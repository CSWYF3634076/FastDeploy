"""
# Copyright (c) 2026  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""

from __future__ import annotations

import logging
from typing import Callable, Iterable, Optional

import paddle
from paddle import nn

logger = logging.getLogger(__name__)


_CPU_OFFLOAD_BYTES = 0
_CPU_OFFLOAD_MAX_BYTES = 0

_DTYPE_TO_BYTES = {
    paddle.bool: 1,
    paddle.int8: 1,
    paddle.uint8: 1,
    paddle.int16: 2,
    paddle.float16: 2,
    paddle.bfloat16: 2,
    paddle.int32: 4,
    paddle.float32: 4,
    paddle.int64: 8,
    paddle.float64: 8,
}


def set_cpu_offload_max_bytes(max_bytes: int) -> None:
    """Set global CPU offload budget in bytes."""
    global _CPU_OFFLOAD_MAX_BYTES, _CPU_OFFLOAD_BYTES
    _CPU_OFFLOAD_BYTES = 0
    _CPU_OFFLOAD_MAX_BYTES = max(0, int(max_bytes))


def _param_nbytes(param: paddle.Tensor) -> int:
    dtype_size = _DTYPE_TO_BYTES.get(param.dtype, 4)
    shape = getattr(param, "shape", None)
    if shape is None:
        return 0
    numel = 1
    for dim in shape:
        if dim is None or dim < 0:
            return 0
        numel *= int(dim)
    return numel * dtype_size


def _layer_nbytes(params: Iterable[paddle.Tensor]) -> int:
    return sum(_param_nbytes(p) for p in params)


def _exclude_offload_layerlist(name: str) -> bool:
    # Keep vision tower layers on GPU to avoid multimodal image understanding regressions.
    name_l = name.lower()
    return any(key in name_l for key in ("vision", "visual", "resampler", "projector"))


def _offload_layer_to_cpu(layer: nn.Layer) -> None:
    if getattr(layer, "_fd_cpu_offloaded", False):
        return
    _offload_layer_params_to_pinned(layer)
    layer._fd_cpu_offloaded = True


def _load_layer_to_device(layer: nn.Layer, device_place: paddle.Place) -> None:
    if not getattr(layer, "_fd_cpu_offloaded", False):
        return
    _load_layer_params_to_device(layer, device_place)
    layer._fd_cpu_offloaded = False


def _offload_layer_params_to_pinned(layer: nn.Layer) -> None:
    if not paddle.is_compiled_with_cuda():
        layer.to(paddle.CPUPlace())
        return
    try:
        pinned_place = paddle.CUDAPinnedPlace()
    except Exception:
        logger.warning("CUDAPinnedPlace is unavailable, fallback to CPUPlace for offload buffers.")
        pinned_place = paddle.CPUPlace()
    for param in layer.parameters():
        if hasattr(param, "_is_initialized") and not param._is_initialized():
            shape = getattr(param, "shape", None)
            if shape is None or any(dim is None or dim < 0 for dim in shape):
                continue
            try:
                cpu_data = paddle.zeros(shape, dtype=param.dtype, device=pinned_place, pin_memory=True)
            except Exception:
                # Some Paddle builds do not support direct pinned allocation in zeros().
                cpu_data = paddle.zeros(shape, dtype=param.dtype, device=paddle.CPUPlace())
                if not isinstance(pinned_place, paddle.CPUPlace):
                    try:
                        cpu_data = cpu_data._copy_to(pinned_place, True)
                    except Exception:
                        pass
            logger.debug(
                "CPU offload init param: shape=%s dtype=%s param_place=%s cpu_place=%s",
                shape,
                param.dtype,
                param.place,
                cpu_data.place,
            )
            param._fd_cpu_data = cpu_data
            cpu_tensor = cpu_data.value().get_tensor()
            param_tensor = param.value().get_tensor()
            param_tensor._share_data_with(cpu_tensor)
            continue
        cpu_data = getattr(param, "_fd_cpu_data", None)
        if cpu_data is None:
            try:
                cpu_data = param._copy_to(pinned_place, False)
            except Exception:
                cpu_data = param._copy_to(pinned_place, True)
            param._fd_cpu_data = cpu_data
        cpu_tensor = cpu_data.value().get_tensor()
        param_tensor = param.value().get_tensor()
        param_tensor._share_data_with(cpu_tensor)
        gpu_data = getattr(param, "_fd_gpu_data", None)
        if gpu_data is not None:
            gpu_data.value().get_tensor()._clear()
            param._fd_gpu_data = None

    # logger.warning("offload_layer_params_to_pinned %s", layer)


def _load_layer_params_to_device(layer: nn.Layer, device_place: paddle.Place) -> None:
    for param in layer.parameters():
        cpu_data = getattr(param, "_fd_cpu_data", None)
        if cpu_data is None:
            continue
        try:
            gpu_data = cpu_data._copy_to(device_place, False)
        except Exception:
            gpu_data = cpu_data._copy_to(device_place, True)
        param._fd_gpu_data = gpu_data
        gpu_tensor = gpu_data.value().get_tensor()
        param_tensor = param.value().get_tensor()
        param_tensor._share_data_with(gpu_tensor)


def _resolve_device_place(device: str, device_id: Optional[int]) -> Optional[paddle.Place]:
    if device.startswith("gpu") and paddle.is_compiled_with_cuda():
        device_index = None
        if ":" in device:
            try:
                _, device_id_str = device.split(":")
                device_index = int(device_id_str)
            except Exception:
                device_index = None
        if device_index is None:
            try:
                current_device = paddle.device.get_device()
                if current_device.startswith("gpu:"):
                    device_index = int(current_device.split(":")[1])
            except Exception:
                device_index = None
        if device_index is None:
            device_index = 0
        try:
            return paddle.CUDAPlace(device_index)
        except Exception:
            if device_id is not None:
                try:
                    return paddle.CUDAPlace(int(device_id))
                except Exception:
                    return None
            return None
    return None


def maybe_offload_layer(
    layer: nn.Layer,
    device_place: paddle.Place,
    *,
    wrap_forward: bool,
) -> nn.Layer:
    global _CPU_OFFLOAD_BYTES
    if _CPU_OFFLOAD_MAX_BYTES <= 0:
        return layer
    if _CPU_OFFLOAD_BYTES >= _CPU_OFFLOAD_MAX_BYTES:
        return layer
    if getattr(layer, "_fd_cpu_offload_wrapped", False):
        return layer
    params = list(layer.parameters())
    if not params:
        return layer
    layer_bytes = _layer_nbytes(params)
    if _CPU_OFFLOAD_BYTES + layer_bytes > _CPU_OFFLOAD_MAX_BYTES:
        return layer
    _CPU_OFFLOAD_BYTES += layer_bytes
    if wrap_forward:
        _wrap_layer_forward_for_offload(layer, device_place)
    _offload_layer_to_cpu(layer)
    return layer


def _wrap_layer_forward_for_offload(layer: nn.Layer, device_place: paddle.Place) -> None:
    if getattr(layer, "_fd_cpu_offload_wrapped", False):
        return
    original_forward: Callable[..., paddle.Tensor] = layer.forward

    def wrapped_forward(*args, **kwargs):
        _load_layer_to_device(layer, device_place)
        try:
            return original_forward(*args, **kwargs)
        finally:
            _offload_layer_to_cpu(layer)

    layer.forward = wrapped_forward
    layer._fd_cpu_offload_wrapped = True


def apply_cpu_offload_to_model(
    model: nn.Layer,
    cpu_offload_gb: float,
    *,
    device: Optional[str] = None,
    device_id: Optional[int] = None,
    show_progress: bool = False,
) -> None:
    max_bytes = int(cpu_offload_gb * 1024**3)
    if max_bytes <= 0:
        return
    if device is None:
        try:
            device = paddle.device.get_device()
        except Exception:
            device = "gpu:0"
    device_place = _resolve_device_place(device, device_id)
    if device_place is None:
        logger.warning("cpu_offload_gb is set but device %s does not support CPU offload.", device)
        return
    if getattr(model, "_fd_cpu_offload_enabled", False):
        return
    set_cpu_offload_max_bytes(max_bytes)
    logger.info("Enable CPU offload: max %.2f GiB.", cpu_offload_gb)
    logger.warning(
        "CPU offload uses layer swapping on Paddle; decoding throughput will drop significantly versus vLLM."
    )
    offloaded_layers = _apply_offload_to_layerlists(
        model,
        device_place,
        wrap_forward=True,
        show_progress=show_progress,
    )
    if offloaded_layers > 0:
        model._fd_cpu_offload_enabled = True


def _apply_offload_to_layerlists(
    model: nn.Layer,
    device_place: paddle.Place,
    *,
    wrap_forward: bool,
    show_progress: bool,
) -> int:
    layer_lists = [
        sublayer
        for name, sublayer in model.named_sublayers()
        if isinstance(sublayer, nn.LayerList) and not _exclude_offload_layerlist(name)
    ]
    total_layers = sum(len(layer_list) for layer_list in layer_lists)
    if total_layers == 0:
        return 0
    processed = 0
    offloaded_layers = 0
    for layer_list in layer_lists:
        processed, offloaded_layers = _offload_layer_list(
            layer_list,
            device_place,
            wrap_forward=wrap_forward,
            show_progress=show_progress,
            processed=processed,
            total_layers=total_layers,
            offloaded_layers=offloaded_layers,
        )
    return offloaded_layers


def _offload_layer_list(
    layer_list: Iterable[nn.Layer],
    device_place: paddle.Place,
    *,
    wrap_forward: bool,
    show_progress: bool,
    processed: int,
    total_layers: int,
    offloaded_layers: int,
) -> tuple[int, int]:
    for idx, layer in enumerate(layer_list):
        new_layer = maybe_offload_layer(layer, device_place, wrap_forward=wrap_forward)
        if new_layer is not layer:
            layer_list[idx] = new_layer
        if getattr(layer, "_fd_cpu_offloaded", False):
            offloaded_layers += 1
        processed += 1
        if show_progress and _should_log_progress(processed, total_layers):
            logger.info(
                "CPU offload progress: %d/%d (%.1f%%)", processed, total_layers, processed / total_layers * 100
            )
    return processed, offloaded_layers


def _should_log_progress(processed: int, total: int) -> bool:
    if processed == 1 or processed == total:
        return True
    if total <= 10:
        return True
    step = max(1, total // 20)
    return processed % step == 0
