"""
# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
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

import os
import re
import time
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, List, Optional, Union

import paddle
from paddleformers.utils.log import logger

from fastdeploy import envs
from fastdeploy.config import FDConfig
from fastdeploy.model_executor.layers.utils import get_tensor, log_cuda_memory
from fastdeploy.platforms import current_platform

_PINNED_SLICE_PARENTS = {}
_TRANSPOSE_FAST_MAX_PEAK_GB = 1.5
_TRANSPOSE_FAST_MAX_PEAK_RATIO = 0.08
_TRANSPOSE_FAST_SAFETY_MARGIN_MB = 256

_DTYPE_NBYTES = {
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
if hasattr(paddle, "float8_e4m3fn"):
    _DTYPE_NBYTES[paddle.float8_e4m3fn] = 1


def _dtype_nbytes(dtype) -> int:
    return _DTYPE_NBYTES.get(dtype, 4)


def _tensor_nbytes(tensor) -> int:
    shape = getattr(tensor, "shape", None)
    if shape is None:
        return 0
    numel = 1
    for dim in shape:
        if dim is None or dim < 0:
            return 0
        numel *= int(dim)
    return numel * _dtype_nbytes(tensor.dtype)


def _current_cuda_device_id() -> int:
    try:
        current_device = paddle.device.get_device()
        if current_device.startswith("gpu:"):
            return int(current_device.split(":")[1])
    except Exception:
        pass
    return 0


def _enable_gpu_fast_transpose() -> bool:
    value = os.getenv("FD_TRANSPOSE_GPU_FAST", "1").strip().lower()
    return value in ("1", "true", "yes", "on")


def _transpose_gpu_budget(weight_nbytes: int) -> dict[str, int | bool]:
    device_id = _current_cuda_device_id()
    max_peak_bytes = int(_TRANSPOSE_FAST_MAX_PEAK_GB * 1024**3)
    total_memory = 0
    reserved_memory = 0
    try:
        total_memory = int(paddle.device.cuda.get_device_properties(device_id).total_memory)
        reserved_memory = int(paddle.device.cuda.memory_reserved(device_id))
        max_peak_bytes = min(max_peak_bytes, int(total_memory * _TRANSPOSE_FAST_MAX_PEAK_RATIO))
    except Exception:
        pass

    estimated_peak_bytes = int(weight_nbytes) * 2
    safety_margin_bytes = int(_TRANSPOSE_FAST_SAFETY_MARGIN_MB * 1024**2)
    available_bytes = max(0, total_memory - reserved_memory)
    can_use_gpu_fast = total_memory > 0 and estimated_peak_bytes <= max_peak_bytes
    can_use_gpu_fast = can_use_gpu_fast and (estimated_peak_bytes + safety_margin_bytes <= available_bytes)

    return {
        "device_id": device_id,
        "estimated_peak_bytes": estimated_peak_bytes,
        "max_peak_bytes": max_peak_bytes,
        "total_memory": total_memory,
        "reserved_memory": reserved_memory,
        "can_use_gpu_fast": can_use_gpu_fast,
    }


def _transpose_meta(weight) -> tuple[list[int], list[int]]:
    if len(weight.shape) == 2:
        return [1, 0], weight.shape[::-1]
    if len(weight.shape) == 3:
        return [0, 2, 1], [weight.shape[0]] + list(weight.shape[1:][::-1])
    raise ValueError(f"Unsupported weight rank {len(weight.shape)} in process_weight_transpose.")


def _try_gpu_fast_host_transpose(weight, perm, debug_mem_log):
    estimated_peak_bytes = 0
    max_peak_bytes = 0
    if not (_enable_gpu_fast_transpose() and _is_pinned_place(weight) and current_platform.is_cuda()):
        return None, "gpu_fast_unavailable", estimated_peak_bytes, max_peak_bytes

    budget = _transpose_gpu_budget(_tensor_nbytes(weight))
    estimated_peak_bytes = int(budget["estimated_peak_bytes"])
    max_peak_bytes = int(budget["max_peak_bytes"])
    if not budget["can_use_gpu_fast"]:
        return None, "gpu_fast_budget_blocked", estimated_peak_bytes, max_peak_bytes

    src_gpu = None
    transposed_gpu = None
    transposed_cpu = None
    try:
        if debug_mem_log:
            log_cuda_memory(f"[process_weight_transpose] gpu_fast before h2d place={weight.place}")
        src_gpu = weight._copy_to(paddle.CUDAPlace(int(budget["device_id"])), True)
        if debug_mem_log:
            log_cuda_memory("[process_weight_transpose] gpu_fast after h2d")
        transposed_gpu = src_gpu.transpose(perm)
        if debug_mem_log:
            log_cuda_memory("[process_weight_transpose] gpu_fast after transpose")
        transposed_cpu = transposed_gpu._copy_to(paddle.CPUPlace(), True)
        transposed_tensor = transposed_cpu._copy_to(weight.place, True)
        return transposed_tensor, "gpu_fast", estimated_peak_bytes, max_peak_bytes
    except Exception:
        return None, "gpu_fast_error", estimated_peak_bytes, max_peak_bytes
    finally:
        del transposed_cpu, transposed_gpu, src_gpu


class BitMaskTracker:
    def __init__(self, length: int):
        """
        Track filling status along a single dimension using a bitmask.

        Args:
            length (int): Number of positions to track (e.g., columns or rows)
        """
        self.length = length
        self.mask = 0

    def mark(self, start: int, end: int):
        """
        Mark the range [start, end) as filled.

        Args:
            start (int): Start index (inclusive)
            end (int): End index (exclusive)
        """
        if start < 0 or end > self.length or start >= end:
            raise ValueError("Invalid mark range")
        block = ((1 << (end - start)) - 1) << start
        self.mask |= block

    def is_full(self) -> bool:
        """Return True if all positions are filled."""
        return self.mask == (1 << self.length) - 1


class TensorTracker:
    def __init__(self, shape: tuple, output_dim: int):
        """
        Unified tracker for 2D or 3D tensors.

        Args:
            shape (tuple): Tensor shape
            output_dim (bool):
                - 2D: True = track columns (dim=1), False = track rows (dim=0)
                - 3D: True = track columns (dim=2), False = track rows (dim=1)
        """
        self.shape = shape
        self.output_dim = output_dim

        if len(shape) == 2:
            self.track_dim = 1 if output_dim else 0
            self.trackers = [BitMaskTracker(shape[self.track_dim])]
        elif len(shape) == 3:
            batch = shape[0]
            self.track_dim = 2 if output_dim else 1
            self.trackers = [BitMaskTracker(shape[self.track_dim]) for _ in range(batch)]
        else:
            raise ValueError("Only 2D or 3D tensors supported")

    def mark(self, start: int = 0, end: int = None, batch_id: int = None):
        """
        Mark a slice of the tensor as filled.

        Args:
            batch_id (int, optional): Batch index for 3D tensors
            start (int): Start index along tracked dimension
            end (int): End index along tracked dimension
        """
        if end is None:
            end = self.shape[self.track_dim]

        if len(self.shape) == 2:
            self.trackers[0].mark(start, end)
        else:
            if batch_id is None:
                raise ValueError("batch_id must be provided for 3D tensor")
            self.trackers[batch_id].mark(start, end)

    def is_fully_copied(self) -> bool:
        """Return True if the tensor is fully filled along tracked dimension(s)."""
        return all(tr.is_full() for tr in self.trackers)


def set_weight_attrs(param, param_attr_map: Optional[dict[str, Any]]):
    if param_attr_map is None:
        return
    for key, value in param_attr_map.items():
        setattr(param, key, value)


def _is_pinned_place(tensor) -> bool:
    try:
        place = tensor.place
        if isinstance(place, paddle.CUDAPinnedPlace):
            return True
        place_str = str(place).lower()
        return "gpu_pinned" in place_str
    except Exception:
        return False


def _is_gpu_place(tensor) -> bool:
    try:
        place = tensor.place
        if hasattr(place, "is_gpu_place") and place.is_gpu_place():
            return True
        place_str = str(place).lower()
        return "gpu" in place_str and "pinned" not in place_str
    except Exception:
        return False


def _repair_pinned_parent_after_copy(dst) -> None:
    parent = getattr(dst, "_fd_pinned_parent", None)
    if parent is None:
        parent = _PINNED_SLICE_PARENTS.pop(id(dst), None)
    else:
        _PINNED_SLICE_PARENTS.pop(id(dst), None)
    if parent is None:
        return

    tensor_track = getattr(parent, "tensor_track", None)
    if tensor_track is not None:
        try:
            if not tensor_track.is_fully_copied():
                return
        except Exception:
            pass

    if not _is_gpu_place(parent):
        try:
            setattr(dst, "_fd_pinned_parent", None)
        except Exception:
            pass
        return

    try:
        pinned_place = paddle.CUDAPinnedPlace()
    except Exception:
        pinned_place = paddle.CPUPlace()

    try:
        pinned_parent = parent._copy_to(pinned_place, True)
    except Exception:
        pinned_parent = paddle.to_tensor(parent.numpy(), place=pinned_place)

    try:
        parent_tensor = parent.value().get_tensor()
        pinned_tensor = pinned_parent.value().get_tensor()
        parent_tensor._share_data_with(pinned_tensor)
    except Exception:
        parent.set_value(pinned_parent)

    try:
        dst.value().get_tensor()._clear()
    except Exception:
        pass

    _PINNED_SLICE_PARENTS.pop(id(dst), None)
    try:
        setattr(dst, "_fd_pinned_parent", None)
    except Exception:
        pass


def slice_fn(weight_or_paramter, output_dim, start, end, step=1):
    debug_slice_log = os.getenv("FD_CPU_OFFLOAD_DEBUG", "0").strip().lower() in ("1", "true", "yes", "on")

    def _slice_impl(tensor):
        if hasattr(tensor, "get_shape"):
            shape = tensor.get_shape()
        else:
            shape = tensor.shape
        if len(shape) == 1:
            return tensor[start:end:step]
        elif output_dim:
            return tensor[..., start:end:step]
        else:
            return tensor[start:end:step, ...]

    # Paddle can fallback pinned slice to GPU. Keep a reference to the original
    # tensor so h2d_copy can repin after shard writes are finished.
    if _is_pinned_place(weight_or_paramter):
        if debug_slice_log:
            log_cuda_memory(f"[slice_fn] pinned before slice place={weight_or_paramter.place}")
        sliced = _slice_impl(weight_or_paramter)
        if _is_gpu_place(sliced) or _is_gpu_place(weight_or_paramter):
            try:
                setattr(sliced, "_fd_pinned_parent", weight_or_paramter)
            except Exception:
                _PINNED_SLICE_PARENTS[id(sliced)] = weight_or_paramter
        if debug_slice_log:
            log_cuda_memory(f"[slice_fn] pinned after slice src={weight_or_paramter.place} dst={sliced.place}")
        return sliced

    original_tensor = weight_or_paramter
    if hasattr(original_tensor, "get_shape"):
        shape = original_tensor.get_shape()
    else:
        shape = original_tensor.shape
    if debug_slice_log:
        log_cuda_memory(
            f"[slice_fn] before param slice shape={shape} weight_or_paramter.place={original_tensor.place}"
        )
    if len(shape) == 1:
        weight_or_paramter = original_tensor[start:end:step]
    elif output_dim:
        weight_or_paramter = original_tensor[..., start:end:step]
    else:
        weight_or_paramter = original_tensor[start:end:step, ...]
    if _is_gpu_place(original_tensor) and getattr(original_tensor, "_fd_cpu_data", None) is not None:
        try:
            setattr(weight_or_paramter, "_fd_pinned_parent", original_tensor)
        except Exception:
            _PINNED_SLICE_PARENTS[id(weight_or_paramter)] = original_tensor
    if debug_slice_log:
        log_cuda_memory(
            f"[slice_fn] after param slice shape={shape} weight_or_paramter.place={weight_or_paramter.place}"
        )
    return weight_or_paramter


def process_weight_transpose(layer, weight_name):
    debug_mem_log = os.getenv("FD_CPU_OFFLOAD_DEBUG", "0").strip().lower() in ("1", "true", "yes", "on")
    t0 = time.perf_counter()
    weight = getattr(layer, weight_name)
    # Some loaders already transpose per shard during loading.
    if getattr(weight, "_fd_transposed_on_load", False):
        weight._fd_transposed_on_load = False
        t1 = time.perf_counter()
        print(
            f"[process_weight_transpose] {weight_name} | "
            f"path=skip_already_transposed_on_load | total={((t1 - t0) * 1000):.2f}ms"
        )
        return

    in_host_place = _is_pinned_place(weight) or ("cpu" in str(weight.place).lower())
    t1 = time.perf_counter()

    perm, weight_shape = _transpose_meta(weight)
    t2 = time.perf_counter()

    transpose_path = "gpu_direct"
    estimated_peak_bytes = 0
    max_peak_bytes = 0

    if in_host_place:
        transposed_tensor, transpose_path, estimated_peak_bytes, max_peak_bytes = _try_gpu_fast_host_transpose(
            weight, perm, debug_mem_log
        )
        if transposed_tensor is None:
            raise RuntimeError(
                f"process_weight_transpose host path requires gpu_fast, but got path={transpose_path}. "
                "Set FD_TRANSPOSE_GPU_FAST=1 and ensure GPU budget is sufficient."
            )

        if debug_mem_log:
            log_cuda_memory(
                f"[process_weight_transpose] host path={transpose_path} transposed_place={transposed_tensor.place}"
            )

        weight_tmp = layer.create_parameter(
            shape=weight_shape,
            dtype=weight.dtype,
            device=transposed_tensor.place,
            default_initializer=paddle.nn.initializer.Constant(0),
            is_bias=False,
        )

        # Keep an explicit reference to backing host tensor; otherwise some
        # Paddle builds may invalidate guards after temporary tensors are freed.
        # weight_tmp._fd_host_data_ref = transposed_tensor
        if _is_pinned_place(transposed_tensor) or getattr(weight, "_fd_cpu_data", None) is not None:
            weight_tmp._fd_cpu_data = transposed_tensor
            weight_tmp._fd_gpu_data = None

        weight_tmp.set_value(transposed_tensor)
        if debug_mem_log:
            log_cuda_memory(f"[process_weight_transpose] host gpu_fast set_value place={transposed_tensor.place}")
    else:
        weight_tmp = layer.create_parameter(
            shape=weight_shape,
            dtype=weight.dtype,
            device=weight.place,
            default_initializer=paddle.nn.initializer.Constant(0),
            is_bias=False,
        )
        weight_transpose = weight.transpose(perm)
        weight_tmp.copy_(weight_transpose, False)
        if debug_mem_log:
            log_cuda_memory(f"[process_weight_transpose] gpu_direct place={weight.place}")
    t3 = time.perf_counter()

    # Keep runtime attrs used by load/offload paths.
    # for attr_name in (
    #     "weight_loader",
    #     "output_dim",
    #     "weight_need_transpose",
    #     "tensor_track",
    #     "tp_row_bias",
    #     "packed_dim",
    #     "packed_factor",
    #     "pack_factor",
    # ):
    #     if hasattr(weight, attr_name):
    #         setattr(weight_tmp, attr_name, getattr(weight, attr_name))
    t4 = time.perf_counter()

    free_tensor(weight)
    setattr(layer, weight_name, weight_tmp)
    t5 = time.perf_counter()
    print(
        f"[process_weight_transpose] {weight_name} | "
        f"path={transpose_path} | "
        f"est_peak_gb={estimated_peak_bytes / 1024**3:.3f} | "
        f"peak_budget_gb={max_peak_bytes / 1024**3:.3f} | "
        f"prep={((t1 - t0) * 1000):.2f}ms | "
        f"shape={((t2 - t1) * 1000):.2f}ms | "
        f"transpose_copy={((t3 - t2) * 1000):.2f}ms | "
        f"attrs={((t4 - t3) * 1000):.2f}ms | "
        f"finalize={((t5 - t4) * 1000):.2f}ms | "
        f"total={((t5 - t0) * 1000):.2f}ms"
    )


def process_weights_after_loading(sublayers_dict: dict, fd_config: FDConfig):
    """
    process_weights_after_loading:
    """

    def fn(model_sublayer_name: str, param=None):
        from fastdeploy.model_executor.layers.linear import (
            KVBatchLinear,
            UnquantizedLinearMethod,
        )
        from fastdeploy.model_executor.layers.moe.moe import get_moe_method

        if model_sublayer_name not in sublayers_dict:
            return
        model_sublayer = sublayers_dict[model_sublayer_name]
        if isinstance(model_sublayer, KVBatchLinear):
            model_sublayer.process_weights_after_loading()
        if fd_config.quant_config and not fd_config.quant_config.is_checkpoint_bf16:
            # skip for offline quantization
            return
        if hasattr(model_sublayer, "quant_method"):
            quant_method = getattr(model_sublayer, "quant_method", None)
            unquant_moe_layer = get_moe_method()
            if unquant_moe_layer is None:
                unquant_moe_cls = object
            else:
                unquant_moe_cls = type(unquant_moe_layer)
            if type(quant_method) is UnquantizedLinearMethod or type(quant_method) is unquant_moe_cls:
                # skip unquantized linear
                return
            if not hasattr(quant_method, "process_weights_after_loading"):
                return
            if param is not None and hasattr(param, "tensor_track") and param.tensor_track is None:
                return
            if param is not None and hasattr(param, "tensor_track") and not param.tensor_track.is_fully_copied():
                return
            quant_method.process_weights_after_loading(model_sublayer)

    return fn


@dataclass
class WeightsMapper:
    orig_to_new_prefix: Mapping[str, Optional[str]] = field(default_factory=dict)

    def _map_name(self, key: str) -> Optional[str]:
        for prefix, new_key in self.orig_to_new_prefix.items():
            if key.startswith(prefix):
                key = key.replace(prefix, new_key, 1)
        return key

    def apply(self, weight_name):
        return self._map_name(weight_name)


def remap_weight_keys(weights_iterator, mapper: dict, include_keys: Optional[List[str]] = None):
    if include_keys is not None:
        weights_iterator = filter(lambda item: any(key in item[0] for key in include_keys), weights_iterator)

    return (
        (next((key.replace(k, v) for k, v in mapper.items() if k in key), key), value)
        for key, value in weights_iterator
    )


def process_weights_before_loading(
    *, skip_prefixes: Optional[List[str]] = None, mapper: Optional[WeightsMapper] = None
):
    def _can_skip(weight_name):
        return any(weight_name.startswith(p) for p in (skip_prefixes or []))

    def fn(weight_name):
        if mapper is not None:
            weight_name = mapper.apply(weight_name)
        if _can_skip(weight_name):
            weight_name = None
        return weight_name

    return fn


def weight_fully_copied(weight):
    return (
        hasattr(weight, "tensor_track") and weight.tensor_track is not None and weight.tensor_track.is_fully_copied()
    )


def process_final_after_loading(model, fd_config: FDConfig):
    # process_final_after_loading handles the post-loading process for cases other than dynamic quantization.
    from fastdeploy.model_executor.layers.linear import (
        KVBatchLinear,
        UnquantizedLinearMethod,
    )
    from fastdeploy.model_executor.layers.moe.moe import get_moe_method

    for name, sublayer in model.named_sublayers():
        if isinstance(sublayer, KVBatchLinear):
            continue
        quant_method = getattr(sublayer, "quant_method", None)
        if quant_method is not None:
            unquant_moe_layer = get_moe_method()
            if unquant_moe_layer is None:
                unquant_moe_cls = object
            else:
                unquant_moe_cls = type(unquant_moe_layer)
            is_unquant_cls = type(quant_method) is UnquantizedLinearMethod or type(quant_method) is unquant_moe_cls
            is_offline_quantized_ckpt = not (fd_config.quant_config and fd_config.quant_config.is_checkpoint_bf16)
            if is_unquant_cls or is_offline_quantized_ckpt:
                if hasattr(quant_method, "process_weights_after_loading"):
                    quant_method.process_weights_after_loading(sublayer)
                continue
        if not hasattr(sublayer, "process_weights_after_loading"):
            continue
        log_cuda_memory(f"[sublayer.process_weights_after_loading] before {name}")
        sublayer.process_weights_after_loading()


def free_tensor(tensor):
    if hasattr(tensor, "tensor_track"):
        tensor.tensor_track = None
    tensor.value().get_tensor()._clear()
    del tensor


def create_parameter_and_copy(layer: paddle.nn.Layer, name: str, weight: paddle.Tensor) -> None:
    """
    Create a parameter in the layer and copy data from weight.

    Args:
        layer (paddle.nn.Layer): The layer where the parameter will be created.
        name (str): The name of the parameter.
        weight (paddle.Tensor): The source weight tensor.
    """
    setattr(
        layer,
        name,
        layer.create_parameter(
            shape=weight.shape,
            dtype=weight.dtype,
            default_initializer=paddle.nn.initializer.Constant(0),
        ),
    )
    getattr(layer, name).copy_(weight, False)


def fd_cast(weight, param):
    if weight.dtype != param.dtype:
        if weight.dtype == paddle.int8 and param.dtype == paddle.float8_e4m3fn:
            weight = weight.view(param.dtype)
        else:
            weight = weight.cast(param.dtype)
    return weight


def default_weight_loader(fd_config: FDConfig = None) -> None:
    """Default weight loader"""

    def fn(param, loaded_weight, shard_id: Optional[Union[int, str]] = None):
        """fn"""
        output_dim = getattr(param, "output_dim", None)
        weight_need_transpose = getattr(param, "weight_need_transpose", False)
        if hasattr(loaded_weight, "get_shape"):
            loaded_shape = loaded_weight.get_shape()
        else:
            loaded_shape = loaded_weight.shape
        if weight_need_transpose:
            loaded_weight = loaded_weight.transpose([1, 0])
            param._fd_transposed_on_load = True
            param.weight_need_transpose = False
        elif len(loaded_shape) == 2 and param.shape == loaded_shape[::-1]:
            loaded_weight = loaded_weight.transpose([1, 0])
            param._fd_transposed_on_load = True
        elif len(loaded_shape) == 3 and param.shape[0] == loaded_shape[0]:
            if param.shape[1] == loaded_shape[2] and param.shape[2] == loaded_shape[1]:
                loaded_weight = loaded_weight.transpose([0, 2, 1])
                param._fd_transposed_on_load = True
        # Tensor parallelism splits the weight along the output_dim
        if output_dim is not None and fd_config is not None and fd_config.parallel_config.tensor_parallel_size > 1:
            dim = -1 if output_dim else 0
            if isinstance(loaded_weight, paddle.Tensor):
                size = loaded_weight.shape[dim]
            else:
                size = loaded_weight.get_shape()[dim]
            block_size = size // fd_config.parallel_config.tensor_parallel_size
            shard_offset = fd_config.parallel_config.tensor_parallel_rank * block_size
            shard_size = (fd_config.parallel_config.tensor_parallel_rank + 1) * block_size
            loaded_weight = slice_fn(loaded_weight, output_dim, shard_offset, shard_size)

        tp_row_bias = getattr(param, "tp_row_bias", None)
        if tp_row_bias:
            loaded_weight = loaded_weight / fd_config.parallel_config.tensor_parallel_size

        # mlp.gate.weight is precision-sensitive, so we cast it to float32 for computation
        loaded_weight = fd_cast(loaded_weight, param)
        if param.shape != loaded_weight.shape:
            # for e_score_correction_bias
            loaded_weight = loaded_weight.reshape(param.shape)
        assert param.shape == loaded_weight.shape, (
            f" Attempted to load weight ({loaded_weight.shape}) " f"into parameter ({param.shape})"
        )
        loaded_weight = get_tensor(loaded_weight)
        param.copy_(loaded_weight, False)

    return fn


def is_pre_sliced_weight(model_path):
    rank_dirs = [
        f for f in os.listdir(model_path) if f.startswith("rank") and os.path.isdir(os.path.join(model_path, f))
    ]
    return len(rank_dirs) > 1


def is_paddle_support_v1_loader():
    src_shape = [32, 32]
    tgt_shape = [1, 32, 64]
    src_tensor = paddle.ones(src_shape, dtype="float32")
    tgt_tensor = paddle.zeros(tgt_shape, dtype="float32")
    for exp_id in range(tgt_shape[0]):
        # gate
        gate_tgt = tgt_tensor[exp_id][..., : tgt_shape[2] // 2]
        gate_tgt.copy_(src_tensor, False)
        # up
        up_tgt = tgt_tensor[exp_id][..., tgt_shape[2] // 2 :]
        up_tgt.copy_(src_tensor, False)
    is_same = bool(paddle.all(tgt_tensor == 1))
    return is_same


_support_new_h2d = None


def is_paddle_support_new_h2d():
    import subprocess
    import sys

    global _support_new_h2d
    if _support_new_h2d is not None:
        return _support_new_h2d

    code = """
import paddle
import resource

resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
try:
    dst = paddle.zeros([2, 4], dtype='bfloat16')
    src = paddle.ones([2, 2], dtype='bfloat16', device='cpu')
    dst = dst[..., :2]
    dst.copy_(src)
    print(1)
except:
    print(0)
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True)
    _support_new_h2d = result.stdout.strip() == b"1"
    return _support_new_h2d


def h2d_copy(dst, src, blocking=True):
    if not current_platform.is_cuda() or not is_paddle_support_new_h2d():
        # For non-GPU devices, data is transferred to device (H2D) in advance.
        src = get_tensor(src)
    if len(src.shape) == 1:
        # TODO (bukejiyu):A recently merged Paddle PR introduced a hang when copying 1-D non-contiguous tensors. This approach serves as a temporary workaround.
        src = get_tensor(src)
    dst.copy_(src, blocking)
    _repair_pinned_parent_after_copy(dst)


def v1_loader_support(fd_config):
    _v1_no_support_archs = ["Qwen2VLForConditionalGeneration"]

    def _get_unsupported_quant():
        if current_platform.is_cuda():
            return {"w4a8", "wint2"}
        elif current_platform.is_xpu():
            return {"w4a8", "w8a8"}
        return set()

    def _err_msg(msg: str) -> str:
        logger.info(msg + "; fallback to the v0 loader for model loading.")

    if not (
        current_platform.is_cuda()
        or current_platform.is_xpu()
        or current_platform.is_iluvatar()
        or current_platform.is_maca()
        or current_platform.is_intel_hpu()
    ):
        _err_msg("v1loader currently only support backends gpu, xpu, intel_hpu, iluvatar and maca")
        return False

    if is_pre_sliced_weight(fd_config.model_config.model):
        _err_msg("v1 loader currently does not support pre-sliced weights")
        return False

    if envs.FD_MOE_BACKEND.lower() == "marlin":
        _err_msg("v1 loader currently does not support marlin backend")
        return False

    if fd_config.quant_config is not None:
        if fd_config.quant_config.name() == "mix_quant":
            moe_quant_type = fd_config.quant_config.moe_quant_type
            dense_quant_type = fd_config.quant_config.dense_quant_type
        else:
            moe_quant_type = fd_config.quant_config.name()
            dense_quant_type = fd_config.quant_config.name()
        unsupported_quant = _get_unsupported_quant()

        if unsupported_quant & {moe_quant_type, dense_quant_type}:
            _err_msg("v1 loader currently does not support w4a8/w4afp8/win2 quantization")
            return False
    if fd_config.model_config.architectures[0] in _v1_no_support_archs:
        _err_msg(f"v1 loader currently does not support {fd_config.model_config.architectures[0]}")
        return False

    if not is_paddle_support_v1_loader():
        _err_msg("The installed Paddle does not support v1 loader")
        return False
    return True


@contextmanager
def temporary_dtype(dtype: str):
    """Temporarily set Paddle default dtype"""
    orig_dtype = paddle.get_default_dtype()
    try:
        if dtype is not None and dtype == "float32":
            paddle.set_default_dtype(dtype)
        yield
    finally:
        paddle.set_default_dtype(orig_dtype)


@contextmanager
def multi_switch_config_context(*changes):
    """
    changes: (obj, attr, new_value)
    """
    originals = []
    try:
        for obj, attr, new_value in changes:
            old_value = getattr(obj, attr)
            originals.append((obj, attr, old_value))
            setattr(obj, attr, new_value)
        yield
    finally:
        for obj, attr, old_value in originals:
            setattr(obj, attr, old_value)


def rename_offline_ckpt_suffix_to_fd_suffix(
    fd_config,
    ckpt_weight_suffix: str = "quant_weight",
    ckpt_scale_suffix="weight_scale",
    ckpt_act_suffix="activation_scale",
):
    """
    Create a function to rename checkpoint key suffixes for FastDeploy.

    Replaces the original suffix (default "weight_scale") with the FD target
    suffix (default "quant_weight"). Only the suffix is changed.

    Args:
        fd_config: FastDeploy configuration.
        ckpt_weight_suffix: Original checkpoint key suffix.
        ckpt_scale_suffix: Target FastDeploy key suffix.

    Returns:
        Callable: Function that renames checkpoint keys.
    """
    fd_suffix_map = {}  # noqa: F841
    fp8_suffix_map = {
        ckpt_weight_suffix: "weight",
        ckpt_scale_suffix: "weight_scale_inv",
    }
    tensor_wise_fp8_suffix_map = {
        ckpt_weight_suffix: "weight",
        ckpt_act_suffix: "in_scale",
    }
    moe_quant_type = ""
    dense_quant_type = ""
    if fd_config.quant_config is not None:
        if fd_config.quant_config.name() == "mix_quant":
            moe_quant_type = fd_config.quant_config.moe_quant_type
            dense_quant_type = fd_config.quant_config.dense_quant_type
        else:
            moe_quant_type = fd_config.quant_config.name()
            dense_quant_type = fd_config.quant_config.name()

    def fn(loaded_weight_name, is_moe):
        if fd_config.quant_config is None or fd_config.quant_config.is_checkpoint_bf16:
            return loaded_weight_name
        # Can be extended to other offline quantization suffixes if needed.
        if (is_moe and moe_quant_type == "block_wise_fp8") or (not is_moe and dense_quant_type == "block_wise_fp8"):
            fd_suffix_map = fp8_suffix_map
        if (is_moe and moe_quant_type == "tensor_wise_fp8") or (not is_moe and dense_quant_type == "tensor_wise_fp8"):
            fd_suffix_map = tensor_wise_fp8_suffix_map
        else:
            fd_suffix_map = {}
        for ckpt_suffix, fd_suffix in fd_suffix_map.items():
            if re.search(rf"{ckpt_suffix}$", loaded_weight_name):
                loaded_weight_name = loaded_weight_name.replace(ckpt_suffix, fd_suffix)
                return loaded_weight_name
        return loaded_weight_name

    return fn
