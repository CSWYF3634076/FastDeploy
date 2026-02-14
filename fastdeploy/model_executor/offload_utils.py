"""
# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

import re
from dataclasses import dataclass, field

import paddle
from paddle import nn
from paddleformers.utils.log import logger

from fastdeploy.config import FDConfig
from fastdeploy.model_executor.utils import TensorTracker
from fastdeploy.platforms import current_platform

_DECODER_LAYER_NAME_PATTERN = re.compile(r"(?:^|[.])(layers|mtp_block|h|blocks|block)[.]\d+$")
_MEMORY_GROWTH_LOG_STEP_BYTES = 1 << 30

_CPU_OFFLOAD_MAX_BYTES = 0
_CPU_WEIGHT_OFFLOAD_MANAGER = None
_PARAM_OFFLOAD_EMPTY_CACHE_STEP_BYTES = 256 << 20


@dataclass
class _LayerOffloadState:
    name: str
    layer: nn.Layer
    param_id_to_param: dict[int, paddle.Tensor]
    total_bytes: int
    loaded_param_ids: set[int] = field(default_factory=set)
    cpu_param_cache: dict[int, paddle.Tensor] = field(default_factory=dict)
    offloaded: bool = False
    pre_hook_handle: object | None = None
    post_hook_handle: object | None = None
    pre_hook_calls: int = 0
    post_hook_calls: int = 0
    counted_in_total: bool = False
    param_offloaded_bytes: int = 0
    partial_offload_logged: bool = False


def _bytes_to_gib_str(num_bytes: int) -> str:
    return f"{num_bytes / float(1024**3):.3f} GiB"


def _is_decoder_layer(layer_name: str, layer: nn.Layer) -> bool:
    if _DECODER_LAYER_NAME_PATTERN.search(layer_name) is not None:
        return True
    if hasattr(layer, "self_attn") and (hasattr(layer, "mlp") or hasattr(layer, "feed_forward")):
        return True
    return False


def _get_cuda_memory_snapshot() -> tuple[float, float, float, float] | None:
    if not (current_platform.is_cuda() or current_platform.is_maca()):
        return None
    try:
        curr_alloc = paddle.device.cuda.memory_allocated() / (1024**3)
        curr_reserved = paddle.device.cuda.memory_reserved() / (1024**3)
        max_alloc = paddle.device.cuda.max_memory_allocated() / (1024**3)
        max_reserved = paddle.device.cuda.max_memory_reserved() / (1024**3)
    except Exception:
        return None
    return curr_alloc, curr_reserved, max_alloc, max_reserved


def log_cpu_offload_memory(context: str, force: bool = True) -> None:
    manager = _CPU_WEIGHT_OFFLOAD_MANAGER
    if manager is not None:
        manager.log_memory(context=context, force=force)
        return
    snapshot = _get_cuda_memory_snapshot()
    if snapshot is None:
        return
    curr_alloc, curr_reserved, max_alloc, max_reserved = snapshot
    logger.info(
        f"[cpu-offload] {context} | current_alloc={curr_alloc:.3f} GiB "
        f"current_reserved={curr_reserved:.3f} GiB max_alloc={max_alloc:.3f} GiB "
        f"max_reserved={max_reserved:.3f} GiB"
    )


class CPUWeightOffloadManager:
    def __init__(self, max_bytes: int, fd_config: FDConfig) -> None:
        self.max_bytes = max(0, int(max_bytes))
        self.fd_config = fd_config
        self.enabled = self.max_bytes > 0
        self.disable_reason = ""
        self.total_selected_bytes = 0
        self.total_offloaded_bytes = 0
        self.layer_states: dict[str, _LayerOffloadState] = {}
        self.param_id_to_layer_name: dict[int, str] = {}
        self.postprocess_owner_cache: dict[str, str | None] = {}
        self.cpu_place = self._init_offload_cache_place()
        self.cpu_place_desc = self._place_to_str(self.cpu_place)
        self.cache_tensor_place_counter: dict[str, int] = {}
        self.cache_tensor_place_log_count = 0
        self.next_alloc_log_step = _MEMORY_GROWTH_LOG_STEP_BYTES
        self.next_reserved_log_step = _MEMORY_GROWTH_LOG_STEP_BYTES
        self.pending_param_offload_bytes = 0
        self._wrapped_param_ids: set[int] = set()
        self.partial_offload_watermark_bytes = self._init_partial_offload_watermark_bytes()
        self.partial_offload_activated = False

        self._validate_runtime_support()

    def _validate_runtime_support(self) -> None:
        if not self.enabled:
            return
        if not (current_platform.is_cuda() or current_platform.is_maca()):
            self.enabled = False
            self.disable_reason = "only CUDA/Metax GPU is supported"
            return
        if getattr(self.fd_config.load_config, "load_choices", None) != "default_v1":
            self.enabled = False
            self.disable_reason = "only load_choices=default_v1 is supported in this version"
            return
        if self.fd_config.load_config.dynamic_load_weight:
            self.enabled = False
            self.disable_reason = "dynamic_load_weight is not supported in this version"
            return
        if self.fd_config.quant_config is not None and not getattr(
            self.fd_config.quant_config, "is_checkpoint_bf16", True
        ):
            self.enabled = False
            self.disable_reason = (
                "offline quantized checkpoint loading is not supported in this version " "(is_checkpoint_bf16=False)"
            )
            return

    @staticmethod
    def _dtype_num_bytes(dtype) -> int:
        if dtype is None:
            return 0
        dtype_str = str(dtype).lower()
        mapping = (
            ("complex128", 16),
            ("complex64", 8),
            ("float64", 8),
            ("int64", 8),
            ("uint64", 8),
            ("float32", 4),
            ("int32", 4),
            ("uint32", 4),
            ("bfloat16", 2),
            ("float16", 2),
            ("int16", 2),
            ("uint16", 2),
            ("float8", 1),
            ("int8", 1),
            ("uint8", 1),
            ("bool", 1),
        )
        for key, size in mapping:
            if key in dtype_str:
                return size
        return 0

    @staticmethod
    def _place_to_str(place_obj) -> str:
        try:
            return str(place_obj)
        except Exception:
            return "unknown-place"

    @staticmethod
    def _tensor_place_to_str(tensor: paddle.Tensor) -> str:
        if tensor is None:
            return "none"
        try:
            return str(tensor.place)
        except Exception:
            return "unknown-tensor-place"

    @staticmethod
    def _init_offload_cache_place():
        if not (current_platform.is_cuda() or current_platform.is_maca()):
            raise RuntimeError("[cpu-offload] CUDAPinnedPlace is required, but current platform is not CUDA/Metax.")
        try:
            place = paddle.CUDAPinnedPlace()
            logger.info("[cpu-offload] Offload cache place: CUDAPinnedPlace")
            return place
        except Exception as ex:
            raise RuntimeError(f"[cpu-offload] CUDAPinnedPlace is required but unavailable: {ex}") from ex

    @classmethod
    def _param_num_bytes(cls, param: paddle.Tensor) -> int:
        try:
            numel = int(param.numel())
            elem_size = int(param.element_size())
            if numel > 0 and elem_size > 0:
                return numel * elem_size
        except Exception:
            pass

        try:
            shape = tuple(param.shape)
            numel = 1
            for dim in shape:
                dim_val = int(dim)
                if dim_val <= 0:
                    return 0
                numel *= dim_val
            dtype_bytes = cls._dtype_num_bytes(getattr(param, "dtype", None))
            if dtype_bytes > 0:
                return numel * dtype_bytes
        except Exception:
            pass
        return 0

    @staticmethod
    def _get_named_parameters(layer: nn.Layer, include_sublayers: bool) -> dict[str, paddle.Tensor]:
        try:
            return dict(layer.named_parameters(include_sublayers=include_sublayers))
        except TypeError:
            params = dict(layer.named_parameters())
            if include_sublayers:
                return params
            # Fallback for old paddle versions without include_sublayers argument.
            return {name: param for name, param in params.items() if "." not in name}

    def _register_layer_state(
        self,
        layer_name: str,
        layer: nn.Layer,
        param_id_to_param: dict[int, paddle.Tensor],
        layer_bytes: int,
        source: str,
    ) -> bool:
        if layer_name in self.layer_states:
            return False
        if layer_bytes <= 0 or len(param_id_to_param) == 0:
            return False
        for param_id in param_id_to_param:
            if param_id in self.param_id_to_layer_name:
                return False
        if self.total_selected_bytes + layer_bytes > self.max_bytes:
            return False

        layer_state = _LayerOffloadState(
            name=layer_name,
            layer=layer,
            param_id_to_param=param_id_to_param,
            total_bytes=layer_bytes,
        )
        self.layer_states[layer_name] = layer_state
        self.total_selected_bytes += layer_bytes
        for param_id in param_id_to_param:
            self.param_id_to_layer_name[param_id] = layer_name
        logger.info(
            f"[cpu-offload] Select layer {layer_name} for offload, "
            f"layer_size={_bytes_to_gib_str(layer_bytes)} "
            f"selected_total={_bytes_to_gib_str(self.total_selected_bytes)} source={source}"
        )
        self.postprocess_owner_cache.clear()
        return True

    def _prepare_fallback_layers_by_direct_params(self, named_sublayers: list[tuple[str, nn.Layer]]) -> int:
        selected_param_ids = set(self.param_id_to_layer_name.keys())
        candidates: list[tuple[int, str, nn.Layer, dict[int, paddle.Tensor]]] = []
        for layer_name, layer in named_sublayers:
            params = self._get_named_parameters(layer=layer, include_sublayers=False)
            if len(params) == 0:
                continue
            param_id_to_param: dict[int, paddle.Tensor] = {}
            for param in params.values():
                param_id = id(param)
                if param_id in selected_param_ids:
                    continue
                self._attach_tensor_track_if_needed(param)
                param_id_to_param[param_id] = param
            if len(param_id_to_param) == 0:
                continue
            layer_bytes = sum(self._param_num_bytes(param) for param in param_id_to_param.values())
            if layer_bytes <= 0:
                continue
            candidates.append((layer_bytes, layer_name, layer, param_id_to_param))

        candidates.sort(key=lambda x: x[0], reverse=True)
        selected_cnt = 0
        for layer_bytes, layer_name, layer, param_id_to_param in candidates:
            if self._register_layer_state(
                layer_name=layer_name,
                layer=layer,
                param_id_to_param=param_id_to_param,
                layer_bytes=layer_bytes,
                source="direct-param-fallback",
            ):
                selected_cnt += 1
        return selected_cnt

    def _prepare_fallback_layers_by_param_name(
        self, model: nn.Layer, named_sublayers: list[tuple[str, nn.Layer]]
    ) -> int:
        selected_param_ids = set(self.param_id_to_layer_name.keys())
        sublayers_dict = dict(named_sublayers)
        grouped: dict[str, dict[int, paddle.Tensor]] = {}
        for param_name, param in model.named_parameters():
            if "." not in param_name:
                continue
            layer_name = param_name.rsplit(".", 1)[0]
            layer = sublayers_dict.get(layer_name)
            if layer is None:
                continue
            param_id = id(param)
            if param_id in selected_param_ids:
                continue
            self._attach_tensor_track_if_needed(param)
            grouped.setdefault(layer_name, {})[param_id] = param

        candidates: list[tuple[int, str, nn.Layer, dict[int, paddle.Tensor]]] = []
        for layer_name, param_id_to_param in grouped.items():
            layer_bytes = sum(self._param_num_bytes(param) for param in param_id_to_param.values())
            if layer_bytes <= 0:
                continue
            candidates.append((layer_bytes, layer_name, sublayers_dict[layer_name], param_id_to_param))

        candidates.sort(key=lambda x: x[0], reverse=True)
        selected_cnt = 0
        for layer_bytes, layer_name, layer, param_id_to_param in candidates:
            if self._register_layer_state(
                layer_name=layer_name,
                layer=layer,
                param_id_to_param=param_id_to_param,
                layer_bytes=layer_bytes,
                source="param-name-fallback",
            ):
                selected_cnt += 1
        return selected_cnt

    @staticmethod
    def _is_param_fully_loaded(param: paddle.Tensor) -> bool:
        if not hasattr(param, "_is_initialized") or not param._is_initialized():
            return False
        if hasattr(param, "tensor_track") and param.tensor_track is not None:
            return param.tensor_track.is_fully_copied()
        return True

    @staticmethod
    def _attach_tensor_track_if_needed(param: paddle.Tensor) -> None:
        if hasattr(param, "tensor_track"):
            return
        if not hasattr(param, "output_dim"):
            return
        shape = tuple(param.shape)
        if len(shape) not in (2, 3):
            return
        try:
            param.tensor_track = TensorTracker(
                shape=shape,
                output_dim=bool(getattr(param, "output_dim", False)),
            )
        except Exception:
            # Best effort. If tracker cannot be attached, fallback to default behavior.
            pass

    def _wrap_layer_weight_loaders(self) -> None:
        wrapped_count = 0
        wrapped_default_loader_count = 0
        default_loader_fn = None
        total_selected_params = sum(len(layer_state.param_id_to_param) for layer_state in self.layer_states.values())
        for layer_state in self.layer_states.values():
            for param_id, param in layer_state.param_id_to_param.items():
                if param_id in self._wrapped_param_ids:
                    continue
                original_loader = getattr(param, "weight_loader", None)
                if original_loader is None:
                    if default_loader_fn is None:
                        from fastdeploy.model_executor.utils import (
                            default_weight_loader,
                        )

                        default_loader_fn = default_weight_loader(self.fd_config)
                    original_loader = default_loader_fn
                    wrapped_default_loader_count += 1
                wrapped_loader = self._make_weight_loader_wrapper(original_loader=original_loader)
                try:
                    param.weight_loader = wrapped_loader
                except Exception:
                    continue
                self._wrapped_param_ids.add(param_id)
                wrapped_count += 1
        if wrapped_count > 0:
            logger.info(
                f"[cpu-offload] Wrapped weight_loader for selected params: "
                f"{wrapped_count}/{total_selected_params}, default_loader_wrapped={wrapped_default_loader_count}"
            )
        if wrapped_count < total_selected_params:
            logger.warning(
                f"[cpu-offload] Only {wrapped_count}/{total_selected_params} selected params got wrapped "
                "weight_loader. Some params may defer offload until finalize."
            )

    def _init_partial_offload_watermark_bytes(self) -> int:
        if not (current_platform.is_cuda() or current_platform.is_maca()):
            return 0
        total_bytes = 0
        try:
            mem_info = paddle.device.cuda.mem_get_info()
            if isinstance(mem_info, (tuple, list)) and len(mem_info) >= 2:
                total_bytes = int(mem_info[1])
        except Exception:
            total_bytes = 0
        if total_bytes <= 0:
            return 0
        gpu_util = float(getattr(self.fd_config.cache_config, "gpu_memory_utilization", 0.9))
        # Keep partial-param offload as a high-pressure fallback to avoid load-time thrashing.
        watermark_util = min(0.92, max(0.80, gpu_util - 0.02))
        return int(total_bytes * watermark_util)

    def _should_offload_partial_param(self) -> bool:
        if self.partial_offload_watermark_bytes <= 0:
            return False
        snapshot = _get_cuda_memory_snapshot()
        if snapshot is None:
            return False
        curr_alloc, curr_reserved, _, _ = snapshot
        current_used_bytes = int(max(curr_alloc, curr_reserved) * (1024**3))
        should_offload = current_used_bytes >= self.partial_offload_watermark_bytes
        if should_offload and not self.partial_offload_activated:
            self.partial_offload_activated = True
            logger.warning(
                f"[cpu-offload] Activate partial-param offload at "
                f"{_bytes_to_gib_str(current_used_bytes)} "
                f"(watermark={_bytes_to_gib_str(self.partial_offload_watermark_bytes)})"
            )
        return should_offload

    def _make_weight_loader_wrapper(self, original_loader):
        def _wrapped(param, loaded_weight, *args, **kwargs):
            self.materialize_param_for_loading(param=param)
            result = original_loader(param, loaded_weight, *args, **kwargs)
            self.on_parameter_loaded(param=param)
            return result

        return _wrapped

    def log_memory(self, context: str, force: bool = False) -> None:
        snapshot = _get_cuda_memory_snapshot()
        if snapshot is None:
            return
        curr_alloc, curr_reserved, max_alloc, max_reserved = snapshot
        alloc_bytes = int(curr_alloc * (1024**3))
        reserved_bytes = int(curr_reserved * (1024**3))
        should_log = force
        if alloc_bytes >= self.next_alloc_log_step:
            self.next_alloc_log_step = (
                (alloc_bytes // _MEMORY_GROWTH_LOG_STEP_BYTES) + 1
            ) * _MEMORY_GROWTH_LOG_STEP_BYTES
            should_log = True
        if reserved_bytes >= self.next_reserved_log_step:
            self.next_reserved_log_step = (
                (reserved_bytes // _MEMORY_GROWTH_LOG_STEP_BYTES) + 1
            ) * _MEMORY_GROWTH_LOG_STEP_BYTES
            should_log = True
        if not should_log:
            return
        logger.info(
            f"[cpu-offload] {context} | current_alloc={curr_alloc:.3f} GiB "
            f"current_reserved={curr_reserved:.3f} GiB max_alloc={max_alloc:.3f} GiB "
            f"max_reserved={max_reserved:.3f} GiB"
        )

    def prepare_model(self, model: nn.Layer) -> None:
        if not self.enabled:
            if self.max_bytes > 0:
                logger.warning(
                    f"[cpu-offload] Disabled (cpu_offload_gb={self.fd_config.cache_config.cpu_offload_gb}): "
                    f"{self.disable_reason}"
                )
            return

        logger.info(f"[cpu-offload] Preparing model CPU weight offload: max={_bytes_to_gib_str(self.max_bytes)}")
        logger.info(f"[cpu-offload] Offload cache place resolved: {self.cpu_place_desc}")
        if self.partial_offload_watermark_bytes > 0:
            logger.info(
                f"[cpu-offload] partial-param watermark={_bytes_to_gib_str(self.partial_offload_watermark_bytes)}"
            )
        named_sublayers = list(model.named_sublayers())
        decoder_candidate_cnt = 0
        for layer_name, layer in named_sublayers:
            if not _is_decoder_layer(layer_name=layer_name, layer=layer):
                continue
            decoder_candidate_cnt += 1
            params = self._get_named_parameters(layer=layer, include_sublayers=True)
            if len(params) == 0:
                continue
            for param in params.values():
                self._attach_tensor_track_if_needed(param)
            param_id_to_param = {id(param): param for param in params.values()}
            layer_bytes = sum(self._param_num_bytes(param) for param in params.values())
            if layer_bytes <= 0:
                continue
            self._register_layer_state(
                layer_name=layer_name,
                layer=layer,
                param_id_to_param=param_id_to_param,
                layer_bytes=layer_bytes,
                source="decoder",
            )

        logger.info(
            f"[cpu-offload] layer discovery summary: total_sublayers={len(named_sublayers)} "
            f"decoder_candidates={decoder_candidate_cnt}"
        )

        if len(self.layer_states) == 0:
            logger.warning("[cpu-offload] No decoder layers were selected; " "fallback to parameter-owning sublayers.")
            selected_fallback = self._prepare_fallback_layers_by_direct_params(named_sublayers=named_sublayers)
            logger.info(f"[cpu-offload] direct-param fallback selected_layers={selected_fallback}")

        if len(self.layer_states) == 0:
            logger.warning(
                "[cpu-offload] Direct-param fallback selected nothing; " "fallback to parameter-name grouping."
            )
            selected_fallback = self._prepare_fallback_layers_by_param_name(
                model=model, named_sublayers=named_sublayers
            )
            logger.info(f"[cpu-offload] param-name fallback selected_layers={selected_fallback}")

        if len(self.layer_states) == 0:
            sample_names = ", ".join(name for name, _ in named_sublayers[:20])
            logger.warning(
                f"[cpu-offload] No layers were selected under max budget={_bytes_to_gib_str(self.max_bytes)}. "
                f"sublayer_sample=[{sample_names}]"
            )
            return
        self._wrap_layer_weight_loaders()
        self.log_memory(context="after selecting offload layers", force=True)

    def on_parameter_loaded(self, param: paddle.Tensor) -> None:
        if not self.enabled or param is None:
            return
        layer_name = self.param_id_to_layer_name.get(id(param))
        if layer_name is None:
            return
        layer_state = self.layer_states[layer_name]
        if self._is_param_fully_loaded(param):
            layer_state.loaded_param_ids.add(id(param))
        elif not self._should_offload_partial_param():
            return
        offloaded_bytes = self._offload_param_if_possible(layer_state=layer_state, param=param)
        if offloaded_bytes > 0 and not layer_state.partial_offload_logged:
            logger.info(
                f"[cpu-offload] Start param-level offload for {layer_state.name}, "
                f"cached_params={len(layer_state.cpu_param_cache)}/{len(layer_state.param_id_to_param)}"
            )
            layer_state.partial_offload_logged = True
        if len(layer_state.loaded_param_ids) == len(layer_state.param_id_to_param):
            self._mark_layer_fully_offloaded(layer_state=layer_state, reason="all-params-loaded")

    def finalize_after_weight_loading(self) -> None:
        if not self.enabled:
            return
        for layer_state in self.layer_states.values():
            if layer_state.offloaded:
                continue
            all_ready = True
            for param_id, param in layer_state.param_id_to_param.items():
                if param_id in layer_state.cpu_param_cache:
                    continue
                if not self._is_param_fully_loaded(param):
                    all_ready = False
                    break
            if all_ready:
                self._offload_layer(layer_state=layer_state, reason="post-load-finalize")
        logger.info(
            f"[cpu-offload] Finalized. selected={_bytes_to_gib_str(self.total_selected_bytes)} "
            f"offloaded={_bytes_to_gib_str(self.total_offloaded_bytes)} "
            f"layer_count={len(self.layer_states)}"
        )
        self.log_cached_tensor_place_stats(context="finalize_after_weight_loading")
        self.log_memory(context="after cpu-offload finalize", force=True)

    def log_cached_tensor_place_stats(self, context: str) -> None:
        if len(self.cache_tensor_place_counter) == 0:
            logger.warning(f"[cpu-offload] [{context}] no cached tensor place stats collected yet")
            return
        place_items = ", ".join(
            f"{place_name}:{count}" for place_name, count in sorted(self.cache_tensor_place_counter.items())
        )
        pinned_hits = 0
        total_hits = 0
        for place_name, count in self.cache_tensor_place_counter.items():
            total_hits += count
            if "CUDAPinnedPlace" in place_name:
                pinned_hits += count
        logger.info(
            f"[cpu-offload] [{context}] cache_place_target={self.cpu_place_desc}; "
            f"cache_place_stats={place_items}; pinned_hits={pinned_hits}/{total_hits}"
        )

    def _resolve_layer_state_for_sublayer(self, sublayer_name: str) -> _LayerOffloadState | None:
        owner_name = self.postprocess_owner_cache.get(sublayer_name, "__MISS__")
        if owner_name != "__MISS__":
            if owner_name is None:
                return None
            return self.layer_states.get(owner_name)

        if sublayer_name in self.layer_states:
            self.postprocess_owner_cache[sublayer_name] = sublayer_name
            return self.layer_states[sublayer_name]

        best_owner_name = None
        best_owner_len = -1
        for layer_name in self.layer_states.keys():
            if sublayer_name.startswith(layer_name + ".") and len(layer_name) > best_owner_len:
                best_owner_name = layer_name
                best_owner_len = len(layer_name)

        self.postprocess_owner_cache[sublayer_name] = best_owner_name
        if best_owner_name is None:
            return None
        return self.layer_states.get(best_owner_name)

    def resolve_postprocess_owner_name(self, sublayer_name: str) -> str | None:
        layer_state = self._resolve_layer_state_for_sublayer(sublayer_name)
        if layer_state is None:
            return None
        return layer_state.name

    def materialize_layer_for_postprocess(self, sublayer_name: str) -> None:
        if not self.enabled:
            return
        layer_state = self._resolve_layer_state_for_sublayer(sublayer_name)
        if layer_state is None:
            return
        if len(layer_state.cpu_param_cache) == 0:
            return
        for param_id, param in layer_state.param_id_to_param.items():
            cpu_tensor = layer_state.cpu_param_cache.get(param_id)
            if cpu_tensor is None:
                continue
            self._materialize_single_param(param=param, cpu_tensor=cpu_tensor, context=sublayer_name)
        layer_state.offloaded = False

    def materialize_param_for_loading(self, param: paddle.Tensor) -> None:
        if not self.enabled or param is None:
            return
        param_id = id(param)
        layer_name = self.param_id_to_layer_name.get(param_id)
        if layer_name is None:
            return
        layer_state = self.layer_states.get(layer_name)
        if layer_state is None:
            return
        cpu_tensor = layer_state.cpu_param_cache.get(param_id)
        if cpu_tensor is None:
            return
        self._materialize_single_param(param=param, cpu_tensor=cpu_tensor, context=f"{layer_name}/weight-load")
        del layer_state.cpu_param_cache[param_id]
        layer_state.loaded_param_ids.discard(param_id)
        layer_state.offloaded = False

    def reoffload_layer_after_postprocess(self, sublayer_name: str) -> None:
        if not self.enabled:
            return
        layer_state = self._resolve_layer_state_for_sublayer(sublayer_name)
        if layer_state is None or layer_state.offloaded:
            return
        self._refresh_layer_state_params(layer_state=layer_state)
        self._offload_layer(layer_state=layer_state, reason="post-process-layer")

    def _refresh_layer_state_params(self, layer_state: _LayerOffloadState) -> None:
        params = self._get_named_parameters(layer=layer_state.layer, include_sublayers=True)
        if len(params) == 0:
            return
        new_param_id_to_param = {id(param): param for param in params.values()}
        old_param_ids = list(layer_state.param_id_to_param.keys())
        for old_id in old_param_ids:
            if self.param_id_to_layer_name.get(old_id) == layer_state.name:
                del self.param_id_to_layer_name[old_id]
        for new_id, param in new_param_id_to_param.items():
            self._attach_tensor_track_if_needed(param)
            self.param_id_to_layer_name[new_id] = layer_state.name
        layer_state.param_id_to_param = new_param_id_to_param
        layer_state.loaded_param_ids = set()
        layer_state.cpu_param_cache = {}

    def _offload_layer(self, layer_state: _LayerOffloadState, reason: str) -> None:
        offloaded_this_layer = 0
        for param_id, param in layer_state.param_id_to_param.items():
            if param_id in layer_state.cpu_param_cache:
                continue
            offloaded_this_layer += self._offload_param_if_possible(layer_state=layer_state, param=param)

        if offloaded_this_layer == 0 and len(layer_state.cpu_param_cache) == 0:
            return

        if len(layer_state.cpu_param_cache) == len(layer_state.param_id_to_param):
            self._mark_layer_fully_offloaded(
                layer_state=layer_state, reason=reason, offloaded_this_layer=offloaded_this_layer
            )
        elif offloaded_this_layer > 0:
            if current_platform.is_cuda() or current_platform.is_maca():
                try:
                    paddle.device.cuda.empty_cache()
                except Exception:
                    pass
            self.log_memory(context=f"after partial offloading {layer_state.name}", force=False)

    def _ensure_layer_hooks(self, layer_state: _LayerOffloadState) -> None:
        if layer_state.pre_hook_handle is not None:
            return
        layer_state.pre_hook_handle = layer_state.layer.register_forward_pre_hook(
            self._build_pre_hook(layer_state),
        )
        layer_state.post_hook_handle = layer_state.layer.register_forward_post_hook(
            self._build_post_hook(layer_state),
        )

    def _build_pre_hook(self, layer_state: _LayerOffloadState):
        def _pre_hook(layer, inputs):
            for param_id, param in layer_state.param_id_to_param.items():
                cpu_tensor = layer_state.cpu_param_cache.get(param_id)
                if cpu_tensor is None:
                    continue
                self._materialize_single_param(param=param, cpu_tensor=cpu_tensor, context=layer_state.name)
            layer_state.pre_hook_calls += 1
            if layer_state.pre_hook_calls <= 3:
                self.log_memory(
                    context=f"runtime pre-hook materialize {layer_state.name} ({layer_state.pre_hook_calls})",
                    force=False,
                )
            return None

        return _pre_hook

    def _build_post_hook(self, layer_state: _LayerOffloadState):
        def _post_hook(layer, inputs, outputs):
            for param_id in layer_state.cpu_param_cache.keys():
                param = layer_state.param_id_to_param.get(param_id)
                if param is None:
                    continue
                if hasattr(param, "_is_initialized") and param._is_initialized():
                    param._clear_data()
            layer_state.post_hook_calls += 1
            if layer_state.post_hook_calls <= 3:
                self.log_memory(
                    context=f"runtime post-hook evict {layer_state.name} ({layer_state.post_hook_calls})",
                    force=False,
                )
            return outputs

        return _post_hook

    @staticmethod
    def _materialize_single_param(param: paddle.Tensor, cpu_tensor: paddle.Tensor, context: str) -> None:
        if cpu_tensor is None:
            return
        try:
            if hasattr(param, "_is_initialized") and not param._is_initialized():
                param.initialize()
        except Exception:
            pass
        try:
            param.copy_(cpu_tensor, True)
            return
        except Exception:
            pass
        try:
            param.set_value(cpu_tensor)
            return
        except Exception as ex:
            logger.warning(f"[cpu-offload] Failed to materialize param for {context}: {ex}")

    def _offload_param_if_possible(self, layer_state: _LayerOffloadState, param: paddle.Tensor) -> int:
        param_id = id(param)
        if param_id in layer_state.cpu_param_cache:
            return 0
        if not hasattr(param, "_is_initialized") or not param._is_initialized():
            return 0
        try:
            # Keep pinned-cache tensors as raw storage only; avoid tensor ops
            # (e.g., slice/transpose) on cached tensors to prevent GPU fallback.
            cpu_tensor = param._copy_to(self.cpu_place, True)
        except Exception:
            return 0
        layer_state.cpu_param_cache[param_id] = cpu_tensor
        cache_place = self._tensor_place_to_str(cpu_tensor)
        self.cache_tensor_place_counter[cache_place] = self.cache_tensor_place_counter.get(cache_place, 0) + 1
        if self.cache_tensor_place_log_count < 8:
            logger.info(
                f"[cpu-offload] cached tensor place sample: "
                f"layer={layer_state.name} cache_place={cache_place} target_place={self.cpu_place_desc}"
            )
            self.cache_tensor_place_log_count += 1
        offloaded_bytes = self._param_num_bytes(param)
        layer_state.param_offloaded_bytes += offloaded_bytes
        try:
            param._clear_data()
        except Exception:
            pass
        self.pending_param_offload_bytes += offloaded_bytes
        if self.pending_param_offload_bytes >= _PARAM_OFFLOAD_EMPTY_CACHE_STEP_BYTES and (
            current_platform.is_cuda() or current_platform.is_maca()
        ):
            try:
                paddle.device.cuda.empty_cache()
            except Exception:
                pass
            self.pending_param_offload_bytes = 0
            self.log_memory(context=f"after param-level offload {layer_state.name}", force=False)
        self._ensure_layer_hooks(layer_state=layer_state)
        return offloaded_bytes

    def _mark_layer_fully_offloaded(
        self,
        layer_state: _LayerOffloadState,
        reason: str,
        offloaded_this_layer: int = 0,
    ) -> None:
        layer_state.offloaded = True
        if not layer_state.counted_in_total:
            self.total_offloaded_bytes += layer_state.total_bytes
            layer_state.counted_in_total = True
        if current_platform.is_cuda() or current_platform.is_maca():
            try:
                paddle.device.cuda.empty_cache()
            except Exception:
                pass
        logger.info(
            f"[cpu-offload] Offloaded layer {layer_state.name} "
            f"({_bytes_to_gib_str(layer_state.total_bytes)}), reason={reason}, "
            f"new_offloaded={_bytes_to_gib_str(offloaded_this_layer)}, "
            f"total_offloaded={_bytes_to_gib_str(self.total_offloaded_bytes)}"
        )
        self.log_memory(context=f"after offloading {layer_state.name}", force=False)


def prepare_model_cpu_weight_offload(model: nn.Layer, fd_config: FDConfig) -> None:
    global _CPU_WEIGHT_OFFLOAD_MANAGER, _CPU_OFFLOAD_MAX_BYTES
    _CPU_OFFLOAD_MAX_BYTES = max(0, int(fd_config.cache_config.cpu_offload_gb * 1024**3))
    if _CPU_OFFLOAD_MAX_BYTES <= 0:
        _CPU_WEIGHT_OFFLOAD_MANAGER = None
        return
    manager = CPUWeightOffloadManager(max_bytes=_CPU_OFFLOAD_MAX_BYTES, fd_config=fd_config)
    _CPU_WEIGHT_OFFLOAD_MANAGER = manager
    manager.prepare_model(model=model)


def maybe_offload_layer_after_weight_loading(param: paddle.Tensor) -> None:
    manager = _CPU_WEIGHT_OFFLOAD_MANAGER
    if manager is None:
        return
    manager.on_parameter_loaded(param=param)


def maybe_materialize_param_for_loading(param: paddle.Tensor) -> None:
    manager = _CPU_WEIGHT_OFFLOAD_MANAGER
    if manager is None:
        return
    manager.materialize_param_for_loading(param=param)


def finalize_cpu_weight_offload() -> None:
    manager = _CPU_WEIGHT_OFFLOAD_MANAGER
    if manager is None:
        return
    manager.finalize_after_weight_loading()
    manager.log_cached_tensor_place_stats(context="after finalize_cpu_weight_offload()")


def log_model_parameter_place_stats(model: nn.Layer, context: str, sample_limit: int = 12) -> None:
    manager = _CPU_WEIGHT_OFFLOAD_MANAGER
    target_place = manager.cpu_place_desc if manager is not None else "offload-manager-none"
    place_counter: dict[str, int] = {}
    samples: list[str] = []
    total_params = 0
    for param_name, param in model.named_parameters():
        total_params += 1
        try:
            if hasattr(param, "_is_initialized") and not param._is_initialized():
                place_name = "uninitialized"
            else:
                place_name = str(param.place)
        except Exception as ex:
            place_name = f"no-memory:{type(ex).__name__}"
        place_counter[place_name] = place_counter.get(place_name, 0) + 1
        if len(samples) < sample_limit:
            samples.append(f"{param_name}=>{place_name}")

    place_items = ", ".join(f"{name}:{count}" for name, count in sorted(place_counter.items()))
    logger.info(
        f"[cpu-offload] [{context}] model parameter places: total={total_params}; "
        f"offload_cache_target={target_place}; stats={place_items}"
    )
    if len(samples) > 0:
        logger.info(f"[cpu-offload] [{context}] model parameter place samples: {' | '.join(samples)}")


def maybe_materialize_layer_for_postprocess(layer_name: str) -> None:
    manager = _CPU_WEIGHT_OFFLOAD_MANAGER
    if manager is None:
        return
    manager.materialize_layer_for_postprocess(sublayer_name=layer_name)


def maybe_reoffload_layer_after_postprocess(layer_name: str) -> None:
    manager = _CPU_WEIGHT_OFFLOAD_MANAGER
    if manager is None:
        return
    manager.reoffload_layer_after_postprocess(sublayer_name=layer_name)


def get_cpu_offload_postprocess_owner(layer_name: str) -> str | None:
    manager = _CPU_WEIGHT_OFFLOAD_MANAGER
    if manager is None:
        return None
    return manager.resolve_postprocess_owner_name(sublayer_name=layer_name)
