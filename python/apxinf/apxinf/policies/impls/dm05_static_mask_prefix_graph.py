"""Exact static-mask prefix and native-suffix CUDA Graph runtime for DM05.

This executor is a deliberately narrow, default-off production runtime for the
frozen DM05-libero request.  The first eligible request supplies an opaque,
CPU-owned mask-layout key.  Under one lock the runtime binds that key, calls
Transformers' original Gemma3 causal-mask owner twice (retained plus oracle),
captures the ordinary vision/projector/language prefix into an address-stable
34-layer cache, and captures the unchanged ten-step native suffix.  Later
requests with the same layout key may change ordinary token and image content.

There is no eager fallback.  History, layout drift, source drift, model
sharding, and any tensor-contract drift fail closed.  Random noise remains an
ordinary eager global-generator draw between the prefix and suffix replays, so
the accepted native seed/order contract is unchanged.
"""

from __future__ import annotations

import hashlib
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from types import MethodType
from typing import Any, Iterator

import torch
import transformers.masking_utils as transformers_masking_utils
import transformers.models.gemma3.modeling_gemma3 as gemma3_modeling
from transformers.cache_utils import Cache, CacheLayerMixin, DynamicCache
from transformers.models.gemma3.modeling_gemma3 import (
    create_causal_mask_mapping,
)

import opendm.model.dm05.dm05_arch as dm05_arch
from opendm.model.dm05.dm05_utils import (
    HISTORY_PAD_TOKEN_ID,
    mask_history_pad_tokens_in_attention,
)

__all__ = ["DM05StaticMaskPrefixGraphRuntime"]


_SCHEMA = "apxinf.dm05.static-prefix-base.v1"
_EXECUTION_BACKEND = "default_exact_combined"
_ARITHMETIC_BACKEND = "default_eager_sdpa"
_GRAPH_SCOPE = "native_prefix_564_plus_suffix_10step"
_MASK_OWNER_SYMBOL = (
    "transformers.models.gemma3.modeling_gemma3."
    "create_causal_mask_mapping"
)
_MODELING_GEMMA3_SHA256 = (
    "a1115edf9e0c4a3b53657f21e2de5de0a99488767d84181db6e05e082adb4f69"
)
_MASKING_UTILS_SHA256 = (
    "c3c82f7b7b6e03d3f04ba6c6c58a3dd6910623636452ec67ef70e3eb522f9fe7"
)
_DM05_ARCH_SHA256 = (
    "b5ab170374fbc965aa86d7d370075e8c8bc21bcf46bc6de34e7e336df1af9ce8"
)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tensor_sha256(value: torch.Tensor) -> str:
    """Hash logical tensor bytes, including BF16 on NumPy builds without BF16."""
    array = value.detach().contiguous().view(torch.uint8).cpu().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def _tensor_contract(value: torch.Tensor) -> tuple[Any, ...]:
    return (
        tuple(int(item) for item in value.shape),
        tuple(int(item) for item in value.stride()),
        value.dtype,
        value.device,
    )


def _same_shape_dtype_device(
    left: torch.Tensor,
    right: torch.Tensor,
) -> bool:
    return (
        left.shape == right.shape
        and left.dtype == right.dtype
        and left.device == right.device
    )


def _official_euler_suffix(
    model: Any,
    *,
    input_ids: Any,
    prefix_len: int,
    past_key_values: Any,
    initial_noise: Any,
    diffusion_steps: int,
    action_mask: Any,
) -> Any:
    """Run the exact Euler suffix from official OpenDM e41e501.

    Every global operation is resolved through the official ``dm05_arch``
    module. The combined capture runtime can therefore bind its temporary
    metadata/KV/affine specializations without modifying the official source.
    """

    batch_size = int(input_ids.shape[0])
    device = input_ids.device
    dtype = model.model.action_in_proj.weight.dtype
    x_t = initial_noise
    time_value = 1.0
    dt = -1.0 / int(diffusion_steps)
    for _ in range(int(diffusion_steps)):
        time_tensor = dm05_arch.torch.full(
            (batch_size,), time_value, device=device, dtype=dtype
        )
        if action_mask is not None:
            x_t = x_t * action_mask
        suffix_embeds = model.model.action_in_proj(x_t)
        adarms_cond = model._build_adarms_cond(time_tensor, suffix_embeds.dtype)
        suffix_len = int(suffix_embeds.shape[1])
        invisible_prefix_token_ids = (dm05_arch.HISTORY_PAD_TOKEN_ID,)
        suffix_attn_mask = dm05_arch.make_suffix_attn_mask(
            input_ids=input_ids,
            prefix_len=prefix_len,
            suffix_len=suffix_len,
            batch_size=batch_size,
            device=suffix_embeds.device,
            dtype=suffix_embeds.dtype,
            pad_token_id=model.model.vlm.model.language_model.padding_idx,
            invisible_prefix_token_ids=invisible_prefix_token_ids,
        )
        suffix_position_ids = model._build_suffix_position_ids(
            prefix_len,
            suffix_len,
            device,
            input_ids=input_ids,
            pad_token_id=model.model.vlm.model.language_model.padding_idx,
            invisible_prefix_token_ids=invisible_prefix_token_ids,
        )
        suffix_out = model._suffix_forward(
            suffix_embeds=suffix_embeds,
            attention_mask=suffix_attn_mask,
            position_ids=suffix_position_ids,
            past_key_values=past_key_values,
            adarms_cond=adarms_cond,
        )
        velocity = model.model.action_out_proj(suffix_out)
        x_t = x_t + velocity * dt
        time_value += dt
    return x_t


class _StaticPrefixCacheLayer(CacheLayerMixin):
    """One address-stable Gemma3 cache layer owned by the prefix graph."""

    is_sliding = False
    is_compileable = True

    def __init__(
        self,
        reference_layer: Any,
        key_reference: torch.Tensor,
        value_reference: torch.Tensor,
        on_tensor_copy: Any,
    ) -> None:
        super().__init__()
        self.is_sliding = bool(getattr(reference_layer, "is_sliding", False))
        self.sliding_window = int(
            getattr(reference_layer, "sliding_window", 0) or 0
        )
        self.keys = torch.empty_strided(
            key_reference.shape,
            key_reference.stride(),
            device=key_reference.device,
            dtype=key_reference.dtype,
        )
        self.values = torch.empty_strided(
            value_reference.shape,
            value_reference.stride(),
            device=value_reference.device,
            dtype=value_reference.dtype,
        )
        torch._dynamo.mark_static_address(self.keys)
        torch._dynamo.mark_static_address(self.values)
        self.dtype = key_reference.dtype
        self.device = key_reference.device
        self.is_initialized = True
        self.seq_len = 0
        self.cumulative_length = 0
        self._on_tensor_copy = on_tensor_copy

    def reset_for_prefill(self) -> None:
        self.seq_len = 0
        self.cumulative_length = 0

    def reset(self) -> None:
        self.reset_for_prefill()

    def lazy_initialization(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
    ) -> None:
        if not _same_shape_dtype_device(self.keys, key_states) or not (
            _same_shape_dtype_device(self.values, value_states)
        ):
            raise RuntimeError(
                "DM05 static prefix cache lazy-initialization contract drifted."
            )
        self.seq_len = 0

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cache_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del cache_kwargs
        if not _same_shape_dtype_device(self.keys, key_states):
            raise RuntimeError("DM05 static prefix K contract drifted.")
        if not _same_shape_dtype_device(self.values, value_states):
            raise RuntimeError("DM05 static prefix V contract drifted.")
        self.keys.copy_(key_states)
        self.values.copy_(value_states)
        self._on_tensor_copy(2)
        self.seq_len = int(key_states.shape[-2])
        self.cumulative_length = int(key_states.shape[-2])
        return self.keys, self.values

    def get_mask_sizes(self, cache_position: torch.Tensor) -> tuple[int, int]:
        query_length = int(cache_position.shape[0])
        if self.is_sliding:
            if self.sliding_window <= 0:
                raise RuntimeError("DM05 sliding cache omitted sliding_window.")
            is_full = self.cumulative_length >= self.sliding_window
            kv_offset = max(
                self.cumulative_length - self.sliding_window + 1,
                0,
            )
            if is_full:
                return self.sliding_window - 1 + query_length, kv_offset
            return self.cumulative_length + query_length, kv_offset
        return self.seq_len + query_length, 0

    def get_seq_length(self) -> int:
        if self.is_sliding:
            return int(self.cumulative_length)
        return int(self.seq_len)

    def get_max_cache_shape(self) -> int:
        if self.is_sliding:
            return int(self.sliding_window)
        return int(self.keys.shape[-2])


class DM05StaticMaskPrefixGraphRuntime:
    """Capture the exact DM05 prefix and suffix for one frozen LIBERO shape.

    ``mask_layout_key`` is intentionally opaque to this runtime. Its CPU owner must
    include all real mask dependencies and exclude ordinary token/image content.
    ApxInf binds the first key and only checks equality thereafter; this keeps
    dynamic same-layout content eligible without rebuilding or guessing masks.
    """

    _REQUEST_PREFIX_LEN = 564
    _DIFFUSION_STEPS = 10
    _BATCH_SIZE = 1
    _CACHE_LAYER_COUNT = 34
    _WARMUP_COUNT = 2
    _EXPECTED_MASK_KEYS = frozenset(
        {"full_attention", "sliding_attention"}
    )

    def __init__(
        self,
        model: Any,
        *,
        request_prefix_len: int = _REQUEST_PREFIX_LEN,
        diffusion_steps: int = _DIFFUSION_STEPS,
    ) -> None:
        if int(request_prefix_len) != self._REQUEST_PREFIX_LEN:
            raise ValueError(
                "DM05StaticMaskPrefixGraphRuntime is frozen to "
                f"request_prefix_len=564, got {request_prefix_len}."
            )
        if int(diffusion_steps) != self._DIFFUSION_STEPS:
            raise ValueError(
                "DM05StaticMaskPrefixGraphRuntime is frozen to "
                f"diffusion_steps=10, got {diffusion_steps}."
            )
        if model.training:
            raise ValueError(
                "DM05StaticMaskPrefixGraphRuntime requires model.eval()."
            )
        if not torch.cuda.is_available():
            raise RuntimeError(
                "DM05StaticMaskPrefixGraphRuntime requires CUDA."
            )

        self.mask_modeling_source_sha256 = _file_sha256(
            Path(gemma3_modeling.__file__)
        )
        self.mask_utils_source_sha256 = _file_sha256(
            Path(transformers_masking_utils.__file__)
        )
        self.dm05_arch_source_sha256 = _file_sha256(Path(dm05_arch.__file__))
        if self.mask_modeling_source_sha256 != _MODELING_GEMMA3_SHA256:
            raise RuntimeError(
                "DM05 Gemma3 mask-owner source drifted: "
                f"{self.mask_modeling_source_sha256} != "
                f"{_MODELING_GEMMA3_SHA256}."
            )
        if self.mask_utils_source_sha256 != _MASKING_UTILS_SHA256:
            raise RuntimeError(
                "DM05 Transformers masking source drifted: "
                f"{self.mask_utils_source_sha256} != "
                f"{_MASKING_UTILS_SHA256}."
            )
        if self.dm05_arch_source_sha256 != _DM05_ARCH_SHA256:
            raise RuntimeError(
                "DM05 official action source drifted: "
                f"{self.dm05_arch_source_sha256} != {_DM05_ARCH_SHA256}."
            )

        self.model = model
        self.request_prefix_len = int(request_prefix_len)
        self.diffusion_steps = int(diffusion_steps)
        self.batch_size = self._BATCH_SIZE
        self._lock = threading.Lock()
        self._closed = False
        self._initialization_failed = False

        parameter = next(model.parameters())
        self.device = parameter.device
        if self.device.type != "cuda":
            raise RuntimeError(
                "DM05StaticMaskPrefixGraphRuntime requires a CUDA-resident model."
            )
        parameter_devices = {value.device for value in model.parameters()}
        if parameter_devices != {self.device}:
            raise RuntimeError(
                "DM05StaticMaskPrefixGraphRuntime does not support "
                "CPU/disk/model sharding: "
                f"{sorted(map(str, parameter_devices))}."
            )

        action_expert = model.model.action_expert
        action_backend = getattr(action_expert, "_suffix_attn_backend", None)
        if action_backend != "sdpa":
            raise RuntimeError(
                "DM05StaticMaskPrefixGraphRuntime requires the accepted native "
                f"SDPA action backend, got {action_backend!r}."
            )
        self.cache_layer_count = len(action_expert.layers)
        if self.cache_layer_count != self._CACHE_LAYER_COUNT:
            raise RuntimeError(
                "DM05StaticMaskPrefixGraphRuntime requires exactly 34 cache "
                f"layers, got {self.cache_layer_count}."
            )

        self.action_dim = int(model.model.config.action_dim)
        self.chunk_size = int(model.model.config.chunk_size)
        self.noise_dtype = model.model.action_in_proj.weight.dtype

        self.initialized = False
        self.initialization_ms: float | None = None
        self.mask_layout_key_verified = False
        self.mask_helper_build_count = 0
        self.mask_mapping_keys: list[str] = []
        self.mask_static_address_verified = False
        self.mask_immutable_verified = False
        self.prefix_static_cache_address_verified = False
        self.suffix_static_output_address_verified = False

        self.prefix_startup_capture_count = 0
        self.suffix_startup_capture_count = 0
        self.prefix_capture_execution_count = 0
        self.suffix_capture_execution_count = 0
        self.prefix_input_stage_requests = 0
        self.prefix_input_tensor_copies = 0
        self.prefix_graph_replay_count = 0
        self.prefix_graph_cache_write_tensor_copies = 0
        self.eager_noise_count = 0
        self.suffix_input_stage_requests = 0
        self.suffix_input_tensor_copies = 0
        self.suffix_graph_replay_count = 0
        self.prefix_eager_count = 0
        self.post_prefix_cache_stage_requests = 0
        self.post_prefix_cache_tensor_copies = 0
        self.fallback_count = 0
        self.history_count = 0
        self.result_reuse_count = 0
        self.request_prefix_length: int | None = None
        self.selected_prefix_length: int | None = None

        self._mask_layout_key: str | None = None
        self._input_contracts: dict[str, tuple[Any, ...]] = {}
        self._static_inputs: dict[str, torch.Tensor] = {}
        self._static_cache: Cache | None = None
        self._static_cache_addresses: tuple[int, ...] = ()
        self._static_action_mask: torch.Tensor | None = None
        self._static_noise: torch.Tensor | None = None
        self._static_output: torch.Tensor | None = None
        self._static_output_address: int | None = None
        self._prefix_graph: torch.cuda.CUDAGraph | None = None
        self._suffix_graph: torch.cuda.CUDAGraph | None = None
        self._retained_mask_owner: dict[str, Any] | None = None
        self._oracle_mask_owner: dict[str, Any] | None = None
        self._retained_mask_addresses: dict[str, int | None] = {}
        self._retained_mask_states: dict[str, dict[str, Any] | None] = {}
        self._oracle_mask_states: dict[str, dict[str, Any] | None] = {}
        self._capture_cache_write_observation = 0
        self._captured_prefix_cache_writes = 0
        self._image_feature_count: int | None = None
        self._cache_layer_semantics_state: list[tuple[Any, ...]] = []

    @staticmethod
    def _require_shape(
        value: torch.Tensor,
        expected: tuple[int, ...],
        *,
        label: str,
    ) -> None:
        actual = tuple(int(dimension) for dimension in value.shape)
        if actual != expected:
            raise ValueError(f"{label} must have shape {expected}, got {actual}.")

    def _validate_request(
        self,
        *,
        input_ids: torch.Tensor | None,
        attention_mask: torch.Tensor | None,
        pixel_values: torch.Tensor | None,
        token_type_ids: torch.Tensor | None,
        diffusion_steps: int,
        past_key_values: Cache | None,
        action_mask: torch.Tensor | None,
        history_pixel_values: torch.Tensor | None,
        history_mask: torch.Tensor | None,
        states: torch.Tensor | None,
        image_masks: torch.Tensor | None,
        mask_layout_key: str | None,
        unsupported: dict[str, Any],
    ) -> None:
        if self._closed:
            raise RuntimeError("DM05StaticMaskPrefixGraphRuntime is closed.")
        if self._initialization_failed:
            raise RuntimeError(
                "DM05StaticMaskPrefixGraphRuntime initialization previously failed."
            )
        if unsupported:
            raise TypeError(
                "DM05StaticMaskPrefixGraphRuntime does not accept extra options: "
                f"{sorted(unsupported)}."
            )
        if states is not None or image_masks is not None:
            raise ValueError(
                "DM05StaticMaskPrefixGraphRuntime does not support states or "
                "image_masks."
            )
        if past_key_values is not None:
            raise ValueError(
                "DM05StaticMaskPrefixGraphRuntime does not accept a prefix cache."
            )
        if history_pixel_values is not None or history_mask is not None:
            raise ValueError(
                "DM05StaticMaskPrefixGraphRuntime does not support history."
            )
        if int(diffusion_steps) != self.diffusion_steps:
            raise ValueError(
                "DM05StaticMaskPrefixGraphRuntime was captured for "
                f"diffusion_steps=10, got {diffusion_steps}."
            )
        if not isinstance(mask_layout_key, str) or not mask_layout_key:
            raise ValueError(
                "DM05StaticMaskPrefixGraphRuntime requires a non-empty opaque "
                "mask_layout_key."
            )

        required = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "token_type_ids": token_type_ids,
            "action_mask": action_mask,
        }
        missing = sorted(name for name, value in required.items() if value is None)
        if missing:
            raise ValueError(
                "DM05StaticMaskPrefixGraphRuntime is missing required tensors: "
                f"{missing}."
            )

        assert input_ids is not None
        assert attention_mask is not None
        assert pixel_values is not None
        assert token_type_ids is not None
        assert action_mask is not None
        self._require_shape(
            input_ids,
            (self.batch_size, self.request_prefix_len),
            label="input_ids",
        )
        self._require_shape(
            attention_mask,
            (self.batch_size, self.request_prefix_len),
            label="attention_mask",
        )
        self._require_shape(
            token_type_ids,
            (self.batch_size, self.request_prefix_len),
            label="token_type_ids",
        )
        self._require_shape(
            action_mask,
            (self.batch_size, 1, self.action_dim),
            label="action_mask",
        )
        for label, value in required.items():
            assert value is not None
            if value.device != self.device:
                raise ValueError(
                    f"{label} must be on {self.device}, got {value.device}."
                )
        if input_ids.dtype != torch.long:
            raise TypeError(f"input_ids must use torch.long, got {input_ids.dtype}.")
        if token_type_ids.dtype != torch.long:
            raise TypeError(
                "token_type_ids must use torch.long, got "
                f"{token_type_ids.dtype}."
            )
        if action_mask.dtype != self.noise_dtype:
            raise TypeError(
                f"action_mask must use {self.noise_dtype}, got "
                f"{action_mask.dtype}."
            )

        if self.initialized:
            if mask_layout_key != self._mask_layout_key:
                self.mask_layout_key_verified = False
                raise RuntimeError(
                    "DM05 static-mask prefix graph mask_layout_key drifted."
                )
            for name, value in (
                ("input_ids", input_ids),
                ("attention_mask", attention_mask),
                ("pixel_values", pixel_values),
                ("token_type_ids", token_type_ids),
            ):
                if _tensor_contract(value) != self._input_contracts[name]:
                    raise RuntimeError(
                        "DM05 static-mask prefix graph input contract drifted for "
                        f"{name}."
                    )
            self.mask_layout_key_verified = True

    @staticmethod
    def _cache_tensors(
        cache: Cache,
    ) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
        layers = tuple(cache.layers)
        return (
            tuple(layer.keys for layer in layers),
            tuple(layer.values for layer in layers),
        )

    def _observe_cache_writes(self, count: int) -> None:
        self._capture_cache_write_observation += int(count)

    def _make_static_cache(self, reference_cache: Cache) -> Cache:
        reference_keys, reference_values = self._cache_tensors(reference_cache)
        if (
            len(reference_keys) != self.cache_layer_count
            or len(reference_values) != self.cache_layer_count
            or any(value is None for value in reference_keys + reference_values)
        ):
            raise RuntimeError(
                "DM05 PFX_KV_564 reference requires 34 initialized K/V pairs."
            )
        return Cache(
            layers=[
                _StaticPrefixCacheLayer(
                    layer,
                    key,
                    value,
                    self._observe_cache_writes,
                )
                for layer, key, value in zip(
                    reference_cache.layers,
                    reference_keys,
                    reference_values,
                    strict=True,
                )
            ]
        )

    def _reset_static_cache(self) -> None:
        assert self._static_cache is not None
        for layer in self._static_cache.layers:
            layer.reset_for_prefill()

    def _static_tensor_like(self, value: torch.Tensor) -> torch.Tensor:
        result = torch.empty_strided(
            value.shape,
            value.stride(),
            device=value.device,
            dtype=value.dtype,
        )
        result.copy_(value)
        torch._dynamo.mark_static_address(result)
        return result

    def _prefix_metadata(
        self,
        request_inputs: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        prefix_attention_mask = mask_history_pad_tokens_in_attention(
            input_ids=request_inputs["input_ids"],
            attention_mask=request_inputs["attention_mask"],
        )
        assert prefix_attention_mask is not None
        prefix_position_ids = (
            prefix_attention_mask.cumsum(dim=-1) - 1
        ).clamp_min_(0)
        cache_position = torch.arange(self.request_prefix_len, device=self.device)
        return prefix_attention_mask, prefix_position_ids, cache_position

    def _cache_layer_semantics(self) -> list[tuple[Any, ...]]:
        assert self._static_cache is not None
        language_config = self.model.model.language_model.config
        reference = DynamicCache(config=language_config)
        cache_position = torch.arange(self.request_prefix_len, device=self.device)
        semantics: list[tuple[Any, ...]] = []
        for index, (reference_layer, static_layer) in enumerate(
            zip(reference.layers, self._static_cache.layers, strict=True)
        ):
            reference_sizes = tuple(
                int(value)
                for value in reference_layer.get_mask_sizes(cache_position)
            )
            static_sizes = tuple(
                int(value)
                for value in static_layer.get_mask_sizes(cache_position)
            )
            record = (
                index,
                bool(static_layer.is_sliding),
                int(getattr(static_layer, "sliding_window", 0) or 0),
                static_sizes,
            )
            if (
                bool(reference_layer.is_sliding) != bool(static_layer.is_sliding)
                or reference_sizes != static_sizes
            ):
                raise RuntimeError(
                    f"DM05 static cache layer {index} mask semantics drifted."
                )
            semantics.append(record)
        return semantics

    def _build_original_mask_owner(self, label: str) -> dict[str, Any]:
        assert self._static_cache is not None
        vlm_model = self.model.model.vlm.model
        self._reset_static_cache()
        (
            prefix_attention_mask,
            prefix_position_ids,
            cache_position,
        ) = self._prefix_metadata(self._static_inputs)
        llm_input_ids = self._static_inputs["input_ids"]
        image_token_id = int(vlm_model.config.image_token_id)
        if image_token_id >= int(vlm_model.vocab_size):
            special_image_mask = llm_input_ids == image_token_id
            llm_input_ids = llm_input_ids.clone()
            llm_input_ids[special_image_mask] = 0

        with torch.inference_mode(False), torch.no_grad():
            inputs_embeds = vlm_model.get_input_embeddings()(llm_input_ids)
            mapping = create_causal_mask_mapping(
                config=vlm_model.config,
                inputs_embeds=inputs_embeds,
                attention_mask=prefix_attention_mask,
                cache_position=cache_position,
                past_key_values=self._static_cache,
                position_ids=prefix_position_ids,
                token_type_ids=self._static_inputs["token_type_ids"],
                pixel_values=self._static_inputs["pixel_values"],
                is_training=False,
                is_first_iteration=True,
            )
        self.mask_helper_build_count += 1
        if set(mapping) != self._EXPECTED_MASK_KEYS:
            raise RuntimeError(
                f"{label} PFX_MASK_564 keys drifted: "
                f"{set(mapping)} != {set(self._EXPECTED_MASK_KEYS)}."
            )
        return {
            "label": label,
            "mapping": mapping,
            "config": vlm_model.config,
            "inputs_embeds": inputs_embeds,
            "prefix_attention_mask": prefix_attention_mask,
            "prefix_position_ids": prefix_position_ids,
            "cache_position": cache_position,
            "past_key_values": self._static_cache,
            "token_type_ids": self._static_inputs["token_type_ids"],
            "pixel_values": self._static_inputs["pixel_values"],
            "is_training": False,
            "is_first_iteration": True,
        }

    @staticmethod
    def _mask_tensor_state(value: torch.Tensor) -> dict[str, Any]:
        storage = value.untyped_storage()
        storage_offset_bytes = int(value.storage_offset() * value.element_size())
        result = {
            "data_ptr": int(value.data_ptr()),
            "storage_ptr": int(storage.data_ptr()),
            "storage_offset_bytes": storage_offset_bytes,
            "logical_bytes": int(value.numel() * value.element_size()),
            "backing_storage_bytes": int(storage.nbytes()),
            "contract": _tensor_contract(value),
            "sha256": _tensor_sha256(value),
        }
        if result["data_ptr"] != result["storage_ptr"] + storage_offset_bytes:
            raise RuntimeError("DM05 retained mask data/storage pointer drifted.")
        if not 0 <= storage_offset_bytes < result["backing_storage_bytes"]:
            raise RuntimeError("DM05 retained mask storage offset is invalid.")
        return result

    def _bind_mask_owners(self) -> None:
        assert self._static_cache is not None
        self._retained_mask_owner = self._build_original_mask_owner("retained")
        self._oracle_mask_owner = self._build_original_mask_owner("oracle_rebuild")
        if self.mask_helper_build_count != 2:
            raise RuntimeError(
                "DM05 original mask helper must build exactly twice."
            )

        for name in (
            "config",
            "past_key_values",
            "token_type_ids",
            "pixel_values",
        ):
            if self._retained_mask_owner[name] is not (
                self._oracle_mask_owner[name]
            ):
                raise RuntimeError(
                    f"DM05 retained/oracle mask owner identity drifted for {name}."
                )
        for name in (
            "inputs_embeds",
            "prefix_attention_mask",
            "prefix_position_ids",
            "cache_position",
        ):
            retained_argument = self._retained_mask_owner[name]
            oracle_argument = self._oracle_mask_owner[name]
            if (
                _tensor_contract(retained_argument)
                != _tensor_contract(oracle_argument)
                or not bool(torch.equal(retained_argument, oracle_argument))
            ):
                raise RuntimeError(
                    f"DM05 retained/oracle mask argument drifted for {name}."
                )
        if (
            self._retained_mask_owner["is_training"] is not False
            or self._oracle_mask_owner["is_training"] is not False
            or self._retained_mask_owner["is_first_iteration"] is not True
            or self._oracle_mask_owner["is_first_iteration"] is not True
        ):
            raise RuntimeError("DM05 retained/oracle mask flags drifted.")

        retained = self._retained_mask_owner["mapping"]
        oracle = self._oracle_mask_owner["mapping"]
        self.mask_mapping_keys = sorted(retained)
        forbidden_storage = {
            int(value.untyped_storage().data_ptr())
            for value in tuple(self._static_inputs.values())
            + sum(self._cache_tensors(self._static_cache), ())
        }
        seen_storage: set[int] = set()
        for key in self.mask_mapping_keys:
            retained_value = retained[key]
            oracle_value = oracle[key]
            if (retained_value is None) != (oracle_value is None):
                raise RuntimeError(f"DM05 mask None position drifted for {key}.")
            if retained_value is None:
                self._retained_mask_addresses[key] = None
                self._retained_mask_states[key] = None
                self._oracle_mask_states[key] = None
                continue
            retained_state = self._mask_tensor_state(retained_value)
            oracle_state = self._mask_tensor_state(oracle_value)
            if retained_state["storage_ptr"] in forbidden_storage:
                raise RuntimeError(f"DM05 retained mask {key} aliases mutable state.")
            if retained_state["storage_ptr"] in seen_storage:
                raise RuntimeError("DM05 retained mask tensors alias each other.")
            if retained_state["storage_ptr"] == oracle_state["storage_ptr"]:
                raise RuntimeError(f"DM05 retained/oracle mask {key} aliases.")
            if (
                retained_state["contract"] != oracle_state["contract"]
                or retained_state["sha256"] != oracle_state["sha256"]
                or not bool(torch.equal(retained_value, oracle_value))
            ):
                raise RuntimeError(
                    f"DM05 retained/oracle mask {key} is not bitwise exact."
                )
            torch._dynamo.mark_static_address(retained_value)
            seen_storage.add(retained_state["storage_ptr"])
            self._retained_mask_addresses[key] = retained_state["data_ptr"]
            self._retained_mask_states[key] = retained_state
            self._oracle_mask_states[key] = oracle_state
        self._verify_mask_owners(check_bytes=True)

    def _verify_mask_owners(self, *, check_bytes: bool) -> None:
        self.mask_static_address_verified = False
        if check_bytes:
            self.mask_immutable_verified = False
        if self.mask_helper_build_count != 2:
            raise RuntimeError("DM05 original mask helper unexpectedly rebuilt masks.")
        if self._retained_mask_owner is None or self._oracle_mask_owner is None:
            raise RuntimeError("DM05 mask owners are unavailable.")
        retained = self._retained_mask_owner["mapping"]
        oracle = self._oracle_mask_owner["mapping"]
        if set(retained) != self._EXPECTED_MASK_KEYS or set(oracle) != (
            self._EXPECTED_MASK_KEYS
        ):
            raise RuntimeError("DM05 retained/oracle mask keys drifted.")

        static_address_verified = True
        immutable_verified = True
        for key in self.mask_mapping_keys:
            retained_value = retained[key]
            oracle_value = oracle[key]
            expected_address = self._retained_mask_addresses[key]
            expected_retained = self._retained_mask_states[key]
            expected_oracle = self._oracle_mask_states[key]
            if retained_value is None or oracle_value is None:
                if not (
                    retained_value is None
                    and oracle_value is None
                    and expected_address is None
                    and expected_retained is None
                    and expected_oracle is None
                ):
                    raise RuntimeError(
                        f"DM05 retained/oracle mask None state drifted for {key}."
                    )
                continue
            if int(retained_value.data_ptr()) != expected_address:
                static_address_verified = False
            if check_bytes:
                retained_state = self._mask_tensor_state(retained_value)
                oracle_state = self._mask_tensor_state(oracle_value)
                if (
                    retained_state != expected_retained
                    or oracle_state != expected_oracle
                    or not bool(torch.equal(retained_value, oracle_value))
                ):
                    immutable_verified = False
        self.mask_static_address_verified = static_address_verified
        if check_bytes:
            self.mask_immutable_verified = immutable_verified
        if not static_address_verified:
            raise RuntimeError("DM05 retained mask address drifted.")
        if check_bytes and not immutable_verified:
            raise RuntimeError("DM05 retained/oracle mask bytes drifted.")

    @contextmanager
    def _static_placeholder_specialization(self) -> Iterator[None]:
        """Temporarily replace only Gemma3's capture-illegal count assertion."""
        vlm_model = self.model.model.vlm.model
        had_instance_value = "get_placeholder_mask" in vlm_model.__dict__
        old_instance_value = vlm_model.__dict__.get("get_placeholder_mask")

        def fixed_placeholder_mask(
            model_self: Any,
            input_ids: torch.Tensor | None,
            inputs_embeds: torch.Tensor,
            image_features: torch.Tensor,
        ) -> torch.Tensor:
            if input_ids is None:
                raise RuntimeError("DM05 static placeholder requires input_ids.")
            expected = int(image_features.shape[0] * image_features.shape[1])
            if expected != self._image_feature_count:
                raise RuntimeError(
                    "DM05 captured image-feature cardinality drifted."
                )
            mask = input_ids == int(model_self.config.image_token_id)
            return mask.unsqueeze(-1).expand_as(inputs_embeds).to(
                inputs_embeds.device
            )

        vlm_model.get_placeholder_mask = MethodType(
            fixed_placeholder_mask,
            vlm_model,
        )
        try:
            yield
        finally:
            if had_instance_value:
                vlm_model.get_placeholder_mask = old_instance_value
            else:
                delattr(vlm_model, "get_placeholder_mask")

    def _run_static_prefix(self) -> None:
        assert self._retained_mask_owner is not None
        assert self._static_cache is not None
        prefix_attention_mask = mask_history_pad_tokens_in_attention(
            input_ids=self._static_inputs["input_ids"],
            attention_mask=self._static_inputs["attention_mask"],
        )
        assert prefix_attention_mask is not None
        prefix_position_ids = (
            prefix_attention_mask.cumsum(dim=-1) - 1
        ).clamp_min_(0)
        self.model.model.vlm.model(
            input_ids=self._static_inputs["input_ids"],
            attention_mask=self._retained_mask_owner["mapping"],
            position_ids=prefix_position_ids,
            past_key_values=self._static_cache,
            inputs_embeds=None,
            pixel_values=self._static_inputs["pixel_values"],
            token_type_ids=self._static_inputs["token_type_ids"],
            use_cache=True,
        )

    def _run_static_suffix(self) -> None:
        assert self._static_cache is not None
        assert self._static_action_mask is not None
        assert self._static_noise is not None
        assert self._static_output is not None
        output = _official_euler_suffix(
            self.model,
            input_ids=self._static_inputs["input_ids"],
            prefix_len=self.request_prefix_len,
            past_key_values=self._static_cache,
            initial_noise=self._static_noise,
            diffusion_steps=self.diffusion_steps,
            action_mask=self._static_action_mask,
        )
        self._static_output.copy_(output)

    def _assert_prefix_static_addresses(self) -> None:
        assert self._static_cache is not None
        current_cache = tuple(
            int(value.data_ptr())
            for value in sum(self._cache_tensors(self._static_cache), ())
        )
        self.prefix_static_cache_address_verified = (
            current_cache == self._static_cache_addresses
        )
        if not self.prefix_static_cache_address_verified:
            raise RuntimeError("DM05 prefix static-cache address drifted.")
        self._verify_mask_owners(check_bytes=False)

    def _assert_static_addresses(self) -> None:
        self._assert_prefix_static_addresses()
        assert self._static_output is not None
        self.suffix_static_output_address_verified = (
            int(self._static_output.data_ptr()) == self._static_output_address
        )
        if not self.suffix_static_output_address_verified:
            raise RuntimeError("DM05 suffix static-output address drifted.")

    def _assert_prefix_exact(self, reference_cache: Cache) -> None:
        assert self._static_cache is not None
        reference_keys, reference_values = self._cache_tensors(reference_cache)
        static_keys, static_values = self._cache_tensors(self._static_cache)
        for kind, references, candidates in (
            ("K", reference_keys, static_keys),
            ("V", reference_values, static_values),
        ):
            for index, (reference, candidate) in enumerate(
                zip(references, candidates, strict=True)
            ):
                reference_layer = reference_cache.layers[index]
                candidate_layer = self._static_cache.layers[index]
                reference_cumulative = getattr(
                    reference_layer,
                    "cumulative_length",
                    None,
                )
                candidate_cumulative = getattr(
                    candidate_layer,
                    "cumulative_length",
                    None,
                )
                semantics_equal = (
                    bool(reference_layer.is_sliding)
                    == bool(candidate_layer.is_sliding)
                    and int(reference_layer.get_seq_length())
                    == int(candidate_layer.get_seq_length())
                    and (
                        reference_cumulative is None
                        or int(reference_cumulative) == int(candidate_cumulative)
                    )
                )
                if (
                    not semantics_equal
                    or _tensor_contract(reference) != _tensor_contract(candidate)
                    or not bool(torch.equal(reference, candidate))
                ):
                    raise RuntimeError(
                        f"DM05 PFX_KV_564 {kind} layer {index} is not exact."
                    )

    def _initialize_suffix_graph(self) -> None:
        """Warm, capture, and validate the native exact suffix graph.

        The combined fixed-cell runtime overrides this single boundary. The
        accepted prefix owner, request staging, replay order, and public
        counters remain canonical in this class.
        """
        with torch.inference_mode():
            for _ in range(self._WARMUP_COUNT):
                self._run_static_suffix()
        torch.cuda.synchronize(self.device)
        suffix_graph = torch.cuda.CUDAGraph()
        with torch.inference_mode(), torch.cuda.graph(suffix_graph):
            self._run_static_suffix()
        self._suffix_graph = suffix_graph
        self.suffix_startup_capture_count = 1
        self.suffix_capture_execution_count = 1
        suffix_graph.replay()
        torch.cuda.synchronize(self.device)

        self._assert_static_addresses()
        self._verify_mask_owners(check_bytes=True)
        assert self._static_output is not None
        if not bool(torch.isfinite(self._static_output).all().item()):
            raise RuntimeError(
                "DM05 static-mask suffix capture produced non-finite output."
            )

    def _initialize(
        self,
        *,
        request_inputs: dict[str, torch.Tensor],
        action_mask: torch.Tensor,
        mask_layout_key: str,
    ) -> None:
        started = time.perf_counter()
        try:
            history_pad_count = int(
                (
                    request_inputs["input_ids"] == HISTORY_PAD_TOKEN_ID
                ).sum().item()
            )
            if history_pad_count != 0:
                raise RuntimeError(
                    "DM05 static-mask prefix graph does not support history-pad "
                    f"tokens, got {history_pad_count}."
                )

            with torch.inference_mode():
                reference_cache, reference_prefix_len = (
                    self.model._compute_prefix_cache(
                        input_ids=request_inputs["input_ids"],
                        attention_mask=request_inputs["attention_mask"],
                        pixel_values=request_inputs["pixel_values"],
                        token_type_ids=request_inputs["token_type_ids"],
                        history_pixel_values=None,
                        history_mask=None,
                        cache_cls=DynamicCache,
                    )
                )
            if int(reference_prefix_len) != self.request_prefix_len:
                raise RuntimeError(
                    "DM05 eager reference prefix length drifted: "
                    f"{reference_prefix_len}."
                )

            vlm_model = self.model.model.vlm.model
            image_token_id = int(vlm_model.config.image_token_id)
            image_token_count = int(
                (
                    request_inputs["input_ids"] == image_token_id
                ).sum().item()
            )
            with torch.inference_mode():
                image_features = vlm_model.get_image_features(
                    request_inputs["pixel_values"],
                    return_dict=True,
                ).pooler_output
            self._image_feature_count = int(
                image_features.shape[0] * image_features.shape[1]
            )
            if image_token_count != self._image_feature_count:
                raise RuntimeError(
                    "DM05 image-token/image-feature cardinality drifted: "
                    f"{image_token_count} != {self._image_feature_count}."
                )
            del image_features

            self._static_cache = self._make_static_cache(reference_cache)
            self._cache_layer_semantics_state = self._cache_layer_semantics()
            self._static_inputs = {
                name: self._static_tensor_like(value)
                for name, value in request_inputs.items()
            }
            self._input_contracts = {
                name: _tensor_contract(value)
                for name, value in self._static_inputs.items()
            }
            static_keys, static_values = self._cache_tensors(self._static_cache)
            self._static_cache_addresses = tuple(
                int(value.data_ptr()) for value in static_keys + static_values
            )

            self._bind_mask_owners()
            self._verify_mask_owners(check_bytes=True)

            with torch.inference_mode():
                for _ in range(self._WARMUP_COUNT):
                    self._reset_static_cache()
                    self._run_static_prefix()
            torch.cuda.synchronize(self.device)
            self._reset_static_cache()
            self._capture_cache_write_observation = 0
            prefix_graph = torch.cuda.CUDAGraph()
            with (
                torch.inference_mode(),
                self._static_placeholder_specialization(),
                torch.cuda.graph(prefix_graph),
            ):
                self._run_static_prefix()
            self._captured_prefix_cache_writes = int(
                self._capture_cache_write_observation
            )
            if self._captured_prefix_cache_writes != 2 * self.cache_layer_count:
                raise RuntimeError(
                    "DM05 prefix capture cache-write count drifted: "
                    f"{self._captured_prefix_cache_writes}."
                )
            self._prefix_graph = prefix_graph
            self.prefix_startup_capture_count = 1
            self.prefix_capture_execution_count = 1
            prefix_graph.replay()
            torch.cuda.synchronize(self.device)
            self._assert_prefix_static_addresses()
            self._assert_prefix_exact(reference_cache)
            self._verify_mask_owners(check_bytes=True)

            rng_before = torch.cuda.get_rng_state(self.device).clone()
            prefix_graph.replay()
            torch.cuda.synchronize(self.device)
            rng_after = torch.cuda.get_rng_state(self.device)
            if not bool(torch.equal(rng_before, rng_after)):
                raise RuntimeError("DM05 captured prefix changed CUDA RNG state.")
            self._verify_mask_owners(check_bytes=True)

            self._static_action_mask = self._static_tensor_like(action_mask)
            self._static_noise = torch.zeros(
                self.batch_size,
                self.chunk_size,
                self.action_dim,
                device=self.device,
                dtype=self.noise_dtype,
            )
            self._static_output = torch.empty_like(self._static_noise)
            for value in (self._static_noise, self._static_output):
                torch._dynamo.mark_static_address(value)
            self._static_output_address = int(self._static_output.data_ptr())

            self._initialize_suffix_graph()

            self._mask_layout_key = mask_layout_key
            self.mask_layout_key_verified = True
            self.initialization_ms = (time.perf_counter() - started) * 1000.0
            self.initialized = True
        except Exception:
            self._initialization_failed = True
            self._prefix_graph = None
            self._suffix_graph = None
            raise

    def _stage_prefix_inputs(
        self,
        request_inputs: dict[str, torch.Tensor],
    ) -> None:
        for name in (
            "input_ids",
            "attention_mask",
            "pixel_values",
            "token_type_ids",
        ):
            self._static_inputs[name].copy_(request_inputs[name])

    @torch.inference_mode()
    def inference_action(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        token_type_ids: torch.LongTensor | None = None,
        states: torch.FloatTensor | None = None,
        image_masks: torch.BoolTensor | None = None,
        diffusion_steps: int = _DIFFUSION_STEPS,
        past_key_values: Cache | None = None,
        action_mask: torch.Tensor | None = None,
        history_pixel_values: torch.Tensor | None = None,
        history_mask: torch.BoolTensor | None = None,
        mask_layout_key: str | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Run one exact request; initialize lazily on the first eligible one."""
        with self._lock:
            self._validate_request(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                token_type_ids=token_type_ids,
                diffusion_steps=diffusion_steps,
                past_key_values=past_key_values,
                action_mask=action_mask,
                history_pixel_values=history_pixel_values,
                history_mask=history_mask,
                states=states,
                image_masks=image_masks,
                mask_layout_key=mask_layout_key,
                unsupported=kwargs,
            )
            assert input_ids is not None
            assert attention_mask is not None
            assert pixel_values is not None
            assert token_type_ids is not None
            assert action_mask is not None
            assert mask_layout_key is not None
            request_inputs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "pixel_values": pixel_values,
                "token_type_ids": token_type_ids,
            }

            if not self.initialized:
                self._initialize(
                    request_inputs=request_inputs,
                    action_mask=action_mask,
                    mask_layout_key=mask_layout_key,
                )
            elif mask_layout_key != self._mask_layout_key:
                self.mask_layout_key_verified = False
                raise RuntimeError(
                    "DM05 static-mask prefix graph mask_layout_key drifted."
                )

            if self._prefix_graph is None or self._suffix_graph is None:
                raise RuntimeError(
                    "DM05StaticMaskPrefixGraphRuntime graphs are unavailable."
                )
            assert self._static_action_mask is not None
            assert self._static_noise is not None
            assert self._static_output is not None

            self._stage_prefix_inputs(request_inputs)
            self._prefix_graph.replay()
            noise = torch.randn(
                self.batch_size,
                self.chunk_size,
                self.action_dim,
                device=self.device,
                dtype=self.noise_dtype,
            )
            self._static_action_mask.copy_(action_mask)
            self._static_noise.copy_(noise)
            self._suffix_graph.replay()
            self._assert_static_addresses()

            self.prefix_input_stage_requests += 1
            self.prefix_input_tensor_copies += 4
            self.prefix_graph_replay_count += 1
            self.prefix_graph_cache_write_tensor_copies += (
                self._captured_prefix_cache_writes
            )
            self.eager_noise_count += 1
            self.suffix_input_stage_requests += 1
            self.suffix_input_tensor_copies += 2
            self.suffix_graph_replay_count += 1
            self.request_prefix_length = self.request_prefix_len
            self.selected_prefix_length = self.request_prefix_len
            self.mask_layout_key_verified = True
            return self._static_output

    def proof_snapshot(self) -> dict[str, Any]:
        """Return public proof without layout-key digests or device pointers."""
        with self._lock:
            return {
                "schema": _SCHEMA,
                "execution_backend": _EXECUTION_BACKEND,
                "arithmetic_backend": _ARITHMETIC_BACKEND,
                "graph_scope": _GRAPH_SCOPE,
                "profile_prefix_lengths": [self.request_prefix_len],
                "initialized": self.initialized,
                "initialization_ms": self.initialization_ms,
                "mask_owner_symbol": _MASK_OWNER_SYMBOL,
                "mask_modeling_source_sha256": (
                    self.mask_modeling_source_sha256
                ),
                "mask_utils_source_sha256": self.mask_utils_source_sha256,
                "dm05_arch_source_sha256": self.dm05_arch_source_sha256,
                "mask_layout_key_verified": self.mask_layout_key_verified,
                "mask_helper_build_count": self.mask_helper_build_count,
                "mask_mapping_keys": list(self.mask_mapping_keys),
                "mask_static_address_verified": (
                    self.mask_static_address_verified
                ),
                "mask_immutable_verified": self.mask_immutable_verified,
                "prefix_startup_capture_count": (
                    self.prefix_startup_capture_count
                ),
                "suffix_startup_capture_count": (
                    self.suffix_startup_capture_count
                ),
                "prefix_capture_execution_count": (
                    self.prefix_capture_execution_count
                ),
                "suffix_capture_execution_count": (
                    self.suffix_capture_execution_count
                ),
                "prefix_input_stage_requests": self.prefix_input_stage_requests,
                "prefix_input_tensor_copies": self.prefix_input_tensor_copies,
                "prefix_graph_replay_count": self.prefix_graph_replay_count,
                "prefix_graph_cache_write_tensor_copies": (
                    self.prefix_graph_cache_write_tensor_copies
                ),
                "eager_noise_count": self.eager_noise_count,
                "suffix_input_stage_requests": self.suffix_input_stage_requests,
                "suffix_input_tensor_copies": self.suffix_input_tensor_copies,
                "suffix_graph_replay_count": self.suffix_graph_replay_count,
                "prefix_eager_count": self.prefix_eager_count,
                "post_prefix_cache_stage_requests": (
                    self.post_prefix_cache_stage_requests
                ),
                "post_prefix_cache_tensor_copies": (
                    self.post_prefix_cache_tensor_copies
                ),
                "fallback_count": self.fallback_count,
                "history_count": self.history_count,
                "result_reuse_count": self.result_reuse_count,
                "request_prefix_length": self.request_prefix_length,
                "selected_prefix_length": self.selected_prefix_length,
                "cache_layer_count": self.cache_layer_count,
                "prefix_static_cache_address_verified": (
                    self.prefix_static_cache_address_verified
                ),
                "suffix_static_output_address_verified": (
                    self.suffix_static_output_address_verified
                ),
                "closed": self._closed,
            }

    def close(self) -> None:
        """Release captured owners and graphs; safe before init and idempotent."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._prefix_graph = None
            self._suffix_graph = None
            self._static_cache = None
            self._static_inputs = {}
            self._static_action_mask = None
            self._static_noise = None
            self._static_output = None
            self._retained_mask_owner = None
            self._oracle_mask_owner = None
            self._mask_layout_key = None
            self._input_contracts = {}
            self._static_cache_addresses = ()
            self._retained_mask_addresses = {}
            self._retained_mask_states = {}
            self._oracle_mask_states = {}
            self._cache_layer_semantics_state = []
