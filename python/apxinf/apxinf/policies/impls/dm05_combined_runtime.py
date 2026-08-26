"""Exact combined CUDA Graph runtime for the frozen DM05-LIBERO RTX 4090 cell.

This default-off executor composes the accepted static-mask prefix graph with
four exact suffix mechanisms:

* one 10 x 69 table of the original AdaRMS modulation linears;
* one per-replay suffix mask, position, boolean mask, and RoPE owner;
* two contiguous expanded K/V pack workspaces consumed without ``repeat_kv``;
* one PTX-gated Triton post-reduction affine kernel which retains native
  mean-square, epsilon, and rsqrt operations.

The capture patch exists only while warming/capturing the suffix graph and is
always restored.  Unsupported software, hardware, topology, shapes, codegen,
or capture behavior raises; the combined selector never falls back. SASS
qualification remains an external promotion receipt, not a runtime dependency.
"""

from __future__ import annotations

import hashlib
import inspect
import re
import threading
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import MethodType
from typing import Any, Iterator, Mapping, Sequence

import torch

import opendm.model.dm05.dm05_arch as dm05_arch
from apxinf.policies.impls.dm05_static_mask_prefix_graph import (
    DM05StaticMaskPrefixGraphRuntime,
)

__all__ = ["DM05CombinedRuntime"]


_SCHEMA = "apxinf.dm05.exact-combined.v1"
_EXECUTION_BACKEND = "default_exact_combined"
_ARITHMETIC_BACKEND = "native_sdpa_plus_exact_postreduce_triton"
_GRAPH_SCOPE = "native_prefix_564_plus_combined_suffix_10step"
_REQUIRED_TORCH_VERSION = "2.11.0+cu130"
_REQUIRED_CUDA_VERSION = "13.0"
_REQUIRED_TRITON_VERSION = "3.6.0"
_DM05_ARCH_SHA256 = (
    "b5ab170374fbc965aa86d7d370075e8c8bc21bcf46bc6de34e7e336df1af9ce8"
)

_BATCH_SIZE = 1
_PREFIX_LENGTH = 564
_SUFFIX_LENGTH = 10
_KEY_LENGTH = _PREFIX_LENGTH + _SUFFIX_LENGTH
_HIDDEN_SIZE = 1024
_MODEL_ACTION_DIM = 32
_LAYER_COUNT = 34
_SLIDING_LAYERS = 29
_FULL_LAYERS = 5
_QUERY_HEADS = 8
_KV_HEADS = 4
_KV_REPETITIONS = 2
_HEAD_DIM = 256
_MODULATION_WIDTH = 3 * _HIDDEN_SIZE
_SITES_PER_STEP = 2 * _LAYER_COUNT + 1
_ADAPTIVE_CALLS = _SUFFIX_LENGTH * _SITES_PER_STEP
_ATTENTION_CALLS = _SUFFIX_LENGTH * _LAYER_COUNT
_CAT_CALLS = 2 * _ATTENTION_CALLS
_REPEAT_CALLS = 2 * _ATTENTION_CALLS
_BLOCK_SIZE = 256
_NUM_WARPS = 8
_ELEMENTS_PER_AFFINE = _BATCH_SIZE * _SUFFIX_LENGTH * _HIDDEN_SIZE
_KERNEL_NAME = "_dm05_exact_postreduce_affine_kernel"

_PATCH_LOCK = threading.Lock()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _shape(value: Any) -> tuple[int, ...]:
    return tuple(int(item) for item in value.shape)


def _tensor_sha256(value: Any) -> str:
    payload = value.detach().contiguous().view(torch.uint8).cpu().numpy()
    return hashlib.sha256(payload.tobytes()).hexdigest()


def _sequence_sha256(values: Sequence[Any]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(bytes.fromhex(_tensor_sha256(value)))
    return digest.hexdigest()


def _data_addresses(values: Sequence[Any]) -> tuple[int, ...]:
    return tuple(int(value.data_ptr()) for value in values)


def _storage_addresses(values: Sequence[Any]) -> tuple[int, ...]:
    return tuple(int(value.untyped_storage().data_ptr()) for value in values)


def _require_tensor(
    value: Any,
    expected_shape: tuple[int, ...],
    *,
    dtype: Any,
    device: Any,
    contiguous: bool | None,
    label: str,
) -> None:
    if _shape(value) != expected_shape:
        raise RuntimeError(
            f"DM05 combined {label} shape drifted: "
            f"{_shape(value)} != {expected_shape}."
        )
    if value.dtype != dtype or value.device != device:
        raise RuntimeError(
            f"DM05 combined {label} dtype/device drifted: "
            f"{value.dtype}/{value.device}."
        )
    if contiguous is not None and bool(value.is_contiguous()) is not contiguous:
        raise RuntimeError(f"DM05 combined {label} contiguity drifted.")


def _native_time_values() -> tuple[float, ...]:
    dt = -1.0 / _SUFFIX_LENGTH
    current = 1.0
    values = []
    for _ in range(_SUFFIX_LENGTH):
        values.append(current)
        current += dt
    return tuple(values)


def _build_triton_kernel() -> tuple[Any, Any]:
    try:
        import triton
        import triton.language as tl
    except ImportError as exc:
        raise RuntimeError(
            "DM05 default_exact_combined requires triton==3.6.0; "
            "there is no fallback."
        ) from exc

    @triton.jit
    def _dm05_exact_postreduce_affine_kernel(
        x_ptr,
        inv_ptr,
        scale_ptr,
        shift_ptr,
        output_ptr,
        N_ELEMENTS: tl.constexpr,
        HIDDEN: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        tl.static_assert(BLOCK == 256)
        offsets = tl.program_id(axis=0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < N_ELEMENTS
        hidden_offsets = offsets % HIDDEN
        row_offsets = offsets // HIDDEN
        x_f32 = tl.load(x_ptr + offsets, mask=mask).to(tl.float32)
        inv_f32 = tl.load(inv_ptr + row_offsets, mask=mask).to(tl.float32)
        scale_f32 = tl.load(scale_ptr + hidden_offsets, mask=mask).to(tl.float32)
        shift_f32 = tl.load(shift_ptr + hidden_offsets, mask=mask).to(tl.float32)
        normalized = x_f32 * inv_f32
        scaled = scale_f32 + 1.0
        product = normalized * scaled
        output = product + shift_f32
        tl.store(output_ptr + offsets, output.to(tl.bfloat16), mask=mask)

    return triton, _dm05_exact_postreduce_affine_kernel


def _validate_ptx_codegen(
    ptx: str,
    *,
    cubin: Any,
    compiled_name: str,
    device_capability: tuple[int, int],
) -> dict[str, Any]:
    """Validate PTX structure and bind PTX/cubin identity.

    This is intentionally not a SASS claim. Formal cubin/SASS inspection stays
    an external promotion receipt and is never a shipping runtime dependency.
    """
    if not isinstance(ptx, str) or not ptx.strip():
        raise RuntimeError("DM05 combined Triton PTX is unavailable.")
    if not isinstance(cubin, (bytes, bytearray, memoryview)) or not cubin:
        raise RuntimeError("DM05 combined Triton cubin is unavailable.")
    cubin_bytes = bytes(cubin)
    forbidden_patterns = {
        "fma_f32": r"\bfma(?:(?:\.(?:rn|rz|rm|rp|ftz))*)\.f32\b",
        "mad_f32": r"\bmad(?:(?:\.(?:rn|rz|rm|rp|ftz))*)\.f32\b",
        "packed_f32x2": r"\bf32x2\b",
        "local_declaration": r"\.local\b",
        "local_load": r"\bld\.local\b",
        "local_store": r"\bst\.local\b",
    }
    forbidden_hits = {
        name: len(re.findall(pattern, ptx, flags=re.IGNORECASE))
        for name, pattern in forbidden_patterns.items()
    }
    mul_count = len(re.findall(r"\bmul\.rn\.f32\b", ptx, re.IGNORECASE))
    add_count = len(re.findall(r"\badd\.rn\.f32\b", ptx, re.IGNORECASE))
    bf16_count = len(
        re.findall(r"\bcvt\.rn\.bf16\.f32\b", ptx, re.IGNORECASE)
    )
    checks = {
        "device_sm89": tuple(device_capability) == (8, 9),
        "ptx_target_sm89": bool(re.search(r"\.target\s+sm_89\b", ptx)),
        "compiled_entry_selected": _KERNEL_NAME in compiled_name,
        "ptx_entry_selected": _KERNEL_NAME in ptx
        and bool(re.search(r"\.entry\s+[^\n]*" + _KERNEL_NAME, ptx)),
        "two_scalar_fmul_rn_f32": mul_count >= 2,
        "two_scalar_fadd_rn_f32": add_count >= 2,
        "bf16_rne_store": bf16_count >= 1,
        "forbidden_codegen_absent": not any(forbidden_hits.values()),
    }
    return {
        "level": "compiled_ptx",
        "ptx_sha256": hashlib.sha256(ptx.encode()).hexdigest(),
        "cubin_sha256": hashlib.sha256(cubin_bytes).hexdigest(),
        "scalar_fmul_rn_f32_count": mul_count,
        "scalar_fadd_rn_f32_count": add_count,
        "bf16_rne_conversion_count": bf16_count,
        "forbidden_hits": forbidden_hits,
        "checks": checks,
        "ptx_codegen_verified": all(checks.values()),
        "sass_external_receipt_required": True,
    }


class _ExactPostReductionAffine:
    """Strict SM89 launcher.  Compile or codegen failure is terminal."""

    def __init__(self, device: Any) -> None:
        if tuple(torch.cuda.get_device_capability(device)) != (8, 9):
            raise RuntimeError(
                "DM05 default_exact_combined requires an SM89 RTX 4090; "
                "there is no fallback."
            )
        self.triton, self.kernel = _build_triton_kernel()
        if str(self.triton.__version__) != _REQUIRED_TRITON_VERSION:
            raise RuntimeError(
                "DM05 default_exact_combined requires triton==3.6.0, got "
                f"{self.triton.__version__}; there is no fallback."
            )
        self.device = device
        self.compiled: Any | None = None
        self.compile_count = 0
        self.fallback_count = 0
        self.codegen: dict[str, Any] | None = None

    @property
    def grid(self) -> tuple[int]:
        return (self.triton.cdiv(_ELEMENTS_PER_AFFINE, _BLOCK_SIZE),)

    def _validate_inputs(
        self,
        x: Any,
        inv: Any,
        scale: Any,
        shift: Any,
        output: Any,
    ) -> None:
        expected = (
            (x, (_BATCH_SIZE, _SUFFIX_LENGTH, _HIDDEN_SIZE), torch.bfloat16, "x"),
            (inv, (_BATCH_SIZE, _SUFFIX_LENGTH, 1), torch.float32, "inv"),
            (scale, (_BATCH_SIZE, 1, _HIDDEN_SIZE), torch.bfloat16, "scale"),
            (shift, (_BATCH_SIZE, 1, _HIDDEN_SIZE), torch.bfloat16, "shift"),
            (
                output,
                (_BATCH_SIZE, _SUFFIX_LENGTH, _HIDDEN_SIZE),
                torch.bfloat16,
                "output",
            ),
        )
        for value, shape, dtype, label in expected:
            _require_tensor(
                value,
                shape,
                dtype=dtype,
                device=self.device,
                contiguous=True if label in {"x", "inv", "output"} else None,
                label=f"affine {label}",
            )
            if int(value.stride(-1)) != 1:
                raise RuntimeError(
                    f"DM05 combined affine {label} last stride drifted."
                )

    def _compile_and_validate(
        self,
        x: Any,
        inv: Any,
        scale: Any,
        shift: Any,
        output: Any,
    ) -> None:
        kwargs = {
            "N_ELEMENTS": _ELEMENTS_PER_AFFINE,
            "HIDDEN": _HIDDEN_SIZE,
            "BLOCK": _BLOCK_SIZE,
            "num_warps": _NUM_WARPS,
            "enable_fp_fusion": False,
        }
        try:
            self.compiled = self.kernel.warmup(
                x,
                inv,
                scale,
                shift,
                output,
                grid=self.grid,
                **kwargs,
            )
        except Exception as exc:
            raise RuntimeError(
                "DM05 combined exact-affine compile failed; there is no fallback: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        self.compile_count += 1
        asm = getattr(self.compiled, "asm", None)
        if not isinstance(asm, Mapping):
            raise RuntimeError("DM05 combined compiled asm mapping is unavailable.")
        compiled_name = str(getattr(self.compiled, "name", ""))
        capability = tuple(torch.cuda.get_device_capability(self.device))
        self.codegen = _validate_ptx_codegen(
            asm.get("ptx"),
            cubin=asm.get("cubin"),
            compiled_name=compiled_name,
            device_capability=capability,
        )
        if not self.codegen["ptx_codegen_verified"]:
            raise RuntimeError(
                "DM05 combined exact-affine PTX codegen guard failed; "
                "there is no fallback."
            )

    def launch(
        self,
        x: Any,
        inv: Any,
        scale: Any,
        shift: Any,
        output: Any,
    ) -> Any:
        self._validate_inputs(x, inv, scale, shift, output)
        if self.compiled is None:
            self._compile_and_validate(x, inv, scale, shift, output)
        if (
            self.compile_count != 1
            or not self.codegen
            or not self.codegen["ptx_codegen_verified"]
        ):
            raise RuntimeError("DM05 combined exact-affine PTX is not verified.")
        self.kernel[self.grid](
            x,
            inv,
            scale,
            shift,
            output,
            N_ELEMENTS=_ELEMENTS_PER_AFFINE,
            HIDDEN=_HIDDEN_SIZE,
            BLOCK=_BLOCK_SIZE,
            num_warps=_NUM_WARPS,
            enable_fp_fusion=False,
        )
        return output


@dataclass(frozen=True)
class _ModulationSite:
    index: int
    kind: str
    module: Any


@dataclass
class _ModulationTable:
    entries_by_site: tuple[tuple[Any, ...], ...]
    initial_addresses: tuple[int, ...]
    initial_sha256: str
    rng_unchanged: bool

    def tensors(self) -> tuple[Any, ...]:
        return tuple(
            self.entries_by_site[site][step]
            for step in range(_SUFFIX_LENGTH)
            for site in range(_SITES_PER_STEP)
        )


@dataclass
class _MetadataOwner:
    numeric_mask: Any | None = None
    position_ids: Any | None = None
    bool_mask: Any | None = None
    rope: dict[str, tuple[Any, Any]] = field(default_factory=dict)

    def tensors(self) -> tuple[Any, ...]:
        values = [self.numeric_mask, self.position_ids, self.bool_mask]
        for layer_type in ("sliding_attention", "full_attention"):
            pair = self.rope.get(layer_type)
            if pair is not None:
                values.extend(pair)
        return tuple(value for value in values if value is not None)


@dataclass
class _PackWorkspaces:
    key: Any
    value: Any

    def for_kind(self, kind: str) -> Any:
        return self.key if kind == "K" else self.value

    def tensors(self) -> tuple[Any, Any]:
        return self.key, self.value


@dataclass
class _CaptureCensus:
    suffix_mask_calls: int = 0
    suffix_mask_builds: int = 0
    suffix_position_calls: int = 0
    suffix_position_builds: int = 0
    bool_mask_calls: int = 0
    bool_mask_builds: int = 0
    rope_calls: int = 0
    sliding_rope_builds: int = 0
    full_rope_builds: int = 0
    apply_rotary_calls: int = 0
    semantic_cat_calls: int = 0
    direct_pack_out_builds: int = 0
    semantic_repeat_calls: int = 0
    identity_repeat_returns: int = 0
    sdpa_calls: int = 0
    post_sdpa_contiguous_builds: int = 0
    time_cond_calls: int = 0
    adaptive_rmsnorm_calls: int = 0
    table_lookup_calls: int = 0
    native_modulator_linear_calls: int = 0
    native_var_calls: int = 0
    native_rsqrt_calls: int = 0
    exact_affine_kernel_calls: int = 0
    mlp_calls: int = 0
    fallback_count: int = 0

    def snapshot(self) -> dict[str, int]:
        return {name: int(value) for name, value in asdict(self).items()}


def _expected_capture_census() -> dict[str, int]:
    return _CaptureCensus(
        suffix_mask_calls=_SUFFIX_LENGTH,
        suffix_mask_builds=1,
        suffix_position_calls=_SUFFIX_LENGTH,
        suffix_position_builds=1,
        bool_mask_calls=_ATTENTION_CALLS,
        bool_mask_builds=1,
        rope_calls=_ATTENTION_CALLS,
        sliding_rope_builds=1,
        full_rope_builds=1,
        apply_rotary_calls=_ATTENTION_CALLS,
        semantic_cat_calls=_CAT_CALLS,
        direct_pack_out_builds=_CAT_CALLS,
        semantic_repeat_calls=_REPEAT_CALLS,
        identity_repeat_returns=_REPEAT_CALLS,
        sdpa_calls=_ATTENTION_CALLS,
        post_sdpa_contiguous_builds=_ATTENTION_CALLS,
        time_cond_calls=_SUFFIX_LENGTH,
        adaptive_rmsnorm_calls=_ADAPTIVE_CALLS,
        table_lookup_calls=_ADAPTIVE_CALLS,
        native_modulator_linear_calls=0,
        native_var_calls=_ADAPTIVE_CALLS,
        native_rsqrt_calls=_ADAPTIVE_CALLS,
        exact_affine_kernel_calls=_ADAPTIVE_CALLS,
        mlp_calls=_ATTENTION_CALLS,
        fallback_count=0,
    ).snapshot()


class _TorchPackProxy:
    """Forward torch except the exact two K/V cat sites per suffix layer."""

    def __init__(
        self,
        torch_module: Any,
        *,
        census: _CaptureCensus,
        prefix_keys: Sequence[Any],
        prefix_values: Sequence[Any],
        workspaces: _PackWorkspaces,
    ) -> None:
        self._torch = torch_module
        self.census = census
        self.prefix_keys = tuple(prefix_keys)
        self.prefix_values = tuple(prefix_values)
        self.workspaces = workspaces

    def __getattr__(self, name: str) -> Any:
        return getattr(self._torch, name)

    def cat(self, tensors: Sequence[Any], dim: int = 0, *, out: Any = None) -> Any:
        index = self.census.semantic_cat_calls
        if not 0 <= index < _CAT_CALLS:
            raise RuntimeError("DM05 combined received an unexpected torch.cat site.")
        attention_index, kind_index = divmod(index, 2)
        _, layer = divmod(attention_index, _LAYER_COUNT)
        kind = "K" if kind_index == 0 else "V"
        if out is not None or dim != 2 or len(tensors) != 2:
            raise RuntimeError("DM05 combined K/V cat signature drifted.")
        prefix, suffix = tensors
        expected_prefix = (
            self.prefix_keys[layer] if kind == "K" else self.prefix_values[layer]
        )
        if prefix is not expected_prefix:
            raise RuntimeError(
                f"DM05 combined {kind} prefix owner/order drifted at layer {layer}."
            )
        _require_tensor(
            prefix,
            (_BATCH_SIZE, _KV_HEADS, _PREFIX_LENGTH, _HEAD_DIM),
            dtype=prefix.dtype,
            device=prefix.device,
            contiguous=None,
            label=f"{kind} prefix",
        )
        _require_tensor(
            suffix,
            (_BATCH_SIZE, _KV_HEADS, _SUFFIX_LENGTH, _HEAD_DIM),
            dtype=prefix.dtype,
            device=prefix.device,
            contiguous=None,
            label=f"{kind} suffix",
        )
        workspace = self.workspaces.for_kind(kind)
        destination = workspace.view(
            _BATCH_SIZE,
            _KV_HEADS,
            _KV_REPETITIONS,
            _KEY_LENGTH,
            _HEAD_DIM,
        )
        prefix5 = prefix[:, :, None].expand(-1, -1, _KV_REPETITIONS, -1, -1)
        suffix5 = suffix[:, :, None].expand(-1, -1, _KV_REPETITIONS, -1, -1)
        self._torch.cat((prefix5, suffix5), dim=3, out=destination)
        self.census.semantic_cat_calls += 1
        self.census.direct_pack_out_builds += 1
        return workspace


@dataclass
class _CaptureState:
    census: _CaptureCensus
    metadata: _MetadataOwner
    modulation_index: int = 0
    adaptive_index: int = 0
    repeat_index: int = 0


def _instance_binding(instance: Any, name: str) -> tuple[bool, Any]:
    return name in instance.__dict__, instance.__dict__.get(name)


def _restore_instance_binding(
    instance: Any,
    name: str,
    state: tuple[bool, Any],
) -> None:
    if state[0]:
        setattr(instance, name, state[1])
    elif name in instance.__dict__:
        delattr(instance, name)


@contextmanager
def _combined_capture_patch_unlocked(
    runtime: "DM05CombinedRuntime",
    state: _CaptureState,
) -> Iterator[None]:
    """Install the sole combined capture specialization and always restore it."""
    model = runtime.model
    dm05 = model.model
    expert = dm05.action_expert
    rotary = expert.rotary_emb
    arch = dm05_arch
    sites = runtime._modulation_sites
    site_by_module = {id(site.module): site for site in sites}
    prefix_keys, prefix_values = runtime._cache_tensors(runtime._static_cache)

    original_torch = arch.torch
    original_mask = arch.make_suffix_attn_mask
    original_apply = arch.apply_rotary_pos_emb
    original_repeat = arch.repeat_kv
    original_functional = arch.F
    original_position = model._build_suffix_position_ids
    original_rotary = rotary.forward
    original_attention = expert._attn_fn
    original_time = model._build_adarms_cond
    original_adaptive = expert._adaptive_rmsnorm
    original_mlp = [layer.mlp.forward for layer in expert.layers]
    original_modulators = [site.module.forward for site in sites]

    position_state = _instance_binding(model, "_build_suffix_position_ids")
    rotary_state = _instance_binding(rotary, "forward")
    attention_state = _instance_binding(expert, "_attn_fn")
    time_state = _instance_binding(model, "_build_adarms_cond")
    adaptive_state = _instance_binding(expert, "_adaptive_rmsnorm")
    mlp_states = [_instance_binding(layer.mlp, "forward") for layer in expert.layers]
    modulator_states = [
        _instance_binding(site.module, "forward") for site in sites
    ]

    proxy = _TorchPackProxy(
        torch,
        census=state.census,
        prefix_keys=prefix_keys,
        prefix_values=prefix_values,
        workspaces=runtime._pack_workspaces,
    )

    def mask_wrapper(*args: Any, **kwargs: Any) -> Any:
        state.census.suffix_mask_calls += 1
        if state.metadata.numeric_mask is None:
            value = original_mask(*args, **kwargs)
            _require_tensor(
                value,
                (_BATCH_SIZE, 1, _SUFFIX_LENGTH, _KEY_LENGTH),
                dtype=runtime.noise_dtype,
                device=runtime.device,
                contiguous=True,
                label="suffix numeric mask",
            )
            state.metadata.numeric_mask = value
            state.census.suffix_mask_builds += 1
        return state.metadata.numeric_mask

    def position_wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        del self
        state.census.suffix_position_calls += 1
        if state.metadata.position_ids is None:
            value = original_position(*args, **kwargs)
            _require_tensor(
                value,
                (_BATCH_SIZE, _SUFFIX_LENGTH),
                dtype=torch.long,
                device=runtime.device,
                contiguous=True,
                label="suffix position ids",
            )
            state.metadata.position_ids = value
            state.census.suffix_position_builds += 1
        return state.metadata.position_ids

    def rotary_wrapper(
        self: Any,
        x: Any,
        position_ids: Any,
        layer_type: Any = None,
    ) -> tuple[Any, Any]:
        del self
        layer_name = str(layer_type)
        if layer_name not in {"sliding_attention", "full_attention"}:
            raise RuntimeError(f"DM05 combined RoPE type drifted: {layer_name}.")
        state.census.rope_calls += 1
        if layer_name not in state.metadata.rope:
            cos, sin = original_rotary(x, position_ids, layer_type)
            for label, value in (("cos", cos), ("sin", sin)):
                _require_tensor(
                    value,
                    (_BATCH_SIZE, _SUFFIX_LENGTH, _HEAD_DIM),
                    dtype=runtime.noise_dtype,
                    device=runtime.device,
                    contiguous=True,
                    label=f"{layer_name} RoPE {label}",
                )
            state.metadata.rope[layer_name] = (cos, sin)
            if layer_name == "sliding_attention":
                state.census.sliding_rope_builds += 1
            else:
                state.census.full_rope_builds += 1
        return state.metadata.rope[layer_name]

    def apply_wrapper(
        query: Any,
        key: Any,
        cos: Any,
        sin: Any,
        unsqueeze_dim: int = 1,
    ) -> tuple[Any, Any]:
        state.census.apply_rotary_calls += 1
        return original_apply(
            query,
            key,
            cos,
            sin,
            unsqueeze_dim=unsqueeze_dim,
        )

    def repeat_wrapper(value: Any, repetitions: int) -> Any:
        index = state.repeat_index
        if not 0 <= index < _REPEAT_CALLS or repetitions != _KV_REPETITIONS:
            raise RuntimeError("DM05 combined repeat_kv call contract drifted.")
        _, kind_index = divmod(index, 2)
        kind = "K" if kind_index == 0 else "V"
        expected = runtime._pack_workspaces.for_kind(kind)
        if value is not expected:
            raise RuntimeError(
                f"DM05 combined {kind} repeat identity received a foreign tensor."
            )
        state.repeat_index += 1
        state.census.semantic_repeat_calls += 1
        state.census.identity_repeat_returns += 1
        return value

    def attention_wrapper(
        query: Any,
        key: Any,
        value: Any,
        attention_mask: Any,
        layer: Any,
    ) -> Any:
        repeated_key = arch.repeat_kv(key, layer.self_attn.num_key_value_groups)
        repeated_value = arch.repeat_kv(
            value, layer.self_attn.num_key_value_groups
        )
        state.census.bool_mask_calls += 1
        if state.metadata.bool_mask is None:
            bool_mask = attention_mask == 0
            _require_tensor(
                bool_mask,
                (_BATCH_SIZE, 1, _SUFFIX_LENGTH, _KEY_LENGTH),
                dtype=torch.bool,
                device=runtime.device,
                contiguous=True,
                label="suffix boolean mask",
            )
            state.metadata.bool_mask = bool_mask
            state.census.bool_mask_builds += 1
        output = arch.F.scaled_dot_product_attention(
            query,
            repeated_key,
            repeated_value,
            attn_mask=state.metadata.bool_mask,
            scale=layer.self_attn.scaling,
        )
        state.census.sdpa_calls += 1
        output = output.transpose(1, 2).contiguous()
        state.census.post_sdpa_contiguous_builds += 1
        return output

    def time_wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        del self
        state.census.time_cond_calls += 1
        value = original_time(*args, **kwargs)
        _require_tensor(
            value,
            (_BATCH_SIZE, _HIDDEN_SIZE),
            dtype=runtime.noise_dtype,
            device=runtime.device,
            contiguous=True,
            label="AdaRMS condition",
        )
        return value

    def make_modulator_wrapper(site: _ModulationSite):
        def wrapper(self: Any, condition: Any) -> Any:
            del self
            index = state.modulation_index
            if index >= _ADAPTIVE_CALLS:
                raise RuntimeError("DM05 combined table lookup count exceeded 690.")
            expected_site = index % _SITES_PER_STEP
            if site.index != expected_site:
                raise RuntimeError(
                    "DM05 combined modulation site order drifted: "
                    f"{site.index} != {expected_site}."
                )
            _require_tensor(
                condition,
                (_BATCH_SIZE, _HIDDEN_SIZE),
                dtype=runtime.noise_dtype,
                device=runtime.device,
                contiguous=True,
                label="table lookup condition",
            )
            step = index // _SITES_PER_STEP
            state.modulation_index += 1
            state.census.table_lookup_calls += 1
            return runtime._modulation_table.entries_by_site[site.index][step]

        return wrapper

    def adaptive_wrapper(
        self: Any,
        norm: Any,
        x: Any,
        adarms_cond: Any,
        modulator: Any,
    ) -> tuple[Any, Any]:
        del self
        index = state.adaptive_index
        if index >= _ADAPTIVE_CALLS:
            raise RuntimeError("DM05 combined adaptive call count exceeded 690.")
        site = site_by_module.get(id(modulator))
        if site is None or site.index != index % _SITES_PER_STEP:
            raise RuntimeError("DM05 combined adaptive site order drifted.")
        _require_tensor(
            x,
            (_BATCH_SIZE, _SUFFIX_LENGTH, _HIDDEN_SIZE),
            dtype=runtime.noise_dtype,
            device=runtime.device,
            contiguous=True,
            label="adaptive input",
        )
        x_float = x.float()
        variance = torch.mean(torch.square(x_float), dim=-1, keepdim=True)
        state.census.native_var_calls += 1
        eps = getattr(norm, "eps", None)
        if eps is None:
            eps = norm.variance_epsilon
        inverse = torch.rsqrt(variance + eps)
        state.census.native_rsqrt_calls += 1
        modulation = modulator(
            adarms_cond.to(dtype=modulator.weight.dtype)
        )
        if modulation.ndim == 2:
            modulation = modulation[:, None, :]
        _require_tensor(
            modulation,
            (_BATCH_SIZE, 1, _MODULATION_WIDTH),
            dtype=runtime.noise_dtype,
            device=runtime.device,
            contiguous=True,
            label="table modulation",
        )
        scale, shift, gate = torch.chunk(modulation, 3, dim=-1)
        gate = gate.to(x.dtype)
        output = runtime._affine_kernel.launch(
            x,
            inverse,
            scale,
            shift,
            runtime._affine_outputs[index],
        )
        state.adaptive_index += 1
        state.census.adaptive_rmsnorm_calls += 1
        state.census.exact_affine_kernel_calls += 1
        return output, gate

    def make_mlp_wrapper(original: Any):
        def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            del self
            state.census.mlp_calls += 1
            return original(*args, **kwargs)

        return wrapper

    def install() -> None:
        arch.torch = proxy
        arch.make_suffix_attn_mask = mask_wrapper
        arch.apply_rotary_pos_emb = apply_wrapper
        arch.repeat_kv = repeat_wrapper
        model._build_suffix_position_ids = MethodType(position_wrapper, model)
        rotary.forward = MethodType(rotary_wrapper, rotary)
        expert._attn_fn = attention_wrapper
        model._build_adarms_cond = MethodType(time_wrapper, model)
        expert._adaptive_rmsnorm = MethodType(adaptive_wrapper, expert)
        for layer, original in zip(expert.layers, original_mlp):
            layer.mlp.forward = MethodType(make_mlp_wrapper(original), layer.mlp)
        for site in sites:
            site.module.forward = MethodType(
                make_modulator_wrapper(site), site.module
            )

    def restore() -> None:
        arch.torch = original_torch
        arch.make_suffix_attn_mask = original_mask
        arch.apply_rotary_pos_emb = original_apply
        arch.repeat_kv = original_repeat
        arch.F = original_functional
        _restore_instance_binding(
            model, "_build_suffix_position_ids", position_state
        )
        _restore_instance_binding(rotary, "forward", rotary_state)
        _restore_instance_binding(expert, "_attn_fn", attention_state)
        _restore_instance_binding(model, "_build_adarms_cond", time_state)
        _restore_instance_binding(expert, "_adaptive_rmsnorm", adaptive_state)
        for layer, binding in zip(expert.layers, mlp_states):
            _restore_instance_binding(layer.mlp, "forward", binding)
        for site, binding in zip(sites, modulator_states):
            _restore_instance_binding(site.module, "forward", binding)
        restored = (
            arch.torch is original_torch
            and arch.make_suffix_attn_mask is original_mask
            and arch.apply_rotary_pos_emb is original_apply
            and arch.repeat_kv is original_repeat
            and arch.F is original_functional
            and model._build_suffix_position_ids == original_position
            and rotary.forward == original_rotary
            and expert._attn_fn is original_attention
            and model._build_adarms_cond == original_time
            and expert._adaptive_rmsnorm == original_adaptive
            and all(
                layer.mlp.forward == original
                for layer, original in zip(
                    expert.layers, original_mlp
                )
            )
            and all(
                site.module.forward == original
                for site, original in zip(
                    sites, original_modulators
                )
            )
        )
        runtime.combined_patches_restored = bool(restored)
        if not restored:
            raise RuntimeError(
                "DM05 combined capture patch did not restore every binding."
            )

    try:
        install()
        yield
    finally:
        restore()


@contextmanager
def _combined_capture_patch(
    runtime: "DM05CombinedRuntime",
    state: _CaptureState,
) -> Iterator[None]:
    """Serialize module-global patch ownership from snapshot through restore."""
    with _PATCH_LOCK:
        with _combined_capture_patch_unlocked(runtime, state):
            yield


class DM05CombinedRuntime(DM05StaticMaskPrefixGraphRuntime):
    """One exact fixed-cell composition of the accepted mechanisms."""

    def __init__(
        self,
        model: Any,
        *,
        request_prefix_len: int = _PREFIX_LENGTH,
        diffusion_steps: int = _SUFFIX_LENGTH,
    ) -> None:
        super().__init__(
            model,
            request_prefix_len=request_prefix_len,
            diffusion_steps=diffusion_steps,
        )
        if str(torch.__version__) != _REQUIRED_TORCH_VERSION:
            raise RuntimeError(
                "DM05 default_exact_combined requires torch==2.11.0+cu130, got "
                f"{torch.__version__}; there is no fallback."
            )
        if str(torch.version.cuda) != _REQUIRED_CUDA_VERSION:
            raise RuntimeError(
                "DM05 default_exact_combined requires CUDA 13.0 PyTorch, got "
                f"{torch.version.cuda}; there is no fallback."
            )
        source = Path(dm05_arch.__file__)
        self.dm05_arch_source_sha256 = _file_sha256(source)
        if self.dm05_arch_source_sha256 != _DM05_ARCH_SHA256:
            raise RuntimeError(
                "DM05 action architecture source drifted: "
                f"{self.dm05_arch_source_sha256} != {_DM05_ARCH_SHA256}."
            )
        self._modulation_sites = self._build_modulation_sites()
        self._validate_fixed_model_contract()
        self._affine_kernel = _ExactPostReductionAffine(self.device)
        self._modulation_table: _ModulationTable | None = None
        self._metadata_owner: _MetadataOwner | None = None
        self._pack_workspaces: _PackWorkspaces | None = None
        self._affine_outputs: tuple[Any, ...] = ()
        self._pack_initial_addresses: tuple[int, ...] = ()
        self._affine_initial_addresses: tuple[int, ...] = ()
        self._combined_address_owners: dict[str, tuple[int, ...]] = {}
        self._capture_census: dict[str, int] = {}
        self.modulation_table_build_count = 0
        self.modulation_table_entry_count = 0
        self.modulation_table_rng_unchanged = False
        self.modulation_table_addresses_stable = False
        self.modulation_table_immutable = False
        self.metadata_owner_tensor_count = 0
        self.metadata_owner_addresses_stable = False
        self.replay_content_baseline_established = False
        self.metadata_owner_second_replay_exact = False
        self.pack_workspace_count = 0
        self.pack_workspace_addresses_stable = False
        self.pack_workspace_second_replay_exact = False
        self.affine_output_count = 0
        self.affine_output_addresses_stable = False
        self.affine_output_second_replay_exact = False
        self.combined_capture_census_exact = False
        self.combined_patches_restored = True
        self.combined_ready = False
        self.startup_native_suffix_reference_count = 0
        self.startup_graph_replay_count = 0
        self.startup_first_replay_output_bitwise_exact = False
        self.startup_second_replay_output_bitwise_exact = False
        self.startup_changed_noise_control_count = 0
        self.startup_changed_noise_graph_vs_eager_bitwise_exact = False
        self.startup_changed_noise_differs_from_zero_baseline = False
        self.startup_static_zero_input_restored = False
        self.startup_static_zero_repeat_bitwise_exact = False
        self.startup_output_poison_count = 0
        self.startup_native_reference_bitwise = False
        self._metadata_first_replay_sha256 = ""
        self._workspace_first_replay_sha256 = ""
        self._affine_first_replay_sha256 = ""

    def _build_modulation_sites(self) -> tuple[_ModulationSite, ...]:
        expert = self.model.model.action_expert
        sites = []
        for layer in range(len(expert.layers)):
            sites.append(
                _ModulationSite(
                    len(sites), "input", expert.input_time_modulators[layer]
                )
            )
            sites.append(
                _ModulationSite(
                    len(sites), "mlp", expert.mlp_time_modulators[layer]
                )
            )
        sites.append(_ModulationSite(len(sites), "final", expert.final_time_modulator))
        if len(sites) != _SITES_PER_STEP:
            raise RuntimeError("DM05 combined modulation topology drifted.")
        return tuple(sites)

    def _validate_fixed_model_contract(self) -> None:
        dm05 = self.model.model
        expert = dm05.action_expert
        if self.model.training or getattr(expert, "_suffix_attn_backend", None) != "sdpa":
            raise RuntimeError("DM05 combined requires model.eval() and native SDPA.")
        if (
            int(dm05.action_in_proj.out_features) != _HIDDEN_SIZE
            or int(dm05.config.chunk_size) != _SUFFIX_LENGTH
            or int(dm05.config.action_dim) != _MODEL_ACTION_DIM
            or len(expert.layers) != _LAYER_COUNT
        ):
            raise RuntimeError("DM05 combined fixed model dimensions drifted.")
        if dm05.action_in_proj.weight.dtype != torch.bfloat16:
            raise RuntimeError("DM05 combined requires BF16 model weights.")
        for label, linear in (
            ("time_mlp_in", dm05.time_mlp_in),
            ("time_mlp_out", dm05.time_mlp_out),
        ):
            if (
                int(linear.in_features) != _HIDDEN_SIZE
                or int(linear.out_features) != _HIDDEN_SIZE
                or linear.weight.dtype != torch.bfloat16
                or linear.weight.device != self.device
            ):
                raise RuntimeError(
                    f"DM05 combined {label} contract drifted."
                )
        layer_types = [str(layer.attention_type) for layer in expert.layers]
        if (
            layer_types.count("sliding_attention") != _SLIDING_LAYERS
            or layer_types.count("full_attention") != _FULL_LAYERS
        ):
            raise RuntimeError("DM05 combined attention layer topology drifted.")
        for layer in expert.layers:
            attention = layer.self_attn
            if (
                int(attention.head_dim) != _HEAD_DIM
                or int(attention.q_proj.out_features // _HEAD_DIM) != _QUERY_HEADS
                or int(attention.k_proj.out_features // _HEAD_DIM) != _KV_HEADS
                or int(attention.num_key_value_groups) != _KV_REPETITIONS
            ):
                raise RuntimeError("DM05 combined attention head topology drifted.")
        rotary = expert.rotary_emb
        rope_types = {str(key): str(value) for key, value in rotary.rope_type.items()}
        if rope_types != {
            "sliding_attention": "default",
            "full_attention": "linear",
        }:
            raise RuntimeError("DM05 combined RoPE types drifted.")
        if any("dynamic" in value or value == "longrope" for value in rope_types.values()):
            raise RuntimeError("DM05 combined does not support stateful RoPE.")
        parameters = expert.config.rope_parameters
        if (
            float(parameters["sliding_attention"]["rope_theta"]) != 10000.0
            or float(parameters["full_attention"]["rope_theta"]) != 1000000.0
            or float(parameters["full_attention"].get("factor", 0.0)) != 8.0
        ):
            raise RuntimeError("DM05 combined RoPE parameters drifted.")
        for site in self._modulation_sites:
            module = site.module
            if (
                int(module.in_features) != _HIDDEN_SIZE
                or int(module.out_features) != _MODULATION_WIDTH
                or module.bias is None
                or _shape(module.bias) != (_MODULATION_WIDTH,)
                or module.weight.dtype != torch.bfloat16
                or module.weight.device != self.device
            ):
                raise RuntimeError("DM05 combined modulation linear contract drifted.")
        adaptive_source = inspect.getsource(expert.__class__._adaptive_rmsnorm)
        ordered = (
            "var = torch.mean(torch.square(x.float()), dim=-1, keepdim=True)",
            "normed = x * torch.rsqrt(var + eps)",
            "modulation = modulator(adarms_cond.to(dtype=modulator.weight.dtype))",
            "scale, shift, gate = torch.chunk(modulation, 3, dim=-1)",
            "normed = normed * (1.0 + scale.float()) + shift.float()",
            "return normed.to(dtype), gate.to(dtype)",
        )
        offsets = [adaptive_source.find(fragment) for fragment in ordered]
        if any(offset < 0 for offset in offsets) or offsets != sorted(offsets):
            raise RuntimeError("DM05 combined native adaptive RMS order drifted.")

    def _build_modulation_table(self) -> _ModulationTable:
        if self.modulation_table_build_count != 0:
            raise RuntimeError("DM05 combined modulation table may build only once.")
        rng_before_cpu = torch.get_rng_state().clone()
        rng_before_cuda = tuple(
            value.clone() for value in torch.cuda.get_rng_state_all()
        )
        entries = [[] for _ in range(_SITES_PER_STEP)]
        with torch.inference_mode():
            for time_value in _native_time_values():
                time_tensor = torch.full(
                    (1,),
                    time_value,
                    device=self.device,
                    dtype=self.noise_dtype,
                )
                condition = self.model._build_adarms_cond(
                    time_tensor, self.noise_dtype
                )
                _require_tensor(
                    condition,
                    (_BATCH_SIZE, _HIDDEN_SIZE),
                    dtype=self.noise_dtype,
                    device=self.device,
                    contiguous=True,
                    label="table condition",
                )
                for site in self._modulation_sites:
                    value = site.module(condition.to(dtype=site.module.weight.dtype))
                    _require_tensor(
                        value,
                        (_BATCH_SIZE, _MODULATION_WIDTH),
                        dtype=self.noise_dtype,
                        device=self.device,
                        contiguous=True,
                        label="modulation table entry",
                    )
                    torch._dynamo.mark_static_address(value)
                    entries[site.index].append(value)
        rng_after_cpu = torch.get_rng_state()
        rng_after_cuda = tuple(torch.cuda.get_rng_state_all())
        rng_unchanged = bool(torch.equal(rng_before_cpu, rng_after_cpu)) and len(
            rng_before_cuda
        ) == len(rng_after_cuda) and all(
            bool(torch.equal(left, right))
            for left, right in zip(rng_before_cuda, rng_after_cuda)
        )
        table = _ModulationTable(
            entries_by_site=tuple(tuple(values) for values in entries),
            initial_addresses=(),
            initial_sha256="",
            rng_unchanged=rng_unchanged,
        )
        tensors = table.tensors()
        if len(tensors) != _ADAPTIVE_CALLS or not rng_unchanged:
            raise RuntimeError("DM05 combined modulation table contract failed.")
        table.initial_addresses = _data_addresses(tensors)
        table.initial_sha256 = _sequence_sha256(tensors)
        self.modulation_table_build_count = 1
        self.modulation_table_entry_count = len(tensors)
        self.modulation_table_rng_unchanged = True
        return table

    def _allocate_combined_owners(self) -> None:
        workspace_shape = (
            _BATCH_SIZE,
            _QUERY_HEADS,
            _KEY_LENGTH,
            _HEAD_DIM,
        )
        key = torch.empty(
            workspace_shape, device=self.device, dtype=self.noise_dtype
        )
        value = torch.empty(
            workspace_shape, device=self.device, dtype=self.noise_dtype
        )
        for label, tensor in (("K workspace", key), ("V workspace", value)):
            _require_tensor(
                tensor,
                workspace_shape,
                dtype=self.noise_dtype,
                device=self.device,
                contiguous=True,
                label=label,
            )
            torch._dynamo.mark_static_address(tensor)
        if int(key.untyped_storage().data_ptr()) == int(
            value.untyped_storage().data_ptr()
        ):
            raise RuntimeError("DM05 combined K/V workspaces alias.")
        self._pack_workspaces = _PackWorkspaces(key=key, value=value)
        self._affine_outputs = tuple(
            torch.empty(
                (_BATCH_SIZE, _SUFFIX_LENGTH, _HIDDEN_SIZE),
                device=self.device,
                dtype=self.noise_dtype,
            )
            for _ in range(_ADAPTIVE_CALLS)
        )
        for output in self._affine_outputs:
            torch._dynamo.mark_static_address(output)
        self.pack_workspace_count = 2
        self.affine_output_count = len(self._affine_outputs)
        self._pack_initial_addresses = _data_addresses(
            self._pack_workspaces.tensors()
        )
        self._affine_initial_addresses = _data_addresses(self._affine_outputs)

    def _new_capture_state(self) -> _CaptureState:
        return _CaptureState(census=_CaptureCensus(), metadata=_MetadataOwner())

    def _capture_one_suffix_run(self, state: _CaptureState) -> None:
        with _combined_capture_patch(self, state):
            self._run_static_suffix()
        if state.census.snapshot() != _expected_capture_census():
            raise RuntimeError(
                "DM05 combined suffix census drifted: "
                f"{state.census.snapshot()}."
            )

    def _bind_owner_addresses(self) -> None:
        assert self._modulation_table is not None
        assert self._metadata_owner is not None
        assert self._pack_workspaces is not None
        table_tensors = self._modulation_table.tensors()
        metadata_tensors = self._metadata_owner.tensors()
        workspace_tensors = self._pack_workspaces.tensors()
        if len(metadata_tensors) != 7:
            raise RuntimeError(
                f"DM05 combined metadata owner requires 7 tensors, got "
                f"{len(metadata_tensors)}."
            )
        if len(set(_storage_addresses(metadata_tensors))) != len(metadata_tensors):
            raise RuntimeError("DM05 combined metadata tensors alias.")
        all_owned_tensors = (
            table_tensors
            + metadata_tensors
            + workspace_tensors
            + self._affine_outputs
        )
        if len(set(_storage_addresses(all_owned_tensors))) != len(
            all_owned_tensors
        ):
            raise RuntimeError("DM05 combined fixed owners alias each other.")
        if _data_addresses(table_tensors) != self._modulation_table.initial_addresses:
            raise RuntimeError("DM05 combined table address changed during capture.")
        if _data_addresses(workspace_tensors) != self._pack_initial_addresses:
            raise RuntimeError(
                "DM05 combined K/V workspace address changed during capture."
            )
        if _data_addresses(self._affine_outputs) != self._affine_initial_addresses:
            raise RuntimeError(
                "DM05 combined affine output address changed during capture."
            )
        self._combined_address_owners = {
            "table": _data_addresses(table_tensors),
            "metadata": _data_addresses(metadata_tensors),
            "workspaces": _data_addresses(workspace_tensors),
            "affine_outputs": _data_addresses(self._affine_outputs),
        }
        self.metadata_owner_tensor_count = len(metadata_tensors)

    def _record_replay_content_baseline(self) -> None:
        if not self._combined_address_owners:
            raise RuntimeError("DM05 combined owners are not bound.")
        assert self._metadata_owner is not None
        assert self._pack_workspaces is not None
        self._metadata_first_replay_sha256 = _sequence_sha256(
            self._metadata_owner.tensors()
        )
        self._workspace_first_replay_sha256 = _sequence_sha256(
            self._pack_workspaces.tensors()
        )
        self._affine_first_replay_sha256 = _sequence_sha256(
            self._affine_outputs
        )
        self.replay_content_baseline_established = True

    def _assert_combined_owners(self, *, check_bytes: bool) -> None:
        if not self._combined_address_owners:
            raise RuntimeError("DM05 combined owners are not bound.")
        assert self._modulation_table is not None
        assert self._metadata_owner is not None
        assert self._pack_workspaces is not None
        current = {
            "table": _data_addresses(self._modulation_table.tensors()),
            "metadata": _data_addresses(self._metadata_owner.tensors()),
            "workspaces": _data_addresses(self._pack_workspaces.tensors()),
            "affine_outputs": _data_addresses(self._affine_outputs),
        }
        self.modulation_table_addresses_stable = (
            current["table"] == self._combined_address_owners["table"]
        )
        self.metadata_owner_addresses_stable = (
            current["metadata"] == self._combined_address_owners["metadata"]
        )
        self.pack_workspace_addresses_stable = (
            current["workspaces"] == self._combined_address_owners["workspaces"]
        )
        self.affine_output_addresses_stable = (
            current["affine_outputs"]
            == self._combined_address_owners["affine_outputs"]
        )
        if not all(
            (
                self.modulation_table_addresses_stable,
                self.metadata_owner_addresses_stable,
                self.pack_workspace_addresses_stable,
                self.affine_output_addresses_stable,
            )
        ):
            raise RuntimeError("DM05 combined static owner address drifted.")
        if check_bytes:
            self.modulation_table_immutable = (
                _sequence_sha256(self._modulation_table.tensors())
                == self._modulation_table.initial_sha256
            )
            self.metadata_owner_second_replay_exact = (
                _sequence_sha256(self._metadata_owner.tensors())
                == self._metadata_first_replay_sha256
            )
            self.pack_workspace_second_replay_exact = (
                _sequence_sha256(self._pack_workspaces.tensors())
                == self._workspace_first_replay_sha256
            )
            self.affine_output_second_replay_exact = (
                _sequence_sha256(self._affine_outputs)
                == self._affine_first_replay_sha256
            )
            if not self.modulation_table_immutable:
                raise RuntimeError("DM05 combined modulation table bytes drifted.")
            if not self.replay_content_baseline_established:
                raise RuntimeError(
                    "DM05 combined replay content baseline is not established."
                )
            if not self.metadata_owner_second_replay_exact:
                raise RuntimeError(
                    "DM05 combined metadata owner changed for the identical "
                    "second startup replay."
                )
            if not self.pack_workspace_second_replay_exact:
                raise RuntimeError(
                    "DM05 combined K/V workspaces changed for the identical "
                    "second startup replay."
                )
            if not self.affine_output_second_replay_exact:
                raise RuntimeError(
                    "DM05 combined affine outputs changed for the identical "
                    "second startup replay."
                )

    def _assert_static_addresses(self) -> None:
        super()._assert_static_addresses()
        if self.combined_ready:
            self._assert_combined_owners(check_bytes=False)

    def _validate_prefix_cache_contract(self) -> None:
        assert self._static_cache is not None
        keys, values = self._cache_tensors(self._static_cache)
        if len(keys) != _LAYER_COUNT or len(values) != _LAYER_COUNT:
            raise RuntimeError("DM05 combined prefix cache layer count drifted.")
        for label, tensor in (
            [("K", value) for value in keys]
            + [("V", value) for value in values]
        ):
            _require_tensor(
                tensor,
                (_BATCH_SIZE, _KV_HEADS, _PREFIX_LENGTH, _HEAD_DIM),
                dtype=self.noise_dtype,
                device=self.device,
                contiguous=None,
                label=f"prefix {label}",
            )

    def _initialize_suffix_graph(self) -> None:
        """Build all fixed owners and capture exactly one combined suffix path."""
        self._validate_prefix_cache_contract()
        assert self._static_output is not None
        native_reference = torch.empty_like(self._static_output)
        torch._dynamo.mark_static_address(native_reference)
        with torch.inference_mode():
            self._run_static_suffix()
            native_reference.copy_(self._static_output)
        torch.cuda.synchronize(self.device)
        if not bool(torch.isfinite(native_reference).all().item()):
            raise RuntimeError("DM05 official zero-noise baseline is non-finite.")
        self.startup_native_suffix_reference_count = 1

        self._modulation_table = self._build_modulation_table()
        self._allocate_combined_owners()

        with torch.inference_mode():
            for _ in range(self._WARMUP_COUNT):
                self._capture_one_suffix_run(self._new_capture_state())
        torch.cuda.synchronize(self.device)
        if self._affine_kernel.compile_count != 1:
            raise RuntimeError("DM05 combined exact-affine must compile exactly once.")

        capture_state = self._new_capture_state()
        suffix_graph = torch.cuda.CUDAGraph()
        try:
            with (
                torch.inference_mode(),
                _combined_capture_patch(self, capture_state),
                torch.cuda.graph(suffix_graph),
            ):
                self._run_static_suffix()
        except Exception as exc:
            raise RuntimeError(
                "DM05 combined suffix capture failed; there is no fallback: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        self._capture_census = capture_state.census.snapshot()
        self.combined_capture_census_exact = (
            self._capture_census == _expected_capture_census()
        )
        if not self.combined_capture_census_exact:
            raise RuntimeError(
                f"DM05 combined capture census drifted: {self._capture_census}."
            )
        self._metadata_owner = capture_state.metadata
        for value in self._metadata_owner.tensors():
            torch._dynamo.mark_static_address(value)
        self._bind_owner_addresses()
        self._assert_combined_owners(check_bytes=False)

        self._suffix_graph = suffix_graph
        self.suffix_startup_capture_count = 1
        self.suffix_capture_execution_count = 1

        # Captured producers have not established readable content yet. The
        # first replay creates the baseline; the second identical replay must
        # reproduce both the native output and every dynamic owner exactly.
        self._static_output.fill_(float("nan"))
        self.startup_output_poison_count += 1
        torch.cuda.synchronize(self.device)
        suffix_graph.replay()
        self.startup_graph_replay_count += 1
        torch.cuda.synchronize(self.device)
        self.startup_first_replay_output_bitwise_exact = bool(
            torch.equal(native_reference, self._static_output)
        )
        if not self.startup_first_replay_output_bitwise_exact:
            raise RuntimeError(
                "DM05 combined first replay differs from the official suffix."
            )
        self._assert_static_addresses()
        self._verify_mask_owners(check_bytes=True)
        self._assert_combined_owners(check_bytes=False)
        self._record_replay_content_baseline()

        self._static_output.fill_(float("nan"))
        self.startup_output_poison_count += 1
        torch.cuda.synchronize(self.device)
        suffix_graph.replay()
        self.startup_graph_replay_count += 1
        torch.cuda.synchronize(self.device)
        self.startup_second_replay_output_bitwise_exact = bool(
            torch.equal(native_reference, self._static_output)
        )
        if not self.startup_second_replay_output_bitwise_exact:
            raise RuntimeError(
                "DM05 combined second replay differs from the official suffix."
            )
        self._assert_static_addresses()
        self._verify_mask_owners(check_bytes=True)
        self._assert_combined_owners(check_bytes=True)

        assert self._static_noise is not None
        control_noise = (
            (torch.arange(
                self._static_noise.numel(),
                device=self.device,
                dtype=torch.float32,
            ) % 31)
            .sub_(15)
            .div_(32)
            .reshape_as(self._static_noise)
            .to(dtype=self.noise_dtype)
        )
        if int(torch.count_nonzero(control_noise).item()) == 0:
            raise RuntimeError("DM05 changed-noise startup control is zero.")
        self._static_noise.copy_(control_noise)
        with torch.inference_mode():
            self._run_static_suffix()
        torch.cuda.synchronize(self.device)
        if not bool(torch.isfinite(self._static_output).all().item()):
            raise RuntimeError("DM05 changed-noise eager control is non-finite.")
        control_reference = torch.empty_like(self._static_output)
        control_reference.copy_(self._static_output)
        self.startup_changed_noise_control_count = 1
        self.startup_changed_noise_differs_from_zero_baseline = not bool(
            torch.equal(control_reference, native_reference)
        )
        if not self.startup_changed_noise_differs_from_zero_baseline:
            raise RuntimeError(
                "DM05 changed-noise control did not differ from zero baseline."
            )

        self._static_output.fill_(float("nan"))
        self.startup_output_poison_count += 1
        torch.cuda.synchronize(self.device)
        suffix_graph.replay()
        self.startup_graph_replay_count += 1
        torch.cuda.synchronize(self.device)
        self.startup_changed_noise_graph_vs_eager_bitwise_exact = bool(
            torch.equal(control_reference, self._static_output)
        )
        if not self.startup_changed_noise_graph_vs_eager_bitwise_exact:
            raise RuntimeError(
                "DM05 changed-noise graph replay differs from eager control."
            )
        self._assert_static_addresses()
        self._verify_mask_owners(check_bytes=True)
        self._assert_combined_owners(check_bytes=False)

        self._static_noise.zero_()
        self.startup_static_zero_input_restored = (
            int(torch.count_nonzero(self._static_noise).item()) == 0
        )
        if not self.startup_static_zero_input_restored:
            raise RuntimeError("DM05 static zero noise was not restored.")
        self._static_output.fill_(float("nan"))
        self.startup_output_poison_count += 1
        torch.cuda.synchronize(self.device)
        suffix_graph.replay()
        self.startup_graph_replay_count += 1
        torch.cuda.synchronize(self.device)
        self.startup_static_zero_repeat_bitwise_exact = bool(
            torch.equal(native_reference, self._static_output)
        )
        if not self.startup_static_zero_repeat_bitwise_exact:
            raise RuntimeError(
                "DM05 restored zero-noise replay differs from native baseline."
            )
        self._assert_static_addresses()
        self._verify_mask_owners(check_bytes=True)
        self._assert_combined_owners(check_bytes=True)
        self.startup_native_reference_bitwise = True
        assert self._static_output is not None
        if not bool(torch.isfinite(self._static_output).all().item()):
            raise RuntimeError("DM05 combined suffix produced non-finite output.")
        self.combined_ready = True

    def proof_snapshot(self) -> dict[str, Any]:
        """Return fixed-cell path proof without raw process addresses."""
        proof = super().proof_snapshot()
        codegen = self._affine_kernel.codegen
        proof.update(
            {
                "schema": _SCHEMA,
                "selector": _EXECUTION_BACKEND,
                "execution_backend": _EXECUTION_BACKEND,
                "arithmetic_backend": _ARITHMETIC_BACKEND,
                "graph_scope": _GRAPH_SCOPE,
                "combined_ready": self.combined_ready,
                "no_fallback": True,
                "combined_mechanisms": [
                    "static_mask_prefix_suffix_graph",
                    "modulation_table_690",
                    "suffix_metadata_first_owner",
                    "two_expanded_kv_pack_workspaces",
                    "exact_postreduce_affine_triton",
                ],
                "dm05_arch_source_sha256": self.dm05_arch_source_sha256,
                "fixed_cell": {
                    "batch_size": _BATCH_SIZE,
                    "prefix_length": _PREFIX_LENGTH,
                    "suffix_length": _SUFFIX_LENGTH,
                    "hidden_size": _HIDDEN_SIZE,
                    "model_action_dim": _MODEL_ACTION_DIM,
                    "layer_count": _LAYER_COUNT,
                    "query_heads": _QUERY_HEADS,
                    "kv_heads": _KV_HEADS,
                    "head_dim": _HEAD_DIM,
                    "device_capability": [8, 9],
                    "dtype": "torch.bfloat16",
                    "attention_backend": "sdpa",
                    "torch_version": "2.11.0+cu130",
                    "triton_version": "3.6.0",
                },
                "runtime_versions": {
                    "torch": str(torch.__version__),
                    "torch_cuda": str(torch.version.cuda),
                    "triton": str(self._affine_kernel.triton.__version__),
                },
                "combined_capture_census": dict(self._capture_census),
                "combined_expected_census": _expected_capture_census(),
                "combined_capture_census_exact": (
                    self.combined_capture_census_exact
                ),
                "combined_patches_restored": self.combined_patches_restored,
                "modulation_table_build_count": (
                    self.modulation_table_build_count
                ),
                "modulation_table_entry_count": (
                    self.modulation_table_entry_count
                ),
                "modulation_table_rng_unchanged": (
                    self.modulation_table_rng_unchanged
                ),
                "modulation_table_addresses_stable": (
                    self.modulation_table_addresses_stable
                ),
                "modulation_table_immutable": self.modulation_table_immutable,
                "metadata_owner_tensor_count": self.metadata_owner_tensor_count,
                "metadata_owner_addresses_stable": (
                    self.metadata_owner_addresses_stable
                ),
                "replay_content_baseline_established": (
                    self.replay_content_baseline_established
                ),
                "metadata_owner_second_replay_exact": (
                    self.metadata_owner_second_replay_exact
                ),
                "metadata_owner_bytes_are_request_dynamic": True,
                "pack_workspace_count": self.pack_workspace_count,
                "pack_workspace_addresses_stable": (
                    self.pack_workspace_addresses_stable
                ),
                "pack_workspace_second_replay_exact": (
                    self.pack_workspace_second_replay_exact
                ),
                "affine_output_count": self.affine_output_count,
                "affine_output_addresses_stable": (
                    self.affine_output_addresses_stable
                ),
                "affine_output_second_replay_exact": (
                    self.affine_output_second_replay_exact
                ),
                "exact_affine_compile_count": self._affine_kernel.compile_count,
                "exact_affine_fallback_count": self._affine_kernel.fallback_count,
                "exact_affine_ptx_codegen_verified": bool(
                    codegen and codegen["ptx_codegen_verified"]
                ),
                "exact_affine_ptx_sha256": (
                    None if codegen is None else codegen["ptx_sha256"]
                ),
                "exact_affine_cubin_sha256": (
                    None if codegen is None else codegen["cubin_sha256"]
                ),
                "sass_external_receipt_required": True,
                "exact_affine_ptx_codegen": (
                    None if codegen is None else dict(codegen)
                ),
                "startup_native_suffix_reference_count": (
                    self.startup_native_suffix_reference_count
                ),
                "startup_graph_replay_count": self.startup_graph_replay_count,
                "startup_first_replay_output_bitwise_exact": (
                    self.startup_first_replay_output_bitwise_exact
                ),
                "startup_second_replay_output_bitwise_exact": (
                    self.startup_second_replay_output_bitwise_exact
                ),
                "startup_changed_noise_control_count": (
                    self.startup_changed_noise_control_count
                ),
                "startup_changed_noise_graph_vs_eager_bitwise_exact": (
                    self.startup_changed_noise_graph_vs_eager_bitwise_exact
                ),
                "startup_changed_noise_differs_from_zero_baseline": (
                    self.startup_changed_noise_differs_from_zero_baseline
                ),
                "startup_static_zero_input_restored": (
                    self.startup_static_zero_input_restored
                ),
                "startup_static_zero_repeat_bitwise_exact": (
                    self.startup_static_zero_repeat_bitwise_exact
                ),
                "startup_output_poison_count": self.startup_output_poison_count,
                "startup_native_reference_bitwise": (
                    self.startup_native_reference_bitwise
                ),
                "source_candidate_gpu_validation_required": True,
            }
        )
        return proof

    def infer(self, **model_inputs: Any) -> torch.Tensor:
        return self.inference_action(**model_inputs)

    def path_proof(self) -> dict[str, Any]:
        return self.proof_snapshot()

    def close(self) -> None:
        super().close()
        self._modulation_table = None
        self._metadata_owner = None
        self._pack_workspaces = None
        self._affine_outputs = ()
        self._pack_initial_addresses = ()
        self._affine_initial_addresses = ()
        self._combined_address_owners = {}
        self.combined_ready = False
