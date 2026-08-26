"""DM05-libero external-runtime adapter for the pinned official OpenDM source.

The ApxInf-owned boundary is deliberately small: this module validates the
fixed LIBERO request contract, exposes it through :class:`AutoPolicy`, and
returns the ordinary ApxInf ``Policy`` result. OpenDM remains the owner of the
base DM05 graph and ``inference_action``; the selected execution runtime is
injected through a neutral factory seam. Heavy imports
(``torch``, ``transformers``, ``cv2`` and ``opendm``) are deferred until
``from_pretrained`` so importing :mod:`apxinf` stays CPU/CUDA neutral.

This is not a native Rust ``VlaRuntime`` port. The required 8D state field is
validated at the wire boundary but is not consumed by the pinned checkpoint,
whose official LIBERO configuration sets ``add_state=False``.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import importlib.util
import json
import math
import subprocess
import threading
import time
from io import BytesIO
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

import numpy as np
from PIL import Image, UnidentifiedImageError

from ..registry import register_policy

__all__ = ["Dm05Policy"]

MODEL_REVISION = "25a8e0d38a8eaeaae41a44d7b4a2378fd8ce1088"
MODEL_SHA256 = "575d0d8e0f75822e95f7adf3a5e62a7c331da0b82e6fc3efeea19ef1b927353f"
MODEL_SIZE = 11_658_431_136
TOKENIZER_SHA256 = "daab2354f8a74e70d70b4d1f804939b68a8c9624dd06cb7858e52dd8970e9726"
TOKENIZER_SIZE = 33_384_567
OPENDM_COMMIT = "e41e501bb82e9c3cb8138c0fb4687faa5f98c690"

_CHECKPOINT_MANIFEST = {
    "chat_template.jinja": (
        1_532,
        "7de1c58e208eda46e9c7f86397df37ec49883aeece39fb961e0a6b24088dd3c4",
    ),
    "config.json": (
        6_795,
        "43b2a56ed9c79c3068849caa0a140458515e654d34ae0b06bb0bbb3ef4dd0f80",
    ),
    "generation_config.json": (
        204,
        "640dbc106facaf0fb90980b5e182ce0c1fcfad6e88da14737578b5b65cb42f7a",
    ),
    "model.safetensors": (MODEL_SIZE, MODEL_SHA256),
    "norm_stats.json": (
        1_900,
        "06382f26d9f9fdba10ee2dba77783ec8c31e6a6dcb348806583cb6217e18303b",
    ),
    "processor_config.json": (
        560,
        "9eb2e8baf401c81b1517343d1dfc799a4c1b2238acaece111fe68f5fbe3a8d57",
    ),
    "tokenizer.json": (TOKENIZER_SIZE, TOKENIZER_SHA256),
    "tokenizer_config.json": (
        715,
        "eb28e3a9807f77cd74dce1b8aed91884621c0302941794470c5a46f884462615",
    ),
}

EXECUTION_BACKEND_DEFAULT = "default"
EXECUTION_BACKEND_COMBINED = "default_exact_combined"
EXECUTION_BACKENDS = (EXECUTION_BACKEND_DEFAULT, EXECUTION_BACKEND_COMBINED)
HOST_THREAD_POLICY = "fixed_intraop_2"
HOST_TORCH_INTRAOP_THREADS = 2

ACTION_HORIZON = 10
ACTION_DIM = 7
MODEL_ACTION_DIM = 32
STATE_DIM = 8
IMAGE_SIZE = 448
IMAGE_PROMPTS = ("Head", "Left wrist")
DIFFUSION_STEPS = 10
EXACT_SEED = 7
EXACT_PREFIX_LEN = 564
HISTORY_PAD_TOKEN_ID = 7

# The combined runtime installs module-global capture bindings temporarily.
# Serializing every ApxInf-owned DM05 inference makes lazy capture exclusive to
# all default and combined readers in this process.
_DM05_PROCESS_INFERENCE_LOCK = threading.Lock()

_CANONICAL_POLICY_METADATA = {
    "schema",
    "model_type",
    "model_revision",
    "action_horizon",
    "action_dim",
    "model_action_dim",
    "state_dim",
    "image_size",
    "image_prompts",
    "robot_type",
    "diffusion_steps",
    "concurrency",
}

_BACKEND_OWNED_METADATA = {
    "backend",
    "device",
    "execution_backend",
    "runtime_selector",
    "host_thread_policy",
    "torch_intraop_threads",
    "process_inference_policy",
    "opendm_commit",
    "precision",
    "llm_attention",
    "vision_attention",
    "action_attention",
    "path_proof",
}

_PROTECTED_POLICY_METADATA = (
    _CANONICAL_POLICY_METADATA | _BACKEND_OWNED_METADATA
)

_TORCH_SEED_MIN = -(1 << 63)
_TORCH_SEED_MAX = (1 << 64) - 1


def _validate_execution_backend(value: str) -> str:
    if value not in EXECUTION_BACKENDS:
        raise ValueError(
            f"unsupported DM05 execution_backend {value!r}; "
            f"expected one of {EXECUTION_BACKENDS}"
        )
    return value


def _uses_combined_runtime(execution_backend: str) -> bool:
    return execution_backend == EXECUTION_BACKEND_COMBINED


def _validate_host_thread_policy(torch_module: Any, execution_backend: str) -> None:
    if not _uses_combined_runtime(execution_backend):
        return
    observed = int(torch_module.get_num_threads())
    if observed != HOST_TORCH_INTRAOP_THREADS:
        raise RuntimeError(
            "default_exact_combined requires torch intra-op threads="
            f"{HOST_TORCH_INTRAOP_THREADS}, got {observed}; set OMP_NUM_THREADS and "
            "MKL_NUM_THREADS before importing torch"
        )


def _validate_policy_metadata(metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    if metadata is None:
        return {}
    values = dict(metadata)
    conflicts = sorted(set(values) & _PROTECTED_POLICY_METADATA)
    if conflicts:
        raise ValueError(f"DM05 metadata cannot override protected fields: {conflicts}")
    return values


class Dm05Backend(Protocol):
    """The one volatile seam behind the stable ApxInf policy contract."""

    metadata: Mapping[str, Any]

    def infer(
        self,
        *,
        prompt: str,
        state: np.ndarray,
        images: Sequence[Image.Image],
        robot_type: str,
        num_steps: int,
        seed: int,
    ) -> tuple[np.ndarray, float]: ...

    def close(self) -> None: ...


class RuntimeFactory(Protocol):
    """Neutral construction seam selected by ApxInf's runtime selector."""

    def __call__(
        self,
        model: Any,
        *,
        selector: str,
        request_prefix_len: int,
        diffusion_steps: int,
    ) -> Any | None: ...


def _load_default_runtime_factory() -> RuntimeFactory:
    from .dm05_runtime import create_dm05_runtime

    if not callable(create_dm05_runtime):
        raise RuntimeError("DM05 runtime factory is not callable")
    return create_dm05_runtime


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verified_file(path: Path, *, size: int, sha256: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"DM05 required file is missing or not regular: {path}")
    observed_size = path.stat().st_size
    if observed_size != size:
        raise ValueError(
            f"DM05 file size mismatch for {path.name}: {observed_size} != {size}"
        )
    observed_sha = _sha256(path)
    if observed_sha != sha256:
        raise ValueError(
            f"DM05 SHA-256 mismatch for {path.name}: {observed_sha} != {sha256}"
        )


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {label}: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} root must be an object")
    return value


def _validate_checkpoint(model_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    for filename, (size, sha256) in _CHECKPOINT_MANIFEST.items():
        _verified_file(model_dir / filename, size=size, sha256=sha256)

    config = _read_json_object(model_dir / "config.json", label="DM05 config")
    if config.get("model_type") != "dm05":
        raise ValueError("DM05 config.model_type must be 'dm05'")
    if config.get("architectures") != ["DM05ForConditionalGeneration"]:
        raise ValueError("DM05 architecture identity mismatch")
    if config.get("dtype") != "bfloat16":
        raise ValueError("DM05 deployment supports the pinned BF16 checkpoint only")
    if config.get("action_dim") != MODEL_ACTION_DIM:
        raise ValueError("DM05 internal action_dim must be 32")

    document = _read_json_object(model_dir / "norm_stats.json", label="norm stats")
    stats = document.get("norm_stats")
    if not isinstance(stats, dict):
        raise ValueError("norm_stats.json must contain an object named norm_stats")
    state = stats.get("state")
    action = stats.get("action")
    if not isinstance(state, dict) or not isinstance(action, dict):
        raise ValueError("norm_stats.json must contain state and action profiles")

    def quantiles(value: Mapping[str, Any], width: int, label: str):
        lo = np.asarray(value.get("q01"), dtype=np.float32)
        hi = np.asarray(value.get("q99"), dtype=np.float32)
        if lo.shape != (width,) or hi.shape != (width,):
            raise ValueError(f"{label} q01/q99 must have width {width}")
        if not np.isfinite(lo).all() or not np.isfinite(hi).all():
            raise ValueError(f"{label} q01/q99 must be finite")
        if np.any(hi < lo):
            raise ValueError(f"{label} q99 must not be below q01")
        return lo, hi

    quantiles(state, STATE_DIM, "state")
    return quantiles(action, ACTION_DIM, "action")


def _opendm_source_root() -> Path:
    spec = importlib.util.find_spec("opendm")
    if spec is None or spec.origin is None:
        raise RuntimeError(
            f"OpenDM is not importable; put dexmal/opendm@{OPENDM_COMMIT[:8]} "
            "on PYTHONPATH"
        )
    root = Path(spec.origin).resolve().parent.parent
    git = ["git", "-c", f"safe.directory={root}", "-C", str(root)]
    try:
        commit = subprocess.run(
            [*git, "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = subprocess.run(
            [*git, "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("OpenDM source must be an intact Git checkout") from exc
    if commit != OPENDM_COMMIT:
        raise RuntimeError(f"OpenDM commit mismatch: {commit} != {OPENDM_COMMIT}")
    if dirty:
        raise RuntimeError("OpenDM source checkout must be clean")
    return root


def _cpu_array(value: Any, *, label: str) -> np.ndarray:
    """Return a CPU tensor as an ndarray without importing PyTorch here."""

    device = getattr(value, "device", None)
    device_type = getattr(device, "type", None)
    if device is not None and device_type not in (None, "cpu"):
        device_name = str(device)
        if device_name != "cpu":
            raise RuntimeError(
                f"DM05 MASK_KEY_564 {label} must be computed on CPU, got "
                f"{device_name}"
            )
    detached = value.detach() if callable(getattr(value, "detach", None)) else value
    cpu_value = (
        detached.cpu() if callable(getattr(detached, "cpu", None)) else detached
    )
    array = np.asarray(cpu_value)
    if array.dtype.hasobject:
        raise RuntimeError(f"DM05 MASK_KEY_564 {label} has unsupported object dtype")
    return array


def _array_contract(value: Any, *, label: str) -> dict[str, Any]:
    array = _cpu_array(value, label=label)
    stride = getattr(value, "stride", None)
    if callable(stride):
        strides = [int(item) for item in stride()]
    elif array.ndim:
        strides = [int(item // array.itemsize) for item in array.strides]
    else:
        strides = []
    return {
        "shape": [int(item) for item in array.shape],
        "stride": strides,
        "dtype": str(getattr(value, "dtype", array.dtype)),
    }


def _array_fingerprint(value: Any, *, label: str) -> dict[str, Any]:
    array = _cpu_array(value, label=label)
    return {
        **_array_contract(value, label=label),
        "sha256": hashlib.sha256(
            np.ascontiguousarray(array).tobytes(order="C")
        ).hexdigest(),
    }


def _mask_layout_key_sha256(
    values: Mapping[str, Any],
    *,
    image_token_id: int,
    text_config: Any,
    padding_idx: int,
    embedding_dtype: Any,
    embedding_width: int,
    graph_device: str,
    gemma_config_class: str,
) -> str:
    """Own the opaque MASK_KEY_564 from only causal-mask dependencies.

    The processor tensors are still on CPU. Pixel values and ordinary token
    identities are deliberately absent: only history-pad and image-placeholder
    positions derived from ``input_ids`` participate in the key.
    """

    required = {"input_ids", "attention_mask", "token_type_ids"}
    missing = required - set(values)
    if missing:
        raise RuntimeError(
            f"DM05 MASK_KEY_564 input is missing {sorted(missing)}"
        )
    input_ids = _cpu_array(values["input_ids"], label="input_ids")
    attention_mask = _cpu_array(values["attention_mask"], label="attention_mask")
    token_type_ids = _cpu_array(
        values["token_type_ids"], label="token_type_ids"
    )
    if input_ids.ndim != 2 or input_ids.shape != attention_mask.shape:
        raise RuntimeError(
            "DM05 MASK_KEY_564 input_ids/attention_mask must share a 2D shape"
        )
    if token_type_ids.shape != input_ids.shape:
        raise RuntimeError(
            "DM05 MASK_KEY_564 token_type_ids must match input_ids shape"
        )

    history_pad_layout = input_ids == HISTORY_PAD_TOKEN_ID
    image_placeholder_layout = input_ids == int(image_token_id)
    effective_attention = np.array(attention_mask, copy=True)
    effective_attention[history_pad_layout] = 0
    position_ids = np.maximum(
        np.cumsum(effective_attention, axis=-1, dtype=np.int64) - 1,
        0,
    )
    prefix_len = int(input_ids.shape[1])
    cache_position = np.arange(prefix_len, dtype=np.int64)

    layer_types = [str(value) for value in text_config.layer_types]
    sliding_window = int(text_config.sliding_window)
    cache_layer_semantics = [
        {
            "layer": index,
            "layer_type": layer_type,
            "is_sliding": layer_type == "sliding_attention",
            "sliding_window": (
                sliding_window if layer_type == "sliding_attention" else 0
            ),
            "initial_prefill_mask_sizes": [prefix_len, 0],
        }
        for index, layer_type in enumerate(layer_types)
    ]
    payload = {
        "schema": "apxinf.dm05.mask-layout-key.v1",
        "batch_size": int(input_ids.shape[0]),
        "prefix_len": prefix_len,
        "processor_device": "cpu",
        "graph_device": str(graph_device),
        "input_ids_contract": _array_contract(
            values["input_ids"], label="input_ids"
        ),
        "attention_mask_input_contract": _array_contract(
            values["attention_mask"], label="attention_mask"
        ),
        "attention_mask": _array_fingerprint(
            effective_attention, label="effective_attention_mask"
        ),
        "history_pad_layout": _array_fingerprint(
            history_pad_layout, label="history_pad_layout"
        ),
        "position_ids": _array_fingerprint(position_ids, label="position_ids"),
        "cache_position": _array_fingerprint(
            cache_position, label="cache_position"
        ),
        "token_type_ids": _array_fingerprint(
            values["token_type_ids"], label="token_type_ids"
        ),
        "image_placeholder_layout": _array_fingerprint(
            image_placeholder_layout, label="image_placeholder_layout"
        ),
        "mask_inputs_embeds_dtype": str(embedding_dtype),
        "mask_inputs_embeds_device": str(graph_device),
        "mask_inputs_embeds_shape": [
            int(input_ids.shape[0]),
            prefix_len,
            int(embedding_width),
        ],
        "layer_types": layer_types,
        "sliding_window": sliding_window,
        "hidden_size": int(text_config.hidden_size),
        "image_token_id": int(image_token_id),
        "padding_idx": int(padding_idx),
        "text_attention_implementation": str(
            getattr(text_config, "_attn_implementation", None)
        ),
        "cache_layer_semantics": cache_layer_semantics,
        "cache_is_fresh": True,
        "is_first_iteration": True,
        "gemma_config_class": str(gemma_config_class),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _validate_mask_layout_key_sha256(value: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise RuntimeError(
            "DM05 MASK_KEY_564 must be a 64-character lowercase SHA-256 digest"
        )
    return value


def _set_config_attention(config) -> None:
    """Select supported backends before Transformers constructs nested models.

    OpenDM's public runtime setter runs after ``from_pretrained``.  The released
    checkpoint embeds ``flash_attention_2`` in its vision config, so the nested
    SigLIP constructor otherwise imports FlashAttention before that setter can
    honor the requested SDPA backend.
    """

    assignments = (
        (config.vlm_config, "eager"),
        (config.vlm_config.text_config, "eager"),
        (config.vlm_config.vision_config, "sdpa"),
        (config.action_config, "sdpa"),
    )
    for target, implementation in assignments:
        target._attn_implementation_internal = implementation

_COMBINED_SCHEMA = "apxinf.dm05.exact-combined.v1"
_COMBINED_RUNTIME_VERSIONS = {
    "torch": "2.11.0+cu130",
    "torch_cuda": "13.0",
    "triton": "3.6.0",
}
_COMBINED_FIXED_CELL = {
    "batch_size": 1,
    "prefix_length": 564,
    "suffix_length": 10,
    "hidden_size": 1024,
    "model_action_dim": 32,
    "layer_count": 34,
    "query_heads": 8,
    "kv_heads": 4,
    "head_dim": 256,
    "device_capability": [8, 9],
    "dtype": "torch.bfloat16",
    "attention_backend": "sdpa",
    "torch_version": "2.11.0+cu130",
    "triton_version": "3.6.0",
}
_COMBINED_CAPTURE_CENSUS = {
    "suffix_mask_calls": 10,
    "suffix_mask_builds": 1,
    "suffix_position_calls": 10,
    "suffix_position_builds": 1,
    "bool_mask_calls": 340,
    "bool_mask_builds": 1,
    "rope_calls": 340,
    "sliding_rope_builds": 1,
    "full_rope_builds": 1,
    "apply_rotary_calls": 340,
    "semantic_cat_calls": 680,
    "direct_pack_out_builds": 680,
    "semantic_repeat_calls": 680,
    "identity_repeat_returns": 680,
    "sdpa_calls": 340,
    "post_sdpa_contiguous_builds": 340,
    "time_cond_calls": 10,
    "adaptive_rmsnorm_calls": 690,
    "table_lookup_calls": 690,
    "native_modulator_linear_calls": 0,
    "native_var_calls": 690,
    "native_rsqrt_calls": 690,
    "exact_affine_kernel_calls": 690,
    "mlp_calls": 340,
    "fallback_count": 0,
}
_COMBINED_READY_FIELDS = {
    "schema": _COMBINED_SCHEMA,
    "execution_backend": EXECUTION_BACKEND_COMBINED,
    "initialized": True,
    "combined_ready": True,
    "combined_capture_census_exact": True,
    "combined_patches_restored": True,
    "dm05_arch_source_sha256": (
        "b5ab170374fbc965aa86d7d370075e8c8bc21bcf46bc6de34e7e336df1af9ce8"
    ),
    "prefix_startup_capture_count": 1,
    "suffix_startup_capture_count": 1,
    "prefix_capture_execution_count": 1,
    "suffix_capture_execution_count": 1,
    "mask_helper_build_count": 2,
    "mask_mapping_keys": ["full_attention", "sliding_attention"],
    "mask_static_address_verified": True,
    "mask_immutable_verified": True,
    "modulation_table_build_count": 1,
    "modulation_table_entry_count": 690,
    "modulation_table_rng_unchanged": True,
    "modulation_table_addresses_stable": True,
    "modulation_table_immutable": True,
    "metadata_owner_tensor_count": 7,
    "metadata_owner_addresses_stable": True,
    "replay_content_baseline_established": True,
    "metadata_owner_second_replay_exact": True,
    "metadata_owner_bytes_are_request_dynamic": True,
    "pack_workspace_count": 2,
    "pack_workspace_addresses_stable": True,
    "pack_workspace_second_replay_exact": True,
    "affine_output_count": 690,
    "affine_output_addresses_stable": True,
    "affine_output_second_replay_exact": True,
    "exact_affine_compile_count": 1,
    "exact_affine_fallback_count": 0,
    "exact_affine_ptx_codegen_verified": True,
    "sass_external_receipt_required": True,
    "source_candidate_gpu_validation_required": True,
    "startup_native_suffix_reference_count": 1,
    "startup_graph_replay_count": 4,
    "startup_first_replay_output_bitwise_exact": True,
    "startup_second_replay_output_bitwise_exact": True,
    "startup_changed_noise_control_count": 1,
    "startup_changed_noise_graph_vs_eager_bitwise_exact": True,
    "startup_changed_noise_differs_from_zero_baseline": True,
    "startup_static_zero_input_restored": True,
    "startup_static_zero_repeat_bitwise_exact": True,
    "startup_output_poison_count": 4,
    "startup_native_reference_bitwise": True,
    "request_prefix_length": EXACT_PREFIX_LEN,
    "selected_prefix_length": EXACT_PREFIX_LEN,
    "cache_layer_count": 34,
    "mask_layout_key_verified": True,
    "prefix_static_cache_address_verified": True,
    "suffix_static_output_address_verified": True,
    "closed": False,
}
_COMBINED_ZERO_FIELDS = (
    "prefix_eager_count",
    "post_prefix_cache_stage_requests",
    "post_prefix_cache_tensor_copies",
    "fallback_count",
    "history_count",
    "result_reuse_count",
)
_COMBINED_REQUEST_DELTAS = {
    "prefix_input_stage_requests": 1,
    "prefix_input_tensor_copies": 4,
    "prefix_graph_replay_count": 1,
    "prefix_graph_cache_write_tensor_copies": 68,
    "eager_noise_count": 1,
    "suffix_input_stage_requests": 1,
    "suffix_input_tensor_copies": 2,
    "suffix_graph_replay_count": 1,
}
_COMBINED_MECHANISMS = [
    "static_mask_prefix_suffix_graph",
    "modulation_table_690",
    "suffix_metadata_first_owner",
    "two_expanded_kv_pack_workspaces",
    "exact_postreduce_affine_triton",
]
_COMBINED_PTX_CHECKS = {
    "device_sm89": True,
    "ptx_target_sm89": True,
    "compiled_entry_selected": True,
    "ptx_entry_selected": True,
    "two_scalar_fmul_rn_f32": True,
    "two_scalar_fadd_rn_f32": True,
    "bf16_rne_store": True,
    "forbidden_codegen_absent": True,
}
_COMBINED_PTX_FORBIDDEN_HITS = {
    "fma_f32": 0,
    "mad_f32": 0,
    "packed_f32x2": 0,
    "local_declaration": 0,
    "local_load": 0,
    "local_store": 0,
}
_COMBINED_PTX_CODEGEN_FIELDS = (
    "level",
    "ptx_sha256",
    "cubin_sha256",
    "scalar_fmul_rn_f32_count",
    "scalar_fadd_rn_f32_count",
    "bf16_rne_conversion_count",
    "forbidden_hits",
    "checks",
    "ptx_codegen_verified",
    "sass_external_receipt_required",
)
_COMBINED_PROOF_TOP_LEVEL_FIELDS = (
    "schema",
    "selector",
    "execution_backend",
    "arithmetic_backend",
    "graph_scope",
    "profile_prefix_lengths",
    "initialized",
    "initialization_ms",
    "mask_owner_symbol",
    "mask_modeling_source_sha256",
    "mask_utils_source_sha256",
    "dm05_arch_source_sha256",
    "mask_layout_key_verified",
    "mask_helper_build_count",
    "mask_mapping_keys",
    "mask_static_address_verified",
    "mask_immutable_verified",
    "prefix_startup_capture_count",
    "suffix_startup_capture_count",
    "prefix_capture_execution_count",
    "suffix_capture_execution_count",
    "prefix_input_stage_requests",
    "prefix_input_tensor_copies",
    "prefix_graph_replay_count",
    "prefix_graph_cache_write_tensor_copies",
    "eager_noise_count",
    "suffix_input_stage_requests",
    "suffix_input_tensor_copies",
    "suffix_graph_replay_count",
    "prefix_eager_count",
    "post_prefix_cache_stage_requests",
    "post_prefix_cache_tensor_copies",
    "fallback_count",
    "history_count",
    "result_reuse_count",
    "request_prefix_length",
    "selected_prefix_length",
    "cache_layer_count",
    "prefix_static_cache_address_verified",
    "suffix_static_output_address_verified",
    "closed",
    "combined_ready",
    "no_fallback",
    "combined_mechanisms",
    "fixed_cell",
    "runtime_versions",
    "combined_capture_census",
    "combined_expected_census",
    "combined_capture_census_exact",
    "combined_patches_restored",
    "modulation_table_build_count",
    "modulation_table_entry_count",
    "modulation_table_rng_unchanged",
    "modulation_table_addresses_stable",
    "modulation_table_immutable",
    "metadata_owner_tensor_count",
    "metadata_owner_addresses_stable",
    "replay_content_baseline_established",
    "metadata_owner_second_replay_exact",
    "metadata_owner_bytes_are_request_dynamic",
    "pack_workspace_count",
    "pack_workspace_addresses_stable",
    "pack_workspace_second_replay_exact",
    "affine_output_count",
    "affine_output_addresses_stable",
    "affine_output_second_replay_exact",
    "exact_affine_compile_count",
    "exact_affine_fallback_count",
    "exact_affine_ptx_codegen_verified",
    "exact_affine_ptx_sha256",
    "exact_affine_cubin_sha256",
    "sass_external_receipt_required",
    "exact_affine_ptx_codegen",
    "startup_native_suffix_reference_count",
    "startup_graph_replay_count",
    "startup_first_replay_output_bitwise_exact",
    "startup_second_replay_output_bitwise_exact",
    "startup_changed_noise_control_count",
    "startup_changed_noise_graph_vs_eager_bitwise_exact",
    "startup_changed_noise_differs_from_zero_baseline",
    "startup_static_zero_input_restored",
    "startup_static_zero_repeat_bitwise_exact",
    "startup_output_poison_count",
    "startup_native_reference_bitwise",
    "source_candidate_gpu_validation_required",
)
_COMBINED_PROOF_TOP_LEVEL_KEYS = frozenset(
    _COMBINED_PROOF_TOP_LEVEL_FIELDS
)
# The runtime publishes one stable envelope in both lifecycle states. Values
# differ, but neither pre-initialization health nor ready request proofs may add
# fields outside this closed schema.
_COMBINED_PREINIT_TOP_LEVEL_KEYS = _COMBINED_PROOF_TOP_LEVEL_KEYS
_COMBINED_READY_TOP_LEVEL_KEYS = _COMBINED_PROOF_TOP_LEVEL_KEYS
_COMBINED_COMMON_FIELDS = {
    "schema": _COMBINED_SCHEMA,
    "selector": EXECUTION_BACKEND_COMBINED,
    "execution_backend": EXECUTION_BACKEND_COMBINED,
    "arithmetic_backend": "native_sdpa_plus_exact_postreduce_triton",
    "graph_scope": "native_prefix_564_plus_combined_suffix_10step",
    "profile_prefix_lengths": [EXACT_PREFIX_LEN],
    "mask_owner_symbol": (
        "transformers.models.gemma3.modeling_gemma3."
        "create_causal_mask_mapping"
    ),
    "mask_modeling_source_sha256": (
        "a1115edf9e0c4a3b53657f21e2de5de0a99488767d84181db6e05e082adb4f69"
    ),
    "mask_utils_source_sha256": (
        "c3c82f7b7b6e03d3f04ba6c6c58a3dd6910623636452ec67ef70e3eb522f9fe7"
    ),
    "dm05_arch_source_sha256": (
        "b5ab170374fbc965aa86d7d370075e8c8bc21bcf46bc6de34e7e336df1af9ce8"
    ),
    "cache_layer_count": 34,
    "closed": False,
    "no_fallback": True,
    "metadata_owner_bytes_are_request_dynamic": True,
    "sass_external_receipt_required": True,
    "source_candidate_gpu_validation_required": True,
}
_COMBINED_PREINIT_FIELDS = {
    "initialized": False,
    "initialization_ms": None,
    "mask_layout_key_verified": False,
    "mask_helper_build_count": 0,
    "mask_mapping_keys": [],
    "mask_static_address_verified": False,
    "mask_immutable_verified": False,
    "prefix_startup_capture_count": 0,
    "suffix_startup_capture_count": 0,
    "prefix_capture_execution_count": 0,
    "suffix_capture_execution_count": 0,
    "prefix_input_stage_requests": 0,
    "prefix_input_tensor_copies": 0,
    "prefix_graph_replay_count": 0,
    "prefix_graph_cache_write_tensor_copies": 0,
    "eager_noise_count": 0,
    "suffix_input_stage_requests": 0,
    "suffix_input_tensor_copies": 0,
    "suffix_graph_replay_count": 0,
    "prefix_eager_count": 0,
    "post_prefix_cache_stage_requests": 0,
    "post_prefix_cache_tensor_copies": 0,
    "fallback_count": 0,
    "history_count": 0,
    "result_reuse_count": 0,
    "request_prefix_length": None,
    "selected_prefix_length": None,
    "prefix_static_cache_address_verified": False,
    "suffix_static_output_address_verified": False,
    "combined_ready": False,
    "combined_capture_census_exact": False,
    "combined_patches_restored": True,
    "modulation_table_build_count": 0,
    "modulation_table_entry_count": 0,
    "modulation_table_rng_unchanged": False,
    "modulation_table_addresses_stable": False,
    "modulation_table_immutable": False,
    "metadata_owner_tensor_count": 0,
    "metadata_owner_addresses_stable": False,
    "replay_content_baseline_established": False,
    "metadata_owner_second_replay_exact": False,
    "pack_workspace_count": 0,
    "pack_workspace_addresses_stable": False,
    "pack_workspace_second_replay_exact": False,
    "affine_output_count": 0,
    "affine_output_addresses_stable": False,
    "affine_output_second_replay_exact": False,
    "exact_affine_compile_count": 0,
    "exact_affine_fallback_count": 0,
    "exact_affine_ptx_codegen_verified": False,
    "exact_affine_ptx_sha256": None,
    "exact_affine_cubin_sha256": None,
    "exact_affine_ptx_codegen": None,
    "startup_native_suffix_reference_count": 0,
    "startup_graph_replay_count": 0,
    "startup_first_replay_output_bitwise_exact": False,
    "startup_second_replay_output_bitwise_exact": False,
    "startup_changed_noise_control_count": 0,
    "startup_changed_noise_graph_vs_eager_bitwise_exact": False,
    "startup_changed_noise_differs_from_zero_baseline": False,
    "startup_static_zero_input_restored": False,
    "startup_static_zero_repeat_bitwise_exact": False,
    "startup_output_poison_count": 0,
    "startup_native_reference_bitwise": False,
}


def _copy_path_proof(value: Any) -> dict[str, Any]:
    """Copy one runtime proof while rejecting process-local addresses."""

    def visit(item: Any, path: str) -> Any:
        if isinstance(item, Mapping):
            result: dict[str, Any] = {}
            for key, child in item.items():
                if not isinstance(key, str):
                    raise RuntimeError(f"DM05 path proof key at {path} must be a string")
                lowered = key.lower()
                address_attestation = lowered.endswith(
                    (
                        "_address_verified",
                        "_addresses_verified",
                        "_address_stable",
                        "_addresses_stable",
                    )
                ) and isinstance(child, bool)
                if (
                    "data_ptr" in lowered
                    or "storage_ptr" in lowered
                    or "pointer" in lowered
                    or lowered.endswith("_ptr")
                    or ("address" in lowered and not address_attestation)
                ):
                    raise RuntimeError(
                        f"DM05 path proof must not expose process-local field {key!r}"
                    )
                result[key] = visit(child, f"{path}.{key}")
            return result
        if isinstance(item, (list, tuple)):
            return [visit(child, f"{path}[]") for child in item]
        if item is None or isinstance(item, (bool, int, str)):
            return item
        if isinstance(item, float):
            if not math.isfinite(item):
                raise RuntimeError(f"DM05 path proof contains non-finite value at {path}")
            return item
        raise RuntimeError(
            f"DM05 path proof contains unsupported {type(item).__name__} at {path}"
        )

    proof = visit(value, "path_proof")
    if not isinstance(proof, dict):
        raise RuntimeError("DM05 combined runtime proof_snapshot() must return an object")
    return proof


def _require_exact_keys(
    value: Any, expected: frozenset[str], *, label: str
) -> dict[str, Any]:
    if type(value) is not dict:
        raise RuntimeError(f"DM05 {label} must be an object")
    observed = set(value)
    unknown = sorted(observed - expected)
    missing = sorted(expected - observed)
    if unknown:
        raise RuntimeError(f"DM05 {label} has unknown fields: {unknown}")
    if missing:
        raise RuntimeError(f"DM05 {label} omitted fields: {missing}")
    return value


def _require_exact_mapping(
    value: Any, expected: Mapping[str, Any], *, label: str
) -> dict[str, Any]:
    mapping = _require_exact_keys(
        value, frozenset(expected), label=label
    )
    for name, expected_value in expected.items():
        _require_proof_field(mapping, name, expected_value)
    return {name: mapping[name] for name in expected}


def _validate_and_project_ptx_codegen(value: Any) -> dict[str, Any]:
    codegen = _require_exact_keys(
        value,
        frozenset(_COMBINED_PTX_CODEGEN_FIELDS),
        label="combined exact_affine_ptx_codegen",
    )
    expected_scalars = {
        "level": "compiled_ptx",
        "scalar_fmul_rn_f32_count": 2,
        "scalar_fadd_rn_f32_count": 2,
        "bf16_rne_conversion_count": 1,
        "ptx_codegen_verified": True,
        "sass_external_receipt_required": True,
    }
    for name, expected in expected_scalars.items():
        _require_proof_field(codegen, name, expected)
    _require_sha256_field(codegen, "ptx_sha256")
    _require_sha256_field(codegen, "cubin_sha256")
    forbidden = _require_exact_mapping(
        codegen["forbidden_hits"],
        _COMBINED_PTX_FORBIDDEN_HITS,
        label="combined exact_affine_ptx_codegen.forbidden_hits",
    )
    checks = _require_exact_mapping(
        codegen["checks"],
        _COMBINED_PTX_CHECKS,
        label="combined exact_affine_ptx_codegen.checks",
    )
    projected = {name: codegen[name] for name in _COMBINED_PTX_CODEGEN_FIELDS}
    projected["forbidden_hits"] = forbidden
    projected["checks"] = checks
    return projected


def _require_proof_field(
    proof: Mapping[str, Any], name: str, expected: Any
) -> None:
    if name not in proof:
        raise RuntimeError(f"DM05 combined path proof omitted {name}")
    observed = proof[name]
    if isinstance(expected, bool):
        valid_type = type(observed) is bool
    elif isinstance(expected, int):
        valid_type = type(observed) is int
    elif isinstance(expected, str):
        valid_type = type(observed) is str
    elif isinstance(expected, dict):
        valid_type = type(observed) is dict
    elif isinstance(expected, list):
        valid_type = type(observed) is list
    else:
        valid_type = type(observed) is type(expected)
    if not valid_type or observed != expected:
        raise RuntimeError(
            f"DM05 combined path proof mismatch for {name}: "
            f"{observed!r} != {expected!r}"
        )


def _require_nonnegative_counter(proof: Mapping[str, Any], name: str) -> int:
    if name not in proof or type(proof[name]) is not int or proof[name] < 0:
        raise RuntimeError(
            f"DM05 combined path proof {name} must be a non-negative integer"
        )
    return proof[name]


def _require_sha256_field(proof: Mapping[str, Any], name: str) -> None:
    value = proof.get(name)
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise RuntimeError(
            f"DM05 combined path proof {name} must be a lowercase SHA-256"
        )


def _validate_combined_common_proof(proof: dict[str, Any]) -> None:
    _require_exact_keys(
        proof,
        _COMBINED_PROOF_TOP_LEVEL_KEYS,
        label="combined path proof",
    )
    for name, expected in _COMBINED_COMMON_FIELDS.items():
        _require_proof_field(proof, name, expected)
    _require_exact_mapping(
        proof["fixed_cell"],
        _COMBINED_FIXED_CELL,
        label="combined fixed_cell",
    )
    _require_exact_mapping(
        proof["runtime_versions"],
        _COMBINED_RUNTIME_VERSIONS,
        label="combined runtime_versions",
    )
    _require_proof_field(
        proof, "combined_mechanisms", _COMBINED_MECHANISMS
    )
    _require_exact_mapping(
        proof["combined_expected_census"],
        _COMBINED_CAPTURE_CENSUS,
        label="combined expected census",
    )


def _project_combined_proof(
    proof: dict[str, Any], *, ready: bool
) -> dict[str, Any]:
    projected = {
        name: proof[name] for name in _COMBINED_PROOF_TOP_LEVEL_FIELDS
    }
    projected["combined_mechanisms"] = list(_COMBINED_MECHANISMS)
    projected["fixed_cell"] = {
        name: proof["fixed_cell"][name] for name in _COMBINED_FIXED_CELL
    }
    projected["runtime_versions"] = {
        name: proof["runtime_versions"][name]
        for name in _COMBINED_RUNTIME_VERSIONS
    }
    projected["combined_expected_census"] = {
        name: proof["combined_expected_census"][name]
        for name in _COMBINED_CAPTURE_CENSUS
    }
    projected["combined_capture_census"] = (
        {
            name: proof["combined_capture_census"][name]
            for name in _COMBINED_CAPTURE_CENSUS
        }
        if ready
        else {}
    )
    projected["exact_affine_ptx_codegen"] = (
        _validate_and_project_ptx_codegen(proof["exact_affine_ptx_codegen"])
        if ready
        else None
    )
    return projected


def _validate_combined_ready_proof(proof: dict[str, Any]) -> dict[str, Any]:
    for name, expected in _COMBINED_READY_FIELDS.items():
        _require_proof_field(proof, name, expected)
    initialization_ms = proof["initialization_ms"]
    if (
        type(initialization_ms) is not float
        or not math.isfinite(initialization_ms)
        or initialization_ms <= 0.0
    ):
        raise RuntimeError(
            "DM05 combined path proof initialization_ms must be positive and finite"
        )
    _require_exact_mapping(
        proof["combined_capture_census"],
        _COMBINED_CAPTURE_CENSUS,
        label="combined capture census",
    )
    for name in _COMBINED_ZERO_FIELDS:
        _require_proof_field(proof, name, 0)
    for name in _COMBINED_REQUEST_DELTAS:
        _require_nonnegative_counter(proof, name)
    _require_sha256_field(proof, "exact_affine_ptx_sha256")
    _require_sha256_field(proof, "exact_affine_cubin_sha256")
    codegen = _validate_and_project_ptx_codegen(
        proof["exact_affine_ptx_codegen"]
    )
    if codegen["ptx_sha256"] != proof["exact_affine_ptx_sha256"]:
        raise RuntimeError("DM05 combined PTX SHA-256 projection mismatch")
    if codegen["cubin_sha256"] != proof["exact_affine_cubin_sha256"]:
        raise RuntimeError("DM05 combined cubin SHA-256 projection mismatch")
    return proof


def _validated_path_proof(
    value: Any, *, require_ready: bool = False
) -> dict[str, Any]:
    proof = _copy_path_proof(value)
    _validate_combined_common_proof(proof)
    initialized = proof.get("initialized")
    if require_ready or initialized is True:
        _validate_combined_ready_proof(proof)
        return _project_combined_proof(proof, ready=True)
    if initialized is not False:
        raise RuntimeError(
            "DM05 combined path proof initialized must be a strict boolean"
        )
    for name, expected in _COMBINED_PREINIT_FIELDS.items():
        _require_proof_field(proof, name, expected)
    _require_exact_keys(
        proof["combined_capture_census"],
        frozenset(),
        label="combined preinit capture census",
    )
    return _project_combined_proof(proof, ready=False)


def _validate_combined_request_transition(
    before_value: Any, after_value: Any
) -> dict[str, Any]:
    before = _validated_path_proof(before_value)
    after = _validated_path_proof(after_value, require_ready=True)
    for name, expected_delta in _COMBINED_REQUEST_DELTAS.items():
        before_value = _require_nonnegative_counter(before, name)
        after_value = _require_nonnegative_counter(after, name)
        observed_delta = after_value - before_value
        if observed_delta != expected_delta:
            raise RuntimeError(
                f"DM05 combined request counter {name} advanced by "
                f"{observed_delta}, expected {expected_delta}"
            )
    return after


class OpenDMBackend:
    """Pinned OpenDM execution behind an explicit, injectable runtime factory."""

    def __init__(
        self,
        model_dir: Path,
        *,
        device: str = "cuda:0",
        execution_backend: str = EXECUTION_BACKEND_DEFAULT,
        runtime_factory: RuntimeFactory | None = None,
    ) -> None:
        execution_backend = _validate_execution_backend(execution_backend)
        action_lo, action_hi = _validate_checkpoint(model_dir)
        _opendm_source_root()

        # Heavy imports stay after checkpoint/source identity verification.
        import cv2
        import torch
        from transformers import AutoProcessor
        from opendm.model.dm05.dm05_arch import (
            DM05Config,
            DM05ForConditionalGeneration,
        )

        _validate_host_thread_policy(torch, execution_backend)
        self._torch = torch
        self._cv2 = cv2
        self._device = torch.device(device)
        if self._device.type != "cuda":
            raise ValueError("DM05 deployment requires a CUDA device")

        config = DM05Config.from_pretrained(str(model_dir), local_files_only=True)
        config.chunk_size = ACTION_HORIZON
        _set_config_attention(config)
        model = DM05ForConditionalGeneration.from_pretrained(
            str(model_dir),
            config=config,
            torch_dtype=torch.bfloat16,
            local_files_only=True,
        )
        model.set_attention_implementation(
            llm_attn_implementation="eager",
            vision_attn_implementation="sdpa",
            action_attn_implementation="sdpa",
            bf16=True,
        )
        model.eval()
        model.to(self._device)
        devices = {parameter.device.type for parameter in model.parameters()}
        if devices != {"cuda"}:
            raise RuntimeError(f"DM05 hidden CPU/disk offload is unsupported: {devices}")

        runtime = None
        if runtime_factory is not None or _uses_combined_runtime(execution_backend):
            factory = runtime_factory or _load_default_runtime_factory()
            runtime = factory(
                model,
                selector=execution_backend,
                request_prefix_len=EXACT_PREFIX_LEN,
                diffusion_steps=DIFFUSION_STEPS,
            )
        if execution_backend == EXECUTION_BACKEND_DEFAULT:
            if runtime is not None:
                raise RuntimeError("OpenDM default runtime factory must return None")
        elif runtime is None:
            raise RuntimeError("OpenDM combined runtime factory returned None")

        self._model = model
        self._runtime = runtime
        self._execution_backend = execution_backend
        self._inference_lock = threading.Lock()
        self._last_path_proof: dict[str, Any] | None = None
        self._mask_layout_key_config: dict[str, Any] | None = None
        if _uses_combined_runtime(execution_backend):
            vlm_model = model.model.vlm.model
            text_config = vlm_model.config.get_text_config()
            embedding_weight = vlm_model.get_input_embeddings().weight
            self._mask_layout_key_config = {
                "image_token_id": int(vlm_model.config.image_token_id),
                "text_config": text_config,
                "padding_idx": int(vlm_model.language_model.padding_idx),
                "embedding_dtype": embedding_weight.dtype,
                "embedding_width": int(embedding_weight.shape[1]),
                "graph_device": str(self._device),
                "gemma_config_class": type(vlm_model.config).__name__,
            }

        self._processor = AutoProcessor.from_pretrained(
            str(model_dir), trust_remote_code=True, local_files_only=True
        )
        self._action_lo = action_lo
        self._action_hi = action_hi
        self.metadata = {
            "backend": "opendm",
            "execution_backend": execution_backend,
            "runtime_selector": execution_backend,
            "opendm_commit": OPENDM_COMMIT,
            "device": str(self._device),
            "precision": "bf16",
            "llm_attention": "eager",
            "vision_attention": "sdpa",
            "action_attention": "sdpa",
            "model_revision": MODEL_REVISION,
            "process_inference_policy": "serialized_all_dm05",
        }
        if _uses_combined_runtime(execution_backend):
            self.metadata.update(
                {
                    "host_thread_policy": HOST_THREAD_POLICY,
                    "torch_intraop_threads": HOST_TORCH_INTRAOP_THREADS,
                }
            )

    def _image(self, image: Image.Image) -> Image.Image:
        array = np.asarray(image.convert("RGB"))
        height, width = array.shape[:2]
        size = max(height, width)
        top = (size - height) // 2
        bottom = size - height - top
        left = (size - width) // 2
        right = size - width - left
        array = self._cv2.copyMakeBorder(
            array,
            top,
            bottom,
            left,
            right,
            self._cv2.BORDER_CONSTANT,
            value=(0, 0, 0),
        )
        array = self._cv2.resize(
            array, (IMAGE_SIZE, IMAGE_SIZE), interpolation=self._cv2.INTER_LINEAR
        )
        return Image.fromarray(array, mode="RGB")

    def _inputs(
        self, prompt: str, images: Sequence[Image.Image]
    ) -> tuple[dict[str, Any], str | None]:
        content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": (
                    "Robot: Franka\n"
                    "Overall speed: 0.5\n"
                    f"Task: {prompt}.\n"
                    "Head image: "
                ),
            },
            {"type": "image", "image": self._image(images[0])},
            {"type": "text", "text": "Left wrist image: "},
            {"type": "image", "image": self._image(images[1])},
        ]
        values = self._processor.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        required = {"input_ids", "attention_mask", "pixel_values", "token_type_ids"}
        missing = required - set(values)
        if missing:
            raise RuntimeError(f"DM05 processor output is missing {sorted(missing)}")
        mask_layout_key = None
        if self._mask_layout_key_config is not None:
            mask_layout_key = _validate_mask_layout_key_sha256(
                _mask_layout_key_sha256(values, **self._mask_layout_key_config)
            )
        return (
            {name: values[name].to(self._device) for name in required},
            mask_layout_key,
        )

    def infer(
        self,
        *,
        prompt: str,
        state: np.ndarray,
        images: Sequence[Image.Image],
        robot_type: str,
        num_steps: int,
        seed: int,
    ) -> tuple[np.ndarray, float]:
        with _DM05_PROCESS_INFERENCE_LOCK:
            with self._inference_lock:
                return self._infer_locked(
                    prompt=prompt,
                    state=state,
                    images=images,
                    robot_type=robot_type,
                    num_steps=num_steps,
                    seed=seed,
                )

    def _infer_locked(
        self,
        *,
        prompt: str,
        state: np.ndarray,
        images: Sequence[Image.Image],
        robot_type: str,
        num_steps: int,
        seed: int,
    ) -> tuple[np.ndarray, float]:
        del state  # Official LIBERO config sets add_state=False.
        if robot_type != "Franka" or num_steps != DIFFUSION_STEPS:
            raise ValueError("unsupported DM05 deployment request")
        if _uses_combined_runtime(self._execution_backend) and seed != EXACT_SEED:
            raise ValueError(
                f"default_exact_combined requires sampling.seed={EXACT_SEED}"
            )

        torch = self._torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        inputs, mask_layout_key = self._inputs(prompt, images)
        prefix_length = int(inputs["input_ids"].shape[1])
        if (
            _uses_combined_runtime(self._execution_backend)
            and prefix_length != EXACT_PREFIX_LEN
        ):
            raise ValueError(
                "default_exact_combined requires processed prefix length "
                f"{EXACT_PREFIX_LEN}, got {prefix_length}"
            )

        action_mask = torch.zeros(
            1,
            1,
            MODEL_ACTION_DIM,
            device=self._device,
            dtype=self._model.model.action_in_proj.weight.dtype,
        )
        action_mask[..., :ACTION_DIM] = 1.0
        kwargs = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "pixel_values": inputs["pixel_values"],
            "token_type_ids": inputs["token_type_ids"],
            "diffusion_steps": num_steps,
            "action_mask": action_mask,
        }

        torch.cuda.synchronize(self._device)
        started = time.perf_counter()
        self._last_path_proof = None
        before_proof = None
        if self._runtime is not None:
            before_proof = _validated_path_proof(self._runtime.proof_snapshot())
        with torch.inference_mode():
            if self._runtime is None:
                actions = self._model.inference_action(**kwargs)
            else:
                if mask_layout_key is None:
                    raise RuntimeError(
                        "default_exact_combined omitted the causal-mask layout key"
                    )
                actions = self._runtime.inference_action(
                    **kwargs, mask_layout_key=mask_layout_key
                )
                self._last_path_proof = _validate_combined_request_transition(
                    before_proof,
                    self._runtime.proof_snapshot(),
                )
        torch.cuda.synchronize(self._device)
        model_ms = (time.perf_counter() - started) * 1000.0

        raw = actions.detach().to(torch.float32).cpu().numpy()[0, :, :ACTION_DIM]
        result = ((raw + 1.0) / 2.0 * (self._action_hi - self._action_lo + 1e-6))
        result = result + self._action_lo
        return np.asarray(result, dtype=np.float32), model_ms

    def consume_path_proof_snapshot(self) -> dict[str, Any] | None:
        with self._inference_lock:
            value = self._last_path_proof
            self._last_path_proof = None
            return None if value is None else dict(value)

    def path_proof_snapshot(self) -> dict[str, Any] | None:
        with self._inference_lock:
            if self._runtime is None:
                return None
            return _validated_path_proof(self._runtime.proof_snapshot())

    def close(self) -> None:
        torch = self._torch
        with self._inference_lock:
            if self._runtime is not None:
                self._runtime.close()
            self._runtime = None
            self._mask_layout_key_config = None
            self._last_path_proof = None
            self._model = None
            self._processor = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _number_list(value: Any, *, width: int, label: str) -> np.ndarray:
    if not isinstance(value, (list, tuple)) or len(value) != width:
        raise ValueError(f"{label} must contain exactly {width} numbers")
    try:
        result = np.asarray(value, dtype=np.float32)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(
            f"{label} must contain exactly {width} finite numbers"
        ) from exc
    if result.shape != (width,) or not np.isfinite(result).all():
        raise ValueError(f"{label} must contain exactly {width} finite numbers")
    return result


def _decode_image(value: Any, *, label: str) -> Image.Image:
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, np.ndarray):
        array = np.asarray(value)
        if array.ndim != 3 or array.shape[2] not in (3, 4):
            raise ValueError(f"{label} numpy image must be HWC RGB/RGBA")
        if array.dtype != np.uint8:
            raise ValueError(f"{label} numpy image must have dtype uint8")
        return Image.fromarray(array[..., :3], mode="RGB")
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a base64 string, PIL image, or uint8 HWC")
    payload = value.split(",", 1)[1] if value.startswith("data:") and "," in value else value
    try:
        raw = base64.b64decode(payload, validate=True)
        return Image.open(BytesIO(raw)).convert("RGB")
    except (binascii.Error, OSError, UnidentifiedImageError, ValueError) as exc:
        raise ValueError(f"{label} is not a valid base64 image") from exc


@register_policy("dm05")
class Dm05Policy:
    """ApxInf L2 policy for the fixed DM05-libero deployment contract."""

    def __init__(
        self,
        backend: Dm05Backend,
        *,
        default_seed: int = EXACT_SEED,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.backend = backend
        self._infer_lock = threading.Lock()
        self.default_seed = int(default_seed)
        backend_metadata = dict(getattr(backend, "metadata", {}))
        base_metadata = {
            "schema": "apxinf.dm05.libero.policy.v1",
            "model_type": "dm05",
            "action_horizon": ACTION_HORIZON,
            "action_dim": ACTION_DIM,
            "model_action_dim": MODEL_ACTION_DIM,
            "state_dim": STATE_DIM,
            "image_size": [IMAGE_SIZE, IMAGE_SIZE],
            "image_prompts": list(IMAGE_PROMPTS),
            "robot_type": "Franka",
            "diffusion_steps": DIFFUSION_STEPS,
            "concurrency": 1,
        }
        conflicting_contract = sorted(
            key
            for key in set(base_metadata) & set(backend_metadata)
            if backend_metadata[key] != base_metadata[key]
        )
        if conflicting_contract:
            raise RuntimeError(
                "DM05 backend metadata conflicts with the policy contract: "
                f"{conflicting_contract}"
            )

        model_revision = backend_metadata.pop("model_revision", None)
        if model_revision is not None and model_revision != MODEL_REVISION:
            raise RuntimeError(
                "DM05 backend model revision mismatch: "
                f"{model_revision!r} != {MODEL_REVISION!r}"
            )
        backend_precision = backend_metadata.pop("precision", None)
        if backend_precision is not None and backend_precision != "bf16":
            raise RuntimeError(
                "DM05 backend precision mismatch: "
                f"{backend_precision!r} != 'bf16'"
            )
        self.execution_backend = _validate_execution_backend(
            backend_metadata.get("execution_backend", EXECUTION_BACKEND_DEFAULT)
        )
        if (
            _uses_combined_runtime(self.execution_backend)
            and self.default_seed != EXACT_SEED
        ):
            raise ValueError(
                f"default_exact_combined requires default_seed={EXACT_SEED}"
            )
        if _uses_combined_runtime(self.execution_backend):
            expected = {
                "runtime_selector": EXECUTION_BACKEND_COMBINED,
                "host_thread_policy": HOST_THREAD_POLICY,
                "torch_intraop_threads": HOST_TORCH_INTRAOP_THREADS,
                "process_inference_policy": "serialized_all_dm05",
            }
            for field, value in expected.items():
                if backend_metadata.get(field) != value:
                    raise RuntimeError(
                        f"default_exact_combined backend metadata mismatch for {field}: "
                        f"{backend_metadata.get(field)!r} != {value!r}"
                    )
        elif {
            "host_thread_policy",
            "torch_intraop_threads",
        } & set(backend_metadata):
            raise RuntimeError("default DM05 backend returned combined host metadata")

        self.metadata = {
            **base_metadata,
            **({"model_revision": model_revision} if model_revision else {}),
            **({"precision": backend_precision} if backend_precision else {}),
            **backend_metadata,
            **_validate_policy_metadata(metadata),
            "execution_backend": self.execution_backend,
            "runtime_selector": self.execution_backend,
        }

    @classmethod
    def from_pretrained(
        cls,
        model_dir,
        *,
        device: str = "cuda:0",
        precision: str = "bf16",
        default_seed: int = EXACT_SEED,
        execution_backend: str = EXECUTION_BACKEND_DEFAULT,
        metadata: Mapping[str, Any] | None = None,
        **unsupported,
    ) -> "Dm05Policy":
        if unsupported:
            raise TypeError(f"unsupported DM05 options: {sorted(unsupported)}")
        _validate_policy_metadata(metadata)
        if precision.lower() != "bf16":
            raise ValueError("DM05-libero ApxInf deployment supports BF16 only")
        execution_backend = _validate_execution_backend(execution_backend)
        if _uses_combined_runtime(execution_backend) and default_seed != EXACT_SEED:
            raise ValueError(
                f"default_exact_combined requires default_seed={EXACT_SEED}"
            )

        model_dir = Path(model_dir).expanduser().resolve()
        backend = OpenDMBackend(
            model_dir,
            device=device,
            execution_backend=execution_backend,
        )
        return cls(backend, default_seed=default_seed, metadata=metadata)

    @property
    def action_dim(self) -> int:
        return ACTION_DIM

    @property
    def action_horizon(self) -> int:
        return ACTION_HORIZON

    def infer(
        self,
        observation: Mapping[str, Any],
        *,
        noise: np.ndarray | None = None,
    ) -> dict[str, Any]:
        if noise is not None:
            raise ValueError(
                "DM05 does not accept external noise; select stochastic input "
                "with sampling.seed"
            )
        if not isinstance(observation, Mapping):
            raise ValueError("DM05 observation must be an object")
        allowed = {"prompt", "state", "images", "robot_type", "sampling"}
        extra = set(observation) - allowed
        if extra:
            raise ValueError(f"unsupported DM05 observation fields: {sorted(extra)}")
        prompt = observation.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("DM05 prompt must be a non-empty string")
        state = _number_list(observation.get("state"), width=STATE_DIM, label="state")
        if observation.get("robot_type") != "Franka":
            raise ValueError("DM05-libero robot_type must be 'Franka'")
        image_values = observation.get("images")
        if not isinstance(image_values, Mapping) or set(image_values) != {"1", "2"}:
            raise ValueError("DM05-libero images must have exactly slots '1' and '2'")
        images = [
            _decode_image(image_values[str(index)], label=f"images.{index}")
            for index in (1, 2)
        ]
        sampling = observation.get("sampling", {})
        if not isinstance(sampling, Mapping):
            raise ValueError("sampling must be an object")
        extra_sampling = set(sampling) - {"num_steps", "seed"}
        if extra_sampling:
            raise ValueError(f"unsupported sampling fields: {sorted(extra_sampling)}")
        num_steps = sampling.get("num_steps", DIFFUSION_STEPS)
        seed = sampling.get("seed", self.default_seed)
        if isinstance(num_steps, bool) or not isinstance(num_steps, int):
            raise ValueError("sampling.num_steps must be an integer")
        if num_steps != DIFFUSION_STEPS:
            raise ValueError(f"sampling.num_steps must be {DIFFUSION_STEPS}")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("sampling.seed must be an integer")
        if seed < _TORCH_SEED_MIN or seed > _TORCH_SEED_MAX:
            raise ValueError(
                "sampling.seed must be in PyTorch's inclusive range "
                f"[{_TORCH_SEED_MIN}, {_TORCH_SEED_MAX}]"
            )
        if _uses_combined_runtime(self.execution_backend) and seed != EXACT_SEED:
            raise ValueError(
                f"default_exact_combined requires sampling.seed={EXACT_SEED}"
            )

        started = time.perf_counter()
        with self._infer_lock:
            before_proof = None
            if _uses_combined_runtime(self.execution_backend):
                snapshot = getattr(self.backend, "path_proof_snapshot", None)
                if not callable(snapshot):
                    raise RuntimeError(
                        "default_exact_combined backend omitted path_proof_snapshot"
                    )
                before_proof = snapshot()
            actions, model_ms = self.backend.infer(
                prompt=prompt,
                state=state,
                images=images,
                robot_type="Franka",
                num_steps=num_steps,
                seed=seed,
            )
            consume = getattr(self.backend, "consume_path_proof_snapshot", None)
            path_proof = consume() if callable(consume) else None

        if _uses_combined_runtime(self.execution_backend):
            path_proof = _validate_combined_request_transition(
                before_proof, path_proof
            )
        elif path_proof is not None:
            raise RuntimeError("default DM05 backend returned unexpected path proof")

        actions = np.asarray(actions, dtype=np.float32)
        if actions.shape != (ACTION_HORIZON, ACTION_DIM):
            raise RuntimeError(
                f"DM05 backend returned {actions.shape}, expected "
                f"{(ACTION_HORIZON, ACTION_DIM)}"
            )
        if not np.isfinite(actions).all():
            raise RuntimeError("DM05 backend returned non-finite actions")
        model_ms = float(model_ms)
        if not math.isfinite(model_ms) or model_ms <= 0.0:
            raise RuntimeError("DM05 backend returned invalid model latency")
        total_ms = (time.perf_counter() - started) * 1000.0
        result = {
            "actions": np.ascontiguousarray(actions),
            "timing": {"model_ms": model_ms, "total_ms": total_ms},
            "sampling": {"num_steps": num_steps, "seed": seed},
        }
        if path_proof is not None:
            result["path_proof"] = path_proof
        return result

    def path_proof_snapshot(self) -> dict[str, Any] | None:
        snapshot = getattr(self.backend, "path_proof_snapshot", None)
        if not callable(snapshot):
            return None
        value = snapshot()
        if value is None:
            return None
        if not _uses_combined_runtime(self.execution_backend):
            raise RuntimeError("default DM05 backend returned unexpected path proof")
        return _validated_path_proof(value)

    def close(self) -> None:
        self.backend.close()
