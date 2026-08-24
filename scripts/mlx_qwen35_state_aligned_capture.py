#!/usr/bin/env python3
"""Pinned, read-only Qwen3.5 state-aligned capture backend.

The module does not import MLX at import time. Production dependencies and
model weights are opened only by ``Qwen35StateAlignedCaptureBackend.open_pair``
after source, runtime, and bundle custody checks pass.

This module deliberately has no in-process RSS watchdog: the real two-model
run must be wrapped by the caller's independent process/RSS supervisor so an
over-budget model process can be terminated from outside its address space.
"""

from __future__ import annotations

import gc
import hashlib
import importlib
from importlib import metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import stat
import sys
from typing import NoReturn


CAPTURE_FORMAT = "apxinf-mlx-chinese-state-aligned-capture-v1"
SOURCE_CUSTODY_FORMAT = "apxinf-direct-regular-single-link-source-custody-v1"
PROMPT_ID = "chinese-explanation"
CERTIFIED_PROMPT_TOKEN_IDS = (
    248045,
    846,
    198,
    139054,
    99986,
    98682,
    98832,
    101850,
    95761,
    116981,
    3709,
    96172,
    110334,
    115103,
    99222,
    1710,
    248046,
    198,
    248045,
    74455,
    198,
    248068,
    271,
    248069,
    271,
)
CERTIFIED_TEACHER_TOKEN_IDS = (
    101850,
    103783,
    104145,
    3709,
    100033,
    96336,
    104257,
    96875,
    332,
    98183,
    96048,
    135450,
    332,
    100546,
    102081,
    1710,
    105090,
    98868,
    125937,
    4960,
    271,
    16,
    13,
    220,
    2972,
    125318,
    100721,
    97994,
    135450,
    100745,
    96844,
    332,
    198,
    256,
    220,
    98183,
    96048,
    135450,
    109134,
    99509,
    96392,
    96098,
    114951,
    3709,
    96646,
    135450,
    100745,
    95999,
    96743,
    98931,
    123128,
    95896,
    95873,
    96392,
    96098,
    1710,
    113286,
    125318,
    96466,
    96706,
    9616,
    101617,
    96466,
    97460,
)
CERTIFIED_PROMPT_TOKEN_IDS_SHA256 = (
    "2831f3f47ee9fa92a0f819505fee7f0d86301e7a25aacce2e4a40f94bcd7dcb5"
)
CERTIFIED_TEACHER_TOKEN_IDS_SHA256 = (
    "76acdf3f223543d8c4721eb89d694373271f76478e77962ba29286fd2fc2e531"
)
CERTIFIED_INPUT_TOKEN_IDS_SHA256 = (
    "3dba5d2c579177a68559980161fb86dc94be4fc53b17b92b56be160a4bb25de2"
)
CERTIFIED_PROMPT_TOKEN_COUNT = 25
CERTIFIED_TEACHER_TOKEN_COUNT = 64
CERTIFIED_INPUT_TOKEN_COUNT = 88
CERTIFIED_RESPONSE_START = 24
CERTIFIED_PREDICTOR_COUNT = 64
PINNED_PYTHON_VERSION = "3.14.3"
PINNED_PACKAGES = {
    "huggingface-hub": "1.28.0",
    "mlx": "0.32.1",
    "mlx-lm": "0.31.3",
    "mlx-metal": "0.32.1",
    "numpy": "2.5.2",
    "safetensors": "0.8.0",
    "tokenizers": "0.22.2",
    "transformers": "5.15.1",
}
PINNED_SOURCE_SHA256 = {
    "qwen3_5.py": "f0daa30bba5cb521c8bdfa7093101a544c6a37bbba09bca582288219cb04ae3a",
    "qwen3_next.py": "3c572fe3fbb36721efab4d80d1bb6af11beb4ad1caae18deefc9fc84cbcd9b79",
    "mlx/nn/layers/base.py": (
        "ec749e1d50fd1a5e57e0aedc8e6eb13fc697e630f59333a0e24aee62a8dc7f0f"
    ),
    "generate.py": "270778ad53eaca55a8533d82e6752660fe5d2605c4aa0879b48a50a91f69345f",
}
REQUIRED_CAPTURE_CAPABILITIES = {
    "mlx_lm_version": "0.31.3",
    "model_type": "qwen3_5",
    "model_residency": "same-process-two-independent-models-v1",
    "state_alignment": "generate-step-prefill24-then-single-token-bf16-prefix-v2",
    "module_capture": "explicit-read-only-qwen35-forward-wrapper-v1",
    "module_input": (
        "same-bf16-predictor-input-excludes-prefill-per-stateless-weight-module-v2"
    ),
    "cache_state": "independent-per-model-cache-only-v1",
    "logit_margin": "production-logprob-reference-token-margin-micro-v2",
    "dynamic_weight_replacement": False,
    "dynamic_module_replacement": False,
    "weight_writes": False,
    "repeat_count": 2,
    "capture_scope": "certified-Chinese-v1-only-no-expansion",
    "prompt_token_count": CERTIFIED_PROMPT_TOKEN_COUNT,
    "prompt_token_ids_sha256": CERTIFIED_PROMPT_TOKEN_IDS_SHA256,
    "teacher_token_count": CERTIFIED_TEACHER_TOKEN_COUNT,
    "teacher_token_ids_sha256": CERTIFIED_TEACHER_TOKEN_IDS_SHA256,
    "input_token_count": CERTIFIED_INPUT_TOKEN_COUNT,
    "input_token_ids_sha256": CERTIFIED_INPUT_TOKEN_IDS_SHA256,
    "response_start": CERTIFIED_RESPONSE_START,
    "predictor_count": CERTIFIED_PREDICTOR_COUNT,
    "chunk_schedule": "prefill24-no-metrics-then-64-single-token-predictors-v1",
    "manual_exact_gate": "per-predictor-chunk-full-logits-bit-exact-v1",
    "module_error_aggregation": (
        "raw-numerator-denominator-max-across-64-predictors-v1"
    ),
    "rss_supervision": "external-process-supervisor-required-v1",
    "in_process_rss_watchdog": False,
    "pinned_public_api_audit_sha256": (
        "4869781f3226db090937d3a3d886ac04bbaee7027525386d62f1ca1706651c1d"
    ),
}
RETAINED_BF16_PATHS = (
    "language_model.model.layers.12.linear_attn.out_proj",
    "language_model.model.layers.14.linear_attn.out_proj",
    "language_model.model.layers.20.linear_attn.out_proj",
)
_AFFINE_W8_G64 = {"bits": 8, "group_size": 64, "mode": "affine"}
_HYBRID_PRESET = {
    "name": "qwen35-0.8b-affine-w8-g64-gdn-outproj-async-chat-parity-v2",
    "policy_sha256": (
        "64a2ba1741fd5a76a7e72580ce9188d1554e1488ce6504b20054bf42479eaf8f"
    ),
    "retained_bf16_paths": list(RETAINED_BF16_PATHS),
    "source_revision": "2fc06364715b967f1860aea9cf38778875588b17",
    "weight_ledger": {
        "estimated_total_parameter_bytes": 805788352,
        "output_tensor_count": 688,
        "quantized_logical_weight_count": 745603072,
        "quantized_module_count": 184,
        "quantized_module_parameter_bytes": 792203264,
        "retained_bf16_logical_weight_count": 6291456,
        "retained_bf16_module_count": 3,
        "retained_bf16_weight_bytes": 12582912,
    },
}
_FROZEN_TEXT_SCHEMA = {
    "model_type": "qwen3_5_text",
    "hidden_size": 1024,
    "intermediate_size": 3584,
    "num_hidden_layers": 24,
    "num_attention_heads": 8,
    "num_key_value_heads": 2,
    "head_dim": 256,
    "linear_num_value_heads": 16,
    "linear_num_key_heads": 16,
    "linear_key_head_dim": 128,
    "linear_value_head_dim": 128,
    "linear_conv_kernel_dim": 4,
    "full_attention_interval": 4,
    "vocab_size": 248320,
    "num_experts": 0,
    "tie_word_embeddings": True,
    "attention_bias": False,
}
_FIXED_BUNDLE_FILES = frozenset(
    {
        "README.md",
        "chat_template.jinja",
        "config.json",
        "model.safetensors.index.json",
        "tokenizer.json",
        "tokenizer_config.json",
    }
)
_MODEL_SHARD = re.compile(
    r"^model(?:\.safetensors|-[0-9]{5}-of-[0-9]{5}\.safetensors)$"
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_HASH_CHUNK_BYTES = 4 * 1024 * 1024
_MAX_BUNDLE_FILES = 64
_MAX_JSON_BYTES = 64 * 1024 * 1024
_OFFLINE_ENVIRONMENT = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "TOKENIZERS_PARALLELISM": "false",
    "NO_PROXY": "*",
    "no_proxy": "*",
}


class CaptureError(ValueError):
    """A fail-closed runtime, custody, model, or capture violation."""


def _fail(message: str) -> NoReturn:
    raise CaptureError(message)


def canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise CaptureError(f"value is not canonical JSON: {error}") from error


def object_sha256(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _stable_fields(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _canonical_directory(value: object, label: str) -> Path:
    if type(value) is not str:
        _fail(f"{label} path must be an absolute canonical string")
    path = Path(value)
    if not path.is_absolute():
        _fail(f"{label} path must be absolute")
    try:
        resolved = path.resolve(strict=True)
        observed = path.lstat()
    except OSError as error:
        raise CaptureError(f"cannot inspect {label}: {error}") from error
    if (
        resolved != path
        or stat.S_ISLNK(observed.st_mode)
        or not stat.S_ISDIR(observed.st_mode)
    ):
        _fail(f"{label} must be a canonical direct directory")
    return path


def _direct_file_stat(path: Path, label: str) -> os.stat_result:
    try:
        observed = path.lstat()
    except OSError as error:
        raise CaptureError(f"cannot inspect {label}: {error}") from error
    if (
        not stat.S_ISREG(observed.st_mode)
        or stat.S_ISLNK(observed.st_mode)
        or observed.st_nlink != 1
    ):
        _fail(f"{label} must be a direct regular file with one hard link")
    return observed


def _stream_sha256(path: Path, label: str) -> str:
    before = _direct_file_stat(path, label)
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise CaptureError(f"cannot open {label}: {error}") from error
    digest = hashlib.sha256()
    try:
        opened = os.fstat(descriptor)
        if _stable_fields(opened) != _stable_fields(before):
            _fail(f"{label} changed before hashing")
        while True:
            chunk = os.read(descriptor, _HASH_CHUNK_BYTES)
            if not chunk:
                break
            digest.update(chunk)
        finished = os.fstat(descriptor)
        if _stable_fields(opened) != _stable_fields(finished):
            _fail(f"{label} changed while hashing")
    finally:
        os.close(descriptor)
    return digest.hexdigest()


def _source_file_identity(path_value: object, label: str) -> dict[str, object]:
    if isinstance(path_value, Path):
        path = path_value
    elif type(path_value) is str:
        path = Path(path_value)
    else:
        _fail(f"{label} path must be a canonical absolute path")
    if not path.is_absolute():
        _fail(f"{label} path must be absolute")
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise CaptureError(f"cannot resolve {label}: {error}") from error
    if resolved != path:
        _fail(f"{label} must be a canonical direct path")
    before = _direct_file_stat(path, label)
    digest = _stream_sha256(path, label)
    after = _direct_file_stat(path, label)
    if _stable_fields(before) != _stable_fields(after):
        _fail(f"{label} changed while its identity was captured")
    return {"path": str(path), "size": before.st_size, "sha256": digest}


def _backend_source_custody() -> dict[str, object]:
    capture_path = Path(__file__)
    if not capture_path.is_absolute():
        _fail("capture backend __file__ is not absolute")
    loader_path = capture_path.with_name("diagnose_mlx_chinese_hybrid.py")
    return {
        "format": SOURCE_CUSTODY_FORMAT,
        "capture": _source_file_identity(capture_path, "capture backend source"),
        "loader": _source_file_identity(loader_path, "diagnostic loader source"),
    }


def _validate_backend_source_custody(value: object) -> dict[str, object]:
    if (
        type(value) is not dict
        or set(value) != {"format", "capture", "loader"}
        or value.get("format") != SOURCE_CUSTODY_FORMAT
    ):
        _fail("capture backend source custody fields drifted")
    observed = _backend_source_custody()
    if value != observed:
        _fail("capture backend or diagnostic loader source identity drifted")
    return json.loads(canonical_bytes(observed))


def _read_direct_bytes(path: Path, label: str, maximum_bytes: int) -> bytes:
    before = _direct_file_stat(path, label)
    if before.st_size > maximum_bytes:
        _fail(f"{label} exceeds the bounded read limit")
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise CaptureError(f"cannot open {label}: {error}") from error
    payload = bytearray()
    try:
        opened = os.fstat(descriptor)
        if _stable_fields(opened) != _stable_fields(before):
            _fail(f"{label} changed before reading")
        while True:
            chunk = os.read(
                descriptor,
                min(1024 * 1024, maximum_bytes + 1 - len(payload)),
            )
            if not chunk:
                break
            payload.extend(chunk)
            if len(payload) > maximum_bytes:
                _fail(f"{label} exceeds the bounded read limit")
        if _stable_fields(opened) != _stable_fields(os.fstat(descriptor)):
            _fail(f"{label} changed while reading")
    finally:
        os.close(descriptor)
    return bytes(payload)


def _reject_duplicate_keys(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            _fail(f"JSON contains duplicate key: {key}")
        value[key] = item
    return value


def _read_json(path: Path, label: str) -> dict[str, object]:
    try:
        value = json.loads(
            _read_direct_bytes(path, label, _MAX_JSON_BYTES),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda constant: _fail(
                f"{label} contains non-finite JSON number {constant}"
            ),
        )
    except CaptureError:
        raise
    except (UnicodeError, ValueError) as error:
        raise CaptureError(f"{label} is not canonical UTF-8 JSON") from error
    if type(value) is not dict:
        _fail(f"{label} root must be an object")
    return value


def _forbid_remote_code(value: object, label: str) -> None:
    if type(value) is dict:
        for key, item in value.items():
            if (
                key in {"auto_map", "model_file", "custom_pipelines"}
                and item is not None
            ):
                _fail(f"{label} requests remote/custom code through {key}")
            _forbid_remote_code(item, label)
    elif type(value) is list:
        for item in value:
            _forbid_remote_code(item, label)


def _validate_bundle_config(
    config: dict[str, object], precision_profile: str, label: str
) -> None:
    _forbid_remote_code(config, f"{label}/config.json")
    text = config.get("text_config")
    architectures = config.get("architectures")
    if (
        config.get("model_type") != "qwen3_5"
        or type(architectures) is not list
        or "Qwen3_5ForConditionalGeneration" not in architectures
        or config.get("tie_word_embeddings") is not True
        or type(text) is not dict
        or text.get("dtype") != "bfloat16"
        or text.get("tie_word_embeddings") is not True
    ):
        _fail(f"{label} is not the frozen tied Qwen3.5-0.8B schema")
    assert type(text) is dict
    for field, expected in _FROZEN_TEXT_SCHEMA.items():
        observed = text.get(field, 0 if field == "num_experts" else None)
        if observed != expected:
            _fail(f"{label} frozen text field drifted: {field}")
    quantization = config.get("quantization")
    quantization_config = config.get("quantization_config")
    selective = config.get("apxinf_selective_mixed_policy")
    hybrid = config.get("apxinf_hybrid_preset")
    if precision_profile == "bf16":
        if any(
            value is not None
            for value in (quantization, quantization_config, selective, hybrid)
        ):
            _fail(f"{label} BF16 configuration declares a quantization policy")
    elif precision_profile == "hybrid-w8-bf16-g64":
        if (
            quantization != _AFFINE_W8_G64
            or quantization_config != _AFFINE_W8_G64
            or selective is not None
            or hybrid != _HYBRID_PRESET
        ):
            _fail(f"{label} hybrid-W8/BF16 frozen preset drifted")
    else:
        _fail(f"unsupported capture precision profile: {precision_profile}")


def _validate_shard_names(names: list[str], label: str) -> None:
    if names == ["model.safetensors"]:
        return
    parsed = []
    for name in names:
        match = re.fullmatch(r"model-([0-9]{5})-of-([0-9]{5})\.safetensors", name)
        if match is None:
            _fail(f"{label} model shard name drifted")
        parsed.append((int(match.group(1)), int(match.group(2))))
    totals = {total for _, total in parsed}
    if (
        len(totals) != 1
        or next(iter(totals)) != len(parsed)
        or sorted(index for index, _ in parsed) != list(range(1, len(parsed) + 1))
    ):
        _fail(f"{label} model shard sequence is incomplete")


def _snapshot_bundle(
    path_value: object,
    label: str,
    *,
    precision_profile: str,
) -> dict[str, object]:
    path = _canonical_directory(path_value, label)
    try:
        entries = list(os.scandir(path))
    except OSError as error:
        raise CaptureError(f"cannot list {label}: {error}") from error
    if len(entries) > _MAX_BUNDLE_FILES:
        _fail(f"{label} exceeds the flat bundle file limit")
    names = {entry.name for entry in entries}
    shards = sorted(name for name in names if _MODEL_SHARD.fullmatch(name))
    if names != _FIXED_BUNDLE_FILES | set(shards) or not shards:
        _fail(f"{label} controlled flat bundle layout drifted")
    _validate_shard_names(shards, label)
    files = {}
    for name in sorted(names):
        file_path = path / name
        before = _direct_file_stat(file_path, f"{label}/{name}")
        digest = _stream_sha256(file_path, f"{label}/{name}")
        after = _direct_file_stat(file_path, f"{label}/{name}")
        if _stable_fields(before) != _stable_fields(after):
            _fail(f"{label}/{name} changed during bundle snapshot")
        files[name] = {"size": before.st_size, "sha256": digest}
    config = _read_json(path / "config.json", f"{label}/config.json")
    tokenizer_config = _read_json(
        path / "tokenizer_config.json", f"{label}/tokenizer_config.json"
    )
    _forbid_remote_code(tokenizer_config, f"{label}/tokenizer_config.json")
    _validate_bundle_config(config, precision_profile, label)
    index = _read_json(
        path / "model.safetensors.index.json",
        f"{label}/model.safetensors.index.json",
    )
    weight_map = index.get("weight_map")
    if (
        set(index) != {"metadata", "weight_map"}
        or type(index.get("metadata")) is not dict
        or type(weight_map) is not dict
        or not weight_map
        or any(
            type(value) is not str or value not in shards
            for value in weight_map.values()
        )
        or set(weight_map.values()) != set(shards)
    ):
        _fail(f"{label} model index is not bound to every controlled shard")
    return {
        "path": str(path),
        "precision_profile": precision_profile,
        "files": files,
        "file_count": len(files),
        "total_bytes": sum(item["size"] for item in files.values()),
        "manifest_sha256": object_sha256(files),
    }


def _runtime_identity() -> dict[str, object]:
    implementation = platform.python_implementation()
    version = platform.python_version()
    if implementation != "CPython" or version != PINNED_PYTHON_VERSION:
        _fail(
            f"capture requires CPython {PINNED_PYTHON_VERSION}, "
            f"observed {implementation} {version}"
        )
    packages = {}
    for name, expected in PINNED_PACKAGES.items():
        try:
            observed = metadata.version(name)
        except metadata.PackageNotFoundError as error:
            raise CaptureError(f"required package is unavailable: {name}") from error
        if observed != expected:
            _fail(f"{name} must be {expected}, observed {observed}")
        packages[name] = observed
    try:
        executable = Path(sys.executable).resolve(strict=True)
    except OSError as error:
        raise CaptureError(f"cannot resolve Python executable: {error}") from error
    return {
        "python": {
            "implementation": implementation,
            "version": version,
            "executable": str(executable),
        },
        "packages": packages,
    }


def _validate_runtime_identity(value: object) -> dict[str, object]:
    if type(value) is not dict or set(value) != {"python", "packages"}:
        _fail("runtime identity fields drifted")
    python = value.get("python")
    packages = value.get("packages")
    if (
        type(python) is not dict
        or set(python) != {"implementation", "version", "executable"}
        or python.get("implementation") != "CPython"
        or python.get("version") != PINNED_PYTHON_VERSION
        or type(python.get("executable")) is not str
        or not python["executable"]
        or packages != PINNED_PACKAGES
    ):
        _fail("runtime identity does not match the exact eight-package lock")
    return json.loads(canonical_bytes(value))


def _distribution_source_paths() -> dict[str, Path]:
    try:
        mlx_lm_distribution = metadata.distribution("mlx-lm")
        mlx_distribution = metadata.distribution("mlx")
    except metadata.PackageNotFoundError as error:
        raise CaptureError(
            f"cannot locate pinned source distribution: {error}"
        ) from error
    paths = {
        "qwen3_5.py": mlx_lm_distribution.locate_file("mlx_lm/models/qwen3_5.py"),
        "qwen3_next.py": mlx_lm_distribution.locate_file("mlx_lm/models/qwen3_next.py"),
        "mlx/nn/layers/base.py": mlx_distribution.locate_file("mlx/nn/layers/base.py"),
        "generate.py": mlx_lm_distribution.locate_file("mlx_lm/generate.py"),
    }
    observed = {}
    for name, raw_path in paths.items():
        path = Path(raw_path)
        if not path.is_absolute():
            _fail(f"pinned distribution source path is not absolute: {name}")
        try:
            resolved = path.resolve(strict=True)
        except OSError as error:
            raise CaptureError(
                f"cannot resolve pinned source {name}: {error}"
            ) from error
        if resolved != path:
            _fail(f"pinned source is not a canonical direct path: {name}")
        _direct_file_stat(path, f"pinned source {name}")
        observed[name] = path
    return observed


def _source_identity_from_paths(paths: object) -> dict[str, str]:
    if type(paths) is not dict or set(paths) != set(PINNED_SOURCE_SHA256):
        _fail("pinned distribution source path portfolio drifted")
    observed = {}
    for name in PINNED_SOURCE_SHA256:
        path = paths[name]
        if not isinstance(path, Path):
            _fail(f"pinned distribution source path is malformed: {name}")
        observed[name] = _stream_sha256(path, f"pinned source {name}")
    return observed


def _source_identity() -> dict[str, str]:
    return _source_identity_from_paths(_distribution_source_paths())


def _validate_source_identity(value: object) -> dict[str, str]:
    if value != PINNED_SOURCE_SHA256:
        _fail("installed Qwen3.5 public source identity drifted")
    return dict(PINNED_SOURCE_SHA256)


def _validate_imported_source_paths(
    imported_modules: object,
    expected_paths: object,
) -> None:
    if (
        type(imported_modules) is not dict
        or type(expected_paths) is not dict
        or set(imported_modules) != set(PINNED_SOURCE_SHA256)
        or set(expected_paths) != set(PINNED_SOURCE_SHA256)
    ):
        _fail("pinned runtime import path portfolio drifted")
    for name in PINNED_SOURCE_SHA256:
        expected = expected_paths[name]
        imported_file = getattr(imported_modules[name], "__file__", None)
        if not isinstance(expected, Path) or type(imported_file) is not str:
            _fail(f"pinned runtime import path is unavailable: {name}")
        observed = Path(imported_file)
        if not observed.is_absolute():
            _fail(f"pinned runtime import path is not absolute: {name}")
        try:
            resolved = observed.resolve(strict=True)
        except OSError as error:
            raise CaptureError(
                f"cannot resolve pinned runtime import path {name}: {error}"
            ) from error
        if resolved != observed or observed != expected:
            _fail(f"pinned runtime import path differs from distribution: {name}")
        _direct_file_stat(observed, f"imported pinned source {name}")


def _enforce_offline_environment() -> None:
    sensitive = {
        name
        for name in os.environ
        if name.upper()
        in {
            "HTTP_PROXY",
            "HTTPS_PROXY",
            "ALL_PROXY",
            "NO_PROXY",
            "HF_TOKEN",
            "HF_API_TOKEN",
            "HUGGING_FACE_HUB_TOKEN",
            "HUGGINGFACE_TOKEN",
        }
        or name.upper().endswith(("_PROXY", "_TOKEN", "_TOKEN_PATH"))
    }
    for name in sensitive:
        os.environ.pop(name, None)
    for name, value in _OFFLINE_ENVIRONMENT.items():
        os.environ[name] = value


def _load_runtime():
    source_paths = _distribution_source_paths()
    _validate_source_identity(_source_identity_from_paths(source_paths))
    try:
        import mlx.core as mx
        import mlx.nn as nn
        from mlx_lm import utils
        from mlx_lm.models import qwen3_5, qwen3_next
    except (ImportError, ModuleNotFoundError) as error:
        raise CaptureError(f"cannot import pinned MLX runtime: {error}") from error
    try:
        generate_module = importlib.import_module("mlx_lm.generate")
        mlx_base_module = importlib.import_module("mlx.nn.layers.base")
    except (ImportError, ModuleNotFoundError) as error:
        raise CaptureError(
            f"cannot import pinned MLX source modules: {error}"
        ) from error
    _validate_imported_source_paths(
        {
            "qwen3_5.py": qwen3_5,
            "qwen3_next.py": qwen3_next,
            "mlx/nn/layers/base.py": mlx_base_module,
            "generate.py": generate_module,
        },
        source_paths,
    )
    _validate_source_identity(_source_identity_from_paths(source_paths))
    required = (
        utils.load,
        qwen3_5.create_attention_mask,
        qwen3_5.create_ssm_mask,
        qwen3_5.gated_delta_update,
        qwen3_next.scaled_dot_product_attention,
        qwen3_next.swiglu,
    )
    if any(not callable(item) for item in required):
        _fail("pinned Qwen3.5 manual-forward dependencies drifted")

    class Runtime:
        def __init__(self) -> None:
            self.mx = mx
            self.nn = nn
            self.create_attention_mask = qwen3_5.create_attention_mask
            self.create_ssm_mask = qwen3_5.create_ssm_mask
            self.gated_delta_update = qwen3_5.gated_delta_update
            self.scaled_dot_product_attention = qwen3_next.scaled_dot_product_attention
            self.swiglu = qwen3_next.swiglu

        @staticmethod
        def load(path: str, *, tokenizer_config: dict[str, object], lazy: bool):
            return utils.load(path, tokenizer_config=tokenizer_config, lazy=lazy)

        @staticmethod
        def clear_cache() -> None:
            clear_cache = getattr(mx, "clear_cache", None)
            if callable(clear_cache):
                clear_cache()

    return Runtime()


def _token_ids(value: object, label: str) -> list[int]:
    if (
        type(value) is not list
        or not value
        or any(
            type(token) is not int or token < 0 or token >= 248320 for token in value
        )
    ):
        _fail(f"{label} must be a non-empty Qwen3.5 token-ID list")
    return list(value)


def build_teacher_forced_sequence(
    prompt_token_ids: object,
    teacher_token_ids: object,
) -> tuple[list[int], int]:
    """Return the one certified Chinese-v1 teacher-forced sequence."""

    prompt = _token_ids(prompt_token_ids, "prompt")
    teacher = _token_ids(teacher_token_ids, "BF16 teacher")
    sequence = prompt + teacher[:-1]
    response_start = len(prompt) - 1
    if (
        prompt != list(CERTIFIED_PROMPT_TOKEN_IDS)
        or teacher != list(CERTIFIED_TEACHER_TOKEN_IDS)
        or len(prompt) != CERTIFIED_PROMPT_TOKEN_COUNT
        or len(teacher) != CERTIFIED_TEACHER_TOKEN_COUNT
        or object_sha256(prompt) != CERTIFIED_PROMPT_TOKEN_IDS_SHA256
        or object_sha256(teacher) != CERTIFIED_TEACHER_TOKEN_IDS_SHA256
        or len(sequence) != CERTIFIED_INPUT_TOKEN_COUNT
        or response_start != CERTIFIED_RESPONSE_START
        or len(sequence) - response_start != CERTIFIED_PREDICTOR_COUNT
        or object_sha256(sequence) != CERTIFIED_INPUT_TOKEN_IDS_SHA256
    ):
        _fail(
            "capture is restricted to the certified Chinese v1 "
            "25-token prompt and 64-token BF16 teacher"
        )
    return sequence, response_start


def _teacher_forced_chunks(
    prompt_token_ids: object,
    teacher_token_ids: object,
) -> tuple[list[int], list[tuple[list[int], int, int]]]:
    prompt = _token_ids(prompt_token_ids, "prompt")
    teacher = _token_ids(teacher_token_ids, "BF16 teacher")
    sequence, response_start = build_teacher_forced_sequence(prompt, teacher)
    prefill = prompt[:-1]
    chunks = [([prompt[-1]], 0, teacher[0])]
    chunks.extend(
        ([teacher[index - 1]], 0, teacher[index]) for index in range(1, len(teacher))
    )
    flattened = prefill + [chunk[0] for chunk, _position, _target in chunks]
    if (
        len(prefill) != response_start
        or len(chunks) != CERTIFIED_PREDICTOR_COUNT
        or flattened != sequence
        or any(len(chunk) != 1 or position != 0 for chunk, position, _ in chunks)
    ):
        _fail("certified Chinese v1 stepwise teacher-forcing schedule drifted")
    return prefill, chunks


def expected_w8_module_paths() -> list[str]:
    """Return the exact 184-module hybrid-v2 W8 portfolio in canonical order."""

    paths = ["language_model.model.embed_tokens"]
    for layer in range(24):
        prefix = f"language_model.model.layers.{layer}"
        if (layer + 1) % 4:
            paths.extend(
                f"{prefix}.linear_attn.{name}"
                for name in (
                    "in_proj_a",
                    "in_proj_b",
                    "in_proj_qkv",
                    "in_proj_z",
                    "out_proj",
                )
            )
        else:
            paths.extend(
                f"{prefix}.self_attn.{name}"
                for name in ("k_proj", "o_proj", "q_proj", "v_proj")
            )
        paths.extend(
            f"{prefix}.mlp.{name}" for name in ("down_proj", "gate_proj", "up_proj")
        )
    retained = set(RETAINED_BF16_PATHS)
    observed = [path for path in paths if path not in retained]
    if len(observed) != 184 or len(observed) != len(set(observed)):
        _fail("internal hybrid-v2 W8 module portfolio drifted")
    return observed


class _ModuleMetricCollector:
    def __init__(
        self,
        engine: object,
        paths: list[str],
        *,
        predictor_position: int,
        predictor_step_index: int,
    ) -> None:
        self.engine = engine
        self.paths = list(paths)
        self.allowed = set(paths)
        if (
            type(predictor_position) is not int
            or predictor_position < 0
            or type(predictor_step_index) is not int
            or predictor_step_index < 0
            or predictor_step_index >= CERTIFIED_PREDICTOR_COUNT
        ):
            _fail("manual module collector predictor coordinates drifted")
        self.predictor_position = predictor_position
        self.predictor_step_index = predictor_step_index
        self.observations: dict[str, list[dict[str, object]]] = {}
        self.expected_calls = {path: 1 for path in paths}
        self.expected_calls["language_model.model.embed_tokens"] = 2

    def call(self, path: str, reference_module: object, candidate_module: object, x):
        observations = self.observations.setdefault(path, [])
        if path not in self.allowed or len(observations) >= self.expected_calls[path]:
            _fail(f"manual forward emitted an unexpected module invocation: {path}")
        reference_output = reference_module(x)
        candidate_output = candidate_module(x)
        observations.append(
            self.engine.module_observation(
                path,
                reference_output,
                candidate_output,
                predictor_position=self.predictor_position,
                predictor_step_index=self.predictor_step_index,
            )
        )
        return reference_output

    def finish(self) -> dict[str, list[dict[str, object]]]:
        if any(
            len(self.observations.get(path, [])) != self.expected_calls[path]
            for path in self.paths
        ):
            missing = [
                path
                for path in self.paths
                if len(self.observations.get(path, [])) != self.expected_calls[path]
            ]
            _fail(
                "manual forward could not safely capture every W8 module; "
                f"first missing path: {missing[0] if missing else 'unknown'}"
            )
        return {path: list(self.observations[path]) for path in self.paths}


class MlxQwen35ManualEngine:
    """Exact pinned Qwen3.5 loop with read-only candidate module side calls."""

    def __init__(self, api: object) -> None:
        self.api = api
        self.mx = api.mx
        self.nn = api.nn

    @staticmethod
    def _text_model(model: object):
        if getattr(model, "model_type", None) != "qwen3_5":
            _fail("capture model_type is not qwen3_5")
        text = getattr(model, "language_model", None)
        body = getattr(text, "model", None)
        layers = getattr(body, "layers", None)
        args = getattr(text, "args", None)
        if (
            text is None
            or body is None
            or type(layers) is not list
            or len(layers) != 24
            or args is None
            or getattr(body, "ssm_idx", None) != 0
            or getattr(body, "fa_idx", None) != 3
        ):
            _fail("capture model is not the frozen dense 24-layer Qwen3.5 schema")
        if any(
            getattr(args, field, None) != expected
            for field, expected in _FROZEN_TEXT_SCHEMA.items()
        ):
            _fail("capture model Qwen3.5-0.8B text schema drifted")
        for index, layer in enumerate(layers):
            if getattr(layer, "is_linear", None) is not ((index + 1) % 4 != 0):
                _fail("capture model Qwen3.5 attention schedule drifted")
        return text, body

    @staticmethod
    def _named_modules(model: object) -> dict[str, object]:
        named_modules = getattr(model, "named_modules", None)
        if not callable(named_modules):
            _fail("capture model does not expose named_modules")
        pairs = named_modules()
        if type(pairs) is not list:
            _fail("capture model named_modules result drifted")
        modules = {}
        for path, module in pairs:
            if type(path) is not str or path in modules:
                _fail("capture model contains duplicate or invalid module paths")
            modules[path] = module
        return modules

    def validate_loaded_pair(self, reference: object, candidate: object) -> None:
        """Prove the loaded objects match the frozen BF16/hybrid module tiers."""

        if reference is candidate:
            _fail("reference and candidate must be independent model objects")
        self._text_model(reference)
        self._text_model(candidate)
        reference_modules = self._named_modules(reference)
        candidate_modules = self._named_modules(candidate)
        paths = expected_w8_module_paths()
        required = paths + list(RETAINED_BF16_PATHS)
        if any(
            path not in reference_modules or path not in candidate_modules
            for path in required
        ):
            _fail("loaded pair does not expose the frozen hybrid module portfolio")
        quantized_embedding = getattr(self.nn, "QuantizedEmbedding", None)
        quantized_linear = getattr(self.nn, "QuantizedLinear", None)
        embedding = getattr(self.nn, "Embedding", None)
        linear = getattr(self.nn, "Linear", None)
        if any(
            not isinstance(module_type, type)
            for module_type in (
                quantized_embedding,
                quantized_linear,
                embedding,
                linear,
            )
        ):
            _fail("pinned MLX stateless weight-module classes are unavailable")
        quantized_types = (quantized_embedding, quantized_linear)
        reference_quantized = {
            path
            for path, module in reference_modules.items()
            if isinstance(module, quantized_types)
        }
        candidate_quantized = {
            path
            for path, module in candidate_modules.items()
            if isinstance(module, quantized_types)
        }
        if reference_quantized or candidate_quantized != set(paths):
            _fail("loaded pair BF16/W8 module tier portfolio drifted")
        for index, path in enumerate(paths):
            expected_reference_type = embedding if index == 0 else linear
            expected_candidate_type = (
                quantized_embedding if index == 0 else quantized_linear
            )
            reference_module = reference_modules[path]
            candidate_module = candidate_modules[path]
            if (
                type(reference_module) is not expected_reference_type
                or type(candidate_module) is not expected_candidate_type
                or reference_module is candidate_module
                or getattr(candidate_module, "bits", None) != 8
                or getattr(candidate_module, "group_size", None) != 64
                or getattr(candidate_module, "mode", None) != "affine"
            ):
                _fail(f"loaded pair module class or W8 config drifted at {path}")
        for path in RETAINED_BF16_PATHS:
            if (
                type(reference_modules[path]) is not linear
                or type(candidate_modules[path]) is not linear
                or reference_modules[path] is candidate_modules[path]
            ):
                _fail(f"retained BF16 module tier drifted at {path}")
        if (
            getattr(reference, "training", None) is not False
            or getattr(candidate, "training", None) is not False
        ):
            _fail("loaded pair is not in deterministic evaluation mode")

    def make_cache(self, model: object):
        self._text_model(model)
        make_cache = getattr(model, "make_cache", None)
        if not callable(make_cache):
            _fail("capture model does not expose make_cache")
        cache = make_cache()
        if type(cache) is not list or len(cache) != 24:
            _fail("capture model cache portfolio drifted")
        return cache

    def official_forward(self, model: object, sequence: list[int], cache: object):
        self._text_model(model)
        token_array = self.mx.array(sequence)[None]
        logits = model(token_array, cache=cache)
        self.mx.eval(logits)
        return logits

    def prefill_forward(
        self, model: object, sequence: list[int], cache: object
    ) -> None:
        self._text_model(model)
        if (
            type(sequence) is not list
            or len(sequence) != CERTIFIED_RESPONSE_START
            or type(cache) is not list
            or len(cache) != 24
        ):
            _fail("stepwise prompt prefill shape drifted")
        token_array = self.mx.array(sequence)[None]
        model(token_array, cache=cache)
        states = [item.state for item in cache]
        self.mx.eval(states)
        clear_cache = getattr(self.mx, "clear_cache", None)
        if not callable(clear_cache):
            _fail("pinned MLX runtime cannot clear the prefill graph cache")
        clear_cache()

    def module_observation(
        self,
        path: str,
        reference_output,
        candidate_output,
        *,
        predictor_position: int,
        predictor_step_index: int,
    ) -> dict[str, object]:
        if reference_output.shape != candidate_output.shape:
            _fail(f"candidate module output shape drifted at {path}")
        if len(reference_output.shape) < 3 or reference_output.shape[0] != 1:
            _fail(f"module output is not a batched sequence tensor at {path}")
        if (
            type(predictor_position) is not int
            or predictor_position < 0
            or predictor_position >= reference_output.shape[1]
            or type(predictor_step_index) is not int
            or predictor_step_index < 0
            or predictor_step_index >= CERTIFIED_PREDICTOR_COUNT
        ):
            _fail(f"module predictor coordinates drifted at {path}")
        reference = reference_output[
            :, predictor_position : predictor_position + 1, ...
        ].astype(self.mx.float32)
        candidate = candidate_output[
            :, predictor_position : predictor_position + 1, ...
        ].astype(self.mx.float32)
        difference = self.mx.abs(reference - candidate)
        numerator = self.mx.sum(difference)
        denominator = self.mx.sum(self.mx.abs(reference))
        maximum = self.mx.max(difference)
        finite = self.mx.all(self.mx.isfinite(reference)) & self.mx.all(
            self.mx.isfinite(candidate)
        )
        self.mx.eval(numerator, denominator, maximum, finite)
        if not bool(finite.item()):
            _fail(f"module output contains a non-finite value at {path}")
        numerator_value = float(numerator.item())
        denominator_value = float(denominator.item())
        maximum_value = float(maximum.item())
        return {
            "sample_count": 1,
            "predictor_step_index": predictor_step_index,
            "numerator": numerator_value,
            "denominator": denominator_value,
            "maximum": maximum_value,
            "first_nonzero_step": (
                predictor_step_index if maximum_value > 0.0 else None
            ),
        }

    @staticmethod
    def combine_module_observations(
        path: str,
        observations: list[dict[str, object]],
        *,
        predictor_count: int,
    ) -> dict[str, object]:
        if predictor_count != CERTIFIED_PREDICTOR_COUNT or not observations:
            _fail(f"module has no captured invocation: {path}")
        expected_per_step = 2 if path == "language_model.model.embed_tokens" else 1
        expected_fields = {
            "sample_count",
            "predictor_step_index",
            "numerator",
            "denominator",
            "maximum",
            "first_nonzero_step",
        }
        step_counts = {index: 0 for index in range(predictor_count)}
        malformed = len(observations) != predictor_count * expected_per_step
        for item in observations:
            if type(item) is not dict or set(item) != expected_fields:
                malformed = True
                continue
            step_index = item.get("predictor_step_index")
            values = (
                item.get("numerator"),
                item.get("denominator"),
                item.get("maximum"),
            )
            first_nonzero = item.get("first_nonzero_step")
            if (
                item.get("sample_count") != 1
                or type(step_index) is not int
                or step_index not in step_counts
                or any(type(value) not in (int, float) for value in values)
                or any(
                    not math.isfinite(float(value)) or float(value) < 0.0
                    for value in values
                )
                or (
                    (float(values[2]) > 0.0 and first_nonzero != step_index)
                    or (float(values[2]) == 0.0 and first_nonzero is not None)
                )
            ):
                malformed = True
                continue
            step_counts[step_index] += 1
        if malformed or any(
            count != expected_per_step for count in step_counts.values()
        ):
            _fail(f"module observations are malformed at {path}")
        numerator = math.fsum(float(item["numerator"]) for item in observations)
        denominator = math.fsum(float(item["denominator"]) for item in observations)
        maximum = max(float(item["maximum"]) for item in observations)
        if denominator == 0.0:
            if numerator != 0.0:
                _fail(f"relative module error denominator is zero at {path}")
            relative_ppm = 0
        else:
            relative_ppm = int(round(numerator * 1_000_000 / denominator))
        maximum_micro = int(round(maximum * 1_000_000))
        if (
            relative_ppm < 0
            or relative_ppm > 10**12
            or maximum_micro < 0
            or maximum_micro > 10**12
        ):
            _fail(f"module error is outside the bounded diagnostic range at {path}")
        first_nonzero_values = [
            item["first_nonzero_step"]
            for item in observations
            if item.get("first_nonzero_step") is not None
        ]
        return {
            "path": path,
            "tier": "w8",
            "sample_count": predictor_count,
            "relative_l1_error_ppm": relative_ppm,
            "max_abs_error_micro": maximum_micro,
            "first_nonzero_step": (
                min(first_nonzero_values) if first_nonzero_values else None
            ),
        }

    def _linear_attention(
        self,
        layer_index: int,
        attention: object,
        candidate_modules: dict[str, object],
        inputs,
        mask,
        cache,
        collector: _ModuleMetricCollector,
    ):
        if getattr(attention, "sharding_group", None) is not None:
            _fail("distributed/sharded linear attention is outside capture scope")
        prefix = f"language_model.model.layers.{layer_index}.linear_attn"

        def tap(name: str, x):
            path = f"{prefix}.{name}"
            module = getattr(attention, name, None)
            candidate = candidate_modules.get(path)
            if module is None or candidate is None:
                _fail(f"linear-attention module path is unavailable: {path}")
            return collector.call(path, module, candidate, x)

        batch, steps, _ = inputs.shape
        qkv = tap("in_proj_qkv", inputs)
        z = tap("in_proj_z", inputs).reshape(
            batch, steps, attention.num_v_heads, attention.head_v_dim
        )
        b = tap("in_proj_b", inputs)
        a = tap("in_proj_a", inputs)
        if cache is not None and cache[0] is not None:
            conv_state = cache[0]
        else:
            conv_state = self.mx.zeros(
                (batch, attention.conv_kernel_size - 1, attention.conv_dim),
                dtype=inputs.dtype,
            )
        if mask is not None:
            qkv = self.mx.where(mask[..., None], qkv, 0)
        conv_input = self.mx.concatenate([conv_state, qkv], axis=1)
        if cache is not None:
            n_keep = attention.conv_kernel_size - 1
            if cache.lengths is not None:
                ends = self.mx.clip(cache.lengths, 0, steps)
                positions = (ends[:, None] + self.mx.arange(n_keep))[..., None]
                cache[0] = self.mx.take_along_axis(conv_input, positions, axis=1)
            else:
                cache[0] = self.mx.contiguous(conv_input[:, -n_keep:, :])
        conv_out = self.nn.silu(attention.conv1d(conv_input))
        q, k, v = [
            value.reshape(batch, steps, heads, dimension)
            for value, heads, dimension in zip(
                self.mx.split(conv_out, [attention.key_dim, 2 * attention.key_dim], -1),
                [
                    attention.num_k_heads,
                    attention.num_k_heads,
                    attention.num_v_heads,
                ],
                [attention.head_k_dim, attention.head_k_dim, attention.head_v_dim],
            )
        ]
        state = cache[1] if cache else None
        inverse_scale = k.shape[-1] ** -0.5
        q = (inverse_scale**2) * self.mx.fast.rms_norm(q, None, 1e-6)
        k = inverse_scale * self.mx.fast.rms_norm(k, None, 1e-6)
        out, state = self.api.gated_delta_update(
            q,
            k,
            v,
            a,
            b,
            attention.A_log,
            attention.dt_bias,
            state,
            mask,
            use_kernel=not attention.training,
        )
        if cache is not None:
            cache[1] = state
            cache.advance(steps)
        out = attention.norm(out, z)
        out_input = out.reshape(batch, steps, -1)
        path = f"{prefix}.out_proj"
        if path in collector.allowed:
            candidate = candidate_modules.get(path)
            if candidate is None:
                _fail(f"linear-attention module path is unavailable: {path}")
            return collector.call(path, attention.out_proj, candidate, out_input)
        return attention.out_proj(out_input)

    def _full_attention(
        self,
        layer_index: int,
        attention: object,
        candidate_modules: dict[str, object],
        inputs,
        mask,
        cache,
        collector: _ModuleMetricCollector,
    ):
        prefix = f"language_model.model.layers.{layer_index}.self_attn"

        def tap(name: str, x):
            path = f"{prefix}.{name}"
            module = getattr(attention, name, None)
            candidate = candidate_modules.get(path)
            if module is None or candidate is None:
                _fail(f"full-attention module path is unavailable: {path}")
            return collector.call(path, module, candidate, x)

        batch, length, _ = inputs.shape
        q_projection = tap("q_proj", inputs)
        queries, gate = self.mx.split(
            q_projection.reshape(batch, length, attention.num_attention_heads, -1),
            2,
            axis=-1,
        )
        gate = gate.reshape(batch, length, -1)
        keys = tap("k_proj", inputs)
        values = tap("v_proj", inputs)
        queries = attention.q_norm(queries).transpose(0, 2, 1, 3)
        keys = attention.k_norm(
            keys.reshape(batch, length, attention.num_key_value_heads, -1)
        ).transpose(0, 2, 1, 3)
        values = values.reshape(
            batch, length, attention.num_key_value_heads, -1
        ).transpose(0, 2, 1, 3)
        if cache is not None:
            queries = attention.rope(queries, offset=cache.offset)
            keys = attention.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = attention.rope(queries)
            keys = attention.rope(keys)
        output = self.api.scaled_dot_product_attention(
            queries,
            keys,
            values,
            cache=cache,
            scale=attention.scale,
            mask=mask,
        )
        output = output.transpose(0, 2, 1, 3).reshape(batch, length, -1)
        return tap("o_proj", output * self.mx.sigmoid(gate))

    def _mlp(
        self,
        layer_index: int,
        mlp: object,
        candidate_modules: dict[str, object],
        inputs,
        collector: _ModuleMetricCollector,
    ):
        prefix = f"language_model.model.layers.{layer_index}.mlp"

        def tap(name: str, x):
            path = f"{prefix}.{name}"
            module = getattr(mlp, name, None)
            candidate = candidate_modules.get(path)
            if module is None or candidate is None:
                _fail(f"MLP module path is unavailable: {path}")
            return collector.call(path, module, candidate, x)

        gate = tap("gate_proj", inputs)
        up = tap("up_proj", inputs)
        return tap("down_proj", self.api.swiglu(gate, up))

    def manual_reference_forward(
        self,
        reference: object,
        candidate: object,
        sequence: list[int],
        cache: object,
        *,
        predictor_position: int,
        predictor_step_index: int,
        module_paths: list[str],
    ):
        reference_text, reference_body = self._text_model(reference)
        self._text_model(candidate)
        if type(cache) is not list or len(cache) != 24:
            _fail("manual reference cache portfolio drifted")
        candidate_modules = self._named_modules(candidate)
        if any(path not in candidate_modules for path in module_paths):
            _fail("candidate model does not expose the complete W8 module portfolio")
        collector = _ModuleMetricCollector(
            self,
            module_paths,
            predictor_position=predictor_position,
            predictor_step_index=predictor_step_index,
        )
        token_array = self.mx.array(sequence)[None]
        embed_path = "language_model.model.embed_tokens"
        hidden_states = collector.call(
            embed_path,
            reference_body.embed_tokens,
            candidate_modules[embed_path],
            token_array,
        )
        full_attention_mask = self.api.create_attention_mask(
            hidden_states, cache[reference_body.fa_idx]
        )
        ssm_mask = self.api.create_ssm_mask(
            hidden_states, cache[reference_body.ssm_idx]
        )
        for index, (layer, layer_cache) in enumerate(zip(reference_body.layers, cache)):
            normalized = layer.input_layernorm(hidden_states)
            if layer.is_linear:
                update = self._linear_attention(
                    index,
                    layer.linear_attn,
                    candidate_modules,
                    normalized,
                    ssm_mask,
                    layer_cache,
                    collector,
                )
            else:
                update = self._full_attention(
                    index,
                    layer.self_attn,
                    candidate_modules,
                    normalized,
                    full_attention_mask,
                    layer_cache,
                    collector,
                )
            residual = hidden_states + update
            hidden_states = residual + self._mlp(
                index,
                layer.mlp,
                candidate_modules,
                layer.post_attention_layernorm(residual),
                collector,
            )
        hidden_states = reference_body.norm(hidden_states)
        if not reference_text.args.tie_word_embeddings:
            _fail("untied output embeddings are outside the frozen capture schema")
        reference_embedding = reference_body.embed_tokens
        candidate_embedding = candidate_modules[embed_path]
        reference_as_linear = getattr(reference_embedding, "as_linear", None)
        candidate_as_linear = getattr(candidate_embedding, "as_linear", None)
        if not callable(reference_as_linear) or not callable(candidate_as_linear):
            _fail("tied embedding does not expose the pinned as_linear operation")
        logits = collector.call(
            embed_path,
            reference_as_linear,
            candidate_as_linear,
            hidden_states,
        )
        self.mx.eval(logits)
        return logits, collector.finish()

    def assert_exact_logits(self, official, manual, *, predictor_position: int) -> None:
        if official.shape != manual.shape:
            _fail("manual Qwen3.5 logits shape differs from the installed model loop")
        exact = self.mx.all(official == manual)
        finite = self.mx.all(self.mx.isfinite(official)) & self.mx.all(
            self.mx.isfinite(manual)
        )
        self.mx.eval(exact, finite)
        if not bool(finite.item()) or not bool(exact.item()):
            _fail("manual Qwen3.5 loop is not bit-exact with the installed model loop")
        if (
            type(predictor_position) is not int
            or predictor_position < 0
            or official.shape[1] <= predictor_position
        ):
            _fail("manual Qwen3.5 logits omit the stepwise predictor position")

    @staticmethod
    def _micro(value: object, label: str) -> int:
        observed = float(value)
        if not math.isfinite(observed) or abs(observed) > 10**6:
            _fail(f"{label} is non-finite or outside the bounded logit range")
        return int(round(observed * 1_000_000))

    def _production_logprobs(self, logits):
        """Match mlx-lm 0.31.3 generate_step normalization in the input dtype."""

        return logits - self.mx.logsumexp(logits, keepdims=True)

    def step_metric(
        self,
        reference_logits,
        candidate_logits,
        teacher_token_id: int,
        *,
        predictor_position: int,
        predictor_step_index: int,
    ) -> dict[str, object]:
        if (
            reference_logits.shape != candidate_logits.shape
            or reference_logits.shape[0] != 1
            or type(predictor_position) is not int
            or predictor_position < 0
            or reference_logits.shape[1] <= predictor_position
            or reference_logits.shape[2] != 248320
            or type(teacher_token_id) is not int
            or teacher_token_id < 0
            or teacher_token_id >= 248320
            or type(predictor_step_index) is not int
            or predictor_step_index < 0
            or predictor_step_index >= CERTIFIED_PREDICTOR_COUNT
        ):
            _fail("reference/candidate full-vocabulary logits shape drifted")
        reference_raw = reference_logits[
            0, predictor_position : predictor_position + 1, :
        ]
        candidate_raw = candidate_logits[
            0, predictor_position : predictor_position + 1, :
        ]
        reference = self._production_logprobs(reference_raw)
        candidate = self._production_logprobs(candidate_raw)
        teacher = self.mx.array([teacher_token_id])
        reference_top1_ids = self.mx.argmax(reference, axis=-1)
        candidate_top1_ids = self.mx.argmax(candidate, axis=-1)
        reference_top2 = self.mx.topk(reference, k=2, axis=-1)
        candidate_top2 = self.mx.topk(candidate, k=2, axis=-1)
        reference_margin = self.mx.max(reference_top2, axis=-1) - self.mx.min(
            reference_top2, axis=-1
        )
        candidate_top1 = self.mx.max(candidate_top2, axis=-1)
        candidate_second = self.mx.min(candidate_top2, axis=-1)
        teacher_logits = self.mx.take_along_axis(
            candidate, teacher[:, None], axis=-1
        ).squeeze(-1)
        candidate_best_alternative = self.mx.where(
            candidate_top1_ids == teacher,
            candidate_second,
            candidate_top1,
        )
        candidate_teacher_margin = teacher_logits - candidate_best_alternative
        finite = (
            self.mx.all(self.mx.isfinite(reference))
            & self.mx.all(self.mx.isfinite(candidate))
            & self.mx.all(self.mx.isfinite(reference_margin))
            & self.mx.all(self.mx.isfinite(candidate_teacher_margin))
        )
        self.mx.eval(
            reference_top1_ids,
            candidate_top1_ids,
            reference_margin,
            candidate_teacher_margin,
            finite,
        )
        if not bool(finite.item()):
            _fail("teacher-forced logits or margins contain a non-finite value")
        reference_ids = list(reference_top1_ids.tolist())
        candidate_ids = list(candidate_top1_ids.tolist())
        reference_margins = list(reference_margin.tolist())
        candidate_margins = list(candidate_teacher_margin.tolist())
        return {
            "step_index": predictor_step_index,
            "reference_token_id": teacher_token_id,
            "reference_top1_token_id": int(reference_ids[0]),
            "candidate_top1_token_id": int(candidate_ids[0]),
            "reference_top1_margin_micro": self._micro(
                reference_margins[0], "reference top-1 margin"
            ),
            "candidate_reference_token_margin_micro": self._micro(
                candidate_margins[0], "candidate reference-token margin"
            ),
        }


def capture_loaded_models(
    reference: object,
    candidate: object,
    *,
    prompt_token_ids: object,
    teacher_token_ids: object,
    engine: object,
) -> dict[str, object]:
    """Capture two deterministic teacher-forced runs from already-loaded models."""

    prompt = _token_ids(prompt_token_ids, "prompt")
    teacher = _token_ids(teacher_token_ids, "BF16 teacher")
    prefill, chunks = _teacher_forced_chunks(prompt, teacher)
    required_calls = (
        "make_cache",
        "prefill_forward",
        "official_forward",
        "manual_reference_forward",
        "assert_exact_logits",
        "step_metric",
        "combine_module_observations",
    )
    if any(not callable(getattr(engine, name, None)) for name in required_calls):
        _fail("capture engine interface is incomplete")
    module_paths = expected_w8_module_paths()
    runs = []
    for _repeat in range(2):
        reference_official_cache = engine.make_cache(reference)
        reference_manual_cache = engine.make_cache(reference)
        candidate_cache = engine.make_cache(candidate)
        engine.prefill_forward(reference, prefill, reference_official_cache)
        engine.prefill_forward(reference, prefill, reference_manual_cache)
        engine.prefill_forward(candidate, prefill, candidate_cache)
        observations = {path: [] for path in module_paths}
        step_metrics = []
        for step_index, (chunk, predictor_position, teacher_token_id) in enumerate(
            chunks
        ):
            reference_official = engine.official_forward(
                reference, chunk, reference_official_cache
            )
            reference_manual, chunk_observations = engine.manual_reference_forward(
                reference,
                candidate,
                chunk,
                reference_manual_cache,
                predictor_position=predictor_position,
                predictor_step_index=step_index,
                module_paths=module_paths,
            )
            engine.assert_exact_logits(
                reference_official,
                reference_manual,
                predictor_position=predictor_position,
            )
            candidate_logits = engine.official_forward(
                candidate, chunk, candidate_cache
            )
            step_metrics.append(
                engine.step_metric(
                    reference_manual,
                    candidate_logits,
                    teacher_token_id,
                    predictor_position=predictor_position,
                    predictor_step_index=step_index,
                )
            )
            if type(chunk_observations) is not dict or set(chunk_observations) != set(
                module_paths
            ):
                _fail("manual forward chunk module portfolio drifted")
            for path in module_paths:
                items = chunk_observations[path]
                expected_calls = 2 if path == "language_model.model.embed_tokens" else 1
                if type(items) is not list or len(items) != expected_calls:
                    _fail(f"manual forward chunk call count drifted at {path}")
                observations[path].extend(items)
        module_metrics = [
            engine.combine_module_observations(
                path,
                observations[path],
                predictor_count=CERTIFIED_PREDICTOR_COUNT,
            )
            for path in module_paths
        ]
        if (
            type(module_metrics) is not list
            or [
                metric.get("path") if type(metric) is dict else None
                for metric in module_metrics
            ]
            != module_paths
        ):
            _fail("manual forward did not capture the exact 184-module W8 portfolio")
        if type(step_metrics) is not list or len(step_metrics) != len(teacher):
            _fail("capture engine did not return every teacher-forced step")
        runs.append(
            json.loads(
                canonical_bytes(
                    {
                        "step_metrics": step_metrics,
                        "module_metrics": module_metrics,
                    }
                )
            )
        )
    if runs[0] != runs[1]:
        _fail("two state-aligned capture runs are not identical")
    return {
        "format": CAPTURE_FORMAT,
        "prompt_id": PROMPT_ID,
        "prompt_token_ids": prompt,
        "teacher_token_ids": teacher,
        "retained_bf16_paths": list(RETAINED_BF16_PATHS),
        "w8_module_paths": module_paths,
        "w8_module_paths_sha256": object_sha256(module_paths),
        "runs": runs,
    }


class Qwen35StateAlignedCaptureBackend:
    """Frozen diagnostic backend with start/end custody and no model mutation."""

    def __init__(
        self,
        *,
        runtime_loader=None,
        runtime_identity_provider=None,
        source_auditor=None,
        bundle_snapshotter=None,
        engine_factory=None,
        source_custody=None,
    ) -> None:
        self._runtime_loader = runtime_loader or _load_runtime
        self._runtime_identity_provider = runtime_identity_provider or _runtime_identity
        self._source_auditor = source_auditor or _source_identity
        self._bundle_snapshotter = bundle_snapshotter or _snapshot_bundle
        self._engine_factory = engine_factory or MlxQwen35ManualEngine
        self._source_custody = _validate_backend_source_custody(
            source_custody if source_custody is not None else _backend_source_custody()
        )
        self._state: dict[str, object] | None = None

    def capabilities(self) -> dict[str, object]:
        current = _validate_backend_source_custody(self._source_custody)
        return json.loads(
            canonical_bytes(
                {**REQUIRED_CAPTURE_CAPABILITIES, "source_custody": current}
            )
        )

    @staticmethod
    def _inputs(value: object) -> dict[str, str]:
        if type(value) is not dict:
            _fail("capture backend inputs must be an object")
        required = (
            "reference_bundle_path",
            "candidate_bundle_path",
            "reference_manifest_sha256",
            "candidate_manifest_sha256",
        )
        if any(type(value.get(field)) is not str for field in required):
            _fail("capture backend bundle identity inputs are incomplete")
        observed = {field: value[field] for field in required}
        if (
            not Path(observed["reference_bundle_path"]).is_absolute()
            or not Path(observed["candidate_bundle_path"]).is_absolute()
            or observed["reference_bundle_path"] == observed["candidate_bundle_path"]
            or _SHA256.fullmatch(observed["reference_manifest_sha256"]) is None
            or _SHA256.fullmatch(observed["candidate_manifest_sha256"]) is None
        ):
            _fail("capture backend bundle identity inputs are malformed")
        return observed

    @staticmethod
    def _validate_snapshot(
        value: object,
        *,
        path: str,
        manifest_sha256: str,
        label: str,
    ) -> dict[str, object]:
        if (
            type(value) is not dict
            or value.get("path") != path
            or value.get("manifest_sha256") != manifest_sha256
        ):
            _fail(f"{label} bundle snapshot is not bound to certified custody")
        return json.loads(canonical_bytes(value))

    @staticmethod
    def _loaded_model(value: object, label: str) -> tuple[object, object]:
        if type(value) is not tuple or len(value) != 2:
            _fail(f"pinned MLX-LM returned an unexpected {label} load result")
        model, tokenizer = value
        if model is None or tokenizer is None:
            _fail(f"pinned MLX-LM returned an empty {label} handle")
        return model, tokenizer

    def open_pair(self, inputs: object) -> dict[str, object]:
        if self._state is not None:
            _fail("capture backend already has an open model pair")
        _validate_backend_source_custody(self._source_custody)
        identity = self._inputs(inputs)
        runtime_start = _validate_runtime_identity(self._runtime_identity_provider())
        source_start = _validate_source_identity(self._source_auditor())
        reference_before = self._validate_snapshot(
            self._bundle_snapshotter(
                identity["reference_bundle_path"],
                "reference",
                precision_profile="bf16",
            ),
            path=identity["reference_bundle_path"],
            manifest_sha256=identity["reference_manifest_sha256"],
            label="reference",
        )
        candidate_before = self._validate_snapshot(
            self._bundle_snapshotter(
                identity["candidate_bundle_path"],
                "candidate",
                precision_profile="hybrid-w8-bf16-g64",
            ),
            path=identity["candidate_bundle_path"],
            manifest_sha256=identity["candidate_manifest_sha256"],
            label="candidate",
        )
        _enforce_offline_environment()
        runtime = None
        try:
            runtime = self._runtime_loader()
            reference, reference_tokenizer = self._loaded_model(
                runtime.load(
                    identity["reference_bundle_path"],
                    tokenizer_config={
                        "local_files_only": True,
                        "trust_remote_code": False,
                    },
                    lazy=True,
                ),
                "reference",
            )
            candidate, candidate_tokenizer = self._loaded_model(
                runtime.load(
                    identity["candidate_bundle_path"],
                    tokenizer_config={
                        "local_files_only": True,
                        "trust_remote_code": False,
                    },
                    lazy=True,
                ),
                "candidate",
            )
            del reference_tokenizer, candidate_tokenizer
            if reference is candidate:
                _fail("pinned MLX-LM did not create independent model handles")
            engine = self._engine_factory(runtime)
            validate_pair = getattr(engine, "validate_loaded_pair", None)
            if not callable(validate_pair):
                _fail("capture engine cannot validate the loaded BF16/W8 pair")
            validate_pair(reference, candidate)
            process_id = os.getpid()
            pair_seed = {
                "process_id": process_id,
                "reference_manifest_sha256": identity["reference_manifest_sha256"],
                "candidate_manifest_sha256": identity["candidate_manifest_sha256"],
            }
            pair = {
                "pair_id": f"qwen35-state-aligned-{object_sha256(pair_seed)[:24]}",
                "process_id": process_id,
                "reference_handle_id": (
                    f"bf16-{identity['reference_manifest_sha256'][:24]}"
                ),
                "candidate_handle_id": (
                    f"hybrid-{identity['candidate_manifest_sha256'][:24]}"
                ),
                "reference_manifest_sha256": identity["reference_manifest_sha256"],
                "candidate_manifest_sha256": identity["candidate_manifest_sha256"],
            }
            self._state = {
                "identity": identity,
                "runtime_start": runtime_start,
                "source_start": source_start,
                "reference_before": reference_before,
                "candidate_before": candidate_before,
                "runtime": runtime,
                "engine": engine,
                "reference": reference,
                "candidate": candidate,
                "pair": pair,
            }
            return json.loads(canonical_bytes(pair))
        except CaptureError:
            if runtime is not None:
                clear_cache = getattr(runtime, "clear_cache", None)
                if callable(clear_cache):
                    clear_cache()
            gc.collect()
            raise
        except Exception as error:
            if runtime is not None:
                clear_cache = getattr(runtime, "clear_cache", None)
                if callable(clear_cache):
                    clear_cache()
            gc.collect()
            raise CaptureError(f"cannot open pinned model pair: {error}") from error

    def _require_pair(self, pair: object) -> dict[str, object]:
        if self._state is None or type(pair) is not dict:
            _fail("capture backend model pair is not open")
        if pair != self._state["pair"] or pair.get("process_id") != os.getpid():
            _fail("capture backend pair handle is stale or unbound")
        return self._state

    def capture_state_aligned(
        self,
        pair: object,
        *,
        prompt_token_ids: object,
        teacher_token_ids: object,
        repeats: object,
    ) -> dict[str, object]:
        state = self._require_pair(pair)
        if type(repeats) is not int or repeats != 2:
            _fail("state-aligned capture requires exactly two repeats")
        return capture_loaded_models(
            state["reference"],
            state["candidate"],
            prompt_token_ids=prompt_token_ids,
            teacher_token_ids=teacher_token_ids,
            engine=state["engine"],
        )

    def close_pair(self, pair: object) -> None:
        state = self._require_pair(pair)
        identity = state["identity"]
        assert type(identity) is dict
        runtime = state["runtime"]
        errors = []
        state["reference"] = None
        state["candidate"] = None
        state["engine"] = None
        try:
            clear_cache = getattr(runtime, "clear_cache", None)
            if not callable(clear_cache):
                _fail("pinned runtime cannot clear its capture cache")
            clear_cache()
        except Exception as error:
            errors.append(f"runtime cache cleanup failed: {error}")
        gc.collect()
        try:
            runtime_end = _validate_runtime_identity(self._runtime_identity_provider())
            if runtime_end != state["runtime_start"]:
                _fail("runtime identity changed during capture")
        except Exception as error:
            errors.append(str(error))
        try:
            source_end = _validate_source_identity(self._source_auditor())
            if source_end != state["source_start"]:
                _fail("pinned source identity changed during capture")
        except Exception as error:
            errors.append(str(error))
        try:
            _validate_backend_source_custody(self._source_custody)
        except Exception as error:
            errors.append(str(error))
        for label, profile in (
            ("reference", "bf16"),
            ("candidate", "hybrid-w8-bf16-g64"),
        ):
            try:
                after = self._validate_snapshot(
                    self._bundle_snapshotter(
                        identity[f"{label}_bundle_path"],
                        label,
                        precision_profile=profile,
                    ),
                    path=identity[f"{label}_bundle_path"],
                    manifest_sha256=identity[f"{label}_manifest_sha256"],
                    label=label,
                )
                if after != state[f"{label}_before"]:
                    _fail(f"{label} bundle changed during capture")
            except Exception as error:
                errors.append(str(error))
        self._state = None
        if errors:
            _fail("capture cleanup/custody check failed: " + "; ".join(errors))


def load_backend(*, source_custody=None) -> Qwen35StateAlignedCaptureBackend:
    """Return the production backend without importing MLX or opening weights."""

    return Qwen35StateAlignedCaptureBackend(source_custody=source_custody)
