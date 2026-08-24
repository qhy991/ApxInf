#!/usr/bin/env python3
"""Produce one trusted, offline MLX multi-prompt quality evidence envelope.

The producer loads two direct local bundles, executes the frozen raw-token
suite twice per bundle through ``mlx_lm.generate.generate_step``, rechecks all
input custody, calls the independent validator, and atomically publishes one
no-replace JSON envelope.  A deterministic quality miss is publishable as an
explicit failed comparison; malformed or unstable evidence is not.
"""

from __future__ import annotations

import argparse
import hashlib
from importlib import metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import re
import secrets
import stat
import sys
from typing import Callable, NoReturn


ENVELOPE_FORMAT = "apxinf-mlx-multi-prompt-quality-run-v1"
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
_CANDIDATE_ID = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MAX_BUNDLE_FILES = 64
_MAX_CONFIG_BYTES = 2 * 1024 * 1024
_MAX_INDEX_BYTES = 64 * 1024 * 1024
_MAX_TOKENIZER_CONFIG_BYTES = 2 * 1024 * 1024
_MAX_OUTPUT_BYTES = 8 * 1024 * 1024
_HASH_CHUNK_BYTES = 4 * 1024 * 1024
_SOURCE_REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
_HYBRID_W8_BF16_PROFILE = "hybrid-w8-bf16-g64"
_COUNTERFACTUAL_PROFILE = (
    "hybrid-w8-bf16-g64-chinese-top1-counterfactual-v1"
)
_COUNTERFACTUAL_REFERENCE_MANIFEST_SHA256 = (
    "fdce8bac86b1bbc888cac0139065f0291a9a57ce7f448e591b748f4baaad5dea"
)
_AFFINE_W8_G64 = {"bits": 8, "group_size": 64, "mode": "affine"}
_HYBRID_W8_BF16_PRESET = {
    "name": "qwen35-0.8b-affine-w8-g64-gdn-outproj-async-chat-parity-v2",
    "policy_sha256": (
        "64a2ba1741fd5a76a7e72580ce9188d1554e1488ce6504b20054bf42479eaf8f"
    ),
    "retained_bf16_paths": [
        "language_model.model.layers.12.linear_attn.out_proj",
        "language_model.model.layers.14.linear_attn.out_proj",
        "language_model.model.layers.20.linear_attn.out_proj",
    ],
    "source_revision": _SOURCE_REVISION,
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
_COUNTERFACTUAL_PRESET = {
    "name": (
        "qwen35-0.8b-affine-w8-g64-gdn3-l19-o-proj-chinese-counterfactual-v3"
    ),
    "policy_sha256": (
        "7030fe5a7c4dd55cbf158750e9da3a67c7f8e65944b8f8835c75b1093e12eec9"
    ),
    "retained_bf16_paths": [
        "language_model.model.layers.12.linear_attn.out_proj",
        "language_model.model.layers.14.linear_attn.out_proj",
        "language_model.model.layers.19.self_attn.o_proj",
        "language_model.model.layers.20.linear_attn.out_proj",
    ],
    "source_revision": _SOURCE_REVISION,
    "weight_ledger": {
        "estimated_total_parameter_bytes": 807754432,
        "output_tensor_count": 686,
        "quantized_logical_weight_count": 743505920,
        "quantized_module_count": 183,
        "quantized_module_parameter_bytes": 789975040,
        "retained_bf16_logical_weight_count": 8388608,
        "retained_bf16_module_count": 4,
        "retained_bf16_weight_bytes": 16777216,
    },
    "counterfactual": {
        "format": "apxinf-qwen35-mlx-hybrid-counterfactual-lineage-v1",
        "status": "unvalidated-candidate",
        "selection": {
            "causal_attribution": False,
            "current_tier": "w8",
            "path": "language_model.model.layers.19.self_attn.o_proj",
            "proposed_tier": "bf16",
            "rank": 1,
            "ranking_metric": "same-bf16-input-relative-l1-error-ppm-v1",
            "selection_basis": "trusted-diagnostic-trigger-only-not-causal-proof-v1",
        },
        "diagnostic": {
            "artifact_path": (
                "doc/20260823-qwen35-macos-bringup/"
                "qwen35-hybrid-w8-bf16-g64-chinese-state-aligned-"
                "diagnostic-v1.json"
            ),
            "artifact_sha256": (
                "1b30a3a7f6d609a8265112bde3189b7638a9072561530b852ec86dbc4794b73d"
            ),
            "content_sha256": (
                "e0c207bf46a62b643e3aeadc9398aea0d983426585d9b13ce25d21ce35d21a7f"
            ),
            "format": "apxinf-mlx-chinese-hybrid-diagnostic-receipt-v1",
        },
        "parent": {
            "bundle_manifest_sha256": (
                "5f65527db17cd4fb7a1a46ca6ab3940a8bcc8225c620ca10d3016265d2bf8553"
            ),
            "policy_sha256": (
                "64a2ba1741fd5a76a7e72580ce9188d1554e1488ce6504b20054bf42479eaf8f"
            ),
            "preset": (
                "qwen35-0.8b-affine-w8-g64-gdn-outproj-async-chat-parity-v2"
            ),
        },
        "reference": {
            "bundle_manifest_sha256": (
                "fdce8bac86b1bbc888cac0139065f0291a9a57ce7f448e591b748f4baaad5dea"
            ),
            "precision": "bf16",
        },
        "admission": {
            "formal_performance_claim": False,
            "general_parity": False,
            "parent_bundle_replacement": False,
            "promotion_requires_all_gates": True,
            "required_gates": [
                "apxinf-mlx-counterfactual-deployed-canonical-gate-v1",
                "qwen35-0.8b-mlx-multi-prompt-quality-v1-4-prompts-x2",
            ],
        },
    },
}
_PRECISION_PROFILES = frozenset(
    {
        "bf16",
        "w8-g64",
        "w4-g64",
        _HYBRID_W8_BF16_PROFILE,
        _COUNTERFACTUAL_PROFILE,
        "mixed-w4-w8-bf16",
    }
)
_CLAIMS = frozenset({"fixed-suite-exact-parity", "fixed-suite-threshold-match"})
_OFFLINE_ENVIRONMENT = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "TOKENIZERS_PARALLELISM": "false",
    "NO_PROXY": "*",
    "no_proxy": "*",
}


class ProducerError(ValueError):
    """A fail-closed producer, custody, runtime, or publication violation."""


def _fail(message: str) -> NoReturn:
    raise ProducerError(message)


def _load_validator_module():
    validator_path = (
        Path(__file__).resolve().with_name("validate_mlx_multi_prompt_quality.py")
    )
    specification = importlib.util.spec_from_file_location(
        "apxinf_mlx_multi_prompt_quality_validator_v1", validator_path
    )
    if specification is None or specification.loader is None:
        raise RuntimeError(f"cannot load quality validator: {validator_path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


VALIDATOR = _load_validator_module()


def _canonical_path(path: Path, label: str, *, directory: bool) -> Path:
    path = Path(path)
    if not path.is_absolute():
        _fail(f"{label} must be an absolute direct path")
    try:
        resolved = path.resolve(strict=True)
        observed = path.lstat()
    except OSError as error:
        raise ProducerError(f"cannot inspect {label}: {error}") from error
    if resolved != path or stat.S_ISLNK(observed.st_mode):
        _fail(f"{label} must be canonical and contain no symlink components")
    wanted = stat.S_ISDIR if directory else stat.S_ISREG
    if not wanted(observed.st_mode):
        _fail(
            f"{label} must be a direct {'directory' if directory else 'regular file'}"
        )
    return path


def _prepare_output(path: Path) -> Path:
    path = Path(path)
    if not path.is_absolute():
        _fail("--output must be an absolute no-replace path")
    parent = _canonical_path(path.parent, "output parent", directory=True)
    output = parent / path.name
    if output != path or path.name in {"", ".", ".."}:
        _fail("--output must be a canonical direct child of its absolute parent")
    try:
        output.lstat()
    except FileNotFoundError:
        return output
    except OSError as error:
        raise ProducerError(f"cannot inspect --output: {error}") from error
    _fail("--output already exists; evidence publication is no-replace")


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


def _direct_file_stat(path: Path, label: str) -> os.stat_result:
    try:
        observed = path.lstat()
    except OSError as error:
        raise ProducerError(f"cannot inspect {label}: {error}") from error
    if not stat.S_ISREG(observed.st_mode) or stat.S_ISLNK(observed.st_mode):
        _fail(f"{label} must be a direct regular file")
    if observed.st_nlink != 1:
        _fail(f"{label} must have exactly one hard link")
    return observed


def _stream_sha256(path: Path) -> str:
    """Hash a direct regular file through a no-follow descriptor in chunks."""

    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ProducerError(
            f"cannot open direct file for hashing {path}: {error}"
        ) from error
    digest = hashlib.sha256()
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            _fail(f"hash input is not a regular file: {path}")
        while True:
            chunk = os.read(descriptor, _HASH_CHUNK_BYTES)
            if not chunk:
                break
            digest.update(chunk)
        finished = os.fstat(descriptor)
        if _stable_fields(opened) != _stable_fields(finished):
            _fail(f"file changed while it was hashed: {path}")
    finally:
        os.close(descriptor)
    return digest.hexdigest()


def _read_direct_bytes(path: Path, label: str, maximum_bytes: int) -> bytes:
    before = _direct_file_stat(path, label)
    if before.st_size > maximum_bytes:
        _fail(f"{label} exceeds the {maximum_bytes}-byte limit")
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ProducerError(f"cannot open {label}: {error}") from error
    payload = bytearray()
    try:
        opened = os.fstat(descriptor)
        if _stable_fields(opened) != _stable_fields(before):
            _fail(f"{label} changed before it was read")
        while True:
            chunk = os.read(
                descriptor, min(1024 * 1024, maximum_bytes + 1 - len(payload))
            )
            if not chunk:
                break
            payload.extend(chunk)
            if len(payload) > maximum_bytes:
                _fail(f"{label} exceeds the {maximum_bytes}-byte limit")
        finished = os.fstat(descriptor)
        if _stable_fields(opened) != _stable_fields(finished):
            _fail(f"{label} changed while it was read")
    finally:
        os.close(descriptor)
    return bytes(payload)


def _reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            _fail(f"JSON contains duplicate key: {key}")
        result[key] = value
    return result


def _read_json(path: Path, label: str, maximum_bytes: int) -> dict[str, object]:
    payload = _read_direct_bytes(path, label, maximum_bytes)
    try:
        value = json.loads(
            payload,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda constant: _fail(
                f"{label} contains non-finite JSON number {constant}"
            ),
        )
    except ProducerError:
        raise
    except (UnicodeError, ValueError) as error:
        raise ProducerError(f"{label} is not valid UTF-8 JSON") from error
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


def _validate_quantization(
    config: dict[str, object], precision_profile: str, label: str
) -> None:
    quantization = config.get("quantization")
    quantization_config = config.get("quantization_config")
    selective = config.get("apxinf_selective_mixed_policy")
    hybrid = config.get("apxinf_hybrid_preset")
    if precision_profile == "bf16":
        for value in (quantization, quantization_config):
            if value is not None:
                _fail(f"{label} BF16 reference must not declare quantization")
        if selective is not None or hybrid is not None:
            _fail(f"{label} BF16 reference must not declare a mixed policy")
        return
    expected_bits = (
        8
        if precision_profile
        in {"w8-g64", _HYBRID_W8_BF16_PROFILE, _COUNTERFACTUAL_PROFILE}
        else 4
    )
    if type(quantization) is not dict or type(quantization_config) is not dict:
        _fail(f"{label} must declare affine group-64 quantization")
    for value in (quantization, quantization_config):
        if (
            value.get("bits") != expected_bits
            or value.get("group_size") != 64
            or value.get("mode") != "affine"
        ):
            _fail(f"{label} precision profile does not match its quantization config")
    if precision_profile in {_HYBRID_W8_BF16_PROFILE, _COUNTERFACTUAL_PROFILE}:
        if quantization != _AFFINE_W8_G64 or quantization_config != _AFFINE_W8_G64:
            _fail(f"{label} hybrid-W8/BF16 profile is not global affine W8 group-64")
        if selective is not None:
            _fail(f"{label} hybrid-W8/BF16 profile must not be selectively quantized")
        expected_hybrid = (
            _COUNTERFACTUAL_PRESET
            if precision_profile == _COUNTERFACTUAL_PROFILE
            else _HYBRID_W8_BF16_PRESET
        )
        if hybrid != expected_hybrid:
            _fail(f"{label} hybrid-W8/BF16 frozen preset manifest drifted")
        return
    if precision_profile != "mixed-w4-w8-bf16":
        if selective is not None or hybrid is not None:
            _fail(f"{label} uniform precision profile must not declare a mixed policy")
        return
    if (
        type(selective) is not dict
        or selective.get("format") != "apxinf-mlx-selective-mixed-policy-manifest-v1"
        or selective.get("source_repo_id") != "Qwen/Qwen3.5-0.8B"
        or selective.get("source_revision") != _SOURCE_REVISION
        or type(selective.get("candidate_module_count")) is not int
        or type(selective.get("w8_paths")) is not list
        or not selective["w8_paths"]
        or type(selective.get("retained_bf16_paths")) is not list
        or not selective["retained_bf16_paths"]
    ):
        _fail(f"{label} mixed-W4/W8/BF16 profile lacks its selective policy manifest")
    if hybrid is not None:
        _fail(f"{label} selective mixed profile must not declare a hybrid preset")
    w8_paths = selective["w8_paths"]
    retained_paths = selective["retained_bf16_paths"]
    if (
        any(type(path) is not str or not path for path in w8_paths + retained_paths)
        or len(set(w8_paths)) != len(w8_paths)
        or len(set(retained_paths)) != len(retained_paths)
        or set(w8_paths) & set(retained_paths)
        or selective["candidate_module_count"] <= len(w8_paths) + len(retained_paths)
    ):
        _fail(f"{label} mixed policy tier portfolio is invalid")
    w8_quantization = {"bits": 8, "group_size": 64, "mode": "affine"}
    for path in w8_paths:
        if (
            quantization.get(path) != w8_quantization
            or quantization_config.get(path) != w8_quantization
        ):
            _fail(f"{label} mixed policy W8 override is absent from config")
    if any(
        path in quantization or path in quantization_config for path in retained_paths
    ):
        _fail(f"{label} mixed policy BF16 retention is quantized in config")


def _validate_bundle_config(
    config: dict[str, object], precision_profile: str, label: str
) -> None:
    _forbid_remote_code(config, f"{label}/config.json")
    if config.get("model_type") != "qwen3_5":
        _fail(f"{label} config model_type is not qwen3_5")
    architectures = config.get("architectures")
    if (
        type(architectures) is not list
        or "Qwen3_5ForConditionalGeneration" not in architectures
    ):
        _fail(f"{label} config architecture is not Qwen3.5")
    text_config = config.get("text_config")
    if (
        type(text_config) is not dict
        or text_config.get("model_type") != "qwen3_5_text"
        or text_config.get("dtype") != "bfloat16"
        or text_config.get("vocab_size") != 248320
    ):
        _fail(f"{label} text config is not the frozen Qwen3.5-0.8B BF16 schema")
    _validate_quantization(config, precision_profile, label)


def _validate_index(
    index: dict[str, object], shard_names: list[str], label: str
) -> None:
    if (
        set(index) != {"metadata", "weight_map"}
        or type(index.get("metadata")) is not dict
    ):
        _fail(f"{label}/model.safetensors.index.json fields drifted")
    weight_map = index.get("weight_map")
    if type(weight_map) is not dict or not weight_map:
        _fail(f"{label} model index must contain a non-empty weight_map")
    if any(type(name) is not str or not name for name in weight_map):
        _fail(f"{label} model index contains an invalid tensor name")
    referenced = list(weight_map.values())
    if any(type(name) is not str or name not in shard_names for name in referenced):
        _fail(f"{label} model index references an uncontrolled shard")
    if set(referenced) != set(shard_names):
        _fail(f"{label} model index does not bind every shard")


def _validate_shard_names(shard_names: list[str], label: str) -> None:
    if shard_names == ["model.safetensors"]:
        return
    parsed = []
    for name in shard_names:
        match = re.fullmatch(r"model-([0-9]{5})-of-([0-9]{5})\.safetensors", name)
        if match is None:
            _fail(f"{label} has an invalid model shard portfolio")
        parsed.append((int(match.group(1)), int(match.group(2))))
    totals = {total for _, total in parsed}
    if len(totals) != 1:
        _fail(f"{label} model shard totals disagree")
    total = totals.pop()
    if total != len(parsed) or sorted(index for index, _ in parsed) != list(
        range(1, total + 1)
    ):
        _fail(f"{label} model shard sequence is incomplete")


def _snapshot_bundle(
    path: Path,
    label: str,
    *,
    precision_profile: str,
    expected_tokenizer_sha256: str,
    file_hasher: Callable[[Path], str],
) -> dict[str, object]:
    path = _canonical_path(path, label, directory=True)
    try:
        entries = list(os.scandir(path))
    except OSError as error:
        raise ProducerError(f"cannot list {label}: {error}") from error
    if len(entries) > _MAX_BUNDLE_FILES:
        _fail(f"{label} exceeds the {_MAX_BUNDLE_FILES}-file flat-layout limit")
    names = {entry.name for entry in entries}
    if len(names) != len(entries):
        _fail(f"{label} contains duplicate directory names")
    shard_names = sorted(name for name in names if _MODEL_SHARD.fullmatch(name))
    allowed_names = _FIXED_BUNDLE_FILES | set(shard_names)
    if names != allowed_names:
        missing = sorted(_FIXED_BUNDLE_FILES - names)
        unexpected = sorted(names - allowed_names)
        _fail(
            f"{label} is not the controlled flat bundle layout "
            f"(missing={missing}, unexpected={unexpected})"
        )
    if not shard_names:
        _fail(f"{label} has no model safetensors shard")
    _validate_shard_names(shard_names, label)

    files: dict[str, dict[str, object]] = {}
    for name in sorted(names):
        file_path = path / name
        before = _direct_file_stat(file_path, f"{label}/{name}")
        try:
            digest = file_hasher(file_path)
        except ProducerError:
            raise
        except Exception as error:
            raise ProducerError(f"cannot hash {label}/{name}: {error}") from error
        after = _direct_file_stat(file_path, f"{label}/{name}")
        if _stable_fields(before) != _stable_fields(after):
            _fail(f"{label}/{name} changed while its bundle was inspected")
        if type(digest) is not str or _SHA256.fullmatch(digest) is None:
            _fail(f"{label}/{name} hasher did not return lowercase SHA-256")
        files[name] = {"size": before.st_size, "sha256": digest}

    if files["tokenizer.json"]["sha256"] != expected_tokenizer_sha256:
        _fail(f"{label} tokenizer.json is not the tokenizer frozen by the contract")
    config = _read_json(path / "config.json", f"{label}/config.json", _MAX_CONFIG_BYTES)
    tokenizer_config = _read_json(
        path / "tokenizer_config.json",
        f"{label}/tokenizer_config.json",
        _MAX_TOKENIZER_CONFIG_BYTES,
    )
    _forbid_remote_code(tokenizer_config, f"{label}/tokenizer_config.json")
    _validate_bundle_config(config, precision_profile, label)
    index = _read_json(
        path / "model.safetensors.index.json",
        f"{label}/model.safetensors.index.json",
        _MAX_INDEX_BYTES,
    )
    _validate_index(index, shard_names, label)
    manifest_hash = VALIDATOR.object_sha256(files)
    return {
        "path": str(path),
        "precision_profile": precision_profile,
        "files": files,
        "file_count": len(files),
        "total_bytes": sum(record["size"] for record in files.values()),
        "manifest_sha256": manifest_hash,
    }


def _runtime_identity() -> dict[str, object]:
    observed_python = platform.python_version()
    implementation = platform.python_implementation()
    if implementation != "CPython" or observed_python != PINNED_PYTHON_VERSION:
        _fail(
            f"Python runtime must be CPython {PINNED_PYTHON_VERSION}, "
            f"observed {implementation} {observed_python}"
        )
    packages = {}
    for distribution, expected in PINNED_PACKAGES.items():
        try:
            observed = metadata.version(distribution)
        except metadata.PackageNotFoundError as error:
            raise ProducerError(
                f"required package is unavailable: {distribution}"
            ) from error
        if observed != expected:
            _fail(f"{distribution} must be {expected}, observed {observed}")
        packages[distribution] = observed
    executable = Path(sys.executable)
    try:
        executable = executable.resolve(strict=True)
    except OSError as error:
        raise ProducerError(f"cannot resolve Python executable: {error}") from error
    return {
        "python": {
            "implementation": implementation,
            "version": observed_python,
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
    ):
        _fail(f"runtime identity is not pinned CPython {PINNED_PYTHON_VERSION}")
    if type(packages) is not dict or set(packages) != set(PINNED_PACKAGES):
        _fail("runtime identity does not contain the exact eight-package lock")
    for package, expected in PINNED_PACKAGES.items():
        if packages.get(package) != expected:
            _fail(f"{package} must be {expected}, observed {packages.get(package)}")
    return json.loads(VALIDATOR.canonical_bytes(value))


def _load_runtime():
    try:
        import mlx.core as mx
        from mlx_lm import utils
        from mlx_lm.generate import generate_step as production_generate_step
    except (ImportError, ModuleNotFoundError) as error:
        raise ProducerError(f"cannot import pinned MLX runtime: {error}") from error
    if not callable(getattr(utils, "load", None)):
        _fail("pinned MLX-LM runtime is missing mlx_lm.utils.load")
    if (
        not callable(production_generate_step)
        or getattr(production_generate_step, "__module__", None) != "mlx_lm.generate"
    ):
        _fail("pinned runtime generate_step implementation drifted")

    class Runtime:
        load = staticmethod(utils.load)
        array = staticmethod(mx.array)
        argmax = staticmethod(mx.argmax)
        eval = staticmethod(mx.eval)
        generate_step = staticmethod(production_generate_step)

        @staticmethod
        def clear_cache():
            clear_cache = getattr(mx, "clear_cache", None)
            if callable(clear_cache):
                clear_cache()

    return Runtime()


def _enforce_offline_environment() -> dict[str, str]:
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
    return dict(sorted(_OFFLINE_ENVIRONMENT.items()))


def _run_lane(runtime, bundle: Path, prompts: list[dict[str, object]], label: str):
    try:
        loaded = runtime.load(
            str(bundle),
            tokenizer_config={
                "local_files_only": True,
                "trust_remote_code": False,
            },
            lazy=True,
        )
    except Exception as error:
        raise ProducerError(
            f"cannot load {label} through pinned local MLX-LM: {error}"
        ) from error
    if type(loaded) is not tuple or len(loaded) != 2:
        _fail(f"pinned MLX-LM returned an unexpected {label} load result")
    model, tokenizer = loaded
    del tokenizer
    records = []

    def greedy_argmax(logprobs):
        return runtime.argmax(logprobs, axis=-1)

    try:
        for prompt in prompts:
            runs = []
            hashes = []
            for repeat in range(2):
                try:
                    generated = runtime.generate_step(
                        runtime.array(prompt["prompt_token_ids"]),
                        model,
                        max_tokens=prompt["teacher_steps"],
                        sampler=greedy_argmax,
                    )
                    tokens = []
                    for item in generated:
                        if type(item) is not tuple or len(item) != 2:
                            _fail(f"{label} generate_step yielded an invalid item")
                        token = item[0]
                        runtime.eval(token)
                        value = (
                            token.item()
                            if callable(getattr(token, "item", None))
                            else token
                        )
                        if type(value) is bool or type(value) is not int:
                            _fail(f"{label} generate_step yielded a non-integer token")
                        if value < 0 or value >= 248320:
                            _fail(
                                f"{label} generate_step yielded an invalid Qwen token"
                            )
                        tokens.append(value)
                except ProducerError:
                    raise
                except Exception as error:
                    raise ProducerError(
                        f"{label} prompt {prompt['id']} repeat {repeat + 1} failed: {error}"
                    ) from error
                if len(tokens) != prompt["teacher_steps"]:
                    _fail(
                        f"{label} prompt {prompt['id']} repeat {repeat + 1} "
                        "returned an unexpected token count"
                    )
                runs.append(tokens)
                hashes.append(VALIDATOR.object_sha256(tokens))
            records.append({"runs": runs, "run_sha256s": hashes})
    finally:
        del model
        try:
            runtime.clear_cache()
        except Exception as error:
            raise ProducerError(
                f"cannot clear MLX cache after {label}: {error}"
            ) from error
    return records


def _file_identity(
    path: Path, label: str, file_hasher: Callable[[Path], str]
) -> dict[str, object]:
    path = _canonical_path(path, label, directory=False)
    observed = _direct_file_stat(path, label)
    digest = file_hasher(path)
    if type(digest) is not str or _SHA256.fullmatch(digest) is None:
        _fail(f"{label} hasher did not return lowercase SHA-256")
    after = _direct_file_stat(path, label)
    if _stable_fields(observed) != _stable_fields(after):
        _fail(f"{label} changed while it was hashed")
    return {"path": str(path), "size": observed.st_size, "sha256": digest}


def _comparison_failure_is_explicit(receipt: dict[str, object], claim: str) -> bool:
    problems = receipt.get("problems")
    expected_prefix = (
        "exact fixed-suite mismatch: "
        if claim == "fixed-suite-exact-parity"
        else "fixed-suite threshold mismatch: "
    )
    return (
        receipt.get("accepted") is False
        and receipt.get("claim") is None
        and type(problems) is list
        and bool(problems)
        and all(
            type(problem) is str and problem.startswith(expected_prefix)
            for problem in problems
        )
    )


def _publish_no_replace(path: Path, value: dict[str, object]) -> None:
    payload = VALIDATOR.canonical_bytes(value) + b"\n"
    if len(payload) > _MAX_OUTPUT_BYTES:
        _fail(f"quality evidence exceeds the {_MAX_OUTPUT_BYTES}-byte output limit")
    temporary = path.parent / (
        f".{path.name}.{os.getpid()}.{secrets.token_hex(12)}.tmp"
    )
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = None
    try:
        descriptor = os.open(temporary, flags, 0o600)
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                _fail("short write while staging quality evidence")
            view = view[written:]
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        try:
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError as error:
            raise ProducerError(
                "--output appeared during publication; no file was replaced"
            ) from error
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except ProducerError:
        raise
    except OSError as error:
        raise ProducerError(
            f"cannot publish no-replace quality evidence: {error}"
        ) from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        except OSError:
            pass


def run_quality_gate(
    *,
    contract_path: Path,
    reference_bundle: Path,
    candidate_bundle: Path,
    output_path: Path,
    candidate_id: str,
    precision_profile: str,
    requested_claim: str,
    runtime_loader: Callable[[], object] = _load_runtime,
    identity_provider: Callable[[], dict[str, object]] = _runtime_identity,
    file_hasher: Callable[[Path], str] = _stream_sha256,
) -> dict[str, object]:
    """Run the frozen gate and publish its accepted/failed-comparison envelope."""

    output = _prepare_output(output_path)
    if type(candidate_id) is not str or _CANDIDATE_ID.fullmatch(candidate_id) is None:
        _fail("--candidate-id must be a bounded canonical identifier")
    if precision_profile not in _PRECISION_PROFILES:
        _fail("--precision-profile is outside the frozen quality contract")
    if requested_claim not in _CLAIMS:
        _fail("--requested-claim is outside the frozen quality contract")
    contract_path = _canonical_path(contract_path, "--contract", directory=False)
    reference_bundle = _canonical_path(
        reference_bundle, "--reference-bundle", directory=True
    )
    candidate_bundle = _canonical_path(
        candidate_bundle, "--candidate-bundle", directory=True
    )
    if reference_bundle == candidate_bundle:
        _fail("reference and candidate bundles must be distinct direct directories")
    if output.parent in {reference_bundle, candidate_bundle}:
        _fail("--output must remain outside both bundles")

    offline_environment = _enforce_offline_environment()
    try:
        contract = VALIDATOR.load_contract(contract_path)
    except Exception as error:
        raise ProducerError(f"quality contract validation failed: {error}") from error
    contract_before = _file_identity(contract_path, "--contract", file_hasher)
    producer_path = Path(__file__).resolve()
    producer_before = _file_identity(
        producer_path, "quality evidence producer", file_hasher
    )
    validator_path = Path(VALIDATOR.__file__).resolve()
    validator_before = _file_identity(validator_path, "quality validator", file_hasher)
    expected_tokenizer_sha256 = contract["model"]["tokenizer_sha256"]

    reference_before = _snapshot_bundle(
        reference_bundle,
        "reference bundle",
        precision_profile="bf16",
        expected_tokenizer_sha256=expected_tokenizer_sha256,
        file_hasher=file_hasher,
    )
    candidate_before = _snapshot_bundle(
        candidate_bundle,
        "candidate bundle",
        precision_profile=precision_profile,
        expected_tokenizer_sha256=expected_tokenizer_sha256,
        file_hasher=file_hasher,
    )
    if (
        precision_profile == _COUNTERFACTUAL_PROFILE
        and reference_before["manifest_sha256"]
        != _COUNTERFACTUAL_REFERENCE_MANIFEST_SHA256
    ):
        _fail(
            "counterfactual gate requires the certified BF16 reference manifest"
        )
    if (
        reference_before["files"]["tokenizer.json"]
        != candidate_before["files"]["tokenizer.json"]
    ):
        _fail("reference and candidate tokenizer.json custody differs")

    runtime_before = _validate_runtime_identity(identity_provider())
    try:
        runtime = runtime_loader()
    except ProducerError:
        raise
    except Exception as error:
        raise ProducerError(f"cannot load pinned MLX runtime: {error}") from error
    required_runtime_calls = (
        "load",
        "array",
        "argmax",
        "eval",
        "generate_step",
        "clear_cache",
    )
    if any(
        not callable(getattr(runtime, name, None)) for name in required_runtime_calls
    ):
        _fail("MLX runtime API is incomplete")

    prompts = contract["suite"]["prompts"]
    reference_runs = _run_lane(runtime, reference_bundle, prompts, "BF16 reference")
    candidate_runs = _run_lane(runtime, candidate_bundle, prompts, "candidate")

    reference_after = _snapshot_bundle(
        reference_bundle,
        "reference bundle",
        precision_profile="bf16",
        expected_tokenizer_sha256=expected_tokenizer_sha256,
        file_hasher=file_hasher,
    )
    candidate_after = _snapshot_bundle(
        candidate_bundle,
        "candidate bundle",
        precision_profile=precision_profile,
        expected_tokenizer_sha256=expected_tokenizer_sha256,
        file_hasher=file_hasher,
    )
    if reference_before != reference_after:
        _fail("reference bundle changed during quality generation")
    if candidate_before != candidate_after:
        _fail("candidate bundle changed during quality generation")
    runtime_after = _validate_runtime_identity(identity_provider())
    if runtime_before != runtime_after:
        _fail("pinned runtime identity changed during quality generation")
    try:
        contract_after_value = VALIDATOR.load_contract(contract_path)
    except Exception as error:
        raise ProducerError(
            f"quality contract changed during generation: {error}"
        ) from error
    contract_after = _file_identity(contract_path, "--contract", file_hasher)
    producer_after = _file_identity(
        producer_path, "quality evidence producer", file_hasher
    )
    validator_after = _file_identity(validator_path, "quality validator", file_hasher)
    if contract != contract_after_value or contract_before != contract_after:
        _fail("quality contract changed during quality generation")
    if producer_before != producer_after:
        _fail("quality evidence producer changed during quality generation")
    if validator_before != validator_after:
        _fail("quality validator changed during quality generation")

    records = []
    for index, prompt in enumerate(prompts):
        records.append(
            {
                "prompt_id": prompt["id"],
                "prompt_token_ids": list(prompt["prompt_token_ids"]),
                "teacher_steps": prompt["teacher_steps"],
                "reference": {
                    "precision": "bf16",
                    **reference_runs[index],
                },
                "candidate": {
                    "precision_profile": precision_profile,
                    **candidate_runs[index],
                },
            }
        )
    evidence = {
        "format": "apxinf-mlx-multi-prompt-quality-evidence-v1",
        "schema_version": 1,
        "contract_sha256": contract["content_sha256"],
        "execution": json.loads(VALIDATOR.canonical_bytes(contract["generation"])),
        "candidate": {
            "candidate_id": candidate_id,
            "precision_profile": precision_profile,
            "requested_claim": requested_claim,
            "claims_general_parity": False,
        },
        "records": records,
    }
    try:
        receipt = VALIDATOR.validate_evidence(contract, evidence)
    except Exception as error:
        raise ProducerError(f"quality evidence validation failed: {error}") from error
    if receipt.get("accepted") is True:
        status = "accepted"
    elif _comparison_failure_is_explicit(receipt, requested_claim):
        status = "failed_comparison"
    else:
        _fail("validator returned a non-publishable rejection")

    envelope = {
        "format": ENVELOPE_FORMAT,
        "schema_version": 1,
        "status": status,
        "policy": {
            "network": "hf-offline-direct-local-bundles-v1",
            "remote_code": False,
            "generation": "mlx-lm-generate-step-explicit-axis-minus-one-argmax-v1",
            "publication": "same-filesystem-atomic-no-replace-v1",
            "claim_scope": "fixed-suite-only-never-general-parity-v1",
        },
        "custody": {
            "contract": {"before": contract_before, "after": contract_after},
            "producer": {"before": producer_before, "after": producer_after},
            "validator": {"before": validator_before, "after": validator_after},
            "runtime": {"before": runtime_before, "after": runtime_after},
            "offline_environment": offline_environment,
            "bundles": {
                "reference": {"before": reference_before, "after": reference_after},
                "candidate": {"before": candidate_before, "after": candidate_after},
            },
        },
        "evidence": evidence,
        "validation_receipt": receipt,
    }
    envelope["content_sha256"] = VALIDATOR.object_sha256(envelope)
    _publish_no_replace(output, envelope)
    return envelope


def main(
    argv=None,
    *,
    runtime_loader: Callable[[], object] = _load_runtime,
    identity_provider: Callable[[], dict[str, object]] = _runtime_identity,
    file_hasher: Callable[[Path], str] = _stream_sha256,
) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--reference-bundle", type=Path, required=True)
    parser.add_argument("--candidate-bundle", type=Path, required=True)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument(
        "--precision-profile", choices=sorted(_PRECISION_PROFILES), required=True
    )
    parser.add_argument("--requested-claim", choices=sorted(_CLAIMS), required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)
    try:
        envelope = run_quality_gate(
            contract_path=arguments.contract,
            reference_bundle=arguments.reference_bundle,
            candidate_bundle=arguments.candidate_bundle,
            output_path=arguments.output,
            candidate_id=arguments.candidate_id,
            precision_profile=arguments.precision_profile,
            requested_claim=arguments.requested_claim,
            runtime_loader=runtime_loader,
            identity_provider=identity_provider,
            file_hasher=file_hasher,
        )
    except ProducerError as error:
        summary = {
            "format": "apxinf-mlx-multi-prompt-quality-run-error-v1",
            "status": "error",
            "accepted": False,
            "published": False,
            "problems": [str(error)],
        }
        return_code = 2
    else:
        summary = {
            "format": "apxinf-mlx-multi-prompt-quality-run-summary-v1",
            "status": envelope["status"],
            "accepted": envelope["validation_receipt"]["accepted"],
            "published": True,
            "output": str(arguments.output),
            "content_sha256": envelope["content_sha256"],
        }
        return_code = 0 if envelope["status"] == "accepted" else 1
    sys.stdout.write(VALIDATOR.canonical_bytes(summary).decode("utf-8") + "\n")
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
