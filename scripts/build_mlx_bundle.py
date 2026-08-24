#!/usr/bin/env python3
"""Build or verify a deterministic, offline MLX bundle for Qwen3.5.

The builder deliberately supports one audited source shape and three explicit
quality tiers.  It does not call ``mlx_lm.convert``: the mixed-precision tier
loads the checkpoint and saves it without a blanket dtype cast, while the two
quantized tiers call the pinned ``mlx_lm.utils.quantize_model`` API directly.
An optional frozen policy can selectively override the W4 tier with W8 or BF16
modules, but only behind pre-save and post-save-reload 128-step production gates.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager, redirect_stderr, redirect_stdout
import ctypes
from dataclasses import dataclass
import errno
import hashlib
import importlib.util
from importlib import metadata
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import shutil
import socket
import stat
import struct
import sys
import tempfile
from typing import Callable, Iterator, NoReturn


RECEIPT_FORMAT = "apxinf-mlx-bundle-build-receipt-v1"
ERROR_FORMAT = "apxinf-mlx-bundle-build-error-v1"
SUPPORTED_MODEL_TYPE = "qwen3_5"
SUPPORTED_TEXT_MODEL_TYPE = "qwen3_5_text"
MODES = ("mixed-bf16", "affine-w8-g64", "affine-w4-g64")
HYBRID_PRESET = "qwen35-0.8b-affine-w8-g64-gdn-outproj-parity-v1"
HYBRID_PRESET_V2 = "qwen35-0.8b-affine-w8-g64-gdn-outproj-async-chat-parity-v2"
HYBRID_COUNTERFACTUAL_PRESET_V3 = (
    "qwen35-0.8b-affine-w8-g64-gdn3-l19-o-proj-chinese-counterfactual-v3"
)
HYBRID_SOURCE_REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
HYBRID_POLICY_PATH = (
    Path(__file__).resolve().parents[1] / "doc/20260823-qwen35-macos-bringup/"
    "qwen35-0.8b-mlx-w8-g64-parity-policy-evidence-v1.json"
)
# SHA-256 of the canonical ``policy`` object in HYBRID_POLICY_PATH.
HYBRID_POLICY_SHA256 = (
    "560f9b3df77a650603d91ff2ed60c0a56761f2d3fc408296be0a87a2f13e65cf"
)
HYBRID_POLICY_PATH_V2 = (
    Path(__file__).resolve().parents[1] / "doc/20260823-qwen35-macos-bringup/"
    "qwen35-0.8b-mlx-w8-g64-async-chat-parity-policy-evidence-v2.json"
)
# SHA-256 of the canonical ``policy`` object in HYBRID_POLICY_PATH_V2.
HYBRID_POLICY_SHA256_V2 = (
    "64a2ba1741fd5a76a7e72580ce9188d1554e1488ce6504b20054bf42479eaf8f"
)
HYBRID_COUNTERFACTUAL_POLICY_PATH_V3 = (
    Path(__file__).resolve().parents[1] / "doc/20260823-qwen35-macos-bringup/"
    "qwen35-0.8b-mlx-w8-g64-chinese-top1-counterfactual-policy-v3.json"
)
# SHA-256 of the canonical ``policy`` object in the v3 counterfactual profile.
# The fixture suite patches this value with an independently constructed policy.
HYBRID_COUNTERFACTUAL_POLICY_SHA256_V3 = (
    "7030fe5a7c4dd55cbf158750e9da3a67c7f8e65944b8f8835c75b1093e12eec9"
)
HYBRID_COUNTERFACTUAL_DIAGNOSTIC_PATH = (
    Path(__file__).resolve().parents[1] / "doc/20260823-qwen35-macos-bringup/"
    "qwen35-hybrid-w8-bf16-g64-chinese-state-aligned-diagnostic-v1.json"
)
HYBRID_COUNTERFACTUAL_DIAGNOSTIC_ARTIFACT_SHA256 = (
    "1b30a3a7f6d609a8265112bde3189b7638a9072561530b852ec86dbc4794b73d"
)
HYBRID_COUNTERFACTUAL_DIAGNOSTIC_CONTENT_SHA256 = (
    "e0c207bf46a62b643e3aeadc9398aea0d983426585d9b13ce25d21ce35d21a7f"
)
HYBRID_COUNTERFACTUAL_PARENT_MANIFEST_SHA256 = (
    "5f65527db17cd4fb7a1a46ca6ab3940a8bcc8225c620ca10d3016265d2bf8553"
)
HYBRID_COUNTERFACTUAL_REFERENCE_MANIFEST_SHA256 = (
    "fdce8bac86b1bbc888cac0139065f0291a9a57ce7f448e591b748f4baaad5dea"
)
HYBRID_COUNTERFACTUAL_SELECTED_PATH = (
    "language_model.model.layers.19.self_attn.o_proj"
)
HYBRID_COUNTERFACTUAL_RETAINED_PATHS = (
    "language_model.model.layers.12.linear_attn.out_proj",
    "language_model.model.layers.14.linear_attn.out_proj",
    HYBRID_COUNTERFACTUAL_SELECTED_PATH,
    "language_model.model.layers.20.linear_attn.out_proj",
)
# Frozen BF16 teacher trajectory certified by HYBRID_POLICY_PATH_V2.
SELECTIVE_TEACHER_IDS_SHA256 = (
    "2042d5522ed5e768b938ac3fd9e19d3936dfc6e685fc157d41a8e8132c5b42fe"
)
SELECTIVE_SOURCE_CONFIG_SHA256 = (
    "b90b86f35c8e6925ef74ee04d0e758f0a845c83a42089ad82bbaa948de9b4204"
)
SELECTIVE_SOURCE_SCHEMA_SHA256 = (
    "bdcfd67329be7cc4edfd2f85fb9c4753f3c9faab0ad7cd666d67fc995d0fbd27"
)
SELECTIVE_SOURCE_TENSOR_COUNT = 320
# Certified by configs/hf-onboarding/qwen35-0.8b-macos-cpu.json and the
# semantically hashed .apxinf/onboarding/qwen35-0.8b/source-lock.json.
SELECTIVE_SOURCE_LOCK_CONTENT_SHA256 = (
    "021209cc96e398db4aac6d126890f7bb5a5a3b5fce7204fed0328f544cbb7500"
)
# SHA-256 of the canonical six-artifact {name: {sha256, size}} source map.
SELECTIVE_SOURCE_MANIFEST_SHA256 = (
    "436821ae50e981b9176784ac6ff9548742a865d60d726c58d3bfa9f76d86b500"
)
HYBRID_CONFIG_KEY = "apxinf_hybrid_preset"
SELECTIVE_CONFIG_KEY = "apxinf_selective_mixed_policy"
SELECTIVE_POLICY_MODULE_PATH = (
    Path(__file__).resolve().with_name("mlx_mixed_quant_policy.py")
)
ASYNC_CHAT_PROMPT_IDS = (
    248045,
    846,
    198,
    9419,
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
PINNED_PACKAGES = {
    "mlx": "0.32.1",
    "mlx-metal": "0.32.1",
    "mlx-lm": "0.31.3",
    "transformers": "5.15.1",
    "safetensors": "0.8.0",
    "tokenizers": "0.22.2",
    "huggingface-hub": "1.28.0",
    "numpy": "2.5.2",
}
TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
)
SOURCE_FIXED_FILES = frozenset(
    {
        "config.json",
        "model.safetensors.index.json",
        *TOKENIZER_FILES,
    }
)
OUTPUT_FIXED_FILES = frozenset(
    {
        "README.md",
        "config.json",
        "model.safetensors.index.json",
        *TOKENIZER_FILES,
    }
)
SOURCE_SHARD = re.compile(
    r"^model(?:\.safetensors|\.safetensors-[0-9]{5}-of-[0-9]{5}\.safetensors)$"
)
OUTPUT_SHARD = re.compile(
    r"^model(?:\.safetensors|-[0-9]{5}-of-[0-9]{5}\.safetensors)$"
)
MAX_JSON_BYTES = 64 * 1024 * 1024
MAX_TOKENIZER_BYTES = 64 * 1024 * 1024
MAX_TEMPLATE_BYTES = 8 * 1024 * 1024
MAX_SAFETENSORS_HEADER_BYTES = 64 * 1024 * 1024
HASH_CHUNK_BYTES = 4 * 1024 * 1024
SAFETENSORS_DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E4M3FN": 1,
    "F8_E5M2": 1,
    "F8_E8M0": 1,
    "U16": 2,
    "I16": 2,
    "F16": 2,
    "BF16": 2,
    "U32": 4,
    "I32": 4,
    "F32": 4,
    "U64": 8,
    "I64": 8,
    "F64": 8,
}
RENAME_EXCL = 0x00000004
RENAME_NOREPLACE = 1


class BundleError(ValueError):
    """A deterministic, user-facing bundle contract failure."""


def _load_selective_policy_module() -> object:
    module_name = "_apxinf_mlx_mixed_quant_policy"
    loaded = sys.modules.get(module_name)
    if loaded is not None:
        return loaded
    spec = importlib.util.spec_from_file_location(
        module_name, SELECTIVE_POLICY_MODULE_PATH
    )
    if spec is None or spec.loader is None:
        raise BundleError("cannot load the selective mixed policy validator")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as error:
        sys.modules.pop(module_name, None)
        raise BundleError(
            f"cannot load the selective mixed policy validator: {error}"
        ) from error
    return module


class ReceiptArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        raise BundleError(f"invalid arguments: {message}")


@dataclass(frozen=True)
class FileRecord:
    path: str
    size: int
    sha256: str
    device: int
    inode: int
    mtime_ns: int
    ctime_ns: int

    def public(self) -> dict[str, object]:
        return {"sha256": self.sha256, "size": self.size}


@dataclass(frozen=True)
class SourceBundle:
    directory: Path
    config: dict[str, object]
    records: dict[str, FileRecord]
    tokenizer_payloads: dict[str, bytes]
    tensor_schema: dict[str, tuple[str, tuple[int, ...]]]


@dataclass(frozen=True)
class MlxApi:
    load: Callable[..., object]
    quantize_model: Callable[..., object]
    save: Callable[..., object]
    array: Callable[..., object]
    argmax: Callable[..., object]
    generate_step: Callable[..., object]
    teacher_forced_step: Callable[..., object]


def _reject_constant(value: str) -> NoReturn:
    raise BundleError(f"non-finite JSON number is forbidden: {value}")


def _object_without_duplicates(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise BundleError(f"duplicate JSON key is forbidden: {key}")
        result[key] = value
    return result


def _parse_json(payload: bytes, label: str) -> dict[str, object]:
    if not payload:
        raise BundleError(f"{label} is empty")
    if len(payload) > MAX_JSON_BYTES:
        raise BundleError(f"{label} exceeds {MAX_JSON_BYTES} bytes")
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_object_without_duplicates,
            parse_constant=_reject_constant,
        )
    except UnicodeDecodeError as error:
        raise BundleError(f"{label} is not UTF-8") from error
    except json.JSONDecodeError as error:
        raise BundleError(
            f"{label} is not valid JSON at line {error.lineno} column {error.colno}"
        ) from error
    if type(value) is not dict:
        raise BundleError(f"{label} must contain one JSON object")
    return value


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _manifest_sha256(records: dict[str, FileRecord]) -> str:
    public = {name: records[name].public() for name in sorted(records)}
    return hashlib.sha256(_canonical_bytes(public)).hexdigest()


def _language_schema_sha256(
    schema: dict[str, tuple[str, tuple[int, ...]]],
) -> str:
    public = {
        name: [dtype, list(shape)]
        for name, (dtype, shape) in sorted(_canonical_language_schema(schema).items())
    }
    return hashlib.sha256(_canonical_bytes(public)).hexdigest()


def _token_ids_sha256(token_ids: list[int]) -> str:
    return hashlib.sha256(_canonical_bytes(token_ids)).hexdigest()


def _hybrid_preset_spec(preset: str) -> dict[str, object]:
    if preset == HYBRID_PRESET:
        return {
            "document_format": "apxinf-qwen35-mlx-hybrid-policy-evidence-v1",
            "policy_path": HYBRID_POLICY_PATH,
            "policy_sha256": HYBRID_POLICY_SHA256,
            "quality_tier": "certified-frozen-hello-128-parity-preset-v1",
            "runtime_quality_gate": False,
        }
    if preset == HYBRID_PRESET_V2:
        return {
            "document_format": "apxinf-qwen35-mlx-hybrid-policy-evidence-v2",
            "policy_path": HYBRID_POLICY_PATH_V2,
            "policy_sha256": HYBRID_POLICY_SHA256_V2,
            "quality_tier": (
                "certified-canonical-chat-async-generate-step-parity-preset-v2"
            ),
            "runtime_quality_gate": True,
            "counterfactual": False,
        }
    if preset == HYBRID_COUNTERFACTUAL_PRESET_V3:
        return {
            "document_format": (
                "apxinf-qwen35-mlx-hybrid-counterfactual-policy-v3"
            ),
            "policy_path": HYBRID_COUNTERFACTUAL_POLICY_PATH_V3,
            "policy_sha256": HYBRID_COUNTERFACTUAL_POLICY_SHA256_V3,
            "quality_tier": "diagnostic-chinese-top1-counterfactual-only-v1",
            "runtime_quality_gate": True,
            "counterfactual": True,
            "retained_bf16_paths": list(HYBRID_COUNTERFACTUAL_RETAINED_PATHS),
        }
    raise BundleError(f"unsupported preset: {preset}")


def _validate_v2_quality_policy(policy: dict[str, object]) -> dict[str, object]:
    gate = policy.get("quality_gate")
    if type(gate) is not dict or set(gate) != {
        "api",
        "semantics",
        "prompt_token_ids",
        "teacher_token_ids",
        "teacher_ids_sha256",
        "teacher_steps",
        "free_run_steps",
        "first100_free_run_sha256",
        "repeat_count",
    }:
        raise BundleError("hybrid v2 quality-gate contract drifted")
    teacher = gate.get("teacher_token_ids")
    if (
        gate.get("api") != "mlx_lm.generate.generate_step"
        or gate.get("semantics") != "mlx-generate-step-argmax-v1"
        or gate.get("prompt_token_ids") != list(ASYNC_CHAT_PROMPT_IDS)
        or gate.get("teacher_steps") != 128
        or gate.get("free_run_steps") != 100
        or gate.get("repeat_count") != 2
        or type(teacher) is not list
        or len(teacher) != 128
        or any(type(token) is not int or token < 0 for token in teacher)
    ):
        raise BundleError("hybrid v2 quality-gate semantics drifted")
    if gate.get("teacher_ids_sha256") != _token_ids_sha256(teacher):
        raise BundleError("hybrid v2 teacher trajectory hash drifted")
    if gate.get("first100_free_run_sha256") != _token_ids_sha256(teacher[:100]):
        raise BundleError("hybrid v2 free-run trajectory hash drifted")
    auxiliary = policy.get("auxiliary_raw_prompt_gate")
    if auxiliary != {
        "admission": False,
        "prompt_token_ids": [9419],
        "scope": "legacy-v1-manual-full-prompt",
        "superseded_policy_sha256": HYBRID_POLICY_SHA256,
    }:
        raise BundleError("hybrid v2 auxiliary raw-prompt contract drifted")
    return gate


def _validate_counterfactual_quality_policy(
    policy: dict[str, object],
) -> dict[str, object]:
    gate = policy.get("quality_gate")
    if type(gate) is not dict or set(gate) != {
        "format",
        "api",
        "semantics",
        "prompt_token_ids",
        "teacher_token_ids",
        "teacher_ids_sha256",
        "teacher_steps",
        "free_run_steps",
        "free_run_ids_sha256",
        "repeat_count",
    }:
        raise BundleError("counterfactual canonical quality-gate contract drifted")
    teacher = gate.get("teacher_token_ids")
    if (
        gate.get("format")
        != "apxinf-qwen35-mlx-counterfactual-canonical-gate-v1"
        or gate.get("api") != "mlx_lm.generate.generate_step"
        or gate.get("semantics") != "mlx-generate-step-argmax-v1"
        or gate.get("prompt_token_ids") != list(ASYNC_CHAT_PROMPT_IDS)
        or gate.get("teacher_steps") != 128
        or gate.get("free_run_steps") != 128
        or gate.get("repeat_count") != 2
        or type(teacher) is not list
        or len(teacher) != 128
        or any(type(token) is not int or token < 0 for token in teacher)
    ):
        raise BundleError("counterfactual canonical quality-gate semantics drifted")
    teacher_sha256 = _token_ids_sha256(teacher)
    if (
        gate.get("teacher_ids_sha256") != teacher_sha256
        or gate.get("free_run_ids_sha256") != teacher_sha256
    ):
        raise BundleError("counterfactual canonical trajectory hash drifted")
    auxiliary = policy.get("auxiliary_raw_prompt_gate")
    if auxiliary != {
        "admission": False,
        "prompt_token_ids": [9419],
        "scope": "legacy-v1-manual-full-prompt",
        "superseded_policy_sha256": HYBRID_POLICY_SHA256,
    }:
        raise BundleError("counterfactual auxiliary raw-prompt contract drifted")
    return gate


def _counterfactual_lineage_contract() -> dict[str, object]:
    return {
        "format": "apxinf-qwen35-mlx-hybrid-counterfactual-lineage-v1",
        "status": "unvalidated-candidate",
        "selection": {
            "causal_attribution": False,
            "current_tier": "w8",
            "path": HYBRID_COUNTERFACTUAL_SELECTED_PATH,
            "proposed_tier": "bf16",
            "rank": 1,
            "ranking_metric": "same-bf16-input-relative-l1-error-ppm-v1",
            "selection_basis": (
                "trusted-diagnostic-trigger-only-not-causal-proof-v1"
            ),
        },
        "diagnostic": {
            "artifact_path": (
                "doc/20260823-qwen35-macos-bringup/"
                "qwen35-hybrid-w8-bf16-g64-chinese-state-aligned-"
                "diagnostic-v1.json"
            ),
            "artifact_sha256": (
                HYBRID_COUNTERFACTUAL_DIAGNOSTIC_ARTIFACT_SHA256
            ),
            "content_sha256": HYBRID_COUNTERFACTUAL_DIAGNOSTIC_CONTENT_SHA256,
            "format": "apxinf-mlx-chinese-hybrid-diagnostic-receipt-v1",
        },
        "parent": {
            "bundle_manifest_sha256": (
                HYBRID_COUNTERFACTUAL_PARENT_MANIFEST_SHA256
            ),
            "policy_sha256": HYBRID_POLICY_SHA256_V2,
            "preset": HYBRID_PRESET_V2,
        },
        "reference": {
            "bundle_manifest_sha256": (
                HYBRID_COUNTERFACTUAL_REFERENCE_MANIFEST_SHA256
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
    }


def _validate_counterfactual_diagnostic() -> FileRecord:
    path = HYBRID_COUNTERFACTUAL_DIAGNOSTIC_PATH
    record = _hash_regular(path, "counterfactual diagnostic")
    if record.sha256 != HYBRID_COUNTERFACTUAL_DIAGNOSTIC_ARTIFACT_SHA256:
        raise BundleError("counterfactual diagnostic artifact hash drifted")
    document = _parse_json(
        _read_regular(
            path,
            "counterfactual diagnostic",
            MAX_JSON_BYTES,
            expected=record,
        ),
        "counterfactual diagnostic",
    )
    content_sha256 = document.get("content_sha256")
    body = dict(document)
    body.pop("content_sha256", None)
    if (
        content_sha256 != HYBRID_COUNTERFACTUAL_DIAGNOSTIC_CONTENT_SHA256
        or hashlib.sha256(_canonical_bytes(body)).hexdigest() != content_sha256
    ):
        raise BundleError("counterfactual diagnostic content hash drifted")
    inputs = document.get("inputs")
    localization = document.get("module_localization")
    candidates = localization.get("top_candidates") if type(localization) is dict else None
    selected = candidates[0] if type(candidates) is list and candidates else None
    if (
        document.get("format")
        != "apxinf-mlx-chinese-hybrid-diagnostic-receipt-v1"
        or document.get("status") != "diagnostic-only"
        or type(inputs) is not dict
        or inputs.get("candidate_manifest_sha256")
        != HYBRID_COUNTERFACTUAL_PARENT_MANIFEST_SHA256
        or inputs.get("reference_manifest_sha256")
        != HYBRID_COUNTERFACTUAL_REFERENCE_MANIFEST_SHA256
        or type(localization) is not dict
        or localization.get("ranking_metric")
        != "same-bf16-input-relative-l1-error-ppm-v1"
        or type(selected) is not dict
        or selected.get("rank") != 1
        or selected.get("path") != HYBRID_COUNTERFACTUAL_SELECTED_PATH
        or selected.get("current_tier") != "w8"
        or selected.get("proposed_tier") != "bf16"
    ):
        raise BundleError("counterfactual diagnostic selection contract drifted")
    return record


def _assert_hybrid_evidence_unchanged(hybrid: dict[str, object] | None) -> None:
    if hybrid is None:
        return
    policy_record = hybrid.get("policy_record")
    policy_path = hybrid.get("policy_path")
    if policy_record is not None:
        if not isinstance(policy_record, FileRecord) or not isinstance(
            policy_path, Path
        ):
            raise BundleError("hybrid policy custody record drifted")
        policy_payload = _read_regular(
            policy_path,
            "hybrid policy",
            MAX_JSON_BYTES,
            expected=policy_record,
        )
        if hashlib.sha256(policy_payload).hexdigest() != policy_record.sha256:
            raise BundleError("hybrid policy changed after validation")
    record = hybrid.get("diagnostic_record")
    if record is None:
        return
    if not isinstance(record, FileRecord):
        raise BundleError("counterfactual diagnostic custody record drifted")
    payload = _read_regular(
        HYBRID_COUNTERFACTUAL_DIAGNOSTIC_PATH,
        "counterfactual diagnostic",
        MAX_JSON_BYTES,
        expected=record,
    )
    if hashlib.sha256(payload).hexdigest() != record.sha256:
        raise BundleError("counterfactual diagnostic changed after validation")


def _hybrid_weight_ledger(
    source_schema: dict[str, tuple[str, tuple[int, ...]]],
    retained_paths: frozenset[str],
) -> dict[str, int]:
    dtype_bytes = {"BF16": 2, "F32": 4}
    language = _canonical_language_schema(source_schema)
    observed_retained: set[str] = set()
    quantized_modules = 0
    retained_modules = 0
    quantized_logical = 0
    retained_logical = 0
    quantized_bytes = 0
    retained_bytes = 0
    total_bytes = 0
    output_tensors = 0
    for name, (dtype, shape) in language.items():
        if dtype not in dtype_bytes:
            raise BundleError(f"hybrid policy cannot account dtype {dtype!r}")
        logical = 1
        for dimension in shape:
            logical *= dimension
        eligible = (
            name.endswith(".weight")
            and len(shape) == 2
            and shape[-1] > 0
            and shape[-1] % 64 == 0
        )
        base = name.removesuffix(".weight")
        if eligible and base in retained_paths:
            if dtype != "BF16":
                raise BundleError(f"hybrid retained tensor {name!r} is not BF16")
            observed_retained.add(base)
            retained_modules += 1
            retained_logical += logical
            retained_bytes += logical * 2
            total_bytes += logical * 2
            output_tensors += 1
        elif eligible:
            if dtype != "BF16":
                raise BundleError(f"hybrid quantization candidate {name!r} is not BF16")
            quantized_modules += 1
            quantized_logical += logical
            # W8 packed weights plus two BF16 group-64 parameter tensors.
            module_bytes = logical + 2 * (logical // 64) * 2
            quantized_bytes += module_bytes
            total_bytes += module_bytes
            output_tensors += 3
        else:
            total_bytes += logical * dtype_bytes[dtype]
            output_tensors += 1
    missing = sorted(retained_paths - observed_retained)
    if missing:
        raise BundleError(f"hybrid retained paths are absent or ineligible: {missing}")
    return {
        "quantized_module_count": quantized_modules,
        "retained_bf16_module_count": retained_modules,
        "quantized_logical_weight_count": quantized_logical,
        "retained_bf16_logical_weight_count": retained_logical,
        "quantized_module_parameter_bytes": quantized_bytes,
        "retained_bf16_weight_bytes": retained_bytes,
        "estimated_total_parameter_bytes": total_bytes,
        "output_tensor_count": output_tensors,
    }


def _load_hybrid_policy(
    source: SourceBundle,
    preset: str | None,
    source_revision: str | None,
    mode: str,
) -> dict[str, object] | None:
    if preset is None:
        if source_revision is not None:
            raise BundleError("--source-revision is only valid with --preset")
        return None
    spec = _hybrid_preset_spec(preset)
    if mode != "affine-w8-g64":
        raise BundleError(f"preset {preset} requires mode 'affine-w8-g64'")
    if source_revision != HYBRID_SOURCE_REVISION:
        raise BundleError(
            f"preset {preset} requires source revision {HYBRID_SOURCE_REVISION}"
        )
    policy_path = spec["policy_path"]
    if not isinstance(policy_path, Path):
        raise BundleError("hybrid policy path contract drifted")
    policy_record = _hash_regular(policy_path, "hybrid policy")
    payload = _read_regular(
        policy_path,
        "hybrid policy",
        MAX_JSON_BYTES,
        expected=policy_record,
    )
    document = _parse_json(payload, "hybrid policy")
    if (
        set(document) != {"format", "policy", "evidence"}
        or document.get("format") != spec["document_format"]
    ):
        raise BundleError("hybrid policy document contract drifted")
    policy = document.get("policy")
    if type(policy) is not dict:
        raise BundleError("hybrid policy.policy must be an object")
    policy_sha256 = hashlib.sha256(_canonical_bytes(policy)).hexdigest()
    if policy_sha256 != spec["policy_sha256"]:
        raise BundleError("hybrid policy hash drifted")
    expected_policy_fields = {
        "preset",
        "source",
        "quantization",
        "retained_bf16_paths",
        "ledger",
    }
    if spec["runtime_quality_gate"]:
        expected_policy_fields |= {"quality_gate", "auxiliary_raw_prompt_gate"}
    if spec.get("counterfactual") is True:
        expected_policy_fields.add("counterfactual")
    if set(policy) != expected_policy_fields:
        raise BundleError("hybrid policy fields drifted")
    if policy.get("preset") != preset:
        raise BundleError("hybrid policy preset name drifted")
    source_policy = policy.get("source")
    expected_source_fields = {
        "revision",
        "config_sha256",
        "language_schema_sha256",
        "language_tensor_count",
    }
    if spec.get("counterfactual") is True:
        expected_source_fields.add("source_manifest_sha256")
    if type(source_policy) is not dict or set(source_policy) != expected_source_fields:
        raise BundleError("hybrid policy source contract drifted")
    if source_policy.get("revision") != source_revision:
        raise BundleError("hybrid policy source revision drifted")
    if source_policy.get("config_sha256") != source.records["config.json"].sha256:
        raise BundleError("hybrid preset source config drifted")
    language = _canonical_language_schema(source.tensor_schema)
    if source_policy.get("language_tensor_count") != len(language):
        raise BundleError("hybrid preset source language tensor count drifted")
    if source_policy.get("language_schema_sha256") != _language_schema_sha256(
        source.tensor_schema
    ):
        raise BundleError("hybrid preset source tensor schema drifted")
    if spec.get("counterfactual") is True and source_policy.get(
        "source_manifest_sha256"
    ) != _manifest_sha256(source.records):
        raise BundleError("counterfactual source artifact manifest drifted")
    if policy.get("quantization") != {
        "bits": 8,
        "group_size": 64,
        "mode": "affine",
    }:
        raise BundleError("hybrid policy quantization contract drifted")
    raw_retained = policy.get("retained_bf16_paths")
    if (
        type(raw_retained) is not list
        or not raw_retained
        or any(type(path) is not str or not path for path in raw_retained)
        or raw_retained != sorted(set(raw_retained))
    ):
        raise BundleError("hybrid policy retained path set drifted")
    retained = frozenset(raw_retained)
    expected_retained = spec.get("retained_bf16_paths")
    if expected_retained is not None and raw_retained != expected_retained:
        raise BundleError("counterfactual retained BF16 path portfolio drifted")
    ledger = _hybrid_weight_ledger(source.tensor_schema, retained)
    if policy.get("ledger") != ledger:
        raise BundleError("hybrid policy weight ledger drifted")
    quality_gate = None
    if spec["runtime_quality_gate"]:
        quality_gate = (
            _validate_counterfactual_quality_policy(policy)
            if spec.get("counterfactual") is True
            else _validate_v2_quality_policy(policy)
        )
    counterfactual = None
    diagnostic_record = None
    if spec.get("counterfactual") is True:
        expected_counterfactual = _counterfactual_lineage_contract()
        if policy.get("counterfactual") != expected_counterfactual:
            raise BundleError("counterfactual lineage contract drifted")
        diagnostic_record = _validate_counterfactual_diagnostic()
        counterfactual = expected_counterfactual
    result = {
        "name": preset,
        "policy_sha256": policy_sha256,
        "source_revision": source_revision,
        "retained_bf16_paths": list(raw_retained),
        "weight_ledger": ledger,
        "quality_tier": spec["quality_tier"],
        "quality_gate": quality_gate,
        "policy_path": policy_path,
        "policy_record": policy_record,
    }
    if counterfactual is not None:
        result["counterfactual"] = counterfactual
        result["diagnostic_record"] = diagnostic_record
    return result


def _selective_candidate_modules(
    source_schema: dict[str, tuple[str, tuple[int, ...]]],
) -> list[dict[str, object]]:
    candidates: list[dict[str, object]] = []
    for name, (dtype, shape) in sorted(
        _canonical_language_schema(source_schema).items()
    ):
        eligible = (
            name.endswith(".weight")
            and len(shape) == 2
            and shape[-1] > 0
            and shape[-1] % 64 == 0
        )
        if not eligible:
            continue
        if dtype != "BF16":
            raise BundleError(
                f"selective source quantization candidate {name!r} is not BF16"
            )
        candidates.append(
            {
                "path": name.removesuffix(".weight"),
                "dtype": dtype,
                "shape": list(shape),
            }
        )
    if not candidates:
        raise BundleError("selective source has no affine group-64 candidates")
    return candidates


def _selective_weight_ledger(
    source_schema: dict[str, tuple[str, tuple[int, ...]]],
    tiers: dict[str, str],
) -> dict[str, int]:
    dtype_bytes = {"BF16": 2, "F32": 4}
    candidate_paths = frozenset(tiers)
    observed: set[str] = set()
    module_counts = {"w4": 0, "w8": 0, "bf16": 0}
    logical_counts = {"w4": 0, "w8": 0, "bf16": 0}
    parameter_bytes = {"w4": 0, "w8": 0, "bf16": 0}
    total_bytes = 0
    output_tensors = 0
    for name, (dtype, shape) in _canonical_language_schema(source_schema).items():
        if dtype not in dtype_bytes:
            raise BundleError(f"selective policy cannot account dtype {dtype!r}")
        logical = 1
        for dimension in shape:
            logical *= dimension
        path = name.removesuffix(".weight")
        if path not in candidate_paths:
            total_bytes += logical * dtype_bytes[dtype]
            output_tensors += 1
            continue
        if dtype != "BF16":
            raise BundleError(f"selective candidate {name!r} is not BF16")
        observed.add(path)
        tier = tiers[path]
        module_counts[tier] += 1
        logical_counts[tier] += logical
        if tier == "bf16":
            size = logical * 2
            output_tensors += 1
        else:
            bits = 4 if tier == "w4" else 8
            size = logical * bits // 8 + 2 * (logical // 64) * 2
            output_tensors += 3
        parameter_bytes[tier] += size
        total_bytes += size
    missing = sorted(candidate_paths - observed)
    if missing:
        raise BundleError(f"selective candidate paths are absent: {missing[:3]}")
    return {
        "w4_module_count": module_counts["w4"],
        "w8_module_count": module_counts["w8"],
        "retained_bf16_module_count": module_counts["bf16"],
        "w4_logical_weight_count": logical_counts["w4"],
        "w8_logical_weight_count": logical_counts["w8"],
        "retained_bf16_logical_weight_count": logical_counts["bf16"],
        "w4_parameter_bytes": parameter_bytes["w4"],
        "w8_parameter_bytes": parameter_bytes["w8"],
        "retained_bf16_weight_bytes": parameter_bytes["bf16"],
        "estimated_total_parameter_bytes": total_bytes,
        "output_tensor_count": output_tensors,
    }


def _selective_config_manifest(selective: dict[str, object]) -> dict[str, object]:
    return {
        "format": "apxinf-mlx-selective-mixed-policy-manifest-v1",
        "policy_sha256": selective["policy_sha256"],
        "policy_document_sha256": selective["policy_document_sha256"],
        "search_receipt_sha256": selective["search_receipt_sha256"],
        "search_status": selective["search_status"],
        "source_repo_id": selective["source_repo_id"],
        "source_revision": selective["source_revision"],
        "source_lock_content_sha256": selective["source_lock_content_sha256"],
        "source_manifest_sha256": selective["source_manifest_sha256"],
        "candidate_modules_sha256": selective["candidate_modules_sha256"],
        "candidate_module_count": len(selective["candidate_modules"]),
        "w8_paths": selective["w8_paths"],
        "retained_bf16_paths": selective["retained_bf16_paths"],
        "trace_sha256": selective["trace_sha256"],
        "weight_ledger": selective["weight_ledger"],
    }


def _load_selective_policy(
    source: SourceBundle,
    mixed_policy: str | None,
    source_revision: str | None,
    mode: str,
) -> dict[str, object] | None:
    if mixed_policy is None:
        return None
    if mode != "affine-w4-g64":
        raise BundleError("--mixed-policy requires mode 'affine-w4-g64'")
    if source_revision is None:
        raise BundleError("--mixed-policy requires --source-revision")
    if source_revision != HYBRID_SOURCE_REVISION:
        raise BundleError(
            f"--mixed-policy requires source revision {HYBRID_SOURCE_REVISION}"
        )
    argument = _require_absolute(mixed_policy, "--mixed-policy")
    parent = _require_owned_directory(argument.parent, "mixed policy parent")
    policy_path = parent / argument.name
    payload = _read_regular(policy_path, "mixed policy", MAX_JSON_BYTES)
    document = _parse_json(payload, "mixed policy")
    policy_api = _load_selective_policy_module()
    try:
        validated = policy_api.validate_policy_document(document)
    except Exception as error:
        raise BundleError(f"mixed policy validation failed: {error}") from error
    policy = validated["policy"]
    policy_sha256 = validated["policy_sha256"]
    search_receipt = validated["search_receipt"]
    source_policy = policy["source"]
    if source_policy["repo_id"] != "Qwen/Qwen3.5-0.8B":
        raise BundleError("mixed policy repo_id is not Qwen/Qwen3.5-0.8B")
    if source_policy["revision"] != source_revision:
        raise BundleError("mixed policy source revision drifted")
    if (
        source_policy["config_sha256"] != SELECTIVE_SOURCE_CONFIG_SHA256
        or source_policy["language_schema_sha256"] != SELECTIVE_SOURCE_SCHEMA_SHA256
        or source_policy["language_tensor_count"] != SELECTIVE_SOURCE_TENSOR_COUNT
    ):
        raise BundleError(
            "mixed policy source is not the certified Qwen3.5-0.8B v2 schema"
        )
    source_manifest_sha256 = _manifest_sha256(source.records)
    if (
        source_policy["source_manifest_sha256"] != SELECTIVE_SOURCE_MANIFEST_SHA256
        or source_manifest_sha256 != SELECTIVE_SOURCE_MANIFEST_SHA256
    ):
        raise BundleError("mixed policy source is not the certified source manifest")
    if source_policy["config_sha256"] != source.records["config.json"].sha256:
        raise BundleError("mixed policy source config drifted")
    language = _canonical_language_schema(source.tensor_schema)
    if source_policy["language_tensor_count"] != len(language):
        raise BundleError("mixed policy source language tensor count drifted")
    if source_policy["language_schema_sha256"] != _language_schema_sha256(
        source.tensor_schema
    ):
        raise BundleError("mixed policy source language schema drifted")
    candidates = _selective_candidate_modules(source.tensor_schema)
    if policy["candidate_modules"] != candidates:
        raise BundleError("mixed policy frozen candidate module set drifted")
    overrides = {
        override["path"]: override["tier"]
        for override in policy["quantization"]["overrides"]
    }
    tiers = {
        candidate["path"]: overrides.get(candidate["path"], "w4")
        for candidate in candidates
    }
    ledger = _selective_weight_ledger(source.tensor_schema, tiers)
    trace = policy["trace"]
    if trace["teacher_ids_sha256"] != SELECTIVE_TEACHER_IDS_SHA256:
        raise BundleError(
            "mixed policy teacher trajectory is not the frozen BF16 v2 teacher"
        )
    selective = {
        "policy_path": str(policy_path),
        "policy_sha256": policy_sha256,
        "policy_document_sha256": policy_api.object_sha256(validated),
        "search_receipt": search_receipt,
        "search_receipt_sha256": validated["search_receipt_sha256"],
        "search_status": search_receipt["status"],
        "source_repo_id": source_policy["repo_id"],
        "source_revision": source_revision,
        "source_lock_content_sha256": SELECTIVE_SOURCE_LOCK_CONTENT_SHA256,
        "source_manifest_sha256": source_manifest_sha256,
        "candidate_modules": candidates,
        "candidate_modules_sha256": policy["candidate_modules_sha256"],
        "tiers": tiers,
        "w4_paths": sorted(path for path, tier in tiers.items() if tier == "w4"),
        "w8_paths": sorted(path for path, tier in tiers.items() if tier == "w8"),
        "retained_bf16_paths": sorted(
            path for path, tier in tiers.items() if tier == "bf16"
        ),
        "trace": trace,
        "trace_sha256": policy_api.object_sha256(trace),
        "weight_ledger": ledger,
        "quality_tier": "runtime-canonical-chat-128-exact-candidate-v1",
    }
    selective["config_manifest"] = _selective_config_manifest(selective)
    return selective


def _require_absolute(argument: str, label: str) -> Path:
    path = Path(argument)
    if not path.is_absolute():
        raise BundleError(f"{label} must be an absolute path")
    return path


def _require_owned_directory(path: Path, label: str) -> Path:
    """Resolve a directory only after rejecting symlinks in every component."""

    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        try:
            info = current.lstat()
        except OSError as error:
            raise BundleError(
                f"cannot inspect {label} component {current}: {error.strerror or error}"
            ) from error
        if stat.S_ISLNK(info.st_mode):
            raise BundleError(f"{label} contains a symlink component: {current}")
        if not stat.S_ISDIR(info.st_mode):
            raise BundleError(f"{label} component is not a directory: {current}")
        if current == path and info.st_uid != os.getuid():
            raise BundleError(f"{label} is not owned by the current uid: {current}")
    return path.resolve(strict=True)


def _require_output_absent(output_dir: Path) -> None:
    try:
        output_dir.lstat()
    except FileNotFoundError:
        return
    except OSError as error:
        raise BundleError(
            f"cannot inspect output directory: {error.strerror or error}"
        ) from error
    raise BundleError("output directory already exists; replacement is forbidden")


def _paths_overlap(source: Path, output: Path) -> bool:
    try:
        source.relative_to(output)
        return True
    except ValueError:
        pass
    try:
        output.relative_to(source)
        return True
    except ValueError:
        return False


def _open_regular(path: Path, label: str) -> tuple[int, os.stat_result]:
    try:
        before = path.lstat()
    except OSError as error:
        raise BundleError(
            f"cannot inspect {label}: {error.strerror or error}"
        ) from error
    if (
        stat.S_ISLNK(before.st_mode)
        or not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or before.st_uid != os.getuid()
    ):
        raise BundleError(
            f"{label} must be a current-uid, single-link regular non-symlink file"
        )
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise BundleError(f"cannot open {label}: {error.strerror or error}") from error
    opened = os.fstat(descriptor)
    if (
        not stat.S_ISREG(opened.st_mode)
        or opened.st_nlink != 1
        or opened.st_uid != os.getuid()
        or opened.st_dev != before.st_dev
        or opened.st_ino != before.st_ino
    ):
        os.close(descriptor)
        raise BundleError(f"{label} changed while it was opened")
    return descriptor, opened


def _stat_matches_record(observed: os.stat_result, expected: FileRecord) -> bool:
    return (
        observed.st_dev == expected.device
        and observed.st_ino == expected.inode
        and observed.st_size == expected.size
        and observed.st_mtime_ns == expected.mtime_ns
        and observed.st_ctime_ns == expected.ctime_ns
    )


def _read_regular(
    path: Path,
    label: str,
    maximum: int,
    *,
    expected: FileRecord | None = None,
) -> bytes:
    descriptor, opened = _open_regular(path, label)
    try:
        if expected is not None and not _stat_matches_record(opened, expected):
            raise BundleError(f"{label} changed since its manifest hash was recorded")
        if opened.st_size > maximum:
            raise BundleError(f"{label} exceeds {maximum} bytes")
        chunks: list[bytes] = []
        remaining = opened.st_size
        while remaining:
            chunk = os.read(descriptor, min(HASH_CHUNK_BYTES, remaining))
            if not chunk:
                raise BundleError(f"{label} ended before its declared size")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise BundleError(f"{label} grew while it was read")
        after = os.fstat(descriptor)
        if (
            after.st_size != opened.st_size
            or after.st_mtime_ns != opened.st_mtime_ns
            or after.st_ctime_ns != opened.st_ctime_ns
        ):
            raise BundleError(f"{label} changed while it was read")
        if expected is not None and not _stat_matches_record(after, expected):
            raise BundleError(f"{label} changed since its manifest hash was recorded")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _hash_regular(path: Path, relative_path: str) -> FileRecord:
    descriptor, opened = _open_regular(path, relative_path)
    digest = hashlib.sha256()
    observed = 0
    try:
        while True:
            chunk = os.read(descriptor, HASH_CHUNK_BYTES)
            if not chunk:
                break
            observed += len(chunk)
            digest.update(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if observed != opened.st_size:
        raise BundleError(f"{relative_path} changed size while it was hashed")
    if (
        after.st_size != opened.st_size
        or after.st_mtime_ns != opened.st_mtime_ns
        or after.st_ctime_ns != opened.st_ctime_ns
    ):
        raise BundleError(f"{relative_path} changed while it was hashed")
    return FileRecord(
        path=relative_path,
        size=observed,
        sha256=digest.hexdigest(),
        device=opened.st_dev,
        inode=opened.st_ino,
        mtime_ns=opened.st_mtime_ns,
        ctime_ns=opened.st_ctime_ns,
    )


def _assert_records_current(
    directory: Path,
    records: dict[str, FileRecord],
    label: str,
    *,
    fixed_names: frozenset[str],
    shard_pattern: re.Pattern[str],
) -> None:
    current_files = _scan_flat_directory(
        directory,
        label=label,
        fixed_names=fixed_names,
        shard_pattern=shard_pattern,
    )
    if set(current_files) != set(records):
        raise BundleError(f"{label} entries changed after manifest hashing")
    for name, expected in records.items():
        try:
            observed = current_files[name].lstat()
        except OSError as error:
            raise BundleError(f"cannot re-inspect {label} {name}: {error}") from error
        if (
            stat.S_ISLNK(observed.st_mode)
            or not stat.S_ISREG(observed.st_mode)
            or observed.st_nlink != 1
            or observed.st_uid != os.getuid()
            or not _stat_matches_record(observed, expected)
        ):
            raise BundleError(f"{label} {name} changed after manifest hashing")


def _scan_flat_directory(
    directory: Path,
    *,
    label: str,
    fixed_names: frozenset[str],
    shard_pattern: re.Pattern[str],
) -> dict[str, Path]:
    files: dict[str, Path] = {}
    try:
        entries = list(os.scandir(directory))
    except OSError as error:
        raise BundleError(f"cannot scan {label}: {error.strerror or error}") from error
    for entry in entries:
        name = entry.name
        if name in (".", "..") or PurePosixPath(name).name != name:
            raise BundleError(f"{label} contains an invalid path: {name!r}")
        try:
            info = entry.stat(follow_symlinks=False)
        except OSError as error:
            raise BundleError(
                f"cannot inspect {label} entry {name}: {error.strerror or error}"
            ) from error
        if (
            entry.is_symlink()
            or not stat.S_ISREG(info.st_mode)
            or info.st_nlink != 1
            or info.st_uid != os.getuid()
        ):
            raise BundleError(
                f"{label} entry {name} is not a current-uid, single-link regular file"
            )
        if name not in fixed_names and shard_pattern.fullmatch(name) is None:
            raise BundleError(f"{label} contains unexpected entry: {name}")
        files[name] = directory / name
    missing = fixed_names - set(files)
    if missing:
        raise BundleError(f"{label} is missing required files: {sorted(missing)}")
    shard_names = sorted(name for name in files if shard_pattern.fullmatch(name))
    if not shard_names:
        raise BundleError(f"{label} contains no safetensors weight shard")
    return files


def _validate_no_remote_code(config: dict[str, object], label: str) -> None:
    forbidden = ("auto_map", "model_file", "custom_pipelines")
    for key in forbidden:
        if config.get(key) is not None:
            raise BundleError(f"{label} requests remote/custom code through {key}")


def _validate_source_config(config: dict[str, object]) -> None:
    _validate_no_remote_code(config, "source config.json")
    if config.get("model_type") != SUPPORTED_MODEL_TYPE:
        raise BundleError(
            f"source config.json model_type must be {SUPPORTED_MODEL_TYPE!r}"
        )
    architectures = config.get("architectures")
    if (
        type(architectures) is not list
        or "Qwen3_5ForConditionalGeneration" not in architectures
    ):
        raise BundleError(
            "source config.json has an unsupported architectures contract"
        )
    text_config = config.get("text_config")
    if type(text_config) is not dict:
        raise BundleError("source config.json text_config must be an object")
    _validate_no_remote_code(text_config, "source config.json text_config")
    if text_config.get("model_type") != SUPPORTED_TEXT_MODEL_TYPE:
        raise BundleError(
            f"source text_config.model_type must be {SUPPORTED_TEXT_MODEL_TYPE!r}"
        )
    if text_config.get("dtype") != "bfloat16":
        raise BundleError("source text_config.dtype must be 'bfloat16'")
    for candidate in (config, text_config):
        if (
            candidate.get("quantization") is not None
            or candidate.get("quantization_config") is not None
        ):
            raise BundleError("source checkpoint must not already be quantized")


def _parse_safetensors_schema(
    path: Path,
    label: str,
    *,
    expected: FileRecord | None = None,
) -> dict[str, tuple[str, tuple[int, ...]]]:
    descriptor, opened = _open_regular(path, label)
    try:
        if expected is not None and not _stat_matches_record(opened, expected):
            raise BundleError(f"{label} changed since its manifest hash was recorded")
        prefix = os.read(descriptor, 8)
        if len(prefix) != 8:
            raise BundleError(f"{label} is too short for a safetensors header")
        header_size = struct.unpack("<Q", prefix)[0]
        if header_size <= 1 or header_size > MAX_SAFETENSORS_HEADER_BYTES:
            raise BundleError(f"{label} has an invalid safetensors header size")
        if 8 + header_size > opened.st_size:
            raise BundleError(f"{label} safetensors header exceeds the file size")
        header_parts: list[bytes] = []
        remaining = header_size
        while remaining:
            part = os.read(descriptor, min(HASH_CHUNK_BYTES, remaining))
            if not part:
                raise BundleError(f"{label} has a truncated safetensors header")
            header_parts.append(part)
            remaining -= len(part)
        header = _parse_json(b"".join(header_parts), f"{label} safetensors header")
        after_header = os.fstat(descriptor)
        if (
            after_header.st_dev != opened.st_dev
            or after_header.st_ino != opened.st_ino
            or after_header.st_size != opened.st_size
            or after_header.st_mtime_ns != opened.st_mtime_ns
            or after_header.st_ctime_ns != opened.st_ctime_ns
        ):
            raise BundleError(f"{label} changed while reading its header")
        if expected is not None and not _stat_matches_record(after_header, expected):
            raise BundleError(f"{label} changed since its manifest hash was recorded")
    finally:
        os.close(descriptor)
    data_bytes = opened.st_size - 8 - header_size
    schema: dict[str, tuple[str, tuple[int, ...]]] = {}
    spans: list[tuple[int, int, str]] = []
    for tensor_name, raw in header.items():
        if tensor_name == "__metadata__":
            if type(raw) is not dict:
                raise BundleError(f"{label} __metadata__ must be an object")
            continue
        if type(tensor_name) is not str or not tensor_name or type(raw) is not dict:
            raise BundleError(f"{label} contains invalid tensor metadata")
        if set(raw) != {"dtype", "shape", "data_offsets"}:
            raise BundleError(f"{label} tensor {tensor_name!r} has unexpected metadata")
        dtype = raw["dtype"]
        shape = raw["shape"]
        offsets = raw["data_offsets"]
        if type(dtype) is not str or dtype not in SAFETENSORS_DTYPE_BYTES:
            raise BundleError(f"{label} tensor {tensor_name!r} has invalid dtype")
        if type(shape) is not list or any(
            type(dimension) is not int or dimension < 0 for dimension in shape
        ):
            raise BundleError(f"{label} tensor {tensor_name!r} has invalid shape")
        if (
            type(offsets) is not list
            or len(offsets) != 2
            or any(type(offset) is not int for offset in offsets)
            or offsets[0] < 0
            or offsets[0] > offsets[1]
            or offsets[1] > data_bytes
        ):
            raise BundleError(f"{label} tensor {tensor_name!r} has invalid offsets")
        logical_elements = 1
        for dimension in shape:
            logical_elements *= dimension
        expected_bytes = logical_elements * SAFETENSORS_DTYPE_BYTES[dtype]
        if offsets[1] - offsets[0] != expected_bytes:
            raise BundleError(
                f"{label} tensor {tensor_name!r} byte size does not match dtype/shape"
            )
        spans.append((offsets[0], offsets[1], tensor_name))
        schema[tensor_name] = (dtype, tuple(shape))
    if not schema:
        raise BundleError(f"{label} contains no tensors")
    cursor = 0
    for start, end, tensor_name in sorted(spans):
        if start != cursor:
            raise BundleError(
                f"{label} tensor {tensor_name!r} data ranges overlap or are not contiguous"
            )
        cursor = end
    if cursor != data_bytes:
        raise BundleError(f"{label} tensor data ranges do not cover the payload")
    return schema


def _validate_weight_bundle(
    directory: Path,
    files: dict[str, Path],
    shard_pattern: re.Pattern[str],
    label: str,
    expected_records: dict[str, FileRecord] | None = None,
) -> dict[str, tuple[str, tuple[int, ...]]]:
    shard_names = sorted(name for name in files if shard_pattern.fullmatch(name))
    all_schema: dict[str, tuple[str, tuple[int, ...]]] = {}
    tensor_shards: dict[str, str] = {}
    for name in shard_names:
        schema = _parse_safetensors_schema(
            directory / name,
            f"{label}/{name}",
            expected=expected_records[name] if expected_records is not None else None,
        )
        overlap = set(all_schema) & set(schema)
        if overlap:
            raise BundleError(
                f"{label} duplicates tensors across shards: {sorted(overlap)[:3]}"
            )
        all_schema.update(schema)
        tensor_shards.update({tensor_name: name for tensor_name in schema})
    index_payload = _read_regular(
        directory / "model.safetensors.index.json",
        f"{label}/model.safetensors.index.json",
        MAX_JSON_BYTES,
        expected=(
            expected_records["model.safetensors.index.json"]
            if expected_records is not None
            else None
        ),
    )
    index = _parse_json(index_payload, f"{label}/model.safetensors.index.json")
    weight_map = index.get("weight_map")
    if type(weight_map) is not dict or not weight_map:
        raise BundleError(f"{label} index weight_map must be a non-empty object")
    if set(weight_map) != set(all_schema):
        raise BundleError(
            f"{label} index tensor names do not match safetensors headers"
        )
    referenced: set[str] = set()
    for tensor_name, shard_name in weight_map.items():
        if type(tensor_name) is not str or type(shard_name) is not str:
            raise BundleError(f"{label} index weight_map must map strings to strings")
        if (
            PurePosixPath(shard_name).name != shard_name
            or shard_name not in shard_names
        ):
            raise BundleError(
                f"{label} index references an invalid shard: {shard_name!r}"
            )
        if tensor_shards[tensor_name] != shard_name:
            raise BundleError(
                f"{label} index maps tensor {tensor_name!r} to the wrong weight shard"
            )
        referenced.add(shard_name)
    if referenced != set(shard_names):
        raise BundleError(f"{label} contains an unreferenced weight shard")
    return all_schema


def _inspect_source(source_dir_argument: str) -> SourceBundle:
    source_argument = _require_absolute(source_dir_argument, "--source-dir")
    source_dir = _require_owned_directory(source_argument, "source directory")
    files = _scan_flat_directory(
        source_dir,
        label="source directory",
        fixed_names=SOURCE_FIXED_FILES,
        shard_pattern=SOURCE_SHARD,
    )
    records = {name: _hash_regular(files[name], name) for name in sorted(files)}
    config_payload = _read_regular(
        files["config.json"],
        "source config.json",
        MAX_JSON_BYTES,
        expected=records["config.json"],
    )
    config = _parse_json(config_payload, "source config.json")
    _validate_source_config(config)
    tokenizer_payloads = {
        "tokenizer.json": _read_regular(
            files["tokenizer.json"],
            "source tokenizer.json",
            MAX_TOKENIZER_BYTES,
            expected=records["tokenizer.json"],
        ),
        "tokenizer_config.json": _read_regular(
            files["tokenizer_config.json"],
            "source tokenizer_config.json",
            MAX_TOKENIZER_BYTES,
            expected=records["tokenizer_config.json"],
        ),
        "chat_template.jinja": _read_regular(
            files["chat_template.jinja"],
            "source chat_template.jinja",
            MAX_TEMPLATE_BYTES,
            expected=records["chat_template.jinja"],
        ),
    }
    tokenizer_config = _parse_json(
        tokenizer_payloads["tokenizer_config.json"], "source tokenizer_config.json"
    )
    _validate_no_remote_code(tokenizer_config, "source tokenizer_config.json")
    embedded_template = tokenizer_config.get("chat_template")
    try:
        template_text = tokenizer_payloads["chat_template.jinja"].decode("utf-8")
    except UnicodeDecodeError as error:
        raise BundleError("source chat_template.jinja is not UTF-8") from error
    if type(embedded_template) is not str or embedded_template != template_text:
        raise BundleError(
            "source tokenizer_config.json chat_template differs from chat_template.jinja"
        )
    tensor_schema = _validate_weight_bundle(
        source_dir, files, SOURCE_SHARD, "source directory", records
    )
    expected_text_tensors = {
        name for name in tensor_schema if name.startswith("model.language_model.")
    }
    if not expected_text_tensors:
        raise BundleError("source checkpoint contains no Qwen3.5 language tensors")
    _assert_records_current(
        source_dir,
        records,
        "source directory",
        fixed_names=SOURCE_FIXED_FILES,
        shard_pattern=SOURCE_SHARD,
    )
    return SourceBundle(
        directory=source_dir,
        config=config,
        records=records,
        tokenizer_payloads=tokenizer_payloads,
        tensor_schema=tensor_schema,
    )


def _assert_source_unchanged(source: SourceBundle) -> None:
    files = _scan_flat_directory(
        source.directory,
        label="source directory",
        fixed_names=SOURCE_FIXED_FILES,
        shard_pattern=SOURCE_SHARD,
    )
    if set(files) != set(source.records):
        raise BundleError("source directory entries changed during the build")
    for name, expected in source.records.items():
        try:
            observed = files[name].lstat()
        except OSError as error:
            raise BundleError(f"cannot re-inspect source {name}: {error}") from error
        if (
            observed.st_dev != expected.device
            or observed.st_ino != expected.inode
            or observed.st_size != expected.size
            or observed.st_mtime_ns != expected.mtime_ns
            or observed.st_ctime_ns != expected.ctime_ns
            or observed.st_nlink != 1
        ):
            raise BundleError(f"source file changed during the build: {name}")


def _runtime_versions() -> dict[str, object]:
    packages: dict[str, str] = {}
    for distribution, pinned in PINNED_PACKAGES.items():
        try:
            observed = metadata.version(distribution)
        except metadata.PackageNotFoundError as error:
            raise BundleError(
                f"required package is unavailable: {distribution}"
            ) from error
        if observed != pinned:
            raise BundleError(
                f"{distribution} version must be {pinned}, observed {observed}"
            )
        packages[distribution] = observed
    return {
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
        },
        "packages": packages,
    }


def _load_mlx_api() -> MlxApi:
    try:
        import mlx.core as mx
        from mlx_lm import utils
        from mlx_lm.generate import generate_step
    except (ImportError, ModuleNotFoundError) as error:
        raise BundleError(f"cannot import pinned MLX-LM API: {error}") from error
    for name in ("load", "quantize_model", "save"):
        if not callable(getattr(utils, name, None)):
            raise BundleError(f"pinned MLX-LM API is missing utils.{name}")

    def teacher_forced_step(
        prompt: object,
        model: object,
        teacher_ids: list[int],
    ) -> list[int]:
        observed: list[int] = []

        def force_teacher(logprobs: object) -> object:
            step = len(observed)
            if step < len(teacher_ids):
                predicted = mx.argmax(logprobs, axis=-1)
                mx.eval(predicted)
                observed.append(int(predicted.item()))
                forced = teacher_ids[step]
            else:
                # ``generate_step`` computes one look-ahead token after the last
                # yielded token.  Its prediction is outside the 128-step gate.
                forced = teacher_ids[-1]
            return mx.array([forced])

        for _item in generate_step(
            prompt,
            model,
            max_tokens=len(teacher_ids),
            sampler=force_teacher,
        ):
            pass
        return observed

    return MlxApi(
        load=utils.load,
        quantize_model=utils.quantize_model,
        save=utils.save,
        array=mx.array,
        argmax=mx.argmax,
        generate_step=generate_step,
        teacher_forced_step=teacher_forced_step,
    )


def _run_hybrid_quality_gate(
    model: object,
    api: MlxApi,
    hybrid: dict[str, object],
) -> dict[str, object] | None:
    gate = hybrid.get("quality_gate")
    if gate is None:
        return None
    if type(gate) is not dict:
        raise BundleError("hybrid runtime quality-gate contract drifted")
    prompt_ids = gate["prompt_token_ids"]
    teacher_ids = gate["teacher_token_ids"]
    if type(prompt_ids) is not list or type(teacher_ids) is not list:
        raise BundleError("hybrid runtime quality-gate tokens drifted")
    observed_repeats: list[list[int]] = []

    def greedy_argmax(logprobs: object) -> object:
        return api.argmax(logprobs, axis=-1)

    for repeat in range(gate["repeat_count"]):
        try:
            prompt = api.array(prompt_ids)
            generated = api.generate_step(
                prompt,
                model,
                max_tokens=gate["teacher_steps"],
                sampler=greedy_argmax,
            )
            observed = []
            for item in generated:
                if type(item) is not tuple or len(item) != 2:
                    raise BundleError(
                        "mlx_lm.generate_step yielded an unexpected value"
                    )
                token = item[0]
                if type(token) is bool:
                    raise BundleError("mlx_lm.generate_step yielded a boolean token")
                observed.append(int(token))
        except BundleError:
            raise
        except Exception as error:
            raise BundleError(
                f"hybrid async quality gate repeat {repeat + 1} failed: {error}"
            ) from error
        if len(observed) != gate["teacher_steps"]:
            raise BundleError(
                "hybrid async quality gate returned an unexpected token count"
            )
        errors = [
            index
            for index, (actual, expected) in enumerate(
                zip(observed, teacher_ids, strict=True)
            )
            if actual != expected
        ]
        if errors:
            raise BundleError(
                "hybrid async quality gate diverged from the BF16 teacher "
                f"at steps {errors[:8]}"
            )
        first100 = observed[: gate["free_run_steps"]]
        if _token_ids_sha256(first100) != gate["first100_free_run_sha256"]:
            raise BundleError("hybrid async quality gate free-run hash drifted")
        observed_repeats.append(observed)
    if any(observed != observed_repeats[0] for observed in observed_repeats[1:]):
        raise BundleError("hybrid async quality gate repeats were not identical")
    return {
        "api": gate["api"],
        "semantics": gate["semantics"],
        "prompt_token_ids": prompt_ids,
        "teacher_steps": gate["teacher_steps"],
        "teacher_exact": gate["teacher_steps"],
        "teacher_ids_sha256": gate["teacher_ids_sha256"],
        "first100_free_run_sha256": gate["first100_free_run_sha256"],
        "repeat_count": gate["repeat_count"],
        "repeated_identically": True,
    }


def _run_selective_quality_gate(
    model: object,
    api: MlxApi,
    selective: dict[str, object],
) -> dict[str, object]:
    trace = selective.get("trace")
    if type(trace) is not dict:
        raise BundleError("selective runtime trace contract drifted")
    prompt_ids = trace.get("prompt_token_ids")
    teacher_ids = trace.get("teacher_token_ids")
    if type(prompt_ids) is not list or type(teacher_ids) is not list:
        raise BundleError("selective runtime trace token IDs drifted")
    if (
        trace.get("api") != "mlx_lm.generate.generate_step"
        or trace.get("semantics") != "mlx-generate-step-argmax-v1"
        or trace.get("teacher_steps") != 128
        or trace.get("free_run_steps") != 128
        or trace.get("repeat_count") != 2
        or len(teacher_ids) != 128
        or trace.get("teacher_ids_sha256") != _token_ids_sha256(teacher_ids)
    ):
        raise BundleError("selective runtime 128-step trace contract drifted")
    teacher_observed_repeats: list[list[int]] = []
    for repeat in range(2):
        try:
            result = api.teacher_forced_step(
                api.array(prompt_ids), model, list(teacher_ids)
            )
        except BundleError:
            raise
        except Exception as error:
            raise BundleError(
                "selective teacher-forced quality gate repeat "
                f"{repeat + 1} failed: {error}"
            ) from error
        if type(result) is not list:
            raise BundleError(
                "selective teacher-forced quality gate returned an unexpected value"
            )
        observed: list[int] = []
        for token in result:
            if type(token) is bool:
                raise BundleError(
                    "selective teacher-forced quality gate returned a boolean token"
                )
            try:
                observed.append(int(token))
            except (TypeError, ValueError, OverflowError) as error:
                raise BundleError(
                    "selective teacher-forced quality gate returned an invalid token"
                ) from error
        if len(observed) != 128:
            raise BundleError(
                "selective teacher-forced quality gate returned an unexpected "
                "token count"
            )
        errors = [
            index
            for index, (actual, expected) in enumerate(
                zip(observed, teacher_ids, strict=True)
            )
            if actual != expected
        ]
        if errors:
            raise BundleError(
                "selective teacher-forced quality gate diverged from the BF16 "
                f"teacher at steps {errors[:8]}"
            )
        if _token_ids_sha256(observed) != trace["teacher_ids_sha256"]:
            raise BundleError(
                "selective teacher-forced quality gate trajectory hash drifted"
            )
        teacher_observed_repeats.append(observed)
    if teacher_observed_repeats[0] != teacher_observed_repeats[1]:
        raise BundleError(
            "selective teacher-forced quality gate repeats were not identical"
        )
    observed_repeats: list[list[int]] = []

    def greedy_argmax(logprobs: object) -> object:
        return api.argmax(logprobs, axis=-1)

    for repeat in range(2):
        try:
            generated = api.generate_step(
                api.array(prompt_ids),
                model,
                max_tokens=128,
                sampler=greedy_argmax,
            )
            observed: list[int] = []
            for item in generated:
                if type(item) is not tuple or len(item) != 2:
                    raise BundleError(
                        "mlx_lm.generate_step yielded an unexpected value"
                    )
                token = item[0]
                if type(token) is bool:
                    raise BundleError("mlx_lm.generate_step yielded a boolean token")
                observed.append(int(token))
        except BundleError:
            raise
        except Exception as error:
            raise BundleError(
                f"selective async quality gate repeat {repeat + 1} failed: {error}"
            ) from error
        if len(observed) != 128:
            raise BundleError(
                "selective async quality gate returned an unexpected token count"
            )
        errors = [
            index
            for index, (actual, expected) in enumerate(
                zip(observed, teacher_ids, strict=True)
            )
            if actual != expected
        ]
        if errors:
            raise BundleError(
                "selective async quality gate diverged from the BF16 teacher "
                f"at steps {errors[:8]}"
            )
        if _token_ids_sha256(observed) != trace["teacher_ids_sha256"]:
            raise BundleError("selective async quality gate trajectory hash drifted")
        observed_repeats.append(observed)
    if observed_repeats[0] != observed_repeats[1]:
        raise BundleError("selective async quality gate repeats were not identical")
    return {
        "format": "apxinf-mlx-selective-quality-gate-v2",
        "prompt_token_ids": prompt_ids,
        "teacher_ids_sha256": trace["teacher_ids_sha256"],
        "teacher_forced": {
            "api": "mlx_lm.generate.generate_step",
            "semantics": "mlx-generate-step-cached-teacher-forced-argmax-v1",
            "forced_token_ids_sha256": trace["teacher_ids_sha256"],
            "steps": 128,
            "exact_steps": 128,
            "repeat_count": 2,
            "repeat_sha256": [
                _token_ids_sha256(observed) for observed in teacher_observed_repeats
            ],
            "repeated_identically": True,
        },
        "async_free_run": {
            "api": trace["api"],
            "semantics": trace["semantics"],
            "steps": 128,
            "exact_steps": 128,
            "repeat_count": 2,
            "repeat_sha256": [
                _token_ids_sha256(observed) for observed in observed_repeats
            ],
            "repeated_identically": True,
        },
    }


def _run_counterfactual_quality_gate(
    model: object,
    api: MlxApi,
    hybrid: dict[str, object],
) -> dict[str, object]:
    gate = hybrid.get("quality_gate")
    if type(gate) is not dict:
        raise BundleError("counterfactual runtime quality-gate contract drifted")
    result = _run_selective_quality_gate(model, api, {"trace": gate})
    result["format"] = "apxinf-mlx-counterfactual-canonical-quality-gate-v1"
    return result


def _is_sensitive_environment_key(key: str) -> bool:
    upper = key.upper()
    return upper in {
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "NO_PROXY",
        "HF_TOKEN",
        "HF_API_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
        "HUGGINGFACE_TOKEN",
    } or upper.endswith(("_PROXY", "_TOKEN", "_TOKEN_PATH"))


@contextmanager
def _offline_runtime(cache_dir: Path) -> Iterator[None]:
    controlled = {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "DO_NOT_TRACK": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "HF_HOME": str(cache_dir / "huggingface"),
        "HF_HUB_CACHE": str(cache_dir / "huggingface/hub"),
        "TRANSFORMERS_CACHE": str(cache_dir / "transformers"),
    }
    touched = set(controlled) | {
        key for key in os.environ if _is_sensitive_environment_key(key)
    }
    previous = {key: os.environ.get(key) for key in touched}
    for key in touched:
        os.environ.pop(key, None)
    os.environ.update(controlled)

    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex
    original_create_connection = socket.create_connection

    def blocked_connect(*_args: object, **_kwargs: object) -> NoReturn:
        raise BundleError("network access is forbidden while building an MLX bundle")

    def blocked_connect_ex(*_args: object, **_kwargs: object) -> int:
        return errno.ENETUNREACH

    socket.socket.connect = blocked_connect  # type: ignore[method-assign]
    socket.socket.connect_ex = blocked_connect_ex  # type: ignore[method-assign]
    socket.create_connection = blocked_connect  # type: ignore[assignment]
    try:
        yield
    finally:
        socket.socket.connect = original_connect  # type: ignore[method-assign]
        socket.socket.connect_ex = original_connect_ex  # type: ignore[method-assign]
        socket.create_connection = original_create_connection
        for key in touched:
            os.environ.pop(key, None)
        for key, value in previous.items():
            if value is not None:
                os.environ[key] = value


def _write_private_regular(path: Path, payload: bytes) -> None:
    try:
        info = path.lstat()
    except FileNotFoundError:
        pass
    except OSError as error:
        raise BundleError(f"cannot inspect generated {path.name}: {error}") from error
    else:
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
            raise BundleError(
                f"generated {path.name} is not a regular non-symlink file"
            )
        path.unlink()
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise BundleError(f"short write while restoring {path.name}")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _quantization_for_mode(mode: str) -> dict[str, object] | None:
    if mode == "mixed-bf16":
        return None
    bits = 8 if mode == "affine-w8-g64" else 4
    return {"bits": bits, "group_size": 64, "mode": "affine"}


def _hybrid_config_manifest(hybrid: dict[str, object]) -> dict[str, object]:
    manifest = {
        "name": hybrid["name"],
        "policy_sha256": hybrid["policy_sha256"],
        "source_revision": hybrid["source_revision"],
        "retained_bf16_paths": hybrid["retained_bf16_paths"],
        "weight_ledger": hybrid["weight_ledger"],
    }
    counterfactual = hybrid.get("counterfactual")
    if counterfactual is not None:
        manifest["counterfactual"] = counterfactual
    return manifest


def _build_payload(
    source: SourceBundle,
    payload_dir: Path,
    cache_dir: Path,
    mode: str,
    api: MlxApi | None,
    hybrid: dict[str, object] | None,
    selective: dict[str, object] | None,
) -> dict[str, object] | None:
    quality_gate_result = None
    selective_pre_save_gate = None
    counterfactual_pre_save_gate = None
    with _offline_runtime(cache_dir):
        selected_api = api if api is not None else _load_mlx_api()
        try:
            loaded = selected_api.load(
                str(source.directory),
                tokenizer_config={
                    "local_files_only": True,
                    "trust_remote_code": False,
                },
                lazy=False,
                return_config=True,
            )
        except BundleError:
            raise
        except Exception as error:
            raise BundleError(f"pinned MLX-LM load failed: {error}") from error
        if type(loaded) is not tuple or len(loaded) != 3:
            raise BundleError("pinned MLX-LM load returned an unexpected value")
        model, tokenizer, loaded_config = loaded
        if type(loaded_config) is not dict:
            raise BundleError("pinned MLX-LM load did not return a config object")
        if loaded_config.get("model_type") != SUPPORTED_MODEL_TYPE:
            raise BundleError("loaded MLX config model_type differs from the source")
        quantization = _quantization_for_mode(mode)
        if quantization is not None:
            retained = (
                frozenset(hybrid["retained_bf16_paths"])
                if hybrid is not None
                else frozenset()
            )
            quant_predicate = None
            if selective is not None:
                tiers = selective["tiers"]
                if type(tiers) is not dict:
                    raise BundleError("selective tier map contract drifted")

                def selective_predicate(path: str, _module: object) -> object:
                    tier = tiers.get(path)
                    if tier == "w4":
                        return True
                    if tier == "w8":
                        return {"bits": 8, "group_size": 64, "mode": "affine"}
                    if tier == "bf16":
                        return False
                    raise BundleError(
                        f"MLX exposed an unfrozen quantization candidate: {path}"
                    )

                quant_predicate = selective_predicate
            elif hybrid is not None:

                def hybrid_predicate(path: str, _module: object) -> bool:
                    return path not in retained

                quant_predicate = hybrid_predicate
            try:
                quantized = selected_api.quantize_model(
                    model,
                    loaded_config,
                    group_size=64,
                    bits=quantization["bits"],
                    mode="affine",
                    **(
                        {"quant_predicate": quant_predicate}
                        if quant_predicate is not None
                        else {}
                    ),
                )
            except BundleError:
                raise
            except Exception as error:
                raise BundleError(
                    f"pinned MLX-LM quantization failed: {error}"
                ) from error
            if type(quantized) is not tuple or len(quantized) != 2:
                raise BundleError(
                    "pinned MLX-LM quantization returned an unexpected value"
                )
            model, loaded_config = quantized
            if type(loaded_config) is not dict:
                raise BundleError("pinned MLX-LM quantization did not return a config")
        if hybrid is not None:
            loaded_config[HYBRID_CONFIG_KEY] = _hybrid_config_manifest(hybrid)
            if hybrid.get("counterfactual") is not None:
                counterfactual_pre_save_gate = _run_counterfactual_quality_gate(
                    model, selected_api, hybrid
                )
            else:
                quality_gate_result = _run_hybrid_quality_gate(
                    model, selected_api, hybrid
                )
        if selective is not None:
            loaded_config[SELECTIVE_CONFIG_KEY] = selective["config_manifest"]
            selective_pre_save_gate = _run_selective_quality_gate(
                model, selected_api, selective
            )
        try:
            selected_api.save(
                payload_dir,
                str(source.directory),
                model,
                tokenizer,
                loaded_config,
                donate_model=True,
            )
        except BundleError:
            raise
        except Exception as error:
            raise BundleError(f"pinned MLX-LM save failed: {error}") from error
        is_counterfactual = (
            hybrid is not None and hybrid.get("counterfactual") is not None
        )
        if selective is not None or is_counterfactual:
            del model
            del tokenizer
            try:
                reloaded = selected_api.load(
                    str(payload_dir),
                    tokenizer_config={
                        "local_files_only": True,
                        "trust_remote_code": False,
                    },
                    lazy=False,
                    return_config=True,
                )
            except BundleError:
                raise
            except Exception as error:
                raise BundleError(f"post-save reload failed: {error}") from error
            if type(reloaded) is not tuple or len(reloaded) != 3:
                raise BundleError("post-save reload returned an unexpected value")
            reloaded_model, _reloaded_tokenizer, reloaded_config = reloaded
            if type(reloaded_config) is not dict:
                raise BundleError("post-save reload did not return a config object")
            _validate_output_config(reloaded_config, mode, hybrid, selective)
            if selective is not None:
                try:
                    post_save_gate = _run_selective_quality_gate(
                        reloaded_model, selected_api, selective
                    )
                except BundleError as error:
                    raise BundleError(
                        f"post-save reload quality gate failed: {error}"
                    ) from error
                if selective_pre_save_gate is None:
                    raise BundleError(
                        "selective pre-save quality gate was not recorded"
                    )
                quality_gate_result = {
                    "format": "apxinf-mlx-selective-deployed-quality-gate-v2",
                    "pre_save": selective_pre_save_gate,
                    "post_save_reload": post_save_gate,
                    "deployed_bundle_reloaded": True,
                    "exact_trajectory_claim": True,
                    "formal_performance_claim": False,
                }
            else:
                assert hybrid is not None
                try:
                    post_save_gate = _run_counterfactual_quality_gate(
                        reloaded_model, selected_api, hybrid
                    )
                except BundleError as error:
                    raise BundleError(
                        "post-save reload counterfactual quality gate failed: "
                        f"{error}"
                    ) from error
                if counterfactual_pre_save_gate is None:
                    raise BundleError(
                        "counterfactual pre-save quality gate was not recorded"
                    )
                quality_gate_result = {
                    "format": (
                        "apxinf-mlx-counterfactual-deployed-canonical-gate-v1"
                    ),
                    "pre_save": counterfactual_pre_save_gate,
                    "post_save_reload": post_save_gate,
                    "deployed_bundle_reloaded": True,
                    "canonical_gate_passed": True,
                    "fixed_suite_accepted": False,
                    "promotion_accepted": False,
                    "required_final_gate": (
                        "qwen35-0.8b-mlx-multi-prompt-quality-v1-4-prompts-x2"
                    ),
                    "exact_trajectory_claim": True,
                    "general_parity_claim": False,
                    "formal_performance_claim": False,
                }
    for name, exact_payload in source.tokenizer_payloads.items():
        _write_private_regular(payload_dir / name, exact_payload)
    return quality_gate_result


def _validate_output_config(
    config: dict[str, object],
    mode: str,
    hybrid: dict[str, object] | None,
    selective: dict[str, object] | None,
) -> None:
    _validate_no_remote_code(config, "output config.json")
    if config.get("model_type") != SUPPORTED_MODEL_TYPE:
        raise BundleError("output config.json has an unexpected model_type")
    expected = _quantization_for_mode(mode)
    observed = config.get("quantization")
    observed_compat = config.get("quantization_config")
    if selective is not None:
        expected_selective = dict(expected) if expected is not None else None
        if expected_selective is None:
            raise BundleError("selective output is missing its W4 default")
        for path in selective["w8_paths"]:
            expected_selective[path] = {
                "bits": 8,
                "group_size": 64,
                "mode": "affine",
            }
        if observed != expected_selective or observed_compat != expected_selective:
            raise BundleError(
                "selective output quantization config does not match W4/W8 policy"
            )
    elif expected is None:
        if observed is not None or observed_compat is not None:
            raise BundleError("mixed-bf16 output unexpectedly declares quantization")
    elif observed != expected or observed_compat != expected:
        raise BundleError(f"{mode} output does not declare the exact quantization tier")
    observed_hybrid = config.get(HYBRID_CONFIG_KEY)
    observed_selective = config.get(SELECTIVE_CONFIG_KEY)
    if hybrid is None:
        if observed_hybrid is not None:
            raise BundleError("generic output unexpectedly declares a hybrid preset")
    else:
        expected_hybrid = _hybrid_config_manifest(hybrid)
        if observed_hybrid != expected_hybrid:
            raise BundleError("hybrid output config policy manifest drifted")
    if selective is None:
        if observed_selective is not None:
            raise BundleError(
                "non-selective output unexpectedly declares a selective policy"
            )
    else:
        if observed_hybrid is not None:
            raise BundleError("selective output unexpectedly declares a hybrid preset")
        if observed_selective != selective["config_manifest"]:
            raise BundleError("selective output config policy manifest drifted")


def _validate_mixed_schema(
    source_schema: dict[str, tuple[str, tuple[int, ...]]],
    output_schema: dict[str, tuple[str, tuple[int, ...]]],
) -> None:
    expected = _canonical_language_schema(source_schema)
    if output_schema != expected:
        missing = sorted(set(expected) - set(output_schema))[:3]
        unexpected = sorted(set(output_schema) - set(expected))[:3]
        changed = sorted(
            name
            for name in set(expected) & set(output_schema)
            if expected[name] != output_schema[name]
        )[:3]
        raise BundleError(
            "mixed-bf16 tensor dtype/shape preservation failed "
            f"(missing={missing}, unexpected={unexpected}, changed={changed})"
        )
    dtypes = {dtype for dtype, _shape in output_schema.values()}
    if not {"BF16", "F32"} <= dtypes:
        raise BundleError(
            "mixed-bf16 output did not preserve both BF16 and F32 tensors"
        )


def _canonical_language_schema(
    source_schema: dict[str, tuple[str, tuple[int, ...]]],
) -> dict[str, tuple[str, tuple[int, ...]]]:
    canonical: dict[str, tuple[str, tuple[int, ...]]] = {}
    for source_name, (dtype, source_shape) in source_schema.items():
        if not source_name.startswith("model.language_model."):
            continue
        name = "language_model.model." + source_name.removeprefix(
            "model.language_model."
        )
        shape = source_shape
        # Qwen3.5's HF depthwise Conv1d layout is [channels, 1, kernel].
        # The pinned MLX loader sanitizes it to [channels, kernel, 1].  This is
        # a controlled layout transform, not a dtype cast.
        if name.endswith(".linear_attn.conv1d.weight"):
            if len(shape) != 3 or shape[1] != 1:
                raise BundleError(
                    f"source Qwen3.5 conv tensor {source_name!r} has invalid shape"
                )
            shape = (shape[0], shape[2], shape[1])
        canonical[name] = (dtype, shape)
    return canonical


def _validate_quantized_schema(
    source_schema: dict[str, tuple[str, tuple[int, ...]]],
    output_schema: dict[str, tuple[str, tuple[int, ...]]],
    mode: str,
    retained_paths: frozenset[str] = frozenset(),
) -> tuple[int, int]:
    bits = 8 if mode == "affine-w8-g64" else 4
    source_language = _canonical_language_schema(source_schema)
    expected: dict[str, tuple[str, tuple[int, ...]]] = {}
    quantized_count = 0
    retained_count = 0
    for name, (dtype, shape) in source_language.items():
        eligible = (
            name.endswith(".weight")
            and len(shape) == 2
            and shape[-1] > 0
            and shape[-1] % 64 == 0
        )
        base = name.removesuffix(".weight")
        should_retain = eligible and base in retained_paths
        should_quantize = eligible and not should_retain
        if not should_quantize:
            expected[name] = (dtype, shape)
            if should_retain:
                retained_count += 1
            continue
        if dtype != "BF16":
            raise BundleError(
                f"{mode} source quantization candidate {name!r} is not BF16"
            )
        expected[name] = ("U32", (*shape[:-1], shape[-1] * bits // 32))
        parameter_shape = (*shape[:-1], shape[-1] // 64)
        expected[f"{base}.scales"] = (dtype, parameter_shape)
        expected[f"{base}.biases"] = (dtype, parameter_shape)
        quantized_count += 1
    if quantized_count == 0:
        raise BundleError(f"{mode} source has no eligible affine group-64 tensors")
    if output_schema != expected:
        missing = sorted(set(expected) - set(output_schema))[:3]
        unexpected = sorted(set(output_schema) - set(expected))[:3]
        changed = sorted(
            name
            for name in set(expected) & set(output_schema)
            if expected[name] != output_schema[name]
        )[:3]
        raise BundleError(
            f"{mode} tensor packing/schema validation failed "
            f"(missing={missing}, unexpected={unexpected}, changed={changed})"
        )
    if retained_count != len(retained_paths):
        raise BundleError("hybrid retained tensor set is incomplete")
    return quantized_count, retained_count


def _validate_selective_quantized_schema(
    source_schema: dict[str, tuple[str, tuple[int, ...]]],
    output_schema: dict[str, tuple[str, tuple[int, ...]]],
    selective: dict[str, object],
) -> dict[str, int]:
    tiers = selective.get("tiers")
    if type(tiers) is not dict or not tiers:
        raise BundleError("selective schema validator has no frozen tier map")
    expected: dict[str, tuple[str, tuple[int, ...]]] = {}
    observed_paths: set[str] = set()
    counts = {"w4": 0, "w8": 0, "bf16": 0}
    for name, (dtype, shape) in _canonical_language_schema(source_schema).items():
        path = name.removesuffix(".weight")
        tier = tiers.get(path)
        if tier is None:
            expected[name] = (dtype, shape)
            continue
        observed_paths.add(path)
        if dtype != "BF16" or len(shape) != 2 or shape[-1] % 64 != 0:
            raise BundleError(f"selective candidate {name!r} is not W4/W8 eligible")
        counts[tier] += 1
        if tier == "bf16":
            expected[name] = (dtype, shape)
            continue
        bits = 4 if tier == "w4" else 8
        expected[name] = ("U32", (*shape[:-1], shape[-1] * bits // 32))
        parameter_shape = (*shape[:-1], shape[-1] // 64)
        expected[f"{path}.scales"] = ("BF16", parameter_shape)
        expected[f"{path}.biases"] = ("BF16", parameter_shape)
    missing_paths = sorted(set(tiers) - observed_paths)
    if missing_paths:
        raise BundleError(
            f"selective schema is missing frozen candidates: {missing_paths[:3]}"
        )
    if output_schema != expected:
        missing = sorted(set(expected) - set(output_schema))[:3]
        unexpected = sorted(set(output_schema) - set(expected))[:3]
        changed = sorted(
            name
            for name in set(expected) & set(output_schema)
            if expected[name] != output_schema[name]
        )[:3]
        raise BundleError(
            "selective W4/W8/BF16 tensor packing/schema validation failed "
            f"(missing={missing}, unexpected={unexpected}, changed={changed})"
        )
    return {
        "w4_module_count": counts["w4"],
        "w8_module_count": counts["w8"],
        "retained_bf16_module_count": counts["bf16"],
    }


def _inspect_output(
    output_dir: Path,
    source: SourceBundle,
    mode: str,
    hybrid: dict[str, object] | None,
    selective: dict[str, object] | None,
) -> tuple[dict[str, FileRecord], dict[str, object]]:
    files = _scan_flat_directory(
        output_dir,
        label="MLX output directory",
        fixed_names=OUTPUT_FIXED_FILES,
        shard_pattern=OUTPUT_SHARD,
    )
    records = {name: _hash_regular(files[name], name) for name in sorted(files)}
    output_config_payload = _read_regular(
        files["config.json"],
        "output config.json",
        MAX_JSON_BYTES,
        expected=records["config.json"],
    )
    output_config = _parse_json(output_config_payload, "output config.json")
    _validate_output_config(output_config, mode, hybrid, selective)
    for name, exact_payload in source.tokenizer_payloads.items():
        maximum = (
            MAX_TEMPLATE_BYTES if name == "chat_template.jinja" else MAX_TOKENIZER_BYTES
        )
        observed = _read_regular(
            files[name], f"output {name}", maximum, expected=records[name]
        )
        if observed != exact_payload:
            raise BundleError(f"output {name} is not byte-identical to the source")
    output_schema = _validate_weight_bundle(
        output_dir, files, OUTPUT_SHARD, "MLX output directory", records
    )
    quantized_tensor_count = 0
    retained_bf16_tensor_count = 0
    selective_counts: dict[str, int] | None = None
    if mode == "mixed-bf16":
        _validate_mixed_schema(source.tensor_schema, output_schema)
    elif selective is not None:
        selective_counts = _validate_selective_quantized_schema(
            source.tensor_schema, output_schema, selective
        )
        quantized_tensor_count = (
            selective_counts["w4_module_count"] + selective_counts["w8_module_count"]
        )
        retained_bf16_tensor_count = selective_counts["retained_bf16_module_count"]
    else:
        quantized_tensor_count, retained_bf16_tensor_count = _validate_quantized_schema(
            source.tensor_schema,
            output_schema,
            mode,
            frozenset(hybrid["retained_bf16_paths"])
            if hybrid is not None
            else frozenset(),
        )
    dtype_counts: dict[str, int] = {}
    for dtype, _shape in output_schema.values():
        dtype_counts[dtype] = dtype_counts.get(dtype, 0) + 1
    evidence = {
        "artifact_count": len(records),
        "artifacts": {name: records[name].public() for name in sorted(records)},
        "manifest_sha256": _manifest_sha256(records),
        "total_bytes": sum(record.size for record in records.values()),
        "tensor_count": len(output_schema),
        "quantized_tensor_count": quantized_tensor_count,
        "tensor_dtype_counts": dict(sorted(dtype_counts.items())),
        "tokenizer_bytes_preserved": True,
        "mixed_dtype_schema_preserved": mode == "mixed-bf16",
    }
    if hybrid is not None:
        evidence["retained_bf16_tensor_count"] = retained_bf16_tensor_count
        evidence["weight_ledger"] = hybrid["weight_ledger"]
    if selective is not None:
        assert selective_counts is not None
        evidence.update(selective_counts)
        evidence["selective_mixed_quantization_verified"] = True
        evidence["weight_ledger"] = selective["weight_ledger"]
    _assert_records_current(
        output_dir,
        records,
        "MLX output directory",
        fixed_names=OUTPUT_FIXED_FILES,
        shard_pattern=OUTPUT_SHARD,
    )
    return records, evidence


def _receipt(
    source: SourceBundle,
    output_dir: Path,
    mode: str,
    runtime: dict[str, object],
    output_evidence: dict[str, object],
    *,
    published: bool,
    verify_only: bool,
    hybrid: dict[str, object] | None,
    selective: dict[str, object] | None,
    quality_gate: dict[str, object] | None = None,
) -> dict[str, object]:
    source_public = {
        name: source.records[name].public() for name in sorted(source.records)
    }
    receipt = {
        "format": RECEIPT_FORMAT,
        "passed": True,
        "mode": mode,
        "published": published,
        "verify_only": verify_only,
        "source": {
            "directory": str(source.directory),
            "model_type": source.config["model_type"],
            "artifact_count": len(source.records),
            "artifacts": source_public,
            "manifest_sha256": _manifest_sha256(source.records),
        },
        "output": {"directory": str(output_dir), **output_evidence},
        "runtime": runtime,
        "policy": {
            "network": "python-sockets-blocked-and-hf-offline-v1",
            "credentials": "ambient-hf-tokens-and-proxies-cleared-v1",
            "remote_code": False,
            "publication": "same-filesystem-atomic-no-replace-v1",
            "source_layout": "flat-qwen3.5-single-link-files-v1",
            "tokenizer_copy": "source-byte-identical-v1",
            "mlx_api": "mlx_lm.utils.load+quantize_model+save-v1",
            "blanket_dtype_cast": False,
            "quality_tier": (
                (
                    "static-selective-policy-schema-only-not-runtime-parity"
                    if verify_only
                    else selective["quality_tier"]
                )
                if selective is not None
                else hybrid["quality_tier"]
                if hybrid is not None
                else (
                    "parity-candidate-mixed-precision"
                    if mode == "mixed-bf16"
                    else "explicit-quantized-quality-tier-not-a-parity-claim"
                )
            ),
        },
    }
    if hybrid is not None:
        receipt["preset"] = _hybrid_config_manifest(hybrid)
    if selective is not None:
        receipt["selective_policy"] = {
            "policy_path": selective["policy_path"],
            "policy_sha256": selective["policy_sha256"],
            "policy_document_sha256": selective["policy_document_sha256"],
            "search_receipt": selective["search_receipt"],
            "search_receipt_sha256": selective["search_receipt_sha256"],
            "search_status": selective["search_status"],
            "source_repo_id": selective["source_repo_id"],
            "source_revision": selective["source_revision"],
            "source_lock_content_sha256": selective["source_lock_content_sha256"],
            "source_manifest_sha256": selective["source_manifest_sha256"],
            "candidate_modules": selective["candidate_modules"],
            "candidate_modules_sha256": selective["candidate_modules_sha256"],
            "w4_paths": selective["w4_paths"],
            "w8_paths": selective["w8_paths"],
            "retained_bf16_paths": selective["retained_bf16_paths"],
            "trace": selective["trace"],
            "trace_sha256": selective["trace_sha256"],
            "weight_ledger": selective["weight_ledger"],
            "runtime_gate": "not-run-verify-only" if verify_only else "passed",
            "formal_performance_claim": False,
        }
    if quality_gate is not None:
        receipt["quality_gate"] = quality_gate
    return receipt


def _rename_no_replace(source: Path, destination: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    source_bytes = os.fsencode(source)
    destination_bytes = os.fsencode(destination)
    if sys.platform == "darwin" and hasattr(libc, "renamex_np"):
        rename = libc.renamex_np
        rename.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        rename.restype = ctypes.c_int
        result = rename(source_bytes, destination_bytes, RENAME_EXCL)
    elif sys.platform.startswith("linux") and hasattr(libc, "renameat2"):
        rename = libc.renameat2
        rename.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        rename.restype = ctypes.c_int
        result = rename(-100, source_bytes, -100, destination_bytes, RENAME_NOREPLACE)
    else:
        raise BundleError("this platform has no supported atomic no-replace rename")
    if result == 0:
        return
    observed_errno = ctypes.get_errno()
    if observed_errno in (errno.EEXIST, errno.ENOTEMPTY):
        raise BundleError(
            "output directory appeared during publication; no files replaced"
        )
    raise BundleError(
        f"atomic no-replace publication failed: {os.strerror(observed_errno)}"
    )


def _remove_build_root(build_root: Path, expected_parent: Path) -> None:
    try:
        info = build_root.lstat()
    except FileNotFoundError:
        return
    if (
        build_root.parent != expected_parent
        or not build_root.name.startswith(".apxinf-mlx-build-")
        or stat.S_ISLNK(info.st_mode)
        or not stat.S_ISDIR(info.st_mode)
        or info.st_uid != os.getuid()
    ):
        raise BundleError("refusing to clean an unexpected build directory")
    shutil.rmtree(build_root)


def build_bundle(
    source_dir_argument: str,
    output_dir_argument: str,
    mode: str,
    *,
    verify_only: bool = False,
    api: MlxApi | None = None,
    preset: str | None = None,
    mixed_policy: str | None = None,
    source_revision: str | None = None,
) -> dict[str, object]:
    if mode not in MODES:
        raise BundleError(f"unsupported mode: {mode}")
    if preset is not None and mixed_policy is not None:
        raise BundleError("--preset and --mixed-policy are mutually exclusive")
    source = _inspect_source(source_dir_argument)
    if mixed_policy is None:
        hybrid = _load_hybrid_policy(source, preset, source_revision, mode)
        selective = None
    else:
        hybrid = None
        selective = _load_selective_policy(source, mixed_policy, source_revision, mode)
    runtime = _runtime_versions()
    output_argument = _require_absolute(output_dir_argument, "--output-dir")
    output_parent = _require_owned_directory(output_argument.parent, "output parent")
    output_dir = output_parent / output_argument.name
    if _paths_overlap(source.directory, output_dir):
        raise BundleError("source and output directories must not overlap")

    if verify_only:
        verified_dir = _require_owned_directory(output_dir, "MLX output directory")
        _assert_source_unchanged(source)
        _records, evidence = _inspect_output(
            verified_dir, source, mode, hybrid, selective
        )
        _assert_hybrid_evidence_unchanged(hybrid)
        return _receipt(
            source,
            verified_dir,
            mode,
            runtime,
            evidence,
            published=False,
            verify_only=True,
            hybrid=hybrid,
            selective=selective,
        )

    _require_output_absent(output_dir)
    build_root = Path(
        tempfile.mkdtemp(prefix=".apxinf-mlx-build-", dir=output_parent)
    ).resolve(strict=True)
    build_root.chmod(0o700)
    payload_dir = build_root / "payload"
    cache_dir = build_root / "runtime-cache"
    cache_dir.mkdir(mode=0o700)
    try:
        quality_gate = None
        with open(os.devnull, "w", encoding="utf-8") as dependency_output:
            with redirect_stdout(dependency_output), redirect_stderr(dependency_output):
                quality_gate = _build_payload(
                    source, payload_dir, cache_dir, mode, api, hybrid, selective
                )
        _assert_hybrid_evidence_unchanged(hybrid)
        if not payload_dir.is_dir() or payload_dir.is_symlink():
            raise BundleError("pinned MLX-LM did not create a regular output directory")
        payload_dir.chmod(0o700)
        for entry in os.scandir(payload_dir):
            if entry.is_file(follow_symlinks=False):
                os.chmod(entry.path, 0o600, follow_symlinks=False)
        _assert_source_unchanged(source)
        _records, evidence = _inspect_output(
            payload_dir, source, mode, hybrid, selective
        )
        prepared = _receipt(
            source,
            output_dir,
            mode,
            runtime,
            evidence,
            published=True,
            verify_only=False,
            hybrid=hybrid,
            selective=selective,
            quality_gate=quality_gate,
        )
        _canonical_bytes(prepared)
        _assert_source_unchanged(source)
        _assert_hybrid_evidence_unchanged(hybrid)
        _rename_no_replace(payload_dir, output_dir)
        return prepared
    finally:
        _remove_build_root(build_root, output_parent)


def _parse_arguments(argv: list[str]) -> argparse.Namespace:
    parser = ReceiptArgumentParser(add_help=True)
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mode", required=True, choices=MODES)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--preset",
        choices=(
            HYBRID_PRESET,
            HYBRID_PRESET_V2,
            HYBRID_COUNTERFACTUAL_PRESET_V3,
        ),
    )
    selection.add_argument(
        "--mixed-policy",
        help="absolute path to a frozen selective W4/W8/BF16 policy document",
    )
    parser.add_argument(
        "--source-revision",
        help="required exact frozen source revision when --preset is selected",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="validate an existing output without loading or copying model weights",
    )
    return parser.parse_args(argv)


def _json_line(value: object) -> str:
    return _canonical_bytes(value).decode("utf-8") + "\n"


def main(argv: list[str] | None = None) -> int:
    try:
        arguments = _parse_arguments(sys.argv[1:] if argv is None else argv)
        receipt = build_bundle(
            arguments.source_dir,
            arguments.output_dir,
            arguments.mode,
            verify_only=arguments.verify_only,
            preset=arguments.preset,
            mixed_policy=arguments.mixed_policy,
            source_revision=arguments.source_revision,
        )
        sys.stdout.write(_json_line(receipt))
        sys.stdout.flush()
        return 0
    except BundleError as error:
        message = " ".join(str(error).split())[:2048] or "unknown bundle error"
        sys.stderr.write(
            _json_line(
                {
                    "format": ERROR_FORMAT,
                    "error": {"message": message},
                }
            )
        )
        sys.stderr.flush()
        return 2
    except Exception as error:
        message = " ".join(str(error).split())[:2048] or "unknown internal error"
        sys.stderr.write(
            _json_line(
                {
                    "format": ERROR_FORMAT,
                    "error": {"message": f"unexpected failure: {message}"},
                }
            )
        )
        sys.stderr.flush()
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
