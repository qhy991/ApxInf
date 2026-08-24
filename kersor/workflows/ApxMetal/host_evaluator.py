#!/usr/bin/python3
"""Host-owned evaluator for the ApxInf Metal W8 head workflow.

The workflow gives this program candidate *shader bytes*.  It never gives an
agent a writable source path.  Formal evaluation later in this module copies a
frozen ApxInf snapshot into command-v1's private TMPDIR and changes exactly one
file there: ``crates/apxinf-metal/src/metal_w8.metal``.

This file intentionally uses only the Python 3.9 standard library so the
workflow can invoke the fixed ``/usr/bin/python3 -B`` macOS interpreter.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
import platform
import re
import selectors
import signal
import shutil
import statistics
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import List


SCHEMA_VERSION = 1
CANONICAL_SHADER = "crates/apxinf-metal/src/metal_w8.metal"
MAX_CANDIDATE_BYTES = 64 * 1024
REQUIRED_KERNELS = ("w8_rows_topk4", "w8_final_topk4")
ALLOWED_PREPROCESSOR_LINES = {"#include <metal_stdlib>"}
BLOCK_ORDERS = ("ABBA", "BAAB", "ABBA", "BAAB", "ABBA", "BAAB")
MINIMUM_SPEEDUP = 1.10
TTFT_MAX_RATIO = 1.05
RSS_MAX_DELTA_BYTES = 64 * 1024 * 1024
MAX_COMMAND_OUTPUT_BYTES = 2 * 1024 * 1024
COMMAND_OUTPUT_TAIL_BYTES = 2048
QUIET_MAX_LOAD_PER_LOGICAL_CPU = 0.50
QUIET_MAX_EXTERNAL_PROCESS_CPU_PERCENT = 25.0
COMMAND_TIMEOUT_SECONDS = 900
GENERATION_TIMEOUT_SECONDS = 300
OVERALL_DEADLINE_SECONDS = 1650
_OVERALL_DEADLINE = None
REQUEST_KEYS = {
    "schema_version",
    "candidate_source",
    "candidate_source_sha256",
    "kernel_path",
    "model_path",
    "strategy_id",
    "prompt",
}
SNAPSHOT_IGNORED_NAMES = {
    ".git",
    ".kersor",
    ".apxinf",
    ".venv",
    "node_modules",
    "target",
    "target-kersor",
    "__pycache__",
}
QWEN35_08B_IDENTITY = {
    "model_type": "qwen3_5",
    "architectures": ["Qwen3_5ForConditionalGeneration"],
    "text_model_type": "qwen3_5_text",
    "hidden_size": 1024,
    "intermediate_size": 3584,
    "num_hidden_layers": 24,
    "num_attention_heads": 8,
    "num_key_value_heads": 2,
    "head_dim": 256,
    "vocab_size": 248320,
    "tie_word_embeddings": True,
    "full_attention_interval": 4,
}


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            if (
                _OVERALL_DEADLINE is not None
                and time.monotonic() >= _OVERALL_DEADLINE
            ):
                raise EvaluationError(
                    "blocked_deadline", "Host overall evaluation deadline expired"
                )
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _qwen35_08b_identity(model: Path) -> dict:
    try:
        config = json.loads((model / "config.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as error:
        raise ValueError("model_path config.json is not valid UTF-8 JSON") from error
    if not isinstance(config, dict) or not isinstance(config.get("text_config"), dict):
        raise ValueError("model_path is not the exact Qwen3.5-0.8B configuration")
    text = config["text_config"]
    identity = {
        "model_type": config.get("model_type"),
        "architectures": config.get("architectures"),
        "text_model_type": text.get("model_type"),
        "hidden_size": text.get("hidden_size"),
        "intermediate_size": text.get("intermediate_size"),
        "num_hidden_layers": text.get("num_hidden_layers"),
        "num_attention_heads": text.get("num_attention_heads"),
        "num_key_value_heads": text.get("num_key_value_heads"),
        "head_dim": text.get("head_dim"),
        "vocab_size": text.get("vocab_size"),
        "tie_word_embeddings": text.get("tie_word_embeddings"),
        "full_attention_interval": text.get("full_attention_interval"),
    }
    if identity != QWEN35_08B_IDENTITY:
        raise ValueError("model_path is not the exact Qwen3.5-0.8B configuration")
    return identity


def freeze_model_manifest(model_path: Path) -> dict:
    """Hash every direct model file and bind it to the exact 0.8B config."""

    model = Path(model_path).resolve(strict=True)
    identity = _qwen35_08b_identity(model)
    files = []
    for path in sorted(model.iterdir(), key=lambda item: item.name):
        if path.is_symlink() or not path.is_file():
            raise ValueError("model_path may contain only direct regular files")
        files.append(
            {
                "name": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": _hash_file(path),
            }
        )
    manifest_bytes = json.dumps(
        files, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return {
        "identity": identity,
        "file_count": len(files),
        "files": files,
        "manifest_sha256": _hash_bytes(manifest_bytes),
    }


def validate_candidate_source(source: object) -> List[str]:
    """Return deterministic, fail-closed source-scope violations."""

    problems: List[str] = []
    if not isinstance(source, str):
        return ["candidate_source must be a string"]
    encoded = source.encode("utf-8")
    if not source.strip():
        problems.append("candidate_source must not be empty")
    if len(encoded) > MAX_CANDIDATE_BYTES:
        problems.append("candidate_source exceeds the 65536-byte Host limit")
    if "\x00" in source:
        problems.append("candidate_source contains a NUL byte")
    if ')APX_METAL"' in source:
        problems.append(
            "candidate_source collides with the build.rs raw-string delimiter"
        )

    directives = [
        line.strip() for line in source.splitlines() if line.lstrip().startswith("#")
    ]
    if directives != sorted(ALLOWED_PREPROCESSOR_LINES):
        problems.append(
            "candidate_source preprocessor directives must be exactly one "
            "#include <metal_stdlib>"
        )
    for forbidden in ("__FILE__", "__DATE__", "__TIME__", "#line", "#import"):
        if forbidden in source:
            problems.append("candidate_source contains forbidden token: " + forbidden)

    for kernel in REQUIRED_KERNELS:
        declaration = re.compile(r"\bkernel\s+void\s+" + re.escape(kernel) + r"\s*\(")
        count = len(declaration.findall(source))
        if count != 1:
            problems.append(
                "candidate_source must define exactly one kernel void "
                + kernel
                + " (found "
                + str(count)
                + ")"
            )
    return problems


def validate_request(request: object, project_root: Path) -> List[str]:
    """Validate the complete workflow-to-Host request without normalizing it."""

    if not isinstance(request, dict):
        return ["request must be a JSON object"]
    problems: List[str] = []
    keys = set(request)
    unexpected = sorted(keys - REQUEST_KEYS)
    missing = sorted(
        {
            "schema_version",
            "candidate_source",
            "kernel_path",
            "model_path",
            "strategy_id",
        }
        - keys
    )
    if unexpected:
        problems.append("unexpected request keys: " + ", ".join(unexpected))
    if missing:
        problems.append("missing request keys: " + ", ".join(missing))
    if request.get("schema_version") != SCHEMA_VERSION or isinstance(
        request.get("schema_version"), bool
    ):
        problems.append("schema_version must be integer 1")

    problems.extend(validate_candidate_source(request.get("candidate_source")))
    declared_candidate_hash = request.get("candidate_source_sha256")
    if declared_candidate_hash is not None:
        if declared_candidate_hash != _sha256_text(request.get("candidate_source")):
            problems.append(
                "candidate_source_sha256 does not match candidate_source bytes"
            )

    root = Path(project_root).resolve()
    canonical_kernel = root / CANONICAL_SHADER
    raw_kernel = request.get("kernel_path")
    if not isinstance(raw_kernel, str) or not raw_kernel:
        problems.append("kernel_path must be a non-empty absolute string")
    else:
        kernel = Path(raw_kernel)
        if not kernel.is_absolute():
            problems.append("kernel_path must be absolute")
        else:
            try:
                if kernel.resolve(strict=True) != canonical_kernel.resolve(strict=True):
                    problems.append(
                        "kernel_path is not the canonical ApxInf Metal shader"
                    )
                if kernel.is_symlink():
                    problems.append("kernel_path must not be a symlink")
            except OSError:
                problems.append("kernel_path does not resolve to the canonical shader")

    raw_model = request.get("model_path")
    if not isinstance(raw_model, str) or not raw_model:
        problems.append("model_path is mandatory for Host acceptance")
    else:
        model = Path(raw_model)
        if not model.is_absolute():
            problems.append("model_path must be absolute")
        elif model.is_symlink():
            problems.append("model_path must not be a symlink")
        elif not model.is_dir():
            problems.append("model_path must be an existing directory")
        else:
            for required_name in ("config.json", "tokenizer.json"):
                required = model / required_name
                if required.is_symlink() or not required.is_file():
                    problems.append(
                        "model_path must contain direct regular " + required_name
                    )
            weights = [
                path
                for path in model.glob("*.safetensors")
                if path.is_file() and not path.is_symlink()
            ]
            if not weights:
                problems.append(
                    "model_path must contain at least one direct safetensors shard"
                )
            try:
                _qwen35_08b_identity(model)
            except ValueError as error:
                problems.append(str(error))

    strategy = request.get("strategy_id")
    if not isinstance(strategy, str) or not re.fullmatch(
        r"[a-z0-9][a-z0-9_-]{0,63}", strategy
    ):
        problems.append("strategy_id must be a bounded lowercase identifier")
    prompt = request.get("prompt", "Hello")
    if not isinstance(prompt, str) or not prompt or len(prompt.encode("utf-8")) > 1024:
        problems.append("prompt must be a non-empty UTF-8 string of at most 1024 bytes")
    return problems


def _single_json_value(payload: object, label: str) -> dict:
    if isinstance(payload, bytes):
        try:
            text = payload.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError(label + " is not UTF-8") from error
    elif isinstance(payload, str):
        text = payload
    else:
        raise ValueError(label + " must be bytes or text")
    lines = text.splitlines()
    if len(lines) != 1 or not lines[0]:
        raise ValueError(label + " must contain exactly one JSON line")
    try:
        value = json.loads(lines[0])
    except ValueError as error:
        raise ValueError(label + " is not valid JSON") from error
    if not isinstance(value, dict):
        raise ValueError(label + " JSON value must be an object")
    return value


def _token_trajectory_sha256(token_ids: object) -> str:
    if not isinstance(token_ids, list) or not all(
        isinstance(token, int)
        and not isinstance(token, bool)
        and 0 <= token <= 0xFFFFFFFF
        for token in token_ids
    ):
        raise ValueError("generated_token_ids must be unsigned 32-bit integers")
    payload = json.dumps(token_ids, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_generation_receipt(
    receipt: object, *, expected_tokens: int, expect_metal: bool
) -> dict:
    """Validate the exact ApxInf native-F32 generation evidence shape."""

    if not isinstance(receipt, dict):
        raise ValueError("generation receipt must be an object")
    if receipt.get("format") != "apxinf-generation-v1":
        raise ValueError("generation receipt format is not apxinf-generation-v1")
    if receipt.get("model_type") != "qwen3_5":
        raise ValueError("generation receipt is not Qwen3.5")
    if receipt.get("device") != "cpu" or receipt.get("dtype") != "fp32":
        raise ValueError("generation receipt must use CPU native fp32")
    build = receipt.get("build")
    if not isinstance(build, dict):
        raise ValueError("generation receipt is missing build identity")
    expected_build = {
        "target_os": "macos",
        "target_arch": "aarch64",
        "matmul_feature": "accelerate",
        "metal_w8_lm_head": expect_metal,
    }
    for key, expected in expected_build.items():
        if build.get(key) != expected:
            raise ValueError(
                "generation build." + key + " must be " + json.dumps(expected)
            )
    token_ids = receipt.get("generated_token_ids")
    trajectory = _token_trajectory_sha256(token_ids)
    if len(token_ids) != expected_tokens:
        raise ValueError(
            "generation receipt must contain exactly "
            + str(expected_tokens)
            + " generated tokens"
        )
    profile = receipt.get("profile")
    if not isinstance(profile, dict) or profile.get("output_tokens") != expected_tokens:
        raise ValueError("generation profile output_tokens does not match the request")
    if not _positive_number(profile.get("ttft_ms")):
        raise ValueError("generation profile ttft_ms must be positive")
    if expected_tokens > 1 and not _positive_number(profile.get("generation_tps")):
        raise ValueError("generation profile generation_tps must be positive")
    return {
        "generated_token_ids": token_ids,
        "generated_ids_sha256": trajectory,
        "ttft_ms": profile["ttft_ms"],
        "generation_tps": profile.get("generation_tps"),
        "build": expected_build,
    }


def validate_teacher_receipt(receipt: object) -> None:
    """Require the real-checkpoint top-4 plus production prefill/decode gate."""

    if not isinstance(receipt, dict) or receipt.get("format") != (
        "apxinf-qwen35-metal-w8-top4-teacher-gate-v2"
    ):
        raise ValueError("teacher receipt format is invalid")
    reranked = receipt.get("f32_reranked")
    if (
        receipt.get("comparisons") != 128
        or not isinstance(reranked, dict)
        or reranked.get("matches") != 128
        or reranked.get("match_rate") != 1.0
        or reranked.get("mismatches") != []
    ):
        raise ValueError("teacher gate requires 128/128 native F32 rerank matches")
    production = receipt.get("production_generation")
    if (
        not isinstance(production, dict)
        or production.get("comparisons") != 10
        or not isinstance(production.get("generated_token_ids"), list)
        or len(production["generated_token_ids"]) != 10
    ):
        raise ValueError(
            "teacher gate must exercise the production prefill/decode path for 10 tokens"
        )
    _token_trajectory_sha256(production["generated_token_ids"])
    quantization = receipt.get("quantization")
    if not isinstance(quantization, dict) or quantization != {
        "layout": "hf-row-major",
        "scheme": "symmetric-int8-per-row-group",
        "group_size": 64,
        "scale_dtype": "f32",
    }:
        raise ValueError(
            "teacher gate quantization identity is not W8 group-64 F32-scale"
        )


def parse_time_l(payload: bytes) -> tuple:
    """Parse macOS /usr/bin/time -l peak RSS and per-process swaps."""

    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("/usr/bin/time output is not UTF-8") from error
    rss = re.findall(
        r"^\s*([0-9]+)\s+maximum resident set size\s*$", text, re.MULTILINE
    )
    swaps = re.findall(r"^\s*([0-9]+)\s+swaps\s*$", text, re.MULTILINE)
    if len(rss) != 1 or len(swaps) != 1:
        raise ValueError(
            "/usr/bin/time output must contain unique RSS and swap counters"
        )
    return int(rss[0]), int(swaps[0])


def _positive_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0


def reduce_performance(
    blocks: object,
    expected_trajectory: str,
    schedule_swap_growth_bytes: int = 0,
) -> dict:
    """Reduce the predeclared 24-sample paired protocol without dropping data."""

    problems: List[str] = []
    contamination: List[str] = []
    if (
        not isinstance(schedule_swap_growth_bytes, int)
        or isinstance(schedule_swap_growth_bytes, bool)
        or schedule_swap_growth_bytes < 0
    ):
        problems.append("schedule swap growth must be a non-negative integer")
    elif schedule_swap_growth_bytes != 0:
        contamination.append("formal schedule observed system swap growth")
    if not isinstance(blocks, list):
        return {
            "accepted": False,
            "replacement_required": False,
            "problems": ["blocks must be an array"],
            "preserved_blocks": blocks,
        }
    if len(blocks) != len(BLOCK_ORDERS):
        problems.append("formal performance requires exactly six blocks")

    baseline_samples = []
    candidate_samples = []
    same_direction = 0
    for index, expected_order in enumerate(BLOCK_ORDERS):
        if index >= len(blocks) or not isinstance(blocks[index], dict):
            problems.append("missing or invalid block " + str(index))
            continue
        block = blocks[index]
        order = block.get("order")
        if order != expected_order:
            problems.append("block " + str(index) + " order must be " + expected_order)
        quiet = block.get("quiet_host")
        if not isinstance(quiet, dict) or quiet.get("passed") is not True:
            contamination.append("block " + str(index) + " failed quiet-host gate")
        growth = block.get("system_swap_growth_bytes")
        if growth != 0:
            contamination.append("block " + str(index) + " observed system swap growth")
        samples = block.get("samples")
        if not isinstance(samples, list) or len(samples) != 4:
            problems.append("block " + str(index) + " must retain four samples")
            continue
        observed_order = "".join(
            sample.get("variant", "?") if isinstance(sample, dict) else "?"
            for sample in samples
        )
        if observed_order != expected_order:
            problems.append(
                "block " + str(index) + " sample order differs from declaration"
            )
        block_a = []
        block_b = []
        for sample_index, sample in enumerate(samples):
            if not isinstance(sample, dict):
                problems.append(
                    "block "
                    + str(index)
                    + " sample "
                    + str(sample_index)
                    + " is invalid"
                )
                continue
            label = sample.get("variant")
            if label not in ("A", "B"):
                continue
            for field in ("generation_tps", "ttft_ms", "max_rss_bytes"):
                if not _positive_number(sample.get(field)):
                    problems.append(
                        "block "
                        + str(index)
                        + " sample "
                        + str(sample_index)
                        + " has invalid "
                        + field
                    )
            if sample.get("process_swaps") != 0:
                contamination.append(
                    "block " + str(index) + " sample " + str(sample_index) + " swapped"
                )
            if sample.get("generated_ids_sha256") != expected_trajectory:
                problems.append(
                    "block "
                    + str(index)
                    + " sample "
                    + str(sample_index)
                    + " trajectory mismatch"
                )
            if label == "A":
                baseline_samples.append(sample)
                block_a.append(sample)
            else:
                candidate_samples.append(sample)
                block_b.append(sample)
        if len(block_a) == 2 and len(block_b) == 2:
            if statistics.median(
                item["generation_tps"] for item in block_b
            ) > statistics.median(item["generation_tps"] for item in block_a):
                same_direction += 1

    sample_count = len(baseline_samples) + len(candidate_samples)
    speedup = None
    ttft_ratio = None
    rss_delta = None
    if len(baseline_samples) == 12 and len(candidate_samples) == 12:
        baseline_tps = statistics.median(
            item["generation_tps"] for item in baseline_samples
        )
        candidate_tps = statistics.median(
            item["generation_tps"] for item in candidate_samples
        )
        speedup = candidate_tps / baseline_tps
        ttft_ratio = statistics.median(
            item["ttft_ms"] for item in candidate_samples
        ) / statistics.median(item["ttft_ms"] for item in baseline_samples)
        rss_delta = statistics.median(
            item["max_rss_bytes"] for item in candidate_samples
        ) - statistics.median(item["max_rss_bytes"] for item in baseline_samples)
        if speedup < MINIMUM_SPEEDUP:
            problems.append("median generation_tps speedup is below 1.10x")
        if ttft_ratio > TTFT_MAX_RATIO:
            problems.append("candidate TTFT exceeds the 1.05x guardrail")
        if rss_delta > RSS_MAX_DELTA_BYTES:
            problems.append("candidate RSS exceeds the 64 MiB guardrail")
    else:
        problems.append("formal performance requires twelve A and twelve B samples")
    if same_direction != 6:
        problems.append("candidate must win all six block medians")

    return {
        "accepted": not problems and not contamination,
        "replacement_required": bool(contamination),
        "problems": problems,
        "contamination": contamination,
        "sample_count": sample_count,
        "same_direction_blocks": same_direction,
        "generation_tps_speedup": speedup,
        "ttft_ratio": ttft_ratio,
        "rss_delta_bytes": rss_delta,
        "preserved_blocks": blocks,
    }


def _generation_argv(binary: Path, model_path: Path, prompt: str, tokens: int) -> list:
    return [
        str(binary),
        "generate",
        "--model",
        str(model_path),
        "--prompt",
        prompt,
        "--max-tokens",
        str(tokens),
        "--max-context",
        "4096",
        "--no-eos-stop",
        "--device",
        "cpu",
        "--dtype",
        "fp32",
        "--json",
        "--metal-w8-lm-head",
    ]


def build_command_plan(
    baseline_root: Path,
    candidate_root: Path,
    cargo: Path,
    model_path: Path,
    prompt: str,
) -> dict:
    """Return the frozen direct-argv plan used by the formal evaluator."""

    cargo = Path(cargo)
    baseline_root = Path(baseline_root)
    candidate_root = Path(candidate_root)
    baseline_binary = baseline_root / "target-kersor/release/apxinf"
    candidate_binary = candidate_root / "target-kersor/release/apxinf"
    metal_tests = [
        str(cargo),
        "test",
        "--offline",
        "--locked",
        "-p",
        "apxinf-metal",
    ]
    qwen_tests = [
        str(cargo),
        "test",
        "--offline",
        "--locked",
        "-p",
        "apxinf-model",
        "--features",
        "accelerate,metal-w8",
        "qwen35",
        "--lib",
    ]
    teacher = [
        str(cargo),
        "run",
        "--offline",
        "--locked",
        "--release",
        "-p",
        "apxinf-model",
        "--example",
        "qwen35_metal_w8_gate",
        "--features",
        "accelerate,metal-w8",
        "--",
        str(model_path),
        "128",
        prompt,
    ]
    build = [
        str(cargo),
        "build",
        "--offline",
        "--locked",
        "--release",
        "--features",
        "accelerate,metal-w8",
        "--bin",
        "apxinf",
    ]
    baseline_trajectory = _generation_argv(baseline_binary, model_path, prompt, 100)
    candidate_trajectory = _generation_argv(candidate_binary, model_path, prompt, 100)
    candidate_negative = _generation_argv(candidate_binary, model_path, prompt, 1)
    candidate_negative = [
        item for item in candidate_negative if item != "--metal-w8-lm-head"
    ]
    candidate_positive = _generation_argv(candidate_binary, model_path, prompt, 1)
    return {
        "correctness": [
            {"name": "metal_adversarial_tests", "argv": metal_tests},
            {"name": "qwen35_tests", "argv": qwen_tests},
            {"name": "teacher_forced_native_f32_128", "argv": teacher},
            {
                "name": "trajectory_exact_100",
                "argv": [baseline_trajectory, candidate_trajectory],
            },
            {
                "name": "execution_path_hit_and_negative_control",
                "argv": [candidate_negative, candidate_positive],
            },
        ],
        "build": {
            "baseline": build,
            "candidate": build,
            "baseline_binary": str(baseline_binary),
            "candidate_binary": str(candidate_binary),
        },
        "performance": {
            "block_orders": list(BLOCK_ORDERS),
            "baseline_argv": baseline_trajectory,
            "candidate_argv": candidate_trajectory,
        },
        "commands": [
            metal_tests,
            qwen_tests,
            teacher,
            build,
            baseline_trajectory,
            candidate_trajectory,
            candidate_negative,
            candidate_positive,
        ],
    }


def _sha256_text(value: object):
    if not isinstance(value, str):
        return None
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _snapshot_ignore(_directory: str, names: list) -> set:
    return set(name for name in names if name in SNAPSHOT_IGNORED_NAMES)


def _tree_manifest(root: Path) -> dict:
    manifest = {}
    root = Path(root)
    for directory, directory_names, file_names in os.walk(root, followlinks=False):
        base = Path(directory)
        retained_directories = []
        for name in sorted(directory_names):
            if name in SNAPSHOT_IGNORED_NAMES:
                continue
            path = base / name
            if path.is_symlink():
                relative = path.relative_to(root).as_posix()
                manifest[relative] = "symlink:" + str(path.readlink())
            else:
                retained_directories.append(name)
        directory_names[:] = retained_directories
        for name in sorted(file_names):
            if name in SNAPSHOT_IGNORED_NAMES:
                continue
            path = base / name
            relative = path.relative_to(root).as_posix()
            if path.is_symlink():
                manifest[relative] = "symlink:" + str(path.readlink())
            elif path.is_file():
                manifest[relative] = _hash_file(path)
    return manifest


def freeze_source_manifest(root: Path) -> dict:
    files = _tree_manifest(Path(root))
    payload = json.dumps(files, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return {
        "file_count": len(files),
        "manifest_sha256": _hash_bytes(payload),
        "files": files,
    }


def _compact_source_manifest(root: Path) -> dict:
    manifest = freeze_source_manifest(root)
    return {
        "file_count": manifest["file_count"],
        "manifest_sha256": manifest["manifest_sha256"],
    }


def _workflow_artifact_identity(project_root: Path) -> dict:
    evaluator = Path(__file__).resolve(strict=True)
    contract = Path(project_root) / "kersor/workflows/ApxMetal/host_contract.json"
    for label, path in (("host evaluator", evaluator), ("host contract", contract)):
        if path.is_symlink() or not path.is_file():
            raise EvaluationError(
                "blocked_contract", label + " must be a direct regular file"
            )
    return {
        "host_evaluator_sha256": _hash_file(evaluator),
        "host_contract_sha256": _hash_file(contract),
    }


def _toolchain_identity(toolchain: dict) -> dict:
    return {
        "cargo_sha256": toolchain["cargo_sha256"],
        "rustc_sha256": toolchain["rustc_sha256"],
    }


def _source_pair_identity(baseline_root: Path, candidate_root: Path) -> dict:
    return {
        "baseline": _compact_source_manifest(baseline_root),
        "candidate": _compact_source_manifest(candidate_root),
    }


def prepare_isolated_roots(
    project_root: Path, scratch_root: Path, candidate_source: str
) -> dict:
    """Freeze one snapshot, fork A/B roots, and modify only the B shader."""

    project_root = Path(project_root).resolve(strict=True)
    scratch_root = Path(scratch_root)
    scratch_root.mkdir(parents=True, exist_ok=False)
    frozen_root = scratch_root / "frozen"
    baseline_root = scratch_root / "baseline"
    candidate_root = scratch_root / "candidate"
    shutil.copytree(
        project_root,
        frozen_root,
        symlinks=True,
        ignore=_snapshot_ignore,
    )
    shutil.copytree(frozen_root, baseline_root, symlinks=True)
    shutil.copytree(frozen_root, candidate_root, symlinks=True)
    candidate_shader = candidate_root / CANONICAL_SHADER
    if candidate_shader.is_symlink() or not candidate_shader.is_file():
        raise ValueError("isolated candidate shader must be a direct regular file")
    with candidate_shader.open("w", encoding="utf-8", newline="") as stream:
        stream.write(candidate_source)

    baseline_manifest = _tree_manifest(baseline_root)
    candidate_manifest = _tree_manifest(candidate_root)
    differences = sorted(
        key
        for key in set(baseline_manifest) | set(candidate_manifest)
        if baseline_manifest.get(key) != candidate_manifest.get(key)
    )
    if differences not in ([], [CANONICAL_SHADER]):
        raise ValueError(
            "isolated candidate changed files outside the shader: "
            + ", ".join(differences)
        )
    return {
        "frozen_root": frozen_root,
        "baseline_root": baseline_root,
        "candidate_root": candidate_root,
        "tree_differences": differences,
        "baseline_tree_sha256": hashlib.sha256(
            json.dumps(
                baseline_manifest, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest(),
        "candidate_tree_sha256": hashlib.sha256(
            json.dumps(
                candidate_manifest, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest(),
        "baseline_shader_sha256": baseline_manifest.get(CANONICAL_SHADER),
        "candidate_shader_sha256": candidate_manifest.get(CANONICAL_SHADER),
    }


class EvaluationError(Exception):
    """Fail-closed formal-evaluation error with a stable receipt status."""

    def __init__(self, status: str, message: str, evidence: object = None):
        super().__init__(message)
        self.status = status
        self.evidence = evidence


class EvaluationCancelled(BaseException):
    """Raised by Host signal handlers so active command cleanup can run."""


def _raise_cancellation(signum, _frame) -> None:
    raise EvaluationCancelled("host evaluation cancelled by signal " + str(signum))


def _install_cancellation_handlers() -> dict:
    previous = {}
    for signum in (signal.SIGINT, signal.SIGTERM):
        previous[signum] = signal.getsignal(signum)
        signal.signal(signum, _raise_cancellation)
    return previous


def _restore_signal_handlers(previous: dict) -> None:
    for signum, handler in previous.items():
        signal.signal(signum, handler)


@contextmanager
def _evaluation_deadline(seconds: float):
    """Bound all nested commands by one monotonic Host-owned deadline."""

    global _OVERALL_DEADLINE
    previous = _OVERALL_DEADLINE
    proposed = time.monotonic() + seconds
    _OVERALL_DEADLINE = proposed if previous is None else min(previous, proposed)
    try:
        yield
    finally:
        _OVERALL_DEADLINE = previous


def _hash_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _bounded_tail(payload: bytes) -> str:
    tail = payload[-COMMAND_OUTPUT_TAIL_BYTES:]
    return tail.decode("utf-8", errors="replace")


def _command_evidence(completed: dict) -> dict:
    return {
        "argv": completed["argv"],
        "argv_sha256": _hash_bytes(
            json.dumps(completed["argv"], separators=(",", ":")).encode("utf-8")
        ),
        "exit_code": completed["returncode"],
        "timed_out": completed["timed_out"],
        "overall_deadline_exhausted": completed.get(
            "overall_deadline_exhausted", False
        ),
        "stdout_size_bytes": len(completed["stdout"]),
        "stdout_sha256": _hash_bytes(completed["stdout"]),
        "stdout_tail": _bounded_tail(completed["stdout"]),
        "stderr_size_bytes": len(completed["stderr"]),
        "stderr_sha256": _hash_bytes(completed["stderr"]),
        "stderr_tail": _bounded_tail(completed["stderr"]),
    }


def _terminate_process_tree(process: subprocess.Popen) -> None:
    """Terminate the complete private process group, then reap its leader."""

    process_group = process.pid
    try:
        os.killpg(process_group, signal.SIGTERM)
    except (OSError, ProcessLookupError):
        pass
    try:
        process.wait(timeout=0.25)
    except subprocess.TimeoutExpired:
        pass
    # The group may still contain descendants after its leader exits.
    try:
        os.killpg(process_group, signal.SIGKILL)
    except (OSError, ProcessLookupError):
        pass
    try:
        process.wait(timeout=1.0)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=1.0)


def _run_direct(
    argv: list,
    *,
    cwd: Path,
    environment: dict,
    timeout_seconds: int,
) -> dict:
    """Execute a Host-owned argv without a shell and bound all output."""

    if (
        not isinstance(argv, list)
        or not argv
        or not all(isinstance(item, str) and item for item in argv)
        or not Path(argv[0]).is_absolute()
    ):
        raise EvaluationError("blocked_contract", "formal command argv is not absolute")
    try:
        process = subprocess.Popen(
            argv,
            cwd=str(cwd),
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
    except OSError as error:
        raise EvaluationError(
            "blocked_environment", "formal command could not start: " + argv[0]
        ) from error
    timed_out = False
    stdout = bytearray()
    stderr = bytearray()
    streams = selectors.DefaultSelector()
    streams.register(process.stdout, selectors.EVENT_READ, stdout)
    streams.register(process.stderr, selectors.EVENT_READ, stderr)
    command_deadline = time.monotonic() + timeout_seconds
    deadline = (
        command_deadline
        if _OVERALL_DEADLINE is None
        else min(command_deadline, _OVERALL_DEADLINE)
    )
    overall_deadline_exhausted = False
    try:
        while streams.get_map():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                timed_out = True
                overall_deadline_exhausted = (
                    _OVERALL_DEADLINE is not None
                    and _OVERALL_DEADLINE <= command_deadline
                )
                _terminate_process_tree(process)
                break
            for key, _events in streams.select(timeout=min(0.10, remaining)):
                chunk = os.read(key.fd, 64 * 1024)
                if not chunk:
                    streams.unregister(key.fileobj)
                    key.fileobj.close()
                    continue
                key.data.extend(chunk)
                if len(key.data) > MAX_COMMAND_OUTPUT_BYTES:
                    _terminate_process_tree(process)
                    raise EvaluationError(
                        "rejected_evidence",
                        "formal command exceeded the 2 MiB stream limit",
                    )
        if not timed_out:
            remaining = max(0.0, deadline - time.monotonic())
            try:
                process.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                timed_out = True
                overall_deadline_exhausted = (
                    _OVERALL_DEADLINE is not None
                    and _OVERALL_DEADLINE <= command_deadline
                )
                _terminate_process_tree(process)
    except BaseException:
        if process.poll() is None:
            _terminate_process_tree(process)
        raise
    finally:
        streams.close()
        for stream in (process.stdout, process.stderr):
            if stream is not None and not stream.closed:
                stream.close()
    return {
        "argv": list(argv),
        "returncode": process.returncode,
        "timed_out": timed_out,
        "overall_deadline_exhausted": overall_deadline_exhausted,
        "stdout": bytes(stdout),
        "stderr": bytes(stderr),
    }


def _run_system(argv: list, timeout_seconds: int = 10) -> bytes:
    environment = {
        "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
        "LANG": "C",
        "LC_ALL": "C",
    }
    completed = _run_direct(
        argv,
        cwd=Path("/"),
        environment=environment,
        timeout_seconds=timeout_seconds,
    )
    if completed["timed_out"] or completed["returncode"] != 0:
        raise EvaluationError(
            "blocked_environment", "Host system probe failed: " + argv[0]
        )
    return completed["stdout"]


def _platform_identity() -> dict:
    if sys.platform != "darwin" or platform.machine() != "arm64":
        raise EvaluationError(
            "blocked_environment", "formal evaluation requires macOS arm64"
        )
    brand = (
        _run_system(["/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string"])
        .decode("ascii", errors="strict")
        .strip()
    )
    if not re.fullmatch(r"Apple M4(?: Pro| Max| Ultra)?", brand):
        raise EvaluationError(
            "blocked_environment", "formal evaluation requires an Apple M4 family SoC"
        )
    return {
        "os": "macos",
        "arch": "arm64",
        "soc": brand,
        "python": sys.version.split()[0],
    }


def _toolchain(project_root: Path) -> dict:
    toolchain_root = (
        project_root.parent
        / ".apxinf-toolchains/rustup/toolchains/stable-aarch64-apple-darwin"
    )
    cargo = toolchain_root / "bin/cargo"
    rustc = toolchain_root / "bin/rustc"
    cargo_home = project_root.parent / ".apxinf-toolchains/cargo"
    rustup_home = project_root.parent / ".apxinf-toolchains/rustup"
    for label, path in (("cargo", cargo), ("rustc", rustc)):
        if path.is_symlink() or not path.is_file() or not os.access(path, os.X_OK):
            raise EvaluationError(
                "blocked_environment", label + " toolchain binary is unavailable"
            )
    if not cargo_home.is_dir() or not rustup_home.is_dir():
        raise EvaluationError(
            "blocked_environment", "offline ApxInf Cargo cache is unavailable"
        )
    return {
        "cargo": cargo,
        "rustc": rustc,
        "cargo_home": cargo_home,
        "rustup_home": rustup_home,
        "toolchain_root": toolchain_root,
        "cargo_sha256": _hash_bytes(cargo.read_bytes()),
        "rustc_sha256": _hash_bytes(rustc.read_bytes()),
    }


def _cargo_environment(
    *, snapshot_root: Path, scratch_root: Path, toolchain: dict
) -> dict:
    home = scratch_root / "home"
    temporary = scratch_root / "tmp"
    target = snapshot_root / "target-kersor"
    for directory in (home, temporary, target):
        directory.mkdir(parents=True, exist_ok=True)
    return {
        "PATH": str(toolchain["toolchain_root"] / "bin")
        + ":/usr/bin:/bin:/usr/sbin:/sbin",
        "HOME": str(home),
        "TMPDIR": str(temporary),
        "TMP": str(temporary),
        "TEMP": str(temporary),
        "LANG": "C",
        "LC_ALL": "C",
        "CARGO_HOME": str(toolchain["cargo_home"]),
        "RUSTUP_HOME": str(toolchain["rustup_home"]),
        "RUSTC": str(toolchain["rustc"]),
        "CARGO_TARGET_DIR": str(target),
        "CARGO_NET_OFFLINE": "true",
        "CARGO_INCREMENTAL": "0",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "NO_PROXY": "*",
        "no_proxy": "*",
    }


def _parse_swap_used(payload: bytes) -> int:
    try:
        text = payload.decode("ascii")
    except UnicodeDecodeError as error:
        raise EvaluationError(
            "blocked_environment", "vm.swapusage output is not ASCII"
        ) from error
    match = re.search(r"\bused\s*=\s*([0-9]+(?:\.[0-9]+)?)([KMGTP])\b", text)
    if match is None:
        raise EvaluationError("blocked_environment", "vm.swapusage output is invalid")
    multiplier = 1024 ** {"K": 1, "M": 2, "G": 3, "T": 4, "P": 5}[match.group(2)]
    return int(round(float(match.group(1)) * multiplier))


def _system_swap_used() -> int:
    return _parse_swap_used(_run_system(["/usr/sbin/sysctl", "-n", "vm.swapusage"]))


def quiet_host_gate() -> dict:
    """Predeclared, deterministic pre-block host-noise gate."""

    logical_cpus_text = (
        _run_system(["/usr/sbin/sysctl", "-n", "hw.ncpu"])
        .decode("ascii", errors="strict")
        .strip()
    )
    load_text = (
        _run_system(["/usr/sbin/sysctl", "-n", "vm.loadavg"])
        .decode("ascii", errors="strict")
        .strip()
    )
    process_text = _run_system(["/bin/ps", "-A", "-o", "pid=,pcpu=,comm="]).decode(
        "utf-8", errors="replace"
    )
    if not logical_cpus_text.isdigit():
        raise EvaluationError("blocked_environment", "hw.ncpu output is invalid")
    load_match = re.fullmatch(
        r"\{\s*([0-9]+(?:\.[0-9]+)?)\s+([0-9]+(?:\.[0-9]+)?)\s+([0-9]+(?:\.[0-9]+)?)\s*\}",
        load_text,
    )
    if load_match is None:
        raise EvaluationError("blocked_environment", "vm.loadavg output is invalid")
    logical_cpus = int(logical_cpus_text)
    load_1m = float(load_match.group(1))
    maximum_load = logical_cpus * QUIET_MAX_LOAD_PER_LOGICAL_CPU
    offenders = []
    for line in process_text.splitlines():
        match = re.match(r"^\s*([0-9]+)\s+([0-9]+(?:\.[0-9]+)?)\s+(.+?)\s*$", line)
        if match is None:
            continue
        pid = int(match.group(1))
        cpu = float(match.group(2))
        if pid == os.getpid() or cpu < QUIET_MAX_EXTERNAL_PROCESS_CPU_PERCENT:
            continue
        offenders.append({"pid": pid, "cpu_percent": cpu, "command": match.group(3)})
    offenders.sort(key=lambda item: (-item["cpu_percent"], item["pid"]))
    passed = load_1m <= maximum_load and not offenders
    return {
        "passed": passed,
        "logical_cpus": logical_cpus,
        "load_1m": load_1m,
        "maximum_load_1m": maximum_load,
        "maximum_external_process_cpu_percent": (
            QUIET_MAX_EXTERNAL_PROCESS_CPU_PERCENT
        ),
        "offenders": offenders[:8],
    }


def _run_required_gate(*, name: str, argv: list, cwd: Path, environment: dict) -> dict:
    completed = _run_direct(
        argv,
        cwd=cwd,
        environment=environment,
        timeout_seconds=COMMAND_TIMEOUT_SECONDS,
    )
    evidence = {"name": name, **_command_evidence(completed)}
    evidence["passed"] = not completed["timed_out"] and completed["returncode"] == 0
    if not evidence["passed"]:
        raise EvaluationError(
            "rejected_correctness",
            "correctness gate failed: " + name,
            evidence=evidence,
        )
    return {"evidence": evidence, "completed": completed}


def _run_generation(
    *,
    argv: list,
    cwd: Path,
    environment: dict,
    expected_tokens: int,
    expect_metal: bool,
) -> dict:
    completed = _run_direct(
        argv,
        cwd=cwd,
        environment=environment,
        timeout_seconds=GENERATION_TIMEOUT_SECONDS,
    )
    if completed["timed_out"] or completed["returncode"] != 0:
        raise EvaluationError(
            "rejected_correctness",
            "generation command failed",
            evidence=_command_evidence(completed),
        )
    try:
        receipt = _single_json_value(completed["stdout"], "generation stdout")
        parsed = validate_generation_receipt(
            receipt,
            expected_tokens=expected_tokens,
            expect_metal=expect_metal,
        )
    except ValueError as error:
        raise EvaluationError(
            "rejected_correctness",
            str(error),
            evidence=_command_evidence(completed),
        ) from error
    return {
        "completed": completed,
        "receipt": receipt,
        "parsed": parsed,
        "evidence": _command_evidence(completed),
    }


def _measure_generation(
    *, argv: list, cwd: Path, environment: dict, variant: str, expected_trajectory: str
) -> dict:
    completed = _run_direct(
        ["/usr/bin/time", "-l", *argv],
        cwd=cwd,
        environment=environment,
        timeout_seconds=GENERATION_TIMEOUT_SECONDS,
    )
    if completed["timed_out"] or completed["returncode"] != 0:
        raise EvaluationError(
            "rejected_performance",
            "timed generation command failed",
            evidence=_command_evidence(completed),
        )
    try:
        receipt = _single_json_value(completed["stdout"], "timed generation stdout")
        parsed = validate_generation_receipt(
            receipt, expected_tokens=100, expect_metal=True
        )
        maximum_rss, process_swaps = parse_time_l(completed["stderr"])
    except ValueError as error:
        raise EvaluationError(
            "rejected_performance",
            str(error),
            evidence=_command_evidence(completed),
        ) from error
    if parsed["generated_ids_sha256"] != expected_trajectory:
        raise EvaluationError(
            "rejected_performance", "timed generation trajectory mismatch"
        )
    return {
        "variant": variant,
        "generation_tps": parsed["generation_tps"],
        "ttft_ms": parsed["ttft_ms"],
        "max_rss_bytes": maximum_rss,
        "process_swaps": process_swaps,
        "generated_ids_sha256": parsed["generated_ids_sha256"],
        "command": _command_evidence(completed),
    }


def _command_v1_scratch() -> Path:
    """Require the environment minted by KerSor command-v1 confinement."""

    raw_tmp = os.environ.get("TMPDIR", "")
    raw_home = os.environ.get("HOME", "")
    if not raw_tmp or not raw_home:
        raise EvaluationError(
            "blocked_environment", "command-v1 private scratch is unavailable"
        )
    try:
        temporary = Path(raw_tmp).resolve(strict=True)
        home = Path(raw_home).resolve(strict=True)
    except OSError as error:
        raise EvaluationError(
            "blocked_environment", "command-v1 private scratch does not resolve"
        ) from error
    if (
        temporary != home
        or not temporary.name.startswith("kersor-command-read-only-")
        or os.environ.get("PYTHONDONTWRITEBYTECODE") != "1"
        or not sys.dont_write_bytecode
    ):
        raise EvaluationError(
            "blocked_environment",
            "formal evaluation must run inside read-only KerSor command-v1",
        )
    pycache = Path(os.environ.get("PYTHONPYCACHEPREFIX", ""))
    if not pycache.is_absolute() or pycache.parent != temporary:
        raise EvaluationError(
            "blocked_environment", "command-v1 bytecode cache is not private"
        )
    return temporary


def _base_receipt(request: dict) -> dict:
    return {
        "format": "apxinf-kersor-metal-w8-host-evaluation-v1",
        "schema_version": SCHEMA_VERSION,
        "status": "running",
        "accepted": False,
        "strategy_id": request["strategy_id"],
        "candidate_shader_sha256": _sha256_text(request["candidate_source"]),
        "candidate_scope": [CANONICAL_SHADER],
        "platform": None,
        "snapshot": None,
        "toolchain": None,
        "custody": {
            "protocol": "command-v1",
            "overall_deadline_seconds": OVERALL_DEADLINE_SECONDS,
            "workflow_artifacts": None,
            "model": {"start": None, "end": None, "unchanged": False},
            "sources": {
                "start": None,
                "after_gates": None,
                "end": None,
                "unchanged": False,
            },
            "toolchain": {"start": None, "end": None, "unchanged": False},
        },
        "builds": [],
        "problems": [],
        "correctness": {"executed": False, "passed": False, "gates": []},
        "execution_path": {"passed": False, "evidence": None},
        "formal_benchmark": {
            "executed": False,
            "accepted": False,
            "sample_count": 0,
            "block_orders": list(BLOCK_ORDERS),
            "minimum_speedup": MINIMUM_SPEEDUP,
            "same_direction_blocks_required": 6,
            "replacement_required": False,
            "preserved_blocks": [],
            "generation_tps_speedup": None,
            "ttft_ratio": None,
            "rss_delta_bytes": None,
            "system_swap_used_start_bytes": None,
            "system_swap_used_end_bytes": None,
            "system_swap_growth_bytes": None,
        },
        "quality_claim": "native_f32_only",
        "claims_hf_bf16_parity": False,
    }


def _reject_receipt(
    receipt: dict,
    error: EvaluationError,
    *,
    active_gate: str = "",
) -> dict:
    receipt["status"] = error.status
    receipt["accepted"] = False
    receipt["problems"].append(str(error))
    if active_gate:
        gate = {
            "name": active_gate,
            "passed": False,
            "problem": str(error),
        }
        if error.evidence is not None:
            gate["command"] = error.evidence
        receipt["correctness"]["gates"].append(gate)
        receipt["correctness"]["executed"] = True
    return receipt


def _evaluate_formal(request: dict, project_root: Path) -> dict:
    """Run one evaluation under a deadline shorter than command-v1's timeout."""

    with _evaluation_deadline(OVERALL_DEADLINE_SECONDS):
        return _evaluate_formal_bounded(request, project_root)


def _evaluate_formal_bounded(request: dict, project_root: Path) -> dict:
    """Run the Host-owned correctness and paired-performance state machine."""

    receipt = _base_receipt(request)
    try:
        receipt["platform"] = _platform_identity()
        scratch_parent = _command_v1_scratch()
        receipt["custody"]["workflow_artifacts"] = _workflow_artifact_identity(
            project_root
        )
        receipt["custody"]["model"]["start"] = freeze_model_manifest(
            Path(request["model_path"])
        )
        toolchain = _toolchain(project_root)
        toolchain_start = _toolchain_identity(toolchain)
        receipt["toolchain"] = {**toolchain_start, "offline": True}
        receipt["custody"]["toolchain"]["start"] = toolchain_start
        run_root = Path(
            tempfile.mkdtemp(prefix="apxinf-metal-formal-", dir=str(scratch_parent))
        )
        snapshot = prepare_isolated_roots(
            project_root=project_root,
            scratch_root=run_root / "snapshot",
            candidate_source=request["candidate_source"],
        )
        if snapshot["tree_differences"] != [CANONICAL_SHADER]:
            raise EvaluationError(
                "rejected_scope",
                "candidate must differ from baseline in exactly the Metal shader",
            )
        receipt["snapshot"] = {
            key: value
            for key, value in snapshot.items()
            if key.endswith("sha256") or key == "tree_differences"
        }
        receipt["custody"]["sources"]["start"] = _source_pair_identity(
            snapshot["baseline_root"], snapshot["candidate_root"]
        )

        writable_cargo_home = run_root / "cargo-home"
        shutil.copytree(toolchain["cargo_home"], writable_cargo_home, symlinks=True)
        private_toolchain = dict(toolchain)
        private_toolchain["cargo_home"] = writable_cargo_home
        baseline_root = snapshot["baseline_root"]
        candidate_root = snapshot["candidate_root"]
        baseline_env = _cargo_environment(
            snapshot_root=baseline_root,
            scratch_root=run_root / "baseline-runtime",
            toolchain=private_toolchain,
        )
        candidate_env = _cargo_environment(
            snapshot_root=candidate_root,
            scratch_root=run_root / "candidate-runtime",
            toolchain=private_toolchain,
        )
        plan = build_command_plan(
            baseline_root=baseline_root,
            candidate_root=candidate_root,
            cargo=toolchain["cargo"],
            model_path=Path(request["model_path"]),
            prompt=request.get("prompt", "Hello"),
        )
    except EvaluationError as error:
        return _reject_receipt(receipt, error)
    except (OSError, ValueError, shutil.Error) as error:
        return _reject_receipt(
            receipt,
            EvaluationError(
                "blocked_environment", "snapshot setup failed: " + str(error)
            ),
        )

    # Gate 1: adversarial Metal unit tests.
    for gate_index in range(2):
        gate = plan["correctness"][gate_index]
        try:
            result = _run_required_gate(
                name=gate["name"],
                argv=gate["argv"],
                cwd=candidate_root,
                environment=candidate_env,
            )
        except EvaluationError as error:
            return _reject_receipt(receipt, error, active_gate=gate["name"])
        receipt["correctness"]["executed"] = True
        receipt["correctness"]["gates"].append(result["evidence"])

    # Gate 3: real-model, 128-step native-F32 teacher forcing and production hook.
    teacher_gate = plan["correctness"][2]
    try:
        teacher = _run_required_gate(
            name=teacher_gate["name"],
            argv=teacher_gate["argv"],
            cwd=candidate_root,
            environment=candidate_env,
        )
        teacher_receipt = _single_json_value(
            teacher["completed"]["stdout"], "teacher stdout"
        )
        validate_teacher_receipt(teacher_receipt)
    except ValueError as error:
        failure = EvaluationError(
            "rejected_correctness",
            str(error),
            evidence=(
                _command_evidence(teacher["completed"])
                if "teacher" in locals()
                else None
            ),
        )
        return _reject_receipt(receipt, failure, active_gate=teacher_gate["name"])
    except EvaluationError as error:
        return _reject_receipt(receipt, error, active_gate=teacher_gate["name"])
    teacher_evidence = dict(teacher["evidence"])
    teacher_evidence["teacher_receipt_sha256"] = _hash_bytes(
        json.dumps(teacher_receipt, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    )
    teacher_evidence["native_f32_rerank_matches"] = 128
    teacher_evidence["production_prefill_decode_tokens"] = 10
    receipt["correctness"]["gates"].append(teacher_evidence)

    # Two release builds from the same frozen tree; only their shader hashes differ.
    try:
        for variant, root, environment in (
            ("A", baseline_root, baseline_env),
            ("B", candidate_root, candidate_env),
        ):
            result = _run_required_gate(
                name="release_build_" + variant,
                argv=plan["build"]["baseline" if variant == "A" else "candidate"],
                cwd=root,
                environment=environment,
            )
            binary = Path(
                plan["build"][
                    "baseline_binary" if variant == "A" else "candidate_binary"
                ]
            )
            if binary.is_symlink() or not binary.is_file():
                raise EvaluationError(
                    "rejected_correctness",
                    "release build did not produce a direct binary",
                )
            receipt["builds"].append(
                {
                    "variant": variant,
                    "binary_sha256": _hash_bytes(binary.read_bytes()),
                    "shader_sha256": snapshot[
                        "baseline_shader_sha256"
                        if variant == "A"
                        else "candidate_shader_sha256"
                    ],
                    "command": result["evidence"],
                }
            )
    except EvaluationError as error:
        return _reject_receipt(receipt, error, active_gate="release_builds")

    # Gate 4: baseline and candidate must produce one exact 100-token trajectory.
    trajectory_gate = plan["correctness"][3]
    try:
        baseline_trajectory = _run_generation(
            argv=trajectory_gate["argv"][0],
            cwd=baseline_root,
            environment=baseline_env,
            expected_tokens=100,
            expect_metal=True,
        )
        candidate_trajectory = _run_generation(
            argv=trajectory_gate["argv"][1],
            cwd=candidate_root,
            environment=candidate_env,
            expected_tokens=100,
            expect_metal=True,
        )
        expected_trajectory = baseline_trajectory["parsed"]["generated_ids_sha256"]
        if (
            candidate_trajectory["parsed"]["generated_ids_sha256"]
            != expected_trajectory
        ):
            raise EvaluationError(
                "rejected_correctness",
                "candidate 100-token trajectory differs from baseline",
            )
    except EvaluationError as error:
        return _reject_receipt(receipt, error, active_gate=trajectory_gate["name"])
    receipt["correctness"]["gates"].append(
        {
            "name": trajectory_gate["name"],
            "passed": True,
            "generated_ids_sha256": expected_trajectory,
            "tokens": 100,
            "baseline_command": baseline_trajectory["evidence"],
            "candidate_command": candidate_trajectory["evidence"],
        }
    )

    # Gate 5: candidate bytes in the binary plus same-binary flag negative control.
    path_gate = plan["correctness"][4]
    candidate_binary = Path(plan["build"]["candidate_binary"])
    candidate_bytes = request["candidate_source"].encode("utf-8")
    try:
        binary_bytes = candidate_binary.read_bytes()
        if candidate_bytes not in binary_bytes:
            raise EvaluationError(
                "rejected_correctness",
                "candidate binary does not contain the exact candidate shader bytes",
            )
        negative = _run_generation(
            argv=path_gate["argv"][0],
            cwd=candidate_root,
            environment=candidate_env,
            expected_tokens=1,
            expect_metal=False,
        )
        positive = _run_generation(
            argv=path_gate["argv"][1],
            cwd=candidate_root,
            environment=candidate_env,
            expected_tokens=1,
            expect_metal=True,
        )
        first_expected = baseline_trajectory["parsed"]["generated_token_ids"][0]
        if negative["parsed"]["generated_token_ids"] != [first_expected] or positive[
            "parsed"
        ]["generated_token_ids"] != [first_expected]:
            raise EvaluationError(
                "rejected_correctness", "same-binary negative-control token differs"
            )
    except (OSError, EvaluationError) as error:
        if not isinstance(error, EvaluationError):
            error = EvaluationError("rejected_correctness", str(error))
        return _reject_receipt(receipt, error, active_gate=path_gate["name"])
    path_evidence = {
        "candidate_binary_sha256": _hash_bytes(binary_bytes),
        "candidate_shader_sha256": _hash_bytes(candidate_bytes),
        "exact_shader_bytes_in_binary": True,
        "negative_control_build_flag": False,
        "positive_build_flag": True,
        "one_token_id": first_expected,
        "negative_command": negative["evidence"],
        "positive_command": positive["evidence"],
    }
    receipt["correctness"]["gates"].append(
        {"name": path_gate["name"], "passed": True, **path_evidence}
    )
    receipt["execution_path"] = {"passed": True, "evidence": path_evidence}
    try:
        after_gates = _source_pair_identity(baseline_root, candidate_root)
    except (OSError, EvaluationError) as error:
        if not isinstance(error, EvaluationError):
            error = EvaluationError("rejected_custody", str(error))
        return _reject_receipt(receipt, error, active_gate="source_custody")
    receipt["custody"]["sources"]["after_gates"] = after_gates
    if after_gates != receipt["custody"]["sources"]["start"]:
        return _reject_receipt(
            receipt,
            EvaluationError(
                "rejected_custody",
                "source manifest changed during correctness gates",
            ),
            active_gate="source_custody",
        )
    receipt["correctness"]["passed"] = True

    # Formal primary metric: exactly 3xABBA + 3xBAAB, with pre-block quiet gates.
    blocks = []
    measurement_error = None
    try:
        schedule_swap_start = _system_swap_used()
    except EvaluationError as error:
        return _reject_receipt(receipt, error)
    receipt["formal_benchmark"]["system_swap_used_start_bytes"] = (
        schedule_swap_start
    )
    for block_index, order in enumerate(BLOCK_ORDERS):
        try:
            quiet = quiet_host_gate()
        except EvaluationError as error:
            quiet = {"passed": False, "probe_error": str(error)}
        block = {
            "index": block_index,
            "order": order,
            "quiet_host": quiet,
            "system_swap_used_before_bytes": None,
            "system_swap_used_after_bytes": None,
            "system_swap_growth_bytes": 0,
            "samples": [],
        }
        blocks.append(block)
        if quiet.get("passed") is not True:
            break
        try:
            block["system_swap_used_before_bytes"] = _system_swap_used()
            for label in order:
                if label == "A":
                    argv = plan["performance"]["baseline_argv"]
                    root = baseline_root
                    environment = baseline_env
                else:
                    argv = plan["performance"]["candidate_argv"]
                    root = candidate_root
                    environment = candidate_env
                block["samples"].append(
                    _measure_generation(
                        argv=argv,
                        cwd=root,
                        environment=environment,
                        variant=label,
                        expected_trajectory=expected_trajectory,
                    )
                )
            block["system_swap_used_after_bytes"] = _system_swap_used()
            block["system_swap_growth_bytes"] = max(
                0,
                block["system_swap_used_after_bytes"]
                - block["system_swap_used_before_bytes"],
            )
        except EvaluationError as error:
            measurement_error = error
            block["measurement_error"] = str(error)
            if error.evidence is not None:
                block["failed_command"] = error.evidence
            break
        if block["system_swap_growth_bytes"] != 0 or any(
            sample["process_swaps"] != 0 for sample in block["samples"]
        ):
            break

    try:
        schedule_swap_end = _system_swap_used()
        source_end = _source_pair_identity(baseline_root, candidate_root)
        model_end = freeze_model_manifest(Path(request["model_path"]))
        toolchain_end = _toolchain_identity(_toolchain(project_root))
    except (OSError, ValueError, EvaluationError) as error:
        if not isinstance(error, EvaluationError):
            error = EvaluationError("rejected_custody", str(error))
        return _reject_receipt(receipt, error)

    formal = receipt["formal_benchmark"]
    formal["system_swap_used_end_bytes"] = schedule_swap_end
    formal["system_swap_growth_bytes"] = max(
        0, schedule_swap_end - schedule_swap_start
    )
    receipt["custody"]["sources"]["end"] = source_end
    receipt["custody"]["sources"]["unchanged"] = (
        source_end == receipt["custody"]["sources"]["start"]
    )
    receipt["custody"]["model"]["end"] = model_end
    receipt["custody"]["model"]["unchanged"] = (
        model_end == receipt["custody"]["model"]["start"]
    )
    receipt["custody"]["toolchain"]["end"] = toolchain_end
    receipt["custody"]["toolchain"]["unchanged"] = (
        toolchain_end == receipt["custody"]["toolchain"]["start"]
    )
    if not all(
        (
            receipt["custody"]["sources"]["unchanged"],
            receipt["custody"]["model"]["unchanged"],
            receipt["custody"]["toolchain"]["unchanged"],
        )
    ):
        return _reject_receipt(
            receipt,
            EvaluationError(
                "rejected_custody",
                "model, source, or toolchain identity changed during evaluation",
            ),
        )

    reduced = reduce_performance(
        blocks,
        expected_trajectory=expected_trajectory,
        schedule_swap_growth_bytes=formal["system_swap_growth_bytes"],
    )
    formal.update(reduced)
    formal["executed"] = any(block["samples"] for block in blocks)
    formal["preserved_blocks"] = blocks
    formal["sample_count"] = sum(len(block["samples"]) for block in blocks)
    if measurement_error is not None:
        receipt["status"] = measurement_error.status
        receipt["problems"].append(str(measurement_error))
        return receipt
    if reduced["replacement_required"]:
        receipt["status"] = "replacement_required"
        receipt["problems"].extend(reduced["contamination"])
        return receipt
    if not reduced["accepted"]:
        receipt["status"] = "rejected_performance"
        receipt["problems"].extend(reduced["problems"])
        return receipt

    receipt["status"] = "accepted"
    receipt["accepted"] = True
    formal["accepted"] = True
    return receipt


def blocked_receipt(request: object, problems: List[str]) -> dict:
    candidate = request.get("candidate_source") if isinstance(request, dict) else None
    strategy = request.get("strategy_id") if isinstance(request, dict) else None
    return {
        "format": "apxinf-kersor-metal-w8-host-evaluation-v1",
        "schema_version": SCHEMA_VERSION,
        "status": "blocked_input",
        "accepted": False,
        "strategy_id": strategy,
        "candidate_shader_sha256": _sha256_text(candidate),
        "candidate_scope": [CANONICAL_SHADER],
        "problems": problems,
        "correctness": {"executed": False, "passed": False, "gates": []},
        "formal_benchmark": {
            "executed": False,
            "sample_count": 0,
            "block_orders": list(BLOCK_ORDERS),
            "replacement_required": False,
            "preserved_blocks": [],
        },
        "quality_claim": "native_f32_only",
        "claims_hf_bf16_parity": False,
    }


def evaluate_request(request: object, project_root: Path) -> dict:
    problems = validate_request(request, project_root=project_root)
    if problems:
        return blocked_receipt(request, problems)
    return _evaluate_formal(
        request, project_root=Path(project_root).resolve(strict=True)
    )


def main(argv=None) -> int:
    previous_handlers = _install_cancellation_handlers()
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("--request-json", required=True)
        args = parser.parse_args(argv)
        try:
            request = json.loads(args.request_json)
        except (TypeError, ValueError) as error:
            sys.stderr.write("invalid --request-json: " + str(error) + "\n")
            return 2
        project_root = Path(__file__).resolve().parents[3]
        receipt = evaluate_request(request, project_root=project_root)
        sys.stdout.write(
            json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n"
        )
        return 0 if receipt.get("accepted") is True else 1
    except EvaluationCancelled as error:
        sys.stderr.write(str(error) + "\n")
        return 130
    finally:
        _restore_signal_handlers(previous_handlers)


if __name__ == "__main__":
    raise SystemExit(main())
