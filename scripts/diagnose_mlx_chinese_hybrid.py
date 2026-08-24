#!/usr/bin/env python3
"""Fail-closed Chinese trajectory and state-aligned MLX diagnostic.

The pure trajectory seam never imports MLX or opens model weights. The
production capture backend must satisfy the read-only interface frozen below;
fake tests exercise that orchestration without loading a real model. The real
two-model command must run under an independent outer process/RSS supervisor;
the model process intentionally contains no in-process RSS watchdog.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import secrets
import stat
import sys
from typing import NoReturn


PROMPT_ID = "chinese-explanation"
PRECISION_PROFILE = "hybrid-w8-bf16-g64"
QUALITY_EVIDENCE_CONTENT_SHA256 = (
    "16d7fd7ff43c56ba9d8992f39efe32caca2d1b2790499c78e7ecd5e60a460b0c"
)
CONTRACT_CONTENT_SHA256 = (
    "d52a79e62827913a34e8f3961233aea6b49d91cc317ab6e4a69405b80d9a311f"
)
CANDIDATE_ID = "qwen35-0.8b-hybrid-w8-bf16-g64-certified-v2"
RECEIPT_FORMAT = "apxinf-mlx-chinese-hybrid-diagnostic-receipt-v1"
CAPTURE_FORMAT = "apxinf-mlx-chinese-state-aligned-capture-v1"
SOURCE_CUSTODY_FORMAT = "apxinf-direct-regular-single-link-source-custody-v1"
EXPECTED_CAPTURE_BACKEND_SHA256 = (
    "27d7c62eacb1e7f5a2946048945e35e76a35efc3ae8cf8c6f5ac96199e534849"
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
RETAINED_BF16_PATHS = [
    "language_model.model.layers.12.linear_attn.out_proj",
    "language_model.model.layers.14.linear_attn.out_proj",
    "language_model.model.layers.20.linear_attn.out_proj",
]
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
PINNED_PUBLIC_API_AUDIT = {
    "mlx_lm_version": "0.31.3",
    "qwen3_5_py_sha256": (
        "f0daa30bba5cb521c8bdfa7093101a544c6a37bbba09bca582288219cb04ae3a"
    ),
    "qwen3_next_py_sha256": (
        "3c572fe3fbb36721efab4d80d1bb6af11beb4ad1caae18deefc9fc84cbcd9b79"
    ),
    "mlx_module_base_py_sha256": (
        "ec749e1d50fd1a5e57e0aedc8e6eb13fc697e630f59333a0e24aee62a8dc7f0f"
    ),
    "generate_py_sha256": (
        "270778ad53eaca55a8533d82e6752660fe5d2605c4aa0879b48a50a91f69345f"
    ),
    "model_call_result": "logits-only",
    "named_modules": True,
    "forward_hooks": False,
    "output_hidden_states": False,
}
_PAIR_FIELDS = {
    "pair_id",
    "process_id",
    "reference_handle_id",
    "candidate_handle_id",
    "reference_manifest_sha256",
    "candidate_manifest_sha256",
}
_CAPTURE_FIELDS = {
    "format",
    "prompt_id",
    "prompt_token_ids",
    "teacher_token_ids",
    "retained_bf16_paths",
    "w8_module_paths",
    "w8_module_paths_sha256",
    "runs",
}
_RUN_FIELDS = {"step_metrics", "module_metrics"}
_STEP_FIELDS = {
    "step_index",
    "reference_token_id",
    "reference_top1_token_id",
    "candidate_top1_token_id",
    "reference_top1_margin_micro",
    "candidate_reference_token_margin_micro",
}
_MODULE_FIELDS = {
    "path",
    "tier",
    "sample_count",
    "relative_l1_error_ppm",
    "max_abs_error_micro",
    "first_nonzero_step",
}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_IDENTIFIER = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]{0,127}$")
_MODULE_PATH = re.compile(
    r"^language_model(?:\.model\.layers\.(\d+)\.[a-z0-9_]+\.[a-z0-9_]+|"
    r"\.model\.embed_tokens|\.lm_head)$"
)
_BACKEND_SOURCE_MAX_BYTES = 2 * 1024 * 1024
_HASH_CHUNK_BYTES = 1024 * 1024


class DiagnosticError(ValueError):
    """A malformed, unbound, or unsupported diagnostic input."""


def _fail(message: str) -> NoReturn:
    raise DiagnosticError(message)


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
        raise DiagnosticError(f"value is not canonical JSON: {error}") from error


def object_sha256(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _stable_source_fields(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


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
        before = path.lstat()
    except OSError as error:
        raise DiagnosticError(f"cannot inspect {label}: {error}") from error
    if (
        resolved != path
        or stat.S_ISLNK(before.st_mode)
        or not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or before.st_size > _BACKEND_SOURCE_MAX_BYTES
    ):
        _fail(
            f"{label} must be a bounded canonical direct regular file "
            "with one hard link"
        )
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise DiagnosticError(f"cannot open {label}: {error}") from error
    digest = hashlib.sha256()
    try:
        opened = os.fstat(descriptor)
        if _stable_source_fields(opened) != _stable_source_fields(before):
            _fail(f"{label} changed before hashing")
        while True:
            chunk = os.read(descriptor, _HASH_CHUNK_BYTES)
            if not chunk:
                break
            digest.update(chunk)
        finished = os.fstat(descriptor)
        if _stable_source_fields(opened) != _stable_source_fields(finished):
            _fail(f"{label} changed while hashing")
    finally:
        os.close(descriptor)
    return {"path": str(path), "size": before.st_size, "sha256": digest.hexdigest()}


def _trusted_backend_source_custody() -> dict[str, object]:
    loader_path = Path(__file__)
    if not loader_path.is_absolute():
        _fail("diagnostic loader __file__ is not absolute")
    capture_path = loader_path.with_name("mlx_qwen35_state_aligned_capture.py")
    custody = {
        "format": SOURCE_CUSTODY_FORMAT,
        "capture": _source_file_identity(capture_path, "capture backend source"),
        "loader": _source_file_identity(loader_path, "diagnostic loader source"),
    }
    if custody["capture"]["sha256"] != EXPECTED_CAPTURE_BACKEND_SHA256:
        _fail("capture backend source SHA256 differs from the loader's frozen digest")
    return custody


def _validate_capture_capabilities(value: object) -> dict[str, object]:
    if type(value) is not dict or set(value) != set(REQUIRED_CAPTURE_CAPABILITIES) | {
        "source_custody"
    }:
        _fail("capture backend capability fields drifted")
    observed = dict(value)
    custody = observed.pop("source_custody")
    if observed != REQUIRED_CAPTURE_CAPABILITIES:
        _fail("capture backend cannot prove the frozen read-only interface")
    expected_custody = _trusted_backend_source_custody()
    if custody != expected_custody:
        _fail("capture backend source custody is not bound to this loader")
    return json.loads(canonical_bytes(value))


def _token_run(value: object, label: str) -> list[int]:
    if (
        type(value) is not list
        or not value
        or any(type(token) is not int or token < 0 for token in value)
    ):
        _fail(f"{label} must be a non-empty token-ID list")
    return list(value)


def _certified_chinese_scope(
    prompt_value: object,
    teacher_value: object,
) -> tuple[list[int], list[int]]:
    prompt = _token_run(prompt_value, "certified Chinese prompt")
    teacher = _token_run(teacher_value, "certified Chinese BF16 trajectory")
    sequence = prompt + teacher[:-1]
    if (
        len(prompt) != CERTIFIED_PROMPT_TOKEN_COUNT
        or len(teacher) != CERTIFIED_TEACHER_TOKEN_COUNT
        or object_sha256(prompt) != CERTIFIED_PROMPT_TOKEN_IDS_SHA256
        or object_sha256(teacher) != CERTIFIED_TEACHER_TOKEN_IDS_SHA256
        or len(sequence) != CERTIFIED_INPUT_TOKEN_COUNT
        or len(prompt) - 1 != CERTIFIED_RESPONSE_START
        or len(sequence) - CERTIFIED_RESPONSE_START != CERTIFIED_PREDICTOR_COUNT
        or object_sha256(sequence) != CERTIFIED_INPUT_TOKEN_IDS_SHA256
    ):
        _fail(
            "diagnostic is restricted to the certified Chinese v1 "
            "25-token prompt and 64-token BF16 teacher"
        )
    return prompt, teacher


def analyze_chinese_trajectory(envelope: object) -> dict[str, object]:
    """Locate the first free-run mismatch without making a semantic claim."""

    if type(envelope) is not dict:
        _fail("quality envelope must be an object")
    body = dict(envelope)
    content_sha256 = body.pop("content_sha256", None)
    if (
        envelope.get("format") != "apxinf-mlx-multi-prompt-quality-run-v1"
        or envelope.get("status") != "failed_comparison"
        or content_sha256 != QUALITY_EVIDENCE_CONTENT_SHA256
        or object_sha256(body) != content_sha256
    ):
        _fail("quality envelope is not the certified hybrid comparison")
    evidence = envelope.get("evidence")
    if (
        type(evidence) is not dict
        or evidence.get("contract_sha256") != CONTRACT_CONTENT_SHA256
    ):
        _fail("quality envelope has no inner evidence")
    candidate = evidence.get("candidate")
    if (
        type(candidate) is not dict
        or candidate.get("candidate_id") != CANDIDATE_ID
        or candidate.get("precision_profile") != PRECISION_PROFILE
        or candidate.get("claims_general_parity") is not False
    ):
        _fail("quality evidence is not the scoped hybrid-W8/BF16 candidate")
    records = evidence.get("records")
    if type(records) is not list:
        _fail("quality evidence records are unavailable")
    matches = [
        record
        for record in records
        if type(record) is dict and record.get("prompt_id") == PROMPT_ID
    ]
    if len(matches) != 1:
        _fail("quality evidence must contain exactly one Chinese prompt record")
    record = matches[0]
    reference = record.get("reference")
    observed = record.get("candidate")
    if type(reference) is not dict or type(observed) is not dict:
        _fail("Chinese prompt trajectories are unavailable")
    reference_runs = reference.get("runs")
    candidate_runs = observed.get("runs")
    if (
        type(reference_runs) is not list
        or len(reference_runs) != 2
        or reference_runs[0] != reference_runs[1]
        or type(candidate_runs) is not list
        or len(candidate_runs) != 2
        or candidate_runs[0] != candidate_runs[1]
    ):
        _fail("Chinese prompt trajectories must contain deterministic double runs")
    prompt, teacher = _certified_chinese_scope(
        record.get("prompt_token_ids"), reference_runs[0]
    )
    del prompt
    candidate_run = _token_run(candidate_runs[0], "Chinese candidate trajectory")
    if len(teacher) != len(candidate_run):
        _fail("Chinese trajectory lengths differ")
    first = next(
        (
            index
            for index, pair in enumerate(zip(teacher, candidate_run))
            if pair[0] != pair[1]
        ),
        None,
    )
    if first is None:
        first_divergence = None
        alignment = {
            "classification": "exact",
            "inserted_candidate_token_id": None,
            "truncated_reference_tail_token_id": None,
            "aligned_suffix_tokens": len(teacher),
        }
    else:
        first_divergence = {
            "step_index": first,
            "step_number": first + 1,
            "reference_token_id": teacher[first],
            "candidate_token_id": candidate_run[first],
        }
        shifted = candidate_run[first + 1 :] == teacher[first:-1]
        alignment = {
            "classification": (
                "candidate-single-token-insertion-with-fixed-window-tail-truncation"
                if shifted
                else "nontrivial-token-divergence"
            ),
            "inserted_candidate_token_id": candidate_run[first] if shifted else None,
            "truncated_reference_tail_token_id": teacher[-1] if shifted else None,
            "aligned_suffix_tokens": len(candidate_run[first + 1 :]) if shifted else 0,
        }
    return {
        "prompt_id": PROMPT_ID,
        "teacher_steps": len(teacher),
        "trajectory_exact": first is None,
        "exact_prefix_tokens": len(teacher) if first is None else first,
        "first_divergence": first_divergence,
        "alignment": alignment,
        "semantic_stability": {"assessed": False, "claim": None},
    }


def _quality_inputs(envelope: dict[str, object]) -> dict[str, object]:
    evidence = envelope["evidence"]
    custody = envelope.get("custody")
    if type(evidence) is not dict or type(custody) is not dict:
        _fail("certified quality envelope custody is unavailable")
    bundles = custody.get("bundles")
    if type(bundles) is not dict:
        _fail("certified quality envelope bundle custody is unavailable")
    snapshots = {}
    for lane in ("reference", "candidate"):
        pair = bundles.get(lane)
        if type(pair) is not dict or pair.get("before") != pair.get("after"):
            _fail(f"certified {lane} bundle custody is not stable")
        snapshot = pair.get("before")
        if (
            type(snapshot) is not dict
            or type(snapshot.get("path")) is not str
            or type(snapshot.get("manifest_sha256")) is not str
            or _SHA256.fullmatch(snapshot["manifest_sha256"]) is None
        ):
            _fail(f"certified {lane} bundle snapshot is malformed")
        snapshots[lane] = snapshot
    records = evidence.get("records")
    if type(records) is not list:
        _fail("certified quality evidence records are unavailable")
    chinese = next(
        (
            record
            for record in records
            if type(record) is dict and record.get("prompt_id") == PROMPT_ID
        ),
        None,
    )
    if type(chinese) is not dict:
        _fail("certified Chinese quality record is unavailable")
    reference = chinese.get("reference")
    if type(reference) is not dict or type(reference.get("runs")) is not list:
        _fail("certified Chinese BF16 trajectory is unavailable")
    prompt, teacher = _certified_chinese_scope(
        chinese.get("prompt_token_ids"), reference["runs"][0]
    )
    return {
        "quality_evidence_content_sha256": envelope["content_sha256"],
        "contract_content_sha256": evidence["contract_sha256"],
        "candidate_id": CANDIDATE_ID,
        "reference_bundle_path": snapshots["reference"]["path"],
        "candidate_bundle_path": snapshots["candidate"]["path"],
        "reference_manifest_sha256": snapshots["reference"]["manifest_sha256"],
        "candidate_manifest_sha256": snapshots["candidate"]["manifest_sha256"],
        "prompt_token_ids": prompt,
        "teacher_token_ids": teacher,
    }


def _validate_pair(value: object, inputs: dict[str, object]) -> dict[str, object]:
    if type(value) is not dict or set(value) != _PAIR_FIELDS:
        _fail("capture backend pair identity fields drifted")
    if (
        type(value.get("pair_id")) is not str
        or _IDENTIFIER.fullmatch(value["pair_id"]) is None
        or type(value.get("process_id")) is not int
        or value["process_id"] <= 0
        or type(value.get("reference_handle_id")) is not str
        or type(value.get("candidate_handle_id")) is not str
        or value["reference_handle_id"] == value["candidate_handle_id"]
    ):
        _fail("capture backend did not open two independent same-process handles")
    for lane in ("reference", "candidate"):
        if value.get(f"{lane}_manifest_sha256") != inputs[f"{lane}_manifest_sha256"]:
            _fail(f"capture backend {lane} handle is not bound to bundle custody")
    return json.loads(canonical_bytes(value))


def _expected_w8_module_paths() -> list[str]:
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
    return [path for path in paths if path not in retained]


def _validate_capture(
    value: object,
    inputs: dict[str, object],
    trajectory: dict[str, object],
) -> dict[str, object]:
    if type(value) is not dict or set(value) != _CAPTURE_FIELDS:
        _fail("state-aligned capture fields drifted")
    if (
        value.get("format") != CAPTURE_FORMAT
        or value.get("prompt_id") != PROMPT_ID
        or value.get("prompt_token_ids") != inputs["prompt_token_ids"]
        or value.get("teacher_token_ids") != inputs["teacher_token_ids"]
        or value.get("retained_bf16_paths") != RETAINED_BF16_PATHS
    ):
        _fail("state-aligned capture is not bound to the fixed Chinese teacher prefix")
    paths = value.get("w8_module_paths")
    expected_paths = _expected_w8_module_paths()
    if (
        type(paths) is not list
        or any(type(path) is not str for path in paths)
        or any(_MODULE_PATH.fullmatch(path) is None for path in paths)
        or paths != expected_paths
        or value.get("w8_module_paths_sha256") != object_sha256(paths)
    ):
        _fail("state-aligned capture W8 module portfolio is invalid")
    runs = value.get("runs")
    if type(runs) is not list or len(runs) != 2 or runs[0] != runs[1]:
        _fail("state-aligned capture must contain two identical runs")
    run = runs[0]
    if type(run) is not dict or set(run) != _RUN_FIELDS:
        _fail("state-aligned capture run fields drifted")
    teacher = inputs["teacher_token_ids"]
    assert type(teacher) is list
    steps = run.get("step_metrics")
    if type(steps) is not list or len(steps) != len(teacher):
        _fail("state-aligned capture step metrics are incomplete")
    for index, step in enumerate(steps):
        if type(step) is not dict or set(step) != _STEP_FIELDS:
            _fail("state-aligned capture step metric fields drifted")
        token_fields = (
            step.get("reference_token_id"),
            step.get("reference_top1_token_id"),
            step.get("candidate_top1_token_id"),
        )
        margin_fields = (
            step.get("reference_top1_margin_micro"),
            step.get("candidate_reference_token_margin_micro"),
        )
        if (
            step.get("step_index") != index
            or token_fields[0] != teacher[index]
            or token_fields[1] != teacher[index]
            or any(
                type(token) is not int or token < 0 or token >= 248320
                for token in token_fields
            )
            or any(
                type(margin) is not int or abs(margin) > 10**12
                for margin in margin_fields
            )
            or margin_fields[0] < 0
        ):
            _fail("state-aligned capture step metric is invalid")
    first = trajectory["first_divergence"]
    if type(first) is dict:
        first_index = first["step_index"]
        if any(
            steps[index]["candidate_top1_token_id"] != teacher[index]
            for index in range(first_index)
        ):
            _fail("teacher-forced capture does not reproduce the free-run exact prefix")
        if steps[first_index]["candidate_top1_token_id"] != first["candidate_token_id"]:
            _fail("teacher-forced capture does not reproduce the first free-run flip")
    modules = run.get("module_metrics")
    if type(modules) is not list or len(modules) != len(paths):
        _fail("state-aligned capture module metrics are incomplete")
    for index, metric in enumerate(modules):
        if type(metric) is not dict or set(metric) != _MODULE_FIELDS:
            _fail("state-aligned capture module metric fields drifted")
        first_nonzero = metric.get("first_nonzero_step")
        if (
            metric.get("path") != paths[index]
            or metric.get("tier") != "w8"
            or metric.get("sample_count") != len(teacher)
            or type(metric.get("relative_l1_error_ppm")) is not int
            or metric["relative_l1_error_ppm"] < 0
            or metric["relative_l1_error_ppm"] > 10**12
            or type(metric.get("max_abs_error_micro")) is not int
            or metric["max_abs_error_micro"] < 0
            or metric["max_abs_error_micro"] > 10**12
            or (
                first_nonzero is not None
                and (
                    type(first_nonzero) is not int
                    or first_nonzero < 0
                    or first_nonzero >= len(teacher)
                )
            )
        ):
            _fail("state-aligned capture module metric is invalid")
    return json.loads(canonical_bytes(value))


def _teacher_stability(capture: dict[str, object]) -> dict[str, object]:
    runs = capture["runs"]
    assert type(runs) is list and type(runs[0]) is dict
    steps = runs[0]["step_metrics"]
    assert type(steps) is list
    matches = [
        step["candidate_top1_token_id"] == step["reference_token_id"] for step in steps
    ]
    first_flip = next(
        (index for index, matched in enumerate(matches) if not matched), None
    )
    margins = [step["candidate_reference_token_margin_micro"] for step in steps]
    erosions = [
        max(
            0,
            step["reference_top1_margin_micro"]
            - step["candidate_reference_token_margin_micro"],
        )
        for step in steps
    ]
    return {
        "metric": "bf16-reference-token-teacher-forced-top1-margin-v1",
        "steps": len(steps),
        "reference_top1_exact": True,
        "candidate_teacher_top1_match_tokens": sum(matches),
        "candidate_teacher_top1_match_ppm": sum(matches) * 1_000_000 // len(steps),
        "first_candidate_teacher_flip_step_index": first_flip,
        "minimum_candidate_teacher_margin_micro": min(margins),
        "maximum_margin_erosion_micro": max(erosions),
    }


def _rank_modules(capture: dict[str, object]) -> dict[str, object]:
    runs = capture["runs"]
    assert type(runs) is list and type(runs[0]) is dict
    modules = runs[0]["module_metrics"]
    assert type(modules) is list
    ranked = sorted(
        (metric for metric in modules if metric["relative_l1_error_ppm"] > 0),
        key=lambda metric: (
            -metric["relative_l1_error_ppm"],
            -metric["max_abs_error_micro"],
            metric["path"],
        ),
    )[:3]
    candidates = [
        {
            "rank": index + 1,
            "path": metric["path"],
            "current_tier": "w8",
            "proposed_tier": "bf16",
            "relative_l1_error_ppm": metric["relative_l1_error_ppm"],
            "max_abs_error_micro": metric["max_abs_error_micro"],
            "first_nonzero_step": metric["first_nonzero_step"],
        }
        for index, metric in enumerate(ranked)
    ]
    return {
        "ranking_metric": "same-bf16-input-relative-l1-error-ppm-v1",
        "top1_margin_attribution": "global-step-context-only-not-module-causal-v1",
        "top_k_limit": 3,
        "captured_w8_module_count": len(modules),
        "w8_module_paths_sha256": capture["w8_module_paths_sha256"],
        "top_candidates": candidates,
        "counterfactual_bundle_built": False,
        "dynamic_module_replacement": False,
    }


def diagnose_with_backend(
    envelope: object,
    backend: object,
) -> dict[str, object]:
    """Run a trusted read-only capture backend and construct a scoped receipt."""

    trajectory = analyze_chinese_trajectory(envelope)
    assert type(envelope) is dict
    inputs = _quality_inputs(envelope)
    required_calls = (
        "capabilities",
        "open_pair",
        "capture_state_aligned",
        "close_pair",
    )
    if any(not callable(getattr(backend, name, None)) for name in required_calls):
        _fail("capture backend interface is incomplete")
    capabilities = _validate_capture_capabilities(backend.capabilities())
    pair_raw = None
    try:
        pair_raw = backend.open_pair(inputs)
        pair = _validate_pair(pair_raw, inputs)
        capture_raw = backend.capture_state_aligned(
            pair_raw,
            prompt_token_ids=list(inputs["prompt_token_ids"]),
            teacher_token_ids=list(inputs["teacher_token_ids"]),
            repeats=2,
        )
        capture = _validate_capture(capture_raw, inputs, trajectory)
    except DiagnosticError:
        raise
    except Exception as error:
        raise DiagnosticError(f"capture backend failed: {error}") from error
    finally:
        if pair_raw is not None:
            try:
                backend.close_pair(pair_raw)
            except Exception as error:
                raise DiagnosticError(
                    f"capture backend cleanup failed: {error}"
                ) from error
    stability = _teacher_stability(capture)
    localization = _rank_modules(capture)
    receipt = {
        "format": RECEIPT_FORMAT,
        "schema_version": 1,
        "status": "diagnostic-only",
        "scope": {
            "model": "Qwen/Qwen3.5-0.8B",
            "prompt_id": PROMPT_ID,
            "claim_scope": "fixed-Chinese-prompt-only-never-general-parity-v1",
        },
        "inputs": inputs,
        "pair": pair,
        "execution": capabilities,
        "pinned_public_api_audit": PINNED_PUBLIC_API_AUDIT,
        "trajectory": trajectory,
        "teacher_forced_stability": stability,
        "module_localization": localization,
        "capture_sha256": object_sha256(capture),
        "claims": {
            "trajectory_exact": trajectory["trajectory_exact"],
            "teacher_forced_top1_exact": (
                stability["candidate_teacher_top1_match_tokens"] == stability["steps"]
            ),
            "semantic_equivalence_assessed": False,
            "general_parity": False,
        },
    }
    receipt["content_sha256"] = object_sha256(receipt)
    return receipt


def load_production_backend() -> object:
    """Load the source-bound wrapper without importing MLX or opening weights."""

    source_custody = _trusted_backend_source_custody()
    path = Path(source_custody["capture"]["path"])
    specification = importlib.util.spec_from_file_location(
        "apxinf_mlx_qwen35_state_aligned_capture_v1", path
    )
    if specification is None or specification.loader is None:
        _fail("cannot load the pinned Qwen3.5 state-aligned capture backend")
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    try:
        specification.loader.exec_module(module)
        if _trusted_backend_source_custody() != source_custody:
            _fail("loader or capture backend source changed during import")
        backend = module.load_backend(source_custody=source_custody)
        _validate_capture_capabilities(backend.capabilities())
    except DiagnosticError:
        raise
    except Exception as error:
        raise DiagnosticError(
            f"cannot initialize production capture backend: {error}"
        ) from error
    return backend


def _reject_duplicate_keys(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            _fail(f"JSON contains duplicate key: {key}")
        value[key] = item
    return value


def _read_quality_evidence(path: Path) -> dict[str, object]:
    path = Path(path)
    if not path.is_absolute():
        _fail("--quality-evidence must be an absolute direct regular file")
    try:
        observed = path.lstat()
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise DiagnosticError(f"cannot inspect --quality-evidence: {error}") from error
    if (
        resolved != path
        or stat.S_ISLNK(observed.st_mode)
        or not stat.S_ISREG(observed.st_mode)
        or observed.st_size > 4 * 1024 * 1024
    ):
        _fail("--quality-evidence must be a bounded canonical direct regular file")
    try:
        value = json.loads(
            path.read_bytes(),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda constant: _fail(
                f"quality evidence contains non-finite JSON number {constant}"
            ),
        )
    except DiagnosticError:
        raise
    except (OSError, UnicodeError, ValueError) as error:
        raise DiagnosticError(f"cannot read --quality-evidence: {error}") from error
    if type(value) is not dict:
        _fail("--quality-evidence root must be an object")
    return value


def _prepare_output(path: Path) -> Path:
    path = Path(path)
    if not path.is_absolute():
        _fail("--output must be an absolute no-replace path")
    try:
        parent = path.parent.resolve(strict=True)
        observed = path.parent.lstat()
    except OSError as error:
        raise DiagnosticError(f"cannot inspect output parent: {error}") from error
    if (
        parent != path.parent
        or stat.S_ISLNK(observed.st_mode)
        or not stat.S_ISDIR(observed.st_mode)
        or path != parent / path.name
        or path.name in {"", ".", ".."}
    ):
        _fail("--output must be a canonical direct child of a direct directory")
    try:
        path.lstat()
    except FileNotFoundError:
        return path
    except OSError as error:
        raise DiagnosticError(f"cannot inspect --output: {error}") from error
    _fail("--output already exists; diagnostic publication is no-replace")


def _publish_no_replace(path: Path, value: object) -> None:
    payload = canonical_bytes(value) + b"\n"
    temporary = path.parent / (
        f".{path.name}.apxinf-chinese-diagnostic-{secrets.token_hex(8)}.tmp"
    )
    descriptor = -1
    try:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(temporary, flags, 0o600)
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                _fail("short write while staging diagnostic receipt")
            view = view[written:]
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        try:
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError:
            _fail("--output already exists; diagnostic publication is no-replace")
        directory_flags = os.O_RDONLY
        if hasattr(os, "O_DIRECTORY"):
            directory_flags |= os.O_DIRECTORY
        directory_descriptor = os.open(path.parent, directory_flags)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except DiagnosticError:
        raise
    except OSError as error:
        raise DiagnosticError(f"cannot publish diagnostic receipt: {error}") from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        except OSError:
            pass


def main(argv=None, *, backend_loader=load_production_backend) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quality-evidence", type=Path, required=True)
    parser.add_argument("--inspect-trajectory-only", action="store_true")
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args(argv)
    try:
        envelope = _read_quality_evidence(arguments.quality_evidence)
        trajectory = analyze_chinese_trajectory(envelope)
        if arguments.inspect_trajectory_only:
            if arguments.output is not None:
                _fail("--inspect-trajectory-only does not publish an output")
            summary = {
                "format": "apxinf-mlx-chinese-hybrid-trajectory-summary-v1",
                "status": "trajectory-localized",
                "quality_evidence_content_sha256": envelope["content_sha256"],
                "trajectory": trajectory,
                "claims": {
                    "trajectory_exact": trajectory["trajectory_exact"],
                    "teacher_forced_stability_assessed": False,
                    "semantic_equivalence_assessed": False,
                    "general_parity": False,
                },
            }
        else:
            if arguments.output is None:
                _fail("--output is required for a state-aligned capture receipt")
            output = _prepare_output(arguments.output)
            backend = backend_loader()
            receipt = diagnose_with_backend(envelope, backend)
            _publish_no_replace(output, receipt)
            summary = {
                "format": "apxinf-mlx-chinese-hybrid-diagnostic-summary-v1",
                "status": receipt["status"],
                "published": True,
                "output": str(output),
                "content_sha256": receipt["content_sha256"],
                "claims_general_parity": False,
            }
    except DiagnosticError as error:
        summary = {
            "format": "apxinf-mlx-chinese-hybrid-diagnostic-error-v1",
            "status": "error",
            "published": False,
            "claims_general_parity": False,
            "problems": [str(error)],
        }
        return_code = 2
    else:
        return_code = 0
    sys.stdout.write(canonical_bytes(summary).decode("utf-8") + "\n")
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
