#!/usr/bin/env python3
"""Pure policy primitives for bounded MLX W4/W8/BF16 search.

This module never imports MLX and never opens model weights.  It validates and
hashes offline facts which are later consumed by the bundle builder.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import NoReturn


POLICY_DOCUMENT_FORMAT = "apxinf-mlx-mixed-quant-policy-document-v2"
POLICY_FORMAT = "apxinf-mlx-mixed-quant-policy-v2"
SEARCH_RECEIPT_FORMAT = "apxinf-mlx-mixed-quant-search-receipt-v2"
OBSERVATION_FORMAT = "apxinf-mlx-mixed-quant-observation-v2"
RUNNER_RECEIPT_FORMAT = "apxinf-mlx-mixed-quant-runner-receipt-v2"
CANONICAL_CHAT_PROMPT_IDS = (
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
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REVISION = re.compile(r"^[0-9a-f]{40,64}$")
_SOURCE_FIELDS = {
    "repo_id",
    "revision",
    "source_manifest_sha256",
    "config_sha256",
    "language_schema_sha256",
    "language_tensor_count",
}
_TRACE_FIELDS = {
    "api",
    "semantics",
    "prompt_token_ids",
    "teacher_token_ids",
    "teacher_ids_sha256",
    "teacher_steps",
    "free_run_steps",
    "repeat_count",
}
_POLICY_FIELDS = {
    "format",
    "source",
    "candidate_selector",
    "candidate_modules",
    "candidate_modules_sha256",
    "quantization",
    "trace",
    "quality_suite_sha256",
    "lineage",
    "transition_history",
}
_OBSERVATION_FIELDS = {
    "format",
    "policy_sha256",
    "trace_sha256",
    "quality_suite_sha256",
    "evaluator",
    "localization",
    "runner_receipt_body",
    "runner_receipt_sha256",
}
_RUNNER_RECEIPT_FIELDS = {
    "format",
    "passed",
    "outcome",
    "inputs",
    "input_sha256",
    "program",
    "runtime",
    "bundles",
    "evaluation",
    "decision",
}
_RUNNER_INPUT_FIELDS = {
    "source_manifest_sha256",
    "config_sha256",
    "language_schema_sha256",
    "policy_artifact_sha256",
    "policy_document_sha256",
    "policy_sha256",
    "search_receipt_sha256",
    "candidate_modules_sha256",
    "trace_sha256",
    "quality_suite_sha256",
}
_CANDIDATE_SELECTOR = {
    "format": "canonical-mlx-linear-weight-v1",
    "dtype": "BF16",
    "rank": 2,
    "input_dimension_multiple": 64,
}
_DEFAULT_W4 = {
    "tier": "w4",
    "bits": 4,
    "group_size": 64,
    "mode": "affine",
}


class PolicyError(ValueError):
    """A fail-closed mixed-quantization policy error."""


def _fail(message: str) -> NoReturn:
    raise PolicyError(message)


def canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def object_sha256(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _copy(value: object) -> object:
    return json.loads(canonical_bytes(value))


def _validate_source(source: object) -> dict[str, object]:
    if type(source) is not dict or set(source) != _SOURCE_FIELDS:
        _fail("source contract must contain the exact frozen fields")
    repo_id = source.get("repo_id")
    revision = source.get("revision")
    tensor_count = source.get("language_tensor_count")
    if type(repo_id) is not str or not repo_id or repo_id.strip() != repo_id:
        _fail("source repo_id must be a non-empty canonical string")
    if type(revision) is not str or _REVISION.fullmatch(revision) is None:
        _fail("source revision must be a lowercase immutable commit hash")
    for field in (
        "source_manifest_sha256",
        "config_sha256",
        "language_schema_sha256",
    ):
        value = source.get(field)
        if type(value) is not str or _SHA256.fullmatch(value) is None:
            _fail(f"source {field} must be a lowercase SHA-256")
    if type(tensor_count) is not int or tensor_count <= 0:
        _fail("source language_tensor_count must be a positive integer")
    return _copy(source)  # type: ignore[return-value]


def _validate_candidates(candidates: object) -> list[dict[str, object]]:
    if type(candidates) is not list or not candidates:
        _fail("candidate_modules must be a non-empty list")
    normalized: list[dict[str, object]] = []
    for candidate in candidates:
        if type(candidate) is not dict or set(candidate) != {"path", "dtype", "shape"}:
            _fail("each candidate module must contain path, dtype, and shape")
        path = candidate.get("path")
        shape = candidate.get("shape")
        if (
            type(path) is not str
            or not path
            or path.endswith(".weight")
            or path.strip() != path
        ):
            _fail("candidate paths must be canonical module paths without .weight")
        if candidate.get("dtype") != "BF16":
            _fail("candidate modules must originate from BF16 weights")
        if (
            type(shape) is not list
            or len(shape) != 2
            or any(type(dimension) is not int or dimension <= 0 for dimension in shape)
            or shape[-1] % 64 != 0
        ):
            _fail("candidate shapes must be positive rank-2 and group-64 eligible")
        normalized.append(_copy(candidate))  # type: ignore[arg-type]
    paths = [candidate["path"] for candidate in normalized]
    if paths != sorted(set(paths)):
        _fail("candidate module paths must be unique and sorted")
    return normalized


def _validate_trace(trace: object) -> dict[str, object]:
    if type(trace) is not dict or set(trace) != _TRACE_FIELDS:
        _fail("trace contract fields drifted")
    teacher = trace.get("teacher_token_ids")
    if (
        trace.get("api") != "mlx_lm.generate.generate_step"
        or trace.get("semantics") != "mlx-generate-step-argmax-v1"
        or trace.get("prompt_token_ids") != list(CANONICAL_CHAT_PROMPT_IDS)
        or trace.get("teacher_steps") != 128
        or trace.get("free_run_steps") != 128
        or trace.get("repeat_count") != 2
        or type(teacher) is not list
        or len(teacher) != 128
        or any(type(token) is not int or token < 0 for token in teacher)
    ):
        _fail(
            "trace must be the canonical 13-token, 128-step, repeated argmax contract"
        )
    if trace.get("teacher_ids_sha256") != object_sha256(teacher):
        _fail("teacher trajectory hash does not match its token IDs")
    return _copy(trace)  # type: ignore[return-value]


def create_initial_policy_document(
    source: object,
    candidate_modules: object,
    trace: object,
    quality_suite_sha256: object,
) -> dict[str, object]:
    """Create the deterministic generation-zero, all-W4 policy document."""

    frozen_source = _validate_source(source)
    frozen_candidates = _validate_candidates(candidate_modules)
    frozen_trace = _validate_trace(trace)
    if (
        type(quality_suite_sha256) is not str
        or _SHA256.fullmatch(quality_suite_sha256) is None
    ):
        _fail("quality suite SHA-256 must be a lowercase SHA-256")
    policy = {
        "format": POLICY_FORMAT,
        "source": frozen_source,
        "candidate_selector": dict(_CANDIDATE_SELECTOR),
        "candidate_modules": frozen_candidates,
        "candidate_modules_sha256": object_sha256(frozen_candidates),
        "quantization": {
            "default": dict(_DEFAULT_W4),
            "overrides": [],
        },
        "trace": frozen_trace,
        "quality_suite_sha256": quality_suite_sha256,
        "lineage": {
            "generation": 0,
            "parent_policy_sha256": None,
            "observation_sha256": None,
            "transition": None,
        },
        "transition_history": [],
    }
    policy_sha256 = object_sha256(policy)
    receipt = {
        "format": SEARCH_RECEIPT_FORMAT,
        "status": "initial-all-w4",
        "input_policy_sha256": None,
        "observation_sha256": None,
        "observation": None,
        "output_policy_sha256": policy_sha256,
        "candidate_modules_sha256": policy["candidate_modules_sha256"],
        "trace_sha256": object_sha256(frozen_trace),
        "quality_suite_sha256": quality_suite_sha256,
        "decision": {
            "default_tier": "w4",
            "changed_modules": [],
            "formal_performance_claim": False,
            "exact_trajectory_claim": False,
        },
    }
    return {
        "format": POLICY_DOCUMENT_FORMAT,
        "policy": policy,
        "policy_sha256": policy_sha256,
        "search_receipt": receipt,
        "search_receipt_sha256": object_sha256(receipt),
    }


def _validate_overrides(
    overrides: object, candidate_paths: frozenset[str]
) -> list[dict[str, object]]:
    if type(overrides) is not list:
        _fail("quantization overrides must be a list")
    normalized: list[dict[str, object]] = []
    for override in overrides:
        if type(override) is not dict:
            _fail("each quantization override must be an object")
        tier = override.get("tier")
        if tier == "w8":
            if override != {
                "path": override.get("path"),
                "tier": "w8",
                "bits": 8,
                "group_size": 64,
                "mode": "affine",
            }:
                _fail("W8 overrides must pin affine bits=8 and group_size=64")
        elif tier == "bf16":
            if set(override) != {"path", "tier"}:
                _fail("BF16 overrides may contain only path and tier")
        else:
            _fail("override tier must be w8 or bf16")
        path = override.get("path")
        if type(path) is not str or path not in candidate_paths:
            _fail("override path is outside the frozen candidate set")
        normalized.append(_copy(override))  # type: ignore[arg-type]
    paths = [override["path"] for override in normalized]
    if paths != sorted(set(paths)):
        _fail("override paths must be unique and sorted")
    return normalized


def _validate_transition_history(
    policy: dict[str, object], candidate_paths: frozenset[str]
) -> dict[str, object]:
    lineage = policy.get("lineage")
    history = policy.get("transition_history")
    if type(lineage) is not dict or set(lineage) != {
        "generation",
        "parent_policy_sha256",
        "observation_sha256",
        "transition",
    }:
        _fail("policy lineage fields drifted")
    generation = lineage.get("generation")
    if (
        type(generation) is not int
        or generation < 0
        or type(history) is not list
        or len(history) != generation
    ):
        _fail("policy transition history length does not match generation")

    reconstructed = _copy(policy)
    assert type(reconstructed) is dict
    reconstructed_quantization = reconstructed["quantization"]
    assert type(reconstructed_quantization) is dict
    reconstructed_quantization["overrides"] = []
    reconstructed["lineage"] = {
        "generation": 0,
        "parent_policy_sha256": None,
        "observation_sha256": None,
        "transition": None,
    }
    reconstructed["transition_history"] = []
    expected_parent = object_sha256(reconstructed)
    tiers: dict[str, str] = {}
    normalized_history: list[dict[str, object]] = []
    for index, entry in enumerate(history, start=1):
        if type(entry) is not dict or set(entry) != {
            "generation",
            "parent_policy_sha256",
            "observation_sha256",
            "transition",
        }:
            _fail("policy transition history entry fields drifted")
        parent = entry.get("parent_policy_sha256")
        observation = entry.get("observation_sha256")
        transition = entry.get("transition")
        if (
            entry.get("generation") != index
            or parent != expected_parent
            or type(observation) is not str
            or _SHA256.fullmatch(observation) is None
            or type(transition) is not dict
            or set(transition) != {"path", "from", "to"}
        ):
            _fail("policy transition history hash or generation drifted")
        path = transition.get("path")
        previous = transition.get("from")
        following = transition.get("to")
        current = tiers.get(path, "w4") if type(path) is str else None
        if (
            type(path) is not str
            or path not in candidate_paths
            or current != previous
            or (previous, following) not in {("w4", "w8"), ("w8", "bf16")}
        ):
            _fail("policy transition history contains a non-sequential upgrade")
        assert type(path) is str and type(following) is str
        tiers[path] = following
        overrides: list[dict[str, object]] = []
        for candidate_path in sorted(tiers):
            tier = tiers[candidate_path]
            if tier == "w8":
                overrides.append(
                    {
                        "path": candidate_path,
                        "tier": "w8",
                        "bits": 8,
                        "group_size": 64,
                        "mode": "affine",
                    }
                )
            else:
                overrides.append({"path": candidate_path, "tier": "bf16"})
        normalized_entry = _copy(entry)
        assert type(normalized_entry) is dict
        normalized_history.append(normalized_entry)
        reconstructed_quantization["overrides"] = overrides
        reconstructed["lineage"] = normalized_entry
        reconstructed["transition_history"] = list(normalized_history)
        expected_parent = object_sha256(reconstructed)
    if reconstructed != policy:
        _fail("policy transition history does not reconstruct the policy")
    return lineage


def validate_policy_document(document: object) -> dict[str, object]:
    """Validate a complete policy document and return a detached copy."""

    if type(document) is not dict or set(document) != {
        "format",
        "policy",
        "policy_sha256",
        "search_receipt",
        "search_receipt_sha256",
    }:
        _fail("policy document fields drifted")
    if document.get("format") != POLICY_DOCUMENT_FORMAT:
        _fail("policy document format drifted")
    policy = document.get("policy")
    if type(policy) is not dict or set(policy) != _POLICY_FIELDS:
        _fail("policy fields drifted")
    if policy.get("format") != POLICY_FORMAT:
        _fail("policy format drifted")
    observed_hash = document.get("policy_sha256")
    if type(observed_hash) is not str or observed_hash != object_sha256(policy):
        _fail("policy SHA-256 does not match the canonical policy")
    _validate_source(policy.get("source"))
    candidates = _validate_candidates(policy.get("candidate_modules"))
    if policy.get("candidate_selector") != _CANDIDATE_SELECTOR:
        _fail("candidate selector drifted")
    if policy.get("candidate_modules_sha256") != object_sha256(candidates):
        _fail("candidate module set hash drifted")
    trace = _validate_trace(policy.get("trace"))
    quality_suite_sha256 = policy.get("quality_suite_sha256")
    if (
        type(quality_suite_sha256) is not str
        or _SHA256.fullmatch(quality_suite_sha256) is None
    ):
        _fail("quality suite SHA-256 must be a lowercase SHA-256")
    quantization = policy.get("quantization")
    if type(quantization) is not dict or set(quantization) != {
        "default",
        "overrides",
    }:
        _fail("quantization policy fields drifted")
    if quantization.get("default") != _DEFAULT_W4:
        _fail("mixed search must retain an all-W4 default")
    candidate_paths = frozenset(candidate["path"] for candidate in candidates)
    _validate_overrides(quantization.get("overrides"), candidate_paths)
    lineage = _validate_transition_history(policy, candidate_paths)
    receipt = document.get("search_receipt")
    if type(receipt) is not dict or receipt.get("format") != SEARCH_RECEIPT_FORMAT:
        _fail("search receipt format drifted")
    if document.get("search_receipt_sha256") != object_sha256(receipt):
        _fail("search receipt SHA-256 does not match its canonical receipt")
    if receipt.get("output_policy_sha256") != observed_hash:
        _fail("search receipt is not bound to its output policy")
    if receipt.get("candidate_modules_sha256") != policy.get(
        "candidate_modules_sha256"
    ):
        _fail("search receipt candidate hash drifted")
    if receipt.get("trace_sha256") != object_sha256(trace):
        _fail("search receipt trace hash drifted")
    if receipt.get("quality_suite_sha256") != quality_suite_sha256:
        _fail("search receipt quality suite hash drifted")
    _validate_receipt_semantics(
        receipt,
        policy=policy,
        policy_sha256=observed_hash,
        lineage=lineage,
        trace=trace,
        candidate_paths=candidate_paths,
    )
    return _copy(document)  # type: ignore[return-value]


def _validate_token_runs(
    value: object, *, repeat_count: int, step_count: int, label: str
) -> list[list[int]]:
    if type(value) is not list or len(value) != repeat_count:
        _fail(f"{label} must contain exactly {repeat_count} runs")
    runs: list[list[int]] = []
    for run in value:
        if (
            type(run) is not list
            or len(run) != step_count
            or any(type(token) is not int or token < 0 for token in run)
        ):
            _fail(f"each {label} run must contain {step_count} token IDs")
        runs.append(list(run))
    return runs


def _first_divergence(run: list[int], teacher: list[int]) -> int | None:
    for index, (observed, expected) in enumerate(zip(run, teacher)):
        if observed != expected:
            return index
    return None


def _is_sha256(value: object) -> bool:
    return type(value) is str and _SHA256.fullmatch(value) is not None


def _canonical_artifact_sha256(document: object) -> str:
    return hashlib.sha256(canonical_bytes(document) + b"\n").hexdigest()


def _runner_gate_analysis(
    teacher_runs: list[list[int]],
    async_runs: list[list[int]],
    teacher: list[int],
) -> dict[str, object]:
    teacher_mismatch_count = sum(
        actual != expected
        for run in teacher_runs
        for actual, expected in zip(run, teacher, strict=True)
    )
    async_mismatch_count = sum(
        actual != expected
        for run in async_runs
        for actual, expected in zip(run, teacher, strict=True)
    )

    def first_divergence(run: list[int]) -> int | None:
        return _first_divergence(run, teacher)

    return {
        "teacher_forced_exact": all(run == teacher for run in teacher_runs),
        "async_free_run_exact": all(run == teacher for run in async_runs),
        "repeated_identically": (
            teacher_runs[0] == teacher_runs[1] and async_runs[0] == async_runs[1]
        ),
        "teacher_forced_mismatch_count": teacher_mismatch_count,
        "async_free_run_mismatch_count": async_mismatch_count,
        "mismatch_count": teacher_mismatch_count + async_mismatch_count,
        "teacher_forced_first_divergence_step": first_divergence(teacher_runs[0]),
        "async_free_run_first_divergence_step": first_divergence(async_runs[0]),
        "teacher_forced_repeat_sha256": [object_sha256(run) for run in teacher_runs],
        "async_free_run_repeat_sha256": [object_sha256(run) for run in async_runs],
    }


def _expected_upgrade_transition(
    policy: dict[str, object], path: str
) -> dict[str, object]:
    quantization = policy.get("quantization")
    if type(quantization) is not dict:
        _fail("selected counterfactual policy quantization drifted")
    overrides = quantization.get("overrides")
    if type(overrides) is not list:
        _fail("selected counterfactual policy overrides drifted")
    tiers = {override["path"]: override["tier"] for override in overrides}
    current = tiers.get(path, "w4")
    if current == "w4":
        following = "w8"
    elif current == "w8":
        following = "bf16"
    else:
        _fail("selected counterfactual path is already BF16")
    return {"path": path, "from": current, "to": following}


def _validate_selected_counterfactual(
    *,
    policy: dict[str, object],
    path: str,
    decision: dict[str, object],
    bundles: dict[str, object],
    attribution: dict[str, object],
    current_teacher_runs: list[list[int]],
    current_async_runs: list[list[int]],
    teacher: list[int],
    repeat_count: int,
    teacher_steps: int,
    expected_transition: dict[str, object] | None,
) -> None:
    if expected_transition is None:
        transition = _expected_upgrade_transition(policy, path)
    else:
        transition = expected_transition
        if transition.get("path") != path or (
            transition.get("from"),
            transition.get("to"),
        ) not in {("w4", "w8"), ("w8", "bf16")}:
            _fail("selected counterfactual expected transition drifted")
    if decision.get("transition") != transition:
        _fail("selected counterfactual decision transition drifted")

    selected = attribution.get("selected_counterfactual")
    selected_fields = {
        "path",
        "transition",
        "manifest_sha256",
        "teacher_forced_token_ids",
        "async_free_run_token_ids",
        "analysis",
        "mismatch_improvement",
        "teacher_async_no_regression",
    }
    if type(selected) is not dict or set(selected) != selected_fields:
        _fail("selected counterfactual evidence fields drifted")
    manifest_sha256 = selected.get("manifest_sha256")
    if (
        selected.get("path") != path
        or selected.get("transition") != transition
        or not _is_sha256(manifest_sha256)
    ):
        _fail("selected counterfactual identity drifted")
    selected_teacher_runs = _validate_token_runs(
        selected.get("teacher_forced_token_ids"),
        repeat_count=repeat_count,
        step_count=teacher_steps,
        label="selected counterfactual teacher-forced token IDs",
    )
    selected_async_runs = _validate_token_runs(
        selected.get("async_free_run_token_ids"),
        repeat_count=repeat_count,
        step_count=teacher_steps,
        label="selected counterfactual async free-run token IDs",
    )
    current_analysis = _runner_gate_analysis(
        current_teacher_runs, current_async_runs, teacher
    )
    selected_analysis = _runner_gate_analysis(
        selected_teacher_runs, selected_async_runs, teacher
    )
    if selected.get("analysis") != selected_analysis:
        _fail("selected counterfactual trajectory analysis drifted")
    current_teacher_first = current_analysis["teacher_forced_first_divergence_step"]
    selected_teacher_first = selected_analysis["teacher_forced_first_divergence_step"]
    current_async_first = current_analysis["async_free_run_first_divergence_step"]
    selected_async_first = selected_analysis["async_free_run_first_divergence_step"]
    teacher_no_regression = selected_analysis[
        "teacher_forced_mismatch_count"
    ] <= current_analysis["teacher_forced_mismatch_count"] and (
        128 if selected_teacher_first is None else selected_teacher_first
    ) >= (128 if current_teacher_first is None else current_teacher_first)
    async_no_regression = selected_analysis[
        "async_free_run_mismatch_count"
    ] <= current_analysis["async_free_run_mismatch_count"] and (
        128 if selected_async_first is None else selected_async_first
    ) >= (128 if current_async_first is None else current_async_first)
    improvement = (
        current_analysis["mismatch_count"] - selected_analysis["mismatch_count"]
    )
    if (
        selected_analysis["repeated_identically"] is not True
        or not teacher_no_regression
        or not async_no_regression
        or improvement <= 0
        or selected.get("mismatch_improvement") != improvement
        or selected.get("teacher_async_no_regression") is not True
    ):
        _fail(
            "selected counterfactual failed the deterministic strict improvement gate"
        )

    counterfactuals = bundles.get("counterfactuals")
    assert type(counterfactuals) is list
    descriptors = [item for item in counterfactuals if item.get("path") == path]
    if len(descriptors) != 1:
        _fail("selected counterfactual bundle descriptor is not unique")
    descriptor = descriptors[0]
    if (
        descriptor.get("transition") != transition
        or descriptor.get("manifest_sha256") != manifest_sha256
        or not _is_sha256(descriptor.get("screening_manifest_sha256"))
        or descriptor.get("screening_manifest_sha256") != manifest_sha256
    ):
        _fail("selected counterfactual bundle transition or manifest drifted")

    current_screen = attribution.get("current_screen")
    screen_evidence = attribution.get("counterfactual_screens")
    if (
        type(current_screen) is not dict
        or current_screen.get("steps") != 32
        or type(current_screen.get("aggregate_score_ppm")) is not int
        or type(screen_evidence) is not list
    ):
        _fail("selected counterfactual 32-step screen is unavailable")
    matching_screens = [
        item
        for item in screen_evidence
        if type(item) is dict and item.get("path") == path
    ]
    if len(matching_screens) != 1:
        _fail("selected counterfactual 32-step screen is not unique")
    screened = matching_screens[0]
    screen = screened.get("screen")
    if (
        screened.get("transition") != transition
        or screened.get("manifest_sha256")
        != descriptor.get("screening_manifest_sha256")
        or type(screen) is not dict
        or screen.get("steps") != 32
        or type(screen.get("aggregate_score_ppm")) is not int
    ):
        _fail("selected counterfactual 32-step screen drifted")
    screen_improvement = (
        current_screen["aggregate_score_ppm"] - screen["aggregate_score_ppm"]
    )
    if (
        screen_improvement <= 0
        or screened.get("screen_improvement_ppm") != screen_improvement
    ):
        _fail("selected counterfactual did not improve the 32-step screen")


def _validate_runner_receipt(
    body: object,
    *,
    body_sha256: object,
    policy: dict[str, object],
    policy_sha256: str,
    trace: dict[str, object],
    candidate_paths: frozenset[str],
    teacher_runs: list[list[int]],
    async_runs: list[list[int]],
    analysis: dict[str, object],
    expected_policy_document: dict[str, object] | None,
    expected_runner_transition: dict[str, object] | None,
) -> dict[str, object]:
    if type(body) is not dict or set(body) != _RUNNER_RECEIPT_FIELDS:
        _fail("runner receipt body fields drifted")
    if body.get("format") != RUNNER_RECEIPT_FORMAT:
        _fail("runner receipt body format drifted")
    if not _is_sha256(body_sha256) or body_sha256 != object_sha256(body):
        _fail("runner receipt SHA-256 does not match its canonical body")

    inputs = body.get("inputs")
    if type(inputs) is not dict or set(inputs) != _RUNNER_INPUT_FIELDS:
        _fail("runner receipt input fields drifted")
    if body.get("input_sha256") != object_sha256(inputs):
        _fail("runner receipt input SHA-256 drifted")
    if any(not _is_sha256(value) for value in inputs.values()):
        _fail("runner receipt inputs must be lowercase SHA-256 values")
    source = policy.get("source")
    if type(source) is not dict:
        _fail("runner receipt cannot bind a malformed policy source")
    expected_inputs = {
        "source_manifest_sha256": source.get("source_manifest_sha256"),
        "config_sha256": source.get("config_sha256"),
        "language_schema_sha256": source.get("language_schema_sha256"),
        "policy_sha256": policy_sha256,
        "candidate_modules_sha256": policy.get("candidate_modules_sha256"),
        "trace_sha256": object_sha256(trace),
        "quality_suite_sha256": policy.get("quality_suite_sha256"),
    }
    if any(inputs.get(key) != value for key, value in expected_inputs.items()):
        _fail("runner receipt inputs drifted from the frozen policy")
    if expected_policy_document is not None:
        if (
            inputs.get("policy_artifact_sha256")
            != _canonical_artifact_sha256(expected_policy_document)
            or inputs.get("policy_document_sha256")
            != object_sha256(expected_policy_document)
            or inputs.get("search_receipt_sha256")
            != expected_policy_document.get("search_receipt_sha256")
        ):
            _fail("runner receipt policy artifact binding drifted")

    program = body.get("program")
    if type(program) is not dict or set(program) != {
        "artifacts",
        "program_sha256",
    }:
        _fail("runner receipt program fields drifted")
    artifacts = program.get("artifacts")
    if type(artifacts) is not list or not artifacts:
        _fail("runner receipt program must bind at least one artifact")
    artifact_paths: list[str] = []
    for artifact in artifacts:
        if type(artifact) is not dict or set(artifact) != {"path", "size", "sha256"}:
            _fail("runner receipt program artifact fields drifted")
        path = artifact.get("path")
        size = artifact.get("size")
        if (
            type(path) is not str
            or not path
            or path.startswith("/")
            or path.strip() != path
            or any(part in {"", ".", ".."} for part in path.split("/"))
            or type(size) is not int
            or size < 0
            or not _is_sha256(artifact.get("sha256"))
        ):
            _fail("runner receipt program artifact is not canonical")
        artifact_paths.append(path)
    if artifact_paths != sorted(set(artifact_paths)):
        _fail("runner receipt program artifacts must be unique and sorted")
    if program.get("program_sha256") != object_sha256(artifacts):
        _fail("runner receipt program SHA-256 drifted")

    runtime = body.get("runtime")
    required_runtime = {
        "python_executable_sha256",
        "python_version",
        "packages",
        "offline",
        "network_blocked",
        "trust_remote_code",
    }
    if type(runtime) is not dict or not required_runtime.issubset(runtime):
        _fail("runner receipt runtime fields drifted")
    python_version = runtime.get("python_version")
    if (
        not _is_sha256(runtime.get("python_executable_sha256"))
        or type(python_version) is not str
        or not python_version
        or python_version.strip() != python_version
        or runtime.get("offline") is not True
        or runtime.get("network_blocked") is not True
        or runtime.get("trust_remote_code") is not False
    ):
        _fail("runner receipt runtime is not frozen offline")
    packages = runtime.get("packages")
    if type(packages) is not list or not packages:
        _fail("runner receipt runtime must bind package artifacts")
    package_names: list[str] = []
    for package in packages:
        if type(package) is not dict or set(package) != {"name", "version", "sha256"}:
            _fail("runner receipt package fields drifted")
        name = package.get("name")
        version = package.get("version")
        if (
            type(name) is not str
            or not name
            or name.strip() != name
            or type(version) is not str
            or not version
            or version.strip() != version
            or not _is_sha256(package.get("sha256"))
        ):
            _fail("runner receipt package artifact is not canonical")
        package_names.append(name)
    if package_names != sorted(set(package_names)):
        _fail("runner receipt packages must be unique and sorted")

    bundles = body.get("bundles")
    required_bundles = {
        "bf16_reference",
        "current_candidate",
        "counterfactuals",
        "materialization",
        "dynamic_module_replacement",
        "model_bundle_published",
    }
    if type(bundles) is not dict or not required_bundles.issubset(bundles):
        _fail("runner receipt bundle fields drifted")
    if (
        bundles.get("materialization") != "independent-saved-static-verified-reload-v1"
        or bundles.get("dynamic_module_replacement") is not False
        or bundles.get("model_bundle_published") is not False
    ):
        _fail("runner receipt used a forbidden dynamic or publishing path")
    for label in ("bf16_reference", "current_candidate"):
        descriptor = bundles.get(label)
        if type(descriptor) is not dict or not _is_sha256(
            descriptor.get("manifest_sha256")
        ):
            _fail(f"runner receipt {label} bundle is not manifest-bound")
    counterfactuals = bundles.get("counterfactuals")
    if type(counterfactuals) is not list:
        _fail("runner receipt counterfactual bundles must be a list")
    counterfactual_paths: list[str] = []
    for counterfactual in counterfactuals:
        if type(counterfactual) is not dict:
            _fail("runner receipt counterfactual bundle must be an object")
        path = counterfactual.get("path")
        if (
            type(path) is not str
            or path not in candidate_paths
            or not _is_sha256(counterfactual.get("manifest_sha256"))
        ):
            _fail("runner receipt counterfactual bundle is not policy-bound")
        counterfactual_paths.append(path)
    if counterfactual_paths != sorted(set(counterfactual_paths)):
        _fail("runner receipt counterfactual bundles must be unique and sorted")

    evaluation = body.get("evaluation")
    if type(evaluation) is not dict or set(evaluation) != {
        "bf16_reference",
        "current_candidate",
        "attribution",
    }:
        _fail("runner receipt evaluation fields drifted")
    reference = evaluation.get("bf16_reference")
    current = evaluation.get("current_candidate")
    attribution = evaluation.get("attribution")
    raw_fields = {"teacher_forced_token_ids", "async_free_run_token_ids"}
    if (
        type(reference) is not dict
        or not raw_fields.issubset(reference)
        or type(current) is not dict
        or not raw_fields.issubset(current)
        or type(attribution) is not dict
        or not {"screening_steps", *raw_fields}.issubset(attribution)
    ):
        _fail("runner receipt evaluation lacks raw trajectories")
    repeat_count = trace["repeat_count"]
    teacher_steps = trace["teacher_steps"]
    assert type(repeat_count) is int and type(teacher_steps) is int
    reference_teacher = _validate_token_runs(
        reference.get("teacher_forced_token_ids"),
        repeat_count=repeat_count,
        step_count=teacher_steps,
        label="runner BF16 teacher-forced token IDs",
    )
    reference_async = _validate_token_runs(
        reference.get("async_free_run_token_ids"),
        repeat_count=repeat_count,
        step_count=teacher_steps,
        label="runner BF16 async free-run token IDs",
    )
    teacher = trace["teacher_token_ids"]
    assert type(teacher) is list
    if any(run != teacher for run in reference_teacher + reference_async):
        _fail("runner BF16 reference trajectories drifted from the teacher")
    current_teacher = _validate_token_runs(
        current.get("teacher_forced_token_ids"),
        repeat_count=repeat_count,
        step_count=teacher_steps,
        label="runner current teacher-forced token IDs",
    )
    current_async = _validate_token_runs(
        current.get("async_free_run_token_ids"),
        repeat_count=repeat_count,
        step_count=teacher_steps,
        label="runner current async free-run token IDs",
    )
    if current_teacher != teacher_runs or current_async != async_runs:
        _fail("runner receipt current trajectories drifted from the observation")
    if attribution.get("screening_steps") != 32:
        _fail("runner receipt attribution must bind a 32-step screen")
    _validate_token_runs(
        attribution.get("teacher_forced_token_ids"),
        repeat_count=repeat_count,
        step_count=32,
        label="runner attribution teacher-forced token IDs",
    )
    _validate_token_runs(
        attribution.get("async_free_run_token_ids"),
        repeat_count=repeat_count,
        step_count=32,
        label="runner attribution async free-run token IDs",
    )

    decision = body.get("decision")
    required_decision = {
        "outcome",
        "stop_reason",
        "changed_module_count",
        "changed_module_path",
        "exact_trajectory_claim",
        "general_parity_claim",
        "default_ready_claim",
        "formal_performance_claim",
    }
    if type(decision) is not dict or not required_decision.issubset(decision):
        _fail("runner receipt decision fields drifted")
    if (
        decision.get("general_parity_claim") is not False
        or decision.get("default_ready_claim") is not False
        or decision.get("formal_performance_claim") is not False
    ):
        _fail("runner receipt makes a forbidden readiness or parity claim")
    outcome = body.get("outcome")
    if decision.get("outcome") != outcome:
        _fail("runner receipt decision outcome drifted")
    repeated = bool(
        analysis["teacher_forced_repeated_identically"]
        and analysis["async_free_run_repeated_identically"]
    )
    exact = bool(analysis["teacher_forced_exact"] and analysis["async_free_run_exact"])
    changed_path = decision.get("changed_module_path")
    if outcome == "exact":
        valid_outcome = (
            body.get("passed") is True
            and exact
            and decision.get("changed_module_count") == 0
            and changed_path is None
            and decision.get("exact_trajectory_claim") is True
            and decision.get("stop_reason") is None
        )
    elif outcome == "divergent":
        unique_upgrade = (
            decision.get("changed_module_count") == 1
            and type(changed_path) is str
            and changed_path in candidate_paths
            and changed_path in counterfactual_paths
            and decision.get("stop_reason") is None
        )
        no_unique_winner = (
            decision.get("changed_module_count") == 0
            and changed_path is None
            and decision.get("stop_reason") == "no-unique-sensitive-module"
        )
        if unique_upgrade:
            assert type(changed_path) is str
            _validate_selected_counterfactual(
                policy=policy,
                path=changed_path,
                decision=decision,
                bundles=bundles,
                attribution=attribution,
                current_teacher_runs=teacher_runs,
                current_async_runs=async_runs,
                teacher=teacher,
                repeat_count=repeat_count,
                teacher_steps=teacher_steps,
                expected_transition=expected_runner_transition,
            )
        valid_outcome = (
            body.get("passed") is False
            and repeated
            and not exact
            and (unique_upgrade or no_unique_winner)
            and decision.get("exact_trajectory_claim") is False
        )
    elif outcome == "nondeterministic":
        valid_outcome = (
            body.get("passed") is False
            and not repeated
            and decision.get("changed_module_count") == 0
            and changed_path is None
            and decision.get("exact_trajectory_claim") is False
            and decision.get("stop_reason") == "nondeterministic-repeated-trajectories"
        )
    else:
        valid_outcome = False
    if not valid_outcome:
        _fail("runner receipt outcome does not match its raw trajectories")
    return _copy(body)  # type: ignore[return-value]


def _validate_observation(
    observation: object,
    *,
    policy: dict[str, object],
    policy_sha256: str,
    trace: dict[str, object],
    candidate_paths: frozenset[str],
    expected_policy_document: dict[str, object] | None = None,
    expected_runner_transition: dict[str, object] | None = None,
) -> tuple[dict[str, object], dict[str, object]]:
    if type(observation) is not dict or set(observation) != _OBSERVATION_FIELDS:
        _fail("observation fields drifted")
    if observation.get("format") != OBSERVATION_FORMAT:
        _fail("observation format drifted")
    if observation.get("policy_sha256") != policy_sha256:
        _fail("observation is not bound to the input policy")
    if observation.get("trace_sha256") != object_sha256(trace):
        _fail("observation trace hash drifted")
    if observation.get("quality_suite_sha256") != policy.get("quality_suite_sha256"):
        _fail("observation quality suite hash drifted")
    evaluator = observation.get("evaluator")
    if type(evaluator) is not dict or set(evaluator) != {
        "api",
        "semantics",
        "prompt_token_ids",
        "teacher_forced_token_ids",
        "async_free_run_token_ids",
    }:
        _fail("observation evaluator fields drifted")
    if (
        evaluator.get("api") != trace["api"]
        or evaluator.get("semantics") != trace["semantics"]
        or evaluator.get("prompt_token_ids") != trace["prompt_token_ids"]
    ):
        _fail("observation evaluator semantics drifted")
    repeat_count = trace["repeat_count"]
    step_count = trace["teacher_steps"]
    assert type(repeat_count) is int and type(step_count) is int
    teacher_runs = _validate_token_runs(
        evaluator.get("teacher_forced_token_ids"),
        repeat_count=repeat_count,
        step_count=step_count,
        label="teacher-forced token IDs",
    )
    async_runs = _validate_token_runs(
        evaluator.get("async_free_run_token_ids"),
        repeat_count=repeat_count,
        step_count=step_count,
        label="async free-run token IDs",
    )
    teacher = trace["teacher_token_ids"]
    assert type(teacher) is list
    analysis = {
        "teacher_forced_exact": all(run == teacher for run in teacher_runs),
        "async_free_run_exact": all(run == teacher for run in async_runs),
        "teacher_forced_repeated_identically": all(
            run == teacher_runs[0] for run in teacher_runs[1:]
        ),
        "async_free_run_repeated_identically": all(
            run == async_runs[0] for run in async_runs[1:]
        ),
        "teacher_forced_first_divergence_step": _first_divergence(
            teacher_runs[0], teacher
        ),
        "async_free_run_first_divergence_step": _first_divergence(
            async_runs[0], teacher
        ),
        "teacher_forced_repeat_sha256": [object_sha256(run) for run in teacher_runs],
        "async_free_run_repeat_sha256": [object_sha256(run) for run in async_runs],
    }
    runner_receipt_sha256 = observation.get("runner_receipt_sha256")
    runner_receipt = _validate_runner_receipt(
        observation.get("runner_receipt_body"),
        body_sha256=runner_receipt_sha256,
        policy=policy,
        policy_sha256=policy_sha256,
        trace=trace,
        candidate_paths=candidate_paths,
        teacher_runs=teacher_runs,
        async_runs=async_runs,
        analysis=analysis,
        expected_policy_document=expected_policy_document,
        expected_runner_transition=expected_runner_transition,
    )
    localization = observation.get("localization")
    if analysis["teacher_forced_exact"] and analysis["async_free_run_exact"]:
        if localization is not None:
            _fail("exact trajectories must not invent a sensitive-module attribution")
        return _copy(observation), analysis  # type: ignore[return-value]
    if not (
        analysis["teacher_forced_repeated_identically"]
        and analysis["async_free_run_repeated_identically"]
    ):
        if localization is not None:
            _fail("nondeterministic trajectories must not invent localization")
        return _copy(observation), analysis  # type: ignore[return-value]
    runner_decision = runner_receipt["decision"]
    assert type(runner_decision) is dict
    if localization is None:
        if runner_decision.get("stop_reason") == "no-unique-sensitive-module":
            return _copy(observation), analysis  # type: ignore[return-value]
        _fail("divergent trajectories without localization lack a STOP reason")
    if type(localization) is not dict or set(localization) != {
        "method",
        "scope",
        "screening_steps",
        "gate_steps",
        "grouping",
        "ranking_metric",
        "sensitive_module_path",
        "unique_top_candidate",
        "runner_receipt_sha256",
    }:
        _fail("observation localization fields drifted")
    runner_hash = localization.get("runner_receipt_sha256")
    sensitive_path = localization.get("sensitive_module_path")
    if (
        localization.get("method") != "state-aligned-single-module-attribution-v1"
        or localization.get("scope") != "one-module-counterfactual-no-combinations"
        or localization.get("screening_steps") != 32
        or localization.get("gate_steps") != 128
        or localization.get("grouping") != "layer-family-v1"
        or localization.get("ranking_metric") != "hidden-error-plus-top1-margin-v1"
        or localization.get("unique_top_candidate") is not True
        or type(sensitive_path) is not str
        or sensitive_path not in candidate_paths
        or runner_hash != runner_receipt_sha256
        or runner_receipt["decision"].get("changed_module_path") != sensitive_path
    ):
        _fail("observation does not uniquely localize one frozen candidate")
    return _copy(observation), analysis  # type: ignore[return-value]


def _validate_receipt_semantics(
    receipt: dict[str, object],
    *,
    policy: dict[str, object],
    policy_sha256: str,
    lineage: dict[str, object],
    trace: dict[str, object],
    candidate_paths: frozenset[str],
) -> None:
    expected_fields = {
        "format",
        "status",
        "input_policy_sha256",
        "observation_sha256",
        "observation",
        "output_policy_sha256",
        "candidate_modules_sha256",
        "trace_sha256",
        "quality_suite_sha256",
        "decision",
    }
    if set(receipt) != expected_fields:
        _fail("search receipt semantics have unexpected fields")
    status = receipt.get("status")
    decision = receipt.get("decision")
    if type(decision) is not dict:
        _fail("search receipt semantics require a decision object")
    generation = lineage["generation"]
    assert type(generation) is int
    if status == "initial-all-w4":
        if (
            generation != 0
            or receipt.get("input_policy_sha256") is not None
            or receipt.get("observation_sha256") is not None
            or receipt.get("observation") is not None
            or decision
            != {
                "default_tier": "w4",
                "changed_modules": [],
                "formal_performance_claim": False,
                "exact_trajectory_claim": False,
            }
        ):
            _fail("search receipt semantics do not describe initial all-W4")
        return

    observation = receipt.get("observation")
    observation_hash = receipt.get("observation_sha256")
    input_policy_sha256 = receipt.get("input_policy_sha256")
    if (
        type(observation) is not dict
        or type(observation_hash) is not str
        or observation_hash != object_sha256(observation)
        or type(input_policy_sha256) is not str
        or _SHA256.fullmatch(input_policy_sha256) is None
    ):
        _fail("search receipt semantics are not bound to a full observation")
    expected_observation_policy = (
        lineage["parent_policy_sha256"] if status == "advance" else policy_sha256
    )
    if input_policy_sha256 != expected_observation_policy:
        _fail("search receipt semantics input policy hash drifted")
    frozen_observation, analysis = _validate_observation(
        observation,
        policy=policy,
        policy_sha256=input_policy_sha256,
        trace=trace,
        candidate_paths=candidate_paths,
        expected_runner_transition=(
            lineage["transition"]
            if status == "advance" and type(lineage["transition"]) is dict
            else None
        ),
    )
    if object_sha256(frozen_observation) != observation_hash:
        _fail("search receipt semantics observation canonicalization drifted")

    if status == "advance":
        transition = lineage["transition"]
        assert type(transition) is dict
        localization = frozen_observation["localization"]
        if (
            generation <= 0
            or lineage["observation_sha256"] != observation_hash
            or type(localization) is not dict
            or localization.get("sensitive_module_path") != transition.get("path")
            or not analysis["teacher_forced_repeated_identically"]
            or not analysis["async_free_run_repeated_identically"]
            or (analysis["teacher_forced_exact"] and analysis["async_free_run_exact"])
            or decision
            != {
                "transition": f"{transition['from']}-to-{transition['to']}",
                "changed_module_count": 1,
                "changed_module_path": transition["path"],
                "exact_trajectory_claim": False,
                "formal_performance_claim": False,
                "trajectory_analysis": analysis,
                "localization": localization,
            }
        ):
            _fail("search receipt semantics do not prove one sequential upgrade")
        return

    if input_policy_sha256 != policy_sha256:
        _fail("terminal search receipt semantics must preserve the input policy")
    quantization = policy["quantization"]
    assert type(quantization) is dict
    overrides = quantization["overrides"]
    assert type(overrides) is list
    if status in {"exact-pareto", "exact-candidate"}:
        reverse_ablation_required = bool(overrides)
        expected_status = (
            "exact-candidate" if reverse_ablation_required else "exact-pareto"
        )
        if (
            status != expected_status
            or not analysis["teacher_forced_exact"]
            or not analysis["async_free_run_exact"]
            or decision
            != {
                "changed_module_count": 0,
                "exact_trajectory_claim": True,
                "teacher_steps": 128,
                "async_free_run_steps": 128,
                "reverse_ablation_required": reverse_ablation_required,
                "formal_performance_claim": False,
                "trajectory_analysis": analysis,
            }
        ):
            _fail("search receipt semantics do not prove an exact trajectory")
        return

    if status == "stop":
        repeated = bool(
            analysis["teacher_forced_repeated_identically"]
            and analysis["async_free_run_repeated_identically"]
        )
        if not repeated:
            stop_reason = "nondeterministic-repeated-trajectories"
        else:
            localization = frozen_observation["localization"]
            if localization is None:
                runner_body = frozen_observation["runner_receipt_body"]
                assert type(runner_body) is dict
                runner_decision = runner_body["decision"]
                if (
                    type(runner_decision) is not dict
                    or runner_decision.get("stop_reason")
                    != "no-unique-sensitive-module"
                ):
                    _fail("search receipt semantics STOP lacks runner evidence")
                stop_reason = "no-unique-sensitive-module"
            else:
                if type(localization) is not dict:
                    _fail("search receipt semantics STOP lacks localization evidence")
                sensitive = localization.get("sensitive_module_path")
                if type(sensitive) is not str:
                    _fail("search receipt semantics STOP localization path drifted")
                by_path = {override["path"]: override["tier"] for override in overrides}
                if by_path.get(sensitive) != "bf16":
                    _fail("search receipt semantics STOP has an upgradeable module")
                stop_reason = "sensitive-module-already-bf16"
        if decision != {
            "stop_reason": stop_reason,
            "changed_module_count": 0,
            "exact_trajectory_claim": False,
            "formal_performance_claim": False,
            "trajectory_analysis": analysis,
        }:
            _fail("search receipt semantics STOP decision drifted")
        return
    _fail("search receipt semantics status is unsupported")


def _stop_document(
    policy: dict[str, object],
    policy_sha256: str,
    observation: dict[str, object],
    trace: dict[str, object],
    analysis: dict[str, object],
    reason: str,
) -> dict[str, object]:
    receipt = {
        "format": SEARCH_RECEIPT_FORMAT,
        "status": "stop",
        "input_policy_sha256": policy_sha256,
        "observation_sha256": object_sha256(observation),
        "observation": observation,
        "output_policy_sha256": policy_sha256,
        "candidate_modules_sha256": policy["candidate_modules_sha256"],
        "trace_sha256": object_sha256(trace),
        "quality_suite_sha256": policy["quality_suite_sha256"],
        "decision": {
            "stop_reason": reason,
            "changed_module_count": 0,
            "exact_trajectory_claim": False,
            "formal_performance_claim": False,
            "trajectory_analysis": analysis,
        },
    }
    result = {
        "format": POLICY_DOCUMENT_FORMAT,
        "policy": policy,
        "policy_sha256": policy_sha256,
        "search_receipt": receipt,
        "search_receipt_sha256": object_sha256(receipt),
    }
    return validate_policy_document(result)


def advance_policy_document(document: object, observation: object) -> dict[str, object]:
    """Advance at most one module by one tier using one aligned observation."""

    validated = validate_policy_document(document)
    current_status = validated["search_receipt"]["status"]
    if current_status in {"exact-pareto", "exact-candidate", "stop"}:
        _fail(f"search receipt status {current_status!r} is terminal")
    policy = validated["policy"]
    assert type(policy) is dict
    policy_sha256 = validated["policy_sha256"]
    assert type(policy_sha256) is str
    candidates = policy["candidate_modules"]
    assert type(candidates) is list
    candidate_paths = frozenset(candidate["path"] for candidate in candidates)
    trace = policy["trace"]
    assert type(trace) is dict
    frozen_observation, analysis = _validate_observation(
        observation,
        policy=policy,
        policy_sha256=policy_sha256,
        trace=trace,
        candidate_paths=candidate_paths,
        expected_policy_document=validated,
    )
    if not (
        analysis["teacher_forced_repeated_identically"]
        and analysis["async_free_run_repeated_identically"]
    ):
        return _stop_document(
            policy,
            policy_sha256,
            frozen_observation,
            trace,
            analysis,
            "nondeterministic-repeated-trajectories",
        )
    if analysis["teacher_forced_exact"] and analysis["async_free_run_exact"]:
        observation_sha256 = object_sha256(frozen_observation)
        quantization = policy["quantization"]
        assert type(quantization) is dict
        overrides = quantization["overrides"]
        assert type(overrides) is list
        reverse_ablation_required = bool(overrides)
        receipt = {
            "format": SEARCH_RECEIPT_FORMAT,
            "status": (
                "exact-candidate" if reverse_ablation_required else "exact-pareto"
            ),
            "input_policy_sha256": policy_sha256,
            "observation_sha256": observation_sha256,
            "observation": frozen_observation,
            "output_policy_sha256": policy_sha256,
            "candidate_modules_sha256": policy["candidate_modules_sha256"],
            "trace_sha256": object_sha256(trace),
            "quality_suite_sha256": policy["quality_suite_sha256"],
            "decision": {
                "changed_module_count": 0,
                "exact_trajectory_claim": True,
                "teacher_steps": 128,
                "async_free_run_steps": 128,
                "reverse_ablation_required": reverse_ablation_required,
                "formal_performance_claim": False,
                "trajectory_analysis": analysis,
            },
        }
        result = {
            "format": POLICY_DOCUMENT_FORMAT,
            "policy": policy,
            "policy_sha256": policy_sha256,
            "search_receipt": receipt,
            "search_receipt_sha256": object_sha256(receipt),
        }
        return validate_policy_document(result)

    localization = frozen_observation["localization"]
    if localization is None:
        return _stop_document(
            policy,
            policy_sha256,
            frozen_observation,
            trace,
            analysis,
            "no-unique-sensitive-module",
        )
    assert type(localization) is dict
    path = localization["sensitive_module_path"]
    assert type(path) is str
    quantization = policy["quantization"]
    assert type(quantization) is dict
    overrides = quantization["overrides"]
    assert type(overrides) is list
    by_path = {override["path"]: override for override in overrides}
    current = by_path.get(path)
    if current is None:
        previous_tier = "w4"
        next_tier = "w8"
        next_override = {
            "path": path,
            "tier": "w8",
            "bits": 8,
            "group_size": 64,
            "mode": "affine",
        }
    elif current.get("tier") == "w8":
        previous_tier = "w8"
        next_tier = "bf16"
        next_override = {"path": path, "tier": "bf16"}
    else:
        return _stop_document(
            policy,
            policy_sha256,
            frozen_observation,
            trace,
            analysis,
            "sensitive-module-already-bf16",
        )
    by_path[path] = next_override

    next_policy = _copy(policy)
    assert type(next_policy) is dict
    next_quantization = next_policy["quantization"]
    assert type(next_quantization) is dict
    next_quantization["overrides"] = [by_path[key] for key in sorted(by_path)]
    lineage = policy["lineage"]
    assert type(lineage) is dict and type(lineage["generation"]) is int
    observation_sha256 = object_sha256(frozen_observation)
    transition = {"path": path, "from": previous_tier, "to": next_tier}
    next_lineage = {
        "generation": lineage["generation"] + 1,
        "parent_policy_sha256": policy_sha256,
        "observation_sha256": observation_sha256,
        "transition": transition,
    }
    next_policy["lineage"] = next_lineage
    history = next_policy["transition_history"]
    assert type(history) is list
    history.append(_copy(next_lineage))
    next_policy_sha256 = object_sha256(next_policy)
    receipt = {
        "format": SEARCH_RECEIPT_FORMAT,
        "status": "advance",
        "input_policy_sha256": policy_sha256,
        "observation_sha256": observation_sha256,
        "observation": frozen_observation,
        "output_policy_sha256": next_policy_sha256,
        "candidate_modules_sha256": next_policy["candidate_modules_sha256"],
        "trace_sha256": object_sha256(trace),
        "quality_suite_sha256": next_policy["quality_suite_sha256"],
        "decision": {
            "transition": f"{previous_tier}-to-{next_tier}",
            "changed_module_count": 1,
            "changed_module_path": path,
            "exact_trajectory_claim": False,
            "formal_performance_claim": False,
            "trajectory_analysis": analysis,
            "localization": localization,
        },
    }
    result = {
        "format": POLICY_DOCUMENT_FORMAT,
        "policy": next_policy,
        "policy_sha256": next_policy_sha256,
        "search_receipt": receipt,
        "search_receipt_sha256": object_sha256(receipt),
    }
    return validate_policy_document(result)
