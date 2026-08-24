#!/usr/bin/env python3
"""Evidence-first orchestration for one MLX mixed-quant search generation.

The orchestration core is deliberately independent from MLX.  A trusted
backend must materialize each evaluation-only candidate as an independent,
saved, statically verified, freshly reloaded bundle.  The runner never mutates
modules in place and never publishes a model bundle.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.util
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import re
import sys
import tempfile
from typing import NoReturn


SCRIPT_DIR = Path(__file__).resolve().parent
POLICY_MODULE_PATH = SCRIPT_DIR / "mlx_mixed_quant_policy.py"
BUILDER_MODULE_PATH = SCRIPT_DIR / "build_mlx_bundle.py"
QUALITY_SUITE_FORMAT = "apxinf-mlx-mixed-quant-quality-suite-v1"
SCREEN_FORMAT = "apxinf-mlx-mixed-quant-state-aligned-screen-v1"
RUNNER_RECEIPT_FORMAT = "apxinf-mlx-mixed-quant-runner-receipt-v2"
OBSERVATION_FORMAT = "apxinf-mlx-mixed-quant-observation-v2"
MATERIALIZATION = "independent-saved-static-verified-reload-v1"
EXACT_SCOPE = "single-frozen-canonical-trajectory-only-v1"
MULTI_PROMPT_CONTRACT_SHA256 = (
    "d52a79e62827913a34e8f3961233aea6b49d91cc317ab6e4a69405b80d9a311f"
)
W4_BASELINE_EVIDENCE_SHA256 = (
    "04f40f00cb3031a56c53d6e6bbb861f98ba6cbcd272a2e28a4f7185f7bd8373d"
)
W4_BASELINE = {
    "contract_content_sha256": MULTI_PROMPT_CONTRACT_SHA256,
    "evidence_content_sha256": W4_BASELINE_EVIDENCE_SHA256,
    "status": "failed_comparison",
    "candidate_precision_profile": "w4-g64",
    "deterministic_repeats": True,
    "prompt_metrics": [
        {
            "prompt_id": "chinese-explanation",
            "exact_prefix_tokens": 1,
            "position_match_ppm": 31250,
        },
        {
            "prompt_id": "english-explanation",
            "exact_prefix_tokens": 2,
            "position_match_ppm": 62500,
        },
        {
            "prompt_id": "math-structured-json",
            "exact_prefix_tokens": 10,
            "position_match_ppm": 312500,
        },
        {
            "prompt_id": "python-code",
            "exact_prefix_tokens": 0,
            "position_match_ppm": 0,
        },
    ],
    "search_use": "prioritize-minimal-single-module-upgrades-v1",
    "publication_requirement": "rerun-fixed-suite-before-final-builder-v1",
}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class SearchError(ValueError):
    """A fail-closed search contract error."""


def _fail(message: str) -> NoReturn:
    raise SearchError(message)


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
        raise SearchError(f"value is not canonical JSON: {error}") from error


def object_sha256(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _copy(value: object) -> object:
    return json.loads(canonical_bytes(value))


def _load_policy_api() -> object:
    module_name = "_apxinf_mlx_mixed_quant_policy_for_runner"
    spec = importlib.util.spec_from_file_location(module_name, POLICY_MODULE_PATH)
    if spec is None or spec.loader is None:
        _fail("cannot load the mixed-quant policy validator")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as error:
        raise SearchError(
            f"cannot load mixed-quant policy validator: {error}"
        ) from error
    return module


def _load_builder_api() -> object:
    module_name = "_apxinf_mlx_bundle_builder_for_search_runner"
    spec = importlib.util.spec_from_file_location(module_name, BUILDER_MODULE_PATH)
    if spec is None or spec.loader is None:
        _fail("cannot load the certified MLX bundle inspector")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as error:
        sys.modules.pop(module_name, None)
        raise SearchError(
            f"cannot load certified MLX bundle inspector: {error}"
        ) from error
    return module


@dataclass(frozen=True)
class CertifiedGeneration:
    """Detached, structurally certified inputs for exactly one generation."""

    policy_document: dict[str, object]
    policy: dict[str, object]
    policy_sha256: str
    policy_artifact_sha256: str
    quality_suite: dict[str, object]
    quality_suite_sha256: str
    inputs: dict[str, object]
    input_sha256: str


@dataclass(frozen=True)
class LocalCertification:
    """Local source/file records retained for a final post-run recheck."""

    generation: CertifiedGeneration
    builder_api: object
    source_bundle: object
    policy_path: Path
    quality_suite_path: Path
    quality_evidence_path: Path


def _expected_family(path: str) -> tuple[int, str] | None:
    matched = re.fullmatch(r"language_model\.model\.layers\.(\d+)\.([^.]+)\..+", path)
    if matched is not None:
        return int(matched.group(1)), matched.group(2)
    if path == "language_model.model.embed_tokens":
        return -1, "embedding"
    if path == "language_model.lm_head":
        return -1, "output"
    return None


def _validate_quality_suite(
    suite: object, policy: dict[str, object]
) -> dict[str, object]:
    expected_fields = {
        "format",
        "source",
        "candidate_modules_sha256",
        "trace_sha256",
        "generation",
        "screening",
        "search",
        "multi_prompt_baseline",
        "publication",
    }
    if type(suite) is not dict or set(suite) != expected_fields:
        _fail("quality suite fields drifted")
    if suite.get("format") != QUALITY_SUITE_FORMAT:
        _fail("quality suite format drifted")
    if suite.get("source") != policy.get("source"):
        _fail("quality suite source drifted from policy")
    if suite.get("candidate_modules_sha256") != policy.get("candidate_modules_sha256"):
        _fail("quality suite candidate set drifted from policy")
    trace = policy.get("trace")
    if type(trace) is not dict or suite.get("trace_sha256") != object_sha256(trace):
        _fail("quality suite trace hash drifted from policy")

    generation = suite.get("generation")
    if generation != {
        "api": trace.get("api"),
        "semantics": trace.get("semantics"),
        "sampler": "mx.argmax(logprobs,axis=-1)",
        "teacher_steps": 128,
        "async_free_run_steps": 128,
        "repeat_count": 2,
        "stop_on_eos": False,
    }:
        _fail("quality suite generation contract drifted")

    screening = suite.get("screening")
    screening_fields = {
        "state_alignment",
        "steps",
        "grouping",
        "group_map",
        "hook",
        "tensor_dtype",
        "reduction_order",
        "hidden_error_formula",
        "top1_margin_formula",
        "score_formula",
        "aggregation",
        "non_finite",
        "minimum_gate_improvement",
        "unique_winner_delta",
    }
    if type(screening) is not dict or set(screening) != screening_fields:
        _fail("quality suite screening fields drifted")
    if (
        screening.get("state_alignment") != "prompt-plus-bf16-teacher-prefix-v1"
        or screening.get("steps") != 32
        or screening.get("grouping") != "layer-family-v1"
        or screening.get("hook") != "module-output-counterfactual-on-bf16-input-v1"
        or screening.get("tensor_dtype") != "float32"
        or screening.get("reduction_order") != "path-token-output-v1"
        or screening.get("hidden_error_formula") != "mean-absolute-relative-ppm-v1"
        or screening.get("top1_margin_formula") != "top2-margin-erosion-ppm-v1"
        or screening.get("score_formula")
        != "hidden-error-plus-top1-margin-plus-flip-rate-v1"
        or screening.get("aggregation") != "sum-module-score-ppm-v1"
        or screening.get("non_finite") != "reject"
        or type(screening.get("minimum_gate_improvement")) is not int
        or screening["minimum_gate_improvement"] <= 0
        or type(screening.get("unique_winner_delta")) is not int
        or screening["unique_winner_delta"] <= 0
    ):
        _fail("quality suite screening semantics drifted")

    candidates = policy.get("candidate_modules")
    if type(candidates) is not list:
        _fail("policy candidate module set is unavailable")
    candidate_paths = [candidate.get("path") for candidate in candidates]
    group_map = screening.get("group_map")
    if type(group_map) is not list or len(group_map) != len(candidate_paths):
        _fail("quality suite group map must cover every frozen candidate")
    observed_paths: list[str] = []
    for entry in group_map:
        if type(entry) is not dict or set(entry) != {"path", "layer", "family"}:
            _fail("quality suite group map entry fields drifted")
        path = entry.get("path")
        layer = entry.get("layer")
        family = entry.get("family")
        if type(path) is not str or type(layer) is not int or type(family) is not str:
            _fail("quality suite group map entry types drifted")
        expected = _expected_family(path)
        if expected is None or expected != (layer, family):
            _fail("quality suite group map does not match the module path")
        observed_paths.append(path)
    if observed_paths != sorted(candidate_paths) or observed_paths != sorted(
        set(observed_paths)
    ):
        _fail("quality suite group map paths are incomplete, duplicated, or unsorted")

    search = suite.get("search")
    if search != {
        "max_counterfactuals_per_generation": search.get(
            "max_counterfactuals_per_generation"
        )
        if type(search) is dict
        else None,
        "allowed_transitions": ["w4-to-w8", "w8-to-bf16"],
        "changed_module_count": 1,
        "allow_combinations": False,
        "candidate_materialization": MATERIALIZATION,
        "dynamic_module_replacement": False,
    }:
        _fail("quality suite bounded-search contract drifted")
    assert type(search) is dict
    budget = search["max_counterfactuals_per_generation"]
    if type(budget) is not int or budget <= 0 or budget > len(candidate_paths):
        _fail("quality suite counterfactual budget is invalid")

    if suite.get("publication") != {
        "evaluation_only": True,
        "publishable": False,
        "final_builder_required": True,
        "exact_scope": EXACT_SCOPE,
        "claims_general_parity": False,
        "default_ready": False,
        "formal_performance_claim": False,
    }:
        _fail("quality suite publication/claim boundary drifted")
    if suite.get("multi_prompt_baseline") != W4_BASELINE:
        _fail("quality suite does not bind the certified W4 multi-prompt baseline")
    return _copy(suite)  # type: ignore[return-value]


def certify_documents_for_test(
    policy_document: object,
    quality_suite: object,
    *,
    policy_artifact_sha256: str,
) -> CertifiedGeneration:
    """Pure certification seam used by fake tests; it never inspects a model."""

    if _SHA256.fullmatch(policy_artifact_sha256) is None:
        _fail("policy artifact SHA-256 is invalid")
    policy_api = _load_policy_api()
    try:
        validated = policy_api.validate_policy_document(policy_document)
    except Exception as error:
        raise SearchError(f"policy validation failed: {error}") from error
    expected_artifact_sha256 = hashlib.sha256(
        canonical_bytes(validated) + b"\n"
    ).hexdigest()
    if policy_artifact_sha256 != expected_artifact_sha256:
        _fail("policy artifact SHA-256 does not match canonical policy bytes")
    policy = validated["policy"]
    suite = _validate_quality_suite(quality_suite, policy)
    suite_hash = object_sha256(suite)
    if policy.get("quality_suite_sha256") != suite_hash:
        _fail("policy is not bound to the supplied quality suite")
    source = policy["source"]
    trace = policy["trace"]
    inputs = {
        "source_manifest_sha256": source["source_manifest_sha256"],
        "config_sha256": source["config_sha256"],
        "language_schema_sha256": source["language_schema_sha256"],
        "policy_artifact_sha256": policy_artifact_sha256,
        "policy_document_sha256": object_sha256(validated),
        "policy_sha256": validated["policy_sha256"],
        "search_receipt_sha256": validated["search_receipt_sha256"],
        "candidate_modules_sha256": policy["candidate_modules_sha256"],
        "trace_sha256": object_sha256(trace),
        "quality_suite_sha256": suite_hash,
    }
    return CertifiedGeneration(
        policy_document=_copy(validated),  # type: ignore[arg-type]
        policy=_copy(policy),  # type: ignore[arg-type]
        policy_sha256=validated["policy_sha256"],
        policy_artifact_sha256=policy_artifact_sha256,
        quality_suite=suite,
        quality_suite_sha256=suite_hash,
        inputs=inputs,
        input_sha256=object_sha256(inputs),
    )


def _read_canonical_json_artifact(
    builder_api: object, argument: str, label: str
) -> tuple[Path, dict[str, object], bytes]:
    path = builder_api._require_absolute(argument, label)
    parent = builder_api._require_owned_directory(path.parent, f"{label} parent")
    resolved = parent / path.name
    payload = builder_api._read_regular(resolved, label, builder_api.MAX_JSON_BYTES)
    document = builder_api._parse_json(payload, label)
    if payload != canonical_bytes(document) + b"\n":
        _fail(f"{label} must be canonical JSON followed by one newline")
    return resolved, document, payload


def _validate_w4_baseline_evidence(evidence: object) -> dict[str, object]:
    if type(evidence) is not dict:
        _fail("W4 multi-prompt baseline evidence must be an object")
    frozen = _copy(evidence)
    assert type(frozen) is dict
    content_sha256 = frozen.pop("content_sha256", None)
    if (
        content_sha256 != W4_BASELINE_EVIDENCE_SHA256
        or object_sha256(frozen) != content_sha256
    ):
        _fail("W4 multi-prompt baseline evidence content SHA-256 drifted")
    if evidence.get("status") != "failed_comparison":
        _fail("W4 multi-prompt baseline must remain an explicit failed comparison")
    payload = evidence.get("evidence")
    receipt = evidence.get("validation_receipt")
    if (
        type(payload) is not dict
        or payload.get("contract_sha256") != MULTI_PROMPT_CONTRACT_SHA256
        or type(payload.get("candidate")) is not dict
        or payload["candidate"].get("precision_profile") != "w4-g64"
        or type(receipt) is not dict
        or receipt.get("accepted") is not False
        or receipt.get("claims_general_parity") is not False
    ):
        _fail("W4 multi-prompt baseline claim boundary drifted")
    expected_metrics = {
        item["prompt_id"]: item for item in W4_BASELINE["prompt_metrics"]
    }
    prompts = receipt.get("prompts")
    if type(prompts) is not list or len(prompts) != len(expected_metrics):
        _fail("W4 multi-prompt baseline prompt metrics drifted")
    observed_metrics: dict[str, dict[str, object]] = {}
    for prompt in prompts:
        if type(prompt) is not dict or type(prompt.get("prompt_id")) is not str:
            _fail("W4 multi-prompt baseline prompt metric is malformed")
        ratio = prompt.get("position_match_ratio")
        if type(ratio) not in {int, float}:
            _fail("W4 multi-prompt baseline position ratio is malformed")
        observed_metrics[prompt["prompt_id"]] = {
            "prompt_id": prompt["prompt_id"],
            "exact_prefix_tokens": prompt.get("exact_prefix_tokens"),
            "position_match_ppm": round(ratio * 1_000_000),
        }
    if observed_metrics != expected_metrics:
        _fail("W4 multi-prompt baseline metrics drifted")
    records = payload.get("records")
    if type(records) is not list or len(records) != len(expected_metrics):
        _fail("W4 multi-prompt baseline raw runs drifted")
    for record in records:
        if type(record) is not dict:
            _fail("W4 multi-prompt baseline record is malformed")
        for lane in ("reference", "candidate"):
            lane_value = record.get(lane)
            runs = lane_value.get("runs") if type(lane_value) is dict else None
            if type(runs) is not list or len(runs) != 2 or runs[0] != runs[1]:
                _fail("W4 multi-prompt baseline repeats are not deterministic")
    return _copy(evidence)  # type: ignore[return-value]


def certify_local_generation(
    *,
    source_dir: str,
    source_revision: str,
    policy_path: str,
    quality_suite_path: str,
    quality_evidence_path: str,
) -> LocalCertification:
    """Authenticate local source/policy/suite without importing or running MLX."""

    builder_api = _load_builder_api()
    try:
        source = builder_api._inspect_source(source_dir)
        resolved_policy, policy_document, policy_payload = (
            _read_canonical_json_artifact(builder_api, policy_path, "mixed policy")
        )
        resolved_suite, suite, _ = _read_canonical_json_artifact(
            builder_api, quality_suite_path, "quality suite"
        )
        resolved_evidence, evidence, _ = _read_canonical_json_artifact(
            builder_api, quality_evidence_path, "W4 multi-prompt quality evidence"
        )
        _validate_w4_baseline_evidence(evidence)
        selective = builder_api._load_selective_policy(
            source,
            str(resolved_policy),
            source_revision,
            "affine-w4-g64",
        )
    except SearchError:
        raise
    except Exception as error:
        raise SearchError(
            f"local mixed-search certification failed: {error}"
        ) from error
    if type(selective) is not dict:
        _fail("certified builder did not accept the selective policy")
    generation = certify_documents_for_test(
        policy_document,
        suite,
        policy_artifact_sha256=hashlib.sha256(policy_payload).hexdigest(),
    )
    if (
        selective.get("policy_sha256") != generation.policy_sha256
        or selective.get("source_manifest_sha256")
        != generation.inputs["source_manifest_sha256"]
        or selective.get("candidate_modules_sha256")
        != generation.inputs["candidate_modules_sha256"]
        or selective.get("trace_sha256") != generation.inputs["trace_sha256"]
    ):
        _fail("certified builder facts drifted from runner inputs")
    return LocalCertification(
        generation=generation,
        builder_api=builder_api,
        source_bundle=source,
        policy_path=resolved_policy,
        quality_suite_path=resolved_suite,
        quality_evidence_path=resolved_evidence,
    )


def _recheck_local_certification(certification: LocalCertification) -> None:
    builder_api = certification.builder_api
    source = certification.source_bundle
    try:
        builder_api._assert_records_current(
            source.directory,
            source.records,
            "source directory",
            fixed_names=builder_api.SOURCE_FIXED_FILES,
            shard_pattern=builder_api.SOURCE_SHARD,
        )
        _, policy_document, policy_payload = _read_canonical_json_artifact(
            builder_api, str(certification.policy_path), "mixed policy"
        )
        _, suite, _ = _read_canonical_json_artifact(
            builder_api, str(certification.quality_suite_path), "quality suite"
        )
        _, evidence, _ = _read_canonical_json_artifact(
            builder_api,
            str(certification.quality_evidence_path),
            "W4 multi-prompt quality evidence",
        )
        _validate_w4_baseline_evidence(evidence)
    except SearchError:
        raise
    except Exception as error:
        raise SearchError(f"local mixed-search recheck failed: {error}") from error
    generation = certification.generation
    if (
        hashlib.sha256(policy_payload).hexdigest() != generation.policy_artifact_sha256
        or object_sha256(policy_document) != generation.inputs["policy_document_sha256"]
        or object_sha256(suite) != generation.quality_suite_sha256
    ):
        _fail("policy or quality suite changed during mixed-search evaluation")


def _validate_handle(
    handle: object,
    *,
    policy_sha256: str,
    label: str,
    transition: dict[str, object] | None = None,
) -> dict[str, object]:
    fields = {
        "handle_id",
        "manifest_sha256",
        "policy_sha256",
        "evaluation_only",
        "publishable",
        "materialization",
    }
    if transition is not None:
        fields.add("transition")
    if type(handle) is not dict or set(handle) != fields:
        _fail(f"{label} handle fields drifted")
    if (
        type(handle.get("handle_id")) is not str
        or not handle["handle_id"]
        or type(handle.get("manifest_sha256")) is not str
        or _SHA256.fullmatch(handle["manifest_sha256"]) is None
        or handle.get("policy_sha256") != policy_sha256
        or handle.get("evaluation_only") is not True
        or handle.get("publishable") is not False
        or handle.get("materialization") != MATERIALIZATION
        or (transition is not None and handle.get("transition") != transition)
    ):
        _fail(f"{label} is not an independent evaluation-only materialization")
    return _copy(handle)  # type: ignore[return-value]


def _validate_screen(
    value: object, certified: CertifiedGeneration
) -> dict[str, object]:
    if type(value) is not dict or set(value) != {
        "format",
        "steps",
        "state_alignment",
        "aggregate_score_ppm",
        "module_scores",
    }:
        _fail("state-aligned screening fields drifted")
    if (
        value.get("format") != SCREEN_FORMAT
        or value.get("steps") != 32
        or value.get("state_alignment") != "prompt-plus-bf16-teacher-prefix-v1"
    ):
        _fail("state-aligned screening semantics drifted")
    scores = value.get("module_scores")
    if type(scores) is not list:
        _fail("state-aligned screening scores are unavailable")
    paths = [candidate["path"] for candidate in certified.policy["candidate_modules"]]
    observed: list[str] = []
    total = 0
    for score in scores:
        if type(score) is not dict or set(score) != {
            "path",
            "hidden_error_ppm",
            "top1_margin_erosion_ppm",
            "top1_flip_rate_ppm",
            "score_ppm",
        }:
            _fail("state-aligned module score fields drifted")
        components = [
            score.get("hidden_error_ppm"),
            score.get("top1_margin_erosion_ppm"),
            score.get("top1_flip_rate_ppm"),
        ]
        if any(type(component) is not int or component < 0 for component in components):
            _fail("state-aligned scores must be finite non-negative integers")
        path = score.get("path")
        if type(path) is not str:
            _fail("state-aligned score path drifted")
        computed = sum(components)
        if score.get("score_ppm") != computed:
            _fail("state-aligned module score formula drifted")
        observed.append(path)
        total += computed
    if observed != paths:
        _fail("state-aligned screening must cover frozen candidates in order")
    if value.get("aggregate_score_ppm") != total:
        _fail("state-aligned aggregate score formula drifted")
    return _copy(value)  # type: ignore[return-value]


def _validate_gate(
    value: object, certified: CertifiedGeneration, label: str
) -> dict[str, object]:
    fields = {
        "api",
        "semantics",
        "prompt_token_ids",
        "teacher_forced_token_ids",
        "async_free_run_token_ids",
    }
    trace = certified.policy["trace"]
    if type(value) is not dict or set(value) != fields:
        _fail(f"{label} gate fields drifted")
    if (
        value.get("api") != trace["api"]
        or value.get("semantics") != trace["semantics"]
        or value.get("prompt_token_ids") != trace["prompt_token_ids"]
    ):
        _fail(f"{label} gate execution semantics drifted")
    for run_kind in ("teacher_forced_token_ids", "async_free_run_token_ids"):
        runs = value.get(run_kind)
        if type(runs) is not list or len(runs) != 2:
            _fail(f"{label} {run_kind} must contain two runs")
        for run in runs:
            if (
                type(run) is not list
                or len(run) != 128
                or any(type(token) is not int or token < 0 for token in run)
            ):
                _fail(f"{label} {run_kind} must contain 128 token IDs per run")
    return _copy(value)  # type: ignore[return-value]


def _gate_analysis(gate: dict[str, object], teacher: list[int]) -> dict[str, object]:
    teacher_runs = gate["teacher_forced_token_ids"]
    async_runs = gate["async_free_run_token_ids"]
    assert type(teacher_runs) is list and type(async_runs) is list
    deterministic = (
        teacher_runs[0] == teacher_runs[1] and async_runs[0] == async_runs[1]
    )
    teacher_exact = all(run == teacher for run in teacher_runs)
    async_exact = all(run == teacher for run in async_runs)
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

    def first_divergence(run: object) -> int | None:
        assert type(run) is list
        for index, (actual, expected) in enumerate(zip(run, teacher, strict=True)):
            if actual != expected:
                return index
        return None

    return {
        "teacher_forced_exact": teacher_exact,
        "async_free_run_exact": async_exact,
        "repeated_identically": deterministic,
        "teacher_forced_mismatch_count": teacher_mismatch_count,
        "async_free_run_mismatch_count": async_mismatch_count,
        "mismatch_count": teacher_mismatch_count + async_mismatch_count,
        "teacher_forced_first_divergence_step": first_divergence(teacher_runs[0]),
        "async_free_run_first_divergence_step": first_divergence(async_runs[0]),
        "teacher_forced_repeat_sha256": [object_sha256(run) for run in teacher_runs],
        "async_free_run_repeat_sha256": [object_sha256(run) for run in async_runs],
    }


def _validate_program(program: object) -> dict[str, object]:
    if type(program) is not dict or set(program) != {"artifacts", "program_sha256"}:
        _fail("program receipt fields drifted")
    artifacts = program.get("artifacts")
    if (
        type(artifacts) is not list
        or not artifacts
        or program.get("program_sha256") != object_sha256(artifacts)
    ):
        _fail("program receipt hash drifted")
    paths: list[str] = []
    for artifact in artifacts:
        if type(artifact) is not dict or set(artifact) != {"path", "size", "sha256"}:
            _fail("program artifact fields drifted")
        path = artifact.get("path")
        size = artifact.get("size")
        digest = artifact.get("sha256")
        if (
            type(path) is not str
            or not path
            or path.startswith("/")
            or any(part in {"", ".", ".."} for part in path.split("/"))
            or type(size) is not int
            or size < 0
            or type(digest) is not str
            or _SHA256.fullmatch(digest) is None
        ):
            _fail("program artifact is not canonical")
        paths.append(path)
    if paths != sorted(set(paths)):
        _fail("program artifacts must be unique and sorted")
    return _copy(program)  # type: ignore[return-value]


def _validate_runtime(runtime: object) -> dict[str, object]:
    required = {
        "python_executable_sha256",
        "python_version",
        "packages",
        "offline",
        "network_blocked",
        "trust_remote_code",
    }
    if type(runtime) is not dict or not required.issubset(runtime):
        _fail("runtime receipt fields drifted")
    if (
        type(runtime.get("python_executable_sha256")) is not str
        or _SHA256.fullmatch(runtime["python_executable_sha256"]) is None
        or type(runtime.get("python_version")) is not str
        or not runtime["python_version"]
        or runtime.get("offline") is not True
        or runtime.get("network_blocked") is not True
        or runtime.get("trust_remote_code") is not False
    ):
        _fail("runtime receipt is not frozen offline")
    packages = runtime.get("packages")
    if type(packages) is not list or not packages:
        _fail("runtime receipt must bind package artifacts")
    names: list[str] = []
    for package in packages:
        if type(package) is not dict or set(package) != {"name", "version", "sha256"}:
            _fail("runtime package fields drifted")
        name = package.get("name")
        version = package.get("version")
        digest = package.get("sha256")
        if (
            type(name) is not str
            or not name
            or type(version) is not str
            or not version
            or type(digest) is not str
            or _SHA256.fullmatch(digest) is None
        ):
            _fail("runtime package artifact is not canonical")
        names.append(name)
    if names != sorted(set(names)):
        _fail("runtime packages must be unique and sorted")
    return _copy(runtime)  # type: ignore[return-value]


def _tier_map(policy: dict[str, object]) -> dict[str, str]:
    candidates = policy["candidate_modules"]
    quantization = policy["quantization"]
    assert type(candidates) is list and type(quantization) is dict
    overrides = quantization["overrides"]
    assert type(overrides) is list
    explicit = {override["path"]: override["tier"] for override in overrides}
    return {
        candidate["path"]: explicit.get(candidate["path"], "w4")
        for candidate in candidates
    }


def _next_transition(path: str, tier: str) -> dict[str, object] | None:
    if tier == "w4":
        following = "w8"
    elif tier == "w8":
        following = "bf16"
    else:
        return None
    return {"path": path, "from": tier, "to": following}


def _top_group_paths(
    screen: dict[str, object], certified: CertifiedGeneration
) -> list[str] | None:
    screening = certified.quality_suite["screening"]
    assert type(screening) is dict
    group_map = screening["group_map"]
    assert type(group_map) is list
    by_path = {entry["path"]: (entry["layer"], entry["family"]) for entry in group_map}
    scores = screen["module_scores"]
    assert type(scores) is list
    group_scores: dict[tuple[object, object], int] = {}
    group_paths: dict[tuple[object, object], list[str]] = {}
    for score in scores:
        path = score["path"]
        key = by_path[path]
        group_scores[key] = group_scores.get(key, 0) + score["score_ppm"]
        group_paths.setdefault(key, []).append(path)
    ranked = sorted(group_scores, key=lambda key: (-group_scores[key], key))
    if not ranked:
        return None
    unique_delta = screening["unique_winner_delta"]
    assert type(unique_delta) is int
    if (
        len(ranked) > 1
        and group_scores[ranked[0]] - group_scores[ranked[1]] < unique_delta
    ):
        return None
    return group_paths[ranked[0]]


def _hash_file(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(4 * 1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            digest.update(chunk)
    return size, digest.hexdigest()


def build_program_receipt() -> dict[str, object]:
    """Bind the three local programs that define search and policy semantics."""

    root = SCRIPT_DIR.parent
    artifacts: list[dict[str, object]] = []
    for path in sorted(
        (BUILDER_MODULE_PATH, POLICY_MODULE_PATH, Path(__file__).resolve()),
        key=lambda item: item.relative_to(root).as_posix(),
    ):
        size, digest = _hash_file(path)
        artifacts.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size": size,
                "sha256": digest,
            }
        )
    return {"artifacts": artifacts, "program_sha256": object_sha256(artifacts)}


def build_runtime_receipt() -> dict[str, object]:
    """Bind the interpreter and installed distribution RECORD manifests."""

    executable = Path(sys.executable).resolve(strict=True)
    _, executable_sha256 = _hash_file(executable)
    builder_api = _load_builder_api()
    packages: list[dict[str, object]] = []
    for name in sorted(builder_api.PINNED_PACKAGES):
        try:
            distribution = metadata.distribution(name)
        except metadata.PackageNotFoundError as error:
            raise SearchError(f"runtime package is unavailable: {name}") from error
        version = distribution.version
        expected = builder_api.PINNED_PACKAGES[name]
        record = distribution.read_text("RECORD")
        if version != expected or record is None:
            _fail(f"runtime package {name} is not the pinned RECORD-based artifact")
        packages.append(
            {
                "name": name,
                "version": version,
                "sha256": hashlib.sha256(record.encode("utf-8")).hexdigest(),
            }
        )
    return {
        "python_executable_sha256": executable_sha256,
        "python_version": platform.python_version(),
        "packages": packages,
        "offline": True,
        # Merely setting HF offline variables is not an OS network sandbox.
        # The real backend vertical slice must replace this with verified
        # sandbox evidence before evaluate_local_generation can proceed.
        "network_blocked": False,
        "trust_remote_code": False,
    }


def _write_observation_no_replace(
    builder_api: object, output_argument: str, observation: object
) -> Path:
    requested = builder_api._require_absolute(output_argument, "--output")
    parent = builder_api._require_owned_directory(requested.parent, "output parent")
    destination = parent / requested.name
    builder_api._require_output_absent(destination)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".apxinf-mixed-observation-", suffix=".json", dir=parent
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        payload = canonical_bytes(observation) + b"\n"
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                _fail("short write while staging mixed-search observation")
            view = view[written:]
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        builder_api._rename_no_replace(temporary, destination)
        return destination
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def evaluate_local_generation(
    certification: LocalCertification,
    backend: object,
    *,
    output_path: str,
) -> dict[str, object]:
    """Run once, recheck every input, then publish only the observation JSON."""

    if not isinstance(certification, LocalCertification):
        _fail("local generation was not source-certified")
    observation = evaluate_certified_generation(
        certification.generation,
        backend,
        runtime=build_runtime_receipt(),
        program=build_program_receipt(),
    )
    _recheck_local_certification(certification)
    _write_observation_no_replace(certification.builder_api, output_path, observation)
    return observation


def _close_backend_handles(backend: object, *handles: object | None) -> None:
    failures: list[Exception] = []
    observed: set[int] = set()
    for handle in handles:
        if handle is None or id(handle) in observed:
            continue
        observed.add(id(handle))
        try:
            backend.close(handle)
        except Exception as error:
            failures.append(error)
    if not failures:
        return
    message = "bundle cleanup failed: " + "; ".join(
        " ".join(str(error).split())[:512] or type(error).__name__ for error in failures
    )
    active = sys.exception()
    if active is not None:
        try:
            active.add_note(message)
        except AttributeError:
            pass
        return
    raise SearchError(message) from failures[0]


def evaluate_certified_generation(
    certified: CertifiedGeneration,
    backend: object,
    *,
    runtime: object,
    program: object,
) -> dict[str, object]:
    """Evaluate one certified policy; never save or publish a model bundle."""

    if not isinstance(certified, CertifiedGeneration):
        _fail("generation inputs were not certified")
    frozen_runtime = _validate_runtime(runtime)
    frozen_program = _validate_program(program)
    reference_raw: object | None = None
    current_raw: object | None = None
    try:
        reference_raw = backend.open_bf16_reference(certified)
        reference = _validate_handle(
            reference_raw,
            policy_sha256=certified.policy_sha256,
            label="BF16 reference",
        )
        current_raw = backend.materialize_current(certified)
        current = _validate_handle(
            current_raw,
            policy_sha256=certified.policy_sha256,
            label="current candidate",
        )
        if reference["handle_id"] == current["handle_id"]:
            _fail("BF16 reference and candidate must be independent handles")

        screen = _validate_screen(
            backend.screen_state_aligned(
                reference_raw,
                current_raw,
                certified=certified,
                transition=None,
            ),
            certified,
        )
        reference_gate = _validate_gate(
            backend.evaluate_gate(
                reference_raw,
                certified=certified,
                role="bf16-reference",
            ),
            certified,
            "BF16 reference",
        )
        current_gate = _validate_gate(
            backend.evaluate_gate(
                current_raw,
                certified=certified,
                role="current-candidate",
            ),
            certified,
            "current candidate",
        )
        teacher = certified.policy["trace"]["teacher_token_ids"]
        assert type(teacher) is list
        reference_analysis = _gate_analysis(reference_gate, teacher)
        if not (
            reference_analysis["teacher_forced_exact"]
            and reference_analysis["async_free_run_exact"]
            and reference_analysis["repeated_identically"]
        ):
            _fail("BF16 reference did not reproduce the frozen double gate")
        current_analysis = _gate_analysis(current_gate, teacher)
        # Raw trajectories and the immutable manifest descriptor are detached;
        # release the current model before any counterfactual materialization so
        # the main process holds at most BF16 + one evaluated candidate.
        _close_backend_handles(backend, current_raw)
        current_raw = None
        counterfactual_bundles: dict[str, dict[str, object]] = {}
        counterfactual_screens: list[dict[str, object]] = []
        selected_evaluation: dict[str, object] | None = None
        selected_transition: dict[str, object] | None = None
        if (
            current_analysis["teacher_forced_exact"]
            and current_analysis["async_free_run_exact"]
        ):
            outcome = "exact"
            stop_reason = None
            exact_claim = True
        elif not current_analysis["repeated_identically"]:
            outcome = "nondeterministic"
            stop_reason = "nondeterministic-repeated-trajectories"
            exact_claim = False
        else:
            outcome = "divergent"
            exact_claim = False
            stop_reason = "no-unique-sensitive-module"
            top_paths = _top_group_paths(screen, certified)
            tiers = _tier_map(certified.policy)
            screening_contract = certified.quality_suite["screening"]
            search_contract = certified.quality_suite["search"]
            assert type(screening_contract) is dict and type(search_contract) is dict
            budget = search_contract["max_counterfactuals_per_generation"]
            assert type(budget) is int
            attempts: list[dict[str, object]] = []
            if top_paths is not None:
                for path in top_paths[:budget]:
                    transition = _next_transition(path, tiers[path])
                    if transition is None:
                        continue
                    handle_raw: object | None = None
                    try:
                        handle_raw = backend.materialize_counterfactual(
                            certified, transition
                        )
                        handle = _validate_handle(
                            handle_raw,
                            policy_sha256=certified.policy_sha256,
                            label="counterfactual",
                            transition=transition,
                        )
                        if handle["handle_id"] in {
                            reference["handle_id"],
                            current["handle_id"],
                        }:
                            _fail("counterfactual must use an independent handle")
                        counter_screen = _validate_screen(
                            backend.screen_state_aligned(
                                reference_raw,
                                handle_raw,
                                certified=certified,
                                transition=transition,
                            ),
                            certified,
                        )
                        improvement = (
                            screen["aggregate_score_ppm"]
                            - counter_screen["aggregate_score_ppm"]
                        )
                        evidence = {
                            "path": path,
                            "transition": transition,
                            "manifest_sha256": handle["manifest_sha256"],
                            "screen": counter_screen,
                            "screen_improvement_ppm": improvement,
                        }
                        attempts.append(evidence)
                        counterfactual_screens.append(evidence)
                        counterfactual_bundles[path] = {
                            "path": path,
                            "manifest_sha256": handle["manifest_sha256"],
                            "transition": transition,
                        }
                    finally:
                        if handle_raw is not None:
                            _close_backend_handles(backend, handle_raw)

            minimum = screening_contract["minimum_gate_improvement"]
            unique_delta = screening_contract["unique_winner_delta"]
            assert type(minimum) is int and type(unique_delta) is int
            ranked_attempts = sorted(
                attempts,
                key=lambda attempt: (
                    -attempt["screen_improvement_ppm"],
                    attempt["path"],
                ),
            )
            winner: dict[str, object] | None = None
            if (
                ranked_attempts
                and ranked_attempts[0]["screen_improvement_ppm"] >= minimum
            ):
                if (
                    len(ranked_attempts) == 1
                    or ranked_attempts[0]["screen_improvement_ppm"]
                    - ranked_attempts[1]["screen_improvement_ppm"]
                    >= unique_delta
                ):
                    winner = ranked_attempts[0]

            if winner is not None:
                transition = winner["transition"]
                assert type(transition) is dict
                selected_raw: object | None = None
                try:
                    selected_raw = backend.materialize_counterfactual(
                        certified, transition
                    )
                    selected = _validate_handle(
                        selected_raw,
                        policy_sha256=certified.policy_sha256,
                        label="selected counterfactual",
                        transition=transition,
                    )
                    if selected["manifest_sha256"] != winner["manifest_sha256"]:
                        _fail(
                            "selected counterfactual manifest drifted from its "
                            "32-step screening materialization"
                        )
                    selected_gate = _validate_gate(
                        backend.evaluate_gate(
                            selected_raw,
                            certified=certified,
                            role="selected-counterfactual",
                        ),
                        certified,
                        "selected counterfactual",
                    )
                    selected_analysis = _gate_analysis(selected_gate, teacher)
                    improvement = (
                        current_analysis["mismatch_count"]
                        - selected_analysis["mismatch_count"]
                    )
                    lane_non_regression = (
                        selected_analysis["teacher_forced_mismatch_count"]
                        <= current_analysis["teacher_forced_mismatch_count"]
                        and selected_analysis["async_free_run_mismatch_count"]
                        <= current_analysis["async_free_run_mismatch_count"]
                    )
                    for lane in ("teacher_forced", "async_free_run"):
                        current_first = current_analysis[
                            f"{lane}_first_divergence_step"
                        ]
                        selected_first = selected_analysis[
                            f"{lane}_first_divergence_step"
                        ]
                        current_prefix = 128 if current_first is None else current_first
                        selected_prefix = (
                            128 if selected_first is None else selected_first
                        )
                        lane_non_regression = (
                            lane_non_regression and selected_prefix >= current_prefix
                        )
                    selected_evaluation = {
                        "path": transition["path"],
                        "transition": transition,
                        "manifest_sha256": selected["manifest_sha256"],
                        "teacher_forced_token_ids": selected_gate[
                            "teacher_forced_token_ids"
                        ],
                        "async_free_run_token_ids": selected_gate[
                            "async_free_run_token_ids"
                        ],
                        "analysis": selected_analysis,
                        "mismatch_improvement": improvement,
                        "teacher_async_no_regression": lane_non_regression,
                    }
                    counterfactual_bundles[transition["path"]] = {
                        "path": transition["path"],
                        "manifest_sha256": selected["manifest_sha256"],
                        "screening_manifest_sha256": winner["manifest_sha256"],
                        "transition": transition,
                    }
                    if (
                        selected_analysis["repeated_identically"]
                        and lane_non_regression
                        and improvement >= minimum
                    ):
                        selected_transition = transition
                        stop_reason = None
                    else:
                        _fail(
                            "selected counterfactual failed the deterministic "
                            "128-step improvement gate"
                        )
                finally:
                    if selected_raw is not None:
                        _close_backend_handles(backend, selected_raw)

        decision = {
            "outcome": outcome,
            "stop_reason": stop_reason,
            "changed_module_count": 1 if selected_transition is not None else 0,
            "changed_module_path": (
                selected_transition["path"] if selected_transition is not None else None
            ),
            "transition": selected_transition,
            "exact_trajectory_claim": exact_claim,
            "exact_scope": EXACT_SCOPE if exact_claim else None,
            "general_parity_claim": False,
            "default_ready_claim": False,
            "formal_performance_claim": False,
        }
        receipt_body = {
            "format": RUNNER_RECEIPT_FORMAT,
            "passed": outcome == "exact",
            "outcome": outcome,
            "inputs": certified.inputs,
            "input_sha256": certified.input_sha256,
            "program": frozen_program,
            "runtime": frozen_runtime,
            "bundles": {
                "bf16_reference": reference,
                "current_candidate": current,
                "counterfactuals": [
                    counterfactual_bundles[path]
                    for path in sorted(counterfactual_bundles)
                ],
                "materialization": MATERIALIZATION,
                "dynamic_module_replacement": False,
                "model_bundle_published": False,
            },
            "evaluation": {
                "bf16_reference": {
                    **reference_gate,
                    "analysis": reference_analysis,
                },
                "current_candidate": {
                    **current_gate,
                    "analysis": current_analysis,
                },
                "attribution": {
                    "screening_steps": 32,
                    "teacher_forced_token_ids": [
                        run[:32] for run in current_gate["teacher_forced_token_ids"]
                    ],
                    "async_free_run_token_ids": [
                        run[:32] for run in current_gate["async_free_run_token_ids"]
                    ],
                    "current_screen": screen,
                    "counterfactual_screens": counterfactual_screens,
                    "selected_counterfactual": selected_evaluation,
                    "multi_prompt_baseline": certified.quality_suite[
                        "multi_prompt_baseline"
                    ],
                },
            },
            "decision": decision,
        }
        runner_receipt_sha256 = object_sha256(receipt_body)
        localization = None
        if selected_transition is not None:
            localization = {
                "method": "state-aligned-single-module-attribution-v1",
                "scope": "one-module-counterfactual-no-combinations",
                "screening_steps": 32,
                "gate_steps": 128,
                "grouping": "layer-family-v1",
                "ranking_metric": "hidden-error-plus-top1-margin-v1",
                "sensitive_module_path": selected_transition["path"],
                "unique_top_candidate": True,
                "runner_receipt_sha256": runner_receipt_sha256,
            }
        observation = {
            "format": OBSERVATION_FORMAT,
            "policy_sha256": certified.policy_sha256,
            "trace_sha256": certified.inputs["trace_sha256"],
            "quality_suite_sha256": certified.quality_suite_sha256,
            "evaluator": current_gate,
            "localization": localization,
            "runner_receipt_body": receipt_body,
            "runner_receipt_sha256": runner_receipt_sha256,
        }
        return _copy(observation)  # type: ignore[return-value]
    finally:
        _close_backend_handles(backend, current_raw, reference_raw)
