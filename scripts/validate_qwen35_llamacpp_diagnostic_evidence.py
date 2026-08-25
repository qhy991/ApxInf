#!/usr/bin/env python3
"""Fail-closed validator for the Qwen3.5 ApxInf/llama.cpp diagnostic evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any


SUMMARY_FORMAT = "apxinf-qwen35-vs-llamacpp-raw13-free128-diagnostic-summary-v2"
VALIDATION_FORMAT = "apxinf-qwen35-vs-llamacpp-diagnostic-validation-v2"
CONTRACT_CONTENT_SHA256 = (
    "23f46184dce0882ab15c6e7e0b87832d143194b80bf3929d5b5c13f5f2173d89"
)
PINNED_SUMMARY_CONTENT_SHA256 = (
    "a70a8a3b46dd9efd37d0cd5aac906a5ee10f4a61eaf1960e92ec9ebc690bf884"
)
PINNED_SUMMARY_FILE_SHA256 = (
    "edb86c4e245bb0e5db561e19c7648253bfa485ac245950510c89d68602e770a6"
)
RUNNER_CONTRACT_COMMIT = "27ab4e670b5a523af3f56540eb9c3369fd0e778a"
LLAMA_CPP_COMMIT = "f280b26983ad0fdb705a0d9ebf0503e76f2899b0"
LLAMA_CPP_TREE = "21045aed8b426d7a5e25a98e646054cbd9487e81"
CANONICAL_FREE128_SHA256 = (
    "2042d5522ed5e768b938ac3fd9e19d3936dfc6e685fc157d41a8e8132c5b42fe"
)
PROMPT_TOKEN_IDS = [
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
]


class EvidenceError(ValueError):
    """Raised when checked-in evidence violates the frozen contract."""


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise EvidenceError(f"duplicate key: {key}")
        result[key] = value
    return result


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda value: (_ for _ in ()).throw(
                EvidenceError(f"non-finite JSON number: {value}")
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise EvidenceError(f"cannot read JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise EvidenceError(f"top-level JSON value must be an object: {path}")
    return value


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compact_json_sha256(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode(
        "utf-8"
    )
    return sha256_bytes(encoded)


def object_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256_bytes(encoded)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise EvidenceError(message)


def require_close(actual: float, expected: float, label: str) -> None:
    require(
        math.isfinite(actual)
        and math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-9),
        f"{label}: expected {expected!r}, got {actual!r}",
    )


def resolve_repo_path(root: Path, relative: str) -> Path:
    require(isinstance(relative, str) and relative, "evidence path must be nonempty")
    path = (root / relative).resolve(strict=True)
    try:
        path.relative_to(root)
    except ValueError as error:
        raise EvidenceError(f"evidence path escapes repository: {relative}") from error
    require(path.is_file(), f"evidence path is not a file: {relative}")
    return path


def validate_integrity_entry(root: Path, entry: dict[str, Any], label: str) -> Path:
    require(set(entry) >= {"path", "sha256", "size"}, f"{label}: incomplete")
    path = resolve_repo_path(root, entry["path"])
    require(path.stat().st_size == entry["size"], f"{label}: size mismatch")
    require(file_sha256(path) == entry["sha256"], f"{label}: SHA-256 mismatch")
    return path


def derive_llama_metrics(receipt: dict[str, Any]) -> dict[str, float]:
    timestamps = receipt["output"]["token_ready_elapsed_ns"]
    require(len(timestamps) == 128, "llama.cpp receipt must contain 128 timestamps")
    require(
        all(isinstance(value, int) and value > 0 for value in timestamps),
        "llama.cpp timestamps must be positive integers",
    )
    require(
        all(left < right for left, right in zip(timestamps, timestamps[1:])),
        "llama.cpp timestamps must be strictly increasing",
    )
    ttft_ms = timestamps[0] / 1_000_000
    total_ms = timestamps[-1] / 1_000_000
    tpot_ms = (total_ms - ttft_ms) / 127
    return {
        "ttft_ms": ttft_ms,
        "total_latency_ms": total_ms,
        "tpot_ms": tpot_ms,
        "generation_tps": 1000 / tpot_ms,
    }


def validate_llama_receipt(
    receipt: dict[str, Any], *, lane: str, model_size: int
) -> tuple[list[int], dict[str, float]]:
    require(receipt["schema"] == "apxinf.llama-cpp.raw-token-diagnostic.v2", "schema")
    require(receipt["ok"] is True, "llama.cpp receipt did not pass")
    contract = receipt["contract"]
    require(contract["prompt_token_ids"] == PROMPT_TOKEN_IDS, "llama prompt drift")
    require(contract["sampling"] == "greedy-argmax", "sampling drift")
    require(contract["generated_token_count"] == 128, "generation length drift")
    require(contract["eog_termination"] is False, "EOS policy drift")
    require(
        contract["final_sampled_token_is_not_decoded_in_timed_workload"] is True,
        "timed workload boundary drift",
    )
    require(receipt["model"]["file_size_bytes"] == model_size, "model size drift")
    require(receipt["model"]["file_identity_unchanged"] is True, "model custody failed")
    params = receipt["parameters"]
    require(params["n_ctx_requested"] == 142, "requested context drift")
    require(params["n_ctx_effective"] == 256, "effective context drift")
    require(params["n_threads"] == 4, "llama.cpp thread count drift")
    require(receipt["build"]["llama_cpp_source_id"] == LLAMA_CPP_COMMIT, "source drift")
    output = receipt["output"]["token_ids"]
    require(len(output) == 128, "llama.cpp receipt must contain 128 tokens")
    require(
        all(isinstance(token, int) and 0 <= token < 248320 for token in output),
        "generated token out of vocabulary bounds",
    )
    perf = receipt["llama_perf"]
    require(perf["context"]["n_prompt_eval"] == 13, "prompt perf count drift")
    require(perf["context"]["n_eval"] == 127, "decode perf count drift")
    require(perf["context"]["n_reused"] == 126, "reuse perf count drift")
    require(perf["sampler"]["n_sample"] == 128, "sample perf count drift")
    proof = receipt["post_measurement_execution_proof"]
    require(
        proof["passed"] is True and proof["timing_excluded"] is True, "proof failed"
    )
    require(proof["decode_count"] == 1, "proof decode count drift")
    require(proof["proof_token_id"] == output[-1], "proof token drift")
    require(proof["requested_sentinel_count"] == 26, "proof request drift")
    require(proof["completed_sentinel_count"] == 26, "proof completion drift")
    require(proof["backend_mismatch"] is False, "proof backend mismatch")
    require(proof["duplicate_or_unexpected_callback"] is False, "proof callback drift")
    placement = receipt["placement_attestation"]
    require(placement["passed"] is True, "placement attestation failed")
    if lane == "cpu":
        require(params["lane"] == "cpu-only", "CPU lane drift")
        require(
            params["kv_cache_type_k"] == params["kv_cache_type_v"] == "f32",
            "CPU KV drift",
        )
        require(placement["layers_on_cpu"] == 24, "CPU layer placement drift")
        require(
            placement["memory_by_device_class"]["gpu"]["total_bytes"] == 0,
            "CPU used GPU memory",
        )
        require(proof["completed_on_cpu"] == 26, "CPU execution proof drift")
        require(proof["completed_on_selected_gpu"] == 0, "CPU GPU proof drift")
    elif lane == "metal":
        require(params["lane"] == "gpu-all-layers", "Metal lane drift")
        require(
            params["kv_cache_type_k"] == params["kv_cache_type_v"] == "f16",
            "Metal KV drift",
        )
        require(
            receipt["backend"]["selected_gpu_device"]["name"] == "MTL0", "GPU drift"
        )
        require(
            placement["layers_on_selected_gpu"] == 24, "Metal layer placement drift"
        )
        require(
            placement["input_embedding_buffer_type"] == "CPU", "input placement drift"
        )
        require(proof["completed_on_cpu"] == 1, "Metal CPU fallback proof drift")
        require(proof["completed_on_selected_gpu"] == 25, "Metal proof drift")
    else:
        raise EvidenceError(f"unknown lane: {lane}")
    return output, derive_llama_metrics(receipt)


def validate_ratio_block(
    block: dict[str, Any], apx: dict[str, Any], llama: dict[str, Any], label: str
) -> None:
    expected = {
        "llama_cpp_generation_tps_over_apxinf": llama["generation_tps"]
        / apx["generation_tps"],
        "llama_cpp_generation_tps_delta_percent": (
            llama["generation_tps"] / apx["generation_tps"] - 1
        )
        * 100,
        "llama_cpp_tpot_over_apxinf": llama["tpot_ms"] / apx["tpot_ms"],
        "llama_cpp_tpot_delta_percent": (llama["tpot_ms"] / apx["tpot_ms"] - 1) * 100,
        "llama_cpp_total_latency_over_apxinf": llama["total_latency_ms"]
        / apx["total_latency_ms"],
        "llama_cpp_total_latency_delta_percent": (
            llama["total_latency_ms"] / apx["total_latency_ms"] - 1
        )
        * 100,
        "llama_cpp_ttft_over_apxinf": llama["ttft_ms"] / apx["ttft_ms"],
        "llama_cpp_ttft_delta_percent": (llama["ttft_ms"] / apx["ttft_ms"] - 1) * 100,
    }
    require(set(block) == set(expected), f"{label}: ratio field set drift")
    for key, value in expected.items():
        require_close(block[key], value, f"{label}.{key}")


def validate_summary(summary: dict[str, Any], root: Path) -> dict[str, Any]:
    require(
        set(summary)
        == {
            "format",
            "schema_version",
            "created_at_local_date",
            "publication",
            "classification",
            "scope",
            "contract_integrity",
            "llama_cpp_build_integrity",
            "model_artifacts",
            "receipt_integrity",
            "quality",
            "comparison_lanes",
            "placement_and_memory",
            "unbound_resource_observations",
            "formal_follow_up",
            "content_sha256",
            "_summary_relative_path",
        },
        "summary top-level field set drift",
    )
    require(summary["format"] == SUMMARY_FORMAT, "summary format drift")
    require(summary["schema_version"] == 2, "summary schema version drift")
    unsigned = {
        key: value
        for key, value in summary.items()
        if key not in {"content_sha256", "_summary_relative_path"}
    }
    require(
        object_sha256(unsigned) == summary["content_sha256"],
        "summary canonical content SHA-256 mismatch",
    )
    require(
        summary["content_sha256"] == PINNED_SUMMARY_CONTENT_SHA256,
        "summary content differs from the audited pinned value",
    )
    summary_path = resolve_repo_path(root, str(summary["_summary_relative_path"]))
    require(
        file_sha256(summary_path) == PINNED_SUMMARY_FILE_SHA256,
        "summary file differs from the audited pinned value",
    )
    classification = summary["classification"]
    require(
        classification["formal_performance_result"] is False, "formal claim forbidden"
    )
    require(
        classification["promotion_status"] == "candidate-only-not-promoted",
        "promotion drift",
    )
    require(
        classification["promotion_claim_allowed"] is False, "promotion claim forbidden"
    )
    require(
        set(classification["reasons_not_formal"])
        == {
            "the quiet-host gate failed",
            "system swap was nonzero",
            "ApxInf and llama.cpp thread policies were not aligned",
            "only one timed observation per implementation and tier was collected",
            "the required ABBA/BAAB campaign and untimed warmups were not run",
            "loaded system-library image hashes were not captured",
            "llama.cpp teacher-forced cross-runtime quality was not measured",
            "runner executable SHA-256 was verified outside the run receipts rather than embedded by the runner",
            "GGUF model SHA-256 values were verified outside the runner receipts rather than computed from the pinned file descriptor",
        },
        "non-formal reason set drift",
    )
    require(
        summary["publication"]["runner_contract_commit"] == RUNNER_CONTRACT_COMMIT,
        "publication commit drift",
    )
    require(
        summary["publication"]["apxinf_measurement_code_commit"]
        == "820ee4ed98f66feaec0324e1a8870a7eb0967531",
        "ApxInf measurement commit drift",
    )
    require(
        summary["publication"]["apxinf_evidence_commit"]
        == "cb976735bbd373ed09f8e593af75c13236096f24",
        "ApxInf evidence commit drift",
    )
    scope = summary["scope"]
    require(scope["prompt_token_ids"] == PROMPT_TOKEN_IDS, "summary prompt drift")
    require(
        scope["generated_tokens"] == 128 and scope["eos_stopping"] is False,
        "workload drift",
    )
    require(scope["requested_context_length"] == 142, "context request drift")
    require(
        scope["llama_cpp_effective_context_length"] == 256, "effective context drift"
    )
    require(
        scope["effective_context_allocation_equal"] is False, "false context equality"
    )
    parity = scope["runtime_parity"]
    require(parity["same_physical_host"] is True, "host parity drift")
    require(
        parity["same_prompt_and_generation_contract"] is True, "workload parity drift"
    )
    require(parity["llama_cpp_threads"] == 4, "thread count drift")
    require(
        parity["apxinf_thread_policy"] == "not-explicitly-controllable",
        "ApxInf thread policy drift",
    )
    require(parity["thread_policy_parity"] is False, "false thread parity")
    require(
        compact_json_sha256(PROMPT_TOKEN_IDS)
        == scope["prompt_token_ids_compact_json_sha256"],
        "prompt hash drift",
    )
    require(
        scope["timing_definitions"]
        == {
            "origin": "immediately-before-raw-token-prefill",
            "ttft_ms": "first-token-ready-elapsed",
            "total_latency_ms": "token-128-ready-elapsed",
            "tpot_ms": "(total_latency_ms-ttft_ms)/127",
            "generation_tps": "127000/(total_latency_ms-ttft_ms)",
            "model_load_included": False,
            "llama_cpp_execution_proof_decode_included": False,
        },
        "timing definition drift",
    )

    contract_entry = summary["contract_integrity"]
    contract_path = validate_integrity_entry(
        root, contract_entry, "comparison contract"
    )
    contract = load_json(contract_path)
    require(
        contract["content_sha256"] == CONTRACT_CONTENT_SHA256, "contract content drift"
    )
    unsigned_contract = dict(contract)
    unsigned_contract.pop("content_sha256")
    require(
        object_sha256(unsigned_contract) == CONTRACT_CONTENT_SHA256,
        "contract canonical hash mismatch",
    )
    require(
        contract_entry["content_sha256"] == CONTRACT_CONTENT_SHA256,
        "contract binding drift",
    )
    build = summary["llama_cpp_build_integrity"]
    require(build["source_commit"] == LLAMA_CPP_COMMIT, "llama.cpp commit drift")
    require(build["source_tree"] == LLAMA_CPP_TREE, "llama.cpp source tree drift")
    require(
        build["runner_schema"] == "apxinf.llama-cpp.raw-token-diagnostic.v2",
        "runner schema drift",
    )
    require(
        build["clean_detached_checkout"] is True, "llama.cpp checkout custody drift"
    )
    require(
        build["runner_binary"]["sha256"]
        == contract["llama_cpp"]["formal_observed_build"]["binary"]["sha256"],
        "runner binary drift",
    )
    require(
        build["runner_binary"]["size"]
        == contract["llama_cpp"]["formal_observed_build"]["binary"]["size"],
        "runner binary size drift",
    )
    require(
        build["runner_binary"]["linkage"]
        == "static-llama-ggml-with-system-dynamic-libraries-only",
        "runner linkage drift",
    )
    require(
        build["runner_binary"]["receipt_binding"]
        == "outer-summary-only-not-self-reported-by-runner",
        "runner receipt binding drift",
    )
    require(
        build["runner_binary"]["clean_rebuild_byte_identical"] is True,
        "clean rebuild drift",
    )
    require(
        build["dynamic_backend_scan_invoked"] is False,
        "dynamic backend scan claim drift",
    )
    require(
        build["loaded_system_library_hashes_captured"] is False,
        "system closure claim drift",
    )
    for key in ("runner_source", "cmake_lists"):
        validate_integrity_entry(root, build[key], f"build {key}")
    observed_build = contract["llama_cpp"]["formal_observed_build"]
    require(
        build["runner_source"]["sha256"]
        == observed_build["inputs"]["runner_source"]["sha256"],
        "runner source contract drift",
    )
    require(
        build["cmake_lists"]["sha256"]
        == observed_build["inputs"]["cmake_lists"]["sha256"],
        "CMake contract drift",
    )

    receipt_paths = {
        key: validate_integrity_entry(root, value, key)
        for key, value in summary["receipt_integrity"].items()
    }
    require(
        set(receipt_paths)
        == {
            "apxinf_gate_summary",
            "apxinf_cpu_teacher",
            "apxinf_candidate_teacher",
            "apxinf_cpu_free",
            "apxinf_candidate_free",
            "llama_cpp_f32_cpu",
            "llama_cpp_q8_0_metal",
        },
        "receipt manifest field set drift",
    )
    apx_summary = load_json(receipt_paths["apxinf_gate_summary"])
    apx_cpu_teacher = load_json(receipt_paths["apxinf_cpu_teacher"])
    apx_candidate_teacher = load_json(receipt_paths["apxinf_candidate_teacher"])
    apx_cpu_free = load_json(receipt_paths["apxinf_cpu_free"])
    apx_candidate_free = load_json(receipt_paths["apxinf_candidate_free"])
    require(apx_cpu_teacher["passed"] is True, "ApxInf CPU teacher receipt failed")
    require(
        apx_candidate_teacher["passed"] is True,
        "ApxInf candidate teacher receipt failed",
    )
    require(apx_cpu_teacher["comparisons"] == 128, "ApxInf CPU teacher count drift")
    require(
        apx_candidate_teacher["comparisons"] == 128,
        "ApxInf candidate teacher count drift",
    )
    require(
        len(apx_cpu_teacher["teacher_input_ids"]) == 128, "teacher input length drift"
    )
    require(
        apx_cpu_teacher["teacher_input_ids"]
        == apx_candidate_teacher["teacher_input_ids"],
        "teacher input trajectory drift",
    )
    exactness = apx_candidate_teacher["exactness"]
    for key in (
        "body_token_mismatches",
        "top4_mismatches",
        "direct_rerank_mismatches",
        "end_to_end_mismatches",
    ):
        require(exactness[key] == [], f"ApxInf teacher mismatch: {key}")
    require(
        apx_candidate_teacher["frozen_cpu_expected_output_ids"]
        == apx_candidate_teacher["tail_normalized_hidden_f32_winner_ids"]
        == apx_candidate_teacher["direct_tied_f32_reranked_output_ids"],
        "ApxInf teacher output arrays drift",
    )
    require(
        apx_candidate_teacher["path_checks"]["prefill"]["all_valid"] is True,
        "teacher prefill path failed",
    )
    require(
        apx_candidate_teacher["path_checks"]["final"]["all_valid"] is True,
        "teacher final path failed",
    )
    require(apx_cpu_free["passed"] is True, "ApxInf CPU free receipt failed")
    require(
        apx_candidate_free["passed"] is True, "ApxInf candidate free receipt failed"
    )
    require(apx_candidate_free["mismatches"] == [], "ApxInf free-run mismatch")
    require(
        apx_candidate_free["exact_128_token_trajectory"] is True,
        "ApxInf free exactness drift",
    )
    require(
        apx_candidate_free["path_checks"]["all_valid"] is True,
        "ApxInf free path failed",
    )
    require(
        apx_cpu_free["prompt_token_ids"] == PROMPT_TOKEN_IDS, "ApxInf CPU prompt drift"
    )
    require(
        apx_candidate_free["prompt_token_ids"] == PROMPT_TOKEN_IDS,
        "ApxInf candidate prompt drift",
    )
    apx_cpu_tokens = apx_cpu_free["generated_token_ids"]
    apx_candidate_tokens = apx_candidate_free["generated_token_ids"]
    require(
        len(apx_cpu_tokens) == len(apx_candidate_tokens) == 128,
        "ApxInf token count drift",
    )
    require(apx_cpu_tokens == apx_candidate_tokens, "ApxInf free trajectory mismatch")
    require(
        apx_summary["quality_gate"]["teacher"]["exact_128_of_128"] is True,
        "ApxInf teacher gate failed",
    )
    require(
        apx_summary["quality_gate"]["free_run"]["exact_128_token_trajectory"] is True,
        "ApxInf free gate failed",
    )

    llama_cpu = load_json(receipt_paths["llama_cpp_f32_cpu"])
    llama_metal = load_json(receipt_paths["llama_cpp_q8_0_metal"])
    llama_cpu_tokens, llama_cpu_metrics = validate_llama_receipt(
        llama_cpu, lane="cpu", model_size=3_020_533_248
    )
    llama_metal_tokens, llama_metal_metrics = validate_llama_receipt(
        llama_metal, lane="metal", model_size=811_843_072
    )
    require(
        apx_cpu_tokens == llama_cpu_tokens == llama_metal_tokens,
        "cross-runtime free trajectory mismatch",
    )
    require(
        compact_json_sha256(apx_cpu_tokens) == CANONICAL_FREE128_SHA256,
        "canonical trajectory hash drift",
    )
    quality = summary["quality"]
    require(
        quality["canonical_free128_token_ids_compact_json_sha256"]
        == CANONICAL_FREE128_SHA256,
        "summary trajectory binding drift",
    )
    require(
        quality["free_run"]["all_four_trajectories_identical"] is True,
        "quality summary drift",
    )
    require(quality["free_run"]["position_match_count"] == 128, "position match drift")
    require_close(
        quality["free_run"]["position_match_ratio"], 1.0, "position match ratio"
    )
    require(quality["free_run"]["exact_prefix_tokens"] == 128, "exact prefix drift")
    require(quality["free_run"]["first_mismatch"] is None, "first mismatch drift")
    require(
        quality["free_run"]["general_quality_parity_claim_allowed"] is False,
        "general quality claim forbidden",
    )
    require(
        quality["teacher_forced"]["llama_cpp_cross_runtime_measured"] is False,
        "teacher measurement drift",
    )
    require(
        quality["teacher_forced"]["llama_cpp_cross_runtime"] == "not-measured",
        "unearned teacher claim",
    )
    require(
        quality["teacher_forced"]["cross_runtime_teacher_exactness_claim_allowed"]
        is False,
        "unearned teacher exactness",
    )

    artifacts = summary["model_artifacts"]
    contract_artifacts = contract["llama_cpp"]["model_artifacts"]
    require(
        artifacts["llama_cpp_f32"]["sha256"] == contract_artifacts["f32"]["sha256"]
        and artifacts["llama_cpp_f32"]["size"] == contract_artifacts["f32"]["size"],
        "F32 model artifact drift",
    )
    require(
        artifacts["llama_cpp_f32"]["hash_binding"]
        == "post-run outer-contract verification",
        "F32 hash binding drift",
    )
    require(
        artifacts["llama_cpp_pure_q8_0"]["sha256"]
        == contract_artifacts["pure_q8_0"]["sha256"]
        and artifacts["llama_cpp_pure_q8_0"]["size"]
        == contract_artifacts["pure_q8_0"]["size"],
        "Q8_0 model artifact drift",
    )
    require(
        artifacts["llama_cpp_pure_q8_0"]["hash_binding"]
        == "post-run outer-contract verification",
        "Q8_0 hash binding drift",
    )

    apx_timing = apx_summary["diagnostic_timing"]
    lanes = summary["comparison_lanes"]
    f32_apx = lanes["f32_reference"]["apxinf"]
    q8_apx = lanes["eight_bit_storage"]["apxinf"]
    expected_apx = {
        "f32": {
            "ttft_ms": apx_timing["cpu_free_ttft_ms"],
            "tpot_ms": apx_timing["cpu_free_tpot_ms"],
            "generation_tps": apx_timing["cpu_free_generation_tps"],
        },
        "q8": {
            "ttft_ms": apx_timing["candidate_free_ttft_ms"],
            "tpot_ms": apx_timing["candidate_free_tpot_ms"],
            "generation_tps": apx_timing["candidate_free_generation_tps"],
        },
    }
    for lane_label, actual, expected in (
        ("f32", f32_apx, expected_apx["f32"]),
        ("q8", q8_apx, expected_apx["q8"]),
    ):
        for key, value in expected.items():
            require_close(actual[key], value, f"{lane_label}.apxinf.{key}")
        require_close(
            actual["total_latency_ms"],
            actual["ttft_ms"] + 127 * actual["tpot_ms"],
            f"{lane_label}.apxinf.total",
        )
    for key, value in llama_cpu_metrics.items():
        require_close(
            lanes["f32_reference"]["llama_cpp"][key], value, f"f32.llama_cpp.{key}"
        )
    for key, value in llama_metal_metrics.items():
        require_close(
            lanes["eight_bit_storage"]["llama_cpp"][key], value, f"q8.llama_cpp.{key}"
        )
    require(
        lanes["eight_bit_storage"]["quantization_mechanisms_equal"] is False,
        "false quantization equivalence",
    )
    require(
        lanes["eight_bit_storage"]["weight_regimes_equal"] is False,
        "false weight equivalence",
    )
    validate_ratio_block(
        lanes["f32_reference"]["diagnostic_ratios"],
        f32_apx,
        lanes["f32_reference"]["llama_cpp"],
        "f32",
    )
    validate_ratio_block(
        lanes["eight_bit_storage"]["diagnostic_ratios"],
        q8_apx,
        lanes["eight_bit_storage"]["llama_cpp"],
        "q8",
    )
    require(
        summary["placement_and_memory"]["cross_runtime_memory_ratio_allowed"] is False,
        "memory ratio claim forbidden",
    )
    memory = summary["placement_and_memory"]
    require(
        memory["llama_cpp_f32_cpu"]["gpu_total_bytes"]
        == llama_cpu["placement_attestation"]["memory_by_device_class"]["gpu"][
            "total_bytes"
        ],
        "F32 memory ledger drift",
    )
    q8_memory = llama_metal["placement_attestation"]["memory_by_device_class"]
    require(
        memory["llama_cpp_q8_0_metal"]["gpu_total_bytes"]
        == q8_memory["gpu"]["total_bytes"],
        "Q8 GPU ledger drift",
    )
    require(
        memory["llama_cpp_q8_0_metal"]["cpu_total_bytes"]
        == q8_memory["cpu"]["total_bytes"],
        "Q8 CPU ledger drift",
    )
    require(
        memory["apxinf_metal_w8"]["resident_mtlbuffer_bytes"]
        == apx_summary["aggregate_buffer_ledger"]["total_persistent_mtlbuffer_bytes"],
        "ApxInf Metal ledger drift",
    )
    require(
        summary["unbound_resource_observations"][
            "process_rss_cross_runtime_comparison_allowed"
        ]
        is False,
        "RSS ratio claim forbidden",
    )
    resources = summary["unbound_resource_observations"]
    require(
        resources["classification"]
        == "diagnostic-only-not-atomically-bound-to-json-receipts",
        "resource classification drift",
    )
    require(
        resources["apxinf"]["system_swap_invariant"] is False, "ApxInf swap claim drift"
    )
    require(
        resources["post_llama_run_host_snapshot"]["quiet_host"] is False,
        "quiet-host claim drift",
    )
    require(
        resources["post_llama_run_host_snapshot"][
            "non_allowlisted_process_above_five_percent_cpu"
        ]
        is True,
        "host-load claim drift",
    )
    require(summary["formal_follow_up"]["required"] is True, "formal follow-up drift")
    require(
        set(summary["formal_follow_up"]["blockers"])
        == {
            "quiet-host-gate",
            "zero-and-invariant-system-swap",
            "explicit-thread-policy-parity",
            "three-untimed-warmups-per-implementation",
            "six-block-abba-baab-campaign",
            "twelve-timed-samples-per-implementation-and-tier",
            "loaded-system-library-image-hashes",
            "runner-and-model-sha256-atomically-bound-in-run-receipts",
            "llama-cpp-teacher-forced-quality-receipt",
        },
        "formal blocker set drift",
    )
    return {
        "format": VALIDATION_FORMAT,
        "valid": True,
        "formal_performance_result": False,
        "runner_contract_commit": RUNNER_CONTRACT_COMMIT,
        "summary_sha256": file_sha256(summary_path),
        "canonical_free128_sha256": CANONICAL_FREE128_SHA256,
        "f32_llama_cpp_tps_over_apxinf": lanes["f32_reference"]["diagnostic_ratios"][
            "llama_cpp_generation_tps_over_apxinf"
        ],
        "q8_llama_cpp_tps_over_apxinf": lanes["eight_bit_storage"]["diagnostic_ratios"][
            "llama_cpp_generation_tps_over_apxinf"
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path)
    args = parser.parse_args(argv)
    summary_path = args.summary.resolve(strict=True)
    root = (args.repo_root or Path(__file__).resolve().parents[1]).resolve(strict=True)
    try:
        relative = summary_path.relative_to(root).as_posix()
        summary = load_json(summary_path)
        summary["_summary_relative_path"] = relative
        result = validate_summary(summary, root)
    except (
        EvidenceError,
        KeyError,
        TypeError,
        IndexError,
        OSError,
        ValueError,
    ) as error:
        print(f"invalid diagnostic evidence: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
