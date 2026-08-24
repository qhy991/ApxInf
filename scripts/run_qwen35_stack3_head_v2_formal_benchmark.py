#!/usr/bin/env python3
"""Fail-closed formal benchmark harness for Qwen3.5 Stack3 + lm_head v2.

The default command validates frozen evidence and emits a plan. A real model
can start only when the caller explicitly supplies ``--execute``.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
import stat
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_HARNESS_PATH = REPO_ROOT / "scripts" / "run_qwen35_all18_formal_benchmark.py"


def _load_base_harness():
    module_name = "_apxinf_qwen35_all18_formal_benchmark_base_for_stack3_head_v2"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(module_name, BASE_HARNESS_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import audited base harness: {BASE_HARNESS_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


BASE = _load_base_harness()
HarnessError = BASE.HarnessError
BLOCK_ORDERS = BASE.BLOCK_ORDERS
CPU_MODE = "cpu-free"
CANDIDATE_MODE = "stack3-head-v2-free"
PINNED_SUMMARY_SHA256 = (
    "7aa07e1dd7beda066fa7c7048bfcbb0505b793c1caf43aac5f104c4a45177727"
)
PINNED_BINARY_SHA256 = (
    "0e70fe6589a77c78c79aa5071741eae27ae184b863b7a49adf47228f86ea1812"
)
PINNED_BASE_HARNESS_SHA256 = (
    "971f8ee9778f64c3e68470ff05e5bc71481239054560ec971be0e74811337bfe"
)
SUMMARY_FORMAT = "apxinf-qwen35-metal-w8-stack3-head-v2-real-gate-summary-v1"
RECEIPT_KEYS = (
    "cpu_teacher128",
    "candidate_teacher128",
    "cpu_free128",
    "candidate_free128",
)
RECEIPT_IDENTITY = {
    "cpu_teacher128": (
        "apxinf-qwen35-metal-w8-stack3-head-v2-cpu-teacher-v1",
        "cpu_teacher",
    ),
    "candidate_teacher128": (
        "apxinf-qwen35-metal-w8-stack3-head-v2-teacher-gate-v1",
        "metal_w8_stack3_head_v2_teacher_forced",
    ),
    "cpu_free128": (
        "apxinf-qwen35-metal-w8-stack3-head-v2-cpu-free-run-v1",
        "cpu_free_run",
    ),
    "candidate_free128": (
        "apxinf-qwen35-metal-w8-stack3-head-v2-free-run-gate-v1",
        "metal_w8_stack3_head_v2_free_run",
    ),
}
RUST_SOURCE_KEYS = {
    "apxinf_metal_build",
    "apxinf_metal_lib",
    "gate_evidence",
    "gdn_rust",
    "general",
    "linear_layer_rust",
    "llm_trait",
    "metal_w8_head_bridge",
    "metal_w8_mlp_bridge",
    "stack3_bridge",
    "stack3_rust",
}
SHADER_SOURCE_KEYS = {
    "metal_w8_gdn",
    "metal_w8_gdn_out_g32",
    "metal_w8_head",
    "metal_w8_linear_layer",
    "metal_w8_matvec",
    "metal_w8_mlp",
}
DEFAULT_SUMMARY = REPO_ROOT / (
    "crates/apxinf-metal/evidence/next-hotspot/"
    "qwen35-stack3-head-v2-real-gate-summary-v1-20260824.json"
)
STACK_LAYER_INDICES = (
    (0, 1, 2),
    (4, 5, 6),
    (8, 9, 10),
    (12, 13, 14),
    (16, 17, 18),
    (20, 21, 22),
)
FULL_ATTENTION_LAYER_INDICES = (3, 7, 11, 15, 19, 23)
STACK_LEDGER = {
    "activation_bytes": 133_248,
    "active_state_bytes": 3_440_640,
    "allocated_buffers": 76,
    "command_buffers_per_decode": 1,
    "commits_per_decode": 1,
    "compute_encoders_per_decode": 3,
    "f32_parameter_bytes": 321_408,
    "final_output_finite_checks_per_decode": 1,
    "host_input_bytes_per_decode": 4_096,
    "host_output_bytes_per_decode": 4_096,
    "intermediate_host_finite_checks_per_decode": 0,
    "packed_scale_bytes": 4_429_824,
    "packed_weight_bytes": 64_585_728,
    "private_buffers": 8,
    "scratch_state_bytes": 3_440_640,
    "shared_buffers": 68,
    "state_host_transfer_bytes_per_decode": 0,
    "total_persistent_bytes": 76_351_488,
    "waits_per_decode": 1,
}
MLP_LEDGER = {
    "activation_bytes": 51_200,
    "allocated_buffers": 8,
    "command_buffers_per_decode": 1,
    "commits_per_decode": 1,
    "compute_encoders_per_decode": 3,
    "host_input_bytes_per_decode": 4_096,
    "host_output_bytes_per_decode": 4_096,
    "packed_scale_bytes": 688_128,
    "packed_weight_bytes": 11_010_048,
    "private_buffers": 2,
    "scope": "resident-mtlbuffer-only",
    "shared_buffers": 6,
    "state_host_transfer_bytes_per_decode": 0,
    "total_persistent_bytes": 11_749_376,
    "waits_per_decode": 1,
}
HEAD_LEDGER = {
    "scope": "resident-mtlbuffer-only",
    "exclusions": "host F32 tied embedding and four-candidate rerank, host allocations, Metal pipelines/libraries/queues, command objects, driver allocations, model body, and KV cache",
    "allocated_buffers": 5,
    "shared_buffers": 4,
    "private_buffers": 1,
    "packed_weight_bytes": 254_279_680,
    "packed_scale_bytes": 15_892_480,
    "hidden_bytes": 4_096,
    "partial_topk_bytes": 993_280,
    "output_token_bytes": 16,
    "total_persistent_bytes": 271_169_552,
    "host_input_bytes_per_call": 4_096,
    "host_output_bytes_per_call": 16,
    "state_host_transfer_bytes_per_call": 0,
    "command_buffers_per_call": 1,
    "compute_encoders_per_call": 2,
    "commits_per_call": 1,
    "waits_per_call": 1,
}
BODY_LEDGER_TOTALS = {
    "scope": "resident-mtlbuffer-only",
    "exclusions": "CPU F32 weights, host Vec allocations, Metal pipelines/libraries/queues, driver allocations, KV cache, and lm_head",
    "includes_lm_head": False,
    "total_persistent_mtlbuffer_bytes": 528_605_184,
    "allocated_buffers": 504,
    "shared_buffers": 444,
    "private_buffers": 60,
    "host_to_device_bytes_per_decode": 49_152,
    "device_to_host_bytes_per_decode": 49_152,
    "state_host_transfer_bytes_per_decode": 0,
    "command_buffers_per_decode": 12,
    "compute_encoders_per_decode": 36,
    "commits_per_decode": 12,
    "waits_per_decode": 12,
    "intermediate_host_finite_checks_per_decode": 0,
    "final_output_finite_checks_per_decode": 6,
}
COMPOSITE_LEDGER_TOTALS = {
    "scope": "resident-mtlbuffer-only",
    "exclusions": "host F32 tied embedding and exact four-candidate F32 rerank, other CPU F32 weights, host allocations, Metal pipelines/libraries/queues, command objects, driver allocations, and KV cache",
    "includes_lm_head": True,
    "total_persistent_mtlbuffer_bytes": 799_774_736,
    "allocated_buffers": 509,
    "shared_buffers": 448,
    "private_buffers": 61,
    "host_to_device_bytes_per_call": 53_248,
    "device_to_host_bytes_per_call": 49_168,
    "state_host_transfer_bytes_per_call": 0,
    "command_buffers_per_call": 13,
    "compute_encoders_per_call": 38,
    "commits_per_call": 13,
    "waits_per_call": 13,
    "intermediate_host_finite_checks_per_call": 0,
    "final_output_finite_checks_per_call": 6,
}
CPU_RECEIPT_KEYS = {
    "format",
    "mode",
    "identity",
    "custody_end_verification",
    "prompt",
    "prompt_token_ids",
    "official_layer_schedule_valid",
    "max_new_tokens",
    "eos_stopping",
    "generated_token_ids",
    "generation_path_contract",
    "profile",
    "passed",
}
CANDIDATE_RECEIPT_KEYS = {
    "format",
    "mode",
    "identity",
    "input_receipt",
    "custody_end_verification",
    "prompt",
    "prompt_token_ids",
    "official_layer_schedule_valid",
    "max_new_tokens",
    "eos_stopping",
    "cpu_expected_token_ids",
    "generated_token_ids",
    "mismatches",
    "exact_trajectory",
    "final_generation_path_receipt",
    "generation_path_contract",
    "aggregate_buffer_ledger",
    "path_checks",
    "profile",
    "passed",
}
CANDIDATE_GENERATION_CONTRACT = {
    "schema": "apxinf-qwen35-stack3-lm-head-generation-path-v2",
    "shared_generate_streaming": True,
    "binds_six_stack3_six_full_attention_mlp_and_tied_head": True,
    "body_decode_calls": 127,
    "head_prefill_calls": 1,
    "head_decode_calls": 127,
}
CANDIDATE_PATH_CHECKS = {
    "aggregate_ledger_valid": True,
    "all_valid": True,
    "full_attention_mlp_valid": True,
    "generation_receipt_valid": True,
    "head_execution_valid": True,
    "schedule_valid": True,
    "stack3_execution_valid": True,
    "stack3_mechanism_valid": True,
    "terminal_clear": True,
}


def _json_exact_equal(observed: object, expected: object) -> bool:
    if type(observed) is not type(expected):
        return False
    if isinstance(expected, dict):
        return set(observed) == set(expected) and all(
            _json_exact_equal(observed[key], expected_value)
            for key, expected_value in expected.items()
        )
    if isinstance(expected, list):
        return len(observed) == len(expected) and all(
            _json_exact_equal(left, right) for left, right in zip(observed, expected)
        )
    return observed == expected


def _require_exact_object(value: object, expected: dict, *, label: str) -> dict:
    if not isinstance(value, dict) or not _json_exact_equal(value, expected):
        raise HarnessError(f"{label} drifted")
    return value


def validate_composite_ledger(ledger: object) -> dict:
    """Validate every field of the 509-buffer body + tied-head ledger."""

    if not isinstance(ledger, dict):
        raise HarnessError("Stack3 + lm_head v2 composite ledger is missing")
    expected_keys = set(COMPOSITE_LEDGER_TOTALS) | {"body", "lm_head"}
    if set(ledger) != expected_keys:
        raise HarnessError("Stack3 + lm_head v2 composite ledger drifted")
    for key, expected in COMPOSITE_LEDGER_TOTALS.items():
        if not _json_exact_equal(ledger.get(key), expected):
            raise HarnessError("Stack3 + lm_head v2 composite ledger drifted")
    _require_exact_object(ledger.get("lm_head"), HEAD_LEDGER, label="lm_head ledger")
    body = ledger.get("body")
    if not isinstance(body, dict):
        raise HarnessError("Stack3 body ledger is missing")
    body_keys = set(BODY_LEDGER_TOTALS) | {
        "stacks",
        "full_attention_mlp_layers",
    }
    if set(body) != body_keys:
        raise HarnessError("Stack3 body ledger drifted")
    for key, expected in BODY_LEDGER_TOTALS.items():
        if not _json_exact_equal(body.get(key), expected):
            raise HarnessError("Stack3 body ledger drifted")
    stacks = body.get("stacks")
    full = body.get("full_attention_mlp_layers")
    if not isinstance(stacks, list) or len(stacks) != 6:
        raise HarnessError("Stack3 body ledger must contain six stacks")
    if not isinstance(full, list) or len(full) != 6:
        raise HarnessError("Stack3 body ledger must contain six full-attention MLPs")
    for entry, layer_indices in zip(stacks, STACK_LAYER_INDICES):
        _require_exact_object(
            entry,
            {"layer_indices": list(layer_indices), "ledger": STACK_LEDGER},
            label="Stack3 stack ledger",
        )
    for entry, layer_index in zip(full, FULL_ATTENTION_LAYER_INDICES):
        _require_exact_object(
            entry,
            {"layer_index": layer_index, "ledger": MLP_LEDGER},
            label="full-attention MLP ledger",
        )
    return ledger


def _nonnegative_int(value: object) -> bool:
    return type(value) is int and value >= 0


def validate_generation_receipt(
    receipt: object, *, body_calls: int, head_calls: dict
) -> dict:
    """Bind all body transactions and the tied-head phase counters."""

    if type(body_calls) is not int or body_calls < 0:
        raise HarnessError("expected body call count is invalid")
    expected_head_keys = {"prefill_calls", "decode_calls", "teacher_calls"}
    if (
        not isinstance(head_calls, dict)
        or set(head_calls) != expected_head_keys
        or any(not _nonnegative_int(value) for value in head_calls.values())
    ):
        raise HarnessError("expected lm_head call counts are invalid")
    if not isinstance(receipt, dict):
        raise HarnessError("v2 generation receipt is missing")
    top_expected = {
        "format": "apxinf-qwen35-stack3-lm-head-generation-path-v2",
        "mechanism": "metal-w8-stack3-lm-head-v2",
        "stack3_mechanism": "metal-w8-linear-layer-stack3-v1",
        "full_attention_mlp_mechanism": "metal-w8-mlp-block-g64",
        "metal_w8_complete_linear_layer_stacks": True,
        "metal_w8_full_attention_mlp_blocks": True,
        "metal_w8_tied_lm_head_topk4_f32_rerank": True,
        "intermediate_host_finite_checks": False,
        "final_output_finite_checks": True,
        "terminal_error": False,
    }
    if set(receipt) != set(top_expected) | {
        "stacks",
        "full_attention_mlp_layers",
        "lm_head",
    } or any(
        not _json_exact_equal(receipt.get(key), expected)
        for key, expected in top_expected.items()
    ):
        raise HarnessError("v2 generation mechanism contract drifted")
    stacks = receipt.get("stacks")
    full = receipt.get("full_attention_mlp_layers")
    if not isinstance(stacks, list) or len(stacks) != 6:
        raise HarnessError("v2 generation receipt must contain six Stack3 lanes")
    if not isinstance(full, list) or len(full) != 6:
        raise HarnessError("v2 generation receipt must contain six full-attention MLPs")
    expected_stack = {
        "command_buffers": body_calls,
        "commits": body_calls,
        "committed_stack_version": body_calls,
        "compute_encoders": body_calls * 3,
        "decode_calls": body_calls,
        "device_to_host_bytes": body_calls * 4_096,
        "failed_decodes": 0,
        "final_output_finite_checks_per_decode": 1,
        "gdn_output_group_sizes": [32, 32, 32],
        "host_to_device_bytes": body_calls * 4_096,
        "intermediate_host_finite_checks_per_decode": 0,
        "last_state_commit_mask": 7 if body_calls else 0,
        "mechanism": "metal-w8-linear-layer-stack3-v1",
        "prefill_seed_calls": [1, 1, 1],
        "state_commits": body_calls * 3,
        "successful_decodes": body_calls,
        "terminal_error": False,
        "waits": body_calls,
    }
    for entry, layer_indices in zip(stacks, STACK_LAYER_INDICES):
        if (
            not isinstance(entry, dict)
            or set(entry) != set(expected_stack) | {"layer_indices", "block_elapsed_ns"}
            or not _json_exact_equal(entry.get("layer_indices"), list(layer_indices))
            or not _nonnegative_int(entry.get("block_elapsed_ns"))
            or any(
                not _json_exact_equal(entry.get(key), expected)
                for key, expected in expected_stack.items()
            )
        ):
            raise HarnessError("Stack3 execution counters drifted")
    for entry, layer_index in zip(full, FULL_ATTENTION_LAYER_INDICES):
        if (
            not isinstance(entry, dict)
            or set(entry) != {"layer_index", "decode_calls", "block_elapsed_ns"}
            or not _json_exact_equal(entry.get("layer_index"), layer_index)
            or not _json_exact_equal(entry.get("decode_calls"), body_calls)
            or not _nonnegative_int(entry.get("block_elapsed_ns"))
        ):
            raise HarnessError("full-attention MLP execution counters drifted")
    head = receipt.get("lm_head")
    if (
        not isinstance(head, dict)
        or set(head)
        != {
            "mechanism",
            "prefill_calls",
            "decode_calls",
            "teacher_calls",
            "topk_elapsed_ns",
            "rerank_elapsed_ns",
        }
        or head.get("mechanism") != "metal-w8-top4-f32-rerank"
        or any(
            not _json_exact_equal(head.get(key), expected)
            for key, expected in head_calls.items()
        )
        or not _nonnegative_int(head.get("topk_elapsed_ns"))
        or not _nonnegative_int(head.get("rerank_elapsed_ns"))
    ):
        raise HarnessError("lm_head execution counters drifted")
    return receipt


def build_schedule(output_dir: Path) -> dict:
    """Return the immutable 24-run Stack3 + lm_head v2 schedule."""

    blocks = []
    for block_index, order in enumerate(BLOCK_ORDERS):
        runs = []
        for run_index, variant in enumerate(order):
            mode = CPU_MODE if variant == "A" else CANDIDATE_MODE
            output = output_dir / (
                f"block-{block_index:02d}-run-{run_index:02d}-{variant}.json"
            )
            runs.append(
                {
                    "index": run_index,
                    "variant": variant,
                    "mode": mode,
                    "output": str(output),
                }
            )
        blocks.append({"index": block_index, "order": order, "runs": runs})
    return {"block_orders": list(BLOCK_ORDERS), "blocks": blocks}


def _expected_end_custody(identity: object) -> dict:
    if not isinstance(identity, dict):
        raise HarnessError("v2 gate identity is missing")
    try:
        custody = identity["custody"]
        sources = custody["sources"]
        return {
            "binary": custody["binary"],
            "gate": sources["gate"],
            "rust_and_bridge_sources": sources["rust_and_bridge_sources"],
            "source_closure": sources["closure"],
            "compiled_metal_shader_sources": sources["compiled_metal_shader_sources"],
            "verified_at_end": True,
        }
    except (KeyError, TypeError) as error:
        raise HarnessError("v2 gate identity custody is incomplete") from error


def _validate_free_profile(profile: object, *, candidate: bool) -> dict:
    expected_classification = (
        "candidate-only single pass under an uncontrolled host; never promotion evidence"
        if candidate
        else "CPU reference single-pass diagnostic timing only; never promotion evidence"
    )
    expected_keys = {
        "classification",
        "generation_total_latency_ms",
        "generation_tps",
        "harness_elapsed_ms",
        "input_tokens",
        "output_tokens",
        "setup",
        "tpot_ms",
        "ttft_ms",
    }
    if not isinstance(profile, dict) or set(profile) != expected_keys:
        raise HarnessError("free-run timing profile schema drifted")
    if (
        profile.get("classification") != expected_classification
        or profile.get("input_tokens") != 13
        or profile.get("output_tokens") != 128
        or any(
            not BASE._positive_number(profile.get(key))
            for key in (
                "generation_total_latency_ms",
                "generation_tps",
                "harness_elapsed_ms",
                "tpot_ms",
                "ttft_ms",
            )
        )
    ):
        raise HarnessError("free-run timing profile is invalid")
    if (
        not math.isclose(
            float(profile["generation_tps"]) * float(profile["tpot_ms"]),
            1_000.0,
            rel_tol=1e-9,
            abs_tol=1e-6,
        )
        or not math.isclose(
            float(profile["generation_total_latency_ms"]),
            float(profile["ttft_ms"]) + 127.0 * float(profile["tpot_ms"]),
            rel_tol=1e-9,
            abs_tol=1e-6,
        )
        or float(profile["harness_elapsed_ms"])
        < float(profile["generation_total_latency_ms"])
    ):
        raise HarnessError("free-run timing profile is internally inconsistent")
    setup = profile.get("setup")
    if (
        not isinstance(setup, dict)
        or set(setup)
        != {"checkpoint_load_ms", "model_construct_ms", "timing_classification"}
        or setup.get("timing_classification")
        != "single-pass diagnostic timing only; never formal promotion evidence"
        or not BASE._positive_number(setup.get("checkpoint_load_ms"))
        or not BASE._positive_number(setup.get("model_construct_ms"))
    ):
        raise HarnessError("free-run setup timing profile is invalid")
    return profile


def validate_run_receipt(path: Path, *, variant: str, frozen: dict) -> dict:
    """Validate one timed CPU-free or Stack3 + lm_head v2 free receipt."""

    if variant not in {"A", "B"}:
        raise HarnessError("run receipt variant must be A or B")
    try:
        identity = frozen["identity"]
        receipts = frozen["receipts"]
        reference_record = frozen["receipt_records"]["cpu_free128"]
        reference = receipts["cpu_free128"]
        frozen_candidate = receipts["candidate_free128"]
        expected_tokens = reference["generated_token_ids"]
        expected_prompt = reference["prompt"]
        expected_prompt_ids = reference["prompt_token_ids"]
    except (KeyError, TypeError) as error:
        raise HarnessError("frozen v2 free-run inputs are incomplete") from error
    if (
        not isinstance(expected_tokens, list)
        or len(expected_tokens) != 128
        or any(type(token) is not int or token < 0 for token in expected_tokens)
    ):
        raise HarnessError("frozen CPU-free reference trajectory is invalid")
    if (
        not isinstance(expected_prompt, str)
        or not isinstance(expected_prompt_ids, list)
        or len(expected_prompt_ids) != 13
        or any(type(token) is not int or token < 0 for token in expected_prompt_ids)
    ):
        raise HarnessError("frozen CPU-free prompt identity is invalid")

    record = BASE.direct_file_record(path, label=f"variant {variant} run receipt")
    receipt = BASE.load_json(path, label=f"variant {variant} run receipt")
    expected_keys = CPU_RECEIPT_KEYS if variant == "A" else CANDIDATE_RECEIPT_KEYS
    if set(receipt) != expected_keys:
        raise HarnessError(f"variant {variant} run receipt schema drifted")
    expected_format = (
        "apxinf-qwen35-metal-w8-stack3-head-v2-cpu-free-run-v1"
        if variant == "A"
        else "apxinf-qwen35-metal-w8-stack3-head-v2-free-run-gate-v1"
    )
    expected_mode = (
        "cpu_free_run" if variant == "A" else "metal_w8_stack3_head_v2_free_run"
    )
    if (
        receipt.get("format") != expected_format
        or receipt.get("mode") != expected_mode
        or receipt.get("passed") is not True
        or receipt.get("official_layer_schedule_valid") is not True
        or receipt.get("max_new_tokens") != 128
        or receipt.get("eos_stopping") is not False
        or not _json_exact_equal(receipt.get("prompt"), expected_prompt)
        or not _json_exact_equal(receipt.get("prompt_token_ids"), expected_prompt_ids)
    ):
        raise HarnessError(f"variant {variant} run receipt identity/status drifted")
    if not _json_exact_equal(receipt.get("identity"), identity):
        raise HarnessError(f"variant {variant} run receipt custody identity drifted")
    if not _json_exact_equal(
        receipt.get("custody_end_verification"), _expected_end_custody(identity)
    ):
        raise HarnessError(f"variant {variant} run receipt end custody drifted")
    profile = _validate_free_profile(receipt.get("profile"), candidate=variant == "B")

    if variant == "A":
        if not _json_exact_equal(receipt.get("generated_token_ids"), expected_tokens):
            raise HarnessError("CPU formal run trajectory drifted from frozen oracle")
        if receipt.get("generation_path_contract") is not None:
            raise HarnessError("CPU formal run unexpectedly reported a Metal path")
    else:
        if not _json_exact_equal(receipt.get("input_receipt"), reference_record):
            raise HarnessError("candidate did not bind frozen CPU-free reference")
        if (
            not _json_exact_equal(
                receipt.get("cpu_expected_token_ids"), expected_tokens
            )
            or not _json_exact_equal(
                receipt.get("generated_token_ids"), expected_tokens
            )
            or not _json_exact_equal(receipt.get("mismatches"), [])
            or receipt.get("exact_trajectory") is not True
        ):
            raise HarnessError("candidate formal run trajectory drifted")
        _require_exact_object(
            receipt.get("generation_path_contract"),
            CANDIDATE_GENERATION_CONTRACT,
            label="candidate generation path contract",
        )
        _require_exact_object(
            receipt.get("path_checks"),
            CANDIDATE_PATH_CHECKS,
            label="candidate path checks",
        )
        if not _json_exact_equal(
            receipt.get("path_checks"), frozen_candidate.get("path_checks")
        ):
            raise HarnessError("candidate path checks drifted from frozen receipt")
        ledger = validate_composite_ledger(receipt.get("aggregate_buffer_ledger"))
        if not _json_exact_equal(
            ledger, frozen_candidate.get("aggregate_buffer_ledger")
        ):
            raise HarnessError("candidate composite ledger drifted from frozen receipt")
        validate_generation_receipt(
            receipt.get("final_generation_path_receipt"),
            body_calls=127,
            head_calls={
                "prefill_calls": 1,
                "decode_calls": 127,
                "teacher_calls": 0,
            },
        )

    return {
        "variant": variant,
        "decode_mean_ms": float(profile["tpot_ms"]),
        "ttft_ms": float(profile["ttft_ms"]),
        "trajectory_sha256": BASE.canonical_json_sha256(expected_tokens),
        "path_valid": True,
        "ledger_valid": True,
        "custody_sha256": BASE.canonical_json_sha256(identity),
        "receipt": record,
    }


def build_gate_argv(*, frozen: dict, mode: str, output: Path) -> list[str]:
    """Build one shell-free command bound to the frozen v2 CPU oracle."""

    try:
        custody = frozen["summary"]["custody"]
        binary_path = custody["binary"]["path"]
        model_path = custody["model_dir"]["path"]
        source_lock_path = custody["source_lock"]["path"]
        cpu_reference_path = frozen["receipt_records"]["cpu_free128"]["path"]
    except (KeyError, TypeError) as error:
        raise HarnessError(
            "frozen Stack3 + lm_head v2 gate custody is incomplete"
        ) from error
    argv = [
        "/usr/bin/time",
        "-l",
        binary_path,
        "--model-dir",
        model_path,
        "--source-lock",
        source_lock_path,
        "--mode",
        mode,
    ]
    if mode == CANDIDATE_MODE:
        argv.extend(["--input-receipt", cpu_reference_path])
    elif mode != CPU_MODE:
        raise HarnessError("formal schedule requested an unapproved v2 gate mode")
    argv.extend(["--output", str(output)])
    return argv


def _valid_token_ids(value: object, *, length: int = 128) -> bool:
    return (
        isinstance(value, list)
        and len(value) == length
        and all(type(token) is int and token >= 0 for token in value)
    )


def _validate_summary_ledger(value: object) -> dict:
    if not isinstance(value, dict) or set(value) != {
        "scope",
        "body",
        "lm_head",
        "composite",
        "independent_component_sum_matches_both_candidate_receipts",
        "candidate_teacher_and_free_ledgers_identical",
        "exclusions",
    }:
        raise HarnessError("archived v2 aggregate ledger schema drifted")
    if (
        value.get("scope") != "resident-mtlbuffer-only"
        or value.get("exclusions") != COMPOSITE_LEDGER_TOTALS["exclusions"]
        or value.get("independent_component_sum_matches_both_candidate_receipts")
        is not True
        or value.get("candidate_teacher_and_free_ledgers_identical") is not True
    ):
        raise HarnessError("archived v2 aggregate ledger admission drifted")
    body = value.get("body")
    head = value.get("lm_head")
    composite = value.get("composite")
    if not isinstance(body, dict) or set(body) != {
        "stack3_component",
        "full_attention_mlp_component",
        "aggregate",
    }:
        raise HarnessError("archived v2 body ledger drifted")
    _require_exact_object(
        body.get("stack3_component"),
        {
            "stack_count": 6,
            "layers_per_stack": 3,
            "persistent_bytes_each": 76_351_488,
            "persistent_bytes_subtotal": 458_108_928,
            "allocated_buffers_each": 76,
            "shared_buffers_each": 68,
            "private_buffers_each": 8,
            "command_buffers_per_decode_each": 1,
            "compute_encoders_per_decode_each": 3,
            "commits_per_decode_each": 1,
            "waits_per_decode_each": 1,
            "host_to_device_bytes_per_decode_each": 4_096,
            "device_to_host_bytes_per_decode_each": 4_096,
            "state_host_transfer_bytes_per_decode_each": 0,
        },
        label="archived Stack3 component ledger",
    )
    _require_exact_object(
        body.get("full_attention_mlp_component"),
        {
            "layer_count": 6,
            "persistent_bytes_each": 11_749_376,
            "persistent_bytes_subtotal": 70_496_256,
            "allocated_buffers_each": 8,
            "shared_buffers_each": 6,
            "private_buffers_each": 2,
            "command_buffers_per_decode_each": 1,
            "compute_encoders_per_decode_each": 3,
            "commits_per_decode_each": 1,
            "waits_per_decode_each": 1,
            "host_to_device_bytes_per_decode_each": 4_096,
            "device_to_host_bytes_per_decode_each": 4_096,
            "state_host_transfer_bytes_per_decode_each": 0,
        },
        label="archived full-attention MLP component ledger",
    )
    _require_exact_object(
        body.get("aggregate"),
        {
            "total_persistent_mtlbuffer_bytes": 528_605_184,
            "allocated_buffers": 504,
            "shared_buffers": 444,
            "private_buffers": 60,
            "command_buffers_per_decode": 12,
            "compute_encoders_per_decode": 36,
            "commits_per_decode": 12,
            "waits_per_decode": 12,
            "host_to_device_bytes_per_decode": 49_152,
            "device_to_host_bytes_per_decode": 49_152,
            "state_host_transfer_bytes_per_decode": 0,
        },
        label="archived body aggregate ledger",
    )
    _require_exact_object(
        head,
        {
            "total_persistent_bytes": 271_169_552,
            "allocated_buffers": 5,
            "shared_buffers": 4,
            "private_buffers": 1,
            "command_buffers_per_call": 1,
            "compute_encoders_per_call": 2,
            "commits_per_call": 1,
            "waits_per_call": 1,
            "host_input_bytes_per_call": 4_096,
            "host_output_bytes_per_call": 16,
            "state_host_transfer_bytes_per_call": 0,
        },
        label="archived lm_head aggregate ledger",
    )
    _require_exact_object(
        composite,
        {
            key: COMPOSITE_LEDGER_TOTALS[key]
            for key in (
                "includes_lm_head",
                "total_persistent_mtlbuffer_bytes",
                "allocated_buffers",
                "shared_buffers",
                "private_buffers",
                "command_buffers_per_call",
                "compute_encoders_per_call",
                "commits_per_call",
                "waits_per_call",
                "host_to_device_bytes_per_call",
                "device_to_host_bytes_per_call",
                "state_host_transfer_bytes_per_call",
                "final_output_finite_checks_per_call",
                "intermediate_host_finite_checks_per_call",
            )
        },
        label="archived 509-buffer composite ledger",
    )
    return value


def _validate_teacher_receipts(
    cpu: dict,
    candidate: dict,
    *,
    cpu_record: dict,
) -> None:
    cpu_inputs = cpu.get("teacher_input_ids")
    expected = cpu.get("cpu_expected_output_ids")
    prefill_token = cpu.get("prefill_token")
    if (
        set(cpu)
        != {
            "comparisons",
            "cpu_expected_output_ids",
            "custody_end_verification",
            "format",
            "generation_path_contract",
            "identity",
            "mode",
            "official_layer_schedule_valid",
            "passed",
            "prefill_token",
            "prompt",
            "prompt_token_ids",
            "teacher_input_ids",
            "timing",
        }
        or cpu.get("comparisons") != 128
        or cpu.get("official_layer_schedule_valid") is not True
        or cpu.get("generation_path_contract") is not None
        or not _valid_token_ids(cpu_inputs)
        or not _valid_token_ids(expected)
        or type(prefill_token) is not int
        or cpu_inputs[0] != prefill_token
        or any(cpu_inputs[index] != expected[index - 1] for index in range(1, 128))
    ):
        raise HarnessError("frozen v2 CPU teacher chain is invalid")
    if set(candidate) != {
        "aggregate_buffer_ledger",
        "candidate_hidden_cpu_f32_output_ids",
        "comparisons",
        "custody_end_verification",
        "exactness",
        "f32_reranked_output_ids",
        "final_generation_path_receipt",
        "format",
        "frozen_cpu_expected_output_ids",
        "generation_path_contract",
        "identity",
        "input_receipt",
        "metal_w8_top4_candidate_ids",
        "mode",
        "official_layer_schedule_valid",
        "passed",
        "path_checks",
        "prefill_generation_path_receipt",
        "prefill_token",
        "prompt",
        "prompt_token_ids",
        "teacher_input_ids",
        "timing",
    }:
        raise HarnessError("frozen v2 candidate teacher schema drifted")
    body_actual = candidate.get("candidate_hidden_cpu_f32_output_ids")
    reranked = candidate.get("f32_reranked_output_ids")
    top4 = candidate.get("metal_w8_top4_candidate_ids")
    if (
        candidate.get("comparisons") != 128
        or candidate.get("official_layer_schedule_valid") is not True
        or not _json_exact_equal(candidate.get("input_receipt"), cpu_record)
        or not _json_exact_equal(candidate.get("teacher_input_ids"), cpu_inputs)
        or candidate.get("prefill_token") != prefill_token
        or not _json_exact_equal(
            candidate.get("frozen_cpu_expected_output_ids"), expected
        )
        or not _json_exact_equal(body_actual, expected)
        or not _json_exact_equal(reranked, body_actual)
        or not isinstance(top4, list)
        or len(top4) != 128
        or any(
            not _valid_token_ids(row, length=4) or body_actual[index] not in row
            for index, row in enumerate(top4)
        )
    ):
        raise HarnessError("frozen v2 teacher body/head trajectory drifted")
    _require_exact_object(
        candidate.get("exactness"),
        {
            "body_mismatches": [],
            "candidate_hidden_cpu_matches_frozen_cpu": True,
            "composite_matches_frozen_cpu": True,
            "end_to_end_mismatches": [],
            "head_mismatches": [],
            "top4_contains_candidate_hidden_winner_and_rerank_matches": True,
        },
        label="frozen v2 teacher exactness",
    )
    _require_exact_object(
        candidate.get("generation_path_contract"),
        {
            "binds_six_stack3_six_full_attention_mlp_and_tied_head": True,
            "head_mechanism": "metal-w8-top4-f32-rerank",
            "schema": "apxinf-qwen35-stack3-lm-head-generation-path-v2",
            "teacher_head_calls": 128,
        },
        label="frozen v2 teacher generation path contract",
    )
    _require_exact_object(
        candidate.get("path_checks"),
        {"prefill": CANDIDATE_PATH_CHECKS, "final": CANDIDATE_PATH_CHECKS},
        label="frozen v2 teacher path checks",
    )
    validate_composite_ledger(candidate.get("aggregate_buffer_ledger"))
    validate_generation_receipt(
        candidate.get("prefill_generation_path_receipt"),
        body_calls=0,
        head_calls={"prefill_calls": 0, "decode_calls": 0, "teacher_calls": 0},
    )
    validate_generation_receipt(
        candidate.get("final_generation_path_receipt"),
        body_calls=128,
        head_calls={"prefill_calls": 0, "decode_calls": 0, "teacher_calls": 128},
    )


def _canonical_direct_record(
    record: object,
    *,
    label: str,
    sha_key: str = "sha256",
    size_key: str = "size",
) -> dict:
    if not isinstance(record, dict):
        raise HarnessError(f"{label} custody record is missing")
    value = {
        "path": record.get("path"),
        "sha256": record.get(sha_key),
        "size": record.get(size_key),
        "direct_regular_file": record.get("direct_regular_file"),
        "single_link": record.get("single_link"),
    }
    if (
        not isinstance(value["path"], str)
        or not Path(value["path"]).is_absolute()
        or not isinstance(value["sha256"], str)
        or len(value["sha256"]) != 64
        or type(value["size"]) is not int
        or value["size"] <= 0
        or value["direct_regular_file"] is not True
        or value["single_link"] is not True
    ):
        raise HarnessError(f"{label} custody record is invalid")
    return value


def _canonical_digest_record(record: object, *, label: str) -> dict:
    if not isinstance(record, dict):
        raise HarnessError(f"{label} custody digest is missing")
    value = {"sha256": record.get("sha256"), "size": record.get("size")}
    if (
        not isinstance(value["sha256"], str)
        or len(value["sha256"]) != 64
        or type(value["size"]) is not int
        or value["size"] <= 0
    ):
        raise HarnessError(f"{label} custody digest is invalid")
    return value


def canonical_summary_custody(custody: object) -> dict:
    if not isinstance(custody, dict):
        raise HarnessError("summary custody is missing")
    sources = custody.get("sources")
    model = custody.get("model_dir")
    if not isinstance(sources, dict) or not isinstance(model, dict):
        raise HarnessError("summary v2 custody closure is incomplete")
    rust = sources.get("rust_and_bridge_sources")
    shaders = sources.get("compiled_metal_shader_sources")
    artifacts = model.get("artifacts")
    if (
        sources.get("captured_at_start") is not True
        or sources.get("closure") != "stack3-lm-head-v2-direct-compile-inputs-v1"
        or not isinstance(rust, dict)
        or set(rust) != RUST_SOURCE_KEYS
        or not isinstance(shaders, dict)
        or set(shaders) != SHADER_SOURCE_KEYS
        or not isinstance(artifacts, dict)
    ):
        raise HarnessError("summary v2 source custody closure drifted")
    model_path = model.get("path")
    if not isinstance(model_path, str) or not Path(model_path).is_absolute():
        raise HarnessError("summary model custody path is invalid")
    profile = custody.get("profile")
    source_lock = custody.get("source_lock")
    return {
        "binary": _canonical_direct_record(custody.get("binary"), label="binary"),
        "gate": _canonical_direct_record(sources.get("gate"), label="gate"),
        "rust": {
            key: _canonical_digest_record(record, label=key)
            for key, record in rust.items()
        },
        "shaders": {
            key: _canonical_digest_record(record, label=key)
            for key, record in shaders.items()
        },
        "profile": _canonical_direct_record(profile, label="profile"),
        "source_lock": _canonical_direct_record(
            source_lock, label="source lock", sha_key="file_sha256"
        ),
        "source_lock_content_sha256": (
            source_lock.get("canonical_content_sha256_without_content_field")
            if isinstance(source_lock, dict)
            else None
        ),
        "model": {
            "path": model_path,
            "artifacts": {
                name: _canonical_direct_record(
                    {**record, "path": str(Path(model_path) / name)},
                    label=f"model artifact {name}",
                )
                for name, record in artifacts.items()
            },
        },
    }


def canonical_identity_custody(custody: object) -> dict:
    if not isinstance(custody, dict):
        raise HarnessError("receipt identity custody is missing")
    sources = custody.get("sources")
    model = custody.get("model_dir")
    if not isinstance(sources, dict) or not isinstance(model, dict):
        raise HarnessError("receipt v2 custody closure is incomplete")
    rust = sources.get("rust_and_bridge_sources")
    shaders = sources.get("compiled_metal_shader_sources")
    artifacts = model.get("artifacts")
    if (
        sources.get("captured_at_start") is not True
        or sources.get("closure") != "stack3-lm-head-v2-direct-compile-inputs-v1"
        or not isinstance(rust, dict)
        or set(rust) != RUST_SOURCE_KEYS
        or not isinstance(shaders, dict)
        or set(shaders) != SHADER_SOURCE_KEYS
        or not isinstance(artifacts, dict)
    ):
        raise HarnessError("receipt identity v2 source custody closure drifted")
    return {
        "binary": _canonical_direct_record(custody.get("binary"), label="binary"),
        "gate": _canonical_direct_record(sources.get("gate"), label="gate"),
        "rust": {
            key: _canonical_digest_record(record, label=key)
            for key, record in rust.items()
        },
        "shaders": {
            key: _canonical_digest_record(record, label=key)
            for key, record in shaders.items()
        },
        "profile": _canonical_direct_record(
            custody.get("profile"),
            label="profile",
            sha_key="file_sha256",
            size_key="file_size",
        ),
        "source_lock": _canonical_direct_record(
            custody.get("source_lock"),
            label="source lock",
            sha_key="file_sha256",
            size_key="file_size",
        ),
        "source_lock_content_sha256": custody.get("source_lock", {}).get(
            "content_sha256"
        ),
        "model": {
            "path": model.get("path"),
            "artifacts": {
                name: _canonical_direct_record(record, label=f"model artifact {name}")
                for name, record in artifacts.items()
            },
        },
    }


def _validate_direct_record(
    record: object,
    *,
    label: str,
    sha_key: str = "sha256",
    size_key: str = "size",
) -> dict:
    canonical = _canonical_direct_record(
        record, label=label, sha_key=sha_key, size_key=size_key
    )
    observed = BASE.direct_file_record(
        Path(canonical["path"]), label=f"{label} custody"
    )
    if observed["sha256"] != canonical["sha256"]:
        raise HarnessError(f"{label} custody SHA-256 drifted")
    if observed["size"] != canonical["size"]:
        raise HarnessError(f"{label} custody size drifted")
    return observed


def _validate_base_engine_contract() -> None:
    expected = {
        "BLOCK_ORDERS": BLOCK_ORDERS,
        "QUIET_SAMPLE_COUNT": 5,
        "QUIET_SAMPLE_INTERVAL_SECONDS": 0.5,
        "QUIET_MAX_PROCESS_CPU_PERCENT": 5.0,
        "QUIET_MAX_LOAD_PER_LOGICAL_CPU": 0.50,
        "RUN_TIMEOUT_SECONDS": 600,
        "RUN_RSS_LIMIT_BYTES": 6 * 1024 * 1024 * 1024,
        "RUN_STREAM_LIMIT_BYTES": 4 * 1024 * 1024,
        "RUN_QUIET_SAMPLE_INTERVAL_SECONDS": 1.0,
        "MINIMUM_MEDIAN_SPEEDUP": 1.10,
        "MAXIMUM_TTFT_RATIO": 1.10,
    }
    for name, value in expected.items():
        if not _json_exact_equal(getattr(BASE, name, None), value):
            raise HarnessError(f"audited base campaign contract drifted: {name}")


def validate_harness_custody() -> dict:
    """Rehash this wrapper plus the pinned audited base campaign engine."""

    _validate_base_engine_contract()
    wrapper = Path(__file__).resolve(strict=True)
    base = BASE_HARNESS_PATH.resolve(strict=True)
    records = {
        "wrapper": {
            **BASE.direct_file_record(wrapper, label="v2 formal wrapper"),
            "direct_regular_file": True,
            "single_link": True,
        },
        "audited_base": {
            **BASE.direct_file_record(base, label="audited base campaign engine"),
            "direct_regular_file": True,
            "single_link": True,
        },
    }
    if records["audited_base"]["sha256"] != PINNED_BASE_HARNESS_SHA256:
        raise HarnessError("audited base campaign engine SHA-256 drifted")
    return records


def validate_live_custody(
    summary: dict,
    *,
    expected_binary_sha256: str = PINNED_BINARY_SHA256,
    identity: dict | None = None,
) -> dict:
    """Rehash the frozen binary, model, and complete v2 compile-input closure."""

    custody = summary.get("custody")
    if identity is None or not isinstance(identity, dict):
        raise HarnessError("live v2 custody requires the frozen receipt identity")
    identity_custody = identity.get("custody")
    if canonical_summary_custody(custody) != canonical_identity_custody(
        identity_custody
    ):
        raise HarnessError("summary and receipt v2 custody identities drifted")
    if not isinstance(custody, dict) or not isinstance(identity_custody, dict):
        raise HarnessError("v2 custody is missing")
    binary = _validate_direct_record(identity_custody.get("binary"), label="binary")
    if binary["sha256"] != expected_binary_sha256:
        raise HarnessError("Stack3 + lm_head v2 binary SHA-256 does not match pin")
    sources = identity_custody.get("sources")
    if not isinstance(sources, dict):
        raise HarnessError("v2 source custody is missing")
    gate = _validate_direct_record(sources.get("gate"), label="v2 gate source")
    rust = sources.get("rust_and_bridge_sources")
    shaders = sources.get("compiled_metal_shader_sources")
    if not isinstance(rust, dict) or set(rust) != RUST_SOURCE_KEYS:
        raise HarnessError("v2 Rust/bridge source closure drifted")
    if not isinstance(shaders, dict) or set(shaders) != SHADER_SOURCE_KEYS:
        raise HarnessError("v2 compiled Metal shader closure drifted")
    observed_rust = {
        name: _validate_direct_record(record, label=f"{name} source")
        for name, record in rust.items()
    }
    observed_shaders = {
        name: _validate_direct_record(record, label=f"{name} shader")
        for name, record in shaders.items()
    }
    profile = _validate_direct_record(
        identity_custody.get("profile"),
        label="profile",
        sha_key="file_sha256",
        size_key="file_size",
    )
    source_lock = _validate_direct_record(
        identity_custody.get("source_lock"),
        label="source lock",
        sha_key="file_sha256",
        size_key="file_size",
    )
    model = identity_custody.get("model_dir")
    if not isinstance(model, dict):
        raise HarnessError("model custody is missing")
    model_path_value = model.get("path")
    if (
        not isinstance(model_path_value, str)
        or not Path(model_path_value).is_absolute()
    ):
        raise HarnessError("model custody path must be absolute")
    model_path = Path(model_path_value)
    try:
        entry = model_path.lstat()
    except OSError as error:
        raise HarnessError(f"model custody is unavailable: {error}") from error
    if stat.S_ISLNK(entry.st_mode) or not stat.S_ISDIR(entry.st_mode):
        raise HarnessError("model custody must be a direct directory")
    artifacts = model.get("artifacts")
    if (
        model.get("closure") != "exact-profile-artifacts-plus-safe-cache-v1"
        or model.get("cache_present") is not False
        or not isinstance(artifacts, dict)
        or not artifacts
    ):
        raise HarnessError("model custody closure drifted")
    try:
        actual_names = {child.name for child in model_path.iterdir()}
    except OSError as error:
        raise HarnessError(f"model custody cannot be enumerated: {error}") from error
    if actual_names != set(artifacts):
        raise HarnessError("model custody top-level artifact set drifted")
    observed_artifacts = {}
    for name, record in artifacts.items():
        if Path(name).name != name or not isinstance(record, dict):
            raise HarnessError("model artifact custody is invalid")
        observed_artifacts[name] = _validate_direct_record(
            record, label=f"model artifact {name}"
        )
    for flag in (
        "same_identity_in_all_four_receipts",
        "same_end_verification_in_all_four_receipts",
        "start_and_end_binary_records_equal",
        "start_and_end_source_records_equal",
        "end_verified_unchanged_flag_in_all_four_receipts",
        "model_start_receipt_present_in_all_four_receipts",
    ):
        if custody.get(flag) is not True:
            raise HarnessError("archived v2 custody admission flags drifted")
    if custody.get("model_record_repeated_in_end_json") is not False:
        raise HarnessError("archived v2 model end-verification semantics drifted")
    independent = custody.get("independent_live_rehash")
    if (
        not isinstance(independent, dict)
        or independent.get("performed") is not True
        or independent.get("file_count") != 27
        or independent.get("mismatch_count") != 0
        or independent.get(
            "matched_binary_profile_source_lock_model_and_source_start_records"
        )
        is not True
        or independent.get("model_top_level_artifact_set_exact") is not True
        or independent.get("source_lock_python_canonical_content_sha256_recomputed")
        is not True
    ):
        raise HarnessError("archived independent v2 custody rehash drifted")
    return {
        "binary": binary,
        "sources": {
            "gate": gate,
            "rust_and_bridge_sources": observed_rust,
            "compiled_metal_shader_sources": observed_shaders,
        },
        "profile": profile,
        "source_lock": source_lock,
        "model_dir": {
            "path": str(model_path),
            "artifacts": observed_artifacts,
            "cache_present": False,
        },
    }


def validate_frozen_inputs(
    summary_path: Path,
    *,
    repo_root: Path,
    expected_summary_sha256: str = PINNED_SUMMARY_SHA256,
    expected_binary_sha256: str = PINNED_BINARY_SHA256,
) -> dict:
    """Bind a formal campaign to the exact archived v2 correctness gate."""

    summary_record = BASE.direct_file_record(summary_path, label="frozen v2 summary")
    observed_sha = summary_record["sha256"]
    if observed_sha != expected_summary_sha256:
        raise HarnessError("frozen v2 summary SHA-256 does not match pin")
    summary = BASE.load_json(summary_path, label="frozen v2 summary")
    if (
        set(summary)
        != {
            "format",
            "created_at_local_date",
            "scope",
            "classification",
            "receipt_integrity",
            "trajectory_gate",
            "custody",
            "execution_counts",
            "aggregate_buffer_ledger",
            "resource_observations",
            "diagnostic_timing_observations",
            "review_method",
            "gate_result",
        }
        or summary.get("format") != SUMMARY_FORMAT
    ):
        raise HarnessError("frozen v2 summary schema/format is not admitted")
    integrity = summary.get("receipt_integrity")
    integrity_keys = set(RECEIPT_KEYS) | {
        "all_four_identity_records_equal",
        "identity_record_jq_cS_with_trailing_lf_sha256",
        "all_four_end_custody_records_equal",
        "end_custody_record_jq_cS_with_trailing_lf_sha256",
        "all_four_independently_rehashed",
    }
    if not isinstance(integrity, dict) or set(integrity) != integrity_keys:
        raise HarnessError("frozen v2 receipt integrity is missing")
    if (
        integrity.get("all_four_identity_records_equal") is not True
        or integrity.get("all_four_end_custody_records_equal") is not True
        or integrity.get("all_four_independently_rehashed") is not True
        or any(
            not isinstance(integrity.get(key), str) or len(integrity[key]) != 64
            for key in (
                "identity_record_jq_cS_with_trailing_lf_sha256",
                "end_custody_record_jq_cS_with_trailing_lf_sha256",
            )
        )
    ):
        raise HarnessError("frozen v2 receipt integrity flags drifted")
    receipts = {}
    receipt_records = {}
    for key in RECEIPT_KEYS:
        declared = integrity.get(key)
        if not isinstance(declared, dict):
            raise HarnessError(f"frozen v2 receipt record is missing: {key}")
        expected_record_keys = {
            "path",
            "format",
            "mode",
            "sha256",
            "size",
            "direct_regular_file",
            "single_link",
            "passed",
        }
        if key.startswith("candidate_"):
            expected_record_keys |= {
                "input_receipt_path",
                "input_receipt_sha256",
                "input_receipt_size",
                "embedded_cpu_receipt_reference_matches",
            }
        if (
            set(declared) != expected_record_keys
            or declared.get("direct_regular_file") is not True
            or declared.get("single_link") is not True
            or declared.get("passed") is not True
        ):
            raise HarnessError(f"frozen v2 receipt record drifted: {key}")
        receipt_path = BASE.resolve_repo_file(
            repo_root, declared.get("path"), label=f"{key} receipt"
        )
        record = BASE.direct_file_record(receipt_path, label=f"{key} receipt")
        if record["sha256"] != declared.get("sha256") or record["size"] != declared.get(
            "size"
        ):
            raise HarnessError(f"{key} receipt digest does not match summary")
        receipt = BASE.load_json(receipt_path, label=f"{key} receipt")
        expected_format, expected_mode = RECEIPT_IDENTITY[key]
        if (
            declared.get("format") != expected_format
            or declared.get("mode") != expected_mode
            or receipt.get("format") != expected_format
            or receipt.get("mode") != expected_mode
            or receipt.get("passed") is not True
        ):
            raise HarnessError(f"{key} is not an admitted passing v2 receipt")
        receipts[key] = receipt
        receipt_records[key] = {
            **record,
            "direct_regular_file": True,
            "single_link": True,
        }
    for candidate_key, reference_key in (
        ("candidate_teacher128", "cpu_teacher128"),
        ("candidate_free128", "cpu_free128"),
    ):
        declared = integrity[candidate_key]
        reference_record = receipt_records[reference_key]
        if (
            declared.get("input_receipt_path") != reference_record["path"]
            or declared.get("input_receipt_sha256") != reference_record["sha256"]
            or declared.get("input_receipt_size") != reference_record["size"]
            or declared.get("embedded_cpu_receipt_reference_matches") is not True
        ):
            raise HarnessError(f"{candidate_key} summary input-receipt binding drifted")
    identities = [receipts[key].get("identity") for key in RECEIPT_KEYS]
    if not isinstance(identities[0], dict) or any(
        not _json_exact_equal(identity, identities[0]) for identity in identities[1:]
    ):
        raise HarnessError("the four frozen v2 receipt identities drifted")
    identity = identities[0]
    expected_end = _expected_end_custody(identity)
    if any(
        not _json_exact_equal(
            receipts[key].get("custody_end_verification"), expected_end
        )
        for key in RECEIPT_KEYS
    ):
        raise HarnessError("the four frozen v2 end custody receipts drifted")

    _validate_teacher_receipts(
        receipts["cpu_teacher128"],
        receipts["candidate_teacher128"],
        cpu_record=receipt_records["cpu_teacher128"],
    )
    frozen_partial = {
        "identity": identity,
        "receipts": receipts,
        "receipt_records": receipt_records,
    }
    validate_run_receipt(
        Path(receipt_records["cpu_free128"]["path"]),
        variant="A",
        frozen=frozen_partial,
    )
    validate_run_receipt(
        Path(receipt_records["candidate_free128"]["path"]),
        variant="B",
        frozen=frozen_partial,
    )
    if not _json_exact_equal(
        receipts["candidate_teacher128"].get("aggregate_buffer_ledger"),
        receipts["candidate_free128"].get("aggregate_buffer_ledger"),
    ):
        raise HarnessError("candidate teacher/free composite ledgers drifted")
    if canonical_summary_custody(summary.get("custody")) != canonical_identity_custody(
        identity.get("custody")
    ):
        raise HarnessError("summary and receipt v2 custody identities drifted")

    trajectory = summary.get("trajectory_gate")
    compact = (
        trajectory.get("compact_token_array_without_trailing_lf", {})
        if isinstance(trajectory, dict)
        else {}
    )
    free_gate = compact.get("free_run", {}) if isinstance(compact, dict) else {}
    teacher_gate = (
        compact.get("teacher_forced", {}) if isinstance(compact, dict) else {}
    )
    expected_teacher_sha = BASE.canonical_json_sha256(
        receipts["cpu_teacher128"]["cpu_expected_output_ids"]
    )
    expected_free_sha = BASE.canonical_json_sha256(
        receipts["cpu_free128"]["generated_token_ids"]
    )
    if (
        not isinstance(trajectory, dict)
        or trajectory.get("all_four_receipts_passed") is not True
        or trajectory.get("all_required_candidate_path_checks_passed") is not True
        or trajectory.get(
            "intermediate_host_finite_checks_intentionally_disabled_by_versioned_contract"
        )
        is not True
        or trajectory.get("final_output_finite_checks_enabled") is not True
        or teacher_gate.get("exact_128_of_128") is not True
        or teacher_gate.get("cpu_sha256") != expected_teacher_sha
        or teacher_gate.get("candidate_hidden_cpu_f32_sha256") != expected_teacher_sha
        or teacher_gate.get("candidate_composite_sha256") != expected_teacher_sha
        or free_gate.get("exact_128_tokens") is not True
        or free_gate.get("cpu_sha256") != expected_free_sha
        or free_gate.get("candidate_sha256") != expected_free_sha
    ):
        raise HarnessError("archived v2 trajectory/path gate is not passing")
    _validate_summary_ledger(summary.get("aggregate_buffer_ledger"))
    counts = summary.get("execution_counts")
    if (
        not isinstance(counts, dict)
        or counts.get("duplicate_mlp_execution") is not False
        or counts.get("all_generation_and_aggregate_receipts_consistent") is not True
        or counts.get("teacher_forced_decode", {}).get("lm_head")
        != {"prefill_calls": 0, "decode_calls": 0, "teacher_calls": 128}
        or counts.get("free_run", {}).get("lm_head")
        != {"prefill_calls": 1, "decode_calls": 127, "teacher_calls": 0}
    ):
        raise HarnessError("archived v2 execution counts drifted")
    classification = summary.get("classification")
    gate = summary.get("gate_result")
    scope = summary.get("scope")
    binary_summary = summary.get("custody", {}).get("binary")
    if (
        not isinstance(scope, dict)
        or scope.get("teacher_forced_comparisons") != 128
        or scope.get("free_run_generated_tokens") != 128
        or scope.get("linear_attention_complete_layer_stacks")
        != [list(indices) for indices in STACK_LAYER_INDICES]
        or scope.get("full_attention_cpu_attention_metal_mlp_layers")
        != list(FULL_ATTENTION_LAYER_INDICES)
        or scope.get("stack_mechanism") != "metal-w8-linear-layer-stack3-v1"
        or scope.get("full_attention_mlp_mechanism") != "metal-w8-mlp-block-g64"
        or scope.get("lm_head_mechanism") != "metal-w8-top4-f32-rerank"
        or scope.get("generation_path_schema")
        != "apxinf-qwen35-stack3-lm-head-generation-path-v2"
        or not isinstance(binary_summary, dict)
        or binary_summary.get("sha256") != expected_binary_sha256
        or binary_summary.get("build_profile") != "release"
        or binary_summary.get("features") != ["accelerate", "metal-w8"]
        or not isinstance(classification, dict)
        or classification.get("formal") is not False
        or classification.get("default") is not False
        or classification.get("formal_abba_or_baab") is not False
        or not isinstance(gate, dict)
        or gate.get("correctness_and_path_gate_passed") is not True
        or gate.get("candidate_path_contract_valid") is not True
        or gate.get("custody_valid_for_correctness_archive") is not True
        or gate.get("execution_counts_valid") is not True
        or gate.get("aggregate_ledger_valid") is not True
        or gate.get("performance_promotion_gate_passed") is not False
        or gate.get("default_path_changed") is not False
        or gate.get("independent_archive_review")
        != {"p0_findings": 0, "p1_findings": 0, "p2_findings": 0}
    ):
        raise HarnessError("archived v2 correctness/classification gate drifted")

    harness_custody = validate_harness_custody()
    live_artifacts = validate_live_custody(
        summary,
        expected_binary_sha256=expected_binary_sha256,
        identity=identity,
    )
    return {
        "summary_path": str(summary_path),
        "summary_sha256": observed_sha,
        "summary_record": {
            **summary_record,
            "direct_regular_file": True,
            "single_link": True,
        },
        "summary": summary,
        "receipts": receipts,
        "receipt_records": receipt_records,
        "identity": identity,
        "harness_custody": harness_custody,
        "live_custody": {"v2_artifacts": live_artifacts, "harness": harness_custody},
        "expected_binary_sha256": expected_binary_sha256,
    }


class Stack3HeadV2CampaignLane:
    """Adapter from the v2 receipt schema to the audited campaign engine."""

    report_format = "apxinf-qwen35-stack3-head-v2-formal-benchmark-v1"

    def __init__(self, frozen: dict):
        self.frozen = frozen
        try:
            self.report_identity = {
                "format": "apxinf-qwen35-stack3-head-v2-formal-lane-identity-v1",
                "harness_custody": frozen["harness_custody"],
                "frozen_summary": {
                    "path": frozen["summary_path"],
                    "sha256": frozen["summary_sha256"],
                    "size": frozen["summary_record"]["size"],
                    "direct_regular_file": True,
                    "single_link": True,
                },
                "frozen_reference_oracle": frozen["receipt_records"]["cpu_free128"],
                "frozen_binary_sha256": frozen["expected_binary_sha256"],
                "generation_path_schema": (
                    "apxinf-qwen35-stack3-lm-head-generation-path-v2"
                ),
                "aggregate_buffer_ledger": {
                    "allocated_buffers": 509,
                    "shared_buffers": 448,
                    "private_buffers": 61,
                    "total_persistent_mtlbuffer_bytes": 799_774_736,
                },
            }
        except (KeyError, TypeError) as error:
            raise HarnessError("frozen v2 lane identity is incomplete") from error

    def prepare_campaign(self, frozen: dict) -> dict:
        if frozen is not self.frozen:
            raise HarnessError("v2 campaign adapter received a different freeze")
        try:
            return {
                "identity": frozen["identity"],
                "summary": frozen["summary"],
                "expected_tokens": frozen["receipts"]["cpu_free128"][
                    "generated_token_ids"
                ],
                "live_custody_start": frozen["live_custody"],
            }
        except (KeyError, TypeError) as error:
            raise HarnessError("frozen v2 campaign inputs are incomplete") from error

    def validate_live_custody(self, summary: dict) -> dict:
        try:
            summary_record = self.frozen["summary_record"]
            reference_record = self.frozen["receipt_records"]["cpu_free128"]
            expected_reference = self.frozen["receipts"]["cpu_free128"]
            identity = self.frozen["identity"]
            expected_binary = self.frozen["expected_binary_sha256"]
        except (KeyError, TypeError) as error:
            raise HarnessError("frozen v2 oracle custody is incomplete") from error
        observed_summary_record = BASE.direct_file_record(
            Path(summary_record["path"]), label="frozen v2 summary"
        )
        if not _json_exact_equal(
            observed_summary_record,
            {key: summary_record[key] for key in ("path", "size", "sha256")},
        ):
            raise HarnessError("frozen v2 summary drifted")
        observed_summary = BASE.load_json(
            Path(summary_record["path"]), label="frozen v2 summary"
        )
        if not _json_exact_equal(observed_summary, summary):
            raise HarnessError("frozen v2 summary content drifted")
        observed_record = BASE.direct_file_record(
            Path(reference_record["path"]), label="frozen v2 CPU-free reference"
        )
        if not _json_exact_equal(
            observed_record,
            {key: reference_record[key] for key in ("path", "size", "sha256")},
        ):
            raise HarnessError("frozen v2 CPU-free reference receipt drifted")
        observed_reference = BASE.load_json(
            Path(reference_record["path"]), label="frozen v2 CPU-free reference"
        )
        if not _json_exact_equal(observed_reference, expected_reference):
            raise HarnessError("frozen v2 CPU-free reference is no longer admitted")
        validate_run_receipt(
            Path(reference_record["path"]), variant="A", frozen=self.frozen
        )
        return {
            "v2_artifacts": validate_live_custody(
                summary,
                expected_binary_sha256=expected_binary,
                identity=identity,
            ),
            "harness": validate_harness_custody(),
        }

    @staticmethod
    def build_schedule(output_dir: Path) -> dict:
        return build_schedule(output_dir)

    @staticmethod
    def build_gate_argv(*, frozen: dict, mode: str, output: Path) -> list[str]:
        return build_gate_argv(frozen=frozen, mode=mode, output=output)

    @staticmethod
    def validate_run_receipt(path: Path, *, variant: str, frozen: dict) -> dict:
        return validate_run_receipt(path, variant=variant, frozen=frozen)


def execute_campaign(
    *,
    frozen: dict,
    repo_root: Path,
    output_dir: Path,
    quiet_probe,
    run_quiet_sample_probe=BASE.capture_quiet_host_sample,
    command_runner=BASE.run_supervised,
    swap_probe=None,
) -> dict:
    """Run v2 only through the pinned, audited formal-campaign engine."""

    if not output_dir.is_absolute():
        raise HarnessError("formal output directory must be absolute")
    try:
        model_path = Path(frozen["summary"]["custody"]["model_dir"]["path"]).resolve(
            strict=True
        )
    except (KeyError, TypeError, OSError) as error:
        raise HarnessError("frozen model output exclusion is unavailable") from error
    resolved_output = output_dir.resolve(strict=False)
    try:
        resolved_output.relative_to(model_path)
    except ValueError:
        pass
    else:
        raise HarnessError("formal output directory must remain outside the model")
    return BASE.execute_campaign(
        frozen=frozen,
        repo_root=repo_root,
        output_dir=output_dir,
        quiet_probe=quiet_probe,
        run_quiet_sample_probe=run_quiet_sample_probe,
        command_runner=command_runner,
        swap_probe=swap_probe,
        lane=Stack3HeadV2CampaignLane(frozen),
    )


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    value.add_argument("--output-dir", type=Path)
    actions = value.add_mutually_exclusive_group()
    actions.add_argument("--execute", action="store_true")
    actions.add_argument("--preflight-only", action="store_true")
    return value


def main(argv: list[str] | None = None) -> int:
    arguments = parser().parse_args(argv)
    try:
        if arguments.preflight_only:
            result = BASE.quiet_host_preflight()
            print(json.dumps(result, sort_keys=True, separators=(",", ":")))
            return 0 if result["passed"] else 2
        frozen = validate_frozen_inputs(arguments.summary, repo_root=REPO_ROOT)
        output_dir = arguments.output_dir
        if arguments.execute:
            if output_dir is None or not output_dir.is_absolute():
                raise HarnessError("--execute requires an absolute --output-dir")
            report = execute_campaign(
                frozen=frozen,
                repo_root=REPO_ROOT,
                output_dir=output_dir,
                quiet_probe=BASE.quiet_host_preflight,
            )
            print(json.dumps(report, sort_keys=True, separators=(",", ":")))
            return 0 if report["formal_accepted"] else 3
        dry_output = output_dir or Path(
            "/private/tmp/apxinf-qwen35-stack3-head-v2-formal-not-started"
        )
        plan = {
            "format": "apxinf-qwen35-stack3-head-v2-formal-plan-v1",
            "execution_started": False,
            "requires_explicit_execute": True,
            "frozen_summary_sha256": frozen["summary_sha256"],
            "frozen_summary": frozen["summary_record"],
            "frozen_binary_sha256": frozen["expected_binary_sha256"],
            "frozen_reference_oracle": frozen["receipt_records"]["cpu_free128"],
            "harness_custody": frozen["harness_custody"],
            "formal_contract": {
                "schedule": "3xABBA+3xBAAB",
                "baseline_samples": 12,
                "candidate_samples": 12,
                "same_direction_block_medians_required": 6,
                "median_speedup_minimum": BASE.MINIMUM_MEDIAN_SPEEDUP,
                "ttft_ratio_maximum": BASE.MAXIMUM_TTFT_RATIO,
                "process_group_rss_limit_bytes": BASE.RUN_RSS_LIMIT_BYTES,
                "child_swaps_required": 0,
                "timeout_seconds": BASE.RUN_TIMEOUT_SECONDS,
                "stdout_stderr_limit_bytes_each": BASE.RUN_STREAM_LIMIT_BYTES,
                "trajectory_tokens": 128,
                "generation_path_schema": (
                    "apxinf-qwen35-stack3-lm-head-generation-path-v2"
                ),
                "aggregate_buffers": 509,
            },
            "schedule": build_schedule(dry_output),
        }
        print(json.dumps(plan, sort_keys=True, separators=(",", ":")))
        return 0
    except HarnessError as error:
        print(
            json.dumps(
                {"formal_accepted": False, "status": "blocked", "error": str(error)},
                sort_keys=True,
                separators=(",", ":"),
            ),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
