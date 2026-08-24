#!/usr/bin/env python3
"""Fail-closed formal benchmark harness for the frozen Qwen3.5 Stack3 lane.

The default command validates frozen evidence and emits a plan. A real model
can start only when the caller explicitly supplies ``--execute``.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import stat
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_HARNESS_PATH = REPO_ROOT / "scripts" / "run_qwen35_all18_formal_benchmark.py"


def _load_base_harness():
    module_name = "_apxinf_qwen35_all18_formal_benchmark_base"
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
CANDIDATE_MODE = "all-linear-layer-stacks-v1-free"
PINNED_BINARY_SHA256 = (
    "882441d89c820031bd61afebd2ffdf12de49817f16f73a9e6c48556bc7f55007"
)
PINNED_SUMMARY_SHA256 = (
    "6e4e6551336b55b7ce4131fc94f2ff2820d62bd98cf2951ee327fca488d926c0"
)
SUMMARY_FORMAT = (
    "apxinf-qwen35-metal-w8-all-linear-layer-stacks-v1-real-gate-summary-v1"
)
DEFAULT_SUMMARY = REPO_ROOT / (
    "crates/apxinf-metal/evidence/next-hotspot/"
    "qwen35-all-linear-layer-stacks-v1-real-gate-summary-v1-20260824.json"
)
RECEIPT_KEYS = (
    "cpu_teacher128",
    "candidate_teacher128",
    "cpu_free128",
    "candidate_free128",
)
RECEIPT_IDENTITY = {
    "cpu_teacher128": (
        "apxinf-qwen35-metal-w8-linear-layer-cpu-teacher-v1",
        "linear_layer_cpu_teacher",
    ),
    "candidate_teacher128": (
        "apxinf-qwen35-metal-w8-all-linear-layer-stacks-v1-teacher-gate-v1",
        "metal_w8_all_linear_layer_stacks_v1_teacher_forced",
    ),
    "cpu_free128": (
        "apxinf-qwen35-metal-w8-linear-layer-cpu-free-run-v1",
        "linear_layer_cpu_free_run",
    ),
    "candidate_free128": (
        "apxinf-qwen35-metal-w8-all-linear-layer-stacks-v1-free-run-gate-v1",
        "metal_w8_all_linear_layer_stacks_v1_free_run",
    ),
}
RUST_SOURCE_KEYS = {
    "apxinf_metal_build",
    "gate_evidence",
    "general",
    "stack3_bridge",
    "stack3_rust",
}
SHADER_SOURCE_KEYS = {
    "metal_w8_gdn",
    "metal_w8_gdn_out_g32",
    "metal_w8_linear_layer",
    "metal_w8_mlp",
}
STACK_LAYER_INDICES = (
    (0, 1, 2),
    (4, 5, 6),
    (8, 9, 10),
    (12, 13, 14),
    (16, 17, 18),
    (20, 21, 22),
)
FULL_ATTENTION_LAYER_INDICES = (3, 7, 11, 15, 19, 23)
LEDGER_TOTALS = {
    "allocated_buffers": 504,
    "shared_buffers": 444,
    "private_buffers": 60,
    "total_persistent_mtlbuffer_bytes": 528_605_184,
    "command_buffers_per_decode": 12,
    "compute_encoders_per_decode": 36,
    "commits_per_decode": 12,
    "waits_per_decode": 12,
    "host_to_device_bytes_per_decode": 49_152,
    "device_to_host_bytes_per_decode": 49_152,
    "state_host_transfer_bytes_per_decode": 0,
    "final_output_finite_checks_per_decode": 6,
    "intermediate_host_finite_checks_per_decode": 0,
}


def build_schedule(output_dir: Path) -> dict:
    """Return the immutable 24-run Stack3 paired measurement schedule."""

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
            _json_exact_equal(observed_item, expected_item)
            for observed_item, expected_item in zip(observed, expected)
        )
    return observed == expected


def _require_exact_fields(value: object, expected: dict, *, label: str) -> dict:
    if not isinstance(value, dict) or any(
        key not in value or not _json_exact_equal(value[key], expected_value)
        for key, expected_value in expected.items()
    ):
        raise HarnessError(f"{label} drifted")
    return value


def validate_stack3_ledger(ledger: object) -> dict:
    """Require the audited 504-buffer Stack3 aggregate and its components."""

    value = _require_exact_fields(ledger, LEDGER_TOTALS, label="Stack3 Metal ledger")
    if (
        value.get("scope") != "resident-mtlbuffer-only"
        or value.get("includes_lm_head") is not False
    ):
        raise HarnessError("Stack3 Metal ledger scope drifted")
    stacks = value.get("stacks")
    full_layers = value.get("full_attention_mlp_layers")
    if not isinstance(stacks, list) or len(stacks) != 6:
        raise HarnessError("Stack3 Metal ledger must contain six stacks")
    if not isinstance(full_layers, list) or len(full_layers) != 6:
        raise HarnessError("Stack3 Metal ledger must contain six full-attention MLPs")
    stack_expected = {
        "allocated_buffers": 76,
        "shared_buffers": 68,
        "private_buffers": 8,
        "total_persistent_bytes": 76_351_488,
        "command_buffers_per_decode": 1,
        "compute_encoders_per_decode": 3,
        "commits_per_decode": 1,
        "waits_per_decode": 1,
        "host_input_bytes_per_decode": 4096,
        "host_output_bytes_per_decode": 4096,
        "state_host_transfer_bytes_per_decode": 0,
        "final_output_finite_checks_per_decode": 1,
        "intermediate_host_finite_checks_per_decode": 0,
    }
    for index, expected_indices in enumerate(STACK_LAYER_INDICES):
        stack = stacks[index]
        if not isinstance(stack, dict) or not _json_exact_equal(
            stack.get("layer_indices"), list(expected_indices)
        ):
            raise HarnessError("Stack3 Metal ledger stack schedule drifted")
        _require_exact_fields(
            stack.get("ledger"), stack_expected, label="Stack3 stack ledger"
        )
    mlp_expected = {
        "allocated_buffers": 8,
        "shared_buffers": 6,
        "private_buffers": 2,
        "total_persistent_bytes": 11_749_376,
        "command_buffers_per_decode": 1,
        "compute_encoders_per_decode": 3,
        "commits_per_decode": 1,
        "waits_per_decode": 1,
        "host_input_bytes_per_decode": 4096,
        "host_output_bytes_per_decode": 4096,
        "state_host_transfer_bytes_per_decode": 0,
    }
    for index, expected_layer in enumerate(FULL_ATTENTION_LAYER_INDICES):
        layer = full_layers[index]
        if not isinstance(layer, dict) or not _json_exact_equal(
            layer.get("layer_index"), expected_layer
        ):
            raise HarnessError("Stack3 full-attention MLP ledger schedule drifted")
        _require_exact_fields(
            layer.get("ledger"), mlp_expected, label="Stack3 full-attention MLP ledger"
        )
    return value


def validate_stack3_path_checks(path_checks: object) -> dict:
    phase_expected = {
        "aggregate_ledger_valid": True,
        "all_valid": True,
        "final_output_finite_checks": True,
        "finite_check_contract_valid": True,
        "full_attention_mlp_execution_valid": True,
        "intermediate_host_finite_checks": False,
        "no_duplicate_mlp": True,
        "precision_contract_valid": True,
        "schedule_valid": True,
        "stack_execution_valid": True,
        "terminal_clear": True,
    }
    value = _require_exact_fields(
        path_checks,
        {
            "decode_generation_receipt_valid": True,
            "exact_trajectory": True,
            "prefill_generation_receipt_valid": True,
        },
        label="Stack3 path checks",
    )
    _require_exact_fields(
        value.get("prefill"), phase_expected, label="Stack3 prefill path checks"
    )
    _require_exact_fields(
        value.get("decode"), phase_expected, label="Stack3 decode path checks"
    )
    return value


def _expected_stack_execution(decode_calls: int) -> dict:
    return {
        "decode_calls": decode_calls,
        "successful_decodes": decode_calls,
        "failed_decodes": 0,
        "command_buffers": decode_calls,
        "compute_encoders": decode_calls * 3,
        "commits": decode_calls,
        "waits": decode_calls,
        "state_commits": decode_calls * 3,
        "last_state_commit_mask": 7 if decode_calls else 0,
        "committed_stack_version": decode_calls,
        "host_to_device_bytes": decode_calls * 4096,
        "device_to_host_bytes": decode_calls * 4096,
        "terminal_error": False,
    }


def validate_stack3_execution(receipt: dict, *, decode_calls: int) -> None:
    """Validate all six Stack3 transactions and six full-attention MLPs."""

    aggregate = receipt.get("final_aggregate_path_receipt")
    generation = receipt.get("final_generation_path_receipt")
    expected_execution = _expected_stack_execution(decode_calls)
    if not isinstance(aggregate, dict) or not isinstance(generation, dict):
        raise HarnessError("Stack3 execution receipts are missing")
    if (
        aggregate.get("mechanism") != "metal-w8-linear-layer-stack3-v1"
        or aggregate.get("full_attention_mlp_mechanism") != "metal-w8-mlp-block-g64"
        or aggregate.get("terminal_error") is not False
        or generation.get("format")
        != "apxinf-qwen35-linear-layer-stacks-generation-path-v1"
        or generation.get("mechanism") != "metal-w8-linear-layer-stack3-v1"
        or generation.get("full_attention_mlp_mechanism") != "metal-w8-mlp-block-g64"
        or generation.get("terminal_error") is not False
        or generation.get("metal_w8_complete_linear_layer_stacks") is not True
        or generation.get("metal_w8_full_attention_mlp_blocks") is not True
        or generation.get("metal_w8_lm_head") is not False
        or generation.get("final_output_finite_checks") is not True
        or generation.get("intermediate_host_finite_checks") is not False
    ):
        raise HarnessError(
            "Stack3 execution/full-attention Metal MLP mechanism/finite contract drifted"
        )
    aggregate_stacks = aggregate.get("stacks")
    generation_stacks = generation.get("stacks")
    if (
        not isinstance(aggregate_stacks, list)
        or len(aggregate_stacks) != 6
        or not isinstance(generation_stacks, list)
        or len(generation_stacks) != 6
    ):
        raise HarnessError("Stack3 execution must contain six stacks")
    for index, layer_indices in enumerate(STACK_LAYER_INDICES):
        aggregate_stack = aggregate_stacks[index]
        generation_stack = generation_stacks[index]
        if (
            not isinstance(aggregate_stack, dict)
            or not isinstance(generation_stack, dict)
            or not _json_exact_equal(
                aggregate_stack.get("layer_indices"), list(layer_indices)
            )
            or not _json_exact_equal(
                generation_stack.get("layer_indices"), list(layer_indices)
            )
            or not _json_exact_equal(
                aggregate_stack.get("prefill_seed_calls"), [1, 1, 1]
            )
            or not _json_exact_equal(
                generation_stack.get("prefill_seed_calls"), [1, 1, 1]
            )
            or not _json_exact_equal(
                aggregate_stack.get("final_output_finite_checks_per_decode"), 1
            )
            or not _json_exact_equal(
                aggregate_stack.get("intermediate_host_finite_checks_per_decode"),
                0,
            )
        ):
            raise HarnessError("Stack3 execution stack identity drifted")
        _require_exact_fields(
            aggregate_stack.get("execution"),
            expected_execution,
            label="Stack3 execution counters",
        )
        _require_exact_fields(
            generation_stack,
            expected_execution,
            label="Stack3 generation execution counters",
        )
    for field in ("full_attention_mlp_layers",):
        aggregate_full = aggregate.get(field)
        generation_full = generation.get(field)
        if (
            not isinstance(aggregate_full, list)
            or len(aggregate_full) != 6
            or not isinstance(generation_full, list)
            or len(generation_full) != 6
        ):
            raise HarnessError("Stack3 execution must contain six full-attention MLPs")
        for index, layer_index in enumerate(FULL_ATTENTION_LAYER_INDICES):
            expected = {"layer_index": layer_index, "decode_calls": decode_calls}
            _require_exact_fields(
                aggregate_full[index], expected, label="Stack3 aggregate MLP execution"
            )
            _require_exact_fields(
                generation_full[index],
                expected,
                label="Stack3 generation MLP execution",
            )


def validate_stack3_prefill_execution(receipt: dict) -> None:
    """Validate prefill seeds while proving that no decode transaction ran."""

    aggregate = receipt.get("prefill_aggregate_path_receipt")
    generation = receipt.get("prefill_generation_path_receipt")
    expected_execution = _expected_stack_execution(0)
    if not isinstance(aggregate, dict) or not isinstance(generation, dict):
        raise HarnessError("Stack3 prefill execution receipts are missing")
    if (
        aggregate.get("mechanism") != "metal-w8-linear-layer-stack3-v1"
        or aggregate.get("full_attention_mlp_mechanism") != "metal-w8-mlp-block-g64"
        or aggregate.get("terminal_error") is not False
        or generation.get("format")
        != "apxinf-qwen35-linear-layer-stacks-generation-path-v1"
        or generation.get("mechanism") != "metal-w8-linear-layer-stack3-v1"
        or generation.get("full_attention_mlp_mechanism") != "metal-w8-mlp-block-g64"
        or generation.get("terminal_error") is not False
        or generation.get("metal_w8_complete_linear_layer_stacks") is not True
        or generation.get("metal_w8_full_attention_mlp_blocks") is not True
        or generation.get("metal_w8_lm_head") is not False
        or generation.get("final_output_finite_checks") is not True
        or generation.get("intermediate_host_finite_checks") is not False
    ):
        raise HarnessError(
            "Stack3 prefill/full-attention Metal MLP mechanism/finite contract drifted"
        )
    aggregate_stacks = aggregate.get("stacks")
    generation_stacks = generation.get("stacks")
    if (
        not isinstance(aggregate_stacks, list)
        or len(aggregate_stacks) != 6
        or not isinstance(generation_stacks, list)
        or len(generation_stacks) != 6
    ):
        raise HarnessError("Stack3 prefill execution must contain six stacks")
    for index, layer_indices in enumerate(STACK_LAYER_INDICES):
        aggregate_stack = aggregate_stacks[index]
        generation_stack = generation_stacks[index]
        if (
            not isinstance(aggregate_stack, dict)
            or not isinstance(generation_stack, dict)
            or not _json_exact_equal(
                aggregate_stack.get("layer_indices"), list(layer_indices)
            )
            or not _json_exact_equal(
                generation_stack.get("layer_indices"), list(layer_indices)
            )
            or not _json_exact_equal(
                aggregate_stack.get("prefill_seed_calls"), [1, 1, 1]
            )
            or not _json_exact_equal(
                generation_stack.get("prefill_seed_calls"), [1, 1, 1]
            )
            or not _json_exact_equal(
                aggregate_stack.get("final_output_finite_checks_per_decode"), 1
            )
            or not _json_exact_equal(
                aggregate_stack.get("intermediate_host_finite_checks_per_decode"),
                0,
            )
        ):
            raise HarnessError("Stack3 prefill execution stack identity drifted")
        _require_exact_fields(
            aggregate_stack.get("execution"),
            expected_execution,
            label="Stack3 prefill execution counters",
        )
        _require_exact_fields(
            generation_stack,
            expected_execution,
            label="Stack3 prefill generation execution counters",
        )
    aggregate_full = aggregate.get("full_attention_mlp_layers")
    generation_full = generation.get("full_attention_mlp_layers")
    if (
        not isinstance(aggregate_full, list)
        or len(aggregate_full) != 6
        or not isinstance(generation_full, list)
        or len(generation_full) != 6
    ):
        raise HarnessError(
            "Stack3 prefill execution must contain six full-attention MLPs"
        )
    for index, layer_index in enumerate(FULL_ATTENTION_LAYER_INDICES):
        expected = {"layer_index": layer_index, "decode_calls": 0}
        _require_exact_fields(
            aggregate_full[index],
            expected,
            label="Stack3 prefill aggregate MLP execution",
        )
        _require_exact_fields(
            generation_full[index],
            expected,
            label="Stack3 prefill generation MLP execution",
        )


def _expected_end_custody(identity: dict) -> dict:
    try:
        custody = identity["custody"]
        sources = custody["sources"]
        return {
            "binary": custody["binary"],
            "gate": sources["gate"],
            "rust_and_bridge_sources": sources["rust_and_bridge_sources"],
            "source_closure": sources["closure"],
            "stack3_compiled_shader_sources": sources["stack3_compiled_shader_sources"],
            "verified_at_end": True,
        }
    except (KeyError, TypeError) as error:
        raise HarnessError("Stack3 receipt identity custody is incomplete") from error


def validate_stack3_candidate_receipt(receipt: dict, *, decode_calls: int) -> None:
    validate_stack3_path_checks(receipt.get("path_checks"))
    validate_stack3_ledger(receipt.get("aggregate_buffer_ledger"))
    _require_exact_fields(
        receipt.get("generation_path_contract"),
        {
            "binds_six_three_layer_stacks_and_six_full_attention_mlp_layers": True,
            "final_output_finite_checks": True,
            "intermediate_host_finite_checks": False,
            "schema": "apxinf-qwen35-linear-layer-stacks-generation-path-v1",
            "versioned_stack3_semantics": True,
        },
        label="Stack3 generation path contract",
    )
    validate_stack3_prefill_execution(receipt)
    validate_stack3_execution(receipt, decode_calls=decode_calls)
    schedule = receipt.get("schedule")
    _require_exact_fields(
        schedule,
        {
            "duplicate_mlp_execution": False,
            "full_attention_cpu_attention_metal_mlp_layers": list(
                FULL_ATTENTION_LAYER_INDICES
            ),
            "full_attention_layer_count": 6,
            "layers_per_stack": 3,
            "linear_attention_complete_layer_stacks": [
                list(indices) for indices in STACK_LAYER_INDICES
            ],
            "linear_layer_count": 18,
            "stack_count": 6,
            "total_layers": 24,
        },
        label="Stack3 schedule",
    )
    _require_exact_fields(
        receipt.get("per_stack_transaction_contract"),
        {
            "command_buffers": 1,
            "commits": 1,
            "compute_encoders": 3,
            "final_output_finite_checks": True,
            "intermediate_host_finite_checks": False,
            "state_commit_mask": 7,
            "state_commits": 3,
            "terminal_error": False,
            "waits": 1,
        },
        label="Stack3 per-stack transaction contract",
    )
    identity = receipt.get("identity")
    if not isinstance(identity, dict) or not _json_exact_equal(
        receipt.get("custody_end_verification"), _expected_end_custody(identity)
    ):
        raise HarnessError("Stack3 end custody verification drifted")


def validate_run_receipt(path: Path, *, variant: str, frozen: dict) -> dict:
    """Validate one timed CPU or Stack3 receipt against the frozen oracle."""

    if variant not in {"A", "B"}:
        raise HarnessError("run receipt variant must be A or B")
    try:
        identity = frozen["identity"]
        receipts = frozen["receipts"]
        receipt_records = frozen["receipt_records"]
        expected_tokens = receipts["cpu_free128"]["generated_token_ids"]
    except (KeyError, TypeError) as error:
        raise HarnessError("frozen Stack3 run inputs are incomplete") from error
    if (
        not isinstance(expected_tokens, list)
        or len(expected_tokens) != 128
        or any(type(token) is not int or token < 0 for token in expected_tokens)
    ):
        raise HarnessError("frozen CPU-free reference trajectory is invalid")
    record = BASE.direct_file_record(path, label=f"variant {variant} run receipt")
    receipt = BASE.load_json(path, label=f"variant {variant} run receipt")
    expected_key = "cpu_free128" if variant == "A" else "candidate_free128"
    expected_format, expected_mode = RECEIPT_IDENTITY[expected_key]
    if (
        receipt.get("format") != expected_format
        or receipt.get("mode") != expected_mode
        or receipt.get("passed") is not True
        or receipt.get("generated_tokens") != 128
    ):
        raise HarnessError(f"variant {variant} run receipt identity/status is invalid")
    if not _json_exact_equal(receipt.get("identity"), identity):
        raise HarnessError(f"variant {variant} run receipt custody identity drifted")
    if not _json_exact_equal(
        receipt.get("custody_end_verification"), _expected_end_custody(identity)
    ):
        raise HarnessError(f"variant {variant} run receipt end custody drifted")
    timing = receipt.get("timing")
    if (
        not isinstance(timing, dict)
        or not BASE._positive_number(timing.get("decode_mean_ms"))
        or not BASE._positive_number(timing.get("prefill_ms"))
    ):
        raise HarnessError(f"variant {variant} run timing is invalid")
    if variant == "A":
        if not _json_exact_equal(receipt.get("generated_token_ids"), expected_tokens):
            raise HarnessError(
                "CPU formal run trajectory drifted from the frozen oracle"
            )
        if receipt.get("path_receipt") is not None:
            raise HarnessError("CPU formal run unexpectedly reported a Metal path")
        path_valid = True
        ledger_valid = True
    else:
        if not _json_exact_equal(
            receipt.get("cpu_free_receipt"), receipt_records["cpu_free128"]
        ):
            raise HarnessError("candidate did not bind the frozen CPU-free reference")
        if (
            not _json_exact_equal(
                receipt.get("cpu_generated_token_ids"), expected_tokens
            )
            or not _json_exact_equal(
                receipt.get("metal_w8_all_linear_layer_stacks_v1_generated_token_ids"),
                expected_tokens,
            )
            or not _json_exact_equal(receipt.get("mismatches"), [])
            or receipt.get("first_mismatch") is not None
        ):
            raise HarnessError("candidate formal run trajectory drifted")
        if not _json_exact_equal(
            receipt.get("path_checks"),
            receipts["candidate_free128"].get("path_checks"),
        ):
            raise HarnessError("candidate formal run Stack3 path checks drifted")
        if not _json_exact_equal(
            receipt.get("aggregate_buffer_ledger"),
            receipts["candidate_free128"].get("aggregate_buffer_ledger"),
        ):
            raise HarnessError("candidate formal run Stack3 ledger drifted")
        validate_stack3_candidate_receipt(receipt, decode_calls=127)
        path_valid = True
        ledger_valid = True
    return {
        "variant": variant,
        "decode_mean_ms": float(timing["decode_mean_ms"]),
        "ttft_ms": float(timing["prefill_ms"]),
        "trajectory_sha256": BASE.canonical_json_sha256(expected_tokens),
        "path_valid": path_valid,
        "ledger_valid": ledger_valid,
        "custody_sha256": BASE.canonical_json_sha256(identity),
        "receipt": record,
    }


def build_gate_argv(*, frozen: dict, mode: str, output: Path) -> list[str]:
    """Build one shell-free gate command bound to the frozen Stack3 oracle."""

    try:
        custody = frozen["summary"]["custody"]
        binary_path = custody["binary"]["path"]
        model_path = custody["model_dir"]["path"]
        source_lock_path = custody["source_lock"]["path"]
        cpu_reference_path = frozen["receipt_records"]["cpu_free128"]["path"]
    except (KeyError, TypeError) as error:
        raise HarnessError("frozen Stack3 gate custody is incomplete") from error
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
        raise HarnessError("formal schedule requested an unapproved Stack3 gate mode")
    argv.extend(["--output", str(output)])
    return argv


class Stack3CampaignLane:
    """Lane adapter for the audited generic formal-campaign safety engine."""

    report_format = "apxinf-qwen35-stack3-formal-benchmark-v1"

    def __init__(self, frozen: dict):
        self.frozen = frozen
        try:
            self.report_identity = {
                "format": "apxinf-qwen35-stack3-formal-lane-identity-v1",
                "harness_custody": frozen["harness_custody"],
                "frozen_summary": {
                    "path": frozen["summary_path"],
                    "sha256": frozen["summary_sha256"],
                },
                "frozen_reference_oracle": frozen["receipt_records"]["cpu_free128"],
                "frozen_binary_sha256": frozen["expected_binary_sha256"],
            }
        except (KeyError, TypeError) as error:
            raise HarnessError("frozen Stack3 lane identity is incomplete") from error

    def prepare_campaign(self, frozen: dict) -> dict:
        if frozen is not self.frozen:
            raise HarnessError("Stack3 campaign adapter received a different freeze")
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
            raise HarnessError(
                "frozen Stack3 campaign inputs are incomplete"
            ) from error

    def validate_live_custody(self, summary: dict) -> dict:
        try:
            expected_record = self.frozen["receipt_records"]["cpu_free128"]
            expected_receipt = self.frozen["receipts"]["cpu_free128"]
            expected_binary = self.frozen["expected_binary_sha256"]
        except (KeyError, TypeError) as error:
            raise HarnessError("frozen Stack3 oracle custody is incomplete") from error
        observed_record = BASE.direct_file_record(
            Path(expected_record["path"]), label="frozen CPU-free reference"
        )
        if not _json_exact_equal(
            observed_record,
            {key: expected_record[key] for key in ("path", "size", "sha256")},
        ):
            raise HarnessError("frozen CPU-free reference receipt drifted")
        observed_receipt = BASE.load_json(
            Path(expected_record["path"]), label="frozen CPU-free reference"
        )
        if (
            not _json_exact_equal(observed_receipt, expected_receipt)
            or observed_receipt.get("passed") is not True
            or not _json_exact_equal(observed_receipt.get("generated_tokens"), 128)
            or not _json_exact_equal(
                observed_receipt.get("identity"), self.frozen.get("identity")
            )
            or not _json_exact_equal(
                observed_receipt.get("custody_end_verification"),
                _expected_end_custody(self.frozen["identity"]),
            )
        ):
            raise HarnessError(
                "frozen CPU-free reference receipt is no longer admitted"
            )
        return {
            "stack3_artifacts": validate_live_custody(
                summary, expected_binary_sha256=expected_binary
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
    """Run Stack3 through the single audited formal-campaign safety engine."""

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
        lane=Stack3CampaignLane(frozen),
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
            "/private/tmp/apxinf-qwen35-stack3-formal-not-started"
        )
        plan = {
            "format": "apxinf-qwen35-stack3-formal-plan-v1",
            "execution_started": False,
            "requires_explicit_execute": True,
            "frozen_summary_sha256": frozen["summary_sha256"],
            "frozen_binary_sha256": PINNED_BINARY_SHA256,
            "frozen_reference_oracle_sha256": frozen.get("receipt_records", {})
            .get("cpu_free128", {})
            .get("sha256"),
            "harness_custody": frozen["harness_custody"],
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


def _validate_direct_record(
    record: object,
    *,
    label: str,
    sha_key: str = "sha256",
    size_key: str = "size",
) -> dict:
    if not isinstance(record, dict):
        raise HarnessError(f"{label} custody is missing")
    declared_path = record.get("path")
    if not isinstance(declared_path, str) or not Path(declared_path).is_absolute():
        raise HarnessError(f"{label} custody path must be absolute")
    observed = BASE.direct_file_record(Path(declared_path), label=f"{label} custody")
    if observed["sha256"] != record.get(sha_key):
        raise HarnessError(f"{label} custody SHA-256 drifted")
    if observed["size"] != record.get(size_key):
        raise HarnessError(f"{label} custody size drifted")
    if (
        record.get("direct_regular_file") is not True
        or record.get("single_link") is not True
    ):
        raise HarnessError(f"{label} custody was not archived as a direct file")
    return observed


def validate_harness_custody() -> dict:
    """Rehash the wrapper and the audited base engine as direct files."""

    records = {}
    for key, path in (
        ("wrapper", Path(__file__).resolve(strict=True)),
        ("audited_base", BASE_HARNESS_PATH.resolve(strict=True)),
    ):
        records[key] = {
            **BASE.direct_file_record(path, label=f"{key} formal harness"),
            "direct_regular_file": True,
            "single_link": True,
        }
    return records


def validate_live_custody(
    summary: dict, *, expected_binary_sha256: str = PINNED_BINARY_SHA256
) -> dict:
    """Rehash the complete Stack3 binary/source/profile/model closure."""

    custody = summary.get("custody")
    if not isinstance(custody, dict):
        raise HarnessError("summary custody is missing")
    binary = _validate_direct_record(custody.get("binary"), label="binary")
    if binary["sha256"] != expected_binary_sha256:
        raise HarnessError("Stack3 binary SHA-256 does not match the pin")

    sources = custody.get("sources")
    if not isinstance(sources, dict):
        raise HarnessError("Stack3 source custody is missing")
    if set(sources) != {
        "closure",
        "gate",
        "rust_and_bridge_sources",
        "compiled_shader_sources",
    }:
        raise HarnessError("Stack3 summary source custody closure drifted")
    if sources.get("closure") != "stack3-direct-compile-inputs-v1":
        raise HarnessError("Stack3 source custody closure drifted")
    gate = _validate_direct_record(sources.get("gate"), label="gate source")
    rust_sources = sources.get("rust_and_bridge_sources")
    shader_sources = sources.get("compiled_shader_sources")
    if not isinstance(rust_sources, dict) or set(rust_sources) != RUST_SOURCE_KEYS:
        raise HarnessError("Stack3 Rust/bridge source closure drifted")
    if (
        not isinstance(shader_sources, dict)
        or set(shader_sources) != SHADER_SOURCE_KEYS
    ):
        raise HarnessError("Stack3 compiled shader source closure drifted")
    observed_rust = {
        name: _validate_direct_record(record, label=f"{name} source")
        for name, record in rust_sources.items()
    }
    observed_shaders = {
        name: _validate_direct_record(record, label=f"{name} source")
        for name, record in shader_sources.items()
    }
    profile = _validate_direct_record(
        custody.get("profile"),
        label="profile",
    )
    source_lock = _validate_direct_record(
        custody.get("source_lock"),
        label="source lock",
        sha_key="file_sha256",
    )

    model = custody.get("model_dir")
    if not isinstance(model, dict):
        raise HarnessError("model custody is missing")
    declared_model = model.get("path")
    if not isinstance(declared_model, str) or not Path(declared_model).is_absolute():
        raise HarnessError("model custody path must be absolute")
    model_path = Path(declared_model)
    try:
        model_entry = model_path.lstat()
    except OSError as error:
        raise HarnessError(f"model custody is unavailable: {error}") from error
    if stat.S_ISLNK(model_entry.st_mode) or not stat.S_ISDIR(model_entry.st_mode):
        raise HarnessError("model custody must be a direct directory")
    artifacts = model.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise HarnessError("model custody artifacts are missing")
    try:
        actual_names = {child.name for child in model_path.iterdir()}
    except OSError as error:
        raise HarnessError(f"model custody cannot be enumerated: {error}") from error
    if actual_names != set(artifacts):
        raise HarnessError("model custody closure drifted")
    if model.get("closure") != "exact-profile-artifacts-plus-safe-cache-v1":
        raise HarnessError("model custody closure contract drifted")
    if model.get("cache_present") is not False:
        raise HarnessError("formal benchmark requires the archived cache-free model")
    observed_artifacts = {}
    for name, record in artifacts.items():
        if Path(name).name != name or not isinstance(record, dict):
            raise HarnessError("model custody artifact name/record is invalid")
        declared = {**record, "path": str(model_path / name)}
        observed_artifacts[name] = _validate_direct_record(
            declared, label=f"model artifact {name}"
        )
    for key in (
        "same_identity_in_all_four_receipts",
        "same_end_verification_in_all_four_receipts",
        "start_and_end_binary_and_source_records_equal",
        "independent_live_rehash_matches_embedded_identity",
    ):
        if custody.get(key) is not True:
            raise HarnessError("Stack3 summary custody admission flags drifted")
    if model.get("exact_top_level_artifact_set_verified") is not True:
        raise HarnessError("Stack3 model artifact-set verification drifted")
    return {
        "binary": binary,
        "sources": {
            "gate": gate,
            "rust_and_bridge_sources": observed_rust,
            "stack3_compiled_shader_sources": observed_shaders,
        },
        "profile": profile,
        "source_lock": source_lock,
        "model_dir": {
            "path": str(model_path),
            "artifacts": observed_artifacts,
            "cache_present": False,
        },
    }


def _canonical_custody_record(
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


def canonical_summary_custody(custody: object) -> dict:
    if not isinstance(custody, dict):
        raise HarnessError("summary custody is missing")
    sources = custody.get("sources")
    model = custody.get("model_dir")
    if not isinstance(sources, dict) or not isinstance(model, dict):
        raise HarnessError("summary custody closure is incomplete")
    rust = sources.get("rust_and_bridge_sources")
    shaders = sources.get("compiled_shader_sources")
    artifacts = model.get("artifacts")
    if (
        not isinstance(rust, dict)
        or not isinstance(shaders, dict)
        or not isinstance(artifacts, dict)
    ):
        raise HarnessError("summary custody records are incomplete")
    model_path = model.get("path")
    if not isinstance(model_path, str) or not Path(model_path).is_absolute():
        raise HarnessError("summary model custody path is invalid")
    return {
        "binary": _canonical_custody_record(custody.get("binary"), label="binary"),
        "gate": _canonical_custody_record(sources.get("gate"), label="gate"),
        "rust": {
            key: _canonical_custody_record(value, label=key)
            for key, value in rust.items()
        },
        "shaders": {
            key: _canonical_custody_record(value, label=key)
            for key, value in shaders.items()
        },
        "profile": _canonical_custody_record(custody.get("profile"), label="profile"),
        "source_lock": _canonical_custody_record(
            custody.get("source_lock"), label="source lock", sha_key="file_sha256"
        ),
        "model": {
            "path": model_path,
            "artifacts": {
                name: _canonical_custody_record(
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
        raise HarnessError("receipt identity custody closure is incomplete")
    rust = sources.get("rust_and_bridge_sources")
    shaders = sources.get("stack3_compiled_shader_sources")
    artifacts = model.get("artifacts")
    if (
        sources.get("captured_at_start") is not True
        or sources.get("closure") != "stack3-direct-compile-inputs-v1"
        or not isinstance(rust, dict)
        or set(rust) != RUST_SOURCE_KEYS
        or not isinstance(shaders, dict)
        or set(shaders) != SHADER_SOURCE_KEYS
        or not isinstance(artifacts, dict)
    ):
        raise HarnessError("receipt identity Stack3 custody closure drifted")
    return {
        "binary": _canonical_custody_record(custody.get("binary"), label="binary"),
        "gate": _canonical_custody_record(sources.get("gate"), label="gate"),
        "rust": {
            key: _canonical_custody_record(value, label=key)
            for key, value in rust.items()
        },
        "shaders": {
            key: _canonical_custody_record(value, label=key)
            for key, value in shaders.items()
        },
        "profile": _canonical_custody_record(
            custody.get("profile"),
            label="profile",
            sha_key="file_sha256",
            size_key="file_size",
        ),
        "source_lock": _canonical_custody_record(
            custody.get("source_lock"),
            label="source lock",
            sha_key="file_sha256",
            size_key="file_size",
        ),
        "model": {
            "path": model.get("path"),
            "artifacts": {
                name: _canonical_custody_record(record, label=f"model artifact {name}")
                for name, record in artifacts.items()
            },
        },
    }


def validate_frozen_inputs(
    summary_path: Path,
    *,
    repo_root: Path,
    expected_summary_sha256: str = PINNED_SUMMARY_SHA256,
    expected_binary_sha256: str = PINNED_BINARY_SHA256,
) -> dict:
    """Bind formal work to exact archived Stack3 correctness evidence."""

    summary_record = BASE.direct_file_record(
        summary_path, label="frozen Stack3 summary"
    )
    observed = summary_record["sha256"]
    if observed != expected_summary_sha256:
        raise HarnessError("frozen Stack3 summary SHA-256 does not match the pin")
    summary = BASE.load_json(summary_path, label="frozen Stack3 summary")
    if summary.get("format") != SUMMARY_FORMAT:
        raise HarnessError("frozen Stack3 summary format is not admitted")
    integrity = summary.get("receipt_integrity")
    if not isinstance(integrity, dict):
        raise HarnessError("frozen Stack3 summary receipt integrity is missing")
    receipts = {}
    receipt_records = {}
    for key in RECEIPT_KEYS:
        record = integrity.get(key)
        if not isinstance(record, dict):
            raise HarnessError(f"frozen receipt record is missing: {key}")
        receipt_path = BASE.resolve_repo_file(
            repo_root, record.get("path"), label=f"{key} receipt"
        )
        observed_record = BASE.direct_file_record(receipt_path, label=f"{key} receipt")
        if observed_record["sha256"] != record.get("sha256"):
            raise HarnessError(f"{key} receipt SHA-256 does not match the summary")
        if not _json_exact_equal(observed_record["size"], record.get("size")):
            raise HarnessError(f"{key} receipt size does not match the summary")
        receipt = BASE.load_json(receipt_path, label=f"{key} receipt")
        expected_format, expected_mode = RECEIPT_IDENTITY[key]
        if (
            receipt.get("format") != expected_format
            or receipt.get("mode") != expected_mode
            or receipt.get("passed") is not True
        ):
            raise HarnessError(f"{key} receipt is not an admitted passing receipt")
        receipts[key] = receipt
        receipt_records[key] = {
            **observed_record,
            "direct_regular_file": True,
            "single_link": True,
        }
    identities = [receipts[key].get("identity") for key in RECEIPT_KEYS]
    if not isinstance(identities[0], dict) or any(
        not _json_exact_equal(identity, identities[0]) for identity in identities[1:]
    ):
        raise HarnessError("the four Stack3 receipt identities drift from one another")
    cpu_teacher = receipts["cpu_teacher128"]
    candidate_teacher = receipts["candidate_teacher128"]
    cpu_free = receipts["cpu_free128"]
    candidate_free = receipts["candidate_free128"]
    teacher_tokens = cpu_teacher.get("cpu_expected_output_ids")
    free_tokens = cpu_free.get("generated_token_ids")
    for label, tokens in (("teacher", teacher_tokens), ("free", free_tokens)):
        if (
            not isinstance(tokens, list)
            or len(tokens) != 128
            or any(type(token) is not int or token < 0 for token in tokens)
        ):
            raise HarnessError(f"frozen Stack3 {label} trajectory is invalid")
    if not _json_exact_equal(cpu_teacher.get("comparisons"), 128):
        raise HarnessError("frozen Stack3 CPU teacher comparison count drifted")
    if (
        not _json_exact_equal(
            candidate_teacher.get("cpu_teacher_receipt"),
            receipt_records["cpu_teacher128"],
        )
        or not _json_exact_equal(candidate_teacher.get("comparisons"), 128)
        or not _json_exact_equal(
            candidate_teacher.get("cpu_expected_output_ids"), teacher_tokens
        )
        or not _json_exact_equal(
            candidate_teacher.get(
                "metal_w8_all_linear_layer_stacks_v1_actual_output_ids"
            ),
            teacher_tokens,
        )
        or not _json_exact_equal(candidate_teacher.get("mismatches"), [])
        or candidate_teacher.get("first_mismatch") is not None
    ):
        raise HarnessError(
            "frozen Stack3 teacher reference/candidate trajectory drifted"
        )
    if (
        not _json_exact_equal(cpu_free.get("generated_tokens"), 128)
        or cpu_free.get("path_receipt") is not None
    ):
        raise HarnessError("frozen Stack3 CPU-free reference is invalid")
    if (
        not _json_exact_equal(
            candidate_free.get("cpu_free_receipt"), receipt_records["cpu_free128"]
        )
        or not _json_exact_equal(candidate_free.get("generated_tokens"), 128)
        or not _json_exact_equal(
            candidate_free.get("cpu_generated_token_ids"), free_tokens
        )
        or not _json_exact_equal(
            candidate_free.get(
                "metal_w8_all_linear_layer_stacks_v1_generated_token_ids"
            ),
            free_tokens,
        )
        or not _json_exact_equal(candidate_free.get("mismatches"), [])
        or candidate_free.get("first_mismatch") is not None
    ):
        raise HarnessError("frozen Stack3 free reference/candidate trajectory drifted")
    validate_stack3_candidate_receipt(candidate_teacher, decode_calls=128)
    validate_stack3_candidate_receipt(candidate_free, decode_calls=127)
    if canonical_summary_custody(summary.get("custody")) != canonical_identity_custody(
        identities[0].get("custody")
    ):
        raise HarnessError("summary and receipt custody identities drifted")
    trajectory = summary.get("trajectory_gate")
    gate = summary.get("gate_result")
    ledger = summary.get("aggregate_buffer_ledger")
    if not isinstance(trajectory, dict) or any(
        trajectory.get(key) is not True
        for key in (
            "all_four_receipts_passed",
            "all_required_candidate_prefill_and_decode_path_contract_checks_passed",
        )
    ):
        raise HarnessError("archived Stack3 trajectory/path gate is not passing")
    if not isinstance(gate, dict) or any(
        gate.get(key) is not True
        for key in (
            "correctness_and_path_gate_passed",
            "aggregate_ledger_valid",
            "custody_valid",
        )
    ):
        raise HarnessError("archived Stack3 correctness gate is not passing")
    if (
        not isinstance(ledger, dict)
        or ledger.get("independent_component_sum_matches_both_candidate_receipts")
        is not True
    ):
        raise HarnessError("archived Stack3 ledger gate is not passing")
    harness_custody = validate_harness_custody()
    live_custody = {
        "stack3_artifacts": validate_live_custody(
            summary, expected_binary_sha256=expected_binary_sha256
        ),
        "harness": harness_custody,
    }
    return {
        "summary_path": str(summary_path),
        "summary_sha256": observed,
        "summary": summary,
        "receipts": receipts,
        "receipt_records": receipt_records,
        "identity": identities[0],
        "live_custody": live_custody,
        "harness_custody": harness_custody,
        "expected_binary_sha256": expected_binary_sha256,
    }


if __name__ == "__main__":
    raise SystemExit(main())
