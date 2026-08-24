from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import importlib.util
import copy
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "scripts" / "run_qwen35_stack3_head_v2_formal_benchmark.py"


def load_module():
    spec = importlib.util.spec_from_file_location(
        "run_qwen35_stack3_head_v2_formal_benchmark_for_tests", MODULE_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")), encoding="utf-8"
    )


def direct_record(path: Path) -> dict:
    return {
        "path": str(path),
        "size": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "direct_regular_file": True,
        "single_link": True,
    }


STACK_INDICES = (
    [0, 1, 2],
    [4, 5, 6],
    [8, 9, 10],
    [12, 13, 14],
    [16, 17, 18],
    [20, 21, 22],
)
FULL_INDICES = (3, 7, 11, 15, 19, 23)


def composite_ledger() -> dict:
    stack = {
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
    mlp = {
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
    body = {
        "scope": "resident-mtlbuffer-only",
        "exclusions": "CPU F32 weights, host Vec allocations, Metal pipelines/libraries/queues, driver allocations, KV cache, and lm_head",
        "includes_lm_head": False,
        "stacks": [
            {"layer_indices": indices, "ledger": dict(stack)}
            for indices in STACK_INDICES
        ],
        "full_attention_mlp_layers": [
            {"layer_index": layer, "ledger": dict(mlp)} for layer in FULL_INDICES
        ],
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
    head = {
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
    return {
        "scope": "resident-mtlbuffer-only",
        "exclusions": "host F32 tied embedding and exact four-candidate F32 rerank, other CPU F32 weights, host allocations, Metal pipelines/libraries/queues, command objects, driver allocations, and KV cache",
        "includes_lm_head": True,
        "body": body,
        "lm_head": head,
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


def generation_receipt(
    body_calls: int = 127,
    *,
    head_prefill: int = 1,
    head_decode: int = 127,
    head_teacher: int = 0,
) -> dict:
    execution = {
        "block_elapsed_ns": 1,
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
    return {
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
        "stacks": [
            {"layer_indices": indices, **execution} for indices in STACK_INDICES
        ],
        "full_attention_mlp_layers": [
            {"layer_index": layer, "decode_calls": body_calls, "block_elapsed_ns": 1}
            for layer in FULL_INDICES
        ],
        "lm_head": {
            "mechanism": "metal-w8-top4-f32-rerank",
            "prefill_calls": head_prefill,
            "decode_calls": head_decode,
            "teacher_calls": head_teacher,
            "topk_elapsed_ns": 1,
            "rerank_elapsed_ns": 1,
        },
    }


def identity_fixture() -> dict:
    custody = {
        "binary": {"path": "/bin/gate", "size": 1, "sha256": "b" * 64},
        "sources": {
            "captured_at_start": True,
            "closure": "stack3-lm-head-v2-direct-compile-inputs-v1",
            "gate": {"path": "/src/gate", "size": 1, "sha256": "g" * 64},
            "rust_and_bridge_sources": {"general": {"sha256": "r" * 64}},
            "compiled_metal_shader_sources": {"metal_w8_head": {"sha256": "s" * 64}},
        },
    }
    return {"binary_path": "/bin/gate", "custody": custody}


def custody_end(identity: dict) -> dict:
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


def free_profile(*, candidate: bool) -> dict:
    tpot_ms = 8.0 if candidate else 10.0
    ttft_ms = 105.0 if candidate else 100.0
    total_ms = ttft_ms + 127 * tpot_ms
    return {
        "classification": (
            "candidate-only single pass under an uncontrolled host; never promotion evidence"
            if candidate
            else "CPU reference single-pass diagnostic timing only; never promotion evidence"
        ),
        "generation_total_latency_ms": total_ms,
        "generation_tps": 1_000.0 / tpot_ms,
        "harness_elapsed_ms": total_ms + 0.001,
        "input_tokens": 13,
        "output_tokens": 128,
        "setup": {
            "checkpoint_load_ms": 100.0,
            "model_construct_ms": 200.0,
            "timing_classification": "single-pass diagnostic timing only; never formal promotion evidence",
        },
        "tpot_ms": tpot_ms,
        "ttft_ms": ttft_ms,
    }


def cpu_free_receipt(identity: dict, tokens: list[int]) -> dict:
    return {
        "format": "apxinf-qwen35-metal-w8-stack3-head-v2-cpu-free-run-v1",
        "mode": "cpu_free_run",
        "identity": identity,
        "custody_end_verification": custody_end(identity),
        "prompt": "Hello",
        "prompt_token_ids": list(range(13)),
        "official_layer_schedule_valid": True,
        "max_new_tokens": 128,
        "eos_stopping": False,
        "generated_token_ids": list(tokens),
        "generation_path_contract": None,
        "profile": free_profile(candidate=False),
        "passed": True,
    }


def candidate_free_receipt(identity: dict, tokens: list[int], cpu_record: dict) -> dict:
    return {
        "format": "apxinf-qwen35-metal-w8-stack3-head-v2-free-run-gate-v1",
        "mode": "metal_w8_stack3_head_v2_free_run",
        "identity": identity,
        "input_receipt": cpu_record,
        "custody_end_verification": custody_end(identity),
        "prompt": "Hello",
        "prompt_token_ids": list(range(13)),
        "official_layer_schedule_valid": True,
        "max_new_tokens": 128,
        "eos_stopping": False,
        "cpu_expected_token_ids": list(tokens),
        "generated_token_ids": list(tokens),
        "mismatches": [],
        "exact_trajectory": True,
        "final_generation_path_receipt": generation_receipt(),
        "generation_path_contract": {
            "schema": "apxinf-qwen35-stack3-lm-head-generation-path-v2",
            "shared_generate_streaming": True,
            "binds_six_stack3_six_full_attention_mlp_and_tied_head": True,
            "body_decode_calls": 127,
            "head_prefill_calls": 1,
            "head_decode_calls": 127,
        },
        "aggregate_buffer_ledger": composite_ledger(),
        "path_checks": {
            "aggregate_ledger_valid": True,
            "all_valid": True,
            "full_attention_mlp_valid": True,
            "generation_receipt_valid": True,
            "head_execution_valid": True,
            "schedule_valid": True,
            "stack3_execution_valid": True,
            "stack3_mechanism_valid": True,
            "terminal_clear": True,
        },
        "profile": free_profile(candidate=True),
        "passed": True,
    }


def clean_quiet_sample() -> dict:
    return {
        "logical_cpus": 10,
        "load_1m": 1.0,
        "pages_throttled": 0,
        "swap_used_bytes": 7_000_000_000,
        "processes": [],
    }


def clean_preflight() -> dict:
    return {
        "passed": True,
        "problems": [],
        "final_swap_used_bytes": 7_000_000_000,
    }


def manual_frozen(root: Path, harness) -> dict:
    model_dir = root / "model"
    model_dir.mkdir()
    source_lock = root / "source-lock.json"
    source_lock.write_text("{}", encoding="utf-8")
    gate_binary = root / "v2-gate"
    gate_binary.write_bytes(b"fake-gate-never-executed")
    identity = identity_fixture()
    identity["binary_path"] = str(gate_binary)
    identity["custody"]["binary"]["path"] = str(gate_binary)
    identity["custody"]["model_dir"] = {"path": str(model_dir)}
    identity["custody"]["source_lock"] = {"path": str(source_lock)}
    tokens = list(range(128))
    reference_path = root / "cpu-reference.json"
    write_json(reference_path, cpu_free_receipt(identity, tokens))
    reference_record = direct_record(reference_path)
    receipts = {
        "cpu_free128": cpu_free_receipt(identity, tokens),
        "candidate_free128": candidate_free_receipt(identity, tokens, reference_record),
    }
    harness_custody = {
        "wrapper": {"sha256": "w" * 64},
        "audited_base": {"sha256": "a" * 64},
    }
    live = {
        "v2_artifacts": {"binary": {"sha256": "b" * 64}},
        "harness": harness_custody,
    }
    summary = {
        "custody": {
            "binary": {"path": str(gate_binary)},
            "model_dir": {"path": str(model_dir)},
            "source_lock": {"path": str(source_lock)},
        }
    }
    summary_path = root / "summary.json"
    write_json(summary_path, summary)
    summary_record = direct_record(summary_path)
    return {
        "summary_path": str(summary_path),
        "summary_sha256": summary_record["sha256"],
        "summary_record": summary_record,
        "summary": summary,
        "identity": identity,
        "receipts": receipts,
        "receipt_records": {"cpu_free128": reference_record},
        "harness_custody": harness_custody,
        "live_custody": live,
        "expected_binary_sha256": "b" * 64,
    }


class Stack3HeadV2FormalBenchmarkHarnessTests(unittest.TestCase):
    def test_plan_is_exactly_three_abba_plus_three_baab_with_twelve_per_lane(self):
        harness = load_module()

        plan = harness.build_schedule(Path("/private/tmp/formal-stack3-head-v2"))

        self.assertEqual(
            [block["order"] for block in plan["blocks"]],
            ["ABBA", "BAAB", "ABBA", "BAAB", "ABBA", "BAAB"],
        )
        runs = [run for block in plan["blocks"] for run in block["runs"]]
        self.assertEqual(len(runs), 24)
        self.assertEqual(sum(run["variant"] == "A" for run in runs), 12)
        self.assertEqual(sum(run["variant"] == "B" for run in runs), 12)
        self.assertEqual(
            {run["mode"] for run in runs},
            {"cpu-free", "stack3-head-v2-free"},
        )

    def test_candidate_command_binds_the_frozen_cpu_free_reference(self):
        harness = load_module()
        frozen = {
            "summary": {
                "custody": {
                    "binary": {"path": "/private/tmp/v2-gate"},
                    "model_dir": {"path": "/private/tmp/model"},
                    "source_lock": {"path": "/private/tmp/source-lock.json"},
                }
            },
            "receipt_records": {"cpu_free128": {"path": "/private/tmp/cpu-free.json"}},
        }
        output = Path("/private/tmp/candidate-output.json")

        argv = harness.build_gate_argv(
            frozen=frozen, mode=harness.CANDIDATE_MODE, output=output
        )

        self.assertEqual(argv[:2], ["/usr/bin/time", "-l"])
        self.assertEqual(argv[argv.index("--mode") + 1], "stack3-head-v2-free")
        self.assertEqual(
            argv[argv.index("--input-receipt") + 1],
            "/private/tmp/cpu-free.json",
        )
        self.assertEqual(argv[argv.index("--output") + 1], str(output))

    def test_composite_ledger_binds_every_stack_mlp_and_head_component(self):
        harness = load_module()
        ledger = composite_ledger()

        self.assertIs(harness.validate_composite_ledger(ledger), ledger)

        wrong_head = copy.deepcopy(ledger)
        wrong_head["lm_head"]["compute_encoders_per_call"] = 1
        with self.assertRaisesRegex(harness.HarnessError, "lm_head ledger"):
            harness.validate_composite_ledger(wrong_head)

        wrong_mlp = copy.deepcopy(ledger)
        wrong_mlp["body"]["full_attention_mlp_layers"][0]["ledger"][
            "packed_weight_bytes"
        ] -= 1
        with self.assertRaisesRegex(harness.HarnessError, "full-attention MLP ledger"):
            harness.validate_composite_ledger(wrong_mlp)

    def test_v2_generation_receipt_binds_six_stacks_six_mlps_and_head_phase(self):
        harness = load_module()
        receipt = generation_receipt()

        harness.validate_generation_receipt(
            receipt,
            body_calls=127,
            head_calls={"prefill_calls": 1, "decode_calls": 127, "teacher_calls": 0},
        )

        wrong_head = copy.deepcopy(receipt)
        wrong_head["lm_head"]["prefill_calls"] = 0
        with self.assertRaisesRegex(harness.HarnessError, "lm_head execution"):
            harness.validate_generation_receipt(
                wrong_head,
                body_calls=127,
                head_calls={
                    "prefill_calls": 1,
                    "decode_calls": 127,
                    "teacher_calls": 0,
                },
            )

        wrong_stack = copy.deepcopy(receipt)
        wrong_stack["stacks"][4]["compute_encoders"] = 380
        with self.assertRaisesRegex(harness.HarnessError, "Stack3 execution"):
            harness.validate_generation_receipt(
                wrong_stack,
                body_calls=127,
                head_calls={
                    "prefill_calls": 1,
                    "decode_calls": 127,
                    "teacher_calls": 0,
                },
            )

    def test_timed_a_and_b_receipts_each_match_the_frozen_cpu_oracle(self):
        harness = load_module()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            identity = identity_fixture()
            tokens = list(range(128))
            reference_path = root / "cpu-reference.json"
            write_json(reference_path, cpu_free_receipt(identity, tokens))
            reference_record = direct_record(reference_path)
            frozen = {
                "identity": identity,
                "receipts": {
                    "cpu_free128": cpu_free_receipt(identity, tokens),
                    "candidate_free128": candidate_free_receipt(
                        identity, tokens, reference_record
                    ),
                },
                "receipt_records": {"cpu_free128": reference_record},
            }
            cpu_path = root / "timed-a.json"
            candidate_path = root / "timed-b.json"
            write_json(cpu_path, cpu_free_receipt(identity, tokens))
            write_json(
                candidate_path,
                candidate_free_receipt(identity, tokens, reference_record),
            )

            sample_a = harness.validate_run_receipt(
                cpu_path, variant="A", frozen=frozen
            )
            sample_b = harness.validate_run_receipt(
                candidate_path, variant="B", frozen=frozen
            )

            self.assertEqual(
                sample_a["trajectory_sha256"], sample_b["trajectory_sha256"]
            )
            self.assertEqual(sample_a["decode_mean_ms"], 10.0)
            self.assertEqual(sample_b["decode_mean_ms"], 8.0)
            self.assertTrue(sample_b["path_valid"])
            self.assertTrue(sample_b["ledger_valid"])

            drifted = candidate_free_receipt(identity, tokens, reference_record)
            drifted["generated_token_ids"][91] = 999
            write_json(candidate_path, drifted)
            with self.assertRaisesRegex(harness.HarnessError, "trajectory"):
                harness.validate_run_receipt(candidate_path, variant="B", frozen=frozen)

            inconsistent = candidate_free_receipt(identity, tokens, reference_record)
            inconsistent["profile"]["tpot_ms"] = 7.0
            write_json(candidate_path, inconsistent)
            with self.assertRaisesRegex(harness.HarnessError, "timing profile"):
                harness.validate_run_receipt(candidate_path, variant="B", frozen=frozen)

    def test_frozen_archive_pins_summary_binary_and_all_four_v2_receipts(self):
        harness = load_module()
        expected_live = {"v2_artifacts": {"binary": {"sha256": "b" * 64}}}
        expected_harness = {
            "wrapper": {"sha256": "w" * 64},
            "audited_base": {"sha256": "a" * 64},
        }

        with mock.patch.object(
            harness, "validate_live_custody", return_value=expected_live
        ) as live_probe:
            with mock.patch.object(
                harness, "validate_harness_custody", return_value=expected_harness
            ):
                frozen = harness.validate_frozen_inputs(
                    harness.DEFAULT_SUMMARY,
                    repo_root=harness.REPO_ROOT,
                )

        self.assertEqual(
            frozen["summary_sha256"],
            "7aa07e1dd7beda066fa7c7048bfcbb0505b793c1caf43aac5f104c4a45177727",
        )
        self.assertEqual(
            frozen["expected_binary_sha256"],
            "0e70fe6589a77c78c79aa5071741eae27ae184b863b7a49adf47228f86ea1812",
        )
        self.assertEqual(set(frozen["receipts"]), set(harness.RECEIPT_KEYS))
        head = frozen["receipts"]["candidate_free128"]["final_generation_path_receipt"][
            "lm_head"
        ]
        self.assertEqual(head["mechanism"], "metal-w8-top4-f32-rerank")
        self.assertEqual(
            {
                key: head[key]
                for key in ("prefill_calls", "decode_calls", "teacher_calls")
            },
            {"prefill_calls": 1, "decode_calls": 127, "teacher_calls": 0},
        )
        self.assertEqual(
            frozen["live_custody"],
            {"v2_artifacts": expected_live, "harness": expected_harness},
        )
        live_probe.assert_called_once_with(
            frozen["summary"],
            expected_binary_sha256=frozen["expected_binary_sha256"],
            identity=frozen["identity"],
        )

    def test_fake_24_run_campaign_reuses_base_supervision_and_enforces_ttft(self):
        harness = load_module()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            frozen = manual_frozen(root, harness)

            def run_campaign(output_name: str, candidate_ttft_ms: float):
                calls = []

                def fake_runner(argv, **_kwargs):
                    calls.append(argv)
                    output = Path(argv[argv.index("--output") + 1])
                    mode = argv[argv.index("--mode") + 1]
                    if mode == harness.CPU_MODE:
                        receipt = cpu_free_receipt(frozen["identity"], list(range(128)))
                    else:
                        receipt = candidate_free_receipt(
                            frozen["identity"],
                            list(range(128)),
                            frozen["receipt_records"]["cpu_free128"],
                        )
                        receipt["profile"]["ttft_ms"] = candidate_ttft_ms
                        receipt["profile"]["generation_total_latency_ms"] = (
                            candidate_ttft_ms + 127 * receipt["profile"]["tpot_ms"]
                        )
                        receipt["profile"]["harness_elapsed_ms"] = (
                            receipt["profile"]["generation_total_latency_ms"] + 0.001
                        )
                    write_json(output, receipt)
                    return {
                        "argv": argv,
                        "returncode": 0,
                        "timed_out": False,
                        "termination_reason": None,
                        "peak_group_rss_bytes": 1_000_000_000,
                        "rss_limit_bytes": harness.BASE.RUN_RSS_LIMIT_BYTES,
                        "stdout": b"",
                        "stderr": b" 1000000000 maximum resident set size\n 0 swaps\n",
                        "owned_process_group": 88_888,
                        "quiet_samples": [clean_quiet_sample()],
                    }

                with mock.patch.object(
                    harness,
                    "validate_live_custody",
                    return_value=frozen["live_custody"]["v2_artifacts"],
                ):
                    with mock.patch.object(
                        harness,
                        "validate_harness_custody",
                        return_value=frozen["harness_custody"],
                    ):
                        report = harness.execute_campaign(
                            frozen=frozen,
                            repo_root=root,
                            output_dir=root / output_name,
                            quiet_probe=clean_preflight,
                            run_quiet_sample_probe=lambda _owned=None: (
                                clean_quiet_sample()
                            ),
                            command_runner=fake_runner,
                            swap_probe=lambda: 7_000_000_000,
                        )
                return report, calls

            accepted, calls = run_campaign("accepted", 105.0)
            rejected, _ = run_campaign("ttft-regression", 154.565289)

        self.assertTrue(accepted["formal_accepted"])
        self.assertEqual(accepted["reduction"]["same_direction_blocks"], 6)
        self.assertEqual(accepted["reduction"]["sample_count"], 24)
        self.assertEqual(len(calls), 24)
        self.assertEqual(
            sum(
                argv[argv.index("--mode") + 1] == harness.CANDIDATE_MODE
                for argv in calls
            ),
            12,
        )
        self.assertFalse(rejected["formal_accepted"])
        self.assertAlmostEqual(rejected["reduction"]["ttft_ratio"], 1.54565289)
        self.assertIn(
            "candidate TTFT regressed by more than 10%",
            rejected["reduction"]["problems"],
        )

    def test_cli_defaults_to_dry_plan_and_execute_requires_explicit_flag(self):
        harness = load_module()
        frozen = {
            "summary_sha256": harness.PINNED_SUMMARY_SHA256,
            "summary_record": {
                "path": "/tmp/summary.json",
                "size": 1,
                "sha256": harness.PINNED_SUMMARY_SHA256,
                "direct_regular_file": True,
                "single_link": True,
            },
            "expected_binary_sha256": harness.PINNED_BINARY_SHA256,
            "receipt_records": {
                "cpu_free128": {"sha256": "c" * 64, "path": "/tmp/cpu.json"}
            },
            "harness_custody": {
                "wrapper": {"sha256": "w" * 64},
                "audited_base": {"sha256": harness.PINNED_BASE_HARNESS_SHA256},
            },
        }
        stdout = io.StringIO()
        with mock.patch.object(harness, "validate_frozen_inputs", return_value=frozen):
            with mock.patch.object(
                harness, "execute_campaign", side_effect=AssertionError("must not run")
            ) as execute:
                with redirect_stdout(stdout):
                    returncode = harness.main([])

        plan = json.loads(stdout.getvalue())
        self.assertEqual(returncode, 0)
        execute.assert_not_called()
        self.assertEqual(plan["format"], "apxinf-qwen35-stack3-head-v2-formal-plan-v1")
        self.assertFalse(plan["execution_started"])
        self.assertTrue(plan["requires_explicit_execute"])
        self.assertEqual(plan["formal_contract"]["ttft_ratio_maximum"], 1.10)
        self.assertEqual(
            plan["formal_contract"]["process_group_rss_limit_bytes"], 6 * 1024**3
        )
        self.assertEqual(len(plan["schedule"]["blocks"]), 6)

    def test_cli_execute_flag_wires_v2_campaign_without_running_a_model(self):
        harness = load_module()
        frozen = {"summary_sha256": harness.PINNED_SUMMARY_SHA256}
        accepted = {
            "format": "apxinf-qwen35-stack3-head-v2-formal-benchmark-v1",
            "formal_accepted": True,
            "status": "formal_accepted",
        }
        stdout = io.StringIO()
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary).resolve() / "new-formal-output"
            with mock.patch.object(
                harness, "validate_frozen_inputs", return_value=frozen
            ):
                with mock.patch.object(
                    harness, "execute_campaign", return_value=accepted
                ) as execute:
                    with redirect_stdout(stdout):
                        returncode = harness.main(
                            ["--execute", "--output-dir", str(output)]
                        )

        self.assertEqual(returncode, 0)
        self.assertEqual(json.loads(stdout.getvalue()), accepted)
        execute.assert_called_once_with(
            frozen=frozen,
            repo_root=harness.REPO_ROOT,
            output_dir=output,
            quiet_probe=harness.BASE.quiet_host_preflight,
        )

    def test_post_freeze_source_drift_blocks_real_default_before_any_execution(self):
        harness = load_module()
        stdout = io.StringIO()
        stderr = io.StringIO()
        with mock.patch.object(
            harness, "execute_campaign", side_effect=AssertionError("must not execute")
        ) as execute:
            with redirect_stdout(stdout), redirect_stderr(stderr):
                returncode = harness.main([])

        self.assertEqual(returncode, 2)
        execute.assert_not_called()
        self.assertEqual(stdout.getvalue(), "")
        blocked = json.loads(stderr.getvalue())
        self.assertFalse(blocked["formal_accepted"])
        self.assertEqual(blocked["status"], "blocked")
        self.assertRegex(blocked["error"], r"custody SHA-256 drifted")

    def test_reference_drift_blocks_before_quiet_gate_child_or_output(self):
        harness = load_module()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            frozen = manual_frozen(root, harness)
            reference = Path(frozen["receipt_records"]["cpu_free128"]["path"])
            reference.write_bytes(reference.read_bytes() + b"\n")
            quiet_calls = []
            child_calls = []
            output = root / "formal-output"

            with self.assertRaisesRegex(
                harness.HarnessError, "frozen v2 CPU-free reference"
            ):
                harness.execute_campaign(
                    frozen=frozen,
                    repo_root=root,
                    output_dir=output,
                    quiet_probe=lambda: quiet_calls.append(True),
                    command_runner=lambda *args, **kwargs: child_calls.append(
                        (args, kwargs)
                    ),
                    swap_probe=lambda: 7_000_000_000,
                )

        self.assertEqual(quiet_calls, [])
        self.assertEqual(child_calls, [])
        self.assertFalse(output.exists())

    def test_summary_drift_blocks_before_quiet_gate_child_or_output(self):
        harness = load_module()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            frozen = manual_frozen(root, harness)
            Path(frozen["summary_path"]).write_bytes(
                Path(frozen["summary_path"]).read_bytes() + b"\n"
            )
            quiet_calls = []
            child_calls = []
            output = root / "formal-output"

            with self.assertRaisesRegex(harness.HarnessError, "frozen v2 summary"):
                harness.execute_campaign(
                    frozen=frozen,
                    repo_root=root,
                    output_dir=output,
                    quiet_probe=lambda: quiet_calls.append(True),
                    command_runner=lambda *args, **kwargs: child_calls.append(
                        (args, kwargs)
                    ),
                    swap_probe=lambda: 7_000_000_000,
                )

        self.assertEqual(quiet_calls, [])
        self.assertEqual(child_calls, [])
        self.assertFalse(output.exists())

    def test_output_is_rejected_inside_frozen_model_before_base_engine(self):
        harness = load_module()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            frozen = manual_frozen(root, harness)
            output = Path(frozen["summary"]["custody"]["model_dir"]["path"]) / "formal"
            with mock.patch.object(
                harness.BASE,
                "execute_campaign",
                side_effect=AssertionError("base must not be reached"),
            ) as base_execute:
                with self.assertRaisesRegex(harness.HarnessError, "outside the model"):
                    harness.execute_campaign(
                        frozen=frozen,
                        repo_root=root,
                        output_dir=output,
                        quiet_probe=clean_preflight,
                    )

        base_execute.assert_not_called()
        self.assertFalse(output.exists())

    def test_lane_rehashes_wrapper_and_base_at_campaign_start_and_end(self):
        harness = load_module()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            frozen = manual_frozen(root, harness)
            lane = harness.Stack3HeadV2CampaignLane(frozen)
            changed = copy.deepcopy(frozen["harness_custody"])
            changed["wrapper"]["sha256"] = "0" * 64
            with mock.patch.object(
                harness,
                "validate_live_custody",
                return_value=frozen["live_custody"]["v2_artifacts"],
            ):
                with mock.patch.object(
                    harness,
                    "validate_harness_custody",
                    side_effect=[frozen["harness_custody"], changed],
                ) as custody_probe:
                    start = lane.validate_live_custody(frozen["summary"])
                    end = lane.validate_live_custody(frozen["summary"])

        self.assertEqual(start, frozen["live_custody"])
        self.assertNotEqual(end, frozen["live_custody"])
        self.assertEqual(custody_probe.call_count, 2)
        self.assertEqual(
            lane.report_identity["harness_custody"], frozen["harness_custody"]
        )


if __name__ == "__main__":
    unittest.main()
