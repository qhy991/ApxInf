from __future__ import annotations

from contextlib import redirect_stdout
import importlib.util
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "scripts" / "run_qwen35_stack3_formal_benchmark.py"


def load_module():
    spec = importlib.util.spec_from_file_location(
        "run_qwen35_stack3_formal_benchmark_for_tests", MODULE_PATH
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


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")), encoding="utf-8"
    )


def direct_record(path: Path, *, sha_key: str = "sha256", size_key: str = "size"):
    return {
        "path": str(path),
        sha_key: sha256(path),
        size_key: path.stat().st_size,
        "direct_regular_file": True,
        "single_link": True,
    }


def stack3_ledger() -> dict:
    stack_indices = (
        [0, 1, 2],
        [4, 5, 6],
        [8, 9, 10],
        [12, 13, 14],
        [16, 17, 18],
        [20, 21, 22],
    )
    return {
        "scope": "resident-mtlbuffer-only",
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
        "final_output_finite_checks_per_decode": 6,
        "intermediate_host_finite_checks_per_decode": 0,
        "state_host_transfer_bytes_per_decode": 0,
        "includes_lm_head": False,
        "stacks": [
            {
                "layer_indices": list(indices),
                "ledger": {
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
                },
            }
            for indices in stack_indices
        ],
        "full_attention_mlp_layers": [
            {
                "layer_index": layer_index,
                "ledger": {
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
                },
            }
            for layer_index in (3, 7, 11, 15, 19, 23)
        ],
    }


def stack3_path_checks() -> dict:
    phase = {
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
    return {
        "decode": dict(phase),
        "decode_generation_receipt_valid": True,
        "exact_trajectory": True,
        "prefill": dict(phase),
        "prefill_generation_receipt_valid": True,
    }


def stack3_aggregate_path(decode_calls: int) -> dict:
    stack_indices = (
        [0, 1, 2],
        [4, 5, 6],
        [8, 9, 10],
        [12, 13, 14],
        [16, 17, 18],
        [20, 21, 22],
    )
    execution = {
        "command_buffers": decode_calls,
        "commits": decode_calls,
        "committed_stack_version": decode_calls,
        "compute_encoders": decode_calls * 3,
        "decode_calls": decode_calls,
        "device_to_host_bytes": decode_calls * 4096,
        "failed_decodes": 0,
        "host_to_device_bytes": decode_calls * 4096,
        "last_state_commit_mask": 7 if decode_calls else 0,
        "state_commits": decode_calls * 3,
        "successful_decodes": decode_calls,
        "terminal_error": False,
        "waits": decode_calls,
    }
    return {
        "mechanism": "metal-w8-linear-layer-stack3-v1",
        "full_attention_mlp_mechanism": "metal-w8-mlp-block-g64",
        "terminal_error": False,
        "stacks": [
            {
                "layer_indices": list(indices),
                "prefill_seed_calls": [1, 1, 1],
                "final_output_finite_checks_per_decode": 1,
                "intermediate_host_finite_checks_per_decode": 0,
                "execution": dict(execution),
            }
            for indices in stack_indices
        ],
        "full_attention_mlp_layers": [
            {"layer_index": layer_index, "decode_calls": decode_calls}
            for layer_index in (3, 7, 11, 15, 19, 23)
        ],
    }


def stack3_generation_path(decode_calls: int) -> dict:
    aggregate = stack3_aggregate_path(decode_calls)
    return {
        "format": "apxinf-qwen35-linear-layer-stacks-generation-path-v1",
        "mechanism": "metal-w8-linear-layer-stack3-v1",
        "full_attention_mlp_mechanism": "metal-w8-mlp-block-g64",
        "metal_w8_complete_linear_layer_stacks": True,
        "metal_w8_full_attention_mlp_blocks": True,
        "metal_w8_lm_head": False,
        "final_output_finite_checks": True,
        "intermediate_host_finite_checks": False,
        "terminal_error": False,
        "stacks": [
            {
                "layer_indices": stack["layer_indices"],
                "prefill_seed_calls": stack["prefill_seed_calls"],
                **stack["execution"],
            }
            for stack in aggregate["stacks"]
        ],
        "full_attention_mlp_layers": aggregate["full_attention_mlp_layers"],
    }


def custody_end_from(custody: dict) -> dict:
    sources = custody["sources"]
    return {
        "binary": custody["binary"],
        "gate": sources["gate"],
        "rust_and_bridge_sources": sources["rust_and_bridge_sources"],
        "source_closure": sources["closure"],
        "stack3_compiled_shader_sources": sources["stack3_compiled_shader_sources"],
        "verified_at_end": True,
    }


def stack3_schedule() -> dict:
    return {
        "duplicate_mlp_execution": False,
        "full_attention_cpu_attention_metal_mlp_layers": [3, 7, 11, 15, 19, 23],
        "full_attention_layer_count": 6,
        "layers_per_stack": 3,
        "linear_attention_complete_layer_stacks": [
            [0, 1, 2],
            [4, 5, 6],
            [8, 9, 10],
            [12, 13, 14],
            [16, 17, 18],
            [20, 21, 22],
        ],
        "linear_layer_count": 18,
        "stack_count": 6,
        "total_layers": 24,
    }


def stack3_transaction_contract() -> dict:
    return {
        "command_buffers": 1,
        "commits": 1,
        "compute_encoders": 3,
        "final_output_finite_checks": True,
        "intermediate_host_finite_checks": False,
        "state_commit_mask": 7,
        "state_commits": 3,
        "terminal_error": False,
        "waits": 1,
    }


def generation_path_contract() -> dict:
    return {
        "binds_six_three_layer_stacks_and_six_full_attention_mlp_layers": True,
        "final_output_finite_checks": True,
        "intermediate_host_finite_checks": False,
        "schema": "apxinf-qwen35-linear-layer-stacks-generation-path-v1",
        "versioned_stack3_semantics": True,
    }


def summary_custody_from(identity_custody: dict) -> dict:
    sources = identity_custody["sources"]
    profile = identity_custody["profile"]
    source_lock = identity_custody["source_lock"]
    model = identity_custody["model_dir"]
    return {
        "same_identity_in_all_four_receipts": True,
        "same_end_verification_in_all_four_receipts": True,
        "start_and_end_binary_and_source_records_equal": True,
        "independent_live_rehash_matches_embedded_identity": True,
        "binary": identity_custody["binary"],
        "sources": {
            "closure": sources["closure"],
            "gate": sources["gate"],
            "rust_and_bridge_sources": sources["rust_and_bridge_sources"],
            "compiled_shader_sources": sources["stack3_compiled_shader_sources"],
        },
        "profile": {
            "path": profile["path"],
            "profile_id": profile["profile_id"],
            "sha256": profile["file_sha256"],
            "size": profile["file_size"],
            "direct_regular_file": True,
            "single_link": True,
        },
        "source_lock": {
            "path": source_lock["path"],
            "file_sha256": source_lock["file_sha256"],
            "canonical_content_sha256_without_content_field": source_lock[
                "content_sha256"
            ],
            "size": source_lock["file_size"],
            "direct_regular_file": True,
            "single_link": True,
        },
        "model_dir": {
            "path": model["path"],
            "closure": model["closure"],
            "cache_present": False,
            "exact_top_level_artifact_set_verified": True,
            "artifacts": {
                name: {key: value for key, value in record.items() if key != "path"}
                for name, record in model["artifacts"].items()
            },
        },
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


def make_frozen_fixture(root: Path) -> tuple[Path, str, str]:
    root = root.resolve()
    binary = root / "bin" / "gate"
    gate = root / "src" / "gate.rs"
    profile = root / "config" / "profile.json"
    source_lock = root / "lock" / "source-lock.json"
    rust_source_names = (
        "apxinf_metal_build",
        "gate_evidence",
        "general",
        "stack3_bridge",
        "stack3_rust",
    )
    shader_source_names = (
        "metal_w8_gdn",
        "metal_w8_gdn_out_g32",
        "metal_w8_linear_layer",
        "metal_w8_mlp",
    )
    for path, payload in (
        (binary, b"stack3-gate-binary"),
        (gate, b"gate-source"),
        (profile, b"profile"),
        (source_lock, b"source-lock"),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    rust_sources = {}
    for name in rust_source_names:
        path = root / "src" / f"{name}.src"
        path.write_bytes(name.encode("ascii"))
        rust_sources[name] = direct_record(path)
    shader_sources = {}
    for name in shader_source_names:
        path = root / "src" / f"{name}.metal"
        path.write_bytes(name.encode("ascii"))
        shader_sources[name] = direct_record(path)

    model_dir = root / "model"
    model_dir.mkdir()
    artifacts = {}
    for name in ("config.json", "tokenizer.json", "model.safetensors"):
        path = model_dir / name
        path.write_bytes(("fixture-" + name).encode("ascii"))
        artifacts[name] = direct_record(path)

    custody = {
        "binary": direct_record(binary),
        "sources": {
            "captured_at_start": True,
            "closure": "stack3-direct-compile-inputs-v1",
            "gate": direct_record(gate),
            "rust_and_bridge_sources": rust_sources,
            "stack3_compiled_shader_sources": shader_sources,
        },
        "profile": {
            **direct_record(profile, sha_key="file_sha256", size_key="file_size"),
            "profile_id": "qwen35-0.8b-macos-cpu",
        },
        "source_lock": {
            **direct_record(source_lock, sha_key="file_sha256", size_key="file_size"),
            "content_sha256": "a" * 64,
        },
        "model_dir": {
            "path": str(model_dir),
            "closure": "exact-profile-artifacts-plus-safe-cache-v1",
            "cache_present": False,
            "artifacts": artifacts,
        },
    }
    identity = {
        "binary_path": str(binary),
        "build_profile": "release",
        "model_dir": str(model_dir),
        "source_lock": str(source_lock),
        "custody": custody,
    }
    specs = {
        "cpu_teacher128": (
            "cpu-teacher.json",
            "apxinf-qwen35-metal-w8-linear-layer-cpu-teacher-v1",
            "linear_layer_cpu_teacher",
        ),
        "candidate_teacher128": (
            "candidate-teacher.json",
            "apxinf-qwen35-metal-w8-all-linear-layer-stacks-v1-teacher-gate-v1",
            "metal_w8_all_linear_layer_stacks_v1_teacher_forced",
        ),
        "cpu_free128": (
            "cpu-free.json",
            "apxinf-qwen35-metal-w8-linear-layer-cpu-free-run-v1",
            "linear_layer_cpu_free_run",
        ),
        "candidate_free128": (
            "candidate-free.json",
            "apxinf-qwen35-metal-w8-all-linear-layer-stacks-v1-free-run-gate-v1",
            "metal_w8_all_linear_layer_stacks_v1_free_run",
        ),
    }
    receipt_integrity = {}
    for key, (name, format_name, mode) in specs.items():
        path = root / "receipts" / name
        payload = {
            "format": format_name,
            "mode": mode,
            "passed": True,
            "identity": identity,
            "custody_end_verification": custody_end_from(custody),
        }
        if key == "cpu_free128":
            payload.update(
                {
                    "generated_tokens": 128,
                    "generated_token_ids": list(range(128)),
                    "path_receipt": None,
                    "timing": {"decode_mean_ms": 10.0, "prefill_ms": 100.0},
                }
            )
        elif key == "cpu_teacher128":
            payload.update(
                {"comparisons": 128, "cpu_expected_output_ids": list(range(128))}
            )
        elif key == "candidate_teacher128":
            payload.update(
                {
                    "comparisons": 128,
                    "cpu_expected_output_ids": list(range(128)),
                    "metal_w8_all_linear_layer_stacks_v1_actual_output_ids": list(
                        range(128)
                    ),
                    "mismatches": [],
                    "first_mismatch": None,
                    "path_checks": stack3_path_checks(),
                    "aggregate_buffer_ledger": stack3_ledger(),
                    "final_aggregate_path_receipt": stack3_aggregate_path(128),
                    "final_generation_path_receipt": stack3_generation_path(128),
                    "prefill_aggregate_path_receipt": stack3_aggregate_path(0),
                    "prefill_generation_path_receipt": stack3_generation_path(0),
                    "custody_end_verification": custody_end_from(custody),
                    "schedule": stack3_schedule(),
                    "per_stack_transaction_contract": stack3_transaction_contract(),
                    "generation_path_contract": generation_path_contract(),
                }
            )
        write_json(path, payload)
        receipt_integrity[key] = {
            **direct_record(path),
            "path": str(path.relative_to(root)),
        }
    cpu_free_path = root / "receipts" / "cpu-free.json"
    candidate_free_path = root / "receipts" / "candidate-free.json"
    candidate_free = json.loads(candidate_free_path.read_text(encoding="utf-8"))
    cpu_teacher_path = root / "receipts" / "cpu-teacher.json"
    candidate_teacher_path = root / "receipts" / "candidate-teacher.json"
    candidate_teacher = json.loads(candidate_teacher_path.read_text(encoding="utf-8"))
    candidate_teacher["cpu_teacher_receipt"] = direct_record(cpu_teacher_path)
    write_json(candidate_teacher_path, candidate_teacher)
    receipt_integrity["candidate_teacher128"] = {
        **direct_record(candidate_teacher_path),
        "path": str(candidate_teacher_path.relative_to(root)),
    }
    candidate_free.update(
        {
            "generated_tokens": 128,
            "cpu_free_receipt": direct_record(cpu_free_path),
            "cpu_generated_token_ids": list(range(128)),
            "metal_w8_all_linear_layer_stacks_v1_generated_token_ids": list(range(128)),
            "mismatches": [],
            "first_mismatch": None,
            "path_checks": stack3_path_checks(),
            "aggregate_buffer_ledger": stack3_ledger(),
            "final_aggregate_path_receipt": stack3_aggregate_path(127),
            "final_generation_path_receipt": stack3_generation_path(127),
            "prefill_aggregate_path_receipt": stack3_aggregate_path(0),
            "prefill_generation_path_receipt": stack3_generation_path(0),
            "custody_end_verification": custody_end_from(custody),
            "schedule": stack3_schedule(),
            "per_stack_transaction_contract": stack3_transaction_contract(),
            "generation_path_contract": generation_path_contract(),
            "timing": {"decode_mean_ms": 8.0, "prefill_ms": 105.0},
        }
    )
    write_json(candidate_free_path, candidate_free)
    receipt_integrity["candidate_free128"] = {
        **direct_record(candidate_free_path),
        "path": str(candidate_free_path.relative_to(root)),
    }

    summary = root / "summary.json"
    write_json(
        summary,
        {
            "format": (
                "apxinf-qwen35-metal-w8-all-linear-layer-stacks-v1-real-gate-summary-v1"
            ),
            "receipt_integrity": receipt_integrity,
            "trajectory_gate": {
                "all_four_receipts_passed": True,
                "all_required_candidate_prefill_and_decode_path_contract_checks_passed": True,
                "teacher_forced_exact_128": True,
                "free_run_exact_128": True,
            },
            "custody": summary_custody_from(custody),
            "aggregate_buffer_ledger": {
                "independent_component_sum_matches_both_candidate_receipts": True,
                "aggregate": stack3_ledger(),
            },
            "gate_result": {
                "correctness_and_path_gate_passed": True,
                "aggregate_ledger_valid": True,
                "custody_valid": True,
            },
        },
    )
    return summary, sha256(summary), sha256(binary)


class Stack3FormalBenchmarkHarnessTests(unittest.TestCase):
    def test_plan_is_fixed_to_three_abba_and_three_baab_blocks(self):
        harness = load_module()

        plan = harness.build_schedule(Path("/private/tmp/formal-stack3"))

        self.assertEqual(
            [block["order"] for block in plan["blocks"]],
            ["ABBA", "BAAB", "ABBA", "BAAB", "ABBA", "BAAB"],
        )
        self.assertEqual(sum(len(block["runs"]) for block in plan["blocks"]), 24)
        self.assertEqual(
            {run["mode"] for block in plan["blocks"] for run in block["runs"]},
            {"cpu-free", "all-linear-layer-stacks-v1-free"},
        )
        self.assertEqual(
            sum(
                run["variant"] == "A"
                for block in plan["blocks"]
                for run in block["runs"]
            ),
            12,
        )
        self.assertEqual(
            sum(
                run["variant"] == "B"
                for block in plan["blocks"]
                for run in block["runs"]
            ),
            12,
        )

    def test_frozen_input_loader_pins_summary_binary_and_four_receipts(self):
        harness = load_module()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            summary, summary_sha, binary_sha = make_frozen_fixture(root)

            frozen = harness.validate_frozen_inputs(
                summary,
                repo_root=root,
                expected_summary_sha256=summary_sha,
                expected_binary_sha256=binary_sha,
            )

            self.assertEqual(frozen["summary_sha256"], summary_sha)
            self.assertEqual(
                frozen["receipt_records"]["cpu_free128"]["sha256"],
                sha256(root / "receipts" / "cpu-free.json"),
            )
            self.assertEqual(
                frozen["identity"]["custody"]["binary"]["sha256"], binary_sha
            )
            self.assertEqual(
                frozen["harness_custody"]["wrapper"]["sha256"],
                sha256(MODULE_PATH),
            )
            self.assertEqual(
                frozen["harness_custody"]["audited_base"]["sha256"],
                sha256(ROOT / "scripts" / "run_qwen35_all18_formal_benchmark.py"),
            )
            self.assertTrue(frozen["harness_custody"]["wrapper"]["direct_regular_file"])

            summary.write_bytes(summary.read_bytes() + b"\n")
            with self.assertRaisesRegex(harness.HarnessError, "summary SHA-256"):
                harness.validate_frozen_inputs(
                    summary,
                    repo_root=root,
                    expected_summary_sha256=summary_sha,
                    expected_binary_sha256=binary_sha,
                )

    def test_frozen_reference_oracle_rejects_stack_execution_or_receipt_drift(self):
        harness = load_module()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            summary, _summary_sha, binary_sha = make_frozen_fixture(root)
            candidate_path = root / "receipts" / "candidate-free.json"
            candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
            candidate["final_aggregate_path_receipt"]["stacks"][0]["execution"][
                "compute_encoders"
            ] = 380
            write_json(candidate_path, candidate)
            summary_value = json.loads(summary.read_text(encoding="utf-8"))
            summary_value["receipt_integrity"]["candidate_free128"].update(
                {
                    "sha256": sha256(candidate_path),
                    "size": candidate_path.stat().st_size,
                }
            )
            write_json(summary, summary_value)

            with self.assertRaisesRegex(harness.HarnessError, "Stack3 execution"):
                harness.validate_frozen_inputs(
                    summary,
                    repo_root=root,
                    expected_summary_sha256=sha256(summary),
                    expected_binary_sha256=binary_sha,
                )

    def test_timed_receipts_must_match_frozen_reference_stack_path_and_custody(self):
        harness = load_module()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            summary, summary_sha, binary_sha = make_frozen_fixture(root)
            frozen = harness.validate_frozen_inputs(
                summary,
                repo_root=root,
                expected_summary_sha256=summary_sha,
                expected_binary_sha256=binary_sha,
            )
            cpu_run = root / "timed-a.json"
            write_json(
                cpu_run,
                {
                    "format": harness.RECEIPT_IDENTITY["cpu_free128"][0],
                    "mode": harness.RECEIPT_IDENTITY["cpu_free128"][1],
                    "passed": True,
                    "generated_tokens": 128,
                    "generated_token_ids": list(range(128)),
                    "path_receipt": None,
                    "identity": frozen["identity"],
                    "custody_end_verification": custody_end_from(
                        frozen["identity"]["custody"]
                    ),
                    "timing": {"decode_mean_ms": 10.0, "prefill_ms": 100.0},
                },
            )
            candidate_run = root / "timed-b.json"
            candidate = json.loads(
                (root / "receipts" / "candidate-free.json").read_text(encoding="utf-8")
            )
            candidate["timing"] = {"decode_mean_ms": 8.0, "prefill_ms": 105.0}
            write_json(candidate_run, candidate)

            sample_a = harness.validate_run_receipt(cpu_run, variant="A", frozen=frozen)
            sample_b = harness.validate_run_receipt(
                candidate_run, variant="B", frozen=frozen
            )

            self.assertEqual(
                sample_a["trajectory_sha256"], sample_b["trajectory_sha256"]
            )
            self.assertTrue(sample_b["path_valid"])
            self.assertTrue(sample_b["ledger_valid"])

            candidate["cpu_free_receipt"]["sha256"] = "0" * 64
            write_json(candidate_run, candidate)
            with self.assertRaisesRegex(
                harness.HarnessError, "frozen CPU-free reference"
            ):
                harness.validate_run_receipt(candidate_run, variant="B", frozen=frozen)

    def test_candidate_command_binds_the_frozen_cpu_free_reference(self):
        harness = load_module()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            summary, summary_sha, binary_sha = make_frozen_fixture(root)
            frozen = harness.validate_frozen_inputs(
                summary,
                repo_root=root,
                expected_summary_sha256=summary_sha,
                expected_binary_sha256=binary_sha,
            )
            output = root / "candidate-output.json"

            argv = harness.build_gate_argv(
                frozen=frozen, mode=harness.CANDIDATE_MODE, output=output
            )

            self.assertEqual(argv[:2], ["/usr/bin/time", "-l"])
            self.assertEqual(
                argv[argv.index("--input-receipt") + 1],
                frozen["receipt_records"]["cpu_free128"]["path"],
            )
            self.assertEqual(argv[argv.index("--mode") + 1], harness.CANDIDATE_MODE)
            self.assertEqual(argv[argv.index("--output") + 1], str(output))

    def test_fake_campaign_reuses_audited_safety_engine_and_publishes_24_runs(self):
        harness = load_module()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            summary, summary_sha, binary_sha = make_frozen_fixture(root)
            frozen = harness.validate_frozen_inputs(
                summary,
                repo_root=root,
                expected_summary_sha256=summary_sha,
                expected_binary_sha256=binary_sha,
            )
            output_dir = root / "formal-output"
            calls = []

            def fake_runner(argv, **_kwargs):
                calls.append(argv)
                output = Path(argv[argv.index("--output") + 1])
                mode = argv[argv.index("--mode") + 1]
                if mode == harness.CPU_MODE:
                    write_json(
                        output,
                        {
                            "format": harness.RECEIPT_IDENTITY["cpu_free128"][0],
                            "mode": harness.RECEIPT_IDENTITY["cpu_free128"][1],
                            "passed": True,
                            "generated_tokens": 128,
                            "generated_token_ids": list(range(128)),
                            "path_receipt": None,
                            "identity": frozen["identity"],
                            "custody_end_verification": custody_end_from(
                                frozen["identity"]["custody"]
                            ),
                            "timing": {
                                "decode_mean_ms": 10.0,
                                "prefill_ms": 100.0,
                            },
                        },
                    )
                else:
                    candidate = json.loads(
                        (root / "receipts" / "candidate-free.json").read_text(
                            encoding="utf-8"
                        )
                    )
                    candidate["timing"] = {
                        "decode_mean_ms": 8.0,
                        "prefill_ms": 105.0,
                    }
                    write_json(output, candidate)
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

            report = harness.execute_campaign(
                frozen=frozen,
                repo_root=root,
                output_dir=output_dir,
                quiet_probe=clean_preflight,
                run_quiet_sample_probe=lambda _owned_group=None: clean_quiet_sample(),
                command_runner=fake_runner,
                swap_probe=lambda: 7_000_000_000,
            )

            self.assertTrue(report["formal_accepted"])
            self.assertEqual(
                report["format"], "apxinf-qwen35-stack3-formal-benchmark-v1"
            )
            self.assertEqual(
                report["lane_identity"]["harness_custody"],
                frozen["harness_custody"],
            )
            self.assertEqual(len(calls), 24)
            self.assertEqual(len(list(output_dir.glob("block-*.json"))), 24)
            candidate_calls = [
                argv
                for argv in calls
                if argv[argv.index("--mode") + 1] == harness.CANDIDATE_MODE
            ]
            self.assertEqual(len(candidate_calls), 12)
            self.assertTrue(
                all(
                    argv[argv.index("--input-receipt") + 1]
                    == frozen["receipt_records"]["cpu_free128"]["path"]
                    for argv in candidate_calls
                )
            )

    def test_cli_defaults_to_a_dry_plan_and_never_executes(self):
        harness = load_module()
        execute = mock.Mock(side_effect=AssertionError("must not execute"))
        harness_custody = {
            "wrapper": {"sha256": "w" * 64},
            "audited_base": {"sha256": "b" * 64},
        }
        frozen = {
            "summary_sha256": harness.PINNED_SUMMARY_SHA256,
            "harness_custody": harness_custody,
        }
        stdout = io.StringIO()

        with mock.patch.object(harness, "validate_frozen_inputs", return_value=frozen):
            with mock.patch.object(harness, "execute_campaign", execute):
                with redirect_stdout(stdout):
                    returncode = harness.main([])

        self.assertEqual(returncode, 0)
        execute.assert_not_called()
        plan = json.loads(stdout.getvalue())
        self.assertEqual(plan["format"], "apxinf-qwen35-stack3-formal-plan-v1")
        self.assertFalse(plan["execution_started"])
        self.assertTrue(plan["requires_explicit_execute"])
        self.assertEqual(plan["harness_custody"], harness_custody)
        self.assertEqual(len(plan["schedule"]["blocks"]), 6)

    def test_cli_execute_flag_wires_the_committed_campaign_entrypoint(self):
        harness = load_module()
        frozen = {"summary_sha256": harness.PINNED_SUMMARY_SHA256}
        accepted = {
            "format": "apxinf-qwen35-stack3-formal-benchmark-v1",
            "formal_accepted": True,
            "status": "formal_accepted",
        }
        stdout = io.StringIO()
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary).resolve() / "new-formal-output"
            with mock.patch.object(
                harness, "validate_frozen_inputs", return_value=frozen
            ):
                with mock.patch.object(
                    harness, "execute_campaign", return_value=accepted
                ) as execute:
                    with redirect_stdout(stdout):
                        returncode = harness.main(
                            ["--execute", "--output-dir", str(output_dir)]
                        )

        self.assertEqual(returncode, 0)
        execute.assert_called_once_with(
            frozen=frozen,
            repo_root=harness.REPO_ROOT,
            output_dir=output_dir,
            quiet_probe=harness.BASE.quiet_host_preflight,
        )
        self.assertEqual(json.loads(stdout.getvalue()), accepted)

    def test_reference_drift_blocks_before_quiet_gate_child_or_output(self):
        harness = load_module()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            summary, summary_sha, binary_sha = make_frozen_fixture(root)
            frozen = harness.validate_frozen_inputs(
                summary,
                repo_root=root,
                expected_summary_sha256=summary_sha,
                expected_binary_sha256=binary_sha,
            )
            cpu_reference = root / "receipts" / "cpu-free.json"
            cpu_reference.write_bytes(cpu_reference.read_bytes() + b"\n")
            quiet_calls = []
            child_calls = []
            output_dir = root / "formal-output"

            with self.assertRaisesRegex(
                harness.HarnessError, "frozen CPU-free reference"
            ):
                harness.execute_campaign(
                    frozen=frozen,
                    repo_root=root,
                    output_dir=output_dir,
                    quiet_probe=lambda: quiet_calls.append(True),
                    command_runner=lambda *args, **kwargs: child_calls.append(
                        (args, kwargs)
                    ),
                    swap_probe=lambda: 7_000_000_000,
                )

            self.assertEqual(quiet_calls, [])
            self.assertEqual(child_calls, [])
            self.assertFalse(output_dir.exists())

    def test_boolean_cannot_impersonate_an_integer_in_stack_contract(self):
        harness = load_module()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            summary, summary_sha, binary_sha = make_frozen_fixture(root)
            frozen = harness.validate_frozen_inputs(
                summary,
                repo_root=root,
                expected_summary_sha256=summary_sha,
                expected_binary_sha256=binary_sha,
            )
            candidate_path = root / "timed-b.json"
            candidate = json.loads(
                (root / "receipts" / "candidate-free.json").read_text(encoding="utf-8")
            )
            candidate["per_stack_transaction_contract"]["command_buffers"] = True
            write_json(candidate_path, candidate)

            with self.assertRaisesRegex(
                harness.HarnessError, "per-stack transaction contract"
            ):
                harness.validate_run_receipt(candidate_path, variant="B", frozen=frozen)

    def test_candidate_prefill_stack_and_full_mlp_receipts_are_not_trusted_flags(self):
        harness = load_module()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            summary, summary_sha, binary_sha = make_frozen_fixture(root)
            frozen = harness.validate_frozen_inputs(
                summary,
                repo_root=root,
                expected_summary_sha256=summary_sha,
                expected_binary_sha256=binary_sha,
            )
            candidate_path = root / "timed-b.json"
            candidate = json.loads(
                (root / "receipts" / "candidate-free.json").read_text(encoding="utf-8")
            )
            candidate["prefill_aggregate_path_receipt"]["stacks"][0]["execution"][
                "command_buffers"
            ] = 1
            write_json(candidate_path, candidate)

            with self.assertRaisesRegex(harness.HarnessError, "prefill execution"):
                harness.validate_run_receipt(candidate_path, variant="B", frozen=frozen)

    def test_formal_output_cannot_be_created_inside_the_frozen_model(self):
        harness = load_module()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            summary, summary_sha, binary_sha = make_frozen_fixture(root)
            frozen = harness.validate_frozen_inputs(
                summary,
                repo_root=root,
                expected_summary_sha256=summary_sha,
                expected_binary_sha256=binary_sha,
            )
            output_dir = root / "model" / "formal-output"
            quiet_calls = []
            child_calls = []

            with self.assertRaisesRegex(harness.HarnessError, "outside the model"):
                harness.execute_campaign(
                    frozen=frozen,
                    repo_root=root,
                    output_dir=output_dir,
                    quiet_probe=lambda: quiet_calls.append(True),
                    command_runner=lambda *args, **kwargs: child_calls.append(
                        (args, kwargs)
                    ),
                    swap_probe=lambda: 7_000_000_000,
                )

            self.assertEqual(quiet_calls, [])
            self.assertEqual(child_calls, [])
            self.assertFalse(output_dir.exists())

    def test_full_attention_mlp_mechanism_and_generation_contract_are_binding(self):
        harness = load_module()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            summary, summary_sha, binary_sha = make_frozen_fixture(root)
            frozen = harness.validate_frozen_inputs(
                summary,
                repo_root=root,
                expected_summary_sha256=summary_sha,
                expected_binary_sha256=binary_sha,
            )
            frozen_candidate = json.loads(
                (root / "receipts" / "candidate-free.json").read_text(encoding="utf-8")
            )
            candidate_path = root / "timed-b-mechanism.json"
            candidate = json.loads(json.dumps(frozen_candidate))
            for field in (
                "prefill_aggregate_path_receipt",
                "prefill_generation_path_receipt",
                "final_aggregate_path_receipt",
                "final_generation_path_receipt",
            ):
                candidate[field]["full_attention_mlp_mechanism"] = "cpu-f32"
            write_json(candidate_path, candidate)

            with self.assertRaisesRegex(
                harness.HarnessError, "full-attention Metal MLP mechanism"
            ):
                harness.validate_run_receipt(candidate_path, variant="B", frozen=frozen)

            contract_path = root / "timed-b-contract.json"
            candidate = json.loads(json.dumps(frozen_candidate))
            candidate["generation_path_contract"][
                "binds_six_three_layer_stacks_and_six_full_attention_mlp_layers"
            ] = False
            write_json(contract_path, candidate)
            with self.assertRaisesRegex(
                harness.HarnessError, "generation path contract"
            ):
                harness.validate_run_receipt(contract_path, variant="B", frozen=frozen)

    def test_harness_custody_is_rehashed_at_campaign_start_and_end(self):
        harness = load_module()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            summary, summary_sha, binary_sha = make_frozen_fixture(root)
            frozen = harness.validate_frozen_inputs(
                summary,
                repo_root=root,
                expected_summary_sha256=summary_sha,
                expected_binary_sha256=binary_sha,
            )
            lane = harness.Stack3CampaignLane(frozen)
            drifted = json.loads(json.dumps(frozen["harness_custody"]))
            drifted["wrapper"]["sha256"] = "0" * 64

            with mock.patch.object(
                harness,
                "validate_harness_custody",
                side_effect=[frozen["harness_custody"], drifted],
            ) as custody_probe:
                start = lane.validate_live_custody(frozen["summary"])
                end = lane.validate_live_custody(frozen["summary"])

            self.assertEqual(start, frozen["live_custody"])
            self.assertNotEqual(end, frozen["live_custody"])
            self.assertEqual(custody_probe.call_count, 2)
            self.assertEqual(
                lane.report_identity["harness_custody"],
                frozen["harness_custody"],
            )


if __name__ == "__main__":
    unittest.main()
