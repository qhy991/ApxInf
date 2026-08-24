from __future__ import annotations

from contextlib import redirect_stdout
import importlib.util
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "scripts" / "run_qwen35_all18_formal_benchmark.py"


def load_module():
    spec = importlib.util.spec_from_file_location(
        "run_qwen35_all18_formal_benchmark_for_tests", MODULE_PATH
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
        json.dumps(value, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )


def make_frozen_fixture(root: Path) -> tuple[Path, str]:
    binary = root / "bin" / "gate"
    gate_source = root / "src" / "gate.rs"
    general_source = root / "src" / "general.rs"
    profile = root / "config" / "profile.json"
    source_lock = root / "lock" / "source-lock.json"
    model_dir = root / "model"
    for path, payload in (
        (binary, b"binary"),
        (gate_source, b"gate-source"),
        (general_source, b"general-source"),
        (profile, b"profile"),
        (source_lock, b"source-lock"),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    model_dir.mkdir()
    model_artifacts = {}
    for name in (
        "chat_template.jinja",
        "config.json",
        "model.safetensors-00001-of-00001.safetensors",
        "model.safetensors.index.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ):
        artifact = model_dir / name
        artifact.write_bytes(("fixture-" + name).encode("utf-8"))
        model_artifacts[name] = {
            "sha256": sha256(artifact),
            "size": artifact.stat().st_size,
            "direct_regular_file": True,
            "single_link": True,
        }

    custody = {
        "binary": {
            "path": str(binary),
            "sha256": sha256(binary),
            "size": binary.stat().st_size,
            "build_profile": "release",
            "features": ["accelerate", "metal-w8"],
            "direct_regular_file": True,
            "single_link": True,
        },
        "sources": {
            "gate": {
                "path": str(gate_source),
                "sha256": sha256(gate_source),
                "size": gate_source.stat().st_size,
                "direct_regular_file": True,
                "single_link": True,
            },
            "general": {
                "path": str(general_source),
                "sha256": sha256(general_source),
                "size": general_source.stat().st_size,
                "direct_regular_file": True,
                "single_link": True,
            },
        },
        "profile": {
            "path": str(profile),
            "profile_id": "qwen35-0.8b-macos-cpu",
            "sha256": sha256(profile),
            "size": profile.stat().st_size,
            "direct_regular_file": True,
            "single_link": True,
        },
        "source_lock": {
            "path": str(source_lock),
            "file_sha256": sha256(source_lock),
            "canonical_content_sha256_without_content_field": "a" * 64,
            "size": source_lock.stat().st_size,
            "direct_regular_file": True,
            "single_link": True,
        },
        "model_dir": {
            "path": str(model_dir),
            "closure": "exact-profile-artifacts-plus-safe-cache-v1",
            "cache_present": False,
            "artifacts": model_artifacts,
        },
    }
    receipt_identity = {
        "binary_path": str(binary),
        "build_profile": "release",
        "model_dir": str(model_dir),
        "source_lock": str(source_lock),
        "custody": {
            "binary": {
                "path": str(binary),
                "sha256": sha256(binary),
                "size": binary.stat().st_size,
                "direct_regular_file": True,
                "single_link": True,
            },
            "sources": custody["sources"],
            "model_dir": custody["model_dir"],
            "profile": {
                "path": str(profile),
                "profile_id": "qwen35-0.8b-macos-cpu",
                "file_sha256": sha256(profile),
                "file_size": profile.stat().st_size,
                "direct_regular_file": True,
                "single_link": True,
            },
            "source_lock": {
                "path": str(source_lock),
                "file_sha256": sha256(source_lock),
                "file_size": source_lock.stat().st_size,
                "content_sha256": "a" * 64,
                "direct_regular_file": True,
                "single_link": True,
            },
        },
    }
    receipt_specs = {
        "cpu_teacher128": (
            "cpu-teacher.json",
            "linear_layer_cpu_teacher",
            "apxinf-qwen35-metal-w8-linear-layer-cpu-teacher-v1",
        ),
        "candidate_teacher128": (
            "candidate-teacher.json",
            "metal_w8_all_linear_layers_gdn_out_g32_v2_teacher_forced",
            ("apxinf-qwen35-metal-w8-all-linear-layers-gdn-out-g32-v2-teacher-gate-v1"),
        ),
        "cpu_free128": (
            "cpu-free.json",
            "linear_layer_cpu_free_run",
            "apxinf-qwen35-metal-w8-linear-layer-cpu-free-run-v1",
        ),
        "candidate_free128": (
            "candidate-free.json",
            "metal_w8_all_linear_layers_gdn_out_g32_v2_free_run",
            (
                "apxinf-qwen35-metal-w8-all-linear-layers-gdn-out-g32-v2-"
                "free-run-gate-v1"
            ),
        ),
    }
    integrity = {}
    for key, (name, mode, format_name) in receipt_specs.items():
        path = root / "receipts" / name
        payload = {
            "format": format_name,
            "mode": mode,
            "passed": True,
            "identity": receipt_identity,
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
        elif key == "candidate_free128":
            cpu_path = root / "receipts" / "cpu-free.json"
            payload.update(
                {
                    "generated_tokens": 128,
                    "cpu_free_receipt": {
                        "path": str(cpu_path),
                        "sha256": sha256(cpu_path),
                        "size": cpu_path.stat().st_size,
                        "direct_regular_file": True,
                        "single_link": True,
                    },
                    "cpu_generated_token_ids": list(range(128)),
                    "metal_w8_all_linear_layers_generated_token_ids": list(range(128)),
                    "mismatches": [],
                    "first_mismatch": None,
                    "path_checks": {"exact_trajectory": True},
                    "aggregate_buffer_ledger": {"allocated_buffers": 624},
                    "timing": {"decode_mean_ms": 8.0, "prefill_ms": 105.0},
                }
            )
        write_json(
            path,
            payload,
        )
        integrity[key] = {
            "path": str(path.relative_to(root)),
            "sha256": sha256(path),
            "size": path.stat().st_size,
            "direct_regular_file": True,
            "single_link": True,
        }
    summary = root / "summary.json"
    write_json(
        summary,
        {
            "format": (
                "apxinf-qwen35-metal-w8-all-linear-layers-precision-v2-"
                "real-gate-summary-v1"
            ),
            "receipt_integrity": {
                **integrity,
                "all_four_identity_records_equal": True,
                "independently_rehashed": True,
            },
            "trajectory_gate": {
                "all_four_receipts_passed": True,
                "all_candidate_path_checks_true": True,
                "teacher_forced": {"exact_128_of_128": True},
                "free_run": {"exact_128_tokens": True},
            },
            "custody": custody,
            "aggregate_buffer_ledger": {
                "independent_sum_matches_both_candidate_receipts": True,
                "aggregate": {"allocated_buffers": 624},
            },
            "gate_result": {
                "correctness_and_path_gate_passed": True,
                "aggregate_ledger_valid": True,
            },
        },
    )
    return summary, sha256(summary)


def make_formal_blocks(
    harness,
    *,
    candidate_tps: float = 120.0,
    candidate_ttft_ms: float = 105.0,
) -> list[dict]:
    blocks = []
    for block_index, order in enumerate(harness.BLOCK_ORDERS):
        samples = []
        for run_index, variant in enumerate(order):
            throughput = 100.0 if variant == "A" else candidate_tps
            samples.append(
                {
                    "index": run_index,
                    "variant": variant,
                    "decode_mean_ms": 1000.0 / throughput,
                    "ttft_ms": 100.0 if variant == "A" else candidate_ttft_ms,
                    "trajectory_sha256": "t" * 64,
                    "path_valid": True,
                    "ledger_valid": True,
                    "custody_sha256": "c" * 64,
                    "child_swaps": 0,
                    "peak_group_rss_bytes": 1_000_000_000,
                    "quiet_custody": {
                        "passed": True,
                        "sample_count": 3,
                        "online_sample_count": 1,
                        "max_external_cpu_percent": 0.0,
                        "max_load_1m": 1.0,
                        "pages_throttled_observed": [0, 0, 0],
                        "swap_drift_bytes": 0,
                    },
                }
            )
        blocks.append(
            {
                "index": block_index,
                "order": order,
                "quiet_host": {"passed": True},
                "system_swap_growth_bytes": 0,
                "samples": samples,
            }
        )
    return blocks


def clean_quiet_sample(*, swap_used_bytes: int = 7_000_000_000) -> dict:
    return {
        "logical_cpus": 10,
        "load_1m": 1.0,
        "pages_throttled": 0,
        "swap_used_bytes": swap_used_bytes,
        "processes": [],
    }


def clean_preflight() -> dict:
    return {
        "passed": True,
        "problems": [],
        "final_swap_used_bytes": 7_000_000_000,
    }


def write_candidate_run_receipt(
    path: Path,
    *,
    identity: dict,
    token_ids: list[int],
    cpu_record: dict,
    path_checks: dict,
    ledger: dict,
) -> None:
    write_json(
        path,
        {
            "format": (
                "apxinf-qwen35-metal-w8-all-linear-layers-gdn-out-g32-v2-"
                "free-run-gate-v1"
            ),
            "mode": "metal_w8_all_linear_layers_gdn_out_g32_v2_free_run",
            "passed": True,
            "generated_tokens": len(token_ids),
            "identity": identity,
            "cpu_free_receipt": cpu_record,
            "cpu_generated_token_ids": token_ids,
            "metal_w8_all_linear_layers_generated_token_ids": token_ids,
            "mismatches": [],
            "first_mismatch": None,
            "path_checks": path_checks,
            "aggregate_buffer_ledger": ledger,
            "timing": {"decode_mean_ms": 8.0, "prefill_ms": 100.0},
        },
    )


def make_fake_campaign_runner(harness, frozen, calls, mutate_mid_sample=None):
    def runner(argv, **_kwargs):
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
                    "timing": {"decode_mean_ms": 10.0, "prefill_ms": 100.0},
                },
            )
        else:
            write_candidate_run_receipt(
                output,
                identity=frozen["identity"],
                token_ids=list(range(128)),
                cpu_record=frozen["receipt_records"]["cpu_free128"],
                path_checks=frozen["receipts"]["candidate_free128"]["path_checks"],
                ledger=frozen["receipts"]["candidate_free128"][
                    "aggregate_buffer_ledger"
                ],
            )
        mid_sample = clean_quiet_sample()
        if mutate_mid_sample is not None:
            mutate_mid_sample(mid_sample, len(calls))
        return {
            "argv": argv,
            "returncode": 0,
            "timed_out": False,
            "termination_reason": None,
            "peak_group_rss_bytes": 1_000_000_000,
            "rss_limit_bytes": harness.RUN_RSS_LIMIT_BYTES,
            "stdout": b"",
            "stderr": b" 1000000000 maximum resident set size\n 0 swaps\n",
            "owned_process_group": 88_888,
            "quiet_samples": [mid_sample],
        }

    return runner


class FormalBenchmarkHarnessTests(unittest.TestCase):
    def test_plan_is_fixed_to_three_abba_and_three_baab_blocks(self):
        harness = load_module()

        plan = harness.build_schedule(Path("/private/tmp/formal-all18"))

        self.assertEqual(
            [block["order"] for block in plan["blocks"]],
            ["ABBA", "BAAB", "ABBA", "BAAB", "ABBA", "BAAB"],
        )
        self.assertEqual(sum(len(block["runs"]) for block in plan["blocks"]), 24)
        self.assertEqual(
            {run["mode"] for block in plan["blocks"] for run in block["runs"]},
            {"cpu-free", "all-linear-layers-gdn-out-g32-v2-free"},
        )
        outputs = [run["output"] for block in plan["blocks"] for run in block["runs"]]
        self.assertEqual(len(outputs), len(set(outputs)))

    def test_frozen_input_loader_pins_the_exact_summary_bytes(self):
        harness = load_module()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            summary, summary_sha = make_frozen_fixture(root)

            frozen = harness.validate_frozen_inputs(
                summary,
                repo_root=root,
                expected_summary_sha256=summary_sha,
            )
            self.assertEqual(frozen["summary_sha256"], summary_sha)

            summary.write_bytes(summary.read_bytes() + b"\n")
            with self.assertRaisesRegex(harness.HarnessError, "summary SHA-256"):
                harness.validate_frozen_inputs(
                    summary,
                    repo_root=root,
                    expected_summary_sha256=summary_sha,
                )

    def test_frozen_input_loader_rejects_any_of_the_four_receipts_drifting(self):
        harness = load_module()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            summary, summary_sha = make_frozen_fixture(root)
            candidate_free = root / "receipts" / "candidate-free.json"
            candidate_free.write_bytes(candidate_free.read_bytes() + b"\n")

            with self.assertRaisesRegex(harness.HarnessError, "receipt.*SHA-256"):
                harness.validate_frozen_inputs(
                    summary,
                    repo_root=root,
                    expected_summary_sha256=summary_sha,
                )

    def test_frozen_input_loader_rehashes_binary_source_and_model_custody(self):
        harness = load_module()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            summary, summary_sha = make_frozen_fixture(root)
            (root / "bin" / "gate").write_bytes(b"drifted-binary")

            with self.assertRaisesRegex(harness.HarnessError, "binary custody"):
                harness.validate_frozen_inputs(
                    summary,
                    repo_root=root,
                    expected_summary_sha256=summary_sha,
                )

    def test_quiet_host_rejects_any_non_allowlisted_process_above_five_percent(self):
        harness = load_module()
        samples = [
            {
                "logical_cpus": 10,
                "load_1m": 2.0,
                "pages_throttled": 0,
                "swap_used_bytes": 7_000_000_000,
                "processes": [
                    {"pid": 42, "cpu_percent": 4.9, "command": "allowed-idle"},
                    {
                        "pid": 99,
                        "cpu_percent": 50.1,
                        "command": "logioptionsplus_agent",
                    },
                ],
            }
            for _ in range(harness.QUIET_SAMPLE_COUNT)
        ]

        result = harness.evaluate_quiet_host_samples(samples, allowed_pids={42})

        self.assertFalse(result["passed"])
        self.assertEqual(result["offenders"][0]["pid"], 99)
        self.assertIn("non-allowlisted process exceeded 5% CPU", result["problems"])

    def test_quiet_host_requires_zero_throttled_pages_and_stable_swap(self):
        harness = load_module()

        def clean_sample():
            return {
                "logical_cpus": 10,
                "load_1m": 1.0,
                "pages_throttled": 0,
                "swap_used_bytes": 7_000_000_000,
                "processes": [],
            }

        throttled = [clean_sample() for _ in range(harness.QUIET_SAMPLE_COUNT)]
        throttled[2]["pages_throttled"] = 1
        throttle_result = harness.evaluate_quiet_host_samples(
            throttled, allowed_pids=set()
        )
        self.assertFalse(throttle_result["passed"])
        self.assertIn(
            "memory_pressure Pages throttled must remain zero",
            throttle_result["problems"],
        )

        swap_drift = [clean_sample() for _ in range(harness.QUIET_SAMPLE_COUNT)]
        swap_drift[-1]["swap_used_bytes"] += 4096
        swap_result = harness.evaluate_quiet_host_samples(
            swap_drift, allowed_pids=set()
        )
        self.assertFalse(swap_result["passed"])
        self.assertIn(
            "system swap usage changed during quiet-host sampling",
            swap_result["problems"],
        )

    def test_preflight_rejects_global_load_above_the_declared_threshold(self):
        harness = load_module()
        samples = [clean_quiet_sample() for _ in range(harness.QUIET_SAMPLE_COUNT)]
        samples[3]["load_1m"] = 5.01

        result = harness.evaluate_quiet_host_samples(samples, allowed_pids=set())

        self.assertFalse(result["passed"])
        self.assertIn("load exceeded the quiet-host threshold", result["problems"])

    def test_zero_load_is_a_valid_quiet_host_and_runtime_sample(self):
        harness = load_module()
        samples = [clean_quiet_sample() for _ in range(harness.QUIET_SAMPLE_COUNT)]
        for sample in samples:
            sample["load_1m"] = 0.0

        preflight = harness.evaluate_quiet_host_samples(samples, allowed_pids=set())
        runtime = harness.evaluate_run_quiet_custody(
            start_sample=samples[0],
            online_samples=[samples[1]],
            end_sample=samples[2],
            allowed_pids=set(),
            owned_process_group=44_444,
            baseline_swap_used_bytes=7_000_000_000,
        )
        blocks = make_formal_blocks(harness)
        for block in blocks:
            for sample in block["samples"]:
                sample["quiet_custody"]["max_load_1m"] = 0.0
        reduced = harness.reduce_formal_campaign(
            blocks,
            expected_trajectory_sha256="t" * 64,
            expected_custody_sha256="c" * 64,
        )

        self.assertTrue(preflight["passed"])
        self.assertTrue(runtime["passed"])
        self.assertEqual(runtime["max_load_1m"], 0.0)
        self.assertTrue(reduced["accepted"])

    def test_runtime_load_is_recorded_but_not_used_as_external_noise_gate(self):
        harness = load_module()
        start = clean_quiet_sample()
        online = clean_quiet_sample()
        end = clean_quiet_sample()
        online["load_1m"] = 9.0
        end["load_1m"] = 8.0

        result = harness.evaluate_run_quiet_custody(
            start_sample=start,
            online_samples=[online],
            end_sample=end,
            allowed_pids=set(),
            owned_process_group=44_444,
            baseline_swap_used_bytes=7_000_000_000,
        )

        self.assertTrue(result["passed"])
        self.assertEqual(result["max_load_1m"], 9.0)

    def test_supervisor_timeout_terminates_the_complete_process_group(self):
        harness = load_module()
        started = time.monotonic()
        result = harness.run_supervised(
            [
                "/usr/bin/python3",
                "-I",
                "-B",
                "-c",
                (
                    "import subprocess,sys,time;"
                    "child=subprocess.Popen([sys.executable,'-I','-B','-c',"
                    "'import time;time.sleep(30)']);"
                    "print(child.pid,flush=True);time.sleep(30)"
                ),
            ],
            cwd=ROOT,
            environment={
                "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
                "LANG": "C",
                "LC_ALL": "C",
            },
            timeout_seconds=0.15,
        )

        self.assertTrue(result["timed_out"])
        self.assertEqual(result["termination_reason"], "timeout")
        self.assertLess(time.monotonic() - started, 1.5)
        descendant_pid = int(result["stdout"].decode("ascii").strip())
        time.sleep(0.05)
        with self.assertRaises(ProcessLookupError):
            os.kill(descendant_pid, 0)

    def test_supervisor_bounds_stdout_and_stderr_independently(self):
        harness = load_module()
        environment = {
            "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
            "LANG": "C",
            "LC_ALL": "C",
        }
        for file_descriptor, expected in ((1, "stdout_limit"), (2, "stderr_limit")):
            with self.subTest(stream=expected):
                result = harness.run_supervised(
                    [
                        "/usr/bin/python3",
                        "-I",
                        "-B",
                        "-c",
                        (
                            "import os,time;"
                            f"os.write({file_descriptor},b'x'*4097);"
                            "time.sleep(10)"
                        ),
                    ],
                    cwd=ROOT,
                    environment=environment,
                    timeout_seconds=2,
                    stream_limit_bytes=4096,
                )
                self.assertEqual(result["termination_reason"], expected)

    def test_supervisor_enforces_strictly_less_than_six_gib_group_rss(self):
        harness = load_module()
        result = harness.run_supervised(
            [
                "/usr/bin/python3",
                "-I",
                "-B",
                "-c",
                "import time;time.sleep(10)",
            ],
            cwd=ROOT,
            environment={
                "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
                "LANG": "C",
                "LC_ALL": "C",
            },
            timeout_seconds=2,
            rss_limit_bytes=harness.RUN_RSS_LIMIT_BYTES,
            rss_probe=lambda _process_group: harness.RUN_RSS_LIMIT_BYTES,
        )

        self.assertEqual(result["termination_reason"], "rss_limit")
        self.assertEqual(result["peak_group_rss_bytes"], harness.RUN_RSS_LIMIT_BYTES)

    def test_real_supervisor_samples_owned_run_and_stops_on_external_contamination(
        self,
    ):
        harness = load_module()
        environment = {
            "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
            "LANG": "C",
            "LC_ALL": "C",
        }

        def owned_only(process_group):
            sample = clean_quiet_sample()
            sample["processes"] = [
                {
                    "pid": process_group,
                    "pgid": process_group,
                    "cpu_percent": 99.0,
                    "command": "owned-fixture",
                }
            ]
            return sample

        clean = harness.run_supervised(
            [
                "/usr/bin/python3",
                "-I",
                "-B",
                "-c",
                "import time;time.sleep(0.18)",
            ],
            cwd=ROOT,
            environment=environment,
            timeout_seconds=1,
            rss_probe=lambda _group: 1_000_000,
            quiet_sample_probe=owned_only,
            quiet_allowed_pids=set(),
            quiet_baseline_swap_used_bytes=7_000_000_000,
            quiet_sample_interval_seconds=0.05,
        )
        self.assertIsNone(clean["termination_reason"])
        self.assertGreaterEqual(len(clean["quiet_samples"]), 2)

        calls = []

        def contaminated(process_group):
            sample = owned_only(process_group)
            calls.append(process_group)
            if len(calls) >= 2:
                sample["processes"].append(
                    {
                        "pid": 999_001,
                        "pgid": 999_001,
                        "cpu_percent": 51.0,
                        "command": "logioptionsplus_agent",
                    }
                )
            return sample

        rejected = harness.run_supervised(
            [
                "/usr/bin/python3",
                "-I",
                "-B",
                "-c",
                "import time;time.sleep(10)",
            ],
            cwd=ROOT,
            environment=environment,
            timeout_seconds=1,
            rss_probe=lambda _group: 1_000_000,
            quiet_sample_probe=contaminated,
            quiet_allowed_pids=set(),
            quiet_baseline_swap_used_bytes=7_000_000_000,
            quiet_sample_interval_seconds=0.05,
        )
        self.assertEqual(rejected["termination_reason"], "host_contamination")
        self.assertGreaterEqual(len(rejected["quiet_samples"]), 2)
        self.assertIn(
            "non-allowlisted process exceeded 5% CPU",
            rejected["quiet_contamination"],
        )

    def test_supervised_result_requires_zero_child_swaps(self):
        harness = load_module()
        result = {
            "returncode": 0,
            "timed_out": False,
            "termination_reason": None,
            "peak_group_rss_bytes": 1_000_000,
            "rss_limit_bytes": harness.RUN_RSS_LIMIT_BYTES,
            "stderr": (b" 1000000  maximum resident set size\n 1  swaps\n"),
        }

        with self.assertRaisesRegex(harness.HarnessError, "zero child swaps"):
            harness.validate_supervised_result(result)

    def test_reducer_accepts_only_the_fixed_24_sample_formal_protocol(self):
        harness = load_module()
        blocks = make_formal_blocks(harness)

        result = harness.reduce_formal_campaign(
            blocks,
            expected_trajectory_sha256="t" * 64,
            expected_custody_sha256="c" * 64,
        )

        self.assertTrue(result["accepted"])
        self.assertEqual(result["sample_count"], 24)
        self.assertEqual(result["baseline_sample_count"], 12)
        self.assertEqual(result["candidate_sample_count"], 12)
        self.assertEqual(result["same_direction_blocks"], 6)
        self.assertAlmostEqual(result["median_speedup"], 1.20)
        self.assertAlmostEqual(result["ttft_ratio"], 1.05)

    def test_reducer_requires_candidate_to_win_all_six_block_medians(self):
        harness = load_module()
        blocks = make_formal_blocks(harness)
        for sample in blocks[-1]["samples"]:
            if sample["variant"] == "B":
                sample["decode_mean_ms"] = 1000.0 / 99.0

        result = harness.reduce_formal_campaign(
            blocks,
            expected_trajectory_sha256="t" * 64,
            expected_custody_sha256="c" * 64,
        )

        self.assertFalse(result["accepted"])
        self.assertEqual(result["same_direction_blocks"], 5)
        self.assertGreater(result["median_speedup"], 1.10)
        self.assertIn("candidate must win all six block medians", result["problems"])

    def test_reducer_enforces_speedup_and_ttft_threshold_boundaries(self):
        harness = load_module()
        at_boundary = harness.reduce_formal_campaign(
            make_formal_blocks(harness, candidate_tps=110.0, candidate_ttft_ms=110.0),
            expected_trajectory_sha256="t" * 64,
            expected_custody_sha256="c" * 64,
        )
        self.assertTrue(at_boundary["accepted"])

        slow = harness.reduce_formal_campaign(
            make_formal_blocks(harness, candidate_tps=109.9),
            expected_trajectory_sha256="t" * 64,
            expected_custody_sha256="c" * 64,
        )
        self.assertFalse(slow["accepted"])
        self.assertIn("candidate median speedup is below 1.10x", slow["problems"])

        ttft = harness.reduce_formal_campaign(
            make_formal_blocks(harness, candidate_ttft_ms=110.1),
            expected_trajectory_sha256="t" * 64,
            expected_custody_sha256="c" * 64,
        )
        self.assertFalse(ttft["accepted"])
        self.assertIn("candidate TTFT regressed by more than 10%", ttft["problems"])

    def test_nonfinite_timing_rss_load_and_derived_ratios_fail_closed(self):
        harness = load_module()
        for value in (float("inf"), float("nan")):
            with self.subTest(metric="decode", value=value):
                blocks = make_formal_blocks(harness)
                blocks[0]["samples"][0]["decode_mean_ms"] = value
                result = harness.reduce_formal_campaign(
                    blocks,
                    expected_trajectory_sha256="t" * 64,
                    expected_custody_sha256="c" * 64,
                )
                self.assertFalse(result["accepted"])
                self.assertTrue(
                    any("decode_mean_ms is invalid" in p for p in result["problems"])
                )

        overflow = make_formal_blocks(harness)
        for block in overflow:
            for sample in block["samples"]:
                if sample["variant"] == "B":
                    sample["decode_mean_ms"] = 5e-324
        overflow_result = harness.reduce_formal_campaign(
            overflow,
            expected_trajectory_sha256="t" * 64,
            expected_custody_sha256="c" * 64,
        )
        self.assertFalse(overflow_result["accepted"])
        self.assertTrue(
            any(
                "derived throughput is non-finite or non-positive" in problem
                for problem in overflow_result["problems"]
            )
        )

        load_sample = clean_quiet_sample()
        load_sample["load_1m"] = float("inf")
        load_result = harness.evaluate_quiet_sample(
            load_sample,
            allowed_pids=set(),
            owned_process_group=None,
            baseline_swap_used_bytes=7_000_000_000,
            enforce_load_threshold=False,
        )
        self.assertFalse(load_result["passed"])

        with self.assertRaisesRegex(harness.HarnessError, "group RSS limit"):
            harness.validate_supervised_result(
                {
                    "returncode": 0,
                    "timed_out": False,
                    "termination_reason": None,
                    "peak_group_rss_bytes": float("inf"),
                    "rss_limit_bytes": harness.RUN_RSS_LIMIT_BYTES,
                    "stderr": b" 1000 maximum resident set size\n 0 swaps\n",
                }
            )
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "nonfinite.json"
            with self.assertRaises(ValueError):
                harness.atomic_write_json_no_replace(output, {"value": float("nan")})
            self.assertFalse(output.exists())

    def test_reducer_rejects_contamination_trajectory_path_ledger_and_custody_drift(
        self,
    ):
        harness = load_module()
        blocks = make_formal_blocks(harness)
        blocks[0]["quiet_host"] = {"passed": False}
        blocks[1]["samples"][0]["trajectory_sha256"] = "x" * 64
        blocks[2]["samples"][0]["path_valid"] = False
        blocks[3]["samples"][0]["ledger_valid"] = False
        blocks[4]["samples"][0]["custody_sha256"] = "d" * 64

        result = harness.reduce_formal_campaign(
            blocks,
            expected_trajectory_sha256="t" * 64,
            expected_custody_sha256="c" * 64,
        )

        self.assertFalse(result["accepted"])
        self.assertTrue(result["replacement_required"])
        self.assertIn("formal block 0 was not quiet", result["contamination"])
        joined = "\n".join(result["problems"])
        for phrase in (
            "trajectory drifted",
            "execution path",
            "Metal ledger",
            "custody",
        ):
            self.assertIn(phrase, joined)

    def test_candidate_run_receipt_requires_exact_trajectory_path_ledger_and_identity(
        self,
    ):
        harness = load_module()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            summary, summary_sha = make_frozen_fixture(root)
            frozen = harness.validate_frozen_inputs(
                summary,
                repo_root=root,
                expected_summary_sha256=summary_sha,
            )
            output = root / "candidate-run.json"
            token_ids = list(range(128))
            cpu_path = root / "receipts" / "cpu-free.json"
            cpu_record = {
                "path": str(cpu_path),
                "sha256": sha256(cpu_path),
                "size": cpu_path.stat().st_size,
                "direct_regular_file": True,
                "single_link": True,
            }
            path_checks = {"exact_trajectory": True, "decode": {"all_valid": True}}
            ledger = {"allocated_buffers": 624, "waits_per_decode": 24}
            write_candidate_run_receipt(
                output,
                identity=frozen["identity"],
                token_ids=token_ids,
                cpu_record=cpu_record,
                path_checks=path_checks,
                ledger=ledger,
            )

            sample = harness.validate_run_receipt(
                output,
                variant="B",
                frozen_identity=frozen["identity"],
                expected_token_ids=token_ids,
                expected_cpu_receipt=cpu_record,
                expected_path_checks=path_checks,
                expected_ledger=ledger,
            )

            self.assertTrue(sample["path_valid"])
            self.assertTrue(sample["ledger_valid"])
            self.assertEqual(sample["variant"], "B")
            self.assertEqual(sample["decode_mean_ms"], 8.0)

    def test_candidate_run_receipt_rejects_trajectory_and_custody_drift(self):
        harness = load_module()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            summary, summary_sha = make_frozen_fixture(root)
            frozen = harness.validate_frozen_inputs(
                summary,
                repo_root=root,
                expected_summary_sha256=summary_sha,
            )
            expected_tokens = list(range(128))
            cpu_path = root / "receipts" / "cpu-free.json"
            cpu_record = {
                "path": str(cpu_path),
                "sha256": sha256(cpu_path),
                "size": cpu_path.stat().st_size,
                "direct_regular_file": True,
                "single_link": True,
            }
            path_checks = {"exact_trajectory": True}
            ledger = {"allocated_buffers": 624}

            wrong_trajectory = root / "wrong-trajectory.json"
            actual_tokens = list(expected_tokens)
            actual_tokens[9] = 999_999
            write_candidate_run_receipt(
                wrong_trajectory,
                identity=frozen["identity"],
                token_ids=actual_tokens,
                cpu_record=cpu_record,
                path_checks=path_checks,
                ledger=ledger,
            )
            with self.assertRaisesRegex(harness.HarnessError, "trajectory drifted"):
                harness.validate_run_receipt(
                    wrong_trajectory,
                    variant="B",
                    frozen_identity=frozen["identity"],
                    expected_token_ids=expected_tokens,
                    expected_cpu_receipt=cpu_record,
                    expected_path_checks=path_checks,
                    expected_ledger=ledger,
                )

            wrong_identity = root / "wrong-identity.json"
            drifted_identity = json.loads(json.dumps(frozen["identity"]))
            drifted_identity["build_profile"] = "drifted"
            write_candidate_run_receipt(
                wrong_identity,
                identity=drifted_identity,
                token_ids=expected_tokens,
                cpu_record=cpu_record,
                path_checks=path_checks,
                ledger=ledger,
            )
            with self.assertRaisesRegex(
                harness.HarnessError, "custody identity drifted"
            ):
                harness.validate_run_receipt(
                    wrong_identity,
                    variant="B",
                    frozen_identity=frozen["identity"],
                    expected_token_ids=expected_tokens,
                    expected_cpu_receipt=cpu_record,
                    expected_path_checks=path_checks,
                    expected_ledger=ledger,
                )

    def test_failed_initial_quiet_preflight_starts_no_child_and_writes_nothing(self):
        harness = load_module()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            summary, summary_sha = make_frozen_fixture(root)
            frozen = harness.validate_frozen_inputs(
                summary,
                repo_root=root,
                expected_summary_sha256=summary_sha,
            )
            output_dir = root / "formal-output"
            child_calls = []

            with self.assertRaisesRegex(harness.HarnessError, "quiet-host preflight"):
                harness.execute_campaign(
                    frozen=frozen,
                    repo_root=root,
                    output_dir=output_dir,
                    quiet_probe=lambda: {
                        "passed": False,
                        "problems": ["Logitech process exceeded 5% CPU"],
                    },
                    command_runner=lambda *args, **kwargs: child_calls.append(
                        (args, kwargs)
                    ),
                    swap_probe=lambda: 7_000_000_000,
                )

            self.assertEqual(child_calls, [])
            self.assertFalse(output_dir.exists())

    def test_source_or_model_drift_blocks_before_final_quiet_and_any_output(self):
        harness = load_module()
        for name, relative, expected in (
            (
                "source",
                Path("src/general.rs"),
                "general source custody SHA-256 drifted",
            ),
            (
                "model",
                Path("model/tokenizer.json"),
                "model artifact tokenizer.json custody SHA-256 drifted",
            ),
        ):
            with self.subTest(custody=name):
                with tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary)
                    summary, summary_sha = make_frozen_fixture(root)
                    frozen = harness.validate_frozen_inputs(
                        summary,
                        repo_root=root,
                        expected_summary_sha256=summary_sha,
                    )
                    target = root / relative
                    target.write_bytes(target.read_bytes() + b"drift")
                    quiet_calls = []
                    child_calls = []
                    output_dir = root / "formal-output"

                    with self.assertRaisesRegex(harness.HarnessError, expected):
                        harness.execute_campaign(
                            frozen=frozen,
                            repo_root=root,
                            output_dir=output_dir,
                            quiet_probe=lambda: quiet_calls.append(True),
                            run_quiet_sample_probe=(
                                lambda _owned_group=None: clean_quiet_sample()
                            ),
                            command_runner=lambda *args, **kwargs: child_calls.append(
                                (args, kwargs)
                            ),
                            swap_probe=lambda: 7_000_000_000,
                        )

                    self.assertEqual(quiet_calls, [])
                    self.assertEqual(child_calls, [])
                    self.assertFalse(output_dir.exists())

    def test_expensive_custody_finishes_before_final_quiet_and_first_run(self):
        harness = load_module()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            summary, summary_sha = make_frozen_fixture(root)
            frozen = harness.validate_frozen_inputs(
                summary,
                repo_root=root,
                expected_summary_sha256=summary_sha,
            )
            events = []

            def custody(_summary):
                events.append("custody")
                return frozen["live_custody"]

            def quiet():
                events.append("quiet")
                return clean_preflight()

            def run_sample(_owned_group=None):
                events.append("run-sample")
                return clean_quiet_sample()

            def runner(argv, **_kwargs):
                events.append("runner")
                return {
                    "argv": argv,
                    "returncode": -9,
                    "timed_out": True,
                    "termination_reason": "timeout",
                    "peak_group_rss_bytes": 1_000_000,
                    "rss_limit_bytes": harness.RUN_RSS_LIMIT_BYTES,
                    "stdout": b"",
                    "stderr": b"",
                    "quiet_samples": [clean_quiet_sample()],
                }

            with mock.patch.object(harness, "validate_live_custody", custody):
                report = harness.execute_campaign(
                    frozen=frozen,
                    repo_root=root,
                    output_dir=root / "formal-output",
                    quiet_probe=quiet,
                    run_quiet_sample_probe=run_sample,
                    command_runner=runner,
                    swap_probe=lambda: 7_000_000_000,
                )

            self.assertFalse(report["formal_accepted"])
            self.assertEqual(events[:4], ["custody", "quiet", "run-sample", "runner"])

    def test_campaign_swap_must_equal_the_final_preflight_sample(self):
        harness = load_module()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            summary, summary_sha = make_frozen_fixture(root)
            frozen = harness.validate_frozen_inputs(
                summary,
                repo_root=root,
                expected_summary_sha256=summary_sha,
            )
            child_calls = []
            report = harness.execute_campaign(
                frozen=frozen,
                repo_root=root,
                output_dir=root / "formal-output",
                quiet_probe=clean_preflight,
                run_quiet_sample_probe=lambda _owned_group=None: clean_quiet_sample(),
                command_runner=lambda *args, **kwargs: child_calls.append(
                    (args, kwargs)
                ),
                swap_probe=lambda: 7_000_004_096,
            )

            self.assertEqual(child_calls, [])
            self.assertFalse(report["formal_accepted"])
            self.assertIn("changed after quiet-host preflight", report["error"])

    def test_cli_defaults_to_a_dry_plan_and_never_calls_execute(self):
        harness = load_module()
        execute = mock.Mock(side_effect=AssertionError("must not execute"))
        frozen = {"summary_sha256": harness.PINNED_SUMMARY_SHA256}
        stdout = io.StringIO()
        with mock.patch.object(harness, "validate_frozen_inputs", return_value=frozen):
            with mock.patch.object(harness, "execute_campaign", execute):
                with redirect_stdout(stdout):
                    returncode = harness.main([])

        self.assertEqual(returncode, 0)
        execute.assert_not_called()
        plan = json.loads(stdout.getvalue())
        self.assertFalse(plan["execution_started"])
        self.assertTrue(plan["requires_explicit_execute"])
        self.assertEqual(len(plan["schedule"]["blocks"]), 6)

    def test_fake_campaign_publishes_24_no_replace_receipts_and_one_formal_result(self):
        harness = load_module()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            summary, summary_sha = make_frozen_fixture(root)
            frozen = harness.validate_frozen_inputs(
                summary,
                repo_root=root,
                expected_summary_sha256=summary_sha,
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
                            "timing": {
                                "decode_mean_ms": 10.0,
                                "prefill_ms": 100.0,
                            },
                        },
                    )
                else:
                    write_candidate_run_receipt(
                        output,
                        identity=frozen["identity"],
                        token_ids=list(range(128)),
                        cpu_record=frozen["receipt_records"]["cpu_free128"],
                        path_checks=frozen["receipts"]["candidate_free128"][
                            "path_checks"
                        ],
                        ledger=frozen["receipts"]["candidate_free128"][
                            "aggregate_buffer_ledger"
                        ],
                    )
                return {
                    "argv": argv,
                    "returncode": 0,
                    "timed_out": False,
                    "termination_reason": None,
                    "peak_group_rss_bytes": 1_000_000_000,
                    "rss_limit_bytes": harness.RUN_RSS_LIMIT_BYTES,
                    "stdout": b"",
                    "stderr": (b" 1000000000 maximum resident set size\n 0 swaps\n"),
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
            self.assertEqual(len(calls), 24)
            self.assertEqual(len(list(output_dir.glob("block-*.json"))), 24)
            self.assertTrue((output_dir / "formal-result.json").is_file())
            self.assertEqual(list(output_dir.glob(".*.staging-*")), [])
            with self.assertRaisesRegex(harness.HarnessError, "already exists"):
                harness.atomic_write_json_no_replace(
                    output_dir / "formal-result.json", {"replacement": True}
                )

    def test_mid_campaign_failure_publishes_only_an_explicit_nonaccepted_result(self):
        harness = load_module()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            summary, summary_sha = make_frozen_fixture(root)
            frozen = harness.validate_frozen_inputs(
                summary,
                repo_root=root,
                expected_summary_sha256=summary_sha,
            )
            output_dir = root / "failed-output"
            calls = []

            def timed_out(argv, **_kwargs):
                calls.append(argv)
                return {
                    "argv": argv,
                    "returncode": -9,
                    "timed_out": True,
                    "termination_reason": "timeout",
                    "peak_group_rss_bytes": 1_000_000,
                    "rss_limit_bytes": harness.RUN_RSS_LIMIT_BYTES,
                    "stdout": b"",
                    "stderr": b"",
                }

            report = harness.execute_campaign(
                frozen=frozen,
                repo_root=root,
                output_dir=output_dir,
                quiet_probe=clean_preflight,
                run_quiet_sample_probe=lambda _owned_group=None: clean_quiet_sample(),
                command_runner=timed_out,
                swap_probe=lambda: 7_000_000_000,
            )

            self.assertEqual(len(calls), 1)
            self.assertEqual(report["status"], "failed")
            self.assertFalse(report["formal_accepted"])
            archived = json.loads(
                (output_dir / "formal-result.json").read_text(encoding="utf-8")
            )
            self.assertFalse(archived["formal_accepted"])
            self.assertEqual(list(output_dir.glob("block-*.json")), [])

    def test_interrupts_publish_nonaccepted_best_effort_then_reraise(self):
        harness = load_module()
        for exception in (KeyboardInterrupt(), SystemExit(7)):
            with self.subTest(interruption=type(exception).__name__):
                with tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary)
                    summary, summary_sha = make_frozen_fixture(root)
                    frozen = harness.validate_frozen_inputs(
                        summary,
                        repo_root=root,
                        expected_summary_sha256=summary_sha,
                    )
                    output_dir = root / "interrupted-output"

                    def interrupting_runner(argv, **_kwargs):
                        staging = Path(argv[argv.index("--output") + 1])
                        staging.write_bytes(b"partial")
                        raise exception

                    with self.assertRaises(type(exception)):
                        harness.execute_campaign(
                            frozen=frozen,
                            repo_root=root,
                            output_dir=output_dir,
                            quiet_probe=clean_preflight,
                            run_quiet_sample_probe=(
                                lambda _owned_group=None: clean_quiet_sample()
                            ),
                            command_runner=interrupting_runner,
                            swap_probe=lambda: 7_000_000_000,
                        )

                    archived = json.loads(
                        (output_dir / "formal-result.json").read_text(encoding="utf-8")
                    )
                    self.assertFalse(archived["formal_accepted"])
                    self.assertEqual(archived["status"], "interrupted")
                    self.assertEqual(list(output_dir.glob(".*.staging-*")), [])

    def test_run_two_midflight_logitech_contamination_stops_all_later_runs(self):
        harness = load_module()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            summary, summary_sha = make_frozen_fixture(root)
            frozen = harness.validate_frozen_inputs(
                summary,
                repo_root=root,
                expected_summary_sha256=summary_sha,
            )
            output_dir = root / "contaminated-output"
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
                            "timing": {
                                "decode_mean_ms": 10.0,
                                "prefill_ms": 100.0,
                            },
                        },
                    )
                else:
                    write_candidate_run_receipt(
                        output,
                        identity=frozen["identity"],
                        token_ids=list(range(128)),
                        cpu_record=frozen["receipt_records"]["cpu_free128"],
                        path_checks=frozen["receipts"]["candidate_free128"][
                            "path_checks"
                        ],
                        ledger=frozen["receipts"]["candidate_free128"][
                            "aggregate_buffer_ledger"
                        ],
                    )
                mid_samples = [clean_quiet_sample()]
                if len(calls) == 2:
                    mid_samples[0]["processes"] = [
                        {
                            "pid": 991,
                            "pgid": 991,
                            "cpu_percent": 51.0,
                            "command": "logioptionsplus_agent",
                        }
                    ]
                return {
                    "argv": argv,
                    "returncode": 0,
                    "timed_out": False,
                    "termination_reason": None,
                    "peak_group_rss_bytes": 1_000_000_000,
                    "rss_limit_bytes": harness.RUN_RSS_LIMIT_BYTES,
                    "stdout": b"",
                    "stderr": (b" 1000000000 maximum resident set size\n 0 swaps\n"),
                    "quiet_samples": mid_samples,
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

            self.assertEqual(len(calls), 2)
            self.assertFalse(report["formal_accepted"])
            self.assertEqual(report["status"], "failed")
            self.assertIn("non-allowlisted process exceeded 5% CPU", report["error"])
            self.assertEqual(len(list(output_dir.glob("block-*.json"))), 2)

    def test_run_two_midflight_throttling_or_swap_drift_stops_all_later_runs(self):
        harness = load_module()

        def throttle(sample, call_count):
            if call_count == 2:
                sample["pages_throttled"] = 1

        def swap(sample, call_count):
            if call_count == 2:
                sample["swap_used_bytes"] += 4096

        for name, mutator, expected in (
            (
                "throttled",
                throttle,
                "memory_pressure Pages throttled must remain zero",
            ),
            ("swap", swap, "system swap usage changed during formal measurement"),
        ):
            with self.subTest(contamination=name):
                with tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary)
                    summary, summary_sha = make_frozen_fixture(root)
                    frozen = harness.validate_frozen_inputs(
                        summary,
                        repo_root=root,
                        expected_summary_sha256=summary_sha,
                    )
                    output_dir = root / "contaminated-output"
                    calls = []
                    report = harness.execute_campaign(
                        frozen=frozen,
                        repo_root=root,
                        output_dir=output_dir,
                        quiet_probe=clean_preflight,
                        run_quiet_sample_probe=(
                            lambda _owned_group=None: clean_quiet_sample()
                        ),
                        command_runner=make_fake_campaign_runner(
                            harness, frozen, calls, mutator
                        ),
                        swap_probe=lambda: 7_000_000_000,
                    )

                    self.assertEqual(len(calls), 2)
                    self.assertFalse(report["formal_accepted"])
                    self.assertIn(expected, report["error"])
                    self.assertEqual(len(list(output_dir.glob("block-*.json"))), 2)


if __name__ == "__main__":
    unittest.main()
