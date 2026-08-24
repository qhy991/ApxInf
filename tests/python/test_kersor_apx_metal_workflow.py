import json
import hashlib
import importlib.util
import os
from pathlib import Path
import subprocess
import tempfile
import time
import unittest
import sys


REPO = Path(__file__).resolve().parents[2]
WORKFLOW_ROOT = REPO / "kersor" / "workflows"
WORKFLOW_DIR = WORKFLOW_ROOT / "ApxMetal"


def _write_fixture_qwen35_model(model: Path) -> None:
    model.mkdir()
    config = {
        "model_type": "qwen3_5",
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "text_config": {
            "model_type": "qwen3_5_text",
            "hidden_size": 1024,
            "intermediate_size": 3584,
            "num_hidden_layers": 24,
            "num_attention_heads": 8,
            "num_key_value_heads": 2,
            "head_dim": 256,
            "vocab_size": 248320,
            "tie_word_embeddings": True,
            "full_attention_interval": 4,
        },
    }
    (model / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (model / "tokenizer.json").write_bytes(b"{}")
    (model / "model.safetensors").write_bytes(b"fixture weights")


def _host_evaluator_module():
    path = WORKFLOW_DIR / "host_evaluator.py"
    spec = importlib.util.spec_from_file_location("apxinf_metal_host_evaluator", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module


def _kersor_root() -> Path:
    configured = os.environ.get("APXINF_TEST_KERSOR_ROOT", "").strip()
    if configured:
        return Path(configured)
    matches = sorted(
        (Path.home() / ".codex" / "plugins" / "cache" / "personal" / "kersor").glob(
            "*/scripts/generate-catalog.sh"
        )
    )
    if not matches:
        raise unittest.SkipTest("KerSor catalog generator is not installed")
    return matches[-1].parents[1]


class ApxMetalWorkflowCatalogTests(unittest.TestCase):
    def test_official_catalog_projects_exact_apxinf_metal_contract(self):
        generator = _kersor_root() / "scripts" / "generate-catalog.sh"
        with tempfile.TemporaryDirectory() as temporary:
            catalog_path = Path(temporary) / "catalog.json"
            completed = subprocess.run(
                ["/bin/bash", str(generator), str(WORKFLOW_ROOT), str(catalog_path)],
                cwd=REPO,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            payload = json.loads(catalog_path.read_text(encoding="utf-8"))

        self.assertEqual(len(payload["workflows"]), 1)
        entry = payload["workflows"][0]
        self.assertEqual(entry["name"], "apxinf-metal-w8-head-optimization")
        self.assertEqual(entry["metadata_source"], "manifest")
        self.assertEqual(entry["languages"], ["metal"])
        self.assertEqual(entry["backends"], ["metal"])
        self.assertEqual(
            entry["integration_patterns"],
            ["apxinf_qwen35_metal_w8_head"],
        )
        self.assertEqual(entry["required_args"], ["kernel_path", "model_path"])
        self.assertEqual(entry["seed_contract"], "reference_only")
        self.assertEqual(entry["requires_harness"], "internal")
        self.assertTrue(entry["requires_embedded_registration"])
        self.assertIsNone(entry["known_broken"])

    def test_official_router_reports_one_feasible_exact_integration(self):
        kersor = _kersor_root()
        generator = kersor / "scripts/generate-catalog.sh"
        preflight = kersor / "scripts/preflight-workflow-pool.py"
        python = Path("/opt/homebrew/bin/python3.13")
        if not python.is_file():
            raise unittest.SkipTest("KerSor Python 3.13 is not installed")
        with tempfile.TemporaryDirectory() as temporary:
            catalog_path = Path(temporary) / "catalog.json"
            generated = subprocess.run(
                ["/bin/bash", str(generator), str(WORKFLOW_ROOT), str(catalog_path)],
                cwd=REPO,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(generated.returncode, 0, generated.stderr)
            completed = subprocess.run(
                [
                    str(python),
                    str(preflight),
                    "--catalog",
                    str(catalog_path),
                    "--workflows",
                    "apxinf-metal-w8-head-optimization",
                    "--integration-pattern",
                    "apxinf_qwen35_metal_w8_head",
                    "--require-count",
                    "1",
                    "--require-all",
                    "--input-mode",
                    "existing_kernel",
                ],
                cwd=REPO,
                check=False,
                capture_output=True,
                text=True,
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            report = json.loads(completed.stdout)

        self.assertEqual(report["verdict"], "pass")
        self.assertEqual(report["feasible_count"], 1)
        self.assertEqual(
            report["feasible_workflows"],
            ["apxinf-metal-w8-head-optimization"],
        )
        self.assertEqual(report["rejected_workflows"], [])

    def test_host_contract_freezes_candidate_scope_and_measurement_order(self):
        contract = json.loads(
            (WORKFLOW_DIR / "host_contract.json").read_text(encoding="utf-8")
        )
        self.assertEqual(contract["backend"], "metal")
        self.assertEqual(contract["language"], "metal")
        self.assertEqual(
            contract["integration_pattern"],
            "apxinf_qwen35_metal_w8_head",
        )
        self.assertEqual(
            contract["candidate_scope"]["allowed_files"],
            ["crates/apxinf-metal/src/metal_w8.metal"],
        )
        self.assertEqual(contract["candidate_scope"]["bridge_policy"], "deny")
        self.assertEqual(
            contract["correctness"]["ordered_gates"],
            [
                "metal_adversarial_tests",
                "qwen35_tests",
                "teacher_forced_native_f32_128",
                "trajectory_exact_100",
                "execution_path_hit_and_negative_control",
            ],
        )
        self.assertEqual(
            contract["performance"]["block_orders"],
            ["ABBA", "BAAB", "ABBA", "BAAB", "ABBA", "BAAB"],
        )
        self.assertEqual(contract["performance"]["minimum_speedup"], 1.1)
        self.assertEqual(contract["performance"]["same_direction_blocks"], 6)
        self.assertEqual(contract["quality_claim"], "native_f32_only")
        self.assertFalse(contract["claims_hf_bf16_parity"])
        self.assertEqual(
            contract["execution"]["candidate_evaluator_argv_prefix"][:3],
            ["/usr/bin/python3", "-I", "-B"],
        )
        self.assertEqual(contract["execution"]["overall_deadline_seconds"], 1650)
        self.assertIn("model", contract["custody"])
        self.assertIn("sources", contract["custody"])
        self.assertIn("toolchain", contract["custody"])

    def test_workflow_pins_the_exact_evaluator_and_host_contract_bytes(self):
        workflow = (
            WORKFLOW_DIR / "apxinf-metal-w8-head-optimization.js"
        ).read_text(encoding="utf-8")
        for variable, name in (
            ("HOST_EVALUATOR_SHA256", "host_evaluator.py"),
            ("HOST_CONTRACT_SHA256", "host_contract.json"),
        ):
            digest = hashlib.sha256((WORKFLOW_DIR / name).read_bytes()).hexdigest()
            self.assertIn(f"const {variable} = '{digest}'", workflow)

    def test_candidate_validator_accepts_shader_only_and_rejects_include_escape(self):
        host = _host_evaluator_module()
        source = (REPO / "crates/apxinf-metal/src/metal_w8.metal").read_text(
            encoding="utf-8"
        )
        self.assertEqual(host.validate_candidate_source(source), [])

        escaped = source.replace(
            "#include <metal_stdlib>",
            '#include <metal_stdlib>\n#include "/Users/example/.ssh/id_ed25519"',
            1,
        )
        problems = host.validate_candidate_source(escaped)
        self.assertTrue(any("preprocessor" in problem for problem in problems))

    def test_request_validator_rejects_extra_mutation_fields_and_missing_model(self):
        host = _host_evaluator_module()
        source = (REPO / "crates/apxinf-metal/src/metal_w8.metal").read_text(
            encoding="utf-8"
        )
        request = {
            "schema_version": 1,
            "candidate_source": source,
            "kernel_path": str(REPO / "crates/apxinf-metal/src/metal_w8.metal"),
            "model_path": "",
            "strategy_id": "fixture",
        }
        problems = host.validate_request(request, project_root=REPO)
        self.assertTrue(any("model_path" in problem for problem in problems))

        request["model_path"] = "/tmp/model"
        request["bridge_source"] = "unauthorized"
        problems = host.validate_request(request, project_root=REPO)
        self.assertTrue(
            any("unexpected request keys" in problem for problem in problems)
        )

    def test_request_validator_binds_a_declared_retry_to_the_same_candidate_bytes(self):
        host = _host_evaluator_module()
        with tempfile.TemporaryDirectory() as temporary:
            model = Path(temporary) / "model"
            _write_fixture_qwen35_model(model)
            source = (REPO / host.CANONICAL_SHADER).read_text(encoding="utf-8")
            request = {
                "schema_version": 1,
                "candidate_source": source,
                "candidate_source_sha256": "0" * 64,
                "kernel_path": str(REPO / host.CANONICAL_SHADER),
                "model_path": str(model),
                "strategy_id": "retry-fixture",
            }
            problems = host.validate_request(request, project_root=REPO)

        self.assertIn(
            "candidate_source_sha256 does not match candidate_source bytes",
            problems,
        )
        self.assertFalse(any("unexpected request keys" in p for p in problems))

    def test_performance_reducer_requires_six_same_direction_blocks_and_ten_percent(
        self,
    ):
        host = _host_evaluator_module()
        orders = ["ABBA", "BAAB", "ABBA", "BAAB", "ABBA", "BAAB"]
        blocks = []
        for index, order in enumerate(orders):
            labels = list(order)
            samples = []
            for label in labels:
                samples.append(
                    {
                        "variant": label,
                        "generation_tps": 100.0 if label == "A" else 112.0,
                        "ttft_ms": 20.0,
                        "max_rss_bytes": 1_000_000_000,
                        "process_swaps": 0,
                        "generated_ids_sha256": "same-trajectory",
                    }
                )
            blocks.append(
                {
                    "index": index,
                    "order": order,
                    "quiet_host": {"passed": True},
                    "system_swap_growth_bytes": 0,
                    "samples": samples,
                }
            )

        reduced = host.reduce_performance(blocks, expected_trajectory="same-trajectory")
        self.assertTrue(reduced["accepted"])
        self.assertEqual(reduced["same_direction_blocks"], 6)
        self.assertAlmostEqual(reduced["generation_tps_speedup"], 1.12)
        self.assertEqual(reduced["sample_count"], 24)

    def test_contaminated_measurement_is_retained_and_requires_replacement(self):
        host = _host_evaluator_module()
        blocks = []
        for index, order in enumerate(host.BLOCK_ORDERS):
            samples = [
                {
                    "variant": label,
                    "generation_tps": 100.0 if label == "A" else 120.0,
                    "ttft_ms": 20.0,
                    "max_rss_bytes": 1_000_000_000,
                    "process_swaps": 0,
                    "generated_ids_sha256": "trajectory",
                }
                for label in order
            ]
            blocks.append(
                {
                    "index": index,
                    "order": order,
                    "quiet_host": {"passed": index != 2},
                    "system_swap_growth_bytes": 0,
                    "samples": samples,
                }
            )

        reduced = host.reduce_performance(blocks, expected_trajectory="trajectory")
        self.assertFalse(reduced["accepted"])
        self.assertTrue(reduced["replacement_required"])
        self.assertEqual(reduced["sample_count"], 24)
        self.assertEqual(reduced["preserved_blocks"], blocks)
        self.assertIn("block 2 failed quiet-host gate", reduced["contamination"])

    def test_schedule_level_swap_growth_forces_replacement_even_if_blocks_are_clean(
        self,
    ):
        host = _host_evaluator_module()
        blocks = []
        for index, order in enumerate(host.BLOCK_ORDERS):
            blocks.append(
                {
                    "index": index,
                    "order": order,
                    "quiet_host": {"passed": True},
                    "system_swap_growth_bytes": 0,
                    "samples": [
                        {
                            "variant": label,
                            "generation_tps": 100.0 if label == "A" else 120.0,
                            "ttft_ms": 20.0,
                            "max_rss_bytes": 1_000_000_000,
                            "process_swaps": 0,
                            "generated_ids_sha256": "trajectory",
                        }
                        for label in order
                    ],
                }
            )

        reduced = host.reduce_performance(
            blocks,
            expected_trajectory="trajectory",
            schedule_swap_growth_bytes=4096,
        )
        self.assertFalse(reduced["accepted"])
        self.assertTrue(reduced["replacement_required"])
        self.assertIn("formal schedule observed system swap growth", reduced["contamination"])

    def test_command_plan_is_shell_free_and_orders_correctness_before_timing(self):
        host = _host_evaluator_module()
        plan = host.build_command_plan(
            baseline_root=Path("/private/tmp/run/baseline"),
            candidate_root=Path("/private/tmp/run/candidate"),
            cargo=Path("/toolchain/bin/cargo"),
            model_path=Path("/models/qwen35"),
            prompt="Hello",
        )
        self.assertEqual(
            [gate["name"] for gate in plan["correctness"]],
            [
                "metal_adversarial_tests",
                "qwen35_tests",
                "teacher_forced_native_f32_128",
                "trajectory_exact_100",
                "execution_path_hit_and_negative_control",
            ],
        )
        self.assertEqual(plan["performance"]["block_orders"], list(host.BLOCK_ORDERS))
        for command in plan["commands"]:
            self.assertIsInstance(command, list)
            self.assertNotIn(command[0], {"sh", "bash", "/bin/sh", "/bin/bash"})

    def test_cli_returns_honest_blocked_receipt_before_any_benchmark(self):
        source = (REPO / "crates/apxinf-metal/src/metal_w8.metal").read_text(
            encoding="utf-8"
        )
        request = {
            "schema_version": 1,
            "candidate_source": source,
            "kernel_path": str(REPO / "crates/apxinf-metal/src/metal_w8.metal"),
            "model_path": "",
            "strategy_id": "fixture",
        }
        completed = subprocess.run(
            [
                "/usr/bin/python3",
                "-B",
                str(WORKFLOW_DIR / "host_evaluator.py"),
                "--request-json",
                json.dumps(request),
            ],
            cwd=REPO,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 1, completed.stderr)
        receipt = json.loads(completed.stdout)
        self.assertEqual(receipt["status"], "blocked_input")
        self.assertFalse(receipt["accepted"])
        self.assertFalse(receipt["formal_benchmark"]["executed"])
        self.assertEqual(receipt["quality_claim"], "native_f32_only")

    def test_valid_request_still_requires_command_v1_confinement(self):
        host = _host_evaluator_module()
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary) / "project"
            shader = project / host.CANONICAL_SHADER
            shader.parent.mkdir(parents=True)
            baseline = (
                "#include <metal_stdlib>\n"
                "kernel void w8_rows_topk4() {}\n"
                "kernel void w8_final_topk4() {}\n"
            )
            shader.write_text(baseline, encoding="utf-8")
            model = Path(temporary) / "model"
            _write_fixture_qwen35_model(model)
            request = {
                "schema_version": 1,
                "candidate_source": baseline.replace("{}", "{ int x = 1; }", 1),
                "kernel_path": str(shader),
                "model_path": str(model),
                "strategy_id": "fixture",
            }

            receipt = host.evaluate_request(request, project_root=project)

        self.assertEqual(receipt["status"], "blocked_environment")
        self.assertFalse(receipt["accepted"])
        self.assertFalse(receipt["formal_benchmark"]["executed"])

    def test_model_manifest_pins_exact_qwen35_08b_identity_and_detects_mutation(self):
        host = _host_evaluator_module()
        with tempfile.TemporaryDirectory() as temporary:
            model = Path(temporary) / "model"
            _write_fixture_qwen35_model(model)
            start = host.freeze_model_manifest(model)
            self.assertEqual(start["identity"]["model_type"], "qwen3_5")
            self.assertEqual(start["identity"]["hidden_size"], 1024)
            self.assertEqual(start["identity"]["num_hidden_layers"], 24)
            self.assertEqual(start["file_count"], 3)

            (model / "tokenizer.json").write_bytes(b'{"changed":true}')
            end = host.freeze_model_manifest(model)
            self.assertNotEqual(start["manifest_sha256"], end["manifest_sha256"])

    def test_request_validator_rejects_a_different_qwen35_size(self):
        host = _host_evaluator_module()
        with tempfile.TemporaryDirectory() as temporary:
            model = Path(temporary) / "model"
            _write_fixture_qwen35_model(model)
            config_path = model / "config.json"
            config = json.loads(config_path.read_text(encoding="utf-8"))
            config["text_config"]["hidden_size"] = 2048
            config_path.write_text(json.dumps(config), encoding="utf-8")
            source = (REPO / host.CANONICAL_SHADER).read_text(encoding="utf-8")
            request = {
                "schema_version": 1,
                "candidate_source": source,
                "kernel_path": str(REPO / host.CANONICAL_SHADER),
                "model_path": str(model),
                "strategy_id": "fixture",
            }
            problems = host.validate_request(request, project_root=REPO)

        self.assertTrue(any("Qwen3.5-0.8B" in problem for problem in problems))

    def test_isolated_snapshot_diff_can_only_be_the_shader(self):
        host = _host_evaluator_module()
        with tempfile.TemporaryDirectory() as temporary:
            temporary = Path(temporary)
            project = temporary / "project"
            metal = project / "crates/apxinf-metal/src"
            metal.mkdir(parents=True)
            baseline_source = (
                "#include <metal_stdlib>\n"
                "kernel void w8_rows_topk4() {}\n"
                "kernel void w8_final_topk4() {}\n"
            )
            candidate_source = baseline_source.replace("{}", "{ int x = 1; }", 1)
            (metal / "metal_w8.metal").write_text(baseline_source, encoding="utf-8")
            (metal / "metal_w8_bridge.mm").write_text("protected", encoding="utf-8")
            (project / "Cargo.toml").write_text("[workspace]\n", encoding="utf-8")
            scratch = temporary / "scratch"

            snapshot = host.prepare_isolated_roots(
                project_root=project,
                scratch_root=scratch,
                candidate_source=candidate_source,
            )

            self.assertEqual(snapshot["tree_differences"], [host.CANONICAL_SHADER])
            self.assertEqual(
                (
                    snapshot["candidate_root"]
                    / "crates/apxinf-metal/src/metal_w8_bridge.mm"
                ).read_text(encoding="utf-8"),
                "protected",
            )
            self.assertEqual(
                snapshot["candidate_shader_sha256"],
                host._sha256_text(candidate_source),
            )

    def test_source_manifest_ignores_build_outputs_but_detects_source_mutation(self):
        host = _host_evaluator_module()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "tree"
            source = root / "src/lib.rs"
            source.parent.mkdir(parents=True)
            source.write_text("pub fn value() -> u32 { 1 }\n", encoding="utf-8")
            start = host.freeze_source_manifest(root)

            build_output = root / "target-kersor/release/artifact"
            build_output.parent.mkdir(parents=True)
            build_output.write_bytes(b"ignored build output")
            after_build = host.freeze_source_manifest(root)
            self.assertEqual(start, after_build)

            source.write_text("pub fn value() -> u32 { 2 }\n", encoding="utf-8")
            end = host.freeze_source_manifest(root)
            self.assertNotEqual(start["manifest_sha256"], end["manifest_sha256"])

    def test_source_manifest_custodies_symlinked_directories_without_following_them(self):
        host = _host_evaluator_module()
        with tempfile.TemporaryDirectory() as temporary:
            temporary = Path(temporary)
            root = temporary / "tree"
            root.mkdir()
            first = temporary / "first"
            second = temporary / "second"
            first.mkdir()
            second.mkdir()
            (first / "outside.txt").write_text("one", encoding="utf-8")
            (second / "outside.txt").write_text("two", encoding="utf-8")
            linked = root / "linked"
            linked.symlink_to(first, target_is_directory=True)
            start = host.freeze_source_manifest(root)
            self.assertEqual(start["files"]["linked"], f"symlink:{first}")

            linked.unlink()
            linked.symlink_to(second, target_is_directory=True)
            end = host.freeze_source_manifest(root)
            self.assertNotEqual(start["manifest_sha256"], end["manifest_sha256"])

    def test_generation_receipt_is_bound_to_native_f32_metal_path(self):
        host = _host_evaluator_module()
        receipt = {
            "format": "apxinf-generation-v1",
            "model_type": "qwen3_5",
            "device": "cpu",
            "dtype": "fp32",
            "build": {
                "target_os": "macos",
                "target_arch": "aarch64",
                "matmul_feature": "accelerate",
                "metal_w8_lm_head": True,
            },
            "generated_token_ids": list(range(100)),
            "profile": {
                "output_tokens": 100,
                "ttft_ms": 20.0,
                "generation_tps": 100.0,
            },
        }
        parsed = host.validate_generation_receipt(
            receipt, expected_tokens=100, expect_metal=True
        )
        self.assertEqual(parsed["generated_token_ids"], list(range(100)))

        wrong_dtype = json.loads(json.dumps(receipt))
        wrong_dtype["dtype"] = "bf16"
        with self.assertRaisesRegex(ValueError, "native fp32"):
            host.validate_generation_receipt(
                wrong_dtype, expected_tokens=100, expect_metal=True
            )

        wrong_path = json.loads(json.dumps(receipt))
        wrong_path["build"]["metal_w8_lm_head"] = False
        with self.assertRaisesRegex(ValueError, "metal_w8_lm_head"):
            host.validate_generation_receipt(
                wrong_path, expected_tokens=100, expect_metal=True
            )

    def test_teacher_gate_requires_all_128_native_f32_rerank_matches(self):
        host = _host_evaluator_module()
        receipt = {
            "format": "apxinf-qwen35-metal-w8-top4-teacher-gate-v2",
            "comparisons": 128,
            "f32_reranked": {
                "matches": 128,
                "match_rate": 1.0,
                "mismatches": [],
            },
            "production_generation": {
                "comparisons": 10,
                "generated_token_ids": list(range(10)),
            },
            "quantization": {
                "layout": "hf-row-major",
                "scheme": "symmetric-int8-per-row-group",
                "group_size": 64,
                "scale_dtype": "f32",
            },
        }
        host.validate_teacher_receipt(receipt)
        receipt["f32_reranked"]["matches"] = 127
        receipt["f32_reranked"]["mismatches"] = [{"step": 9}]
        with self.assertRaisesRegex(ValueError, "128/128"):
            host.validate_teacher_receipt(receipt)

    def test_time_parser_requires_unique_rss_and_zero_swap_is_measurable(self):
        host = _host_evaluator_module()
        self.assertEqual(
            host.parse_time_l(b" 123456 maximum resident set size\n 0 swaps\n"),
            (123456, 0),
        )
        with self.assertRaisesRegex(ValueError, "unique"):
            host.parse_time_l(b" 123 maximum resident set size\n")

    def test_host_stops_a_command_as_soon_as_a_stream_exceeds_its_bound(self):
        host = _host_evaluator_module()
        environment = {
            "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
            "LANG": "C",
            "LC_ALL": "C",
        }
        payload_bytes = host.MAX_COMMAND_OUTPUT_BYTES + 1
        started = time.monotonic()
        with self.assertRaisesRegex(host.EvaluationError, "stream limit"):
            host._run_direct(
                [
                    "/usr/bin/python3",
                    "-I",
                    "-B",
                    "-c",
                    (
                        "import os,time;"
                        f"os.write(1,b'x'*{payload_bytes});"
                        "time.sleep(2)"
                    ),
                ],
                cwd=REPO,
                environment=environment,
                timeout_seconds=5,
            )
        self.assertLess(time.monotonic() - started, 1.5)

    def test_each_command_is_capped_by_the_internal_overall_deadline(self):
        host = _host_evaluator_module()
        environment = {
            "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
            "LANG": "C",
            "LC_ALL": "C",
        }
        started = time.monotonic()
        with host._evaluation_deadline(0.10):
            completed = host._run_direct(
                [
                    "/usr/bin/python3",
                    "-I",
                    "-B",
                    "-c",
                    "import time; time.sleep(2)",
                ],
                cwd=REPO,
                environment=environment,
                timeout_seconds=5,
            )
        self.assertTrue(completed["timed_out"])
        self.assertTrue(completed["overall_deadline_exhausted"])
        self.assertLess(time.monotonic() - started, 1.0)

    def test_command_timeout_terminates_the_complete_child_process_group(self):
        host = _host_evaluator_module()
        environment = {
            "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
            "LANG": "C",
            "LC_ALL": "C",
        }
        completed = host._run_direct(
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
            cwd=REPO,
            environment=environment,
            timeout_seconds=0.20,
        )
        self.assertTrue(completed["timed_out"])
        descendant_pid = int(completed["stdout"].decode("ascii").strip())
        with self.assertRaises(ProcessLookupError):
            os.kill(descendant_pid, 0)

    def test_evaluator_cancellation_terminates_its_active_command_tree(self):
        host_path = WORKFLOW_DIR / "host_evaluator.py"
        with tempfile.TemporaryDirectory() as temporary:
            pid_path = Path(temporary) / "command-pids.json"
            command = (
                "import json,os,pathlib,subprocess,sys,time;"
                "child=subprocess.Popen([sys.executable,'-I','-B','-c',"
                "'import time;time.sleep(30)']);"
                f"pathlib.Path({str(pid_path)!r}).write_text("
                "json.dumps([os.getpid(),child.pid]));time.sleep(30)"
            )
            helper = (
                "import importlib.util,os,pathlib,sys;"
                f"p=pathlib.Path({str(host_path)!r});"
                "s=importlib.util.spec_from_file_location('host',p);"
                "m=importlib.util.module_from_spec(s);s.loader.exec_module(m);"
                "m._install_cancellation_handlers();"
                "m._run_direct([sys.executable,'-I','-B','-c',"
                f"{command!r}],cwd=pathlib.Path({str(REPO)!r}),"
                "environment={'PATH':'/usr/bin:/bin:/usr/sbin:/sbin',"
                "'LANG':'C','LC_ALL':'C'},timeout_seconds=30)"
            )
            evaluator = subprocess.Popen(
                ["/usr/bin/python3", "-I", "-B", "-c", helper],
                cwd=REPO,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            pids = None
            try:
                deadline = time.monotonic() + 3.0
                while time.monotonic() < deadline:
                    if pid_path.is_file():
                        pids = json.loads(pid_path.read_text(encoding="utf-8"))
                        break
                    if evaluator.poll() is not None:
                        break
                    time.sleep(0.02)
                if pids is None:
                    _stdout, stderr = evaluator.communicate(timeout=1.0)
                    self.fail(stderr.decode())
                evaluator.terminate()
                evaluator.communicate(timeout=3.0)
                for pid in pids:
                    with self.assertRaises(ProcessLookupError):
                        os.kill(pid, 0)
            finally:
                if evaluator.poll() is None:
                    evaluator.kill()
                    evaluator.communicate()
                if pids:
                    try:
                        os.killpg(pids[0], 9)
                    except ProcessLookupError:
                        pass

    def test_workflow_host_fixture_uses_read_only_agent_and_fixed_evaluator(self):
        fixture = WORKFLOW_DIR / "fixtures/workflow_host_smoke.mjs"
        completed = subprocess.run(
            ["/opt/homebrew/bin/node", str(fixture), str(_kersor_root())],
            cwd=REPO,
            check=False,
            capture_output=True,
            text=True,
            env={**os.environ, "NODE_NO_WARNINGS": "1"},
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        result = json.loads(completed.stdout)
        self.assertEqual(result["run_status"], "completed")
        self.assertEqual(result["workflow_status"], "accepted")
        self.assertEqual(result["agent_calls"], 1)
        self.assertEqual(result["agent_transaction"], None)
        self.assertEqual(result["evaluator_calls"], 1)
        self.assertEqual(result["filesystem_policy"], "read-only")
        self.assertEqual(result["network_policy"], "denied")
        self.assertEqual(
            result["evaluator_argv_prefix"],
            [
                "/usr/bin/python3",
                "-I",
                "-B",
                str(WORKFLOW_DIR / "host_evaluator.py"),
                "--request-json",
            ],
        )
        self.assertEqual(result["evaluator_cwd"], str(REPO))

    def test_authoring_checker_has_only_the_documented_dsh_evaluate_block(self):
        checker = _kersor_root() / "scripts/check-workflow-syntax.py"
        python = Path("/opt/homebrew/bin/python3.13")
        if not python.is_file():
            raise unittest.SkipTest("KerSor Python 3.13 is not installed")
        completed = subprocess.run(
            [
                str(python),
                str(checker),
                str(WORKFLOW_DIR / "apxinf-metal-w8-head-optimization.js"),
            ],
            cwd=REPO,
            check=False,
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
        combined = completed.stdout + completed.stderr
        self.assertEqual(completed.returncode, 1)
        self.assertEqual(combined.count("FAIL:"), 1, combined)
        self.assertIn(
            "evaluate() is not available in the DSH Workflow script body",
            combined,
        )

    def test_workflow_returns_before_agent_when_external_model_input_is_missing(self):
        fixture = WORKFLOW_DIR / "fixtures/workflow_host_smoke.mjs"
        completed = subprocess.run(
            [
                "/opt/homebrew/bin/node",
                str(fixture),
                str(_kersor_root()),
                "--missing-model",
            ],
            cwd=REPO,
            check=False,
            capture_output=True,
            text=True,
            env={**os.environ, "NODE_NO_WARNINGS": "1"},
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        result = json.loads(completed.stdout)
        self.assertEqual(result["workflow_status"], "needs_model_path")
        self.assertEqual(result["agent_calls"], 0)
        self.assertEqual(result["evaluator_calls"], 0)

    def test_workflow_returns_before_agent_when_model_path_is_relative(self):
        fixture = WORKFLOW_DIR / "fixtures/workflow_host_smoke.mjs"
        completed = subprocess.run(
            [
                "/opt/homebrew/bin/node",
                str(fixture),
                str(_kersor_root()),
                "--relative-model",
            ],
            cwd=REPO / "crates/apxinf-metal/src",
            check=False,
            capture_output=True,
            text=True,
            env={**os.environ, "NODE_NO_WARNINGS": "1"},
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        result = json.loads(completed.stdout)
        self.assertEqual(result["workflow_status"], "needs_model_path")
        self.assertEqual(result["agent_calls"], 0)
        self.assertEqual(result["evaluator_calls"], 0)

    def test_same_bytes_schedule_retry_skips_agent_and_preserves_candidate_hash(self):
        fixture = WORKFLOW_DIR / "fixtures/workflow_host_smoke.mjs"
        completed = subprocess.run(
            [
                "/opt/homebrew/bin/node",
                str(fixture),
                str(_kersor_root()),
                "--same-bytes-retry",
            ],
            cwd=REPO,
            check=False,
            capture_output=True,
            text=True,
            env={**os.environ, "NODE_NO_WARNINGS": "1"},
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        result = json.loads(completed.stdout)
        self.assertEqual(result["workflow_status"], "accepted")
        self.assertEqual(result["agent_calls"], 0)
        self.assertEqual(result["evaluator_calls"], 1)
        self.assertTrue(result["retry_used"])
        self.assertEqual(
            result["request_candidate_source_sha256"],
            result["expected_candidate_source_sha256"],
        )

    def test_replacement_receipt_retains_exact_candidate_for_a_same_bytes_retry(self):
        fixture = WORKFLOW_DIR / "fixtures/workflow_host_smoke.mjs"
        completed = subprocess.run(
            [
                "/opt/homebrew/bin/node",
                str(fixture),
                str(_kersor_root()),
                "--replacement-required",
            ],
            cwd=REPO,
            check=False,
            capture_output=True,
            text=True,
            env={**os.environ, "NODE_NO_WARNINGS": "1"},
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        result = json.loads(completed.stdout)
        self.assertEqual(result["workflow_status"], "replacement_required")
        self.assertFalse(result["workflow_ok"])
        self.assertEqual(result["retry_candidate_source"], result["expected_source"])
        self.assertEqual(
            result["retry_candidate_sha256"],
            result["expected_candidate_source_sha256"],
        )

    def test_same_bytes_retry_rejects_a_host_receipt_for_a_different_candidate(self):
        fixture = WORKFLOW_DIR / "fixtures/workflow_host_smoke.mjs"
        completed = subprocess.run(
            [
                "/opt/homebrew/bin/node",
                str(fixture),
                str(_kersor_root()),
                "--same-bytes-retry",
                "--tamper-candidate-hash",
            ],
            cwd=REPO,
            check=False,
            capture_output=True,
            text=True,
            env={**os.environ, "NODE_NO_WARNINGS": "1"},
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        result = json.loads(completed.stdout)
        self.assertEqual(result["workflow_status"], "invalid_host_receipt")
        self.assertEqual(result["agent_calls"], 0)
        self.assertFalse(result["workflow_ok"])
        self.assertIsNone(result["best_kernel_code"])

    def test_workflow_rejects_tampered_host_receipt_and_command_custody(self):
        fixture = WORKFLOW_DIR / "fixtures/workflow_host_smoke.mjs"
        for mode in (
            "--tamper-gate",
            "--tamper-command",
            "--tamper-artifact",
            "--tamper-build",
            "--tamper-block",
            "--missing-field",
            "--tamper-source-end",
            "--tamper-model-end",
            "--tamper-toolchain-end",
            "--non-command-v1",
            "--evaluator-timeout",
            "--stream-violation",
            "--tamper-candidate-hash",
        ):
            with self.subTest(mode=mode):
                completed = subprocess.run(
                    [
                        "/opt/homebrew/bin/node",
                        str(fixture),
                        str(_kersor_root()),
                        mode,
                    ],
                    cwd=REPO,
                    check=False,
                    capture_output=True,
                    text=True,
                    env={**os.environ, "NODE_NO_WARNINGS": "1"},
                )
                self.assertEqual(completed.returncode, 0, completed.stderr)
                result = json.loads(completed.stdout)
                self.assertEqual(result["workflow_status"], "invalid_host_receipt")
                self.assertFalse(result["workflow_ok"])
                self.assertIsNone(result["best_kernel_code"])


if __name__ == "__main__":
    unittest.main()
