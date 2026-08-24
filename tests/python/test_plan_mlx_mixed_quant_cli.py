from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

from tests.python.test_build_mlx_bundle import (
    ASYNC_CHAT_PROMPT_IDS,
    FIXTURE_ASYNC_TEACHER_IDS,
    HYBRID_REVISION,
    ids_sha256,
    make_selective_source,
)
from tests.python.test_plan_mlx_mixed_quant import (
    QUALITY_SUITE_SHA256,
    TRACE as POLICY_TRACE,
    bound_divergent_observation,
)


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "scripts/plan_mlx_mixed_quant.py"
SPEC = importlib.util.spec_from_file_location("plan_mlx_mixed_quant", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class MixedQuantCliTests(unittest.TestCase):
    def test_init_scans_source_offline_and_writes_canonical_policy_no_replace(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            source = make_selective_source(root)
            trace_path = root / "trace.json"
            trace_path.write_text(
                json.dumps(
                    {
                        "api": "mlx_lm.generate.generate_step",
                        "semantics": "mlx-generate-step-argmax-v1",
                        "prompt_token_ids": ASYNC_CHAT_PROMPT_IDS,
                        "teacher_token_ids": FIXTURE_ASYNC_TEACHER_IDS,
                        "teacher_ids_sha256": ids_sha256(FIXTURE_ASYNC_TEACHER_IDS),
                        "teacher_steps": 128,
                        "free_run_steps": 128,
                        "repeat_count": 2,
                    },
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            quality_suite = {
                "format": "apxinf-mlx-mixed-quant-quality-suite-v1",
                "trace_sha256": MODULE.POLICY_API.object_sha256(
                    json.loads(trace_path.read_text(encoding="utf-8"))
                ),
            }
            quality_suite_path = root / "quality-suite.json"
            quality_suite_path.write_text(
                json.dumps(quality_suite, sort_keys=True), encoding="utf-8"
            )
            output = root / "initial-policy.json"
            arguments = [
                "init",
                "--source-dir",
                str(source),
                "--repo-id",
                "Qwen/Qwen3.5-0.8B",
                "--revision",
                HYBRID_REVISION,
                "--trace-contract",
                str(trace_path),
                "--quality-suite",
                str(quality_suite_path),
                "--output",
                str(output),
            ]
            stdout = io.StringIO()
            stderr = io.StringIO()

            with redirect_stdout(stdout), redirect_stderr(stderr):
                result = MODULE.main(arguments)

            self.assertEqual(result, 0)
            self.assertEqual(stderr.getvalue(), "")
            document = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(document["search_receipt"]["status"], "initial-all-w4")
            self.assertEqual(
                document["policy"]["quality_suite_sha256"],
                MODULE.POLICY_API.object_sha256(quality_suite),
            )
            self.assertEqual(len(document["policy"]["candidate_modules"]), 3)
            self.assertEqual(
                output.read_bytes(),
                MODULE.POLICY_API.canonical_bytes(document) + b"\n",
            )
            original = output.read_bytes()
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                second = MODULE.main(arguments)
            self.assertEqual(second, 2)
            self.assertEqual(output.read_bytes(), original)

    def test_advance_consumes_bound_observation_and_writes_one_tier_change(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            trace = json.loads(json.dumps(POLICY_TRACE))
            path = "language_model.model.layers.0.mlp.down_proj"
            initial = MODULE.POLICY_API.create_initial_policy_document(
                {
                    "repo_id": "Qwen/Qwen3.5-0.8B",
                    "revision": HYBRID_REVISION,
                    "source_manifest_sha256": "0" * 64,
                    "config_sha256": "1" * 64,
                    "language_schema_sha256": "2" * 64,
                    "language_tensor_count": 1,
                },
                [{"path": path, "dtype": "BF16", "shape": [2, 64]}],
                trace,
                QUALITY_SUITE_SHA256,
            )
            policy_path = root / "policy.json"
            policy_path.write_bytes(MODULE.POLICY_API.canonical_bytes(initial) + b"\n")
            observation = bound_divergent_observation(initial, path)
            observation_path = root / "observation.json"
            observation_path.write_bytes(
                MODULE.POLICY_API.canonical_bytes(observation) + b"\n"
            )
            output = root / "w8-policy.json"

            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                result = MODULE.main(
                    [
                        "advance",
                        "--policy",
                        str(policy_path),
                        "--observation",
                        str(observation_path),
                        "--output",
                        str(output),
                    ]
                )

            self.assertEqual(result, 0)
            advanced = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(
                advanced["policy"]["quantization"]["overrides"][0]["tier"],
                "w8",
            )
            self.assertEqual(
                advanced["search_receipt"]["decision"]["changed_module_count"],
                1,
            )

            observation["format"] = "apxinf-mlx-mixed-quant-observation-v1"
            observation_path.write_bytes(
                MODULE.POLICY_API.canonical_bytes(observation) + b"\n"
            )
            rejected_output = root / "rejected-v1-policy.json"
            stderr = io.StringIO()
            with redirect_stdout(io.StringIO()), redirect_stderr(stderr):
                rejected = MODULE.main(
                    [
                        "advance",
                        "--policy",
                        str(policy_path),
                        "--observation",
                        str(observation_path),
                        "--output",
                        str(rejected_output),
                    ]
                )

            self.assertEqual(rejected, 2)
            self.assertFalse(rejected_output.exists())
            error = json.loads(stderr.getvalue())
            self.assertEqual(error["format"], MODULE.ERROR_FORMAT)
            self.assertIn("format drifted", error["error"]["message"])


if __name__ == "__main__":
    unittest.main()
