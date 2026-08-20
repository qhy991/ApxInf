from __future__ import annotations

import importlib.util
import json
import math
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


score_module = load_module("score_submission", HERE / "score_submission.py")
efficiency_module = load_module("compute_efficiency", HERE / "compute_efficiency.py")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


class EvaluationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract = read_json(HERE / "contract-v1.json")
        cls.vllm = read_json(HERE / "fixtures" / "vllm-reference.json")
        cls.apxinf = read_json(HERE / "fixtures" / "apxinf-marlin-current.json")

    def test_vllm_public_calibration_score_is_frozen(self):
        result = score_module.score_submission(
            self.contract, self.vllm, "public_calibration"
        )
        self.assertTrue(result["eligible"])
        self.assertAlmostEqual(result["leaderboard_score"], 86.31210181113707)
        self.assertEqual(result["section_scores"]["ttft"], 30.0)
        self.assertAlmostEqual(result["section_scores"]["tpot"], 6.312101811137062)

    def test_current_apxinf_receives_public_calibration_score(self):
        result = score_module.score_submission(
            self.contract, self.apxinf, "public_calibration"
        )
        self.assertTrue(result["eligible"])
        self.assertAlmostEqual(result["leaderboard_score"], 65.00528137194189)
        self.assertAlmostEqual(result["section_scores"]["ttft"], 1.0052813719418892)
        self.assertEqual(result["section_scores"]["tpot"], 20.0)
        self.assertEqual(result["eligibility_failures"], [])

    def test_midterm_requires_hidden_correctness(self):
        result = score_module.score_submission(
            self.contract, self.vllm, "midterm_leaderboard"
        )
        self.assertFalse(result["eligible"])
        self.assertIn(
            "hidden_pass_rate_below_threshold_or_missing", result["eligibility_failures"]
        )

    def test_efficiency_separates_proxy_from_profiled_metrics(self):
        result = efficiency_module.compute_efficiency(self.contract, self.apxinf)
        decode = result["cells"]["text-perf-1024"]["decode"]
        self.assertGreater(decode["minimum_model_bwu_pct"], 80.0)
        self.assertLess(decode["minimum_model_bwu_pct"], 100.0)
        self.assertLess(decode["estimated_mfu_bf16_equivalent_pct"], 5.0)
        self.assertIsNone(decode["profiled"])
        self.assertTrue(math.isfinite(decode["estimated_tflops"]))

    def test_context_score_is_logarithmic_and_capped(self):
        candidate = json.loads(json.dumps(self.vllm))
        candidate["context"]["max_verified_prompt_tokens"] = 8192
        result = score_module.score_submission(
            self.contract, candidate, "public_calibration"
        )
        self.assertEqual(result["section_scores"]["context"], 9.0)
        candidate["context"]["max_verified_prompt_tokens"] = 100_000
        result = score_module.score_submission(
            self.contract, candidate, "public_calibration"
        )
        self.assertEqual(result["section_scores"]["context"], 15.0)


if __name__ == "__main__":
    unittest.main()
