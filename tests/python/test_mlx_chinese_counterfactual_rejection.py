import hashlib
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
DOC_ROOT = ROOT / "doc/20260823-qwen35-macos-bringup"
POLICY_PATH = (
    DOC_ROOT / "qwen35-0.8b-mlx-w8-g64-chinese-top1-counterfactual-policy-v3.json"
)
DIAGNOSTIC_PATH = (
    DOC_ROOT / "qwen35-hybrid-w8-bf16-g64-chinese-state-aligned-diagnostic-v1.json"
)
REJECTION_PATH = (
    DOC_ROOT / "qwen35-chinese-top3-o-proj-counterfactual-rejection-v1.json"
)
RUNBOOK_PATH = DOC_ROOT / "qwen35-chinese-top1-counterfactual-v3-runbook.md"


def canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class MlxChineseCounterfactualRejectionEvidenceTest(unittest.TestCase):
    def test_rejection_evidence_closes_the_exact_top3_o_proj_subset(self) -> None:
        evidence = json.loads(REJECTION_PATH.read_text(encoding="utf-8"))
        claimed_content_sha256 = evidence.pop("content_sha256")

        self.assertEqual(
            hashlib.sha256(canonical_bytes(evidence)).hexdigest(),
            claimed_content_sha256,
        )
        self.assertEqual(
            evidence["status"], "REAL_REJECTED_CANONICAL_STEP9"
        )
        self.assertEqual(evidence["screen"]["case_count"], 7)
        self.assertTrue(
            evidence["screen"]["exhaustive_nonempty_subsets_of_diagnostic_top3"]
        )
        observed_subsets = {
            tuple(case["diagnostic_ranks"])
            for case in evidence["screen"]["cases"]
        }
        self.assertEqual(
            observed_subsets,
            {(1,), (2,), (3,), (1, 2), (1, 3), (2, 3), (1, 2, 3)},
        )
        shared = evidence["screen"]["shared_result"]
        self.assertEqual(shared["mismatch_indices"], [9])
        self.assertEqual(shared["expected_token_id_at_step9"], 25677)
        self.assertEqual(shared["actual_token_id_at_step9"], 248046)
        self.assertTrue(shared["token_pattern_observed_across_all_seven_cases"])
        self.assertTrue(shared["not_a_per_case_repeat_claim"])
        self.assertNotIn("repeat_count", shared)
        self.assertNotIn("repeated_identically", shared)

        observed_run_counts = {
            case["case_id"]: case["observed_run_count"]
            for case in evidence["screen"]["cases"]
        }
        self.assertEqual(observed_run_counts["rank1-l19"], 2)
        self.assertEqual(
            {
                count
                for case_id, count in observed_run_counts.items()
                if case_id != "rank1-l19"
            },
            {1},
        )
        rank1 = next(
            case
            for case in evidence["screen"]["cases"]
            if case["case_id"] == "rank1-l19"
        )
        self.assertEqual(
            [observation["kind"] for observation in rank1["observations"]],
            [
                "real-build-pre-save-teacher-repeat1",
                "independent-first10-reproduction",
            ],
        )
        self.assertTrue(
            rank1["observations"][0]["actual_token_id_was_not_emitted_by_cli_error"]
        )
        parent_control = evidence["parent_control"]
        self.assertEqual(parent_control["observed_run_count"], 1)
        self.assertFalse(parent_control["repeatability_claim"])
        self.assertNotIn("repeat_count", parent_control)
        self.assertNotIn("repeated_identically", parent_control)

        decision = evidence["decision"]
        self.assertEqual(decision["top3_o_proj_restore"], "STOP")
        self.assertEqual(decision["top4_screen"], "DO_NOT_RUN")
        self.assertEqual(decision["additional_combinations"], "DO_NOT_RUN")
        self.assertFalse(decision["candidate_target_published"])
        self.assertFalse(decision["four_prompt_quality_gate_run"])
        self.assertFalse(decision["formal_performance_claim"])
        self.assertFalse(decision["general_parity_claim"])

    def test_rejection_binds_unchanged_frozen_inputs(self) -> None:
        evidence = json.loads(REJECTION_PATH.read_text(encoding="utf-8"))
        custody = evidence["custody"]
        self.assertEqual(
            file_sha256(POLICY_PATH),
            "e8b5ea2e3a5804772f5c1a3c9936abdc2eabde08b470dc49bebbfd8b02d0df7a",
        )
        policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))["policy"]
        self.assertEqual(
            hashlib.sha256(canonical_bytes(policy)).hexdigest(),
            "7030fe5a7c4dd55cbf158750e9da3a67c7f8e65944b8f8835c75b1093e12eec9",
        )
        self.assertEqual(
            file_sha256(DIAGNOSTIC_PATH),
            "1b30a3a7f6d609a8265112bde3189b7638a9072561530b852ec86dbc4794b73d",
        )
        self.assertEqual(
            custody["builder"]["sha256"],
            file_sha256(ROOT / "scripts/build_mlx_bundle.py"),
        )
        self.assertTrue(evidence["policy_status"]["frozen_policy_unchanged"])
        self.assertEqual(
            evidence["policy_status"]["effective_status"],
            "REAL_REJECTED_CANONICAL_STEP9",
        )

    def test_runbook_retains_no_live_build_or_quality_command(self) -> None:
        runbook = RUNBOOK_PATH.read_text(encoding="utf-8")
        self.assertIn("REAL_REJECTED_CANONICAL_STEP9", runbook)
        self.assertIn(REJECTION_PATH.name, runbook)
        self.assertIn("top3 `self_attn.o_proj` restore family: **STOP**", runbook)
        self.assertNotIn("scripts/build_mlx_bundle.py \\", runbook)
        self.assertNotIn("scripts/run_mlx_multi_prompt_quality.py \\", runbook)


if __name__ == "__main__":
    unittest.main()
