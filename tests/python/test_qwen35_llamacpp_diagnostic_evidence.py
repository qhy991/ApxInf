from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
SUMMARY_RELATIVE = (
    "crates/apxinf-metal/evidence/llama-cpp/"
    "qwen35-0.8b-apxinf-vs-llamacpp-raw13-free128-"
    "diagnostic-summary-v2-20260825.json"
)
SUMMARY_PATH = ROOT / SUMMARY_RELATIVE
MODULE_PATH = ROOT / "scripts" / "validate_qwen35_llamacpp_diagnostic_evidence.py"


def load_module():
    spec = importlib.util.spec_from_file_location(
        "validate_qwen35_llamacpp_diagnostic_evidence_for_tests", MODULE_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module


def prepared_summary(module):
    value = module.load_json(SUMMARY_PATH)
    value["_summary_relative_path"] = SUMMARY_RELATIVE
    return value


def rehash_summary(module, value):
    value.pop("_summary_relative_path", None)
    value.pop("content_sha256", None)
    value["content_sha256"] = module.object_sha256(value)
    value["_summary_relative_path"] = SUMMARY_RELATIVE


class Qwen35LlamaCppDiagnosticEvidenceTests(unittest.TestCase):
    def test_checked_in_summary_and_all_referenced_receipts_validate(self):
        completed = subprocess.run(
            [
                "/usr/bin/python3",
                "-I",
                "-B",
                str(MODULE_PATH),
                "--summary",
                str(SUMMARY_PATH),
            ],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        receipt = json.loads(completed.stdout)
        self.assertTrue(receipt["valid"])
        self.assertFalse(receipt["formal_performance_result"])
        self.assertEqual(
            receipt["canonical_free128_sha256"],
            "2042d5522ed5e768b938ac3fd9e19d3936dfc6e685fc157d41a8e8132c5b42fe",
        )

    def test_summary_records_exact_free_run_without_overclaiming_teacher_quality(self):
        summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
        free_run = summary["quality"]["free_run"]
        self.assertTrue(free_run["all_four_trajectories_identical"])
        self.assertEqual(free_run["position_match_count"], 128)
        self.assertEqual(free_run["position_match_ratio"], 1.0)
        self.assertEqual(free_run["exact_prefix_tokens"], 128)
        self.assertIsNone(free_run["first_mismatch"])
        self.assertFalse(free_run["general_quality_parity_claim_allowed"])
        teacher = summary["quality"]["teacher_forced"]
        self.assertFalse(teacher["llama_cpp_cross_runtime_measured"])
        self.assertFalse(teacher["cross_runtime_teacher_exactness_claim_allowed"])

    def test_rehashed_semantic_and_custody_drift_are_rejected(self):
        module = load_module()
        mutations = (
            (
                "formal claim",
                lambda value: value["classification"].__setitem__(
                    "formal_performance_result", True
                ),
            ),
            (
                "F32 ratio direction",
                lambda value: value["comparison_lanes"]["f32_reference"][
                    "diagnostic_ratios"
                ].__setitem__("llama_cpp_generation_tps_over_apxinf", 0.5),
            ),
            (
                "teacher exactness",
                lambda value: value["quality"]["teacher_forced"].__setitem__(
                    "cross_runtime_teacher_exactness_claim_allowed", True
                ),
            ),
            (
                "memory comparison",
                lambda value: value["placement_and_memory"].__setitem__(
                    "cross_runtime_memory_ratio_allowed", True
                ),
            ),
            (
                "timing formula",
                lambda value: value["scope"]["timing_definitions"].__setitem__(
                    "generation_tps", "128000/(total_latency_ms-ttft_ms)"
                ),
            ),
            (
                "model hash binding",
                lambda value: value["model_artifacts"]["llama_cpp_f32"].__setitem__(
                    "hash_binding", "runner-pinned-fd-sha256"
                ),
            ),
            (
                "non-formal reasons",
                lambda value: value["classification"].__setitem__(
                    "reasons_not_formal", []
                ),
            ),
            (
                "llama.cpp source tree",
                lambda value: value["llama_cpp_build_integrity"].__setitem__(
                    "source_tree", "0" * 40
                ),
            ),
            (
                "runner binary size",
                lambda value: value["llama_cpp_build_integrity"][
                    "runner_binary"
                ].__setitem__("size", 1),
            ),
            (
                "runner receipt binding",
                lambda value: value["llama_cpp_build_integrity"][
                    "runner_binary"
                ].__setitem__("receipt_binding", "self-reported"),
            ),
        )
        for label, mutate in mutations:
            with self.subTest(label=label):
                changed = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
                mutate(changed)
                rehash_summary(module, changed)
                with self.assertRaises(module.EvidenceError):
                    module.validate_summary(changed, ROOT)

    def test_rehashed_receipt_hash_drift_is_rejected(self):
        module = load_module()
        summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
        changed = dict(summary["receipt_integrity"]["llama_cpp_f32_cpu"])
        changed["sha256"] = "0" * 64
        with self.assertRaisesRegex(module.EvidenceError, "SHA-256 mismatch"):
            module.validate_integrity_entry(ROOT, changed, "llama.cpp F32 receipt")

    def test_duplicate_json_keys_are_rejected(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "duplicate.json"
            path.write_text('{"format":"one","format":"two"}\n', encoding="utf-8")
            with self.assertRaisesRegex(module.EvidenceError, "duplicate key"):
                module.load_json(path)


if __name__ == "__main__":
    unittest.main()
