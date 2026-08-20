from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "orchestrate_cohort_test", HERE / "teacher" / "orchestrate_cohort.py"
)
assert spec and spec.loader
orchestrator = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = orchestrator
spec.loader.exec_module(orchestrator)


def entry(name, role, revision="a" * 40):
    return {
        "name": name,
        "role": role,
        "repository": "/not-read-during-structural-validation",
        "checkout_revision": revision,
        "serve_argv": ["serve"],
        "runner_argv": ["run"],
        "health_url": "http://127.0.0.1:8000/health",
    }


class TeacherOrchestrationTest(unittest.TestCase):
    def test_plan_requires_one_control_and_full_commit_ids(self):
        valid = {
            "schema": orchestrator.PLAN_SCHEMA,
            "round_id": "midterm-2026-v1",
            "profile": "midterm_leaderboard",
            "entries": [entry("vllm", "control"), entry("pr-001", "candidate")],
        }
        orchestrator.validate_plan(valid)

        two_controls = {**valid, "entries": [*valid["entries"], entry("vllm-2", "control")]}
        with self.assertRaisesRegex(ValueError, "exactly one"):
            orchestrator.validate_plan(two_controls)

        short_sha = {
            **valid,
            "entries": [entry("vllm", "control"), entry("pr-001", "candidate", "abc123")],
        }
        with self.assertRaisesRegex(ValueError, "full commit SHA"):
            orchestrator.validate_plan(short_sha)


if __name__ == "__main__":
    unittest.main()
