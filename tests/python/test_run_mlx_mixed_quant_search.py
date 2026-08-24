from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
POLICY_PATH = ROOT / "scripts/mlx_mixed_quant_policy.py"
RUNNER_PATH = ROOT / "scripts/run_mlx_mixed_quant_search.py"


def load_module(name: str, path: Path) -> object:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


POLICY = load_module("test_mlx_mixed_quant_policy_for_runner", POLICY_PATH)
RUNNER = load_module("_apxinf_mixed_quant_runner_under_test", RUNNER_PATH)


def object_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


SOURCE = {
    "repo_id": "Qwen/Qwen3.5-0.8B",
    "revision": "2fc06364715b967f1860aea9cf38778875588b17",
    "source_manifest_sha256": "0" * 64,
    "config_sha256": "1" * 64,
    "language_schema_sha256": "2" * 64,
    "language_tensor_count": 7,
}
CANDIDATES = [
    {
        "path": "language_model.model.layers.0.mlp.down_proj",
        "dtype": "BF16",
        "shape": [128, 64],
    },
    {
        "path": "language_model.model.layers.0.self_attn.q_proj",
        "dtype": "BF16",
        "shape": [64, 64],
    },
]
TEACHER_IDS = list(range(128))
TRACE = {
    "api": "mlx_lm.generate.generate_step",
    "semantics": "mlx-generate-step-argmax-v1",
    "prompt_token_ids": list(POLICY.CANONICAL_CHAT_PROMPT_IDS),
    "teacher_token_ids": TEACHER_IDS,
    "teacher_ids_sha256": object_sha256(TEACHER_IDS),
    "teacher_steps": 128,
    "free_run_steps": 128,
    "repeat_count": 2,
}


def quality_suite() -> dict[str, object]:
    return {
        "format": "apxinf-mlx-mixed-quant-quality-suite-v1",
        "source": SOURCE,
        "candidate_modules_sha256": object_sha256(CANDIDATES),
        "trace_sha256": object_sha256(TRACE),
        "generation": {
            "api": "mlx_lm.generate.generate_step",
            "semantics": "mlx-generate-step-argmax-v1",
            "sampler": "mx.argmax(logprobs,axis=-1)",
            "teacher_steps": 128,
            "async_free_run_steps": 128,
            "repeat_count": 2,
            "stop_on_eos": False,
        },
        "screening": {
            "state_alignment": "prompt-plus-bf16-teacher-prefix-v1",
            "steps": 32,
            "grouping": "layer-family-v1",
            "group_map": [
                {
                    "path": CANDIDATES[0]["path"],
                    "layer": 0,
                    "family": "mlp",
                },
                {
                    "path": CANDIDATES[1]["path"],
                    "layer": 0,
                    "family": "self_attn",
                },
            ],
            "hook": "module-output-counterfactual-on-bf16-input-v1",
            "tensor_dtype": "float32",
            "reduction_order": "path-token-output-v1",
            "hidden_error_formula": "mean-absolute-relative-ppm-v1",
            "top1_margin_formula": "top2-margin-erosion-ppm-v1",
            "score_formula": "hidden-error-plus-top1-margin-plus-flip-rate-v1",
            "aggregation": "sum-module-score-ppm-v1",
            "non_finite": "reject",
            "minimum_gate_improvement": 1,
            "unique_winner_delta": 1,
        },
        "search": {
            "max_counterfactuals_per_generation": 2,
            "allowed_transitions": ["w4-to-w8", "w8-to-bf16"],
            "changed_module_count": 1,
            "allow_combinations": False,
            "candidate_materialization": (
                "independent-saved-static-verified-reload-v1"
            ),
            "dynamic_module_replacement": False,
        },
        "multi_prompt_baseline": json.loads(json.dumps(RUNNER.W4_BASELINE)),
        "publication": {
            "evaluation_only": True,
            "publishable": False,
            "final_builder_required": True,
            "exact_scope": "single-frozen-canonical-trajectory-only-v1",
            "claims_general_parity": False,
            "default_ready": False,
            "formal_performance_claim": False,
        },
    }


def policy_document() -> dict[str, object]:
    suite = quality_suite()
    return POLICY.create_initial_policy_document(
        SOURCE, CANDIDATES, TRACE, object_sha256(suite)
    )


def exact_gate() -> dict[str, object]:
    return {
        "api": TRACE["api"],
        "semantics": TRACE["semantics"],
        "prompt_token_ids": TRACE["prompt_token_ids"],
        "teacher_forced_token_ids": [TEACHER_IDS, TEACHER_IDS],
        "async_free_run_token_ids": [TEACHER_IDS, TEACHER_IDS],
    }


def screen(score0: int = 30, score1: int = 10) -> dict[str, object]:
    scores = []
    for candidate, score_value in zip(CANDIDATES, [score0, score1], strict=True):
        scores.append(
            {
                "path": candidate["path"],
                "hidden_error_ppm": score_value,
                "top1_margin_erosion_ppm": 0,
                "top1_flip_rate_ppm": 0,
                "score_ppm": score_value,
            }
        )
    return {
        "format": "apxinf-mlx-mixed-quant-state-aligned-screen-v1",
        "steps": 32,
        "state_alignment": "prompt-plus-bf16-teacher-prefix-v1",
        "aggregate_score_ppm": score0 + score1,
        "module_scores": scores,
    }


def runtime_receipt() -> dict[str, object]:
    return {
        "python_executable_sha256": "5" * 64,
        "python_version": "3.14.3",
        "packages": [{"name": "mlx", "version": "0.32.1", "sha256": "6" * 64}],
        "offline": True,
        "network_blocked": True,
        "trust_remote_code": False,
    }


def program_receipt() -> dict[str, object]:
    artifacts = [
        {
            "path": "scripts/run_mlx_mixed_quant_search.py",
            "size": 123,
            "sha256": "7" * 64,
        }
    ]
    return {"artifacts": artifacts, "program_sha256": object_sha256(artifacts)}


class FakeBackend:
    def __init__(self) -> None:
        self.events: list[object] = []
        self.current_gate = exact_gate()
        self.reference_gate = exact_gate()
        self.current_screen = screen()
        self.counterfactual_screens: dict[str, dict[str, object]] = {}
        self.counterfactual_gates: dict[str, dict[str, object]] = {}
        self.counterfactual_builds = 0

    @staticmethod
    def _handle(name: str, policy_sha256: str) -> dict[str, object]:
        return {
            "handle_id": name,
            "manifest_sha256": hashlib.sha256(name.encode()).hexdigest(),
            "policy_sha256": policy_sha256,
            "evaluation_only": True,
            "publishable": False,
            "materialization": "independent-saved-static-verified-reload-v1",
        }

    def open_bf16_reference(self, certified: object) -> dict[str, object]:
        self.events.append("open-bf16")
        return self._handle("bf16-reference", certified.policy_sha256)

    def materialize_current(self, certified: object) -> dict[str, object]:
        self.events.append("materialize-current")
        return self._handle("current-candidate", certified.policy_sha256)

    def screen_state_aligned(
        self,
        reference: dict[str, object],
        candidate: dict[str, object],
        *,
        certified: object,
        transition: dict[str, object] | None,
    ) -> dict[str, object]:
        self.events.append(("screen", reference["handle_id"], candidate["handle_id"]))
        if transition is None:
            return json.loads(json.dumps(self.current_screen))
        path = str(transition["path"])
        return json.loads(json.dumps(self.counterfactual_screens[path]))

    def evaluate_gate(
        self,
        handle: dict[str, object],
        *,
        certified: object,
        role: str,
    ) -> dict[str, object]:
        self.events.append(("gate", role, handle["handle_id"]))
        if role == "bf16-reference":
            return json.loads(json.dumps(self.reference_gate))
        if role == "current-candidate":
            return json.loads(json.dumps(self.current_gate))
        path = str(handle["transition"]["path"])
        return json.loads(json.dumps(self.counterfactual_gates[path]))

    def materialize_counterfactual(
        self, certified: object, transition: dict[str, object]
    ) -> dict[str, object]:
        path = str(transition["path"])
        self.counterfactual_builds += 1
        self.events.append(("materialize-counterfactual", path, transition["to"]))
        handle = self._handle(
            f"counterfactual-{self.counterfactual_builds}-{path}-{transition['to']}",
            certified.policy_sha256,
        )
        handle["manifest_sha256"] = hashlib.sha256(
            f"counterfactual-{path}-{transition['to']}".encode()
        ).hexdigest()
        handle["transition"] = json.loads(json.dumps(transition))
        return handle

    def close(self, handle: dict[str, object]) -> None:
        self.events.append(("close", handle["handle_id"]))


class MixedQuantSearchRunnerTests(unittest.TestCase):
    def certified(self) -> object:
        document = policy_document()
        suite = quality_suite()
        return RUNNER.certify_documents_for_test(
            document,
            suite,
            policy_artifact_sha256=hashlib.sha256(
                POLICY.canonical_bytes(document) + b"\n"
            ).hexdigest(),
        )

    def test_exact_candidate_runs_screen_before_distinct_128_step_double_gates(
        self,
    ) -> None:
        backend = FakeBackend()
        certified = self.certified()

        observation = RUNNER.evaluate_certified_generation(
            certified,
            backend,
            runtime=runtime_receipt(),
            program=program_receipt(),
        )

        self.assertEqual(
            backend.events[:5],
            [
                "open-bf16",
                "materialize-current",
                ("screen", "bf16-reference", "current-candidate"),
                ("gate", "bf16-reference", "bf16-reference"),
                ("gate", "current-candidate", "current-candidate"),
            ],
        )
        self.assertNotEqual(
            observation["runner_receipt_body"]["bundles"]["bf16_reference"][
                "handle_id"
            ],
            observation["runner_receipt_body"]["bundles"]["current_candidate"][
                "handle_id"
            ],
        )
        self.assertEqual(observation["evaluator"], exact_gate())
        self.assertIsNone(observation["localization"])
        self.assertEqual(
            observation["runner_receipt_sha256"],
            object_sha256(observation["runner_receipt_body"]),
        )
        decision = observation["runner_receipt_body"]["decision"]
        self.assertEqual(decision["outcome"], "exact")
        self.assertIsNone(decision["stop_reason"])
        self.assertTrue(decision["exact_trajectory_claim"])
        self.assertEqual(
            decision["exact_scope"],
            "single-frozen-canonical-trajectory-only-v1",
        )
        self.assertFalse(decision["general_parity_claim"])
        self.assertFalse(decision["default_ready_claim"])
        self.assertFalse(decision["formal_performance_claim"])
        self.assertFalse(
            observation["runner_receipt_body"]["bundles"]["model_bundle_published"]
        )
        self.assertEqual(
            observation["runner_receipt_body"]["evaluation"]["attribution"][
                "multi_prompt_baseline"
            ]["evidence_content_sha256"],
            RUNNER.W4_BASELINE_EVIDENCE_SHA256,
        )
        planned = POLICY.advance_policy_document(policy_document(), observation)
        self.assertEqual(planned["search_receipt"]["status"], "exact-pareto")

    def test_divergence_builds_one_tier_counterfactual_and_gates_unique_winner(
        self,
    ) -> None:
        backend = FakeBackend()
        divergent = list(TEACHER_IDS)
        divergent[9] = 999
        backend.current_gate = {
            **exact_gate(),
            "teacher_forced_token_ids": [divergent, divergent],
            "async_free_run_token_ids": [divergent, divergent],
        }
        path = str(CANDIDATES[0]["path"])
        backend.counterfactual_screens[path] = screen(5, 10)
        backend.counterfactual_gates[path] = exact_gate()

        observation = RUNNER.evaluate_certified_generation(
            self.certified(),
            backend,
            runtime=runtime_receipt(),
            program=program_receipt(),
        )

        self.assertEqual(observation["runner_receipt_body"]["outcome"], "divergent")
        self.assertEqual(observation["localization"]["sensitive_module_path"], path)
        self.assertTrue(observation["localization"]["unique_top_candidate"])
        decision = observation["runner_receipt_body"]["decision"]
        self.assertEqual(decision["changed_module_count"], 1)
        self.assertEqual(
            decision["transition"], {"path": path, "from": "w4", "to": "w8"}
        )
        self.assertIsNone(decision["stop_reason"])
        self.assertFalse(decision["exact_trajectory_claim"])
        self.assertEqual(backend.counterfactual_builds, 2)
        screen_build = next(
            index
            for index, event in enumerate(backend.events)
            if event == ("materialize-counterfactual", path, "w8")
        )
        selected_gate = next(
            index
            for index, event in enumerate(backend.events)
            if type(event) is tuple and event[:2] == ("gate", "selected-counterfactual")
        )
        self.assertLess(screen_build, selected_gate)
        self.assertFalse(
            observation["runner_receipt_body"]["bundles"]["dynamic_module_replacement"]
        )
        self.assertFalse(
            observation["runner_receipt_body"]["bundles"]["model_bundle_published"]
        )
        planned = POLICY.advance_policy_document(policy_document(), observation)
        self.assertEqual(planned["search_receipt"]["status"], "advance")
        self.assertEqual(
            planned["policy"]["lineage"]["transition"],
            {"path": path, "from": "w4", "to": "w8"},
        )

    def test_teacher_exact_but_async_divergent_is_not_mislabeled_exact(self) -> None:
        backend = FakeBackend()
        divergent = list(TEACHER_IDS)
        divergent[31] = 999
        backend.current_gate = {
            **exact_gate(),
            "async_free_run_token_ids": [divergent, divergent],
        }
        path = str(CANDIDATES[0]["path"])
        backend.counterfactual_screens[path] = screen(5, 10)
        backend.counterfactual_gates[path] = exact_gate()

        observation = RUNNER.evaluate_certified_generation(
            self.certified(),
            backend,
            runtime=runtime_receipt(),
            program=program_receipt(),
        )

        decision = observation["runner_receipt_body"]["decision"]
        self.assertEqual(decision["outcome"], "divergent")
        self.assertFalse(decision["exact_trajectory_claim"])
        self.assertEqual(decision["changed_module_count"], 1)

    def test_tied_group_stops_without_guessing_or_building_counterfactual(self) -> None:
        backend = FakeBackend()
        divergent = list(TEACHER_IDS)
        divergent[7] = 999
        backend.current_gate = {
            **exact_gate(),
            "teacher_forced_token_ids": [divergent, divergent],
            "async_free_run_token_ids": [divergent, divergent],
        }
        backend.current_screen = screen(20, 20)

        observation = RUNNER.evaluate_certified_generation(
            self.certified(),
            backend,
            runtime=runtime_receipt(),
            program=program_receipt(),
        )

        self.assertIsNone(observation["localization"])
        decision = observation["runner_receipt_body"]["decision"]
        self.assertEqual(decision["stop_reason"], "no-unique-sensitive-module")
        self.assertEqual(decision["changed_module_count"], 0)
        self.assertEqual(backend.counterfactual_builds, 0)
        planned = POLICY.advance_policy_document(policy_document(), observation)
        self.assertEqual(planned["search_receipt"]["status"], "stop")

    def test_nondeterministic_repeats_stop_before_counterfactual_build(self) -> None:
        backend = FakeBackend()
        first = list(TEACHER_IDS)
        second = list(TEACHER_IDS)
        first[7] = 999
        second[8] = 999
        backend.current_gate = {
            **exact_gate(),
            "teacher_forced_token_ids": [first, first],
            "async_free_run_token_ids": [first, second],
        }

        observation = RUNNER.evaluate_certified_generation(
            self.certified(),
            backend,
            runtime=runtime_receipt(),
            program=program_receipt(),
        )

        self.assertIsNone(observation["localization"])
        decision = observation["runner_receipt_body"]["decision"]
        self.assertEqual(decision["outcome"], "nondeterministic")
        self.assertEqual(
            decision["stop_reason"], "nondeterministic-repeated-trajectories"
        )
        self.assertEqual(backend.counterfactual_builds, 0)
        planned = POLICY.advance_policy_document(policy_document(), observation)
        self.assertEqual(planned["search_receipt"]["status"], "stop")

    def test_next_generation_allows_only_w8_to_bf16_for_same_module(self) -> None:
        first_backend = FakeBackend()
        divergent = list(TEACHER_IDS)
        divergent[9] = 999
        first_backend.current_gate = {
            **exact_gate(),
            "teacher_forced_token_ids": [divergent, divergent],
            "async_free_run_token_ids": [divergent, divergent],
        }
        path = str(CANDIDATES[0]["path"])
        first_backend.counterfactual_screens[path] = screen(5, 10)
        first_backend.counterfactual_gates[path] = exact_gate()
        first_observation = RUNNER.evaluate_certified_generation(
            self.certified(),
            first_backend,
            runtime=runtime_receipt(),
            program=program_receipt(),
        )
        advanced = POLICY.advance_policy_document(policy_document(), first_observation)
        certified = RUNNER.certify_documents_for_test(
            advanced,
            quality_suite(),
            policy_artifact_sha256=hashlib.sha256(
                POLICY.canonical_bytes(advanced) + b"\n"
            ).hexdigest(),
        )
        second_backend = FakeBackend()
        second_backend.current_gate = {
            **exact_gate(),
            "teacher_forced_token_ids": [divergent, divergent],
            "async_free_run_token_ids": [divergent, divergent],
        }
        second_backend.counterfactual_screens[path] = screen(5, 10)
        second_backend.counterfactual_gates[path] = exact_gate()

        observation = RUNNER.evaluate_certified_generation(
            certified,
            second_backend,
            runtime=runtime_receipt(),
            program=program_receipt(),
        )

        self.assertEqual(
            observation["runner_receipt_body"]["decision"]["transition"],
            {"path": path, "from": "w8", "to": "bf16"},
        )

    def test_selected_counterfactual_failure_raises_and_closes_all_handles(
        self,
    ) -> None:
        backend = FakeBackend()
        divergent = list(TEACHER_IDS)
        divergent[9] = 999
        backend.current_gate = {
            **exact_gate(),
            "teacher_forced_token_ids": [divergent, divergent],
            "async_free_run_token_ids": [divergent, divergent],
        }
        path = str(CANDIDATES[0]["path"])
        backend.counterfactual_screens[path] = screen(5, 10)
        backend.counterfactual_gates[path] = backend.current_gate

        with self.assertRaisesRegex(RUNNER.SearchError, "128-step improvement"):
            RUNNER.evaluate_certified_generation(
                self.certified(),
                backend,
                runtime=runtime_receipt(),
                program=program_receipt(),
            )

        closes = [event for event in backend.events if event[0] == "close"]
        self.assertEqual(len(closes), 4)

    def test_policy_artifact_hash_is_recomputed_before_backend_use(self) -> None:
        with self.assertRaisesRegex(RUNNER.SearchError, "artifact SHA-256"):
            RUNNER.certify_documents_for_test(
                policy_document(),
                quality_suite(),
                policy_artifact_sha256="f" * 64,
            )

    def test_failed_gate_does_not_publish_even_an_observation_file(self) -> None:
        backend = FakeBackend()
        divergent = list(TEACHER_IDS)
        divergent[9] = 999
        backend.current_gate = {
            **exact_gate(),
            "teacher_forced_token_ids": [divergent, divergent],
            "async_free_run_token_ids": [divergent, divergent],
        }
        path = str(CANDIDATES[0]["path"])
        backend.counterfactual_screens[path] = screen(5, 10)
        backend.counterfactual_gates[path] = backend.current_gate
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "observation.json"
            with self.assertRaisesRegex(RUNNER.SearchError, "128-step improvement"):
                RUNNER.evaluate_certified_generation(
                    self.certified(),
                    backend,
                    runtime=runtime_receipt(),
                    program=program_receipt(),
                )
            self.assertFalse(output.exists())

    def test_real_failed_w4_multiprompt_evidence_is_structurally_consumable(
        self,
    ) -> None:
        evidence_path = (
            ROOT / "doc/20260823-qwen35-macos-bringup/"
            "qwen35-w4-multi-prompt-quality-v1.json"
        )
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))

        validated = RUNNER._validate_w4_baseline_evidence(evidence)

        self.assertEqual(validated["status"], "failed_comparison")
        self.assertEqual(
            validated["content_sha256"], RUNNER.W4_BASELINE_EVIDENCE_SHA256
        )

    def test_selected_rematerialization_manifest_drift_fails_closed(self) -> None:
        backend = FakeBackend()
        divergent = list(TEACHER_IDS)
        divergent[9] = 999
        backend.current_gate = {
            **exact_gate(),
            "teacher_forced_token_ids": [divergent, divergent],
            "async_free_run_token_ids": [divergent, divergent],
        }
        path = str(CANDIDATES[0]["path"])
        backend.counterfactual_screens[path] = screen(5, 10)
        backend.counterfactual_gates[path] = exact_gate()
        original = backend.materialize_counterfactual

        def drifting_materialization(
            certified: object, transition: dict[str, object]
        ) -> dict[str, object]:
            handle = original(certified, transition)
            if backend.counterfactual_builds == 2:
                handle["manifest_sha256"] = "e" * 64
            return handle

        backend.materialize_counterfactual = drifting_materialization

        with self.assertRaisesRegex(RUNNER.SearchError, "manifest drifted"):
            RUNNER.evaluate_certified_generation(
                self.certified(),
                backend,
                runtime=runtime_receipt(),
                program=program_receipt(),
            )

    def test_unprotected_runtime_receipt_does_not_claim_network_blocked(self) -> None:
        class FakeDistribution:
            version = "0.32.1"

            @staticmethod
            def read_text(name: str) -> str | None:
                return "mlx/__init__.py,sha256=fake,1\n" if name == "RECORD" else None

        fake_builder = type("FakeBuilder", (), {"PINNED_PACKAGES": {"mlx": "0.32.1"}})()
        with (
            mock.patch.object(RUNNER, "_load_builder_api", return_value=fake_builder),
            mock.patch.object(
                RUNNER.metadata,
                "distribution",
                return_value=FakeDistribution(),
            ),
        ):
            receipt = RUNNER.build_runtime_receipt()

        self.assertFalse(receipt["network_blocked"])

    def test_total_improvement_cannot_hide_teacher_lane_regression(self) -> None:
        backend = FakeBackend()
        current_teacher = list(TEACHER_IDS)
        current_teacher[100] = 999
        current_async = list(TEACHER_IDS)
        for index in range(10):
            current_async[index] = 999
        backend.current_gate = {
            **exact_gate(),
            "teacher_forced_token_ids": [current_teacher, current_teacher],
            "async_free_run_token_ids": [current_async, current_async],
        }
        selected_teacher = list(TEACHER_IDS)
        selected_teacher[5] = 998
        selected_teacher[6] = 998
        selected_gate = {
            **exact_gate(),
            "teacher_forced_token_ids": [selected_teacher, selected_teacher],
        }
        path = str(CANDIDATES[0]["path"])
        backend.counterfactual_screens[path] = screen(5, 10)
        backend.counterfactual_gates[path] = selected_gate

        with self.assertRaisesRegex(RUNNER.SearchError, "128-step improvement"):
            RUNNER.evaluate_certified_generation(
                self.certified(),
                backend,
                runtime=runtime_receipt(),
                program=program_receipt(),
            )

    def test_close_failure_does_not_skip_remaining_bundle_cleanup(self) -> None:
        backend = FakeBackend()
        original_close = backend.close

        def failing_close(handle: dict[str, object]) -> None:
            original_close(handle)
            if handle["handle_id"] == "current-candidate":
                raise RuntimeError("injected current cleanup failure")

        backend.close = failing_close

        with self.assertRaisesRegex(RUNNER.SearchError, "bundle cleanup failed"):
            RUNNER.evaluate_certified_generation(
                self.certified(),
                backend,
                runtime=runtime_receipt(),
                program=program_receipt(),
            )

        self.assertIn(("close", "bf16-reference"), backend.events)


if __name__ == "__main__":
    unittest.main()
