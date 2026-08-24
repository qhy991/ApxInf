from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "scripts/mlx_mixed_quant_policy.py"
SPEC = importlib.util.spec_from_file_location("mlx_mixed_quant_policy", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def sha256(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def artifact_sha256(value: object) -> str:
    return hashlib.sha256(MODULE.canonical_bytes(value) + b"\n").hexdigest()


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
    "prompt_token_ids": list(MODULE.CANONICAL_CHAT_PROMPT_IDS),
    "teacher_token_ids": TEACHER_IDS,
    "teacher_ids_sha256": sha256(TEACHER_IDS),
    "teacher_steps": 128,
    "free_run_steps": 128,
    "repeat_count": 2,
}
QUALITY_SUITE_SHA256 = "4" * 64


def gate_analysis(
    teacher_runs: list[list[int]], async_runs: list[list[int]]
) -> dict[str, object]:
    teacher_mismatch_count = sum(
        actual != expected
        for run in teacher_runs
        for actual, expected in zip(run, TEACHER_IDS, strict=True)
    )
    async_mismatch_count = sum(
        actual != expected
        for run in async_runs
        for actual, expected in zip(run, TEACHER_IDS, strict=True)
    )

    def first_divergence(run: list[int]) -> int | None:
        for index, (actual, expected) in enumerate(zip(run, TEACHER_IDS, strict=True)):
            if actual != expected:
                return index
        return None

    return {
        "teacher_forced_exact": all(run == TEACHER_IDS for run in teacher_runs),
        "async_free_run_exact": all(run == TEACHER_IDS for run in async_runs),
        "repeated_identically": (
            teacher_runs[0] == teacher_runs[1] and async_runs[0] == async_runs[1]
        ),
        "teacher_forced_mismatch_count": teacher_mismatch_count,
        "async_free_run_mismatch_count": async_mismatch_count,
        "mismatch_count": teacher_mismatch_count + async_mismatch_count,
        "teacher_forced_first_divergence_step": first_divergence(teacher_runs[0]),
        "async_free_run_first_divergence_step": first_divergence(async_runs[0]),
        "teacher_forced_repeat_sha256": [sha256(run) for run in teacher_runs],
        "async_free_run_repeat_sha256": [sha256(run) for run in async_runs],
    }


def screen_receipt(primary_score: int) -> dict[str, object]:
    scores = []
    for index, candidate in enumerate(CANDIDATES):
        hidden_error = primary_score if index == 0 else 10
        scores.append(
            {
                "path": candidate["path"],
                "hidden_error_ppm": hidden_error,
                "top1_margin_erosion_ppm": 0,
                "top1_flip_rate_ppm": 0,
                "score_ppm": hidden_error,
            }
        )
    return {
        "format": "apxinf-mlx-mixed-quant-state-aligned-screen-v1",
        "steps": 32,
        "state_alignment": "prompt-plus-bf16-teacher-prefix-v1",
        "aggregate_score_ppm": sum(score["score_ppm"] for score in scores),
        "module_scores": scores,
    }


def runner_receipt_body(
    document: dict[str, object],
    evaluator: dict[str, object],
    *,
    outcome: str,
    path: str | None,
    stop_reason: str | None = None,
) -> dict[str, object]:
    policy = document["policy"]
    inputs = {
        "source_manifest_sha256": policy["source"]["source_manifest_sha256"],
        "config_sha256": policy["source"]["config_sha256"],
        "language_schema_sha256": policy["source"]["language_schema_sha256"],
        "policy_artifact_sha256": artifact_sha256(document),
        "policy_document_sha256": sha256(document),
        "policy_sha256": document["policy_sha256"],
        "search_receipt_sha256": document["search_receipt_sha256"],
        "candidate_modules_sha256": policy["candidate_modules_sha256"],
        "trace_sha256": sha256(policy["trace"]),
        "quality_suite_sha256": policy["quality_suite_sha256"],
    }
    artifacts = [
        {
            "path": "scripts/run_mlx_mixed_quant_quality.py",
            "size": 123,
            "sha256": "5" * 64,
        }
    ]
    teacher_runs = evaluator["teacher_forced_token_ids"]
    async_runs = evaluator["async_free_run_token_ids"]
    exact = outcome == "exact"
    transition = None
    selected_gate = None
    counterfactual_screens = []
    current_screen = screen_receipt(30)
    counterfactuals = []
    if path is not None:
        overrides = {
            override["path"]: override["tier"]
            for override in policy["quantization"]["overrides"]
        }
        current_tier = overrides.get(path, "w4")
        next_tier = "w8" if current_tier == "w4" else "bf16"
        transition = {"path": path, "from": current_tier, "to": next_tier}
        selected_screen = screen_receipt(5)
        selected_manifest = "a" * 64
        screening_manifest = selected_manifest
        screen_improvement = (
            current_screen["aggregate_score_ppm"]
            - selected_screen["aggregate_score_ppm"]
        )
        counterfactual_screens = [
            {
                "path": path,
                "transition": transition,
                "manifest_sha256": screening_manifest,
                "screen": selected_screen,
                "screen_improvement_ppm": screen_improvement,
            }
        ]
        selected_teacher = [TEACHER_IDS, TEACHER_IDS]
        selected_async = [TEACHER_IDS, TEACHER_IDS]
        current_mismatches = gate_analysis(teacher_runs, async_runs)["mismatch_count"]
        selected_gate = {
            "path": path,
            "transition": transition,
            "manifest_sha256": selected_manifest,
            "teacher_forced_token_ids": selected_teacher,
            "async_free_run_token_ids": selected_async,
            "analysis": gate_analysis(selected_teacher, selected_async),
            "mismatch_improvement": current_mismatches,
            "teacher_async_no_regression": True,
        }
        counterfactuals = [
            {
                "path": path,
                "manifest_sha256": selected_manifest,
                "screening_manifest_sha256": screening_manifest,
                "transition": transition,
            }
        ]
    return {
        "format": MODULE.RUNNER_RECEIPT_FORMAT,
        "passed": exact,
        "outcome": outcome,
        "inputs": inputs,
        "input_sha256": sha256(inputs),
        "program": {
            "artifacts": artifacts,
            "program_sha256": sha256(artifacts),
        },
        "runtime": {
            "python_executable_sha256": "6" * 64,
            "python_version": "3.14.0",
            "packages": [{"name": "mlx", "version": "0.29.3", "sha256": "7" * 64}],
            "offline": True,
            "network_blocked": True,
            "trust_remote_code": False,
        },
        "bundles": {
            "bf16_reference": {"manifest_sha256": "8" * 64},
            "current_candidate": {"manifest_sha256": "9" * 64},
            "counterfactuals": counterfactuals,
            "materialization": "independent-saved-static-verified-reload-v1",
            "dynamic_module_replacement": False,
            "model_bundle_published": False,
        },
        "evaluation": {
            "bf16_reference": {
                "teacher_forced_token_ids": [TEACHER_IDS, TEACHER_IDS],
                "async_free_run_token_ids": [TEACHER_IDS, TEACHER_IDS],
            },
            "current_candidate": {
                "teacher_forced_token_ids": json.loads(json.dumps(teacher_runs)),
                "async_free_run_token_ids": json.loads(json.dumps(async_runs)),
            },
            "attribution": {
                "screening_steps": 32,
                "teacher_forced_token_ids": [run[:32] for run in teacher_runs],
                "async_free_run_token_ids": [run[:32] for run in async_runs],
                "current_screen": current_screen,
                "counterfactual_screens": counterfactual_screens,
                "selected_counterfactual": selected_gate,
            },
        },
        "decision": {
            "outcome": outcome,
            "stop_reason": (
                "nondeterministic-repeated-trajectories"
                if outcome == "nondeterministic"
                else stop_reason
            ),
            "changed_module_count": 0 if path is None else 1,
            "changed_module_path": path,
            "transition": transition,
            "exact_trajectory_claim": exact,
            "general_parity_claim": False,
            "default_ready_claim": False,
            "formal_performance_claim": False,
        },
    }


def bound_exact_observation(document: dict[str, object]) -> dict[str, object]:
    evaluator = {
        "api": TRACE["api"],
        "semantics": TRACE["semantics"],
        "prompt_token_ids": TRACE["prompt_token_ids"],
        "teacher_forced_token_ids": [TEACHER_IDS, TEACHER_IDS],
        "async_free_run_token_ids": [TEACHER_IDS, TEACHER_IDS],
    }
    body = runner_receipt_body(document, evaluator, outcome="exact", path=None)
    return {
        "format": MODULE.OBSERVATION_FORMAT,
        "policy_sha256": document["policy_sha256"],
        "trace_sha256": sha256(TRACE),
        "quality_suite_sha256": QUALITY_SUITE_SHA256,
        "evaluator": evaluator,
        "localization": None,
        "runner_receipt_body": body,
        "runner_receipt_sha256": sha256(body),
    }


def bound_divergent_observation(
    document: dict[str, object], path: str
) -> dict[str, object]:
    generated = list(TEACHER_IDS)
    generated[7] = 999
    evaluator = {
        "api": TRACE["api"],
        "semantics": TRACE["semantics"],
        "prompt_token_ids": TRACE["prompt_token_ids"],
        "teacher_forced_token_ids": [generated, generated],
        "async_free_run_token_ids": [generated, generated],
    }
    body = runner_receipt_body(document, evaluator, outcome="divergent", path=path)
    body_sha256 = sha256(body)
    return {
        "format": MODULE.OBSERVATION_FORMAT,
        "policy_sha256": document["policy_sha256"],
        "trace_sha256": sha256(TRACE),
        "quality_suite_sha256": QUALITY_SUITE_SHA256,
        "evaluator": evaluator,
        "localization": {
            "method": "state-aligned-single-module-attribution-v1",
            "scope": "one-module-counterfactual-no-combinations",
            "screening_steps": 32,
            "gate_steps": 128,
            "grouping": "layer-family-v1",
            "ranking_metric": "hidden-error-plus-top1-margin-v1",
            "sensitive_module_path": path,
            "unique_top_candidate": True,
            "runner_receipt_sha256": body_sha256,
        },
        "runner_receipt_body": body,
        "runner_receipt_sha256": body_sha256,
    }


def bound_nondeterministic_observation(
    document: dict[str, object],
) -> dict[str, object]:
    first = list(TEACHER_IDS)
    first[7] = 999
    second = list(first)
    second[9] = 998
    evaluator = {
        "api": TRACE["api"],
        "semantics": TRACE["semantics"],
        "prompt_token_ids": TRACE["prompt_token_ids"],
        "teacher_forced_token_ids": [first, first],
        "async_free_run_token_ids": [first, second],
    }
    body = runner_receipt_body(
        document, evaluator, outcome="nondeterministic", path=None
    )
    return {
        "format": MODULE.OBSERVATION_FORMAT,
        "policy_sha256": document["policy_sha256"],
        "trace_sha256": sha256(TRACE),
        "quality_suite_sha256": QUALITY_SUITE_SHA256,
        "evaluator": evaluator,
        "localization": None,
        "runner_receipt_body": body,
        "runner_receipt_sha256": sha256(body),
    }


def bound_no_unique_observation(
    document: dict[str, object],
) -> dict[str, object]:
    generated = list(TEACHER_IDS)
    generated[7] = 999
    evaluator = {
        "api": TRACE["api"],
        "semantics": TRACE["semantics"],
        "prompt_token_ids": TRACE["prompt_token_ids"],
        "teacher_forced_token_ids": [generated, generated],
        "async_free_run_token_ids": [generated, generated],
    }
    body = runner_receipt_body(
        document,
        evaluator,
        outcome="divergent",
        path=None,
        stop_reason="no-unique-sensitive-module",
    )
    return {
        "format": MODULE.OBSERVATION_FORMAT,
        "policy_sha256": document["policy_sha256"],
        "trace_sha256": sha256(TRACE),
        "quality_suite_sha256": QUALITY_SUITE_SHA256,
        "evaluator": evaluator,
        "localization": None,
        "runner_receipt_body": body,
        "runner_receipt_sha256": sha256(body),
    }


def initial_document() -> dict[str, object]:
    return MODULE.create_initial_policy_document(
        SOURCE, CANDIDATES, TRACE, QUALITY_SUITE_SHA256
    )


class MixedQuantPolicyTests(unittest.TestCase):
    def test_initial_policy_is_all_w4_and_binds_source_candidates_and_trace(
        self,
    ) -> None:
        document = MODULE.create_initial_policy_document(
            SOURCE, CANDIDATES, TRACE, QUALITY_SUITE_SHA256
        )

        self.assertEqual(document["format"], MODULE.POLICY_DOCUMENT_FORMAT)
        policy = document["policy"]
        self.assertEqual(document["policy_sha256"], sha256(policy))
        self.assertEqual(
            document["search_receipt_sha256"],
            sha256(document["search_receipt"]),
        )
        self.assertEqual(policy["source"], SOURCE)
        self.assertEqual(policy["candidate_modules"], CANDIDATES)
        self.assertEqual(policy["candidate_modules_sha256"], sha256(CANDIDATES))
        self.assertEqual(
            policy["candidate_selector"],
            {
                "format": "canonical-mlx-linear-weight-v1",
                "dtype": "BF16",
                "rank": 2,
                "input_dimension_multiple": 64,
            },
        )
        self.assertEqual(
            policy["quantization"],
            {
                "default": {
                    "tier": "w4",
                    "bits": 4,
                    "group_size": 64,
                    "mode": "affine",
                },
                "overrides": [],
            },
        )
        self.assertEqual(policy["trace"], TRACE)
        self.assertEqual(policy["quality_suite_sha256"], QUALITY_SUITE_SHA256)
        self.assertEqual(
            policy["lineage"],
            {
                "generation": 0,
                "parent_policy_sha256": None,
                "observation_sha256": None,
                "transition": None,
            },
        )
        self.assertEqual(document["search_receipt"]["status"], "initial-all-w4")
        self.assertEqual(
            document["search_receipt"]["quality_suite_sha256"],
            QUALITY_SUITE_SHA256,
        )

        with self.assertRaisesRegex(MODULE.PolicyError, "quality suite"):
            MODULE.create_initial_policy_document(SOURCE, CANDIDATES, TRACE, "bad")

    def test_divergent_w4_policy_advances_only_one_sensitive_module_to_w8(
        self,
    ) -> None:
        initial = MODULE.create_initial_policy_document(
            SOURCE, CANDIDATES, TRACE, QUALITY_SUITE_SHA256
        )
        observation = bound_divergent_observation(initial, CANDIDATES[0]["path"])

        advanced = MODULE.advance_policy_document(initial, observation)

        policy = advanced["policy"]
        self.assertEqual(
            policy["quantization"]["overrides"],
            [
                {
                    "path": CANDIDATES[0]["path"],
                    "tier": "w8",
                    "bits": 8,
                    "group_size": 64,
                    "mode": "affine",
                }
            ],
        )
        self.assertEqual(policy["lineage"]["generation"], 1)
        self.assertEqual(
            policy["lineage"]["parent_policy_sha256"],
            initial["policy_sha256"],
        )
        self.assertEqual(policy["lineage"]["observation_sha256"], sha256(observation))
        self.assertEqual(
            policy["lineage"]["transition"],
            {
                "path": CANDIDATES[0]["path"],
                "from": "w4",
                "to": "w8",
            },
        )
        receipt = advanced["search_receipt"]
        self.assertEqual(receipt["status"], "advance")
        self.assertEqual(receipt["decision"]["transition"], "w4-to-w8")
        self.assertEqual(receipt["decision"]["changed_module_count"], 1)
        self.assertFalse(receipt["decision"]["exact_trajectory_claim"])
        self.assertEqual(
            receipt["observation"]["localization"]["runner_receipt_sha256"],
            receipt["observation"]["runner_receipt_sha256"],
        )

    def test_exact_observation_binds_full_canonical_runner_receipt(self) -> None:
        initial = MODULE.create_initial_policy_document(
            SOURCE, CANDIDATES, TRACE, QUALITY_SUITE_SHA256
        )
        observation = bound_exact_observation(initial)

        exact = MODULE.advance_policy_document(initial, observation)

        self.assertEqual(exact["search_receipt"]["status"], "exact-pareto")
        self.assertEqual(
            exact["search_receipt"]["observation"]["runner_receipt_body"],
            observation["runner_receipt_body"],
        )
        self.assertEqual(
            observation["runner_receipt_sha256"],
            sha256(observation["runner_receipt_body"]),
        )

    def test_nondeterministic_stop_accepts_null_localization_with_full_receipt(
        self,
    ) -> None:
        initial = MODULE.create_initial_policy_document(
            SOURCE, CANDIDATES, TRACE, QUALITY_SUITE_SHA256
        )
        observation = bound_nondeterministic_observation(initial)

        stopped = MODULE.advance_policy_document(initial, observation)

        self.assertEqual(stopped["search_receipt"]["status"], "stop")
        self.assertEqual(
            stopped["search_receipt"]["decision"]["stop_reason"],
            "nondeterministic-repeated-trajectories",
        )
        self.assertIsNone(stopped["search_receipt"]["observation"]["localization"])
        self.assertEqual(
            stopped["search_receipt"]["observation"]["runner_receipt_body"],
            observation["runner_receipt_body"],
        )

    def test_divergent_without_unique_winner_stops_with_null_localization(
        self,
    ) -> None:
        initial = MODULE.create_initial_policy_document(
            SOURCE, CANDIDATES, TRACE, QUALITY_SUITE_SHA256
        )
        observation = bound_no_unique_observation(initial)

        stopped = MODULE.advance_policy_document(initial, observation)

        self.assertEqual(stopped["search_receipt"]["status"], "stop")
        self.assertEqual(
            stopped["search_receipt"]["decision"]["stop_reason"],
            "no-unique-sensitive-module",
        )
        self.assertIsNone(stopped["search_receipt"]["observation"]["localization"])

    def test_same_sensitive_module_advances_from_w8_to_bf16_before_stop(
        self,
    ) -> None:
        path = CANDIDATES[0]["path"]
        initial = initial_document()
        w8 = MODULE.advance_policy_document(
            initial, bound_divergent_observation(initial, path)
        )

        bf16 = MODULE.advance_policy_document(w8, bound_divergent_observation(w8, path))

        self.assertEqual(
            bf16["policy"]["quantization"]["overrides"],
            [{"path": path, "tier": "bf16"}],
        )
        self.assertEqual(
            bf16["policy"]["lineage"]["transition"],
            {"path": path, "from": "w8", "to": "bf16"},
        )
        self.assertEqual(bf16["search_receipt"]["decision"]["transition"], "w8-to-bf16")

    def test_full_128_step_exact_teacher_and_async_trajectory_is_terminal_pareto(
        self,
    ) -> None:
        initial = initial_document()

        result = MODULE.advance_policy_document(
            initial, bound_exact_observation(initial)
        )

        self.assertEqual(result["policy_sha256"], initial["policy_sha256"])
        self.assertEqual(result["policy"], initial["policy"])
        receipt = result["search_receipt"]
        self.assertEqual(receipt["status"], "exact-pareto")
        self.assertTrue(receipt["decision"]["exact_trajectory_claim"])
        self.assertEqual(receipt["decision"]["teacher_steps"], 128)
        self.assertEqual(receipt["decision"]["async_free_run_steps"], 128)
        self.assertEqual(receipt["decision"]["changed_module_count"], 0)
        self.assertFalse(receipt["decision"]["formal_performance_claim"])

    def test_noise_or_bf16_exhaustion_returns_evidence_bound_stop_without_mutation(
        self,
    ) -> None:
        path = CANDIDATES[0]["path"]
        initial = initial_document()
        noisy_observation = bound_nondeterministic_observation(initial)

        noisy = MODULE.advance_policy_document(initial, noisy_observation)

        self.assertEqual(noisy["policy_sha256"], initial["policy_sha256"])
        self.assertEqual(noisy["search_receipt"]["status"], "stop")
        self.assertEqual(
            noisy["search_receipt"]["decision"]["stop_reason"],
            "nondeterministic-repeated-trajectories",
        )
        self.assertEqual(
            noisy["search_receipt"]["observation_sha256"],
            sha256(noisy_observation),
        )

        w8 = MODULE.advance_policy_document(
            initial, bound_divergent_observation(initial, path)
        )
        bf16 = MODULE.advance_policy_document(w8, bound_divergent_observation(w8, path))
        exhausted_observation = bound_no_unique_observation(bf16)
        exhausted = MODULE.advance_policy_document(bf16, exhausted_observation)

        self.assertEqual(exhausted["policy_sha256"], bf16["policy_sha256"])
        self.assertEqual(exhausted["search_receipt"]["status"], "stop")
        self.assertEqual(
            exhausted["search_receipt"]["decision"]["stop_reason"],
            "no-unique-sensitive-module",
        )
        self.assertEqual(
            exhausted["search_receipt"]["decision"]["changed_module_count"], 0
        )

    def test_exact_upgraded_policy_is_not_mislabeled_pareto_before_ablation(
        self,
    ) -> None:
        path = CANDIDATES[0]["path"]
        initial = initial_document()
        upgraded = MODULE.advance_policy_document(
            initial, bound_divergent_observation(initial, path)
        )

        exact = MODULE.advance_policy_document(
            upgraded, bound_exact_observation(upgraded)
        )

        self.assertEqual(exact["search_receipt"]["status"], "exact-candidate")
        self.assertTrue(
            exact["search_receipt"]["decision"]["reverse_ablation_required"]
        )
        self.assertNotIn("pareto", exact["search_receipt"]["status"])

    def test_partial_or_unbound_exact_claim_fails_closed(self) -> None:
        initial = initial_document()
        partial = bound_exact_observation(initial)
        divergent = list(TEACHER_IDS)
        divergent[127] = 999
        partial["evaluator"]["async_free_run_token_ids"] = [divergent, divergent]

        with self.assertRaisesRegex(MODULE.PolicyError, "current trajectories"):
            MODULE.advance_policy_document(initial, partial)

        outside = bound_divergent_observation(
            initial, "language_model.model.layers.99.fake"
        )
        with self.assertRaisesRegex(MODULE.PolicyError, "policy-bound"):
            MODULE.advance_policy_document(initial, outside)

        malformed = bound_divergent_observation(initial, CANDIDATES[0]["path"])
        malformed["localization"]["sensitive_module_path"] = []
        with self.assertRaisesRegex(MODULE.PolicyError, "frozen candidate"):
            MODULE.advance_policy_document(initial, malformed)

    def test_runner_receipt_suite_and_policy_inputs_are_cross_bound(self) -> None:
        initial = initial_document()
        observation = bound_exact_observation(initial)
        observation["runner_receipt_body"]["inputs"]["quality_suite_sha256"] = "b" * 64
        observation["runner_receipt_body"]["input_sha256"] = sha256(
            observation["runner_receipt_body"]["inputs"]
        )
        observation["runner_receipt_sha256"] = sha256(
            observation["runner_receipt_body"]
        )

        with self.assertRaisesRegex(MODULE.PolicyError, "frozen policy"):
            MODULE.advance_policy_document(initial, observation)

        observation = bound_exact_observation(initial)
        observation["runner_receipt_body"]["passed"] = False
        with self.assertRaisesRegex(MODULE.PolicyError, "runner receipt SHA-256"):
            MODULE.advance_policy_document(initial, observation)

    def test_runner_receipt_forbids_dynamic_replacement_publish_and_ready_claims(
        self,
    ) -> None:
        cases = (
            ("bundles", "dynamic_module_replacement", True, "dynamic"),
            ("bundles", "model_bundle_published", True, "publishing"),
            ("decision", "general_parity_claim", True, "parity"),
            ("decision", "default_ready_claim", True, "readiness"),
        )
        for section, field, value, message in cases:
            with self.subTest(field=field):
                initial = initial_document()
                observation = bound_exact_observation(initial)
                observation["runner_receipt_body"][section][field] = value
                observation["runner_receipt_sha256"] = sha256(
                    observation["runner_receipt_body"]
                )

                with self.assertRaisesRegex(MODULE.PolicyError, message):
                    MODULE.advance_policy_document(initial, observation)

    def test_runner_raw_trajectories_and_localization_share_one_receipt_hash(
        self,
    ) -> None:
        initial = initial_document()
        observation = bound_divergent_observation(initial, CANDIDATES[0]["path"])
        receipt_run = list(
            observation["runner_receipt_body"]["evaluation"]["current_candidate"][
                "async_free_run_token_ids"
            ][0]
        )
        receipt_run[12] = 777
        observation["runner_receipt_body"]["evaluation"]["current_candidate"][
            "async_free_run_token_ids"
        ][0] = receipt_run
        observation["runner_receipt_sha256"] = sha256(
            observation["runner_receipt_body"]
        )
        observation["localization"]["runner_receipt_sha256"] = observation[
            "runner_receipt_sha256"
        ]

        with self.assertRaisesRegex(MODULE.PolicyError, "current trajectories"):
            MODULE.advance_policy_document(initial, observation)

        observation = bound_divergent_observation(initial, CANDIDATES[0]["path"])
        observation["localization"]["runner_receipt_sha256"] = "c" * 64
        with self.assertRaisesRegex(MODULE.PolicyError, "uniquely localize"):
            MODULE.advance_policy_document(initial, observation)

    def test_rehashed_selected_counterfactual_must_strictly_improve_both_lanes(
        self,
    ) -> None:
        initial = initial_document()
        observation = bound_divergent_observation(initial, CANDIDATES[0]["path"])
        selected = observation["runner_receipt_body"]["evaluation"]["attribution"][
            "selected_counterfactual"
        ]
        regressed = list(TEACHER_IDS)
        regressed[6] = 997
        regressed[8] = 998
        selected["async_free_run_token_ids"] = [regressed, regressed]
        selected["analysis"] = gate_analysis(
            selected["teacher_forced_token_ids"],
            selected["async_free_run_token_ids"],
        )
        current_analysis = gate_analysis(
            observation["evaluator"]["teacher_forced_token_ids"],
            observation["evaluator"]["async_free_run_token_ids"],
        )
        selected["mismatch_improvement"] = (
            current_analysis["mismatch_count"] - selected["analysis"]["mismatch_count"]
        )
        selected["teacher_async_no_regression"] = False
        body = observation["runner_receipt_body"]
        observation["runner_receipt_sha256"] = sha256(body)
        observation["localization"]["runner_receipt_sha256"] = observation[
            "runner_receipt_sha256"
        ]

        with self.assertRaisesRegex(MODULE.PolicyError, "selected counterfactual"):
            MODULE.advance_policy_document(initial, observation)

    def test_selected_gate_manifest_must_match_its_32_step_materialization(
        self,
    ) -> None:
        initial = initial_document()
        observation = bound_divergent_observation(initial, CANDIDATES[0]["path"])
        body = observation["runner_receipt_body"]
        selected = body["evaluation"]["attribution"]["selected_counterfactual"]
        descriptor = body["bundles"]["counterfactuals"][0]
        selected["manifest_sha256"] = "c" * 64
        descriptor["manifest_sha256"] = "c" * 64
        observation["runner_receipt_sha256"] = sha256(body)
        observation["localization"]["runner_receipt_sha256"] = observation[
            "runner_receipt_sha256"
        ]

        with self.assertRaisesRegex(MODULE.PolicyError, "manifest"):
            MODULE.advance_policy_document(initial, observation)

    def test_rehashed_selected_transition_cannot_skip_the_current_policy_tier(
        self,
    ) -> None:
        initial = initial_document()
        observation = bound_divergent_observation(initial, CANDIDATES[0]["path"])
        body = observation["runner_receipt_body"]
        skipped = {
            "path": CANDIDATES[0]["path"],
            "from": "w4",
            "to": "bf16",
        }
        body["decision"]["transition"] = skipped
        body["bundles"]["counterfactuals"][0]["transition"] = skipped
        attribution = body["evaluation"]["attribution"]
        attribution["counterfactual_screens"][0]["transition"] = skipped
        attribution["selected_counterfactual"]["transition"] = skipped
        observation["runner_receipt_sha256"] = sha256(body)
        observation["localization"]["runner_receipt_sha256"] = observation[
            "runner_receipt_sha256"
        ]

        with self.assertRaisesRegex(MODULE.PolicyError, "transition"):
            MODULE.advance_policy_document(initial, observation)

    def test_v1_observation_is_explicitly_rejected_for_real_advancement(self) -> None:
        initial = initial_document()
        observation = bound_exact_observation(initial)
        observation["format"] = "apxinf-mlx-mixed-quant-observation-v1"

        with self.assertRaisesRegex(MODULE.PolicyError, "format drifted"):
            MODULE.advance_policy_document(initial, observation)

    def test_search_receipt_tampering_is_detected_before_policy_use(self) -> None:
        document = initial_document()
        tampered = json.loads(json.dumps(document))
        tampered["search_receipt"]["decision"]["formal_performance_claim"] = True

        with self.assertRaisesRegex(MODULE.PolicyError, "receipt SHA-256"):
            MODULE.validate_policy_document(tampered)

    def test_rehashed_forged_exact_receipt_is_rejected_semantically(self) -> None:
        document = initial_document()
        forged = json.loads(json.dumps(document))
        receipt = forged["search_receipt"]
        receipt["status"] = "exact-pareto"
        receipt["decision"]["exact_trajectory_claim"] = True
        forged["search_receipt_sha256"] = sha256(receipt)

        with self.assertRaisesRegex(MODULE.PolicyError, "receipt semantics"):
            MODULE.validate_policy_document(forged)

    def test_rehashed_multimodule_generation_jump_is_rejected(self) -> None:
        document = initial_document()
        forged = json.loads(json.dumps(document))
        policy = forged["policy"]
        policy["quantization"]["overrides"] = [
            {"path": CANDIDATES[0]["path"], "tier": "bf16"},
            {"path": CANDIDATES[1]["path"], "tier": "bf16"},
        ]
        policy["lineage"] = {
            "generation": 1,
            "parent_policy_sha256": "3" * 64,
            "observation_sha256": "4" * 64,
            "transition": {
                "path": CANDIDATES[0]["path"],
                "from": "w4",
                "to": "w8",
            },
        }
        forged["policy_sha256"] = sha256(policy)
        forged["search_receipt"]["output_policy_sha256"] = forged["policy_sha256"]
        forged["search_receipt_sha256"] = sha256(forged["search_receipt"])

        with self.assertRaisesRegex(MODULE.PolicyError, "transition history"):
            MODULE.validate_policy_document(forged)

    def test_exact_or_stop_receipt_is_terminal_and_cannot_advance_again(self) -> None:
        path = CANDIDATES[0]["path"]
        initial = initial_document()
        exact = MODULE.advance_policy_document(
            initial, bound_exact_observation(initial)
        )
        with self.assertRaisesRegex(MODULE.PolicyError, "terminal"):
            MODULE.advance_policy_document(
                exact, bound_divergent_observation(exact, path)
            )

        noisy_observation = bound_nondeterministic_observation(initial)
        stopped = MODULE.advance_policy_document(initial, noisy_observation)
        with self.assertRaisesRegex(MODULE.PolicyError, "terminal"):
            MODULE.advance_policy_document(
                stopped, bound_divergent_observation(stopped, path)
            )


if __name__ == "__main__":
    unittest.main()
