from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = ROOT / "configs" / "qwen35-0.8b-cross-runtime-formal-v3.json"
MODULE_PATH = ROOT / "scripts" / "validate_qwen35_cross_runtime_formal_contract.py"


def load_module():
    spec = importlib.util.spec_from_file_location(
        "validate_qwen35_cross_runtime_formal_contract_for_tests", MODULE_PATH
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


def checked_in_contract() -> dict:
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


class Qwen35CrossRuntimeFormalContractTests(unittest.TestCase):
    def test_validator_accepts_the_checked_in_fail_closed_predeclaration(self):
        module = load_module()

        contract = module.load_contract(CONTRACT_PATH)

        self.assertEqual(
            contract["format"],
            "apxinf-qwen35-cross-runtime-formal-predeclaration-v3",
        )
        readiness = contract["core_parity_contract"]["current_readiness"]
        self.assertFalse(readiness["formal_campaign_may_start_now"])
        self.assertIn(
            "V3_CORE_DRIVER_AND_BINARY_HASHES_NOT_CAPTURED",
            readiness["blocker_codes"],
        )

    def test_parsed_contract_canonical_semantics_are_wholly_pinned(self):
        module = load_module()
        contract = checked_in_contract()
        contract["authored_at_utc"] = "2026-08-25T21:07:11Z"

        with self.assertRaisesRegex(
            module.FormalContractError, "canonical contract semantic pin"
        ):
            module.validate_contract(contract)

    def test_all_three_statistics_and_decision_contracts_are_exactly_frozen(self):
        module = load_module()
        mutations = (
            (
                "core stability",
                lambda statistics: statistics["CORE_A_VS_L"][
                    "stability_gates"
                ].__setitem__("A_tpot_population_cv_max", 0.04),
            ),
            (
                "native decision",
                lambda statistics: statistics["NATIVE_A_VS_L"][
                    "decision_rules"
                ].__setitem__("INCONCLUSIVE", "pick the faster point estimate"),
            ),
            (
                "gateway precedence",
                lambda statistics: statistics["GATEWAY_B_VS_G"].__setitem__(
                    "decision_precedence",
                    [
                        "UNINTERPRETABLE",
                        "POSITIVE_GATEWAY_PATH_OVERHEAD",
                        "PRACTICALLY_EQUIVALENT_GATEWAY_PATH",
                        "INCONCLUSIVE",
                    ],
                ),
            ),
        )
        for label, mutate in mutations:
            with self.subTest(label=label):
                contract = checked_in_contract()
                mutate(contract["statistics_and_decisions"])
                with self.assertRaisesRegex(
                    module.FormalContractError, "statistics and decisions"
                ):
                    module.validate_contract(contract)

    def test_campaign_scope_lineage_and_edge_marker_ids_are_frozen_and_cross_bound(self):
        module = load_module()
        mutations = (
            (
                "campaign id",
                lambda contract: contract.__setitem__("campaign_id", "replacement-campaign"),
            ),
            (
                "scope",
                lambda contract: contract["scope"]["host"].__setitem__(
                    "chip", "Apple M5"
                ),
            ),
            (
                "lineage",
                lambda contract: contract["lineage"].__setitem__(
                    "v2_samples_reused_by_v3", True
                ),
            ),
            (
                "edge id",
                lambda contract: contract["comparison_graph"]["edges"][
                    "NATIVE_A_VS_L"
                ].__setitem__("subcampaign_id", "native-edge-drift"),
            ),
            (
                "marker id",
                lambda contract: contract["failure_contract"][
                    "subcampaign_markers"
                ]["NATIVE_A_VS_L"].__setitem__(
                    "subcampaign_id", "native-marker-drift"
                ),
            ),
        )
        for label, mutate in mutations:
            with self.subTest(label=label):
                contract = checked_in_contract()
                mutate(contract)
                with self.assertRaisesRegex(
                    module.FormalContractError, "campaign binding"
                ):
                    module.validate_contract(contract)

    def test_cli_emits_a_machine_readable_three_edge_readiness_receipt(self):
        completed = subprocess.run(
            [
                "/usr/bin/python3",
                "-I",
                "-B",
                str(MODULE_PATH),
                "--contract",
                str(CONTRACT_PATH),
            ],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        receipt = json.loads(completed.stdout)
        self.assertTrue(receipt["valid"])
        self.assertEqual(
            receipt["format"],
            "apxinf-qwen35-cross-runtime-formal-validation-v3",
        )
        self.assertEqual(
            receipt["campaign_id"],
            "qwen35-0.8b-cross-runtime-formal-v3-20260826",
        )
        self.assertEqual(
            receipt["contract_file_sha256"],
            hashlib.sha256(CONTRACT_PATH.read_bytes()).hexdigest(),
        )
        self.assertFalse(receipt["edges"]["CORE_A_VS_L"]["ready"])
        self.assertFalse(receipt["edges"]["NATIVE_A_VS_L"]["ready"])
        self.assertFalse(receipt["edges"]["GATEWAY_B_VS_G"]["ready"])
        self.assertEqual(
            receipt["edges"]["NATIVE_A_VS_L"]["claim_class"],
            "named-deployment-only-with-disclosed-regime-differences",
        )

    def test_cli_rejects_a_semantically_tampered_contract_with_exit_one(self):
        contract = checked_in_contract()
        contract["timing_contract"]["CORE_A_VS_L"][
            "common_token_ready_boundary"
        ] = "logits-ready"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tampered.json"
            path.write_text(json.dumps(contract) + "\n", encoding="utf-8")
            completed = subprocess.run(
                [
                    "/usr/bin/python3",
                    "-I",
                    "-B",
                    str(MODULE_PATH),
                    "--contract",
                    str(path),
                ],
                cwd=ROOT,
                check=False,
                capture_output=True,
                text=True,
            )

        self.assertEqual(completed.returncode, 1)
        self.assertEqual(completed.stdout, "")
        self.assertIn("formal contract rejected", completed.stderr)
        self.assertIn("timing boundary", completed.stderr)

    def test_cli_parses_and_hashes_one_identical_read_bytes_snapshot(self):
        module = load_module()
        raw = CONTRACT_PATH.read_bytes()
        stdout = io.StringIO()
        stderr = io.StringIO()
        with mock.patch.object(
            module.Path, "read_bytes", side_effect=[raw]
        ) as read_bytes, mock.patch.object(
            module.Path,
            "read_text",
            side_effect=AssertionError("CLI must not use a second text read"),
        ), contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            returncode = module.main(["--contract", str(CONTRACT_PATH)])

        self.assertEqual(returncode, 0, stderr.getvalue())
        self.assertEqual(read_bytes.call_count, 1)
        receipt = json.loads(stdout.getvalue())
        self.assertEqual(
            receipt["contract_file_sha256"], hashlib.sha256(raw).hexdigest()
        )
        self.assertEqual(receipt["contract_file_size_bytes"], len(raw))

    def test_cli_rejects_nonfinite_json_constants_during_parse(self):
        contract = checked_in_contract()
        contract["sampling_state_at_authoring"]["v3_generation_requests"] = float(
            "nan"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nan.json"
            path.write_text(json.dumps(contract) + "\n", encoding="utf-8")
            completed = subprocess.run(
                [
                    "/usr/bin/python3",
                    "-I",
                    "-B",
                    str(MODULE_PATH),
                    "--contract",
                    str(path),
                ],
                cwd=ROOT,
                check=False,
                capture_output=True,
                text=True,
            )

        self.assertEqual(completed.returncode, 1)
        self.assertEqual(completed.stdout, "")
        self.assertIn("non-finite JSON constant", completed.stderr)

    def test_cli_rejects_contract_bytes_that_are_not_strict_utf8(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid-utf8.json"
            path.write_bytes(b'{"format":"\xff"}\n')
            completed = subprocess.run(
                [
                    "/usr/bin/python3",
                    "-I",
                    "-B",
                    str(MODULE_PATH),
                    "--contract",
                    str(path),
                ],
                cwd=ROOT,
                check=False,
                capture_output=True,
                text=True,
            )

        self.assertEqual(completed.returncode, 1)
        self.assertEqual(completed.stdout, "")
        self.assertIn("strict UTF-8", completed.stderr)

    def test_cli_rejects_nonfinite_validation_receipt_output(self):
        module = load_module()
        stdout = io.StringIO()
        stderr = io.StringIO()
        with mock.patch.object(
            module, "validation_receipt", return_value={"invalid": float("nan")}
        ), contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            returncode = module.main(["--contract", str(CONTRACT_PATH)])

        self.assertEqual(returncode, 1)
        self.assertEqual(stdout.getvalue(), "")
        self.assertIn("non-finite validation receipt", stderr.getvalue())

    def test_loader_rejects_duplicate_json_keys(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "duplicate.json"
            path.write_text(
                '{"format":"first","format":"second"}\n', encoding="utf-8"
            )

            with self.assertRaisesRegex(module.FormalContractError, "duplicate key"):
                module.load_contract(path)

    def test_core_timing_ends_only_when_the_same_greedy_token_is_ready(self):
        module = load_module()
        mutations = (
            (
                "common endpoint",
                lambda value: value.__setitem__(
                    "common_token_ready_boundary", "device-logits-ready"
                ),
            ),
            (
                "current ApxInf rerank",
                lambda value: value[
                    "current_nonadmitted_ApxInf_boundary_includes"
                ].remove(
                    "F32 exact top-4 rerank"
                ),
            ),
            (
                "formal ApxInf argmax",
                lambda value: value[
                    "formal_ApxInf_selected_lane_boundary_includes"
                ].remove("full-vocabulary unbiased greedy argmax"),
            ),
            (
                "llama sampler",
                lambda value: value["llama_cpp_boundary_includes"].remove(
                    "sampler or argmax execution"
                ),
            ),
            (
                "argmax exclusion",
                lambda value: value.__setitem__(
                    "sampling_or_argmax_may_be_excluded", True
                ),
            ),
            (
                "early stop",
                lambda value: value.__setitem__("end", "128th-logits-ready"),
            ),
        )

        for label, mutate in mutations:
            with self.subTest(label=label):
                contract = checked_in_contract()
                mutate(contract["timing_contract"]["CORE_A_VS_L"])
                with self.assertRaisesRegex(
                    module.FormalContractError, "timing boundary"
                ):
                    module.validate_contract(contract)

    def test_core_lane_requires_exact_gguf_q8_weights_quantization_and_f16_kv(self):
        module = load_module()
        mutations = (
            (
                "selected lane",
                lambda parity: parity.__setitem__("selected_lane", "CUSTOM_W8"),
            ),
            (
                "source revision shortcut",
                lambda parity: parity["logical_weights"].__setitem__(
                    "same_source_revision_is_sufficient_for_equality", True
                ),
            ),
            (
                "logical manifest equality",
                lambda parity: parity["logical_weights"].__setitem__(
                    "A_and_L_logical_manifest_root_sha256_must_equal", False
                ),
            ),
            (
                "runtime traceability",
                lambda parity: parity["logical_weights"].__setitem__(
                    "ApxInf_must_trace_every_runtime_tensor_to_exact_GGUF_source_payload",
                    False,
                ),
            ),
            (
                "signed q8 values",
                lambda parity: parity["quantization"].__setitem__(
                    "same_signed_q8_values_required", False
                ),
            ),
            (
                "f16 scale bits",
                lambda parity: parity["quantization"].__setitem__(
                    "same_f16_scale_bits_required", False
                ),
            ),
            (
                "runtime requantization",
                lambda parity: parity["quantization"].__setitem__(
                    "runtime_requantization_allowed", True
                ),
            ),
            (
                "KV value dtype",
                lambda parity: parity["kv_cache"].__setitem__(
                    "value_dtype", "f32"
                ),
            ),
            (
                "KV reuse",
                lambda parity: parity["kv_cache"].__setitem__(
                    "prefix_or_cross_sample_reuse_allowed", True
                ),
            ),
        )

        for label, mutate in mutations:
            with self.subTest(label=label):
                contract = checked_in_contract()
                mutate(contract["core_parity_contract"])
                with self.assertRaisesRegex(
                    module.FormalContractError, "parity contract"
                ):
                    module.validate_contract(contract)

    def test_core_edge_arms_workload_artifact_prefill_head_and_execution_are_exact(self):
        module = load_module()
        mutations = (
            (
                "A arm",
                lambda contract: contract["comparison_graph"]["arms"]["A"].__setitem__(
                    "kind", "current-hybrid-core"
                ),
            ),
            (
                "L arm",
                lambda contract: contract["comparison_graph"]["arms"]["L"].__setitem__(
                    "transport_in_primary_timing", "HTTP"
                ),
            ),
            (
                "edge",
                lambda contract: contract["comparison_graph"]["edges"][
                    "CORE_A_VS_L"
                ].__setitem__("workload_id", "NATIVE_RAW13_FREE128_V3"),
            ),
            (
                "workload",
                lambda contract: contract["workload_contracts"][
                    "CORE_RAW13_FREE128_V3"
                ]["generation"].__setitem__("effective_batch_tokens", 256),
            ),
            (
                "artifact",
                lambda contract: contract["core_parity_contract"][
                    "model_artifact"
                ].__setitem__("secondary_quantization_allowed", True),
            ),
            (
                "prefill",
                lambda contract: contract["core_parity_contract"][
                    "prefill_and_recurrent_state"
                ].__setitem__("ApxInf_CPU_F32_prefill_allowed", True),
            ),
            (
                "state",
                lambda contract: contract["core_parity_contract"][
                    "prefill_and_recurrent_state"
                ].__setitem__("A_and_L_GDN_state_policy_manifest_sha256_must_equal", False),
            ),
            (
                "head",
                lambda contract: contract["core_parity_contract"][
                    "output_head_and_selection"
                ].__setitem__("ApxInf_F32_tied_embedding_top4_rerank_allowed", True),
            ),
            (
                "execution",
                lambda contract: contract["core_parity_contract"][
                    "execution"
                ].__setitem__("cpu_worker_threads", 8),
            ),
        )
        for label, mutate in mutations:
            with self.subTest(label=label):
                contract = checked_in_contract()
                mutate(contract)
                with self.assertRaisesRegex(module.FormalContractError, "CORE contract"):
                    module.validate_contract(contract)

    def test_unproven_exact_lane_and_null_manifest_must_block_core_sampling(self):
        module = load_module()
        required_blockers = (
            "APXINF_Q8_0_PREFILL_STATE_PARITY_NOT_PROVEN",
            "APXINF_Q8_0_HEAD_ARGMAX_PARITY_NOT_PROVEN",
        )
        for blocker in required_blockers:
            with self.subTest(missing_blocker=blocker):
                contract = checked_in_contract()
                contract["core_parity_contract"]["current_readiness"][
                    "blocker_codes"
                ].remove(blocker)
                with self.assertRaisesRegex(module.FormalContractError, "readiness"):
                    module.validate_contract(contract)

        mutations = (
            ("admitted", "formally_admitted", True),
            ("instantiable", "selected_lane_instantiable_now", True),
            ("campaign start", "formal_campaign_may_start_now", True),
            (
                "null hash policy",
                "null_common_parameters_hash_is_intentional_and_blocks_sampling",
                False,
            ),
            (
                "custom W8 eligibility",
                "existing_custom_G32_G64_W8_hybrid_is_eligible",
                True,
            ),
            (
                "trajectory shortcut",
                "matching_the_128_token_trajectory_does_not_clear_weight_prefill_state_kv_or_head_gates",
                False,
            ),
        )
        for label, field, value in mutations:
            with self.subTest(label=label):
                contract = checked_in_contract()
                contract["core_parity_contract"]["current_readiness"][field] = value
                with self.assertRaisesRegex(module.FormalContractError, "readiness"):
                    module.validate_contract(contract)

    def test_each_executable_edge_uses_fixed_abba_baab_with_at_least_twelve_per_arm(self):
        module = load_module()
        mutations = (
            (
                "core too few",
                lambda protocol: protocol["CORE_A_VS_L"].update(
                    {
                        "timed_block_orders": ["ABBA", "BAAB"] * 2,
                        "timed_blocks": 4,
                        "timed_samples_per_arm": 8,
                        "timed_samples_total": 16,
                    }
                ),
            ),
            (
                "core one order",
                lambda protocol: protocol["CORE_A_VS_L"].__setitem__(
                    "timed_block_orders", ["ABBA"] * 8
                ),
            ),
            (
                "core count lie",
                lambda protocol: protocol["CORE_A_VS_L"].__setitem__(
                    "timed_samples_per_arm", 15
                ),
            ),
            (
                "native count lie",
                lambda protocol: protocol["NATIVE_A_VS_L"].__setitem__(
                    "timed_samples_per_arm", 15
                ),
            ),
            (
                "gateway bad order",
                lambda protocol: protocol["GATEWAY_B_VS_G"].__setitem__(
                    "odd_macroblock_abstract_orders", ["ABBA", "AABB"]
                ),
            ),
            (
                "gateway count lie",
                lambda protocol: protocol["GATEWAY_B_VS_G"].__setitem__(
                    "timed_samples_per_arm", 63
                ),
            ),
            (
                "post-hoc extension",
                lambda protocol: protocol.__setitem__(
                    "sample_extension_after_looking_at_results_allowed", True
                ),
            ),
        )
        for label, mutate in mutations:
            with self.subTest(label=label):
                contract = checked_in_contract()
                mutate(contract["execution_protocol"])
                with self.assertRaisesRegex(module.FormalContractError, "schedule"):
                    module.validate_contract(contract)

    def test_quiet_host_gate_is_continuous_strict_and_fail_closed(self):
        module = load_module()
        mutations = (
            (
                "allowlist scope",
                lambda gate: gate["process_policy"].__setitem__(
                    "allowlist_scope", "all processes owned by the current user"
                ),
            ),
            (
                "preflight samples",
                lambda gate: gate["preflight"].__setitem__("snapshot_count", 4),
            ),
            (
                "continuous monitor",
                lambda gate: gate["continuous_monitor"].__setitem__(
                    "starts_before_first_warmup", False
                ),
            ),
            (
                "postflight samples",
                lambda gate: gate["postflight"].__setitem__("snapshot_count", 4),
            ),
            (
                "single process CPU",
                lambda gate: gate["process_policy"].__setitem__(
                    "maximum_single_non_allowlisted_process_cpu_percent", 10.1
                ),
            ),
            (
                "aggregate CPU",
                lambda gate: gate["process_policy"].__setitem__(
                    "maximum_aggregate_non_allowlisted_process_cpu_percent", 25.1
                ),
            ),
            (
                "driver kills processes",
                lambda gate: gate["process_policy"].__setitem__(
                    "user_or_system_process_termination_by_driver_allowed", True
                ),
            ),
            (
                "swap",
                lambda gate: gate["system_policy"].__setitem__(
                    "system_swap_delta_bytes_required", 1
                ),
            ),
            (
                "thermal drift",
                lambda gate: gate["system_policy"].__setitem__(
                    "power_or_thermal_state_change_allowed", True
                ),
            ),
        )
        for label, mutate in mutations:
            with self.subTest(label=label):
                contract = checked_in_contract()
                mutate(contract["host_quiet_gate"])
                with self.assertRaisesRegex(module.FormalContractError, "quiet-host"):
                    module.validate_contract(contract)

    def test_native_edge_is_named_deployment_only_and_discloses_every_regime_difference(self):
        module = load_module()
        mutations = (
            (
                "engine-only edge",
                lambda contract: contract["comparison_graph"]["edges"][
                    "NATIVE_A_VS_L"
                ].__setitem__("engine_only_ranking_edge", True),
            ),
            (
                "false identical edge",
                lambda contract: contract["comparison_graph"]["edges"][
                    "NATIVE_A_VS_L"
                ].__setitem__(
                    "identical_weight_quantization_kv_or_placement_edge", True
                ),
            ),
            (
                "quantization hidden",
                lambda contract: contract["native_deployment_contract"][
                    "machine_disclosures"
                ].__setitem__("same_quantization_scheme", True),
            ),
            (
                "prefill hidden",
                lambda contract: contract["native_deployment_contract"][
                    "machine_disclosures"
                ].__setitem__("same_prefill_precision_or_placement", True),
            ),
            (
                "KV hidden",
                lambda contract: contract["native_deployment_contract"][
                    "machine_disclosures"
                ].__setitem__("same_KV_dtype", True),
            ),
            (
                "head hidden",
                lambda contract: contract["native_deployment_contract"][
                    "machine_disclosures"
                ].__setitem__("same_output_head_mechanism", True),
            ),
            (
                "engine conclusion",
                lambda contract: contract["native_deployment_contract"][
                    "claim_scope"
                ].__setitem__("engine_only_conclusion_allowed", True),
            ),
            (
                "premature admission",
                lambda contract: contract["native_deployment_contract"][
                    "current_readiness"
                ].__setitem__("formally_admitted", True),
            ),
            (
                "statistics engine winner",
                lambda contract: contract["statistics_and_decisions"][
                    "NATIVE_A_VS_L"
                ].__setitem__("engine_only_winner_claim_allowed", True),
            ),
        )
        for label, mutate in mutations:
            with self.subTest(label=label):
                contract = checked_in_contract()
                mutate(contract)
                with self.assertRaisesRegex(
                    module.FormalContractError, "native deployment"
                ):
                    module.validate_contract(contract)

    def test_native_deployment_identity_freezes_both_complete_runtime_configurations(self):
        module = load_module()
        mutations = (
            (
                "deleted AN checkpoint hash",
                lambda deployments: deployments["AN"]["source_weights"].pop(
                    "checkpoint_sha256"
                ),
            ),
            (
                "cleared AN decode",
                lambda deployments: deployments["AN"].__setitem__("decode", {}),
            ),
            (
                "tampered AN weight regime",
                lambda deployments: deployments["AN"]["weight_regime"].__setitem__(
                    "group_sizes_present", [32]
                ),
            ),
            (
                "tampered L source hash",
                lambda deployments: deployments["L"]["source_weights"].__setitem__(
                    "artifact_sha256", "0" * 64
                ),
            ),
            (
                "tampered L placement",
                lambda deployments: deployments["L"].__setitem__(
                    "placement", "unspecified"
                ),
            ),
            (
                "tampered L threads",
                lambda deployments: deployments["L"]["thread_policy"].__setitem__(
                    "threads", 8
                ),
            ),
        )
        for label, mutate in mutations:
            with self.subTest(label=label):
                contract = checked_in_contract()
                mutate(contract["native_deployment_contract"]["deployments"])
                with self.assertRaisesRegex(
                    module.FormalContractError, "native deployment identity"
                ):
                    module.validate_contract(contract)

    def test_raw_prompt_and_every_free_run_trajectory_are_hash_pinned_and_pairwise_exact(self):
        module = load_module()
        mutations = (
            (
                "prompt count",
                lambda workloads: workloads["shared_prompt"].__setitem__(
                    "token_count", 12
                ),
            ),
            (
                "prompt hash",
                lambda workloads: workloads["shared_prompt"].__setitem__(
                    "sha256", "0" * 64
                ),
            ),
            (
                "core raw ingress",
                lambda workloads: workloads["CORE_RAW13_FREE128_V3"].__setitem__(
                    "both_arms_receive_raw_token_ids_directly", False
                ),
            ),
            (
                "core trajectory hash",
                lambda workloads: workloads["CORE_RAW13_FREE128_V3"][
                    "trajectory_admission"
                ].__setitem__("expected_sha256", "0" * 64),
            ),
            (
                "core trajectory equality",
                lambda workloads: workloads["CORE_RAW13_FREE128_V3"][
                    "trajectory_admission"
                ].__setitem__("pairwise_A_L_token_ids_must_be_bitwise_equal", False),
            ),
            (
                "gateway rendered prompt",
                lambda workloads: workloads["GATEWAY_RAW13_FREE128_V3"].__setitem__(
                    "backend_rendered_prompt_token_ids_must_equal_shared_prompt", False
                ),
            ),
            (
                "gateway trajectory equality",
                lambda workloads: workloads["GATEWAY_RAW13_FREE128_V3"][
                    "trajectory_admission"
                ].__setitem__("pairwise_B_G_token_ids_must_be_bitwise_equal", False),
            ),
            (
                "native free trajectory hash",
                lambda workloads: workloads["NATIVE_RAW13_FREE128_V3"][
                    "free_run_trajectory_admission"
                ].__setitem__("expected_sha256", "0" * 64),
            ),
            (
                "hash-only admission",
                lambda workloads: workloads["NATIVE_RAW13_FREE128_V3"][
                    "free_run_trajectory_admission"
                ].__setitem__("hash_only_without_raw_ids_is_sufficient", True),
            ),
        )
        for label, mutate in mutations:
            with self.subTest(label=label):
                contract = checked_in_contract()
                mutate(contract["workload_contracts"])
                with self.assertRaisesRegex(module.FormalContractError, "workload"):
                    module.validate_contract(contract)

    def test_native_teacher_forced_admission_requires_all_128_exact_argmax_receipts(self):
        module = load_module()
        mutations = (
            ("steps", lambda teacher: teacher.__setitem__("steps", 127)),
            (
                "teacher inputs",
                lambda teacher: teacher.__setitem__(
                    "same_teacher_input_token_ids_for_AN_and_L", False
                ),
            ),
            (
                "receipt fields",
                lambda teacher: teacher["required_receipt_fields"].remove(
                    "observed_argmax_token_ids"
                ),
            ),
            (
                "AN mismatches",
                lambda teacher: teacher.__setitem__(
                    "AN_zero_argmax_mismatches_required", False
                ),
            ),
            (
                "L mismatches",
                lambda teacher: teacher.__setitem__(
                    "L_zero_argmax_mismatches_required", False
                ),
            ),
            (
                "pairwise argmax",
                lambda teacher: teacher.__setitem__(
                    "AN_and_L_observed_argmax_token_ids_must_be_bitwise_equal", False
                ),
            ),
            (
                "receipt publication",
                lambda teacher: teacher.__setitem__(
                    "receipts_must_be_committed_and_pushed_before_performance_sampling",
                    False,
                ),
            ),
            (
                "free-run substitute",
                lambda teacher: teacher.__setitem__(
                    "prior_free_run_identity_is_not_a_substitute", False
                ),
            ),
        )
        for label, mutate in mutations:
            with self.subTest(label=label):
                contract = checked_in_contract()
                teacher = contract["workload_contracts"][
                    "NATIVE_RAW13_FREE128_V3"
                ]["teacher_forced_admission"]
                mutate(teacher)
                with self.assertRaisesRegex(
                    module.FormalContractError, "teacher-forced"
                ):
                    module.validate_contract(contract)

    def test_native_teacher_inputs_are_prebound_to_prompt_and_canonical_free_prefix(self):
        module = load_module()
        mutations = (
            (
                "short token list",
                lambda teacher: teacher["teacher_input_token_ids"].pop(),
            ),
            (
                "wrong prompt last token",
                lambda teacher: teacher["teacher_input_token_ids"].__setitem__(0, 0),
            ),
            (
                "wrong canonical free prefix",
                lambda teacher: teacher["teacher_input_token_ids"].__setitem__(1, 0),
            ),
            (
                "wrong canonical hash",
                lambda teacher: teacher.__setitem__(
                    "teacher_input_token_ids_sha256", "0" * 64
                ),
            ),
            (
                "derivation not recomputed",
                lambda teacher: teacher.__setitem__(
                    "teacher_input_derivation_must_be_recomputed_and_match_before_each_receipt",
                    False,
                ),
            ),
            (
                "cleared derivation",
                lambda teacher: teacher.__setitem__("teacher_input_derivation", ""),
            ),
        )
        for label, mutate in mutations:
            with self.subTest(label=label):
                contract = checked_in_contract()
                teacher = contract["workload_contracts"][
                    "NATIVE_RAW13_FREE128_V3"
                ]["teacher_forced_admission"]
                mutate(teacher)
                with self.assertRaisesRegex(
                    module.FormalContractError, "teacher prebinding"
                ):
                    module.validate_contract(contract)

    def test_native_timing_includes_top4_rerank_and_full_vocabulary_argmax_to_token_ready(self):
        module = load_module()
        mutations = (
            (
                "endpoint",
                lambda timing: timing.__setitem__(
                    "common_token_ready_boundary", "logits-ready"
                ),
            ),
            (
                "ApxInf rerank",
                lambda timing: timing["ApxInf_boundary_includes"].remove(
                    "F32 tied-embedding exact top-4 rerank"
                ),
            ),
            (
                "llama argmax",
                lambda timing: timing["llama_cpp_boundary_includes"].remove(
                    "sampler or full-vocabulary greedy argmax"
                ),
            ),
            (
                "false internal equality",
                lambda timing: timing.__setitem__("internal_operations_are_identical", True),
            ),
            (
                "endpoint inequality",
                lambda timing: timing.__setitem__("endpoint_semantics_are_identical", False),
            ),
            (
                "selection excluded",
                lambda timing: timing.__setitem__(
                    "sampling_argmax_top4_transfer_or_F32_rerank_may_be_excluded",
                    True,
                ),
            ),
        )
        for label, mutate in mutations:
            with self.subTest(label=label):
                contract = checked_in_contract()
                mutate(contract["timing_contract"]["NATIVE_A_VS_L"])
                with self.assertRaisesRegex(
                    module.FormalContractError, "native timing"
                ):
                    module.validate_contract(contract)

    def test_activation_and_runtime_custody_must_bind_public_git_and_all_hashes(self):
        module = load_module()
        mutations = (
            (
                "public before requests",
                lambda contract: contract["activation_contract"].__setitem__(
                    "predeclaration_must_be_committed_and_pushed_before_any_v3_generation_request",
                    False,
                ),
            ),
            (
                "receipt binding",
                lambda contract: contract["activation_contract"].__setitem__(
                    "campaign_start_receipt_must_record_predeclaration_commit_size_and_sha256",
                    False,
                ),
            ),
            (
                "HEAD binding",
                lambda contract: contract["activation_contract"].__setitem__(
                    "campaign_start_receipt_must_prove_head_equals_origin_main", False
                ),
            ),
            (
                "clean tree",
                lambda contract: contract["activation_contract"].__setitem__(
                    "campaign_start_receipt_must_prove_clean_worktree", False
                ),
            ),
            (
                "post-start edits",
                lambda contract: contract["activation_contract"].__setitem__(
                    "editing_this_contract_after_the_first_v3_generation_request_allowed",
                    True,
                ),
            ),
            (
                "null ApxInf hashes no longer block",
                lambda contract: contract["runtime_custody"]["ApxInf_native"].__setitem__(
                    "null_hashes_block_sampling", False
                ),
            ),
            (
                "model hash",
                lambda contract: contract["source_model_custody"]["checkpoint"].__setitem__(
                    "sha256", "0" * 64
                ),
            ),
            (
                "llama binary hash",
                lambda contract: contract["runtime_custody"][
                    "pinned_llama_cpp_core"
                ].__setitem__("runner_binary_sha256", "0" * 64),
            ),
            (
                "OmniInfer hash",
                lambda contract: contract["runtime_custody"]["gateway_cohort"][
                    "omniinfer"
                ].__setitem__("cli_sha256", "0" * 64),
            ),
            (
                "backend hash",
                lambda contract: contract["runtime_custody"]["gateway_cohort"][
                    "backend"
                ].__setitem__("llama_server_sha256", "0" * 64),
            ),
            (
                "marker overwrite",
                lambda contract: contract["failure_contract"][
                    "subcampaign_markers"
                ]["NATIVE_A_VS_L"].__setitem__("create_new_only", False),
            ),
            (
                "marker not public",
                lambda contract: contract["failure_contract"][
                    "subcampaign_markers"
                ]["GATEWAY_B_VS_G"].__setitem__(
                    "must_be_committed_and_pushed_before_first_gateway_generation_request",
                    False,
                ),
            ),
        )
        for label, mutate in mutations:
            with self.subTest(label=label):
                contract = checked_in_contract()
                mutate(contract)
                with self.assertRaisesRegex(
                    module.FormalContractError, "activation|custody"
                ):
                    module.validate_contract(contract)

    def test_activation_requires_live_frozen_github_remote_not_a_local_ref_substitute(self):
        module = load_module()
        mutations = (
            (
                "missing frozen origin",
                lambda activation: activation.pop("frozen_origin_remote_url"),
            ),
            (
                "different remote",
                lambda activation: activation.__setitem__(
                    "frozen_origin_remote_url", "https://example.invalid/ApxInf.git"
                ),
            ),
            (
                "local tracking ref substitute",
                lambda activation: activation.__setitem__(
                    "local_tracking_ref_equality_is_sufficient_publication_proof",
                    True,
                ),
            ),
        )
        for label, mutate in mutations:
            with self.subTest(label=label):
                contract = checked_in_contract()
                mutate(contract["activation_contract"])
                with self.assertRaisesRegex(
                    module.FormalContractError, "live remote publication"
                ):
                    module.validate_contract(contract)

    def test_gateway_edge_is_same_backend_same_request_client_wall_only(self):
        module = load_module()
        mutations = (
            (
                "resident process lifetime",
                lambda contract: contract["execution_protocol"][
                    "GATEWAY_B_VS_G"
                ].__setitem__("process_state", "fresh-process-per-sample"),
            ),
            (
                "client connections",
                lambda contract: contract["execution_protocol"][
                    "GATEWAY_B_VS_G"
                ].__setitem__("client_connections", "new connection per request"),
            ),
            (
                "warmup count",
                lambda contract: contract["execution_protocol"][
                    "GATEWAY_B_VS_G"
                ].__setitem__("untimed_warmups_per_arm", 2),
            ),
            (
                "generation finish",
                lambda contract: contract["workload_contracts"][
                    "GATEWAY_RAW13_FREE128_V3"
                ]["generation"].__setitem__("finish_reason", "stop"),
            ),
            (
                "cache acknowledgement",
                lambda contract: contract["workload_contracts"][
                    "GATEWAY_RAW13_FREE128_V3"
                ]["generation"].__setitem__(
                    "cache_clear_acknowledgement_before_every_arm", [1]
                ),
            ),
            (
                "same backend",
                lambda contract: contract["comparison_graph"]["edges"][
                    "GATEWAY_B_VS_G"
                ].__setitem__("same_backend_process_required", False),
            ),
            (
                "same model",
                lambda contract: contract["comparison_graph"]["edges"][
                    "GATEWAY_B_VS_G"
                ].__setitem__("same_loaded_model_file_description_required", False),
            ),
            (
                "engine ranking",
                lambda contract: contract["comparison_graph"]["edges"][
                    "GATEWAY_B_VS_G"
                ].__setitem__("engine_ranking_edge", True),
            ),
            (
                "same process identity",
                lambda contract: contract["runtime_custody"]["gateway_cohort"][
                    "backend"
                ].__setitem__(
                    "same_pid_start_time_argv_environment_and_loaded_model_fd_for_B_and_G",
                    False,
                ),
            ),
            (
                "different body",
                lambda contract: contract["workload_contracts"][
                    "GATEWAY_RAW13_FREE128_V3"
                ]["request"].__setitem__("same_client_body_for_B_and_G", False),
            ),
            (
                "cache enabled",
                lambda contract: contract["workload_contracts"][
                    "GATEWAY_RAW13_FREE128_V3"
                ]["request"].__setitem__("cache_prompt", True),
            ),
            (
                "early client stop",
                lambda contract: contract["timing_contract"][
                    "GATEWAY_B_VS_G"
                ].__setitem__("end", "after-response-headers"),
            ),
            (
                "cross-edge subtraction",
                lambda contract: contract["timing_contract"][
                    "GATEWAY_B_VS_G"
                ].__setitem__(
                    "gateway_overhead_subtraction_from_ApxInf_or_llama_core_allowed",
                    True,
                ),
            ),
            (
                "premature gateway admission",
                lambda contract: contract["runtime_custody"]["gateway_cohort"][
                    "current_readiness"
                ].__setitem__("formally_admitted", True),
            ),
        )
        for label, mutate in mutations:
            with self.subTest(label=label):
                contract = checked_in_contract()
                mutate(contract)
                with self.assertRaisesRegex(module.FormalContractError, "gateway"):
                    module.validate_contract(contract)

    def test_claim_and_machine_receipt_require_every_gate_and_forbid_cross_edge_relabeling(self):
        module = load_module()
        mutations = (
            (
                "native gate",
                lambda contract: contract["machine_receipt_contract"][
                    "required_true_gate_ids_for_NATIVE_A_VS_L"
                ].remove("TEACHER_FORCED_128_EXACT"),
            ),
            (
                "core gate",
                lambda contract: contract["machine_receipt_contract"][
                    "required_true_gate_ids_for_CORE_A_VS_L"
                ].remove("Q8_0_PREFILL_STATE_POLICY_EQUAL"),
            ),
            (
                "gateway gate",
                lambda contract: contract["machine_receipt_contract"][
                    "required_true_gate_ids_for_GATEWAY_B_VS_G"
                ].remove("SAME_RESIDENT_BACKEND_PROCESS"),
            ),
            (
                "missing gate allowed",
                lambda contract: contract["machine_receipt_contract"].__setitem__(
                    "missing_gate_is_failure", False
                ),
            ),
            (
                "null hash allowed",
                lambda contract: contract["machine_receipt_contract"].__setitem__(
                    "null_required_hash_is_failure", False
                ),
            ),
            (
                "native-as-engine claim",
                lambda contract: contract["claim_policy"]["always_forbidden"].remove(
                    "relabeling the NATIVE_A_VS_L result as ApxInf-engine versus llama.cpp-engine performance"
                ),
            ),
            (
                "A-vs-G claim",
                lambda contract: contract["claim_policy"]["always_forbidden"].remove(
                    "ApxInf versus OmniInfer end-to-end speed ratio delta winner or ranking"
                ),
            ),
            (
                "cross-edge join",
                lambda contract: contract["comparison_graph"].__setitem__(
                    "cross_edge_result_join_allowed", True
                ),
            ),
            (
                "missing forbidden edge",
                lambda contract: contract["comparison_graph"]["forbidden_edges"].remove(
                    "A_VS_G_END_TO_END"
                ),
            ),
        )
        for label, mutate in mutations:
            with self.subTest(label=label):
                contract = checked_in_contract()
                mutate(contract)
                with self.assertRaisesRegex(
                    module.FormalContractError, "claim|receipt"
                ):
                    module.validate_contract(contract)

    def test_dynamic_receipt_and_subcampaign_marker_bindings_are_exactly_frozen(self):
        module = load_module()
        mutations = (
            (
                "missing live oid",
                lambda receipt: receipt["required_dynamic_contract_binding"][
                    "required_fields"
                ].remove("ls_remote_live_oid"),
            ),
            (
                "cleared constraints",
                lambda receipt: receipt["required_dynamic_contract_binding"].__setitem__(
                    "constraints", []
                ),
            ),
            (
                "missing binding allowed",
                lambda receipt: receipt["required_dynamic_contract_binding"].__setitem__(
                    "any_missing_or_unresolved_dynamic_binding_is_failure", False
                ),
            ),
            (
                "cleared marker bindings",
                lambda receipt: receipt.__setitem__("subcampaign_marker_bindings", {}),
            ),
            (
                "tampered marker path",
                lambda receipt: receipt["subcampaign_marker_bindings"][
                    "CORE_A_VS_L"
                ].__setitem__("expected_marker_repository_path", "wrong.json"),
            ),
        )
        for label, mutate in mutations:
            with self.subTest(label=label):
                contract = checked_in_contract()
                mutate(contract["machine_receipt_contract"])
                with self.assertRaisesRegex(
                    module.FormalContractError, "dynamic receipt|marker bindings"
                ):
                    module.validate_contract(contract)

    def test_failure_dispatch_common_and_per_edge_hard_stops_are_exactly_frozen(self):
        module = load_module()
        mutations = (
            (
                "tampered dispatch",
                lambda failure: failure.__setitem__("hard_stop_dispatch", ""),
            ),
            (
                "inactive edge applies",
                lambda failure: failure.__setitem__(
                    "inactive_edge_hard_stop_conditions_apply", True
                ),
            ),
            (
                "cleared common stops",
                lambda failure: failure.__setitem__("common_hard_stop_conditions", []),
            ),
            (
                "cleared native stops",
                lambda failure: failure["per_edge_hard_stop_conditions"].__setitem__(
                    "NATIVE_A_VS_L", []
                ),
            ),
            (
                "deleted gateway stops",
                lambda failure: failure["per_edge_hard_stop_conditions"].pop(
                    "GATEWAY_B_VS_G"
                ),
            ),
        )
        for label, mutate in mutations:
            with self.subTest(label=label):
                contract = checked_in_contract()
                mutate(contract["failure_contract"])
                with self.assertRaisesRegex(
                    module.FormalContractError, "failure hard-stop"
                ):
                    module.validate_contract(contract)

    def test_claim_policy_gate_bound_allowlist_is_exact_not_just_nonempty(self):
        module = load_module()
        mutations = (
            (
                "deleted decision",
                lambda allowed: allowed.pop(),
            ),
            (
                "cleared decisions",
                lambda allowed: allowed.clear(),
            ),
            (
                "tampered decision",
                lambda allowed: allowed.__setitem__(
                    0, "generic engine ranking after any convenient gate passes"
                ),
            ),
        )
        for label, mutate in mutations:
            with self.subTest(label=label):
                contract = checked_in_contract()
                mutate(
                    contract["claim_policy"][
                        "allowed_only_after_corresponding_machine_gates_pass"
                    ]
                )
                with self.assertRaisesRegex(module.FormalContractError, "claim policy"):
                    module.validate_contract(contract)


if __name__ == "__main__":
    unittest.main()
