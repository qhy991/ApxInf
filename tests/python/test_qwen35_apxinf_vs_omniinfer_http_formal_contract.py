from __future__ import annotations

import contextlib
import copy
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
CONTRACT_PATH = ROOT / "configs" / "qwen35-0.8b-apxinf-vs-omniinfer-http-formal-v1.json"
MODULE_PATH = (
    ROOT / "scripts" / "validate_qwen35_apxinf_vs_omniinfer_http_formal_contract.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location(
        "validate_qwen35_apxinf_vs_omniinfer_http_formal_contract_for_tests",
        MODULE_PATH,
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


class Qwen35ApxInfVsOmniInferHttpFormalContractTests(unittest.TestCase):
    def test_checked_in_static_contract_is_valid_but_campaign_is_not_ready(self):
        module = load_module()

        contract, raw = module.load_contract(CONTRACT_PATH)

        self.assertEqual(contract["deployment_edge_id"], "DEPLOYMENT_AH_VS_OG")
        self.assertEqual(hashlib.sha256(raw).hexdigest(), module.PINNED_FILE_SHA256)
        self.assertFalse(contract["current_readiness"]["formal_campaign_may_start_now"])
        self.assertIn(
            "FORMAL_V1_DRIVER_NOT_IMPLEMENTED",
            contract["current_readiness"]["blocker_codes"],
        )
        self.assertEqual(
            contract["lineage"]["formal_driver_status"],
            "NOT_IMPLEMENTED_BY_THIS_CONTRACT_SLICE",
        )

    def test_cli_emits_static_only_machine_readable_validation_receipt(self):
        completed = subprocess.run(
            ["/usr/bin/python3", "-I", "-B", str(MODULE_PATH)],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        receipt = json.loads(completed.stdout)
        self.assertTrue(receipt["valid"])
        self.assertEqual(receipt["validation_scope"], "STATIC_PREDECLARATION_ONLY")
        self.assertEqual(receipt["deployment_edge_id"], "DEPLOYMENT_AH_VS_OG")
        self.assertFalse(receipt["formal_campaign_ready"])
        self.assertFalse(receipt["formal_driver_implemented_by_validator"])
        self.assertFalse(receipt["services_started_or_generation_requests_made"])
        self.assertFalse(receipt["future_run_receipt_validated"])
        self.assertFalse(receipt["prior_nonformal_evidence_upgraded"])
        self.assertIsNone(receipt["performance_result"])

    def test_required_semantic_gates_reject_negative_mutations(self):
        module = load_module()
        mutations = (
            (
                "independent deployment edge",
                "deployment edge ID",
                lambda value: value.__setitem__("deployment_edge_id", "A_VS_G"),
            ),
            (
                "nonformal samples",
                "diagnostic lineage",
                lambda value: value["lineage"].__setitem__(
                    "existing_nonformal_samples_reused", True
                ),
            ),
            (
                "serialized artifacts differ",
                "named deployment boundary",
                lambda value: value["named_deployment_boundary"].__setitem__(
                    "same_serialized_weight_bytes", True
                ),
            ),
            (
                "trajectories need not match",
                "named deployment boundary",
                lambda value: value["named_deployment_boundary"].__setitem__(
                    "cross_arm_generated_trajectory_equality_required", True
                ),
            ),
            (
                "canonical request",
                "383-byte request object",
                lambda value: value["workload_contract"]["request"][
                    "canonical_json_object"
                ].__setitem__("max_tokens", 127),
            ),
            (
                "raw 13 token prompt",
                "raw 13-token prompt",
                lambda value: value["workload_contract"]["prompt"][
                    "token_ids"
                ].__setitem__(0, 1),
            ),
            (
                "single external tokenize preflight",
                "OmniInfer /tokenize",
                lambda value: value["workload_contract"][
                    "omniinfer_external_tokenize_admission"
                ].__setitem__("expected_request_count", 136),
            ),
            (
                "no tokenize between slots",
                "OmniInfer /tokenize",
                lambda value: value["workload_contract"][
                    "omniinfer_external_tokenize_admission"
                ].__setitem__("repeated_during_warmup_or_measured_slots_allowed", True),
            ),
            (
                "tokenize parses special tokens",
                "OmniInfer /tokenize",
                lambda value: value["workload_contract"][
                    "omniinfer_external_tokenize_admission"
                ]["tokenize_request"].__setitem__("parse_special", False),
            ),
            (
                "tokenize returns raw integer IDs",
                "OmniInfer /tokenize",
                lambda value: value["workload_contract"][
                    "omniinfer_external_tokenize_admission"
                ]["tokenize_request"].__setitem__("with_pieces", True),
            ),
            (
                "tokenize response exact schema",
                "OmniInfer /tokenize",
                lambda value: value["workload_contract"][
                    "omniinfer_external_tokenize_admission"
                ].__setitem__("response_required_exact_keys", ["tokens", "model"]),
            ),
            (
                "five EOG negative infinity",
                "five-EOG",
                lambda value: value["workload_contract"][
                    "suppressed_eog_policy"
                ].__setitem__("token_ids", [248044, 248046, 248063, 248064]),
            ),
            (
                "AH launch features",
                "AH formal launch",
                lambda value: value["named_deployment_boundary"]["arms"]["AH"][
                    "formal_runtime_requirements"
                ]["cargo_build_argv_without_toolchain_path"].__setitem__(9, "metal-w8"),
            ),
            (
                "OG GPU layers",
                "OG formal launch",
                lambda value: value["named_deployment_boundary"]["arms"]["OG"][
                    "formal_runtime_requirements"
                ].__setitem__("gpu_layers", 0),
            ),
            (
                "AH full path counters",
                "compact generation-path",
                lambda value: value["workload_contract"][
                    "apxinf_compact_generation_path_external_validation"
                ]["decode_head_required_values"].__setitem__("excluded_calls", 0),
            ),
            (
                "cold slot each request",
                "cold-cache",
                lambda value: value["workload_contract"][
                    "cold_slot_contract"
                ].__setitem__("clear_immediately_before_every_arm_request", False),
            ),
            (
                "exact OG clear endpoint",
                "cold-cache",
                lambda value: value["workload_contract"][
                    "cold_slot_contract"
                ].__setitem__(
                    "OG_clear_method_path_and_body",
                    ["POST", "/slots/0?action=erase", ""],
                ),
            ),
            (
                "fixed 16 blocks",
                "paired 16-block schedule",
                lambda value: value["schedule_contract"].__setitem__(
                    "measured_blocks", 8
                ),
            ),
            (
                "parse and semantics inside wall",
                "full HTTP wall timing",
                lambda value: value["timing_contract"].__setitem__(
                    "semantic_response_validation_before_end", False
                ),
            ),
            (
                "quiet passed boolean not trusted",
                "arbitrary passed=true",
                lambda value: value["quiet_host_receipt_contract"].__setitem__(
                    "passed_field_is_sufficient_without_recomputation", True
                ),
            ),
            (
                "continuous host coverage",
                "continuous pre/during/post",
                lambda value: value["quiet_host_receipt_contract"][
                    "continuous"
                ].__setitem__("covers_all_warmup_and_measured_requests", False),
            ),
            (
                "live loaded libraries",
                "loaded-library custody",
                lambda value: value["runtime_custody_contract"][
                    "loaded_library_custody"
                ].__setitem__(
                    "closure_root_start_end_and_all_checkpoints_equal", False
                ),
            ),
            (
                "live model FD",
                "model file-descriptor custody",
                lambda value: value["runtime_custody_contract"][
                    "model_file_descriptor_custody"
                ].__setitem__("path_only_or_hash_only_is_sufficient", True),
            ),
            (
                "one shot marker",
                "one-shot prepare/run marker",
                lambda value: value["execution_protocol"][
                    "one_shot_marker"
                ].__setitem__("second_run_under_same_campaign_id_allowed", True),
            ),
            (
                "durable per slot",
                "crash-safe durable per-slot",
                lambda value: value["execution_protocol"][
                    "raw_per_slot_receipts"
                ].__setitem__("aggregate_only_receipt_is_sufficient", True),
            ),
            (
                "external compact path validation",
                "Apx compact-path external validation",
                lambda value: value["machine_receipt_contract"].__setitem__(
                    "apxinf_server_boolean_or_compact_receipt_without_external_field_validation_is_sufficient",
                    True,
                ),
            ),
            (
                "df 15",
                "16 blocks, df=15",
                lambda value: value["statistics_and_decision_contract"].__setitem__(
                    "degrees_of_freedom", 14
                ),
            ),
            (
                "same halves",
                "same-halves",
                lambda value: value["statistics_and_decision_contract"][
                    "stability_gates"
                ].__setitem__(
                    "first8_and_last8_ratios_must_support_same_decision", False
                ),
            ),
            (
                "five percent threshold",
                "5% thresholds",
                lambda value: value["statistics_and_decision_contract"].__setitem__(
                    "practical_thresholds_AH_over_OG", [0.90, 1.10]
                ),
            ),
            (
                "not ready",
                "must not claim",
                lambda value: value["current_readiness"].__setitem__(
                    "formal_campaign_may_start_now", True
                ),
            ),
            (
                "validator only",
                "validator scope",
                lambda value: value["validator_scope"].__setitem__(
                    "implements_formal_driver", True
                ),
            ),
        )
        for label, pattern, mutate in mutations:
            with self.subTest(label=label):
                contract = copy.deepcopy(checked_in_contract())
                mutate(contract)
                with self.assertRaisesRegex(module.FormalContractError, pattern):
                    module.validate_contract(contract)

    def test_whole_document_semantic_pin_rejects_otherwise_unchecked_drift(self):
        module = load_module()
        contract = checked_in_contract()
        contract["authored_at_utc"] = "2026-08-26T09:20:01Z"

        with self.assertRaisesRegex(
            module.FormalContractError, "canonical semantic pin"
        ):
            module.validate_contract(contract)

    def test_loader_rejects_same_semantics_with_different_file_bytes(self):
        module = load_module()
        compact = json.dumps(
            checked_in_contract(), ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reformatted.json"
            path.write_bytes(compact)
            with self.assertRaisesRegex(
                module.FormalContractError, "file-byte pin mismatch"
            ):
                module.load_contract(path)

    def test_cli_rejects_semantically_weakened_contract(self):
        contract = checked_in_contract()
        contract["quiet_host_receipt_contract"][
            "passed_field_is_sufficient_without_recomputation"
        ] = True
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weakened.json"
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
        self.assertIn("arbitrary passed=true", completed.stderr)

    def test_parser_rejects_duplicate_nonfinite_trailing_bom_and_invalid_utf8(self):
        module = load_module()
        invalid_documents = (
            (b'{"x":1,"x":2}', "duplicate key"),
            (b'{"x":NaN}', "non-finite"),
            (b'{"x":1}[]', "trailing"),
            (b'\xef\xbb\xbf{"x":1}', "BOM"),
            (b'{"x":"\xff"}', "strict UTF-8"),
        )
        for raw, pattern in invalid_documents:
            with self.subTest(pattern=pattern):
                with self.assertRaisesRegex(module.FormalContractError, pattern):
                    module.parse_strict_json(raw)

    def test_cli_reads_one_snapshot_for_hash_and_parse(self):
        module = load_module()
        raw = CONTRACT_PATH.read_bytes()
        metadata = CONTRACT_PATH.lstat()
        stdout = io.StringIO()
        stderr = io.StringIO()
        with (
            mock.patch.object(
                module.Path, "lstat", return_value=metadata
            ) as lstat_call,
            mock.patch.object(module.Path, "read_bytes", return_value=raw) as read_call,
            contextlib.redirect_stdout(stdout),
            contextlib.redirect_stderr(stderr),
        ):
            returncode = module.main(["--contract", str(CONTRACT_PATH)])

        self.assertEqual(returncode, 0, stderr.getvalue())
        self.assertEqual(lstat_call.call_count, 1)
        self.assertEqual(read_call.call_count, 1)
        receipt = json.loads(stdout.getvalue())
        self.assertEqual(receipt["contract_file_size_bytes"], len(raw))
        self.assertEqual(
            receipt["contract_file_sha256"], hashlib.sha256(raw).hexdigest()
        )


if __name__ == "__main__":
    unittest.main()
