import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "scripts" / "validate_mlx_multi_prompt_quality.py"
CONTRACT_PATH = ROOT / "configs" / "qwen35-0.8b-mlx-multi-prompt-quality-v1.json"
REAL_W4_ENVELOPE_PATH = (
    ROOT
    / "doc"
    / "20260823-qwen35-macos-bringup"
    / "qwen35-w4-multi-prompt-quality-v1.json"
)


def _module():
    spec = importlib.util.spec_from_file_location(
        "validate_mlx_multi_prompt_quality", MODULE_PATH
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


def _synthetic_evidence(
    module,
    contract,
    *,
    claim="fixed-suite-exact-parity",
    precision_profile="mixed-w4-w8-bf16",
):
    records = []
    for prompt_index, prompt in enumerate(contract["suite"]["prompts"]):
        teacher = [
            1000 + prompt_index * 100 + step for step in range(prompt["teacher_steps"])
        ]
        records.append(
            {
                "prompt_id": prompt["id"],
                "prompt_token_ids": list(prompt["prompt_token_ids"]),
                "teacher_steps": prompt["teacher_steps"],
                "reference": {
                    "precision": "bf16",
                    "runs": [list(teacher), list(teacher)],
                    "run_sha256s": [
                        module.object_sha256(teacher),
                        module.object_sha256(teacher),
                    ],
                },
                "candidate": {
                    "precision_profile": precision_profile,
                    "runs": [list(teacher), list(teacher)],
                    "run_sha256s": [
                        module.object_sha256(teacher),
                        module.object_sha256(teacher),
                    ],
                },
            }
        )
    return {
        "format": "apxinf-mlx-multi-prompt-quality-evidence-v1",
        "schema_version": 1,
        "contract_sha256": contract["content_sha256"],
        "execution": json.loads(module.canonical_bytes(contract["generation"])),
        "candidate": {
            "candidate_id": "synthetic-mixed-candidate",
            "precision_profile": precision_profile,
            "requested_claim": claim,
            "claims_general_parity": False,
        },
        "records": records,
    }


def _run_cli(evidence_path):
    return subprocess.run(
        [
            "/usr/bin/python3",
            "-I",
            "-B",
            str(MODULE_PATH),
            "--contract",
            str(CONTRACT_PATH),
            "--evidence",
            str(evidence_path),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def _write_rehashed_envelope(module, path, envelope):
    envelope.pop("content_sha256", None)
    envelope["content_sha256"] = module.object_sha256(envelope)
    path.write_bytes(module.canonical_bytes(envelope) + b"\n")


class MultiPromptQualityContractTests(unittest.TestCase):
    def test_contract_freezes_four_domains_and_production_generate_step(self):
        module = _module()
        contract = module.load_contract(CONTRACT_PATH)

        self.assertEqual(
            [prompt["domain"] for prompt in contract["suite"]["prompts"]],
            ["english", "chinese", "code", "math_structured"],
        )
        self.assertEqual(contract["generation"]["api"], "mlx_lm.generate.generate_step")
        self.assertEqual(
            contract["generation"]["semantics"],
            "mlx-generate-step-argmax-v1",
        )
        self.assertEqual(contract["generation"]["repeat_count"], 2)
        self.assertTrue(
            all(
                prompt["teacher_steps"] >= 32 for prompt in contract["suite"]["prompts"]
            )
        )
        self.assertGreaterEqual(
            sum(
                prompt["teacher_steps"] >= 64 for prompt in contract["suite"]["prompts"]
            ),
            2,
        )

    def test_exact_mode_accepts_all_four_prompts_with_two_identical_bf16_and_candidate_runs(
        self,
    ):
        module = _module()
        contract = module.load_contract(CONTRACT_PATH)
        evidence = _synthetic_evidence(module, contract)

        receipt = module.validate_evidence(contract, evidence)

        self.assertTrue(receipt["accepted"])
        self.assertEqual(receipt["claim"], "fixed-suite-exact-parity")
        self.assertEqual(receipt["prompt_count"], 4)
        self.assertTrue(all(item["exact"] for item in receipt["prompts"]))
        self.assertFalse(receipt["claims_general_parity"])

    def test_hybrid_w8_bf16_profile_is_a_supported_fixed_suite_tier(self):
        module = _module()
        contract = module.load_contract(CONTRACT_PATH)
        evidence = _synthetic_evidence(
            module,
            contract,
            precision_profile="hybrid-w8-bf16-g64",
        )

        receipt = module.validate_evidence(contract, evidence)

        self.assertTrue(receipt["accepted"])
        self.assertEqual(receipt["precision_profile"], "hybrid-w8-bf16-g64")
        self.assertFalse(receipt["claims_general_parity"])

    def test_chinese_top1_counterfactual_is_only_a_fixed_suite_tier(self):
        module = _module()
        contract = module.load_contract(CONTRACT_PATH)
        evidence = _synthetic_evidence(
            module,
            contract,
            precision_profile=(
                "hybrid-w8-bf16-g64-chinese-top1-counterfactual-v1"
            ),
        )

        receipt = module.validate_evidence(contract, evidence)

        self.assertTrue(receipt["accepted"])
        self.assertEqual(
            receipt["precision_profile"],
            "hybrid-w8-bf16-g64-chinese-top1-counterfactual-v1",
        )
        self.assertFalse(receipt["claims_general_parity"])

    def test_reference_and_candidate_each_require_two_identical_runs(self):
        module = _module()
        contract = module.load_contract(CONTRACT_PATH)
        for lane in ("reference", "candidate"):
            with self.subTest(lane=lane):
                evidence = _synthetic_evidence(module, contract)
                second = evidence["records"][0][lane]["runs"][1]
                second[-1] += 1
                evidence["records"][0][lane]["run_sha256s"][1] = module.object_sha256(
                    second
                )
                with self.assertRaisesRegex(module.QualityGateError, "not identical"):
                    module.validate_evidence(contract, evidence)

    def test_threshold_mode_uses_explicit_per_prompt_prefix_and_ratio_floors(self):
        module = _module()
        contract = module.load_contract(CONTRACT_PATH)
        evidence = _synthetic_evidence(
            module,
            contract,
            claim="fixed-suite-threshold-match",
        )
        for record in evidence["records"]:
            teacher = record["reference"]["runs"][0]
            observed = list(teacher)
            mismatch_count = len(observed) // 4
            for index in range(len(observed) - mismatch_count, len(observed)):
                observed[index] = (observed[index] + 1) % 248320
            digest = module.object_sha256(observed)
            record["candidate"]["runs"] = [list(observed), list(observed)]
            record["candidate"]["run_sha256s"] = [digest, digest]

        receipt = module.validate_evidence(contract, evidence)

        self.assertTrue(receipt["accepted"])
        self.assertEqual(receipt["claim"], "fixed-suite-threshold-match")
        self.assertTrue(all(not item["exact"] for item in receipt["prompts"]))
        self.assertTrue(
            all(item["position_match_ratio"] >= 0.75 for item in receipt["prompts"])
        )
        self.assertFalse(receipt["claims_general_parity"])

    def test_exact_claim_fails_if_any_noncanonical_prompt_diverges(self):
        module = _module()
        contract = module.load_contract(CONTRACT_PATH)
        evidence = _synthetic_evidence(module, contract)
        record = evidence["records"][1]
        observed = list(record["candidate"]["runs"][0])
        observed[-1] += 1
        digest = module.object_sha256(observed)
        record["candidate"]["runs"] = [list(observed), list(observed)]
        record["candidate"]["run_sha256s"] = [digest, digest]

        receipt = module.validate_evidence(contract, evidence)

        self.assertFalse(receipt["accepted"])
        self.assertIsNone(receipt["claim"])
        self.assertIn("chinese-explanation", receipt["problems"][0])

    def test_threshold_claim_fails_when_one_prompt_drops_below_frozen_ratio(self):
        module = _module()
        contract = module.load_contract(CONTRACT_PATH)
        evidence = _synthetic_evidence(
            module,
            contract,
            claim="fixed-suite-threshold-match",
        )
        record = evidence["records"][2]
        observed = list(record["candidate"]["runs"][0])
        for index in range(8, len(observed)):
            observed[index] = (observed[index] + 1) % 248320
        digest = module.object_sha256(observed)
        record["candidate"]["runs"] = [list(observed), list(observed)]
        record["candidate"]["run_sha256s"] = [digest, digest]

        receipt = module.validate_evidence(contract, evidence)

        self.assertFalse(receipt["accepted"])
        self.assertIsNone(receipt["claim"])
        self.assertIn("python-code", receipt["problems"][0])

    def test_single_prompt_or_general_parity_claim_fails_closed(self):
        module = _module()
        contract = module.load_contract(CONTRACT_PATH)
        single_prompt = _synthetic_evidence(module, contract)
        single_prompt["records"] = single_prompt["records"][:1]
        with self.assertRaisesRegex(module.QualityGateError, "every fixed prompt"):
            module.validate_evidence(contract, single_prompt)

        universal = _synthetic_evidence(module, contract)
        universal["candidate"]["claims_general_parity"] = True
        universal["candidate"]["requested_claim"] = "general-parity"
        with self.assertRaisesRegex(module.QualityGateError, "fixed-suite"):
            module.validate_evidence(contract, universal)

    def test_semantics_drift_cannot_be_relabelled_as_production_reference(self):
        module = _module()
        contract = module.load_contract(CONTRACT_PATH)
        evidence = _synthetic_evidence(module, contract)
        evidence["execution"]["semantics"] = "manual-model-call-argmax"

        with self.assertRaisesRegex(
            module.QualityGateError, "production generate_step"
        ):
            module.validate_evidence(contract, evidence)

    def test_cli_emits_one_offline_receipt_without_loading_a_model(self):
        module = _module()
        contract = module.load_contract(CONTRACT_PATH)
        evidence = _synthetic_evidence(module, contract)
        with tempfile.TemporaryDirectory() as temporary:
            evidence_path = Path(temporary) / "synthetic-evidence.json"
            evidence_path.write_bytes(module.canonical_bytes(evidence) + b"\n")
            completed = subprocess.run(
                [
                    "/usr/bin/python3",
                    "-I",
                    "-B",
                    str(MODULE_PATH),
                    "--contract",
                    str(CONTRACT_PATH),
                    "--evidence",
                    str(evidence_path),
                ],
                cwd=ROOT,
                check=False,
                capture_output=True,
                text=True,
            )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(len(completed.stdout.splitlines()), 1)
        receipt = json.loads(completed.stdout)
        self.assertTrue(receipt["accepted"])
        self.assertEqual(receipt["prompt_count"], 4)

    def test_cli_accepts_the_real_producer_envelope_and_recomputes_its_failed_receipt(
        self,
    ):
        envelope = json.loads(REAL_W4_ENVELOPE_PATH.read_text(encoding="utf-8"))

        completed = subprocess.run(
            [
                "/usr/bin/python3",
                "-I",
                "-B",
                str(MODULE_PATH),
                "--contract",
                str(CONTRACT_PATH),
                "--evidence",
                str(REAL_W4_ENVELOPE_PATH),
            ],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(completed.returncode, 1, completed.stderr)
        self.assertEqual(json.loads(completed.stdout), envelope["validation_receipt"])

    def test_cli_rejects_rehashed_envelope_if_bundle_custody_changes(self):
        module = _module()
        envelope = json.loads(REAL_W4_ENVELOPE_PATH.read_text(encoding="utf-8"))
        envelope["custody"]["bundles"]["candidate"]["after"]["total_bytes"] += 1
        with tempfile.TemporaryDirectory() as temporary:
            evidence_path = Path(temporary) / "tampered-envelope.json"
            _write_rehashed_envelope(module, evidence_path, envelope)
            completed = _run_cli(evidence_path)

        self.assertEqual(completed.returncode, 2, completed.stderr)
        self.assertIn("custody", json.loads(completed.stdout)["problems"][0])

    def test_cli_fails_closed_on_outer_hash_policy_or_receipt_tampering(self):
        module = _module()
        original = json.loads(REAL_W4_ENVELOPE_PATH.read_text(encoding="utf-8"))
        cases = (
            (
                "self hash",
                lambda envelope: envelope.__setitem__("status", "accepted"),
                False,
                "content_sha256",
            ),
            (
                "policy",
                lambda envelope: envelope["policy"].__setitem__(
                    "network", "network-allowed"
                ),
                True,
                "policy",
            ),
            (
                "inner receipt",
                lambda envelope: envelope["validation_receipt"].__setitem__(
                    "evidence_sha256", "0" * 64
                ),
                True,
                "receipt",
            ),
        )
        for label, mutate, rehash, expected_problem in cases:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as temporary:
                envelope = json.loads(module.canonical_bytes(original))
                mutate(envelope)
                evidence_path = Path(temporary) / "tampered-envelope.json"
                if rehash:
                    _write_rehashed_envelope(module, evidence_path, envelope)
                else:
                    evidence_path.write_bytes(module.canonical_bytes(envelope) + b"\n")
                completed = _run_cli(evidence_path)

            self.assertEqual(completed.returncode, 2, completed.stderr)
            self.assertIn(
                expected_problem,
                json.loads(completed.stdout)["problems"][0],
            )


if __name__ == "__main__":
    unittest.main()
