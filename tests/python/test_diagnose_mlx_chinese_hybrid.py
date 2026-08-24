from __future__ import annotations

from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "scripts" / "diagnose_mlx_chinese_hybrid.py"
REAL_HYBRID_EVIDENCE = (
    ROOT
    / "doc"
    / "20260823-qwen35-macos-bringup"
    / "qwen35-hybrid-w8-bf16-g64-multi-prompt-quality-v1.json"
)
RETAINED_BF16_PATHS = [
    "language_model.model.layers.12.linear_attn.out_proj",
    "language_model.model.layers.14.linear_attn.out_proj",
    "language_model.model.layers.20.linear_attn.out_proj",
]


def _hybrid_w8_paths():
    paths = ["language_model.model.embed_tokens"]
    for layer in range(24):
        prefix = f"language_model.model.layers.{layer}"
        if (layer + 1) % 4:
            paths.extend(
                f"{prefix}.linear_attn.{name}"
                for name in (
                    "in_proj_a",
                    "in_proj_b",
                    "in_proj_qkv",
                    "in_proj_z",
                    "out_proj",
                )
            )
        else:
            paths.extend(
                f"{prefix}.self_attn.{name}"
                for name in ("k_proj", "o_proj", "q_proj", "v_proj")
            )
        paths.extend(
            f"{prefix}.mlp.{name}" for name in ("down_proj", "gate_proj", "up_proj")
        )
    retained = set(RETAINED_BF16_PATHS)
    return [path for path in paths if path not in retained]


W8_PATHS = _hybrid_w8_paths()


def _module():
    specification = importlib.util.spec_from_file_location(
        "diagnose_mlx_chinese_hybrid_for_tests", MODULE_PATH
    )
    if specification is None or specification.loader is None:
        raise RuntimeError(f"cannot import {MODULE_PATH}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        specification.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module


class _FakeCaptureBackend:
    def __init__(self, module) -> None:
        self.module = module
        self.events = []

    def capabilities(self):
        self.events.append("capabilities")
        return json.loads(
            json.dumps(
                {
                    **self.module.REQUIRED_CAPTURE_CAPABILITIES,
                    "source_custody": self.module._trusted_backend_source_custody(),
                }
            )
        )

    def open_pair(self, inputs):
        self.events.append(("open_pair", inputs["candidate_manifest_sha256"]))
        return {
            "pair_id": "synthetic-same-process-pair",
            "process_id": 4242,
            "reference_handle_id": "bf16-handle",
            "candidate_handle_id": "hybrid-handle",
            "reference_manifest_sha256": inputs["reference_manifest_sha256"],
            "candidate_manifest_sha256": inputs["candidate_manifest_sha256"],
        }

    def capture_state_aligned(
        self,
        pair,
        *,
        prompt_token_ids,
        teacher_token_ids,
        repeats,
    ):
        self.events.append(("capture", pair["pair_id"], repeats))
        steps = []
        for index, teacher_token in enumerate(teacher_token_ids):
            flipped = index == 46
            steps.append(
                {
                    "step_index": index,
                    "reference_token_id": teacher_token,
                    "reference_top1_token_id": teacher_token,
                    "candidate_top1_token_id": 110926 if flipped else teacher_token,
                    "reference_top1_margin_micro": 200000,
                    "candidate_reference_token_margin_micro": (
                        -10000 if flipped else 180000
                    ),
                }
            )
        error_by_path = {
            "language_model.model.layers.8.linear_attn.in_proj_qkv": 900,
            "language_model.model.layers.9.mlp.down_proj": 7000,
            "language_model.model.layers.10.linear_attn.out_proj": 5000,
            "language_model.model.layers.11.self_attn.o_proj": 3000,
        }
        modules = [
            {
                "path": path,
                "tier": "w8",
                "sample_count": len(teacher_token_ids),
                "relative_l1_error_ppm": error_by_path.get(path, 1),
                "max_abs_error_micro": error_by_path.get(path, 1) * 2,
                "first_nonzero_step": 0,
            }
            for path in W8_PATHS
        ]
        run = {"step_metrics": steps, "module_metrics": modules}
        return {
            "format": "apxinf-mlx-chinese-state-aligned-capture-v1",
            "prompt_id": "chinese-explanation",
            "prompt_token_ids": list(prompt_token_ids),
            "teacher_token_ids": list(teacher_token_ids),
            "retained_bf16_paths": list(RETAINED_BF16_PATHS),
            "w8_module_paths": list(W8_PATHS),
            "w8_module_paths_sha256": self.module.object_sha256(W8_PATHS),
            "runs": [json.loads(json.dumps(run)) for _ in range(repeats)],
        }

    def close_pair(self, pair):
        self.events.append(("close_pair", pair["pair_id"]))


class ChineseHybridDiagnosticTests(unittest.TestCase):
    def test_certified_chinese_v1_scope_is_exactly_25_plus_64(self):
        module = _module()
        envelope = json.loads(REAL_HYBRID_EVIDENCE.read_text(encoding="utf-8"))

        inputs = module._quality_inputs(envelope)

        self.assertEqual(len(inputs["prompt_token_ids"]), 25)
        self.assertEqual(len(inputs["teacher_token_ids"]), 64)
        self.assertEqual(
            module.object_sha256(inputs["prompt_token_ids"]),
            module.CERTIFIED_PROMPT_TOKEN_IDS_SHA256,
        )
        self.assertEqual(
            module.object_sha256(inputs["teacher_token_ids"]),
            module.CERTIFIED_TEACHER_TOKEN_IDS_SHA256,
        )
        sequence = inputs["prompt_token_ids"] + inputs["teacher_token_ids"][:-1]
        self.assertEqual(len(sequence), 88)
        self.assertEqual(len(sequence) - 24, 64)
        self.assertEqual(
            module.object_sha256(sequence),
            module.CERTIFIED_INPUT_TOKEN_IDS_SHA256,
        )

    def test_real_quality_evidence_localizes_fixed_window_candidate_insertion(self):
        module = _module()
        envelope = json.loads(REAL_HYBRID_EVIDENCE.read_text(encoding="utf-8"))

        trajectory = module.analyze_chinese_trajectory(envelope)

        self.assertEqual(trajectory["prompt_id"], "chinese-explanation")
        self.assertFalse(trajectory["trajectory_exact"])
        self.assertEqual(trajectory["exact_prefix_tokens"], 46)
        self.assertEqual(
            trajectory["first_divergence"],
            {
                "step_index": 46,
                "step_number": 47,
                "reference_token_id": 100745,
                "candidate_token_id": 110926,
            },
        )
        self.assertEqual(
            trajectory["alignment"],
            {
                "classification": (
                    "candidate-single-token-insertion-with-fixed-window-tail-truncation"
                ),
                "inserted_candidate_token_id": 110926,
                "truncated_reference_tail_token_id": 97460,
                "aligned_suffix_tokens": 17,
            },
        )
        self.assertEqual(
            trajectory["semantic_stability"],
            {"assessed": False, "claim": None},
        )

    def test_trajectory_analysis_rejects_a_rehashed_noncertified_envelope(self):
        module = _module()
        envelope = json.loads(REAL_HYBRID_EVIDENCE.read_text(encoding="utf-8"))
        chinese = next(
            record
            for record in envelope["evidence"]["records"]
            if record["prompt_id"] == "chinese-explanation"
        )
        chinese["candidate"]["runs"][0][46] = 7
        chinese["candidate"]["runs"][1][46] = 7
        envelope.pop("content_sha256")
        envelope["content_sha256"] = module.object_sha256(envelope)

        with self.assertRaisesRegex(module.DiagnosticError, "certified hybrid"):
            module.analyze_chinese_trajectory(envelope)

    def test_read_only_same_process_capture_ranks_at_most_three_w8_restorations(self):
        module = _module()
        envelope = json.loads(REAL_HYBRID_EVIDENCE.read_text(encoding="utf-8"))
        backend = _FakeCaptureBackend(module)

        receipt = module.diagnose_with_backend(envelope, backend)

        self.assertEqual(receipt["format"], module.RECEIPT_FORMAT)
        self.assertEqual(receipt["status"], "diagnostic-only")
        self.assertEqual(
            [item["path"] for item in receipt["module_localization"]["top_candidates"]],
            [
                "language_model.model.layers.9.mlp.down_proj",
                "language_model.model.layers.10.linear_attn.out_proj",
                "language_model.model.layers.11.self_attn.o_proj",
            ],
        )
        self.assertTrue(
            all(
                item["proposed_tier"] == "bf16"
                for item in receipt["module_localization"]["top_candidates"]
            )
        )
        self.assertEqual(receipt["module_localization"]["top_k_limit"], 3)
        self.assertFalse(receipt["claims"]["trajectory_exact"])
        self.assertFalse(receipt["claims"]["teacher_forced_top1_exact"])
        self.assertFalse(receipt["claims"]["semantic_equivalence_assessed"])
        self.assertFalse(receipt["claims"]["general_parity"])
        self.assertEqual(
            receipt["execution"]["source_custody"],
            module._trusted_backend_source_custody(),
        )
        self.assertEqual(
            receipt["execution"]["rss_supervision"],
            "external-process-supervisor-required-v1",
        )
        self.assertFalse(receipt["execution"]["in_process_rss_watchdog"])
        stability = receipt["teacher_forced_stability"]
        self.assertEqual(stability["candidate_teacher_top1_match_tokens"], 63)
        self.assertEqual(stability["candidate_teacher_top1_match_ppm"], 984375)
        self.assertEqual(stability["first_candidate_teacher_flip_step_index"], 46)
        self.assertEqual(stability["minimum_candidate_teacher_margin_micro"], -10000)
        self.assertEqual(stability["maximum_margin_erosion_micro"], 210000)
        self.assertEqual(
            backend.events,
            [
                "capabilities",
                (
                    "open_pair",
                    "5f65527db17cd4fb7a1a46ca6ab3940a8bcc8225c620ca10d3016265d2bf8553",
                ),
                ("capture", "synthetic-same-process-pair", 2),
                ("close_pair", "synthetic-same-process-pair"),
            ],
        )
        body = dict(receipt)
        digest = body.pop("content_sha256")
        self.assertEqual(digest, module.object_sha256(body))

    def test_production_loader_returns_source_bound_backend_without_importing_mlx(self):
        module = _module()
        before = set(sys.modules)

        backend = module.load_production_backend()

        capabilities = backend.capabilities()
        custody = capabilities.pop("source_custody")
        self.assertEqual(capabilities, module.REQUIRED_CAPTURE_CAPABILITIES)
        self.assertEqual(
            custody["format"],
            "apxinf-direct-regular-single-link-source-custody-v1",
        )
        self.assertEqual(
            custody["capture"]["sha256"],
            module.EXPECTED_CAPTURE_BACKEND_SHA256,
        )
        self.assertEqual(
            custody["capture"]["path"],
            str((ROOT / "scripts" / "mlx_qwen35_state_aligned_capture.py").resolve()),
        )
        self.assertEqual(
            custody["loader"]["path"],
            str(MODULE_PATH.resolve()),
        )
        for identity in (custody["capture"], custody["loader"]):
            path = Path(identity["path"])
            observed = path.lstat()
            self.assertEqual(identity["size"], observed.st_size)
            self.assertEqual(observed.st_nlink, 1)
            self.assertEqual(len(identity["sha256"]), 64)
        newly_imported = set(sys.modules) - before
        self.assertFalse(
            any(name == "mlx" or name.startswith("mlx.") for name in newly_imported)
        )
        self.assertFalse(
            any(
                name == "mlx_lm" or name.startswith("mlx_lm.")
                for name in newly_imported
            )
        )

    def test_loader_rejects_capture_digest_drift_before_import(self):
        module = _module()
        original = module.EXPECTED_CAPTURE_BACKEND_SHA256
        module.EXPECTED_CAPTURE_BACKEND_SHA256 = "0" * 64
        try:
            with self.assertRaisesRegex(module.DiagnosticError, "frozen digest"):
                module.load_production_backend()
        finally:
            module.EXPECTED_CAPTURE_BACKEND_SHA256 = original

    def test_source_custody_rejects_a_second_hard_link(self):
        module = _module()
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary).resolve() / "source.py"
            alias = source.with_name("alias.py")
            source.write_text("# source\n", encoding="utf-8")
            alias.hardlink_to(source)

            with self.assertRaisesRegex(module.DiagnosticError, "one hard link"):
                module._source_file_identity(source, "test source")

    def test_partial_w8_module_capture_is_rejected_and_pair_is_closed(self):
        module = _module()
        envelope = json.loads(REAL_HYBRID_EVIDENCE.read_text(encoding="utf-8"))
        backend = _FakeCaptureBackend(module)
        original = backend.capture_state_aligned

        def partial_capture(*args, **kwargs):
            capture = original(*args, **kwargs)
            capture["w8_module_paths"].pop()
            capture["w8_module_paths_sha256"] = module.object_sha256(
                capture["w8_module_paths"]
            )
            for run in capture["runs"]:
                run["module_metrics"].pop()
            return capture

        backend.capture_state_aligned = partial_capture

        with self.assertRaisesRegex(module.DiagnosticError, "portfolio"):
            module.diagnose_with_backend(envelope, backend)

        self.assertEqual(
            backend.events[-1], ("close_pair", "synthetic-same-process-pair")
        )

    def test_capture_must_use_bf16_teacher_top1_and_never_dynamic_replacement(self):
        module = _module()
        envelope = json.loads(REAL_HYBRID_EVIDENCE.read_text(encoding="utf-8"))

        unsafe = _FakeCaptureBackend(module)
        unsafe_capabilities = unsafe.capabilities

        def replacement_capabilities():
            capabilities = unsafe_capabilities()
            capabilities["dynamic_module_replacement"] = True
            return capabilities

        unsafe.capabilities = replacement_capabilities
        with self.assertRaisesRegex(module.DiagnosticError, "read-only interface"):
            module.diagnose_with_backend(envelope, unsafe)
        self.assertNotIn(
            "open_pair",
            [event if type(event) is str else event[0] for event in unsafe.events],
        )

        wrong_teacher = _FakeCaptureBackend(module)
        original = wrong_teacher.capture_state_aligned

        def wrong_teacher_capture(*args, **kwargs):
            capture = original(*args, **kwargs)
            for run in capture["runs"]:
                run["step_metrics"][12]["reference_top1_token_id"] += 1
            return capture

        wrong_teacher.capture_state_aligned = wrong_teacher_capture
        with self.assertRaisesRegex(module.DiagnosticError, "step metric"):
            module.diagnose_with_backend(envelope, wrong_teacher)
        self.assertEqual(
            wrong_teacher.events[-1],
            ("close_pair", "synthetic-same-process-pair"),
        )

    def test_capture_source_custody_must_match_this_loader_before_open(self):
        module = _module()
        envelope = json.loads(REAL_HYBRID_EVIDENCE.read_text(encoding="utf-8"))
        backend = _FakeCaptureBackend(module)
        original = backend.capabilities

        def unbound_capabilities():
            capabilities = original()
            capabilities["source_custody"]["capture"]["size"] += 1
            return capabilities

        backend.capabilities = unbound_capabilities

        with self.assertRaisesRegex(module.DiagnosticError, "source custody"):
            module.diagnose_with_backend(envelope, backend)

        self.assertNotIn(
            "open_pair",
            [event if type(event) is str else event[0] for event in backend.events],
        )

    def test_teacher_forced_capture_rejects_a_candidate_flip_before_free_run_flip(self):
        module = _module()
        envelope = json.loads(REAL_HYBRID_EVIDENCE.read_text(encoding="utf-8"))
        backend = _FakeCaptureBackend(module)
        original = backend.capture_state_aligned

        def premature_flip(*args, **kwargs):
            capture = original(*args, **kwargs)
            for run in capture["runs"]:
                run["step_metrics"][12]["candidate_top1_token_id"] += 1
            return capture

        backend.capture_state_aligned = premature_flip

        with self.assertRaisesRegex(module.DiagnosticError, "free-run exact prefix"):
            module.diagnose_with_backend(envelope, backend)

        self.assertEqual(
            backend.events[-1],
            ("close_pair", "synthetic-same-process-pair"),
        )

    def test_cli_can_report_the_certified_trajectory_without_loading_mlx(self):
        module = _module()
        stdout = io.StringIO()

        with redirect_stdout(stdout):
            return_code = module.main(
                [
                    "--quality-evidence",
                    str(REAL_HYBRID_EVIDENCE.resolve()),
                    "--inspect-trajectory-only",
                ],
                backend_loader=lambda: self.fail("must not load an MLX backend"),
            )

        self.assertEqual(return_code, 0)
        summary = json.loads(stdout.getvalue())
        self.assertEqual(
            summary["format"],
            "apxinf-mlx-chinese-hybrid-trajectory-summary-v1",
        )
        self.assertEqual(summary["trajectory"]["exact_prefix_tokens"], 46)
        self.assertFalse(summary["claims"]["trajectory_exact"])
        self.assertFalse(summary["claims"]["semantic_equivalence_assessed"])

    def test_fake_capture_cli_publishes_one_versioned_no_replace_receipt(self):
        module = _module()
        backend = _FakeCaptureBackend(module)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary).resolve() / "chinese-diagnostic-v1.json"
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                return_code = module.main(
                    [
                        "--quality-evidence",
                        str(REAL_HYBRID_EVIDENCE.resolve()),
                        "--output",
                        str(output),
                    ],
                    backend_loader=lambda: backend,
                )
            published = json.loads(output.read_text(encoding="utf-8"))

            second_backend = _FakeCaptureBackend(module)
            second_stdout = io.StringIO()
            with redirect_stdout(second_stdout):
                second_return_code = module.main(
                    [
                        "--quality-evidence",
                        str(REAL_HYBRID_EVIDENCE.resolve()),
                        "--output",
                        str(output),
                    ],
                    backend_loader=lambda: second_backend,
                )

        self.assertEqual(return_code, 0)
        summary = json.loads(stdout.getvalue())
        self.assertEqual(summary["status"], "diagnostic-only")
        self.assertTrue(summary["published"])
        self.assertEqual(summary["content_sha256"], published["content_sha256"])
        self.assertEqual(published["format"], module.RECEIPT_FORMAT)
        self.assertEqual(second_return_code, 2)
        self.assertIn(
            "already exists", json.loads(second_stdout.getvalue())["problems"][0]
        )
        self.assertEqual(second_backend.events, [])


if __name__ == "__main__":
    unittest.main()
