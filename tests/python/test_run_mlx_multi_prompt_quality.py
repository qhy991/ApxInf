from __future__ import annotations

from contextlib import redirect_stdout
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "scripts" / "run_mlx_multi_prompt_quality.py"
CONTRACT_PATH = ROOT / "configs" / "qwen35-0.8b-mlx-multi-prompt-quality-v1.json"
HYBRID_W8_BF16_PROFILE = "hybrid-w8-bf16-g64"
COUNTERFACTUAL_PROFILE = (
    "hybrid-w8-bf16-g64-chinese-top1-counterfactual-v1"
)
HYBRID_W8_BF16_PRESET = {
    "name": "qwen35-0.8b-affine-w8-g64-gdn-outproj-async-chat-parity-v2",
    "policy_sha256": (
        "64a2ba1741fd5a76a7e72580ce9188d1554e1488ce6504b20054bf42479eaf8f"
    ),
    "retained_bf16_paths": [
        "language_model.model.layers.12.linear_attn.out_proj",
        "language_model.model.layers.14.linear_attn.out_proj",
        "language_model.model.layers.20.linear_attn.out_proj",
    ],
    "source_revision": "2fc06364715b967f1860aea9cf38778875588b17",
    "weight_ledger": {
        "estimated_total_parameter_bytes": 805788352,
        "output_tensor_count": 688,
        "quantized_logical_weight_count": 745603072,
        "quantized_module_count": 184,
        "quantized_module_parameter_bytes": 792203264,
        "retained_bf16_logical_weight_count": 6291456,
        "retained_bf16_module_count": 3,
        "retained_bf16_weight_bytes": 12582912,
    },
}
COUNTERFACTUAL_PRESET = {
    "name": (
        "qwen35-0.8b-affine-w8-g64-gdn3-l19-o-proj-chinese-counterfactual-v3"
    ),
    "policy_sha256": (
        "7030fe5a7c4dd55cbf158750e9da3a67c7f8e65944b8f8835c75b1093e12eec9"
    ),
    "retained_bf16_paths": [
        "language_model.model.layers.12.linear_attn.out_proj",
        "language_model.model.layers.14.linear_attn.out_proj",
        "language_model.model.layers.19.self_attn.o_proj",
        "language_model.model.layers.20.linear_attn.out_proj",
    ],
    "source_revision": "2fc06364715b967f1860aea9cf38778875588b17",
    "weight_ledger": {
        "estimated_total_parameter_bytes": 807754432,
        "output_tensor_count": 686,
        "quantized_logical_weight_count": 743505920,
        "quantized_module_count": 183,
        "quantized_module_parameter_bytes": 789975040,
        "retained_bf16_logical_weight_count": 8388608,
        "retained_bf16_module_count": 4,
        "retained_bf16_weight_bytes": 16777216,
    },
    "counterfactual": {
        "format": "apxinf-qwen35-mlx-hybrid-counterfactual-lineage-v1",
        "status": "unvalidated-candidate",
        "selection": {
            "causal_attribution": False,
            "current_tier": "w8",
            "path": "language_model.model.layers.19.self_attn.o_proj",
            "proposed_tier": "bf16",
            "rank": 1,
            "ranking_metric": "same-bf16-input-relative-l1-error-ppm-v1",
            "selection_basis": "trusted-diagnostic-trigger-only-not-causal-proof-v1",
        },
        "diagnostic": {
            "artifact_path": (
                "doc/20260823-qwen35-macos-bringup/"
                "qwen35-hybrid-w8-bf16-g64-chinese-state-aligned-"
                "diagnostic-v1.json"
            ),
            "artifact_sha256": (
                "1b30a3a7f6d609a8265112bde3189b7638a9072561530b852ec86dbc4794b73d"
            ),
            "content_sha256": (
                "e0c207bf46a62b643e3aeadc9398aea0d983426585d9b13ce25d21ce35d21a7f"
            ),
            "format": "apxinf-mlx-chinese-hybrid-diagnostic-receipt-v1",
        },
        "parent": {
            "bundle_manifest_sha256": (
                "5f65527db17cd4fb7a1a46ca6ab3940a8bcc8225c620ca10d3016265d2bf8553"
            ),
            "policy_sha256": (
                "64a2ba1741fd5a76a7e72580ce9188d1554e1488ce6504b20054bf42479eaf8f"
            ),
            "preset": (
                "qwen35-0.8b-affine-w8-g64-gdn-outproj-async-chat-parity-v2"
            ),
        },
        "reference": {
            "bundle_manifest_sha256": (
                "fdce8bac86b1bbc888cac0139065f0291a9a57ce7f448e591b748f4baaad5dea"
            ),
            "precision": "bf16",
        },
        "admission": {
            "formal_performance_claim": False,
            "general_parity": False,
            "parent_bundle_replacement": False,
            "promotion_requires_all_gates": True,
            "required_gates": [
                "apxinf-mlx-counterfactual-deployed-canonical-gate-v1",
                "qwen35-0.8b-mlx-multi-prompt-quality-v1-4-prompts-x2",
            ],
        },
    },
}


def _module():
    spec = importlib.util.spec_from_file_location(
        "run_mlx_multi_prompt_quality_for_tests", MODULE_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module


def _write_bundle(path: Path, *, precision: str) -> Path:
    path.mkdir()
    config = {
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "model_type": "qwen3_5",
        "text_config": {
            "dtype": "bfloat16",
            "model_type": "qwen3_5_text",
            "vocab_size": 248320,
        },
    }
    if precision in {HYBRID_W8_BF16_PROFILE, COUNTERFACTUAL_PROFILE}:
        config["quantization"] = {"bits": 8, "group_size": 64, "mode": "affine"}
        config["quantization_config"] = dict(config["quantization"])
        preset = (
            COUNTERFACTUAL_PRESET
            if precision == COUNTERFACTUAL_PROFILE
            else HYBRID_W8_BF16_PRESET
        )
        config["apxinf_hybrid_preset"] = json.loads(json.dumps(preset))
    elif precision == "mixed-w4-w8-bf16":
        w8_path = "language_model.model.layers.1.self_attn.q_proj"
        retained_path = "language_model.model.layers.2.linear_attn.out_proj"
        config["quantization"] = {
            "bits": 4,
            "group_size": 64,
            "mode": "affine",
            w8_path: {"bits": 8, "group_size": 64, "mode": "affine"},
        }
        config["quantization_config"] = dict(config["quantization"])
        config["apxinf_selective_mixed_policy"] = {
            "format": "apxinf-mlx-selective-mixed-policy-manifest-v1",
            "name": "synthetic-policy-v1",
            "source_repo_id": "Qwen/Qwen3.5-0.8B",
            "source_revision": "2fc06364715b967f1860aea9cf38778875588b17",
            "candidate_module_count": 3,
            "w8_paths": [w8_path],
            "retained_bf16_paths": [retained_path],
        }
    (path / "README.md").write_text("synthetic fixture\n", encoding="utf-8")
    (path / "chat_template.jinja").write_text(
        "{{ messages[-1].content }}\n", encoding="utf-8"
    )
    (path / "config.json").write_text(
        json.dumps(config, sort_keys=True), encoding="utf-8"
    )
    (path / "tokenizer.json").write_text(
        json.dumps({"fixture": "raw-token tests do not invoke tokenizer"}),
        encoding="utf-8",
    )
    (path / "tokenizer_config.json").write_text(
        json.dumps({"tokenizer_class": "Qwen2Tokenizer"}), encoding="utf-8"
    )
    (path / "model.safetensors").write_bytes(b"synthetic-safetensors-shard")
    (path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": 27},
                "weight_map": {"model.synthetic.weight": "model.safetensors"},
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return path.resolve()


def _identity(module):
    return {
        "python": {
            "implementation": "CPython",
            "version": module.PINNED_PYTHON_VERSION,
            "executable": "/synthetic/pinned/python3.14",
        },
        "packages": dict(module.PINNED_PACKAGES),
    }


class _Token:
    def __init__(self, value: int) -> None:
        self.value = value

    def item(self) -> int:
        return self.value


class _FakeRuntime:
    def __init__(
        self,
        *,
        mismatch_prompt: int | None = None,
        nondeterministic_lane: str | None = None,
        mutate_after_candidate_run=None,
    ) -> None:
        self.mismatch_prompt = mismatch_prompt
        self.nondeterministic_lane = nondeterministic_lane
        self.mutate_after_candidate_run = mutate_after_candidate_run
        self.load_calls = []
        self.generate_calls = []
        self.argmax_axes = []
        self.evaluated = 0
        self._lane = None
        self._lane_runs = {}

    def load(self, path, *, tokenizer_config, lazy):
        assert os.environ["HF_HUB_OFFLINE"] == "1"
        assert os.environ["TRANSFORMERS_OFFLINE"] == "1"
        assert os.environ["HF_DATASETS_OFFLINE"] == "1"
        assert os.environ["NO_PROXY"] == "*"
        assert tokenizer_config == {
            "local_files_only": True,
            "trust_remote_code": False,
        }
        assert lazy is True
        path = Path(path)
        self._lane = "reference" if "reference" in path.name else "candidate"
        self.load_calls.append(
            {
                "path": str(path),
                "tokenizer_config": dict(tokenizer_config),
                "lazy": lazy,
            }
        )
        return self._lane, object()

    def array(self, values):
        return tuple(values)

    def argmax(self, _logprobs, *, axis):
        self.argmax_axes.append(axis)
        return 0

    def eval(self, _value):
        self.evaluated += 1

    def generate_step(self, prompt, model, *, max_tokens, sampler):
        lane = model
        key = (lane, tuple(prompt))
        run_index = self._lane_runs.get(key, 0)
        self._lane_runs[key] = run_index + 1
        prompt_index = {
            24: 0,
            25: 1,
            28: 2,
            38: 3,
        }[len(prompt)]
        self.generate_calls.append(
            {
                "lane": lane,
                "prompt_index": prompt_index,
                "run_index": run_index,
                "max_tokens": max_tokens,
            }
        )
        sampler(object())
        values = [1000 + prompt_index * 100 + index for index in range(max_tokens)]
        if lane == "candidate" and self.mismatch_prompt == prompt_index:
            values[-1] += 1
        if lane == self.nondeterministic_lane and run_index == 1:
            values[-1] += 2
        for value in values:
            yield _Token(value), object()
        if lane == "candidate" and self.mutate_after_candidate_run is not None:
            callback = self.mutate_after_candidate_run
            self.mutate_after_candidate_run = None
            callback()

    def clear_cache(self):
        return None


class MlxMultiPromptEvidenceProducerTests(unittest.TestCase):
    def test_counterfactual_profile_matches_the_checked_in_policy_contract(self):
        module = _module()
        policy_path = (
            ROOT
            / "doc"
            / "20260823-qwen35-macos-bringup"
            / "qwen35-0.8b-mlx-w8-g64-chinese-top1-counterfactual-policy-v3.json"
        )
        policy = json.loads(policy_path.read_text(encoding="utf-8"))["policy"]
        observed_sha256 = hashlib.sha256(
            json.dumps(
                policy,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()

        self.assertEqual(
            observed_sha256,
            module._COUNTERFACTUAL_PRESET["policy_sha256"],
        )
        self.assertEqual(
            module._COUNTERFACTUAL_PRESET,
            {
                "name": policy["preset"],
                "policy_sha256": observed_sha256,
                "retained_bf16_paths": policy["retained_bf16_paths"],
                "source_revision": policy["source"]["revision"],
                "weight_ledger": policy["ledger"],
                "counterfactual": policy["counterfactual"],
            },
        )

    def _fixture(self, root: Path):
        reference = _write_bundle(root / "reference-bf16", precision="bf16")
        candidate = _write_bundle(
            root / "candidate-mixed", precision="mixed-w4-w8-bf16"
        )
        output = (root / "evidence" / "quality-run.json").resolve()
        output.parent.mkdir()
        return reference, candidate, output

    def _tokenizer_aware_hasher(self, module):
        real_hasher = module._stream_sha256
        contract = module.VALIDATOR.load_contract(CONTRACT_PATH)

        def digest(path):
            if Path(path).name == "tokenizer.json":
                return contract["model"]["tokenizer_sha256"]
            return real_hasher(path)

        return digest

    def _run(self, module, reference, candidate, output, runtime, **kwargs):
        return module.run_quality_gate(
            contract_path=CONTRACT_PATH,
            reference_bundle=reference,
            candidate_bundle=candidate,
            output_path=output,
            candidate_id="synthetic-mixed-candidate",
            precision_profile="mixed-w4-w8-bf16",
            requested_claim="fixed-suite-exact-parity",
            runtime_loader=lambda: runtime,
            identity_provider=lambda: _identity(module),
            file_hasher=self._tokenizer_aware_hasher(module),
            **kwargs,
        )

    def test_fake_runtime_produces_valid_accepted_evidence_with_full_custody(self):
        module = _module()
        runtime = _FakeRuntime()
        with tempfile.TemporaryDirectory() as temporary:
            reference, candidate, output = self._fixture(Path(temporary).resolve())

            envelope = self._run(module, reference, candidate, output, runtime)
            published = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(envelope, published)
        self.assertEqual(envelope["status"], "accepted")
        self.assertTrue(envelope["validation_receipt"]["accepted"])
        self.assertEqual(len(envelope["evidence"]["records"]), 4)
        self.assertEqual(len(runtime.generate_calls), 16)
        self.assertEqual(
            [(item["lane"], item["run_index"]) for item in runtime.generate_calls],
            [
                (lane, repeat)
                for lane in ("reference", "candidate")
                for _prompt in range(4)
                for repeat in range(2)
            ],
        )
        self.assertEqual(runtime.argmax_axes, [-1] * 16)
        self.assertEqual(runtime.evaluated, 2 * 2 * (64 + 64 + 32 + 32))
        self.assertEqual(
            [item["tokenizer_config"] for item in runtime.load_calls],
            [
                {"local_files_only": True, "trust_remote_code": False},
                {"local_files_only": True, "trust_remote_code": False},
            ],
        )
        for lane in ("reference", "candidate"):
            custody = envelope["custody"]["bundles"][lane]
            self.assertEqual(custody["before"], custody["after"])
            self.assertEqual(
                set(custody["before"]["files"]),
                {
                    "README.md",
                    "chat_template.jinja",
                    "config.json",
                    "model.safetensors",
                    "model.safetensors.index.json",
                    "tokenizer.json",
                    "tokenizer_config.json",
                },
            )
        self.assertEqual(
            envelope["custody"]["runtime"]["before"],
            envelope["custody"]["runtime"]["after"],
        )
        self.assertEqual(
            set(envelope["custody"]["runtime"]["before"]["packages"]),
            set(module.PINNED_PACKAGES),
        )
        self.assertEqual(len(module.PINNED_PACKAGES), 8)
        body = dict(envelope)
        digest = body.pop("content_sha256")
        self.assertEqual(digest, module.VALIDATOR.object_sha256(body))

    def test_hybrid_w8_bf16_profile_runs_the_fixed_suite_with_full_custody(self):
        module = _module()
        runtime = _FakeRuntime()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            reference = _write_bundle(root / "reference-bf16", precision="bf16")
            candidate = _write_bundle(
                root / "candidate-hybrid-w8-bf16",
                precision=HYBRID_W8_BF16_PROFILE,
            )
            output = (root / "quality-run.json").resolve()

            envelope = module.run_quality_gate(
                contract_path=CONTRACT_PATH,
                reference_bundle=reference,
                candidate_bundle=candidate,
                output_path=output,
                candidate_id="synthetic-hybrid-w8-bf16-candidate",
                precision_profile=HYBRID_W8_BF16_PROFILE,
                requested_claim="fixed-suite-threshold-match",
                runtime_loader=lambda: runtime,
                identity_provider=lambda: _identity(module),
                file_hasher=self._tokenizer_aware_hasher(module),
            )

        self.assertEqual(envelope["status"], "accepted")
        self.assertEqual(
            envelope["validation_receipt"]["precision_profile"],
            HYBRID_W8_BF16_PROFILE,
        )
        self.assertEqual(
            envelope["custody"]["bundles"]["candidate"]["before"]["precision_profile"],
            HYBRID_W8_BF16_PROFILE,
        )
        self.assertFalse(envelope["validation_receipt"]["claims_general_parity"])

    def test_counterfactual_profile_runs_the_unchanged_four_prompt_gate_twice(self):
        module = _module()
        runtime = _FakeRuntime()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            reference = _write_bundle(root / "reference-bf16", precision="bf16")
            candidate = _write_bundle(
                root / "candidate-counterfactual",
                precision=COUNTERFACTUAL_PROFILE,
            )
            output = (root / "quality-run.json").resolve()
            hasher = self._tokenizer_aware_hasher(module)
            contract = module.VALIDATOR.load_contract(CONTRACT_PATH)
            reference_manifest = module._snapshot_bundle(
                reference,
                "fixture BF16 reference",
                precision_profile="bf16",
                expected_tokenizer_sha256=contract["model"]["tokenizer_sha256"],
                file_hasher=hasher,
            )["manifest_sha256"]

            with mock.patch.object(
                module,
                "_COUNTERFACTUAL_REFERENCE_MANIFEST_SHA256",
                reference_manifest,
            ):
                envelope = module.run_quality_gate(
                    contract_path=CONTRACT_PATH,
                    reference_bundle=reference,
                    candidate_bundle=candidate,
                    output_path=output,
                    candidate_id="synthetic-chinese-top1-counterfactual-v1",
                    precision_profile=COUNTERFACTUAL_PROFILE,
                    requested_claim="fixed-suite-exact-parity",
                    runtime_loader=lambda: runtime,
                    identity_provider=lambda: _identity(module),
                    file_hasher=hasher,
                )

        self.assertEqual(envelope["status"], "accepted")
        self.assertEqual(
            envelope["validation_receipt"]["precision_profile"],
            COUNTERFACTUAL_PROFILE,
        )
        self.assertEqual(len(runtime.generate_calls), 16)
        self.assertFalse(envelope["validation_receipt"]["claims_general_parity"])
        self.assertEqual(
            envelope["custody"]["bundles"]["candidate"]["before"][
                "precision_profile"
            ],
            COUNTERFACTUAL_PROFILE,
        )

    def test_counterfactual_profile_rejects_any_extra_bf16_path_before_load(self):
        module = _module()
        runtime = _FakeRuntime()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            reference = _write_bundle(root / "reference-bf16", precision="bf16")
            candidate = _write_bundle(
                root / "candidate-counterfactual",
                precision=COUNTERFACTUAL_PROFILE,
            )
            config_path = candidate / "config.json"
            config = json.loads(config_path.read_text(encoding="utf-8"))
            config["apxinf_hybrid_preset"]["retained_bf16_paths"].append(
                "language_model.model.layers.21.self_attn.o_proj"
            )
            config_path.write_text(json.dumps(config), encoding="utf-8")
            output = (root / "quality-run.json").resolve()

            with self.assertRaisesRegex(
                module.ProducerError, "frozen preset manifest drifted"
            ):
                module.run_quality_gate(
                    contract_path=CONTRACT_PATH,
                    reference_bundle=reference,
                    candidate_bundle=candidate,
                    output_path=output,
                    candidate_id="synthetic-chinese-top1-counterfactual-v1",
                    precision_profile=COUNTERFACTUAL_PROFILE,
                    requested_claim="fixed-suite-exact-parity",
                    runtime_loader=lambda: runtime,
                    identity_provider=lambda: _identity(module),
                    file_hasher=self._tokenizer_aware_hasher(module),
                )

            self.assertEqual(runtime.load_calls, [])
            self.assertFalse(output.exists())

    def test_counterfactual_profile_rejects_an_uncertified_bf16_reference(self):
        module = _module()
        runtime = _FakeRuntime()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            reference = _write_bundle(root / "reference-bf16", precision="bf16")
            candidate = _write_bundle(
                root / "candidate-counterfactual",
                precision=COUNTERFACTUAL_PROFILE,
            )
            output = (root / "quality-run.json").resolve()

            with self.assertRaisesRegex(
                module.ProducerError, "certified BF16 reference manifest"
            ):
                module.run_quality_gate(
                    contract_path=CONTRACT_PATH,
                    reference_bundle=reference,
                    candidate_bundle=candidate,
                    output_path=output,
                    candidate_id="synthetic-chinese-top1-counterfactual-v1",
                    precision_profile=COUNTERFACTUAL_PROFILE,
                    requested_claim="fixed-suite-exact-parity",
                    runtime_loader=lambda: runtime,
                    identity_provider=lambda: _identity(module),
                    file_hasher=self._tokenizer_aware_hasher(module),
                )

            self.assertEqual(runtime.load_calls, [])
            self.assertFalse(output.exists())

    def test_hybrid_w8_bf16_profile_rejects_any_frozen_config_drift_before_load(self):
        module = _module()

        def add_global_override(config):
            config["quantization"]["language_model.model.layers.0.mlp.gate_proj"] = {
                "bits": 8,
                "group_size": 64,
                "mode": "affine",
            }

        def change_compat_bits(config):
            config["quantization_config"]["bits"] = 4

        def remove_hybrid(config):
            del config["apxinf_hybrid_preset"]

        def add_hybrid_field(config):
            config["apxinf_hybrid_preset"]["format"] = "unbound-format"

        def change_hybrid_name(config):
            config["apxinf_hybrid_preset"]["name"] = "different-preset"

        def change_hybrid_revision(config):
            config["apxinf_hybrid_preset"]["source_revision"] = "0" * 40

        def change_hybrid_policy_hash(config):
            config["apxinf_hybrid_preset"]["policy_sha256"] = "0" * 64

        def change_retained_paths(config):
            config["apxinf_hybrid_preset"]["retained_bf16_paths"].pop()

        def change_weight_ledger(config):
            config["apxinf_hybrid_preset"]["weight_ledger"][
                "retained_bf16_module_count"
            ] = 2

        def add_selective_manifest(config):
            config["apxinf_selective_mixed_policy"] = {"format": "unexpected"}

        cases = (
            ("global quantization override", add_global_override),
            ("compat quantization drift", change_compat_bits),
            ("missing hybrid preset", remove_hybrid),
            ("hybrid field-set drift", add_hybrid_field),
            ("hybrid name drift", change_hybrid_name),
            ("hybrid revision drift", change_hybrid_revision),
            ("hybrid policy hash drift", change_hybrid_policy_hash),
            ("retained BF16 path drift", change_retained_paths),
            ("weight ledger drift", change_weight_ledger),
            ("selective manifest coexistence", add_selective_manifest),
        )
        for label, mutate in cases:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary).resolve()
                reference = _write_bundle(root / "reference-bf16", precision="bf16")
                candidate = _write_bundle(
                    root / "candidate-hybrid-w8-bf16",
                    precision=HYBRID_W8_BF16_PROFILE,
                )
                output = (root / "quality-run.json").resolve()
                config_path = candidate / "config.json"
                config = json.loads(config_path.read_text(encoding="utf-8"))
                mutate(config)
                config_path.write_text(json.dumps(config), encoding="utf-8")
                runtime = _FakeRuntime()

                with self.assertRaises(module.ProducerError):
                    module.run_quality_gate(
                        contract_path=CONTRACT_PATH,
                        reference_bundle=reference,
                        candidate_bundle=candidate,
                        output_path=output,
                        candidate_id="synthetic-hybrid-w8-bf16-candidate",
                        precision_profile=HYBRID_W8_BF16_PROFILE,
                        requested_claim="fixed-suite-threshold-match",
                        runtime_loader=lambda: runtime,
                        identity_provider=lambda: _identity(module),
                        file_hasher=self._tokenizer_aware_hasher(module),
                    )

                self.assertFalse(output.exists())
                self.assertEqual(runtime.load_calls, [])

    def test_production_runtime_import_binds_generate_step_without_loading_model(self):
        module = _module()
        calls = []
        mlx_package = types.ModuleType("mlx")
        mlx_package.__path__ = []
        core = types.ModuleType("mlx.core")
        core.array = lambda values: tuple(values)

        def argmax(logprobs, *, axis):
            calls.append(("argmax", logprobs, axis))
            return 17

        core.argmax = argmax
        core.eval = lambda value: calls.append(("eval", value))
        core.clear_cache = lambda: calls.append(("clear_cache",))
        mlx_package.core = core

        mlx_lm = types.ModuleType("mlx_lm")
        mlx_lm.__path__ = []
        utils = types.ModuleType("mlx_lm.utils")
        utils.load = lambda *args, **kwargs: (args, kwargs)
        generate = types.ModuleType("mlx_lm.generate")
        exec(
            "def generate_step(prompt, model, *, max_tokens, sampler):\n"
            "    return (prompt, model, max_tokens, sampler)\n",
            generate.__dict__,
        )
        mlx_lm.utils = utils

        fake_modules = {
            "mlx": mlx_package,
            "mlx.core": core,
            "mlx_lm": mlx_lm,
            "mlx_lm.utils": utils,
            "mlx_lm.generate": generate,
        }
        with mock.patch.dict(sys.modules, fake_modules):
            runtime = module._load_runtime()

        sampler_result = runtime.argmax("logprobs", axis=-1)
        generated = runtime.generate_step(
            (1, 2), "model", max_tokens=3, sampler=runtime.argmax
        )
        runtime.eval(17)
        runtime.clear_cache()

        self.assertIs(runtime.generate_step, generate.generate_step)
        self.assertEqual(sampler_result, 17)
        self.assertEqual(generated[:3], ((1, 2), "model", 3))
        self.assertEqual(
            calls, [("argmax", "logprobs", -1), ("eval", 17), ("clear_cache",)]
        )

    def test_existing_output_is_rejected_before_hashing_or_loading(self):
        module = _module()
        runtime = _FakeRuntime()
        with tempfile.TemporaryDirectory() as temporary:
            reference, candidate, output = self._fixture(Path(temporary).resolve())
            output.write_text("do-not-replace\n", encoding="utf-8")

            with self.assertRaisesRegex(module.ProducerError, "already exists"):
                self._run(module, reference, candidate, output, runtime)

            self.assertEqual(output.read_text(encoding="utf-8"), "do-not-replace\n")
        self.assertEqual(runtime.load_calls, [])

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            reference, candidate, _output = self._fixture(root)
            inside_bundle = candidate / "quality-run.json"
            runtime = _FakeRuntime()
            with self.assertRaisesRegex(module.ProducerError, "outside both bundles"):
                self._run(module, reference, candidate, inside_bundle, runtime)

            self.assertFalse(inside_bundle.exists())
            self.assertEqual(runtime.load_calls, [])

    def test_non_absolute_or_polluted_bundle_is_rejected_before_runtime_load(self):
        module = _module()
        cases = ("unexpected_file", "nested_directory", "symlink")
        for case in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary).resolve()
                reference, candidate, output = self._fixture(root)
                if case == "unexpected_file":
                    (candidate / "custom_model.py").write_text(
                        "raise RuntimeError('must never execute')\n", encoding="utf-8"
                    )
                elif case == "nested_directory":
                    (candidate / "nested").mkdir()
                else:
                    (candidate / "alias.safetensors").symlink_to(
                        candidate / "model.safetensors"
                    )
                runtime = _FakeRuntime()

                with self.assertRaises(module.ProducerError):
                    self._run(module, reference, candidate, output, runtime)

                self.assertFalse(output.exists())
                self.assertEqual(runtime.load_calls, [])

        with tempfile.TemporaryDirectory() as temporary:
            reference, candidate, output = self._fixture(Path(temporary).resolve())
            with self.assertRaisesRegex(module.ProducerError, "absolute"):
                module.run_quality_gate(
                    contract_path=Path("relative-contract.json"),
                    reference_bundle=reference,
                    candidate_bundle=candidate,
                    output_path=output,
                    candidate_id="synthetic-mixed-candidate",
                    precision_profile="mixed-w4-w8-bf16",
                    requested_claim="fixed-suite-exact-parity",
                    runtime_loader=lambda: _FakeRuntime(),
                    identity_provider=lambda: _identity(module),
                    file_hasher=self._tokenizer_aware_hasher(module),
                )

    def test_remote_code_config_and_wrong_tokenizer_hash_fail_before_load(self):
        module = _module()
        with tempfile.TemporaryDirectory() as temporary:
            reference, candidate, output = self._fixture(Path(temporary).resolve())
            config_path = candidate / "config.json"
            config = json.loads(config_path.read_text(encoding="utf-8"))
            config["model_file"] = "custom_model.py"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            runtime = _FakeRuntime()

            with self.assertRaisesRegex(module.ProducerError, "remote/custom code"):
                self._run(module, reference, candidate, output, runtime)

            self.assertFalse(output.exists())
            self.assertEqual(runtime.load_calls, [])

        with tempfile.TemporaryDirectory() as temporary:
            reference, candidate, output = self._fixture(Path(temporary).resolve())
            runtime = _FakeRuntime()
            with self.assertRaisesRegex(module.ProducerError, "tokenizer.json"):
                module.run_quality_gate(
                    contract_path=CONTRACT_PATH,
                    reference_bundle=reference,
                    candidate_bundle=candidate,
                    output_path=output,
                    candidate_id="synthetic-mixed-candidate",
                    precision_profile="mixed-w4-w8-bf16",
                    requested_claim="fixed-suite-exact-parity",
                    runtime_loader=lambda: runtime,
                    identity_provider=lambda: _identity(module),
                )

            self.assertFalse(output.exists())
            self.assertEqual(runtime.load_calls, [])

    def test_mixed_precision_label_must_match_frozen_w4_w8_bf16_config(self):
        module = _module()
        with tempfile.TemporaryDirectory() as temporary:
            reference, candidate, output = self._fixture(Path(temporary).resolve())
            config_path = candidate / "config.json"
            config = json.loads(config_path.read_text(encoding="utf-8"))
            w8_path = config["apxinf_selective_mixed_policy"]["w8_paths"][0]
            config["quantization"][w8_path]["bits"] = 4
            config_path.write_text(json.dumps(config), encoding="utf-8")
            runtime = _FakeRuntime()

            with self.assertRaisesRegex(module.ProducerError, "W8 override"):
                self._run(module, reference, candidate, output, runtime)

            self.assertFalse(output.exists())
            self.assertEqual(runtime.load_calls, [])

    def test_real_file_hasher_streams_files_larger_than_one_chunk(self):
        module = _module()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary).resolve() / "large.synthetic"
            payload = b"a" * module._HASH_CHUNK_BYTES + b"bounded-tail"
            path.write_bytes(payload)

            observed = module._stream_sha256(path)

        self.assertEqual(observed, hashlib.sha256(payload).hexdigest())

    def test_bundle_change_during_generation_prevents_publication(self):
        module = _module()
        with tempfile.TemporaryDirectory() as temporary:
            reference, candidate, output = self._fixture(Path(temporary).resolve())
            runtime = _FakeRuntime(
                mutate_after_candidate_run=lambda: (
                    candidate / "model.safetensors"
                ).write_bytes(b"mutated-after-start")
            )

            with self.assertRaisesRegex(module.ProducerError, "changed during"):
                self._run(module, reference, candidate, output, runtime)

            self.assertFalse(output.exists())

    def test_nondeterministic_lane_is_invalid_and_never_published(self):
        module = _module()
        for lane in ("reference", "candidate"):
            with self.subTest(lane=lane), tempfile.TemporaryDirectory() as temporary:
                reference, candidate, output = self._fixture(Path(temporary).resolve())
                runtime = _FakeRuntime(nondeterministic_lane=lane)

                with self.assertRaisesRegex(
                    module.ProducerError, "two runs are not identical"
                ):
                    self._run(module, reference, candidate, output, runtime)

                self.assertFalse(output.exists())

    def test_deterministic_quality_mismatch_publishes_explicit_failed_comparison(self):
        module = _module()
        runtime = _FakeRuntime(mismatch_prompt=1)
        with tempfile.TemporaryDirectory() as temporary:
            reference, candidate, output = self._fixture(Path(temporary).resolve())

            envelope = self._run(module, reference, candidate, output, runtime)

            self.assertTrue(output.is_file())
            self.assertEqual(envelope["status"], "failed_comparison")
            receipt = envelope["validation_receipt"]
            self.assertFalse(receipt["accepted"])
            self.assertIsNone(receipt["claim"])
            self.assertIn("chinese-explanation", receipt["problems"][0])
            self.assertFalse(receipt["claims_general_parity"])

    def test_toolchain_drift_before_or_after_generation_never_publishes(self):
        module = _module()
        for phase in ("before", "after"):
            with self.subTest(phase=phase), tempfile.TemporaryDirectory() as temporary:
                reference, candidate, output = self._fixture(Path(temporary).resolve())
                calls = 0

                def identity_provider():
                    nonlocal calls
                    calls += 1
                    identity = _identity(module)
                    if phase == "before" or calls == 2:
                        identity["packages"]["mlx-lm"] = "0.31.4"
                    return identity

                runtime = _FakeRuntime()
                with self.assertRaisesRegex(module.ProducerError, "mlx-lm"):
                    module.run_quality_gate(
                        contract_path=CONTRACT_PATH,
                        reference_bundle=reference,
                        candidate_bundle=candidate,
                        output_path=output,
                        candidate_id="synthetic-mixed-candidate",
                        precision_profile="mixed-w4-w8-bf16",
                        requested_claim="fixed-suite-exact-parity",
                        runtime_loader=lambda: runtime,
                        identity_provider=identity_provider,
                        file_hasher=self._tokenizer_aware_hasher(module),
                    )

                self.assertFalse(output.exists())
                if phase == "before":
                    self.assertEqual(runtime.load_calls, [])

    def test_main_emits_one_summary_and_maps_comparison_failure_to_exit_one(self):
        module = _module()
        runtime = _FakeRuntime(mismatch_prompt=3)
        with tempfile.TemporaryDirectory() as temporary:
            reference, candidate, output = self._fixture(Path(temporary).resolve())
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                return_code = module.main(
                    [
                        "--contract",
                        str(CONTRACT_PATH),
                        "--reference-bundle",
                        str(reference),
                        "--candidate-bundle",
                        str(candidate),
                        "--candidate-id",
                        "synthetic-mixed-candidate",
                        "--precision-profile",
                        "mixed-w4-w8-bf16",
                        "--requested-claim",
                        "fixed-suite-exact-parity",
                        "--output",
                        str(output),
                    ],
                    runtime_loader=lambda: runtime,
                    identity_provider=lambda: _identity(module),
                    file_hasher=self._tokenizer_aware_hasher(module),
                )

        self.assertEqual(return_code, 1)
        summary = json.loads(stdout.getvalue())
        self.assertEqual(summary["status"], "failed_comparison")
        self.assertEqual(summary["output"], str(output))
        self.assertFalse(summary["accepted"])


if __name__ == "__main__":
    unittest.main()
