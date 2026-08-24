from __future__ import annotations

import hashlib
import importlib.util
import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
BACKEND_PATH = ROOT / "scripts/mlx_mixed_quant_backend.py"


def load_module(name: str, path: Path) -> object:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


BACKEND = load_module("_apxinf_mixed_quant_backend_under_test", BACKEND_PATH)


def canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


class FakeBuilder:
    PINNED_PACKAGES = {f"package-{index}": "1.0" for index in range(8)}

    def __init__(self, source_manifest_sha256: str) -> None:
        self.source_manifest_sha256 = source_manifest_sha256
        self.source_checks = 0

    def _assert_source_unchanged(self, _source: object) -> None:
        self.source_checks += 1

    def _load_selective_policy(
        self,
        source: object,
        policy_path: str,
        source_revision: str,
        mode: str,
    ) -> dict[str, object]:
        assert source.directory.is_dir()
        assert Path(policy_path).is_file()
        assert mode == "affine-w4-g64"
        return {
            "policy_sha256": "a" * 64,
            "policy_document_sha256": "b" * 64,
            "search_receipt_sha256": "c" * 64,
            "search_status": "initial-all-w4",
            "source_repo_id": "Qwen/Qwen3.5-0.8B",
            "source_revision": source_revision,
            "source_lock_content_sha256": "d" * 64,
            "source_manifest_sha256": self.source_manifest_sha256,
            "candidate_modules": [
                {
                    "path": "language_model.model.layers.0.mlp.down_proj",
                    "dtype": "BF16",
                    "shape": [8, 64],
                }
            ],
            "candidate_modules_sha256": "e" * 64,
            "tiers": {"language_model.model.layers.0.mlp.down_proj": "w4"},
            "w4_paths": ["language_model.model.layers.0.mlp.down_proj"],
            "w8_paths": [],
            "retained_bf16_paths": [],
            "trace": {},
            "trace_sha256": "f" * 64,
            "weight_ledger": {"w4_module_count": 1},
            "quality_tier": "evaluation-only",
            "config_manifest": {"w8_paths": [], "retained_bf16_paths": []},
        }

    @staticmethod
    def _selective_weight_ledger(
        _schema: object, tiers: dict[str, str]
    ) -> dict[str, int]:
        return {
            "w4_module_count": sum(tier == "w4" for tier in tiers.values()),
            "w8_module_count": sum(tier == "w8" for tier in tiers.values()),
            "retained_bf16_module_count": sum(
                tier == "bf16" for tier in tiers.values()
            ),
        }

    @staticmethod
    def _selective_config_manifest(selective: dict[str, object]) -> dict[str, object]:
        return {
            "w8_paths": selective["w8_paths"],
            "retained_bf16_paths": selective["retained_bf16_paths"],
        }


class FakeAdapter:
    def __init__(self, packages: dict[str, str]) -> None:
        self.packages = packages
        self.events: list[tuple[object, ...]] = []
        self.fail_reload = False
        self.record_sha256 = "8" * 64

    def runtime_identity(self) -> dict[str, object]:
        return {
            "python": {
                "implementation": "CPython",
                "version": "3.14.3",
                "executable_sha256": "9" * 64,
            },
            "packages": [
                {"name": name, "version": version, "record_sha256": "8" * 64}
                | {"record_sha256": self.record_sha256}
                for name, version in sorted(self.packages.items())
            ],
            "offline": True,
            "network_blocked": False,
            "trust_remote_code": False,
        }

    def save_candidate(
        self,
        source: object,
        bundle_dir: Path,
        cache_dir: Path,
        *,
        mode: str,
        selective: dict[str, object] | None,
    ) -> None:
        self.events.append(("save", bundle_dir, mode, selective is not None))
        assert source.directory.is_dir()
        assert cache_dir.is_dir()
        bundle_dir.mkdir(mode=0o700)
        (bundle_dir / "config.json").write_bytes(
            canonical_bytes(
                {
                    "mode": mode,
                    "tiers": selective["tiers"] if selective is not None else None,
                }
            )
        )
        (bundle_dir / "model.safetensors").write_bytes(b"tiny-model")

    def inspect_saved(
        self,
        source: object,
        bundle_dir: Path,
        *,
        mode: str,
        selective: dict[str, object] | None,
    ) -> dict[str, object]:
        self.events.append(("inspect", bundle_dir, mode, selective is not None))
        digest = hashlib.sha256()
        for path in sorted(bundle_dir.iterdir()):
            digest.update(path.name.encode())
            digest.update(path.read_bytes())
        return {"manifest_sha256": digest.hexdigest()}

    def reload_candidate(
        self,
        bundle_dir: Path,
        cache_dir: Path,
        *,
        mode: str,
        selective: dict[str, object] | None,
    ) -> object:
        self.events.append(("reload", bundle_dir, mode, selective is not None))
        if self.fail_reload:
            raise RuntimeError("injected reload failure")
        return {"bundle_dir": bundle_dir, "closed": False}

    def close_loaded(self, loaded: dict[str, object]) -> None:
        self.events.append(("close", loaded["bundle_dir"]))
        loaded["closed"] = True

    def evaluate_gate(
        self,
        loaded: dict[str, object],
        cache_dir: Path,
        *,
        trace: dict[str, object],
        role: str,
    ) -> dict[str, object]:
        self.events.append(("gate", role, loaded["bundle_dir"], cache_dir))
        return {
            "api": trace["api"],
            "semantics": trace["semantics"],
            "prompt_token_ids": trace["prompt_token_ids"],
            "teacher_forced_token_ids": [[1, 2], [1, 2]],
            "async_free_run_token_ids": [[1, 2], [1, 2]],
        }

    def screen_state_aligned(
        self,
        reference: dict[str, object],
        candidate: dict[str, object],
        cache_dir: Path,
        *,
        trace: dict[str, object],
        candidate_modules: list[dict[str, object]],
        transition: dict[str, object] | None,
    ) -> dict[str, object]:
        self.events.append(
            (
                "screen",
                reference["bundle_dir"],
                candidate["bundle_dir"],
                cache_dir,
                transition,
            )
        )
        scores = [
            {
                "path": item["path"],
                "hidden_error_ppm": 3,
                "top1_margin_erosion_ppm": 2,
                "top1_flip_rate_ppm": 1,
                "score_ppm": 6,
            }
            for item in candidate_modules
        ]
        return {
            "format": "apxinf-mlx-mixed-quant-state-aligned-screen-v1",
            "steps": 32,
            "state_alignment": "prompt-plus-bf16-teacher-prefix-v1",
            "aggregate_score_ppm": 6 * len(scores),
            "module_scores": scores,
        }


class MixedQuantBackendTests(unittest.TestCase):
    def fixture(self, root: Path) -> tuple[object, FakeAdapter, Path]:
        source_dir = root / "source"
        source_dir.mkdir()
        (source_dir / "model.safetensors").write_bytes(b"source")
        source_manifest = hashlib.sha256(b"source-manifest").hexdigest()
        policy = {
            "format": "fake-policy-document",
            "policy": {
                "source": {
                    "repo_id": "Qwen/Qwen3.5-0.8B",
                    "revision": "1" * 40,
                },
                "candidate_modules": [
                    {
                        "path": "language_model.model.layers.0.mlp.down_proj",
                        "dtype": "BF16",
                        "shape": [8, 64],
                    }
                ],
                "quantization": {"default": {"tier": "w4"}, "overrides": []},
                "trace": {
                    "api": "mlx_lm.generate.generate_step",
                    "semantics": "mlx-generate-step-argmax-v1",
                    "prompt_token_ids": [7, 8],
                    "teacher_token_ids": [1, 2],
                },
            },
        }
        policy_path = root / "policy.json"
        policy_payload = canonical_bytes(policy) + b"\n"
        policy_path.write_bytes(policy_payload)
        generation = SimpleNamespace(
            policy=policy["policy"],
            policy_sha256="a" * 64,
            policy_artifact_sha256=hashlib.sha256(policy_payload).hexdigest(),
            inputs={
                "source_manifest_sha256": source_manifest,
                "policy_document_sha256": hashlib.sha256(
                    canonical_bytes(policy)
                ).hexdigest(),
            },
        )
        builder = FakeBuilder(source_manifest)
        source = SimpleNamespace(directory=source_dir, records={}, tensor_schema={})
        certification = SimpleNamespace(
            generation=generation,
            builder_api=builder,
            source_bundle=source,
            policy_path=policy_path,
        )
        scratch = root / "scratch"
        scratch.mkdir()
        scratch = scratch.resolve(strict=True)
        adapter = FakeAdapter(builder.PINNED_PACKAGES)
        backend = BACKEND.MlxMixedQuantBackend(certification, scratch, adapter)
        return backend, adapter, scratch

    def test_current_candidate_is_saved_verified_and_reloaded_before_return(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            backend, adapter, scratch = self.fixture(Path(temporary))

            handle = backend.materialize_current(backend.certified_generation)

            self.assertEqual(
                [event[0] for event in adapter.events],
                ["save", "inspect", "reload"],
            )
            paths = [event[1] for event in adapter.events]
            self.assertEqual(paths, [paths[0], paths[0], paths[0]])
            self.assertTrue(paths[0].is_dir())
            self.assertEqual(
                set(handle),
                {
                    "handle_id",
                    "manifest_sha256",
                    "policy_sha256",
                    "evaluation_only",
                    "publishable",
                    "materialization",
                },
            )
            self.assertTrue(handle["evaluation_only"])
            self.assertFalse(handle["publishable"])

            backend.close(handle)

            self.assertFalse(paths[0].exists())
            self.assertEqual(adapter.events[-1][0], "close")
            self.assertEqual(list(scratch.iterdir()), [])

    def test_bf16_reference_and_current_candidate_are_independent_reloads(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            backend, adapter, scratch = self.fixture(Path(temporary))

            reference = backend.open_bf16_reference(backend.certified_generation)
            current = backend.materialize_current(backend.certified_generation)

            saves = [event for event in adapter.events if event[0] == "save"]
            self.assertEqual(
                [(event[2], event[3]) for event in saves],
                [("mixed-bf16", False), ("affine-w4-g64", True)],
            )
            self.assertNotEqual(saves[0][1], saves[1][1])
            self.assertNotEqual(reference["handle_id"], current["handle_id"])
            backend.close(current)
            self.assertTrue(saves[0][1].is_dir())
            backend.close(reference)
            self.assertEqual(list(scratch.iterdir()), [])

    def test_counterfactual_changes_one_saved_descriptor_before_reload(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            backend, adapter, _scratch = self.fixture(Path(temporary))
            transition = {
                "path": "language_model.model.layers.0.mlp.down_proj",
                "from": "w4",
                "to": "w8",
            }

            handle = backend.materialize_counterfactual(
                backend.certified_generation, transition
            )

            save = next(event for event in adapter.events if event[0] == "save")
            self.assertEqual(save[2:], ("affine-w4-g64", True))
            self.assertEqual(
                json.loads((save[1] / "config.json").read_bytes())["tiers"],
                {"language_model.model.layers.0.mlp.down_proj": "w8"},
            )
            self.assertEqual(handle["transition"], transition)
            backend.close(handle)

    def test_gate_and_state_aligned_screen_use_only_registered_reloads(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            backend, adapter, _scratch = self.fixture(Path(temporary))
            reference = backend.open_bf16_reference(backend.certified_generation)
            current = backend.materialize_current(backend.certified_generation)

            gate = backend.evaluate_gate(
                current,
                certified=backend.certified_generation,
                role="current-candidate",
            )
            screen = backend.screen_state_aligned(
                reference,
                current,
                certified=backend.certified_generation,
                transition=None,
            )

            self.assertEqual(gate["teacher_forced_token_ids"], [[1, 2], [1, 2]])
            self.assertEqual(screen["aggregate_score_ppm"], 6)
            self.assertEqual(
                [
                    event[0]
                    for event in adapter.events
                    if event[0] in {"gate", "screen"}
                ],
                ["gate", "screen"],
            )
            forged = dict(current)
            forged["manifest_sha256"] = "0" * 64
            with self.assertRaisesRegex(BACKEND.BackendError, "unknown or changed"):
                backend.evaluate_gate(
                    forged,
                    certified=backend.certified_generation,
                    role="current-candidate",
                )
            backend.close(current)
            backend.close(reference)

    def test_reload_failure_leaves_no_candidate_or_session_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            backend, adapter, scratch = self.fixture(Path(temporary))
            adapter.fail_reload = True

            with self.assertRaisesRegex(RuntimeError, "injected reload failure"):
                backend.materialize_current(backend.certified_generation)

            self.assertEqual(list(scratch.iterdir()), [])

    def test_policy_identity_drift_fails_before_model_work_and_cleans_session(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            backend, adapter, scratch = self.fixture(root)
            (root / "policy.json").write_text("{}\n", encoding="utf-8")

            with self.assertRaisesRegex(
                BACKEND.BackendError, "policy artifact changed"
            ):
                backend.materialize_current(backend.certified_generation)

            self.assertEqual(adapter.events, [])
            self.assertEqual(list(scratch.iterdir()), [])

    def test_eight_package_record_drift_fails_before_model_work(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            backend, adapter, scratch = self.fixture(Path(temporary))
            adapter.record_sha256 = "6" * 64

            with self.assertRaisesRegex(
                BACKEND.BackendError, "runtime identity changed"
            ):
                backend.materialize_current(backend.certified_generation)

            self.assertEqual(adapter.events, [])
            self.assertEqual(list(scratch.iterdir()), [])

    def test_pinned_adapter_uses_local_save_reload_and_has_no_implicit_hook(
        self,
    ) -> None:
        class TinyBuilder:
            SELECTIVE_CONFIG_KEY = "apxinf_selective_mixed_policy"
            SUPPORTED_MODEL_TYPE = "qwen3_5"

            def __init__(self) -> None:
                self.events: list[tuple[object, ...]] = []

            @contextmanager
            def _offline_runtime(self, cache_dir: Path):
                self.events.append(("offline-enter", cache_dir))
                try:
                    yield
                finally:
                    self.events.append(("offline-exit", cache_dir))

            def _validate_output_config(
                self,
                config: dict[str, object],
                mode: str,
                hybrid: object,
                selective: dict[str, object] | None,
            ) -> None:
                self.events.append(("validate-config", mode, hybrid, selective))
                assert config[self.SELECTIVE_CONFIG_KEY] == selective["config_manifest"]

            def _assert_source_unchanged(self, source: object) -> None:
                self.events.append(("source-check", source.directory))

            def _inspect_output(
                self,
                output: Path,
                source: object,
                mode: str,
                hybrid: object,
                selective: dict[str, object] | None,
            ) -> tuple[dict[str, object], dict[str, object]]:
                self.events.append(
                    (
                        "static-inspect",
                        output,
                        source.directory,
                        mode,
                        hybrid,
                        selective,
                    )
                )
                return {}, {"manifest_sha256": "7" * 64}

        class TinyMlxApi:
            def __init__(self, source_dir: Path) -> None:
                self.source_dir = source_dir
                self.events: list[tuple[object, ...]] = []
                self.saved_config: dict[str, object] | None = None
                self.async_ids = list(range(128))

            def load(
                self, path: str, **kwargs: object
            ) -> tuple[object, object, object]:
                self.events.append(("load", Path(path), kwargs))
                config = (
                    {"model_type": "qwen3_5"}
                    if Path(path) == self.source_dir
                    else json.loads(json.dumps(self.saved_config))
                )
                return object(), object(), config

            def quantize_model(
                self,
                model: object,
                config: dict[str, object],
                **kwargs: object,
            ) -> tuple[object, dict[str, object]]:
                predicate = kwargs["quant_predicate"]
                decisions = {
                    path: predicate(path, object())
                    for path in ("module.w4", "module.w8", "module.bf16")
                }
                self.events.append(("quantize", kwargs, decisions))
                return model, dict(config)

            def save(
                self,
                output: Path,
                source: str,
                model: object,
                tokenizer: object,
                config: dict[str, object],
                *,
                donate_model: bool,
            ) -> None:
                self.events.append(("save", output, source, donate_model))
                output.mkdir(mode=0o700)
                (output / "model.safetensors").write_bytes(b"tiny")
                (output / "config.json").write_bytes(canonical_bytes(config))
                self.saved_config = json.loads(json.dumps(config))

            @staticmethod
            def array(value: object) -> object:
                return value

            @staticmethod
            def argmax(value: object, *, axis: int) -> object:
                assert axis == -1
                return value

            @staticmethod
            def teacher_forced_step(
                prompt: object, model: object, teacher_ids: list[int]
            ) -> list[int]:
                return list(teacher_ids)

            def generate_step(self, prompt: object, model: object, **kwargs: object):
                self.events.append(("generate", prompt, kwargs))
                for token in self.async_ids[: int(kwargs["max_tokens"])]:
                    yield token, None

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve(strict=True)
            source_dir = root / "source"
            source_dir.mkdir()
            cache_dir = root / "cache"
            cache_dir.mkdir()
            bundle_dir = root / "candidate"
            builder = TinyBuilder()
            api = TinyMlxApi(source_dir)
            adapter = BACKEND.PinnedMlxAdapter(builder, mlx_api=api)
            source = SimpleNamespace(
                directory=source_dir,
                tokenizer_payloads={},
            )
            selective = {
                "tiers": {
                    "module.w4": "w4",
                    "module.w8": "w8",
                    "module.bf16": "bf16",
                },
                "config_manifest": {"policy_sha256": "a" * 64},
            }

            adapter.save_candidate(
                source,
                bundle_dir,
                cache_dir,
                mode="affine-w4-g64",
                selective=selective,
            )
            evidence = adapter.inspect_saved(
                source,
                bundle_dir,
                mode="affine-w4-g64",
                selective=selective,
            )
            loaded = adapter.reload_candidate(
                bundle_dir,
                cache_dir,
                mode="affine-w4-g64",
                selective=selective,
            )

            self.assertEqual(evidence["manifest_sha256"], "7" * 64)
            self.assertEqual(
                [event[0] for event in builder.events],
                [
                    "offline-enter",
                    "offline-exit",
                    "source-check",
                    "static-inspect",
                    "source-check",
                    "offline-enter",
                    "offline-exit",
                    "validate-config",
                ],
            )
            trace = {
                "api": "mlx_lm.generate.generate_step",
                "semantics": "mlx-generate-step-argmax-v1",
                "prompt_token_ids": [7, 8],
                "teacher_token_ids": list(range(128)),
                "teacher_steps": 128,
                "free_run_steps": 128,
                "repeat_count": 2,
            }
            gate = adapter.evaluate_gate(
                loaded,
                cache_dir,
                trace=trace,
                role="current-candidate",
            )
            self.assertEqual(
                gate["teacher_forced_token_ids"],
                [list(range(128)), list(range(128))],
            )
            self.assertEqual(
                gate["async_free_run_token_ids"],
                [list(range(128)), list(range(128))],
            )
            loads = [event for event in api.events if event[0] == "load"]
            self.assertEqual([event[1] for event in loads], [source_dir, bundle_dir])
            for event in loads:
                self.assertEqual(
                    event[2]["tokenizer_config"],
                    {"local_files_only": True, "trust_remote_code": False},
                )
            quantize = next(event for event in api.events if event[0] == "quantize")
            self.assertEqual(
                quantize[2],
                {
                    "module.w4": True,
                    "module.w8": {"bits": 8, "group_size": 64, "mode": "affine"},
                    "module.bf16": False,
                },
            )
            with self.assertRaisesRegex(
                BACKEND.BackendError, "audited state-aligned capture hook"
            ):
                adapter.screen_state_aligned(
                    loaded,
                    loaded,
                    cache_dir,
                    trace={},
                    candidate_modules=[],
                    transition=None,
                )
            adapter.close_loaded(loaded)
            with self.assertRaisesRegex(BACKEND.BackendError, "closed MLX bundle"):
                adapter.evaluate_gate(
                    loaded,
                    cache_dir,
                    trace=trace,
                    role="current-candidate",
                )

    def test_pinned_adapter_runtime_identity_binds_cpython_and_eight_records(
        self,
    ) -> None:
        class Distribution:
            version = "1.0"

            def __init__(self, name: str) -> None:
                self.name = name

            def read_text(self, filename: str) -> str | None:
                assert filename == "RECORD"
                return f"{self.name},sha256=tiny,1\n"

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve(strict=True)
            executable = root / "python"
            executable.write_bytes(b"tiny-cpython")
            builder = FakeBuilder("0" * 64)
            adapter = BACKEND.PinnedMlxAdapter(builder, mlx_api=object())
            with (
                mock.patch.object(BACKEND.sys, "executable", str(executable)),
                mock.patch.object(
                    BACKEND.metadata,
                    "distribution",
                    side_effect=lambda name: Distribution(name),
                ),
                mock.patch.object(
                    BACKEND.platform, "python_implementation", return_value="CPython"
                ),
                mock.patch.object(
                    BACKEND.platform, "python_version", return_value="3.14.3"
                ),
            ):
                identity = adapter.runtime_identity()

            self.assertEqual(len(identity["packages"]), 8)
            self.assertEqual(
                [item["name"] for item in identity["packages"]],
                sorted(builder.PINNED_PACKAGES),
            )
            self.assertEqual(
                identity["python"]["executable_sha256"],
                hashlib.sha256(b"tiny-cpython").hexdigest(),
            )
            self.assertTrue(identity["offline"])
            self.assertFalse(identity["network_blocked"])
            self.assertFalse(identity["trust_remote_code"])


if __name__ == "__main__":
    unittest.main()
