from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "scripts" / "mlx_qwen35_state_aligned_capture.py"


def _module():
    specification = importlib.util.spec_from_file_location(
        "mlx_qwen35_state_aligned_capture_for_tests", MODULE_PATH
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


class _FakeEngine:
    def __init__(self, module) -> None:
        self.module = module
        self.events = []
        self._cache_index = 0

    def make_cache(self, model):
        self._cache_index += 1
        cache = {"cache_id": self._cache_index, "role": model["role"]}
        self.events.append(("make_cache", model["role"], self._cache_index))
        return cache

    def validate_loaded_pair(self, reference, candidate):
        self.events.append(
            ("validate_loaded_pair", reference["role"], candidate["role"])
        )

    def official_forward(self, model, sequence, cache):
        self.events.append(
            ("official_forward", model["role"], tuple(sequence), cache["cache_id"])
        )
        return {
            "role": model["role"],
            "sequence": list(sequence),
            "cache_id": cache["cache_id"],
        }

    def prefill_forward(self, model, sequence, cache):
        self.events.append(
            ("prefill_forward", model["role"], tuple(sequence), cache["cache_id"])
        )

    def manual_reference_forward(
        self,
        reference,
        candidate,
        sequence,
        cache,
        *,
        predictor_position,
        predictor_step_index,
        module_paths,
    ):
        self.events.append(
            (
                "manual_reference_forward",
                reference["role"],
                candidate["role"],
                tuple(sequence),
                predictor_position,
                predictor_step_index,
                len(module_paths),
                cache["cache_id"],
            )
        )
        observations = {}
        for index, path in enumerate(module_paths):
            observation = {
                "sample_count": 1,
                "predictor_step_index": predictor_step_index,
                "numerator": float(index + 1),
                "denominator": float((index + 1) * 10),
                "maximum": float(index + 1) / 1000.0,
                "first_nonzero_step": predictor_step_index,
            }
            calls = 2 if path == "language_model.model.embed_tokens" else 1
            observations[path] = [dict(observation) for _ in range(calls)]
        return {
            "role": reference["role"],
            "sequence": list(sequence),
            "cache_id": cache["cache_id"],
        }, observations

    def assert_exact_logits(self, official, manual, *, predictor_position):
        self.events.append(
            (
                "assert_exact_logits",
                official["role"],
                manual["role"],
                predictor_position,
            )
        )
        if official["sequence"] != manual["sequence"]:
            raise AssertionError("manual sequence drifted")

    def step_metric(
        self,
        reference_logits,
        candidate_logits,
        teacher_token_id,
        *,
        predictor_position,
        predictor_step_index,
    ):
        self.events.append(
            (
                "step_metrics",
                reference_logits["role"],
                candidate_logits["role"],
                predictor_position,
                predictor_step_index,
            )
        )
        return {
            "step_index": predictor_step_index,
            "reference_token_id": teacher_token_id,
            "reference_top1_token_id": teacher_token_id,
            "candidate_top1_token_id": (
                110926 if predictor_step_index == 46 else teacher_token_id
            ),
            "reference_top1_margin_micro": 20,
            "candidate_reference_token_margin_micro": (
                -10 if predictor_step_index == 46 else 18
            ),
        }

    def combine_module_observations(self, path, observations, *, predictor_count):
        return self.module.MlxQwen35ManualEngine.combine_module_observations(
            path,
            observations,
            predictor_count=predictor_count,
        )


class Qwen35StateAlignedCaptureTests(unittest.TestCase):
    @staticmethod
    def _runtime_identity(module):
        return {
            "python": {
                "implementation": "CPython",
                "version": module.PINNED_PYTHON_VERSION,
                "executable": "/pinned/python3.14",
            },
            "packages": dict(module.PINNED_PACKAGES),
        }

    def test_teacher_forced_sequence_uses_prompt_and_all_but_last_bf16_token(self):
        module = _module()

        sequence, response_start = module.build_teacher_forced_sequence(
            list(module.CERTIFIED_PROMPT_TOKEN_IDS),
            list(module.CERTIFIED_TEACHER_TOKEN_IDS),
        )

        self.assertEqual(len(sequence), 88)
        self.assertEqual(response_start, 24)
        self.assertEqual(len(sequence) - response_start, 64)
        self.assertEqual(
            module.object_sha256(sequence),
            "3dba5d2c579177a68559980161fb86dc94be4fc53b17b92b56be160a4bb25de2",
        )

    def test_chinese_v1_rejects_noncertified_token_scope_before_engine(self):
        module = _module()
        engine = _FakeEngine(module)

        with self.assertRaisesRegex(module.CaptureError, "certified Chinese v1"):
            module.capture_loaded_models(
                {"role": "bf16"},
                {"role": "hybrid"},
                prompt_token_ids=[5, 6, 7],
                teacher_token_ids=[1, 2, 3, 4],
                engine=engine,
            )

        self.assertEqual(engine.events, [])

    def test_imported_runtime_modules_must_resolve_to_prehashed_distribution_paths(
        self,
    ):
        module = _module()
        names = tuple(module.PINNED_SOURCE_SHA256)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            expected = {}
            imported = {}
            for index, name in enumerate(names):
                path = root / f"source-{index}.py"
                path.write_text(f"# {name}\n", encoding="utf-8")
                expected[name] = path
                imported[name] = SimpleNamespace(__file__=str(path))

            module._validate_imported_source_paths(imported, expected)

            impostor = root / "impostor.py"
            impostor.write_text("# wrong module\n", encoding="utf-8")
            imported["qwen3_5.py"] = SimpleNamespace(__file__=str(impostor))
            with self.assertRaisesRegex(module.CaptureError, "import path"):
                module._validate_imported_source_paths(imported, expected)

    def test_tiny_engine_runs_two_read_only_state_aligned_captures_with_fresh_caches(
        self,
    ):
        module = _module()
        engine = _FakeEngine(module)
        reference = {"role": "bf16"}
        candidate = {"role": "hybrid"}

        capture = module.capture_loaded_models(
            reference,
            candidate,
            prompt_token_ids=list(module.CERTIFIED_PROMPT_TOKEN_IDS),
            teacher_token_ids=list(module.CERTIFIED_TEACHER_TOKEN_IDS),
            engine=engine,
        )

        self.assertEqual(capture["format"], module.CAPTURE_FORMAT)
        self.assertEqual(capture["prompt_id"], "chinese-explanation")
        self.assertEqual(
            capture["prompt_token_ids"], list(module.CERTIFIED_PROMPT_TOKEN_IDS)
        )
        self.assertEqual(
            capture["teacher_token_ids"], list(module.CERTIFIED_TEACHER_TOKEN_IDS)
        )
        self.assertEqual(len(capture["w8_module_paths"]), 184)
        self.assertEqual(
            capture["w8_module_paths_sha256"],
            module.object_sha256(capture["w8_module_paths"]),
        )
        self.assertEqual(capture["runs"][0], capture["runs"][1])
        self.assertEqual(len(capture["runs"][0]["module_metrics"]), 184)
        self.assertEqual(len(capture["runs"][0]["step_metrics"]), 64)
        self.assertTrue(
            all(
                metric["sample_count"] == 64
                for metric in capture["runs"][0]["module_metrics"]
            )
        )
        cache_events = [event for event in engine.events if event[0] == "make_cache"]
        self.assertEqual(len(cache_events), 6)
        self.assertEqual(len({event[2] for event in cache_events}), 6)
        official_sequences = [
            event[2] for event in engine.events if event[0] == "official_forward"
        ]
        expected_chunks = [(module.CERTIFIED_PROMPT_TOKEN_IDS[-1],)] + [
            (token,) for token in module.CERTIFIED_TEACHER_TOKEN_IDS[:-1]
        ]
        self.assertEqual(
            official_sequences,
            [chunk for chunk in expected_chunks for _lane in range(2)] * 2,
        )

    def test_capture_replays_generate_step_chunk_schedule_on_three_persistent_caches(
        self,
    ):
        module = _module()
        engine = _FakeEngine(module)
        prompt = list(module.CERTIFIED_PROMPT_TOKEN_IDS)
        teacher = list(module.CERTIFIED_TEACHER_TOKEN_IDS)

        capture = module.capture_loaded_models(
            {"role": "bf16"},
            {"role": "hybrid"},
            prompt_token_ids=prompt,
            teacher_token_ids=teacher,
            engine=engine,
        )

        expected_chunks = [(prompt[-1],)] + [(token,) for token in teacher[:-1]]
        prefill = [event for event in engine.events if event[0] == "prefill_forward"]
        self.assertEqual(len(prefill), 2 * 3)
        for cache_id in (1, 2, 3, 4, 5, 6):
            self.assertEqual(
                [event[2] for event in prefill if event[3] == cache_id],
                [tuple(prompt[:-1])],
            )
        official = [event for event in engine.events if event[0] == "official_forward"]
        self.assertEqual(len(official), 2 * 2 * 64)
        for cache_id in (1, 3, 4, 6):
            self.assertEqual(
                [event[2] for event in official if event[3] == cache_id],
                expected_chunks,
            )
        manual = [
            event for event in engine.events if event[0] == "manual_reference_forward"
        ]
        self.assertEqual(len(manual), 2 * 64)
        for cache_id in (2, 5):
            self.assertEqual(
                [event[3] for event in manual if event[7] == cache_id],
                expected_chunks,
            )
            self.assertEqual(
                [event[5] for event in manual if event[7] == cache_id],
                list(range(64)),
            )
            step_46 = next(
                event for event in manual if event[7] == cache_id and event[5] == 46
            )
            self.assertEqual(step_46[3], (teacher[45],))
        exact_gates = [
            event for event in engine.events if event[0] == "assert_exact_logits"
        ]
        self.assertEqual(len(exact_gates), 2 * 64)
        self.assertTrue(all(event[3] == 0 for event in exact_gates))
        self.assertEqual(
            capture["runs"][0]["step_metrics"][46],
            {
                "step_index": 46,
                "reference_token_id": teacher[46],
                "reference_top1_token_id": teacher[46],
                "candidate_top1_token_id": 110926,
                "reference_top1_margin_micro": 20,
                "candidate_reference_token_margin_micro": -10,
            },
        )

    def test_frozen_model_schema_is_tied_and_has_the_exact_184_w8_paths(self):
        module = _module()
        layers = [
            SimpleNamespace(is_linear=(index + 1) % 4 != 0) for index in range(24)
        ]
        args = SimpleNamespace(
            model_type="qwen3_5_text",
            hidden_size=1024,
            intermediate_size=3584,
            num_hidden_layers=24,
            num_attention_heads=8,
            num_key_value_heads=2,
            head_dim=256,
            linear_num_value_heads=16,
            linear_num_key_heads=16,
            linear_key_head_dim=128,
            linear_value_head_dim=128,
            linear_conv_kernel_dim=4,
            full_attention_interval=4,
            vocab_size=248320,
            num_experts=0,
            tie_word_embeddings=True,
            attention_bias=False,
        )
        model = SimpleNamespace(
            model_type="qwen3_5",
            language_model=SimpleNamespace(
                args=args,
                model=SimpleNamespace(layers=layers, ssm_idx=0, fa_idx=3),
            ),
        )

        _text, body = module.MlxQwen35ManualEngine._text_model(model)
        paths = module.expected_w8_module_paths()

        self.assertIs(body.layers, layers)
        self.assertEqual(len(paths), 184)
        self.assertEqual(len(paths), len(set(paths)))
        self.assertEqual(
            module.object_sha256(paths),
            "a913a999195653916fc77a6f6b7a1f2c3ccdffd709f92fae9111784b9c2bb349",
        )
        self.assertNotIn("language_model.model.layers.12.linear_attn.out_proj", paths)
        self.assertNotIn("language_model.lm_head", paths)

    def test_collector_accumulates_both_tied_embedding_calls_and_exactly_one_other_call(
        self,
    ):
        module = _module()
        paths = module.expected_w8_module_paths()

        class TinyMetricEngine:
            @staticmethod
            def module_observation(
                path,
                reference_output,
                candidate_output,
                *,
                predictor_position,
                predictor_step_index,
            ):
                return {
                    "sample_count": 1,
                    "predictor_step_index": predictor_step_index,
                    "numerator": candidate_output - reference_output,
                    "denominator": reference_output,
                    "maximum": candidate_output - reference_output,
                    "first_nonzero_step": predictor_step_index,
                }

        collector = module._ModuleMetricCollector(
            TinyMetricEngine(),
            paths,
            predictor_position=0,
            predictor_step_index=7,
        )

        def reference(value):
            return value + 10

        def candidate(value):
            return value + 11

        for path in paths:
            collector.call(path, reference, candidate, 1)
        collector.call(paths[0], reference, candidate, 2)

        observations = collector.finish()

        self.assertEqual(len(observations), 184)
        self.assertEqual(len(observations[paths[0]]), 2)
        self.assertEqual(len(observations[paths[1]]), 1)
        self.assertEqual(
            {item["predictor_step_index"] for item in observations[paths[0]]},
            {7},
        )
        with self.assertRaisesRegex(
            module.CaptureError, "unexpected module invocation"
        ):
            collector.call(paths[0], reference, candidate, 3)

    def test_module_metrics_aggregate_raw_weighted_errors_across_exactly_64_steps(self):
        module = _module()
        path = "language_model.model.layers.0.mlp.down_proj"
        observations = []
        for step_index in range(64):
            denominator = 1000.0 if step_index == 63 else 1.0
            ratio_ppm = 1.49 if step_index == 63 else 0.49
            numerator = denominator * ratio_ppm / 1_000_000
            observations.append(
                {
                    "sample_count": 1,
                    "predictor_step_index": step_index,
                    "numerator": numerator,
                    "denominator": denominator,
                    "maximum": numerator,
                    "first_nonzero_step": step_index,
                }
            )

        metric = module.MlxQwen35ManualEngine.combine_module_observations(
            path,
            observations,
            predictor_count=64,
        )

        self.assertEqual(metric["sample_count"], 64)
        self.assertEqual(metric["relative_l1_error_ppm"], 1)
        self.assertEqual(metric["max_abs_error_micro"], 1490)
        self.assertEqual(metric["first_nonzero_step"], 0)

    def test_prefill_materializes_only_cache_state_then_clears_graph_cache(self):
        module = _module()
        events = []

        class TinyArray:
            def __init__(self, values):
                self.values = tuple(values)

            def __getitem__(self, key):
                self.key = key
                return self

        class TinyMx:
            @staticmethod
            def array(values):
                return TinyArray(values)

            @staticmethod
            def eval(states):
                events.append(("eval", tuple(states)))

            @staticmethod
            def clear_cache():
                events.append(("clear_cache",))

        class TinyModel:
            def __call__(self, tokens, *, cache):
                events.append(("model", tokens.values, len(cache)))
                return object()

        engine = object.__new__(module.MlxQwen35ManualEngine)
        engine.mx = TinyMx
        engine._text_model = lambda _model: None
        cache = [SimpleNamespace(state=f"state-{index}") for index in range(24)]

        engine.prefill_forward(TinyModel(), list(range(24)), cache)

        self.assertEqual(events[0], ("model", tuple(range(24)), 24))
        self.assertEqual(events[1], ("eval", tuple(item.state for item in cache)))
        self.assertEqual(events[2], ("clear_cache",))

    def test_production_logprob_normalization_preserves_original_dtype_order(self):
        module = _module()
        events = []

        class Symbol:
            def __sub__(self, other):
                events.append(("subtract", self, other))
                return ("production-logprobs", self, other)

        class TinyMx:
            @staticmethod
            def logsumexp(value, *, keepdims):
                events.append(("logsumexp", value, keepdims))
                return ("normalizer", value)

        raw_logits = Symbol()
        engine = object.__new__(module.MlxQwen35ManualEngine)
        engine.mx = TinyMx

        scores = engine._production_logprobs(raw_logits)

        self.assertEqual(events[0], ("logsumexp", raw_logits, True))
        self.assertEqual(events[1][0], "subtract")
        self.assertIs(events[1][1], raw_logits)
        self.assertEqual(scores[0], "production-logprobs")

    def test_step_metric_resolves_a_low_precision_near_tie_after_production_normalization(
        self,
    ):
        import numpy as np

        module = _module()
        argmax_inputs = []

        class TinyMx:
            @staticmethod
            def array(values):
                return np.array(values)

            @staticmethod
            def logsumexp(value, *, keepdims):
                maximum = np.max(value.astype(np.float64), axis=-1, keepdims=True)
                total = np.sum(
                    np.exp(value.astype(np.float64) - maximum),
                    axis=-1,
                    keepdims=True,
                )
                result = maximum + np.log(total)
                if not keepdims:
                    result = np.squeeze(result, axis=-1)
                return result.astype(value.dtype)

            @staticmethod
            def argmax(value, *, axis):
                argmax_inputs.append(value.copy())
                return np.argmax(value, axis=axis)

            @staticmethod
            def topk(value, *, k, axis):
                return np.partition(value, -k, axis=axis)[..., -k:]

            max = staticmethod(np.max)
            min = staticmethod(np.min)
            take_along_axis = staticmethod(np.take_along_axis)
            where = staticmethod(np.where)
            isfinite = staticmethod(np.isfinite)
            all = staticmethod(np.all)

            @staticmethod
            def eval(*_values):
                return None

        raw = np.full((1, 1, 248320), -20.0, dtype=np.float16)
        raw[0, 0, 0] = np.float16(-0.4)
        raw[0, 0, 1] = np.nextafter(
            np.float16(-0.4), np.float16(np.inf), dtype=np.float16
        )
        self.assertEqual(int(np.argmax(raw, axis=-1)[0, 0]), 1)

        engine = object.__new__(module.MlxQwen35ManualEngine)
        engine.mx = TinyMx
        metric = engine.step_metric(
            raw,
            raw.copy(),
            1,
            predictor_position=0,
            predictor_step_index=0,
        )

        self.assertEqual(metric["reference_top1_token_id"], 0)
        self.assertEqual(metric["candidate_top1_token_id"], 0)
        self.assertEqual(metric["candidate_reference_token_margin_micro"], 0)
        self.assertEqual(len(argmax_inputs), 2)
        self.assertEqual(argmax_inputs[0].dtype, np.float16)
        self.assertEqual(argmax_inputs[0][0, 0], argmax_inputs[0][0, 1])

    def test_tiny_fake_loaded_pair_proves_exact_module_classes_without_replacement(
        self,
    ):
        module = _module()

        class Embedding:
            pass

        class Linear:
            pass

        class QuantizedEmbedding:
            def __init__(self):
                self.bits = 8
                self.group_size = 64
                self.mode = "affine"

        class QuantizedLinear(QuantizedEmbedding):
            pass

        class TinyModel:
            def __init__(self, named):
                self.model_type = "qwen3_5"
                self.training = False
                self.language_model = SimpleNamespace(
                    args=SimpleNamespace(
                        **{
                            **module._FROZEN_TEXT_SCHEMA,
                        }
                    ),
                    model=SimpleNamespace(
                        layers=[
                            SimpleNamespace(is_linear=(index + 1) % 4 != 0)
                            for index in range(24)
                        ],
                        ssm_idx=0,
                        fa_idx=3,
                    ),
                )
                self._named = named

            def named_modules(self):
                return list(self._named.items())

            def update_modules(self, *_args, **_kwargs):
                raise AssertionError("read-only capture must not replace modules")

        paths = module.expected_w8_module_paths()
        reference_modules = {
            path: Embedding() if index == 0 else Linear()
            for index, path in enumerate(paths)
        }
        candidate_modules = {
            path: QuantizedEmbedding() if index == 0 else QuantizedLinear()
            for index, path in enumerate(paths)
        }
        for path in module.RETAINED_BF16_PATHS:
            reference_modules[path] = Linear()
            candidate_modules[path] = Linear()
        engine = object.__new__(module.MlxQwen35ManualEngine)
        engine.nn = SimpleNamespace(
            Embedding=Embedding,
            Linear=Linear,
            QuantizedEmbedding=QuantizedEmbedding,
            QuantizedLinear=QuantizedLinear,
        )

        engine.validate_loaded_pair(
            TinyModel(reference_modules), TinyModel(candidate_modules)
        )

        candidate_modules[paths[-1]].group_size = 32
        with self.assertRaisesRegex(module.CaptureError, "W8 config"):
            engine.validate_loaded_pair(
                TinyModel(reference_modules), TinyModel(candidate_modules)
            )

    def test_backend_matches_frozen_protocol_and_rechecks_runtime_source_and_bundle_custody(
        self,
    ):
        module = _module()
        reference_manifest = "a" * 64
        candidate_manifest = "b" * 64
        snapshot_calls = []
        identity_calls = []
        source_calls = []

        def snapshotter(path, label, *, precision_profile):
            snapshot_calls.append((str(path), label, precision_profile))
            manifest = (
                reference_manifest if label == "reference" else candidate_manifest
            )
            return {
                "path": str(path),
                "precision_profile": precision_profile,
                "files": {"config.json": {"size": 1, "sha256": "c" * 64}},
                "file_count": 1,
                "total_bytes": 1,
                "manifest_sha256": manifest,
            }

        def identity_provider():
            identity_calls.append("identity")
            return self._runtime_identity(module)

        def source_auditor():
            source_calls.append("source")
            return dict(module.PINNED_SOURCE_SHA256)

        class FakeRuntime:
            def __init__(self):
                self.loads = []
                self.cleared = 0

            def load(self, path, *, tokenizer_config, lazy):
                self.loads.append((path, tokenizer_config, lazy))
                return {"role": "bf16" if "reference" in path else "hybrid"}, object()

            def clear_cache(self):
                self.cleared += 1

        runtime = FakeRuntime()
        engine = _FakeEngine(module)
        backend = module.Qwen35StateAlignedCaptureBackend(
            runtime_loader=lambda: runtime,
            runtime_identity_provider=identity_provider,
            source_auditor=source_auditor,
            bundle_snapshotter=snapshotter,
            engine_factory=lambda _runtime: engine,
        )
        inputs = {
            "reference_bundle_path": "/controlled/reference",
            "candidate_bundle_path": "/controlled/candidate",
            "reference_manifest_sha256": reference_manifest,
            "candidate_manifest_sha256": candidate_manifest,
        }

        capabilities = backend.capabilities()
        custody = capabilities.pop("source_custody")
        self.assertEqual(capabilities, module.REQUIRED_CAPTURE_CAPABILITIES)
        self.assertEqual(custody, module._backend_source_custody())
        self.assertEqual(
            custody["format"],
            "apxinf-direct-regular-single-link-source-custody-v1",
        )
        pair = backend.open_pair(inputs)
        capture = backend.capture_state_aligned(
            pair,
            prompt_token_ids=list(module.CERTIFIED_PROMPT_TOKEN_IDS),
            teacher_token_ids=list(module.CERTIFIED_TEACHER_TOKEN_IDS),
            repeats=2,
        )
        backend.close_pair(pair)

        self.assertEqual(capture["format"], module.CAPTURE_FORMAT)
        self.assertEqual(pair["process_id"], module.os.getpid())
        self.assertNotEqual(pair["reference_handle_id"], pair["candidate_handle_id"])
        self.assertEqual(
            runtime.loads,
            [
                (
                    "/controlled/reference",
                    {"local_files_only": True, "trust_remote_code": False},
                    True,
                ),
                (
                    "/controlled/candidate",
                    {"local_files_only": True, "trust_remote_code": False},
                    True,
                ),
            ],
        )
        self.assertEqual(
            snapshot_calls,
            [
                ("/controlled/reference", "reference", "bf16"),
                (
                    "/controlled/candidate",
                    "candidate",
                    "hybrid-w8-bf16-g64",
                ),
                ("/controlled/reference", "reference", "bf16"),
                (
                    "/controlled/candidate",
                    "candidate",
                    "hybrid-w8-bf16-g64",
                ),
            ],
        )
        self.assertEqual(identity_calls, ["identity", "identity"])
        self.assertEqual(source_calls, ["source", "source"])
        self.assertEqual(runtime.cleared, 1)

    def test_backend_fails_before_load_when_pinned_source_identity_drifts(self):
        module = _module()
        runtime_loads = []

        class FakeRuntime:
            def load(self, *args, **kwargs):
                runtime_loads.append((args, kwargs))

        snapshots = {
            "reference": "a" * 64,
            "candidate": "b" * 64,
        }
        backend = module.Qwen35StateAlignedCaptureBackend(
            runtime_loader=lambda: FakeRuntime(),
            runtime_identity_provider=lambda: self._runtime_identity(module),
            source_auditor=lambda: {
                **module.PINNED_SOURCE_SHA256,
                "qwen3_5.py": "0" * 64,
            },
            bundle_snapshotter=lambda path, label, precision_profile: {
                "path": str(path),
                "manifest_sha256": snapshots[label],
            },
            engine_factory=lambda runtime: _FakeEngine(module),
        )

        with self.assertRaisesRegex(module.CaptureError, "source identity"):
            backend.open_pair(
                {
                    "reference_bundle_path": "/controlled/reference",
                    "candidate_bundle_path": "/controlled/candidate",
                    "reference_manifest_sha256": snapshots["reference"],
                    "candidate_manifest_sha256": snapshots["candidate"],
                }
            )

        self.assertEqual(runtime_loads, [])

    def test_backend_close_fails_closed_on_bundle_drift_and_releases_pair(self):
        module = _module()
        manifests = {"reference": "a" * 64, "candidate": "b" * 64}
        calls = {"reference": 0, "candidate": 0}

        def snapshotter(path, label, *, precision_profile):
            calls[label] += 1
            return {
                "path": str(path),
                "precision_profile": precision_profile,
                "manifest_sha256": manifests[label],
                "files": {"generation": calls[label]},
            }

        class FakeRuntime:
            @staticmethod
            def load(path, *, tokenizer_config, lazy):
                del tokenizer_config, lazy
                return {"role": "bf16" if "reference" in path else "hybrid"}, object()

            @staticmethod
            def clear_cache():
                return None

        backend = module.Qwen35StateAlignedCaptureBackend(
            runtime_loader=FakeRuntime,
            runtime_identity_provider=lambda: self._runtime_identity(module),
            source_auditor=lambda: dict(module.PINNED_SOURCE_SHA256),
            bundle_snapshotter=snapshotter,
            engine_factory=lambda _runtime: _FakeEngine(module),
        )
        pair = backend.open_pair(
            {
                "reference_bundle_path": "/controlled/reference",
                "candidate_bundle_path": "/controlled/candidate",
                "reference_manifest_sha256": manifests["reference"],
                "candidate_manifest_sha256": manifests["candidate"],
            }
        )

        with self.assertRaisesRegex(module.CaptureError, "bundle changed"):
            backend.close_pair(pair)
        with self.assertRaisesRegex(module.CaptureError, "not open"):
            backend.close_pair(pair)


if __name__ == "__main__":
    unittest.main()
