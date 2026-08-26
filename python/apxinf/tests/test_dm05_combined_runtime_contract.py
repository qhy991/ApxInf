"""CPU-only contracts for the default-off exact combined DM05 runtime."""

from __future__ import annotations

import importlib.util
import inspect
import subprocess
import sys
import threading
import types
import unittest
from pathlib import Path
from unittest import mock


class _FakeBaseRuntime:
    def proof_snapshot(self):
        return {
            "initialized": True,
            "prefix_graph_replay_count": 3,
            "suffix_graph_replay_count": 3,
            "request_prefix_length": 564,
            "fallback_count": 0,
        }

    def close(self):
        self._closed = True


def _load_combined_module():
    fake_torch = types.ModuleType("torch")
    fake_opendm = types.ModuleType("opendm")
    fake_opendm.__path__ = []
    fake_model_package = types.ModuleType("opendm.model")
    fake_model_package.__path__ = []
    fake_dm05_package = types.ModuleType("opendm.model.dm05")
    fake_dm05_package.__path__ = []
    fake_arch = types.ModuleType("opendm.model.dm05.dm05_arch")
    fake_opendm.model = fake_model_package
    fake_model_package.dm05 = fake_dm05_package
    fake_dm05_package.dm05_arch = fake_arch
    fake_base = types.ModuleType(
        "apxinf.policies.impls.dm05_static_mask_prefix_graph"
    )
    fake_base.DM05StaticMaskPrefixGraphRuntime = _FakeBaseRuntime
    replacements = {
        "torch": fake_torch,
        "opendm": fake_opendm,
        "opendm.model": fake_model_package,
        "opendm.model.dm05": fake_dm05_package,
        "opendm.model.dm05.dm05_arch": fake_arch,
        "apxinf.policies.impls.dm05_static_mask_prefix_graph": fake_base,
    }
    previous = {name: sys.modules.get(name) for name in replacements}
    sys.modules.update(replacements)
    module_name = "_dm05_combined_runtime_contract_test"
    source = (
        Path(__file__).parents[1]
        / "apxinf"
        / "policies"
        / "impls"
        / "dm05_combined_runtime.py"
    )
    spec = importlib.util.spec_from_file_location(module_name, source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(module_name, None)
        for name, old_value in previous.items():
            if old_value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old_value
    return module, fake_arch, fake_torch


MODULE, ARCH, FAKE_TORCH = _load_combined_module()


class _Storage:
    def __init__(self, pointer):
        self.pointer = pointer

    def data_ptr(self):
        return self.pointer


class _FakeTensor:
    def __init__(
        self,
        pointer,
        *,
        shape=(1,),
        dtype="bf16",
        device="cuda:0",
        contiguous=True,
    ):
        self.pointer = pointer
        self.shape = tuple(shape)
        self.dtype = dtype
        self.device = device
        self.contiguous = contiguous
        self.content = 0

    def data_ptr(self):
        return self.pointer

    def untyped_storage(self):
        return _Storage(self.pointer)

    def is_contiguous(self):
        return self.contiguous

    def stride(self, dimension=None):
        values = tuple(1 for _ in self.shape)
        return values if dimension is None else values[dimension]


def _patch_fixture():
    original_torch = object()

    def original_mask(*_args, **_kwargs):
        return None

    def original_apply(*_args, **_kwargs):
        return None

    def original_repeat(*_args, **_kwargs):
        return None

    original_functional = object()
    ARCH.torch = original_torch
    ARCH.make_suffix_attn_mask = original_mask
    ARCH.apply_rotary_pos_emb = original_apply
    ARCH.repeat_kv = original_repeat
    ARCH.F = original_functional

    def position(*_args, **_kwargs):
        return None

    def rotary_forward(*_args, **_kwargs):
        return None

    def attention(*_args, **_kwargs):
        return None

    def time_cond(*_args, **_kwargs):
        return None

    def adaptive(*_args, **_kwargs):
        return None

    def mlp_forward(*_args, **_kwargs):
        return None

    def modulator_forward(*_args, **_kwargs):
        return None

    modulator = types.SimpleNamespace(forward=modulator_forward)
    layer = types.SimpleNamespace(
        mlp=types.SimpleNamespace(forward=mlp_forward)
    )
    rotary = types.SimpleNamespace(forward=rotary_forward)
    expert = types.SimpleNamespace(
        rotary_emb=rotary,
        _attn_fn=attention,
        _adaptive_rmsnorm=adaptive,
        layers=[layer],
    )
    model = types.SimpleNamespace(
        model=types.SimpleNamespace(action_expert=expert),
        _build_suffix_position_ids=position,
        _build_adarms_cond=time_cond,
    )
    runtime = types.SimpleNamespace(
        model=model,
        _modulation_sites=(MODULE._ModulationSite(0, "final", modulator),),
        _static_cache=object(),
        _cache_tensors=lambda _cache: ((), ()),
        _pack_workspaces=types.SimpleNamespace(),
        combined_patches_restored=False,
    )
    state = MODULE._CaptureState(
        census=MODULE._CaptureCensus(),
        metadata=MODULE._MetadataOwner(),
    )
    originals = {
        "torch": original_torch,
        "mask": original_mask,
        "apply": original_apply,
        "repeat": original_repeat,
        "functional": original_functional,
        "position": position,
        "rotary": rotary_forward,
        "attention": attention,
        "time": time_cond,
        "adaptive": adaptive,
        "mlp": mlp_forward,
        "modulator": modulator_forward,
    }
    return runtime, state, originals, layer, modulator


def _contains_pointer_key(value):
    if isinstance(value, dict):
        return any(
            "ptr" in str(key).lower()
            or _contains_pointer_key(item)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(_contains_pointer_key(item) for item in value)
    return False


class CombinedRuntimeContractTest(unittest.TestCase):
    def test_capture_census_is_one_exact_composition(self):
        census = MODULE._expected_capture_census()

        self.assertEqual(census["suffix_mask_builds"], 1)
        self.assertEqual(census["suffix_position_builds"], 1)
        self.assertEqual(census["bool_mask_builds"], 1)
        self.assertEqual(census["sliding_rope_builds"], 1)
        self.assertEqual(census["full_rope_builds"], 1)
        self.assertEqual(census["direct_pack_out_builds"], 680)
        self.assertEqual(census["identity_repeat_returns"], 680)
        self.assertEqual(census["table_lookup_calls"], 690)
        self.assertEqual(census["native_modulator_linear_calls"], 0)
        self.assertEqual(census["native_var_calls"], 690)
        self.assertEqual(census["native_rsqrt_calls"], 690)
        self.assertEqual(census["exact_affine_kernel_calls"], 690)
        self.assertEqual(census["fallback_count"], 0)

    def test_capture_patch_restores_every_binding_after_exception(self):
        runtime, state, originals, layer, modulator = _patch_fixture()

        with self.assertRaisesRegex(RuntimeError, "synthetic capture failure"):
            with MODULE._combined_capture_patch(runtime, state):
                self.assertIsNot(ARCH.torch, originals["torch"])
                self.assertIsNot(
                    runtime.model.model.action_expert._attn_fn,
                    originals["attention"],
                )
                raise RuntimeError("synthetic capture failure")

        self.assertIs(ARCH.torch, originals["torch"])
        self.assertIs(ARCH.make_suffix_attn_mask, originals["mask"])
        self.assertIs(ARCH.apply_rotary_pos_emb, originals["apply"])
        self.assertIs(ARCH.repeat_kv, originals["repeat"])
        self.assertIs(ARCH.F, originals["functional"])
        self.assertIs(
            runtime.model._build_suffix_position_ids, originals["position"]
        )
        self.assertIs(
            runtime.model.model.action_expert.rotary_emb.forward,
            originals["rotary"],
        )
        self.assertIs(
            runtime.model.model.action_expert._attn_fn,
            originals["attention"],
        )
        self.assertIs(runtime.model._build_adarms_cond, originals["time"])
        self.assertIs(
            runtime.model.model.action_expert._adaptive_rmsnorm,
            originals["adaptive"],
        )
        self.assertIs(layer.mlp.forward, originals["mlp"])
        self.assertIs(modulator.forward, originals["modulator"])
        self.assertTrue(runtime.combined_patches_restored)

    def test_patch_lock_is_held_before_any_binding_mutation(self):
        runtime, state, originals, _layer, _modulator = _patch_fixture()
        entered = threading.Event()
        finished = threading.Event()
        errors = []

        def worker():
            try:
                with MODULE._combined_capture_patch(runtime, state):
                    entered.set()
            except Exception as exc:  # pragma: no cover - asserted below
                errors.append(exc)
            finally:
                finished.set()

        MODULE._PATCH_LOCK.acquire()
        try:
            thread = threading.Thread(target=worker)
            thread.start()
            self.assertFalse(entered.wait(0.05))
            self.assertIs(ARCH.torch, originals["torch"])
        finally:
            MODULE._PATCH_LOCK.release()
        self.assertTrue(finished.wait(1.0))
        thread.join()
        self.assertEqual(errors, [])
        self.assertIs(ARCH.torch, originals["torch"])

    def test_request_dynamic_metadata_requires_addresses_not_immutable_bytes(self):
        table = (_FakeTensor(10),)
        metadata = (_FakeTensor(20), _FakeTensor(21))
        workspaces = (_FakeTensor(30), _FakeTensor(31))
        outputs = (_FakeTensor(40),)
        runtime = object.__new__(MODULE.DM05CombinedRuntime)
        runtime._modulation_table = types.SimpleNamespace(tensors=lambda: table)
        runtime._metadata_owner = types.SimpleNamespace(tensors=lambda: metadata)
        runtime._pack_workspaces = types.SimpleNamespace(tensors=lambda: workspaces)
        runtime._affine_outputs = outputs
        runtime._combined_address_owners = {
            "table": MODULE._data_addresses(table),
            "metadata": MODULE._data_addresses(metadata),
            "workspaces": MODULE._data_addresses(workspaces),
            "affine_outputs": MODULE._data_addresses(outputs),
        }
        metadata[0].content = 7

        runtime._assert_combined_owners(check_bytes=False)

        self.assertTrue(runtime.metadata_owner_addresses_stable)

    def test_codegen_guard_requires_sm89_exact_scalar_order_and_no_local(self):
        ptx = """
.version 8.0
.target sm_89
.visible .entry _dm05_exact_postreduce_affine_kernel_0() {
  mul.rn.f32 %f1, %f2, %f3;
  add.rn.f32 %f4, %f1, %f5;
  mul.rn.f32 %f6, %f4, %f7;
  add.rn.f32 %f8, %f6, %f9;
  cvt.rn.bf16.f32 %h1, %f8;
}
"""
        record = MODULE._validate_ptx_codegen(
            ptx,
            cubin=b"fixture-cubin",
            compiled_name="_dm05_exact_postreduce_affine_kernel_0",
            device_capability=(8, 9),
        )
        bad = MODULE._validate_ptx_codegen(
            ptx + "\nfma.rn.f32 %f1, %f2, %f3, %f4;\n",
            cubin=b"fixture-cubin",
            compiled_name="_dm05_exact_postreduce_affine_kernel_0",
            device_capability=(8, 9),
        )
        wrong_arch = MODULE._validate_ptx_codegen(
            ptx,
            cubin=b"fixture-cubin",
            compiled_name="_dm05_exact_postreduce_affine_kernel_0",
            device_capability=(9, 0),
        )

        self.assertTrue(record["ptx_codegen_verified"])
        from apxinf.policies.impls import dm05

        self.assertEqual(
            tuple(record), dm05._COMBINED_PTX_CODEGEN_FIELDS
        )
        self.assertEqual(record["checks"], dm05._COMBINED_PTX_CHECKS)
        self.assertEqual(
            record["forbidden_hits"], dm05._COMBINED_PTX_FORBIDDEN_HITS
        )
        self.assertEqual(len(record["ptx_sha256"]), 64)
        self.assertEqual(len(record["cubin_sha256"]), 64)
        self.assertTrue(record["sass_external_receipt_required"])
        self.assertFalse(bad["ptx_codegen_verified"])
        self.assertFalse(wrong_arch["ptx_codegen_verified"])

    def test_codegen_guard_rejects_every_mad_rounding_mode(self):
        base = """
.version 8.0
.target sm_89
.visible .entry _dm05_exact_postreduce_affine_kernel_0() {
  mul.rn.f32 %f1, %f2, %f3;
  add.rn.f32 %f4, %f1, %f5;
  mul.rn.f32 %f6, %f4, %f7;
  add.rn.f32 %f8, %f6, %f9;
  cvt.rn.bf16.f32 %h1, %f8;
  {instruction}
}
"""
        for mode in ("", ".rn", ".rz", ".rm", ".rp", ".rn.ftz", ".ftz"):
            record = MODULE._validate_ptx_codegen(
                base.replace(
                    "{instruction}",
                    f"mad{mode}.f32 %f1, %f2, %f3, %f4;",
                ),
                cubin=b"fixture-cubin",
                compiled_name="_dm05_exact_postreduce_affine_kernel_0",
                device_capability=(8, 9),
            )
            self.assertFalse(record["ptx_codegen_verified"], mode)

    def test_missing_triton_propagates_without_fallback(self):
        old_torch = MODULE.torch
        MODULE.torch = types.SimpleNamespace(
            cuda=types.SimpleNamespace(
                get_device_capability=lambda _device: (8, 9)
            )
        )
        try:
            with mock.patch.object(
                MODULE,
                "_build_triton_kernel",
                side_effect=RuntimeError("there is no fallback"),
            ):
                with self.assertRaisesRegex(RuntimeError, "no fallback"):
                    MODULE._ExactPostReductionAffine("cuda:0")
        finally:
            MODULE.torch = old_torch

    def test_proof_is_pointer_free_and_exposes_startup_oracle(self):
        runtime = object.__new__(MODULE.DM05CombinedRuntime)
        runtime._affine_kernel = types.SimpleNamespace(
            compile_count=1,
            fallback_count=0,
            codegen={
                "ptx_codegen_verified": True,
                "ptx_sha256": "a" * 64,
                "cubin_sha256": "b" * 64,
                "sass_external_receipt_required": True,
            },
            triton=types.SimpleNamespace(__version__="3.6.0"),
        )
        old_torch = MODULE.torch
        MODULE.torch = types.SimpleNamespace(
            __version__="2.11.0+cu130",
            version=types.SimpleNamespace(cuda="13.0"),
        )
        runtime.combined_ready = True
        runtime.dm05_arch_source_sha256 = "b" * 64
        runtime._capture_census = MODULE._expected_capture_census()
        runtime.combined_capture_census_exact = True
        runtime.combined_patches_restored = True
        runtime.modulation_table_build_count = 1
        runtime.modulation_table_entry_count = 690
        runtime.modulation_table_rng_unchanged = True
        runtime.modulation_table_addresses_stable = True
        runtime.modulation_table_immutable = True
        runtime.metadata_owner_tensor_count = 7
        runtime.metadata_owner_addresses_stable = True
        runtime.replay_content_baseline_established = True
        runtime.metadata_owner_second_replay_exact = True
        runtime.pack_workspace_count = 2
        runtime.pack_workspace_addresses_stable = True
        runtime.pack_workspace_second_replay_exact = True
        runtime.affine_output_count = 690
        runtime.affine_output_addresses_stable = True
        runtime.affine_output_second_replay_exact = True
        runtime.startup_native_suffix_reference_count = 1
        runtime.startup_graph_replay_count = 4
        runtime.startup_first_replay_output_bitwise_exact = True
        runtime.startup_second_replay_output_bitwise_exact = True
        runtime.startup_changed_noise_control_count = 1
        runtime.startup_changed_noise_graph_vs_eager_bitwise_exact = True
        runtime.startup_changed_noise_differs_from_zero_baseline = True
        runtime.startup_static_zero_input_restored = True
        runtime.startup_static_zero_repeat_bitwise_exact = True
        runtime.startup_output_poison_count = 4
        runtime.startup_native_reference_bitwise = True

        try:
            proof = runtime.proof_snapshot()
        finally:
            MODULE.torch = old_torch

        self.assertEqual(proof["selector"], "default_exact_combined")
        self.assertEqual(proof["execution_backend"], "default_exact_combined")
        self.assertEqual(proof["schema"], "apxinf.dm05.exact-combined.v1")
        self.assertTrue(proof["no_fallback"])
        self.assertTrue(proof["initialized"])
        self.assertEqual(proof["prefix_graph_replay_count"], 3)
        self.assertEqual(proof["suffix_graph_replay_count"], 3)
        self.assertTrue(proof["startup_native_reference_bitwise"])
        self.assertEqual(proof["startup_graph_replay_count"], 4)
        self.assertEqual(proof["startup_output_poison_count"], 4)
        self.assertEqual(proof["startup_changed_noise_control_count"], 1)
        self.assertTrue(
            proof["startup_changed_noise_graph_vs_eager_bitwise_exact"]
        )
        self.assertTrue(
            proof["startup_changed_noise_differs_from_zero_baseline"]
        )
        self.assertTrue(proof["startup_static_zero_input_restored"])
        self.assertTrue(proof["startup_static_zero_repeat_bitwise_exact"])
        self.assertTrue(proof["replay_content_baseline_established"])
        self.assertTrue(proof["metadata_owner_second_replay_exact"])
        self.assertTrue(proof["pack_workspace_second_replay_exact"])
        self.assertTrue(proof["affine_output_second_replay_exact"])
        self.assertTrue(proof["exact_affine_ptx_codegen_verified"])
        self.assertTrue(proof["sass_external_receipt_required"])
        self.assertTrue(proof["metadata_owner_bytes_are_request_dynamic"])
        self.assertNotIn("metadata_owner_immutable", proof)
        self.assertFalse(_contains_pointer_key(proof))

    def test_native_oracle_and_version_checks_precede_ready(self):
        source = inspect.getsource(
            MODULE.DM05CombinedRuntime._initialize_suffix_graph
        )
        constructor = inspect.getsource(MODULE.DM05CombinedRuntime.__init__)

        self.assertLess(
            source.index("native_reference.copy_"),
            source.index("self._build_modulation_table"),
        )
        self.assertEqual(source.count("torch.equal(native_reference"), 3)
        self.assertEqual(source.count('fill_(float("nan"))'), 4)
        self.assertEqual(source.count("suffix_graph.replay()"), 4)
        cursor = 0
        replay_positions = []
        for _ in range(4):
            poison = source.index('fill_(float("nan"))', cursor)
            replay = source.index("suffix_graph.replay()", poison)
            self.assertLess(poison, replay)
            replay_positions.append(replay)
            cursor = replay + 1
        baseline = source.index("_record_replay_content_baseline()")
        self.assertLess(replay_positions[0], baseline)
        self.assertLess(baseline, replay_positions[1])
        control = source.index("control_noise =")
        restore = source.index("self._static_noise.zero_()")
        self.assertLess(replay_positions[1], control)
        self.assertLess(control, replay_positions[2])
        self.assertLess(replay_positions[2], restore)
        self.assertLess(restore, replay_positions[3])
        self.assertLess(
            source.index("startup_native_reference_bitwise = True"),
            source.index("combined_ready = True"),
        )
        self.assertIn("_REQUIRED_TORCH_VERSION", constructor)
        self.assertIn("_REQUIRED_CUDA_VERSION", constructor)

    def test_close_drops_every_combined_owner(self):
        runtime = object.__new__(MODULE.DM05CombinedRuntime)
        runtime._modulation_table = object()
        runtime._metadata_owner = object()
        runtime._pack_workspaces = object()
        runtime._affine_outputs = (object(),)
        runtime._combined_address_owners = {"table": (1,)}
        runtime.combined_ready = True

        runtime.close()

        self.assertTrue(runtime._closed)
        self.assertIsNone(runtime._modulation_table)
        self.assertIsNone(runtime._metadata_owner)
        self.assertIsNone(runtime._pack_workspaces)
        self.assertEqual(runtime._affine_outputs, ())
        self.assertEqual(runtime._combined_address_owners, {})
        self.assertFalse(runtime.combined_ready)


class RuntimeSelectorContractTest(unittest.TestCase):
    def test_policy_default_factory_is_apxinf_owned(self):
        from apxinf.policies.impls.dm05 import _load_default_runtime_factory

        factory = _load_default_runtime_factory()
        self.assertEqual(
            factory.__module__, "apxinf.policies.impls.dm05_runtime"
        )

    def test_selector_is_default_off_lazy_and_fail_closed(self):
        from apxinf.policies.impls.dm05_runtime import create_dm05_runtime

        model = object()
        self.assertIsNone(create_dm05_runtime(model))
        with self.assertRaisesRegex(ValueError, "Unsupported DM05 runtime selector"):
            create_dm05_runtime(model, selector="fast_or_fallback")

        fake_module = types.ModuleType(
            "apxinf.policies.impls.dm05_combined_runtime"
        )

        class FakeCombinedRuntime:
            def __init__(self, bound_model, **kwargs):
                self.model = bound_model
                self.kwargs = kwargs

        fake_module.DM05CombinedRuntime = FakeCombinedRuntime
        with mock.patch.dict(
            sys.modules,
            {"apxinf.policies.impls.dm05_combined_runtime": fake_module},
        ):
            runtime = create_dm05_runtime(
                model,
                selector="default_exact_combined",
                request_prefix_len=564,
                diffusion_steps=10,
            )
        self.assertIs(runtime.model, model)
        self.assertEqual(
            runtime.kwargs,
            {"request_prefix_len": 564, "diffusion_steps": 10},
        )

        subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import sys; "
                    "from apxinf.policies.impls.dm05_runtime import "
                    "create_dm05_runtime; "
                    "assert create_dm05_runtime(object()) is None; "
                    "assert 'apxinf.policies.impls.dm05_combined_runtime' "
                    "not in sys.modules"
                ),
            ],
            check=True,
        )

    def test_product_docs_name_only_the_two_public_selectors(self):
        root = Path(__file__).parents[3]
        text = (root / "README.md").read_text(encoding="utf-8")
        self.assertIn("`default_exact_combined`", text)
        self.assertIn("`default`", text)

    def test_official_source_helper_is_model_local_and_patchable(self):
        source = (
            Path(__file__).parents[1]
            / "apxinf"
            / "policies"
            / "impls"
            / "dm05_static_mask_prefix_graph.py"
        ).read_text(encoding="utf-8")
        helper = source.split("def _official_euler_suffix(", 1)[1].split(
            "class _StaticPrefixCacheLayer", 1
        )[0]
        self.assertIn(
            "b5ab170374fbc965aa86d7d370075e8c8bc21bcf46bc6de34e7e336df1af9ce8",
            source,
        )
        self.assertNotIn("model._decode_action_suffix", source)
        self.assertIn("dm05_arch.torch.full", helper)
        self.assertIn("dm05_arch.make_suffix_attn_mask", helper)
        self.assertIn("model._suffix_forward", helper)
        self.assertIn("model.model.action_out_proj", helper)


if __name__ == "__main__":
    unittest.main()
