from __future__ import annotations

import base64
import builtins
import importlib.util
import io
import os
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image


class FakeBackend:
    metadata = {"backend": "fake", "execution_backend": "default"}

    def __init__(self):
        self.calls = []
        self.closed = False

    def infer(self, **kwargs):
        self.calls.append(kwargs)
        return np.arange(70, dtype=np.float32).reshape(10, 7), 12.5

    def close(self):
        self.closed = True


def agreed_external_combined_proof(count: int) -> dict:
    """Literal fixture frozen from ApxInf's exact-combined runtime schema."""

    ready = count > 0
    census = {
        "suffix_mask_calls": 10,
        "suffix_mask_builds": 1,
        "suffix_position_calls": 10,
        "suffix_position_builds": 1,
        "bool_mask_calls": 340,
        "bool_mask_builds": 1,
        "rope_calls": 340,
        "sliding_rope_builds": 1,
        "full_rope_builds": 1,
        "apply_rotary_calls": 340,
        "semantic_cat_calls": 680,
        "direct_pack_out_builds": 680,
        "semantic_repeat_calls": 680,
        "identity_repeat_returns": 680,
        "sdpa_calls": 340,
        "post_sdpa_contiguous_builds": 340,
        "time_cond_calls": 10,
        "adaptive_rmsnorm_calls": 690,
        "table_lookup_calls": 690,
        "native_modulator_linear_calls": 0,
        "native_var_calls": 690,
        "native_rsqrt_calls": 690,
        "exact_affine_kernel_calls": 690,
        "mlp_calls": 340,
        "fallback_count": 0,
    }
    codegen = (
        {
            "level": "compiled_ptx",
            "ptx_sha256": "a" * 64,
            "cubin_sha256": "b" * 64,
            "scalar_fmul_rn_f32_count": 2,
            "scalar_fadd_rn_f32_count": 2,
            "bf16_rne_conversion_count": 1,
            "forbidden_hits": {
                "fma_f32": 0,
                "mad_f32": 0,
                "packed_f32x2": 0,
                "local_declaration": 0,
                "local_load": 0,
                "local_store": 0,
            },
            "checks": {
                "device_sm89": True,
                "ptx_target_sm89": True,
                "compiled_entry_selected": True,
                "ptx_entry_selected": True,
                "two_scalar_fmul_rn_f32": True,
                "two_scalar_fadd_rn_f32": True,
                "bf16_rne_store": True,
                "forbidden_codegen_absent": True,
            },
            "ptx_codegen_verified": True,
            "sass_external_receipt_required": True,
        }
        if ready
        else None
    )
    return {
        "schema": "apxinf.dm05.exact-combined.v1",
        "selector": "default_exact_combined",
        "execution_backend": "default_exact_combined",
        "arithmetic_backend": "native_sdpa_plus_exact_postreduce_triton",
        "graph_scope": "native_prefix_564_plus_combined_suffix_10step",
        "profile_prefix_lengths": [564],
        "initialized": ready,
        "initialization_ms": 123.0 if ready else None,
        "mask_owner_symbol": (
            "transformers.models.gemma3.modeling_gemma3."
            "create_causal_mask_mapping"
        ),
        "mask_modeling_source_sha256": (
            "a1115edf9e0c4a3b53657f21e2de5de0a99488767d84181db6e05e082adb4f69"
        ),
        "mask_utils_source_sha256": (
            "c3c82f7b7b6e03d3f04ba6c6c58a3dd6910623636452ec67ef70e3eb522f9fe7"
        ),
        "dm05_arch_source_sha256": (
            "b5ab170374fbc965aa86d7d370075e8c8bc21bcf46bc6de34e7e336df1af9ce8"
        ),
        "mask_layout_key_verified": ready,
        "mask_helper_build_count": 2 if ready else 0,
        "mask_mapping_keys": (
            ["full_attention", "sliding_attention"] if ready else []
        ),
        "mask_static_address_verified": ready,
        "mask_immutable_verified": ready,
        "prefix_startup_capture_count": 1 if ready else 0,
        "suffix_startup_capture_count": 1 if ready else 0,
        "prefix_capture_execution_count": 1 if ready else 0,
        "suffix_capture_execution_count": 1 if ready else 0,
        "prefix_input_stage_requests": count,
        "prefix_input_tensor_copies": count * 4,
        "prefix_graph_replay_count": count,
        "prefix_graph_cache_write_tensor_copies": count * 68,
        "eager_noise_count": count,
        "suffix_input_stage_requests": count,
        "suffix_input_tensor_copies": count * 2,
        "suffix_graph_replay_count": count,
        "prefix_eager_count": 0,
        "post_prefix_cache_stage_requests": 0,
        "post_prefix_cache_tensor_copies": 0,
        "fallback_count": 0,
        "history_count": 0,
        "result_reuse_count": 0,
        "request_prefix_length": 564 if ready else None,
        "selected_prefix_length": 564 if ready else None,
        "cache_layer_count": 34,
        "prefix_static_cache_address_verified": ready,
        "suffix_static_output_address_verified": ready,
        "closed": False,
        "combined_ready": ready,
        "no_fallback": True,
        "combined_mechanisms": [
            "static_mask_prefix_suffix_graph",
            "modulation_table_690",
            "suffix_metadata_first_owner",
            "two_expanded_kv_pack_workspaces",
            "exact_postreduce_affine_triton",
        ],
        "fixed_cell": {
            "batch_size": 1,
            "prefix_length": 564,
            "suffix_length": 10,
            "hidden_size": 1024,
            "model_action_dim": 32,
            "layer_count": 34,
            "query_heads": 8,
            "kv_heads": 4,
            "head_dim": 256,
            "device_capability": [8, 9],
            "dtype": "torch.bfloat16",
            "attention_backend": "sdpa",
            "torch_version": "2.11.0+cu130",
            "triton_version": "3.6.0",
        },
        "runtime_versions": {
            "torch": "2.11.0+cu130",
            "torch_cuda": "13.0",
            "triton": "3.6.0",
        },
        "combined_capture_census": dict(census) if ready else {},
        "combined_expected_census": dict(census),
        "combined_capture_census_exact": ready,
        "combined_patches_restored": True,
        "modulation_table_build_count": 1 if ready else 0,
        "modulation_table_entry_count": 690 if ready else 0,
        "modulation_table_rng_unchanged": ready,
        "modulation_table_addresses_stable": ready,
        "modulation_table_immutable": ready,
        "metadata_owner_tensor_count": 7 if ready else 0,
        "metadata_owner_addresses_stable": ready,
        "replay_content_baseline_established": ready,
        "metadata_owner_second_replay_exact": ready,
        "metadata_owner_bytes_are_request_dynamic": True,
        "pack_workspace_count": 2 if ready else 0,
        "pack_workspace_addresses_stable": ready,
        "pack_workspace_second_replay_exact": ready,
        "affine_output_count": 690 if ready else 0,
        "affine_output_addresses_stable": ready,
        "affine_output_second_replay_exact": ready,
        "exact_affine_compile_count": 1 if ready else 0,
        "exact_affine_fallback_count": 0,
        "exact_affine_ptx_codegen_verified": ready,
        "exact_affine_ptx_sha256": "a" * 64 if ready else None,
        "exact_affine_cubin_sha256": "b" * 64 if ready else None,
        "sass_external_receipt_required": True,
        "exact_affine_ptx_codegen": codegen,
        "startup_native_suffix_reference_count": 1 if ready else 0,
        "startup_graph_replay_count": 4 if ready else 0,
        "startup_first_replay_output_bitwise_exact": ready,
        "startup_second_replay_output_bitwise_exact": ready,
        "startup_changed_noise_control_count": 1 if ready else 0,
        "startup_changed_noise_graph_vs_eager_bitwise_exact": ready,
        "startup_changed_noise_differs_from_zero_baseline": ready,
        "startup_static_zero_input_restored": ready,
        "startup_static_zero_repeat_bitwise_exact": ready,
        "startup_output_poison_count": 4 if ready else 0,
        "startup_native_reference_bitwise": ready,
        "source_candidate_gpu_validation_required": True,
    }
class FakeCombinedBackend(FakeBackend):
    metadata = {
        "backend": "fake",
        "execution_backend": "default_exact_combined",
        "runtime_selector": "default_exact_combined",
        "host_thread_policy": "fixed_intraop_2",
        "torch_intraop_threads": 2,
        "process_inference_policy": "serialized_all_dm05",
        "precision": "bf16",
        "llm_attention": "eager",
        "vision_attention": "sdpa",
        "action_attention": "sdpa",
    }

    def __init__(self):
        super().__init__()
        self.request_count = 0
        self.pending = None
        self.proof_mutator = None

    def _proof(self):
        proof = agreed_external_combined_proof(self.request_count)
        if self.proof_mutator is not None:
            self.proof_mutator(proof)
        return proof

    def infer(self, **kwargs):
        self.request_count += 1
        self.pending = self._proof()
        return super().infer(**kwargs)

    def consume_path_proof_snapshot(self):
        value = self.pending
        self.pending = None
        return value

    def path_proof_snapshot(self):
        return self._proof()


def image_b64() -> str:
    image = Image.new("RGB", (8, 6), color=(1, 2, 3))
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    return base64.b64encode(stream.getvalue()).decode()


def observation():
    return {
        "prompt": "pick up the bowl",
        "state": [0.0] * 8,
        "images": {"1": image_b64(), "2": image_b64()},
        "robot_type": "Franka",
        "sampling": {"num_steps": 10, "seed": 7},
    }


def test_dm05_registration_is_runtime_lazy():
    from apxinf import Dm05Policy
    from apxinf.policies import available_policies, get_policy

    assert get_policy("dm05") is Dm05Policy
    assert "dm05" in available_policies()
    subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys, apxinf; "
                "assert 'apxinf.policies.impls.dm05_runtime' not in sys.modules; "
                "assert 'apxinf.policies.impls.dm05_combined_runtime' not in sys.modules; "
                "assert 'apxinf.policies.impls.dm05_static_mask_prefix_graph' "
                "not in sys.modules"
            ),
        ],
        check=True,
    )


def test_dm05_public_selector_set_is_exact():
    from apxinf.policies.impls import dm05

    assert dm05.EXECUTION_BACKENDS == ("default", "default_exact_combined")
    for legacy in ("legacy_graph", "legacy_host", "legacy_prefix"):
        with pytest.raises(ValueError, match="execution_backend"):
            dm05._validate_execution_backend(legacy)


def test_dm05_pins_official_reachable_opendm_revision():
    from apxinf.policies.impls.dm05 import OPENDM_COMMIT

    assert OPENDM_COMMIT == "e41e501bb82e9c3cb8138c0fb4687faa5f98c690"


def test_dm05_checkpoint_manifest_covers_complete_semantic_snapshot():
    from apxinf.policies.impls import dm05

    assert dm05._CHECKPOINT_MANIFEST == {
        "chat_template.jinja": (
            1_532,
            "7de1c58e208eda46e9c7f86397df37ec49883aeece39fb961e0a6b24088dd3c4",
        ),
        "config.json": (
            6_795,
            "43b2a56ed9c79c3068849caa0a140458515e654d34ae0b06bb0bbb3ef4dd0f80",
        ),
        "generation_config.json": (
            204,
            "640dbc106facaf0fb90980b5e182ce0c1fcfad6e88da14737578b5b65cb42f7a",
        ),
        "model.safetensors": (
            11_658_431_136,
            "575d0d8e0f75822e95f7adf3a5e62a7c331da0b82e6fc3efeea19ef1b927353f",
        ),
        "norm_stats.json": (
            1_900,
            "06382f26d9f9fdba10ee2dba77783ec8c31e6a6dcb348806583cb6217e18303b",
        ),
        "processor_config.json": (
            560,
            "9eb2e8baf401c81b1517343d1dfc799a4c1b2238acaece111fe68f5fbe3a8d57",
        ),
        "tokenizer.json": (
            33_384_567,
            "daab2354f8a74e70d70b4d1f804939b68a8c9624dd06cb7858e52dd8970e9726",
        ),
        "tokenizer_config.json": (
            715,
            "eb28e3a9807f77cd74dce1b8aed91884621c0302941794470c5a46f884462615",
        ),
    }


def test_dm05_manifest_is_verified_before_semantic_files_are_parsed(
    monkeypatch, tmp_path
):
    from apxinf.policies.impls import dm05

    events = []

    def fake_verified(path, *, size, sha256):
        events.append(("verify", path.name, size, sha256))

    def fake_read(path, *, label):
        events.append(("read", path.name, label))
        if path.name == "config.json":
            return {
                "model_type": "dm05",
                "architectures": ["DM05ForConditionalGeneration"],
                "dtype": "bfloat16",
                "action_dim": 32,
            }
        assert path.name == "norm_stats.json"
        return {
            "norm_stats": {
                "state": {"q01": [0.0] * 8, "q99": [1.0] * 8},
                "action": {"q01": [0.0] * 7, "q99": [1.0] * 7},
            }
        }

    monkeypatch.setattr(dm05, "_verified_file", fake_verified)
    monkeypatch.setattr(dm05, "_read_json_object", fake_read)
    lo, hi = dm05._validate_checkpoint(tmp_path)
    assert lo.shape == hi.shape == (7,)
    manifest_count = len(dm05._CHECKPOINT_MANIFEST)
    assert [event[0] for event in events[:manifest_count]] == [
        "verify"
    ] * manifest_count
    assert [event[1] for event in events[:manifest_count]] == list(
        dm05._CHECKPOINT_MANIFEST
    )
    assert [event[0] for event in events[manifest_count:]] == ["read", "read"]


def test_dm05_git_identity_uses_safe_directory_for_broker_owner(
    monkeypatch, tmp_path
):
    from apxinf.policies.impls import dm05

    package = tmp_path / "opendm"
    package.mkdir()
    origin = package / "__init__.py"
    origin.write_text("", encoding="utf-8")
    root = tmp_path.resolve()
    calls = []

    def fake_run(argv, **kwargs):
        calls.append((argv, kwargs))
        required_prefix = [
            "git",
            "-c",
            f"safe.directory={root}",
            "-C",
            str(root),
        ]
        if argv[:5] != required_prefix:
            raise AssertionError("broker-owned checkout omitted safe.directory")
        stdout = (
            dm05.OPENDM_COMMIT + "\n"
            if argv[5:] == ["rev-parse", "HEAD"]
            else ""
        )
        return SimpleNamespace(stdout=stdout)

    monkeypatch.setattr(
        dm05.importlib.util,
        "find_spec",
        lambda _name: SimpleNamespace(origin=str(origin)),
    )
    monkeypatch.setattr(dm05.subprocess, "run", fake_run)
    assert dm05._opendm_source_root() == root
    assert [call[0] for call in calls] == [
        [
            "git",
            "-c",
            f"safe.directory={root}",
            "-C",
            str(root),
            "rev-parse",
            "HEAD",
        ],
        [
            "git",
            "-c",
            f"safe.directory={root}",
            "-C",
            str(root),
            "status",
            "--porcelain",
        ],
    ]
    assert all(
        kwargs
        == {"check": True, "capture_output": True, "text": True}
        for _, kwargs in calls
    )


def test_dm05_policy_validates_and_returns_policy_contract():
    from apxinf import Dm05Policy

    backend = FakeBackend()
    policy = Dm05Policy(backend)
    result = policy.infer(observation())
    assert result["actions"].shape == (10, 7)
    assert result["actions"].dtype == np.float32
    assert result["timing"]["model_ms"] == 12.5
    assert result["timing"]["total_ms"] > 0
    assert backend.calls[0]["seed"] == 7
    assert [image.size for image in backend.calls[0]["images"]] == [(8, 6), (8, 6)]
    assert "path_proof" not in result
    policy.close()
    assert backend.closed


def test_dm05_default_keeps_arbitrary_integer_seed():
    from apxinf import Dm05Policy

    value = observation()
    value["sampling"]["seed"] = 19
    result = Dm05Policy(FakeBackend()).infer(value)
    assert result["sampling"]["seed"] == 19


def test_dm05_combined_requires_exact_seed_and_proof():
    from apxinf import Dm05Policy

    backend = FakeCombinedBackend()
    policy = Dm05Policy(backend)
    result = policy.infer(observation())
    assert result["path_proof"]["execution_backend"] == "default_exact_combined"
    assert result["path_proof"]["prefix_graph_replay_count"] == 1
    assert policy.path_proof_snapshot()["prefix_graph_replay_count"] == 1

    value = observation()
    value["sampling"]["seed"] = 8
    with pytest.raises(ValueError, match="sampling.seed=7"):
        policy.infer(value)
    assert len(backend.calls) == 1


def test_dm05_agreed_external_proof_fixture_uses_action_hidden_size():
    from apxinf import Dm05Policy

    fixture = agreed_external_combined_proof(1)
    assert fixture["fixed_cell"]["hidden_size"] == 1024
    result = Dm05Policy(FakeCombinedBackend()).infer(observation())
    assert result["path_proof"]["fixed_cell"] == fixture["fixed_cell"]


def test_dm05_combined_rejects_missing_or_process_local_proof():
    from apxinf import Dm05Policy

    backend = FakeCombinedBackend()
    backend.consume_path_proof_snapshot = lambda: None
    with pytest.raises(RuntimeError, match="proof_snapshot"):
        Dm05Policy(backend).infer(observation())

    backend = FakeCombinedBackend()
    backend.consume_path_proof_snapshot = lambda: {
        "execution_backend": "default_exact_combined",
        "static_output_data_ptr": 123,
    }
    with pytest.raises(RuntimeError, match="process-local"):
        Dm05Policy(backend).infer(observation())


@pytest.mark.parametrize(
    "raw_field",
    [
        {"nested": {"storage_ptr": 123}},
        {"nested": {"runtime_address": "0x123"}},
        {"nested": {"mask_static_address_verified": 123}},
    ],
)
def test_dm05_combined_rejects_nested_raw_address_fields(raw_field):
    from apxinf import Dm05Policy

    backend = FakeCombinedBackend()
    backend.consume_path_proof_snapshot = lambda: {
        "execution_backend": "default_exact_combined",
        **raw_field,
    }
    with pytest.raises(RuntimeError, match="process-local"):
        Dm05Policy(backend).infer(observation())


def test_dm05_combined_allows_boolean_address_attestations():
    from apxinf import Dm05Policy

    result = Dm05Policy(FakeCombinedBackend()).infer(observation())
    proof = result["path_proof"]
    assert proof["mask_static_address_verified"] is True
    assert proof["pack_workspace_addresses_stable"] is True


def test_dm05_combined_rejects_wrong_proof_selector():
    from apxinf import Dm05Policy

    backend = FakeCombinedBackend()
    backend.proof_mutator = lambda proof: proof.update(execution_backend="default") \
        if proof["initialized"] else None
    with pytest.raises(RuntimeError, match="execution_backend"):
        Dm05Policy(backend).infer(observation())


def test_dm05_combined_rejects_selector_only_proof():
    from apxinf import Dm05Policy

    backend = FakeCombinedBackend()
    backend.consume_path_proof_snapshot = lambda: {
        "execution_backend": "default_exact_combined"
    }
    with pytest.raises(RuntimeError, match="omitted fields"):
        Dm05Policy(backend).infer(observation())


@pytest.mark.parametrize(
    ("mutator", "match"),
    [
        (
            lambda proof: proof["combined_capture_census"].update(sdpa_calls=339)
            if proof["initialized"]
            else None,
            "sdpa_calls",
        ),
        (
            lambda proof: proof.pop("modulation_table_entry_count", None)
            if proof["initialized"]
            else None,
            "modulation_table_entry_count",
        ),
        (
            lambda proof: proof.update(fallback_count=1)
            if proof["initialized"]
            else None,
            "fallback_count",
        ),
        (
            lambda proof: proof.update(prefix_graph_replay_count=0)
            if proof["initialized"]
            else None,
            "advanced by 0",
        ),
    ],
)
def test_dm05_combined_strict_guard_rejects_invalid_success(mutator, match):
    from apxinf import Dm05Policy

    backend = FakeCombinedBackend()
    backend.proof_mutator = mutator
    with pytest.raises(RuntimeError, match=match):
        Dm05Policy(backend).infer(observation())


def test_dm05_combined_returns_canonical_projection():
    from apxinf import Dm05Policy
    from apxinf.policies.impls import dm05

    proof = Dm05Policy(FakeCombinedBackend()).infer(observation())["path_proof"]
    assert tuple(proof) == dm05._COMBINED_PROOF_TOP_LEVEL_FIELDS
    assert tuple(proof["fixed_cell"]) == tuple(dm05._COMBINED_FIXED_CELL)
    assert tuple(proof["runtime_versions"]) == tuple(
        dm05._COMBINED_RUNTIME_VERSIONS
    )
    assert tuple(proof["combined_capture_census"]) == tuple(
        dm05._COMBINED_CAPTURE_CENSUS
    )
    assert tuple(proof["exact_affine_ptx_codegen"]) == (
        dm05._COMBINED_PTX_CODEGEN_FIELDS
    )


def test_dm05_combined_rejects_innocuous_unknown_runtime_value():
    from apxinf import Dm05Policy

    backend = FakeCombinedBackend()
    backend.proof_mutator = lambda proof: proof.update(
        opaque_runtime_value=0x12345678
    )
    with pytest.raises(RuntimeError, match="unknown fields.*opaque_runtime_value"):
        Dm05Policy(backend).path_proof_snapshot()


@pytest.mark.parametrize(
    ("mutator", "match"),
    [
        (
            lambda proof: proof["fixed_cell"].update(opaque=0),
            "fixed_cell has unknown fields",
        ),
        (
            lambda proof: proof["runtime_versions"].update(opaque="0"),
            "runtime_versions has unknown fields",
        ),
        (
            lambda proof: proof["combined_capture_census"].update(opaque=0),
            "capture census has unknown fields",
        ),
        (
            lambda proof: proof["combined_expected_census"].update(opaque=0),
            "expected census has unknown fields",
        ),
        (
            lambda proof: proof["combined_mechanisms"].append("opaque"),
            "combined_mechanisms",
        ),
        (
            lambda proof: proof["exact_affine_ptx_codegen"].update(opaque=0),
            "exact_affine_ptx_codegen has unknown fields",
        ),
        (
            lambda proof: proof["exact_affine_ptx_codegen"]["checks"].update(
                opaque=True
            ),
            "checks has unknown fields",
        ),
        (
            lambda proof: proof["exact_affine_ptx_codegen"][
                "forbidden_hits"
            ].update(opaque=0),
            "forbidden_hits has unknown fields",
        ),
    ],
)
def test_dm05_combined_rejects_unknown_nested_proof_fields(mutator, match):
    from apxinf import Dm05Policy

    backend = FakeCombinedBackend()
    backend.proof_mutator = lambda proof: mutator(proof) \
        if proof["initialized"] else None
    with pytest.raises(RuntimeError, match=match):
        Dm05Policy(backend).infer(observation())


def test_dm05_policy_construction_uses_only_builtin_factory(monkeypatch, tmp_path):
    from apxinf import Dm05Policy
    from apxinf.policies.impls import dm05

    calls = []

    def fake_backend(model_dir, *, device, execution_backend):
        calls.append((model_dir, device, execution_backend))
        return (
            FakeCombinedBackend()
            if execution_backend == "default_exact_combined"
            else FakeBackend()
        )

    monkeypatch.setattr(dm05, "OpenDMBackend", fake_backend)
    Dm05Policy.from_pretrained(tmp_path)
    Dm05Policy.from_pretrained(
        tmp_path,
        execution_backend="default_exact_combined",
    )
    assert calls == [
        (tmp_path.resolve(), "cuda:0", "default"),
        (tmp_path.resolve(), "cuda:0", "default_exact_combined"),
    ]


def test_dm05_policy_rejects_public_runtime_factory_override(tmp_path):
    from apxinf import Dm05Policy
    from apxinf.policies.impls import dm05

    assert dm05.__all__ == ["Dm05Policy"]

    with pytest.raises(TypeError, match="runtime_factory"):
        Dm05Policy.from_pretrained(
            tmp_path,
            runtime_factory=object(),
        )


def test_dm05_combined_metadata_is_fail_closed():
    from apxinf import Dm05Policy

    for field in (
        "runtime_selector",
        "host_thread_policy",
        "torch_intraop_threads",
        "process_inference_policy",
    ):
        backend = FakeCombinedBackend()
        backend.metadata = dict(backend.metadata)
        backend.metadata.pop(field)
        with pytest.raises(RuntimeError, match=field):
            Dm05Policy(backend)


@pytest.mark.parametrize(
    "field",
    [
        "execution_backend",
        "runtime_selector",
        "host_thread_policy",
        "torch_intraop_threads",
        "process_inference_policy",
        "opendm_commit",
        "precision",
        "llm_attention",
        "vision_attention",
        "action_attention",
        "path_proof",
    ],
)
def test_dm05_policy_rejects_protected_metadata_overrides(field):
    from apxinf import Dm05Policy

    with pytest.raises(ValueError, match="protected fields"):
        Dm05Policy(FakeBackend(), metadata={field: "forged"})


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value.update(robot_type="Aloha"),
        lambda value: value.update(state=[0.0] * 7),
        lambda value: value.update(images={"1": image_b64()}),
        lambda value: value["sampling"].update(num_steps=9),
        lambda value: value.update(extra=True),
    ],
)
def test_dm05_policy_fails_closed(mutate):
    from apxinf import Dm05Policy

    value = observation()
    mutate(value)
    with pytest.raises(ValueError):
        Dm05Policy(FakeBackend()).infer(value)


def test_dm05_policy_rejects_bad_backend_output():
    from apxinf import Dm05Policy

    backend = FakeBackend()
    backend.infer = lambda **kwargs: (np.zeros((50, 32), dtype=np.float32), 1.0)
    with pytest.raises(RuntimeError, match="expected"):
        Dm05Policy(backend).infer(observation())


def test_dm05_attention_is_selected_before_model_construction():
    from apxinf.policies.impls.dm05 import _set_config_attention

    config = SimpleNamespace(
        vlm_config=SimpleNamespace(
            text_config=SimpleNamespace(),
            vision_config=SimpleNamespace(),
        ),
        action_config=SimpleNamespace(),
    )
    _set_config_attention(config)
    assert config.vlm_config._attn_implementation_internal == "eager"
    assert config.vlm_config.text_config._attn_implementation_internal == "eager"
    assert config.vlm_config.vision_config._attn_implementation_internal == "sdpa"
    assert config.action_config._attn_implementation_internal == "sdpa"


def test_dm05_mask_layout_key_excludes_pixels_and_ordinary_token_content():
    from apxinf.policies.impls.dm05 import _mask_layout_key_sha256

    text_config = SimpleNamespace(
        layer_types=["full_attention", "sliding_attention"] * 17,
        sliding_window=1024,
        hidden_size=2560,
        _attn_implementation="eager",
    )
    input_ids = np.arange(564, dtype=np.int64)[None, :] + 100
    input_ids[0, 20:276] = 262144
    values = {
        "input_ids": input_ids,
        "attention_mask": np.ones((1, 564), dtype=np.int64),
        "token_type_ids": np.zeros((1, 564), dtype=np.int64),
        "pixel_values": np.zeros((2, 3, 2, 2), dtype=np.float32),
    }
    kwargs = {
        "image_token_id": 262144,
        "text_config": text_config,
        "padding_idx": 0,
        "embedding_dtype": "torch.bfloat16",
        "embedding_width": 2560,
        "graph_device": "cuda:0",
        "gemma_config_class": "Gemma3Config",
    }
    canonical = _mask_layout_key_sha256(values, **kwargs)

    changed_content = {name: np.array(value, copy=True) for name, value in values.items()}
    changed_content["pixel_values"].fill(17)
    changed_content["input_ids"][0, 400] += 13
    assert _mask_layout_key_sha256(changed_content, **kwargs) == canonical

    changed_layout = {name: np.array(value, copy=True) for name, value in values.items()}
    changed_layout["input_ids"][0, 20] = 123
    assert _mask_layout_key_sha256(changed_layout, **kwargs) != canonical


def test_dm05_cli_has_only_two_selectors():
    script = Path(__file__).resolve().parents[3] / "scripts" / "dm05_http_server.py"
    spec = importlib.util.spec_from_file_location("dm05_server_selector_test", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    parser = module.build_parser()
    required = ["--model-dir", "/model", "--opendm-root", "/opendm"]
    assert parser.parse_args(required).execution_backend == "default"
    assert (
        parser.parse_args(
            required + ["--execution-backend", "default_exact_combined"]
        ).execution_backend
        == "default_exact_combined"
    )
    for legacy in ("legacy_graph", "legacy_host", "legacy_prefix"):
        with pytest.raises(SystemExit):
            parser.parse_args(required + ["--execution-backend", legacy])


def test_dm05_combined_cli_sets_host_policy_before_apxinf_import(monkeypatch):
    script = Path(__file__).resolve().parents[3] / "scripts" / "dm05_http_server.py"
    spec = importlib.util.spec_from_file_location("dm05_server_host_test", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    monkeypatch.setenv("OMP_NUM_THREADS", "17")
    monkeypatch.setenv("MKL_NUM_THREADS", "19")
    imports = []

    class FakePolicy:
        metadata = {}

        def close(self):
            pass

    class FakeAutoPolicy:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            assert kwargs["execution_backend"] == "default_exact_combined"
            return FakePolicy()

    real_import = builtins.__import__

    def checked_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name in {"apxinf", "apxinf.serving"}:
            imports.append(
                (
                    name,
                    os.environ.get("OMP_NUM_THREADS"),
                    os.environ.get("MKL_NUM_THREADS"),
                )
            )
            if name == "apxinf":
                return SimpleNamespace(AutoPolicy=FakeAutoPolicy)
            return SimpleNamespace(serve_dm05_http=lambda *args, **kwargs: None)
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", checked_import)
    module.main(
        [
            "--model-dir",
            "/model",
            "--opendm-root",
            "/opendm",
            "--execution-backend",
            "default_exact_combined",
        ]
    )
    assert imports == [
        ("apxinf", "2", "2"),
        ("apxinf.serving", "2", "2"),
    ]


def test_dm05_host_policy_fails_closed_on_wrong_torch_threads():
    from apxinf.policies.impls.dm05 import _validate_host_thread_policy

    wrong = SimpleNamespace(get_num_threads=lambda: 1)
    with pytest.raises(RuntimeError, match="intra-op threads=2"):
        _validate_host_thread_policy(wrong, "default_exact_combined")
    assert _validate_host_thread_policy(wrong, "default") is None


def test_dm05_process_lock_serializes_default_and_combined_readers():
    from apxinf.policies.impls import dm05

    backend = dm05.OpenDMBackend.__new__(dm05.OpenDMBackend)
    backend._inference_lock = threading.Lock()
    entered = threading.Event()
    finished = threading.Event()
    errors = []

    def fake_infer_locked(**_kwargs):
        entered.set()
        return np.zeros((10, 7), dtype=np.float32), 1.0

    backend._infer_locked = fake_infer_locked

    def worker():
        try:
            backend.infer(
                prompt="task",
                state=np.zeros(8, dtype=np.float32),
                images=(),
                robot_type="Franka",
                num_steps=10,
                seed=7,
            )
        except Exception as exc:  # pragma: no cover - asserted below
            errors.append(exc)
        finally:
            finished.set()

    dm05._DM05_PROCESS_INFERENCE_LOCK.acquire()
    try:
        thread = threading.Thread(target=worker)
        thread.start()
        assert not entered.wait(0.05)
    finally:
        dm05._DM05_PROCESS_INFERENCE_LOCK.release()
    assert finished.wait(1.0)
    thread.join()
    assert errors == []
    assert entered.is_set()
