from __future__ import annotations

import base64
import importlib.util
import io
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image


class FakeBackend:
    metadata = {
        "backend": "apxinf-native",
        "model_revision": "25a8e0d38a8eaeaae41a44d7b4a2378fd8ce1088",
        "device": "cuda:0",
        "precision": "bf16",
    }

    def __init__(self):
        self.calls = []
        self.closed = False

    def infer(self, rgb_u8, token_ids, *, seed, noise):
        self.calls.append(
            {
                "rgb_u8": np.array(rgb_u8, copy=True),
                "token_ids": np.array(token_ids, copy=True),
                "seed": seed,
                "noise": None if noise is None else np.array(noise, copy=True),
            }
        )
        output = np.zeros((10, 32), dtype=np.float32)
        output[:, :7] = np.arange(70, dtype=np.float32).reshape(10, 7) / 100.0
        return output, 12.5

    def close(self):
        self.closed = True


class FakeProcessor:
    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        ids = np.full((1, 564), 2, dtype=np.int64)
        ids[:, 37:293] = 262_144
        ids[:, 301:557] = 262_144
        return {"input_ids": ids}


class FakeCv2:
    BORDER_CONSTANT = 0
    INTER_LINEAR = 1

    @staticmethod
    def copyMakeBorder(array, top, bottom, left, right, border, value):
        assert border == FakeCv2.BORDER_CONSTANT
        return np.pad(
            array,
            ((top, bottom), (left, right), (0, 0)),
            mode="constant",
            constant_values=0,
        )

    @staticmethod
    def resize(array, size, interpolation):
        assert interpolation == FakeCv2.INTER_LINEAR
        return np.asarray(Image.fromarray(array).resize(size, Image.Resampling.BILINEAR))


def image_b64(color=(1, 2, 3)) -> str:
    image = Image.new("RGB", (8, 6), color=color)
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    return base64.b64encode(stream.getvalue()).decode()


def observation():
    return {
        "prompt": "pick up the bowl",
        "state": [0.0] * 8,
        "images": {"1": image_b64(), "2": image_b64((4, 5, 6))},
        "robot_type": "Franka",
        "sampling": {"num_steps": 10, "seed": 7},
    }


def policy(backend=None):
    from apxinf import Dm05Policy

    return Dm05Policy(
        backend or FakeBackend(),
        processor=FakeProcessor(),
        cv2_module=FakeCv2,
        action_low=np.full(7, -1.0, dtype=np.float32),
        action_high=np.full(7, 1.0, dtype=np.float32),
    )


def test_dm05_registration_is_runtime_lazy():
    from apxinf import Dm05Policy
    from apxinf.policies import available_policies, get_policy

    assert get_policy("dm05") is Dm05Policy
    assert "dm05" in available_policies()
    assert "apxinf_py" not in sys.modules
    assert "opendm" not in sys.modules


def test_dm05_checkpoint_manifest_covers_complete_semantic_snapshot():
    from apxinf.policies.impls import dm05

    assert set(dm05._CHECKPOINT_MANIFEST) == {
        "chat_template.jinja",
        "config.json",
        "generation_config.json",
        "model.safetensors",
        "norm_stats.json",
        "processor_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
    }
    assert dm05._CHECKPOINT_MANIFEST["model.safetensors"] == (
        11_658_431_136,
        "575d0d8e0f75822e95f7adf3a5e62a7c331da0b82e6fc3efeea19ef1b927353f",
    )


def test_dm05_native_policy_forwards_exact_public_contract_and_noise():
    backend = FakeBackend()
    instance = policy(backend)
    noise = np.arange(320, dtype=np.float32).reshape(10, 32)
    result = instance.infer(observation(), noise=noise)

    assert result["actions"].shape == (10, 7)
    assert result["normalized_actions"].shape == (10, 32)
    assert np.allclose(
        result["actions"], result["normalized_actions"][:, :7], atol=1e-6
    )
    assert result["timing"]["model_ms"] == 12.5
    call = backend.calls[0]
    assert call["rgb_u8"].shape == (2, 448, 448, 3)
    assert call["rgb_u8"].dtype == np.uint8
    assert call["token_ids"].shape == (564,)
    assert call["token_ids"].dtype == np.uint32
    assert np.array_equal(call["noise"], noise)


def test_dm05_processor_template_is_exact_and_state_is_not_consumed():
    instance = policy()
    first = observation()
    second = observation()
    second["state"] = [0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8]
    left = instance.infer(first, noise=np.zeros((10, 32), dtype=np.float32))
    right = instance.infer(second, noise=np.zeros((10, 32), dtype=np.float32))
    assert np.array_equal(left["actions"], right["actions"])

    messages, kwargs = instance.processor.calls[0]
    content = messages[0]["content"]
    assert content[0]["text"] == (
        "Robot: Franka\nOverall speed: 0.5\nTask: pick up the bowl.\nHead image: "
    )
    assert content[2]["text"] == "Left wrist image: "
    assert kwargs["add_generation_prompt"] is True
    assert kwargs["return_tensors"] == "np"
    assert "state" not in repr(messages).lower()


def test_dm05_seed_path_is_explicit_and_negative_seed_is_mapped():
    backend = FakeBackend()
    instance = policy(backend)
    value = observation()
    value["sampling"]["seed"] = -1
    instance.infer(value)
    assert backend.calls[0]["seed"] == -1
    assert backend.calls[0]["noise"] is None


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value.update(robot_type="Aloha"),
        lambda value: value.update(state=[0.0] * 7),
        lambda value: value.update(images={"1": image_b64()}),
        lambda value: value["sampling"].update(num_steps=9),
        lambda value: value["sampling"].update(seed=1 << 64),
        lambda value: value.update(extra=True),
    ],
)
def test_dm05_policy_fails_closed_on_wire_drift(mutate):
    value = observation()
    mutate(value)
    with pytest.raises(ValueError):
        policy().infer(value)


def test_dm05_policy_rejects_bad_native_output_and_noise():
    backend = FakeBackend()
    backend.infer = lambda *args, **kwargs: (
        np.zeros((50, 32), dtype=np.float32),
        1.0,
    )
    with pytest.raises(RuntimeError, match="expected"):
        policy(backend).infer(observation())

    with pytest.raises(ValueError, match="noise"):
        policy().infer(observation(), noise=np.zeros((10, 7), dtype=np.float32))


@pytest.mark.parametrize(
    "field",
    [
        "schema",
        "model_type",
        "model_revision",
        "backend",
        "device",
        "precision",
        "action_horizon",
        "action_dim",
        "model_action_dim",
        "state_dim",
        "state_conditioned",
        "image_size",
        "image_prompts",
        "robot_type",
        "diffusion_steps",
        "concurrency",
    ],
)
def test_dm05_policy_rejects_canonical_metadata_override(field):
    from apxinf import Dm05Policy

    with pytest.raises(ValueError, match="canonical fields"):
        Dm05Policy(
            FakeBackend(),
            processor=FakeProcessor(),
            cv2_module=FakeCv2,
            action_low=np.full(7, -1.0, dtype=np.float32),
            action_high=np.full(7, 1.0, dtype=np.float32),
            metadata={field: "forged"},
        )


def test_dm05_from_pretrained_binds_only_native_backend(monkeypatch, tmp_path):
    from apxinf import Dm05Policy
    from apxinf.policies.impls import dm05

    calls = []
    monkeypatch.setattr(
        dm05,
        "_validate_checkpoint",
        lambda path: (
            np.full(7, -1.0, dtype=np.float32),
            np.full(7, 1.0, dtype=np.float32),
        ),
    )

    class ProcessorFactory:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return FakeProcessor()

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(__version__="5.3.0", AutoProcessor=ProcessorFactory),
    )
    monkeypatch.setitem(sys.modules, "cv2", FakeCv2)

    def native(path, *, device, default_seed):
        calls.append((path, device, default_seed))
        return FakeBackend()

    monkeypatch.setattr(dm05, "ApxInfNativeBackend", native)
    loaded = Dm05Policy.from_pretrained(tmp_path, default_seed=11)
    assert calls == [(tmp_path.resolve(), "cuda:0", 11)]
    assert loaded.metadata["backend"] == "apxinf-native"

    with pytest.raises(TypeError, match="execution_backend"):
        Dm05Policy.from_pretrained(tmp_path, execution_backend="default")
    with pytest.raises(TypeError, match="opendm_root"):
        Dm05Policy.from_pretrained(tmp_path, opendm_root=tmp_path)


def test_dm05_cli_has_no_external_runtime_surface():
    script = Path(__file__).resolve().parents[3] / "scripts" / "dm05_http_server.py"
    spec = importlib.util.spec_from_file_location("dm05_native_server_test", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    parser = module.build_parser()
    args = parser.parse_args(["--model-dir", "/model"])
    assert args.model_dir == "/model"
    with pytest.raises(SystemExit):
        parser.parse_args(["--model-dir", "/model", "--opendm-root", "/opendm"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--model-dir", "/model", "--execution-backend", "default"])


def test_dm05_removed_external_runtime_modules_are_absent():
    root = Path(__file__).resolve().parents[1] / "apxinf/policies/impls"
    assert not (root / "dm05_runtime.py").exists()
    assert not (root / "dm05_static_mask_prefix_graph.py").exists()
    assert not (root / "dm05_combined_runtime.py").exists()
