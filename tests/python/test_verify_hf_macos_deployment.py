from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import stat
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "scripts/verify_hf_macos_deployment.py"
PROFILE_PATH = ROOT / "configs/hf-onboarding/qwen35-0.8b-macos-cpu.json"
SPEC = importlib.util.spec_from_file_location("verify_hf_macos_deployment", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


class DeploymentFixture:
    repo_id = "org/tiny-qwen"
    commit = "a" * 40
    token_ids = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

    def __init__(self, root: Path) -> None:
        self.root = root
        self.model_dir = root / "model"
        self.model_dir.mkdir()
        self.payloads = {
            "config.json": b'{"model_type":"qwen3_5"}\n',
            "model.safetensors.index.json": b'{"weight_map":{"x":"model.safetensors-00001-of-00001.safetensors"}}\n',
            "model.safetensors-00001-of-00001.safetensors": b"tiny-safe-tensors-fixture",
            "tokenizer.json": b'{"version":"1.0"}\n',
            "tokenizer_config.json": b'{"tokenizer_class":"Qwen"}\n',
            "chat_template.jinja": b"{{ messages }}\n",
        }
        for name, payload in self.payloads.items():
            (self.model_dir / name).write_bytes(payload)
        cache = self.model_dir / ".cache/huggingface/trees"
        cache.mkdir(parents=True)
        (cache / f"{self.commit}.json").write_text("{}\n", encoding="utf-8")

        self.artifacts = {
            name: {"size": len(payload), "sha256": digest(payload)}
            for name, payload in self.payloads.items()
        }
        self.runtime = {
            "torch": "9.1.0",
            "transformers": "9.2.0",
            "safetensors": "9.3.0",
            "device": "cpu",
            "dtype": "float32",
            "attention_implementation": "eager",
            "use_hub_kernels": False,
            "optional_kernel_packages": [],
            "threads": 1,
            "deterministic_algorithms": True,
        }
        self.source_lock_path = root / "source-lock.json"
        self.manifest_path = root / "manifest.json"
        self.metrics_path = root / "metrics.json"
        self.generation_receipt_path = root / "generation-receipt.json"
        self.memory_receipt_path = root / "memory-receipt.json"
        self.profile_path = root / "profile.json"
        self.binary_path = root / "apxinf"
        self.output_path = root / "deployment-lock.json"

        self.source_lock = self.make_source_lock()
        write_json(self.source_lock_path, self.source_lock)
        self.manifest = self.make_manifest()
        write_json(self.manifest_path, self.manifest)
        self.metrics = self.make_metrics()
        write_json(self.metrics_path, self.metrics)
        self.generation_receipt = self.make_generation_receipt()
        write_json(self.generation_receipt_path, self.generation_receipt)
        self.write_binary(arm64=True, executable=True)
        self.memory_receipt = self.make_memory_receipt()
        write_json(self.memory_receipt_path, self.memory_receipt)
        self.profile = self.make_profile()
        write_json(self.profile_path, self.profile)

    def make_source_lock(self) -> dict[str, object]:
        metadata_names = (
            "config.json",
            "model.safetensors.index.json",
            "tokenizer_config.json",
        )
        checkpoint = self.artifacts["model.safetensors-00001-of-00001.safetensors"]
        value: dict[str, object] = {
            "format": "apxinf-hf-source-lock-v1",
            "repo_id": self.repo_id,
            "requested_revision": self.commit,
            "resolved_commit": self.commit,
            "source": {
                "url": f"https://huggingface.co/{self.repo_id}",
                "license": "apache-2.0",
                "private": False,
                "gated": False,
                "disabled": False,
            },
            "architecture": {"config_sha256": self.artifacts["config.json"]["sha256"]},
            "metadata": {
                "files": [
                    {"path": name, **self.artifacts[name]} for name in metadata_names
                ]
            },
            "weights": {
                "format": "safetensors",
                "index_file": "model.safetensors.index.json",
                "files": [
                    {
                        "path": "model.safetensors-00001-of-00001.safetensors",
                        **checkpoint,
                    }
                ],
                "total_bytes": checkpoint["size"],
            },
            "security": {
                "safetensors_only_plan": True,
                "unsafe_weight_files": [],
                "remote_code_indicators": {"python_files": [], "auto_map_keys": []},
            },
            "policy_receipt": {
                "metadata_only": True,
                "weight_payload_bytes_downloaded": 0,
                "remote_code_executed": False,
                "hf_token_read": False,
            },
        }
        value["content_sha256"] = digest(MODULE.canonical_bytes(value))
        return value

    def make_manifest(self) -> dict[str, object]:
        checkpoint = self.artifacts["model.safetensors-00001-of-00001.safetensors"]
        return {
            "format": "fixture-transformers-oracle-v1",
            "repo_id": self.repo_id,
            "revision": self.commit,
            "checkpoint_sha256": checkpoint["sha256"],
            "runtime": copy.deepcopy(self.runtime),
            "model_dir": str(self.model_dir),
            "uses_locked_default_ids": True,
            "locked_chat": {"enable_thinking": False},
            "greedy_trajectory": {
                "generated_ids": self.token_ids,
                "length": 10,
                "minimum_length": 10,
                "do_sample": False,
                "use_cache": True,
                "eos_stopping": False,
            },
            "snapshot": {
                "checkpoint_sha256": checkpoint["sha256"],
                "verified_files": copy.deepcopy(self.artifacts),
            },
        }

    def make_metrics(self) -> dict[str, object]:
        checkpoint = self.artifacts["model.safetensors-00001-of-00001.safetensors"]
        return {
            "format": "fixture-oracle-comparison-v1",
            "verification": {
                "passed": True,
                "status": "pass",
                "failures": [],
                "checks": [
                    {
                        "name": "greedy_trajectory",
                        "passed": True,
                        "expected": {"generated_ids": self.token_ids, "length": 10},
                        "observed": {"generated_ids": self.token_ids, "length": 10},
                    }
                ],
                "manifest": {
                    "format": "fixture-frozen-gate-v1",
                    "frozen": True,
                    "threshold_overrides_supported": False,
                    "calibration": {
                        "checkpoint_sha256": checkpoint["sha256"],
                        "comparison_format": "fixture-oracle-comparison-v1",
                        "device": "cpu",
                        "dtype": "float32",
                        "matmul_feature": "accelerate",
                    },
                },
            },
            "greedy_trajectory": {
                "apxinf_ids": self.token_ids,
                "expected_ids": self.token_ids,
                "length": 10,
                "minimum_length": 10,
                "exact_match": True,
                "eos_stopping": False,
            },
            "apxinf": {"device": "cpu", "matmul_feature": "accelerate", "max_context": 32},
        }

    def make_profile(self) -> dict[str, object]:
        return {
            "format": "apxinf-hf-macos-deployment-profile-v1",
            "profile_id": "tiny-qwen-macos-cpu",
            "source": {
                "repo_id": self.repo_id,
                "resolved_commit": self.commit,
                "license": "Apache-2.0",
                "source_lock_content_sha256": self.source_lock["content_sha256"],
                "config_sha256": self.artifacts["config.json"]["sha256"],
            },
            "artifacts": copy.deepcopy(self.artifacts),
            "binary": {
                "size": self.binary_path.stat().st_size,
                "sha256": digest(self.binary_path.read_bytes()),
                "build": {
                    "target_os": "macos",
                    "target_arch": "aarch64",
                    "matmul_feature": "accelerate",
                },
            },
            "runtime": {
                "target": "macos-arm64",
                "provider": "native-apxinf-cpu",
                "device": "cpu",
                "dtype": "fp32",
                "matmul_feature": "accelerate",
            },
            "gate": {
                "generation_receipt_format": "apxinf-generation-v1",
                "max_context": 32,
                "max_tokens": 10,
                "no_eos_stop": True,
                "prompt": "Hello",
                "prompt_token_count": 13,
                "generated_token_ids": self.token_ids,
            },
            "memory_smoke": {
                "receipt_format": "apxinf-macos-memory-smoke-v1",
                "measurement": "macos-time-l-vm-stat-v1",
                "max_peak_rss_bytes": 1024 * 1024,
                "max_process_swaps": 0,
                "non_authoritative_evidence": [
                    "pageout_delta_bytes",
                    "swap_delta_bytes",
                    "swap_growth_bytes",
                ],
                "sandbox": "macos-seatbelt-deny-network-write-home-read-v1",
                "timeout_seconds": 5,
            },
            "oracle": {
                "manifest_format": "fixture-transformers-oracle-v1",
                "manifest_sha256": digest(self.manifest_path.read_bytes()),
                "metrics_format": "fixture-oracle-comparison-v1",
                "metrics_sha256": digest(self.metrics_path.read_bytes()),
                "gate_format": "fixture-frozen-gate-v1",
                "runtime": copy.deepcopy(self.runtime),
            },
        }

    def make_generation_receipt(self) -> dict[str, object]:
        return {
            "format": "apxinf-generation-v1",
            "model_type": "qwen3_5",
            "device": "cpu",
            "dtype": "fp32",
            "build": {
                "target_os": "macos",
                "target_arch": "aarch64",
                "matmul_feature": "accelerate",
            },
            "prompt_token_count": 13,
            "generated_token_ids": self.token_ids,
            "profile": {
                "input_tokens": 13,
                "output_tokens": 10,
                "ttft_ms": 10.0,
                "tpot_ms": 2.0,
                "generation_tps": 500.0,
                "total_latency_ms": 28.0,
            },
        }

    def make_memory_receipt(self, *, source: str = "fixture") -> dict[str, object]:
        checkpoint_name = "model.safetensors-00001-of-00001.safetensors"
        stdout_receipt = copy.deepcopy(self.generation_receipt)
        body: dict[str, object] = {
            "format": "apxinf-macos-memory-smoke-v1",
            "measurement": {
                "source": source,
                "platform": "macos",
                "tool": "/usr/bin/time",
                "mode": "-l",
                "sandbox": "/usr/bin/sandbox-exec",
                "sandbox_profile_sha256": digest(
                    MODULE._seatbelt_profile(
                        binary_path=self.binary_path, model_dir=self.model_dir
                    ).encode("utf-8")
                ),
            },
            "binary": {
                "path": str(self.binary_path),
                "sha256": digest(self.binary_path.read_bytes()),
            },
            "model": {
                "directory": str(self.model_dir),
                "checkpoint": checkpoint_name,
                "checkpoint_sha256": self.artifacts[checkpoint_name]["sha256"],
            },
            "argv": [
                str(self.binary_path),
                "generate",
                "--model",
                str(self.model_dir),
                "--prompt",
                "Hello",
                "--max-tokens",
                "10",
                "--max-context",
                "32",
                "--no-eos-stop",
                "--device",
                "cpu",
                "--dtype",
                "fp32",
                "--json",
            ],
            "generation": {
                "input_receipt_sha256": digest(self.generation_receipt_path.read_bytes()),
                "stdout_receipt_sha256": digest(MODULE.canonical_bytes(stdout_receipt)),
                "stdout_receipt": stdout_receipt,
            },
            "result": {
                "exit_code": 0,
                "peak_rss_bytes": 65536,
                "process_swaps": 0,
                "page_size_bytes": 4096,
                "pageouts_before": 100,
                "pageouts_after": 100,
                "pageout_delta": 0,
                "pageout_delta_bytes": 0,
                "swap_used_before_bytes": 0,
                "swap_used_after_bytes": 0,
                "swap_delta_bytes": 0,
                "swap_growth_bytes": 0,
            },
        }
        body["content_sha256"] = digest(MODULE.canonical_bytes(body))
        return body

    def write_memory_receipt(self, receipt: dict[str, object]) -> None:
        body = dict(receipt)
        body.pop("content_sha256", None)
        receipt["content_sha256"] = digest(MODULE.canonical_bytes(body))
        write_json(self.memory_receipt_path, receipt)

    def live_memory_receipt(self) -> dict[str, object]:
        receipt = copy.deepcopy(self.memory_receipt)
        receipt["measurement"]["source"] = "live"
        body = dict(receipt)
        del body["content_sha256"]
        receipt["content_sha256"] = digest(MODULE.canonical_bytes(body))
        return receipt

    def refresh_profile_manifest_hash(self) -> None:
        self.profile["oracle"]["manifest_sha256"] = digest(self.manifest_path.read_bytes())
        write_json(self.profile_path, self.profile)

    def refresh_profile_metrics_hash(self) -> None:
        self.profile["oracle"]["metrics_sha256"] = digest(self.metrics_path.read_bytes())
        write_json(self.profile_path, self.profile)

    def write_binary(self, *, arm64: bool, executable: bool, file_type: int = 2) -> None:
        cpu_type = 0x0100000C if arm64 else 0x01000007
        header = struct.pack("<IiiIIIII", 0xFEEDFACF, cpu_type, 0, file_type, 0, 0, 0, 0)
        self.binary_path.write_bytes(header)
        self.binary_path.chmod(0o755 if executable else 0o644)

    def command(
        self, *, output: bool = False, memory: bool = False, measure: bool = False
    ) -> list[str]:
        command = [
            sys.executable,
            str(MODULE_PATH),
            "--profile",
            str(self.profile_path),
            "--source-lock",
            str(self.source_lock_path),
            "--model-dir",
            str(self.model_dir),
            "--oracle-manifest",
            str(self.manifest_path),
            "--oracle-metrics",
            str(self.metrics_path),
            "--generation-receipt",
            str(self.generation_receipt_path),
            "--binary",
            str(self.binary_path),
        ]
        if memory:
            command.extend(["--memory-receipt", str(self.memory_receipt_path)])
        if measure:
            command.append("--measure-smoke")
        if output:
            command.extend(["--output", str(self.output_path)])
        return command

    def run(
        self, *, output: bool = False, memory: bool = False, measure: bool = False
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            self.command(output=output, memory=memory, measure=measure),
            text=True,
            capture_output=True,
            check=False,
        )


class DeploymentGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.fixture = DeploymentFixture(Path(self.temporary.name))

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def assert_failure(self, result: subprocess.CompletedProcess[str]) -> dict[str, object]:
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertEqual(result.stderr, "")
        self.assertEqual(len(result.stdout.splitlines()), 1)
        receipt = json.loads(result.stdout)
        self.assertIs(receipt["passed"], False)
        return receipt

    def test_valid_fixture_emits_one_json_receipt_without_running_binary(self) -> None:
        before = self.fixture.binary_path.stat().st_mtime_ns
        result = self.fixture.run()
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertEqual(result.stderr, "")
        self.assertEqual(len(result.stdout.splitlines()), 1)
        receipt = json.loads(result.stdout)
        self.assertIs(receipt["passed"], True)
        self.assertIsNone(receipt["output"])
        self.assertEqual(self.fixture.binary_path.stat().st_mtime_ns, before)

    def test_trusted_model_content_can_move_without_rewriting_oracle_provenance(self) -> None:
        oracle_model_dir = self.fixture.manifest["model_dir"]
        relocated_parent = self.fixture.root / "relocated"
        relocated_parent.mkdir()
        relocated_model_dir = relocated_parent / "model"
        self.fixture.model_dir.rename(relocated_model_dir)
        self.fixture.model_dir = relocated_model_dir

        self.assertTrue(relocated_model_dir.is_absolute())
        self.assertNotEqual(str(relocated_model_dir), oracle_model_dir)
        self.assertEqual(self.fixture.manifest["model_dir"], oracle_model_dir)
        result = self.fixture.run()
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertEqual(result.stderr, "")
        self.assertIs(json.loads(result.stdout)["passed"], True)

    def test_tampered_model_artifact_is_rejected(self) -> None:
        (self.fixture.model_dir / "tokenizer.json").write_bytes(b"tampered")
        receipt = self.assert_failure(self.fixture.run())
        self.assertIn("tokenizer.json", receipt["error"]["message"])

    def test_model_artifact_symlink_is_rejected(self) -> None:
        target = self.fixture.root / "replacement-config.json"
        target.write_bytes(self.fixture.payloads["config.json"])
        artifact = self.fixture.model_dir / "config.json"
        artifact.unlink()
        artifact.symlink_to(target)
        receipt = self.assert_failure(self.fixture.run())
        self.assertIn("non-symlink", receipt["error"]["message"])

    def test_cache_symlink_and_script_are_rejected(self) -> None:
        cache = self.fixture.model_dir / ".cache"
        target = self.fixture.root / "outside.json"
        target.write_text("{}\n", encoding="utf-8")
        link = cache / "unsafe-link"
        link.symlink_to(target)
        receipt = self.assert_failure(self.fixture.run())
        self.assertIn("symlink", receipt["error"]["message"])
        link.unlink()
        (cache / "payload.py").write_text("print('never execute')\n", encoding="utf-8")
        receipt = self.assert_failure(self.fixture.run())
        self.assertIn("script", receipt["error"]["message"])

    def test_extra_top_level_python_file_is_rejected(self) -> None:
        (self.fixture.model_dir / "modeling_remote.py").write_text(
            "raise RuntimeError('never execute')\n", encoding="utf-8"
        )
        receipt = self.assert_failure(self.fixture.run())
        self.assertIn("unexpected", receipt["error"]["message"])

    def test_semantically_stale_oracle_is_rejected_even_when_profile_hash_is_refreshed(self) -> None:
        self.fixture.manifest["revision"] = "b" * 40
        write_json(self.fixture.manifest_path, self.fixture.manifest)
        self.fixture.refresh_profile_manifest_hash()
        receipt = self.assert_failure(self.fixture.run())
        self.assertIn("revision", receipt["error"]["message"])

    def test_failed_metrics_are_rejected(self) -> None:
        self.fixture.metrics["verification"]["passed"] = False
        write_json(self.fixture.metrics_path, self.fixture.metrics)
        self.fixture.refresh_profile_metrics_hash()
        receipt = self.assert_failure(self.fixture.run())
        self.assertIn("did not pass", receipt["error"]["message"])

    def test_metrics_file_is_bound_by_profile_hash(self) -> None:
        self.fixture.metrics["forged_pass"] = True
        write_json(self.fixture.metrics_path, self.fixture.metrics)
        receipt = self.assert_failure(self.fixture.run())
        self.assertIn("metrics SHA-256", receipt["error"]["message"])

    def test_stale_generation_receipt_is_rejected(self) -> None:
        self.fixture.generation_receipt["generated_token_ids"][-1] += 1
        write_json(
            self.fixture.generation_receipt_path,
            self.fixture.generation_receipt,
        )
        receipt = self.assert_failure(self.fixture.run())
        self.assertIn("trajectory", receipt["error"]["message"])

    def test_fake_memory_receipt_is_strictly_validated_for_audit(self) -> None:
        result = self.fixture.run(memory=True)
        self.assertEqual(result.returncode, 0, result.stdout)
        receipt = json.loads(result.stdout)
        self.assertEqual(receipt["memory_smoke"]["origin"], "file")
        self.assertIs(receipt["memory_smoke"]["present"], True)

    def test_memory_receipt_binds_binary_argv_and_generation_receipt(self) -> None:
        cases = (
            ("binary", lambda value: value["binary"].update({"sha256": "0" * 64})),
            ("argv", lambda value: value["argv"].pop()),
            (
                "generation receipt",
                lambda value: value["generation"].update(
                    {"input_receipt_sha256": "0" * 64}
                ),
            ),
        )
        for expected_message, mutate in cases:
            with self.subTest(binding=expected_message):
                candidate = copy.deepcopy(self.fixture.memory_receipt)
                mutate(candidate)
                self.fixture.write_memory_receipt(candidate)
                receipt = self.assert_failure(self.fixture.run(memory=True))
                self.assertIn(expected_message, receipt["error"]["message"])

    def test_memory_receipt_enforces_rss_and_child_swap_limits(self) -> None:
        cases = (
            ("peak RSS", {"peak_rss_bytes": 1024 * 1024 + 1}),
            ("process swaps", {"process_swaps": 1}),
        )
        for expected_message, changes in cases:
            with self.subTest(limit=expected_message):
                candidate = copy.deepcopy(self.fixture.memory_receipt)
                candidate["result"].update(changes)
                self.fixture.write_memory_receipt(candidate)
                receipt = self.assert_failure(self.fixture.run(memory=True))
                self.assertIn(expected_message, receipt["error"]["message"])

    def test_global_swap_and_pageout_deltas_are_evidence_not_hard_gates(self) -> None:
        candidate = copy.deepcopy(self.fixture.memory_receipt)
        candidate["result"].update(
            {
                "swap_used_after_bytes": 4096,
                "swap_delta_bytes": 4096,
                "swap_growth_bytes": 4096,
                "pageouts_after": 101,
                "pageout_delta": 1,
                "pageout_delta_bytes": 4096,
            }
        )
        self.fixture.write_memory_receipt(candidate)
        result = self.fixture.run(memory=True)
        self.assertEqual(result.returncode, 0, result.stdout)

    def test_binary_must_be_executable_arm64_macho(self) -> None:
        self.fixture.write_binary(arm64=True, executable=False)
        receipt = self.assert_failure(self.fixture.run())
        self.assertIn("executable", receipt["error"]["message"])
        self.fixture.write_binary(arm64=False, executable=True)
        receipt = self.assert_failure(self.fixture.run())
        self.assertIn("arm64", receipt["error"]["message"])
        self.fixture.write_binary(arm64=True, executable=True, file_type=6)
        receipt = self.assert_failure(self.fixture.run())
        self.assertIn("file type", receipt["error"]["message"])

    def test_arm64_stub_cannot_replace_the_profile_pinned_binary(self) -> None:
        self.fixture.binary_path.write_bytes(self.fixture.binary_path.read_bytes() + b"stub")
        self.fixture.binary_path.chmod(0o755)
        receipt = self.assert_failure(self.fixture.run())
        self.assertIn("trusted profile", receipt["error"]["message"])

    def test_production_output_rejects_an_injected_receipt(self) -> None:
        receipt = self.assert_failure(self.fixture.run(output=True, memory=True))
        self.assertIn("requires a live --measure-smoke", receipt["error"]["message"])
        self.assertFalse(self.fixture.output_path.exists())

    def test_lock_writer_is_self_hashed_atomic_and_exclusive(self) -> None:
        arguments = self.fixture.command(memory=True)[2:]
        args = MODULE.parser().parse_args(arguments)
        receipt, lock_value = MODULE.verify(args)
        self.assertIs(receipt["passed"], True)
        MODULE._exclusive_atomic_json(self.fixture.output_path, lock_value)
        lock = json.loads(self.fixture.output_path.read_text(encoding="utf-8"))
        expected = lock.pop("content_sha256")
        self.assertEqual(digest(MODULE.canonical_bytes(lock)), expected)
        self.assertEqual(stat.S_IMODE(self.fixture.output_path.stat().st_mode), 0o600)
        self.assertEqual(lock["memory_smoke"]["origin"], "file")
        self.assertEqual(
            lock["memory_smoke"]["file"]["path"], str(self.fixture.memory_receipt_path)
        )
        self.assertEqual(lock["memory_smoke"]["receipt"]["result"]["peak_rss_bytes"], 65536)
        self.assertEqual(lock["launch"]["program"], str(self.fixture.binary_path))
        self.assertEqual(
            lock["launch"]["fixed_args"],
            [
                "generate",
                "--model",
                str(self.fixture.model_dir),
                "--device",
                "cpu",
                "--dtype",
                "fp32",
                "--max-context",
                "32",
            ],
        )
        original = self.fixture.output_path.read_bytes()
        with self.assertRaisesRegex(MODULE.DeploymentError, "already exists"):
            MODULE._exclusive_atomic_json(self.fixture.output_path, lock_value)
        self.assertEqual(self.fixture.output_path.read_bytes(), original)

    def test_macos_measurement_parsers_are_deterministic(self) -> None:
        vm = (
            b"Mach Virtual Memory Statistics: (page size of 16384 bytes)\n"
            b"Pageouts: 42.\n"
        )
        self.assertEqual(MODULE._parse_vm_stat(vm), (16384, 42))
        self.assertEqual(
            MODULE._parse_swap_used(b"total = 1.00G used = 1.25M free = 0.00M\n"),
            1310720,
        )
        timing = b" 123456 maximum resident set size\n 0 swaps\n"
        self.assertEqual(MODULE._parse_time_l(timing), (123456, 0))

    def test_live_measurement_builds_a_strict_bound_receipt_without_shell(self) -> None:
        measured = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=MODULE.canonical_bytes(self.fixture.generation_receipt) + b"\n",
            stderr=b" 65536 maximum resident set size\n 0 swaps\n",
        )
        with (
            mock.patch.object(MODULE.sys, "platform", "darwin"),
            mock.patch.object(MODULE, "_require_system_tool"),
            mock.patch.object(
                MODULE,
                "_mac_memory_snapshot",
                side_effect=[(4096, 50, 0), (4096, 50, 0)],
            ),
            mock.patch.object(MODULE, "_run_memory_command", return_value=measured) as run,
        ):
            receipt = MODULE._measure_memory_smoke(
                binary_path=self.fixture.binary_path,
                binary_sha=digest(self.fixture.binary_path.read_bytes()),
                model_dir=self.fixture.model_dir,
                profile=self.fixture.profile,
                generation_receipt_sha=digest(
                    self.fixture.generation_receipt_path.read_bytes()
                ),
            )
        command = run.call_args.args[0]
        self.assertEqual(command[0:2], ["/usr/bin/sandbox-exec", "-p"])
        self.assertIn("(deny network*)", command[2])
        self.assertIn("(deny file-write*)", command[2])
        self.assertEqual(command[3:6], ["/usr/bin/time", "-l", str(self.fixture.binary_path)])
        self.assertNotIsInstance(command, str)
        self.assertEqual(receipt["measurement"]["source"], "live")
        MODULE._validate_memory_receipt(
            receipt,
            self.fixture.profile,
            binary_path=self.fixture.binary_path,
            binary_sha=digest(self.fixture.binary_path.read_bytes()),
            model_dir=self.fixture.model_dir,
            generation_receipt_sha=digest(self.fixture.generation_receipt_path.read_bytes()),
            require_live=True,
        )

    def test_live_verify_rechecks_all_inputs_after_measurement(self) -> None:
        args = MODULE.parser().parse_args(self.fixture.command(measure=True)[2:])
        with mock.patch.object(
            MODULE,
            "_measure_memory_smoke",
            return_value=self.fixture.live_memory_receipt(),
        ):
            receipt, _ = MODULE.verify(args)
        self.assertIs(receipt["passed"], True)

    def test_live_verify_rejects_model_drift_during_measurement(self) -> None:
        args = MODULE.parser().parse_args(self.fixture.command(measure=True)[2:])

        def mutate_model(**_arguments: object) -> dict[str, object]:
            (self.fixture.model_dir / "tokenizer.json").write_bytes(b"changed")
            return self.fixture.live_memory_receipt()

        with (
            mock.patch.object(MODULE, "_measure_memory_smoke", side_effect=mutate_model),
            self.assertRaisesRegex(MODULE.DeploymentError, "tokenizer.json|model snapshot"),
        ):
            MODULE.verify(args)


class RealProfileTests(unittest.TestCase):
    def test_checked_in_qwen_profile_has_valid_strict_schema(self) -> None:
        profile = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
        validated = MODULE.validate_profile(profile)
        self.assertEqual(validated["source"]["repo_id"], "Qwen/Qwen3.5-0.8B")
        self.assertEqual(validated["runtime"]["matmul_feature"], "accelerate")
        self.assertEqual(validated["binary"]["size"], 8163904)
        self.assertEqual(
            validated["binary"]["sha256"],
            "d9cb4de44b236b5b3f216a81079b11102220939a2b179cbc2678442ff947803b",
        )
        self.assertEqual(
            validated["oracle"]["metrics_sha256"],
            "d440a40b9add739718ccc718b8bad39470d41fda5d2552c383cad115b544f91e",
        )
        self.assertEqual(
            validated["memory_smoke"]["non_authoritative_evidence"],
            ["pageout_delta_bytes", "swap_delta_bytes", "swap_growth_bytes"],
        )
        self.assertEqual(len(validated["gate"]["generated_token_ids"]), 10)


if __name__ == "__main__":
    unittest.main()
