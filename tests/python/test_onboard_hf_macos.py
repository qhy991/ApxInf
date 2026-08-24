from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "scripts/onboard_hf_macos.py"
SPEC = importlib.util.spec_from_file_location("onboard_hf_macos", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class Fixture:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.model_dir = root / "model"
        self.oracle_dir = root / "oracle"
        self.model_dir.mkdir()
        self.oracle_dir.mkdir()
        (self.oracle_dir / "manifest.json").write_text("{}\n", encoding="utf-8")
        (self.oracle_dir / "apxinf-metrics.json").write_text("{}\n", encoding="utf-8")
        self.binary = root / "apxinf"
        self.binary.write_bytes(b"fixture executable")
        self.binary.chmod(0o755)
        self.source_lock = root / "source-lock.json"
        self.source_lock.write_text("{}\n", encoding="utf-8")
        self.receipt = root / "generation-receipt.json"
        self.lock = root / "deployment-lock.json"

    def argv(self, *, dry_run: bool = False) -> list[str]:
        result = [
            MODULE.MODEL_URL,
            "--revision",
            MODULE.REVISION,
            "--source-lock",
            str(self.source_lock),
            "--model-dir",
            str(self.model_dir),
            "--oracle-dir",
            str(self.oracle_dir),
            "--binary",
            str(self.binary),
            "--receipt-output",
            str(self.receipt),
            "--lock-output",
            str(self.lock),
            "--offline",
        ]
        if dry_run:
            result.append("--dry-run")
        return result


def source_receipt() -> dict[str, object]:
    profile = MODULE._load_profile()
    return {
        "passed": True,
        "format": "apxinf-hf-source-lock-v1",
        "repo_id": MODULE.REPO_ID,
        "requested_revision": MODULE.REVISION,
        "resolved_commit": MODULE.REVISION,
        "content_sha256": profile["source"]["source_lock_content_sha256"],
        "metadata_bytes": 1,
        "weight_payload_bytes_downloaded": 0,
    }


def generation_result() -> dict[str, object]:
    profile = MODULE._load_profile()
    gate = profile["gate"]
    return {
        "build": {
            "target_os": "macos",
            "target_arch": "aarch64",
            "matmul_feature": "accelerate",
        },
        "format": MODULE.GENERATION_FORMAT,
        "model_type": "qwen3_5",
        "device": "cpu",
        "dtype": "fp32",
        "prompt_token_count": gate["prompt_token_count"],
        "generated_token_ids": gate["generated_token_ids"],
        "profile": {
            "input_tokens": gate["prompt_token_count"],
            "output_tokens": gate["max_tokens"],
            "ttft_ms": 10.0,
            "tpot_ms": 2.0,
            "generation_tps": 500.0,
            "total_latency_ms": 28.0,
        },
    }


def stager_receipt(model_dir: Path, *, action: str = "staged") -> dict[str, object]:
    profile = MODULE._load_profile()
    artifacts = MODULE._profile_artifacts(profile)
    total = sum(record["size"] for record in artifacts)
    existing = action == "reused-existing"
    return {
        "format": MODULE.STAGER_FORMAT,
        "passed": True,
        "action": action,
        "profile_id": MODULE.PROFILE_ID,
        "repo_id": MODULE.REPO_ID,
        "resolved_commit": MODULE.REVISION,
        "source_lock_content_sha256": profile["source"]["source_lock_content_sha256"],
        "artifact_manifest_sha256": MODULE._artifact_manifest_sha256(profile),
        "model_dir": str(model_dir),
        "artifacts": artifacts,
        "total_bytes": total,
        "published": True,
        "downloaded_bytes": 0 if existing else total,
        "resumed_from_bytes": 0,
        "reused_bytes": total if existing else 0,
        "policy": {
            "network": {
                "https_only": True,
                "approved_domain_suffixes": ["huggingface.co", "hf.co"],
                "ambient_proxy_forbidden": True,
                "authorization_forbidden": True,
                "remote_code_forbidden": True,
                "transfer_encoding_forbidden": True,
            },
            "filesystem": {
                "trust_boundary": "same-uid-local-filesystem-v1",
                "concurrency": "cooperative-adjacent-flock-v1",
                "atomic_publish": "macos-renamex-noreplace-v1",
            },
            "operation": {"existing_only_requested": existing},
            "recovery": {"max_restart_from_zero_per_artifact": 1},
        },
        "evidence": {
            "builtin_opener": not existing,
            "opener_injected": False,
            "ambient_proxy_disabled": not existing,
            "authorization_header_omitted": not existing,
            "lock_acquired": True,
            "network_used": not existing,
            "network_request_count": 0 if existing else 1,
            "existing_bundle_verified": existing,
            "existing_only_enforced": existing,
            "cache_tree_present": False,
            "cache_entry_count": 0,
            "cache_total_bytes": 0,
            "published_by_this_invocation": not existing,
            "atomic_no_replace_publish_observed": not existing,
            "recovered_artifacts": [],
            "recovery_bytes_discarded": 0,
        },
    }


def completed(argv: list[str], payload: object) -> subprocess.CompletedProcess[bytes]:
    encoded = MODULE._canonical_bytes(payload) + b"\n"
    return subprocess.CompletedProcess(argv, 0, stdout=encoded, stderr=b"")


class OnboardControllerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.fixture = Fixture(Path(self.temporary.name))

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def invoke(self, argv: list[str]) -> tuple[int, str, str]:
        stdout = io.StringIO()
        stderr = io.StringIO()
        with (
            mock.patch.object(MODULE, "_verify_pinned_binary"),
            mock.patch.object(MODULE, "_validate_existing_source_lock"),
            mock.patch.object(MODULE, "_validate_oracle_provenance"),
            redirect_stdout(stdout),
            redirect_stderr(stderr),
        ):
            status = MODULE.main(argv)
        return status, stdout.getvalue(), stderr.getvalue()

    def test_dry_run_emits_exact_stages_without_executing(self) -> None:
        with mock.patch.object(MODULE.subprocess, "run") as run:
            status, stdout, stderr = self.invoke(self.fixture.argv(dry_run=True))

        self.assertEqual(status, 0)
        self.assertEqual(stderr, "")
        self.assertEqual(len(stdout.splitlines()), 1)
        run.assert_not_called()
        plan = json.loads(stdout)
        self.assertEqual(plan["format"], MODULE.PLAN_FORMAT)
        self.assertTrue(plan["dry_run"])
        self.assertNotIn("existing_bundle_only", plan)
        self.assertNotIn("downloads_weights", plan)
        self.assertEqual(plan["bundle"]["disposition"], "reused")
        self.assertEqual(
            plan["bundle"]["stager_receipt"]["expected_action"],
            "reused-existing",
        )
        self.assertEqual(
            plan["bundle"]["stager_receipt"]["resolved_commit"], MODULE.REVISION
        )
        self.assertEqual(
            plan["bundle"]["stager_receipt"]["total_bytes"],
            plan["bundle"]["total_bytes"],
        )
        self.assertEqual(plan["source_lock"]["disposition"], "reused")
        self.assertFalse(plan["starts_agent"])
        self.assertEqual(
            [stage["name"] for stage in plan["stages"]],
            [
                "verify_source_lock",
                "ensure_model_bundle",
                "run_generation_gate",
                "publish_generation_receipt",
                "verify_and_publish_deployment",
            ],
        )
        ensure = plan["stages"][1]
        self.assertEqual(ensure["network_policy"], "offline")
        self.assertEqual(ensure["timeout_seconds"], MODULE.BUNDLE_STAGE_TIMEOUT)
        self.assertIn("--existing-only", ensure["argv"])
        self.assertTrue(plan["bundle"]["stager_receipt"]["existing_only"])
        generation = plan["stages"][2]
        self.assertEqual(generation["argv"][-1], "--json")
        self.assertEqual(generation["argv"][0], "/usr/bin/sandbox-exec")
        self.assertEqual(generation["argv"][1], "-p")
        self.assertIn("(deny network*)", generation["argv"][2])
        self.assertIn("(deny file-write*)", generation["argv"][2])
        self.assertIn(str(self.fixture.binary), generation["argv"])
        self.assertIn("--no-eos-stop", generation["argv"])
        self.assertEqual(generation["network_policy"], "offline")
        self.assertNotIn("HF_TOKEN", generation["environment_keys"])
        self.assertIn("--measure-smoke", plan["stages"][-1]["argv"])
        self.assertNotIn("kersor", json.dumps(plan).casefold())

    def test_online_dry_run_starts_with_metadata_only_resolution(self) -> None:
        self.fixture.source_lock.unlink()
        argv = self.fixture.argv(dry_run=True)
        argv.remove("--offline")
        with mock.patch.object(MODULE.subprocess, "run") as run:
            status, stdout, stderr = self.invoke(argv)

        self.assertEqual(status, 0)
        self.assertEqual(stderr, "")
        run.assert_not_called()
        plan = json.loads(stdout)
        resolution = plan["stages"][0]
        self.assertEqual(resolution["name"], "resolve_source_lock")
        self.assertEqual(
            resolution["network_policy"],
            "metadata-only-https-huggingface.co",
        )
        self.assertEqual(resolution["argv"][2], MODULE.MODEL_URL)
        self.assertIn("--output", resolution["argv"])

    def test_noncanonical_url_is_a_single_json_error(self) -> None:
        argv = self.fixture.argv(dry_run=True)
        argv[0] = MODULE.MODEL_URL + "/"
        with mock.patch.object(MODULE.subprocess, "run") as run:
            status, stdout, stderr = self.invoke(argv)

        self.assertNotEqual(status, 0)
        self.assertEqual(stdout, "")
        self.assertEqual(len(stderr.splitlines()), 1)
        run.assert_not_called()
        error = json.loads(stderr)
        self.assertFalse(error["passed"])
        self.assertIn("canonical URL", error["error"]["message"])

    def test_generation_result_requires_locked_build(self) -> None:
        profile = MODULE._load_profile()
        receipt = MODULE._generation_receipt(generation_result(), profile)
        self.assertEqual(
            receipt["build"],
            {
                "target_os": "macos",
                "target_arch": "aarch64",
                "matmul_feature": "accelerate",
            },
        )

        tampered = generation_result()
        tampered["generated_token_ids"] = [1]
        with self.assertRaises(MODULE.OnboardError):
            MODULE._generation_receipt(tampered, profile)
        extra = generation_result()
        extra["unexpected"] = True
        with self.assertRaises(MODULE.OnboardError):
            MODULE._generation_receipt(extra, profile)
        missing_build = generation_result()
        del missing_build["build"]
        with self.assertRaises(MODULE.OnboardError):
            MODULE._generation_receipt(missing_build, profile)

    def test_binary_identity_is_checked_before_execution(self) -> None:
        profile = MODULE._load_profile()
        with self.assertRaises(MODULE.OnboardError):
            MODULE._verify_pinned_binary(self.fixture.binary, profile)

    def test_subprocess_contract_uses_argv_timeout_and_clean_environment(self) -> None:
        stage = MODULE.Stage("fixture", ("/bin/tool", "--json"), 17, "offline")
        child = completed(list(stage.argv), {"passed": True})
        with mock.patch.object(MODULE.subprocess, "run", return_value=child) as run:
            result = MODULE._run_json_stage(stage)

        self.assertTrue(result["passed"])
        _, kwargs = run.call_args
        self.assertIs(kwargs["shell"], False)
        self.assertEqual(kwargs["timeout"], 17)
        self.assertEqual(kwargs["stdin"], subprocess.DEVNULL)
        self.assertEqual(kwargs["stdout"], subprocess.PIPE)
        self.assertEqual(kwargs["stderr"], subprocess.PIPE)
        self.assertNotIn("HF_TOKEN", kwargs["env"])
        self.assertNotIn("HUGGINGFACE_HUB_TOKEN", kwargs["env"])
        self.assertEqual(kwargs["env"]["HF_HUB_OFFLINE"], "1")

    def test_offline_existing_bundle_publishes_receipt_and_lock(self) -> None:
        calls: list[list[str]] = []

        def run(
            argv: list[str], **kwargs: object
        ) -> subprocess.CompletedProcess[bytes]:
            del kwargs
            calls.append(argv)
            if len(argv) > 1 and argv[1] == str(MODULE.RESOLVER):
                return completed(argv, source_receipt())
            if len(argv) > 1 and argv[1] == str(MODULE.STAGER):
                return completed(
                    argv,
                    stager_receipt(self.fixture.model_dir, action="reused-existing"),
                )
            if argv[0] == str(MODULE.SANDBOX_EXEC):
                return completed(argv, generation_result())
            if len(argv) > 1 and argv[1] == str(MODULE.DEPLOYMENT_VERIFIER):
                memory_hash = "a" * 64
                body: dict[str, object] = {
                    "format": MODULE.DEPLOYMENT_FORMAT,
                    "fixture": True,
                    "memory_smoke": {
                        "origin": "live",
                        "content_sha256": memory_hash,
                    },
                }
                body["content_sha256"] = MODULE._sha256(MODULE._canonical_bytes(body))
                self.fixture.lock.write_text(
                    json.dumps(body, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                return completed(
                    argv,
                    {
                        "format": MODULE.DEPLOYMENT_RECEIPT_FORMAT,
                        "passed": True,
                        "profile_id": MODULE.PROFILE_ID,
                        "repo_id": MODULE.REPO_ID,
                        "resolved_commit": MODULE.REVISION,
                        "deployment_lock_sha256": body["content_sha256"],
                        "memory_smoke": {
                            "present": True,
                            "origin": "live",
                            "content_sha256": memory_hash,
                        },
                        "output": str(self.fixture.lock),
                    },
                )
            raise AssertionError(f"unexpected subprocess: {argv}")

        with mock.patch.object(MODULE.subprocess, "run", side_effect=run):
            status, stdout, stderr = self.invoke(self.fixture.argv())

        self.assertEqual(status, 0, stderr)
        self.assertEqual(stderr, "")
        self.assertEqual(len(stdout.splitlines()), 1)
        self.assertEqual(len(calls), 4)
        result = json.loads(stdout)
        self.assertTrue(result["passed"])
        self.assertEqual(result["bundle"]["disposition"], "reused")
        self.assertEqual(result["bundle"]["bytes"]["downloaded"], 0)
        self.assertEqual(
            result["bundle"]["bytes"]["reused"], result["bundle"]["total_bytes"]
        )
        self.assertEqual(
            result["bundle"]["stager_receipt"]["action"], "reused-existing"
        )
        self.assertEqual(
            result["bundle"]["stager_receipt"]["source_lock_content_sha256"],
            result["source_lock"]["content_sha256"],
        )
        self.assertTrue(
            result["bundle"]["stager_receipt"]["evidence"]["existing_bundle_verified"]
        )
        self.assertEqual(result["source_lock"]["disposition"], "reused")
        self.assertNotIn("existing_bundle_only", result)
        self.assertNotIn("downloads_weights", result)
        self.assertEqual(result["deployment_lock"]["path"], str(self.fixture.lock))
        receipt = json.loads(self.fixture.receipt.read_text(encoding="utf-8"))
        self.assertEqual(
            receipt["generated_token_ids"], generation_result()["generated_token_ids"]
        )
        self.assertEqual(stat.S_IMODE(self.fixture.receipt.stat().st_mode), 0o600)

        receipt_before = self.fixture.receipt.read_bytes()
        lock_before = self.fixture.lock.read_bytes()
        with mock.patch.object(MODULE.subprocess, "run") as rerun:
            status, stdout, stderr = self.invoke(self.fixture.argv())
        self.assertNotEqual(status, 0)
        self.assertEqual(stdout, "")
        self.assertIn("overwrite", json.loads(stderr)["error"]["message"])
        rerun.assert_not_called()
        self.assertEqual(self.fixture.receipt.read_bytes(), receipt_before)
        self.assertEqual(self.fixture.lock.read_bytes(), lock_before)

    def test_online_existing_source_lock_is_verified_and_reused(self) -> None:
        argv = self.fixture.argv(dry_run=True)
        argv.remove("--offline")
        with mock.patch.object(MODULE.subprocess, "run") as run:
            status, stdout, stderr = self.invoke(argv)

        self.assertEqual(status, 0, stderr)
        self.assertEqual(stderr, "")
        run.assert_not_called()
        plan = json.loads(stdout)
        self.assertEqual(plan["source_lock"]["disposition"], "reused")
        names = [stage["name"] for stage in plan["stages"]]
        self.assertEqual(names[0], "verify_source_lock")
        self.assertNotIn("resolve_source_lock", names)

    def test_existing_source_lock_must_be_exact_before_reuse(self) -> None:
        body: dict[str, object] = {
            "format": "apxinf-hf-source-lock-v1",
            "repo_id": MODULE.REPO_ID,
            "requested_revision": MODULE.REVISION,
            "resolved_commit": MODULE.REVISION,
            "policy_receipt": {
                "metadata_only": True,
                "weight_payload_bytes_downloaded": 0,
                "remote_code_executed": False,
                "hf_token_read": False,
            },
        }
        digest = MODULE._sha256(MODULE._canonical_bytes(body))
        lock = {**body, "content_sha256": digest}
        self.fixture.source_lock.write_text(
            json.dumps(lock, sort_keys=True) + "\n", encoding="utf-8"
        )
        profile = MODULE._load_profile()
        profile["source"] = dict(profile["source"])
        profile["source"]["source_lock_content_sha256"] = digest
        verified = MODULE._validate_existing_source_lock(
            self.fixture.source_lock, profile
        )
        self.assertEqual(verified["content_sha256"], digest)

        lock["repo_id"] = "attacker/model"
        self.fixture.source_lock.write_text(
            json.dumps(lock, sort_keys=True) + "\n", encoding="utf-8"
        )
        with self.assertRaisesRegex(MODULE.OnboardError, "hash mismatch"):
            MODULE._validate_existing_source_lock(self.fixture.source_lock, profile)

    def test_oracle_provenance_does_not_bind_relocated_target(self) -> None:
        manifest = self.fixture.oracle_dir / "manifest.json"
        manifest.write_text(
            json.dumps({"model_dir": str(self.fixture.model_dir)}) + "\n",
            encoding="utf-8",
        )
        MODULE._validate_oracle_provenance(manifest)
        # The path is historical provenance. A fresh target is identified by the
        # profile and per-artifact hashes, so relocation must not be rejected.
        relocated = self.fixture.root / "different-model"
        self.assertNotEqual(relocated, self.fixture.model_dir)
        MODULE._validate_oracle_provenance(manifest)
        manifest.write_text('{"model_dir":"relative/model"}\n', encoding="utf-8")
        with self.assertRaisesRegex(MODULE.OnboardError, "provenance"):
            MODULE._validate_oracle_provenance(manifest)

    def test_missing_model_requires_online_download_authority(self) -> None:
        self.fixture.model_dir.rmdir()
        online = self.fixture.argv(dry_run=True)
        online.remove("--offline")
        with mock.patch.object(MODULE.subprocess, "run") as run:
            status, stdout, stderr = self.invoke(online)
        self.assertNotEqual(status, 0)
        self.assertEqual(stdout, "")
        self.assertIn("--download-missing", json.loads(stderr)["error"]["message"])
        run.assert_not_called()

        offline = self.fixture.argv(dry_run=True)
        offline.append("--download-missing")
        with mock.patch.object(MODULE.subprocess, "run") as run:
            status, stdout, stderr = self.invoke(offline)
        self.assertNotEqual(status, 0)
        self.assertEqual(stdout, "")
        self.assertIn("offline", json.loads(stderr)["error"]["message"])
        run.assert_not_called()

    def test_outputs_cannot_overlap_stager_work_paths(self) -> None:
        argv = self.fixture.argv(dry_run=True)
        receipt_index = argv.index("--receipt-output") + 1
        argv[receipt_index] = str(
            self.fixture.model_dir.parent
            / f".{self.fixture.model_dir.name}.apxinf-stage.lock"
        )
        with mock.patch.object(MODULE.subprocess, "run") as run:
            status, stdout, stderr = self.invoke(argv)
        self.assertNotEqual(status, 0)
        self.assertEqual(stdout, "")
        self.assertIn("stager work", json.loads(stderr)["error"]["message"])
        run.assert_not_called()

    def test_missing_model_dry_run_plans_exact_stager_without_writes(self) -> None:
        self.fixture.model_dir.rmdir()
        self.fixture.source_lock.unlink()
        argv = self.fixture.argv(dry_run=True)
        argv.remove("--offline")
        argv.append("--download-missing")
        with mock.patch.object(MODULE.subprocess, "run") as run:
            status, stdout, stderr = self.invoke(argv)

        self.assertEqual(status, 0, stderr)
        self.assertEqual(stderr, "")
        run.assert_not_called()
        self.assertFalse(self.fixture.model_dir.exists())
        self.assertFalse(self.fixture.source_lock.exists())
        plan = json.loads(stdout)
        self.assertEqual(plan["source_lock"]["disposition"], "created")
        self.assertEqual(plan["bundle"]["disposition"], "staged")
        self.assertEqual(
            plan["bundle"]["stager_receipt"]["format"], MODULE.STAGER_FORMAT
        )
        names = [stage["name"] for stage in plan["stages"]]
        self.assertEqual(
            names,
            [
                "resolve_source_lock",
                "verify_source_lock",
                "ensure_model_bundle",
                "run_generation_gate",
                "publish_generation_receipt",
                "verify_and_publish_deployment",
            ],
        )
        stage = plan["stages"][2]
        self.assertEqual(stage["argv"][1], str(MODULE.STAGER))
        self.assertNotIn("--existing-only", stage["argv"])
        self.assertFalse(plan["bundle"]["stager_receipt"]["existing_only"])
        self.assertIn("--profile", stage["argv"])
        self.assertIn("--source-lock", stage["argv"])
        self.assertIn("--model-dir", stage["argv"])
        self.assertIn(str(MODULE.BUNDLE_STAGE_TIMEOUT), stage["argv"])
        self.assertIn(str(MODULE.STAGER_MAX_TOTAL_BYTES), stage["argv"])
        self.assertEqual(stage["timeout_seconds"], 7200)

    def test_downloaded_bundle_receipt_is_strictly_bound_before_generation(
        self,
    ) -> None:
        self.fixture.model_dir.rmdir()
        argv = self.fixture.argv()
        argv.remove("--offline")
        argv.append("--download-missing")
        calls: list[list[str]] = []

        def run(
            argv: list[str], **kwargs: object
        ) -> subprocess.CompletedProcess[bytes]:
            del kwargs
            calls.append(argv)
            if len(argv) > 1 and argv[1] == str(MODULE.RESOLVER):
                return completed(argv, source_receipt())
            if len(argv) > 1 and argv[1] == str(MODULE.STAGER):
                return completed(argv, stager_receipt(self.fixture.model_dir))
            if argv[0] == str(MODULE.SANDBOX_EXEC):
                return completed(argv, generation_result())
            if len(argv) > 1 and argv[1] == str(MODULE.DEPLOYMENT_VERIFIER):
                memory_hash = "a" * 64
                body: dict[str, object] = {
                    "format": MODULE.DEPLOYMENT_FORMAT,
                    "fixture": True,
                    "memory_smoke": {
                        "origin": "live",
                        "content_sha256": memory_hash,
                    },
                }
                body["content_sha256"] = MODULE._sha256(MODULE._canonical_bytes(body))
                self.fixture.lock.write_text(
                    json.dumps(body, sort_keys=True) + "\n", encoding="utf-8"
                )
                return completed(
                    argv,
                    {
                        "format": MODULE.DEPLOYMENT_RECEIPT_FORMAT,
                        "passed": True,
                        "profile_id": MODULE.PROFILE_ID,
                        "repo_id": MODULE.REPO_ID,
                        "resolved_commit": MODULE.REVISION,
                        "deployment_lock_sha256": body["content_sha256"],
                        "memory_smoke": {
                            "present": True,
                            "origin": "live",
                            "content_sha256": memory_hash,
                        },
                        "output": str(self.fixture.lock),
                    },
                )
            raise AssertionError(f"unexpected subprocess: {argv}")

        with mock.patch.object(MODULE.subprocess, "run", side_effect=run):
            status, stdout, stderr = self.invoke(argv)

        self.assertEqual(status, 0, stderr)
        self.assertEqual(stderr, "")
        self.assertEqual(
            [
                "resolver"
                if call[1] == str(MODULE.RESOLVER)
                else "stager"
                if len(call) > 1 and call[1] == str(MODULE.STAGER)
                else "generation"
                if call[0] == str(MODULE.SANDBOX_EXEC)
                else "deployment"
                for call in calls
            ],
            ["resolver", "stager", "generation", "deployment"],
        )
        result = json.loads(stdout)
        bundle = result["bundle"]
        self.assertEqual(bundle["disposition"], "staged")
        self.assertEqual(bundle["bytes"]["downloaded"], bundle["total_bytes"])
        self.assertEqual(bundle["bytes"]["resumed"], 0)
        self.assertEqual(bundle["bytes"]["reused"], 0)
        self.assertEqual(bundle["bytes"]["recovery_discarded"], 0)
        self.assertEqual(bundle["stager_receipt"]["format"], MODULE.STAGER_FORMAT)
        self.assertRegex(
            bundle["stager_receipt"]["canonical_sha256"], r"^[0-9a-f]{64}$"
        )
        self.assertEqual(result["source_lock"]["disposition"], "reused")

    def test_tampered_stager_receipt_stops_before_binary_execution(self) -> None:
        self.fixture.model_dir.rmdir()
        argv = self.fixture.argv()
        argv.remove("--offline")
        argv.append("--download-missing")
        calls: list[list[str]] = []

        def run(
            arguments: list[str], **kwargs: object
        ) -> subprocess.CompletedProcess[bytes]:
            del kwargs
            calls.append(arguments)
            if len(arguments) > 1 and arguments[1] == str(MODULE.RESOLVER):
                return completed(arguments, source_receipt())
            if len(arguments) > 1 and arguments[1] == str(MODULE.STAGER):
                receipt = stager_receipt(self.fixture.model_dir)
                receipt["unexpected"] = True
                return completed(arguments, receipt)
            raise AssertionError(
                "binary or deployment verifier ran after a bad stager receipt"
            )

        with mock.patch.object(MODULE.subprocess, "run", side_effect=run):
            status, stdout, stderr = self.invoke(argv)

        self.assertNotEqual(status, 0)
        self.assertEqual(stdout, "")
        self.assertIn("missing or unknown", json.loads(stderr)["error"]["message"])
        self.assertEqual(len(calls), 2)

    def test_existing_bundle_stager_failure_stops_before_binary_execution(self) -> None:
        calls: list[list[str]] = []

        def run(
            arguments: list[str], **kwargs: object
        ) -> subprocess.CompletedProcess[bytes]:
            del kwargs
            calls.append(arguments)
            if len(arguments) > 1 and arguments[1] == str(MODULE.RESOLVER):
                return completed(arguments, source_receipt())
            if len(arguments) > 1 and arguments[1] == str(MODULE.STAGER):
                self.assertIn("--existing-only", arguments)
                payload = {
                    "format": MODULE.STAGER_FORMAT,
                    "passed": False,
                    "error": {
                        "code": "BUNDLE_INVALID",
                        "message": "top-level allowlist mismatch",
                    },
                }
                encoded = MODULE._canonical_bytes(payload) + b"\n"
                return subprocess.CompletedProcess(
                    arguments, 2, stdout=encoded, stderr=b""
                )
            raise AssertionError("generation ran after stager rejected the bundle")

        with mock.patch.object(MODULE.subprocess, "run", side_effect=run):
            status, stdout, stderr = self.invoke(self.fixture.argv())

        self.assertNotEqual(status, 0)
        self.assertEqual(stdout, "")
        self.assertIn("top-level allowlist", json.loads(stderr)["error"]["message"])
        self.assertEqual(len(calls), 2)

    def test_existing_stager_receipt_accepts_safe_cache_and_rejects_network(
        self,
    ) -> None:
        profile = MODULE._load_profile()
        receipt = stager_receipt(self.fixture.model_dir, action="reused-existing")
        receipt["evidence"]["cache_tree_present"] = True
        receipt["evidence"]["cache_entry_count"] = 18
        receipt["evidence"]["cache_total_bytes"] = 2657
        verified = MODULE._validate_stager_receipt(
            receipt,
            profile,
            self.fixture.model_dir,
            expected_action="reused-existing",
        )
        self.assertTrue(verified["evidence"]["cache_tree_present"])

        receipt["evidence"]["network_used"] = True
        receipt["evidence"]["network_request_count"] = 1
        with self.assertRaisesRegex(MODULE.OnboardError, "existing bundle"):
            MODULE._validate_stager_receipt(
                receipt,
                profile,
                self.fixture.model_dir,
                expected_action="reused-existing",
            )

    def test_staged_receipt_accepts_one_safe_restart_per_artifact(self) -> None:
        profile = MODULE._load_profile()
        receipt = stager_receipt(self.fixture.model_dir)
        total = receipt["total_bytes"]
        receipt["downloaded_bytes"] = 2 * total
        receipt["evidence"]["network_request_count"] = 2 * len(receipt["artifacts"])
        receipt["evidence"]["recovered_artifacts"] = sorted(
            artifact["path"] for artifact in receipt["artifacts"]
        )
        receipt["evidence"]["recovery_bytes_discarded"] = total
        verified = MODULE._validate_stager_receipt(
            receipt,
            profile,
            self.fixture.model_dir,
            expected_action="staged",
        )
        self.assertEqual(verified["downloaded_bytes"], 2 * total)
        self.assertEqual(verified["evidence"]["recovery_bytes_discarded"], total)

        receipt["downloaded_bytes"] += 1
        with self.assertRaisesRegex(MODULE.OnboardError, "byte accounting"):
            MODULE._validate_stager_receipt(
                receipt,
                profile,
                self.fixture.model_dir,
                expected_action="staged",
            )

    def test_existing_lock_is_rejected_before_any_subprocess(self) -> None:
        self.fixture.lock.write_text("do not overwrite\n", encoding="utf-8")
        with mock.patch.object(MODULE.subprocess, "run") as run:
            status, stdout, stderr = self.invoke(self.fixture.argv())

        self.assertNotEqual(status, 0)
        self.assertEqual(stdout, "")
        self.assertIn("overwrite", json.loads(stderr)["error"]["message"])
        run.assert_not_called()
        self.assertEqual(
            self.fixture.lock.read_text(encoding="utf-8"), "do not overwrite\n"
        )


if __name__ == "__main__":
    unittest.main()
