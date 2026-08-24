from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
LOCK_MODULE_PATH = ROOT / "scripts/kersor_runtime_lock.py"
LAUNCHER_PATH = ROOT / "scripts/run_locked_kersor_mission.py"
PREPARE_MODULE_PATH = ROOT / "scripts/prepare_hf_macos_intake.py"
SPEC = importlib.util.spec_from_file_location("kersor_runtime_lock", LOCK_MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
LOCK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(LOCK)
PREPARE_SPEC = importlib.util.spec_from_file_location(
    "prepare_hf_macos_intake_for_runtime_lock_test", PREPARE_MODULE_PATH
)
assert PREPARE_SPEC is not None and PREPARE_SPEC.loader is not None
PREPARE = importlib.util.module_from_spec(PREPARE_SPEC)
PREPARE_SPEC.loader.exec_module(PREPARE)


def write(path: Path, payload: str | bytes, *, executable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(payload, bytes):
        path.write_bytes(payload)
    else:
        path.write_text(payload, encoding="utf-8")
    if executable:
        path.chmod(0o755)


FAKE_EVOLVE = (
    r"""#!/usr/bin/env python3
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import sys

def raw_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def dump(path, value, mode=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    if mode is not None: path.chmod(mode)

root = Path(__file__).resolve().parents[1]
mission_path = Path(sys.argv[1]).resolve()
mission = json.loads(mission_path.read_text(encoding="utf-8"))
run_dir = Path(sys.argv[sys.argv.index("--run-dir") + 1]).resolve()
runtime_path = Path(sys.argv[sys.argv.index("--runtime-config") + 1]).resolve()
capture = Path.cwd() / "fake-evolve-capture.json"
history = json.loads(capture.read_text()) if capture.exists() else []
history.append({"argv": sys.argv, "environment": dict(os.environ)})
capture.write_text(json.dumps(history, sort_keys=True), encoding="utf-8")

if "--admit-only" in sys.argv:
    run_dir.mkdir(parents=True, exist_ok=False)
    dump(run_dir / "mission.json", mission)
    shutil.copyfile(runtime_path, run_dir / "runtime-config.json")
    shutil.copyfile(root / "runtime/autonomous-controller.js", run_dir / "controller.js")
    (run_dir / "controller.js").chmod((root / "runtime/autonomous-controller.js").stat().st_mode & 0o7777)
    session = Path(mission["session"])
    config = json.loads((session / "session-config.json").read_text())
    state = json.loads((session / "state.json").read_text())
    dump(run_dir / "session-snapshot.json", {"schema_version": 1, "config": config, "state": state})
    paths = """
    + repr(list(LOCK.EXECUTOR_RUNTIME_PATHS))
    + r"""
    files = []
    for relative in paths:
        source = root / relative
        target = run_dir / "executor-runtime" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        mode = source.stat().st_mode & 0o7777
        target.chmod(mode)
        files.append({"path": relative, "sha256": raw_sha(source), "size_bytes": source.stat().st_size, "mode": f"{mode:04o}"})
    manifest_path = run_dir / "executor-runtime-manifest.json"
    dump(manifest_path, {"schema_version": 1, "root": "executor-runtime", "files": files}, 0o600)
    executor = {"schema_version": 1, "root": "executor-runtime", "manifest": "executor-runtime-manifest.json", "manifest_sha256": raw_sha(manifest_path), "manifest_mode": "0600", "dispatch": "scripts/dispatch-workflow.sh", "workflow_host": "runtime/workflow-host.mjs"}
    frozen_mission = run_dir / "mission.json"
    frozen_runtime = run_dir / "runtime-config.json"
    controller = run_dir / "controller.js"
    snapshot = run_dir / "session-snapshot.json"
    pretty = lambda value: hashlib.sha256((json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n").encode()).hexdigest()
    binding = {
        "schema_version": 1, "run_id": mission["mission"]["mission_id"],
        "created_at": "2026-08-24T00:00:00+00:00", "session_dir": str(session),
        "session_id": state["session_id"], "session_schema_version": state["schema_version"],
        "session_config_sha256": pretty(config), "session_state_sha256": pretty(state),
        "session_snapshot_sha256": raw_sha(snapshot), "source_mission_sha256": raw_sha(mission_path),
        "mission_sha256": raw_sha(frozen_mission), "controller_path": "controller.js",
        "controller_sha256": raw_sha(controller), "controller_size_bytes": controller.stat().st_size,
        "controller_mode": f"{controller.stat().st_mode & 0o7777:04o}",
        "source_runtime_config_sha256": raw_sha(runtime_path), "runtime_config_sha256": raw_sha(frozen_runtime),
        "executor_runtime": executor, "runtime": "codex", "project_root": str(Path.cwd().resolve()),
    }
    binding_path = run_dir / "binding.json"
    dump(binding_path, binding)
    receipt = {
        "contract_version": "kersor-autonomous-admission-v1", "status": "admitted", "revision": 0,
        "run_dir": str(run_dir), "binding_sha256": raw_sha(binding_path),
        "source_mission_sha256": raw_sha(mission_path), "mission_sha256": raw_sha(frozen_mission),
        "controller_sha256": raw_sha(controller), "source_runtime_config_sha256": raw_sha(runtime_path),
        "runtime_config_sha256": raw_sha(frozen_runtime), "session_snapshot_sha256": raw_sha(snapshot),
        "executor_runtime_manifest_sha256": raw_sha(manifest_path),
    }
    if mission.get("_test_bad_admission"):
        receipt["binding_sha256"] = "0" * 64
    rendered = json.dumps(receipt, separators=(",", ":"), sort_keys=True)
    if mission.get("_test_duplicate_admission_key"):
        rendered = rendered[:-1] + ',"status":"admitted"}'
    print(rendered)
    raise SystemExit(0)

if "--resume" not in sys.argv:
    raise SystemExit(91)
(run_dir / ".runtime").mkdir(exist_ok=True)
if mission.get("_test_mutate_runtime"):
    target = root / "runtime/autonomous-controller.js"
    target.write_text(target.read_text() + "// drift\n")
raise SystemExit(int(mission.get("_test_exit_code", 0)))
"""
)


class FakeRuntime:
    def __init__(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name).resolve()
        self.kersor = self.base / "kersor"
        self.workspace = self.base / "workspace"
        self.session = self.base / "session"
        self.runtime_config = self.base / "runtime.json"
        self.mission = self.base / "mission.json"
        self.codex = self.base / "bin/codex-native"
        self.node = self.base / "bin/node"
        self.auth_home = self.base / "auth-home"
        self.source_lock = self.workspace / ".apxinf/source-lock.json"
        self.lock_path = self.base / "runtime.lock.json"
        self.capture = self.workspace / "fake-evolve-capture.json"
        self.workspace.mkdir()
        self.session.mkdir()
        self.auth_home.mkdir(mode=0o700)
        write(
            self.auth_home / "auth.json", '{"token":"AUTH-CONTENT-MUST-NOT-BE-READ"}\n'
        )
        (self.auth_home / "auth.json").chmod(0o600)
        self._build_workspace()
        self._build_kersor()
        self._build_commands()
        self._write_contract()
        self._init_git()

    def cleanup(self) -> None:
        self.temporary.cleanup()

    def _build_workspace(self) -> None:
        for relative in LOCK.APXINF_RUNTIME_PATHS:
            source = ROOT / relative
            target = self.workspace / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        write(self.workspace / "scripts/source_validator.py", "import json\n")
        write(self.workspace / "scripts/manifest_validator.py", "import pathlib\n")
        write(self.workspace / "scripts/resolve_hf_source.py", "import json\n")
        write(
            self.workspace / "scripts/validate_hf_port_manifest.py",
            "import pathlib\n",
        )
        write(
            self.workspace / "configs/hf-onboarding/qwen35-0.8b-macos-cpu.json",
            '{"profile":"fixture"}\n',
        )
        body = {
            "format": "apxinf-hf-source-lock-v1",
            "repo_id": "org/model",
            "requested_revision": "main",
            "resolved_commit": "a" * 40,
        }
        body["content_sha256"] = LOCK.semantic_json_sha256(body)
        write(self.source_lock, json.dumps(body, sort_keys=True) + "\n")

    def _build_kersor(self) -> None:
        write(
            self.kersor / ".codex-plugin/plugin.json",
            json.dumps({"name": "kersor", "version": "9.9.0-test"}),
        )
        for relative in LOCK.EXECUTOR_RUNTIME_PATHS:
            content = "{}\n" if relative.endswith(".json") else f"// {relative}\n"
            if relative.endswith(".py"):
                content = "# fixture\n"
            if relative.endswith(".sh"):
                content = "#!/bin/sh\nexit 0\n"
            write(self.kersor / relative, content, executable=relative.endswith(".sh"))
        write(
            self.kersor / "runtime/autonomous-controller.js",
            "export const controller = 1\n",
        )
        write(
            self.kersor / "runtime/brokers/codex-exec.mjs",
            "export const CODEX_AUTH_CUSTODY_MECHANISM='codex-named-permissions-auth-read-deny-v2'\n"
            "export const CODEX_COMMAND_READ_SCOPE_MECHANISM='codex-minimal-project-read-v1'\n"
            "export const CODEX_AUTH_PERMISSION_PROFILE='kersor_auth_custody'\n"
            "export const sandboxMechanism='codex-named-permissions-profile-v1'\n",
        )
        write(self.kersor / "kersor_core/__init__.py", "VERSION = 1\n")
        write(self.kersor / "kersor_core/cli.py", "def main(): return 0\n")
        write(self.kersor / "kersor_core/session.py", "class SessionStore: pass\n")
        write(self.kersor / "config/default_config.json", "{}\n")
        write(
            self.kersor / "docs/autonomous-workflow-runtime.md",
            "# Test autonomous runtime\n",
        )
        write(self.kersor / "skills/kersor-evolve/SKILL.md", "# evolve\n")
        write(self.kersor / "skills/kersor-protocol/SKILL.md", "# protocol\n")
        write(self.kersor / "scripts/evolve.sh", FAKE_EVOLVE, executable=True)
        write(
            self.kersor / "scripts/run-autonomous-workflow.py",
            "#!/usr/bin/env python3\n",
            executable=True,
        )
        write(
            self.kersor / "scripts/verify-autonomous-run.py",
            "#!/usr/bin/env python3\n",
            executable=True,
        )

    def _build_commands(self) -> None:
        write(
            self.codex,
            '#!/bin/sh\n[ "$1" = "--version" ] && { echo \'codex-cli 9.9.0-test\'; exit 0; }\nexit 97\n',
            executable=True,
        )
        write(self.node, "#!/bin/sh\nexit 0\n", executable=True)

    def _write_contract(self, *, mission_id: str = "fixture", **extra: object) -> None:
        config = {
            "contract_version": "akw-js-runtime-v1",
            "broker": {
                "type": "codex-exec",
                "command": "codex",
                "sandbox": "read-only",
                "approval_policy": "never",
                "ephemeral": True,
                "disable_nested_agents": True,
                "extra_args": LOCK.SAFE_CODEX_EXTRA_ARGS,
            },
        }
        write(self.runtime_config, json.dumps(config, sort_keys=True) + "\n")
        session_config = {
            "schema_version": 2,
            "input_mode": "repository",
            "runner_kind": "stable",
            "task_dir": str(self.workspace),
        }
        session_state = {
            "schema_version": 2,
            "session_id": mission_id,
            "backend": "cpu",
            "integration_pattern": "hf-macos-intake",
        }
        write(
            self.session / "session-config.json",
            json.dumps(session_config, sort_keys=True) + "\n",
        )
        write(
            self.session / "state.json",
            json.dumps(session_state, sort_keys=True) + "\n",
        )
        python = str(Path(sys.executable).resolve())
        source = json.loads(self.source_lock.read_text())
        mission = {
            "contract_version": "kersor-mission-v1",
            "workspace": str(self.workspace),
            "session": str(self.session),
            "runtime": "codex",
            "runtime_config": str(self.runtime_config),
            "mission": {
                "mission_id": mission_id,
                "goal": "test only",
                "authority": [],
                "required_artifacts": [],
                "required_facts": {},
                "max_revisions": 1,
            },
            "capabilities": [
                {
                    "name": "verify_source_lock",
                    "side_effect": "none",
                    "execution": {
                        "kind": "host_evaluator",
                        "retryable": False,
                        "request": {
                            "protocol": "command-v1",
                            "filesystem_policy": "read-only",
                            "network_policy": "denied",
                            "output_policy": "sealed",
                            "argv": [
                                python,
                                "-S",
                                "-B",
                                str(self.workspace / "scripts/resolve_hf_source.py"),
                                "--verify",
                                str(self.source_lock),
                                "--expected-sha256",
                                source["content_sha256"],
                            ],
                            "cwd": ".",
                            "artifacts": [".apxinf/source-lock.json"],
                            "timeout_seconds": 60,
                            "max_output_bytes": 65536,
                        },
                        "fact_projections": [
                            {
                                "output_name": "source_lock_valid",
                                "result_path": "passed",
                            }
                        ],
                    },
                },
                {
                    "name": "validate_port_manifest",
                    "side_effect": "none",
                    "execution": {
                        "kind": "host_evaluator",
                        "retryable": False,
                        "request": {
                            "protocol": "command-v1",
                            "filesystem_policy": "read-only",
                            "network_policy": "denied",
                            "output_policy": "sealed",
                            "argv": [
                                python,
                                "-S",
                                "-B",
                                str(
                                    self.workspace
                                    / "scripts/validate_hf_port_manifest.py"
                                ),
                                "--workspace",
                                str(self.workspace),
                                "--json",
                                "",
                                "--source-lock",
                                str(self.source_lock),
                                "--deployment-profile",
                                str(
                                    self.workspace
                                    / "configs/hf-onboarding/qwen35-0.8b-macos-cpu.json"
                                ),
                                "--expected-repo-id",
                                source["repo_id"],
                                "--expected-requested-revision",
                                source["requested_revision"],
                                "--expected-resolved-commit",
                                source["resolved_commit"],
                                "--expected-source-lock-content-sha256",
                                source["content_sha256"],
                                "--require-ready-existing",
                            ],
                            "cwd": ".",
                            "artifacts": [],
                            "timeout_seconds": 60,
                            "max_output_bytes": 65536,
                        },
                        "input_artifact_field": "argv.7",
                        "fact_projections": [
                            {
                                "output_name": "port_manifest_valid",
                                "result_path": "passed",
                            },
                            {
                                "output_name": "route_verified",
                                "result_path": "passed",
                            },
                            {
                                "output_name": "decision_complete",
                                "result_path": "passed",
                            },
                        ],
                    },
                },
            ],
            **extra,
        }
        write(self.mission, json.dumps(mission, sort_keys=True) + "\n")

    def _init_git(self) -> None:
        for command in (
            ["git", "init", "-q", str(self.kersor)],
            ["git", "-C", str(self.kersor), "add", "."],
            [
                "git",
                "-C",
                str(self.kersor),
                "-c",
                "user.name=Fixture",
                "-c",
                "user.email=fixture@example.invalid",
                "commit",
                "-qm",
                "fixture",
            ],
        ):
            subprocess.run(command, check=True, capture_output=True)

    def build_lock(
        self,
        *,
        write_lock: bool = True,
        host_python_path: Path | None = None,
        source_lock_path: Path | None = None,
    ) -> dict[str, object]:
        relocated_module = self.workspace / "scripts/kersor_runtime_lock.py"
        with mock.patch.object(LOCK, "__file__", str(relocated_module)):
            lock = LOCK.build_runtime_lock(
                kersor_root=self.kersor,
                mission_path=self.mission,
                runtime_config_path=self.runtime_config,
                codex_command=self.codex,
                node_command=self.node,
                source_lock_path=source_lock_path or self.source_lock,
                host_python_path=host_python_path,
            )
        if write_lock:
            LOCK.write_runtime_lock(self.lock_path, lock)
        return lock

    def launcher(
        self,
        *,
        dry_run: bool = False,
        mode: str = "admit-only",
        ambient: dict[str, str] | None = None,
    ):
        command = [
            sys.executable,
            str(LAUNCHER_PATH),
            "--lock",
            str(self.lock_path),
            "--auth-home",
            str(self.auth_home),
            "--mission",
            str(self.mission),
            "--runtime-config",
            str(self.runtime_config),
            "--codex",
            str(self.codex),
            "--node",
            str(self.node),
            f"--{mode}",
        ]
        if dry_run:
            command.append("--dry-run")
        environment = os.environ.copy()
        environment.update(ambient or {})
        return subprocess.run(
            command, text=True, capture_output=True, check=False, env=environment
        )


class RuntimeLockTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = FakeRuntime()

    def tearDown(self) -> None:
        self.fixture.cleanup()

    def prepared_mission(self) -> dict[str, object]:
        source = json.loads(self.fixture.source_lock.read_text())
        return PREPARE.build_mission(
            workspace=self.fixture.workspace,
            session=self.fixture.session,
            runtime_config=self.fixture.runtime_config,
            repo_id=source["repo_id"],
            revision=source["requested_revision"],
            resolved_commit=source["resolved_commit"],
            source_lock=self.fixture.source_lock,
            source_lock_sha256=source["content_sha256"],
            mission_id="fixture",
        )

    def test_v2_file_record_binds_fd_identity_metadata(self) -> None:
        record = LOCK.file_record(self.fixture.mission)
        self.assertEqual(
            set(record),
            {"path", "sha256", "size", "mode", "dev", "ino", "uid", "gid", "nlink"},
        )

    def test_lock_binds_v2_execution_closure_and_layers(self) -> None:
        lock = self.fixture.build_lock()
        receipt = LOCK.validate_runtime_lock(lock, node_command=self.fixture.node)
        self.assertTrue(receipt["passed"])
        self.assertEqual(lock["contract"], "apxinf-kersor-runtime-lock-v2")
        self.assertEqual(
            lock["kersor"]["auth_custody"]["mechanism"], LOCK.AUTH_CUSTODY_MECHANISM
        )
        self.assertEqual(
            lock["kersor"]["auth_custody"]["command_read_scope_mechanism"],
            LOCK.COMMAND_READ_SCOPE_MECHANISM,
        )
        self.assertEqual(
            set(lock["kersor"]["layers"]["fresh_snapshot_sources"]),
            set(LOCK.EXECUTOR_RUNTIME_PATHS),
        )
        self.assertIn(
            "scripts/run-autonomous-workflow.py",
            lock["kersor"]["layers"]["always_live_admission"],
        )
        self.assertEqual(
            lock["source_lock"]["content_sha256"],
            json.loads(self.fixture.source_lock.read_text())["content_sha256"],
        )
        self.assertEqual(len(lock["host_evaluators"]), 2)
        self.assertEqual(set(lock["apxinf"]), {"root", "files"})
        self.assertIn("dynamic_libraries", lock["runtime"]["node"])
        self.assertIn("dynamic_libraries", lock["runtime"]["host_python"])
        self.assertEqual(
            lock["runtime"]["host_python"]["closure_scope"],
            LOCK.HOST_PYTHON_CLOSURE_SCOPE,
        )

    def test_validation_executes_no_version_git_or_otool_probe(self) -> None:
        lock = self.fixture.build_lock()
        with mock.patch.object(
            LOCK.subprocess, "run", side_effect=AssertionError("must not execute")
        ):
            self.assertTrue(
                LOCK.validate_runtime_lock(lock, node_command=self.fixture.node)[
                    "passed"
                ]
            )

    def test_runtime_mission_session_source_host_and_binary_drift_fail_closed(
        self,
    ) -> None:
        targets = (
            self.fixture.kersor / "runtime/autonomous-controller.js",
            self.fixture.mission,
            self.fixture.runtime_config,
            self.fixture.session / "state.json",
            self.fixture.source_lock,
            self.fixture.workspace / "scripts/resolve_hf_source.py",
            self.fixture.codex,
            self.fixture.node,
        )
        for path in targets:
            with self.subTest(path=path):
                lock = self.fixture.build_lock(write_lock=False)
                original, mode = path.read_bytes(), path.stat().st_mode
                path.write_bytes(original + b"\n")
                path.chmod(mode)
                with self.assertRaises(LOCK.RuntimeLockError):
                    LOCK.validate_runtime_lock(lock, node_command=self.fixture.node)
                path.write_bytes(original)
                path.chmod(mode)

    def test_added_runtime_file_is_detected_without_git_probe(self) -> None:
        lock = self.fixture.build_lock()
        write(self.fixture.kersor / "runtime/injected.mjs", "export default 1\n")
        with self.assertRaises(LOCK.RuntimeLockError):
            LOCK.validate_runtime_lock(lock, node_command=self.fixture.node)

    def test_legacy_v1_is_formally_fail_closed(self) -> None:
        with self.assertRaisesRegex(LOCK.RuntimeLockError, "legacy runtime lock v1"):
            LOCK.validate_runtime_lock(
                {"contract": LOCK.LEGACY_LOCK_CONTRACT, "schema_version": 1}
            )

    def test_runtime_config_rejects_authority_expansion(self) -> None:
        payload = json.loads(self.fixture.runtime_config.read_text())
        for key, value in (
            ("outer_sandbox", "workspace-write"),
            ("additional_dirs", ["/tmp"]),
            ("sandbox_preflight", False),
        ):
            with self.subTest(key=key):
                candidate = json.loads(json.dumps(payload))
                candidate["broker"][key] = value
                with self.assertRaises(LOCK.RuntimeLockError):
                    LOCK.validate_runtime_config_policy(candidate)

    def test_host_evaluator_materialization_is_rejected_by_lock_creation(self) -> None:
        mission = json.loads(self.fixture.mission.read_text())
        mission["capabilities"][0]["execution"]["request"]["materialize"] = [
            {"path": "src/main.rs", "content": "must-not-be-written"}
        ]
        write(self.fixture.mission, json.dumps(mission, sort_keys=True) + "\n")

        with self.assertRaisesRegex(
            LOCK.RuntimeLockError, "Host evaluator materialize must be absent or empty"
        ):
            self.fixture.build_lock(write_lock=False)

    def test_lock_records_the_complete_host_evaluator_request_policy(self) -> None:
        mission = json.loads(self.fixture.mission.read_text())
        expected_request = mission["capabilities"][0]["execution"]["request"]

        lock = self.fixture.build_lock(write_lock=False)
        observed = lock["host_evaluators"][0]

        self.assertEqual(observed["request"], expected_request)
        self.assertEqual(
            observed["request_semantic_sha256"],
            LOCK.semantic_json_sha256(expected_request),
        )

    def test_prepare_mission_builds_a_lock_with_exact_artifact_input_binding(
        self,
    ) -> None:
        mission = self.prepared_mission()
        write(self.fixture.mission, json.dumps(mission, sort_keys=True) + "\n")

        lock = self.fixture.build_lock(write_lock=False)

        evaluator = next(
            item
            for item in lock["host_evaluators"]
            if item["name"] == "validate_port_manifest"
        )
        self.assertEqual(evaluator["input_artifact_field"], "argv.7")
        self.assertEqual(evaluator["argv"][6:8], ["--json", ""])
        self.assertEqual(evaluator["argv"][-1], "--require-ready-existing")
        self.assertEqual(
            evaluator["request"]["output_policy"],
            "sealed",
        )
        self.assertEqual(
            {
                projection["output_name"]: projection["result_path"]
                for projection in mission["capabilities"][-1]["execution"][
                    "fact_projections"
                ]
            },
            {
                "port_manifest_valid": "passed",
                "route_verified": "passed",
                "decision_complete": "passed",
            },
        )
        sealed_result = {"passed": True, "stdout_json": None}
        self.assertTrue(
            all(
                sealed_result[projection["result_path"]]
                for projection in mission["capabilities"][-1]["execution"][
                    "fact_projections"
                ]
            )
        )

    def test_artifact_input_binding_rejects_name_value_and_position_drift(
        self,
    ) -> None:
        for mutation in (
            "other-evaluator",
            "wrong-field",
            "wrong-flag-position",
            "nonempty-slot",
            "missing-field",
        ):
            with self.subTest(mutation=mutation):
                mission = self.prepared_mission()
                capabilities = {item["name"]: item for item in mission["capabilities"]}
                validator = capabilities["validate_port_manifest"]
                execution = validator["execution"]
                if mutation == "other-evaluator":
                    validator["name"] = "renamed_port_manifest_validator"
                elif mutation == "wrong-field":
                    execution["input_artifact_field"] = "argv.8"
                elif mutation == "wrong-flag-position":
                    execution["request"]["argv"][6] = "--artifact-json"
                elif mutation == "nonempty-slot":
                    execution["request"]["argv"][7] = "{}"
                else:
                    del execution["input_artifact_field"]
                write(
                    self.fixture.mission,
                    json.dumps(mission, sort_keys=True) + "\n",
                )
                with self.assertRaisesRegex(
                    LOCK.RuntimeLockError,
                    "input_artifact_field|fixed intake evaluator",
                ):
                    self.fixture.build_lock(write_lock=False)

        mission = self.prepared_mission()
        capabilities = {item["name"]: item for item in mission["capabilities"]}
        capabilities["verify_source_lock"]["execution"]["input_artifact_field"] = (
            "argv.7"
        )
        write(self.fixture.mission, json.dumps(mission, sort_keys=True) + "\n")
        with self.assertRaisesRegex(
            LOCK.RuntimeLockError,
            "only allowed for validate_port_manifest|fixed intake evaluator",
        ):
            self.fixture.build_lock(write_lock=False)

    def test_host_evaluator_policy_is_exact_and_rejects_unknown_fields(self) -> None:
        mutations = (
            ("protocol", "command-v2"),
            ("filesystem_policy", "workspace-write"),
            ("network_policy", "allowed"),
            ("output_policy", "materialized"),
            ("cwd", "/tmp"),
            ("shell", True),
        )
        for field, value in mutations:
            with self.subTest(field=field):
                self.fixture._write_contract()
                mission = json.loads(self.fixture.mission.read_text())
                mission["capabilities"][0]["execution"]["request"][field] = value
                write(self.fixture.mission, json.dumps(mission, sort_keys=True) + "\n")
                with self.assertRaises(LOCK.RuntimeLockError):
                    self.fixture.build_lock(write_lock=False)

    def test_host_evaluator_python_command_is_isolated_and_direct(self) -> None:
        mission = json.loads(self.fixture.mission.read_text())
        mission["capabilities"][0]["execution"]["request"]["argv"][1:3] = [
            "-B",
            "-S",
        ]
        write(self.fixture.mission, json.dumps(mission, sort_keys=True) + "\n")
        with self.assertRaisesRegex(LOCK.RuntimeLockError, "exact '-S -B'"):
            self.fixture.build_lock(write_lock=False)

        self.fixture._write_contract()
        script = self.fixture.workspace / "scripts/resolve_hf_source.py"
        target = self.fixture.workspace / "scripts/resolve_hf_source_impl.py"
        script.rename(target)
        script.symlink_to(target.name)
        with self.assertRaisesRegex(LOCK.RuntimeLockError, "script.*direct"):
            self.fixture.build_lock(write_lock=False)

    def test_intake_lock_rejects_an_unregistered_workspace_evaluator_script(
        self,
    ) -> None:
        mission = json.loads(self.fixture.mission.read_text())
        mission["capabilities"][0]["execution"]["request"]["argv"][3] = str(
            self.fixture.workspace / "scripts/source_validator.py"
        )
        write(self.fixture.mission, json.dumps(mission, sort_keys=True) + "\n")

        with self.assertRaisesRegex(LOCK.RuntimeLockError, "fixed intake evaluator"):
            self.fixture.build_lock(write_lock=False)

    def test_fixed_intake_evaluator_contract_rejects_shape_and_identity_drift(
        self,
    ) -> None:
        for mutation in (
            "extra-host",
            "duplicate-agent-name",
            "order",
            "argv",
            "timeout",
            "output-cap",
            "artifacts",
            "profile",
            "source",
            "source-identity",
            "fact-projection",
            "request-key",
        ):
            with self.subTest(mutation=mutation):
                self.fixture._write_contract()
                mission = json.loads(self.fixture.mission.read_text())
                source = mission["capabilities"][0]
                manifest = mission["capabilities"][1]
                if mutation == "extra-host":
                    mission["capabilities"].append(json.loads(json.dumps(source)))
                elif mutation == "duplicate-agent-name":
                    mission["capabilities"].append(
                        {
                            "name": "verify_source_lock",
                            "execution": {"kind": "agent"},
                        }
                    )
                elif mutation == "order":
                    mission["capabilities"][:2] = [manifest, source]
                elif mutation == "argv":
                    source["execution"]["request"]["argv"][4] = "--check"
                elif mutation == "timeout":
                    source["execution"]["request"]["timeout_seconds"] = 61
                elif mutation == "output-cap":
                    manifest["execution"]["request"]["max_output_bytes"] = 65535
                elif mutation == "artifacts":
                    source["execution"]["request"]["artifacts"] = []
                elif mutation == "profile":
                    manifest["execution"]["request"]["argv"][11] = str(
                        self.fixture.source_lock
                    )
                elif mutation == "source":
                    manifest["execution"]["request"]["argv"][9] = str(
                        self.fixture.workspace / ".apxinf/other.json"
                    )
                elif mutation == "source-identity":
                    manifest["execution"]["request"]["argv"][13] = "org/other"
                elif mutation == "fact-projection":
                    manifest["execution"]["fact_projections"][1]["result_path"] = (
                        "stdout_json.route_verified"
                    )
                else:
                    del manifest["execution"]["request"]["artifacts"]
                write(
                    self.fixture.mission,
                    json.dumps(mission, sort_keys=True) + "\n",
                )
                with self.assertRaises(LOCK.RuntimeLockError):
                    self.fixture.build_lock(write_lock=False)

    def test_agent_capabilities_reject_all_mutation_contract_fields(self) -> None:
        for field, value in (
            ("side_effect", "write"),
            ("transaction_artifacts", []),
            ("commit_failed_outputs", False),
            ("candidate_verifier", "verify_candidate"),
        ):
            with self.subTest(field=field):
                self.fixture._write_contract()
                mission = json.loads(self.fixture.mission.read_text())
                capability = {
                    "name": "analyze",
                    "required_authorities": [],
                    "produces_artifacts": [],
                    "produces_facts": [],
                    field: value,
                }
                mission["capabilities"].append(capability)
                write(self.fixture.mission, json.dumps(mission, sort_keys=True) + "\n")
                with self.assertRaisesRegex(
                    LOCK.RuntimeLockError, "read-only Agent capability"
                ):
                    self.fixture.build_lock(write_lock=False)

    def test_mission_workspace_must_be_the_locked_apxinf_root(self) -> None:
        other = self.fixture.base / "other-workspace"
        other.mkdir()
        mission = json.loads(self.fixture.mission.read_text())
        mission["workspace"] = str(other)
        write(self.fixture.mission, json.dumps(mission, sort_keys=True) + "\n")
        with self.assertRaisesRegex(LOCK.RuntimeLockError, "ApxInf root"):
            self.fixture.build_lock(write_lock=False)

    def test_session_task_dir_and_state_id_are_cross_bound(self) -> None:
        cases = (("task_dir", str(self.fixture.base)), ("session_id", "other"))
        for field, value in cases:
            with self.subTest(field=field):
                self.fixture._write_contract()
                filename = (
                    "session-config.json" if field == "task_dir" else "state.json"
                )
                path = self.fixture.session / filename
                payload = json.loads(path.read_text())
                payload[field] = value
                write(path, json.dumps(payload, sort_keys=True) + "\n")
                with self.assertRaisesRegex(LOCK.RuntimeLockError, "Session"):
                    self.fixture.build_lock(write_lock=False)

    def test_explicit_source_lock_must_match_the_evaluator_binding(self) -> None:
        other = self.fixture.workspace / ".apxinf/other-source-lock.json"
        payload = json.loads(self.fixture.source_lock.read_text())
        payload["repo_id"] = "org/other-model"
        body = dict(payload)
        body.pop("content_sha256")
        payload["content_sha256"] = LOCK.semantic_json_sha256(body)
        write(other, json.dumps(payload, sort_keys=True) + "\n")
        with self.assertRaisesRegex(LOCK.RuntimeLockError, "source lock.*Mission"):
            self.fixture.build_lock(write_lock=False, source_lock_path=other)

    def test_locked_host_python_must_be_the_current_interpreter(self) -> None:
        with self.assertRaisesRegex(LOCK.RuntimeLockError, "current interpreter"):
            self.fixture.build_lock(
                write_lock=False, host_python_path=self.fixture.node
            )

    def test_host_python_closure_is_required_during_validation(self) -> None:
        lock = self.fixture.build_lock(write_lock=False)
        del lock["runtime"]["host_python"]["dynamic_libraries"]
        lock["lock_sha256"] = LOCK.lock_self_sha256(lock)
        with self.assertRaisesRegex(LOCK.RuntimeLockError, "Host Python"):
            LOCK.validate_runtime_lock(lock, node_command=self.fixture.node)

    def test_host_evaluator_workspace_input_file_drift_fails_closed(self) -> None:
        profile = (
            self.fixture.workspace / "configs/hf-onboarding/qwen35-0.8b-macos-cpu.json"
        )
        lock = self.fixture.build_lock(write_lock=False)
        locked_inputs = lock["host_evaluators"][1]["input_files"]
        self.assertIn(str(profile), {item["path"] for item in locked_inputs})
        write(profile, '{"profile":"drifted"}\n')
        with self.assertRaisesRegex(LOCK.RuntimeLockError, "Host evaluator"):
            LOCK.validate_runtime_lock(lock, node_command=self.fixture.node)

    def test_missing_command_read_scope_marker_fails_closed(self) -> None:
        broker = self.fixture.kersor / "runtime/brokers/codex-exec.mjs"
        payload = broker.read_text().replace(
            "export const CODEX_COMMAND_READ_SCOPE_MECHANISM='codex-minimal-project-read-v1'\n",
            "",
        )
        write(broker, payload)
        with self.assertRaisesRegex(LOCK.RuntimeLockError, "read-scope marker"):
            self.fixture.build_lock(write_lock=False)


class LauncherTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = FakeRuntime()

    def tearDown(self) -> None:
        self.fixture.cleanup()

    def test_dry_run_has_one_explicit_admission_argv_and_empty_ambient_allowlist(
        self,
    ) -> None:
        self.fixture.build_lock()
        result = self.fixture.launcher(
            dry_run=True,
            ambient={
                "HF_TOKEN": "HF-SENTINEL",
                "KERSOR_CODEX_OUTER_SANDBOX": "workspace-write",
                "KERSOR_CODEX_COMMAND": "/tmp/attacker",
                "NODE_OPTIONS": "--require=/tmp/x",
            },
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        receipt = json.loads(result.stdout)
        self.assertEqual(receipt["mode"], "admit-only")
        self.assertEqual(receipt["environment"]["ambient_allowlist"], [])
        self.assertIn("--admit-only", receipt["argv"])
        self.assertNotIn("--resume", receipt["argv"])
        self.assertEqual(receipt["environment"]["codex"], str(self.fixture.codex))
        self.assertEqual(receipt["environment"]["node"], str(self.fixture.node))
        self.assertEqual(receipt["environment"]["accelerator_activity"], "0")
        self.assertEqual(
            receipt["command_read_scope_mechanism"],
            LOCK.COMMAND_READ_SCOPE_MECHANISM,
        )
        for value in (
            "HF-SENTINEL",
            "workspace-write",
            "/tmp/attacker",
            "AUTH-CONTENT-MUST-NOT-BE-READ",
        ):
            self.assertNotIn(value, result.stdout)
        self.assertFalse(self.fixture.capture.exists())

    def test_admission_and_later_explicit_resume_are_two_separate_invocations(
        self,
    ) -> None:
        self.fixture.build_lock()
        admitted = self.fixture.launcher(
            ambient={
                "HF_TOKEN": "HF-SENTINEL",
                "KERSOR_AUTONOMOUS_RUNNER": "/tmp/attacker.py",
                "KERSOR_NODE_BIN": "/tmp/node",
                "NODE_OPTIONS": "--no-warnings",
            }
        )
        self.assertEqual(admitted.returncode, 0, admitted.stdout + admitted.stderr)
        self.assertEqual(len(admitted.stdout.splitlines()), 1)
        admission_receipt = json.loads(admitted.stdout)
        self.assertEqual(admission_receipt["status"], "admitted")
        self.assertFalse(admission_receipt["agent_started"])
        self.assertEqual(
            admission_receipt["command_read_scope_mechanism"],
            LOCK.COMMAND_READ_SCOPE_MECHANISM,
        )
        history = json.loads(self.fixture.capture.read_text())
        self.assertEqual(len(history), 1)
        self.assertIn("--admit-only", history[0]["argv"])
        self.assertEqual(history[0]["environment"]["KERSOR_ACCELERATOR_ACTIVITY"], "0")

        resumed = self.fixture.launcher(
            mode="resume",
            ambient={
                "HF_TOKEN": "HF-SENTINEL",
                "KERSOR_AUTONOMOUS_RUNNER": "/tmp/attacker.py",
                "KERSOR_NODE_BIN": "/tmp/node",
                "NODE_OPTIONS": "--no-warnings",
            },
        )
        self.assertEqual(resumed.returncode, 0, resumed.stdout + resumed.stderr)
        history = json.loads(self.fixture.capture.read_text())
        self.assertEqual(len(history), 2)
        self.assertIn("--resume", history[1]["argv"])
        env = history[1]["environment"]
        for key in ("HF_TOKEN", "DYLD_INSERT_LIBRARIES", "NODE_OPTIONS"):
            self.assertNotIn(key, env)
        self.assertEqual(env["KERSOR_NODE_BIN"], str(self.fixture.node))
        self.assertEqual(env["KERSOR_CODEX_COMMAND"], str(self.fixture.codex))
        self.assertEqual(env["KERSOR_ACCELERATOR_ACTIVITY"], "0")
        self.assertEqual(
            env["KERSOR_AUTONOMOUS_RUNNER"],
            str(self.fixture.kersor / "scripts/run-autonomous-workflow.py"),
        )
        self.assertTrue(env["PYTHONPYCACHEPREFIX"].endswith("/pycache"))

    def test_existing_admission_refuses_recreation_and_fresh_flag_is_rejected(
        self,
    ) -> None:
        self.fixture.build_lock()
        self.assertEqual(self.fixture.launcher().returncode, 0)
        second = self.fixture.launcher(mode="admit-only")
        self.assertEqual(second.returncode, 2)
        self.assertIn("--admit-only refuses an existing run", second.stdout)

        deprecated = self.fixture.launcher(mode="fresh")
        self.assertEqual(deprecated.returncode, 2)
        self.assertIn(
            "one of the arguments --admit-only --resume is required",
            deprecated.stderr,
        )
        history = json.loads(self.fixture.capture.read_text())
        self.assertEqual(len(history), 1)

        resumed = self.fixture.launcher(mode="resume")
        self.assertEqual(resumed.returncode, 0, resumed.stdout + resumed.stderr)

    def test_bad_admission_receipt_stops_before_resume(self) -> None:
        self.fixture._write_contract(_test_bad_admission=True)
        self.fixture.build_lock()
        result = self.fixture.launcher()
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        history = json.loads(self.fixture.capture.read_text())
        self.assertEqual(len(history), 1)
        self.assertIn("--admit-only", history[0]["argv"])

    def test_duplicate_key_admission_receipt_stops_before_resume(self) -> None:
        self.fixture._write_contract(_test_duplicate_admission_key=True)
        self.fixture.build_lock()

        result = self.fixture.launcher()

        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertIn("duplicate JSON key", result.stdout)
        history = json.loads(self.fixture.capture.read_text())
        self.assertEqual(len(history), 1)
        self.assertIn("--admit-only", history[0]["argv"])

    def test_resume_refuses_an_admission_with_execution_evidence(self) -> None:
        lock = self.fixture.build_lock()
        admitted = self.fixture.launcher()
        self.assertEqual(admitted.returncode, 0, admitted.stdout + admitted.stderr)
        run_dir = Path(lock["mission_binding"]["run_dir"])
        (run_dir / ".runtime").mkdir()

        resumed = self.fixture.launcher(mode="resume")

        self.assertEqual(resumed.returncode, 2, resumed.stdout + resumed.stderr)
        self.assertIn("pristine admission", resumed.stdout)
        history = json.loads(self.fixture.capture.read_text())
        self.assertEqual(len(history), 1)
        self.assertIn("--admit-only", history[0]["argv"])

    def test_resume_refuses_an_unknown_admission_top_level_file(self) -> None:
        lock = self.fixture.build_lock()
        admitted = self.fixture.launcher()
        self.assertEqual(admitted.returncode, 0, admitted.stdout + admitted.stderr)
        run_dir = Path(lock["mission_binding"]["run_dir"])
        write(run_dir / "injected.txt", "harmless injected fixture\n")

        resumed = self.fixture.launcher(mode="resume")

        self.assertEqual(resumed.returncode, 2, resumed.stdout + resumed.stderr)
        self.assertIn("pristine admission inventory", resumed.stdout)
        history = json.loads(self.fixture.capture.read_text())
        self.assertEqual(len(history), 1)

    def test_resume_refuses_an_injected_executor_runtime_file(self) -> None:
        lock = self.fixture.build_lock()
        admitted = self.fixture.launcher()
        self.assertEqual(admitted.returncode, 0, admitted.stdout + admitted.stderr)
        run_dir = Path(lock["mission_binding"]["run_dir"])
        write(
            run_dir / "executor-runtime/runtime/injected.mjs",
            "export default 'harmless fixture'\n",
        )

        resumed = self.fixture.launcher(mode="resume")

        self.assertEqual(resumed.returncode, 2, resumed.stdout + resumed.stderr)
        self.assertIn("executor runtime tree", resumed.stdout)
        history = json.loads(self.fixture.capture.read_text())
        self.assertEqual(len(history), 1)

    def test_resume_refuses_unknown_executor_manifest_fields(self) -> None:
        lock = self.fixture.build_lock()
        admitted = self.fixture.launcher()
        self.assertEqual(admitted.returncode, 0, admitted.stdout + admitted.stderr)
        manifest_path = (
            Path(lock["mission_binding"]["run_dir"]) / "executor-runtime-manifest.json"
        )
        manifest = json.loads(manifest_path.read_text())
        manifest["unknown"] = "not locked"
        write(manifest_path, json.dumps(manifest, sort_keys=True) + "\n")

        resumed = self.fixture.launcher(mode="resume")

        self.assertEqual(resumed.returncode, 2, resumed.stdout + resumed.stderr)
        self.assertIn("manifest schema is malformed", resumed.stdout)

    def test_resume_refuses_duplicate_executor_manifest_paths(self) -> None:
        lock = self.fixture.build_lock()
        admitted = self.fixture.launcher()
        self.assertEqual(admitted.returncode, 0, admitted.stdout + admitted.stderr)
        manifest_path = (
            Path(lock["mission_binding"]["run_dir"]) / "executor-runtime-manifest.json"
        )
        manifest = json.loads(manifest_path.read_text())
        manifest["files"].append(dict(manifest["files"][0]))
        write(manifest_path, json.dumps(manifest, sort_keys=True) + "\n")

        resumed = self.fixture.launcher(mode="resume")

        self.assertEqual(resumed.returncode, 2, resumed.stdout + resumed.stderr)
        self.assertIn("frozen inventory is incomplete", resumed.stdout)

    def test_resume_refuses_non_private_executor_manifest(self) -> None:
        lock = self.fixture.build_lock()
        admitted = self.fixture.launcher()
        self.assertEqual(admitted.returncode, 0, admitted.stdout + admitted.stderr)
        manifest_path = (
            Path(lock["mission_binding"]["run_dir"]) / "executor-runtime-manifest.json"
        )
        manifest_path.chmod(0o644)

        resumed = self.fixture.launcher(mode="resume")

        self.assertEqual(resumed.returncode, 2, resumed.stdout + resumed.stderr)
        self.assertIn("manifest must have exact mode 0600", resumed.stdout)

    def test_resume_refuses_hard_linked_controller(self) -> None:
        lock = self.fixture.build_lock()
        admitted = self.fixture.launcher()
        self.assertEqual(admitted.returncode, 0, admitted.stdout + admitted.stderr)
        controller = Path(lock["mission_binding"]["run_dir"]) / "controller.js"
        os.link(controller, self.fixture.base / "controller-hard-link.js")

        resumed = self.fixture.launcher(mode="resume")

        self.assertEqual(resumed.returncode, 2, resumed.stdout + resumed.stderr)
        self.assertIn("controller must have exactly one hard link", resumed.stdout)

    def test_resume_refuses_binding_session_id_drift(self) -> None:
        lock = self.fixture.build_lock()
        admitted = self.fixture.launcher()
        self.assertEqual(admitted.returncode, 0, admitted.stdout + admitted.stderr)
        binding_path = Path(lock["mission_binding"]["run_dir"]) / "binding.json"
        binding = json.loads(binding_path.read_text())
        binding["session_id"] = "different-session"
        write(binding_path, json.dumps(binding, sort_keys=True) + "\n")

        resumed = self.fixture.launcher(mode="resume")

        self.assertEqual(resumed.returncode, 2, resumed.stdout + resumed.stderr)
        self.assertIn("binding mismatch for session_id", resumed.stdout)

    def test_resume_refuses_binding_session_schema_drift(self) -> None:
        lock = self.fixture.build_lock()
        admitted = self.fixture.launcher()
        self.assertEqual(admitted.returncode, 0, admitted.stdout + admitted.stderr)
        binding_path = Path(lock["mission_binding"]["run_dir"]) / "binding.json"
        binding = json.loads(binding_path.read_text())
        binding["session_schema_version"] += 1
        write(binding_path, json.dumps(binding, sort_keys=True) + "\n")

        resumed = self.fixture.launcher(mode="resume")

        self.assertEqual(resumed.returncode, 2, resumed.stdout + resumed.stderr)
        self.assertIn("binding mismatch for session_schema_version", resumed.stdout)

    def test_post_run_runtime_drift_overrides_child_success(self) -> None:
        self.fixture._write_contract(_test_mutate_runtime=True)
        self.fixture.build_lock()
        admitted = self.fixture.launcher()
        self.assertEqual(admitted.returncode, 0, admitted.stdout + admitted.stderr)
        result = self.fixture.launcher(mode="resume")
        self.assertEqual(result.returncode, 78, result.stdout + result.stderr)
        self.assertEqual(
            json.loads(result.stderr)["error"]["code"], "KERSOR_POST_RUN_DRIFT"
        )

    def test_auth_metadata_profile_fails_closed_without_reading_secret(self) -> None:
        self.fixture.build_lock()
        auth = self.fixture.auth_home / "auth.json"
        auth.chmod(0o644)
        result = self.fixture.launcher(dry_run=True)
        self.assertEqual(result.returncode, 2)
        self.assertNotIn("AUTH-CONTENT-MUST-NOT-BE-READ", result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
