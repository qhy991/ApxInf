#!/usr/bin/env python3
"""Admit or explicitly resume one KerSor Mission under runtime lock v2."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from kersor_runtime_lock import (  # noqa: E402
    AUTH_CUSTODY_MECHANISM,
    EXECUTOR_RUNTIME_PATHS,
    RuntimeLockError,
    canonical_json_bytes,
    file_record,
    load_runtime_lock,
    parse_json,
    read_file_bytes,
    semantic_json_sha256,
    sha256_bytes,
    validate_runtime_lock,
)


class LaunchError(RuntimeLockError):
    """Raised when Host custody checks refuse to launch KerSor."""


ADMISSION_CONTRACT = "kersor-autonomous-admission-v1"
EXECUTION_EVIDENCE_PATHS = (".runtime", "dispatch.json", "result.json", "artifacts")
ADMISSION_TOP_LEVEL_FILES = frozenset(
    {
        "binding.json",
        "controller.js",
        "executor-runtime-manifest.json",
        "mission.json",
        "runtime-config.json",
        "session-snapshot.json",
    }
)
ADMISSION_TOP_LEVEL_DIRECTORIES = frozenset({"executor-runtime"})


def _inside(root: Path, candidate: Path) -> bool:
    try:
        candidate.relative_to(root)
    except ValueError:
        return False
    return True


def _safe_parent_chain(path: Path, *, uid: int) -> None:
    current = Path(path.anchor)
    for component in path.parts[1:]:
        current /= component
        try:
            entry = current.lstat()
        except OSError as exc:
            raise LaunchError(
                f"credential parent path is unavailable: {current}: {exc}"
            ) from exc
        if stat.S_ISLNK(entry.st_mode) or not stat.S_ISDIR(entry.st_mode):
            raise LaunchError(
                f"credential parent path must be a direct directory: {current}"
            )
        if entry.st_uid not in {0, uid}:
            raise LaunchError(f"credential parent has an unsafe owner: {current}")
        if stat.S_IMODE(entry.st_mode) & 0o022:
            raise LaunchError(f"credential parent is group/other writable: {current}")


def validate_auth_home(auth_home: Path, *, forbidden_roots: list[Path]) -> Path:
    """Validate credential metadata only; never open or hash ``auth.json``."""

    uid = os.getuid()
    declared_home = Path(os.path.abspath(os.fspath(auth_home.expanduser())))
    _safe_parent_chain(declared_home.parent, uid=uid)
    try:
        home_entry = declared_home.lstat()
    except OSError as exc:
        raise LaunchError(f"Codex auth home is unavailable: {exc}") from exc
    if stat.S_ISLNK(home_entry.st_mode) or not stat.S_ISDIR(home_entry.st_mode):
        raise LaunchError("Codex auth home must be a direct non-symlink directory")
    if home_entry.st_uid != uid or stat.S_IMODE(home_entry.st_mode) != 0o700:
        raise LaunchError(
            "Codex auth home must be owned by the current uid with mode 0700"
        )
    canonical_home = declared_home.resolve(strict=True)
    if canonical_home != declared_home:
        raise LaunchError("Codex auth home must not traverse symlinked parents")

    declared_auth = canonical_home / "auth.json"
    try:
        auth_entry = declared_auth.lstat()
    except OSError as exc:
        raise LaunchError(f"Codex auth.json is unavailable: {exc}") from exc
    if stat.S_ISLNK(auth_entry.st_mode) or not stat.S_ISREG(auth_entry.st_mode):
        raise LaunchError("Codex auth.json must be a direct regular file")
    if auth_entry.st_uid != uid:
        raise LaunchError("Codex auth.json must be owned by the current uid")
    if auth_entry.st_nlink != 1:
        raise LaunchError("Codex auth.json must have exactly one hard link")
    if stat.S_IMODE(auth_entry.st_mode) != 0o600:
        raise LaunchError("Codex auth.json must have exact mode 0600")
    canonical_auth = declared_auth.resolve(strict=True)
    if canonical_auth != declared_auth:
        raise LaunchError("Codex auth.json must not resolve through a link")
    for forbidden in forbidden_roots:
        try:
            root = forbidden.resolve(strict=True)
        except OSError:
            root = forbidden.absolute()
        if _inside(root, canonical_auth):
            raise LaunchError(
                f"Codex auth.json must remain outside locked writable/runtime root: {root}"
            )
    return canonical_home


def _trusted_path(lock: dict[str, Any]) -> str:
    runtime = lock["runtime"]
    candidates = [
        Path(runtime["codex"]["path"]).parent,
        Path(runtime["node"]["executable"]["path"]).parent,
        Path(runtime["host_python"]["path"]).parent,
        *(
            Path(record["path"]).parent
            for record in runtime["external_commands"].values()
        ),
    ]
    values: list[str] = []
    for candidate in candidates:
        rendered = str(candidate)
        if candidate.is_dir() and rendered not in values:
            values.append(rendered)
    return os.pathsep.join(values)


def build_sanitized_environment(
    *, lock: dict[str, Any], auth_home: Path, neutral_home: Path
) -> dict[str, str]:
    """Return a constant allowlist; no ambient variable or value is copied."""

    runtime = lock["runtime"]
    root = Path(lock["kersor"]["root"])
    return {
        "PATH": _trusted_path(lock),
        "HOME": str(neutral_home),
        "TMPDIR": str(neutral_home / "tmp"),
        "LANG": "C",
        "LC_ALL": "C",
        "PYTHONIOENCODING": "utf-8",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPYCACHEPREFIX": str(neutral_home / "pycache"),
        "PYTHONPATH": str(root),
        "NO_COLOR": "1",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_TERMINAL_PROMPT": "0",
        "GCM_INTERACTIVE": "Never",
        "KERSOR_CODEX_AUTH_HOME": str(auth_home),
        "KERSOR_CODEX_COMMAND": str(runtime["codex"]["path"]),
        "KERSOR_NODE_BIN": str(runtime["node"]["executable"]["path"]),
        "KERSOR_PYTHON": str(runtime["host_python"]["path"]),
        "KERSOR_ACCELERATOR_ACTIVITY": "0",
        "KERSOR_AUTONOMOUS_RUNNER": str(root / "scripts/run-autonomous-workflow.py"),
    }


def exact_evolve_argv(lock: dict[str, Any], *, mode: str) -> list[str]:
    if mode not in {"admit", "resume"}:
        raise LaunchError(f"unknown evolve mode: {mode}")
    try:
        root = Path(lock["kersor"]["root"])
        mission = str(lock["mission"]["path"])
        mission_sha256 = str(lock["mission"]["sha256"])
        runtime_config = str(lock["runtime_config"]["path"])
        runtime_config_sha256 = str(lock["runtime_config"]["sha256"])
        run_dir = str(lock["mission_binding"]["run_dir"])
    except (KeyError, TypeError) as exc:
        raise LaunchError(f"runtime lock is missing evolve bindings: {exc}") from exc
    return [
        str(root / "scripts/evolve.sh"),
        str(mission),
        "--runtime",
        "codex",
        "--runtime-config",
        runtime_config,
        "--run-dir",
        run_dir,
        "--expected-contract-sha256",
        mission_sha256,
        "--expected-runtime-config-sha256",
        runtime_config_sha256,
        "--admit-only" if mode == "admit" else "--resume",
    ]


def _direct_json(path: Path, label: str) -> tuple[dict[str, Any], dict[str, object]]:
    record = file_record(path)
    if record["nlink"] != 1:
        raise LaunchError(f"{label} must have exactly one hard link")
    payload = parse_json(read_file_bytes(path, label=label), label=label)
    return payload, record


def _pretty_semantic_sha(value: object) -> str:
    payload = (
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    ).encode("utf-8")
    return sha256_bytes(payload)


def _parse_admission_stdout(stdout: str) -> dict[str, Any]:
    lines = stdout.splitlines()
    if len(lines) != 1:
        raise LaunchError("KerSor admission must emit exactly one JSON receipt line")
    return parse_json(lines[0].encode("utf-8"), label="KerSor admission receipt")


def _verify_pristine_top_level_inventory(run_dir: Path) -> None:
    observed_files: set[str] = set()
    observed_directories: set[str] = set()
    try:
        children = list(run_dir.iterdir())
    except OSError as exc:
        raise LaunchError(
            f"pristine admission inventory is unavailable: {exc}"
        ) from exc
    for child in children:
        try:
            entry = child.lstat()
        except OSError as exc:
            raise LaunchError(
                f"pristine admission inventory entry is unavailable: {child.name}"
            ) from exc
        if stat.S_ISLNK(entry.st_mode):
            raise LaunchError(
                f"pristine admission inventory contains a symlink: {child.name}"
            )
        if stat.S_ISREG(entry.st_mode):
            observed_files.add(child.name)
        elif stat.S_ISDIR(entry.st_mode):
            observed_directories.add(child.name)
        else:
            raise LaunchError(
                f"pristine admission inventory contains a special file: {child.name}"
            )
    if (
        observed_files != ADMISSION_TOP_LEVEL_FILES
        or observed_directories != ADMISSION_TOP_LEVEL_DIRECTORIES
    ):
        raise LaunchError("pristine admission inventory differs from the admitted set")


def _verify_executor_runtime_tree(executor_root: Path) -> None:
    expected_files = set(EXECUTOR_RUNTIME_PATHS)
    expected_directories: set[str] = set()
    for relative in expected_files:
        parent = Path(relative).parent
        while parent != Path("."):
            expected_directories.add(parent.as_posix())
            parent = parent.parent

    try:
        root_entry = executor_root.lstat()
    except OSError as exc:
        raise LaunchError("executor runtime tree is unavailable") from exc
    if (
        stat.S_ISLNK(root_entry.st_mode)
        or not stat.S_ISDIR(root_entry.st_mode)
        or executor_root.resolve(strict=True) != executor_root
    ):
        raise LaunchError("executor runtime tree must be one direct directory")

    observed_files: set[str] = set()
    observed_directories: set[str] = set()
    pending = [executor_root]
    while pending:
        current = pending.pop()
        try:
            children = list(current.iterdir())
        except OSError as exc:
            raise LaunchError("executor runtime tree could not be enumerated") from exc
        for child in children:
            relative = child.relative_to(executor_root).as_posix()
            try:
                entry = child.lstat()
            except OSError as exc:
                raise LaunchError(
                    f"executor runtime tree entry is unavailable: {relative}"
                ) from exc
            if stat.S_ISLNK(entry.st_mode):
                raise LaunchError(
                    f"executor runtime tree contains a symlink: {relative}"
                )
            if stat.S_ISDIR(entry.st_mode):
                observed_directories.add(relative)
                pending.append(child)
            elif stat.S_ISREG(entry.st_mode):
                observed_files.add(relative)
            else:
                raise LaunchError(
                    f"executor runtime tree contains a special file: {relative}"
                )
    if observed_files != expected_files or observed_directories != expected_directories:
        raise LaunchError("executor runtime tree differs from its locked inventory")


def verify_frozen_admission(
    lock: dict[str, Any],
    *,
    admission_receipt: dict[str, Any] | None = None,
    require_pristine: bool = False,
) -> dict[str, Any]:
    """Verify the complete pre-execution admission snapshot using hashes only."""

    run_dir = Path(lock["mission_binding"]["run_dir"])
    try:
        entry = run_dir.lstat()
    except OSError as exc:
        raise LaunchError(f"frozen run admission is unavailable: {exc}") from exc
    if stat.S_ISLNK(entry.st_mode) or not stat.S_ISDIR(entry.st_mode):
        raise LaunchError("frozen run admission must be one direct directory")
    if run_dir.resolve(strict=True) != run_dir:
        raise LaunchError("frozen run admission traverses a symlink")

    binding, binding_record = _direct_json(run_dir / "binding.json", "run binding")
    mission, mission_record = _direct_json(run_dir / "mission.json", "frozen Mission")
    runtime_config, runtime_record = _direct_json(
        run_dir / "runtime-config.json", "frozen runtime config"
    )
    snapshot, snapshot_record = _direct_json(
        run_dir / "session-snapshot.json", "frozen Session snapshot"
    )
    manifest, manifest_record = _direct_json(
        run_dir / "executor-runtime-manifest.json", "executor runtime manifest"
    )
    controller_record = file_record(run_dir / "controller.js")
    if manifest_record["mode"] != 0o600:
        raise LaunchError("executor runtime manifest must have exact mode 0600")
    if controller_record["nlink"] != 1:
        raise LaunchError("frozen controller must have exactly one hard link")

    if semantic_json_sha256(mission) != lock["mission"]["semantic_sha256"]:
        raise LaunchError("frozen Mission semantic identity differs from runtime lock")
    if (
        semantic_json_sha256(runtime_config)
        != lock["runtime_config"]["semantic_sha256"]
    ):
        raise LaunchError(
            "frozen runtime config semantic identity differs from runtime lock"
        )
    if snapshot.get("schema_version") != 1 or set(snapshot) != {
        "schema_version",
        "config",
        "state",
    }:
        raise LaunchError("frozen Session snapshot schema is malformed")
    if (
        semantic_json_sha256(snapshot["config"])
        != lock["session"]["config"]["semantic_sha256"]
    ):
        raise LaunchError("frozen Session config differs from runtime lock")
    if (
        semantic_json_sha256(snapshot["state"])
        != lock["session"]["state"]["semantic_sha256"]
    ):
        raise LaunchError("frozen Session state differs from runtime lock")

    closure_by_path = {
        item["path"]: item for item in lock["kersor"]["closure"]["files"]
    }
    controller_source = closure_by_path.get("runtime/autonomous-controller.js")
    if not isinstance(controller_source, dict) or any(
        controller_record[key] != controller_source[key]
        for key in ("sha256", "size", "mode")
    ):
        raise LaunchError("frozen controller differs from its locked KerSor source")
    if set(manifest) != {"schema_version", "root", "files"} or (
        manifest.get("schema_version") != 1
        or manifest.get("root") != "executor-runtime"
    ):
        raise LaunchError("executor runtime manifest schema is malformed")
    entries = manifest.get("files")
    expected_runtime_paths = set(EXECUTOR_RUNTIME_PATHS)
    if not isinstance(entries, list) or len(entries) != len(expected_runtime_paths):
        raise LaunchError("executor runtime frozen inventory is incomplete")
    manifest_paths = [
        item.get("path") if isinstance(item, dict) else None for item in entries
    ]
    if (
        any(type(path) is not str for path in manifest_paths)
        or len(set(manifest_paths)) != len(manifest_paths)
        or set(manifest_paths) != expected_runtime_paths
    ):
        raise LaunchError("executor runtime frozen inventory is incomplete")
    _verify_executor_runtime_tree(run_dir / "executor-runtime")
    for item in entries:
        if not isinstance(item, dict) or set(item) != {
            "path",
            "sha256",
            "size_bytes",
            "mode",
        }:
            raise LaunchError("executor runtime manifest entry is malformed")
        source = closure_by_path.get(item["path"])
        frozen = file_record(run_dir / "executor-runtime" / item["path"])
        if (
            not isinstance(source, dict)
            or item["sha256"] != source["sha256"]
            or item["size_bytes"] != source["size"]
            or item["mode"] != f"{source['mode']:04o}"
            or frozen["sha256"] != item["sha256"]
            or frozen["size"] != item["size_bytes"]
            or frozen["mode"] != int(item["mode"], 8)
            or frozen["nlink"] != 1
        ):
            raise LaunchError(
                f"executor runtime snapshot differs from lock: {item['path']}"
            )

    executor_binding = {
        "schema_version": 1,
        "root": "executor-runtime",
        "manifest": "executor-runtime-manifest.json",
        "manifest_sha256": manifest_record["sha256"],
        "manifest_mode": "0600",
        "dispatch": "scripts/dispatch-workflow.sh",
        "workflow_host": "runtime/workflow-host.mjs",
    }
    expected_binding_fields = {
        "schema_version": 1,
        "run_id": lock["mission_binding"]["mission_id"],
        "session_dir": lock["mission_binding"]["session"],
        "source_mission_sha256": lock["mission"]["sha256"],
        "mission_sha256": mission_record["sha256"],
        "controller_path": "controller.js",
        "controller_sha256": controller_record["sha256"],
        "controller_size_bytes": controller_record["size"],
        "controller_mode": f"{controller_record['mode']:04o}",
        "source_runtime_config_sha256": lock["runtime_config"]["sha256"],
        "runtime_config_sha256": runtime_record["sha256"],
        "session_snapshot_sha256": snapshot_record["sha256"],
        "session_config_sha256": _pretty_semantic_sha(snapshot["config"]),
        "session_state_sha256": _pretty_semantic_sha(snapshot["state"]),
        "session_id": lock["mission_binding"]["mission_id"],
        "session_schema_version": snapshot["state"]["schema_version"],
        "executor_runtime": executor_binding,
        "runtime": "codex",
        "project_root": lock["mission_binding"]["workspace"],
    }
    for key, expected in expected_binding_fields.items():
        if binding.get(key) != expected:
            raise LaunchError(f"frozen run binding mismatch for {key}")
    expected_binding_keys = {
        *expected_binding_fields,
        "created_at",
        "session_id",
        "session_schema_version",
    }
    if set(binding) != expected_binding_keys or not isinstance(
        binding.get("created_at"), str
    ):
        raise LaunchError("frozen run binding has missing or unknown fields")

    if admission_receipt is not None or require_pristine:
        for relative in EXECUTION_EVIDENCE_PATHS:
            candidate = run_dir / relative
            if candidate.exists() or candidate.is_symlink():
                raise LaunchError(
                    f"pristine admission unexpectedly contains execution evidence: {relative}"
                )
        _verify_pristine_top_level_inventory(run_dir)
    if admission_receipt is not None:
        expected_receipt = {
            "contract_version": ADMISSION_CONTRACT,
            "status": "admitted",
            "revision": 0,
            "run_dir": str(run_dir),
            "binding_sha256": binding_record["sha256"],
            "source_mission_sha256": lock["mission"]["sha256"],
            "mission_sha256": mission_record["sha256"],
            "controller_sha256": controller_record["sha256"],
            "source_runtime_config_sha256": lock["runtime_config"]["sha256"],
            "runtime_config_sha256": runtime_record["sha256"],
            "session_snapshot_sha256": snapshot_record["sha256"],
            "executor_runtime_manifest_sha256": manifest_record["sha256"],
        }
        if admission_receipt != expected_receipt:
            raise LaunchError(
                "KerSor admission receipt differs from the frozen admission"
            )
    return {
        "passed": True,
        "contract": "apxinf-kersor-frozen-admission-verification-v1",
        "run_dir": str(run_dir),
        "binding_sha256": binding_record["sha256"],
    }


def _assert_lock_file_unchanged(
    lock_path: Path, expected_lock: dict[str, Any], expected_file_sha256: str
) -> None:
    observed_lock, observed_file_sha256 = load_runtime_lock(lock_path)
    if observed_file_sha256 != expected_file_sha256 or observed_lock != expected_lock:
        raise LaunchError("runtime lock file changed while KerSor was running")


def _emit(value: object, *, stream: Any = sys.stdout) -> None:
    print(canonical_json_bytes(value).decode("utf-8"), file=stream)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--auth-home", type=Path, required=True)
    parser.add_argument("--mission", type=Path)
    parser.add_argument("--runtime-config", type=Path)
    parser.add_argument("--codex")
    parser.add_argument("--node")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--admit-only", action="store_true")
    mode.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _postcheck(
    *,
    arguments: argparse.Namespace,
    lock: dict[str, Any],
    lock_file_sha256: str,
    auth_home: Path,
    forbidden_roots: list[Path],
    require_pristine: bool = False,
) -> None:
    _assert_lock_file_unchanged(arguments.lock, lock, lock_file_sha256)
    validate_runtime_lock(
        lock,
        mission_path=arguments.mission,
        runtime_config_path=arguments.runtime_config,
        codex_command=arguments.codex,
        node_command=arguments.node,
    )
    validate_auth_home(auth_home, forbidden_roots=forbidden_roots)
    verify_frozen_admission(lock, require_pristine=require_pristine)


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        lock, lock_file_sha256 = load_runtime_lock(arguments.lock)
        verification = validate_runtime_lock(
            lock,
            mission_path=arguments.mission,
            runtime_config_path=arguments.runtime_config,
            codex_command=arguments.codex,
            node_command=arguments.node,
        )
        root = Path(lock["kersor"]["root"])
        workspace = Path(lock["mission_binding"]["workspace"])
        session = Path(lock["mission_binding"]["session"])
        run_dir = Path(lock["mission_binding"]["run_dir"])
        forbidden = [root, workspace, session, run_dir]
        auth_home = validate_auth_home(arguments.auth_home, forbidden_roots=forbidden)
        command_read_scope = lock["kersor"]["auth_custody"][
            "command_read_scope_mechanism"
        ]
        mode = "admit-only" if arguments.admit_only else "resume"
        if mode == "admit-only" and (run_dir.exists() or run_dir.is_symlink()):
            raise LaunchError(
                "--admit-only refuses an existing run; choose --resume explicitly"
            )
        if mode == "resume":
            verify_frozen_admission(lock, require_pristine=True)
        evolve_argv = exact_evolve_argv(
            lock, mode="admit" if mode == "admit-only" else "resume"
        )
        _assert_lock_file_unchanged(arguments.lock, lock, lock_file_sha256)

        if arguments.dry_run:
            environment = build_sanitized_environment(
                lock=lock,
                auth_home=auth_home,
                neutral_home=Path("/nonexistent/apxinf-kersor-launch-home"),
            )
            _emit(
                {
                    "passed": True,
                    "contract": "apxinf-locked-kersor-dry-run-receipt-v2",
                    "dry_run": True,
                    "mode": mode,
                    "lock_sha256": verification["lock_sha256"],
                    "lock_file_sha256": lock_file_sha256,
                    "kersor_root": verification["kersor_root"],
                    "mission_sha256": verification["mission_sha256"],
                    "runtime_config_sha256": verification["runtime_config_sha256"],
                    "auth_custody_mechanism": AUTH_CUSTODY_MECHANISM,
                    "command_read_scope_mechanism": command_read_scope,
                    "agent_started": False,
                    "argv": evolve_argv,
                    "environment": {
                        "keys": sorted(environment),
                        "ambient_allowlist": [],
                        "ambient_values_copied": False,
                        "codex": environment["KERSOR_CODEX_COMMAND"],
                        "node": environment["KERSOR_NODE_BIN"],
                        "private_pycache": environment["PYTHONPYCACHEPREFIX"],
                        "accelerator_activity": environment[
                            "KERSOR_ACCELERATOR_ACTIVITY"
                        ],
                    },
                }
            )
            return 0

        with tempfile.TemporaryDirectory(prefix="apxinf-kersor-home-") as temporary:
            neutral_home = Path(temporary).resolve(strict=True)
            for name in ("tmp", "pycache"):
                (neutral_home / name).mkdir(mode=0o700)
            environment = build_sanitized_environment(
                lock=lock, auth_home=auth_home, neutral_home=neutral_home
            )
            if mode == "admit-only":
                try:
                    admitted = subprocess.run(
                        evolve_argv,
                        stdin=subprocess.DEVNULL,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        check=False,
                        cwd=workspace,
                        env=environment,
                    )
                except OSError as exc:
                    raise LaunchError(
                        f"failed to execute KerSor admission: {exc}"
                    ) from exc
                if admitted.returncode != 0:
                    raise LaunchError(
                        f"KerSor admission failed closed with return code {admitted.returncode}"
                    )
                receipt = _parse_admission_stdout(admitted.stdout)
                frozen = verify_frozen_admission(lock, admission_receipt=receipt)
                _postcheck(
                    arguments=arguments,
                    lock=lock,
                    lock_file_sha256=lock_file_sha256,
                    auth_home=auth_home,
                    forbidden_roots=forbidden,
                    require_pristine=True,
                )
                _emit(
                    {
                        "passed": True,
                        "contract": "apxinf-locked-kersor-admission-receipt-v1",
                        "status": "admitted",
                        "agent_started": False,
                        "mode": mode,
                        "lock_sha256": verification["lock_sha256"],
                        "lock_file_sha256": lock_file_sha256,
                        "run_dir": frozen["run_dir"],
                        "binding_sha256": frozen["binding_sha256"],
                        "frozen_verification_contract": frozen["contract"],
                        "kersor_admission_contract": receipt["contract_version"],
                        "auth_custody_mechanism": AUTH_CUSTODY_MECHANISM,
                        "command_read_scope_mechanism": command_read_scope,
                    }
                )
                return 0
            try:
                completed = subprocess.run(
                    evolve_argv,
                    stdin=None,
                    stdout=None,
                    stderr=None,
                    check=False,
                    cwd=workspace,
                    env=environment,
                )
            except OSError as exc:
                raise LaunchError(f"failed to resume locked KerSor run: {exc}") from exc

        try:
            _postcheck(
                arguments=arguments,
                lock=lock,
                lock_file_sha256=lock_file_sha256,
                auth_home=auth_home,
                forbidden_roots=forbidden,
            )
        except RuntimeLockError as exc:
            _emit(
                {
                    "passed": False,
                    "error": {"code": "KERSOR_POST_RUN_DRIFT", "message": str(exc)},
                    "child_returncode": completed.returncode,
                },
                stream=sys.stderr,
            )
            return 78
        return completed.returncode
    except RuntimeLockError as exc:
        _emit(
            {
                "passed": False,
                "error": {"code": "KERSOR_LOCKED_LAUNCH_REFUSED", "message": str(exc)},
            }
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
