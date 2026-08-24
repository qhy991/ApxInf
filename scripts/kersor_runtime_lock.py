#!/usr/bin/env python3
"""Create and verify the fail-closed ApxInf/KerSor runtime lock v2.

Creation may perform bounded identity probes (Codex ``--version`` and macOS
Mach-O dependency discovery).  Verification never executes a locked program:
it re-opens every bound file with ``O_NOFOLLOW`` and hashes that file descriptor.
Credentials are deliberately outside this module's byte-reading surface.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
from typing import Any, Iterable
import re


LOCK_CONTRACT = "apxinf-kersor-runtime-lock-v2"
LOCK_SCHEMA_VERSION = 2
LEGACY_LOCK_CONTRACT = "apxinf-kersor-runtime-lock-v1"
HEX_DIGITS = frozenset("0123456789abcdef")
SCRIPT_SUFFIXES = frozenset({".py", ".sh", ".js", ".mjs"})
SAFE_MISSION_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
REQUIRED_CLOSURE_PATHS = (
    "scripts/evolve.sh",
    "scripts/run-autonomous-workflow.py",
    "scripts/dispatch-workflow.sh",
    "scripts/verify-autonomous-run.py",
    "config/default_config.json",
    "docs/autonomous-workflow-runtime.md",
    "skills/kersor-evolve/SKILL.md",
    "skills/kersor-protocol/SKILL.md",
)
APXINF_RUNTIME_PATHS = (
    "scripts/kersor_runtime_lock.py",
    "scripts/run_locked_kersor_mission.py",
    "scripts/prepare_hf_macos_intake.py",
)
EXECUTOR_RUNTIME_PATHS = (
    "config/default_config.json",
    "runtime/workflow-host.mjs",
    "runtime/brokers/claude-code-exec.mjs",
    "runtime/brokers/codex-exec.mjs",
    "runtime/brokers/dsh-host-rpc.mjs",
    "runtime/brokers/index.mjs",
    "runtime/brokers/kernelowl-exec.mjs",
    "runtime/brokers/pi-exec.mjs",
    "runtime/evaluators/command.mjs",
    "runtime/evaluators/index.mjs",
    "runtime/evaluators/sol-execbench.mjs",
    "scripts/accelerator-activity.py",
    "scripts/dispatch-workflow.sh",
    "scripts/kersor-config.sh",
)
ALWAYS_LIVE_PREFIXES = ("kersor_core/",)
ALWAYS_LIVE_PATHS = (
    "scripts/evolve.sh",
    "scripts/run-autonomous-workflow.py",
)
AUTH_CUSTODY_MARKER_NAME = "CODEX_AUTH_CUSTODY_MECHANISM"
AUTH_CUSTODY_MECHANISM = "codex-named-permissions-auth-read-deny-v2"
COMMAND_READ_SCOPE_MARKER_NAME = "CODEX_COMMAND_READ_SCOPE_MECHANISM"
COMMAND_READ_SCOPE_MECHANISM = "codex-minimal-project-read-v1"
SYSTEM_MACHO_PREFIXES = ("/usr/lib/", "/System/Library/")
HOST_PYTHON_CLOSURE_SCOPE = {
    "executable": "sha256-file-record",
    "non_system_macho": "recursive-sha256-file-records",
    "system_macho": "platform-tcb",
    "python_stdlib": "platform-tcb",
    "site_initialization": "disabled-by--S",
    "bytecode_writes": "disabled-by--B",
}
# Auth custody is a parent-Codex boundary.  No runtime-supplied Codex argument
# is accepted there; the outer launcher supplies the complete child env.
SAFE_CODEX_EXTRA_ARGS: list[str] = []
ALLOWED_RUNTIME_TOP_LEVEL = frozenset(
    {"contract_version", "max_concurrency", "budget", "broker"}
)
ALLOWED_BROKER_FIELDS = frozenset(
    {
        "type",
        "command",
        "sandbox",
        "ephemeral",
        "approval_policy",
        "skip_git_repo_check",
        "disable_nested_agents",
        "default_model",
        "default_reasoning_effort",
        "worker_preamble",
        "timeout_seconds",
        "extra_args",
        "model_roles",
        "model_aliases",
    }
)
ALLOWED_HOST_EVALUATOR_REQUEST_FIELDS = frozenset(
    {
        "protocol",
        "filesystem_policy",
        "network_policy",
        "output_policy",
        "argv",
        "cwd",
        "artifacts",
        "timeout_seconds",
        "max_output_bytes",
        "materialize",
    }
)
FORBIDDEN_READ_ONLY_AGENT_FIELDS = frozenset(
    {"transaction_artifacts", "commit_failed_outputs", "candidate_verifier"}
)
ALLOWED_HOST_EVALUATOR_EXECUTION_FIELDS = frozenset(
    {
        "kind",
        "retryable",
        "request",
        "input_artifact_field",
        "fact_projections",
    }
)
FIXED_INTAKE_HOST_EVALUATOR_NAMES = (
    "verify_source_lock",
    "validate_port_manifest",
)
FIXED_HOST_EVALUATOR_REQUEST_FIELDS = frozenset(
    {
        "protocol",
        "filesystem_policy",
        "network_policy",
        "output_policy",
        "argv",
        "cwd",
        "artifacts",
        "timeout_seconds",
        "max_output_bytes",
    }
)
FIXED_SOURCE_FACT_PROJECTIONS = [
    {"output_name": "source_lock_valid", "result_path": "passed"}
]
FIXED_MANIFEST_FACT_PROJECTIONS = [
    {"output_name": "port_manifest_valid", "result_path": "passed"},
    {"output_name": "route_verified", "result_path": "passed"},
    {"output_name": "decision_complete", "result_path": "passed"},
]


class RuntimeLockError(ValueError):
    """Raised when a runtime identity cannot be frozen or revalidated."""


def canonical_json_bytes(value: object) -> bytes:
    """Return the single canonical encoding used by the lock contract."""

    try:
        rendered = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise RuntimeLockError(f"value is not canonical JSON: {exc}") from exc
    return rendered.encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    return str(file_record(path)["sha256"])


def read_file_bytes(path: Path, *, label: str) -> bytes:
    canonical = _regular_file(path, label=label)
    payload, _, _ = _read_fd(canonical, label=label, retain_payload=True)
    assert payload is not None
    return payload


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise RuntimeLockError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise RuntimeLockError(f"non-finite JSON number is forbidden: {value}")


def parse_json(payload: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            payload,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except RuntimeLockError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeLockError(f"{label} is not valid UTF-8 JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeLockError(f"{label} must contain one JSON object")
    return value


def _regular_file(path: Path, *, label: str, executable: bool = False) -> Path:
    expanded = path.expanduser()
    try:
        entry = expanded.lstat()
    except OSError as exc:
        raise RuntimeLockError(f"{label} is unavailable at {expanded}: {exc}") from exc
    if stat.S_ISLNK(entry.st_mode) or not stat.S_ISREG(entry.st_mode):
        raise RuntimeLockError(
            f"{label} must be a regular non-symlink file: {expanded}"
        )
    canonical = expanded.resolve(strict=True)
    if executable and not os.access(canonical, os.X_OK):
        raise RuntimeLockError(f"{label} is not executable: {canonical}")
    return canonical


def _canonical_directory(path: Path, *, label: str) -> Path:
    expanded = path.expanduser()
    try:
        canonical = expanded.resolve(strict=True)
    except OSError as exc:
        raise RuntimeLockError(f"{label} is unavailable at {expanded}: {exc}") from exc
    if not canonical.is_dir():
        raise RuntimeLockError(f"{label} is not a directory: {canonical}")
    return canonical


def file_record(path: Path, *, relative_to: Path | None = None) -> dict[str, object]:
    canonical = _regular_file(path, label="locked file")
    if relative_to is None:
        rendered_path = str(canonical)
    else:
        try:
            rendered_path = canonical.relative_to(relative_to).as_posix()
        except ValueError as exc:
            raise RuntimeLockError(
                f"locked file escapes canonical KerSor root: {canonical}"
            ) from exc
    _, digest, metadata = _read_fd(canonical, label="locked file", retain_payload=False)
    return {
        "path": rendered_path,
        "sha256": digest,
        "size": metadata.st_size,
        "mode": stat.S_IMODE(metadata.st_mode),
        "dev": metadata.st_dev,
        "ino": metadata.st_ino,
        "uid": metadata.st_uid,
        "gid": metadata.st_gid,
        "nlink": metadata.st_nlink,
    }


def _read_fd(
    path: Path, *, label: str, retain_payload: bool
) -> tuple[bytes | None, str, os.stat_result]:
    """Read/hash one direct file through an ``O_NOFOLLOW`` descriptor."""

    nofollow = getattr(os, "O_NOFOLLOW", 0)
    if nofollow == 0:
        raise RuntimeLockError("O_NOFOLLOW is required by runtime lock v2")
    flags = os.O_RDONLY | nofollow | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise RuntimeLockError(
            f"cannot open {label} without following links: {path}: {exc}"
        ) from exc
    digest = hashlib.sha256()
    chunks: list[bytes] | None = [] if retain_payload else None
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise RuntimeLockError(f"{label} is not a regular file: {path}")
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            if chunks is not None:
                chunks.append(chunk)
        after = os.fstat(descriptor)
    except OSError as exc:
        raise RuntimeLockError(f"cannot read {label} at {path}: {exc}") from exc
    finally:
        os.close(descriptor)
    identity_fields = (
        "st_dev",
        "st_ino",
        "st_size",
        "st_mode",
        "st_uid",
        "st_gid",
        "st_nlink",
        "st_mtime_ns",
        "st_ctime_ns",
    )
    if any(getattr(before, key) != getattr(after, key) for key in identity_fields):
        raise RuntimeLockError(
            f"{label} changed while its descriptor was hashed: {path}"
        )
    try:
        pathname = path.lstat()
    except OSError as exc:
        raise RuntimeLockError(
            f"{label} pathname disappeared after hashing: {path}: {exc}"
        ) from exc
    if stat.S_ISLNK(pathname.st_mode) or (
        pathname.st_dev,
        pathname.st_ino,
        pathname.st_size,
        pathname.st_mode,
        pathname.st_uid,
        pathname.st_gid,
        pathname.st_nlink,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mode,
        after.st_uid,
        after.st_gid,
        after.st_nlink,
    ):
        raise RuntimeLockError(f"{label} pathname changed while it was hashed: {path}")
    return (b"".join(chunks) if chunks is not None else None, digest.hexdigest(), after)


def _trusted_path(extra: Iterable[Path] = ()) -> str:
    candidates = [
        *(str(path) for path in extra),
        str(Path(sys.executable).resolve().parent),
        "/opt/homebrew/bin",
        "/usr/local/bin",
        "/usr/bin",
        "/bin",
        "/usr/sbin",
        "/sbin",
    ]
    unique: list[str] = []
    for value in candidates:
        if value not in unique and Path(value).is_dir():
            unique.append(value)
    return os.pathsep.join(unique)


def _probe_environment(extra_path: Iterable[Path] = ()) -> dict[str, str]:
    return {
        "PATH": _trusted_path(extra_path),
        "HOME": "/nonexistent/apxinf-kersor-probe-home",
        "LC_ALL": "C",
        "LANG": "C",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_TERMINAL_PROMPT": "0",
    }


def _git_command(root: Path, args: list[str]) -> subprocess.CompletedProcess[bytes]:
    command = [
        "git",
        "-c",
        "core.fsmonitor=false",
        "-c",
        "core.untrackedCache=false",
        "-C",
        str(root),
        *args,
    ]
    try:
        return subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=30,
            env=_probe_environment(),
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeLockError(f"cannot inspect KerSor Git identity: {exc}") from exc


def git_identity(root: Path) -> dict[str, object]:
    probe = _git_command(root, ["rev-parse", "--show-toplevel"])
    if probe.returncode != 0:
        return {
            "present": False,
            "top_level": None,
            "head": None,
            "dirty": None,
            "status_sha256": None,
            "status_entries": None,
        }

    try:
        top_level = Path(os.fsdecode(probe.stdout).strip()).resolve(strict=True)
    except (OSError, UnicodeDecodeError) as exc:
        raise RuntimeLockError(f"KerSor Git top-level is invalid: {exc}") from exc
    if top_level != root:
        raise RuntimeLockError(
            f"canonical KerSor root must equal its Git top-level: {root} != {top_level}"
        )

    head_result = _git_command(root, ["rev-parse", "--verify", "HEAD"])
    if head_result.returncode != 0:
        raise RuntimeLockError("KerSor Git repository has no resolvable HEAD")
    head = os.fsdecode(head_result.stdout).strip()
    if len(head) not in {40, 64} or any(
        character not in HEX_DIGITS for character in head
    ):
        raise RuntimeLockError(
            f"KerSor Git HEAD is not a lowercase object ID: {head!r}"
        )

    status_result = _git_command(
        root, ["status", "--porcelain=v1", "-z", "--untracked-files=all"]
    )
    if status_result.returncode != 0:
        detail = os.fsdecode(status_result.stderr[-2048:]).strip()
        raise RuntimeLockError(f"cannot inspect KerSor dirty state: {detail}")
    status_payload = status_result.stdout
    return {
        "present": True,
        "top_level": str(root),
        "head": head,
        "dirty": bool(status_payload),
        "status_sha256": sha256_bytes(status_payload),
        "status_entries": status_payload.count(b"\0"),
    }


def _plugin_manifest(root: Path) -> tuple[Path, str]:
    candidates = (
        root / ".codex-plugin/plugin.json",
        root / ".claude-plugin/plugin.json",
    )
    for candidate in candidates:
        if not candidate.exists() and not candidate.is_symlink():
            continue
        canonical = _regular_file(candidate, label="KerSor plugin manifest")
        payload = parse_json(
            read_file_bytes(canonical, label="KerSor plugin manifest"),
            label="KerSor plugin manifest",
        )
        version = payload.get("version")
        if not isinstance(version, str) or not version.strip() or len(version) > 200:
            raise RuntimeLockError("KerSor plugin manifest has no valid version")
        return canonical, version
    raise RuntimeLockError("KerSor plugin manifest was not found")


def enumerate_closure(root: Path) -> list[Path]:
    """Enumerate the complete execution/protocol closure bound by the lock."""

    selected: set[Path] = set()
    for relative in REQUIRED_CLOSURE_PATHS:
        selected.add(
            _regular_file(root / relative, label=f"KerSor closure file {relative}")
        )

    runtime_root = _canonical_directory(
        root / "runtime", label="KerSor runtime directory"
    )
    runtime_files = [
        path
        for path in runtime_root.rglob("*")
        if path.suffix in {".js", ".mjs"} and (path.is_file() or path.is_symlink())
    ]
    if not runtime_files:
        raise RuntimeLockError("KerSor runtime contains no JavaScript modules")
    selected.update(
        _regular_file(path, label="KerSor runtime module") for path in runtime_files
    )

    core_root = _canonical_directory(
        root / "kersor_core", label="KerSor core directory"
    )
    core_files = [
        path for path in core_root.rglob("*.py") if path.is_file() or path.is_symlink()
    ]
    if not core_files:
        raise RuntimeLockError("KerSor core contains no Python modules")
    selected.update(
        _regular_file(path, label="KerSor core module") for path in core_files
    )

    scripts_root = _canonical_directory(
        root / "scripts", label="KerSor scripts directory"
    )
    script_files = [
        path
        for path in scripts_root.rglob("*")
        if path.suffix in SCRIPT_SUFFIXES and (path.is_file() or path.is_symlink())
    ]
    selected.update(_regular_file(path, label="KerSor script") for path in script_files)

    config_root = _canonical_directory(root / "config", label="KerSor config directory")
    config_files = [
        path
        for path in config_root.rglob("*.json")
        if path.is_file() or path.is_symlink()
    ]
    selected.update(_regular_file(path, label="KerSor config") for path in config_files)

    manifest, _ = _plugin_manifest(root)
    selected.add(manifest)
    return sorted(selected, key=lambda path: path.relative_to(root).as_posix())


def closure_records(root: Path) -> list[dict[str, object]]:
    return [file_record(path, relative_to=root) for path in enumerate_closure(root)]


def _resolve_executable(command: str | Path) -> Path:
    rendered = os.fspath(command)
    if not rendered.strip():
        raise RuntimeLockError("Codex executable is empty")
    if os.sep in rendered:
        candidate = Path(rendered)
    else:
        located = shutil.which(rendered, path=_trusted_path())
        if located is None:
            raise RuntimeLockError(f"Codex executable was not found: {rendered}")
        candidate = Path(located)
    try:
        canonical = candidate.expanduser().resolve(strict=True)
    except OSError as exc:
        raise RuntimeLockError(
            f"Codex executable cannot be canonicalized: {exc}"
        ) from exc
    if not canonical.is_file() or not os.access(canonical, os.X_OK):
        raise RuntimeLockError(
            f"Codex executable is not a regular executable: {canonical}"
        )
    return canonical


def _macho_non_system_closure(executable: Path) -> list[dict[str, object]]:
    """Bind every non-system dylib reported by macOS ``otool -L`` recursively."""

    if sys.platform != "darwin":
        return []
    otool = Path("/usr/bin/otool")
    if not otool.is_file():
        raise RuntimeLockError("macOS runtime closure requires /usr/bin/otool")
    pending = [executable]
    observed: set[Path] = set()
    dependencies: dict[str, Path] = {}
    while pending:
        current = pending.pop()
        if current in observed:
            continue
        observed.add(current)
        try:
            completed = subprocess.run(
                [str(otool), "-L", str(current)],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                timeout=15,
                env=_probe_environment(),
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise RuntimeLockError(
                f"cannot inspect Mach-O dependencies for {current}: {exc}"
            ) from exc
        if completed.returncode != 0:
            # Shell/Python scripts are valid locked commands but not Mach-O.
            if current == executable:
                return []
            detail = os.fsdecode(completed.stderr[-1024:]).strip()
            raise RuntimeLockError(
                f"cannot inspect Mach-O dependency {current}: {detail}"
            )
        for raw_line in os.fsdecode(completed.stdout).splitlines()[1:]:
            value = raw_line.strip().split(" (", 1)[0]
            if not value or value.startswith(SYSTEM_MACHO_PREFIXES):
                continue
            if value.startswith("@loader_path/"):
                declared_dependency = current.parent / value.removeprefix(
                    "@loader_path/"
                )
            elif value.startswith("@"):
                raise RuntimeLockError(
                    f"unresolved non-system Mach-O dependency for {current}: {value}"
                )
            else:
                declared_dependency = Path(value)
            try:
                physical_candidate = declared_dependency.resolve(strict=True)
            except OSError as exc:
                raise RuntimeLockError(
                    f"cannot resolve Mach-O dependency {value}: {exc}"
                ) from exc
            dependency = _regular_file(physical_candidate, label="Mach-O dependency")
            dependencies[f"{current}\0{value}"] = dependency
            if dependency not in observed:
                pending.append(dependency)
    return [
        {
            "owner": key.split("\0", 1)[0],
            "load_path": key.split("\0", 1)[1],
            "payload": file_record(path),
        }
        for key, path in sorted(dependencies.items())
    ]


def _codex_native_from_entrypoint(entrypoint: Path) -> Path:
    if entrypoint.suffix != ".js":
        return entrypoint
    package_root = entrypoint.parent.parent
    candidates = sorted(
        (
            candidate.resolve(strict=True)
            for candidate in (package_root / "node_modules/@openai").glob(
                "codex-*/vendor/*/bin/codex"
            )
            if candidate.is_file() and os.access(candidate, os.X_OK)
        ),
        key=str,
    )
    if len(candidates) != 1:
        raise RuntimeLockError(
            "Codex JavaScript entrypoint must resolve to exactly one packaged native payload"
        )
    return candidates[0]


def codex_identity(command: str | Path) -> dict[str, object]:
    entrypoint = _resolve_executable(command)
    executable = _codex_native_from_entrypoint(entrypoint)
    before = file_record(executable)
    try:
        completed = subprocess.run(
            [str(executable), "--version"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=15,
            env=_probe_environment([executable.parent]),
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeLockError(f"cannot query Codex version: {exc}") from exc
    if completed.returncode != 0:
        detail = os.fsdecode((completed.stderr or completed.stdout)[-2048:]).strip()
        raise RuntimeLockError(f"Codex --version failed: {detail}")
    version = os.fsdecode(completed.stdout).strip()
    if not version or "\n" in version or len(version) > 1024:
        raise RuntimeLockError(
            "Codex --version must return one bounded non-empty stdout line"
        )
    record = file_record(executable)
    if record != before:
        raise RuntimeLockError("Codex executable changed while its version was queried")
    return {
        **record,
        "version": version,
        "entrypoint": file_record(entrypoint),
        "native_payload": True,
        "dynamic_libraries": _macho_non_system_closure(executable),
    }


def node_identity(command: str | Path = "node") -> dict[str, object]:
    executable = _resolve_executable(command)
    return {
        "executable": file_record(executable),
        "dynamic_libraries": _macho_non_system_closure(executable),
    }


def command_identities() -> dict[str, dict[str, object]]:
    names = (
        "bash",
        "jq",
        "realpath",
        "sha256sum",
        "awk",
        "mktemp",
        "cp",
        "dirname",
        "mkdir",
        "rm",
    )
    return {name: file_record(_resolve_executable(name)) for name in names}


def _resolve_json_path(value: object, *, base: Path, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise RuntimeLockError(f"{label} must be a non-empty path")
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = base / candidate
    return candidate.resolve(strict=True)


def semantic_json_sha256(value: object) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def validate_runtime_config_policy(payload: dict[str, Any]) -> dict[str, object]:
    if set(payload) - ALLOWED_RUNTIME_TOP_LEVEL:
        raise RuntimeLockError("runtime config has unknown top-level fields")
    if payload.get("contract_version") != "akw-js-runtime-v1":
        raise RuntimeLockError(
            "runtime config contract_version must be akw-js-runtime-v1"
        )
    broker = payload.get("broker")
    if not isinstance(broker, dict):
        raise RuntimeLockError("runtime config has no broker object")
    unknown = set(broker) - ALLOWED_BROKER_FIELDS
    if unknown:
        raise RuntimeLockError(
            f"runtime config broker has forbidden/unknown fields: {sorted(unknown)}"
        )
    required = {
        "type": "codex-exec",
        "sandbox": "read-only",
        "approval_policy": "never",
        "ephemeral": True,
        "disable_nested_agents": True,
    }
    for key, expected in required.items():
        if broker.get(key) != expected:
            raise RuntimeLockError(
                f"runtime config broker.{key} must equal {expected!r}"
            )
    if broker.get("command") not in {None, "codex"}:
        raise RuntimeLockError(
            "runtime config broker.command must be the outer-bound codex alias"
        )
    if broker.get("extra_args") != SAFE_CODEX_EXTRA_ARGS:
        raise RuntimeLockError(
            "runtime config auth custody requires broker.extra_args=[]"
        )
    model_roles = broker.get("model_roles", {})
    if not isinstance(model_roles, dict):
        raise RuntimeLockError("runtime config broker.model_roles must be an object")
    for role, settings in model_roles.items():
        if not isinstance(role, str) or not isinstance(settings, dict):
            raise RuntimeLockError("runtime config model role is malformed")
        if settings.get("profile") is not None:
            raise RuntimeLockError(
                "runtime config Codex profiles are forbidden under auth custody"
            )
    encoded = canonical_json_bytes(payload).decode("utf-8").lower()
    for forbidden in (
        "danger-full-access",
        "workspace-write",
        "outer_sandbox",
        "outer-sandbox",
        "additional_dirs",
        "additionaldirs",
        "add_dirs",
        "add-dir",
        "sandbox_preflight",
        "bypass-approvals",
        "approve-for-me",
        "default_permissions",
        "sandbox_mode",
    ):
        if forbidden in encoded:
            raise RuntimeLockError(
                f"runtime config contains forbidden authority selector: {forbidden}"
            )
    return {
        "sandbox": "read-only",
        "approval_policy": "never",
        "environment_inheritance": "none",
        "additional_writable_directories": [],
        "outer_sandbox": None,
    }


def _host_evaluator_argv(mission_payload: dict[str, Any]) -> list[dict[str, Any]]:
    capabilities = mission_payload.get("capabilities")
    if not isinstance(capabilities, list):
        raise RuntimeLockError("Mission capabilities must be an array")
    evaluators: list[dict[str, Any]] = []
    capability_names: set[str] = set()
    for capability in capabilities:
        if not isinstance(capability, dict):
            raise RuntimeLockError("Mission capability must be an object")
        name = capability.get("name")
        if not isinstance(name, str) or not name:
            raise RuntimeLockError("Mission capability must have a non-empty name")
        if name in capability_names:
            raise RuntimeLockError(f"Mission capability name is duplicated: {name}")
        capability_names.add(name)
        execution = capability.get("execution")
        if "execution" in capability and not isinstance(execution, dict):
            raise RuntimeLockError("Mission capability execution must be an object")
        kind = execution.get("kind") if isinstance(execution, dict) else "agent"
        if kind not in {"agent", "host_evaluator"}:
            raise RuntimeLockError(
                f"Mission capability has unknown execution kind: {kind}"
            )
        if kind == "agent":
            forbidden = set(capability) & FORBIDDEN_READ_ONLY_AGENT_FIELDS
            if capability.get("side_effect", "none") != "none":
                forbidden.add("side_effect")
            if forbidden:
                raise RuntimeLockError(
                    "read-only Agent capability contains transaction/mutation fields: "
                    f"{sorted(forbidden)}"
                )
            if isinstance(execution, dict) and set(execution) != {"kind"}:
                raise RuntimeLockError(
                    "read-only Agent capability execution may only declare kind=agent"
                )
            continue
        if capability.get("side_effect", "none") != "none":
            raise RuntimeLockError("Host evaluator side_effect must equal 'none'")
        assert isinstance(execution, dict)
        if name not in FIXED_INTAKE_HOST_EVALUATOR_NAMES:
            raise RuntimeLockError(
                "Host evaluator is not one of the two fixed intake evaluators"
            )
        expected_execution_fields = {
            "kind",
            "retryable",
            "request",
            "fact_projections",
        }
        if name == "validate_port_manifest":
            expected_execution_fields.add("input_artifact_field")
        if set(execution) != expected_execution_fields:
            raise RuntimeLockError(
                f"fixed intake evaluator execution fields drifted: {name}"
            )
        unknown_execution_fields = (
            set(execution) - ALLOWED_HOST_EVALUATOR_EXECUTION_FIELDS
        )
        if unknown_execution_fields:
            raise RuntimeLockError(
                "Host evaluator execution has forbidden/unknown fields: "
                f"{sorted(unknown_execution_fields)}"
            )
        if execution.get("retryable") is not False:
            raise RuntimeLockError("Host evaluator retryable must equal false")
        request = execution.get("request")
        if not isinstance(request, dict):
            raise RuntimeLockError("Host evaluator request must be an object")
        unknown_request_fields = set(request) - ALLOWED_HOST_EVALUATOR_REQUEST_FIELDS
        if unknown_request_fields:
            raise RuntimeLockError(
                "Host evaluator request has forbidden/unknown fields: "
                f"{sorted(unknown_request_fields)}"
            )
        required_policy = {
            "protocol": "command-v1",
            "filesystem_policy": "read-only",
            "network_policy": "denied",
            "output_policy": "sealed",
        }
        for field, expected in required_policy.items():
            if request.get(field) != expected:
                raise RuntimeLockError(
                    f"Host evaluator {field} must equal {expected!r}"
                )
        materialize = request.get("materialize", [])
        if not isinstance(materialize, list) or materialize:
            raise RuntimeLockError("Host evaluator materialize must be absent or empty")
        if request.get("cwd", ".") != ".":
            raise RuntimeLockError(
                "Host evaluator cwd must be absent or the exact ApxInf project root selector '.'"
            )
        if set(request) != FIXED_HOST_EVALUATOR_REQUEST_FIELDS:
            raise RuntimeLockError(
                f"fixed intake evaluator request fields drifted: {name}"
            )
        if request.get("timeout_seconds") != 60:
            raise RuntimeLockError(
                f"fixed intake evaluator timeout_seconds drifted: {name}"
            )
        if request.get("max_output_bytes") != 65536:
            raise RuntimeLockError(
                f"fixed intake evaluator max_output_bytes drifted: {name}"
            )
        artifacts = request.get("artifacts")
        if name == "validate_port_manifest":
            if artifacts != []:
                raise RuntimeLockError(
                    "validate_port_manifest artifacts must be exactly empty"
                )
        elif (
            not isinstance(artifacts, list)
            or len(artifacts) != 1
            or not isinstance(artifacts[0], str)
            or not artifacts[0]
        ):
            raise RuntimeLockError(
                "verify_source_lock artifacts must contain its exact source lock"
            )
        argv = request.get("argv") if isinstance(request, dict) else None
        if (
            not isinstance(argv, list)
            or len(argv) < 4
            or not all(isinstance(value, str) for value in argv)
            or not argv[0]
            or not argv[3]
        ):
            raise RuntimeLockError("Host evaluator argv must be a bounded string array")
        if argv[1:3] != ["-S", "-B"]:
            raise RuntimeLockError(
                "Host evaluator Python argv must use exact '-S -B' isolation flags"
            )
        if len(argv) > 128 or sum(len(value) for value in argv) > 64 * 1024:
            raise RuntimeLockError("Host evaluator argv exceeds the lock bound")
        expected_projections = (
            FIXED_MANIFEST_FACT_PROJECTIONS
            if name == "validate_port_manifest"
            else FIXED_SOURCE_FACT_PROJECTIONS
        )
        if execution.get("fact_projections") != expected_projections:
            raise RuntimeLockError(
                f"fixed intake evaluator fact projections drifted: {name}"
            )
        input_artifact_field = execution.get("input_artifact_field")
        if name == "validate_port_manifest":
            if (
                input_artifact_field != "argv.7"
                or len(argv) <= 7
                or argv[6] != "--json"
                or argv[7] != ""
            ):
                raise RuntimeLockError(
                    "validate_port_manifest input_artifact_field must bind the exact empty argv.7 after --json"
                )
        elif "input_artifact_field" in execution:
            raise RuntimeLockError(
                "input_artifact_field is only allowed for validate_port_manifest"
            )
        evaluators.append(
            {
                "name": name,
                "argv": argv,
                "script": argv[3],
                "request": request,
                "input_artifact_field": input_artifact_field,
            }
        )
    if not evaluators:
        raise RuntimeLockError("Mission must contain at least one Host evaluator")
    if [item["name"] for item in evaluators] != list(FIXED_INTAKE_HOST_EVALUATOR_NAMES):
        raise RuntimeLockError(
            "Mission must contain exactly the two ordered fixed intake Host evaluators"
        )
    return evaluators


def _validate_fixed_intake_evaluator_requests(
    evaluators: list[dict[str, Any]],
    *,
    workspace: Path,
    host_python: Path,
    source_lock: Path,
    source_payload: dict[str, Any],
) -> None:
    try:
        source_relative = source_lock.relative_to(workspace).as_posix()
    except ValueError as exc:
        raise RuntimeLockError("fixed intake source lock escapes ApxInf") from exc
    identity_fields = {
        field: source_payload.get(field)
        for field in (
            "repo_id",
            "requested_revision",
            "resolved_commit",
            "content_sha256",
        )
    }
    if not all(isinstance(value, str) and value for value in identity_fields.values()):
        raise RuntimeLockError("fixed intake source-lock identity is malformed")
    python = str(host_python)
    source_argv = [
        python,
        "-S",
        "-B",
        str(workspace / "scripts/resolve_hf_source.py"),
        "--verify",
        str(source_lock),
        "--expected-sha256",
        identity_fields["content_sha256"],
    ]
    manifest_argv = [
        python,
        "-S",
        "-B",
        str(workspace / "scripts/validate_hf_port_manifest.py"),
        "--workspace",
        str(workspace),
        "--json",
        "",
        "--source-lock",
        str(source_lock),
        "--deployment-profile",
        str(workspace / "configs/hf-onboarding/qwen35-0.8b-macos-cpu.json"),
        "--expected-repo-id",
        identity_fields["repo_id"],
        "--expected-requested-revision",
        identity_fields["requested_revision"],
        "--expected-resolved-commit",
        identity_fields["resolved_commit"],
        "--expected-source-lock-content-sha256",
        identity_fields["content_sha256"],
        "--require-ready-existing",
    ]
    common_request = {
        "protocol": "command-v1",
        "filesystem_policy": "read-only",
        "network_policy": "denied",
        "output_policy": "sealed",
        "cwd": ".",
        "timeout_seconds": 60,
        "max_output_bytes": 65536,
    }
    expected_requests = {
        "verify_source_lock": {
            **common_request,
            "argv": source_argv,
            "artifacts": [source_relative],
        },
        "validate_port_manifest": {
            **common_request,
            "argv": manifest_argv,
            "artifacts": [],
        },
    }
    observed = {item["name"]: item for item in evaluators}
    for name, expected_request in expected_requests.items():
        if observed[name]["request"] != expected_request:
            raise RuntimeLockError(
                f"fixed intake evaluator request/argv binding drifted: {name}"
            )


def _python_local_imports(entrypoint: Path, workspace: Path) -> list[Path]:
    """Resolve transitive workspace-local Python imports without executing code."""

    pending = [entrypoint]
    observed: set[Path] = set()
    while pending:
        current = pending.pop()
        if current in observed:
            continue
        current = _regular_file(current, label="Host evaluator Python script")
        try:
            current.relative_to(workspace)
        except ValueError as exc:
            raise RuntimeLockError(
                f"Host evaluator script escapes the ApxInf workspace: {current}"
            ) from exc
        observed.add(current)
        try:
            tree = ast.parse(
                read_file_bytes(current, label="Host evaluator Python script").decode(
                    "utf-8"
                ),
                filename=str(current),
            )
        except (UnicodeDecodeError, SyntaxError) as exc:
            raise RuntimeLockError(
                f"cannot statically parse Host evaluator {current}: {exc}"
            ) from exc
        modules: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                modules.add(node.module)
        for module in modules:
            relative = Path(*module.split("."))
            candidates = (
                workspace / f"{relative}.py",
                workspace / relative / "__init__.py",
                workspace / "scripts" / f"{relative}.py",
            )
            for candidate in candidates:
                if candidate.exists() and not candidate.is_symlink():
                    resolved = candidate.resolve(strict=True)
                    if resolved not in observed:
                        pending.append(resolved)
                    break
    return sorted(observed, key=str)


def host_evaluator_identity(
    mission_payload: dict[str, Any],
    workspace: Path,
    host_python: Path,
    *,
    source_lock: Path,
    source_payload: dict[str, Any],
) -> list[dict[str, Any]]:
    identities: list[dict[str, Any]] = []
    evaluators = _host_evaluator_argv(mission_payload)
    _validate_fixed_intake_evaluator_requests(
        evaluators,
        workspace=workspace,
        host_python=host_python,
        source_lock=source_lock,
        source_payload=source_payload,
    )
    for evaluator in evaluators:
        argv = evaluator["argv"]
        command = _resolve_executable(argv[0])
        if command != host_python or argv[0] != str(host_python):
            raise RuntimeLockError(
                f"Host evaluator {evaluator['name']} does not use the exact locked Host Python path"
            )
        declared_script = Path(evaluator["script"])
        if not declared_script.is_absolute():
            declared_script = workspace / declared_script
        declared_script = declared_script.absolute()
        if evaluator["script"] != str(declared_script):
            raise RuntimeLockError(
                "Host evaluator script argv must use its exact absolute path"
            )
        try:
            script_entry = declared_script.lstat()
        except OSError as exc:
            raise RuntimeLockError(
                f"Host evaluator script is unavailable: {declared_script}: {exc}"
            ) from exc
        if (
            stat.S_ISLNK(script_entry.st_mode)
            or not stat.S_ISREG(script_entry.st_mode)
            or script_entry.st_nlink != 1
        ):
            raise RuntimeLockError(
                "Host evaluator script must be a direct single-link regular file"
            )
        script = declared_script.resolve(strict=True)
        if script != declared_script:
            raise RuntimeLockError(
                "Host evaluator script must use its exact canonical direct path"
            )
        scripts = _python_local_imports(script, workspace)
        script_records = [file_record(path) for path in scripts]
        if any(record["nlink"] != 1 for record in script_records):
            raise RuntimeLockError(
                "Host evaluator transitive scripts must be single-link regular files"
            )
        input_files: dict[Path, dict[str, object]] = {}
        for raw_argument in argv[4:]:
            if not raw_argument:
                continue
            declared = Path(raw_argument)
            if not declared.is_absolute():
                declared = workspace / declared
            if not (declared.exists() or declared.is_symlink()):
                continue
            try:
                declared.absolute().relative_to(workspace)
            except ValueError:
                continue
            try:
                entry = declared.lstat()
            except OSError as exc:
                raise RuntimeLockError(
                    f"Host evaluator input is unavailable: {declared}: {exc}"
                ) from exc
            if stat.S_ISDIR(entry.st_mode):
                continue
            if (
                stat.S_ISLNK(entry.st_mode)
                or not stat.S_ISREG(entry.st_mode)
                or entry.st_nlink != 1
            ):
                raise RuntimeLockError(
                    "Host evaluator workspace input must be a direct single-link "
                    f"regular file: {declared}"
                )
            canonical_input = declared.resolve(strict=True)
            if canonical_input != declared.absolute():
                raise RuntimeLockError(
                    f"Host evaluator workspace input traverses a path alias: {declared}"
                )
            try:
                canonical_input.relative_to(workspace)
            except ValueError as exc:
                raise RuntimeLockError(
                    f"Host evaluator workspace input escapes ApxInf: {declared}"
                ) from exc
            input_files[canonical_input] = file_record(canonical_input)
        identities.append(
            {
                "name": evaluator["name"],
                "argv": argv,
                "request": evaluator["request"],
                "input_artifact_field": evaluator["input_artifact_field"],
                "request_semantic_sha256": semantic_json_sha256(evaluator["request"]),
                "command": file_record(command),
                "transitive_scripts": script_records,
                "input_files": [
                    input_files[path] for path in sorted(input_files, key=str)
                ],
            }
        )
    return identities


def _discover_source_lock(
    mission_payload: dict[str, Any], workspace: Path
) -> tuple[Path, dict[str, Any]]:
    matches: dict[Path, dict[str, Any]] = {}
    for evaluator in _host_evaluator_argv(mission_payload):
        for value in evaluator["argv"][2:]:
            candidate = Path(value)
            if not candidate.is_absolute():
                candidate = workspace / candidate
            try:
                canonical = candidate.resolve(strict=True)
            except OSError:
                continue
            try:
                canonical.relative_to(workspace)
            except ValueError:
                continue
            if not canonical.is_file() or canonical.suffix != ".json":
                continue
            try:
                parsed = parse_json(
                    read_file_bytes(canonical, label="candidate source lock"),
                    label="candidate source lock",
                )
            except RuntimeLockError:
                continue
            if parsed.get("format") == "apxinf-hf-source-lock-v1":
                matches[canonical] = parsed
    if len(matches) != 1:
        raise RuntimeLockError(
            "Mission Host evaluators must bind exactly one apxinf-hf-source-lock-v1"
        )
    return next(iter(matches.items()))


def _auth_custody_marker(root: Path) -> dict[str, str]:
    broker = root / "runtime/brokers/codex-exec.mjs"
    text = read_file_bytes(broker, label="KerSor Codex broker").decode("utf-8")
    pattern = re.compile(
        rf"export\s+const\s+{AUTH_CUSTODY_MARKER_NAME}\s*=\s*['\"]{re.escape(AUTH_CUSTODY_MECHANISM)}['\"]"
    )
    if pattern.search(text) is None:
        raise RuntimeLockError(
            "KerSor Codex broker lacks the required auth command-read deny marker"
        )
    read_scope_pattern = re.compile(
        rf"export\s+const\s+{COMMAND_READ_SCOPE_MARKER_NAME}\s*=\s*['\"]{re.escape(COMMAND_READ_SCOPE_MECHANISM)}['\"]"
    )
    if read_scope_pattern.search(text) is None:
        raise RuntimeLockError(
            "KerSor Codex broker lacks the required command read-scope marker"
        )
    if "codex-named-permissions-profile-v1" not in text:
        raise RuntimeLockError(
            "KerSor Codex broker lacks the named-permissions sandbox marker"
        )
    return {
        "export": AUTH_CUSTODY_MARKER_NAME,
        "mechanism": AUTH_CUSTODY_MECHANISM,
        "command_read_scope_export": COMMAND_READ_SCOPE_MARKER_NAME,
        "command_read_scope_mechanism": COMMAND_READ_SCOPE_MECHANISM,
        "sandbox_mechanism": "codex-named-permissions-profile-v1",
    }


def _bound_run_dir(session: Path, mission_id: object) -> Path:
    if not isinstance(mission_id, str) or SAFE_MISSION_ID.fullmatch(mission_id) is None:
        raise RuntimeLockError(
            "Mission mission_id must be one safe path segment of at most 128 characters"
        )
    run_root = session / "autonomous-runs"
    if run_root.exists() or run_root.is_symlink():
        try:
            root_entry = run_root.lstat()
        except OSError as exc:
            raise RuntimeLockError(
                f"Mission autonomous run root is unavailable: {exc}"
            ) from exc
        if stat.S_ISLNK(root_entry.st_mode) or not stat.S_ISDIR(root_entry.st_mode):
            raise RuntimeLockError(
                "Mission autonomous-runs root must be a regular non-symlink directory"
            )
        canonical_root = run_root.resolve(strict=True)
    else:
        canonical_root = run_root

    candidate = canonical_root / mission_id
    if candidate.exists() or candidate.is_symlink():
        try:
            run_entry = candidate.lstat()
        except OSError as exc:
            raise RuntimeLockError(
                f"Mission run directory is unavailable: {exc}"
            ) from exc
        if stat.S_ISLNK(run_entry.st_mode) or not stat.S_ISDIR(run_entry.st_mode):
            raise RuntimeLockError(
                "Mission run directory must be a regular non-symlink directory when present"
            )
        canonical_candidate = candidate.resolve(strict=True)
    else:
        canonical_candidate = candidate
    if canonical_candidate.parent != canonical_root:
        raise RuntimeLockError(
            "Mission run directory escapes its Session autonomous-runs root"
        )
    try:
        relative = canonical_candidate.relative_to(session)
    except ValueError as exc:
        raise RuntimeLockError(
            "Mission run directory escapes its canonical Session"
        ) from exc
    if relative.parts != ("autonomous-runs", mission_id):
        raise RuntimeLockError(
            "Mission run directory is not the exact locked Session child"
        )
    return canonical_candidate


def validate_mission_binding(
    mission: Path, runtime_config: Path, *, expected_workspace: Path
) -> dict[str, str]:
    payload = parse_json(
        read_file_bytes(mission, label="KerSor Mission"), label="KerSor Mission"
    )
    if payload.get("contract_version") != "kersor-mission-v1":
        raise RuntimeLockError("launcher accepts only kersor-mission-v1 contracts")
    if payload.get("runtime", "codex") != "codex":
        raise RuntimeLockError("KerSor Mission runtime must be codex")
    bound_config = _resolve_json_path(
        payload.get("runtime_config"),
        base=mission.parent,
        label="Mission runtime_config",
    )
    if bound_config != runtime_config:
        raise RuntimeLockError(
            f"Mission runtime_config differs from locked config: {bound_config} != {runtime_config}"
        )
    if payload.get("runtime_config") != str(runtime_config):
        raise RuntimeLockError(
            "Mission runtime_config must use the exact canonical locked path"
        )
    workspace = _resolve_json_path(
        payload.get("workspace"), base=mission.parent, label="Mission workspace"
    )
    if not workspace.is_dir():
        raise RuntimeLockError(f"Mission workspace is not a directory: {workspace}")
    canonical_expected_workspace = _canonical_directory(
        expected_workspace, label="ApxInf root"
    )
    if workspace != canonical_expected_workspace or payload.get("workspace") != str(
        canonical_expected_workspace
    ):
        raise RuntimeLockError(
            "Mission workspace must be the exact canonical ApxInf root"
        )
    session = _resolve_json_path(
        payload.get("session"), base=mission.parent, label="Mission session"
    )
    if not session.is_dir():
        raise RuntimeLockError(f"Mission session is not a directory: {session}")
    if payload.get("session") != str(session):
        raise RuntimeLockError("Mission session must use its exact canonical path")
    mission_contract = payload.get("mission")
    if not isinstance(mission_contract, dict):
        raise RuntimeLockError("KerSor Mission has no mission object")
    mission_id = mission_contract.get("mission_id")
    run_dir = _bound_run_dir(session, mission_id)

    config_payload = parse_json(
        read_file_bytes(runtime_config, label="KerSor runtime config"),
        label="KerSor runtime config",
    )
    validate_runtime_config_policy(config_payload)
    assert isinstance(mission_id, str)
    return {
        "workspace": str(workspace),
        "session": str(session),
        "mission_id": mission_id,
        "run_dir": str(run_dir),
    }


def validate_session_cross_binding(
    *,
    config: dict[str, Any],
    state: dict[str, Any],
    workspace: Path,
    mission_id: str,
) -> None:
    task_dir = config.get("task_dir")
    if task_dir != str(workspace):
        raise RuntimeLockError(
            "Session config task_dir must equal the exact Mission workspace"
        )
    if state.get("session_id") != mission_id:
        raise RuntimeLockError(
            "Session state session_id must equal the exact Mission mission_id"
        )


def _without_self_hash(lock: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in lock.items() if key != "lock_sha256"}


def lock_self_sha256(lock: dict[str, Any]) -> str:
    return sha256_bytes(canonical_json_bytes(_without_self_hash(lock)))


def build_runtime_lock(
    *,
    kersor_root: Path,
    mission_path: Path,
    runtime_config_path: Path,
    codex_command: str | Path,
    node_command: str | Path = "node",
    source_lock_path: Path | None = None,
    host_python_path: Path | None = None,
) -> dict[str, Any]:
    root = _canonical_directory(kersor_root, label="KerSor root")
    apxinf_root = _canonical_directory(
        Path(__file__).resolve().parents[1], label="ApxInf root"
    )
    mission = _regular_file(mission_path, label="KerSor Mission")
    runtime_config = _regular_file(runtime_config_path, label="KerSor runtime config")
    mission_record = file_record(mission)
    runtime_config_record = file_record(runtime_config)
    mission_binding = validate_mission_binding(
        mission, runtime_config, expected_workspace=apxinf_root
    )
    mission_payload = parse_json(
        read_file_bytes(mission, label="KerSor Mission"), label="KerSor Mission"
    )
    runtime_payload = parse_json(
        read_file_bytes(runtime_config, label="KerSor runtime config"),
        label="KerSor runtime config",
    )
    runtime_policy = validate_runtime_config_policy(runtime_payload)
    workspace = Path(mission_binding["workspace"])
    session = Path(mission_binding["session"])
    session_config_path = _regular_file(
        session / "session-config.json", label="Session config"
    )
    session_state_path = _regular_file(session / "state.json", label="Session state")
    session_config_payload = parse_json(
        read_file_bytes(session_config_path, label="Session config"),
        label="Session config",
    )
    session_state_payload = parse_json(
        read_file_bytes(session_state_path, label="Session state"),
        label="Session state",
    )
    validate_session_cross_binding(
        config=session_config_payload,
        state=session_state_payload,
        workspace=workspace,
        mission_id=mission_binding["mission_id"],
    )
    current_python = _resolve_executable(sys.executable)
    host_python = _resolve_executable(host_python_path or current_python)
    if host_python != current_python:
        raise RuntimeLockError(
            "locked Host Python must equal the current interpreter executable"
        )
    source_lock, source_payload = _discover_source_lock(mission_payload, workspace)
    if source_lock_path is not None:
        asserted_source_lock = _regular_file(source_lock_path, label="HF source lock")
        try:
            asserted_source_lock.relative_to(workspace)
        except ValueError as exc:
            raise RuntimeLockError(
                "HF source lock must remain inside the workspace"
            ) from exc
        if asserted_source_lock != source_lock:
            raise RuntimeLockError(
                "explicit source lock must equal the unique Mission evaluator binding"
            )
    source_content_sha = source_payload.get("content_sha256")
    if (
        not isinstance(source_content_sha, str)
        or re.fullmatch(r"[0-9a-f]{64}", source_content_sha) is None
    ):
        raise RuntimeLockError("HF source lock has no valid semantic content_sha256")
    source_body = dict(source_payload)
    del source_body["content_sha256"]
    if semantic_json_sha256(source_body) != source_content_sha:
        raise RuntimeLockError("HF source lock semantic content hash is invalid")
    evaluators = host_evaluator_identity(
        mission_payload,
        workspace,
        host_python,
        source_lock=source_lock,
        source_payload=source_payload,
    )
    if file_record(mission) != mission_record:
        raise RuntimeLockError("KerSor Mission changed while its binding was inspected")
    if file_record(runtime_config) != runtime_config_record:
        raise RuntimeLockError(
            "KerSor runtime config changed while its binding was inspected"
        )
    manifest, plugin_version = _plugin_manifest(root)
    git_before = git_identity(root)
    closure = closure_records(root)
    git_after = git_identity(root)
    if git_after != git_before:
        raise RuntimeLockError(
            "KerSor Git identity changed while its closure was hashed"
        )

    closure_paths = [str(item["path"]) for item in closure]
    always_live = sorted(
        path
        for path in closure_paths
        if path in ALWAYS_LIVE_PATHS or path.startswith(ALWAYS_LIVE_PREFIXES)
    )
    missing_fresh = sorted(set(EXECUTOR_RUNTIME_PATHS) - set(closure_paths))
    if missing_fresh:
        raise RuntimeLockError(
            f"KerSor fresh snapshot source closure is incomplete: {missing_fresh}"
        )
    if any(path not in closure_paths for path in ALWAYS_LIVE_PATHS):
        raise RuntimeLockError("KerSor always-live admission closure is incomplete")
    auth_marker = _auth_custody_marker(root)
    apxinf_files = [
        file_record(apxinf_root / relative, relative_to=apxinf_root)
        for relative in APXINF_RUNTIME_PATHS
    ]

    lock: dict[str, Any] = {
        "contract": LOCK_CONTRACT,
        "schema_version": LOCK_SCHEMA_VERSION,
        "kersor": {
            "root": str(root),
            "plugin_version": plugin_version,
            "plugin_manifest": manifest.relative_to(root).as_posix(),
            "git": git_after,
            "auth_custody": auth_marker,
            "closure": {
                "algorithm": "sha256",
                "files": closure,
            },
            "layers": {
                "always_live_admission": always_live,
                "fresh_snapshot_sources": list(EXECUTOR_RUNTIME_PATHS),
            },
        },
        "apxinf": {
            "root": str(apxinf_root),
            "files": apxinf_files,
        },
        "mission": {
            **mission_record,
            "semantic_sha256": semantic_json_sha256(mission_payload),
        },
        "runtime_config": {
            **runtime_config_record,
            "semantic_sha256": semantic_json_sha256(runtime_payload),
            "policy": runtime_policy,
        },
        "mission_binding": mission_binding,
        "session": {
            "root": str(session),
            "config": {
                **file_record(session_config_path),
                "semantic_sha256": semantic_json_sha256(session_config_payload),
            },
            "state": {
                **file_record(session_state_path),
                "semantic_sha256": semantic_json_sha256(session_state_payload),
            },
        },
        "source_lock": {
            **file_record(source_lock),
            "format": source_payload.get("format"),
            "content_sha256": source_content_sha,
            "semantic_sha256": semantic_json_sha256(source_payload),
        },
        "host_evaluators": evaluators,
        "runtime": {
            "host_python": {
                **file_record(host_python),
                "dynamic_libraries": _macho_non_system_closure(host_python),
                "closure_scope": dict(HOST_PYTHON_CLOSURE_SCOPE),
            },
            "node": node_identity(node_command),
            "codex": codex_identity(codex_command),
            "external_commands": command_identities(),
        },
    }
    lock["lock_sha256"] = lock_self_sha256(lock)
    return lock


def _locked_path(lock: dict[str, Any], section: str) -> Path:
    value = lock.get(section)
    if not isinstance(value, dict) or not isinstance(value.get("path"), str):
        raise RuntimeLockError(f"runtime lock has no valid {section}.path")
    return Path(value["path"])


FILE_RECORD_FIELDS = (
    "path",
    "sha256",
    "size",
    "mode",
    "dev",
    "ino",
    "uid",
    "gid",
    "nlink",
)


def _validate_file_record(
    record: object, *, base: Path | None = None, label: str
) -> dict[str, object]:
    if not isinstance(record, dict) or any(
        key not in record for key in FILE_RECORD_FIELDS
    ):
        raise RuntimeLockError(f"runtime lock has malformed {label} file record")
    raw_path = record.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        raise RuntimeLockError(f"runtime lock has malformed {label} path")
    candidate = Path(raw_path) if base is None else base / raw_path
    current = file_record(candidate, relative_to=base)
    locked = {key: record.get(key) for key in FILE_RECORD_FIELDS}
    if current != locked:
        raise RuntimeLockError(f"locked file identity drifted: {label}: {raw_path}")
    return current


def _validate_record_list(records: object, *, base: Path | None, label: str) -> None:
    if not isinstance(records, list):
        raise RuntimeLockError(f"runtime lock {label} must be a file-record array")
    for index, record in enumerate(records):
        _validate_file_record(record, base=base, label=f"{label}[{index}]")


def _validate_dynamic_libraries(records: object, *, label: str) -> None:
    if not isinstance(records, list):
        raise RuntimeLockError(f"runtime lock {label} must be a dylib array")
    for index, item in enumerate(records):
        if not isinstance(item, dict) or set(item) != {"owner", "load_path", "payload"}:
            raise RuntimeLockError(f"runtime lock {label}[{index}] is malformed")
        load_path = item.get("load_path")
        owner = item.get("owner")
        if not isinstance(load_path, str) or not isinstance(owner, str):
            raise RuntimeLockError(
                f"runtime lock {label}[{index}] load path is malformed"
            )
        if load_path.startswith("@loader_path/"):
            declared = Path(owner).parent / load_path.removeprefix("@loader_path/")
        elif load_path.startswith("/"):
            declared = Path(load_path)
        else:
            raise RuntimeLockError(
                f"runtime lock {label}[{index}] load path is unresolved"
            )
        try:
            current_target = declared.resolve(strict=True)
        except OSError as exc:
            raise RuntimeLockError(
                f"runtime dependency load path drifted: {load_path}: {exc}"
            ) from exc
        payload = item.get("payload")
        if not isinstance(payload, dict) or current_target != Path(
            str(payload.get("path"))
        ):
            raise RuntimeLockError(
                f"runtime dependency load path target drifted: {load_path}"
            )
        _validate_file_record(payload, label=f"{label}[{index}] payload")


def validate_runtime_lock(
    lock: dict[str, Any],
    *,
    mission_path: Path | None = None,
    runtime_config_path: Path | None = None,
    codex_command: str | Path | None = None,
    node_command: str | Path | None = None,
) -> dict[str, object]:
    if lock.get("contract") == LEGACY_LOCK_CONTRACT or lock.get("schema_version") == 1:
        raise RuntimeLockError(
            "legacy runtime lock v1 is development-only and formally fail-closed"
        )
    if set(lock) != {
        "contract",
        "schema_version",
        "kersor",
        "apxinf",
        "mission",
        "runtime_config",
        "mission_binding",
        "session",
        "source_lock",
        "host_evaluators",
        "runtime",
        "lock_sha256",
    }:
        raise RuntimeLockError("runtime lock has missing or unknown top-level fields")
    if lock.get("contract") != LOCK_CONTRACT or lock.get("schema_version") != 2:
        raise RuntimeLockError("unsupported runtime lock contract")
    observed_self_hash = lock.get("lock_sha256")
    expected_self_hash = lock_self_sha256(lock)
    if observed_self_hash != expected_self_hash:
        raise RuntimeLockError(
            f"runtime lock self hash mismatch: expected {expected_self_hash}, observed {observed_self_hash}"
        )

    kersor = lock.get("kersor")
    if not isinstance(kersor, dict) or not isinstance(kersor.get("root"), str):
        raise RuntimeLockError("runtime lock has no canonical KerSor root")
    root = _canonical_directory(Path(kersor["root"]), label="locked KerSor root")
    locked_mission = _locked_path(lock, "mission")
    locked_config = _locked_path(lock, "runtime_config")
    runtime = lock.get("runtime")
    if not isinstance(runtime, dict):
        raise RuntimeLockError("runtime lock has no runtime identity")
    codex = runtime.get("codex")
    node = runtime.get("node")
    host_python = runtime.get("host_python")
    if not isinstance(codex, dict) or not isinstance(codex.get("path"), str):
        raise RuntimeLockError("runtime lock has no native Codex payload path")
    if not isinstance(node, dict) or not isinstance(node.get("executable"), dict):
        raise RuntimeLockError("runtime lock has no Node runtime identity")
    if not isinstance(host_python, dict) or set(host_python) != {
        *FILE_RECORD_FIELDS,
        "dynamic_libraries",
        "closure_scope",
    }:
        raise RuntimeLockError("runtime lock has malformed Host Python identity")
    if host_python.get("closure_scope") != HOST_PYTHON_CLOSURE_SCOPE:
        raise RuntimeLockError("runtime lock has malformed Host Python closure scope")

    if mission_path is not None:
        asserted = _regular_file(mission_path, label="asserted Mission")
        if asserted != locked_mission:
            raise RuntimeLockError(
                f"asserted Mission path differs from lock: {asserted}"
            )
    if runtime_config_path is not None:
        asserted = _regular_file(runtime_config_path, label="asserted runtime config")
        if asserted != locked_config:
            raise RuntimeLockError(
                f"asserted runtime config path differs from lock: {asserted}"
            )
    asserted_codex = codex_command if codex_command is not None else codex["path"]
    asserted_codex_entry = _resolve_executable(asserted_codex)
    if _codex_native_from_entrypoint(asserted_codex_entry) != Path(codex["path"]):
        raise RuntimeLockError("asserted Codex executable path differs from lock")
    asserted_node = (
        node_command if node_command is not None else node["executable"].get("path")
    )
    if _resolve_executable(asserted_node) != Path(str(node["executable"].get("path"))):
        raise RuntimeLockError("asserted Node executable path differs from lock")

    # Every check below is an fd hash/stat or pure parsing.  In particular,
    # validation never re-runs Codex --version, otool, git, or any evaluator.
    closure = kersor.get("closure")
    if not isinstance(closure, dict) or closure.get("algorithm") != "sha256":
        raise RuntimeLockError("KerSor closure binding is malformed")
    locked_closure = closure.get("files")
    _validate_record_list(locked_closure, base=root, label="KerSor closure")
    if closure_records(root) != locked_closure:
        raise RuntimeLockError("KerSor closure inventory drifted")
    layers = kersor.get("layers")
    paths = {record["path"] for record in locked_closure if isinstance(record, dict)}
    expected_always_live = sorted(
        path
        for path in paths
        if path in ALWAYS_LIVE_PATHS or str(path).startswith(ALWAYS_LIVE_PREFIXES)
    )
    if layers != {
        "always_live_admission": expected_always_live,
        "fresh_snapshot_sources": list(EXECUTOR_RUNTIME_PATHS),
    }:
        raise RuntimeLockError("KerSor fresh/always-live layer binding drifted")
    if kersor.get("auth_custody") != _auth_custody_marker(root):
        raise RuntimeLockError("KerSor auth custody marker drifted")

    apxinf = lock.get("apxinf")
    if not isinstance(apxinf, dict) or not isinstance(apxinf.get("root"), str):
        raise RuntimeLockError("ApxInf runtime identity is malformed")
    apxinf_root = _canonical_directory(Path(apxinf["root"]), label="locked ApxInf root")
    expected_apx_paths = list(APXINF_RUNTIME_PATHS)
    apx_files = apxinf.get("files")
    _validate_record_list(apx_files, base=apxinf_root, label="ApxInf runtime")
    if (
        not isinstance(apx_files, list)
        or [item.get("path") for item in apx_files] != expected_apx_paths
    ):
        raise RuntimeLockError("ApxInf launcher/lock/prepare binding is incomplete")

    _validate_file_record(lock["mission"], label="Mission")
    _validate_file_record(lock["runtime_config"], label="runtime config")
    mission_payload = parse_json(
        read_file_bytes(locked_mission, label="KerSor Mission"), label="KerSor Mission"
    )
    runtime_payload = parse_json(
        read_file_bytes(locked_config, label="KerSor runtime config"),
        label="KerSor runtime config",
    )
    if lock["mission"].get("semantic_sha256") != semantic_json_sha256(mission_payload):
        raise RuntimeLockError("Mission semantic identity drifted")
    policy = validate_runtime_config_policy(runtime_payload)
    if (
        lock["runtime_config"].get("semantic_sha256")
        != semantic_json_sha256(runtime_payload)
        or lock["runtime_config"].get("policy") != policy
    ):
        raise RuntimeLockError("runtime config semantic policy drifted")
    current_binding = validate_mission_binding(
        locked_mission, locked_config, expected_workspace=apxinf_root
    )
    if current_binding != lock.get("mission_binding"):
        raise RuntimeLockError("Mission routing binding drifted")

    session = lock.get("session")
    if (
        not isinstance(session, dict)
        or session.get("root") != current_binding["session"]
    ):
        raise RuntimeLockError("Session binding is malformed")
    session_payloads: dict[str, dict[str, Any]] = {}
    for field, filename in (("config", "session-config.json"), ("state", "state.json")):
        record = session.get(field)
        _validate_file_record(record, label=f"Session {field}")
        if Path(str(record.get("path"))) != Path(session["root"]) / filename:
            raise RuntimeLockError(f"Session {field} path is not exact")
        parsed = parse_json(
            read_file_bytes(Path(str(record["path"])), label=f"Session {field}"),
            label=f"Session {field}",
        )
        if record.get("semantic_sha256") != semantic_json_sha256(parsed):
            raise RuntimeLockError(f"Session {field} semantic identity drifted")
        session_payloads[field] = parsed
    validate_session_cross_binding(
        config=session_payloads["config"],
        state=session_payloads["state"],
        workspace=Path(current_binding["workspace"]),
        mission_id=current_binding["mission_id"],
    )

    source = lock.get("source_lock")
    _validate_file_record(source, label="HF source lock")
    source_payload = parse_json(
        read_file_bytes(Path(str(source["path"])), label="HF source lock"),
        label="HF source lock",
    )
    source_body = dict(source_payload)
    content_sha = source_body.pop("content_sha256", None)
    if (
        source.get("format") != "apxinf-hf-source-lock-v1"
        or content_sha != source.get("content_sha256")
        or semantic_json_sha256(source_body) != content_sha
        or source.get("semantic_sha256") != semantic_json_sha256(source_payload)
    ):
        raise RuntimeLockError("HF source-lock semantic identity drifted")

    _validate_file_record(host_python, label="Host Python")
    if _resolve_executable(sys.executable) != Path(str(host_python["path"])):
        raise RuntimeLockError(
            "locked Host Python differs from the current interpreter executable"
        )
    _validate_dynamic_libraries(
        host_python.get("dynamic_libraries"), label="Host Python dylibs"
    )
    current_evaluators = host_evaluator_identity(
        mission_payload,
        Path(current_binding["workspace"]),
        Path(str(host_python["path"])),
        source_lock=Path(str(source["path"])),
        source_payload=source_payload,
    )
    if current_evaluators != lock.get("host_evaluators"):
        raise RuntimeLockError("Host evaluator argv or transitive scripts drifted")
    _validate_file_record(codex, label="Codex native payload")
    _validate_file_record(codex.get("entrypoint"), label="Codex entrypoint provenance")
    _validate_dynamic_libraries(codex.get("dynamic_libraries"), label="Codex dylibs")
    _validate_file_record(node.get("executable"), label="Node executable")
    _validate_dynamic_libraries(node.get("dynamic_libraries"), label="Node dylibs")
    commands = runtime.get("external_commands")
    if not isinstance(commands, dict) or set(commands) != {
        "bash",
        "jq",
        "realpath",
        "sha256sum",
        "awk",
        "mktemp",
        "cp",
        "dirname",
        "mkdir",
        "rm",
    }:
        raise RuntimeLockError("external command identity set is incomplete")
    for name, record in commands.items():
        _validate_file_record(record, label=f"external command {name}")
    return {
        "passed": True,
        "contract": "apxinf-kersor-runtime-lock-verification-v2",
        "lock_sha256": observed_self_hash,
        "kersor_root": kersor["root"],
        "plugin_version": kersor.get("plugin_version"),
        "mission_sha256": lock["mission"]["sha256"],
        "runtime_config_sha256": lock["runtime_config"]["sha256"],
        "codex": {
            "path": codex["path"],
            "sha256": codex["sha256"],
            "version": codex["version"],
        },
        "node": {
            "path": node["executable"]["path"],
            "sha256": node["executable"]["sha256"],
        },
        "auth_custody_mechanism": AUTH_CUSTODY_MECHANISM,
        "command_read_scope_mechanism": COMMAND_READ_SCOPE_MECHANISM,
    }


def load_runtime_lock(path: Path) -> tuple[dict[str, Any], str]:
    canonical = _regular_file(path, label="runtime lock")
    payload = read_file_bytes(canonical, label="runtime lock")
    return parse_json(payload, label="runtime lock"), sha256_bytes(payload)


def write_runtime_lock(path: Path, lock: dict[str, Any]) -> None:
    destination = path.expanduser().absolute()
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json_bytes(lock) + b"\n"
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, destination, follow_symlinks=False)
        except FileExistsError as exc:
            raise RuntimeLockError(
                f"runtime lock already exists; refusing to overwrite: {destination}"
            ) from exc
        temporary.unlink()
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except RuntimeLockError:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    except OSError as exc:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise RuntimeLockError(f"cannot write runtime lock: {exc}") from exc


def _emit(value: object) -> None:
    print(canonical_json_bytes(value).decode("utf-8"))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("create", help="create a deterministic runtime lock")
    create.add_argument("--kersor-root", type=Path, required=True)
    create.add_argument("--mission", type=Path, required=True)
    create.add_argument("--runtime-config", type=Path, required=True)
    create.add_argument("--codex", required=True)
    create.add_argument("--node", default="node")
    create.add_argument("--source-lock", type=Path)
    create.add_argument("--host-python", type=Path)
    create.add_argument("--output", type=Path, required=True)
    verify = commands.add_parser("verify", help="revalidate an existing runtime lock")
    verify.add_argument("--lock", type=Path, required=True)
    verify.add_argument("--mission", type=Path)
    verify.add_argument("--runtime-config", type=Path)
    verify.add_argument("--codex")
    verify.add_argument("--node")
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        if arguments.command == "create":
            lock = build_runtime_lock(
                kersor_root=arguments.kersor_root,
                mission_path=arguments.mission,
                runtime_config_path=arguments.runtime_config,
                codex_command=arguments.codex,
                node_command=arguments.node,
                source_lock_path=arguments.source_lock,
                host_python_path=arguments.host_python,
            )
            write_runtime_lock(arguments.output, lock)
            _emit(
                {
                    "passed": True,
                    "contract": "apxinf-kersor-runtime-lock-created-v2",
                    "lock_path": str(arguments.output.expanduser().absolute()),
                    "lock_sha256": lock["lock_sha256"],
                }
            )
            return 0
        lock, _ = load_runtime_lock(arguments.lock)
        _emit(
            validate_runtime_lock(
                lock,
                mission_path=arguments.mission,
                runtime_config_path=arguments.runtime_config,
                codex_command=arguments.codex,
                node_command=arguments.node,
            )
        )
        return 0
    except RuntimeLockError as exc:
        _emit(
            {
                "passed": False,
                "error": {"code": "KERSOR_RUNTIME_LOCK_INVALID", "message": str(exc)},
            }
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
