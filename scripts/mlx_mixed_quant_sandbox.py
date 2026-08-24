#!/usr/bin/env python3
"""Launch one MLX mixed-quant backend child in a fixed macOS Seatbelt.

The launcher is intentionally independent from the search runner.  A future
runner integration may set ``network_blocked=true`` only after validating the
receipt returned here; the runner's current default remains unchanged.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import platform
import selectors
import signal
import stat
import subprocess
import sys
import time
from typing import NoReturn


RECEIPT_FORMAT = "apxinf-mlx-mixed-quant-sandbox-launch-receipt-v1"
SANDBOX_POLICY = "macos-seatbelt-mixed-quant-backend-v1"
SANDBOX_EXEC = Path("/usr/bin/sandbox-exec")
MAX_PATH_BYTES = 4096
MAX_ARGV_BYTES = 64 * 1024
_SHA256 = frozenset("0123456789abcdef")


class SandboxError(RuntimeError):
    """A fail-closed launcher, custody, or receipt validation error."""


def _fail(message: str) -> NoReturn:
    raise SandboxError(message)


def _canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise SandboxError(f"value is not canonical JSON: {error}") from error


def _object_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in _SHA256 for character in value)
    )


def _stable_stat(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _hash_file(path: Path, label: str) -> dict[str, object]:
    try:
        before = path.lstat()
    except OSError as error:
        raise SandboxError(f"cannot inspect {label}: {error}") from error
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        _fail(f"{label} must be a regular non-symlink file")
    digest = hashlib.sha256()
    size = 0
    try:
        with path.open("rb") as stream:
            opened = os.fstat(stream.fileno())
            if (opened.st_dev, opened.st_ino, opened.st_size) != (
                before.st_dev,
                before.st_ino,
                before.st_size,
            ):
                _fail(f"{label} changed while it was opened")
            while True:
                chunk = stream.read(4 * 1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                digest.update(chunk)
            after = os.fstat(stream.fileno())
    except OSError as error:
        raise SandboxError(f"cannot hash {label}: {error}") from error
    if _stable_stat(before) != _stable_stat(after) or size != before.st_size:
        _fail(f"{label} changed while it was hashed")
    return {"path": str(path), "size": size, "sha256": digest.hexdigest()}


def _canonical_directory(path: Path, label: str, *, mode_0700: bool = False) -> Path:
    try:
        info = path.lstat()
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise SandboxError(f"cannot inspect {label}: {error}") from error
    if (
        path != resolved
        or stat.S_ISLNK(info.st_mode)
        or not stat.S_ISDIR(info.st_mode)
        or info.st_uid != os.getuid()
    ):
        _fail(f"{label} must be an absolute owned real directory")
    if mode_0700 and stat.S_IMODE(info.st_mode) != 0o700:
        _fail(f"{label} mode must be exactly 0700")
    return path


def _canonical_file(path: Path, label: str) -> Path:
    try:
        info = path.lstat()
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise SandboxError(f"cannot inspect {label}: {error}") from error
    if path != resolved or stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        _fail(f"{label} must be an absolute regular non-symlink file")
    return path


@dataclass(frozen=True)
class SandboxLaunchSpec:
    python_path: Path
    backend_script_path: Path
    toolchain_dir: Path
    source_dir: Path
    source_manifest_sha256: str
    policy_path: Path
    scratch_dir: Path
    child_argv: tuple[str, ...]
    timeout_seconds: float = 120.0
    stdout_limit_bytes: int = 1024 * 1024
    stderr_limit_bytes: int = 1024 * 1024


@dataclass(frozen=True)
class _Prepared:
    python: Path
    script: Path
    toolchain: Path
    source: Path
    policy: Path
    scratch: Path
    sandbox: Path
    profile: str
    environment: dict[str, str]
    argv: list[str]
    identities: dict[str, object]


def _sbpl_string(value: str) -> str:
    if (
        not value
        or len(value.encode("utf-8")) > MAX_PATH_BYTES
        or any(ord(character) < 0x20 for character in value)
    ):
        _fail("sandbox path contains an unsupported character or length")
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _seatbelt_profile(
    *,
    source_dir: Path,
    policy_path: Path,
    toolchain_dir: Path,
    script_path: Path,
    scratch_dir: Path,
) -> str:
    source = _sbpl_string(str(source_dir))
    policy = _sbpl_string(str(policy_path))
    toolchain = _sbpl_string(str(toolchain_dir))
    script = _sbpl_string(str(script_path))
    scratch = _sbpl_string(str(scratch_dir))
    return "\n".join(
        [
            "(version 1)",
            "(allow default)",
            "(deny network*)",
            "(deny file-write*)",
            f"(allow file-write* (subpath {scratch}))",
            "(deny file-read*",
            '  (subpath "/Users")',
            '  (subpath "/private/var/folders")',
            '  (subpath "/private/tmp")',
            '  (subpath "/Volumes")',
            '  (subpath "/Network"))',
            "(allow file-read-metadata)",
            "(allow file-read*",
            f"  (subpath {source})",
            f"  (literal {policy})",
            f"  (subpath {toolchain})",
            f"  (literal {script})",
            f"  (subpath {scratch}))",
        ]
    )


def _environment(scratch: Path) -> dict[str, str]:
    directories = {
        "HOME": scratch / "home",
        "TMPDIR": scratch / "tmp",
        "HF_HOME": scratch / "huggingface",
        "HF_HUB_CACHE": scratch / "huggingface/hub",
        "TRANSFORMERS_CACHE": scratch / "transformers",
    }
    for name, directory in directories.items():
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        try:
            info = directory.lstat()
        except OSError as error:
            raise SandboxError(f"cannot inspect sandbox {name}: {error}") from error
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            _fail(f"sandbox {name} is not a real directory")
        os.chmod(directory, 0o700, follow_symlinks=False)
        _canonical_directory(directory, f"sandbox {name}", mode_0700=True)
    return {
        "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
        "LANG": "C",
        "LC_ALL": "C",
        "__CF_USER_TEXT_ENCODING": f"0x{os.getuid():X}:0x0:0x0",
        "HOME": str(directories["HOME"]),
        "TMPDIR": str(directories["TMPDIR"]),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "DO_NOT_TRACK": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "HF_HOME": str(directories["HF_HOME"]),
        "HF_HUB_CACHE": str(directories["HF_HUB_CACHE"]),
        "TRANSFORMERS_CACHE": str(directories["TRANSFORMERS_CACHE"]),
    }


def _prepare(spec: SandboxLaunchSpec) -> _Prepared:
    if not isinstance(spec, SandboxLaunchSpec):
        _fail("launcher requires a SandboxLaunchSpec")
    if sys.platform != "darwin":
        _fail("MLX Seatbelt launcher requires macOS")
    python = _canonical_file(Path(spec.python_path), "pinned CPython")
    script = _canonical_file(Path(spec.backend_script_path), "backend child script")
    toolchain = _canonical_directory(Path(spec.toolchain_dir), "MLX toolchain")
    source = _canonical_directory(Path(spec.source_dir), "frozen source")
    policy = _canonical_file(Path(spec.policy_path), "mixed policy")
    scratch = _canonical_directory(
        Path(spec.scratch_dir), "backend scratch", mode_0700=True
    )
    sandbox = _canonical_file(SANDBOX_EXEC, "sandbox-exec")
    if sandbox.stat().st_uid != 0 or not os.access(sandbox, os.X_OK):
        _fail("sandbox-exec must be the root-owned executable system tool")
    if python != Path(sys.executable).resolve(strict=True):
        _fail("launcher itself must run under the pinned child CPython")
    try:
        python.relative_to(toolchain)
    except ValueError as error:
        raise SandboxError("pinned CPython is outside the fixed toolchain") from error
    if not _is_sha256(spec.source_manifest_sha256):
        _fail("source manifest identity must be a lowercase SHA-256")
    if any(
        scratch == path or scratch in path.parents or path in scratch.parents
        for path in (source, policy, toolchain, script)
    ):
        _fail("writable scratch must not overlap any read-only input")
    if (
        type(spec.child_argv) is not tuple
        or any(type(value) is not str or "\x00" in value for value in spec.child_argv)
        or sum(len(value.encode("utf-8")) + 1 for value in spec.child_argv)
        > MAX_ARGV_BYTES
    ):
        _fail("child argv is not a bounded tuple of strings")
    if (
        type(spec.timeout_seconds) not in {int, float}
        or not 0 < spec.timeout_seconds <= 3600
        or type(spec.stdout_limit_bytes) is not int
        or type(spec.stderr_limit_bytes) is not int
        or not 1 <= spec.stdout_limit_bytes <= 16 * 1024 * 1024
        or not 1 <= spec.stderr_limit_bytes <= 16 * 1024 * 1024
    ):
        _fail("timeout or child stream limit is invalid")
    profile = _seatbelt_profile(
        source_dir=source,
        policy_path=policy,
        toolchain_dir=toolchain,
        script_path=script,
        scratch_dir=scratch,
    )
    environment = _environment(scratch)
    argv = [
        str(sandbox),
        "-p",
        profile,
        str(python),
        "-I",
        "-B",
        str(script),
        *spec.child_argv,
    ]
    identities = {
        "sandbox_exec": _hash_file(sandbox, "sandbox-exec"),
        "python": {
            **_hash_file(python, "pinned CPython"),
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
        },
        "backend_script": _hash_file(script, "backend child script"),
        "policy": _hash_file(policy, "mixed policy"),
        "source": {
            "directory": str(source),
            "manifest_sha256": spec.source_manifest_sha256,
        },
        "toolchain_directory": str(toolchain),
    }
    return _Prepared(
        python,
        script,
        toolchain,
        source,
        policy,
        scratch,
        sandbox,
        profile,
        environment,
        argv,
        identities,
    )


def _kill_process_group(process: subprocess.Popen[bytes]) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    except OSError:
        process.kill()


def _reap_killed_process(process: subprocess.Popen[bytes]) -> None:
    for stream in (process.stdout, process.stderr):
        if stream is not None and not stream.closed:
            stream.close()
    process.wait()


def _wait_for_leader_exit_without_reaping(
    process: subprocess.Popen[bytes], deadline: float
) -> None:
    flags = os.WEXITED | os.WNOHANG | os.WNOWAIT
    while True:
        try:
            observed = os.waitid(os.P_PID, process.pid, flags)
        except (ChildProcessError, OSError) as error:
            raise SandboxError(
                f"cannot observe sandboxed backend leader: {error}"
            ) from error
        if observed is not None:
            return
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            _kill_process_group(process)
            _reap_killed_process(process)
            _fail(
                "sandboxed backend timed out after closing its streams; "
                "owned process group was killed"
            )
        time.sleep(min(remaining, 0.01))


def _run_child(
    prepared: _Prepared, spec: SandboxLaunchSpec
) -> subprocess.CompletedProcess[bytes]:
    process = subprocess.Popen(
        prepared.argv,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=prepared.scratch,
        env=prepared.environment,
        start_new_session=True,
    )
    if process.stdout is None or process.stderr is None:
        _kill_process_group(process)
        _reap_killed_process(process)
        _fail("sandboxed backend pipes were not created")
    streams = selectors.DefaultSelector()
    streams.register(process.stdout, selectors.EVENT_READ, "stdout")
    streams.register(process.stderr, selectors.EVENT_READ, "stderr")
    buffers = {"stdout": bytearray(), "stderr": bytearray()}
    limits = {
        "stdout": spec.stdout_limit_bytes,
        "stderr": spec.stderr_limit_bytes,
    }
    deadline = time.monotonic() + spec.timeout_seconds
    failure: str | None = None
    try:
        while streams.get_map():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                failure = "sandboxed backend timed out"
                break
            for key, _mask in streams.select(timeout=min(remaining, 0.1)):
                lane = key.data
                assert lane in {"stdout", "stderr"}
                try:
                    chunk = os.read(key.fd, 64 * 1024)
                except OSError as error:
                    failure = f"sandboxed backend {lane} read failed: {error}"
                    break
                if not chunk:
                    streams.unregister(key.fileobj)
                    key.fileobj.close()
                    continue
                room = limits[lane] - len(buffers[lane])
                if len(chunk) > room:
                    if room > 0:
                        buffers[lane].extend(chunk[:room])
                    failure = f"sandboxed backend {lane} exceeded its fixed limit"
                    break
                buffers[lane].extend(chunk)
            if failure is not None:
                break
    finally:
        streams.close()
    if failure is not None:
        _kill_process_group(process)
        _reap_killed_process(process)
        raise SandboxError(f"{failure}; owned process group was killed")
    _wait_for_leader_exit_without_reaping(process, deadline)
    _kill_process_group(process)
    returncode = process.wait()
    return subprocess.CompletedProcess(
        prepared.argv,
        returncode,
        bytes(buffers["stdout"]),
        bytes(buffers["stderr"]),
    )


def _output_record(payload: bytes) -> dict[str, object]:
    return {
        "size": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "base64": base64.b64encode(payload).decode("ascii"),
    }


def launch_backend_child(spec: SandboxLaunchSpec) -> dict[str, object]:
    """Launch one child and return a self-hashed, non-publishing receipt."""

    prepared = _prepare(spec)
    result = _run_child(prepared, spec)
    post = _prepare(spec)
    if (
        post.profile != prepared.profile
        or post.argv != prepared.argv
        or post.environment != prepared.environment
        or post.identities != prepared.identities
    ):
        _fail("sandbox inputs changed while the backend child was running")
    stdout = _output_record(result.stdout)
    stderr = _output_record(result.stderr)
    body = {
        "format": RECEIPT_FORMAT,
        "passed": result.returncode == 0,
        "network_blocked": True,
        "trust_remote_code": False,
        "sandbox": {
            "policy": SANDBOX_POLICY,
            "profile_sha256": hashlib.sha256(
                prepared.profile.encode("utf-8")
            ).hexdigest(),
            "network": "deny-network-star-v1",
            "file_reads": "system-runtime-plus-fixed-user-inputs-only-v1",
            "writes": "scratch-only-v1",
        },
        "identities": prepared.identities,
        "process": {
            "argv": prepared.argv,
            "argv_sha256": _object_sha256(prepared.argv),
            "cwd": str(prepared.scratch),
            "environment": prepared.environment,
            "environment_sha256": _object_sha256(prepared.environment),
            "start_new_session": True,
            "process_group_kill_on_fault": True,
            "process_group_swept_before_leader_reap": True,
            "timeout_seconds": spec.timeout_seconds,
            "stdout_limit_bytes": spec.stdout_limit_bytes,
            "stderr_limit_bytes": spec.stderr_limit_bytes,
            "exit_code": result.returncode,
            "timed_out": False,
            "output_limited": False,
        },
        "output": {
            "stdout_size": stdout["size"],
            "stdout_sha256": stdout["sha256"],
            "stdout_base64": stdout["base64"],
            "stderr_size": stderr["size"],
            "stderr_sha256": stderr["sha256"],
            "stderr_base64": stderr["base64"],
        },
    }
    return {
        "format": RECEIPT_FORMAT,
        "body": body,
        "receipt_sha256": _object_sha256(body),
    }


def verify_launch_receipt(
    receipt: object, spec: SandboxLaunchSpec
) -> dict[str, object]:
    """Reconstruct the fixed launch contract and validate a completed receipt."""

    if type(receipt) is not dict or set(receipt) != {
        "format",
        "body",
        "receipt_sha256",
    }:
        _fail("sandbox launch receipt fields drifted")
    body = receipt.get("body")
    if (
        receipt.get("format") != RECEIPT_FORMAT
        or type(body) is not dict
        or receipt.get("receipt_sha256") != _object_sha256(body)
    ):
        _fail("sandbox launch receipt hash or format is invalid")
    prepared = _prepare(spec)
    expected_fields = {
        "format",
        "passed",
        "network_blocked",
        "trust_remote_code",
        "sandbox",
        "identities",
        "process",
        "output",
    }
    if set(body) != expected_fields:
        _fail("sandbox launch receipt body fields drifted")
    if (
        body.get("format") != RECEIPT_FORMAT
        or body.get("passed") is not True
        or body.get("network_blocked") is not True
        or body.get("trust_remote_code") is not False
        or body.get("identities") != prepared.identities
    ):
        _fail("sandbox launch did not produce a trusted successful result")
    sandbox = body.get("sandbox")
    expected_sandbox = {
        "policy": SANDBOX_POLICY,
        "profile_sha256": hashlib.sha256(prepared.profile.encode("utf-8")).hexdigest(),
        "network": "deny-network-star-v1",
        "file_reads": "system-runtime-plus-fixed-user-inputs-only-v1",
        "writes": "scratch-only-v1",
    }
    if sandbox != expected_sandbox:
        _fail("sandbox policy receipt drifted")
    process = body.get("process")
    if process != {
        "argv": prepared.argv,
        "argv_sha256": _object_sha256(prepared.argv),
        "cwd": str(prepared.scratch),
        "environment": prepared.environment,
        "environment_sha256": _object_sha256(prepared.environment),
        "start_new_session": True,
        "process_group_kill_on_fault": True,
        "process_group_swept_before_leader_reap": True,
        "timeout_seconds": spec.timeout_seconds,
        "stdout_limit_bytes": spec.stdout_limit_bytes,
        "stderr_limit_bytes": spec.stderr_limit_bytes,
        "exit_code": 0,
        "timed_out": False,
        "output_limited": False,
    }:
        _fail("sandbox process receipt drifted")
    output = body.get("output")
    if type(output) is not dict or set(output) != {
        "stdout_size",
        "stdout_sha256",
        "stdout_base64",
        "stderr_size",
        "stderr_sha256",
        "stderr_base64",
    }:
        _fail("sandbox output receipt fields drifted")
    for lane in ("stdout", "stderr"):
        try:
            payload = base64.b64decode(output[f"{lane}_base64"], validate=True)
        except (TypeError, ValueError) as error:
            raise SandboxError(
                f"sandbox {lane} receipt is not canonical base64"
            ) from error
        limit = spec.stdout_limit_bytes if lane == "stdout" else spec.stderr_limit_bytes
        if (
            output[f"{lane}_size"] != len(payload)
            or output[f"{lane}_sha256"] != hashlib.sha256(payload).hexdigest()
            or len(payload) > limit
        ):
            _fail(f"sandbox {lane} receipt hash or limit is invalid")
    return json.loads(_canonical_bytes(receipt))
