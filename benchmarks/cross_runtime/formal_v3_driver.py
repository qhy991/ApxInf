#!/usr/bin/env python3
"""Fail-closed NATIVE_A_VS_L campaign driver for the frozen v3 contract.

Formal execution is deliberately two phase.  ``prepare`` proves every
pre-marker prerequisite and exclusively creates the campaign-start marker.
After that marker and its referenced receipts have been committed and pushed,
``run`` re-proves publication and custody before dispatching the immutable
warmup/timed schedule.  ``self-test`` uses only in-process fixtures and can
never create a production marker or invoke a model.
"""

from __future__ import annotations

import copy
import argparse
import base64
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import secrets
import signal
import stat
import struct
import subprocess
import sys
import threading
import time
import tempfile
from typing import Any


EDGE_ID = "NATIVE_A_VS_L"
SAMPLE_FORMAT = "apxinf-qwen35-native-free-sample-receipt-v3"
TEACHER_FORMAT = "apxinf-qwen35-native-teacher-receipt-v3"
HOST_FORMAT = "apxinf-qwen35-native-host-custody-receipt-v3"
PLAN_FORMAT = "apxinf-qwen35-native-formal-execution-plan-v3"
MARKER_FORMAT = "apxinf-qwen35-native-formal-campaign-start-v3"
RAW_FORMAT = "apxinf-qwen35-native-formal-raw-campaign-v3"
FROZEN_CAMPAIGN_ID = "qwen35-0.8b-cross-runtime-formal-v3-20260826"
FROZEN_NATIVE_SUBCAMPAIGN_ID = (
    "qwen35-0.8b-native-apxinf-vs-llamacpp-formal-v3-20260826"
)
FROZEN_CONTRACT_REPOSITORY_PATH = (
    "configs/qwen35-0.8b-cross-runtime-formal-v3.json"
)
FROZEN_VALIDATOR_REPOSITORY_PATH = (
    "scripts/validate_qwen35_cross_runtime_formal_contract.py"
)
FROZEN_DRIVER_REPOSITORY_PATH = "benchmarks/cross_runtime/formal_v3_driver.py"
FROZEN_NATIVE_MARKER_REPOSITORY_PATH = (
    "crates/apxinf-metal/evidence/llama-cpp/qwen35-0.8b-native-apxinf-"
    "vs-llamacpp-formal-v3-campaign-start-20260826.json"
)
EXECUTION_PLAN_FIELDS = frozenset(
    {
        "format",
        "schema_version",
        "edge_id",
        "repository_root",
        "contract_repository_path",
        "validator_repository_path",
        "driver_repository_path",
        "plan_repository_path",
        "marker_repository_path",
        "raw_output_path",
        "timeout_seconds",
        "commands",
        "artifacts",
        "teacher_receipts",
    }
)
REJECTED_LEGACY_LLAMA_RUNNER_SHA256 = (
    "ccfa5ecd78119d4f8cdd8721e7faae360cb94b8334f9d61ed47e2e00290f2716"
)
FROZEN_CONTRACT_SHA256 = (
    "caa46b953f0abc0e58ffaa3725257fbbfabe4be49ca84aa0c523de8a16efb301"
)
FROZEN_VALIDATOR_SHA256 = (
    "9e4586d60839180cf7be55b63f53ac0c9dea811149ee3b4b1c9c9ccd6f9a11cf"
)
ROLE_TO_ARM = {"A": "AN", "B": "L"}
WARMUP_ROLES = ("A", "B", "B", "A", "A", "B")
TIMED_BLOCK_ORDERS = (
    "ABBA",
    "BAAB",
    "ABBA",
    "BAAB",
    "ABBA",
    "BAAB",
    "ABBA",
    "BAAB",
)

AN_CONSTRUCTOR_ID = (
    "from_weights_with_metal_w8_mlp_stack3_boundary_tail_head_gdn_core_fused_v1"
)
AN_PROFILE_ID = "metal-w8-mlp-stack3-boundary-tail-head-gdn-core-fused-v1"
AN_THREAD_OVERRIDE_ENVIRONMENT = (
    "VECLIB_MAXIMUM_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
)
HOST_CPU_WINDOW_MAX_OVERRUN_MS = 50.0


class CampaignError(RuntimeError):
    """A fail-closed formal-campaign condition was not proved."""


class ReceiptError(CampaignError):
    """An external runtime receipt did not satisfy the frozen contract."""


class RuntimeInvocationError(CampaignError):
    """One and only one attempted slot failed, with raw evidence retained."""

    def __init__(self, message: str, observation: dict[str, Any] | None = None):
        super().__init__(message)
        self.observation = observation or {}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CampaignError(message)


def _receipt_require(condition: bool, message: str) -> None:
    if not condition:
        raise ReceiptError(message)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _positive_number(value: Any, label: str) -> float:
    _receipt_require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        f"{label} must be numeric",
    )
    result = float(value)
    _receipt_require(math.isfinite(result) and result > 0.0, f"{label} must be finite and positive")
    return result


def compact_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, allow_nan=False, separators=(",", ":")
    ).encode("utf-8")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ReceiptError(f"JSON receipt contains duplicate key: {key}")
        result[key] = value
    return result


def _reject_nonfinite_constant(value: str) -> Any:
    raise ReceiptError(f"JSON receipt contains non-finite constant: {value}")


def parse_single_json_line(raw: bytes) -> dict[str, Any]:
    """Parse exactly one LF-terminated, strict-UTF8 JSON object."""

    _receipt_require(isinstance(raw, bytes), "runtime stdout is not bytes")
    _receipt_require(0 < len(raw) <= 8 * 1024 * 1024, "runtime receipt size is invalid")
    _receipt_require(raw.endswith(b"\n") and raw.count(b"\n") == 1, "runtime stdout must be exactly one LF-terminated JSON line")
    _receipt_require(b"\r" not in raw, "runtime receipt must use LF, not CRLF")
    try:
        text = raw[:-1].decode("utf-8", errors="strict")
        value = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite_constant,
        )
    except UnicodeDecodeError as error:
        raise ReceiptError(f"runtime receipt is not strict UTF-8: {error}") from error
    except json.JSONDecodeError as error:
        raise ReceiptError(f"runtime receipt is not strict JSON: {error}") from error
    _receipt_require(isinstance(value, dict), "runtime receipt must be a JSON object")
    return value


def _sha256_compact(value: Any) -> str:
    return hashlib.sha256(compact_json_bytes(value)).hexdigest()


def _file_snapshot(path_value: Path | str) -> tuple[Path, bytes, dict[str, Any]]:
    path = Path(path_value).resolve(strict=True)
    _require(path.is_file() and not path.is_symlink(), f"not a direct regular file: {path}")
    raw = path.read_bytes()
    stat = path.stat()
    return path, raw, {
        "absolute_path": str(path),
        "size_bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "device": stat.st_dev,
        "inode": stat.st_ino,
        "mode": stat.st_mode,
        "ctime_ns": stat.st_ctime_ns,
    }


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def file_custody(expected: dict[str, Any]) -> dict[str, Any]:
    """Hash one immutable regular file through one O_NOFOLLOW descriptor."""

    _require(isinstance(expected, dict), "artifact expectation must be an object")
    path_text = expected.get("absolute_path")
    _require(isinstance(path_text, str) and path_text.startswith("/"), "artifact path must be absolute")
    _require(os.path.normpath(path_text) == path_text, "artifact path must be normalized")
    expected_size = expected.get("size_bytes")
    expected_hash = expected.get("sha256")
    _require(_is_int(expected_size) and expected_size > 0, "artifact expected size is invalid")
    _require(_valid_sha256(expected_hash), "artifact expected SHA256 is invalid")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path_text, flags)
    except OSError as error:
        raise CampaignError(f"artifact O_NOFOLLOW open failed for {path_text}: {error}") from error
    try:
        before = os.fstat(descriptor)
        _require(stat.S_ISREG(before.st_mode), f"artifact is not a regular file: {path_text}")
        digest = hashlib.sha256()
        total = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            total += len(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity_fields = (
        "st_dev",
        "st_ino",
        "st_mode",
        "st_size",
        "st_nlink",
        "st_ctime_ns",
    )
    identity_equal = all(getattr(before, field) == getattr(after, field) for field in identity_fields)
    actual_hash = digest.hexdigest()
    _require(identity_equal, f"artifact identity changed while hashing: {path_text}")
    _require(total == before.st_size == expected_size, f"artifact size mismatch: {path_text}")
    _require(actual_hash == expected_hash, f"artifact SHA256 mismatch: {path_text}")
    return {
        "absolute_path": path_text,
        "device": before.st_dev,
        "inode": before.st_ino,
        "mode": before.st_mode,
        "size_bytes": before.st_size,
        "hard_link_count": before.st_nlink,
        "ctime_ns": before.st_ctime_ns,
        "sha256": actual_hash,
        "open_flags": ["O_RDONLY", "O_CLOEXEC", "O_NOFOLLOW"],
        "identity_before_after_equal": identity_equal,
    }


def _system_command_runner(
    argv: list[str],
    cwd: Path | str,
    timeout_seconds: float,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    completed = subprocess.run(
        argv,
        cwd=str(cwd),
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout_seconds,
        check=False,
    )
    return {
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def _invoke_command(
    runner: Any,
    argv: list[str],
    cwd: Path | str,
    timeout_seconds: float = 30.0,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    result = runner(argv, cwd, timeout_seconds, env)
    _require(isinstance(result, dict), f"command runner returned no result: {argv}")
    _require(_is_int(result.get("returncode")), f"command exit status is invalid: {argv}")
    _require(isinstance(result.get("stdout"), bytes), f"command stdout is not bytes: {argv}")
    _require(isinstance(result.get("stderr"), bytes), f"command stderr is not bytes: {argv}")
    return result


def _git_success(
    runner: Any, repo: Path, arguments: list[str], allow_stdout: bool = True
) -> bytes:
    argv = ["/usr/bin/git", *arguments]
    result = _invoke_command(
        runner,
        argv,
        repo,
        env=git_custody_environment(),
    )
    _require(result["returncode"] == 0, f"git {' '.join(arguments)} failed")
    _require(result["stderr"] == b"", f"git {' '.join(arguments)} wrote stderr")
    if not allow_stdout:
        _require(result["stdout"] == b"", f"git {' '.join(arguments)} wrote unexpected stdout")
    return result["stdout"]


def git_custody_environment() -> dict[str, str]:
    """Return the complete, non-inherited environment for custody Git calls.

    In particular, Git must not inherit transport rewrites, proxy variables,
    credential helpers, askpass programs, or system/global configuration from
    the interactive account that happens to launch the campaign.
    """

    return {
        "LANG": "C",
        "LC_ALL": "C",
        "TZ": "UTC",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_SYSTEM": "/dev/null",
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_CONFIG_COUNT": "0",
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_ASKPASS": "/usr/bin/false",
        "GIT_PROTOCOL_FROM_USER": "0",
        "GIT_ALLOW_PROTOCOL": "https",
    }


def reject_git_transport_overrides(
    repository_root: Path | str,
    command_runner: Any = _system_command_runner,
) -> list[str]:
    """Reject local-repository settings capable of changing Git transport.

    The live publication proof names an HTTPS URL explicitly.  Local Git
    configuration is still consulted for ``remote.origin.url``, so reject
    every namespace that can redirect that URL or insert a transport/proxy.
    """

    repo = Path(repository_root).resolve(strict=True)
    raw = _git_success(
        command_runner,
        repo,
        ["config", "--local", "--name-only", "--null", "--list"],
    )
    _require(raw == b"" or raw.endswith(b"\0"), "local Git config key list is malformed")
    encoded_keys = raw[:-1].split(b"\0") if raw else []
    keys: list[str] = []
    for encoded in encoded_keys:
        _require(encoded != b"", "local Git config key list contains an empty key")
        try:
            key = encoded.decode("utf-8", errors="strict")
        except UnicodeDecodeError as error:
            raise CampaignError("local Git config key is not strict UTF-8") from error
        _require(
            re.fullmatch(r"[A-Za-z][A-Za-z0-9-]*(?:\.[^\x00\r\n]+)+", key)
            is not None,
            f"local Git config key is malformed: {key!r}",
        )
        lowered = key.lower()
        components = lowered.split(".")
        dangerous = (
            lowered.startswith(("url.", "http.", "https.", "include.", "includeif.", "protocol."))
            or lowered in ("core.gitproxy", "core.sshcommand")
            or lowered.startswith("credential.")
            or (
                len(components) >= 3
                and components[0] == "remote"
                and components[-1]
                in {
                    "proxy",
                    "proxyauthmethod",
                    "vcs",
                    "uploadpack",
                    "receivepack",
                }
            )
        )
        _require(not dangerous, f"local Git transport override is forbidden: {key}")
        keys.append(key)
    return keys


def _parse_oid_line(raw: bytes, length: int, label: str) -> str:
    _require(raw.endswith(b"\n") and raw.count(b"\n") == 1, f"{label} output shape is invalid")
    try:
        value = raw[:-1].decode("ascii", errors="strict")
    except UnicodeDecodeError as error:
        raise CampaignError(f"{label} object ID is not ASCII") from error
    _require(
        len(value) == length and all(character in "0123456789abcdef" for character in value),
        f"{label} object ID is invalid",
    )
    return value


def collect_git_custody(
    repository_root: Path | str,
    contract: dict[str, Any],
    tracked_paths: dict[str, str],
    *,
    command_runner: Any = _system_command_runner,
    published_marker_label: str | None = None,
) -> dict[str, Any]:
    """Prove clean local Git state and live public-main equality.

    ``refs/remotes/origin/main`` is recorded only as a secondary observation;
    the exact ``git ls-remote`` response is the publication authority.
    """

    repo = Path(repository_root).resolve(strict=True)
    _require(repo.is_dir(), "repository root is not a directory")
    _require(isinstance(tracked_paths, dict) and "contract" in tracked_paths, "tracked contract path is absent")
    local_config_keys = reject_git_transport_overrides(repo, command_runner)
    activation = contract["activation_contract"]
    remote_url = activation["frozen_origin_remote_url"]
    remote_ref = activation["frozen_live_remote_ref"]
    remote_raw = _git_success(command_runner, repo, ["config", "--get", "remote.origin.url"])
    _require(remote_raw == remote_url.encode("utf-8") + b"\n", "remote.origin.url is not exact")
    object_format_raw = _git_success(command_runner, repo, ["rev-parse", "--show-object-format"])
    _require(object_format_raw in (b"sha1\n", b"sha256\n"), "Git object format is unsupported")
    object_format = object_format_raw[:-1].decode("ascii")
    oid_length = 40 if object_format == "sha1" else 64
    head = _parse_oid_line(
        _git_success(command_runner, repo, ["rev-parse", "--verify", "HEAD^{commit}"]),
        oid_length,
        "HEAD",
    )
    tracking_ref = "refs/remotes/origin/main"
    tracking = _parse_oid_line(
        _git_success(
            command_runner,
            repo,
            ["rev-parse", "--verify", f"{tracking_ref}^{{commit}}"],
        ),
        oid_length,
        "origin/main",
    )
    ls_remote_arguments = ["ls-remote", "--exit-code", remote_url, remote_ref]
    ls_remote_stdout = _git_success(command_runner, repo, ls_remote_arguments)
    expected_remote_suffix = b"\t" + remote_ref.encode("utf-8") + b"\n"
    _require(ls_remote_stdout.endswith(expected_remote_suffix), "live ls-remote response shape is invalid")
    live_oid = _parse_oid_line(
        ls_remote_stdout[: -len(expected_remote_suffix)] + b"\n",
        oid_length,
        "live remote main",
    )
    _require(head == tracking == live_oid, "HEAD, origin/main and live remote main are not equal")
    status = _git_success(
        command_runner,
        repo,
        ["status", "--porcelain=v1", "-z", "--untracked-files=all"],
    )
    _require(status == b"", "Git worktree is not clean")
    head_tree = _parse_oid_line(
        _git_success(command_runner, repo, ["rev-parse", "--verify", "HEAD^{tree}"]),
        oid_length,
        "HEAD tree",
    )

    tracked: dict[str, Any] = {}
    for label, repository_path in tracked_paths.items():
        _require(isinstance(label, str) and label, "tracked file label is invalid")
        _require(
            isinstance(repository_path, str)
            and repository_path
            and not repository_path.startswith("/")
            and os.path.normpath(repository_path) == repository_path
            and not repository_path.startswith("../"),
            f"tracked repository path is invalid: {repository_path!r}",
        )
        ls_tree = _git_success(
            command_runner,
            repo,
            ["ls-tree", "-z", "HEAD", "--", repository_path],
        )
        _require(ls_tree.endswith(b"\0") and ls_tree.count(b"\0") == 1, f"tracked file is absent or ambiguous: {repository_path}")
        try:
            metadata_raw, path_raw = ls_tree[:-1].split(b"\t", 1)
            mode_raw, kind_raw, oid_raw = metadata_raw.split(b" ", 2)
            recorded_path = path_raw.decode("utf-8", errors="strict")
            blob_oid = oid_raw.decode("ascii", errors="strict")
        except (ValueError, UnicodeDecodeError) as error:
            raise CampaignError(f"Git tree entry is malformed: {repository_path}") from error
        _require(mode_raw in (b"100644", b"100755") and kind_raw == b"blob", f"tracked path is not a regular blob: {repository_path}")
        _require(recorded_path == repository_path, f"Git tree path mismatch: {repository_path}")
        _require(len(blob_oid) == oid_length and all(character in "0123456789abcdef" for character in blob_oid), f"Git blob object ID is invalid: {repository_path}")
        blob_bytes = _git_success(command_runner, repo, ["cat-file", "blob", blob_oid])
        working_path = repo / repository_path
        try:
            working_bytes = working_path.read_bytes()
        except OSError as error:
            raise CampaignError(f"cannot read tracked file {repository_path}: {error}") from error
        _require(working_bytes == blob_bytes, f"tracked file differs bytewise from HEAD: {repository_path}")
        tracked[label] = {
            "repository_path": repository_path,
            "blob_oid": blob_oid,
            "blob_size_bytes": len(blob_bytes),
            "blob_sha256": hashlib.sha256(blob_bytes).hexdigest(),
            "observed_size_bytes": len(working_bytes),
            "observed_sha256": hashlib.sha256(working_bytes).hexdigest(),
        }

    contract_path = tracked_paths["contract"]
    contract_commit = _parse_oid_line(
        _git_success(
            command_runner,
            repo,
            ["log", "-1", "--format=%H", "--", contract_path],
        ),
        oid_length,
        "contract commit",
    )
    contract_tree = _parse_oid_line(
        _git_success(
            command_runner,
            repo,
            ["rev-parse", "--verify", f"{contract_commit}^{{tree}}"],
        ),
        oid_length,
        "contract tree",
    )
    ancestor = _invoke_command(
        command_runner,
        ["/usr/bin/git", "merge-base", "--is-ancestor", contract_commit, head],
        repo,
        env=git_custody_environment(),
    )
    _require(ancestor["returncode"] == 0 and ancestor["stdout"] == b"" and ancestor["stderr"] == b"", "contract commit is not an ancestor of activation HEAD")
    if published_marker_label is not None:
        _require(published_marker_label in tracked, "published marker was not included in Git custody")
        marker_path = tracked_paths[published_marker_label]
        marker_commit = _parse_oid_line(
            _git_success(
                command_runner,
                repo,
                ["log", "-1", "--format=%H", "--", marker_path],
            ),
            oid_length,
            "activation marker commit",
        )
        _require(marker_commit == head, "activation marker is not committed at public HEAD")
    else:
        marker_commit = None
    return {
        "repository_url": activation["repository"],
        "remote_origin_url": remote_url,
        "object_format": object_format,
        "local_tracking_ref": tracking_ref,
        "local_tracking_oid": tracking,
        "live_remote_url": remote_url,
        "live_remote_ref": remote_ref,
        "ls_remote_argv": ["git", *ls_remote_arguments],
        "ls_remote_exit_code": 0,
        "ls_remote_stdout_sha256": hashlib.sha256(ls_remote_stdout).hexdigest(),
        "ls_remote_live_oid": live_oid,
        "head_commit": head,
        "head_tree": head_tree,
        "contract_commit": contract_commit,
        "contract_tree": contract_tree,
        "activation_commit": marker_commit,
        "contract_commit_is_ancestor_of_activation_commit": True,
        "activation_commit_equals_head_and_live_remote_oid": (
            marker_commit == head == live_oid if marker_commit is not None else False
        ),
        "local_tracking_ref_used_as_publication_proof": False,
        "worktree_clean": True,
        "sanitized_git_environment": git_custody_environment(),
        "local_config_keys": local_config_keys,
        "local_transport_overrides_absent": True,
        "tracked_files": tracked,
    }


def validate_an_campaign_commit(
    plan: dict[str, Any],
    git_custody: dict[str, Any],
    repository_root: Path | str,
    command_runner: Any = _system_command_runner,
) -> dict[str, Any]:
    """Prove the native source commit/tree is in live campaign ancestry."""

    _require(isinstance(plan, dict), "execution plan is absent")
    _require(isinstance(git_custody, dict), "Git custody is absent")
    artifacts = plan.get("artifacts")
    _require(isinstance(artifacts, dict), "execution plan artifacts are absent")
    native = artifacts.get("AN")
    _require(isinstance(native, dict), "AN artifact custody is absent")
    campaign_commit = native.get("runtime_source_commit")
    public_head = git_custody.get("head_commit")
    _require(
        isinstance(campaign_commit, str)
        and len(campaign_commit) == 40
        and all(character in "0123456789abcdef" for character in campaign_commit),
        "AN campaign commit is invalid",
    )
    _require(
        isinstance(public_head, str)
        and len(public_head) in (40, 64)
        and all(character in "0123456789abcdef" for character in public_head),
        "live campaign Git HEAD is invalid",
    )
    _require(git_custody.get("worktree_clean") is True, "AN campaign checkout is not clean")
    repo = Path(repository_root).resolve(strict=True)
    object_format = git_custody.get("object_format", "sha1")
    oid_length = 40 if object_format == "sha1" else 64
    resolved_commit = _parse_oid_line(
        _git_success(
            command_runner,
            repo,
            ["rev-parse", "--verify", f"{campaign_commit}^{{commit}}"],
        ),
        oid_length,
        "AN campaign source commit",
    )
    _require(
        resolved_commit == campaign_commit,
        "AN campaign source object did not resolve to the planned commit",
    )
    campaign_tree = _parse_oid_line(
        _git_success(
            command_runner,
            repo,
            ["rev-parse", "--verify", f"{campaign_commit}^{{tree}}"],
        ),
        oid_length,
        "AN campaign source tree",
    )
    ancestor = _invoke_command(
        command_runner,
        [
            "/usr/bin/git",
            "merge-base",
            "--is-ancestor",
            campaign_commit,
            public_head,
        ],
        repo,
        env=git_custody_environment(),
    )
    _require(
        ancestor["returncode"] == 0
        and ancestor["stdout"] == b""
        and ancestor["stderr"] == b"",
        "AN campaign source commit is not an ancestor of live campaign Git",
    )
    return {
        "campaign_commit": campaign_commit,
        "campaign_tree": campaign_tree,
        "live_head": public_head,
        "clean_checkout": True,
        "is_ancestor_of_live_head": True,
    }


def load_frozen_contract(
    contract_path: Path | str, validator_path: Path | str
) -> dict[str, Any]:
    """Load and execute the exact reviewed validator over exact contract bytes."""

    _, validator_bytes, validator_file = _file_snapshot(validator_path)
    _require(
        validator_file["sha256"] == FROZEN_VALIDATOR_SHA256,
        "formal contract validator SHA256 is not the frozen v3 validator",
    )
    namespace: dict[str, Any] = {
        "__name__": "_apxinf_frozen_cross_runtime_validator_v3",
        "__file__": validator_file["absolute_path"],
    }
    try:
        source = validator_bytes.decode("utf-8", errors="strict")
        exec(compile(source, validator_file["absolute_path"], "exec"), namespace)
    except (UnicodeDecodeError, SyntaxError) as error:
        raise CampaignError(f"cannot load the frozen formal validator: {error}") from error

    _, contract_bytes, contract_file = _file_snapshot(contract_path)
    _require(
        contract_file["sha256"] == FROZEN_CONTRACT_SHA256,
        "formal contract SHA256 is not the frozen v3 contract",
    )
    try:
        contract_text = contract_bytes.decode("utf-8", errors="strict")
        contract = json.loads(
            contract_text,
            object_pairs_hook=namespace["_reject_duplicate_keys"],
            parse_constant=namespace["_reject_nonfinite_constant"],
        )
        validated = namespace["validate_contract"](contract)
        validation = namespace["validation_receipt"](validated, contract_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError, KeyError, ValueError) as error:
        raise CampaignError(f"frozen formal contract rejected: {error}") from error
    _require(validation.get("valid") is True, "frozen formal validator did not admit the contract")
    return {
        "contract": validated,
        "contract_file": contract_file,
        "validator_file": validator_file,
        "validation": validation,
    }


def _same_number(actual: Any, expected: float, label: str) -> None:
    value = _positive_number(actual, label)
    _receipt_require(
        math.isclose(value, expected, rel_tol=1e-12, abs_tol=1e-12),
        f"{label} is arithmetically inconsistent",
    )


def declared_schedule() -> list[dict[str, Any]]:
    """Return a fresh copy of the exact frozen native campaign schedule."""

    schedule: list[dict[str, Any]] = []
    for warmup_index, role in enumerate(WARMUP_ROLES):
        schedule.append(
            {
                "sequence_index": len(schedule),
                "phase": "warmup",
                "warmup_index": warmup_index,
                "block_index": None,
                "slot_index": None,
                "role": role,
                "arm": ROLE_TO_ARM[role],
            }
        )
    for block_index, order in enumerate(TIMED_BLOCK_ORDERS):
        for slot_index, role in enumerate(order):
            schedule.append(
                {
                    "sequence_index": len(schedule),
                    "phase": "timed",
                    "warmup_index": None,
                    "block_index": block_index,
                    "slot_index": slot_index,
                    "role": role,
                    "arm": ROLE_TO_ARM[role],
                }
            )
    return schedule


def expected_deployment_receipt(arm: str) -> dict[str, Any]:
    if arm == "AN":
        return {
            "constructor_id": AN_CONSTRUCTOR_ID,
            "profile_id": AN_PROFILE_ID,
            "context_capacity_tokens": 256,
            "prefill_device": "CPU",
            "prefill_precision": "F32",
            "full_attention_device": "CPU",
            "full_attention_precision": "F32",
            "kv_key_dtype": "F32",
            "kv_value_dtype": "F32",
            "head": "F32 tied embedding top-4 exact rerank",
            "metal_build_input_count": 17,
            "exact_live_execution_ledger": True,
            "thread_policy": {
                "policy": "Accelerate OS-managed default",
                "fixed_worker_count_claimed": False,
                "VECLIB_MAXIMUM_THREADS_present": False,
                "OMP_NUM_THREADS_present": False,
                "OPENBLAS_NUM_THREADS_present": False,
                "MKL_NUM_THREADS_present": False,
            },
        }
    if arm == "L":
        return {
            "context_capacity_tokens": 256,
            "model_type": "GGUF-Q8_0",
            "kv_key_dtype": "F16",
            "kv_value_dtype": "F16",
            "threads": 4,
            "batch_threads": 4,
            "transformer_layers_on_mtl0": 24,
            "input_embedding_cpu_fallback_observed": True,
            "dynamic_backend_scan": False,
        }
    raise CampaignError(f"unknown native arm: {arm}")


def _validate_repository_path(value: Any, label: str) -> str:
    _require(
        isinstance(value, str)
        and value
        and not value.startswith("/")
        and os.path.normpath(value) == value
        and value != ".."
        and not value.startswith("../"),
        f"{label} must be a normalized repository-relative path",
    )
    return value


def _validate_file_expectation(value: Any, label: str) -> dict[str, Any]:
    _require(isinstance(value, dict), f"{label} file expectation is absent")
    _require(
        set(value) == {"absolute_path", "size_bytes", "sha256"},
        f"{label} file expectation fields differ",
    )
    path = value.get("absolute_path")
    _require(
        isinstance(path, str)
        and path.startswith("/")
        and os.path.normpath(path) == path,
        f"{label} file path is not normalized absolute",
    )
    _require(
        _is_int(value.get("size_bytes")) and value["size_bytes"] > 0,
        f"{label} file size is invalid",
    )
    _require(_valid_sha256(value.get("sha256")), f"{label} file SHA256 is invalid")
    return value


def expected_free_runtime_argv(
    arm: str, repository_root: str, custody: dict[str, Any]
) -> list[str]:
    """Return the sole admitted, fixed-order free-run argv for one arm."""

    runner = custody["runner"]["absolute_path"]
    model = custody["model"]["absolute_path"]
    if arm == "AN":
        return [
            runner,
            "--mode",
            "native-v3-free",
            "--model-dir",
            str(Path(model).parent),
            "--source-lock",
            str(
                Path(repository_root)
                / ".apxinf/onboarding/qwen35-0.8b/source-lock.json"
            ),
        ]
    if arm == "L":
        return [
            runner,
            "--model",
            model,
            "--gpu-layers",
            "-1",
            "--gpu-device",
            "MTL0",
            "--threads",
            "4",
            "--mode",
            "native-v3-free",
        ]
    raise CampaignError(f"unknown native arm: {arm}")


def validate_execution_plan(
    plan: Any, contract: dict[str, Any]
) -> dict[str, Any]:
    """Validate immutable local inputs without treating authored blockers as final."""

    _require(isinstance(plan, dict), "execution plan must be an object")
    _require(
        set(plan) == EXECUTION_PLAN_FIELDS,
        "execution plan fields differ from v3",
    )
    _require(plan.get("format") == PLAN_FORMAT, "execution plan format mismatch")
    _require(plan.get("schema_version") == 3, "execution plan schema mismatch")
    _require(plan.get("edge_id") == EDGE_ID, "execution plan edge mismatch")
    repository_root = plan.get("repository_root")
    _require(
        isinstance(repository_root, str)
        and repository_root.startswith("/")
        and os.path.normpath(repository_root) == repository_root,
        "execution plan repository root is invalid",
    )
    _require(
        plan.get("contract_repository_path") == contract["activation_contract"]["path"],
        "execution plan contract path mismatch",
    )
    _require(
        plan.get("validator_repository_path")
        == "scripts/validate_qwen35_cross_runtime_formal_contract.py",
        "execution plan validator path mismatch",
    )
    _require(
        plan.get("driver_repository_path")
        == contract["runtime_custody"]["ApxInf_native"]["runner_source_path"],
        "execution plan driver path mismatch",
    )
    _validate_repository_path(plan.get("plan_repository_path"), "execution plan")
    marker_binding = contract["machine_receipt_contract"]["subcampaign_marker_bindings"][EDGE_ID]
    _require(
        plan.get("marker_repository_path")
        == marker_binding["expected_marker_repository_path"],
        "execution plan marker path mismatch",
    )
    raw_output_path = plan.get("raw_output_path")
    _require(
        isinstance(raw_output_path, str)
        and raw_output_path.startswith("/")
        and os.path.normpath(raw_output_path) == raw_output_path,
        "execution plan raw output path is invalid",
    )
    timeout = plan.get("timeout_seconds")
    _require(_is_int(timeout) and 1 <= timeout <= 3600, "execution plan timeout is invalid")

    protocol = contract["execution_protocol"][EDGE_ID]
    _require(
        list(WARMUP_ROLES) == protocol["untimed_warmup_order"]
        and list(TIMED_BLOCK_ORDERS) == protocol["timed_block_orders"]
        and protocol["untimed_warmups_per_arm"] == 3
        and protocol["timed_blocks"] == 8
        and protocol["timed_samples_per_arm"] == 16
        and protocol["timed_samples_total"] == 32,
        "contract native execution schedule is not the implemented frozen schedule",
    )
    artifacts = plan.get("artifacts")
    commands = plan.get("commands")
    teachers = plan.get("teacher_receipts")
    _require(isinstance(artifacts, dict) and set(artifacts) == {"AN", "L"}, "execution plan artifact arms differ")
    _require(isinstance(commands, dict) and set(commands) == {"AN", "L"}, "execution plan command arms differ")
    _require(isinstance(teachers, dict) and set(teachers) == {"AN", "L"}, "execution plan teacher arms differ")
    deployments = contract["native_deployment_contract"]["deployments"]
    llama_contract = contract["runtime_custody"]["pinned_llama_cpp_core"]
    source_checkpoint = contract["source_model_custody"]["checkpoint"]
    _require(
        llama_contract["runner_binary_sha256"]
        != REJECTED_LEGACY_LLAMA_RUNNER_SHA256,
        "legacy v2 llama.cpp runner cannot satisfy native-v3 mode/context/teacher receipts",
    )
    for arm in ("AN", "L"):
        custody = artifacts[arm]
        _require(isinstance(custody, dict), f"{arm} artifact custody is absent")
        required = {
            "configuration_id",
            "runner",
            "model",
            "runtime_source_commit",
            "loaded_non_system_library_closure_sha256",
            "packed_weight_and_resident_buffer_manifest_sha256",
            "deployment",
        }
        _require(set(custody) == required, f"{arm} artifact custody fields differ")
        _require(
            custody["configuration_id"] == deployments[arm]["configuration_id"],
            f"{arm} named deployment configuration mismatch",
        )
        runner = _validate_file_expectation(custody["runner"], f"{arm} runner")
        model = _validate_file_expectation(custody["model"], f"{arm} model")
        _require(
            isinstance(custody["runtime_source_commit"], str)
            and len(custody["runtime_source_commit"]) == 40
            and all(character in "0123456789abcdef" for character in custody["runtime_source_commit"]),
            f"{arm} runtime source commit is invalid",
        )
        _require(
            _valid_sha256(custody["loaded_non_system_library_closure_sha256"]),
            f"{arm} loaded library closure hash is invalid",
        )
        _require(custody["deployment"] == expected_deployment_receipt(arm), f"{arm} deployment disclosure mismatch")
        if arm == "AN":
            _require(
                model["size_bytes"] == source_checkpoint["size_bytes"]
                and model["sha256"] == source_checkpoint["sha256"],
                "AN source checkpoint custody mismatch",
            )
            _require(
                _valid_sha256(custody["packed_weight_and_resident_buffer_manifest_sha256"]),
                "AN packed/resident manifest hash is invalid",
            )
        else:
            _require(
                runner["size_bytes"] == llama_contract["runner_binary_size_bytes"]
                and runner["sha256"] == llama_contract["runner_binary_sha256"],
                "L runner differs from the repinned v3 contract",
            )
            source_weights = deployments["L"]["source_weights"]
            _require(
                model["size_bytes"] == source_weights["artifact_size_bytes"]
                and model["sha256"] == source_weights["artifact_sha256"],
                "L GGUF artifact custody mismatch",
            )
            _require(
                custody["runtime_source_commit"] == llama_contract["source_commit"],
                "L source commit mismatch",
            )
            _require(
                custody["packed_weight_and_resident_buffer_manifest_sha256"] is None,
                "L must not claim an ApxInf packed-weight manifest",
            )

        command = commands[arm]
        _require(isinstance(command, dict) and set(command) == {"argv", "environment"}, f"{arm} command fields differ")
        argv = command["argv"]
        _require(
            isinstance(argv, list)
            and len(argv) >= 3
            and all(isinstance(item, str) and item and "\0" not in item and "\n" not in item for item in argv),
            f"{arm} command argv is invalid",
        )
        _require(argv[0] == runner["absolute_path"], f"{arm} command executable differs from runner custody")
        _require(
            sum(item == "--mode" for item in argv) == 1
            and any(
                argv[index : index + 2] == ["--mode", "native-v3-free"]
                for index in range(len(argv) - 1)
            ),
            f"{arm} command must explicitly select native-v3-free",
        )
        _require(
            argv == expected_free_runtime_argv(
                arm, repository_root, custody
            ),
            f"{arm} command argv differs from the fixed-order custody-bound v3 argv",
        )
        _require(
            command["environment"] == {"LC_ALL": "C", "TZ": "UTC"},
            f"{arm} command environment is not the frozen minimal environment",
        )
        teacher = teachers[arm]
        _require(
            isinstance(teacher, dict)
            and set(teacher)
            == {
                "reference_repository_path",
                "runtime_repository_path",
            },
            f"{arm} teacher receipt paths differ",
        )
        for field, value in teacher.items():
            _validate_repository_path(value, f"{arm} teacher {field}")
    return plan


def validate_sample_receipt(
    receipt: Any,
    slot: dict[str, Any],
    nonce: str,
    contract: dict[str, Any],
    expected_custody: dict[str, Any],
) -> dict[str, Any]:
    """Validate one external runtime's single free-run JSON receipt.

    The runtime is untrusted.  The driver therefore rechecks schedule binding,
    raw IDs, the complete trajectory, timing arithmetic and immutable custody;
    an exit status or a trajectory hash alone is never admission evidence.
    """

    _receipt_require(isinstance(receipt, dict), "sample receipt must be an object")
    _receipt_require(receipt.get("format") == SAMPLE_FORMAT, "legacy or unknown sample receipt format")
    _receipt_require(receipt.get("schema_version") == 3, "sample receipt schema is not v3")
    _receipt_require(receipt.get("campaign_id") == contract.get("campaign_id"), "sample campaign binding mismatch")
    edge = contract["comparison_graph"]["edges"][EDGE_ID]
    _receipt_require(receipt.get("subcampaign_id") == edge.get("subcampaign_id"), "sample subcampaign binding mismatch")
    _receipt_require(receipt.get("edge_id") == EDGE_ID, "sample edge binding mismatch")
    _receipt_require(receipt.get("mode") == "native-v3-free", "legacy or teacher mode cannot enter the performance schedule")

    request = receipt.get("request")
    _receipt_require(isinstance(request, dict), "sample request binding is absent")
    expected_request = {
        "nonce": nonce,
        "sequence_index": slot["sequence_index"],
        "phase": slot["phase"],
        "warmup_index": slot["warmup_index"],
        "block_index": slot["block_index"],
        "slot_index": slot["slot_index"],
        "role": slot["role"],
        "arm": slot["arm"],
    }
    _receipt_require(request == expected_request, "sample request nonce or schedule binding mismatch")

    native = contract["workload_contracts"]["NATIVE_RAW13_FREE128_V3"]
    generation = native["generation"]
    shared_prompt = contract["workload_contracts"]["shared_prompt"]
    trajectory = native["free_run_trajectory_admission"]
    workload = receipt.get("workload")
    _receipt_require(isinstance(workload, dict), "sample workload receipt is absent")
    expected_workload_scalars = {
        "ingress_semantics": "raw-token-ids",
        "prefill_token_count": shared_prompt["token_count"],
        "generated_token_count": trajectory["generated_token_ids_count"],
        "sampling": generation["sampling"],
        "temperature": generation["temperature"],
        "eog_policy": generation["eog_policy"],
        "speculative_decoding": generation["speculative_decoding_allowed"],
        "continuous_batching": generation["continuous_batching_allowed"],
        "sequence_count": generation["sequence_count"],
        "requested_context_tokens": generation["requested_context_tokens"],
        "effective_context_tokens": generation["effective_context_tokens"],
        "requested_batch_tokens": generation["requested_batch_tokens"],
        "effective_batch_tokens": generation["effective_batch_tokens"],
        "requested_ubatch_tokens": generation["requested_ubatch_tokens"],
        "effective_ubatch_tokens": generation["effective_ubatch_tokens"],
        "empty_state_before_prefill": True,
        "prompt_cache_reused": False,
    }
    for field, expected in expected_workload_scalars.items():
        _receipt_require(workload.get(field) == expected, f"sample workload {field} mismatch")
    prompt_ids = workload.get("prompt_token_ids")
    _receipt_require(prompt_ids == shared_prompt["token_ids"], "sample raw13 prompt IDs mismatch")
    _receipt_require(_sha256_compact(prompt_ids) == shared_prompt["sha256"], "sample raw13 prompt hash mismatch")
    generated_ids = workload.get("generated_token_ids")
    _receipt_require(
        isinstance(generated_ids, list)
        and len(generated_ids) == trajectory["generated_token_ids_count"]
        and all(_is_int(token) and token >= 0 for token in generated_ids),
        "sample must record exactly 128 nonnegative integer generated token IDs",
    )
    generated_hash = _sha256_compact(generated_ids)
    _receipt_require(generated_hash == trajectory["expected_sha256"], "sample free128 trajectory mismatch")
    _receipt_require(workload.get("generated_token_ids_sha256") == generated_hash, "sample-reported free128 hash mismatch")

    timing_contract = contract["timing_contract"][EDGE_ID]
    timing = receipt.get("timing")
    _receipt_require(isinstance(timing, dict), "sample timing receipt is absent")
    timing_expected = {
        "clock": contract["timing_contract"]["clock"],
        "start_boundary": timing_contract["start"],
        "common_token_ready_boundary": timing_contract["common_token_ready_boundary"],
        "end_boundary": timing_contract["end"],
        "selection_work_included": True,
        "accelerator_completion_before_each_token_ready_timestamp": timing_contract[
            "accelerator_completion_before_each_token_ready_timestamp"
        ],
        "final_sampled_token_decoded_inside_timed_region": timing_contract[
            "final_sampled_token_decoded_inside_timed_region"
        ],
    }
    for field, expected in timing_expected.items():
        _receipt_require(timing.get(field) == expected, f"sample timing {field} mismatch")
    _receipt_require(
        isinstance(timing.get("clock_identity"), str) and timing["clock_identity"],
        "sample clock identity is absent",
    )
    _receipt_require(
        _is_int(timing.get("clock_resolution_ns")) and timing["clock_resolution_ns"] > 0,
        "sample clock resolution is invalid",
    )
    timestamps = [
        timing.get("prefill_start_ns"),
        timing.get("token_1_ready_ns"),
        timing.get("token_128_ready_ns"),
    ]
    _receipt_require(all(_is_int(value) for value in timestamps), "sample token-ready timestamps must be integer nanoseconds")
    start_ns, first_ns, last_ns = timestamps
    _receipt_require(start_ns < first_ns < last_ns, "sample token-ready timestamps are not strictly ordered")
    _same_number(timing.get("ttft_ms"), (first_ns - start_ns) / 1_000_000, "ttft_ms")
    _same_number(timing.get("total_latency_ms"), (last_ns - start_ns) / 1_000_000, "total_latency_ms")
    _same_number(timing.get("tpot_ms"), (last_ns - first_ns) / 127 / 1_000_000, "tpot_ms")
    _same_number(timing.get("generation_tps"), 127_000_000_000 / (last_ns - first_ns), "generation_tps")

    custody = receipt.get("custody")
    _receipt_require(isinstance(custody, dict), "sample custody receipt is absent")
    for field in (
        "configuration_id",
        "runner",
        "model",
        "runtime_source_commit",
        "loaded_non_system_library_closure_sha256",
        "packed_weight_and_resident_buffer_manifest_sha256",
        "deployment",
    ):
        _receipt_require(custody.get(field) == expected_custody.get(field), f"sample custody {field} mismatch")
    _receipt_require(custody.get("fresh_process") is True, "sample did not use a fresh process")
    _receipt_require(custody.get("start_end_identity_equal") is True, "sample runtime identity changed")
    if slot["arm"] == "L":
        _receipt_require(custody.get("ggml_backend_path_unset") is True, "llama.cpp dynamic backend search was not disabled")
    _receipt_require(
        custody.get("deployment") == expected_deployment_receipt(slot["arm"]),
        "sample live deployment receipt does not prove the named configuration",
    )
    return receipt


def _require_mapping(value: Any, label: str) -> dict[str, Any]:
    _receipt_require(isinstance(value, dict), f"{label} is absent")
    return value


def _require_exact_fields(
    value: dict[str, Any],
    expected: dict[str, Any],
    label: str,
    *,
    additional_fields: tuple[str, ...] = (),
) -> None:
    _receipt_require(
        set(value) == set(expected).union(additional_fields),
        f"{label} fields differ from the strict v3 schema",
    )
    for field, expected_value in expected.items():
        _receipt_require(
            value.get(field) == expected_value,
            f"{label} {field} mismatch",
        )


def _validate_llama_runtime_closure(
    raw: dict[str, Any], expected_custody: dict[str, Any], label: str
) -> dict[str, Any]:
    runtime = _require_mapping(raw.get("runtime_custody"), f"{label} runtime custody")
    expected_fields = {
        "loaded_non_system_library_closure",
        "loaded_non_system_library_closure_start",
        "loaded_non_system_library_closure_end",
        "loaded_non_system_library_closure_sha256",
        "loaded_non_system_library_closure_start_sha256",
        "loaded_non_system_library_closure_end_sha256",
    }
    _receipt_require(
        set(runtime) == expected_fields,
        f"{label} runtime custody fields differ from the raw-v3 schema",
    )
    closure = runtime["loaded_non_system_library_closure"]
    start = runtime["loaded_non_system_library_closure_start"]
    end = runtime["loaded_non_system_library_closure_end"]
    expected_hash = expected_custody[
        "loaded_non_system_library_closure_sha256"
    ]
    _receipt_require(
        isinstance(closure, list)
        and start == end == closure
        and _sha256_compact(closure) == expected_hash
        and runtime["loaded_non_system_library_closure_sha256"] == expected_hash
        and runtime["loaded_non_system_library_closure_start_sha256"]
        == expected_hash
        and runtime["loaded_non_system_library_closure_end_sha256"]
        == expected_hash,
        f"{label} raw loaded-library closure start/end or plan hash mismatch",
    )
    required_entry_fields = {
        "absolute_path",
        "size_bytes",
        "sha256",
        "device",
        "inode",
        "change_time_seconds",
        "change_time_nanoseconds",
    }
    previous_path: str | None = None
    for entry in closure:
        _receipt_require(
            isinstance(entry, dict) and set(entry) == required_entry_fields,
            f"{label} loaded-library entry fields differ",
        )
        path = entry.get("absolute_path")
        _receipt_require(
            isinstance(path, str)
            and path.startswith("/")
            and os.path.normpath(path) == path
            and (previous_path is None or previous_path < path),
            f"{label} loaded-library paths are invalid or not uniquely sorted",
        )
        _receipt_require(
            _is_int(entry.get("size_bytes"))
            and entry["size_bytes"] > 0
            and _valid_sha256(entry.get("sha256"))
            and _is_int(entry.get("device"))
            and entry["device"] > 0
            and _is_int(entry.get("inode"))
            and entry["inode"] > 0
            and _is_int(entry.get("change_time_seconds"))
            and entry["change_time_seconds"] > 0
            and _is_int(entry.get("change_time_nanoseconds"))
            and 0 <= entry["change_time_nanoseconds"] < 1_000_000_000,
            f"{label} loaded-library identity is invalid",
        )
        previous_path = path
    return copy.deepcopy(runtime)


def _validate_llama_model_preflight_binding(
    model: dict[str, Any],
    expected_custody: dict[str, Any],
    expected_artifact_observation: dict[str, Any],
) -> dict[str, Any]:
    observation = _require_mapping(
        expected_artifact_observation, "llama preflight artifact observation"
    )
    _receipt_require(
        observation.get("configuration_id")
        == expected_custody.get("configuration_id")
        and observation.get("runtime_source_commit")
        == expected_custody.get("runtime_source_commit")
        and observation.get("loaded_non_system_library_closure_sha256")
        == expected_custody.get("loaded_non_system_library_closure_sha256")
        and observation.get("deployment") == expected_custody.get("deployment"),
        "llama preflight artifact observation is not bound to the plan",
    )
    observed_model = _require_mapping(
        observation.get("model"), "llama preflight model observation"
    )
    expected_model = _require_mapping(
        expected_custody.get("model"), "llama expected model custody"
    )
    _receipt_require(
        observed_model.get("absolute_path") == expected_model.get("absolute_path")
        and observed_model.get("size_bytes") == expected_model.get("size_bytes")
        and observed_model.get("sha256") == expected_model.get("sha256")
        and observed_model.get("open_flags")
        == ["O_RDONLY", "O_CLOEXEC", "O_NOFOLLOW"]
        and observed_model.get("identity_before_after_equal") is True
        and _is_int(observed_model.get("mode"))
        and stat.S_ISREG(observed_model["mode"])
        and _is_int(observed_model.get("device"))
        and observed_model["device"] > 0
        and _is_int(observed_model.get("inode"))
        and observed_model["inode"] > 0
        and _is_int(observed_model.get("hard_link_count"))
        and observed_model["hard_link_count"] >= 1
        and _is_int(observed_model.get("ctime_ns"))
        and observed_model["ctime_ns"] > 0,
        "llama preflight O_NOFOLLOW model observation is invalid",
    )
    identity = _require_mapping(
        model.get("file_identity_start"), "llama model file identity start"
    )
    _receipt_require(
        identity.get("device") == observed_model["device"]
        and identity.get("inode") == observed_model["inode"]
        and identity.get("size_bytes") == observed_model["size_bytes"]
        and identity.get("hard_link_count")
        == observed_model["hard_link_count"]
        and identity.get("change_time_seconds")
        == observed_model["ctime_ns"] // 1_000_000_000
        and identity.get("change_time_nanoseconds")
        == observed_model["ctime_ns"] % 1_000_000_000,
        "llama runtime model descriptor is not the preflight hashed file identity",
    )
    return copy.deepcopy(observed_model)


def adapt_llama_free_receipt(
    raw: Any,
    slot: dict[str, Any],
    nonce: str,
    contract: dict[str, Any],
    expected_custody: dict[str, Any],
    expected_artifact_observation: dict[str, Any],
) -> dict[str, Any]:
    """Validate the pinned C++ raw-v3 schema and form the common receipt."""

    _receipt_require(slot.get("arm") == "L", "llama adapter received a non-L schedule slot")
    raw = _require_mapping(raw, "llama raw receipt")
    _require_exact_fields(
        raw,
        {
            "schema": "apxinf.llama-cpp.raw-token-diagnostic.v3",
            "ok": True,
            "mode": "native-v3-free",
            "token_ready_boundary": "next-greedy-token-ready",
            "selection_work_included": True,
            "accelerator_completion_before_each_token_ready_timestamp": True,
        },
        "llama raw receipt",
        additional_fields=(
            "contract",
            "model",
            "parameters",
            "output",
            "timings",
            "llama_perf",
            "runtime_custody",
            "backend",
            "placement_attestation",
            "post_measurement_execution_proof",
            "build",
        ),
    )
    shared_prompt = contract["workload_contracts"]["shared_prompt"]
    native = contract["workload_contracts"]["NATIVE_RAW13_FREE128_V3"]
    generation = native["generation"]
    raw_contract = _require_mapping(raw.get("contract"), "llama raw contract")
    _require_exact_fields(
        raw_contract,
        {
            "prompt_token_ids": shared_prompt["token_ids"],
            "mode": "native-v3-free",
            "sampling": "greedy-argmax",
            "generated_token_count": 128,
            "eog_termination": False,
            "token_ready_elapsed_ns_origin": "immediately-before-prompt-decode",
            "final_sampled_token_is_not_decoded_in_timed_workload": True,
            "final_sampled_token_decoded_once_post_measurement_for_execution_proof": True,
        },
        "llama raw contract",
    )
    parameters = _require_mapping(raw.get("parameters"), "llama raw parameters")
    _require_exact_fields(
        parameters,
        {
            "n_ctx_requested": 256,
            "n_ctx_effective": 256,
            "n_ctx_per_sequence_effective": 256,
            "n_batch_requested": 13,
            "n_batch_effective": 13,
            "n_ubatch_requested": 13,
            "n_ubatch_effective": 13,
            "n_seq_max_requested": 1,
            "n_seq_max_effective": 1,
            "n_threads": 4,
            "n_threads_batch": 4,
            "lane": "gpu-all-layers",
            "n_gpu_layers": -1,
            "kv_cache_type_k": "f16",
            "kv_cache_type_v": "f16",
            "flash_attention": "auto",
            "offload_kqv": True,
            "op_offload": True,
            "swa_full": False,
            "kv_unified": False,
            "model_load_mode": "none-from-pinned-file-pointer",
            "use_mmap": False,
            "use_direct_io": False,
            "use_mlock": False,
            "check_tensors": False,
        },
        "llama raw parameters",
    )

    model = _require_mapping(raw.get("model"), "llama raw model")
    _require_exact_fields(
        model,
        {
            "requested_path": expected_custody["model"]["absolute_path"],
            "load_binding": "pinned-file-descriptor",
            "open_flags": "O_RDONLY|O_NOFOLLOW|O_CLOEXEC",
            "file_identity_unchanged": True,
            "file_size_bytes": expected_custody["model"]["size_bytes"],
            "file_type": "Q8_0",
            "vocabulary_size": 248320,
            "layer_count": 24,
            "is_recurrent": False,
            "is_hybrid": True,
        },
        "llama raw model",
        additional_fields=(
            "file_identity_start",
            "file_identity_after_load",
            "file_identity_before_receipt",
            "description",
            "parameter_count",
            "tensor_size_bytes",
            "file_type_code",
        ),
    )
    identity_labels = (
        "file_identity_start",
        "file_identity_after_load",
        "file_identity_before_receipt",
    )
    identities = [_require_mapping(model.get(label), f"llama {label}") for label in identity_labels]
    for label, model_identity in zip(identity_labels, identities):
        _receipt_require(
            set(model_identity)
            == {
                "device",
                "inode",
                "size_bytes",
                "hard_link_count",
                "change_time_seconds",
                "change_time_nanoseconds",
            },
            f"llama {label} fields differ from the strict v3 schema",
        )
    _receipt_require(identities[0] == identities[1] == identities[2], "llama model file identity changed")
    identity = identities[0]
    _receipt_require(
        all(
            _is_int(identity.get(field))
            for field in (
                "device",
                "inode",
                "size_bytes",
                "hard_link_count",
                "change_time_seconds",
                "change_time_nanoseconds",
            )
        )
        and identity["device"] > 0
        and identity["inode"] > 0
        and identity["size_bytes"] == expected_custody["model"]["size_bytes"]
        and identity["hard_link_count"] >= 1,
        "llama model file identity is invalid",
    )
    model_observation = _validate_llama_model_preflight_binding(
        model, expected_custody, expected_artifact_observation
    )
    runtime_custody = _validate_llama_runtime_closure(
        raw, expected_custody, "llama"
    )

    output = _require_mapping(raw.get("output"), "llama raw output")
    _receipt_require(
        set(output) == {"token_ids", "token_ready_elapsed_ns"},
        "llama raw output fields differ from the strict v3 schema",
    )
    token_ids = output.get("token_ids")
    expected_trajectory = native["free_run_trajectory_admission"]
    _receipt_require(
        isinstance(token_ids, list)
        and len(token_ids) == 128
        and all(_is_int(token) and token >= 0 for token in token_ids)
        and _sha256_compact(token_ids) == expected_trajectory["expected_sha256"],
        "llama raw free128 trajectory mismatch",
    )
    elapsed = output.get("token_ready_elapsed_ns")
    _receipt_require(
        isinstance(elapsed, list)
        and len(elapsed) == 128
        and all(_is_int(value) and value > 0 for value in elapsed)
        and all(left < right for left, right in zip(elapsed, elapsed[1:])),
        "llama raw token-ready elapsed times are not 128 strictly increasing integers",
    )
    timings = _require_mapping(raw.get("timings"), "llama raw timings")
    _require_exact_fields(
        timings,
        {
            "clock_identity": "std::chrono::steady_clock",
            "clock_is_steady": True,
        },
        "llama raw timings",
        additional_fields=(
            "clock_resolution_ns",
            "clock_period_numerator",
            "clock_period_denominator",
            "generation_start_ns",
            "model_load_elapsed_ns",
            "context_init_elapsed_ns",
            "generation_elapsed_ns",
            "measurement_scope_elapsed_ns",
            "post_measurement_execution_proof_elapsed_ns",
            "receipt_ready_elapsed_ns",
        ),
    )
    start_ns = timings.get("generation_start_ns")
    resolution_ns = timings.get("clock_resolution_ns")
    period_numerator = timings.get("clock_period_numerator")
    period_denominator = timings.get("clock_period_denominator")
    _receipt_require(_is_int(start_ns) and start_ns > 0, "llama generation_start_ns is invalid")
    _receipt_require(_is_int(resolution_ns) and resolution_ns > 0, "llama clock resolution is invalid")
    _receipt_require(
        _is_int(period_numerator)
        and period_numerator > 0
        and _is_int(period_denominator)
        and period_denominator > 0
        and resolution_ns
        == math.ceil(period_numerator * 1_000_000_000 / period_denominator),
        "llama clock period and nanosecond resolution are inconsistent",
    )
    for field in (
        "model_load_elapsed_ns",
        "context_init_elapsed_ns",
        "generation_elapsed_ns",
        "measurement_scope_elapsed_ns",
        "post_measurement_execution_proof_elapsed_ns",
        "receipt_ready_elapsed_ns",
    ):
        _receipt_require(_is_int(timings.get(field)) and timings[field] > 0, f"llama timing {field} is invalid")
    _receipt_require(
        timings["generation_elapsed_ns"] >= elapsed[-1],
        "llama generation duration ends before token 128 is ready",
    )

    perf = _require_mapping(raw.get("llama_perf"), "llama performance counters")
    _receipt_require(
        set(perf)
        == {
            "context",
            "sampler",
            "captured_before_post_measurement_execution_proof",
        },
        "llama performance counter fields differ from the strict v3 schema",
    )
    _require_exact_fields(
        _require_mapping(perf.get("context"), "llama context performance counters"),
        {"n_prompt_eval": 13, "n_eval": 127, "n_reused": 126},
        "llama context performance counters",
        additional_fields=(
            "t_start_ms",
            "t_load_ms",
            "t_prompt_eval_ms",
            "t_eval_ms",
        ),
    )
    _require_exact_fields(
        _require_mapping(perf.get("sampler"), "llama sampler performance counters"),
        {"n_sample": 0},
        "llama sampler performance counters",
        additional_fields=("t_sample_ms",),
    )
    _receipt_require(perf.get("captured_before_post_measurement_execution_proof") is True, "llama performance counters include the execution proof")

    backend = _require_mapping(raw.get("backend"), "llama backend receipt")
    _require_exact_fields(
        backend,
        {
            "registration_mode": "linked-static-registry-only",
            "dynamic_backend_scan_invoked": False,
            "backend_directory_option_supported": False,
            "ggml_backend_path_present": False,
            "supports_gpu_offload": True,
        },
        "llama backend receipt",
        additional_fields=(
            "selected_gpu_device",
            "registered_devices_after_generation",
            "system_info",
        ),
    )
    selected_gpu = _require_mapping(backend.get("selected_gpu_device"), "llama selected GPU")
    _require_exact_fields(
        selected_gpu,
        {"name": "MTL0", "description": "Apple M4", "type": "gpu"},
        "llama selected GPU",
    )
    devices = backend.get("registered_devices_after_generation")
    _receipt_require(isinstance(devices, list), "llama registered device inventory is absent")
    _receipt_require(
        sum(
            isinstance(device, dict)
            and device.get("name") == "MTL0"
            and device.get("description") == "Apple M4"
            and device.get("type") == "gpu"
            for device in devices
        )
        == 1
        and sum(
            isinstance(device, dict)
            and device.get("name") == "CPU"
            and device.get("description") == "Apple M4"
            and device.get("type") == "cpu"
            for device in devices
        )
        == 1,
        "llama registered Metal/CPU device inventory mismatch",
    )
    placement = _require_mapping(raw.get("placement_attestation"), "llama placement attestation")
    _require_exact_fields(
        placement,
        {
            "method": "pinned-llama-internal-layer-assignments-plus-memory-breakdown-v1",
            "passed": True,
            "model_selected_device_count": 1,
            "transformer_layer_count": 24,
            "layers_on_selected_gpu": 24,
            "layers_on_cpu": 0,
            "output_on_selected_gpu": True,
            "output_on_cpu": False,
            "input_embedding_buffer_type": "CPU",
        },
        "llama placement attestation",
        additional_fields=(
            "input_embedding_device",
            "memory_by_device_class",
            "memory_by_buffer_type",
        ),
    )
    proof = _require_mapping(raw.get("post_measurement_execution_proof"), "llama execution proof")
    _require_exact_fields(
        proof,
        {
            "method": "scheduler-callback-completed-sentinels-v1",
            "passed": True,
            "timing_excluded": True,
            "decode_count": 1,
            "requested_sentinel_count": 26,
            "completed_sentinel_count": 26,
            "completed_input_embedding_on_cpu": True,
            "completed_transformer_layer_endpoints": 24,
            "completed_output_head": True,
            "completed_on_selected_gpu": 25,
            "completed_on_cpu": 1,
            "backend_mismatch": False,
            "duplicate_or_unexpected_callback": False,
        },
        "llama execution proof",
        additional_fields=("proof_token_id",),
    )
    build = _require_mapping(raw.get("build"), "llama build receipt")
    _require_exact_fields(
        build,
        {
            "llama_cpp_source_id": expected_custody["runtime_source_commit"],
            "llama_cpp_source_id_provenance": "clean-git-head",
            "cmake_build_type": "Release",
            "build_shared_libs": False,
            "ggml_backend_dl": False,
            "ggml_metal": True,
            "ggml_metal_embed_library": True,
            "ggml_accelerate": True,
            "ggml_native": True,
        },
        "llama build receipt",
        additional_fields=(
            "llama_cpp_version",
            "cmake_version",
            "cxx_compiler_id",
            "cxx_compiler_version",
            "cxx_compiler_banner",
        ),
    )

    first_ns = start_ns + elapsed[0]
    last_ns = start_ns + elapsed[-1]
    raw_bytes = compact_json_bytes(raw)
    wrapped = {
        "format": SAMPLE_FORMAT,
        "schema_version": 3,
        "campaign_id": contract["campaign_id"],
        "subcampaign_id": contract["comparison_graph"]["edges"][EDGE_ID]["subcampaign_id"],
        "edge_id": EDGE_ID,
        "mode": "native-v3-free",
        "request": {
            "nonce": nonce,
            "sequence_index": slot["sequence_index"],
            "phase": slot["phase"],
            "warmup_index": slot["warmup_index"],
            "block_index": slot["block_index"],
            "slot_index": slot["slot_index"],
            "role": slot["role"],
            "arm": "L",
        },
        "workload": {
            "ingress_semantics": "raw-token-ids",
            "prompt_token_ids": token_ids[:0] + shared_prompt["token_ids"],
            "prefill_token_count": 13,
            "generated_token_ids": token_ids,
            "generated_token_ids_sha256": _sha256_compact(token_ids),
            "generated_token_count": 128,
            "sampling": generation["sampling"],
            "temperature": generation["temperature"],
            "eog_policy": generation["eog_policy"],
            "speculative_decoding": False,
            "continuous_batching": False,
            "sequence_count": 1,
            "requested_context_tokens": 256,
            "effective_context_tokens": 256,
            "requested_batch_tokens": 13,
            "effective_batch_tokens": 13,
            "requested_ubatch_tokens": 13,
            "effective_ubatch_tokens": 13,
            "empty_state_before_prefill": True,
            "prompt_cache_reused": False,
        },
        "timing": {
            "clock": "monotonic",
            "clock_identity": timings["clock_identity"],
            "clock_resolution_ns": resolution_ns,
            "start_boundary": contract["timing_contract"][EDGE_ID]["start"],
            "common_token_ready_boundary": "next-greedy-token-ready",
            "end_boundary": "128th-next-greedy-token-ready",
            "selection_work_included": True,
            "accelerator_completion_before_each_token_ready_timestamp": True,
            "final_sampled_token_decoded_inside_timed_region": False,
            "prefill_start_ns": start_ns,
            "token_1_ready_ns": first_ns,
            "token_128_ready_ns": last_ns,
            "ttft_ms": elapsed[0] / 1_000_000,
            "total_latency_ms": elapsed[-1] / 1_000_000,
            "tpot_ms": (elapsed[-1] - elapsed[0]) / 127 / 1_000_000,
            "generation_tps": 127_000_000_000 / (elapsed[-1] - elapsed[0]),
        },
        "custody": {
            **expected_custody,
            "fresh_process": True,
            "start_end_identity_equal": True,
            "ggml_backend_path_unset": True,
            "model_preflight_observation": model_observation,
            **runtime_custody,
        },
        "external_receipt": {
            "schema": raw["schema"],
            "size_bytes": len(raw_bytes),
            "sha256": hashlib.sha256(raw_bytes).hexdigest(),
            "raw": raw,
        },
    }
    validate_sample_receipt(wrapped, slot, nonce, contract, expected_custody)
    return wrapped


def validate_an_free_receipt(
    receipt: Any,
    slot: dict[str, Any],
    nonce: str,
    contract: dict[str, Any],
    expected_custody: dict[str, Any],
    expected_source_binding: dict[str, Any],
) -> dict[str, Any]:
    """Validate the native runner's common wrapper plus its fused-path proof."""

    _receipt_require(slot.get("arm") == "AN", "AN adapter received a non-AN schedule slot")
    receipt = validate_sample_receipt(
        receipt, slot, nonce, contract, expected_custody
    )
    _receipt_require(receipt.get("passed") is True, "AN free runner did not pass")
    timing = _require_mapping(receipt.get("timing"), "AN timing")
    _receipt_require(
        timing.get("clock_identity") == "Darwin CLOCK_MONOTONIC_RAW",
        "AN timing clock identity is not the pinned monotonic raw clock",
    )
    ready = timing.get("token_ready_ns")
    elapsed = timing.get("next_greedy_token_ready_elapsed_ns")
    _receipt_require(
        isinstance(ready, list)
        and len(ready) == 128
        and all(_is_int(value) and value > 0 for value in ready)
        and all(left < right for left, right in zip(ready, ready[1:])),
        "AN token-ready timestamp ledger is invalid",
    )
    start_ns = timing["prefill_start_ns"]
    _receipt_require(
        ready[0] == timing["token_1_ready_ns"]
        and ready[-1] == timing["token_128_ready_ns"]
        and elapsed == [value - start_ns for value in ready],
        "AN token-ready timestamp and elapsed ledgers disagree",
    )
    final_path = _require_mapping(receipt.get("final_path"), "AN final fused path")
    path_checks = _require_mapping(final_path.get("path_checks"), "AN fused path checks")
    required_path_checks = {
        "schedule_valid",
        "mechanism_and_precision_valid",
        "six_region_execution_valid",
        "tail_execution_and_phase_valid",
        "aggregate_ledger_valid",
        "generation_receipt_valid",
        "terminal_clear",
        "all_valid",
    }
    _receipt_require(
        required_path_checks.issubset(path_checks)
        and all(path_checks.get(field) is True for field in required_path_checks),
        "AN fused path did not prove every live execution check",
    )
    ledger = _require_mapping(
        final_path.get("aggregate_buffer_ledger"),
        "AN packed/resident aggregate ledger",
    )
    custody = _require_mapping(receipt.get("custody"), "AN sample custody")
    _receipt_require(
        _sha256_compact(ledger)
        == custody.get("packed_weight_and_resident_buffer_manifest_sha256")
        == expected_custody["packed_weight_and_resident_buffer_manifest_sha256"],
        "AN packed/resident ledger hash mismatch",
    )
    _validate_an_dynamic_custody(
        custody,
        contract,
        expected_custody["loaded_non_system_library_closure_sha256"],
        "AN free",
        expected_source_binding,
    )
    source_start = _require_mapping(
        custody.get("source_custody_start"), "AN source custody start"
    )
    source_end = _require_mapping(
        custody.get("source_custody_end"), "AN source custody end"
    )
    if source_start != source_end:
        _receipt_require(
            source_start.get("binary") == source_end.get("binary")
            and source_start.get("model_dir", {}).get("path")
            == source_end.get("model_dir", {}).get("path")
            and source_start.get("model_dir", {}).get("artifacts")
            == source_end.get("model_dir", {}).get("artifacts")
            and source_start.get("sources", {}).get("gate")
            == source_end.get("gate")
            and source_start.get("sources", {}).get("rust_and_bridge_sources")
            == source_end.get("rust_and_bridge_sources")
            and source_start.get("sources", {}).get(
                "compiled_metal_shader_sources"
            )
            == source_end.get("compiled_metal_shader_sources")
            and source_end.get("verified_at_end") is True,
            "AN source/model/binary custody start/end evidence disagrees",
        )
    _receipt_require(
        custody.get("start_end_identity_equal") is True,
        "AN source custody changed during the sample",
    )
    return receipt


def validate_teacher_receipt(
    receipt: Any,
    arm: str,
    contract: dict[str, Any],
    reference_file: dict[str, Any],
    runtime_file: dict[str, Any],
    expected_custody: dict[str, Any],
) -> dict[str, Any]:
    """Prove the independent prompt12 + 128 teacher-forced admission lane."""

    _receipt_require(arm in ("AN", "L"), "teacher arm is not native AN or L")
    _receipt_require(isinstance(receipt, dict), "teacher receipt must be an object")
    native = contract["workload_contracts"]["NATIVE_RAW13_FREE128_V3"]
    teacher = native["teacher_forced_admission"]
    prompt = contract["workload_contracts"]["shared_prompt"]["token_ids"]
    edge = contract["comparison_graph"]["edges"][EDGE_ID]
    scalar_bindings = {
        "format": TEACHER_FORMAT,
        "schema_version": 3,
        "campaign_id": contract["campaign_id"],
        "subcampaign_id": edge["subcampaign_id"],
        "edge_id": EDGE_ID,
        "arm": arm,
        "mode": "native-v3-teacher",
        "prefill_prompt_token_ids_count": len(prompt) - 1,
        "teacher_input_token_ids_sha256": teacher[
            "teacher_input_token_ids_sha256"
        ],
        "reference_receipt_size_bytes": reference_file.get("size_bytes"),
        "reference_receipt_sha256": reference_file.get("sha256"),
        "runtime_receipt_size_bytes": runtime_file.get("size_bytes"),
        "runtime_receipt_sha256": runtime_file.get("sha256"),
    }
    for field, expected in scalar_bindings.items():
        _receipt_require(receipt.get(field) == expected, f"teacher receipt {field} mismatch")
    _receipt_require(
        receipt.get("prefill_prompt_token_ids") == prompt[:-1],
        "teacher prefill must contain raw prompt tokens 0 through 11 only",
    )
    teacher_inputs = receipt.get("teacher_input_token_ids")
    _receipt_require(
        teacher_inputs == teacher["teacher_input_token_ids"],
        "teacher inputs differ from prompt[-1] plus canonical free-run prefix127",
    )
    _receipt_require(
        teacher_inputs[0] == prompt[-1]
        and _sha256_compact(teacher_inputs) == teacher["teacher_input_token_ids_sha256"],
        "teacher input derivation or canonical hash mismatch",
    )
    reference = receipt.get("reference_argmax_token_ids")
    observed = receipt.get("observed_argmax_token_ids")
    expected_trajectory_hash = native["free_run_trajectory_admission"][
        "expected_sha256"
    ]
    _receipt_require(
        isinstance(reference, list)
        and len(reference) == teacher["steps"]
        and all(_is_int(token) and token >= 0 for token in reference),
        "teacher reference must contain 128 raw argmax token IDs",
    )
    _receipt_require(
        _sha256_compact(reference) == expected_trajectory_hash,
        "teacher reference argmax trajectory is not the frozen free128 trajectory",
    )
    _receipt_require(observed == reference, "teacher observed argmax trajectory diverged")
    _receipt_require(receipt.get("mismatch_positions") == [], "teacher mismatch positions are not empty")
    _receipt_require(receipt.get("first_mismatch") is None, "teacher first mismatch is not null")
    custody = receipt.get("custody")
    _receipt_require(custody == expected_custody, "teacher runtime custody mismatch")
    return receipt


def _validate_native_teacher_common(
    raw: Any,
    *,
    role: str,
    arm: str,
    contract: dict[str, Any],
) -> tuple[dict[str, Any], list[int]]:
    raw = _require_mapping(raw, f"{role} teacher raw receipt")
    teacher = contract["workload_contracts"]["NATIVE_RAW13_FREE128_V3"][
        "teacher_forced_admission"
    ]
    prompt = contract["workload_contracts"]["shared_prompt"]["token_ids"]
    _require_exact_fields(
        raw,
        {
            "format": "apxinf-qwen35-native-teacher-runtime-receipt-v3",
            "schema_version": 3,
            "campaign_id": contract["campaign_id"],
            "subcampaign_id": contract["comparison_graph"]["edges"][EDGE_ID][
                "subcampaign_id"
            ],
            "edge_id": EDGE_ID,
            "mode": "native-v3-teacher",
            "teacher_role": role,
            "arm": arm,
            "prefill_prompt_token_ids": prompt[:-1],
            "prefill_prompt_token_ids_count": 12,
            "teacher_input_token_ids": teacher["teacher_input_token_ids"],
            "teacher_input_token_ids_sha256": teacher[
                "teacher_input_token_ids_sha256"
            ],
            "eog_termination": False,
            "passed": True,
        },
        f"{role} teacher raw receipt",
        additional_fields=(
            ("reference_argmax_token_ids", "custody")
            if role == "reference"
            else (
                "reference_argmax_token_ids",
                "tail_normalized_hidden_f32_argmax_token_ids",
                "tail_top4_candidate_token_ids",
                "observed_argmax_token_ids",
                "mismatch_positions",
                "first_mismatch",
                "next_greedy_token_ready_elapsed_ns",
                "accelerator_candidate_elapsed_ns",
                "f32_tied_rerank_elapsed_ns",
                "selection_work_included",
                "accelerator_completion_before_each_token_ready_timestamp",
                "prefill_path",
                "final_path",
                "custody",
            )
        ),
    )
    reference = raw.get("reference_argmax_token_ids")
    _receipt_require(
        isinstance(reference, list)
        and len(reference) == 128
        and all(_is_int(token) and token >= 0 for token in reference)
        and _sha256_compact(reference)
        == contract["workload_contracts"]["NATIVE_RAW13_FREE128_V3"][
            "free_run_trajectory_admission"
        ]["expected_sha256"],
        f"{role} teacher reference trajectory mismatch",
    )
    return raw, reference


def _custody_core_matches(
    actual: dict[str, Any], expected: dict[str, Any], label: str
) -> None:
    for field in (
        "runner",
        "model",
        "runtime_source_commit",
        "loaded_non_system_library_closure_sha256",
    ):
        _receipt_require(actual.get(field) == expected.get(field), f"{label} custody {field} mismatch")
    _receipt_require(
        actual.get("fresh_process") is True
        and actual.get("start_end_identity_equal") is True,
        f"{label} did not prove fresh-process immutable custody",
    )


def _validate_an_source_custody(custody: dict[str, Any], label: str) -> None:
    """Require the runner's independently captured start/end source closure."""

    start = _require_mapping(
        custody.get("source_custody_start"), f"{label} source custody start"
    )
    end = _require_mapping(
        custody.get("source_custody_end"), f"{label} source custody end"
    )
    _receipt_require(start and end, f"{label} source custody start/end is empty")
    start_binary = _require_mapping(start.get("binary"), f"{label} start binary")
    end_binary = _require_mapping(end.get("binary"), f"{label} end binary")
    start_profile = _require_mapping(
        start.get("profile"), f"{label} start deployment profile"
    )
    end_profile = _require_mapping(
        end.get("deployment_profile"), f"{label} end deployment profile"
    )
    start_lock = _require_mapping(
        start.get("source_lock"), f"{label} start source lock"
    )
    end_lock = _require_mapping(
        end.get("source_lock"), f"{label} end source lock"
    )
    start_model = _require_mapping(start.get("model_dir"), f"{label} start model directory")
    end_model = _require_mapping(end.get("model_dir"), f"{label} end model directory")
    start_sources = _require_mapping(start.get("sources"), f"{label} start source set")
    start_gate = _require_mapping(start_sources.get("gate"), f"{label} start gate source")
    end_gate = _require_mapping(end.get("gate"), f"{label} end gate source")
    start_rust = _require_mapping(
        start_sources.get("rust_and_bridge_sources"),
        f"{label} start Rust/bridge sources",
    )
    end_rust = _require_mapping(
        end.get("rust_and_bridge_sources"),
        f"{label} end Rust/bridge sources",
    )
    start_metal = _require_mapping(
        start_sources.get("compiled_metal_shader_sources"),
        f"{label} start Metal sources",
    )
    end_metal = _require_mapping(
        end.get("compiled_metal_shader_sources"),
        f"{label} end Metal sources",
    )
    start_artifacts = _require_mapping(
        start_model.get("artifacts"), f"{label} start model artifacts"
    )
    end_artifacts = _require_mapping(
        end_model.get("artifacts"), f"{label} end model artifacts"
    )
    _receipt_require(
        start_binary
        and start_profile
        and start_lock
        and start_gate
        and start_rust
        and start_metal
        and start_artifacts,
        f"{label} source/model/binary closure is empty",
    )
    _receipt_require(
        end.get("verified_at_end") is True
        and start_binary == end_binary
        and start_profile.get("path") == end_profile.get("path")
        and start_profile.get("file_size") == end_profile.get("size")
        and start_profile.get("file_sha256") == end_profile.get("sha256")
        and start_profile.get("direct_regular_file") is True
        and end_profile.get("direct_regular_file") is True
        and start_profile.get("single_link") is True
        and end_profile.get("single_link") is True
        and start_lock.get("path") == end_lock.get("path")
        and start_lock.get("file_size") == end_lock.get("size")
        and start_lock.get("file_sha256") == end_lock.get("sha256")
        and start_lock.get("direct_regular_file") is True
        and end_lock.get("direct_regular_file") is True
        and start_lock.get("single_link") is True
        and end_lock.get("single_link") is True
        and start_model.get("path") == end_model.get("path")
        and start_artifacts == end_artifacts
        and start_model.get("cache_present")
        == end_model.get("cache_present")
        and end_model.get("loaded_from_start_pinned_artifacts") is True
        and start_gate == end_gate
        and start_rust == end_rust
        and start_metal == end_metal
        and start_sources.get("captured_at_start") is True
        and start_sources.get(
            "binary_attestation_authoritative_for_full_executable"
        )
        is True
        and isinstance(start_sources.get("set_id"), str)
        and bool(start_sources["set_id"])
        and isinstance(start_sources.get("coverage"), str)
        and bool(start_sources["coverage"])
        and start_sources.get("set_id") == end.get("source_set_id")
        and start_sources.get("coverage") == end.get("source_set_coverage"),
        f"{label} source/model/binary custody start/end evidence disagrees",
    )


def _an_repository_source_file_claims(
    custody: dict[str, Any], repository_root: Path, label: str
) -> dict[str, dict[str, Any]]:
    """Flatten every repository source attestation emitted by the AN runner."""

    start = _require_mapping(
        custody.get("source_custody_start"), f"{label} source custody start"
    )
    sources = _require_mapping(start.get("sources"), f"{label} source set")
    candidates: list[tuple[str, dict[str, Any], str, str]] = [
        (
            "deployment_profile",
            _require_mapping(start.get("profile"), f"{label} profile"),
            "file_size",
            "file_sha256",
        ),
        (
            "source_lock",
            _require_mapping(start.get("source_lock"), f"{label} source lock"),
            "file_size",
            "file_sha256",
        ),
        (
            "gate",
            _require_mapping(sources.get("gate"), f"{label} gate source"),
            "size",
            "sha256",
        ),
    ]
    for group_name, field in (
        ("rust_and_bridge_sources", "rust_and_bridge_sources"),
        ("compiled_metal_shader_sources", "compiled_metal_shader_sources"),
    ):
        group = _require_mapping(
            sources.get(field), f"{label} {group_name}"
        )
        for name, attestation in sorted(group.items()):
            _receipt_require(
                isinstance(name, str) and name,
                f"{label} source receipt name is invalid",
            )
            candidates.append(
                (
                    f"{group_name}.{name}",
                    _require_mapping(
                        attestation, f"{label} source {group_name}.{name}"
                    ),
                    "size",
                    "sha256",
                )
            )

    root_text = str(repository_root)
    prefix = root_text + os.sep
    claims: dict[str, dict[str, Any]] = {}
    for receipt_name, attestation, size_field, hash_field in candidates:
        absolute_path = attestation.get("path")
        size_bytes = attestation.get(size_field)
        sha256 = attestation.get(hash_field)
        _receipt_require(
            isinstance(absolute_path, str)
            and absolute_path.startswith(prefix)
            and os.path.normpath(absolute_path) == absolute_path,
            f"{label} repository source path is not normalized under the checkout",
        )
        repository_path = absolute_path[len(prefix) :]
        _receipt_require(
            repository_path
            and repository_path not in claims
            and _is_int(size_bytes)
            and size_bytes > 0
            and _valid_sha256(sha256),
            f"{label} repository source file claim is invalid or duplicated",
        )
        claims[repository_path] = {
            "receipt_name": receipt_name,
            "repository_path": repository_path,
            "size_bytes": size_bytes,
            "sha256": sha256,
        }
    _receipt_require(claims, f"{label} repository source file set is empty")
    return dict(sorted(claims.items()))


def _git_blob_at_revision(
    repository_root: Path,
    revision: str,
    repository_path: str,
    oid_length: int,
    command_runner: Any,
) -> dict[str, Any]:
    raw = _git_success(
        command_runner,
        repository_root,
        ["ls-tree", "-z", revision, "--", repository_path],
    )
    _require(
        raw.endswith(b"\0") and raw.count(b"\0") == 1,
        f"Git source blob is absent or ambiguous: {revision}:{repository_path}",
    )
    try:
        metadata, path_raw = raw[:-1].split(b"\t", 1)
        mode_raw, kind_raw, oid_raw = metadata.split(b" ", 2)
        path = path_raw.decode("utf-8", errors="strict")
        oid = oid_raw.decode("ascii", errors="strict")
    except (ValueError, UnicodeDecodeError) as error:
        raise CampaignError(
            f"Git source tree entry is malformed: {revision}:{repository_path}"
        ) from error
    _require(
        mode_raw in (b"100644", b"100755")
        and kind_raw == b"blob"
        and path == repository_path
        and len(oid) == oid_length
        and all(character in "0123456789abcdef" for character in oid),
        f"Git source tree entry is invalid: {revision}:{repository_path}",
    )
    blob = _git_success(
        command_runner, repository_root, ["cat-file", "blob", oid]
    )
    return {
        "repository_path": repository_path,
        "blob_oid": oid,
        "size_bytes": len(blob),
        "sha256": hashlib.sha256(blob).hexdigest(),
    }


def _an_evidence_only_descendant_paths(plan: dict[str, Any]) -> set[str]:
    allowed = {plan["plan_repository_path"], plan["marker_repository_path"]}
    for arm in ("AN", "L"):
        allowed.add(
            plan["teacher_receipts"][arm]["reference_repository_path"]
        )
        allowed.add(
            plan["teacher_receipts"][arm]["runtime_repository_path"]
        )
    return allowed


def bind_an_source_custodies_to_git(
    repository_root: Path | str,
    plan: dict[str, Any],
    git_custody: dict[str, Any],
    source_custodies: list[dict[str, Any]],
    *,
    command_runner: Any = _system_command_runner,
    expected_source_tree: str | None = None,
) -> dict[str, Any]:
    """Bind all AN raw source receipts to one ancestor tree and live HEAD."""

    repo = Path(repository_root).resolve(strict=True)
    commit = validate_an_campaign_commit(
        plan, git_custody, repo, command_runner
    )
    if expected_source_tree is not None:
        _require(
            commit["campaign_tree"] == expected_source_tree,
            "AN campaign source tree drifted after pre-marker admission",
        )
    _require(
        isinstance(source_custodies, list) and source_custodies,
        "AN raw source custodities are absent",
    )
    claim_sets: list[dict[str, dict[str, Any]]] = []
    for index, custody_value in enumerate(source_custodies):
        label = f"AN raw source custody {index}"
        custody = _require_mapping(custody_value, label)
        _validate_an_source_custody(custody, label)
        claim_sets.append(
            _an_repository_source_file_claims(custody, repo, label)
        )
    canonical_claims = claim_sets[0]
    _receipt_require(
        all(claims == canonical_claims for claims in claim_sets[1:]),
        "AN teacher raw source receipts disagree",
    )
    oid_length = 40 if git_custody.get("object_format", "sha1") == "sha1" else 64
    files: dict[str, Any] = {}
    for repository_path, claim in canonical_claims.items():
        source_blob = _git_blob_at_revision(
            repo,
            commit["campaign_commit"],
            repository_path,
            oid_length,
            command_runner,
        )
        head_blob = _git_blob_at_revision(
            repo,
            git_custody["head_commit"],
            repository_path,
            oid_length,
            command_runner,
        )
        _receipt_require(
            claim["size_bytes"]
            == source_blob["size_bytes"]
            == head_blob["size_bytes"]
            and claim["sha256"]
            == source_blob["sha256"]
            == head_blob["sha256"]
            and source_blob["blob_oid"] == head_blob["blob_oid"],
            f"AN raw source blob/size/hash drifted: {repository_path}",
        )
        files[repository_path] = {
            **copy.deepcopy(claim),
            "source_commit_blob_oid": source_blob["blob_oid"],
            "live_head_blob_oid": head_blob["blob_oid"],
            "source_and_live_blob_equal": True,
        }

    diff_raw = _git_success(
        command_runner,
        repo,
        [
            "diff",
            "--name-only",
            "-z",
            "--no-renames",
            commit["campaign_commit"],
            git_custody["head_commit"],
            "--",
        ],
    )
    _require(
        diff_raw == b"" or diff_raw.endswith(b"\0"),
        "AN source-to-live Git diff path list is malformed",
    )
    try:
        changed_paths = (
            {
                item.decode("utf-8", errors="strict")
                for item in diff_raw[:-1].split(b"\0")
            }
            if diff_raw
            else set()
        )
    except UnicodeDecodeError as error:
        raise CampaignError("AN source-to-live Git diff path is not UTF-8") from error
    allowed_evidence = _an_evidence_only_descendant_paths(plan)
    _require(
        changed_paths.issubset(allowed_evidence),
        "AN source commit descendants changed non-evidence paths",
    )
    for path in changed_paths:
        source_entry = _git_success(
            command_runner,
            repo,
            [
                "ls-tree",
                "-z",
                commit["campaign_commit"],
                "--",
                path,
            ],
        )
        _require(
            source_entry == b"",
            f"AN source descendant modified rather than appended evidence: {path}",
        )
    return {
        **commit,
        "repository_root": str(repo),
        "source_file_count": len(files),
        "source_files": files,
        "source_files_sha256": _sha256_compact(files),
        "live_descendant_changed_paths": sorted(changed_paths),
        "live_descendant_changes_are_append_only_evidence": True,
    }


def _validate_an_source_custody_binding(
    custody: dict[str, Any], expected_binding: dict[str, Any], label: str
) -> None:
    binding = _require_mapping(expected_binding, f"{label} Git source binding")
    root_text = binding.get("repository_root")
    _receipt_require(
        isinstance(root_text, str) and root_text.startswith("/"),
        f"{label} Git source binding repository root is invalid",
    )
    claims = _an_repository_source_file_claims(
        custody, Path(root_text), label
    )
    files = _require_mapping(
        binding.get("source_files"), f"{label} Git source files"
    )
    expected_claims = {
        path: {
            field: entry[field]
            for field in (
                "receipt_name",
                "repository_path",
                "size_bytes",
                "sha256",
            )
        }
        for path, entry in files.items()
        if isinstance(entry, dict)
        and all(
            field in entry
            for field in (
                "receipt_name",
                "repository_path",
                "size_bytes",
                "sha256",
            )
        )
    }
    _receipt_require(
        claims == expected_claims
        and custody.get("runtime_source_commit")
        == binding.get("campaign_commit"),
        f"{label} raw source custody is not bound to the admitted Git tree",
    )


def _validate_an_dynamic_custody(
    custody: dict[str, Any],
    contract: dict[str, Any],
    expected_closure_sha256: str,
    label: str,
    expected_source_binding: dict[str, Any] | None = None,
) -> None:
    closure = custody.get("loaded_non_system_library_closure")
    closure_start = custody.get("loaded_non_system_library_closure_start")
    closure_end = custody.get("loaded_non_system_library_closure_end")
    closure_sha256 = custody.get("loaded_non_system_library_closure_sha256")
    _receipt_require(
        isinstance(closure, list)
        and closure_start == closure_end == closure
        and _sha256_compact(closure) == closure_sha256 == expected_closure_sha256
        and custody.get("loaded_non_system_library_closure_start_sha256")
        == closure_sha256
        and custody.get("loaded_non_system_library_closure_end_sha256")
        == closure_sha256,
        f"{label} loaded non-system library closure start/end mismatch",
    )
    runtime = _require_mapping(
        custody.get("thread_policy_runtime"), f"{label} thread-policy runtime"
    )
    _require_exact_fields(
        runtime,
        {
            "logical_cpu_count": contract["scope"]["host"][
                "logical_cpu_count"
            ],
            "logical_cpu_count_source": "std::thread::available_parallelism",
            "fixed_worker_count_claimed": False,
            "environment_overrides_absent": True,
            "absent_environment_overrides": list(
                AN_THREAD_OVERRIDE_ENVIRONMENT
            ),
        },
        f"{label} thread-policy runtime",
    )
    _validate_an_source_custody(custody, label)
    if expected_source_binding is not None:
        _validate_an_source_custody_binding(
            custody, expected_source_binding, label
        )


def build_teacher_admission_receipt(
    arm: str,
    reference_raw: Any,
    runtime_raw: Any,
    reference_file: dict[str, Any],
    runtime_file: dict[str, Any],
    contract: dict[str, Any],
    expected_custody: dict[str, Any],
    reference_expected_custody: dict[str, Any] | None = None,
    expected_artifact_observation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Merge separately persisted CPU-reference and observed teacher receipts."""

    _receipt_require(arm in ("AN", "L"), "teacher admission arm is invalid")
    reference_expected = reference_expected_custody or expected_custody
    reference_raw, reference = _validate_native_teacher_common(
        reference_raw,
        role="reference",
        arm="CPU_REFERENCE",
        contract=contract,
    )
    reference_custody = _require_mapping(
        reference_raw.get("custody"), "CPU reference custody"
    )
    _custody_core_matches(reference_custody, reference_expected, "CPU reference")
    _validate_an_dynamic_custody(
        reference_custody,
        contract,
        reference_expected["loaded_non_system_library_closure_sha256"],
        "CPU reference",
    )
    _receipt_require(
        reference_custody.get("configuration_id")
        == "ApxInf-native-CPU-F32-teacher-reference-v3"
        and reference_custody.get("packed_weight_and_resident_buffer_manifest_sha256")
        is None
        and reference_custody.get("deployment")
        == {
            "constructor_id": "from_weights",
            "context_capacity_tokens": 256,
            "prefill_device": "CPU",
            "prefill_precision": "F32",
            "full_attention_device": "CPU",
            "full_attention_precision": "F32",
            "kv_key_dtype": "F32",
            "kv_value_dtype": "F32",
            "head": "CPU/F32 full-vocabulary tied argmax",
            "teacher_reference_only": True,
        },
        "CPU reference named deployment receipt mismatch",
    )

    if arm == "AN":
        runtime_raw, runtime_reference = _validate_native_teacher_common(
            runtime_raw,
            role="observed",
            arm="AN",
            contract=contract,
        )
        observed = runtime_raw.get("observed_argmax_token_ids")
        _receipt_require(
            runtime_reference == reference
            and observed == reference
            and runtime_raw.get("tail_normalized_hidden_f32_argmax_token_ids")
            == reference
            and runtime_raw.get("mismatch_positions") == []
            and runtime_raw.get("first_mismatch") is None,
            "AN teacher observed trajectory diverged",
        )
        candidates = runtime_raw.get("tail_top4_candidate_token_ids")
        _receipt_require(
            isinstance(candidates, list)
            and len(candidates) == 128
            and all(
                isinstance(row, list)
                and len(row) == 4
                and len(set(row)) == 4
                and reference[index] in row
                and all(_is_int(token) and token >= 0 for token in row)
                for index, row in enumerate(candidates)
            ),
            "AN teacher top-4 exact-winner custody failed",
        )
        for field in (
            "next_greedy_token_ready_elapsed_ns",
            "accelerator_candidate_elapsed_ns",
            "f32_tied_rerank_elapsed_ns",
        ):
            values = runtime_raw.get(field)
            _receipt_require(
                isinstance(values, list)
                and len(values) == 128
                and all(_is_int(value) and value > 0 for value in values),
                f"AN teacher {field} is invalid",
            )
            if field == "next_greedy_token_ready_elapsed_ns":
                _receipt_require(
                    all(left < right for left, right in zip(values, values[1:])),
                    "AN teacher next-greedy-token-ready times are not strictly ordered",
                )
        _receipt_require(
            runtime_raw.get("selection_work_included") is True
            and runtime_raw.get(
                "accelerator_completion_before_each_token_ready_timestamp"
            )
            is True,
            "AN teacher next-greedy-token-ready boundary is incomplete",
        )
        for phase in ("prefill_path", "final_path"):
            path = _require_mapping(runtime_raw.get(phase), f"AN teacher {phase}")
            checks = _require_mapping(path.get("path_checks"), f"AN teacher {phase} checks")
            _receipt_require(checks.get("all_valid") is True, f"AN teacher {phase} failed")
        runtime_custody = _require_mapping(
            runtime_raw.get("custody"), "AN teacher runtime custody"
        )
        _custody_core_matches(runtime_custody, expected_custody, "AN teacher")
        _validate_an_dynamic_custody(
            runtime_custody,
            contract,
            expected_custody["loaded_non_system_library_closure_sha256"],
            "AN teacher",
        )
        _receipt_require(
            runtime_custody.get("configuration_id")
            == expected_custody["configuration_id"]
            and runtime_custody.get("deployment")
            == expected_deployment_receipt("AN")
            and runtime_custody.get(
                "packed_weight_and_resident_buffer_manifest_sha256"
            )
            == expected_custody[
                "packed_weight_and_resident_buffer_manifest_sha256"
            ],
            "AN teacher named deployment custody mismatch",
        )
    else:
        _receipt_require(
            expected_artifact_observation is not None,
            "llama teacher preflight artifact observation is absent",
        )
        observed = _validate_llama_teacher_raw(
            runtime_raw,
            reference,
            contract,
            expected_custody,
            expected_artifact_observation,
        )

    admission = {
        "format": TEACHER_FORMAT,
        "schema_version": 3,
        "campaign_id": contract["campaign_id"],
        "subcampaign_id": contract["comparison_graph"]["edges"][EDGE_ID][
            "subcampaign_id"
        ],
        "edge_id": EDGE_ID,
        "arm": arm,
        "mode": "native-v3-teacher",
        "prefill_prompt_token_ids": contract["workload_contracts"][
            "shared_prompt"
        ]["token_ids"][:-1],
        "prefill_prompt_token_ids_count": 12,
        "teacher_input_token_ids": contract["workload_contracts"][
            "NATIVE_RAW13_FREE128_V3"
        ]["teacher_forced_admission"]["teacher_input_token_ids"],
        "teacher_input_token_ids_sha256": contract["workload_contracts"][
            "NATIVE_RAW13_FREE128_V3"
        ]["teacher_forced_admission"]["teacher_input_token_ids_sha256"],
        "reference_argmax_token_ids": reference,
        "observed_argmax_token_ids": observed,
        "mismatch_positions": [],
        "first_mismatch": None,
        "reference_receipt_size_bytes": reference_file.get("size_bytes"),
        "reference_receipt_sha256": reference_file.get("sha256"),
        "runtime_receipt_size_bytes": runtime_file.get("size_bytes"),
        "runtime_receipt_sha256": runtime_file.get("sha256"),
        "custody": expected_custody,
        "source_receipts": {
            "reference": reference_raw,
            "runtime": runtime_raw,
        },
    }
    return validate_teacher_receipt(
        admission,
        arm,
        contract,
        reference_file,
        runtime_file,
        expected_custody,
    )


def _validate_llama_teacher_raw(
    raw: Any,
    reference: list[int],
    contract: dict[str, Any],
    expected_custody: dict[str, Any],
    expected_artifact_observation: dict[str, Any],
) -> list[int]:
    """Strict adapter hook for the pinned llama native-v3 teacher schema."""

    raw = _require_mapping(raw, "llama teacher raw receipt")
    _require_exact_fields(
        raw,
        {
            "schema": "apxinf.llama-cpp.raw-token-diagnostic.v3",
            "ok": True,
            "mode": "native-v3-teacher",
            "token_ready_boundary": "next-greedy-token-ready",
            "selection_work_included": True,
            "accelerator_completion_before_each_token_ready_timestamp": True,
        },
        "llama teacher raw receipt",
        additional_fields=(
            "contract",
            "model",
            "parameters",
            "output",
            "teacher_forced",
            "timings",
            "llama_perf",
            "runtime_custody",
            "backend",
            "placement_attestation",
            "post_measurement_execution_proof",
            "build",
        ),
    )
    prompt = contract["workload_contracts"]["shared_prompt"]["token_ids"]
    teacher_contract = contract["workload_contracts"]["NATIVE_RAW13_FREE128_V3"][
        "teacher_forced_admission"
    ]
    raw_contract = _require_mapping(raw.get("contract"), "llama teacher contract")
    _require_exact_fields(
        raw_contract,
        {
            "prompt_token_ids": prompt,
            "mode": "native-v3-teacher",
            "sampling": "greedy-argmax",
            "generated_token_count": 128,
            "eog_termination": False,
            "token_ready_elapsed_ns_origin": "immediately-before-teacher-step-0-decode",
            "final_sampled_token_is_not_decoded_in_timed_workload": True,
            "final_sampled_token_decoded_once_post_measurement_for_execution_proof": True,
        },
        "llama teacher contract",
    )
    teacher = _require_mapping(raw.get("teacher_forced"), "llama teacher-forced proof")
    _require_exact_fields(
        teacher,
        {
            "raw_prompt_token_ids": prompt,
            "raw_prefill_token_ids": prompt[:-1],
            "teacher_input_token_ids": teacher_contract["teacher_input_token_ids"],
            "reference_argmax_token_ids": reference,
            "observed_argmax_token_ids": reference,
            "mismatch_positions": [],
            "first_mismatch": None,
            "mismatch_count": 0,
            "exact_128_of_128": True,
            "raw_prefill_token_count": 12,
            "teacher_step_count": 128,
            "teacher_step_input_position_first_zero_based": 12,
            "teacher_step_input_position_last_zero_based": 139,
            "context_token_count_before_execution_proof": 140,
            "teacher_input_derivation": "prompt[-1]+canonical_free[:127]",
            "teacher_input_derivation_recomputed_and_matched": True,
            "teacher_input_token_ids_sha256": teacher_contract[
                "teacher_input_token_ids_sha256"
            ],
            "reference_argmax_token_ids_sha256": contract[
                "workload_contracts"
            ]["NATIVE_RAW13_FREE128_V3"]["free_run_trajectory_admission"][
                "expected_sha256"
            ],
            "argmax_scope": "all-248320-raw-logits-lowest-token-id-wins-ties",
            "argmax_timing_included": True,
            "eog_termination": False,
        },
        "llama teacher-forced proof",
        additional_fields=("next_greedy_token_ready_elapsed_ns",),
    )
    next_ready = teacher.get("next_greedy_token_ready_elapsed_ns")
    _receipt_require(
        isinstance(next_ready, list)
        and len(next_ready) == 128
        and all(_is_int(value) and value > 0 for value in next_ready),
        "llama teacher per-step token-ready times are invalid",
    )
    timings = _require_mapping(raw.get("timings"), "llama teacher timings")
    _receipt_require(
        _is_int(timings.get("teacher_prefill_elapsed_ns"))
        and timings["teacher_prefill_elapsed_ns"] > 0,
        "llama teacher raw12 prefill timing is invalid",
    )
    perf = _require_mapping(raw.get("llama_perf"), "llama teacher performance counters")
    _require_exact_fields(
        _require_mapping(perf.get("context"), "llama teacher context counters"),
        {"n_prompt_eval": 12, "n_eval": 128, "n_reused": 127},
        "llama teacher context counters",
        additional_fields=(
            "t_start_ms",
            "t_load_ms",
            "t_prompt_eval_ms",
            "t_eval_ms",
        ),
    )
    _require_exact_fields(
        _require_mapping(perf.get("sampler"), "llama teacher sampler counters"),
        {"n_sample": 0},
        "llama teacher sampler counters",
        additional_fields=("t_sample_ms",),
    )

    # Reuse the free adapter for the identical model, context, backend,
    # placement, execution-proof, build and token-ready ledgers.  Only fields
    # whose teacher semantics were checked above are projected to their free
    # counterparts; no unvalidated evidence is invented.
    common = copy.deepcopy(raw)
    common["mode"] = "native-v3-free"
    common["contract"]["mode"] = "native-v3-free"
    common["contract"][
        "token_ready_elapsed_ns_origin"
    ] = "immediately-before-prompt-decode"
    common["llama_perf"]["context"]["n_prompt_eval"] = 13
    common["llama_perf"]["context"]["n_eval"] = 127
    common["llama_perf"]["context"]["n_reused"] = 126
    common["timings"].pop("teacher_prefill_elapsed_ns", None)
    common.pop("teacher_forced", None)
    slot = next(
        item
        for item in declared_schedule()
        if item["arm"] == "L" and item["phase"] == "warmup"
    )
    adapt_llama_free_receipt(
        common,
        slot,
        "0" * 64,
        contract,
        expected_custody,
        expected_artifact_observation,
    )
    output = _require_mapping(raw.get("output"), "llama teacher output")
    observed = output.get("token_ids")
    _receipt_require(observed == reference, "llama teacher output trajectory diverged")
    return observed


def validate_host_receipt(
    receipt: Any, phase: str, contract: dict[str, Any]
) -> dict[str, Any]:
    """Validate an exact-host, power, memory and process quiet-gate receipt."""

    _require(phase in ("preflight", "continuous", "postflight"), "unknown host receipt phase")
    _require(isinstance(receipt, dict), "host receipt must be an object")
    _require(receipt.get("format") == HOST_FORMAT, "host receipt format mismatch")
    _require(receipt.get("schema_version") == 3, "host receipt schema mismatch")
    _require(receipt.get("phase") == phase, "host receipt phase mismatch")
    gate = contract["host_quiet_gate"]
    expected_host = contract["scope"]["host"]
    actual_host = receipt.get("host")
    _require(isinstance(actual_host, dict), "host identity receipt is absent")
    for field in (
        "model_identifier",
        "chip",
        "architecture",
        "logical_cpu_count",
        "memory_bytes",
        "os_product",
        "os_version",
        "os_build",
    ):
        _require(actual_host.get(field) == expected_host[field], f"host identity {field} mismatch")
    _require(receipt.get("power_source") == gate["power_source_required"], "host is not on required AC power")
    _require(receipt.get("thermal_warning") is False, "host thermal warning is active")
    _require(receipt.get("performance_warning") is False, "host performance warning is active")
    _require(receipt.get("processes_terminated_or_modified") is False, "host processes were modified by the campaign")
    _require(receipt.get("passed") is True, "quiet-host gate did not pass")
    phase_contract = gate[phase] if phase != "continuous" else gate["continuous_monitor"]
    expected_interval = (
        phase_contract["sample_interval_ms"]
        if phase == "continuous"
        else phase_contract["snapshot_interval_ms"]
    )
    _require(receipt.get("snapshot_interval_ms") == expected_interval, "host snapshot interval mismatch")
    swap_start = receipt.get("system_swap_used_bytes_start")
    throttled_start = receipt.get("memory_pressure_pages_throttled_start")
    _require(
        _is_int(swap_start) and swap_start >= 0,
        "host initial system swap observation is invalid",
    )
    _require(
        _is_int(throttled_start) and throttled_start >= 0,
        "host initial memory-pressure observation is invalid",
    )
    snapshots = receipt.get("snapshots")
    _require(isinstance(snapshots, list), "host snapshots are absent")
    if phase in ("preflight", "postflight"):
        _require(len(snapshots) == phase_contract["snapshot_count"], "host snapshot count mismatch")
    else:
        _require(len(snapshots) >= 1, "continuous host monitor recorded no snapshot")
    maximum_single = gate["process_policy"]["maximum_single_non_allowlisted_process_cpu_percent"]
    maximum_aggregate = gate["process_policy"]["maximum_aggregate_non_allowlisted_process_cpu_percent"]
    maximum_load = gate["system_policy"]["maximum_load_average_per_logical_cpu"]
    previous_monotonic: int | None = None
    for expected_index, snapshot in enumerate(snapshots):
        _require(isinstance(snapshot, dict), "host snapshot is not an object")
        _require(snapshot.get("index") == expected_index, "host snapshot index drifted")
        monotonic_ns = snapshot.get("monotonic_ns")
        window_start_ns = snapshot.get("cpu_window_start_monotonic_ns")
        _require(_is_int(monotonic_ns) and monotonic_ns >= 0, "host snapshot monotonic time is invalid")
        _require(
            _is_int(window_start_ns)
            and 0 <= window_start_ns < monotonic_ns,
            "host CPU window start is invalid",
        )
        actual_window_ms = (monotonic_ns - window_start_ns) / 1_000_000
        reported_window_ms = snapshot.get("cpu_percent_window_ms")
        _require(
            isinstance(reported_window_ms, (int, float))
            and not isinstance(reported_window_ms, bool)
            and math.isfinite(float(reported_window_ms))
            and math.isclose(
                float(reported_window_ms),
                actual_window_ms,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
            and expected_interval
            <= actual_window_ms
            <= expected_interval + HOST_CPU_WINDOW_MAX_OVERRUN_MS,
            "host CPU measurement window is not the actual bounded 250-ms window",
        )
        _require(
            snapshot.get("cpu_measurement_source")
            == "libproc-PROC_PIDTASKINFO-delta",
            "host CPU measurement source mismatch",
        )
        if previous_monotonic is not None:
            _require(
                window_start_ns == previous_monotonic
                and monotonic_ns > previous_monotonic,
                "host CPU windows are not contiguous",
            )
        previous_monotonic = monotonic_ns
        allowlist = snapshot.get("resolved_allowlist")
        _require(isinstance(allowlist, list), "host snapshot resolved allowlist is absent")
        roles = {
            entry.get("role")
            for entry in allowlist
            if isinstance(entry, dict)
        }
        _require(
            {"campaign_orchestrator", "custody_monitor"}.issubset(roles),
            "host snapshot did not resolve orchestrator and custody monitor identities",
        )
        processes = snapshot.get("nonallowlisted_processes")
        _require(isinstance(processes, list), "host snapshot nonallowlisted process inventory is absent")
        _require(
            snapshot.get("vanished_nonallowlisted_processes") == []
            and snapshot.get("cpu_window_proof_complete") is True,
            "host CPU inventory proof is incomplete",
        )
        cpu_values: list[float] = []
        for process in processes:
            _require(isinstance(process, dict), "host process entry is not an object")
            _require(_is_int(process.get("pid")) and process["pid"] > 0, "host process PID is invalid")
            _require(isinstance(process.get("process_start_time"), str) and process["process_start_time"], "host process start identity is absent")
            cpu = process.get("cpu_percent")
            _require(isinstance(cpu, (int, float)) and not isinstance(cpu, bool) and math.isfinite(float(cpu)) and cpu >= 0, "host process CPU is invalid")
            cpu_values.append(float(cpu))
        observed_max = max(cpu_values, default=0.0)
        observed_total = sum(cpu_values)
        _require(
            math.isclose(
                float(snapshot.get("maximum_single_nonallowlisted_process_cpu_percent", -1)),
                observed_max,
                rel_tol=0.0,
                abs_tol=1e-9,
            ),
            "quiet-host maximum process CPU arithmetic mismatch",
        )
        _require(
            math.isclose(
                float(snapshot.get("aggregate_nonallowlisted_process_cpu_percent", -1)),
                observed_total,
                rel_tol=0.0,
                abs_tol=1e-9,
            ),
            "quiet-host aggregate process CPU arithmetic mismatch",
        )
        _require(observed_max <= maximum_single, "quiet-host single process CPU threshold failed")
        _require(observed_total <= maximum_aggregate, "quiet-host aggregate process CPU threshold failed")
        load = snapshot.get("load_average_per_logical_cpu")
        _require(isinstance(load, (int, float)) and not isinstance(load, bool) and math.isfinite(float(load)) and 0 <= load <= maximum_load, "quiet-host load-average threshold failed")
        _require(snapshot.get("campaign_process_swap_bytes") == 0, "campaign process used swap")
        _require(
            snapshot.get("power_source") == gate["power_source_required"]
            and snapshot.get("thermal_warning") is False
            and snapshot.get("performance_warning") is False,
            "host power or thermal state failed during a snapshot",
        )
        _require(
            snapshot.get("system_swap_used_bytes") == swap_start
            and snapshot.get("memory_pressure_pages_throttled")
            == throttled_start
            and snapshot.get("system_state_matches_gate_start") is True,
            "host swap or memory-pressure state changed during a snapshot",
        )
        _require(
            isinstance(snapshot.get("campaign_process_swap_observations"), list)
            and isinstance(
                snapshot.get("campaign_swap_probe_vanished_processes"), list
            )
            and (
                snapshot.get("active_runtime_root_present") is None
                or isinstance(
                    snapshot.get("active_runtime_root_present"), bool
                )
            )
            and isinstance(
                snapshot.get("active_runtime_swap_proof_complete"), bool
            ),
            "campaign PID swap custody shape is invalid",
        )
        _require(snapshot.get("passed") is True, "quiet-host snapshot failed")
    _require(receipt.get("swap_delta_bytes") == 0, "host swap changed during gate")
    _require(receipt.get("memory_pressure_pages_throttled_delta") == 0, "host memory-pressure pages throttled changed")
    _require(receipt.get("power_or_thermal_state_changed") is False, "host power or thermal state changed")
    accepted_runtime_proofs = receipt.get("accepted_runtime_swap_proofs")
    _require(
        isinstance(accepted_runtime_proofs, list),
        "accepted runtime PID swap proofs are absent",
    )
    if phase != "continuous":
        _require(
            accepted_runtime_proofs == [],
            "non-continuous host gate contains runtime swap proofs",
        )
    seen_process_identities: set[tuple[int, str]] = set()
    for proof in accepted_runtime_proofs:
        observed_identity = (
            proof.get("observed_process_identity")
            if isinstance(proof, dict)
            else None
        )
        identity = (
            (proof.get("pid"), observed_identity.get("process_start_time"))
            if isinstance(observed_identity, dict)
            else None
        )
        _require(
            isinstance(proof, dict)
            and _is_int(proof.get("pid"))
            and proof["pid"] > 0
            and proof.get("arm") in ("AN", "L")
            and proof.get("swap_zero_proven") is True
            and isinstance(observed_identity, dict)
            and observed_identity.get("pid") == proof["pid"]
            and isinstance(observed_identity.get("process_start_time"), str)
            and bool(observed_identity["process_start_time"])
            and observed_identity.get("swapped_bytes") == 0
            and identity not in seen_process_identities,
            "accepted runtime PID swap proof is invalid or duplicated",
        )
        seen_process_identities.add(identity)
    return receipt


def _json_line_bytes(value: Any) -> bytes:
    return compact_json_bytes(value) + b"\n"


def _fsync_directory(directory: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(str(directory), flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_synced_temporary_json(path: Path, value: Any) -> Path:
    parent = path.parent.resolve(strict=True)
    _require(parent.is_dir(), f"JSON parent is not a directory: {parent}")
    temporary = parent / (
        f".formal-v3-{os.getpid()}-{secrets.token_hex(16)}.tmp"
    )
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(str(temporary), flags, 0o600)
    try:
        payload = _json_line_bytes(value)
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            _require(written > 0, f"short JSON write for {path}")
            offset += written
        os.fchmod(descriptor, 0o644)
        os.fsync(descriptor)
    except BaseException:
        os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise
    else:
        os.close(descriptor)
    return temporary


def atomic_create_json(path_value: Path | str, value: Any) -> None:
    """Durably publish one JSON line without ever replacing an existing path."""

    path = Path(path_value)
    _require(path.name not in ("", ".", ".."), "create-new JSON path is invalid")
    temporary = _write_synced_temporary_json(path, value)
    try:
        try:
            os.link(str(temporary), str(path), follow_symlinks=False)
        except FileExistsError as error:
            raise CampaignError(f"create-new JSON path already exists: {path}") from error
        _fsync_directory(path.parent.resolve(strict=True))
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        _fsync_directory(path.parent.resolve(strict=True))


def atomic_replace_json(path_value: Path | str, value: Any) -> None:
    """Durably replace one existing regular JSON file through same-dir rename."""

    path = Path(path_value)
    try:
        current = path.lstat()
    except FileNotFoundError as error:
        raise CampaignError(f"atomic JSON replacement target is absent: {path}") from error
    _require(stat.S_ISREG(current.st_mode), f"atomic JSON target is not regular: {path}")
    temporary = _write_synced_temporary_json(path, value)
    try:
        os.replace(str(temporary), str(path))
        _fsync_directory(path.parent.resolve(strict=True))
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def prepare_marker_after_preflight(
    marker_path: Path | str, preflight: Any
) -> dict[str, Any]:
    """Run every supplied pre-marker proof before the irreversible create-new."""

    _require(callable(preflight), "pre-marker preflight is not callable")
    marker = preflight()
    _require(isinstance(marker, dict), "pre-marker receipt is not an object")
    _require(marker.get("format") == MARKER_FORMAT, "campaign marker format mismatch")
    _require(marker.get("schema_version") == 3, "campaign marker schema mismatch")
    admission = marker.get("pre_marker_admission")
    _require(
        isinstance(admission, dict) and admission.get("all_passed") is True,
        "not every pre-marker admission gate passed",
    )
    atomic_create_json(marker_path, marker)
    return marker


def _population_cv(values: list[float], label: str) -> float:
    _require(values and all(math.isfinite(value) and value > 0 for value in values), f"{label} values are invalid")
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return math.sqrt(variance) / mean


def _mean(values: list[float], label: str) -> float:
    _require(values and all(math.isfinite(value) for value in values), f"{label} values are invalid")
    return sum(values) / len(values)


def _half_ratio_classification(ratio: float) -> str:
    if ratio < 0.95:
        return "AN_FASTER"
    if ratio > 1.05:
        return "L_FASTER"
    return "WITHIN_EQUIVALENCE_BAND"


def compute_native_statistics(
    samples: list[dict[str, Any]], contract: dict[str, Any]
) -> dict[str, Any]:
    """Compute the frozen eight-block log-ratio estimator and stability gates."""

    _require(isinstance(samples, list), "sample collection is not a list")
    by_sequence: dict[int, dict[str, Any]] = {}
    for sample in samples:
        _require(isinstance(sample, dict), "statistics sample is not an object")
        request = sample.get("request")
        _require(isinstance(request, dict), "statistics sample request is absent")
        sequence = request.get("sequence_index")
        _require(_is_int(sequence) and sequence not in by_sequence, "statistics sample sequence is invalid or duplicated")
        by_sequence[sequence] = sample
    timed_slots = [slot for slot in declared_schedule() if slot["phase"] == "timed"]
    _require(
        all(slot["sequence_index"] in by_sequence for slot in timed_slots),
        "statistics requires every frozen timed slot",
    )
    timed = [by_sequence[slot["sequence_index"]] for slot in timed_slots]
    _require(len(timed) == 32, "statistics timed sample count is not 32")
    arm_values: dict[str, list[float]] = {"AN": [], "L": []}
    block_values: list[float] = []
    block_orders: list[str] = []
    for block_index, order in enumerate(TIMED_BLOCK_ORDERS):
        block_samples = [
            sample
            for sample in timed
            if sample["request"].get("block_index") == block_index
        ]
        block_samples.sort(key=lambda sample: sample["request"].get("slot_index"))
        _require(
            len(block_samples) == 4
            and "".join(sample["request"]["role"] for sample in block_samples)
            == order,
            f"statistics block {block_index} schedule drifted",
        )
        logs: dict[str, list[float]] = {"AN": [], "L": []}
        for sample in block_samples:
            arm = sample["request"].get("arm")
            _require(arm in ("AN", "L"), "statistics arm is invalid")
            tpot = _positive_number(sample.get("timing", {}).get("tpot_ms"), "statistics tpot_ms")
            arm_values[arm].append(tpot)
            logs[arm].append(math.log(tpot))
        _require(len(logs["AN"]) == len(logs["L"]) == 2, "statistics block is not arm-balanced")
        block_values.append(_mean(logs["AN"], "AN block logs") - _mean(logs["L"], "L block logs"))
        block_orders.append(order)

    log_mean = _mean(block_values, "block log ratios")
    sample_variance = sum((value - log_mean) ** 2 for value in block_values) / 7
    standard_error = math.sqrt(sample_variance) / math.sqrt(8)
    critical = 2.364624251
    lower = math.exp(log_mean - critical * standard_error)
    upper = math.exp(log_mean + critical * standard_error)
    point = math.exp(log_mean)
    first_half_ratio = math.exp(_mean(block_values[:4], "first-half block ratios"))
    second_half_ratio = math.exp(_mean(block_values[4:], "second-half block ratios"))
    abba_mean = _mean(
        [value for value, order in zip(block_values, block_orders) if order == "ABBA"],
        "ABBA block ratios",
    )
    baab_mean = _mean(
        [value for value, order in zip(block_values, block_orders) if order == "BAAB"],
        "BAAB block ratios",
    )
    first_last_difference = abs(
        _mean(block_values[:4], "first four blocks")
        - _mean(block_values[4:], "last four blocks")
    )
    statistical_contract = contract["statistics_and_decisions"][EDGE_ID]
    limits = statistical_contract["stability_gates"]
    an_cv = _population_cv(arm_values["AN"], "AN TPOT")
    l_cv = _population_cv(arm_values["L"], "L TPOT")
    half_classes = [
        _half_ratio_classification(first_half_ratio),
        _half_ratio_classification(second_half_ratio),
    ]
    stability = {
        "A_tpot_population_cv": an_cv <= limits["A_tpot_population_cv_max"],
        "L_tpot_population_cv": l_cv <= limits["L_tpot_population_cv_max"],
        "absolute_ABBA_BAAB_log_ratio_mean_difference": abs(abba_mean - baab_mean)
        <= limits["absolute_ABBA_BAAB_log_ratio_mean_difference_max"],
        "absolute_first4_last4_block_log_ratio_mean_difference": first_last_difference
        <= limits["absolute_first4_last4_block_log_ratio_mean_difference_max"],
        "both_half_campaign_ratios_support_same_decision": half_classes[0]
        == half_classes[1],
    }
    all_stable = all(stability.values())
    if not all_stable:
        decision = "UNRANKABLE"
    elif upper < 0.95 and first_half_ratio < 0.95 and second_half_ratio < 0.95:
        decision = "NAMED_APXINF_DEPLOYMENT_AT_LEAST_5_PERCENT_FASTER"
    elif lower > 1.05 and first_half_ratio > 1.05 and second_half_ratio > 1.05:
        decision = "NAMED_LLAMA_CPP_DEPLOYMENT_AT_LEAST_5_PERCENT_FASTER"
    elif lower >= 0.95 and upper <= 1.05:
        decision = "NAMED_DEPLOYMENTS_PRACTICALLY_EQUIVALENT_WITHIN_5_PERCENT"
    else:
        decision = "INCONCLUSIVE"
    return {
        "primary_observation": "tpot_ms",
        "timed_sample_count": 32,
        "timed_samples_per_arm": {"AN": 16, "L": 16},
        "block_orders": block_orders,
        "block_log_ratios": block_values,
        "mean_block_log_ratio": log_mean,
        "block_log_ratio_sample_variance": sample_variance,
        "block_log_ratio_standard_error": standard_error,
        "student_t_df": 7,
        "student_t_critical_0_975": critical,
        "point_ratio_A_over_L": point,
        "lower_ci95_A_over_L": lower,
        "upper_ci95_A_over_L": upper,
        "first_half_ratio_A_over_L": first_half_ratio,
        "second_half_ratio_A_over_L": second_half_ratio,
        "half_ratio_classifications": half_classes,
        "A_tpot_population_cv": an_cv,
        "L_tpot_population_cv": l_cv,
        "ABBA_log_ratio_mean": abba_mean,
        "BAAB_log_ratio_mean": baab_mean,
        "absolute_ABBA_BAAB_log_ratio_mean_difference": abs(abba_mean - baab_mean),
        "absolute_first4_last4_block_log_ratio_mean_difference": first_last_difference,
        "stability_gates": stability,
        "all_stability_gates_passed": all_stable,
        "decision": decision,
    }


def _safe_failure_observation(value: Any) -> Any:
    if isinstance(value, bytes):
        return {
            "encoding": "base64",
            "size_bytes": len(value),
            "sha256": hashlib.sha256(value).hexdigest(),
            "data": base64.b64encode(value).decode("ascii"),
        }
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, list):
        return [_safe_failure_observation(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _safe_failure_observation(item) for key, item in value.items()}
    return {"python_type": type(value).__name__, "repr": repr(value)}


def _required_native_gates(passed: bool, stability: bool) -> dict[str, bool]:
    gate_ids = (
        "PREDECLARATION_PUBLIC_BEFORE_SAMPLING",
        "GIT_CUSTODY",
        "HOST_IDENTITY",
        "QUIET_HOST_CONTINUOUS",
        "POWER_THERMAL_MEMORY",
        "SAME_MODEL_REVISION_LINEAGE",
        "NATIVE_DEPLOYMENT_IDENTITIES_AND_HASHES",
        "NATIVE_DIFFERENCE_DISCLOSURES_COMPLETE",
        "RAW_PROMPT_IDS_EQUAL",
        "TEACHER_FORCED_128_EXACT",
        "FREE128_TRAJECTORY_EQUAL",
        "NEXT_GREEDY_TOKEN_READY_BOUNDARY_EQUAL",
        "FIXED_ABBA_BAAB_SCHEDULE_COMPLETE",
        "NO_RETRY_REPLACEMENT_OUTLIER_REMOVAL",
    )
    result = {gate: passed for gate in gate_ids}
    result["NATIVE_STABILITY"] = passed and stability
    return result


def validate_native_machine_receipt(
    receipt: dict[str, Any], contract: dict[str, Any]
) -> None:
    required_objects = contract["machine_receipt_contract"][
        "required_top_level_objects"
    ]
    _require(
        all(field in receipt and receipt[field] is not None for field in required_objects)
        and isinstance(receipt.get("samples"), list)
        and all(
            isinstance(receipt.get(field), dict)
            for field in required_objects
            if field != "samples"
        ),
        "formal machine receipt is missing a required top-level object",
    )
    _require(receipt.get("status") == "FORMAL_COMPLETE", "formal machine receipt is not complete")
    binding = receipt["contract_binding"]
    _require(
        binding.get("campaign_id") == contract["campaign_id"]
        and binding.get("schema_version") == 3
        and binding.get("edge_id") == EDGE_ID
        and binding.get("subcampaign_id")
        == contract["comparison_graph"]["edges"][EDGE_ID]["subcampaign_id"],
        "formal machine contract binding identity mismatch",
    )
    required_binding_fields = contract["machine_receipt_contract"][
        "required_dynamic_contract_binding"
    ]["required_fields"]
    _require(
        all(field in binding and binding[field] is not None for field in required_binding_fields),
        "formal machine dynamic contract binding is incomplete",
    )
    _require(
        binding["activation_commit"]
        == binding["head_commit"]
        == binding["ls_remote_live_oid"]
        == binding["local_tracking_oid"]
        and binding["contract_commit_is_ancestor_of_activation_commit"] is True
        and binding["activation_commit_equals_head_and_live_remote_oid"] is True
        and binding["local_tracking_ref_used_as_publication_proof"] is False
        and binding["worktree_clean"] is True,
        "formal machine Git publication booleans are invalid",
    )
    host = receipt["host_custody"]
    for field in (
        "model_identifier",
        "chip",
        "architecture",
        "logical_cpu_count",
        "memory_bytes",
        "os_product",
        "os_version",
        "os_build",
    ):
        _require(host.get(field) == contract["scope"]["host"][field], f"formal machine host {field} mismatch")
    validate_host_receipt(host.get("preflight"), "preflight", contract)
    validate_host_receipt(host.get("continuous"), "continuous", contract)
    validate_host_receipt(host.get("postflight"), "postflight", contract)
    artifacts = receipt["artifact_custody"]
    exact = contract["machine_receipt_contract"]["required_exact_bindings"]
    _require(
        artifacts.get("model", {}).get("sha256")
        == exact["/artifact_custody/model/sha256"]
        and artifacts.get("llama_cpp", {}).get("source_commit")
        == exact["/artifact_custody/llama_cpp/source_commit"]
        and artifacts.get("omniinfer", {}).get("source_commit")
        == exact["/artifact_custody/omniinfer/source_commit"]
        and artifacts.get("gateway_backend", {}).get("source_commit")
        == exact["/artifact_custody/gateway_backend/source_commit"],
        "formal machine artifact exact binding mismatch",
    )
    parity = receipt["parity_admission"]
    _require(parity.get("all_passed") is True, "formal machine teacher admission is not exact")
    _require(
        set(parity.get("admissions", {})) == {"AN", "L"},
        "formal machine teacher arms are incomplete",
    )
    schedule = receipt["schedule_receipt"]
    _require(
        schedule.get("attempted_count") == 38
        and schedule.get("accepted_count") == 38
        and schedule.get("failed_count") == 0
        and schedule.get("remaining_unattempted_count") == 0
        and schedule.get("stopped_at_first_failure") is False
        and schedule.get("retry_replacement_or_extension_performed") is False
        and [entry.get("slot") for entry in schedule.get("slots", [])]
        == declared_schedule()
        and all(entry.get("status") == "accepted" and entry.get("attempt_count") == 1 for entry in schedule["slots"]),
        "formal machine fixed schedule/retry receipt mismatch",
    )
    _require(len(receipt.get("samples", [])) == 38, "formal machine sample count is not 38")
    required_gates = set(
        contract["machine_receipt_contract"][
            "required_true_gate_ids_for_NATIVE_A_VS_L"
        ]
    )
    _require(
        set(receipt["gates"]) == required_gates
        and all(receipt["gates"].values()),
        "formal machine native gate set is not exactly all true",
    )
    statistics = receipt["statistics"]
    _require(
        statistics.get("timed_sample_count") == 32
        and statistics.get("all_stability_gates_passed") is True
        and receipt["decision"].get("label") == statistics.get("decision")
        and receipt["decision"].get("formal_summary_allowed") is True,
        "formal machine statistics or decision mismatch",
    )


def execute_formal_schedule(
    contract: dict[str, Any],
    plan: dict[str, Any],
    binding_evidence: dict[str, Any],
    raw_output_path: Path | str,
    *,
    sample_collector: Any,
    postflight_collector: Any,
    nonce_factory: Any | None = None,
    before_first_slot: Any | None = None,
) -> dict[str, Any]:
    """Consume the exact 38 slots once, durably recording every transition."""

    _require(callable(sample_collector), "sample collector is not callable")
    _require(callable(postflight_collector), "postflight collector is not callable")
    _require(
        isinstance(binding_evidence, dict)
        and binding_evidence.get("blocker_resolution", {}).get("all_resolved")
        is True,
        "campaign-start authored blockers were not all externally resolved",
    )
    schedule = declared_schedule()
    raw_path = Path(raw_output_path)
    slots = [
        {
            "slot": copy.deepcopy(slot),
            "status": "unattempted",
            "attempt_count": 0,
            "nonce": None,
            "receipt_sha256": None,
            "failure_index": None,
        }
        for slot in schedule
    ]
    record: dict[str, Any] = {
        "format": RAW_FORMAT,
        "schema_version": 3,
        "campaign_id": contract["campaign_id"],
        "subcampaign_id": contract["comparison_graph"]["edges"][EDGE_ID][
            "subcampaign_id"
        ],
        "edge_id": EDGE_ID,
        "status": "RUNNING",
        "raw_output_path": str(raw_path),
        "contract_binding": copy.deepcopy(binding_evidence["contract_binding"]),
        "git_custody": copy.deepcopy(binding_evidence["git_custody"]),
        "host_custody": {
            **{
                field: contract["scope"]["host"][field]
                for field in (
                    "model_identifier",
                    "chip",
                    "architecture",
                    "logical_cpu_count",
                    "memory_bytes",
                    "os_product",
                    "os_version",
                    "os_build",
                )
            },
            **copy.deepcopy(binding_evidence["host_custody"]),
        },
        "artifact_custody": copy.deepcopy(binding_evidence["artifact_custody"]),
        "parity_admission": copy.deepcopy(binding_evidence["parity_admission"]),
        "blocker_resolution": copy.deepcopy(binding_evidence["blocker_resolution"]),
        "schedule_receipt": {
            "process_state": "fresh-process-per-sample",
            "warmup_order": list(WARMUP_ROLES),
            "timed_block_orders": list(TIMED_BLOCK_ORDERS),
            "slots": slots,
            "attempted_count": 0,
            "accepted_count": 0,
            "failed_count": 0,
            "remaining_unattempted_count": 38,
            "stopped_at_first_failure": False,
            "retry_replacement_or_extension_performed": False,
        },
        "samples": [],
        "failures": [],
        "statistics": None,
        "gates": _required_native_gates(False, False),
        "decision": {"label": "UNRANKABLE", "formal_summary_allowed": False},
    }
    atomic_create_json(raw_path, record)
    generated_nonces: set[str] = set()
    stopped = False
    if before_first_slot is not None:
        try:
            _require(callable(before_first_slot), "before-first-slot hook is not callable")
            before_first_slot()
        except BaseException as error:
            record["failures"].append(
                {
                    "stage": "continuous-monitor-start",
                    "sequence_index": None,
                    "arm": None,
                    "exception_type": type(error).__name__,
                    "message": str(error),
                    "observation": {},
                    "remaining_slots_marked_unattempted": True,
                    "failed_observation_retained": True,
                }
            )
            record["status"] = "CONSUMED_FIRST_POST_MARKER_FAILURE"
            record["schedule_receipt"]["stopped_at_first_failure"] = True
            atomic_replace_json(raw_path, record)
            stopped = True
    factory = nonce_factory or (lambda _slot: secrets.token_hex(32))
    for slot, slot_record in zip(schedule, slots):
        if stopped:
            break
        try:
            nonce = factory(copy.deepcopy(slot))
            _require(
                isinstance(nonce, str)
                and _valid_sha256(nonce)
                and nonce not in generated_nonces,
                "sample request nonce is invalid or reused",
            )
            generated_nonces.add(nonce)
            slot_record["status"] = "attempting"
            slot_record["attempt_count"] = 1
            slot_record["nonce"] = nonce
            record["schedule_receipt"]["attempted_count"] += 1
            record["schedule_receipt"]["remaining_unattempted_count"] -= 1
            atomic_replace_json(raw_path, record)
            sample = sample_collector(copy.deepcopy(slot), nonce)
            validate_sample_receipt(
                sample,
                slot,
                nonce,
                contract,
                plan["artifacts"][slot["arm"]],
            )
            slot_record["status"] = "accepted"
            slot_record["receipt_sha256"] = _sha256_compact(sample)
            record["samples"].append(sample)
            record["schedule_receipt"]["accepted_count"] += 1
            atomic_replace_json(raw_path, record)
        except BaseException as error:
            slot_record["status"] = "failed"
            slot_record["failure_index"] = len(record["failures"])
            observation = (
                error.observation
                if isinstance(error, RuntimeInvocationError)
                else {}
            )
            record["failures"].append(
                {
                    "stage": "sample",
                    "sequence_index": slot["sequence_index"],
                    "arm": slot["arm"],
                    "exception_type": type(error).__name__,
                    "message": str(error),
                    "observation": _safe_failure_observation(observation),
                    "remaining_slots_marked_unattempted": True,
                    "failed_observation_retained": True,
                }
            )
            record["schedule_receipt"]["failed_count"] += 1
            record["schedule_receipt"]["stopped_at_first_failure"] = True
            record["status"] = "CONSUMED_FIRST_POST_MARKER_FAILURE"
            atomic_replace_json(raw_path, record)
            stopped = True

    try:
        postflight = postflight_collector()
        _require(isinstance(postflight, dict), "postflight collector returned no receipt")
        continuous = validate_host_receipt(
            postflight.get("continuous"), "continuous", contract
        )
        postflight_host = validate_host_receipt(
            postflight.get("postflight"), "postflight", contract
        )
        if binding_evidence.get("enforce_machine_contract") is True:
            expected_runtime_proofs = [
                {
                    "pid": sample["driver_process_observation"]["pid"],
                    "arm": sample["request"]["arm"],
                    "proof_sha256": sample["driver_process_observation"][
                        "runtime_swap_proof_sha256"
                    ],
                }
                for sample in record["samples"]
            ]
            observed_runtime_proofs = [
                {
                    "pid": proof["pid"],
                    "arm": proof["arm"],
                    "proof_sha256": _sha256_compact(proof),
                }
                for proof in continuous["accepted_runtime_swap_proofs"]
            ]
            _require(
                observed_runtime_proofs == expected_runtime_proofs,
                "continuous host custody does not cover every accepted runtime PID",
            )
        _require(
            postflight.get("artifact_custody_end") == plan["artifacts"],
            "artifact custody changed between campaign start and postflight",
        )
        if "artifact_file_observations" in binding_evidence:
            _require(
                postflight.get("artifact_file_observations_end")
                == binding_evidence["artifact_file_observations"],
                "artifact file identity/hash observations changed at postflight",
            )
        record["host_custody"]["continuous"] = continuous
        record["host_custody"]["postflight"] = postflight_host
        record["artifact_custody_end"] = postflight["artifact_custody_end"]
        if "artifact_file_observations_end" in postflight:
            record["artifact_file_observations_end"] = postflight[
                "artifact_file_observations_end"
            ]
    except BaseException as error:
        record["failures"].append(
            {
                "stage": "postflight",
                "sequence_index": None,
                "arm": None,
                "exception_type": type(error).__name__,
                "message": str(error),
                "observation": {},
                "remaining_slots_marked_unattempted": True,
                "failed_observation_retained": True,
            }
        )
        record["status"] = "CONSUMED_FIRST_POST_MARKER_FAILURE"
        record["schedule_receipt"]["stopped_at_first_failure"] = True

    if not record["failures"]:
        _require(len(record["samples"]) == 38, "formal schedule did not admit exactly 38 samples")
        statistics = compute_native_statistics(record["samples"], contract)
        record["statistics"] = statistics
        stable = statistics["all_stability_gates_passed"]
        record["gates"] = _required_native_gates(True, stable)
        record["decision"] = {
            "label": statistics["decision"],
            "formal_summary_allowed": stable,
            "result_subject": "unmatched-numerical-regime named-native-deployment single-workload comparison",
        }
        record["status"] = "FORMAL_COMPLETE" if stable else "FORMAL_UNRANKABLE"
    else:
        record["statistics"] = None
        record["gates"] = _required_native_gates(False, False)
        record["decision"] = {
            "label": "UNRANKABLE",
            "formal_summary_allowed": False,
        }
    record["schedule_receipt"]["accepted_count"] = len(record["samples"])
    record["schedule_receipt"]["failed_count"] = sum(
        entry["stage"] == "sample" for entry in record["failures"]
    )
    record["schedule_receipt"]["remaining_unattempted_count"] = sum(
        entry["status"] == "unattempted" for entry in slots
    )
    if (
        binding_evidence.get("enforce_machine_contract") is True
        and record["status"] == "FORMAL_COMPLETE"
    ):
        try:
            validate_native_machine_receipt(record, contract)
        except BaseException as error:
            record["failures"].append(
                {
                    "stage": "final-machine-receipt-validation",
                    "sequence_index": None,
                    "arm": None,
                    "exception_type": type(error).__name__,
                    "message": str(error),
                    "observation": {},
                    "remaining_slots_marked_unattempted": True,
                    "failed_observation_retained": True,
                }
            )
            record["status"] = "CONSUMED_FIRST_POST_MARKER_FAILURE"
            record["statistics"] = None
            record["gates"] = _required_native_gates(False, False)
            record["decision"] = {
                "label": "UNRANKABLE",
                "formal_summary_allowed": False,
            }
    atomic_replace_json(raw_path, record)
    return record


def _load_strict_json_document(path_value: Path | str) -> tuple[dict[str, Any], dict[str, Any]]:
    path, raw, observation = _file_snapshot(path_value)
    _require(0 < len(raw) <= 8 * 1024 * 1024, f"JSON document size is invalid: {path}")
    try:
        value = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CampaignError(f"strict JSON document rejected at {path}: {error}") from error
    _require(isinstance(value, dict), f"JSON document is not an object: {path}")
    return value, observation


def _read_bootstrap_marker(
    marker_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Read the already-created marker without following a replacement link."""

    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(str(marker_path), flags)
    except OSError as error:
        raise CampaignError(
            f"cannot open the campaign-start marker directly: {marker_path}: {error}"
        ) from error
    try:
        before = os.fstat(descriptor)
        _require(
            stat.S_ISREG(before.st_mode) and before.st_nlink == 1,
            "campaign-start marker is not a direct single-link regular file",
        )
        chunks: list[bytes] = []
        size = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            _require(size <= 8 * 1024 * 1024, "campaign-start marker is too large")
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity = (
        before.st_dev,
        before.st_ino,
        before.st_mode,
        before.st_nlink,
        before.st_size,
        before.st_ctime_ns,
    )
    _require(
        identity
        == (
            after.st_dev,
            after.st_ino,
            after.st_mode,
            after.st_nlink,
            after.st_size,
            after.st_ctime_ns,
        ),
        "campaign-start marker changed while loading the run bootstrap",
    )
    raw = b"".join(chunks)
    marker = parse_single_json_line(raw)
    return marker, {
        "absolute_path": str(marker_path),
        "size_bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "device": before.st_dev,
        "inode": before.st_ino,
        "mode": before.st_mode,
        "hard_link_count": before.st_nlink,
        "ctime_ns": before.st_ctime_ns,
        "open_flags": ["O_RDONLY", "O_CLOEXEC", "O_NOFOLLOW"],
        "identity_before_after_equal": True,
    }


def _bootstrap_post_marker_run(plan_path_value: Path | str) -> dict[str, Any]:
    """Recover only marker-bound paths needed to persist a context-load failure."""

    plan_path = Path(plan_path_value).resolve(strict=True)
    preliminary, plan_file = _load_strict_json_document(plan_path)
    _require(
        set(preliminary) == EXECUTION_PLAN_FIELDS,
        "execution plan fields differ from v3 during run bootstrap",
    )
    _require(
        preliminary.get("format") == PLAN_FORMAT
        and preliminary.get("schema_version") == 3
        and preliminary.get("edge_id") == EDGE_ID,
        "execution plan identity differs during run bootstrap",
    )
    root_text = preliminary.get("repository_root")
    _require(
        isinstance(root_text, str)
        and root_text.startswith("/")
        and os.path.normpath(root_text) == root_text,
        "execution plan repository root is invalid during run bootstrap",
    )
    repository_root = Path(root_text).resolve(strict=True)
    _require(
        repository_root.is_dir() and str(repository_root) == root_text,
        "execution plan repository root is not a canonical directory",
    )
    _require(
        preliminary.get("contract_repository_path")
        == FROZEN_CONTRACT_REPOSITORY_PATH
        and preliminary.get("validator_repository_path")
        == FROZEN_VALIDATOR_REPOSITORY_PATH
        and preliminary.get("driver_repository_path")
        == FROZEN_DRIVER_REPOSITORY_PATH,
        "execution plan frozen source paths differ during run bootstrap",
    )
    plan_repository_path = _validate_repository_path(
        preliminary.get("plan_repository_path"), "execution plan"
    )
    _require(
        (repository_root / plan_repository_path).resolve(strict=True)
        == plan_path,
        "execution plan CLI path differs from its run-bootstrap binding",
    )
    _require(
        preliminary.get("marker_repository_path")
        == FROZEN_NATIVE_MARKER_REPOSITORY_PATH,
        "execution plan marker path differs during run bootstrap",
    )
    raw_text = preliminary.get("raw_output_path")
    _require(
        isinstance(raw_text, str)
        and raw_text.startswith("/")
        and os.path.normpath(raw_text) == raw_text,
        "execution plan raw output path is invalid during run bootstrap",
    )
    raw_path = Path(raw_text)
    _require(
        raw_path.parent.resolve(strict=True).is_dir(),
        "formal raw-output parent is absent during run bootstrap",
    )
    _marker_path_absent(raw_path, "formal raw output")

    marker_path = repository_root / FROZEN_NATIVE_MARKER_REPOSITORY_PATH
    marker, marker_file = _read_bootstrap_marker(marker_path)
    _require(
        marker.get("format") == MARKER_FORMAT
        and marker.get("schema_version") == 3
        and marker.get("campaign_id") == FROZEN_CAMPAIGN_ID
        and marker.get("subcampaign_id") == FROZEN_NATIVE_SUBCAMPAIGN_ID
        and marker.get("edge_id") == EDGE_ID,
        "campaign-start marker identity differs during run bootstrap",
    )
    _require(
        marker.get("plan_repository_path") == plan_repository_path
        and marker.get("plan_blob_size_bytes") == plan_file["size_bytes"]
        and marker.get("plan_blob_sha256") == plan_file["sha256"]
        and marker.get("marker_repository_path")
        == FROZEN_NATIVE_MARKER_REPOSITORY_PATH,
        "campaign-start marker does not bind the loaded execution plan",
    )
    _require(
        marker.get("sampling_state_at_marker_creation")
        == {"generation_requests": 0, "warmup_samples": 0, "timed_samples": 0}
        and marker.get("pre_marker_admission", {}).get("all_passed") is True,
        "campaign-start marker does not prove a pre-generation admission",
    )
    return {
        "repository_root": repository_root,
        "plan": preliminary,
        "plan_file": plan_file,
        "marker": marker,
        "marker_file": marker_file,
        "marker_path": marker_path,
        "raw_path": raw_path,
    }


def _read_tracked_single_line_receipt(
    repository_root: Path,
    repository_path: str,
    tracked: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    absolute = repository_root / repository_path
    expected = {
        "absolute_path": str(absolute),
        "size_bytes": tracked["blob_size_bytes"],
        "sha256": tracked["blob_sha256"],
    }
    observation = file_custody(expected)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(str(absolute), flags)
    try:
        before = os.fstat(descriptor)
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            _require(total <= 8 * 1024 * 1024, f"tracked receipt is too large: {repository_path}")
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    _require(
        (before.st_dev, before.st_ino, before.st_size, before.st_ctime_ns)
        == (after.st_dev, after.st_ino, after.st_size, after.st_ctime_ns),
        f"tracked receipt changed while reading: {repository_path}",
    )
    raw = b"".join(chunks)
    _require(
        len(raw) == expected["size_bytes"]
        and hashlib.sha256(raw).hexdigest() == expected["sha256"],
        f"tracked receipt bytes changed after custody proof: {repository_path}",
    )
    receipt = parse_single_json_line(raw)
    return receipt, observation


def load_execution_context(plan_path_value: Path | str) -> dict[str, Any]:
    """Load the plan, frozen contract and frozen validator without network access."""

    plan_path = Path(plan_path_value).resolve(strict=True)
    preliminary, plan_file = _load_strict_json_document(plan_path)
    root_text = preliminary.get("repository_root")
    _require(isinstance(root_text, str) and root_text.startswith("/"), "plan repository root is absent")
    repository_root = Path(root_text).resolve(strict=True)
    _require(repository_root.is_dir(), "plan repository root is not a directory")
    contract_repository_path = preliminary.get("contract_repository_path")
    validator_repository_path = preliminary.get("validator_repository_path")
    _require(isinstance(contract_repository_path, str), "plan contract path is absent")
    _require(isinstance(validator_repository_path, str), "plan validator path is absent")
    frozen = load_frozen_contract(
        repository_root / contract_repository_path,
        repository_root / validator_repository_path,
    )
    plan = validate_execution_plan(preliminary, frozen["contract"])
    expected_plan_path = repository_root / plan["plan_repository_path"]
    _require(
        expected_plan_path.resolve(strict=True) == plan_path,
        "execution plan CLI path differs from its repository binding",
    )
    return {
        "repository_root": repository_root,
        "plan": plan,
        "plan_file": plan_file,
        **frozen,
    }


def _tracked_campaign_paths(
    plan: dict[str, Any], *, include_marker: bool
) -> dict[str, str]:
    result = {
        "contract": plan["contract_repository_path"],
        "validator": plan["validator_repository_path"],
        "driver": plan["driver_repository_path"],
        "plan": plan["plan_repository_path"],
    }
    for arm in ("AN", "L"):
        result[f"teacher_{arm}_reference"] = plan["teacher_receipts"][arm][
            "reference_repository_path"
        ]
        result[f"teacher_{arm}_runtime"] = plan["teacher_receipts"][arm][
            "runtime_repository_path"
        ]
    if include_marker:
        result["activation_marker"] = plan["marker_repository_path"]
    return result


def verify_plan_artifacts(plan: dict[str, Any]) -> dict[str, Any]:
    observations: dict[str, Any] = {}
    for arm in ("AN", "L"):
        expected = plan["artifacts"][arm]
        observations[arm] = {
            "configuration_id": expected["configuration_id"],
            "runner": file_custody(expected["runner"]),
            "model": file_custody(expected["model"]),
            "runtime_source_commit": expected["runtime_source_commit"],
            "loaded_non_system_library_closure_sha256": expected[
                "loaded_non_system_library_closure_sha256"
            ],
            "packed_weight_and_resident_buffer_manifest_sha256": expected[
                "packed_weight_and_resident_buffer_manifest_sha256"
            ],
            "deployment": copy.deepcopy(expected["deployment"]),
        }
    return observations


def collect_teacher_admissions(
    repository_root: Path,
    plan: dict[str, Any],
    contract: dict[str, Any],
    git_custody: dict[str, Any],
    artifact_observations: dict[str, Any],
) -> dict[str, Any]:
    tracked = git_custody["tracked_files"]
    admissions: dict[str, Any] = {}
    files: dict[str, Any] = {}
    for arm in ("AN", "L"):
        paths = plan["teacher_receipts"][arm]
        reference_label = f"teacher_{arm}_reference"
        runtime_label = f"teacher_{arm}_runtime"
        reference_raw, reference_file = _read_tracked_single_line_receipt(
            repository_root,
            paths["reference_repository_path"],
            tracked[reference_label],
        )
        runtime_raw, runtime_file = _read_tracked_single_line_receipt(
            repository_root,
            paths["runtime_repository_path"],
            tracked[runtime_label],
        )
        admission = build_teacher_admission_receipt(
            arm,
            reference_raw,
            runtime_raw,
            reference_file,
            runtime_file,
            contract,
            plan["artifacts"][arm],
            reference_expected_custody=plan["artifacts"]["AN"],
            expected_artifact_observation=artifact_observations[arm],
        )
        validate_teacher_receipt(
            admission,
            arm,
            contract,
            reference_file,
            runtime_file,
            plan["artifacts"][arm],
        )
        admissions[arm] = admission
        files[arm] = {
            "reference_repository_path": paths["reference_repository_path"],
            "reference": reference_file,
            "runtime_repository_path": paths["runtime_repository_path"],
            "runtime": runtime_file,
        }
    return {"admissions": admissions, "files": files, "all_passed": True}


def _teacher_an_source_custodies(
    teacher: dict[str, Any]
) -> list[dict[str, Any]]:
    admissions = _require_mapping(
        teacher.get("admissions"), "teacher admissions"
    )
    result: list[dict[str, Any]] = []
    for arm in ("AN", "L"):
        admission = _require_mapping(
            admissions.get(arm), f"{arm} teacher admission"
        )
        sources = _require_mapping(
            admission.get("source_receipts"),
            f"{arm} teacher source receipts",
        )
        reference = _require_mapping(
            sources.get("reference"), f"{arm} CPU teacher raw receipt"
        )
        result.append(
            _require_mapping(
                reference.get("custody"),
                f"{arm} CPU teacher raw custody",
            )
        )
        if arm == "AN":
            runtime = _require_mapping(
                sources.get("runtime"), "AN teacher raw receipt"
            )
            result.append(
                _require_mapping(
                    runtime.get("custody"), "AN teacher raw custody"
                )
            )
    return result


NATIVE_AUTHORED_BLOCKERS = (
    "LLAMA_CPP_TEACHER_FORCED_128_RECEIPT_NOT_CAPTURED",
    "V3_NATIVE_DRIVER_AND_BINARY_HASHES_NOT_CAPTURED",
    "NATIVE_SUBCAMPAIGN_NOT_PUBLICLY_ACTIVATED",
    "QUIET_HOST_GATE_NOT_YET_PASSED",
)


def native_blocker_resolution(
    contract: dict[str, Any],
    *,
    teacher_evidence: Any,
    binary_evidence: Any,
    quiet_host_evidence: Any,
    public_activation_evidence: Any | None,
) -> dict[str, Any]:
    authored = contract["native_deployment_contract"]["current_readiness"]
    _require(
        tuple(authored["blocker_codes"]) == NATIVE_AUTHORED_BLOCKERS
        and authored["formal_campaign_may_start_now"] is False,
        "native authored-state blocker set drifted",
    )
    resolutions = {
        NATIVE_AUTHORED_BLOCKERS[0]: {
            "resolved": teacher_evidence is not None,
            "evidence": teacher_evidence,
        },
        NATIVE_AUTHORED_BLOCKERS[1]: {
            "resolved": binary_evidence is not None,
            "evidence": binary_evidence,
        },
        NATIVE_AUTHORED_BLOCKERS[2]: {
            "resolved": public_activation_evidence is not None,
            "evidence": public_activation_evidence,
        },
        NATIVE_AUTHORED_BLOCKERS[3]: {
            "resolved": quiet_host_evidence is not None,
            "evidence": quiet_host_evidence,
        },
    }
    pre_marker = all(
        resolutions[code]["resolved"]
        for code in NATIVE_AUTHORED_BLOCKERS
        if code != "NATIVE_SUBCAMPAIGN_NOT_PUBLICLY_ACTIVATED"
    )
    return {
        "authored_formal_campaign_may_start_now": False,
        "authored_blocker_codes": list(NATIVE_AUTHORED_BLOCKERS),
        "resolution_map": resolutions,
        "all_pre_marker_blockers_except_public_activation_resolved": pre_marker,
        "all_resolved": all(item["resolved"] for item in resolutions.values()),
        "authored_state_was_not_mutated": True,
    }


def _checked_host_command(argv: list[str], timeout_seconds: float = 10.0) -> bytes:
    result = _system_command_runner(
        argv,
        Path("/"),
        timeout_seconds,
        {"LC_ALL": "C", "TZ": "UTC"},
    )
    _require(result["returncode"] == 0, f"host command failed: {argv}")
    _require(result["stderr"] == b"", f"host command wrote stderr: {argv}")
    return result["stdout"]


def _one_host_line(argv: list[str]) -> str:
    raw = _checked_host_command(argv)
    _require(raw.endswith(b"\n") and raw.count(b"\n") == 1, f"host command output shape drifted: {argv}")
    try:
        return raw[:-1].decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise CampaignError(f"host command output is not UTF-8: {argv}") from error


def collect_mac_host_identity() -> dict[str, Any]:
    _require(platform.system() == "Darwin", "formal native v3 host must be macOS")
    sw_vers = _checked_host_command(["/usr/bin/sw_vers"])
    try:
        lines = sw_vers.decode("utf-8", errors="strict").splitlines()
        values = {
            key.strip(): value.strip()
            for key, value in (line.split(":", 1) for line in lines)
        }
    except (UnicodeDecodeError, ValueError) as error:
        raise CampaignError("sw_vers output is malformed") from error
    _require(set(values) == {"ProductName", "ProductVersion", "BuildVersion"}, "sw_vers fields drifted")
    logical = _one_host_line(["/usr/sbin/sysctl", "-n", "hw.logicalcpu"])
    memory = _one_host_line(["/usr/sbin/sysctl", "-n", "hw.memsize"])
    _require(logical.isdigit() and memory.isdigit(), "host CPU or memory sysctl is invalid")
    return {
        "model_identifier": _one_host_line(["/usr/sbin/sysctl", "-n", "hw.model"]),
        "chip": _one_host_line(
            ["/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string"]
        ),
        "architecture": _one_host_line(["/usr/bin/uname", "-m"]),
        "logical_cpu_count": int(logical),
        "memory_bytes": int(memory),
        "os_product": values["ProductName"],
        "os_version": values["ProductVersion"],
        "os_build": values["BuildVersion"],
    }


def _collect_power_thermal_state() -> dict[str, Any]:
    battery = _checked_host_command(["/usr/bin/pmset", "-g", "batt"])
    try:
        battery_text = battery.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise CampaignError("pmset battery output is not UTF-8") from error
    match = re.search(r"^Now drawing from '([^']+)'$", battery_text, re.MULTILINE)
    _require(match is not None, "pmset did not report an exact power source")
    thermal = _checked_host_command(["/usr/bin/pmset", "-g", "therm"])
    try:
        thermal_text = thermal.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise CampaignError("pmset thermal output is not UTF-8") from error
    no_thermal = "Note: No thermal warning level has been recorded" in thermal_text
    no_performance = "Note: No performance warning level has been recorded" in thermal_text
    _require(no_thermal or "Thermal Warning Level: 0" in thermal_text, "thermal warning state is not provably clear")
    _require(no_performance or "Performance Warning Level: 0" in thermal_text, "performance warning state is not provably clear")
    return {
        "power_source": match.group(1),
        "thermal_warning": False,
        "performance_warning": False,
        "pmset_batt_sha256": hashlib.sha256(battery).hexdigest(),
        "pmset_therm_sha256": hashlib.sha256(thermal).hexdigest(),
    }


def _system_swap_used_bytes() -> int:
    raw = _checked_host_command(["/usr/sbin/sysctl", "-b", "vm.swapusage"])
    _require(len(raw) == 32, "binary vm.swapusage structure size drifted")
    total, available, used, page_size, encrypted = struct.unpack("<QQQII", raw)
    _require(
        total == available + used
        and page_size > 0
        and encrypted in (0, 1),
        "binary vm.swapusage structure is inconsistent",
    )
    return used


def _pages_throttled() -> int:
    raw = _checked_host_command(["/usr/bin/vm_stat"])
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise CampaignError("vm_stat output is not UTF-8") from error
    match = re.search(r"^Pages throttled:\s+([0-9]+)\.$", text, re.MULTILINE)
    _require(match is not None, "vm_stat pages-throttled field is absent")
    return int(match.group(1))


class _ProcTaskInfo(ctypes.Structure):
    _fields_ = [
        ("virtual_size", ctypes.c_uint64),
        ("resident_size", ctypes.c_uint64),
        ("total_user", ctypes.c_uint64),
        ("total_system", ctypes.c_uint64),
        ("threads_user", ctypes.c_uint64),
        ("threads_system", ctypes.c_uint64),
        ("policy", ctypes.c_int32),
        ("faults", ctypes.c_int32),
        ("pageins", ctypes.c_int32),
        ("cow_faults", ctypes.c_int32),
        ("messages_sent", ctypes.c_uint32),
        ("messages_received", ctypes.c_uint32),
        ("syscalls_mach", ctypes.c_uint32),
        ("syscalls_unix", ctypes.c_uint32),
        ("context_switches", ctypes.c_uint32),
        ("thread_count", ctypes.c_uint32),
        ("running_threads", ctypes.c_uint32),
        ("priority", ctypes.c_int32),
    ]


_LIBPROC: Any | None = None


def _process_cpu_time_ns(pid: int) -> int | None:
    global _LIBPROC
    if _LIBPROC is None:
        _LIBPROC = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
        _LIBPROC.proc_pidinfo.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_uint64,
            ctypes.c_void_p,
            ctypes.c_int,
        ]
        _LIBPROC.proc_pidinfo.restype = ctypes.c_int
    info = _ProcTaskInfo()
    size = ctypes.sizeof(info)
    result = _LIBPROC.proc_pidinfo(pid, 4, 0, ctypes.byref(info), size)
    if result != size:
        return None
    return int(info.total_user + info.total_system)


def _process_inventory() -> dict[int, dict[str, Any]]:
    raw = _checked_host_command(
        ["/bin/ps", "-axo", "pid=,ppid=,pgid=,lstart=,command="]
    )
    try:
        lines = raw.decode("utf-8", errors="strict").splitlines()
    except UnicodeDecodeError as error:
        raise CampaignError("ps process inventory is not UTF-8") from error
    result: dict[int, dict[str, Any]] = {}
    for line in lines:
        parts = line.strip().split(None, 8)
        _require(len(parts) == 9, f"ps process inventory row drifted: {line!r}")
        pid_text, ppid_text, pgid_text = parts[:3]
        _require(pid_text.isdigit() and ppid_text.isdigit() and pgid_text.isdigit(), "ps PID fields are invalid")
        pid = int(pid_text)
        result[pid] = {
            "pid": pid,
            "ppid": int(ppid_text),
            "process_group_id": int(pgid_text),
            "process_start_time": " ".join(parts[3:8]),
            "command": parts[8],
            "cpu_time_ns": _process_cpu_time_ns(pid),
        }
    return result


def _hash_regular_file_unpinned(path_value: Path | str) -> dict[str, Any]:
    path = Path(path_value).resolve(strict=True)
    stat_value = path.stat()
    _require(stat.S_ISREG(stat_value.st_mode), f"executable is not regular: {path}")
    raw = path.read_bytes()
    return {
        "absolute_path": str(path),
        "size_bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def _process_swapped_bytes(pid: int) -> int:
    raw = _checked_host_command(
        ["/usr/bin/footprint", "--swapped", "-f", "bytes", "-p", str(pid)],
        timeout_seconds=15.0,
    )
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise CampaignError("footprint output is not UTF-8") from error
    match = re.search(
        r"^\s*[0-9]+ B\s+([0-9]+) B\s+[0-9]+ B\s+[0-9]+ B\s+[0-9]+\s+TOTAL$",
        text,
        re.MULTILINE,
    )
    _require(match is not None, f"footprint swapped TOTAL is absent for PID {pid}")
    return int(match.group(1))


class MacQuietHostProbe:
    """Stateful 250-ms CPU-window and exact-process-identity probe."""

    def __init__(self, contract: dict[str, Any]):
        self.contract = contract
        self.host = collect_mac_host_identity()
        self._orchestrator_pid = os.getpid()
        self._executable = _hash_regular_file_unpinned(sys.executable)
        self._argv_sha256 = hashlib.sha256(
            compact_json_bytes(sys.argv)
        ).hexdigest()
        self._previous_inventory: dict[int, dict[str, Any]] | None = None
        self._previous_allowed: set[tuple[int, str]] = set()
        self._previous_monotonic_ns: int | None = None

    def _descendants(self, inventory: dict[int, dict[str, Any]], root_pid: int | None) -> set[int]:
        if root_pid is None or root_pid not in inventory:
            return set()
        descendants = {root_pid}
        changed = True
        while changed:
            changed = False
            for pid, process in inventory.items():
                if process["ppid"] in descendants and pid not in descendants:
                    descendants.add(pid)
                    changed = True
        return descendants

    def _allowlist(
        self,
        inventory: dict[int, dict[str, Any]],
        active_runtime: dict[str, Any] | None,
    ) -> tuple[set[tuple[int, str]], list[dict[str, Any]], set[int]]:
        orchestrator = inventory.get(self._orchestrator_pid)
        _require(orchestrator is not None, "campaign orchestrator disappeared")
        active_root = active_runtime.get("pid") if active_runtime else None
        active_pids = self._descendants(inventory, active_root)
        helper_pids = {
            pid
            for pid, process in inventory.items()
            if process["ppid"] == self._orchestrator_pid
            and process["command"].startswith(
                (
                    "/bin/ps -axo ",
                    "/usr/bin/footprint --swapped -f bytes -p ",
                    "/usr/bin/pmset -g ",
                    "/usr/sbin/sysctl ",
                    "/usr/bin/vm_stat",
                    "/usr/bin/sw_vers",
                    "/usr/bin/uname -m",
                )
            )
        }
        allowed_pids = {self._orchestrator_pid, *active_pids, *helper_pids}
        identities = {
            (pid, inventory[pid]["process_start_time"])
            for pid in allowed_pids
            if pid in inventory
        }
        resolved = [
            {
                "role": "campaign_orchestrator",
                "pid": self._orchestrator_pid,
                "process_start_time": orchestrator["process_start_time"],
                "executable_path": self._executable["absolute_path"],
                "executable_sha256": self._executable["sha256"],
                "argv_sha256": self._argv_sha256,
                "process_group_id": orchestrator["process_group_id"],
            },
            {
                "role": "custody_monitor",
                "pid": self._orchestrator_pid,
                "process_start_time": orchestrator["process_start_time"],
                "executable_path": self._executable["absolute_path"],
                "executable_sha256": self._executable["sha256"],
                "argv_sha256": self._argv_sha256,
                "thread_native_id": threading.get_native_id(),
            },
        ]
        if active_runtime is not None and active_root in inventory:
            root = inventory[active_root]
            resolved.append(
                {
                    "role": "active_measured_runtime_tree",
                    "root_pid": active_root,
                    "root_process_start_time": root["process_start_time"],
                    "edge_id": EDGE_ID,
                    "descendant_pid_start_time_pairs": [
                        [pid, inventory[pid]["process_start_time"]]
                        for pid in sorted(active_pids)
                    ],
                    "executable_and_library_hashes": copy.deepcopy(
                        active_runtime["executable_and_library_hashes"]
                    ),
                }
            )
        return identities, resolved, active_pids

    def prime(self, active_runtime: dict[str, Any] | None = None) -> None:
        inventory = _process_inventory()
        allowed, _, _ = self._allowlist(inventory, active_runtime)
        self._previous_inventory = inventory
        self._previous_allowed = allowed
        self._previous_monotonic_ns = time.monotonic_ns()

    def seconds_until_next_window(self, interval_ms: int) -> float:
        _require(
            self._previous_monotonic_ns is not None,
            "host probe was not primed",
        )
        _require(_is_int(interval_ms) and interval_ms > 0, "host interval is invalid")
        target = self._previous_monotonic_ns + interval_ms * 1_000_000
        return max(0.0, (target - time.monotonic_ns()) / 1_000_000_000)

    def snapshot(
        self, index: int, active_runtime: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        _require(self._previous_inventory is not None and self._previous_monotonic_ns is not None, "host probe was not primed")
        inventory = _process_inventory()
        now = time.monotonic_ns()
        allowed, resolved, active_pids = self._allowlist(inventory, active_runtime)
        window_start_ns = self._previous_monotonic_ns
        elapsed_ns = now - window_start_ns
        _require(elapsed_ns > 0, "host CPU measurement window did not advance")
        vanished_unallowlisted = [
            process
            for pid, process in self._previous_inventory.items()
            if pid not in inventory
            and (pid, process["process_start_time"]) not in self._previous_allowed
        ]
        processes: list[dict[str, Any]] = []
        proof_complete = not vanished_unallowlisted
        maximum = self.contract["host_quiet_gate"]["process_policy"][
            "maximum_single_non_allowlisted_process_cpu_percent"
        ]
        for pid, process in inventory.items():
            identity = (pid, process["process_start_time"])
            if identity in allowed:
                continue
            previous = self._previous_inventory.get(pid)
            if (
                previous is None
                or previous["process_start_time"] != process["process_start_time"]
                or previous["cpu_time_ns"] is None
                or process["cpu_time_ns"] is None
            ):
                cpu_percent = maximum + 1.0
                proof_complete = False
            else:
                delta = process["cpu_time_ns"] - previous["cpu_time_ns"]
                if delta < 0:
                    cpu_percent = maximum + 1.0
                    proof_complete = False
                else:
                    cpu_percent = delta * 100.0 / elapsed_ns
            processes.append(
                {
                    "pid": pid,
                    "process_start_time": process["process_start_time"],
                    "ppid": process["ppid"],
                    "process_group_id": process["process_group_id"],
                    "command_sha256": hashlib.sha256(
                        process["command"].encode("utf-8")
                    ).hexdigest(),
                    "cpu_percent": cpu_percent,
                }
            )
        cpu_values = [process["cpu_percent"] for process in processes]
        observed_max = max(cpu_values, default=0.0)
        observed_total = sum(cpu_values)
        logical = self.contract["scope"]["host"]["logical_cpu_count"]
        load = os.getloadavg()[0] / logical
        campaign_pids = {self._orchestrator_pid, *active_pids}
        campaign_swap = 0
        swapped_observed: list[dict[str, Any]] = []
        vanished_during_swap_probe: list[dict[str, Any]] = []
        for pid in sorted(campaign_pids):
            process_identity = inventory[pid]
            try:
                swapped_bytes = _process_swapped_bytes(pid)
            except CampaignError:
                latest = _process_inventory()
                latest_process = latest.get(pid)
                if (
                    latest_process is not None
                    and all(
                        latest_process[field] == process_identity[field]
                        for field in (
                            "ppid",
                            "process_group_id",
                            "process_start_time",
                            "command",
                        )
                    )
                ):
                    raise
                vanished_during_swap_probe.append(
                    {
                        "pid": pid,
                        "process_start_time": process_identity[
                            "process_start_time"
                        ],
                    }
                )
                continue
            campaign_swap += swapped_bytes
            swapped_observed.append(
                {
                    "pid": pid,
                    "process_start_time": process_identity[
                        "process_start_time"
                    ],
                    "ppid": process_identity["ppid"],
                    "process_group_id": process_identity[
                        "process_group_id"
                    ],
                    "command_sha256": hashlib.sha256(
                        process_identity["command"].encode("utf-8")
                    ).hexdigest(),
                    "swapped_bytes": swapped_bytes,
                }
            )
        active_root = active_runtime.get("pid") if active_runtime else None
        active_root_present = (
            active_root in inventory if active_root is not None else None
        )
        observed_swap_pids = {entry["pid"] for entry in swapped_observed}
        active_runtime_swap_proof_complete = (
            active_root_present is True
            and active_pids.issubset(observed_swap_pids)
            and not any(
                entry["pid"] in active_pids
                for entry in vanished_during_swap_probe
            )
        )
        gate = self.contract["host_quiet_gate"]
        window_ms = elapsed_ns / 1_000_000
        window_valid = (
            250.0
            <= window_ms
            <= 250.0 + HOST_CPU_WINDOW_MAX_OVERRUN_MS
        )
        passed = (
            window_valid
            and proof_complete
            and observed_max
            <= gate["process_policy"][
                "maximum_single_non_allowlisted_process_cpu_percent"
            ]
            and observed_total
            <= gate["process_policy"][
                "maximum_aggregate_non_allowlisted_process_cpu_percent"
            ]
            and load
            <= gate["system_policy"]["maximum_load_average_per_logical_cpu"]
            and campaign_swap == 0
        )
        snapshot = {
            "index": index,
            "cpu_window_start_monotonic_ns": window_start_ns,
            "monotonic_ns": now,
            "cpu_percent_window_ms": window_ms,
            "cpu_measurement_source": "libproc-PROC_PIDTASKINFO-delta",
            "resolved_allowlist": resolved,
            "nonallowlisted_processes": processes,
            "vanished_nonallowlisted_processes": vanished_unallowlisted,
            "cpu_window_proof_complete": proof_complete,
            "maximum_single_nonallowlisted_process_cpu_percent": observed_max,
            "aggregate_nonallowlisted_process_cpu_percent": observed_total,
            "load_average_per_logical_cpu": load,
            "campaign_process_swap_bytes": campaign_swap,
            "campaign_process_swap_observations": swapped_observed,
            "campaign_swap_probe_vanished_processes": vanished_during_swap_probe,
            "active_runtime_root_present": active_root_present,
            "active_runtime_swap_proof_complete": active_runtime_swap_proof_complete,
            "passed": passed,
        }
        self._previous_inventory = inventory
        self._previous_allowed = allowed
        self._previous_monotonic_ns = now
        return snapshot


def _attach_snapshot_system_state(
    snapshot: dict[str, Any],
    start_state: dict[str, Any],
    swap_start: int,
    throttled_start: int,
) -> dict[str, Any]:
    current_state = _collect_power_thermal_state()
    current_swap = _system_swap_used_bytes()
    current_throttled = _pages_throttled()
    matches_start = (
        current_state == start_state
        and current_swap == swap_start
        and current_throttled == throttled_start
    )
    snapshot.update(
        {
            "power_source": current_state["power_source"],
            "thermal_warning": current_state["thermal_warning"],
            "performance_warning": current_state["performance_warning"],
            "system_swap_used_bytes": current_swap,
            "memory_pressure_pages_throttled": current_throttled,
            "system_state_matches_gate_start": matches_start,
            "passed": snapshot.get("passed") is True and matches_start,
        }
    )
    return snapshot


def _host_gate_receipt(
    contract: dict[str, Any],
    phase: str,
    host: dict[str, Any],
    start_state: dict[str, Any],
    end_state: dict[str, Any],
    swap_start: int,
    swap_end: int,
    throttled_start: int,
    throttled_end: int,
    snapshots: list[dict[str, Any]],
    accepted_runtime_swap_proofs: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    receipt = {
        "format": HOST_FORMAT,
        "schema_version": 3,
        "phase": phase,
        "host": host,
        "power_source": start_state["power_source"],
        "thermal_warning": start_state["thermal_warning"],
        "performance_warning": start_state["performance_warning"],
        "snapshot_interval_ms": 250,
        "snapshots": snapshots,
        "system_swap_used_bytes_start": swap_start,
        "memory_pressure_pages_throttled_start": throttled_start,
        "swap_delta_bytes": swap_end - swap_start,
        "memory_pressure_pages_throttled_delta": throttled_end
        - throttled_start,
        "power_or_thermal_state_changed": start_state != end_state,
        "processes_terminated_or_modified": False,
        "accepted_runtime_swap_proofs": copy.deepcopy(
            accepted_runtime_swap_proofs or []
        ),
        "passed": (
            all(snapshot["passed"] for snapshot in snapshots)
            and swap_end == swap_start
            and throttled_end == throttled_start
            and start_state == end_state
        ),
    }
    return receipt


def collect_host_gate(
    phase: str,
    contract: dict[str, Any],
    *,
    sleeper: Any = time.sleep,
) -> dict[str, Any]:
    _require(phase in ("preflight", "postflight"), "host gate collector phase is invalid")
    probe = MacQuietHostProbe(contract)
    start_state = _collect_power_thermal_state()
    swap_start = _system_swap_used_bytes()
    throttled_start = _pages_throttled()
    phase_contract = contract["host_quiet_gate"][phase]
    if phase == "postflight":
        sleeper(phase_contract["cooldown_before_snapshots_ms"] / 1000)
    probe.prime()
    snapshots = []
    for index in range(phase_contract["snapshot_count"]):
        sleeper(
            probe.seconds_until_next_window(
                phase_contract["snapshot_interval_ms"]
            )
        )
        snapshots.append(
            _attach_snapshot_system_state(
                probe.snapshot(index),
                start_state,
                swap_start,
                throttled_start,
            )
        )
    end_state = _collect_power_thermal_state()
    receipt = _host_gate_receipt(
        contract,
        phase,
        probe.host,
        start_state,
        end_state,
        swap_start,
        _system_swap_used_bytes(),
        throttled_start,
        _pages_throttled(),
        snapshots,
    )
    return validate_host_receipt(receipt, phase, contract)


class ContinuousHostMonitor:
    """Continuous 250-ms quiet gate that can abort an active runtime tree."""

    def __init__(self, contract: dict[str, Any]):
        self.contract = contract
        self.probe = MacQuietHostProbe(contract)
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._lock = threading.Lock()
        self._active_runtime: dict[str, Any] | None = None
        self._completed_runtime_proofs: dict[int, dict[str, Any]] = {}
        self._accepted_runtime_swap_proofs: list[dict[str, Any]] = []
        self._failure: BaseException | None = None
        self._snapshots: list[dict[str, Any]] = []
        self._thread: threading.Thread | None = None
        self._start_state: dict[str, Any] | None = None
        self._swap_start: int | None = None
        self._throttled_start: int | None = None

    def start(self) -> None:
        _require(self._thread is None, "continuous monitor already started")
        self._start_state = _collect_power_thermal_state()
        self._swap_start = _system_swap_used_bytes()
        self._throttled_start = _pages_throttled()
        self.probe.prime()
        self._thread = threading.Thread(
            target=self._run,
            name="formal-v3-custody-monitor",
            daemon=False,
        )
        self._thread.start()

    def _run(self) -> None:
        interval_ms = self.contract["host_quiet_gate"]["continuous_monitor"][
            "sample_interval_ms"
        ]
        while True:
            delay = self.probe.seconds_until_next_window(interval_ms)
            if self._stop.wait(delay):
                return
            try:
                with self._lock:
                    active = copy.deepcopy(self._active_runtime)
                    snapshot = self.probe.snapshot(
                        len(self._snapshots), active
                    )
                    if (
                        active is not None
                        and snapshot.get("active_runtime_swap_proof_complete")
                        is True
                        and snapshot.get("campaign_process_swap_bytes") == 0
                        and self._active_runtime is not None
                        and self._active_runtime["pid"] == active["pid"]
                    ):
                        self._active_runtime["swap_zero_observed"] = True
                        self._active_runtime[
                            "observed_process_identity"
                        ] = next(
                            (
                                copy.deepcopy(entry)
                                for entry in snapshot[
                                    "campaign_process_swap_observations"
                                ]
                                if entry["pid"] == active["pid"]
                            ),
                            None,
                        )
                snapshot = _attach_snapshot_system_state(
                    snapshot,
                    self._start_state,
                    self._swap_start,
                    self._throttled_start,
                )
                self._snapshots.append(snapshot)
                self._ready.set()
                if not snapshot["passed"]:
                    raise CampaignError("continuous quiet-host snapshot failed")
            except BaseException as error:
                self._failure = error
                self._ready.set()
                self._stop.set()
                return

    def wait_until_ready(self, timeout_seconds: float = 30.0) -> None:
        _require(self._ready.wait(timeout_seconds), "continuous monitor produced no first snapshot")
        self.assert_healthy()

    def assert_healthy(self) -> None:
        if self._failure is not None:
            raise CampaignError(f"continuous quiet-host gate failed: {self._failure}")

    def set_active_runtime(
        self,
        pid: int,
        arm: str,
        executable_and_library_hashes: dict[str, Any],
    ) -> None:
        with self._lock:
            _require(self._active_runtime is None, "two measured runtimes would overlap")
            self._active_runtime = {
                "pid": pid,
                "arm": arm,
                "executable_and_library_hashes": copy.deepcopy(
                    executable_and_library_hashes
                ),
                "swap_zero_observed": False,
                "observed_process_identity": None,
            }

    def clear_active_runtime(self, pid: int) -> None:
        with self._lock:
            if self._active_runtime is not None:
                _require(self._active_runtime["pid"] == pid, "active runtime PID identity drifted")
                _require(
                    pid not in self._completed_runtime_proofs,
                    "runtime PID proof was reused before receipt validation",
                )
                self._completed_runtime_proofs[pid] = {
                    "pid": pid,
                    "arm": self._active_runtime["arm"],
                    "swap_zero_proven": self._active_runtime[
                        "swap_zero_observed"
                    ],
                    "observed_process_identity": copy.deepcopy(
                        self._active_runtime["observed_process_identity"]
                    ),
                }
                self._active_runtime = None

    def confirm_runtime_receipt(self, pid: int) -> dict[str, Any]:
        """Fail closed unless a live PID/start identity had a zero-swap sample."""

        with self._lock:
            proof = self._completed_runtime_proofs.pop(pid, None)
            _require(proof is not None, "runtime receipt has no completed PID custody proof")
            _require(
                proof.get("swap_zero_proven") is True
                and isinstance(proof.get("observed_process_identity"), dict),
                "runtime exited before any complete zero-swap PID custody snapshot",
            )
            self._accepted_runtime_swap_proofs.append(copy.deepcopy(proof))
            return copy.deepcopy(proof)

    def stop_and_receipt(self) -> dict[str, Any]:
        _require(self._thread is not None, "continuous monitor was never started")
        self._stop.set()
        self._thread.join(timeout=30.0)
        _require(not self._thread.is_alive(), "continuous monitor did not stop")
        end_state = _collect_power_thermal_state()
        receipt = _host_gate_receipt(
            self.contract,
            "continuous",
            self.probe.host,
            self._start_state,
            end_state,
            self._swap_start,
            _system_swap_used_bytes(),
            self._throttled_start,
            _pages_throttled(),
            self._snapshots,
            self._accepted_runtime_swap_proofs,
        )
        self.assert_healthy()
        return validate_host_receipt(receipt, "continuous", self.contract)


def _marker_path_absent(path: Path, label: str) -> None:
    _require(not os.path.lexists(path), f"{label} already exists: {path}")


def _build_pre_marker_receipt(
    context: dict[str, Any],
    git_custody: dict[str, Any],
    artifact_observations: dict[str, Any],
    teacher: dict[str, Any],
    host_preflight: dict[str, Any],
    blocker_resolution: dict[str, Any],
    an_source_binding: dict[str, Any],
) -> dict[str, Any]:
    contract = context["contract"]
    plan = context["plan"]
    tracked = git_custody["tracked_files"]
    _require(
        blocker_resolution[
            "all_pre_marker_blockers_except_public_activation_resolved"
        ]
        is True
        and blocker_resolution["all_resolved"] is False,
        "pre-marker blocker resolution state is invalid",
    )
    contract_tracked = tracked["contract"]
    return {
        "format": MARKER_FORMAT,
        "schema_version": 3,
        "campaign_id": contract["campaign_id"],
        "subcampaign_id": contract["comparison_graph"]["edges"][EDGE_ID][
            "subcampaign_id"
        ],
        "edge_id": EDGE_ID,
        "contract_repository_path": plan["contract_repository_path"],
        "contract_blob_size_bytes": contract_tracked["blob_size_bytes"],
        "contract_blob_sha256": contract_tracked["blob_sha256"],
        "validator_repository_path": plan["validator_repository_path"],
        "validator_blob_sha256": tracked["validator"]["blob_sha256"],
        "driver_repository_path": plan["driver_repository_path"],
        "driver_blob_sha256": tracked["driver"]["blob_sha256"],
        "plan_repository_path": plan["plan_repository_path"],
        "plan_blob_size_bytes": tracked["plan"]["blob_size_bytes"],
        "plan_blob_sha256": tracked["plan"]["blob_sha256"],
        "marker_repository_path": plan["marker_repository_path"],
        "pre_marker_git_custody": git_custody,
        "an_campaign_source": copy.deepcopy(an_source_binding),
        "artifact_custody": copy.deepcopy(plan["artifacts"]),
        "artifact_file_observations": artifact_observations,
        "parity_admission": teacher,
        "host_preflight": host_preflight,
        "blocker_resolution": blocker_resolution,
        "declared_schedule": declared_schedule(),
        "sampling_state_at_marker_creation": {
            "generation_requests": 0,
            "warmup_samples": 0,
            "timed_samples": 0,
        },
        "pre_marker_admission": {
            "immutable_contract_and_validator": True,
            "live_public_git_and_clean_worktree": True,
            "artifacts_and_named_deployments": True,
            "teacher_reference_and_both_runtime_receipts": True,
            "quiet_host": True,
            "marker_was_absent_during_every_preflight": True,
            "all_passed": True,
        },
        "next_required_action": (
            "commit this create-new marker, push it to live refs/heads/main, "
            "then run the same driver; no generation is requested by prepare"
        ),
    }


def prepare_campaign(
    plan_path: Path | str,
    *,
    command_runner: Any = _system_command_runner,
    host_gate_collector: Any = collect_host_gate,
) -> dict[str, Any]:
    """Prove every pre-marker gate and exclusively create the frozen marker."""

    context = load_execution_context(plan_path)
    plan = context["plan"]
    contract = context["contract"]
    repository_root = context["repository_root"]
    marker_path = repository_root / plan["marker_repository_path"]
    raw_path = Path(plan["raw_output_path"])

    def preflight() -> dict[str, Any]:
        _marker_path_absent(marker_path, "campaign-start marker")
        _marker_path_absent(raw_path, "formal raw output")
        _require(marker_path.parent.resolve(strict=True).is_dir(), "marker parent is absent")
        _require(raw_path.parent.resolve(strict=True).is_dir(), "raw-output parent is absent")
        git_custody = collect_git_custody(
            repository_root,
            contract,
            _tracked_campaign_paths(plan, include_marker=False),
            command_runner=command_runner,
        )
        tracked = git_custody["tracked_files"]
        _require(
            tracked["contract"]["blob_sha256"]
            == context["contract_file"]["sha256"]
            == FROZEN_CONTRACT_SHA256,
            "Git contract blob differs from the frozen loaded bytes",
        )
        _require(
            tracked["validator"]["blob_sha256"]
            == context["validator_file"]["sha256"]
            == FROZEN_VALIDATOR_SHA256,
            "Git validator blob differs from the frozen loaded bytes",
        )
        _require(
            tracked["plan"]["blob_sha256"] == context["plan_file"]["sha256"],
            "Git plan blob differs from the loaded plan bytes",
        )
        artifacts = verify_plan_artifacts(plan)
        teacher = collect_teacher_admissions(
            repository_root, plan, contract, git_custody, artifacts
        )
        an_source_binding = bind_an_source_custodies_to_git(
            repository_root,
            plan,
            git_custody,
            _teacher_an_source_custodies(teacher),
            command_runner=command_runner,
        )
        host_preflight = host_gate_collector("preflight", contract)
        validate_host_receipt(host_preflight, "preflight", contract)
        _marker_path_absent(marker_path, "campaign-start marker")
        resolution = native_blocker_resolution(
            contract,
            teacher_evidence={
                arm: _sha256_compact(teacher["admissions"][arm])
                for arm in ("AN", "L")
            },
            binary_evidence={
                "driver_blob_sha256": tracked["driver"]["blob_sha256"],
                "runner_sha256": {
                    arm: plan["artifacts"][arm]["runner"]["sha256"]
                    for arm in ("AN", "L")
                },
                "artifact_file_observations_sha256": _sha256_compact(
                    artifacts
                ),
            },
            quiet_host_evidence=_sha256_compact(host_preflight),
            public_activation_evidence=None,
        )
        return _build_pre_marker_receipt(
            context,
            git_custody,
            artifacts,
            teacher,
            host_preflight,
            resolution,
            an_source_binding,
        )

    return prepare_marker_after_preflight(marker_path, preflight)


def _prove_git_ancestor(
    repository_root: Path,
    ancestor: str,
    descendant: str,
    command_runner: Any,
) -> None:
    result = _invoke_command(
        command_runner,
        ["/usr/bin/git", "merge-base", "--is-ancestor", ancestor, descendant],
        repository_root,
        env=git_custody_environment(),
    )
    _require(
        result["returncode"] == 0
        and result["stdout"] == b""
        and result["stderr"] == b"",
        "pre-marker public commit is not an ancestor of the marker commit",
    )


def _validate_published_marker(
    marker: dict[str, Any],
    marker_file: dict[str, Any],
    context: dict[str, Any],
    git_custody: dict[str, Any],
    artifacts: dict[str, Any],
    teacher: dict[str, Any],
    an_source_binding: dict[str, Any],
    command_runner: Any,
) -> None:
    contract = context["contract"]
    plan = context["plan"]
    _require(marker.get("format") == MARKER_FORMAT, "published marker format mismatch")
    _require(marker.get("schema_version") == 3, "published marker schema mismatch")
    _require(marker.get("campaign_id") == contract["campaign_id"], "published marker campaign mismatch")
    _require(
        marker.get("subcampaign_id")
        == contract["comparison_graph"]["edges"][EDGE_ID]["subcampaign_id"]
        and marker.get("edge_id") == EDGE_ID,
        "published marker edge binding mismatch",
    )
    tracked = git_custody["tracked_files"]
    _require(
        marker.get("contract_blob_size_bytes")
        == tracked["contract"]["blob_size_bytes"]
        and marker.get("contract_blob_sha256")
        == tracked["contract"]["blob_sha256"]
        == FROZEN_CONTRACT_SHA256,
        "published marker contract binding mismatch",
    )
    _require(
        marker.get("validator_blob_sha256")
        == tracked["validator"]["blob_sha256"]
        == FROZEN_VALIDATOR_SHA256
        and marker.get("driver_blob_sha256")
        == tracked["driver"]["blob_sha256"]
        and marker.get("plan_blob_sha256") == tracked["plan"]["blob_sha256"],
        "published marker validator/driver/plan binding mismatch",
    )
    _require(
        marker.get("marker_repository_path") == plan["marker_repository_path"]
        and marker_file["size_bytes"]
        == tracked["activation_marker"]["blob_size_bytes"]
        and marker_file["sha256"]
        == tracked["activation_marker"]["blob_sha256"],
        "published marker file/blob custody mismatch",
    )
    pre_git = marker.get("pre_marker_git_custody")
    _require(
        isinstance(pre_git, dict)
        and pre_git.get("worktree_clean") is True
        and pre_git.get("head_commit") == pre_git.get("ls_remote_live_oid"),
        "marker did not bind a clean live-public pre-marker commit",
    )
    pre_source = marker.get("an_campaign_source")
    _require(
        isinstance(pre_source, dict)
        and pre_source.get("campaign_commit")
        == an_source_binding.get("campaign_commit")
        and pre_source.get("campaign_tree")
        == an_source_binding.get("campaign_tree")
        and pre_source.get("repository_root")
        == an_source_binding.get("repository_root")
        and pre_source.get("source_file_count")
        == an_source_binding.get("source_file_count")
        and pre_source.get("source_files")
        == an_source_binding.get("source_files")
        and pre_source.get("source_files_sha256")
        == an_source_binding.get("source_files_sha256")
        and pre_source.get("live_head") == pre_git.get("head_commit")
        and pre_source.get("is_ancestor_of_live_head") is True
        and pre_source.get("clean_checkout") is True
        and pre_source.get(
            "live_descendant_changes_are_append_only_evidence"
        )
        is True,
        "published marker AN campaign source binding mismatch",
    )
    for label, entry in pre_git["tracked_files"].items():
        _require(label in tracked, f"published marker tracked label vanished: {label}")
        _require(
            entry["repository_path"] == tracked[label]["repository_path"]
            and entry["blob_oid"] == tracked[label]["blob_oid"]
            and entry["blob_sha256"] == tracked[label]["blob_sha256"],
            f"tracked pre-marker evidence changed before activation: {label}",
        )
    _prove_git_ancestor(
        context["repository_root"],
        pre_git["head_commit"],
        git_custody["activation_commit"],
        command_runner,
    )
    _require(
        marker.get("artifact_custody") == plan["artifacts"]
        and marker.get("artifact_file_observations") == artifacts,
        "published marker artifact custody changed",
    )
    _require(marker.get("parity_admission") == teacher, "published marker teacher admission changed")
    validate_host_receipt(marker.get("host_preflight"), "preflight", contract)
    _require(
        marker.get("sampling_state_at_marker_creation")
        == {"generation_requests": 0, "warmup_samples": 0, "timed_samples": 0},
        "published marker claims pre-marker generation",
    )
    resolution = marker.get("blocker_resolution")
    _require(
        isinstance(resolution, dict)
        and resolution.get(
            "all_pre_marker_blockers_except_public_activation_resolved"
        )
        is True
        and resolution.get("all_resolved") is False
        and resolution["resolution_map"][
            "NATIVE_SUBCAMPAIGN_NOT_PUBLICLY_ACTIVATED"
        ]["resolved"]
        is False,
        "published marker pre-activation blocker map is invalid",
    )
    _require(
        marker.get("pre_marker_admission", {}).get("all_passed") is True,
        "published marker was not created after all pre-marker gates",
    )


def _final_contract_binding(
    context: dict[str, Any], git_custody: dict[str, Any]
) -> dict[str, Any]:
    plan = context["plan"]
    tracked = git_custody["tracked_files"]
    contract_entry = tracked["contract"]
    marker_entry = tracked["activation_marker"]
    return {
        "campaign_id": context["contract"]["campaign_id"],
        "schema_version": 3,
        "edge_id": EDGE_ID,
        "subcampaign_id": context["contract"]["comparison_graph"]["edges"][
            EDGE_ID
        ]["subcampaign_id"],
        "repository_url": git_custody["repository_url"],
        "remote_origin_url": git_custody["remote_origin_url"],
        "local_tracking_ref": git_custody["local_tracking_ref"],
        "local_tracking_oid": git_custody["local_tracking_oid"],
        "live_remote_url": git_custody["live_remote_url"],
        "live_remote_ref": git_custody["live_remote_ref"],
        "ls_remote_argv": git_custody["ls_remote_argv"],
        "ls_remote_exit_code": git_custody["ls_remote_exit_code"],
        "ls_remote_live_oid": git_custody["ls_remote_live_oid"],
        "head_commit": git_custody["head_commit"],
        "contract_repository_path": plan["contract_repository_path"],
        "contract_commit": git_custody["contract_commit"],
        "contract_tree": git_custody["contract_tree"],
        "contract_blob_oid": contract_entry["blob_oid"],
        "contract_blob_size_bytes": contract_entry["blob_size_bytes"],
        "contract_blob_sha256": contract_entry["blob_sha256"],
        "observed_file_size_bytes": contract_entry["observed_size_bytes"],
        "observed_file_sha256": contract_entry["observed_sha256"],
        "activation_commit": git_custody["activation_commit"],
        "activation_tree": git_custody["head_tree"],
        "activation_contract_blob_oid": contract_entry["blob_oid"],
        "activation_contract_blob_size_bytes": contract_entry["blob_size_bytes"],
        "activation_contract_blob_sha256": contract_entry["blob_sha256"],
        "activation_marker_repository_path": plan["marker_repository_path"],
        "activation_marker_blob_oid": marker_entry["blob_oid"],
        "activation_marker_blob_size_bytes": marker_entry["blob_size_bytes"],
        "activation_marker_blob_sha256": marker_entry["blob_sha256"],
        "contract_commit_is_ancestor_of_activation_commit": True,
        "activation_commit_equals_head_and_live_remote_oid": True,
        "local_tracking_ref_used_as_publication_proof": False,
        "worktree_clean": True,
    }


def _machine_artifact_custody(
    contract: dict[str, Any],
    plan: dict[str, Any],
    observations: dict[str, Any],
) -> dict[str, Any]:
    gateway = contract["runtime_custody"]["gateway_cohort"]
    return {
        "model": {
            "sha256": plan["artifacts"]["L"]["model"]["sha256"],
            "size_bytes": plan["artifacts"]["L"]["model"]["size_bytes"],
        },
        "llama_cpp": {
            "source_commit": plan["artifacts"]["L"]["runtime_source_commit"],
            "runner": copy.deepcopy(plan["artifacts"]["L"]["runner"]),
        },
        "omniinfer": {
            "source_commit": gateway["omniinfer"]["source_commit"],
            "inactive_edge_contract_only": True,
        },
        "gateway_backend": {
            "source_commit": gateway["backend"]["source_commit"],
            "inactive_edge_contract_only": True,
        },
        "native_arms": copy.deepcopy(plan["artifacts"]),
        "file_observations_start": copy.deepcopy(observations),
    }


def _terminate_measured_process(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def invoke_fresh_runtime_process(
    repository_root: Path,
    plan: dict[str, Any],
    slot: dict[str, Any],
    nonce: str,
    monitor: ContinuousHostMonitor,
) -> dict[str, Any]:
    """Invoke one measured arm once in its own process group."""

    arm = slot["arm"]
    command = plan["commands"][arm]
    request = {
        "nonce": nonce,
        "sequence_index": slot["sequence_index"],
        "phase": slot["phase"],
        "warmup_index": slot["warmup_index"],
        "block_index": slot["block_index"],
        "slot_index": slot["slot_index"],
        "role": slot["role"],
        "arm": arm,
    }
    environment = dict(command["environment"])
    environment["APXINF_FORMAL_V3_REQUEST_JSON"] = compact_json_bytes(
        request
    ).decode("utf-8")
    started_ns = time.monotonic_ns()
    process = subprocess.Popen(
        command["argv"],
        cwd=str(repository_root),
        env=environment,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    try:
        monitor.set_active_runtime(
            process.pid,
            arm,
            {
                "runner_sha256": plan["artifacts"][arm]["runner"]["sha256"],
                "loaded_non_system_library_closure_sha256": plan["artifacts"][arm][
                    "loaded_non_system_library_closure_sha256"
                ],
            },
        )
    except BaseException as error:
        _terminate_measured_process(process)
        stdout, stderr = process.communicate()
        raise RuntimeInvocationError(
            f"could not bind active runtime custody in slot {slot['sequence_index']}: {error}",
            {
                "pid": process.pid,
                "returncode": process.returncode,
                "stdout": stdout,
                "stderr": stderr,
            },
        ) from error
    stdout = b""
    stderr = b""
    timed_out = False
    monitor_failure: BaseException | None = None
    try:
        deadline = time.monotonic() + plan["timeout_seconds"]
        while True:
            try:
                stdout, stderr = process.communicate(timeout=0.05)
                break
            except subprocess.TimeoutExpired:
                try:
                    monitor.assert_healthy()
                except BaseException as error:
                    monitor_failure = error
                    _terminate_measured_process(process)
                    stdout, stderr = process.communicate()
                    break
                if time.monotonic() >= deadline:
                    timed_out = True
                    _terminate_measured_process(process)
                    stdout, stderr = process.communicate()
                    break
    except BaseException:
        _terminate_measured_process(process)
        stdout, stderr = process.communicate()
        raise
    finally:
        monitor.clear_active_runtime(process.pid)
    observation = {
        "pid": process.pid,
        "process_group_id": process.pid,
        "fresh_process": True,
        "started_monotonic_ns": started_ns,
        "ended_monotonic_ns": time.monotonic_ns(),
        "returncode": process.returncode,
        "timed_out": timed_out,
        "argv": command["argv"],
        "environment_keys": sorted(environment),
        "stdout": stdout,
        "stderr": stderr,
    }
    if monitor_failure is not None:
        raise RuntimeInvocationError(
            f"continuous host monitor failed during slot {slot['sequence_index']}: {monitor_failure}",
            observation,
        )
    if timed_out:
        raise RuntimeInvocationError(
            f"runtime timed out in slot {slot['sequence_index']}", observation
        )
    if process.returncode != 0:
        raise RuntimeInvocationError(
            f"runtime exited {process.returncode} in slot {slot['sequence_index']}",
            observation,
        )
    if stderr != b"":
        raise RuntimeInvocationError(
            f"runtime wrote stderr in slot {slot['sequence_index']}", observation
        )
    return observation


def _sample_from_external_process(
    repository_root: Path,
    plan: dict[str, Any],
    contract: dict[str, Any],
    slot: dict[str, Any],
    nonce: str,
    monitor: ContinuousHostMonitor,
    artifact_observations: dict[str, Any],
    an_source_binding: dict[str, Any],
) -> dict[str, Any]:
    observation = invoke_fresh_runtime_process(
        repository_root, plan, slot, nonce, monitor
    )
    try:
        raw = parse_single_json_line(observation["stdout"])
        if slot["arm"] == "AN":
            sample = validate_an_free_receipt(
                raw,
                slot,
                nonce,
                contract,
                plan["artifacts"]["AN"],
                an_source_binding,
            )
        else:
            sample = adapt_llama_free_receipt(
                raw,
                slot,
                nonce,
                contract,
                plan["artifacts"]["L"],
                artifact_observations["L"],
            )
        runtime_swap_proof = monitor.confirm_runtime_receipt(
            observation["pid"]
        )
        sample["driver_process_observation"] = {
            "pid": observation["pid"],
            "process_group_id": observation["process_group_id"],
            "fresh_process": True,
            "started_monotonic_ns": observation["started_monotonic_ns"],
            "ended_monotonic_ns": observation["ended_monotonic_ns"],
            "returncode": observation["returncode"],
            "timed_out": observation["timed_out"],
            "argv_sha256": _sha256_compact(observation["argv"]),
            "environment_keys": observation["environment_keys"],
            "stdout_size_bytes": len(observation["stdout"]),
            "stdout_sha256": hashlib.sha256(observation["stdout"]).hexdigest(),
            "stderr_size_bytes": len(observation["stderr"]),
            "stderr_sha256": hashlib.sha256(observation["stderr"]).hexdigest(),
            "runtime_swap_proof_sha256": _sha256_compact(
                runtime_swap_proof
            ),
        }
        return sample
    except BaseException as error:
        raise RuntimeInvocationError(
            f"runtime receipt rejected in slot {slot['sequence_index']}: {error}",
            observation,
        ) from error


def _pre_schedule_failure_record(
    contract: dict[str, Any],
    plan: dict[str, Any],
    error: BaseException,
    evidence: dict[str, Any],
) -> dict[str, Any]:
    slots = [
        {
            "slot": slot,
            "status": "unattempted",
            "attempt_count": 0,
            "nonce": None,
            "receipt_sha256": None,
            "failure_index": None,
        }
        for slot in declared_schedule()
    ]
    return {
        "format": RAW_FORMAT,
        "schema_version": 3,
        "campaign_id": contract["campaign_id"],
        "subcampaign_id": contract["comparison_graph"]["edges"][EDGE_ID][
            "subcampaign_id"
        ],
        "edge_id": EDGE_ID,
        "status": "CONSUMED_FIRST_POST_MARKER_FAILURE",
        "raw_output_path": plan["raw_output_path"],
        "contract_binding": evidence.get("contract_binding", {}),
        "git_custody": evidence.get("git_custody", {}),
        "host_custody": {
            **{
                field: contract["scope"]["host"][field]
                for field in (
                    "model_identifier",
                    "chip",
                    "architecture",
                    "logical_cpu_count",
                    "memory_bytes",
                    "os_product",
                    "os_version",
                    "os_build",
                )
            },
            **evidence.get("host_custody", {}),
        },
        "artifact_custody": evidence.get("artifact_custody", plan["artifacts"]),
        "parity_admission": evidence.get("parity_admission", {}),
        "blocker_resolution": evidence.get("blocker_resolution", {}),
        "schedule_receipt": {
            "process_state": "fresh-process-per-sample",
            "warmup_order": list(WARMUP_ROLES),
            "timed_block_orders": list(TIMED_BLOCK_ORDERS),
            "slots": slots,
            "attempted_count": 0,
            "accepted_count": 0,
            "failed_count": 0,
            "remaining_unattempted_count": 38,
            "stopped_at_first_failure": True,
            "retry_replacement_or_extension_performed": False,
        },
        "samples": [],
        "failures": [
            {
                "stage": "campaign-start-reproof",
                "sequence_index": None,
                "arm": None,
                "exception_type": type(error).__name__,
                "message": str(error),
                "observation": {},
                "remaining_slots_marked_unattempted": True,
                "failed_observation_retained": True,
            }
        ],
        "statistics": None,
        "gates": _required_native_gates(False, False),
        "decision": {"label": "UNRANKABLE", "formal_summary_allowed": False},
    }


def _execution_context_failure_record(
    bootstrap: dict[str, Any], error: BaseException
) -> dict[str, Any]:
    """Record a consumed marker even when the frozen context cannot load."""

    plan = bootstrap["plan"]
    slots = [
        {
            "slot": slot,
            "status": "unattempted",
            "attempt_count": 0,
            "nonce": None,
            "receipt_sha256": None,
            "failure_index": None,
        }
        for slot in declared_schedule()
    ]
    return {
        "format": RAW_FORMAT,
        "schema_version": 3,
        "campaign_id": FROZEN_CAMPAIGN_ID,
        "subcampaign_id": FROZEN_NATIVE_SUBCAMPAIGN_ID,
        "edge_id": EDGE_ID,
        "status": "CONSUMED_FIRST_POST_MARKER_FAILURE",
        "raw_output_path": plan["raw_output_path"],
        "contract_binding": {
            "campaign_id": FROZEN_CAMPAIGN_ID,
            "schema_version": 3,
            "edge_id": EDGE_ID,
            "subcampaign_id": FROZEN_NATIVE_SUBCAMPAIGN_ID,
            "execution_context_loaded": False,
            "plan_repository_path": plan["plan_repository_path"],
            "plan_blob_size_bytes": bootstrap["plan_file"]["size_bytes"],
            "plan_blob_sha256": bootstrap["plan_file"]["sha256"],
            "activation_marker_repository_path": plan[
                "marker_repository_path"
            ],
            "activation_marker_size_bytes": bootstrap["marker_file"][
                "size_bytes"
            ],
            "activation_marker_sha256": bootstrap["marker_file"]["sha256"],
        },
        "git_custody": {},
        "host_custody": {},
        "artifact_custody": {},
        "parity_admission": {},
        "blocker_resolution": {},
        "schedule_receipt": {
            "process_state": "fresh-process-per-sample",
            "warmup_order": list(WARMUP_ROLES),
            "timed_block_orders": list(TIMED_BLOCK_ORDERS),
            "slots": slots,
            "attempted_count": 0,
            "accepted_count": 0,
            "failed_count": 0,
            "remaining_unattempted_count": 38,
            "stopped_at_first_failure": True,
            "retry_replacement_or_extension_performed": False,
        },
        "samples": [],
        "failures": [
            {
                "stage": "execution-context-load",
                "sequence_index": None,
                "arm": None,
                "exception_type": type(error).__name__,
                "message": str(error),
                "observation": {
                    "plan_file": copy.deepcopy(bootstrap["plan_file"]),
                    "marker_file": copy.deepcopy(bootstrap["marker_file"]),
                },
                "remaining_slots_marked_unattempted": True,
                "failed_observation_retained": True,
            }
        ],
        "statistics": None,
        "gates": _required_native_gates(False, False),
        "decision": {"label": "UNRANKABLE", "formal_summary_allowed": False},
    }


def run_campaign(
    plan_path: Path | str,
    *,
    command_runner: Any = _system_command_runner,
    host_gate_collector: Any = collect_host_gate,
    monitor_factory: Any = ContinuousHostMonitor,
) -> dict[str, Any]:
    """Re-prove the published marker, then consume the immutable 38-slot run."""

    bootstrap = _bootstrap_post_marker_run(plan_path)
    try:
        context = load_execution_context(plan_path)
        _require(
            context["repository_root"] == bootstrap["repository_root"]
            and context["plan_file"] == bootstrap["plan_file"]
            and context["plan"] == bootstrap["plan"],
            "execution context changed after the post-marker bootstrap",
        )
    except BaseException as error:
        partial = _execution_context_failure_record(bootstrap, error)
        atomic_create_json(bootstrap["raw_path"], partial)
        return partial
    repository_root = context["repository_root"]
    plan = context["plan"]
    contract = context["contract"]
    marker_path = repository_root / plan["marker_repository_path"]
    raw_path = Path(plan["raw_output_path"])
    _require(os.path.lexists(marker_path), "campaign-start marker is absent; run prepare first")
    evidence: dict[str, Any] = {}
    try:
        git_custody = collect_git_custody(
            repository_root,
            contract,
            _tracked_campaign_paths(plan, include_marker=True),
            command_runner=command_runner,
            published_marker_label="activation_marker",
        )
        evidence["git_custody"] = git_custody
        marker, marker_file = _read_tracked_single_line_receipt(
            repository_root,
            plan["marker_repository_path"],
            git_custody["tracked_files"]["activation_marker"],
        )
        artifacts = verify_plan_artifacts(plan)
        teacher = collect_teacher_admissions(
            repository_root, plan, contract, git_custody, artifacts
        )
        marker_source = _require_mapping(
            marker.get("an_campaign_source"),
            "published marker AN source binding",
        )
        an_source_binding = bind_an_source_custodies_to_git(
            repository_root,
            plan,
            git_custody,
            _teacher_an_source_custodies(teacher),
            command_runner=command_runner,
            expected_source_tree=marker_source.get("campaign_tree"),
        )
        _validate_published_marker(
            marker,
            marker_file,
            context,
            git_custody,
            artifacts,
            teacher,
            an_source_binding,
            command_runner,
        )
        host_preflight = host_gate_collector("preflight", contract)
        validate_host_receipt(host_preflight, "preflight", contract)
        contract_binding = _final_contract_binding(context, git_custody)
        resolution = native_blocker_resolution(
            contract,
            teacher_evidence={
                arm: _sha256_compact(teacher["admissions"][arm])
                for arm in ("AN", "L")
            },
            binary_evidence={
                "driver_blob_sha256": git_custody["tracked_files"]["driver"][
                    "blob_sha256"
                ],
                "runner_sha256": {
                    arm: plan["artifacts"][arm]["runner"]["sha256"]
                    for arm in ("AN", "L")
                },
                "artifact_file_observations_sha256": _sha256_compact(
                    artifacts
                ),
            },
            quiet_host_evidence=_sha256_compact(host_preflight),
            public_activation_evidence={
                "activation_commit": git_custody["activation_commit"],
                "live_remote_oid": git_custody["ls_remote_live_oid"],
                "marker_blob_sha256": marker_file["sha256"],
            },
        )
        _require(resolution["all_resolved"] is True, "campaign-start blocker resolution is incomplete")
        evidence.update(
            {
                "contract_binding": contract_binding,
                "host_custody": {"preflight": host_preflight},
                "artifact_custody": _machine_artifact_custody(
                    contract, plan, artifacts
                ),
                "artifact_file_observations": artifacts,
                "an_source_binding": an_source_binding,
                "parity_admission": teacher,
                "blocker_resolution": resolution,
                "enforce_machine_contract": True,
            }
        )
    except BaseException as error:
        partial = _pre_schedule_failure_record(
            contract, plan, error, evidence
        )
        atomic_create_json(raw_path, partial)
        return partial

    monitor = monitor_factory(contract)

    def before_first_slot() -> None:
        monitor.start()
        monitor.wait_until_ready()

    def sample_collector(slot: dict[str, Any], nonce: str) -> dict[str, Any]:
        monitor.assert_healthy()
        return _sample_from_external_process(
            repository_root,
            plan,
            contract,
            slot,
            nonce,
            monitor,
            artifacts,
            an_source_binding,
        )

    def postflight_collector() -> dict[str, Any]:
        postflight_error: BaseException | None = None
        postflight: dict[str, Any] | None = None
        continuous: dict[str, Any] | None = None
        try:
            postflight = host_gate_collector("postflight", contract)
        except BaseException as error:
            postflight_error = error
        try:
            continuous = monitor.stop_and_receipt()
        except BaseException as error:
            if postflight_error is None:
                postflight_error = error
        if postflight_error is not None:
            raise CampaignError(f"host postflight or continuous monitor failed: {postflight_error}")
        artifact_end = verify_plan_artifacts(plan)
        return {
            "continuous": continuous,
            "postflight": postflight,
            "artifact_custody_end": copy.deepcopy(plan["artifacts"]),
            "artifact_file_observations_end": artifact_end,
        }

    return execute_formal_schedule(
        contract,
        plan,
        evidence,
        raw_path,
        sample_collector=sample_collector,
        postflight_collector=postflight_collector,
        before_first_slot=before_first_slot,
    )


def _self_test_custody(arm: str, contract: dict[str, Any]) -> dict[str, Any]:
    return {
        "configuration_id": contract["native_deployment_contract"]["deployments"][
            arm
        ]["configuration_id"],
        "runner": {
            "absolute_path": f"/fixture/{arm.lower()}-runner",
            "size_bytes": 100,
            "sha256": ("a" if arm == "AN" else "b") * 64,
        },
        "model": {
            "absolute_path": f"/fixture/{arm.lower()}-model",
            "size_bytes": 200,
            "sha256": ("c" if arm == "AN" else "d") * 64,
        },
        "runtime_source_commit": ("e" * 40 if arm == "AN" else "f" * 40),
        "loaded_non_system_library_closure_sha256": "1" * 64,
        "packed_weight_and_resident_buffer_manifest_sha256": (
            "2" * 64 if arm == "AN" else None
        ),
        "deployment": expected_deployment_receipt(arm),
    }


def _self_test_sample(
    contract: dict[str, Any],
    artifacts: dict[str, Any],
    slot: dict[str, Any],
    nonce: str,
) -> dict[str, Any]:
    prompt = contract["workload_contracts"]["shared_prompt"]["token_ids"]
    trajectory_contract = contract["workload_contracts"][
        "NATIVE_RAW13_FREE128_V3"
    ]["free_run_trajectory_admission"]
    teacher_inputs = contract["workload_contracts"][
        "NATIVE_RAW13_FREE128_V3"
    ]["teacher_forced_admission"]["teacher_input_token_ids"]
    generated = teacher_inputs[1:] + [198]
    _require(
        _sha256_compact(generated) == trajectory_contract["expected_sha256"],
        "self-test canonical trajectory reconstruction drifted",
    )
    start = 1_000_000_000
    first = 1_010_000_000
    last = 2_280_000_000
    generation = contract["workload_contracts"]["NATIVE_RAW13_FREE128_V3"][
        "generation"
    ]
    return {
        "format": SAMPLE_FORMAT,
        "schema_version": 3,
        "campaign_id": contract["campaign_id"],
        "subcampaign_id": contract["comparison_graph"]["edges"][EDGE_ID][
            "subcampaign_id"
        ],
        "edge_id": EDGE_ID,
        "mode": "native-v3-free",
        "request": {
            "nonce": nonce,
            "sequence_index": slot["sequence_index"],
            "phase": slot["phase"],
            "warmup_index": slot["warmup_index"],
            "block_index": slot["block_index"],
            "slot_index": slot["slot_index"],
            "role": slot["role"],
            "arm": slot["arm"],
        },
        "workload": {
            "ingress_semantics": "raw-token-ids",
            "prompt_token_ids": prompt,
            "prefill_token_count": 13,
            "generated_token_ids": generated,
            "generated_token_ids_sha256": _sha256_compact(generated),
            "generated_token_count": 128,
            "sampling": generation["sampling"],
            "temperature": generation["temperature"],
            "eog_policy": generation["eog_policy"],
            "speculative_decoding": False,
            "continuous_batching": False,
            "sequence_count": 1,
            "requested_context_tokens": 256,
            "effective_context_tokens": 256,
            "requested_batch_tokens": 13,
            "effective_batch_tokens": 13,
            "requested_ubatch_tokens": 13,
            "effective_ubatch_tokens": 13,
            "empty_state_before_prefill": True,
            "prompt_cache_reused": False,
        },
        "timing": {
            "clock": "monotonic",
            "clock_identity": "fixture-monotonic-ns",
            "clock_resolution_ns": 1,
            "start_boundary": contract["timing_contract"][EDGE_ID]["start"],
            "common_token_ready_boundary": "next-greedy-token-ready",
            "end_boundary": contract["timing_contract"][EDGE_ID]["end"],
            "selection_work_included": True,
            "accelerator_completion_before_each_token_ready_timestamp": True,
            "final_sampled_token_decoded_inside_timed_region": False,
            "prefill_start_ns": start,
            "token_1_ready_ns": first,
            "token_128_ready_ns": last,
            "ttft_ms": (first - start) / 1_000_000,
            "total_latency_ms": (last - start) / 1_000_000,
            "tpot_ms": (last - first) / 127 / 1_000_000,
            "generation_tps": 127_000_000_000 / (last - first),
        },
        "custody": {
            **copy.deepcopy(artifacts[slot["arm"]]),
            "fresh_process": True,
            "start_end_identity_equal": True,
            "ggml_backend_path_unset": slot["arm"] == "L",
        },
    }


def _self_test_host_receipt(
    contract: dict[str, Any], phase: str
) -> dict[str, Any]:
    count = 5 if phase in ("preflight", "postflight") else 1
    snapshots = []
    for index in range(count):
        window_start_ns = 1_000_000_000 + index * 250_000_000
        snapshots.append(
            {
                "index": index,
                "cpu_window_start_monotonic_ns": window_start_ns,
                "monotonic_ns": window_start_ns + 250_000_000,
                "cpu_percent_window_ms": 250.0,
                "cpu_measurement_source": "libproc-PROC_PIDTASKINFO-delta",
                "resolved_allowlist": [
                    {
                        "role": "campaign_orchestrator",
                        "pid": 1,
                        "process_start_time": "fixture",
                    },
                    {
                        "role": "custody_monitor",
                        "pid": 1,
                        "process_start_time": "fixture",
                    },
                ],
                "nonallowlisted_processes": [],
                "vanished_nonallowlisted_processes": [],
                "cpu_window_proof_complete": True,
                "maximum_single_nonallowlisted_process_cpu_percent": 0.0,
                "aggregate_nonallowlisted_process_cpu_percent": 0.0,
                "load_average_per_logical_cpu": 0.0,
                "campaign_process_swap_bytes": 0,
                "campaign_process_swap_observations": [],
                "campaign_swap_probe_vanished_processes": [],
                "active_runtime_root_present": None,
                "active_runtime_swap_proof_complete": False,
                "power_source": "AC Power",
                "thermal_warning": False,
                "performance_warning": False,
                "system_swap_used_bytes": 0,
                "memory_pressure_pages_throttled": 0,
                "system_state_matches_gate_start": True,
                "passed": True,
            }
        )
    return {
        "format": HOST_FORMAT,
        "schema_version": 3,
        "phase": phase,
        "host": copy.deepcopy(contract["scope"]["host"]),
        "power_source": "AC Power",
        "thermal_warning": False,
        "performance_warning": False,
        "snapshot_interval_ms": 250,
        "snapshots": snapshots,
        "system_swap_used_bytes_start": 0,
        "memory_pressure_pages_throttled_start": 0,
        "swap_delta_bytes": 0,
        "memory_pressure_pages_throttled_delta": 0,
        "power_or_thermal_state_changed": False,
        "processes_terminated_or_modified": False,
        "accepted_runtime_swap_proofs": [],
        "passed": True,
    }


def run_fixture_self_test() -> dict[str, Any]:
    """Exercise both terminal paths without network, host probes or model processes."""

    repository_root = Path(__file__).resolve().parents[2]
    loaded = load_frozen_contract(
        repository_root / "configs/qwen35-0.8b-cross-runtime-formal-v3.json",
        repository_root
        / "scripts/validate_qwen35_cross_runtime_formal_contract.py",
    )
    contract = loaded["contract"]
    artifacts = {arm: _self_test_custody(arm, contract) for arm in ("AN", "L")}
    plan = {"artifacts": artifacts}
    evidence = {
        "contract_binding": {"fixture": True},
        "git_custody": {"fixture": True},
        "host_custody": {
            "preflight": _self_test_host_receipt(contract, "preflight")
        },
        "artifact_custody": artifacts,
        "parity_admission": {"AN": {"passed": True}, "L": {"passed": True}},
        "blocker_resolution": {"all_resolved": True},
    }
    complete_calls: list[int] = []
    failure_calls: list[int] = []

    def postflight() -> dict[str, Any]:
        return {
            "continuous": _self_test_host_receipt(contract, "continuous"),
            "postflight": _self_test_host_receipt(contract, "postflight"),
            "artifact_custody_end": artifacts,
        }

    with tempfile.TemporaryDirectory(prefix="apxinf-formal-v3-self-test-") as directory:
        temporary = Path(directory)

        def complete_collector(slot: dict[str, Any], nonce: str) -> dict[str, Any]:
            complete_calls.append(slot["sequence_index"])
            return _self_test_sample(contract, artifacts, slot, nonce)

        complete = execute_formal_schedule(
            contract,
            plan,
            evidence,
            temporary / "complete.json",
            sample_collector=complete_collector,
            postflight_collector=postflight,
            nonce_factory=lambda slot: f"{slot['sequence_index'] + 1:064x}",
        )

        def failure_collector(slot: dict[str, Any], nonce: str) -> dict[str, Any]:
            failure_calls.append(slot["sequence_index"])
            if slot["sequence_index"] == 3:
                raise RuntimeInvocationError(
                    "injected fixture crash", {"returncode": -6}
                )
            return _self_test_sample(contract, artifacts, slot, nonce)

        failed = execute_formal_schedule(
            contract,
            plan,
            evidence,
            temporary / "failed.json",
            sample_collector=failure_collector,
            postflight_collector=postflight,
            nonce_factory=lambda slot: f"{slot['sequence_index'] + 101:064x}",
        )
        _require(
            parse_single_json_line((temporary / "complete.json").read_bytes())
            == complete
            and parse_single_json_line((temporary / "failed.json").read_bytes())
            == failed,
            "self-test atomic raw receipts did not round-trip",
        )
    passed = (
        complete["status"] == "FORMAL_COMPLETE"
        and complete_calls == list(range(38))
        and failed["status"] == "CONSUMED_FIRST_POST_MARKER_FAILURE"
        and failure_calls == list(range(4))
        and all(
            entry["status"] == "unattempted"
            for entry in failed["schedule_receipt"]["slots"][4:]
        )
    )
    _require(passed, "fixture self-test terminal-state assertions failed")
    return {
        "format": "apxinf-qwen35-native-formal-v3-driver-self-test",
        "schema_version": 3,
        "passed": True,
        "complete_path_invocations": len(complete_calls),
        "failure_path_invocations": len(failure_calls),
        "network_used": False,
        "model_process_used": False,
    }


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fail-closed Qwen3.5 native ApxInf-vs-llama.cpp formal-v3 driver"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("prepare", "run"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--plan", required=True, type=Path)
    subparsers.add_parser("self-test")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _argument_parser().parse_args(argv)
    try:
        if args.command == "self-test":
            result = run_fixture_self_test()
        elif args.command == "prepare":
            marker = prepare_campaign(args.plan)
            result = {
                "status": "MARKER_CREATED_REQUIRES_COMMIT_AND_PUSH",
                "campaign_id": marker["campaign_id"],
                "edge_id": EDGE_ID,
                "marker_repository_path": marker["marker_repository_path"],
                "generation_requests": 0,
            }
        else:
            campaign = run_campaign(args.plan)
            result = {
                "status": campaign["status"],
                "campaign_id": campaign["campaign_id"],
                "edge_id": EDGE_ID,
                "raw_output_path": campaign["raw_output_path"],
                "accepted_count": campaign["schedule_receipt"]["accepted_count"],
                "decision": campaign["decision"]["label"],
            }
        sys.stdout.buffer.write(_json_line_bytes(result))
        return 0 if result.get("status") not in (
            "CONSUMED_FIRST_POST_MARKER_FAILURE",
            "FORMAL_UNRANKABLE",
        ) else 2
    except (CampaignError, OSError, subprocess.SubprocessError) as error:
        failure = {
            "format": "apxinf-qwen35-native-formal-v3-driver-error",
            "schema_version": 3,
            "error_type": type(error).__name__,
            "message": str(error),
        }
        sys.stderr.buffer.write(_json_line_bytes(failure))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
