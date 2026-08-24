#!/usr/bin/env python3
"""Onboard the pinned Qwen3.5 macOS bundle, staging it when authorized."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import tempfile
from typing import NoReturn, TextIO


ROOT = Path(__file__).resolve().parents[1]
RESOLVER = ROOT / "scripts/resolve_hf_source.py"
STAGER = ROOT / "scripts/stage_hf_bundle.py"
DEPLOYMENT_VERIFIER = ROOT / "scripts/verify_hf_macos_deployment.py"
PROFILE = ROOT / "configs/hf-onboarding/qwen35-0.8b-macos-cpu.json"

MODEL_URL = "https://huggingface.co/Qwen/Qwen3.5-0.8B"
REPO_ID = "Qwen/Qwen3.5-0.8B"
REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
PROFILE_ID = "qwen35-0.8b-macos-cpu"
BINARY_SIZE = 8_163_904
BINARY_SHA256 = "d9cb4de44b236b5b3f216a81079b11102220939a2b179cbc2678442ff947803b"
SANDBOX_EXEC = Path("/usr/bin/sandbox-exec")
SANDBOX_POLICY = "macos-seatbelt-deny-network-write-home-read-v1"
CONTROLLER_FORMAT = "apxinf-hf-macos-onboard-receipt-v2"
PLAN_FORMAT = "apxinf-hf-macos-onboard-plan-v2"
STAGER_FORMAT = "apxinf-hf-bundle-stage-receipt-v1"
GENERATION_FORMAT = "apxinf-generation-v1"
DEPLOYMENT_FORMAT = "apxinf-deployment-lock-v1"
DEPLOYMENT_RECEIPT_FORMAT = "apxinf-deployment-verification-receipt-v1"
MAX_JSON_BYTES = 16 * 1024 * 1024
MAX_CHILD_OUTPUT_BYTES = 2 * 1024 * 1024
STAGER_MAX_TOTAL_BYTES = 2 * 1024 * 1024 * 1024
SHA256 = re.compile(r"^[0-9a-f]{64}$")
ARTIFACT_MANIFEST_SHA256 = (
    "436821ae50e981b9176784ac6ff9548742a865d60d726c58d3bfa9f76d86b500"
)
REQUIRED_ARTIFACTS = frozenset(
    {
        "chat_template.jinja",
        "config.json",
        "model.safetensors-00001-of-00001.safetensors",
        "model.safetensors.index.json",
        "tokenizer.json",
        "tokenizer_config.json",
    }
)
SOURCE_RESOLVE_TIMEOUT = 120
SOURCE_VERIFY_TIMEOUT = 30
GENERATION_TIMEOUT = 300
DEPLOYMENT_VERIFY_TIMEOUT = 180
BUNDLE_STAGE_TIMEOUT = 2 * 60 * 60


class OnboardError(ValueError):
    """A deterministic, user-correctable onboarding failure."""


class JsonArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        raise OnboardError(f"invalid arguments: {message}")


@dataclass(frozen=True)
class Stage:
    name: str
    argv: tuple[str, ...]
    timeout_seconds: int
    network_policy: str

    def plan(self) -> dict[str, object]:
        return {
            "name": self.name,
            "argv": list(self.argv),
            "timeout_seconds": self.timeout_seconds,
            "network_policy": self.network_policy,
            "environment_keys": sorted(
                _subprocess_env(self.network_policy == "offline")
            ),
        }


@dataclass(frozen=True)
class PreflightState:
    source_lock_exists: bool
    model_dir_exists: bool


def _absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path.expanduser())))


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _reject_constant(value: str) -> NoReturn:
    raise OnboardError(f"non-finite JSON number is not allowed: {value}")


def _object_without_duplicates(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise OnboardError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _parse_json(payload: bytes, label: str) -> dict[str, object]:
    if len(payload) > MAX_JSON_BYTES:
        raise OnboardError(f"{label} exceeds {MAX_JSON_BYTES} bytes")
    try:
        text = payload.decode("utf-8")
        value = json.loads(
            text,
            object_pairs_hook=_object_without_duplicates,
            parse_constant=_reject_constant,
        )
    except UnicodeDecodeError as error:
        raise OnboardError(f"{label} is not UTF-8") from error
    except json.JSONDecodeError as error:
        raise OnboardError(
            f"{label} is not valid JSON at line {error.lineno} column {error.colno}"
        ) from error
    if type(value) is not dict:
        raise OnboardError(f"{label} must contain one JSON object")
    return value


def _regular_file(path: Path, label: str, *, executable: bool = False) -> None:
    try:
        info = path.lstat()
    except OSError as error:
        raise OnboardError(f"cannot inspect {label}: {error}") from error
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise OnboardError(f"{label} must be a regular non-symlink file")
    if executable and info.st_mode & 0o111 == 0:
        raise OnboardError(f"{label} must have an executable permission bit")


def _directory(path: Path, label: str) -> None:
    try:
        info = path.lstat()
    except OSError as error:
        raise OnboardError(f"cannot inspect {label}: {error}") from error
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise OnboardError(f"{label} must be a non-symlink directory")


def _read_json_file(path: Path, label: str) -> dict[str, object]:
    _regular_file(path, label)
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if before.st_size > MAX_JSON_BYTES:
            raise OnboardError(f"{label} exceeds {MAX_JSON_BYTES} bytes")
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                raise OnboardError(f"{label} ended before its declared size")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise OnboardError(f"{label} changed while it was read")
        after = os.fstat(descriptor)
        if after.st_size != before.st_size or after.st_mtime_ns != before.st_mtime_ns:
            raise OnboardError(f"{label} changed while it was read")
    finally:
        os.close(descriptor)
    return _parse_json(b"".join(chunks), label)


def _hash_regular_file(path: Path, label: str) -> tuple[int, str]:
    _regular_file(path, label)
    before = path.lstat()
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise OnboardError(f"cannot open {label}: {error}") from error
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_dev != before.st_dev
            or opened.st_ino != before.st_ino
        ):
            raise OnboardError(f"{label} changed while it was being opened")
        digest = hashlib.sha256()
        total = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            total += len(chunk)
        after = os.fstat(descriptor)
        if (
            total != opened.st_size
            or after.st_size != opened.st_size
            or after.st_mtime_ns != opened.st_mtime_ns
        ):
            raise OnboardError(f"{label} changed while it was being hashed")
        return total, digest.hexdigest()
    finally:
        os.close(descriptor)


def _profile_artifacts(profile: dict[str, object]) -> list[dict[str, object]]:
    artifacts = profile.get("artifacts")
    if type(artifacts) is not dict or set(artifacts) != REQUIRED_ARTIFACTS:
        raise OnboardError("checked-in deployment artifact map is invalid")
    result: list[dict[str, object]] = []
    for name in sorted(artifacts):
        record = artifacts[name]
        if (
            type(name) is not str
            or "/" in name
            or name in {"", ".", ".."}
            or type(record) is not dict
            or set(record) != {"size", "sha256"}
            or type(record.get("size")) is not int
            or record["size"] <= 0
            or type(record.get("sha256")) is not str
            or not SHA256.fullmatch(record["sha256"])
        ):
            raise OnboardError(f"checked-in deployment artifact is invalid: {name!r}")
        result.append(
            {"path": name, "size": record["size"], "sha256": record["sha256"]}
        )
    if _sha256(_canonical_bytes(artifacts)) != ARTIFACT_MANIFEST_SHA256:
        raise OnboardError("checked-in deployment artifact manifest is not pinned")
    return result


def _artifact_manifest_sha256(profile: dict[str, object]) -> str:
    return _sha256(_canonical_bytes(profile["artifacts"]))


def _output_path(path: Path, label: str) -> None:
    parent = path.parent
    _directory(parent, f"{label} parent")
    if os.path.lexists(path):
        raise OnboardError(f"refusing to overwrite existing {label}: {path}")


def _is_within(path: Path, directory: Path) -> bool:
    try:
        path.relative_to(directory)
        return True
    except ValueError:
        return False


def _load_profile() -> dict[str, object]:
    profile = _read_json_file(PROFILE, "checked-in deployment profile")
    source = profile.get("source")
    binary = profile.get("binary")
    runtime = profile.get("runtime")
    gate = profile.get("gate")
    memory_smoke = profile.get("memory_smoke")
    if (
        set(profile)
        != {
            "format",
            "profile_id",
            "source",
            "artifacts",
            "binary",
            "runtime",
            "gate",
            "memory_smoke",
            "oracle",
        }
        or profile.get("format") != "apxinf-hf-macos-deployment-profile-v1"
        or profile.get("profile_id") != PROFILE_ID
        or type(source) is not dict
        or source.get("repo_id") != REPO_ID
        or source.get("resolved_commit") != REVISION
        or not isinstance(source.get("source_lock_content_sha256"), str)
        or not SHA256.fullmatch(source["source_lock_content_sha256"])
    ):
        raise OnboardError("checked-in Qwen3.5 deployment profile identity is invalid")
    if binary != {
        "size": BINARY_SIZE,
        "sha256": BINARY_SHA256,
        "build": {
            "target_os": "macos",
            "target_arch": "aarch64",
            "matmul_feature": "accelerate",
        },
    }:
        raise OnboardError("checked-in Qwen3.5 deployment binary identity is invalid")
    if runtime != {
        "target": "macos-arm64",
        "provider": "native-apxinf-cpu",
        "device": "cpu",
        "dtype": "fp32",
        "matmul_feature": "accelerate",
    }:
        raise OnboardError("checked-in Qwen3.5 deployment runtime is invalid")
    expected_gate = {
        "generation_receipt_format": GENERATION_FORMAT,
        "max_context": 32,
        "max_tokens": 10,
        "no_eos_stop": True,
        "prompt": "Hello",
        "prompt_token_count": 13,
        "generated_token_ids": [
            9419,
            0,
            2500,
            628,
            353,
            1438,
            488,
            3242,
            30,
            25677,
        ],
    }
    if gate != expected_gate:
        raise OnboardError("checked-in Qwen3.5 deployment gate is invalid")
    if memory_smoke != {
        "max_peak_rss_bytes": 6 * 1024 * 1024 * 1024,
        "max_process_swaps": 0,
        "measurement": "macos-time-l-vm-stat-v1",
        "non_authoritative_evidence": [
            "pageout_delta_bytes",
            "swap_delta_bytes",
            "swap_growth_bytes",
        ],
        "receipt_format": "apxinf-macos-memory-smoke-v1",
        "sandbox": SANDBOX_POLICY,
        "timeout_seconds": 120,
    }:
        raise OnboardError("checked-in Qwen3.5 memory-smoke gate is invalid")
    _profile_artifacts(profile)
    return profile


def _verify_pinned_binary(path: Path, profile: dict[str, object]) -> None:
    size, digest = _hash_regular_file(path, "ApxInf binary")
    expected = profile["binary"]
    if size != expected["size"] or digest != expected["sha256"]:
        raise OnboardError(
            "ApxInf binary does not match the trusted deployment profile"
        )


def _sbpl_string(value: str) -> str:
    if not value or any(ord(character) < 0x20 for character in value):
        raise OnboardError("sandbox path contains an unsupported control character")
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _seatbelt_profile(*, binary_path: Path, model_dir: Path) -> str:
    binary = _sbpl_string(str(_absolute(binary_path)))
    model = _sbpl_string(str(_absolute(model_dir)))
    return "\n".join(
        [
            "(version 1)",
            "(allow default)",
            "(deny network*)",
            "(deny file-write*)",
            '(deny file-read* (subpath "/Users"))',
            f"(allow file-read* (subpath {model}) (literal {binary}))",
        ]
    )


def _subprocess_env(offline: bool) -> dict[str, str]:
    environment = {"PATH": os.defpath, "LANG": "C", "LC_ALL": "C"}
    if offline:
        environment.update(
            {
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
            }
        )
    return environment


def _one_json_line(payload: bytes, label: str) -> dict[str, object]:
    if len(payload) > MAX_CHILD_OUTPUT_BYTES:
        raise OnboardError(f"{label} output exceeds {MAX_CHILD_OUTPUT_BYTES} bytes")
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise OnboardError(f"{label} output is not UTF-8") from error
    lines = text.splitlines()
    if len(lines) != 1 or not lines[0].strip():
        raise OnboardError(f"{label} must emit exactly one JSON line")
    return _parse_json(lines[0].encode("utf-8"), f"{label} output")


def _run_json_stage(stage: Stage) -> dict[str, object]:
    try:
        result = subprocess.run(
            list(stage.argv),
            cwd=ROOT,
            env=_subprocess_env(stage.network_policy == "offline"),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=stage.timeout_seconds,
            check=False,
            shell=False,
        )
    except subprocess.TimeoutExpired as error:
        raise OnboardError(
            f"stage {stage.name} timed out after {stage.timeout_seconds} seconds"
        ) from error
    except OSError as error:
        raise OnboardError(f"cannot start stage {stage.name}: {error}") from error

    stdout = (
        result.stdout if isinstance(result.stdout, bytes) else result.stdout.encode()
    )
    stderr = (
        result.stderr if isinstance(result.stderr, bytes) else result.stderr.encode()
    )
    if len(stdout) > MAX_CHILD_OUTPUT_BYTES or len(stderr) > MAX_CHILD_OUTPUT_BYTES:
        raise OnboardError(f"stage {stage.name} exceeded the output byte limit")
    if result.returncode != 0:
        detail = ""
        if stdout:
            try:
                child = _one_json_line(stdout, stage.name)
                error = child.get("error")
                if isinstance(error, dict):
                    detail = str(error.get("message", ""))
                elif error is not None:
                    detail = str(error)
            except OnboardError:
                detail = ""
        if not detail and stderr:
            detail = stderr.decode("utf-8", errors="replace").strip()
        suffix = f": {detail[:4096]}" if detail else ""
        raise OnboardError(
            f"stage {stage.name} failed with exit code {result.returncode}{suffix}"
        )
    if stderr:
        raise OnboardError(f"stage {stage.name} wrote unexpected stderr output")
    return _one_json_line(stdout, stage.name)


def _validate_source_receipt(
    receipt: dict[str, object], profile: dict[str, object]
) -> None:
    source = profile["source"]
    if (
        receipt.get("passed") is not True
        or receipt.get("format") != "apxinf-hf-source-lock-v1"
        or receipt.get("repo_id") != REPO_ID
        or receipt.get("requested_revision") != REVISION
        or receipt.get("resolved_commit") != REVISION
        or receipt.get("content_sha256") != source["source_lock_content_sha256"]
        or receipt.get("weight_payload_bytes_downloaded") != 0
    ):
        raise OnboardError(
            "source-lock verification receipt does not match the profile"
        )


def _validate_existing_source_lock(
    path: Path, profile: dict[str, object]
) -> dict[str, object]:
    lock = _read_json_file(path, "source lock")
    digest = lock.get("content_sha256")
    if type(digest) is not str or not SHA256.fullmatch(digest):
        raise OnboardError("source lock content_sha256 is invalid")
    body = dict(lock)
    del body["content_sha256"]
    if _sha256(_canonical_bytes(body)) != digest:
        raise OnboardError("source lock content hash mismatch")
    if (
        digest != profile["source"]["source_lock_content_sha256"]
        or lock.get("format") != "apxinf-hf-source-lock-v1"
        or lock.get("repo_id") != REPO_ID
        or lock.get("requested_revision") != REVISION
        or lock.get("resolved_commit") != REVISION
        or lock.get("policy_receipt")
        != {
            "metadata_only": True,
            "weight_payload_bytes_downloaded": 0,
            "remote_code_executed": False,
            "hf_token_read": False,
        }
    ):
        raise OnboardError("existing source lock is not the pinned metadata-only lock")
    return lock


def _validate_oracle_provenance(manifest_path: Path) -> None:
    manifest = _read_json_file(manifest_path, "oracle manifest")
    recorded = manifest.get("model_dir")
    if type(recorded) is not str or not recorded or not Path(recorded).is_absolute():
        raise OnboardError("oracle manifest model_dir provenance is invalid")


def _finite_positive(value: object, label: str) -> float:
    if type(value) not in {int, float}:
        raise OnboardError(f"{label} must be a number")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise OnboardError(f"{label} must be finite and positive")
    return result


def _generation_receipt(
    result: dict[str, object], profile: dict[str, object]
) -> dict[str, object]:
    base_keys = {
        "device",
        "dtype",
        "format",
        "generated_token_ids",
        "model_type",
        "profile",
        "prompt_token_count",
    }
    observed = set(result)
    if observed != base_keys | {"build"}:
        raise OnboardError("generation result contains missing or unknown fields")
    gate = profile["gate"]
    runtime = profile["runtime"]
    if (
        result.get("format") != GENERATION_FORMAT
        or result.get("model_type") != "qwen3_5"
        or result.get("device") != runtime["device"]
        or result.get("dtype") != runtime["dtype"]
        or result.get("prompt_token_count") != gate["prompt_token_count"]
        or result.get("generated_token_ids") != gate["generated_token_ids"]
    ):
        raise OnboardError("generation result does not match the frozen Qwen3.5 gate")
    timings = result.get("profile")
    expected_timing_keys = {
        "generation_tps",
        "input_tokens",
        "output_tokens",
        "total_latency_ms",
        "tpot_ms",
        "ttft_ms",
    }
    if type(timings) is not dict or set(timings) != expected_timing_keys:
        raise OnboardError("generation profile contains missing or unknown fields")
    if (
        timings["input_tokens"] != gate["prompt_token_count"]
        or timings["output_tokens"] != gate["max_tokens"]
    ):
        raise OnboardError("generation profile token counts are stale")
    for name in ("ttft_ms", "tpot_ms", "generation_tps", "total_latency_ms"):
        _finite_positive(timings[name], f"generation profile {name}")
    if float(timings["total_latency_ms"]) < float(timings["ttft_ms"]):
        raise OnboardError("generation total latency is below TTFT")

    expected_build = {
        "target_os": "macos",
        "target_arch": "aarch64",
        "matmul_feature": "accelerate",
    }
    if result["build"] != expected_build:
        raise OnboardError("generation build does not match macOS arm64 Accelerate")
    return dict(result)


def _nonnegative_int(value: object, label: str) -> int:
    if type(value) is not int or value < 0:
        raise OnboardError(f"{label} must be a non-negative integer")
    return value


def _validate_stager_receipt(
    receipt: dict[str, object],
    profile: dict[str, object],
    model_dir: Path,
    *,
    expected_action: str,
) -> dict[str, object]:
    expected_keys = {
        "format",
        "passed",
        "action",
        "profile_id",
        "repo_id",
        "resolved_commit",
        "source_lock_content_sha256",
        "artifact_manifest_sha256",
        "model_dir",
        "artifacts",
        "total_bytes",
        "published",
        "downloaded_bytes",
        "resumed_from_bytes",
        "reused_bytes",
        "policy",
        "evidence",
    }
    if set(receipt) != expected_keys:
        raise OnboardError("bundle stager receipt contains missing or unknown fields")
    expected_artifacts = _profile_artifacts(profile)
    expected_total = sum(record["size"] for record in expected_artifacts)
    expected_manifest = _artifact_manifest_sha256(profile)
    if expected_action not in {"staged", "reused-existing"}:
        raise OnboardError("internal expected stager action is invalid")
    if (
        receipt["format"] != STAGER_FORMAT
        or receipt["passed"] is not True
        or receipt["action"] != expected_action
        or receipt["profile_id"] != PROFILE_ID
        or receipt["repo_id"] != REPO_ID
        or receipt["resolved_commit"] != REVISION
        or receipt["source_lock_content_sha256"]
        != profile["source"]["source_lock_content_sha256"]
        or receipt["artifact_manifest_sha256"] != expected_manifest
        or receipt["model_dir"] != str(model_dir)
        or receipt["artifacts"] != expected_artifacts
        or receipt["total_bytes"] != expected_total
        or receipt["published"] is not True
    ):
        raise OnboardError("bundle stager receipt does not match the pinned bundle")
    downloaded = _nonnegative_int(
        receipt["downloaded_bytes"], "bundle downloaded_bytes"
    )
    resumed = _nonnegative_int(
        receipt["resumed_from_bytes"], "bundle resumed_from_bytes"
    )
    reused = _nonnegative_int(receipt["reused_bytes"], "bundle reused_bytes")
    policy = receipt["policy"]
    expected_policy = {
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
        "operation": {"existing_only_requested": expected_action == "reused-existing"},
        "recovery": {"max_restart_from_zero_per_artifact": 1},
    }
    if policy != expected_policy:
        raise OnboardError("bundle stager policy receipt is invalid")
    evidence = receipt["evidence"]
    evidence_keys = {
        "ambient_proxy_disabled",
        "atomic_no_replace_publish_observed",
        "authorization_header_omitted",
        "builtin_opener",
        "cache_entry_count",
        "cache_total_bytes",
        "cache_tree_present",
        "existing_bundle_verified",
        "existing_only_enforced",
        "lock_acquired",
        "network_request_count",
        "network_used",
        "opener_injected",
        "published_by_this_invocation",
        "recovered_artifacts",
        "recovery_bytes_discarded",
    }
    if type(evidence) is not dict or set(evidence) != evidence_keys:
        raise OnboardError("bundle stager evidence contains missing or unknown fields")
    boolean_fields = {
        "ambient_proxy_disabled",
        "atomic_no_replace_publish_observed",
        "authorization_header_omitted",
        "builtin_opener",
        "cache_tree_present",
        "existing_bundle_verified",
        "existing_only_enforced",
        "lock_acquired",
        "network_used",
        "opener_injected",
        "published_by_this_invocation",
    }
    if any(type(evidence[name]) is not bool for name in boolean_fields):
        raise OnboardError("bundle stager evidence boolean is invalid")
    cache_entries = _nonnegative_int(
        evidence["cache_entry_count"], "bundle cache_entry_count"
    )
    cache_bytes = _nonnegative_int(
        evidence["cache_total_bytes"], "bundle cache_total_bytes"
    )
    request_count = _nonnegative_int(
        evidence["network_request_count"], "bundle network_request_count"
    )
    discarded = _nonnegative_int(
        evidence["recovery_bytes_discarded"], "bundle recovery_bytes_discarded"
    )
    recovered = evidence["recovered_artifacts"]
    artifact_names = {record["path"] for record in expected_artifacts}
    if (
        type(recovered) is not list
        or any(type(name) is not str for name in recovered)
        or recovered != sorted(set(recovered))
        or not set(recovered).issubset(artifact_names)
    ):
        raise OnboardError("bundle stager recovery evidence is invalid")
    if (
        evidence["opener_injected"] is not False
        or evidence["lock_acquired"] is not True
    ):
        raise OnboardError(
            "bundle stager did not use the production lock/opener contract"
        )
    if evidence["cache_tree_present"] is False and (cache_entries or cache_bytes):
        raise OnboardError("bundle stager cache evidence is inconsistent")
    if evidence["network_used"] != (request_count > 0):
        raise OnboardError("bundle stager network evidence is inconsistent")

    if expected_action == "reused-existing":
        if (
            downloaded != 0
            or resumed != 0
            or reused != expected_total
            or evidence["existing_bundle_verified"] is not True
            or evidence["existing_only_enforced"] is not True
            or evidence["network_used"] is not False
            or evidence["builtin_opener"] is not False
            or evidence["ambient_proxy_disabled"] is not False
            or evidence["authorization_header_omitted"] is not False
            or evidence["published_by_this_invocation"] is not False
            or evidence["atomic_no_replace_publish_observed"] is not False
            or recovered != []
            or discarded != 0
        ):
            raise OnboardError("existing bundle stager evidence is inconsistent")
    else:
        accounted = downloaded + resumed + reused
        if (
            resumed + reused > expected_total
            or accounted < expected_total
            or accounted > 2 * expected_total
            or evidence["existing_bundle_verified"] is not False
            or evidence["existing_only_enforced"] is not False
            or evidence["cache_tree_present"] is not False
            or cache_entries != 0
            or cache_bytes != 0
            or evidence["published_by_this_invocation"] is not True
            or evidence["atomic_no_replace_publish_observed"] is not True
        ):
            raise OnboardError(
                "staged bundle evidence or byte accounting is inconsistent"
            )
        if downloaded > 0:
            if (
                evidence["network_used"] is not True
                or evidence["builtin_opener"] is not True
                or evidence["ambient_proxy_disabled"] is not True
                or evidence["authorization_header_omitted"] is not True
            ):
                raise OnboardError("staged bundle network evidence is incomplete")
        elif (
            evidence["network_used"] is not False
            or evidence["builtin_opener"] is not False
            or evidence["ambient_proxy_disabled"] is not False
            or evidence["authorization_header_omitted"] is not False
        ):
            raise OnboardError("network-free staged bundle evidence is inconsistent")
    return {
        "format": receipt["format"],
        "action": receipt["action"],
        "canonical_sha256": _sha256(_canonical_bytes(receipt)),
        "profile_id": receipt["profile_id"],
        "repo_id": receipt["repo_id"],
        "resolved_commit": receipt["resolved_commit"],
        "source_lock_content_sha256": receipt["source_lock_content_sha256"],
        "model_dir": receipt["model_dir"],
        "artifact_manifest_sha256": expected_manifest,
        "total_bytes": expected_total,
        "published": receipt["published"],
        "policy_sha256": _sha256(_canonical_bytes(policy)),
        "evidence": dict(evidence),
        "downloaded_bytes": downloaded,
        "resumed_from_bytes": resumed,
        "reused_bytes": reused,
    }


def _bundle_summary(
    profile: dict[str, object],
    model_dir: Path,
    *,
    disposition: str,
    stager: dict[str, object],
) -> dict[str, object]:
    if disposition not in {"staged", "reused"}:
        raise OnboardError("internal bundle disposition is invalid")
    artifacts = _profile_artifacts(profile)
    downloaded = stager["downloaded_bytes"]
    resumed = stager["resumed_from_bytes"]
    reused = stager["reused_bytes"]
    return {
        "disposition": disposition,
        "model_dir": str(model_dir),
        "artifact_manifest_sha256": _artifact_manifest_sha256(profile),
        "artifacts": artifacts,
        "total_bytes": sum(record["size"] for record in artifacts),
        "cache_tree_present": stager["evidence"]["cache_tree_present"],
        "bytes": {
            "downloaded": downloaded,
            "resumed": resumed,
            "reused": reused,
            "recovery_discarded": stager["evidence"]["recovery_bytes_discarded"],
        },
        "stager_receipt": stager,
    }


def _exclusive_json_write(path: Path, value: object) -> None:
    _output_path(path, "generation receipt")
    payload = (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8")
        + b"\n"
    )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        offset = 0
        while offset < len(payload):
            offset += os.write(descriptor, payload[offset:])
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        try:
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError as error:
            raise OnboardError(
                f"refusing to overwrite existing generation receipt: {path}"
            ) from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


def _validate_lock(
    path: Path, verifier_receipt: dict[str, object]
) -> dict[str, object]:
    lock = _read_json_file(path, "deployment lock")
    digest = lock.get("content_sha256")
    if lock.get("format") != DEPLOYMENT_FORMAT or not isinstance(digest, str):
        raise OnboardError("deployment verifier produced an invalid lock")
    if not SHA256.fullmatch(digest):
        raise OnboardError("deployment lock content hash is invalid")
    body = dict(lock)
    del body["content_sha256"]
    if _sha256(_canonical_bytes(body)) != digest:
        raise OnboardError("deployment lock content hash mismatch")
    memory = verifier_receipt.get("memory_smoke")
    lock_memory = lock.get("memory_smoke")
    if (
        verifier_receipt.get("format") != DEPLOYMENT_RECEIPT_FORMAT
        or verifier_receipt.get("passed") is not True
        or verifier_receipt.get("profile_id") != PROFILE_ID
        or verifier_receipt.get("repo_id") != REPO_ID
        or verifier_receipt.get("resolved_commit") != REVISION
        or verifier_receipt.get("deployment_lock_sha256") != digest
        or verifier_receipt.get("output") != str(path)
        or type(memory) is not dict
        or memory.get("present") is not True
        or memory.get("origin") != "live"
        or not isinstance(memory.get("content_sha256"), str)
        or not SHA256.fullmatch(memory["content_sha256"])
        or type(lock_memory) is not dict
        or lock_memory.get("origin") != "live"
        or lock_memory.get("content_sha256") != memory["content_sha256"]
    ):
        raise OnboardError("deployment verifier receipt does not match its lock")
    return lock


def _preflight(
    args: argparse.Namespace,
) -> tuple[dict[str, object], dict[str, Path], PreflightState]:
    if args.model_url != MODEL_URL:
        raise OnboardError(f"only the canonical URL {MODEL_URL} is supported")
    if args.revision != REVISION:
        raise OnboardError(f"revision must be the pinned commit {REVISION}")

    paths = {
        "source_lock": _absolute(args.source_lock),
        "model_dir": _absolute(args.model_dir),
        "oracle_dir": _absolute(args.oracle_dir),
        "binary": _absolute(args.binary),
        "receipt_output": _absolute(args.receipt_output),
        "lock_output": _absolute(args.lock_output),
    }
    paths["oracle_manifest"] = paths["oracle_dir"] / "manifest.json"
    paths["oracle_metrics"] = paths["oracle_dir"] / "apxinf-metrics.json"

    _regular_file(RESOLVER, "source resolver")
    _regular_file(STAGER, "bundle stager")
    _regular_file(DEPLOYMENT_VERIFIER, "deployment verifier")
    _directory(paths["oracle_dir"], "oracle directory")
    _regular_file(paths["oracle_manifest"], "oracle manifest")
    _regular_file(paths["oracle_metrics"], "oracle metrics")
    _validate_oracle_provenance(paths["oracle_manifest"])
    _regular_file(paths["binary"], "ApxInf binary", executable=True)
    _regular_file(SANDBOX_EXEC, "macOS sandbox tool", executable=True)

    profile = _load_profile()
    source_lock_exists = os.path.lexists(paths["source_lock"])
    if source_lock_exists:
        _regular_file(paths["source_lock"], "source lock")
        _validate_existing_source_lock(paths["source_lock"], profile)
    elif args.offline:
        raise OnboardError("offline onboarding requires an existing source lock")
    else:
        _output_path(paths["source_lock"], "source lock")

    model_dir_exists = os.path.lexists(paths["model_dir"])
    if model_dir_exists:
        _directory(paths["model_dir"], "model directory")
    else:
        _directory(paths["model_dir"].parent, "model directory parent")
        if args.offline:
            raise OnboardError(
                "offline onboarding cannot stage a missing model directory"
            )
        if not args.download_missing:
            raise OnboardError(
                "model directory is missing; pass --download-missing to authorize "
                "the pinned bundle download"
            )

    _output_path(paths["receipt_output"], "generation receipt")
    _output_path(paths["lock_output"], "deployment lock")

    mutable_paths = {
        paths["source_lock"],
        paths["receipt_output"],
        paths["lock_output"],
    }
    if len(mutable_paths) != 3:
        raise OnboardError(
            "source lock, receipt, and deployment lock paths must be distinct"
        )
    for output in (
        paths["source_lock"],
        paths["receipt_output"],
        paths["lock_output"],
    ):
        if _is_within(output, paths["model_dir"]) or _is_within(
            output, paths["oracle_dir"]
        ):
            raise OnboardError("outputs must be outside model and oracle directories")
    stager_work = paths["model_dir"].parent / (
        f".{paths['model_dir'].name}.apxinf-staging"
    )
    stager_lock = paths["model_dir"].parent / (
        f".{paths['model_dir'].name}.apxinf-stage.lock"
    )
    for output in mutable_paths:
        if output == stager_lock or _is_within(output, stager_work):
            raise OnboardError("outputs must not overlap the bundle stager work paths")
    _verify_pinned_binary(paths["binary"], profile)
    return (
        profile,
        paths,
        PreflightState(
            source_lock_exists=source_lock_exists,
            model_dir_exists=model_dir_exists,
        ),
    )


def _stages(
    args: argparse.Namespace,
    profile: dict[str, object],
    paths: dict[str, Path],
    state: PreflightState,
) -> list[Stage]:
    python = str(_absolute(Path(sys.executable)))
    stages: list[Stage] = []
    if not args.offline and not state.source_lock_exists:
        stages.append(
            Stage(
                "resolve_source_lock",
                (
                    python,
                    str(RESOLVER),
                    MODEL_URL,
                    "--revision",
                    REVISION,
                    "--output",
                    str(paths["source_lock"]),
                ),
                SOURCE_RESOLVE_TIMEOUT,
                "metadata-only-https-huggingface.co",
            )
        )
    stages.append(
        Stage(
            "verify_source_lock",
            (
                python,
                str(RESOLVER),
                "--verify",
                str(paths["source_lock"]),
                "--expected-sha256",
                profile["source"]["source_lock_content_sha256"],
            ),
            SOURCE_VERIFY_TIMEOUT,
            "offline",
        )
    )
    stages.append(
        Stage(
            "ensure_model_bundle",
            (
                python,
                str(STAGER),
                "--profile",
                str(PROFILE),
                "--source-lock",
                str(paths["source_lock"]),
                "--model-dir",
                str(paths["model_dir"]),
                "--timeout-seconds",
                str(BUNDLE_STAGE_TIMEOUT),
                "--max-total-bytes",
                str(STAGER_MAX_TOTAL_BYTES),
                *(("--existing-only",) if state.model_dir_exists else ()),
            ),
            BUNDLE_STAGE_TIMEOUT,
            (
                "offline"
                if state.model_dir_exists
                else "public-model-payload-https-hugging-face-owned-domains"
            ),
        )
    )
    gate = profile["gate"]
    runtime = profile["runtime"]
    generation_argv = [
        str(SANDBOX_EXEC),
        "-p",
        _seatbelt_profile(binary_path=paths["binary"], model_dir=paths["model_dir"]),
        str(paths["binary"]),
        "generate",
        "--model",
        str(paths["model_dir"]),
        "--prompt",
        gate["prompt"],
        "--max-tokens",
        str(gate["max_tokens"]),
        "--max-context",
        str(gate["max_context"]),
    ]
    if gate["no_eos_stop"]:
        generation_argv.append("--no-eos-stop")
    generation_argv.extend(
        [
            "--device",
            runtime["device"],
            "--dtype",
            runtime["dtype"],
            "--json",
        ]
    )
    stages.append(
        Stage(
            "run_generation_gate",
            tuple(generation_argv),
            GENERATION_TIMEOUT,
            "offline",
        )
    )
    stages.append(
        Stage(
            "verify_and_publish_deployment",
            (
                python,
                str(DEPLOYMENT_VERIFIER),
                "--profile",
                str(PROFILE),
                "--source-lock",
                str(paths["source_lock"]),
                "--model-dir",
                str(paths["model_dir"]),
                "--oracle-manifest",
                str(paths["oracle_manifest"]),
                "--oracle-metrics",
                str(paths["oracle_metrics"]),
                "--generation-receipt",
                str(paths["receipt_output"]),
                "--binary",
                str(paths["binary"]),
                "--measure-smoke",
                "--output",
                str(paths["lock_output"]),
            ),
            DEPLOYMENT_VERIFY_TIMEOUT,
            "offline",
        )
    )
    return stages


def execute(args: argparse.Namespace) -> dict[str, object]:
    profile, paths, state = _preflight(args)
    stages = _stages(args, profile, paths, state)
    plan: list[dict[str, object]] = []
    for stage in stages:
        plan.append(stage.plan())
        if stage.name == "run_generation_gate":
            plan.append(
                {
                    "name": "publish_generation_receipt",
                    "operation": "exclusive-atomic-json-write",
                    "path": str(paths["receipt_output"]),
                    "overwrite": False,
                }
            )
    expected_artifacts = _profile_artifacts(profile)
    expected_total = sum(record["size"] for record in expected_artifacts)
    bundle_disposition = "reused" if state.model_dir_exists else "staged"
    planned_bundle = {
        "disposition": bundle_disposition,
        "model_dir": str(paths["model_dir"]),
        "artifact_manifest_sha256": _artifact_manifest_sha256(profile),
        "artifacts": expected_artifacts,
        "total_bytes": expected_total,
        "bytes": {
            "maximum_download": 0 if state.model_dir_exists else expected_total,
            "reused": expected_total if state.model_dir_exists else None,
            "resumed": 0 if state.model_dir_exists else None,
            "recovery_discarded": 0 if state.model_dir_exists else None,
        },
        "stager_receipt": {
            "format": STAGER_FORMAT,
            "expected_action": (
                "reused-existing" if state.model_dir_exists else "staged"
            ),
            "profile_id": PROFILE_ID,
            "repo_id": REPO_ID,
            "resolved_commit": REVISION,
            "source_lock_content_sha256": profile["source"][
                "source_lock_content_sha256"
            ],
            "artifact_manifest_sha256": _artifact_manifest_sha256(profile),
            "model_dir": str(paths["model_dir"]),
            "total_bytes": expected_total,
            "existing_only": state.model_dir_exists,
            "script": str(STAGER),
        },
    }
    if args.dry_run:
        return {
            "format": PLAN_FORMAT,
            "passed": True,
            "dry_run": True,
            "model_url": MODEL_URL,
            "repo_id": REPO_ID,
            "revision": REVISION,
            "profile_id": PROFILE_ID,
            "starts_agent": False,
            "source_lock": {
                "path": str(paths["source_lock"]),
                "disposition": ("reused" if state.source_lock_exists else "created"),
                "content_sha256": profile["source"]["source_lock_content_sha256"],
            },
            "bundle": planned_bundle,
            "stages": plan,
        }

    stage_by_name = {stage.name: stage for stage in stages}
    if "resolve_source_lock" in stage_by_name:
        resolution = _run_json_stage(stage_by_name["resolve_source_lock"])
        _validate_source_receipt(resolution, profile)
    source_receipt = _run_json_stage(stage_by_name["verify_source_lock"])
    _validate_source_receipt(source_receipt, profile)
    _validate_existing_source_lock(paths["source_lock"], profile)

    staged = _run_json_stage(stage_by_name["ensure_model_bundle"])
    stager_identity = _validate_stager_receipt(
        staged,
        profile,
        paths["model_dir"],
        expected_action=("reused-existing" if state.model_dir_exists else "staged"),
    )
    bundle = _bundle_summary(
        profile,
        paths["model_dir"],
        disposition=bundle_disposition,
        stager=stager_identity,
    )

    generation = _run_json_stage(stage_by_name["run_generation_gate"])
    _verify_pinned_binary(paths["binary"], profile)
    generation_receipt = _generation_receipt(generation, profile)
    _exclusive_json_write(paths["receipt_output"], generation_receipt)
    deployment_receipt = _run_json_stage(stage_by_name["verify_and_publish_deployment"])
    lock = _validate_lock(paths["lock_output"], deployment_receipt)
    return {
        "format": CONTROLLER_FORMAT,
        "passed": True,
        "dry_run": False,
        "model_url": MODEL_URL,
        "repo_id": REPO_ID,
        "revision": REVISION,
        "profile_id": PROFILE_ID,
        "starts_agent": False,
        "source_lock": {
            "path": str(paths["source_lock"]),
            "disposition": "reused" if state.source_lock_exists else "created",
            "content_sha256": source_receipt["content_sha256"],
        },
        "bundle": bundle,
        "generation_receipt": {
            "path": str(paths["receipt_output"]),
            "generated_token_ids": generation_receipt["generated_token_ids"],
        },
        "deployment_lock": {
            "path": str(paths["lock_output"]),
            "content_sha256": lock["content_sha256"],
        },
    }


def parser() -> argparse.ArgumentParser:
    result = JsonArgumentParser(description=__doc__)
    result.add_argument("model_url", help=f"must be exactly {MODEL_URL}")
    result.add_argument("--revision", required=True)
    result.add_argument("--source-lock", required=True, type=Path)
    result.add_argument("--model-dir", required=True, type=Path)
    result.add_argument("--oracle-dir", required=True, type=Path)
    result.add_argument("--binary", required=True, type=Path)
    result.add_argument("--receipt-output", required=True, type=Path)
    result.add_argument("--lock-output", required=True, type=Path)
    result.add_argument(
        "--offline",
        action="store_true",
        help="reuse and verify the existing source lock without network access",
    )
    result.add_argument(
        "--download-missing",
        action="store_true",
        help="authorize staging the pinned public bundle when --model-dir is absent",
    )
    result.add_argument(
        "--dry-run",
        action="store_true",
        help="validate paths and emit the exact stage plan without execution",
    )
    return result


def _emit(stream: TextIO, value: object) -> None:
    stream.write(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    )
    stream.flush()


def main(argv: list[str] | None = None) -> int:
    try:
        args = parser().parse_args(argv)
        receipt = execute(args)
        _emit(sys.stdout, receipt)
        return 0
    except (OnboardError, OSError, UnicodeError, json.JSONDecodeError) as error:
        _emit(
            sys.stderr,
            {
                "format": CONTROLLER_FORMAT,
                "passed": False,
                "error": {
                    "code": "HF_MACOS_ONBOARD_FAILED",
                    "message": str(error) or type(error).__name__,
                },
            },
        )
        return 2
    except Exception as error:
        _emit(
            sys.stderr,
            {
                "format": CONTROLLER_FORMAT,
                "passed": False,
                "error": {
                    "code": "HF_MACOS_ONBOARD_INTERNAL_ERROR",
                    "message": str(error) or type(error).__name__,
                },
            },
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
