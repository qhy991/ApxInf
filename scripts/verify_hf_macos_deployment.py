#!/usr/bin/env python3
"""Fail-closed, offline deployment gate for pinned Hugging Face snapshots.

Normal validation never imports model code or launches the candidate binary.
The explicit ``--measure-smoke`` production gate runs one fixed, offline macOS
command under system memory accounting before atomically publishing a lock.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import stat
import struct
import subprocess
import sys
import tempfile
from typing import NoReturn


PROFILE_FORMAT = "apxinf-hf-macos-deployment-profile-v1"
SOURCE_LOCK_FORMAT = "apxinf-hf-source-lock-v1"
DEPLOYMENT_LOCK_FORMAT = "apxinf-deployment-lock-v1"
RECEIPT_FORMAT = "apxinf-deployment-verification-receipt-v1"
MEMORY_RECEIPT_FORMAT = "apxinf-macos-memory-smoke-v1"
MAX_JSON_BYTES = 16 * 1024 * 1024
MAX_CACHE_ENTRIES = 100_000
SHA256 = re.compile(r"^[0-9a-f]{64}$")
SHA1 = re.compile(r"^[0-9a-f]{40}$")
REPO_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}/[A-Za-z0-9][A-Za-z0-9._-]{0,95}$")
PROFILE_ID = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")
REQUIRED_ARTIFACTS = frozenset(
    {
        "config.json",
        "model.safetensors.index.json",
        "model.safetensors-00001-of-00001.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
    }
)
SCRIPT_SUFFIXES = frozenset(
    {
        ".py",
        ".pyc",
        ".pyo",
        ".pyw",
        ".sh",
        ".bash",
        ".zsh",
        ".fish",
        ".command",
        ".pl",
        ".rb",
        ".js",
        ".mjs",
        ".cjs",
    }
)
CPU_TYPE_ARM64 = 0x0100000C


class DeploymentError(ValueError):
    """A deterministic, user-correctable deployment validation failure."""


class JsonArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        raise DeploymentError(f"invalid arguments: {message}")


def canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _reject_constant(value: str) -> NoReturn:
    raise DeploymentError(f"non-finite JSON number is not allowed: {value}")


def _object_without_duplicate_keys(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise DeploymentError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _open_regular_nofollow(path: Path, label: str) -> tuple[int, os.stat_result]:
    path = _absolute(path)
    try:
        before = path.lstat()
    except OSError as error:
        raise DeploymentError(
            f"cannot inspect {label}: {error.strerror or error}"
        ) from error
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise DeploymentError(f"{label} must be a regular non-symlink file")
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise DeploymentError(f"cannot open {label}: {error.strerror or error}") from error
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_dev != before.st_dev
            or opened.st_ino != before.st_ino
        ):
            raise DeploymentError(f"{label} changed while it was being opened")
        return descriptor, opened
    except Exception:
        os.close(descriptor)
        raise


def _read_regular(path: Path, label: str, maximum: int) -> tuple[bytes, os.stat_result]:
    descriptor, opened = _open_regular_nofollow(path, label)
    try:
        if opened.st_size > maximum:
            raise DeploymentError(f"{label} exceeds {maximum} bytes")
        chunks: list[bytes] = []
        remaining = opened.st_size
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                raise DeploymentError(f"{label} ended before its declared size")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise DeploymentError(f"{label} changed while it was being read")
        after = os.fstat(descriptor)
        if after.st_size != opened.st_size or after.st_mtime_ns != opened.st_mtime_ns:
            raise DeploymentError(f"{label} changed while it was being read")
        return b"".join(chunks), opened
    finally:
        os.close(descriptor)


def _hash_regular(
    path: Path, label: str, *, expected_size: int | None = None
) -> tuple[int, str, os.stat_result]:
    descriptor, opened = _open_regular_nofollow(path, label)
    try:
        if expected_size is not None and opened.st_size != expected_size:
            raise DeploymentError(
                f"{label} size mismatch: expected {expected_size}, observed {opened.st_size}"
            )
        digest = hashlib.sha256()
        total = 0
        while True:
            chunk = os.read(descriptor, 8 * 1024 * 1024)
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
            raise DeploymentError(f"{label} changed while it was being hashed")
        return total, digest.hexdigest(), opened
    finally:
        os.close(descriptor)


def _parse_json(payload: bytes, label: str) -> dict[str, object]:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise DeploymentError(f"{label} is not valid UTF-8") from error
    try:
        value = json.loads(
            text,
            object_pairs_hook=_object_without_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except json.JSONDecodeError as error:
        raise DeploymentError(
            f"invalid {label} JSON at line {error.lineno} column {error.colno}"
        ) from error
    if type(value) is not dict:
        raise DeploymentError(f"{label} root must be a JSON object")
    return value


def _read_json(path: Path, label: str) -> tuple[dict[str, object], int, str]:
    payload, info = _read_regular(path, label, MAX_JSON_BYTES)
    return _parse_json(payload, label), info.st_size, sha256_bytes(payload)


def _exact_keys(value: object, expected: set[str], label: str) -> dict[str, object]:
    if type(value) is not dict:
        raise DeploymentError(f"{label} must be an object")
    observed = set(value)
    if observed != expected:
        missing = sorted(expected - observed)
        unknown = sorted(observed - expected)
        raise DeploymentError(
            f"{label} keys mismatch (missing={missing}, unknown={unknown})"
        )
    return value


def _string(value: object, label: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise DeploymentError(f"{label} must be a non-empty string without outer whitespace")
    return value


def _sha256(value: object, label: str) -> str:
    result = _string(value, label)
    if not SHA256.fullmatch(result):
        raise DeploymentError(f"{label} must be a lowercase SHA-256")
    return result


def _positive_int(value: object, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise DeploymentError(f"{label} must be a positive integer")
    return value


def _is_exact_token_ids(value: object, expected: list[int]) -> bool:
    return (
        type(value) is list
        and len(value) == len(expected)
        and all(type(token) is int and token >= 0 for token in value)
        and value == expected
    )


def validate_profile(profile: object) -> dict[str, object]:
    root = _exact_keys(
        profile,
        {
            "format",
            "profile_id",
            "source",
            "artifacts",
            "binary",
            "runtime",
            "gate",
            "memory_smoke",
            "oracle",
        },
        "profile",
    )
    if root["format"] != PROFILE_FORMAT:
        raise DeploymentError(f"profile format must be {PROFILE_FORMAT}")
    profile_id = _string(root["profile_id"], "profile.profile_id")
    if not PROFILE_ID.fullmatch(profile_id):
        raise DeploymentError("profile.profile_id is not canonical")

    source = _exact_keys(
        root["source"],
        {
            "repo_id",
            "resolved_commit",
            "license",
            "source_lock_content_sha256",
            "config_sha256",
        },
        "profile.source",
    )
    repo_id = _string(source["repo_id"], "profile.source.repo_id")
    if not REPO_ID.fullmatch(repo_id):
        raise DeploymentError("profile.source.repo_id must be owner/model")
    commit = _string(source["resolved_commit"], "profile.source.resolved_commit")
    if not SHA1.fullmatch(commit):
        raise DeploymentError("profile.source.resolved_commit must be a lowercase commit")
    if source["license"] != "Apache-2.0":
        raise DeploymentError("profile.source.license must be Apache-2.0")
    _sha256(source["source_lock_content_sha256"], "profile.source.source_lock_content_sha256")
    config_sha = _sha256(source["config_sha256"], "profile.source.config_sha256")

    artifacts = root["artifacts"]
    if type(artifacts) is not dict or set(artifacts) != REQUIRED_ARTIFACTS:
        raise DeploymentError(
            "profile.artifacts must contain exactly the pinned Qwen deployment files"
        )
    for name, record_value in artifacts.items():
        record = _exact_keys(record_value, {"size", "sha256"}, f"profile.artifacts.{name}")
        _positive_int(record["size"], f"profile.artifacts.{name}.size")
        _sha256(record["sha256"], f"profile.artifacts.{name}.sha256")
    if artifacts["config.json"]["sha256"] != config_sha:
        raise DeploymentError("profile source config hash does not match config.json")

    binary = _exact_keys(root["binary"], {"size", "sha256", "build"}, "profile.binary")
    _positive_int(binary["size"], "profile.binary.size")
    _sha256(binary["sha256"], "profile.binary.sha256")
    binary_build = _exact_keys(
        binary["build"],
        {"target_os", "target_arch", "matmul_feature"},
        "profile.binary.build",
    )
    if binary_build != {
        "target_os": "macos",
        "target_arch": "aarch64",
        "matmul_feature": "accelerate",
    }:
        raise DeploymentError("profile binary is not the trusted macOS arm64 Accelerate build")

    runtime = _exact_keys(
        root["runtime"],
        {"target", "provider", "device", "dtype", "matmul_feature"},
        "profile.runtime",
    )
    required_runtime = {
        "target": "macos-arm64",
        "provider": "native-apxinf-cpu",
        "device": "cpu",
        "dtype": "fp32",
        "matmul_feature": "accelerate",
    }
    if runtime != required_runtime:
        raise DeploymentError(f"profile.runtime must equal {required_runtime}")

    gate = _exact_keys(
        root["gate"],
        {
            "generation_receipt_format",
            "max_context",
            "max_tokens",
            "no_eos_stop",
            "prompt",
            "prompt_token_count",
            "generated_token_ids",
        },
        "profile.gate",
    )
    if (
        gate["generation_receipt_format"] != "apxinf-generation-v1"
        or gate["max_context"] != 32
        or gate["max_tokens"] != 10
        or gate["no_eos_stop"] is not True
        or gate["prompt"] != "Hello"
        or gate["prompt_token_count"] != 13
    ):
        raise DeploymentError("profile gate must lock context=32, tokens=10, and no_eos_stop=true")
    ids = gate["generated_token_ids"]
    if (
        type(ids) is not list
        or len(ids) != 10
        or any(type(token) is not int or token < 0 for token in ids)
    ):
        raise DeploymentError("profile.gate.generated_token_ids must contain 10 token IDs")

    memory_smoke = _exact_keys(
        root["memory_smoke"],
        {
            "receipt_format",
            "measurement",
            "max_peak_rss_bytes",
            "max_process_swaps",
            "non_authoritative_evidence",
            "sandbox",
            "timeout_seconds",
        },
        "profile.memory_smoke",
    )
    if (
        memory_smoke["receipt_format"] != MEMORY_RECEIPT_FORMAT
        or memory_smoke["measurement"] != "macos-time-l-vm-stat-v1"
        or memory_smoke["sandbox"]
        != "macos-seatbelt-deny-network-write-home-read-v1"
    ):
        raise DeploymentError("profile memory smoke format or measurement is unsupported")
    _positive_int(memory_smoke["max_peak_rss_bytes"], "profile memory peak RSS limit")
    if type(memory_smoke["max_process_swaps"]) is not int or memory_smoke["max_process_swaps"] < 0:
        raise DeploymentError("profile.memory_smoke.max_process_swaps must be non-negative")
    if memory_smoke["non_authoritative_evidence"] != [
        "pageout_delta_bytes",
        "swap_delta_bytes",
        "swap_growth_bytes",
    ]:
        raise DeploymentError("profile must mark global pageout and swap deltas as evidence-only")
    _positive_int(memory_smoke["timeout_seconds"], "profile memory smoke timeout")

    oracle = _exact_keys(
        root["oracle"],
        {
            "manifest_format",
            "manifest_sha256",
            "metrics_format",
            "metrics_sha256",
            "gate_format",
            "runtime",
        },
        "profile.oracle",
    )
    for key in ("manifest_format", "metrics_format", "gate_format"):
        _string(oracle[key], f"profile.oracle.{key}")
    _sha256(oracle["manifest_sha256"], "profile.oracle.manifest_sha256")
    _sha256(oracle["metrics_sha256"], "profile.oracle.metrics_sha256")
    oracle_runtime = _exact_keys(
        oracle["runtime"],
        {
            "torch",
            "transformers",
            "safetensors",
            "device",
            "dtype",
            "attention_implementation",
            "use_hub_kernels",
            "optional_kernel_packages",
            "threads",
            "deterministic_algorithms",
        },
        "profile.oracle.runtime",
    )
    for key in ("torch", "transformers", "safetensors"):
        _string(oracle_runtime[key], f"profile.oracle.runtime.{key}")
    if (
        oracle_runtime["device"] != "cpu"
        or oracle_runtime["dtype"] != "float32"
        or oracle_runtime["attention_implementation"] != "eager"
        or oracle_runtime["use_hub_kernels"] is not False
        or oracle_runtime["optional_kernel_packages"] != []
        or type(oracle_runtime["threads"]) is not int
        or oracle_runtime["threads"] != 1
        or oracle_runtime["deterministic_algorithms"] is not True
    ):
        raise DeploymentError("profile.oracle.runtime is not the locked deterministic CPU runtime")
    return root


def _validate_source_lock(source_lock: dict[str, object], profile: dict[str, object]) -> None:
    source_profile = profile["source"]
    if source_lock.get("format") != SOURCE_LOCK_FORMAT:
        raise DeploymentError(f"source lock format must be {SOURCE_LOCK_FORMAT}")
    content_hash = _sha256(source_lock.get("content_sha256"), "source_lock.content_sha256")
    body = dict(source_lock)
    del body["content_sha256"]
    if sha256_bytes(canonical_bytes(body)) != content_hash:
        raise DeploymentError("source lock content hash mismatch")
    if content_hash != source_profile["source_lock_content_sha256"]:
        raise DeploymentError("source lock is not bound to the deployment profile")
    if source_lock.get("repo_id") != source_profile["repo_id"]:
        raise DeploymentError("source lock repo_id does not match the profile")
    if source_lock.get("resolved_commit") != source_profile["resolved_commit"]:
        raise DeploymentError("source lock resolved_commit does not match the profile")
    if source_lock.get("requested_revision") != source_profile["resolved_commit"]:
        raise DeploymentError("source lock requested_revision is not the pinned commit")

    source = source_lock.get("source")
    if type(source) is not dict:
        raise DeploymentError("source lock source section is invalid")
    if source.get("license", "").casefold() != source_profile["license"].casefold():
        raise DeploymentError("source lock license does not match the profile")
    if source.get("private") is not False or source.get("gated") is not False or source.get("disabled") is not False:
        raise DeploymentError("source lock is not an unrestricted public source")
    if source.get("url") != f"https://huggingface.co/{source_profile['repo_id']}":
        raise DeploymentError("source lock URL does not match the profile")
    if source_lock.get("policy_receipt") != {
        "metadata_only": True,
        "weight_payload_bytes_downloaded": 0,
        "remote_code_executed": False,
        "hf_token_read": False,
    }:
        raise DeploymentError("source lock policy receipt is not metadata-only")
    security = source_lock.get("security")
    if type(security) is not dict:
        raise DeploymentError("source lock security section is invalid")
    indicators = security.get("remote_code_indicators")
    if (
        security.get("safetensors_only_plan") is not True
        or security.get("unsafe_weight_files") != []
        or type(indicators) is not dict
        or indicators.get("python_files") != []
        or indicators.get("auto_map_keys") != []
    ):
        raise DeploymentError("source lock contains an unsafe execution or weight indicator")
    architecture = source_lock.get("architecture")
    if type(architecture) is not dict or architecture.get("config_sha256") != source_profile["config_sha256"]:
        raise DeploymentError("source lock config hash does not match the profile")

    artifacts = profile["artifacts"]
    metadata = source_lock.get("metadata")
    if type(metadata) is not dict or type(metadata.get("files")) is not list:
        raise DeploymentError("source lock metadata file list is invalid")
    metadata_by_name: dict[str, dict[str, object]] = {}
    for value in metadata["files"]:
        if type(value) is not dict or type(value.get("path")) is not str:
            raise DeploymentError("source lock metadata record is invalid")
        name = value["path"]
        if name in metadata_by_name:
            raise DeploymentError(f"duplicate source lock metadata record: {name}")
        metadata_by_name[name] = value
    for name in ("config.json", "model.safetensors.index.json", "tokenizer_config.json"):
        record = metadata_by_name.get(name)
        expected = artifacts[name]
        if record is None or record.get("size") != expected["size"] or record.get("sha256") != expected["sha256"]:
            raise DeploymentError(f"source lock metadata does not bind {name}")

    weights = source_lock.get("weights")
    if type(weights) is not dict or weights.get("format") != "safetensors":
        raise DeploymentError("source lock weights are not SafeTensors")
    if weights.get("index_file") != "model.safetensors.index.json":
        raise DeploymentError("source lock weight index is not the pinned index")
    expected_weight_names = sorted(name for name in artifacts if name.endswith(".safetensors"))
    records = weights.get("files")
    if type(records) is not list:
        raise DeploymentError("source lock weight file list is invalid")
    weight_by_name: dict[str, dict[str, object]] = {}
    for value in records:
        if type(value) is not dict or type(value.get("path")) is not str:
            raise DeploymentError("source lock weight record is invalid")
        name = value["path"]
        if name in weight_by_name:
            raise DeploymentError(f"duplicate source lock weight record: {name}")
        weight_by_name[name] = value
    if sorted(weight_by_name) != expected_weight_names:
        raise DeploymentError("source lock weight files do not match the profile")
    total = 0
    for name in expected_weight_names:
        expected = artifacts[name]
        record = weight_by_name[name]
        if record.get("size") != expected["size"] or record.get("sha256") != expected["sha256"]:
            raise DeploymentError(f"source lock weight does not bind {name}")
        total += expected["size"]
    if weights.get("total_bytes") != total:
        raise DeploymentError("source lock total weight size is inconsistent")


def _cache_file_is_script(path: Path, mode: int) -> bool:
    if path.suffix.casefold() in SCRIPT_SUFFIXES or mode & 0o111:
        return True
    descriptor, _ = _open_regular_nofollow(path, f"cache file {path}")
    try:
        return os.read(descriptor, 2) == b"#!"
    finally:
        os.close(descriptor)


def _validate_cache_tree(cache: Path) -> None:
    try:
        root_info = cache.lstat()
    except OSError as error:
        raise DeploymentError(f"cannot inspect model .cache: {error.strerror or error}") from error
    if stat.S_ISLNK(root_info.st_mode) or not stat.S_ISDIR(root_info.st_mode):
        raise DeploymentError("model .cache must be a non-symlink directory")
    pending = [cache]
    count = 0
    while pending:
        directory = pending.pop()
        try:
            entries = list(os.scandir(directory))
        except OSError as error:
            raise DeploymentError(f"cannot inspect model cache: {error.strerror or error}") from error
        for entry in entries:
            count += 1
            if count > MAX_CACHE_ENTRIES:
                raise DeploymentError("model .cache exceeds the safety entry limit")
            path = Path(entry.path)
            try:
                info = entry.stat(follow_symlinks=False)
            except OSError as error:
                raise DeploymentError(f"cannot inspect cache entry: {error.strerror or error}") from error
            if stat.S_ISLNK(info.st_mode):
                raise DeploymentError(f"model .cache contains a symlink: {path.relative_to(cache)}")
            if stat.S_ISDIR(info.st_mode):
                pending.append(path)
            elif stat.S_ISREG(info.st_mode):
                if _cache_file_is_script(path, info.st_mode):
                    raise DeploymentError(f"model .cache contains a script: {path.relative_to(cache)}")
            else:
                raise DeploymentError(f"model .cache contains a non-regular entry: {path.relative_to(cache)}")


def _validate_model_dir(model_dir: Path, profile: dict[str, object]) -> dict[str, object]:
    model_dir = _absolute(model_dir)
    try:
        root_info = model_dir.lstat()
    except OSError as error:
        raise DeploymentError(f"cannot inspect model directory: {error.strerror or error}") from error
    if stat.S_ISLNK(root_info.st_mode) or not stat.S_ISDIR(root_info.st_mode):
        raise DeploymentError("model directory must be a non-symlink directory")
    try:
        entries = list(os.scandir(model_dir))
    except OSError as error:
        raise DeploymentError(f"cannot list model directory: {error.strerror or error}") from error
    allowed = set(profile["artifacts"]) | {".cache"}
    observed = {entry.name for entry in entries}
    unexpected = sorted(observed - allowed)
    missing = sorted(set(profile["artifacts"]) - observed)
    if unexpected or missing:
        raise DeploymentError(
            f"model top-level allowlist mismatch (missing={missing}, unexpected={unexpected})"
        )
    if ".cache" in observed:
        _validate_cache_tree(model_dir / ".cache")

    verified: dict[str, object] = {}
    for name in sorted(profile["artifacts"]):
        expected = profile["artifacts"][name]
        size, digest, _ = _hash_regular(
            model_dir / name, f"model artifact {name}", expected_size=expected["size"]
        )
        if digest != expected["sha256"]:
            raise DeploymentError(f"model artifact {name} SHA-256 mismatch")
        verified[name] = {"size": size, "sha256": digest}
    return verified


def _checkpoint_record(profile: dict[str, object]) -> tuple[str, dict[str, object]]:
    matches = [(name, value) for name, value in profile["artifacts"].items() if name.endswith(".safetensors")]
    if len(matches) != 1:
        raise DeploymentError("profile must contain exactly one SafeTensors checkpoint")
    return matches[0]


def _validate_oracle_manifest(
    manifest: dict[str, object], manifest_sha: str, profile: dict[str, object]
) -> None:
    oracle_profile = profile["oracle"]
    source_profile = profile["source"]
    gate = profile["gate"]
    checkpoint_name, checkpoint = _checkpoint_record(profile)
    if manifest_sha != oracle_profile["manifest_sha256"]:
        raise DeploymentError("oracle manifest SHA-256 does not match the profile")
    if manifest.get("format") != oracle_profile["manifest_format"]:
        raise DeploymentError("oracle manifest format does not match the profile")
    if manifest.get("repo_id") != source_profile["repo_id"]:
        raise DeploymentError("oracle manifest repo_id does not match the profile")
    if manifest.get("revision") != source_profile["resolved_commit"]:
        raise DeploymentError("oracle manifest revision does not match the profile")
    if manifest.get("checkpoint_sha256") != checkpoint["sha256"]:
        raise DeploymentError("oracle manifest checkpoint does not match the profile")
    recorded_model_dir = manifest.get("model_dir")
    if (
        type(recorded_model_dir) is not str
        or not recorded_model_dir
        or not Path(recorded_model_dir).is_absolute()
    ):
        raise DeploymentError("oracle manifest model_dir provenance is invalid")
    # The v1 path records where the oracle was generated; it is not model identity.
    # The profile and snapshot hashes bind the bytes deployed at any absolute location.
    runtime = manifest.get("runtime")
    if type(runtime) is not dict:
        raise DeploymentError("oracle manifest runtime is invalid")
    for key, expected in oracle_profile["runtime"].items():
        if runtime.get(key) != expected:
            raise DeploymentError(f"oracle manifest runtime mismatch for {key}")
    trajectory = manifest.get("greedy_trajectory")
    if type(trajectory) is not dict:
        raise DeploymentError("oracle manifest greedy trajectory is invalid")
    expected_ids = gate["generated_token_ids"]
    if (
        not _is_exact_token_ids(trajectory.get("generated_ids"), expected_ids)
        or trajectory.get("length") != gate["max_tokens"]
        or trajectory.get("minimum_length") != gate["max_tokens"]
        or trajectory.get("do_sample") is not False
        or trajectory.get("use_cache") is not True
        or trajectory.get("eos_stopping") is not False
    ):
        raise DeploymentError("oracle manifest greedy trajectory does not match the profile")
    if manifest.get("uses_locked_default_ids") is not True:
        raise DeploymentError("oracle manifest did not use locked default IDs")
    locked_chat = manifest.get("locked_chat")
    if type(locked_chat) is not dict or locked_chat.get("enable_thinking") is not False:
        raise DeploymentError("oracle manifest is not the locked non-thinking chat gate")
    snapshot = manifest.get("snapshot")
    if type(snapshot) is not dict or snapshot.get("checkpoint_sha256") != checkpoint["sha256"]:
        raise DeploymentError("oracle snapshot checkpoint does not match the profile")
    files = snapshot.get("verified_files")
    if type(files) is not dict or set(files) != set(profile["artifacts"]):
        raise DeploymentError("oracle snapshot file set does not match the profile")
    for name, expected in profile["artifacts"].items():
        record = files.get(name)
        if type(record) is not dict or record.get("size") != expected["size"]:
            raise DeploymentError(f"oracle snapshot size mismatch for {name}")
        if "sha256" in record and record["sha256"] != expected["sha256"]:
            raise DeploymentError(f"oracle snapshot hash mismatch for {name}")
    if files[checkpoint_name].get("sha256") != checkpoint["sha256"]:
        raise DeploymentError("oracle snapshot checkpoint hash is missing or stale")


def _validate_metrics(
    metrics: dict[str, object], metrics_sha: str, profile: dict[str, object]
) -> None:
    oracle_profile = profile["oracle"]
    gate = profile["gate"]
    _, checkpoint = _checkpoint_record(profile)
    if metrics_sha != oracle_profile["metrics_sha256"]:
        raise DeploymentError("oracle metrics SHA-256 does not match the profile")
    if metrics.get("format") != oracle_profile["metrics_format"]:
        raise DeploymentError("oracle metrics format does not match the profile")
    verification = metrics.get("verification")
    if type(verification) is not dict:
        raise DeploymentError("oracle metrics verification is invalid")
    checks = verification.get("checks")
    if (
        verification.get("passed") is not True
        or verification.get("status") != "pass"
        or verification.get("failures") != []
        or type(checks) is not list
        or not checks
        or any(type(check) is not dict or check.get("passed") is not True for check in checks)
    ):
        raise DeploymentError("oracle metrics verification did not pass every frozen check")
    gate_manifest = verification.get("manifest")
    if (
        type(gate_manifest) is not dict
        or gate_manifest.get("format") != oracle_profile["gate_format"]
        or gate_manifest.get("frozen") is not True
        or gate_manifest.get("threshold_overrides_supported") is not False
    ):
        raise DeploymentError("oracle metrics did not use the frozen gate")
    calibration = gate_manifest.get("calibration")
    if (
        type(calibration) is not dict
        or calibration.get("checkpoint_sha256") != checkpoint["sha256"]
        or calibration.get("comparison_format") != oracle_profile["metrics_format"]
        or calibration.get("device") != profile["runtime"]["device"]
        or calibration.get("dtype") != oracle_profile["runtime"]["dtype"]
        or calibration.get("matmul_feature") != profile["runtime"]["matmul_feature"]
    ):
        raise DeploymentError("oracle metrics calibration is stale")

    expected_ids = gate["generated_token_ids"]
    trajectory = metrics.get("greedy_trajectory")
    if (
        type(trajectory) is not dict
        or not _is_exact_token_ids(trajectory.get("apxinf_ids"), expected_ids)
        or not _is_exact_token_ids(trajectory.get("expected_ids"), expected_ids)
        or trajectory.get("length") != gate["max_tokens"]
        or trajectory.get("minimum_length") != gate["max_tokens"]
        or trajectory.get("exact_match") is not True
        or trajectory.get("eos_stopping") is not False
    ):
        raise DeploymentError("oracle metrics greedy trajectory does not match the profile")
    greedy_checks = [check for check in checks if check.get("name") == "greedy_trajectory"]
    if len(greedy_checks) != 1:
        raise DeploymentError("oracle metrics must contain exactly one greedy trajectory check")
    greedy_check = greedy_checks[0]
    expected = greedy_check.get("expected")
    observed = greedy_check.get("observed")
    if (
        type(expected) is not dict
        or type(observed) is not dict
        or not _is_exact_token_ids(expected.get("generated_ids"), expected_ids)
        or expected.get("length") != gate["max_tokens"]
        or not _is_exact_token_ids(observed.get("generated_ids"), expected_ids)
        or observed.get("length") != gate["max_tokens"]
    ):
        raise DeploymentError("oracle metrics greedy check contains stale token IDs")
    apxinf = metrics.get("apxinf")
    if (
        type(apxinf) is not dict
        or apxinf.get("device") != profile["runtime"]["device"]
        or apxinf.get("matmul_feature") != profile["runtime"]["matmul_feature"]
        or apxinf.get("max_context") != gate["max_context"]
    ):
        raise DeploymentError("oracle metrics runtime does not match the deployment profile")


def _finite_number(value: object, label: str, *, positive: bool = False) -> float:
    if type(value) not in {int, float}:
        raise DeploymentError(f"{label} must be a finite number")
    result = float(value)
    if not (result == result and abs(result) != float("inf")):
        raise DeploymentError(f"{label} must be a finite number")
    if (positive and result <= 0) or (not positive and result < 0):
        qualifier = "positive" if positive else "non-negative"
        raise DeploymentError(f"{label} must be {qualifier}")
    return result


def _validate_generation_receipt(
    receipt: dict[str, object], profile: dict[str, object]
) -> None:
    root = _exact_keys(
        receipt,
        {
            "build",
            "device",
            "dtype",
            "format",
            "generated_token_ids",
            "model_type",
            "profile",
            "prompt_token_count",
        },
        "generation receipt",
    )
    gate = profile["gate"]
    runtime = profile["runtime"]
    if root["format"] != gate["generation_receipt_format"]:
        raise DeploymentError("generation receipt format does not match the profile")
    if root["model_type"] != "qwen3_5":
        raise DeploymentError("generation receipt model_type is not qwen3_5")
    if root["device"] != runtime["device"] or root["dtype"] != runtime["dtype"]:
        raise DeploymentError("generation receipt device or dtype does not match the profile")
    if root["prompt_token_count"] != gate["prompt_token_count"]:
        raise DeploymentError("generation receipt prompt token count does not match the profile")
    if not _is_exact_token_ids(root["generated_token_ids"], gate["generated_token_ids"]):
        raise DeploymentError("generation receipt token trajectory does not match the profile")

    build = _exact_keys(
        root["build"],
        {"target_os", "target_arch", "matmul_feature"},
        "generation receipt build",
    )
    if build != profile["binary"]["build"]:
        raise DeploymentError("generation receipt is not the pinned macOS arm64 Accelerate build")

    timings = _exact_keys(
        root["profile"],
        {
            "generation_tps",
            "input_tokens",
            "output_tokens",
            "total_latency_ms",
            "tpot_ms",
            "ttft_ms",
        },
        "generation receipt profile",
    )
    if timings["input_tokens"] != gate["prompt_token_count"]:
        raise DeploymentError("generation receipt profile input token count is stale")
    if timings["output_tokens"] != gate["max_tokens"]:
        raise DeploymentError("generation receipt profile output token count is stale")
    ttft = _finite_number(timings["ttft_ms"], "generation receipt TTFT", positive=True)
    _finite_number(timings["tpot_ms"], "generation receipt TPOT", positive=True)
    _finite_number(
        timings["generation_tps"], "generation receipt generation throughput", positive=True
    )
    total = _finite_number(
        timings["total_latency_ms"], "generation receipt total latency", positive=True
    )
    if total < ttft:
        raise DeploymentError("generation receipt total latency is below TTFT")


def _smoke_fixed_args(model_dir: Path, profile: dict[str, object]) -> list[str]:
    return [
        "generate",
        "--model",
        str(_absolute(model_dir)),
        "--prompt",
        profile["gate"]["prompt"],
        "--max-tokens",
        str(profile["gate"]["max_tokens"]),
        "--max-context",
        str(profile["gate"]["max_context"]),
        "--no-eos-stop",
        "--device",
        profile["runtime"]["device"],
        "--dtype",
        profile["runtime"]["dtype"],
        "--json",
    ]


def _non_negative_int(value: object, label: str) -> int:
    if type(value) is not int or value < 0:
        raise DeploymentError(f"{label} must be a non-negative integer")
    return value


def _signed_int(value: object, label: str) -> int:
    if type(value) is not int:
        raise DeploymentError(f"{label} must be an integer")
    return value


def _validate_memory_receipt(
    receipt: dict[str, object],
    profile: dict[str, object],
    *,
    binary_path: Path,
    binary_sha: str,
    model_dir: Path,
    generation_receipt_sha: str,
    require_live: bool,
) -> None:
    root = _exact_keys(
        receipt,
        {
            "format",
            "content_sha256",
            "measurement",
            "binary",
            "model",
            "argv",
            "generation",
            "result",
        },
        "memory receipt",
    )
    memory_profile = profile["memory_smoke"]
    if root["format"] != memory_profile["receipt_format"]:
        raise DeploymentError("memory receipt format does not match the profile")
    content_sha = _sha256(root["content_sha256"], "memory receipt content_sha256")
    body = dict(root)
    del body["content_sha256"]
    if sha256_bytes(canonical_bytes(body)) != content_sha:
        raise DeploymentError("memory receipt content hash mismatch")

    measurement = _exact_keys(
        root["measurement"],
        {
            "source",
            "platform",
            "tool",
            "mode",
            "sandbox",
            "sandbox_profile_sha256",
        },
        "memory measurement",
    )
    if (
        measurement["source"] not in {"live", "fixture"}
        or measurement["platform"] != "macos"
        or measurement["tool"] != "/usr/bin/time"
        or measurement["mode"] != "-l"
        or measurement["sandbox"] != "/usr/bin/sandbox-exec"
    ):
        raise DeploymentError("memory receipt measurement identity is invalid")
    expected_sandbox_sha = sha256_bytes(
        _seatbelt_profile(binary_path=binary_path, model_dir=model_dir).encode("utf-8")
    )
    if measurement["sandbox_profile_sha256"] != expected_sandbox_sha:
        raise DeploymentError("memory receipt sandbox policy does not match the fixed policy")
    if require_live and measurement["source"] != "live":
        raise DeploymentError("production deployment requires a live macOS memory measurement")

    binary = _exact_keys(root["binary"], {"path", "sha256"}, "memory receipt binary")
    if (
        binary["path"] != str(_absolute(binary_path))
        or binary["sha256"] != binary_sha
    ):
        raise DeploymentError("memory receipt is bound to a different binary")
    checkpoint_name, checkpoint = _checkpoint_record(profile)
    model = _exact_keys(
        root["model"],
        {"directory", "checkpoint", "checkpoint_sha256"},
        "memory receipt model",
    )
    if (
        model["directory"] != str(_absolute(model_dir))
        or model["checkpoint"] != checkpoint_name
        or model["checkpoint_sha256"] != checkpoint["sha256"]
    ):
        raise DeploymentError("memory receipt is bound to a different model snapshot")
    expected_argv = [str(_absolute(binary_path)), *_smoke_fixed_args(model_dir, profile)]
    if (
        type(root["argv"]) is not list
        or any(type(argument) is not str for argument in root["argv"])
        or root["argv"] != expected_argv
    ):
        raise DeploymentError("memory receipt argv does not match the fixed smoke command")

    generation = _exact_keys(
        root["generation"],
        {"input_receipt_sha256", "stdout_receipt_sha256", "stdout_receipt"},
        "memory receipt generation binding",
    )
    if generation["input_receipt_sha256"] != generation_receipt_sha:
        raise DeploymentError("memory receipt is bound to a different generation receipt")
    stdout_receipt = generation["stdout_receipt"]
    if type(stdout_receipt) is not dict:
        raise DeploymentError("memory receipt stdout receipt must be an object")
    stdout_sha = _sha256(
        generation["stdout_receipt_sha256"],
        "memory receipt stdout receipt SHA-256",
    )
    if sha256_bytes(canonical_bytes(stdout_receipt)) != stdout_sha:
        raise DeploymentError("memory receipt stdout receipt hash mismatch")
    _validate_generation_receipt(stdout_receipt, profile)

    result = _exact_keys(
        root["result"],
        {
            "exit_code",
            "peak_rss_bytes",
            "process_swaps",
            "page_size_bytes",
            "pageouts_before",
            "pageouts_after",
            "pageout_delta",
            "pageout_delta_bytes",
            "swap_used_before_bytes",
            "swap_used_after_bytes",
            "swap_delta_bytes",
            "swap_growth_bytes",
        },
        "memory receipt result",
    )
    exit_code = _signed_int(result["exit_code"], "memory smoke exit code")
    peak_rss = _positive_int(result["peak_rss_bytes"], "memory smoke peak RSS")
    process_swaps = _non_negative_int(result["process_swaps"], "memory smoke process swaps")
    page_size = _positive_int(result["page_size_bytes"], "memory smoke page size")
    pageouts_before = _non_negative_int(result["pageouts_before"], "memory smoke pageouts before")
    pageouts_after = _non_negative_int(result["pageouts_after"], "memory smoke pageouts after")
    pageout_delta = _non_negative_int(result["pageout_delta"], "memory smoke pageout delta")
    pageout_delta_bytes = _non_negative_int(
        result["pageout_delta_bytes"], "memory smoke pageout byte delta"
    )
    swap_before = _non_negative_int(
        result["swap_used_before_bytes"], "memory smoke swap before"
    )
    swap_after = _non_negative_int(
        result["swap_used_after_bytes"], "memory smoke swap after"
    )
    swap_delta = _signed_int(result["swap_delta_bytes"], "memory smoke swap delta")
    swap_growth = _non_negative_int(result["swap_growth_bytes"], "memory smoke swap growth")
    if exit_code != 0:
        raise DeploymentError("memory smoke command did not exit successfully")
    if pageouts_after < pageouts_before or pageout_delta != pageouts_after - pageouts_before:
        raise DeploymentError("memory receipt pageout counters are inconsistent")
    if pageout_delta_bytes != pageout_delta * page_size:
        raise DeploymentError("memory receipt pageout byte delta is inconsistent")
    if swap_delta != swap_after - swap_before or swap_growth != max(0, swap_delta):
        raise DeploymentError("memory receipt swap counters are inconsistent")
    if peak_rss > memory_profile["max_peak_rss_bytes"]:
        raise DeploymentError("memory smoke peak RSS exceeds the profile limit")
    if process_swaps > memory_profile["max_process_swaps"]:
        raise DeploymentError("memory smoke process swaps exceed the profile limit")
    # vm_stat and vm.swapusage are host-wide counters.  Preserve them as
    # self-consistent evidence, but do not attribute concurrent host activity
    # to this child; /usr/bin/time's per-child `swaps` remains the hard gate.


def _require_system_tool(path: Path) -> None:
    descriptor, info = _open_regular_nofollow(path, f"system tool {path}")
    os.close(descriptor)
    if info.st_mode & 0o111 == 0:
        raise DeploymentError(f"system tool is not executable: {path}")


def _run_system_tool(command: list[str], *, timeout: int) -> subprocess.CompletedProcess[bytes]:
    try:
        return subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=timeout,
            env={"PATH": "/usr/bin:/bin:/usr/sbin:/sbin", "LANG": "C", "LC_ALL": "C"},
        )
    except subprocess.TimeoutExpired as error:
        raise DeploymentError(f"system measurement tool timed out: {command[0]}") from error


def _parse_vm_stat(payload: bytes) -> tuple[int, int]:
    try:
        text = payload.decode("ascii")
    except UnicodeDecodeError as error:
        raise DeploymentError("vm_stat output is not ASCII") from error
    page_size_match = re.search(r"page size of ([0-9]+) bytes", text)
    pageouts_match = re.search(r"^Pageouts:\s+([0-9]+)\.\s*$", text, re.MULTILINE)
    if page_size_match is None or pageouts_match is None:
        raise DeploymentError("vm_stat output is missing page size or Pageouts")
    page_size = int(page_size_match.group(1))
    pageouts = int(pageouts_match.group(1))
    if page_size <= 0:
        raise DeploymentError("vm_stat reported an invalid page size")
    return page_size, pageouts


def _parse_swap_used(payload: bytes) -> int:
    try:
        text = payload.decode("ascii")
    except UnicodeDecodeError as error:
        raise DeploymentError("sysctl swapusage output is not ASCII") from error
    match = re.search(r"\bused\s*=\s*([0-9]+(?:\.[0-9]+)?)([KMGTP])\b", text)
    if match is None:
        raise DeploymentError("sysctl swapusage output is invalid")
    multiplier = 1024 ** {"K": 1, "M": 2, "G": 3, "T": 4, "P": 5}[match.group(2)]
    return int(round(float(match.group(1)) * multiplier))


def _mac_memory_snapshot(timeout: int) -> tuple[int, int, int]:
    vm_stat_path = Path("/usr/bin/vm_stat")
    sysctl_path = Path("/usr/sbin/sysctl")
    _require_system_tool(vm_stat_path)
    _require_system_tool(sysctl_path)
    vm_stat = _run_system_tool([str(vm_stat_path)], timeout=timeout)
    swap = _run_system_tool([str(sysctl_path), "-n", "vm.swapusage"], timeout=timeout)
    if vm_stat.returncode != 0 or swap.returncode != 0:
        raise DeploymentError("macOS memory counter tool failed")
    page_size, pageouts = _parse_vm_stat(vm_stat.stdout)
    return page_size, pageouts, _parse_swap_used(swap.stdout)


def _parse_time_l(payload: bytes) -> tuple[int, int]:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise DeploymentError("/usr/bin/time output is not UTF-8") from error
    peak_matches = re.findall(r"^\s*([0-9]+)\s+maximum resident set size\s*$", text, re.MULTILINE)
    swap_matches = re.findall(r"^\s*([0-9]+)\s+swaps\s*$", text, re.MULTILINE)
    if len(peak_matches) != 1 or len(swap_matches) != 1:
        raise DeploymentError("/usr/bin/time -l output is missing unique RSS or swap counters")
    return int(peak_matches[0]), int(swap_matches[0])


def _parse_generation_stdout(payload: bytes) -> dict[str, object]:
    if not payload or len(payload) > MAX_JSON_BYTES:
        raise DeploymentError("memory smoke stdout receipt is empty or too large")
    lines = payload.splitlines()
    if len(lines) != 1 or not lines[0]:
        raise DeploymentError("memory smoke stdout must contain exactly one JSON receipt")
    return _parse_json(lines[0], "memory smoke stdout receipt")


def _sbpl_string(value: str) -> str:
    if not value or any(ord(character) < 0x20 for character in value):
        raise DeploymentError("sandbox path contains an unsupported control character")
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


def _run_memory_command(
    command: list[str], *, timeout: int, environment: dict[str, str]
) -> subprocess.CompletedProcess[bytes]:
    process = subprocess.Popen(
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=environment,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired as error:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except OSError:
            process.kill()
        process.communicate()
        raise DeploymentError("memory smoke command timed out and was terminated") from error
    return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)


def _measure_memory_smoke(
    *,
    binary_path: Path,
    binary_sha: str,
    model_dir: Path,
    profile: dict[str, object],
    generation_receipt_sha: str,
) -> dict[str, object]:
    if sys.platform != "darwin":
        raise DeploymentError("--measure-smoke requires macOS")
    time_path = Path("/usr/bin/time")
    sandbox_path = Path("/usr/bin/sandbox-exec")
    _require_system_tool(time_path)
    _require_system_tool(sandbox_path)
    timeout = profile["memory_smoke"]["timeout_seconds"]
    page_size_before, pageouts_before, swap_before = _mac_memory_snapshot(timeout)
    argv = [str(_absolute(binary_path)), *_smoke_fixed_args(model_dir, profile)]
    environment = {
        "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
        "LANG": "C",
        "LC_ALL": "C",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "NO_PROXY": "*",
        "no_proxy": "*",
    }
    sandbox_profile = _seatbelt_profile(binary_path=binary_path, model_dir=model_dir)
    measured = _run_memory_command(
        [str(sandbox_path), "-p", sandbox_profile, str(time_path), "-l", *argv],
        timeout=timeout,
        environment=environment,
    )
    page_size_after, pageouts_after, swap_after = _mac_memory_snapshot(timeout)
    if page_size_after != page_size_before:
        raise DeploymentError("macOS VM page size changed during measurement")
    peak_rss, process_swaps = _parse_time_l(measured.stderr)
    stdout_receipt = _parse_generation_stdout(measured.stdout)
    swap_delta = swap_after - swap_before
    pageout_delta = pageouts_after - pageouts_before
    if pageout_delta < 0:
        raise DeploymentError("macOS Pageouts counter decreased during measurement")
    checkpoint_name, checkpoint = _checkpoint_record(profile)
    body: dict[str, object] = {
        "format": profile["memory_smoke"]["receipt_format"],
        "measurement": {
            "source": "live",
            "platform": "macos",
            "tool": "/usr/bin/time",
            "mode": "-l",
            "sandbox": "/usr/bin/sandbox-exec",
            "sandbox_profile_sha256": sha256_bytes(sandbox_profile.encode("utf-8")),
        },
        "binary": {"path": str(_absolute(binary_path)), "sha256": binary_sha},
        "model": {
            "directory": str(_absolute(model_dir)),
            "checkpoint": checkpoint_name,
            "checkpoint_sha256": checkpoint["sha256"],
        },
        "argv": argv,
        "generation": {
            "input_receipt_sha256": generation_receipt_sha,
            "stdout_receipt_sha256": sha256_bytes(canonical_bytes(stdout_receipt)),
            "stdout_receipt": stdout_receipt,
        },
        "result": {
            "exit_code": measured.returncode,
            "peak_rss_bytes": peak_rss,
            "process_swaps": process_swaps,
            "page_size_bytes": page_size_before,
            "pageouts_before": pageouts_before,
            "pageouts_after": pageouts_after,
            "pageout_delta": pageout_delta,
            "pageout_delta_bytes": pageout_delta * page_size_before,
            "swap_used_before_bytes": swap_before,
            "swap_used_after_bytes": swap_after,
            "swap_delta_bytes": swap_delta,
            "swap_growth_bytes": max(0, swap_delta),
        },
    }
    body["content_sha256"] = sha256_bytes(canonical_bytes(body))
    return body


def _read_at(descriptor: int, size: int, offset: int) -> bytes:
    if hasattr(os, "pread"):
        return os.pread(descriptor, size, offset)
    current = os.lseek(descriptor, 0, os.SEEK_CUR)
    try:
        os.lseek(descriptor, offset, os.SEEK_SET)
        return os.read(descriptor, size)
    finally:
        os.lseek(descriptor, current, os.SEEK_SET)


def _thin_macho_header(header: bytes, label: str) -> tuple[str, int, int]:
    if len(header) < 32:
        raise DeploymentError(f"{label} has a truncated Mach-O 64-bit header")
    magic = header[:4]
    if magic == b"\xcf\xfa\xed\xfe":
        byte_order = "<"
    elif magic == b"\xfe\xed\xfa\xcf":
        byte_order = ">"
    else:
        raise DeploymentError(f"{label} is not a Mach-O 64-bit image")
    cpu_type = struct.unpack_from(f"{byte_order}I", header, 4)[0]
    file_type = struct.unpack_from(f"{byte_order}I", header, 12)[0]
    return byte_order, cpu_type, file_type


def _parse_macho(descriptor: int, size: int) -> dict[str, object]:
    header = _read_at(descriptor, min(size, 4096), 0)
    if len(header) < 8:
        raise DeploymentError("binary is too short to contain a Mach-O header")
    magic = header[:4]
    if magic in {b"\xcf\xfa\xed\xfe", b"\xfe\xed\xfa\xcf"}:
        _, cpu_type, file_type = _thin_macho_header(header, "binary")
        if cpu_type != CPU_TYPE_ARM64:
            raise DeploymentError("binary Mach-O architecture is not arm64")
        if file_type != 2:
            raise DeploymentError("binary Mach-O file type is not executable")
        return {"container": "thin", "architectures": ["arm64"], "bits": 64}

    fat_layouts = {
        b"\xca\xfe\xba\xbe": (">", False),
        b"\xbe\xba\xfe\xca": ("<", False),
        b"\xca\xfe\xba\xbf": (">", True),
        b"\xbf\xba\xfe\xca": ("<", True),
    }
    if magic not in fat_layouts:
        raise DeploymentError("binary is not a Mach-O 64-bit arm64 executable")
    byte_order, is_64 = fat_layouts[magic]
    count = struct.unpack_from(f"{byte_order}I", header, 4)[0]
    if count == 0 or count > 128:
        raise DeploymentError("binary has an invalid Mach-O universal architecture count")
    record_size = 32 if is_64 else 20
    needed = 8 + count * record_size
    if needed > len(header):
        header = _read_at(descriptor, needed, 0)
    if len(header) < needed:
        raise DeploymentError("binary has a truncated Mach-O universal header")
    architectures: list[str] = []
    for index in range(count):
        offset = 8 + index * record_size
        cpu_type = struct.unpack_from(f"{byte_order}I", header, offset)[0]
        if is_64:
            slice_offset, slice_size = struct.unpack_from(f"{byte_order}QQ", header, offset + 8)
        else:
            slice_offset, slice_size = struct.unpack_from(f"{byte_order}II", header, offset + 8)
        if slice_size == 0 or slice_offset > size or slice_size > size - slice_offset:
            raise DeploymentError("binary has an out-of-bounds Mach-O universal slice")
        if cpu_type == CPU_TYPE_ARM64:
            slice_header = _read_at(descriptor, 32, slice_offset)
            _, slice_cpu_type, slice_file_type = _thin_macho_header(
                slice_header, "binary arm64 universal slice"
            )
            if slice_cpu_type != CPU_TYPE_ARM64 or slice_file_type != 2:
                raise DeploymentError("binary arm64 universal slice is not an executable arm64 image")
            architectures.append("arm64")
        else:
            architectures.append(f"cpu-{cpu_type:#x}")
    if "arm64" not in architectures:
        raise DeploymentError("binary Mach-O universal image does not contain arm64")
    return {"container": "universal", "architectures": architectures, "bits": 64}


def _validate_binary(binary: Path) -> dict[str, object]:
    descriptor, opened = _open_regular_nofollow(binary, "binary")
    try:
        if opened.st_mode & 0o111 == 0:
            raise DeploymentError("binary must have an executable permission bit")
        macho = _parse_macho(descriptor, opened.st_size)
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
            raise DeploymentError("binary changed while it was being verified")
        return {"size": total, "sha256": digest.hexdigest(), "macho": macho}
    finally:
        os.close(descriptor)


def _validate_profile_binary(binary: dict[str, object], profile: dict[str, object]) -> None:
    expected = profile["binary"]
    if binary["size"] != expected["size"] or binary["sha256"] != expected["sha256"]:
        raise DeploymentError("binary size or SHA-256 does not match the trusted profile")


def _build_lock(
    *,
    profile_path: Path,
    profile_size: int,
    profile_file_sha: str,
    profile: dict[str, object],
    source_lock_path: Path,
    source_lock_size: int,
    source_lock_file_sha: str,
    model_dir: Path,
    artifacts: dict[str, object],
    oracle_manifest_path: Path,
    oracle_manifest_size: int,
    oracle_manifest_sha: str,
    oracle_metrics_path: Path,
    oracle_metrics_size: int,
    oracle_metrics_sha: str,
    generation_receipt_path: Path,
    generation_receipt_size: int,
    generation_receipt_sha: str,
    generation_receipt: dict[str, object],
    memory_receipt_path: Path | None,
    memory_receipt_size: int | None,
    memory_receipt_file_sha: str | None,
    memory_receipt: dict[str, object] | None,
    memory_receipt_live: bool,
    binary_path: Path,
    binary: dict[str, object],
) -> dict[str, object]:
    model_dir = _absolute(model_dir)
    binary_path = _absolute(binary_path)
    body: dict[str, object] = {
        "format": DEPLOYMENT_LOCK_FORMAT,
        "profile": {
            "id": profile["profile_id"],
            "path": str(_absolute(profile_path)),
            "size": profile_size,
            "sha256": profile_file_sha,
        },
        "source": {
            "repo_id": profile["source"]["repo_id"],
            "resolved_commit": profile["source"]["resolved_commit"],
            "license": profile["source"]["license"],
            "source_lock": {
                "path": str(_absolute(source_lock_path)),
                "size": source_lock_size,
                "file_sha256": source_lock_file_sha,
                "content_sha256": profile["source"]["source_lock_content_sha256"],
            },
        },
        "model": {"directory": str(model_dir), "artifacts": artifacts},
        "oracle": {
            "manifest": {
                "path": str(_absolute(oracle_manifest_path)),
                "size": oracle_manifest_size,
                "sha256": oracle_manifest_sha,
            },
            "metrics": {
                "path": str(_absolute(oracle_metrics_path)),
                "size": oracle_metrics_size,
                "sha256": oracle_metrics_sha,
            },
            "generated_token_ids": profile["gate"]["generated_token_ids"],
        },
        "runtime": dict(profile["runtime"]),
        "gate": dict(profile["gate"]),
        "smoke": {
            "receipt": {
                "path": str(_absolute(generation_receipt_path)),
                "size": generation_receipt_size,
                "sha256": generation_receipt_sha,
            },
            "result": generation_receipt,
            "fixed_args": _smoke_fixed_args(model_dir, profile),
        },
        "memory_smoke": None,
        "binary": {"path": str(binary_path), **binary},
        "launch": {
            "program": str(binary_path),
            "fixed_args": [
                "generate",
                "--model",
                str(model_dir),
                "--device",
                profile["runtime"]["device"],
                "--dtype",
                profile["runtime"]["dtype"],
                "--max-context",
                str(profile["gate"]["max_context"]),
            ],
        },
    }
    if memory_receipt is not None:
        body["memory_smoke"] = {
            "origin": "live" if memory_receipt_live else "file",
            "file": (
                {
                    "path": str(_absolute(memory_receipt_path)),
                    "size": memory_receipt_size,
                    "sha256": memory_receipt_file_sha,
                }
                if memory_receipt_path is not None
                else None
            ),
            "content_sha256": memory_receipt["content_sha256"],
            "receipt": memory_receipt,
        }
    body["content_sha256"] = sha256_bytes(canonical_bytes(body))
    return body


def _exclusive_atomic_json(path: Path, value: object) -> None:
    path = _absolute(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    parent_info = path.parent.lstat()
    if stat.S_ISLNK(parent_info.st_mode) or not stat.S_ISDIR(parent_info.st_mode):
        raise DeploymentError("output parent must be a non-symlink directory")
    if os.path.lexists(path):
        raise DeploymentError("output already exists")
    payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    installed = False
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
            raise DeploymentError("output already exists") from error
        installed = True
        temporary.unlink()
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except Exception:
        if installed:
            try:
                path.unlink()
            except OSError:
                pass
        raise
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def verify(args: argparse.Namespace) -> tuple[dict[str, object], dict[str, object]]:
    profile, profile_size, profile_file_sha = _read_json(args.profile, "profile")
    profile = validate_profile(profile)
    if args.output is not None and not args.measure_smoke:
        raise DeploymentError("production --output requires a live --measure-smoke run")
    binary = _validate_binary(args.binary)
    _validate_profile_binary(binary, profile)
    source_lock, source_size, source_file_sha = _read_json(args.source_lock, "source lock")
    _validate_source_lock(source_lock, profile)
    artifacts = _validate_model_dir(args.model_dir, profile)
    manifest, manifest_size, manifest_sha = _read_json(args.oracle_manifest, "oracle manifest")
    _validate_oracle_manifest(manifest, manifest_sha, profile)
    metrics, metrics_size, metrics_sha = _read_json(args.oracle_metrics, "oracle metrics")
    _validate_metrics(metrics, metrics_sha, profile)
    generation_receipt, generation_receipt_size, generation_receipt_sha = _read_json(
        args.generation_receipt, "generation receipt"
    )
    _validate_generation_receipt(generation_receipt, profile)
    memory_receipt_path: Path | None = None
    memory_receipt_size: int | None = None
    memory_receipt_file_sha: str | None = None
    memory_receipt: dict[str, object] | None = None
    memory_receipt_live = False
    if args.measure_smoke:
        memory_receipt = _measure_memory_smoke(
            binary_path=args.binary,
            binary_sha=binary["sha256"],
            model_dir=args.model_dir,
            profile=profile,
            generation_receipt_sha=generation_receipt_sha,
        )
        if type(memory_receipt) is not dict:
            raise DeploymentError("memory smoke measurer did not return a receipt")
        memory_receipt_live = True
        _validate_memory_receipt(
            memory_receipt,
            profile,
            binary_path=args.binary,
            binary_sha=binary["sha256"],
            model_dir=args.model_dir,
            generation_receipt_sha=generation_receipt_sha,
            require_live=True,
        )
        binary_after = _validate_binary(args.binary)
        _validate_profile_binary(binary_after, profile)
        if binary_after != binary:
            raise DeploymentError("binary changed during the live memory smoke")
        post_profile, post_profile_size, post_profile_sha = _read_json(
            args.profile, "post-smoke profile"
        )
        if (
            post_profile_size != profile_size
            or post_profile_sha != profile_file_sha
            or post_profile != profile
        ):
            raise DeploymentError("profile changed during the live memory smoke")
        post_source, post_source_size, post_source_sha = _read_json(
            args.source_lock, "post-smoke source lock"
        )
        if (
            post_source_size != source_size
            or post_source_sha != source_file_sha
            or post_source != source_lock
        ):
            raise DeploymentError("source lock changed during the live memory smoke")
        post_artifacts = _validate_model_dir(args.model_dir, profile)
        if post_artifacts != artifacts:
            raise DeploymentError("model snapshot changed during the live memory smoke")
        post_manifest, post_manifest_size, post_manifest_sha = _read_json(
            args.oracle_manifest, "post-smoke oracle manifest"
        )
        if (
            post_manifest_size != manifest_size
            or post_manifest_sha != manifest_sha
            or post_manifest != manifest
        ):
            raise DeploymentError("oracle manifest changed during the live memory smoke")
        post_metrics, post_metrics_size, post_metrics_sha = _read_json(
            args.oracle_metrics, "post-smoke oracle metrics"
        )
        if (
            post_metrics_size != metrics_size
            or post_metrics_sha != metrics_sha
            or post_metrics != metrics
        ):
            raise DeploymentError("oracle metrics changed during the live memory smoke")
        post_generation, post_generation_size, post_generation_sha = _read_json(
            args.generation_receipt, "post-smoke generation receipt"
        )
        if (
            post_generation_size != generation_receipt_size
            or post_generation_sha != generation_receipt_sha
            or post_generation != generation_receipt
        ):
            raise DeploymentError("generation receipt changed during the live memory smoke")
    elif args.memory_receipt is not None:
        memory_receipt_path = args.memory_receipt
        memory_receipt, memory_receipt_size, memory_receipt_file_sha = _read_json(
            args.memory_receipt, "memory receipt"
        )
        _validate_memory_receipt(
            memory_receipt,
            profile,
            binary_path=args.binary,
            binary_sha=binary["sha256"],
            model_dir=args.model_dir,
            generation_receipt_sha=generation_receipt_sha,
            require_live=False,
        )
    lock = _build_lock(
        profile_path=args.profile,
        profile_size=profile_size,
        profile_file_sha=profile_file_sha,
        profile=profile,
        source_lock_path=args.source_lock,
        source_lock_size=source_size,
        source_lock_file_sha=source_file_sha,
        model_dir=args.model_dir,
        artifacts=artifacts,
        oracle_manifest_path=args.oracle_manifest,
        oracle_manifest_size=manifest_size,
        oracle_manifest_sha=manifest_sha,
        oracle_metrics_path=args.oracle_metrics,
        oracle_metrics_size=metrics_size,
        oracle_metrics_sha=metrics_sha,
        generation_receipt_path=args.generation_receipt,
        generation_receipt_size=generation_receipt_size,
        generation_receipt_sha=generation_receipt_sha,
        generation_receipt=generation_receipt,
        memory_receipt_path=memory_receipt_path,
        memory_receipt_size=memory_receipt_size,
        memory_receipt_file_sha=memory_receipt_file_sha,
        memory_receipt=memory_receipt,
        memory_receipt_live=memory_receipt_live,
        binary_path=args.binary,
        binary=binary,
    )
    if args.output is not None:
        _exclusive_atomic_json(args.output, lock)
    receipt: dict[str, object] = {
        "format": RECEIPT_FORMAT,
        "passed": True,
        "profile_id": profile["profile_id"],
        "repo_id": profile["source"]["repo_id"],
        "resolved_commit": profile["source"]["resolved_commit"],
        "deployment_lock_sha256": lock["content_sha256"],
        "memory_smoke": (
            {
                "present": True,
                "origin": "live" if memory_receipt_live else "file",
                "content_sha256": memory_receipt["content_sha256"],
            }
            if memory_receipt is not None
            else {"present": False}
        ),
        "output": str(_absolute(args.output)) if args.output is not None else None,
    }
    return receipt, lock


def parser() -> argparse.ArgumentParser:
    result = JsonArgumentParser(
        description="Offline verification gate for one pinned ApxInf macOS deployment."
    )
    result.add_argument("--profile", required=True, type=Path)
    result.add_argument("--source-lock", required=True, type=Path)
    result.add_argument("--model-dir", required=True, type=Path)
    result.add_argument("--oracle-manifest", required=True, type=Path)
    result.add_argument("--oracle-metrics", required=True, type=Path)
    result.add_argument("--generation-receipt", required=True, type=Path)
    result.add_argument("--binary", required=True, type=Path)
    memory = result.add_mutually_exclusive_group()
    memory.add_argument(
        "--memory-receipt",
        type=Path,
        help="validate an existing memory receipt (audit only; cannot publish --output)",
    )
    memory.add_argument(
        "--measure-smoke",
        action="store_true",
        help="run the fixed offline generation command under macOS memory accounting",
    )
    result.add_argument("--output", type=Path)
    return result


def _emit(value: object) -> None:
    sys.stdout.write(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")


def main(argv: list[str] | None = None) -> int:
    try:
        args = parser().parse_args(argv)
        receipt, _ = verify(args)
        _emit(receipt)
        return 0
    except DeploymentError as error:
        _emit(
            {
                "error": {"code": "DEPLOYMENT_GATE_FAILED", "message": str(error)},
                "passed": False,
            }
        )
        return 2
    except Exception as error:
        _emit(
            {
                "error": {
                    "code": "DEPLOYMENT_GATE_IO_ERROR",
                    "message": str(error) or type(error).__name__,
                },
                "passed": False,
            }
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
