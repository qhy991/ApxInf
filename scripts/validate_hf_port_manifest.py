#!/usr/bin/env python3
"""Validate an HF macOS port manifest without network or workspace mutation.

The validator is the fail-closed boundary between the read-only intake Mission
and a later controller that may compile a write Mission.  It deliberately does
not interpret model metadata, run commands, follow symlinks, or create paths.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import stat
import sys
import unicodedata
from typing import Any, NoReturn


SCRIPT_DIR = str(Path(__file__).resolve().parent)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from resolve_hf_source import (  # noqa: E402
    SourceLockError,
    _read_json_bytes as read_source_lock_json,
    validate_source_lock,
)


MAX_INPUT_BYTES = 1024 * 1024
MAX_PATHS = 256
MAX_PATH_BYTES = 1024
MAX_BLOCKERS = 128
MAX_GATES = 128
MAX_SOURCE_LOCK_BYTES = 32 * 1024 * 1024
MAX_PROFILE_BYTES = 1024 * 1024

QWEN_PROFILE_RELATIVE_PATH = Path("configs/hf-onboarding/qwen35-0.8b-macos-cpu.json")
QWEN_PROFILE_FILE_SHA256 = (
    "e5ba485f295b0e05eb78de817bff16c0fe7965557460ced874982ab882846816"
)
QWEN_PROFILE_ID = "qwen35-0.8b-macos-cpu"
QWEN_PROVIDER = "native-apxinf-cpu"
QWEN_READY_GATES = (
    "source-lock",
    "bundle-integrity",
    "pinned-macos-arm64-binary",
    "exact-greedy-token-trajectory",
    "transformers-oracle-parity",
    "macos-memory-smoke",
)

EXPECTED_KEYS = frozenset(
    {
        "schema_version",
        "repo_id",
        "requested_revision",
        "resolved_commit",
        "source_lock_content_sha256",
        "task",
        "input_modalities",
        "profile_id",
        "target",
        "route",
        "provider",
        "blockers",
        "user_checkpoint_required",
        "transaction_paths",
        "new_paths",
        "required_gates",
    }
)
ROUTES = frozenset(
    {
        "READY_EXISTING",
        "FAMILY_ADAPTER",
        "PORT_MODEL",
        "EXTEND_BACKEND",
        "EXTERNAL_PROVIDER",
        "BLOCKED",
    }
)
GENERIC_ROUTE_PROVIDERS = {
    "FAMILY_ADAPTER": frozenset({"native-apxinf-cpu"}),
    "PORT_MODEL": frozenset({"native-apxinf-cpu"}),
    "EXTEND_BACKEND": frozenset({"native-apxinf-cpu"}),
    "EXTERNAL_PROVIDER": frozenset({"coreml", "llama-cpp", "mlx", "mlx-lm"}),
    "BLOCKED": frozenset({"none"}),
}
GENERIC_ROUTE_GATES = {
    "FAMILY_ADAPTER": frozenset(
        {"source-lock", "macos-arm64-build", "tokenizer-parity", "numerical-parity"}
    ),
    "PORT_MODEL": frozenset(
        {
            "source-lock",
            "model-unit-tests",
            "macos-arm64-build",
            "tokenizer-parity",
            "numerical-parity",
        }
    ),
    "EXTEND_BACKEND": frozenset(
        {"source-lock", "backend-unit-tests", "macos-arm64-build", "numerical-parity"}
    ),
    "EXTERNAL_PROVIDER": frozenset(
        {"source-lock", "provider-availability", "offline-smoke", "numerical-parity"}
    ),
    "BLOCKED": frozenset(),
}
HF_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$")
HF_REVISION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,199}$")
COMMIT = re.compile(r"^[0-9a-f]{40}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
PROVIDER = re.compile(r"^[a-z0-9][a-z0-9._+-]{0,63}$")
PROFILE_ID = re.compile(r"^[a-z0-9][a-z0-9._+-]{0,95}$")
GATE = re.compile(r"^[a-z][a-z0-9]*(?:[._-][a-z0-9]+)*$")
PROFILE_ARTIFACT = re.compile(r"^[A-Za-z0-9._+@-]+(?:/[A-Za-z0-9._+@-]+)*$")

FORBIDDEN_COMPONENTS = frozenset(
    {
        ".git",
        ".kersor",
        ".cache",
        "cache",
        "caches",
        "hf-cache",
        "huggingface-cache",
        ".huggingface",
        "target",
        "weights",
        "model-weights",
        "checkpoints",
        "snapshots",
        "blobs",
        ".apxinf-models",
        "credentials",
        ".credentials",
        "secrets",
        ".secrets",
        "tokens",
        ".tokens",
        ".env",
        ".netrc",
        "netrc",
        "id_rsa",
        "id_ed25519",
        "autonomous-runs",
        "host-runs",
        "host-run",
        "run-dir",
        "run-dirs",
    }
)
FORBIDDEN_WEIGHT_MARKERS = (
    ".safetensors",
    ".gguf",
    ".ckpt",
    ".onnx",
    ".tflite",
    ".mlmodel",
)
FORBIDDEN_WEIGHT_SUFFIXES = (".bin", ".pt", ".pth", ".npz", ".npy")
FORBIDDEN_CREDENTIAL_SUFFIXES = (".pem", ".key", ".p12", ".pfx", ".jks")
FORBIDDEN_CREDENTIAL_NAMES = frozenset(
    {
        "auth",
        "credential",
        "credentials",
        "secret",
        "secrets",
        "token",
        "tokens",
        "hf_token",
        "access_token",
        "auth_token",
        "api_key",
        "secret_key",
    }
)


class ManifestError(ValueError):
    """A deterministic validation failure safe to expose in the receipt."""


class ReceiptArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        raise ManifestError(f"invalid arguments: {message}")


def _exact_dict(value: object, keys: set[str], field: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != keys:
        raise ManifestError(f"{field} has an invalid schema")
    return value


def _read_workspace_file(
    path: Path,
    workspace: Path,
    *,
    field: str,
    maximum: int,
) -> tuple[Path, bytes]:
    declared = Path(os.path.abspath(os.fspath(path)))
    try:
        info = declared.lstat()
    except OSError as error:
        raise ManifestError(f"cannot inspect {field}") from error
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise ManifestError(f"{field} must be a regular non-symlink file")
    if info.st_size > maximum:
        raise ManifestError(f"{field} exceeds its byte cap")
    resolved = declared.resolve(strict=True)
    try:
        resolved.relative_to(workspace)
    except ValueError as error:
        raise ManifestError(f"{field} must be inside the workspace") from error
    try:
        payload = resolved.read_bytes()
    except OSError as error:
        raise ManifestError(f"cannot read {field}") from error
    if len(payload) > maximum:
        raise ManifestError(f"{field} exceeds its byte cap")
    return resolved, payload


def _parse_profile(payload: bytes) -> dict[str, Any]:
    try:
        profile = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_object_without_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except ManifestError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ManifestError("deployment profile is not valid UTF-8 JSON") from error
    root = _exact_dict(
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
        "deployment profile",
    )
    if root["format"] != "apxinf-hf-macos-deployment-profile-v1":
        raise ManifestError("deployment profile format is invalid")
    if root["profile_id"] != QWEN_PROFILE_ID:
        raise ManifestError("deployment profile id is not trusted")
    source = _exact_dict(
        root["source"],
        {
            "repo_id",
            "resolved_commit",
            "license",
            "source_lock_content_sha256",
            "config_sha256",
        },
        "deployment profile source",
    )
    runtime = _exact_dict(
        root["runtime"],
        {"target", "provider", "device", "dtype", "matmul_feature"},
        "deployment profile runtime",
    )
    if runtime != {
        "target": "macos-arm64",
        "provider": QWEN_PROVIDER,
        "device": "cpu",
        "dtype": "fp32",
        "matmul_feature": "accelerate",
    }:
        raise ManifestError(
            "deployment profile runtime is not the native macOS contract"
        )
    for field in ("resolved_commit", "source_lock_content_sha256", "config_sha256"):
        value = source[field]
        pattern = COMMIT if field == "resolved_commit" else SHA256
        if type(value) is not str or not pattern.fullmatch(value):
            raise ManifestError(f"deployment profile source {field} is invalid")
    artifacts = root["artifacts"]
    if type(artifacts) is not dict or not artifacts or len(artifacts) > 128:
        raise ManifestError("deployment profile artifacts are invalid")
    for name, value in artifacts.items():
        if (
            type(name) is not str
            or len(name.encode("utf-8")) > MAX_PATH_BYTES
            or not PROFILE_ARTIFACT.fullmatch(name)
            or any(part in {"", ".", ".."} for part in name.split("/"))
        ):
            raise ManifestError("deployment profile artifact path is invalid")
        record = _exact_dict(
            value, {"size", "sha256"}, "deployment profile artifact record"
        )
        if type(record["size"]) is not int or record["size"] <= 0:
            raise ManifestError("deployment profile artifact size is invalid")
        if type(record["sha256"]) is not str or not SHA256.fullmatch(record["sha256"]):
            raise ManifestError("deployment profile artifact SHA-256 is invalid")
    for field in ("binary", "gate", "memory_smoke", "oracle"):
        if type(root[field]) is not dict or not root[field]:
            raise ManifestError(f"deployment profile {field} section is invalid")
    return root


def _load_host_source_and_profile(
    *,
    workspace: Path,
    source_lock_path: Path,
    deployment_profile_path: Path,
    expected_identity: dict[str, str],
) -> tuple[dict[str, Any], dict[str, Any], bool]:
    _, source_payload = _read_workspace_file(
        source_lock_path,
        workspace,
        field="source lock",
        maximum=MAX_SOURCE_LOCK_BYTES,
    )
    try:
        source_lock = read_source_lock_json(source_payload, "source lock")
        source_receipt = validate_source_lock(source_lock)
    except (SourceLockError, UnicodeError, ValueError) as error:
        raise ManifestError(
            "source lock failed deterministic Host validation"
        ) from error
    source_identity = {
        "repo_id": source_receipt["repo_id"],
        "requested_revision": source_receipt["requested_revision"],
        "resolved_commit": source_receipt["resolved_commit"],
        "source_lock_content_sha256": source_receipt["content_sha256"],
    }
    if source_identity != expected_identity:
        raise ManifestError("source lock identity differs from Host-bound expectations")

    profile_path, profile_payload = _read_workspace_file(
        deployment_profile_path,
        workspace,
        field="deployment profile",
        maximum=MAX_PROFILE_BYTES,
    )
    expected_profile_path = (workspace / QWEN_PROFILE_RELATIVE_PATH).resolve()
    if profile_path != expected_profile_path:
        raise ManifestError("deployment profile is not the checked-in Host profile")
    if hashlib.sha256(profile_payload).hexdigest() != QWEN_PROFILE_FILE_SHA256:
        raise ManifestError("checked-in deployment profile bytes are not trusted")
    profile = _parse_profile(profile_payload)

    architecture = source_lock["architecture"]
    metadata_records = {
        record["path"]: record for record in source_lock["metadata"]["files"]
    }
    config_record = metadata_records.get("config.json")
    if type(config_record) is not dict or architecture[
        "config_sha256"
    ] != config_record.get("sha256"):
        raise ManifestError("source lock config identity is internally inconsistent")
    profile_source = profile["source"]
    profile_artifacts = profile["artifacts"]
    profile_config = profile_artifacts.get("config.json")
    source_weights = sorted(
        [
            {
                "path": record["path"],
                "size": record["size"],
                "sha256": record["sha256"],
            }
            for record in source_lock["weights"]["files"]
        ],
        key=lambda record: record["path"],
    )
    profile_weights = sorted(
        [
            {
                "path": name,
                "size": record["size"],
                "sha256": record["sha256"],
            }
            for name, record in profile_artifacts.items()
            if name.endswith(".safetensors")
        ],
        key=lambda record: record["path"],
    )
    current_qwen = (
        profile_source["repo_id"] == source_lock["repo_id"]
        and profile_source["resolved_commit"] == source_lock["resolved_commit"]
        and profile_source["source_lock_content_sha256"]
        == source_lock["content_sha256"]
        and profile_source["config_sha256"] == architecture["config_sha256"]
        and profile_config
        == {"size": config_record["size"], "sha256": config_record["sha256"]}
        and profile_weights == source_weights
    )
    return source_lock, profile, current_qwen


def _reject_constant(value: str) -> NoReturn:
    raise ManifestError(f"non-finite JSON number is not allowed: {value}")


def _object_without_duplicate_keys(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ManifestError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def parse_json(payload: bytes) -> dict[str, object]:
    if not payload:
        raise ManifestError("manifest input is empty")
    if len(payload) > MAX_INPUT_BYTES:
        raise ManifestError(f"manifest exceeds {MAX_INPUT_BYTES} bytes")
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ManifestError("manifest is not valid UTF-8") from error
    try:
        value = json.loads(
            text,
            object_pairs_hook=_object_without_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except json.JSONDecodeError as error:
        raise ManifestError(
            f"invalid JSON at line {error.lineno} column {error.colno}"
        ) from error
    if type(value) is not dict:
        raise ManifestError("manifest root must be a JSON object")
    return value


def _require_exact_string(value: object, field: str, *, max_length: int) -> str:
    if type(value) is not str:
        raise ManifestError(f"{field} must be a string")
    if not value or value != value.strip():
        raise ManifestError(f"{field} must be non-empty and have no outer whitespace")
    if len(value) > max_length:
        raise ManifestError(f"{field} exceeds {max_length} characters")
    if any(unicodedata.category(char) == "Cc" for char in value):
        raise ManifestError(f"{field} contains a control character")
    return value


def _require_string_list(
    value: object,
    field: str,
    *,
    maximum: int,
    item_max_length: int,
) -> list[str]:
    if type(value) is not list:
        raise ManifestError(f"{field} must be an array")
    if len(value) > maximum:
        raise ManifestError(f"{field} contains more than {maximum} entries")
    result: list[str] = []
    seen: set[str] = set()
    for index, item in enumerate(value):
        text = _require_exact_string(
            item, f"{field}[{index}]", max_length=item_max_length
        )
        collision_key = unicodedata.normalize("NFC", text).casefold()
        if collision_key in seen:
            raise ManifestError(f"{field} contains duplicate entry: {text}")
        seen.add(collision_key)
        result.append(text)
    return result


def _validate_repo_id(value: object) -> str:
    repo_id = _require_exact_string(value, "repo_id", max_length=192)
    components = repo_id.split("/")
    if len(components) != 2 or not all(
        HF_COMPONENT.fullmatch(item) for item in components
    ):
        raise ManifestError("repo_id must be a canonical owner/model id")
    return repo_id


def _validate_revision(value: object) -> str:
    revision = _require_exact_string(value, "requested_revision", max_length=200)
    parts = revision.split("/")
    if (
        not HF_REVISION.fullmatch(revision)
        or "//" in revision
        or any(part in {"", ".", ".."} for part in parts)
    ):
        raise ManifestError("requested_revision is not a safe Hugging Face revision")
    return revision


def _collision_key(value: str) -> str:
    return unicodedata.normalize("NFC", value).casefold()


def _path_parts(value: str, field: str) -> tuple[str, ...]:
    if len(value.encode("utf-8")) > MAX_PATH_BYTES:
        raise ManifestError(f"{field} exceeds {MAX_PATH_BYTES} UTF-8 bytes")
    if "\\" in value:
        raise ManifestError(f"{field} must use canonical POSIX separators")
    posix = PurePosixPath(value)
    windows = PureWindowsPath(value)
    raw_parts = value.split("/")
    if posix.is_absolute() or windows.is_absolute() or windows.drive:
        raise ManifestError(f"{field} must be workspace-relative")
    if any(part in {"", ".", ".."} for part in raw_parts):
        raise ManifestError(f"{field} is not a canonical lexical path")
    if posix.as_posix() != value:
        raise ManifestError(f"{field} is not a canonical lexical path")
    if any(unicodedata.category(char).startswith("C") for char in value):
        raise ManifestError(f"{field} contains a control character")
    if raw_parts[0].startswith("~"):
        raise ManifestError(f"{field} must not use home-directory notation")
    _reject_forbidden_path(raw_parts, field)
    return tuple(raw_parts)


def _reject_forbidden_path(parts: list[str], field: str) -> None:
    lowered = [_collision_key(part) for part in parts]
    for component in lowered:
        if component in FORBIDDEN_COMPONENTS:
            raise ManifestError(f"{field} enters forbidden path component: {component}")
        if component.startswith(".env."):
            raise ManifestError(f"{field} enters a credentials path")
        credential_stem = re.split(r"[._-]", component.lstrip("."), maxsplit=1)[0]
        if component in FORBIDDEN_CREDENTIAL_NAMES or credential_stem in {
            "auth",
            "credential",
            "credentials",
            "secret",
            "secrets",
            "token",
            "tokens",
        }:
            raise ManifestError(f"{field} enters a credentials path")
        if component.endswith(FORBIDDEN_CREDENTIAL_SUFFIXES):
            raise ManifestError(f"{field} enters a credentials path")
    basename = lowered[-1]
    if any(marker in basename for marker in FORBIDDEN_WEIGHT_MARKERS):
        raise ManifestError(f"{field} names a model weight artifact")
    if basename.endswith(FORBIDDEN_WEIGHT_SUFFIXES):
        raise ManifestError(f"{field} names a model weight artifact")
    if any(f"{suffix}." in basename for suffix in FORBIDDEN_WEIGHT_SUFFIXES):
        raise ManifestError(f"{field} names a model weight artifact")


def _lstat(path: Path, field: str) -> os.stat_result | None:
    try:
        return path.lstat()
    except FileNotFoundError:
        return None
    except OSError as error:
        raise ManifestError(
            f"cannot inspect {field}: {error.strerror or error}"
        ) from error


def _reject_case_collision(parent: Path, component: str, field: str) -> None:
    wanted = _collision_key(component)
    try:
        with os.scandir(parent) as entries:
            for entry in entries:
                if entry.name != component and _collision_key(entry.name) == wanted:
                    raise ManifestError(
                        f"{field} has a case/Unicode collision with existing entry: {entry.name}"
                    )
    except ManifestError:
        raise
    except OSError as error:
        raise ManifestError(
            f"cannot inspect parent of {field}: {error.strerror or error}"
        ) from error


def _inspect_transaction_path(
    workspace: Path,
    value: str,
    parts: tuple[str, ...],
    *,
    must_be_new: bool,
) -> str:
    current = workspace
    for index, component in enumerate(parts):
        is_leaf = index == len(parts) - 1
        _reject_case_collision(current, component, value)
        candidate = current / component
        info = _lstat(candidate, value)
        if info is None:
            if not is_leaf:
                raise ManifestError(
                    f"{value} has a missing parent component: {component}"
                )
            if not must_be_new:
                raise ManifestError(
                    f"missing transaction path is not declared in new_paths: {value}"
                )
            return "new"
        if stat.S_ISLNK(info.st_mode):
            raise ManifestError(f"{value} crosses or targets a symbolic link")
        if not is_leaf:
            if not stat.S_ISDIR(info.st_mode):
                raise ManifestError(
                    f"{value} has a non-directory parent component: {component}"
                )
            current = candidate
            continue
        if must_be_new:
            raise ManifestError(f"new_paths entry already exists: {value}")
        if stat.S_ISDIR(info.st_mode):
            raise ManifestError(
                f"existing directory cannot be a transaction path: {value}"
            )
        if not stat.S_ISREG(info.st_mode):
            raise ManifestError(f"transaction path is not a regular file: {value}")
        if info.st_nlink != 1:
            raise ManifestError(f"transaction path has hard links: {value}")
        return "existing-file"
    raise ManifestError(f"empty transaction path: {value}")


def _validate_path_sets(
    workspace: Path, transaction_paths: list[str], new_paths: list[str]
) -> list[dict[str, str]]:
    parsed: dict[str, tuple[str, ...]] = {}
    canonical_to_value: dict[str, str] = {}
    for index, value in enumerate(transaction_paths):
        parts = _path_parts(value, f"transaction_paths[{index}]")
        key = "/".join(_collision_key(part) for part in parts)
        if key in canonical_to_value:
            raise ManifestError(
                f"transaction_paths collide: {canonical_to_value[key]} and {value}"
            )
        canonical_to_value[key] = value
        parsed[value] = parts

    new_keys: set[str] = set()
    for index, value in enumerate(new_paths):
        parts = _path_parts(value, f"new_paths[{index}]")
        key = "/".join(_collision_key(part) for part in parts)
        if key in new_keys:
            raise ManifestError(f"new_paths contains duplicate entry: {value}")
        new_keys.add(key)
        if key not in canonical_to_value:
            raise ManifestError(
                f"new_paths entry is not present in transaction_paths: {value}"
            )
        if canonical_to_value[key] != value:
            raise ManifestError(
                f"new_paths entry must exactly match transaction path: {value}"
            )

    ordered = [(value, parsed[value]) for value in transaction_paths]
    for left_index, (left, left_parts) in enumerate(ordered):
        left_key = tuple(_collision_key(part) for part in left_parts)
        for right, right_parts in ordered[left_index + 1 :]:
            right_key = tuple(_collision_key(part) for part in right_parts)
            shared = min(len(left_key), len(right_key))
            if left_key[:shared] == right_key[:shared]:
                raise ManifestError(
                    f"transaction paths overlap as parent/child: {left} and {right}"
                )

    facts: list[dict[str, str]] = []
    for value, parts in ordered:
        key = "/".join(_collision_key(part) for part in parts)
        state = _inspect_transaction_path(
            workspace, value, parts, must_be_new=key in new_keys
        )
        facts.append({"path": value, "state": state})
    return facts


def validate_manifest(
    manifest: dict[str, object],
    workspace: Path,
    *,
    expected_repo_id: str,
    expected_requested_revision: str,
    expected_resolved_commit: str,
    expected_source_lock_content_sha256: str,
    source_lock_path: Path | None = None,
    deployment_profile_path: Path | None = None,
    require_ready_existing: bool = False,
) -> dict[str, object]:
    keys = set(manifest)
    missing = sorted(EXPECTED_KEYS - keys)
    unknown = sorted(keys - EXPECTED_KEYS)
    if missing:
        raise ManifestError(f"missing manifest keys: {', '.join(missing)}")
    if unknown:
        raise ManifestError(f"unknown manifest keys: {', '.join(unknown)}")

    schema_version = manifest["schema_version"]
    if type(schema_version) is not int or schema_version != 2:
        raise ManifestError("schema_version must be integer 2")
    repo_id = _validate_repo_id(manifest["repo_id"])
    requested_revision = _validate_revision(manifest["requested_revision"])
    resolved_commit = _require_exact_string(
        manifest["resolved_commit"], "resolved_commit", max_length=40
    )
    if not COMMIT.fullmatch(resolved_commit):
        raise ManifestError("resolved_commit must be a lowercase 40-hex commit")
    source_lock_content_sha256 = _require_exact_string(
        manifest["source_lock_content_sha256"],
        "source_lock_content_sha256",
        max_length=64,
    )
    if not SHA256.fullmatch(source_lock_content_sha256):
        raise ManifestError(
            "source_lock_content_sha256 must be 64 lowercase hex characters"
        )
    if manifest["task"] != "text-generation":
        raise ManifestError("task must be text-generation")
    if manifest["input_modalities"] != ["text"]:
        raise ManifestError("input_modalities must be exactly ['text']")
    profile_value = manifest["profile_id"]
    if profile_value is None:
        profile_id: str | None = None
    else:
        profile_id = _require_exact_string(profile_value, "profile_id", max_length=96)
        if not PROFILE_ID.fullmatch(profile_id):
            raise ManifestError("profile_id must be a canonical lowercase profile id")

    locked_repo_id = _validate_repo_id(expected_repo_id)
    locked_requested_revision = _validate_revision(expected_requested_revision)
    locked_resolved_commit = _require_exact_string(
        expected_resolved_commit, "expected resolved commit", max_length=40
    )
    if not COMMIT.fullmatch(locked_resolved_commit):
        raise ManifestError(
            "expected resolved commit must be a lowercase 40-hex commit"
        )
    locked_source_hash = _require_exact_string(
        expected_source_lock_content_sha256,
        "expected source-lock content SHA-256",
        max_length=64,
    )
    if not SHA256.fullmatch(locked_source_hash):
        raise ManifestError(
            "expected source-lock content SHA-256 must be 64 lowercase hex characters"
        )
    expected_identity = {
        "repo_id": locked_repo_id,
        "requested_revision": locked_requested_revision,
        "resolved_commit": locked_resolved_commit,
        "source_lock_content_sha256": locked_source_hash,
    }
    observed_identity = {
        "repo_id": repo_id,
        "requested_revision": requested_revision,
        "resolved_commit": resolved_commit,
        "source_lock_content_sha256": source_lock_content_sha256,
    }
    if observed_identity != expected_identity:
        mismatches = sorted(
            key
            for key in expected_identity
            if observed_identity[key] != expected_identity[key]
        )
        raise ManifestError(
            "manifest identity differs from Host-bound expectations: "
            + ", ".join(mismatches)
        )
    if manifest["target"] != "macos-arm64":
        raise ManifestError("target must be macos-arm64")
    route = manifest["route"]
    if type(route) is not str or route not in ROUTES:
        raise ManifestError(f"route must be one of: {', '.join(sorted(ROUTES))}")
    if route == "READY_EXISTING" and profile_id is None:
        raise ManifestError("READY_EXISTING requires a Host-verifiable profile_id")
    if route != "READY_EXISTING" and profile_id is not None:
        raise ManifestError("only READY_EXISTING may name a checked-in profile")
    provider = _require_exact_string(manifest["provider"], "provider", max_length=64)
    if not PROVIDER.fullmatch(provider):
        raise ManifestError("provider must be a canonical lowercase provider id")
    if route != "READY_EXISTING" and provider not in GENERIC_ROUTE_PROVIDERS[route]:
        raise ManifestError("provider is not allowlisted for the selected route")

    blockers = _require_string_list(
        manifest["blockers"],
        "blockers",
        maximum=MAX_BLOCKERS,
        item_max_length=1024,
    )
    checkpoint = manifest["user_checkpoint_required"]
    if type(checkpoint) is not bool:
        raise ManifestError("user_checkpoint_required must be a boolean")
    transaction_paths = _require_string_list(
        manifest["transaction_paths"],
        "transaction_paths",
        maximum=MAX_PATHS,
        item_max_length=MAX_PATH_BYTES,
    )
    new_paths = _require_string_list(
        manifest["new_paths"],
        "new_paths",
        maximum=MAX_PATHS,
        item_max_length=MAX_PATH_BYTES,
    )
    gates = _require_string_list(
        manifest["required_gates"],
        "required_gates",
        maximum=MAX_GATES,
        item_max_length=96,
    )
    for gate in gates:
        if not GATE.fullmatch(gate):
            raise ManifestError(f"required gate is not a canonical gate id: {gate}")
    if route != "READY_EXISTING" and not set(gates).issubset(
        GENERIC_ROUTE_GATES[route]
    ):
        raise ManifestError(
            "required_gates contains a gate not allowlisted for the route"
        )
    if route != "BLOCKED" and not gates:
        raise ManifestError(
            "non-BLOCKED routes require at least one required_gates entry"
        )
    if route == "BLOCKED" and not blockers:
        raise ManifestError("BLOCKED route requires at least one blocker")
    if route == "BLOCKED" and (transaction_paths or new_paths):
        raise ManifestError("BLOCKED route must not propose transaction paths")
    if checkpoint and not blockers:
        raise ManifestError(
            "user checkpoint requires at least one blocker explaining it"
        )
    if route != "READY_EXISTING" and (not checkpoint or not blockers):
        raise ManifestError(
            "non-READY routes require a user checkpoint and explanatory blocker"
        )

    workspace = Path(os.path.abspath(os.fspath(workspace)))
    workspace_info = _lstat(workspace, "workspace")
    if workspace_info is None or not stat.S_ISDIR(workspace_info.st_mode):
        raise ManifestError("workspace must be an existing directory")
    if stat.S_ISLNK(workspace_info.st_mode):
        raise ManifestError("workspace must not be a symbolic link")
    if (source_lock_path is None) != (deployment_profile_path is None):
        raise ManifestError(
            "source lock and deployment profile Host bindings must be supplied together"
        )
    current_qwen = False
    profile_file_sha256: str | None = None
    if source_lock_path is not None and deployment_profile_path is not None:
        _, _, current_qwen = _load_host_source_and_profile(
            workspace=workspace,
            source_lock_path=source_lock_path,
            deployment_profile_path=deployment_profile_path,
            expected_identity=expected_identity,
        )
        profile_file_sha256 = QWEN_PROFILE_FILE_SHA256
    if route == "READY_EXISTING":
        if not current_qwen:
            raise ManifestError(
                "READY_EXISTING is not verified by the checked-in Host profile"
            )
        if profile_id != QWEN_PROFILE_ID:
            raise ManifestError("READY_EXISTING profile_id is not the verified profile")
        if provider != QWEN_PROVIDER:
            raise ManifestError("READY_EXISTING provider is not the verified provider")
        if blockers or checkpoint or transaction_paths or new_paths:
            raise ManifestError(
                "READY_EXISTING must have no blockers, checkpoint, or transaction paths"
            )
        if gates != list(QWEN_READY_GATES):
            raise ManifestError(
                "READY_EXISTING required_gates do not match the Host contract"
            )
    elif current_qwen:
        raise ManifestError(
            "the Host-verified Qwen source must use route READY_EXISTING"
        )
    host_decision_complete = route == "READY_EXISTING" and current_qwen
    if require_ready_existing and not host_decision_complete:
        raise ManifestError(
            "this intake requires the exact Host-verified READY_EXISTING route"
        )
    path_facts = _validate_path_sets(workspace, transaction_paths, new_paths)

    canonical_payload = json.dumps(
        manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return {
        "manifest_sha256": hashlib.sha256(canonical_payload).hexdigest(),
        "new_paths": sorted(new_paths),
        "passed": True,
        "route_verified": host_decision_complete,
        "decision_complete": host_decision_complete,
        "path_facts": path_facts,
        "provider": provider,
        "repo_id": repo_id,
        "requested_revision": requested_revision,
        "resolved_commit": resolved_commit,
        "source_lock_content_sha256": source_lock_content_sha256,
        "route": route,
        "schema_version": 2,
        "task": "text-generation",
        "input_modalities": ["text"],
        "profile_id": profile_id,
        "profile_file_sha256": profile_file_sha256,
        "target": "macos-arm64",
        "transaction_paths": sorted(transaction_paths),
        "user_checkpoint_required": checkpoint,
        "workspace": str(workspace),
        "blocker_count": len(blockers),
        "required_gates": gates,
    }


def _read_input(source: str) -> bytes:
    if source == "-":
        payload = sys.stdin.buffer.read(MAX_INPUT_BYTES + 1)
    else:
        path = Path(source)
        try:
            info = path.lstat()
        except OSError as error:
            raise ManifestError(
                f"cannot inspect manifest input: {error.strerror or error}"
            ) from error
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
            raise ManifestError("manifest input must be a regular non-symlink file")
        if info.st_size > MAX_INPUT_BYTES:
            raise ManifestError(f"manifest exceeds {MAX_INPUT_BYTES} bytes")
        try:
            payload = path.read_bytes()
        except OSError as error:
            raise ManifestError(
                f"cannot read manifest input: {error.strerror or error}"
            ) from error
    if len(payload) > MAX_INPUT_BYTES:
        raise ManifestError(f"manifest exceeds {MAX_INPUT_BYTES} bytes")
    return payload


def parser() -> argparse.ArgumentParser:
    result = ReceiptArgumentParser(
        description="Validate one metadata-only HF port_manifest before write-Mission compilation."
    )
    result.add_argument(
        "manifest",
        nargs="?",
        default="-",
        help="manifest JSON file, or -/omitted for stdin",
    )
    result.add_argument(
        "--json",
        dest="inline_json",
        help="manifest JSON bytes supplied directly by a KerSor artifact binding",
    )
    result.add_argument(
        "--workspace",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="workspace against which transaction paths are inspected",
    )
    result.add_argument(
        "--source-lock",
        type=Path,
        required=True,
        help="Host-bound source-lock JSON validated independently of the Agent",
    )
    result.add_argument(
        "--deployment-profile",
        type=Path,
        required=True,
        help="absolute checked-in deployment profile used for READY_EXISTING",
    )
    result.add_argument("--expected-repo-id", required=True)
    result.add_argument("--expected-requested-revision", required=True)
    result.add_argument("--expected-resolved-commit", required=True)
    result.add_argument("--expected-source-lock-content-sha256", required=True)
    result.add_argument("--require-ready-existing", action="store_true")
    return result


def _emit(receipt: dict[str, object]) -> None:
    sys.stdout.write(
        json.dumps(receipt, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    )


def main(argv: list[str] | None = None) -> int:
    try:
        args = parser().parse_args(argv)
        if args.inline_json is not None and args.manifest != "-":
            raise ManifestError("--json cannot be combined with a manifest file")
        payload = (
            args.inline_json.encode("utf-8")
            if args.inline_json is not None
            else _read_input(args.manifest)
        )
        manifest = parse_json(payload)
        _emit(
            validate_manifest(
                manifest,
                args.workspace,
                expected_repo_id=args.expected_repo_id,
                expected_requested_revision=args.expected_requested_revision,
                expected_resolved_commit=args.expected_resolved_commit,
                expected_source_lock_content_sha256=(
                    args.expected_source_lock_content_sha256
                ),
                source_lock_path=args.source_lock,
                deployment_profile_path=args.deployment_profile,
                require_ready_existing=args.require_ready_existing,
            )
        )
        return 0
    except ManifestError as error:
        _emit(
            {
                "error": {"code": "INVALID_PORT_MANIFEST", "message": str(error)},
                "passed": False,
            }
        )
        return 2
    except (OSError, UnicodeError, ValueError) as error:
        _emit(
            {
                "error": {
                    "code": "VALIDATION_IO_ERROR",
                    "message": str(error) or type(error).__name__,
                },
                "passed": False,
            }
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
