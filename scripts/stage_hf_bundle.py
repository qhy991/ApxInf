#!/usr/bin/env python3
"""Stage the pinned Qwen3.5 bundle from Hugging Face, then publish atomically.

This is intentionally not a general-purpose Hub downloader.  The checked-in
deployment profile and the metadata-only source lock jointly define the only
accepted repository, commit, filenames, sizes, and SHA-256 digests.  Downloads
use no ambient proxy or credentials, and an incomplete sibling staging
directory is retained so a later invocation can resume with a verified Range
request.

The local security boundary trusts the current UID.  Owner checks, no-follow
opens, and the adjacent flock prevent accidents and coordinate cooperating
processes; they do not claim to defend against a malicious process running as
the same user.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import ctypes
from dataclasses import dataclass
import errno
import fcntl
import hashlib
from http.client import HTTPException, IncompleteRead
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys
import time
from types import MappingProxyType
from typing import Any, Iterator, Mapping, NoReturn
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urljoin, urlparse
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROFILE = ROOT / "configs/hf-onboarding/qwen35-0.8b-macos-cpu.json"
DEFAULT_SOURCE_LOCK = ROOT / ".apxinf/onboarding/qwen35-0.8b/source-lock.json"

RECEIPT_FORMAT = "apxinf-hf-bundle-stage-receipt-v1"
PROFILE_FORMAT = "apxinf-hf-macos-deployment-profile-v1"
SOURCE_LOCK_FORMAT = "apxinf-hf-source-lock-v1"
MAX_JSON_BYTES = 16 * 1024 * 1024
HARD_MAX_TOTAL_BYTES = 2 * 1024 * 1024 * 1024
DEFAULT_TIMEOUT_SECONDS = 2 * 60 * 60
MAX_TIMEOUT_SECONDS = 24 * 60 * 60
READ_CHUNK_BYTES = 1024 * 1024
MAX_CRITICAL_HEADER_BYTES = 128
MAX_NUMERIC_HEADER_DIGITS = 20
MAX_CACHE_ENTRIES = 4096
MAX_CACHE_FILE_BYTES = 16 * 1024 * 1024
MAX_CACHE_TOTAL_BYTES = 64 * 1024 * 1024
SHA1 = re.compile(r"^[0-9a-f]{40}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
CONTENT_RANGE = re.compile(r"^bytes ([0-9]+)-([0-9]+)/([0-9]+)$")

# Hugging Face currently serves immutable resolve URLs from huggingface.co and
# signed content URLs below domains it owns (notably cas-bridge.xethub.hf.co).
# Matching is label-aware: "evilhf.co" and "huggingface.co.example" do not pass.
ALLOWED_DOWNLOAD_DOMAIN_SUFFIXES = ("huggingface.co", "hf.co")

PROFILE_KEYS = frozenset(
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
    }
)
PROFILE_SOURCE_KEYS = frozenset(
    {
        "config_sha256",
        "license",
        "repo_id",
        "resolved_commit",
        "source_lock_content_sha256",
    }
)
SOURCE_LOCK_METADATA_ARTIFACTS = frozenset(
    {"config.json", "model.safetensors.index.json", "tokenizer_config.json"}
)
FORBIDDEN_ARTIFACT_SUFFIXES = (
    ".py",
    ".pyc",
    ".pyd",
    ".so",
    ".dylib",
    ".dll",
    ".exe",
    ".sh",
)
FORBIDDEN_CACHE_SCRIPT_SUFFIXES = FORBIDDEN_ARTIFACT_SUFFIXES + (
    ".bash",
    ".command",
    ".fish",
    ".pl",
    ".ps1",
    ".rb",
    ".zsh",
)


class StageError(ValueError):
    """A fail-closed staging error safe to expose in a local JSON receipt."""


class ArtifactIntegrityError(StageError):
    """A completed HTTP payload whose exact pinned SHA-256 did not match."""

    def __init__(self, message: str, *, downloaded_bytes: int) -> None:
        super().__init__(message)
        self.downloaded_bytes = downloaded_bytes


class ReceiptArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        raise StageError(f"invalid arguments: {message}")


@dataclass(frozen=True)
class Artifact:
    path: str
    size: int
    sha256: str


@dataclass(frozen=True)
class BundlePlan:
    profile_id: str
    repo_id: str
    resolved_commit: str
    source_lock_content_sha256: str
    artifact_manifest_sha256: str
    artifacts: tuple[Artifact, ...]

    @property
    def total_bytes(self) -> int:
        return sum(artifact.size for artifact in self.artifacts)


@dataclass(frozen=True)
class ProfileContract:
    profile_id: str
    repo_id: str
    resolved_commit: str
    source_lock_content_sha256: str
    source: Mapping[str, object]
    artifacts: Mapping[str, Mapping[str, object]]


QWEN_ARTIFACTS: Mapping[str, Mapping[str, object]] = MappingProxyType(
    {
        "chat_template.jinja": MappingProxyType(
            {
                "size": 7_755,
                "sha256": "273d8e0e683b885071fb17e08d71e5f2a5ddfb5309756181681de4f5a1822d80",
            }
        ),
        "config.json": MappingProxyType(
            {
                "size": 2_907,
                "sha256": "b90b86f35c8e6925ef74ee04d0e758f0a845c83a42089ad82bbaa948de9b4204",
            }
        ),
        "model.safetensors-00001-of-00001.safetensors": MappingProxyType(
            {
                "size": 1_746_942_600,
                "sha256": "04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696",
            }
        ),
        "model.safetensors.index.json": MappingProxyType(
            {
                "size": 50_900,
                "sha256": "d8a08838a613b025eb7952ed9db11696213e57e76a375661ef5c12f9dd5dcf4e",
            }
        ),
        "tokenizer.json": MappingProxyType(
            {
                "size": 12_807_982,
                "sha256": "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42",
            }
        ),
        "tokenizer_config.json": MappingProxyType(
            {
                "size": 16_709,
                "sha256": "49e2b6e395f959f077f1e992b338919c0d4a9732fc6e613995e06557f843500c",
            }
        ),
    }
)
QWEN_SOURCE: Mapping[str, object] = MappingProxyType(
    {
        "config_sha256": "b90b86f35c8e6925ef74ee04d0e758f0a845c83a42089ad82bbaa948de9b4204",
        "license": "Apache-2.0",
        "repo_id": "Qwen/Qwen3.5-0.8B",
        "resolved_commit": "2fc06364715b967f1860aea9cf38778875588b17",
        "source_lock_content_sha256": "021209cc96e398db4aac6d126890f7bb5a5a3b5fce7204fed0328f544cbb7500",
    }
)
QWEN_CONTRACT = ProfileContract(
    profile_id="qwen35-0.8b-macos-cpu",
    repo_id="Qwen/Qwen3.5-0.8B",
    resolved_commit="2fc06364715b967f1860aea9cf38778875588b17",
    source_lock_content_sha256=(
        "021209cc96e398db4aac6d126890f7bb5a5a3b5fce7204fed0328f544cbb7500"
    ),
    source=QWEN_SOURCE,
    artifacts=QWEN_ARTIFACTS,
)


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _reject_constant(value: str) -> NoReturn:
    raise StageError(f"non-finite JSON number is forbidden: {value}")


def _object_without_duplicates(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise StageError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _parse_json(payload: bytes, label: str) -> dict[str, object]:
    if not payload:
        raise StageError(f"{label} is empty")
    if len(payload) > MAX_JSON_BYTES:
        raise StageError(f"{label} exceeds {MAX_JSON_BYTES} bytes")
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_object_without_duplicates,
            parse_constant=_reject_constant,
        )
    except UnicodeDecodeError as error:
        raise StageError(f"{label} is not UTF-8") from error
    except json.JSONDecodeError as error:
        raise StageError(
            f"{label} is not valid JSON at line {error.lineno} column {error.colno}"
        ) from error
    if type(value) is not dict:
        raise StageError(f"{label} must contain one JSON object")
    return value


def _read_json_regular(path: Path, label: str) -> dict[str, object]:
    try:
        before = path.lstat()
    except OSError as error:
        raise StageError(
            f"cannot inspect {label}: {error.strerror or error}"
        ) from error
    if (
        stat.S_ISLNK(before.st_mode)
        or not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
    ):
        raise StageError(f"{label} must be a single-link regular non-symlink file")
    if before.st_size > MAX_JSON_BYTES:
        raise StageError(f"{label} exceeds {MAX_JSON_BYTES} bytes")
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise StageError(f"cannot open {label}: {error.strerror or error}") from error
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_dev != before.st_dev
            or opened.st_ino != before.st_ino
            or opened.st_nlink != 1
        ):
            raise StageError(f"{label} changed while it was opened")
        remaining = opened.st_size
        chunks: list[bytes] = []
        while remaining:
            chunk = os.read(descriptor, min(READ_CHUNK_BYTES, remaining))
            if not chunk:
                raise StageError(f"{label} ended before its declared size")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise StageError(f"{label} grew while it was read")
        after = os.fstat(descriptor)
        if (
            after.st_size != opened.st_size
            or after.st_mtime_ns != opened.st_mtime_ns
            or after.st_ctime_ns != opened.st_ctime_ns
            or after.st_nlink != 1
        ):
            raise StageError(f"{label} changed while it was read")
    finally:
        os.close(descriptor)
    return _parse_json(b"".join(chunks), label)


def _plain_artifact_map(
    artifacts: Mapping[str, Mapping[str, object]],
) -> dict[str, dict[str, object]]:
    return {
        name: {"sha256": record["sha256"], "size": record["size"]}
        for name, record in artifacts.items()
    }


def _validate_artifact_name(name: object) -> str:
    if type(name) is not str or not name or len(name.encode("utf-8")) > 240:
        raise StageError("profile contains an invalid artifact name")
    posix = PurePosixPath(name)
    if (
        posix.is_absolute()
        or posix.as_posix() != name
        or len(posix.parts) != 1
        or name in {".", ".."}
        or "\\" in name
        or name.startswith(".")
    ):
        raise StageError(f"profile contains an unsafe artifact path: {name!r}")
    if name.casefold().endswith(FORBIDDEN_ARTIFACT_SUFFIXES):
        raise StageError(f"profile attempts to stage executable code: {name}")
    return name


def _validate_artifact_map(value: object) -> dict[str, dict[str, object]]:
    if type(value) is not dict or not value:
        raise StageError("deployment profile artifacts must be a non-empty object")
    result: dict[str, dict[str, object]] = {}
    collision_keys: set[str] = set()
    for raw_name, raw_record in value.items():
        name = _validate_artifact_name(raw_name)
        collision = name.casefold()
        if collision in collision_keys:
            raise StageError(f"profile contains a case-colliding artifact: {name}")
        collision_keys.add(collision)
        if type(raw_record) is not dict or set(raw_record) != {"size", "sha256"}:
            raise StageError(f"profile artifact record is invalid: {name}")
        size = raw_record.get("size")
        digest = raw_record.get("sha256")
        if type(size) is not int or size <= 0 or size > HARD_MAX_TOTAL_BYTES:
            raise StageError(f"profile artifact size is invalid: {name}")
        if type(digest) is not str or not SHA256.fullmatch(digest):
            raise StageError(f"profile artifact SHA-256 is invalid: {name}")
        result[name] = {"size": size, "sha256": digest}
    if sum(record["size"] for record in result.values()) > HARD_MAX_TOTAL_BYTES:
        raise StageError("deployment profile exceeds the hard bundle byte cap")
    return result


def _validate_source_lock(
    lock: dict[str, object],
    profile_artifacts: Mapping[str, Mapping[str, object]],
    contract: ProfileContract,
) -> None:
    if lock.get("format") != SOURCE_LOCK_FORMAT:
        raise StageError(f"source lock format must be {SOURCE_LOCK_FORMAT}")
    content_digest = lock.get("content_sha256")
    if type(content_digest) is not str or not SHA256.fullmatch(content_digest):
        raise StageError("source lock content_sha256 is invalid")
    body = dict(lock)
    del body["content_sha256"]
    if _sha256_bytes(_canonical_bytes(body)) != content_digest:
        raise StageError("source lock content hash mismatch")
    if content_digest != contract.source_lock_content_sha256:
        raise StageError("source lock is not the pinned Qwen3.5 source lock")
    if (
        lock.get("repo_id") != contract.repo_id
        or lock.get("requested_revision") != contract.resolved_commit
        or lock.get("resolved_commit") != contract.resolved_commit
    ):
        raise StageError("source lock model identity is not the pinned Qwen3.5 commit")

    if lock.get("policy_receipt") != {
        "metadata_only": True,
        "weight_payload_bytes_downloaded": 0,
        "remote_code_executed": False,
        "hf_token_read": False,
    }:
        raise StageError("source lock policy receipt is unsafe")
    source = lock.get("source")
    if type(source) is not dict or (
        source.get("url") != f"https://huggingface.co/{contract.repo_id}"
        or source.get("private") is not False
        or source.get("gated") is not False
        or source.get("disabled") is not False
    ):
        raise StageError("source lock does not describe an open public model")
    security = lock.get("security")
    if type(security) is not dict or security != {
        "remote_code_indicators": {"auto_map_keys": [], "python_files": []},
        "unsafe_weight_files": [],
        "safetensors_only_plan": True,
    }:
        raise StageError("source lock permits remote code or unsafe weight formats")

    weights = lock.get("weights")
    if type(weights) is not dict or weights.get("format") != "safetensors":
        raise StageError("source lock weight plan is not SafeTensors-only")
    weight_records = weights.get("files")
    if type(weight_records) is not list or not weight_records:
        raise StageError("source lock contains no SafeTensors shards")
    expected_weight_names = {
        name for name in profile_artifacts if name.endswith(".safetensors")
    }
    observed_weight_names: set[str] = set()
    observed_weight_total = 0
    for record in weight_records:
        if type(record) is not dict or set(record) != {
            "path",
            "size",
            "sha256",
            "git_blob_sha1",
        }:
            raise StageError("source lock contains an invalid weight record")
        name = _validate_artifact_name(record.get("path"))
        if name in observed_weight_names or name not in expected_weight_names:
            raise StageError(f"source lock contains an unexpected weight shard: {name}")
        if (
            record.get("size") != profile_artifacts[name]["size"]
            or record.get("sha256") != profile_artifacts[name]["sha256"]
            or type(record.get("git_blob_sha1")) is not str
            or not SHA1.fullmatch(record["git_blob_sha1"])
        ):
            raise StageError(f"source lock disagrees with profile artifact: {name}")
        observed_weight_names.add(name)
        observed_weight_total += record["size"]
    if observed_weight_names != expected_weight_names:
        raise StageError("source lock does not bind every profile SafeTensors shard")
    if weights.get("total_bytes") != observed_weight_total:
        raise StageError("source lock SafeTensors byte total is inconsistent")
    if weights.get("index_file") != "model.safetensors.index.json":
        raise StageError("source lock does not bind the expected SafeTensors index")

    metadata = lock.get("metadata")
    if type(metadata) is not dict or type(metadata.get("files")) is not list:
        raise StageError("source lock metadata records are invalid")
    metadata_by_name: dict[str, dict[str, object]] = {}
    for record in metadata["files"]:
        if type(record) is not dict:
            raise StageError("source lock contains an invalid metadata record")
        name = record.get("path")
        if type(name) is not str or name in metadata_by_name:
            raise StageError("source lock contains duplicate or invalid metadata paths")
        metadata_by_name[name] = record
    for name in SOURCE_LOCK_METADATA_ARTIFACTS:
        record = metadata_by_name.get(name)
        if record is None:
            raise StageError(f"source lock does not bind required metadata: {name}")
        if (
            record.get("size") != profile_artifacts[name]["size"]
            or record.get("sha256") != profile_artifacts[name]["sha256"]
            or type(record.get("git_blob_sha1")) is not str
            or not SHA1.fullmatch(record["git_blob_sha1"])
        ):
            raise StageError(f"source lock disagrees with profile metadata: {name}")


def validate_profile_and_source_lock(
    profile: dict[str, object],
    source_lock: dict[str, object],
    *,
    contract: ProfileContract = QWEN_CONTRACT,
) -> BundlePlan:
    if set(profile) != PROFILE_KEYS or profile.get("format") != PROFILE_FORMAT:
        raise StageError("deployment profile schema or format is invalid")
    if profile.get("profile_id") != contract.profile_id:
        raise StageError("deployment profile is not the pinned Qwen3.5 profile")
    source = profile.get("source")
    expected_source = dict(contract.source)
    if (
        type(source) is not dict
        or set(source) != PROFILE_SOURCE_KEYS
        or source != expected_source
    ):
        raise StageError("deployment profile source identity is not pinned")
    profile_artifacts = _validate_artifact_map(profile.get("artifacts"))
    expected_artifacts = _plain_artifact_map(contract.artifacts)
    if profile_artifacts != expected_artifacts:
        raise StageError("deployment profile artifact allowlist is not pinned")
    _validate_source_lock(source_lock, profile_artifacts, contract)
    artifact_manifest_sha256 = _sha256_bytes(_canonical_bytes(profile_artifacts))
    artifacts = tuple(
        Artifact(name, record["size"], record["sha256"])
        for name, record in sorted(profile_artifacts.items())
    )
    return BundlePlan(
        profile_id=contract.profile_id,
        repo_id=contract.repo_id,
        resolved_commit=contract.resolved_commit,
        source_lock_content_sha256=contract.source_lock_content_sha256,
        artifact_manifest_sha256=artifact_manifest_sha256,
        artifacts=artifacts,
    )


def load_fixed_plan(profile_path: Path, source_lock_path: Path) -> BundlePlan:
    profile = _read_json_regular(profile_path, "checked-in deployment profile")
    source_lock = _read_json_regular(source_lock_path, "pinned source lock")
    return validate_profile_and_source_lock(profile, source_lock)


def _host_matches_suffix(host: str, suffix: str) -> bool:
    return host == suffix or host.endswith(f".{suffix}")


def _validate_download_url(url: str, *, label: str) -> None:
    try:
        parsed = urlparse(url)
        port = parsed.port
    except ValueError as error:
        raise StageError(f"{label} is not a valid URL") from error
    host = (parsed.hostname or "").rstrip(".").lower()
    if (
        parsed.scheme != "https"
        or not any(
            _host_matches_suffix(host, suffix)
            for suffix in ALLOWED_DOWNLOAD_DOMAIN_SUFFIXES
        )
        or parsed.username is not None
        or parsed.password is not None
        or port not in {None, 443}
        or parsed.fragment
    ):
        raise StageError(f"{label} left approved Hugging Face HTTPS domains")


class _HuggingFaceContentRedirectHandler(HTTPRedirectHandler):
    """Validate a redirect target before urllib can issue the next request."""

    def redirect_request(
        self,
        req: Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> Request | None:
        target = urljoin(req.full_url, newurl)
        _validate_download_url(target, label="download redirect")
        return super().redirect_request(req, fp, code, msg, headers, target)


def _build_http_opener() -> Any:
    # ProxyHandler({}) suppresses environment and macOS system proxy discovery.
    return build_opener(ProxyHandler({}), _HuggingFaceContentRedirectHandler())


def _artifact_url(plan: BundlePlan, artifact: Artifact) -> str:
    repo = "/".join(quote(part, safe="") for part in plan.repo_id.split("/"))
    filename = quote(artifact.path, safe="")
    url = f"https://huggingface.co/{repo}/resolve/{plan.resolved_commit}/{filename}"
    _validate_download_url(url, label="artifact URL")
    return url


def _canonical_model_dir(declared: Path) -> Path:
    expanded = declared.expanduser()
    if not expanded.is_absolute():
        raise StageError("--model-dir must be an absolute path")
    if ".." in expanded.parts or expanded.name in {"", ".", ".."}:
        raise StageError("--model-dir must be a canonical absolute path")
    try:
        declared_parent = expanded.parent
        parent = declared_parent.resolve(strict=True)
        parent_info = parent.lstat()
    except OSError as error:
        raise StageError(
            f"cannot inspect --model-dir parent: {error.strerror or error}"
        ) from error
    if stat.S_ISLNK(parent_info.st_mode) or not stat.S_ISDIR(parent_info.st_mode):
        raise StageError("--model-dir parent must be a regular non-symlink directory")
    if declared_parent != parent:
        raise StageError("--model-dir parent must not traverse a symlink")
    return parent / expanded.name


def _staging_path(target: Path) -> Path:
    return target.parent / f".{target.name}.apxinf-staging"


def _lock_path(target: Path) -> Path:
    return target.parent / f".{target.name}.apxinf-stage.lock"


@contextmanager
def _exclusive_stage_lock(target: Path) -> Iterator[None]:
    """Coordinate cooperating same-UID processes across inspect and publish.

    flock is advisory on macOS.  It prevents two instances of this stager from
    appending to one partial file, but is not an adversarial same-UID boundary.
    """

    path = _lock_path(target)
    flags = os.O_RDWR | os.O_CREAT
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, 0o600)
    except OSError as error:
        raise StageError(
            f"cannot open staging lock: {error.strerror or error}"
        ) from error
    try:
        info = os.fstat(descriptor)
        try:
            path_info = path.lstat()
        except OSError as error:
            raise StageError(
                f"cannot inspect staging lock: {error.strerror or error}"
            ) from error
        if (
            stat.S_ISLNK(path_info.st_mode)
            or not stat.S_ISREG(info.st_mode)
            or info.st_dev != path_info.st_dev
            or info.st_ino != path_info.st_ino
            or info.st_nlink != 1
            or info.st_uid != os.getuid()
            or stat.S_IMODE(info.st_mode) != 0o600
        ):
            raise StageError(
                "staging lock must be an owner-only, single-link regular file"
            )
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise StageError(
                "another bundle staging process holds the destination lock"
            ) from error
        # Revalidate after acquisition so a same-user pathname swap cannot make
        # the receipt appear to lock a different inode.
        locked_path_info = path.lstat()
        if (
            locked_path_info.st_dev != info.st_dev
            or locked_path_info.st_ino != info.st_ino
            or locked_path_info.st_nlink != 1
        ):
            raise StageError("staging lock path changed during acquisition")
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        except OSError:
            pass
        os.close(descriptor)


def _private_directory(path: Path, *, create: bool) -> bool:
    if not os.path.lexists(path):
        if not create:
            return False
        try:
            os.mkdir(path, 0o700)
        except FileExistsError:
            pass
        except OSError as error:
            raise StageError(
                f"cannot create private staging directory: {error.strerror or error}"
            ) from error
    try:
        info = path.lstat()
    except OSError as error:
        raise StageError(
            f"cannot inspect private staging directory: {error.strerror or error}"
        ) from error
    if (
        stat.S_ISLNK(info.st_mode)
        or not stat.S_ISDIR(info.st_mode)
        or info.st_uid != os.getuid()
        or stat.S_IMODE(info.st_mode) != 0o700
    ):
        raise StageError(
            "staging path must be an owner-only regular non-symlink directory"
        )
    if info.st_dev != path.parent.lstat().st_dev:
        raise StageError("staging directory is not on the destination filesystem")
    return True


def _inspect_private_file(path: Path, label: str) -> os.stat_result:
    try:
        info = path.lstat()
    except OSError as error:
        raise StageError(
            f"cannot inspect {label}: {error.strerror or error}"
        ) from error
    if (
        stat.S_ISLNK(info.st_mode)
        or not stat.S_ISREG(info.st_mode)
        or info.st_nlink != 1
        or info.st_uid != os.getuid()
        or stat.S_IMODE(info.st_mode) != 0o600
    ):
        raise StageError(
            f"{label} must be an owner-only, single-link regular non-symlink file"
        )
    return info


def _hash_private_file(path: Path, label: str) -> tuple[int, str]:
    before = _inspect_private_file(path, label)
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise StageError(f"cannot open {label}: {error.strerror or error}") from error
    try:
        opened = os.fstat(descriptor)
        if (
            opened.st_dev != before.st_dev
            or opened.st_ino != before.st_ino
            or opened.st_nlink != 1
        ):
            raise StageError(f"{label} changed while it was opened")
        digest = hashlib.sha256()
        total = 0
        while True:
            chunk = os.read(descriptor, READ_CHUNK_BYTES)
            if not chunk:
                break
            digest.update(chunk)
            total += len(chunk)
        after = os.fstat(descriptor)
        if (
            total != opened.st_size
            or after.st_size != opened.st_size
            or after.st_mtime_ns != opened.st_mtime_ns
            or after.st_ctime_ns != opened.st_ctime_ns
            or after.st_nlink != 1
        ):
            raise StageError(f"{label} changed while it was hashed")
        return total, digest.hexdigest()
    finally:
        os.close(descriptor)


def _truncate_private_partial(
    path: Path, *, expected_identity: os.stat_result | None = None
) -> int:
    """Reset one stager-owned partial after fd-level identity validation.

    Callers must hold the adjacent cooperative lock.  Only a private 0600,
    owner-owned, single-link regular file is mutable; every other type fails.
    """

    before = _inspect_private_file(path, "recoverable partial artifact")
    if expected_identity is not None and (
        before.st_dev != expected_identity.st_dev
        or before.st_ino != expected_identity.st_ino
    ):
        raise StageError("recoverable partial changed before reset")
    flags = os.O_RDWR
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise StageError(
            f"cannot open recoverable partial: {error.strerror or error}"
        ) from error
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_dev != before.st_dev
            or opened.st_ino != before.st_ino
            or opened.st_nlink != 1
            or opened.st_uid != os.getuid()
            or stat.S_IMODE(opened.st_mode) != 0o600
        ):
            raise StageError("recoverable partial changed while it was opened")
        discarded = opened.st_size
        os.ftruncate(descriptor, 0)
        os.fsync(descriptor)
        after = os.fstat(descriptor)
        if after.st_size != 0 or after.st_nlink != 1:
            raise StageError("recoverable partial did not reset safely")
        return discarded
    finally:
        os.close(descriptor)


def _reasonable_existing_directory(info: os.stat_result, label: str) -> None:
    mode = stat.S_IMODE(info.st_mode)
    if (
        not stat.S_ISDIR(info.st_mode)
        or info.st_uid != os.getuid()
        or mode & 0o7000
        or mode & 0o002
        or mode & 0o500 != 0o500
    ):
        raise StageError(
            f"{label} must be an owner-readable/executable, non-world-writable directory"
        )


def _reasonable_existing_artifact(info: os.stat_result, label: str) -> None:
    mode = stat.S_IMODE(info.st_mode)
    if (
        not stat.S_ISREG(info.st_mode)
        or info.st_nlink != 1
        or info.st_uid != os.getuid()
        or mode & ~0o666
        or mode & 0o022
        or mode & 0o400 == 0
    ):
        raise StageError(
            f"{label} must be an owner-readable, single-link, non-executable file "
            "with no group/other write permission"
        )


def _hash_existing_artifact(path: Path, artifact: Artifact) -> tuple[int, str]:
    label = f"existing model artifact {artifact.path}"
    try:
        before = path.lstat()
    except OSError as error:
        raise StageError(
            f"cannot inspect {label}: {error.strerror or error}"
        ) from error
    if stat.S_ISLNK(before.st_mode):
        raise StageError(f"{label} must not be a symlink")
    _reasonable_existing_artifact(before, label)
    if before.st_size != artifact.size:
        raise StageError(f"{label} size does not match the pinned profile")
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise StageError(f"cannot open {label}: {error.strerror or error}") from error
    try:
        opened = os.fstat(descriptor)
        _reasonable_existing_artifact(opened, label)
        if opened.st_dev != before.st_dev or opened.st_ino != before.st_ino:
            raise StageError(f"{label} changed while it was opened")
        digest = hashlib.sha256()
        total = 0
        while total < artifact.size:
            chunk = os.read(descriptor, min(READ_CHUNK_BYTES, artifact.size - total))
            if not chunk:
                raise StageError(f"{label} ended before its pinned size")
            digest.update(chunk)
            total += len(chunk)
        if os.read(descriptor, 1):
            raise StageError(f"{label} exceeds its pinned size")
        after = os.fstat(descriptor)
        if (
            after.st_size != opened.st_size
            or after.st_mtime_ns != opened.st_mtime_ns
            or after.st_ctime_ns != opened.st_ctime_ns
            or after.st_nlink != 1
        ):
            raise StageError(f"{label} changed while it was hashed")
        observed = digest.hexdigest()
        if observed != artifact.sha256:
            raise StageError(f"{label} SHA-256 mismatch")
        return total, observed
    finally:
        os.close(descriptor)


def _cache_file_has_shebang(path: Path, before: os.stat_result) -> bool:
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_dev != before.st_dev
            or opened.st_ino != before.st_ino
            or opened.st_nlink != 1
        ):
            raise StageError("model .cache entry changed while it was opened")
        return os.read(descriptor, 2) == b"#!"
    finally:
        os.close(descriptor)


def _validate_optional_cache_tree(cache: Path) -> tuple[int, int]:
    try:
        root_info = cache.lstat()
    except OSError as error:
        raise StageError(
            f"cannot inspect model .cache: {error.strerror or error}"
        ) from error
    if stat.S_ISLNK(root_info.st_mode):
        raise StageError("model .cache must not be a symlink")
    _reasonable_existing_directory(root_info, "model .cache")
    pending = [cache]
    entry_count = 0
    total_bytes = 0
    while pending:
        directory = pending.pop()
        try:
            entries = list(os.scandir(directory))
        except OSError as error:
            raise StageError(
                f"cannot inspect model .cache: {error.strerror or error}"
            ) from error
        for entry in entries:
            entry_count += 1
            if entry_count > MAX_CACHE_ENTRIES:
                raise StageError("model .cache exceeds the safety entry cap")
            path = Path(entry.path)
            try:
                info = entry.stat(follow_symlinks=False)
            except OSError as error:
                raise StageError(
                    f"cannot inspect model .cache entry: {error.strerror or error}"
                ) from error
            if stat.S_ISLNK(info.st_mode):
                raise StageError("model .cache contains a symlink")
            if stat.S_ISDIR(info.st_mode):
                _reasonable_existing_directory(info, "model .cache directory")
                pending.append(path)
                continue
            mode = stat.S_IMODE(info.st_mode)
            if (
                not stat.S_ISREG(info.st_mode)
                or info.st_nlink != 1
                or info.st_uid != os.getuid()
                or mode & 0o7000
                or mode & 0o002
                or mode & 0o111
                or mode & 0o400 == 0
            ):
                raise StageError(
                    "model .cache contains an unsafe, linked, or executable entry"
                )
            if info.st_size > MAX_CACHE_FILE_BYTES:
                raise StageError("model .cache file exceeds the safety byte cap")
            total_bytes += info.st_size
            if total_bytes > MAX_CACHE_TOTAL_BYTES:
                raise StageError("model .cache exceeds the total safety byte cap")
            if path.name.casefold().endswith(FORBIDDEN_CACHE_SCRIPT_SUFFIXES):
                raise StageError("model .cache contains a script-like file")
            try:
                if _cache_file_has_shebang(path, info):
                    raise StageError("model .cache contains a shebang script")
            except OSError as error:
                raise StageError(
                    f"cannot read model .cache entry: {error.strerror or error}"
                ) from error
    return entry_count, total_bytes


def _validate_existing_bundle(plan: BundlePlan, target: Path) -> dict[str, object]:
    try:
        root_info = target.lstat()
    except OSError as error:
        raise StageError(
            f"cannot inspect existing --model-dir: {error.strerror or error}"
        ) from error
    if stat.S_ISLNK(root_info.st_mode):
        raise StageError("existing --model-dir must not be a symlink")
    _reasonable_existing_directory(root_info, "existing --model-dir")
    try:
        names = {entry.name for entry in os.scandir(target)}
    except OSError as error:
        raise StageError(
            f"cannot inspect existing --model-dir: {error.strerror or error}"
        ) from error
    expected = {artifact.path for artifact in plan.artifacts}
    allowed = expected | {".cache"}
    missing = sorted(expected - names)
    unexpected = sorted(names - allowed)
    if missing or unexpected:
        raise StageError(
            "existing --model-dir allowlist mismatch "
            f"(missing={missing}, unexpected={unexpected})"
        )
    for artifact in plan.artifacts:
        _hash_existing_artifact(target / artifact.path, artifact)
    cache_present = ".cache" in names
    cache_entries = 0
    cache_bytes = 0
    if cache_present:
        cache_entries, cache_bytes = _validate_optional_cache_tree(target / ".cache")
    return {
        "cache_tree_present": cache_present,
        "cache_entry_count": cache_entries,
        "cache_total_bytes": cache_bytes,
    }


def _inspect_staging(
    plan: BundlePlan, staging: Path, *, repair_corrupt: bool = False
) -> tuple[dict[str, int], dict[str, int], dict[str, int]]:
    if not _private_directory(staging, create=False):
        return {}, {}, {}
    by_name = {artifact.path: artifact for artifact in plan.artifacts}
    allowed_names = set(by_name)
    allowed_names.update(f"{name}.part" for name in by_name)
    try:
        names = {entry.name for entry in os.scandir(staging)}
    except OSError as error:
        raise StageError(
            f"cannot inspect staging contents: {error.strerror or error}"
        ) from error
    unexpected = sorted(names - allowed_names)
    if unexpected:
        raise StageError(
            f"staging directory contains an unexpected entry: {unexpected[0]}"
        )
    completed: dict[str, int] = {}
    partial: dict[str, int] = {}
    recovered: dict[str, int] = {}
    for artifact in plan.artifacts:
        final_path = staging / artifact.path
        part_path = staging / f"{artifact.path}.part"
        final_exists = os.path.lexists(final_path)
        part_exists = os.path.lexists(part_path)
        if final_exists and part_exists:
            raise StageError(
                f"staging contains both complete and partial copies: {artifact.path}"
            )
        if final_exists:
            info = _inspect_private_file(final_path, f"staged {artifact.path}")
            if info.st_size == artifact.size:
                size, digest = _hash_private_file(final_path, f"staged {artifact.path}")
            else:
                size, digest = info.st_size, ""
            if size == artifact.size and digest == artifact.sha256:
                completed[artifact.path] = size
            elif not repair_corrupt:
                raise StageError(
                    f"completed staged artifact failed integrity: {artifact.path}"
                )
            else:
                _exclusive_rename(final_path, part_path)
                recovered[artifact.path] = _truncate_private_partial(
                    part_path, expected_identity=info
                )
                _fsync_directory(staging)
                partial[artifact.path] = 0
        elif part_exists:
            info = _inspect_private_file(part_path, f"partial {artifact.path}")
            if info.st_size > artifact.size:
                raise StageError(
                    f"partial artifact exceeds its pinned size: {artifact.path}"
                )
            if info.st_size == artifact.size:
                size, digest = _hash_private_file(part_path, f"partial {artifact.path}")
                if size == artifact.size and digest != artifact.sha256:
                    if not repair_corrupt:
                        raise StageError(
                            f"full partial artifact failed integrity: {artifact.path}"
                        )
                    recovered[artifact.path] = _truncate_private_partial(
                        part_path, expected_identity=info
                    )
                    partial[artifact.path] = 0
                    continue
            partial[artifact.path] = info.st_size
    return completed, partial, recovered


def _open_partial(path: Path, *, existing_size: int) -> int:
    flags = os.O_RDWR
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    if existing_size == 0 and not os.path.lexists(path):
        flags |= os.O_CREAT | os.O_EXCL
    try:
        descriptor = os.open(path, flags, 0o600)
    except OSError as error:
        raise StageError(
            f"cannot open partial artifact: {error.strerror or error}"
        ) from error
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or opened.st_uid != os.getuid()
            or stat.S_IMODE(opened.st_mode) != 0o600
            or opened.st_size != existing_size
        ):
            raise StageError("partial artifact changed while it was opened")
    except Exception:
        os.close(descriptor)
        raise
    return descriptor


def _single_critical_header(headers: Any, name: str) -> str | None:
    get_all = getattr(headers, "get_all", None)
    if callable(get_all):
        values = get_all(name, [])
    else:
        value = headers.get(name)
        values = [] if value is None else [value]
    if len(values) > 1:
        raise StageError(f"download response contains duplicate {name}")
    if not values:
        return None
    value = values[0]
    if type(value) is not str or len(value.encode("latin-1", errors="replace")) > (
        MAX_CRITICAL_HEADER_BYTES
    ):
        raise StageError(f"download response has an invalid or oversized {name}")
    return value


def _parse_nonnegative_header(value: str | None, label: str) -> int:
    if (
        value is None
        or len(value) > MAX_NUMERIC_HEADER_DIGITS
        or not re.fullmatch(r"0|[1-9][0-9]*", value)
    ):
        raise StageError(f"download response has invalid or missing {label}")
    return int(value)


def _validate_response(response: Any, *, artifact: Artifact, start: int) -> int:
    try:
        response_url = response.geturl()
    except Exception as error:
        raise StageError("download response did not expose its final URL") from error
    _validate_download_url(response_url, label="download response URL")
    status = getattr(response, "status", None)
    if status is None:
        status = response.getcode()
    expected_remaining = artifact.size - start
    headers = response.headers
    transfer_encoding = _single_critical_header(headers, "Transfer-Encoding")
    if transfer_encoding is not None:
        raise StageError("download response must not use Transfer-Encoding")
    content_encoding = _single_critical_header(headers, "Content-Encoding")
    if content_encoding not in {None, "", "identity"}:
        raise StageError("download response used a transformed content encoding")
    declared_length = _parse_nonnegative_header(
        _single_critical_header(headers, "Content-Length"), "Content-Length"
    )
    if declared_length != expected_remaining:
        raise StageError(
            "download Content-Length does not match the pinned artifact size"
        )
    content_range = _single_critical_header(headers, "Content-Range")
    if start == 0:
        if status != 200 or content_range is not None:
            raise StageError("initial download must be an exact HTTP 200 response")
    else:
        match = CONTENT_RANGE.fullmatch(content_range or "")
        if (
            status != 206
            or match is None
            or any(len(group) > MAX_NUMERIC_HEADER_DIGITS for group in match.groups())
            or int(match.group(1)) != start
            or int(match.group(2)) != artifact.size - 1
            or int(match.group(3)) != artifact.size
        ):
            raise StageError("resume response has an invalid Content-Range")
    return expected_remaining


def _remaining_seconds(deadline: float) -> float:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise StageError("bundle staging exceeded --timeout-seconds")
    return remaining


def _download_artifact(
    plan: BundlePlan,
    artifact: Artifact,
    staging: Path,
    *,
    start: int,
    opener: Any,
    deadline: float,
) -> int:
    part_path = staging / f"{artifact.path}.part"
    descriptor = _open_partial(part_path, existing_size=start)
    try:
        digest = hashlib.sha256()
        os.lseek(descriptor, 0, os.SEEK_SET)
        remaining_prefix = start
        while remaining_prefix:
            chunk = os.read(descriptor, min(READ_CHUNK_BYTES, remaining_prefix))
            if not chunk:
                raise StageError(f"partial artifact ended early: {artifact.path}")
            digest.update(chunk)
            remaining_prefix -= len(chunk)
        os.lseek(descriptor, start, os.SEEK_SET)

        headers = {
            "Accept": "application/octet-stream",
            "Accept-Encoding": "identity",
            "User-Agent": "ApxInf-macOS-bundle-stager/1",
        }
        if start:
            headers["Range"] = f"bytes={start}-"
        request = Request(_artifact_url(plan, artifact), headers=headers, method="GET")
        try:
            response = opener.open(
                request, timeout=min(_remaining_seconds(deadline), 60.0)
            )
        except (
            HTTPError,
            URLError,
            HTTPException,
            IncompleteRead,
            OSError,
            TimeoutError,
        ) as error:
            raise StageError(f"download request failed for {artifact.path}") from error
        try:
            expected_remaining = _validate_response(
                response, artifact=artifact, start=start
            )
            downloaded = 0
            while downloaded < expected_remaining:
                _remaining_seconds(deadline)
                try:
                    chunk = response.read(
                        min(READ_CHUNK_BYTES, expected_remaining - downloaded + 1)
                    )
                except (
                    HTTPException,
                    IncompleteRead,
                    OSError,
                    TimeoutError,
                ) as error:
                    raise StageError(
                        f"download read failed for {artifact.path}"
                    ) from error
                if not chunk:
                    break
                downloaded += len(chunk)
                if downloaded > expected_remaining:
                    raise StageError(f"download exceeded pinned size: {artifact.path}")
                digest.update(chunk)
                view = memoryview(chunk)
                while view:
                    written = os.write(descriptor, view)
                    if written <= 0:
                        raise StageError(
                            f"cannot write partial artifact: {artifact.path}"
                        )
                    view = view[written:]
            if downloaded != expected_remaining:
                raise StageError(f"download ended before pinned size: {artifact.path}")
            try:
                trailing = response.read(1)
            except (
                HTTPException,
                IncompleteRead,
                OSError,
                TimeoutError,
            ) as error:
                raise StageError(f"download read failed for {artifact.path}") from error
            if trailing:
                raise StageError(f"download exceeded pinned size: {artifact.path}")
        finally:
            try:
                response.close()
            except (HTTPException, OSError) as error:
                raise StageError(
                    f"download response close failed for {artifact.path}"
                ) from error
        os.fsync(descriptor)
        after = os.fstat(descriptor)
        if after.st_size != artifact.size or after.st_nlink != 1:
            raise StageError(f"staged artifact size changed: {artifact.path}")
        if digest.hexdigest() != artifact.sha256:
            raise ArtifactIntegrityError(
                f"staged artifact SHA-256 mismatch: {artifact.path}",
                downloaded_bytes=downloaded,
            )
        return downloaded
    finally:
        os.close(descriptor)


def _exclusive_rename(source: Path, destination: Path) -> None:
    """Atomically rename without ever replacing an existing destination."""

    if sys.platform != "darwin":
        raise StageError("exclusive bundle publication requires macOS renamex_np")
    libc = ctypes.CDLL(None, use_errno=True)
    renamex_np = libc.renamex_np
    renamex_np.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
    renamex_np.restype = ctypes.c_int
    result = renamex_np(
        os.fsencode(source),
        os.fsencode(destination),
        0x00000004,  # RENAME_EXCL from <sys/stdio.h>
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise StageError("refusing to overwrite an existing destination")
    raise StageError(f"exclusive atomic rename failed: {os.strerror(error_number)}")


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _artifact_receipts(plan: BundlePlan) -> list[dict[str, object]]:
    return [
        {"path": artifact.path, "sha256": artifact.sha256, "size": artifact.size}
        for artifact in plan.artifacts
    ]


def _base_receipt(
    plan: BundlePlan,
    target: Path,
    *,
    opener_injected: bool,
    existing_only: bool,
) -> dict[str, object]:
    return {
        "format": RECEIPT_FORMAT,
        "passed": True,
        "profile_id": plan.profile_id,
        "repo_id": plan.repo_id,
        "resolved_commit": plan.resolved_commit,
        "source_lock_content_sha256": plan.source_lock_content_sha256,
        "artifact_manifest_sha256": plan.artifact_manifest_sha256,
        "model_dir": str(target),
        "artifacts": _artifact_receipts(plan),
        "total_bytes": plan.total_bytes,
        "policy": {
            "network": {
                "https_only": True,
                "approved_domain_suffixes": list(ALLOWED_DOWNLOAD_DOMAIN_SUFFIXES),
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
            "operation": {"existing_only_requested": existing_only},
            "recovery": {"max_restart_from_zero_per_artifact": 1},
        },
        "evidence": {
            "builtin_opener": False,
            "opener_injected": opener_injected,
            "ambient_proxy_disabled": False,
            "authorization_header_omitted": False,
            "lock_acquired": False,
            "network_used": False,
            "network_request_count": 0,
            "existing_bundle_verified": False,
            "existing_only_enforced": existing_only,
            "cache_tree_present": False,
            "cache_entry_count": 0,
            "cache_total_bytes": 0,
            "published_by_this_invocation": False,
            "atomic_no_replace_publish_observed": False,
            "recovered_artifacts": [],
            "recovery_bytes_discarded": 0,
        },
    }


def stage_bundle(
    plan: BundlePlan,
    model_dir: Path,
    *,
    dry_run: bool = False,
    existing_only: bool = False,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    max_total_bytes: int = HARD_MAX_TOTAL_BYTES,
    opener: Any | None = None,
) -> dict[str, object]:
    if type(existing_only) is not bool:
        raise StageError("existing_only must be a boolean")
    if (
        type(timeout_seconds) is not int
        or timeout_seconds <= 0
        or timeout_seconds > MAX_TIMEOUT_SECONDS
    ):
        raise StageError(f"--timeout-seconds must be in 1..{MAX_TIMEOUT_SECONDS}")
    if (
        type(max_total_bytes) is not int
        or max_total_bytes <= 0
        or max_total_bytes > HARD_MAX_TOTAL_BYTES
    ):
        raise StageError(f"--max-total-bytes must be in 1..{HARD_MAX_TOTAL_BYTES}")
    if plan.total_bytes > max_total_bytes:
        raise StageError("pinned bundle exceeds --max-total-bytes")
    target = _canonical_model_dir(model_dir)
    target_exists = os.path.lexists(target)
    if existing_only and not target_exists:
        raise StageError("--existing-only requires an existing --model-dir")
    receipt = _base_receipt(
        plan,
        target,
        opener_injected=opener is not None,
        existing_only=existing_only,
    )
    if dry_run:
        if target_exists:
            existing = _validate_existing_bundle(plan, target)
            completed = {artifact.path: artifact.size for artifact in plan.artifacts}
            partial: dict[str, int] = {}
            receipt["evidence"].update(existing)
            receipt["evidence"]["existing_bundle_verified"] = True
            would_action = "reused-existing"
        else:
            staging = _staging_path(target)
            completed, partial, _ = _inspect_staging(plan, staging)
            would_action = "publish-new"
        receipt.update(
            {
                "action": "dry-run",
                "would_action": would_action,
                "published": False,
                "downloaded_bytes": 0,
                "resumed_from_bytes": sum(partial.values()),
                "reused_bytes": sum(completed.values()),
            }
        )
        receipt["would_download_bytes"] = (
            plan.total_bytes - sum(completed.values()) - sum(partial.values())
        )
        return receipt

    with _exclusive_stage_lock(target):
        receipt["evidence"]["lock_acquired"] = True
        target_exists_after_lock = os.path.lexists(target)
        if existing_only and not target_exists_after_lock:
            raise StageError(
                "--existing-only target disappeared before locked validation"
            )
        if target_exists_after_lock:
            existing = _validate_existing_bundle(plan, target)
            receipt["evidence"].update(existing)
            receipt["evidence"]["existing_bundle_verified"] = True
            receipt.update(
                {
                    "action": "reused-existing",
                    "published": True,
                    "downloaded_bytes": 0,
                    "resumed_from_bytes": 0,
                    "reused_bytes": plan.total_bytes,
                }
            )
            return receipt
        staging = _staging_path(target)
        completed, partial, recovered = _inspect_staging(
            plan, staging, repair_corrupt=True
        )
        restarted = set(recovered)
        receipt["evidence"]["recovered_artifacts"] = sorted(recovered)
        receipt["evidence"]["recovery_bytes_discarded"] = sum(recovered.values())
        receipt.update(
            {
                "action": "staged",
                "published": False,
                "downloaded_bytes": 0,
                "resumed_from_bytes": sum(partial.values()),
                "reused_bytes": sum(completed.values()),
            }
        )
        _private_directory(staging, create=True)
        active_opener = opener
        deadline = time.monotonic() + timeout_seconds
        downloaded_bytes = 0
        for artifact in plan.artifacts:
            if artifact.path in completed:
                continue
            part_path = staging / f"{artifact.path}.part"
            final_path = staging / artifact.path
            start = partial.get(artifact.path, 0)
            while start < artifact.size:
                if active_opener is None:
                    active_opener = _build_http_opener()
                    receipt["evidence"]["builtin_opener"] = True
                    receipt["evidence"]["ambient_proxy_disabled"] = True
                receipt["evidence"]["network_used"] = True
                receipt["evidence"]["authorization_header_omitted"] = True
                receipt["evidence"]["network_request_count"] += 1
                try:
                    downloaded_bytes += _download_artifact(
                        plan,
                        artifact,
                        staging,
                        start=start,
                        opener=active_opener,
                        deadline=deadline,
                    )
                    break
                except ArtifactIntegrityError as error:
                    downloaded_bytes += error.downloaded_bytes
                    if artifact.path in restarted:
                        raise StageError(
                            "artifact SHA-256 remained invalid after one safe "
                            f"restart: {artifact.path}"
                        ) from error
                    discarded = _truncate_private_partial(part_path)
                    restarted.add(artifact.path)
                    recovered[artifact.path] = (
                        recovered.get(artifact.path, 0) + discarded
                    )
                    receipt["evidence"]["recovered_artifacts"] = sorted(recovered)
                    receipt["evidence"]["recovery_bytes_discarded"] = sum(
                        recovered.values()
                    )
                    start = 0
            _exclusive_rename(part_path, final_path)
            completed[artifact.path] = artifact.size
        completed_after, partial_after, recovered_after = _inspect_staging(
            plan, staging
        )
        if (
            set(completed_after) != {artifact.path for artifact in plan.artifacts}
            or partial_after
            or recovered_after
        ):
            raise StageError("staging directory is incomplete after download")
        _fsync_directory(staging)
        _exclusive_rename(staging, target)
        _fsync_directory(target.parent)
        receipt["published"] = True
        receipt["downloaded_bytes"] = downloaded_bytes
        receipt["evidence"]["published_by_this_invocation"] = True
        receipt["evidence"]["atomic_no_replace_publish_observed"] = True
        return receipt


def parser() -> argparse.ArgumentParser:
    result = ReceiptArgumentParser(description=__doc__)
    result.add_argument(
        "--profile",
        type=Path,
        default=DEFAULT_PROFILE,
        help="pinned deployment profile",
    )
    result.add_argument(
        "--source-lock",
        type=Path,
        default=DEFAULT_SOURCE_LOCK,
        help="metadata-only source lock",
    )
    result.add_argument(
        "--model-dir", type=Path, required=True, help="new absolute bundle directory"
    )
    result.add_argument(
        "--dry-run", action="store_true", help="validate without network or writes"
    )
    result.add_argument(
        "--existing-only",
        action="store_true",
        help="require and validate an existing bundle; never inspect staging or download",
    )
    result.add_argument("--timeout-seconds", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    result.add_argument("--max-total-bytes", type=int, default=HARD_MAX_TOTAL_BYTES)
    return result


def main(argv: list[str] | None = None) -> int:
    try:
        args = parser().parse_args(argv)
        plan = load_fixed_plan(args.profile.expanduser(), args.source_lock.expanduser())
        receipt = stage_bundle(
            plan,
            args.model_dir,
            dry_run=args.dry_run,
            existing_only=args.existing_only,
            timeout_seconds=args.timeout_seconds,
            max_total_bytes=args.max_total_bytes,
        )
        print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
        return 0
    except KeyboardInterrupt:
        receipt = {
            "format": RECEIPT_FORMAT,
            "passed": False,
            "error": {
                "code": "HF_BUNDLE_STAGE_INTERRUPTED",
                "message": "bundle staging interrupted by user",
            },
        }
        print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
        return 130
    except StageError as error:
        receipt = {
            "format": RECEIPT_FORMAT,
            "passed": False,
            "error": {"code": "HF_BUNDLE_STAGE_FAILED", "message": str(error)},
        }
        print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
        return 2
    except (
        HTTPError,
        URLError,
        HTTPException,
        IncompleteRead,
        TimeoutError,
    ):
        receipt = {
            "format": RECEIPT_FORMAT,
            "passed": False,
            "error": {
                "code": "HF_BUNDLE_STAGE_FAILED",
                "message": "bundle staging transport failed",
            },
        }
        print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
        return 2
    except OSError as error:
        receipt = {
            "format": RECEIPT_FORMAT,
            "passed": False,
            "error": {"code": "HF_BUNDLE_STAGE_FAILED", "message": str(error)},
        }
        print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
        return 2
    except Exception as error:
        receipt = {
            "format": RECEIPT_FORMAT,
            "passed": False,
            "error": {
                "code": "HF_BUNDLE_STAGE_INTERNAL_ERROR",
                "message": f"unexpected internal staging failure ({type(error).__name__})",
            },
        }
        print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
