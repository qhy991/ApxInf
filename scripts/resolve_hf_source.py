#!/usr/bin/env python3
"""Resolve a public Hugging Face model into a bounded, metadata-only source lock.

The resolver never reads an HF token, downloads weight payloads, executes model
repository code, or renders model-card/template text.  Its JSON output is the
trusted input boundary for the later KerSor architecture analysis.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import sys
import tempfile
from typing import Any, Callable
from urllib.parse import quote, urlparse
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

SCRIPT_DIR = str(Path(__file__).resolve().parent)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from prepare_hf_macos_intake import (  # noqa: E402
    IntakeError,
    model_slug,
    parse_model_reference,
)


LOCK_FORMAT = "apxinf-hf-source-lock-v1"
MAX_API_BYTES = 16 * 1024 * 1024
MAX_METADATA_FILE_BYTES = 16 * 1024 * 1024
MAX_METADATA_TOTAL_BYTES = 32 * 1024 * 1024
MAX_CONFIG_KEYS = 4096
MAX_AUTO_MAP_KEYS = 256
MAX_INDICATOR_FILES = 4096
SHA1 = re.compile(r"^[0-9a-f]{40}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
SAFE_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+:/-]{0,255}$")
SAFE_REPO_PATH = re.compile(r"^[A-Za-z0-9._+@-]+(?:/[A-Za-z0-9._+@-]+)*$")
SOURCE_SLUG = re.compile(r"^[a-z0-9][a-z0-9._+-]{0,95}$")
AUTO_MAP_KEY = re.compile(r"^Auto[A-Za-z0-9_]{0,92}$")
PIPELINE_TAG_ALLOWLIST = frozenset(
    {
        "audio-classification",
        "automatic-speech-recognition",
        "feature-extraction",
        "fill-mask",
        "image-classification",
        "image-segmentation",
        "image-text-to-text",
        "image-to-text",
        "object-detection",
        "question-answering",
        "sentence-similarity",
        "summarization",
        "text-classification",
        "text-generation",
        "text-to-image",
        "text2text-generation",
        "token-classification",
        "translation",
        "zero-shot-classification",
    }
)
LIBRARY_NAME_ALLOWLIST = frozenset(
    {
        "adapter-transformers",
        "diffusers",
        "keras",
        "sentence-transformers",
        "spacy",
        "stable-baselines3",
        "timm",
        "transformers",
    }
)
HUGGING_FACE_HOSTS = frozenset({"huggingface.co", "www.huggingface.co"})

METADATA_FILES = {
    "config.json",
    "generation_config.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "preprocessor_config.json",
    "processor_config.json",
    "video_preprocessor_config.json",
    "model.safetensors.index.json",
}
UNSAFE_WEIGHT_SUFFIXES = (
    ".bin",
    ".ckpt",
    ".joblib",
    ".pickle",
    ".pkl",
    ".pt",
    ".pth",
)
STRUCTURAL_STRING_KEYS = {
    "architectures",
    "attention_type",
    "dtype",
    "hidden_act",
    "image_processor_type",
    "layer_types",
    "model_type",
    "processor_class",
    "rope_type",
    "tokenizer_class",
    "torch_dtype",
    "transformers_version",
}
SOURCE_KEYS = frozenset(
    {"url", "private", "gated", "disabled", "license", "pipeline_tag", "library_name"}
)
ARCHITECTURE_KEYS = frozenset(
    {"config_sha256", "config_keys", "structural_config", "tokenizer"}
)
SECURITY_KEYS = frozenset(
    {"remote_code_indicators", "unsafe_weight_files", "safetensors_only_plan"}
)
REMOTE_CODE_KEYS = frozenset({"auto_map_keys", "python_files"})


class SourceLockError(ValueError):
    """Raised when deterministic source resolution must fail closed."""


def _optional_source_slug(value: object, *, field: str) -> str | None:
    if value is None:
        return None
    if type(value) is not str or not SOURCE_SLUG.fullmatch(value):
        raise SourceLockError(f"Hub {field} is not a canonical metadata slug")
    return value


def _allowlisted_source_slug(value: object, *, field: str) -> str | None:
    slug = _optional_source_slug(value, field=field)
    if slug is None:
        return None
    allowlist = (
        PIPELINE_TAG_ALLOWLIST if field == "pipeline_tag" else LIBRARY_NAME_ALLOWLIST
    )
    return slug if slug in allowlist else None


def _gated_value(value: object) -> bool | str:
    if type(value) is bool or (
        type(value) is str and value in {"auto", "manual"}
    ):
        return value
    raise SourceLockError("Hub gated metadata is not an allowed policy value")


def _canonical_mapping_keys(
    value: object,
    *,
    field: str,
    maximum: int,
    pattern: re.Pattern[str] = SAFE_IDENTIFIER,
) -> list[str]:
    if type(value) is not dict:
        raise SourceLockError(f"{field} must be a JSON object")
    if len(value) > maximum:
        raise SourceLockError(f"{field} exceeds the key-count policy")
    keys = sorted(value)
    if any(type(key) is not str or not pattern.fullmatch(key) for key in keys):
        raise SourceLockError(f"{field} contains a non-canonical key")
    return keys


def _validate_canonical_string_list(
    value: object,
    *,
    field: str,
    maximum: int,
    pattern: re.Pattern[str],
) -> list[str]:
    if type(value) is not list or len(value) > maximum:
        raise SourceLockError(f"{field} is not a bounded array")
    if any(type(item) is not str or not pattern.fullmatch(item) for item in value):
        raise SourceLockError(f"{field} contains a non-canonical identifier")
    if value != sorted(set(value)):
        raise SourceLockError(f"{field} must be sorted and unique")
    return value


def _validate_repo_path_list(
    value: object, *, field: str, maximum: int
) -> list[str]:
    if type(value) is not list or len(value) > maximum:
        raise SourceLockError(f"{field} is not a bounded array")
    if any(type(item) is not str for item in value):
        raise SourceLockError(f"{field} contains a non-string path")
    for item in value:
        safe_repo_path(item)
    if value != sorted(set(value)):
        raise SourceLockError(f"{field} must be sorted and unique")
    return value


def _validate_hugging_face_url(url: str, *, label: str) -> None:
    try:
        parsed = urlparse(url)
        port = parsed.port
    except ValueError as error:
        raise SourceLockError(f"{label} is not a valid Hugging Face URL: {url}") from error
    if (
        parsed.scheme != "https"
        or (parsed.hostname or "").lower() not in HUGGING_FACE_HOSTS
        or parsed.username is not None
        or parsed.password is not None
        or port not in {None, 443}
        or parsed.fragment
    ):
        raise SourceLockError(f"{label} left HTTPS huggingface.co: {url}")


class _HuggingFaceRedirectHandler(HTTPRedirectHandler):
    """Reject redirects before urllib can issue an off-domain request."""

    def redirect_request(
        self,
        req: Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> Request | None:
        _validate_hugging_face_url(newurl, label="metadata redirect")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _build_http_opener() -> Any:
    # An explicit empty proxy map prevents HTTP(S)_PROXY and system proxy
    # settings from redirecting trusted metadata requests through ambient hosts.
    return build_opener(ProxyHandler({}), _HuggingFaceRedirectHandler())


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise SourceLockError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> None:
    raise SourceLockError(f"non-finite JSON number is forbidden: {value}")


def canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def git_blob_sha1(value: bytes) -> str:
    header = f"blob {len(value)}\0".encode("ascii")
    return hashlib.sha1(header + value).hexdigest()


def safe_repo_path(value: str) -> str:
    if not isinstance(value, str) or not SAFE_REPO_PATH.fullmatch(value):
        raise SourceLockError(f"unsafe repository path: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise SourceLockError(f"unsafe repository path: {value!r}")
    return value


def _safe_scalar(value: object, *, allow_string: bool) -> object | None:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if allow_string and isinstance(value, str) and SAFE_IDENTIFIER.fullmatch(value):
        return value
    return None


def structural_projection(value: object, *, key: str = "", depth: int = 0) -> object:
    """Keep shape/config data while excluding templates, prose, and executable text."""

    if depth > 8:
        return {"truncated": True}
    scalar = _safe_scalar(value, allow_string=key in STRUCTURAL_STRING_KEYS)
    if scalar is not None or value is None:
        return scalar
    if isinstance(value, list):
        projected: list[object] = []
        for item in value[:4096]:
            child = structural_projection(item, key=key, depth=depth + 1)
            if child is not None:
                projected.append(child)
        return projected
    if isinstance(value, dict):
        projected_dict: dict[str, object] = {}
        for child_key in sorted(value):
            if not isinstance(child_key, str) or not SAFE_IDENTIFIER.fullmatch(child_key):
                continue
            child = structural_projection(
                value[child_key], key=child_key, depth=depth + 1
            )
            if child is not None:
                projected_dict[child_key] = child
        return projected_dict
    return None


def _read_json_bytes(payload: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            payload,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite,
        )
    except SourceLockError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SourceLockError(f"{label} is not valid UTF-8 JSON: {error}") from error
    if not isinstance(value, dict):
        raise SourceLockError(f"{label} must contain one JSON object")
    return value


def bounded_get(url: str, *, max_bytes: int) -> bytes:
    _validate_hugging_face_url(url, label="metadata URL")
    request = Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "ApxInf-macOS-source-lock/1",
        },
        method="GET",
    )
    with _build_http_opener().open(request, timeout=30) as response:
        _validate_hugging_face_url(response.geturl(), label="metadata response URL")
        length = response.headers.get("Content-Length")
        if length is not None:
            try:
                declared = int(length)
            except ValueError as error:
                raise SourceLockError("invalid Content-Length from Hugging Face") from error
            if declared < 0 or declared > max_bytes:
                raise SourceLockError(
                    f"metadata response exceeds byte cap: {declared} > {max_bytes}"
                )
        payload = response.read(max_bytes + 1)
    if len(payload) > max_bytes:
        raise SourceLockError(f"metadata response exceeds byte cap: > {max_bytes}")
    return payload


def _metadata_url(repo_id: str, commit: str, filename: str) -> str:
    encoded_repo = quote(repo_id, safe="/")
    encoded_file = "/".join(quote(part, safe="") for part in filename.split("/"))
    return f"https://huggingface.co/{encoded_repo}/resolve/{commit}/{encoded_file}"


def _api_url(repo_id: str, revision: str) -> str:
    return (
        "https://huggingface.co/api/models/"
        f"{quote(repo_id, safe='/')}/revision/{quote(revision, safe='')}?blobs=true"
    )


def _sibling_table(model_info: dict[str, Any]) -> dict[str, dict[str, Any]]:
    raw = model_info.get("siblings")
    if not isinstance(raw, list):
        raise SourceLockError("Hub model metadata does not contain a siblings list")
    siblings: dict[str, dict[str, Any]] = {}
    for item in raw:
        if not isinstance(item, dict):
            raise SourceLockError("Hub sibling entry is not an object")
        name = safe_repo_path(item.get("rfilename"))
        if name in siblings:
            raise SourceLockError(f"duplicate Hub sibling: {name}")
        size = item.get("size")
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise SourceLockError(f"Hub sibling has invalid size: {name}")
        blob_id = item.get("blobId")
        if not isinstance(blob_id, str) or not SHA1.fullmatch(blob_id):
            raise SourceLockError(f"Hub sibling has invalid blobId: {name}")
        lfs = item.get("lfs")
        if lfs is not None:
            if not isinstance(lfs, dict):
                raise SourceLockError(f"Hub sibling has invalid LFS metadata: {name}")
            lfs_sha = lfs.get("sha256")
            lfs_size = lfs.get("size")
            if not isinstance(lfs_sha, str) or not SHA256.fullmatch(lfs_sha):
                raise SourceLockError(f"Hub sibling has invalid LFS SHA-256: {name}")
            if not isinstance(lfs_size, int) or lfs_size != size:
                raise SourceLockError(f"Hub sibling has inconsistent LFS size: {name}")
        siblings[name] = {"path": name, "size": size, "git_blob_sha1": blob_id, "lfs": lfs}
    return siblings


def _metadata_names(siblings: dict[str, dict[str, Any]]) -> list[str]:
    names = sorted(name for name in METADATA_FILES if name in siblings)
    if "config.json" not in names:
        raise SourceLockError("model does not contain config.json")
    return names


def _weight_plan(
    *, siblings: dict[str, dict[str, Any]], metadata_payloads: dict[str, bytes]
) -> dict[str, Any]:
    index_name = "model.safetensors.index.json"
    tensor_names: list[str] = []
    if index_name in metadata_payloads:
        index = _read_json_bytes(metadata_payloads[index_name], index_name)
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise SourceLockError("SafeTensors index has no weight_map")
        weight_files: set[str] = set()
        for tensor_name, filename in weight_map.items():
            if not isinstance(tensor_name, str) or not tensor_name or len(tensor_name) > 1024:
                raise SourceLockError("SafeTensors index contains an invalid tensor name")
            safe_repo_path(tensor_name)
            weight_files.add(safe_repo_path(filename))
            tensor_names.append(tensor_name)
    else:
        weight_files = {
            name
            for name in siblings
            if name.endswith(".safetensors") and "/" not in name
        }
        if not weight_files:
            raise SourceLockError("model has no SafeTensors weight file or index")

    records: list[dict[str, Any]] = []
    total = 0
    for name in sorted(weight_files):
        sibling = siblings.get(name)
        if sibling is None:
            raise SourceLockError(f"SafeTensors index references a missing shard: {name}")
        if not name.endswith(".safetensors"):
            raise SourceLockError(f"SafeTensors index references a non-SafeTensors shard: {name}")
        lfs = sibling.get("lfs")
        if not isinstance(lfs, dict):
            raise SourceLockError(f"weight shard is not bound to LFS SHA-256: {name}")
        records.append(
            {
                "path": name,
                "size": sibling["size"],
                "sha256": lfs["sha256"],
                "git_blob_sha1": sibling["git_blob_sha1"],
            }
        )
        total += sibling["size"]
    return {
        "format": "safetensors",
        "index_file": index_name if index_name in metadata_payloads else None,
        "files": records,
        "total_bytes": total,
        "tensor_count": len(tensor_names) if tensor_names else None,
        "tensor_names": sorted(tensor_names),
    }


def build_source_lock(
    *,
    repo_id: str,
    requested_revision: str,
    get_bytes: Callable[..., bytes] = bounded_get,
) -> dict[str, Any]:
    api_payload = get_bytes(_api_url(repo_id, requested_revision), max_bytes=MAX_API_BYTES)
    model_info = _read_json_bytes(api_payload, "Hub model metadata")
    model_id = model_info.get("modelId") or model_info.get("id")
    if model_id != repo_id:
        raise SourceLockError(
            f"Hub returned a different model id: expected {repo_id}, observed {model_id!r}"
        )
    commit = model_info.get("sha")
    if not isinstance(commit, str) or not SHA1.fullmatch(commit):
        raise SourceLockError("Hub did not resolve the revision to a 40-hex commit")
    siblings = _sibling_table(model_info)

    metadata_payloads: dict[str, bytes] = {}
    metadata_records: list[dict[str, Any]] = []
    total_metadata_bytes = len(api_payload)
    for name in _metadata_names(siblings):
        sibling = siblings[name]
        if sibling["size"] > MAX_METADATA_FILE_BYTES:
            raise SourceLockError(f"metadata file exceeds byte cap: {name}")
        payload = get_bytes(
            _metadata_url(repo_id, commit, name), max_bytes=MAX_METADATA_FILE_BYTES
        )
        if len(payload) != sibling["size"]:
            raise SourceLockError(
                f"metadata file size mismatch for {name}: {len(payload)} != {sibling['size']}"
            )
        observed_blob = git_blob_sha1(payload)
        if observed_blob != sibling["git_blob_sha1"]:
            raise SourceLockError(f"metadata Git blob hash mismatch: {name}")
        total_metadata_bytes += len(payload)
        if total_metadata_bytes > MAX_METADATA_TOTAL_BYTES:
            raise SourceLockError("metadata download exceeds total byte cap")
        metadata_payloads[name] = payload
        metadata_records.append(
            {
                "path": name,
                "size": len(payload),
                "git_blob_sha1": observed_blob,
                "sha256": sha256_bytes(payload),
            }
        )

    config = _read_json_bytes(metadata_payloads["config.json"], "config.json")
    config_keys = _canonical_mapping_keys(
        config, field="config.json", maximum=MAX_CONFIG_KEYS
    )
    tokenizer = (
        _read_json_bytes(metadata_payloads["tokenizer_config.json"], "tokenizer_config.json")
        if "tokenizer_config.json" in metadata_payloads
        else {}
    )
    weight_plan = _weight_plan(
        siblings=siblings, metadata_payloads=metadata_payloads
    )
    python_files = sorted(name for name in siblings if name.endswith(".py"))
    unsafe_weight_files = sorted(
        name for name in siblings if name.lower().endswith(UNSAFE_WEIGHT_SUFFIXES)
    )
    auto_map = config.get("auto_map")
    auto_map_keys = (
        []
        if auto_map is None
        else _canonical_mapping_keys(
            auto_map,
            field="config.json auto_map",
            maximum=MAX_AUTO_MAP_KEYS,
            pattern=AUTO_MAP_KEY,
        )
    )
    license_id = (model_info.get("cardData") or {}).get("license")
    if license_id is not None and (
        not isinstance(license_id, str) or not SAFE_IDENTIFIER.fullmatch(license_id)
    ):
        license_id = None

    lock: dict[str, Any] = {
        "format": LOCK_FORMAT,
        "repo_id": repo_id,
        "requested_revision": requested_revision,
        "resolved_commit": commit,
        "source": {
            "url": f"https://huggingface.co/{repo_id}",
            "private": model_info.get("private") is True,
            "gated": _gated_value(model_info.get("gated", False)),
            "disabled": model_info.get("disabled") is True,
            "license": license_id,
            "pipeline_tag": _allowlisted_source_slug(
                model_info.get("pipeline_tag"), field="pipeline_tag"
            ),
            "library_name": _allowlisted_source_slug(
                model_info.get("library_name"), field="library_name"
            ),
        },
        "architecture": {
            "config_sha256": sha256_bytes(metadata_payloads["config.json"]),
            "config_keys": config_keys,
            "structural_config": structural_projection(config),
            "tokenizer": structural_projection(tokenizer),
        },
        "security": {
            "remote_code_indicators": {
                "auto_map_keys": auto_map_keys,
                "python_files": python_files,
            },
            "unsafe_weight_files": unsafe_weight_files,
            "safetensors_only_plan": not unsafe_weight_files,
        },
        "weights": weight_plan,
        "metadata": {
            "api_sha256": sha256_bytes(api_payload),
            "downloaded_bytes": total_metadata_bytes,
            "files": metadata_records,
            "file_cap_bytes": MAX_METADATA_FILE_BYTES,
            "total_cap_bytes": MAX_METADATA_TOTAL_BYTES,
        },
        "policy_receipt": {
            "metadata_only": True,
            "weight_payload_bytes_downloaded": 0,
            "remote_code_executed": False,
            "hf_token_read": False,
        },
    }
    lock["content_sha256"] = sha256_bytes(canonical_bytes(lock))
    return lock


def validate_source_lock(lock: object, *, expected_sha256: str | None = None) -> dict[str, Any]:
    if not isinstance(lock, dict) or lock.get("format") != LOCK_FORMAT:
        raise SourceLockError(f"source lock format must be {LOCK_FORMAT}")
    content_hash = lock.get("content_sha256")
    if not isinstance(content_hash, str) or not SHA256.fullmatch(content_hash):
        raise SourceLockError("source lock content_sha256 is invalid")
    body = dict(lock)
    del body["content_sha256"]
    observed = sha256_bytes(canonical_bytes(body))
    if observed != content_hash:
        raise SourceLockError("source lock content hash mismatch")
    if expected_sha256 is not None and content_hash != expected_sha256:
        raise SourceLockError("source lock does not match --expected-sha256")
    repo_value = lock.get("repo_id")
    requested_value = lock.get("requested_revision")
    if not isinstance(repo_value, str) or not isinstance(requested_value, str):
        raise SourceLockError("source lock model identity is invalid")
    repo_id, revision = parse_model_reference(repo_value, None)
    if revision != "main":
        raise SourceLockError("source lock repo_id unexpectedly includes a revision")
    _, requested = parse_model_reference(repo_id, requested_value)
    commit = lock.get("resolved_commit")
    if not isinstance(commit, str) or not SHA1.fullmatch(commit):
        raise SourceLockError("source lock resolved_commit is invalid")
    source = lock.get("source")
    if type(source) is not dict or set(source) != SOURCE_KEYS:
        raise SourceLockError("source lock source section has an invalid schema")
    if source["url"] != f"https://huggingface.co/{repo_id}":
        raise SourceLockError("source lock source URL is not canonical")
    if type(source["private"]) is not bool or type(source["disabled"]) is not bool:
        raise SourceLockError("source lock source policy flags must be booleans")
    _gated_value(source["gated"])
    license_id = source["license"]
    if license_id is not None and (
        type(license_id) is not str or not SAFE_IDENTIFIER.fullmatch(license_id)
    ):
        raise SourceLockError("source lock license is not a canonical identifier")
    for field in ("pipeline_tag", "library_name"):
        if _allowlisted_source_slug(source[field], field=field) != source[field]:
            raise SourceLockError(f"source lock {field} is not allowlisted")
    architecture = lock.get("architecture")
    if type(architecture) is not dict or set(architecture) != ARCHITECTURE_KEYS:
        raise SourceLockError("source lock architecture section has an invalid schema")
    config_sha256 = architecture["config_sha256"]
    if type(config_sha256) is not str or not SHA256.fullmatch(config_sha256):
        raise SourceLockError("source lock config SHA-256 is invalid")
    config_keys = _validate_canonical_string_list(
        architecture["config_keys"],
        field="source lock config_keys",
        maximum=MAX_CONFIG_KEYS,
        pattern=SAFE_IDENTIFIER,
    )
    structural_config = architecture["structural_config"]
    tokenizer = architecture["tokenizer"]
    if (
        type(structural_config) is not dict
        or structural_projection(structural_config) != structural_config
        or not set(structural_config).issubset(config_keys)
    ):
        raise SourceLockError("source lock structural config is not a safe projection")
    if type(tokenizer) is not dict or structural_projection(tokenizer) != tokenizer:
        raise SourceLockError("source lock tokenizer config is not a safe projection")
    security = lock.get("security")
    if type(security) is not dict or set(security) != SECURITY_KEYS:
        raise SourceLockError("source lock security section has an invalid schema")
    remote_code = security["remote_code_indicators"]
    if type(remote_code) is not dict or set(remote_code) != REMOTE_CODE_KEYS:
        raise SourceLockError("source lock remote-code section has an invalid schema")
    _validate_canonical_string_list(
        remote_code["auto_map_keys"],
        field="source lock auto_map_keys",
        maximum=MAX_AUTO_MAP_KEYS,
        pattern=AUTO_MAP_KEY,
    )
    python_files = _validate_repo_path_list(
        remote_code["python_files"],
        field="source lock python_files",
        maximum=MAX_INDICATOR_FILES,
    )
    if any(not item.endswith(".py") for item in python_files):
        raise SourceLockError("source lock python_files contains a non-Python path")
    unsafe_weight_files = _validate_repo_path_list(
        security["unsafe_weight_files"],
        field="source lock unsafe_weight_files",
        maximum=MAX_INDICATOR_FILES,
    )
    if any(
        not item.lower().endswith(UNSAFE_WEIGHT_SUFFIXES)
        for item in unsafe_weight_files
    ):
        raise SourceLockError("source lock unsafe_weight_files contains a safe suffix")
    if type(security["safetensors_only_plan"]) is not bool or security[
        "safetensors_only_plan"
    ] is not (not unsafe_weight_files):
        raise SourceLockError("source lock SafeTensors policy is inconsistent")
    receipt = lock.get("policy_receipt")
    if receipt != {
        "metadata_only": True,
        "weight_payload_bytes_downloaded": 0,
        "remote_code_executed": False,
        "hf_token_read": False,
    }:
        raise SourceLockError("source lock policy receipt is not metadata-only")
    metadata = lock.get("metadata")
    if not isinstance(metadata, dict):
        raise SourceLockError("source lock metadata section is invalid")
    downloaded = metadata.get("downloaded_bytes")
    if not isinstance(downloaded, int) or downloaded < 0 or downloaded > MAX_METADATA_TOTAL_BYTES:
        raise SourceLockError("source lock metadata byte total exceeds policy")
    files = metadata.get("files")
    if not isinstance(files, list) or not files:
        raise SourceLockError("source lock metadata files are missing")
    seen: set[str] = set()
    for record in files:
        if not isinstance(record, dict):
            raise SourceLockError("source lock metadata record is invalid")
        name = safe_repo_path(record.get("path"))
        if name not in METADATA_FILES or name in seen:
            raise SourceLockError(f"source lock contains an unexpected metadata file: {name}")
        seen.add(name)
        if (
            type(record.get("size")) is not int
            or record["size"] < 0
            or record["size"] > MAX_METADATA_FILE_BYTES
        ):
            raise SourceLockError(f"source lock metadata size is invalid: {name}")
        if not isinstance(record.get("git_blob_sha1"), str) or not SHA1.fullmatch(record["git_blob_sha1"]):
            raise SourceLockError(f"source lock metadata Git hash is invalid: {name}")
        if not isinstance(record.get("sha256"), str) or not SHA256.fullmatch(record["sha256"]):
            raise SourceLockError(f"source lock metadata SHA-256 is invalid: {name}")
    weights = lock.get("weights")
    if not isinstance(weights, dict) or weights.get("format") != "safetensors":
        raise SourceLockError("source lock weight plan is not SafeTensors")
    weight_files = weights.get("files")
    if not isinstance(weight_files, list) or not weight_files:
        raise SourceLockError("source lock weight plan has no files")
    total = 0
    seen_weights: set[str] = set()
    for record in weight_files:
        if not isinstance(record, dict):
            raise SourceLockError("source lock weight record is invalid")
        name = safe_repo_path(record.get("path"))
        if name in seen_weights:
            raise SourceLockError(f"source lock contains a duplicate weight: {name}")
        seen_weights.add(name)
        if not name.endswith(".safetensors"):
            raise SourceLockError(f"source lock weight is not SafeTensors: {name}")
        size = record.get("size")
        digest = record.get("sha256")
        if type(size) is not int or size <= 0:
            raise SourceLockError(f"source lock weight size is invalid: {name}")
        if not isinstance(digest, str) or not SHA256.fullmatch(digest):
            raise SourceLockError(f"source lock weight SHA-256 is invalid: {name}")
        total += size
    if weights.get("total_bytes") != total:
        raise SourceLockError("source lock weight byte total is inconsistent")
    return {
        "passed": True,
        "format": LOCK_FORMAT,
        "repo_id": repo_id,
        "requested_revision": requested,
        "resolved_commit": commit,
        "content_sha256": content_hash,
        "metadata_bytes": downloaded,
        "weight_payload_bytes_downloaded": 0,
    }


def _exclusive_json_write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            os.chmod(temporary, 0o600)
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    except FileExistsError as error:
        raise SourceLockError(f"refusing to overwrite existing source lock: {path}") from error
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("model", nargs="?", help="owner/model or canonical Hugging Face URL")
    result.add_argument("--revision", help="branch, tag, or commit; defaults to main")
    result.add_argument("--output", type=Path, help="exclusive source-lock output path")
    result.add_argument("--verify", type=Path, help="verify one existing source lock offline")
    result.add_argument("--expected-sha256", help="expected source-lock content hash")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.verify is not None:
            if args.model is not None or args.output is not None or args.revision is not None:
                raise SourceLockError("--verify cannot be combined with resolve arguments")
            declared = args.verify.expanduser()
            info = declared.lstat()
            if declared.is_symlink() or not declared.is_file():
                raise SourceLockError("source lock must be a regular non-symlink file")
            if info.st_size > MAX_METADATA_TOTAL_BYTES:
                raise SourceLockError("source lock exceeds the metadata byte cap")
            lock = _read_json_bytes(declared.read_bytes(), "source lock")
            receipt = validate_source_lock(lock, expected_sha256=args.expected_sha256)
        else:
            if args.model is None:
                raise SourceLockError("model is required unless --verify is used")
            repo_id, revision = parse_model_reference(args.model, args.revision)
            output = (
                args.output.expanduser().resolve()
                if args.output is not None
                else Path.cwd()
                / ".apxinf"
                / "onboarding"
                / model_slug(repo_id)
                / "source-lock.json"
            )
            lock = build_source_lock(repo_id=repo_id, requested_revision=revision)
            _exclusive_json_write(output, lock)
            receipt = validate_source_lock(lock)
            receipt["path"] = str(output)
        print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
        return 0
    except (IntakeError, SourceLockError, OSError, json.JSONDecodeError) as error:
        print(
            json.dumps(
                {"passed": False, "error": str(error), "error_type": type(error).__name__},
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
