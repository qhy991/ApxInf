#!/usr/bin/env python3
"""Persistent, single-model MLX-LM service over a strict local JSONL pipe.

This is a trusted-local/offline process boundary, not an operating-system
network sandbox.  The caller must select trusted, direct local runtime files.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
import contextlib
from dataclasses import dataclass
import hashlib
import importlib.util
import inspect
import os
from pathlib import Path
import platform
import re
import stat
import sys
import time
from typing import Any, NoReturn


_HELPER_PATH = Path(__file__).with_name("apxinf_mlx_generate.py")
_HELPER_SPEC = importlib.util.spec_from_file_location(
    "_apxinf_mlx_generate_service_helper", _HELPER_PATH
)
if _HELPER_SPEC is None or _HELPER_SPEC.loader is None:
    raise RuntimeError("cannot load the adjacent MLX generation helper")
one_shot = importlib.util.module_from_spec(_HELPER_SPEC)
_HELPER_SPEC.loader.exec_module(one_shot)


PROTOCOL = "apxinf-mlx-service-v1"
READY_FORMAT = "apxinf-mlx-service-ready-v1"
REQUEST_FORMAT = "apxinf-mlx-service-request-v1"
RESPONSE_FORMAT = "apxinf-mlx-service-response-v1"
RESPONSE_ERROR_FORMAT = "apxinf-mlx-service-response-error-v1"
CONTROL_FORMAT = "apxinf-mlx-service-control-v1"
SHUTDOWN_FORMAT = "apxinf-mlx-service-shutdown-v1"
SESSION_PROTOCOL = "apxinf-mlx-session-v1"
SESSION_REQUEST_FORMAT = "apxinf-mlx-session-request-v1"
SESSION_RESPONSE_FORMAT = "apxinf-mlx-session-response-v1"
SESSION_RESPONSE_ERROR_FORMAT = "apxinf-mlx-session-response-error-v1"
SESSION_CONTROL_FORMAT = "apxinf-mlx-session-control-v1"
SESSION_RESET_FORMAT = "apxinf-mlx-session-reset-v1"
SESSION_BINDING_FORMAT = "apxinf-mlx-session-binding-v1"
SESSION_PREFIX_FORMAT = "apxinf-mlx-session-prefix-v1"
SESSION_CACHE_READY_FORMAT = "apxinf-mlx-session-cache-ready-v1"
SESSION_CACHE_POLICY = "exact-append-only-in-process-lru-v1"
MAX_LINE_BYTES = one_shot.MAX_REQUEST_BYTES
MAX_OUTPUT_BYTES = one_shot.MAX_OUTPUT_BYTES
MAX_REQUESTS = 1_000_000
MAX_BUNDLE_FILES = 4096
MAX_SESSIONS = 4
MAX_SESSION_CACHE_BYTES = 512 * 1024 * 1024
REQUEST_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z")
SESSION_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,63}\Z")
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
GREEDY_STRATEGY = one_shot.GREEDY_STRATEGY


class ServiceError(one_shot.WorkerError):
    """A deterministic service protocol or inference failure."""


class JsonArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        raise ServiceError("invalid_arguments", message)


@dataclass
class SessionEntry:
    token_ids: list[int]
    token_ids_sha256: str
    prompt_cache: list[Any]
    cache_bytes: int
    binding: dict[str, object]


def _identity(path: Path) -> dict[str, str]:
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise ServiceError(
            "invalid_runtime", f"cannot resolve runtime file: {error}"
        ) from error
    if not resolved.is_file() or resolved.is_symlink():
        raise ServiceError("invalid_runtime", "runtime file must be a regular file")
    return {"path": str(resolved), "sha256": one_shot._file_sha256(resolved)}


def _bundle_identity(model_dir: Path) -> dict[str, object]:
    try:
        entries = sorted(os.scandir(model_dir), key=lambda entry: entry.name)
    except OSError as error:
        raise ServiceError(
            "invalid_model", f"cannot scan model bundle: {error}"
        ) from error
    if not entries or len(entries) > MAX_BUNDLE_FILES:
        raise ServiceError(
            "invalid_model", f"model bundle must contain 1..{MAX_BUNDLE_FILES} files"
        )
    manifest = hashlib.sha256(b"apxinf-local-bundle-manifest-v1\0")
    total_bytes = 0
    for entry in entries:
        if (
            not entry.name
            or any(
                ord(character) < 32 or ord(character) == 127 for character in entry.name
            )
            or "/" in entry.name
        ):
            raise ServiceError("invalid_model", "model bundle filename is unsafe")
        try:
            selected = entry.stat(follow_symlinks=False)
        except OSError as error:
            raise ServiceError(
                "invalid_model", f"cannot inspect model bundle file: {error}"
            ) from error
        if not stat.S_ISREG(selected.st_mode):
            raise ServiceError(
                "invalid_model", "model bundle must contain only direct regular files"
            )
        digest = hashlib.sha256()
        try:
            with open(entry.path, "rb") as source:
                opened = os.fstat(source.fileno())
                if (
                    opened.st_dev != selected.st_dev
                    or opened.st_ino != selected.st_ino
                    or opened.st_size != selected.st_size
                    or opened.st_mode != selected.st_mode
                ):
                    raise ServiceError(
                        "invalid_model", "model bundle file changed while opening"
                    )
                while chunk := source.read(1024 * 1024):
                    digest.update(chunk)
                after = os.fstat(source.fileno())
        except OSError as error:
            raise ServiceError(
                "invalid_model", f"cannot hash model bundle file: {error}"
            ) from error
        if (
            after.st_dev != opened.st_dev
            or after.st_ino != opened.st_ino
            or after.st_size != opened.st_size
            or after.st_mode != opened.st_mode
        ):
            raise ServiceError(
                "invalid_model", "model bundle file changed while hashing"
            )
        encoded_name = entry.name.encode("utf-8")
        manifest.update(encoded_name)
        manifest.update(b"\0")
        manifest.update(str(opened.st_size).encode("ascii"))
        manifest.update(b"\0")
        manifest.update(digest.hexdigest().encode("ascii"))
        manifest.update(b"\n")
        total_bytes += opened.st_size
    return {
        "format": "apxinf-local-bundle-manifest-v1",
        "file_count": len(entries),
        "total_bytes": total_bytes,
        "sha256": manifest.hexdigest(),
    }


def _request_id(value: object) -> str:
    if type(value) is not str or REQUEST_ID_PATTERN.fullmatch(value) is None:
        raise ServiceError(
            "invalid_request",
            "request_id must be 1..128 safe ASCII characters",
        )
    return value


def _session_id(value: object) -> str:
    if type(value) is not str or SESSION_ID_PATTERN.fullmatch(value) is None:
        raise ServiceError(
            "invalid_request",
            "session_id must be 1..64 safe ASCII characters",
        )
    return value


def _session_prefix(value: object) -> dict[str, object]:
    if type(value) is not dict or set(value) != {
        "format",
        "token_count",
        "token_ids_sha256",
    }:
        raise ServiceError(
            "invalid_request", "expected_prefix keys do not match the contract"
        )
    if value["format"] != SESSION_PREFIX_FORMAT:
        raise ServiceError("invalid_request", "expected_prefix format is invalid")
    token_count = value["token_count"]
    if (
        type(token_count) is not int
        or token_count < 0
        or token_count > one_shot.MAX_PROMPT_TOKENS
    ):
        raise ServiceError("invalid_request", "expected prefix token count is invalid")
    digest = value["token_ids_sha256"]
    if type(digest) is not str or SHA256_PATTERN.fullmatch(digest) is None:
        raise ServiceError("invalid_request", "expected prefix token hash is invalid")
    return {
        "format": SESSION_PREFIX_FORMAT,
        "token_count": token_count,
        "token_ids_sha256": digest,
    }


def _session_binding(value: object) -> dict[str, object]:
    expected = {
        "format",
        "model_config_sha256",
        "model_bundle_sha256",
        "greedy_strategy",
        "cache_policy",
    }
    if type(value) is not dict or set(value) != expected:
        raise ServiceError(
            "invalid_request", "session binding keys do not match the contract"
        )
    if value["format"] != SESSION_BINDING_FORMAT:
        raise ServiceError("invalid_request", "session binding format is invalid")
    for field in ("model_config_sha256", "model_bundle_sha256"):
        digest = value[field]
        if type(digest) is not str or SHA256_PATTERN.fullmatch(digest) is None:
            raise ServiceError("invalid_request", f"session binding {field} is invalid")
    for field in ("greedy_strategy", "cache_policy"):
        selected = value[field]
        if (
            type(selected) is not str
            or not selected
            or selected != selected.strip()
            or len(selected) > 128
            or any(
                ord(character) < 32 or ord(character) == 127 for character in selected
            )
        ):
            raise ServiceError("invalid_request", f"session binding {field} is invalid")
    return dict(value)


def _read_line() -> bytes | None:
    payload = sys.stdin.buffer.readline(MAX_LINE_BYTES + 1)
    if not payload:
        return None
    if len(payload) > MAX_LINE_BYTES:
        raise ServiceError("invalid_request", f"request exceeds {MAX_LINE_BYTES} bytes")
    if payload[-1:] != b"\n" or b"\r" in payload:
        raise ServiceError(
            "invalid_request", "request must be one newline-terminated JSON line"
        )
    body = payload[:-1]
    if not body:
        raise ServiceError("invalid_request", "request line must not be empty")
    return body


def _write_line(value: object) -> None:
    payload = one_shot._json_line(value)
    if len(payload) > MAX_OUTPUT_BYTES:
        raise ServiceError(
            "output_too_large", f"response exceeds {MAX_OUTPUT_BYTES} bytes"
        )
    sys.stdout.buffer.write(payload)
    sys.stdout.buffer.flush()


def _safe_error(error: one_shot.WorkerError) -> dict[str, object]:
    message = " ".join(str(error).split())[:1024] or "unknown service error"
    return {"code": error.code, "message": message}


class LoadedService:
    def __init__(self, model_dir_argument: str) -> None:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
        os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        self.model_dir, self.config, self.config_sha256 = one_shot._read_model_config(
            model_dir_argument
        )
        self.bundle_identity = _bundle_identity(self.model_dir)
        self.packages = one_shot._pinned_toolchain_versions()
        try:
            python_path = Path(sys.executable).resolve(strict=True)
            runner_path = Path(__file__).resolve(strict=True)
            helper_path = Path(one_shot.__file__).resolve(strict=True)
        except OSError as error:
            raise ServiceError(
                "invalid_runtime", f"cannot resolve runtime identity: {error}"
            ) from error
        self.runtime = {
            "policy": "trusted-local-offline-environment-v1",
            "offline_environment": True,
            "os_network_sandbox": False,
            "trust_remote_code": False,
            "python_version": platform.python_version(),
            "python": _identity(python_path),
            "runner": _identity(runner_path),
            "generation_helper": _identity(helper_path),
        }

        try:
            import mlx.core as mx
            import mlx_lm
            from mlx_lm.generate import generate_step
            from mlx_lm.models.cache import make_prompt_cache
        except (ImportError, ModuleNotFoundError) as error:
            raise ServiceError(
                "dependency_unavailable", f"cannot import MLX runtime: {error}"
            ) from error

        self.mx = mx
        self.generate_step = generate_step
        self.make_prompt_cache = make_prompt_cache
        try:
            generate_parameters = inspect.signature(generate_step).parameters
        except (TypeError, ValueError) as error:
            raise ServiceError(
                "dependency_unavailable",
                "cannot inspect MLX-LM generate_step cache contract",
            ) from error
        if "prompt_cache" not in generate_parameters:
            raise ServiceError(
                "dependency_unavailable",
                "MLX-LM generate_step does not expose prompt_cache",
            )
        self.mx.reset_peak_memory()
        load_start = time.perf_counter_ns()
        try:
            loaded = mlx_lm.load(
                str(self.model_dir),
                tokenizer_config={
                    "local_files_only": True,
                    "trust_remote_code": False,
                },
                lazy=False,
                return_config=True,
            )
        except Exception as error:
            raise ServiceError(
                "model_load_failed", f"MLX-LM model load failed: {error}"
            ) from error
        load_end = time.perf_counter_ns()
        if type(loaded) is not tuple or len(loaded) != 3:
            raise ServiceError(
                "model_load_failed", "MLX-LM load returned an unexpected value"
            )
        self.model, self.tokenizer, loaded_config = loaded
        if type(loaded_config) is not dict:
            raise ServiceError(
                "model_load_failed", "MLX-LM load did not return model config"
            )
        if loaded_config.get("model_type") != self.config["model_type"]:
            raise ServiceError(
                "model_load_failed", "loaded model_type differs from config.json"
            )
        self.load_ms = (load_end - load_start) / 1_000_000
        self.model_identity = {
            "model_dir": str(self.model_dir),
            "model_type": self.config["model_type"],
            "quantization": one_shot._configured_quantization(self.config),
            "config_sha256": self.config_sha256,
            "bundle": self.bundle_identity,
        }
        self.sessions: OrderedDict[str, SessionEntry] = OrderedDict()
        self.session_cache_bytes = 0

    def ready(self) -> dict[str, object]:
        return {
            "format": READY_FORMAT,
            "protocol": PROTOCOL,
            "model": self.model_identity,
            "packages": self.packages,
            "runtime": self.runtime,
            "limits": {
                "max_line_bytes": MAX_LINE_BYTES,
                "max_output_bytes": MAX_OUTPUT_BYTES,
                "max_prompt_tokens": one_shot.MAX_PROMPT_TOKENS,
                "max_generated_tokens": one_shot.MAX_GENERATED_TOKENS,
                "max_requests": MAX_REQUESTS,
            },
            "metrics": {"load_ms": self.load_ms},
            "greedy_strategy": GREEDY_STRATEGY,
            "session_cache": {
                "format": SESSION_CACHE_READY_FORMAT,
                "protocol": SESSION_PROTOCOL,
                "policy": SESSION_CACHE_POLICY,
                "request_format": SESSION_REQUEST_FORMAT,
                "control_format": SESSION_CONTROL_FORMAT,
                "max_sessions": MAX_SESSIONS,
                "max_bytes": MAX_SESSION_CACHE_BYTES,
            },
        }

    def session_binding(self) -> dict[str, object]:
        return {
            "format": SESSION_BINDING_FORMAT,
            "model_config_sha256": self.config_sha256,
            "model_bundle_sha256": self.bundle_identity["sha256"],
            "greedy_strategy": GREEDY_STRATEGY,
            "cache_policy": SESSION_CACHE_POLICY,
        }

    def generate(
        self,
        request_id: str,
        request: dict[str, object],
        *,
        prompt_cache: list[Any] | None = None,
    ) -> dict[str, object]:
        requested_eos = request["eos_token_id"]
        if requested_eos is not None:
            eos_ids = [requested_eos]
        else:
            eos_ids = one_shot._normalise_eos_ids(
                getattr(self.tokenizer, "eos_token_ids", None)
            )
            if not eos_ids:
                eos_ids = one_shot._config_eos_ids(self.config)
        if request["stop_on_eos"] and request["max_tokens"] > 0 and not eos_ids:
            raise ServiceError(
                "invalid_model", "stop_on_eos requires an effective EOS token id"
            )

        self.mx.reset_peak_memory()
        generated_ids: list[int] = []
        generation_start = time.perf_counter_ns()
        first_token_at: int | None = None
        stop_reason = "length"
        if request["max_tokens"] > 0:
            try:
                prompt = self.mx.array(request["prompt_token_ids"])

                def greedy_argmax(logprobs):
                    return self.mx.argmax(logprobs, axis=-1)

                generate_arguments = {
                    "max_tokens": request["max_tokens"],
                    "sampler": greedy_argmax,
                }
                if prompt_cache is not None:
                    generate_arguments["prompt_cache"] = prompt_cache
                steps = self.generate_step(prompt, self.model, **generate_arguments)
                for raw_token, _logprobs in steps:
                    observed_at = time.perf_counter_ns()
                    if type(raw_token) is not int:
                        try:
                            raw_token = raw_token.item()
                        except (AttributeError, TypeError, ValueError) as error:
                            raise ServiceError(
                                "generation_failed",
                                "MLX-LM produced a non-integer token",
                            ) from error
                    token = one_shot._token_id(raw_token, "generated token")
                    if first_token_at is None:
                        first_token_at = observed_at
                    generated_ids.append(token)
                    if request["stop_on_eos"] and token in eos_ids:
                        stop_reason = "eos"
                        break
                    if len(generated_ids) == request["max_tokens"]:
                        break
                close_steps = getattr(steps, "close", None)
                if close_steps is not None:
                    close_steps()
            except one_shot.WorkerError:
                raise
            except Exception as error:
                raise ServiceError(
                    "generation_failed", f"MLX-LM generation failed: {error}"
                ) from error
        generation_end = time.perf_counter_ns()
        if request["max_tokens"] > 0 and (not generated_ids or first_token_at is None):
            raise ServiceError("generation_failed", "MLX-LM produced no tokens")
        if stop_reason == "length" and len(generated_ids) != request["max_tokens"]:
            raise ServiceError(
                "generation_failed", "MLX-LM ended before max_tokens without EOS"
            )

        decode_tokens = max(0, len(generated_ids) - 1)
        decode_ns = (
            max(1, generation_end - first_token_at)
            if decode_tokens and first_token_at is not None
            else 0
        )
        try:
            self.mx.synchronize()
            peak_memory = int(self.mx.get_peak_memory())
        except (RuntimeError, TypeError, ValueError, OverflowError) as error:
            raise ServiceError(
                "generation_failed", "MLX returned an invalid peak memory measurement"
            ) from error
        if peak_memory < 0:
            raise ServiceError(
                "generation_failed", "MLX returned a negative peak memory measurement"
            )

        return {
            "format": RESPONSE_FORMAT,
            "protocol": PROTOCOL,
            "request_id": request_id,
            "request": {
                "prompt_token_count": len(request["prompt_token_ids"]),
                "prompt_token_ids_sha256": one_shot._token_ids_sha256(
                    request["prompt_token_ids"]
                ),
                "max_tokens": request["max_tokens"],
                "stop_on_eos": request["stop_on_eos"],
                "greedy_strategy": GREEDY_STRATEGY,
                "requested_eos_token_id": requested_eos,
                "effective_eos_token_ids": eos_ids,
            },
            "model": self.model_identity,
            "packages": self.packages,
            "runtime": self.runtime,
            "metrics": {
                "request_ms": (generation_end - generation_start) / 1_000_000,
                "ttft_ms": (
                    (first_token_at - generation_start) / 1_000_000
                    if first_token_at is not None
                    else 0.0
                ),
                "tpot_ms": (
                    decode_ns / decode_tokens / 1_000_000 if decode_tokens else 0.0
                ),
                "tps": (
                    decode_tokens * 1_000_000_000 / decode_ns if decode_tokens else 0.0
                ),
                "timed_decode_tokens": decode_tokens,
                "mlx_peak_memory_bytes": peak_memory,
            },
            "generation": {
                "generated_token_ids": generated_ids,
                "generated_token_count": len(generated_ids),
                "stop_reason": stop_reason,
            },
        }

    def _new_prompt_cache(self) -> list[Any]:
        try:
            prompt_cache = self.make_prompt_cache(self.model)
        except Exception as error:
            raise ServiceError(
                "session_cache_failed", f"cannot construct model prompt cache: {error}"
            ) from error
        if type(prompt_cache) is not list or not prompt_cache:
            raise ServiceError(
                "session_cache_failed",
                "model prompt cache must be a non-empty list",
            )
        return prompt_cache

    @staticmethod
    def _prompt_cache_nbytes(prompt_cache: list[Any]) -> int:
        total = 0
        for index, cache in enumerate(prompt_cache):
            try:
                selected = cache.nbytes
            except Exception as error:
                raise ServiceError(
                    "session_cache_failed",
                    f"cannot read prompt cache byte size at index {index}",
                ) from error
            if type(selected) is not int or selected < 0:
                raise ServiceError(
                    "session_cache_failed",
                    f"prompt cache byte size at index {index} is invalid",
                )
            total += selected
            if total > (1 << 63) - 1:
                raise ServiceError(
                    "session_cache_failed", "prompt cache byte size overflowed"
                )
        return total

    def _drop_session(self, session_id: str) -> SessionEntry | None:
        entry = self.sessions.pop(session_id, None)
        if entry is not None:
            self.session_cache_bytes -= entry.cache_bytes
            if self.session_cache_bytes < 0:
                self.sessions.clear()
                self.session_cache_bytes = 0
                raise ServiceError(
                    "session_cache_failed", "session cache accounting failed closed"
                )
        return entry

    def _store_session(self, session_id: str, entry: SessionEntry) -> list[str]:
        if entry.cache_bytes > MAX_SESSION_CACHE_BYTES:
            self.mx.clear_cache()
            raise ServiceError(
                "session_cache_limit",
                "one session exceeds the configured prompt-cache byte limit",
            )
        if session_id in self.sessions:
            raise ServiceError(
                "session_cache_failed", "session was unexpectedly present at commit"
            )
        self.sessions[session_id] = entry
        self.session_cache_bytes += entry.cache_bytes
        evicted: list[str] = []
        while (
            len(self.sessions) > MAX_SESSIONS
            or self.session_cache_bytes > MAX_SESSION_CACHE_BYTES
        ):
            selected, removed = self.sessions.popitem(last=False)
            self.session_cache_bytes -= removed.cache_bytes
            evicted.append(selected)
        if evicted:
            self.mx.clear_cache()
        return evicted

    def generate_session(
        self, request_id: str, request: dict[str, object]
    ) -> dict[str, object]:
        session_id = request["session_id"]
        assert type(session_id) is str
        operation = request["operation"]
        assert operation in ("create", "append")
        full_prompt = request["prompt_token_ids"]
        assert type(full_prompt) is list
        expected_prefix = request["expected_prefix"]
        binding = request["binding"]
        assert type(expected_prefix) is dict and type(binding) is dict

        live_binding = self.session_binding()
        if binding != live_binding:
            raise ServiceError(
                "session_binding_mismatch",
                "session binding differs from the loaded model or strategy",
            )
        empty_hash = one_shot._token_ids_sha256([])
        reused_prefix_count = 0
        entry: SessionEntry | None = None
        if operation == "create":
            if session_id in self.sessions:
                raise ServiceError("session_exists", "session_id already exists")
            if (
                expected_prefix["token_count"] != 0
                or expected_prefix["token_ids_sha256"] != empty_hash
            ):
                raise ServiceError(
                    "session_prefix_mismatch",
                    "session creation must bind the empty prefix",
                )
            prompt_cache = self._new_prompt_cache()
            evaluated_prompt = full_prompt
        else:
            entry = self.sessions.get(session_id)
            if entry is None:
                raise ServiceError("session_not_found", "session_id is not resident")
            if entry.binding != live_binding:
                self._drop_session(session_id)
                raise ServiceError(
                    "session_binding_mismatch",
                    "resident session binding failed closed",
                )
            entry_hash = one_shot._token_ids_sha256(entry.token_ids)
            if entry_hash != entry.token_ids_sha256:
                self._drop_session(session_id)
                raise ServiceError(
                    "session_cache_failed", "resident session prefix integrity failed"
                )
            reused_prefix_count = len(entry.token_ids)
            if (
                expected_prefix["token_count"] != reused_prefix_count
                or expected_prefix["token_ids_sha256"] != entry.token_ids_sha256
                or len(full_prompt) <= reused_prefix_count
                or full_prompt[:reused_prefix_count] != entry.token_ids
            ):
                raise ServiceError(
                    "session_prefix_mismatch",
                    "full prompt is not an exact append to the resident prefix",
                )
            evaluated_prompt = full_prompt[reused_prefix_count:]
            removed = self._drop_session(session_id)
            assert removed is entry
            prompt_cache = entry.prompt_cache

        internal_request = {
            "prompt_token_ids": evaluated_prompt,
            "max_tokens": request["max_tokens"],
            "stop_on_eos": request["stop_on_eos"],
            "eos_token_id": request["eos_token_id"],
        }
        try:
            ordinary = self.generate(
                request_id,
                internal_request,
                prompt_cache=prompt_cache,
            )
            generated = ordinary["generation"]["generated_token_ids"]
            assert type(generated) is list
            new_prefix = [*full_prompt, *generated]
            cache_bytes = self._prompt_cache_nbytes(prompt_cache)
            new_entry = SessionEntry(
                token_ids=new_prefix,
                token_ids_sha256=one_shot._token_ids_sha256(new_prefix),
                prompt_cache=prompt_cache,
                cache_bytes=cache_bytes,
                binding=dict(live_binding),
            )
            evicted = self._store_session(session_id, new_entry)
        except Exception:
            if session_id in self.sessions:
                self._drop_session(session_id)
            self.mx.clear_cache()
            raise

        ordinary_request = ordinary["request"]
        assert type(ordinary_request) is dict
        return {
            "format": SESSION_RESPONSE_FORMAT,
            "protocol": SESSION_PROTOCOL,
            "request_id": request_id,
            "request": {
                "operation": operation,
                "prompt_token_count": len(full_prompt),
                "prompt_token_ids_sha256": one_shot._token_ids_sha256(full_prompt),
                "expected_prefix": dict(expected_prefix),
                "evaluated_prompt_token_count": len(evaluated_prompt),
                "evaluated_prompt_token_ids_sha256": one_shot._token_ids_sha256(
                    evaluated_prompt
                ),
                "max_tokens": request["max_tokens"],
                "stop_on_eos": request["stop_on_eos"],
                "greedy_strategy": GREEDY_STRATEGY,
                "requested_eos_token_id": request["eos_token_id"],
                "effective_eos_token_ids": ordinary_request["effective_eos_token_ids"],
                "binding": dict(live_binding),
            },
            "session": {
                "session_id": session_id,
                "prefix_token_count": len(new_prefix),
                "prefix_token_ids_sha256": new_entry.token_ids_sha256,
                "reused_prefix_token_count": reused_prefix_count,
                "evaluated_prompt_token_count": len(evaluated_prompt),
                "cache_bytes": cache_bytes,
            },
            "session_cache": {
                "policy": SESSION_CACHE_POLICY,
                "session_count": len(self.sessions),
                "total_cache_bytes": self.session_cache_bytes,
                "max_sessions": MAX_SESSIONS,
                "max_bytes": MAX_SESSION_CACHE_BYTES,
                "evicted_session_ids": evicted,
            },
            "model": ordinary["model"],
            "packages": ordinary["packages"],
            "runtime": ordinary["runtime"],
            "metrics": ordinary["metrics"],
            "generation": ordinary["generation"],
        }

    def reset_session(
        self, request_id: str, request: dict[str, object]
    ) -> dict[str, object]:
        session_id = request["session_id"]
        binding = request["binding"]
        expected_prefix = request["expected_prefix"]
        assert type(session_id) is str
        assert type(binding) is dict and type(expected_prefix) is dict
        live_binding = self.session_binding()
        if binding != live_binding:
            raise ServiceError(
                "session_binding_mismatch",
                "session binding differs from the loaded model or strategy",
            )
        entry = self.sessions.get(session_id)
        if entry is None:
            raise ServiceError("session_not_found", "session_id is not resident")
        if entry.binding != live_binding:
            self._drop_session(session_id)
            raise ServiceError(
                "session_binding_mismatch",
                "resident session binding failed closed",
            )
        if one_shot._token_ids_sha256(entry.token_ids) != entry.token_ids_sha256:
            self._drop_session(session_id)
            raise ServiceError(
                "session_cache_failed", "resident session prefix integrity failed"
            )
        if (
            expected_prefix["token_count"] != len(entry.token_ids)
            or expected_prefix["token_ids_sha256"] != entry.token_ids_sha256
        ):
            raise ServiceError(
                "session_prefix_mismatch",
                "reset prefix does not match the resident session",
            )
        removed = self._drop_session(session_id)
        assert removed is not None
        self.mx.clear_cache()
        return {
            "format": SESSION_RESET_FORMAT,
            "protocol": SESSION_PROTOCOL,
            "request_id": request_id,
            "session_id": session_id,
            "released_cache_bytes": removed.cache_bytes,
            "previous_prefix": {
                "format": SESSION_PREFIX_FORMAT,
                "token_count": len(removed.token_ids),
                "token_ids_sha256": removed.token_ids_sha256,
            },
            "binding": dict(live_binding),
            "session_cache": {
                "policy": SESSION_CACHE_POLICY,
                "session_count": len(self.sessions),
                "total_cache_bytes": self.session_cache_bytes,
                "max_sessions": MAX_SESSIONS,
                "max_bytes": MAX_SESSION_CACHE_BYTES,
            },
        }


def _validate_envelope(value: object) -> tuple[str, str, dict[str, object] | None]:
    if type(value) is not dict:
        raise ServiceError("invalid_request", "request root must be an object")
    request_id = _request_id(value.get("request_id"))
    message_format = value.get("format")
    if message_format == CONTROL_FORMAT:
        if set(value) != {"format", "request_id", "operation"}:
            raise ServiceError(
                "invalid_request", "control keys do not match the contract"
            )
        if value["operation"] != "shutdown":
            raise ServiceError("invalid_request", "unsupported control operation")
        return "shutdown", request_id, None
    if message_format == SESSION_CONTROL_FORMAT:
        if set(value) != {
            "format",
            "request_id",
            "operation",
            "session_id",
            "expected_prefix",
            "binding",
        }:
            raise ServiceError(
                "invalid_request", "session control keys do not match the contract"
            )
        if value["operation"] != "reset":
            raise ServiceError(
                "invalid_request", "unsupported session control operation"
            )
        return (
            "reset_session",
            request_id,
            {
                "session_id": _session_id(value["session_id"]),
                "expected_prefix": _session_prefix(value["expected_prefix"]),
                "binding": _session_binding(value["binding"]),
            },
        )
    if message_format not in (REQUEST_FORMAT, SESSION_REQUEST_FORMAT):
        raise ServiceError("invalid_request", "request format is unsupported")
    expected = {
        "format",
        "request_id",
        "prompt_token_ids",
        "max_tokens",
        "stop_on_eos",
    }
    if message_format == SESSION_REQUEST_FORMAT:
        expected |= {
            "session_id",
            "operation",
            "expected_prefix",
            "binding",
        }
    allowed = expected | {"eos_token_id"}
    observed = set(value)
    if not expected <= observed or not observed <= allowed:
        raise ServiceError("invalid_request", "request keys do not match the contract")
    one_shot_request = {
        "format": one_shot.REQUEST_FORMAT,
        "prompt_token_ids": value["prompt_token_ids"],
        "max_tokens": value["max_tokens"],
        "stop_on_eos": value["stop_on_eos"],
    }
    if "eos_token_id" in value:
        one_shot_request["eos_token_id"] = value["eos_token_id"]
    validated = one_shot._validate_request(one_shot_request)
    if message_format == REQUEST_FORMAT:
        return "generate", request_id, validated
    operation = value["operation"]
    if operation not in ("create", "append"):
        raise ServiceError("invalid_request", "unsupported session request operation")
    if validated["max_tokens"] < 1:
        raise ServiceError(
            "invalid_request", "session generation requires max_tokens >= 1"
        )
    if (
        len(validated["prompt_token_ids"]) + validated["max_tokens"]
        > one_shot.MAX_PROMPT_TOKENS
    ):
        raise ServiceError(
            "invalid_request", "session prompt plus generation exceeds token limit"
        )
    validated.update(
        {
            "session_id": _session_id(value["session_id"]),
            "operation": operation,
            "expected_prefix": _session_prefix(value["expected_prefix"]),
            "binding": _session_binding(value["binding"]),
        }
    )
    return "generate_session", request_id, validated


def _parse_arguments(argv: list[str]) -> argparse.Namespace:
    parser = JsonArgumentParser(add_help=False)
    parser.add_argument("--model-dir", required=True)
    return parser.parse_args(argv)


def _serve(service: LoadedService) -> int:
    _write_line(service.ready())
    observed_ids: set[str] = set()
    while True:
        payload = _read_line()
        if payload is None:
            raise ServiceError("protocol_closed", "stdin closed without shutdown")
        value = one_shot._parse_json(payload, "request")
        request_id: str | None = None
        try:
            if type(value) is not dict:
                raise ServiceError("invalid_request", "request root must be an object")
            request_id = _request_id(value.get("request_id"))
            if request_id in observed_ids:
                raise ServiceError(
                    "duplicate_request_id", "request_id was already used"
                )
            if len(observed_ids) >= MAX_REQUESTS:
                raise ServiceError("request_limit", "service request limit reached")
            observed_ids.add(request_id)
            operation, validated_request_id, request = _validate_envelope(value)
            assert validated_request_id == request_id
            if operation == "shutdown":
                _write_line(
                    {
                        "format": SHUTDOWN_FORMAT,
                        "protocol": PROTOCOL,
                        "request_id": request_id,
                    }
                )
                return 0
            assert request is not None
            if operation == "generate":
                response = service.generate(request_id, request)
            elif operation == "generate_session":
                response = service.generate_session(request_id, request)
            else:
                assert operation == "reset_session"
                response = service.reset_session(request_id, request)
            _write_line(response)
        except one_shot.WorkerError as error:
            if request_id is None:
                raise
            session_message = type(value) is dict and value.get("format") in (
                SESSION_REQUEST_FORMAT,
                SESSION_CONTROL_FORMAT,
            )
            _write_line(
                {
                    "format": (
                        SESSION_RESPONSE_ERROR_FORMAT
                        if session_message
                        else RESPONSE_ERROR_FORMAT
                    ),
                    "protocol": SESSION_PROTOCOL if session_message else PROTOCOL,
                    "request_id": request_id,
                    "error": _safe_error(error),
                }
            )


def main(argv: list[str] | None = None) -> int:
    try:
        arguments = _parse_arguments(sys.argv[1:] if argv is None else argv)
        with open(os.devnull, "w", encoding="utf-8") as dependency_output:
            with (
                contextlib.redirect_stdout(dependency_output),
                contextlib.redirect_stderr(dependency_output),
            ):
                service = LoadedService(arguments.model_dir)
        return _serve(service)
    except one_shot.WorkerError as error:
        one_shot._write_error(error)
        return 2
    except Exception as error:
        one_shot._write_error(
            ServiceError("internal_error", f"unexpected service failure: {error}")
        )
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
