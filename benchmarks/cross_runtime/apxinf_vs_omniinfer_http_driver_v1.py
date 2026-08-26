#!/usr/bin/env python3
"""Strict NON_FORMAL paired ApxInf-versus-OmniInfer HTTP diagnostic.

The two model processes must already be resident.  Arm B is the ApxInf
benchmark HTTP adapter and arm G is the OmniInfer gateway.  The driver opens
each measurement connection exactly once, clears the corresponding cache
outside the primary interval before every generation, and times one complete
pre-serialized HTTP/1.1 request wire through the complete response-body read.
JSON parsing and semantic validation intentionally happen after the end clock.

This driver can never produce formal evidence or an engine-ranking claim.  It
does not require the two engines to produce the same trajectory.  Instead, it
requires the same five-token EOG suppression policy and a deterministic
trajectory independently within each arm.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import http.client
import json
import math
import os
from pathlib import Path
import platform
import socket
import statistics
import sys
import time
import traceback
import urllib.parse
from typing import Any, Callable


FORMAT = "apxinf-vs-omniinfer-resident-http-paired-diagnostic-v1"
QUALIFICATION = "NON_FORMAL_DIAGNOSTIC_NOT_IN_FROZEN_GATE_CUSTODY"
APX_SERVER_QUALIFICATION = (
    "NON_FORMAL_DIAGNOSTIC_HTTP_ADAPTER_NOT_IN_FROZEN_GATE_CUSTODY"
)
MODEL_ALIAS = (
    "/Users/haiyan-mini/Agent4Kernel/models/Qwen3.5-0.8B-2fc063647-GGUF/"
    "Qwen3.5-0.8B-2fc063647-Q8_0-llama.gguf"
)
RENDERED_PROMPT = (
    "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
)
PROMPT_TOKEN_IDS = [
    248045,
    846,
    198,
    9419,
    248046,
    198,
    248045,
    74455,
    198,
    248068,
    271,
    248069,
    271,
]
SUPPRESSED_EOG_TOKEN_IDS = [248044, 248046, 248063, 248064, 248065]
REQUEST: dict[str, Any] = {
    "cache_prompt": False,
    "chat_template_kwargs": {"enable_thinking": False},
    "id_slot": 0,
    "ignore_eos": True,
    "max_tokens": 128,
    "messages": [{"content": "Hello", "role": "user"}],
    "model": MODEL_ALIAS,
    "reasoning_format": "none",
    "return_tokens": True,
    "seed": 0,
    "stream": False,
    "temperature": 0,
    "verbose": True,
}
REQUEST_SIZE = 383
REQUEST_SHA256 = "7773f5337693843f1e8cf3017b98868517cbddd3bc32649e550d8f2fec1d5cf6"
WARMUP_ORDERS = ("BG", "GB", "GB", "BG")
ODD_BLOCK_ORDERS = ("BG", "GB", "GB", "BG")
EVEN_BLOCK_ORDERS = ("GB", "BG", "BG", "GB")
MEASURED_BLOCKS = 16
PAIRS_PER_BLOCK = 4
EXPECTED_REQUESTS_PER_ARM = len(WARMUP_ORDERS) + MEASURED_BLOCKS * PAIRS_PER_BLOCK
T_CRITICAL_DF15_975 = 2.131449545559323
MAX_RESPONSE_BYTES = 16 * 1024 * 1024
MAX_QUIET_HOST_RECEIPT_BYTES = 1024 * 1024


class AdmissionError(RuntimeError):
    """A diagnostic contract or semantic admission failed."""


class TransportFailure(AdmissionError):
    """A terminal persistent-transport failure with captured observations."""

    def __init__(self, message: str, observation: dict[str, Any]):
        super().__init__(message)
        self.observation = observation


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AdmissionError(message)


def is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


REQUEST_BYTES = canonical_json_bytes(REQUEST)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> Any:
    raise ValueError(f"non-finite JSON number {value}")


def parse_strict_json_document(raw: bytes) -> dict[str, Any]:
    """Parse one UTF-8 JSON object, rejecting duplicates and non-finite values."""

    require(not raw.startswith(b"\xef\xbb\xbf"), "JSON response has a UTF-8 BOM")
    try:
        text = raw.decode("utf-8", "strict")
    except UnicodeDecodeError as error:
        raise AdmissionError("JSON response is not strict UTF-8") from error
    decoder = json.JSONDecoder(
        object_pairs_hook=_reject_duplicate_keys,
        parse_constant=_reject_nonfinite,
    )
    start = len(text) - len(text.lstrip())
    try:
        value, end = decoder.raw_decode(text, start)
    except (json.JSONDecodeError, ValueError) as error:
        raise AdmissionError(f"invalid strict JSON response: {error}") from error
    require(text[end:].strip() == "", "JSON response has trailing non-whitespace data")
    require(isinstance(value, dict), "JSON response is not an object")
    return value


def validate_static_contract() -> dict[str, Any]:
    require(len(REQUEST_BYTES) == REQUEST_SIZE, "canonical request size drifted")
    require(
        sha256_bytes(REQUEST_BYTES) == REQUEST_SHA256, "canonical request SHA drifted"
    )
    require(
        json.loads(REQUEST_BYTES) == REQUEST, "canonical request does not round-trip"
    )
    require(len(PROMPT_TOKEN_IDS) == 13, "prompt token contract drifted")
    require(len(SUPPRESSED_EOG_TOKEN_IDS) == 5, "EOG policy count drifted")
    require(
        len(set(SUPPRESSED_EOG_TOKEN_IDS)) == len(SUPPRESSED_EOG_TOKEN_IDS),
        "EOG policy contains duplicates",
    )
    schedule = declared_schedule()
    warmups = [entry for entry in schedule if entry["phase"] == "warmup"]
    measured = [entry for entry in schedule if entry["phase"] == "measured"]
    require(len(warmups) == 4, "warmup schedule count drifted")
    require(len(measured) == 64, "measured schedule count drifted")
    require(
        sum(entry["order"] == "BG" for entry in measured)
        == sum(entry["order"] == "GB" for entry in measured)
        == 32,
        "measured order schedule is not balanced",
    )
    return {
        "request_size_bytes": len(REQUEST_BYTES),
        "request_sha256": sha256_bytes(REQUEST_BYTES),
        "warmup_pairs": len(warmups),
        "measured_pairs": len(measured),
        "requests_per_arm": EXPECTED_REQUESTS_PER_ARM,
    }


def declared_schedule() -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for pair_index, order in enumerate(WARMUP_ORDERS, start=1):
        result.append(
            {
                "phase": "warmup",
                "block": 0,
                "pair_index": pair_index,
                "order": order,
            }
        )
    for block in range(1, MEASURED_BLOCKS + 1):
        orders = ODD_BLOCK_ORDERS if block % 2 else EVEN_BLOCK_ORDERS
        for pair_index, order in enumerate(orders, start=1):
            result.append(
                {
                    "phase": "measured",
                    "block": block,
                    "pair_index": pair_index,
                    "order": order,
                }
            )
    return result


def parse_loopback_endpoint(base_url: str, label: str) -> tuple[str, int]:
    parsed = urllib.parse.urlsplit(base_url)
    require(parsed.scheme == "http", f"{label} must use http://")
    require(parsed.hostname == "127.0.0.1", f"{label} must use 127.0.0.1")
    require(parsed.port is not None, f"{label} must include a port")
    require(
        parsed.username is None and parsed.password is None, f"{label} has credentials"
    )
    require(parsed.path in ("", "/"), f"{label} must not include a path")
    require(
        not parsed.query and not parsed.fragment, f"{label} has query or fragment data"
    )
    return parsed.hostname, parsed.port


def validate_http_path(path: str, label: str) -> None:
    require(path.startswith("/"), f"{label} does not start with /")
    require("\r" not in path and "\n" not in path, f"{label} contains a line break")
    try:
        path.encode("ascii")
    except UnicodeEncodeError as error:
        raise AdmissionError(f"{label} is not ASCII") from error


def clock_receipt() -> dict[str, Any]:
    info = time.get_clock_info("monotonic")
    return {
        "clock": "time.monotonic_ns",
        "implementation": info.implementation,
        "resolution_ns": int(round(info.resolution * 1_000_000_000)),
        "monotonic": info.monotonic,
        "adjustable": info.adjustable,
    }


def host_observation() -> dict[str, Any]:
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version,
        "logical_cpu_count": os.cpu_count(),
        "load_average": list(os.getloadavg()),
    }


class PersistentHttpConnection:
    """One raw HTTP/1.1 connection; reconnect and request retry are impossible."""

    def __init__(
        self,
        base_url: str,
        label: str,
        *,
        timeout_seconds: float = 600.0,
        socket_factory: Callable[..., Any] = socket.create_connection,
        response_factory: Callable[..., Any] = http.client.HTTPResponse,
        clock_ns: Callable[[], int] = time.monotonic_ns,
    ):
        self.host, self.port = parse_loopback_endpoint(base_url, label)
        require(timeout_seconds > 0.0, f"{label} timeout must be positive")
        self.base_url = base_url.rstrip("/")
        self.label = label
        self.timeout_seconds = timeout_seconds
        self.socket_factory = socket_factory
        self.response_factory = response_factory
        self.clock_ns = clock_ns
        self.sock: Any | None = None
        self.baseline: dict[str, Any] | None = None
        self.request_count = 0

    def socket_identity(self) -> dict[str, Any]:
        require(self.sock is not None, f"{self.label} socket is absent")
        descriptor = self.sock.fileno()
        require(
            is_int(descriptor) and descriptor >= 0, f"{self.label} socket is closed"
        )
        return {
            "python_object_id": id(self.sock),
            "file_descriptor": descriptor,
            "local_address": list(self.sock.getsockname()),
            "peer_address": list(self.sock.getpeername()),
        }

    def connect(self) -> dict[str, Any]:
        require(self.sock is None, f"{self.label} was already connected")
        self.sock = self.socket_factory(
            (self.host, self.port), timeout=self.timeout_seconds
        )
        try:
            self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except (AttributeError, OSError):
            pass
        self.baseline = self.socket_identity()
        return dict(self.baseline)

    def serialize_request(self, method: str, path: str, body: bytes | None) -> bytes:
        require(method in ("GET", "POST"), "only GET and POST are admitted")
        validate_http_path(path, "HTTP request path")
        require(
            body is None or isinstance(body, bytes), "HTTP body must be bytes or absent"
        )
        headers = [
            f"{method} {path} HTTP/1.1",
            f"Host: {self.host}:{self.port}",
            "Accept: application/json",
            "Accept-Encoding: identity",
            "Connection: keep-alive",
        ]
        if body is not None:
            if body:
                headers.append("Content-Type: application/json")
            headers.append(f"Content-Length: {len(body)}")
        return ("\r\n".join(headers) + "\r\n\r\n").encode("ascii") + (body or b"")

    def request_json(
        self,
        method: str,
        path: str,
        body: bytes | None,
        *,
        primary_timed: bool,
        semantic_validator: Callable[[dict[str, Any]], dict[str, Any]],
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        require(callable(semantic_validator), "semantic validator is not callable")
        require(self.baseline is not None, f"{self.label} was not preconnected")
        before = self.socket_identity()
        require(before == self.baseline, f"{self.label} socket changed before request")
        wire = self.serialize_request(method, path, body)
        body_bytes = body or b""
        body_offset = len(wire) - len(body_bytes)
        require(wire[body_offset:] == body_bytes, "serialized request body differs")
        canonical_body = body == REQUEST_BYTES
        timing_events = ["complete-http-request-wire-serialized"]
        started_ns: int | None = None
        ended_ns: int | None = None
        raw = b""
        sendall_calls = 0
        stage = "before-sendall"
        try:
            started_ns = self.clock_ns()
            self.sock.sendall(wire)
            sendall_calls += 1
            timing_events.extend(
                [
                    "start-monotonic-timestamp-captured",
                    "single-sendall-complete",
                ]
            )
            stage = "response-headers"
            response = self.response_factory(self.sock)
            response.begin()
            timing_events.append("response-headers-read")
            status = response.status
            version = response.version
            will_close = response.will_close
            headers = response.getheaders()
            stage = "response-body"
            try:
                raw = response.read(MAX_RESPONSE_BYTES + 1)
            except http.client.IncompleteRead as error:
                raw = bytes(error.partial)
                raise
            ended_ns = self.clock_ns()
            timing_events.extend(
                [
                    "complete-response-body-read",
                    "end-monotonic-timestamp-captured",
                ]
            )
            response.close()
            require(
                len(raw) <= MAX_RESPONSE_BYTES, f"{self.label} response is oversized"
            )
            require(status == 200, f"{self.label} returned HTTP {status}")
            require(version == 11, f"{self.label} response is not HTTP/1.1")
            require(will_close is False, f"{self.label} response closes the connection")
            header_map: dict[str, list[str]] = {}
            for name, value in headers:
                header_map.setdefault(name.lower(), []).append(value)
            content_types = header_map.get("content-type", [])
            require(
                len(content_types) == 1
                and content_types[0].lower().startswith("application/json"),
                f"{self.label} response Content-Type is not one application/json value",
            )
            content_lengths = header_map.get("content-length", [])
            require(len(content_lengths) == 1, f"{self.label} needs one Content-Length")
            require(
                content_lengths[0].isdigit() and int(content_lengths[0]) == len(raw),
                f"{self.label} response Content-Length differs",
            )
            require(
                not header_map.get("transfer-encoding"),
                f"{self.label} chunked response is outside this diagnostic contract",
            )
            stage = "strict-json-parse-after-wall"
            parse_started_ns = self.clock_ns()
            payload = parse_strict_json_document(raw)
            parse_ended_ns = self.clock_ns()
            timing_events.append("strict-json-parse-complete-after-end")
            stage = "semantic-validation-after-wall"
            validation_started_ns = self.clock_ns()
            validated = semantic_validator(payload)
            validation_ended_ns = self.clock_ns()
            timing_events.append("semantic-validation-complete-after-end")
            after = self.socket_identity()
            require(
                after == self.baseline, f"{self.label} socket reconnected or closed"
            )
        except BaseException as error:
            observation = {
                "connection": self.label,
                "stage": stage,
                "started_monotonic_ns": started_ns,
                "ended_monotonic_ns": ended_ns,
                "sendall_call_count": sendall_calls,
                "request_wire_size_bytes": len(wire),
                "request_wire_sha256": sha256_bytes(wire),
                "raw_response_size_bytes": len(raw),
                "raw_response_sha256": sha256_bytes(raw),
                "raw_response_base64": base64.b64encode(raw).decode("ascii"),
                "exception_type": type(error).__name__,
                "message": str(error),
            }
            raise TransportFailure(
                f"{self.label} failed terminally during {stage}: {error}", observation
            ) from error
        self.request_count += 1
        assert started_ns is not None and ended_ns is not None
        transport = {
            "connection": self.label,
            "connection_generation": 1,
            "request_index_on_connection": self.request_count,
            "socket": dict(self.baseline),
            "socket_start_end_equal": True,
            "reconnect_count": 0,
            "method": method,
            "path": path,
            "primary_timed_interval": primary_timed,
            "request_wire_size_bytes": len(wire),
            "request_wire_sha256": sha256_bytes(wire),
            "request_wire_base64": base64.b64encode(wire).decode("ascii"),
            "request_body_offset_bytes": body_offset,
            "request_body_size_bytes": len(body_bytes),
            "request_body_sha256": sha256_bytes(body_bytes),
            "request_body_equals_canonical_383_bytes": canonical_body,
            "complete_request_serialized_before_start": True,
            "single_sendall_call_count": sendall_calls,
            "started_monotonic_ns": started_ns,
            "ended_monotonic_ns": ended_ns,
            "client_full_response_wall_ns": ended_ns - started_ns,
            "client_full_response_wall_ms": (ended_ns - started_ns) / 1_000_000,
            "wall_start_boundary": (
                "immediately-before-single-sendall-of-complete-pre-serialized-http-wire"
            ),
            "wall_end_boundary": "immediately-after-complete-response-body-read",
            "json_parse_excluded_from_wall": True,
            "json_parse_ns": parse_ended_ns - parse_started_ns,
            "semantic_validation_excluded_from_wall": True,
            "semantic_validation_ns": validation_ended_ns - validation_started_ns,
            "timing_event_order": timing_events,
            "status": status,
            "http_version": version,
            "response_headers": header_map,
            "response_size_bytes": len(raw),
            "response_sha256": sha256_bytes(raw),
            "response_base64": base64.b64encode(raw).decode("ascii"),
        }
        return payload, validated, transport

    def close(self) -> None:
        if self.sock is not None:
            self.sock.close()
            self.sock = None


def validate_apx_health(payload: dict[str, Any]) -> dict[str, Any]:
    require(payload.get("status") == "ok", "ApxInf health status differs")
    require(payload.get("resident") is True, "ApxInf does not report a resident model")
    require(
        payload.get("formal_evidence_eligible") is False, "ApxInf formal flag differs"
    )
    require(
        payload.get("qualification") == APX_SERVER_QUALIFICATION,
        "ApxInf qualification differs",
    )
    return {"status": "ok", "resident": True, "formal_evidence_eligible": False}


def validate_apx_state(
    payload: dict[str, Any], *, expected_completed: int
) -> dict[str, Any]:
    require(payload.get("ok") is True, "ApxInf state is not acknowledged")
    require(
        payload.get("formal_evidence_eligible") is False,
        "ApxInf state formal flag differs",
    )
    require(
        payload.get("qualification") == APX_SERVER_QUALIFICATION,
        "ApxInf state qualification differs",
    )
    require(
        payload.get("connection_generation") == 1,
        "ApxInf connection generation differs",
    )
    require(
        payload.get("expected_generation_requests") == EXPECTED_REQUESTS_PER_ARM,
        "ApxInf expected generation count differs",
    )
    require(
        payload.get("completed_generation_requests") == expected_completed,
        "ApxInf completed generation count differs",
    )
    require(
        payload.get("armed_epoch") is None, "ApxInf reset epoch is unexpectedly armed"
    )
    require(payload.get("poisoned") is False, "ApxInf server is poisoned")
    canonical = payload.get("canonical_request")
    require(isinstance(canonical, dict), "ApxInf state lacks canonical request receipt")
    require(
        canonical.get("size_bytes") == REQUEST_SIZE, "ApxInf canonical size differs"
    )
    require(canonical.get("sha256") == REQUEST_SHA256, "ApxInf canonical SHA differs")
    engine = payload.get("engine")
    require(isinstance(engine, dict), "ApxInf state lacks engine receipt")
    require(engine.get("resident_model") is True, "ApxInf model is not resident")
    require(
        engine.get("resident_tokenizer") is True, "ApxInf tokenizer is not resident"
    )
    require(
        engine.get("generation_policy") == "-inf-before-greedy", "ApxInf policy differs"
    )
    require(
        engine.get("suppressed_eog_token_ids") == SUPPRESSED_EOG_TOKEN_IDS,
        "ApxInf state EOG list differs",
    )
    candidate_commit = engine.get("candidate_commit")
    require(
        isinstance(candidate_commit, str)
        and len(candidate_commit) == 40
        and all(character in "0123456789abcdef" for character in candidate_commit),
        "ApxInf candidate commit is invalid",
    )
    return {
        "completed_generation_requests": expected_completed,
        "candidate_commit": candidate_commit,
        "actual_model_dir": engine.get("actual_model_dir"),
        "source_lock": engine.get("source_lock"),
        "generation_policy": engine.get("generation_policy"),
        "suppressed_eog_token_ids": engine.get("suppressed_eog_token_ids"),
    }


def validate_omni_health(payload: dict[str, Any]) -> dict[str, Any]:
    require(payload.get("status") == "ok", "OmniInfer health status differs")
    omni = payload.get("omni")
    require(isinstance(omni, dict), "OmniInfer deep health lacks runtime state")
    require(omni.get("backend_ready") is True, "OmniInfer backend is not ready")
    backend_health = payload.get("backend_health")
    require(
        isinstance(backend_health, dict) and backend_health.get("status") == "ok",
        "OmniInfer deep backend health differs",
    )
    return {"status": "ok", "backend_ready": True, "backend_health": "ok"}


def validate_omni_state(payload: dict[str, Any]) -> dict[str, Any]:
    require(
        payload.get("backend_ready") is True, "OmniInfer state backend is not ready"
    )
    backend_pid = payload.get("backend_pid")
    generation = payload.get("generation")
    require(is_int(backend_pid) and backend_pid > 0, "OmniInfer backend PID is invalid")
    require(
        is_int(generation) and generation > 0,
        "OmniInfer generation identity is invalid",
    )
    client_endpoint = payload.get("client_endpoint")
    require(isinstance(client_endpoint, str), "OmniInfer client endpoint is absent")
    parse_loopback_endpoint(client_endpoint, "OmniInfer resident backend endpoint")
    model_path = payload.get("model_path")
    require(model_path == MODEL_ALIAS, "OmniInfer loaded model path differs")
    return {
        "backend": payload.get("backend"),
        "backend_pid": backend_pid,
        "generation": generation,
        "client_endpoint": client_endpoint,
        "model_path": model_path,
    }


def validate_apx_clear(payload: dict[str, Any], expected_epoch: int) -> dict[str, Any]:
    require(payload.get("ok") is True, "ApxInf cache clear is not acknowledged")
    require(
        payload.get("formal_evidence_eligible") is None,
        "unexpected ApxInf clear formal flag",
    )
    require(
        payload.get("qualification") == APX_SERVER_QUALIFICATION,
        "ApxInf clear qualification differs",
    )
    require(
        payload.get("cache_policy")
        == "checked-reset-exactly-once-before-each-generation",
        "ApxInf checked reset policy differs",
    )
    require(payload.get("cleared_slots") == [0], "ApxInf clear slot differs")
    require(payload.get("epoch") == expected_epoch, "ApxInf reset epoch differs")
    require(
        payload.get("checked_reset_calls_this_epoch") == 1,
        "ApxInf checked reset call count differs",
    )
    return {
        "acknowledged": True,
        "cleared_slots": [0],
        "epoch": expected_epoch,
        "checked_reset_calls": 1,
    }


def validate_omni_clear(
    payload: dict[str, Any], *, contract: str, path: str
) -> dict[str, Any]:
    if contract == "omni-gateway":
        require(
            payload.get("ok") is True, "OmniInfer gateway clear is not acknowledged"
        )
        require(
            payload.get("cache_policy") == "cleared_each_run",
            "OmniInfer gateway clear policy differs",
        )
        require(
            payload.get("cleared_slots") == [0], "OmniInfer gateway clear slot differs"
        )
        return {"contract": contract, "acknowledged": True, "cleared_slots": [0]}
    require(contract == "llama-slot-erase", "unknown OmniInfer clear contract")
    require(
        path == "/slots/0?action=erase", "llama slot erase path is not exact slot 0"
    )
    require(payload.get("id_slot") == 0, "llama slot erase response slot differs")
    n_erased = payload.get("n_erased")
    require(is_int(n_erased) and n_erased >= 0, "llama slot erase count is invalid")
    return {
        "contract": contract,
        "acknowledged": True,
        "cleared_slots": [0],
        "n_erased": n_erased,
    }


def validate_generation_policy(settings: Any, arm: str) -> dict[str, Any]:
    require(isinstance(settings, dict), f"arm {arm} generation settings are absent")
    require(settings.get("ignore_eos") is True, f"arm {arm} ignore_eos differs")
    require(settings.get("max_tokens") == 128, f"arm {arm} max_tokens differs")
    require(settings.get("seed") == 0, f"arm {arm} seed differs")
    temperature = settings.get("temperature")
    require(
        isinstance(temperature, (int, float))
        and not isinstance(temperature, bool)
        and float(temperature) == 0.0,
        f"arm {arm} temperature differs",
    )
    if arm == "B":
        require(
            settings.get("policy") == "-inf-before-greedy",
            "ApxInf generation policy differs",
        )
        require(
            settings.get("suppressed_eog_token_ids") == SUPPRESSED_EOG_TOKEN_IDS,
            "ApxInf generation EOG list differs",
        )
        return {
            "runtime": "apxinf",
            "semantics": "negative-infinity-before-greedy",
            "suppressed_eog_token_ids": list(SUPPRESSED_EOG_TOKEN_IDS),
            "source_field": "suppressed_eog_token_ids",
        }
    require(arm == "G", "unknown generation arm")
    logit_bias = settings.get("logit_bias")
    require(isinstance(logit_bias, list), "OmniInfer logit_bias is absent")
    require(
        len(logit_bias) == len(SUPPRESSED_EOG_TOKEN_IDS),
        "OmniInfer logit_bias count differs",
    )
    observed: list[int] = []
    for entry in logit_bias:
        require(isinstance(entry, dict), "OmniInfer logit_bias entry is not an object")
        require(
            set(entry) == {"bias", "token"}, "OmniInfer logit_bias entry shape differs"
        )
        token = entry.get("token")
        require(is_int(token), "OmniInfer logit_bias token is invalid")
        require(
            entry.get("bias") is None,
            "OmniInfer ignore_eos bias is not JSON null (-inf serialization)",
        )
        observed.append(token)
    require(
        sorted(observed) == SUPPRESSED_EOG_TOKEN_IDS, "OmniInfer EOG bias tokens differ"
    )
    return {
        "runtime": "omniinfer-llama.cpp",
        "semantics": "negative-infinity-before-greedy",
        "suppressed_eog_token_ids": list(SUPPRESSED_EOG_TOKEN_IDS),
        "source_field": "logit_bias-null-serialization-of-negative-infinity",
    }


def validate_chat_response(response: dict[str, Any], arm: str) -> dict[str, Any]:
    runtime = "apxinf" if arm == "B" else "omniinfer"
    require(response.get("object") == "chat.completion", f"{runtime} object differs")
    require(response.get("model") == MODEL_ALIAS, f"{runtime} response model differs")
    choices = response.get("choices")
    require(
        isinstance(choices, list) and len(choices) == 1,
        f"{runtime} choice count differs",
    )
    choice = choices[0]
    require(isinstance(choice, dict), f"{runtime} choice is not an object")
    require(choice.get("finish_reason") == "length", f"{runtime} finish reason differs")
    message = choice.get("message")
    require(isinstance(message, dict), f"{runtime} response message is absent")
    require(message.get("role") == "assistant", f"{runtime} message role differs")
    content = message.get("content")
    require(isinstance(content, str), f"{runtime} content is not text")
    usage = response.get("usage")
    require(isinstance(usage, dict), f"{runtime} usage is absent")
    require(usage.get("prompt_tokens") == 13, f"{runtime} prompt usage differs")
    require(
        usage.get("completion_tokens") == 128, f"{runtime} completion usage differs"
    )
    require(usage.get("total_tokens") == 141, f"{runtime} total usage differs")
    verbose = response.get("__verbose")
    require(isinstance(verbose, dict), f"{runtime} verbose receipt is absent")
    require(verbose.get("tokens_evaluated") == 13, f"{runtime} evaluated count differs")
    require(
        verbose.get("tokens_predicted") == 128, f"{runtime} predicted count differs"
    )
    require(verbose.get("stop_type") == "limit", f"{runtime} stop type differs")
    require(
        verbose.get("prompt") == RENDERED_PROMPT, f"{runtime} rendered prompt differs"
    )
    tokens = verbose.get("tokens")
    require(
        isinstance(tokens, list)
        and len(tokens) == 128
        and all(is_int(token) and token >= 0 for token in tokens),
        f"{runtime} raw generated token vector differs",
    )
    eog_hits = sorted(set(tokens).intersection(SUPPRESSED_EOG_TOKEN_IDS))
    require(not eog_hits, f"{runtime} generated suppressed EOG tokens {eog_hits}")
    if arm == "B":
        require(
            verbose.get("qualification") == APX_SERVER_QUALIFICATION,
            "ApxInf response qualification differs",
        )
        require(
            verbose.get("formal_evidence_eligible") is False,
            "ApxInf response formal flag differs",
        )
        require(
            verbose.get("prompt_token_ids") == PROMPT_TOKEN_IDS,
            "ApxInf prompt token IDs differ",
        )
    policy = validate_generation_policy(verbose.get("generation_settings"), arm)
    token_hash = sha256_bytes(canonical_json_bytes(tokens))
    content_bytes = content.encode("utf-8")
    return {
        "runtime": runtime,
        "finish_reason": "length",
        "usage": {"prompt_tokens": 13, "completion_tokens": 128, "total_tokens": 141},
        "prompt_token_ids": list(PROMPT_TOKEN_IDS),
        "generated_token_ids": list(tokens),
        "generated_token_ids_sha256": token_hash,
        "generated_eog_hits": [],
        "content": content,
        "content_sha256": sha256_bytes(content_bytes),
        "generation_policy": policy,
    }


def _mean(values: list[float], label: str) -> float:
    require(values, f"{label} has no observations")
    require(
        all(math.isfinite(value) and value > 0.0 for value in values),
        f"{label} is invalid",
    )
    return statistics.fmean(values)


def summary_stats(values: list[float], label: str) -> dict[str, Any]:
    mean = _mean(values, label)
    require(len(values) > 1, f"{label} needs multiple observations")
    population_sd = statistics.pstdev(values)
    return {
        "count": len(values),
        "samples": values,
        "mean": mean,
        "median": statistics.median(values),
        "population_sd": population_sd,
        "population_cv": population_sd / mean,
        "min": min(values),
        "max": max(values),
    }


def block_t_interval(values: list[float], label: str) -> dict[str, Any]:
    require(len(values) == MEASURED_BLOCKS, f"{label} block count differs")
    require(
        all(math.isfinite(value) for value in values), f"{label} has a non-finite block"
    )
    mean = statistics.fmean(values)
    sample_sd = statistics.stdev(values)
    standard_error = sample_sd / math.sqrt(len(values))
    half_width = T_CRITICAL_DF15_975 * standard_error
    return {
        "block_means": values,
        "mean": mean,
        "sample_sd": sample_sd,
        "standard_error": standard_error,
        "degrees_of_freedom": 15,
        "t_critical": T_CRITICAL_DF15_975,
        "ci95_lower": mean - half_width,
        "ci95_upper": mean + half_width,
        "ci95_half_width": half_width,
    }


def analyze_measured_pairs(
    pairs: list[dict[str, Any]], *, quiet_host_passed: bool
) -> dict[str, Any]:
    require(
        len(pairs) == MEASURED_BLOCKS * PAIRS_PER_BLOCK, "measured pair count differs"
    )
    apx_walls: list[float] = []
    omni_walls: list[float] = []
    deltas: list[float] = []
    log_ratios: list[float] = []
    by_block_delta: list[list[float]] = [[] for _ in range(MEASURED_BLOCKS)]
    by_block_log: list[list[float]] = [[] for _ in range(MEASURED_BLOCKS)]
    by_order: dict[str, list[float]] = {"BG": [], "GB": []}
    for pair in pairs:
        samples = pair.get("samples")
        require(
            isinstance(samples, list) and len(samples) == 2, "pair sample count differs"
        )
        by_arm = {sample["arm"]: sample for sample in samples}
        require(set(by_arm) == {"B", "G"}, "pair arms differ")
        apx = float(by_arm["B"]["transport"]["client_full_response_wall_ms"])
        omni = float(by_arm["G"]["transport"]["client_full_response_wall_ms"])
        require(apx > 0.0 and omni > 0.0, "pair wall time is not positive")
        delta = omni - apx
        log_ratio = math.log(omni / apx)
        block_index = int(pair["block"]) - 1
        require(0 <= block_index < MEASURED_BLOCKS, "pair block is invalid")
        order = pair["order"]
        require(order in by_order, "pair order is invalid")
        apx_walls.append(apx)
        omni_walls.append(omni)
        deltas.append(delta)
        log_ratios.append(log_ratio)
        by_block_delta[block_index].append(delta)
        by_block_log[block_index].append(log_ratio)
        by_order[order].append(delta)
    require(
        all(len(block) == PAIRS_PER_BLOCK for block in by_block_delta),
        "measured block population differs",
    )
    block_delta = [statistics.fmean(block) for block in by_block_delta]
    block_log = [statistics.fmean(block) for block in by_block_log]
    delta_interval = block_t_interval(block_delta, "OmniInfer-minus-ApxInf delta")
    log_interval = block_t_interval(block_log, "OmniInfer-over-ApxInf log ratio")
    ratio_interval = {
        **log_interval,
        "geometric_mean_ratio": math.exp(log_interval["mean"]),
        "ci95_ratio_lower": math.exp(log_interval["ci95_lower"]),
        "ci95_ratio_upper": math.exp(log_interval["ci95_upper"]),
    }
    apx_stats = summary_stats(apx_walls, "ApxInf wall")
    omni_stats = summary_stats(omni_walls, "OmniInfer wall")
    pooled_wall = statistics.fmean(apx_walls + omni_walls)
    order_means = {
        order: statistics.fmean(values) for order, values in by_order.items()
    }
    order_difference = abs(order_means["BG"] - order_means["GB"])
    front_back_difference = abs(
        statistics.fmean(block_delta[:8]) - statistics.fmean(block_delta[8:])
    )
    drift_threshold = max(2.0, 0.002 * pooled_wall)
    gates = {
        "quiet_host_operator_gate_passed": quiet_host_passed,
        "apxinf_wall_population_cv_le_1pct": apx_stats["population_cv"] <= 0.01,
        "omniinfer_wall_population_cv_le_1pct": omni_stats["population_cv"] <= 0.01,
        "paired_delta_sd_over_pooled_wall_le_1pct": (
            statistics.pstdev(deltas) / pooled_wall <= 0.01
        ),
        "order_stratum_difference_within_threshold": order_difference
        <= drift_threshold,
        "front_back_block_difference_within_threshold": (
            front_back_difference <= drift_threshold
        ),
        "primary_ci95_half_width_le_2ms": delta_interval["ci95_half_width"] <= 2.0,
    }
    return {
        "primary_omniinfer_minus_apxinf_client_wall_ms": delta_interval,
        "secondary_omniinfer_over_apxinf_client_wall_ratio": ratio_interval,
        "apxinf_client_wall_ms": apx_stats,
        "omniinfer_client_wall_ms": omni_stats,
        "raw_paired_deltas_ms": deltas,
        "raw_paired_ratios": [math.exp(value) for value in log_ratios],
        "order_strata": {"samples": by_order, "means": order_means},
        "order_stratum_difference_ms": order_difference,
        "front_back_block_difference_ms": front_back_difference,
        "order_and_drift_threshold_ms": drift_threshold,
        "diagnostic_stability_gates": gates,
        "all_diagnostic_stability_gates_passed": all(gates.values()),
        "qualification": QUALIFICATION,
        "formal_summary_allowed": False,
        "engine_winner_or_ranking_claim_allowed": False,
    }


def validate_per_runtime_determinism(
    warmup_pairs: list[dict[str, Any]], measured_pairs: list[dict[str, Any]]
) -> dict[str, Any]:
    by_arm: dict[str, dict[str, set[str]]] = {
        "B": {"tokens": set(), "content": set()},
        "G": {"tokens": set(), "content": set()},
    }
    counts = {"B": 0, "G": 0}
    for pair in [*warmup_pairs, *measured_pairs]:
        for sample in pair["samples"]:
            arm = sample["arm"]
            require(arm in by_arm, "determinism receipt has an unknown arm")
            validated = sample["validated"]
            by_arm[arm]["tokens"].add(validated["generated_token_ids_sha256"])
            by_arm[arm]["content"].add(validated["content_sha256"])
            counts[arm] += 1
    result: dict[str, Any] = {}
    for arm, runtime in (("B", "apxinf"), ("G", "omniinfer")):
        require(
            counts[arm] == EXPECTED_REQUESTS_PER_ARM, f"arm {arm} sample count differs"
        )
        require(
            len(by_arm[arm]["tokens"]) == 1,
            f"arm {arm} trajectory is not deterministic",
        )
        require(
            len(by_arm[arm]["content"]) == 1, f"arm {arm} content is not deterministic"
        )
        result[runtime] = {
            "sample_count": counts[arm],
            "generated_token_ids_sha256": next(iter(by_arm[arm]["tokens"])),
            "content_sha256": next(iter(by_arm[arm]["content"])),
            "deterministic_within_runtime": True,
        }
    return {
        "per_runtime": result,
        "cross_runtime_trajectory_equality_required": False,
        "cross_runtime_trajectory_hash_comparison_omitted": True,
    }


class ExclusiveJsonOutput:
    """Reserve one absolute output path and write one durable JSON document."""

    def __init__(self, path_value: str):
        path = Path(path_value)
        require(path.is_absolute(), "--output must be an absolute path")
        parent = path.parent.resolve(strict=True)
        require(parent.is_dir(), "--output parent is not a directory")
        self.path = parent / path.name
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        self.fd = os.open(self.path, flags, 0o600)
        self.written = False
        self.write_started = False

    def write(self, value: dict[str, Any]) -> dict[str, Any]:
        require(not self.write_started, "exclusive output write was already attempted")
        self.write_started = True
        raw = canonical_json_bytes(value) + b"\n"
        try:
            offset = 0
            while offset < len(raw):
                written = os.write(self.fd, raw[offset:])
                if written <= 0:
                    raise OSError("exclusive output write made no progress")
                offset += written
            os.fsync(self.fd)
            os.close(self.fd)
            self.fd = -1
        except BaseException:
            if self.fd >= 0:
                os.close(self.fd)
                self.fd = -1
            raise
        self.written = True
        return {
            "path": str(self.path),
            "size_bytes": len(raw),
            "sha256": sha256_bytes(raw),
            "exclusive_create": True,
        }

    def close_empty(self) -> None:
        if self.fd >= 0:
            os.close(self.fd)
            self.fd = -1


def quiet_host_gate(args: argparse.Namespace) -> dict[str, Any]:
    receipt: dict[str, Any] | None = None
    if args.quiet_host_receipt is not None:
        path = Path(args.quiet_host_receipt)
        require(path.is_absolute(), "--quiet-host-receipt must be absolute")
        metadata = path.lstat()
        require(
            path.is_file() and not path.is_symlink(),
            "quiet-host receipt is not a direct file",
        )
        require(
            metadata.st_size <= MAX_QUIET_HOST_RECEIPT_BYTES,
            "quiet-host receipt is oversized",
        )
        raw = path.read_bytes()
        parsed = parse_strict_json_document(raw)
        receipt = {
            "path": str(path.resolve(strict=True)),
            "size_bytes": len(raw),
            "sha256": sha256_bytes(raw),
            "payload": parsed,
        }
    if args.quiet_host_status == "not-evaluated":
        require(
            receipt is None,
            "not-evaluated quiet-host status must not include a receipt",
        )
    else:
        require(
            receipt is not None,
            "passed/failed quiet-host status requires --quiet-host-receipt",
        )
        require(
            receipt["payload"].get("passed") is (args.quiet_host_status == "passed"),
            "quiet-host status differs from receipt passed field",
        )
    return {
        "status": args.quiet_host_status,
        "evaluated": args.quiet_host_status != "not-evaluated",
        "passed": args.quiet_host_status == "passed",
        "operator_supplied_receipt": receipt,
        "driver_does_not_promote_this_to_formal_evidence": True,
    }


def _control_request(
    connection: PersistentHttpConnection,
    method: str,
    path: str,
    body: bytes | None,
    validator: Callable[[dict[str, Any]], dict[str, Any]],
) -> dict[str, Any]:
    response, validated, transport = connection.request_json(
        method,
        path,
        body,
        primary_timed=False,
        semantic_validator=validator,
    )
    return {"response": response, "validated": validated, "transport": transport}


def run_one_sample(
    *,
    arm: str,
    sequence_index: int,
    schedule_entry: dict[str, Any],
    apx: PersistentHttpConnection,
    omni: PersistentHttpConnection,
    omni_clear: PersistentHttpConnection,
    args: argparse.Namespace,
) -> dict[str, Any]:
    if arm == "B":
        clear_response, clear_validated, clear_transport = apx.request_json(
            "POST",
            "/apxinf/cache/clear",
            b"{}",
            primary_timed=False,
            semantic_validator=lambda value: validate_apx_clear(value, sequence_index),
        )
        target = apx
    else:
        require(arm == "G", "sample arm is invalid")
        clear_body = b"{}" if args.omni_clear_body == "empty-object" else b""
        clear_response, clear_validated, clear_transport = omni_clear.request_json(
            "POST",
            args.omni_clear_path,
            clear_body,
            primary_timed=False,
            semantic_validator=lambda value: validate_omni_clear(
                value,
                contract=args.omni_clear_contract,
                path=args.omni_clear_path,
            ),
        )
        target = omni
    response, validated, transport = target.request_json(
        "POST",
        "/v1/chat/completions",
        REQUEST_BYTES,
        primary_timed=schedule_entry["phase"] == "measured",
        semantic_validator=lambda value: validate_chat_response(value, arm),
    )
    require(
        clear_transport["ended_monotonic_ns"] <= transport["started_monotonic_ns"],
        "cache clear did not complete before generation timing started",
    )
    return {
        "sequence_index_for_arm": sequence_index,
        "phase": schedule_entry["phase"],
        "block": schedule_entry["block"],
        "pair_index": schedule_entry["pair_index"],
        "order": schedule_entry["order"],
        "arm": arm,
        "runtime": "apxinf-resident-http" if arm == "B" else "omniinfer-resident-http",
        "cache_clear_immediately_before": {
            "outside_primary_timed_interval": True,
            "response": clear_response,
            "validated": clear_validated,
            "transport": clear_transport,
        },
        "request_contract": {
            "size_bytes": len(REQUEST_BYTES),
            "sha256": sha256_bytes(REQUEST_BYTES),
            "canonical_body_identical_between_arms": True,
        },
        "transport": transport,
        "validated": validated,
        "response": response,
    }


def run_diagnostic(
    args: argparse.Namespace, progress: dict[str, Any]
) -> dict[str, Any]:
    static = validate_static_contract()
    quiet_gate = quiet_host_gate(args)
    host_start = host_observation()
    progress["host_observation_start"] = host_start
    apx_address = parse_loopback_endpoint(args.apx_endpoint, "ApxInf endpoint")
    omni_address = parse_loopback_endpoint(args.omni_endpoint, "OmniInfer endpoint")
    parse_loopback_endpoint(args.omni_clear_endpoint, "OmniInfer clear endpoint")
    require(
        apx_address != omni_address,
        "ApxInf and OmniInfer generation endpoints are identical",
    )
    validate_http_path(args.omni_health_path, "OmniInfer health path")
    validate_http_path(args.omni_state_path, "OmniInfer state path")
    validate_http_path(args.omni_clear_path, "OmniInfer clear path")
    if args.omni_clear_contract == "omni-gateway":
        require(
            args.omni_clear_body == "empty-object",
            "gateway clear body must be exact {}",
        )
    else:
        require(args.omni_clear_body == "empty", "llama slot erase body must be empty")
        require(
            args.omni_clear_path == "/slots/0?action=erase",
            "llama slot erase must target exact /slots/0?action=erase",
        )

    timeout = float(args.timeout_seconds)
    require(math.isfinite(timeout) and timeout > 0.0, "timeout must be positive")
    apx = PersistentHttpConnection(
        args.apx_endpoint, "apxinf-single", timeout_seconds=timeout
    )
    omni = PersistentHttpConnection(
        args.omni_endpoint, "omniinfer-generation", timeout_seconds=timeout
    )
    omni_clear = PersistentHttpConnection(
        args.omni_clear_endpoint, "omniinfer-clear", timeout_seconds=timeout
    )
    connections: dict[str, Any] = {}
    progress["stage"] = "connect"
    try:
        connections["apxinf_single"] = apx.connect()
        connections["omniinfer_generation"] = omni.connect()
        connections["omniinfer_clear"] = omni_clear.connect()
        progress["stage"] = "preflight"
        apx_health = _control_request(apx, "GET", "/health", None, validate_apx_health)
        apx_state_start = _control_request(
            apx,
            "GET",
            "/apxinf/state",
            None,
            lambda value: validate_apx_state(value, expected_completed=0),
        )
        omni_health = _control_request(
            omni, "GET", args.omni_health_path, None, validate_omni_health
        )
        omni_state_start = _control_request(
            omni, "GET", args.omni_state_path, None, validate_omni_state
        )
        progress["preflight"] = {
            "apxinf_health": apx_health,
            "apxinf_state": apx_state_start,
            "omniinfer_health": omni_health,
            "omniinfer_state": omni_state_start,
        }
        sequence = {"B": 0, "G": 0}
        for entry in declared_schedule():
            progress["stage"] = (
                f"{entry['phase']}-block-{entry['block']}-pair-{entry['pair_index']}"
            )
            pair = {**entry, "samples": []}
            destination = (
                progress["warmup_pairs"]
                if entry["phase"] == "warmup"
                else progress["measured_pairs"]
            )
            destination.append(pair)
            for arm in entry["order"]:
                sequence[arm] += 1
                pair["samples"].append(
                    run_one_sample(
                        arm=arm,
                        sequence_index=sequence[arm],
                        schedule_entry=entry,
                        apx=apx,
                        omni=omni,
                        omni_clear=omni_clear,
                        args=args,
                    )
                )
        require(
            sequence
            == {"B": EXPECTED_REQUESTS_PER_ARM, "G": EXPECTED_REQUESTS_PER_ARM},
            "arm request count differs",
        )
        progress["stage"] = "postflight"
        apx_state_end = _control_request(
            apx,
            "GET",
            "/apxinf/state",
            None,
            lambda value: validate_apx_state(
                value, expected_completed=EXPECTED_REQUESTS_PER_ARM
            ),
        )
        omni_state_end = _control_request(
            omni, "GET", args.omni_state_path, None, validate_omni_state
        )
        require(
            omni_state_start["validated"] == omni_state_end["validated"],
            "OmniInfer resident backend identity changed",
        )
        require(
            apx.request_count == 139, "ApxInf single-connection request count differs"
        )
        require(omni.request_count == 71, "OmniInfer generation request count differs")
        require(omni_clear.request_count == 68, "OmniInfer clear request count differs")
        progress["postflight"] = {
            "apxinf_state": apx_state_end,
            "omniinfer_state": omni_state_end,
            "omniinfer_resident_identity_start_end_equal": True,
            "connection_request_counts": {
                "apxinf_single": apx.request_count,
                "omniinfer_generation": omni.request_count,
                "omniinfer_clear": omni_clear.request_count,
            },
        }
        progress["stage"] = "analysis"
        determinism = validate_per_runtime_determinism(
            progress["warmup_pairs"], progress["measured_pairs"]
        )
        statistics_receipt = analyze_measured_pairs(
            progress["measured_pairs"], quiet_host_passed=quiet_gate["passed"]
        )
        host_end = host_observation()
        progress["stage"] = "complete"
        return {
            "format": FORMAT,
            "schema_version": 1,
            "status": "COMPLETE_NON_FORMAL_DIAGNOSTIC",
            "ok": True,
            "qualification": QUALIFICATION,
            "formal_evidence_eligible": False,
            "formal_summary_allowed": False,
            "engine_winner_or_ranking_claim_allowed": False,
            "arm_mapping": {
                "B": "ApxInf resident HTTP named deployment",
                "G": "OmniInfer resident HTTP named deployment",
            },
            "claim_boundary": (
                "client-observed full HTTP response wall for two named deployments; "
                "not a same-engine gateway increment and not pure kernel time"
            ),
            "static_contract": static,
            "request_contract": {
                "canonical_json_object": REQUEST,
                "canonical_utf8": REQUEST_BYTES.decode("utf-8"),
                "size_bytes": len(REQUEST_BYTES),
                "sha256": sha256_bytes(REQUEST_BYTES),
                "same_body_for_B_and_G": True,
            },
            "eog_policy_contract": {
                "suppressed_eog_token_ids": SUPPRESSED_EOG_TOKEN_IDS,
                "same_five_token_negative_infinity_policy_required": True,
                "no_generated_eog_required": True,
                "cross_runtime_trajectory_equality_required": False,
            },
            "timing_contract": {
                "start": "immediately before one sendall of the complete serialized request wire",
                "end": "immediately after complete response body read",
                "json_parse_after_end": True,
                "semantic_validation_after_end": True,
                "cache_clear_outside_primary_interval": True,
                "request_authority_header_inside_primary_interval": True,
            },
            "schedule": {
                "warmup_orders": list(WARMUP_ORDERS),
                "odd_block_orders": list(ODD_BLOCK_ORDERS),
                "even_block_orders": list(EVEN_BLOCK_ORDERS),
                "measured_blocks": MEASURED_BLOCKS,
                "pairs_per_block": PAIRS_PER_BLOCK,
                "no_retry": True,
                "no_resample": True,
                "no_outlier_removal": True,
            },
            "endpoints": {
                "apxinf": args.apx_endpoint,
                "omniinfer_generation": args.omni_endpoint,
                "omniinfer_clear": args.omni_clear_endpoint,
                "omniinfer_health_path": args.omni_health_path,
                "omniinfer_state_path": args.omni_state_path,
                "omniinfer_clear_path": args.omni_clear_path,
                "omniinfer_clear_contract": args.omni_clear_contract,
                "omniinfer_clear_body": args.omni_clear_body,
            },
            "connections": connections,
            "clock": clock_receipt(),
            "quiet_host_gate": quiet_gate,
            "host_observation_start": host_start,
            "host_observation_end": host_end,
            "preflight": progress["preflight"],
            "warmup_pairs": progress["warmup_pairs"],
            "measured_pairs": progress["measured_pairs"],
            "per_runtime_determinism": determinism,
            "statistics": statistics_receipt,
            "postflight": progress["postflight"],
        }
    finally:
        apx.close()
        omni.close()
        omni_clear.close()


def run_self_test(_: argparse.Namespace) -> dict[str, Any]:
    static = validate_static_contract()
    require(
        parse_strict_json_document(b'{"ok":true}') == {"ok": True},
        "strict JSON fixture failed",
    )
    try:
        parse_strict_json_document(b'{"ok":true,"ok":false}')
    except AdmissionError:
        duplicate_rejected = True
    else:
        duplicate_rejected = False
    require(duplicate_rejected, "strict JSON accepted a duplicate key")
    return {
        "format": f"{FORMAT}-self-test",
        "ok": True,
        "zero_generation_requests": True,
        "static_contract": static,
        "strict_duplicate_key_rejected": True,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    self_test = commands.add_parser(
        "self-test", help="run zero-network contract checks"
    )
    self_test.set_defaults(operation=run_self_test)
    run = commands.add_parser("run", help="run one exclusive NON_FORMAL diagnostic")
    run.add_argument("--apx-endpoint", required=True, help="http://127.0.0.1:PORT")
    run.add_argument("--omni-endpoint", required=True, help="http://127.0.0.1:PORT")
    run.add_argument(
        "--omni-clear-endpoint", required=True, help="http://127.0.0.1:PORT"
    )
    run.add_argument("--omni-health-path", default="/health?deep=true")
    run.add_argument("--omni-state-path", default="/omni/state")
    run.add_argument("--omni-clear-path", required=True)
    run.add_argument(
        "--omni-clear-contract",
        required=True,
        choices=("omni-gateway", "llama-slot-erase"),
    )
    run.add_argument(
        "--omni-clear-body", required=True, choices=("empty-object", "empty")
    )
    run.add_argument("--output", required=True, help="absolute absent JSON path")
    run.add_argument("--timeout-seconds", type=float, default=600.0)
    run.add_argument(
        "--quiet-host-status",
        choices=("passed", "failed", "not-evaluated"),
        default="not-evaluated",
    )
    run.add_argument("--quiet-host-receipt")
    run.set_defaults(operation=None)
    return parser


def emit(value: dict[str, Any]) -> None:
    sys.stdout.buffer.write(canonical_json_bytes(value) + b"\n")
    sys.stdout.buffer.flush()


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "self-test":
        try:
            emit(args.operation(args))
            return 0
        except Exception as error:
            emit(
                {
                    "format": f"{FORMAT}-self-test-failure",
                    "ok": False,
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                }
            )
            return 1

    progress: dict[str, Any] = {
        "stage": "reserve-exclusive-output",
        "preflight": None,
        "warmup_pairs": [],
        "measured_pairs": [],
        "postflight": None,
    }
    output: ExclusiveJsonOutput | None = None
    try:
        output = ExclusiveJsonOutput(args.output)
        record = run_diagnostic(args, progress)
        output_receipt = output.write(record)
        emit(
            {
                "format": f"{FORMAT}-completion",
                "ok": True,
                "qualification": QUALIFICATION,
                "output": output_receipt,
            }
        )
        return 0
    except Exception as error:
        failure = {
            "format": f"{FORMAT}-failure",
            "schema_version": 1,
            "ok": False,
            "status": "TERMINAL_NON_FORMAL_DIAGNOSTIC_FAILURE",
            "qualification": QUALIFICATION,
            "formal_evidence_eligible": False,
            "engine_winner_or_ranking_claim_allowed": False,
            "stage": progress["stage"],
            "error_type": type(error).__name__,
            "error": str(error),
            "transport_observation": (
                error.observation if isinstance(error, TransportFailure) else None
            ),
            "progress": progress,
            "traceback": traceback.format_exc(),
            "no_retry_or_resample_attempted": True,
        }
        output_receipt = None
        if output is not None:
            try:
                output_receipt = output.write(failure)
            except Exception:
                output.close_empty()
        emit(
            {
                "format": f"{FORMAT}-failure-summary",
                "ok": False,
                "qualification": QUALIFICATION,
                "error_type": type(error).__name__,
                "error": str(error),
                "output": output_receipt,
            }
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
