"""Serial HTTP transport for the official DM05 ``POST /v1/infer`` contract."""

from __future__ import annotations

import copy
import json
import math
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Mapping

import numpy as np

__all__ = ["Dm05HttpService", "serve_dm05_http"]


class Dm05HttpService:
    """Transport shell over an ApxInf DM05 policy; no model imports here."""

    def __init__(self, policy, *, max_body_bytes: int = 32 * 1024 * 1024) -> None:
        self.policy = policy
        self.max_body_bytes = int(max_body_bytes)

    def health(self) -> dict[str, Any]:
        policy_metadata = dict(self.policy.metadata)
        snapshot = getattr(self.policy, "path_proof_snapshot", None)
        if callable(snapshot):
            path_proof = snapshot()
            if path_proof is not None:
                policy_metadata["path_proof"] = copy.deepcopy(path_proof)
        return {
            "status": "ok",
            "schema": "apxinf.dm05.libero.http.v1",
            "policy": policy_metadata,
        }

    def infer(self, body: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(body, Mapping) or set(body) - {"observation", "sampling"}:
            raise ValueError("request must contain only observation and optional sampling")
        observation = body.get("observation")
        if not isinstance(observation, Mapping):
            raise ValueError("observation must be an object")
        request = dict(observation)
        request["sampling"] = body.get("sampling", {})
        result = self.policy.infer(request)
        actions = np.asarray(result["actions"], dtype=np.float32)
        latency_ms = float(result["timing"]["total_ms"])
        if not math.isfinite(latency_ms) or latency_ms <= 0.0:
            raise RuntimeError("policy returned invalid total latency")
        metadata = {
            "latency_ms": latency_ms,
            "model_latency_ms": float(result["timing"]["model_ms"]),
            "schema": "apxinf.dm05.libero.response.v1",
        }
        policy_metadata = dict(getattr(self.policy, "metadata", {}))
        if policy_metadata.get("execution_backend") == "default_exact_combined":
            fields = [
                "execution_backend",
                "runtime_selector",
                "host_thread_policy",
                "torch_intraop_threads",
            ]
            for field in fields:
                if field not in policy_metadata:
                    raise RuntimeError(
                        f"DM05 optimized policy metadata omitted {field}"
                    )
                metadata[field] = copy.deepcopy(policy_metadata[field])
        path_proof = result.get("path_proof")
        if path_proof is not None:
            if not isinstance(path_proof, Mapping):
                raise RuntimeError("policy returned invalid path proof")
            path_proof = copy.deepcopy(dict(path_proof))
            metadata["path_proof"] = path_proof
            for field in (
                "precision",
                "llm_attention",
                "vision_attention",
                "action_attention",
            ):
                if field not in policy_metadata:
                    raise RuntimeError(f"DM05 policy metadata omitted {field}")
                metadata[field] = copy.deepcopy(policy_metadata[field])
        return {
            "actions": actions.tolist(),
            "metadata": metadata,
        }

    def handle(
        self,
        method: str,
        path: str,
        body: bytes = b"",
        *,
        content_type: str = "application/json",
    ) -> tuple[int, dict[str, Any]]:
        if method == "GET" and path in {"/health", "/healthz"}:
            return 200, self.health()
        if path != "/v1/infer":
            return 404, {"error": "not found"}
        if method != "POST":
            return 405, {"error": "method not allowed"}
        if content_type.split(";", 1)[0].strip().lower() != "application/json":
            return 415, {"error": "Content-Type must be application/json"}
        if len(body) > self.max_body_bytes:
            return 413, {"error": "request body too large"}
        try:
            document = json.loads(body.decode("utf-8"))
            return 200, self.infer(document)
        except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
            return 400, {"error": str(exc)}


def serve_dm05_http(policy, host: str, port: int) -> None:
    """Serve one request at a time, matching the fixed deployment contract."""

    service = Dm05HttpService(policy)

    class Handler(BaseHTTPRequestHandler):
        server_version = "ApxInf-DM05/1"

        def _dispatch(self) -> None:
            length_header = self.headers.get("Content-Length", "0")
            try:
                length = int(length_header)
            except ValueError:
                length = -1
            if length < 0 or length > service.max_body_bytes:
                status, response = 413, {"error": "request body too large"}
            else:
                body = self.rfile.read(length) if length else b""
                status, response = service.handle(
                    self.command,
                    self.path.split("?", 1)[0],
                    body,
                    content_type=self.headers.get("Content-Type", "application/json"),
                )
            payload = (json.dumps(response, separators=(",", ":")) + "\n").encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        do_GET = _dispatch
        do_POST = _dispatch

        def log_message(self, format: str, *args) -> None:
            print(f"[apxinf-dm05] {self.address_string()} {format % args}")

    server = HTTPServer((host, int(port)), Handler)
    try:
        server.serve_forever()
    finally:
        server.server_close()
