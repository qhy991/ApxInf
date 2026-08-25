#!/usr/bin/env python3
"""Fail-closed paired driver for OmniInfer gateway-path overhead.

The measured arms share one resident llama-server process.  Arm B addresses
the backend endpoint directly; arm G addresses the OmniInfer gateway.  The
driver writes no files except for the campaign command's exclusive one-shot
marker.  Every invocation emits exactly one compact JSON record on stdout.
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import math
import os
import pathlib
import statistics
import subprocess
import sys
import time
import traceback
import urllib.parse
from typing import Any


CAMPAIGN_ID = "qwen35-0.8b-omniinfer-gateway-overhead-v1-20260826"
REPOSITORY_ROOT = "/Users/haiyan-mini/Agent4Kernel/ApxInf"
CAMPAIGN_START_PATH = (
    "crates/apxinf-metal/evidence/llama-cpp/"
    "qwen35-0.8b-omniinfer-gateway-overhead-campaign-start-diagnostic-v1-20260826.json"
)
EXPECTED_DRIVER_PATH = (
    "/Users/haiyan-mini/Agent4Kernel/ApxInf/benchmarks/cross_runtime/"
    "omniinfer_gateway_overhead_driver.py"
)
EXPECTED_PYTHON_PATH = (
    "/opt/homebrew/Cellar/python@3.14/3.14.3/Frameworks/Python.framework/"
    "Versions/3.14/bin/python3.14"
)
EXPECTED_PYTHON_SIZE = 52448
EXPECTED_PYTHON_SHA256 = "ff73afba45e095e0dadc9b51deb9b994ef90d097c36a211955c57336ce76508f"
MODEL_PATH = (
    "/Users/haiyan-mini/Agent4Kernel/models/Qwen3.5-0.8B-2fc063647-GGUF/"
    "Qwen3.5-0.8B-2fc063647-Q8_0-llama.gguf"
)
MODEL_SIZE = 811843072
MODEL_SHA256 = "427555e797eefeb62e1c8ef71510ce062027b0c7f1674cc4cfcb352710e0908c"
OMNI_BASE_URL = "http://127.0.0.1:9000"
OMNI_CLI_PATH = "/tmp/apxinf-omniinfer-prep.F7GNFt/OmniInfer/omniinfer"
OMNI_CLI_SIZE = 9719136
OMNI_CLI_SHA256 = "65487424ca9179850b80079beafa5ad69a66e0841d328ee8dd8a1fd4b613d661"
OMNI_LLAMA_SERVER_PATH = (
    "/tmp/apxinf-omniinfer-prep.F7GNFt/runtime/llama.cpp-mac/bin/llama-server"
)
OMNI_LLAMA_SERVER_SIZE = 33472
OMNI_LLAMA_SERVER_SHA256 = "02723fc39fbeebd9849ce4c9ca3799649df3cf91f101c2cd56b8756e1db54d28"
OMNI_RUNTIME_LOGS_PATH = "/tmp/apxinf-omniinfer-prep.F7GNFt/runtime/llama.cpp-mac/logs"
OMNI_SLOT_SAVE_PATH = "/tmp/apxinf-omniinfer-prep.F7GNFt/slots-gateway-overhead-v1"
HISTORY_ROOT = "/tmp/apxinf-omniinfer-prep.F7GNFt/state/.local/request_history"
ONE_SHOT_MARKER = (
    "/tmp/apxinf-omniinfer-prep.F7GNFt/"
    "gateway-overhead-v1-campaign-consumed.marker"
)
PROMPT_TOKEN_IDS = [
    248045, 846, 198, 9419, 248046, 198, 248045,
    74455, 198, 248068, 271, 248069, 271,
]
RENDERED_PROMPT = (
    "<|im_start|>user\nHello<|im_end|>\n"
    "<|im_start|>assistant\n<think>\n\n</think>\n\n"
)
REQUEST: dict[str, Any] = {
    "cache_prompt": False,
    "chat_template_kwargs": {"enable_thinking": False},
    "id_slot": 0,
    "ignore_eos": True,
    "max_tokens": 128,
    "messages": [{"content": "Hello", "role": "user"}],
    "model": MODEL_PATH,
    "reasoning_format": "none",
    "return_tokens": True,
    "seed": 0,
    "stream": False,
    "temperature": 0,
    "verbose": True,
}
REQUEST_SIZE = 383
REQUEST_SHA256 = "7773f5337693843f1e8cf3017b98868517cbddd3bc32649e550d8f2fec1d5cf6"
EXPECTED_CONTENT_SHA256 = "c65d3fd8040fc05441d86adf0965a9f3c12bd801e965fae9cd2aa87d444df7ad"
# Frozen from the separate exploratory probe before the one-shot campaign.
EXPECTED_TOKEN_IDS_SHA256: str | None = (
    "0a8a6c5ceeb831528480ebcad172fbcdda4ac23478ab051b1f74a00ec6d4f8e4"
)
EXPECTED_GENERATION_SETTINGS_SHA256: str | None = (
    "99e6940bc3b693a57a15c128b6ac0c0fb713b1b169ebbd4c9fa41763582d32f3"
)
EXPECTED_VERBOSE_TOKENS_CACHED = 140
WARMUP_ORDERS = ["BG", "GB", "GB", "BG"]
ODD_BLOCK_ORDERS = ["BG", "GB", "GB", "BG"]
EVEN_BLOCK_ORDERS = ["GB", "BG", "BG", "GB"]
MEASURED_BLOCKS = 16
PAIRS_PER_BLOCK = 4
T_CRITICAL_DF15_975 = 2.131449545559323


class AdmissionError(RuntimeError):
    """A frozen-contract admission check failed."""


class CampaignInvalidError(AdmissionError):
    """A consumed campaign failed, with its captured raw record preserved."""

    def __init__(self, record: dict[str, Any]):
        super().__init__(str(record.get("error", "campaign invalid")))
        self.record = record


def compact_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")


REQUEST_BYTES = compact_json_bytes(REQUEST)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def emit(value: dict[str, Any]) -> None:
    sys.stdout.buffer.write(compact_json_bytes(value) + b"\n")
    sys.stdout.buffer.flush()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AdmissionError(message)


def positive_number(value: Any, label: str) -> float:
    require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        f"{label} is not numeric",
    )
    result = float(value)
    require(math.isfinite(result) and result > 0.0, f"{label} is not positive")
    return result


def relative_close(actual: float, expected: float, tolerance: float = 1e-5) -> bool:
    return abs(actual - expected) <= tolerance * max(abs(actual), abs(expected), 1.0)


def git_stdout(arguments: list[str]) -> str:
    process = subprocess.run(
        ["/usr/bin/git", *arguments],
        cwd=REPOSITORY_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30.0,
        check=False,
    )
    require(process.returncode == 0, f"git {' '.join(arguments)} failed")
    return process.stdout.decode("utf-8", "strict").strip()


def file_receipt(path_text: str, expected_size: int, expected_sha256: str) -> dict[str, Any]:
    path = pathlib.Path(path_text).resolve(strict=True)
    require(path.is_file() and not path.is_symlink(), f"not a direct regular file: {path}")
    stat = path.stat()
    require(stat.st_size == expected_size, f"size changed: {path}")
    digest = sha256_file(path)
    require(digest == expected_sha256, f"SHA256 changed: {path}")
    return {
        "path": str(path),
        "device": stat.st_dev,
        "inode": stat.st_ino,
        "size_bytes": stat.st_size,
        "ctime_ns": stat.st_ctime_ns,
        "sha256": digest,
    }


def execution_binding() -> dict[str, Any]:
    driver = pathlib.Path(sys.argv[0]).resolve(strict=True)
    python = pathlib.Path(sys.executable).resolve(strict=True)
    require(str(driver) == EXPECTED_DRIVER_PATH, "driver path is not the fixed absolute path")
    require(str(python) == EXPECTED_PYTHON_PATH, "Python path differs from the fixed interpreter")
    python_receipt = file_receipt(
        str(python), EXPECTED_PYTHON_SIZE, EXPECTED_PYTHON_SHA256
    )
    return {
        "argv": [str(driver), *sys.argv[1:]],
        "driver": {
            "path": str(driver),
            "size_bytes": driver.stat().st_size,
            "sha256": sha256_file(driver),
        },
        "python": {
            **python_receipt,
            "version": sys.version,
        },
    }


def validate_campaign_start_binding(args: argparse.Namespace) -> dict[str, Any]:
    require(
        len(args.campaign_start_sha256) == 64
        and all(c in "0123456789abcdef" for c in args.campaign_start_sha256),
        "campaign-start SHA256 is invalid",
    )
    require(
        len(args.campaign_start_commit) == 40
        and all(c in "0123456789abcdef" for c in args.campaign_start_commit),
        "campaign-start commit is invalid",
    )
    start = (pathlib.Path(REPOSITORY_ROOT) / CAMPAIGN_START_PATH).resolve(strict=True)
    require(start.is_file(), "campaign-start is absent")
    require(sha256_file(start) == args.campaign_start_sha256, "campaign-start SHA differs")
    head = git_stdout(["rev-parse", "HEAD"])
    origin = git_stdout(["rev-parse", "origin/main"])
    require(head == args.campaign_start_commit, "HEAD differs from campaign-start commit")
    require(origin == args.campaign_start_commit, "origin/main differs from campaign-start commit")
    require(
        git_stdout(["status", "--porcelain=v1", "--untracked-files=all"]) == "",
        "worktree is not clean",
    )
    return {
        "path": CAMPAIGN_START_PATH,
        "size_bytes": start.stat().st_size,
        "sha256": args.campaign_start_sha256,
        "commit": args.campaign_start_commit,
        "head": head,
        "origin_main": origin,
    }


def history_snapshot() -> dict[str, Any]:
    root = pathlib.Path(HISTORY_ROOT)
    entries: list[dict[str, Any]] = []
    if root.exists():
        for path in sorted(root.rglob("*"), key=lambda item: str(item.relative_to(root))):
            if path.is_file() and not path.is_symlink():
                stat = path.stat()
                entries.append({
                    "relative_path": str(path.relative_to(root)),
                    "size_bytes": stat.st_size,
                    "sha256": sha256_file(path),
                })
    return {
        "root": str(root),
        "entries": entries,
        "canonical_sha256": sha256_bytes(compact_json_bytes(entries)),
    }


def gateway_process_proof() -> dict[str, Any]:
    process = subprocess.run(
        ["/usr/sbin/lsof", "-nP", "-t", "-iTCP:9000", "-sTCP:LISTEN"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=10.0,
        check=False,
    )
    require(process.returncode == 0, "port 9000 has no listening gateway")
    pids = [int(line) for line in process.stdout.splitlines() if line.strip()]
    require(len(pids) == 1, "port 9000 listener PID is not unique")
    pid = pids[0]
    ps = subprocess.run(
        ["/bin/ps", "eww", "-p", str(pid), "-o", "command="],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=10.0,
        check=False,
    )
    require(ps.returncode == 0, "cannot inspect gateway process environment")
    command_and_env = ps.stdout.decode("utf-8", "replace").strip()
    require("OMNIINFER_REQUEST_HISTORY=0" in command_and_env, "request history is not disabled")
    require(OMNI_CLI_PATH in command_and_env, "listener is not the pinned OmniInfer binary")
    return {
        "pid": pid,
        "command_and_environment_sha256": sha256_bytes(ps.stdout),
        "request_history_environment": "OMNIINFER_REQUEST_HISTORY=0",
        "pinned_binary_present": True,
    }


def backend_process_proof(
    state: dict[str, Any], gateway_pid: int
) -> dict[str, Any]:
    backend_pid = state.get("backend_pid")
    backend_port = state.get("backend_port")
    launch_command = state.get("launch_command")
    require(isinstance(backend_pid, int), "backend process proof lacks PID")
    require(isinstance(backend_port, int), "backend process proof lacks port")
    require(
        isinstance(launch_command, list)
        and all(isinstance(item, str) for item in launch_command),
        "backend process proof lacks launch command",
    )

    listener = subprocess.run(
        [
            "/usr/sbin/lsof", "-nP", "-t", f"-iTCP:{backend_port}",
            "-sTCP:LISTEN",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=10.0,
        check=False,
    )
    require(listener.returncode == 0, "backend port has no listener")
    listener_pids = [
        int(line) for line in listener.stdout.splitlines() if line.strip()
    ]
    require(listener_pids == [backend_pid], "backend listener PID differs from state")

    parent = subprocess.run(
        ["/bin/ps", "-p", str(backend_pid), "-o", "ppid="],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=10.0,
        check=False,
    )
    require(parent.returncode == 0, "cannot inspect backend parent PID")
    try:
        parent_pid = int(parent.stdout.strip())
    except ValueError as error:
        raise AdmissionError("backend parent PID is invalid") from error
    require(parent_pid == gateway_pid, "backend is not a child of the gateway")

    command = subprocess.run(
        ["/bin/ps", "-p", str(backend_pid), "-o", "command="],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=10.0,
        check=False,
    )
    require(command.returncode == 0, "cannot inspect backend command")
    command_text = command.stdout.decode("utf-8", "strict").strip()
    require(
        command_text == " ".join(launch_command),
        "backend OS command differs from the pinned launch command",
    )

    executable = subprocess.run(
        ["/usr/sbin/lsof", "-a", "-p", str(backend_pid), "-d", "txt", "-Fn"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=10.0,
        check=False,
    )
    require(executable.returncode == 0, "cannot inspect backend executable mapping")
    executable_lines = executable.stdout.decode("utf-8", "strict").splitlines()
    resolved_executable = str(pathlib.Path(OMNI_LLAMA_SERVER_PATH).resolve(strict=True))
    require(
        f"n{resolved_executable}" in executable_lines,
        "backend PID is not mapped to the pinned llama-server executable",
    )
    return {
        "pid": backend_pid,
        "port": backend_port,
        "unique_listener_pid": backend_pid,
        "parent_gateway_pid": parent_pid,
        "command": command_text,
        "command_sha256": sha256_bytes(command.stdout),
        "pinned_executable_path": OMNI_LLAMA_SERVER_PATH,
        "resolved_executable_mapping": resolved_executable,
        "pinned_executable_mapped": True,
    }


def expected_launch_args() -> list[str]:
    return [
        "--slot-prompt-similarity", "0", "--cache-idle-slots",
        "-ngl", "999", "--cache-ram", "8192",
        "-ngl", "999", "-b", "13", "-ub", "13",
        "-t", "4", "-tb", "4", "-np", "1",
        "--cache-ram", "0", "--slots", "--no-cache-idle-slots",
        "--no-cache-prompt", "--slot-prompt-similarity", "0",
        "--slot-save-path", OMNI_SLOT_SAVE_PATH,
    ]


def essential_state(state: dict[str, Any]) -> dict[str, Any]:
    runtime = state.get("runtime") if isinstance(state.get("runtime"), dict) else {}
    return {
        "backend": state.get("backend"),
        "backend_ready": state.get("backend_ready"),
        "model_path": state.get("model_path"),
        "ctx_size": state.get("ctx_size"),
        "backend_pid": state.get("backend_pid"),
        "backend_port": state.get("backend_port"),
        "client_endpoint": state.get("client_endpoint"),
        "generation": state.get("generation"),
        "launch_command": state.get("launch_command"),
        "launch_args": state.get("launch_args"),
        "request_defaults": state.get("request_defaults"),
        "effective_parameters": state.get("effective_parameters"),
        "proxy_model": state.get("proxy_model"),
        "runtime": {
            "mode": runtime.get("mode"),
            "pid": runtime.get("pid"),
            "port": runtime.get("port"),
            "client_endpoint": runtime.get("client_endpoint"),
            "launch_command": runtime.get("launch_command"),
        },
    }


def validate_state(state: dict[str, Any]) -> dict[str, Any]:
    require(state.get("backend") == "llama.cpp-mac", "backend is not llama.cpp-mac")
    require(state.get("backend_ready") is True, "backend is not ready")
    require(state.get("model_path") == MODEL_PATH, "loaded model path differs")
    require(state.get("default_model") == MODEL_PATH, "default model differs")
    require(state.get("mmproj") is None, "mmproj is not null")
    require(state.get("ctx_size") == 256, "context is not 256")
    require(state.get("request_defaults") == {}, "request defaults are not empty")
    require(state.get("effective_parameters") == {}, "effective parameters are not empty")
    require(state.get("proxy_model") is None, "proxy model is not null")
    require(state.get("public_model_id") is None, "public model id is not null")
    require(isinstance(state.get("backend_pid"), int), "backend PID is absent")
    require(isinstance(state.get("backend_port"), int), "backend port is absent")
    endpoint = f"http://127.0.0.1:{state['backend_port']}"
    require(state.get("client_endpoint") == endpoint, "backend endpoint differs")
    args = expected_launch_args()
    require(state.get("launch_args") == args, "backend launch args differ")
    command = [
        OMNI_LLAMA_SERVER_PATH, "-m", MODEL_PATH,
        "--host", "127.0.0.1", "--port", str(state["backend_port"]),
        "--no-webui", "--slot-save-path", OMNI_RUNTIME_LOGS_PATH,
        *args, "-c", "256",
    ]
    require(state.get("launch_command") == command, "backend launch command differs")
    runtime = state.get("runtime")
    require(isinstance(runtime, dict), "runtime object is absent")
    require(runtime.get("pid") == state["backend_pid"], "runtime PID differs")
    require(runtime.get("port") == state["backend_port"], "runtime port differs")
    require(runtime.get("client_endpoint") == endpoint, "runtime endpoint differs")
    require(runtime.get("launch_command") == command, "runtime launch command differs")
    loaded = state.get("loaded_models")
    require(isinstance(loaded, list) and len(loaded) == 1, "loaded-model set differs")
    require(loaded[0].get("model_path") == MODEL_PATH, "loaded-model path differs")
    require(loaded[0].get("request_defaults") == {}, "loaded defaults differ")
    restore = state.get("restore_selection")
    require(isinstance(restore, dict), "restore selection is absent")
    require(restore.get("model") == MODEL_PATH, "restore model differs")
    require(restore.get("ctx_size") == 256, "restore context differs")
    require(restore.get("request_defaults") == {}, "restore defaults differ")
    return essential_state(state)


def validate_props(props: dict[str, Any]) -> None:
    require(props.get("build_info") == "b10280-61881b1f7", "backend build differs")
    require(props.get("model_path") == MODEL_PATH, "props model path differs")
    require(props.get("model_ftype") == "Q8_0", "props model type differs")
    require(props.get("endpoint_slots") is True, "slots endpoint is disabled")
    require(props.get("total_slots") == 1, "slot count differs")
    settings = props.get("default_generation_settings")
    require(isinstance(settings, dict) and settings.get("n_ctx") == 256, "props context differs")


class PersistentJsonConnection:
    """One preconnected HTTP/1.1 connection with zero-reconnect custody."""

    def __init__(self, base_url: str, label: str, timeout_seconds: float = 600.0):
        parsed = urllib.parse.urlsplit(base_url)
        require(parsed.scheme == "http", f"{label} URL is not HTTP")
        require(parsed.hostname == "127.0.0.1", f"{label} host is not loopback")
        require(parsed.port is not None, f"{label} port is absent")
        self.host = parsed.hostname
        self.port = parsed.port
        self.label = label
        self.connection = http.client.HTTPConnection(self.host, self.port, timeout=timeout_seconds)
        self.baseline: dict[str, Any] | None = None
        self.request_count = 0

    def socket_identity(self) -> dict[str, Any]:
        sock = self.connection.sock
        require(sock is not None, f"{self.label} socket is closed")
        return {
            "object_id": id(sock),
            "fileno": sock.fileno(),
            "local": list(sock.getsockname()),
            "peer": list(sock.getpeername()),
        }

    def connect(self) -> dict[str, Any]:
        self.connection.connect()
        self.baseline = self.socket_identity()
        return dict(self.baseline)

    def request_json(
        self, method: str, path: str, body: bytes | None, measured: bool
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        require(self.baseline is not None, f"{self.label} connection was not preconnected")
        before = self.socket_identity()
        require(before == self.baseline, f"{self.label} connection changed before request")
        started_ns = time.monotonic_ns()
        self.connection.putrequest(
            method, path, skip_host=True, skip_accept_encoding=True
        )
        self.connection.putheader("Host", f"{self.host}:{self.port}")
        if body is not None:
            self.connection.putheader("Content-Type", "application/json")
            self.connection.putheader("Content-Length", str(len(body)))
        self.connection.endheaders(body)
        response = self.connection.getresponse()
        raw = response.read()
        ended_ns = time.monotonic_ns()
        parse_started_ns = time.monotonic_ns()
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as error:
            raise AdmissionError(f"{self.label} returned non-JSON") from error
        parse_ended_ns = time.monotonic_ns()
        require(response.status == 200, f"{self.label} returned HTTP {response.status}")
        require(isinstance(payload, dict), f"{self.label} response is not an object")
        after = self.socket_identity()
        require(after == self.baseline, f"{self.label} connection reconnected or closed")
        self.request_count += 1
        headers = {name.lower(): value for name, value in response.getheaders()}
        receipt = {
            "connection": self.label,
            "connection_generation": 1,
            "request_index_on_connection": self.request_count,
            "socket": dict(self.baseline),
            "method": method,
            "path": path,
            "measured_wall_boundary": measured,
            "started_monotonic_ns": started_ns,
            "ended_monotonic_ns": ended_ns,
            "wall_ns": ended_ns - started_ns,
            "wall_ms": (ended_ns - started_ns) / 1e6,
            "client_json_parse_excluded_from_wall": True,
            "client_json_parse_ns": parse_ended_ns - parse_started_ns,
            "client_json_parse_ms": (parse_ended_ns - parse_started_ns) / 1e6,
            "status": response.status,
            "http_version": response.version,
            "response_content_type": headers.get("content-type"),
            "response_size_bytes": len(raw),
            "response_sha256": sha256_bytes(raw),
        }
        return payload, receipt

    def close(self) -> None:
        self.connection.close()


def validate_slot_directory() -> None:
    slot = pathlib.Path(OMNI_SLOT_SAVE_PATH)
    require(slot.is_absolute(), "slot directory is not absolute")
    require(slot.resolve(strict=True).is_dir(), "slot directory does not exist")


def artifact_receipts() -> dict[str, Any]:
    return {
        "omniinfer_cli": file_receipt(OMNI_CLI_PATH, OMNI_CLI_SIZE, OMNI_CLI_SHA256),
        "llama_server": file_receipt(
            OMNI_LLAMA_SERVER_PATH,
            OMNI_LLAMA_SERVER_SIZE,
            OMNI_LLAMA_SERVER_SHA256,
        ),
        "model": file_receipt(MODEL_PATH, MODEL_SIZE, MODEL_SHA256),
    }


def one_shot_json(
    base_url: str, label: str, method: str, path: str, body: bytes | None
) -> tuple[dict[str, Any], dict[str, Any]]:
    connection = PersistentJsonConnection(base_url, label, timeout_seconds=60.0)
    connection.connect()
    try:
        return connection.request_json(method, path, body, measured=False)
    finally:
        connection.close()


def clear_cache(admin: PersistentJsonConnection) -> dict[str, Any]:
    payload, transport = admin.request_json("POST", "/omni/cache/clear", b"{}", measured=False)
    require(payload.get("ok") is True, "cache clear was not acknowledged")
    require(payload.get("cache_policy") == "cleared_each_run", "cache policy differs")
    require(payload.get("cleared_slots") == [0], "cache clear did not erase exact slot 0")
    return {"response": payload, "transport": transport}


def tokenize_ids(payload: dict[str, Any]) -> list[int]:
    raw = payload.get("tokens")
    require(isinstance(raw, list), "tokenize response lacks tokens")
    result: list[int] = []
    for entry in raw:
        value = entry.get("id") if isinstance(entry, dict) else entry
        require(isinstance(value, int) and not isinstance(value, bool), "token ID is invalid")
        result.append(value)
    return result


def validate_response(response: dict[str, Any], frozen: bool) -> dict[str, Any]:
    require(response.get("object") == "chat.completion", "response object differs")
    require(response.get("model") == MODEL_PATH, "response model differs")
    require(response.get("system_fingerprint") == "b10280-61881b1f7", "fingerprint differs")
    choices = response.get("choices")
    require(isinstance(choices, list) and len(choices) == 1, "choice count differs")
    choice = choices[0]
    require(isinstance(choice, dict), "choice is not an object")
    require(choice.get("finish_reason") == "length", "finish reason is not length")
    message = choice.get("message")
    require(isinstance(message, dict), "message is absent")
    require(message.get("role") == "assistant", "message role differs")
    content = message.get("content")
    require(isinstance(content, str), "message content is not a string")
    content_bytes = content.encode("utf-8")
    content_sha256 = sha256_bytes(content_bytes)
    require(content_sha256 == EXPECTED_CONTENT_SHA256, "response content hash differs")

    usage = response.get("usage")
    require(isinstance(usage, dict), "usage is absent")
    require(usage.get("prompt_tokens") == 13, "usage prompt count differs")
    require(usage.get("completion_tokens") == 128, "usage completion count differs")
    require(usage.get("total_tokens") == 141, "usage total count differs")
    prompt_details = usage.get("prompt_tokens_details")
    require(
        isinstance(prompt_details, dict) and prompt_details.get("cached_tokens") == 0,
        "usage cached token count differs",
    )

    timings = response.get("timings")
    require(isinstance(timings, dict), "timings are absent")
    require(timings.get("prompt_n") == 13, "native prompt count differs")
    require(timings.get("predicted_n") == 128, "native predicted count differs")
    require(timings.get("cache_n") == 0, "native cache count differs")
    prompt_ms = positive_number(timings.get("prompt_ms"), "prompt_ms")
    predicted_ms = positive_number(timings.get("predicted_ms"), "predicted_ms")
    prompt_tps = positive_number(timings.get("prompt_per_second"), "prompt_per_second")
    predicted_tps = positive_number(
        timings.get("predicted_per_second"), "predicted_per_second"
    )
    require(relative_close(prompt_tps, 13000.0 / prompt_ms), "prompt TPS formula differs")
    require(
        relative_close(predicted_tps, 128000.0 / predicted_ms),
        "predicted TPS formula differs",
    )
    require(
        relative_close(
            positive_number(timings.get("prompt_per_token_ms"), "prompt_per_token_ms"),
            prompt_ms / 13.0,
        ),
        "prompt per-token formula differs",
    )
    require(
        relative_close(
            positive_number(
                timings.get("predicted_per_token_ms"), "predicted_per_token_ms"
            ),
            predicted_ms / 128.0,
        ),
        "predicted per-token formula differs",
    )

    verbose = response.get("__verbose")
    require(isinstance(verbose, dict), "__verbose response is absent")
    require(verbose.get("id_slot") == 0, "verbose slot differs")
    require(verbose.get("tokens_predicted") == 128, "verbose predicted count differs")
    require(verbose.get("tokens_evaluated") == 13, "verbose evaluated count differs")
    verbose_tokens_cached = verbose.get("tokens_cached")
    require(
        isinstance(verbose_tokens_cached, int)
        and not isinstance(verbose_tokens_cached, bool)
        and verbose_tokens_cached >= 0,
        "verbose cached count is invalid",
    )
    require(verbose.get("stop_type") == "limit", "verbose stop type differs")
    require(verbose.get("truncated") is False, "verbose response was truncated")
    prompt = verbose.get("prompt")
    require(isinstance(prompt, str) and prompt == RENDERED_PROMPT, "verbose prompt differs")
    tokens = verbose.get("tokens")
    require(
        isinstance(tokens, list)
        and len(tokens) == 128
        and all(isinstance(token, int) and not isinstance(token, bool) for token in tokens),
        "verbose generated token IDs differ in shape",
    )
    token_sha256 = sha256_bytes(compact_json_bytes(tokens))
    settings = verbose.get("generation_settings")
    require(isinstance(settings, dict), "verbose generation settings are absent")
    settings_sha256 = sha256_bytes(compact_json_bytes(settings))
    if frozen:
        require(EXPECTED_TOKEN_IDS_SHA256 is not None, "expected token hash was not frozen")
        require(
            EXPECTED_GENERATION_SETTINGS_SHA256 is not None,
            "expected generation-settings hash was not frozen",
        )
        require(token_sha256 == EXPECTED_TOKEN_IDS_SHA256, "generated trajectory hash differs")
        require(
            settings_sha256 == EXPECTED_GENERATION_SETTINGS_SHA256,
            "generation-settings hash differs",
        )
        require(
            verbose_tokens_cached == EXPECTED_VERBOSE_TOKENS_CACHED,
            "verbose cached count differs from the exploratory freeze",
        )
    return {
        "content": content,
        "content_size_bytes": len(content_bytes),
        "content_sha256": content_sha256,
        "token_ids": tokens,
        "token_ids_compact_json_sha256": token_sha256,
        "rendered_prompt": prompt,
        "rendered_prompt_sha256": sha256_bytes(prompt.encode("utf-8")),
        "generation_settings_sha256": settings_sha256,
        "verbose_tokens_cached": verbose_tokens_cached,
        "native": {
            "prompt_ms": prompt_ms,
            "predicted_ms": predicted_ms,
            "prompt_tps": prompt_tps,
            "predicted_tps": predicted_tps,
            "total_ms": prompt_ms + predicted_ms,
        },
    }


def validate_pair_equal(pair: dict[str, Any]) -> None:
    samples = pair.get("samples")
    require(isinstance(samples, list) and len(samples) == 2, "pair sample count differs")
    by_arm = {sample["arm"]: sample for sample in samples}
    require(set(by_arm) == {"B", "G"}, "pair arms differ")
    left = by_arm["B"]["validated"]
    right = by_arm["G"]["validated"]
    for key in [
        "content",
        "content_sha256",
        "token_ids",
        "token_ids_compact_json_sha256",
        "rendered_prompt",
        "rendered_prompt_sha256",
        "generation_settings_sha256",
        "verbose_tokens_cached",
    ]:
        require(left[key] == right[key], f"pair B/G {key} differs")


def open_runtime_connections(gateway_pid: int) -> tuple[
    PersistentJsonConnection,
    PersistentJsonConnection,
    PersistentJsonConnection,
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    admin = PersistentJsonConnection(OMNI_BASE_URL, "gateway-admin")
    admin_socket = admin.connect()
    state, _ = admin.request_json("GET", "/omni/state", None, measured=False)
    props, _ = admin.request_json("GET", "/omni/backend/props", None, measured=False)
    state_receipt = validate_state(state)
    validate_props(props)
    backend_process = backend_process_proof(state_receipt, gateway_pid)
    backend = PersistentJsonConnection(state["client_endpoint"], "backend-measurement")
    gateway = PersistentJsonConnection(OMNI_BASE_URL, "gateway-measurement")
    backend_socket = backend.connect()
    gateway_socket = gateway.connect()
    connections = {
        "admin": admin_socket,
        "backend_measurement": backend_socket,
        "gateway_measurement": gateway_socket,
    }
    return admin, backend, gateway, state_receipt, connections, backend_process


def run_sample(
    arm: str,
    sequence_index: int,
    block: int,
    pair_index: int,
    order: str,
    measured: bool,
    admin: PersistentJsonConnection,
    backend: PersistentJsonConnection,
    gateway: PersistentJsonConnection,
    frozen: bool,
) -> dict[str, Any]:
    clear = clear_cache(admin)
    target = backend if arm == "B" else gateway
    response, transport = target.request_json(
        "POST", "/v1/chat/completions", REQUEST_BYTES, measured=measured
    )
    validated = validate_response(response, frozen=frozen)
    return {
        "sequence_index": sequence_index,
        "block": block,
        "pair_index": pair_index,
        "order": order,
        "arm": arm,
        "endpoint": (
            f"backend:{target.host}:{target.port}" if arm == "B" else OMNI_BASE_URL
        ),
        "measured": measured,
        "request_size_bytes": len(REQUEST_BYTES),
        "request_sha256": sha256_bytes(REQUEST_BYTES),
        "request": REQUEST,
        "cache_clear_immediately_before": clear,
        "transport": transport,
        "validated": validated,
        "response": response,
    }


def summary_stats(values: list[float]) -> dict[str, Any]:
    require(len(values) > 1, "statistics require multiple observations")
    mean = statistics.fmean(values)
    return {
        "samples": values,
        "count": len(values),
        "mean": mean,
        "median": statistics.median(values),
        "population_sd": statistics.pstdev(values),
        "population_cv": statistics.pstdev(values) / mean,
        "min": min(values),
        "max": max(values),
    }


def block_t_interval(block_means: list[float]) -> dict[str, Any]:
    require(len(block_means) == MEASURED_BLOCKS, "block mean count differs")
    mean = statistics.fmean(block_means)
    standard_error = statistics.stdev(block_means) / math.sqrt(len(block_means))
    half_width = T_CRITICAL_DF15_975 * standard_error
    return {
        "block_means": block_means,
        "mean": mean,
        "sample_sd": statistics.stdev(block_means),
        "standard_error": standard_error,
        "t_critical": T_CRITICAL_DF15_975,
        "degrees_of_freedom": 15,
        "ci95_lower": mean - half_width,
        "ci95_upper": mean + half_width,
        "ci95_half_width": half_width,
    }


def analyze_pairs(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    require(len(pairs) == 64, "measured pair count differs")
    backend_walls: list[float] = []
    gateway_walls: list[float] = []
    backend_predicted: list[float] = []
    gateway_predicted: list[float] = []
    deltas: list[float] = []
    log_ratios: list[float] = []
    adjusted_deltas: list[float] = []
    strata: dict[str, list[float]] = {"BG": [], "GB": []}
    by_block_delta: list[list[float]] = [[] for _ in range(MEASURED_BLOCKS)]
    by_block_log: list[list[float]] = [[] for _ in range(MEASURED_BLOCKS)]
    by_block_adjusted: list[list[float]] = [[] for _ in range(MEASURED_BLOCKS)]
    for pair in pairs:
        validate_pair_equal(pair)
        by_arm = {sample["arm"]: sample for sample in pair["samples"]}
        backend = by_arm["B"]
        gateway = by_arm["G"]
        backend_wall = float(backend["transport"]["wall_ms"])
        gateway_wall = float(gateway["transport"]["wall_ms"])
        backend_native = float(backend["validated"]["native"]["total_ms"])
        gateway_native = float(gateway["validated"]["native"]["total_ms"])
        delta = gateway_wall - backend_wall
        log_ratio = math.log(gateway_wall / backend_wall)
        adjusted = (gateway_wall - gateway_native) - (backend_wall - backend_native)
        block_index = int(pair["block"]) - 1
        backend_walls.append(backend_wall)
        gateway_walls.append(gateway_wall)
        backend_predicted.append(float(backend["validated"]["native"]["predicted_ms"]))
        gateway_predicted.append(float(gateway["validated"]["native"]["predicted_ms"]))
        deltas.append(delta)
        log_ratios.append(log_ratio)
        adjusted_deltas.append(adjusted)
        strata[pair["order"]].append(delta)
        by_block_delta[block_index].append(delta)
        by_block_log[block_index].append(log_ratio)
        by_block_adjusted[block_index].append(adjusted)
        pair["derived"] = {
            "gateway_minus_backend_wall_ms": delta,
            "gateway_over_backend_wall_ratio": gateway_wall / backend_wall,
            "log_wall_ratio": log_ratio,
            "native_adjusted_residual_delta_ms": adjusted,
        }
    require(all(len(block) == 4 for block in by_block_delta), "pairs per block differ")
    block_delta = [statistics.fmean(block) for block in by_block_delta]
    block_log = [statistics.fmean(block) for block in by_block_log]
    block_adjusted = [statistics.fmean(block) for block in by_block_adjusted]
    primary = block_t_interval(block_delta)
    log_interval = block_t_interval(block_log)
    adjusted_interval = block_t_interval(block_adjusted)
    ratio = {
        **log_interval,
        "geometric_mean_ratio": math.exp(log_interval["mean"]),
        "ci95_ratio_lower": math.exp(log_interval["ci95_lower"]),
        "ci95_ratio_upper": math.exp(log_interval["ci95_upper"]),
    }
    backend_stats = summary_stats(backend_walls)
    gateway_stats = summary_stats(gateway_walls)
    backend_predicted_stats = summary_stats(backend_predicted)
    gateway_predicted_stats = summary_stats(gateway_predicted)
    pooled_wall = statistics.fmean(backend_walls + gateway_walls)
    order_means = {order: statistics.fmean(values) for order, values in strata.items()}
    order_difference = abs(order_means["BG"] - order_means["GB"])
    drift = abs(statistics.fmean(block_delta[:8]) - statistics.fmean(block_delta[8:]))
    drift_threshold = max(2.0, 0.002 * pooled_wall)
    gates = {
        "backend_wall_population_cv_le_1pct": backend_stats["population_cv"] <= 0.01,
        "gateway_wall_population_cv_le_1pct": gateway_stats["population_cv"] <= 0.01,
        "backend_predicted_ms_population_cv_le_1pct": (
            backend_predicted_stats["population_cv"] <= 0.01
        ),
        "gateway_predicted_ms_population_cv_le_1pct": (
            gateway_predicted_stats["population_cv"] <= 0.01
        ),
        "delta_sd_over_pooled_wall_le_1pct": statistics.pstdev(deltas) / pooled_wall <= 0.01,
        "order_stratum_mean_difference_within_threshold": order_difference <= drift_threshold,
        "front_back_block_mean_difference_within_threshold": drift <= drift_threshold,
        "primary_ci95_half_width_le_2ms": primary["ci95_half_width"] <= 2.0,
    }
    gates_passed = all(gates.values())
    positive_overhead = (
        gates_passed
        and primary["ci95_lower"] > 0.0
        and ratio["ci95_ratio_lower"] > 1.0
        and order_means["BG"] > 0.0
        and order_means["GB"] > 0.0
        and adjusted_interval["ci95_lower"] > 0.0
    )
    practical_equivalence = (
        gates_passed
        and primary["ci95_lower"] >= -5.0
        and primary["ci95_upper"] <= 5.0
        and ratio["ci95_ratio_lower"] >= 0.995
        and ratio["ci95_ratio_upper"] <= 1.005
    )
    return {
        "primary_gateway_minus_backend_wall_ms": primary,
        "secondary_geometric_wall_ratio": ratio,
        "native_adjusted_residual_sensitivity_ms": adjusted_interval,
        "backend_wall_ms": backend_stats,
        "gateway_wall_ms": gateway_stats,
        "backend_native_predicted_ms": backend_predicted_stats,
        "gateway_native_predicted_ms": gateway_predicted_stats,
        "raw_pair_deltas_ms": deltas,
        "raw_pair_ratios": [math.exp(value) for value in log_ratios],
        "order_strata": {"samples": strata, "means": order_means},
        "order_stratum_mean_difference_ms": order_difference,
        "front_back_block_mean_difference_ms": drift,
        "order_and_drift_threshold_ms": drift_threshold,
        "stability_gates": gates,
        "stability_gates_passed": gates_passed,
        "positive_gateway_path_overhead_detected": positive_overhead,
        "practical_equivalence_within_5ms_and_0.5pct": practical_equivalence,
        "claim_boundary": (
            "client-observed full-response OmniInfer gateway-path overhead under "
            "warmed persistent HTTP/1.1 connections; not pure Rust/JSON CPU time"
        ),
    }


def run_self_test(_: argparse.Namespace) -> dict[str, Any]:
    binding = execution_binding()
    require(len(REQUEST_BYTES) == REQUEST_SIZE, "canonical request size differs")
    require(sha256_bytes(REQUEST_BYTES) == REQUEST_SHA256, "canonical request SHA differs")
    require(json.loads(REQUEST_BYTES) == REQUEST, "canonical request does not round-trip")
    require(
        sum(len(ODD_BLOCK_ORDERS if block % 2 else EVEN_BLOCK_ORDERS)
            for block in range(1, MEASURED_BLOCKS + 1)) == 64,
        "measured schedule pair count differs",
    )
    flat_orders = [
        order
        for block in range(1, MEASURED_BLOCKS + 1)
        for order in (ODD_BLOCK_ORDERS if block % 2 else EVEN_BLOCK_ORDERS)
    ]
    require(flat_orders.count("BG") == 32 and flat_orders.count("GB") == 32, "order balance differs")
    fixture_blocks = [float(index) / 100.0 for index in range(1, 17)]
    interval = block_t_interval(fixture_blocks)
    require(interval["degrees_of_freedom"] == 15, "fixture CI degrees of freedom differ")
    return {
        "format": "apxinf-omniinfer-gateway-overhead-driver-self-test-v1",
        "ok": True,
        "campaign_id": CAMPAIGN_ID,
        "generation_requests": 0,
        "execution_binding": binding,
        "request": {
            "value": REQUEST,
            "size_bytes": len(REQUEST_BYTES),
            "sha256": sha256_bytes(REQUEST_BYTES),
            "utf8": REQUEST_BYTES.decode("utf-8"),
        },
        "schedule": {
            "warmup_orders": WARMUP_ORDERS,
            "measured_blocks": MEASURED_BLOCKS,
            "pairs_per_block": PAIRS_PER_BLOCK,
            "measured_pairs": len(flat_orders),
            "BG_pairs": flat_orders.count("BG"),
            "GB_pairs": flat_orders.count("GB"),
        },
        "fixture_block_interval": interval,
        "probe_frozen_hashes": {
            "token_ids_sha256": EXPECTED_TOKEN_IDS_SHA256,
            "generation_settings_sha256": EXPECTED_GENERATION_SETTINGS_SHA256,
        },
    }


def run_preflight(_: argparse.Namespace) -> dict[str, Any]:
    binding = execution_binding()
    validate_slot_directory()
    require(not pathlib.Path(ONE_SHOT_MARKER).exists(), "campaign marker already exists")
    process = gateway_process_proof()
    artifacts = artifact_receipts()
    history_before = history_snapshot()
    health, health_transport = one_shot_json(
        OMNI_BASE_URL, "preflight-health", "GET", "/health?deep=true", None
    )
    state, state_transport = one_shot_json(
        OMNI_BASE_URL, "preflight-state", "GET", "/omni/state", None
    )
    props, props_transport = one_shot_json(
        OMNI_BASE_URL, "preflight-props", "GET", "/omni/backend/props", None
    )
    state_receipt = validate_state(state)
    validate_props(props)
    backend_process = backend_process_proof(state_receipt, process["pid"])
    require(health.get("status") == "ok", "gateway health is not ok")
    require(
        isinstance(health.get("backend_health"), dict)
        and health["backend_health"].get("status") == "ok",
        "backend deep health is not ok",
    )
    apply_request = compact_json_bytes({
        "add_generation_prompt": True,
        "chat_template_kwargs": {"enable_thinking": False},
        "messages": [{"content": "Hello", "role": "user"}],
    })
    applied, apply_transport = one_shot_json(
        state["client_endpoint"], "preflight-apply-template", "POST", "/apply-template", apply_request
    )
    require(applied.get("prompt") == RENDERED_PROMPT, "rendered prompt differs")
    tokenized, tokenize_transport = one_shot_json(
        OMNI_BASE_URL,
        "preflight-tokenize",
        "POST",
        "/tokenize",
        compact_json_bytes({
            "add_special": False,
            "content": RENDERED_PROMPT,
            "with_pieces": True,
        }),
    )
    ids = tokenize_ids(tokenized)
    require(ids == PROMPT_TOKEN_IDS, "rendered prompt token IDs differ")
    admin = PersistentJsonConnection(OMNI_BASE_URL, "preflight-admin", timeout_seconds=60.0)
    admin.connect()
    try:
        cache_clear = clear_cache(admin)
        state_after, state_after_transport = admin.request_json(
            "GET", "/omni/state", None, measured=False
        )
    finally:
        admin.close()
    require(validate_state(state_after) == state_receipt, "state changed during preflight")
    history_after = history_snapshot()
    require(history_after == history_before, "request-history files changed during preflight")
    require(artifact_receipts() == artifacts, "immutable artifacts changed during preflight")
    return {
        "format": "apxinf-omniinfer-gateway-overhead-preflight-v1",
        "ok": True,
        "campaign_id": CAMPAIGN_ID,
        "generation_requests": 0,
        "execution_binding": binding,
        "gateway_process": process,
        "backend_process": backend_process,
        "artifacts": artifacts,
        "history_before_and_after_equal": True,
        "history": history_before,
        "state_before_and_after_equal": True,
        "state": state_receipt,
        "health": health,
        "props": props,
        "prompt": {
            "rendered": RENDERED_PROMPT,
            "token_ids": ids,
            "token_ids_sha256": sha256_bytes(compact_json_bytes(ids)),
        },
        "cache_clear": cache_clear,
        "transport_receipts": {
            "health": health_transport,
            "state": state_transport,
            "props": props_transport,
            "apply_template": apply_transport,
            "tokenize": tokenize_transport,
            "state_after": state_after_transport,
        },
    }


def run_probe(_: argparse.Namespace) -> dict[str, Any]:
    binding = execution_binding()
    validate_slot_directory()
    require(not pathlib.Path(ONE_SHOT_MARKER).exists(), "campaign marker already exists")
    process = gateway_process_proof()
    artifacts = artifact_receipts()
    history_before = history_snapshot()
    (
        admin,
        backend,
        gateway,
        state_before,
        connections,
        backend_process,
    ) = open_runtime_connections(process["pid"])
    try:
        backend_sample = run_sample(
            "B", 1, 0, 0, "BG", False, admin, backend, gateway, frozen=False
        )
        gateway_sample = run_sample(
            "G", 2, 0, 0, "BG", False, admin, backend, gateway, frozen=False
        )
        pair = {
            "block": 0,
            "pair_index": 0,
            "order": "BG",
            "measured": False,
            "classification": "exploratory final-shape probe before predeclaration",
            "samples": [backend_sample, gateway_sample],
        }
        validate_pair_equal(pair)
        state_after_raw, state_after_transport = admin.request_json(
            "GET", "/omni/state", None, measured=False
        )
        state_after = validate_state(state_after_raw)
    finally:
        backend.close()
        gateway.close()
        admin.close()
    require(state_after == state_before, "state changed during exploratory probe")
    history_after = history_snapshot()
    require(history_after == history_before, "request-history files changed during probe")
    require(artifact_receipts() == artifacts, "immutable artifacts changed during probe")
    observed = backend_sample["validated"]
    return {
        "format": "apxinf-omniinfer-gateway-overhead-exploratory-probe-v1",
        "ok": True,
        "campaign_id": CAMPAIGN_ID,
        "classification": "exploratory setup; excluded from the future one-shot campaign",
        "generation_requests": 2,
        "performance_samples": 0,
        "execution_binding": binding,
        "gateway_process": process,
        "backend_process": backend_process,
        "artifacts": artifacts,
        "connections": connections,
        "history_before_and_after_equal": True,
        "history": history_before,
        "state_before_and_after_equal": True,
        "state": state_before,
        "state_after_transport": state_after_transport,
        "request_size_bytes": len(REQUEST_BYTES),
        "request_sha256": sha256_bytes(REQUEST_BYTES),
        "observed_freeze_values": {
            "content_sha256": observed["content_sha256"],
            "token_ids": observed["token_ids"],
            "token_ids_compact_json_sha256": observed["token_ids_compact_json_sha256"],
            "rendered_prompt_sha256": observed["rendered_prompt_sha256"],
            "generation_settings_sha256": observed["generation_settings_sha256"],
        },
        "pair": pair,
    }


def create_one_shot_marker(binding: dict[str, Any]) -> dict[str, Any]:
    marker = pathlib.Path(ONE_SHOT_MARKER)
    require(marker.is_absolute(), "campaign marker path is not absolute")
    require(marker.parent.resolve(strict=True).is_dir(), "campaign marker parent is absent")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    payload = compact_json_bytes({
        "campaign_id": CAMPAIGN_ID,
        "campaign_start": binding,
        "created_wall_unix_ns": time.time_ns(),
    }) + b"\n"
    try:
        descriptor = os.open(marker, flags, 0o600)
    except FileExistsError as error:
        raise AdmissionError("one-shot campaign marker already exists") from error
    try:
        written = os.write(descriptor, payload)
        require(written == len(payload), "campaign marker write was incomplete")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return {
        "path": str(marker),
        "size_bytes": marker.stat().st_size,
        "sha256": sha256_file(marker),
        "exclusive_create": True,
    }


def capture_observation(operation: Any) -> dict[str, Any]:
    """Best-effort evidence capture that cannot hide an earlier campaign failure."""
    try:
        return {"ok": True, "value": operation()}
    except Exception as error:
        return {
            "ok": False,
            "error_type": type(error).__name__,
            "error": str(error),
        }


def run_campaign(args: argparse.Namespace) -> dict[str, Any]:
    require(EXPECTED_TOKEN_IDS_SHA256 is not None, "probe token hash has not been frozen")
    require(
        EXPECTED_GENERATION_SETTINGS_SHA256 is not None,
        "probe generation-settings hash has not been frozen",
    )
    binding = execution_binding()
    campaign_start = validate_campaign_start_binding(args)
    validate_slot_directory()
    require(not pathlib.Path(ONE_SHOT_MARKER).exists(), "campaign marker already exists")
    process = gateway_process_proof()
    require(
        process["pid"] != args.exploratory_gateway_pid,
        "campaign gateway PID equals the exploratory probe gateway PID",
    )
    artifacts_before = artifact_receipts()
    history_before = history_snapshot()
    (
        admin,
        backend,
        gateway,
        state_before,
        connections,
        backend_process,
    ) = open_runtime_connections(process["pid"])
    require(
        state_before["backend_pid"] != args.exploratory_backend_pid,
        "campaign backend PID equals the exploratory probe backend PID",
    )
    marker: dict[str, Any] | None = None
    warmup_pairs: list[dict[str, Any]] = []
    measured_pairs: list[dict[str, Any]] = []
    block_state_receipts: list[dict[str, Any]] = []
    sequence_index = 0
    global_pair_index = 0
    in_progress_pair: dict[str, Any] | None = None
    state_after: dict[str, Any] | None = None
    state_after_transport: dict[str, Any] | None = None
    active_phase = "marker"
    cleanup_errors: list[dict[str, str]] = []

    def invalid_record(error: Exception, error_traceback: str) -> dict[str, Any]:
        captured_pairs = [*warmup_pairs, *measured_pairs]
        captured_samples = sum(len(pair["samples"]) for pair in captured_pairs)
        if in_progress_pair is not None:
            captured_samples += len(in_progress_pair["samples"])
        marker_path = pathlib.Path(ONE_SHOT_MARKER)
        return {
            "format": "apxinf-omniinfer-gateway-overhead-raw-campaign-invalid-v1",
            "ok": False,
            "campaign_id": CAMPAIGN_ID,
            "classification": "consumed campaign; invalid diagnostic; no retry permitted",
            "campaign_start_binding": campaign_start,
            "campaign_consumed": marker_path.exists(),
            "one_shot_marker": marker,
            "execution_binding": binding,
            "gateway_process": process,
            "backend_process": backend_process,
            "artifacts_before": artifacts_before,
            "history_before": history_before,
            "state_before": state_before,
            "connections": connections,
            "active_phase_at_failure": active_phase,
            "scheduled_generation_slots_entered": sequence_index,
            "captured_generation_responses": captured_samples,
            "captured_warmup_pairs": len(warmup_pairs),
            "captured_measured_pairs": len(measured_pairs),
            "in_progress_pair": in_progress_pair,
            "warmup_pairs": warmup_pairs,
            "measured_pairs": measured_pairs,
            "block_state_receipts": block_state_receipts,
            "state_after_if_captured": state_after,
            "state_after_transport_if_captured": state_after_transport,
            "post_failure_observations": {
                "history": capture_observation(history_snapshot),
                "artifacts": capture_observation(artifact_receipts),
                "marker_exists": marker_path.exists(),
            },
            "cleanup_errors": cleanup_errors,
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": error_traceback,
        }

    campaign_error: tuple[Exception, str] | None = None
    try:
        marker = create_one_shot_marker(campaign_start)
        active_phase = "warmup"
        for warmup_index, order in enumerate(WARMUP_ORDERS, start=1):
            in_progress_pair = {
                "block": 0,
                "pair_index": warmup_index,
                "order": order,
                "measured": False,
                "samples": [],
            }
            for arm in order:
                sequence_index += 1
                in_progress_pair["samples"].append(run_sample(
                    arm,
                    sequence_index,
                    0,
                    warmup_index,
                    order,
                    False,
                    admin,
                    backend,
                    gateway,
                    frozen=True,
                ))
            validate_pair_equal(in_progress_pair)
            warmup_pairs.append(in_progress_pair)
            in_progress_pair = None
        active_phase = "measured"
        for block in range(1, MEASURED_BLOCKS + 1):
            orders = ODD_BLOCK_ORDERS if block % 2 else EVEN_BLOCK_ORDERS
            for order in orders:
                global_pair_index += 1
                in_progress_pair = {
                    "block": block,
                    "pair_index": global_pair_index,
                    "order": order,
                    "measured": True,
                    "samples": [],
                }
                for arm in order:
                    sequence_index += 1
                    in_progress_pair["samples"].append(run_sample(
                        arm,
                        sequence_index,
                        block,
                        global_pair_index,
                        order,
                        True,
                        admin,
                        backend,
                        gateway,
                        frozen=True,
                    ))
                validate_pair_equal(in_progress_pair)
                measured_pairs.append(in_progress_pair)
                in_progress_pair = None
            state_block_raw, state_transport = admin.request_json(
                "GET", "/omni/state", None, measured=False
            )
            state_block = validate_state(state_block_raw)
            require(state_block == state_before, f"state changed after block {block}")
            block_state_receipts.append({
                "block": block,
                "state": state_block,
                "transport": state_transport,
            })
        active_phase = "final_state"
        state_after_raw, state_after_transport = admin.request_json(
            "GET", "/omni/state", None, measured=False
        )
        state_after = validate_state(state_after_raw)
    except Exception as error:
        campaign_error = (error, traceback.format_exc())
    finally:
        for connection in [backend, gateway, admin]:
            try:
                connection.close()
            except Exception as error:
                cleanup_errors.append({
                    "connection": connection.label,
                    "error_type": type(error).__name__,
                    "error": str(error),
                })
                if campaign_error is None:
                    campaign_error = (error, traceback.format_exc())
    if campaign_error is not None:
        error, error_traceback = campaign_error
        raise CampaignInvalidError(invalid_record(error, error_traceback)) from error
    try:
        active_phase = "postconditions"
        require(marker is not None, "campaign marker was not created")
        require(sequence_index == 136, "request sequence count differs")
        require(global_pair_index == 64, "measured pair index differs")
        require(
            len(warmup_pairs) == 4 and len(measured_pairs) == 64,
            "schedule counts differ",
        )
        require(state_after == state_before, "state changed during campaign")
        history_after = history_snapshot()
        require(
            history_after == history_before,
            "request-history files changed during campaign",
        )
        artifacts_after = artifact_receipts()
        require(artifacts_after == artifacts_before, "immutable artifacts changed during campaign")
        active_phase = "analysis"
        analysis = analyze_pairs(measured_pairs)
    except Exception as error:
        raise CampaignInvalidError(
            invalid_record(error, traceback.format_exc())
        ) from error
    return {
        "format": "apxinf-omniinfer-gateway-overhead-raw-campaign-v1",
        "ok": True,
        "campaign_id": CAMPAIGN_ID,
        "classification": "same-resident-backend paired diagnostic",
        "claim_boundary": analysis["claim_boundary"],
        "campaign_start_binding": campaign_start,
        "campaign_consumed": True,
        "one_shot_marker": marker,
        "execution_binding": binding,
        "gateway_process": process,
        "backend_process": backend_process,
        "fresh_restart_after_exploratory_probe": {
            "exploratory_gateway_pid": args.exploratory_gateway_pid,
            "campaign_gateway_pid": process["pid"],
            "gateway_pid_changed": True,
            "exploratory_backend_pid": args.exploratory_backend_pid,
            "campaign_backend_pid": state_before["backend_pid"],
            "backend_pid_changed": True,
        },
        "artifacts_before": artifacts_before,
        "artifacts_after_equal": True,
        "history_before_and_after_equal": True,
        "history": history_before,
        "state_before_and_after_equal": True,
        "state": state_before,
        "state_after_transport": state_after_transport,
        "block_state_receipts": block_state_receipts,
        "connections": connections,
        "request_contract": {
            "request": REQUEST,
            "utf8": REQUEST_BYTES.decode("utf-8"),
            "size_bytes": len(REQUEST_BYTES),
            "sha256": sha256_bytes(REQUEST_BYTES),
            "client_body_identical_for_B_and_G": True,
            "gateway_forwarded_body_identity": {
                "classification": "source-derived expectation; upstream socket bytes not captured",
                "claim_requires_external_source_binding": True,
                "omniinfer_release_commit": "79af77228f329a79ac665014089e23983e69e79f",
                "normalizer": "normalize_chat_request_with_defaults",
                "runtime_defaults_required": {},
                "proxy_model_required": None,
            },
        },
        "schedule": {
            "warmup_orders": WARMUP_ORDERS,
            "warmup_pairs": 4,
            "measured_blocks": MEASURED_BLOCKS,
            "pairs_per_block": PAIRS_PER_BLOCK,
            "measured_pairs": 64,
            "measured_requests": 128,
            "total_generation_requests": 136,
            "retry_resample_replacement_or_outlier_removal": False,
        },
        "warmup_pairs": warmup_pairs,
        "measured_pairs": measured_pairs,
        "analysis": analysis,
    }


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    subcommands = root.add_subparsers(dest="command", required=True)
    self_test = subcommands.add_parser("self-test", help="run zero-generation fixture checks")
    self_test.set_defaults(run=run_self_test)
    preflight = subcommands.add_parser("preflight", help="run zero-generation runtime checks")
    preflight.set_defaults(run=run_preflight)
    probe = subcommands.add_parser("probe", help="run one exploratory B/G pair")
    probe.set_defaults(run=run_probe)
    campaign = subcommands.add_parser("campaign", help="run the frozen one-shot campaign")
    campaign.add_argument("--campaign-start-sha256", required=True)
    campaign.add_argument("--campaign-start-commit", required=True)
    campaign.add_argument("--exploratory-gateway-pid", required=True, type=int)
    campaign.add_argument("--exploratory-backend-pid", required=True, type=int)
    campaign.set_defaults(run=run_campaign)
    return root


def main() -> int:
    args = parser().parse_args()
    try:
        result = args.run(args)
    except CampaignInvalidError as error:
        emit(error.record)
        return 1
    except Exception as error:
        emit({
            "format": "apxinf-omniinfer-gateway-overhead-driver-failure-v1",
            "ok": False,
            "campaign_id": CAMPAIGN_ID,
            "command": getattr(args, "command", None),
            "campaign_marker_exists": pathlib.Path(ONE_SHOT_MARKER).exists(),
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        })
        return 1
    emit(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
