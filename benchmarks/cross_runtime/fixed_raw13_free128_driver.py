#!/usr/bin/env python3
"""Narrow, fail-closed driver for the Qwen3.5 raw13/free128 diagnostic.

The driver writes no files.  Each invocation emits exactly one compact JSON
record on stdout and exits non-zero after emitting a failure record when an
admission check fails.  Campaign ordering, publication, and host custody stay
in the outer evidence protocol.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pathlib
import statistics
import subprocess
import sys
import time
import traceback
import urllib.error
import urllib.request
from typing import Any


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
PROMPT_TOKEN_IDS_SHA256 = "4b890fa15ee3d7db4e9dd18bd79c6362d40e9e016ae4f9f74cb7fc420ef3b6d3"
CANONICAL_OUTPUT_IDS_SHA256 = "2042d5522ed5e768b938ac3fd9e19d3936dfc6e685fc157d41a8e8132c5b42fe"
RENDERED_PROMPT = (
    "<|im_start|>user\n"
    "Hello<|im_end|>\n"
    "<|im_start|>assistant\n"
    "<think>\n\n</think>\n"
    "\n"
)
APPLY_TEMPLATE_REQUEST = {
    "messages": [{"role": "user", "content": "Hello"}],
    "add_generation_prompt": True,
    "chat_template_kwargs": {"enable_thinking": False},
}
OMNI_REQUEST = {
    "messages": [{"role": "user", "content": "Hello"}],
    "temperature": 0,
    "max_tokens": 128,
    "stream": False,
    "think": False,
    "cache_prompt": False,
    "ignore_eos": True,
}
CAMPAIGN_ID = "qwen35-0.8b-raw13-free128-omniinfer-diagnostic-v2-20260826"
CAMPAIGN_START_PATH = (
    "crates/apxinf-metal/evidence/llama-cpp/"
    "qwen35-0.8b-apxinf-fused-c-vs-llamacpp-and-omniinfer-raw13-free128-"
    "campaign-start-diagnostic-v2-20260826.json"
)
REPOSITORY_ROOT = "/Users/haiyan-mini/Agent4Kernel/ApxInf"
MODEL_PATH = (
    "/Users/haiyan-mini/Agent4Kernel/models/Qwen3.5-0.8B-2fc063647-GGUF/"
    "Qwen3.5-0.8B-2fc063647-Q8_0-llama.gguf"
)
MODEL_SIZE = 811843072
DIRECT_RUNNER_PATH = "/tmp/apxinf-llama-runner-build/bin/apxinf-llama-cpp-raw-token-runner"
DIRECT_RUNNER_SIZE = 6499056
DIRECT_RUNNER_SHA256 = "ccfa5ecd78119d4f8cdd8721e7faae360cb94b8334f9d61ed47e2e00290f2716"
DIRECT_SOURCE_COMMIT = "f280b26983ad0fdb705a0d9ebf0503e76f2899b0"
OMNI_BASE_URL = "http://127.0.0.1:9000"
OMNI_SLOT_SAVE_PATH = "/tmp/apxinf-omniinfer-prep.F7GNFt/slots-cold-v2"
EXPECTED_DRIVER_PATH = (
    "/Users/haiyan-mini/Agent4Kernel/ApxInf/benchmarks/cross_runtime/"
    "fixed_raw13_free128_driver.py"
)
EXPECTED_PYTHON_PATH = (
    "/opt/homebrew/Cellar/python@3.14/3.14.3/Frameworks/Python.framework/"
    "Versions/3.14/bin/python3.14"
)
EXPECTED_PYTHON_SIZE = 52448
EXPECTED_PYTHON_SHA256 = "ff73afba45e095e0dadc9b51deb9b994ef90d097c36a211955c57336ce76508f"
EXPECTED_PYTHON_VERSION_PREFIX = "3.14.3 (main, Feb  3 2026, 15:32:20)"
OMNI_LLAMA_SERVER_PATH = (
    "/tmp/apxinf-omniinfer-prep.F7GNFt/runtime/llama.cpp-mac/bin/llama-server"
)
OMNI_RUNTIME_LOGS_PATH = "/tmp/apxinf-omniinfer-prep.F7GNFt/runtime/llama.cpp-mac/logs"


class AdmissionError(RuntimeError):
    """A fixed-contract admission check failed."""


class NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Reject redirects so the receipt stays bound to the declared endpoint."""

    def redirect_request(self, req: Any, fp: Any, code: int, msg: str, headers: Any, newurl: str) -> None:
        return None


def compact_json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


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
    require(isinstance(value, (int, float)) and not isinstance(value, bool), f"{label} is not numeric")
    result = float(value)
    require(math.isfinite(result) and result > 0.0, f"{label} is not finite and positive")
    return result


def relative_close(actual: float, expected: float, tolerance: float = 1e-5) -> bool:
    return abs(actual - expected) <= tolerance * max(abs(actual), abs(expected), 1.0)


def http_json(
    method: str,
    url: str,
    payload: dict[str, Any] | None,
    timeout_seconds: float,
) -> dict[str, Any]:
    data = None if payload is None else compact_json_bytes(payload)
    request = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={"Content-Type": "application/json", "Connection": "close"},
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirectHandler())
    try:
        with opener.open(request, timeout=timeout_seconds) as response:
            body = response.read()
            status = response.status
    except urllib.error.HTTPError as error:
        body = error.read()
        raise AdmissionError(
            f"HTTP {error.code} from {url}: {body[-2000:].decode('utf-8', 'replace')}"
        ) from error
    except urllib.error.URLError as error:
        raise AdmissionError(f"request to {url} failed: {error}") from error
    require(status == 200, f"HTTP status from {url} was {status}")
    try:
        decoded = json.loads(body)
    except json.JSONDecodeError as error:
        raise AdmissionError(f"response from {url} was not JSON") from error
    require(isinstance(decoded, dict), f"response from {url} was not a JSON object")
    return decoded


def essential_state(state: dict[str, Any]) -> dict[str, Any]:
    runtime = state.get("runtime") if isinstance(state.get("runtime"), dict) else {}
    return {
        "backend": state.get("backend"),
        "backend_ready": state.get("backend_ready"),
        "model": state.get("model"),
        "model_path": state.get("model_path"),
        "mmproj": state.get("mmproj"),
        "ctx_size": state.get("ctx_size"),
        "backend_pid": state.get("backend_pid"),
        "backend_port": state.get("backend_port"),
        "client_endpoint": state.get("client_endpoint"),
        "generation": state.get("generation"),
        "launch_command": state.get("launch_command"),
        "launch_args": state.get("launch_args"),
        "request_defaults": state.get("request_defaults"),
        "effective_parameters": state.get("effective_parameters"),
        "restore_selection_request_defaults": (
            state.get("restore_selection", {}).get("request_defaults")
            if isinstance(state.get("restore_selection"), dict)
            else None
        ),
        "runtime": {
            "mode": runtime.get("mode"),
            "client_endpoint": runtime.get("client_endpoint"),
            "pid": runtime.get("pid"),
            "port": runtime.get("port"),
            "launch_command": runtime.get("launch_command"),
        },
    }


def expected_omni_launch_args(slot_save_path: str) -> list[str]:
    return [
        "--slot-prompt-similarity",
        "0",
        "--cache-idle-slots",
        "-ngl",
        "999",
        "--cache-ram",
        "8192",
        "-ngl",
        "999",
        "-b",
        "13",
        "-ub",
        "13",
        "-t",
        "4",
        "-tb",
        "4",
        "-np",
        "1",
        "--cache-ram",
        "0",
        "--slots",
        "--no-cache-idle-slots",
        "--no-cache-prompt",
        "--slot-prompt-similarity",
        "0",
        "--slot-save-path",
        slot_save_path,
    ]


def validate_omni_state(state: dict[str, Any], model: str, slot_save_path: str) -> None:
    require(state.get("backend") == "llama.cpp-mac", "OmniInfer backend is not llama.cpp-mac")
    require(state.get("backend_ready") is True, "OmniInfer backend is not ready")
    require(state.get("model_path") == model, "OmniInfer loaded model path differs from the bound GGUF")
    require("mmproj" in state and state.get("mmproj") is None, "OmniInfer unexpectedly loaded mmproj")
    require(state.get("request_defaults") == {}, "OmniInfer request defaults are not empty")
    require(state.get("effective_parameters") == {}, "OmniInfer effective parameters are not empty")
    require(state.get("default_model") == model, "OmniInfer default model differs")
    require(state.get("proxy_model") is None, "OmniInfer unexpectedly uses a proxy model")
    require(state.get("public_model_id") is None, "OmniInfer unexpectedly uses a public model ID")
    require(state.get("ctx_size") == 256, "OmniInfer context length is not 256")
    require(isinstance(state.get("backend_pid"), int), "OmniInfer state has no backend PID")
    require(isinstance(state.get("backend_port"), int), "OmniInfer state has no backend port")
    launch = state.get("launch_command")
    require(isinstance(launch, list), "OmniInfer state has no launch command array")
    launch_args = state.get("launch_args")
    expected_args = expected_omni_launch_args(slot_save_path)
    require(launch_args == expected_args, "OmniInfer launch_args differ from the exact frozen list")
    expected_command = [
        OMNI_LLAMA_SERVER_PATH,
        "-m",
        model,
        "--host",
        "127.0.0.1",
        "--port",
        str(state["backend_port"]),
        "--no-webui",
        "--slot-save-path",
        OMNI_RUNTIME_LOGS_PATH,
        *expected_args,
        "-c",
        "256",
    ]
    require(launch == expected_command, "OmniInfer launch command differs from the exact frozen list")
    runtime = state.get("runtime")
    require(isinstance(runtime, dict), "OmniInfer state has no runtime object")
    require(runtime.get("launch_command") == launch, "runtime launch command differs from top-level")
    require(runtime.get("pid") == state.get("backend_pid"), "runtime PID differs from backend PID")
    require(runtime.get("port") == state.get("backend_port"), "runtime port differs from backend port")
    expected_endpoint = f"http://127.0.0.1:{state['backend_port']}"
    require(state.get("client_endpoint") == expected_endpoint, "backend client endpoint differs")
    require(runtime.get("client_endpoint") == expected_endpoint, "runtime client endpoint differs")
    loaded_models = state.get("loaded_models")
    require(isinstance(loaded_models, list) and len(loaded_models) == 1, "loaded model set differs")
    loaded = loaded_models[0]
    require(isinstance(loaded, dict), "loaded model state is not an object")
    require(loaded.get("model_path") == model, "loaded model state path differs")
    require(loaded.get("request_defaults") == {}, "loaded model request defaults are not empty")
    require(loaded.get("mmproj") is None, "loaded model state unexpectedly has mmproj")
    restore = state.get("restore_selection")
    require(isinstance(restore, dict), "OmniInfer restore selection is absent")
    require(restore.get("model") == model, "OmniInfer restore model differs")
    require(restore.get("ctx_size") == 256, "OmniInfer restore context differs")
    require(restore.get("mmproj") is None, "OmniInfer restore selection has mmproj")
    require(restore.get("no_mmproj") is True, "OmniInfer restore selection does not disable mmproj")
    require(restore.get("request_defaults") == {}, "OmniInfer restore request defaults are not empty")


def validate_omni_props(props: dict[str, Any], model: str) -> int:
    require(props.get("build_info") == "b10280-61881b1f7", "unexpected llama-server build_info")
    require(props.get("model_path") == model, "backend props model path differs")
    require(props.get("model_ftype") == "Q8_0", "backend props model type is not Q8_0")
    require(props.get("endpoint_slots") is True, "backend slot endpoint is not enabled")
    require(props.get("total_slots") == 1, "backend slot count is not one")
    defaults = props.get("default_generation_settings")
    require(isinstance(defaults, dict), "backend props lacks default generation settings")
    require(defaults.get("n_ctx") == 256, "backend effective default context is not 256")
    return 1


def clear_omni_cache(base: str, total_slots: int, timeout_seconds: float) -> dict[str, Any]:
    cache_clear = http_json("POST", f"{base}/omni/cache/clear", {}, timeout_seconds)
    cleared_slots = cache_clear.get("cleared_slots")
    require(cache_clear.get("ok") is True, "OmniInfer cache clear was not acknowledged")
    require(
        cleared_slots == list(range(total_slots)),
        "cache clear did not erase each advertised slot exactly once",
    )
    return cache_clear


def validate_fixed_direct_args(args: argparse.Namespace) -> None:
    require(args.runner == DIRECT_RUNNER_PATH, "direct runner argument differs from fixed path")
    require(args.runner_size == DIRECT_RUNNER_SIZE, "direct runner size argument differs")
    require(args.runner_sha256 == DIRECT_RUNNER_SHA256, "direct runner SHA256 argument differs")
    require(args.source_commit == DIRECT_SOURCE_COMMIT, "direct source commit argument differs")
    require(args.model == MODEL_PATH, "direct model argument differs from fixed path")
    require(args.model_size == MODEL_SIZE, "direct model size argument differs")


def validate_fixed_omni_args(args: argparse.Namespace) -> None:
    require(args.base_url == OMNI_BASE_URL, "OmniInfer base URL differs from fixed URL")
    require(args.model == MODEL_PATH, "OmniInfer model argument differs from fixed path")
    require(args.slot_save_path == OMNI_SLOT_SAVE_PATH, "OmniInfer slot path differs")


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


def validate_campaign_start_binding(args: argparse.Namespace) -> dict[str, Any]:
    require(
        len(args.campaign_start_sha256) == 64
        and all(character in "0123456789abcdef" for character in args.campaign_start_sha256),
        "campaign-start SHA256 is not lowercase hexadecimal",
    )
    require(
        len(args.campaign_start_commit) == 40
        and all(character in "0123456789abcdef" for character in args.campaign_start_commit),
        "campaign-start commit is not lowercase hexadecimal",
    )
    repository = pathlib.Path(REPOSITORY_ROOT).resolve(strict=True)
    start = (repository / CAMPAIGN_START_PATH).resolve(strict=True)
    require(start.is_file(), "campaign-start path is not a regular file")
    start_stat = start.stat()
    require(sha256_file(start) == args.campaign_start_sha256, "campaign-start SHA256 differs")
    head = git_stdout(["rev-parse", "HEAD"])
    origin_main = git_stdout(["rev-parse", "origin/main"])
    require(head == args.campaign_start_commit, "Git HEAD differs from campaign-start commit")
    require(origin_main == args.campaign_start_commit, "origin/main differs from campaign-start commit")
    require(
        git_stdout(["status", "--porcelain=v1", "--untracked-files=all"]) == "",
        "worktree is not clean during campaign sampling",
    )
    return {
        "campaign_id": CAMPAIGN_ID,
        "campaign_start_path": CAMPAIGN_START_PATH,
        "campaign_start_sha256": args.campaign_start_sha256,
        "campaign_start_commit": args.campaign_start_commit,
        "campaign_start_size_bytes": start_stat.st_size,
        "git_head": head,
        "git_origin_main": origin_main,
        "git_worktree_clean": True,
    }


def execution_binding() -> dict[str, Any]:
    driver = pathlib.Path(__file__).resolve(strict=True)
    python = pathlib.Path(sys.executable).resolve(strict=True)
    require(str(driver) == EXPECTED_DRIVER_PATH, "driver path differs from the fixed campaign path")
    require(str(python) == EXPECTED_PYTHON_PATH, "Python interpreter path differs")
    python_stat = python.stat()
    require(python_stat.st_size == EXPECTED_PYTHON_SIZE, "Python interpreter size differs")
    require(sha256_file(python) == EXPECTED_PYTHON_SHA256, "Python interpreter SHA256 differs")
    require(sys.version.startswith(EXPECTED_PYTHON_VERSION_PREFIX), "Python version differs")
    driver_stat = driver.stat()
    return {
        "transport": "absolute file-backed Python invocation; no python -c, eval, or source argument",
        "argv": sys.argv,
        "driver": {
            "path": str(driver),
            "size_bytes": driver_stat.st_size,
            "sha256": sha256_file(driver),
        },
        "python": {
            "path": str(python),
            "size_bytes": python_stat.st_size,
            "sha256": EXPECTED_PYTHON_SHA256,
            "version": sys.version,
        },
    }


def tokenize_ids(tokenize: dict[str, Any]) -> list[int]:
    tokens = tokenize.get("tokens")
    require(isinstance(tokens, list), "tokenize response has no tokens array")
    result: list[int] = []
    for token in tokens:
        if isinstance(token, int):
            result.append(token)
        elif isinstance(token, dict) and isinstance(token.get("id"), int):
            result.append(token["id"])
        else:
            raise AdmissionError("tokenize response contains an unrecognized token record")
    return result


def validate_omni_response(response: dict[str, Any]) -> dict[str, float]:
    usage = response.get("usage")
    timings = response.get("timings")
    choices = response.get("choices")
    require(isinstance(usage, dict), "OmniInfer response has no usage object")
    require(isinstance(timings, dict), "OmniInfer response has no timings object")
    require(isinstance(choices, list) and len(choices) == 1, "OmniInfer response choices are not singular")
    require(usage.get("prompt_tokens") == 13, "OmniInfer prompt token count is not 13")
    require(usage.get("completion_tokens") == 128, "OmniInfer completion token count is not 128")
    require(usage.get("total_tokens") == 141, "OmniInfer total token count is not 141")
    require(timings.get("prompt_n") == 13, "llama-server prompt_n is not 13")
    require(timings.get("predicted_n") == 128, "llama-server predicted_n is not 128")
    require(timings.get("cache_n") in (0, 0.0), "llama-server reported a prompt cache hit")
    prompt_ms = positive_number(timings.get("prompt_ms"), "timings.prompt_ms")
    predicted_ms = positive_number(timings.get("predicted_ms"), "timings.predicted_ms")
    prompt_tps = positive_number(timings.get("prompt_per_second"), "timings.prompt_per_second")
    predicted_tps = positive_number(
        timings.get("predicted_per_second"), "timings.predicted_per_second"
    )
    require(relative_close(prompt_tps, 13000.0 / prompt_ms), "prompt TPS does not match prompt_ms")
    require(
        relative_close(predicted_tps, 128000.0 / predicted_ms),
        "predicted TPS does not match predicted_ms",
    )
    choice = choices[0]
    require(isinstance(choice, dict), "OmniInfer choice is not an object")
    require(choice.get("finish_reason") == "length", "OmniInfer did not finish at max_tokens")
    message = choice.get("message")
    require(isinstance(message, dict), "OmniInfer choice has no message")
    content = message.get("content")
    require(isinstance(content, str), "OmniInfer response content is not a string")
    return {
        "prompt_ms": prompt_ms,
        "prompt_tps": prompt_tps,
        "predicted_ms": predicted_ms,
        "predicted_tps": predicted_tps,
        "content_size_bytes": float(len(content.encode("utf-8"))),
    }


def run_self_test(_: argparse.Namespace) -> dict[str, Any]:
    bound_execution = execution_binding()
    require(sha256_bytes(compact_json_bytes(PROMPT_TOKEN_IDS)) == PROMPT_TOKEN_IDS_SHA256, "prompt hash")
    fixture = {
        "usage": {"prompt_tokens": 13, "completion_tokens": 128, "total_tokens": 141},
        "timings": {
            "cache_n": 0,
            "prompt_n": 13,
            "prompt_ms": 10.0,
            "prompt_per_second": 1300.0,
            "predicted_n": 128,
            "predicted_ms": 2000.0,
            "predicted_per_second": 64.0,
        },
        "choices": [{"finish_reason": "length", "message": {"content": "fixture"}}],
    }
    metrics = validate_omni_response(fixture)
    samples = [63.0, 64.0, 65.0]
    mean = statistics.fmean(samples)
    population_cv = statistics.pstdev(samples) / mean
    require(relative_close(mean, 64.0), "statistics mean")
    require(population_cv > 0.0, "statistics CV")
    return {
        "format": "apxinf-cross-runtime-fixed-raw13-free128-driver-self-test-v2",
        "ok": True,
        "campaign_id": CAMPAIGN_ID,
        "execution_binding": bound_execution,
        "prompt_token_ids_sha256": PROMPT_TOKEN_IDS_SHA256,
        "fixture_metrics": metrics,
        "fixture_population_cv": population_cv,
    }


def run_preflight(args: argparse.Namespace) -> dict[str, Any]:
    bound_execution = execution_binding()
    validate_fixed_omni_args(args)
    base = args.base_url.rstrip("/")
    slot_save_path_arg = pathlib.Path(args.slot_save_path)
    require(slot_save_path_arg.is_absolute(), "slot save path is not absolute")
    slot_save_path_resolved = slot_save_path_arg.resolve(strict=True)
    require(slot_save_path_resolved.is_dir(), "slot save path is not a directory")
    health = http_json("GET", f"{base}/health?deep=true", None, args.timeout_seconds)
    state = http_json("GET", f"{base}/omni/state", None, args.timeout_seconds)
    props = http_json("GET", f"{base}/omni/backend/props", None, args.timeout_seconds)
    validate_omni_state(state, args.model, args.slot_save_path)
    applied_template = http_json(
        "POST",
        f"{state['client_endpoint']}/apply-template",
        APPLY_TEMPLATE_REQUEST,
        args.timeout_seconds,
    )
    require(
        applied_template.get("prompt") == RENDERED_PROMPT,
        "backend apply-template output differs from the frozen rendered prompt",
    )
    tokenized = http_json(
        "POST",
        f"{base}/tokenize",
        {"content": RENDERED_PROMPT, "add_special": False, "with_pieces": True},
        args.timeout_seconds,
    )
    require(health.get("status") == "ok", "gateway health status is not ok")
    omni_health = health.get("omni")
    backend_health = health.get("backend_health")
    require(isinstance(omni_health, dict), "health response lacks omni state")
    require(omni_health.get("backend_ready") is True, "deep health backend is not ready")
    require(isinstance(backend_health, dict), "health response lacks backend health")
    require(backend_health.get("status") == "ok", "backend health status is not ok")
    total_slots = validate_omni_props(props, args.model)
    ids = tokenize_ids(tokenized)
    require(ids == PROMPT_TOKEN_IDS, f"OmniInfer rendered prompt token IDs differ: {ids!r}")
    cache_clear = clear_omni_cache(base, total_slots, args.timeout_seconds)
    state_after = http_json("GET", f"{base}/omni/state", None, args.timeout_seconds)
    validate_omni_state(state_after, args.model, args.slot_save_path)
    require(
        essential_state(state) == essential_state(state_after),
        "OmniInfer essential state changed during zero-generation preflight",
    )
    return {
        "format": "apxinf-cross-runtime-fixed-raw13-free128-omniinfer-preflight-v2",
        "ok": True,
        "campaign_id": CAMPAIGN_ID,
        "execution_binding": bound_execution,
        "generation_requests": 0,
        "state_before_and_after_equal": True,
        "cache_clear": cache_clear,
        "state": essential_state(state),
        "health": {
            "status": health.get("status"),
            "backend_status": backend_health.get("status"),
        },
        "props": {
            "build_info": props.get("build_info"),
            "model_path": props.get("model_path"),
            "model_ftype": props.get("model_ftype"),
            "total_slots": props.get("total_slots"),
            "default_generation_settings": props.get("default_generation_settings"),
        },
        "prompt": {
            "apply_template_request": APPLY_TEMPLATE_REQUEST,
            "rendered": RENDERED_PROMPT,
            "token_ids": ids,
            "token_ids_sha256": sha256_bytes(compact_json_bytes(ids)),
        },
    }


def run_direct(args: argparse.Namespace) -> dict[str, Any]:
    bound_execution = execution_binding()
    validate_fixed_direct_args(args)
    bound_campaign = validate_campaign_start_binding(args)
    require(args.measured == (args.slot != "D_warmup"), "direct slot measured flag differs")
    runner = pathlib.Path(args.runner).resolve(strict=True)
    model = pathlib.Path(args.model).resolve(strict=True)
    require(runner.is_file(), "direct runner is not a regular file")
    require(model.is_file(), "model is not a regular file")
    runner_stat = runner.stat()
    model_before = model.stat()
    require(runner_stat.st_size == args.runner_size, "direct runner size changed")
    require(sha256_file(runner) == args.runner_sha256, "direct runner SHA256 changed")
    require(model_before.st_size == args.model_size, "model size changed")
    command = [
        args.runner,
        "--model",
        str(model),
        "--gpu-layers",
        "-1",
        "--gpu-device",
        "MTL0",
        "--threads",
        "4",
    ]
    child_env = os.environ.copy()
    child_env.pop("GGML_BACKEND_PATH", None)
    started_ns = time.monotonic_ns()
    process = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=child_env,
        timeout=args.timeout_seconds,
        check=False,
    )
    ended_ns = time.monotonic_ns()
    model_after = model.stat()
    require(process.returncode == 0, f"direct runner exited {process.returncode}")
    require(
        process.stdout.endswith(b"\n") and process.stdout.count(b"\n") == 1,
        "direct runner stdout was not exactly one newline-terminated JSON record",
    )
    try:
        receipt = json.loads(process.stdout)
    except json.JSONDecodeError as error:
        raise AdmissionError("direct runner stdout was not JSON") from error
    require(isinstance(receipt, dict) and receipt.get("ok") is True, "direct runner receipt failed")
    contract = receipt.get("contract")
    output = receipt.get("output")
    parameters = receipt.get("parameters")
    require(isinstance(contract, dict), "direct receipt has no contract")
    require(isinstance(output, dict), "direct receipt has no output")
    require(isinstance(parameters, dict), "direct receipt has no parameters")
    require(contract.get("prompt_token_ids") == PROMPT_TOKEN_IDS, "direct prompt IDs differ")
    require(contract.get("sampling") == "greedy-argmax", "direct sampling is not greedy argmax")
    require(contract.get("generated_token_count") == 128, "direct output count is not 128")
    require(contract.get("eog_termination") is False, "direct runner enabled EOG termination")
    ids = output.get("token_ids")
    times = output.get("token_ready_elapsed_ns")
    require(isinstance(ids, list) and len(ids) == 128, "direct output IDs length is not 128")
    require(all(isinstance(value, int) for value in ids), "direct output IDs are not integers")
    require(isinstance(times, list) and len(times) == 128, "direct token times length is not 128")
    require(all(isinstance(value, int) and value > 0 for value in times), "direct token times invalid")
    require(all(right > left for left, right in zip(times, times[1:])), "direct token times not strict")
    trajectory_sha256 = sha256_bytes(compact_json_bytes(ids))
    require(trajectory_sha256 == CANONICAL_OUTPUT_IDS_SHA256, "direct trajectory hash differs")
    require(parameters.get("n_ctx_requested") == 142, "direct requested context is not 142")
    require(parameters.get("n_ctx_effective") == 256, "direct effective context is not 256")
    require(
        parameters.get("n_ctx_per_sequence_effective") == 256,
        "direct per-sequence effective context is not 256",
    )
    require(parameters.get("n_batch_requested") == 13, "direct requested batch is not 13")
    require(parameters.get("n_batch_effective") == 13, "direct effective batch is not 13")
    require(parameters.get("n_ubatch_requested") == 13, "direct requested ubatch is not 13")
    require(parameters.get("n_ubatch_effective") == 13, "direct effective ubatch is not 13")
    require(parameters.get("n_seq_max_requested") == 1, "direct requested sequence count is not one")
    require(parameters.get("n_seq_max_effective") == 1, "direct effective sequence count is not one")
    require(parameters.get("n_threads") == 4, "direct threads is not four")
    require(parameters.get("n_threads_batch") == 4, "direct batch threads is not four")
    require(parameters.get("lane") == "gpu-all-layers", "direct lane is not GPU all layers")
    require(parameters.get("n_gpu_layers") == -1, "direct GPU layer request is not all layers")
    model_receipt = receipt.get("model")
    require(isinstance(model_receipt, dict), "direct receipt has no model object")
    require(model_receipt.get("requested_path") == str(model), "direct receipt model path differs")
    require(model_receipt.get("file_size_bytes") == args.model_size, "direct receipt model size differs")
    require(model_receipt.get("file_type") == "Q8_0", "direct receipt model type is not Q8_0")
    require(model_receipt.get("file_identity_unchanged") is True, "direct model identity changed")
    build = receipt.get("build")
    require(isinstance(build, dict), "direct receipt has no build")
    require(build.get("llama_cpp_source_id") == args.source_commit, "direct source ID differs")
    backend = receipt.get("backend")
    require(isinstance(backend, dict), "direct receipt has no backend object")
    require(backend.get("ggml_backend_path_present") is False, "direct backend path was injected")
    selected_device = backend.get("selected_gpu_device")
    require(
        isinstance(selected_device, dict) and selected_device.get("name") == "MTL0",
        "direct selected GPU is not MTL0",
    )
    placement = receipt.get("placement_attestation")
    proof = receipt.get("post_measurement_execution_proof")
    require(isinstance(placement, dict) and placement.get("passed") is True, "placement failed")
    require(isinstance(proof, dict) and proof.get("passed") is True, "execution proof failed")
    identity_before = (model_before.st_dev, model_before.st_ino, model_before.st_size, model_before.st_ctime_ns)
    identity_after = (model_after.st_dev, model_after.st_ino, model_after.st_size, model_after.st_ctime_ns)
    require(identity_before == identity_after, "model identity changed during direct run")
    ttft_ms = times[0] / 1e6
    total_latency_ms = times[-1] / 1e6
    tpot_ms = (times[-1] - times[0]) / 127e6
    return {
        "format": "apxinf-cross-runtime-fixed-raw13-free128-direct-sample-v2",
        "ok": True,
        "campaign_binding": bound_campaign,
        "execution_binding": bound_execution,
        "slot": args.slot,
        "measured": args.measured,
        "command": command,
        "GGML_BACKEND_PATH_unset": "GGML_BACKEND_PATH" not in child_env,
        "process_wall_ms": (ended_ns - started_ns) / 1e6,
        "stdout_size_bytes": len(process.stdout),
        "stdout_sha256": sha256_bytes(process.stdout),
        "stderr_size_bytes": len(process.stderr),
        "stderr_sha256": sha256_bytes(process.stderr),
        "trajectory_compact_json_sha256": trajectory_sha256,
        "derived": {
            "ttft_ms": ttft_ms,
            "total_latency_ms": total_latency_ms,
            "tpot_ms": tpot_ms,
            "generation_tps": 1000.0 / tpot_ms,
        },
        "model_identity_unchanged": True,
        "receipt": receipt,
    }


def run_omni(args: argparse.Namespace) -> dict[str, Any]:
    bound_execution = execution_binding()
    validate_fixed_omni_args(args)
    bound_campaign = validate_campaign_start_binding(args)
    require(args.measured == (args.slot != "N_warmup"), "OmniInfer slot measured flag differs")
    base = args.base_url.rstrip("/")
    slot_save_path_arg = pathlib.Path(args.slot_save_path)
    require(slot_save_path_arg.is_absolute(), "slot save path is not absolute")
    slot_save_path_resolved = slot_save_path_arg.resolve(strict=True)
    require(slot_save_path_resolved.is_dir(), "slot save path is not a directory")
    state_before = http_json("GET", f"{base}/omni/state", None, args.timeout_seconds)
    props = http_json("GET", f"{base}/omni/backend/props", None, args.timeout_seconds)
    validate_omni_state(state_before, args.model, args.slot_save_path)
    total_slots = validate_omni_props(props, args.model)
    cache_clear = clear_omni_cache(base, total_slots, args.timeout_seconds)
    started_ns = time.monotonic_ns()
    response = http_json(
        "POST", f"{base}/v1/chat/completions", OMNI_REQUEST, args.timeout_seconds
    )
    ended_ns = time.monotonic_ns()
    metrics = validate_omni_response(response)
    state_after = http_json("GET", f"{base}/omni/state", None, args.timeout_seconds)
    validate_omni_state(state_after, args.model, args.slot_save_path)
    before_essential = essential_state(state_before)
    after_essential = essential_state(state_after)
    require(before_essential == after_essential, "OmniInfer state changed during the request")
    content = response["choices"][0]["message"]["content"]
    return {
        "format": "apxinf-cross-runtime-fixed-raw13-free128-omniinfer-sample-v2",
        "ok": True,
        "campaign_binding": bound_campaign,
        "execution_binding": bound_execution,
        "slot": args.slot,
        "measured": args.measured,
        "request": OMNI_REQUEST,
        "cache_clear": cache_clear,
        "state_before_and_after_equal": True,
        "state": before_essential,
        "gateway_wall_time_ms": (ended_ns - started_ns) / 1e6,
        "native": {
            "prompt_ms": metrics["prompt_ms"],
            "prompt_tps": metrics["prompt_tps"],
            "predicted_ms": metrics["predicted_ms"],
            "predicted_tps": metrics["predicted_tps"],
        },
        "response_content_size_bytes": len(content.encode("utf-8")),
        "response_content_sha256": sha256_bytes(content.encode("utf-8")),
        "response": response,
    }


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    subcommands = root.add_subparsers(dest="command", required=True)
    self_test = subcommands.add_parser("self-test", help="run fixture-only checks")
    self_test.set_defaults(run=run_self_test)

    preflight = subcommands.add_parser("preflight", help="validate OmniInfer without generation")
    preflight.add_argument("--base-url", required=True)
    preflight.add_argument("--model", required=True)
    preflight.add_argument("--slot-save-path", required=True)
    preflight.add_argument("--timeout-seconds", type=float, default=30.0)
    preflight.set_defaults(run=run_preflight)

    direct = subcommands.add_parser("direct", help="run one direct llama.cpp sample")
    direct.add_argument("--slot", choices=["D_warmup", "D1", "D2", "D3"], required=True)
    direct.add_argument("--measured", action="store_true")
    direct.add_argument("--campaign-start-sha256", required=True)
    direct.add_argument("--campaign-start-commit", required=True)
    direct.add_argument("--runner", required=True)
    direct.add_argument("--runner-size", type=int, required=True)
    direct.add_argument("--runner-sha256", required=True)
    direct.add_argument("--source-commit", required=True)
    direct.add_argument("--model", required=True)
    direct.add_argument("--model-size", type=int, required=True)
    direct.add_argument("--timeout-seconds", type=float, default=600.0)
    direct.set_defaults(run=run_direct)

    omni = subcommands.add_parser("omni", help="run one cold-KV OmniInfer sample")
    omni.add_argument("--slot", choices=["N_warmup", "N1", "N2", "N3"], required=True)
    omni.add_argument("--measured", action="store_true")
    omni.add_argument("--campaign-start-sha256", required=True)
    omni.add_argument("--campaign-start-commit", required=True)
    omni.add_argument("--base-url", required=True)
    omni.add_argument("--model", required=True)
    omni.add_argument("--slot-save-path", required=True)
    omni.add_argument("--timeout-seconds", type=float, default=600.0)
    omni.set_defaults(run=run_omni)
    return root


def main() -> int:
    args = parser().parse_args()
    try:
        result = args.run(args)
    except Exception as error:  # fail closed while preserving a machine-readable receipt
        emit(
            {
                "format": "apxinf-cross-runtime-fixed-raw13-free128-driver-failure-v2",
                "ok": False,
                "campaign_id": CAMPAIGN_ID,
                "command": args.command,
                "slot": getattr(args, "slot", None),
                "campaign_start_sha256": getattr(args, "campaign_start_sha256", None),
                "campaign_start_commit": getattr(args, "campaign_start_commit", None),
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            }
        )
        return 1
    emit(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
