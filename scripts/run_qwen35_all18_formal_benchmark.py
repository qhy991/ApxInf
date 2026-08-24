#!/usr/bin/env python3
"""Fail-closed formal benchmark harness for the frozen Qwen3.5 all18 lane.

The default command only validates inputs and emits a plan.  A real model is
started only when the caller supplies ``--execute``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import secrets
import selectors
import signal
import stat
import statistics
import subprocess
import sys
import time


BLOCK_ORDERS = ("ABBA", "BAAB", "ABBA", "BAAB", "ABBA", "BAAB")
CPU_MODE = "cpu-free"
CANDIDATE_MODE = "all-linear-layers-gdn-out-g32-v2-free"
PINNED_SUMMARY_SHA256 = (
    "3695081d1a5fc1bd54fc041a91596b46fbf0d7d7d2d4befd9bce0efc1b1b9e0d"
)
RECEIPT_KEYS = (
    "cpu_teacher128",
    "candidate_teacher128",
    "cpu_free128",
    "candidate_free128",
)
RECEIPT_IDENTITY = {
    "cpu_teacher128": (
        "apxinf-qwen35-metal-w8-linear-layer-cpu-teacher-v1",
        "linear_layer_cpu_teacher",
    ),
    "candidate_teacher128": (
        "apxinf-qwen35-metal-w8-all-linear-layers-gdn-out-g32-v2-teacher-gate-v1",
        "metal_w8_all_linear_layers_gdn_out_g32_v2_teacher_forced",
    ),
    "cpu_free128": (
        "apxinf-qwen35-metal-w8-linear-layer-cpu-free-run-v1",
        "linear_layer_cpu_free_run",
    ),
    "candidate_free128": (
        "apxinf-qwen35-metal-w8-all-linear-layers-gdn-out-g32-v2-free-run-gate-v1",
        "metal_w8_all_linear_layers_gdn_out_g32_v2_free_run",
    ),
}
SUMMARY_FORMAT = (
    "apxinf-qwen35-metal-w8-all-linear-layers-precision-v2-real-gate-summary-v1"
)
QUIET_SAMPLE_COUNT = 5
QUIET_SAMPLE_INTERVAL_SECONDS = 0.5
QUIET_MAX_PROCESS_CPU_PERCENT = 5.0
QUIET_MAX_LOAD_PER_LOGICAL_CPU = 0.50
RUN_TIMEOUT_SECONDS = 600
RUN_RSS_LIMIT_BYTES = 6 * 1024 * 1024 * 1024
RUN_STREAM_LIMIT_BYTES = 4 * 1024 * 1024
RUN_QUIET_SAMPLE_INTERVAL_SECONDS = 1.0
MINIMUM_MEDIAN_SPEEDUP = 1.10
MAXIMUM_TTFT_RATIO = 1.10
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY = REPO_ROOT / (
    "crates/apxinf-metal/evidence/next-hotspot/"
    "qwen35-all-linear-layers-precision-v2-real-gate-summary-v1-20260824.json"
)


class HarnessError(RuntimeError):
    """Raised when formal measurement must fail closed."""


def terminate_process_group(process: subprocess.Popen) -> None:
    """Terminate and reap a private child process group."""

    try:
        os.killpg(process.pid, signal.SIGTERM)
    except (OSError, ProcessLookupError):
        pass
    try:
        process.wait(timeout=0.25)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except (OSError, ProcessLookupError):
        pass
    try:
        process.wait(timeout=1.0)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=1.0)


def process_group_rss_bytes(process_group: int) -> int:
    """Return summed RSS for one private process group using the fixed ps tool."""

    try:
        completed = subprocess.run(
            ["/bin/ps", "-axo", "pid=,pgid=,rss="],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=1.0,
            env={"PATH": "/usr/bin:/bin:/usr/sbin:/sbin", "LANG": "C", "LC_ALL": "C"},
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise HarnessError("process-group RSS probe failed") from error
    if completed.returncode != 0:
        raise HarnessError("process-group RSS probe failed")
    total_kib = 0
    for line in completed.stdout.decode("ascii", errors="strict").splitlines():
        fields = line.split()
        if len(fields) != 3 or not all(field.isdigit() for field in fields):
            raise HarnessError("process-group RSS probe returned malformed output")
        if int(fields[1]) == process_group:
            total_kib += int(fields[2])
    return total_kib * 1024


def run_supervised(
    argv: list[str],
    *,
    cwd: Path,
    environment: dict[str, str],
    timeout_seconds: float = RUN_TIMEOUT_SECONDS,
    rss_limit_bytes: int = RUN_RSS_LIMIT_BYTES,
    stream_limit_bytes: int = RUN_STREAM_LIMIT_BYTES,
    rss_probe=process_group_rss_bytes,
    quiet_sample_probe=None,
    quiet_allowed_pids: set[int] | None = None,
    quiet_baseline_swap_used_bytes: int | None = None,
    quiet_sample_interval_seconds: float = RUN_QUIET_SAMPLE_INTERVAL_SECONDS,
) -> dict:
    """Run one shell-free command with bounded time, group RSS, and streams."""

    if (
        not isinstance(argv, list)
        or not argv
        or not all(isinstance(item, str) and item for item in argv)
        or not Path(argv[0]).is_absolute()
    ):
        raise HarnessError("supervised argv must start with an absolute executable")
    if (
        not _positive_number(timeout_seconds)
        or rss_limit_bytes <= 0
        or stream_limit_bytes <= 0
        or not _positive_number(quiet_sample_interval_seconds)
    ):
        raise HarnessError("supervisor limits must be positive")
    if quiet_sample_probe is not None and (
        type(quiet_baseline_swap_used_bytes) is not int
        or quiet_baseline_swap_used_bytes < 0
    ):
        raise HarnessError("runtime quiet monitor requires a valid swap baseline")
    try:
        process = subprocess.Popen(
            argv,
            cwd=str(cwd),
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
    except OSError as error:
        raise HarnessError(f"supervised command could not start: {argv[0]}") from error
    stdout = bytearray()
    stderr = bytearray()
    streams = selectors.DefaultSelector()
    streams.register(process.stdout, selectors.EVENT_READ, stdout)
    streams.register(process.stderr, selectors.EVENT_READ, stderr)
    deadline = time.monotonic() + timeout_seconds
    termination_reason = None
    peak_group_rss_bytes = 0
    quiet_samples = []
    quiet_contamination = []
    next_quiet_sample = time.monotonic()
    try:
        while streams.get_map():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                termination_reason = "timeout"
                terminate_process_group(process)
                break
            if quiet_sample_probe is not None and time.monotonic() >= next_quiet_sample:
                try:
                    quiet_sample = quiet_sample_probe(process.pid)
                    quiet_samples.append(quiet_sample)
                    quiet_result = evaluate_quiet_sample(
                        quiet_sample,
                        allowed_pids=quiet_allowed_pids or set(),
                        owned_process_group=process.pid,
                        baseline_swap_used_bytes=quiet_baseline_swap_used_bytes,
                        enforce_load_threshold=False,
                    )
                except Exception:
                    termination_reason = "quiet_probe_failed"
                    terminate_process_group(process)
                    break
                if quiet_result["passed"] is not True:
                    quiet_contamination = quiet_result["problems"]
                    termination_reason = "host_contamination"
                    terminate_process_group(process)
                    break
                next_quiet_sample += quiet_sample_interval_seconds
            try:
                group_rss = rss_probe(process.pid)
            except Exception:
                termination_reason = "rss_probe_failed"
                terminate_process_group(process)
                break
            if type(group_rss) is not int or group_rss < 0:
                termination_reason = "rss_probe_failed"
                terminate_process_group(process)
                break
            peak_group_rss_bytes = max(peak_group_rss_bytes, group_rss)
            if group_rss >= rss_limit_bytes:
                termination_reason = "rss_limit"
                terminate_process_group(process)
                break
            for key, _events in streams.select(timeout=min(0.05, remaining)):
                chunk = os.read(key.fd, 64 * 1024)
                if not chunk:
                    streams.unregister(key.fileobj)
                    key.fileobj.close()
                    continue
                key.data.extend(chunk)
                if len(key.data) > stream_limit_bytes:
                    termination_reason = (
                        "stdout_limit" if key.data is stdout else "stderr_limit"
                    )
                    terminate_process_group(process)
                    break
            if termination_reason is not None:
                break
        if termination_reason is None:
            remaining = max(0.0, deadline - time.monotonic())
            try:
                process.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                termination_reason = "timeout"
                terminate_process_group(process)
    except BaseException:
        if process.poll() is None:
            terminate_process_group(process)
        raise
    finally:
        streams.close()
        for stream in (process.stdout, process.stderr):
            if stream is not None and not stream.closed:
                stream.close()
    return {
        "argv": list(argv),
        "returncode": process.returncode,
        "timed_out": termination_reason == "timeout",
        "termination_reason": termination_reason,
        "peak_group_rss_bytes": peak_group_rss_bytes,
        "rss_limit_bytes": rss_limit_bytes,
        "stdout": bytes(stdout),
        "stderr": bytes(stderr),
        "owned_process_group": process.pid,
        "quiet_samples": quiet_samples,
        "quiet_contamination": quiet_contamination,
    }


def parse_time_l(payload: bytes) -> tuple[int, int]:
    """Parse unique macOS ``/usr/bin/time -l`` RSS and child swap counters."""

    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise HarnessError("time -l output is not UTF-8") from error
    rss = re.findall(
        r"^\s*([0-9]+)\s+maximum resident set size\s*$", text, re.MULTILINE
    )
    swaps = re.findall(r"^\s*([0-9]+)\s+swaps\s*$", text, re.MULTILINE)
    if len(rss) != 1 or len(swaps) != 1:
        raise HarnessError("time -l output must contain unique RSS and swap counters")
    return int(rss[0]), int(swaps[0])


def validate_supervised_result(result: object) -> dict:
    """Admit one completed child only when every resource boundary passed."""

    if not isinstance(result, dict):
        raise HarnessError("supervised result is invalid")
    if result.get("termination_reason") is not None:
        raise HarnessError(
            "supervised run was terminated: " + str(result["termination_reason"])
        )
    if result.get("timed_out") is not False or result.get("returncode") != 0:
        raise HarnessError("supervised run did not exit successfully")
    peak = result.get("peak_group_rss_bytes")
    limit = result.get("rss_limit_bytes")
    if (
        type(peak) is not int
        or type(limit) is not int
        or peak <= 0
        or limit <= 0
        or peak >= limit
    ):
        raise HarnessError("supervised run did not remain below the group RSS limit")
    stderr = result.get("stderr")
    if not isinstance(stderr, bytes):
        raise HarnessError("supervised stderr evidence is invalid")
    time_l_rss, child_swaps = parse_time_l(stderr)
    if child_swaps != 0:
        raise HarnessError("formal run requires zero child swaps")
    if time_l_rss <= 0 or time_l_rss >= limit:
        raise HarnessError("time -l RSS did not remain below the RSS limit")
    return {"time_l_max_rss_bytes": time_l_rss, "child_swaps": child_swaps}


def _positive_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value > 0
    )


def evaluate_quiet_sample(
    sample: object,
    *,
    allowed_pids: set[int],
    owned_process_group: int | None,
    baseline_swap_used_bytes: int,
    enforce_load_threshold: bool,
) -> dict:
    """Validate one host sample, optionally enforcing the global load gate."""

    problems = []
    offenders = []
    swap_values = []
    throttled_values = []
    loads = []
    max_external_cpu = 0.0
    if not isinstance(sample, dict):
        return {
            "passed": False,
            "problems": ["run quiet sample is invalid"],
            "sample": sample,
        }
    for sample_index, sample in enumerate([sample]):
        cpus = sample.get("logical_cpus")
        load = sample.get("load_1m")
        throttled = sample.get("pages_throttled")
        swap = sample.get("swap_used_bytes")
        processes = sample.get("processes")
        if type(cpus) is not int or cpus <= 0:
            problems.append(f"run quiet sample {sample_index} has invalid CPU count")
        if (
            not isinstance(load, (int, float))
            or isinstance(load, bool)
            or not math.isfinite(load)
            or load < 0
        ):
            problems.append(f"run quiet sample {sample_index} has invalid load")
        else:
            loads.append(float(load))
            if (
                enforce_load_threshold
                and type(cpus) is int
                and cpus > 0
                and load > (cpus * QUIET_MAX_LOAD_PER_LOGICAL_CPU)
            ):
                problems.append("load exceeded the quiet-host threshold")
        if type(throttled) is not int or throttled < 0:
            problems.append(
                f"run quiet sample {sample_index} has invalid throttled pages"
            )
        else:
            throttled_values.append(throttled)
            if throttled != 0:
                problems.append("memory_pressure Pages throttled must remain zero")
        if type(swap) is not int or swap < 0:
            problems.append(f"run quiet sample {sample_index} has invalid swap usage")
        else:
            swap_values.append(swap)
            if swap != baseline_swap_used_bytes:
                problems.append("system swap usage changed during formal measurement")
        if not isinstance(processes, list):
            problems.append(f"run quiet sample {sample_index} process list is invalid")
            continue
        for process in processes:
            if not isinstance(process, dict):
                problems.append(
                    f"run quiet sample {sample_index} contains an invalid process"
                )
                continue
            pid = process.get("pid")
            pgid = process.get("pgid")
            cpu = process.get("cpu_percent")
            command = process.get("command")
            if (
                type(pid) is not int
                or pid <= 0
                or (pgid is not None and (type(pgid) is not int or pgid <= 0))
                or not isinstance(cpu, (int, float))
                or isinstance(cpu, bool)
                or not math.isfinite(cpu)
                or cpu < 0
                or not isinstance(command, str)
                or not command
            ):
                problems.append(
                    f"run quiet sample {sample_index} contains an invalid process"
                )
                continue
            if pid in allowed_pids or (
                owned_process_group is not None and pgid == owned_process_group
            ):
                continue
            max_external_cpu = max(max_external_cpu, float(cpu))
            if cpu > QUIET_MAX_PROCESS_CPU_PERCENT:
                offenders.append(
                    {
                        "sample_index": sample_index,
                        "pid": pid,
                        "pgid": pgid,
                        "cpu_percent": cpu,
                        "command": command,
                    }
                )
    if offenders:
        problems.append("non-allowlisted process exceeded 5% CPU")
    problems = list(dict.fromkeys(problems))
    offenders.sort(
        key=lambda item: (-item["cpu_percent"], item["sample_index"], item["pid"])
    )
    return {
        "passed": not problems,
        "problems": problems,
        "sample_count": 1,
        "max_external_cpu_percent": max_external_cpu,
        "max_load_1m": max(loads) if loads else None,
        "pages_throttled_observed": throttled_values,
        "swap_used_bytes_observed": swap_values,
        "swap_drift_bytes": max(
            (abs(value - baseline_swap_used_bytes) for value in swap_values),
            default=0,
        ),
        "offenders": offenders,
        "sample": sample,
    }


def evaluate_run_quiet_custody(
    *,
    start_sample: object,
    online_samples: object,
    end_sample: object,
    allowed_pids: set[int],
    owned_process_group: int | None,
    baseline_swap_used_bytes: int,
) -> dict:
    """Require strict boundaries plus at least one owned-run online sample."""

    if not isinstance(online_samples, list) or not online_samples:
        return {
            "passed": False,
            "problems": ["run quiet custody requires at least one online sample"],
            "sample_count": 2,
            "samples": {
                "start": start_sample,
                "online": online_samples,
                "end": end_sample,
            },
        }
    evaluations = [
        evaluate_quiet_sample(
            start_sample,
            allowed_pids=allowed_pids,
            owned_process_group=None,
            baseline_swap_used_bytes=baseline_swap_used_bytes,
            enforce_load_threshold=False,
        ),
        *[
            evaluate_quiet_sample(
                sample,
                allowed_pids=allowed_pids,
                owned_process_group=owned_process_group,
                baseline_swap_used_bytes=baseline_swap_used_bytes,
                enforce_load_threshold=False,
            )
            for sample in online_samples
        ],
        evaluate_quiet_sample(
            end_sample,
            allowed_pids=allowed_pids,
            owned_process_group=None,
            baseline_swap_used_bytes=baseline_swap_used_bytes,
            enforce_load_threshold=False,
        ),
    ]
    problems = list(
        dict.fromkeys(
            problem for evaluation in evaluations for problem in evaluation["problems"]
        )
    )
    loads = [
        evaluation["max_load_1m"]
        for evaluation in evaluations
        if evaluation.get("max_load_1m") is not None
    ]
    return {
        "passed": not problems,
        "problems": problems,
        "sample_count": len(evaluations),
        "online_sample_count": len(online_samples),
        "max_external_cpu_percent": max(
            evaluation.get("max_external_cpu_percent", 0.0)
            for evaluation in evaluations
        ),
        "max_load_1m": max(loads) if loads else None,
        "pages_throttled_observed": [
            value
            for evaluation in evaluations
            for value in evaluation.get("pages_throttled_observed", [])
        ],
        "swap_used_bytes_observed": [
            value
            for evaluation in evaluations
            for value in evaluation.get("swap_used_bytes_observed", [])
        ],
        "swap_drift_bytes": max(
            evaluation.get("swap_drift_bytes", 0) for evaluation in evaluations
        ),
        "offenders": [
            offender
            for evaluation in evaluations
            for offender in evaluation.get("offenders", [])
        ],
        "samples": {
            "start": start_sample,
            "online": online_samples,
            "end": end_sample,
        },
    }


def canonical_json_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _all_boolean_leaves_true(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if not isinstance(value, dict) or not value:
        return False
    return all(_all_boolean_leaves_true(child) for child in value.values())


def validate_run_receipt(
    path: Path,
    *,
    variant: str,
    frozen_identity: dict,
    expected_token_ids: list[int],
    expected_cpu_receipt: dict | None = None,
    expected_path_checks: dict | None = None,
    expected_ledger: dict | None = None,
) -> dict:
    """Validate one independently-created CPU or all18 free-run receipt."""

    if variant not in {"A", "B"}:
        raise HarnessError("run receipt variant must be A or B")
    if (
        not isinstance(expected_token_ids, list)
        or len(expected_token_ids) != 128
        or any(type(token) is not int or token < 0 for token in expected_token_ids)
    ):
        raise HarnessError("formal trajectory must contain exactly 128 token IDs")
    record = direct_file_record(path, label=f"variant {variant} run receipt")
    receipt = load_json(path, label=f"variant {variant} run receipt")
    expected_key = "cpu_free128" if variant == "A" else "candidate_free128"
    expected_format, expected_mode = RECEIPT_IDENTITY[expected_key]
    if (
        receipt.get("format") != expected_format
        or receipt.get("mode") != expected_mode
        or receipt.get("passed") is not True
        or receipt.get("generated_tokens") != 128
    ):
        raise HarnessError(f"variant {variant} run receipt identity/status is invalid")
    if receipt.get("identity") != frozen_identity:
        raise HarnessError(f"variant {variant} run receipt custody identity drifted")
    timing = receipt.get("timing")
    if (
        not isinstance(timing, dict)
        or not _positive_number(timing.get("decode_mean_ms"))
        or not _positive_number(timing.get("prefill_ms"))
    ):
        raise HarnessError(f"variant {variant} run timing is invalid")
    if variant == "A":
        if receipt.get("generated_token_ids") != expected_token_ids:
            raise HarnessError("CPU formal run trajectory drifted")
        if receipt.get("path_receipt") is not None:
            raise HarnessError("CPU formal run unexpectedly reported a Metal path")
        path_valid = True
        ledger_valid = True
    else:
        if receipt.get("cpu_free_receipt") != expected_cpu_receipt:
            raise HarnessError("candidate input CPU receipt identity drifted")
        if (
            receipt.get("cpu_generated_token_ids") != expected_token_ids
            or receipt.get("metal_w8_all_linear_layers_generated_token_ids")
            != expected_token_ids
        ):
            raise HarnessError("candidate formal run trajectory drifted")
        if receipt.get("mismatches") != [] or receipt.get("first_mismatch") is not None:
            raise HarnessError("candidate formal run reported a trajectory mismatch")
        path_valid = receipt.get("path_checks") == expected_path_checks and (
            _all_boolean_leaves_true(receipt.get("path_checks"))
        )
        if not path_valid:
            raise HarnessError("candidate formal run execution path drifted")
        ledger_valid = receipt.get("aggregate_buffer_ledger") == expected_ledger
        if not ledger_valid:
            raise HarnessError("candidate formal run Metal ledger drifted")
    return {
        "variant": variant,
        "decode_mean_ms": float(timing["decode_mean_ms"]),
        "ttft_ms": float(timing["prefill_ms"]),
        "trajectory_sha256": canonical_json_sha256(expected_token_ids),
        "path_valid": path_valid,
        "ledger_valid": ledger_valid,
        "custody_sha256": canonical_json_sha256(frozen_identity),
        "receipt": record,
    }


def publish_file_no_replace(staging: Path, destination: Path) -> None:
    """Atomically expose one direct file without replacing an existing path."""

    direct_file_record(staging, label="staged formal output")
    try:
        os.link(staging, destination, follow_symlinks=False)
    except FileExistsError as error:
        raise HarnessError(f"formal output already exists: {destination}") from error
    except OSError as error:
        raise HarnessError(
            f"formal output could not be published: {destination}"
        ) from error
    try:
        staging.unlink()
    except OSError as error:
        raise HarnessError(
            "published formal output could not drop its staging link"
        ) from error
    direct_file_record(destination, label="published formal output")


def atomic_write_json_no_replace(path: Path, value: object) -> None:
    """Write canonical JSON through a same-directory staging link."""

    payload = (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    staging = path.with_name("." + path.name + ".staging-" + secrets.token_hex(8))
    try:
        descriptor = os.open(staging, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        publish_file_no_replace(staging, path)
    except BaseException:
        try:
            staging.unlink()
        except FileNotFoundError:
            pass
        raise


def cleanup_staging_best_effort(staging: Path | None) -> str | None:
    """Remove only this harness-owned staging file without masking an error."""

    if staging is None:
        return None
    try:
        entry = staging.lstat()
    except FileNotFoundError:
        return None
    except OSError as error:
        return str(error)
    if not stat.S_ISREG(entry.st_mode) or stat.S_ISLNK(entry.st_mode):
        return "staging path is no longer a direct regular file"
    try:
        staging.unlink()
    except OSError as error:
        return str(error)
    return None


def formal_environment() -> dict[str, str]:
    return {
        "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
        "LANG": "C",
        "LC_ALL": "C",
        "NO_COLOR": "1",
    }


def build_gate_argv(*, frozen: dict, mode: str, output: Path) -> list[str]:
    custody = frozen["summary"]["custody"]
    argv = [
        "/usr/bin/time",
        "-l",
        custody["binary"]["path"],
        "--model-dir",
        custody["model_dir"]["path"],
        "--source-lock",
        custody["source_lock"]["path"],
        "--mode",
        mode,
    ]
    if mode == CANDIDATE_MODE:
        argv.extend(
            [
                "--input-receipt",
                frozen["receipt_records"]["cpu_free128"]["path"],
            ]
        )
    elif mode != CPU_MODE:
        raise HarnessError("formal schedule requested an unapproved gate mode")
    argv.extend(["--output", str(output)])
    return argv


def command_evidence(result: dict) -> dict:
    stdout = result.get("stdout", b"")
    stderr = result.get("stderr", b"")
    return {
        "argv": result.get("argv"),
        "returncode": result.get("returncode"),
        "termination_reason": result.get("termination_reason"),
        "stdout_size_bytes": len(stdout),
        "stdout_sha256": hashlib.sha256(stdout).hexdigest(),
        "stderr_size_bytes": len(stderr),
        "stderr_sha256": hashlib.sha256(stderr).hexdigest(),
    }


def run_system_probe(argv: list[str]) -> bytes:
    try:
        completed = subprocess.run(
            argv,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=3.0,
            env=formal_environment(),
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise HarnessError(f"host probe failed: {argv[0]}") from error
    if (
        completed.returncode != 0
        or len(completed.stdout) > 1024 * 1024
        or len(completed.stderr) > 1024 * 1024
    ):
        raise HarnessError(f"host probe failed: {argv[0]}")
    return completed.stdout


def parse_swap_usage(payload: bytes) -> int:
    try:
        text = payload.decode("ascii")
    except UnicodeDecodeError as error:
        raise HarnessError("vm.swapusage output is not ASCII") from error
    match = re.search(r"\bused\s*=\s*([0-9]+(?:\.[0-9]+)?)([KMGTP])\b", text)
    if match is None:
        raise HarnessError("vm.swapusage output is invalid")
    multiplier = 1024 ** {"K": 1, "M": 2, "G": 3, "T": 4, "P": 5}[match.group(2)]
    return int(round(float(match.group(1)) * multiplier))


def system_swap_used_bytes() -> int:
    return parse_swap_usage(
        run_system_probe(["/usr/sbin/sysctl", "-n", "vm.swapusage"])
    )


def capture_quiet_host_sample(_owned_process_group: int | None = None) -> dict:
    memory = run_system_probe(["/usr/bin/memory_pressure"]).decode(
        "ascii", errors="strict"
    )
    throttled_matches = re.findall(
        r"^Pages throttled:\s*([0-9]+)\s*$", memory, re.MULTILINE
    )
    if len(throttled_matches) != 1:
        raise HarnessError("memory_pressure lacks a unique Pages throttled counter")
    cpu_text = (
        run_system_probe(["/usr/sbin/sysctl", "-n", "hw.ncpu"])
        .decode("ascii", errors="strict")
        .strip()
    )
    load_text = (
        run_system_probe(["/usr/sbin/sysctl", "-n", "vm.loadavg"])
        .decode("ascii", errors="strict")
        .strip()
    )
    if not cpu_text.isdigit():
        raise HarnessError("hw.ncpu output is invalid")
    load_match = re.fullmatch(
        r"\{\s*([0-9]+(?:\.[0-9]+)?)\s+[0-9.]+\s+[0-9.]+\s*\}",
        load_text,
    )
    if load_match is None:
        raise HarnessError("vm.loadavg output is invalid")
    process_text = run_system_probe(
        ["/bin/ps", "-A", "-o", "pid=,pgid=,pcpu=,comm="]
    ).decode("utf-8", errors="strict")
    processes = []
    for line in process_text.splitlines():
        match = re.match(
            r"^\s*([0-9]+)\s+([0-9]+)\s+([0-9]+(?:\.[0-9]+)?)\s+(.+?)\s*$",
            line,
        )
        if match is None:
            if line.strip():
                raise HarnessError("ps process sample is malformed")
            continue
        processes.append(
            {
                "pid": int(match.group(1)),
                "pgid": int(match.group(2)),
                "cpu_percent": float(match.group(3)),
                "command": match.group(4),
            }
        )
    return {
        "logical_cpus": int(cpu_text),
        "load_1m": float(load_match.group(1)),
        "pages_throttled": int(throttled_matches[0]),
        "swap_used_bytes": system_swap_used_bytes(),
        "processes": processes,
    }


def quiet_host_preflight() -> dict:
    samples = []
    for index in range(QUIET_SAMPLE_COUNT):
        samples.append(capture_quiet_host_sample())
        if index + 1 < QUIET_SAMPLE_COUNT:
            time.sleep(QUIET_SAMPLE_INTERVAL_SECONDS)
    return evaluate_quiet_host_samples(samples, allowed_pids={os.getpid()})


def execute_campaign(
    *,
    frozen: dict,
    repo_root: Path,
    output_dir: Path,
    quiet_probe,
    run_quiet_sample_probe=capture_quiet_host_sample,
    command_runner=run_supervised,
    swap_probe=None,
    lane=None,
) -> dict:
    """Execute the formal schedule only after a fail-closed quiet-host gate."""

    if output_dir.exists():
        raise HarnessError("formal output directory already exists")
    if swap_probe is None:
        swap_probe = system_swap_used_bytes
    if lane is None:
        try:
            identity = frozen["identity"]
            summary = frozen["summary"]
            receipts = frozen["receipts"]
            cpu_record = frozen["receipt_records"]["cpu_free128"]
            expected_tokens = receipts["cpu_free128"]["generated_token_ids"]
            expected_path_checks = receipts["candidate_free128"]["path_checks"]
            expected_ledger = receipts["candidate_free128"]["aggregate_buffer_ledger"]
            live_custody_start = frozen["live_custody"]
        except (KeyError, TypeError) as error:
            raise HarnessError("frozen formal inputs are incomplete") from error
        custody_validator = validate_live_custody
        schedule_builder = build_schedule
        gate_argv_builder = build_gate_argv
        receipt_validator = None
        report_format = "apxinf-qwen35-all18-formal-benchmark-v1"
        lane_identity = None
    else:
        try:
            prepared = lane.prepare_campaign(frozen)
            identity = prepared["identity"]
            summary = prepared["summary"]
            expected_tokens = prepared["expected_tokens"]
            live_custody_start = prepared["live_custody_start"]
            custody_validator = lane.validate_live_custody
            schedule_builder = lane.build_schedule
            gate_argv_builder = lane.build_gate_argv
            receipt_validator = lane.validate_run_receipt
            report_format = lane.report_format
            lane_identity = lane.report_identity
        except (AttributeError, KeyError, TypeError) as error:
            raise HarnessError("formal lane adapter is incomplete") from error
        cpu_record = None
        expected_path_checks = None
        expected_ledger = None
    if custody_validator(summary) != live_custody_start:
        raise HarnessError("binary/source/model custody drifted before measurement")
    expected_trajectory = canonical_json_sha256(expected_tokens)
    expected_custody = canonical_json_sha256(identity)
    plan = schedule_builder(output_dir)
    quiet = quiet_probe()
    if not isinstance(quiet, dict) or quiet.get("passed") is not True:
        raise HarnessError("quiet-host preflight failed; no formal child was started")
    preflight_swap = quiet.get("final_swap_used_bytes")
    if type(preflight_swap) is not int or preflight_swap < 0:
        raise HarnessError("quiet-host preflight lacks its final swap baseline")
    output_dir.mkdir(mode=0o700)
    blocks = []
    report = None
    active_staging = None
    try:
        schedule_swap_start = swap_probe()
        if type(schedule_swap_start) is not int or schedule_swap_start < 0:
            raise HarnessError("system swap probe returned an invalid value")
        if schedule_swap_start != preflight_swap:
            raise HarnessError("system swap changed after quiet-host preflight")
        for planned_block in plan["blocks"]:
            block_swap_start = swap_probe()
            if block_swap_start != schedule_swap_start:
                raise HarnessError("system swap changed before a formal block")
            block = {
                "index": planned_block["index"],
                "order": planned_block["order"],
                "quiet_host": {"passed": False, "complete": False},
                "system_swap_used_before_bytes": block_swap_start,
                "samples": [],
            }
            blocks.append(block)
            for planned_run in planned_block["runs"]:
                final_path = Path(planned_run["output"])
                staging_path = final_path.with_name(
                    "." + final_path.name + ".staging-" + secrets.token_hex(8)
                )
                active_staging = staging_path
                argv = gate_argv_builder(
                    frozen=frozen,
                    mode=planned_run["mode"],
                    output=staging_path,
                )
                run_start_sample = run_quiet_sample_probe(None)
                start_quiet = evaluate_quiet_sample(
                    run_start_sample,
                    allowed_pids={os.getpid()},
                    owned_process_group=None,
                    baseline_swap_used_bytes=preflight_swap,
                    enforce_load_threshold=False,
                )
                if start_quiet["passed"] is not True:
                    raise HarnessError("; ".join(start_quiet["problems"]))
                result = command_runner(
                    argv,
                    cwd=repo_root,
                    environment=formal_environment(),
                    timeout_seconds=RUN_TIMEOUT_SECONDS,
                    rss_limit_bytes=RUN_RSS_LIMIT_BYTES,
                    stream_limit_bytes=RUN_STREAM_LIMIT_BYTES,
                    quiet_sample_probe=run_quiet_sample_probe,
                    quiet_allowed_pids={os.getpid()},
                    quiet_baseline_swap_used_bytes=preflight_swap,
                    quiet_sample_interval_seconds=RUN_QUIET_SAMPLE_INTERVAL_SECONDS,
                )
                run_end_sample = run_quiet_sample_probe(None)
                run_quiet = evaluate_run_quiet_custody(
                    start_sample=run_start_sample,
                    online_samples=result.get("quiet_samples", []),
                    end_sample=run_end_sample,
                    allowed_pids={os.getpid()},
                    owned_process_group=result.get("owned_process_group"),
                    baseline_swap_used_bytes=preflight_swap,
                )
                block["last_run_quiet_custody"] = run_quiet
                resources = validate_supervised_result(result)
                if receipt_validator is None:
                    sample = validate_run_receipt(
                        staging_path,
                        variant=planned_run["variant"],
                        frozen_identity=identity,
                        expected_token_ids=expected_tokens,
                        expected_cpu_receipt=cpu_record,
                        expected_path_checks=expected_path_checks,
                        expected_ledger=expected_ledger,
                    )
                else:
                    sample = receipt_validator(
                        staging_path,
                        variant=planned_run["variant"],
                        frozen=frozen,
                    )
                publish_file_no_replace(staging_path, final_path)
                active_staging = None
                sample.update(
                    {
                        "index": planned_run["index"],
                        "child_swaps": resources["child_swaps"],
                        "peak_group_rss_bytes": result["peak_group_rss_bytes"],
                        "time_l_max_rss_bytes": resources["time_l_max_rss_bytes"],
                        "receipt": direct_file_record(
                            final_path, label="published formal run receipt"
                        ),
                        "command": command_evidence(result),
                        "quiet_custody": run_quiet,
                    }
                )
                block["samples"].append(sample)
                if run_quiet["passed"] is not True:
                    raise HarnessError("; ".join(run_quiet["problems"]))
                if swap_probe() != schedule_swap_start:
                    raise HarnessError("system swap changed during a formal run")
            block["quiet_host"] = {
                "passed": all(
                    sample["quiet_custody"]["passed"] is True
                    for sample in block["samples"]
                ),
                "complete": len(block["samples"]) == 4,
                "source": "per-run-start-monitor-end-v1",
            }
            block_swap_end = swap_probe()
            block["system_swap_used_after_bytes"] = block_swap_end
            block["system_swap_growth_bytes"] = max(
                0, block_swap_end - block_swap_start
            )
            if block_swap_end != block_swap_start:
                raise HarnessError("system swap changed during a formal block")
        schedule_swap_end = swap_probe()
        if schedule_swap_end != schedule_swap_start:
            raise HarnessError("system swap changed during the formal campaign")
        live_custody_end = custody_validator(summary)
        if live_custody_end != live_custody_start:
            raise HarnessError("binary/source/model custody changed during measurement")
        reduction = reduce_formal_campaign(
            blocks,
            expected_trajectory_sha256=expected_trajectory,
            expected_custody_sha256=expected_custody,
        )
        report = {
            "format": report_format,
            **({"lane_identity": lane_identity} if lane_identity is not None else {}),
            "status": "formal_accepted" if reduction["accepted"] else "rejected",
            "formal_accepted": reduction["accepted"],
            "frozen_summary_sha256": frozen["summary_sha256"],
            "quiet_host_preflight": quiet,
            "system_swap_used_start_bytes": schedule_swap_start,
            "system_swap_used_end_bytes": schedule_swap_end,
            "reduction": reduction,
        }
    except (KeyboardInterrupt, SystemExit) as interruption:
        cleanup_error = cleanup_staging_best_effort(active_staging)
        interrupted_report = {
            "format": report_format,
            **({"lane_identity": lane_identity} if lane_identity is not None else {}),
            "status": "interrupted",
            "formal_accepted": False,
            "frozen_summary_sha256": frozen.get("summary_sha256"),
            "quiet_host_preflight": quiet,
            "error": type(interruption).__name__,
            "cleanup_error": cleanup_error,
            "preserved_blocks": blocks,
        }
        try:
            atomic_write_json_no_replace(
                output_dir / "formal-result.json", interrupted_report
            )
        except BaseException:
            pass
        raise
    except Exception as error:
        cleanup_error = cleanup_staging_best_effort(active_staging)
        report = {
            "format": report_format,
            **({"lane_identity": lane_identity} if lane_identity is not None else {}),
            "status": "failed",
            "formal_accepted": False,
            "frozen_summary_sha256": frozen.get("summary_sha256"),
            "quiet_host_preflight": quiet,
            "error": str(error),
            "cleanup_error": cleanup_error,
            "preserved_blocks": blocks,
        }
    atomic_write_json_no_replace(output_dir / "formal-result.json", report)
    return report


def reduce_formal_campaign(
    blocks: object,
    *,
    expected_trajectory_sha256: str,
    expected_custody_sha256: str,
) -> dict:
    """Reduce the complete 3xABBA + 3xBAAB protocol without dropping samples."""

    problems = []
    contamination = []
    if not isinstance(blocks, list):
        return {
            "accepted": False,
            "problems": ["formal blocks must be an array"],
            "contamination": [],
            "preserved_blocks": blocks,
        }
    if len(blocks) != len(BLOCK_ORDERS):
        problems.append("formal campaign requires exactly six blocks")
    baseline = []
    candidate = []
    same_direction_blocks = 0
    for block_index, expected_order in enumerate(BLOCK_ORDERS):
        if block_index >= len(blocks) or not isinstance(blocks[block_index], dict):
            problems.append(f"formal block {block_index} is missing or invalid")
            continue
        block = blocks[block_index]
        if block.get("index") != block_index or block.get("order") != expected_order:
            problems.append(f"formal block {block_index} order/identity drifted")
        quiet = block.get("quiet_host")
        if not isinstance(quiet, dict) or quiet.get("passed") is not True:
            contamination.append(f"formal block {block_index} was not quiet")
        if block.get("system_swap_growth_bytes") != 0:
            contamination.append(f"formal block {block_index} observed swap growth")
        samples = block.get("samples")
        if not isinstance(samples, list) or len(samples) != 4:
            problems.append(f"formal block {block_index} must retain four samples")
            continue
        observed_order = "".join(
            sample.get("variant", "?") if isinstance(sample, dict) else "?"
            for sample in samples
        )
        if observed_order != expected_order:
            problems.append(f"formal block {block_index} sample order drifted")
        block_a = []
        block_b = []
        for sample_index, sample in enumerate(samples):
            label = f"formal block {block_index} sample {sample_index}"
            if not isinstance(sample, dict):
                problems.append(label + " is invalid")
                continue
            variant = sample.get("variant")
            if sample.get("index") != sample_index or variant not in {"A", "B"}:
                problems.append(label + " identity drifted")
                continue
            decode_mean_ms = sample.get("decode_mean_ms")
            ttft_ms = sample.get("ttft_ms")
            if not _positive_number(decode_mean_ms):
                problems.append(label + " decode_mean_ms is invalid")
                continue
            if not _positive_number(ttft_ms):
                problems.append(label + " ttft_ms is invalid")
                continue
            if sample.get("trajectory_sha256") != expected_trajectory_sha256:
                problems.append(label + " trajectory drifted")
            if sample.get("path_valid") is not True:
                problems.append(label + " execution path is invalid")
            if sample.get("ledger_valid") is not True:
                problems.append(label + " Metal ledger is invalid")
            if sample.get("custody_sha256") != expected_custody_sha256:
                problems.append(label + " custody drifted")
            if sample.get("child_swaps") != 0:
                contamination.append(label + " swapped")
            run_quiet = sample.get("quiet_custody")
            throttled_observed = (
                run_quiet.get("pages_throttled_observed")
                if isinstance(run_quiet, dict)
                else None
            )
            max_external_cpu = (
                run_quiet.get("max_external_cpu_percent")
                if isinstance(run_quiet, dict)
                else None
            )
            max_load = (
                run_quiet.get("max_load_1m") if isinstance(run_quiet, dict) else None
            )
            if (
                not isinstance(run_quiet, dict)
                or run_quiet.get("passed") is not True
                or type(run_quiet.get("sample_count")) is not int
                or run_quiet["sample_count"] < 3
                or type(run_quiet.get("online_sample_count")) is not int
                or run_quiet["online_sample_count"] < 1
                or type(run_quiet.get("swap_drift_bytes")) is not int
                or run_quiet["swap_drift_bytes"] != 0
                or not isinstance(throttled_observed, list)
                or not throttled_observed
                or any(
                    type(value) is not int or value != 0 for value in throttled_observed
                )
                or not isinstance(max_external_cpu, (int, float))
                or isinstance(max_external_cpu, bool)
                or not math.isfinite(max_external_cpu)
                or max_external_cpu < 0
                or not isinstance(max_load, (int, float))
                or isinstance(max_load, bool)
                or not math.isfinite(max_load)
                or max_load < 0
            ):
                contamination.append(label + " lacks complete quiet custody")
            peak = sample.get("peak_group_rss_bytes")
            if type(peak) is not int or peak <= 0 or peak >= RUN_RSS_LIMIT_BYTES:
                problems.append(label + " exceeded the group RSS contract")
            throughput_tps = 1000.0 / float(decode_mean_ms)
            if not math.isfinite(throughput_tps) or throughput_tps <= 0:
                problems.append(
                    label + " derived throughput is non-finite or non-positive"
                )
                continue
            measured = {
                **sample,
                "throughput_tps": throughput_tps,
            }
            if variant == "A":
                baseline.append(measured)
                block_a.append(measured)
            else:
                candidate.append(measured)
                block_b.append(measured)
        if len(block_a) == 2 and len(block_b) == 2:
            block_a_tps = statistics.median(
                sample["throughput_tps"] for sample in block_a
            )
            block_b_tps = statistics.median(
                sample["throughput_tps"] for sample in block_b
            )
            if block_b_tps > block_a_tps:
                same_direction_blocks += 1

    median_speedup = None
    ttft_ratio = None
    if len(baseline) == 12 and len(candidate) == 12:
        median_speedup = statistics.median(
            sample["throughput_tps"] for sample in candidate
        ) / statistics.median(sample["throughput_tps"] for sample in baseline)
        ttft_ratio = statistics.median(
            sample["ttft_ms"] for sample in candidate
        ) / statistics.median(sample["ttft_ms"] for sample in baseline)
        if not math.isfinite(median_speedup) or median_speedup <= 0:
            problems.append("candidate median speedup is non-finite or non-positive")
        elif median_speedup + 1e-12 < MINIMUM_MEDIAN_SPEEDUP:
            problems.append("candidate median speedup is below 1.10x")
        if not math.isfinite(ttft_ratio) or ttft_ratio <= 0:
            problems.append("candidate TTFT ratio is non-finite or non-positive")
        elif ttft_ratio - 1e-12 > MAXIMUM_TTFT_RATIO:
            problems.append("candidate TTFT regressed by more than 10%")
    else:
        problems.append(
            "formal campaign requires exactly twelve A and twelve B samples"
        )
    if same_direction_blocks != 6:
        problems.append("candidate must win all six block medians")
    problems = list(dict.fromkeys(problems))
    contamination = list(dict.fromkeys(contamination))
    return {
        "accepted": not problems and not contamination,
        "replacement_required": bool(contamination),
        "problems": problems,
        "contamination": contamination,
        "sample_count": len(baseline) + len(candidate),
        "baseline_sample_count": len(baseline),
        "candidate_sample_count": len(candidate),
        "same_direction_blocks": same_direction_blocks,
        "median_speedup": median_speedup,
        "ttft_ratio": ttft_ratio,
        "preserved_blocks": blocks,
    }


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def direct_file_record(path: Path, *, label: str) -> dict:
    try:
        entry = path.lstat()
    except OSError as error:
        raise HarnessError(f"{label} is unavailable: {error}") from error
    if stat.S_ISLNK(entry.st_mode) or not stat.S_ISREG(entry.st_mode):
        raise HarnessError(f"{label} must be a direct regular file")
    if entry.st_nlink != 1:
        raise HarnessError(f"{label} must have exactly one hard link")
    return {"path": str(path), "size": entry.st_size, "sha256": hash_file(path)}


def load_json(path: Path, *, label: str) -> dict:
    direct_file_record(path, label=label)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as error:
        raise HarnessError(f"{label} must be valid UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise HarnessError(f"{label} must be a JSON object")
    return value


def resolve_repo_file(repo_root: Path, declared: object, *, label: str) -> Path:
    if not isinstance(declared, str) or not declared or Path(declared).is_absolute():
        raise HarnessError(f"{label} path must be repo-relative")
    root = repo_root.resolve(strict=True)
    candidate = Path(os.path.abspath(os.fspath(root / declared)))
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise HarnessError(f"{label} path escapes the repository") from error
    return candidate


def validate_live_custody(summary: dict) -> dict:
    """Rehash the binary, sources, profile, source lock, and model closure."""

    custody = summary.get("custody")
    if not isinstance(custody, dict):
        raise HarnessError("summary custody is missing")

    def validate_record(label: str, record: object, *, sha_key: str = "sha256") -> dict:
        if not isinstance(record, dict):
            raise HarnessError(f"{label} custody is missing")
        declared_path = record.get("path")
        if not isinstance(declared_path, str) or not Path(declared_path).is_absolute():
            raise HarnessError(f"{label} custody path must be absolute")
        observed = direct_file_record(Path(declared_path), label=f"{label} custody")
        if observed["sha256"] != record.get(sha_key):
            raise HarnessError(f"{label} custody SHA-256 drifted")
        if observed["size"] != record.get("size"):
            raise HarnessError(f"{label} custody size drifted")
        if (
            record.get("direct_regular_file") is not True
            or record.get("single_link") is not True
        ):
            raise HarnessError(f"{label} custody was not archived as a direct file")
        return observed

    observed = {
        "binary": validate_record("binary", custody.get("binary")),
        "sources": {},
    }
    sources = custody.get("sources")
    if not isinstance(sources, dict) or set(sources) != {"gate", "general"}:
        raise HarnessError("source custody must contain exactly gate and general")
    for key in ("gate", "general"):
        observed["sources"][key] = validate_record(f"{key} source", sources.get(key))
    observed["profile"] = validate_record("profile", custody.get("profile"))
    observed["source_lock"] = validate_record(
        "source lock", custody.get("source_lock"), sha_key="file_sha256"
    )

    model = custody.get("model_dir")
    if not isinstance(model, dict):
        raise HarnessError("model custody is missing")
    declared_model = model.get("path")
    if not isinstance(declared_model, str) or not Path(declared_model).is_absolute():
        raise HarnessError("model custody path must be absolute")
    model_path = Path(declared_model)
    try:
        model_entry = model_path.lstat()
    except OSError as error:
        raise HarnessError(f"model custody is unavailable: {error}") from error
    if stat.S_ISLNK(model_entry.st_mode) or not stat.S_ISDIR(model_entry.st_mode):
        raise HarnessError("model custody must be a direct directory")
    artifacts = model.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise HarnessError("model custody artifacts are missing")
    try:
        actual_names = {child.name for child in model_path.iterdir()}
    except OSError as error:
        raise HarnessError(f"model custody cannot be enumerated: {error}") from error
    if actual_names != set(artifacts):
        raise HarnessError("model custody closure drifted")
    observed_artifacts = {}
    for name, record in artifacts.items():
        if Path(name).name != name:
            raise HarnessError("model custody artifact name is invalid")
        artifact_record = dict(record) if isinstance(record, dict) else record
        if not isinstance(artifact_record, dict):
            raise HarnessError(f"model artifact custody is invalid: {name}")
        artifact_record["path"] = str(model_path / name)
        observed_artifacts[name] = validate_record(
            f"model artifact {name}", artifact_record
        )
    if model.get("closure") != "exact-profile-artifacts-plus-safe-cache-v1":
        raise HarnessError("model custody closure contract drifted")
    if model.get("cache_present") is not False:
        raise HarnessError("formal benchmark requires the archived cache-free model")
    observed["model_dir"] = {
        "path": str(model_path),
        "artifacts": observed_artifacts,
        "cache_present": False,
    }
    return observed


def evaluate_quiet_host_samples(samples: object, *, allowed_pids: set[int]) -> dict:
    """Reduce a fixed host sample window; any ambiguous value fails closed."""

    problems = []
    offenders = []
    if not isinstance(samples, list) or len(samples) != QUIET_SAMPLE_COUNT:
        return {
            "passed": False,
            "problems": [f"quiet-host preflight requires {QUIET_SAMPLE_COUNT} samples"],
            "offenders": [],
            "samples": samples,
        }
    swap_values = []
    logical_cpus = None
    for sample_index, sample in enumerate(samples):
        if not isinstance(sample, dict):
            problems.append(f"quiet-host sample {sample_index} is invalid")
            continue
        cpus = sample.get("logical_cpus")
        load = sample.get("load_1m")
        throttled = sample.get("pages_throttled")
        swap = sample.get("swap_used_bytes")
        processes = sample.get("processes")
        if type(cpus) is not int or cpus <= 0:
            problems.append(f"quiet-host sample {sample_index} has invalid CPU count")
        else:
            if logical_cpus is None:
                logical_cpus = cpus
            elif cpus != logical_cpus:
                problems.append("logical CPU count drifted during quiet-host sampling")
        if (
            not isinstance(load, (int, float))
            or isinstance(load, bool)
            or load < 0
            or type(cpus) is not int
            or cpus <= 0
        ):
            problems.append(f"quiet-host sample {sample_index} has invalid load")
        elif not math.isfinite(load):
            problems.append(f"quiet-host sample {sample_index} has invalid load")
        elif load > cpus * QUIET_MAX_LOAD_PER_LOGICAL_CPU:
            problems.append("load exceeded the quiet-host threshold")
        if type(throttled) is not int or throttled != 0:
            problems.append("memory_pressure Pages throttled must remain zero")
        if type(swap) is not int or swap < 0:
            problems.append(f"quiet-host sample {sample_index} has invalid swap usage")
        else:
            swap_values.append(swap)
        if not isinstance(processes, list):
            problems.append(f"quiet-host sample {sample_index} process list is invalid")
            continue
        for process in processes:
            if not isinstance(process, dict):
                problems.append(
                    f"quiet-host sample {sample_index} contains an invalid process"
                )
                continue
            pid = process.get("pid")
            cpu = process.get("cpu_percent")
            command = process.get("command")
            if (
                type(pid) is not int
                or pid <= 0
                or not isinstance(cpu, (int, float))
                or isinstance(cpu, bool)
                or not math.isfinite(cpu)
                or cpu < 0
                or not isinstance(command, str)
                or not command
            ):
                problems.append(
                    f"quiet-host sample {sample_index} contains an invalid process"
                )
                continue
            if pid not in allowed_pids and cpu > QUIET_MAX_PROCESS_CPU_PERCENT:
                offenders.append(
                    {
                        "sample_index": sample_index,
                        "pid": pid,
                        "cpu_percent": cpu,
                        "command": command,
                    }
                )
    if swap_values and len(set(swap_values)) != 1:
        problems.append("system swap usage changed during quiet-host sampling")
    if offenders:
        problems.append("non-allowlisted process exceeded 5% CPU")
    unique_problems = list(dict.fromkeys(problems))
    offenders.sort(
        key=lambda item: (-item["cpu_percent"], item["sample_index"], item["pid"])
    )
    return {
        "passed": not unique_problems,
        "problems": unique_problems,
        "offenders": offenders,
        "samples": samples,
        "maximum_process_cpu_percent": QUIET_MAX_PROCESS_CPU_PERCENT,
        "maximum_load_per_logical_cpu": QUIET_MAX_LOAD_PER_LOGICAL_CPU,
        "final_swap_used_bytes": swap_values[-1] if swap_values else None,
    }


def validate_frozen_inputs(
    summary_path: Path,
    *,
    repo_root: Path,
    expected_summary_sha256: str = PINNED_SUMMARY_SHA256,
) -> dict:
    """Bind formal work to the exact archived all18 correctness summary."""

    summary_record = direct_file_record(summary_path, label="frozen all18 summary")
    observed = summary_record["sha256"]
    if observed != expected_summary_sha256:
        raise HarnessError("frozen all18 summary SHA-256 does not match the pin")
    summary = load_json(summary_path, label="frozen all18 summary")
    if summary.get("format") != SUMMARY_FORMAT:
        raise HarnessError("frozen all18 summary format is not admitted")
    integrity = summary.get("receipt_integrity")
    if not isinstance(integrity, dict):
        raise HarnessError("frozen all18 summary receipt integrity is missing")
    receipts = {}
    receipt_records = {}
    for key in RECEIPT_KEYS:
        record = integrity.get(key)
        if not isinstance(record, dict):
            raise HarnessError(f"frozen receipt record is missing: {key}")
        receipt_path = resolve_repo_file(
            repo_root, record.get("path"), label=f"{key} receipt"
        )
        observed_record = direct_file_record(receipt_path, label=f"{key} receipt")
        if observed_record["sha256"] != record.get("sha256"):
            raise HarnessError(f"{key} receipt SHA-256 does not match the summary")
        if observed_record["size"] != record.get("size"):
            raise HarnessError(f"{key} receipt size does not match the summary")
        receipt = load_json(receipt_path, label=f"{key} receipt")
        expected_format, expected_mode = RECEIPT_IDENTITY[key]
        if (
            receipt.get("format") != expected_format
            or receipt.get("mode") != expected_mode
            or receipt.get("passed") is not True
        ):
            raise HarnessError(f"{key} receipt is not an admitted passing receipt")
        receipts[key] = receipt
        receipt_records[key] = {
            **observed_record,
            "direct_regular_file": True,
            "single_link": True,
        }
    identities = [receipts[key].get("identity") for key in RECEIPT_KEYS]
    if not isinstance(identities[0], dict) or any(
        identity != identities[0] for identity in identities[1:]
    ):
        raise HarnessError("the four receipt identities drift from one another")
    trajectory = summary.get("trajectory_gate")
    gate = summary.get("gate_result")
    ledger = summary.get("aggregate_buffer_ledger")
    if not isinstance(trajectory, dict) or any(
        trajectory.get(key) is not True
        for key in ("all_four_receipts_passed", "all_candidate_path_checks_true")
    ):
        raise HarnessError("archived all18 trajectory/path gate is not passing")
    if not isinstance(gate, dict) or any(
        gate.get(key) is not True
        for key in ("correctness_and_path_gate_passed", "aggregate_ledger_valid")
    ):
        raise HarnessError("archived all18 correctness gate is not passing")
    if (
        not isinstance(ledger, dict)
        or ledger.get("independent_sum_matches_both_candidate_receipts") is not True
    ):
        raise HarnessError("archived all18 ledger gate is not passing")
    live_custody = validate_live_custody(summary)
    return {
        "summary_path": str(summary_path),
        "summary_sha256": observed,
        "summary": summary,
        "receipts": receipts,
        "receipt_records": receipt_records,
        "identity": identities[0],
        "live_custody": live_custody,
    }


def build_schedule(output_dir: Path) -> dict:
    """Return the immutable 24-run paired measurement schedule."""

    blocks = []
    for block_index, order in enumerate(BLOCK_ORDERS):
        runs = []
        for run_index, variant in enumerate(order):
            mode = CPU_MODE if variant == "A" else CANDIDATE_MODE
            output = output_dir / (
                f"block-{block_index:02d}-run-{run_index:02d}-{variant}.json"
            )
            runs.append(
                {
                    "index": run_index,
                    "variant": variant,
                    "mode": mode,
                    "output": str(output),
                }
            )
        blocks.append({"index": block_index, "order": order, "runs": runs})
    return {"block_orders": list(BLOCK_ORDERS), "blocks": blocks}


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    value.add_argument("--output-dir", type=Path)
    actions = value.add_mutually_exclusive_group()
    actions.add_argument("--execute", action="store_true")
    actions.add_argument("--preflight-only", action="store_true")
    return value


def main(argv: list[str] | None = None) -> int:
    arguments = parser().parse_args(argv)
    try:
        if arguments.preflight_only:
            result = quiet_host_preflight()
            print(json.dumps(result, sort_keys=True, separators=(",", ":")))
            return 0 if result["passed"] else 2
        frozen = validate_frozen_inputs(arguments.summary, repo_root=REPO_ROOT)
        output_dir = arguments.output_dir
        if arguments.execute:
            if output_dir is None or not output_dir.is_absolute():
                raise HarnessError("--execute requires an absolute --output-dir")
            report = execute_campaign(
                frozen=frozen,
                repo_root=REPO_ROOT,
                output_dir=output_dir,
                quiet_probe=quiet_host_preflight,
            )
            print(json.dumps(report, sort_keys=True, separators=(",", ":")))
            return 0 if report["formal_accepted"] else 3
        dry_output = output_dir or Path(
            "/private/tmp/apxinf-qwen35-all18-formal-not-started"
        )
        plan = {
            "format": "apxinf-qwen35-all18-formal-plan-v1",
            "execution_started": False,
            "requires_explicit_execute": True,
            "frozen_summary_sha256": frozen["summary_sha256"],
            "schedule": build_schedule(dry_output),
        }
        print(json.dumps(plan, sort_keys=True, separators=(",", ":")))
        return 0
    except HarnessError as error:
        print(
            json.dumps(
                {"formal_accepted": False, "status": "blocked", "error": str(error)},
                sort_keys=True,
                separators=(",", ":"),
            ),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
