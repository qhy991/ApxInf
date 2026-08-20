#!/usr/bin/env python3
"""Run a frozen single-GPU cohort from clean Git worktrees.

The plan is teacher-only. Commands are argv arrays, never shell strings. Every
entry is built and served from its exact commit in a detached worktree, while
the evaluator and scorer are loaded from a separately pinned teacher directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PLAN_SCHEMA = "apxinf.qwen38_27b.cohort_plan.v1"
ROUND_SCHEMA = "apxinf.qwen38_27b.cohort_provenance.v1"
SAFE_NAME = re.compile(r"[A-Za-z0-9_.-]+")
COMMIT = re.compile(r"[0-9a-f]{40}")


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: top-level value must be an object")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def expand_argv(value: Any, variables: dict[str, str], field: str) -> list[str]:
    if not isinstance(value, list) or not value or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise ValueError(f"{field} must be a non-empty argv array")
    try:
        return [item.format_map(variables) for item in value]
    except KeyError as error:
        raise ValueError(f"{field} uses unknown placeholder {error}") from error


def run_logged(argv: list[str], cwd: Path, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("wb") as log:
        completed = subprocess.run(
            argv,
            cwd=cwd,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode:
        raise RuntimeError(
            f"command failed with exit {completed.returncode}; see {log_path}"
        )


def git_output(repository: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repository), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def wait_for_health(url: str, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    last_error = "not attempted"
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=3) as response:
                if response.status == 200:
                    return
                last_error = f"HTTP {response.status}"
        except Exception as error:  # startup deliberately tolerates connection errors
            last_error = f"{type(error).__name__}: {error}"
        time.sleep(1)
    raise TimeoutError(f"service did not become healthy at {url}: {last_error}")


def stop_process(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGINT)
    try:
        process.wait(timeout=30)
        return
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)


def validate_plan(plan: dict[str, Any]) -> None:
    if plan.get("schema") != PLAN_SCHEMA:
        raise ValueError(f"plan.schema must equal {PLAN_SCHEMA}")
    if not isinstance(plan.get("round_id"), str) or not SAFE_NAME.fullmatch(
        plan["round_id"]
    ):
        raise ValueError("round_id must be a filesystem-safe name")
    if plan.get("profile") not in ("public_calibration", "midterm_leaderboard"):
        raise ValueError("profile must be public_calibration or midterm_leaderboard")
    entries = plan.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("entries must be a non-empty array")
    names: set[str] = set()
    controls = 0
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"entries[{index}] must be an object")
        name = entry.get("name")
        if not isinstance(name, str) or not SAFE_NAME.fullmatch(name) or name in names:
            raise ValueError(f"entries[{index}].name must be unique and filesystem-safe")
        names.add(name)
        if entry.get("role") not in ("candidate", "control"):
            raise ValueError(f"entries[{index}].role must be candidate or control")
        controls += int(entry["role"] == "control")
        revision = entry.get("checkout_revision")
        if not isinstance(revision, str) or not COMMIT.fullmatch(revision):
            raise ValueError(f"entries[{index}].checkout_revision must be a full commit SHA")
        for field in ("serve_argv", "runner_argv"):
            if not isinstance(entry.get(field), list) or not entry[field]:
                raise ValueError(f"entries[{index}].{field} must be a non-empty argv array")
        if "build_argv" in entry and not isinstance(entry["build_argv"], list):
            raise ValueError(f"entries[{index}].build_argv must be an argv array")
    if controls != 1:
        raise ValueError("exactly one teacher control entry is required")
    if not any(entry["role"] == "candidate" for entry in entries):
        raise ValueError("at least one candidate entry is required")


def collect_gpu_environment() -> dict[str, Any]:
    fields = (
        "name,uuid,driver_version,pci.bus_id,power.limit,clocks.current.sm,"
        "clocks.current.memory,memory.total"
    )
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                f"--query-gpu={fields}",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
        return {
            "available": completed.returncode == 0,
            "query_fields": fields.split(","),
            "rows": completed.stdout.strip().splitlines(),
            "error": completed.stderr.strip() or None,
        }
    except Exception as error:
        return {"available": False, "error": f"{type(error).__name__}: {error}"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--evaluator-root", type=Path, required=True)
    parser.add_argument("--artifacts-root", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    plan = load_json(args.plan)
    validate_plan(plan)
    evaluator_root = args.evaluator_root.resolve(strict=True)
    contract_path = evaluator_root / "contract-v1.json"
    runner_path = evaluator_root / "run_evaluation.py"
    scorer_path = evaluator_root / "score_submission.py"
    for path in (contract_path, runner_path, scorer_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    resolved_entries: list[dict[str, Any]] = []
    for entry in plan["entries"]:
        repository = Path(entry["repository"]).resolve(strict=True)
        actual = git_output(repository, "rev-parse", f"{entry['checkout_revision']}^{{commit}}")
        if actual != entry["checkout_revision"]:
            raise ValueError(f"{entry['name']}: resolved revision {actual} is not exact")
        resolved_entries.append({**entry, "repository": str(repository)})
    if args.validate_only:
        print(json.dumps({"valid": True, "entries": resolved_entries}, indent=2))
        return 0

    round_dir = args.artifacts_root.resolve() / plan["round_id"]
    round_dir.mkdir(parents=True, exist_ok=False)
    worktrees_dir = round_dir / "worktrees"
    worktrees_dir.mkdir()
    started = datetime.now(timezone.utc).isoformat()
    entry_records: list[dict[str, Any]] = []
    candidate_submissions: list[Path] = []
    control_submissions: list[Path] = []

    for entry in resolved_entries:
        name = entry["name"]
        repository = Path(entry["repository"])
        checkout = worktrees_dir / name
        entry_dir = round_dir / "entries" / name
        entry_dir.mkdir(parents=True)
        subprocess.run(
            [
                "git",
                "-C",
                str(repository),
                "worktree",
                "add",
                "--detach",
                str(checkout),
                entry["checkout_revision"],
            ],
            check=True,
        )
        variables = {
            "checkout": str(checkout),
            "entry_artifacts": str(entry_dir),
            "round_artifacts": str(round_dir),
            "evaluator_root": str(evaluator_root),
            "python": sys.executable,
        }
        service: subprocess.Popen[bytes] | None = None
        try:
            build_argv = entry.get("build_argv", [])
            if build_argv:
                run_logged(
                    expand_argv(build_argv, variables, f"{name}.build_argv"),
                    checkout,
                    entry_dir / "build.log",
                )
            serve_argv = expand_argv(entry["serve_argv"], variables, f"{name}.serve_argv")
            service_log_path = entry_dir / "service.log"
            service_log = service_log_path.open("wb")
            try:
                service = subprocess.Popen(
                    serve_argv,
                    cwd=checkout,
                    stdin=subprocess.DEVNULL,
                    stdout=service_log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                wait_for_health(
                    entry["health_url"], float(entry.get("startup_timeout_s", 300))
                )
                runner_argv = expand_argv(
                    entry["runner_argv"], variables, f"{name}.runner_argv"
                ) + [
                    "--output-dir",
                    str(entry_dir),
                    "--run-id",
                    "evaluation",
                ]
                run_logged(runner_argv, evaluator_root, entry_dir / "runner.log")
            finally:
                if service is not None:
                    stop_process(service)
                service_log.close()

            submission = entry_dir / "evaluation" / "submission.json"
            if not submission.is_file():
                raise FileNotFoundError(f"{name}: evaluator produced no {submission}")
            target = control_submissions if entry["role"] == "control" else candidate_submissions
            target.append(submission)
            entry_records.append(
                {
                    "name": name,
                    "role": entry["role"],
                    "repository": str(repository),
                    "checkout_revision": entry["checkout_revision"],
                    "tree_sha256": hashlib.sha256(
                        git_output(checkout, "ls-tree", "-r", "--full-tree", "HEAD").encode()
                    ).hexdigest(),
                    "submission": str(submission.relative_to(round_dir)),
                    "submission_sha256": sha256_file(submission),
                    "build_log_sha256": sha256_file(entry_dir / "build.log")
                    if (entry_dir / "build.log").is_file()
                    else None,
                    "service_log_sha256": sha256_file(entry_dir / "service.log"),
                    "runner_log_sha256": sha256_file(entry_dir / "runner.log"),
                }
            )
        finally:
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(repository),
                    "worktree",
                    "remove",
                    "--force",
                    str(checkout),
                ],
                check=False,
            )

    snapshot_path = round_dir / "leaderboard-snapshot.json"
    score_argv = [
        sys.executable,
        str(scorer_path),
        "--contract",
        str(contract_path),
        "--profile",
        plan["profile"],
        "--require-official-control",
        "--output",
        str(snapshot_path),
    ]
    for submission in control_submissions:
        score_argv.extend(("--control-submission", str(submission)))
    for submission in candidate_submissions:
        score_argv.extend(("--submission", str(submission)))
    run_logged(score_argv, evaluator_root, round_dir / "scorer.log")

    provenance = {
        "schema": ROUND_SCHEMA,
        "round_id": plan["round_id"],
        "profile": plan["profile"],
        "started_utc": started,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "plan_sha256": sha256_file(args.plan),
        "contract_sha256": sha256_file(contract_path),
        "evaluator_sha256": sha256_file(runner_path),
        "scorer_sha256": sha256_file(scorer_path),
        "snapshot_sha256": sha256_file(snapshot_path),
        "gpu_environment": collect_gpu_environment(),
        "entries": entry_records,
    }
    provenance_path = round_dir / "provenance.json"
    provenance_path.write_bytes(canonical_json(provenance) + b"\n")
    print(
        json.dumps(
            {
                "round_dir": str(round_dir),
                "snapshot": str(snapshot_path),
                "snapshot_sha256": provenance["snapshot_sha256"],
                "provenance": str(provenance_path),
                "provenance_sha256": sha256_file(provenance_path),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
