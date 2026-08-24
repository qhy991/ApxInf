#!/usr/bin/env python3
"""Compile a frozen Hugging Face source lock into a read-only KerSor Mission.

This is intentionally the agent-analysis half of onboarding.  Source resolution
is performed first by ``resolve_hf_source.py``; the Mission can only consume its
bounded source lock and the ApxInf workspace.  Host evaluators independently
validate both that lock and the returned ``port_manifest``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import stat
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from urllib.parse import unquote, urlparse


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from kersor_runtime_lock import (  # noqa: E402
    RuntimeLockError,
    build_runtime_lock,
    read_file_bytes,
    validate_runtime_config_policy,
    write_runtime_lock,
)


HF_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
HF_REVISION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,199}$")
SAFE_RUN_COMPONENT = re.compile(r"[^a-z0-9._-]+")
EXPECTED_HASH_FLAG = "--expected-contract-sha256"
EXPECTED_RUNTIME_HASH_FLAG = "--expected-runtime-config-sha256"


class IntakeError(ValueError):
    """Raised when deterministic intake compilation must fail closed."""


def parse_model_reference(
    reference: str, explicit_revision: str | None
) -> tuple[str, str]:
    """Return ``(repo_id, revision)`` for a canonical HF model reference."""

    raw = reference.strip()
    if not raw:
        raise IntakeError("model reference is empty")

    url_revision: str | None = None
    if "://" in raw:
        parsed = urlparse(raw)
        if parsed.scheme != "https":
            raise IntakeError("Hugging Face URL must use https")
        if (parsed.hostname or "").lower() not in {
            "huggingface.co",
            "www.huggingface.co",
        }:
            raise IntakeError("only canonical huggingface.co model URLs are accepted")
        if parsed.username or parsed.password or parsed.port:
            raise IntakeError(
                "credentials and custom ports are not allowed in model URLs"
            )
        if parsed.query or parsed.fragment or parsed.params:
            raise IntakeError(
                "query strings and fragments are not allowed in model URLs"
            )
        parts = [unquote(part) for part in parsed.path.split("/") if part]
        if len(parts) < 2:
            raise IntakeError("model URL must contain owner/model")
        owner, model = parts[:2]
        if len(parts) > 2:
            if parts[2] != "tree" or len(parts) == 3:
                raise IntakeError(
                    "use a model root URL or /tree/<revision>, not a file URL"
                )
            url_revision = "/".join(parts[3:])
        repo_id = f"{owner}/{model}"
    else:
        parts = raw.split("/")
        if len(parts) != 2:
            raise IntakeError("model id must use owner/model")
        owner, model = parts
        repo_id = raw

    if not HF_COMPONENT.fullmatch(owner) or not HF_COMPONENT.fullmatch(model):
        raise IntakeError("owner and model contain unsupported characters")

    requested = explicit_revision.strip() if explicit_revision else None
    if requested == "":
        raise IntakeError("revision cannot be empty")
    if requested and url_revision and requested != url_revision:
        raise IntakeError("--revision conflicts with the revision embedded in the URL")
    revision = requested or url_revision or "main"
    if (
        not HF_REVISION.fullmatch(revision)
        or "//" in revision
        or any(part in {".", ".."} for part in revision.split("/"))
    ):
        raise IntakeError("revision is not safe")
    return repo_id, revision


def model_slug(repo_id: str) -> str:
    normalized = SAFE_RUN_COMPONENT.sub("-", repo_id.lower().replace("/", "--")).strip(
        "-."
    )
    digest = hashlib.sha256(repo_id.encode("utf-8")).hexdigest()[:8]
    stem = normalized[:72].rstrip("-.") or "model"
    return f"{stem}-{digest}"


def discover_kersor_root(explicit: Path | None, workspace: Path) -> Path:
    if explicit is not None:
        candidates = [explicit]
    else:
        candidates = [workspace.parent / "kersor"]

    for candidate in candidates:
        root = candidate.expanduser().resolve()
        required = (
            root / "scripts/evolve.sh",
            root / "scripts/verify-autonomous-run.py",
            root / "scripts/run-autonomous-workflow.py",
            root / "config/runtime-codex-autonomous.json",
            root / "kersor_core/cli.py",
            root / "runtime/autonomous-controller.js",
            root / "runtime/workflow-host.mjs",
            root / "runtime/brokers/codex-exec.mjs",
        )
        if not all(path.is_file() for path in required):
            continue
        try:
            evolve_source = (root / "scripts/evolve.sh").read_text(encoding="utf-8")
        except OSError:
            continue
        if (
            EXPECTED_HASH_FLAG in evolve_source
            and EXPECTED_RUNTIME_HASH_FLAG in evolve_source
        ):
            return root
    rendered = ", ".join(str(path.expanduser()) for path in candidates)
    raise IntakeError(f"KerSor root not found; checked: {rendered}")


def validate_runtime_config(path: Path) -> Path:
    declared = path.expanduser()
    try:
        info = declared.lstat()
    except OSError as error:
        raise IntakeError(
            f"cannot inspect runtime config {declared}: {error}"
        ) from error
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise IntakeError("runtime config must be a regular non-symlink file")
    resolved = declared.resolve()
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise IntakeError(f"invalid runtime config {resolved}: {error}") from error
    if not isinstance(payload, dict):
        raise IntakeError("runtime config must contain one JSON object")
    try:
        validate_runtime_config_policy(payload)
    except RuntimeLockError as error:
        raise IntakeError(str(error)) from error
    return resolved


def load_source_lock(
    path: Path, workspace: Path
) -> tuple[Path, dict[str, object], dict[str, object], str]:
    declared = path.expanduser()
    try:
        info = declared.lstat()
    except OSError as error:
        raise IntakeError(f"cannot inspect source lock {declared}: {error}") from error
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise IntakeError("source lock must be a regular non-symlink file")
    resolved = declared.resolve()
    try:
        resolved.relative_to(workspace)
    except ValueError as error:
        raise IntakeError("source lock must be inside the ApxInf workspace") from error
    try:
        if info.st_size > 32 * 1024 * 1024:
            raise IntakeError("source lock exceeds the metadata byte cap")
        from resolve_hf_source import _read_json_bytes, validate_source_lock

        raw_payload = read_file_bytes(resolved, label="HF source lock")
        payload = _read_json_bytes(raw_payload, "source lock")
        receipt = validate_source_lock(payload)
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise IntakeError(f"invalid source lock {resolved}: {error}") from error
    return resolved, payload, receipt, hashlib.sha256(raw_payload).hexdigest()


def kersor_version(kersor_root: Path) -> str:
    for relative in (".codex-plugin/plugin.json", ".claude-plugin/plugin.json"):
        path = kersor_root / relative
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        value = payload.get("version")
        if isinstance(value, str) and value:
            return value
    return "unknown"


def build_mission(
    *,
    workspace: Path,
    session: Path,
    runtime_config: Path,
    repo_id: str,
    revision: str,
    resolved_commit: str,
    source_lock: Path,
    source_lock_sha256: str,
    mission_id: str,
) -> dict[str, object]:
    source_lock_relative = source_lock.relative_to(workspace).as_posix()
    source_authority = (
        f"read the frozen source lock {source_lock_relative} with SHA-256 "
        f"{source_lock_sha256}"
    )
    workspace_authority = "read the ApxInf workspace without modifying it"
    analyze_authority = "analyze macOS compatibility and resource feasibility"
    validate_authority = "run deterministic read-only onboarding validators"
    source_validator = workspace / "scripts/resolve_hf_source.py"
    manifest_validator = workspace / "scripts/validate_hf_port_manifest.py"
    deployment_profile = workspace / "configs/hf-onboarding/qwen35-0.8b-macos-cpu.json"
    host_python = str(Path(sys.executable).resolve(strict=True))

    return {
        "contract_version": "kersor-mission-v1",
        "workspace": str(workspace),
        "session": str(session),
        "runtime": "codex",
        "runtime_config": str(runtime_config),
        "planner_model": "opus",
        "worker_model": "opus",
        "planner_timeout_seconds": 900,
        "max_plan_attempts": 3,
        "mission": {
            "mission_id": mission_id,
            "goal": (
                f"Produce an evidence-backed, metadata-only macOS onboarding decision for "
                f"Hugging Face model {repo_id} at immutable commit {resolved_commit}. Use only "
                f"the validated source lock {source_lock_relative} and the ApxInf workspace; do "
                "not browse the network, download files, execute repository code, inspect Host "
                "internals or credentials, accept gated terms, or modify the workspace. The "
                "support claim is limited to task=text-generation with input_modalities=['text']; "
                "vision, video, audio, or MTP branches are outside READY_EXISTING scope. The "
                "port_manifest must describe the smallest safe next action and exact candidate "
                "paths; it must not claim deployment success."
            ),
            "authority": [
                source_authority,
                workspace_authority,
                analyze_authority,
                validate_authority,
            ],
            "required_artifacts": [
                "source_lock_receipt",
                "source_resolution",
                "security_assessment",
                "resource_plan",
                "architecture_fingerprint",
                "support_decision",
                "port_manifest",
                "port_manifest_receipt",
            ],
            "required_facts": {
                "source_lock_valid": True,
                "port_manifest_valid": True,
                "route_verified": True,
                "decision_complete": True,
            },
            "max_revisions": 3,
        },
        "capabilities": [
            {
                "name": "verify_source_lock",
                "description": (
                    "Independently validate the frozen source-lock schema, content hash, bounded "
                    "metadata receipt, immutable commit, and SafeTensors-only weight plan."
                ),
                "required_authorities": [validate_authority],
                "side_effect": "none",
                "evidence_prefixes": ["host-evaluator:"],
                "produces_artifacts": ["source_lock_receipt"],
                "produces_facts": ["source_lock_valid"],
                "execution": {
                    "kind": "host_evaluator",
                    "retryable": False,
                    "request": {
                        "protocol": "command-v1",
                        "filesystem_policy": "read-only",
                        "network_policy": "denied",
                        "output_policy": "sealed",
                        "argv": [
                            host_python,
                            "-S",
                            "-B",
                            str(source_validator),
                            "--verify",
                            str(source_lock),
                            "--expected-sha256",
                            source_lock_sha256,
                        ],
                        "cwd": ".",
                        "artifacts": [source_lock_relative],
                        "timeout_seconds": 60,
                        "max_output_bytes": 65536,
                    },
                    "fact_projections": [
                        {"output_name": "source_lock_valid", "result_path": "passed"}
                    ],
                },
            },
            {
                "name": "inspect_model_source",
                "description": (
                    f"Inspect only the validated source lock for {repo_id}@{resolved_commit}. "
                    "Assess gated/private/license metadata, remote-code indicators, formats, "
                    "file sizes, structural config, tokenizer/processor classes, tensor names, "
                    "and a conservative M4 16 GiB resource plan. The source lock contains no "
                    "model-card or template prose. Never access the network or model cache."
                ),
                "required_authorities": [source_authority, analyze_authority],
                "requires_artifacts": ["source_lock_receipt"],
                "evidence_prefixes": [str(source_lock)],
                "produces_artifacts": [
                    "source_resolution",
                    "security_assessment",
                    "resource_plan",
                ],
                "produces_facts": [],
                "planner_projection": {
                    "artifacts": [
                        "source_resolution",
                        "security_assessment",
                        "resource_plan",
                    ],
                    "max_chars": 32768,
                },
            },
            {
                "name": "classify_model_support",
                "description": (
                    "Compare the frozen source evidence with the ApxInf registry, strict loader "
                    "requirements, backend primitives, tokenizer/processor behavior, and macOS "
                    "resource limits. Classify exactly one route: READY_EXISTING, FAMILY_ADAPTER, "
                    "PORT_MODEL, EXTEND_BACKEND, EXTERNAL_PROVIDER, or BLOCKED. Do not modify files "
                    "and do not infer compatibility from model_type alone. Classify only "
                    "task=text-generation with input_modalities=['text']; READY_EXISTING must not "
                    "cover vision, video, audio, or MTP behavior."
                ),
                "required_authorities": [workspace_authority, analyze_authority],
                "requires_artifacts": [
                    "source_resolution",
                    "security_assessment",
                    "resource_plan",
                ],
                "evidence_prefixes": [str(workspace)],
                "produces_artifacts": [
                    "architecture_fingerprint",
                    "support_decision",
                ],
                "produces_facts": [],
                "planner_projection": {
                    "artifacts": ["architecture_fingerprint", "support_decision"],
                    "max_chars": 32768,
                },
            },
            {
                "name": "compile_port_manifest",
                "description": (
                    "Produce a machine-readable JSON port_manifest from all intake evidence. It "
                    "must contain schema_version=2, repo_id, requested_revision, resolved_commit, "
                    "source_lock_content_sha256, task=text-generation, "
                    "input_modalities=['text'], profile_id, target=macos-arm64, route, provider, "
                    "blockers, user_checkpoint_required, "
                    "transaction_paths, new_paths, and required_gates. Paths must be workspace-"
                    "relative, minimal, non-overlapping, and must exclude .git, .kersor, caches, "
                    "credentials, model weights, and Host run directories. Existing directories "
                    "must not be transaction paths. For the exact checked-in Qwen3.5-0.8B "
                    "profile use route=READY_EXISTING, profile_id=qwen35-0.8b-macos-cpu, "
                    "provider=native-apxinf-cpu, empty blockers/transaction_paths/new_paths, no "
                    "checkpoint, and required_gates in this exact order: source-lock, "
                    "bundle-integrity, pinned-macos-arm64-binary, "
                    "exact-greedy-token-trajectory, transformers-oracle-parity, "
                    "macos-memory-smoke. Other routes must set profile_id=null. This artifact is "
                    "a proposal for a later deterministic compiler, not authority to mutate files."
                ),
                "required_authorities": [workspace_authority, analyze_authority],
                "requires_artifacts": [
                    "source_resolution",
                    "security_assessment",
                    "resource_plan",
                    "architecture_fingerprint",
                    "support_decision",
                ],
                "evidence_prefixes": [str(workspace)],
                "produces_artifacts": ["port_manifest"],
                "produces_facts": [],
                "planner_projection": {
                    "artifacts": ["port_manifest"],
                    "max_chars": 32768,
                },
            },
            {
                "name": "validate_port_manifest",
                "description": (
                    "Validate the exact port_manifest schema and fail closed on unsafe, broad, "
                    "overlapping, credential, cache, weight, symlink, hardlink, or Host paths."
                ),
                "required_authorities": [validate_authority],
                "requires_artifacts": ["port_manifest"],
                "side_effect": "none",
                "evidence_prefixes": ["host-evaluator:"],
                "produces_artifacts": ["port_manifest_receipt"],
                "produces_facts": [
                    "port_manifest_valid",
                    "route_verified",
                    "decision_complete",
                ],
                "execution": {
                    "kind": "host_evaluator",
                    "retryable": False,
                    "request": {
                        "protocol": "command-v1",
                        "filesystem_policy": "read-only",
                        "network_policy": "denied",
                        "output_policy": "sealed",
                        "argv": [
                            host_python,
                            "-S",
                            "-B",
                            str(manifest_validator),
                            "--workspace",
                            str(workspace),
                            "--json",
                            "",
                            "--source-lock",
                            str(source_lock),
                            "--deployment-profile",
                            str(deployment_profile),
                            "--expected-repo-id",
                            repo_id,
                            "--expected-requested-revision",
                            revision,
                            "--expected-resolved-commit",
                            resolved_commit,
                            "--expected-source-lock-content-sha256",
                            source_lock_sha256,
                            "--require-ready-existing",
                        ],
                        "cwd": ".",
                        "artifacts": [],
                        "timeout_seconds": 60,
                        "max_output_bytes": 65536,
                    },
                    "input_artifact_field": "argv.7",
                    "fact_projections": [
                        {"output_name": "port_manifest_valid", "result_path": "passed"},
                        {"output_name": "route_verified", "result_path": "passed"},
                        {
                            "output_name": "decision_complete",
                            "result_path": "passed",
                        },
                    ],
                },
            },
        ],
    }


def session_cli_environment(
    kersor_root: Path,
    *,
    neutral_home: Path = Path("/nonexistent/apxinf-kersor-session-home"),
) -> dict[str, str]:
    trusted_path = os.pathsep.join(
        path
        for path in (
            str(Path(sys.executable).resolve().parent),
            "/opt/homebrew/bin",
            "/usr/local/bin",
            "/usr/bin",
            "/bin",
            "/usr/sbin",
            "/sbin",
        )
        if Path(path).is_dir()
    )
    return {
        "PATH": trusted_path,
        "HOME": str(neutral_home),
        "TMPDIR": str(neutral_home / "tmp"),
        "LANG": "C",
        "LC_ALL": "C",
        "PYTHONIOENCODING": "utf-8",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPYCACHEPREFIX": str(neutral_home / "pycache"),
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_TERMINAL_PROMPT": "0",
        "PYTHONPATH": str(kersor_root),
    }


def create_or_validate_session(
    *, session: Path, workspace: Path, kersor_root: Path, session_id: str
) -> None:
    config_path = session / "session-config.json"
    state_path = session / "state.json"
    with tempfile.TemporaryDirectory(prefix="apxinf-kersor-session-home-") as temporary:
        neutral_home = Path(temporary).resolve(strict=True)
        (neutral_home / "tmp").mkdir(mode=0o700)
        (neutral_home / "pycache").mkdir(mode=0o700)
        env = session_cli_environment(kersor_root, neutral_home=neutral_home)
        if session.exists() and config_path.is_file() and state_path.is_file():
            command = [
                sys.executable,
                "-m",
                "kersor_core.cli",
                "migrate-session",
                str(session),
                "--check",
            ]
            subprocess.run(command, cwd=kersor_root, env=env, check=True)
        elif session.exists():
            raise IntakeError(
                f"refusing incomplete or legacy Session directory: {session}"
            )
        else:
            session.mkdir(parents=True, exist_ok=False)

            payload = {
                "config": {
                    "schema_version": 2,
                    "kersor_version": kersor_version(kersor_root),
                    "max_workflows": 1,
                    "mode": "auto",
                    "runner_kind": "stable",
                    "input_mode": "repository",
                    "task_dir": str(workspace),
                    "user_note": "ApxInf Hugging Face macOS metadata-only intake",
                },
                "state": {
                    "schema_version": 2,
                    "phase": "optimizing",
                    "current_round": 1,
                    "session_id": session_id,
                    "target_speedup": None,
                    "seed_origin": "repository",
                    "kernel_language": "rust",
                    "backend": "cpu",
                    "integration_pattern": "hf-macos-intake",
                },
            }
            command = [
                sys.executable,
                "-m",
                "kersor_core.cli",
                "create-session",
                str(session),
            ]
            subprocess.run(
                command,
                cwd=kersor_root,
                env=env,
                input=json.dumps(payload),
                text=True,
                check=True,
            )

    if config_path.is_file() and state_path.is_file():
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise IntakeError(f"cannot validate existing Session: {error}") from error
        expected_config = {
            "schema_version": 2,
            "input_mode": "repository",
            "runner_kind": "stable",
            "task_dir": str(workspace),
        }
        expected_state = {
            "schema_version": 2,
            "session_id": session_id,
            "backend": "cpu",
            "integration_pattern": "hf-macos-intake",
        }
        for key, expected in expected_config.items():
            if config.get(key) != expected:
                raise IntakeError(
                    f"existing Session config mismatch for {key}: {config.get(key)!r}"
                )
        for key, expected in expected_state.items():
            if state.get(key) != expected:
                raise IntakeError(
                    f"existing Session state mismatch for {key}: {state.get(key)!r}"
                )
        return
    raise IntakeError(f"Session creation did not produce v2 state: {session}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def locked_launcher_commands(
    base: list[str],
) -> tuple[list[str], list[str], list[str]]:
    if any(
        flag in base for flag in ("--fresh", "--admit-only", "--resume", "--dry-run")
    ):
        raise IntakeError("launcher base command already contains a state-control flag")
    admit = [*base, "--admit-only"]
    return [*admit, "--dry-run"], admit, [*base, "--resume"]


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Prepare a read-only KerSor Mission for one HF macOS onboarding intake."
    )
    result.add_argument(
        "model", help="owner/model or canonical https://huggingface.co URL"
    )
    result.add_argument(
        "--revision", help="requested branch, tag, or commit; defaults to main"
    )
    result.add_argument(
        "--source-lock",
        type=Path,
        required=True,
        help="validated metadata-only source-lock.json created by resolve_hf_source.py",
    )
    result.add_argument(
        "--workspace",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="ApxInf workspace (default: this checkout)",
    )
    result.add_argument(
        "--kersor-root",
        type=Path,
        help="one exact KerSor installation/source root; never mix roots in one run",
    )
    result.add_argument(
        "--runtime-config",
        type=Path,
        help=(
            "read-only Codex runtime with a cleared worker environment "
            "(default: kersor/runtime-codex-hf-intake.json in this workspace)"
        ),
    )
    result.add_argument(
        "--session",
        type=Path,
        help="Session v2 directory (default: .kersor/hf-macos/<model>-<timestamp>)",
    )
    result.add_argument(
        "--output",
        type=Path,
        help="Mission JSON path (default: <session>/intake-mission.json)",
    )
    result.add_argument(
        "--codex",
        default="codex",
        help="exact Codex executable name or path to hash-bind (default: codex)",
    )
    result.add_argument(
        "--node",
        default="node",
        help="exact Node executable name or path to hash-bind (default: node)",
    )
    result.add_argument(
        "--codex-auth-home",
        type=Path,
        default=Path.home() / ".codex",
        help="directory containing a regular auth.json for launcher preflight",
    )
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        workspace = args.workspace.expanduser().resolve()
        if (
            not (workspace / "Cargo.toml").is_file()
            or not (workspace / "crates/apxinf-core").is_dir()
        ):
            raise IntakeError(f"not an ApxInf workspace: {workspace}")
        repo_id, revision = parse_model_reference(args.model, args.revision)
        slug = model_slug(repo_id)
        kersor_root = discover_kersor_root(args.kersor_root, workspace)
        runtime_config = validate_runtime_config(
            args.runtime_config
            if args.runtime_config is not None
            else workspace / "kersor/runtime-codex-hf-intake.json"
        )
        (
            source_lock,
            _,
            source_receipt,
            source_lock_file_sha256,
        ) = load_source_lock(args.source_lock, workspace)
        if source_receipt["repo_id"] != repo_id:
            raise IntakeError(
                f"source lock repo mismatch: {source_receipt['repo_id']} != {repo_id}"
            )
        if source_receipt["requested_revision"] != revision:
            raise IntakeError(
                "source lock requested revision does not match the command"
            )
        resolved_commit = str(source_receipt["resolved_commit"])
        source_lock_sha256 = str(source_receipt["content_sha256"])
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        mission_id = f"hf-intake-{slug}-{timestamp.lower()}"
        session = (
            args.session.expanduser().resolve()
            if args.session
            else workspace / ".kersor" / "hf-macos" / f"{slug}-{timestamp}"
        )
        output = (
            args.output.expanduser().resolve()
            if args.output
            else session / "intake-mission.json"
        )
        if output.parent != session:
            raise IntakeError(
                "Mission output must be an immediate child of its Session directory"
            )

        create_or_validate_session(
            session=session,
            workspace=workspace,
            kersor_root=kersor_root,
            session_id=mission_id,
        )
        mission = build_mission(
            workspace=workspace,
            session=session,
            runtime_config=runtime_config,
            repo_id=repo_id,
            revision=revision,
            resolved_commit=resolved_commit,
            source_lock=source_lock,
            source_lock_sha256=source_lock_sha256,
            mission_id=mission_id,
        )
        try:
            with output.open("x", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(mission, ensure_ascii=False, indent=2, sort_keys=True)
                    + "\n"
                )
                handle.flush()
                os.fsync(handle.fileno())
        except FileExistsError as error:
            raise IntakeError(
                f"refusing to overwrite an existing Mission; resume the frozen run: {output}"
            ) from error

        contract_hash = sha256_file(output)
        runtime_hash = sha256_file(runtime_config)
        runtime_lock_path = session / "kersor-runtime-lock.json"
        runtime_lock = build_runtime_lock(
            kersor_root=kersor_root,
            mission_path=output,
            runtime_config_path=runtime_config,
            codex_command=args.codex,
            node_command=args.node,
            source_lock_path=source_lock,
            host_python_path=Path(sys.executable),
        )
        write_runtime_lock(runtime_lock_path, runtime_lock)
        run_dir = Path(runtime_lock["mission_binding"]["run_dir"])
        launcher = workspace / "scripts/run_locked_kersor_mission.py"
        launcher_base = [
            sys.executable,
            str(launcher),
            "--lock",
            str(runtime_lock_path),
            "--auth-home",
            str(args.codex_auth_home.expanduser().resolve()),
            "--mission",
            str(output),
            "--runtime-config",
            str(runtime_config),
            "--codex",
            str(runtime_lock["runtime"]["codex"]["path"]),
            "--node",
            str(runtime_lock["runtime"]["node"]["executable"]["path"]),
        ]
        command, admit, resume = locked_launcher_commands(launcher_base)
        verify = [
            sys.executable,
            str(kersor_root / "scripts/verify-autonomous-run.py"),
            "--run-dir",
            str(run_dir),
        ]
        print(f"MISSION={output}")
        print(f"SOURCE_LOCK={source_lock}")
        print(f"SOURCE_LOCK_CONTENT_SHA256={source_lock_sha256}")
        print(f"SOURCE_LOCK_FILE_SHA256={source_lock_file_sha256}")
        print(f"MISSION_SHA256={contract_hash}")
        print(f"RUNTIME_CONFIG_SHA256={runtime_hash}")
        print(f"KERSOR_RUNTIME_LOCK={runtime_lock_path}")
        print(f"KERSOR_RUNTIME_LOCK_SHA256={runtime_lock['lock_sha256']}")
        print(f"RUN_DRY={shlex.join(command)}")
        print(f"FORMAL_ADMIT={shlex.join(admit)}")
        print(f"FORMAL_RESUME={shlex.join(resume)}")
        print(
            "ADMISSION_PROTOCOL=admit-only -> verify frozen admission -> explicit resume"
        )
        print(f"VERIFY={shlex.join(verify)}")
        return 0
    except (
        IntakeError,
        RuntimeLockError,
        OSError,
        subprocess.CalledProcessError,
    ) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
