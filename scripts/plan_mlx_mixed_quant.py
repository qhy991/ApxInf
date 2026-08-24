#!/usr/bin/env python3
"""Create and advance deterministic offline MLX mixed-quant policies.

The tool scans only local source metadata and safetensors headers.  It never
imports MLX, loads tensor payloads, or runs a model.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
import sys
import tempfile


SCRIPT_DIR = Path(__file__).resolve().parent
ERROR_FORMAT = "apxinf-mlx-mixed-quant-plan-error-v1"


def _load_sibling(module_name: str, filename: str) -> object:
    cached = sys.modules.get(module_name)
    if cached is not None:
        return cached
    path = SCRIPT_DIR / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load required module {filename}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


BUILDER_API = _load_sibling(
    "_apxinf_bundle_builder_for_mixed_plan", "build_mlx_bundle.py"
)
POLICY_API = _load_sibling(
    "_apxinf_mixed_quant_policy_for_plan", "mlx_mixed_quant_policy.py"
)


def _read_json(argument: str, label: str) -> dict[str, object]:
    path = BUILDER_API._require_absolute(argument, label)
    parent = BUILDER_API._require_owned_directory(path.parent, f"{label} parent")
    payload = BUILDER_API._read_regular(
        parent / path.name, label, BUILDER_API.MAX_JSON_BYTES
    )
    return BUILDER_API._parse_json(payload, label)


def _write_canonical_no_replace(argument: str, value: object) -> Path:
    requested = BUILDER_API._require_absolute(argument, "--output")
    parent = BUILDER_API._require_owned_directory(requested.parent, "output parent")
    destination = parent / requested.name
    BUILDER_API._require_output_absent(destination)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".apxinf-mixed-policy-", suffix=".json", dir=parent
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        payload = POLICY_API.canonical_bytes(value) + b"\n"
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise BUILDER_API.BundleError("short write while staging policy")
            view = view[written:]
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        BUILDER_API._rename_no_replace(temporary, destination)
        return destination
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _init(arguments: argparse.Namespace) -> dict[str, object]:
    source = BUILDER_API._inspect_source(arguments.source_dir)
    candidates = BUILDER_API._selective_candidate_modules(source.tensor_schema)
    source_contract = {
        "repo_id": arguments.repo_id,
        "revision": arguments.revision,
        "source_manifest_sha256": BUILDER_API._manifest_sha256(source.records),
        "config_sha256": source.records["config.json"].sha256,
        "language_schema_sha256": BUILDER_API._language_schema_sha256(
            source.tensor_schema
        ),
        "language_tensor_count": len(
            BUILDER_API._canonical_language_schema(source.tensor_schema)
        ),
    }
    trace = _read_json(arguments.trace_contract, "--trace-contract")
    quality_suite = _read_json(arguments.quality_suite, "--quality-suite")
    document = POLICY_API.create_initial_policy_document(
        source_contract,
        candidates,
        trace,
        POLICY_API.object_sha256(quality_suite),
    )
    _write_canonical_no_replace(arguments.output, document)
    return document


def _advance(arguments: argparse.Namespace) -> dict[str, object]:
    policy = _read_json(arguments.policy, "--policy")
    observation = _read_json(arguments.observation, "--observation")
    document = POLICY_API.advance_policy_document(policy, observation)
    _write_canonical_no_replace(arguments.output, document)
    return document


def _validate(arguments: argparse.Namespace) -> dict[str, object]:
    policy = _read_json(arguments.policy, "--policy")
    return POLICY_API.validate_policy_document(policy)


def _parse_arguments(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init")
    init.add_argument("--source-dir", required=True)
    init.add_argument("--repo-id", required=True)
    init.add_argument("--revision", required=True)
    init.add_argument("--trace-contract", required=True)
    init.add_argument("--quality-suite", required=True)
    init.add_argument("--output", required=True)
    advance = commands.add_parser("advance")
    advance.add_argument("--policy", required=True)
    advance.add_argument("--observation", required=True)
    advance.add_argument("--output", required=True)
    validate = commands.add_parser("validate")
    validate.add_argument("--policy", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    try:
        arguments = _parse_arguments(sys.argv[1:] if argv is None else argv)
        if arguments.command == "init":
            result = _init(arguments)
        elif arguments.command == "advance":
            result = _advance(arguments)
        else:
            result = _validate(arguments)
        sys.stdout.write(POLICY_API.canonical_bytes(result).decode("utf-8") + "\n")
        sys.stdout.flush()
        return 0
    except (BUILDER_API.BundleError, POLICY_API.PolicyError, OSError) as error:
        message = " ".join(str(error).split())[:2048] or "unknown planning error"
        payload = {
            "format": ERROR_FORMAT,
            "error": {"message": message},
        }
        sys.stderr.write(POLICY_API.canonical_bytes(payload).decode("utf-8") + "\n")
        sys.stderr.flush()
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
