#!/usr/bin/env python3
"""Freeze content-addressed metadata for one assignment release."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA = "apxinf.qwen38_27b.release_manifest.v1"


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: top-level value must be an object")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    return {"name": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def git(repository: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repository), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def manifest_record(path: Path, *, private: bool) -> dict[str, Any]:
    value = load_json(path)
    return {
        "private": private,
        "manifest_sha256": sha256_file(path),
        "contract_sha256": value.get("contract_sha256"),
        "cases_jsonl_sha256": value.get("cases_jsonl_sha256"),
        "case_count": value.get("case_count"),
        "split": value.get("split"),
        "suite": value.get("suite"),
        "seed_sha256": value.get("seed_sha256"),
        "tokenizer_combined_sha256": value.get("tokenizer", {}).get(
            "combined_sha256"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-id", required=True)
    parser.add_argument("--starter-repository", type=Path, required=True)
    parser.add_argument("--starter-public-url", required=True)
    parser.add_argument("--starter-revision", required=True)
    parser.add_argument("--evaluator-root", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--public-manifest", type=Path, required=True)
    parser.add_argument("--hidden-manifest", type=Path, required=True)
    parser.add_argument("--context-manifest", type=Path, required=True)
    parser.add_argument("--qualitative-manifest", type=Path, required=True)
    parser.add_argument("--multimodal-public-manifest", type=Path, required=True)
    parser.add_argument("--multimodal-hidden-manifest", type=Path, required=True)
    parser.add_argument("--trajectory-reference", type=Path, required=True)
    parser.add_argument(
        "--validation-submission",
        type=Path,
        action="append",
        default=[],
    )
    parser.add_argument("--container-image-digest")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    evaluator_root = args.evaluator_root.resolve(strict=True)
    contract_path = evaluator_root / "contract-v1.json"
    contract = load_json(contract_path)
    if contract.get("status") != "released":
        raise ValueError("contract status must be released before freezing metadata")
    multimodal_contract_path = evaluator_root / "multimodal-contract-v1.json"
    multimodal_contract = load_json(multimodal_contract_path)
    if multimodal_contract.get("status") != "released":
        raise ValueError("multimodal contract status must be released before freezing metadata")
    if multimodal_contract.get("overlay_for", {}).get("contract_sha256") != sha256_file(
        contract_path
    ):
        raise ValueError("multimodal overlay does not reference the released text contract")
    starter_repository = args.starter_repository.resolve(strict=True)
    revision = git(
        starter_repository,
        "rev-parse",
        f"{args.starter_revision}^{{commit}}",
    )
    if revision != args.starter_revision:
        raise ValueError("starter revision must be a full immutable commit SHA")

    evaluator_names = (
        "ASSIGNMENT.md",
        "INTERFACE.md",
        "README.md",
        "contract-v1.json",
        "multimodal-contract-v1.json",
        "submission-schema-v1.json",
        "multimodal-report-schema-v1.json",
        "fetch_public_corpus.py",
        "generate_evaluation_cases.py",
        "generate_multimodal_cases.py",
        "run_evaluation.py",
        "run_multimodal.py",
        "score_submission.py",
        "score_multimodal.py",
        "compute_efficiency.py",
        "test_evaluation.py",
        "test_multimodal_evaluation.py",
        "test_runner_protocol.py",
        "test_teacher_orchestration.py",
        "teacher/orchestrate_cohort.py",
        "teacher/generate_hidden_cases.py",
        "teacher/generate_hidden_multimodal_cases.py",
        "teacher/freeze_release.py",
    )
    evaluator_files = [file_record(evaluator_root / name) for name in evaluator_names]
    model_dir = args.model_dir.resolve(strict=True)
    model_paths = [
        path
        for name in (
            "config.json",
            "generation_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "model.safetensors.index.json",
        )
        if (path := model_dir / name).is_file()
    ]
    model_paths.extend(sorted(model_dir.glob("*.safetensors")))
    if not model_paths or not any(path.suffix == ".safetensors" for path in model_paths):
        raise ValueError("model directory has no safetensors checkpoint shards")
    model_files = [file_record(path) for path in model_paths]
    validation_submissions = []
    for path in args.validation_submission:
        submission = load_json(path)
        validation_submissions.append(
            {
                "sha256": sha256_file(path),
                "implementation": submission.get("implementation"),
                "correctness": submission.get("correctness"),
                "context": submission.get("context"),
                "reliability": submission.get("reliability"),
            }
        )

    public = manifest_record(args.public_manifest, private=False)
    hidden = manifest_record(args.hidden_manifest, private=True)
    context = manifest_record(args.context_manifest, private=False)
    qualitative = manifest_record(args.qualitative_manifest, private=False)
    for label, record in (
        ("public", public),
        ("hidden", hidden),
        ("context", context),
        ("qualitative", qualitative),
    ):
        if record["contract_sha256"] != sha256_file(contract_path):
            raise ValueError(f"{label} manifest was not generated from this contract")
    multimodal_public = manifest_record(args.multimodal_public_manifest, private=False)
    multimodal_hidden = manifest_record(args.multimodal_hidden_manifest, private=True)
    for label, record in (
        ("multimodal_public", multimodal_public),
        ("multimodal_hidden", multimodal_hidden),
    ):
        if record["contract_sha256"] != sha256_file(multimodal_contract_path):
            raise ValueError(
                f"{label} manifest was not generated from this multimodal contract"
            )

    release = {
        "schema": SCHEMA,
        "status": "released",
        "release_id": args.release_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "starter": {
            "revision": revision,
            "tree": git(starter_repository, "rev-parse", f"{revision}^{{tree}}"),
            "public_url": args.starter_public_url,
        },
        "evaluator": {
            "files": evaluator_files,
            "bundle_sha256": canonical_sha256(evaluator_files),
        },
        "contract_sha256": sha256_file(contract_path),
        "multimodal_contract_sha256": sha256_file(multimodal_contract_path),
        "model": {
            "repo_id": contract["model"]["repo_id"],
            "revision": contract["model"]["revision"],
            "files": model_files,
            "bundle_sha256": canonical_sha256(model_files),
        },
        "datasets": {
            "public": public,
            "hidden": hidden,
            "context": context,
            "qualitative": qualitative,
            "multimodal_public": multimodal_public,
            "multimodal_hidden": multimodal_hidden,
        },
        "trajectory_reference": {
            "sha256": sha256_file(args.trajectory_reference),
            "case_count": len(load_json(args.trajectory_reference).get("cases", {})),
        },
        "container_image_digest": args.container_image_digest,
        "validation_submissions": validation_submissions,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(release, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "release_manifest_sha256": sha256_file(args.output),
                "evaluator_bundle_sha256": release["evaluator"]["bundle_sha256"],
                "model_bundle_sha256": release["model"]["bundle_sha256"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
