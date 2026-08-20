#!/usr/bin/env python3
"""Generate and freeze the teacher-only image-input suite from a secret seed."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path


def load_generator(root: Path):
    path = root / "generate_multimodal_cases.py"
    spec = importlib.util.spec_from_file_location("qwen38_multimodal_generator", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--contract",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "multimodal-contract-v1.json",
    )
    args = parser.parse_args()
    seed_bytes = args.seed_file.read_bytes()
    if not seed_bytes:
        raise ValueError("hidden seed file is empty")
    seed_hash = hashlib.sha256(seed_bytes).hexdigest()
    seed = int(seed_hash[:16], 16)
    generator = load_generator(Path(__file__).resolve().parents[1])
    generated = generator.generate_cases(seed, "hidden-mm", 2)
    manifest = generator.write_suite(
        generated,
        args.output_dir,
        args.contract,
        "hidden",
        seed_sha256=seed_hash,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
