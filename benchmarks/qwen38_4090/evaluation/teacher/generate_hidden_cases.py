#!/usr/bin/env python3
"""Generate the teacher-only hidden functional and trajectory dataset."""

from __future__ import annotations

import argparse
import json
import os
import random
import secrets
import sys
from pathlib import Path
from typing import Any


EVALUATION_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EVALUATION_ROOT))
import generate_evaluation_cases as generation  # noqa: E402


def hidden_cases(
    contract: dict[str, Any],
    tokenizer,
    corpus: str,
    rng: random.Random,
) -> list[dict[str, Any]]:
    output_budget = int(
        contract["correctness_workload"]["hidden_functional_suite"][
            "output_tokens_per_case"
        ]
    )
    definitions = []
    for index, (length, position) in enumerate(
        zip((2048, 4096, 8192, 16384), (0.17, 0.39, 0.64, 0.83)),
        start=1,
    ):
        definitions.append(
            (
                "exact_retrieval",
                length,
                f"retrieval-{index}",
                generation.retrieval_semantics(rng, position, f"H{index}"),
            )
        )
    for index, length in enumerate((4096, 12288), start=1):
        definitions.append(
            (
                "distractor_disambiguation",
                length,
                f"distractor-{index}",
                generation.distractor_semantics(rng),
            )
        )
    for index, length in enumerate((8192, 16384), start=1):
        definitions.append(
            (
                "multi_hop",
                length,
                f"multihop-{index}",
                generation.multihop_semantics(rng),
            )
        )
    for index, length in enumerate((8192, 12288), start=1):
        definitions.append(
            (
                "revision_resolution",
                length,
                f"revision-{index}",
                generation.revision_semantics(rng),
            )
        )
    for index, (length, record_count) in enumerate(
        ((8192, 2), (16384, 3)), start=1
    ):
        definitions.append(
            (
                "aggregate",
                length,
                f"aggregate-{index}",
                generation.aggregate_semantics(rng, record_count=record_count),
            )
        )

    cases: list[dict[str, Any]] = []
    for category, length, label, semantics in definitions:
        opaque_id = generation.sha256_bytes(
            f"{label}:{rng.random()}".encode("utf-8")
        )[:16]
        cases.append(
            generation.make_case(
                tokenizer=tokenizer,
                corpus=corpus,
                rng=rng,
                case_id=f"hidden-{opaque_id}",
                split="hidden",
                suite="functional",
                category=category,
                target_tokens=length,
                semantics=semantics,
                max_new_tokens=output_budget,
                ignore_eos=False,
                validation="normalized_exact",
                include_prompt=False,
            )
        )
    for index, length in enumerate((6144, 14336), start=1):
        opaque_id = generation.sha256_bytes(
            f"trajectory-{index}:{rng.random()}".encode("utf-8")
        )[:16]
        cases.append(
            generation.make_case(
                tokenizer=tokenizer,
                corpus=corpus,
                rng=rng,
                case_id=f"hidden-trajectory-{opaque_id}",
                split="hidden",
                suite="trajectory",
                category="fixed_decode_budget",
                target_tokens=length,
                semantics=(
                    "下面是教师隐藏的长文本 trajectory 负载。",
                    [(0.50, "【课程私有档案】隐藏性能标记 HIDDEN-QWEN38。")],
                    "任务：先输出 HIDDEN-QWEN38，然后持续输出词语 trajectory。",
                    None,
                ),
                max_new_tokens=128,
                ignore_eos=True,
                validation="nonempty",
                include_prompt=False,
                roles=("hidden_trajectory",),
            )
        )
    return cases


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=EVALUATION_ROOT / "contract-v1.json")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument(
        "--corpus", type=Path, default=EVALUATION_ROOT / ".cache" / "pg24264.txt"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hidden-seed-file", type=Path, required=True)
    parser.add_argument("--create-hidden-seed", action="store_true")
    args = parser.parse_args()

    contract = generation.load_json(args.contract)
    corpus = generation.extract_corpus(
        args.corpus, contract["data_generation"]["corpus"]["sha256"]
    )
    tokenizer = generation.load_tokenizer(args.model_dir)
    if args.create_hidden_seed:
        args.hidden_seed_file.parent.mkdir(parents=True, exist_ok=True)
        seed = secrets.randbits(256)
        with args.hidden_seed_file.open("x", encoding="ascii") as handle:
            handle.write(f"{seed}\n")
        os.chmod(args.hidden_seed_file, 0o600)
    else:
        seed = int(args.hidden_seed_file.read_text(encoding="ascii").strip())
    rng = random.Random(seed)
    manifest = generation.write_dataset(
        args.output_dir,
        hidden_cases(contract, tokenizer, corpus, rng),
        contract_path=args.contract,
        contract=contract,
        corpus_path=args.corpus,
        model_dir=args.model_dir,
        split="hidden",
        seed=seed,
        suite="hidden",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
