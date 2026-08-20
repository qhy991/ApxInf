#!/usr/bin/env python3
"""Balanced same-process M8 versus Marlin-M64 service screen."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

from run_benchmark import (
    HardwareSampler,
    health_check,
    load_tokenizer,
    read_json,
    read_jsonl,
    request_once,
    sha256_file,
    utc_now,
)


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, default=here / "spec.json")
    parser.add_argument("--data-dir", type=Path, default=here / "data")
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--pairs", type=int, required=True)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--base-url", default="http://127.0.0.1:8001")
    parser.add_argument("--model", default="Qwen3.8-27B-AWQ-INT4")
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def median(rows: list[dict[str, Any]], key: str) -> float:
    return statistics.median(float(row[key]) for row in rows)


def optional_median(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return statistics.median(values) if values else None


def token_comparison(left: list[int], right: list[int]) -> dict[str, Any]:
    compared = min(len(left), len(right))
    exact = sum(left[index] == right[index] for index in range(compared))
    prefix = 0
    for baseline, candidate in zip(left, right):
        if baseline != candidate:
            break
        prefix += 1
    denominator = max(len(left), len(right), 1)
    return {
        "baseline_count": len(left),
        "candidate_count": len(right),
        "compared": compared,
        "exact": exact,
        "exact_rate": exact / denominator,
        "common_prefix": prefix,
        "first_exact": bool(left and right and left[0] == right[0]),
    }


def main() -> None:
    args = parse_args()
    if args.pairs <= 0 or args.warmups < 0:
        raise SystemExit("pairs must be positive and warmups non-negative")
    spec = read_json(args.spec)
    manifest_path = args.data_dir / "manifest.json"
    manifest = read_json(manifest_path)
    text_path = args.data_dir / "text.jsonl"
    if sha256_file(text_path) != manifest["text_jsonl_sha256"]:
        raise SystemExit("text manifest hash mismatch")
    cases = {case["id"]: case for case in read_jsonl(text_path)}
    if args.case_id not in cases:
        raise SystemExit(f"unknown text case: {args.case_id}")
    case = cases[args.case_id]
    tokenizer = load_tokenizer(Path(spec["model"]["local_path"]))
    health_check(args.base_url, args.timeout)

    sampler = HardwareSampler(spec["telemetry"]["sample_interval_ms"])
    sampler.start()
    time.sleep(max(0.5, sampler.interval_ms / 1000 * 2))
    records: list[dict[str, Any]] = []
    try:
        for warmup in range(args.warmups):
            for mode in ("m8", "marlin-m64"):
                row = request_once(
                    case,
                    args.data_dir,
                    args.base_url,
                    args.model,
                    args.timeout,
                    spec["server"]["chat_template_kwargs"],
                    tokenizer,
                    sampler,
                    mode,
                )
                print(
                    f"warmup[{warmup}] {mode} success={row['success']} "
                    f"functional={row['functional_pass']} ttft={row['ttft_s']}"
                )
                if not row["success"]:
                    raise SystemExit(f"warmup failed for {mode}: {row['error']}")
        for pair in range(args.pairs):
            order = ("m8", "marlin-m64") if pair % 2 == 0 else ("marlin-m64", "m8")
            for order_index, mode in enumerate(order):
                row = request_once(
                    case,
                    args.data_dir,
                    args.base_url,
                    args.model,
                    args.timeout,
                    spec["server"]["chat_template_kwargs"],
                    tokenizer,
                    sampler,
                    mode,
                )
                row.update(
                    {
                        "pair": pair,
                        "order": "AB" if pair % 2 == 0 else "BA",
                        "order_index": order_index,
                        "prefill_mode": mode,
                    }
                )
                path = row.get("apxinf") or {}
                row["path_proof"] = {
                    "reported_mode": path.get("prefill_mode"),
                    "marlin_m64_tiles": path.get("marlin_m64_tiles"),
                    "m8_tiles": path.get("m8_tiles"),
                    "m1_tokens": path.get("m1_tokens"),
                    "pass": path.get("prefill_mode") == mode
                    and (
                        mode == "m8"
                        or int(path.get("marlin_m64_tiles") or 0) > 0
                    ),
                }
                records.append(row)
                print(
                    f"pair[{pair}] {mode} success={row['success']} "
                    f"functional={row['functional_pass']} ttft={row['ttft_s']} "
                    f"tpot={row['tpot_s']} path={row['path_proof']}"
                )
    finally:
        sampler.stop()

    baseline = [row for row in records if row["prefill_mode"] == "m8"]
    candidate = [row for row in records if row["prefill_mode"] == "marlin-m64"]
    comparisons = []
    for pair in range(args.pairs):
        left = next(row for row in baseline if row["pair"] == pair)
        right = next(row for row in candidate if row["pair"] == pair)
        comparison = token_comparison(
            list((left.get("apxinf") or {}).get("token_ids") or []),
            list((right.get("apxinf") or {}).get("token_ids") or []),
        )
        comparison.update(
            {
                "pair": pair,
                "ttft_speedup": left["ttft_s"] / right["ttft_s"],
                "candidate_ttft_win": right["ttft_s"] < left["ttft_s"],
                "baseline_functional": left["functional_pass"],
                "candidate_functional": right["functional_pass"],
                "functional_non_regression": (
                    not left["functional_pass"] or right["functional_pass"]
                ),
            }
        )
        comparisons.append(comparison)

    ttft_speedup = median(baseline, "ttft_s") / median(candidate, "ttft_s")
    baseline_tpot = optional_median(baseline, "tpot_s")
    candidate_tpot = optional_median(candidate, "tpot_s")
    tpot_ratio = (
        candidate_tpot / baseline_tpot
        if baseline_tpot is not None
        and baseline_tpot > 0.0
        and candidate_tpot is not None
        else None
    )
    wins = sum(item["candidate_ttft_win"] for item in comparisons)
    total_exact = sum(item["exact"] for item in comparisons)
    total_tokens = sum(max(item["baseline_count"], item["candidate_count"]) for item in comparisons)
    trajectory_rate = total_exact / max(total_tokens, 1)
    all_success = all(row["success"] for row in records)
    all_functional = all(row["functional_pass"] for row in records)
    functional_non_regression = all(
        item["functional_non_regression"] for item in comparisons
    )
    all_paths = all(row["path_proof"]["pass"] for row in records)
    all_first_exact = all(item["first_exact"] for item in comparisons)
    target_context = int(case.get("target_context_tokens") or 0)
    performance_case = case["suite"] == "text-performance"
    required_wins = 4 if target_context == 1024 else 2 if target_context == 8192 else 0
    performance_pass = (
        not performance_case
        or (
            ttft_speedup >= 1.50
            and wins >= required_wins
            and tpot_ratio is not None
            and tpot_ratio <= 1.05
            and trajectory_rate >= 0.90
            and all_first_exact
        )
    )
    admission_pass = (
        all_success and functional_non_regression and all_paths and performance_pass
    )
    result = {
        "schema": "apxinf.qwen38_27b.prefill_mode_ab.v1",
        "timestamp_utc": utc_now(),
        "case": {
            "id": case["id"],
            "suite": case["suite"],
            "target_context_tokens": target_context,
            "input_sha256": case["input_sha256"],
            "max_tokens": case["max_tokens"],
        },
        "contract": {
            "baseline": "m8",
            "candidate": "marlin-m64",
            "pairs": args.pairs,
            "warmups_per_arm": args.warmups,
            "same_resident_process": True,
            "timing": "client send through SSE",
        },
        "records": records,
        "comparisons": comparisons,
        "summary": {
            "all_success": all_success,
            "all_functional": all_functional,
            "functional_non_regression": functional_non_regression,
            "all_paths": all_paths,
            "all_first_tokens_exact": all_first_exact,
            "trajectory_exact_rate": trajectory_rate,
            "baseline_ttft_median_s": median(baseline, "ttft_s"),
            "candidate_ttft_median_s": median(candidate, "ttft_s"),
            "ttft_speedup": ttft_speedup,
            "candidate_ttft_wins": wins,
            "baseline_tpot_median_s": baseline_tpot,
            "candidate_tpot_median_s": candidate_tpot,
            "candidate_over_baseline_tpot": tpot_ratio,
            "performance_pass": performance_pass,
            "admission_pass": admission_pass,
        },
        "evidence": {
            "spec_sha256": sha256_file(args.spec),
            "dataset_manifest_sha256": sha256_file(manifest_path),
        },
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if not admission_pass:
        raise SystemExit("prefill-mode admission failed")


if __name__ == "__main__":
    main()
