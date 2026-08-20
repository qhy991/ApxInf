#!/usr/bin/env python3
"""Score one Qwen3.8-27B leaderboard result using the frozen v1 contract."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


SUBMISSION_SCHEMA = "apxinf.qwen38_27b.leaderboard_submission.v1"


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: top-level JSON value must be an object")
    return value


def _rate(value: Any, field: str, *, allow_none: bool = False) -> float | None:
    if value is None and allow_none:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a number in [0, 1]")
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"{field} must be a finite number in [0, 1]")
    return number


def _positive(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) and number > 0.0 else None


def _validate_top_level(submission: dict[str, Any]) -> None:
    required = ("implementation", "correctness", "cells", "context", "reliability")
    if submission.get("schema") != SUBMISSION_SCHEMA:
        raise ValueError(f"schema must equal {SUBMISSION_SCHEMA!r}")
    missing = [field for field in required if not isinstance(submission.get(field), dict)]
    if missing:
        raise ValueError(f"missing object field(s): {', '.join(missing)}")
    implementation = submission["implementation"]
    for field in ("name", "revision", "backend"):
        if not isinstance(implementation.get(field), str) or not implementation[field].strip():
            raise ValueError(f"implementation.{field} must be a non-empty string")


def _score_latency_cells(
    definitions: list[dict[str, Any]],
    cells: dict[str, Any],
    metric: str,
    minimum_measured_repeats: int,
    required_warmup_repeats: int,
) -> tuple[float, list[dict[str, Any]]]:
    total = 0.0
    details: list[dict[str, Any]] = []
    for definition in definitions:
        cell_id = definition["id"]
        observed_cell = cells.get(cell_id)
        reasons: list[str] = []
        observed = None
        if not isinstance(observed_cell, dict):
            reasons.append("missing_cell")
        else:
            if observed_cell.get("actual_prompt_tokens") != definition["prompt_tokens"]:
                reasons.append("prompt_token_count_mismatch")
            if observed_cell.get("completion_tokens") != definition["output_tokens"]:
                reasons.append("output_token_count_mismatch")
            success_rate = _rate(
                observed_cell.get("success_rate"),
                f"cells.{cell_id}.success_rate",
                allow_none=True,
            )
            if success_rate != 1.0:
                reasons.append("success_rate_not_one")
            measured_repeats = observed_cell.get("measured_repeats")
            if (
                not isinstance(measured_repeats, int)
                or isinstance(measured_repeats, bool)
                or measured_repeats < minimum_measured_repeats
            ):
                reasons.append("insufficient_measured_repeats")
            warmup_repeats = observed_cell.get("warmup_repeats")
            if (
                not isinstance(warmup_repeats, int)
                or isinstance(warmup_repeats, bool)
                or warmup_repeats < required_warmup_repeats
            ):
                reasons.append("insufficient_warmup_repeats")
            observed = _positive(observed_cell.get(metric))
            if observed is None:
                reasons.append(f"missing_or_invalid_{metric}")

        points = 0.0
        if not reasons and observed is not None:
            points = float(definition["weight"]) * min(
                1.0, float(definition["anchor_seconds"]) / observed
            )
        total += points
        details.append(
            {
                "id": cell_id,
                "metric": metric,
                "eligible": not reasons,
                "observed_seconds": observed,
                "anchor_seconds": float(definition["anchor_seconds"]),
                "weight": float(definition["weight"]),
                "points": points,
                "reasons": reasons,
            }
        )
    return total, details


def score_submission(
    contract: dict[str, Any],
    submission: dict[str, Any],
    profile_name: str,
) -> dict[str, Any]:
    _validate_top_level(submission)
    profiles = contract.get("score_profiles", {})
    if profile_name not in profiles:
        raise ValueError(f"unknown score profile {profile_name!r}; choose from {sorted(profiles)}")
    profile = profiles[profile_name]

    correctness = submission["correctness"]
    protocol_pass = correctness.get("protocol_pass") is True
    public_rate = _rate(correctness.get("public_pass_rate"), "correctness.public_pass_rate")
    hidden_rate = _rate(
        correctness.get("hidden_pass_rate"),
        "correctness.hidden_pass_rate",
        allow_none=True,
    )
    trajectory_rate = _rate(
        correctness.get("token_trajectory_rate"),
        "correctness.token_trajectory_rate",
    )
    correctness_weights = profile["correctness"]
    correctness_points = {
        "protocol": float(correctness_weights["protocol"]) * float(protocol_pass),
        "public_cases": float(correctness_weights["public_cases"]) * float(public_rate),
        "hidden_cases": float(correctness_weights["hidden_cases"]) * float(hidden_rate or 0.0),
        "token_trajectory": float(correctness_weights["token_trajectory"]) * float(trajectory_rate),
    }

    latency = contract["latency_scoring"]
    measurement = profile["measurement"]
    minimum_measured_repeats = int(measurement["minimum_measured_repeats_per_latency_cell"])
    required_warmup_repeats = int(measurement["required_warmup_repeats"])
    ttft_points, ttft_details = _score_latency_cells(
        latency["ttft_cells"],
        submission["cells"],
        "ttft_s",
        minimum_measured_repeats,
        required_warmup_repeats,
    )
    tpot_points, tpot_details = _score_latency_cells(
        latency["tpot_cells"],
        submission["cells"],
        "tpot_s",
        minimum_measured_repeats,
        required_warmup_repeats,
    )

    context = submission["context"]
    context_contract = contract["context_scoring"]
    max_prompt = context.get("max_verified_prompt_tokens")
    output_tokens = context.get("verified_output_tokens")
    if not isinstance(max_prompt, int) or max_prompt < 0:
        raise ValueError("context.max_verified_prompt_tokens must be a non-negative integer")
    if not isinstance(output_tokens, int) or output_tokens < 0:
        raise ValueError("context.verified_output_tokens must be a non-negative integer")
    context_reasons: list[str] = []
    if output_tokens < int(context_contract["minimum_output_tokens"]):
        context_reasons.append("insufficient_verified_output_tokens")
    if context.get("service_healthy_after_failure") is not True:
        context_reasons.append("service_not_healthy_after_failure")
    context_points = 0.0
    if not context_reasons:
        minimum = float(context_contract["minimum_prompt_tokens"])
        maximum = float(context_contract["full_score_prompt_tokens"])
        if max_prompt > 0:
            progress = math.log2(max(float(max_prompt), minimum) / minimum) / math.log2(
                maximum / minimum
            )
            context_points = float(context_contract["weight"]) * min(1.0, max(0.0, progress))

    reliability = submission["reliability"]
    reliability_contract = contract["reliability_scoring"]
    request_success_rate = _rate(
        reliability.get("request_success_rate"), "reliability.request_success_rate"
    )
    boolean_checks: dict[str, bool] = {}
    for field in reliability_contract["boolean_checks"]:
        if not isinstance(reliability.get(field), bool):
            raise ValueError(f"reliability.{field} must be boolean")
        boolean_checks[field] = reliability[field]
    boolean_weight_total = float(reliability_contract["weight"]) - float(
        reliability_contract["request_success_rate_weight"]
    )
    boolean_weight = boolean_weight_total / len(boolean_checks)
    reliability_points = float(reliability_contract["request_success_rate_weight"]) * float(
        request_success_rate
    ) + boolean_weight * sum(boolean_checks.values())

    eligibility_contract = profile["eligibility"]
    eligibility_failures: list[str] = []
    if not protocol_pass:
        eligibility_failures.append("protocol_failed")
    if public_rate < float(eligibility_contract["public_pass_rate_min"]):
        eligibility_failures.append("public_pass_rate_below_threshold")
    hidden_min = eligibility_contract["hidden_pass_rate_min"]
    if hidden_min is not None and (hidden_rate is None or hidden_rate < float(hidden_min)):
        eligibility_failures.append("hidden_pass_rate_below_threshold_or_missing")
    if trajectory_rate < float(eligibility_contract["token_trajectory_rate_min"]):
        eligibility_failures.append("token_trajectory_rate_below_threshold")
    failed_checks = [field for field, passed in boolean_checks.items() if not passed]
    eligibility_failures.extend(f"reliability_check_failed:{field}" for field in failed_checks)

    section_scores = {
        "correctness": sum(correctness_points.values()),
        "ttft": ttft_points,
        "tpot": tpot_points,
        "context": context_points,
        "reliability": reliability_points,
    }
    diagnostic_score = sum(section_scores.values())
    eligible = not eligibility_failures
    return {
        "schema": "apxinf.qwen38_27b.leaderboard_score.v1",
        "implementation": submission["implementation"],
        "profile": profile_name,
        "eligible": eligible,
        "leaderboard_score": diagnostic_score if eligible else None,
        "diagnostic_score": diagnostic_score,
        "section_scores": section_scores,
        "correctness_details": correctness_points,
        "latency_details": {"ttft": ttft_details, "tpot": tpot_details},
        "context_details": {
            "max_verified_prompt_tokens": max_prompt,
            "points": context_points,
            "reasons": context_reasons,
        },
        "reliability_details": {
            "request_success_rate": request_success_rate,
            "boolean_checks": boolean_checks,
            "points": reliability_points,
        },
        "eligibility_failures": eligibility_failures,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parent
    parser.add_argument("--contract", type=Path, default=here / "contract-v1.json")
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument(
        "--profile",
        choices=("public_calibration", "midterm_leaderboard"),
        default="public_calibration",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    result = score_submission(_load_json(args.contract), _load_json(args.submission), args.profile)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
