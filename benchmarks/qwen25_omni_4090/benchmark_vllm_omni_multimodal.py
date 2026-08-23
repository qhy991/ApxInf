#!/usr/bin/env python3
"""Real PNG/WAV capability and latency gate for vLLM-Omni."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path
from typing import Any

from benchmark_service import Case, HardwareSampler, utc_now
from benchmark_vllm_omni import (
    get_json,
    health_status,
    stream_openai_chat,
    summarize,
)


SCHEMA = "apxinf.qwen25_omni.vllm_omni_multimodal.v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def request_body(
    model: str, kind: str, media: Path, prompt: str, output_tokens: int
) -> dict[str, Any]:
    encoded = base64.b64encode(media.read_bytes()).decode("ascii")
    if kind == "image":
        media_part = {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{encoded}"},
        }
    elif kind == "audio":
        media_part = {
            "type": "input_audio",
            "input_audio": {"format": "wav", "data": encoded},
        }
    else:
        raise ValueError(f"unsupported media kind: {kind}")
    return {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [media_part, {"type": "text", "text": prompt}],
            }
        ],
        "modalities": ["text"],
        "max_tokens": output_tokens,
        "temperature": 0,
        "top_p": 1,
        "seed": 42,
        "repetition_penalty": 1.0,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    }


def run_case(
    base_url: str,
    model: str,
    case: Case,
    kind: str,
    media: Path,
    prompt: str,
    required_text: str,
    timeout: float,
    sampler: HardwareSampler,
) -> dict[str, Any]:
    trial = stream_openai_chat(
        base_url,
        request_body(model, kind, media, prompt, case.output_tokens),
        case.prompt_tokens,
        case.output_tokens,
        timeout,
        sampler,
    )
    output_text = str(trial.get("output_text", ""))
    semantic_passed = required_text.casefold() in output_text.casefold()
    trial["semantic_required_text"] = required_text
    trial["semantic_passed"] = semantic_passed
    trial["passed"] = bool(trial.get("passed")) and semantic_passed
    if not semantic_passed and trial.get("error") is None:
        trial["error"] = f"response did not contain required text: {required_text}"
    return trial


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8003")
    parser.add_argument("--model")
    parser.add_argument(
        "--image", type=Path, default=Path("scripts/roofline_decode_throughput.png")
    )
    parser.add_argument(
        "--audio",
        type=Path,
        default=Path("/var/lib/agent-gpu-broker/apxinf-omni-tone.wav"),
    )
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--sample-interval-ms", type=int, default=100)
    parser.add_argument("--apxinf-reference", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.warmups < 0 or args.repeats < 1:
        raise SystemExit("warmups must be nonnegative and repeats must be positive")
    image = args.image.resolve(strict=True)
    audio = args.audio.resolve(strict=True)
    base_url = args.base_url.rstrip("/")
    if health_status(base_url, args.timeout) != 200:
        raise SystemExit("vLLM-Omni health endpoint is not ready")
    version = get_json(f"{base_url}/version", args.timeout)
    models = get_json(f"{base_url}/v1/models", args.timeout)
    model = args.model or models["data"][0]["id"]

    cases = [
        {
            "case": Case("real_png_chart_title", 1760, 16, "multimodal"),
            "kind": "image",
            "media": image,
            "prompt": "Read the chart title. Answer with the title only.",
            "required_text": "TinyLlama-1.1B Decode Throughput vs Position",
        },
        {
            "case": Case("real_wav_sine_description", 52, 16, "multimodal"),
            "kind": "audio",
            "media": audio,
            "prompt": "Describe this audio signal briefly.",
            "required_text": "sine wave",
        },
    ]
    sampler = HardwareSampler(args.sample_interval_ms)
    sampler.start()
    started_at = utc_now()
    rows: list[dict[str, Any]] = []
    try:
        for item in cases:
            case = item["case"]
            warmup_failure = None
            for _ in range(args.warmups):
                warmup = run_case(
                    base_url,
                    model,
                    case,
                    item["kind"],
                    item["media"],
                    item["prompt"],
                    item["required_text"],
                    args.timeout,
                    sampler,
                )
                if not warmup["passed"]:
                    warmup_failure = warmup
                    break
            trials = []
            if warmup_failure is None:
                trials = [
                    run_case(
                        base_url,
                        model,
                        case,
                        item["kind"],
                        item["media"],
                        item["prompt"],
                        item["required_text"],
                        args.timeout,
                        sampler,
                    )
                    for _ in range(args.repeats)
                ]
            rows.append(
                {
                    "case": {
                        "case_id": case.case_id,
                        "prompt_tokens": case.prompt_tokens,
                        "output_tokens": case.output_tokens,
                        "kind": "image/png" if item["kind"] == "image" else "audio/wav",
                        "path": str(item["media"]),
                        "sha256": sha256_file(item["media"]),
                        "prompt": item["prompt"],
                        "semantic_required_text": item["required_text"],
                    },
                    "warmup_failure": warmup_failure,
                    "trials": trials,
                    "summary": summarize(case, trials) if trials else None,
                }
            )
    finally:
        sampler.stop()

    reference = None
    if args.apxinf_reference is not None:
        reference_path = args.apxinf_reference.resolve(strict=True)
        reference = {
            "path": str(reference_path),
            "sha256": sha256_file(reference_path),
        }
    report = {
        "schema": SCHEMA,
        "started_at": started_at,
        "completed_at": utc_now(),
        "engine": "vLLM-Omni",
        "engine_version": version,
        "model": model,
        "models_response": models,
        "endpoint": "/v1/chat/completions",
        "contract": {
            "single_request": True,
            "sampling": "greedy",
            "temperature": 0,
            "ignore_eos": True,
            "stream": True,
            "output_modalities": ["text"],
            "max_tokens": 16,
            "warmups": args.warmups,
            "repeats": args.repeats,
        },
        "apxinf_reference": reference,
        "cases": rows,
        "passed": all(
            row["warmup_failure"] is None
            and row["trials"]
            and all(trial["passed"] for trial in row["trials"])
            for row in rows
        ),
        "limitations": [
            "Semantic checks cover one real PNG and one real WAV only",
            "Text output only; Talker and Code2Wav are excluded from both engines",
            "Cross-engine token identity is not required because kernel stacks differ",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "passed": report["passed"],
                "summaries": [row["summary"] for row in rows],
            },
            ensure_ascii=False,
        )
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
