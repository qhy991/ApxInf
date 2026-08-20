#!/usr/bin/env python3
"""Generate deterministic text and multimodal workloads for Qwen3.8.

The generated JSONL files are derived artifacts. spec.json is the single
source of truth for target lengths, seeds, model identity, and output budgets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Callable

from PIL import Image, ImageDraw, ImageFont
from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, default=here / "spec.json")
    parser.add_argument("--output-dir", type=Path, default=here / "data")
    parser.add_argument("--model-dir", type=Path)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_tokenizer(model_dir: Path):
    return AutoTokenizer.from_pretrained(
        str(model_dir),
        trust_remote_code=True,
        local_files_only=True,
        use_fast=True,
    )


def chat_token_ids(tokenizer, content: str) -> list[int]:
    messages = [{"role": "user", "content": content}]
    kwargs = {
        "tokenize": True,
        "add_generation_prompt": True,
        "enable_thinking": False,
        "preserve_thinking": False,
    }
    try:
        ids = tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("preserve_thinking", None)
        try:
            ids = tokenizer.apply_chat_template(messages, **kwargs)
        except TypeError:
            kwargs.pop("enable_thinking", None)
            ids = tokenizer.apply_chat_template(messages, **kwargs)
    if isinstance(ids, Mapping):
        ids = ids["input_ids"]
    return list(ids)


FILLER_UNIT = (
    " Archive record: the north wind crossed the silver valley while the "
    "cobalt instrument retained its original checksum."
)


def fit_prompt(
    token_count: Callable[[str], int],
    target_tokens: int,
    prefix: str,
    middle: str,
    suffix: str,
    middle_fraction: float,
) -> tuple[str, int]:
    """Fit a prompt at or just below target_tokens without truncating semantics."""

    def compose(units: int, calibration: str = "") -> str:
        left_units = int(units * middle_fraction)
        right_units = units - left_units
        return (
            prefix
            + FILLER_UNIT * left_units
            + middle
            + FILLER_UNIT * right_units
            + calibration
            + suffix
        )

    shell_tokens = token_count(compose(0))
    if shell_tokens > target_tokens:
        raise ValueError(f"target {target_tokens} is smaller than semantic prompt shell")

    # Derive the search bound from this tokenizer instead of guessing how many
    # tokens a filler unit contains. The previous character-scale guess could
    # tokenize a 300K+ temporary string while fitting a 32K prompt.
    one_unit_tokens = token_count(compose(1))
    unit_cost = max(1, one_unit_tokens - shell_tokens)
    estimated_units = max(0, (target_tokens - shell_tokens) // unit_cost)
    low, high = 0, max(1, estimated_units + 64)
    while token_count(compose(high)) <= target_tokens:
        low, high = high, high * 2
    while low + 1 < high:
        mid = (low + high) // 2
        if token_count(compose(mid)) <= target_tokens:
            low = mid
        else:
            high = mid

    calibration = ""
    current = token_count(compose(low, calibration))
    # For the Qwen tokenizer, " x" is normally one token. Keep the loop
    # bounded and preserve the final question even if a future tokenizer
    # makes exact fitting impossible.
    while current < target_tokens:
        candidate = calibration + " x"
        candidate_count = token_count(compose(low, candidate))
        if candidate_count <= current or candidate_count > target_tokens:
            break
        calibration, current = candidate, candidate_count
    return compose(low, calibration), current


def text_cases(spec: dict[str, Any], tokenizer) -> list[dict[str, Any]]:
    text_spec = spec["text"]
    count = lambda prompt: len(chat_token_ids(tokenizer, prompt))
    cases: list[dict[str, Any]] = []

    for target in text_spec["context_lengths"]:
        for position in text_spec["needle_positions"]:
            position_tag = f"p{int(position * 100):02d}"
            answer = f"K-{target}-{position_tag}-7291"
            prefix = (
                "You are reading an immutable archive. Retain any explicit "
                "TARGET_KEY exactly. Do not infer or alter it.\n"
            )
            needle = f"\nTARGET_KEY={answer}\n"
            suffix = "\nQuestion: What is TARGET_KEY? Answer with the key only."
            prompt, actual = fit_prompt(count, target, prefix, needle, suffix, position)
            case = {
                "schema_version": 1,
                "id": f"text-niah-{target}-{position_tag}",
                "suite": "text-capability",
                "modality": "text",
                "target_context_tokens": target,
                "manifest_prompt_tokens": actual,
                "needle_position_fraction": position,
                "prompt": prompt,
                "expected": answer,
                "validation": "normalized_exact",
                "max_tokens": text_spec["capability_output_tokens"],
                "ignore_eos": False,
                "temperature": 0.0,
            }
            case["input_sha256"] = sha256_bytes(canonical_json(case))
            cases.append(case)

        prefix = (
            "Performance workload. Read the records, then emit the word "
            "benchmark repeatedly, separated by one space.\n"
        )
        middle = "\nPERFORMANCE_MARKER=QWEN38-4090\n"
        suffix = "\nBegin the repeated output now."
        prompt, actual = fit_prompt(count, target, prefix, middle, suffix, 0.5)
        case = {
            "schema_version": 1,
            "id": f"text-perf-{target}",
            "suite": "text-performance",
            "modality": "text",
            "target_context_tokens": target,
            "manifest_prompt_tokens": actual,
            "prompt": prompt,
            "validation": "nonempty",
            "max_tokens": text_spec["performance_output_tokens"],
            "ignore_eos": True,
            "temperature": 0.0,
        }
        case["input_sha256"] = sha256_bytes(canonical_json(case))
        cases.append(case)
    return cases


def font(size: int):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def draw_shape_count(path: Path) -> None:
    image = Image.new("RGB", (768, 512), "white")
    draw = ImageDraw.Draw(image)
    red_triangles = [(100, 100), (350, 110), (600, 100)]
    for x, y in red_triangles:
        draw.polygon([(x, y + 70), (x + 40, y), (x + 80, y + 70)], fill="#d62828", outline="black")
    for x, y in [(90, 300), (260, 300), (430, 300), (600, 300)]:
        draw.ellipse((x, y, x + 72, y + 72), fill="#247ba0", outline="black", width=3)
    for x, y in [(260, 205), (480, 205)]:
        draw.rectangle((x, y, x + 70, y + 70), fill="#2a9d8f", outline="black", width=3)
    image.save(path)


def draw_ocr(path: Path) -> None:
    image = Image.new("RGB", (1100, 300), "#f7f7f2")
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((35, 35, 1065, 265), radius=24, outline="#222222", width=5, fill="white")
    draw.text((85, 105), "APX-4090-Q38-1729", fill="#111111", font=font(64))
    image.save(path)


def draw_bar_chart(path: Path) -> None:
    image = Image.new("RGB", (768, 512), "white")
    draw = ImageDraw.Draw(image)
    draw.line((90, 420, 700, 420), fill="black", width=4)
    draw.line((90, 60, 90, 420), fill="black", width=4)
    values = {"A": 3, "B": 7, "C": 5, "D": 9}
    colors = ["#457b9d", "#e76f51", "#2a9d8f", "#f4a261"]
    for index, ((label, value), color) in enumerate(zip(values.items(), colors)):
        x0 = 140 + index * 135
        y0 = 420 - value * 36
        draw.rectangle((x0, y0, x0 + 80, 420), fill=color, outline="black", width=2)
        draw.text((x0 + 25, 435), label, fill="black", font=font(32))
        draw.text((x0 + 28, y0 - 40), str(value), fill="black", font=font(28))
    image.save(path)


def draw_spatial(path: Path) -> None:
    image = Image.new("RGB", (800, 360), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((120, 130, 260, 270), fill="#2979ff", outline="black", width=4)
    draw.text((128, 285), "blue square", fill="black", font=font(24))
    cx, cy, r1, r2 = 440, 200, 80, 34
    points = []
    for i in range(10):
        angle = -math.pi / 2 + i * math.pi / 5
        radius = r1 if i % 2 == 0 else r2
        points.append((cx + radius * math.cos(angle), cy + radius * math.sin(angle)))
    draw.polygon(points, fill="#ffd60a", outline="black")
    draw.text((375, 285), "yellow star", fill="black", font=font(24))
    draw.ellipse((610, 130, 750, 270), fill="#e63946", outline="black", width=4)
    draw.text((620, 285), "red circle", fill="black", font=font(24))
    image.save(path)


def draw_circle_panel(path: Path, count: int, label: str) -> None:
    image = Image.new("RGB", (640, 400), "white")
    draw = ImageDraw.Draw(image)
    draw.text((30, 25), label, fill="black", font=font(34))
    for index in range(count):
        row, col = divmod(index, 4)
        x, y = 80 + col * 130, 110 + row * 135
        draw.ellipse((x, y, x + 80, y + 80), fill="#d62828", outline="black", width=3)
    image.save(path)


def multimodal_cases(spec: dict[str, Any], output_dir: Path) -> list[dict[str, Any]]:
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    builders = {
        "shape-count.png": draw_shape_count,
        "ocr-code.png": draw_ocr,
        "bar-chart.png": draw_bar_chart,
        "spatial.png": draw_spatial,
    }
    for name, builder in builders.items():
        builder(image_dir / name)
    draw_circle_panel(image_dir / "compare-left.png", 3, "Panel One")
    draw_circle_panel(image_dir / "compare-right.png", 5, "Panel Two")

    max_tokens = spec["multimodal"]["output_tokens"]
    definitions = [
        (
            "mm-shape-count",
            ["images/shape-count.png"],
            "How many red triangles are in the image? Answer with one integer only.",
            "3",
            "normalized_exact",
            "counting",
        ),
        (
            "mm-ocr-code",
            ["images/ocr-code.png"],
            "Transcribe the large code exactly. Return only the code.",
            "APX-4090-Q38-1729",
            "normalized_exact",
            "ocr",
        ),
        (
            "mm-bar-chart",
            ["images/bar-chart.png"],
            "Which labeled bar has the largest value? Answer with the label only.",
            "D",
            "normalized_exact",
            "chart",
        ),
        (
            "mm-spatial",
            ["images/spatial.png"],
            "Which named shape is immediately to the left of the yellow star? Answer with the two-word label.",
            "blue square",
            "normalized_exact",
            "spatial",
        ),
        (
            "mm-two-image-difference",
            ["images/compare-left.png", "images/compare-right.png"],
            "How many more red circles are in Panel Two than Panel One? Answer with one integer only.",
            "2",
            "normalized_exact",
            "multi-image",
        ),
    ]
    cases = []
    for case_id, images, prompt, expected, validation, category in definitions:
        image_meta = []
        for relative in images:
            path = output_dir / relative
            with Image.open(path) as image:
                image_meta.append(
                    {
                        "path": relative,
                        "sha256": sha256_file(path),
                        "width": image.width,
                        "height": image.height,
                        "bytes": path.stat().st_size,
                    }
                )
        case = {
            "schema_version": 1,
            "id": case_id,
            "suite": "multimodal",
            "modality": "image-text",
            "category": category,
            "images": image_meta,
            "prompt": prompt,
            "expected": expected,
            "validation": validation,
            "max_tokens": max_tokens,
            "ignore_eos": False,
            "temperature": 0.0,
        }
        case["input_sha256"] = sha256_bytes(canonical_json(case))
        cases.append(case)
    return cases


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    args = parse_args()
    spec = read_json(args.spec)
    model_dir = args.model_dir or Path(spec["model"]["local_path"])
    tokenizer = load_tokenizer(model_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    text = text_cases(spec, tokenizer)
    multimodal = multimodal_cases(spec, args.output_dir)
    text_path = args.output_dir / "text.jsonl"
    multimodal_path = args.output_dir / "multimodal.jsonl"
    write_jsonl(text_path, text)
    write_jsonl(multimodal_path, multimodal)

    manifest = {
        "schema_version": 1,
        "spec_sha256": sha256_file(args.spec),
        "model_revision": spec["model"]["revision"],
        "tokenizer_path": str(model_dir / "tokenizer.json"),
        "text_cases": len(text),
        "multimodal_cases": len(multimodal),
        "text_jsonl_sha256": sha256_file(text_path),
        "multimodal_jsonl_sha256": sha256_file(multimodal_path),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
