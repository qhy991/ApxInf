#!/usr/bin/env python3
"""Compare ApxInf's tiny Qwen4-Exp vision encoder with Transformers."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import Qwen4ExpVisionConfig, Qwen4ExpVisionModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/qwen4-exp/synthetic-tiny.json"))
    parser.add_argument("--output-dir", type=Path, default=Path(".apxinf/qwen4-exp-vision-oracle"))
    parser.add_argument(
        "--rust-binary",
        type=Path,
        default=Path("target/release/examples/qwen4_exp_vision"),
    )
    parser.add_argument("--seed", type=int, default=38)
    parser.add_argument("--max-abs", type=float, default=1e-4)
    parser.add_argument("--grid", default="1,2,2")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.rust_binary.is_file() or args.max_abs <= 0:
        raise SystemExit("rust-binary/max-abs contract is invalid")
    root = json.loads(args.config.read_text(encoding="utf-8"))
    config = Qwen4ExpVisionConfig(**root["vision_config"])
    config._attn_implementation = "eager"
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    model = Qwen4ExpVisionModel(config).eval()
    with torch.no_grad():
        for tensor in list(model.parameters()) + list(model.buffers()):
            if tensor.is_floating_point():
                tensor.copy_(tensor.to(torch.bfloat16).float())

    grid_values = [int(value) for value in args.grid.split(",")]
    if len(grid_values) != 3:
        raise SystemExit("--grid must be T,H,W")
    patches = math.prod(grid_values)
    patch_width = (
        config.in_channels
        * config.temporal_patch_size
        * config.patch_size
        * config.patch_size
    )
    pixels = torch.arange(patches * patch_width, dtype=torch.float32).reshape(patches, patch_width) / 100.0
    grid = torch.tensor([grid_values], dtype=torch.long)
    with torch.inference_mode():
        expected = model(pixels, grid).pooler_output.float().cpu()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    config_path = args.output_dir / "config.json"
    checkpoint_path = args.output_dir / "model.safetensors"
    report_path = args.output_dir / "report.json"
    config_path.write_text(json.dumps(root, sort_keys=True) + "\n", encoding="utf-8")
    state = {
        "model.visual." + name: tensor.to(torch.bfloat16).contiguous()
        for name, tensor in model.state_dict().items()
        if tensor.is_floating_point()
    }
    save_file(state, checkpoint_path)
    completed = subprocess.run(
        [
            str(args.rust_binary.resolve()),
            str(config_path),
            str(checkpoint_path),
            args.grid,
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    rust = json.loads(completed.stdout)
    actual = torch.tensor(rust["output"], dtype=torch.float32).reshape_as(expected)
    difference = (actual - expected).abs()
    max_abs = difference.max().item()
    mean_abs = difference.mean().item()
    passed = bool(torch.isfinite(actual).all()) and max_abs <= args.max_abs
    report = {
        "format": "apxinf-qwen4-exp-vision-oracle-v1",
        "seed": args.seed,
        "grid_thw": grid.tolist(),
        "checkpoint_tensors": len(state),
        "comparison": {
            "max_abs": max_abs,
            "mean_abs": mean_abs,
            "threshold": args.max_abs,
        },
        "passed": passed,
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
