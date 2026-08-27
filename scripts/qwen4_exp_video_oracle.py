#!/usr/bin/env python3
"""Compare tiny Qwen4-Exp video timestamp groups with Transformers."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import Qwen4ExpConfig, Qwen4ExpForConditionalGeneration

from qwen4_exp_multimodal_oracle import published_state


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/qwen4-exp/synthetic-tiny.json"))
    parser.add_argument("--output-dir", type=Path, default=Path(".apxinf/qwen4-exp-video-oracle"))
    parser.add_argument(
        "--rust-binary",
        type=Path,
        default=Path("target/release/examples/qwen4_exp_video"),
    )
    parser.add_argument(
        "--rust-vision-binary",
        type=Path,
        default=Path("target/release/examples/qwen4_exp_vision"),
    )
    parser.add_argument("--seed", type=int, default=38)
    parser.add_argument("--max-abs", type=float, default=1e-4)
    args = parser.parse_args()

    root = json.loads(args.config.read_text(encoding="utf-8"))
    root["text_config"]["indexer_budget"] = 16
    text_config = dict(root["text_config"])
    text_config["dtype"] = "float32"
    config = Qwen4ExpConfig(
        text_config=text_config,
        vision_config=root["vision_config"],
        image_token_id=root["image_token_id"],
        video_token_id=root["video_token_id"],
        vision_start_token_id=root["vision_start_token_id"],
        vision_end_token_id=root["vision_end_token_id"],
    )
    config._attn_implementation = "eager"
    config.text_config._attn_implementation = "eager"
    config.vision_config._attn_implementation = "eager"
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    model = Qwen4ExpForConditionalGeneration(config).eval()
    with torch.no_grad():
        for tensor in list(model.parameters()) + list(model.buffers()):
            if tensor.is_floating_point():
                tensor.copy_(tensor.to(torch.bfloat16).float())

    tokens = torch.tensor([[1, 29, 29, 29, 29, 2, 29, 29, 29, 29, 3]], dtype=torch.long)
    token_types = torch.tensor([[0, 2, 2, 2, 2, 0, 2, 2, 2, 2, 0]], dtype=torch.int)
    grid = torch.tensor([[2, 4, 4]], dtype=torch.long)
    patch_width = 3 * 2 * 2 * 2
    pixels = torch.arange(32 * patch_width, dtype=torch.float32).reshape(32, patch_width) / 100.0
    with torch.inference_mode():
        expected_vision = model.model.visual(pixels, grid).pooler_output.float().cpu()
        prefill = model(
            input_ids=tokens,
            pixel_values_videos=pixels,
            video_grid_thw=grid,
            mm_token_type_ids=token_types,
            use_cache=True,
        )
        decode = model(
            input_ids=torch.tensor([[4]], dtype=torch.long),
            past_key_values=prefill.past_key_values,
            use_cache=True,
        )
        expected = torch.cat([prefill.logits[0], decode.logits[0]], dim=0).float().cpu()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "config.json").write_text(json.dumps(root, sort_keys=True) + "\n", encoding="utf-8")
    state = published_state(model, config.text_config.split_ngram_parts)
    save_file(state, args.output_dir / "model.safetensors")
    completed = subprocess.run(
        [str(args.rust_binary.resolve()), str(args.output_dir.resolve())],
        check=True,
        text=True,
        capture_output=True,
    )
    rust = json.loads(completed.stdout)
    vision_completed = subprocess.run(
        [
            str(args.rust_vision_binary.resolve()),
            str((args.output_dir / "config.json").resolve()),
            str((args.output_dir / "model.safetensors").resolve()),
            "2,4,4",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    rust_vision = json.loads(vision_completed.stdout)
    actual_vision = torch.tensor(rust_vision["output"], dtype=torch.float32).reshape_as(expected_vision)
    vision_difference = (actual_vision - expected_vision).abs()
    actual = torch.tensor(rust["logits"], dtype=torch.float32).reshape_as(expected)
    difference = (actual - expected).abs()
    top1 = actual.argmax(-1).eq(expected.argmax(-1))
    path = rust.get("generation_path", {})
    max_abs = difference.max().item()
    passed = (
        bool(torch.isfinite(actual).all())
        and max_abs <= args.max_abs
        and bool(top1.all())
        and path.get("video") is True
        and path.get("rope_delta") == -4
    )
    report = {
        "format": "apxinf-qwen4-exp-video-oracle-v1",
        "tokens": tokens.tolist()[0],
        "grid_thw": grid.tolist(),
        "checkpoint_tensors": len(state),
        "indexer_budget": config.text_config.indexer_budget,
        "rope_delta": path.get("rope_delta"),
        "comparison": {
            "max_abs": max_abs,
            "mean_abs": difference.mean().item(),
            "row_max_abs": difference.max(dim=-1).values.tolist(),
            "threshold": args.max_abs,
            "top1_equal": top1.tolist(),
        },
        "vision_comparison": {
            "max_abs": vision_difference.max().item(),
            "mean_abs": vision_difference.mean().item(),
        },
        "passed": passed,
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
