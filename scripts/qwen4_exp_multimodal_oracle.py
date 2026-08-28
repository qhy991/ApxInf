#!/usr/bin/env python3
"""Compare full tiny Qwen4-Exp image-text prefill with Transformers."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import Qwen4ExpConfig, Qwen4ExpForConditionalGeneration


DERIVED_SUFFIXES = (
    ".ple_embedding.layer_multipliers",
    ".ple_embedding.ngram_heads_offsets",
    ".ple_embedding.ngram_heads_vocab_sizes",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/qwen4-exp/synthetic-tiny.json"))
    parser.add_argument("--output-dir", type=Path, default=Path(".apxinf/qwen4-exp-multimodal-oracle"))
    parser.add_argument(
        "--rust-binary",
        type=Path,
        default=Path("target/release/examples/qwen4_exp_multimodal"),
    )
    parser.add_argument("--seed", type=int, default=38)
    parser.add_argument("--max-abs", type=float, default=1e-4)
    return parser.parse_args()


def published_state(model, split_parts: int) -> dict[str, torch.Tensor]:
    output = {}
    for name, tensor in model.state_dict().items():
        if name.endswith(DERIVED_SUFFIXES):
            continue
        stored = tensor.to(torch.bfloat16) if tensor.is_floating_point() else tensor
        if name.endswith(".ple_embedding.ngram_embedding.weight"):
            prefix = name.removesuffix(".weight")
            for index, shard in enumerate(torch.tensor_split(stored, split_parts, dim=0)):
                output[f"{prefix}.shard_{index}.weight"] = shard.contiguous()
        elif tensor.is_floating_point():
            output[name] = stored.contiguous()
    return output


def main() -> int:
    args = parse_args()
    root = json.loads(args.config.read_text(encoding="utf-8"))
    oracle_text_config = dict(root["text_config"])
    oracle_text_config["dtype"] = "float32"
    config = Qwen4ExpConfig(
        text_config=oracle_text_config,
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

    tokens = torch.tensor([[1, 28, 28, 28, 28, 3]], dtype=torch.long)
    token_types = torch.tensor([[0, 1, 1, 1, 1, 0]], dtype=torch.int)
    grid = torch.tensor([[1, 4, 4]], dtype=torch.long)
    patch_width = 3 * 2 * 2 * 2
    pixels = torch.arange(16 * patch_width, dtype=torch.float32).reshape(16, patch_width) / 100.0
    with torch.inference_mode():
        prefill = model(
            input_ids=tokens,
            pixel_values=pixels,
            image_grid_thw=grid,
            mm_token_type_ids=token_types,
            use_cache=True,
        )
        decode = model(
            input_ids=torch.tensor([[4]], dtype=torch.long),
            past_key_values=prefill.past_key_values,
            use_cache=True,
        )
        expected = torch.cat([prefill.logits[0], decode.logits[0]], dim=0).float().cpu()
        expected_generation = torch.cat(
            [prefill.logits[0, -1:], decode.logits[0]], dim=0
        ).float().cpu()

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
    actual = torch.tensor(rust["logits"], dtype=torch.float32).reshape_as(expected)
    generation_actual = torch.tensor(
        rust["generation_logits"], dtype=torch.float32
    ).reshape_as(expected_generation)
    difference = (actual - expected).abs()
    generation_difference = (generation_actual - expected_generation).abs()
    path = rust.get("generation_path", {})
    generation_path = rust.get("generation_prefill_path", {})
    top1 = actual.argmax(-1).eq(expected.argmax(-1))
    generation_top1 = generation_actual.argmax(-1).eq(expected_generation.argmax(-1))
    max_abs = difference.max().item()
    passed = (
        bool(torch.isfinite(actual).all())
        and max_abs <= args.max_abs
        and bool(top1.all())
        and path.get("multimodal_prefill") is True
        and path.get("checkpoint_payloads_mmap") is True
        and path.get("rope_delta") == -2
        and generation_difference.max().item() <= args.max_abs
        and bool(generation_top1.all())
        and rust.get("generation_logits_shape") == [2, expected.shape[-1]]
        and generation_path.get("generation_prefill_logits_rows") == 1
        and generation_path.get("rope_delta") == -2
    )
    report = {
        "format": "apxinf-qwen4-exp-multimodal-oracle-v1",
        "seed": args.seed,
        "tokens": tokens.tolist()[0],
        "grid_thw": grid.tolist(),
        "checkpoint_tensors": len(state),
        "comparison": {
            "max_abs": max_abs,
            "mean_abs": difference.mean().item(),
            "threshold": args.max_abs,
            "top1_equal": top1.tolist(),
        },
        "rope_delta": path.get("rope_delta"),
        "generation_prefill_comparison": {
            "max_abs": generation_difference.max().item(),
            "mean_abs": generation_difference.mean().item(),
            "top1_equal": generation_top1.tolist(),
            "logits_rows": rust.get("generation_logits_shape", [None])[0],
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
