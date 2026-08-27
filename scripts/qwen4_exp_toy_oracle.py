#!/usr/bin/env python3
"""Build one tiny official Qwen4-Exp checkpoint and compare ApxInf logits."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import Qwen4ExpForCausalLM, Qwen4ExpTextConfig


TRANSFORMERS_COMMIT = "19876312341f42cf49467bb24d67271cf28cb599"
DERIVED_SUFFIXES = (
    ".ple_embedding.layer_multipliers",
    ".ple_embedding.ngram_heads_offsets",
    ".ple_embedding.ngram_heads_vocab_sizes",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/qwen4-exp/synthetic-tiny.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(".apxinf/qwen4-exp-toy-oracle"),
    )
    parser.add_argument(
        "--rust-binary",
        type=Path,
        default=Path("target/release/examples/qwen4_exp_checkpoint"),
    )
    parser.add_argument("--tokens", default="1,3,5,7,9")
    parser.add_argument("--seed", type=int, default=38)
    parser.add_argument("--max-abs", type=float, default=1e-4)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def published_name(name: str) -> str:
    if name == "lm_head.weight":
        return name
    if not name.startswith("model."):
        raise RuntimeError(f"unexpected state_dict name {name}")
    return "model.language_model." + name.removeprefix("model.")


def published_state_dict(
    model: Qwen4ExpForCausalLM, split_ngram_parts: int
) -> dict[str, torch.Tensor]:
    converted: dict[str, torch.Tensor] = {}
    for name, tensor in model.state_dict().items():
        if name.endswith(DERIVED_SUFFIXES):
            continue
        if name.endswith(".ple_embedding.ngram_embedding.weight"):
            prefix = published_name(name).removesuffix(".weight")
            shards = torch.tensor_split(tensor, split_ngram_parts, dim=0)
            for index, shard in enumerate(shards):
                converted[f"{prefix}.shard_{index}.weight"] = shard.contiguous()
            continue
        converted[published_name(name)] = tensor.contiguous()
    return converted


def torch_logits(
    model: Qwen4ExpForCausalLM, tokens: list[int]
) -> tuple[torch.Tensor, torch.Tensor]:
    input_ids = torch.tensor([tokens], dtype=torch.long)
    with torch.inference_mode():
        one_shot = model(input_ids=input_ids, use_cache=False).logits[0].float().cpu()
        past = None
        steps = []
        for token in tokens:
            output = model(
                input_ids=torch.tensor([[token]], dtype=torch.long),
                past_key_values=past,
                use_cache=True,
            )
            past = output.past_key_values
            steps.append(output.logits[0, -1].float().cpu())
    return one_shot, torch.stack(steps)


def main() -> int:
    args = parse_args()
    tokens = [int(value) for value in args.tokens.split(",")]
    if not tokens or args.max_abs <= 0 or not args.rust_binary.is_file():
        raise SystemExit("tokens/max-abs/rust-binary contract is invalid")
    root = json.loads(args.config.read_text(encoding="utf-8"))
    text_config = dict(root["text_config"])
    config = Qwen4ExpTextConfig(**text_config)
    config._attn_implementation = "eager"
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    model = Qwen4ExpForCausalLM(config).eval()
    one_shot, stepwise = torch_logits(model, tokens)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    config_path = args.output_dir / "config.json"
    checkpoint_path = args.output_dir / "model.safetensors"
    report_path = args.output_dir / "report.json"
    config_path.write_text(json.dumps(root, sort_keys=True) + "\n", encoding="utf-8")
    state = published_state_dict(model, config.split_ngram_parts)
    save_file(state, checkpoint_path)

    command = [
        str(args.rust_binary.resolve()),
        str(config_path.resolve()),
        str(checkpoint_path.resolve()),
        ",".join(str(token) for token in tokens),
    ]
    completed = subprocess.run(command, check=True, text=True, capture_output=True)
    rust = json.loads(completed.stdout)
    rust_logits = torch.tensor(rust["logits"], dtype=torch.float32).reshape_as(stepwise)
    difference = (rust_logits - stepwise).abs()
    max_abs = difference.max().item()
    mean_abs = difference.mean().item()
    top1_equal = rust_logits.argmax(dim=-1).eq(stepwise.argmax(dim=-1))
    one_shot_stepwise_max_abs = (one_shot - stepwise).abs().max().item()
    finite = bool(torch.isfinite(rust_logits).all() and torch.isfinite(stepwise).all())
    checkpoint_path_observed = rust.get("generation_path", {}).get("weights") == "checkpoint"
    passed = (
        finite
        and max_abs <= args.max_abs
        and bool(top1_equal.all())
        and checkpoint_path_observed
    )
    report = {
        "format": "apxinf-qwen4-exp-toy-oracle-v1",
        "transformers_commit": TRANSFORMERS_COMMIT,
        "transformers_version": __import__("transformers").__version__,
        "torch_version": torch.__version__,
        "seed": args.seed,
        "tokens": tokens,
        "checkpoint": {
            "sha256": sha256(checkpoint_path),
            "bytes": checkpoint_path.stat().st_size,
            "tensor_count": len(state),
        },
        "comparison": {
            "finite": finite,
            "max_abs": max_abs,
            "mean_abs": mean_abs,
            "threshold": args.max_abs,
            "top1_equal": top1_equal.tolist(),
            "one_shot_vs_stepwise_max_abs": one_shot_stepwise_max_abs,
            "checkpoint_path_observed": checkpoint_path_observed,
        },
        "rust": {
            "checksum": rust["checksum"],
            "elapsed_ms": rust["elapsed_ms"],
        },
        "passed": passed,
    }
    if not all(math.isfinite(value) for value in (max_abs, mean_abs, one_shot_stepwise_max_abs)):
        report["passed"] = False
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
