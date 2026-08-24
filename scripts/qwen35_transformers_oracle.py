#!/usr/bin/env python3
"""Generate a pinned Transformers oracle for Qwen3.5-0.8B on CPU.

The output is split into a small JSON manifest and an F32 SafeTensors payload.
The companion Rust example consumes both files and feeds the raw token ids
directly to ApxInf, so tokenizer or chat-template differences cannot affect the
numeric comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from typing import Any


# These must be set before importing Transformers. In particular, disabling hub
# kernels is not equivalent to merely selecting eager full attention: Qwen3.5
# also has optional FLA and causal-conv implementations.
os.environ["USE_HUB_KERNELS"] = "0"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
import transformers
from safetensors import __version__ as safetensors_version
from safetensors.torch import save_file
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration


REPO_ID = "Qwen/Qwen3.5-0.8B"
LOCKED_REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
LOCKED_CHECKPOINT_SHA256 = (
    "04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696"
)
LOCKED_INPUT_IDS = [
    248045,
    846,
    198,
    9419,
    248046,
    198,
    248045,
    74455,
    198,
    248068,
    271,
    248069,
    271,
]
DEFAULT_PROBE_TOKEN_ID = 9419
DEFAULT_GREEDY_LENGTH = 10
MIN_GREEDY_LENGTH = 10
EXPECTED_VERSIONS = {
    "torch": "2.13.0",
    "transformers": "5.15.1",
    "safetensors": "0.8.0",
}
REQUIRED_SNAPSHOT_FILES = (
    "config.json",
    "model.safetensors.index.json",
    "model.safetensors-00001-of-00001.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
)
OPTIONAL_KERNEL_PACKAGES = ("fla", "causal_conv1d", "kernels", "flash_attn")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a CPU/FP32 official Transformers reference for the pinned "
            "Qwen/Qwen3.5-0.8B checkpoint."
        )
    )
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-reference", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument(
        "--revision",
        default=LOCKED_REVISION,
        help="Must remain the oracle's locked Hugging Face commit.",
    )
    parser.add_argument(
        "--input-ids",
        default=",".join(str(token) for token in LOCKED_INPUT_IDS),
        help="Comma-separated raw token ids. The locked non-thinking chat ids are the default.",
    )
    parser.add_argument("--probe-token-id", type=int, default=DEFAULT_PROBE_TOKEN_ID)
    parser.add_argument(
        "--greedy-length",
        type=int,
        default=DEFAULT_GREEDY_LENGTH,
        help=(
            "Number of generated tokens frozen into the exact greedy trajectory; "
            f"must be at least {MIN_GREEDY_LENGTH}."
        ),
    )
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace existing output files.",
    )
    return parser.parse_args()


def _parse_input_ids(raw: str) -> list[int]:
    try:
        ids = [int(part.strip()) for part in raw.split(",") if part.strip()]
    except ValueError as error:
        raise ValueError(f"invalid --input-ids: {error}") from error
    if not ids:
        raise ValueError("--input-ids must contain at least one token")
    if any(token < 0 for token in ids):
        raise ValueError("--input-ids cannot contain negative token ids")
    return ids


def _version_without_local_suffix(version: str) -> str:
    return version.split("+", maxsplit=1)[0]


def _verify_runtime() -> dict[str, str]:
    versions = {
        "torch": _version_without_local_suffix(torch.__version__),
        "transformers": transformers.__version__,
        "safetensors": safetensors_version,
    }
    if versions != EXPECTED_VERSIONS:
        raise RuntimeError(
            f"oracle runtime version mismatch: expected {EXPECTED_VERSIONS}, got {versions}"
        )
    installed_optional = [
        package for package in OPTIONAL_KERNEL_PACKAGES if importlib.util.find_spec(package)
    ]
    if installed_optional:
        raise RuntimeError(
            "the oracle requires Transformers' pure PyTorch fallbacks, but these optional "
            f"kernel packages are installed: {installed_optional}"
        )
    if os.environ.get("USE_HUB_KERNELS") != "0":
        raise RuntimeError("USE_HUB_KERNELS must be 0 before importing Transformers")
    return versions


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_blob_sha1(path: Path) -> str:
    digest = hashlib.sha1()
    size = path.stat().st_size
    digest.update(f"blob {size}\0".encode())
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_pinned_snapshot(model_dir: Path, revision: str) -> dict[str, Any]:
    if revision != LOCKED_REVISION:
        raise RuntimeError(
            f"this oracle is locked to {LOCKED_REVISION}; received revision {revision}"
        )
    model_dir = model_dir.resolve()
    tree_path = model_dir / ".cache" / "huggingface" / "trees" / f"{revision}.json"
    if not tree_path.is_file():
        raise RuntimeError(
            f"missing pinned snapshot provenance {tree_path}; use snapshot_download with "
            f"revision={revision!r} and local_dir={str(model_dir)!r}"
        )
    tree = json.loads(tree_path.read_text(encoding="utf-8"))
    files = tree.get("files")
    if tree.get("format_version") != 1 or not isinstance(files, dict):
        raise RuntimeError(f"invalid Hugging Face local snapshot tree: {tree_path}")

    verified: dict[str, dict[str, Any]] = {}
    for name in REQUIRED_SNAPSHOT_FILES:
        path = model_dir / name
        entry = files.get(name)
        if not path.is_file() or not isinstance(entry, dict):
            raise RuntimeError(f"pinned snapshot is missing required file {name}")
        actual_size = path.stat().st_size
        expected_size = entry.get("size")
        if actual_size != expected_size:
            raise RuntimeError(
                f"{name} size mismatch: expected {expected_size}, got {actual_size}"
            )
        lfs_sha256 = entry.get("lfs_sha256")
        if lfs_sha256:
            actual_hash = _sha256(path)
            if actual_hash != lfs_sha256:
                raise RuntimeError(
                    f"{name} SHA-256 mismatch: expected {lfs_sha256}, got {actual_hash}"
                )
            verified[name] = {
                "size": actual_size,
                "sha256": actual_hash,
            }
        else:
            actual_hash = _git_blob_sha1(path)
            expected_hash = entry.get("blob_id")
            if actual_hash != expected_hash:
                raise RuntimeError(
                    f"{name} git-blob hash mismatch: expected {expected_hash}, got {actual_hash}"
                )
            verified[name] = {
                "size": actual_size,
                "git_blob_sha1": actual_hash,
            }

    checkpoint_hash = verified[
        "model.safetensors-00001-of-00001.safetensors"
    ]["sha256"]
    if checkpoint_hash != LOCKED_CHECKPOINT_SHA256:
        raise RuntimeError(
            f"checkpoint hash mismatch: expected {LOCKED_CHECKPOINT_SHA256}, got {checkpoint_hash}"
        )
    return {
        "tree_path": str(tree_path),
        "verified_files": verified,
        "checkpoint_sha256": checkpoint_hash,
    }


def _verify_config_contract(model_dir: Path) -> dict[str, Any]:
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    text = config.get("text_config", {})
    expected = {
        "model_type": "qwen3_5",
        "hidden_size": 1024,
        "vocab_size": 248320,
        "num_hidden_layers": 24,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "head_dim": 256,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 16,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "linear_conv_kernel_dim": 4,
    }
    actual = {"model_type": config.get("model_type")}
    actual.update({key: text.get(key) for key in expected if key != "model_type"})
    if actual != expected:
        raise RuntimeError(f"Qwen3.5 config contract mismatch: expected {expected}, got {actual}")
    expected_layers = [
        "full_attention" if layer % 4 == 3 else "linear_attention" for layer in range(24)
    ]
    if text.get("layer_types") != expected_layers:
        raise RuntimeError("Qwen3.5 hybrid layer schedule does not match the locked 18/6 layout")
    return {**expected, "layer_types": expected_layers}


def _prepare_outputs(reference: Path, manifest: Path, force: bool) -> None:
    if reference.resolve() == manifest.resolve():
        raise ValueError("--output-reference and --output-manifest must be different paths")
    for path in (reference, manifest):
        if path.exists() and not force:
            raise FileExistsError(f"refusing to replace {path}; pass --force to overwrite it")
        path.parent.mkdir(parents=True, exist_ok=True)


def _tensor(value: torch.Tensor) -> torch.Tensor:
    return value.detach().to(device="cpu", dtype=torch.float32).contiguous().clone()


def _metrics(candidate: torch.Tensor, reference: torch.Tensor) -> dict[str, float | int]:
    candidate = _tensor(candidate).reshape(-1)
    reference = _tensor(reference).reshape(-1)
    if candidate.shape != reference.shape:
        raise ValueError(f"metric shape mismatch: {candidate.shape} != {reference.shape}")
    if not torch.isfinite(candidate).all() or not torch.isfinite(reference).all():
        raise ValueError("metric inputs must be finite")
    delta = candidate - reference
    count = delta.numel()
    mean_abs = delta.abs().mean().item()
    rmse = delta.square().mean().sqrt().item()
    reference_rms = reference.square().mean().sqrt().item()
    candidate_rms = candidate.square().mean().sqrt().item()
    denominator = candidate.norm().item() * reference.norm().item()
    cosine = (
        torch.dot(candidate, reference).item() / denominator
        if denominator != 0.0
        else float(torch.equal(candidate, reference))
    )
    return {
        "numel": count,
        "max_abs": delta.abs().max().item() if count else 0.0,
        "mean_abs": mean_abs,
        "rmse": rmse,
        "nrmse": rmse / max(reference_rms, sys.float_info.min),
        "cosine": cosine,
        "candidate_rms": candidate_rms,
        "reference_rms": reference_rms,
    }


def _capture_hidden(outputs: Any) -> torch.Tensor:
    hidden_states = outputs.hidden_states
    if hidden_states is None or len(hidden_states) != 25:
        raise RuntimeError(
            f"expected embedding plus 24 hidden states, got {None if hidden_states is None else len(hidden_states)}"
        )
    return torch.stack([_tensor(hidden[0, -1]) for hidden in hidden_states])


def _capture_cache(
    cache: Any,
    text_config: Any,
    prefix: str,
    tensors: dict[str, torch.Tensor],
) -> dict[str, Any]:
    sequence_length = int(cache.get_seq_length())
    if len(cache.layers) != text_config.num_hidden_layers:
        raise RuntimeError(
            f"cache has {len(cache.layers)} layers, expected {text_config.num_hidden_layers}"
        )
    key_width = text_config.linear_num_key_heads * text_config.linear_key_head_dim
    value_width = text_config.linear_num_value_heads * text_config.linear_value_head_dim
    summary: dict[str, Any] = {
        "sequence_length": sequence_length,
        "linear": {},
        "full": {},
    }
    for layer_index, layer_type in enumerate(text_config.layer_types):
        layer = cache.layers[layer_index]
        layer_name = f"{layer_index:02d}"
        if layer_type == "linear_attention":
            conv = layer.conv_states[0]
            recurrent = layer.recurrent_states[0]
            if conv is None or recurrent is None:
                raise RuntimeError(f"linear layer {layer_index} did not initialize its state")
            expected_conv_shape = (
                1,
                key_width * 2 + value_width,
                text_config.linear_conv_kernel_dim,
            )
            expected_recurrent_shape = (
                1,
                text_config.linear_num_value_heads,
                text_config.linear_key_head_dim,
                text_config.linear_value_head_dim,
            )
            if tuple(conv.shape) != expected_conv_shape:
                raise RuntimeError(
                    f"linear layer {layer_index} conv shape {tuple(conv.shape)} != {expected_conv_shape}"
                )
            if tuple(recurrent.shape) != expected_recurrent_shape:
                raise RuntimeError(
                    f"linear layer {layer_index} recurrent shape {tuple(recurrent.shape)} "
                    f"!= {expected_recurrent_shape}"
                )
            query, key, value = conv[0].split(
                [key_width, key_width, value_width], dim=0
            )
            names = {
                "q_conv": f"{prefix}.linear.{layer_name}.q_conv",
                "k_conv": f"{prefix}.linear.{layer_name}.k_conv",
                "v_conv": f"{prefix}.linear.{layer_name}.v_conv",
                "recurrent": f"{prefix}.linear.{layer_name}.recurrent",
            }
            tensors[names["q_conv"]] = _tensor(query.T)
            tensors[names["k_conv"]] = _tensor(key.T)
            tensors[names["v_conv"]] = _tensor(value.T)
            tensors[names["recurrent"]] = _tensor(recurrent[0])
            summary["linear"][layer_name] = {
                name: list(tensors[tensor_name].shape)
                for name, tensor_name in names.items()
            }
        elif layer_type == "full_attention":
            keys = layer.keys
            values = layer.values
            expected_shape = (
                1,
                text_config.num_key_value_heads,
                sequence_length,
                text_config.head_dim,
            )
            if keys is None or values is None:
                raise RuntimeError(f"full-attention layer {layer_index} did not initialize KV")
            if tuple(keys.shape) != expected_shape or tuple(values.shape) != expected_shape:
                raise RuntimeError(
                    f"full-attention layer {layer_index} KV shapes "
                    f"{tuple(keys.shape)}/{tuple(values.shape)} != {expected_shape}"
                )
            key_name = f"{prefix}.full.{layer_name}.key"
            value_name = f"{prefix}.full.{layer_name}.value"
            tensors[key_name] = _tensor(keys[0])
            tensors[value_name] = _tensor(values[0])
            summary["full"][layer_name] = {
                "key": list(tensors[key_name].shape),
                "value": list(tensors[value_name].shape),
            }
        else:
            raise RuntimeError(f"unexpected layer type {layer_type!r} at layer {layer_index}")
    return summary


def _top_tokens(
    logits: torch.Tensor, tokenizer: Any, top_k: int
) -> list[dict[str, Any]]:
    values, ids = torch.topk(_tensor(logits), k=top_k, sorted=True)
    output = []
    for token, value in zip(ids.tolist(), values.tolist(), strict=True):
        token_text = tokenizer.convert_ids_to_tokens(int(token))
        try:
            decoded = tokenizer.decode(
                [int(token)],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
        except (KeyError, TypeError, ValueError):
            decoded = None
        output.append(
            {
                "id": int(token),
                "logit": float(value),
                "token": token_text,
                "decoded": decoded,
            }
        )
    return output


def main() -> None:
    args = _parse_args()
    input_ids_list = _parse_input_ids(args.input_ids)
    if args.top_k <= 0:
        raise ValueError("--top-k must be greater than zero")
    if args.threads <= 0:
        raise ValueError("--threads must be greater than zero")
    if args.probe_token_id < 0:
        raise ValueError("--probe-token-id cannot be negative")
    if args.greedy_length < MIN_GREEDY_LENGTH:
        raise ValueError(
            f"--greedy-length cannot be below the frozen minimum {MIN_GREEDY_LENGTH}"
        )

    model_dir = args.model_dir.resolve()
    output_reference = args.output_reference.resolve()
    output_manifest = args.output_manifest.resolve()
    _prepare_outputs(output_reference, output_manifest, args.force)
    versions = _verify_runtime()
    snapshot = _verify_pinned_snapshot(model_dir, args.revision)
    config_contract = _verify_config_contract(model_dir)
    vocab_size = int(config_contract["vocab_size"])
    if any(token >= vocab_size for token in input_ids_list):
        raise ValueError(f"input token ids must be below vocabulary size {vocab_size}")
    if args.probe_token_id >= vocab_size:
        raise ValueError(f"probe token id must be below vocabulary size {vocab_size}")
    if args.top_k > vocab_size:
        raise ValueError(f"--top-k cannot exceed vocabulary size {vocab_size}")

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")

    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        local_files_only=True,
        use_fast=True,
    )
    locked_chat = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Hello"}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_dict=True,
        return_tensors="pt",
    )
    if locked_chat["input_ids"].tolist() != [LOCKED_INPUT_IDS]:
        raise RuntimeError(
            "the pinned tokenizer/chat template no longer produces the locked non-thinking ids"
        )
    if not bool(locked_chat["attention_mask"].all()):
        raise RuntimeError("the locked no-padding chat unexpectedly contains masked tokens")

    input_ids = torch.tensor([input_ids_list], dtype=torch.long, device="cpu")
    probe = torch.tensor([[args.probe_token_id]], dtype=torch.long, device="cpu")

    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        model_dir,
        local_files_only=True,
        dtype=torch.float32,
        attn_implementation="eager",
        use_kernels=False,
    ).eval()
    first_parameter = next(model.parameters())
    if first_parameter.device.type != "cpu" or first_parameter.dtype != torch.float32:
        raise RuntimeError(
            f"oracle model must be CPU/FP32, got {first_parameter.device}/{first_parameter.dtype}"
        )
    if model.config.text_config._attn_implementation != "eager":
        raise RuntimeError("Qwen3.5 text attention implementation is not eager")

    reference_tensors: dict[str, torch.Tensor] = {}
    cache_summaries: dict[str, Any] = {}
    with torch.inference_mode():
        prefill = model(
            input_ids=input_ids,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
            logits_to_keep=0,
        )
        expected_prefill_shape = (1, len(input_ids_list), vocab_size)
        if tuple(prefill.logits.shape) != expected_prefill_shape:
            raise RuntimeError(
                f"prefill logits shape {tuple(prefill.logits.shape)} != {expected_prefill_shape}"
            )
        reference_tensors["prefill.logits"] = _tensor(prefill.logits[0])
        reference_tensors["prefill.hidden_last_by_layer"] = _capture_hidden(prefill)
        cache_summaries["prefill"] = _capture_cache(
            prefill.past_key_values,
            model.config.text_config,
            "prefill",
            reference_tensors,
        )

        cached_probe = model(
            input_ids=probe,
            past_key_values=prefill.past_key_values,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
            logits_to_keep=0,
        )
        reference_tensors["cached_probe.logits"] = _tensor(cached_probe.logits[0])
        reference_tensors["cached_probe.hidden_last_by_layer"] = _capture_hidden(cached_probe)
        cache_summaries["cached_probe"] = _capture_cache(
            cached_probe.past_key_values,
            model.config.text_config,
            "cached_probe",
            reference_tensors,
        )

        # Text-only calls do not set rope_deltas, but clearing it makes the
        # fresh-path contract explicit if this script is later extended.
        model.model.rope_deltas = None
        fresh_ids = torch.cat([input_ids, probe], dim=1)
        fresh_probe = model(
            input_ids=fresh_ids,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
            logits_to_keep=0,
        )
        reference_tensors["fresh_probe.logits"] = _tensor(fresh_probe.logits[0])
        reference_tensors["fresh_probe.hidden_last_by_layer"] = _capture_hidden(fresh_probe)

        # Freeze a request-level trajectory in addition to the tensor-level
        # probes above. EOS stopping is deliberately disabled so the artifact
        # always binds exactly `greedy_length` next-token decisions.
        model.model.rope_deltas = None
        generated = model.generate(
            input_ids=input_ids,
            max_new_tokens=args.greedy_length,
            do_sample=False,
            use_cache=True,
            eos_token_id=None,
            pad_token_id=tokenizer.pad_token_id,
            return_dict_in_generate=False,
        )
        greedy_token_ids = generated[0, input_ids.shape[1] :].tolist()
        if len(greedy_token_ids) != args.greedy_length:
            raise RuntimeError(
                f"greedy generation returned {len(greedy_token_ids)} tokens, "
                f"expected exactly {args.greedy_length}"
            )

    prefill_last = reference_tensors["prefill.logits"][-1]
    cached_last = reference_tensors["cached_probe.logits"][-1]
    fresh_last = reference_tensors["fresh_probe.logits"][-1]
    cached_hidden = reference_tensors["cached_probe.hidden_last_by_layer"]
    fresh_hidden = reference_tensors["fresh_probe.hidden_last_by_layer"]
    hidden_layer_max_abs = (
        (cached_hidden - fresh_hidden).abs().amax(dim=1).tolist()
    )
    top_tokens = _top_tokens(prefill_last, tokenizer, args.top_k)
    if greedy_token_ids[0] != top_tokens[0]["id"]:
        raise RuntimeError(
            "model.generate first token disagrees with the direct prefill argmax"
        )

    metadata = {
        "format": "apxinf-qwen35-transformers-oracle-v1",
        "repo_id": REPO_ID,
        "revision": LOCKED_REVISION,
        "checkpoint_sha256": LOCKED_CHECKPOINT_SHA256,
        "torch_version": versions["torch"],
        "transformers_version": versions["transformers"],
        "safetensors_version": versions["safetensors"],
        "input_ids": json.dumps(input_ids_list, separators=(",", ":")),
        "probe_token_id": str(args.probe_token_id),
        "greedy_length": str(args.greedy_length),
        "greedy_token_ids": json.dumps(greedy_token_ids, separators=(",", ":")),
    }
    save_file(reference_tensors, output_reference, metadata=metadata)

    manifest = {
        "format": metadata["format"],
        "repo_id": REPO_ID,
        "revision": LOCKED_REVISION,
        "checkpoint_sha256": snapshot["checkpoint_sha256"],
        "runtime": {
            **versions,
            "device": "cpu",
            "dtype": "float32",
            "attention_implementation": "eager",
            "use_hub_kernels": False,
            "optional_kernel_packages": [],
            "threads": args.threads,
            "deterministic_algorithms": True,
        },
        "model_dir": str(model_dir),
        "reference_path": str(output_reference),
        "snapshot": snapshot,
        "config": config_contract,
        "input_ids": input_ids_list,
        "uses_locked_default_ids": input_ids_list == LOCKED_INPUT_IDS,
        "locked_chat": {
            "messages": [{"role": "user", "content": "Hello"}],
            "enable_thinking": False,
            "input_ids": LOCKED_INPUT_IDS,
            "rendered": tokenizer.decode(
                LOCKED_INPUT_IDS,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            ),
        },
        "probe_token_id": args.probe_token_id,
        "first_step": {
            "argmax_token_id": top_tokens[0]["id"],
            "top_tokens": top_tokens,
        },
        "greedy_trajectory": {
            "length": args.greedy_length,
            "minimum_length": MIN_GREEDY_LENGTH,
            "do_sample": False,
            "use_cache": True,
            "eos_stopping": False,
            "generated_ids": greedy_token_ids,
            "decoded": tokenizer.decode(
                greedy_token_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            ),
        },
        "official_internal_consistency": {
            "cached_probe_logits_vs_fresh_last": _metrics(cached_last, fresh_last),
            "cached_probe_hidden_vs_fresh_last_by_layer": _metrics(
                cached_hidden, fresh_hidden
            ),
            "hidden_layer_max_abs": hidden_layer_max_abs,
        },
        "cache": cache_summaries,
        "tensors": {
            name: {"shape": list(tensor.shape), "dtype": str(tensor.dtype)}
            for name, tensor in sorted(reference_tensors.items())
        },
    }
    output_manifest.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "reference": str(output_reference),
                "manifest": str(output_manifest),
                "input_tokens": len(input_ids_list),
                "first_step_argmax": top_tokens[0]["id"],
                "greedy_length": args.greedy_length,
                "greedy_token_ids": greedy_token_ids,
                "cached_vs_fresh": manifest["official_internal_consistency"][
                    "cached_probe_logits_vs_fresh_last"
                ],
            },
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
