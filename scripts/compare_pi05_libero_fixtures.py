#!/usr/bin/env python3
"""Compare real ApxInf LIBERO fixtures with an independent Mizar BF16 path."""

from __future__ import annotations

import argparse
import json
import os
import pathlib

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    parser.add_argument("--reference-output-dir", type=pathlib.Path)
    parser.add_argument("--min-cosine", type=float, default=0.997)
    parser.add_argument("--max-relative-l2", type=float, default=0.10)
    parser.add_argument("fixtures", nargs="+", type=pathlib.Path)
    return parser.parse_args()


def images_from_patches(patches: np.ndarray) -> np.ndarray:
    patches = np.asarray(patches, dtype=np.float32)
    if patches.shape != (512, 588):
        raise ValueError(f"expected [512,588] patches, got {patches.shape}")
    images = patches.reshape(2, 16, 16, 3, 14, 14)
    return images.transpose(0, 3, 1, 4, 2, 5).reshape(2, 3, 224, 224)


def metrics(actual: np.ndarray, expected: np.ndarray) -> dict:
    actual = np.asarray(actual, dtype=np.float64).reshape(-1)
    expected = np.asarray(expected, dtype=np.float64).reshape(-1)
    if actual.shape != expected.shape or actual.size == 0:
        raise ValueError(f"comparison shape mismatch: {actual.shape} versus {expected.shape}")
    if not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise FloatingPointError("comparison contains non-finite values")
    error = actual - expected
    actual_l2 = np.linalg.norm(actual)
    expected_l2 = np.linalg.norm(expected)
    cosine = (
        float(np.dot(actual, expected) / (actual_l2 * expected_l2))
        if actual_l2 and expected_l2
        else float(actual_l2 == expected_l2)
    )
    return {
        "cosine": cosine,
        "relative_l2": float(np.linalg.norm(error) / expected_l2) if expected_l2 else None,
        "max_abs": float(np.max(np.abs(error))),
        "mean_abs": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "actual_abs_checksum": float(np.abs(actual).sum()),
        "reference_abs_checksum": float(np.abs(expected).sum()),
    }


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.min_cosine <= 1.0:
        raise ValueError("--min-cosine must be in [0,1]")
    if args.max_relative_l2 < 0.0:
        raise ValueError("--max-relative-l2 must be non-negative")

    # Every supplied prefix token is valid, so these fast-path flags are
    # mathematically equivalent to the explicit OpenPI masks used by BF16.
    os.environ.setdefault("MIZAR_ASSUME_PREFIX_UNMASKED", "1")
    os.environ.setdefault("MIZAR_SKIP_LA_MASKS", "1")
    os.environ.setdefault("MIZAR_COMPACT_VALID_PREFIX", "1")
    os.environ.setdefault("MIZAR_SKIP_OPENPI_FP32_RESTORE", "0")
    os.environ.setdefault("MIZAR_DISABLE_PRECOMPUTED_STYLES", "1")
    os.environ.setdefault("MIZAR_FORCE_TORCH_ADARMS", "1")
    os.environ.setdefault("MIZAR_FORCE_TORCH_GATED_RESIDUAL", "1")
    os.environ.setdefault("MIZAR_FORCE_TORCH_LINEAR", "1")
    os.environ.setdefault("MIZAR_FORCE_TORCH_SDPA", "1")
    os.environ.setdefault("MIZAR_USE_KN_GEMM", "0")
    os.environ.setdefault("MIZAR_USE_KN_GEMM_ATTN", "0")

    from mizar_robo.model_executor.models.pi05.model import Pi05Model
    from mizar_robo.model_executor.preprocessing.vla_preprocessor import PreprocessedData

    # Mizar currently pre-initializes a custom attention workspace even when
    # use_mizar_kernel=False. The independent BF16 path below uses PyTorch SDPA,
    # so bypass that unrelated custom-op probe (some reference images do not
    # ship the attention_qkv operator).
    Pi05Model._preinit_cublaslt_workspace = lambda self: None

    overrides = {
        "dtype": "bfloat16",
        "compute_dtype": "bfloat16",
        "execution_mode": "separated",
        "action_horizon": 10,
        "discrete_state_input": False,
        "attn_impl": "sdpa",
        "quantize": "off",
        "use_compile": False,
        "use_pipeline": False,
        "compile_v_stage": False,
        "compile_l_stage": False,
        "compile_a_stage": False,
        "enable_prefix_cache": False,
        "use_mizar_kernel": False,
        "fuse_qkv": False,
        "fuse_qkv_rope_kvcache": False,
        "use_decomposed_attn": False,
        "use_strided_fmha": False,
        "verbose_timing": False,
    }
    model = Pi05Model.from_pretrained(
        str(args.model_dir),
        device="cuda:0",
        dtype="bfloat16",
        num_warmup_steps=0,
        config_overrides=overrides,
    )

    comparisons = []
    all_passed = True
    with torch.inference_mode():
        for path in args.fixtures:
            fixture = np.load(path, allow_pickle=False)
            images = images_from_patches(fixture["patches"])
            tokens = np.asarray(fixture["tokens"], dtype=np.int32)
            noise = np.asarray(fixture["noise"], dtype=np.float32)
            apxinf_actions = np.asarray(fixture["normalized_actions"], dtype=np.float32)
            if not 0 < tokens.size <= 200:
                raise ValueError(f"invalid fixture token count {tokens.size} in {path}")

            preprocessed = PreprocessedData(
                images={
                    "base_0_rgb": torch.from_numpy(images[0:1]).to("cuda:0"),
                    "left_wrist_0_rgb": torch.from_numpy(images[1:2]).to("cuda:0"),
                },
                image_masks={
                    "base_0_rgb": torch.ones(1, dtype=torch.bool, device="cuda:0"),
                    "left_wrist_0_rgb": torch.ones(1, dtype=torch.bool, device="cuda:0"),
                },
                state=torch.zeros(1, 32, dtype=torch.float32, device="cuda:0"),
                tokenized_prompt=torch.from_numpy(tokens[None]).to("cuda:0"),
                tokenized_prompt_mask=torch.ones(
                    1, tokens.size, dtype=torch.bool, device="cuda:0"
                ),
                prefix_cache_key=None,
            )
            model._model._external_noise = torch.from_numpy(noise[None]).to("cuda:0")
            reference = model.sample_actions_from_preprocessed(preprocessed, num_flow_steps=10)[0]
            if args.reference_output_dir is not None:
                args.reference_output_dir.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    args.reference_output_dir / path.name,
                    task_id=fixture["task_id"],
                    trial_id=fixture["trial_id"],
                    tokens=tokens,
                    noise=np.asarray(fixture["noise"]),
                    patches=np.asarray(fixture["patches"]),
                    normalized_actions=reference,
                )
            measured = metrics(apxinf_actions, reference)
            relative_l2 = measured["relative_l2"]
            passed = measured["cosine"] >= args.min_cosine and (
                relative_l2 is not None and relative_l2 <= args.max_relative_l2
            )
            measured.update(
                {
                    "fixture": str(path),
                    "task_id": int(fixture["task_id"]),
                    "trial_id": int(fixture["trial_id"]),
                    "token_count": int(tokens.size),
                    "passed": passed,
                }
            )
            comparisons.append(measured)
            all_passed &= passed
            print(
                f"{path.name}: cosine={measured['cosine']:.9f} "
                f"relative_l2={relative_l2:.6f} passed={passed}",
                flush=True,
            )
    model._model._external_noise = None

    result = {
        "schema": "apxinf.pi05.libero-bf16-parity.v1",
        "reference": "Mizar BF16 mathematical path",
        "model_dir": str(args.model_dir),
        "thresholds": {
            "minimum_cosine": args.min_cosine,
            "maximum_relative_l2": args.max_relative_l2,
        },
        "passed": all_passed,
        "comparisons": comparisons,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if not all_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
