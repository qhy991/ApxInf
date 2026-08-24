from __future__ import annotations

from contextlib import contextmanager, nullcontext, redirect_stderr, redirect_stdout
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import socket
import struct
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "scripts/build_mlx_bundle.py"
SPEC = importlib.util.spec_from_file_location("build_mlx_bundle", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

POLICY_MODULE_PATH = ROOT / "scripts/mlx_mixed_quant_policy.py"
POLICY_SPEC = importlib.util.spec_from_file_location(
    "mlx_mixed_quant_policy_for_builder_tests", POLICY_MODULE_PATH
)
assert POLICY_SPEC is not None and POLICY_SPEC.loader is not None
POLICY_MODULE = importlib.util.module_from_spec(POLICY_SPEC)
sys.modules[POLICY_SPEC.name] = POLICY_MODULE
POLICY_SPEC.loader.exec_module(POLICY_MODULE)


PINNED = {
    "mlx": "0.32.1",
    "mlx-metal": "0.32.1",
    "mlx-lm": "0.31.3",
    "transformers": "5.15.1",
    "safetensors": "0.8.0",
    "tokenizers": "0.22.2",
    "huggingface-hub": "1.28.0",
    "numpy": "2.5.2",
}
HYBRID_PRESET = "qwen35-0.8b-affine-w8-g64-gdn-outproj-parity-v1"
HYBRID_PRESET_V2 = "qwen35-0.8b-affine-w8-g64-gdn-outproj-async-chat-parity-v2"
HYBRID_COUNTERFACTUAL_PRESET_V3 = (
    "qwen35-0.8b-affine-w8-g64-gdn3-l19-o-proj-chinese-counterfactual-v3"
)
HYBRID_REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
ASYNC_CHAT_PROMPT_IDS = [
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
FIXTURE_ASYNC_TEACHER_IDS = list(range(128))
SELECTIVE_TEACHER_IDS = json.loads(
    (
        ROOT / "doc/20260823-qwen35-macos-bringup/"
        "qwen35-0.8b-mlx-w8-g64-async-chat-parity-policy-evidence-v2.json"
    ).read_text(encoding="utf-8")
)["policy"]["quality_gate"]["teacher_token_ids"]
SELECTIVE_QUALITY_SUITE_SHA256 = "4" * 64


def write_safetensors(
    path: Path,
    tensors: dict[str, tuple[str, list[int]]],
) -> None:
    header: dict[str, object] = {}
    cursor = 0
    for name, (dtype, shape) in tensors.items():
        logical_elements = 1
        for dimension in shape:
            logical_elements *= dimension
        tensor_bytes = logical_elements * MODULE.SAFETENSORS_DTYPE_BYTES[dtype]
        header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [cursor, cursor + tensor_bytes],
        }
        cursor += tensor_bytes
    payload = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
    path.write_bytes(struct.pack("<Q", len(payload)) + payload + bytes(cursor))


def write_index(path: Path, shard: str, tensor_names: list[str]) -> None:
    path.write_text(
        json.dumps(
            {
                "metadata": {"total_size": 0},
                "weight_map": {name: shard for name in sorted(tensor_names)},
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def make_source(root: Path) -> Path:
    source = root / "source"
    source.mkdir()
    source_config = {
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "model_type": "qwen3_5",
        "text_config": {
            "dtype": "bfloat16",
            "model_type": "qwen3_5_text",
        },
    }
    (source / "config.json").write_text(
        json.dumps(source_config, sort_keys=True), encoding="utf-8"
    )
    template = "{% for message in messages %}{{ message.content }}{% endfor %}"
    (source / "chat_template.jinja").write_text(template, encoding="utf-8")
    (source / "tokenizer.json").write_bytes(b'{"fixture":"exact bytes"}\n')
    (source / "tokenizer_config.json").write_text(
        json.dumps(
            {"chat_template": template, "tokenizer_class": "Qwen2Tokenizer"},
            sort_keys=True,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    tensors = {
        "model.language_model.layers.0.weight": ("BF16", [2, 64]),
        "model.language_model.layers.0.norm.weight": ("F32", [2]),
        "model.language_model.layers.0.linear_attn.conv1d.weight": (
            "BF16",
            [2, 1, 4],
        ),
    }
    write_safetensors(source / "model.safetensors", tensors)
    write_index(
        source / "model.safetensors.index.json",
        "model.safetensors",
        list(tensors),
    )
    return source.resolve()


def make_hybrid_source(root: Path) -> Path:
    source = make_source(root)
    tensors = {
        "model.language_model.layers.0.weight": ("BF16", [2, 64]),
        "model.language_model.layers.0.norm.weight": ("F32", [2]),
        "model.language_model.layers.0.linear_attn.conv1d.weight": (
            "BF16",
            [2, 1, 4],
        ),
        "model.language_model.layers.12.linear_attn.out_proj.weight": (
            "BF16",
            [2, 64],
        ),
    }
    write_safetensors(source / "model.safetensors", tensors)
    write_index(
        source / "model.safetensors.index.json", "model.safetensors", list(tensors)
    )
    return source


def make_hybrid_v2_source(root: Path) -> Path:
    source = make_source(root)
    tensors = {
        "model.language_model.layers.0.weight": ("BF16", [2, 64]),
        "model.language_model.layers.0.norm.weight": ("F32", [2]),
        "model.language_model.layers.0.linear_attn.conv1d.weight": (
            "BF16",
            [2, 1, 4],
        ),
        **{
            f"model.language_model.layers.{layer}.linear_attn.out_proj.weight": (
                "BF16",
                [2, 64],
            )
            for layer in (12, 14, 20)
        },
    }
    write_safetensors(source / "model.safetensors", tensors)
    write_index(
        source / "model.safetensors.index.json", "model.safetensors", list(tensors)
    )
    return source


def make_hybrid_counterfactual_v3_source(root: Path) -> Path:
    source = make_hybrid_v2_source(root)
    tensors = MODULE._parse_safetensors_schema(
        source / "model.safetensors", "fixture counterfactual source"
    )
    rewritten = {
        name: (dtype, list(shape)) for name, (dtype, shape) in tensors.items()
    }
    rewritten[
        "model.language_model.layers.19.self_attn.o_proj.weight"
    ] = ("BF16", [2, 64])
    write_safetensors(source / "model.safetensors", rewritten)
    write_index(
        source / "model.safetensors.index.json",
        "model.safetensors",
        list(rewritten),
    )
    return source


def make_selective_source(root: Path) -> Path:
    source = make_source(root)
    tensors = {
        "model.language_model.layers.0.mlp.down_proj.weight": (
            "BF16",
            [2, 64],
        ),
        "model.language_model.layers.0.self_attn.q_proj.weight": (
            "BF16",
            [2, 64],
        ),
        "model.language_model.layers.0.self_attn.v_proj.weight": (
            "BF16",
            [2, 64],
        ),
        "model.language_model.layers.0.norm.weight": ("F32", [2]),
        "model.language_model.layers.0.linear_attn.conv1d.weight": (
            "BF16",
            [2, 1, 4],
        ),
    }
    write_safetensors(source / "model.safetensors", tensors)
    write_index(
        source / "model.safetensors.index.json",
        "model.safetensors",
        list(tensors),
    )
    return source


def bound_selective_observation(
    document: dict[str, object],
    generated: list[int],
    *,
    outcome: str,
    path: str | None,
) -> dict[str, object]:
    policy = document["policy"]
    trace = policy["trace"]
    teacher = list(trace["teacher_token_ids"])
    evaluator = {
        "api": trace["api"],
        "semantics": trace["semantics"],
        "prompt_token_ids": trace["prompt_token_ids"],
        "teacher_forced_token_ids": [list(generated), list(generated)],
        "async_free_run_token_ids": [list(generated), list(generated)],
    }
    inputs = {
        "source_manifest_sha256": policy["source"]["source_manifest_sha256"],
        "config_sha256": policy["source"]["config_sha256"],
        "language_schema_sha256": policy["source"]["language_schema_sha256"],
        "policy_artifact_sha256": hashlib.sha256(
            POLICY_MODULE.canonical_bytes(document) + b"\n"
        ).hexdigest(),
        "policy_document_sha256": POLICY_MODULE.object_sha256(document),
        "policy_sha256": document["policy_sha256"],
        "search_receipt_sha256": document["search_receipt_sha256"],
        "candidate_modules_sha256": policy["candidate_modules_sha256"],
        "trace_sha256": POLICY_MODULE.object_sha256(trace),
        "quality_suite_sha256": policy["quality_suite_sha256"],
    }
    artifacts = [
        {
            "path": "scripts/run_mlx_mixed_quant_quality.py",
            "size": 123,
            "sha256": "5" * 64,
        }
    ]
    exact = outcome == "exact"
    transition = None
    counterfactuals = []
    current_screen = None
    counterfactual_screens = []
    selected_counterfactual = None
    if path is not None:
        overrides = {
            override["path"]: override["tier"]
            for override in policy["quantization"]["overrides"]
        }
        current_tier = overrides.get(path, "w4")
        next_tier = "w8" if current_tier == "w4" else "bf16"
        transition = {"path": path, "from": current_tier, "to": next_tier}
        manifest_sha256 = "a" * 64

        def screen(selected_score: int) -> dict[str, object]:
            scores = []
            for candidate in policy["candidate_modules"]:
                hidden_error = selected_score if candidate["path"] == path else 10
                scores.append(
                    {
                        "path": candidate["path"],
                        "hidden_error_ppm": hidden_error,
                        "top1_margin_erosion_ppm": 0,
                        "top1_flip_rate_ppm": 0,
                        "score_ppm": hidden_error,
                    }
                )
            return {
                "format": "apxinf-mlx-mixed-quant-state-aligned-screen-v1",
                "steps": 32,
                "state_alignment": "prompt-plus-bf16-teacher-prefix-v1",
                "aggregate_score_ppm": sum(item["score_ppm"] for item in scores),
                "module_scores": scores,
            }

        def analysis(runs: list[list[int]]) -> dict[str, object]:
            mismatch_count = sum(
                actual != expected
                for run in runs
                for actual, expected in zip(run, teacher, strict=True)
            )
            first = next(
                (
                    index
                    for index, (actual, expected) in enumerate(
                        zip(runs[0], teacher, strict=True)
                    )
                    if actual != expected
                ),
                None,
            )
            return {
                "teacher_forced_exact": mismatch_count == 0,
                "async_free_run_exact": mismatch_count == 0,
                "repeated_identically": runs[0] == runs[1],
                "teacher_forced_mismatch_count": mismatch_count,
                "async_free_run_mismatch_count": mismatch_count,
                "mismatch_count": mismatch_count * 2,
                "teacher_forced_first_divergence_step": first,
                "async_free_run_first_divergence_step": first,
                "teacher_forced_repeat_sha256": [
                    POLICY_MODULE.object_sha256(run) for run in runs
                ],
                "async_free_run_repeat_sha256": [
                    POLICY_MODULE.object_sha256(run) for run in runs
                ],
            }

        current_screen = screen(30)
        selected_screen = screen(5)
        counterfactual_screens = [
            {
                "path": path,
                "transition": transition,
                "manifest_sha256": manifest_sha256,
                "screen": selected_screen,
                "screen_improvement_ppm": (
                    current_screen["aggregate_score_ppm"]
                    - selected_screen["aggregate_score_ppm"]
                ),
            }
        ]
        selected_runs = [teacher, teacher]
        current_analysis = analysis([list(generated), list(generated)])
        selected_analysis = analysis(selected_runs)
        selected_counterfactual = {
            "path": path,
            "transition": transition,
            "manifest_sha256": manifest_sha256,
            "teacher_forced_token_ids": selected_runs,
            "async_free_run_token_ids": selected_runs,
            "analysis": selected_analysis,
            "mismatch_improvement": (
                current_analysis["mismatch_count"] - selected_analysis["mismatch_count"]
            ),
            "teacher_async_no_regression": True,
        }
        counterfactuals = [
            {
                "path": path,
                "manifest_sha256": manifest_sha256,
                "screening_manifest_sha256": manifest_sha256,
                "transition": transition,
            }
        ]
    body = {
        "format": POLICY_MODULE.RUNNER_RECEIPT_FORMAT,
        "passed": exact,
        "outcome": outcome,
        "inputs": inputs,
        "input_sha256": POLICY_MODULE.object_sha256(inputs),
        "program": {
            "artifacts": artifacts,
            "program_sha256": POLICY_MODULE.object_sha256(artifacts),
        },
        "runtime": {
            "python_executable_sha256": "6" * 64,
            "python_version": "3.14.0",
            "packages": [{"name": "mlx", "version": "0.32.1", "sha256": "7" * 64}],
            "offline": True,
            "network_blocked": True,
            "trust_remote_code": False,
        },
        "bundles": {
            "bf16_reference": {"manifest_sha256": "8" * 64},
            "current_candidate": {"manifest_sha256": "9" * 64},
            "counterfactuals": counterfactuals,
            "materialization": "independent-saved-static-verified-reload-v1",
            "dynamic_module_replacement": False,
            "model_bundle_published": False,
        },
        "evaluation": {
            "bf16_reference": {
                "teacher_forced_token_ids": [teacher, teacher],
                "async_free_run_token_ids": [teacher, teacher],
            },
            "current_candidate": {
                "teacher_forced_token_ids": [list(generated), list(generated)],
                "async_free_run_token_ids": [list(generated), list(generated)],
            },
            "attribution": {
                "screening_steps": 32,
                "teacher_forced_token_ids": [
                    generated[:32],
                    generated[:32],
                ],
                "async_free_run_token_ids": [
                    generated[:32],
                    generated[:32],
                ],
                "current_screen": current_screen,
                "counterfactual_screens": counterfactual_screens,
                "selected_counterfactual": selected_counterfactual,
            },
        },
        "decision": {
            "outcome": outcome,
            "stop_reason": None,
            "changed_module_count": 0 if path is None else 1,
            "changed_module_path": path,
            "transition": transition,
            "exact_trajectory_claim": exact,
            "general_parity_claim": False,
            "default_ready_claim": False,
            "formal_performance_claim": False,
        },
    }
    body_sha256 = POLICY_MODULE.object_sha256(body)
    return {
        "format": POLICY_MODULE.OBSERVATION_FORMAT,
        "policy_sha256": document["policy_sha256"],
        "trace_sha256": POLICY_MODULE.object_sha256(trace),
        "quality_suite_sha256": policy["quality_suite_sha256"],
        "evaluator": evaluator,
        "localization": (
            {
                "method": "state-aligned-single-module-attribution-v1",
                "scope": "one-module-counterfactual-no-combinations",
                "screening_steps": 32,
                "gate_steps": 128,
                "grouping": "layer-family-v1",
                "ranking_metric": "hidden-error-plus-top1-margin-v1",
                "sensitive_module_path": path,
                "unique_top_candidate": True,
                "runner_receipt_sha256": body_sha256,
            }
            if path is not None
            else None
        ),
        "runner_receipt_body": body,
        "runner_receipt_sha256": body_sha256,
    }


def write_selective_policy(path: Path, source: Path, *, advance: bool = True) -> str:
    inspected = MODULE._inspect_source(str(source))
    schema = MODULE._parse_safetensors_schema(
        source / "model.safetensors", "fixture selective source"
    )
    source_contract = {
        "repo_id": "Qwen/Qwen3.5-0.8B",
        "revision": HYBRID_REVISION,
        "source_manifest_sha256": MODULE._manifest_sha256(inspected.records),
        "config_sha256": hashlib.sha256(
            (source / "config.json").read_bytes()
        ).hexdigest(),
        "language_schema_sha256": MODULE._language_schema_sha256(schema),
        "language_tensor_count": len(MODULE._canonical_language_schema(schema)),
    }
    candidates = [
        {
            "path": "language_model.model.layers.0.mlp.down_proj",
            "dtype": "BF16",
            "shape": [2, 64],
        },
        {
            "path": "language_model.model.layers.0.self_attn.q_proj",
            "dtype": "BF16",
            "shape": [2, 64],
        },
        {
            "path": "language_model.model.layers.0.self_attn.v_proj",
            "dtype": "BF16",
            "shape": [2, 64],
        },
    ]
    teacher = list(SELECTIVE_TEACHER_IDS)
    trace = {
        "api": "mlx_lm.generate.generate_step",
        "semantics": "mlx-generate-step-argmax-v1",
        "prompt_token_ids": ASYNC_CHAT_PROMPT_IDS,
        "teacher_token_ids": teacher,
        "teacher_ids_sha256": ids_sha256(teacher),
        "teacher_steps": 128,
        "free_run_steps": 128,
        "repeat_count": 2,
    }
    document = POLICY_MODULE.create_initial_policy_document(
        source_contract, candidates, trace, SELECTIVE_QUALITY_SUITE_SHA256
    )
    if not advance:
        path.write_bytes(POLICY_MODULE.canonical_bytes(document))
        return document["policy_sha256"]

    q_path = candidates[1]["path"]
    v_path = candidates[2]["path"]
    generated = list(teacher)
    generated[7] = 999
    document = POLICY_MODULE.advance_policy_document(
        document,
        bound_selective_observation(
            document, generated, outcome="divergent", path=q_path
        ),
    )
    document = POLICY_MODULE.advance_policy_document(
        document,
        bound_selective_observation(
            document, generated, outcome="divergent", path=v_path
        ),
    )
    document = POLICY_MODULE.advance_policy_document(
        document,
        bound_selective_observation(
            document, generated, outcome="divergent", path=v_path
        ),
    )
    path.write_bytes(POLICY_MODULE.canonical_bytes(document))
    return document["policy_sha256"]


@contextmanager
def fake_selective_certification(source: Path, *, full_manifest: bool = True):
    inspected = MODULE._inspect_source(str(source))
    manifest_patch = (
        mock.patch.object(
            MODULE,
            "SELECTIVE_SOURCE_MANIFEST_SHA256",
            MODULE._manifest_sha256(inspected.records),
        )
        if full_manifest
        else nullcontext()
    )
    with (
        mock.patch.object(
            MODULE,
            "SELECTIVE_SOURCE_CONFIG_SHA256",
            inspected.records["config.json"].sha256,
        ),
        mock.patch.object(
            MODULE,
            "SELECTIVE_SOURCE_SCHEMA_SHA256",
            MODULE._language_schema_sha256(inspected.tensor_schema),
        ),
        mock.patch.object(
            MODULE,
            "SELECTIVE_SOURCE_TENSOR_COUNT",
            len(MODULE._canonical_language_schema(inspected.tensor_schema)),
        ),
        manifest_patch,
    ):
        yield


def ids_sha256(ids: list[int]) -> str:
    return hashlib.sha256(
        json.dumps(ids, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def write_hybrid_v2_policy(path: Path, source: Path) -> str:
    retained = [
        f"language_model.model.layers.{layer}.linear_attn.out_proj"
        for layer in (12, 14, 20)
    ]
    source_schema = MODULE._parse_safetensors_schema(
        source / "model.safetensors", "fixture v2 source"
    )
    policy = {
        "preset": HYBRID_PRESET_V2,
        "source": {
            "revision": HYBRID_REVISION,
            "config_sha256": hashlib.sha256(
                (source / "config.json").read_bytes()
            ).hexdigest(),
            "language_schema_sha256": MODULE._language_schema_sha256(source_schema),
            "language_tensor_count": 6,
        },
        "quantization": {"bits": 8, "group_size": 64, "mode": "affine"},
        "retained_bf16_paths": retained,
        "ledger": {
            "quantized_module_count": 1,
            "retained_bf16_module_count": 3,
            "quantized_logical_weight_count": 128,
            "retained_bf16_logical_weight_count": 384,
            "quantized_module_parameter_bytes": 136,
            "retained_bf16_weight_bytes": 768,
            "estimated_total_parameter_bytes": 928,
            "output_tensor_count": 8,
        },
        "quality_gate": {
            "api": "mlx_lm.generate.generate_step",
            "semantics": "mlx-generate-step-argmax-v1",
            "prompt_token_ids": ASYNC_CHAT_PROMPT_IDS,
            "teacher_token_ids": FIXTURE_ASYNC_TEACHER_IDS,
            "teacher_ids_sha256": ids_sha256(FIXTURE_ASYNC_TEACHER_IDS),
            "teacher_steps": 128,
            "free_run_steps": 100,
            "first100_free_run_sha256": ids_sha256(FIXTURE_ASYNC_TEACHER_IDS[:100]),
            "repeat_count": 2,
        },
        "auxiliary_raw_prompt_gate": {
            "admission": False,
            "prompt_token_ids": [9419],
            "scope": "legacy-v1-manual-full-prompt",
            "superseded_policy_sha256": (
                "560f9b3df77a650603d91ff2ed60c0a56761f2d3fc408296be0a87a2f13e65cf"
            ),
        },
    }
    payload = {
        "format": "apxinf-qwen35-mlx-hybrid-policy-evidence-v2",
        "policy": policy,
        "evidence": {},
    }
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return hashlib.sha256(
        json.dumps(policy, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def write_counterfactual_diagnostic(path: Path) -> tuple[str, str]:
    body = {
        "format": "apxinf-mlx-chinese-hybrid-diagnostic-receipt-v1",
        "status": "diagnostic-only",
        "inputs": {
            "candidate_manifest_sha256": (
                "5f65527db17cd4fb7a1a46ca6ab3940a8bcc8225c620ca10d3016265d2bf8553"
            ),
            "reference_manifest_sha256": (
                "fdce8bac86b1bbc888cac0139065f0291a9a57ce7f448e591b748f4baaad5dea"
            ),
        },
        "module_localization": {
            "ranking_metric": "same-bf16-input-relative-l1-error-ppm-v1",
            "top_candidates": [
                {
                    "current_tier": "w8",
                    "path": "language_model.model.layers.19.self_attn.o_proj",
                    "proposed_tier": "bf16",
                    "rank": 1,
                }
            ],
        },
    }
    content_sha256 = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    document = {**body, "content_sha256": content_sha256}
    payload = (
        json.dumps(document, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    )
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest(), content_sha256


def write_hybrid_counterfactual_v3_policy(
    path: Path,
    source: Path,
    *,
    diagnostic_artifact_sha256: str,
    diagnostic_content_sha256: str,
) -> str:
    retained = [
        "language_model.model.layers.12.linear_attn.out_proj",
        "language_model.model.layers.14.linear_attn.out_proj",
        "language_model.model.layers.19.self_attn.o_proj",
        "language_model.model.layers.20.linear_attn.out_proj",
    ]
    source_schema = MODULE._parse_safetensors_schema(
        source / "model.safetensors", "fixture counterfactual source"
    )
    inspected = MODULE._inspect_source(str(source))
    policy = {
        "preset": HYBRID_COUNTERFACTUAL_PRESET_V3,
        "source": {
            "revision": HYBRID_REVISION,
            "config_sha256": hashlib.sha256(
                (source / "config.json").read_bytes()
            ).hexdigest(),
            "language_schema_sha256": MODULE._language_schema_sha256(source_schema),
            "language_tensor_count": 7,
            "source_manifest_sha256": MODULE._manifest_sha256(inspected.records),
        },
        "quantization": {"bits": 8, "group_size": 64, "mode": "affine"},
        "retained_bf16_paths": retained,
        "ledger": {
            "quantized_module_count": 1,
            "retained_bf16_module_count": 4,
            "quantized_logical_weight_count": 128,
            "retained_bf16_logical_weight_count": 512,
            "quantized_module_parameter_bytes": 136,
            "retained_bf16_weight_bytes": 1024,
            "estimated_total_parameter_bytes": 1184,
            "output_tensor_count": 9,
        },
        "quality_gate": {
            "format": "apxinf-qwen35-mlx-counterfactual-canonical-gate-v1",
            "api": "mlx_lm.generate.generate_step",
            "semantics": "mlx-generate-step-argmax-v1",
            "prompt_token_ids": ASYNC_CHAT_PROMPT_IDS,
            "teacher_token_ids": FIXTURE_ASYNC_TEACHER_IDS,
            "teacher_ids_sha256": ids_sha256(FIXTURE_ASYNC_TEACHER_IDS),
            "teacher_steps": 128,
            "free_run_steps": 128,
            "free_run_ids_sha256": ids_sha256(FIXTURE_ASYNC_TEACHER_IDS),
            "repeat_count": 2,
        },
        "auxiliary_raw_prompt_gate": {
            "admission": False,
            "prompt_token_ids": [9419],
            "scope": "legacy-v1-manual-full-prompt",
            "superseded_policy_sha256": (
                "560f9b3df77a650603d91ff2ed60c0a56761f2d3fc408296be0a87a2f13e65cf"
            ),
        },
        "counterfactual": {
            "format": "apxinf-qwen35-mlx-hybrid-counterfactual-lineage-v1",
            "status": "unvalidated-candidate",
            "selection": {
                "causal_attribution": False,
                "current_tier": "w8",
                "path": "language_model.model.layers.19.self_attn.o_proj",
                "proposed_tier": "bf16",
                "rank": 1,
                "ranking_metric": "same-bf16-input-relative-l1-error-ppm-v1",
                "selection_basis": (
                    "trusted-diagnostic-trigger-only-not-causal-proof-v1"
                ),
            },
            "diagnostic": {
                "artifact_path": (
                    "doc/20260823-qwen35-macos-bringup/"
                    "qwen35-hybrid-w8-bf16-g64-chinese-state-aligned-"
                    "diagnostic-v1.json"
                ),
                "artifact_sha256": diagnostic_artifact_sha256,
                "content_sha256": diagnostic_content_sha256,
                "format": "apxinf-mlx-chinese-hybrid-diagnostic-receipt-v1",
            },
            "parent": {
                "bundle_manifest_sha256": (
                    "5f65527db17cd4fb7a1a46ca6ab3940a8bcc8225c620ca10d3016265d2bf8553"
                ),
                "policy_sha256": MODULE.HYBRID_POLICY_SHA256_V2,
                "preset": HYBRID_PRESET_V2,
            },
            "reference": {
                "bundle_manifest_sha256": (
                    "fdce8bac86b1bbc888cac0139065f0291a9a57ce7f448e591b748f4baaad5dea"
                ),
                "precision": "bf16",
            },
            "admission": {
                "formal_performance_claim": False,
                "general_parity": False,
                "parent_bundle_replacement": False,
                "promotion_requires_all_gates": True,
                "required_gates": [
                    "apxinf-mlx-counterfactual-deployed-canonical-gate-v1",
                    "qwen35-0.8b-mlx-multi-prompt-quality-v1-4-prompts-x2",
                ],
            },
        },
    }
    payload = {
        "format": "apxinf-qwen35-mlx-hybrid-counterfactual-policy-v3",
        "policy": policy,
        "evidence": {"status": "fixture-only-no-real-bundle"},
    }
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return hashlib.sha256(
        json.dumps(policy, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def write_hybrid_policy(path: Path, source: Path) -> str:
    retained = ["language_model.model.layers.12.linear_attn.out_proj"]
    schema = {
        "language_model.model.layers.0.weight": ["BF16", [2, 64]],
        "language_model.model.layers.0.norm.weight": ["F32", [2]],
        "language_model.model.layers.0.linear_attn.conv1d.weight": [
            "BF16",
            [2, 4, 1],
        ],
        "language_model.model.layers.12.linear_attn.out_proj.weight": [
            "BF16",
            [2, 64],
        ],
    }
    policy = {
        "preset": HYBRID_PRESET,
        "source": {
            "revision": HYBRID_REVISION,
            "config_sha256": hashlib.sha256(
                (source / "config.json").read_bytes()
            ).hexdigest(),
            "language_schema_sha256": hashlib.sha256(
                json.dumps(schema, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
            "language_tensor_count": 4,
        },
        "quantization": {"bits": 8, "group_size": 64, "mode": "affine"},
        "retained_bf16_paths": retained,
        "ledger": {
            "quantized_module_count": 1,
            "retained_bf16_module_count": 1,
            "quantized_logical_weight_count": 128,
            "retained_bf16_logical_weight_count": 128,
            "quantized_module_parameter_bytes": 136,
            "retained_bf16_weight_bytes": 256,
            "estimated_total_parameter_bytes": 416,
            "output_tensor_count": 6,
        },
    }
    payload = {
        "format": "apxinf-qwen35-mlx-hybrid-policy-evidence-v1",
        "policy": policy,
        "evidence": {},
    }
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return hashlib.sha256(
        json.dumps(policy, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


class FakeApi:
    def __init__(
        self,
        source: Path,
        *,
        mutate_source: bool = False,
        generated_ids: list[int] | None = None,
        post_reload_generated_ids: list[int] | None = None,
        teacher_forced_ids: list[int] | None = None,
        post_reload_teacher_forced_ids: list[int] | None = None,
    ) -> None:
        self.source = source
        self.mutate_source = mutate_source
        self.load_calls: list[tuple[str, dict[str, object]]] = []
        self.quantize_calls: list[dict[str, object]] = []
        self.save_calls: list[dict[str, object]] = []
        self.network_blocked = False
        self.environment: dict[str, str | None] = {}
        self.generated_ids = generated_ids
        self.post_reload_generated_ids = post_reload_generated_ids
        self.teacher_forced_ids = teacher_forced_ids
        self.post_reload_teacher_forced_ids = post_reload_teacher_forced_ids
        self.generate_calls: list[dict[str, object]] = []
        self.teacher_forced_calls: list[dict[str, object]] = []
        self.argmax_calls: list[tuple[object, int]] = []
        self.quantization_decisions: dict[str, object] = {}

    def array(self, values: list[int]) -> list[int]:
        return list(values)

    def argmax(self, values: object, *, axis: int = -1) -> object:
        self.argmax_calls.append((values, axis))
        return values

    def generate_step(
        self,
        prompt: list[int],
        model: object,
        *,
        max_tokens: int,
        sampler: object,
    ) -> object:
        self.generate_calls.append(
            {
                "prompt": list(prompt),
                "model": model,
                "max_tokens": max_tokens,
                "sampler": sampler,
            }
        )
        sampler("fixture-logprobs")
        selected_ids = self.generated_ids
        if (
            type(model) is dict
            and model.get("loaded_from") != str(self.source)
            and self.post_reload_generated_ids is not None
        ):
            selected_ids = self.post_reload_generated_ids
        if selected_ids is None:
            raise AssertionError("unexpected generate_step call")
        return ((token, object()) for token in selected_ids[:max_tokens])

    def teacher_forced_step(
        self,
        prompt: list[int],
        model: object,
        teacher_ids: list[int],
    ) -> list[int]:
        self.teacher_forced_calls.append(
            {
                "prompt": list(prompt),
                "model": model,
                "teacher_ids": list(teacher_ids),
            }
        )
        selected_ids = self.teacher_forced_ids
        if selected_ids is None:
            selected_ids = self.generated_ids
        if (
            type(model) is dict
            and model.get("loaded_from") != str(self.source)
            and self.post_reload_teacher_forced_ids is not None
        ):
            selected_ids = self.post_reload_teacher_forced_ids
        if selected_ids is None:
            raise AssertionError("unexpected teacher_forced_step call")
        return list(selected_ids)

    def load(
        self, path: str, **kwargs: object
    ) -> tuple[object, object, dict[str, object]]:
        self.load_calls.append((path, kwargs))
        self.environment = {
            key: os.environ.get(key)
            for key in (
                "HF_HUB_OFFLINE",
                "TRANSFORMERS_OFFLINE",
                "HF_HUB_DISABLE_TELEMETRY",
                "HTTP_PROXY",
                "HF_TOKEN",
            )
        }
        probe = socket.socket()
        try:
            probe.connect(("127.0.0.1", 9))
        except MODULE.BundleError:
            self.network_blocked = True
        finally:
            probe.close()
        loaded_path = Path(path)
        if self.mutate_source and loaded_path == self.source:
            (self.source / "tokenizer.json").write_bytes(b"changed")
        config = json.loads((loaded_path / "config.json").read_text(encoding="utf-8"))
        return {"loaded_from": path}, object(), config

    def quantize_model(
        self,
        model: object,
        config: dict[str, object],
        **kwargs: object,
    ) -> tuple[object, dict[str, object]]:
        self.quantize_calls.append(dict(kwargs))
        updated = dict(config)
        quantization = {
            "bits": kwargs["bits"],
            "group_size": kwargs["group_size"],
            "mode": kwargs["mode"],
        }
        source_schema = MODULE._parse_safetensors_schema(
            self.source / "model.safetensors", "fixture source"
        )
        canonical = MODULE._canonical_language_schema(source_schema)
        predicate = kwargs.get("quant_predicate")
        for name, (_dtype, shape) in canonical.items():
            eligible = (
                name.endswith(".weight")
                and len(shape) == 2
                and shape[-1] > 0
                and shape[-1] % kwargs["group_size"] == 0
            )
            if not eligible:
                continue
            path = name.removesuffix(".weight")
            decision = True if predicate is None else predicate(path, object())
            self.quantization_decisions[path] = decision
            if type(decision) is dict:
                quantization[path] = dict(decision)
        updated["quantization"] = quantization
        updated["quantization_config"] = dict(quantization)
        return model, updated

    def save(
        self,
        destination: Path,
        source: str,
        model: object,
        tokenizer: object,
        config: dict[str, object],
        **kwargs: object,
    ) -> None:
        self.save_calls.append(
            {"source": source, "config": config, "kwargs": dict(kwargs)}
        )
        destination.mkdir(parents=True)
        (destination / "README.md").write_text("fixture MLX model\n", encoding="utf-8")
        (destination / "config.json").write_text(
            json.dumps(config, sort_keys=True), encoding="utf-8"
        )
        for name in MODULE.TOKENIZER_FILES:
            (destination / name).write_bytes(b"fake tokenizer rewrite")
        if "quantization" in config:
            default_bits = config["quantization"]["bits"]
            source_schema = MODULE._parse_safetensors_schema(
                self.source / "model.safetensors", "fixture source"
            )
            canonical = MODULE._canonical_language_schema(source_schema)
            tensors = {}
            for name, (dtype, shape) in canonical.items():
                eligible = (
                    name.endswith(".weight") and len(shape) == 2 and shape[-1] % 64 == 0
                )
                base = name.removesuffix(".weight")
                decision = self.quantization_decisions.get(base, True)
                should_quantize = eligible and decision is not False
                if should_quantize:
                    bits = decision["bits"] if type(decision) is dict else default_bits
                    tensors[name] = ("U32", [*shape[:-1], shape[-1] * bits // 32])
                    tensors[f"{base}.scales"] = ("BF16", [*shape[:-1], shape[-1] // 64])
                    tensors[f"{base}.biases"] = ("BF16", [*shape[:-1], shape[-1] // 64])
                else:
                    tensors[name] = (dtype, list(shape))
        else:
            tensors = {
                "language_model.model.layers.0.weight": ("BF16", [2, 64]),
                "language_model.model.layers.0.norm.weight": ("F32", [2]),
                "language_model.model.layers.0.linear_attn.conv1d.weight": (
                    "BF16",
                    [2, 4, 1],
                ),
            }
        write_safetensors(destination / "model.safetensors", tensors)
        write_index(
            destination / "model.safetensors.index.json",
            "model.safetensors",
            list(tensors),
        )


def version_side_effect(distribution: str) -> str:
    return PINNED[distribution]


class BundleBuilderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.source = make_source(self.root)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_checked_in_v3_counterfactual_policy_matches_the_frozen_hash(self) -> None:
        document = json.loads(
            MODULE.HYBRID_COUNTERFACTUAL_POLICY_PATH_V3.read_text(encoding="utf-8")
        )
        observed = hashlib.sha256(
            json.dumps(
                document["policy"],
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()

        self.assertEqual(
            observed,
            MODULE.HYBRID_COUNTERFACTUAL_POLICY_SHA256_V3,
        )
        self.assertFalse(document["evidence"]["real_bundle_built"])
        self.assertEqual(
            document["policy"]["counterfactual"]["admission"]["required_gates"],
            [
                "apxinf-mlx-counterfactual-deployed-canonical-gate-v1",
                "qwen35-0.8b-mlx-multi-prompt-quality-v1-4-prompts-x2",
            ],
        )

    def build(
        self,
        mode: str,
        *,
        name: str = "output",
        api: FakeApi | None = None,
    ) -> tuple[dict[str, object], Path, FakeApi]:
        output = self.root / name
        selected = api if api is not None else FakeApi(self.source)
        with mock.patch.object(
            MODULE.metadata, "version", side_effect=version_side_effect
        ):
            receipt = MODULE.build_bundle(
                str(self.source), str(output), mode, api=selected
            )
        return receipt, output, selected

    def test_safetensors_schema_rejects_zero_payload_and_overlapping_ranges(
        self,
    ) -> None:
        zero = self.root / "zero.safetensors"
        zero_header = {
            "tensor": {
                "dtype": "BF16",
                "shape": [2, 64],
                "data_offsets": [0, 0],
            }
        }
        zero_payload = json.dumps(
            zero_header, sort_keys=True, separators=(",", ":")
        ).encode()
        zero.write_bytes(struct.pack("<Q", len(zero_payload)) + zero_payload)
        with self.assertRaisesRegex(MODULE.BundleError, "byte size"):
            MODULE._parse_safetensors_schema(zero, "zero fixture")

        overlap = self.root / "overlap.safetensors"
        overlap_header = {
            "first": {
                "dtype": "BF16",
                "shape": [2],
                "data_offsets": [0, 4],
            },
            "second": {
                "dtype": "BF16",
                "shape": [2],
                "data_offsets": [0, 4],
            },
        }
        overlap_payload = json.dumps(
            overlap_header, sort_keys=True, separators=(",", ":")
        ).encode()
        overlap.write_bytes(
            struct.pack("<Q", len(overlap_payload)) + overlap_payload + bytes(4)
        )
        with self.assertRaisesRegex(MODULE.BundleError, "overlap|contiguous"):
            MODULE._parse_safetensors_schema(overlap, "overlap fixture")

    def test_mixed_build_is_offline_atomic_and_preserves_exact_bytes_and_dtypes(
        self,
    ) -> None:
        old_proxy = os.environ.get("HTTP_PROXY")
        old_token = os.environ.get("HF_TOKEN")
        os.environ["HTTP_PROXY"] = "http://ambient.invalid:8080"
        os.environ["HF_TOKEN"] = "secret-token"
        try:
            receipt, output, api = self.build("mixed-bf16")
        finally:
            if old_proxy is None:
                os.environ.pop("HTTP_PROXY", None)
            else:
                os.environ["HTTP_PROXY"] = old_proxy
            if old_token is None:
                os.environ.pop("HF_TOKEN", None)
            else:
                os.environ["HF_TOKEN"] = old_token

        self.assertTrue(receipt["passed"])
        self.assertTrue(receipt["published"])
        self.assertFalse(receipt["verify_only"])
        self.assertEqual(receipt["mode"], "mixed-bf16")
        self.assertEqual(receipt["format"], MODULE.RECEIPT_FORMAT)
        self.assertEqual(receipt["output"]["directory"], str(output))
        self.assertTrue(receipt["output"]["tokenizer_bytes_preserved"])
        self.assertTrue(receipt["output"]["mixed_dtype_schema_preserved"])
        self.assertEqual(
            receipt["output"]["tensor_dtype_counts"], {"BF16": 2, "F32": 1}
        )
        self.assertFalse(receipt["policy"]["blanket_dtype_cast"])
        self.assertEqual(api.quantize_calls, [])
        self.assertEqual(len(api.load_calls), 1)
        self.assertEqual(api.load_calls[0][0], str(self.source))
        self.assertEqual(
            api.load_calls[0][1],
            {
                "tokenizer_config": {
                    "local_files_only": True,
                    "trust_remote_code": False,
                },
                "lazy": False,
                "return_config": True,
            },
        )
        self.assertTrue(api.network_blocked)
        self.assertEqual(api.environment["HF_HUB_OFFLINE"], "1")
        self.assertEqual(api.environment["TRANSFORMERS_OFFLINE"], "1")
        self.assertEqual(api.environment["HTTP_PROXY"], None)
        self.assertEqual(api.environment["HF_TOKEN"], None)
        self.assertEqual(output.stat().st_mode & 0o777, 0o700)
        for name in MODULE.TOKENIZER_FILES:
            self.assertEqual(
                (output / name).read_bytes(), (self.source / name).read_bytes()
            )
            self.assertEqual((output / name).stat().st_nlink, 1)
        output_artifacts = receipt["output"]["artifacts"]
        self.assertEqual(
            set(output_artifacts), {path.name for path in output.iterdir()}
        )
        for name, evidence in output_artifacts.items():
            payload = (output / name).read_bytes()
            self.assertEqual(evidence["size"], len(payload))
            self.assertEqual(evidence["sha256"], hashlib.sha256(payload).hexdigest())

    def test_quantized_modes_use_only_the_exact_pinned_affine_contract(self) -> None:
        for mode, bits in (("affine-w8-g64", 8), ("affine-w4-g64", 4)):
            with self.subTest(mode=mode):
                receipt, output, api = self.build(mode, name=mode)
                self.assertEqual(
                    api.quantize_calls,
                    [{"bits": bits, "group_size": 64, "mode": "affine"}],
                )
                config = json.loads((output / "config.json").read_text())
                expected = {"bits": bits, "group_size": 64, "mode": "affine"}
                self.assertEqual(config["quantization"], expected)
                self.assertEqual(config["quantization_config"], expected)
                self.assertFalse(receipt["output"]["mixed_dtype_schema_preserved"])
                self.assertEqual(receipt["output"]["quantized_tensor_count"], 1)
                self.assertIn("not-a-parity-claim", receipt["policy"]["quality_tier"])

    def test_selective_policy_roundtrips_exact_w4_w8_bf16_config_and_schema(
        self,
    ) -> None:
        case = self.root / "selective-case"
        case.mkdir()
        source = make_selective_source(case)
        policy_path = case / "policy.json"
        policy_sha256 = write_selective_policy(policy_path, source)
        output = case / "output"
        api = FakeApi(source, generated_ids=SELECTIVE_TEACHER_IDS)

        with (
            fake_selective_certification(source),
            mock.patch.object(
                MODULE.metadata, "version", side_effect=version_side_effect
            ),
        ):
            built = MODULE.build_bundle(
                str(source),
                str(output),
                "affine-w4-g64",
                mixed_policy=str(policy_path),
                source_revision=HYBRID_REVISION,
                api=api,
            )

        down = "language_model.model.layers.0.mlp.down_proj"
        q_path = "language_model.model.layers.0.self_attn.q_proj"
        v_path = "language_model.model.layers.0.self_attn.v_proj"
        self.assertEqual(
            api.quantization_decisions,
            {
                down: True,
                q_path: {"bits": 8, "group_size": 64, "mode": "affine"},
                v_path: False,
            },
        )
        config = json.loads((output / "config.json").read_text(encoding="utf-8"))
        expected_quantization = {
            "bits": 4,
            "group_size": 64,
            "mode": "affine",
            q_path: {"bits": 8, "group_size": 64, "mode": "affine"},
        }
        self.assertEqual(config["quantization"], expected_quantization)
        self.assertEqual(config["quantization_config"], expected_quantization)
        self.assertEqual(
            config[MODULE.SELECTIVE_CONFIG_KEY]["policy_sha256"], policy_sha256
        )
        schema = MODULE._parse_safetensors_schema(
            output / "model.safetensors", "fixture selective output"
        )
        self.assertEqual(schema[f"{down}.weight"], ("U32", (2, 8)))
        self.assertEqual(schema[f"{q_path}.weight"], ("U32", (2, 16)))
        self.assertEqual(schema[f"{v_path}.weight"], ("BF16", (2, 64)))
        self.assertEqual(built["output"]["w4_module_count"], 1)
        self.assertEqual(built["output"]["w8_module_count"], 1)
        self.assertEqual(built["output"]["retained_bf16_module_count"], 1)
        self.assertTrue(built["output"]["selective_mixed_quantization_verified"])
        self.assertEqual(len(api.load_calls), 2)
        self.assertEqual(len(api.teacher_forced_calls), 4)
        self.assertEqual(len(api.generate_calls), 4)
        self.assertTrue(built["quality_gate"]["deployed_bundle_reloaded"])
        self.assertEqual(
            built["quality_gate"]["format"],
            "apxinf-mlx-selective-deployed-quality-gate-v2",
        )
        self.assertEqual(
            built["quality_gate"]["pre_save"]["teacher_forced"]["exact_steps"],
            128,
        )
        self.assertEqual(
            built["quality_gate"]["post_save_reload"]["async_free_run"]["exact_steps"],
            128,
        )

        with (
            fake_selective_certification(source),
            mock.patch.object(
                MODULE.metadata, "version", side_effect=version_side_effect
            ),
            mock.patch.object(
                MODULE, "_load_mlx_api", side_effect=AssertionError("MLX imported")
            ),
        ):
            verified = MODULE.build_bundle(
                str(source),
                str(output),
                "affine-w4-g64",
                mixed_policy=str(policy_path),
                source_revision=HYBRID_REVISION,
                verify_only=True,
            )
        self.assertEqual(
            verified["output"]["manifest_sha256"],
            built["output"]["manifest_sha256"],
        )
        self.assertNotIn("quality_gate", verified)
        self.assertEqual(verified["selective_policy"]["policy_sha256"], policy_sha256)

        corrupted = {
            name: (dtype, list(shape)) for name, (dtype, shape) in schema.items()
        }
        corrupted[f"{q_path}.weight"] = ("U32", [2, 8])
        write_safetensors(output / "model.safetensors", corrupted)
        write_index(
            output / "model.safetensors.index.json",
            "model.safetensors",
            list(corrupted),
        )
        with (
            fake_selective_certification(source),
            mock.patch.object(
                MODULE.metadata, "version", side_effect=version_side_effect
            ),
            mock.patch.object(
                MODULE, "_load_mlx_api", side_effect=AssertionError("MLX imported")
            ),
            self.assertRaisesRegex(MODULE.BundleError, "W4/W8/BF16"),
        ):
            MODULE.build_bundle(
                str(source),
                str(output),
                "affine-w4-g64",
                mixed_policy=str(policy_path),
                source_revision=HYBRID_REVISION,
                verify_only=True,
            )

    def test_selective_policy_rejects_conflicts_and_source_drift_before_mlx_load(
        self,
    ) -> None:
        case = self.root / "selective-reject"
        case.mkdir()
        source = make_selective_source(case)
        policy_path = case / "policy.json"
        write_selective_policy(policy_path, source)
        api = FakeApi(source, generated_ids=SELECTIVE_TEACHER_IDS)

        with self.assertRaisesRegex(MODULE.BundleError, "mutually exclusive"):
            MODULE.build_bundle(
                str(source),
                str(case / "conflict"),
                "affine-w4-g64",
                preset=HYBRID_PRESET,
                mixed_policy=str(policy_path),
                source_revision=HYBRID_REVISION,
                api=api,
            )
        with self.assertRaisesRegex(MODULE.BundleError, "requires mode"):
            MODULE.build_bundle(
                str(source),
                str(case / "wrong-mode"),
                "affine-w8-g64",
                mixed_policy=str(policy_path),
                source_revision=HYBRID_REVISION,
                api=api,
            )

        schema = MODULE._parse_safetensors_schema(
            source / "model.safetensors", "fixture selective source"
        )
        tensors = {
            name: (dtype, list(shape)) for name, (dtype, shape) in schema.items()
        }
        tensors["model.language_model.layers.1.mlp.down_proj.weight"] = (
            "BF16",
            [2, 64],
        )
        write_safetensors(source / "model.safetensors", tensors)
        write_index(
            source / "model.safetensors.index.json",
            "model.safetensors",
            list(tensors),
        )
        with self.assertRaisesRegex(
            MODULE.BundleError, "artifact manifest|tensor count|schema"
        ):
            MODULE.build_bundle(
                str(source),
                str(case / "source-drift"),
                "affine-w4-g64",
                mixed_policy=str(policy_path),
                source_revision=HYBRID_REVISION,
                api=api,
            )
        self.assertEqual(api.load_calls, [])

    def test_selective_policy_rejects_rehashed_weights_outside_the_source_lock(
        self,
    ) -> None:
        case = self.root / "selective-source-lock"
        case.mkdir()
        source = make_selective_source(case)
        shard = source / "model.safetensors"
        with shard.open("r+b") as handle:
            handle.seek(-1, os.SEEK_END)
            old = handle.read(1)
            handle.seek(-1, os.SEEK_END)
            handle.write(bytes([old[0] ^ 1]))
        policy_path = case / "policy.json"
        write_selective_policy(policy_path, source)
        api = FakeApi(source, generated_ids=SELECTIVE_TEACHER_IDS)

        with (
            fake_selective_certification(source, full_manifest=False),
            mock.patch.object(
                MODULE.metadata, "version", side_effect=version_side_effect
            ),
            self.assertRaisesRegex(MODULE.BundleError, "certified source manifest"),
        ):
            MODULE.build_bundle(
                str(source),
                str(case / "output"),
                "affine-w4-g64",
                mixed_policy=str(policy_path),
                source_revision=HYBRID_REVISION,
                api=api,
            )
        self.assertEqual(api.load_calls, [])

    def test_selective_policy_never_saves_or_publishes_a_divergent_trajectory(
        self,
    ) -> None:
        case = self.root / "selective-divergence"
        case.mkdir()
        source = make_selective_source(case)
        policy_path = case / "policy.json"
        write_selective_policy(policy_path, source, advance=False)
        generated = list(SELECTIVE_TEACHER_IDS)
        generated[127] = 999
        api = FakeApi(
            source,
            generated_ids=generated,
            teacher_forced_ids=SELECTIVE_TEACHER_IDS,
        )
        output = case / "output"

        with (
            fake_selective_certification(source),
            mock.patch.object(
                MODULE.metadata, "version", side_effect=version_side_effect
            ),
            self.assertRaisesRegex(MODULE.BundleError, r"async.*steps \[127\]"),
        ):
            MODULE.build_bundle(
                str(source),
                str(output),
                "affine-w4-g64",
                mixed_policy=str(policy_path),
                source_revision=HYBRID_REVISION,
                api=api,
            )
        self.assertEqual(api.save_calls, [])
        self.assertFalse(output.exists())

    def test_selective_policy_never_publishes_teacher_forced_divergence(
        self,
    ) -> None:
        case = self.root / "selective-teacher-forced-divergence"
        case.mkdir()
        source = make_selective_source(case)
        policy_path = case / "policy.json"
        write_selective_policy(policy_path, source, advance=False)
        teacher_forced = list(SELECTIVE_TEACHER_IDS)
        teacher_forced[23] = 999
        api = FakeApi(
            source,
            generated_ids=SELECTIVE_TEACHER_IDS,
            teacher_forced_ids=teacher_forced,
        )
        output = case / "output"

        with (
            fake_selective_certification(source),
            mock.patch.object(
                MODULE.metadata, "version", side_effect=version_side_effect
            ),
            self.assertRaisesRegex(MODULE.BundleError, r"teacher-forced.*steps \[23\]"),
        ):
            MODULE.build_bundle(
                str(source),
                str(output),
                "affine-w4-g64",
                mixed_policy=str(policy_path),
                source_revision=HYBRID_REVISION,
                api=api,
            )
        self.assertEqual(api.save_calls, [])
        self.assertFalse(output.exists())

    def test_selective_quality_gate_separates_teacher_and_async_evidence(
        self,
    ) -> None:
        case = self.root / "selective-double-gate-evidence"
        case.mkdir()
        source = make_selective_source(case)
        policy_path = case / "policy.json"
        write_selective_policy(policy_path, source, advance=False)
        output = case / "output"
        api = FakeApi(
            source,
            generated_ids=SELECTIVE_TEACHER_IDS,
            teacher_forced_ids=SELECTIVE_TEACHER_IDS,
        )

        with (
            fake_selective_certification(source),
            mock.patch.object(
                MODULE.metadata, "version", side_effect=version_side_effect
            ),
        ):
            receipt = MODULE.build_bundle(
                str(source),
                str(output),
                "affine-w4-g64",
                mixed_policy=str(policy_path),
                source_revision=HYBRID_REVISION,
                api=api,
            )

        self.assertEqual(len(api.teacher_forced_calls), 4)
        self.assertEqual(len(api.generate_calls), 4)
        quality_gate = receipt["quality_gate"]
        self.assertEqual(
            quality_gate["format"],
            "apxinf-mlx-selective-deployed-quality-gate-v2",
        )
        expected_hashes = [ids_sha256(SELECTIVE_TEACHER_IDS)] * 2
        for phase in ("pre_save", "post_save_reload"):
            evidence = quality_gate[phase]
            self.assertEqual(
                evidence,
                {
                    "format": "apxinf-mlx-selective-quality-gate-v2",
                    "prompt_token_ids": ASYNC_CHAT_PROMPT_IDS,
                    "teacher_ids_sha256": ids_sha256(SELECTIVE_TEACHER_IDS),
                    "teacher_forced": {
                        "api": "mlx_lm.generate.generate_step",
                        "semantics": (
                            "mlx-generate-step-cached-teacher-forced-argmax-v1"
                        ),
                        "forced_token_ids_sha256": ids_sha256(SELECTIVE_TEACHER_IDS),
                        "steps": 128,
                        "exact_steps": 128,
                        "repeat_count": 2,
                        "repeat_sha256": expected_hashes,
                        "repeated_identically": True,
                    },
                    "async_free_run": {
                        "api": "mlx_lm.generate.generate_step",
                        "semantics": "mlx-generate-step-argmax-v1",
                        "steps": 128,
                        "exact_steps": 128,
                        "repeat_count": 2,
                        "repeat_sha256": expected_hashes,
                        "repeated_identically": True,
                    },
                },
            )
        self.assertTrue(quality_gate["deployed_bundle_reloaded"])
        self.assertTrue(quality_gate["exact_trajectory_claim"])
        self.assertFalse(quality_gate["formal_performance_claim"])

    def test_selective_policy_reloads_staging_and_rejects_serialized_divergence(
        self,
    ) -> None:
        case = self.root / "selective-reload-divergence"
        case.mkdir()
        source = make_selective_source(case)
        policy_path = case / "policy.json"
        write_selective_policy(policy_path, source, advance=False)
        post_reload = list(SELECTIVE_TEACHER_IDS)
        post_reload[5] = 999
        api = FakeApi(
            source,
            generated_ids=SELECTIVE_TEACHER_IDS,
            post_reload_generated_ids=post_reload,
            teacher_forced_ids=SELECTIVE_TEACHER_IDS,
        )
        output = case / "output"

        with (
            fake_selective_certification(source),
            mock.patch.object(
                MODULE.metadata, "version", side_effect=version_side_effect
            ),
            self.assertRaisesRegex(MODULE.BundleError, "post-save reload"),
        ):
            MODULE.build_bundle(
                str(source),
                str(output),
                "affine-w4-g64",
                mixed_policy=str(policy_path),
                source_revision=HYBRID_REVISION,
                api=api,
            )
        self.assertEqual(len(api.load_calls), 2)
        self.assertEqual(len(api.save_calls), 1)
        self.assertFalse(output.exists())

    def test_selective_policy_rejects_post_reload_teacher_forced_divergence(
        self,
    ) -> None:
        case = self.root / "selective-reload-teacher-forced-divergence"
        case.mkdir()
        source = make_selective_source(case)
        policy_path = case / "policy.json"
        write_selective_policy(policy_path, source, advance=False)
        post_reload_teacher = list(SELECTIVE_TEACHER_IDS)
        post_reload_teacher[41] = 999
        api = FakeApi(
            source,
            generated_ids=SELECTIVE_TEACHER_IDS,
            teacher_forced_ids=SELECTIVE_TEACHER_IDS,
            post_reload_teacher_forced_ids=post_reload_teacher,
        )
        output = case / "output"

        with (
            fake_selective_certification(source),
            mock.patch.object(
                MODULE.metadata, "version", side_effect=version_side_effect
            ),
            self.assertRaisesRegex(
                MODULE.BundleError,
                r"post-save reload.*teacher-forced.*steps \[41\]",
            ),
        ):
            MODULE.build_bundle(
                str(source),
                str(output),
                "affine-w4-g64",
                mixed_policy=str(policy_path),
                source_revision=HYBRID_REVISION,
                api=api,
            )
        self.assertEqual(len(api.load_calls), 2)
        self.assertEqual(len(api.save_calls), 1)
        self.assertFalse(output.exists())

    def test_selective_policy_cannot_choose_a_self_consistent_fake_teacher(
        self,
    ) -> None:
        case = self.root / "selective-fake-teacher"
        case.mkdir()
        source = make_selective_source(case)
        trusted_path = case / "trusted.json"
        write_selective_policy(trusted_path, source)
        trusted = json.loads(trusted_path.read_text(encoding="utf-8"))
        trace = dict(trusted["policy"]["trace"])
        fake_teacher = list(trace["teacher_token_ids"])
        fake_teacher[0] += 1
        trace["teacher_token_ids"] = fake_teacher
        trace["teacher_ids_sha256"] = ids_sha256(fake_teacher)
        fake = POLICY_MODULE.create_initial_policy_document(
            trusted["policy"]["source"],
            trusted["policy"]["candidate_modules"],
            trace,
            trusted["policy"]["quality_suite_sha256"],
        )
        fake_path = case / "fake.json"
        fake_path.write_bytes(POLICY_MODULE.canonical_bytes(fake))
        api = FakeApi(source, generated_ids=fake_teacher)

        with (
            fake_selective_certification(source),
            self.assertRaisesRegex(MODULE.BundleError, "frozen BF16 v2 teacher"),
        ):
            MODULE.build_bundle(
                str(source),
                str(case / "output"),
                "affine-w4-g64",
                mixed_policy=str(fake_path),
                source_revision=HYBRID_REVISION,
                api=api,
            )
        self.assertEqual(api.load_calls, [])

    def test_selective_verify_binds_the_exact_search_receipt_document(self) -> None:
        case = self.root / "selective-receipt-binding"
        case.mkdir()
        source = make_selective_source(case)
        policy_path = case / "policy.json"
        write_selective_policy(policy_path, source)
        original = json.loads(policy_path.read_text(encoding="utf-8"))
        output = case / "output"
        with (
            fake_selective_certification(source),
            mock.patch.object(
                MODULE.metadata, "version", side_effect=version_side_effect
            ),
        ):
            MODULE.build_bundle(
                str(source),
                str(output),
                "affine-w4-g64",
                mixed_policy=str(policy_path),
                source_revision=HYBRID_REVISION,
                api=FakeApi(source, generated_ids=SELECTIVE_TEACHER_IDS),
            )

        exact_observation = bound_selective_observation(
            original,
            SELECTIVE_TEACHER_IDS,
            outcome="exact",
            path=None,
        )
        alternate = POLICY_MODULE.advance_policy_document(original, exact_observation)
        self.assertEqual(alternate["policy_sha256"], original["policy_sha256"])
        self.assertNotEqual(
            alternate["search_receipt_sha256"],
            original["search_receipt_sha256"],
        )
        policy_path.write_bytes(POLICY_MODULE.canonical_bytes(alternate))

        with (
            fake_selective_certification(source),
            mock.patch.object(
                MODULE.metadata, "version", side_effect=version_side_effect
            ),
            self.assertRaisesRegex(
                MODULE.BundleError, "policy document|receipt|manifest"
            ),
        ):
            MODULE.build_bundle(
                str(source),
                str(output),
                "affine-w4-g64",
                mixed_policy=str(policy_path),
                source_revision=HYBRID_REVISION,
                verify_only=True,
            )

    def test_hybrid_w8_parity_preset_builds_only_the_pinned_selective_policy(
        self,
    ) -> None:
        case = self.root / "hybrid-case"
        case.mkdir()
        source = make_hybrid_source(case)
        policy_path = case / "policy.json"
        policy_sha256 = write_hybrid_policy(policy_path, source)
        output = case / "output"
        api = FakeApi(source)
        with (
            mock.patch.object(
                MODULE.metadata, "version", side_effect=version_side_effect
            ),
            mock.patch.object(MODULE, "HYBRID_POLICY_PATH", policy_path),
            mock.patch.object(MODULE, "HYBRID_POLICY_SHA256", policy_sha256),
        ):
            receipt = MODULE.build_bundle(
                str(source),
                str(output),
                "affine-w8-g64",
                preset=HYBRID_PRESET,
                source_revision=HYBRID_REVISION,
                api=api,
            )
        self.assertEqual(receipt["preset"]["name"], HYBRID_PRESET)
        self.assertEqual(receipt["preset"]["policy_sha256"], policy_sha256)
        self.assertEqual(
            receipt["preset"]["retained_bf16_paths"],
            ["language_model.model.layers.12.linear_attn.out_proj"],
        )
        self.assertEqual(receipt["output"]["quantized_tensor_count"], 1)
        self.assertEqual(receipt["output"]["retained_bf16_tensor_count"], 1)
        self.assertEqual(
            receipt["output"]["weight_ledger"]["estimated_total_parameter_bytes"], 416
        )
        self.assertIn("quant_predicate", api.quantize_calls[0])
        self.assertEqual(
            receipt["policy"]["quality_tier"],
            "certified-frozen-hello-128-parity-preset-v1",
        )

    def test_v2_async_chat_preset_runs_the_pinned_production_gate_before_publish(
        self,
    ) -> None:
        case = self.root / "hybrid-v2-case"
        case.mkdir()
        source = make_hybrid_v2_source(case)
        policy_path = case / "policy-v2.json"
        policy_sha256 = write_hybrid_v2_policy(policy_path, source)
        output = case / "output"
        api = FakeApi(source, generated_ids=FIXTURE_ASYNC_TEACHER_IDS)
        with (
            mock.patch.object(
                MODULE.metadata, "version", side_effect=version_side_effect
            ),
            mock.patch.object(MODULE, "HYBRID_POLICY_PATH_V2", policy_path),
            mock.patch.object(MODULE, "HYBRID_POLICY_SHA256_V2", policy_sha256),
        ):
            receipt = MODULE.build_bundle(
                str(source),
                str(output),
                "affine-w8-g64",
                preset=HYBRID_PRESET_V2,
                source_revision=HYBRID_REVISION,
                api=api,
            )

        self.assertEqual(receipt["preset"]["name"], HYBRID_PRESET_V2)
        self.assertEqual(
            receipt["preset"]["retained_bf16_paths"],
            [
                "language_model.model.layers.12.linear_attn.out_proj",
                "language_model.model.layers.14.linear_attn.out_proj",
                "language_model.model.layers.20.linear_attn.out_proj",
            ],
        )
        self.assertEqual(
            [
                (call["prompt"], call["max_tokens"], callable(call["sampler"]))
                for call in api.generate_calls
            ],
            [
                (ASYNC_CHAT_PROMPT_IDS, 128, True),
                (ASYNC_CHAT_PROMPT_IDS, 128, True),
            ],
        )
        self.assertEqual(
            api.argmax_calls,
            [("fixture-logprobs", -1), ("fixture-logprobs", -1)],
        )
        self.assertEqual(
            receipt["quality_gate"],
            {
                "api": "mlx_lm.generate.generate_step",
                "first100_free_run_sha256": ids_sha256(FIXTURE_ASYNC_TEACHER_IDS[:100]),
                "prompt_token_ids": ASYNC_CHAT_PROMPT_IDS,
                "repeat_count": 2,
                "repeated_identically": True,
                "semantics": "mlx-generate-step-argmax-v1",
                "teacher_exact": 128,
                "teacher_ids_sha256": ids_sha256(FIXTURE_ASYNC_TEACHER_IDS),
                "teacher_steps": 128,
            },
        )
        self.assertEqual(receipt["output"]["retained_bf16_tensor_count"], 3)
        self.assertEqual(
            receipt["policy"]["quality_tier"],
            "certified-canonical-chat-async-generate-step-parity-preset-v2",
        )

    def test_v3_counterfactual_builds_only_the_diagnostic_top1_four_path_profile(
        self,
    ) -> None:
        case = self.root / "hybrid-counterfactual-v3-case"
        case.mkdir()
        source = make_hybrid_counterfactual_v3_source(case)
        diagnostic_path = case / "diagnostic.json"
        diagnostic_sha256, diagnostic_content_sha256 = (
            write_counterfactual_diagnostic(diagnostic_path)
        )
        policy_path = case / "policy-v3.json"
        policy_sha256 = write_hybrid_counterfactual_v3_policy(
            policy_path,
            source,
            diagnostic_artifact_sha256=diagnostic_sha256,
            diagnostic_content_sha256=diagnostic_content_sha256,
        )
        output = case / "output"
        api = FakeApi(source, generated_ids=FIXTURE_ASYNC_TEACHER_IDS)
        with (
            mock.patch.object(
                MODULE.metadata, "version", side_effect=version_side_effect
            ),
            mock.patch.object(
                MODULE, "HYBRID_COUNTERFACTUAL_POLICY_PATH_V3", policy_path
            ),
            mock.patch.object(
                MODULE, "HYBRID_COUNTERFACTUAL_POLICY_SHA256_V3", policy_sha256
            ),
            mock.patch.object(
                MODULE, "HYBRID_COUNTERFACTUAL_DIAGNOSTIC_PATH", diagnostic_path
            ),
            mock.patch.object(
                MODULE,
                "HYBRID_COUNTERFACTUAL_DIAGNOSTIC_ARTIFACT_SHA256",
                diagnostic_sha256,
            ),
            mock.patch.object(
                MODULE,
                "HYBRID_COUNTERFACTUAL_DIAGNOSTIC_CONTENT_SHA256",
                diagnostic_content_sha256,
            ),
        ):
            receipt = MODULE.build_bundle(
                str(source),
                str(output),
                "affine-w8-g64",
                preset=HYBRID_COUNTERFACTUAL_PRESET_V3,
                source_revision=HYBRID_REVISION,
                api=api,
            )

        retained = [
            "language_model.model.layers.12.linear_attn.out_proj",
            "language_model.model.layers.14.linear_attn.out_proj",
            "language_model.model.layers.19.self_attn.o_proj",
            "language_model.model.layers.20.linear_attn.out_proj",
        ]
        self.assertEqual(receipt["preset"]["retained_bf16_paths"], retained)
        self.assertEqual(receipt["preset"]["counterfactual"]["selection"]["path"], retained[2])
        self.assertEqual(receipt["output"]["retained_bf16_tensor_count"], 4)
        self.assertEqual(receipt["output"]["quantized_tensor_count"], 1)
        self.assertEqual(
            receipt["policy"]["quality_tier"],
            "diagnostic-chinese-top1-counterfactual-only-v1",
        )
        self.assertEqual(len(api.load_calls), 2)
        self.assertEqual(len(api.teacher_forced_calls), 4)
        self.assertEqual(len(api.generate_calls), 4)
        self.assertEqual(
            receipt["quality_gate"]["format"],
            "apxinf-mlx-counterfactual-deployed-canonical-gate-v1",
        )
        self.assertTrue(receipt["quality_gate"]["deployed_bundle_reloaded"])
        self.assertFalse(receipt["quality_gate"]["fixed_suite_accepted"])
        self.assertEqual(
            {
                path: decision
                for path, decision in api.quantization_decisions.items()
                if decision is False
            },
            {path: False for path in retained},
        )

    def test_v3_counterfactual_never_publishes_if_its_diagnostic_changes(self) -> None:
        case = self.root / "hybrid-counterfactual-v3-diagnostic-race"
        case.mkdir()
        source = make_hybrid_counterfactual_v3_source(case)
        diagnostic_path = case / "diagnostic.json"
        diagnostic_sha256, diagnostic_content_sha256 = (
            write_counterfactual_diagnostic(diagnostic_path)
        )
        policy_path = case / "policy-v3.json"
        policy_sha256 = write_hybrid_counterfactual_v3_policy(
            policy_path,
            source,
            diagnostic_artifact_sha256=diagnostic_sha256,
            diagnostic_content_sha256=diagnostic_content_sha256,
        )
        output = case / "output"
        api = FakeApi(source, generated_ids=FIXTURE_ASYNC_TEACHER_IDS)
        original_save = api.save

        def mutate_diagnostic_after_save(*args: object, **kwargs: object) -> None:
            original_save(*args, **kwargs)
            diagnostic_path.write_bytes(diagnostic_path.read_bytes() + b" ")

        api.save = mutate_diagnostic_after_save  # type: ignore[method-assign]
        with (
            mock.patch.object(
                MODULE.metadata, "version", side_effect=version_side_effect
            ),
            mock.patch.object(
                MODULE, "HYBRID_COUNTERFACTUAL_POLICY_PATH_V3", policy_path
            ),
            mock.patch.object(
                MODULE, "HYBRID_COUNTERFACTUAL_POLICY_SHA256_V3", policy_sha256
            ),
            mock.patch.object(
                MODULE, "HYBRID_COUNTERFACTUAL_DIAGNOSTIC_PATH", diagnostic_path
            ),
            mock.patch.object(
                MODULE,
                "HYBRID_COUNTERFACTUAL_DIAGNOSTIC_ARTIFACT_SHA256",
                diagnostic_sha256,
            ),
            mock.patch.object(
                MODULE,
                "HYBRID_COUNTERFACTUAL_DIAGNOSTIC_CONTENT_SHA256",
                diagnostic_content_sha256,
            ),
            self.assertRaisesRegex(MODULE.BundleError, "diagnostic.*changed"),
        ):
            MODULE.build_bundle(
                str(source),
                str(output),
                "affine-w8-g64",
                preset=HYBRID_COUNTERFACTUAL_PRESET_V3,
                source_revision=HYBRID_REVISION,
                api=api,
            )

        self.assertFalse(output.exists())

    def test_v3_counterfactual_rejects_a_rehashed_arbitrary_bf16_path_before_mlx(
        self,
    ) -> None:
        case = self.root / "hybrid-counterfactual-v3-arbitrary-path"
        case.mkdir()
        source = make_hybrid_counterfactual_v3_source(case)
        diagnostic_path = case / "diagnostic.json"
        diagnostic_sha256, diagnostic_content_sha256 = (
            write_counterfactual_diagnostic(diagnostic_path)
        )
        policy_path = case / "policy-v3.json"
        write_hybrid_counterfactual_v3_policy(
            policy_path,
            source,
            diagnostic_artifact_sha256=diagnostic_sha256,
            diagnostic_content_sha256=diagnostic_content_sha256,
        )
        document = json.loads(policy_path.read_text(encoding="utf-8"))
        document["policy"]["retained_bf16_paths"].append(
            "language_model.model.layers.21.self_attn.o_proj"
        )
        policy_path.write_text(json.dumps(document, sort_keys=True), encoding="utf-8")
        rehashed_policy = hashlib.sha256(
            json.dumps(
                document["policy"], sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest()
        api = FakeApi(source, generated_ids=FIXTURE_ASYNC_TEACHER_IDS)
        output = case / "output"
        with (
            mock.patch.object(
                MODULE.metadata, "version", side_effect=version_side_effect
            ),
            mock.patch.object(
                MODULE, "HYBRID_COUNTERFACTUAL_POLICY_PATH_V3", policy_path
            ),
            mock.patch.object(
                MODULE, "HYBRID_COUNTERFACTUAL_POLICY_SHA256_V3", rehashed_policy
            ),
            mock.patch.object(
                MODULE, "HYBRID_COUNTERFACTUAL_DIAGNOSTIC_PATH", diagnostic_path
            ),
            mock.patch.object(
                MODULE,
                "HYBRID_COUNTERFACTUAL_DIAGNOSTIC_ARTIFACT_SHA256",
                diagnostic_sha256,
            ),
            mock.patch.object(
                MODULE,
                "HYBRID_COUNTERFACTUAL_DIAGNOSTIC_CONTENT_SHA256",
                diagnostic_content_sha256,
            ),
            self.assertRaisesRegex(
                MODULE.BundleError, "retained BF16 path portfolio drifted"
            ),
        ):
            MODULE.build_bundle(
                str(source),
                str(output),
                "affine-w8-g64",
                preset=HYBRID_COUNTERFACTUAL_PRESET_V3,
                source_revision=HYBRID_REVISION,
                api=api,
            )

        self.assertEqual(api.load_calls, [])
        self.assertFalse(output.exists())

    def test_v3_counterfactual_rechecks_diagnostic_immediately_before_publish(
        self,
    ) -> None:
        case = self.root / "hybrid-counterfactual-v3-late-diagnostic-race"
        case.mkdir()
        source = make_hybrid_counterfactual_v3_source(case)
        diagnostic_path = case / "diagnostic.json"
        diagnostic_sha256, diagnostic_content_sha256 = (
            write_counterfactual_diagnostic(diagnostic_path)
        )
        policy_path = case / "policy-v3.json"
        policy_sha256 = write_hybrid_counterfactual_v3_policy(
            policy_path,
            source,
            diagnostic_artifact_sha256=diagnostic_sha256,
            diagnostic_content_sha256=diagnostic_content_sha256,
        )
        output = case / "output"
        api = FakeApi(source, generated_ids=FIXTURE_ASYNC_TEACHER_IDS)
        original_receipt = MODULE._receipt

        def mutate_diagnostic_after_receipt(*args: object, **kwargs: object):
            receipt = original_receipt(*args, **kwargs)
            if receipt["published"]:
                diagnostic_path.write_bytes(diagnostic_path.read_bytes() + b" ")
            return receipt

        with (
            mock.patch.object(
                MODULE.metadata, "version", side_effect=version_side_effect
            ),
            mock.patch.object(
                MODULE, "HYBRID_COUNTERFACTUAL_POLICY_PATH_V3", policy_path
            ),
            mock.patch.object(
                MODULE, "HYBRID_COUNTERFACTUAL_POLICY_SHA256_V3", policy_sha256
            ),
            mock.patch.object(
                MODULE, "HYBRID_COUNTERFACTUAL_DIAGNOSTIC_PATH", diagnostic_path
            ),
            mock.patch.object(
                MODULE,
                "HYBRID_COUNTERFACTUAL_DIAGNOSTIC_ARTIFACT_SHA256",
                diagnostic_sha256,
            ),
            mock.patch.object(
                MODULE,
                "HYBRID_COUNTERFACTUAL_DIAGNOSTIC_CONTENT_SHA256",
                diagnostic_content_sha256,
            ),
            mock.patch.object(
                MODULE, "_receipt", side_effect=mutate_diagnostic_after_receipt
            ),
            self.assertRaisesRegex(MODULE.BundleError, "diagnostic.*changed"),
        ):
            MODULE.build_bundle(
                str(source),
                str(output),
                "affine-w8-g64",
                preset=HYBRID_COUNTERFACTUAL_PRESET_V3,
                source_revision=HYBRID_REVISION,
                api=api,
            )

        self.assertFalse(output.exists())

    def test_v3_counterfactual_rechecks_policy_immediately_before_publish(self) -> None:
        case = self.root / "hybrid-counterfactual-v3-late-policy-race"
        case.mkdir()
        source = make_hybrid_counterfactual_v3_source(case)
        diagnostic_path = case / "diagnostic.json"
        diagnostic_sha256, diagnostic_content_sha256 = (
            write_counterfactual_diagnostic(diagnostic_path)
        )
        policy_path = case / "policy-v3.json"
        policy_sha256 = write_hybrid_counterfactual_v3_policy(
            policy_path,
            source,
            diagnostic_artifact_sha256=diagnostic_sha256,
            diagnostic_content_sha256=diagnostic_content_sha256,
        )
        output = case / "output"
        api = FakeApi(source, generated_ids=FIXTURE_ASYNC_TEACHER_IDS)
        original_receipt = MODULE._receipt

        def mutate_policy_after_receipt(*args: object, **kwargs: object):
            receipt = original_receipt(*args, **kwargs)
            if receipt["published"]:
                policy_path.write_bytes(policy_path.read_bytes() + b" ")
            return receipt

        with (
            mock.patch.object(
                MODULE.metadata, "version", side_effect=version_side_effect
            ),
            mock.patch.object(
                MODULE, "HYBRID_COUNTERFACTUAL_POLICY_PATH_V3", policy_path
            ),
            mock.patch.object(
                MODULE, "HYBRID_COUNTERFACTUAL_POLICY_SHA256_V3", policy_sha256
            ),
            mock.patch.object(
                MODULE, "HYBRID_COUNTERFACTUAL_DIAGNOSTIC_PATH", diagnostic_path
            ),
            mock.patch.object(
                MODULE,
                "HYBRID_COUNTERFACTUAL_DIAGNOSTIC_ARTIFACT_SHA256",
                diagnostic_sha256,
            ),
            mock.patch.object(
                MODULE,
                "HYBRID_COUNTERFACTUAL_DIAGNOSTIC_CONTENT_SHA256",
                diagnostic_content_sha256,
            ),
            mock.patch.object(
                MODULE, "_receipt", side_effect=mutate_policy_after_receipt
            ),
            self.assertRaisesRegex(MODULE.BundleError, "policy.*changed"),
        ):
            MODULE.build_bundle(
                str(source),
                str(output),
                "affine-w8-g64",
                preset=HYBRID_COUNTERFACTUAL_PRESET_V3,
                source_revision=HYBRID_REVISION,
                api=api,
            )

        self.assertFalse(output.exists())

    def test_v2_async_chat_preset_does_not_publish_a_divergent_model(self) -> None:
        case = self.root / "hybrid-v2-divergence"
        case.mkdir()
        source = make_hybrid_v2_source(case)
        policy_path = case / "policy-v2.json"
        policy_sha256 = write_hybrid_v2_policy(policy_path, source)
        generated = list(FIXTURE_ASYNC_TEACHER_IDS)
        generated[9] = 999
        api = FakeApi(source, generated_ids=generated)
        output = case / "output"
        with (
            mock.patch.object(
                MODULE.metadata, "version", side_effect=version_side_effect
            ),
            mock.patch.object(MODULE, "HYBRID_POLICY_PATH_V2", policy_path),
            mock.patch.object(MODULE, "HYBRID_POLICY_SHA256_V2", policy_sha256),
            self.assertRaisesRegex(MODULE.BundleError, r"steps \[9\]"),
        ):
            MODULE.build_bundle(
                str(source),
                str(output),
                "affine-w8-g64",
                preset=HYBRID_PRESET_V2,
                source_revision=HYBRID_REVISION,
                api=api,
            )

        self.assertFalse(output.exists())
        self.assertEqual(api.save_calls, [])

    def test_v2_policy_rejects_deleted_added_or_renamed_retained_layer(self) -> None:
        for mutation in ("delete", "add", "rename"):
            with self.subTest(mutation=mutation):
                case = self.root / f"hybrid-v2-policy-{mutation}"
                case.mkdir()
                source = make_hybrid_v2_source(case)
                policy_path = case / "policy-v2.json"
                expected_sha256 = write_hybrid_v2_policy(policy_path, source)
                document = json.loads(policy_path.read_text(encoding="utf-8"))
                retained = document["policy"]["retained_bf16_paths"]
                if mutation == "delete":
                    retained.pop(1)
                elif mutation == "add":
                    retained.append(
                        "language_model.model.layers.21.linear_attn.out_proj"
                    )
                else:
                    retained[1] = (
                        "language_model.model.layers.14.linear_attn.renamed_proj"
                    )
                policy_path.write_text(json.dumps(document), encoding="utf-8")
                api = FakeApi(source, generated_ids=FIXTURE_ASYNC_TEACHER_IDS)
                with (
                    mock.patch.object(
                        MODULE.metadata, "version", side_effect=version_side_effect
                    ),
                    mock.patch.object(MODULE, "HYBRID_POLICY_PATH_V2", policy_path),
                    mock.patch.object(
                        MODULE, "HYBRID_POLICY_SHA256_V2", expected_sha256
                    ),
                    self.assertRaisesRegex(MODULE.BundleError, "policy hash drifted"),
                ):
                    MODULE.build_bundle(
                        str(source),
                        str(case / "output"),
                        "affine-w8-g64",
                        preset=HYBRID_PRESET_V2,
                        source_revision=HYBRID_REVISION,
                        api=api,
                    )
                self.assertEqual(api.load_calls, [])

    def test_v2_verify_only_reuses_the_exact_bundle_without_loading_mlx(self) -> None:
        case = self.root / "hybrid-v2-verify"
        case.mkdir()
        source = make_hybrid_v2_source(case)
        policy_path = case / "policy-v2.json"
        policy_sha256 = write_hybrid_v2_policy(policy_path, source)
        output = case / "output"
        with (
            mock.patch.object(
                MODULE.metadata, "version", side_effect=version_side_effect
            ),
            mock.patch.object(MODULE, "HYBRID_POLICY_PATH_V2", policy_path),
            mock.patch.object(MODULE, "HYBRID_POLICY_SHA256_V2", policy_sha256),
        ):
            built = MODULE.build_bundle(
                str(source),
                str(output),
                "affine-w8-g64",
                preset=HYBRID_PRESET_V2,
                source_revision=HYBRID_REVISION,
                api=FakeApi(source, generated_ids=FIXTURE_ASYNC_TEACHER_IDS),
            )
        with (
            mock.patch.object(
                MODULE.metadata, "version", side_effect=version_side_effect
            ),
            mock.patch.object(MODULE, "HYBRID_POLICY_PATH_V2", policy_path),
            mock.patch.object(MODULE, "HYBRID_POLICY_SHA256_V2", policy_sha256),
            mock.patch.object(
                MODULE, "_load_mlx_api", side_effect=AssertionError("MLX imported")
            ),
        ):
            verified = MODULE.build_bundle(
                str(source),
                str(output),
                "affine-w8-g64",
                preset=HYBRID_PRESET_V2,
                source_revision=HYBRID_REVISION,
                verify_only=True,
            )

        self.assertTrue(verified["verify_only"])
        self.assertFalse(verified["published"])
        self.assertNotIn("quality_gate", verified)
        self.assertEqual(
            verified["output"]["manifest_sha256"],
            built["output"]["manifest_sha256"],
        )
        self.assertEqual(verified["preset"]["policy_sha256"], policy_sha256)

    def test_v2_preset_rejects_wrong_revision_and_non_qwen_source(self) -> None:
        case = self.root / "hybrid-v2-source-contract"
        case.mkdir()
        source = make_hybrid_v2_source(case)
        policy_path = case / "policy-v2.json"
        policy_sha256 = write_hybrid_v2_policy(policy_path, source)
        with (
            mock.patch.object(
                MODULE.metadata, "version", side_effect=version_side_effect
            ),
            mock.patch.object(MODULE, "HYBRID_POLICY_PATH_V2", policy_path),
            mock.patch.object(MODULE, "HYBRID_POLICY_SHA256_V2", policy_sha256),
            self.assertRaisesRegex(MODULE.BundleError, "requires source revision"),
        ):
            MODULE.build_bundle(
                str(source),
                str(case / "wrong-revision"),
                "affine-w8-g64",
                preset=HYBRID_PRESET_V2,
                source_revision="different-revision",
                api=FakeApi(source, generated_ids=FIXTURE_ASYNC_TEACHER_IDS),
            )

    def test_v2_preset_rejects_deleted_added_or_renamed_source_layer(self) -> None:
        for mutation in ("delete", "add", "rename"):
            with self.subTest(mutation=mutation):
                case = self.root / f"hybrid-v2-source-{mutation}"
                case.mkdir()
                source = make_hybrid_v2_source(case)
                policy_path = case / "policy-v2.json"
                policy_sha256 = write_hybrid_v2_policy(policy_path, source)
                tensors = {
                    "model.language_model.layers.0.weight": ("BF16", [2, 64]),
                    "model.language_model.layers.0.norm.weight": ("F32", [2]),
                    "model.language_model.layers.0.linear_attn.conv1d.weight": (
                        "BF16",
                        [2, 1, 4],
                    ),
                    **{
                        f"model.language_model.layers.{layer}.linear_attn.out_proj.weight": (
                            "BF16",
                            [2, 64],
                        )
                        for layer in (12, 14, 20)
                    },
                }
                retained = "model.language_model.layers.14.linear_attn.out_proj.weight"
                if mutation == "delete":
                    tensors.pop(retained)
                elif mutation == "add":
                    tensors[
                        "model.language_model.layers.21.linear_attn.out_proj.weight"
                    ] = ("BF16", [2, 64])
                else:
                    tensors[
                        "model.language_model.layers.14.linear_attn.renamed_proj.weight"
                    ] = tensors.pop(retained)
                write_safetensors(source / "model.safetensors", tensors)
                write_index(
                    source / "model.safetensors.index.json",
                    "model.safetensors",
                    list(tensors),
                )
                with (
                    mock.patch.object(
                        MODULE.metadata, "version", side_effect=version_side_effect
                    ),
                    mock.patch.object(MODULE, "HYBRID_POLICY_PATH_V2", policy_path),
                    mock.patch.object(MODULE, "HYBRID_POLICY_SHA256_V2", policy_sha256),
                    self.assertRaisesRegex(MODULE.BundleError, "tensor"),
                ):
                    MODULE.build_bundle(
                        str(source),
                        str(case / "output"),
                        "affine-w8-g64",
                        preset=HYBRID_PRESET_V2,
                        source_revision=HYBRID_REVISION,
                        api=FakeApi(source, generated_ids=FIXTURE_ASYNC_TEACHER_IDS),
                    )

    def test_v2_policy_rejects_prompt_or_generation_semantics_drift(self) -> None:
        for mutation in ("prompt", "semantics"):
            with self.subTest(mutation=mutation):
                case = self.root / f"hybrid-v2-gate-{mutation}"
                case.mkdir()
                source = make_hybrid_v2_source(case)
                policy_path = case / "policy-v2.json"
                write_hybrid_v2_policy(policy_path, source)
                document = json.loads(policy_path.read_text(encoding="utf-8"))
                gate = document["policy"]["quality_gate"]
                if mutation == "prompt":
                    gate["prompt_token_ids"][0] += 1
                else:
                    gate["semantics"] = "implicit-default-sampler"
                policy_path.write_text(json.dumps(document), encoding="utf-8")
                mutated_sha256 = hashlib.sha256(
                    json.dumps(
                        document["policy"],
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                ).hexdigest()
                with (
                    mock.patch.object(
                        MODULE.metadata, "version", side_effect=version_side_effect
                    ),
                    mock.patch.object(MODULE, "HYBRID_POLICY_PATH_V2", policy_path),
                    mock.patch.object(
                        MODULE, "HYBRID_POLICY_SHA256_V2", mutated_sha256
                    ),
                    self.assertRaisesRegex(
                        MODULE.BundleError, "quality-gate semantics drifted"
                    ),
                ):
                    MODULE.build_bundle(
                        str(source),
                        str(case / "output"),
                        "affine-w8-g64",
                        preset=HYBRID_PRESET_V2,
                        source_revision=HYBRID_REVISION,
                        api=FakeApi(source, generated_ids=FIXTURE_ASYNC_TEACHER_IDS),
                    )

    def test_v2_verify_only_rejects_deleted_added_or_renamed_output_layer(
        self,
    ) -> None:
        for mutation in ("delete", "add", "rename"):
            with self.subTest(mutation=mutation):
                case = self.root / f"hybrid-v2-output-{mutation}"
                case.mkdir()
                source = make_hybrid_v2_source(case)
                policy_path = case / "policy-v2.json"
                policy_sha256 = write_hybrid_v2_policy(policy_path, source)
                output = case / "output"
                with (
                    mock.patch.object(
                        MODULE.metadata, "version", side_effect=version_side_effect
                    ),
                    mock.patch.object(MODULE, "HYBRID_POLICY_PATH_V2", policy_path),
                    mock.patch.object(MODULE, "HYBRID_POLICY_SHA256_V2", policy_sha256),
                ):
                    MODULE.build_bundle(
                        str(source),
                        str(output),
                        "affine-w8-g64",
                        preset=HYBRID_PRESET_V2,
                        source_revision=HYBRID_REVISION,
                        api=FakeApi(source, generated_ids=FIXTURE_ASYNC_TEACHER_IDS),
                    )
                observed = MODULE._parse_safetensors_schema(
                    output / "model.safetensors", "fixture v2 output"
                )
                tensors = {
                    name: (dtype, list(shape))
                    for name, (dtype, shape) in observed.items()
                }
                retained = "language_model.model.layers.14.linear_attn.out_proj.weight"
                if mutation == "delete":
                    tensors.pop(retained)
                elif mutation == "add":
                    tensors[
                        "language_model.model.layers.21.linear_attn.out_proj.weight"
                    ] = ("BF16", [2, 64])
                else:
                    tensors[
                        "language_model.model.layers.14.linear_attn.renamed_proj.weight"
                    ] = tensors.pop(retained)
                write_safetensors(output / "model.safetensors", tensors)
                write_index(
                    output / "model.safetensors.index.json",
                    "model.safetensors",
                    list(tensors),
                )
                with (
                    mock.patch.object(
                        MODULE.metadata, "version", side_effect=version_side_effect
                    ),
                    mock.patch.object(MODULE, "HYBRID_POLICY_PATH_V2", policy_path),
                    mock.patch.object(MODULE, "HYBRID_POLICY_SHA256_V2", policy_sha256),
                    self.assertRaisesRegex(
                        MODULE.BundleError, "packing/schema validation"
                    ),
                ):
                    MODULE.build_bundle(
                        str(source),
                        str(output),
                        "affine-w8-g64",
                        preset=HYBRID_PRESET_V2,
                        source_revision=HYBRID_REVISION,
                        verify_only=True,
                    )

        config_path = source / "config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config["model_type"] = "not_qwen"
        config_path.write_text(json.dumps(config), encoding="utf-8")
        with self.assertRaisesRegex(MODULE.BundleError, "model_type"):
            MODULE.build_bundle(
                str(source),
                str(case / "non-qwen"),
                "affine-w8-g64",
                preset=HYBRID_PRESET_V2,
                source_revision=HYBRID_REVISION,
                api=FakeApi(source, generated_ids=FIXTURE_ASYNC_TEACHER_IDS),
            )

    def test_hybrid_policy_rejects_deleted_added_or_renamed_retained_layer(
        self,
    ) -> None:
        for mutation in ("delete", "add", "rename"):
            with self.subTest(mutation=mutation):
                case = self.root / f"policy-{mutation}"
                case.mkdir()
                source = make_hybrid_source(case)
                policy_path = case / "policy.json"
                expected_sha256 = write_hybrid_policy(policy_path, source)
                document = json.loads(policy_path.read_text(encoding="utf-8"))
                retained = document["policy"]["retained_bf16_paths"]
                if mutation == "delete":
                    retained.clear()
                elif mutation == "add":
                    retained.append(
                        "language_model.model.layers.13.linear_attn.out_proj"
                    )
                else:
                    retained[0] = (
                        "language_model.model.layers.12.linear_attn.renamed_proj"
                    )
                policy_path.write_text(json.dumps(document), encoding="utf-8")
                api = FakeApi(source)
                with (
                    mock.patch.object(
                        MODULE.metadata, "version", side_effect=version_side_effect
                    ),
                    mock.patch.object(MODULE, "HYBRID_POLICY_PATH", policy_path),
                    mock.patch.object(MODULE, "HYBRID_POLICY_SHA256", expected_sha256),
                    self.assertRaisesRegex(MODULE.BundleError, "policy hash drifted"),
                ):
                    MODULE.build_bundle(
                        str(source),
                        str(case / "output"),
                        "affine-w8-g64",
                        preset=HYBRID_PRESET,
                        source_revision=HYBRID_REVISION,
                        api=api,
                    )
                self.assertEqual(api.load_calls, [])

    def test_hybrid_preset_rejects_non_qwen_and_wrong_revision(self) -> None:
        case = self.root / "preset-source-contract"
        case.mkdir()
        source = make_hybrid_source(case)
        policy_path = case / "policy.json"
        policy_sha256 = write_hybrid_policy(policy_path, source)
        with (
            mock.patch.object(
                MODULE.metadata, "version", side_effect=version_side_effect
            ),
            mock.patch.object(MODULE, "HYBRID_POLICY_PATH", policy_path),
            mock.patch.object(MODULE, "HYBRID_POLICY_SHA256", policy_sha256),
            self.assertRaisesRegex(MODULE.BundleError, "requires source revision"),
        ):
            MODULE.build_bundle(
                str(source),
                str(case / "wrong-revision"),
                "affine-w8-g64",
                preset=HYBRID_PRESET,
                source_revision="different-revision",
                api=FakeApi(source),
            )
        config_path = source / "config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config["model_type"] = "not_qwen"
        config_path.write_text(json.dumps(config), encoding="utf-8")
        with self.assertRaisesRegex(MODULE.BundleError, "model_type"):
            MODULE.build_bundle(
                str(source),
                str(case / "non-qwen"),
                "affine-w8-g64",
                preset=HYBRID_PRESET,
                source_revision=HYBRID_REVISION,
                api=FakeApi(source),
            )

    def test_hybrid_preset_rejects_config_and_source_schema_drift(self) -> None:
        for mutation in ("config", "delete", "add", "rename"):
            with self.subTest(mutation=mutation):
                case = self.root / f"source-drift-{mutation}"
                case.mkdir()
                source = make_hybrid_source(case)
                policy_path = case / "policy.json"
                policy_sha256 = write_hybrid_policy(policy_path, source)
                if mutation == "config":
                    config_path = source / "config.json"
                    config = json.loads(config_path.read_text(encoding="utf-8"))
                    config["fixture_drift"] = True
                    config_path.write_text(json.dumps(config), encoding="utf-8")
                    expected = "source config drifted"
                else:
                    tensors = {
                        "model.language_model.layers.0.weight": ("BF16", [2, 64]),
                        "model.language_model.layers.0.norm.weight": ("F32", [2]),
                        "model.language_model.layers.0.linear_attn.conv1d.weight": (
                            "BF16",
                            [2, 1, 4],
                        ),
                        "model.language_model.layers.12.linear_attn.out_proj.weight": (
                            "BF16",
                            [2, 64],
                        ),
                    }
                    retained = (
                        "model.language_model.layers.12.linear_attn.out_proj.weight"
                    )
                    if mutation == "delete":
                        tensors.pop(retained)
                    elif mutation == "add":
                        tensors[
                            "model.language_model.layers.13.linear_attn.out_proj.weight"
                        ] = ("BF16", [2, 64])
                    else:
                        tensors[
                            "model.language_model.layers.12.linear_attn.renamed_proj.weight"
                        ] = tensors.pop(retained)
                    write_safetensors(source / "model.safetensors", tensors)
                    write_index(
                        source / "model.safetensors.index.json",
                        "model.safetensors",
                        list(tensors),
                    )
                    expected = "tensor"
                with (
                    mock.patch.object(
                        MODULE.metadata, "version", side_effect=version_side_effect
                    ),
                    mock.patch.object(MODULE, "HYBRID_POLICY_PATH", policy_path),
                    mock.patch.object(MODULE, "HYBRID_POLICY_SHA256", policy_sha256),
                    self.assertRaisesRegex(MODULE.BundleError, expected),
                ):
                    MODULE.build_bundle(
                        str(source),
                        str(case / "output"),
                        "affine-w8-g64",
                        preset=HYBRID_PRESET,
                        source_revision=HYBRID_REVISION,
                        api=FakeApi(source),
                    )

    def test_hybrid_verify_only_reuses_exact_policy_and_rejects_manifest_drift(
        self,
    ) -> None:
        case = self.root / "hybrid-verify"
        case.mkdir()
        source = make_hybrid_source(case)
        policy_path = case / "policy.json"
        policy_sha256 = write_hybrid_policy(policy_path, source)
        output = case / "output"
        patches = (
            mock.patch.object(
                MODULE.metadata, "version", side_effect=version_side_effect
            ),
            mock.patch.object(MODULE, "HYBRID_POLICY_PATH", policy_path),
            mock.patch.object(MODULE, "HYBRID_POLICY_SHA256", policy_sha256),
        )
        with patches[0], patches[1], patches[2]:
            built = MODULE.build_bundle(
                str(source),
                str(output),
                "affine-w8-g64",
                preset=HYBRID_PRESET,
                source_revision=HYBRID_REVISION,
                api=FakeApi(source),
            )
        with (
            mock.patch.object(
                MODULE.metadata, "version", side_effect=version_side_effect
            ),
            mock.patch.object(MODULE, "HYBRID_POLICY_PATH", policy_path),
            mock.patch.object(MODULE, "HYBRID_POLICY_SHA256", policy_sha256),
            mock.patch.object(
                MODULE, "_load_mlx_api", side_effect=AssertionError("MLX imported")
            ),
        ):
            verified = MODULE.build_bundle(
                str(source),
                str(output),
                "affine-w8-g64",
                preset=HYBRID_PRESET,
                source_revision=HYBRID_REVISION,
                verify_only=True,
            )
        self.assertTrue(verified["verify_only"])
        self.assertEqual(
            verified["output"]["manifest_sha256"], built["output"]["manifest_sha256"]
        )
        with (
            mock.patch.object(
                MODULE.metadata, "version", side_effect=version_side_effect
            ),
            self.assertRaisesRegex(MODULE.BundleError, "unexpectedly declares"),
        ):
            MODULE.build_bundle(
                str(source),
                str(output),
                "affine-w8-g64",
                verify_only=True,
            )
        config_path = output / "config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config[MODULE.HYBRID_CONFIG_KEY]["retained_bf16_paths"].pop()
        config_path.write_text(json.dumps(config), encoding="utf-8")
        with (
            mock.patch.object(
                MODULE.metadata, "version", side_effect=version_side_effect
            ),
            mock.patch.object(MODULE, "HYBRID_POLICY_PATH", policy_path),
            mock.patch.object(MODULE, "HYBRID_POLICY_SHA256", policy_sha256),
            self.assertRaisesRegex(MODULE.BundleError, "policy manifest drifted"),
        ):
            MODULE.build_bundle(
                str(source),
                str(output),
                "affine-w8-g64",
                preset=HYBRID_PRESET,
                source_revision=HYBRID_REVISION,
                verify_only=True,
            )

    def test_hybrid_verify_only_rejects_deleted_added_or_renamed_output_layer(
        self,
    ) -> None:
        for mutation in ("delete", "add", "rename"):
            with self.subTest(mutation=mutation):
                case = self.root / f"output-drift-{mutation}"
                case.mkdir()
                source = make_hybrid_source(case)
                policy_path = case / "policy.json"
                policy_sha256 = write_hybrid_policy(policy_path, source)
                output = case / "output"
                with (
                    mock.patch.object(
                        MODULE.metadata, "version", side_effect=version_side_effect
                    ),
                    mock.patch.object(MODULE, "HYBRID_POLICY_PATH", policy_path),
                    mock.patch.object(MODULE, "HYBRID_POLICY_SHA256", policy_sha256),
                ):
                    MODULE.build_bundle(
                        str(source),
                        str(output),
                        "affine-w8-g64",
                        preset=HYBRID_PRESET,
                        source_revision=HYBRID_REVISION,
                        api=FakeApi(source),
                    )
                tensors = {
                    "language_model.model.layers.0.weight": ("U32", [2, 16]),
                    "language_model.model.layers.0.scales": ("BF16", [2, 1]),
                    "language_model.model.layers.0.biases": ("BF16", [2, 1]),
                    "language_model.model.layers.0.norm.weight": ("F32", [2]),
                    "language_model.model.layers.0.linear_attn.conv1d.weight": (
                        "BF16",
                        [2, 4, 1],
                    ),
                    "language_model.model.layers.12.linear_attn.out_proj.weight": (
                        "BF16",
                        [2, 64],
                    ),
                }
                retained = "language_model.model.layers.12.linear_attn.out_proj.weight"
                if mutation == "delete":
                    tensors.pop(retained)
                elif mutation == "add":
                    tensors[
                        "language_model.model.layers.13.linear_attn.out_proj.weight"
                    ] = ("BF16", [2, 64])
                else:
                    tensors[
                        "language_model.model.layers.12.linear_attn.renamed_proj.weight"
                    ] = tensors.pop(retained)
                write_safetensors(output / "model.safetensors", tensors)
                write_index(
                    output / "model.safetensors.index.json",
                    "model.safetensors",
                    list(tensors),
                )
                with (
                    mock.patch.object(
                        MODULE.metadata, "version", side_effect=version_side_effect
                    ),
                    mock.patch.object(MODULE, "HYBRID_POLICY_PATH", policy_path),
                    mock.patch.object(MODULE, "HYBRID_POLICY_SHA256", policy_sha256),
                    self.assertRaisesRegex(
                        MODULE.BundleError, "packing/schema validation"
                    ),
                ):
                    MODULE.build_bundle(
                        str(source),
                        str(output),
                        "affine-w8-g64",
                        preset=HYBRID_PRESET,
                        source_revision=HYBRID_REVISION,
                        verify_only=True,
                    )

    def test_verify_only_reuses_an_existing_bundle_without_calling_mlx(self) -> None:
        built, output, _api = self.build("mixed-bf16")
        with (
            mock.patch.object(
                MODULE.metadata, "version", side_effect=version_side_effect
            ),
            mock.patch.object(
                MODULE, "_load_mlx_api", side_effect=AssertionError("MLX imported")
            ),
        ):
            verified = MODULE.build_bundle(
                str(self.source),
                str(output),
                "mixed-bf16",
                verify_only=True,
            )
        self.assertFalse(verified["published"])
        self.assertTrue(verified["verify_only"])
        self.assertEqual(
            verified["output"]["manifest_sha256"],
            built["output"]["manifest_sha256"],
        )

    def test_existing_output_is_never_replaced(self) -> None:
        output = self.root / "output"
        output.mkdir()
        marker = output / "owner-data"
        marker.write_bytes(b"keep")
        api = FakeApi(self.source)
        with (
            mock.patch.object(
                MODULE.metadata, "version", side_effect=version_side_effect
            ),
            self.assertRaisesRegex(MODULE.BundleError, "already exists"),
        ):
            MODULE.build_bundle(str(self.source), str(output), "mixed-bf16", api=api)
        self.assertEqual(marker.read_bytes(), b"keep")
        self.assertEqual(api.load_calls, [])

    def test_publication_race_preserves_the_competing_output(self) -> None:
        output = self.root / "output"
        original = MODULE._rename_no_replace

        def race(source: Path, destination: Path) -> None:
            destination.mkdir()
            (destination / "winner").write_bytes(b"other process")
            original(source, destination)

        with (
            mock.patch.object(
                MODULE.metadata, "version", side_effect=version_side_effect
            ),
            mock.patch.object(MODULE, "_rename_no_replace", side_effect=race),
            self.assertRaisesRegex(MODULE.BundleError, "appeared during publication"),
        ):
            MODULE.build_bundle(
                str(self.source),
                str(output),
                "mixed-bf16",
                api=FakeApi(self.source),
            )
        self.assertEqual((output / "winner").read_bytes(), b"other process")
        self.assertEqual(
            list(self.root.glob(".apxinf-mlx-build-*")),
            [],
            "private staging directories must be cleaned",
        )

    def test_rejects_symlink_hardlink_and_unexpected_path_hazards(self) -> None:
        for hazard in ("symlink", "hardlink", "directory", "python"):
            with self.subTest(hazard=hazard):
                case_root = self.root / hazard
                case_root.mkdir()
                source = make_source(case_root)
                if hazard == "symlink":
                    target = case_root / "external-tokenizer"
                    target.write_bytes((source / "tokenizer.json").read_bytes())
                    (source / "tokenizer.json").unlink()
                    (source / "tokenizer.json").symlink_to(target)
                elif hazard == "hardlink":
                    target = case_root / "external-tokenizer"
                    target.write_bytes((source / "tokenizer.json").read_bytes())
                    (source / "tokenizer.json").unlink()
                    os.link(target, source / "tokenizer.json")
                elif hazard == "directory":
                    (source / "nested").mkdir()
                else:
                    (source / "modeling_qwen.py").write_text("raise SystemExit")
                with self.assertRaises(MODULE.BundleError):
                    MODULE.build_bundle(
                        str(source),
                        str(case_root / "output"),
                        "mixed-bf16",
                        api=FakeApi(source),
                    )

    def test_rejects_remote_code_wrong_model_and_missing_template(self) -> None:
        cases = ("remote", "wrong-model", "missing-template")
        for case in cases:
            with self.subTest(case=case):
                case_root = self.root / case
                case_root.mkdir()
                source = make_source(case_root)
                if case == "missing-template":
                    (source / "chat_template.jinja").unlink()
                else:
                    config_path = source / "config.json"
                    config = json.loads(config_path.read_text())
                    if case == "remote":
                        config["auto_map"] = {"AutoModel": "remote.module"}
                    else:
                        config["model_type"] = "not_qwen"
                    config_path.write_text(json.dumps(config), encoding="utf-8")
                with self.assertRaises(MODULE.BundleError):
                    MODULE.build_bundle(
                        str(source),
                        str(case_root / "output"),
                        "mixed-bf16",
                        api=FakeApi(source),
                    )

    def test_rejects_source_mutation_before_publication(self) -> None:
        output = self.root / "output"
        api = FakeApi(self.source, mutate_source=True)
        with (
            mock.patch.object(
                MODULE.metadata, "version", side_effect=version_side_effect
            ),
            self.assertRaisesRegex(MODULE.BundleError, "source file changed"),
        ):
            MODULE.build_bundle(str(self.source), str(output), "mixed-bf16", api=api)
        self.assertFalse(output.exists())

    def test_rejects_unpinned_runtime_before_loading_mlx(self) -> None:
        output = self.root / "output"
        api = FakeApi(self.source)

        def wrong_version(distribution: str) -> str:
            return "99.0" if distribution == "mlx-lm" else PINNED[distribution]

        with (
            mock.patch.object(MODULE.metadata, "version", side_effect=wrong_version),
            self.assertRaisesRegex(MODULE.BundleError, "mlx-lm version must be"),
        ):
            MODULE.build_bundle(str(self.source), str(output), "mixed-bf16", api=api)
        self.assertEqual(api.load_calls, [])
        self.assertFalse(output.exists())

    def test_verify_only_rejects_tokenizer_drift(self) -> None:
        _receipt, output, _api = self.build("mixed-bf16")
        (output / "tokenizer.json").write_bytes(b"drift")
        with (
            mock.patch.object(
                MODULE.metadata, "version", side_effect=version_side_effect
            ),
            self.assertRaisesRegex(MODULE.BundleError, "not byte-identical"),
        ):
            MODULE.build_bundle(
                str(self.source),
                str(output),
                "mixed-bf16",
                verify_only=True,
            )

    def test_verify_only_rejects_quantized_tensor_packing_drift(self) -> None:
        _receipt, output, _api = self.build("affine-w8-g64")
        tensors = {
            "language_model.model.layers.0.weight": ("U32", [2, 15]),
            "language_model.model.layers.0.scales": ("BF16", [2, 1]),
            "language_model.model.layers.0.biases": ("BF16", [2, 1]),
            "language_model.model.layers.0.norm.weight": ("F32", [2]),
            "language_model.model.layers.0.linear_attn.conv1d.weight": (
                "BF16",
                [2, 4, 1],
            ),
        }
        write_safetensors(output / "model.safetensors", tensors)
        write_index(
            output / "model.safetensors.index.json",
            "model.safetensors",
            list(tensors),
        )
        with (
            mock.patch.object(
                MODULE.metadata, "version", side_effect=version_side_effect
            ),
            self.assertRaisesRegex(MODULE.BundleError, "packing/schema validation"),
        ):
            MODULE.build_bundle(
                str(self.source),
                str(output),
                "affine-w8-g64",
                verify_only=True,
            )

    def test_verify_only_rejects_tensor_mapped_to_the_wrong_weight_shard(
        self,
    ) -> None:
        _receipt, output, _api = self.build("mixed-bf16")
        schema = MODULE._parse_safetensors_schema(
            output / "model.safetensors", "fixture output"
        )
        tensor_names = sorted(schema)
        first_names = tensor_names[:1]
        second_names = tensor_names[1:]
        first_shard = "model-00001-of-00002.safetensors"
        second_shard = "model-00002-of-00002.safetensors"
        (output / "model.safetensors").unlink()
        write_safetensors(
            output / first_shard,
            {name: (schema[name][0], list(schema[name][1])) for name in first_names},
        )
        write_safetensors(
            output / second_shard,
            {name: (schema[name][0], list(schema[name][1])) for name in second_names},
        )
        (output / "model.safetensors.index.json").write_text(
            json.dumps(
                {
                    "metadata": {"total_size": 0},
                    "weight_map": {
                        **{name: second_shard for name in first_names},
                        **{name: first_shard for name in second_names},
                    },
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )

        with (
            mock.patch.object(
                MODULE.metadata, "version", side_effect=version_side_effect
            ),
            self.assertRaisesRegex(MODULE.BundleError, "wrong weight shard"),
        ):
            MODULE.build_bundle(
                str(self.source),
                str(output),
                "mixed-bf16",
                verify_only=True,
            )

    def test_verify_only_rejects_a_file_added_after_the_manifest_scan(self) -> None:
        _receipt, output, _api = self.build("mixed-bf16")
        shard = output / "model.safetensors"
        original_parse = MODULE._parse_safetensors_schema
        added = False

        def race(
            path: Path,
            label: str,
            *,
            expected: MODULE.FileRecord | None = None,
        ) -> dict[str, tuple[str, tuple[int, ...]]]:
            nonlocal added
            schema = original_parse(path, label, expected=expected)
            if path == shard and not added:
                added = True
                (output / "late-added.bin").write_bytes(b"late")
            return schema

        with (
            mock.patch.object(
                MODULE.metadata, "version", side_effect=version_side_effect
            ),
            mock.patch.object(MODULE, "_parse_safetensors_schema", side_effect=race),
            self.assertRaisesRegex(MODULE.BundleError, "unexpected entry"),
        ):
            MODULE.build_bundle(
                str(self.source),
                str(output),
                "mixed-bf16",
                verify_only=True,
            )
        self.assertTrue(added)

    def test_verify_rejects_weight_mutation_between_manifest_hash_and_schema(
        self,
    ) -> None:
        _receipt, output, _api = self.build("affine-w8-g64")
        shard = output / "model.safetensors"
        original_parse = MODULE._parse_safetensors_schema
        mutated = False

        def race(
            path: Path,
            label: str,
            *,
            expected: MODULE.FileRecord | None = None,
        ) -> dict[str, tuple[str, tuple[int, ...]]]:
            nonlocal mutated
            if path == shard and not mutated:
                mutated = True
                with path.open("r+b") as handle:
                    handle.seek(-1, os.SEEK_END)
                    old = handle.read(1)
                    handle.seek(-1, os.SEEK_END)
                    handle.write(bytes([old[0] ^ 1]))
                    handle.flush()
                    os.fsync(handle.fileno())
            return original_parse(path, label, expected=expected)

        with (
            mock.patch.object(
                MODULE.metadata, "version", side_effect=version_side_effect
            ),
            mock.patch.object(MODULE, "_parse_safetensors_schema", side_effect=race),
            self.assertRaisesRegex(MODULE.BundleError, "manifest hash"),
        ):
            MODULE.build_bundle(
                str(self.source),
                str(output),
                "affine-w8-g64",
                verify_only=True,
            )
        self.assertTrue(mutated)

    def test_main_emits_exactly_one_canonical_json_line(self) -> None:
        _receipt, output, _api = self.build("mixed-bf16")
        stdout = io.StringIO()
        stderr = io.StringIO()
        with (
            mock.patch.object(
                MODULE.metadata, "version", side_effect=version_side_effect
            ),
            redirect_stdout(stdout),
            redirect_stderr(stderr),
        ):
            result = MODULE.main(
                [
                    "--source-dir",
                    str(self.source),
                    "--output-dir",
                    str(output),
                    "--mode",
                    "mixed-bf16",
                    "--verify-only",
                ]
            )
        self.assertEqual(result, 0)
        self.assertEqual(stderr.getvalue(), "")
        self.assertEqual(stdout.getvalue().count("\n"), 1)
        parsed = json.loads(stdout.getvalue())
        self.assertEqual(parsed["format"], MODULE.RECEIPT_FORMAT)
        self.assertEqual(
            stdout.getvalue(),
            json.dumps(
                parsed,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n",
        )


if __name__ == "__main__":
    unittest.main()
