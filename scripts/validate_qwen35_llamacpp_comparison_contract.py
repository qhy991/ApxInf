#!/usr/bin/env python3
"""Offline fail-closed validator for the frozen ApxInf/llama.cpp contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys


CONTRACT_FORMAT = "apxinf-qwen35-llamacpp-comparison-contract-v1"
VALIDATION_FORMAT = "apxinf-qwen35-llamacpp-comparison-validation-v1"
PINNED_CONTENT_SHA256 = (
    "23f46184dce0882ab15c6e7e0b87832d143194b80bf3929d5b5c13f5f2173d89"
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_TOP_LEVEL_FIELDS = {
    "format",
    "schema_version",
    "source_model",
    "workload",
    "apxinf",
    "llama_cpp",
    "comparison_tiers",
    "quality_protocol",
    "formal_protocol",
    "claims",
    "content_sha256",
}
_RAW_TOKEN_IDS = [
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
_SOURCE_MODEL = {
    "repo_id": "Qwen/Qwen3.5-0.8B",
    "revision": "2fc06364715b967f1860aea9cf38778875588b17",
    "checkpoint": {
        "name": "model.safetensors-00001-of-00001.safetensors",
        "sha256": (
            "04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696"
        ),
        "size": 1_746_942_600,
    },
}
_MODEL_ARTIFACTS = {
    "f32": {
        "name": "Qwen3.5-0.8B-2fc063647-F32.gguf",
        "sha256": (
            "69ad6b3ef11f0fb4d9af2d9f59a235c8576d9ef2e64b4375274ca35fc34530e4"
        ),
        "size": 3_020_533_248,
    },
    "pure_q8_0": {
        "name": "Qwen3.5-0.8B-2fc063647-Q8_0-llama.gguf",
        "parent_f32_sha256": (
            "69ad6b3ef11f0fb4d9af2d9f59a235c8576d9ef2e64b4375274ca35fc34530e4"
        ),
        "quantization": "llama.cpp-pure-Q8_0",
        "sha256": (
            "427555e797eefeb62e1c8ef71510ce062027b0c7f1674cc4cfcb352710e0908c"
        ),
        "size": 811_843_072,
    },
}
_FORMAL_SOURCE = {
    "repository": "https://github.com/ggml-org/llama.cpp.git",
    "commit": "f280b26983ad0fdb705a0d9ebf0503e76f2899b0",
    "tree": "21045aed8b426d7a5e25a98e646054cbd9487e81",
    "clean_detached_checkout_required": True,
    "build_admission": (
        "formal-only-after-new-executable-and-loaded-library-closure-hashes-are-captured"
    ),
}
_FORMAL_OBSERVED_BUILD = {
    "runner_schema": "apxinf.llama-cpp.raw-token-diagnostic.v2",
    "source": {
        "commit": "f280b26983ad0fdb705a0d9ebf0503e76f2899b0",
        "tree": "21045aed8b426d7a5e25a98e646054cbd9487e81",
        "clean_detached_checkout": True,
    },
    "inputs": {
        "runner_source": {
            "name": "benchmarks/llama_cpp/raw_token_runner.cpp",
            "sha256": (
                "76a5a354f729d22659387557ef368b75e83910e28a09d52876ddb366106c66e4"
            ),
            "size": 41_807,
        },
        "cmake_lists": {
            "name": "benchmarks/llama_cpp/CMakeLists.txt",
            "sha256": (
                "50c8bd83995b73f239dc4e3e4573127952e35d360f3584823a9529626904201d"
            ),
            "size": 5_757,
        },
    },
    "binary": {
        "name": "apxinf-llama-cpp-raw-token-runner",
        "build_type": "Release",
        "linkage": "static-llama-ggml-with-system-dynamic-libraries-only",
        "sha256": (
            "ccfa5ecd78119d4f8cdd8721e7faae360cb94b8334f9d61ed47e2e00290f2716"
        ),
        "size": 6_499_056,
    },
    "cmake_options": {
        "BUILD_SHARED_LIBS": False,
        "GGML_BACKEND_DL": False,
        "GGML_STATIC": False,
        "GGML_LTO": False,
        "GGML_CCACHE": False,
        "GGML_METAL": True,
        "GGML_METAL_EMBED_LIBRARY": True,
        "GGML_ACCELERATE": True,
        "GGML_CPU": True,
        "GGML_CPU_REPACK": True,
        "GGML_BLAS": True,
        "GGML_BLAS_VENDOR": "Apple",
        "GGML_LLAMAFILE": True,
        "GGML_NATIVE": True,
        "GGML_OPENMP": False,
        "GGML_METAL_NDEBUG": False,
        "GGML_METAL_SHADER_DEBUG": False,
        "GGML_CUDA": False,
        "GGML_MUSA": False,
        "GGML_HIP": False,
        "GGML_VULKAN": False,
        "GGML_WEBGPU": False,
        "GGML_ZDNN": False,
        "GGML_VIRTGPU": False,
        "GGML_VIRTGPU_BACKEND": False,
        "GGML_RPC": False,
        "GGML_SYCL": False,
        "GGML_OPENVINO": False,
        "GGML_ET": False,
        "GGML_OPENCL": False,
        "GGML_HEXAGON": False,
        "GGML_ZENDNN": False,
    },
    "otool_runtime_closure": {
        "non_system_dependencies": [],
        "classification": "otool-reports-system-frameworks-and-libraries-only",
        "dynamic_loader_symbols_present": ["dlopen", "dlsym"],
        "dynamic_loader_symbols_absent_claim_allowed": False,
        "claim_scope": (
            "runner-policy-disables-backend-load-all-and-default-scan; "
            "not-a-claim-that-dlopen-or-dlsym-symbols-are-absent"
        ),
    },
    "backend_policy": {
        "registration_mode": "linked-static-registry-only",
        "ggml_backend_load_all_called": False,
        "default_backend_scan_invoked": False,
        "backend_directory_option_policy": "rejected",
        "ggml_backend_path_policy": "must-be-absent-or-run-fails",
        "gpu_lane": {
            "device_selection": (
                "explicit-name-must-resolve-to-exactly-one-registered-gpu"
            ),
            "model_selected_device_count": 1,
            "transformer_layer_count": 24,
            "layers_on_selected_gpu": 24,
            "output_device_pointer_equals_selected_gpu": True,
            "gpu_memory_bytes_required": {
                "model": "strictly-positive",
                "context": "strictly-positive",
                "compute": "strictly-positive",
            },
        },
        "cpu_lane": {
            "model_selected_device_count": 0,
            "transformer_layer_count": 24,
            "layers_on_cpu": 24,
            "output_on_cpu": True,
            "gpu_memory_bytes_required": {
                "model": 0,
                "context": 0,
                "compute": 0,
            },
        },
        "kv_cache": {"cpu_lane": "f32", "gpu_lane": "f16"},
        "q8_0_metal_observed_placement": {
            "input_embedding_buffer_type": "CPU",
            "input_embedding_device_pointer": None,
            "cpu_model_buffer_bytes": 270_172_160,
            "pure_all_device_memory": False,
            "required_report_disclosure": (
                "Q8_0 Metal keeps the input embedding in a CPU fallback buffer; "
                "it is not pure all-device memory"
            ),
        },
    },
    "execution_placement_proof": {
        "method": "scheduler-callback-completed-sentinels-v1",
        "internal_api_binding": {
            "llama_cpp_commit": "f280b26983ad0fdb705a0d9ebf0503e76f2899b0",
            "experimental_internal_api": True,
            "headers": ["llama-context.h", "llama-ext.h", "llama-model.h"],
            "source_upgrade_policy": "rebuild-and-reaudit-required",
        },
        "phase_order": {
            "same_context_as_measured_generation": True,
            "proof_token": "sampled-token-128",
            "token_timing_complete_before_proof": True,
            "perf_counters_captured_before_proof": True,
            "proof_decode_count": 1,
            "timing_excluded": True,
            "excluded_from": [
                "token_ready_elapsed_ns",
                "generation_elapsed_ns",
                "measurement_scope_elapsed_ns",
                "llama_perf",
            ],
            "separately_recorded_as": (
                "post_measurement_execution_proof_elapsed_ns"
            ),
        },
        "scheduler_callback": {
            "requested_sentinel_count_ask_true": 26,
            "completion_callback_ask_value": False,
            "completed_sentinel_count_ask_false": 26,
            "input_sentinel": "model.input_embed",
            "layer_sentinel_pattern": "l_out-0..23",
            "layer_sentinel_count": 24,
            "output_sentinel": "result_output",
        },
        "gpu_lane_completion": {
            "model.input_embed": "CPU",
            "l_out-0..23": "MTL0",
            "result_output": "MTL0",
            "completed_on_cpu": 1,
            "completed_on_selected_gpu": 25,
        },
        "cpu_lane_completion": {
            "all_26_sentinels": "CPU",
            "completed_on_cpu": 26,
        },
        "fail_closed_on": [
            "proof-decode-error",
            "missing-sentinel",
            "duplicate-or-unexpected-callback",
            "wrong-backend",
        ],
        "receipt_contract": {
            "recorded": True,
            "passed": True,
            "timing_excluded": True,
            "decode_count": 1,
            "proof_token_id_recorded": True,
            "requested_sentinel_count": 26,
            "completed_sentinel_count": 26,
            "completed_input_embedding_on_cpu": True,
            "completed_transformer_layer_endpoints": 24,
            "completed_output_head": True,
            "backend_mismatch": False,
            "duplicate_or_unexpected_callback": False,
        },
    },
    "source_custody_eligible": True,
    "formal_campaign_eligible": False,
    "formal_campaign_blockers": [
        "apxinf-thread-policy-parity-not-established",
        "quiet-host-gate-not-passed",
    ],
}
_CONTEXT_CONTRACT = {
    "same_requested_context": True,
    "requested_context_length_each": 142,
    "pinned_llama_cpp_effective_context_length": 256,
    "pinned_llama_cpp_behavior": "implementation-rounds-request-142-up-to-256",
    "effective_context_equality_claim_allowed": False,
    "required_report_disclosure": (
        "both engines receive request 142; pinned llama.cpp Qwen3.5 reports "
        "effective 256 due to implementation rounding"
    ),
}
_BLOCK_ORDERS = ["ABBA", "BAAB", "ABBA", "BAAB", "ABBA", "BAAB"]


class ComparisonContractError(ValueError):
    """A fail-closed comparison-contract violation."""


def _fail(message):
    raise ComparisonContractError(message)


def canonical_bytes(value):
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def object_sha256(value):
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _reject_duplicate_keys(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            _fail(f"JSON contains duplicate key: {key}")
        value[key] = item
    return value


def _expect(actual, expected, label):
    if actual != expected:
        _fail(f"{label} drifted from the frozen comparison contract")


def _object(value, label):
    if not isinstance(value, dict):
        _fail(f"{label} must be an object")
    return value


def validate_contract(contract):
    contract = _object(contract, "contract")
    _expect(set(contract), _TOP_LEVEL_FIELDS, "top-level fields")
    _expect(contract.get("format"), CONTRACT_FORMAT, "format")
    _expect(contract.get("schema_version"), 1, "schema version")

    claimed_hash = contract.get("content_sha256")
    if not isinstance(claimed_hash, str) or not _SHA256.fullmatch(claimed_hash):
        _fail("content_sha256 must be a lowercase SHA-256")
    unsigned = dict(contract)
    del unsigned["content_sha256"]
    observed_hash = object_sha256(unsigned)
    _expect(observed_hash, claimed_hash, "content SHA-256")
    _expect(claimed_hash, PINNED_CONTENT_SHA256, "pinned contract SHA-256")

    _expect(contract.get("source_model"), _SOURCE_MODEL, "source model")
    workload = _object(contract.get("workload"), "workload")
    prompt = _object(workload.get("prompt"), "workload.prompt")
    _expect(prompt.get("raw_token_ids"), _RAW_TOKEN_IDS, "raw 13-token prompt")
    _expect(prompt.get("token_count"), 13, "raw prompt token count")
    _expect(prompt.get("ingress"), "raw-token-ids-only", "prompt ingress")
    _expect(
        prompt.get("token_ids_hash_encoding"),
        "sha256-canonical-compact-json-array-utf8-v1",
        "prompt token hash encoding",
    )
    _expect(prompt.get("token_ids_sha256"), object_sha256(_RAW_TOKEN_IDS), "prompt hash")
    generation = _object(workload.get("generation"), "workload.generation")
    _expect(generation.get("max_new_tokens"), 128, "generation length")
    _expect(generation.get("context_length"), 142, "context length")
    _expect(
        generation.get("context_arithmetic"),
        {"prompt_tokens": 13, "generated_tokens": 128, "spare_tokens": 1},
        "context arithmetic",
    )
    _expect(
        generation.get("context_contract"),
        _CONTEXT_CONTRACT,
        "requested/effective context disclosure",
    )
    _expect(generation.get("sampling"), "greedy-argmax", "sampling")
    _expect(generation.get("stop_policy"), "fixed-128-ignore-eos", "stop policy")
    _expect(
        generation.get("process_state"),
        "fresh-process-per-sample",
        "process state",
    )

    apxinf = _object(contract.get("apxinf"), "apxinf")
    lane = _object(apxinf.get("metal_w8_lane"), "ApxInf Metal W8 lane")
    _expect(
        lane.get("constructor"),
        "GeneralQwen35::from_weights_with_metal_w8_mlp_stack3_boundary_tail_head_v1",
        "ApxInf lane constructor",
    )
    _expect(
        lane.get("mechanism"),
        "metal-w8-mlp-stack3-boundary-tail-head-v1",
        "ApxInf mechanism",
    )
    _expect(lane.get("initial_stack_layers"), [0, 1, 2], "initial stack")
    _expect(
        lane.get("boundary_regions"),
        [
            {"full_attention_mlp_layer": 3, "linear_stack_layers": [4, 5, 6]},
            {"full_attention_mlp_layer": 7, "linear_stack_layers": [8, 9, 10]},
            {
                "full_attention_mlp_layer": 11,
                "linear_stack_layers": [12, 13, 14],
            },
            {
                "full_attention_mlp_layer": 15,
                "linear_stack_layers": [16, 17, 18],
            },
            {
                "full_attention_mlp_layer": 19,
                "linear_stack_layers": [20, 21, 22],
            },
        ],
        "boundary regions",
    )
    _expect(lane.get("tail_layer"), 23, "tail layer")
    quantization = _object(lane.get("quantization"), "ApxInf quantization")
    _expect(
        quantization.get("scheme"),
        "symmetric-signed-int8-per-row-group-round-clamp-minus127-plus127-f32-scale",
        "ApxInf W8 scheme",
    )
    _expect(
        quantization.get("linear_projection_group_sizes"),
        {
            "gdn_input": 64,
            "gdn_output": 32,
            "mlp_gate": 64,
            "mlp_up": 64,
            "mlp_down": 64,
        },
        "ApxInf W8 group sizes",
    )
    ledger = _object(lane.get("resident_metal_ledger"), "ApxInf Metal ledger")
    _expect(ledger.get("total_persistent_bytes"), 799_543_312, "Metal bytes")
    _expect(ledger.get("allocated_buffers"), 494, "Metal buffers")
    _expect(ledger.get("host_to_device_bytes_per_decode"), 28_672, "H2D bytes")
    _expect(ledger.get("device_to_host_bytes_per_decode"), 28_688, "D2H bytes")
    _expect(ledger.get("state_host_transfer_bytes_per_decode"), 0, "state transfer")
    _expect(ledger.get("command_buffers_per_decode"), 7, "command buffers")
    _expect(ledger.get("compute_encoders_per_decode"), 24, "encoders")
    _expect(ledger.get("kernel_dispatches_per_decode"), 267, "dispatches")

    llama = _object(contract.get("llama_cpp"), "llama_cpp")
    _expect(llama.get("model_artifacts"), _MODEL_ARTIFACTS, "GGUF artifacts")
    _expect(llama.get("formal_source"), _FORMAL_SOURCE, "formal llama.cpp source")
    _expect(
        llama.get("formal_observed_build"),
        _FORMAL_OBSERVED_BUILD,
        "formal observed llama.cpp build",
    )
    derivation = _object(llama.get("model_derivation"), "GGUF derivation")
    pure_q8 = _object(derivation.get("pure_q8_0"), "pure Q8_0 derivation")
    _expect(
        pure_q8.get("parent_f32_sha256"),
        _MODEL_ARTIFACTS["f32"]["sha256"],
        "pure Q8_0 parent",
    )
    _expect(pure_q8.get("quantization_type"), "Q8_0", "Q8_0 type")
    _expect(pure_q8.get("pure"), True, "Q8_0 pure flag")
    local = _object(llama.get("local_observed_build"), "local llama.cpp build")
    _expect(local.get("source_commit"), None, "local llama.cpp source commit")
    _expect(local.get("version_output"), "version: 0 (unknown)", "local version")
    _expect(local.get("formal_eligible"), False, "local formal eligibility")
    _expect(local.get("classification"), "diagnostic-only", "local classification")

    tiers = _object(contract.get("comparison_tiers"), "comparison tiers")
    _expect(set(tiers), {"f32_reference", "eight_bit_storage"}, "comparison tiers")
    eight_bit = _object(tiers.get("eight_bit_storage"), "eight-bit tier")
    _expect(
        eight_bit.get("quantization_mechanisms_equal"),
        False,
        "quantization equivalence",
    )
    _expect(eight_bit.get("weight_regimes_equal"), False, "weight-regime equivalence")

    quality = _object(contract.get("quality_protocol"), "quality protocol")
    _expect(quality.get("must_complete_before_formal_timing"), True, "quality ordering")
    _expect(quality["teacher_forced"].get("steps"), 128, "teacher-forced steps")
    _expect(quality["free_run"].get("steps"), 128, "free-run steps")
    _expect(
        quality.get("apxinf_metal_w8_admission"),
        "exact-teacher128-and-free128-versus-same-process-apxinf-cpu-f32-oracle",
        "ApxInf quality admission",
    )
    _expect(
        quality.get("speed_claim_must_not_imply_quality_parity"),
        True,
        "quality/performance separation",
    )

    formal = _object(contract.get("formal_protocol"), "formal protocol")
    schedule = _object(formal.get("schedule"), "formal schedule")
    _expect(schedule.get("block_orders"), _BLOCK_ORDERS, "ABBA/BAAB schedule")
    _expect(schedule.get("untimed_warmups_per_implementation"), 3, "warmups")
    _expect(schedule.get("timed_samples_total"), 24, "total timed samples")
    _expect(schedule.get("timed_samples_per_implementation"), 12, "per-lane samples")
    joined = "".join(_BLOCK_ORDERS)
    if len(joined) != 24 or joined.count("A") != 12 or joined.count("B") != 12:
        _fail("formal schedule is not a balanced 24-sample design")
    resources = _object(formal.get("resource_gates"), "resource gates")
    _expect(resources.get("process_group_rss_comparison"), "strictly-less-than", "RSS comparator")
    _expect(resources.get("process_group_rss_limit_bytes"), 6 * 1024**3, "RSS limit")
    _expect(resources.get("child_swaps_required"), 0, "child swaps")
    _expect(resources.get("system_swap_delta_bytes_required"), 0, "system swap delta")
    quiet = _object(formal.get("quiet_host_gate"), "quiet-host gate")
    _expect(quiet.get("preflight_sample_count"), 5, "quiet-host sample count")
    _expect(
        quiet.get("maximum_non_allowlisted_process_cpu_percent"),
        5.0,
        "quiet-host process threshold",
    )
    _expect(
        quiet.get("must_remain_valid_during_every_timed_sample"),
        True,
        "runtime quiet-host gate",
    )
    runtime_parity = _object(formal.get("runtime_parity"), "runtime parity")
    _expect(runtime_parity.get("same_thread_policy_required"), True, "thread parity")
    _expect(
        runtime_parity.get("current_thread_parity_status"),
        "blocked-apxinf-thread-policy-not-explicitly-controllable",
        "current thread-parity status",
    )
    _expect(formal.get("failure_policy"), "fail-closed-no-partial-formal-claim", "failure policy")

    claims = _object(contract.get("claims"), "claims")
    forbidden = claims.get("forbidden_claims")
    if not isinstance(forbidden, list) or not {
        "identical-quantization",
        "identical-weight-regime",
        "general-model-parity",
        "official-llama.cpp-version-from-version-zero-unknown",
        "identical-effective-context-allocation",
        "pure-all-device-q8_0-metal",
        "binary-has-no-dlopen-or-dlsym-symbols",
    }.issubset(set(forbidden)):
        _fail("required forbidden claims are missing")
    return contract


def load_contract(path):
    try:
        raw = Path(path).read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        _fail(f"cannot read contract: {error}")
    try:
        contract = json.loads(raw, object_pairs_hook=_reject_duplicate_keys)
    except json.JSONDecodeError as error:
        _fail(f"contract is not valid JSON: {error}")
    return validate_contract(contract)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        contract = load_contract(args.contract)
    except ComparisonContractError as error:
        print(f"comparison contract rejected: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "format": VALIDATION_FORMAT,
                "valid": True,
                "contract_sha256": contract["content_sha256"],
                "formal_llama_cpp_commit": contract["llama_cpp"]["formal_source"][
                    "commit"
                ],
                "formal_llama_cpp_binary_sha256": contract["llama_cpp"][
                    "formal_observed_build"
                ]["binary"]["sha256"],
                "formal_build_source_custody_eligible": contract["llama_cpp"][
                    "formal_observed_build"
                ]["source_custody_eligible"],
                "formal_campaign_eligible": contract["llama_cpp"][
                    "formal_observed_build"
                ]["formal_campaign_eligible"],
                "local_unknown_build_classification": contract["llama_cpp"][
                    "local_observed_build"
                ]["classification"],
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
