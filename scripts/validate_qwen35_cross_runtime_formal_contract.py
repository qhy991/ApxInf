#!/usr/bin/env python3
"""Fail-closed validator for the Qwen3.5 cross-runtime v3 predeclaration."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


CONTRACT_FORMAT = "apxinf-qwen35-cross-runtime-formal-predeclaration-v3"
VALIDATION_FORMAT = "apxinf-qwen35-cross-runtime-formal-validation-v3"
PINNED_CANONICAL_CONTRACT_SHA256 = (
    "6711abb9596ba75b087c5254e384ddd3df4452fadd1a144d800b6a8e9153ea4e"
)
_CAMPAIGN_ID = "qwen35-0.8b-cross-runtime-formal-v3-20260826"
_SCOPE_SHA256 = "5a9869cdd6866bef4c030ba98144d98963d2a03a14bcded6ee12f0f180f51d18"
_LINEAGE_SHA256 = "b2f6309af6b7e8fa7e06caf8dd414620ae0e445b4d44582563241f295bf42f62"
_SUBCAMPAIGN_IDS = {
    "NATIVE_A_VS_L": "qwen35-0.8b-native-apxinf-vs-llamacpp-formal-v3-20260826",
    "CORE_A_VS_L": "qwen35-0.8b-core-apxinf-vs-llamacpp-formal-v3-20260826",
    "GATEWAY_B_VS_G": "qwen35-0.8b-omniinfer-gateway-increment-formal-v3-20260826",
}
_STATISTICS_SHA256 = {
    "NATIVE_A_VS_L": "e715671fed9c4a9d8c49011042dd35b7125e6d1eb0387286b69b3ee43adb41e1",
    "CORE_A_VS_L": "1c5e5c04076d05697659c7e5be2742e4389d7413e4082e9397d30b5e060e579f",
    "GATEWAY_B_VS_G": "fb2381567e58c4c9e4d33ca022bc408f489f7443df9b30987ec0abce3f874f4d",
}
_CORE_CONTRACT_SHA256 = {
    "arm_A": "f35f7c1ae6f041343d94903323ba27412891d3e752b3b6624c71177e9ab338b9",
    "arm_L": "18420317b4871ca0943bc2a041430a9adcd0cd0b1eedf947b0790dcf48cac61a",
    "edge": "4788b24174256d4add11c5cf26a41f593a57640ef3917898973fc93f1ff0a038",
    "workload": "7309c169e306e6c3677fbccd267f64adf55589f068f75d8d4cb68383dab2ed70",
    "model_artifact": "02be7ecc6c02ab398292c776cfdb6b1acba7c1d47d789c8e04e6cf4bd54697e9",
    "prefill_and_recurrent_state": "b8497ce4c7965f2f23946411acc4ab9a02b768c77cb821f95fa88be9b244800c",
    "output_head_and_selection": "5ffddec7478e37c957fbce9a8ae587985e17bbd8e20dc621a94e838680a3a7f0",
    "execution": "0165a715c50e47903efc1b67ebe334589250bd678a1ba458b3166e8f65af6a8a",
}
_GATEWAY_CONTRACT_SHA256 = {
    "arm_B": "9e4ae268962dfe86c4853e7c5a943835124b621cd947a193bb0916e4979cb827",
    "arm_G": "473e476e15cec0fc12c8144571165a3b5ab5760c3f2772753da55e88e1bff850",
    "edge": "891abaaacd9b5cc382b843be3a4c9afd2cf175b9cad65d6caa36e5848da24b1f",
    "workload": "6cbbea28e0022bba95e943b92f5c3d4073892370875ceb14993f7a58587d1085",
    "runtime": "6fc68682b83633524b81a27b8d6fa66f4e840d0612da72830bf4967a913df820",
    "execution": "08cb4a5be8a812252dd9917472bdb459df90ff5fbbc6966bf217e65105a5ab60",
    "timing": "ebd4fad4c98d85cfbe9f67bc7067479596a5c929b9a046316e837b0aad1647ae",
}
_DYNAMIC_RECEIPT_SHA256 = (
    "ce1de36206d2a242c3680a7159d2a6dabb0967e9f10193565dbb1347b6ed7e0d"
)
_SUBCAMPAIGN_MARKER_BINDINGS_SHA256 = (
    "31f94e6568192d20b93d81752d5d09bb3f34bf60753cbb0854a61ccc48d6c3c4"
)
_FAILURE_CONTRACT_SHA256 = (
    "ac008e0db12bdd3b009505fcb18d430b39616d5efe313cb2981628c4e548745a"
)
_TEACHER_INPUT_TOKEN_IDS_SHA256 = (
    "c33b70a7626fbf3aaa9b8b09e03ce55b5d0e9a1b6ba7068d29067ccb6209a70d"
)
_CANONICAL_FREE128_PREFIX127_SHA256 = (
    "e771ed4facdc8d343b72ae59de586bf567fcc480a5d0edfc5e054cb68011742f"
)
_NATIVE_DEPLOYMENTS_SHA256 = (
    "d44bec19694ff2451c963136333d80b0ee5da9c4e86cc68d11d7625019a2b7f7"
)
_TOP_LEVEL_FIELDS = {
    "format",
    "schema_version",
    "campaign_id",
    "authored_at_utc",
    "document_role",
    "result_status",
    "sampling_state_at_authoring",
    "activation_contract",
    "lineage",
    "scope",
    "comparison_graph",
    "source_model_custody",
    "workload_contracts",
    "core_parity_contract",
    "native_deployment_contract",
    "runtime_custody",
    "execution_protocol",
    "timing_contract",
    "host_quiet_gate",
    "statistics_and_decisions",
    "failure_contract",
    "machine_receipt_contract",
    "claim_policy",
}


class FormalContractError(ValueError):
    """A fail-closed formal comparison predeclaration violation."""


def _fail(message: str) -> None:
    raise FormalContractError(message)


def _object(value: object, label: str) -> dict:
    if not isinstance(value, dict):
        _fail(f"{label} must be an object")
    return value


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict:
    value = {}
    for key, item in pairs:
        if key in value:
            _fail(f"JSON contains duplicate key: {key}")
        value[key] = item
    return value


def _reject_nonfinite_constant(value: str) -> object:
    _fail(f"non-finite JSON constant is forbidden: {value}")


def _required_strings(value: object, required: set[str], label: str) -> None:
    if (
        not isinstance(value, list)
        or any(not isinstance(item, str) for item in value)
        or not required.issubset(set(value))
    ):
        _fail(f"{label} is incomplete")


def _validate_core_timing(timing_value: object) -> None:
    timing = _object(timing_value, "timing boundary")
    if timing.get("clock") != "monotonic":
        _fail("timing boundary must use a monotonic clock")
    if timing.get("timestamp_resolution_and_clock_identity_must_be_recorded") is not True:
        _fail("timing boundary clock identity must be recorded")
    core = _object(timing.get("CORE_A_VS_L"), "CORE_A_VS_L timing boundary")
    expected = {
        "start": "immediately-before-first-raw-token-prefill-dispatch",
        "common_token_ready_boundary": "next-greedy-token-ready",
        "ttft_end": "first-next-greedy-token-ready",
        "end": "128th-next-greedy-token-ready",
        "logits_ready_is_a_valid_endpoint": False,
        "sampling_or_argmax_may_be_excluded": False,
        "device_completion_before_each_token_ready_timestamp": True,
        "final_sampled_token_decoded_inside_timed_region": False,
        "ttft_ms": "token_1_ready_time - prefill_start_time",
        "total_latency_ms": "token_128_ready_time - prefill_start_time",
        "tpot_ms": "(token_128_ready_time - token_1_ready_time) / 127",
        "generation_tps": "127000 / (token_128_ready_ms - token_1_ready_ms)",
        "primary_metric": "tpot_ms",
        "process_wall_time_is_primary": False,
        "model_load_time_is_primary": False,
    }
    if any(core.get(field) != expected_value for field, expected_value in expected.items()):
        _fail("CORE_A_VS_L timing boundary is not the frozen token-ready boundary")
    _required_strings(
        core.get("formal_ApxInf_selected_lane_boundary_includes"),
        {
            "device-complete exact-GGUF-Q8_0 full-vocabulary logits production",
            "full-vocabulary unbiased greedy argmax",
            "greedy token selection",
        },
        "CORE_A_VS_L timing boundary formal ApxInf selection",
    )
    _required_strings(
        core.get("current_nonadmitted_ApxInf_boundary_includes"),
        {
            "device-complete logits production",
            "top-4 candidate transfer",
            "F32 exact top-4 rerank",
            "greedy token selection",
        },
        "CORE_A_VS_L timing boundary current ApxInf selection",
    )
    if (
        core.get(
            "current_nonadmitted_ApxInf_boundary_must_still_include_top4_and_F32_rerank_in_any_diagnostic"
        )
        is not True
        or core.get("current_nonadmitted_ApxInf_boundary_is_formal_lane_eligible")
        is not False
    ):
        _fail("CORE_A_VS_L timing boundary misclassifies the current ApxInf lane")
    _required_strings(
        core.get("llama_cpp_boundary_includes"),
        {
            "device-complete logits production",
            "sampler or argmax execution",
            "greedy token selection",
        },
        "CORE_A_VS_L timing boundary llama.cpp selection",
    )


def _validate_native_timing(timing_value: object) -> None:
    timing = _object(timing_value, "native timing contract")
    native = _object(timing.get("NATIVE_A_VS_L"), "native timing boundary")
    expected = {
        "start": "immediately-before-first-raw-token-prefill-dispatch",
        "common_token_ready_boundary": "next-greedy-token-ready",
        "ttft_end": "first-next-greedy-token-ready",
        "end": "128th-next-greedy-token-ready",
        "internal_operations_are_identical": False,
        "endpoint_semantics_are_identical": True,
        "logits_ready_is_a_valid_endpoint": False,
        "sampling_argmax_top4_transfer_or_F32_rerank_may_be_excluded": False,
        "accelerator_completion_before_each_token_ready_timestamp": True,
        "final_sampled_token_decoded_inside_timed_region": False,
        "ttft_ms": "token_1_ready_time - prefill_start_time",
        "total_latency_ms": "token_128_ready_time - prefill_start_time",
        "tpot_ms": "(token_128_ready_time - token_1_ready_time) / 127",
        "generation_tps": "127000 / (token_128_ready_ms - token_1_ready_ms)",
        "primary_metric": "tpot_ms",
        "process_wall_time_is_primary": False,
        "model_load_time_is_primary": False,
    }
    if any(native.get(field) != value for field, value in expected.items()):
        _fail("native timing boundary is not the frozen next-greedy-token-ready boundary")
    _required_strings(
        native.get("ApxInf_boundary_includes"),
        {
            "CPU/F32 raw-token prefill and initial KV and GDN state production for TTFT",
            "all hybrid Metal W8 and CPU/F32 decode work",
            "CPU/F32 full-attention work",
            "top-4 candidate transfer",
            "F32 tied-embedding exact top-4 rerank",
            "greedy token selection",
        },
        "native timing ApxInf boundary",
    )
    _required_strings(
        native.get("llama_cpp_boundary_includes"),
        {
            "Q8_0 Metal raw-token prefill and initial F16 KV production for TTFT",
            "all Q8_0 Metal decode work and disclosed CPU fallback",
            "device-complete logits production",
            "sampler or full-vocabulary greedy argmax",
            "greedy token selection",
        },
        "native timing llama.cpp boundary",
    )


def _validate_core_parity(parity_value: object) -> None:
    parity = _object(parity_value, "core parity contract")
    if parity.get("selected_lane") != "MATCHED_GGUF_Q8_0_F16_KV_METAL_V1":
        _fail("core parity contract selected lane drifted")

    logical = _object(parity.get("logical_weights"), "core parity contract weights")
    logical_expected = {
        "same_source_revision_is_sufficient_for_equality": False,
        "same_serialized_container_bytes_required": False,
        "canonical_tensor_manifest_required_for_each_arm": True,
        "quantized_value_and_scale_hashes_are_null_only_for_unquantized_source_tensors": True,
        "logical_manifest_root_sha256_required": True,
        "A_and_L_logical_manifest_root_sha256_must_equal": True,
        "runtime_storage_manifest_required_per_arm": True,
        "runtime_storage_manifest_roots_may_differ": True,
        "ApxInf_must_trace_every_runtime_tensor_to_exact_GGUF_source_payload": True,
        "missing_extra_duplicated_or_requantized_tensor_allowed": False,
    }
    if any(
        logical.get(field) != expected
        for field, expected in logical_expected.items()
    ):
        _fail("core parity contract logical weight identity is weakened")
    _required_strings(
        logical.get("canonical_logical_tensor_manifest_fields"),
        {
            "canonical_tensor_name",
            "dimensions",
            "source_ggml_type",
            "element_count",
            "source_payload_size_bytes",
            "source_payload_sha256",
            "quantized_value_payload_sha256",
            "scale_payload_sha256",
        },
        "core parity contract logical tensor manifest",
    )
    _required_strings(
        logical.get("runtime_storage_manifest_fields"),
        {
            "canonical_tensor_name",
            "runtime_buffer_size_bytes",
            "runtime_buffer_sha256",
            "lossless_repack_recipe_id",
            "source_payload_sha256",
        },
        "core parity contract runtime storage manifest",
    )

    quantization = _object(
        parity.get("quantization"), "core parity contract quantization"
    )
    quantization_expected = {
        "scheme": "llama.cpp-GGML_TYPE_Q8_0",
        "same_scheme_required": True,
        "same_tensor_type_map_required": True,
        "same_block_size_required": True,
        "same_signed_q8_values_required": True,
        "same_f16_scale_bits_required": True,
        "same_unquantized_tensor_dtype_and_bits_required": True,
        "rounding_or_clamping_differences_allowed": False,
        "runtime_requantization_allowed": False,
        "lossless_memory_layout_repack_allowed": True,
        "dequantized_value_tolerance_is_not_a_substitute_for_payload_identity": True,
    }
    if any(
        quantization.get(field) != expected
        for field, expected in quantization_expected.items()
    ):
        _fail("core parity contract quantization is not exact GGUF Q8_0")

    kv_cache = _object(parity.get("kv_cache"), "core parity contract KV cache")
    kv_expected = {
        "key_dtype": "f16",
        "value_dtype": "f16",
        "same_dtype_required": True,
        "same_capacity_tokens": 256,
        "same_initial_logical_length": 0,
        "same_final_logical_length": 140,
        "kv_quantization_allowed": False,
        "prefix_or_cross_sample_reuse_allowed": False,
        "kv_offload_required": True,
        "same_physical_layout_required": False,
        "physical_layout_and_allocated_bytes_must_be_reported": True,
        "logical_value_bit_identity_required": False,
    }
    if any(kv_cache.get(field) != expected for field, expected in kv_expected.items()):
        _fail("core parity contract KV policy is not matched F16 KV")
    _validate_core_readiness(parity.get("current_readiness"))


def _validate_core_readiness(readiness_value: object) -> None:
    readiness = _object(readiness_value, "core parity readiness")
    for field in (
        "formally_admitted",
        "selected_lane_instantiable_now",
        "formal_campaign_may_start_now",
        "existing_custom_G32_G64_W8_hybrid_is_eligible",
    ):
        if readiness.get(field) is not False:
            _fail(f"core parity readiness {field} must remain false")
    if readiness.get("common_parameters_manifest_root_sha256") is not None:
        _fail("core parity readiness common manifest must remain unresolved here")
    if (
        readiness.get("null_common_parameters_hash_is_intentional_and_blocks_sampling")
        is not True
    ):
        _fail("core parity readiness null common manifest must block sampling")
    required_blockers = {
        "APXINF_EXACT_GGUF_Q8_0_LOSSLESS_LANE_NOT_YET_PROVEN",
        "MATCHED_TENSOR_MANIFEST_ROOTS_NOT_CAPTURED",
        "APXINF_Q8_0_PREFILL_STATE_PARITY_NOT_PROVEN",
        "APXINF_CPU_F32_PREFILL_DIFFERS_FROM_LLAMA_Q8_0_PREFILL",
        "APXINF_F32_GDN_STATE_POLICY_DIFFERS_FROM_REQUIRED_MATCHED_STATE_POLICY",
        "APXINF_F16_KV_PARITY_NOT_PROVEN",
        "APXINF_Q8_0_HEAD_ARGMAX_PARITY_NOT_PROVEN",
        "APXINF_F32_TIED_EMBEDDING_TOP4_RERANK_DIFFERS_FROM_LLAMA_Q8_0_FULL_VOCAB_ARGMAX",
        "EXACT_Q8_0_HEAD_AND_ARGMAX_PARITY_NOT_PROVEN",
        "APXINF_EXPLICIT_FOUR_THREAD_POLICY_NOT_PROVEN",
        "V3_CORE_DRIVER_AND_BINARY_HASHES_NOT_CAPTURED",
        "QUIET_HOST_GATE_NOT_YET_PASSED",
    }
    blocker_codes = readiness.get("blocker_codes")
    if not isinstance(blocker_codes, list) or not required_blockers.issubset(
        set(blocker_codes)
    ):
        _fail("core parity readiness blocker codes are incomplete")
    if (
        readiness.get(
            "matching_the_128_token_trajectory_does_not_clear_weight_prefill_state_kv_or_head_gates"
        )
        is not True
    ):
        _fail("core parity readiness cannot treat trajectory identity as parity proof")
    for field in (
        "existing_custom_G32_G64_W8_reason",
        "existing_prefill_and_state_reason",
        "existing_output_head_reason",
    ):
        if not isinstance(readiness.get(field), str) or not readiness[field]:
            _fail(f"core parity readiness {field} must disclose the mismatch")
    if readiness.get("future_resolution_may_be_proven_in_campaign_start_without_editing_this_contract") is not True:
        _fail("core parity readiness future proof binding is missing")


def _validate_balanced_blocks(edge: dict, label: str) -> None:
    orders = edge.get("timed_block_orders")
    if (
        not isinstance(orders, list)
        or not orders
        or any(order not in {"ABBA", "BAAB"} for order in orders)
        or set(orders) != {"ABBA", "BAAB"}
    ):
        _fail(f"{label} schedule must use both ABBA and BAAB blocks exclusively")
    if edge.get("timed_blocks") != len(orders):
        _fail(f"{label} schedule block count is inconsistent")
    if edge.get("timed_samples_per_block") != 4:
        _fail(f"{label} schedule blocks must each contain four samples")
    total = len(orders) * 4
    if edge.get("timed_samples_total") != total:
        _fail(f"{label} schedule total sample count is inconsistent")
    per_arm = total // 2
    if per_arm < 12 or edge.get("timed_samples_per_arm") != per_arm:
        _fail(f"{label} schedule must contain at least 12 samples per arm")


def _validate_execution_protocol(protocol_value: object) -> None:
    protocol = _object(protocol_value, "schedule")
    if protocol.get("campaigns_are_independent") is not True:
        _fail("schedule edges must remain independent campaigns")
    for field in (
        "fixed_schedule_mutation_allowed",
        "retry_allowed_after_campaign_start",
        "resampling_allowed",
        "replacement_allowed",
        "reordering_allowed",
        "outlier_removal_allowed",
        "sample_extension_after_looking_at_results_allowed",
    ):
        if protocol.get(field) is not False:
            _fail(f"schedule {field} must remain false")

    core = _object(protocol.get("CORE_A_VS_L"), "CORE_A_VS_L schedule")
    if core.get("process_state") != "fresh-process-per-sample":
        _fail("CORE_A_VS_L schedule must use fresh processes")
    if core.get("internal_timing_excludes_process_start_model_load_and_receipt_serialization") is not True:
        _fail("CORE_A_VS_L schedule timing exclusions drifted")
    warmups = core.get("untimed_warmups_per_arm")
    warmup_order = core.get("untimed_warmup_order")
    if (
        isinstance(warmups, bool)
        or not isinstance(warmups, int)
        or warmups < 3
        or not isinstance(warmup_order, list)
        or warmup_order.count("A") != warmups
        or warmup_order.count("B") != warmups
    ):
        _fail("CORE_A_VS_L schedule warmups are incomplete")
    if core.get("role_binding") != {
        "A": "ApxInf_future_exact_GGUF_Q8_0_F16_KV_core",
        "B": "pinned_llama_cpp_core",
    }:
        _fail("CORE_A_VS_L schedule role binding drifted")
    _validate_balanced_blocks(core, "CORE_A_VS_L")

    native = _object(protocol.get("NATIVE_A_VS_L"), "NATIVE_A_VS_L schedule")
    if (
        native.get("process_state") != "fresh-process-per-sample"
        or native.get(
            "internal_timing_excludes_process_start_model_load_and_receipt_serialization"
        )
        is not True
    ):
        _fail("NATIVE_A_VS_L schedule process or timing policy drifted")
    native_warmups = native.get("untimed_warmups_per_arm")
    native_warmup_order = native.get("untimed_warmup_order")
    if (
        native_warmups != 3
        or not isinstance(native_warmup_order, list)
        or native_warmup_order.count("A") != native_warmups
        or native_warmup_order.count("B") != native_warmups
    ):
        _fail("NATIVE_A_VS_L schedule warmups are incomplete")
    if native.get("role_binding") != {
        "A": "ApxInf_native_hybrid_G32_G64_W8_CPU_F32_remainder_F32_KV",
        "B": "pinned_llama_cpp_core",
    }:
        _fail("NATIVE_A_VS_L schedule role binding drifted")
    _validate_balanced_blocks(native, "NATIVE_A_VS_L")

    gateway = _object(
        protocol.get("GATEWAY_B_VS_G"), "GATEWAY_B_VS_G schedule"
    )
    odd = gateway.get("odd_macroblock_abstract_orders")
    even = gateway.get("even_macroblock_abstract_orders")
    for label, orders in (("odd", odd), ("even", even)):
        if (
            not isinstance(orders, list)
            or len(orders) != 2
            or set(orders) != {"ABBA", "BAAB"}
        ):
            _fail(f"GATEWAY_B_VS_G schedule {label} macroblocks are not balanced")
    macroblocks = gateway.get("timed_macroblock_count")
    if macroblocks != 16:
        _fail("GATEWAY_B_VS_G schedule must contain exactly 16 macroblocks")
    subblocks = macroblocks * 2
    if gateway.get("timed_subblocks_total") != subblocks:
        _fail("GATEWAY_B_VS_G schedule subblock count is inconsistent")
    if gateway.get("timed_samples_per_subblock") != 4:
        _fail("GATEWAY_B_VS_G schedule subblocks must contain four samples")
    total = subblocks * 4
    if gateway.get("timed_samples_total") != total:
        _fail("GATEWAY_B_VS_G schedule total sample count is inconsistent")
    per_arm = total // 2
    if per_arm != 64 or gateway.get("timed_samples_per_arm") != per_arm:
        _fail("GATEWAY_B_VS_G schedule must contain exactly 64 samples per arm")
    if gateway.get("role_binding") != {
        "A": "direct_resident_llama_server",
        "B": "omniinfer_gateway_to_same_resident_llama_server",
    }:
        _fail("GATEWAY_B_VS_G schedule role binding drifted")
    if gateway.get("cache_clear_and_admission_occur_outside_each_timed_interval") is not True:
        _fail("GATEWAY_B_VS_G schedule cache admission timing drifted")


def _validate_quiet_host(gate_value: object) -> None:
    gate = _object(gate_value, "quiet-host gate")
    top_expected = {
        "must_pass_before_create_new_campaign_start_marker": True,
        "failed_pre_marker_preflight_consumes_campaign": False,
        "may_retry_preflight_without_sampling_after_host_becomes_quiet": True,
        "power_source_required": "AC Power",
        "thermal_warning_allowed": False,
        "performance_warning_allowed": False,
        "failure_effect": "FORMAL_RESULT_REJECTED_NO_RANKING_NO_EQUIVALENCE",
    }
    if any(gate.get(field) != expected for field, expected in top_expected.items()):
        _fail("quiet-host gate top-level policy drifted")
    for phase in ("preflight", "postflight"):
        phase_gate = _object(gate.get(phase), f"quiet-host {phase}")
        count = phase_gate.get("snapshot_count")
        interval = phase_gate.get("snapshot_interval_ms")
        if (
            isinstance(count, bool)
            or not isinstance(count, int)
            or count < 5
            or isinstance(interval, bool)
            or not isinstance(interval, int)
            or interval < 250
            or phase_gate.get("every_snapshot_must_pass") is not True
        ):
            _fail(f"quiet-host {phase} sampling is too weak")
    postflight = gate["postflight"]
    cooldown = postflight.get("cooldown_before_snapshots_ms")
    if isinstance(cooldown, bool) or not isinstance(cooldown, int) or cooldown < 1000:
        _fail("quiet-host postflight cooldown is too short")

    monitor = _object(gate.get("continuous_monitor"), "quiet-host continuous monitor")
    monitor_interval = monitor.get("sample_interval_ms")
    if (
        isinstance(monitor_interval, bool)
        or not isinstance(monitor_interval, int)
        or monitor_interval < 1
        or monitor_interval > 250
        or monitor.get("starts_before_first_warmup") is not True
        or monitor.get("ends_after_postflight") is not True
        or monitor.get("every_snapshot_must_pass") is not True
    ):
        _fail("quiet-host continuous monitor is incomplete")

    process = _object(gate.get("process_policy"), "quiet-host process policy")
    if process.get("allowlist_scope") != (
        "exact campaign orchestrator process group, active measured runtime "
        "process group, and the custody monitor only"
    ):
        _fail("quiet-host process allowlist_scope drifted")
    thresholds = (
        ("maximum_single_non_allowlisted_process_cpu_percent", 10.0),
        ("maximum_aggregate_non_allowlisted_process_cpu_percent", 25.0),
    )
    for field, maximum in thresholds:
        value = process.get(field)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or value <= 0.0
            or value > maximum
        ):
            _fail(f"quiet-host process threshold {field} is too weak")
    if (
        process.get("cpu_percent_window_ms") != 250
        or process.get("user_or_system_process_termination_by_driver_allowed")
        is not False
    ):
        _fail("quiet-host process measurement or termination policy drifted")

    system = _object(gate.get("system_policy"), "quiet-host system policy")
    system_expected = {
        "maximum_load_average_per_logical_cpu": 0.5,
        "system_swap_delta_bytes_required": 0,
        "campaign_process_swap_bytes_required": 0,
        "memory_pressure_pages_throttled_delta_required": 0,
        "power_or_thermal_state_change_allowed": False,
    }
    if any(system.get(field) != expected for field, expected in system_expected.items()):
        _fail("quiet-host system policy drifted")


def _validate_native_deployment(
    graph_value: object, native_value: object, statistics_value: object
) -> None:
    graph = _object(graph_value, "native deployment comparison graph")
    arms = _object(graph.get("arms"), "native deployment arms")
    if arms.get("AN") != {
        "name": "ApxInf_native_hybrid_G32_G64_W8_CPU_F32_remainder_F32_KV",
        "kind": "named_native_deployment",
        "transport_in_primary_timing": "none",
    }:
        _fail("native deployment AN arm identity drifted")
    edges = _object(graph.get("edges"), "native deployment edges")
    edge = _object(edges.get("NATIVE_A_VS_L"), "native deployment edge")
    edge_expected = {
        "members": ["AN", "L"],
        "estimand": "performance ratio of two explicitly named native deployment configurations",
        "workload_id": "NATIVE_RAW13_FREE128_V3",
        "rankable_only_after_all_declared_native-deployment_gates": True,
        "engine_only_ranking_edge": False,
        "identical_weight_quantization_kv_or_placement_edge": False,
        "omniinfer_gateway_included": False,
    }
    if any(edge.get(field) != expected for field, expected in edge_expected.items()):
        _fail("native deployment edge scope drifted")

    native = _object(native_value, "native deployment contract")
    native_expected = {
        "comparison_edge": "NATIVE_A_VS_L",
        "comparison_class": "FORMAL_NAMED_DEPLOYMENT_COMPARISON_WITH_DISCLOSED_NUMERICAL_REGIME_DIFFERENCES",
        "same_model_revision_required": True,
        "same_raw_prompt_and_generation_contract_required": True,
        "same_teacher_forced_128_and_free_run_128_admission_required": True,
        "same_next_greedy_token_ready_timing_boundary_required": True,
        "same_weights_quantization_kv_or_placement_claim_allowed": False,
    }
    if any(native.get(field) != expected for field, expected in native_expected.items()):
        _fail("native deployment contract scope drifted")

    deployments = _object(native.get("deployments"), "native deployment identities")
    if _canonical_sha256(deployments) != _NATIVE_DEPLOYMENTS_SHA256:
        _fail("native deployment identity definitions drifted")
    if set(deployments) != {"AN", "L"}:
        _fail("native deployment identities are incomplete")
    arm_a = _object(deployments["AN"], "native deployment AN")
    arm_l = _object(deployments["L"], "native deployment L")
    if arm_a.get("configuration_id") != "ApxInf-native-hybrid-G32-G64-W8-CPU-F32-remainder-F32-KV-v3":
        _fail("native deployment AN configuration identity drifted")
    if arm_l.get("configuration_id") != "llama.cpp-f280b269-Q8_0-Metal-F16-KV-threads4-v3":
        _fail("native deployment L configuration identity drifted")
    if _object(arm_a.get("weight_regime"), "native deployment AN weights").get(
        "classification"
    ) != "custom-hybrid-not-GGUF-Q8_0":
        _fail("native deployment AN weight regime is not disclosed")
    if _object(arm_l.get("weight_regime"), "native deployment L weights").get(
        "classification"
    ) != "llama.cpp-pure-Q8_0":
        _fail("native deployment L weight regime is not disclosed")
    if _object(arm_a.get("prefill"), "native deployment AN prefill") != {
        "device": "CPU",
        "weight_and_compute_precision": "F32",
    }:
        _fail("native deployment AN prefill regime is not disclosed")
    if _object(arm_a.get("kv_cache"), "native deployment AN KV cache") != {
        "key_dtype": "F32",
        "value_dtype": "F32",
        "capacity_tokens": 256,
        "empty_before_each_sample": True,
    }:
        _fail("native deployment AN F32 KV cache is not disclosed")
    if _object(arm_l.get("kv_cache"), "native deployment L KV cache") != {
        "key_dtype": "F16",
        "value_dtype": "F16",
        "capacity_tokens": 256,
        "empty_before_each_sample": True,
    }:
        _fail("native deployment L F16 KV cache is not disclosed")
    a_decode = _object(arm_a.get("decode"), "native deployment AN decode")
    l_decode = _object(arm_l.get("decode"), "native deployment L decode")
    if a_decode.get("head") != "F32 tied embedding top-4 exact rerank":
        _fail("native deployment AN head mechanism is not disclosed")
    if l_decode.get("head") != "GGUF artifact head path":
        _fail("native deployment L head mechanism is not disclosed")

    disclosures = _object(
        native.get("machine_disclosures"), "native deployment disclosures"
    )
    disclosure_expected = {
        "same_source_revision": True,
        "same_source_checkpoint_lineage": True,
        "same_serialized_artifact": False,
        "same_logical_weight_payload": False,
        "same_quantization_scheme": False,
        "same_prefill_precision_or_placement": False,
        "same_KV_dtype": False,
        "same_output_head_mechanism": False,
        "same_effective_CPU_thread_policy_claimed": False,
        "same_128_token_free_run_trajectory_required": True,
        "same_128_step_teacher_argmax_trajectory_required": True,
        "same_timing_endpoint_required": True,
        "same_physical_host_required": True,
    }
    if any(
        disclosures.get(field) != expected
        for field, expected in disclosure_expected.items()
    ):
        _fail("native deployment difference disclosures are incomplete")

    readiness = _object(native.get("current_readiness"), "native deployment readiness")
    readiness_expected = {
        "named_deployment_implementations_exist": True,
        "formally_admitted": False,
        "formal_campaign_may_start_now": False,
        "existing_v2_free_run_receipts_reusable_as_formal_v3_samples": False,
        "future_resolution_may_be_proven_without_changing_named_deployment_semantics": True,
    }
    if any(readiness.get(field) != expected for field, expected in readiness_expected.items()):
        _fail("native deployment readiness must remain fail closed")
    blockers = readiness.get("blocker_codes")
    required_blockers = {
        "LLAMA_CPP_TEACHER_FORCED_128_RECEIPT_NOT_CAPTURED",
        "V3_NATIVE_DRIVER_AND_BINARY_HASHES_NOT_CAPTURED",
        "NATIVE_SUBCAMPAIGN_NOT_PUBLICLY_ACTIVATED",
        "QUIET_HOST_GATE_NOT_YET_PASSED",
    }
    if not isinstance(blockers, list) or not required_blockers.issubset(set(blockers)):
        _fail("native deployment readiness blockers are incomplete")

    claim_scope = _object(native.get("claim_scope"), "native deployment claim scope")
    for field in (
        "engine_only_conclusion_allowed",
        "identical_quantization_or_KV_conclusion_allowed",
        "general_model_or_prompt_conclusion_allowed",
    ):
        if claim_scope.get(field) is not False:
            _fail(f"native deployment claim scope {field} must remain false")
    if not isinstance(claim_scope.get("allowed_subject"), str) or not claim_scope[
        "allowed_subject"
    ]:
        _fail("native deployment claim subject is missing")

    statistics = _object(statistics_value, "native deployment statistics")
    native_statistics = _object(
        statistics.get("NATIVE_A_VS_L"), "native deployment statistics"
    )
    if (
        native_statistics.get("ratio_subject_A") != arm_a["configuration_id"]
        or native_statistics.get("ratio_subject_L") != arm_l["configuration_id"]
        or native_statistics.get("engine_only_winner_claim_allowed") is not False
        or native_statistics.get(
            "identical_weight_quantization_kv_or_placement_claim_allowed"
        )
        is not False
    ):
        _fail("native deployment statistics broaden the named-configuration claim")


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_statistics(statistics_value: object) -> None:
    statistics = _object(statistics_value, "statistics and decisions")
    if set(statistics) != set(_STATISTICS_SHA256):
        _fail("statistics and decisions edge set is incomplete or unknown")
    for edge, expected_hash in _STATISTICS_SHA256.items():
        if _canonical_sha256(statistics.get(edge)) != expected_hash:
            _fail(f"statistics and decisions for {edge} drifted")


def _validate_exact_core_contract(
    graph_value: object, workloads_value: object, parity_value: object
) -> None:
    graph = _object(graph_value, "CORE contract graph")
    arms = _object(graph.get("arms"), "CORE contract arms")
    edges = _object(graph.get("edges"), "CORE contract edges")
    workloads = _object(workloads_value, "CORE contract workloads")
    parity = _object(parity_value, "CORE contract parity")
    objects = {
        "arm_A": arms.get("A"),
        "arm_L": arms.get("L"),
        "edge": edges.get("CORE_A_VS_L"),
        "workload": workloads.get("CORE_RAW13_FREE128_V3"),
        "model_artifact": parity.get("model_artifact"),
        "prefill_and_recurrent_state": parity.get("prefill_and_recurrent_state"),
        "output_head_and_selection": parity.get("output_head_and_selection"),
        "execution": parity.get("execution"),
    }
    for label, value in objects.items():
        if _canonical_sha256(value) != _CORE_CONTRACT_SHA256[label]:
            _fail(f"CORE contract {label} drifted")


def _validate_exact_gateway_contract(
    graph_value: object,
    workloads_value: object,
    runtime_value: object,
    execution_value: object,
    timing_value: object,
) -> None:
    graph = _object(graph_value, "gateway contract graph")
    arms = _object(graph.get("arms"), "gateway contract arms")
    edges = _object(graph.get("edges"), "gateway contract edges")
    workloads = _object(workloads_value, "gateway contract workloads")
    runtime = _object(runtime_value, "gateway contract runtime")
    execution = _object(execution_value, "gateway contract execution")
    timing = _object(timing_value, "gateway contract timing")
    objects = {
        "arm_B": arms.get("B"),
        "arm_G": arms.get("G"),
        "edge": edges.get("GATEWAY_B_VS_G"),
        "workload": workloads.get("GATEWAY_RAW13_FREE128_V3"),
        "runtime": runtime.get("gateway_cohort"),
        "execution": execution.get("GATEWAY_B_VS_G"),
        "timing": timing.get("GATEWAY_B_VS_G"),
    }
    error_labels = {
        "runtime": "runtime custody",
        "execution": "execution schedule",
    }
    for label, value in objects.items():
        if _canonical_sha256(value) != _GATEWAY_CONTRACT_SHA256[label]:
            _fail(f"gateway contract {error_labels.get(label, label)} drifted")


def _validate_campaign_binding(
    campaign_id: object,
    scope_value: object,
    lineage_value: object,
    graph_value: object,
    failure_value: object,
) -> None:
    if campaign_id != _CAMPAIGN_ID:
        _fail("campaign binding campaign_id drifted")
    if _canonical_sha256(scope_value) != _SCOPE_SHA256:
        _fail("campaign binding scope drifted")
    if _canonical_sha256(lineage_value) != _LINEAGE_SHA256:
        _fail("campaign binding lineage drifted")
    graph = _object(graph_value, "campaign binding graph")
    edges = _object(graph.get("edges"), "campaign binding edges")
    failure = _object(failure_value, "campaign binding failure contract")
    markers = _object(
        failure.get("subcampaign_markers"), "campaign binding markers"
    )
    for edge, expected_id in _SUBCAMPAIGN_IDS.items():
        edge_contract = _object(edges.get(edge), f"campaign binding {edge} edge")
        marker = _object(markers.get(edge), f"campaign binding {edge} marker")
        if (
            edge_contract.get("subcampaign_id") != expected_id
            or marker.get("subcampaign_id") != expected_id
            or edge_contract.get("subcampaign_id") != marker.get("subcampaign_id")
        ):
            _fail(f"campaign binding {edge} edge and marker IDs disagree")


def _validate_free_trajectory(
    admission_value: object, expected_hash: str, pair_field: str, label: str
) -> None:
    admission = _object(admission_value, f"workload {label} trajectory")
    expected = {
        "required_for_every_warmup_and_timed_sample": True,
        "generated_token_ids_must_be_recorded": True,
        "generated_token_ids_count": 128,
        "canonicalization": "UTF-8 bytes of compact JSON array with no whitespace",
        "expected_sha256": expected_hash,
        pair_field: True,
        "hash_only_without_raw_ids_is_sufficient": False,
    }
    if any(admission.get(field) != value for field, value in expected.items()):
        _fail(f"workload {label} trajectory admission drifted")


def _validate_workloads(workloads_value: object) -> None:
    workloads = _object(workloads_value, "workload contracts")
    prompt = _object(workloads.get("shared_prompt"), "workload shared prompt")
    token_ids = prompt.get("token_ids")
    if (
        not isinstance(token_ids, list)
        or len(token_ids) != 13
        or any(
            isinstance(token, bool) or not isinstance(token, int) or token < 0
            for token in token_ids
        )
        or prompt.get("ingress_semantics") != "raw-token-ids"
        or prompt.get("token_count") != len(token_ids)
        or prompt.get("canonicalization")
        != "UTF-8 bytes of compact JSON array with no whitespace"
        or prompt.get("sha256") != _canonical_sha256(token_ids)
        or prompt.get("sha256")
        != "4b890fa15ee3d7db4e9dd18bd79c6362d40e9e016ae4f9f74cb7fc420ef3b6d3"
    ):
        _fail("workload shared raw prompt is not exactly pinned")

    core = _object(
        workloads.get("CORE_RAW13_FREE128_V3"), "workload CORE_A_VS_L"
    )
    if (
        core.get("arms") != ["A", "L"]
        or core.get("prompt_binding") != "workload_contracts.shared_prompt"
        or core.get("both_arms_receive_raw_token_ids_directly") is not True
    ):
        _fail("workload CORE_A_VS_L raw-token ingress drifted")
    core_generation = _object(core.get("generation"), "workload CORE generation")
    generation_expected = {
        "generated_token_count": 128,
        "sampling": "unbiased-greedy-argmax",
        "temperature": 0,
        "seed_effect": "none",
        "eog_policy": "select-and-feed-back-eog-without-termination-and-without-eog-logit-suppression",
        "speculative_decoding_allowed": False,
        "continuous_batching_allowed": False,
        "sequence_count": 1,
        "requested_context_tokens": 256,
        "effective_context_tokens": 256,
        "requested_batch_tokens": 13,
        "effective_batch_tokens": 13,
        "requested_ubatch_tokens": 13,
        "effective_ubatch_tokens": 13,
        "prompt_cache_reuse_allowed": False,
    }
    if any(
        core_generation.get(field) != expected
        for field, expected in generation_expected.items()
    ) or core_generation.get("empty_kv_before_prefill") is not True:
        _fail("workload CORE generation semantics drifted")
    core_hash = "2042d5522ed5e768b938ac3fd9e19d3936dfc6e685fc157d41a8e8132c5b42fe"
    _validate_free_trajectory(
        core.get("trajectory_admission"),
        core_hash,
        "pairwise_A_L_token_ids_must_be_bitwise_equal",
        "CORE_A_VS_L",
    )

    gateway = _object(
        workloads.get("GATEWAY_RAW13_FREE128_V3"), "workload GATEWAY_B_VS_G"
    )
    if (
        gateway.get("arms") != ["B", "G"]
        or gateway.get("prompt_binding") != "workload_contracts.shared_prompt"
        or gateway.get("backend_rendered_prompt_token_ids_must_equal_shared_prompt")
        is not True
    ):
        _fail("workload GATEWAY_B_VS_G prompt binding drifted")
    gateway_hash = "0a8a6c5ceeb831528480ebcad172fbcdda4ac23478ab051b1f74a00ec6d4f8e4"
    _validate_free_trajectory(
        gateway.get("trajectory_admission"),
        gateway_hash,
        "pairwise_B_G_token_ids_must_be_bitwise_equal",
        "GATEWAY_B_VS_G",
    )
    gateway_admission = gateway["trajectory_admission"]
    if gateway_admission.get("pairwise_content_and_usage_must_be_equal") is not True:
        _fail("workload GATEWAY_B_VS_G content and usage equality is missing")
    if (
        gateway.get("same_as_core_trajectory") is not False
        or gateway.get("cross_cohort_trajectory_comparison_allowed") is not False
    ):
        _fail("workload gateway and core trajectories must not be joined")

    native = _object(
        workloads.get("NATIVE_RAW13_FREE128_V3"), "workload NATIVE_A_VS_L"
    )
    if (
        native.get("arms") != ["AN", "L"]
        or native.get("prompt_binding") != "workload_contracts.shared_prompt"
        or native.get("both_arms_receive_raw_token_ids_directly") is not True
    ):
        _fail("workload NATIVE_A_VS_L raw-token ingress drifted")
    native_generation = _object(
        native.get("generation"), "workload NATIVE generation"
    )
    if any(
        native_generation.get(field) != expected
        for field, expected in generation_expected.items()
    ) or native_generation.get("empty_kv_and_GDN_state_before_prefill") is not True:
        _fail("workload NATIVE generation semantics drifted")
    teacher = _object(
        native.get("teacher_forced_admission"),
        "native teacher-forced workload admission",
    )
    teacher_ids = teacher.get("teacher_input_token_ids")
    teacher_binding_expected = {
        "teacher_input_derivation": "shared_prompt.token_ids[-1] followed by the first 127 token IDs of the frozen canonical free-run trajectory",
        "teacher_input_derivation_shared_prompt_last_token_id": token_ids[-1],
        "teacher_input_derivation_free_run_trajectory_sha256": core_hash,
        "teacher_input_token_ids_count": 128,
        "teacher_input_token_ids_canonicalization": "UTF-8 bytes of compact JSON array with no whitespace",
        "teacher_input_token_ids_sha256": _TEACHER_INPUT_TOKEN_IDS_SHA256,
        "teacher_input_derivation_must_be_recomputed_and_match_before_each_receipt": True,
    }
    if (
        any(
            teacher.get(field) != expected
            for field, expected in teacher_binding_expected.items()
        )
        or not isinstance(teacher_ids, list)
        or len(teacher_ids) != 128
        or any(
            isinstance(token, bool) or not isinstance(token, int) or token < 0
            for token in teacher_ids
        )
        or teacher_ids[0] != token_ids[-1]
        or _canonical_sha256(teacher_ids) != _TEACHER_INPUT_TOKEN_IDS_SHA256
        or _canonical_sha256(teacher_ids[1:])
        != _CANONICAL_FREE128_PREFIX127_SHA256
    ):
        _fail("native teacher prebinding to prompt and canonical free run drifted")
    teacher_expected = {
        "must_complete_before_timed_campaign_marker": True,
        "steps": 128,
        "reference": "same-revision ApxInf CPU/F32 oracle",
        "same_teacher_input_token_ids_for_AN_and_L": True,
        "AN_zero_argmax_mismatches_required": True,
        "L_zero_argmax_mismatches_required": True,
        "AN_and_L_observed_argmax_token_ids_must_be_bitwise_equal": True,
        "receipts_must_be_committed_and_pushed_before_performance_sampling": True,
        "prior_v2_llama_teacher_receipt_exists": False,
        "prior_free_run_identity_is_not_a_substitute": True,
    }
    if any(teacher.get(field) != expected for field, expected in teacher_expected.items()):
        _fail("native teacher-forced workload admission drifted")
    required_receipt_fields = {
        "teacher_input_token_ids",
        "reference_argmax_token_ids",
        "observed_argmax_token_ids",
        "mismatch_positions",
        "first_mismatch",
        "reference_receipt_size_bytes",
        "reference_receipt_sha256",
        "runtime_receipt_size_bytes",
        "runtime_receipt_sha256",
    }
    receipt_fields = teacher.get("required_receipt_fields")
    if not isinstance(receipt_fields, list) or not required_receipt_fields.issubset(
        set(receipt_fields)
    ):
        _fail("native teacher-forced workload receipt fields are incomplete")
    _validate_free_trajectory(
        native.get("free_run_trajectory_admission"),
        core_hash,
        "pairwise_AN_L_token_ids_must_be_bitwise_equal",
        "NATIVE_A_VS_L free run",
    )


def _validate_activation_and_custody(
    activation_value: object, source_value: object, runtime_value: object
) -> None:
    activation = _object(activation_value, "activation contract")
    activation_expected = {
        "path": "configs/qwen35-0.8b-cross-runtime-formal-v3.json",
        "repository": "https://github.com/qhy991/ApxInf.git",
        "authored_from_commit": "df5c55a06140107b24964d9e4abbefd9fa2a4733",
        "predeclaration_must_be_committed_and_pushed_before_any_v3_generation_request": True,
        "campaign_start_receipt_must_record_predeclaration_commit_size_and_sha256": True,
        "campaign_start_receipt_must_prove_head_equals_origin_main": True,
        "frozen_origin_remote_url": "https://github.com/qhy991/ApxInf.git",
        "frozen_live_remote_ref": "refs/heads/main",
        "remote_origin_url_normalization": {
            "allowed": False,
            "rule_id": "EXACT_UTF8_BYTE_EQUALITY_NO_NORMALIZATION",
            "required_value": "https://github.com/qhy991/ApxInf.git",
        },
        "campaign_start_receipt_must_record_actual_remote_origin_url": True,
        "campaign_start_receipt_must_prove_remote_origin_url_exact": True,
        "campaign_start_receipt_must_run_live_ls_remote": True,
        "campaign_start_receipt_live_ls_remote_argv": [
            "git",
            "ls-remote",
            "--exit-code",
            "https://github.com/qhy991/ApxInf.git",
            "refs/heads/main",
        ],
        "campaign_start_receipt_must_prove_head_activation_and_live_remote_main_equal": True,
        "local_tracking_ref_equality_is_sufficient_publication_proof": False,
        "campaign_start_receipt_must_prove_clean_worktree": True,
        "editing_this_contract_after_the_first_v3_generation_request_allowed": False,
        "activation_is_not_proven_by_this_document_alone": True,
    }
    if activation != activation_expected:
        _fail(
            "activation live remote publication contract does not bind the frozen "
            "GitHub URL and live refs/heads/main before sampling"
        )

    source = _object(source_value, "source model custody")
    if source.get("checkpoint") != {
        "name": "model.safetensors-00001-of-00001.safetensors",
        "size_bytes": 1_746_942_600,
        "sha256": "04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696",
    }:
        _fail("source model custody checkpoint drifted")
    if (
        source.get("tokenizer_and_config_manifest_sha256_required") is not True
        or source.get("network_download_during_campaign_allowed") is not False
        or source.get("artifact_mutation_after_campaign_start_allowed") is not False
    ):
        _fail("source model custody policy drifted")

    runtime = _object(runtime_value, "runtime custody")
    base_commit = "df5c55a06140107b24964d9e4abbefd9fa2a4733"
    for name, extra_null_fields in (
        ("ApxInf_native", {"packed_weight_and_resident_buffer_manifest_sha256"}),
        ("ApxInf_exact_Q8_core", set()),
    ):
        lane = _object(runtime.get(name), f"runtime custody {name}")
        expected = {
            "repository": "https://github.com/qhy991/ApxInf.git",
            "predeclaration_base_commit": base_commit,
            "campaign_commit_required": True,
            "campaign_tree_required": True,
            "clean_checkout_required": True,
            "runner_source_path": "benchmarks/cross_runtime/formal_v3_driver.py",
            "runner_source_sha256": None,
            "release_binary_sha256": None,
            "loaded_non_system_library_closure_sha256": None,
            "null_hashes_block_sampling": True,
        }
        if any(lane.get(field) != value for field, value in expected.items()):
            _fail(f"runtime custody {name} unresolved hashes must block sampling")
        if any(lane.get(field) is not None for field in extra_null_fields):
            _fail(f"runtime custody {name} unresolved manifest must block sampling")

    llama = _object(runtime.get("pinned_llama_cpp_core"), "runtime custody llama.cpp")
    llama_expected = {
        "repository": "https://github.com/ggml-org/llama.cpp.git",
        "source_commit": "f280b26983ad0fdb705a0d9ebf0503e76f2899b0",
        "source_tree": "21045aed8b426d7a5e25a98e646054cbd9487e81",
        "clean_detached_checkout_required": True,
        "runner_binary_name": "apxinf-llama-cpp-raw-token-runner",
        "runner_binary_size_bytes": 6_499_056,
        "runner_binary_sha256": "ccfa5ecd78119d4f8cdd8721e7faae360cb94b8334f9d61ed47e2e00290f2716",
        "loaded_non_system_library_closure_sha256_required": True,
        "GGML_BACKEND_PATH_must_be_unset": True,
        "dynamic_backend_scan_allowed": False,
    }
    if any(llama.get(field) != expected for field, expected in llama_expected.items()):
        _fail("runtime custody pinned llama.cpp identity drifted")

    gateway = _object(runtime.get("gateway_cohort"), "runtime custody gateway cohort")
    omni = _object(gateway.get("omniinfer"), "runtime custody OmniInfer")
    omni_expected = {
        "repository": "https://github.com/omnimind-ai/OmniInfer.git",
        "release": "v0.3.26",
        "source_commit": "79af77228f329a79ac665014089e23983e69e79f",
        "release_archive_sha256": "0f83ea36aad7126976ff2a53a58f0ce20e934d2bd0133d40f4ce974658a48cf4",
        "cli_size_bytes": 9_719_136,
        "cli_sha256": "65487424ca9179850b80079beafa5ad69a66e0841d328ee8dd8a1fd4b613d661",
    }
    if any(omni.get(field) != expected for field, expected in omni_expected.items()):
        _fail("runtime custody OmniInfer identity drifted")
    backend = _object(gateway.get("backend"), "runtime custody gateway backend")
    backend_expected = {
        "runtime": "llama.cpp-mac",
        "release": "b10280",
        "source_commit": "61881b1f7f0b13d9e46d561fc25afcd6bbaec479",
        "release_archive_sha256": "5dc4b11192ef34895c7f92a9f1dd3bd3d5864a63976ea2327fe2e0944891cb75",
        "llama_server_size_bytes": 33_472,
        "llama_server_sha256": "02723fc39fbeebd9849ce4c9ca3799649df3cf91f101c2cd56b8756e1db54d28",
        "model_artifact_sha256": "427555e797eefeb62e1c8ef71510ce062027b0c7f1674cc4cfcb352710e0908c",
        "same_pid_start_time_argv_environment_and_loaded_model_fd_for_B_and_G": True,
        "same_runtime_closure_start_and_end_sha256_required": True,
        "slots": 1,
        "effective_context_tokens": 256,
        "batch_tokens": 13,
        "ubatch_tokens": 13,
        "threads": 4,
        "batch_threads": 4,
    }
    if any(
        backend.get(field) != expected for field, expected in backend_expected.items()
    ):
        _fail("runtime custody gateway backend identity drifted")

    required_executable_fields = {
        "absolute_path",
        "lstat_device_inode_mode_size_ctime",
        "sha256",
        "code_signature_identity_or_explicit_unsigned_state",
        "loaded_image_paths_and_sha256",
        "start_end_identity_equality",
    }
    executable_fields = runtime.get("required_for_every_executable_or_library")
    if not isinstance(executable_fields, list) or not required_executable_fields.issubset(
        set(executable_fields)
    ):
        _fail("runtime custody executable closure fields are incomplete")
    required_model_fields = {
        "O_RDONLY",
        "O_NOFOLLOW",
        "file_descriptor_identity_before_load",
        "file_descriptor_identity_after_load",
        "file_descriptor_identity_before_receipt",
        "size_bytes",
        "sha256",
    }
    model_fields = runtime.get("required_for_every_model_open")
    if not isinstance(model_fields, list) or not required_model_fields.issubset(
        set(model_fields)
    ):
        _fail("runtime custody model file fields are incomplete")


def _validate_gateway_edge(
    graph_value: object,
    workloads_value: object,
    runtime_value: object,
    timing_value: object,
    statistics_value: object,
) -> None:
    graph = _object(graph_value, "gateway comparison graph")
    edge = _object(
        _object(graph.get("edges"), "gateway comparison edges").get(
            "GATEWAY_B_VS_G"
        ),
        "gateway edge",
    )
    edge_expected = {
        "members": ["B", "G"],
        "estimand": "incremental client-observed full-response OmniInfer gateway-path latency over the exact same resident llama-server backend",
        "workload_id": "GATEWAY_RAW13_FREE128_V3",
        "same_backend_process_required": True,
        "same_loaded_model_file_description_required": True,
        "engine_ranking_edge": False,
    }
    if any(edge.get(field) != expected for field, expected in edge_expected.items()):
        _fail("gateway edge must compare paths to the same resident backend only")

    workloads = _object(workloads_value, "gateway workload contracts")
    workload = _object(
        workloads.get("GATEWAY_RAW13_FREE128_V3"), "gateway workload"
    )
    request = _object(workload.get("request"), "gateway request")
    request_expected = {
        "method": "POST",
        "endpoint": "/v1/chat/completions",
        "canonicalization": "UTF-8 JSON with sorted keys, ensure_ascii=false, and compact separators",
        "size_bytes": 383,
        "sha256": "7773f5337693843f1e8cf3017b98868517cbddd3bc32649e550d8f2fec1d5cf6",
        "same_client_body_for_B_and_G": True,
        "temperature": 0,
        "seed": 0,
        "max_tokens": 128,
        "stream": False,
        "ignore_eos": True,
        "cache_prompt": False,
        "enable_thinking": False,
        "reasoning_format": "none",
        "return_tokens": True,
    }
    if any(request.get(field) != expected for field, expected in request_expected.items()):
        _fail("gateway request is not the exact same canonical request for B and G")

    runtime = _object(runtime_value, "gateway runtime custody")
    cohort = _object(runtime.get("gateway_cohort"), "gateway cohort")
    if (
        cohort.get("request_history_disabled") is not True
        or cohort.get("request_defaults_exact") != {}
        or cohort.get("effective_parameters_exact") != {}
        or cohort.get("proxy_model_exact") is not None
        or cohort.get("gateway_and_backend_mutable_logs_excluded_but_start_end_hashed")
        is not True
    ):
        _fail("gateway cohort mutable state is not fail closed")
    readiness = _object(cohort.get("current_readiness"), "gateway readiness")
    readiness_expected = {
        "formally_admitted": False,
        "same_resident_backend_design_proven_by_v1_diagnostic": True,
        "v1_diagnostic_samples_reusable": False,
        "future_resolution_may_be_proven_in_campaign_start_without_editing_this_contract": True,
    }
    if any(readiness.get(field) != expected for field, expected in readiness_expected.items()):
        _fail("gateway readiness must remain fail closed")
    blockers = readiness.get("blocker_codes")
    required_blockers = {
        "V3_GATEWAY_SUBCAMPAIGN_NOT_PUBLICLY_ACTIVATED",
        "V3_FORMAL_DRIVER_HASH_NOT_CAPTURED",
        "QUIET_HOST_GATE_NOT_YET_PASSED",
    }
    if not isinstance(blockers, list) or not required_blockers.issubset(set(blockers)):
        _fail("gateway readiness blockers are incomplete")

    timing = _object(timing_value, "gateway timing contract")
    gateway_timing = _object(timing.get("GATEWAY_B_VS_G"), "gateway timing boundary")
    timing_expected = {
        "start": "immediately-before-writing-the-first-byte-of-the-identical-canonical-request-on-a-warmed-persistent-client-connection",
        "end": "after-reading-the-full-response-body-and-validating-and-parsing-the-complete-JSON-response",
        "primary_metric": "client_full_response_wall_ms",
        "paired_delta_ms": "G_client_full_response_wall_ms - B_client_full_response_wall_ms",
        "paired_ratio": "G_client_full_response_wall_ms / B_client_full_response_wall_ms",
        "backend_native_prompt_and_predicted_ms_are_sensitivity_metrics_only": True,
        "gateway_overhead_subtraction_from_ApxInf_or_llama_core_allowed": False,
        "pure_gateway_CPU_time_claim_allowed": False,
    }
    if any(
        gateway_timing.get(field) != expected
        for field, expected in timing_expected.items()
    ):
        _fail("gateway client full-response timing boundary drifted")

    statistics = _object(statistics_value, "gateway statistics")
    gateway_statistics = _object(
        statistics.get("GATEWAY_B_VS_G"), "gateway statistics"
    )
    if (
        gateway_statistics.get("primary_observation")
        != "paired G_minus_B client_full_response_wall_ms"
        or gateway_statistics.get("engine_winner_or_ranking_claim_allowed") is not False
    ):
        _fail("gateway statistics must not make an engine ranking claim")


def _validate_claim_and_receipt(
    graph_value: object, receipt_value: object, claim_value: object
) -> None:
    graph = _object(graph_value, "claim comparison graph")
    required_forbidden_edges = {
        "NATIVE_EDGE_RELABELED_AS_ENGINE_ONLY",
        "NATIVE_EDGE_RELABELED_AS_MATCHED_QUANTIZATION",
        "NATIVE_EDGE_JOINED_TO_CORE_EDGE",
        "NATIVE_EDGE_JOINED_TO_GATEWAY_EDGE",
        "A_VS_G_END_TO_END",
        "A_VS_B_END_TO_END",
        "L_VS_G_END_TO_END",
        "CORE_EDGE_JOINED_TO_GATEWAY_EDGE",
    }
    forbidden_edges = graph.get("forbidden_edges")
    if (
        not isinstance(forbidden_edges, list)
        or not required_forbidden_edges.issubset(set(forbidden_edges))
        or graph.get("cross_edge_result_join_allowed") is not False
    ):
        _fail("claim comparison graph permits a forbidden edge or cross-edge join")

    receipt = _object(receipt_value, "machine receipt contract")
    if (
        _canonical_sha256(receipt.get("required_dynamic_contract_binding"))
        != _DYNAMIC_RECEIPT_SHA256
    ):
        _fail("machine dynamic receipt contract drifted")
    if (
        _canonical_sha256(receipt.get("subcampaign_marker_bindings"))
        != _SUBCAMPAIGN_MARKER_BINDINGS_SHA256
    ):
        _fail("machine receipt subcampaign marker bindings drifted")
    required_top_level = {
        "contract_binding",
        "git_custody",
        "host_custody",
        "artifact_custody",
        "parity_admission",
        "schedule_receipt",
        "samples",
        "statistics",
        "gates",
        "decision",
    }
    top_level = receipt.get("required_top_level_objects")
    if (
        receipt.get("pointer_syntax") != "RFC6901"
        or not isinstance(top_level, list)
        or not required_top_level.issubset(set(top_level))
    ):
        _fail("machine receipt shape is incomplete")
    exact_bindings = receipt.get("required_exact_bindings")
    expected_bindings = {
        "/contract_binding/campaign_id": "qwen35-0.8b-cross-runtime-formal-v3-20260826",
        "/contract_binding/schema_version": 3,
        "/host_custody/model_identifier": "Mac16,10",
        "/host_custody/chip": "Apple M4",
        "/host_custody/os_build": "25C56",
        "/artifact_custody/model/sha256": "427555e797eefeb62e1c8ef71510ce062027b0c7f1674cc4cfcb352710e0908c",
        "/artifact_custody/llama_cpp/source_commit": "f280b26983ad0fdb705a0d9ebf0503e76f2899b0",
        "/artifact_custody/omniinfer/source_commit": "79af77228f329a79ac665014089e23983e69e79f",
        "/artifact_custody/gateway_backend/source_commit": "61881b1f7f0b13d9e46d561fc25afcd6bbaec479",
    }
    if not isinstance(exact_bindings, dict) or any(
        exact_bindings.get(pointer) != expected
        for pointer, expected in expected_bindings.items()
    ):
        _fail("machine receipt exact custody bindings drifted")

    required_gates = {
        "NATIVE_A_VS_L": {
            "PREDECLARATION_PUBLIC_BEFORE_SAMPLING",
            "GIT_CUSTODY",
            "HOST_IDENTITY",
            "QUIET_HOST_CONTINUOUS",
            "POWER_THERMAL_MEMORY",
            "SAME_MODEL_REVISION_LINEAGE",
            "NATIVE_DEPLOYMENT_IDENTITIES_AND_HASHES",
            "NATIVE_DIFFERENCE_DISCLOSURES_COMPLETE",
            "RAW_PROMPT_IDS_EQUAL",
            "TEACHER_FORCED_128_EXACT",
            "FREE128_TRAJECTORY_EQUAL",
            "NEXT_GREEDY_TOKEN_READY_BOUNDARY_EQUAL",
            "FIXED_ABBA_BAAB_SCHEDULE_COMPLETE",
            "NO_RETRY_REPLACEMENT_OUTLIER_REMOVAL",
            "NATIVE_STABILITY",
        },
        "CORE_A_VS_L": {
            "PREDECLARATION_PUBLIC_BEFORE_SAMPLING",
            "GIT_CUSTODY",
            "HOST_IDENTITY",
            "QUIET_HOST_CONTINUOUS",
            "POWER_THERMAL_MEMORY",
            "MODEL_ARTIFACT_CUSTODY",
            "LOGICAL_WEIGHT_MANIFEST_EQUAL",
            "Q8_0_PAYLOAD_AND_SCALE_BITS_EQUAL",
            "Q8_0_PREFILL_STATE_POLICY_EQUAL",
            "F16_KV_POLICY_EQUAL",
            "Q8_0_HEAD_ARGMAX_POLICY_EQUAL",
            "CONTEXT_BATCH_THREAD_POLICY_EQUAL",
            "SAME_PHYSICAL_GPU",
            "PLACEMENT_RECEIPTS_COMPLETE",
            "RAW_PROMPT_IDS_EQUAL",
            "FREE128_TRAJECTORY_EQUAL",
            "TIMING_BOUNDARY_EQUAL",
            "FIXED_ABBA_BAAB_SCHEDULE_COMPLETE",
            "NO_RETRY_REPLACEMENT_OUTLIER_REMOVAL",
            "CORE_STABILITY",
        },
        "GATEWAY_B_VS_G": {
            "PREDECLARATION_PUBLIC_BEFORE_SAMPLING",
            "GIT_CUSTODY",
            "HOST_IDENTITY",
            "QUIET_HOST_CONTINUOUS",
            "POWER_THERMAL_MEMORY",
            "MODEL_ARTIFACT_CUSTODY",
            "SAME_RESIDENT_BACKEND_PROCESS",
            "SAME_CANONICAL_REQUEST",
            "SAME_RENDERED_RAW_PROMPT_IDS",
            "EMPTY_CACHE_BEFORE_EVERY_ARM",
            "FREE128_TRAJECTORY_EQUAL",
            "CLIENT_WALL_BOUNDARY_EQUAL",
            "FIXED_ABBA_BAAB_SCHEDULE_COMPLETE",
            "NO_RETRY_REPLACEMENT_OUTLIER_REMOVAL",
            "GATEWAY_STABILITY",
        },
    }
    for edge, expected_gates in required_gates.items():
        gate_ids = receipt.get(f"required_true_gate_ids_for_{edge}")
        if not isinstance(gate_ids, list) or not expected_gates.issubset(set(gate_ids)):
            _fail(f"machine receipt gates for {edge} are incomplete")
    for field in (
        "all_listed_gates_must_be_boolean_true",
        "missing_gate_is_failure",
        "null_required_hash_is_failure",
        "unknown_or_not_applicable_gate_is_failure",
    ):
        if receipt.get(field) is not True:
            _fail(f"machine receipt {field} must fail closed")

    claim = _object(claim_value, "claim policy")
    required_forbidden_claims = {
        "relabeling the NATIVE_A_VS_L result as ApxInf-engine versus llama.cpp-engine performance",
        "omitting the native edge's quantization prefill KV head placement or thread-policy differences",
        "claiming native edge weight quantization KV placement or numerical-regime identity",
        "ApxInf versus OmniInfer end-to-end speed ratio delta winner or ranking",
        "OmniInfer inference engine versus llama.cpp inference engine winner or ranking",
        "joining CORE_A_VS_L and GATEWAY_B_VS_G into one derived ranking",
        "ranking when any formal or stability gate fails",
        "relabeling v1 or v2 diagnostic observations as v3 formal samples",
    }
    forbidden_claims = claim.get("always_forbidden")
    if not isinstance(forbidden_claims, list) or not required_forbidden_claims.issubset(
        set(forbidden_claims)
    ):
        _fail("claim policy is missing a required forbidden claim")
    expected_labels = {
        "NATIVE_A_VS_L": "unmatched-numerical-regime named-native-deployment single-workload comparison",
        "CORE_A_VS_L": "matched-Q8_0-F16-KV single-workload core comparison",
        "GATEWAY_B_VS_G": "same-resident-backend client-observed OmniInfer gateway-path increment",
    }
    if claim.get("mandatory_result_labels") != expected_labels:
        _fail("claim policy mandatory result labels drifted")
    allowed = claim.get("allowed_only_after_corresponding_machine_gates_pass")
    expected_allowed = [
        "one NATIVE_A_VS_L decision about its two exact configuration_id values from its predeclared five-way decision table",
        "one CORE_A_VS_L decision from its predeclared five-way decision table",
        "one GATEWAY_B_VS_G decision from its predeclared four-way decision table",
        "raw samples, point estimates, intervals, stability statistics, and exact workload disclosures",
    ]
    if allowed != expected_allowed:
        _fail("claim policy gate-bound edge decision allowlist drifted")


def _validate_subcampaign_activation(failure_value: object) -> None:
    failure = _object(failure_value, "activation failure contract")
    if _canonical_sha256(failure) != _FAILURE_CONTRACT_SHA256:
        _fail("activation failure hard-stop contract drifted")
    if (
        failure.get("fail_closed") is not True
        or failure.get("one_subcampaign_failure_consumes_the_other_subcampaign")
        is not False
        or failure.get("one_subcampaign_result_may_fill_missing_other_edge") is not False
    ):
        _fail("activation subcampaign independence is not fail closed")
    markers = _object(
        failure.get("subcampaign_markers"), "activation subcampaign markers"
    )
    if set(markers) != {"NATIVE_A_VS_L", "CORE_A_VS_L", "GATEWAY_B_VS_G"}:
        _fail("activation subcampaign marker set is incomplete")
    edge_specific_fields = {
        "NATIVE_A_VS_L": (
            "created_only_after_all_native_configuration_teacher_free-trajectory_immutable_and_quiet_preconditions_pass",
            "must_be_committed_and_pushed_before_first_native_performance_generation_request",
            "creation_consumes_only_native_subcampaign_if_any_later_native_step_fails",
        ),
        "CORE_A_VS_L": (
            "created_only_after_all_core_immutable_parity_and_quiet_preconditions_pass",
            "must_be_committed_and_pushed_before_first_core_generation_request",
            "creation_consumes_only_core_subcampaign_if_any_later_core_step_fails",
        ),
        "GATEWAY_B_VS_G": (
            "created_only_after_all_gateway_immutable_same-backend_and_quiet_preconditions_pass",
            "must_be_committed_and_pushed_before_first_gateway_generation_request",
            "creation_consumes_only_gateway_subcampaign_if_any_later_gateway_step_fails",
        ),
    }
    for edge, fields in edge_specific_fields.items():
        marker = _object(markers.get(edge), f"activation {edge} marker")
        if (
            marker.get("create_new_only") is not True
            or marker.get("must_not_exist_before_creation") is not True
            or any(marker.get(field) is not True for field in fields)
        ):
            _fail(f"activation {edge} marker is not immutable and public before sampling")
    post_failure = _object(
        failure.get("first_post_marker_failure_action"),
        "activation post-marker failure action",
    )
    post_expected = {
        "scope": "active subcampaign only",
        "stop_immediately": True,
        "remaining_slots_marked_unattempted": True,
        "raw_partial_receipt_published": True,
        "failed_observations_retained": True,
        "campaign_consumed": True,
        "formal_summary_allowed": False,
        "replacement_campaign_under_same_id_allowed": False,
    }
    if any(post_failure.get(field) != value for field, value in post_expected.items()):
        _fail("activation post-marker failure action is not fail closed")


def validate_contract(contract: object) -> dict:
    contract = _object(contract, "contract")
    if set(contract) != _TOP_LEVEL_FIELDS:
        _fail("top-level fields are incomplete or unknown")
    if contract.get("format") != CONTRACT_FORMAT:
        _fail("format must identify the cross-runtime v3 predeclaration")
    if contract.get("schema_version") != 3:
        _fail("schema_version must be 3")
    if contract.get("document_role") != "PREDECLARATION_ONLY":
        _fail("document_role must remain PREDECLARATION_ONLY")
    if contract.get("result_status") != "NO_V3_PERFORMANCE_RESULT":
        _fail("predeclaration must not contain a v3 performance result")
    if contract.get("sampling_state_at_authoring") != {
        "v3_generation_requests": 0,
        "v3_warmup_samples": 0,
        "v3_timed_samples": 0,
        "performance_numbers_in_this_document": False,
    }:
        _fail("predeclaration sampling state must remain zero")
    _validate_campaign_binding(
        contract.get("campaign_id"),
        contract.get("scope"),
        contract.get("lineage"),
        contract.get("comparison_graph"),
        contract.get("failure_contract"),
    )
    _validate_exact_core_contract(
        contract.get("comparison_graph"),
        contract.get("workload_contracts"),
        contract.get("core_parity_contract"),
    )
    _validate_exact_gateway_contract(
        contract.get("comparison_graph"),
        contract.get("workload_contracts"),
        contract.get("runtime_custody"),
        contract.get("execution_protocol"),
        contract.get("timing_contract"),
    )
    _validate_activation_and_custody(
        contract.get("activation_contract"),
        contract.get("source_model_custody"),
        contract.get("runtime_custody"),
    )
    _validate_core_parity(contract.get("core_parity_contract"))
    _validate_core_timing(contract.get("timing_contract"))
    _validate_native_timing(contract.get("timing_contract"))
    _validate_execution_protocol(contract.get("execution_protocol"))
    _validate_quiet_host(contract.get("host_quiet_gate"))
    _validate_native_deployment(
        contract.get("comparison_graph"),
        contract.get("native_deployment_contract"),
        contract.get("statistics_and_decisions"),
    )
    _validate_statistics(contract.get("statistics_and_decisions"))
    _validate_workloads(contract.get("workload_contracts"))
    _validate_gateway_edge(
        contract.get("comparison_graph"),
        contract.get("workload_contracts"),
        contract.get("runtime_custody"),
        contract.get("timing_contract"),
        contract.get("statistics_and_decisions"),
    )
    _validate_claim_and_receipt(
        contract.get("comparison_graph"),
        contract.get("machine_receipt_contract"),
        contract.get("claim_policy"),
    )
    _validate_subcampaign_activation(contract.get("failure_contract"))
    if _canonical_sha256(contract) != PINNED_CANONICAL_CONTRACT_SHA256:
        _fail("canonical contract semantic pin drifted")
    return contract


def _load_contract_snapshot(path: Path | str) -> tuple[dict, bytes]:
    try:
        raw = Path(path).read_bytes()
    except OSError as error:
        _fail(f"cannot read formal contract: {error}")
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        _fail(f"formal contract is not strict UTF-8: {error}")
    try:
        contract = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite_constant,
        )
    except json.JSONDecodeError as error:
        _fail(f"formal contract is not valid JSON: {error}")
    return validate_contract(contract), raw


def load_contract(path: Path | str) -> dict:
    contract, _ = _load_contract_snapshot(path)
    return contract


def validation_receipt(contract: dict, contract_bytes: bytes) -> dict:
    if not isinstance(contract_bytes, bytes):
        _fail("validation receipt requires the parsed contract byte snapshot")
    raw = contract_bytes
    core_readiness = contract["core_parity_contract"]["current_readiness"]
    native_readiness = contract["native_deployment_contract"]["current_readiness"]
    gateway_readiness = contract["runtime_custody"]["gateway_cohort"][
        "current_readiness"
    ]
    return {
        "format": VALIDATION_FORMAT,
        "valid": True,
        "campaign_id": contract["campaign_id"],
        "contract_file_size_bytes": len(raw),
        "contract_file_sha256": hashlib.sha256(raw).hexdigest(),
        "edges": {
            "CORE_A_VS_L": {
                "ready": core_readiness["formal_campaign_may_start_now"],
                "claim_class": "matched-exact-GGUF-Q8_0-F16-KV-core",
                "blocker_codes": core_readiness["blocker_codes"],
            },
            "NATIVE_A_VS_L": {
                "ready": native_readiness["formal_campaign_may_start_now"],
                "claim_class": "named-deployment-only-with-disclosed-regime-differences",
                "blocker_codes": native_readiness["blocker_codes"],
            },
            "GATEWAY_B_VS_G": {
                "ready": gateway_readiness["formally_admitted"],
                "claim_class": "same-resident-backend-gateway-path-increment-not-engine-ranking",
                "blocker_codes": gateway_readiness["blocker_codes"],
            },
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        contract, raw = _load_contract_snapshot(args.contract)
        receipt = validation_receipt(contract, raw)
        try:
            encoded_receipt = json.dumps(
                receipt,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        except ValueError as error:
            _fail(f"non-finite validation receipt is forbidden: {error}")
        except (TypeError, OverflowError) as error:
            _fail(f"validation receipt is not strict JSON: {error}")
    except FormalContractError as error:
        print(f"formal contract rejected: {error}", file=sys.stderr)
        return 1
    print(encoded_receipt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
