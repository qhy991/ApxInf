#!/usr/bin/env python3
"""Fail-closed static validator for the ApxInf/OmniInfer HTTP formal-v1 contract.

This program validates only the checked-in predeclaration.  It deliberately
does not implement the formal driver, inspect a future run receipt, start a
runtime, make a generation request, or turn prior diagnostics into evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import stat
import sys
from typing import Any


CONTRACT_FORMAT = "apxinf-qwen35-apxinf-vs-omniinfer-http-formal-predeclaration-v1"
VALIDATION_FORMAT = (
    "apxinf-qwen35-apxinf-vs-omniinfer-http-formal-contract-validation-v1"
)
CAMPAIGN_ID = "qwen35-0.8b-apxinf-vs-omniinfer-http-formal-v1-20260826"
EDGE_ID = "DEPLOYMENT_AH_VS_OG"
PINNED_FILE_SHA256 = "6614977266b9b455b2592c41ade2eb2879bce45738ab198b94e4920b46e184f6"
PINNED_CANONICAL_SEMANTIC_SHA256 = (
    "8e6f908677191b56d1800eff4e2dff1eabf0002cc9daf3042bcde9b104bfcc6e"
)
MAX_CONTRACT_BYTES = 1024 * 1024
DEFAULT_CONTRACT = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "qwen35-0.8b-apxinf-vs-omniinfer-http-formal-v1.json"
)

PROMPT_TOKEN_IDS = [
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
SUPPRESSED_EOG_TOKEN_IDS = [248044, 248046, 248063, 248064, 248065]
RENDERED_PROMPT = (
    "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
)
REQUEST = {
    "cache_prompt": False,
    "chat_template_kwargs": {"enable_thinking": False},
    "id_slot": 0,
    "ignore_eos": True,
    "max_tokens": 128,
    "messages": [{"content": "Hello", "role": "user"}],
    "model": (
        "/Users/haiyan-mini/Agent4Kernel/models/"
        "Qwen3.5-0.8B-2fc063647-GGUF/"
        "Qwen3.5-0.8B-2fc063647-Q8_0-llama.gguf"
    ),
    "reasoning_format": "none",
    "return_tokens": True,
    "seed": 0,
    "stream": False,
    "temperature": 0,
    "verbose": True,
}
TOKENIZE_REQUEST = {
    "add_special": False,
    "content": RENDERED_PROMPT,
    "parse_special": True,
    "with_pieces": False,
}
TOP_LEVEL_FIELDS = {
    "format",
    "schema_version",
    "campaign_id",
    "deployment_edge_id",
    "authored_at_utc",
    "document_role",
    "result_status",
    "sampling_state_at_authoring",
    "activation_contract",
    "lineage",
    "scope",
    "named_deployment_boundary",
    "workload_contract",
    "schedule_contract",
    "timing_contract",
    "quiet_host_receipt_contract",
    "runtime_custody_contract",
    "execution_protocol",
    "statistics_and_decision_contract",
    "failure_contract",
    "machine_receipt_contract",
    "current_readiness",
    "claim_policy",
    "validator_scope",
}
CUSTODY_CHECKPOINTS = [
    "prepare-before-runtime-launch",
    "runtime-preflight-start",
    "runtime-preflight-end",
    "before-first-warmup",
    "after-warmups",
    *[f"after-measured-block-{index}" for index in range(1, 17)],
    "postflight-before-shutdown",
    "raw-receipt-durable-before-cleanup",
]
REQUIRED_GATE_IDS = [
    "CONTRACT_PUBLIC_BEFORE_PREPARE",
    "ONE_SHOT_MARKER_CLAIMED",
    "HOST_IDENTITY",
    "QUIET_HOST_SCHEMA_VALID",
    "QUIET_HOST_CONTINUOUS",
    "LIVE_SOURCE_CUSTODY",
    "LIVE_BINARY_CUSTODY",
    "LIVE_LIBRARY_CUSTODY",
    "LIVE_MODEL_FD_CUSTODY",
    "NAMED_DEPLOYMENTS_EXACT",
    "CANONICAL_383_BYTE_REQUEST",
    "EXACT_RAW13_OMNI_TOKENIZE",
    "FIVE_EOG_NEGATIVE_INFINITY",
    "SLOT0_COLD_BEFORE_EVERY_REQUEST",
    "APX_COMPACT_GENERATION_PATH_EXTERNAL_VALIDATION",
    "PER_ARM_DETERMINISTIC_FREE128",
    "FULL_HTTP_WALL_BOUNDARY",
    "FIXED_16_BLOCK_SCHEDULE",
    "DURABLE_136_SLOT_RECEIPTS",
    "NO_RETRY_REPLACEMENT_OR_OUTLIER_REMOVAL",
    "STABILITY_AND_SAME_HALVES",
]
BLOCKER_CODES = [
    "FORMAL_V1_DRIVER_NOT_IMPLEMENTED",
    "FORMAL_APXINF_RUNTIME_CUSTODY_ADAPTER_NOT_IMPLEMENTED",
    "FORMAL_PREPARE_RUN_MARKER_NOT_IMPLEMENTED",
    "FORMAL_CONTINUOUS_HOST_MONITOR_NOT_IMPLEMENTED",
    "FORMAL_LIVE_FD_AND_LIBRARY_CUSTODY_NOT_IMPLEMENTED",
    "QUIET_HOST_GATE_NOT_PASSED",
]


class FormalContractError(ValueError):
    """The static formal-v1 contract is malformed or weakened."""


def _fail(message: str) -> None:
    raise FormalContractError(message)


def _require(condition: bool, message: str) -> None:
    if not condition:
        _fail(message)


def _object(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        _fail(f"{label} must be an object")
    return value


def _exact_keys(value: dict[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        _fail(f"{label} field set drifted")


def _canonical_json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise FormalContractError(
            f"value is not canonical strict JSON: {error}"
        ) from error


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            _fail(f"JSON contains duplicate key: {key}")
        result[key] = value
    return result


def _reject_nonfinite_constant(value: str) -> object:
    _fail(f"non-finite JSON constant is forbidden: {value}")


def parse_strict_json(raw: bytes) -> dict[str, Any]:
    if raw.startswith(b"\xef\xbb\xbf"):
        _fail("contract must not contain a UTF-8 BOM")
    try:
        text = raw.decode("utf-8", "strict")
    except UnicodeDecodeError as error:
        raise FormalContractError("contract is not strict UTF-8") from error
    decoder = json.JSONDecoder(
        object_pairs_hook=_reject_duplicate_keys,
        parse_constant=_reject_nonfinite_constant,
    )
    start = len(text) - len(text.lstrip())
    try:
        value, end = decoder.raw_decode(text, start)
    except (json.JSONDecodeError, FormalContractError) as error:
        if isinstance(error, FormalContractError):
            raise
        raise FormalContractError(f"contract is not strict JSON: {error}") from error
    if text[end:].strip():
        _fail("contract has trailing non-whitespace data")
    return _object(value, "contract")


def _validate_identity_and_lineage(contract: dict[str, Any]) -> None:
    _exact_keys(contract, TOP_LEVEL_FIELDS, "top-level contract")
    _require(contract.get("format") == CONTRACT_FORMAT, "contract format drifted")
    _require(contract.get("schema_version") == 1, "schema version drifted")
    _require(contract.get("campaign_id") == CAMPAIGN_ID, "campaign ID drifted")
    _require(
        contract.get("deployment_edge_id") == EDGE_ID, "deployment edge ID drifted"
    )
    _require(
        contract.get("document_role")
        == "FROZEN_PREDECLARATION_AND_STATIC_CONTRACT_ONLY",
        "document role drifted",
    )
    _require(
        contract.get("result_status") == "NO_FORMAL_DRIVER_NO_FORMAL_SAMPLES_NO_RESULT",
        "result status must disclose that no formal result exists",
    )
    sampling = _object(contract.get("sampling_state_at_authoring"), "sampling state")
    _require(
        sampling
        == {
            "formal_generation_requests": 0,
            "formal_warmup_samples": 0,
            "formal_timed_samples": 0,
            "performance_numbers_in_this_document": False,
        },
        "sampling state must remain zero-result",
    )

    activation = _object(contract.get("activation_contract"), "activation contract")
    activation_expected = {
        "contract_and_validator_must_be_committed_and_pushed_before_prepare": True,
        "prepare_must_run_live_git_ls_remote": True,
        "prepare_must_prove_head_equals_live_remote_ref": True,
        "prepare_must_prove_clean_worktree": True,
        "local_tracking_ref_alone_is_publication_proof": False,
        "editing_after_prepare_marker_creation_allowed": False,
        "this_document_alone_activates_campaign": False,
    }
    _require(
        all(
            activation.get(key) == expected
            for key, expected in activation_expected.items()
        ),
        "activation/publication contract was weakened",
    )
    _require(
        activation.get("frozen_live_remote_ref")
        == "refs/heads/perf/qwen35-http-server-v5",
        "activation live remote ref drifted",
    )

    lineage = _object(contract.get("lineage"), "diagnostic lineage")
    lineage_expected = {
        "supersedes_existing_cross_runtime_v3": False,
        "independent_from_cross_runtime_v3": True,
        "cross_runtime_v3_forbidden_edge_being_replaced": "A_VS_G_END_TO_END",
        "existing_nonformal_driver_may_be_used_as_formal_driver": False,
        "existing_nonformal_samples_reused": False,
        "existing_nonformal_samples_relabelled": False,
        "existing_nonformal_statistics_reused": False,
        "formal_driver_status": "NOT_IMPLEMENTED_BY_THIS_CONTRACT_SLICE",
    }
    _require(
        all(lineage.get(key) == expected for key, expected in lineage_expected.items()),
        "diagnostic lineage would reuse or relabel nonformal evidence",
    )
    _require(
        lineage.get("formal_driver_path_reserved")
        == "benchmarks/cross_runtime/apxinf_vs_omniinfer_http_formal_v1_driver.py",
        "formal driver reservation drifted",
    )


def _validate_named_deployment_boundary(contract: dict[str, Any]) -> None:
    boundary = _object(
        contract.get("named_deployment_boundary"), "named deployment boundary"
    )
    _require(boundary.get("edge_id") == EDGE_ID, "deployment edge binding drifted")
    arms = _object(boundary.get("arms"), "named deployment arms")
    _require(set(arms) == {"AH", "OG"}, "named deployment arm set drifted")
    ah = _object(arms.get("AH"), "AH deployment")
    og = _object(arms.get("OG"), "OG deployment")
    _require(
        ah.get("inference_implementation_base_commit")
        == "80049e7f15df67356b3932370b7ab3cc06e938f8"
        and ah.get("formal_runtime_source_commit_binding")
        == "exact prepare HEAD equal to live remote ref and descendant of inference implementation base commit"
        and ah.get("serialized_weight_size_bytes") == 1746942600
        and ah.get("serialized_weight_sha256")
        == "04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696"
        and ah.get("existing_adapter_is_formal_runtime") is False,
        "AH named deployment identity or nonformal-adapter boundary drifted",
    )
    _require(
        og.get("omniinfer_source_commit") == "79af77228f329a79ac665014089e23983e69e79f"
        and og.get("omniinfer_cli_sha256")
        == "65487424ca9179850b80079beafa5ad69a66e0841d328ee8dd8a1fd4b613d661"
        and og.get("backend_source_commit")
        == "61881b1f7f0b13d9e46d561fc25afcd6bbaec479"
        and og.get("backend_binary_sha256")
        == "02723fc39fbeebd9849ce4c9ca3799649df3cf91f101c2cd56b8756e1db54d28"
        and og.get("serialized_weight_size_bytes") == 811843072
        and og.get("serialized_weight_sha256")
        == "427555e797eefeb62e1c8ef71510ce062027b0c7f1674cc4cfcb352710e0908c",
        "OG named deployment identity drifted",
    )
    expected_boundary = {
        "same_upstream_model_revision": True,
        "same_serialized_weight_bytes": False,
        "same_quantization_layout": False,
        "same_runtime_or_kernel_graph": False,
        "same_kv_numeric_regime": False,
        "cross_arm_generated_trajectory_equality_required": False,
        "per_arm_deterministic_trajectory_required": True,
        "all_68_warmup_and_measured_trajectories_within_each_arm_must_be_bitwise_equal": True,
        "raw_128_generated_token_ids_required_for_every_sample": True,
        "different_trajectories_do_not_invalidate_named_deployment_estimand": True,
        "engine_only_ranking_allowed": False,
        "identical_artifact_or_numerical_parity_claim_allowed": False,
    }
    _require(
        all(
            boundary.get(key) == expected for key, expected in expected_boundary.items()
        ),
        "named deployment boundary misstates serialized weights or trajectories",
    )
    _require(
        ah.get("formal_runtime_requirements")
        == {
            "bind_endpoint": "http://127.0.0.1:9100",
            "cargo_build_argv_without_toolchain_path": [
                "cargo",
                "build",
                "--locked",
                "--release",
                "--no-default-features",
                "-p",
                "apxinf-model",
                "--features",
                "accelerate,metal-w8",
                "--example",
                "qwen35_benchmark_http_formal_v1_server",
            ],
            "cargo_profile": "release",
            "rustflags_environment_required": "unset",
            "embedded_candidate_commit_must_equal_formal_runtime_HEAD": True,
            "model_directory": "/Users/haiyan-mini/Agent4Kernel/.apxinf-models/Qwen3.5-0.8B-2fc063647",
            "source_lock": "/Users/haiyan-mini/Agent4Kernel/ApxInf/.apxinf/onboarding/qwen35-0.8b/source-lock.json",
            "device": "Metal with declared CPU F32 remainder",
            "effective_context_tokens": 256,
            "expected_generation_requests": 68,
            "resident_model_and_tokenizer_before_preflight": True,
            "checked_reset_before_every_generation": True,
        },
        "AH formal launch configuration drifted",
    )
    _require(
        og.get("formal_runtime_requirements")
        == {
            "gateway_endpoint": "http://127.0.0.1:9000",
            "isolated_absent_state_and_slot_roots_before_launch": True,
            "request_history_enabled": False,
            "model_path": REQUEST["model"],
            "mmproj": None,
            "effective_context_tokens": 256,
            "gpu_layers": 999,
            "batch_tokens": 13,
            "ubatch_tokens": 13,
            "threads": 4,
            "batch_threads": 4,
            "parallel_slots": 1,
            "slots_endpoint_enabled": True,
            "cache_ram_bytes": 0,
            "prompt_cache_enabled": False,
            "cache_idle_slots_enabled": False,
            "slot_prompt_similarity": 0,
            "flash_attention_flag_present": False,
            "cache_type_k_override_flag_present": False,
            "cache_type_v_override_flag_present": False,
            "effective_K_and_V_cache_type": "F16",
            "model_offload_policy": "999 requested GPU layers on pinned Metal backend",
            "flash_attention_policy": "no explicit flag; pinned b10280 default",
            "extra_backend_performance_flags_allowed": False,
            "gateway_and_backend_resident_before_preflight": True,
        },
        "OG formal launch configuration drifted",
    )


def _validate_workload(contract: dict[str, Any]) -> None:
    workload = _object(contract.get("workload_contract"), "workload contract")
    request = _object(workload.get("request"), "request contract")
    _require(
        request.get("canonical_json_object") == REQUEST,
        "383-byte request object drifted",
    )
    request_bytes = _canonical_json_bytes(request["canonical_json_object"])
    _require(
        len(request_bytes) == request.get("size_bytes") == 383
        and _sha256(request_bytes)
        == request.get("sha256")
        == "7773f5337693843f1e8cf3017b98868517cbddd3bc32649e550d8f2fec1d5cf6",
        "383-byte request size or SHA drifted",
    )
    _require(
        request.get("same_body_bytes_for_AH_and_OG") is True
        and request.get("complete_wire_is_pre_serialized_per_arm") is True
        and request.get(
            "only_authority_header_and_destination_differ_between_arm_wires"
        )
        is True,
        "request wire parity was weakened",
    )

    prompt = _object(workload.get("prompt"), "prompt contract")
    _require(
        prompt.get("rendered_utf8") == RENDERED_PROMPT
        and len(RENDERED_PROMPT.encode("utf-8"))
        == prompt.get("rendered_size_bytes")
        == 74
        and _sha256(RENDERED_PROMPT.encode("utf-8"))
        == prompt.get("rendered_sha256")
        == "13071b0f0e23e97681f6f39247dbb715973302580660a81dbae994fe8064e7d3"
        and prompt.get("token_count") == 13
        and prompt.get("token_ids") == PROMPT_TOKEN_IDS
        and _sha256(_canonical_json_bytes(PROMPT_TOKEN_IDS))
        == prompt.get("token_ids_canonical_sha256")
        == "4b890fa15ee3d7db4e9dd18bd79c6362d40e9e016ae4f9f74cb7fc420ef3b6d3",
        "raw 13-token prompt contract drifted",
    )

    tokenize = _object(
        workload.get("omniinfer_external_tokenize_admission"),
        "OmniInfer external tokenization contract",
    )
    tokenize_bytes = _canonical_json_bytes(tokenize.get("tokenize_request"))
    _require(
        tokenize.get("expected_request_count") == 1
        and tokenize.get("owned_by_arm") == "OG"
        and tokenize.get(
            "required_once_in_runtime_preflight_before_any_warmup_or_cache_clear"
        )
        is True
        and tokenize.get("repeated_during_warmup_or_measured_slots_allowed") is False
        and tokenize.get("outside_primary_timed_interval") is True
        and tokenize.get("authority") == "OmniInfer generation gateway"
        and tokenize.get("tokenize_endpoint") == "/tokenize"
        and tokenize.get("tokenize_request") == TOKENIZE_REQUEST
        and len(tokenize_bytes) == tokenize.get("tokenize_request_size_bytes") == 156
        and _sha256(tokenize_bytes)
        == tokenize.get("tokenize_request_sha256")
        == "617df3df640c21bf6c3c6460f78589476d50f0ee149e1d5699ff41f99502677b"
        and tokenize.get("response_required_exact_keys") == ["tokens"]
        and tokenize.get(
            "response_tokens_must_be_array_of_exactly_13_nonnegative_integers"
        )
        is True
        and tokenize.get("exact_token_ids_required") == PROMPT_TOKEN_IDS
        and tokenize.get("exact_token_ids_canonical_sha256")
        == "4b890fa15ee3d7db4e9dd18bd79c6362d40e9e016ae4f9f74cb7fc420ef3b6d3"
        and tokenize.get("generation_request_count_increment_required") == 0
        and tokenize.get("receipt_must_be_durable_before_first_warmup_cache_clear")
        is True
        and tokenize.get(
            "every_slot_receipt_must_reference_the_same_preflight_tokenize_receipt_and_current_custody_checkpoint"
        )
        is True
        and tokenize.get(
            "every_OG_generation_verbose_prompt_must_equal_tokenize_content_bytes"
        )
        is True
        and tokenize.get("every_OG_generation_verbose_prompt_required_size_bytes") == 74
        and tokenize.get("every_OG_generation_verbose_prompt_required_sha256")
        == "13071b0f0e23e97681f6f39247dbb715973302580660a81dbae994fe8064e7d3"
        and tokenize.get("apply_template_call_required_or_allowed_as_substitute")
        is False
        and tokenize.get("prompt_token_count_report_alone_is_sufficient") is False
        and tokenize.get("server_internal_token_ids_claim_alone_is_sufficient")
        is False,
        "OmniInfer /tokenize exact-13 admission was weakened",
    )

    generation = _object(workload.get("generation"), "generation contract")
    _require(
        generation
        == {
            "completion_tokens": 128,
            "temperature": 0,
            "seed": 0,
            "stream": False,
            "concurrency": 1,
            "speculative_decoding_allowed": False,
            "finish_reason": "length",
            "usage_prompt_completion_total": [13, 128, 141],
        },
        "128-token generation contract drifted",
    )
    eog = _object(workload.get("suppressed_eog_policy"), "five-EOG policy")
    _require(
        eog.get("token_ids") == SUPPRESSED_EOG_TOKEN_IDS
        and len(set(eog["token_ids"])) == 5
        and _sha256(_canonical_json_bytes(eog["token_ids"]))
        == eog.get("token_ids_canonical_sha256")
        == "656e15a6ba9c76f492ba6bb34a0f2af4095ec3850dbb09b468228c2055ece9ca"
        and eog.get("same_five_token_policy_for_both_arms") is True
        and eog.get(
            "apxinf_external_compact_path_receipt_must_prove_masked_prefill_and_all_127_decode_calls"
        )
        is True
        and eog.get("canonical_request_ignore_eos_must_be_true") is True
        and eog.get("AH_generation_settings_policy_required") == "-inf-before-greedy"
        and eog.get("AH_generation_settings_suppressed_eog_token_ids_required")
        == SUPPRESSED_EOG_TOKEN_IDS
        and eog.get("every_OG_generation_verbose_logit_bias_required_count") == 5
        and eog.get("every_OG_generation_verbose_logit_bias_entry_exact_keys")
        == ["bias", "token"]
        and eog.get("every_OG_generation_verbose_logit_bias_token_set_required")
        == SUPPRESSED_EOG_TOKEN_IDS
        and eog.get("every_OG_generation_verbose_logit_bias_bias_required") is None
        and eog.get("null_bias_is_negative_infinity_serialization") is True
        and eog.get("generated_suppressed_eog_hits_required") == 0,
        "five-EOG negative-infinity policy drifted",
    )
    compact = _object(
        workload.get("apxinf_compact_generation_path_external_validation"),
        "Apx compact generation-path external validation contract",
    )
    _require(
        compact
        == {
            "full_unmodified_generation_path_receipt_required_in_AH_response": True,
            "server_provided_compact_summary_or_boolean_alone_is_sufficient": False,
            "raw_top_level_required_values": {
                "format": "apxinf-qwen35-mlp-stack3-boundary-tail-head-generation-path-v1",
                "mechanism": "metal-w8-mlp-stack3-boundary-tail-head-gdn-core-fused-v1",
                "gdn_core_profile": "gdn-core-fused-v1",
                "prefill_body_calls": 1,
                "terminal_error": False,
            },
            "prefill_head_required_values": {
                "mechanism": "cpu-f32-tied",
                "calls": 1,
                "tail_transactions": 0,
            },
            "initial_stack_required_values": {
                "mechanism": "metal-w8-linear-layer-stack3-gdn-core-fused-v1",
                "gdn_core_profile": "gdn-core-fused-v1",
                "decode_calls": 127,
                "successful_decodes": 127,
                "failed_decodes": 0,
                "terminal_error": False,
            },
            "boundaries_required_count": 5,
            "every_boundary_required_values": {
                "mechanism": "metal-w8-mlp-stack3-boundary-gdn-core-fused-v1",
                "gdn_core_profile": "gdn-core-fused-v1",
                "decode_calls": 127,
                "successful_decodes": 127,
                "failed_decodes": 0,
                "terminal_error": False,
            },
            "decode_head_required_values": {
                "mechanism": "metal-w8-tail-v1",
                "layer_index": 23,
                "calls": 127,
                "excluded_calls": 127,
                "teacher_calls": 0,
                "tail_transactions": 127,
                "successful_transactions": 127,
                "failed_transactions": 0,
                "terminal_error": False,
            },
            "externally_rederived_compact_exact_values": {
                "expected_decode_calls": 127,
                "decode_api_calls": 127,
                "excluded_decode_api_calls": 127,
                "teacher_calls": 0,
                "tail_transactions": 127,
                "successful_tail_transactions": 127,
                "failed_tail_transactions": 0,
                "initial_valid": True,
                "boundaries_valid": True,
                "optimized_excluding_decode_api_hit": True,
                "terminal_error": False,
            },
            "required_subtrees_must_reject_missing_wrong_type_or_nonexact_required_values": True,
            "additional_raw_telemetry_fields_allowed": True,
            "additional_raw_telemetry_cannot_substitute_for_required_fields": True,
        },
        "Apx compact generation-path exact schema or counters drifted",
    )
    cold = _object(workload.get("cold_slot_contract"), "cold slot contract")
    cold_expected = {
        "slot_id": 0,
        "AH_clear_authority": "http://127.0.0.1:9100",
        "AH_clear_method_path_and_body": ["POST", "/apxinf/cache/clear", "{}"],
        "AH_clear_required_response_fields": {
            "ok": True,
            "cache_policy": "checked-reset-exactly-once-before-each-generation",
            "cleared_slots": [0],
            "checked_reset_calls_this_epoch": 1,
        },
        "AH_clear_response_required_exact_keys": [
            "cache_policy",
            "checked_reset_calls_this_epoch",
            "cleared_slots",
            "epoch",
            "format",
            "ok",
            "qualification",
        ],
        "AH_clear_required_format": "apxinf-qwen35-benchmark-http-formal-v1-server",
        "AH_clear_required_qualification": "FORMAL_CANDIDATE_HTTP_ADAPTER_REQUIRES_EXTERNAL_CAMPAIGN_GATES",
        "AH_clear_epoch_must_equal_one_based_arm_sequence_index": True,
        "AH_generation_verbose_epoch_must_equal_immediately_preceding_clear_epoch": True,
        "AH_clear_must_use_same_persistent_connection_and_authority_as_generation": True,
        "OG_clear_authority": "http://127.0.0.1:9000",
        "OG_clear_method_path_and_body": ["POST", "/omni/cache/clear", "{}"],
        "OG_clear_required_response_fields": {
            "ok": True,
            "cache_policy": "cleared_each_run",
            "cleared_slots": [0],
        },
        "OG_clear_response_required_exact_keys": [
            "cache_policy",
            "cleared_slots",
            "message",
            "ok",
        ],
        "OG_clear_endpoint_must_be_same_gateway_authority_as_generation_and_tokenize": True,
        "OG_gateway_state_backend_PID_port_endpoint_and_model_binding_required_before_and_after_campaign": True,
        "direct_backend_slot_erase_is_allowed_substitute": False,
        "clear_immediately_before_every_arm_request": True,
        "clear_outside_primary_timed_interval": True,
        "clear_must_be_acknowledged": True,
        "clear_once_then_generate_once": True,
        "prompt_cache_allowed": False,
        "omniinfer_usage_cached_tokens_required": 0,
        "omniinfer_native_cache_n_required": 0,
        "omniinfer_tokens_cached_after_generation_required": 140,
        "retry_after_clear_or_generation_failure_allowed": False,
    }
    _require(cold == cold_expected, "per-request slot-0 cold-cache contract drifted")


def _validate_schedule_and_timing(contract: dict[str, Any]) -> None:
    schedule = _object(contract.get("schedule_contract"), "paired schedule contract")
    schedule_expected = {
        "warmup_pair_orders": ["AH_OG", "OG_AH", "OG_AH", "AH_OG"],
        "measured_blocks": 16,
        "measured_block_indices": list(range(1, 17)),
        "pairs_per_block": 4,
        "odd_block_pair_orders": ["AH_OG", "OG_AH", "OG_AH", "AH_OG"],
        "even_block_pair_orders": ["OG_AH", "AH_OG", "AH_OG", "OG_AH"],
        "warmup_pairs": 4,
        "measured_pairs": 64,
        "warmup_samples": 8,
        "measured_samples": 128,
        "requests_per_arm_including_warmup": 68,
        "total_generation_requests": 136,
        "one_resident_process_tree_per_arm": True,
        "one_warmed_persistent_generation_connection_per_arm": True,
        "no_retry": True,
        "no_replacement": True,
        "no_resample": True,
        "no_outlier_removal": True,
        "no_schedule_extension_or_reordering": True,
    }
    _require(schedule == schedule_expected, "fixed paired 16-block schedule drifted")

    timing = _object(contract.get("timing_contract"), "full HTTP timing contract")
    timing_expected = {
        "clock": "monotonic",
        "clock_identity_and_resolution_recorded": True,
        "primary_metric": "client_full_response_wall_ms",
        "start": "immediately-before-the-single-sendall-call-for-the-complete-pre-serialized-arm-specific-HTTP/1.1-request-wire-containing-the-identical-canonical-383-byte-JSON-body-on-a-warmed-persistent-client-connection",
        "end": "immediately-after-full-response-body-read-strict-JSON-parse-and-external-arm-specific-semantic-validation",
        "single_sendall_for_complete_request_wire_required": True,
        "full_response_body_read_before_end": True,
        "strict_duplicate-key-and-nonfinite-rejecting_JSON_parse_before_end": True,
        "semantic_response_validation_before_end": True,
        "apxinf_compact_generation_path_receipt_external_validation_before_end": True,
        "omniinfer_verbose_cache_usage_token_and_EOG_validation_before_end": True,
        "single_tokenize_preflight_occurs_before_all_warmup_and_measured_intervals": True,
        "tokenize_calls_between_slots_allowed": False,
        "cache_clear_before_start": True,
        "request_serialization_before_start": True,
        "model_load_process_start_and_shutdown_excluded": True,
        "body_only_or_runtime_native_timing_is_primary": False,
        "runtime_native_timing_fields_are_secondary_only": True,
    }
    _require(timing == timing_expected, "full HTTP wall timing boundary drifted")


def _validate_quiet_host(contract: dict[str, Any]) -> None:
    quiet = _object(
        contract.get("quiet_host_receipt_contract"), "quiet-host receipt contract"
    )
    _require(
        quiet.get("receipt_format") == "apxinf-qwen35-continuous-quiet-host-receipt-v1"
        and quiet.get("receipt_schema_version") == 1
        and quiet.get("strict_JSON_duplicate_keys_nonfinite_and_trailing_data_rejected")
        is True
        and quiet.get("maximum_receipt_size_bytes") == 16777216,
        "quiet-host receipt schema identity drifted",
    )
    _require(
        quiet.get("top_level_required_fields")
        == [
            "format",
            "schema_version",
            "host_identity",
            "monitor_identity",
            "allowlist",
            "thresholds",
            "preflight",
            "continuous",
            "postflight",
            "derived_gates",
            "passed",
        ]
        and quiet.get("top_level_unknown_fields_allowed") is False
        and quiet.get("host_identity_required_exact")
        == {
            "model_identifier": "Mac16,10",
            "chip": "Apple M4",
            "architecture": "arm64",
            "logical_cpu_count": 10,
            "memory_bytes": 17179869184,
            "os_build": "25C56",
        }
        and quiet.get("monitor_identity_required_fields")
        == [
            "pid",
            "process_start_time",
            "absolute_executable_path",
            "executable_sha256",
            "argv",
            "argv_sha256",
            "process_group_id",
        ]
        and quiet.get("allowlist_required_roles")
        == [
            "campaign_orchestrator",
            "custody_monitor",
            "active_AH_runtime_tree",
            "active_OG_gateway_and_backend_tree",
        ]
        and quiet.get("derived_gate_ids")
        == [
            "HOST_IDENTITY",
            "MONITOR_IDENTITY",
            "SNAPSHOT_SCHEMA",
            "TEMPORAL_COVERAGE",
            "POWER_SOURCE",
            "THERMAL_AND_PERFORMANCE",
            "LOAD_AVERAGE",
            "PROCESS_CPU",
            "SWAP",
            "MEMORY_PRESSURE",
            "ALL_SNAPSHOTS",
        ]
        and quiet.get("all_derived_gates_must_be_boolean_true") is True
        and quiet.get("passed_must_equal_logical_AND_of_all_derived_gates") is True
        and quiet.get("passed_field_is_sufficient_without_recomputation") is False
        and quiet.get(
            "validator_or_driver_must_recompute_every_gate_from_raw_snapshots"
        )
        is True
        and quiet.get("missing_snapshot_or_required_field_is_failure") is True
        and quiet.get("unknown_or_not_evaluated_gate_is_failure") is True,
        "quiet-host arbitrary passed=true would be accepted",
    )
    _require(
        quiet.get("preflight")
        == {
            "snapshot_count": 5,
            "snapshot_interval_ms": 250,
            "every_snapshot_must_pass": True,
            "must_complete_before_prepare_marker_creation": True,
            "failed_pre_marker_preflight_consumes_campaign": False,
            "retry_without_marker_or_generation_after_host_becomes_quiet_allowed": True,
        },
        "quiet-host preflight schema drifted",
    )
    _require(
        quiet.get("continuous")
        == {
            "sample_interval_ms": 250,
            "maximum_inter_sample_gap_ms": 500,
            "starts_before_preflight_first_snapshot": True,
            "covers_prepare_marker_creation_and_run_claim": True,
            "covers_runtime_launch_and_preflight": True,
            "starts_before_first_warmup_cache_clear": True,
            "covers_all_warmup_and_measured_requests": True,
            "continues_through_postflight_last_snapshot": True,
            "every_snapshot_must_pass": True,
            "raw_snapshots_must_be_crash_safe_and_durable": True,
        },
        "quiet-host continuous pre/during/post coverage drifted",
    )
    _require(
        quiet.get("postflight")
        == {
            "cooldown_ms": 1000,
            "snapshot_count": 5,
            "snapshot_interval_ms": 250,
            "every_snapshot_must_pass": True,
        },
        "quiet-host postflight schema drifted",
    )
    required_snapshot_fields = {
        "sequence",
        "phase",
        "monotonic_ns",
        "wall_utc",
        "power_source",
        "thermal_warning",
        "performance_warning",
        "load_average_1m",
        "swap_used_bytes",
        "campaign_process_swap_bytes",
        "pages_throttled",
        "memory_pressure",
        "processes",
        "resolved_allowlist",
        "non_allowlisted_single_cpu_percent_max",
        "non_allowlisted_aggregate_cpu_percent",
    }
    _require(
        set(quiet.get("snapshot_required_fields", [])) == required_snapshot_fields
        and quiet.get("snapshot_unknown_fields_allowed") is False,
        "quiet-host raw snapshot schema is incomplete",
    )
    thresholds = _object(quiet.get("thresholds"), "quiet-host thresholds")
    _require(
        thresholds
        == {
            "power_source_required": "AC Power",
            "thermal_warning_allowed": False,
            "performance_warning_allowed": False,
            "maximum_load_average_per_logical_cpu": 0.5,
            "maximum_single_non_allowlisted_process_cpu_percent": 10.0,
            "maximum_aggregate_non_allowlisted_process_cpu_percent": 25.0,
            "system_swap_delta_bytes_required": 0,
            "campaign_process_swap_bytes_required": 0,
            "memory_pressure_pages_throttled_delta_required": 0,
            "power_or_thermal_state_change_allowed": False,
        }
        and quiet.get("allowlist_matching")
        == "exact PID plus process-start identity plus declared ancestry"
        and quiet.get("process_name_path_or_bundle_prefix_wildcards_allowed") is False
        and quiet.get("driver_may_terminate_user_or_system_processes") is False
        and quiet.get("failed_pre_marker_effect")
        == "NO_CAMPAIGN_MARKER_NO_GENERATION_MAY_RETRY_AFTER_HOST_IS_QUIET"
        and quiet.get("post_marker_failure_effect")
        == "FORMAL_CAMPAIGN_CONSUMED_UNRANKABLE",
        "quiet-host thresholds or fail-closed policy drifted",
    )


def _validate_runtime_custody(contract: dict[str, Any]) -> None:
    custody = _object(
        contract.get("runtime_custody_contract"), "live runtime custody contract"
    )
    _require(
        custody.get("custody_checkpoints") == CUSTODY_CHECKPOINTS,
        "live custody checkpoint coverage drifted",
    )
    process = _object(custody.get("process_identity"), "live process custody")
    _require(
        set(process.get("required_for_driver_gateway_apxinf_and_backend", []))
        == {
            "pid",
            "process_start_time",
            "parent_pid",
            "process_group_id",
            "absolute_executable_path",
            "executable_device_inode_mode_size_ctime",
            "executable_sha256",
            "argv",
            "argv_sha256",
            "environment_allowlist_values_and_sha256",
            "code_signature_identity_or_explicit_unsigned_state",
        }
        and process.get("unexpected_restart_allowed") is False
        and process.get("start_checkpoint_and_end_checkpoint_identity_must_equal")
        is True,
        "live binary/process custody drifted",
    )
    source = _object(custody.get("live_source_tree_custody"), "live source custody")
    _require(
        source.get("apxinf_inference_base_commit_required")
        == "80049e7f15df67356b3932370b7ab3cc06e938f8"
        and source.get("apxinf_inference_base_tree_required")
        == "f922486224d5f8b8d68e14d36b4c0fec75a711bb"
        and source.get(
            "formal_runtime_HEAD_must_equal_prepare_HEAD_and_live_remote_ref"
        )
        is True
        and source.get("formal_runtime_HEAD_must_descend_from_inference_base_commit")
        is True
        and source.get(
            "formal_runtime_commit_and_tree_must_be_recorded_in_prepare_marker"
        )
        is True
        and source.get("clean_worktree_required") is True
        and source.get("build_input_manifest_with_git_blob_and_sha256_required") is True
        and source.get("binary_embedded_source_commit_must_equal_formal_runtime_HEAD")
        is True
        and source.get("source_manifest_root_start_end_and_all_checkpoints_equal")
        is True
        and source.get("omniinfer_and_backend_clean_source_checkouts_required") is True
        and source.get(
            "omniinfer_and_backend_build_input_manifest_with_git_blob_and_sha256_required"
        )
        is True,
        "live source-tree custody drifted",
    )
    libraries = _object(custody.get("loaded_library_custody"), "loaded library custody")
    _require(
        all(
            libraries.get(field) is True
            for field in (
                "all_non_system_loaded_images_recorded_with_path_inode_size_ctime_and_sha256",
                "system_library_paths_and_code_signature_identities_recorded",
                "loaded_image_closure_root_required_per_process",
                "closure_root_start_end_and_all_checkpoints_equal",
            )
        )
        and libraries.get("dynamic_backend_scan_or_late_library_load_allowed") is False,
        "live loaded-library custody drifted",
    )
    model_fd = _object(
        custody.get("model_file_descriptor_custody"), "live model-FD custody"
    )
    _require(
        model_fd.get("controller_opens_all_model_and_tokenizer_artifacts_with")
        == ["O_RDONLY", "O_NOFOLLOW", "O_CLOEXEC"]
        and model_fd.get("controller_holds_descriptors_from")
        == "prepare-before-runtime-launch"
        and model_fd.get("controller_holds_descriptors_through")
        == "raw-receipt-durable-before-cleanup"
        and set(model_fd.get("required_identity_fields", []))
        == {
            "absolute_path",
            "device",
            "inode",
            "mode",
            "link_count",
            "size_bytes",
            "ctime_ns",
            "sha256",
        }
        and model_fd.get("single_link_regular_files_required") is True
        and model_fd.get(
            "runtime_loaded_descriptor_or_mapped_vnode_must_equal_controller_identity"
        )
        is True
        and model_fd.get("independent_runtime_observers")
        == ["libproc-PROC_PIDLISTFDS+PROC_PIDFDVNODEPATHINFO", "lsof", "vmmap"]
        and model_fd.get("independent_observers_must_agree") is True
        and model_fd.get("identity_checked_at_every_custody_checkpoint") is True
        and model_fd.get("path_only_or_hash_only_is_sufficient") is False,
        "live model file-descriptor custody drifted",
    )
    apx = _object(custody.get("apxinf_artifact_custody"), "ApxInf artifact custody")
    _require(
        apx.get("safetensors_size_bytes") == 1746942600
        and apx.get("safetensors_sha256")
        == "04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696"
        and apx.get("source_lock_size_bytes") == 40486
        and apx.get("source_lock_sha256")
        == "55a43c039fe03cbe0c5a2feefffa46eba526db394a2ce3fa03075945d50ca268"
        and apx.get("source_lock_must_be_strictly_schema_validated") is True
        and apx.get(
            "source_lock_model_revision_and_every_artifact_hash_must_match_live_files"
        )
        is True
        and apx.get("tokenizer_config_manifest_root_required") is True
        and apx.get("resident_packed_weight_and_buffer_manifest_root_required") is True
        and apx.get("server_reported_raw_source_lock_hash_alone_is_sufficient")
        is False,
        "ApxInf model/source-lock custody drifted",
    )
    omni = _object(
        custody.get("omniinfer_artifact_custody"), "OmniInfer artifact custody"
    )
    _require(
        omni.get("omniinfer_cli_sha256")
        == "65487424ca9179850b80079beafa5ad69a66e0841d328ee8dd8a1fd4b613d661"
        and omni.get("backend_binary_sha256")
        == "02723fc39fbeebd9849ce4c9ca3799649df3cf91f101c2cd56b8756e1db54d28"
        and omni.get("gguf_size_bytes") == 811843072
        and omni.get("gguf_sha256")
        == "427555e797eefeb62e1c8ef71510ce062027b0c7f1674cc4cfcb352710e0908c"
        and omni.get("gateway_backend_pid_port_endpoint_state_cross_check_required")
        is True
        and omni.get("gateway_backend_identity_start_end_and_all_checkpoints_equal")
        is True
        and custody.get("network_download_or_artifact_mutation_after_prepare_allowed")
        is False
        and custody.get("null_missing_or_mismatched_custody_value_is_failure") is True,
        "OmniInfer artifact/runtime custody drifted",
    )


def _validate_execution_and_receipts(contract: dict[str, Any]) -> None:
    execution = _object(contract.get("execution_protocol"), "prepare/run protocol")
    _require(
        execution.get("commands") == ["prepare", "run"]
        and execution.get("prepare_makes_generation_requests") is False
        and execution.get("run_requires_successful_prepare") is True,
        "one-shot prepare/run protocol drifted",
    )
    marker = _object(execution.get("one_shot_marker"), "one-shot marker")
    marker_true = {
        "path_must_be_absolute_and_absent",
        "created_with_O_CREAT_O_EXCL",
        "contains_contract_validator_driver_binary_library_source_model_and_quiet_preflight_bindings",
        "file_and_parent_directory_fsync_required",
        "run_claims_marker_by_atomic_no_replace_transition",
        "claimed_marker_cannot_be_reused",
        "marker_creation_or_claim_failure_is_terminal",
    }
    _require(
        all(marker.get(field) is True for field in marker_true)
        and marker.get("second_run_under_same_campaign_id_allowed") is False,
        "one-shot prepare/run marker was weakened",
    )
    slots = _object(execution.get("raw_per_slot_receipts"), "raw per-slot receipts")
    _require(
        slots.get("expected_count") == 136
        and slots.get("one_receipt_for_every_warmup_and_measured_arm_slot") is True
        and slots.get("predeclared_slot_identity_and_order_required") is True
        and slots.get(
            "write_sibling_temp_fsync_atomic_no_replace_rename_and_parent_fsync"
        )
        is True
        and slots.get("failed_slot_observation_must_be_durable") is True
        and slots.get("aggregate_only_receipt_is_sufficient") is False
        and slots.get("retry_or_replacement_after_failed_or_partial_slot_allowed")
        is False
        and slots.get(
            "summary_created_only_after_all_slots_host_postflight_and_custody_are_durable"
        )
        is True
        and execution.get("host_monitor_snapshots_are_durable_during_run") is True
        and execution.get("crash_after_marker_claim_consumes_campaign") is True
        and execution.get("crash_preserves_all_previously_durable_raw_receipts") is True
        and execution.get("cleanup_before_raw_receipt_durable_allowed") is False,
        "crash-safe durable per-slot receipt protocol drifted",
    )

    machine = _object(
        contract.get("machine_receipt_contract"), "machine receipt contract"
    )
    _require(
        machine.get("formal_driver_must_strictly_validate_receipts") is True
        and machine.get("required_true_gate_ids") == REQUIRED_GATE_IDS
        and machine.get("all_listed_gates_must_be_boolean_true") is True
        and machine.get("missing_null_unknown_or_not_applicable_gate_is_failure")
        is True
        and machine.get(
            "apxinf_server_boolean_or_compact_receipt_without_external_field_validation_is_sufficient"
        )
        is False
        and machine.get(
            "aggregate_statistics_without_raw_per_slot_receipts_are_formal_evidence"
        )
        is False,
        "machine receipt or Apx compact-path external validation drifted",
    )
    _require(
        "full_generation_path_receipt_raw"
        in machine.get("AH_per_slot_additional_required_fields", [])
        and "server_compact_generation_path_receipt_raw"
        in machine.get("AH_per_slot_additional_required_fields", [])
        and "externally_rederived_compact_generation_path_receipt"
        in machine.get("AH_per_slot_additional_required_fields", [])
        and "compact_generation_path_external_validation"
        in machine.get("AH_per_slot_additional_required_fields", [])
        and "five_EOG_logit_bias_receipt"
        in machine.get("OG_per_slot_additional_required_fields", []),
        "arm-specific raw receipt schema is incomplete",
    )
    _require(
        "preflight_prompt_tokenize_receipt_reference"
        in machine.get("per_slot_required_fields", [])
        and "prompt_tokenize_receipt"
        not in machine.get("per_slot_required_fields", []),
        "per-slot tokenization must reference the one durable preflight receipt",
    )


def _validate_statistics_and_claims(contract: dict[str, Any]) -> None:
    statistics = _object(
        contract.get("statistics_and_decision_contract"),
        "statistics and decisions",
    )
    _require(
        statistics.get("primary_observation") == "client_full_response_wall_ms"
        and statistics.get("ratio_orientation") == "AH_over_OG"
        and statistics.get("ratio_interpretation")
        == "ratio below one means the named AH deployment has lower client wall latency"
        and statistics.get("warmups_in_statistics") is False
        and statistics.get("measured_pair_count") == 64
        and statistics.get("finite_strictly_positive_observation_required") is True
        and statistics.get("pair_log_ratio")
        == "log(AH_client_full_response_wall_ms / OG_client_full_response_wall_ms)"
        and statistics.get("block_mean_is_unweighted_arithmetic_mean") is True
        and statistics.get("number_of_block_values") == 16
        and statistics.get("degrees_of_freedom") == 15
        and statistics.get("sample_standard_deviation_denominator") == 15
        and statistics.get("standard_error")
        == "sample standard deviation of sixteen block log ratios divided by sqrt(16)"
        and statistics.get("t_critical_0_975") == 2.131449545559323
        and statistics.get("half_campaign_ratios")
        == "exponentiated unweighted means of blocks 1-8 and blocks 9-16 separately"
        and statistics.get("order_strata")
        == "AH-first pair log ratios versus OG-first pair log ratios"
        and statistics.get("population_cv_definition")
        == "population standard deviation with denominator 64 divided by arithmetic mean across the 64 measured wall observations of that arm"
        and statistics.get("order_stratum_difference_definition")
        == "absolute difference of unweighted means of the 32 AH-first and 32 OG-first measured pair log ratios"
        and statistics.get("first8_last8_difference_definition")
        == "absolute difference of unweighted means of block log ratios 1-8 and 9-16"
        and statistics.get("practical_thresholds_AH_over_OG") == [0.95, 1.05],
        "statistics and decisions did not retain 16 blocks, df=15, and 5% thresholds",
    )
    stability = _object(statistics.get("stability_gates"), "statistics stability gates")
    _require(
        stability
        == {
            "AH_wall_population_cv_max": 0.03,
            "OG_wall_population_cv_max": 0.03,
            "absolute_order_stratum_pair_log_ratio_mean_difference_max": 0.02,
            "absolute_first8_last8_block_log_ratio_mean_difference_max": 0.02,
            "first8_and_last8_ratios_must_support_same_decision": True,
            "all_admission_custody_quiet_schedule_timing_and_semantic_gates_must_pass": True,
        },
        "statistics stability and same-halves gates drifted",
    )
    decisions = _object(statistics.get("decision_rules"), "statistics decision rules")
    _require(
        decisions
        == {
            "NAMED_APXINF_DEPLOYMENT_AT_LEAST_5_PERCENT_FASTER": "all gates pass AND upper_ci95_AH_over_OG < 0.95 AND first8_ratio < 0.95 AND last8_ratio < 0.95",
            "NAMED_OMNIINFER_DEPLOYMENT_AT_LEAST_5_PERCENT_FASTER": "all gates pass AND lower_ci95_AH_over_OG > 1.05 AND first8_ratio > 1.05 AND last8_ratio > 1.05",
            "NAMED_DEPLOYMENTS_PRACTICALLY_EQUIVALENT_WITHIN_5_PERCENT": "all gates pass AND lower_ci95_AH_over_OG >= 0.95 AND upper_ci95_AH_over_OG <= 1.05 AND first8_ratio in [0.95,1.05] AND last8_ratio in [0.95,1.05]",
            "INCONCLUSIVE": "all gates pass and every other confidence-interval or same-halves outcome",
            "UNRANKABLE": "any admission custody quiet-host schedule timing semantic determinism or stability gate fails",
        }
        and statistics.get("decision_precedence")
        == [
            "UNRANKABLE",
            "NAMED_APXINF_DEPLOYMENT_AT_LEAST_5_PERCENT_FASTER",
            "NAMED_OMNIINFER_DEPLOYMENT_AT_LEAST_5_PERCENT_FASTER",
            "NAMED_DEPLOYMENTS_PRACTICALLY_EQUIVALENT_WITHIN_5_PERCENT",
            "INCONCLUSIVE",
        ]
        and statistics.get("point_estimate_alone_may_select_winner") is False
        and statistics.get("secondary_metrics_may_change_primary_decision") is False,
        "statistics CI, practical-threshold, or same-halves decision rules drifted",
    )

    failure = _object(contract.get("failure_contract"), "failure contract")
    _require(
        failure.get("fail_closed") is True
        and failure.get("first_post_marker_failure_stops_dispatch") is True
        and failure.get("campaign_consumed") is True
        and failure.get("retry_replacement_resampling_or_same_id_restart_allowed")
        is False
        and failure.get("formal_summary_or_ranking_allowed_after_failure") is False,
        "failure contract is not fail-closed",
    )
    claims = _object(contract.get("claim_policy"), "named-deployment claim policy")
    forbidden = set(claims.get("always_forbidden", []))
    _require(
        claims.get("mandatory_result_label")
        == "unmatched-serialized-weight-and-trajectory named-resident-deployment single-request HTTP-wall comparison"
        and "ApxInf engine versus OmniInfer engine general ranking" in forbidden
        and "same serialized weights quantization KV numerical regime or trajectory"
        in forbidden
        and "relabeling the v1 NON_FORMAL diagnostic as formal" in forbidden
        and "winner claim from point estimate or CI without same-halves support"
        in forbidden,
        "named-deployment claim boundary was weakened",
    )


def _validate_readiness_and_scope(contract: dict[str, Any]) -> None:
    readiness = _object(contract.get("current_readiness"), "current readiness")
    _require(
        readiness.get("formal_campaign_may_start_now") is False
        and readiness.get("blocker_codes") == BLOCKER_CODES
        and readiness.get("validator_success_clears_any_blocker") is False,
        "static validator must not claim the formal campaign is ready",
    )
    scope = _object(contract.get("validator_scope"), "validator scope")
    _require(
        scope
        == {
            "validates_only_this_static_predeclaration": True,
            "implements_formal_driver": False,
            "starts_services_or_makes_generation_requests": False,
            "validates_a_future_quiet_host_receipt_instance": False,
            "produces_or_upgrades_performance_evidence": False,
            "validator_success_means_campaign_ready": False,
        },
        "validator scope expanded into a driver, run validator, or evidence producer",
    )


def validate_contract(contract: dict[str, Any]) -> None:
    """Validate parsed semantics, ending with a whole-document semantic pin."""

    _validate_identity_and_lineage(contract)
    _validate_named_deployment_boundary(contract)
    _validate_workload(contract)
    _validate_schedule_and_timing(contract)
    _validate_quiet_host(contract)
    _validate_runtime_custody(contract)
    _validate_execution_and_receipts(contract)
    _validate_statistics_and_claims(contract)
    _validate_readiness_and_scope(contract)
    observed = _sha256(_canonical_json_bytes(contract))
    _require(
        observed == PINNED_CANONICAL_SEMANTIC_SHA256,
        "whole-contract canonical semantic pin mismatch",
    )


def load_contract(path: Path) -> tuple[dict[str, Any], bytes]:
    """Read one direct file snapshot and validate both its bytes and semantics."""

    try:
        metadata = path.lstat()
    except OSError as error:
        raise FormalContractError(f"cannot lstat contract: {error}") from error
    _require(stat.S_ISREG(metadata.st_mode), "contract must be a direct regular file")
    _require(metadata.st_size <= MAX_CONTRACT_BYTES, "contract file is oversized")
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise FormalContractError(f"cannot read contract: {error}") from error
    _require(len(raw) == metadata.st_size, "contract size changed while reading")
    contract = parse_strict_json(raw)
    validate_contract(contract)
    _require(
        _sha256(raw) == PINNED_FILE_SHA256, "frozen contract file-byte pin mismatch"
    )
    return contract, raw


def validation_receipt(
    contract: dict[str, Any], raw: bytes, path: Path
) -> dict[str, Any]:
    return {
        "format": VALIDATION_FORMAT,
        "schema_version": 1,
        "valid": True,
        "validation_scope": "STATIC_PREDECLARATION_ONLY",
        "campaign_id": CAMPAIGN_ID,
        "deployment_edge_id": EDGE_ID,
        "contract_path": str(path.resolve()),
        "contract_file_size_bytes": len(raw),
        "contract_file_sha256": _sha256(raw),
        "contract_canonical_semantic_sha256": _sha256(_canonical_json_bytes(contract)),
        "formal_driver_implemented_by_validator": False,
        "services_started_or_generation_requests_made": False,
        "future_run_receipt_validated": False,
        "formal_campaign_ready": False,
        "blocker_codes": list(BLOCKER_CODES),
        "prior_nonformal_evidence_upgraded": False,
        "performance_result": None,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        contract, raw = load_contract(args.contract)
        receipt = validation_receipt(contract, raw, args.contract)
        output = _canonical_json_bytes(receipt)
    except (FormalContractError, OSError) as error:
        print(f"formal contract rejected: {error}", file=sys.stderr)
        return 1
    sys.stdout.write(output.decode("utf-8") + "\n")
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
