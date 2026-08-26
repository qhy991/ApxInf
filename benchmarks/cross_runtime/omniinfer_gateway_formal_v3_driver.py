#!/usr/bin/env python3
"""Fail-closed formal-v3 driver for the OmniInfer gateway increment edge.

This is deliberately a separate campaign from the native-engine comparison.
Arm B talks directly to one already-resident llama-server and arm G talks to
that same process through OmniInfer.  ``prepare`` performs no generation and
creates the campaign marker only after every immutable/runtime/host proof is
complete.  ``run`` requires that marker at live ``refs/heads/main`` before it
opens the two persistent measurement connections or sends the first warmup.

Because the pinned upstream llama-server does not expose a pre-load descriptor
receipt, a driver-owned custodian opens and hashes the model with
O_RDONLY|O_NOFOLLOW|O_CLOEXEC before it launches the gateway/backend tree.  It
holds that FD through public activation and postflight; lsof and libproc must
independently agree that the backend's live descriptor names the same vnode.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import copy
import ctypes
import fcntl
import hashlib
import http.client
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import re
import secrets
import select
import signal
import socket
import stat
import struct
import subprocess
import sys
import threading
import time
import traceback
import urllib.parse
from typing import Any


_NATIVE_DRIVER_PATH = Path(__file__).with_name("formal_v3_driver.py")
_NATIVE_SPEC = importlib.util.spec_from_file_location(
    "_apxinf_formal_v3_shared", _NATIVE_DRIVER_PATH
)
if _NATIVE_SPEC is None or _NATIVE_SPEC.loader is None:
    raise RuntimeError("cannot locate formal-v3 shared implementation")
_SHARED = importlib.util.module_from_spec(_NATIVE_SPEC)
_NATIVE_SPEC.loader.exec_module(_SHARED)

CampaignError = _SHARED.CampaignError
ReceiptError = _SHARED.ReceiptError

EDGE_ID = "GATEWAY_B_VS_G"
PLAN_FORMAT = "apxinf-qwen35-omniinfer-gateway-formal-execution-plan-v3"
MARKER_FORMAT = "apxinf-qwen35-omniinfer-gateway-formal-campaign-start-v3"
SAMPLE_FORMAT = "apxinf-qwen35-omniinfer-gateway-sample-receipt-v3"
RAW_FORMAT = "apxinf-qwen35-omniinfer-gateway-formal-raw-campaign-v3"
HOST_FORMAT = _SHARED.HOST_FORMAT
CUSTODIAN_READY_FORMAT = "apxinf-omniinfer-gateway-custodian-ready-v3"
CUSTODIAN_ATTESTATION_FORMAT = "apxinf-omniinfer-gateway-custodian-attestation-v3"
CUSTODIAN_CLEANUP_FORMAT = "apxinf-omniinfer-gateway-custodian-cleanup-v3"
CUSTODIAN_MODEL_LOAD_COMPLETION_MARGIN_SECONDS = 30.0
CUSTODIAN_PARENT_READY_RECEIPT_MARGIN_SECONDS = 30.0
CUSTODIAN_CLEANUP_RESPONSE_MARGIN_SECONDS = 5.0
CUSTODIAN_CONTROL_REQUEST_MAX_BYTES = 1024 * 1024
CUSTODIAN_CONTROL_RESPONSE_MAX_BYTES = 16 * 1024 * 1024

SAMPLE_REQUIRED_FIELDS_V3 = frozenset(
    {
        "format",
        "schema_version",
        "campaign_id",
        "subcampaign_id",
        "edge_id",
        "slot",
        "request",
        "cache_clear",
        "workload",
        "timing",
        "native_sensitivity",
        "connection",
        "response",
        "response_bytes",
    }
)
SAMPLE_OPTIONAL_FIELDS_V3 = frozenset({"segment_state_after"})
SAMPLE_NESTED_FIELDS_V3 = {
    "request": frozenset(
        {
            "canonical_json_object",
            "canonical_utf8",
            "size_bytes",
            "sha256",
            "same_body_for_B_and_G",
        }
    ),
    "cache_clear": frozenset(
        {
            "acknowledged",
            "cleared_slots",
            "response",
            "outside_primary_timed_interval",
            "transport",
        }
    ),
    "workload": frozenset(
        {
            "rendered_prompt",
            "prompt_token_ids",
            "generated_token_ids",
            "generated_token_ids_sha256",
            "content",
            "content_sha256",
            "usage",
            "usage_object",
            "generation_settings",
            "generation_settings_sha256",
            "per_sample_tokenize_admission_outside_timed_interval",
        }
    ),
    "timing": frozenset(
        {
            "clock",
            "clock_identity",
            "clock_resolution_ns",
            "clock_is_monotonic",
            "clock_is_adjustable",
            "start_boundary",
            "end_boundary",
            "implementation_start_boundary",
            "implementation_end_boundary",
            "request_serialization_before_start",
            "first_wire_byte_send_call_immediately_after_start",
            "full_response_body_read_before_end",
            "strict_json_parse_before_end",
            "semantic_validation_before_end",
            "json_parse_excluded_from_wall",
            "semantic_validation_excluded_from_wall",
            "request_wire_size_bytes",
            "request_wire_sha256",
            "request_wire_base64",
            "request_wire_body_offset_bytes",
            "request_wire_body_size_bytes",
            "request_wire_body_sha256",
            "request_wire_body_equals_request_body",
            "single_sendall_call_count",
            "single_sendall_argument_size_bytes",
            "single_sendall_argument_sha256",
            "timing_event_order",
            "complete_HTTP_request_wire_serialization_before_start",
            "single_sendall_call_for_complete_request_wire_required",
            "canonical_383_byte_JSON_body_identical_between_B_and_G",
            "arm_specific_HTTP_authority_header_difference_is_inside_timed_region",
            "body_only_timing_allowed",
            "started_monotonic_ns",
            "ended_monotonic_ns",
            "client_full_response_wall_ns",
            "client_full_response_wall_ms",
        }
    ),
    "native_sensitivity": frozenset(
        {"prompt_ms", "predicted_ms", "prompt_tps", "predicted_tps"}
    ),
    "connection": frozenset(
        {
            "connection_generation",
            "request_index_on_connection",
            "socket",
            "socket_start_end_equal",
            "reconnect_count",
        }
    ),
    "response_bytes": frozenset({"encoding", "base64", "size_bytes", "sha256"}),
}
SAMPLE_TOKENIZE_ADMISSION_FIELDS_V3 = frozenset({"token_ids", "transport"})
SAMPLE_SEGMENT_STATE_FIELDS_V3 = frozenset(
    {
        "state",
        "transport",
        "controller_backend_fd_custody",
        "outside_primary_timed_interval",
    }
)
GATEWAY_MACHINE_RECEIPT_FIELDS_V3 = frozenset(
    {
        "format",
        "schema_version",
        "campaign_id",
        "subcampaign_id",
        "edge_id",
        "campaign_consumed",
        "status",
        "raw_output_path",
        "contract_binding",
        "git_custody",
        "host_custody",
        "artifact_custody",
        "artifact_custody_end",
        "parity_admission",
        "schedule_receipt",
        "samples",
        "statistics",
        "gates",
        "decision",
        "postflight",
        "failures",
        "custodian_cleanup",
    }
)
GATEWAY_CONTRACT_BINDING_FIELDS_V3 = frozenset(
    {
        "campaign_id",
        "schema_version",
        "edge_id",
        "subcampaign_id",
        "repository_url",
        "remote_origin_url",
        "local_tracking_ref",
        "local_tracking_oid",
        "live_remote_url",
        "live_remote_ref",
        "ls_remote_argv",
        "ls_remote_exit_code",
        "ls_remote_live_oid",
        "head_commit",
        "contract_repository_path",
        "contract_commit",
        "contract_tree",
        "contract_blob_oid",
        "contract_blob_size_bytes",
        "contract_blob_sha256",
        "observed_file_size_bytes",
        "observed_file_sha256",
        "gateway_driver_repository_path",
        "gateway_driver_blob_sha256",
        "validator_repository_path",
        "validator_blob_sha256",
        "shared_formal_driver_repository_path",
        "shared_formal_driver_blob_sha256",
        "plan_repository_path",
        "plan_blob_sha256",
        "activation_commit",
        "activation_tree",
        "activation_contract_blob_oid",
        "activation_contract_blob_size_bytes",
        "activation_contract_blob_sha256",
        "activation_marker_repository_path",
        "activation_marker_blob_oid",
        "activation_marker_blob_size_bytes",
        "activation_marker_blob_sha256",
        "contract_commit_is_ancestor_of_activation_commit",
        "activation_commit_equals_head_and_live_remote_oid",
        "local_tracking_ref_used_as_publication_proof",
        "worktree_clean",
    }
)
GATEWAY_DECISION_FIELDS_V3 = frozenset(
    {
        "label",
        "formal_summary_allowed",
        "engine_winner_or_ranking_claim_allowed",
    }
)
GATEWAY_MARKER_FIELDS_V3 = frozenset(
    {
        "format",
        "schema_version",
        "campaign_id",
        "subcampaign_id",
        "edge_id",
        "contract_repository_path",
        "contract_blob_size_bytes",
        "contract_blob_sha256",
        "validator_repository_path",
        "validator_blob_sha256",
        "driver_repository_path",
        "driver_blob_sha256",
        "shared_formal_driver_repository_path",
        "shared_formal_driver_blob_sha256",
        "plan_repository_path",
        "plan_blob_size_bytes",
        "plan_blob_sha256",
        "marker_repository_path",
        "pre_marker_git_custody",
        "artifact_expectations",
        "artifact_file_observations",
        "runtime_preflight",
        "host_preflight",
        "blocker_resolution",
        "declared_schedule",
        "sampling_state_at_marker_creation",
        "pre_marker_admission",
        "next_required_action",
    }
)
GATEWAY_MARKER_SAMPLING_STATE_FIELDS_V3 = frozenset(
    {"generation_requests", "warmup_samples", "timed_samples"}
)
GATEWAY_MARKER_ADMISSION_FIELDS_V3 = frozenset(
    {
        "immutable_contract_validator_driver_plan",
        "live_public_git_and_clean_worktree",
        "O_NOFOLLOW_artifact_custody",
        "same_resident_backend_pid_start_argv_environment_model_fd_and_closure",
        "controller_preload_fd_and_backend_loaded_fd_crosscheck",
        "canonical_request_and_raw_prompt",
        "request_history_disabled",
        "quiet_host",
        "marker_absent_through_preflight",
        "all_passed",
    }
)
GATEWAY_MARKER_BLOCKER_RESOLUTION_FIELDS_V3 = frozenset(
    {
        "authored_formally_admitted",
        "authored_blocker_codes",
        "resolution_map",
        "all_pre_marker_blockers_except_public_activation_resolved",
        "all_resolved",
        "authored_state_was_not_mutated",
    }
)
GATEWAY_MARKER_BLOCKER_ENTRY_FIELDS_V3 = frozenset({"resolved", "evidence"})

CONTRACT_REPOSITORY_PATH = "configs/qwen35-0.8b-cross-runtime-formal-v3.json"
VALIDATOR_REPOSITORY_PATH = "scripts/validate_qwen35_cross_runtime_formal_contract.py"
DRIVER_REPOSITORY_PATH = (
    "benchmarks/cross_runtime/omniinfer_gateway_formal_v3_driver.py"
)
SHARED_DRIVER_REPOSITORY_PATH = "benchmarks/cross_runtime/formal_v3_driver.py"
MARKER_REPOSITORY_PATH = (
    "crates/apxinf-metal/evidence/llama-cpp/"
    "qwen35-0.8b-omniinfer-gateway-increment-formal-v3-"
    "campaign-start-20260826.json"
)

MODEL_PATH = (
    "/Users/haiyan-mini/Agent4Kernel/models/Qwen3.5-0.8B-2fc063647-GGUF/"
    "Qwen3.5-0.8B-2fc063647-Q8_0-llama.gguf"
)
MODEL_SIZE = 811_843_072
MODEL_SHA256 = "427555e797eefeb62e1c8ef71510ce062027b0c7f1674cc4cfcb352710e0908c"
OMNI_SOURCE_COMMIT = "79af77228f329a79ac665014089e23983e69e79f"
OMNI_ARCHIVE_SHA256 = "0f83ea36aad7126976ff2a53a58f0ce20e934d2bd0133d40f4ce974658a48cf4"
OMNI_CLI_SIZE = 9_719_136
OMNI_CLI_SHA256 = "65487424ca9179850b80079beafa5ad69a66e0841d328ee8dd8a1fd4b613d661"
BACKEND_SOURCE_COMMIT = "61881b1f7f0b13d9e46d561fc25afcd6bbaec479"
BACKEND_ARCHIVE_SHA256 = (
    "5dc4b11192ef34895c7f92a9f1dd3bd3d5864a63976ea2327fe2e0944891cb75"
)
BACKEND_BINARY_SIZE = 33_472
BACKEND_BINARY_SHA256 = (
    "02723fc39fbeebd9849ce4c9ca3799649df3cf91f101c2cd56b8756e1db54d28"
)
BACKEND_BUILD_INFO = "b10280-61881b1f7"
PINNED_CORE_SOURCE_COMMIT = "f280b26983ad0fdb705a0d9ebf0503e76f2899b0"

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
RENDERED_PROMPT = (
    "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
)
REQUEST: dict[str, Any] = {
    "cache_prompt": False,
    "chat_template_kwargs": {"enable_thinking": False},
    "id_slot": 0,
    "ignore_eos": True,
    "max_tokens": 128,
    "messages": [{"content": "Hello", "role": "user"}],
    "model": MODEL_PATH,
    "reasoning_format": "none",
    "return_tokens": True,
    "seed": 0,
    "stream": False,
    "temperature": 0,
    "verbose": True,
}
REQUEST_SIZE = 383
REQUEST_SHA256 = "7773f5337693843f1e8cf3017b98868517cbddd3bc32649e550d8f2fec1d5cf6"
EXPECTED_VERBOSE_TOKENS_CACHED = 140
T_CRITICAL_DF15_975 = 2.131449545559323
MAX_RESPONSE_BYTES = 16 * 1024 * 1024
GATEWAY_TIMING_START_V3 = (
    "immediately-before-the-single-sendall-call-for-the-complete-pre-serialized-"
    "arm-specific-HTTP/1.1-request-wire-containing-the-identical-canonical-383-byte-"
    "JSON-body-on-a-warmed-persistent-client-connection"
)
COMPLETE_WIRE_TIMING_EVENT_ORDER_V3 = [
    "complete-HTTP-request-wire-serialized",
    "start-monotonic-timestamp-captured",
    "single-sendall-call-for-complete-request-wire",
    "response-headers-read",
    "full-response-body-read",
    "strict-JSON-parse-complete",
    "semantic-validation-complete",
    "end-monotonic-timestamp-captured",
]
GATEWAY_COMPLETE_WIRE_TIMING_CONTRACT_V3 = {
    "complete_HTTP_request_wire_serialization_before_start": True,
    "single_sendall_call_for_complete_request_wire_required": True,
    "canonical_383_byte_JSON_body_identical_between_B_and_G": True,
    "arm_specific_HTTP_authority_header_difference_is_inside_timed_region": True,
    "body_only_timing_allowed": False,
}

GATEWAY_GATE_IDS = (
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
)

GATEWAY_AUTHORED_BLOCKERS = (
    "V3_GATEWAY_SUBCAMPAIGN_NOT_PUBLICLY_ACTIVATED",
    "V3_FORMAL_DRIVER_HASH_NOT_CAPTURED",
    "QUIET_HOST_GATE_NOT_YET_PASSED",
)

GATEWAY_MODEL_CUSTODY_PROOF_ID = "GATEWAY_CONTROLLER_HELD_VNODE_AND_BACKEND_FD_V3"
GATEWAY_MODEL_FD_CHECKPOINTS_V3 = [
    "prepare-before-any-generation",
    "runtime-preflight-start",
    "runtime-preflight-end",
    "before-first-generation",
    "after-warmups",
    *[f"after-timed-macroblock-{index}" for index in range(16)],
    "runtime-postflight-before-cleanup",
]
GATEWAY_MODEL_CUSTODY_EDGE_FIELDS_V3: dict[str, Any] = {
    "same_loaded_model_file_description_required": False,
    "same_backend_loaded_model_fd_for_B_and_G_required": True,
    "controller_and_backend_same_open_file_description_claim_allowed": False,
    "controller_and_backend_same_vnode_identity_required": True,
    "model_custody_proof_id": GATEWAY_MODEL_CUSTODY_PROOF_ID,
}
GATEWAY_MODEL_CUSTODY_CONTRACT_V3: dict[str, Any] = {
    "proof_id": GATEWAY_MODEL_CUSTODY_PROOF_ID,
    "controller_preload_fd": {
        "owner": "driver-owned-custodian-daemon",
        "required_open_flags": ["O_RDONLY", "O_NOFOLLOW", "O_CLOEXEC"],
        "required_file_type": "single-link-regular",
        "required_hard_link_count": 1,
        "full_sha256_required": True,
        "open_and_full_hash_must_complete_before": "gateway-and-backend-launch",
        "held_continuously_from": "prepare-before-runtime-launch",
        "held_continuously_through": "run-raw-receipt-durable-before-cleanup",
    },
    "backend_loaded_fd": {
        "required_access_mode": "read-only",
        "required_file_type": "single-link-regular",
        "independent_observers": [
            "libproc-PROC_PIDLISTFDS+PROC_PIDFDVNODEPATHINFO",
            "lsof",
        ],
        "observers_must_agree": True,
        "required_vnode_identity_fields": [
            "device",
            "inode",
            "mode",
            "link_count",
            "size_bytes",
            "ctime_ns",
            "absolute_path",
        ],
        "must_equal_controller_preload_fd_vnode_identity": True,
        "required_checkpoints": GATEWAY_MODEL_FD_CHECKPOINTS_V3,
    },
    "claim_semantics": {
        "same_backend_loaded_model_fd_shared_by_B_and_G_required": True,
        "controller_and_backend_same_vnode_identity_required": True,
        "controller_and_backend_same_open_file_description_required": False,
        "controller_and_backend_same_open_file_description_claim_allowed": False,
    },
}
GATEWAY_RUNTIME_PREFLIGHT_FIELDS_V3 = frozenset(
    {
        "format",
        "schema_version",
        "edge_id",
        "generation_requests",
        "same_resident_backend_process_for_B_and_G",
        "direct_arm_backend_endpoint",
        "gateway_arm_endpoint",
        "gateway_process_start",
        "gateway_process_end",
        "backend_process_start",
        "backend_process_end",
        "backend_start_end_identity_equal",
        "gateway_start_end_identity_equal",
        "custodian_binding",
        "zero_generation_model_load",
        "custodian_process_start",
        "custodian_process_end",
        "controller_backend_model_fd_custody",
        "state",
        "state_before_after_equal",
        "health",
        "props",
        "canonical_request",
        "rendered_prompt",
        "rendered_prompt_token_ids",
        "cache_clear",
        "history_start",
        "history_end",
        "history_start_end_equal",
        "mutable_logs_start",
        "mutable_logs_end",
        "mutable_logs_equality_not_required",
        "transport_receipts",
        "all_passed",
    }
)
GATEWAY_MODEL_FD_CUSTODY_RECEIPT_FIELDS_V3 = frozenset(
    {
        "start",
        "end",
        "controller_and_backend_same_vnode_identity",
        "same_open_file_description_not_claimed",
        "controller_fd_open_completed_before_gateway_backend_launch",
    }
)
ZERO_GENERATION_MODEL_LOAD_FIELDS_V3 = frozenset(
    {
        "format",
        "schema_version",
        "edge_id",
        "management_api",
        "management_request_sequence",
        "model_select_request_count",
        "request",
        "response",
        "pre_load_state",
        "loaded_state",
        "transport",
        "gateway_pid",
        "gateway_start_identity",
        "gateway_end_identity",
        "gateway_process_start_end_equal",
        "gateway_parent_pid_start",
        "gateway_parent_pid_end",
        "gateway_parent_start_end_equal",
        "backend_pid",
        "backend_start_identity",
        "gateway_is_direct_parent_of_backend",
        "history_start",
        "history_end",
        "history_start_end_equal",
        "generation_requests",
        "generation_endpoint_paths_called",
        "request_history_records_created",
        "all_passed",
    }
)
ZERO_GENERATION_MANAGEMENT_TRANSPORT_FIELDS_V3 = frozenset(
    {
        "connection",
        "method",
        "path",
        "request_body_size_bytes",
        "request_body_sha256",
        "request_wire_size_bytes",
        "request_wire_sha256",
        "request_wire_base64",
        "request_wire_body_offset_bytes",
        "request_wire_body_size_bytes",
        "request_wire_body_sha256",
        "request_wire_body_equals_request_body",
        "single_sendall_call_count",
        "single_sendall_argument_size_bytes",
        "single_sendall_argument_sha256",
        "status",
        "http_version",
        "response_size_bytes",
        "response_sha256",
        "response_base64",
        "request_serialization_before_start",
        "first_wire_byte_send_call_immediately_after_start",
        "complete_HTTP_request_wire_serialization_before_start",
        "single_sendall_call_for_complete_request_wire_required",
        "full_response_body_read_before_end",
        "strict_json_parse_before_end",
        "semantic_validation_before_end",
        "setup_only_zero_generation_management_request",
    }
)
CUSTODIAN_READY_FIELDS_V3 = frozenset(
    {
        "format",
        "schema_version",
        "edge_id",
        "nonce",
        "custodian_pid",
        "custodian_start_identity",
        "gateway_pid",
        "gateway_start_identity",
        "backend_pid",
        "backend_start_identity",
        "controller_preload_fd",
        "backend_loaded_fd",
        "lifecycle_sequence",
        "control_socket",
        "zero_generation_model_load",
        "generation_requests",
        "passed",
    }
)
CAMPAIGN_DIRECTORY_INITIALIZATION_FIELDS_V3 = frozenset(
    {
        "format",
        "schema_version",
        "edge_id",
        "campaign_root",
        "expected_directory_paths",
        "preexisting_directory_paths",
        "created_directory_paths",
        "directory_observations",
        "retry_policy",
        "initial_tree_sha256",
        "generation_requests",
        "runtime_processes_started",
        "marker_created",
        "raw_created",
        "all_passed",
    }
)
CAMPAIGN_DIRECTORY_OBSERVATION_FIELDS_V3 = frozenset(
    {
        "absolute_path",
        "relative_path",
        "device",
        "inode",
        "mode",
        "permission_bits",
        "uid",
        "gid",
        "expected_child_names",
        "observed_child_names",
        "direct_directory_no_symlink",
        "owner_matches_controller",
        "permissions_are_0700",
    }
)
CAMPAIGN_DIRECTORY_CLEANUP_FIELDS_V3 = frozenset(
    {
        "format",
        "schema_version",
        "edge_id",
        "reason",
        "campaign_root",
        "initialization_receipt_sha256",
        "preexisting_directory_paths",
        "cleanup_eligible",
        "contamination_detected",
        "removed_paths",
        "root_removed",
        "marker_present",
        "raw_present",
        "signals_sent",
        "all_passed",
    }
)
GATEWAY_PRELOAD_STATE_FIELDS_V3 = frozenset(
    {
        "backend",
        "backend_ready",
        "model",
        "public_model_id",
        "mmproj",
        "ctx_size",
        "request_defaults",
        "runtime_mode",
        "backend_pid",
        "backend_port",
        "launch_args",
        "cuda_visible_devices",
        "warning",
        "launch_command",
        "proxy_model",
        "external_server_protocol",
        "client_endpoint",
        "openai_compatible",
        "backend_log",
        "effective_parameters",
        "runtime",
        "loaded_models",
        "default_model",
        "restore_selection",
        "restore_status",
        "restore_completed",
        "resource_ledger",
        "available_backends",
    }
)
GATEWAY_LOADED_STATE_FIELDS_V3 = frozenset(
    {
        "backend",
        "backend_ready",
        "model",
        "model_path",
        "public_model_id",
        "owner_admin_id",
        "mmproj",
        "ctx_size",
        "request_defaults",
        "runtime_mode",
        "backend_pid",
        "backend_port",
        "generation",
        "route_state",
        "allocation_id",
        "resource_budget",
        "speculative_admission",
        "launch_args",
        "cuda_visible_devices",
        "warning",
        "launch_command",
        "proxy_model",
        "external_server_protocol",
        "client_endpoint",
        "openai_compatible",
        "backend_log",
        "effective_parameters",
        "runtime",
        "log_path",
        "loaded_models",
        "default_model",
        "restore_selection",
        "restore_status",
        "restore_completed",
        "resource_ledger",
        "available_backends",
    }
)


class RuntimeObservationError(CampaignError):
    """A live request failed after its slot became irrevocably attempted."""

    def __init__(self, message: str, observation: dict[str, Any] | None = None):
        super().__init__(message)
        self.observation = observation or {}


class PreflightBlockedError(CampaignError):
    """A prepare-only proof was unavailable; no marker was consumed."""

    def __init__(self, receipt: dict[str, Any]):
        blockers = receipt.get("blocker_codes", ["UNKNOWN_PREFLIGHT_BLOCKER"])
        super().__init__("gateway preflight blocked: " + ",".join(blockers))
        self.receipt = receipt


def raise_preflight_blocker(
    code: str, explanation: str, evidence: dict[str, Any] | None = None
) -> None:
    raise PreflightBlockedError(
        {
            "format": "apxinf-qwen35-omniinfer-gateway-preflight-blocked-v3",
            "schema_version": 3,
            "edge_id": EDGE_ID,
            "generation_requests": 0,
            "marker_created": False,
            "blocker_codes": [code],
            "explanation": explanation,
            "evidence": copy.deepcopy(evidence or {}),
        }
    )


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CampaignError(message)


def receipt_require(condition: bool, message: str) -> None:
    if not condition:
        raise ReceiptError(message)


def _remaining_deadline_seconds(
    deadline: float,
    label: str,
    *,
    monotonic: Any = time.monotonic,
) -> float:
    remaining = deadline - monotonic()
    require(remaining > 0.0, f"{label} exceeded its total deadline")
    return remaining


def _custodian_child_ready_budget_seconds(runtime: dict[str, Any]) -> float:
    """Bound listener wait plus one synchronous select/load and exact state GET."""

    listener_budget = float(runtime["custodian_ready_timeout_seconds"])
    require(listener_budget > 0.0, "custodian listener budget is invalid")
    return listener_budget * 2.0 + CUSTODIAN_MODEL_LOAD_COMPLETION_MARGIN_SECONDS


def _custodian_parent_ready_budget_seconds(runtime: dict[str, Any]) -> float:
    return (
        _custodian_child_ready_budget_seconds(runtime)
        + CUSTODIAN_PARENT_READY_RECEIPT_MARGIN_SECONDS
    )


def contains_forbidden_engine_ranking_claim(value: Any) -> bool:
    """Reject positive engine winner/ranking claims anywhere in gateway evidence."""

    if isinstance(value, dict):
        for key, nested in value.items():
            normalized = key.casefold() if isinstance(key, str) else ""
            if (
                "engine" in normalized
                and ("winner" in normalized or "ranking" in normalized)
                and nested not in (False, None, "", [], {})
            ):
                return True
            if contains_forbidden_engine_ranking_claim(nested):
                return True
    elif isinstance(value, list):
        return any(contains_forbidden_engine_ranking_claim(item) for item in value)
    return False


def is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def positive_number(value: Any, label: str) -> float:
    receipt_require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        f"{label} is not numeric",
    )
    result = float(value)
    receipt_require(math.isfinite(result) and result > 0.0, f"{label} is invalid")
    return result


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def json_line_bytes(value: Any) -> bytes:
    return canonical_json_bytes(value) + b"\n"


REQUEST_BYTES = canonical_json_bytes(REQUEST)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_canonical(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ReceiptError(f"JSON contains duplicate key: {key}")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> Any:
    raise ReceiptError(f"JSON contains non-finite constant: {value}")


def parse_strict_json_document(raw: bytes) -> dict[str, Any]:
    """Parse one bounded strict-UTF8 JSON object, rejecting duplicates/NaN."""

    receipt_require(isinstance(raw, bytes), "JSON body is not bytes")
    receipt_require(0 < len(raw) <= MAX_RESPONSE_BYTES, "JSON body size is invalid")
    try:
        text = raw.decode("utf-8", errors="strict")
        value = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite,
        )
    except UnicodeDecodeError as error:
        raise ReceiptError(f"JSON body is not strict UTF-8: {error}") from error
    except json.JSONDecodeError as error:
        raise ReceiptError(f"JSON body is not strict JSON: {error}") from error
    receipt_require(isinstance(value, dict), "JSON body is not an object")
    return value


def parse_strict_json_line(raw: bytes) -> dict[str, Any]:
    receipt_require(
        raw.endswith(b"\n") and raw.count(b"\n") == 1 and b"\r" not in raw,
        "receipt must be one LF-terminated JSON line",
    )
    return parse_strict_json_document(raw[:-1])


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def relative_close(actual: float, expected: float, tolerance: float = 1e-5) -> bool:
    return abs(actual - expected) <= tolerance * max(abs(actual), abs(expected), 1.0)


def clock_receipt() -> dict[str, Any]:
    info = time.get_clock_info("monotonic")
    resolution_ns = max(1, math.ceil(info.resolution * 1_000_000_000))
    require(
        info.monotonic and not info.adjustable, "Python monotonic clock is unsuitable"
    )
    return {
        "clock": "monotonic",
        "clock_identity": f"Python time.monotonic_ns/{info.implementation}",
        "clock_resolution_ns": resolution_ns,
        "clock_is_monotonic": info.monotonic,
        "clock_is_adjustable": info.adjustable,
    }


def validate_frozen_gateway_contract(contract: dict[str, Any]) -> None:
    edge = contract["comparison_graph"]["edges"][EDGE_ID]
    require(edge["members"] == ["B", "G"], "gateway edge members drifted")
    runtime_custody = contract.get("runtime_custody")
    custody_object = (
        runtime_custody.get("gateway_controller_backend_model_custody_v3")
        if isinstance(runtime_custody, dict)
        else None
    )
    edge_custody = {
        field: edge.get(field) for field in GATEWAY_MODEL_CUSTODY_EDGE_FIELDS_V3
    }
    if (
        edge_custody != GATEWAY_MODEL_CUSTODY_EDGE_FIELDS_V3
        or custody_object != GATEWAY_MODEL_CUSTODY_CONTRACT_V3
    ):
        raise_preflight_blocker(
            "GATEWAY_MODEL_CUSTODY_CONTRACT_ALTERNATIVE_NOT_ACTIVATED",
            "GATEWAY_B_VS_G still requires an impossible controller/backend "
            "same-open-file-description claim or lacks the exact approved "
            "same-vnode custodian proof contract",
            {
                "observed_edge_fields": edge_custody,
                "required_edge_fields": GATEWAY_MODEL_CUSTODY_EDGE_FIELDS_V3,
                "observed_runtime_custody_object": custody_object,
                "required_runtime_custody_object": GATEWAY_MODEL_CUSTODY_CONTRACT_V3,
            },
        )
    request = contract["workload_contracts"]["GATEWAY_RAW13_FREE128_V3"]["request"]
    require(
        request["canonical_json_object"] == REQUEST, "gateway request object drifted"
    )
    require(request["size_bytes"] == REQUEST_SIZE, "gateway request size drifted")
    require(request["sha256"] == REQUEST_SHA256, "gateway request hash drifted")
    require(len(REQUEST_BYTES) == REQUEST_SIZE, "local request size drifted")
    require(sha256_bytes(REQUEST_BYTES) == REQUEST_SHA256, "local request hash drifted")
    timing = contract["timing_contract"][EDGE_ID]
    require(
        timing.get("start") == GATEWAY_TIMING_START_V3
        and all(
            timing.get(field) is expected
            for field, expected in GATEWAY_COMPLETE_WIRE_TIMING_CONTRACT_V3.items()
        ),
        "gateway complete-wire timing contract drifted",
    )
    protocol = contract["execution_protocol"][EDGE_ID]
    require(
        protocol["untimed_warmup_abstract_orders"] == ["ABBA", "BAAB"]
        and protocol["untimed_warmups_per_arm"] == 4
        and protocol["timed_macroblock_count"] == 16
        and protocol["odd_macroblock_abstract_orders"] == ["ABBA", "BAAB"]
        and protocol["even_macroblock_abstract_orders"] == ["BAAB", "ABBA"]
        and protocol["timed_samples_total"] == 128,
        "gateway schedule contract drifted",
    )
    required_gates = contract["machine_receipt_contract"][
        "required_true_gate_ids_for_GATEWAY_B_VS_G"
    ]
    require(tuple(required_gates) == GATEWAY_GATE_IDS, "gateway gate IDs drifted")


def _expand_abstract_subblock(order: str) -> list[str]:
    require(order in ("ABBA", "BAAB"), "abstract gateway subblock is invalid")
    role_to_arm = {"A": "B", "B": "G"}
    return [role_to_arm[role] for role in order]


def declared_schedule(contract: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the exact 8 warmup + 128 timed generation slots."""

    protocol = contract["execution_protocol"][EDGE_ID]
    require(
        protocol["untimed_warmup_abstract_orders"] == ["ABBA", "BAAB"],
        "warmup abstract schedule drifted",
    )
    slots: list[dict[str, Any]] = []

    def append_subblock(
        phase: str,
        abstract_order: str,
        *,
        warmup_subblock_index: int | None,
        macroblock_index: int | None,
        subblock_index: int,
    ) -> None:
        arms = _expand_abstract_subblock(abstract_order)
        for local_index, arm in enumerate(arms):
            pair_local_index = local_index // 2
            pair_arms = arms[pair_local_index * 2 : pair_local_index * 2 + 2]
            slots.append(
                {
                    "sequence_index": len(slots),
                    "phase": phase,
                    "warmup_subblock_index": warmup_subblock_index,
                    "macroblock_index": macroblock_index,
                    "subblock_index": subblock_index,
                    "slot_index_in_subblock": local_index,
                    "pair_index_in_subblock": pair_local_index,
                    "pair_order": "".join(pair_arms),
                    "abstract_subblock_order": abstract_order,
                    "arm": arm,
                }
            )

    for subblock_index, abstract in enumerate(
        protocol["untimed_warmup_abstract_orders"]
    ):
        append_subblock(
            "warmup",
            abstract,
            warmup_subblock_index=subblock_index,
            macroblock_index=None,
            subblock_index=subblock_index,
        )
    for macroblock in range(16):
        key = (
            "odd_macroblock_abstract_orders"
            if macroblock % 2 == 0
            else "even_macroblock_abstract_orders"
        )
        for subblock_index, abstract in enumerate(protocol[key]):
            append_subblock(
                "timed",
                abstract,
                warmup_subblock_index=None,
                macroblock_index=macroblock,
                subblock_index=subblock_index,
            )
    require(len(slots) == 136, "gateway generation slot count drifted")
    warmup = [slot for slot in slots if slot["phase"] == "warmup"]
    timed = [slot for slot in slots if slot["phase"] == "timed"]
    require(
        len(warmup) == 8
        and sum(slot["arm"] == "B" for slot in warmup) == 4
        and sum(slot["arm"] == "G" for slot in warmup) == 4,
        "gateway warmup arm counts drifted",
    )
    require(
        len(timed) == 128
        and sum(slot["arm"] == "B" for slot in timed) == 64
        and sum(slot["arm"] == "G" for slot in timed) == 64,
        "gateway timed arm counts drifted",
    )
    return slots


def validate_gateway_response(
    response: dict[str, Any], contract: dict[str, Any], expected_model_path: str
) -> dict[str, Any]:
    """Validate every semantic field before the client-wall end timestamp."""

    receipt_require(
        response.get("object") == "chat.completion", "response object drifted"
    )
    receipt_require(
        response.get("model") == expected_model_path, "response model drifted"
    )
    receipt_require(
        response.get("system_fingerprint") == BACKEND_BUILD_INFO,
        "response backend fingerprint drifted",
    )
    choices = response.get("choices")
    receipt_require(
        isinstance(choices, list) and len(choices) == 1, "choice count drifted"
    )
    choice = choices[0]
    receipt_require(isinstance(choice, dict), "choice is not an object")
    receipt_require(choice.get("finish_reason") == "length", "finish reason drifted")
    message = choice.get("message")
    receipt_require(isinstance(message, dict), "response message is absent")
    receipt_require(message.get("role") == "assistant", "response role drifted")
    content = message.get("content")
    receipt_require(isinstance(content, str), "response content is not a string")

    usage = response.get("usage")
    receipt_require(isinstance(usage, dict), "response usage is absent")
    usage_tuple = [
        usage.get("prompt_tokens"),
        usage.get("completion_tokens"),
        usage.get("total_tokens"),
    ]
    expected = contract["workload_contracts"]["GATEWAY_RAW13_FREE128_V3"]
    receipt_require(
        usage_tuple == expected["generation"]["usage_prompt_completion_total"],
        "response usage counts drifted",
    )
    prompt_details = usage.get("prompt_tokens_details")
    receipt_require(
        isinstance(prompt_details, dict) and prompt_details.get("cached_tokens") == 0,
        "response usage reports cached prompt tokens",
    )

    timings = response.get("timings")
    receipt_require(isinstance(timings, dict), "native timings are absent")
    native_counts = [
        timings.get("prompt_n"),
        timings.get("predicted_n"),
        timings.get("cache_n"),
    ]
    receipt_require(
        native_counts == expected["generation"]["native_prompt_predicted_cache_n"],
        "native prompt/predicted/cache counts drifted",
    )
    prompt_ms = positive_number(timings.get("prompt_ms"), "prompt_ms")
    predicted_ms = positive_number(timings.get("predicted_ms"), "predicted_ms")
    prompt_tps = positive_number(timings.get("prompt_per_second"), "prompt_per_second")
    predicted_tps = positive_number(
        timings.get("predicted_per_second"), "predicted_per_second"
    )
    receipt_require(
        relative_close(prompt_tps, 13_000.0 / prompt_ms), "prompt TPS formula drifted"
    )
    receipt_require(
        relative_close(predicted_tps, 128_000.0 / predicted_ms),
        "predicted TPS formula drifted",
    )
    receipt_require(
        relative_close(
            positive_number(timings.get("prompt_per_token_ms"), "prompt_per_token_ms"),
            prompt_ms / 13,
        ),
        "prompt per-token formula drifted",
    )
    receipt_require(
        relative_close(
            positive_number(
                timings.get("predicted_per_token_ms"), "predicted_per_token_ms"
            ),
            predicted_ms / 128,
        ),
        "predicted per-token formula drifted",
    )

    verbose = response.get("__verbose")
    receipt_require(isinstance(verbose, dict), "verbose response is absent")
    receipt_require(verbose.get("id_slot") == 0, "verbose slot drifted")
    receipt_require(
        verbose.get("tokens_predicted") == 128, "verbose predicted count drifted"
    )
    receipt_require(
        verbose.get("tokens_evaluated") == 13, "verbose evaluated count drifted"
    )
    receipt_require(
        verbose.get("tokens_cached") == EXPECTED_VERBOSE_TOKENS_CACHED,
        "verbose cache extent drifted",
    )
    receipt_require(verbose.get("stop_type") == "limit", "verbose stop type drifted")
    receipt_require(verbose.get("truncated") is False, "response was truncated")
    rendered = verbose.get("prompt")
    receipt_require(rendered == RENDERED_PROMPT, "backend rendered prompt drifted")
    generated = verbose.get("tokens")
    trajectory = expected["trajectory_admission"]
    receipt_require(
        isinstance(generated, list)
        and len(generated) == trajectory["generated_token_ids_count"]
        and all(is_int(token) and token >= 0 for token in generated),
        "raw generated token trajectory shape drifted",
    )
    generated_hash = sha256_canonical(generated)
    receipt_require(
        generated_hash == trajectory["expected_sha256"],
        "raw free128 trajectory hash drifted",
    )
    settings = verbose.get("generation_settings")
    receipt_require(isinstance(settings, dict), "generation settings are absent")
    return {
        "prompt_token_ids": copy.deepcopy(
            contract["workload_contracts"]["shared_prompt"]["token_ids"]
        ),
        "rendered_prompt": rendered,
        "rendered_prompt_sha256": sha256_bytes(rendered.encode("utf-8")),
        "generated_token_ids": copy.deepcopy(generated),
        "generated_token_ids_sha256": generated_hash,
        "content": content,
        "content_sha256": sha256_bytes(content.encode("utf-8")),
        "usage": usage_tuple,
        "usage_object": copy.deepcopy(usage),
        "generation_settings": copy.deepcopy(settings),
        "generation_settings_sha256": sha256_canonical(settings),
        "native": {
            "prompt_ms": prompt_ms,
            "predicted_ms": predicted_ms,
            "prompt_tps": prompt_tps,
            "predicted_tps": predicted_tps,
        },
    }


class PersistentHttpJsonConnection:
    """One raw HTTP/1.1 socket; reconnects and parse-excluded timing are impossible."""

    def __init__(
        self,
        base_url: str,
        label: str,
        *,
        timeout_seconds: float = 600.0,
        socket_factory: Any = socket.create_connection,
        response_factory: Any = http.client.HTTPResponse,
        clock_ns: Any = time.monotonic_ns,
    ):
        parsed = urllib.parse.urlsplit(base_url)
        require(parsed.scheme == "http", f"{label} scheme is not HTTP")
        require(parsed.hostname == "127.0.0.1", f"{label} is not loopback")
        require(parsed.port is not None, f"{label} port is absent")
        require(
            parsed.path in ("", "/") and not parsed.query,
            f"{label} base URL has a path",
        )
        self.host = parsed.hostname
        self.port = parsed.port
        self.label = label
        self.timeout_seconds = timeout_seconds
        self.socket_factory = socket_factory
        self.response_factory = response_factory
        self.clock_ns = clock_ns
        self.sock: Any | None = None
        self.baseline: dict[str, Any] | None = None
        self.request_count = 0
        self.reconnect_count = 0
        self.last_raw_response = b""

    def socket_identity(self) -> dict[str, Any]:
        require(self.sock is not None, f"{self.label} socket is absent")
        descriptor = self.sock.fileno()
        require(
            is_int(descriptor) and descriptor >= 0, f"{self.label} socket is closed"
        )
        return {
            "python_object_id": id(self.sock),
            "file_descriptor": descriptor,
            "local_address": list(self.sock.getsockname()),
            "peer_address": list(self.sock.getpeername()),
        }

    def connect(self) -> dict[str, Any]:
        require(self.sock is None, f"{self.label} connection was already opened")
        self.sock = self.socket_factory(
            (self.host, self.port), timeout=self.timeout_seconds
        )
        try:
            self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except AttributeError:
            pass
        self.baseline = self.socket_identity()
        return copy.deepcopy(self.baseline)

    def _wire_request(self, method: str, path: str, body: bytes | None) -> bytes:
        require(method in ("GET", "POST"), "HTTP method is not admitted")
        require(
            path.startswith("/") and "\r" not in path and "\n" not in path,
            "HTTP path is invalid",
        )
        headers = [
            f"{method} {path} HTTP/1.1",
            f"Host: {self.host}:{self.port}",
            "Accept: application/json",
            "Connection: keep-alive",
        ]
        if body is not None:
            headers.extend(
                [
                    "Content-Type: application/json",
                    f"Content-Length: {len(body)}",
                ]
            )
        return ("\r\n".join(headers) + "\r\n\r\n").encode("ascii") + (body or b"")

    def request_json(
        self,
        method: str,
        path: str,
        body: bytes | None,
        semantic_validator: Any,
    ) -> tuple[dict[str, Any], Any, dict[str, Any]]:
        require(callable(semantic_validator), "semantic validator is not callable")
        require(self.baseline is not None, f"{self.label} was not preconnected")
        before = self.socket_identity()
        require(before == self.baseline, f"{self.label} socket changed before request")
        wire = self._wire_request(method, path, body)
        body_bytes = body or b""
        body_offset = len(wire) - len(body_bytes)
        require(
            body_offset >= 0 and wire[body_offset:] == body_bytes,
            f"{self.label} complete request wire does not contain its exact body",
        )
        wire_sha256 = sha256_bytes(wire)
        canonical_body = (
            len(body_bytes) == REQUEST_SIZE
            and sha256_bytes(body_bytes) == REQUEST_SHA256
            and body_bytes == REQUEST_BYTES
        )
        timing_events = [COMPLETE_WIRE_TIMING_EVENT_ORDER_V3[0]]
        sendall_call_count = 0
        started_ns: int | None = None
        raw = b""
        self.last_raw_response = b""
        stage = "before-first-wire-byte"
        try:
            started_ns = self.clock_ns()
            self.sock.sendall(wire)
            sendall_call_count += 1
            timing_events.extend(COMPLETE_WIRE_TIMING_EVENT_ORDER_V3[1:3])
            stage = "response-headers"
            response = self.response_factory(self.sock)
            response.begin()
            timing_events.append(COMPLETE_WIRE_TIMING_EVENT_ORDER_V3[3])
            status = response.status
            version = response.version
            will_close = response.will_close
            headers = response.getheaders()
            stage = "full-response-body"
            try:
                raw = response.read(MAX_RESPONSE_BYTES + 1)
            except http.client.IncompleteRead as error:
                raw = bytes(error.partial)
                self.last_raw_response = raw
                raise
            timing_events.append(COMPLETE_WIRE_TIMING_EVENT_ORDER_V3[4])
            self.last_raw_response = raw
            response.close()
            require(
                len(raw) <= MAX_RESPONSE_BYTES, f"{self.label} response is oversized"
            )
            receipt_require(status == 200, f"{self.label} HTTP status is {status}")
            receipt_require(version == 11, f"{self.label} did not use HTTP/1.1")
            receipt_require(
                will_close is False, f"{self.label} response closes the connection"
            )
            header_map: dict[str, list[str]] = {}
            for name, value in headers:
                header_map.setdefault(name.lower(), []).append(value)
            content_types = header_map.get("content-type", [])
            receipt_require(
                len(content_types) == 1
                and content_types[0].lower().startswith("application/json"),
                f"{self.label} response content type drifted",
            )
            stage = "strict-json-parse"
            payload = parse_strict_json_document(raw)
            timing_events.append(COMPLETE_WIRE_TIMING_EVENT_ORDER_V3[5])
            stage = "semantic-validation"
            validated = semantic_validator(payload)
            timing_events.append(COMPLETE_WIRE_TIMING_EVENT_ORDER_V3[6])
            ended_ns = self.clock_ns()
            timing_events.append(COMPLETE_WIRE_TIMING_EVENT_ORDER_V3[7])
            stage = "socket-postvalidation"
            after = self.socket_identity()
            require(
                after == self.baseline, f"{self.label} socket reconnected or closed"
            )
        except BaseException as error:
            observation = {
                "connection": self.label,
                "stage": stage,
                "started_monotonic_ns": started_ns,
                "raw_response": raw,
                "request_body": body,
                "request_wire_size_bytes": len(wire),
                "request_wire_sha256": wire_sha256,
                "request_wire_body_offset_bytes": body_offset,
                "single_sendall_call_count": sendall_call_count,
                "timing_event_order": timing_events,
                "exception_type": type(error).__name__,
                "message": str(error),
            }
            raise RuntimeObservationError(
                f"{self.label} request failed during {stage}: {error}", observation
            ) from error
        self.request_count += 1
        timing_clock = clock_receipt()
        receipt = {
            "connection": self.label,
            "connection_generation": 1,
            "request_index_on_connection": self.request_count,
            "socket": copy.deepcopy(self.baseline),
            "socket_start_end_equal": True,
            "reconnect_count": self.reconnect_count,
            "method": method,
            "path": path,
            "request_body_size_bytes": len(body) if body is not None else 0,
            "request_body_sha256": sha256_bytes(body) if body is not None else None,
            "request_wire_size_bytes": len(wire),
            "request_wire_sha256": wire_sha256,
            "request_wire_base64": base64.b64encode(wire).decode("ascii"),
            "request_wire_body_offset_bytes": body_offset,
            "request_wire_body_size_bytes": len(body_bytes),
            "request_wire_body_sha256": sha256_bytes(body_bytes),
            "request_wire_body_equals_request_body": wire[body_offset:] == body_bytes,
            "single_sendall_call_count": sendall_call_count,
            "single_sendall_argument_size_bytes": len(wire),
            "single_sendall_argument_sha256": wire_sha256,
            "timing_event_order": timing_events,
            "status": status,
            "http_version": version,
            "response_headers": header_map,
            "response_size_bytes": len(raw),
            "response_sha256": sha256_bytes(raw),
            **timing_clock,
            "start_boundary": GATEWAY_TIMING_START_V3,
            "end_boundary": "immediately-after-full-body-strict-JSON-parse-and-semantic-validation",
            "request_serialization_before_start": True,
            "first_wire_byte_send_call_immediately_after_start": True,
            "complete_HTTP_request_wire_serialization_before_start": True,
            "single_sendall_call_for_complete_request_wire_required": (
                sendall_call_count == 1
            ),
            "canonical_383_byte_JSON_body_identical_between_B_and_G": canonical_body,
            "arm_specific_HTTP_authority_header_difference_is_inside_timed_region": True,
            "body_only_timing_allowed": False,
            "full_response_body_read_before_end": True,
            "strict_json_parse_before_end": True,
            "semantic_validation_before_end": True,
            "json_parse_excluded_from_wall": False,
            "semantic_validation_excluded_from_wall": False,
            "started_monotonic_ns": started_ns,
            "ended_monotonic_ns": ended_ns,
            "client_full_response_wall_ns": ended_ns - started_ns,
            "client_full_response_wall_ms": (ended_ns - started_ns) / 1_000_000,
        }
        return payload, validated, receipt

    def close(self) -> None:
        if self.sock is not None:
            self.sock.close()
            self.sock = None


def validate_complete_wire_timing_receipt(timing: dict[str, Any]) -> dict[str, Any]:
    try:
        wire = base64.b64decode(
            timing.get("request_wire_base64", "").encode("ascii"), validate=True
        )
    except (UnicodeEncodeError, binascii.Error, ValueError) as error:
        raise ReceiptError(
            "sample complete request wire is not strict base64"
        ) from error
    offset = timing.get("request_wire_body_offset_bytes")
    receipt_require(
        is_int(offset) and 0 < offset <= len(wire),
        "sample complete request wire body offset is invalid",
    )
    header = wire[:offset]
    body = wire[offset:]
    try:
        header_text = header.decode("ascii", errors="strict")
    except UnicodeDecodeError as error:
        raise ReceiptError("sample request wire header is not ASCII") from error
    host_lines = [
        line.removeprefix("Host: ")
        for line in header_text.split("\r\n")
        if line.startswith("Host: ")
    ]
    authority_match = (
        re.fullmatch(r"127\.0\.0\.1:([1-9][0-9]{0,4})", host_lines[0])
        if len(host_lines) == 1
        else None
    )
    authority_port = int(authority_match.group(1)) if authority_match else 0
    expected_header = (
        "POST /v1/chat/completions HTTP/1.1\r\n"
        f"Host: {host_lines[0] if len(host_lines) == 1 else ''}\r\n"
        "Accept: application/json\r\n"
        "Connection: keep-alive\r\n"
        "Content-Type: application/json\r\n"
        f"Content-Length: {REQUEST_SIZE}\r\n\r\n"
    ).encode("ascii")
    receipt_require(
        authority_match is not None
        and authority_port <= 65_535
        and header == expected_header,
        "sample complete request wire HTTP/1.1 header drifted",
    )
    wire_sha256 = sha256_bytes(wire)
    receipt_require(
        timing.get("request_wire_size_bytes") == len(wire)
        and timing.get("request_wire_sha256") == wire_sha256
        and timing.get("request_wire_body_size_bytes") == len(body) == REQUEST_SIZE
        and timing.get("request_wire_body_sha256")
        == sha256_bytes(body)
        == REQUEST_SHA256
        and body == REQUEST_BYTES
        and timing.get("request_wire_body_equals_request_body") is True,
        "sample complete request wire does not contain the canonical 383-byte body",
    )
    receipt_require(
        timing.get("single_sendall_call_count") == 1
        and timing.get("single_sendall_argument_size_bytes") == len(wire)
        and timing.get("single_sendall_argument_sha256") == wire_sha256
        and timing.get("timing_event_order") == COMPLETE_WIRE_TIMING_EVENT_ORDER_V3,
        "sample complete request wire was not sent by one ordered sendall call",
    )
    receipt_require(
        all(
            timing.get(field) is expected
            for field, expected in GATEWAY_COMPLETE_WIRE_TIMING_CONTRACT_V3.items()
        ),
        "sample complete-wire timing proof drifted",
    )
    return {
        "wire_sha256": wire_sha256,
        "body_sha256": sha256_bytes(body),
        "body_size_bytes": len(body),
        "authority": host_lines[0],
    }


def validate_sample_receipt(
    sample: dict[str, Any], slot: dict[str, Any], contract: dict[str, Any]
) -> None:
    receipt_require(isinstance(sample, dict), "sample receipt is not an object")
    receipt_require(
        set(sample)
        in (
            SAMPLE_REQUIRED_FIELDS_V3,
            SAMPLE_REQUIRED_FIELDS_V3 | SAMPLE_OPTIONAL_FIELDS_V3,
        ),
        "sample top-level schema drifted",
    )
    receipt_require(
        not contains_forbidden_engine_ranking_claim(sample),
        "sample contains a forbidden engine winner or ranking claim",
    )
    for field, expected_fields in SAMPLE_NESTED_FIELDS_V3.items():
        nested = sample.get(field)
        receipt_require(
            isinstance(nested, dict) and set(nested) == expected_fields,
            f"sample {field} schema drifted",
        )
    tokenize_admission = sample["workload"].get(
        "per_sample_tokenize_admission_outside_timed_interval"
    )
    receipt_require(
        isinstance(tokenize_admission, dict)
        and set(tokenize_admission) == SAMPLE_TOKENIZE_ADMISSION_FIELDS_V3,
        "sample tokenize-admission schema drifted",
    )
    if "segment_state_after" in sample:
        segment = sample["segment_state_after"]
        receipt_require(
            isinstance(segment, dict)
            and set(segment) == SAMPLE_SEGMENT_STATE_FIELDS_V3,
            "sample segment-state schema drifted",
        )
    receipt_require(
        sample.get("format") == SAMPLE_FORMAT, "sample format is not formal v3"
    )
    receipt_require(sample.get("schema_version") == 3, "sample schema is not v3")
    receipt_require(
        sample.get("campaign_id") == contract["campaign_id"], "sample campaign drifted"
    )
    receipt_require(
        sample.get("subcampaign_id")
        == contract["comparison_graph"]["edges"][EDGE_ID]["subcampaign_id"],
        "sample subcampaign drifted",
    )
    receipt_require(sample.get("edge_id") == EDGE_ID, "sample edge drifted")
    receipt_require(sample.get("slot") == slot, "sample slot binding drifted")
    request = sample.get("request")
    expected_request = contract["workload_contracts"]["GATEWAY_RAW13_FREE128_V3"][
        "request"
    ]
    receipt_require(
        isinstance(request, dict)
        and request.get("canonical_json_object") == REQUEST
        and request.get("canonical_utf8") == REQUEST_BYTES.decode("utf-8")
        and request.get("size_bytes") == expected_request["size_bytes"]
        and request.get("sha256") == expected_request["sha256"]
        and request.get("same_body_for_B_and_G") is True,
        "sample canonical request drifted",
    )
    cache = sample.get("cache_clear")
    receipt_require(
        isinstance(cache, dict)
        and cache.get("acknowledged") is True
        and cache.get("cleared_slots") == [0]
        and cache.get("outside_primary_timed_interval") is True,
        "sample cache clear admission drifted",
    )
    cache_response = cache.get("response")
    cache_transport = cache.get("transport")
    admitted_cache = _cache_clear_validator(cache_response)
    receipt_require(
        admitted_cache["acknowledged"] == cache["acknowledged"]
        and admitted_cache["cleared_slots"] == cache["cleared_slots"]
        and isinstance(cache_transport, dict)
        and cache_transport.get("method") == "POST"
        and cache_transport.get("path") == "/omni/cache/clear"
        and cache_transport.get("request_body_size_bytes") == 2
        and cache_transport.get("request_body_sha256") == sha256_bytes(b"{}"),
        "sample cache clear response/transport drifted",
    )
    workload = sample.get("workload")
    trajectory = contract["workload_contracts"]["GATEWAY_RAW13_FREE128_V3"][
        "trajectory_admission"
    ]
    receipt_require(isinstance(workload, dict), "sample workload is absent")
    receipt_require(
        workload.get("prompt_token_ids")
        == contract["workload_contracts"]["shared_prompt"]["token_ids"],
        "sample raw prompt IDs drifted",
    )
    generated = workload.get("generated_token_ids")
    receipt_require(
        isinstance(generated, list)
        and len(generated) == trajectory["generated_token_ids_count"]
        and all(is_int(token) and token >= 0 for token in generated)
        and workload.get("generated_token_ids_sha256")
        == sha256_canonical(generated)
        == trajectory["expected_sha256"],
        "sample raw free128 trajectory drifted",
    )
    receipt_require(workload.get("usage") == [13, 128, 141], "sample usage drifted")
    receipt_require(
        isinstance(workload.get("usage_object"), dict),
        "sample raw usage object is absent",
    )
    receipt_require(
        isinstance(workload.get("generation_settings"), dict)
        and workload.get("generation_settings_sha256")
        == sha256_canonical(workload["generation_settings"]),
        "sample generation-settings custody drifted",
    )
    response_bytes = sample.get("response_bytes")
    receipt_require(
        isinstance(response_bytes, dict)
        and response_bytes.get("encoding") == "base64"
        and isinstance(response_bytes.get("base64"), str)
        and is_int(response_bytes.get("size_bytes"))
        and response_bytes["size_bytes"] > 0
        and _valid_sha256(response_bytes.get("sha256")),
        "sample exact response bytes are absent",
    )
    try:
        raw_response = base64.b64decode(
            response_bytes["base64"].encode("ascii"), validate=True
        )
    except (UnicodeEncodeError, binascii.Error, ValueError) as error:
        raise ReceiptError("sample response base64 is invalid") from error
    receipt_require(
        len(raw_response) == response_bytes["size_bytes"]
        and sha256_bytes(raw_response) == response_bytes["sha256"],
        "sample exact response byte custody drifted",
    )
    parsed_response = parse_strict_json_document(raw_response)
    receipt_require(
        parsed_response == sample.get("response"),
        "sample parsed response differs from retained strict JSON bytes",
    )
    semantic = validate_gateway_response(parsed_response, contract, MODEL_PATH)
    receipt_require(
        workload.get("rendered_prompt") == semantic["rendered_prompt"]
        and workload.get("prompt_token_ids") == semantic["prompt_token_ids"]
        and workload.get("generated_token_ids") == semantic["generated_token_ids"]
        and workload.get("generated_token_ids_sha256")
        == semantic["generated_token_ids_sha256"]
        and workload.get("content") == semantic["content"]
        and workload.get("content_sha256") == semantic["content_sha256"]
        and workload.get("usage") == semantic["usage"]
        and workload.get("usage_object") == semantic["usage_object"]
        and workload.get("generation_settings") == semantic["generation_settings"]
        and workload.get("generation_settings_sha256")
        == semantic["generation_settings_sha256"]
        and sample.get("native_sensitivity") == semantic["native"],
        "sample response semantics differ from admitted workload/native evidence",
    )
    tokenize = workload.get("per_sample_tokenize_admission_outside_timed_interval")
    receipt_require(
        isinstance(tokenize, dict)
        and tokenize.get("token_ids") == semantic["prompt_token_ids"]
        and isinstance(tokenize.get("transport"), dict),
        "sample per-sample tokenize admission is absent",
    )
    timing = sample.get("timing")
    timing_contract = contract["timing_contract"][EDGE_ID]
    receipt_require(isinstance(timing, dict), "sample timing is absent")
    receipt_require(
        timing.get("clock") == "monotonic"
        and isinstance(timing.get("clock_identity"), str)
        and timing["clock_identity"]
        and is_int(timing.get("clock_resolution_ns"))
        and timing["clock_resolution_ns"] > 0
        and timing.get("clock_is_monotonic") is True
        and timing.get("clock_is_adjustable") is False,
        "sample clock receipt drifted",
    )
    receipt_require(
        timing.get("start_boundary")
        == timing_contract["start"]
        == GATEWAY_TIMING_START_V3
        and timing.get("end_boundary") == timing_contract["end"]
        and all(
            timing_contract.get(field) is expected
            for field, expected in GATEWAY_COMPLETE_WIRE_TIMING_CONTRACT_V3.items()
        ),
        "sample timing contract labels drifted",
    )
    receipt_require(
        timing.get("implementation_start_boundary") == GATEWAY_TIMING_START_V3
        and timing.get("implementation_end_boundary")
        == "immediately-after-full-body-strict-JSON-parse-and-semantic-validation",
        "sample implementation timing boundaries drifted",
    )
    validate_complete_wire_timing_receipt(timing)
    for field in (
        "request_serialization_before_start",
        "first_wire_byte_send_call_immediately_after_start",
        "full_response_body_read_before_end",
        "strict_json_parse_before_end",
        "semantic_validation_before_end",
    ):
        receipt_require(
            timing.get(field) is True, f"sample timing proof failed: {field}"
        )
    receipt_require(
        timing.get("json_parse_excluded_from_wall") is False
        and timing.get("semantic_validation_excluded_from_wall") is False,
        "sample excludes parse or semantic validation from client wall",
    )
    started = timing.get("started_monotonic_ns")
    ended = timing.get("ended_monotonic_ns")
    wall_ns = timing.get("client_full_response_wall_ns")
    wall = positive_number(timing.get("client_full_response_wall_ms"), "client wall ms")
    receipt_require(
        is_int(started)
        and is_int(ended)
        and ended > started
        and wall_ns == ended - started
        and relative_close(wall, (ended - started) / 1_000_000, 1e-9),
        "sample wall arithmetic drifted",
    )
    connection = sample.get("connection")
    schedule = declared_schedule(contract)
    expected_request_index = (
        sum(
            prior["arm"] == slot["arm"]
            for prior in schedule
            if prior["sequence_index"] < slot["sequence_index"]
        )
        + 2
    )
    receipt_require(
        isinstance(connection, dict)
        and connection.get("connection_generation") == 1
        and connection.get("request_index_on_connection") == expected_request_index
        and isinstance(connection.get("socket"), dict)
        and connection.get("socket_start_end_equal") is True
        and connection.get("reconnect_count") == 0,
        "sample persistent connection custody drifted",
    )


def validate_pair_equal(left: dict[str, Any], right: dict[str, Any]) -> None:
    by_arm = {left["slot"]["arm"]: left, right["slot"]["arm"]: right}
    receipt_require(set(by_arm) == {"B", "G"}, "adjacent pair arms are not B/G")
    b = by_arm["B"]["workload"]
    g = by_arm["G"]["workload"]
    for field in (
        "prompt_token_ids",
        "generated_token_ids",
        "generated_token_ids_sha256",
        "content",
        "usage",
        "usage_object",
        "generation_settings",
        "generation_settings_sha256",
    ):
        receipt_require(b[field] == g[field], f"paired B/G {field} drifted")
    b_wire = validate_complete_wire_timing_receipt(by_arm["B"]["timing"])
    g_wire = validate_complete_wire_timing_receipt(by_arm["G"]["timing"])
    receipt_require(
        b_wire["body_size_bytes"] == g_wire["body_size_bytes"] == REQUEST_SIZE
        and b_wire["body_sha256"] == g_wire["body_sha256"] == REQUEST_SHA256
        and b_wire["authority"] != g_wire["authority"],
        "paired B/G wires lack distinct authorities with one identical canonical body",
    )


def _mean(values: list[float], label: str) -> float:
    require(
        values and all(math.isfinite(value) for value in values), f"{label} is invalid"
    )
    return sum(values) / len(values)


def _population_sd(values: list[float], label: str) -> float:
    mean = _mean(values, label)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))


def _population_cv(values: list[float], label: str) -> float:
    mean = _mean(values, label)
    require(mean > 0, f"{label} mean is not positive")
    return _population_sd(values, label) / mean


def _summary(values: list[float], label: str) -> dict[str, Any]:
    return {
        "count": len(values),
        "mean": _mean(values, label),
        "population_sd": _population_sd(values, label),
        "population_cv": _population_cv(values, label),
        "min": min(values),
        "max": max(values),
        "samples": values,
    }


def _t_interval(values: list[float], label: str) -> dict[str, Any]:
    require(len(values) == 16, f"{label} must contain sixteen block values")
    mean = _mean(values, label)
    sample_sd = math.sqrt(sum((value - mean) ** 2 for value in values) / 15)
    standard_error = sample_sd / 4.0
    half_width = T_CRITICAL_DF15_975 * standard_error
    return {
        "block_values": values,
        "mean": mean,
        "sample_sd": sample_sd,
        "standard_error": standard_error,
        "degrees_of_freedom": 15,
        "t_critical_0_975": T_CRITICAL_DF15_975,
        "ci95_lower": mean - half_width,
        "ci95_upper": mean + half_width,
        "ci95_half_width": half_width,
    }


def compute_gateway_statistics(
    samples: list[dict[str, Any]], contract: dict[str, Any]
) -> dict[str, Any]:
    """Compute the frozen paired-delta/log-ratio estimator and four-state decision."""

    schedule = declared_schedule(contract)
    timed_slots = [slot for slot in schedule if slot["phase"] == "timed"]
    by_sequence: dict[int, dict[str, Any]] = {}
    for sample in samples:
        slot = sample.get("slot")
        require(isinstance(slot, dict), "statistics sample slot is absent")
        sequence = slot.get("sequence_index")
        require(
            is_int(sequence) and sequence not in by_sequence,
            "sample sequence duplicated",
        )
        by_sequence[sequence] = sample
    require(
        len(samples) == 128
        and all(slot["sequence_index"] in by_sequence for slot in timed_slots),
        "statistics does not contain the exact 128 timed samples",
    )
    ordered = [by_sequence[slot["sequence_index"]] for slot in timed_slots]
    b_walls: list[float] = []
    g_walls: list[float] = []
    b_native: list[float] = []
    g_native: list[float] = []
    deltas: list[float] = []
    log_ratios: list[float] = []
    strata: dict[str, list[float]] = {"BG": [], "GB": []}
    by_block_delta: list[list[float]] = [[] for _ in range(16)]
    by_block_log: list[list[float]] = [[] for _ in range(16)]
    for pair_start in range(0, 128, 2):
        pair = ordered[pair_start : pair_start + 2]
        validate_pair_equal(pair[0], pair[1])
        by_arm = {sample["slot"]["arm"]: sample for sample in pair}
        b_wall = positive_number(
            by_arm["B"]["timing"]["client_full_response_wall_ms"], "B wall"
        )
        g_wall = positive_number(
            by_arm["G"]["timing"]["client_full_response_wall_ms"], "G wall"
        )
        b_predicted = positive_number(
            by_arm["B"]["native_sensitivity"]["predicted_ms"], "B native predicted"
        )
        g_predicted = positive_number(
            by_arm["G"]["native_sensitivity"]["predicted_ms"], "G native predicted"
        )
        delta = g_wall - b_wall
        log_ratio = math.log(g_wall / b_wall)
        macroblock = pair[0]["slot"]["macroblock_index"]
        require(
            pair[1]["slot"]["macroblock_index"] == macroblock
            and is_int(macroblock)
            and 0 <= macroblock < 16,
            "pair macroblock binding drifted",
        )
        pair_order = pair[0]["slot"]["pair_order"]
        require(
            pair[1]["slot"]["pair_order"] == pair_order, "pair order binding drifted"
        )
        b_walls.append(b_wall)
        g_walls.append(g_wall)
        b_native.append(b_predicted)
        g_native.append(g_predicted)
        deltas.append(delta)
        log_ratios.append(log_ratio)
        strata[pair_order].append(delta)
        by_block_delta[macroblock].append(delta)
        by_block_log[macroblock].append(log_ratio)
    require(
        all(len(block) == 4 for block in by_block_delta),
        "macroblock pair count drifted",
    )
    block_delta = [_mean(block, "macroblock deltas") for block in by_block_delta]
    block_log = [_mean(block, "macroblock log ratios") for block in by_block_log]
    delta_interval = _t_interval(block_delta, "block deltas")
    log_interval = _t_interval(block_log, "block log ratios")
    ratio = {
        **log_interval,
        "point_ratio": math.exp(log_interval["mean"]),
        "ci95_ratio_lower": math.exp(log_interval["ci95_lower"]),
        "ci95_ratio_upper": math.exp(log_interval["ci95_upper"]),
    }
    b_stats = _summary(b_walls, "B wall")
    g_stats = _summary(g_walls, "G wall")
    b_native_stats = _summary(b_native, "B native predicted")
    g_native_stats = _summary(g_native, "G native predicted")
    pooled_wall = _mean(b_walls + g_walls, "pooled wall")
    order_means = {
        order: _mean(values, f"{order} strata") for order, values in strata.items()
    }
    order_difference = abs(order_means["BG"] - order_means["GB"])
    front_back_difference = abs(
        _mean(block_delta[:8], "first eight blocks")
        - _mean(block_delta[8:], "last eight blocks")
    )
    dynamic_threshold = max(2.0, 0.002 * pooled_wall)
    limits = contract["statistics_and_decisions"][EDGE_ID]["stability_gates"]
    stability = {
        "B_wall_population_cv": b_stats["population_cv"]
        <= limits["B_wall_population_cv_max"],
        "G_wall_population_cv": g_stats["population_cv"]
        <= limits["G_wall_population_cv_max"],
        "B_native_predicted_ms_population_cv": b_native_stats["population_cv"]
        <= limits["B_native_predicted_ms_population_cv_max"],
        "G_native_predicted_ms_population_cv": g_native_stats["population_cv"]
        <= limits["G_native_predicted_ms_population_cv_max"],
        "population_sd_pair_delta_over_pooled_wall": _population_sd(
            deltas, "pair deltas"
        )
        / pooled_wall
        <= limits["population_sd_pair_delta_over_pooled_wall_max"],
        "absolute_order_stratum_mean_difference": order_difference <= dynamic_threshold,
        "absolute_first8_last8_block_mean_difference": front_back_difference
        <= dynamic_threshold,
        "ci95_delta_half_width": delta_interval["ci95_half_width"]
        <= limits["ci95_delta_half_width_max_ms"],
    }
    all_stable = all(stability.values())
    equivalence = (
        all_stable
        and delta_interval["ci95_lower"] >= -5.0
        and delta_interval["ci95_upper"] <= 5.0
        and ratio["ci95_ratio_lower"] >= 0.995
        and ratio["ci95_ratio_upper"] <= 1.005
    )
    positive = (
        all_stable
        and not equivalence
        and delta_interval["ci95_lower"] > 0.0
        and ratio["ci95_ratio_lower"] > 1.0
        and order_means["BG"] > 0.0
        and order_means["GB"] > 0.0
    )
    if not all_stable:
        decision = "UNINTERPRETABLE"
    elif equivalence:
        decision = "PRACTICALLY_EQUIVALENT_GATEWAY_PATH"
    elif positive:
        decision = "POSITIVE_GATEWAY_PATH_OVERHEAD"
    else:
        decision = "INCONCLUSIVE"
    return {
        "primary_observation": "paired G_minus_B client_full_response_wall_ms",
        "timed_samples": 128,
        "timed_samples_per_arm": {"B": 64, "G": 64},
        "paired_observations": 64,
        "macroblock_values": 16,
        "primary_delta_ms": delta_interval,
        "secondary_log_ratio": ratio,
        "B_wall_ms": b_stats,
        "G_wall_ms": g_stats,
        "B_native_predicted_ms": b_native_stats,
        "G_native_predicted_ms": g_native_stats,
        "pair_deltas_ms": deltas,
        "pair_log_ratios": log_ratios,
        "order_strata": {"samples": strata, "means": order_means},
        "absolute_order_stratum_mean_difference_ms": order_difference,
        "absolute_first8_last8_block_mean_difference_ms": front_back_difference,
        "dynamic_stability_threshold_ms": dynamic_threshold,
        "stability_gates": stability,
        "all_stability_gates_passed": all_stable,
        "decision": decision,
        "engine_winner_or_ranking_claim_allowed": False,
    }


def require_same_backend_identity(start: dict[str, Any], end: dict[str, Any]) -> None:
    for field in (
        "pid",
        "process_start_identity",
        "argv_sha256",
        "environment_sha256",
        "loaded_model_fd",
        "runtime_closure_sha256",
    ):
        require(start.get(field) == end.get(field), f"backend {field} changed")


def _safe_observation(value: Any) -> Any:
    if isinstance(value, bytes):
        return {
            "encoding": "base64",
            "size_bytes": len(value),
            "sha256": sha256_bytes(value),
            "data": base64.b64encode(value).decode("ascii"),
        }
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, list):
        return [_safe_observation(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _safe_observation(item) for key, item in value.items()}
    return {"python_type": type(value).__name__, "repr": repr(value)}


def _gateway_gates(admitted: bool, stable: bool) -> dict[str, bool]:
    result = {gate: admitted for gate in GATEWAY_GATE_IDS}
    result["GATEWAY_STABILITY"] = admitted and stable
    return result


def execute_formal_schedule(
    contract: dict[str, Any],
    plan: dict[str, Any],
    binding_evidence: dict[str, Any],
    *,
    sample_collector: Any,
    postflight_collector: Any,
    before_first_slot: Any | None = None,
) -> dict[str, Any]:
    """Run the fixed schedule once, replacing only the crash-safe raw receipt."""

    require(callable(sample_collector), "sample collector is not callable")
    require(callable(postflight_collector), "postflight collector is not callable")
    schedule = declared_schedule(contract)
    raw_path = Path(plan["raw_output_path"])
    slots = [
        {
            "slot": copy.deepcopy(slot),
            "status": "unattempted",
            "attempt_count": 0,
            "receipt_sha256": None,
            "failure_index": None,
        }
        for slot in schedule
    ]
    record: dict[str, Any] = {
        "format": RAW_FORMAT,
        "schema_version": 3,
        "campaign_id": contract["campaign_id"],
        "subcampaign_id": contract["comparison_graph"]["edges"][EDGE_ID][
            "subcampaign_id"
        ],
        "edge_id": EDGE_ID,
        "campaign_consumed": True,
        "status": "RUNNING",
        "raw_output_path": str(raw_path),
        "contract_binding": copy.deepcopy(binding_evidence.get("contract_binding", {})),
        "git_custody": copy.deepcopy(binding_evidence.get("git_custody", {})),
        "host_custody": copy.deepcopy(binding_evidence.get("host_custody", {})),
        "artifact_custody": copy.deepcopy(binding_evidence.get("artifact_custody", {})),
        "parity_admission": copy.deepcopy(binding_evidence.get("parity_admission", {})),
        "schedule_receipt": {
            "process_state": "one-resident-backend-and-gateway-for-entire-campaign",
            "client_connections": "one warmed persistent HTTP/1.1 connection per arm",
            "warmup_abstract_orders": ["ABBA", "BAAB"],
            "timed_macroblocks": 16,
            "slots": slots,
            "attempted_count": 0,
            "accepted_count": 0,
            "failed_count": 0,
            "remaining_unattempted_count": 136,
            "stopped_at_first_failure": False,
            "retry_replacement_reordering_outlier_removal_or_extension_performed": False,
        },
        "samples": [],
        "statistics": None,
        "gates": _gateway_gates(False, False),
        "decision": {"label": "UNINTERPRETABLE", "formal_summary_allowed": False},
        "failures": [],
    }
    _SHARED.atomic_create_json(raw_path, record)
    stopped = False
    if before_first_slot is not None:
        try:
            before_first_slot()
        except BaseException as error:
            record["failures"].append(
                {
                    "stage": "before-first-warmup",
                    "sequence_index": None,
                    "exception_type": type(error).__name__,
                    "message": str(error),
                    "observation": {},
                    "remaining_slots_marked_unattempted": True,
                }
            )
            record["status"] = "CONSUMED_FIRST_POST_MARKER_FAILURE"
            record["schedule_receipt"]["stopped_at_first_failure"] = True
            _SHARED.atomic_replace_json(raw_path, record)
            stopped = True
    for slot, slot_state in zip(schedule, slots):
        if stopped:
            break
        try:
            slot_state["status"] = "attempting"
            slot_state["attempt_count"] = 1
            record["schedule_receipt"]["attempted_count"] += 1
            record["schedule_receipt"]["remaining_unattempted_count"] -= 1
            _SHARED.atomic_replace_json(raw_path, record)
            sample = sample_collector(copy.deepcopy(slot))
            validate_sample_receipt(sample, slot, contract)
            slot_state["status"] = "accepted"
            slot_state["receipt_sha256"] = sha256_canonical(sample)
            record["samples"].append(sample)
            record["schedule_receipt"]["accepted_count"] += 1
            _SHARED.atomic_replace_json(raw_path, record)
        except BaseException as error:
            slot_state["status"] = "failed"
            slot_state["failure_index"] = len(record["failures"])
            observation = (
                error.observation if isinstance(error, RuntimeObservationError) else {}
            )
            record["failures"].append(
                {
                    "stage": "sample",
                    "sequence_index": slot["sequence_index"],
                    "arm": slot["arm"],
                    "exception_type": type(error).__name__,
                    "message": str(error),
                    "observation": _safe_observation(observation),
                    "remaining_slots_marked_unattempted": True,
                    "failed_observation_retained": True,
                }
            )
            record["schedule_receipt"]["failed_count"] += 1
            record["schedule_receipt"]["stopped_at_first_failure"] = True
            record["status"] = "CONSUMED_FIRST_POST_MARKER_FAILURE"
            _SHARED.atomic_replace_json(raw_path, record)
            stopped = True
    try:
        postflight = postflight_collector()
        require(isinstance(postflight, dict), "postflight receipt is absent")
        require(postflight.get("passed") is True, "postflight admission failed")
        record["postflight"] = copy.deepcopy(postflight)
        if "continuous_host" in postflight:
            record["host_custody"]["continuous"] = copy.deepcopy(
                postflight["continuous_host"]
            )
        if "host_postflight" in postflight:
            record["host_custody"]["postflight"] = copy.deepcopy(
                postflight["host_postflight"]
            )
        if "artifact_custody_end" in postflight:
            record["artifact_custody_end"] = copy.deepcopy(
                postflight["artifact_custody_end"]
            )
    except BaseException as error:
        if not stopped:
            record["failures"].append(
                {
                    "stage": "postflight",
                    "sequence_index": None,
                    "exception_type": type(error).__name__,
                    "message": str(error),
                    "observation": {},
                    "remaining_slots_marked_unattempted": True,
                }
            )
            record["status"] = "CONSUMED_FIRST_POST_MARKER_FAILURE"
            record["schedule_receipt"]["stopped_at_first_failure"] = True
            stopped = True
    if not stopped:
        try:
            timed = [
                sample
                for sample in record["samples"]
                if sample["slot"]["phase"] == "timed"
            ]
            statistics = compute_gateway_statistics(timed, contract)
            stable = statistics["all_stability_gates_passed"]
            record["statistics"] = statistics
            record["gates"] = _gateway_gates(True, stable)
            record["decision"] = {
                "label": statistics["decision"],
                "formal_summary_allowed": stable,
                "engine_winner_or_ranking_claim_allowed": False,
            }
            record["status"] = "FORMAL_COMPLETE" if stable else "FORMAL_UNINTERPRETABLE"
        except BaseException as error:
            record["failures"].append(
                {
                    "stage": "statistics",
                    "sequence_index": None,
                    "exception_type": type(error).__name__,
                    "message": str(error),
                    "observation": {},
                    "remaining_slots_marked_unattempted": True,
                }
            )
            record["status"] = "CONSUMED_FIRST_POST_MARKER_FAILURE"
            record["schedule_receipt"]["stopped_at_first_failure"] = True
    _SHARED.atomic_replace_json(raw_path, record)
    return record


def prepare_marker_after_preflight(path: Path | str, preflight: Any) -> dict[str, Any]:
    require(callable(preflight), "preflight is not callable")
    marker = preflight()
    require(isinstance(marker, dict), "preflight marker is not an object")
    require(marker.get("format") == MARKER_FORMAT, "marker format drifted")
    require(marker.get("schema_version") == 3, "marker schema drifted")
    admission = marker.get("pre_marker_admission")
    require(
        isinstance(admission, dict) and admission.get("all_passed") is True,
        "not every pre-marker gateway admission passed",
    )
    _SHARED.atomic_create_json(path, marker)
    return marker


def _fixture_gateway_response(predicted_ms: float) -> dict[str, Any]:
    generated = list(range(128))
    return {
        "object": "chat.completion",
        "model": MODEL_PATH,
        "system_fingerprint": BACKEND_BUILD_INFO,
        "choices": [
            {
                "finish_reason": "length",
                "message": {"role": "assistant", "content": "fixture"},
            }
        ],
        "usage": {
            "prompt_tokens": 13,
            "completion_tokens": 128,
            "total_tokens": 141,
            "prompt_tokens_details": {"cached_tokens": 0},
        },
        "timings": {
            "prompt_n": 13,
            "predicted_n": 128,
            "cache_n": 0,
            "prompt_ms": 10.0,
            "predicted_ms": predicted_ms,
            "prompt_per_second": 1300.0,
            "predicted_per_second": 128_000.0 / predicted_ms,
            "prompt_per_token_ms": 10.0 / 13.0,
            "predicted_per_token_ms": predicted_ms / 128.0,
        },
        "__verbose": {
            "id_slot": 0,
            "tokens_predicted": 128,
            "tokens_evaluated": 13,
            "tokens_cached": EXPECTED_VERBOSE_TOKENS_CACHED,
            "stop_type": "limit",
            "truncated": False,
            "prompt": RENDERED_PROMPT,
            "tokens": generated,
            "generation_settings": {"seed": 0, "temperature": 0},
        },
    }


def _fixture_generation_wire_timing(arm: str) -> dict[str, Any]:
    port = 19_001 if arm == "B" else 19_000
    header = (
        "POST /v1/chat/completions HTTP/1.1\r\n"
        f"Host: 127.0.0.1:{port}\r\n"
        "Accept: application/json\r\n"
        "Connection: keep-alive\r\n"
        "Content-Type: application/json\r\n"
        f"Content-Length: {REQUEST_SIZE}\r\n\r\n"
    ).encode("ascii")
    wire = header + REQUEST_BYTES
    wire_sha256 = sha256_bytes(wire)
    return {
        "request_wire_size_bytes": len(wire),
        "request_wire_sha256": wire_sha256,
        "request_wire_base64": base64.b64encode(wire).decode("ascii"),
        "request_wire_body_offset_bytes": len(header),
        "request_wire_body_size_bytes": REQUEST_SIZE,
        "request_wire_body_sha256": REQUEST_SHA256,
        "request_wire_body_equals_request_body": True,
        "single_sendall_call_count": 1,
        "single_sendall_argument_size_bytes": len(wire),
        "single_sendall_argument_sha256": wire_sha256,
        "timing_event_order": list(COMPLETE_WIRE_TIMING_EVENT_ORDER_V3),
        **GATEWAY_COMPLETE_WIRE_TIMING_CONTRACT_V3,
    }


def _fixture_sample(
    contract: dict[str, Any], slot: dict[str, Any], wall_ms: float
) -> dict[str, Any]:
    generated = list(range(128))
    trajectory_hash = sha256_canonical(generated)
    predicted_ms = wall_ms - 1.0
    response = _fixture_gateway_response(predicted_ms)
    response_raw = canonical_json_bytes(response)
    request_index = (
        sum(
            prior["arm"] == slot["arm"]
            for prior in declared_schedule(contract)
            if prior["sequence_index"] < slot["sequence_index"]
        )
        + 2
    )
    return {
        "format": SAMPLE_FORMAT,
        "schema_version": 3,
        "campaign_id": contract["campaign_id"],
        "subcampaign_id": contract["comparison_graph"]["edges"][EDGE_ID][
            "subcampaign_id"
        ],
        "edge_id": EDGE_ID,
        "slot": copy.deepcopy(slot),
        "request": {
            "canonical_json_object": REQUEST,
            "canonical_utf8": REQUEST_BYTES.decode("utf-8"),
            "size_bytes": REQUEST_SIZE,
            "sha256": REQUEST_SHA256,
            "same_body_for_B_and_G": True,
        },
        "cache_clear": {
            "acknowledged": True,
            "cleared_slots": [0],
            "outside_primary_timed_interval": True,
            "response": {
                "ok": True,
                "cache_policy": "cleared_each_run",
                "cleared_slots": [0],
            },
            "transport": {
                "method": "POST",
                "path": "/omni/cache/clear",
                "request_body_size_bytes": 2,
                "request_body_sha256": sha256_bytes(b"{}"),
            },
        },
        "workload": {
            "rendered_prompt": RENDERED_PROMPT,
            "prompt_token_ids": PROMPT_TOKEN_IDS,
            "generated_token_ids": generated,
            "generated_token_ids_sha256": trajectory_hash,
            "content": "fixture",
            "content_sha256": sha256_bytes(b"fixture"),
            "usage": [13, 128, 141],
            "usage_object": {
                "prompt_tokens": 13,
                "completion_tokens": 128,
                "total_tokens": 141,
                "prompt_tokens_details": {"cached_tokens": 0},
            },
            "generation_settings": {"seed": 0, "temperature": 0},
            "generation_settings_sha256": sha256_canonical(
                {"seed": 0, "temperature": 0}
            ),
            "per_sample_tokenize_admission_outside_timed_interval": {
                "token_ids": PROMPT_TOKEN_IDS,
                "transport": {},
            },
        },
        "timing": {
            **clock_receipt(),
            "start_boundary": contract["timing_contract"][EDGE_ID]["start"],
            "end_boundary": contract["timing_contract"][EDGE_ID]["end"],
            "implementation_start_boundary": GATEWAY_TIMING_START_V3,
            "implementation_end_boundary": "immediately-after-full-body-strict-JSON-parse-and-semantic-validation",
            "request_serialization_before_start": True,
            "first_wire_byte_send_call_immediately_after_start": True,
            **_fixture_generation_wire_timing(slot["arm"]),
            "full_response_body_read_before_end": True,
            "strict_json_parse_before_end": True,
            "semantic_validation_before_end": True,
            "json_parse_excluded_from_wall": False,
            "semantic_validation_excluded_from_wall": False,
            "started_monotonic_ns": 1_000_000,
            "ended_monotonic_ns": 1_000_000 + round(wall_ms * 1_000_000),
            "client_full_response_wall_ns": round(wall_ms * 1_000_000),
            "client_full_response_wall_ms": wall_ms,
        },
        "native_sensitivity": {
            "prompt_ms": 10.0,
            "predicted_ms": predicted_ms,
            "prompt_tps": 1300.0,
            "predicted_tps": 128_000.0 / predicted_ms,
        },
        "connection": {
            "connection_generation": 1,
            "request_index_on_connection": request_index,
            "socket": {"fixture": slot["arm"]},
            "socket_start_end_equal": True,
            "reconnect_count": 0,
        },
        "response": response,
        "response_bytes": {
            "encoding": "base64",
            "base64": base64.b64encode(response_raw).decode("ascii"),
            "size_bytes": len(response_raw),
            "sha256": sha256_bytes(response_raw),
        },
    }


def _fixture_contract() -> dict[str, Any]:
    generated = list(range(128))
    return {
        "campaign_id": "fixture-campaign",
        "comparison_graph": {
            "edges": {EDGE_ID: {"subcampaign_id": "fixture-subcampaign"}}
        },
        "workload_contracts": {
            "shared_prompt": {"token_ids": PROMPT_TOKEN_IDS},
            "GATEWAY_RAW13_FREE128_V3": {
                "request": {
                    "canonical_json_object": REQUEST,
                    "size_bytes": REQUEST_SIZE,
                    "sha256": REQUEST_SHA256,
                },
                "trajectory_admission": {
                    "generated_token_ids_count": 128,
                    "expected_sha256": sha256_canonical(generated),
                },
                "generation": {
                    "usage_prompt_completion_total": [13, 128, 141],
                    "native_prompt_predicted_cache_n": [13, 128, 0],
                },
            },
        },
        "execution_protocol": {
            EDGE_ID: {
                "untimed_warmup_abstract_orders": ["ABBA", "BAAB"],
                "odd_macroblock_abstract_orders": ["ABBA", "BAAB"],
                "even_macroblock_abstract_orders": ["BAAB", "ABBA"],
            }
        },
        "timing_contract": {
            EDGE_ID: {
                "start": GATEWAY_TIMING_START_V3,
                "end": "after-reading-the-full-response-body-and-validating-and-parsing-the-complete-JSON-response",
                **GATEWAY_COMPLETE_WIRE_TIMING_CONTRACT_V3,
            }
        },
        "statistics_and_decisions": {
            EDGE_ID: {
                "stability_gates": {
                    "B_wall_population_cv_max": 0.01,
                    "G_wall_population_cv_max": 0.01,
                    "B_native_predicted_ms_population_cv_max": 0.01,
                    "G_native_predicted_ms_population_cv_max": 0.01,
                    "population_sd_pair_delta_over_pooled_wall_max": 0.01,
                    "ci95_delta_half_width_max_ms": 2.0,
                }
            }
        },
    }


def run_fixture_self_test() -> dict[str, Any]:
    """Exercise success/failure state machines without sockets, Git, or models."""

    contract = _fixture_contract()
    schedule = declared_schedule(contract)
    success_calls: list[int] = []
    failure_calls: list[int] = []
    with __import__("tempfile").TemporaryDirectory(
        prefix="apxinf-gateway-formal-v3-self-test-"
    ) as directory:
        root = Path(directory)

        def success_collector(slot: dict[str, Any]) -> dict[str, Any]:
            success_calls.append(slot["sequence_index"])
            wall = 1000.0 if slot["arm"] == "B" else 1001.0
            return _fixture_sample(contract, slot, wall)

        success = execute_formal_schedule(
            contract,
            {"raw_output_path": str(root / "success.json")},
            {"fixture": True},
            sample_collector=success_collector,
            postflight_collector=lambda: {"passed": True},
        )

        def failure_collector(slot: dict[str, Any]) -> dict[str, Any]:
            failure_calls.append(slot["sequence_index"])
            if slot["sequence_index"] == 3:
                raise RuntimeObservationError("fixture failure", {"raw": b"x"})
            wall = 1000.0 if slot["arm"] == "B" else 1001.0
            return _fixture_sample(contract, slot, wall)

        failed = execute_formal_schedule(
            contract,
            {"raw_output_path": str(root / "failed.json")},
            {"fixture": True},
            sample_collector=failure_collector,
            postflight_collector=lambda: {"passed": True},
        )
        require(
            parse_strict_json_line((root / "success.json").read_bytes()) == success
            and parse_strict_json_line((root / "failed.json").read_bytes()) == failed,
            "fixture raw receipts did not round-trip",
        )
    require(len(schedule) == 136, "fixture schedule drifted")
    require(success_calls == list(range(136)), "fixture complete path drifted")
    require(failure_calls == [0, 1, 2, 3], "fixture failure did not stop immediately")
    validate_controller_launch_sequence(
        {
            "controller_fd_open_started_monotonic_ns": 1,
            "controller_fd_custody_complete_monotonic_ns": 2,
            "gateway_spawn_invocation_monotonic_ns": 3,
            "gateway_kernel_identity_observed_monotonic_ns": 4,
            "backend_kernel_identity_observed_monotonic_ns": 5,
        }
    )
    require(success["status"] == "FORMAL_COMPLETE", "fixture success state drifted")
    require(
        failed["status"] == "CONSUMED_FIRST_POST_MARKER_FAILURE",
        "fixture failure state drifted",
    )
    return {
        "format": "apxinf-qwen35-omniinfer-gateway-formal-v3-self-test",
        "schema_version": 3,
        "passed": True,
        "complete_path_generation_invocations": len(success_calls),
        "failure_path_generation_invocations": len(failure_calls),
        "network_used": False,
        "model_process_used": False,
        "marker_created": False,
        "custodian_daemon_started": False,
        "controller_model_fd_opened": False,
        "campaign_directory_tree_initialized": False,
        "custodian_lifecycle_fixture_validated": True,
    }


_HARDENED_COMMAND_ENV = {
    "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
    "LC_ALL": "C",
    "LANG": "C",
    "TZ": "UTC",
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_CONFIG_GLOBAL": "/dev/null",
    "GIT_CONFIG_SYSTEM": "/dev/null",
    "GIT_TERMINAL_PROMPT": "0",
}

_CUSTODIAN_ENV = {
    "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
    "LC_ALL": "C",
    "LANG": "C",
    "TZ": "UTC",
    "PYTHONHASHSEED": "0",
}


def hardened_command_runner(
    argv: list[str],
    cwd: Path | str,
    timeout_seconds: float,
    approved_environment: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Run custody commands without ambient Git/configuration environment."""

    require(
        isinstance(argv, list)
        and argv
        and all(isinstance(item, str) and item for item in argv),
        "command argv is invalid",
    )
    if approved_environment is None:
        command_environment = dict(_HARDENED_COMMAND_ENV)
    else:
        require(
            argv[0] == "/usr/bin/git"
            and approved_environment == _SHARED.git_custody_environment(),
            "custody caller supplied an unapproved command environment",
        )
        command_environment = dict(approved_environment)
    completed = subprocess.run(
        argv,
        cwd=str(cwd),
        env=command_environment,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout_seconds,
        check=False,
    )
    return {
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def _run_checked(
    argv: list[str], cwd: Path | str = Path("/"), timeout_seconds: float = 30.0
) -> bytes:
    result = hardened_command_runner(argv, cwd, timeout_seconds)
    require(result["returncode"] == 0, f"custody command failed: {argv}")
    require(result["stderr"] == b"", f"custody command wrote stderr: {argv}")
    return result["stdout"]


def _read_regular_no_follow(
    path_value: Path | str, maximum: int
) -> tuple[bytes, dict[str, Any]]:
    path = Path(path_value)
    require(path.is_absolute(), "custody path must be absolute")
    require(os.path.normpath(str(path)) == str(path), "custody path is not normalized")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(str(path), flags)
    except OSError as error:
        raise CampaignError(f"O_NOFOLLOW open failed for {path}: {error}") from error
    try:
        before = os.fstat(descriptor)
        require(stat.S_ISREG(before.st_mode), f"custody path is not regular: {path}")
        chunks: list[bytes] = []
        size = 0
        while True:
            chunk = os.read(descriptor, min(1024 * 1024, maximum + 1 - size))
            if not chunk:
                break
            chunks.append(chunk)
            size += len(chunk)
            require(size <= maximum, f"custody file is oversized: {path}")
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity = ("st_dev", "st_ino", "st_mode", "st_size", "st_ctime_ns")
    require(
        all(getattr(before, field) == getattr(after, field) for field in identity),
        f"custody file changed while read: {path}",
    )
    raw = b"".join(chunks)
    require(len(raw) == before.st_size, f"custody file read was incomplete: {path}")
    return raw, {
        "absolute_path": str(path),
        "device": before.st_dev,
        "inode": before.st_ino,
        "mode": before.st_mode,
        "size_bytes": before.st_size,
        "ctime_ns": before.st_ctime_ns,
        "sha256": sha256_bytes(raw),
        "open_flags": ["O_RDONLY", "O_CLOEXEC", "O_NOFOLLOW"],
        "identity_before_after_equal": True,
    }


class ControllerModelFd:
    """One driver-owned model FD held from before runtime spawn through cleanup."""

    def __init__(
        self,
        path_value: Path | str,
        expected_size: int,
        expected_sha256: str,
        *,
        clock_ns: Any = time.monotonic_ns,
    ):
        path = Path(path_value)
        require(path.is_absolute(), "controller model path must be absolute")
        require(
            os.path.normpath(str(path)) == str(path),
            "controller model path is not normalized",
        )
        require(
            hasattr(os, "O_NOFOLLOW") and hasattr(os, "O_CLOEXEC"),
            "platform lacks required O_NOFOLLOW or O_CLOEXEC model-open flags",
        )
        require(
            is_int(expected_size) and expected_size > 0,
            "controller expected model size is invalid",
        )
        require(_valid_sha256(expected_sha256), "controller expected hash is invalid")
        self.path = str(path)
        self.expected_size = expected_size
        self.expected_sha256 = expected_sha256
        self._clock_ns = clock_ns
        self.open_started_monotonic_ns = clock_ns()
        flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC
        try:
            self.fd = os.open(self.path, flags)
        except OSError as error:
            raise CampaignError(
                f"controller O_NOFOLLOW model open failed: {self.path}: {error}"
            ) from error
        try:
            before = os.fstat(self.fd)
            require(stat.S_ISREG(before.st_mode), "controller model FD is not regular")
            require(
                before.st_nlink == 1,
                "controller model FD must be a single-link regular file",
            )
            require(
                before.st_size == expected_size,
                "controller model FD size differs from predeclaration",
            )
            digest = hashlib.sha256()
            offset = 0
            while offset < before.st_size:
                chunk = os.pread(
                    self.fd, min(1024 * 1024, before.st_size - offset), offset
                )
                require(chunk, "controller model FD hash read was incomplete")
                digest.update(chunk)
                offset += len(chunk)
            after = os.fstat(self.fd)
            identity_fields = (
                "st_dev",
                "st_ino",
                "st_mode",
                "st_nlink",
                "st_size",
                "st_ctime_ns",
            )
            require(
                all(
                    getattr(before, field) == getattr(after, field)
                    for field in identity_fields
                ),
                "controller model FD changed during initial hash",
            )
            require(
                digest.hexdigest() == expected_sha256,
                "controller model FD hash differs from predeclaration",
            )
            require(
                fcntl.fcntl(self.fd, fcntl.F_GETFD) & fcntl.FD_CLOEXEC,
                "controller model FD is not FD_CLOEXEC",
            )
            self._identity = {
                "device": before.st_dev,
                "inode": before.st_ino,
                "mode": before.st_mode,
                "link_count": before.st_nlink,
                "size_bytes": before.st_size,
                "ctime_ns": before.st_ctime_ns,
            }
            self.custody_complete_monotonic_ns = clock_ns()
            require(
                self.custody_complete_monotonic_ns > self.open_started_monotonic_ns,
                "controller model FD custody clock did not advance",
            )
        except BaseException:
            os.close(self.fd)
            self.fd = -1
            raise

    def observe(self, stage: str) -> dict[str, Any]:
        require(self.fd >= 0, "controller model FD is closed")
        require(isinstance(stage, str) and stage, "controller FD stage is invalid")
        observed = os.fstat(self.fd)
        identity = {
            "device": observed.st_dev,
            "inode": observed.st_ino,
            "mode": observed.st_mode,
            "link_count": observed.st_nlink,
            "size_bytes": observed.st_size,
            "ctime_ns": observed.st_ctime_ns,
        }
        require(identity == self._identity, "controller model FD identity changed")
        require(
            fcntl.fcntl(self.fd, fcntl.F_GETFD) & fcntl.FD_CLOEXEC,
            "controller model FD lost FD_CLOEXEC",
        )
        return {
            "format": "apxinf-controller-held-model-fd-v3",
            "schema_version": 3,
            "stage": stage,
            "controller_pid": os.getpid(),
            "fd": self.fd,
            "absolute_path": self.path,
            **copy.deepcopy(identity),
            "sha256": self.expected_sha256,
            "open_flags": ["O_RDONLY", "O_NOFOLLOW", "O_CLOEXEC"],
            "fd_cloexec_observed": True,
            "controller_fd_open_started_monotonic_ns": self.open_started_monotonic_ns,
            "controller_fd_custody_complete_monotonic_ns": self.custody_complete_monotonic_ns,
        }

    def close(self) -> None:
        if self.fd >= 0:
            os.close(self.fd)
            self.fd = -1

    def __enter__(self) -> ControllerModelFd:
        return self

    def __exit__(self, *_error: Any) -> None:
        self.close()


def _load_strict_json_file(
    path_value: Path | str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    raw, custody = _read_regular_no_follow(path_value, 8 * 1024 * 1024)
    return parse_strict_json_document(raw), custody


def _repository_path(value: Any, label: str) -> str:
    require(
        isinstance(value, str)
        and value
        and not value.startswith("/")
        and os.path.normpath(value) == value
        and not value.startswith("../"),
        f"{label} repository path is invalid",
    )
    return value


def _file_expectation(value: Any, label: str) -> dict[str, Any]:
    require(isinstance(value, dict), f"{label} expectation is not an object")
    require(
        set(value) == {"absolute_path", "size_bytes", "sha256"},
        f"{label} expectation fields drifted",
    )
    path = value["absolute_path"]
    require(
        isinstance(path, str)
        and path.startswith("/")
        and os.path.normpath(path) == path,
        f"{label} path is invalid",
    )
    require(
        is_int(value["size_bytes"]) and value["size_bytes"] > 0,
        f"{label} size is invalid",
    )
    require(_valid_sha256(value["sha256"]), f"{label} SHA256 is invalid")
    return copy.deepcopy(value)


def validate_execution_plan(
    plan: dict[str, Any], contract: dict[str, Any]
) -> dict[str, Any]:
    require(isinstance(plan, dict), "gateway execution plan is not an object")
    required = {
        "format",
        "schema_version",
        "edge_id",
        "repository_root",
        "contract_repository_path",
        "validator_repository_path",
        "driver_repository_path",
        "plan_repository_path",
        "marker_repository_path",
        "raw_output_path",
        "artifacts",
        "runtime",
    }
    require(set(plan) == required, "gateway execution plan fields drifted")
    require(plan["format"] == PLAN_FORMAT, "gateway plan format drifted")
    require(plan["schema_version"] == 3, "gateway plan schema drifted")
    require(plan["edge_id"] == EDGE_ID, "gateway plan edge drifted")
    root = plan["repository_root"]
    require(
        isinstance(root, str)
        and root.startswith("/")
        and os.path.normpath(root) == root,
        "repository root is invalid",
    )
    require(
        plan["contract_repository_path"] == CONTRACT_REPOSITORY_PATH
        and plan["validator_repository_path"] == VALIDATOR_REPOSITORY_PATH
        and plan["driver_repository_path"] == DRIVER_REPOSITORY_PATH,
        "plan code/contract path binding drifted",
    )
    _repository_path(plan["plan_repository_path"], "plan")
    marker_binding = contract["machine_receipt_contract"][
        "subcampaign_marker_bindings"
    ][EDGE_ID]
    require(
        plan["marker_repository_path"]
        == marker_binding["expected_marker_repository_path"]
        == MARKER_REPOSITORY_PATH,
        "gateway marker path drifted",
    )
    raw_path = plan["raw_output_path"]
    require(
        isinstance(raw_path, str)
        and raw_path.startswith("/")
        and os.path.normpath(raw_path) == raw_path,
        "gateway raw output path is invalid",
    )

    artifacts = plan["artifacts"]
    require(
        isinstance(artifacts, dict)
        and set(artifacts) == {"model", "omniinfer_cli", "gateway_backend"},
        "gateway artifact set drifted",
    )
    model = _file_expectation(artifacts["model"], "model")
    omni = _file_expectation(artifacts["omniinfer_cli"], "OmniInfer CLI")
    backend = _file_expectation(artifacts["gateway_backend"], "gateway backend")
    require(
        model
        == {
            "absolute_path": MODEL_PATH,
            "size_bytes": MODEL_SIZE,
            "sha256": MODEL_SHA256,
        },
        "gateway model artifact drifted",
    )
    require(
        omni["size_bytes"] == OMNI_CLI_SIZE and omni["sha256"] == OMNI_CLI_SHA256,
        "OmniInfer CLI artifact drifted",
    )
    require(
        backend["size_bytes"] == BACKEND_BINARY_SIZE
        and backend["sha256"] == BACKEND_BINARY_SHA256,
        "gateway backend artifact drifted",
    )

    runtime = plan["runtime"]
    runtime_fields = {
        "omni_base_url",
        "expected_gateway_argv",
        "expected_gateway_environment",
        "slot_save_path",
        "runtime_logs_path",
        "gateway_logs_path",
        "history_root",
        "mutable_log_roots",
        "custodian_control_socket_path",
        "custodian_ready_timeout_seconds",
        "custodian_shutdown_timeout_seconds",
    }
    require(
        isinstance(runtime, dict) and set(runtime) == runtime_fields,
        "gateway runtime plan fields drifted",
    )
    parsed = urllib.parse.urlsplit(runtime["omni_base_url"])
    require(
        parsed.scheme == "http"
        and parsed.hostname == "127.0.0.1"
        and parsed.port is not None
        and parsed.path in ("", "/")
        and not parsed.query,
        "OmniInfer base URL is not exact loopback HTTP",
    )
    gateway_argv = runtime["expected_gateway_argv"]
    require(
        isinstance(gateway_argv, list)
        and gateway_argv
        and all(isinstance(item, str) and item for item in gateway_argv)
        and gateway_argv[0] == omni["absolute_path"],
        "gateway argv is not fully predeclared",
    )
    gateway_environment = runtime["expected_gateway_environment"]
    require(
        isinstance(gateway_environment, dict)
        and gateway_environment
        and all(
            isinstance(name, str)
            and name
            and "=" not in name
            and isinstance(value, str)
            and "\0" not in value
            for name, value in gateway_environment.items()
        )
        and gateway_environment.get("OMNIINFER_REQUEST_HISTORY") == "0"
        and not any(
            name.startswith("DYLD_") or name.startswith("GIT_")
            for name in gateway_environment
        ),
        "gateway environment is not fully predeclared and hardened",
    )
    for field in (
        "slot_save_path",
        "runtime_logs_path",
        "gateway_logs_path",
        "history_root",
    ):
        value = runtime[field]
        require(
            isinstance(value, str)
            and value.startswith("/")
            and os.path.normpath(value) == value,
            f"runtime {field} is invalid",
        )
    mutable = runtime["mutable_log_roots"]
    require(
        isinstance(mutable, list)
        and mutable
        and all(
            isinstance(path, str)
            and path.startswith("/")
            and os.path.normpath(path) == path
            for path in mutable
        )
        and len(set(mutable)) == len(mutable),
        "mutable log roots are invalid",
    )
    require(
        set(mutable) == {runtime["runtime_logs_path"], runtime["gateway_logs_path"]},
        "mutable log roots do not exactly cover gateway and backend logs",
    )
    control_socket = runtime["custodian_control_socket_path"]
    require(
        isinstance(control_socket, str)
        and control_socket.startswith("/")
        and os.path.normpath(control_socket) == control_socket
        and len(os.fsencode(control_socket)) <= 103
        and str(Path(control_socket).parent) == runtime["gateway_logs_path"],
        "custodian control socket is invalid or outside the exact gateway log root",
    )
    require(
        runtime["custodian_ready_timeout_seconds"] == 180
        and runtime["custodian_shutdown_timeout_seconds"] == 30,
        "custodian timeout contract drifted",
    )
    layout = campaign_directory_layout(plan)
    campaign_root = Path(layout["campaign_root"])
    state_root = campaign_root / "state"
    runtime_root = campaign_root / "runtime"
    require(
        gateway_argv
        == [
            omni["absolute_path"],
            "--state-root",
            str(state_root),
            "--runtime-root",
            str(runtime_root),
            "gateway",
            "--host",
            "127.0.0.1",
            "--port",
            str(parsed.port),
            "--startup-timeout",
            str(runtime["custodian_ready_timeout_seconds"]),
        ],
        "gateway argv differs from the exact pinned campaign launch",
    )
    require(
        gateway_environment
        == {
            "LANG": "C",
            "OMNIINFER_LLAMA_CPP_MAC_LAUNCHER_PATH": backend["absolute_path"],
            "OMNIINFER_REQUEST_HISTORY": "0",
            "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
        },
        "gateway environment differs from the exact pinned launch environment",
    )
    validate_frozen_gateway_contract(contract)
    return copy.deepcopy(plan)


def load_execution_context(plan_path_value: Path | str) -> dict[str, Any]:
    plan_path = Path(plan_path_value).resolve(strict=True)
    plan_raw, plan_file = _read_regular_no_follow(plan_path, 8 * 1024 * 1024)
    plan_preliminary = parse_strict_json_document(plan_raw)
    repository_root_value = plan_preliminary.get("repository_root")
    require(
        isinstance(repository_root_value, str)
        and repository_root_value.startswith("/"),
        "plan repository root is absent",
    )
    repository_root = Path(repository_root_value).resolve(strict=True)
    frozen = _SHARED.load_frozen_contract(
        repository_root / CONTRACT_REPOSITORY_PATH,
        repository_root / VALIDATOR_REPOSITORY_PATH,
    )
    plan = validate_execution_plan(plan_preliminary, frozen["contract"])
    expected_plan = repository_root / plan["plan_repository_path"]
    require(
        expected_plan.resolve(strict=True) == plan_path,
        "plan CLI path differs from repository binding",
    )
    require(
        (repository_root / DRIVER_REPOSITORY_PATH).resolve(strict=True)
        == Path(__file__).resolve(strict=True),
        "executed gateway driver differs from repository binding",
    )
    require(
        (repository_root / SHARED_DRIVER_REPOSITORY_PATH).resolve(strict=True)
        == _NATIVE_DRIVER_PATH.resolve(strict=True),
        "loaded shared formal driver differs from repository binding",
    )
    return {
        "repository_root": repository_root,
        "plan": plan,
        "plan_file": plan_file,
        **frozen,
    }


def tracked_campaign_paths(
    plan: dict[str, Any], *, include_marker: bool
) -> dict[str, str]:
    result = {
        "contract": plan["contract_repository_path"],
        "validator": plan["validator_repository_path"],
        "driver": plan["driver_repository_path"],
        "shared_formal_driver": SHARED_DRIVER_REPOSITORY_PATH,
        "plan": plan["plan_repository_path"],
    }
    if include_marker:
        result["activation_marker"] = plan["marker_repository_path"]
    return result


def collect_git_custody(
    repository_root: Path,
    contract: dict[str, Any],
    tracked: dict[str, str],
    *,
    include_marker: bool,
) -> dict[str, Any]:
    """Use shared Git parsing with a non-ambient command environment."""

    return _SHARED.collect_git_custody(
        repository_root,
        contract,
        tracked,
        command_runner=hardened_command_runner,
        published_marker_label="activation_marker" if include_marker else None,
    )


def verify_plan_artifacts(plan: dict[str, Any]) -> dict[str, Any]:
    return {
        label: _SHARED.file_custody(expectation)
        for label, expectation in plan["artifacts"].items()
    }


def tree_manifest(path_value: Path | str) -> dict[str, Any]:
    """Hash a tree without following any directory entry symlink."""

    root = Path(path_value)
    require(root.is_absolute(), "tree manifest root must be absolute")
    try:
        root_lstat = root.lstat()
    except FileNotFoundError:
        entries: list[dict[str, Any]] = []
        return {
            "absolute_root": str(root),
            "exists": False,
            "entries": entries,
            "canonical_sha256": sha256_canonical(entries),
        }
    require(
        stat.S_ISDIR(root_lstat.st_mode), f"tree root is not a direct directory: {root}"
    )
    root_before = root.stat(follow_symlinks=False)
    entries = []
    pending = [root]
    while pending:
        directory = pending.pop()
        with os.scandir(directory) as iterator:
            children = sorted(iterator, key=lambda entry: entry.name)
        for child in children:
            path = Path(child.path)
            relative = str(path.relative_to(root))
            child_stat = child.stat(follow_symlinks=False)
            require(
                not stat.S_ISLNK(child_stat.st_mode), f"tree contains symlink: {path}"
            )
            if stat.S_ISDIR(child_stat.st_mode):
                entries.append({"relative_path": relative, "kind": "directory"})
                pending.append(path)
            elif stat.S_ISREG(child_stat.st_mode):
                _, receipt = _read_regular_no_follow(path, 512 * 1024 * 1024)
                entries.append(
                    {
                        "relative_path": relative,
                        "kind": "file",
                        "device": receipt["device"],
                        "inode": receipt["inode"],
                        "mode": receipt["mode"],
                        "size_bytes": receipt["size_bytes"],
                        "ctime_ns": receipt["ctime_ns"],
                        "sha256": receipt["sha256"],
                    }
                )
            elif stat.S_ISSOCK(child_stat.st_mode):
                entries.append(
                    {
                        "relative_path": relative,
                        "kind": "unix-domain-socket",
                        "device": child_stat.st_dev,
                        "inode": child_stat.st_ino,
                        "mode": child_stat.st_mode,
                        "uid": child_stat.st_uid,
                        "ctime_ns": child_stat.st_ctime_ns,
                    }
                )
            else:
                raise CampaignError(f"tree contains unsupported entry: {path}")
    entries.sort(key=lambda entry: entry["relative_path"])
    root_after = root.stat(follow_symlinks=False)
    root_identity = ("st_dev", "st_ino", "st_mode", "st_ctime_ns")
    require(
        all(
            getattr(root_before, field) == getattr(root_after, field)
            for field in root_identity
        ),
        f"tree root changed while manifesting: {root}",
    )
    return {
        "absolute_root": str(root),
        "exists": True,
        "root_device": root_before.st_dev,
        "root_inode": root_before.st_ino,
        "root_mode": root_before.st_mode,
        "root_ctime_ns": root_before.st_ctime_ns,
        "root_start_end_identity_equal": True,
        "entries": entries,
        "canonical_sha256": sha256_canonical(entries),
    }


def campaign_directory_layout(plan: dict[str, Any]) -> dict[str, Any]:
    """Derive the one admitted private campaign tree from the frozen plan paths."""

    runtime = plan["runtime"]
    raw_path = Path(plan["raw_output_path"])
    slots = Path(runtime["slot_save_path"])
    gateway_logs = Path(runtime["gateway_logs_path"])
    history = Path(runtime["history_root"])
    runtime_logs = Path(runtime["runtime_logs_path"])
    control_socket = Path(runtime["custodian_control_socket_path"])
    candidates = (raw_path, slots, gateway_logs, history, runtime_logs, control_socket)
    require(
        all(
            path.is_absolute() and os.path.normpath(str(path)) == str(path)
            for path in candidates
        ),
        "campaign directory plan paths are not exact absolute paths",
    )
    state_root = gateway_logs.parent.parent
    runtime_root = runtime_logs.parent.parent
    campaign_root = state_root.parent
    expected = (
        campaign_root,
        campaign_root / "state",
        campaign_root / "state" / ".local",
        campaign_root / "state" / ".local" / "logs",
        campaign_root / "state" / ".local" / "request_history",
        campaign_root / "runtime",
        campaign_root / "runtime" / "llama.cpp-mac",
        campaign_root / "runtime" / "llama.cpp-mac" / "logs",
        campaign_root / "raw",
        campaign_root / "slots",
    )
    require(
        state_root == campaign_root / "state"
        and runtime_root == campaign_root / "runtime"
        and raw_path.parent == campaign_root / "raw"
        and raw_path.parent != raw_path
        and slots == campaign_root / "slots"
        and gateway_logs == campaign_root / "state" / ".local" / "logs"
        and history == campaign_root / "state" / ".local" / "request_history"
        and runtime_logs == campaign_root / "runtime" / "llama.cpp-mac" / "logs"
        and control_socket.parent == gateway_logs
        and campaign_root != Path("/")
        and campaign_root.parent != campaign_root,
        "campaign paths do not form the exact isolated state/runtime/raw/slots tree",
    )

    argv = runtime["expected_gateway_argv"]

    def unique_option(option: str) -> str:
        indexes = [index for index, value in enumerate(argv) if value == option]
        require(
            len(indexes) == 1 and indexes[0] + 1 < len(argv),
            f"gateway argv does not contain one exact {option} binding",
        )
        return argv[indexes[0] + 1]

    require(
        unique_option("--state-root") == str(state_root)
        and unique_option("--runtime-root") == str(runtime_root),
        "gateway argv state/runtime roots differ from campaign directory custody",
    )
    children: dict[str, list[str]] = {str(path): [] for path in expected}
    expected_set = set(expected)
    for path in expected[1:]:
        require(path.parent in expected_set, "campaign tree has an unbound parent")
        children[str(path.parent)].append(path.name)
    for names in children.values():
        names.sort()
    return {
        "campaign_root": str(campaign_root),
        "expected_directory_paths": [str(path) for path in expected],
        "expected_child_names": children,
    }


def _campaign_directory_identity(path: Path) -> os.stat_result:
    require(
        hasattr(os, "O_NOFOLLOW")
        and hasattr(os, "O_CLOEXEC")
        and hasattr(os, "O_DIRECTORY"),
        "platform lacks fail-closed directory open flags",
    )
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_DIRECTORY
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise CampaignError(
            f"campaign directory O_NOFOLLOW open failed: {path}"
        ) from error
    try:
        observed = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    require(stat.S_ISDIR(observed.st_mode), f"campaign path is not a directory: {path}")
    require(
        observed.st_uid == os.geteuid(),
        f"campaign directory ownership drifted: {path}",
    )
    require(
        stat.S_IMODE(observed.st_mode) == 0o700,
        f"campaign directory permissions are not 0700: {path}",
    )
    require(
        path.resolve(strict=True) == path,
        f"campaign directory path contains a symlink or alias: {path}",
    )
    return observed


def _observe_initialized_campaign_directory(
    path: Path, campaign_root: Path, expected_children: list[str]
) -> dict[str, Any]:
    observed = _campaign_directory_identity(path)
    child_names = sorted(os.listdir(path))
    require(
        child_names == expected_children,
        f"campaign directory contains an unexpected or absent entry: {path}",
    )
    return {
        "absolute_path": str(path),
        "relative_path": str(path.relative_to(campaign_root)),
        "device": observed.st_dev,
        "inode": observed.st_ino,
        "mode": observed.st_mode,
        "permission_bits": stat.S_IMODE(observed.st_mode),
        "uid": observed.st_uid,
        "gid": observed.st_gid,
        "expected_child_names": copy.deepcopy(expected_children),
        "observed_child_names": child_names,
        "direct_directory_no_symlink": True,
        "owner_matches_controller": True,
        "permissions_are_0700": True,
    }


def initialize_campaign_directory_tree(
    plan: dict[str, Any], *, mkdir: Any = os.mkdir
) -> dict[str, Any]:
    """Create only the exact empty private campaign tree, before runtime launch."""

    layout = campaign_directory_layout(plan)
    campaign_root = Path(layout["campaign_root"])
    expected_paths = [Path(value) for value in layout["expected_directory_paths"]]
    expected_set = set(expected_paths)
    root_parent = campaign_root.parent
    try:
        parent_lstat = root_parent.lstat()
    except OSError as error:
        raise CampaignError("campaign root parent is absent") from error
    require(
        stat.S_ISDIR(parent_lstat.st_mode)
        and not stat.S_ISLNK(parent_lstat.st_mode)
        and root_parent.resolve(strict=True) == root_parent,
        "campaign root parent is not a direct canonical directory",
    )
    require(
        not os.path.lexists(plan["raw_output_path"]),
        "gateway raw output already exists before directory initialization",
    )
    require(
        not os.path.lexists(plan["runtime"]["custodian_control_socket_path"]),
        "custodian socket already exists before directory initialization",
    )

    preexisting: list[str] = []
    for path in expected_paths:
        if not os.path.lexists(path):
            continue
        preexisting.append(str(path))
        observed = _campaign_directory_identity(path)
        require(
            stat.S_ISDIR(observed.st_mode), f"campaign path is not a directory: {path}"
        )
        with os.scandir(path) as iterator:
            children = list(iterator)
        for child in children:
            child_path = Path(child.path)
            child_stat = child.stat(follow_symlinks=False)
            require(
                child_path in expected_set
                and stat.S_ISDIR(child_stat.st_mode)
                and not stat.S_ISLNK(child_stat.st_mode),
                f"campaign tree contains an unexpected preexisting entry: {child_path}",
            )

    require(callable(mkdir), "campaign directory mkdir boundary is not callable")
    created: list[str] = []
    created_identities: dict[str, tuple[int, int, int, int]] = {}
    for path in expected_paths:
        if os.path.lexists(path):
            continue
        try:
            mkdir(path, 0o700)
        except OSError as error:
            for created_path_text in reversed(created):
                created_path = Path(created_path_text)
                identity = created_identities[created_path_text]
                try:
                    current = created_path.lstat()
                    if (
                        (
                            current.st_dev,
                            current.st_ino,
                            current.st_mode,
                            current.st_uid,
                        )
                        == identity
                        and stat.S_ISDIR(current.st_mode)
                        and not stat.S_ISLNK(current.st_mode)
                    ):
                        os.rmdir(created_path)
                except OSError:
                    pass
            raise CampaignError(
                f"campaign directory creation failed: {path}"
            ) from error
        created.append(str(path))
        observed = path.lstat()
        created_identities[str(path)] = (
            observed.st_dev,
            observed.st_ino,
            observed.st_mode,
            observed.st_uid,
        )

    observations = [
        _observe_initialized_campaign_directory(
            path,
            campaign_root,
            layout["expected_child_names"][str(path)],
        )
        for path in expected_paths
    ]
    receipt = {
        "format": "apxinf-omniinfer-gateway-campaign-directory-initialization-v3",
        "schema_version": 3,
        "edge_id": EDGE_ID,
        "campaign_root": str(campaign_root),
        "expected_directory_paths": [str(path) for path in expected_paths],
        "preexisting_directory_paths": preexisting,
        "created_directory_paths": created,
        "directory_observations": observations,
        "retry_policy": {
            "exact_empty_partial_tree_reusable": True,
            "exact_empty_complete_tree_reusable": True,
            "preexisting_content_or_unexpected_entry_rejected": True,
            "symlink_or_non_directory_rejected": True,
            "cleanup_limited_to_directories_created_by_this_attempt": True,
        },
        "initial_tree_sha256": sha256_canonical(observations),
        "generation_requests": 0,
        "runtime_processes_started": 0,
        "marker_created": False,
        "raw_created": False,
        "all_passed": True,
    }
    validate_campaign_directory_initialization_receipt(receipt, plan, verify_live=True)
    return receipt


def validate_campaign_directory_initialization_receipt(
    receipt: dict[str, Any], plan: dict[str, Any], *, verify_live: bool = False
) -> dict[str, Any]:
    layout = campaign_directory_layout(plan)
    expected_paths = layout["expected_directory_paths"]
    require(
        isinstance(receipt, dict)
        and set(receipt) == CAMPAIGN_DIRECTORY_INITIALIZATION_FIELDS_V3
        and receipt.get("format")
        == "apxinf-omniinfer-gateway-campaign-directory-initialization-v3"
        and receipt.get("schema_version") == 3
        and receipt.get("edge_id") == EDGE_ID
        and receipt.get("campaign_root") == layout["campaign_root"]
        and receipt.get("expected_directory_paths") == expected_paths
        and receipt.get("generation_requests") == 0
        and receipt.get("runtime_processes_started") == 0
        and receipt.get("marker_created") is False
        and receipt.get("raw_created") is False
        and receipt.get("all_passed") is True,
        "campaign directory initialization receipt drifted",
    )
    preexisting = receipt.get("preexisting_directory_paths")
    created = receipt.get("created_directory_paths")
    require(
        isinstance(preexisting, list)
        and isinstance(created, list)
        and len(preexisting) == len(set(preexisting))
        and len(created) == len(set(created))
        and not set(preexisting).intersection(created)
        and set(preexisting).union(created) == set(expected_paths)
        and preexisting == [path for path in expected_paths if path in preexisting]
        and created == [path for path in expected_paths if path in created],
        "campaign directory creation/preexistence partition drifted",
    )
    require(
        receipt.get("retry_policy")
        == {
            "exact_empty_partial_tree_reusable": True,
            "exact_empty_complete_tree_reusable": True,
            "preexisting_content_or_unexpected_entry_rejected": True,
            "symlink_or_non_directory_rejected": True,
            "cleanup_limited_to_directories_created_by_this_attempt": True,
        },
        "campaign directory retry policy drifted",
    )
    observations = receipt.get("directory_observations")
    require(
        isinstance(observations, list)
        and len(observations) == len(expected_paths)
        and receipt.get("initial_tree_sha256") == sha256_canonical(observations),
        "campaign directory observation set drifted",
    )
    campaign_root = Path(layout["campaign_root"])
    require(
        len(expected_paths) == len(observations),
        "campaign directory observation length drifted",
    )
    for path_text, observation in zip(expected_paths, observations):
        expected_children = layout["expected_child_names"][path_text]
        require(
            isinstance(observation, dict)
            and set(observation) == CAMPAIGN_DIRECTORY_OBSERVATION_FIELDS_V3
            and observation.get("absolute_path") == path_text
            and observation.get("relative_path")
            == str(Path(path_text).relative_to(campaign_root))
            and is_int(observation.get("device"))
            and is_int(observation.get("inode"))
            and stat.S_ISDIR(observation.get("mode", 0))
            and observation.get("permission_bits") == 0o700
            and observation.get("uid") == os.geteuid()
            and is_int(observation.get("gid"))
            and observation.get("expected_child_names") == expected_children
            and observation.get("observed_child_names") == expected_children
            and observation.get("direct_directory_no_symlink") is True
            and observation.get("owner_matches_controller") is True
            and observation.get("permissions_are_0700") is True,
            "campaign directory observation drifted",
        )
        if verify_live:
            live = _campaign_directory_identity(Path(path_text))
            require(
                live.st_dev == observation["device"]
                and live.st_ino == observation["inode"]
                and live.st_mode == observation["mode"],
                f"campaign directory identity changed: {path_text}",
            )
            require(
                all((Path(path_text) / child).is_dir() for child in expected_children),
                f"campaign directory lost an expected child: {path_text}",
            )
    return copy.deepcopy(receipt)


def cleanup_failed_prepare_campaign_tree(
    plan: dict[str, Any],
    receipt: dict[str, Any],
    reason: str,
    *,
    before_first_removal: Any | None = None,
) -> dict[str, Any]:
    """Remove one exclusively new tree through fixed parent/root directory FDs."""

    admitted = validate_campaign_directory_initialization_receipt(receipt, plan)
    require(isinstance(reason, str) and reason, "campaign cleanup reason is absent")
    root = Path(admitted["campaign_root"])
    marker = Path(plan["repository_root"]) / plan["marker_repository_path"]
    raw = Path(plan["raw_output_path"])
    base = {
        "format": "apxinf-omniinfer-gateway-failed-prepare-directory-cleanup-v3",
        "schema_version": 3,
        "edge_id": EDGE_ID,
        "reason": reason,
        "campaign_root": str(root),
        "initialization_receipt_sha256": sha256_canonical(admitted),
        "preexisting_directory_paths": copy.deepcopy(
            admitted["preexisting_directory_paths"]
        ),
        "cleanup_eligible": False,
        "contamination_detected": False,
        "removed_paths": [],
        "root_removed": False,
        "marker_present": os.path.lexists(marker),
        "raw_present": os.path.lexists(raw),
        "signals_sent": 0,
        "all_passed": False,
    }
    require(set(base) == CAMPAIGN_DIRECTORY_CLEANUP_FIELDS_V3, "cleanup schema drifted")
    if base["marker_present"] or base["raw_present"]:
        base["contamination_detected"] = True
        return base
    if admitted["preexisting_directory_paths"]:
        return base
    require(
        before_first_removal is None or callable(before_first_removal),
        "campaign cleanup mutation hook is not callable",
    )
    directory_flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_DIRECTORY
    parent_fd = -1
    directory_fds: dict[tuple[str, ...], int] = {}
    try:
        parent_fd = os.open(root.parent, directory_flags)
        try:
            root_fd = os.open(root.name, directory_flags, dir_fd=parent_fd)
        except FileNotFoundError:
            base.update(cleanup_eligible=True, root_removed=True, all_passed=True)
            return base
        directory_fds[()] = root_fd
        root_live = os.fstat(root_fd)
        root_observation = admitted["directory_observations"][0]
        if not (
            root_live.st_dev == root_observation["device"]
            and root_live.st_ino == root_observation["inode"]
            and root_live.st_mode == root_observation["mode"]
            and root_live.st_uid == os.geteuid()
        ):
            base["contamination_detected"] = True
            return base

        def root_name_still_bound() -> bool:
            try:
                current = os.stat(root.name, dir_fd=parent_fd, follow_symlinks=False)
            except OSError:
                return False
            return (
                current.st_dev == root_live.st_dev
                and current.st_ino == root_live.st_ino
                and current.st_mode == root_live.st_mode
                and not stat.S_ISLNK(current.st_mode)
            )

        if not root_name_still_bound():
            base["contamination_detected"] = True
            return base

        entries: list[tuple[tuple[str, ...], str, os.stat_result]] = []

        def scan_directory(parts: tuple[str, ...], descriptor: int) -> None:
            require(len(entries) <= 100_000, "campaign cleanup tree is oversized")
            for name in sorted(os.listdir(descriptor)):
                require(
                    name not in (".", "..") and "/" not in name and "\x00" not in name,
                    "campaign cleanup entry name is invalid",
                )
                observed = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
                require(
                    not stat.S_ISLNK(observed.st_mode)
                    and observed.st_uid == os.geteuid()
                    and (
                        stat.S_ISDIR(observed.st_mode)
                        or stat.S_ISREG(observed.st_mode)
                        or stat.S_ISSOCK(observed.st_mode)
                    )
                    and (not stat.S_ISREG(observed.st_mode) or observed.st_nlink == 1),
                    "campaign cleanup tree contains an unsafe entry",
                )
                entries.append((parts, name, observed))
                if stat.S_ISDIR(observed.st_mode):
                    child_parts = (*parts, name)
                    child_fd = os.open(name, directory_flags, dir_fd=descriptor)
                    child_live = os.fstat(child_fd)
                    require(
                        child_live.st_dev == observed.st_dev
                        and child_live.st_ino == observed.st_ino
                        and child_live.st_mode == observed.st_mode,
                        "campaign child directory changed while opening",
                    )
                    directory_fds[child_parts] = child_fd
                    scan_directory(child_parts, child_fd)

        try:
            scan_directory((), root_fd)
        except (OSError, CampaignError):
            base["contamination_detected"] = True
            return base
        base["cleanup_eligible"] = True
        if before_first_removal is not None:
            before_first_removal()
        removed: list[str] = []
        for parent_parts, name, initial in sorted(
            entries, key=lambda item: len(item[0]) + 1, reverse=True
        ):
            if not root_name_still_bound():
                base["contamination_detected"] = True
                base["removed_paths"] = removed
                return base
            parent_descriptor = directory_fds[parent_parts]
            try:
                current = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
            except OSError:
                base["contamination_detected"] = True
                base["removed_paths"] = removed
                return base
            if not (
                current.st_dev == initial.st_dev
                and current.st_ino == initial.st_ino
                and current.st_mode == initial.st_mode
                and current.st_uid == initial.st_uid
                and not stat.S_ISLNK(current.st_mode)
            ):
                base["contamination_detected"] = True
                base["removed_paths"] = removed
                return base
            try:
                if stat.S_ISDIR(current.st_mode):
                    os.rmdir(name, dir_fd=parent_descriptor)
                else:
                    os.unlink(name, dir_fd=parent_descriptor)
            except OSError:
                base["removed_paths"] = removed
                return base
            removed.append(str(root.joinpath(*parent_parts, name)))
        if not root_name_still_bound():
            base["contamination_detected"] = True
            base["removed_paths"] = removed
            return base
        try:
            os.rmdir(root.name, dir_fd=parent_fd)
        except OSError:
            base["removed_paths"] = removed
            return base
        removed.append(str(root))
        base.update(
            removed_paths=removed,
            root_removed=True,
            marker_present=os.path.lexists(marker),
            raw_present=os.path.lexists(raw),
            all_passed=(not os.path.lexists(root)),
        )
        return base
    except OSError:
        base["contamination_detected"] = True
        return base
    finally:
        for descriptor in reversed(list(directory_fds.values())):
            try:
                os.close(descriptor)
            except OSError:
                pass
        if parent_fd >= 0:
            try:
                os.close(parent_fd)
            except OSError:
                pass


class _ProcBSDInfo(ctypes.Structure):
    _fields_ = [
        ("pbi_flags", ctypes.c_uint32),
        ("pbi_status", ctypes.c_uint32),
        ("pbi_xstatus", ctypes.c_uint32),
        ("pbi_pid", ctypes.c_uint32),
        ("pbi_ppid", ctypes.c_uint32),
        ("pbi_uid", ctypes.c_uint32),
        ("pbi_gid", ctypes.c_uint32),
        ("pbi_ruid", ctypes.c_uint32),
        ("pbi_rgid", ctypes.c_uint32),
        ("pbi_svuid", ctypes.c_uint32),
        ("pbi_svgid", ctypes.c_uint32),
        ("pbi_rfu_1", ctypes.c_uint32),
        ("pbi_comm", ctypes.c_char * 16),
        ("pbi_name", ctypes.c_char * 32),
        ("pbi_nfiles", ctypes.c_uint32),
        ("pbi_pgid", ctypes.c_uint32),
        ("pbi_pjobc", ctypes.c_uint32),
        ("e_tdev", ctypes.c_uint32),
        ("e_tpgid", ctypes.c_uint32),
        ("pbi_nice", ctypes.c_int32),
        ("pbi_start_tvsec", ctypes.c_uint64),
        ("pbi_start_tvusec", ctypes.c_uint64),
    ]


class _ProcFdInfo(ctypes.Structure):
    _fields_ = [
        ("proc_fd", ctypes.c_int32),
        ("proc_fdtype", ctypes.c_uint32),
    ]


class _ProcFileInfo(ctypes.Structure):
    _fields_ = [
        ("fi_openflags", ctypes.c_uint32),
        ("fi_status", ctypes.c_uint32),
        ("fi_offset", ctypes.c_int64),
        ("fi_type", ctypes.c_int32),
        ("fi_guardflags", ctypes.c_uint32),
    ]


class _VinfoStat(ctypes.Structure):
    _fields_ = [
        ("vst_dev", ctypes.c_uint32),
        ("vst_mode", ctypes.c_uint16),
        ("vst_nlink", ctypes.c_uint16),
        ("vst_ino", ctypes.c_uint64),
        ("vst_uid", ctypes.c_uint32),
        ("vst_gid", ctypes.c_uint32),
        ("vst_atime", ctypes.c_int64),
        ("vst_atimensec", ctypes.c_int64),
        ("vst_mtime", ctypes.c_int64),
        ("vst_mtimensec", ctypes.c_int64),
        ("vst_ctime", ctypes.c_int64),
        ("vst_ctimensec", ctypes.c_int64),
        ("vst_birthtime", ctypes.c_int64),
        ("vst_birthtimensec", ctypes.c_int64),
        ("vst_size", ctypes.c_int64),
        ("vst_blocks", ctypes.c_int64),
        ("vst_blksize", ctypes.c_int32),
        ("vst_flags", ctypes.c_uint32),
        ("vst_gen", ctypes.c_uint32),
        ("vst_rdev", ctypes.c_uint32),
        ("vst_qspare", ctypes.c_int64 * 2),
    ]


class _Fsid(ctypes.Structure):
    _fields_ = [("val", ctypes.c_int32 * 2)]


class _VnodeInfo(ctypes.Structure):
    _fields_ = [
        ("vi_stat", _VinfoStat),
        ("vi_type", ctypes.c_int32),
        ("vi_pad", ctypes.c_int32),
        ("vi_fsid", _Fsid),
    ]


class _VnodeInfoPath(ctypes.Structure):
    _fields_ = [
        ("vip_vi", _VnodeInfo),
        ("vip_path", ctypes.c_char * 1024),
    ]


class _VnodeFdInfoWithPath(ctypes.Structure):
    _fields_ = [
        ("pfi", _ProcFileInfo),
        ("pvip", _VnodeInfoPath),
    ]


_LIBPROC: Any | None = None
_LIBSYSTEM: Any | None = None


def _load_libproc() -> Any:
    global _LIBPROC
    if _LIBPROC is None:
        _LIBPROC = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
    _LIBPROC.proc_pidinfo.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_uint64,
        ctypes.c_void_p,
        ctypes.c_int,
    ]
    _LIBPROC.proc_pidinfo.restype = ctypes.c_int
    _LIBPROC.proc_pidfdinfo.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_int,
    ]
    _LIBPROC.proc_pidfdinfo.restype = ctypes.c_int
    return _LIBPROC


def process_start_identity(pid: int) -> dict[str, int]:
    """Read the microsecond kernel process birth identity through libproc."""

    if platform.system() != "Darwin":
        raise_preflight_blocker(
            "MACOS_KERNEL_PROCESS_START_IDENTITY_UNAVAILABLE",
            "formal same-resident process custody requires Darwin libproc",
        )
    require(is_int(pid) and pid > 0, "process PID is invalid")
    libproc = _load_libproc()
    info = _ProcBSDInfo()
    size = ctypes.sizeof(info)
    result = libproc.proc_pidinfo(pid, 3, 0, ctypes.byref(info), size)
    if result != size:
        raise_preflight_blocker(
            "MACOS_KERNEL_PROCESS_START_IDENTITY_UNAVAILABLE",
            f"PROC_PIDTBSDINFO did not return an exact proc_bsdinfo for PID {pid}",
            {"pid": pid, "expected_size": size, "returned_size": result},
        )
    require(info.pbi_pid == pid, f"libproc returned a different PID for {pid}")
    require(info.pbi_start_tvsec > 0, f"PID {pid} has invalid birth seconds")
    require(
        info.pbi_start_tvusec < 1_000_000, f"PID {pid} has invalid birth microseconds"
    )
    return {
        "seconds": int(info.pbi_start_tvsec),
        "microseconds": int(info.pbi_start_tvusec),
    }


def proc_vnode_fd_entries(pid: int) -> list[dict[str, Any]]:
    """Enumerate vnode FDs through libproc, failing on any ambiguous race."""

    if platform.system() != "Darwin":
        raise_preflight_blocker(
            "MACOS_PROC_PIDFDINFO_UNAVAILABLE",
            "formal loaded-model FD custody requires Darwin libproc",
        )
    require(is_int(pid) and pid > 0, "process PID is invalid")
    libproc = _load_libproc()
    start = process_start_identity(pid)
    capacity = 1024 * 1024
    buffer = ctypes.create_string_buffer(capacity)
    returned = libproc.proc_pidinfo(pid, 1, 0, buffer, capacity)
    item_size = ctypes.sizeof(_ProcFdInfo)
    if returned <= 0 or returned >= capacity or returned % item_size != 0:
        raise_preflight_blocker(
            "MACOS_PROC_PIDLISTFDS_AMBIGUOUS",
            f"PROC_PIDLISTFDS was incomplete for PID {pid}",
            {
                "pid": pid,
                "returned_size": returned,
                "buffer_size": capacity,
                "entry_size": item_size,
                "errno": ctypes.get_errno(),
            },
        )
    count = returned // item_size
    descriptors = ctypes.cast(buffer, ctypes.POINTER(_ProcFdInfo))
    vnode_size = ctypes.sizeof(_VnodeFdInfoWithPath)
    result: list[dict[str, Any]] = []
    seen: set[int] = set()
    for index in range(count):
        descriptor = descriptors[index]
        fd = int(descriptor.proc_fd)
        require(fd >= 0 and fd not in seen, f"PID {pid} FD list is invalid")
        seen.add(fd)
        if descriptor.proc_fdtype != 1:
            continue
        info = _VnodeFdInfoWithPath()
        actual = libproc.proc_pidfdinfo(pid, fd, 2, ctypes.byref(info), vnode_size)
        if actual != vnode_size:
            raise_preflight_blocker(
                "MACOS_PROC_PIDFDINFO_RACE",
                f"PROC_PIDFDVNODEPATHINFO changed while enumerating PID {pid} FD {fd}",
                {
                    "pid": pid,
                    "fd": fd,
                    "expected_size": vnode_size,
                    "returned_size": actual,
                    "errno": ctypes.get_errno(),
                },
            )
        raw_path = bytes(info.pvip.vip_path).split(b"\0", 1)[0]
        try:
            path = raw_path.decode("utf-8", errors="strict")
        except UnicodeDecodeError as error:
            raise CampaignError(f"PID {pid} FD {fd} vnode path is not UTF-8") from error
        vnode = info.pvip.vip_vi.vi_stat
        result.append(
            {
                "fd": fd,
                "fd_type": "vnode",
                "open_flags": int(info.pfi.fi_openflags),
                "close_on_exec": bool(info.pfi.fi_status & 2),
                "device": int(vnode.vst_dev),
                "inode": int(vnode.vst_ino),
                "mode": int(vnode.vst_mode),
                "link_count": int(vnode.vst_nlink),
                "size_bytes": int(vnode.vst_size),
                "ctime_ns": int(vnode.vst_ctime) * 1_000_000_000
                + int(vnode.vst_ctimensec),
                "path": path,
            }
        )
    require(
        process_start_identity(pid) == start,
        f"PID {pid} changed while enumerating file descriptors",
    )
    return result


def process_argv_environment(pid: int) -> dict[str, Any]:
    """Read exact NUL-delimited argv/environment bytes through KERN_PROCARGS2."""

    if platform.system() != "Darwin":
        raise_preflight_blocker(
            "MACOS_KERNEL_ARGV_ENVIRONMENT_UNAVAILABLE",
            "formal argv/environment custody requires Darwin KERN_PROCARGS2",
        )
    require(is_int(pid) and pid > 0, "process PID is invalid")
    global _LIBSYSTEM
    if _LIBSYSTEM is None:
        _LIBSYSTEM = ctypes.CDLL("/usr/lib/libSystem.B.dylib", use_errno=True)
        _LIBSYSTEM.sysctl.argtypes = [
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_uint,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.c_void_p,
            ctypes.c_size_t,
        ]
        _LIBSYSTEM.sysctl.restype = ctypes.c_int
    maximum_raw = _run_checked(["/usr/sbin/sysctl", "-n", "kern.argmax"])
    try:
        maximum_text = maximum_raw.decode("ascii", errors="strict").strip()
    except UnicodeDecodeError as error:
        raise CampaignError("kern.argmax output is not ASCII") from error
    require(maximum_text.isdigit(), "kern.argmax is invalid")
    maximum = int(maximum_text)
    require(4096 <= maximum <= 16 * 1024 * 1024, "kern.argmax is outside bounds")
    mib = (ctypes.c_int * 3)(1, 49, pid)
    buffer = ctypes.create_string_buffer(maximum)
    size = ctypes.c_size_t(maximum)
    result = _LIBSYSTEM.sysctl(mib, 3, buffer, ctypes.byref(size), None, 0)
    if result != 0:
        raise_preflight_blocker(
            "MACOS_KERNEL_ARGV_ENVIRONMENT_UNAVAILABLE",
            f"KERN_PROCARGS2 failed for PID {pid}",
            {"pid": pid, "errno": ctypes.get_errno()},
        )
    raw = buffer.raw[: size.value]
    require(len(raw) >= 5, f"KERN_PROCARGS2 result is truncated for PID {pid}")
    argc = struct.unpack_from("=i", raw, 0)[0]
    require(0 < argc <= 4096, f"KERN_PROCARGS2 argc is invalid for PID {pid}")
    offset = 4

    def take_string() -> bytes:
        nonlocal offset
        end = raw.find(b"\0", offset)
        require(end >= offset, f"KERN_PROCARGS2 string is unterminated for PID {pid}")
        value = raw[offset:end]
        offset = end + 1
        return value

    executable_raw = take_string()
    while offset < len(raw) and raw[offset] == 0:
        offset += 1
    argv_raw = [take_string() for _ in range(argc)]
    environment_raw: list[bytes] = []
    while offset < len(raw):
        if raw[offset] == 0:
            if any(byte != 0 for byte in raw[offset:]):
                offset += 1
                continue
            break
        environment_raw.append(take_string())
    try:
        executable = executable_raw.decode("utf-8", errors="strict")
        argv = [value.decode("utf-8", errors="strict") for value in argv_raw]
        environment = [
            value.decode("utf-8", errors="strict") for value in environment_raw
        ]
    except UnicodeDecodeError as error:
        raise CampaignError(
            f"PID {pid} argv/environment is not strict UTF-8"
        ) from error
    require(
        all("=" in entry and not entry.startswith("=") for entry in environment),
        f"PID {pid} environment entry is malformed",
    )
    environment_names = [entry.split("=", 1)[0] for entry in environment]
    require(
        len(set(environment_names)) == len(environment_names),
        f"PID {pid} environment has duplicate names",
    )
    return {
        "kernel_executable_path": executable,
        "argv": argv,
        "argv_sha256": sha256_canonical(argv),
        "environment_entry_count": len(environment),
        "environment_variable_names": sorted(environment_names),
        "environment_sha256": sha256_canonical(environment),
        "environment_sorted_sha256": sha256_canonical(sorted(environment)),
        "disclosed_policy_environment": {
            name: entry.split("=", 1)[1]
            for entry in environment
            for name in [entry.split("=", 1)[0]]
            if name == "OMNIINFER_REQUEST_HISTORY"
        },
        "kern_procargs2_raw_size_bytes": len(raw),
        "kern_procargs2_raw_sha256": sha256_bytes(raw),
    }


def _parse_lsof_entries(raw: bytes, expected_pid: int) -> list[dict[str, Any]]:
    try:
        lines = raw.decode("utf-8", errors="strict").splitlines()
    except UnicodeDecodeError as error:
        raise CampaignError("lsof output is not strict UTF-8") from error
    require(lines and lines[0] == f"p{expected_pid}", "lsof PID header drifted")
    entries: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    field_names = {
        "f": "fd",
        "a": "access",
        "t": "type",
        "D": "device_text",
        "i": "inode_text",
        "s": "size_text",
        "n": "path",
    }
    for line in lines[1:]:
        require(line, "lsof output contains an empty field line")
        code, value = line[0], line[1:]
        require(code in field_names, f"lsof output contains unknown field: {code}")
        if code == "f":
            if current is not None:
                entries.append(current)
            current = {"fd": value}
        else:
            require(current is not None, "lsof field preceded its descriptor")
            name = field_names[code]
            require(name not in current, f"lsof descriptor duplicates {name}")
            current[name] = value
    if current is not None:
        entries.append(current)
    require(entries, f"lsof returned no entries for PID {expected_pid}")
    return entries


def lsof_entries(pid: int) -> list[dict[str, Any]]:
    result = hardened_command_runner(
        ["/usr/sbin/lsof", "-nP", "-a", "-p", str(pid), "-FpfatDins"],
        Path("/"),
        30.0,
    )
    require(result["returncode"] == 0, f"lsof failed for PID {pid}")
    require(result["stderr"] == b"", f"lsof wrote stderr for PID {pid}")
    return _parse_lsof_entries(result["stdout"], pid)


def code_signature_receipt(path: str) -> dict[str, Any]:
    result = hardened_command_runner(
        ["/usr/bin/codesign", "-d", "--verbose=4", path], Path("/"), 30.0
    )
    output = result["stdout"] + result["stderr"]
    require(len(output) <= 1024 * 1024, f"codesign output is oversized: {path}")
    try:
        text = output.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise CampaignError(f"codesign output is not UTF-8: {path}") from error
    if result["returncode"] == 0:
        identifier = re.search(r"^Identifier=(.+)$", text, re.MULTILINE)
        cdhash = re.search(r"^CDHash=([0-9a-fA-F]+)$", text, re.MULTILINE)
        require(
            identifier is not None and cdhash is not None,
            f"signed identity is incomplete: {path}",
        )
        return {
            "state": "signed",
            "identifier": identifier.group(1),
            "cdhash": cdhash.group(1).lower(),
            "codesign_output_sha256": sha256_bytes(output),
        }
    require(
        "code object is not signed at all" in text,
        f"code signature state is ambiguous: {path}",
    )
    return {
        "state": "explicitly-unsigned",
        "codesign_output_sha256": sha256_bytes(output),
    }


def _unpinned_executable_custody(path: str) -> dict[str, Any]:
    raw_before, receipt_before = _read_regular_no_follow(path, 512 * 1024 * 1024)
    require(
        raw_before or receipt_before["size_bytes"] == 0,
        f"cannot read executable: {path}",
    )
    signature = code_signature_receipt(path)
    raw_after, receipt_after = _read_regular_no_follow(path, 512 * 1024 * 1024)
    require(
        raw_before == raw_after and receipt_before == receipt_after,
        f"executable changed across code-signature observation: {path}",
    )
    return {
        **receipt_after,
        "code_signature_identity_or_explicit_unsigned_state": signature,
        "O_NOFOLLOW_identity_and_bytes_stable_across_code_signature": True,
    }


def _lsof_text_vnode(entry: dict[str, Any], path: str) -> dict[str, int]:
    device_text = entry.get("device_text")
    inode_text = entry.get("inode_text")
    size_text = entry.get("size_text")
    require(
        isinstance(device_text, str)
        and re.fullmatch(r"(?:0[xX][0-9a-fA-F]+|[0-9]+)", device_text) is not None,
        f"lsof text mapping device is absent or invalid: {path}",
    )
    require(
        isinstance(inode_text, str) and re.fullmatch(r"[0-9]+", inode_text),
        f"lsof text mapping inode is absent or invalid: {path}",
    )
    require(
        isinstance(size_text, str) and re.fullmatch(r"[0-9]+", size_text),
        f"lsof text mapping size is absent or invalid: {path}",
    )
    device = (
        int(device_text, 16)
        if device_text.lower().startswith("0x")
        else int(device_text, 10)
    )
    return {
        "device": device,
        "inode": int(inode_text, 10),
        "size_bytes": int(size_text, 10),
    }


def runtime_closure(pid: int, expected_executable: str) -> dict[str, Any]:
    start_before = process_start_identity(pid)
    entries = lsof_entries(pid)
    observed_paths: list[str] = []
    observed_vnodes: dict[str, dict[str, int]] = {}
    for entry in entries:
        path = entry.get("path")
        if not (
            entry.get("fd") == "txt"
            and entry.get("type") == "REG"
            and isinstance(path, str)
            and path.startswith("/")
            and not path.startswith(("/System/Library/", "/usr/lib/"))
        ):
            continue
        canonical = str(Path(path).resolve(strict=True))
        vnode = _lsof_text_vnode(entry, canonical)
        if canonical in observed_vnodes:
            require(
                observed_vnodes[canonical] == vnode,
                f"lsof reported conflicting text vnodes for {canonical}",
            )
        else:
            observed_paths.append(path)
            observed_vnodes[canonical] = vnode
    canonical_expected = str(Path(expected_executable).resolve(strict=True))
    paths = sorted(observed_vnodes)
    if canonical_expected not in paths:
        raise_preflight_blocker(
            "MACOS_RUNTIME_IMAGE_CLOSURE_AMBIGUOUS",
            f"lsof text mappings did not contain the exact executable for PID {pid}",
            {
                "pid": pid,
                "expected_executable": canonical_expected,
                "observed_text_paths": observed_paths,
            },
        )
    closure: list[dict[str, Any]] = []
    for path in paths:
        file_receipt = _unpinned_executable_custody(path)
        vnode = observed_vnodes[path]
        require(
            {
                "device": file_receipt["device"],
                "inode": file_receipt["inode"],
                "size_bytes": file_receipt["size_bytes"],
            }
            == vnode,
            f"live path bytes differ from the lsof-loaded text vnode: {path}",
        )
        closure.append(
            {
                "loaded_image_path": path,
                "lsof_text_vnode": vnode,
                "file": file_receipt,
            }
        )
    start_after = process_start_identity(pid)
    require(
        start_after == start_before,
        f"PID {pid} changed while binding its loaded image closure",
    )
    return {
        "scope": "lsof-txt-regular-files-excluding-/System/Library-and-/usr/lib",
        "process_start_identity_before": copy.deepcopy(start_before),
        "process_start_identity_after": copy.deepcopy(start_after),
        "loaded_image_paths_and_sha256": closure,
        "runtime_closure_sha256": sha256_canonical(closure),
    }


MODEL_FD_IDENTITY_FIELDS = (
    "device",
    "inode",
    "mode",
    "link_count",
    "size_bytes",
    "ctime_ns",
)


def _model_fd_identity(value: dict[str, Any]) -> dict[str, int]:
    result: dict[str, int] = {}
    for field in MODEL_FD_IDENTITY_FIELDS:
        item = value.get(field)
        require(is_int(item) and item >= 0, f"model FD {field} is invalid")
        result[field] = item
    require(stat.S_ISREG(result["mode"]), "model FD is not a regular file")
    require(result["link_count"] == 1, "model FD is not single-link")
    return result


def backend_loaded_model_fd_libproc(
    pid: int,
    controller_observation: dict[str, Any],
    expected_path: str = MODEL_PATH,
) -> dict[str, Any]:
    """Prove one backend FD from libproc without spawning a daemon-side helper."""

    require(
        isinstance(expected_path, str)
        and expected_path.startswith("/")
        and os.path.normpath(expected_path) == expected_path,
        "model FD expected path is invalid",
    )
    controller_identity = _model_fd_identity(controller_observation)
    require(
        controller_observation.get("sha256") == MODEL_SHA256
        and controller_identity["size_bytes"] == MODEL_SIZE,
        "controller model artifact hash or size drifted",
    )
    proc_matches = [
        entry
        for entry in proc_vnode_fd_entries(pid)
        if entry.get("path") == expected_path
        or all(
            entry.get(field) == controller_identity[field]
            for field in MODEL_FD_IDENTITY_FIELDS
        )
    ]
    if len(proc_matches) != 1:
        raise_preflight_blocker(
            "MACOS_LOADED_MODEL_FD_IDENTITY_AMBIGUOUS",
            f"backend PID {pid} did not expose exactly one libproc model descriptor",
            {
                "pid": pid,
                "model_path": expected_path,
                "libproc_match_count": len(proc_matches),
            },
        )
    proc_entry = proc_matches[0]
    require(
        proc_entry["path"] == expected_path,
        "backend model FD path and file identity did not identify the same vnode",
    )
    require(
        proc_entry["open_flags"] & 3 == 1,
        "backend model FD lacks the exact libproc FREAD-without-FWRITE state",
    )
    require(
        _model_fd_identity(proc_entry) == controller_identity,
        "backend libproc model FD differs from controller-held FD",
    )
    return {
        **copy.deepcopy(proc_entry),
        "access": "libproc-FREAD-without-FWRITE (O_RDONLY)",
        "artifact_sha256_via_controller_fd": controller_observation["sha256"],
    }


def backend_loaded_model_fd_proof(
    pid: int,
    controller_observation: dict[str, Any],
    expected_path: str = MODEL_PATH,
) -> dict[str, Any]:
    """Cross-check one backend model FD through independent lsof and libproc views."""

    controller_identity = _model_fd_identity(controller_observation)
    proc_entry = backend_loaded_model_fd_libproc(
        pid, controller_observation, expected_path
    )

    lsof_matches = [
        entry
        for entry in lsof_entries(pid)
        if entry.get("path") == expected_path
        and isinstance(entry.get("fd"), str)
        and re.fullmatch(r"[0-9]+", entry["fd"]) is not None
    ]
    if len(lsof_matches) != 1:
        raise_preflight_blocker(
            "MACOS_LOADED_MODEL_FD_LSOF_AMBIGUOUS",
            f"backend PID {pid} did not expose exactly one lsof model descriptor",
            {
                "pid": pid,
                "model_path": expected_path,
                "lsof_match_count": len(lsof_matches),
            },
        )
    lsof_entry = lsof_matches[0]
    require(
        lsof_entry.get("type") == "REG" and lsof_entry.get("access") == "r",
        "lsof backend model FD is not a read-only regular file",
    )
    try:
        lsof_identity = {
            "fd": int(lsof_entry["fd"]),
            "device": int(lsof_entry["device_text"], 0),
            "inode": int(lsof_entry["inode_text"]),
            "size_bytes": int(lsof_entry["size_text"]),
            "path": lsof_entry["path"],
            "access": "read-only",
            "type": "REG",
        }
    except (KeyError, ValueError) as error:
        raise CampaignError("lsof loaded-model FD fields are invalid") from error
    require(
        lsof_identity
        == {
            "fd": proc_entry["fd"],
            "device": proc_entry["device"],
            "inode": proc_entry["inode"],
            "size_bytes": proc_entry["size_bytes"],
            "path": proc_entry["path"],
            "access": "read-only",
            "type": "REG",
        },
        "lsof and libproc disagree about the backend model FD",
    )
    return {
        "proof_format": "macos-controller-backend-model-fd-crosscheck-v3",
        "backend_pid": pid,
        "controller_preload_fd_identity": controller_identity,
        "backend_loaded_fd": proc_entry,
        "lsof_observation": copy.deepcopy(lsof_identity),
        "lsof_libproc_agree": True,
        "controller_backend_file_identity_equal": True,
    }


def loaded_model_fd(
    pid: int, model_expectation: dict[str, Any], model_observation: dict[str, Any]
) -> dict[str, Any]:
    proof = backend_loaded_model_fd_proof(
        pid, model_observation, model_expectation["absolute_path"]
    )
    return copy.deepcopy(proof["backend_loaded_fd"])


def validate_controller_launch_sequence(receipt: dict[str, Any]) -> None:
    fields = (
        "controller_fd_open_started_monotonic_ns",
        "controller_fd_custody_complete_monotonic_ns",
        "gateway_spawn_invocation_monotonic_ns",
        "gateway_kernel_identity_observed_monotonic_ns",
        "backend_kernel_identity_observed_monotonic_ns",
    )
    values = [receipt.get(field) for field in fields]
    require(
        all(is_int(value) and value >= 0 for value in values)
        and values == sorted(values)
        and len(set(values)) == len(values),
        "controller FD was not proven complete before gateway/backend launch",
    )


def listener_pid(port: int, label: str) -> int:
    result = hardened_command_runner(
        [
            "/usr/sbin/lsof",
            "-nP",
            "-t",
            f"-iTCP:{port}",
            "-sTCP:LISTEN",
        ],
        Path("/"),
        15.0,
    )
    require(
        result["returncode"] == 0 and result["stderr"] == b"",
        f"{label} listener is absent",
    )
    try:
        values = [int(line) for line in result["stdout"].splitlines() if line]
    except ValueError as error:
        raise CampaignError(f"{label} listener PID is malformed") from error
    require(len(values) == 1 and values[0] > 0, f"{label} listener PID is ambiguous")
    return values[0]


def _process_parent_pid(pid: int) -> int:
    raw = _run_checked(["/bin/ps", "-p", str(pid), "-o", "ppid="])
    try:
        text = raw.decode("ascii", errors="strict").strip()
    except UnicodeDecodeError as error:
        raise CampaignError(f"parent PID is not ASCII for {pid}") from error
    require(text.isdigit(), f"parent PID is invalid for {pid}")
    return int(text)


def expected_backend_launch_args(plan: dict[str, Any], backend_port: int) -> list[str]:
    runtime = plan["runtime"]
    backend = plan["artifacts"]["gateway_backend"]["absolute_path"]
    return [
        backend,
        "-m",
        MODEL_PATH,
        "--host",
        "127.0.0.1",
        "--port",
        str(backend_port),
        "--no-webui",
        "--slot-save-path",
        runtime["runtime_logs_path"],
        "--slot-prompt-similarity",
        "0",
        "--cache-idle-slots",
        "-ngl",
        "999",
        "--cache-ram",
        "8192",
        "-ngl",
        "999",
        "-b",
        "13",
        "-ub",
        "13",
        "-t",
        "4",
        "-tb",
        "4",
        "-np",
        "1",
        "--cache-ram",
        "0",
        "--slots",
        "--no-cache-idle-slots",
        "--no-cache-prompt",
        "--slot-prompt-similarity",
        "0",
        "--slot-save-path",
        runtime["slot_save_path"],
        "-c",
        "256",
    ]


def validate_gateway_state(
    state: dict[str, Any], plan: dict[str, Any]
) -> dict[str, Any]:
    receipt_require(
        isinstance(state, dict) and set(state) == GATEWAY_LOADED_STATE_FIELDS_V3,
        "loaded gateway state schema drifted",
    )
    receipt_require(state.get("backend") == "llama.cpp-mac", "gateway backend drifted")
    receipt_require(state.get("backend_ready") is True, "gateway backend is not ready")
    receipt_require(state.get("model_path") == MODEL_PATH, "gateway model path drifted")
    receipt_require(
        state.get("default_model") == MODEL_PATH, "gateway default model drifted"
    )
    receipt_require(
        state.get("model") == MODEL_PATH
        and state.get("public_model_id") is None
        and state.get("owner_admin_id") == "local",
        "gateway loaded model identity drifted",
    )
    receipt_require(state.get("mmproj") is None, "gateway mmproj is not null")
    receipt_require(state.get("ctx_size") == 256, "gateway context drifted")
    receipt_require(
        state.get("request_defaults") == {}, "gateway request defaults drifted"
    )
    receipt_require(
        state.get("effective_parameters") == {}, "gateway effective parameters drifted"
    )
    receipt_require(state.get("proxy_model") is None, "gateway proxy model drifted")
    receipt_require(
        state.get("public_model_id") is None, "gateway public model ID drifted"
    )
    backend_pid = state.get("backend_pid")
    backend_port = state.get("backend_port")
    receipt_require(is_int(backend_pid) and backend_pid > 0, "backend PID is absent")
    receipt_require(is_int(backend_port) and backend_port > 0, "backend port is absent")
    endpoint = f"http://127.0.0.1:{backend_port}"
    receipt_require(
        state.get("client_endpoint") == endpoint, "backend endpoint drifted"
    )
    command = expected_backend_launch_args(plan, backend_port)
    launch_args = command[10:-2]
    expected_log_path = expected_gateway_backend_log_path(plan)
    receipt_require(
        state.get("launch_command") == command, "backend launch command drifted"
    )
    receipt_require(
        state.get("launch_args") == launch_args,
        "backend launch args drifted",
    )
    receipt_require(
        state.get("runtime_mode") == "external_server"
        and state.get("generation") == 1
        and state.get("route_state") == "ready"
        and state.get("allocation_id") == 1
        and state.get("resource_budget") == _expected_gateway_resource_budget()
        and state.get("speculative_admission") is None
        and state.get("cuda_visible_devices") is None
        and state.get("warning") is None
        and state.get("external_server_protocol") == "llama.cpp-server"
        and state.get("openai_compatible") is True
        and state.get("backend_log") == expected_log_path
        and state.get("log_path") == expected_log_path,
        "gateway loaded runtime metadata drifted",
    )
    runtime = state.get("runtime")
    receipt_require(
        isinstance(runtime, dict)
        and set(runtime)
        == {
            "mode",
            "host",
            "port",
            "pid",
            "cuda_visible_devices",
            "launch_command",
            "log_path",
            "proxy_model_ref",
            "external_server_protocol",
            "client_endpoint",
            "openai_compatible",
        },
        "gateway runtime object schema drifted",
    )
    receipt_require(
        runtime.get("mode") == "external_server"
        and runtime.get("host") == "127.0.0.1"
        and runtime.get("pid") == backend_pid
        and runtime.get("port") == backend_port
        and runtime.get("cuda_visible_devices") is None
        and runtime.get("launch_command") == command
        and runtime.get("log_path") == expected_log_path
        and runtime.get("proxy_model_ref") is None
        and runtime.get("external_server_protocol") == "llama.cpp-server"
        and runtime.get("client_endpoint") == endpoint
        and runtime.get("openai_compatible") is True,
        "gateway runtime backend identity drifted",
    )
    loaded = state.get("loaded_models")
    receipt_require(
        isinstance(loaded, list) and len(loaded) == 1, "loaded-model set drifted"
    )
    receipt_require(
        isinstance(loaded[0], dict)
        and set(loaded[0])
        == {
            "id",
            "owner_admin_id",
            "backend",
            "model",
            "model_path",
            "public_model_id",
            "mmproj",
            "ctx_size",
            "request_defaults",
            "runtime_mode",
            "backend_pid",
            "backend_port",
            "generation",
            "route_state",
            "allocation_id",
            "resource_budget",
            "speculative_admission",
            "launch_args",
            "cuda_visible_devices",
            "warning",
            "launch_command",
            "proxy_model",
            "external_server_protocol",
            "client_endpoint",
            "openai_compatible",
            "backend_log",
        }
        and loaded[0]
        == {
            "id": MODEL_PATH,
            "owner_admin_id": "local",
            "backend": "llama.cpp-mac",
            "model": MODEL_PATH,
            "model_path": MODEL_PATH,
            "public_model_id": None,
            "mmproj": None,
            "ctx_size": 256,
            "request_defaults": {},
            "runtime_mode": "external_server",
            "backend_pid": backend_pid,
            "backend_port": backend_port,
            "generation": 1,
            "route_state": "ready",
            "allocation_id": 1,
            "resource_budget": _expected_gateway_resource_budget(),
            "speculative_admission": None,
            "launch_args": launch_args,
            "cuda_visible_devices": None,
            "warning": None,
            "launch_command": command,
            "proxy_model": None,
            "external_server_protocol": "llama.cpp-server",
            "client_endpoint": endpoint,
            "openai_compatible": True,
            "backend_log": expected_log_path,
        },
        "loaded-model receipt drifted",
    )
    restore = state.get("restore_selection")
    receipt_require(
        isinstance(restore, dict)
        and set(restore)
        == {"backend", "model", "mmproj", "no_mmproj", "ctx_size", "request_defaults"}
        and restore.get("backend") == "llama.cpp-mac"
        and restore.get("model") == MODEL_PATH
        and restore.get("mmproj") is None
        and restore.get("no_mmproj") is True
        and restore.get("ctx_size") == 256
        and restore.get("request_defaults") == {},
        "gateway restore selection drifted",
    )
    receipt_require(
        state.get("restore_status") == "loaded"
        and state.get("restore_completed") is True,
        "gateway restore completion drifted",
    )
    resource_ledger = state.get("resource_ledger")
    receipt_require(
        isinstance(resource_ledger, dict)
        and set(resource_ledger)
        == {
            "capacity_snapshot_id",
            "capacity_bytes",
            "reserved_bytes",
            "committed_bytes",
            "available_bytes",
        }
        and resource_ledger.get("capacity_snapshot_id") == 1
        and resource_ledger.get("reserved_bytes") == {}
        and resource_ledger.get("committed_bytes")
        == _expected_gateway_resource_budget()["domains_bytes"],
        "gateway loaded resource ledger drifted",
    )
    capacity = resource_ledger["capacity_bytes"]
    available = resource_ledger["available_bytes"]
    committed = resource_ledger["committed_bytes"]
    receipt_require(
        isinstance(capacity, dict)
        and isinstance(available, dict)
        and set(capacity) == set(available) == set(committed)
        and all(
            isinstance(domain, str)
            and is_int(value)
            and value >= committed[domain]
            and available[domain] == value - committed[domain]
            for domain, value in capacity.items()
        ),
        "gateway loaded resource-ledger arithmetic drifted",
    )
    available_backends = state.get("available_backends")
    receipt_require(
        isinstance(available_backends, list)
        and any(
            isinstance(entry, dict) and entry.get("id") == "llama.cpp-mac"
            for entry in available_backends
        ),
        "gateway available-backend registry lacks llama.cpp-mac",
    )
    return {
        "backend": "llama.cpp-mac",
        "backend_ready": True,
        "model_path": MODEL_PATH,
        "ctx_size": 256,
        "backend_pid": backend_pid,
        "backend_port": backend_port,
        "client_endpoint": endpoint,
        "model": MODEL_PATH,
        "owner_admin_id": "local",
        "generation": 1,
        "route_state": "ready",
        "allocation_id": 1,
        "resource_budget": _expected_gateway_resource_budget(),
        "resource_ledger": copy.deepcopy(resource_ledger),
        "loaded_models": copy.deepcopy(loaded),
        "restore_selection": copy.deepcopy(restore),
        "restore_status": "loaded",
        "restore_completed": True,
        "runtime": copy.deepcopy(runtime),
        "runtime_mode": "external_server",
        "external_server_protocol": "llama.cpp-server",
        "openai_compatible": True,
        "backend_log": expected_log_path,
        "available_backends": copy.deepcopy(available_backends),
        "launch_command": command,
        "request_defaults": {},
        "effective_parameters": {},
        "proxy_model": None,
    }


def validate_backend_props(props: dict[str, Any]) -> dict[str, Any]:
    receipt_require(
        props.get("build_info") == BACKEND_BUILD_INFO, "backend build info drifted"
    )
    receipt_require(
        props.get("model_path") == MODEL_PATH, "backend props model drifted"
    )
    receipt_require(props.get("model_ftype") == "Q8_0", "backend model type drifted")
    receipt_require(
        props.get("endpoint_slots") is True, "backend slots endpoint is disabled"
    )
    receipt_require(props.get("total_slots") == 1, "backend slot count drifted")
    settings = props.get("default_generation_settings")
    receipt_require(
        isinstance(settings, dict) and settings.get("n_ctx") == 256,
        "backend props context drifted",
    )
    return {
        "build_info": BACKEND_BUILD_INFO,
        "model_path": MODEL_PATH,
        "model_ftype": "Q8_0",
        "endpoint_slots": True,
        "total_slots": 1,
        "n_ctx": 256,
    }


def validate_health(health: dict[str, Any]) -> dict[str, Any]:
    receipt_require(health.get("status") == "ok", "gateway health is not ok")
    backend = health.get("backend_health")
    receipt_require(
        isinstance(backend, dict) and backend.get("status") == "ok",
        "gateway deep backend health is not ok",
    )
    return {"status": "ok", "backend_status": "ok"}


def gateway_model_select_launch_args(plan: dict[str, Any]) -> list[str]:
    return [
        "-ngl",
        "999",
        "-b",
        "13",
        "-ub",
        "13",
        "-t",
        "4",
        "-tb",
        "4",
        "-np",
        "1",
        "--cache-ram",
        "0",
        "--slots",
        "--no-cache-idle-slots",
        "--no-cache-prompt",
        "--slot-prompt-similarity",
        "0",
        "--slot-save-path",
        plan["runtime"]["slot_save_path"],
    ]


def gateway_model_select_request(plan: dict[str, Any]) -> dict[str, Any]:
    return {
        "backend": "llama.cpp-mac",
        "model": MODEL_PATH,
        "no_mmproj": True,
        "ctx_size": 256,
        "request_defaults": {},
        "launch_args": gateway_model_select_launch_args(plan),
    }


def gateway_model_select_request_bytes(plan: dict[str, Any]) -> bytes:
    return canonical_json_bytes(gateway_model_select_request(plan))


def expected_gateway_backend_log_path(plan: dict[str, Any]) -> str:
    sanitized = "".join(
        character
        if character.isascii() and (character.isalnum() or character in (".", "_", "-"))
        else "_"
        for character in MODEL_PATH
    )
    return str(Path(plan["runtime"]["runtime_logs_path"]) / f"runtime-{sanitized}.log")


def validate_gateway_preload_state(state: dict[str, Any]) -> dict[str, Any]:
    receipt_require(
        isinstance(state, dict) and set(state) == GATEWAY_PRELOAD_STATE_FIELDS_V3,
        "gateway pre-load state schema drifted",
    )
    receipt_require(
        state.get("backend") is None
        and state.get("backend_ready") is False
        and state.get("model") is None
        and state.get("public_model_id") is None
        and state.get("mmproj") is None
        and state.get("ctx_size") is None
        and state.get("request_defaults") == {}
        and state.get("runtime_mode") is None
        and state.get("backend_pid") is None
        and state.get("backend_port") is None
        and state.get("launch_args") == []
        and state.get("cuda_visible_devices") is None
        and state.get("warning") is None
        and state.get("launch_command") == []
        and state.get("proxy_model") is None
        and state.get("external_server_protocol") is None
        and state.get("client_endpoint") is None
        and state.get("openai_compatible") is False
        and state.get("backend_log") is None
        and state.get("effective_parameters") == {}
        and state.get("runtime") is None
        and state.get("default_model") is None
        and state.get("loaded_models") == []
        and state.get("restore_selection") is None
        and state.get("restore_status") == "not_configured"
        and state.get("restore_completed") is False,
        "hidden gateway was not in the exact fresh unloaded state",
    )
    receipt_require(
        state.get("resource_ledger") is None,
        "fresh gateway unexpectedly has a resource ledger",
    )
    available_backends = state.get("available_backends")
    receipt_require(
        isinstance(available_backends, list)
        and any(
            isinstance(entry, dict) and entry.get("id") == "llama.cpp-mac"
            for entry in available_backends
        ),
        "fresh gateway available-backend registry lacks llama.cpp-mac",
    )
    return {
        "backend": None,
        "backend_ready": False,
        "backend_pid": None,
        "backend_port": None,
        "model": None,
        "public_model_id": None,
        "mmproj": None,
        "ctx_size": None,
        "request_defaults": {},
        "runtime_mode": None,
        "launch_args": [],
        "launch_command": [],
        "client_endpoint": None,
        "effective_parameters": {},
        "runtime": None,
        "loaded_models": [],
        "restore_status": "not_configured",
        "restore_completed": False,
        "resource_ledger": None,
        "available_backends": copy.deepcopy(available_backends),
    }


def _expected_gateway_resource_budget() -> dict[str, Any]:
    return {
        "domains_bytes": {"unified:system": 1_784_921_600},
        "components": [
            {
                "name": "weights",
                "domain": "unified:system",
                "bytes": 811_843_072,
            },
            {
                "name": "kv_cache",
                "domain": "unified:system",
                "bytes": 268_435_456,
            },
            {
                "name": "activation",
                "domain": "unified:system",
                "bytes": 134_217_728,
            },
            {
                "name": "framework_overhead",
                "domain": "unified:system",
                "bytes": 402_653_184,
            },
            {
                "name": "allocator_slack",
                "domain": "unified:system",
                "bytes": 167_772_160,
            },
        ],
    }


def validate_gateway_model_select_response(
    response: dict[str, Any], plan: dict[str, Any]
) -> dict[str, Any]:
    required = {
        "ok",
        "already_loaded",
        "requires_reload",
        "model",
        "owner_admin_id",
        "selected_backend",
        "selected_model",
        "selected_public_model_id",
        "selected_mmproj",
        "selected_ctx_size",
        "request_defaults",
        "backend_pid",
        "backend_port",
        "generation",
        "route_state",
        "allocation_id",
        "resource_budget",
        "speculative_admission",
        "launch_command",
        "log_path",
        "external_server_protocol",
        "client_endpoint",
        "openai_compatible",
    }
    receipt_require(
        isinstance(response, dict) and set(response) == required,
        "OmniInfer model-select response schema drifted",
    )
    backend_pid = response["backend_pid"]
    backend_port = response["backend_port"]
    receipt_require(
        response["ok"] is True
        and response["already_loaded"] is False
        and response["requires_reload"] is False
        and response["model"] == MODEL_PATH
        and response["owner_admin_id"] == "local"
        and response["selected_backend"] == "llama.cpp-mac"
        and response["selected_model"] == MODEL_PATH
        and response["selected_public_model_id"] is None
        and response["selected_mmproj"] is None
        and response["selected_ctx_size"] == 256
        and response["request_defaults"] == {}
        and is_int(backend_pid)
        and backend_pid > 0
        and is_int(backend_port)
        and 0 < backend_port <= 65_535
        and response["generation"] == 1
        and response["route_state"] == "ready"
        and response["allocation_id"] == 1
        and response["resource_budget"] == _expected_gateway_resource_budget()
        and response["speculative_admission"] is None
        and response["launch_command"]
        == expected_backend_launch_args(plan, backend_port)
        and response["log_path"] == expected_gateway_backend_log_path(plan)
        and response["external_server_protocol"] == "llama.cpp-server"
        and response["client_endpoint"] == f"http://127.0.0.1:{backend_port}"
        and response["openai_compatible"] is True,
        "OmniInfer model-select response semantics drifted",
    )
    return {
        "backend_pid": backend_pid,
        "backend_port": backend_port,
        "runtime_generation": 1,
        "allocation_id": 1,
        "route_state": "ready",
        "launch_command": copy.deepcopy(response["launch_command"]),
        "client_endpoint": response["client_endpoint"],
    }


def validate_zero_generation_management_transport(
    transport: dict[str, Any],
    base_url: str,
    label: str,
    method: str,
    path: str,
    body: bytes | None,
    response: dict[str, Any],
) -> dict[str, Any]:
    receipt_require(
        method in ("GET", "POST")
        and path in ("/omni/state", "/omni/model/select")
        and isinstance(transport, dict),
        "zero-generation management transport endpoint drifted",
    )
    parsed = urllib.parse.urlsplit(base_url)
    receipt_require(
        parsed.scheme == "http"
        and parsed.hostname == "127.0.0.1"
        and parsed.port is not None,
        "zero-generation management transport base URL drifted",
    )
    headers = [
        f"{method} {path} HTTP/1.1",
        f"Host: 127.0.0.1:{parsed.port}",
        "Accept: application/json",
        "Connection: keep-alive",
    ]
    if body is not None:
        headers.extend(
            ["Content-Type: application/json", f"Content-Length: {len(body)}"]
        )
    body_bytes = body or b""
    expected_wire = ("\r\n".join(headers) + "\r\n\r\n").encode("ascii") + body_bytes
    try:
        observed_wire = base64.b64decode(
            transport.get("request_wire_base64", "").encode("ascii"), validate=True
        )
        response_bytes = base64.b64decode(
            transport.get("response_base64", "").encode("ascii"), validate=True
        )
    except (AttributeError, UnicodeEncodeError, binascii.Error, ValueError) as error:
        raise ReceiptError(
            "zero-generation management transport bytes are not strict base64"
        ) from error
    body_offset = len(expected_wire) - len(body_bytes)
    receipt_require(
        transport.get("connection") == label
        and transport.get("method") == method
        and transport.get("path") == path
        and transport.get("request_body_size_bytes") == len(body_bytes)
        and transport.get("request_body_sha256")
        == (sha256_bytes(body) if body is not None else None)
        and observed_wire == expected_wire
        and transport.get("request_wire_size_bytes") == len(expected_wire)
        and transport.get("request_wire_sha256") == sha256_bytes(expected_wire)
        and transport.get("request_wire_body_offset_bytes") == body_offset
        and transport.get("request_wire_body_size_bytes") == len(body_bytes)
        and transport.get("request_wire_body_sha256") == sha256_bytes(body_bytes)
        and transport.get("request_wire_body_equals_request_body") is True
        and transport.get("single_sendall_call_count") == 1
        and transport.get("single_sendall_argument_size_bytes") == len(expected_wire)
        and transport.get("single_sendall_argument_sha256")
        == sha256_bytes(expected_wire)
        and transport.get("status") == 200
        and transport.get("http_version") == 11
        and transport.get("response_size_bytes") == len(response_bytes)
        and transport.get("response_sha256") == sha256_bytes(response_bytes)
        and parse_strict_json_document(response_bytes) == response
        and transport.get("request_serialization_before_start") is True
        and transport.get("first_wire_byte_send_call_immediately_after_start") is True
        and transport.get("complete_HTTP_request_wire_serialization_before_start")
        is True
        and transport.get("single_sendall_call_for_complete_request_wire_required")
        is True
        and transport.get("full_response_body_read_before_end") is True
        and transport.get("strict_json_parse_before_end") is True
        and transport.get("semantic_validation_before_end") is True,
        "zero-generation management transport receipt drifted",
    )
    return {
        "connection": label,
        "method": method,
        "path": path,
        "request_body_size_bytes": len(body_bytes),
        "request_body_sha256": sha256_bytes(body) if body is not None else None,
        "request_wire_size_bytes": len(expected_wire),
        "request_wire_sha256": sha256_bytes(expected_wire),
        "request_wire_base64": transport["request_wire_base64"],
        "request_wire_body_offset_bytes": body_offset,
        "request_wire_body_size_bytes": len(body_bytes),
        "request_wire_body_sha256": sha256_bytes(body_bytes),
        "request_wire_body_equals_request_body": True,
        "single_sendall_call_count": 1,
        "single_sendall_argument_size_bytes": len(expected_wire),
        "single_sendall_argument_sha256": sha256_bytes(expected_wire),
        "status": 200,
        "http_version": 11,
        "response_size_bytes": len(response_bytes),
        "response_sha256": sha256_bytes(response_bytes),
        "response_base64": transport["response_base64"],
        "request_serialization_before_start": True,
        "first_wire_byte_send_call_immediately_after_start": True,
        "complete_HTTP_request_wire_serialization_before_start": True,
        "single_sendall_call_for_complete_request_wire_required": True,
        "full_response_body_read_before_end": True,
        "strict_json_parse_before_end": True,
        "semantic_validation_before_end": True,
        "setup_only_zero_generation_management_request": True,
    }


def admit_zero_generation_model_load(
    plan: dict[str, Any],
    gateway_process: Any,
    gateway_start: dict[str, int],
    *,
    expected_gateway_parent_pid: int | None = None,
    request_json: Any | None = None,
    process_start_reader: Any = process_start_identity,
    parent_pid_reader: Any = _process_parent_pid,
    tree_reader: Any = tree_manifest,
    monotonic: Any = time.monotonic,
    sleeper: Any = time.sleep,
) -> dict[str, Any]:
    """Load the formal backend through the pinned zero-generation management API."""

    runtime = plan["runtime"]
    started = monotonic()
    child_deadline = started + _custodian_child_ready_budget_seconds(runtime)
    listener_deadline = started + float(runtime["custodian_ready_timeout_seconds"])

    def requester(*args: Any) -> tuple[dict[str, Any], Any, dict[str, Any]]:
        if request_json is not None:
            return request_json(*args)
        return _one_shot_json(
            *args,
            timeout_seconds=_remaining_deadline_seconds(
                child_deadline,
                "zero-generation model-load request",
                monotonic=monotonic,
            ),
        )

    gateway_parent_pid_start = parent_pid_reader(gateway_process.pid)
    require(
        is_int(gateway_parent_pid_start)
        and gateway_parent_pid_start > 0
        and (
            expected_gateway_parent_pid is None
            or gateway_parent_pid_start == expected_gateway_parent_pid
        ),
        "gateway was not the expected custodian child before model load",
    )
    history_start = tree_reader(runtime["history_root"])
    management_request_sequence: list[list[str]] = []
    pre_load_raw: dict[str, Any] | None = None
    pre_load_core: dict[str, Any] | None = None
    pre_load_transport: dict[str, Any] | None = None
    while monotonic() < listener_deadline:
        require(
            gateway_process.poll() is None, "gateway exited before listener readiness"
        )
        require(
            process_start_reader(gateway_process.pid) == gateway_start,
            "gateway PID changed before listener readiness",
        )
        try:
            pre_load_raw, _, pre_load_transport = requester(
                runtime["omni_base_url"],
                "custodian-pre-load-state",
                "GET",
                "/omni/state",
                None,
                lambda value: copy.deepcopy(value),
            )
        except OSError:
            sleeper(0.1)
            continue
        management_request_sequence.append(["GET", "/omni/state"])
        pre_load_core = validate_gateway_preload_state(pre_load_raw)
        pre_load_transport = validate_zero_generation_management_transport(
            pre_load_transport,
            runtime["omni_base_url"],
            "custodian-pre-load-state",
            "GET",
            "/omni/state",
            None,
            pre_load_raw,
        )
        break
    if pre_load_raw is None or pre_load_core is None or pre_load_transport is None:
        raise_preflight_blocker(
            "OMNIINFER_GATEWAY_LISTENER_NOT_READY_FOR_ZERO_GENERATION_MODEL_LOAD",
            "hidden OmniInfer gateway did not expose its management listener before timeout",
            {"generation_requests": 0, "management_endpoint": "/omni/state"},
        )

    require(
        gateway_process.poll() is None
        and process_start_reader(gateway_process.pid) == gateway_start
        and parent_pid_reader(gateway_process.pid) == gateway_parent_pid_start,
        "gateway identity changed before zero-generation model select",
    )
    _remaining_deadline_seconds(
        child_deadline,
        "zero-generation model-select start",
        monotonic=monotonic,
    )
    request_object = gateway_model_select_request(plan)
    request_body = gateway_model_select_request_bytes(plan)
    request_receipt = {
        "object": request_object,
        "canonical_utf8": request_body.decode("utf-8"),
        "size_bytes": len(request_body),
        "sha256": sha256_bytes(request_body),
    }
    management_request_sequence.append(["POST", "/omni/model/select"])
    response_raw, response_core, response_transport = requester(
        runtime["omni_base_url"],
        "custodian-zero-generation-model-select",
        "POST",
        "/omni/model/select",
        request_body,
        lambda value: validate_gateway_model_select_response(value, plan),
    )
    _remaining_deadline_seconds(
        child_deadline,
        "zero-generation model-select completion",
        monotonic=monotonic,
    )
    response_revalidated = validate_gateway_model_select_response(response_raw, plan)
    receipt_require(
        response_revalidated == response_core,
        "model-select response changed after transport validation",
    )
    response_core = response_revalidated
    response_transport = validate_zero_generation_management_transport(
        response_transport,
        runtime["omni_base_url"],
        "custodian-zero-generation-model-select",
        "POST",
        "/omni/model/select",
        request_body,
        response_raw,
    )
    response_receipt = {
        "object": copy.deepcopy(response_raw),
        "canonical_sha256": sha256_canonical(response_raw),
        "validated": copy.deepcopy(response_core),
        "runtime_generation_is_not_an_inference_generation": True,
    }

    require(gateway_process.poll() is None, "gateway exited after model-select success")
    require(
        process_start_reader(gateway_process.pid) == gateway_start,
        "gateway PID changed after model-select success",
    )
    _remaining_deadline_seconds(
        child_deadline,
        "post-select exact state request",
        monotonic=monotonic,
    )
    loaded_raw, loaded_core, loaded_transport = requester(
        runtime["omni_base_url"],
        "custodian-post-load-state",
        "GET",
        "/omni/state",
        None,
        lambda value: validate_gateway_state(value, plan),
    )
    _remaining_deadline_seconds(
        child_deadline,
        "post-select exact state completion",
        monotonic=monotonic,
    )
    management_request_sequence.append(["GET", "/omni/state"])
    loaded_revalidated = validate_gateway_state(loaded_raw, plan)
    receipt_require(
        loaded_revalidated == loaded_core,
        "post-select loaded state changed after transport validation",
    )
    loaded_core = loaded_revalidated
    loaded_transport = validate_zero_generation_management_transport(
        loaded_transport,
        runtime["omni_base_url"],
        "custodian-post-load-state",
        "GET",
        "/omni/state",
        None,
        loaded_raw,
    )
    post_load_state_transports = [copy.deepcopy(loaded_transport)]

    backend_pid = loaded_core["backend_pid"]
    require(
        backend_pid == response_core["backend_pid"]
        and loaded_core["backend_port"] == response_core["backend_port"],
        "model-select response and loaded gateway state disagree",
    )
    backend_start = process_start_reader(backend_pid)
    gateway_parent_pid_end = parent_pid_reader(gateway_process.pid)
    require(
        gateway_process.poll() is None
        and process_start_reader(gateway_process.pid) == gateway_start
        and gateway_parent_pid_end == gateway_parent_pid_start
        and (
            expected_gateway_parent_pid is None
            or gateway_parent_pid_end == expected_gateway_parent_pid
        ),
        "gateway identity or parent changed across zero-generation model load",
    )
    require(
        parent_pid_reader(backend_pid) == gateway_process.pid,
        "loaded backend is not the original hidden gateway child",
    )
    history_end = tree_reader(runtime["history_root"])
    require(
        history_end == history_start,
        "request history changed during zero-generation model load",
    )
    return {
        "format": "apxinf-omniinfer-gateway-zero-generation-model-load-v3",
        "schema_version": 3,
        "edge_id": EDGE_ID,
        "management_api": {
            "method": "POST",
            "path": "/omni/model/select",
            "pinned_omniinfer_source_commit": OMNI_SOURCE_COMMIT,
            "alias_used": False,
        },
        "management_request_sequence": management_request_sequence,
        "model_select_request_count": 1,
        "request": request_receipt,
        "response": response_receipt,
        "pre_load_state": {
            "object": copy.deepcopy(pre_load_raw),
            "validated": copy.deepcopy(pre_load_core),
        },
        "loaded_state": {
            "object": copy.deepcopy(loaded_raw),
            "validated": copy.deepcopy(loaded_core),
        },
        "transport": {
            "pre_load_state": copy.deepcopy(pre_load_transport),
            "model_select": copy.deepcopy(response_transport),
            "post_load_state_attempts": post_load_state_transports,
            "loaded_state": copy.deepcopy(loaded_transport),
        },
        "gateway_pid": gateway_process.pid,
        "gateway_start_identity": copy.deepcopy(gateway_start),
        "gateway_end_identity": copy.deepcopy(gateway_start),
        "gateway_process_start_end_equal": True,
        "gateway_parent_pid_start": gateway_parent_pid_start,
        "gateway_parent_pid_end": gateway_parent_pid_end,
        "gateway_parent_start_end_equal": True,
        "backend_pid": backend_pid,
        "backend_start_identity": copy.deepcopy(backend_start),
        "gateway_is_direct_parent_of_backend": True,
        "history_start": history_start,
        "history_end": history_end,
        "history_start_end_equal": True,
        "generation_requests": 0,
        "generation_endpoint_paths_called": [],
        "request_history_records_created": 0,
        "all_passed": True,
    }


def _one_shot_json(
    base_url: str,
    label: str,
    method: str,
    path: str,
    body: bytes | None,
    validator: Any,
    *,
    timeout_seconds: float = 60.0,
) -> tuple[dict[str, Any], Any, dict[str, Any]]:
    connection = PersistentHttpJsonConnection(
        base_url, label, timeout_seconds=timeout_seconds
    )
    connection.connect()
    try:
        payload, validated, transport = connection.request_json(
            method, path, body, validator
        )
        transport["response_base64"] = base64.b64encode(
            connection.last_raw_response
        ).decode("ascii")
        return payload, validated, transport
    finally:
        connection.close()


def collect_process_identity(
    pid: int,
    expected_argv: list[str],
    expected_executable: str,
    *,
    model_expectation: dict[str, Any] | None = None,
    model_observation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    start_before = process_start_identity(pid)
    process = process_argv_environment(pid)
    start_after = process_start_identity(pid)
    require(start_before == start_after, f"PID {pid} changed while collecting argv")
    require(
        process["argv"] == expected_argv, f"PID {pid} argv differs from predeclaration"
    )
    canonical_expected = str(Path(expected_executable).resolve(strict=True))
    require(
        str(Path(process["kernel_executable_path"]).resolve(strict=True))
        == canonical_expected
        and str(Path(expected_argv[0]).resolve(strict=True)) == canonical_expected,
        f"PID {pid} executable differs from predeclared artifact",
    )
    closure = runtime_closure(pid, canonical_expected)
    result = {
        "pid": pid,
        "process_start_identity": start_before,
        "kernel_executable_path": process["kernel_executable_path"],
        "canonical_executable_path": canonical_expected,
        "argv": copy.deepcopy(process["argv"]),
        "argv_sha256": process["argv_sha256"],
        "environment_entry_count": process["environment_entry_count"],
        "environment_variable_names": process["environment_variable_names"],
        "environment_sha256": process["environment_sha256"],
        "environment_sorted_sha256": process["environment_sorted_sha256"],
        "kern_procargs2_raw_size_bytes": process["kern_procargs2_raw_size_bytes"],
        "kern_procargs2_raw_sha256": process["kern_procargs2_raw_sha256"],
        "disclosed_policy_environment": process["disclosed_policy_environment"],
        "runtime_closure": closure["loaded_image_paths_and_sha256"],
        "runtime_closure_scope": closure["scope"],
        "runtime_closure_sha256": closure["runtime_closure_sha256"],
    }
    if model_expectation is not None:
        require(model_observation is not None, "model observation is absent")
        result["loaded_model_fd"] = loaded_model_fd(
            pid, model_expectation, model_observation
        )
    require(
        process_start_identity(pid) == start_before,
        f"PID {pid} changed while collecting runtime closure",
    )
    return result


def _custodian_command(plan_path: Path | str, nonce: str) -> list[str]:
    require(
        re.fullmatch(r"[0-9a-f]{64}", nonce) is not None, "custodian nonce is invalid"
    )
    return [
        sys.executable,
        "-I",
        "-B",
        str(Path(__file__).resolve(strict=True)),
        "_custodian",
        "--plan",
        str(Path(plan_path).resolve(strict=True)),
        "--nonce",
        nonce,
    ]


def _sorted_environment(environment: dict[str, str]) -> dict[str, str]:
    return {name: environment[name] for name in sorted(environment)}


def _environment_entries(environment: dict[str, str]) -> list[str]:
    return [f"{name}={environment[name]}" for name in sorted(environment)]


def _require_exact_process_environment(
    identity: dict[str, Any], expected: dict[str, str], label: str
) -> None:
    entries = _environment_entries(expected)
    require(
        identity["environment_entry_count"] == len(entries)
        and identity["environment_variable_names"] == sorted(expected)
        and identity["environment_sorted_sha256"] == sha256_canonical(entries),
        f"{label} environment differs from exact predeclaration",
    )


def _lsof_numeric_fd(pid: int, fd: int, expected_path: str) -> dict[str, Any]:
    matches = [
        entry
        for entry in lsof_entries(pid)
        if entry.get("fd") == str(fd) and entry.get("path") == expected_path
    ]
    require(len(matches) == 1, f"PID {pid} FD {fd} lsof identity is ambiguous")
    entry = matches[0]
    require(
        entry.get("type") == "REG" and entry.get("access") == "r",
        f"PID {pid} FD {fd} is not a read-only regular file in lsof",
    )
    try:
        return {
            "fd": fd,
            "device": int(entry["device_text"], 0),
            "inode": int(entry["inode_text"]),
            "size_bytes": int(entry["size_text"]),
            "path": entry["path"],
            "access": "read-only",
            "type": "REG",
        }
    except (KeyError, ValueError) as error:
        raise CampaignError(f"PID {pid} FD {fd} lsof fields are invalid") from error


def controller_fd_process_proof(
    controller_pid: int, held: dict[str, Any]
) -> dict[str, Any]:
    require(
        held.get("format") == "apxinf-controller-held-model-fd-v3"
        and held.get("schema_version") == 3
        and held.get("controller_pid") == controller_pid
        and held.get("absolute_path") == MODEL_PATH
        and held.get("open_flags") == ["O_RDONLY", "O_NOFOLLOW", "O_CLOEXEC"]
        and held.get("fd_cloexec_observed") is True
        and held.get("sha256") == MODEL_SHA256,
        "controller-held model FD receipt drifted",
    )
    fd = held.get("fd")
    require(is_int(fd) and fd >= 0, "controller-held model FD number is invalid")
    matches = [
        entry for entry in proc_vnode_fd_entries(controller_pid) if entry["fd"] == fd
    ]
    require(len(matches) == 1, "controller-held FD is absent from libproc")
    proc_entry = matches[0]
    require(
        proc_entry["path"] == MODEL_PATH
        and proc_entry["open_flags"] & 3 == 1
        and proc_entry["close_on_exec"] is True
        and _model_fd_identity(proc_entry) == _model_fd_identity(held),
        "controller-held FD libproc identity or flags drifted",
    )
    lsof_entry = _lsof_numeric_fd(controller_pid, fd, MODEL_PATH)
    require(
        lsof_entry
        == {
            "fd": fd,
            "device": proc_entry["device"],
            "inode": proc_entry["inode"],
            "size_bytes": proc_entry["size_bytes"],
            "path": proc_entry["path"],
            "access": "read-only",
            "type": "REG",
        },
        "controller-held FD lsof/libproc disagreement",
    )
    return {
        "controller_pid": controller_pid,
        "controller_held_fd": copy.deepcopy(held),
        "libproc_observation": copy.deepcopy(proc_entry),
        "lsof_observation": lsof_entry,
        "lsof_libproc_agree": True,
        "O_RDONLY_O_NOFOLLOW_O_CLOEXEC_proven": True,
    }


def _socket_custody(path_value: Path | str) -> dict[str, Any]:
    path = Path(path_value)
    require(path.is_absolute(), "custodian socket path is not absolute")
    observed = path.lstat()
    require(stat.S_ISSOCK(observed.st_mode), "custodian control path is not a socket")
    require(
        stat.S_IMODE(observed.st_mode) == 0o600,
        "custodian control socket mode is not 0600",
    )
    require(observed.st_uid == os.getuid(), "custodian control socket owner drifted")
    return {
        "absolute_path": str(path),
        "device": observed.st_dev,
        "inode": observed.st_ino,
        "mode": observed.st_mode,
        "uid": observed.st_uid,
        "ctime_ns": observed.st_ctime_ns,
    }


def _read_pipe_json_line_with_total_deadline(
    descriptor: int,
    timeout_seconds: float,
    maximum_bytes: int,
) -> dict[str, Any]:
    require(timeout_seconds > 0.0, "pipe receipt timeout is invalid")
    deadline = time.monotonic() + timeout_seconds
    raw = bytearray()
    while b"\n" not in raw and len(raw) <= maximum_bytes:
        remaining = _remaining_deadline_seconds(
            deadline, "custodian readiness receipt", monotonic=time.monotonic
        )
        readable, _, _ = select.select([descriptor], [], [], remaining)
        require(readable, "custodian produced no complete readiness receipt")
        chunk = os.read(descriptor, min(64 * 1024, maximum_bytes + 1 - len(raw)))
        require(chunk, "custodian readiness pipe closed before one complete line")
        raw.extend(chunk)
    require(len(raw) <= maximum_bytes, "custodian readiness is oversized")
    return parse_strict_json_line(bytes(raw))


def _receive_control_request(
    connection: socket.socket,
    timeout_seconds: float,
    *,
    monotonic: Any = time.monotonic,
) -> dict[str, Any]:
    require(timeout_seconds > 0.0, "custodian control read timeout is invalid")
    deadline = monotonic() + timeout_seconds
    raw = bytearray()
    while len(raw) <= CUSTODIAN_CONTROL_REQUEST_MAX_BYTES:
        connection.settimeout(
            _remaining_deadline_seconds(
                deadline,
                "custodian control request read",
                monotonic=monotonic,
            )
        )
        chunk = connection.recv(
            min(64 * 1024, CUSTODIAN_CONTROL_REQUEST_MAX_BYTES + 1 - len(raw))
        )
        if not chunk:
            break
        raw.extend(chunk)
    require(
        len(raw) <= CUSTODIAN_CONTROL_REQUEST_MAX_BYTES,
        "custodian control request is oversized",
    )
    return parse_strict_json_line(bytes(raw))


def _send_control_response(connection: socket.socket, value: dict[str, Any]) -> None:
    connection.sendall(canonical_json_bytes(value) + b"\n")
    connection.shutdown(socket.SHUT_WR)


def _custodian_light_attestation(
    plan: dict[str, Any],
    nonce: str,
    challenge: str,
    stage: str,
    controller: ControllerModelFd,
    gateway_pid: int,
    gateway_start: dict[str, int],
    backend_pid: int,
    backend_start: dict[str, int],
    lifecycle: dict[str, Any],
) -> dict[str, Any]:
    require(
        process_start_identity(os.getpid()) == lifecycle["custodian_start_identity"],
        "custodian PID identity changed",
    )
    require(
        process_start_identity(gateway_pid) == gateway_start,
        "gateway PID identity changed",
    )
    require(
        process_start_identity(backend_pid) == backend_start,
        "backend PID identity changed",
    )
    held = controller.observe(stage)
    backend_fd = backend_loaded_model_fd_libproc(backend_pid, held)
    core = {
        "format": CUSTODIAN_ATTESTATION_FORMAT,
        "schema_version": 3,
        "edge_id": EDGE_ID,
        "nonce": nonce,
        "challenge": challenge,
        "stage": stage,
        "custodian_pid": os.getpid(),
        "custodian_start_identity": copy.deepcopy(
            lifecycle["custodian_start_identity"]
        ),
        "gateway_pid": gateway_pid,
        "gateway_start_identity": copy.deepcopy(gateway_start),
        "backend_pid": backend_pid,
        "backend_start_identity": copy.deepcopy(backend_start),
        "controller_preload_fd": held,
        "backend_loaded_fd": backend_fd,
        "controller_fd_preceded_gateway_and_backend_launch": True,
        "lifecycle_sequence": copy.deepcopy(lifecycle),
    }
    return {
        **core,
        "challenge_response_sha256": sha256_canonical(core),
        "passed": True,
    }


def _terminate_gateway_child(
    process: subprocess.Popen[bytes],
    expected_start: dict[str, int],
    backend_pid: int,
    backend_start: dict[str, int],
    timeout: float,
) -> dict[str, Any]:
    require(timeout > 0.0, "gateway cleanup timeout is invalid")
    deadline = time.monotonic() + timeout

    def bounded_slice(fraction: float, label: str) -> float:
        return min(
            timeout * fraction,
            _remaining_deadline_seconds(deadline, label, monotonic=time.monotonic),
        )

    require(
        process_start_identity(process.pid) == expected_start,
        "refusing to terminate a reused gateway PID",
    )
    process.terminate()
    forced = False
    try:
        returncode = process.wait(timeout=bounded_slice(0.15, "gateway SIGTERM wait"))
    except subprocess.TimeoutExpired:
        require(
            process_start_identity(process.pid) == expected_start,
            "refusing to kill a reused gateway PID",
        )
        process.kill()
        forced = True
        returncode = process.wait(timeout=bounded_slice(0.15, "gateway SIGKILL wait"))
    backend_grace_deadline = min(deadline, time.monotonic() + timeout * 0.10)
    backend_termination = "exited-with-gateway"
    while time.monotonic() < backend_grace_deadline:
        try:
            current = process_start_identity(backend_pid)
        except PreflightBlockedError:
            break
        if current != backend_start:
            backend_termination = "original-exited-and-pid-was-reused"
            break
        time.sleep(min(0.05, backend_grace_deadline - time.monotonic()))
    else:
        require(
            process_start_identity(backend_pid) == backend_start,
            "refusing to terminate a reused backend PID",
        )
        os.kill(backend_pid, signal.SIGTERM)
        backend_termination = "explicit-sigterm"
        backend_term_deadline = min(deadline, time.monotonic() + timeout * 0.30)
        while time.monotonic() < backend_term_deadline:
            try:
                current = process_start_identity(backend_pid)
            except PreflightBlockedError:
                break
            if current != backend_start:
                backend_termination = "explicit-sigterm-original-exited-pid-reused"
                break
            time.sleep(min(0.05, backend_term_deadline - time.monotonic()))
        else:
            require(
                process_start_identity(backend_pid) == backend_start,
                "refusing to kill a reused backend PID",
            )
            os.kill(backend_pid, signal.SIGKILL)
            backend_termination = "explicit-sigkill"
            while time.monotonic() < deadline:
                try:
                    current = process_start_identity(backend_pid)
                except PreflightBlockedError:
                    break
                if current != backend_start:
                    backend_termination = "explicit-sigkill-original-exited-pid-reused"
                    break
                time.sleep(min(0.05, deadline - time.monotonic()))
            else:
                raise CampaignError("backend survived exact-PID SIGKILL cleanup")
    return {
        "gateway_pid": process.pid,
        "gateway_start_identity": copy.deepcopy(expected_start),
        "termination_requested": True,
        "forced_kill_required": forced,
        "returncode": returncode,
        "backend_pid": backend_pid,
        "backend_start_identity": copy.deepcopy(backend_start),
        "backend_termination": backend_termination,
    }


def _pid_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _terminate_exact_managed_pid(
    pid: int,
    expected_start: dict[str, int],
    timeout: float,
    label: str,
) -> dict[str, Any]:
    require(timeout > 0.0, f"{label} cleanup timeout is invalid")
    total_deadline = time.monotonic() + timeout
    try:
        current = process_start_identity(pid)
    except PreflightBlockedError:
        require(not _pid_exists(pid), f"{label} PID identity became ambiguous")
        return {"state": "already-exited", "pid": pid}
    if current != expected_start:
        return {"state": "original-exited-pid-reused", "pid": pid}
    os.kill(pid, signal.SIGTERM)
    forced = False
    term_deadline = min(total_deadline, time.monotonic() + timeout * 0.5)
    while time.monotonic() < term_deadline:
        try:
            current = process_start_identity(pid)
        except PreflightBlockedError:
            require(not _pid_exists(pid), f"{label} exit identity became ambiguous")
            return {"state": "terminated", "pid": pid, "forced": forced}
        if current != expected_start:
            return {"state": "terminated-pid-reused", "pid": pid, "forced": forced}
        time.sleep(min(0.05, term_deadline - time.monotonic()))
    require(
        process_start_identity(pid) == expected_start,
        f"refusing to kill a reused {label} PID",
    )
    os.kill(pid, signal.SIGKILL)
    forced = True
    while time.monotonic() < total_deadline:
        try:
            current = process_start_identity(pid)
        except PreflightBlockedError:
            require(not _pid_exists(pid), f"{label} kill identity became ambiguous")
            return {"state": "killed", "pid": pid, "forced": forced}
        if current != expected_start:
            return {"state": "killed-pid-reused", "pid": pid, "forced": forced}
        time.sleep(min(0.05, total_deadline - time.monotonic()))
    raise CampaignError(f"{label} survived exact-PID SIGKILL cleanup")


def _custodian_group_exists(process_group_id: int) -> bool:
    try:
        os.killpg(process_group_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError as error:
        raise CampaignError(
            "custodian process-group identity is inaccessible"
        ) from error
    return True


def _cleanup_unattested_custodian_start(
    process: subprocess.Popen[bytes], timeout: float
) -> dict[str, Any]:
    """Never signal a group when the spawned leader's start identity was unavailable."""

    require(timeout > 0.0, "unattested custodian cleanup timeout is invalid")
    require(
        process.poll() is not None,
        "custodian start identity is unavailable; refusing unbound process-group signal",
    )
    process.wait(timeout=timeout)
    require(
        not _custodian_group_exists(process.pid),
        "custodian leader exited but an unbound/reused process group still exists",
    )
    return {
        "process_group_id": process.pid,
        "state": "leader-exited-and-group-absent-without-signal",
        "signal_sent": False,
    }


def _terminate_custodian_process_group(
    process: subprocess.Popen[bytes],
    expected_start: dict[str, int],
    timeout: float,
) -> dict[str, Any]:
    group_id = process.pid
    require(timeout > 0.0, "custodian process-group cleanup timeout is invalid")
    deadline = time.monotonic() + timeout

    def require_bound_live_leader() -> None:
        require(
            process.poll() is None
            and process_start_identity(process.pid) == expected_start
            and os.getpgid(process.pid) == group_id
            and os.getsid(process.pid) == group_id,
            "refusing to signal an unbound custodian process group",
        )

    if not _custodian_group_exists(group_id):
        process.wait(
            timeout=_remaining_deadline_seconds(
                deadline,
                "custodian process-group reap",
                monotonic=time.monotonic,
            )
        )
        return {"process_group_id": group_id, "state": "already-exited"}
    require_bound_live_leader()
    os.killpg(group_id, signal.SIGTERM)
    forced = False
    while time.monotonic() < deadline:
        process.poll()
        if not _custodian_group_exists(group_id):
            process.wait(
                timeout=_remaining_deadline_seconds(
                    deadline,
                    "custodian process-group reap",
                    monotonic=time.monotonic,
                )
            )
            return {
                "process_group_id": group_id,
                "state": "terminated",
                "forced": forced,
            }
        require_bound_live_leader()
        time.sleep(0.05)
    require_bound_live_leader()
    os.killpg(group_id, signal.SIGKILL)
    forced = True
    while time.monotonic() < deadline:
        process.poll()
        if not _custodian_group_exists(group_id):
            process.wait(
                timeout=_remaining_deadline_seconds(
                    deadline,
                    "custodian process-group reap",
                    monotonic=time.monotonic,
                )
            )
            return {
                "process_group_id": group_id,
                "state": "killed",
                "forced": forced,
            }
        require_bound_live_leader()
        time.sleep(0.05)
    raise CampaignError("custodian process group survived SIGKILL")


class _CustodianTerminationRequested(BaseException):
    pass


def run_custodian(plan_path: Path | str, nonce: str) -> int:
    """Hidden daemon: hold model FD, own runtime launch, and serve attestations."""

    gateway_process: subprocess.Popen[bytes] | None = None
    gateway_start: dict[str, int] | None = None
    backend_pid: int | None = None
    backend_start: dict[str, int] | None = None
    controller: ControllerModelFd | None = None
    server: socket.socket | None = None
    socket_path: Path | None = None
    ready_emitted = False
    cleanup_requested = False

    def request_graceful_termination(_signal: int, _frame: Any) -> None:
        raise _CustodianTerminationRequested()

    signal.signal(signal.SIGTERM, request_graceful_termination)
    try:
        context = load_execution_context(plan_path)
        plan = context["plan"]
        runtime = plan["runtime"]
        socket_path = Path(runtime["custodian_control_socket_path"])
        parent = socket_path.parent
        require(
            parent.is_dir() and not parent.is_symlink(),
            "custodian socket parent is not a real directory",
        )
        require(
            parent.resolve(strict=True) == parent,
            "custodian socket parent canonical path drifted",
        )
        require(
            not os.path.lexists(socket_path), "custodian control socket already exists"
        )

        controller = ControllerModelFd(MODEL_PATH, MODEL_SIZE, MODEL_SHA256)
        held_before_spawn = controller.observe("before-gateway-spawn")
        lifecycle = {
            "controller_fd_open_started_monotonic_ns": controller.open_started_monotonic_ns,
            "controller_fd_custody_complete_monotonic_ns": controller.custody_complete_monotonic_ns,
            "gateway_spawn_invocation_monotonic_ns": time.monotonic_ns(),
        }
        gateway_process = subprocess.Popen(
            runtime["expected_gateway_argv"],
            cwd="/",
            env=_sorted_environment(runtime["expected_gateway_environment"]),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            close_fds=True,
            start_new_session=False,
        )
        gateway_start = process_start_identity(gateway_process.pid)
        lifecycle["gateway_kernel_identity_observed_monotonic_ns"] = time.monotonic_ns()
        lifecycle["custodian_start_identity"] = process_start_identity(os.getpid())
        zero_generation_model_load = admit_zero_generation_model_load(
            plan,
            gateway_process,
            gateway_start,
            expected_gateway_parent_pid=os.getpid(),
        )
        backend_pid = zero_generation_model_load["backend_pid"]
        require(
            _process_parent_pid(backend_pid) == gateway_process.pid,
            "backend is not the custodian-launched gateway child",
        )
        backend_start = process_start_identity(backend_pid)
        require(
            backend_start == zero_generation_model_load["backend_start_identity"],
            "backend identity changed after zero-generation model load admission",
        )
        lifecycle["backend_kernel_identity_observed_monotonic_ns"] = time.monotonic_ns()
        validate_controller_launch_sequence(lifecycle)
        initial = _custodian_light_attestation(
            plan,
            nonce,
            "0" * 64,
            "ready-zero-generation",
            controller,
            gateway_process.pid,
            gateway_start,
            backend_pid,
            backend_start,
            lifecycle,
        )
        require(
            initial["controller_preload_fd"]
            == held_before_spawn | {"stage": "ready-zero-generation"},
            "controller held-FD identity drifted across launch",
        )

        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server.bind(str(socket_path))
        os.chmod(socket_path, 0o600)
        server.listen(1)
        socket_receipt = _socket_custody(socket_path)
        ready = {
            "format": CUSTODIAN_READY_FORMAT,
            "schema_version": 3,
            "edge_id": EDGE_ID,
            "nonce": nonce,
            "custodian_pid": os.getpid(),
            "custodian_start_identity": lifecycle["custodian_start_identity"],
            "gateway_pid": gateway_process.pid,
            "gateway_start_identity": gateway_start,
            "backend_pid": backend_pid,
            "backend_start_identity": backend_start,
            "controller_preload_fd": initial["controller_preload_fd"],
            "backend_loaded_fd": initial["backend_loaded_fd"],
            "lifecycle_sequence": lifecycle,
            "control_socket": socket_receipt,
            "zero_generation_model_load": zero_generation_model_load,
            "generation_requests": 0,
            "passed": True,
        }
        sys.stdout.buffer.write(canonical_json_bytes(ready) + b"\n")
        sys.stdout.buffer.flush()
        ready_emitted = True
        seen_challenges: set[str] = set()
        while not cleanup_requested:
            connection, _ = server.accept()
            with connection:
                request = _receive_control_request(
                    connection,
                    runtime["custodian_shutdown_timeout_seconds"],
                )
                require(
                    set(request) == {"command", "nonce", "challenge", "stage"}
                    and request["command"] in ("attest", "shutdown")
                    and request["nonce"] == nonce
                    and re.fullmatch(r"[0-9a-f]{64}", request["challenge"]) is not None
                    and request["challenge"] not in seen_challenges
                    and isinstance(request["stage"], str)
                    and request["stage"],
                    "custodian control request failed exact admission",
                )
                seen_challenges.add(request["challenge"])
                attestation = _custodian_light_attestation(
                    plan,
                    nonce,
                    request["challenge"],
                    request["stage"],
                    controller,
                    gateway_process.pid,
                    gateway_start,
                    backend_pid,
                    backend_start,
                    lifecycle,
                )
                if request["command"] == "attest":
                    _send_control_response(connection, attestation)
                    continue
                cleanup = _terminate_gateway_child(
                    gateway_process,
                    gateway_start,
                    backend_pid,
                    backend_start,
                    runtime["custodian_shutdown_timeout_seconds"]
                    - CUSTODIAN_CLEANUP_RESPONSE_MARGIN_SECONDS,
                )
                cleanup_receipt = {
                    "format": CUSTODIAN_CLEANUP_FORMAT,
                    "schema_version": 3,
                    "edge_id": EDGE_ID,
                    "nonce": nonce,
                    "challenge": request["challenge"],
                    "pre_cleanup_attestation": attestation,
                    "runtime_cleanup": cleanup,
                    "controller_fd_still_held_while_cleanup_response_built": controller.observe(
                        "cleanup-response"
                    ),
                    "passed": True,
                }
                _send_control_response(connection, cleanup_receipt)
                cleanup_requested = True
        return 0
    except BaseException as error:
        if not ready_emitted:
            failure = {
                "format": CUSTODIAN_READY_FORMAT,
                "schema_version": 3,
                "edge_id": EDGE_ID,
                "nonce": nonce,
                "passed": False,
                "exception_type": type(error).__name__,
                "message": str(error),
            }
            try:
                sys.stdout.buffer.write(canonical_json_bytes(failure) + b"\n")
                sys.stdout.buffer.flush()
            except BaseException:
                pass
        return 3
    finally:
        if gateway_process is not None and gateway_process.poll() is None:
            try:
                if (
                    gateway_start is not None
                    and backend_pid is not None
                    and backend_start is not None
                ):
                    _terminate_gateway_child(
                        gateway_process,
                        gateway_start,
                        backend_pid,
                        backend_start,
                        30.0,
                    )
                else:
                    gateway_process.terminate()
                    try:
                        gateway_process.wait(timeout=30.0)
                    except subprocess.TimeoutExpired:
                        gateway_process.kill()
                        gateway_process.wait(timeout=30.0)
            except BaseException:
                pass
        if backend_pid is not None and backend_start is not None:
            try:
                _terminate_exact_managed_pid(
                    backend_pid, backend_start, 30.0, "custodian backend"
                )
            except BaseException:
                pass
        if controller is not None:
            controller.close()
        if server is not None:
            server.close()
        if socket_path is not None and os.path.lexists(socket_path):
            try:
                if stat.S_ISSOCK(socket_path.lstat().st_mode):
                    socket_path.unlink()
            except OSError:
                pass


def custodian_control_request(
    socket_path_value: Path | str,
    nonce: str,
    command: str,
    stage: str,
    timeout_seconds: float,
) -> dict[str, Any]:
    require(command in ("attest", "shutdown"), "custodian command is invalid")
    require(
        re.fullmatch(r"[0-9a-f]{64}", nonce) is not None, "custodian nonce is invalid"
    )
    socket_path = Path(socket_path_value)
    socket_before = _socket_custody(socket_path)
    challenge = secrets.token_hex(32)
    connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    require(timeout_seconds > 0.0, "custodian control timeout is invalid")
    deadline = time.monotonic() + timeout_seconds
    try:
        connection.settimeout(
            _remaining_deadline_seconds(
                deadline, "custodian control connect", monotonic=time.monotonic
            )
        )
        connection.connect(str(socket_path))
        connection.settimeout(
            _remaining_deadline_seconds(
                deadline, "custodian control send", monotonic=time.monotonic
            )
        )
        connection.sendall(
            canonical_json_bytes(
                {
                    "command": command,
                    "nonce": nonce,
                    "challenge": challenge,
                    "stage": stage,
                }
            )
            + b"\n"
        )
        connection.shutdown(socket.SHUT_WR)
        raw = bytearray()
        while len(raw) <= CUSTODIAN_CONTROL_RESPONSE_MAX_BYTES:
            connection.settimeout(
                _remaining_deadline_seconds(
                    deadline,
                    "custodian control response read",
                    monotonic=time.monotonic,
                )
            )
            chunk = connection.recv(
                min(
                    64 * 1024,
                    CUSTODIAN_CONTROL_RESPONSE_MAX_BYTES + 1 - len(raw),
                )
            )
            if not chunk:
                break
            raw.extend(chunk)
        require(
            len(raw) <= CUSTODIAN_CONTROL_RESPONSE_MAX_BYTES,
            "custodian response is oversized",
        )
    finally:
        connection.close()
    response = parse_strict_json_line(bytes(raw))
    require(
        response.get("format")
        == (
            CUSTODIAN_ATTESTATION_FORMAT
            if command == "attest"
            else CUSTODIAN_CLEANUP_FORMAT
        )
        and response.get("schema_version") == 3
        and response.get("edge_id") == EDGE_ID
        and response.get("nonce") == nonce
        and response.get("challenge") == challenge
        and response.get("passed") is True,
        "custodian challenge response binding drifted",
    )
    if command == "attest":
        core = {
            key: value
            for key, value in response.items()
            if key not in ("challenge_response_sha256", "passed")
        }
        require(
            response["challenge_response_sha256"] == sha256_canonical(core),
            "custodian challenge-response digest drifted",
        )
        require(
            _socket_custody(socket_path) == socket_before,
            "custodian socket identity changed during challenge",
        )
    return response


def attest_custodian(
    plan: dict[str, Any], binding: dict[str, Any], stage: str
) -> dict[str, Any]:
    runtime = plan["runtime"]
    response = custodian_control_request(
        runtime["custodian_control_socket_path"],
        binding["nonce"],
        "attest",
        stage,
        runtime["custodian_ready_timeout_seconds"],
    )
    require(
        response["stage"] == stage
        and response["custodian_pid"] == binding["custodian_pid"]
        and response["custodian_start_identity"] == binding["custodian_start_identity"]
        and response["gateway_pid"] == binding["gateway_pid"]
        and response["gateway_start_identity"] == binding["gateway_start_identity"]
        and response["backend_pid"] == binding["backend_pid"]
        and response["backend_start_identity"] == binding["backend_start_identity"]
        and response["lifecycle_sequence"] == binding["lifecycle_sequence"],
        "custodian live attestation differs from its marker-bound identity",
    )
    controller_external = controller_fd_process_proof(
        binding["custodian_pid"], response["controller_preload_fd"]
    )
    backend_external = backend_loaded_model_fd_proof(
        binding["backend_pid"], response["controller_preload_fd"]
    )
    require(
        response["backend_loaded_fd"] == backend_external["backend_loaded_fd"],
        "custodian and controller independently disagree on backend model FD",
    )
    require(
        process_start_identity(binding["custodian_pid"])
        == binding["custodian_start_identity"]
        and process_start_identity(binding["gateway_pid"])
        == binding["gateway_start_identity"]
        and process_start_identity(binding["backend_pid"])
        == binding["backend_start_identity"],
        "custodian-managed process identity changed during external attestation",
    )
    return {
        "daemon_challenge_response": response,
        "controller_fd_external_proof": controller_external,
        "backend_fd_external_proof": backend_external,
        "same_controller_and_backend_file_identity": True,
        "passed": True,
    }


def validate_zero_generation_model_load_receipt(
    receipt: dict[str, Any],
    plan: dict[str, Any],
    *,
    gateway_pid: int | None = None,
    gateway_start: dict[str, int] | None = None,
    backend_pid: int | None = None,
    backend_start: dict[str, int] | None = None,
    custodian_pid: int | None = None,
) -> dict[str, Any]:
    require(
        isinstance(receipt, dict)
        and set(receipt) == ZERO_GENERATION_MODEL_LOAD_FIELDS_V3,
        "zero-generation model-load receipt schema drifted",
    )
    require(
        receipt.get("format")
        == "apxinf-omniinfer-gateway-zero-generation-model-load-v3"
        and receipt.get("schema_version") == 3
        and receipt.get("edge_id") == EDGE_ID
        and receipt.get("model_select_request_count") == 1
        and receipt.get("generation_requests") == 0
        and receipt.get("generation_endpoint_paths_called") == []
        and receipt.get("request_history_records_created") == 0
        and receipt.get("all_passed") is True,
        "zero-generation model-load admission drifted",
    )
    management_api = receipt.get("management_api")
    require(
        isinstance(management_api, dict)
        and set(management_api)
        == {"method", "path", "pinned_omniinfer_source_commit", "alias_used"}
        and management_api
        == {
            "method": "POST",
            "path": "/omni/model/select",
            "pinned_omniinfer_source_commit": OMNI_SOURCE_COMMIT,
            "alias_used": False,
        },
        "zero-generation model-load API binding drifted",
    )
    sequence = receipt.get("management_request_sequence")
    require(
        isinstance(sequence, list)
        and all(item == ["GET", "/omni/state"] for item in sequence[:1])
        and sequence.count(["POST", "/omni/model/select"]) == 1,
        "zero-generation management request sequence drifted",
    )
    select_index = sequence.index(["POST", "/omni/model/select"])
    require(
        select_index > 0
        and select_index < len(sequence) - 1
        and all(item == ["GET", "/omni/state"] for item in sequence[:select_index])
        and all(
            item == ["GET", "/omni/state"] for item in sequence[select_index + 1 :]
        ),
        "model-select was not bracketed by state-only management requests",
    )

    request = receipt.get("request")
    request_bytes = gateway_model_select_request_bytes(plan)
    require(
        isinstance(request, dict)
        and set(request) == {"object", "canonical_utf8", "size_bytes", "sha256"}
        and request["object"] == gateway_model_select_request(plan)
        and request["canonical_utf8"] == request_bytes.decode("utf-8")
        and request["size_bytes"] == len(request_bytes)
        and request["sha256"] == sha256_bytes(request_bytes),
        "zero-generation model-select request drifted",
    )
    response = receipt.get("response")
    require(
        isinstance(response, dict)
        and set(response)
        == {
            "object",
            "canonical_sha256",
            "validated",
            "runtime_generation_is_not_an_inference_generation",
        }
        and response["canonical_sha256"] == sha256_canonical(response["object"])
        and response["runtime_generation_is_not_an_inference_generation"] is True,
        "zero-generation model-select response binding drifted",
    )
    response_validated = validate_gateway_model_select_response(
        response["object"], plan
    )
    require(
        response["validated"] == response_validated,
        "zero-generation model-select response validation drifted",
    )

    pre_load = receipt.get("pre_load_state")
    loaded = receipt.get("loaded_state")
    require(
        isinstance(pre_load, dict)
        and set(pre_load) == {"object", "validated"}
        and pre_load["validated"] == validate_gateway_preload_state(pre_load["object"])
        and isinstance(loaded, dict)
        and set(loaded) == {"object", "validated"}
        and loaded["validated"] == validate_gateway_state(loaded["object"], plan),
        "zero-generation gateway state admission drifted",
    )
    require(
        loaded["validated"]["backend_pid"] == response_validated["backend_pid"]
        and loaded["validated"]["backend_port"] == response_validated["backend_port"],
        "zero-generation response/state backend identity drifted",
    )

    transport = receipt.get("transport")
    require(
        isinstance(transport, dict)
        and set(transport)
        == {
            "pre_load_state",
            "model_select",
            "post_load_state_attempts",
            "loaded_state",
        },
        "zero-generation transport collection schema drifted",
    )

    def validate_projected(
        value: Any,
        label: str,
        method: str,
        path: str,
        body: bytes | None,
        response_object: dict[str, Any],
    ) -> dict[str, Any]:
        require(
            isinstance(value, dict)
            and set(value) == ZERO_GENERATION_MANAGEMENT_TRANSPORT_FIELDS_V3,
            "zero-generation projected transport schema drifted",
        )
        projected = validate_zero_generation_management_transport(
            value,
            plan["runtime"]["omni_base_url"],
            label,
            method,
            path,
            body,
            response_object,
        )
        require(projected == value, "zero-generation projected transport drifted")
        return projected

    validate_projected(
        transport["pre_load_state"],
        "custodian-pre-load-state",
        "GET",
        "/omni/state",
        None,
        pre_load["object"],
    )
    validate_projected(
        transport["model_select"],
        "custodian-zero-generation-model-select",
        "POST",
        "/omni/model/select",
        request_bytes,
        response["object"],
    )
    attempts = transport["post_load_state_attempts"]
    require(
        isinstance(attempts, list) and attempts, "post-load state transports are absent"
    )
    for attempt in attempts:
        try:
            raw = base64.b64decode(
                attempt["response_base64"].encode("ascii"), validate=True
            )
        except (
            KeyError,
            AttributeError,
            UnicodeEncodeError,
            binascii.Error,
            ValueError,
        ) as error:
            raise CampaignError(
                "post-load state transport response is invalid"
            ) from error
        validate_projected(
            attempt,
            "custodian-post-load-state",
            "GET",
            "/omni/state",
            None,
            parse_strict_json_document(raw),
        )
    require(
        attempts[-1] == transport["loaded_state"],
        "final loaded-state transport differs from its attempt ledger",
    )
    validate_projected(
        transport["loaded_state"],
        "custodian-post-load-state",
        "GET",
        "/omni/state",
        None,
        loaded["object"],
    )

    observed_gateway_pid = receipt.get("gateway_pid")
    observed_backend_pid = receipt.get("backend_pid")
    require(
        is_int(observed_gateway_pid)
        and observed_gateway_pid > 0
        and is_int(observed_backend_pid)
        and observed_backend_pid > 0
        and observed_backend_pid == loaded["validated"]["backend_pid"]
        and receipt.get("gateway_start_identity") == receipt.get("gateway_end_identity")
        and receipt.get("gateway_process_start_end_equal") is True
        and receipt.get("gateway_parent_pid_start")
        == receipt.get("gateway_parent_pid_end")
        and receipt.get("gateway_parent_start_end_equal") is True
        and receipt.get("gateway_is_direct_parent_of_backend") is True,
        "zero-generation process ancestry receipt drifted",
    )
    if gateway_pid is not None:
        require(
            observed_gateway_pid == gateway_pid,
            "ready gateway PID differs from load receipt",
        )
    if gateway_start is not None:
        require(
            receipt["gateway_start_identity"] == gateway_start,
            "ready gateway start differs from load receipt",
        )
    if backend_pid is not None:
        require(
            observed_backend_pid == backend_pid,
            "ready backend PID differs from load receipt",
        )
    if backend_start is not None:
        require(
            receipt["backend_start_identity"] == backend_start,
            "ready backend start differs from load receipt",
        )
    if custodian_pid is not None:
        require(
            receipt["gateway_parent_pid_start"] == custodian_pid,
            "load receipt gateway is not the ready custodian child",
        )
    require(
        receipt.get("history_start") == receipt.get("history_end")
        and receipt.get("history_start_end_equal") is True,
        "request history changed during zero-generation model load",
    )
    return copy.deepcopy(receipt)


def _custodian_binding_from_ready(
    ready: dict[str, Any], plan_path: Path | str, plan: dict[str, Any]
) -> dict[str, Any]:
    require(
        isinstance(ready, dict)
        and set(ready) == CUSTODIAN_READY_FIELDS_V3
        and ready.get("format") == CUSTODIAN_READY_FORMAT
        and ready.get("schema_version") == 3
        and ready.get("edge_id") == EDGE_ID
        and ready.get("passed") is True
        and ready.get("generation_requests") == 0,
        "custodian readiness receipt failed",
    )
    nonce = ready.get("nonce")
    require(
        re.fullmatch(r"[0-9a-f]{64}", nonce or "") is not None,
        "custodian readiness nonce is invalid",
    )
    lifecycle = ready.get("lifecycle_sequence")
    require(isinstance(lifecycle, dict), "custodian lifecycle sequence is absent")
    validate_controller_launch_sequence(lifecycle)
    require(
        lifecycle.get("custodian_start_identity")
        == ready.get("custodian_start_identity"),
        "custodian lifecycle start binding drifted",
    )
    zero_generation_model_load = validate_zero_generation_model_load_receipt(
        ready["zero_generation_model_load"],
        plan,
        gateway_pid=ready["gateway_pid"],
        gateway_start=ready["gateway_start_identity"],
        backend_pid=ready["backend_pid"],
        backend_start=ready["backend_start_identity"],
        custodian_pid=ready["custodian_pid"],
    )
    return {
        "format": "apxinf-omniinfer-gateway-custodian-binding-v3",
        "schema_version": 3,
        "edge_id": EDGE_ID,
        "nonce": nonce,
        "control_socket": copy.deepcopy(ready["control_socket"]),
        "custodian_pid": ready["custodian_pid"],
        "custodian_start_identity": copy.deepcopy(ready["custodian_start_identity"]),
        "gateway_pid": ready["gateway_pid"],
        "gateway_start_identity": copy.deepcopy(ready["gateway_start_identity"]),
        "backend_pid": ready["backend_pid"],
        "backend_start_identity": copy.deepcopy(ready["backend_start_identity"]),
        "lifecycle_sequence": copy.deepcopy(lifecycle),
        "zero_generation_model_load": zero_generation_model_load,
        "expected_custodian_argv": _custodian_command(plan_path, nonce),
        "expected_custodian_environment": copy.deepcopy(_CUSTODIAN_ENV),
        "ready_receipt": copy.deepcopy(ready),
    }


def _require_loopback_listener_absent(port: int, label: str) -> None:
    result = hardened_command_runner(
        ["/usr/sbin/lsof", "-nP", "-t", f"-iTCP:{port}", "-sTCP:LISTEN"],
        Path("/"),
        15.0,
    )
    require(
        result["returncode"] == 1
        and result["stdout"] == b""
        and result["stderr"] == b"",
        f"{label} listener must be absent before custodian-owned launch",
    )


def start_custodian(
    plan_path: Path | str,
    plan: dict[str, Any],
    campaign_directory_initialization: dict[str, Any],
) -> dict[str, Any]:
    """Start the only allowed daemon before it opens the model and launches runtime."""

    runtime = plan["runtime"]
    directory_receipt = validate_campaign_directory_initialization_receipt(
        campaign_directory_initialization, plan, verify_live=True
    )
    parsed = urllib.parse.urlsplit(runtime["omni_base_url"])
    require(parsed.port is not None, "gateway plan port is absent")
    _require_loopback_listener_absent(parsed.port, "gateway")
    socket_path = Path(runtime["custodian_control_socket_path"])
    require(not os.path.lexists(socket_path), "custodian socket already exists")
    nonce = secrets.token_hex(32)
    command = _custodian_command(plan_path, nonce)
    process = subprocess.Popen(
        command,
        cwd="/",
        env=_sorted_environment(_CUSTODIAN_ENV),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        close_fds=True,
        start_new_session=True,
    )
    spawned_start: dict[str, int] | None = None
    try:
        spawned_start = process_start_identity(process.pid)
        require(
            os.getpgid(process.pid) == process.pid
            and os.getsid(process.pid) == process.pid,
            "custodian did not become an exclusive process-group/session leader",
        )
        require(process.stdout is not None, "custodian readiness pipe is absent")
        ready = _read_pipe_json_line_with_total_deadline(
            process.stdout.fileno(),
            _custodian_parent_ready_budget_seconds(runtime),
            16 * 1024 * 1024,
        )
        binding = _custodian_binding_from_ready(ready, plan_path, plan)
        binding["campaign_directory_initialization"] = directory_receipt
        binding["custodian_process_group_id"] = process.pid
        binding["custodian_session_id"] = process.pid
        require(
            binding["custodian_pid"] == process.pid
            and process.poll() is None
            and process_start_identity(process.pid)
            == binding["custodian_start_identity"],
            "spawned custodian PID/start identity drifted",
        )
        require(
            _socket_custody(socket_path) == binding["control_socket"],
            "custodian readiness socket identity drifted",
        )
        initial = attest_custodian(plan, binding, "prepare-before-any-generation")
        binding["initial_external_attestation"] = initial
        process.stdout.close()
        return binding
    except BaseException as launch_error:
        try:
            if process.poll() is None and os.path.lexists(socket_path):
                try:
                    custodian_control_request(
                        socket_path,
                        nonce,
                        "shutdown",
                        "prepare-start-failure-cleanup",
                        runtime["custodian_shutdown_timeout_seconds"],
                    )
                except BaseException:
                    pass
            if spawned_start is None:
                _cleanup_unattested_custodian_start(
                    process, runtime["custodian_shutdown_timeout_seconds"]
                )
            else:
                _terminate_custodian_process_group(
                    process,
                    spawned_start,
                    runtime["custodian_shutdown_timeout_seconds"],
                )
        except BaseException as cleanup_error:
            combined = CampaignError(
                f"custodian launch failed and exact process-group cleanup failed: {cleanup_error}"
            )
            setattr(combined, "custodian_launch_cleanup_complete", False)
            raise combined from cleanup_error
        setattr(launch_error, "custodian_launch_cleanup_complete", True)
        raise
    finally:
        if process.stdout is not None and not process.stdout.closed:
            process.stdout.close()


def collect_custodian_process_custody(
    plan_path: Path | str, plan: dict[str, Any], binding: dict[str, Any]
) -> dict[str, Any]:
    custodian = collect_process_identity(
        binding["custodian_pid"],
        binding["expected_custodian_argv"],
        binding["expected_custodian_argv"][0],
    )
    require(
        binding["expected_custodian_argv"][1:]
        == _custodian_command(plan_path, binding["nonce"])[1:],
        "custodian marker argv differs from the active plan/driver binding",
    )
    require(
        binding.get("custodian_process_group_id") == binding["custodian_pid"]
        and binding.get("custodian_session_id") == binding["custodian_pid"]
        and os.getpgid(binding["custodian_pid"]) == binding["custodian_pid"]
        and os.getsid(binding["custodian_pid"]) == binding["custodian_pid"],
        "custodian exclusive process-group/session binding drifted",
    )
    _require_exact_process_environment(
        custodian, binding["expected_custodian_environment"], "custodian"
    )
    gateway = collect_process_identity(
        binding["gateway_pid"],
        plan["runtime"]["expected_gateway_argv"],
        plan["artifacts"]["omniinfer_cli"]["absolute_path"],
    )
    _require_exact_process_environment(
        gateway,
        plan["runtime"]["expected_gateway_environment"],
        "gateway",
    )
    backend_port = urllib.parse.urlsplit(plan["runtime"]["omni_base_url"]).port
    require(backend_port is not None, "gateway public port is absent")
    require(
        _process_parent_pid(binding["gateway_pid"]) == binding["custodian_pid"]
        and _process_parent_pid(binding["backend_pid"]) == binding["gateway_pid"],
        "custodian/gateway/backend parent chain drifted",
    )
    driver_raw, driver_source = _read_regular_no_follow(
        Path(__file__).resolve(strict=True), 16 * 1024 * 1024
    )
    require(driver_raw, "custodian driver source is empty")
    shared_raw, shared_driver_source = _read_regular_no_follow(
        _NATIVE_DRIVER_PATH.resolve(strict=True), 16 * 1024 * 1024
    )
    require(shared_raw, "custodian shared formal-driver source is empty")
    return {
        "custodian_process": custodian,
        "gateway_process": gateway,
        "driver_source": driver_source,
        "shared_formal_driver_source": shared_driver_source,
        "controller_before_gateway_before_backend": True,
        "control_socket": _socket_custody(
            plan["runtime"]["custodian_control_socket_path"]
        ),
        "passed": True,
    }


def shutdown_custodian(
    plan: dict[str, Any], binding: dict[str, Any], stage: str
) -> dict[str, Any]:
    cleanup = custodian_control_request(
        plan["runtime"]["custodian_control_socket_path"],
        binding["nonce"],
        "shutdown",
        stage,
        plan["runtime"]["custodian_shutdown_timeout_seconds"],
    )
    require(
        cleanup["pre_cleanup_attestation"]["custodian_pid"] == binding["custodian_pid"]
        and cleanup["pre_cleanup_attestation"]["gateway_pid"] == binding["gateway_pid"]
        and cleanup["pre_cleanup_attestation"]["backend_pid"] == binding["backend_pid"],
        "custodian cleanup identity drifted",
    )
    pre_cleanup = cleanup["pre_cleanup_attestation"]
    require(
        pre_cleanup["custodian_start_identity"] == binding["custodian_start_identity"]
        and pre_cleanup["gateway_start_identity"] == binding["gateway_start_identity"]
        and pre_cleanup["backend_start_identity"] == binding["backend_start_identity"]
        and _stable_controller_fd_receipt(
            cleanup["controller_fd_still_held_while_cleanup_response_built"]
        )
        == _stable_controller_fd_receipt(pre_cleanup["controller_preload_fd"]),
        "custodian cleanup lost process or controller-FD custody before response",
    )
    runtime_cleanup = cleanup["runtime_cleanup"]
    require(
        runtime_cleanup["gateway_pid"] == binding["gateway_pid"]
        and runtime_cleanup["gateway_start_identity"]
        == binding["gateway_start_identity"]
        and runtime_cleanup["backend_pid"] == binding["backend_pid"]
        and runtime_cleanup["backend_start_identity"]
        == binding["backend_start_identity"]
        and runtime_cleanup["termination_requested"] is True,
        "custodian runtime cleanup receipt drifted",
    )
    deadline = time.monotonic() + plan["runtime"]["custodian_shutdown_timeout_seconds"]
    socket_path = Path(plan["runtime"]["custodian_control_socket_path"])
    original_exit = False
    pid_reused = False
    while time.monotonic() < deadline:
        try:
            current = process_start_identity(binding["custodian_pid"])
        except PreflightBlockedError:
            original_exit = True
        else:
            if current != binding["custodian_start_identity"]:
                original_exit = True
                pid_reused = True
        if original_exit and not os.path.lexists(socket_path):
            break
        time.sleep(0.05)
    require(
        original_exit and not os.path.lexists(socket_path),
        "custodian did not exit and remove its exact control socket after cleanup",
    )
    managed_exit: dict[str, Any] = {}
    for label, pid, expected in (
        (
            "gateway",
            binding["gateway_pid"],
            binding["gateway_start_identity"],
        ),
        (
            "backend",
            binding["backend_pid"],
            binding["backend_start_identity"],
        ),
    ):
        try:
            current = process_start_identity(pid)
        except PreflightBlockedError:
            require(
                not _pid_exists(pid),
                f"{label} exit identity is ambiguous after custodian cleanup",
            )
            managed_exit[label] = {
                "original_process_exited": True,
                "pid_reused_after_exit": False,
            }
        else:
            require(
                current != expected,
                f"original {label} remained alive after custodian cleanup",
            )
            managed_exit[label] = {
                "original_process_exited": True,
                "pid_reused_after_exit": True,
                "reused_start_identity": current,
            }
    parsed = urllib.parse.urlsplit(plan["runtime"]["omni_base_url"])
    require(parsed.port is not None, "gateway cleanup port is absent")
    _require_loopback_listener_absent(parsed.port, "post-custodian-cleanup gateway")
    cleanup["cleanup_completion"] = {
        "custodian_original_process_exited": True,
        "custodian_pid_reused_after_exit": pid_reused,
        "gateway": managed_exit["gateway"],
        "backend": managed_exit["backend"],
        "gateway_listener_absent": True,
        "control_socket_removed": True,
        "controller_fd_closed_by_process_exit": True,
    }
    return cleanup


def _fallback_cleanup_marker_bound_processes(
    plan: dict[str, Any], binding: dict[str, Any], stage: str
) -> dict[str, Any]:
    """Best-effort cleanup that never signals without exact live identity/ancestry."""

    timeout = float(plan["runtime"]["custodian_shutdown_timeout_seconds"])
    require(timeout > 0.0, "fallback cleanup timeout is invalid")
    deadline = time.monotonic() + timeout
    steps: list[dict[str, Any]] = []
    blocked = False
    specifications = (
        (
            "backend",
            binding["backend_pid"],
            binding["backend_start_identity"],
            binding["gateway_pid"],
            binding["gateway_start_identity"],
        ),
        (
            "gateway",
            binding["gateway_pid"],
            binding["gateway_start_identity"],
            binding["custodian_pid"],
            binding["custodian_start_identity"],
        ),
        (
            "custodian",
            binding["custodian_pid"],
            binding["custodian_start_identity"],
            None,
            None,
        ),
    )
    for index, (label, pid, expected, parent_pid, parent_expected) in enumerate(
        specifications
    ):
        step: dict[str, Any] = {
            "label": label,
            "pid": pid,
            "expected_start_identity": copy.deepcopy(expected),
            "signal_sent": False,
        }
        if blocked:
            step.update(
                {
                    "state": "unattempted-after-prior-identity-or-ancestry-blocker",
                    "passed": False,
                }
            )
            steps.append(step)
            continue
        try:
            current = process_start_identity(pid)
        except PreflightBlockedError as error:
            if _pid_exists(pid):
                step.update(
                    {
                        "state": "identity-ambiguous-no-signal",
                        "error": str(error),
                        "passed": False,
                    }
                )
                blocked = True
            else:
                step.update({"state": "already-exited", "passed": True})
            steps.append(step)
            continue
        if current != expected:
            step.update(
                {
                    "state": "original-exited-pid-reused-no-signal",
                    "observed_start_identity": copy.deepcopy(current),
                    "passed": True,
                }
            )
            steps.append(step)
            continue
        try:
            if parent_pid is None:
                require(
                    os.getpgid(pid) == pid and os.getsid(pid) == pid,
                    "custodian session/group ancestry changed",
                )
            else:
                require(
                    process_start_identity(parent_pid) == parent_expected
                    and _process_parent_pid(pid) == parent_pid,
                    f"{label} parent ancestry changed",
                )
            remaining_steps = len(specifications) - index
            per_process_budget = (
                _remaining_deadline_seconds(
                    deadline,
                    "fallback exact-PID cleanup",
                    monotonic=time.monotonic,
                )
                / remaining_steps
            )
            termination = _terminate_exact_managed_pid(
                pid, expected, per_process_budget, f"fallback {label}"
            )
            step.update(
                {
                    "state": termination["state"],
                    "signal_sent": True,
                    "termination": termination,
                    "passed": True,
                }
            )
        except BaseException as error:
            step.update(
                {
                    "state": "identity-or-ancestry-blocked-no-further-signals",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "passed": False,
                }
            )
            blocked = True
        steps.append(step)

    socket_path = Path(plan["runtime"]["custodian_control_socket_path"])
    socket_cleanup: dict[str, Any]
    listener_absent = False
    if all(step["passed"] for step in steps):
        try:
            if os.path.lexists(socket_path):
                require(
                    _socket_custody(socket_path) == binding["control_socket"],
                    "fallback control socket identity drifted",
                )
                socket_path.unlink()
            require(
                not os.path.lexists(socket_path),
                "fallback control socket remained present",
            )
            socket_cleanup = {"state": "exact-socket-absent", "passed": True}
            parsed = urllib.parse.urlsplit(plan["runtime"]["omni_base_url"])
            require(parsed.port is not None, "fallback gateway port is absent")
            _require_loopback_listener_absent(parsed.port, "fallback gateway")
            listener_absent = True
        except BaseException as error:
            socket_cleanup = {
                "state": "socket-or-listener-cleanup-blocked",
                "error_type": type(error).__name__,
                "error": str(error),
                "passed": False,
            }
    else:
        socket_cleanup = {
            "state": "unattempted-after-process-identity-blocker",
            "passed": False,
        }
    passed = (
        all(step["passed"] for step in steps)
        and socket_cleanup["passed"]
        and listener_absent
    )
    return {
        "format": "apxinf-omniinfer-gateway-exact-pid-fallback-cleanup-v3",
        "schema_version": 3,
        "edge_id": EDGE_ID,
        "stage": stage,
        "process_steps": steps,
        "control_socket_cleanup": socket_cleanup,
        "gateway_listener_absent": listener_absent,
        "passed": passed,
    }


def _cleanup_failed_prepare_resources(
    plan: dict[str, Any],
    binding: dict[str, Any] | None,
    directory_initialization: dict[str, Any] | None,
    marker_path: Path,
    raw_path: Path,
    stage: str,
    *,
    runtime_cleanup_already_complete: bool = False,
) -> dict[str, Any]:
    """Stop a bound runtime regardless of marker-race state; delete only fresh tree."""

    shutdown_receipt: dict[str, Any] | None = None
    shutdown_error: dict[str, str] | None = None
    fallback: dict[str, Any] | None = None
    runtime_cleanup_complete = runtime_cleanup_already_complete
    if binding is not None:
        try:
            shutdown_receipt = shutdown_custodian(plan, binding, stage)
            runtime_cleanup_complete = True
        except BaseException as error:
            shutdown_error = {
                "exception_type": type(error).__name__,
                "message": str(error),
            }
            fallback = _fallback_cleanup_marker_bound_processes(plan, binding, stage)
            runtime_cleanup_complete = fallback["passed"] is True
    directory_cleanup: dict[str, Any] | None = None
    if (
        directory_initialization is not None
        and runtime_cleanup_complete
        and not os.path.lexists(marker_path)
        and not os.path.lexists(raw_path)
    ):
        directory_cleanup = cleanup_failed_prepare_campaign_tree(
            plan, directory_initialization, stage
        )
    return {
        "runtime_cleanup_complete": runtime_cleanup_complete,
        "shutdown_receipt": shutdown_receipt,
        "shutdown_error": shutdown_error,
        "exact_pid_fallback": fallback,
        "directory_cleanup": directory_cleanup,
        "marker_present_prevented_directory_cleanup": os.path.lexists(marker_path),
        "raw_present_prevented_directory_cleanup": os.path.lexists(raw_path),
    }


def _cache_clear_validator(payload: dict[str, Any]) -> dict[str, Any]:
    receipt_require(payload.get("ok") is True, "cache clear was not acknowledged")
    receipt_require(
        payload.get("cache_policy") == "cleared_each_run", "cache clear policy drifted"
    )
    receipt_require(
        payload.get("cleared_slots") == [0], "cache clear did not acknowledge slot zero"
    )
    return {
        "acknowledged": True,
        "cleared_slots": [0],
        "response": copy.deepcopy(payload),
    }


def _tokenize_ids(payload: dict[str, Any]) -> list[int]:
    tokens = payload.get("tokens")
    receipt_require(isinstance(tokens, list), "tokenize response lacks tokens")
    result: list[int] = []
    for entry in tokens:
        value = entry.get("id") if isinstance(entry, dict) else entry
        receipt_require(
            is_int(value) and value >= 0, "tokenize response has invalid ID"
        )
        result.append(value)
    receipt_require(result == PROMPT_TOKEN_IDS, "rendered raw prompt token IDs drifted")
    return result


def _apply_template_validator(payload: dict[str, Any]) -> str:
    rendered = payload.get("prompt")
    receipt_require(rendered == RENDERED_PROMPT, "backend rendered prompt drifted")
    return rendered


def _stable_controller_fd_receipt(receipt: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in receipt.items() if key != "stage"}


def require_same_custodian_attestation(
    start: dict[str, Any], end: dict[str, Any]
) -> None:
    start_daemon = start["daemon_challenge_response"]
    end_daemon = end["daemon_challenge_response"]
    for field in (
        "custodian_pid",
        "custodian_start_identity",
        "gateway_pid",
        "gateway_start_identity",
        "backend_pid",
        "backend_start_identity",
        "lifecycle_sequence",
    ):
        require(start_daemon[field] == end_daemon[field], f"custodian {field} changed")
    require(
        _stable_controller_fd_receipt(start_daemon["controller_preload_fd"])
        == _stable_controller_fd_receipt(end_daemon["controller_preload_fd"]),
        "controller-held model FD changed",
    )
    require(
        start_daemon["backend_loaded_fd"] == end_daemon["backend_loaded_fd"],
        "backend loaded-model FD changed",
    )


def reject_unreviewed_same_ofd_claims(value: Any, path: tuple[str, ...] = ()) -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            require(isinstance(key, str), "runtime custody contains a non-string key")
            next_path = (*path, key)
            if "open_file_description" in key:
                require(
                    next_path == ("same_open_file_description_not_claimed",)
                    and nested is True,
                    "runtime custody contains an unreviewed same-OFD claim",
                )
            reject_unreviewed_same_ofd_claims(nested, next_path)
    elif isinstance(value, list):
        for nested in value:
            reject_unreviewed_same_ofd_claims(nested, path)


def validate_published_runtime_preflight(
    marker_runtime: dict[str, Any],
    runtime_now: dict[str, Any],
    plan: dict[str, Any],
) -> None:
    """Admit only the exact reviewed runtime/custody receipt vocabulary."""

    for label, runtime in (("published marker", marker_runtime), ("live", runtime_now)):
        require(
            isinstance(runtime, dict)
            and set(runtime) == GATEWAY_RUNTIME_PREFLIGHT_FIELDS_V3,
            f"{label} runtime-preflight schema drifted",
        )
        require(
            runtime.get("format")
            == "apxinf-qwen35-omniinfer-gateway-runtime-preflight-v3"
            and runtime.get("schema_version") == 3
            and runtime.get("edge_id") == EDGE_ID
            and runtime.get("generation_requests") == 0
            and runtime.get("same_resident_backend_process_for_B_and_G") is True
            and runtime.get("backend_start_end_identity_equal") is True
            and runtime.get("gateway_start_end_identity_equal") is True
            and runtime.get("state_before_after_equal") is True
            and runtime.get("history_start_end_equal") is True
            and runtime.get("mutable_logs_equality_not_required") is True
            and runtime.get("all_passed") is True,
            f"{label} runtime-preflight admission drifted",
        )
        custody = runtime.get("controller_backend_model_fd_custody")
        require(
            isinstance(custody, dict)
            and set(custody) == GATEWAY_MODEL_FD_CUSTODY_RECEIPT_FIELDS_V3
            and custody.get("controller_and_backend_same_vnode_identity") is True
            and custody.get("same_open_file_description_not_claimed") is True
            and custody.get(
                "controller_fd_open_completed_before_gateway_backend_launch"
            )
            is True,
            f"{label} controller/backend model-FD custody schema drifted",
        )
        reject_unreviewed_same_ofd_claims(custody)
        require_same_custodian_attestation(custody["start"], custody["end"])
        binding = runtime.get("custodian_binding")
        require(isinstance(binding, dict), f"{label} custodian binding is absent")
        validate_campaign_directory_initialization_receipt(
            binding.get("campaign_directory_initialization"), plan
        )
        zero_generation_model_load = validate_zero_generation_model_load_receipt(
            runtime.get("zero_generation_model_load"),
            plan,
            gateway_pid=binding.get("gateway_pid"),
            gateway_start=binding.get("gateway_start_identity"),
            backend_pid=binding.get("backend_pid"),
            backend_start=binding.get("backend_start_identity"),
            custodian_pid=binding.get("custodian_pid"),
        )
        require(
            binding.get("zero_generation_model_load") == zero_generation_model_load
            and isinstance(binding.get("ready_receipt"), dict)
            and binding["ready_receipt"].get("zero_generation_model_load")
            == zero_generation_model_load,
            f"{label} custodian/model-load binding drifted",
        )

    for field in (
        "gateway_process_start",
        "backend_process_start",
        "custodian_binding",
        "zero_generation_model_load",
        "custodian_process_start",
        "state",
        "canonical_request",
        "rendered_prompt",
        "rendered_prompt_token_ids",
        "history_start",
    ):
        require(
            marker_runtime[field] == runtime_now[field],
            f"resident runtime changed after marker publication: {field}",
        )
    require_same_custodian_attestation(
        marker_runtime["controller_backend_model_fd_custody"]["start"],
        runtime_now["controller_backend_model_fd_custody"]["end"],
    )


def validate_custodian_attestation_receipt(
    attestation: dict[str, Any], expected_stage: str, expected_nonce: str
) -> str:
    require(isinstance(attestation, dict), "custodian attestation is absent")
    daemon = attestation.get("daemon_challenge_response")
    controller_external = attestation.get("controller_fd_external_proof")
    backend_external = attestation.get("backend_fd_external_proof")
    require(
        isinstance(daemon, dict)
        and isinstance(controller_external, dict)
        and isinstance(backend_external, dict),
        "custodian attestation proof sources are incomplete",
    )
    require(
        daemon.get("format") == CUSTODIAN_ATTESTATION_FORMAT
        and daemon.get("schema_version") == 3
        and daemon.get("edge_id") == EDGE_ID
        and daemon.get("stage") == expected_stage
        and daemon.get("nonce") == expected_nonce
        and re.fullmatch(r"[0-9a-f]{64}", daemon.get("challenge", "")) is not None
        and daemon.get("passed") is True
        and attestation.get("passed") is True
        and attestation.get("same_controller_and_backend_file_identity") is True,
        f"custodian attestation stage or admission drifted: {expected_stage}",
    )
    core = {
        key: value
        for key, value in daemon.items()
        if key not in ("challenge_response_sha256", "passed")
    }
    require(
        daemon.get("challenge_response_sha256") == sha256_canonical(core),
        "custodian attestation challenge digest drifted",
    )
    held = daemon.get("controller_preload_fd")
    backend_loaded = daemon.get("backend_loaded_fd")
    require(
        isinstance(held, dict)
        and held.get("format") == "apxinf-controller-held-model-fd-v3"
        and held.get("schema_version") == 3
        and held.get("stage") == expected_stage
        and held.get("controller_pid") == daemon.get("custodian_pid")
        and is_int(held.get("fd"))
        and held["fd"] >= 0
        and held.get("absolute_path") == MODEL_PATH
        and held.get("open_flags") == ["O_RDONLY", "O_NOFOLLOW", "O_CLOEXEC"]
        and held.get("fd_cloexec_observed") is True
        and held.get("sha256") == MODEL_SHA256,
        "controller preload FD receipt drifted",
    )
    require(isinstance(backend_loaded, dict), "backend loaded FD receipt is absent")
    controller_identity = _model_fd_identity(held)
    backend_identity = _model_fd_identity(backend_loaded)
    validate_controller_launch_sequence(daemon.get("lifecycle_sequence", {}))
    require(
        controller_identity == backend_identity
        and backend_loaded.get("fd_type") == "vnode"
        and is_int(backend_loaded.get("fd"))
        and backend_loaded["fd"] >= 0
        and backend_loaded.get("path") == MODEL_PATH
        and is_int(backend_loaded.get("open_flags"))
        and backend_loaded["open_flags"] & 3 == 1
        and backend_loaded.get("access") == "libproc-FREAD-without-FWRITE (O_RDONLY)"
        and backend_loaded.get("artifact_sha256_via_controller_fd") == MODEL_SHA256,
        "controller and backend do not name the same single-link model vnode",
    )
    controller_proc = controller_external.get("libproc_observation")
    controller_lsof = controller_external.get("lsof_observation")
    require(
        isinstance(controller_proc, dict)
        and controller_proc.get("fd") == held["fd"]
        and controller_proc.get("fd_type") == "vnode"
        and controller_proc.get("path") == MODEL_PATH
        and is_int(controller_proc.get("open_flags"))
        and controller_proc["open_flags"] & 3 == 1
        and controller_proc.get("close_on_exec") is True
        and _model_fd_identity(controller_proc) == controller_identity,
        "controller libproc FD proof drifted",
    )
    expected_controller_lsof = {
        "fd": held["fd"],
        "device": controller_identity["device"],
        "inode": controller_identity["inode"],
        "size_bytes": controller_identity["size_bytes"],
        "path": MODEL_PATH,
        "access": "read-only",
        "type": "REG",
    }
    require(
        controller_external.get("controller_pid") == daemon.get("custodian_pid")
        and controller_external.get("controller_held_fd") == held
        and controller_external.get("lsof_libproc_agree") is True
        and controller_external.get("O_RDONLY_O_NOFOLLOW_O_CLOEXEC_proven") is True,
        "controller external FD proof drifted",
    )
    require(
        controller_lsof == expected_controller_lsof,
        "controller lsof FD proof drifted",
    )
    expected_backend_lsof = {
        "fd": backend_loaded["fd"],
        "device": backend_identity["device"],
        "inode": backend_identity["inode"],
        "size_bytes": backend_identity["size_bytes"],
        "path": MODEL_PATH,
        "access": "read-only",
        "type": "REG",
    }
    require(
        backend_external.get("proof_format")
        == "macos-controller-backend-model-fd-crosscheck-v3"
        and backend_external.get("backend_pid") == daemon.get("backend_pid")
        and backend_external.get("controller_preload_fd_identity")
        == controller_identity
        and backend_external.get("backend_loaded_fd") == backend_loaded
        and backend_external.get("lsof_libproc_agree") is True
        and backend_external.get("controller_backend_file_identity_equal") is True,
        "backend libproc/lsof FD proof drifted",
    )
    require(
        backend_external.get("lsof_observation") == expected_backend_lsof,
        "backend lsof FD proof drifted",
    )
    return daemon["challenge"]


def validate_model_fd_checkpoint_schedule(receipt: dict[str, Any]) -> None:
    runtime = receipt.get("parity_admission", {})
    binding = runtime.get("custodian_binding", {})
    nonce = binding.get("nonce")
    require(
        re.fullmatch(r"[0-9a-f]{64}", nonce or "") is not None,
        "model-FD checkpoint nonce is invalid",
    )
    custody = runtime.get("controller_backend_model_fd_custody", {})
    start = custody.get("start")
    preflight_end = custody.get("end")
    initial = binding.get("initial_external_attestation")
    postflight = receipt.get("postflight", {})
    before_first = postflight.get("before_first_generation_custody")
    final = postflight.get("runtime_postflight", {}).get(
        "controller_backend_model_fd_custody_end"
    )
    fixed = (
        (initial, "prepare-before-any-generation"),
        (start, "runtime-preflight-start"),
        (preflight_end, "runtime-preflight-end"),
        (before_first, "before-first-generation"),
        (final, "runtime-postflight-before-cleanup"),
    )
    challenges: set[str] = set()
    for attestation, stage in fixed:
        challenge = validate_custodian_attestation_receipt(attestation, stage, nonce)
        require(
            challenge not in challenges, "custodian checkpoint challenge was replayed"
        )
        challenges.add(challenge)
        require_same_custodian_attestation(start, attestation)

    expected_segments = {7: "after-warmups"}
    expected_segments.update(
        {
            15 + macroblock * 8: f"after-timed-macroblock-{macroblock}"
            for macroblock in range(16)
        }
    )
    samples = receipt.get("samples")
    require(
        isinstance(samples, list) and len(samples) == 136,
        "model-FD checkpoint sample schedule is incomplete",
    )
    for index, sample in enumerate(samples):
        segment = sample.get("segment_state_after")
        if index not in expected_segments:
            require(
                segment is None, f"unexpected model-FD checkpoint at sample {index}"
            )
            continue
        require(
            isinstance(segment, dict)
            and segment.get("outside_primary_timed_interval") is True,
            f"model-FD checkpoint is absent at sample {index}",
        )
        attestation = segment.get("controller_backend_fd_custody")
        challenge = validate_custodian_attestation_receipt(
            attestation, expected_segments[index], nonce
        )
        require(
            challenge not in challenges, "custodian checkpoint challenge was replayed"
        )
        challenges.add(challenge)
        require_same_custodian_attestation(start, attestation)


def collect_runtime_preflight(
    plan_path: Path | str,
    plan: dict[str, Any],
    artifact_observations: dict[str, Any],
    custodian_binding: dict[str, Any],
) -> dict[str, Any]:
    """Collect zero-generation same-backend and request/trajectory prerequisites."""

    runtime = plan["runtime"]
    validate_campaign_directory_initialization_receipt(
        custodian_binding.get("campaign_directory_initialization"),
        plan,
        verify_live=True,
    )
    base_url = runtime["omni_base_url"]
    parsed = urllib.parse.urlsplit(base_url)
    require(parsed.port is not None, "gateway port is absent")
    custodian_attestation_start = attest_custodian(
        plan, custodian_binding, "runtime-preflight-start"
    )
    gateway_pid = listener_pid(parsed.port, "OmniInfer gateway")
    require(
        gateway_pid == custodian_binding["gateway_pid"],
        "gateway listener is not custodian-owned",
    )
    history_before = tree_manifest(runtime["history_root"])
    mutable_before = [tree_manifest(path) for path in runtime["mutable_log_roots"]]
    _, state_core, state_transport = _one_shot_json(
        base_url,
        "preflight-state",
        "GET",
        "/omni/state",
        None,
        lambda value: validate_gateway_state(value, plan),
    )
    backend_pid = state_core["backend_pid"]
    backend_port = state_core["backend_port"]
    require(
        listener_pid(backend_port, "resident backend") == backend_pid,
        "backend listener PID differs from state",
    )
    require(
        backend_pid == custodian_binding["backend_pid"],
        "backend is not custodian-owned",
    )
    require(
        _process_parent_pid(backend_pid) == gateway_pid,
        "resident backend is not the gateway child",
    )
    gateway_identity_start = collect_process_identity(
        gateway_pid,
        runtime["expected_gateway_argv"],
        plan["artifacts"]["omniinfer_cli"]["absolute_path"],
    )
    require(
        gateway_identity_start["disclosed_policy_environment"]
        == {"OMNIINFER_REQUEST_HISTORY": "0"},
        "OmniInfer request history is not exactly disabled",
    )
    _require_exact_process_environment(
        gateway_identity_start,
        runtime["expected_gateway_environment"],
        "gateway",
    )
    backend_identity_start = collect_process_identity(
        backend_pid,
        expected_backend_launch_args(plan, backend_port),
        plan["artifacts"]["gateway_backend"]["absolute_path"],
    )
    backend_identity_start["loaded_model_fd"] = copy.deepcopy(
        custodian_attestation_start["backend_fd_external_proof"]["backend_loaded_fd"]
    )
    custodian_process_start = collect_custodian_process_custody(
        plan_path, plan, custodian_binding
    )
    _, health, health_transport = _one_shot_json(
        base_url,
        "preflight-health",
        "GET",
        "/health?deep=true",
        None,
        validate_health,
    )
    _, props, props_transport = _one_shot_json(
        base_url,
        "preflight-props",
        "GET",
        "/omni/backend/props",
        None,
        validate_backend_props,
    )
    apply_body = canonical_json_bytes(
        {
            "add_generation_prompt": True,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"content": "Hello", "role": "user"}],
        }
    )
    _, rendered, apply_transport = _one_shot_json(
        state_core["client_endpoint"],
        "preflight-apply-template",
        "POST",
        "/apply-template",
        apply_body,
        _apply_template_validator,
    )
    tokenize_body = canonical_json_bytes(
        {
            "add_special": False,
            "content": rendered,
            "with_pieces": True,
        }
    )
    _, prompt_ids, tokenize_transport = _one_shot_json(
        base_url,
        "preflight-tokenize",
        "POST",
        "/tokenize",
        tokenize_body,
        _tokenize_ids,
    )
    _, cache_clear, cache_transport = _one_shot_json(
        base_url,
        "preflight-cache-clear",
        "POST",
        "/omni/cache/clear",
        b"{}",
        _cache_clear_validator,
    )
    _, state_after_core, state_after_transport = _one_shot_json(
        base_url,
        "preflight-state-after",
        "GET",
        "/omni/state",
        None,
        lambda value: validate_gateway_state(value, plan),
    )
    require(
        state_after_core == state_core,
        "gateway essential state changed during preflight",
    )
    backend_identity_end = collect_process_identity(
        backend_pid,
        expected_backend_launch_args(plan, backend_port),
        plan["artifacts"]["gateway_backend"]["absolute_path"],
    )
    gateway_identity_end = collect_process_identity(
        gateway_pid,
        runtime["expected_gateway_argv"],
        plan["artifacts"]["omniinfer_cli"]["absolute_path"],
    )
    _require_exact_process_environment(
        gateway_identity_end,
        runtime["expected_gateway_environment"],
        "gateway",
    )
    custodian_attestation_end = attest_custodian(
        plan, custodian_binding, "runtime-preflight-end"
    )
    require_same_custodian_attestation(
        custodian_attestation_start, custodian_attestation_end
    )
    backend_identity_end["loaded_model_fd"] = copy.deepcopy(
        custodian_attestation_end["backend_fd_external_proof"]["backend_loaded_fd"]
    )
    require_same_backend_identity(backend_identity_start, backend_identity_end)
    require(
        backend_identity_start == backend_identity_end,
        "backend runtime identity changed during preflight",
    )
    require(
        gateway_identity_start == gateway_identity_end,
        "gateway runtime identity changed during preflight",
    )
    custodian_process_end = collect_custodian_process_custody(
        plan_path, plan, custodian_binding
    )
    require(
        custodian_process_end == custodian_process_start,
        "custodian process/source/closure custody changed during preflight",
    )
    history_after = tree_manifest(runtime["history_root"])
    require(
        history_after == history_before, "request-history tree changed during preflight"
    )
    mutable_after = [tree_manifest(path) for path in runtime["mutable_log_roots"]]
    artifact_end = verify_plan_artifacts(plan)
    require(
        artifact_end == artifact_observations,
        "immutable artifacts changed during preflight",
    )
    return {
        "format": "apxinf-qwen35-omniinfer-gateway-runtime-preflight-v3",
        "schema_version": 3,
        "edge_id": EDGE_ID,
        "generation_requests": 0,
        "same_resident_backend_process_for_B_and_G": True,
        "direct_arm_backend_endpoint": state_core["client_endpoint"],
        "gateway_arm_endpoint": base_url,
        "gateway_process_start": gateway_identity_start,
        "gateway_process_end": gateway_identity_end,
        "backend_process_start": backend_identity_start,
        "backend_process_end": backend_identity_end,
        "backend_start_end_identity_equal": True,
        "gateway_start_end_identity_equal": True,
        "custodian_binding": copy.deepcopy(custodian_binding),
        "zero_generation_model_load": copy.deepcopy(
            custodian_binding["zero_generation_model_load"]
        ),
        "custodian_process_start": custodian_process_start,
        "custodian_process_end": custodian_process_end,
        "controller_backend_model_fd_custody": {
            "start": custodian_attestation_start,
            "end": custodian_attestation_end,
            "controller_and_backend_same_vnode_identity": True,
            "same_open_file_description_not_claimed": True,
            "controller_fd_open_completed_before_gateway_backend_launch": True,
        },
        "state": state_core,
        "state_before_after_equal": True,
        "health": health,
        "props": props,
        "canonical_request": {
            "object": copy.deepcopy(REQUEST),
            "size_bytes": len(REQUEST_BYTES),
            "sha256": sha256_bytes(REQUEST_BYTES),
            "B_G_body_identity_required": True,
        },
        "rendered_prompt": rendered,
        "rendered_prompt_token_ids": prompt_ids,
        "cache_clear": cache_clear,
        "history_start": history_before,
        "history_end": history_after,
        "history_start_end_equal": True,
        "mutable_logs_start": mutable_before,
        "mutable_logs_end": mutable_after,
        "mutable_logs_equality_not_required": True,
        "transport_receipts": {
            "state": state_transport,
            "health": health_transport,
            "props": props_transport,
            "apply_template": apply_transport,
            "tokenize": tokenize_transport,
            "cache_clear": cache_transport,
            "state_after": state_after_transport,
        },
        "all_passed": True,
    }


def collect_runtime_postflight(
    plan_path: Path | str,
    plan: dict[str, Any],
    artifact_observations: dict[str, Any],
    runtime_start: dict[str, Any],
) -> dict[str, Any]:
    base_url = plan["runtime"]["omni_base_url"]
    _, state_core, state_transport = _one_shot_json(
        base_url,
        "postflight-state",
        "GET",
        "/omni/state",
        None,
        lambda value: validate_gateway_state(value, plan),
    )
    backend_pid = runtime_start["backend_process_start"]["pid"]
    gateway_pid = runtime_start["gateway_process_start"]["pid"]
    require(
        state_core == runtime_start["state"], "gateway state changed during campaign"
    )
    backend_end = collect_process_identity(
        backend_pid,
        expected_backend_launch_args(plan, state_core["backend_port"]),
        plan["artifacts"]["gateway_backend"]["absolute_path"],
    )
    gateway_end = collect_process_identity(
        gateway_pid,
        plan["runtime"]["expected_gateway_argv"],
        plan["artifacts"]["omniinfer_cli"]["absolute_path"],
    )
    _require_exact_process_environment(
        gateway_end,
        plan["runtime"]["expected_gateway_environment"],
        "gateway",
    )
    custodian_end = attest_custodian(
        plan, runtime_start["custodian_binding"], "runtime-postflight-before-cleanup"
    )
    require_same_custodian_attestation(
        runtime_start["controller_backend_model_fd_custody"]["start"],
        custodian_end,
    )
    backend_end["loaded_model_fd"] = copy.deepcopy(
        custodian_end["backend_fd_external_proof"]["backend_loaded_fd"]
    )
    require_same_backend_identity(runtime_start["backend_process_start"], backend_end)
    require(
        runtime_start["backend_process_start"] == backend_end,
        "backend process/closure changed during campaign",
    )
    require(
        runtime_start["gateway_process_start"] == gateway_end,
        "gateway process/closure changed during campaign",
    )
    custodian_process_end = collect_custodian_process_custody(
        plan_path, plan, runtime_start["custodian_binding"]
    )
    require(
        custodian_process_end == runtime_start["custodian_process_start"],
        "custodian process/source/closure changed during campaign",
    )
    history_end = tree_manifest(plan["runtime"]["history_root"])
    require(
        history_end == runtime_start["history_start"],
        "request-history tree changed during campaign",
    )
    artifact_end = verify_plan_artifacts(plan)
    require(
        artifact_end == artifact_observations,
        "immutable artifacts changed during campaign",
    )
    return {
        "state": state_core,
        "state_transport": state_transport,
        "backend_process_end": backend_end,
        "gateway_process_end": gateway_end,
        "custodian_process_end": custodian_process_end,
        "controller_backend_model_fd_custody_end": custodian_end,
        "history_end": history_end,
        "mutable_logs_end": [
            tree_manifest(path) for path in plan["runtime"]["mutable_log_roots"]
        ],
        "artifact_file_observations_end": artifact_end,
        "passed": True,
    }


HOST_INTERVAL_TOLERANCE_MS = _SHARED.HOST_CPU_WINDOW_MAX_OVERRUN_MS


def validate_actual_host_windows(
    receipt: dict[str, Any], contract: dict[str, Any]
) -> None:
    expected = contract["host_quiet_gate"]["continuous_monitor"]["sample_interval_ms"]
    require(expected == 250, "host interval contract drifted")
    snapshots = receipt.get("snapshots")
    require(isinstance(snapshots, list) and snapshots, "host receipt has no snapshots")
    previous_end: int | None = None
    for snapshot in snapshots:
        start = snapshot.get("cpu_window_start_monotonic_ns")
        end = snapshot.get("monotonic_ns")
        window = snapshot.get("cpu_percent_window_ms")
        require(
            is_int(start)
            and is_int(end)
            and 0 <= start < end
            and isinstance(window, (int, float))
            and not isinstance(window, bool)
            and math.isfinite(float(window))
            and math.isclose(
                float(window),
                (end - start) / 1_000_000,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
            and expected <= float(window) <= expected + HOST_INTERVAL_TOLERANCE_MS,
            "host CPU sampling window is not the actual bounded 250ms window",
        )
        require(
            previous_end is None or start == previous_end,
            "host CPU sampling windows are not contiguous",
        )
        require(
            snapshot.get("system_state_matches_gate_start") is True,
            "power, thermal, swap, or memory state was not checked unchanged in a host window",
        )
        previous_end = end


class GatewayQuietHostProbe(_SHARED.MacQuietHostProbe):
    """Native probe with exact persistent gateway/backend birth identities."""

    def __init__(
        self,
        contract: dict[str, Any],
        process_start_reader: Any = process_start_identity,
    ):
        super().__init__(contract)
        self.process_start_reader = process_start_reader

    def _allowlist(
        self,
        inventory: dict[int, dict[str, Any]],
        active_runtime: dict[str, Any] | None,
    ) -> tuple[set[tuple[int, str]], list[dict[str, Any]], set[int]]:
        orchestrator = inventory.get(self._orchestrator_pid)
        require(orchestrator is not None, "campaign orchestrator disappeared")
        active_pids: set[int] = set()
        if active_runtime is not None:
            custodian_pid = active_runtime["custodian_pid"]
            gateway_pid = active_runtime["gateway_pid"]
            backend_pid = active_runtime["backend_pid"]
            require(
                custodian_pid in inventory,
                "custodian exited during continuous custody",
            )
            require(
                gateway_pid in inventory, "gateway exited during continuous custody"
            )
            require(
                backend_pid in inventory, "backend exited during continuous custody"
            )
            require(
                self.process_start_reader(custodian_pid)
                == active_runtime["custodian_process_start_identity"],
                "custodian PID was reused during continuous custody",
            )
            require(
                self.process_start_reader(gateway_pid)
                == active_runtime["gateway_process_start_identity"],
                "gateway PID was reused during continuous custody",
            )
            require(
                self.process_start_reader(backend_pid)
                == active_runtime["backend_process_start_identity"],
                "backend PID was reused during continuous custody",
            )
            active_pids = self._descendants(inventory, custodian_pid)
            require(
                gateway_pid in active_pids and backend_pid in active_pids,
                "gateway/backend left the custodian process tree",
            )
        helper_prefixes = (
            "/bin/ps -axo ",
            "/bin/ps -p ",
            "/usr/bin/footprint --swapped -f bytes -p ",
            "/usr/bin/pmset -g ",
            "/usr/sbin/lsof ",
            "/usr/bin/codesign ",
            "/usr/sbin/sysctl ",
            "/usr/bin/vm_stat",
            "/usr/bin/sw_vers",
            "/usr/bin/uname -m",
        )
        helper_pids = {
            pid
            for pid, process in inventory.items()
            if process["ppid"] == self._orchestrator_pid
            and process["command"].startswith(helper_prefixes)
        }
        allowed_pids = {self._orchestrator_pid, *active_pids, *helper_pids}
        identities = {
            (pid, inventory[pid]["process_start_time"])
            for pid in allowed_pids
            if pid in inventory
        }
        resolved = [
            {
                "role": "campaign_orchestrator",
                "pid": self._orchestrator_pid,
                "process_start_time": orchestrator["process_start_time"],
                "executable_path": self._executable["absolute_path"],
                "executable_sha256": self._executable["sha256"],
                "argv_sha256": self._argv_sha256,
                "process_group_id": orchestrator["process_group_id"],
            },
            {
                "role": "custody_monitor",
                "pid": self._orchestrator_pid,
                "process_start_time": orchestrator["process_start_time"],
                "executable_path": self._executable["absolute_path"],
                "executable_sha256": self._executable["sha256"],
                "argv_sha256": self._argv_sha256,
                "thread_native_id": threading.get_native_id(),
            },
        ]
        if active_runtime is not None:
            resolved.append(
                {
                    "role": "active_measured_runtime_tree",
                    "root_pid": active_runtime["custodian_pid"],
                    "root_process_start_time": inventory[
                        active_runtime["custodian_pid"]
                    ]["process_start_time"],
                    "root_kernel_start_identity": active_runtime[
                        "custodian_process_start_identity"
                    ],
                    "gateway_pid": active_runtime["gateway_pid"],
                    "gateway_kernel_start_identity": active_runtime[
                        "gateway_process_start_identity"
                    ],
                    "backend_pid": active_runtime["backend_pid"],
                    "backend_process_start_time": inventory[
                        active_runtime["backend_pid"]
                    ]["process_start_time"],
                    "backend_kernel_start_identity": active_runtime[
                        "backend_process_start_identity"
                    ],
                    "edge_id": EDGE_ID,
                    "descendant_pid_start_time_pairs": [
                        [pid, inventory[pid]["process_start_time"]]
                        for pid in sorted(active_pids)
                    ],
                    "executable_and_library_hashes": copy.deepcopy(
                        active_runtime["executable_and_library_hashes"]
                    ),
                }
            )
        return identities, resolved, active_pids

    def snapshot(
        self, index: int, active_runtime: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        snapshot = super().snapshot(index, active_runtime)
        if active_runtime is not None:
            require(
                self.process_start_reader(active_runtime["custodian_pid"])
                == active_runtime["custodian_process_start_identity"]
                and self.process_start_reader(active_runtime["gateway_pid"])
                == active_runtime["gateway_process_start_identity"]
                and self.process_start_reader(active_runtime["backend_pid"])
                == active_runtime["backend_process_start_identity"],
                "managed runtime PID identity changed during a host snapshot",
            )
            complete = (
                snapshot.get("active_runtime_root_present") is True
                and snapshot.get("active_runtime_swap_proof_complete") is True
                and snapshot.get("campaign_swap_probe_vanished_processes") == []
            )
            snapshot["managed_runtime_swap_proof_complete"] = complete
            snapshot["passed"] = snapshot.get("passed") is True and complete
        else:
            snapshot["managed_runtime_swap_proof_complete"] = None
        return snapshot


def gateway_host_active_receipt(runtime_identity: dict[str, Any]) -> dict[str, Any]:
    binding = runtime_identity["custodian_binding"]
    gateway = runtime_identity["gateway_process_start"]
    backend = runtime_identity["backend_process_start"]
    return {
        "pid": binding["custodian_pid"],
        "custodian_pid": binding["custodian_pid"],
        "custodian_process_start_identity": binding["custodian_start_identity"],
        "gateway_pid": gateway["pid"],
        "gateway_process_start_identity": gateway["process_start_identity"],
        "backend_pid": backend["pid"],
        "backend_process_start_identity": backend["process_start_identity"],
        "executable_and_library_hashes": {
            "custodian_runtime_closure_sha256": runtime_identity[
                "custodian_process_start"
            ]["custodian_process"]["runtime_closure_sha256"],
            "gateway_runtime_closure_sha256": gateway["runtime_closure_sha256"],
            "backend_runtime_closure_sha256": backend["runtime_closure_sha256"],
        },
    }


def collect_gateway_host_gate(
    phase: str,
    contract: dict[str, Any],
    *,
    runtime_identity: dict[str, Any] | None = None,
    sleeper: Any = time.sleep,
) -> dict[str, Any]:
    require(phase in ("preflight", "postflight"), "host gate phase is invalid")
    probe = GatewayQuietHostProbe(contract)
    start_power = _SHARED._collect_power_thermal_state()
    swap_start = _SHARED._system_swap_used_bytes()
    throttled_start = _SHARED._pages_throttled()
    phase_contract = contract["host_quiet_gate"][phase]
    if phase == "postflight":
        sleeper(phase_contract["cooldown_before_snapshots_ms"] / 1000)
    active = (
        gateway_host_active_receipt(runtime_identity)
        if runtime_identity is not None
        else None
    )
    probe.prime(active)
    snapshots = []
    for index in range(phase_contract["snapshot_count"]):
        sleeper(probe.seconds_until_next_window(phase_contract["snapshot_interval_ms"]))
        snapshots.append(
            _SHARED._attach_snapshot_system_state(
                probe.snapshot(index, active), start_power, swap_start, throttled_start
            )
        )
    end_power = _SHARED._collect_power_thermal_state()
    receipt = _SHARED._host_gate_receipt(
        contract,
        phase,
        probe.host,
        start_power,
        end_power,
        swap_start,
        _SHARED._system_swap_used_bytes(),
        throttled_start,
        _SHARED._pages_throttled(),
        snapshots,
    )
    receipt["continuous_power_thermal_observation_per_snapshot"] = True
    receipt["actual_interval_tolerance_ms"] = HOST_INTERVAL_TOLERANCE_MS
    _SHARED.validate_host_receipt(receipt, phase, contract)
    validate_actual_host_windows(receipt, contract)
    return receipt


class GatewayContinuousHostMonitor:
    """250-ms continuous host/runtime custody with no PID-only allowance."""

    def __init__(
        self,
        contract: dict[str, Any],
        runtime_identity: dict[str, Any],
        *,
        process_start_reader: Any = process_start_identity,
    ):
        self.contract = contract
        self.runtime_identity = copy.deepcopy(runtime_identity)
        self.process_start_reader = process_start_reader
        self.probe = GatewayQuietHostProbe(contract, process_start_reader)
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._thread: threading.Thread | None = None
        self._failure: BaseException | None = None
        self._snapshots: list[dict[str, Any]] = []
        self._start_power: dict[str, Any] | None = None
        self._swap_start: int | None = None
        self._throttled_start: int | None = None

    def _active_receipt(self) -> dict[str, Any]:
        return gateway_host_active_receipt(self.runtime_identity)

    def start(self) -> None:
        require(self._thread is None, "continuous host monitor already started")
        active = self._active_receipt()
        self._start_power = _SHARED._collect_power_thermal_state()
        self._swap_start = _SHARED._system_swap_used_bytes()
        self._throttled_start = _SHARED._pages_throttled()
        self.probe.prime(active)
        self._thread = threading.Thread(
            target=self._run,
            name="gateway-formal-v3-custody-monitor",
            daemon=False,
        )
        self._thread.start()

    def _run(self) -> None:
        interval_ms = self.contract["host_quiet_gate"]["continuous_monitor"][
            "sample_interval_ms"
        ]
        active = self._active_receipt()
        while True:
            delay = self.probe.seconds_until_next_window(interval_ms)
            if self._stop.wait(delay):
                return
            try:
                snapshot = self.probe.snapshot(len(self._snapshots), active)
                _SHARED._attach_snapshot_system_state(
                    snapshot,
                    self._start_power,
                    self._swap_start,
                    self._throttled_start,
                )
                window = float(snapshot["cpu_percent_window_ms"])
                require(
                    interval_ms <= window <= interval_ms + HOST_INTERVAL_TOLERANCE_MS,
                    "continuous host monitor missed the actual 250ms window",
                )
                self._snapshots.append(snapshot)
                self._ready.set()
                require(snapshot["passed"], "continuous quiet-host snapshot failed")
            except BaseException as error:
                self._failure = error
                self._ready.set()
                self._stop.set()
                return

    def wait_until_ready(self, timeout_seconds: float = 30.0) -> None:
        require(
            self._ready.wait(timeout_seconds),
            "continuous host monitor produced no snapshot",
        )
        self.assert_healthy()

    def assert_healthy(self) -> None:
        if self._failure is not None:
            raise CampaignError(f"continuous host gate failed: {self._failure}")
        active = self._active_receipt()
        require(
            self.process_start_reader(active["custodian_pid"])
            == active["custodian_process_start_identity"],
            "custodian exited or PID was reused between monitor snapshots",
        )
        require(
            self.process_start_reader(active["gateway_pid"])
            == active["gateway_process_start_identity"],
            "gateway exited or PID was reused between monitor snapshots",
        )
        require(
            self.process_start_reader(active["backend_pid"])
            == active["backend_process_start_identity"],
            "backend exited or PID was reused between monitor snapshots",
        )

    def stop_and_receipt(self) -> dict[str, Any]:
        require(self._thread is not None, "continuous host monitor was never started")
        self._stop.set()
        self._thread.join(timeout=30.0)
        require(not self._thread.is_alive(), "continuous host monitor did not stop")
        self.assert_healthy()
        end_power = _SHARED._collect_power_thermal_state()
        receipt = _SHARED._host_gate_receipt(
            self.contract,
            "continuous",
            self.probe.host,
            self._start_power,
            end_power,
            self._swap_start,
            _SHARED._system_swap_used_bytes(),
            self._throttled_start,
            _SHARED._pages_throttled(),
            self._snapshots,
        )
        receipt["continuous_power_thermal_observation_per_snapshot"] = True
        receipt["actual_interval_tolerance_ms"] = HOST_INTERVAL_TOLERANCE_MS
        _SHARED.validate_host_receipt(receipt, "continuous", self.contract)
        validate_actual_host_windows(receipt, self.contract)
        return receipt


def open_campaign_connections(
    plan: dict[str, Any], runtime_preflight: dict[str, Any]
) -> tuple[
    PersistentHttpJsonConnection,
    PersistentHttpJsonConnection,
    PersistentHttpJsonConnection,
    dict[str, Any],
]:
    admin = PersistentHttpJsonConnection(
        plan["runtime"]["omni_base_url"], "gateway-admin"
    )
    backend = PersistentHttpJsonConnection(
        runtime_preflight["direct_arm_backend_endpoint"], "B-direct-measurement"
    )
    gateway = PersistentHttpJsonConnection(
        runtime_preflight["gateway_arm_endpoint"], "G-gateway-measurement"
    )
    try:
        sockets = {
            "admin": admin.connect(),
            "B": backend.connect(),
            "G": gateway.connect(),
        }
        _, _, b_warm = backend.request_json(
            "GET",
            "/health",
            None,
            lambda value: (
                receipt_require(
                    value.get("status") == "ok", "backend connection warm health failed"
                )
                or {"status": "ok"}
            ),
        )
        _, _, g_warm = gateway.request_json(
            "GET", "/health?deep=true", None, validate_health
        )
    except BaseException:
        for connection in (gateway, backend, admin):
            try:
                connection.close()
            except BaseException:
                pass
        raise
    return (
        admin,
        backend,
        gateway,
        {
            "sockets": sockets,
            "warm_non_generation_requests": {"B": b_warm, "G": g_warm},
            "generation_requests_before_schedule": 0,
            "one_persistent_connection_per_arm": True,
        },
    )


def _sample_from_connections(
    slot: dict[str, Any],
    plan: dict[str, Any],
    contract: dict[str, Any],
    admin: PersistentHttpJsonConnection,
    backend: PersistentHttpJsonConnection,
    gateway: PersistentHttpJsonConnection,
    monitor: GatewayContinuousHostMonitor,
    pair_buffer: list[dict[str, Any]],
) -> dict[str, Any]:
    monitor.assert_healthy()
    _, clear, clear_transport = admin.request_json(
        "POST", "/omni/cache/clear", b"{}", _cache_clear_validator
    )
    monitor.assert_healthy()
    target = backend if slot["arm"] == "B" else gateway
    payload, validated, transport = target.request_json(
        "POST",
        "/v1/chat/completions",
        REQUEST_BYTES,
        lambda response: validate_gateway_response(response, contract, MODEL_PATH),
    )
    raw_generation_response = bytes(target.last_raw_response)
    require(
        len(raw_generation_response) == transport["response_size_bytes"]
        and sha256_bytes(raw_generation_response) == transport["response_sha256"],
        "generation response bytes changed after timed admission",
    )
    monitor.assert_healthy()
    tokenize_body = canonical_json_bytes(
        {
            "add_special": False,
            "content": validated["rendered_prompt"],
            "with_pieces": True,
        }
    )
    _, prompt_ids, tokenize_transport = admin.request_json(
        "POST", "/tokenize", tokenize_body, _tokenize_ids
    )
    receipt_require(
        prompt_ids == validated["prompt_token_ids"] == PROMPT_TOKEN_IDS,
        "per-sample raw rendered prompt IDs drifted",
    )
    timing_contract = contract["timing_contract"][EDGE_ID]
    sample = {
        "format": SAMPLE_FORMAT,
        "schema_version": 3,
        "campaign_id": contract["campaign_id"],
        "subcampaign_id": contract["comparison_graph"]["edges"][EDGE_ID][
            "subcampaign_id"
        ],
        "edge_id": EDGE_ID,
        "slot": copy.deepcopy(slot),
        "request": {
            "canonical_json_object": copy.deepcopy(REQUEST),
            "canonical_utf8": REQUEST_BYTES.decode("utf-8"),
            "size_bytes": len(REQUEST_BYTES),
            "sha256": sha256_bytes(REQUEST_BYTES),
            "same_body_for_B_and_G": True,
        },
        "cache_clear": {
            **clear,
            "outside_primary_timed_interval": True,
            "transport": clear_transport,
        },
        "workload": {
            "rendered_prompt": validated["rendered_prompt"],
            "prompt_token_ids": prompt_ids,
            "generated_token_ids": validated["generated_token_ids"],
            "generated_token_ids_sha256": validated["generated_token_ids_sha256"],
            "content": validated["content"],
            "content_sha256": validated["content_sha256"],
            "usage": validated["usage"],
            "usage_object": validated["usage_object"],
            "generation_settings": validated["generation_settings"],
            "generation_settings_sha256": validated["generation_settings_sha256"],
            "per_sample_tokenize_admission_outside_timed_interval": {
                "token_ids": prompt_ids,
                "transport": tokenize_transport,
            },
        },
        "timing": {
            "clock": transport["clock"],
            "clock_identity": transport["clock_identity"],
            "clock_resolution_ns": transport["clock_resolution_ns"],
            "clock_is_monotonic": transport["clock_is_monotonic"],
            "clock_is_adjustable": transport["clock_is_adjustable"],
            "start_boundary": timing_contract["start"],
            "end_boundary": timing_contract["end"],
            "implementation_start_boundary": transport["start_boundary"],
            "implementation_end_boundary": transport["end_boundary"],
            "request_serialization_before_start": transport[
                "request_serialization_before_start"
            ],
            "first_wire_byte_send_call_immediately_after_start": transport[
                "first_wire_byte_send_call_immediately_after_start"
            ],
            "request_wire_size_bytes": transport["request_wire_size_bytes"],
            "request_wire_sha256": transport["request_wire_sha256"],
            "request_wire_base64": transport["request_wire_base64"],
            "request_wire_body_offset_bytes": transport[
                "request_wire_body_offset_bytes"
            ],
            "request_wire_body_size_bytes": transport["request_wire_body_size_bytes"],
            "request_wire_body_sha256": transport["request_wire_body_sha256"],
            "request_wire_body_equals_request_body": transport[
                "request_wire_body_equals_request_body"
            ],
            "single_sendall_call_count": transport["single_sendall_call_count"],
            "single_sendall_argument_size_bytes": transport[
                "single_sendall_argument_size_bytes"
            ],
            "single_sendall_argument_sha256": transport[
                "single_sendall_argument_sha256"
            ],
            "timing_event_order": copy.deepcopy(transport["timing_event_order"]),
            "complete_HTTP_request_wire_serialization_before_start": transport[
                "complete_HTTP_request_wire_serialization_before_start"
            ],
            "single_sendall_call_for_complete_request_wire_required": transport[
                "single_sendall_call_for_complete_request_wire_required"
            ],
            "canonical_383_byte_JSON_body_identical_between_B_and_G": transport[
                "canonical_383_byte_JSON_body_identical_between_B_and_G"
            ],
            "arm_specific_HTTP_authority_header_difference_is_inside_timed_region": transport[
                "arm_specific_HTTP_authority_header_difference_is_inside_timed_region"
            ],
            "body_only_timing_allowed": transport["body_only_timing_allowed"],
            "full_response_body_read_before_end": transport[
                "full_response_body_read_before_end"
            ],
            "strict_json_parse_before_end": transport["strict_json_parse_before_end"],
            "semantic_validation_before_end": transport[
                "semantic_validation_before_end"
            ],
            "json_parse_excluded_from_wall": transport["json_parse_excluded_from_wall"],
            "semantic_validation_excluded_from_wall": transport[
                "semantic_validation_excluded_from_wall"
            ],
            "started_monotonic_ns": transport["started_monotonic_ns"],
            "ended_monotonic_ns": transport["ended_monotonic_ns"],
            "client_full_response_wall_ns": transport["client_full_response_wall_ns"],
            "client_full_response_wall_ms": transport["client_full_response_wall_ms"],
        },
        "native_sensitivity": copy.deepcopy(validated["native"]),
        "connection": {
            "connection_generation": transport["connection_generation"],
            "request_index_on_connection": transport["request_index_on_connection"],
            "socket": transport["socket"],
            "socket_start_end_equal": transport["socket_start_end_equal"],
            "reconnect_count": transport["reconnect_count"],
        },
        "response": payload,
        "response_bytes": {
            "encoding": "base64",
            "base64": base64.b64encode(raw_generation_response).decode("ascii"),
            "size_bytes": transport["response_size_bytes"],
            "sha256": transport["response_sha256"],
        },
    }
    pair_buffer.append(sample)
    if len(pair_buffer) == 2:
        validate_pair_equal(pair_buffer[0], pair_buffer[1])
        pair_buffer.clear()
    elif len(pair_buffer) > 2:
        raise CampaignError("adjacent pair buffer overflowed")
    end_of_subcampaign_segment = slot["slot_index_in_subblock"] == 3 and (
        (slot["phase"] == "warmup" and slot["warmup_subblock_index"] == 1)
        or (slot["phase"] == "timed" and slot["subblock_index"] == 1)
    )
    if end_of_subcampaign_segment:
        monitor.assert_healthy()
        custody = attest_custodian(
            plan,
            monitor.runtime_identity["custodian_binding"],
            (
                "after-warmups"
                if slot["phase"] == "warmup"
                else f"after-timed-macroblock-{slot['macroblock_index']}"
            ),
        )
        require_same_custodian_attestation(
            monitor.runtime_identity["controller_backend_model_fd_custody"]["start"],
            custody,
        )
        _, state_core, state_transport = admin.request_json(
            "GET",
            "/omni/state",
            None,
            lambda value: validate_gateway_state(value, plan),
        )
        expected_backend = monitor.runtime_identity["backend_process_start"]
        require(
            state_core["backend_pid"] == expected_backend["pid"]
            and process_start_identity(state_core["backend_pid"])
            == expected_backend["process_start_identity"],
            "resident backend identity changed at schedule segment boundary",
        )
        sample["segment_state_after"] = {
            "state": state_core,
            "transport": state_transport,
            "controller_backend_fd_custody": custody,
            "outside_primary_timed_interval": True,
        }
        monitor.assert_healthy()
    return sample


def collect_sample_with_failed_raw_retention(
    slot: dict[str, Any],
    backend: PersistentHttpJsonConnection,
    gateway: PersistentHttpJsonConnection,
    collector: Any,
) -> dict[str, Any]:
    """Retain exact generation bytes when later per-sample admission fails."""

    target = backend if slot["arm"] == "B" else gateway
    requests_before = target.request_count
    try:
        return collector()
    except BaseException as error:
        if target.request_count == requests_before + 1:
            observation = {
                "stage": "after-generation-before-sample-admission-complete",
                "arm": slot["arm"],
                "sequence_index": slot["sequence_index"],
                "raw_generation_response": target.last_raw_response,
                "raw_generation_response_size_bytes": len(target.last_raw_response),
                "raw_generation_response_sha256": sha256_bytes(
                    target.last_raw_response
                ),
                "downstream_exception_type": type(error).__name__,
                "downstream_message": str(error),
                "downstream_observation": (
                    error.observation
                    if isinstance(error, RuntimeObservationError)
                    else {}
                ),
            }
            raise RuntimeObservationError(
                f"sample admission failed after generation response: {error}",
                observation,
            ) from error
        raise


def validate_connection_postflight(
    backend: PersistentHttpJsonConnection,
    gateway: PersistentHttpJsonConnection,
    admin: PersistentHttpJsonConnection,
) -> dict[str, Any]:
    expected_generation_requests = 68
    require(
        backend.request_count == expected_generation_requests + 1,
        "B persistent connection request count drifted",
    )
    require(
        gateway.request_count == expected_generation_requests + 1,
        "G persistent connection request count drifted",
    )
    require(
        backend.reconnect_count
        == gateway.reconnect_count
        == admin.reconnect_count
        == 0,
        "a campaign connection reconnected",
    )
    require(
        backend.socket_identity() == backend.baseline
        and gateway.socket_identity() == gateway.baseline
        and admin.socket_identity() == admin.baseline,
        "a persistent campaign socket changed before postflight",
    )
    return {
        "B": {
            "socket_start": copy.deepcopy(backend.baseline),
            "socket_end": backend.socket_identity(),
            "socket_start_end_equal": backend.socket_identity() == backend.baseline,
            "request_count": backend.request_count,
            "generation_request_count": 68,
            "reconnect_count": 0,
        },
        "G": {
            "socket_start": copy.deepcopy(gateway.baseline),
            "socket_end": gateway.socket_identity(),
            "socket_start_end_equal": gateway.socket_identity() == gateway.baseline,
            "request_count": gateway.request_count,
            "generation_request_count": 68,
            "reconnect_count": 0,
        },
        "admin": {
            "socket_start": copy.deepcopy(admin.baseline),
            "socket_end": admin.socket_identity(),
            "socket_start_end_equal": admin.socket_identity() == admin.baseline,
            "request_count": admin.request_count,
            "reconnect_count": 0,
        },
        "passed": True,
    }


def gateway_blocker_resolution(
    contract: dict[str, Any],
    *,
    driver_evidence: Any,
    quiet_host_evidence: Any,
    runtime_evidence: Any,
    public_activation_evidence: Any | None,
) -> dict[str, Any]:
    readiness = contract["runtime_custody"]["gateway_cohort"]["current_readiness"]
    require(
        tuple(readiness["blocker_codes"]) == GATEWAY_AUTHORED_BLOCKERS
        and readiness["formally_admitted"] is False,
        "gateway authored blocker set drifted",
    )
    resolutions = {
        GATEWAY_AUTHORED_BLOCKERS[0]: {
            "resolved": public_activation_evidence is not None,
            "evidence": public_activation_evidence,
        },
        GATEWAY_AUTHORED_BLOCKERS[1]: {
            "resolved": driver_evidence is not None,
            "evidence": driver_evidence,
        },
        GATEWAY_AUTHORED_BLOCKERS[2]: {
            "resolved": quiet_host_evidence is not None,
            "evidence": quiet_host_evidence,
        },
        "SAME_RESIDENT_BACKEND_AND_CONTROLLER_MODEL_FD_CUSTODY": {
            "resolved": runtime_evidence is not None,
            "evidence": runtime_evidence,
        },
    }
    pre_marker = all(
        value["resolved"]
        for key, value in resolutions.items()
        if key != "V3_GATEWAY_SUBCAMPAIGN_NOT_PUBLICLY_ACTIVATED"
    )
    return {
        "authored_formally_admitted": False,
        "authored_blocker_codes": list(GATEWAY_AUTHORED_BLOCKERS),
        "resolution_map": resolutions,
        "all_pre_marker_blockers_except_public_activation_resolved": pre_marker,
        "all_resolved": all(value["resolved"] for value in resolutions.values()),
        "authored_state_was_not_mutated": True,
    }


def machine_artifact_custody(
    plan: dict[str, Any], observations: dict[str, Any], runtime: dict[str, Any]
) -> dict[str, Any]:
    return {
        "model": {
            "absolute_path": MODEL_PATH,
            "size_bytes": MODEL_SIZE,
            "sha256": MODEL_SHA256,
            "O_NOFOLLOW_observation": copy.deepcopy(observations["model"]),
        },
        "llama_cpp": {
            "source_commit": PINNED_CORE_SOURCE_COMMIT,
            "inactive_core_lane_contract_binding_only": True,
        },
        "omniinfer": {
            "source_commit": OMNI_SOURCE_COMMIT,
            "release_archive_sha256": OMNI_ARCHIVE_SHA256,
            "cli": copy.deepcopy(observations["omniinfer_cli"]),
            "runtime_process": copy.deepcopy(runtime["gateway_process_start"]),
        },
        "gateway_backend": {
            "source_commit": BACKEND_SOURCE_COMMIT,
            "release_archive_sha256": BACKEND_ARCHIVE_SHA256,
            "llama_server": copy.deepcopy(observations["gateway_backend"]),
            "runtime_process": copy.deepcopy(runtime["backend_process_start"]),
            "controller_backend_model_fd_custody": copy.deepcopy(
                runtime["controller_backend_model_fd_custody"]
            ),
            "custodian_process": copy.deepcopy(runtime["custodian_process_start"]),
        },
        "file_observations_start": copy.deepcopy(observations),
    }


def _read_tracked_receipt(
    repository_root: Path,
    repository_path: str,
    tracked: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    raw, file_receipt = _read_regular_no_follow(
        repository_root / repository_path, 64 * 1024 * 1024
    )
    require(
        file_receipt["size_bytes"] == tracked["blob_size_bytes"]
        and file_receipt["sha256"] == tracked["blob_sha256"],
        "tracked receipt differs from Git blob",
    )
    return parse_strict_json_line(raw), file_receipt


def _prove_ancestor(repository_root: Path, ancestor: str, descendant: str) -> None:
    result = hardened_command_runner(
        [
            "/usr/bin/git",
            "merge-base",
            "--is-ancestor",
            ancestor,
            descendant,
        ],
        repository_root,
        30.0,
    )
    require(
        result["returncode"] == 0
        and result["stdout"] == b""
        and result["stderr"] == b"",
        "pre-marker public commit is not an ancestor of activation commit",
    )


def final_contract_binding(
    context: dict[str, Any], git_custody: dict[str, Any]
) -> dict[str, Any]:
    plan = context["plan"]
    tracked = git_custody["tracked_files"]
    contract_file = tracked["contract"]
    marker_file = tracked["activation_marker"]
    return {
        "campaign_id": context["contract"]["campaign_id"],
        "schema_version": 3,
        "edge_id": EDGE_ID,
        "subcampaign_id": context["contract"]["comparison_graph"]["edges"][EDGE_ID][
            "subcampaign_id"
        ],
        "repository_url": git_custody["repository_url"],
        "remote_origin_url": git_custody["remote_origin_url"],
        "local_tracking_ref": git_custody["local_tracking_ref"],
        "local_tracking_oid": git_custody["local_tracking_oid"],
        "live_remote_url": git_custody["live_remote_url"],
        "live_remote_ref": git_custody["live_remote_ref"],
        "ls_remote_argv": git_custody["ls_remote_argv"],
        "ls_remote_exit_code": git_custody["ls_remote_exit_code"],
        "ls_remote_live_oid": git_custody["ls_remote_live_oid"],
        "head_commit": git_custody["head_commit"],
        "contract_repository_path": plan["contract_repository_path"],
        "contract_commit": git_custody["contract_commit"],
        "contract_tree": git_custody["contract_tree"],
        "contract_blob_oid": contract_file["blob_oid"],
        "contract_blob_size_bytes": contract_file["blob_size_bytes"],
        "contract_blob_sha256": contract_file["blob_sha256"],
        "observed_file_size_bytes": contract_file["observed_size_bytes"],
        "observed_file_sha256": contract_file["observed_sha256"],
        "gateway_driver_repository_path": plan["driver_repository_path"],
        "gateway_driver_blob_sha256": tracked["driver"]["blob_sha256"],
        "validator_repository_path": plan["validator_repository_path"],
        "validator_blob_sha256": tracked["validator"]["blob_sha256"],
        "shared_formal_driver_repository_path": SHARED_DRIVER_REPOSITORY_PATH,
        "shared_formal_driver_blob_sha256": tracked["shared_formal_driver"][
            "blob_sha256"
        ],
        "plan_repository_path": plan["plan_repository_path"],
        "plan_blob_sha256": tracked["plan"]["blob_sha256"],
        "activation_commit": git_custody["activation_commit"],
        "activation_tree": git_custody["head_tree"],
        "activation_contract_blob_oid": contract_file["blob_oid"],
        "activation_contract_blob_size_bytes": contract_file["blob_size_bytes"],
        "activation_contract_blob_sha256": contract_file["blob_sha256"],
        "activation_marker_repository_path": plan["marker_repository_path"],
        "activation_marker_blob_oid": marker_file["blob_oid"],
        "activation_marker_blob_size_bytes": marker_file["blob_size_bytes"],
        "activation_marker_blob_sha256": marker_file["blob_sha256"],
        "contract_commit_is_ancestor_of_activation_commit": True,
        "activation_commit_equals_head_and_live_remote_oid": True,
        "local_tracking_ref_used_as_publication_proof": False,
        "worktree_clean": True,
    }


def build_marker(
    context: dict[str, Any],
    git_custody: dict[str, Any],
    artifacts: dict[str, Any],
    runtime_preflight: dict[str, Any],
    host_preflight: dict[str, Any],
    resolution: dict[str, Any],
) -> dict[str, Any]:
    plan = context["plan"]
    contract = context["contract"]
    tracked = git_custody["tracked_files"]
    require(
        resolution["all_pre_marker_blockers_except_public_activation_resolved"] is True
        and resolution["all_resolved"] is False,
        "gateway pre-marker blocker state is invalid",
    )
    return {
        "format": MARKER_FORMAT,
        "schema_version": 3,
        "campaign_id": contract["campaign_id"],
        "subcampaign_id": contract["comparison_graph"]["edges"][EDGE_ID][
            "subcampaign_id"
        ],
        "edge_id": EDGE_ID,
        "contract_repository_path": plan["contract_repository_path"],
        "contract_blob_size_bytes": tracked["contract"]["blob_size_bytes"],
        "contract_blob_sha256": tracked["contract"]["blob_sha256"],
        "validator_repository_path": plan["validator_repository_path"],
        "validator_blob_sha256": tracked["validator"]["blob_sha256"],
        "driver_repository_path": plan["driver_repository_path"],
        "driver_blob_sha256": tracked["driver"]["blob_sha256"],
        "shared_formal_driver_repository_path": SHARED_DRIVER_REPOSITORY_PATH,
        "shared_formal_driver_blob_sha256": tracked["shared_formal_driver"][
            "blob_sha256"
        ],
        "plan_repository_path": plan["plan_repository_path"],
        "plan_blob_size_bytes": tracked["plan"]["blob_size_bytes"],
        "plan_blob_sha256": tracked["plan"]["blob_sha256"],
        "marker_repository_path": plan["marker_repository_path"],
        "pre_marker_git_custody": copy.deepcopy(git_custody),
        "artifact_expectations": copy.deepcopy(plan["artifacts"]),
        "artifact_file_observations": copy.deepcopy(artifacts),
        "runtime_preflight": copy.deepcopy(runtime_preflight),
        "host_preflight": copy.deepcopy(host_preflight),
        "blocker_resolution": copy.deepcopy(resolution),
        "declared_schedule": declared_schedule(contract),
        "sampling_state_at_marker_creation": {
            "generation_requests": 0,
            "warmup_samples": 0,
            "timed_samples": 0,
        },
        "pre_marker_admission": {
            "immutable_contract_validator_driver_plan": True,
            "live_public_git_and_clean_worktree": True,
            "O_NOFOLLOW_artifact_custody": True,
            "same_resident_backend_pid_start_argv_environment_model_fd_and_closure": True,
            "controller_preload_fd_and_backend_loaded_fd_crosscheck": True,
            "canonical_request_and_raw_prompt": True,
            "request_history_disabled": True,
            "quiet_host": True,
            "marker_absent_through_preflight": True,
            "all_passed": True,
        },
        "next_required_action": (
            "commit this create-new marker and push it to live refs/heads/main; "
            "only then may run send the first formal warmup"
        ),
    }


def prepare_campaign(
    plan_path: Path | str,
    *,
    host_gate_collector: Any = collect_gateway_host_gate,
) -> dict[str, Any]:
    """Perform only zero-generation proofs, then exclusively create the marker."""

    context = load_execution_context(plan_path)
    plan = context["plan"]
    contract = context["contract"]
    repository_root = context["repository_root"]
    marker_path = repository_root / plan["marker_repository_path"]
    raw_path = Path(plan["raw_output_path"])
    custodian_binding: dict[str, Any] | None = None
    campaign_directory_initialization: dict[str, Any] | None = None
    custodian_start_attempted = False

    def preflight() -> dict[str, Any]:
        require(not os.path.lexists(marker_path), "gateway marker already exists")
        require(not os.path.lexists(raw_path), "gateway raw output already exists")
        require(
            marker_path.parent.resolve(strict=True).is_dir(), "marker parent is absent"
        )
        nonlocal campaign_directory_initialization
        campaign_directory_initialization = initialize_campaign_directory_tree(plan)
        require(
            raw_path.parent.resolve(strict=True).is_dir(), "raw-output parent is absent"
        )
        git_custody = collect_git_custody(
            repository_root,
            contract,
            tracked_campaign_paths(plan, include_marker=False),
            include_marker=False,
        )
        tracked = git_custody["tracked_files"]
        require(
            tracked["contract"]["blob_sha256"]
            == context["contract_file"]["sha256"]
            == _SHARED.FROZEN_CONTRACT_SHA256,
            "Git contract bytes differ from frozen contract",
        )
        require(
            tracked["validator"]["blob_sha256"]
            == context["validator_file"]["sha256"]
            == _SHARED.FROZEN_VALIDATOR_SHA256,
            "Git validator bytes differ from frozen validator",
        )
        require(
            tracked["plan"]["blob_sha256"] == context["plan_file"]["sha256"],
            "Git plan bytes differ from loaded plan",
        )
        artifacts = verify_plan_artifacts(plan)
        nonlocal custodian_binding, custodian_start_attempted
        custodian_start_attempted = True
        custodian_binding = start_custodian(
            plan_path, plan, campaign_directory_initialization
        )
        runtime_before_host = collect_runtime_preflight(
            plan_path, plan, artifacts, custodian_binding
        )
        host_preflight = (
            host_gate_collector(
                "preflight",
                contract,
                runtime_identity=runtime_before_host,
            )
            if host_gate_collector is collect_gateway_host_gate
            else host_gate_collector("preflight", contract)
        )
        _SHARED.validate_host_receipt(host_preflight, "preflight", contract)
        validate_actual_host_windows(host_preflight, contract)
        runtime_after_host = collect_runtime_preflight(
            plan_path, plan, artifacts, custodian_binding
        )
        require(
            runtime_after_host["custodian_process_start"]["driver_source"]["sha256"]
            == tracked["driver"]["blob_sha256"],
            "custodian-executed driver source differs from the tracked driver blob",
        )
        require(
            runtime_after_host["custodian_process_start"][
                "shared_formal_driver_source"
            ]["sha256"]
            == tracked["shared_formal_driver"]["blob_sha256"],
            "custodian-loaded helper source differs from the tracked helper blob",
        )
        require(
            runtime_before_host["backend_process_start"]
            == runtime_after_host["backend_process_start"]
            and runtime_before_host["gateway_process_start"]
            == runtime_after_host["gateway_process_start"]
            and runtime_before_host["custodian_binding"]
            == runtime_after_host["custodian_binding"],
            "resident runtime identity changed across host preflight",
        )
        require_same_custodian_attestation(
            runtime_before_host["controller_backend_model_fd_custody"]["start"],
            runtime_after_host["controller_backend_model_fd_custody"]["end"],
        )
        require(
            not os.path.lexists(marker_path), "gateway marker appeared during preflight"
        )
        resolution = gateway_blocker_resolution(
            contract,
            driver_evidence={
                "driver_blob_sha256": tracked["driver"]["blob_sha256"],
                "shared_formal_driver_blob_sha256": tracked["shared_formal_driver"][
                    "blob_sha256"
                ],
                "plan_blob_sha256": tracked["plan"]["blob_sha256"],
                "artifact_observations_sha256": sha256_canonical(artifacts),
            },
            quiet_host_evidence=sha256_canonical(host_preflight),
            runtime_evidence=sha256_canonical(runtime_after_host),
            public_activation_evidence=None,
        )
        return build_marker(
            context,
            git_custody,
            artifacts,
            runtime_after_host,
            host_preflight,
            resolution,
        )

    try:
        return prepare_marker_after_preflight(marker_path, preflight)
    except BaseException as error:
        cleanup_evidence = _cleanup_failed_prepare_resources(
            plan,
            custodian_binding,
            campaign_directory_initialization,
            marker_path,
            raw_path,
            "prepare-failed-before-marker-cleanup",
            runtime_cleanup_already_complete=(
                not custodian_start_attempted
                or getattr(error, "custodian_launch_cleanup_complete", False) is True
            ),
        )
        if isinstance(error, PreflightBlockedError):
            error.receipt["failed_prepare_cleanup"] = cleanup_evidence
        else:
            setattr(error, "failed_prepare_cleanup", cleanup_evidence)
        raise


def validate_shared_formal_driver_binding(
    marker: dict[str, Any],
    tracked: dict[str, Any],
    marker_runtime: dict[str, Any],
) -> None:
    shared = tracked.get("shared_formal_driver")
    require(isinstance(shared, dict), "tracked shared formal driver is absent")
    expected_hash = shared.get("blob_sha256")
    require(
        _valid_sha256(expected_hash), "tracked shared formal-driver hash is invalid"
    )
    require(
        marker.get("shared_formal_driver_repository_path")
        == SHARED_DRIVER_REPOSITORY_PATH
        and marker.get("shared_formal_driver_blob_sha256") == expected_hash,
        "published marker shared formal-driver binding drifted",
    )
    runtime_source = marker_runtime.get("custodian_process_start", {}).get(
        "shared_formal_driver_source", {}
    )
    require(
        runtime_source.get("absolute_path")
        == str(_NATIVE_DRIVER_PATH.resolve(strict=True))
        and runtime_source.get("sha256") == expected_hash,
        "marker-bound custodian helper differs from activation helper blob",
    )


def validate_gateway_marker_schema(marker: dict[str, Any]) -> None:
    require(
        isinstance(marker, dict) and set(marker) == GATEWAY_MARKER_FIELDS_V3,
        "published gateway marker top-level schema drifted",
    )
    require(
        not contains_forbidden_engine_ranking_claim(marker),
        "published gateway marker contains a forbidden engine winner or ranking claim",
    )
    sampling_state = marker.get("sampling_state_at_marker_creation")
    require(
        isinstance(sampling_state, dict)
        and set(sampling_state) == GATEWAY_MARKER_SAMPLING_STATE_FIELDS_V3,
        "published gateway marker sampling-state schema drifted",
    )
    admission = marker.get("pre_marker_admission")
    require(
        isinstance(admission, dict)
        and set(admission) == GATEWAY_MARKER_ADMISSION_FIELDS_V3,
        "published gateway marker admission schema drifted",
    )
    resolution = marker.get("blocker_resolution")
    require(
        isinstance(resolution, dict)
        and set(resolution) == GATEWAY_MARKER_BLOCKER_RESOLUTION_FIELDS_V3,
        "published gateway marker blocker-resolution schema drifted",
    )
    resolution_map = resolution.get("resolution_map")
    expected_blockers = {
        *GATEWAY_AUTHORED_BLOCKERS,
        "SAME_RESIDENT_BACKEND_AND_CONTROLLER_MODEL_FD_CUSTODY",
    }
    require(
        isinstance(resolution_map, dict) and set(resolution_map) == expected_blockers,
        "published gateway marker blocker-map schema drifted",
    )
    require(
        all(
            isinstance(entry, dict)
            and set(entry) == GATEWAY_MARKER_BLOCKER_ENTRY_FIELDS_V3
            for entry in resolution_map.values()
        ),
        "published gateway marker blocker-entry schema drifted",
    )


def validate_published_marker(
    marker: dict[str, Any],
    marker_file: dict[str, Any],
    context: dict[str, Any],
    git_custody: dict[str, Any],
    artifacts: dict[str, Any],
    runtime_now: dict[str, Any],
) -> None:
    plan = context["plan"]
    contract = context["contract"]
    tracked = git_custody["tracked_files"]
    validate_gateway_marker_schema(marker)
    require(
        marker.get("format") == MARKER_FORMAT, "published gateway marker format drifted"
    )
    require(
        marker.get("schema_version") == 3, "published gateway marker schema drifted"
    )
    require(
        marker.get("campaign_id") == contract["campaign_id"]
        and marker.get("subcampaign_id")
        == contract["comparison_graph"]["edges"][EDGE_ID]["subcampaign_id"]
        and marker.get("edge_id") == EDGE_ID,
        "published gateway marker binding drifted",
    )
    require(
        marker.get("contract_blob_size_bytes") == tracked["contract"]["blob_size_bytes"]
        and marker.get("contract_blob_sha256")
        == tracked["contract"]["blob_sha256"]
        == _SHARED.FROZEN_CONTRACT_SHA256,
        "published gateway marker contract drifted",
    )
    require(
        marker.get("validator_blob_sha256")
        == tracked["validator"]["blob_sha256"]
        == _SHARED.FROZEN_VALIDATOR_SHA256
        and marker.get("driver_blob_sha256") == tracked["driver"]["blob_sha256"]
        and marker.get("shared_formal_driver_repository_path")
        == SHARED_DRIVER_REPOSITORY_PATH
        and marker.get("shared_formal_driver_blob_sha256")
        == tracked["shared_formal_driver"]["blob_sha256"]
        and marker.get("plan_blob_sha256") == tracked["plan"]["blob_sha256"],
        "published gateway marker code/plan drifted",
    )
    require(
        marker.get("marker_repository_path") == plan["marker_repository_path"]
        and marker_file["size_bytes"] == tracked["activation_marker"]["blob_size_bytes"]
        and marker_file["sha256"] == tracked["activation_marker"]["blob_sha256"],
        "published gateway marker file differs from Git blob",
    )
    pre_git = marker.get("pre_marker_git_custody")
    require(
        isinstance(pre_git, dict)
        and pre_git.get("worktree_clean") is True
        and pre_git.get("head_commit") == pre_git.get("ls_remote_live_oid"),
        "gateway marker lacks live pre-marker Git custody",
    )
    for label, entry in pre_git["tracked_files"].items():
        require(label in tracked, f"pre-marker tracked label vanished: {label}")
        require(
            entry["repository_path"] == tracked[label]["repository_path"]
            and entry["blob_oid"] == tracked[label]["blob_oid"]
            and entry["blob_sha256"] == tracked[label]["blob_sha256"],
            f"pre-marker tracked bytes changed: {label}",
        )
    _prove_ancestor(
        context["repository_root"],
        pre_git["head_commit"],
        git_custody["activation_commit"],
    )
    require(
        marker.get("artifact_expectations") == plan["artifacts"]
        and marker.get("artifact_file_observations") == artifacts,
        "published gateway artifact custody changed",
    )
    marker_runtime = marker.get("runtime_preflight")
    require(
        isinstance(marker_runtime, dict), "published marker runtime preflight is absent"
    )
    validate_shared_formal_driver_binding(marker, tracked, marker_runtime)
    validate_published_runtime_preflight(marker_runtime, runtime_now, plan)
    require(
        marker_runtime["custodian_process_start"]["driver_source"]["sha256"]
        == tracked["driver"]["blob_sha256"],
        "marker-bound custodian source differs from activation driver blob",
    )
    require(
        marker_runtime["custodian_process_start"]["shared_formal_driver_source"][
            "sha256"
        ]
        == tracked["shared_formal_driver"]["blob_sha256"],
        "marker-bound custodian helper differs from activation helper blob",
    )
    _SHARED.validate_host_receipt(marker.get("host_preflight"), "preflight", contract)
    validate_actual_host_windows(marker["host_preflight"], contract)
    require(
        marker.get("declared_schedule") == declared_schedule(contract),
        "published marker schedule drifted",
    )
    require(
        marker.get("sampling_state_at_marker_creation")
        == {"generation_requests": 0, "warmup_samples": 0, "timed_samples": 0},
        "published marker claims pre-marker generation",
    )
    resolution = marker.get("blocker_resolution")
    require(
        isinstance(resolution, dict)
        and resolution.get("all_pre_marker_blockers_except_public_activation_resolved")
        is True
        and resolution.get("all_resolved") is False
        and resolution["resolution_map"][
            "V3_GATEWAY_SUBCAMPAIGN_NOT_PUBLICLY_ACTIVATED"
        ]["resolved"]
        is False,
        "published marker pre-activation blocker map drifted",
    )
    require(
        marker.get("pre_marker_admission", {}).get("all_passed") is True,
        "published marker admission is not complete",
    )


def validate_custodian_cleanup_receipt(
    receipt: dict[str, Any], expected_stage: str
) -> None:
    cleanup = receipt.get("custodian_cleanup")
    runtime = receipt.get("parity_admission", {})
    binding = runtime.get("custodian_binding", {})
    require(
        isinstance(cleanup, dict)
        and cleanup.get("format") == CUSTODIAN_CLEANUP_FORMAT
        and cleanup.get("schema_version") == 3
        and cleanup.get("edge_id") == EDGE_ID
        and cleanup.get("nonce") == binding.get("nonce")
        and re.fullmatch(r"[0-9a-f]{64}", cleanup.get("challenge", "")) is not None
        and cleanup.get("passed") is True,
        "formal gateway custodian cleanup receipt drifted",
    )
    pre_cleanup = cleanup.get("pre_cleanup_attestation")
    require(
        isinstance(pre_cleanup, dict)
        and pre_cleanup.get("format") == CUSTODIAN_ATTESTATION_FORMAT
        and pre_cleanup.get("schema_version") == 3
        and pre_cleanup.get("edge_id") == EDGE_ID
        and pre_cleanup.get("nonce") == binding.get("nonce")
        and pre_cleanup.get("challenge") == cleanup.get("challenge")
        and pre_cleanup.get("stage") == expected_stage
        and pre_cleanup.get("passed") is True,
        "cleanup pre-termination attestation drifted",
    )
    core = {
        key: value
        for key, value in pre_cleanup.items()
        if key not in ("challenge_response_sha256", "passed")
    }
    require(
        pre_cleanup.get("challenge_response_sha256") == sha256_canonical(core),
        "cleanup pre-termination challenge digest drifted",
    )
    baseline = (
        runtime.get("controller_backend_model_fd_custody", {})
        .get("start", {})
        .get("daemon_challenge_response", {})
    )
    validate_controller_launch_sequence(pre_cleanup.get("lifecycle_sequence", {}))
    for field in (
        "custodian_pid",
        "custodian_start_identity",
        "gateway_pid",
        "gateway_start_identity",
        "backend_pid",
        "backend_start_identity",
        "lifecycle_sequence",
        "backend_loaded_fd",
    ):
        require(
            pre_cleanup.get(field) == baseline.get(field),
            f"cleanup pre-termination {field} changed",
        )
    require(
        _stable_controller_fd_receipt(pre_cleanup.get("controller_preload_fd", {}))
        == _stable_controller_fd_receipt(baseline.get("controller_preload_fd", {}))
        == _stable_controller_fd_receipt(
            cleanup.get("controller_fd_still_held_while_cleanup_response_built", {})
        ),
        "cleanup did not retain the exact controller FD through response creation",
    )
    runtime_cleanup = cleanup.get("runtime_cleanup")
    require(
        isinstance(runtime_cleanup, dict)
        and runtime_cleanup.get("gateway_pid") == binding.get("gateway_pid")
        and runtime_cleanup.get("gateway_start_identity")
        == binding.get("gateway_start_identity")
        and runtime_cleanup.get("backend_pid") == binding.get("backend_pid")
        and runtime_cleanup.get("backend_start_identity")
        == binding.get("backend_start_identity")
        and runtime_cleanup.get("termination_requested") is True
        and is_int(runtime_cleanup.get("returncode"))
        and isinstance(runtime_cleanup.get("forced_kill_required"), bool)
        and runtime_cleanup.get("backend_termination")
        in {
            "exited-with-gateway",
            "original-exited-and-pid-was-reused",
            "explicit-sigterm",
            "explicit-sigterm-original-exited-pid-reused",
            "explicit-sigkill",
            "explicit-sigkill-original-exited-pid-reused",
        },
        "cleanup runtime termination binding drifted",
    )
    completion = cleanup.get("cleanup_completion")
    require(
        isinstance(completion, dict)
        and set(completion)
        == {
            "custodian_original_process_exited",
            "custodian_pid_reused_after_exit",
            "control_socket_removed",
            "controller_fd_closed_by_process_exit",
        }
        and completion["custodian_original_process_exited"] is True
        and isinstance(completion["custodian_pid_reused_after_exit"], bool)
        and completion["control_socket_removed"] is True
        and completion["controller_fd_closed_by_process_exit"] is True,
        "custodian cleanup completion proof drifted",
    )


def validate_machine_git_custody(
    receipt: dict[str, Any], contract: dict[str, Any]
) -> None:
    git = receipt.get("git_custody")
    binding = receipt.get("contract_binding", {})
    require(isinstance(git, dict), "formal gateway Git custody is absent")
    activation = contract["activation_contract"]
    oid_length = 40 if git.get("object_format") == "sha1" else 64
    oids = (
        git.get("head_commit"),
        git.get("local_tracking_oid"),
        git.get("ls_remote_live_oid"),
        git.get("activation_commit"),
    )
    require(
        git.get("object_format") in ("sha1", "sha256")
        and all(
            isinstance(oid, str)
            and len(oid) == oid_length
            and all(character in "0123456789abcdef" for character in oid)
            for oid in oids
        )
        and len(set(oids)) == 1
        and git.get("worktree_clean") is True
        and git.get("activation_commit_equals_head_and_live_remote_oid") is True
        and git.get("contract_commit_is_ancestor_of_activation_commit") is True
        and git.get("local_tracking_ref_used_as_publication_proof") is False
        and git.get("local_transport_overrides_absent") is True
        and git.get("sanitized_git_environment") == _SHARED.git_custody_environment()
        and git.get("remote_origin_url") == activation["frozen_origin_remote_url"]
        and git.get("live_remote_url") == activation["frozen_origin_remote_url"]
        and git.get("live_remote_ref") == activation["frozen_live_remote_ref"]
        and git.get("ls_remote_exit_code") == 0,
        "formal gateway live Git activation custody drifted",
    )
    tracked = git.get("tracked_files")
    expected_paths = {
        "contract": CONTRACT_REPOSITORY_PATH,
        "validator": VALIDATOR_REPOSITORY_PATH,
        "driver": DRIVER_REPOSITORY_PATH,
        "shared_formal_driver": SHARED_DRIVER_REPOSITORY_PATH,
        "plan": binding.get("plan_repository_path"),
        "activation_marker": MARKER_REPOSITORY_PATH,
    }
    require(
        isinstance(tracked, dict) and set(tracked) == set(expected_paths),
        "formal gateway tracked Git dependency set drifted",
    )
    for label, expected_path in expected_paths.items():
        entry = tracked[label]
        require(
            isinstance(expected_path, str)
            and entry.get("repository_path") == expected_path
            and _valid_sha256(entry.get("blob_sha256"))
            and entry.get("blob_size_bytes") == entry.get("observed_size_bytes")
            and entry.get("blob_sha256") == entry.get("observed_sha256"),
            f"formal gateway tracked Git file drifted: {label}",
        )
    require(
        tracked["contract"]["blob_sha256"]
        == binding.get("contract_blob_sha256")
        == binding.get("activation_contract_blob_sha256")
        == _SHARED.FROZEN_CONTRACT_SHA256
        and tracked["validator"]["blob_sha256"]
        == binding.get("validator_blob_sha256")
        == _SHARED.FROZEN_VALIDATOR_SHA256
        and tracked["driver"]["blob_sha256"]
        == binding.get("gateway_driver_blob_sha256")
        and tracked["shared_formal_driver"]["blob_sha256"]
        == binding.get("shared_formal_driver_blob_sha256")
        and tracked["plan"]["blob_sha256"] == binding.get("plan_blob_sha256")
        and tracked["activation_marker"]["blob_sha256"]
        == binding.get("activation_marker_blob_sha256")
        and git.get("head_tree") == binding.get("activation_tree")
        and git.get("head_commit") == binding.get("activation_commit"),
        "formal gateway Git blobs differ from dynamic contract binding",
    )


def validate_machine_artifact_custody(receipt: dict[str, Any]) -> None:
    artifacts = receipt.get("artifact_custody")
    require(isinstance(artifacts, dict), "formal gateway artifact custody is absent")
    observations = artifacts.get("file_observations_start")
    require(
        isinstance(observations, dict)
        and set(observations) == {"model", "omniinfer_cli", "gateway_backend"}
        and receipt.get("artifact_custody_end") == observations,
        "formal gateway artifact start/end observation set drifted",
    )
    expected = {
        "model": (MODEL_PATH, MODEL_SIZE, MODEL_SHA256),
        "omniinfer_cli": (
            artifacts.get("omniinfer", {}).get("cli", {}).get("absolute_path"),
            OMNI_CLI_SIZE,
            OMNI_CLI_SHA256,
        ),
        "gateway_backend": (
            artifacts.get("gateway_backend", {})
            .get("llama_server", {})
            .get("absolute_path"),
            BACKEND_BINARY_SIZE,
            BACKEND_BINARY_SHA256,
        ),
    }
    for label, (path, size, digest) in expected.items():
        item = observations[label]
        require(
            isinstance(path, str)
            and path.startswith("/")
            and item.get("absolute_path") == path
            and item.get("size_bytes") == size
            and item.get("sha256") == digest
            and is_int(item.get("mode"))
            and stat.S_ISREG(item["mode"])
            and is_int(item.get("device"))
            and is_int(item.get("inode"))
            and is_int(item.get("ctime_ns"))
            and is_int(item.get("hard_link_count"))
            and item["hard_link_count"] >= 1
            and item.get("open_flags") == ["O_RDONLY", "O_CLOEXEC", "O_NOFOLLOW"]
            and item.get("identity_before_after_equal") is True,
            f"formal gateway artifact O_NOFOLLOW custody drifted: {label}",
        )
    require(
        observations["model"]["hard_link_count"] == 1,
        "formal gateway model artifact is not single-link",
    )
    runtime = receipt.get("parity_admission", {})
    for label, process_key in (
        ("omniinfer_cli", "gateway_process_start"),
        ("gateway_backend", "backend_process_start"),
    ):
        item = observations[label]
        process = runtime.get(process_key)
        require(
            isinstance(process, dict)
            and process.get("canonical_executable_path") == item["absolute_path"],
            f"formal gateway process executable differs from artifact: {label}",
        )
        matches = [
            image
            for image in process.get("runtime_closure", [])
            if image.get("loaded_image_path") == item["absolute_path"]
        ]
        require(len(matches) == 1, f"runtime closure lacks exact artifact: {label}")
        closure_file = matches[0].get("file", {})
        for field in ("device", "inode", "mode", "size_bytes", "ctime_ns", "sha256"):
            require(
                closure_file.get(field) == item[field],
                f"runtime closure artifact identity drifted: {label}/{field}",
            )
    held = (
        runtime.get("controller_backend_model_fd_custody", {})
        .get("start", {})
        .get("daemon_challenge_response", {})
        .get("controller_preload_fd", {})
    )
    model = observations["model"]
    require(
        held.get("absolute_path") == model["absolute_path"]
        and held.get("device") == model["device"]
        and held.get("inode") == model["inode"]
        and held.get("mode") == model["mode"]
        and held.get("link_count") == model["hard_link_count"]
        and held.get("size_bytes") == model["size_bytes"]
        and held.get("ctime_ns") == model["ctime_ns"]
        and held.get("sha256") == model["sha256"],
        "controller preload FD differs from model artifact custody",
    )


def validate_machine_connections(receipt: dict[str, Any]) -> None:
    postflight = receipt.get("postflight", {})
    starts = postflight.get("connections_start")
    ending = postflight.get("connection_postflight")
    require(
        isinstance(starts, dict)
        and isinstance(ending, dict)
        and ending.get("passed") is True
        and postflight.get("connections_closed") is True
        and postflight.get("close_errors") == [],
        "formal gateway persistent-connection receipt is incomplete",
    )
    sockets = starts.get("sockets")
    require(
        isinstance(sockets, dict) and set(sockets) == {"admin", "B", "G"},
        "formal gateway initial socket set drifted",
    )
    for arm in ("B", "G"):
        item = ending.get(arm)
        require(
            isinstance(item, dict)
            and item.get("socket_start") == sockets[arm]
            and item.get("socket_end") == sockets[arm]
            and item.get("socket_start_end_equal") is True
            and item.get("request_count") == 69
            and item.get("generation_request_count") == 68
            and item.get("reconnect_count") == 0,
            f"formal gateway {arm} persistent connection drifted",
        )
    admin = ending.get("admin")
    require(
        isinstance(admin, dict)
        and admin.get("socket_start") == sockets["admin"]
        and admin.get("socket_end") == sockets["admin"]
        and admin.get("socket_start_end_equal") is True
        and admin.get("request_count") == 289
        and admin.get("reconnect_count") == 0,
        "formal gateway admin persistent connection drifted",
    )
    warm = starts.get("warm_non_generation_requests")
    require(
        isinstance(warm, dict)
        and warm.get("B", {}).get("socket") == sockets["B"]
        and warm.get("B", {}).get("request_index_on_connection") == 1
        and warm.get("B", {}).get("path") == "/health"
        and warm.get("G", {}).get("socket") == sockets["G"]
        and warm.get("G", {}).get("request_index_on_connection") == 1
        and warm.get("G", {}).get("path") == "/health?deep=true",
        "formal gateway persistent connection warmup drifted",
    )
    for sample in receipt.get("samples", []):
        require(
            sample.get("connection", {}).get("socket") == sockets[sample["slot"]["arm"]]
            and sample.get("cache_clear", {}).get("transport", {}).get("socket")
            == sockets["admin"]
            and sample.get("workload", {})
            .get("per_sample_tokenize_admission_outside_timed_interval", {})
            .get("transport", {})
            .get("socket")
            == sockets["admin"],
            "formal gateway sample used an unbound persistent socket",
        )
        segment = sample.get("segment_state_after")
        if segment is not None:
            require(
                segment.get("transport", {}).get("socket") == sockets["admin"],
                "formal gateway segment state used an unbound admin socket",
            )


def validate_recomputed_gateway_statistics(
    samples: list[dict[str, Any]],
    recorded: dict[str, Any],
    contract: dict[str, Any],
) -> bool:
    expected = compute_gateway_statistics(
        [sample for sample in samples if sample["slot"]["phase"] == "timed"],
        contract,
    )
    require(
        recorded == expected,
        "formal gateway statistics were not exactly derived from timed samples",
    )
    return expected["all_stability_gates_passed"] is True


def validate_gateway_machine_schema(
    receipt: dict[str, Any], *, require_cleanup: bool
) -> None:
    expected_fields = GATEWAY_MACHINE_RECEIPT_FIELDS_V3
    if not require_cleanup:
        expected_fields = expected_fields - {"custodian_cleanup"}
    require(
        isinstance(receipt, dict) and set(receipt) == expected_fields,
        "formal gateway machine-receipt top-level schema drifted",
    )
    require(
        not contains_forbidden_engine_ranking_claim(receipt),
        "formal gateway machine receipt contains a forbidden engine winner or ranking claim",
    )
    contract_binding = receipt.get("contract_binding")
    require(
        isinstance(contract_binding, dict)
        and set(contract_binding) == GATEWAY_CONTRACT_BINDING_FIELDS_V3,
        "formal gateway contract-binding schema drifted",
    )
    decision = receipt.get("decision")
    require(
        isinstance(decision, dict) and set(decision) == GATEWAY_DECISION_FIELDS_V3,
        "formal gateway decision schema drifted",
    )


def validate_gateway_machine_receipt(
    receipt: dict[str, Any],
    contract: dict[str, Any],
    *,
    require_cleanup: bool = True,
) -> None:
    validate_gateway_machine_schema(receipt, require_cleanup=require_cleanup)
    require(
        receipt.get("format") == RAW_FORMAT
        and receipt.get("schema_version") == 3
        and receipt.get("campaign_id") == contract["campaign_id"]
        and receipt.get("subcampaign_id")
        == contract["comparison_graph"]["edges"][EDGE_ID]["subcampaign_id"]
        and receipt.get("edge_id") == EDGE_ID
        and receipt.get("campaign_consumed") is True
        and receipt.get("failures") == [],
        "formal gateway raw envelope is not an admitted v3 completion",
    )
    raw_output_path = receipt.get("raw_output_path")
    require(
        isinstance(raw_output_path, str)
        and raw_output_path.startswith("/")
        and os.path.normpath(raw_output_path) == raw_output_path,
        "formal gateway raw output path is invalid",
    )
    for field in contract["machine_receipt_contract"]["required_top_level_objects"]:
        require(field in receipt, f"formal gateway receipt lacks {field}")
    validate_machine_git_custody(receipt, contract)
    validate_machine_artifact_custody(receipt)
    binding = receipt["contract_binding"]
    require(
        binding.get("campaign_id") == contract["campaign_id"]
        and binding.get("schema_version") == 3
        and binding.get("edge_id") == EDGE_ID
        and binding.get("subcampaign_id")
        == contract["comparison_graph"]["edges"][EDGE_ID]["subcampaign_id"],
        "formal gateway dynamic contract binding drifted",
    )
    host = receipt["host_custody"]
    for field in ("model_identifier", "chip", "os_build"):
        require(
            host.get(field) == contract["scope"]["host"][field], f"host {field} drifted"
        )
    for phase in ("preflight", "continuous", "postflight"):
        _SHARED.validate_host_receipt(host.get(phase), phase, contract)
        validate_actual_host_windows(host[phase], contract)
    artifacts = receipt["artifact_custody"]
    require(
        artifacts.get("model", {}).get("sha256") == MODEL_SHA256
        and artifacts.get("llama_cpp", {}).get("source_commit")
        == PINNED_CORE_SOURCE_COMMIT
        and artifacts.get("omniinfer", {}).get("source_commit") == OMNI_SOURCE_COMMIT
        and artifacts.get("gateway_backend", {}).get("source_commit")
        == BACKEND_SOURCE_COMMIT,
        "formal gateway machine artifact bindings drifted",
    )
    require(
        receipt.get("artifact_custody_end") == artifacts.get("file_observations_start"),
        "formal gateway artifact O_NOFOLLOW observations changed start-to-end",
    )
    require(
        receipt.get("postflight", {}).get("runtime_postflight", {}).get("passed")
        is True
        and receipt.get("postflight", {}).get("connection_postflight", {}).get("passed")
        is True,
        "formal gateway runtime or persistent-connection postflight failed",
    )
    schedule = receipt["schedule_receipt"]
    require(
        len(schedule["slots"]) == 136
        and schedule["attempted_count"] == 136
        and schedule["accepted_count"] == 136
        and schedule["failed_count"] == 0
        and schedule["remaining_unattempted_count"] == 0
        and schedule.get("stopped_at_first_failure") is False
        and schedule.get(
            "retry_replacement_reordering_outlier_removal_or_extension_performed"
        )
        is False
        and schedule.get("process_state")
        == "one-resident-backend-and-gateway-for-entire-campaign"
        and schedule.get("client_connections")
        == "one warmed persistent HTTP/1.1 connection per arm"
        and schedule.get("warmup_abstract_orders") == ["ABBA", "BAAB"]
        and schedule.get("timed_macroblocks") == 16
        and all(
            slot["attempt_count"] == 1
            and slot.get("failure_index") is None
            and slot.get("status") == "accepted"
            for slot in schedule["slots"]
        ),
        "formal gateway completed schedule drifted",
    )
    declared = declared_schedule(contract)
    samples = receipt.get("samples")
    require(
        isinstance(samples, list) and len(samples) == len(declared),
        "formal gateway sample list is incomplete",
    )
    validate_model_fd_checkpoint_schedule(receipt)
    connection_postflight = receipt["postflight"]["connection_postflight"]
    for expected_slot, slot_state, sample in zip(declared, schedule["slots"], samples):
        require(
            slot_state["slot"] == expected_slot
            and slot_state["status"] == "accepted"
            and slot_state["receipt_sha256"] == sha256_canonical(sample),
            "formal gateway slot/sample hash binding drifted",
        )
        validate_sample_receipt(sample, expected_slot, contract)
        arm = expected_slot["arm"]
        require(
            sample["connection"]["socket"]
            == connection_postflight[arm]["socket_start"],
            f"formal gateway {arm} sample used a different persistent socket",
        )
    for pair_start in range(0, len(samples), 2):
        validate_pair_equal(samples[pair_start], samples[pair_start + 1])
    validate_machine_connections(receipt)
    statistics = receipt["statistics"]
    stable = validate_recomputed_gateway_statistics(samples, statistics, contract)
    expected_gates = _gateway_gates(True, stable)
    require(receipt["gates"] == expected_gates, "formal gateway gate map drifted")
    require(
        set(receipt["gates"])
        == set(
            contract["machine_receipt_contract"][
                "required_true_gate_ids_for_GATEWAY_B_VS_G"
            ]
        ),
        "formal gateway gate IDs drifted",
    )
    require(
        receipt["decision"]["label"] == statistics["decision"]
        and receipt["decision"]["engine_winner_or_ranking_claim_allowed"] is False,
        "formal gateway decision drifted",
    )
    require(
        receipt.get("status")
        == ("FORMAL_COMPLETE" if stable else "FORMAL_UNINTERPRETABLE"),
        "formal gateway completion status differs from recomputed stability",
    )
    if stable:
        require(
            receipt["decision"]["formal_summary_allowed"] is True,
            "stable result was suppressed",
        )
    else:
        require(
            receipt["decision"]["label"] == "UNINTERPRETABLE"
            and receipt["decision"]["formal_summary_allowed"] is False,
            "unstable gateway result was interpreted",
        )
    if require_cleanup:
        validate_custodian_cleanup_receipt(
            receipt, "run-finished-after-raw-evidence-durable"
        )


def _pre_schedule_failure_record(
    contract: dict[str, Any],
    plan: dict[str, Any],
    error: BaseException,
    evidence: dict[str, Any],
) -> dict[str, Any]:
    schedule = declared_schedule(contract)
    slots = [
        {
            "slot": slot,
            "status": "unattempted",
            "attempt_count": 0,
            "receipt_sha256": None,
            "failure_index": None,
        }
        for slot in schedule
    ]
    host_static = {
        field: contract["scope"]["host"][field]
        for field in (
            "model_identifier",
            "chip",
            "architecture",
            "logical_cpu_count",
            "memory_bytes",
            "os_product",
            "os_version",
            "os_build",
        )
    }
    return {
        "format": RAW_FORMAT,
        "schema_version": 3,
        "campaign_id": contract["campaign_id"],
        "subcampaign_id": contract["comparison_graph"]["edges"][EDGE_ID][
            "subcampaign_id"
        ],
        "edge_id": EDGE_ID,
        "campaign_consumed": True,
        "status": "CONSUMED_FIRST_POST_MARKER_FAILURE",
        "raw_output_path": plan["raw_output_path"],
        "contract_binding": copy.deepcopy(evidence.get("contract_binding", {})),
        "git_custody": copy.deepcopy(evidence.get("git_custody", {})),
        "host_custody": {
            **host_static,
            **copy.deepcopy(evidence.get("host_custody", {})),
        },
        "artifact_custody": copy.deepcopy(evidence.get("artifact_custody", {})),
        "parity_admission": copy.deepcopy(evidence.get("parity_admission", {})),
        "schedule_receipt": {
            "process_state": "one-resident-backend-and-gateway-for-entire-campaign",
            "slots": slots,
            "attempted_count": 0,
            "accepted_count": 0,
            "failed_count": 0,
            "remaining_unattempted_count": 136,
            "stopped_at_first_failure": True,
            "retry_replacement_reordering_outlier_removal_or_extension_performed": False,
        },
        "samples": [],
        "statistics": None,
        "gates": _gateway_gates(False, False),
        "decision": {
            "label": "UNINTERPRETABLE",
            "formal_summary_allowed": False,
            "engine_winner_or_ranking_claim_allowed": False,
        },
        "failures": [
            {
                "stage": "campaign-start-reproof",
                "sequence_index": None,
                "exception_type": type(error).__name__,
                "message": str(error),
                "observation": _safe_observation(
                    error.receipt if isinstance(error, PreflightBlockedError) else {}
                ),
                "remaining_slots_marked_unattempted": True,
                "failed_observation_retained": True,
            }
        ],
    }


def _attach_custodian_cleanup_after_raw(
    record: dict[str, Any],
    raw_path: Path,
    plan: dict[str, Any],
    binding: dict[str, Any] | None,
    stage: str,
) -> dict[str, Any]:
    """Only stop managed processes after their failure/success evidence is durable."""

    durable_raw, _ = _read_regular_no_follow(raw_path, 512 * 1024 * 1024)
    require(
        parse_strict_json_line(durable_raw) == record,
        "refusing custodian cleanup before exact raw evidence is durable",
    )
    if binding is None:
        record["custodian_cleanup"] = {
            "attempted": False,
            "reason": "no marker-bound custodian identity was admitted",
        }
        _SHARED.atomic_replace_json(raw_path, record)
        return record
    try:
        record["custodian_cleanup"] = shutdown_custodian(plan, binding, stage)
    except BaseException as error:
        try:
            exact_pid_fallback = _fallback_cleanup_marker_bound_processes(
                plan, binding, stage
            )
        except BaseException as fallback_error:
            exact_pid_fallback = {
                "format": "apxinf-omniinfer-gateway-exact-pid-fallback-cleanup-v3",
                "passed": False,
                "exception_type": type(fallback_error).__name__,
                "message": str(fallback_error),
                "no_unbound_signal_claimed": True,
            }
        record["failures"].append(
            {
                "stage": "custodian-cleanup",
                "sequence_index": None,
                "exception_type": type(error).__name__,
                "message": str(error),
                "observation": {},
                "remaining_slots_marked_unattempted": True,
                "failed_observation_retained": True,
            }
        )
        record["status"] = "CONSUMED_FIRST_POST_MARKER_FAILURE"
        record["gates"] = _gateway_gates(False, False)
        record["decision"] = {
            "label": "UNINTERPRETABLE",
            "formal_summary_allowed": False,
            "engine_winner_or_ranking_claim_allowed": False,
        }
        record["custodian_cleanup"] = {
            "attempted": True,
            "passed": False,
            "exception_type": type(error).__name__,
            "message": str(error),
            "exact_pid_fallback": exact_pid_fallback,
        }
    _SHARED.atomic_replace_json(raw_path, record)
    return record


def run_campaign(
    plan_path: Path | str,
    *,
    host_gate_collector: Any = collect_gateway_host_gate,
    monitor_factory: Any = GatewayContinuousHostMonitor,
) -> dict[str, Any]:
    """Re-prove public activation, then consume the exact 136-slot campaign."""

    context = load_execution_context(plan_path)
    plan = context["plan"]
    contract = context["contract"]
    repository_root = context["repository_root"]
    marker_path = repository_root / plan["marker_repository_path"]
    raw_path = Path(plan["raw_output_path"])
    require(os.path.lexists(marker_path), "gateway marker is absent; run prepare first")
    require(not os.path.lexists(raw_path), "gateway raw output already exists")
    evidence: dict[str, Any] = {}
    custodian_binding: dict[str, Any] | None = None
    try:
        git_custody = collect_git_custody(
            repository_root,
            contract,
            tracked_campaign_paths(plan, include_marker=True),
            include_marker=True,
        )
        evidence["git_custody"] = git_custody
        evidence["contract_binding"] = final_contract_binding(context, git_custody)
        marker, marker_file = _read_tracked_receipt(
            repository_root,
            plan["marker_repository_path"],
            git_custody["tracked_files"]["activation_marker"],
        )
        marker_runtime = marker.get("runtime_preflight")
        require(isinstance(marker_runtime, dict), "marker runtime custody is absent")
        candidate_binding = marker_runtime.get("custodian_binding")
        require(
            isinstance(candidate_binding, dict),
            "marker custodian binding is absent",
        )
        custodian_binding = copy.deepcopy(candidate_binding)
        artifacts = verify_plan_artifacts(plan)
        evidence["artifact_custody"] = {
            "model": {
                "size_bytes": MODEL_SIZE,
                "sha256": MODEL_SHA256,
                "O_NOFOLLOW_observation": copy.deepcopy(artifacts["model"]),
            },
            "llama_cpp": {"source_commit": PINNED_CORE_SOURCE_COMMIT},
            "omniinfer": {
                "source_commit": OMNI_SOURCE_COMMIT,
                "cli": copy.deepcopy(artifacts["omniinfer_cli"]),
            },
            "gateway_backend": {
                "source_commit": BACKEND_SOURCE_COMMIT,
                "llama_server": copy.deepcopy(artifacts["gateway_backend"]),
            },
        }
        runtime_preflight = collect_runtime_preflight(
            plan_path, plan, artifacts, custodian_binding
        )
        evidence["parity_admission"] = copy.deepcopy(runtime_preflight)
        validate_published_marker(
            marker,
            marker_file,
            context,
            git_custody,
            artifacts,
            runtime_preflight,
        )
        host_preflight = (
            host_gate_collector(
                "preflight", contract, runtime_identity=runtime_preflight
            )
            if host_gate_collector is collect_gateway_host_gate
            else host_gate_collector("preflight", contract)
        )
        _SHARED.validate_host_receipt(host_preflight, "preflight", contract)
        validate_actual_host_windows(host_preflight, contract)
        binding = final_contract_binding(context, git_custody)
        resolution = gateway_blocker_resolution(
            contract,
            driver_evidence={
                "driver_blob_sha256": git_custody["tracked_files"]["driver"][
                    "blob_sha256"
                ],
                "shared_formal_driver_blob_sha256": git_custody["tracked_files"][
                    "shared_formal_driver"
                ]["blob_sha256"],
                "plan_blob_sha256": git_custody["tracked_files"]["plan"]["blob_sha256"],
            },
            quiet_host_evidence=sha256_canonical(host_preflight),
            runtime_evidence=sha256_canonical(runtime_preflight),
            public_activation_evidence={
                "activation_commit": git_custody["activation_commit"],
                "live_remote_oid": git_custody["ls_remote_live_oid"],
                "marker_blob_sha256": marker_file["sha256"],
            },
        )
        require(
            resolution["all_resolved"] is True,
            "gateway blocker resolution is incomplete",
        )
        artifact_custody = machine_artifact_custody(plan, artifacts, runtime_preflight)
        evidence.update(
            {
                "contract_binding": binding,
                "host_custody": {
                    **{
                        field: contract["scope"]["host"][field]
                        for field in (
                            "model_identifier",
                            "chip",
                            "architecture",
                            "logical_cpu_count",
                            "memory_bytes",
                            "os_product",
                            "os_version",
                            "os_build",
                        )
                    },
                    "preflight": host_preflight,
                },
                "artifact_custody": artifact_custody,
                "parity_admission": runtime_preflight,
                "blocker_resolution": resolution,
            }
        )
    except BaseException as error:
        partial = _pre_schedule_failure_record(contract, plan, error, evidence)
        _SHARED.atomic_create_json(raw_path, partial)
        return _attach_custodian_cleanup_after_raw(
            partial,
            raw_path,
            plan,
            custodian_binding,
            "run-pre-schedule-reproof-failure",
        )

    try:
        admin, backend, gateway, connections_start = open_campaign_connections(
            plan, runtime_preflight
        )
    except BaseException as error:
        partial = _pre_schedule_failure_record(contract, plan, error, evidence)
        _SHARED.atomic_create_json(raw_path, partial)
        return _attach_custodian_cleanup_after_raw(
            partial,
            raw_path,
            plan,
            custodian_binding,
            "run-connection-open-failure",
        )

    try:
        monitor = monitor_factory(contract, runtime_preflight)
    except BaseException as error:
        partial = _pre_schedule_failure_record(contract, plan, error, evidence)
        _SHARED.atomic_create_json(raw_path, partial)
        close_errors: list[dict[str, str]] = []
        for connection in (gateway, backend, admin):
            try:
                connection.close()
            except BaseException as close_error:
                close_errors.append(
                    {
                        "connection": connection.label,
                        "exception_type": type(close_error).__name__,
                        "message": str(close_error),
                    }
                )
        partial["connection_cleanup_after_monitor_factory_failure"] = {
            "all_close_calls_attempted": True,
            "close_errors": close_errors,
            "passed": not close_errors,
        }
        _SHARED.atomic_replace_json(raw_path, partial)
        return _attach_custodian_cleanup_after_raw(
            partial,
            raw_path,
            plan,
            custodian_binding,
            "run-monitor-factory-failure",
        )
    pair_buffer: list[dict[str, Any]] = []
    monitor_started = False
    monitor_stopped = False
    before_first_generation_custody: dict[str, Any] | None = None

    def before_first_slot() -> None:
        nonlocal monitor_started, before_first_generation_custody
        monitor_started = True
        monitor.start()
        monitor.wait_until_ready()
        before_first_generation_custody = attest_custodian(
            plan, runtime_preflight["custodian_binding"], "before-first-generation"
        )
        require_same_custodian_attestation(
            runtime_preflight["controller_backend_model_fd_custody"]["start"],
            before_first_generation_custody,
        )

    def sample_collector(slot: dict[str, Any]) -> dict[str, Any]:
        return collect_sample_with_failed_raw_retention(
            slot,
            backend,
            gateway,
            lambda: _sample_from_connections(
                slot,
                plan,
                contract,
                admin,
                backend,
                gateway,
                monitor,
                pair_buffer,
            ),
        )

    def postflight_collector() -> dict[str, Any]:
        nonlocal monitor_stopped
        primary_error: BaseException | None = None
        host_postflight: dict[str, Any] | None = None
        runtime_postflight: dict[str, Any] | None = None
        connection_postflight: dict[str, Any] | None = None
        continuous: dict[str, Any] | None = None
        close_errors: list[dict[str, str]] = []
        try:
            monitor.assert_healthy()
            host_postflight = (
                host_gate_collector(
                    "postflight", contract, runtime_identity=runtime_preflight
                )
                if host_gate_collector is collect_gateway_host_gate
                else host_gate_collector("postflight", contract)
            )
            _SHARED.validate_host_receipt(host_postflight, "postflight", contract)
            validate_actual_host_windows(host_postflight, contract)
            monitor.assert_healthy()
            runtime_postflight = collect_runtime_postflight(
                plan_path, plan, artifacts, runtime_preflight
            )
            connection_postflight = validate_connection_postflight(
                backend, gateway, admin
            )
        except BaseException as error:
            primary_error = error
        finally:
            if monitor_started and not monitor_stopped:
                try:
                    continuous = monitor.stop_and_receipt()
                    monitor_stopped = True
                except BaseException as error:
                    if primary_error is None:
                        primary_error = error
            for connection in (gateway, backend, admin):
                try:
                    connection.close()
                except BaseException as error:
                    close_errors.append(
                        {
                            "connection": connection.label,
                            "exception_type": type(error).__name__,
                            "message": str(error),
                        }
                    )
                    if primary_error is None:
                        primary_error = error
        if primary_error is not None:
            raise CampaignError(
                f"gateway postflight failed: {primary_error}"
            ) from primary_error
        require(not pair_buffer, "campaign ended with an incomplete adjacent B/G pair")
        return {
            "host_postflight": host_postflight,
            "continuous_host": continuous,
            "runtime_postflight": runtime_postflight,
            "connection_postflight": connection_postflight,
            "connections_start": connections_start,
            "before_first_generation_custody": before_first_generation_custody,
            "connections_closed": True,
            "close_errors": close_errors,
            "artifact_custody_end": copy.deepcopy(
                runtime_postflight["artifact_file_observations_end"]
            ),
            "passed": True,
        }

    record = execute_formal_schedule(
        contract,
        plan,
        evidence,
        sample_collector=sample_collector,
        postflight_collector=postflight_collector,
        before_first_slot=before_first_slot,
    )
    if record["status"] in ("FORMAL_COMPLETE", "FORMAL_UNINTERPRETABLE"):
        try:
            validate_gateway_machine_receipt(record, contract, require_cleanup=False)
        except BaseException as error:
            record["failures"].append(
                {
                    "stage": "machine-receipt-validation",
                    "sequence_index": None,
                    "exception_type": type(error).__name__,
                    "message": str(error),
                    "observation": {},
                    "remaining_slots_marked_unattempted": True,
                    "failed_observation_retained": True,
                }
            )
            record["status"] = "CONSUMED_FIRST_POST_MARKER_FAILURE"
            record["gates"] = _gateway_gates(False, False)
            record["decision"] = {
                "label": "UNINTERPRETABLE",
                "formal_summary_allowed": False,
                "engine_winner_or_ranking_claim_allowed": False,
            }
            _SHARED.atomic_replace_json(raw_path, record)
    record = _attach_custodian_cleanup_after_raw(
        record,
        raw_path,
        plan,
        custodian_binding,
        "run-finished-after-raw-evidence-durable",
    )
    if record["status"] in ("FORMAL_COMPLETE", "FORMAL_UNINTERPRETABLE"):
        try:
            validate_gateway_machine_receipt(record, contract)
        except BaseException as error:
            record["failures"].append(
                {
                    "stage": "final-machine-receipt-and-cleanup-validation",
                    "sequence_index": None,
                    "exception_type": type(error).__name__,
                    "message": str(error),
                    "observation": {},
                    "remaining_slots_marked_unattempted": True,
                    "failed_observation_retained": True,
                }
            )
            record["status"] = "CONSUMED_FIRST_POST_MARKER_FAILURE"
            record["gates"] = _gateway_gates(False, False)
            record["decision"] = {
                "label": "UNINTERPRETABLE",
                "formal_summary_allowed": False,
                "engine_winner_or_ranking_claim_allowed": False,
            }
            _SHARED.atomic_replace_json(raw_path, record)
    return record


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fail-closed formal-v3 same-backend OmniInfer gateway driver"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("prepare", "run"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--plan", required=True, type=Path)
    custodian = subparsers.add_parser("_custodian", help=argparse.SUPPRESS)
    custodian.add_argument("--plan", required=True, type=Path)
    custodian.add_argument("--nonce", required=True)
    subparsers.add_parser("self-test")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _argument_parser().parse_args(argv)
    try:
        if args.command == "_custodian":
            return run_custodian(args.plan, args.nonce)
        if args.command == "self-test":
            result = run_fixture_self_test()
        elif args.command == "prepare":
            marker = prepare_campaign(args.plan)
            result = {
                "status": "MARKER_CREATED_REQUIRES_COMMIT_AND_PUSH",
                "campaign_id": marker["campaign_id"],
                "edge_id": EDGE_ID,
                "marker_repository_path": marker["marker_repository_path"],
                "generation_requests": 0,
            }
        else:
            campaign = run_campaign(args.plan)
            result = {
                "status": campaign["status"],
                "campaign_id": campaign["campaign_id"],
                "edge_id": EDGE_ID,
                "raw_output_path": campaign["raw_output_path"],
                "accepted_count": campaign["schedule_receipt"]["accepted_count"],
                "decision": campaign["decision"]["label"],
            }
        sys.stdout.buffer.write(json_line_bytes(result))
        return (
            2
            if result.get("status")
            in ("CONSUMED_FIRST_POST_MARKER_FAILURE", "FORMAL_UNINTERPRETABLE")
            else 0
        )
    except PreflightBlockedError as error:
        sys.stdout.buffer.write(json_line_bytes(error.receipt))
        return 3
    except (CampaignError, OSError, subprocess.SubprocessError) as error:
        failure = {
            "format": "apxinf-qwen35-omniinfer-gateway-formal-v3-driver-error",
            "schema_version": 3,
            "edge_id": EDGE_ID,
            "command": getattr(args, "command", None),
            "error_type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }
        sys.stderr.buffer.write(json_line_bytes(failure))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
