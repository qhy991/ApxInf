#!/usr/bin/env python3
"""Zero-network tests for the formal-v3 OmniInfer gateway driver."""

from __future__ import annotations

import base64
import copy
import hashlib
import http.client
import importlib.util
import json
import os
from pathlib import Path
import platform
import stat
import tempfile
import unittest
from unittest import mock
import urllib.parse


ROOT = Path(__file__).resolve().parents[2]
DRIVER_PATH = ROOT / "benchmarks/cross_runtime/omniinfer_gateway_formal_v3_driver.py"
SPEC = importlib.util.spec_from_file_location("gateway_formal_v3", DRIVER_PATH)
assert SPEC is not None and SPEC.loader is not None
DRIVER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DRIVER)


def compact(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, allow_nan=False, separators=(",", ":")
    ).encode("utf-8")


def fixture_generation_wire_timing(arm: str) -> dict[str, object]:
    port = 19001 if arm == "B" else 19000
    header = (
        "POST /v1/chat/completions HTTP/1.1\r\n"
        f"Host: 127.0.0.1:{port}\r\n"
        "Accept: application/json\r\n"
        "Connection: keep-alive\r\n"
        "Content-Type: application/json\r\n"
        f"Content-Length: {DRIVER.REQUEST_SIZE}\r\n\r\n"
    ).encode("ascii")
    wire = header + DRIVER.REQUEST_BYTES
    wire_sha256 = hashlib.sha256(wire).hexdigest()
    return {
        "request_wire_size_bytes": len(wire),
        "request_wire_sha256": wire_sha256,
        "request_wire_base64": base64.b64encode(wire).decode("ascii"),
        "request_wire_body_offset_bytes": len(header),
        "request_wire_body_size_bytes": DRIVER.REQUEST_SIZE,
        "request_wire_body_sha256": DRIVER.REQUEST_SHA256,
        "request_wire_body_equals_request_body": True,
        "single_sendall_call_count": 1,
        "single_sendall_argument_size_bytes": len(wire),
        "single_sendall_argument_sha256": wire_sha256,
        "timing_event_order": list(DRIVER.COMPLETE_WIRE_TIMING_EVENT_ORDER_V3),
        "complete_HTTP_request_wire_serialization_before_start": True,
        "single_sendall_call_for_complete_request_wire_required": True,
        "canonical_383_byte_JSON_body_identical_between_B_and_G": True,
        "arm_specific_HTTP_authority_header_difference_is_inside_timed_region": True,
        "body_only_timing_allowed": False,
    }


def fixture_contract(tokens: list[int] | None = None) -> dict[str, object]:
    generated = tokens if tokens is not None else list(range(128))
    return {
        "campaign_id": "fixture-campaign",
        "comparison_graph": {
            "edges": {
                "GATEWAY_B_VS_G": {"subcampaign_id": "fixture-gateway-subcampaign"}
            }
        },
        "workload_contracts": {
            "shared_prompt": {
                "token_ids": DRIVER.PROMPT_TOKEN_IDS,
            },
            "GATEWAY_RAW13_FREE128_V3": {
                "request": {
                    "canonical_json_object": DRIVER.REQUEST,
                    "size_bytes": DRIVER.REQUEST_SIZE,
                    "sha256": DRIVER.REQUEST_SHA256,
                },
                "generation": {
                    "generated_token_count": 128,
                    "finish_reason": "length",
                    "usage_prompt_completion_total": [13, 128, 141],
                    "native_prompt_predicted_cache_n": [13, 128, 0],
                    "cache_clear_acknowledgement_before_every_arm": [0],
                },
                "trajectory_admission": {
                    "generated_token_ids_count": 128,
                    "expected_sha256": hashlib.sha256(compact(generated)).hexdigest(),
                },
            },
        },
        "execution_protocol": {
            "GATEWAY_B_VS_G": {
                "process_state": "one-resident-backend-and-gateway-for-entire-campaign",
                "client_connections": "one warmed persistent HTTP/1.1 connection per arm",
                "upstream_connection_reuse_claim_allowed": False,
                "untimed_warmup_abstract_orders": ["ABBA", "BAAB"],
                "untimed_warmups_per_arm": 4,
                "timed_macroblock_count": 16,
                "odd_macroblock_abstract_orders": ["ABBA", "BAAB"],
                "even_macroblock_abstract_orders": ["BAAB", "ABBA"],
                "role_binding": {
                    "A": "direct_resident_llama_server",
                    "B": "omniinfer_gateway_to_same_resident_llama_server",
                },
                "timed_subblocks_total": 32,
                "timed_samples_per_subblock": 4,
                "timed_samples_per_arm": 64,
                "timed_samples_total": 128,
                "cache_clear_and_admission_occur_outside_each_timed_interval": True,
            }
        },
        "timing_contract": {
            "clock": "monotonic",
            "GATEWAY_B_VS_G": {
                "start": DRIVER.GATEWAY_TIMING_START_V3,
                "end": "after-reading-the-full-response-body-and-validating-and-parsing-the-complete-JSON-response",
                "primary_metric": "client_full_response_wall_ms",
                **DRIVER.GATEWAY_COMPLETE_WIRE_TIMING_CONTRACT_V3,
            },
        },
        "statistics_and_decisions": {
            "GATEWAY_B_VS_G": {
                "stability_gates": {
                    "B_wall_population_cv_max": 0.01,
                    "G_wall_population_cv_max": 0.01,
                    "B_native_predicted_ms_population_cv_max": 0.01,
                    "G_native_predicted_ms_population_cv_max": 0.01,
                    "population_sd_pair_delta_over_pooled_wall_max": 0.01,
                    "absolute_order_stratum_mean_difference_max": "max(2 ms, 0.2% of pooled wall ms)",
                    "absolute_first8_last8_block_mean_difference_max": "max(2 ms, 0.2% of pooled wall ms)",
                    "ci95_delta_half_width_max_ms": 2.0,
                }
            }
        },
        "machine_receipt_contract": {
            "required_true_gate_ids_for_GATEWAY_B_VS_G": list(DRIVER.GATEWAY_GATE_IDS)
        },
    }


def fixture_response(
    tokens: list[int] | None = None, predicted_ms: float = 990.0
) -> dict[str, object]:
    generated = tokens if tokens is not None else list(range(128))
    return {
        "object": "chat.completion",
        "model": DRIVER.MODEL_PATH,
        "system_fingerprint": DRIVER.BACKEND_BUILD_INFO,
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
            "predicted_per_second": 128000.0 / predicted_ms,
            "prompt_per_token_ms": 10.0 / 13.0,
            "predicted_per_token_ms": predicted_ms / 128.0,
        },
        "__verbose": {
            "id_slot": 0,
            "tokens_predicted": 128,
            "tokens_evaluated": 13,
            "tokens_cached": 140,
            "stop_type": "limit",
            "truncated": False,
            "prompt": DRIVER.RENDERED_PROMPT,
            "tokens": generated,
            "generation_settings": {"seed": 0, "temperature": 0},
        },
    }


def fixture_sample(slot: dict[str, object], wall_ms: float) -> dict[str, object]:
    native_ms = wall_ms - 1.0
    response = fixture_response(predicted_ms=native_ms)
    response_raw = compact(response)
    request_index = (
        sum(
            prior["arm"] == slot["arm"]
            for prior in DRIVER.declared_schedule(fixture_contract())
            if prior["sequence_index"] < slot["sequence_index"]
        )
        + 2
    )
    return {
        "format": DRIVER.SAMPLE_FORMAT,
        "schema_version": 3,
        "campaign_id": "fixture-campaign",
        "subcampaign_id": "fixture-gateway-subcampaign",
        "edge_id": DRIVER.EDGE_ID,
        "slot": copy.deepcopy(slot),
        "request": {
            "canonical_json_object": DRIVER.REQUEST,
            "canonical_utf8": DRIVER.REQUEST_BYTES.decode("utf-8"),
            "size_bytes": DRIVER.REQUEST_SIZE,
            "sha256": DRIVER.REQUEST_SHA256,
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
                "request_body_sha256": hashlib.sha256(b"{}").hexdigest(),
            },
        },
        "workload": {
            "rendered_prompt": DRIVER.RENDERED_PROMPT,
            "prompt_token_ids": DRIVER.PROMPT_TOKEN_IDS,
            "generated_token_ids": list(range(128)),
            "generated_token_ids_sha256": hashlib.sha256(
                compact(list(range(128)))
            ).hexdigest(),
            "content": "fixture",
            "content_sha256": hashlib.sha256(b"fixture").hexdigest(),
            "usage": [13, 128, 141],
            "usage_object": {
                "prompt_tokens": 13,
                "completion_tokens": 128,
                "total_tokens": 141,
                "prompt_tokens_details": {"cached_tokens": 0},
            },
            "generation_settings": {"seed": 0, "temperature": 0},
            "generation_settings_sha256": hashlib.sha256(
                compact({"seed": 0, "temperature": 0})
            ).hexdigest(),
            "per_sample_tokenize_admission_outside_timed_interval": {
                "token_ids": DRIVER.PROMPT_TOKEN_IDS,
                "transport": {},
            },
        },
        "timing": {
            "clock": "monotonic",
            "clock_identity": "fixture-monotonic",
            "clock_resolution_ns": 1,
            "clock_is_monotonic": True,
            "clock_is_adjustable": False,
            "start_boundary": fixture_contract()["timing_contract"][DRIVER.EDGE_ID][
                "start"
            ],
            "end_boundary": fixture_contract()["timing_contract"][DRIVER.EDGE_ID][
                "end"
            ],
            "implementation_start_boundary": DRIVER.GATEWAY_TIMING_START_V3,
            "implementation_end_boundary": "immediately-after-full-body-strict-JSON-parse-and-semantic-validation",
            "request_serialization_before_start": True,
            "first_wire_byte_send_call_immediately_after_start": True,
            **fixture_generation_wire_timing(slot["arm"]),
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
            "predicted_ms": native_ms,
            "prompt_tps": 1300.0,
            "predicted_tps": 128000.0 / native_ms,
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
            "sha256": hashlib.sha256(response_raw).hexdigest(),
        },
    }


def fixture_execution_plan() -> dict[str, object]:
    campaign_root = Path("/private/tmp/gateway-formal-v3-campaign-fixture")
    state_root = campaign_root / "state"
    runtime_root = campaign_root / "runtime"
    log_root = state_root / ".local" / "logs"
    runtime_logs = runtime_root / "llama.cpp-mac" / "logs"
    return {
        "format": DRIVER.PLAN_FORMAT,
        "schema_version": 3,
        "edge_id": DRIVER.EDGE_ID,
        "repository_root": str(ROOT),
        "contract_repository_path": DRIVER.CONTRACT_REPOSITORY_PATH,
        "validator_repository_path": DRIVER.VALIDATOR_REPOSITORY_PATH,
        "driver_repository_path": DRIVER.DRIVER_REPOSITORY_PATH,
        "plan_repository_path": "benchmarks/cross_runtime/fixture-plan.json",
        "marker_repository_path": DRIVER.MARKER_REPOSITORY_PATH,
        "raw_output_path": str(campaign_root / "raw" / "receipt.json"),
        "artifacts": {
            "model": {
                "absolute_path": DRIVER.MODEL_PATH,
                "size_bytes": DRIVER.MODEL_SIZE,
                "sha256": DRIVER.MODEL_SHA256,
            },
            "omniinfer_cli": {
                "absolute_path": "/opt/pinned/omniinfer",
                "size_bytes": DRIVER.OMNI_CLI_SIZE,
                "sha256": DRIVER.OMNI_CLI_SHA256,
            },
            "gateway_backend": {
                "absolute_path": "/opt/pinned/llama-server",
                "size_bytes": DRIVER.BACKEND_BINARY_SIZE,
                "sha256": DRIVER.BACKEND_BINARY_SHA256,
            },
        },
        "runtime": {
            "omni_base_url": "http://127.0.0.1:19000",
            "expected_gateway_argv": [
                "/opt/pinned/omniinfer",
                "--state-root",
                str(state_root),
                "--runtime-root",
                str(runtime_root),
                "gateway",
                "--host",
                "127.0.0.1",
                "--port",
                "19000",
                "--startup-timeout",
                "180",
            ],
            "expected_gateway_environment": {
                "LANG": "C",
                "OMNIINFER_LLAMA_CPP_MAC_LAUNCHER_PATH": "/opt/pinned/llama-server",
                "OMNIINFER_REQUEST_HISTORY": "0",
                "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
            },
            "slot_save_path": str(campaign_root / "slots"),
            "runtime_logs_path": str(runtime_logs),
            "gateway_logs_path": str(log_root),
            "history_root": str(state_root / ".local" / "request_history"),
            "mutable_log_roots": [
                str(runtime_logs),
                str(log_root),
            ],
            "custodian_control_socket_path": str(log_root / "custodian.sock"),
            "custodian_ready_timeout_seconds": 180,
            "custodian_shutdown_timeout_seconds": 30,
        },
    }


def fixture_campaign_directory_plan(parent: Path) -> dict[str, object]:
    plan = fixture_execution_plan()
    campaign_root = parent / "campaign"
    state_root = campaign_root / "state"
    runtime_root = campaign_root / "runtime"
    gateway_logs = state_root / ".local" / "logs"
    history_root = state_root / ".local" / "request_history"
    runtime_logs = runtime_root / "llama.cpp-mac" / "logs"
    plan["raw_output_path"] = str(campaign_root / "raw" / "receipt.json")
    plan["runtime"] = {
        **plan["runtime"],
        "expected_gateway_argv": [
            "/opt/pinned/omniinfer",
            "--state-root",
            str(state_root),
            "--runtime-root",
            str(runtime_root),
            "gateway",
            "--host",
            "127.0.0.1",
            "--port",
            "19000",
            "--startup-timeout",
            "180",
        ],
        "slot_save_path": str(campaign_root / "slots"),
        "runtime_logs_path": str(runtime_logs),
        "gateway_logs_path": str(gateway_logs),
        "history_root": str(history_root),
        "mutable_log_roots": [str(runtime_logs), str(gateway_logs)],
        "custodian_control_socket_path": str(gateway_logs / "custodian.sock"),
    }
    return plan


def fixture_campaign_directory_initialization_receipt(
    plan: dict[str, object] | None = None,
) -> dict[str, object]:
    selected_plan = plan or fixture_execution_plan()
    layout = DRIVER.campaign_directory_layout(selected_plan)
    root = Path(layout["campaign_root"])
    observations = []
    for index, path_text in enumerate(layout["expected_directory_paths"], start=1):
        expected_children = layout["expected_child_names"][path_text]
        observations.append(
            {
                "absolute_path": path_text,
                "relative_path": str(Path(path_text).relative_to(root)),
                "device": 1,
                "inode": index,
                "mode": stat.S_IFDIR | 0o700,
                "permission_bits": 0o700,
                "uid": os.geteuid(),
                "gid": os.getegid(),
                "expected_child_names": copy.deepcopy(expected_children),
                "observed_child_names": copy.deepcopy(expected_children),
                "direct_directory_no_symlink": True,
                "owner_matches_controller": True,
                "permissions_are_0700": True,
            }
        )
    return {
        "format": "apxinf-omniinfer-gateway-campaign-directory-initialization-v3",
        "schema_version": 3,
        "edge_id": DRIVER.EDGE_ID,
        "campaign_root": layout["campaign_root"],
        "expected_directory_paths": copy.deepcopy(layout["expected_directory_paths"]),
        "preexisting_directory_paths": [],
        "created_directory_paths": copy.deepcopy(layout["expected_directory_paths"]),
        "directory_observations": observations,
        "retry_policy": {
            "exact_empty_partial_tree_reusable": True,
            "exact_empty_complete_tree_reusable": True,
            "preexisting_content_or_unexpected_entry_rejected": True,
            "symlink_or_non_directory_rejected": True,
            "cleanup_limited_to_directories_created_by_this_attempt": True,
        },
        "initial_tree_sha256": DRIVER.sha256_canonical(observations),
        "generation_requests": 0,
        "runtime_processes_started": 0,
        "marker_created": False,
        "raw_created": False,
        "all_passed": True,
    }


def fixture_model_select_response(
    plan: dict[str, object], backend_pid: int = 12, backend_port: int = 19001
) -> dict[str, object]:
    command = DRIVER.expected_backend_launch_args(plan, backend_port)
    return {
        "ok": True,
        "already_loaded": False,
        "requires_reload": False,
        "model": DRIVER.MODEL_PATH,
        "owner_admin_id": "local",
        "selected_backend": "llama.cpp-mac",
        "selected_model": DRIVER.MODEL_PATH,
        "selected_public_model_id": None,
        "selected_mmproj": None,
        "selected_ctx_size": 256,
        "request_defaults": {},
        "backend_pid": backend_pid,
        "backend_port": backend_port,
        "generation": 1,
        "route_state": "ready",
        "allocation_id": 1,
        "resource_budget": {
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
        },
        "speculative_admission": None,
        "launch_command": command,
        "log_path": DRIVER.expected_gateway_backend_log_path(plan),
        "external_server_protocol": "llama.cpp-server",
        "client_endpoint": f"http://127.0.0.1:{backend_port}",
        "openai_compatible": True,
    }


def fixture_unloaded_gateway_state() -> dict[str, object]:
    return {
        "backend": None,
        "backend_ready": False,
        "model": None,
        "public_model_id": None,
        "mmproj": None,
        "ctx_size": None,
        "request_defaults": {},
        "runtime_mode": None,
        "backend_pid": None,
        "backend_port": None,
        "launch_args": [],
        "cuda_visible_devices": None,
        "warning": None,
        "launch_command": [],
        "proxy_model": None,
        "external_server_protocol": None,
        "client_endpoint": None,
        "openai_compatible": False,
        "backend_log": None,
        "effective_parameters": {},
        "runtime": None,
        "default_model": None,
        "loaded_models": [],
        "restore_selection": None,
        "restore_status": "not_configured",
        "restore_completed": False,
        "resource_ledger": None,
        "available_backends": [{"id": "llama.cpp-mac"}],
    }


def fixture_loaded_gateway_state(
    plan: dict[str, object], backend_pid: int = 12, backend_port: int = 19001
) -> dict[str, object]:
    command = DRIVER.expected_backend_launch_args(plan, backend_port)
    launch_args = command[10:-2]
    endpoint = f"http://127.0.0.1:{backend_port}"
    log_path = DRIVER.expected_gateway_backend_log_path(plan)
    resource_budget = DRIVER._expected_gateway_resource_budget()
    committed = resource_budget["domains_bytes"]
    capacity = {"unified:system": 8_000_000_000}
    resource_ledger = {
        "capacity_snapshot_id": 1,
        "capacity_bytes": capacity,
        "reserved_bytes": {},
        "committed_bytes": committed,
        "available_bytes": {
            domain: capacity[domain] - value for domain, value in committed.items()
        },
    }
    loaded_model = {
        "id": DRIVER.MODEL_PATH,
        "owner_admin_id": "local",
        "backend": "llama.cpp-mac",
        "model": DRIVER.MODEL_PATH,
        "model_path": DRIVER.MODEL_PATH,
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
        "resource_budget": resource_budget,
        "speculative_admission": None,
        "launch_args": launch_args,
        "cuda_visible_devices": None,
        "warning": None,
        "launch_command": command,
        "proxy_model": None,
        "external_server_protocol": "llama.cpp-server",
        "client_endpoint": endpoint,
        "openai_compatible": True,
        "backend_log": log_path,
    }
    return {
        "backend": "llama.cpp-mac",
        "backend_ready": True,
        "model": DRIVER.MODEL_PATH,
        "model_path": DRIVER.MODEL_PATH,
        "default_model": DRIVER.MODEL_PATH,
        "public_model_id": None,
        "owner_admin_id": "local",
        "mmproj": None,
        "ctx_size": 256,
        "request_defaults": {},
        "runtime_mode": "external_server",
        "effective_parameters": {},
        "proxy_model": None,
        "backend_pid": backend_pid,
        "backend_port": backend_port,
        "generation": 1,
        "route_state": "ready",
        "allocation_id": 1,
        "resource_budget": resource_budget,
        "speculative_admission": None,
        "client_endpoint": endpoint,
        "launch_command": command,
        "launch_args": launch_args,
        "cuda_visible_devices": None,
        "warning": None,
        "external_server_protocol": "llama.cpp-server",
        "openai_compatible": True,
        "backend_log": log_path,
        "log_path": log_path,
        "runtime": {
            "mode": "external_server",
            "host": "127.0.0.1",
            "pid": backend_pid,
            "port": backend_port,
            "cuda_visible_devices": None,
            "client_endpoint": endpoint,
            "launch_command": command,
            "log_path": log_path,
            "proxy_model_ref": None,
            "external_server_protocol": "llama.cpp-server",
            "openai_compatible": True,
        },
        "loaded_models": [loaded_model],
        "restore_selection": {
            "backend": "llama.cpp-mac",
            "model": DRIVER.MODEL_PATH,
            "mmproj": None,
            "no_mmproj": True,
            "ctx_size": 256,
            "request_defaults": {},
        },
        "restore_status": "loaded",
        "restore_completed": True,
        "resource_ledger": resource_ledger,
        "available_backends": [{"id": "llama.cpp-mac"}],
    }


def fixture_management_transport(
    base_url: str,
    label: str,
    method: str,
    path: str,
    body: bytes | None,
    response: dict[str, object],
) -> dict[str, object]:
    port = urllib.parse.urlsplit(base_url).port
    assert port is not None
    headers = [
        f"{method} {path} HTTP/1.1",
        f"Host: 127.0.0.1:{port}",
        "Accept: application/json",
        "Connection: keep-alive",
    ]
    if body is not None:
        headers.extend(
            ["Content-Type: application/json", f"Content-Length: {len(body)}"]
        )
    body_bytes = body or b""
    wire = ("\r\n".join(headers) + "\r\n\r\n").encode("ascii") + body_bytes
    response_bytes = compact(response)
    return {
        "connection": label,
        "method": method,
        "path": path,
        "request_body_size_bytes": len(body_bytes),
        "request_body_sha256": hashlib.sha256(body).hexdigest()
        if body is not None
        else None,
        "request_wire_size_bytes": len(wire),
        "request_wire_sha256": hashlib.sha256(wire).hexdigest(),
        "request_wire_base64": base64.b64encode(wire).decode("ascii"),
        "request_wire_body_offset_bytes": len(wire) - len(body_bytes),
        "request_wire_body_size_bytes": len(body_bytes),
        "request_wire_body_sha256": hashlib.sha256(body_bytes).hexdigest(),
        "request_wire_body_equals_request_body": True,
        "single_sendall_call_count": 1,
        "single_sendall_argument_size_bytes": len(wire),
        "single_sendall_argument_sha256": hashlib.sha256(wire).hexdigest(),
        "status": 200,
        "http_version": 11,
        "response_size_bytes": len(response_bytes),
        "response_sha256": hashlib.sha256(response_bytes).hexdigest(),
        "response_base64": base64.b64encode(response_bytes).decode("ascii"),
        "request_serialization_before_start": True,
        "first_wire_byte_send_call_immediately_after_start": True,
        "complete_HTTP_request_wire_serialization_before_start": True,
        "single_sendall_call_for_complete_request_wire_required": True,
        "full_response_body_read_before_end": True,
        "strict_json_parse_before_end": True,
        "semantic_validation_before_end": True,
    }


def fixture_zero_generation_model_load_receipt(
    plan: dict[str, object] | None = None,
) -> dict[str, object]:
    selected_plan = plan or fixture_execution_plan()
    gateway_start = {"seconds": 3, "microseconds": 4}
    backend_start = {"seconds": 5, "microseconds": 6}
    payloads = iter(
        (
            fixture_unloaded_gateway_state(),
            fixture_model_select_response(selected_plan),
            fixture_loaded_gateway_state(selected_plan),
        )
    )

    class GatewayProcess:
        pid = 11

        @staticmethod
        def poll() -> None:
            return None

    def request(
        base_url: str,
        label: str,
        method: str,
        path: str,
        body: bytes | None,
        validator: object,
    ) -> tuple[dict[str, object], object, dict[str, object]]:
        payload = next(payloads)
        return (
            payload,
            validator(payload),
            fixture_management_transport(base_url, label, method, path, body, payload),
        )

    history = {"exists": False, "entries": [], "canonical_sha256": "0" * 64}
    return DRIVER.admit_zero_generation_model_load(
        selected_plan,
        GatewayProcess(),
        gateway_start,
        expected_gateway_parent_pid=10,
        request_json=request,
        process_start_reader=lambda pid: copy.deepcopy(
            gateway_start if pid == 11 else backend_start
        ),
        parent_pid_reader=lambda pid: 11 if pid == 12 else 10,
        tree_reader=lambda _path: copy.deepcopy(history),
        monotonic=lambda: 0.0,
        sleeper=lambda _seconds: None,
    )


def fixture_runtime_preflight(
    plan: dict[str, object] | None = None,
) -> dict[str, object]:
    selected_plan = plan or fixture_execution_plan()
    zero_generation_model_load = fixture_zero_generation_model_load_receipt(
        selected_plan
    )
    attestation = {
        "daemon_challenge_response": {
            "custodian_pid": 10,
            "custodian_start_identity": {"seconds": 1, "microseconds": 2},
            "gateway_pid": 11,
            "gateway_start_identity": {"seconds": 3, "microseconds": 4},
            "backend_pid": 12,
            "backend_start_identity": {"seconds": 5, "microseconds": 6},
            "lifecycle_sequence": {"fixed": True},
            "controller_preload_fd": {"fd": 7},
            "backend_loaded_fd": {"fd": 8},
        }
    }
    return {
        "format": "apxinf-qwen35-omniinfer-gateway-runtime-preflight-v3",
        "schema_version": 3,
        "edge_id": DRIVER.EDGE_ID,
        "generation_requests": 0,
        "same_resident_backend_process_for_B_and_G": True,
        "direct_arm_backend_endpoint": "http://127.0.0.1:19001",
        "gateway_arm_endpoint": "http://127.0.0.1:19000",
        "gateway_process_start": {"pid": 11},
        "gateway_process_end": {"pid": 11},
        "backend_process_start": {"pid": 12},
        "backend_process_end": {"pid": 12},
        "backend_start_end_identity_equal": True,
        "gateway_start_end_identity_equal": True,
        "custodian_binding": {
            "nonce": "a" * 64,
            "custodian_pid": 10,
            "gateway_pid": 11,
            "gateway_start_identity": {"seconds": 3, "microseconds": 4},
            "backend_pid": 12,
            "backend_start_identity": {"seconds": 5, "microseconds": 6},
            "zero_generation_model_load": copy.deepcopy(zero_generation_model_load),
            "ready_receipt": {
                "zero_generation_model_load": copy.deepcopy(zero_generation_model_load)
            },
            "campaign_directory_initialization": (
                fixture_campaign_directory_initialization_receipt(selected_plan)
            ),
        },
        "zero_generation_model_load": copy.deepcopy(zero_generation_model_load),
        "custodian_process_start": {"pid": 10},
        "custodian_process_end": {"pid": 10},
        "controller_backend_model_fd_custody": {
            "start": copy.deepcopy(attestation),
            "end": copy.deepcopy(attestation),
            "controller_and_backend_same_vnode_identity": True,
            "same_open_file_description_not_claimed": True,
            "controller_fd_open_completed_before_gateway_backend_launch": True,
        },
        "state": {"status": "ready"},
        "state_before_after_equal": True,
        "health": {"status": "ok"},
        "props": {"status": "ok"},
        "canonical_request": {
            "object": DRIVER.REQUEST,
            "size_bytes": DRIVER.REQUEST_SIZE,
            "sha256": DRIVER.REQUEST_SHA256,
            "B_G_body_identity_required": True,
        },
        "rendered_prompt": DRIVER.RENDERED_PROMPT,
        "rendered_prompt_token_ids": DRIVER.PROMPT_TOKEN_IDS,
        "cache_clear": {"acknowledged": True},
        "history_start": {},
        "history_end": {},
        "history_start_end_equal": True,
        "mutable_logs_start": [],
        "mutable_logs_end": [],
        "mutable_logs_equality_not_required": True,
        "transport_receipts": {},
        "all_passed": True,
    }


class StrictJsonTests(unittest.TestCase):
    def test_rejects_duplicate_keys_and_nonfinite_constants(self) -> None:
        for raw in (b'{"x":1,"x":2}', b'{"x":NaN}', b'{"x":Infinity}'):
            with self.subTest(raw=raw):
                with self.assertRaises(DRIVER.ReceiptError):
                    DRIVER.parse_strict_json_document(raw)

    def test_canonical_request_is_exact(self) -> None:
        self.assertEqual(len(DRIVER.REQUEST_BYTES), 383)
        self.assertEqual(
            hashlib.sha256(DRIVER.REQUEST_BYTES).hexdigest(),
            "7773f5337693843f1e8cf3017b98868517cbddd3bc32649e550d8f2fec1d5cf6",
        )


class ScheduleTests(unittest.TestCase):
    def test_exact_warmup_and_macroblock_schedule(self) -> None:
        schedule = DRIVER.declared_schedule(fixture_contract())
        self.assertEqual(len(schedule), 136)
        warmup = [slot for slot in schedule if slot["phase"] == "warmup"]
        timed = [slot for slot in schedule if slot["phase"] == "timed"]
        self.assertEqual([slot["arm"] for slot in warmup], list("BGGBGBBG"))
        self.assertEqual(sum(slot["arm"] == "B" for slot in warmup), 4)
        self.assertEqual(sum(slot["arm"] == "G" for slot in warmup), 4)
        self.assertEqual(len(timed), 128)
        self.assertEqual(sum(slot["arm"] == "B" for slot in timed), 64)
        self.assertEqual(sum(slot["arm"] == "G" for slot in timed), 64)
        self.assertEqual([slot["arm"] for slot in timed[:8]], list("BGGBGBBG"))
        self.assertEqual([slot["arm"] for slot in timed[8:16]], list("GBBGBGGB"))


class SemanticAdmissionTests(unittest.TestCase):
    def test_sample_rejects_unknown_top_level_claim_fields(self) -> None:
        contract = fixture_contract()
        slot = DRIVER.declared_schedule(contract)[0]
        sample = fixture_sample(slot, 1000.0)
        DRIVER.validate_sample_receipt(sample, slot, contract)

        for field, value in (
            ("controller_and_backend_same_open_file_description_claimed", True),
            ("decision", {"engine_winner_or_ranking_claim_allowed": True}),
        ):
            changed = copy.deepcopy(sample)
            changed[field] = value
            with self.subTest(field=field):
                with self.assertRaises(DRIVER.ReceiptError):
                    DRIVER.validate_sample_receipt(changed, slot, contract)

    def test_sample_rejects_unknown_fields_in_key_nested_receipts(self) -> None:
        contract = fixture_contract()
        slot = DRIVER.declared_schedule(contract)[0]
        sample = fixture_sample(slot, 1000.0)
        paths = (
            ("request",),
            ("cache_clear",),
            ("workload",),
            ("workload", "per_sample_tokenize_admission_outside_timed_interval"),
            ("timing",),
            ("native_sensitivity",),
            ("connection",),
            ("response_bytes",),
        )
        for path in paths:
            changed = copy.deepcopy(sample)
            target = changed
            for field in path:
                target = target[field]
            target["engine_winner_or_ranking_claim_allowed"] = True
            with self.subTest(path=path):
                with self.assertRaises(DRIVER.ReceiptError):
                    DRIVER.validate_sample_receipt(changed, slot, contract)

    def test_sample_rejects_engine_claim_hidden_in_response_bytes(self) -> None:
        contract = fixture_contract()
        slot = DRIVER.declared_schedule(contract)[0]
        sample = fixture_sample(slot, 1000.0)
        sample["response"].update(
            {"engine_ranking": True, "engine_winner": "OmniInfer"}
        )
        raw = compact(sample["response"])
        sample["response_bytes"] = {
            "encoding": "base64",
            "base64": base64.b64encode(raw).decode("ascii"),
            "size_bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
        with self.assertRaises(DRIVER.ReceiptError):
            DRIVER.validate_sample_receipt(sample, slot, contract)

    def test_response_records_raw_prompt_and_free128(self) -> None:
        tokens = list(range(128))
        admitted = DRIVER.validate_gateway_response(
            fixture_response(tokens), fixture_contract(tokens), DRIVER.MODEL_PATH
        )
        self.assertEqual(admitted["prompt_token_ids"], DRIVER.PROMPT_TOKEN_IDS)
        self.assertEqual(admitted["generated_token_ids"], tokens)

    def test_wrong_trajectory_is_rejected_even_with_128_tokens(self) -> None:
        response = fixture_response()
        response["__verbose"]["tokens"][0] = 999
        with self.assertRaises(DRIVER.ReceiptError):
            DRIVER.validate_gateway_response(
                response, fixture_contract(), DRIVER.MODEL_PATH
            )

    def test_v1_v2_or_parse_excluded_sample_is_never_reusable(self) -> None:
        contract = fixture_contract()
        slot = DRIVER.declared_schedule(contract)[0]
        sample = fixture_sample(slot, 1000.0)
        for mutation in ("schema", "format", "parse"):
            candidate = copy.deepcopy(sample)
            if mutation == "schema":
                candidate["schema_version"] = 2
            elif mutation == "format":
                candidate["format"] = "diagnostic-v1"
            else:
                candidate["timing"]["json_parse_excluded_from_wall"] = True
            with self.subTest(mutation=mutation):
                with self.assertRaises(DRIVER.ReceiptError):
                    DRIVER.validate_sample_receipt(candidate, slot, contract)

    def test_backend_identity_any_drift_is_rejected(self) -> None:
        identity = {
            "pid": 42,
            "process_start_identity": {"seconds": 1, "microseconds": 2},
            "argv_sha256": "a" * 64,
            "environment_sha256": "b" * 64,
            "loaded_model_fd": {"fd": 7, "device": 1, "inode": 2},
            "runtime_closure_sha256": "c" * 64,
        }
        DRIVER.require_same_backend_identity(identity, copy.deepcopy(identity))
        for key in identity:
            changed = copy.deepcopy(identity)
            changed[key] = 999 if key == "pid" else {"drift": True}
            with self.subTest(key=key):
                with self.assertRaises(DRIVER.CampaignError):
                    DRIVER.require_same_backend_identity(identity, changed)

    def test_controller_and_backend_fd_require_lsof_libproc_exact_agreement(
        self,
    ) -> None:
        controller = {
            "device": 1,
            "inode": 2,
            "mode": 0o100400,
            "link_count": 1,
            "size_bytes": DRIVER.MODEL_SIZE,
            "ctime_ns": 123,
            "sha256": DRIVER.MODEL_SHA256,
        }
        proc_entry = {
            "fd": 7,
            "fd_type": "vnode",
            "open_flags": 1,
            "close_on_exec": False,
            "device": 1,
            "inode": 2,
            "mode": 0o100400,
            "link_count": 1,
            "size_bytes": DRIVER.MODEL_SIZE,
            "ctime_ns": 123,
            "path": DRIVER.MODEL_PATH,
        }
        lsof_entry = {
            "fd": "7",
            "access": "r",
            "type": "REG",
            "device_text": "0x1",
            "inode_text": "2",
            "size_text": str(DRIVER.MODEL_SIZE),
            "path": DRIVER.MODEL_PATH,
        }
        with (
            mock.patch.object(
                DRIVER, "proc_vnode_fd_entries", return_value=[proc_entry]
            ),
            mock.patch.object(DRIVER, "lsof_entries", return_value=[lsof_entry]),
        ):
            proof = DRIVER.backend_loaded_model_fd_proof(42, controller)
        self.assertTrue(proof["lsof_libproc_agree"])
        self.assertTrue(proof["controller_backend_file_identity_equal"])
        self.assertEqual(proof["backend_loaded_fd"]["fd"], 7)

        changed = copy.deepcopy(proc_entry)
        changed["ctime_ns"] += 1
        with (
            mock.patch.object(DRIVER, "proc_vnode_fd_entries", return_value=[changed]),
            mock.patch.object(DRIVER, "lsof_entries", return_value=[lsof_entry]),
        ):
            with self.assertRaises(DRIVER.CampaignError):
                DRIVER.backend_loaded_model_fd_proof(42, controller)

        changed = copy.deepcopy(proc_entry)
        changed["link_count"] = 2
        with (
            mock.patch.object(DRIVER, "proc_vnode_fd_entries", return_value=[changed]),
            mock.patch.object(DRIVER, "lsof_entries", return_value=[lsof_entry]),
        ):
            with self.assertRaises(DRIVER.CampaignError):
                DRIVER.backend_loaded_model_fd_proof(42, controller)

    def test_controller_fd_lifecycle_sequence_must_precede_gateway_spawn(self) -> None:
        DRIVER.validate_controller_launch_sequence(
            {
                "controller_fd_open_started_monotonic_ns": 10,
                "controller_fd_custody_complete_monotonic_ns": 20,
                "gateway_spawn_invocation_monotonic_ns": 21,
                "gateway_kernel_identity_observed_monotonic_ns": 30,
                "backend_kernel_identity_observed_monotonic_ns": 40,
            }
        )
        with self.assertRaises(DRIVER.CampaignError):
            DRIVER.validate_controller_launch_sequence(
                {
                    "controller_fd_open_started_monotonic_ns": 10,
                    "controller_fd_custody_complete_monotonic_ns": 22,
                    "gateway_spawn_invocation_monotonic_ns": 21,
                    "gateway_kernel_identity_observed_monotonic_ns": 30,
                    "backend_kernel_identity_observed_monotonic_ns": 40,
                }
            )

    def test_controller_fd_is_no_follow_cloexec_and_held(self) -> None:
        payload = b"fixture-model-bytes"
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "model.gguf"
            model.write_bytes(payload)
            ticks = iter((10, 20))
            custody = DRIVER.ControllerModelFd(
                model,
                len(payload),
                hashlib.sha256(payload).hexdigest(),
                clock_ns=lambda: next(ticks),
            )
            try:
                first = custody.observe("before-runtime-spawn")
                second = custody.observe("postflight-before-cleanup")
            finally:
                custody.close()
            self.assertEqual(first["fd"], second["fd"])
            self.assertEqual(first["device"], second["device"])
            self.assertEqual(
                first["open_flags"], ["O_RDONLY", "O_NOFOLLOW", "O_CLOEXEC"]
            )

            link = Path(directory) / "model-link.gguf"
            link.symlink_to(model)
            with self.assertRaises(DRIVER.CampaignError):
                DRIVER.ControllerModelFd(
                    link, len(payload), hashlib.sha256(payload).hexdigest()
                )

            hardlink = Path(directory) / "model-hardlink.gguf"
            os.link(model, hardlink)
            with self.assertRaises(DRIVER.CampaignError):
                DRIVER.ControllerModelFd(
                    model, len(payload), hashlib.sha256(payload).hexdigest()
                )

    def test_cache_clear_ack_is_exact_not_truthy(self) -> None:
        contract = fixture_contract()
        slot = DRIVER.declared_schedule(contract)[0]
        sample = fixture_sample(slot, 1000.0)
        sample["cache_clear"]["cleared_slots"] = [0, 1]
        with self.assertRaises(DRIVER.ReceiptError):
            DRIVER.validate_sample_receipt(sample, slot, contract)

        sample = fixture_sample(slot, 1000.0)
        sample["cache_clear"]["response"]["cleared_slots"] = [0, 1]
        with self.assertRaises(DRIVER.ReceiptError):
            DRIVER.validate_sample_receipt(sample, slot, contract)

    def test_retained_response_bytes_are_reparsed_and_semantically_revalidated(
        self,
    ) -> None:
        contract = fixture_contract()
        slot = DRIVER.declared_schedule(contract)[0]
        sample = fixture_sample(slot, 1000.0)
        invalid = copy.deepcopy(sample["response"])
        invalid["choices"][0]["finish_reason"] = "stop"
        raw = compact(invalid)
        sample["response"] = invalid
        sample["response_bytes"] = {
            "encoding": "base64",
            "base64": base64.b64encode(raw).decode("ascii"),
            "size_bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
        with self.assertRaises(DRIVER.ReceiptError):
            DRIVER.validate_sample_receipt(sample, slot, contract)


class TimingBoundaryTests(unittest.TestCase):
    def test_sample_admits_the_exact_complete_wire_timing_contract(self) -> None:
        contract = fixture_contract()
        contract["timing_contract"][DRIVER.EDGE_ID].update(
            {
                "start": DRIVER.GATEWAY_TIMING_START_V3,
                "complete_HTTP_request_wire_serialization_before_start": True,
                "single_sendall_call_for_complete_request_wire_required": True,
                "canonical_383_byte_JSON_body_identical_between_B_and_G": True,
                "arm_specific_HTTP_authority_header_difference_is_inside_timed_region": True,
                "body_only_timing_allowed": False,
            }
        )
        slot = DRIVER.declared_schedule(contract)[0]
        sample = fixture_sample(slot, 1000.0)
        sample["timing"].update(fixture_generation_wire_timing(slot["arm"]))
        sample["timing"]["start_boundary"] = DRIVER.GATEWAY_TIMING_START_V3
        sample["timing"]["implementation_start_boundary"] = (
            DRIVER.GATEWAY_TIMING_START_V3
        )
        DRIVER.validate_sample_receipt(sample, slot, contract)

    def test_pair_requires_distinct_authorities_with_identical_canonical_bodies(
        self,
    ) -> None:
        contract = fixture_contract()
        left_slot, right_slot = DRIVER.declared_schedule(contract)[:2]
        left = fixture_sample(left_slot, 1000.0)
        right = fixture_sample(right_slot, 1001.0)
        left["timing"].update(fixture_generation_wire_timing(left_slot["arm"]))
        right["timing"].update(fixture_generation_wire_timing(right_slot["arm"]))
        DRIVER.validate_pair_equal(left, right)

        same_authority = copy.deepcopy(right)
        same_authority["timing"].update(
            fixture_generation_wire_timing(left_slot["arm"])
        )
        with self.assertRaises(DRIVER.ReceiptError):
            DRIVER.validate_pair_equal(left, same_authority)

        extra_header = copy.deepcopy(right)
        timing = extra_header["timing"]
        wire = base64.b64decode(timing["request_wire_base64"])
        header, body = wire.split(b"\r\n\r\n", 1)
        changed_wire = header + b"\r\nX-Unbound: true\r\n\r\n" + body
        changed_hash = hashlib.sha256(changed_wire).hexdigest()
        timing.update(
            {
                "request_wire_size_bytes": len(changed_wire),
                "request_wire_sha256": changed_hash,
                "request_wire_base64": base64.b64encode(changed_wire).decode("ascii"),
                "request_wire_body_offset_bytes": len(changed_wire) - len(body),
                "single_sendall_argument_size_bytes": len(changed_wire),
                "single_sendall_argument_sha256": changed_hash,
            }
        )
        with self.assertRaises(DRIVER.ReceiptError):
            DRIVER.validate_pair_equal(left, extra_header)

    def test_timer_covers_send_read_parse_and_semantic_validation(self) -> None:
        events: list[str] = []
        sent_wires: list[bytes] = []

        class FakeSocket:
            def __init__(self) -> None:
                self._fd = 9

            def fileno(self) -> int:
                return self._fd

            def getsockname(self) -> tuple[str, int]:
                return ("127.0.0.1", 40000)

            def getpeername(self) -> tuple[str, int]:
                return ("127.0.0.1", 9000)

            def sendall(self, wire: bytes) -> None:
                sent_wires.append(wire)
                events.append("send")

            def close(self) -> None:
                self._fd = -1

        class FakeResponse:
            status = 200
            version = 11
            will_close = False

            def __init__(self, _: object) -> None:
                pass

            def begin(self) -> None:
                events.append("headers")

            def read(self, _amount: int | None = None) -> bytes:
                events.append("read")
                return b'{"ok":true}'

            def getheaders(self) -> list[tuple[str, str]]:
                return [("Content-Type", "application/json")]

            def close(self) -> None:
                events.append("response-close")

        ticks = iter((100, 200))

        def clock_ns() -> int:
            value = next(ticks)
            events.append("clock-start" if value == 100 else "clock-end")
            return value

        connection = DRIVER.PersistentHttpJsonConnection(
            "http://127.0.0.1:9000",
            "fixture",
            socket_factory=lambda *_args, **_kwargs: FakeSocket(),
            response_factory=FakeResponse,
            clock_ns=clock_ns,
        )
        connection.connect()

        def validate(value: dict[str, object]) -> dict[str, object]:
            events.append("semantic")
            return value

        _, validated, receipt = connection.request_json(
            "POST", "/v1/chat/completions", DRIVER.REQUEST_BYTES, validate
        )
        self.assertEqual(validated, {"ok": True})
        self.assertLess(events.index("clock-start"), events.index("send"))
        self.assertLess(events.index("send"), events.index("read"))
        self.assertLess(events.index("read"), events.index("semantic"))
        self.assertLess(events.index("semantic"), events.index("clock-end"))
        self.assertEqual(receipt["started_monotonic_ns"], 100)
        self.assertEqual(receipt["ended_monotonic_ns"], 200)
        self.assertFalse(receipt["json_parse_excluded_from_wall"])
        self.assertFalse(receipt["semantic_validation_excluded_from_wall"])
        self.assertEqual(len(sent_wires), 1)
        self.assertTrue(sent_wires[0].endswith(DRIVER.REQUEST_BYTES))
        self.assertEqual(receipt["single_sendall_call_count"], 1)
        self.assertEqual(
            receipt["single_sendall_argument_sha256"],
            hashlib.sha256(sent_wires[0]).hexdigest(),
        )
        self.assertEqual(
            receipt["timing_event_order"], DRIVER.COMPLETE_WIRE_TIMING_EVENT_ORDER_V3
        )
        self.assertTrue(
            receipt["complete_HTTP_request_wire_serialization_before_start"]
        )
        self.assertTrue(
            receipt["single_sendall_call_for_complete_request_wire_required"]
        )
        self.assertTrue(
            receipt["canonical_383_byte_JSON_body_identical_between_B_and_G"]
        )
        self.assertTrue(
            receipt[
                "arm_specific_HTTP_authority_header_difference_is_inside_timed_region"
            ]
        )
        self.assertFalse(receipt["body_only_timing_allowed"])

    def test_incomplete_http_body_retains_exact_received_prefix(self) -> None:
        class FakeSocket:
            def fileno(self) -> int:
                return 9

            def getsockname(self) -> tuple[str, int]:
                return ("127.0.0.1", 40000)

            def getpeername(self) -> tuple[str, int]:
                return ("127.0.0.1", 9000)

            def sendall(self, _: bytes) -> None:
                pass

        class IncompleteResponse:
            status = 200
            version = 11
            will_close = False

            def __init__(self, _: object) -> None:
                pass

            def begin(self) -> None:
                pass

            def getheaders(self) -> list[tuple[str, str]]:
                return [("Content-Type", "application/json")]

            def read(self, _amount: int) -> bytes:
                raise http.client.IncompleteRead(b'{"partial":', 5)

        connection = DRIVER.PersistentHttpJsonConnection(
            "http://127.0.0.1:9000",
            "fixture",
            socket_factory=lambda *_args, **_kwargs: FakeSocket(),
            response_factory=IncompleteResponse,
        )
        connection.connect()
        with self.assertRaises(DRIVER.RuntimeObservationError) as caught:
            connection.request_json(
                "POST", "/v1/chat/completions", b"{}", lambda value: value
            )
        self.assertEqual(caught.exception.observation["raw_response"], b'{"partial":')
        self.assertEqual(connection.last_raw_response, b'{"partial":')


class StatisticsTests(unittest.TestCase):
    def samples(self, delta: float) -> list[dict[str, object]]:
        contract = fixture_contract()
        result = []
        for slot in DRIVER.declared_schedule(contract):
            if slot["phase"] != "timed":
                continue
            wall = 1000.0 if slot["arm"] == "B" else 1000.0 + delta
            result.append(fixture_sample(slot, wall))
        return result

    def test_four_state_decision_table(self) -> None:
        contract = fixture_contract()
        equivalent = DRIVER.compute_gateway_statistics(self.samples(1.0), contract)
        positive = DRIVER.compute_gateway_statistics(self.samples(10.0), contract)
        inconclusive = DRIVER.compute_gateway_statistics(self.samples(-10.0), contract)
        unstable_samples = self.samples(1.0)
        for index, sample in enumerate(unstable_samples):
            sample["timing"]["client_full_response_wall_ms"] += index * 10.0
        unstable = DRIVER.compute_gateway_statistics(unstable_samples, contract)
        self.assertEqual(equivalent["decision"], "PRACTICALLY_EQUIVALENT_GATEWAY_PATH")
        self.assertEqual(positive["decision"], "POSITIVE_GATEWAY_PATH_OVERHEAD")
        self.assertEqual(inconclusive["decision"], "INCONCLUSIVE")
        self.assertEqual(unstable["decision"], "UNINTERPRETABLE")

    def test_recorded_statistics_must_equal_recomputation(self) -> None:
        contract = fixture_contract()
        samples = self.samples(1.0)
        recorded = DRIVER.compute_gateway_statistics(samples, contract)
        self.assertIsInstance(
            DRIVER.validate_recomputed_gateway_statistics(samples, recorded, contract),
            bool,
        )
        changed = copy.deepcopy(recorded)
        changed["decision"] = "POSITIVE_GATEWAY_PATH_OVERHEAD"
        with self.assertRaises(DRIVER.CampaignError):
            DRIVER.validate_recomputed_gateway_statistics(samples, changed, contract)

    def test_statistics_use_the_frozen_engine_claim_field(self) -> None:
        statistics = DRIVER.compute_gateway_statistics(
            self.samples(1.0), fixture_contract()
        )
        self.assertIs(statistics["engine_winner_or_ranking_claim_allowed"], False)
        self.assertNotIn("engine_ranking_claim_allowed", statistics)


class MachineSchemaTests(unittest.TestCase):
    def test_machine_receipt_rejects_unknown_claim_fields(self) -> None:
        receipt = {field: None for field in DRIVER.GATEWAY_MACHINE_RECEIPT_FIELDS_V3}
        receipt["contract_binding"] = {
            field: None for field in DRIVER.GATEWAY_CONTRACT_BINDING_FIELDS_V3
        }
        receipt["decision"] = {
            "label": "INCONCLUSIVE",
            "formal_summary_allowed": True,
            "engine_winner_or_ranking_claim_allowed": False,
        }
        DRIVER.validate_gateway_machine_schema(receipt, require_cleanup=True)

        top_level = copy.deepcopy(receipt)
        top_level["controller_and_backend_same_open_file_description_claimed"] = True
        with self.assertRaises(DRIVER.CampaignError):
            DRIVER.validate_gateway_machine_schema(top_level, require_cleanup=True)

        decision = copy.deepcopy(receipt)
        decision["decision"]["engine_winner_claim_allowed"] = True
        with self.assertRaises(DRIVER.CampaignError):
            DRIVER.validate_gateway_machine_schema(decision, require_cleanup=True)

        nested = copy.deepcopy(receipt)
        nested["contract_binding"].update(
            {"engine_ranking": True, "engine_winner": "OmniInfer"}
        )
        with self.assertRaises(DRIVER.CampaignError):
            DRIVER.validate_gateway_machine_schema(nested, require_cleanup=True)

        nonauthored_nested = copy.deepcopy(receipt)
        nonauthored_nested["host_custody"] = {
            "engine_ranking": True,
            "engine_winner": "OmniInfer",
        }
        with self.assertRaises(DRIVER.CampaignError):
            DRIVER.validate_gateway_machine_schema(
                nonauthored_nested, require_cleanup=True
            )


class MarkerSchemaTests(unittest.TestCase):
    def marker(self) -> dict[str, object]:
        marker = {field: None for field in DRIVER.GATEWAY_MARKER_FIELDS_V3}
        marker["sampling_state_at_marker_creation"] = {
            field: 0 for field in DRIVER.GATEWAY_MARKER_SAMPLING_STATE_FIELDS_V3
        }
        marker["pre_marker_admission"] = {
            field: True for field in DRIVER.GATEWAY_MARKER_ADMISSION_FIELDS_V3
        }
        marker["blocker_resolution"] = {
            "authored_formally_admitted": False,
            "authored_blocker_codes": list(DRIVER.GATEWAY_AUTHORED_BLOCKERS),
            "resolution_map": {
                blocker: {"resolved": False, "evidence": None}
                for blocker in (
                    *DRIVER.GATEWAY_AUTHORED_BLOCKERS,
                    "SAME_RESIDENT_BACKEND_AND_CONTROLLER_MODEL_FD_CUSTODY",
                )
            },
            "all_pre_marker_blockers_except_public_activation_resolved": True,
            "all_resolved": False,
            "authored_state_was_not_mutated": True,
        }
        return marker

    def test_marker_rejects_engine_claim_in_every_authored_schema(self) -> None:
        marker = self.marker()
        DRIVER.validate_gateway_marker_schema(marker)
        paths = (
            (),
            ("sampling_state_at_marker_creation",),
            ("pre_marker_admission",),
            ("blocker_resolution",),
            (
                "blocker_resolution",
                "resolution_map",
                DRIVER.GATEWAY_AUTHORED_BLOCKERS[0],
            ),
        )
        for path in paths:
            changed = copy.deepcopy(marker)
            target = changed
            for field in path:
                target = target[field]
            target.update({"engine_ranking": True, "engine_winner": "OmniInfer"})
            with self.subTest(path=path):
                with self.assertRaises(DRIVER.CampaignError):
                    DRIVER.validate_gateway_marker_schema(changed)

        runtime_nested = copy.deepcopy(marker)
        runtime_nested["runtime_preflight"] = {
            "engine_ranking": True,
            "engine_winner": "OmniInfer",
        }
        with self.assertRaises(DRIVER.CampaignError):
            DRIVER.validate_gateway_marker_schema(runtime_nested)


class CrashSafetyTests(unittest.TestCase):
    def test_downstream_failure_retains_exact_generation_response(self) -> None:
        class Connection:
            request_count = 0
            last_raw_response = b""

        backend = Connection()
        gateway = Connection()
        slot = {"arm": "B", "sequence_index": 7}

        def fail_after_generation() -> None:
            backend.request_count += 1
            backend.last_raw_response = b'{"exact":"wire bytes"}'
            raise DRIVER.CampaignError("segment custody failed")

        with self.assertRaises(DRIVER.RuntimeObservationError) as caught:
            DRIVER.collect_sample_with_failed_raw_retention(
                slot, backend, gateway, fail_after_generation
            )
        self.assertEqual(
            caught.exception.observation["raw_generation_response"],
            b'{"exact":"wire bytes"}',
        )

    def test_first_failure_stops_and_keeps_remaining_slots_unattempted(self) -> None:
        contract = fixture_contract()
        calls: list[int] = []

        def collect(slot: dict[str, object]) -> dict[str, object]:
            calls.append(slot["sequence_index"])
            if slot["sequence_index"] == 3:
                raise DRIVER.RuntimeObservationError(
                    "fixture crash", {"raw_response": b"broken"}
                )
            wall = 1000.0 if slot["arm"] == "B" else 1001.0
            return fixture_sample(slot, wall)

        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory) / "raw.json"
            record = DRIVER.execute_formal_schedule(
                contract,
                {"raw_output_path": str(raw)},
                {"fixture": True},
                sample_collector=collect,
                postflight_collector=lambda: {"passed": True},
            )
            self.assertEqual(DRIVER.parse_strict_json_line(raw.read_bytes()), record)
        self.assertEqual(calls, [0, 1, 2, 3])
        self.assertEqual(record["status"], "CONSUMED_FIRST_POST_MARKER_FAILURE")
        self.assertEqual(record["schedule_receipt"]["slots"][3]["status"], "failed")
        self.assertTrue(
            all(
                slot["status"] == "unattempted"
                for slot in record["schedule_receipt"]["slots"][4:]
            )
        )
        observation = record["failures"][0]["observation"]["raw_response"]
        self.assertEqual(observation["sha256"], hashlib.sha256(b"broken").hexdigest())

    def test_failed_preflight_never_creates_marker(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / "marker.json"
            with self.assertRaises(DRIVER.CampaignError):
                DRIVER.prepare_marker_after_preflight(
                    marker,
                    lambda: {
                        "format": DRIVER.MARKER_FORMAT,
                        "schema_version": 3,
                        "pre_marker_admission": {
                            "all_passed": False,
                            "blocker_codes": ["AMBIGUOUS_MACOS_RUNTIME_CLOSURE"],
                        },
                    },
                )
            self.assertFalse(marker.exists())

    def test_before_first_warmup_failure_attempts_no_slot(self) -> None:
        contract = fixture_contract()
        calls: list[int] = []
        with tempfile.TemporaryDirectory() as directory:
            record = DRIVER.execute_formal_schedule(
                contract,
                {"raw_output_path": str(Path(directory) / "raw.json")},
                {},
                sample_collector=lambda slot: calls.append(slot["sequence_index"]),
                postflight_collector=lambda: {"passed": True},
                before_first_slot=lambda: (_ for _ in ()).throw(
                    DRIVER.CampaignError("monitor start failed")
                ),
            )
        self.assertEqual(calls, [])
        self.assertEqual(record["schedule_receipt"]["attempted_count"], 0)
        self.assertTrue(
            all(
                slot["status"] == "unattempted"
                for slot in record["schedule_receipt"]["slots"]
            )
        )


class CustodyHardeningTests(unittest.TestCase):
    def test_published_runtime_preflight_rejects_same_ofd_claim_mutations(self) -> None:
        plan = fixture_execution_plan()
        marker_runtime = fixture_runtime_preflight(plan)
        runtime_now = copy.deepcopy(marker_runtime)
        DRIVER.validate_published_runtime_preflight(marker_runtime, runtime_now, plan)

        for field, value in (
            ("controller_and_backend_same_vnode_identity", False),
            ("same_open_file_description_not_claimed", False),
            ("controller_and_backend_same_open_file_description_claimed", True),
        ):
            changed = copy.deepcopy(marker_runtime)
            changed["controller_backend_model_fd_custody"][field] = value
            with self.subTest(field=field):
                with self.assertRaises(DRIVER.CampaignError):
                    DRIVER.validate_published_runtime_preflight(
                        changed, runtime_now, plan
                    )

        nested_claim = copy.deepcopy(marker_runtime)
        nested_claim["controller_backend_model_fd_custody"]["start"][
            "controller_and_backend_same_open_file_description_claimed"
        ] = True
        with self.assertRaises(DRIVER.CampaignError):
            DRIVER.validate_published_runtime_preflight(nested_claim, runtime_now, plan)

    def test_published_runtime_preflight_rejects_model_load_receipt_mutations(
        self,
    ) -> None:
        plan = fixture_execution_plan()
        runtime_now = fixture_runtime_preflight(plan)
        DRIVER.validate_published_runtime_preflight(runtime_now, runtime_now, plan)

        mutations = {
            "select-before-listener-ready": lambda receipt: receipt.update(
                {
                    "management_request_sequence": [
                        ["POST", "/omni/model/select"],
                        ["GET", "/omni/state"],
                    ]
                }
            ),
            "generation-request": lambda receipt: receipt.update(
                {"generation_requests": 1}
            ),
            "response-drift": lambda receipt: receipt["response"]["object"].update(
                {"engine_winner": "OmniInfer"}
            ),
        }
        for label, mutate in mutations.items():
            changed = copy.deepcopy(runtime_now)
            for receipt in (
                changed["zero_generation_model_load"],
                changed["custodian_binding"]["zero_generation_model_load"],
                changed["custodian_binding"]["ready_receipt"][
                    "zero_generation_model_load"
                ],
            ):
                mutate(receipt)
            with self.subTest(label=label):
                with self.assertRaises(DRIVER.CampaignError):
                    DRIVER.validate_published_runtime_preflight(
                        changed, runtime_now, plan
                    )

        changed = copy.deepcopy(runtime_now)
        for receipt in (
            changed["zero_generation_model_load"],
            changed["custodian_binding"]["zero_generation_model_load"],
            changed["custodian_binding"]["ready_receipt"]["zero_generation_model_load"],
        ):
            receipt["history_start"] = {"exists": False, "entries": [], "v": 2}
            receipt["history_end"] = copy.deepcopy(receipt["history_start"])
        with self.assertRaises(DRIVER.CampaignError):
            DRIVER.validate_published_runtime_preflight(changed, runtime_now, plan)

    def test_gateway_model_custody_requires_explicit_same_vnode_contract(self) -> None:
        contract = fixture_contract()
        edge = contract["comparison_graph"]["edges"][DRIVER.EDGE_ID]
        edge.update(
            {
                "members": ["B", "G"],
                "same_loaded_model_file_description_required": True,
            }
        )
        contract["runtime_custody"] = {}

        with self.assertRaises(DRIVER.PreflightBlockedError) as blocked:
            DRIVER.validate_frozen_gateway_contract(contract)
        self.assertEqual(
            blocked.exception.receipt["blocker_codes"],
            ["GATEWAY_MODEL_CUSTODY_CONTRACT_ALTERNATIVE_NOT_ACTIVATED"],
        )
        self.assertEqual(blocked.exception.receipt["generation_requests"], 0)
        self.assertFalse(blocked.exception.receipt["marker_created"])

        edge.update(copy.deepcopy(DRIVER.GATEWAY_MODEL_CUSTODY_EDGE_FIELDS_V3))
        contract["runtime_custody"]["gateway_controller_backend_model_custody_v3"] = (
            copy.deepcopy(DRIVER.GATEWAY_MODEL_CUSTODY_CONTRACT_V3)
        )
        DRIVER.validate_frozen_gateway_contract(contract)


class ZeroGenerationModelLoadTests(unittest.TestCase):
    def test_unloaded_gateway_listener_timeout_is_zero_generation_blocker(self) -> None:
        plan = fixture_execution_plan()
        plan["runtime"]["custodian_ready_timeout_seconds"] = 1
        gateway_start = {"seconds": 3, "microseconds": 4}
        calls: list[tuple[str, str]] = []

        class GatewayProcess:
            pid = 11

            @staticmethod
            def poll() -> None:
                return None

        def unavailable_request(
            _base_url: str,
            _label: str,
            method: str,
            path: str,
            _body: bytes | None,
            _validator: object,
        ) -> object:
            calls.append((method, path))
            raise OSError("listener absent")

        ticks = iter((0.0, 0.0, 1.0))
        with self.assertRaises(DRIVER.PreflightBlockedError) as blocked:
            DRIVER.admit_zero_generation_model_load(
                plan,
                GatewayProcess(),
                gateway_start,
                request_json=unavailable_request,
                process_start_reader=lambda _pid: copy.deepcopy(gateway_start),
                parent_pid_reader=lambda _pid: 11,
                tree_reader=lambda _path: {"exists": False, "entries": []},
                monotonic=lambda: next(ticks),
                sleeper=lambda _seconds: None,
            )
        self.assertEqual(
            blocked.exception.receipt["blocker_codes"],
            ["OMNIINFER_GATEWAY_LISTENER_NOT_READY_FOR_ZERO_GENERATION_MODEL_LOAD"],
        )
        self.assertEqual(blocked.exception.receipt["generation_requests"], 0)
        self.assertFalse(blocked.exception.receipt["marker_created"])
        self.assertTrue(calls)
        self.assertEqual(set(calls), {("GET", "/omni/state")})

    def test_model_select_success_binds_zero_generation_receipt(self) -> None:
        plan = fixture_execution_plan()
        gateway_start = {"seconds": 3, "microseconds": 4}
        backend_start = {"seconds": 5, "microseconds": 6}
        unloaded = fixture_unloaded_gateway_state()
        response = fixture_model_select_response(plan)
        loaded = fixture_loaded_gateway_state(plan)
        calls: list[tuple[str, str, bytes | None]] = []

        class GatewayProcess:
            pid = 11

            @staticmethod
            def poll() -> None:
                return None

        def request(
            _base_url: str,
            label: str,
            method: str,
            path: str,
            body: bytes | None,
            validator: object,
        ) -> tuple[dict[str, object], object, dict[str, object]]:
            calls.append((method, path, body))
            if len(calls) == 1:
                payload = unloaded
            elif len(calls) == 2:
                payload = response
            else:
                payload = loaded
            return (
                payload,
                validator(payload),
                fixture_management_transport(
                    _base_url, label, method, path, body, payload
                ),
            )

        history = {"exists": False, "entries": [], "canonical_sha256": "0" * 64}
        ticks = iter((0.0, 0.0, 1.2, 1.25, 1.3, 1.35))
        receipt = DRIVER.admit_zero_generation_model_load(
            plan,
            GatewayProcess(),
            gateway_start,
            request_json=request,
            process_start_reader=lambda pid: copy.deepcopy(
                gateway_start if pid == 11 else backend_start
            ),
            parent_pid_reader=lambda pid: 11 if pid == 12 else 10,
            tree_reader=lambda _path: copy.deepcopy(history),
            monotonic=lambda: next(ticks),
            sleeper=lambda _seconds: None,
        )

        self.assertEqual(
            [(method, path) for method, path, _body in calls],
            [
                ("GET", "/omni/state"),
                ("POST", "/omni/model/select"),
                ("GET", "/omni/state"),
            ],
        )
        self.assertEqual(calls[1][2], DRIVER.gateway_model_select_request_bytes(plan))
        self.assertEqual(receipt["generation_requests"], 0)
        self.assertEqual(receipt["generation_endpoint_paths_called"], [])
        self.assertEqual(receipt["model_select_request_count"], 1)
        self.assertEqual(
            receipt["request"]["object"], DRIVER.gateway_model_select_request(plan)
        )
        self.assertEqual(receipt["response"]["object"], response)
        self.assertEqual(receipt["loaded_state"]["validated"]["backend_pid"], 12)
        self.assertTrue(receipt["history_start_end_equal"])
        self.assertTrue(receipt["gateway_process_start_end_equal"])
        self.assertTrue(receipt["gateway_is_direct_parent_of_backend"])
        self.assertTrue(receipt["all_passed"])

    def test_model_select_response_drift_fails_closed_before_state_wait(self) -> None:
        plan = fixture_execution_plan()
        gateway_start = {"seconds": 3, "microseconds": 4}

        class GatewayProcess:
            pid = 11

            @staticmethod
            def poll() -> None:
                return None

        for mutation in ("extra-field", "already-loaded", "generation", "mmproj"):
            calls: list[tuple[str, str]] = []
            changed = fixture_model_select_response(plan)
            if mutation == "extra-field":
                changed["engine_winner"] = "OmniInfer"
            elif mutation == "already-loaded":
                changed["already_loaded"] = True
            elif mutation == "mmproj":
                changed["selected_mmproj"] = "/private/tmp/forbidden-mmproj.gguf"
            else:
                changed["generation"] = 2

            def request(
                _base_url: str,
                _label: str,
                method: str,
                path: str,
                _body: bytes | None,
                validator: object,
            ) -> tuple[dict[str, object], object, dict[str, object]]:
                calls.append((method, path))
                if len(calls) == 1:
                    payload = fixture_unloaded_gateway_state()
                    return (
                        payload,
                        validator(payload),
                        fixture_management_transport(
                            _base_url, _label, method, path, _body, payload
                        ),
                    )
                if len(calls) == 2:
                    admitted = fixture_model_select_response(plan)
                    return (
                        changed,
                        validator(admitted),
                        fixture_management_transport(
                            _base_url, _label, method, path, _body, changed
                        ),
                    )
                payload = fixture_loaded_gateway_state(plan)
                return (
                    payload,
                    validator(payload),
                    fixture_management_transport(
                        _base_url, _label, method, path, _body, payload
                    ),
                )

            with self.subTest(mutation=mutation):
                with self.assertRaises(DRIVER.ReceiptError):
                    DRIVER.admit_zero_generation_model_load(
                        plan,
                        GatewayProcess(),
                        gateway_start,
                        request_json=request,
                        process_start_reader=lambda _pid: copy.deepcopy(gateway_start),
                        parent_pid_reader=lambda _pid: 11,
                        tree_reader=lambda _path: {
                            "exists": False,
                            "entries": [],
                        },
                        monotonic=lambda: 0.0,
                        sleeper=lambda _seconds: None,
                    )
            self.assertEqual(
                calls,
                [("GET", "/omni/state"), ("POST", "/omni/model/select")],
            )
            self.assertNotIn(("POST", "/v1/chat/completions"), calls)

    def test_first_post_select_state_must_be_exact_ready(self) -> None:
        plan = fixture_execution_plan()
        plan["runtime"]["custodian_ready_timeout_seconds"] = 1
        gateway_start = {"seconds": 3, "microseconds": 4}
        response = fixture_model_select_response(plan)
        calls: list[tuple[str, str]] = []

        class GatewayProcess:
            pid = 11

            @staticmethod
            def poll() -> None:
                return None

        def request(
            _base_url: str,
            label: str,
            method: str,
            path: str,
            _body: bytes | None,
            validator: object,
        ) -> tuple[dict[str, object], object, dict[str, object]]:
            calls.append((method, path))
            payload = response if len(calls) == 2 else fixture_unloaded_gateway_state()
            return (
                payload,
                validator(payload),
                fixture_management_transport(
                    _base_url, label, method, path, _body, payload
                ),
            )

        history = {"exists": False, "entries": [], "canonical_sha256": "0" * 64}
        sleeps: list[float] = []
        with self.assertRaises(DRIVER.ReceiptError):
            DRIVER.admit_zero_generation_model_load(
                plan,
                GatewayProcess(),
                gateway_start,
                request_json=request,
                process_start_reader=lambda _pid: copy.deepcopy(gateway_start),
                parent_pid_reader=lambda _pid: 11,
                tree_reader=lambda _path: copy.deepcopy(history),
                monotonic=lambda: 0.0,
                sleeper=sleeps.append,
            )
        self.assertEqual(
            calls,
            [
                ("GET", "/omni/state"),
                ("POST", "/omni/model/select"),
                ("GET", "/omni/state"),
            ],
        )
        self.assertEqual(sleeps, [])

    def test_preload_state_wrong_model_runtime_or_schema_is_rejected(self) -> None:
        mutations = {
            "model": lambda state: state.update({"model": DRIVER.MODEL_PATH}),
            "runtime": lambda state: state.update({"runtime": {"pid": 12}}),
            "resource-ledger": lambda state: state.update({"resource_ledger": {}}),
            "extra": lambda state: state.update({"engine_winner": "OmniInfer"}),
        }
        for label, mutate in mutations.items():
            state = fixture_unloaded_gateway_state()
            mutate(state)
            with self.subTest(label=label):
                with self.assertRaises(DRIVER.ReceiptError):
                    DRIVER.validate_gateway_preload_state(state)

    def test_loaded_state_core_and_nested_drift_is_rejected(self) -> None:
        plan = fixture_execution_plan()
        mutations = {
            "owner": lambda state: state.update({"owner_admin_id": None}),
            "generation": lambda state: state.update({"generation": 2}),
            "restore": lambda state: state.update({"restore_status": "pending"}),
            "runtime": lambda state: state["runtime"].update({"mode": "wrong"}),
            "loaded-model": lambda state: state["loaded_models"][0].update(
                {"route_state": "draining"}
            ),
            "resource-ledger": lambda state: state["resource_ledger"].update(
                {"committed_bytes": {}}
            ),
            "extra": lambda state: state.update({"engine_ranking": True}),
        }
        for label, mutate in mutations.items():
            state = fixture_loaded_gateway_state(plan)
            mutate(state)
            with self.subTest(label=label):
                with self.assertRaises(DRIVER.ReceiptError):
                    DRIVER.validate_gateway_state(state, plan)

    def test_gateway_parent_change_during_model_load_fails_closed(self) -> None:
        plan = fixture_execution_plan()
        gateway_start = {"seconds": 3, "microseconds": 4}
        backend_start = {"seconds": 5, "microseconds": 6}
        response = fixture_model_select_response(plan)
        payloads = iter(
            (
                fixture_unloaded_gateway_state(),
                response,
                fixture_loaded_gateway_state(plan),
            )
        )

        class GatewayProcess:
            pid = 11

            @staticmethod
            def poll() -> None:
                return None

        def request(
            _base_url: str,
            _label: str,
            _method: str,
            _path: str,
            _body: bytes | None,
            validator: object,
        ) -> tuple[dict[str, object], object, dict[str, object]]:
            payload = next(payloads)
            return (
                payload,
                validator(payload),
                fixture_management_transport(
                    _base_url, _label, _method, _path, _body, payload
                ),
            )

        gateway_parents = iter((10, 10, 99))

        def parent(pid: int) -> int:
            return next(gateway_parents) if pid == 11 else 11

        with self.assertRaises(DRIVER.CampaignError):
            DRIVER.admit_zero_generation_model_load(
                plan,
                GatewayProcess(),
                gateway_start,
                expected_gateway_parent_pid=10,
                request_json=request,
                process_start_reader=lambda pid: copy.deepcopy(
                    gateway_start if pid == 11 else backend_start
                ),
                parent_pid_reader=parent,
                tree_reader=lambda _path: {"exists": False, "entries": []},
                monotonic=lambda: 0.0,
                sleeper=lambda _seconds: None,
            )

    def test_management_transport_drift_fails_closed(self) -> None:
        plan = fixture_execution_plan()
        gateway_start = {"seconds": 3, "microseconds": 4}
        backend_start = {"seconds": 5, "microseconds": 6}
        payloads = iter(
            (
                fixture_unloaded_gateway_state(),
                fixture_model_select_response(plan),
                fixture_loaded_gateway_state(plan),
            )
        )

        class GatewayProcess:
            pid = 11

            @staticmethod
            def poll() -> None:
                return None

        def request(
            _base_url: str,
            _label: str,
            _method: str,
            _path: str,
            _body: bytes | None,
            validator: object,
        ) -> tuple[dict[str, object], object, dict[str, object]]:
            payload = next(payloads)
            return payload, validator(payload), {"passed": True}

        with self.assertRaises(DRIVER.ReceiptError):
            DRIVER.admit_zero_generation_model_load(
                plan,
                GatewayProcess(),
                gateway_start,
                request_json=request,
                process_start_reader=lambda pid: copy.deepcopy(
                    gateway_start if pid == 11 else backend_start
                ),
                parent_pid_reader=lambda pid: 11 if pid == 12 else 10,
                tree_reader=lambda _path: {"exists": False, "entries": []},
                monotonic=lambda: 0.0,
                sleeper=lambda _seconds: None,
            )

    def test_shared_formal_helper_is_tracked_and_marker_bound(self) -> None:
        paths = DRIVER.tracked_campaign_paths(
            {
                "contract_repository_path": "contract.json",
                "validator_repository_path": "validator.py",
                "driver_repository_path": DRIVER.DRIVER_REPOSITORY_PATH,
                "plan_repository_path": "plan.json",
                "marker_repository_path": "marker.json",
            },
            include_marker=True,
        )
        self.assertEqual(
            paths["shared_formal_driver"], DRIVER.SHARED_DRIVER_REPOSITORY_PATH
        )
        digest = "a" * 64
        marker = {
            "shared_formal_driver_repository_path": DRIVER.SHARED_DRIVER_REPOSITORY_PATH,
            "shared_formal_driver_blob_sha256": digest,
        }
        tracked = {"shared_formal_driver": {"blob_sha256": digest}}
        runtime = {
            "custodian_process_start": {
                "shared_formal_driver_source": {
                    "absolute_path": str(
                        DRIVER._NATIVE_DRIVER_PATH.resolve(strict=True)
                    ),
                    "sha256": digest,
                }
            }
        }
        DRIVER.validate_shared_formal_driver_binding(marker, tracked, runtime)
        changed = copy.deepcopy(marker)
        changed["shared_formal_driver_blob_sha256"] = "b" * 64
        with self.assertRaises(DRIVER.CampaignError):
            DRIVER.validate_shared_formal_driver_binding(changed, tracked, runtime)

    @unittest.skipUnless(platform.system() == "Darwin", "Darwin libproc only")
    def test_libproc_vnode_layout_observes_own_fixture_fd(self) -> None:
        payload = b"libproc-layout-fixture"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.bin"
            path.write_bytes(payload)
            fd = os.open(
                path,
                os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC,
            )
            try:
                matches = [
                    entry
                    for entry in DRIVER.proc_vnode_fd_entries(os.getpid())
                    if entry["fd"] == fd
                ]
            finally:
                os.close(fd)
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0]["size_bytes"], len(payload))
        self.assertTrue(matches[0]["close_on_exec"])
        self.assertEqual(matches[0]["open_flags"] & 3, 1)

    def test_plan_requires_driver_owned_custodian_lifecycle_fields(self) -> None:
        contract = json.loads(
            (ROOT / DRIVER.CONTRACT_REPOSITORY_PATH).read_text(encoding="utf-8")
        )
        contract["comparison_graph"]["edges"][DRIVER.EDGE_ID].update(
            copy.deepcopy(DRIVER.GATEWAY_MODEL_CUSTODY_EDGE_FIELDS_V3)
        )
        contract["runtime_custody"]["gateway_controller_backend_model_custody_v3"] = (
            copy.deepcopy(DRIVER.GATEWAY_MODEL_CUSTODY_CONTRACT_V3)
        )
        plan = fixture_execution_plan()
        admitted = DRIVER.validate_execution_plan(plan, contract)
        self.assertEqual(
            admitted["runtime"]["custodian_control_socket_path"],
            "/private/tmp/gateway-formal-v3-campaign-fixture/state/.local/logs/custodian.sock",
        )
        for mutation in (
            "old-pointer",
            "history-enabled",
            "socket-outside",
            "port-mismatch",
            "extra-argv",
            "wrong-launcher",
            "extra-environment",
            "wrong-lang",
        ):
            changed = copy.deepcopy(plan)
            if mutation == "old-pointer":
                changed["runtime"]["instrumented_model_fd_state_pointer"] = (
                    "/runtime/model_fd_custody"
                )
            elif mutation == "history-enabled":
                changed["runtime"]["expected_gateway_environment"][
                    "OMNIINFER_REQUEST_HISTORY"
                ] = "1"
            elif mutation == "socket-outside":
                changed["runtime"]["custodian_control_socket_path"] = (
                    "/private/tmp/outside.sock"
                )
            elif mutation == "port-mismatch":
                changed["runtime"]["omni_base_url"] = "http://127.0.0.1:19002"
            elif mutation == "extra-argv":
                changed["runtime"]["expected_gateway_argv"].append("--verbose")
            elif mutation == "wrong-launcher":
                changed["runtime"]["expected_gateway_environment"][
                    "OMNIINFER_LLAMA_CPP_MAC_LAUNCHER_PATH"
                ] = "/bin/false"
            elif mutation == "extra-environment":
                changed["runtime"]["expected_gateway_environment"]["EXTRA"] = "1"
            else:
                changed["runtime"]["expected_gateway_environment"]["LANG"] = "en_US"
            with self.subTest(mutation=mutation):
                with self.assertRaises(DRIVER.CampaignError):
                    DRIVER.validate_execution_plan(changed, contract)

    def test_custodian_ready_binding_rejects_launch_order_mutation(self) -> None:
        lifecycle = {
            "controller_fd_open_started_monotonic_ns": 10,
            "controller_fd_custody_complete_monotonic_ns": 20,
            "gateway_spawn_invocation_monotonic_ns": 30,
            "gateway_kernel_identity_observed_monotonic_ns": 40,
            "backend_kernel_identity_observed_monotonic_ns": 50,
            "custodian_start_identity": {"seconds": 1, "microseconds": 2},
        }
        ready = {
            "format": DRIVER.CUSTODIAN_READY_FORMAT,
            "schema_version": 3,
            "edge_id": DRIVER.EDGE_ID,
            "passed": True,
            "generation_requests": 0,
            "nonce": "a" * 64,
            "custodian_pid": 10,
            "custodian_start_identity": {"seconds": 1, "microseconds": 2},
            "gateway_pid": 11,
            "gateway_start_identity": {"seconds": 3, "microseconds": 4},
            "backend_pid": 12,
            "backend_start_identity": {"seconds": 5, "microseconds": 6},
            "controller_preload_fd": {"fd": 7},
            "backend_loaded_fd": {"fd": 8},
            "lifecycle_sequence": lifecycle,
            "control_socket": {"absolute_path": "/private/tmp/c.sock"},
            "zero_generation_model_load": fixture_zero_generation_model_load_receipt(),
        }
        plan = fixture_execution_plan()
        binding = DRIVER._custodian_binding_from_ready(ready, DRIVER_PATH, plan)
        self.assertEqual(binding["nonce"], "a" * 64)
        self.assertEqual(
            binding["zero_generation_model_load"], ready["zero_generation_model_load"]
        )
        changed = copy.deepcopy(ready)
        changed["lifecycle_sequence"]["gateway_spawn_invocation_monotonic_ns"] = 15
        with self.assertRaises(DRIVER.CampaignError):
            DRIVER._custodian_binding_from_ready(changed, DRIVER_PATH, plan)
        changed = copy.deepcopy(ready)
        changed["zero_generation_model_load"]["generation_requests"] = 1
        with self.assertRaises(DRIVER.CampaignError):
            DRIVER._custodian_binding_from_ready(changed, DRIVER_PATH, plan)

    def test_custodian_attestation_stability_rejects_backend_fd_mutation(self) -> None:
        controller = {
            "format": "apxinf-controller-held-model-fd-v3",
            "schema_version": 3,
            "stage": "start",
            "fd": 7,
        }
        backend = {"fd": 8, "device": 1, "inode": 2}
        daemon = {
            "custodian_pid": 10,
            "custodian_start_identity": {"seconds": 1, "microseconds": 2},
            "gateway_pid": 11,
            "gateway_start_identity": {"seconds": 3, "microseconds": 4},
            "backend_pid": 12,
            "backend_start_identity": {"seconds": 5, "microseconds": 6},
            "lifecycle_sequence": {"fixed": True},
            "controller_preload_fd": controller,
            "backend_loaded_fd": backend,
        }
        start = {"daemon_challenge_response": daemon}
        end = copy.deepcopy(start)
        end["daemon_challenge_response"]["controller_preload_fd"]["stage"] = "end"
        DRIVER.require_same_custodian_attestation(start, end)
        end["daemon_challenge_response"]["backend_loaded_fd"]["fd"] = 9
        with self.assertRaises(DRIVER.CampaignError):
            DRIVER.require_same_custodian_attestation(start, end)

    def test_fd_checkpoint_schedule_requires_warmup_every_macroblock_and_postflight(
        self,
    ) -> None:
        counter = iter(range(1, 23))

        def checkpoint(stage: str) -> dict[str, str]:
            return {"stage": stage, "challenge": f"{next(counter):064x}"}

        initial = checkpoint("prepare-before-any-generation")
        start = checkpoint("runtime-preflight-start")
        preflight_end = checkpoint("runtime-preflight-end")
        before = checkpoint("before-first-generation")
        final = checkpoint("runtime-postflight-before-cleanup")
        samples: list[dict[str, object]] = [{} for _ in range(136)]
        expected = {7: "after-warmups"}
        expected.update(
            {15 + index * 8: f"after-timed-macroblock-{index}" for index in range(16)}
        )
        for index, stage in expected.items():
            samples[index] = {
                "segment_state_after": {
                    "outside_primary_timed_interval": True,
                    "controller_backend_fd_custody": checkpoint(stage),
                }
            }
        receipt = {
            "parity_admission": {
                "custodian_binding": {
                    "nonce": "a" * 64,
                    "initial_external_attestation": initial,
                },
                "controller_backend_model_fd_custody": {
                    "start": start,
                    "end": preflight_end,
                },
            },
            "postflight": {
                "before_first_generation_custody": before,
                "runtime_postflight": {
                    "controller_backend_model_fd_custody_end": final
                },
            },
            "samples": samples,
        }

        def validate(attestation: dict[str, str], stage: str, _nonce: str) -> str:
            self.assertEqual(attestation["stage"], stage)
            return attestation["challenge"]

        with (
            mock.patch.object(
                DRIVER, "validate_custodian_attestation_receipt", side_effect=validate
            ),
            mock.patch.object(DRIVER, "require_same_custodian_attestation"),
        ):
            DRIVER.validate_model_fd_checkpoint_schedule(receipt)
            changed = copy.deepcopy(receipt)
            del changed["samples"][15]["segment_state_after"]
            with self.assertRaises(DRIVER.CampaignError):
                DRIVER.validate_model_fd_checkpoint_schedule(changed)

    def test_cleanup_is_requested_only_after_raw_receipt_exists(self) -> None:
        record = {
            "status": "FORMAL_COMPLETE",
            "failures": [],
            "gates": {},
            "decision": {},
        }
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory) / "raw.json"
            with mock.patch.object(DRIVER, "shutdown_custodian") as shutdown:
                with self.assertRaises(DRIVER.CampaignError):
                    DRIVER._attach_custodian_cleanup_after_raw(
                        record,
                        raw,
                        {},
                        {"nonce": "a" * 64},
                        "fixture-cleanup",
                    )
                shutdown.assert_not_called()
            DRIVER._SHARED.atomic_create_json(raw, record)

            def cleanup(*_args: object) -> dict[str, object]:
                self.assertTrue(raw.exists())
                return {
                    "format": DRIVER.CUSTODIAN_CLEANUP_FORMAT,
                    "passed": True,
                }

            with mock.patch.object(DRIVER, "shutdown_custodian", side_effect=cleanup):
                updated = DRIVER._attach_custodian_cleanup_after_raw(
                    record,
                    raw,
                    {},
                    {"nonce": "a" * 64},
                    "fixture-cleanup",
                )
            self.assertTrue(updated["custodian_cleanup"]["passed"])
            self.assertEqual(DRIVER.parse_strict_json_line(raw.read_bytes()), updated)

    def test_raw_cleanup_rpc_failure_uses_marker_bound_exact_pid_fallback(
        self,
    ) -> None:
        record = {
            "status": "FORMAL_COMPLETE",
            "failures": [],
            "gates": {},
            "decision": {},
        }
        fallback_receipt = {
            "format": "apxinf-omniinfer-gateway-exact-pid-fallback-cleanup-v3",
            "passed": True,
        }
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory) / "raw.json"
            DRIVER._SHARED.atomic_create_json(raw, record)
            with (
                mock.patch.object(
                    DRIVER, "shutdown_custodian", side_effect=OSError("rpc failed")
                ),
                mock.patch.object(
                    DRIVER,
                    "_fallback_cleanup_marker_bound_processes",
                    return_value=fallback_receipt,
                ) as fallback,
            ):
                updated = DRIVER._attach_custodian_cleanup_after_raw(
                    record,
                    raw,
                    fixture_execution_plan(),
                    {"nonce": "a" * 64},
                    "fixture-cleanup",
                )
            fallback.assert_called_once()
            self.assertEqual(
                updated["custodian_cleanup"]["exact_pid_fallback"],
                fallback_receipt,
            )
            self.assertEqual(DRIVER.parse_strict_json_line(raw.read_bytes()), updated)

    def test_fallback_never_signals_reused_marker_bound_pids(self) -> None:
        plan = fixture_execution_plan()
        binding = {
            "backend_pid": 12,
            "backend_start_identity": {"seconds": 1, "microseconds": 1},
            "gateway_pid": 11,
            "gateway_start_identity": {"seconds": 2, "microseconds": 2},
            "custodian_pid": 10,
            "custodian_start_identity": {"seconds": 3, "microseconds": 3},
            "control_socket": {"fixture": True},
        }
        with (
            mock.patch.object(
                DRIVER,
                "process_start_identity",
                return_value={"seconds": 999, "microseconds": 999},
            ),
            mock.patch.object(DRIVER.os, "kill") as kill,
            mock.patch.object(DRIVER.os, "killpg") as killpg,
            mock.patch.object(DRIVER, "_require_loopback_listener_absent"),
        ):
            receipt = DRIVER._fallback_cleanup_marker_bound_processes(
                plan, binding, "fixture-reuse"
            )
        kill.assert_not_called()
        killpg.assert_not_called()
        self.assertTrue(receipt["passed"])
        self.assertTrue(
            all(
                step["state"] == "original-exited-pid-reused-no-signal"
                for step in receipt["process_steps"]
            )
        )

    def test_command_runner_does_not_inherit_ambient_git_configuration(self) -> None:
        completed = type(
            "Completed",
            (),
            {"returncode": 0, "stdout": b"", "stderr": b""},
        )()
        with mock.patch.object(DRIVER.subprocess, "run", return_value=completed) as run:
            with self.assertRaises(DRIVER.CampaignError):
                DRIVER.hardened_command_runner(
                    ["/usr/bin/git", "status"], Path("/"), 1.0, {"EVIL": "1"}
                )
            DRIVER.hardened_command_runner(
                ["/usr/bin/git", "status"],
                Path("/"),
                1.0,
                DRIVER._SHARED.git_custody_environment(),
            )
        environment = run.call_args.kwargs["env"]
        self.assertEqual(environment, DRIVER._SHARED.git_custody_environment())
        self.assertNotIn("EVIL", environment)
        self.assertEqual(environment["GIT_CONFIG_GLOBAL"], "/dev/null")
        self.assertEqual(environment["GIT_CONFIG_SYSTEM"], "/dev/null")

    def test_actual_host_window_and_power_are_fail_closed(self) -> None:
        contract = {
            "host_quiet_gate": {"continuous_monitor": {"sample_interval_ms": 250}}
        }
        base = {
            "snapshots": [
                {
                    "cpu_window_start_monotonic_ns": 1_000_000_000,
                    "monotonic_ns": 1_250_000_000,
                    "cpu_percent_window_ms": 250.0,
                    "system_state_matches_gate_start": True,
                }
            ]
        }
        DRIVER.validate_actual_host_windows(base, contract)
        for field, value in (
            ("monotonic_ns", 1_249_999_999),
            ("cpu_percent_window_ms", 250.000001),
            ("system_state_matches_gate_start", False),
        ):
            changed = copy.deepcopy(base)
            changed["snapshots"][0][field] = value
            with self.subTest(field=field):
                with self.assertRaises(DRIVER.CampaignError):
                    DRIVER.validate_actual_host_windows(changed, contract)

    def test_actual_host_windows_must_be_contiguous(self) -> None:
        contract = {
            "host_quiet_gate": {"continuous_monitor": {"sample_interval_ms": 250}}
        }
        receipt = {
            "snapshots": [
                {
                    "cpu_window_start_monotonic_ns": 1_000_000_000,
                    "monotonic_ns": 1_250_000_000,
                    "cpu_percent_window_ms": 250.0,
                    "system_state_matches_gate_start": True,
                },
                {
                    "cpu_window_start_monotonic_ns": 1_250_000_001,
                    "monotonic_ns": 1_500_000_001,
                    "cpu_percent_window_ms": 250.0,
                    "system_state_matches_gate_start": True,
                },
            ]
        }
        with self.assertRaises(DRIVER.CampaignError):
            DRIVER.validate_actual_host_windows(receipt, contract)

    def test_monitor_rejects_pid_reuse_between_snapshots(self) -> None:
        monitor = object.__new__(DRIVER.GatewayContinuousHostMonitor)
        monitor._failure = None
        monitor.runtime_identity = {
            "custodian_binding": {
                "custodian_pid": 9,
                "custodian_start_identity": {"seconds": 0, "microseconds": 1},
            },
            "custodian_process_start": {
                "custodian_process": {"runtime_closure_sha256": "0" * 64}
            },
            "gateway_process_start": {
                "pid": 10,
                "process_start_identity": {"seconds": 1, "microseconds": 2},
                "runtime_closure_sha256": "a" * 64,
            },
            "backend_process_start": {
                "pid": 11,
                "process_start_identity": {"seconds": 3, "microseconds": 4},
                "runtime_closure_sha256": "b" * 64,
            },
        }
        monitor.process_start_reader = lambda _pid: {
            "seconds": 999,
            "microseconds": 0,
        }
        with self.assertRaises(DRIVER.CampaignError):
            monitor.assert_healthy()

    def test_exact_backend_and_custodian_group_cleanup_are_fail_safe(self) -> None:
        start = {"seconds": 1, "microseconds": 2}
        absent = DRIVER.PreflightBlockedError(
            {
                "blocker_codes": ["fixture-absent"],
                "format": "fixture",
            }
        )
        with (
            mock.patch.object(
                DRIVER, "process_start_identity", side_effect=[start, absent]
            ),
            mock.patch.object(DRIVER, "_pid_exists", return_value=False),
            mock.patch.object(DRIVER.os, "kill") as kill,
        ):
            result = DRIVER._terminate_exact_managed_pid(42, start, 1.0, "backend")
        kill.assert_called_once_with(42, DRIVER.signal.SIGTERM)
        self.assertEqual(result["state"], "terminated")

        process = mock.Mock()
        process.pid = 55
        process.poll.return_value = None
        process.wait.return_value = 0
        with (
            mock.patch.object(DRIVER, "process_start_identity", return_value=start),
            mock.patch.object(DRIVER.os, "getpgid", return_value=55),
            mock.patch.object(DRIVER.os, "getsid", return_value=55),
            mock.patch.object(
                DRIVER, "_custodian_group_exists", side_effect=[True, False]
            ),
            mock.patch.object(DRIVER.os, "killpg") as killpg,
        ):
            group = DRIVER._terminate_custodian_process_group(process, start, 1.0)
        killpg.assert_called_once_with(55, DRIVER.signal.SIGTERM)
        self.assertEqual(group["state"], "terminated")

    def test_exited_custodian_leader_never_signals_a_reused_process_group(
        self,
    ) -> None:
        process = mock.Mock()
        process.pid = 55
        process.poll.return_value = 0
        with (
            mock.patch.object(DRIVER, "_custodian_group_exists", return_value=True),
            mock.patch.object(DRIVER, "process_start_identity") as identity,
            mock.patch.object(DRIVER.os, "killpg") as killpg,
        ):
            with self.assertRaises(DRIVER.CampaignError):
                DRIVER._terminate_custodian_process_group(
                    process, {"seconds": 1, "microseconds": 2}, 1.0
                )
        identity.assert_not_called()
        killpg.assert_not_called()

    def test_unattested_custodian_start_never_signals_a_process_group(self) -> None:
        process = mock.Mock()
        process.pid = 55
        process.poll.return_value = 0
        process.wait.return_value = 0
        with (
            mock.patch.object(DRIVER, "_custodian_group_exists", return_value=True),
            mock.patch.object(DRIVER.os, "killpg") as killpg,
        ):
            with self.assertRaises(DRIVER.CampaignError):
                DRIVER._cleanup_unattested_custodian_start(process, 1.0)
        killpg.assert_not_called()

    def test_gateway_backend_cleanup_uses_one_total_deadline(self) -> None:
        class Clock:
            value = 0.0

            def monotonic(self) -> float:
                return self.value

            def advance(self, seconds: float) -> None:
                self.value += seconds

        class Process:
            pid = 55

            def __init__(self, clock: Clock) -> None:
                self.clock = clock
                self.waits: list[float] = []
                self.wait_count = 0

            def terminate(self) -> None:
                pass

            def kill(self) -> None:
                pass

            def wait(self, timeout: float) -> int:
                self.waits.append(timeout)
                self.clock.advance(timeout)
                self.wait_count += 1
                if self.wait_count == 1:
                    raise DRIVER.subprocess.TimeoutExpired(["gateway"], timeout)
                return -9

        clock = Clock()
        process = Process(clock)
        gateway_start = {"seconds": 1, "microseconds": 2}
        absent = DRIVER.PreflightBlockedError(
            {"blocker_codes": ["fixture-absent"], "format": "fixture"}
        )

        def identity(pid: int) -> dict[str, int]:
            if pid == process.pid:
                return gateway_start
            raise absent

        with (
            mock.patch.object(DRIVER.time, "monotonic", side_effect=clock.monotonic),
            mock.patch.object(DRIVER, "process_start_identity", side_effect=identity),
        ):
            receipt = DRIVER._terminate_gateway_child(
                process,
                gateway_start,
                77,
                {"seconds": 3, "microseconds": 4},
                20.0,
            )
        self.assertTrue(receipt["forced_kill_required"])
        self.assertLessEqual(sum(process.waits), 20.0)

    def test_control_request_read_has_one_total_deadline(self) -> None:
        class TricklingSocket:
            def __init__(self) -> None:
                self.timeouts: list[float] = []
                self.chunks = iter((b'{"command":', b'"attest"', b""))

            def settimeout(self, timeout: float) -> None:
                self.timeouts.append(timeout)

            def recv(self, _size: int) -> bytes:
                return next(self.chunks)

        connection = TricklingSocket()
        ticks = iter((0.0, 0.1, 0.6, 1.1))
        with self.assertRaises(DRIVER.CampaignError):
            DRIVER._receive_control_request(
                connection, 1.0, monotonic=lambda: next(ticks)
            )
        self.assertGreaterEqual(len(connection.timeouts), 2)
        self.assertGreater(connection.timeouts[0], connection.timeouts[1])

    def test_custodian_parent_ready_budget_strictly_covers_child_load_budget(
        self,
    ) -> None:
        runtime = fixture_execution_plan()["runtime"]
        child = DRIVER._custodian_child_ready_budget_seconds(runtime)
        parent = DRIVER._custodian_parent_ready_budget_seconds(runtime)
        self.assertEqual(child, 390.0)
        self.assertGreaterEqual(parent - child, 30.0)

    def test_long_listener_and_synchronous_load_fit_one_proven_child_budget(
        self,
    ) -> None:
        plan = fixture_execution_plan()
        gateway_start = {"seconds": 3, "microseconds": 4}
        backend_start = {"seconds": 5, "microseconds": 6}
        history = {"exists": False, "entries": [], "canonical_sha256": "0" * 64}

        class Clock:
            value = 0.0

            def monotonic(self) -> float:
                return self.value

        class GatewayProcess:
            pid = 11

            @staticmethod
            def poll() -> None:
                return None

        clock = Clock()
        calls = 0

        def request(
            base_url: str,
            label: str,
            method: str,
            path: str,
            body: bytes | None,
            validator: object,
        ) -> tuple[dict[str, object], object, dict[str, object]]:
            nonlocal calls
            calls += 1
            if calls == 1:
                clock.value = 179.0
                raise OSError("listener still starting")
            if calls == 2:
                payload = fixture_unloaded_gateway_state()
            elif calls == 3:
                clock.value += 180.0
                payload = fixture_model_select_response(plan)
            else:
                clock.value += 30.0
                payload = fixture_loaded_gateway_state(plan)
            return (
                payload,
                validator(payload),
                fixture_management_transport(
                    base_url, label, method, path, body, payload
                ),
            )

        receipt = DRIVER.admit_zero_generation_model_load(
            plan,
            GatewayProcess(),
            gateway_start,
            request_json=request,
            process_start_reader=lambda pid: copy.deepcopy(
                gateway_start if pid == 11 else backend_start
            ),
            parent_pid_reader=lambda pid: 11 if pid == 12 else 10,
            tree_reader=lambda _path: copy.deepcopy(history),
            monotonic=clock.monotonic,
            sleeper=lambda _seconds: None,
        )
        self.assertEqual(clock.value, 389.0)
        self.assertTrue(receipt["all_passed"])

    def test_runtime_closure_binds_lsof_text_vnode_to_read_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            executable = Path(directory) / "runner"
            executable.write_bytes(b"exact mapped image")
            observed = executable.stat()
            entry = {
                "fd": "txt",
                "type": "REG",
                "path": str(executable),
                "device_text": hex(observed.st_dev),
                "inode_text": str(observed.st_ino),
                "size_text": str(observed.st_size),
            }
            start = {"seconds": 1, "microseconds": 2}
            with (
                mock.patch.object(DRIVER, "lsof_entries", return_value=[entry]),
                mock.patch.object(DRIVER, "process_start_identity", return_value=start),
                mock.patch.object(
                    DRIVER,
                    "code_signature_receipt",
                    return_value={"state": "explicitly-unsigned"},
                ),
            ):
                closure = DRIVER.runtime_closure(42, str(executable))
            image = closure["loaded_image_paths_and_sha256"][0]
            self.assertEqual(
                image["lsof_text_vnode"],
                {
                    "device": observed.st_dev,
                    "inode": observed.st_ino,
                    "size_bytes": observed.st_size,
                },
            )
            self.assertEqual(closure["process_start_identity_before"], start)
            self.assertEqual(closure["process_start_identity_after"], start)

            for field, mutation in (
                ("device_text", hex(observed.st_dev + 1)),
                ("inode_text", str(observed.st_ino + 1)),
                ("size_text", str(observed.st_size + 1)),
            ):
                changed = copy.deepcopy(entry)
                changed[field] = mutation
                with (
                    self.subTest(field=field),
                    mock.patch.object(DRIVER, "lsof_entries", return_value=[changed]),
                    mock.patch.object(
                        DRIVER, "process_start_identity", return_value=start
                    ),
                    mock.patch.object(
                        DRIVER,
                        "code_signature_receipt",
                        return_value={"state": "explicitly-unsigned"},
                    ),
                ):
                    with self.assertRaises(DRIVER.CampaignError):
                        DRIVER.runtime_closure(42, str(executable))

    def test_host_snapshot_rejects_incomplete_managed_runtime_swap_proof(self) -> None:
        probe = object.__new__(DRIVER.GatewayQuietHostProbe)
        identities = {
            9: {"seconds": 1, "microseconds": 1},
            10: {"seconds": 2, "microseconds": 2},
            11: {"seconds": 3, "microseconds": 3},
        }
        probe.process_start_reader = lambda pid: identities[pid]
        active = {
            "custodian_pid": 9,
            "custodian_process_start_identity": identities[9],
            "gateway_pid": 10,
            "gateway_process_start_identity": identities[10],
            "backend_pid": 11,
            "backend_process_start_identity": identities[11],
        }
        base = {
            "active_runtime_root_present": True,
            "active_runtime_swap_proof_complete": False,
            "campaign_swap_probe_vanished_processes": [],
            "passed": True,
        }
        with mock.patch.object(
            DRIVER._SHARED.MacQuietHostProbe,
            "snapshot",
            return_value=base,
        ):
            receipt = DRIVER.GatewayQuietHostProbe.snapshot(probe, 0, active)
        self.assertFalse(receipt["managed_runtime_swap_proof_complete"])
        self.assertFalse(receipt["passed"])


class CampaignDirectoryInitializationTests(unittest.TestCase):
    def test_fresh_campaign_tree_is_created_before_runtime_with_exact_custody(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(dir="/private/tmp") as directory:
            plan = fixture_campaign_directory_plan(Path(directory))
            campaign_root = Path(directory) / "campaign"
            self.assertFalse(campaign_root.exists())

            receipt = DRIVER.initialize_campaign_directory_tree(plan)

            expected = [Path(path) for path in receipt["expected_directory_paths"]]
            self.assertEqual(
                set(receipt["created_directory_paths"]), set(map(str, expected))
            )
            self.assertEqual(receipt["preexisting_directory_paths"], [])
            self.assertTrue(receipt["all_passed"])
            self.assertEqual(receipt["generation_requests"], 0)
            self.assertEqual(receipt["runtime_processes_started"], 0)
            self.assertFalse(Path(plan["raw_output_path"]).exists())
            for path in expected:
                self.assertTrue(path.is_dir())
                self.assertEqual(path.stat().st_mode & 0o777, 0o700)
                self.assertEqual(path.stat().st_uid, os.geteuid())
            DRIVER.validate_campaign_directory_initialization_receipt(
                receipt, plan, verify_live=True
            )

    def test_unsafe_preexisting_campaign_entries_are_rejected(self) -> None:
        for mutation in (
            "root-symlink",
            "expected-nondirectory",
            "unexpected-file",
            "unexpected-directory",
            "unsafe-permissions",
        ):
            with self.subTest(mutation=mutation):
                with tempfile.TemporaryDirectory(dir="/private/tmp") as directory:
                    parent = Path(directory)
                    plan = fixture_campaign_directory_plan(parent)
                    root = parent / "campaign"
                    if mutation == "root-symlink":
                        target = parent / "target"
                        target.mkdir(mode=0o700)
                        root.symlink_to(target, target_is_directory=True)
                    else:
                        root.mkdir(mode=0o700)
                        if mutation == "expected-nondirectory":
                            (root / "state").write_text(
                                "not a directory", encoding="utf-8"
                            )
                        elif mutation == "unsafe-permissions":
                            root.chmod(0o755)
                        elif mutation == "unexpected-file":
                            unexpected = root / "unexpected"
                            unexpected.write_text("not admitted", encoding="utf-8")
                        else:
                            unexpected = root / "unexpected"
                            unexpected.mkdir(mode=0o700)

                    with self.assertRaises(DRIVER.CampaignError):
                        DRIVER.initialize_campaign_directory_tree(plan)
                    self.assertFalse(Path(plan["raw_output_path"]).exists())

    def test_exact_empty_partial_and_complete_tree_are_retryable(self) -> None:
        with tempfile.TemporaryDirectory(dir="/private/tmp") as directory:
            plan = fixture_campaign_directory_plan(Path(directory))
            root = Path(directory) / "campaign"
            state = root / "state"
            root.mkdir(mode=0o700)
            state.mkdir(mode=0o700)

            repaired = DRIVER.initialize_campaign_directory_tree(plan)
            self.assertEqual(
                repaired["preexisting_directory_paths"], [str(root), str(state)]
            )
            self.assertNotEqual(repaired["created_directory_paths"], [])

            repeated = DRIVER.initialize_campaign_directory_tree(plan)
            self.assertEqual(
                repeated["preexisting_directory_paths"],
                repeated["expected_directory_paths"],
            )
            self.assertEqual(repeated["created_directory_paths"], [])
            DRIVER.validate_campaign_directory_initialization_receipt(
                repeated, plan, verify_live=True
            )

    def test_partial_mkdir_failure_rolls_back_for_a_fresh_retry(self) -> None:
        with tempfile.TemporaryDirectory(dir="/private/tmp") as directory:
            plan = fixture_campaign_directory_plan(Path(directory))
            calls = 0

            def fail_fourth_mkdir(path: Path, mode: int) -> None:
                nonlocal calls
                calls += 1
                if calls == 4:
                    raise OSError("fixture mkdir failure")
                os.mkdir(path, mode)

            with self.assertRaises(DRIVER.CampaignError):
                DRIVER.initialize_campaign_directory_tree(plan, mkdir=fail_fourth_mkdir)
            self.assertFalse((Path(directory) / "campaign").exists())

            retry = DRIVER.initialize_campaign_directory_tree(plan)
            self.assertEqual(retry["preexisting_directory_paths"], [])

    def test_fresh_failed_prepare_tree_cleanup_allows_fresh_retry(self) -> None:
        with tempfile.TemporaryDirectory(dir="/private/tmp") as directory:
            parent = Path(directory)
            plan = fixture_campaign_directory_plan(parent)
            plan["repository_root"] = str(parent)
            plan["marker_repository_path"] = "marker.json"
            receipt = DRIVER.initialize_campaign_directory_tree(plan)
            root = parent / "campaign"
            (root / "state" / "selected.json").write_text("{}", encoding="utf-8")
            (root / "state" / ".local" / "logs" / "gateway.log").write_text(
                "fixture", encoding="utf-8"
            )
            (root / "runtime" / "llama.cpp-mac" / "logs" / "backend.log").write_text(
                "fixture", encoding="utf-8"
            )

            cleanup = DRIVER.cleanup_failed_prepare_campaign_tree(
                plan, receipt, "host-preflight-failure"
            )

            self.assertTrue(cleanup["all_passed"])
            self.assertTrue(cleanup["root_removed"])
            self.assertFalse(root.exists())
            self.assertFalse(Path(plan["raw_output_path"]).exists())
            retry = DRIVER.initialize_campaign_directory_tree(plan)
            self.assertEqual(retry["preexisting_directory_paths"], [])

    def test_prepare_host_gate_failure_stops_runtime_and_allows_fresh_retry(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(dir="/private/tmp") as directory:
            parent = Path(directory)
            plan = fixture_campaign_directory_plan(parent)
            plan["repository_root"] = str(parent)
            plan["marker_repository_path"] = "formal-marker.json"
            plan_path = parent / "execution-plan.json"
            marker = parent / plan["marker_repository_path"]
            raw = Path(plan["raw_output_path"])
            campaign_root = parent / "campaign"
            sibling = parent / "not-owned-by-campaign.txt"
            sibling.write_text("must survive", encoding="utf-8")
            binding = {"nonce": "a" * 64}
            plan_hash = "c" * 64
            context = {
                "plan": plan,
                "contract": fixture_contract(),
                "repository_root": parent,
                "contract_file": {"sha256": DRIVER._SHARED.FROZEN_CONTRACT_SHA256},
                "validator_file": {"sha256": DRIVER._SHARED.FROZEN_VALIDATOR_SHA256},
                "plan_file": {"sha256": plan_hash},
            }
            git_custody = {
                "tracked_files": {
                    "contract": {"blob_sha256": DRIVER._SHARED.FROZEN_CONTRACT_SHA256},
                    "validator": {
                        "blob_sha256": DRIVER._SHARED.FROZEN_VALIDATOR_SHA256
                    },
                    "plan": {"blob_sha256": plan_hash},
                }
            }

            def fake_start(
                _plan_path: Path,
                received_plan: dict[str, object],
                receipt: dict[str, object],
            ) -> dict[str, object]:
                self.assertIs(received_plan, plan)
                DRIVER.validate_campaign_directory_initialization_receipt(
                    receipt, plan, verify_live=True
                )
                (campaign_root / "state" / "selected.json").write_text(
                    "{}", encoding="utf-8"
                )
                (
                    campaign_root / "state" / ".local" / "logs" / "gateway.log"
                ).write_text("fixture", encoding="utf-8")
                (
                    campaign_root / "runtime" / "llama.cpp-mac" / "logs" / "backend.log"
                ).write_text("fixture", encoding="utf-8")
                return binding

            def fail_host_gate(_stage: str, _contract: dict[str, object]) -> None:
                self.assertTrue((campaign_root / "state" / "selected.json").is_file())
                raise DRIVER.CampaignError("fixture quiet-host rejection")

            with (
                mock.patch.object(
                    DRIVER, "load_execution_context", return_value=context
                ),
                mock.patch.object(
                    DRIVER, "collect_git_custody", return_value=git_custody
                ),
                mock.patch.object(
                    DRIVER, "verify_plan_artifacts", return_value={"fixture": True}
                ),
                mock.patch.object(
                    DRIVER, "start_custodian", side_effect=fake_start
                ) as start,
                mock.patch.object(
                    DRIVER,
                    "collect_runtime_preflight",
                    return_value={"fixture": "runtime"},
                ),
                mock.patch.object(
                    DRIVER, "shutdown_custodian", return_value={"passed": True}
                ) as shutdown,
            ):
                with self.assertRaises(DRIVER.CampaignError) as failed:
                    DRIVER.prepare_campaign(
                        plan_path, host_gate_collector=fail_host_gate
                    )

            start.assert_called_once()
            shutdown.assert_called_once_with(
                plan, binding, "prepare-failed-before-marker-cleanup"
            )
            cleanup = failed.exception.failed_prepare_cleanup
            self.assertTrue(cleanup["runtime_cleanup_complete"])
            self.assertTrue(cleanup["directory_cleanup"]["all_passed"])
            self.assertTrue(cleanup["directory_cleanup"]["root_removed"])
            self.assertFalse(campaign_root.exists())
            self.assertFalse(marker.exists())
            self.assertFalse(raw.exists())
            self.assertEqual(sibling.read_text(encoding="utf-8"), "must survive")

            retry = DRIVER.initialize_campaign_directory_tree(plan)
            self.assertEqual(retry["preexisting_directory_paths"], [])

    def test_marker_create_race_still_stops_runtime_without_deleting_tree(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(dir="/private/tmp") as directory:
            parent = Path(directory)
            plan = fixture_campaign_directory_plan(parent)
            plan["repository_root"] = str(parent)
            plan["marker_repository_path"] = "marker.json"
            receipt = DRIVER.initialize_campaign_directory_tree(plan)
            marker = parent / "marker.json"
            marker.write_text("race winner", encoding="utf-8")
            binding = {"nonce": "a" * 64}
            with (
                mock.patch.object(
                    DRIVER,
                    "shutdown_custodian",
                    return_value={"passed": True},
                ) as shutdown,
                mock.patch.object(
                    DRIVER, "cleanup_failed_prepare_campaign_tree"
                ) as directory_cleanup,
            ):
                cleanup = DRIVER._cleanup_failed_prepare_resources(
                    plan,
                    binding,
                    receipt,
                    marker,
                    Path(plan["raw_output_path"]),
                    "marker-create-race",
                )
            shutdown.assert_called_once_with(plan, binding, "marker-create-race")
            directory_cleanup.assert_not_called()
            self.assertTrue(cleanup["runtime_cleanup_complete"])
            self.assertIsNone(cleanup["directory_cleanup"])
            self.assertTrue((parent / "campaign").is_dir())

    def test_cleanup_root_swap_never_deletes_replacement_tree(self) -> None:
        with tempfile.TemporaryDirectory(dir="/private/tmp") as directory:
            parent = Path(directory)
            plan = fixture_campaign_directory_plan(parent)
            plan["repository_root"] = str(parent)
            plan["marker_repository_path"] = "marker.json"
            receipt = DRIVER.initialize_campaign_directory_tree(plan)
            root = parent / "campaign"
            moved = parent / "original-campaign"
            replacement = root / "replacement.txt"

            def swap_root() -> None:
                root.rename(moved)
                root.mkdir(mode=0o700)
                replacement.write_text("must survive", encoding="utf-8")

            cleanup = DRIVER.cleanup_failed_prepare_campaign_tree(
                plan,
                receipt,
                "root-swap-mutation",
                before_first_removal=swap_root,
            )

            self.assertFalse(cleanup["all_passed"])
            self.assertTrue(cleanup["contamination_detected"])
            self.assertEqual(cleanup["removed_paths"], [])
            self.assertEqual(replacement.read_text(encoding="utf-8"), "must survive")
            self.assertTrue(moved.is_dir())


class SelfTestTests(unittest.TestCase):
    def test_self_test_is_fixture_only(self) -> None:
        receipt = DRIVER.run_fixture_self_test()
        self.assertTrue(receipt["passed"])
        self.assertFalse(receipt["network_used"])
        self.assertFalse(receipt["model_process_used"])
        self.assertFalse(receipt["marker_created"])
        self.assertFalse(receipt["campaign_directory_tree_initialized"])


if __name__ == "__main__":
    unittest.main()
