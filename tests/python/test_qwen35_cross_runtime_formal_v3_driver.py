from __future__ import annotations

import importlib.util
import hashlib
import json
import copy
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "benchmarks" / "cross_runtime" / "formal_v3_driver.py"
CONTRACT_PATH = ROOT / "configs" / "qwen35-0.8b-cross-runtime-formal-v3.json"
VALIDATOR_PATH = ROOT / "scripts" / "validate_qwen35_cross_runtime_formal_contract.py"


def load_module():
    spec = importlib.util.spec_from_file_location(
        "qwen35_cross_runtime_formal_v3_driver_for_tests", MODULE_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module


def contract_fixture() -> dict:
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def canonical_free_ids(contract: dict) -> list[int]:
    teacher = contract["workload_contracts"]["NATIVE_RAW13_FREE128_V3"][
        "teacher_forced_admission"
    ]["teacher_input_token_ids"]
    result = teacher[1:] + [198]
    expected = contract["workload_contracts"]["NATIVE_RAW13_FREE128_V3"][
        "free_run_trajectory_admission"
    ]["expected_sha256"]
    assert (
        hashlib.sha256(json.dumps(result, separators=(",", ":")).encode()).hexdigest()
        == expected
    )
    return result


def custody_fixture(arm: str) -> dict:
    suffix = "a" if arm == "AN" else "b"
    return {
        "configuration_id": (
            "ApxInf-native-hybrid-G32-G64-W8-CPU-F32-remainder-F32-KV-v3"
            if arm == "AN"
            else "llama.cpp-f280b269-Q8_0-Metal-F16-KV-threads4-v3"
        ),
        "runner": {
            "absolute_path": f"/fixture/{arm.lower()}-runner",
            "size_bytes": 100,
            "sha256": suffix * 64,
        },
        "model": {
            "absolute_path": f"/fixture/{arm.lower()}-model",
            "size_bytes": 200,
            "sha256": (
                "04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696"
                if arm == "AN"
                else "427555e797eefeb62e1c8ef71510ce062027b0c7f1674cc4cfcb352710e0908c"
            ),
        },
        "runtime_source_commit": (
            "c" * 40
            if arm == "AN"
            else "f280b26983ad0fdb705a0d9ebf0503e76f2899b0"
        ),
        "loaded_non_system_library_closure_sha256": hashlib.sha256(b"[]").hexdigest(),
        "packed_weight_and_resident_buffer_manifest_sha256": (
            "e" * 64 if arm == "AN" else None
        ),
        "deployment": (
            {
                "constructor_id": (
                    "from_weights_with_metal_w8_mlp_stack3_boundary_tail_head_"
                    "gdn_core_fused_v1"
                ),
                "profile_id": "metal-w8-mlp-stack3-boundary-tail-head-gdn-core-fused-v1",
                "context_capacity_tokens": 256,
                "prefill_device": "CPU",
                "prefill_precision": "F32",
                "full_attention_device": "CPU",
                "full_attention_precision": "F32",
                "kv_key_dtype": "F32",
                "kv_value_dtype": "F32",
                "head": "F32 tied embedding top-4 exact rerank",
                "metal_build_input_count": 17,
                "exact_live_execution_ledger": True,
                "thread_policy": {
                    "policy": "Accelerate OS-managed default",
                    "fixed_worker_count_claimed": False,
                    "VECLIB_MAXIMUM_THREADS_present": False,
                    "OMP_NUM_THREADS_present": False,
                    "OPENBLAS_NUM_THREADS_present": False,
                    "MKL_NUM_THREADS_present": False,
                },
            }
            if arm == "AN"
            else {
                "context_capacity_tokens": 256,
                "model_type": "GGUF-Q8_0",
                "kv_key_dtype": "F16",
                "kv_value_dtype": "F16",
                "threads": 4,
                "batch_threads": 4,
                "transformer_layers_on_mtl0": 24,
                "input_embedding_cpu_fallback_observed": True,
                "dynamic_backend_scan": False,
            }
        ),
    }


def artifact_observation_fixture(custody: dict) -> dict:
    """One preflight O_NOFOLLOW snapshot matching the llama raw fixture."""

    return {
        "configuration_id": custody["configuration_id"],
        "runner": {
            "absolute_path": custody["runner"]["absolute_path"],
            "device": 11,
            "inode": 12,
            "mode": 0o100755,
            "size_bytes": custody["runner"]["size_bytes"],
            "hard_link_count": 1,
            "ctime_ns": 13_000_000_014,
            "sha256": custody["runner"]["sha256"],
            "open_flags": ["O_RDONLY", "O_CLOEXEC", "O_NOFOLLOW"],
            "identity_before_after_equal": True,
        },
        "model": {
            "absolute_path": custody["model"]["absolute_path"],
            "device": 1,
            "inode": 2,
            "mode": 0o100644,
            "size_bytes": custody["model"]["size_bytes"],
            "hard_link_count": 1,
            "ctime_ns": 3_000_000_004,
            "sha256": custody["model"]["sha256"],
            "open_flags": ["O_RDONLY", "O_CLOEXEC", "O_NOFOLLOW"],
            "identity_before_after_equal": True,
        },
        "runtime_source_commit": custody["runtime_source_commit"],
        "loaded_non_system_library_closure_sha256": custody[
            "loaded_non_system_library_closure_sha256"
        ],
        "packed_weight_and_resident_buffer_manifest_sha256": custody[
            "packed_weight_and_resident_buffer_manifest_sha256"
        ],
        "deployment": copy.deepcopy(custody["deployment"]),
    }


def an_source_custody_fixture() -> tuple[dict, dict]:
    file_receipt = {
        "path": "/fixture/file",
        "size": 1,
        "sha256": "1" * 64,
        "device": 1,
        "inode": 2,
        "change_time_seconds": 3,
        "change_time_nanoseconds": 4,
        "direct_regular_file": True,
        "single_link": True,
    }
    artifacts = {"model.safetensors": copy.deepcopy(file_receipt)}
    gate = {**copy.deepcopy(file_receipt), "path": "/fixture/gate.rs"}
    rust = {
        "runner": {
            **copy.deepcopy(file_receipt),
            "path": "/fixture/runner.rs",
        }
    }
    metal = {
        "kernel": {
            **copy.deepcopy(file_receipt),
            "path": "/fixture/kernel.metal",
        }
    }
    start = {
        "profile": {
            "profile_id": "fixture-profile",
            "path": "/fixture/profile.json",
            "file_size": 1,
            "file_sha256": "2" * 64,
            "direct_regular_file": True,
            "single_link": True,
        },
        "source_lock": {
            "path": "/fixture/source-lock.json",
            "file_size": 1,
            "file_sha256": "3" * 64,
            "content_sha256": "4" * 64,
            "direct_regular_file": True,
            "single_link": True,
        },
        "binary": copy.deepcopy(file_receipt),
        "model_dir": {
            "path": "/fixture/model-dir",
            "cache_present": False,
            "artifacts": artifacts,
        },
        "sources": {
            "set_id": "fixture-source-set-v1",
            "coverage": "fixture-explicit-set-v1",
            "captured_at_start": True,
            "binary_attestation_authoritative_for_full_executable": True,
            "gate": copy.deepcopy(gate),
            "rust_and_bridge_sources": rust,
            "compiled_metal_shader_sources": metal,
        },
    }
    end = {
        "verified_at_end": True,
        "source_set_id": "fixture-source-set-v1",
        "source_set_coverage": "fixture-explicit-set-v1",
        "deployment_profile": {
            **copy.deepcopy(file_receipt),
            "path": "/fixture/profile.json",
            "sha256": "2" * 64,
        },
        "source_lock": {
            **copy.deepcopy(file_receipt),
            "path": "/fixture/source-lock.json",
            "sha256": "3" * 64,
        },
        "binary": copy.deepcopy(file_receipt),
        "model_dir": {
            "path": "/fixture/model-dir",
            "cache_present": False,
            "artifacts": copy.deepcopy(artifacts),
            "loaded_from_start_pinned_artifacts": True,
        },
        "gate": copy.deepcopy(gate),
        "rust_and_bridge_sources": copy.deepcopy(rust),
        "compiled_metal_shader_sources": copy.deepcopy(metal),
    }
    return start, end


def an_source_binding_fixture(module, custody: dict) -> dict:
    claims = module._an_repository_source_file_claims(
        custody, Path("/fixture"), "fixture AN"
    )
    files = {
        path: {
            **claim,
            "source_commit_blob_oid": "5" * 40,
            "live_head_blob_oid": "5" * 40,
            "source_and_live_blob_equal": True,
        }
        for path, claim in claims.items()
    }
    return {
        "repository_root": "/fixture",
        "campaign_commit": custody["runtime_source_commit"],
        "campaign_tree": "6" * 40,
        "source_files": files,
        "source_files_sha256": hashlib.sha256(
            json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }


def sample_receipt_fixture(module, contract: dict, slot: dict, nonce: str) -> dict:
    prompt = contract["workload_contracts"]["shared_prompt"]["token_ids"]
    generated = canonical_free_ids(contract)
    custody = custody_fixture(slot["arm"])
    start_ns = 1_000_000_000
    first_ns = 1_010_000_000
    last_ns = 2_280_000_000
    return {
        "format": module.SAMPLE_FORMAT,
        "schema_version": 3,
        "campaign_id": contract["campaign_id"],
        "subcampaign_id": contract["comparison_graph"]["edges"][
            "NATIVE_A_VS_L"
        ]["subcampaign_id"],
        "edge_id": "NATIVE_A_VS_L",
        "mode": "native-v3-free",
        "request": {
            "nonce": nonce,
            "sequence_index": slot["sequence_index"],
            "phase": slot["phase"],
            "warmup_index": slot["warmup_index"],
            "block_index": slot["block_index"],
            "slot_index": slot["slot_index"],
            "role": slot["role"],
            "arm": slot["arm"],
        },
        "workload": {
            "ingress_semantics": "raw-token-ids",
            "prompt_token_ids": prompt,
            "prefill_token_count": 13,
            "generated_token_ids": generated,
            "generated_token_ids_sha256": hashlib.sha256(
                json.dumps(generated, separators=(",", ":")).encode()
            ).hexdigest(),
            "generated_token_count": 128,
            "sampling": "unbiased-greedy-argmax",
            "temperature": 0,
            "eog_policy": (
                "select-and-feed-back-eog-without-termination-and-without-"
                "eog-logit-suppression"
            ),
            "speculative_decoding": False,
            "continuous_batching": False,
            "sequence_count": 1,
            "requested_context_tokens": 256,
            "effective_context_tokens": 256,
            "requested_batch_tokens": 13,
            "effective_batch_tokens": 13,
            "requested_ubatch_tokens": 13,
            "effective_ubatch_tokens": 13,
            "empty_state_before_prefill": True,
            "prompt_cache_reused": False,
        },
        "timing": {
            "clock": "monotonic",
            "clock_identity": "fixture-monotonic-ns",
            "clock_resolution_ns": 1,
            "start_boundary": "immediately-before-first-raw-token-prefill-dispatch",
            "common_token_ready_boundary": "next-greedy-token-ready",
            "end_boundary": "128th-next-greedy-token-ready",
            "selection_work_included": True,
            "accelerator_completion_before_each_token_ready_timestamp": True,
            "final_sampled_token_decoded_inside_timed_region": False,
            "prefill_start_ns": start_ns,
            "token_1_ready_ns": first_ns,
            "token_128_ready_ns": last_ns,
            "ttft_ms": (first_ns - start_ns) / 1_000_000,
            "total_latency_ms": (last_ns - start_ns) / 1_000_000,
            "tpot_ms": (last_ns - first_ns) / 127 / 1_000_000,
            "generation_tps": 127_000_000_000 / (last_ns - first_ns),
        },
        "custody": {
            **custody,
            "fresh_process": True,
            "start_end_identity_equal": True,
            "ggml_backend_path_unset": slot["arm"] == "L",
        },
    }


def teacher_receipt_fixture(module, contract: dict, arm: str) -> dict:
    prompt = contract["workload_contracts"]["shared_prompt"]["token_ids"]
    native = contract["workload_contracts"]["NATIVE_RAW13_FREE128_V3"]
    teacher = native["teacher_forced_admission"]
    canonical = canonical_free_ids(contract)
    return {
        "format": module.TEACHER_FORMAT,
        "schema_version": 3,
        "campaign_id": contract["campaign_id"],
        "subcampaign_id": contract["comparison_graph"]["edges"][
            "NATIVE_A_VS_L"
        ]["subcampaign_id"],
        "edge_id": "NATIVE_A_VS_L",
        "arm": arm,
        "mode": "native-v3-teacher",
        "prefill_prompt_token_ids": prompt[:-1],
        "prefill_prompt_token_ids_count": 12,
        "teacher_input_token_ids": teacher["teacher_input_token_ids"],
        "teacher_input_token_ids_sha256": teacher["teacher_input_token_ids_sha256"],
        "reference_argmax_token_ids": canonical,
        "observed_argmax_token_ids": canonical,
        "mismatch_positions": [],
        "first_mismatch": None,
        "reference_receipt_size_bytes": 41,
        "reference_receipt_sha256": "8" * 64,
        "runtime_receipt_size_bytes": 43,
        "runtime_receipt_sha256": "9" * 64,
        "custody": custody_fixture(arm),
    }


def host_receipt_fixture(module, contract: dict, phase: str = "preflight") -> dict:
    host = contract["scope"]["host"]
    snapshot_count = 5 if phase in ("preflight", "postflight") else 1
    snapshots = []
    for index in range(snapshot_count):
        window_start_ns = 10_000 + index * 250_000_000
        monotonic_ns = window_start_ns + 250_000_000
        snapshots.append(
            {
                "index": index,
                "cpu_window_start_monotonic_ns": window_start_ns,
                "monotonic_ns": monotonic_ns,
                "cpu_percent_window_ms": 250.0,
                "cpu_measurement_source": "libproc-PROC_PIDTASKINFO-delta",
                "resolved_allowlist": [
                    {
                        "role": "campaign_orchestrator",
                        "pid": 10,
                        "process_start_time": "fixture-start",
                        "executable_path": "/fixture/python",
                        "executable_sha256": "a" * 64,
                        "argv_sha256": "b" * 64,
                        "process_group_id": 10,
                    },
                    {
                        "role": "custody_monitor",
                        "pid": 10,
                        "process_start_time": "fixture-start",
                        "executable_path": "/fixture/python",
                        "executable_sha256": "a" * 64,
                        "argv_sha256": "b" * 64,
                    },
                ],
                "nonallowlisted_processes": [],
                "vanished_nonallowlisted_processes": [],
                "cpu_window_proof_complete": True,
                "maximum_single_nonallowlisted_process_cpu_percent": 0.0,
                "aggregate_nonallowlisted_process_cpu_percent": 0.0,
                "load_average_per_logical_cpu": 0.1,
                "campaign_process_swap_bytes": 0,
                "campaign_process_swap_observations": [],
                "campaign_swap_probe_vanished_processes": [],
                "active_runtime_root_present": None,
                "active_runtime_swap_proof_complete": False,
                "power_source": "AC Power",
                "thermal_warning": False,
                "performance_warning": False,
                "system_swap_used_bytes": 0,
                "memory_pressure_pages_throttled": 0,
                "system_state_matches_gate_start": True,
                "passed": True,
            }
        )
    return {
        "format": module.HOST_FORMAT,
        "schema_version": 3,
        "phase": phase,
        "host": {
            "model_identifier": host["model_identifier"],
            "chip": host["chip"],
            "architecture": host["architecture"],
            "logical_cpu_count": host["logical_cpu_count"],
            "memory_bytes": host["memory_bytes"],
            "os_product": host["os_product"],
            "os_version": host["os_version"],
            "os_build": host["os_build"],
        },
        "power_source": "AC Power",
        "thermal_warning": False,
        "performance_warning": False,
        "snapshot_interval_ms": 250,
        "snapshots": snapshots,
        "system_swap_used_bytes_start": 0,
        "memory_pressure_pages_throttled_start": 0,
        "swap_delta_bytes": 0,
        "memory_pressure_pages_throttled_delta": 0,
        "power_or_thermal_state_changed": False,
        "processes_terminated_or_modified": False,
        "accepted_runtime_swap_proofs": [],
        "passed": True,
    }


def execution_plan_fixture(module, contract: dict) -> dict:
    artifacts = {arm: custody_fixture(arm) for arm in ("AN", "L")}
    artifacts["AN"]["model"]["size_bytes"] = contract["source_model_custody"][
        "checkpoint"
    ]["size_bytes"]
    artifacts["L"]["runner"]["size_bytes"] = contract["runtime_custody"][
        "pinned_llama_cpp_core"
    ]["runner_binary_size_bytes"]
    artifacts["L"]["runner"]["sha256"] = contract["runtime_custody"][
        "pinned_llama_cpp_core"
    ]["runner_binary_sha256"]
    artifacts["L"]["model"]["size_bytes"] = contract["native_deployment_contract"][
        "deployments"
    ]["L"]["source_weights"]["artifact_size_bytes"]
    return {
        "format": module.PLAN_FORMAT,
        "schema_version": 3,
        "edge_id": "NATIVE_A_VS_L",
        "repository_root": str(ROOT),
        "contract_repository_path": "configs/qwen35-0.8b-cross-runtime-formal-v3.json",
        "validator_repository_path": "scripts/validate_qwen35_cross_runtime_formal_contract.py",
        "driver_repository_path": "benchmarks/cross_runtime/formal_v3_driver.py",
        "plan_repository_path": "configs/qwen35-native-formal-v3-execution-plan.json",
        "marker_repository_path": (
            "crates/apxinf-metal/evidence/llama-cpp/qwen35-0.8b-native-apxinf-"
            "vs-llamacpp-formal-v3-campaign-start-20260826.json"
        ),
        "raw_output_path": "/fixture/native-v3-raw-partial.json",
        "timeout_seconds": 900,
        "commands": {
            "AN": {
                "argv": [
                    artifacts["AN"]["runner"]["absolute_path"],
                    "--mode",
                    "native-v3-free",
                    "--model-dir",
                    str(Path(artifacts["AN"]["model"]["absolute_path"]).parent),
                    "--source-lock",
                    str(ROOT / ".apxinf/onboarding/qwen35-0.8b/source-lock.json"),
                ],
                "environment": {"LC_ALL": "C", "TZ": "UTC"},
            },
            "L": {
                "argv": [
                    artifacts["L"]["runner"]["absolute_path"],
                    "--model",
                    artifacts["L"]["model"]["absolute_path"],
                    "--gpu-layers",
                    "-1",
                    "--gpu-device",
                    "MTL0",
                    "--threads",
                    "4",
                    "--mode",
                    "native-v3-free",
                ],
                "environment": {"LC_ALL": "C", "TZ": "UTC"},
            },
        },
        "artifacts": artifacts,
        "teacher_receipts": {
            arm: {
                "reference_repository_path": "evidence/reference-teacher-raw.json",
                "runtime_repository_path": f"evidence/{arm.lower()}-teacher-raw.json",
            }
            for arm in ("AN", "L")
        },
    }


def llama_raw_free_fixture(contract: dict, custody: dict) -> dict:
    prompt = contract["workload_contracts"]["shared_prompt"]["token_ids"]
    generated = canonical_free_ids(contract)
    elapsed = [10_000_000 + index * 10_000_000 for index in range(128)]
    identity = {
        "device": 1,
        "inode": 2,
        "size_bytes": custody["model"]["size_bytes"],
        "hard_link_count": 1,
        "change_time_seconds": 3,
        "change_time_nanoseconds": 4,
    }
    return {
        "schema": "apxinf.llama-cpp.raw-token-diagnostic.v3",
        "ok": True,
        "mode": "native-v3-free",
        "token_ready_boundary": "next-greedy-token-ready",
        "selection_work_included": True,
        "accelerator_completion_before_each_token_ready_timestamp": True,
        "contract": {
            "prompt_token_ids": prompt,
            "mode": "native-v3-free",
            "sampling": "greedy-argmax",
            "generated_token_count": 128,
            "eog_termination": False,
            "token_ready_elapsed_ns_origin": "immediately-before-prompt-decode",
            "final_sampled_token_is_not_decoded_in_timed_workload": True,
            "final_sampled_token_decoded_once_post_measurement_for_execution_proof": True,
        },
        "model": {
            "requested_path": custody["model"]["absolute_path"],
            "load_binding": "pinned-file-descriptor",
            "open_flags": "O_RDONLY|O_NOFOLLOW|O_CLOEXEC",
            "file_identity_start": identity,
            "file_identity_after_load": identity,
            "file_identity_before_receipt": identity,
            "file_identity_unchanged": True,
            "file_size_bytes": custody["model"]["size_bytes"],
            "description": "Qwen3.5 fixture",
            "parameter_count": 800_000_000,
            "tensor_size_bytes": custody["model"]["size_bytes"],
            "file_type": "Q8_0",
            "file_type_code": 7,
            "vocabulary_size": 248320,
            "layer_count": 24,
            "is_recurrent": False,
            "is_hybrid": True,
        },
        "parameters": {
            "n_ctx_requested": 256,
            "n_ctx_effective": 256,
            "n_ctx_per_sequence_effective": 256,
            "n_batch_requested": 13,
            "n_batch_effective": 13,
            "n_ubatch_requested": 13,
            "n_ubatch_effective": 13,
            "n_seq_max_requested": 1,
            "n_seq_max_effective": 1,
            "n_threads": 4,
            "n_threads_batch": 4,
            "lane": "gpu-all-layers",
            "n_gpu_layers": -1,
            "kv_cache_type_k": "f16",
            "kv_cache_type_v": "f16",
            "flash_attention": "auto",
            "offload_kqv": True,
            "op_offload": True,
            "swa_full": False,
            "kv_unified": False,
            "model_load_mode": "none-from-pinned-file-pointer",
            "use_mmap": False,
            "use_direct_io": False,
            "use_mlock": False,
            "check_tensors": False,
        },
        "output": {
            "token_ids": generated,
            "token_ready_elapsed_ns": elapsed,
        },
        "timings": {
            "clock_identity": "std::chrono::steady_clock",
            "clock_is_steady": True,
            "clock_resolution_ns": 1,
            "clock_period_numerator": 1,
            "clock_period_denominator": 1_000_000_000,
            "generation_start_ns": 5_000_000_000,
            "model_load_elapsed_ns": 1,
            "context_init_elapsed_ns": 1,
            "generation_elapsed_ns": elapsed[-1] + 1,
            "measurement_scope_elapsed_ns": elapsed[-1] + 2,
            "post_measurement_execution_proof_elapsed_ns": 1,
            "receipt_ready_elapsed_ns": elapsed[-1] + 3,
        },
        "llama_perf": {
            "context": {
                "t_start_ms": 1.0,
                "t_load_ms": 2.0,
                "t_prompt_eval_ms": 3.0,
                "t_eval_ms": 4.0,
                "n_prompt_eval": 13,
                "n_eval": 127,
                "n_reused": 126,
            },
            "sampler": {"t_sample_ms": 0.0, "n_sample": 0},
            "captured_before_post_measurement_execution_proof": True,
        },
        "runtime_custody": {
            "loaded_non_system_library_closure": [],
            "loaded_non_system_library_closure_start": [],
            "loaded_non_system_library_closure_end": [],
            "loaded_non_system_library_closure_sha256": custody[
                "loaded_non_system_library_closure_sha256"
            ],
            "loaded_non_system_library_closure_start_sha256": custody[
                "loaded_non_system_library_closure_sha256"
            ],
            "loaded_non_system_library_closure_end_sha256": custody[
                "loaded_non_system_library_closure_sha256"
            ],
        },
        "backend": {
            "registration_mode": "linked-static-registry-only",
            "dynamic_backend_scan_invoked": False,
            "backend_directory_option_supported": False,
            "ggml_backend_path_present": False,
            "supports_gpu_offload": True,
            "selected_gpu_device": {
                "name": "MTL0",
                "description": "Apple M4",
                "type": "gpu",
            },
            "registered_devices_after_generation": [
                {"name": "MTL0", "description": "Apple M4", "type": "gpu"},
                {"name": "CPU", "description": "Apple M4", "type": "cpu"},
            ],
            "system_info": "fixture-system-info",
        },
        "placement_attestation": {
            "method": "pinned-llama-internal-layer-assignments-plus-memory-breakdown-v1",
            "passed": True,
            "model_selected_device_count": 1,
            "transformer_layer_count": 24,
            "layers_on_selected_gpu": 24,
            "layers_on_cpu": 0,
            "output_on_selected_gpu": True,
            "output_on_cpu": False,
            "input_embedding_buffer_type": "CPU",
            "input_embedding_device": {
                "name": "CPU",
                "description": "Apple M4",
                "type": "cpu",
            },
            "memory_by_device_class": {
                "gpu": {"buffer_bytes": 1, "tensor_bytes": 1},
                "cpu": {"buffer_bytes": 1, "tensor_bytes": 1},
                "accelerator": {"buffer_bytes": 0, "tensor_bytes": 0},
                "other": {"buffer_bytes": 0, "tensor_bytes": 0},
            },
            "memory_by_buffer_type": [],
        },
        "post_measurement_execution_proof": {
            "method": "scheduler-callback-completed-sentinels-v1",
            "passed": True,
            "timing_excluded": True,
            "decode_count": 1,
            "proof_token_id": generated[-1],
            "requested_sentinel_count": 26,
            "completed_sentinel_count": 26,
            "completed_input_embedding_on_cpu": True,
            "completed_transformer_layer_endpoints": 24,
            "completed_output_head": True,
            "completed_on_selected_gpu": 25,
            "completed_on_cpu": 1,
            "backend_mismatch": False,
            "duplicate_or_unexpected_callback": False,
        },
        "build": {
            "llama_cpp_source_id": custody["runtime_source_commit"],
            "llama_cpp_source_id_provenance": "clean-git-head",
            "llama_cpp_version": "fixture-version",
            "cmake_version": "fixture-cmake",
            "cxx_compiler_id": "AppleClang",
            "cxx_compiler_version": "fixture-compiler-version",
            "cmake_build_type": "Release",
            "build_shared_libs": False,
            "ggml_backend_dl": False,
            "ggml_metal": True,
            "ggml_metal_embed_library": True,
            "ggml_accelerate": True,
            "ggml_native": True,
            "cxx_compiler_banner": "fixture-compiler-banner",
        },
    }


def an_raw_free_fixture(module, contract: dict, slot: dict, nonce: str):
    custody = copy.deepcopy(custody_fixture("AN"))
    library_closure = [
        {
            "absolute_path": "/fixture/libmetal.dylib",
            "size_bytes": 10,
            "sha256": "7" * 64,
            "device": 1,
            "inode": 2,
            "change_time_seconds": 3,
            "change_time_nanoseconds": 4,
        }
    ]
    ledger = {"scope": "fixture-exact-ledger", "allocated_buffers": 494}
    custody["loaded_non_system_library_closure_sha256"] = hashlib.sha256(
        json.dumps(library_closure, separators=(",", ":")).encode()
    ).hexdigest()
    custody["packed_weight_and_resident_buffer_manifest_sha256"] = hashlib.sha256(
        json.dumps(ledger, separators=(",", ":")).encode()
    ).hexdigest()
    receipt = sample_receipt_fixture(module, contract, slot, nonce)
    source_start, source_end = an_source_custody_fixture()
    receipt["custody"] = {
        **custody,
        "fresh_process": True,
        "start_end_identity_equal": True,
        "ggml_backend_path_unset": False,
        "loaded_non_system_library_closure": library_closure,
        "loaded_non_system_library_closure_start": library_closure,
        "loaded_non_system_library_closure_end": library_closure,
        "loaded_non_system_library_closure_start_sha256": custody[
            "loaded_non_system_library_closure_sha256"
        ],
        "loaded_non_system_library_closure_end_sha256": custody[
            "loaded_non_system_library_closure_sha256"
        ],
        "thread_policy_runtime": {
            "logical_cpu_count": contract["scope"]["host"]["logical_cpu_count"],
            "logical_cpu_count_source": "std::thread::available_parallelism",
            "fixed_worker_count_claimed": False,
            "environment_overrides_absent": True,
            "absent_environment_overrides": [
                "VECLIB_MAXIMUM_THREADS",
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
            ],
        },
        "source_custody_start": source_start,
        "source_custody_end": source_end,
    }
    start = receipt["timing"]["prefill_start_ns"]
    first = receipt["timing"]["token_1_ready_ns"]
    last = receipt["timing"]["token_128_ready_ns"]
    stride = (last - first) // 127
    token_ready = [first + index * stride for index in range(128)]
    token_ready[-1] = last
    receipt["timing"]["token_ready_ns"] = token_ready
    receipt["timing"]["clock_identity"] = "Darwin CLOCK_MONOTONIC_RAW"
    receipt["timing"]["next_greedy_token_ready_elapsed_ns"] = [
        value - start for value in token_ready
    ]
    receipt["final_path"] = {
        "path_checks": {
            "schedule_valid": True,
            "mechanism_and_precision_valid": True,
            "six_region_execution_valid": True,
            "tail_execution_and_phase_valid": True,
            "aggregate_ledger_valid": True,
            "generation_receipt_valid": True,
            "terminal_clear": True,
            "all_valid": True,
        },
        "aggregate_buffer_ledger": ledger,
    }
    receipt["passed"] = True
    return receipt, custody


def native_teacher_raw_fixtures(module, contract: dict, custody: dict):
    prompt = contract["workload_contracts"]["shared_prompt"]["token_ids"]
    teacher = contract["workload_contracts"]["NATIVE_RAW13_FREE128_V3"][
        "teacher_forced_admission"
    ]
    canonical = canonical_free_ids(contract)
    base = {
        "format": "apxinf-qwen35-native-teacher-runtime-receipt-v3",
        "schema_version": 3,
        "campaign_id": contract["campaign_id"],
        "subcampaign_id": contract["comparison_graph"]["edges"][
            "NATIVE_A_VS_L"
        ]["subcampaign_id"],
        "edge_id": "NATIVE_A_VS_L",
        "mode": "native-v3-teacher",
        "prefill_prompt_token_ids": prompt[:-1],
        "prefill_prompt_token_ids_count": 12,
        "teacher_input_token_ids": teacher["teacher_input_token_ids"],
        "teacher_input_token_ids_sha256": teacher["teacher_input_token_ids_sha256"],
        "reference_argmax_token_ids": canonical,
        "eog_termination": False,
        "passed": True,
    }
    source_start, source_end = an_source_custody_fixture()
    reference_custody = {
        **copy.deepcopy(custody),
        "configuration_id": "ApxInf-native-CPU-F32-teacher-reference-v3",
        "packed_weight_and_resident_buffer_manifest_sha256": None,
        "deployment": {
            "constructor_id": "from_weights",
            "context_capacity_tokens": 256,
            "prefill_device": "CPU",
            "prefill_precision": "F32",
            "full_attention_device": "CPU",
            "full_attention_precision": "F32",
            "kv_key_dtype": "F32",
            "kv_value_dtype": "F32",
            "head": "CPU/F32 full-vocabulary tied argmax",
            "teacher_reference_only": True,
        },
        "fresh_process": True,
        "start_end_identity_equal": True,
        "loaded_non_system_library_closure": [],
        "loaded_non_system_library_closure_start": [],
        "loaded_non_system_library_closure_end": [],
        "loaded_non_system_library_closure_start_sha256": hashlib.sha256(
            b"[]"
        ).hexdigest(),
        "loaded_non_system_library_closure_end_sha256": hashlib.sha256(
            b"[]"
        ).hexdigest(),
        "thread_policy_runtime": {
            "logical_cpu_count": contract["scope"]["host"]["logical_cpu_count"],
            "logical_cpu_count_source": "std::thread::available_parallelism",
            "fixed_worker_count_claimed": False,
            "environment_overrides_absent": True,
            "absent_environment_overrides": [
                "VECLIB_MAXIMUM_THREADS",
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
            ],
        },
        "source_custody_start": copy.deepcopy(source_start),
        "source_custody_end": copy.deepcopy(source_end),
    }
    reference = {
        **copy.deepcopy(base),
        "teacher_role": "reference",
        "arm": "CPU_REFERENCE",
        "custody": reference_custody,
    }
    observed_custody = {
        **copy.deepcopy(custody),
        "fresh_process": True,
        "start_end_identity_equal": True,
        "loaded_non_system_library_closure": [],
        "loaded_non_system_library_closure_start": [],
        "loaded_non_system_library_closure_end": [],
        "loaded_non_system_library_closure_start_sha256": hashlib.sha256(
            b"[]"
        ).hexdigest(),
        "loaded_non_system_library_closure_end_sha256": hashlib.sha256(
            b"[]"
        ).hexdigest(),
        "thread_policy_runtime": {
            "logical_cpu_count": contract["scope"]["host"]["logical_cpu_count"],
            "logical_cpu_count_source": "std::thread::available_parallelism",
            "fixed_worker_count_claimed": False,
            "environment_overrides_absent": True,
            "absent_environment_overrides": [
                "VECLIB_MAXIMUM_THREADS",
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
            ],
        },
        "source_custody_start": copy.deepcopy(source_start),
        "source_custody_end": copy.deepcopy(source_end),
    }
    observed = {
        **copy.deepcopy(base),
        "teacher_role": "observed",
        "arm": "AN",
        "tail_normalized_hidden_f32_argmax_token_ids": canonical,
        "tail_top4_candidate_token_ids": [
            [token, (token + 1) % 248320, (token + 2) % 248320, (token + 3) % 248320]
            for token in canonical
        ],
        "observed_argmax_token_ids": canonical,
        "mismatch_positions": [],
        "first_mismatch": None,
        "next_greedy_token_ready_elapsed_ns": [index + 1 for index in range(128)],
        "accelerator_candidate_elapsed_ns": [1] * 128,
        "f32_tied_rerank_elapsed_ns": [1] * 128,
        "selection_work_included": True,
        "accelerator_completion_before_each_token_ready_timestamp": True,
        "prefill_path": {"path_checks": {"all_valid": True}},
        "final_path": {"path_checks": {"all_valid": True}},
        "custody": observed_custody,
    }
    return reference, observed


def llama_teacher_raw_fixture(contract: dict, custody: dict) -> dict:
    raw = llama_raw_free_fixture(contract, custody)
    raw["mode"] = "native-v3-teacher"
    raw["contract"]["mode"] = "native-v3-teacher"
    raw["contract"][
        "token_ready_elapsed_ns_origin"
    ] = "immediately-before-teacher-step-0-decode"
    teacher = contract["workload_contracts"]["NATIVE_RAW13_FREE128_V3"][
        "teacher_forced_admission"
    ]
    canonical = canonical_free_ids(contract)
    elapsed = raw["output"]["token_ready_elapsed_ns"]
    raw["teacher_forced"] = {
        "raw_prompt_token_ids": contract["workload_contracts"]["shared_prompt"][
            "token_ids"
        ],
        "raw_prefill_token_ids": contract["workload_contracts"]["shared_prompt"][
            "token_ids"
        ][:-1],
        "teacher_input_token_ids": teacher["teacher_input_token_ids"],
        "reference_argmax_token_ids": canonical,
        "observed_argmax_token_ids": canonical,
        "mismatch_positions": [],
        "first_mismatch": None,
        "mismatch_count": 0,
        "exact_128_of_128": True,
        "raw_prefill_token_count": 12,
        "teacher_step_count": 128,
        "teacher_step_input_position_first_zero_based": 12,
        "teacher_step_input_position_last_zero_based": 139,
        "context_token_count_before_execution_proof": 140,
        "teacher_input_derivation": "prompt[-1]+canonical_free[:127]",
        "teacher_input_derivation_recomputed_and_matched": True,
        "teacher_input_token_ids_sha256": teacher["teacher_input_token_ids_sha256"],
        "reference_argmax_token_ids_sha256": contract["workload_contracts"][
            "NATIVE_RAW13_FREE128_V3"
        ]["free_run_trajectory_admission"]["expected_sha256"],
        "argmax_scope": "all-248320-raw-logits-lowest-token-id-wins-ties",
        "argmax_timing_included": True,
        "eog_termination": False,
        "next_greedy_token_ready_elapsed_ns": elapsed,
    }
    raw["timings"]["teacher_prefill_elapsed_ns"] = 1
    raw["llama_perf"]["context"].update(
        {
            "n_prompt_eval": 12,
            "n_eval": 128,
            "n_reused": 127,
        }
    )
    return raw


class Qwen35CrossRuntimeFormalV3DriverTests(unittest.TestCase):
    def test_schedule_is_the_frozen_three_warmups_and_eight_balanced_blocks(self):
        module = load_module()

        schedule = module.declared_schedule()

        self.assertEqual(
            [slot["arm"] for slot in schedule if slot["phase"] == "warmup"],
            ["AN", "L", "L", "AN", "AN", "L"],
        )
        timed = [slot for slot in schedule if slot["phase"] == "timed"]
        self.assertEqual(len(timed), 32)
        self.assertEqual(
            [
                "".join(slot["role"] for slot in timed[index : index + 4])
                for index in range(0, len(timed), 4)
            ],
            ["ABBA", "BAAB", "ABBA", "BAAB", "ABBA", "BAAB", "ABBA", "BAAB"],
        )
        self.assertEqual(
            {arm: sum(slot["arm"] == arm for slot in timed) for arm in ("AN", "L")},
            {"AN": 16, "L": 16},
        )

    def test_sample_receipt_accepts_only_the_bound_raw13_free128_token_ready_sample(self):
        module = load_module()
        contract = contract_fixture()
        slot = declared = module.declared_schedule()[0]
        nonce = "1" * 64
        receipt = sample_receipt_fixture(module, contract, slot, nonce)

        validated = module.validate_sample_receipt(
            receipt, declared, nonce, contract, custody_fixture("AN")
        )

        self.assertIs(validated, receipt)

    def test_teacher_receipt_proves_prompt12_then_the_prebound_128_steps(self):
        module = load_module()
        contract = contract_fixture()
        receipt = teacher_receipt_fixture(module, contract, "AN")

        validated = module.validate_teacher_receipt(
            receipt,
            "AN",
            contract,
            {"size_bytes": 41, "sha256": "8" * 64},
            {"size_bytes": 43, "sha256": "9" * 64},
            custody_fixture("AN"),
        )

        self.assertIs(validated, receipt)

    def test_driver_loads_one_hash_pinned_snapshot_of_contract_and_validator(self):
        module = load_module()

        loaded = module.load_frozen_contract(CONTRACT_PATH, VALIDATOR_PATH)

        self.assertEqual(
            loaded["contract"]["campaign_id"],
            "qwen35-0.8b-cross-runtime-formal-v3-20260826",
        )
        self.assertEqual(
            loaded["contract_file"]["sha256"],
            "caa46b953f0abc0e58ffaa3725257fbbfabe4be49ca84aa0c523de8a16efb301",
        )
        self.assertEqual(
            loaded["validator_file"]["sha256"],
            "9e4586d60839180cf7be55b63f53ac0c9dea811149ee3b4b1c9c9ccd6f9a11cf",
        )
        self.assertTrue(loaded["validation"]["valid"])

    def test_sample_receipt_rejects_legacy_mode_hash_only_and_wrong_timing_or_custody(self):
        module = load_module()
        contract = contract_fixture()
        slot = module.declared_schedule()[7]
        nonce = "2" * 64
        original = sample_receipt_fixture(module, contract, slot, nonce)
        mutations = (
            ("legacy mode", lambda value: value.__setitem__("mode", "free")),
            (
                "hash only",
                lambda value: value["workload"].__setitem__("generated_token_ids", None),
            ),
            (
                "raw prompt",
                lambda value: value["workload"]["prompt_token_ids"].__setitem__(0, 1),
            ),
            (
                "logits endpoint",
                lambda value: value["timing"].__setitem__(
                    "common_token_ready_boundary", "logits-ready"
                ),
            ),
            (
                "timing arithmetic",
                lambda value: value["timing"].__setitem__("tpot_ms", 1.0),
            ),
            (
                "runner custody",
                lambda value: value["custody"]["runner"].__setitem__(
                    "sha256", "0" * 64
                ),
            ),
        )
        for label, mutate in mutations:
            with self.subTest(label=label):
                receipt = copy.deepcopy(original)
                mutate(receipt)
                with self.assertRaises(module.ReceiptError):
                    module.validate_sample_receipt(
                        receipt, slot, nonce, contract, custody_fixture(slot["arm"])
                    )

    def test_teacher_receipt_rejects_prompt13_or_any_argmax_divergence(self):
        module = load_module()
        contract = contract_fixture()
        original = teacher_receipt_fixture(module, contract, "L")
        mutations = (
            (
                "prompt13",
                lambda value: value.__setitem__(
                    "prefill_prompt_token_ids",
                    contract["workload_contracts"]["shared_prompt"]["token_ids"],
                ),
            ),
            (
                "teacher input",
                lambda value: value["teacher_input_token_ids"].__setitem__(0, 0),
            ),
            (
                "observed argmax",
                lambda value: value["observed_argmax_token_ids"].__setitem__(17, 7),
            ),
            ("legacy mode", lambda value: value.__setitem__("mode", "teacher")),
        )
        for label, mutate in mutations:
            with self.subTest(label=label):
                receipt = copy.deepcopy(original)
                mutate(receipt)
                with self.assertRaises(module.ReceiptError):
                    module.validate_teacher_receipt(
                        receipt,
                        "L",
                        contract,
                        {"size_bytes": 41, "sha256": "8" * 64},
                        {"size_bytes": 43, "sha256": "9" * 64},
                        custody_fixture("L"),
                    )

    def test_file_custody_hashes_one_nofollow_file_descriptor_snapshot(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "artifact.bin"
            path.write_bytes(b"immutable fixture artifact")
            expected = {
                "absolute_path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }

            receipt = module.file_custody(expected)

            self.assertEqual(receipt["sha256"], expected["sha256"])
            self.assertTrue(receipt["identity_before_after_equal"])
            symlink = Path(directory) / "artifact-link"
            os.symlink(path, symlink)
            with self.assertRaisesRegex(module.CampaignError, "NOFOLLOW"):
                module.file_custody({**expected, "absolute_path": str(symlink)})

    def test_runtime_stdout_must_be_exactly_one_strict_json_line(self):
        module = load_module()
        self.assertEqual(module.parse_single_json_line(b'{"ok":true}\n'), {"ok": True})
        invalid = (
            b'{"ok":true}',
            b'{"ok":true}\n\n',
            b'{"ok":true}\n{"extra":true}\n',
            b'{"ok":true,"ok":false}\n',
            b'{"value":NaN}\n',
            b'\xff\n',
        )
        for raw in invalid:
            with self.subTest(raw=raw):
                with self.assertRaises(module.ReceiptError):
                    module.parse_single_json_line(raw)

    def test_host_receipt_requires_exact_mac_and_every_quiet_snapshot(self):
        module = load_module()
        contract = contract_fixture()
        receipt = host_receipt_fixture(module, contract)

        validated = module.validate_host_receipt(receipt, "preflight", contract)

        self.assertIs(validated, receipt)
        noisy = copy.deepcopy(receipt)
        noisy["snapshots"][3][
            "maximum_single_nonallowlisted_process_cpu_percent"
        ] = 10.1
        noisy["snapshots"][3]["passed"] = False
        noisy["passed"] = False
        with self.assertRaisesRegex(module.CampaignError, "quiet-host"):
            module.validate_host_receipt(noisy, "preflight", contract)

    def test_host_receipt_rejects_transient_system_state_or_non_250ms_cpu_proof(self):
        module = load_module()
        contract = contract_fixture()
        original = host_receipt_fixture(module, contract, "continuous")
        mutations = (
            (
                "transient thermal warning",
                lambda value: value["snapshots"][0].__setitem__(
                    "thermal_warning", True
                ),
            ),
            (
                "diluted CPU window",
                lambda value: value["snapshots"][0].__setitem__(
                    "cpu_percent_window_ms", 400.0
                ),
            ),
            (
                "incomplete CPU inventory",
                lambda value: value["snapshots"][0].__setitem__(
                    "cpu_window_proof_complete", False
                ),
            ),
            (
                "transient system swap",
                lambda value: value["snapshots"][0].__setitem__(
                    "system_swap_used_bytes", 1
                ),
            ),
        )
        for label, mutate in mutations:
            with self.subTest(label=label):
                receipt = copy.deepcopy(original)
                mutate(receipt)
                with self.assertRaises(module.CampaignError):
                    module.validate_host_receipt(receipt, "continuous", contract)

    def test_quiet_probe_never_footprints_an_active_pid_absent_from_inventory(self):
        module = load_module()
        contract = contract_fixture()
        probe = object.__new__(module.MacQuietHostProbe)
        probe.contract = contract
        probe.host = contract["scope"]["host"]
        probe._orchestrator_pid = 1
        probe._executable = {
            "absolute_path": "/fixture/python",
            "sha256": "a" * 64,
        }
        probe._argv_sha256 = "b" * 64
        orchestrator = {
            "pid": 1,
            "ppid": 0,
            "process_group_id": 1,
            "process_start_time": "fixture-start",
            "command": "python",
            "cpu_time_ns": 100,
        }
        probe._previous_inventory = {1: orchestrator}
        probe._previous_allowed = {(1, "fixture-start")}
        probe._previous_monotonic_ns = 1
        calls = []
        old_inventory = module._process_inventory
        old_swapped = module._process_swapped_bytes
        old_monotonic = module.time.monotonic_ns
        old_load = module.os.getloadavg
        try:
            module._process_inventory = lambda: {
                1: {**orchestrator, "cpu_time_ns": 101}
            }

            def swapped(pid):
                calls.append(pid)
                return 0

            module._process_swapped_bytes = swapped
            module.time.monotonic_ns = lambda: 250_000_001
            module.os.getloadavg = lambda: (0.0, 0.0, 0.0)
            snapshot = probe.snapshot(
                0,
                {
                    "pid": 42,
                    "arm": "L",
                    "executable_and_library_hashes": {},
                },
            )
        finally:
            module._process_inventory = old_inventory
            module._process_swapped_bytes = old_swapped
            module.time.monotonic_ns = old_monotonic
            module.os.getloadavg = old_load

        self.assertEqual(calls, [1])
        self.assertFalse(snapshot["active_runtime_root_present"])

    def test_runtime_receipt_fails_closed_without_a_live_zero_swap_pid_proof(self):
        module = load_module()
        monitor = object.__new__(module.ContinuousHostMonitor)
        monitor._lock = module.threading.Lock()
        monitor._completed_runtime_proofs = {
            42: {
                "pid": 42,
                "arm": "L",
                "swap_zero_proven": False,
                "observed_process_identity": None,
            }
        }
        monitor._accepted_runtime_swap_proofs = []

        with self.assertRaisesRegex(module.CampaignError, "exited before"):
            monitor.confirm_runtime_receipt(42)

        monitor._completed_runtime_proofs[43] = {
            "pid": 43,
            "arm": "AN",
            "swap_zero_proven": True,
            "observed_process_identity": {
                "pid": 43,
                "process_start_time": "fixture-start",
                "swapped_bytes": 0,
            },
        }
        proof = monitor.confirm_runtime_receipt(43)
        self.assertEqual(proof["pid"], 43)
        self.assertEqual(monitor._accepted_runtime_swap_proofs, [proof])

    def test_git_custody_uses_live_ls_remote_and_binds_the_tracked_contract_blob(self):
        module = load_module()
        contract = contract_fixture()
        head = "1" * 40
        tree = "2" * 40
        contract_commit = "3" * 40
        contract_tree = "4" * 40
        blob = "5" * 40
        contract_bytes = CONTRACT_PATH.read_bytes()
        calls = []

        def git_fixture(argv, cwd, timeout_seconds, env=None):
            del cwd, timeout_seconds
            self.assertEqual(env, module.git_custody_environment())
            calls.append(argv)
            args = argv[1:]
            if args == ["config", "--local", "--name-only", "--null", "--list"]:
                stdout = (
                    b"core.repositoryformatversion\0core.filemode\0"
                    b"remote.origin.url\0remote.origin.fetch\0"
                )
                code = 0
            elif args == ["config", "--get", "remote.origin.url"]:
                stdout = b"https://github.com/qhy991/ApxInf.git\n"
                code = 0
            elif args == ["rev-parse", "--show-object-format"]:
                stdout, code = b"sha1\n", 0
            elif args == ["rev-parse", "--verify", "HEAD^{commit}"]:
                stdout, code = f"{head}\n".encode(), 0
            elif args == [
                "rev-parse",
                "--verify",
                "refs/remotes/origin/main^{commit}",
            ]:
                stdout, code = f"{head}\n".encode(), 0
            elif args == [
                "ls-remote",
                "--exit-code",
                "https://github.com/qhy991/ApxInf.git",
                "refs/heads/main",
            ]:
                stdout, code = f"{head}\trefs/heads/main\n".encode(), 0
            elif args == ["status", "--porcelain=v1", "-z", "--untracked-files=all"]:
                stdout, code = b"", 0
            elif args == ["rev-parse", "--verify", "HEAD^{tree}"]:
                stdout, code = f"{tree}\n".encode(), 0
            elif args == [
                "log",
                "-1",
                "--format=%H",
                "--",
                "configs/qwen35-0.8b-cross-runtime-formal-v3.json",
            ]:
                stdout, code = f"{contract_commit}\n".encode(), 0
            elif args == [
                "rev-parse",
                "--verify",
                f"{contract_commit}^{{tree}}",
            ]:
                stdout, code = f"{contract_tree}\n".encode(), 0
            elif args == [
                "ls-tree",
                "-z",
                "HEAD",
                "--",
                "configs/qwen35-0.8b-cross-runtime-formal-v3.json",
            ]:
                stdout, code = (
                    f"100644 blob {blob}\tconfigs/qwen35-0.8b-cross-runtime-formal-v3.json\0".encode(),
                    0,
                )
            elif args == ["cat-file", "blob", blob]:
                stdout, code = contract_bytes, 0
            elif args == [
                "merge-base",
                "--is-ancestor",
                contract_commit,
                head,
            ]:
                stdout, code = b"", 0
            else:
                raise AssertionError(f"unexpected git command: {argv}")
            return {"returncode": code, "stdout": stdout, "stderr": b""}

        receipt = module.collect_git_custody(
            ROOT,
            contract,
            {"contract": "configs/qwen35-0.8b-cross-runtime-formal-v3.json"},
            command_runner=git_fixture,
        )

        self.assertEqual(receipt["head_commit"], head)
        self.assertEqual(receipt["ls_remote_live_oid"], head)
        self.assertEqual(receipt["tracked_files"]["contract"]["blob_oid"], blob)
        self.assertIn(
            [
                "/usr/bin/git",
                "ls-remote",
                "--exit-code",
                "https://github.com/qhy991/ApxInf.git",
                "refs/heads/main",
            ],
            calls,
        )

    def test_git_custody_rejects_repository_transport_redirect_or_proxy_config(self):
        module = load_module()

        for dangerous_key in (
            "url.file:///fixture/.insteadof",
            "http.https://github.com/.proxy",
            "remote.origin.proxy",
            "core.gitproxy",
            "include.path",
            "protocol.file.allow",
        ):
            with self.subTest(dangerous_key=dangerous_key):
                def runner(argv, cwd, timeout_seconds, env=None):
                    del cwd, timeout_seconds
                    self.assertEqual(env, module.git_custody_environment())
                    self.assertEqual(
                        argv,
                        [
                            "/usr/bin/git",
                            "config",
                            "--local",
                            "--name-only",
                            "--null",
                            "--list",
                        ],
                    )
                    return {
                        "returncode": 0,
                        "stdout": dangerous_key.encode() + b"\0",
                        "stderr": b"",
                    }

                with self.assertRaisesRegex(module.CampaignError, "transport"):
                    module.reject_git_transport_overrides(ROOT, runner)

    def test_execution_plan_binds_only_explicit_native_v3_commands_and_named_deployments(self):
        module = load_module()
        contract = contract_fixture()
        plan = execution_plan_fixture(module, contract)

        validated = module.validate_execution_plan(plan, contract)

        self.assertIs(validated, plan)
        legacy_contract = copy.deepcopy(contract)
        legacy_contract["runtime_custody"]["pinned_llama_cpp_core"][
            "runner_binary_size_bytes"
        ] = 6_499_056
        legacy_contract["runtime_custody"]["pinned_llama_cpp_core"][
            "runner_binary_sha256"
        ] = module.REJECTED_LEGACY_LLAMA_RUNNER_SHA256
        legacy_plan = execution_plan_fixture(module, legacy_contract)
        with self.assertRaisesRegex(module.CampaignError, "legacy v2 llama.cpp"):
            module.validate_execution_plan(legacy_plan, legacy_contract)
        legacy = copy.deepcopy(plan)
        legacy["commands"]["L"]["argv"][-1] = "free"
        with self.assertRaisesRegex(module.CampaignError, "native-v3-free"):
            module.validate_execution_plan(legacy, contract)
        command_mutations = (
            (
                "AN missing source lock",
                lambda value: value["commands"]["AN"]["argv"].__delitem__(
                    slice(-2, None)
                ),
            ),
            (
                "AN flag reordering",
                lambda value: value["commands"]["AN"].__setitem__(
                    "argv",
                    [
                        value["commands"]["AN"]["argv"][0],
                        *value["commands"]["AN"]["argv"][3:5],
                        *value["commands"]["AN"]["argv"][1:3],
                        *value["commands"]["AN"]["argv"][5:7],
                    ],
                ),
            ),
            (
                "AN unknown flag",
                lambda value: value["commands"]["AN"]["argv"].extend(
                    ["--unknown", "1"]
                ),
            ),
            (
                "L unbound model",
                lambda value: value["commands"]["L"]["argv"].__setitem__(
                    2, "/fixture/other-model"
                ),
            ),
            (
                "L duplicate threads",
                lambda value: value["commands"]["L"]["argv"].extend(
                    ["--threads", "4"]
                ),
            ),
            (
                "L flag reordering",
                lambda value: value["commands"]["L"]["argv"].__setitem__(
                    slice(1, 5),
                    value["commands"]["L"]["argv"][3:5]
                    + value["commands"]["L"]["argv"][1:3],
                ),
            ),
        )
        for label, mutate in command_mutations:
            with self.subTest(label=label):
                mutated = copy.deepcopy(plan)
                mutate(mutated)
                with self.assertRaisesRegex(module.CampaignError, "argv"):
                    module.validate_execution_plan(mutated, contract)

    def test_an_runtime_source_commit_is_a_real_ancestor_with_a_stable_tree(self):
        module = load_module()
        contract = contract_fixture()
        plan = execution_plan_fixture(module, contract)
        source_commit = plan["artifacts"]["AN"]["runtime_source_commit"]
        live_head = "8" * 40
        source_tree = "7" * 40
        custody = {
            "head_commit": live_head,
            "head_tree": "1" * 40,
            "object_format": "sha1",
            "worktree_clean": True,
        }

        def runner(argv, cwd, timeout_seconds, env=None):
            del cwd, timeout_seconds
            self.assertEqual(env, module.git_custody_environment())
            args = argv[1:]
            if args == [
                "rev-parse",
                "--verify",
                f"{source_commit}^{{commit}}",
            ]:
                return {
                    "returncode": 0,
                    "stdout": f"{source_commit}\n".encode(),
                    "stderr": b"",
                }
            if args == [
                "rev-parse",
                "--verify",
                f"{source_commit}^{{tree}}",
            ]:
                return {
                    "returncode": 0,
                    "stdout": f"{source_tree}\n".encode(),
                    "stderr": b"",
                }
            if args == [
                "merge-base",
                "--is-ancestor",
                source_commit,
                live_head,
            ]:
                return {"returncode": 0, "stdout": b"", "stderr": b""}
            raise AssertionError(argv)

        receipt = module.validate_an_campaign_commit(
            plan, custody, ROOT, runner
        )

        self.assertEqual(receipt["campaign_commit"], source_commit)
        self.assertEqual(receipt["campaign_tree"], source_tree)

        def nonexistent(argv, cwd, timeout_seconds, env=None):
            del argv, cwd, timeout_seconds, env
            return {"returncode": 1, "stdout": b"", "stderr": b"missing"}

        with self.assertRaisesRegex(module.CampaignError, "git rev-parse"):
            module.validate_an_campaign_commit(
                plan, custody, ROOT, nonexistent
            )

        def nonancestor(argv, cwd, timeout_seconds, env=None):
            result = runner(argv, cwd, timeout_seconds, env)
            if argv[1] == "merge-base":
                return {"returncode": 1, "stdout": b"", "stderr": b""}
            return result

        with self.assertRaisesRegex(module.CampaignError, "not an ancestor"):
            module.validate_an_campaign_commit(
                plan, custody, ROOT, nonancestor
            )

    def test_an_raw_source_files_bind_to_the_ancestor_tree_and_live_blobs(self):
        module = load_module()
        contract = contract_fixture()
        plan = execution_plan_fixture(module, contract)
        source_commit = plan["artifacts"]["AN"]["runtime_source_commit"]
        live_head = "8" * 40
        source_tree = "7" * 40
        git_custody = {
            "head_commit": live_head,
            "head_tree": "9" * 40,
            "object_format": "sha1",
            "worktree_clean": True,
        }
        source_start, source_end = an_source_custody_fixture()

        def relocate(value):
            if isinstance(value, dict):
                return {key: relocate(item) for key, item in value.items()}
            if isinstance(value, list):
                return [relocate(item) for item in value]
            if isinstance(value, str) and value.startswith("/fixture/"):
                return str(ROOT / value[len("/fixture/") :])
            return value

        source_start = relocate(source_start)
        source_end = relocate(source_end)
        raw_custody = {
            **copy.deepcopy(plan["artifacts"]["AN"]),
            "runtime_source_commit": source_commit,
            "source_custody_start": source_start,
            "source_custody_end": source_end,
        }
        claim_locations = [
            (source_start["profile"], "file_size", "file_sha256"),
            (source_start["source_lock"], "file_size", "file_sha256"),
            (source_start["sources"]["gate"], "size", "sha256"),
            *[
                (entry, "size", "sha256")
                for entry in source_start["sources"][
                    "rust_and_bridge_sources"
                ].values()
            ],
            *[
                (entry, "size", "sha256")
                for entry in source_start["sources"][
                    "compiled_metal_shader_sources"
                ].values()
            ],
        ]
        payloads = {}
        for entry, size_field, hash_field in claim_locations:
            payload = ("blob:" + entry["path"]).encode()
            entry[size_field] = len(payload)
            entry[hash_field] = hashlib.sha256(payload).hexdigest()
            payloads[entry["path"].removeprefix(str(ROOT) + "/")] = payload
        source_end["deployment_profile"]["size"] = source_start["profile"][
            "file_size"
        ]
        source_end["deployment_profile"]["sha256"] = source_start["profile"][
            "file_sha256"
        ]
        source_end["source_lock"]["size"] = source_start["source_lock"][
            "file_size"
        ]
        source_end["source_lock"]["sha256"] = source_start["source_lock"][
            "file_sha256"
        ]
        source_end["gate"] = copy.deepcopy(source_start["sources"]["gate"])
        source_end["rust_and_bridge_sources"] = copy.deepcopy(
            source_start["sources"]["rust_and_bridge_sources"]
        )
        source_end["compiled_metal_shader_sources"] = copy.deepcopy(
            source_start["sources"]["compiled_metal_shader_sources"]
        )
        blob_oids = {
            path: hashlib.sha1(payload).hexdigest()
            for path, payload in payloads.items()
        }

        def runner(argv, cwd, timeout_seconds, env=None):
            del cwd, timeout_seconds
            self.assertEqual(env, module.git_custody_environment())
            args = argv[1:]
            if args == [
                "rev-parse",
                "--verify",
                f"{source_commit}^{{commit}}",
            ]:
                stdout, code = f"{source_commit}\n".encode(), 0
            elif args == [
                "rev-parse",
                "--verify",
                f"{source_commit}^{{tree}}",
            ]:
                stdout, code = f"{source_tree}\n".encode(), 0
            elif args == [
                "merge-base",
                "--is-ancestor",
                source_commit,
                live_head,
            ]:
                stdout, code = b"", 0
            elif args[:2] == ["ls-tree", "-z"] and len(args) == 5:
                revision, path = args[2], args[4]
                self.assertIn(revision, (source_commit, live_head))
                oid = blob_oids[path]
                stdout, code = (
                    f"100644 blob {oid}\t{path}\0".encode(),
                    0,
                )
            elif args[:2] == ["cat-file", "blob"]:
                oid = args[2]
                path = next(
                    path for path, candidate in blob_oids.items() if candidate == oid
                )
                stdout, code = payloads[path], 0
            elif args == [
                "diff",
                "--name-only",
                "-z",
                "--no-renames",
                source_commit,
                live_head,
                "--",
            ]:
                stdout, code = b"", 0
            else:
                raise AssertionError(argv)
            return {"returncode": code, "stdout": stdout, "stderr": b""}

        binding = module.bind_an_source_custodies_to_git(
            ROOT,
            plan,
            git_custody,
            [raw_custody],
            command_runner=runner,
        )
        self.assertEqual(binding["campaign_tree"], source_tree)
        self.assertEqual(binding["source_file_count"], len(payloads))

        with self.assertRaisesRegex(module.CampaignError, "tree drifted"):
            module.bind_an_source_custodies_to_git(
                ROOT,
                plan,
                git_custody,
                [raw_custody],
                command_runner=runner,
                expected_source_tree="9" * 40,
            )

        drift = copy.deepcopy(raw_custody)
        drift["source_custody_start"]["sources"]["gate"]["sha256"] = (
            "0" * 64
        )
        drift["source_custody_end"]["gate"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(module.ReceiptError, "blob/size/hash"):
            module.bind_an_source_custodies_to_git(
                ROOT,
                plan,
                git_custody,
                [drift],
                command_runner=runner,
            )

    def test_llama_raw_v3_adapter_proves_the_real_runner_seam_before_wrapping(self):
        module = load_module()
        contract = contract_fixture()
        custody = execution_plan_fixture(module, contract)["artifacts"]["L"]
        slot = module.declared_schedule()[7]
        nonce = "3" * 64
        raw = llama_raw_free_fixture(contract, custody)

        wrapped = module.adapt_llama_free_receipt(
            raw,
            slot,
            nonce,
            contract,
            custody,
            artifact_observation_fixture(custody),
        )

        self.assertEqual(wrapped["format"], module.SAMPLE_FORMAT)
        self.assertEqual(wrapped["mode"], "native-v3-free")
        self.assertEqual(
            wrapped["external_receipt"]["schema"],
            "apxinf.llama-cpp.raw-token-diagnostic.v3",
        )
        module.validate_sample_receipt(wrapped, slot, nonce, contract, custody)

    def test_llama_raw_adapter_rejects_legacy_quality_timing_placement_and_custody_drift(self):
        module = load_module()
        contract = contract_fixture()
        custody = execution_plan_fixture(module, contract)["artifacts"]["L"]
        slot = module.declared_schedule()[7]
        nonce = "4" * 64
        original = llama_raw_free_fixture(contract, custody)
        mutations = (
            (
                "unexpected schema field",
                lambda value: value.__setitem__("unexpected", True),
            ),
            ("legacy schema", lambda value: value.__setitem__("schema", "apxinf.llama-cpp.raw-token-diagnostic.v2")),
            ("failed", lambda value: value.__setitem__("ok", False)),
            ("teacher mode", lambda value: value.__setitem__("mode", "native-v3-teacher")),
            ("context142", lambda value: value["parameters"].__setitem__("n_ctx_requested", 142)),
            ("raw12", lambda value: value["contract"]["prompt_token_ids"].pop()),
            ("trajectory", lambda value: value["output"]["token_ids"].__setitem__(10, 1)),
            (
                "nonmonotonic",
                lambda value: value["output"]["token_ready_elapsed_ns"].__setitem__(
                    1, value["output"]["token_ready_elapsed_ns"][0]
                ),
            ),
            ("execution proof", lambda value: value["post_measurement_execution_proof"].__setitem__("passed", False)),
            ("placement", lambda value: value["placement_attestation"].__setitem__("layers_on_selected_gpu", 23)),
            ("backend scan", lambda value: value["backend"].__setitem__("dynamic_backend_scan_invoked", True)),
            (
                "model identity",
                lambda value: value["model"].__setitem__(
                    "file_identity_after_load",
                    {**value["model"]["file_identity_after_load"], "inode": 9},
                ),
            ),
        )
        for label, mutate in mutations:
            with self.subTest(label=label):
                receipt = copy.deepcopy(original)
                mutate(receipt)
                with self.assertRaises(module.ReceiptError):
                    module.adapt_llama_free_receipt(
                        receipt,
                        slot,
                        nonce,
                        contract,
                        custody,
                        artifact_observation_fixture(custody),
                    )

    def test_llama_adapter_rejects_unbound_model_identity_or_synthetic_library_closure(self):
        module = load_module()
        contract = contract_fixture()
        custody = execution_plan_fixture(module, contract)["artifacts"]["L"]
        observation = artifact_observation_fixture(custody)
        slot = module.declared_schedule()[1]
        nonce = "6" * 64
        original = llama_raw_free_fixture(contract, custody)
        mutations = (
            (
                "preflight inode",
                lambda value: value["model"]["file_identity_start"].__setitem__(
                    "inode", 999
                ),
            ),
            (
                "closure plan-only hash",
                lambda value: value.pop("runtime_custody"),
            ),
            (
                "closure start/end",
                lambda value: value["runtime_custody"][
                    "loaded_non_system_library_closure_end"
                ].append({"absolute_path": "/fixture/injected.dylib"}),
            ),
            (
                "free reuse counter",
                lambda value: value["llama_perf"]["context"].__setitem__(
                    "n_reused", 0
                ),
            ),
        )
        for label, mutate in mutations:
            with self.subTest(label=label):
                raw = copy.deepcopy(original)
                mutate(raw)
                if label == "preflight inode":
                    raw["model"]["file_identity_after_load"] = copy.deepcopy(
                        raw["model"]["file_identity_start"]
                    )
                    raw["model"]["file_identity_before_receipt"] = copy.deepcopy(
                        raw["model"]["file_identity_start"]
                    )
                with self.assertRaises(module.ReceiptError):
                    module.adapt_llama_free_receipt(
                        raw,
                        slot,
                        nonce,
                        contract,
                        custody,
                        observation,
                    )

    def test_an_raw_adapter_requires_the_live_fused_path_and_custody_hashes(self):
        module = load_module()
        contract = contract_fixture()
        slot = module.declared_schedule()[0]
        nonce = "5" * 64
        receipt, custody = an_raw_free_fixture(module, contract, slot, nonce)

        validated = module.validate_an_free_receipt(
            receipt,
            slot,
            nonce,
            contract,
            custody,
            an_source_binding_fixture(module, receipt["custody"]),
        )

        self.assertIs(validated, receipt)
        mutations = (
            (
                "fused path",
                lambda value: value["final_path"]["path_checks"].__setitem__(
                    "aggregate_ledger_valid", False
                ),
            ),
            (
                "thread policy",
                lambda value: value["custody"]["thread_policy_runtime"].__setitem__(
                    "logical_cpu_count", 8
                ),
            ),
            (
                "dyld closure",
                lambda value: value["custody"][
                    "loaded_non_system_library_closure_end"
                ].append({"absolute_path": "/fixture/lazy.dylib"}),
            ),
            (
                "clock identity",
                lambda value: value["timing"].__setitem__(
                    "clock_identity", "std::chrono::steady_clock"
                ),
            ),
            (
                "empty source custody",
                lambda value: value["custody"].update(
                    {"source_custody_start": {}, "source_custody_end": {}}
                ),
            ),
            (
                "source identity drift",
                lambda value: value["custody"]["source_custody_end"][
                    "binary"
                ].__setitem__("inode", 99),
            ),
        )
        for label, mutate in mutations:
            with self.subTest(label=label):
                broken = copy.deepcopy(receipt)
                mutate(broken)
                with self.assertRaises(module.ReceiptError):
                    module.validate_an_free_receipt(
                        broken,
                        slot,
                        nonce,
                        contract,
                        custody,
                        an_source_binding_fixture(
                            module, receipt["custody"]
                        ),
                    )

    def test_teacher_preflight_combines_independent_cpu_reference_and_an_observed_files(self):
        module = load_module()
        contract = contract_fixture()
        custody = custody_fixture("AN")
        reference, observed = native_teacher_raw_fixtures(
            module, contract, custody
        )
        reference_file = {"size_bytes": 41, "sha256": "8" * 64}
        runtime_file = {"size_bytes": 43, "sha256": "9" * 64}

        admission = module.build_teacher_admission_receipt(
            "AN",
            reference,
            observed,
            reference_file,
            runtime_file,
            contract,
            custody,
            expected_artifact_observation=artifact_observation_fixture(custody),
        )

        self.assertEqual(admission["format"], module.TEACHER_FORMAT)
        self.assertEqual(admission["mismatch_positions"], [])
        module.validate_teacher_receipt(
            admission,
            "AN",
            contract,
            reference_file,
            runtime_file,
            custody,
        )

    def test_teacher_preflight_adapts_llama_raw12_plus_128_teacher_schema(self):
        module = load_module()
        contract = contract_fixture()
        an_custody = custody_fixture("AN")
        reference, _ = native_teacher_raw_fixtures(
            module, contract, an_custody
        )
        l_custody = execution_plan_fixture(module, contract)["artifacts"]["L"]
        runtime = llama_teacher_raw_fixture(contract, l_custody)
        reference_file = {"size_bytes": 41, "sha256": "8" * 64}
        runtime_file = {"size_bytes": 43, "sha256": "9" * 64}

        admission = module.build_teacher_admission_receipt(
            "L",
            reference,
            runtime,
            reference_file,
            runtime_file,
            contract,
            l_custody,
            reference_expected_custody=an_custody,
            expected_artifact_observation=artifact_observation_fixture(l_custody),
        )

        self.assertEqual(admission["arm"], "L")
        self.assertEqual(admission["observed_argmax_token_ids"], canonical_free_ids(contract))

    def test_atomic_json_create_is_create_new_and_replace_never_exposes_partial_json(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "raw.json"

            module.atomic_create_json(path, {"generation": 0})

            self.assertEqual(path.read_bytes(), b'{"generation":0}\n')
            with self.assertRaisesRegex(module.CampaignError, "already exists"):
                module.atomic_create_json(path, {"generation": 99})
            module.atomic_replace_json(path, {"generation": 1, "passed": True})
            self.assertEqual(
                module.parse_single_json_line(path.read_bytes()),
                {"generation": 1, "passed": True},
            )
            self.assertEqual(list(Path(directory).glob(".formal-v3-*.tmp")), [])

    def test_run_persists_an_unattempted_partial_when_context_loading_fails_after_marker(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as directory:
            repository_root = Path(directory).resolve()
            plan = execution_plan_fixture(module, contract_fixture())
            plan["repository_root"] = str(repository_root)
            plan["plan_repository_path"] = "configs/formal-v3-plan.json"
            plan["raw_output_path"] = str(repository_root / "raw-partial.json")
            plan_path = repository_root / plan["plan_repository_path"]
            plan_path.parent.mkdir(parents=True)
            plan_bytes = json.dumps(
                plan, ensure_ascii=False, separators=(",", ":")
            ).encode("utf-8")
            plan_path.write_bytes(plan_bytes)

            validator_path = repository_root / plan["validator_repository_path"]
            validator_path.parent.mkdir(parents=True)
            validator_path.write_bytes(VALIDATOR_PATH.read_bytes())
            contract_path = repository_root / plan["contract_repository_path"]
            contract_path.parent.mkdir(parents=True, exist_ok=True)
            contract_path.write_bytes(b'{"corrupted":true}\n')

            marker_path = repository_root / plan["marker_repository_path"]
            marker_path.parent.mkdir(parents=True)
            marker = {
                "format": module.MARKER_FORMAT,
                "schema_version": 3,
                "campaign_id": "qwen35-0.8b-cross-runtime-formal-v3-20260826",
                "subcampaign_id": (
                    "qwen35-0.8b-native-apxinf-vs-llamacpp-formal-v3-20260826"
                ),
                "edge_id": module.EDGE_ID,
                "plan_repository_path": plan["plan_repository_path"],
                "plan_blob_size_bytes": len(plan_bytes),
                "plan_blob_sha256": hashlib.sha256(plan_bytes).hexdigest(),
                "marker_repository_path": plan["marker_repository_path"],
                "sampling_state_at_marker_creation": {
                    "generation_requests": 0,
                    "warmup_samples": 0,
                    "timed_samples": 0,
                },
                "pre_marker_admission": {"all_passed": True},
            }
            marker_path.write_bytes(
                json.dumps(marker, separators=(",", ":")).encode("utf-8") + b"\n"
            )

            result = module.run_campaign(
                plan_path,
                command_runner=lambda *args, **kwargs: self.fail(
                    "Git or runtime command must not run before context admission"
                ),
                host_gate_collector=lambda *args, **kwargs: self.fail(
                    "host sampling must not run before context admission"
                ),
            )

            raw_path = Path(plan["raw_output_path"])
            self.assertEqual(
                module.parse_single_json_line(raw_path.read_bytes()), result
            )
            self.assertEqual(
                result["status"], "CONSUMED_FIRST_POST_MARKER_FAILURE"
            )
            self.assertEqual(result["failures"][0]["stage"], "execution-context-load")
            schedule = result["schedule_receipt"]
            self.assertEqual(schedule["attempted_count"], 0)
            self.assertEqual(schedule["remaining_unattempted_count"], 38)
            self.assertTrue(schedule["stopped_at_first_failure"])
            self.assertEqual(
                [entry["status"] for entry in schedule["slots"]],
                ["unattempted"] * 38,
            )

    def test_statistics_use_only_32_timed_samples_and_the_frozen_block_estimator(self):
        module = load_module()
        contract = contract_fixture()
        samples = []
        for slot in module.declared_schedule():
            receipt = sample_receipt_fixture(
                module, contract, slot, f"{slot['sequence_index']:064x}"
            )
            if slot["phase"] == "timed":
                receipt["timing"]["tpot_ms"] = 10.0
            samples.append(receipt)

        statistics = module.compute_native_statistics(samples, contract)

        self.assertEqual(statistics["timed_sample_count"], 32)
        self.assertEqual(statistics["timed_samples_per_arm"], {"AN": 16, "L": 16})
        self.assertEqual(statistics["block_log_ratios"], [0.0] * 8)
        self.assertEqual(statistics["point_ratio_A_over_L"], 1.0)
        self.assertEqual(statistics["decision"], "NAMED_DEPLOYMENTS_PRACTICALLY_EQUIVALENT_WITHIN_5_PERCENT")
        self.assertTrue(all(statistics["stability_gates"].values()))

    def test_exact_schedule_stops_once_and_atomically_preserves_failed_and_unattempted_slots(self):
        module = load_module()
        contract = contract_fixture()
        plan = execution_plan_fixture(module, contract)
        calls = []

        def collect(slot, nonce):
            calls.append(slot["sequence_index"])
            if slot["sequence_index"] == 5:
                raise module.RuntimeInvocationError(
                    "fixture runtime crash",
                    {
                        "returncode": -6,
                        "timed_out": False,
                        "stdout_size_bytes": 0,
                        "stderr_size_bytes": 13,
                    },
                )
            receipt = sample_receipt_fixture(module, contract, slot, nonce)
            receipt["custody"].update(copy.deepcopy(plan["artifacts"][slot["arm"]]))
            return receipt

        with tempfile.TemporaryDirectory() as directory:
            raw_path = Path(directory) / "partial.json"
            result = module.execute_formal_schedule(
                contract,
                plan,
                {
                    "contract_binding": {"fixture": True},
                    "git_custody": {"fixture": True},
                    "host_custody": {"preflight": host_receipt_fixture(module, contract)},
                    "artifact_custody": plan["artifacts"],
                    "parity_admission": {"AN": {"passed": True}, "L": {"passed": True}},
                    "blocker_resolution": {"all_resolved": True},
                },
                raw_path,
                sample_collector=collect,
                postflight_collector=lambda: {
                    "continuous": host_receipt_fixture(module, contract, "continuous"),
                    "postflight": host_receipt_fixture(module, contract, "postflight"),
                    "artifact_custody_end": plan["artifacts"],
                },
                nonce_factory=lambda slot: f"{slot['sequence_index']:064x}",
            )

            persisted = module.parse_single_json_line(raw_path.read_bytes())
            self.assertEqual(calls, list(range(6)))
            self.assertEqual(result, persisted)
            self.assertEqual(result["status"], "CONSUMED_FIRST_POST_MARKER_FAILURE")
            statuses = [entry["status"] for entry in result["schedule_receipt"]["slots"]]
            self.assertEqual(statuses[:5], ["accepted"] * 5)
            self.assertEqual(statuses[5], "failed")
            self.assertEqual(statuses[6:], ["unattempted"] * 32)
            self.assertEqual(result["schedule_receipt"]["slots"][5]["attempt_count"], 1)
            self.assertEqual(len(result["failures"]), 1)
            self.assertIsNone(result["statistics"])
            self.assertEqual(result["decision"]["label"], "UNRANKABLE")

    def test_exact_schedule_accepts_38_fresh_samples_then_runs_postflight_and_statistics(self):
        module = load_module()
        contract = contract_fixture()
        plan = execution_plan_fixture(module, contract)
        calls = []
        postflight_calls = []

        def collect(slot, nonce):
            calls.append((slot["sequence_index"], nonce))
            receipt = sample_receipt_fixture(module, contract, slot, nonce)
            receipt["custody"].update(copy.deepcopy(plan["artifacts"][slot["arm"]]))
            return receipt

        def postflight():
            postflight_calls.append(True)
            return {
                "continuous": host_receipt_fixture(module, contract, "continuous"),
                "postflight": host_receipt_fixture(module, contract, "postflight"),
                "artifact_custody_end": plan["artifacts"],
            }

        with tempfile.TemporaryDirectory() as directory:
            result = module.execute_formal_schedule(
                contract,
                plan,
                {
                    "contract_binding": {"fixture": True},
                    "git_custody": {"fixture": True},
                    "host_custody": {"preflight": host_receipt_fixture(module, contract)},
                    "artifact_custody": plan["artifacts"],
                    "parity_admission": {"AN": {"passed": True}, "L": {"passed": True}},
                    "blocker_resolution": {"all_resolved": True},
                },
                Path(directory) / "complete.json",
                sample_collector=collect,
                postflight_collector=postflight,
                nonce_factory=lambda slot: f"{slot['sequence_index']:064x}",
            )

        self.assertEqual([index for index, _ in calls], list(range(38)))
        self.assertEqual(len({nonce for _, nonce in calls}), 38)
        self.assertEqual(postflight_calls, [True])
        self.assertEqual(result["status"], "FORMAL_COMPLETE")
        self.assertEqual(result["schedule_receipt"]["accepted_count"], 38)
        self.assertEqual(result["statistics"]["timed_sample_count"], 32)
        self.assertEqual(result["decision"]["label"], "NAMED_DEPLOYMENTS_PRACTICALLY_EQUIVALENT_WITHIN_5_PERCENT")

    def test_prepare_helper_never_creates_marker_when_any_preflight_raises(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / "marker.json"

            def fail_preflight():
                raise module.CampaignError("teacher receipt drift")

            with self.assertRaisesRegex(module.CampaignError, "teacher receipt drift"):
                module.prepare_marker_after_preflight(marker, fail_preflight)
            self.assertFalse(marker.exists())

            created = module.prepare_marker_after_preflight(
                marker,
                lambda: {
                    "format": module.MARKER_FORMAT,
                    "schema_version": 3,
                    "pre_marker_admission": {"all_passed": True},
                },
            )
            self.assertTrue(marker.exists())
            self.assertEqual(created["pre_marker_admission"]["all_passed"], True)
            with self.assertRaisesRegex(module.CampaignError, "already exists"):
                module.prepare_marker_after_preflight(marker, lambda: created)

    def test_cli_self_test_is_fixture_only_and_exercises_complete_and_failure_paths(self):
        completed = subprocess.run(
            [sys.executable, "-I", "-B", str(MODULE_PATH), "self-test"],
            cwd=ROOT,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
            check=False,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr.decode())
        self.assertEqual(completed.stderr, b"")
        receipt = json.loads(completed.stdout)
        self.assertTrue(receipt["passed"])
        self.assertEqual(receipt["complete_path_invocations"], 38)
        self.assertEqual(receipt["failure_path_invocations"], 4)
        self.assertTrue(receipt["network_used"] is False)
        self.assertTrue(receipt["model_process_used"] is False)


if __name__ == "__main__":
    unittest.main()
