//! Single-process, fixed-schedule production A/C continuation gate for the
//! explicit Qwen3.5 MLP→Stack3 boundary body + fused tail-head v1 path.
//! This campaign-only example exposes no arbitrary arm or order selector and
//! is not reachable from CLI, AutoModel, registry, or a default constructor.

#[path = "support/qwen35_boundary_tail_head_v1_gate_evidence.rs"]
mod gate_evidence;

use std::collections::HashMap;
use std::ffi::{CString, OsString};
use std::fs::{self, File};
use std::io::Write;
use std::os::fd::{AsRawFd, FromRawFd};
use std::os::unix::ffi::OsStrExt;
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::{error::Error, time::Instant};

use serde_json::{json, Value};

use apxinf_core::{Device, Tensor};
use apxinf_model::qwen35::general::{
    Qwen35MetalW8MlpStack3BoundaryTailHeadV1AggregateLedger,
    Qwen35MetalW8MlpStack3BoundaryTailHeadV1Stats,
};
use apxinf_model::{GeneralQwen35, LlmInput, LlmTrait, Qwen35Config, Qwen35LayerType};
use apxinf_tokenizer::{ChatMessage, Tokenizer};

const CAMPAIGN_FORMAT: &str = "apxinf-qwen35-gdn-core-fused-v1-production-ac-v1-raw-campaign-v1";
const CAMPAIGN_SENTINEL_FORMAT: &str =
    "apxinf-qwen35-gdn-core-fused-v1-production-ac-v1-campaign-start-v1";
const PREDECLARATION_FORMAT: &str =
    "apxinf-qwen35-gdn-core-fused-v1-production-ac-v1-predeclared-gate-v1";
const SOURCE_LOCK_FORMAT: &str = "apxinf-hf-source-lock-v1";
const REPO_ID: &str = "Qwen/Qwen3.5-0.8B";
const LOCKED_REVISION: &str = "2fc06364715b967f1860aea9cf38778875588b17";
const LOCKED_CHECKPOINT: &str = "model.safetensors-00001-of-00001.safetensors";
const LOCKED_CHECKPOINT_SHA256: &str =
    "04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696";
const LOCKED_CHECKPOINT_BYTES: u64 = 1_746_942_600;
const PROMPT: &str = "Hello";
const PROMPT_TOKEN_IDS: [u32; 13] = [
    248045, 846, 198, 9419, 248046, 198, 248045, 74455, 198, 248068, 271, 248069, 271,
];
const GATE_SOURCE_NAME: &str = "qwen35_metal_w8_boundary_tail_head_v1_gate.rs";
const GATE_SOURCE_BYTES: &[u8] = include_bytes!("qwen35_metal_w8_boundary_tail_head_v1_gate.rs");
const BASELINE_PARENT_COMMIT: &str = "ac9f72885bb9f31c5a85bf7d30269b1a59aba382";
const EXPECTED_ORIGIN_URL: &str = "https://github.com/qhy991/ApxInf.git";
const EMBEDDED_CANDIDATE_COMMIT: Option<&str> = option_env!("APXINF_CANDIDATE_COMMIT");
const PREDECLARATION_RELATIVE_PATH: &str = "crates/apxinf-metal/evidence/next-hotspot/qwen35-gdn-core-fused-v1-production-ac-v1-predeclared-gate-v1-20260826.json";
const PRIMITIVE_RAW_RELATIVE_PATH: &str = "crates/apxinf-metal/evidence/next-hotspot/qwen35-gdn-core-fused-v1-primitive-abc-raw-v1-20260825.json";
const PRIMITIVE_ACCEPTED_SUMMARY_RELATIVE_PATH: &str = "crates/apxinf-metal/evidence/next-hotspot/qwen35-gdn-core-fused-v1-accepted-diagnostic-summary-v1-20260825.json";
const CAMPAIGN_SENTINEL_RELATIVE_PATH: &str = "crates/apxinf-metal/evidence/next-hotspot/qwen35-gdn-core-fused-v1-production-ac-v1-campaign-start-v1-20260826.json";
const RAW_RECEIPT_RELATIVE_PATH: &str = "crates/apxinf-metal/evidence/next-hotspot/qwen35-gdn-core-fused-v1-production-ac-v1-raw-campaign-v1-20260826.json";
const BODY_IMPROVEMENT_THRESHOLD_PERCENT: f64 = 3.0;
const POOLED_TPOT_IMPROVEMENT_THRESHOLD_PERCENT: f64 = 1.5;

const EXPECTED_CANDIDATE_CHANGED_PATHS: [&str; 12] = [
    PREDECLARATION_RELATIVE_PATH,
    "crates/apxinf-metal/src/gdn_core_fused_profile_v1.rs",
    "crates/apxinf-metal/src/linear_layer/mlp_stack3_boundary.rs",
    "crates/apxinf-metal/src/linear_layer/stack3.rs",
    "crates/apxinf-metal/src/metal_w8_linear_layer_stack3_bridge.mm",
    "crates/apxinf-metal/src/metal_w8_mlp_stack3_boundary_v1_bridge.mm",
    "crates/apxinf-metal/tests/linear_layer_stack3.rs",
    "crates/apxinf-metal/tests/mlp_stack3_boundary_v1.rs",
    "crates/apxinf-model/examples/qwen35_metal_w8_boundary_tail_head_v1_gate.rs",
    "crates/apxinf-model/examples/qwen35_metal_w8_linear_layer_gate.rs",
    "crates/apxinf-model/examples/support/qwen35_boundary_tail_head_v1_gate_evidence.rs",
    "crates/apxinf-model/src/qwen35/general.rs",
];

struct PinnedOutputTarget {
    requested_path: PathBuf,
    requested_parent: PathBuf,
    canonical_parent: PathBuf,
    parent_device: u64,
    parent_inode: u64,
    parent_dir: File,
    file_name: CString,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Arm {
    A,
    C,
}

impl Arm {
    const fn core_profile(self) -> apxinf_metal::GdnCoreProfileV1 {
        match self {
            Self::A => apxinf_metal::GdnCoreProfileV1::LegacyFourDispatch,
            Self::C => apxinf_metal::GdnCoreProfileV1::Fused128,
        }
    }

    const fn short(self) -> &'static str {
        match self {
            Self::A => "A",
            Self::C => "C",
        }
    }

    const fn profile(self) -> &'static str {
        match self {
            Self::A => "legacy-four-dispatch",
            Self::C => "gdn-core-fused-v1",
        }
    }

    const fn function_chain(self) -> &'static str {
        match self {
            Self::A => {
                "gdn_depthwise_preprocess|gdn_normalize_qk|gdn_recurrent_update|gdn_norm_gate"
            }
            Self::C => "gdn_core_fused_v1",
        }
    }

    const fn top_mechanism(self) -> &'static str {
        match self {
            Self::A => "metal-w8-mlp-stack3-boundary-tail-head-v1",
            Self::C => "metal-w8-mlp-stack3-boundary-tail-head-gdn-core-fused-v1",
        }
    }

    const fn initial_mechanism(self) -> &'static str {
        match self {
            Self::A => "metal-w8-linear-layer-stack3-v1",
            Self::C => "metal-w8-linear-layer-stack3-gdn-core-fused-v1",
        }
    }

    const fn boundary_mechanism(self) -> &'static str {
        match self {
            Self::A => "metal-w8-mlp-stack3-boundary-v1",
            Self::C => "metal-w8-mlp-stack3-boundary-gdn-core-fused-v1",
        }
    }

    const fn full_dispatches(self) -> usize {
        match self {
            Self::A => 267,
            Self::C => 213,
        }
    }

    const fn full_broad_barriers(self) -> usize {
        match self {
            Self::A => 243,
            Self::C => 189,
        }
    }

    const fn initial_dispatches(self) -> usize {
        match self {
            Self::A => 39,
            Self::C => 30,
        }
    }

    const fn initial_broad_barriers(self) -> usize {
        match self {
            Self::A => 36,
            Self::C => 27,
        }
    }

    const fn boundary_dispatches(self) -> usize {
        match self {
            Self::A => 44,
            Self::C => 35,
        }
    }

    const fn boundary_broad_barriers(self) -> usize {
        match self {
            Self::A => 40,
            Self::C => 31,
        }
    }

    const fn core_kernel_output_groups_per_row(self) -> u32 {
        match self {
            Self::A => 64,
            Self::C => 32,
        }
    }
}

// Put C in the unavoidable first system-wide Metal decode position. Any
// residual driver cold-start cost therefore works conservatively against the
// candidate instead of making A look slower.
const TEACHER_ORDER: [Arm; 4] = [Arm::C, Arm::A, Arm::A, Arm::C];
const FREE_BLOCK_1_ORDER: [Arm; 4] = [Arm::A, Arm::C, Arm::C, Arm::A];
const FREE_BLOCK_2_ORDER: [Arm; 4] = [Arm::C, Arm::A, Arm::A, Arm::C];

const fn gdn_core_profile_label(profile: apxinf_metal::GdnCoreProfileV1) -> &'static str {
    match profile {
        apxinf_metal::GdnCoreProfileV1::LegacyFourDispatch => "legacy-four-dispatch",
        apxinf_metal::GdnCoreProfileV1::QkStagedFourDispatch => "qk-staged-four-dispatch-control",
        apxinf_metal::GdnCoreProfileV1::Fused128 => "gdn-core-fused-v1",
    }
}

const STEPS: usize = 128;
const ALL_LINEAR_LAYER_INDICES: [usize; 18] = [
    0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 20, 21, 22,
];
const FULL_ATTENTION_LAYER_INDICES: [usize; 6] = [3, 7, 11, 15, 19, 23];
const BOUNDARY_REGIONS: [(usize, [usize; 3]); 5] = [
    (3, [4, 5, 6]),
    (7, [8, 9, 10]),
    (11, [12, 13, 14]),
    (15, [16, 17, 18]),
    (19, [20, 21, 22]),
];

fn validate_official_schedule(layer_types: &[Qwen35LayerType]) -> Result<(), String> {
    if layer_types.len() != 24 {
        return Err(format!(
            "boundary + tail-head v1 gate requires exactly 24 layers, got {}",
            layer_types.len()
        ));
    }
    let linear = layer_types
        .iter()
        .enumerate()
        .filter_map(|(index, layer_type)| {
            (*layer_type == Qwen35LayerType::LinearAttention).then_some(index)
        })
        .collect::<Vec<_>>();
    let full = layer_types
        .iter()
        .enumerate()
        .filter_map(|(index, layer_type)| {
            (*layer_type == Qwen35LayerType::FullAttention).then_some(index)
        })
        .collect::<Vec<_>>();
    if linear != ALL_LINEAR_LAYER_INDICES || full != FULL_ATTENTION_LAYER_INDICES {
        return Err(format!(
            "boundary + tail-head v1 requires linear={ALL_LINEAR_LAYER_INDICES:?} and full={FULL_ATTENTION_LAYER_INDICES:?}, got linear={linear:?}, full={full:?}"
        ));
    }
    Ok(())
}

fn validate_source_lock(lock: &Value) -> Result<(), String> {
    if lock.get("format").and_then(Value::as_str) != Some(SOURCE_LOCK_FORMAT)
        || lock.get("repo_id").and_then(Value::as_str) != Some(REPO_ID)
        || lock.get("resolved_commit").and_then(Value::as_str) != Some(LOCKED_REVISION)
    {
        return Err("source lock does not identify the frozen official Qwen3.5-0.8B source".into());
    }
    let files = lock
        .pointer("/weights/files")
        .and_then(Value::as_array)
        .ok_or("source lock weights.files must be an array")?;
    if files.len() != 1
        || files[0].get("path").and_then(Value::as_str) != Some(LOCKED_CHECKPOINT)
        || files[0].get("sha256").and_then(Value::as_str) != Some(LOCKED_CHECKPOINT_SHA256)
        || files[0].get("size").and_then(Value::as_u64) != Some(LOCKED_CHECKPOINT_BYTES)
    {
        return Err("source lock does not bind the frozen Qwen3.5-0.8B checkpoint".into());
    }
    Ok(())
}

fn argmax(logits: &Tensor, vocab_size: usize) -> Result<u32, Box<dyn Error>> {
    if logits.shape().dims() != [1, vocab_size] {
        return Err(format!("expected logits [1, {vocab_size}], got {}", logits.shape()).into());
    }
    let mut best_score = f32::NEG_INFINITY;
    let mut best_token = 0u32;
    for (token, &score) in logits.as_f32()?.iter().enumerate() {
        if score > best_score {
            best_score = score;
            best_token = u32::try_from(token)?;
        }
    }
    Ok(best_token)
}

fn official_initial_stack_ledger_is_exact(
    arm: Arm,
    ledger: apxinf_metal::LinearLayerStack3BufferLedger,
) -> bool {
    let profile = arm.core_profile();
    ledger.gdn_core_profile == profile
        && ledger.gdn_function_chain == arm.function_chain()
        && ledger.allocated_buffers == 76
        && ledger.shared_buffers == 68
        && ledger.private_buffers == 8
        && ledger.packed_weight_bytes == 64_585_728
        && ledger.packed_scale_bytes == 4_429_824
        && ledger.f32_parameter_bytes == 321_408
        && ledger.active_state_bytes == 3_440_640
        && ledger.scratch_state_bytes == 3_440_640
        && ledger.activation_bytes == 133_248
        && ledger.total_persistent_bytes == 76_351_488
        && ledger.host_input_bytes_per_decode == 4_096
        && ledger.host_output_bytes_per_decode == 4_096
        && ledger.state_host_transfer_bytes_per_decode == 0
        && ledger.command_buffers_per_decode == 1
        && ledger.compute_encoders_per_decode == 3
        && ledger.kernel_dispatches_per_decode == arm.initial_dispatches()
        && ledger.explicit_buffer_barriers_per_decode == arm.initial_broad_barriers()
        && ledger.gdn_core_seams_per_decode == 3
        && ledger.gdn_core_kernel_dispatches_per_decode
            == profile.gdn_core_dispatches_for_seams(3) as usize
        && ledger.gdn_core_explicit_buffer_barriers_per_decode
            == profile.gdn_core_dispatches_for_seams(3) as usize
        && ledger.gdn_core_recurrent_or_fused_threads_per_threadgroup
            == profile.recurrent_threads_per_threadgroup() as usize
        && ledger.gdn_core_threadgroups_per_decode
            == profile.gdn_core_threadgroups_for_seams(3) as usize
        && ledger.gdn_core_launched_threads_per_decode
            == profile.gdn_core_launched_threads_for_seams(3) as usize
        && ledger.gdn_core_source_declared_threadgroup_memory_bytes
            == profile.source_declared_threadgroup_memory_bytes() as usize
        && ledger.gdn_core_expected_pipeline_static_threadgroup_memory_bytes
            == profile.expected_pipeline_static_threadgroup_memory_bytes() as usize
        && ledger.gdn_core_internal_threadgroup_barrier_sites_per_threadgroup
            == profile.internal_threadgroup_barrier_sites_per_threadgroup() as usize
        && ledger.commits_per_decode == 1
        && ledger.waits_per_decode == 1
        && ledger.intermediate_host_finite_checks_per_decode == 0
        && ledger.final_output_finite_checks_per_decode == 1
}

fn official_boundary_ledger_is_exact(
    arm: Arm,
    ledger: apxinf_metal::MlpStack3BoundaryBufferLedgerV1,
) -> bool {
    let profile = arm.core_profile();
    ledger.gdn_core_profile == profile
        && ledger.gdn_function_chain == arm.function_chain()
        && ledger.scope == "resident-mtlbuffer-only"
        && ledger.exclusions
            == "CPU packed weights, host allocations, Metal pipelines/libraries/queues, command objects, driver allocations, attention/KV, model loader, and language-model head"
        && ledger.abi_version == 1
        && ledger.stack_depth == 3
        && ledger.allocated_buffers == 81
        && ledger.shared_buffers == 73
        && ledger.private_buffers == 8
        && ledger.packed_weight_bytes == 75_595_776
        && ledger.packed_scale_bytes == 5_117_952
        && ledger.f32_parameter_bytes == 325_504
        && ledger.active_state_bytes == 3_440_640
        && ledger.scratch_state_bytes == 3_440_640
        && ledger.activation_bytes == 133_248
        && ledger.total_persistent_bytes == 88_053_760
        && ledger.host_input_bytes_per_decode == 4_096
        && ledger.host_output_bytes_per_decode == 4_096
        && ledger.state_host_transfer_bytes_per_decode == 0
        && ledger.command_buffers_per_decode == 1
        && ledger.compute_encoders_per_decode == 4
        && ledger.kernel_dispatches_per_decode == arm.boundary_dispatches()
        && ledger.explicit_buffer_barriers_per_decode == arm.boundary_broad_barriers()
        && ledger.gdn_core_seams_per_decode == 3
        && ledger.gdn_core_kernel_dispatches_per_decode
            == profile.gdn_core_dispatches_for_seams(3) as usize
        && ledger.gdn_core_explicit_buffer_barriers_per_decode
            == profile.gdn_core_dispatches_for_seams(3) as usize
        && ledger.gdn_core_recurrent_or_fused_threads_per_threadgroup
            == profile.recurrent_threads_per_threadgroup() as usize
        && ledger.gdn_core_threadgroups_per_decode
            == profile.gdn_core_threadgroups_for_seams(3) as usize
        && ledger.gdn_core_launched_threads_per_decode
            == profile.gdn_core_launched_threads_for_seams(3) as usize
        && ledger.gdn_core_source_declared_threadgroup_memory_bytes
            == profile.source_declared_threadgroup_memory_bytes() as usize
        && ledger.gdn_core_expected_pipeline_static_threadgroup_memory_bytes
            == profile.expected_pipeline_static_threadgroup_memory_bytes() as usize
        && ledger.gdn_core_internal_threadgroup_barrier_sites_per_threadgroup
            == profile.internal_threadgroup_barrier_sites_per_threadgroup() as usize
        && ledger.commits_per_decode == 1
        && ledger.waits_per_decode == 1
        && ledger.intermediate_host_finite_checks_per_decode == 0
        && ledger.final_output_finite_checks_per_decode == 1
}

fn official_tail_ledger_is_exact(ledger: apxinf_metal::TailMlpHeadBufferLedgerV1) -> bool {
    ledger.scope == "resident-mtlbuffer-only"
        && ledger.exclusions
            == "CPU packed weights, host F32 tied embedding and exact four-candidate rerank, host allocations, Metal pipelines/libraries/queues, command objects, driver allocations, attention/KV, model loader, and all earlier model layers"
        && ledger.abi_version == 1
        && ledger.allocated_buffers == 13
        && ledger.shared_buffers == 10
        && ledger.private_buffers == 3
        && ledger.packed_weight_bytes == 265_289_728
        && ledger.packed_scale_bytes == 16_580_608
        && ledger.f32_parameter_bytes == 8_192
        && ledger.hidden_activation_bytes == 8_192
        && ledger.mlp_activation_bytes == 43_008
        && ledger.partial_topk_bytes == 993_280
        && ledger.output_token_bytes == 16
        && ledger.total_persistent_bytes == 282_923_024
        && ledger.host_input_bytes_per_decode == 4_096
        && ledger.host_output_bytes_per_decode == 4_112
        && ledger.state_host_transfer_bytes_per_decode == 0
        && ledger.command_buffers_per_decode == 1
        && ledger.compute_encoders_per_decode == 1
        && ledger.kernel_dispatches_per_decode == 8
        && ledger.buffer_barriers_per_decode == 7
        && ledger.commits_per_decode == 1
        && ledger.waits_per_decode == 1
}

fn official_aggregate_ledger_is_exact(
    arm: Arm,
    aggregate: &Qwen35MetalW8MlpStack3BoundaryTailHeadV1AggregateLedger,
) -> bool {
    if aggregate.scope != "resident-mtlbuffer-only"
        || aggregate.exclusions
            != "CPU F32 weights, host F32 tied embedding and exact four-candidate rerank, host allocations, Metal pipelines/libraries/queues, command objects, driver allocations, full attention/KV, model loader, and prefill CPU head"
        || !aggregate.includes_lm_head
        || aggregate.gdn_core_profile != arm.core_profile()
        || aggregate.gdn_function_chain != arm.function_chain()
        || aggregate.initial_stack.layer_indices != [0, 1, 2]
        || !official_initial_stack_ledger_is_exact(arm, aggregate.initial_stack.ledger)
        || aggregate.tail_layer_index != 23
        || !official_tail_ledger_is_exact(aggregate.tail)
        || aggregate.boundaries.len() != BOUNDARY_REGIONS.len()
        || !aggregate
            .boundaries
            .iter()
            .zip(BOUNDARY_REGIONS)
            .all(|(entry, (boundary_mlp_layer_index, stack_layer_indices))| {
                entry.boundary_mlp_layer_index == boundary_mlp_layer_index
                    && entry.stack_layer_indices == stack_layer_indices
                    && official_boundary_ledger_is_exact(arm, entry.ledger)
            })
    {
        return false;
    }
    let initial = aggregate.initial_stack.ledger;
    let boundaries = &aggregate.boundaries;
    let tail = aggregate.tail;
    macro_rules! sum {
        ($initial_field:ident, $boundary_field:ident, $tail_field:ident) => {{
            boundaries
                .iter()
                .try_fold(initial.$initial_field, |total, entry| {
                    total.checked_add(entry.ledger.$boundary_field)
                })
                .and_then(|total| total.checked_add(tail.$tail_field))
        }};
    }
    let recomputed_dispatches = boundaries
        .iter()
        .try_fold(arm.initial_dispatches(), |total, entry| {
            total.checked_add(entry.ledger.kernel_dispatches_per_decode)
        })
        .and_then(|total| total.checked_add(tail.kernel_dispatches_per_decode));
    aggregate.total_persistent_mtlbuffer_bytes
        == sum!(
            total_persistent_bytes,
            total_persistent_bytes,
            total_persistent_bytes
        )
        .unwrap_or(0)
        && aggregate.allocated_buffers
            == sum!(allocated_buffers, allocated_buffers, allocated_buffers).unwrap_or(0)
        && aggregate.shared_buffers
            == sum!(shared_buffers, shared_buffers, shared_buffers).unwrap_or(0)
        && aggregate.private_buffers
            == sum!(private_buffers, private_buffers, private_buffers).unwrap_or(0)
        && aggregate.host_to_device_bytes_per_decode
            == sum!(
                host_input_bytes_per_decode,
                host_input_bytes_per_decode,
                host_input_bytes_per_decode
            )
            .unwrap_or(0)
        && aggregate.device_to_host_bytes_per_decode
            == sum!(
                host_output_bytes_per_decode,
                host_output_bytes_per_decode,
                host_output_bytes_per_decode
            )
            .unwrap_or(0)
        && aggregate.state_host_transfer_bytes_per_decode
            == sum!(
                state_host_transfer_bytes_per_decode,
                state_host_transfer_bytes_per_decode,
                state_host_transfer_bytes_per_decode
            )
            .unwrap_or(usize::MAX)
        && aggregate.command_buffers_per_decode
            == sum!(
                command_buffers_per_decode,
                command_buffers_per_decode,
                command_buffers_per_decode
            )
            .unwrap_or(0)
        && aggregate.compute_encoders_per_decode
            == sum!(
                compute_encoders_per_decode,
                compute_encoders_per_decode,
                compute_encoders_per_decode
            )
            .unwrap_or(0)
        && aggregate.kernel_dispatches_per_decode == recomputed_dispatches.unwrap_or(0)
        && aggregate.commits_per_decode
            == sum!(commits_per_decode, commits_per_decode, commits_per_decode).unwrap_or(0)
        && aggregate.waits_per_decode
            == sum!(waits_per_decode, waits_per_decode, waits_per_decode).unwrap_or(0)
        && aggregate.total_persistent_mtlbuffer_bytes == 799_543_312
        && aggregate.allocated_buffers == 494
        && aggregate.shared_buffers == 443
        && aggregate.private_buffers == 51
        && aggregate.host_to_device_bytes_per_decode == 28_672
        && aggregate.device_to_host_bytes_per_decode == 28_688
        && aggregate.state_host_transfer_bytes_per_decode == 0
        && aggregate.command_buffers_per_decode == 7
        && aggregate.compute_encoders_per_decode == 24
        && aggregate.kernel_dispatches_per_decode == arm.full_dispatches()
        && aggregate.explicit_buffer_barriers_per_decode == arm.full_broad_barriers()
        && aggregate.commits_per_decode == 7
        && aggregate.waits_per_decode == 7
}

fn stack3_ledger_json(ledger: apxinf_metal::LinearLayerStack3BufferLedger) -> Value {
    json!({
        "gdn_core_profile": gdn_core_profile_label(ledger.gdn_core_profile),
        "gdn_function_chain": ledger.gdn_function_chain,
        "allocated_buffers": ledger.allocated_buffers,
        "shared_buffers": ledger.shared_buffers,
        "private_buffers": ledger.private_buffers,
        "packed_weight_bytes": ledger.packed_weight_bytes,
        "packed_scale_bytes": ledger.packed_scale_bytes,
        "f32_parameter_bytes": ledger.f32_parameter_bytes,
        "active_state_bytes": ledger.active_state_bytes,
        "scratch_state_bytes": ledger.scratch_state_bytes,
        "activation_bytes": ledger.activation_bytes,
        "total_persistent_bytes": ledger.total_persistent_bytes,
        "host_input_bytes_per_decode": ledger.host_input_bytes_per_decode,
        "host_output_bytes_per_decode": ledger.host_output_bytes_per_decode,
        "state_host_transfer_bytes_per_decode": ledger.state_host_transfer_bytes_per_decode,
        "command_buffers_per_decode": ledger.command_buffers_per_decode,
        "compute_encoders_per_decode": ledger.compute_encoders_per_decode,
        "kernel_dispatches_per_decode": ledger.kernel_dispatches_per_decode,
        "explicit_buffer_barriers_per_decode": ledger.explicit_buffer_barriers_per_decode,
        "gdn_core_seams_per_decode": ledger.gdn_core_seams_per_decode,
        "gdn_core_kernel_dispatches_per_decode": ledger.gdn_core_kernel_dispatches_per_decode,
        "gdn_core_explicit_buffer_barriers_per_decode": ledger.gdn_core_explicit_buffer_barriers_per_decode,
        "gdn_core_recurrent_or_fused_threads_per_threadgroup": ledger.gdn_core_recurrent_or_fused_threads_per_threadgroup,
        "gdn_core_threadgroups_per_decode": ledger.gdn_core_threadgroups_per_decode,
        "gdn_core_launched_threads_per_decode": ledger.gdn_core_launched_threads_per_decode,
        "gdn_core_source_declared_threadgroup_memory_bytes": ledger.gdn_core_source_declared_threadgroup_memory_bytes,
        "gdn_core_expected_pipeline_static_threadgroup_memory_bytes": ledger.gdn_core_expected_pipeline_static_threadgroup_memory_bytes,
        "gdn_core_internal_threadgroup_barrier_sites_per_threadgroup": ledger.gdn_core_internal_threadgroup_barrier_sites_per_threadgroup,
        "commits_per_decode": ledger.commits_per_decode,
        "waits_per_decode": ledger.waits_per_decode,
        "intermediate_host_finite_checks_per_decode": ledger.intermediate_host_finite_checks_per_decode,
        "final_output_finite_checks_per_decode": ledger.final_output_finite_checks_per_decode,
    })
}

fn boundary_ledger_json(ledger: apxinf_metal::MlpStack3BoundaryBufferLedgerV1) -> Value {
    json!({
        "gdn_core_profile": gdn_core_profile_label(ledger.gdn_core_profile),
        "gdn_function_chain": ledger.gdn_function_chain,
        "scope": ledger.scope,
        "exclusions": ledger.exclusions,
        "abi_version": ledger.abi_version,
        "stack_depth": ledger.stack_depth,
        "allocated_buffers": ledger.allocated_buffers,
        "shared_buffers": ledger.shared_buffers,
        "private_buffers": ledger.private_buffers,
        "packed_weight_bytes": ledger.packed_weight_bytes,
        "packed_scale_bytes": ledger.packed_scale_bytes,
        "f32_parameter_bytes": ledger.f32_parameter_bytes,
        "active_state_bytes": ledger.active_state_bytes,
        "scratch_state_bytes": ledger.scratch_state_bytes,
        "activation_bytes": ledger.activation_bytes,
        "total_persistent_bytes": ledger.total_persistent_bytes,
        "host_input_bytes_per_decode": ledger.host_input_bytes_per_decode,
        "host_output_bytes_per_decode": ledger.host_output_bytes_per_decode,
        "state_host_transfer_bytes_per_decode": ledger.state_host_transfer_bytes_per_decode,
        "command_buffers_per_decode": ledger.command_buffers_per_decode,
        "compute_encoders_per_decode": ledger.compute_encoders_per_decode,
        "kernel_dispatches_per_decode": ledger.kernel_dispatches_per_decode,
        "explicit_buffer_barriers_per_decode": ledger.explicit_buffer_barriers_per_decode,
        "gdn_core_seams_per_decode": ledger.gdn_core_seams_per_decode,
        "gdn_core_kernel_dispatches_per_decode": ledger.gdn_core_kernel_dispatches_per_decode,
        "gdn_core_explicit_buffer_barriers_per_decode": ledger.gdn_core_explicit_buffer_barriers_per_decode,
        "gdn_core_recurrent_or_fused_threads_per_threadgroup": ledger.gdn_core_recurrent_or_fused_threads_per_threadgroup,
        "gdn_core_threadgroups_per_decode": ledger.gdn_core_threadgroups_per_decode,
        "gdn_core_launched_threads_per_decode": ledger.gdn_core_launched_threads_per_decode,
        "gdn_core_source_declared_threadgroup_memory_bytes": ledger.gdn_core_source_declared_threadgroup_memory_bytes,
        "gdn_core_expected_pipeline_static_threadgroup_memory_bytes": ledger.gdn_core_expected_pipeline_static_threadgroup_memory_bytes,
        "gdn_core_internal_threadgroup_barrier_sites_per_threadgroup": ledger.gdn_core_internal_threadgroup_barrier_sites_per_threadgroup,
        "commits_per_decode": ledger.commits_per_decode,
        "waits_per_decode": ledger.waits_per_decode,
        "intermediate_host_finite_checks_per_decode": ledger.intermediate_host_finite_checks_per_decode,
        "final_output_finite_checks_per_decode": ledger.final_output_finite_checks_per_decode,
    })
}

fn tail_ledger_json(ledger: apxinf_metal::TailMlpHeadBufferLedgerV1) -> Value {
    json!({
        "scope": ledger.scope,
        "exclusions": ledger.exclusions,
        "abi_version": ledger.abi_version,
        "allocated_buffers": ledger.allocated_buffers,
        "shared_buffers": ledger.shared_buffers,
        "private_buffers": ledger.private_buffers,
        "packed_weight_bytes": ledger.packed_weight_bytes,
        "packed_scale_bytes": ledger.packed_scale_bytes,
        "f32_parameter_bytes": ledger.f32_parameter_bytes,
        "hidden_activation_bytes": ledger.hidden_activation_bytes,
        "mlp_activation_bytes": ledger.mlp_activation_bytes,
        "partial_topk_bytes": ledger.partial_topk_bytes,
        "output_token_bytes": ledger.output_token_bytes,
        "total_persistent_bytes": ledger.total_persistent_bytes,
        "host_input_bytes_per_decode": ledger.host_input_bytes_per_decode,
        "host_output_bytes_per_decode": ledger.host_output_bytes_per_decode,
        "state_host_transfer_bytes_per_decode": ledger.state_host_transfer_bytes_per_decode,
        "command_buffers_per_decode": ledger.command_buffers_per_decode,
        "compute_encoders_per_decode": ledger.compute_encoders_per_decode,
        "kernel_dispatches_per_decode": ledger.kernel_dispatches_per_decode,
        "buffer_barriers_per_decode": ledger.buffer_barriers_per_decode,
        "commits_per_decode": ledger.commits_per_decode,
        "waits_per_decode": ledger.waits_per_decode,
    })
}

fn aggregate_ledger_json(
    arm: Arm,
    ledger: &Qwen35MetalW8MlpStack3BoundaryTailHeadV1AggregateLedger,
) -> Value {
    json!({
        "scope": ledger.scope,
        "exclusions": ledger.exclusions,
        "includes_lm_head": ledger.includes_lm_head,
        "gdn_core_profile": gdn_core_profile_label(ledger.gdn_core_profile),
        "gdn_function_chain": ledger.gdn_function_chain,
        "initial_stack": {
            "layer_indices": ledger.initial_stack.layer_indices,
            "ledger": stack3_ledger_json(ledger.initial_stack.ledger),
        },
        "boundaries": ledger.boundaries.iter().map(|entry| json!({
            "boundary_mlp_layer_index": entry.boundary_mlp_layer_index,
            "stack_layer_indices": entry.stack_layer_indices,
            "ledger": boundary_ledger_json(entry.ledger),
        })).collect::<Vec<_>>(),
        "tail_layer_index": ledger.tail_layer_index,
        "tail": tail_ledger_json(ledger.tail),
        "total_persistent_mtlbuffer_bytes": ledger.total_persistent_mtlbuffer_bytes,
        "allocated_buffers": ledger.allocated_buffers,
        "shared_buffers": ledger.shared_buffers,
        "private_buffers": ledger.private_buffers,
        "host_to_device_bytes_per_decode": ledger.host_to_device_bytes_per_decode,
        "device_to_host_bytes_per_decode": ledger.device_to_host_bytes_per_decode,
        "state_host_transfer_bytes_per_decode": ledger.state_host_transfer_bytes_per_decode,
        "command_buffers_per_decode": ledger.command_buffers_per_decode,
        "compute_encoders_per_decode": ledger.compute_encoders_per_decode,
        "kernel_dispatches_per_decode": ledger.kernel_dispatches_per_decode,
        "explicit_buffer_barriers_per_decode": ledger.explicit_buffer_barriers_per_decode,
        "commits_per_decode": ledger.commits_per_decode,
        "waits_per_decode": ledger.waits_per_decode,
        "component_sum_recomputed_and_exact": official_aggregate_ledger_is_exact(arm, ledger),
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct TailPhaseCounts {
    prefill: usize,
    decode: usize,
    teacher: usize,
}

impl TailPhaseCounts {
    const fn teacher(calls: usize) -> Self {
        Self {
            prefill: 0,
            decode: 0,
            teacher: calls,
        }
    }

    const fn free(calls: usize) -> Self {
        Self {
            prefill: 0,
            decode: calls,
            teacher: 0,
        }
    }
}

fn boundary_tail_generation_receipt_is_exact(
    arm: Arm,
    receipt: &Value,
    body_tail_calls: usize,
    phase: TailPhaseCounts,
) -> bool {
    if phase.prefill != 0
        || phase.decode.checked_add(phase.teacher) != Some(body_tail_calls)
        || receipt.get("format").and_then(Value::as_str)
            != Some("apxinf-qwen35-mlp-stack3-boundary-tail-head-generation-path-v1")
        || receipt.get("mechanism").and_then(Value::as_str) != Some(arm.top_mechanism())
        || receipt.get("gdn_core_profile").and_then(Value::as_str) != Some(arm.profile())
        || receipt.get("gdn_function_chain").and_then(Value::as_str) != Some(arm.function_chain())
        || receipt
            .get("cpu_full_attention_and_kv")
            .and_then(Value::as_bool)
            != Some(true)
        || receipt
            .get("cpu_prefill_all_24_layers")
            .and_then(Value::as_bool)
            != Some(true)
        || receipt
            .get("metal_w8_initial_complete_linear_layer_stack3")
            .and_then(Value::as_bool)
            != Some(true)
        || receipt
            .get("metal_w8_mlp_stack3_boundaries")
            .and_then(Value::as_bool)
            != Some(true)
        || receipt
            .get("metal_w8_tail_layer23_mlp_final_rms_top4")
            .and_then(Value::as_bool)
            != Some(true)
        || receipt
            .get("standalone_layer23_mlp")
            .and_then(Value::as_bool)
            != Some(false)
        || receipt
            .get("standalone_metal_lm_head")
            .and_then(Value::as_bool)
            != Some(false)
        || receipt
            .get("f32_tied_four_candidate_rerank")
            .and_then(Value::as_bool)
            != Some(true)
        || receipt.get("prefill_body_calls").and_then(Value::as_u64) != Some(1)
        || receipt.get("terminal_error").and_then(Value::as_bool) != Some(false)
    {
        return false;
    }
    let calls = body_tail_calls as u64;
    let Some(three_calls) = body_tail_calls.checked_mul(3).map(|value| value as u64) else {
        return false;
    };
    let Some(four_calls) = body_tail_calls.checked_mul(4).map(|value| value as u64) else {
        return false;
    };
    let Some(transfer) = body_tail_calls.checked_mul(4_096).map(|value| value as u64) else {
        return false;
    };
    let state_mask = if body_tail_calls == 0 { 0 } else { 0b111 };
    let Some(initial) = receipt.get("initial_stack") else {
        return false;
    };
    let initial_valid = initial.get("layer_indices") == Some(&json!([0, 1, 2]))
        && initial.get("mechanism").and_then(Value::as_str) == Some(arm.initial_mechanism())
        && initial.get("gdn_core_profile").and_then(Value::as_str) == Some(arm.profile())
        && initial.get("gdn_function_chain").and_then(Value::as_str) == Some(arm.function_chain())
        && initial
            .get("kernel_dispatches_per_decode")
            .and_then(Value::as_u64)
            == Some(arm.initial_dispatches() as u64)
        && initial
            .get("explicit_buffer_barriers_per_decode")
            .and_then(Value::as_u64)
            == Some(arm.initial_broad_barriers() as u64)
        && production_receipt_json_is_exact(
            initial.get("last_gdn_core_receipt").unwrap_or(&Value::Null),
            arm,
            body_tail_calls,
        )
        && initial.get("prefill_seed_calls") == Some(&json!([1, 1, 1]))
        && initial.get("decode_calls").and_then(Value::as_u64) == Some(calls)
        && initial.get("successful_decodes").and_then(Value::as_u64) == Some(calls)
        && initial.get("failed_decodes").and_then(Value::as_u64) == Some(0)
        && initial.get("command_buffers").and_then(Value::as_u64) == Some(calls)
        && initial.get("compute_encoders").and_then(Value::as_u64) == Some(three_calls)
        && initial.get("commits").and_then(Value::as_u64) == Some(calls)
        && initial.get("waits").and_then(Value::as_u64) == Some(calls)
        && initial.get("host_to_device_bytes").and_then(Value::as_u64) == Some(transfer)
        && initial.get("device_to_host_bytes").and_then(Value::as_u64) == Some(transfer)
        && initial.get("state_commits").and_then(Value::as_u64) == Some(three_calls)
        && initial
            .get("last_state_commit_mask")
            .and_then(Value::as_u64)
            == Some(state_mask)
        && initial
            .get("committed_stack_version")
            .and_then(Value::as_u64)
            == Some(calls)
        && initial.get("terminal_error").and_then(Value::as_bool) == Some(false);
    let Some(boundaries) = receipt.get("boundaries").and_then(Value::as_array) else {
        return false;
    };
    let boundaries_valid = boundaries.len() == BOUNDARY_REGIONS.len()
        && boundaries.iter().zip(BOUNDARY_REGIONS).all(
            |(entry, (boundary_mlp_layer_index, stack_layer_indices))| {
                entry
                    .get("boundary_mlp_layer_index")
                    .and_then(Value::as_u64)
                    == Some(boundary_mlp_layer_index as u64)
                    && entry.get("stack_layer_indices") == Some(&json!(stack_layer_indices))
                    && entry.get("mechanism").and_then(Value::as_str)
                        == Some(arm.boundary_mechanism())
                    && entry.get("gdn_core_profile").and_then(Value::as_str) == Some(arm.profile())
                    && entry.get("gdn_function_chain").and_then(Value::as_str)
                        == Some(arm.function_chain())
                    && entry
                        .get("kernel_dispatches_per_decode")
                        .and_then(Value::as_u64)
                        == Some(arm.boundary_dispatches() as u64)
                    && entry
                        .get("explicit_buffer_barriers_per_decode")
                        .and_then(Value::as_u64)
                        == Some(arm.boundary_broad_barriers() as u64)
                    && production_receipt_json_is_exact(
                        entry.get("last_gdn_core_receipt").unwrap_or(&Value::Null),
                        arm,
                        body_tail_calls,
                    )
                    && entry.get("prefill_seed_calls") == Some(&json!([1, 1, 1]))
                    && entry.get("decode_calls").and_then(Value::as_u64) == Some(calls)
                    && entry.get("successful_decodes").and_then(Value::as_u64) == Some(calls)
                    && entry.get("failed_decodes").and_then(Value::as_u64) == Some(0)
                    && entry.get("command_buffers").and_then(Value::as_u64) == Some(calls)
                    && entry.get("compute_encoders").and_then(Value::as_u64) == Some(four_calls)
                    && entry.get("commits").and_then(Value::as_u64) == Some(calls)
                    && entry.get("waits").and_then(Value::as_u64) == Some(calls)
                    && entry.get("host_to_device_bytes").and_then(Value::as_u64) == Some(transfer)
                    && entry.get("device_to_host_bytes").and_then(Value::as_u64) == Some(transfer)
                    && entry.get("state_commits").and_then(Value::as_u64) == Some(three_calls)
                    && entry.get("last_state_commit_mask").and_then(Value::as_u64)
                        == Some(state_mask)
                    && entry.get("committed_stack_version").and_then(Value::as_u64) == Some(calls)
                    && entry.get("terminal_error").and_then(Value::as_bool) == Some(false)
            },
        );
    let Some(prefill_head) = receipt.get("prefill_head") else {
        return false;
    };
    let prefill_valid = prefill_head.get("mechanism").and_then(Value::as_str)
        == Some("cpu-f32-tied")
        && prefill_head.get("calls").and_then(Value::as_u64) == Some(1)
        && prefill_head
            .get("tail_transactions")
            .and_then(Value::as_u64)
            == Some(phase.prefill as u64);
    let Some(tail) = receipt.get("decode_head") else {
        return false;
    };
    let tail_valid = tail.get("mechanism").and_then(Value::as_str) == Some("metal-w8-tail-v1")
        && tail.get("layer_index").and_then(Value::as_u64) == Some(23)
        && tail.get("calls").and_then(Value::as_u64) == Some(phase.decode as u64)
        && tail.get("teacher_calls").and_then(Value::as_u64) == Some(phase.teacher as u64)
        && tail.get("tail_transactions").and_then(Value::as_u64) == Some(calls)
        && tail.get("successful_transactions").and_then(Value::as_u64) == Some(calls)
        && tail.get("failed_transactions").and_then(Value::as_u64) == Some(0)
        && tail.get("command_buffers").and_then(Value::as_u64) == Some(calls)
        && tail.get("compute_encoders").and_then(Value::as_u64) == Some(calls)
        && tail.get("kernel_dispatches").and_then(Value::as_u64)
            == body_tail_calls.checked_mul(8).map(|value| value as u64)
        && tail.get("commits").and_then(Value::as_u64) == Some(calls)
        && tail.get("waits").and_then(Value::as_u64) == Some(calls)
        && tail.get("host_to_device_bytes").and_then(Value::as_u64) == Some(transfer)
        && tail.get("device_to_host_bytes").and_then(Value::as_u64)
            == body_tail_calls.checked_mul(4_112).map(|value| value as u64)
        && tail.get("output_commits").and_then(Value::as_u64)
            == body_tail_calls.checked_mul(2).map(|value| value as u64)
        && tail.get("last_output_commit_mask").and_then(Value::as_u64)
            == Some(if body_tail_calls == 0 { 0 } else { 0b11 })
        && tail.get("terminal_error").and_then(Value::as_bool) == Some(false);
    let Some(aggregate) = receipt.get("aggregate") else {
        return false;
    };
    let aggregate_valid = aggregate.get("scope").and_then(Value::as_str)
        == Some("resident-mtlbuffer-only")
        && aggregate.get("includes_lm_head").and_then(Value::as_bool) == Some(true)
        && aggregate.get("gdn_core_profile").and_then(Value::as_str) == Some(arm.profile())
        && aggregate.get("gdn_function_chain").and_then(Value::as_str)
            == Some(arm.function_chain())
        && aggregate
            .get("persistent_mtlbuffer_bytes")
            .and_then(Value::as_u64)
            == Some(799_543_312)
        && aggregate.get("allocated_buffers").and_then(Value::as_u64) == Some(494)
        && aggregate.get("shared_buffers").and_then(Value::as_u64) == Some(443)
        && aggregate.get("private_buffers").and_then(Value::as_u64) == Some(51)
        && aggregate
            .get("host_to_device_bytes_per_decode")
            .and_then(Value::as_u64)
            == Some(28_672)
        && aggregate
            .get("device_to_host_bytes_per_decode")
            .and_then(Value::as_u64)
            == Some(28_688)
        && aggregate
            .get("state_host_transfer_bytes_per_decode")
            .and_then(Value::as_u64)
            == Some(0)
        && aggregate
            .get("command_buffers_per_decode")
            .and_then(Value::as_u64)
            == Some(7)
        && aggregate
            .get("compute_encoders_per_decode")
            .and_then(Value::as_u64)
            == Some(24)
        && aggregate
            .get("kernel_dispatches_per_decode")
            .and_then(Value::as_u64)
            == Some(arm.full_dispatches() as u64)
        && aggregate
            .get("explicit_buffer_barriers_per_decode")
            .and_then(Value::as_u64)
            == Some(arm.full_broad_barriers() as u64)
        && aggregate.get("commits_per_decode").and_then(Value::as_u64) == Some(7)
        && aggregate.get("waits_per_decode").and_then(Value::as_u64) == Some(7);
    initial_valid && boundaries_valid && prefill_valid && tail_valid && aggregate_valid
}

fn production_receipt_json_is_exact(receipt: &Value, arm: Arm, calls: usize) -> bool {
    if calls == 0 {
        return receipt.is_null();
    }
    let profile = arm.core_profile();
    receipt.get("profile").and_then(Value::as_str) == Some(arm.profile())
        && receipt.get("function_chain").and_then(Value::as_str) == Some(arm.function_chain())
        && receipt.get("gdn_core_seams").and_then(Value::as_u64) == Some(3)
        && receipt
            .get("persistent_output_groups_per_row")
            .and_then(Value::as_u64)
            == Some(64)
        && receipt
            .get("core_kernel_output_groups_per_row")
            .and_then(Value::as_u64)
            == Some(arm.core_kernel_output_groups_per_row() as u64)
        && receipt.get("kernel_dispatches").and_then(Value::as_u64)
            == Some(profile.gdn_core_dispatches_for_seams(3) as u64)
        && receipt
            .get("explicit_buffer_barriers")
            .and_then(Value::as_u64)
            == Some(profile.gdn_core_dispatches_for_seams(3) as u64)
        && receipt
            .get("recurrent_or_fused_threads_per_threadgroup")
            .and_then(Value::as_u64)
            == Some(profile.recurrent_threads_per_threadgroup() as u64)
        && receipt.get("threadgroups").and_then(Value::as_u64)
            == Some(profile.gdn_core_threadgroups_for_seams(3) as u64)
        && receipt.get("launched_threads").and_then(Value::as_u64)
            == Some(profile.gdn_core_launched_threads_for_seams(3) as u64)
        && receipt
            .get("pipeline_thread_execution_width")
            .and_then(Value::as_u64)
            == Some(32)
        && receipt
            .get("source_declared_threadgroup_memory_bytes")
            .and_then(Value::as_u64)
            == Some(profile.source_declared_threadgroup_memory_bytes() as u64)
        && receipt
            .get("pipeline_static_threadgroup_memory_bytes")
            .and_then(Value::as_u64)
            == Some(profile.expected_pipeline_static_threadgroup_memory_bytes() as u64)
        && receipt
            .get("internal_threadgroup_barrier_sites_per_threadgroup")
            .and_then(Value::as_u64)
            == Some(profile.internal_threadgroup_barrier_sites_per_threadgroup() as u64)
        && receipt
            .get("fixed_shape_validated")
            .and_then(Value::as_bool)
            == Some(true)
        && receipt.get("rms_norm_eps_bits").and_then(Value::as_u64)
            == Some(1.0e-6_f32.to_bits() as u64)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct BoundaryTailPathChecks {
    schedule_valid: bool,
    mechanism_and_precision_valid: bool,
    six_region_execution_valid: bool,
    tail_execution_and_phase_valid: bool,
    aggregate_ledger_valid: bool,
    generation_receipt_valid: bool,
    terminal_clear: bool,
}

impl BoundaryTailPathChecks {
    fn all_valid(self) -> bool {
        self.schedule_valid
            && self.mechanism_and_precision_valid
            && self.six_region_execution_valid
            && self.tail_execution_and_phase_valid
            && self.aggregate_ledger_valid
            && self.generation_receipt_valid
            && self.terminal_clear
    }

    fn receipt_json(self) -> Value {
        json!({
            "schedule_valid": self.schedule_valid,
            "mechanism_and_precision_valid": self.mechanism_and_precision_valid,
            "six_region_execution_valid": self.six_region_execution_valid,
            "tail_execution_and_phase_valid": self.tail_execution_and_phase_valid,
            "aggregate_ledger_valid": self.aggregate_ledger_valid,
            "generation_receipt_valid": self.generation_receipt_valid,
            "terminal_clear": self.terminal_clear,
            "all_valid": self.all_valid(),
        })
    }
}

fn quantization_profile_is_exact(ledger: apxinf_metal::LinearLayerQuantizationLedger) -> bool {
    ledger.gdn_input_group_size == apxinf_metal::W8GroupSize::G64
        && ledger.gdn_output_group_size == apxinf_metal::W8GroupSize::G32
        && ledger.mlp_gate_group_size == apxinf_metal::W8GroupSize::G64
        && ledger.mlp_up_group_size == apxinf_metal::W8GroupSize::G64
        && ledger.mlp_down_group_size == apxinf_metal::W8GroupSize::G64
}

fn boundary_tail_path_checks(
    arm: Arm,
    stats: &Qwen35MetalW8MlpStack3BoundaryTailHeadV1Stats,
    aggregate: &Qwen35MetalW8MlpStack3BoundaryTailHeadV1AggregateLedger,
    generation_receipt: &Value,
    body_tail_calls: usize,
    phase: TailPhaseCounts,
) -> BoundaryTailPathChecks {
    let calls = body_tail_calls;
    let triple = calls.checked_mul(3);
    let quadruple = calls.checked_mul(4);
    let h4 = calls.checked_mul(4_096);
    let h4_top4 = calls.checked_mul(4_112);
    let state_mask = if calls == 0 { 0 } else { 0b111 };
    let initial_execution = stats.initial_stack.execution;
    let initial_valid = triple.is_some()
        && h4.is_some()
        && stats.initial_stack.prefill_seed_calls == [1, 1, 1]
        && initial_execution.decode_calls == calls
        && initial_execution.successful_decodes == calls
        && initial_execution.failed_decodes == 0
        && initial_execution.command_buffers == calls
        && initial_execution.compute_encoders == triple.unwrap()
        && initial_execution.commits == calls
        && initial_execution.waits == calls
        && initial_execution.host_to_device_bytes == h4.unwrap()
        && initial_execution.device_to_host_bytes == h4.unwrap()
        && initial_execution.state_commits == triple.unwrap()
        && initial_execution.last_state_commit_mask == state_mask
        && initial_execution.committed_stack_version == calls as u64
        && !initial_execution.terminal_error
        && !stats.initial_stack.terminal_error;
    let boundaries_valid = triple.is_some()
        && quadruple.is_some()
        && h4.is_some()
        && stats.boundaries.iter().all(|region| {
            let execution = region.execution;
            region.prefill_seed_calls == [1, 1, 1]
                && execution.decode_calls == calls
                && execution.successful_decodes == calls
                && execution.failed_decodes == 0
                && execution.command_buffers == calls
                && execution.compute_encoders == quadruple.unwrap()
                && execution.commits == calls
                && execution.waits == calls
                && execution.host_to_device_bytes == h4.unwrap()
                && execution.device_to_host_bytes == h4.unwrap()
                && execution.state_commits == triple.unwrap()
                && execution.last_state_commit_mask == state_mask
                && execution.committed_stack_version == calls as u64
                && !execution.terminal_error
                && !region.terminal_error
        });
    let tail = stats.tail;
    let tail_valid = h4.is_some()
        && h4_top4.is_some()
        && stats.prefill_body_calls == 1
        && stats.prefill_cpu_head_calls == 1
        && stats.decode_calls == phase.decode
        && stats.teacher_calls == phase.teacher
        && phase.prefill == 0
        && phase.decode.checked_add(phase.teacher) == Some(calls)
        && tail.decode_calls == calls
        && tail.successful_decodes == calls
        && tail.failed_decodes == 0
        && tail.host_to_device_bytes == h4.unwrap()
        && tail.device_to_host_bytes == h4_top4.unwrap()
        && tail.command_buffers == calls
        && tail.compute_encoders == calls
        && tail.kernel_dispatches == calls.checked_mul(8).unwrap_or(usize::MAX)
        && tail.buffer_barriers == calls.checked_mul(7).unwrap_or(usize::MAX)
        && tail.commits == calls
        && tail.waits == calls
        && tail.output_commits == calls.checked_mul(2).unwrap_or(usize::MAX)
        && tail.last_output_commit_mask == if calls == 0 { 0 } else { 0b11 }
        && !tail.terminal_error;
    let stats_boundaries = stats
        .boundaries
        .iter()
        .map(|entry| (entry.boundary_mlp_layer_index, entry.stack_layer_indices))
        .collect::<Vec<_>>();
    BoundaryTailPathChecks {
        schedule_valid: stats.initial_stack.layer_indices == [0, 1, 2]
            && stats_boundaries == BOUNDARY_REGIONS
            && stats.tail_layer_index == 23,
        mechanism_and_precision_valid: stats.mechanism == arm.top_mechanism()
            && stats.gdn_core_profile == arm.core_profile()
            && stats.gdn_function_chain == arm.function_chain()
            && stats.initial_stack.mechanism == arm.initial_mechanism()
            && stats.initial_stack.gdn_core_profile == arm.core_profile()
            && stats.initial_stack.gdn_function_chain == arm.function_chain()
            && stats.initial_stack.kernel_dispatches_per_decode == arm.initial_dispatches()
            && stats.initial_stack.explicit_buffer_barriers_per_decode
                == arm.initial_broad_barriers()
            && production_receipt_struct_is_exact(
                stats.initial_stack.last_gdn_core_receipt,
                arm,
                calls,
            )
            && stats
                .initial_stack
                .quantization
                .iter()
                .copied()
                .all(quantization_profile_is_exact)
            && stats.boundaries.iter().all(|region| {
                region.mechanism == arm.boundary_mechanism()
                    && region.gdn_core_profile == arm.core_profile()
                    && region.gdn_function_chain == arm.function_chain()
                    && region.kernel_dispatches_per_decode == arm.boundary_dispatches()
                    && region.explicit_buffer_barriers_per_decode == arm.boundary_broad_barriers()
                    && production_receipt_struct_is_exact(region.last_gdn_core_receipt, arm, calls)
                    && region
                        .quantization
                        .iter()
                        .copied()
                        .all(quantization_profile_is_exact)
            }),
        six_region_execution_valid: initial_valid && boundaries_valid,
        tail_execution_and_phase_valid: tail_valid,
        aggregate_ledger_valid: official_aggregate_ledger_is_exact(arm, aggregate),
        generation_receipt_valid: boundary_tail_generation_receipt_is_exact(
            arm,
            generation_receipt,
            body_tail_calls,
            phase,
        ),
        terminal_clear: !stats.terminal_error
            && generation_receipt
                .get("terminal_error")
                .and_then(Value::as_bool)
                == Some(false),
    }
}

fn production_receipt_struct_is_exact(
    receipt: Option<apxinf_metal::GdnCoreProductionReceiptV1>,
    arm: Arm,
    calls: usize,
) -> bool {
    if calls == 0 {
        return receipt.is_none();
    }
    let Some(receipt) = receipt else {
        return false;
    };
    let profile = arm.core_profile();
    receipt.profile == profile
        && receipt.function_chain == arm.function_chain()
        && receipt.gdn_core_seams == 3
        && receipt.persistent_output_groups_per_row == 64
        && receipt.core_kernel_output_groups_per_row == arm.core_kernel_output_groups_per_row()
        && receipt.kernel_dispatches == profile.gdn_core_dispatches_for_seams(3)
        && receipt.explicit_buffer_barriers == profile.gdn_core_dispatches_for_seams(3)
        && receipt.recurrent_or_fused_threads_per_threadgroup
            == profile.recurrent_threads_per_threadgroup()
        && receipt.threadgroups == profile.gdn_core_threadgroups_for_seams(3)
        && receipt.launched_threads == profile.gdn_core_launched_threads_for_seams(3)
        && receipt.pipeline_thread_execution_width == 32
        && receipt.source_declared_threadgroup_memory_bytes
            == profile.source_declared_threadgroup_memory_bytes()
        && receipt.pipeline_static_threadgroup_memory_bytes
            == profile.expected_pipeline_static_threadgroup_memory_bytes()
        && receipt.internal_threadgroup_barrier_sites_per_threadgroup
            == profile.internal_threadgroup_barrier_sites_per_threadgroup()
        && receipt.fixed_shape_validated
        && receipt.rms_norm_eps_bits == 1.0e-6_f32.to_bits()
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct TeacherOracle {
    teacher_inputs: Vec<u32>,
    expected_outputs: Vec<u32>,
}

struct TeacherGateEvaluation {
    passed: bool,
    body_token_mismatches: Vec<Value>,
    top4_mismatches: Vec<Value>,
    rerank_mismatches: Vec<Value>,
    end_to_end_mismatches: Vec<Value>,
}

fn compute_same_process_teacher_oracle(
    model: &mut GeneralQwen35,
    prompt_tokens: &[u32],
    vocab_size: usize,
) -> Result<(u32, TeacherOracle), Box<dyn Error>> {
    let prefill_logits = model.prefill_for_generation(LlmInput::text(prompt_tokens))?;
    let prefill_token = argmax(&prefill_logits, vocab_size)?;
    let mut teacher = prefill_token;
    let mut teacher_inputs = Vec::with_capacity(STEPS);
    let mut expected_outputs = Vec::with_capacity(STEPS);
    for step in 0..STEPS {
        teacher_inputs.push(teacher);
        let position = prompt_tokens
            .len()
            .checked_add(step)
            .ok_or("same-process teacher position overflow")?;
        let logits = model.forward(&[teacher], u32::try_from(position)?)?;
        teacher = argmax(&logits, vocab_size)?;
        expected_outputs.push(teacher);
    }
    Ok((
        prefill_token,
        TeacherOracle {
            teacher_inputs,
            expected_outputs,
        },
    ))
}

fn evaluate_teacher_candidate(
    oracle: &TeacherOracle,
    tail_normalized_f32_tokens: &[u32],
    tail_top4_candidates: &[[u32; 4]],
    directly_reranked_tokens: &[u32],
    vocab_size: usize,
) -> Result<TeacherGateEvaluation, String> {
    if vocab_size < 4
        || oracle.teacher_inputs.len() != STEPS
        || oracle.expected_outputs.len() != STEPS
        || tail_normalized_f32_tokens.len() != STEPS
        || tail_top4_candidates.len() != STEPS
        || directly_reranked_tokens.len() != STEPS
    {
        return Err("teacher candidate evidence must contain exactly 128 valid steps".into());
    }
    let mut body_token_mismatches = Vec::new();
    let mut top4_mismatches = Vec::new();
    let mut rerank_mismatches = Vec::new();
    let mut end_to_end_mismatches = Vec::new();
    for step in 0..STEPS {
        let frozen_expected = oracle.expected_outputs[step];
        let normalized_f32_winner = tail_normalized_f32_tokens[step];
        let candidates = tail_top4_candidates[step];
        let reranked = directly_reranked_tokens[step];
        if normalized_f32_winner != frozen_expected {
            body_token_mismatches.push(json!({
                "step": step,
                "input": oracle.teacher_inputs[step],
                "expected": frozen_expected,
                "actual": normalized_f32_winner,
                "teacher_input": oracle.teacher_inputs[step],
                "frozen_cpu_expected": frozen_expected,
                "tail_normalized_hidden_f32_winner": normalized_f32_winner,
            }));
        }
        let unique = candidates
            .iter()
            .copied()
            .collect::<std::collections::BTreeSet<_>>()
            .len()
            == 4;
        let in_range = candidates
            .iter()
            .all(|candidate| (*candidate as usize) < vocab_size);
        let contains_winner = candidates.contains(&normalized_f32_winner);
        if !unique || !in_range || !contains_winner {
            top4_mismatches.push(json!({
                "step": step,
                "input": oracle.teacher_inputs[step],
                "expected": {
                    "four_unique_candidates": true,
                    "candidate_upper_bound_exclusive": vocab_size,
                    "contains_tail_normalized_f32_winner": normalized_f32_winner,
                },
                "actual": candidates,
                "teacher_input": oracle.teacher_inputs[step],
                "frozen_cpu_expected": frozen_expected,
                "tail_normalized_hidden_f32_winner": normalized_f32_winner,
                "actual_tail_top4": candidates,
                "unique": unique,
                "in_range": in_range,
                "contains_f32_winner": contains_winner,
            }));
        }
        if reranked != normalized_f32_winner {
            rerank_mismatches.push(json!({
                "step": step,
                "input": oracle.teacher_inputs[step],
                "expected": normalized_f32_winner,
                "actual": reranked,
                "teacher_input": oracle.teacher_inputs[step],
                "frozen_cpu_expected": frozen_expected,
                "tail_normalized_hidden_f32_winner": normalized_f32_winner,
                "actual_direct_tied_f32_rerank": reranked,
            }));
        }
        if reranked != frozen_expected {
            end_to_end_mismatches.push(json!({
                "step": step,
                "input": oracle.teacher_inputs[step],
                "expected": frozen_expected,
                "actual": reranked,
                "teacher_input": oracle.teacher_inputs[step],
                "frozen_cpu_expected": frozen_expected,
                "actual_direct_tied_f32_rerank": reranked,
            }));
        }
    }
    Ok(TeacherGateEvaluation {
        passed: body_token_mismatches.is_empty()
            && top4_mismatches.is_empty()
            && rerank_mismatches.is_empty()
            && end_to_end_mismatches.is_empty(),
        body_token_mismatches,
        top4_mismatches,
        rerank_mismatches,
        end_to_end_mismatches,
    })
}

fn evaluate_free_trajectory(expected: &[u32], actual: &[u32]) -> Result<Vec<Value>, String> {
    if expected.len() != STEPS || actual.len() != STEPS {
        return Err("free-run comparison requires two complete 128-token trajectories".into());
    }
    Ok(expected
        .iter()
        .zip(actual)
        .enumerate()
        .filter_map(|(step, (&cpu_expected, &candidate_actual))| {
            (cpu_expected != candidate_actual).then(|| {
                json!({
                    "step": step,
                    "input": "shared_prompt_plus_prior_generated_tokens",
                    "expected": cpu_expected,
                    "actual": candidate_actual,
                    "cpu_expected": cpu_expected,
                    "boundary_tail_v1_actual": candidate_actual,
                })
            })
        })
        .collect())
}

struct Args {
    model_dir: PathBuf,
    source_lock: PathBuf,
    output: PathBuf,
    candidate_commit: String,
}

fn usage() -> &'static str {
    "Usage: qwen35_metal_w8_boundary_tail_head_v1_gate \
  --model-dir OFFICIAL_LOCAL_QWEN35_0_8B \
  --source-lock SOURCE_LOCK.json \
  --candidate-commit FULL_40_HEX_SHA \
  --output NEW_RECEIPT.json"
}

fn parse_args_from<I, S>(args: I) -> Result<Args, String>
where
    I: IntoIterator<Item = S>,
    S: Into<OsString>,
{
    let mut model_dir = None;
    let mut source_lock = None;
    let mut output = None;
    let mut candidate_commit = None;
    let mut iter = args.into_iter().map(Into::into).skip(1);
    while let Some(raw_flag) = iter.next() {
        let flag = raw_flag.to_string_lossy();
        let value = iter
            .next()
            .ok_or_else(|| format!("{flag} requires a value"))?;
        match flag.as_ref() {
            "--model-dir" => {
                if model_dir.replace(PathBuf::from(value)).is_some() {
                    return Err("duplicate --model-dir".into());
                }
            }
            "--source-lock" => {
                if source_lock.replace(PathBuf::from(value)).is_some() {
                    return Err("duplicate --source-lock".into());
                }
            }
            "--candidate-commit" => {
                if candidate_commit
                    .replace(value.to_string_lossy().into_owned())
                    .is_some()
                {
                    return Err("duplicate --candidate-commit".into());
                }
            }
            "--output" => {
                if output.replace(PathBuf::from(value)).is_some() {
                    return Err("duplicate --output".into());
                }
            }
            other => return Err(format!("unknown argument {other:?}\n{}", usage())),
        }
    }
    let args = Args {
        model_dir: model_dir.ok_or_else(|| format!("--model-dir is required\n{}", usage()))?,
        source_lock: source_lock
            .ok_or_else(|| format!("--source-lock is required\n{}", usage()))?,
        output: output.ok_or_else(|| format!("--output is required\n{}", usage()))?,
        candidate_commit: candidate_commit
            .ok_or_else(|| format!("--candidate-commit is required\n{}", usage()))?,
    };
    if !args.model_dir.is_absolute()
        || !args.source_lock.is_absolute()
        || !args.output.is_absolute()
    {
        return Err("all gate paths must be absolute".into());
    }
    if args.output.exists() {
        return Err("--output must not already exist".into());
    }
    if args.candidate_commit.len() != 40
        || !args
            .candidate_commit
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit())
    {
        return Err("--candidate-commit must be a full 40-character hexadecimal commit".into());
    }
    Ok(args)
}

impl PinnedOutputTarget {
    fn capture(path: &Path) -> Result<Self, String> {
        let requested_parent = path
            .parent()
            .ok_or_else(|| "--output must have a parent directory".to_string())?
            .to_path_buf();
        let canonical_parent = std::fs::canonicalize(&requested_parent)
            .map_err(|error| format!("cannot canonicalize output parent directory: {error}"))?;
        let parent_dir = File::open(&canonical_parent)
            .map_err(|error| format!("cannot pin output parent directory: {error}"))?;
        let metadata = parent_dir
            .metadata()
            .map_err(|error| format!("cannot inspect pinned output parent directory: {error}"))?;
        if !metadata.is_dir() {
            return Err("--output parent must be a directory".into());
        }
        let file_name = path
            .file_name()
            .ok_or_else(|| "--output must name a file".to_string())?;
        if file_name == "." || file_name == ".." {
            return Err("--output must name a regular directory entry".into());
        }
        let file_name = CString::new(file_name.as_bytes())
            .map_err(|_| "--output file name must not contain NUL".to_string())?;
        Ok(Self {
            requested_path: path.to_path_buf(),
            requested_parent,
            canonical_parent,
            parent_device: metadata.dev(),
            parent_inode: metadata.ino(),
            parent_dir,
            file_name,
        })
    }

    fn verify_path_binding(&self) -> Result<(), String> {
        let current_canonical = std::fs::canonicalize(&self.requested_parent).map_err(|error| {
            format!("cannot re-canonicalize output parent before publication: {error}")
        })?;
        if current_canonical != self.canonical_parent {
            return Err("--output parent path changed during gate execution".into());
        }
        let current = std::fs::metadata(&current_canonical)
            .map_err(|error| format!("cannot re-inspect output parent: {error}"))?;
        let pinned = self
            .parent_dir
            .metadata()
            .map_err(|error| format!("cannot re-inspect pinned output parent: {error}"))?;
        if current.dev() != self.parent_device
            || current.ino() != self.parent_inode
            || pinned.dev() != self.parent_device
            || pinned.ino() != self.parent_inode
        {
            return Err("--output parent directory identity changed during gate execution".into());
        }
        if self.requested_path.exists() {
            return Err("--output must not already exist".into());
        }
        Ok(())
    }
}

#[cfg(target_os = "macos")]
fn publish_receipt_create_new(
    target: &PinnedOutputTarget,
    receipt: &Value,
) -> Result<(), Box<dyn std::error::Error>> {
    use std::os::raw::{c_char, c_int, c_uint};
    use std::sync::atomic::{AtomicU64, Ordering};

    unsafe extern "C" {
        fn openat(dirfd: c_int, path: *const c_char, oflag: c_int, ...) -> c_int;
        fn renameatx_np(
            fromfd: c_int,
            from: *const c_char,
            tofd: c_int,
            to: *const c_char,
            flags: c_uint,
        ) -> c_int;
        fn unlinkat(dirfd: c_int, path: *const c_char, flags: c_int) -> c_int;
    }

    // Darwin fcntl.h values. The gate is macOS-only and the release entrypoint
    // rejects every other target before model loading.
    const O_WRONLY: c_int = 0x0001;
    const O_NOFOLLOW: c_int = 0x0100;
    const O_CREAT: c_int = 0x0200;
    const O_EXCL: c_int = 0x0800;
    const O_CLOEXEC: c_int = 0x0100_0000;
    const RENAME_EXCL: c_uint = 0x0000_0004;
    static NEXT_TEMP_RECEIPT: AtomicU64 = AtomicU64::new(0);

    target.verify_path_binding()?;
    let mut bytes = serde_json::to_vec(receipt)?;
    bytes.push(b'\n');
    let (raw_fd, temp_name) = (0..64)
        .find_map(|_| {
            let serial = NEXT_TEMP_RECEIPT.fetch_add(1, Ordering::Relaxed);
            let name = CString::new(format!(
                ".apxinf-receipt-{}-{serial}.tmp",
                std::process::id()
            ))
            .expect("generated receipt temp name cannot contain NUL");
            let fd = unsafe {
                openat(
                    target.parent_dir.as_raw_fd(),
                    name.as_ptr(),
                    O_WRONLY | O_NOFOLLOW | O_CREAT | O_EXCL | O_CLOEXEC,
                    0o600 as c_uint,
                )
            };
            if fd >= 0 {
                return Some(Ok((fd, name)));
            }
            let error = std::io::Error::last_os_error();
            if error.kind() == std::io::ErrorKind::AlreadyExists {
                None
            } else {
                Some(Err(error))
            }
        })
        .ok_or("could not allocate a unique receipt staging entry")??;
    let mut output = unsafe { File::from_raw_fd(raw_fd) };
    let staged = output.write_all(&bytes).and_then(|_| output.sync_all());
    drop(output);
    if let Err(error) = staged {
        unsafe {
            unlinkat(target.parent_dir.as_raw_fd(), temp_name.as_ptr(), 0);
        }
        return Err(error.into());
    }
    if let Err(error) = target.verify_path_binding() {
        unsafe {
            unlinkat(target.parent_dir.as_raw_fd(), temp_name.as_ptr(), 0);
        }
        return Err(error.into());
    }
    let renamed = unsafe {
        renameatx_np(
            target.parent_dir.as_raw_fd(),
            temp_name.as_ptr(),
            target.parent_dir.as_raw_fd(),
            target.file_name.as_ptr(),
            RENAME_EXCL,
        )
    };
    if renamed != 0 {
        let error = std::io::Error::last_os_error();
        unsafe {
            unlinkat(target.parent_dir.as_raw_fd(), temp_name.as_ptr(), 0);
        }
        return Err(error.into());
    }
    target.parent_dir.sync_all()?;
    Ok(())
}

#[cfg(not(target_os = "macos"))]
fn publish_receipt_create_new(
    _target: &PinnedOutputTarget,
    _receipt: &Value,
) -> Result<(), Box<dyn std::error::Error>> {
    Err("boundary-tail v1 receipt publication requires macOS".into())
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct TeacherTrace {
    normalized_f32_tokens: Vec<u32>,
    top4_candidates: Vec<[u32; 4]>,
    reranked_tokens: Vec<u32>,
}

#[derive(Clone, Debug)]
struct TeacherSample {
    arm: Arm,
    body_ns_per_call: f64,
}

#[derive(Clone, Debug)]
struct FreeSample {
    arm: Arm,
    block: usize,
    tpot_ms: f64,
}

struct PredeclarationBinding {
    value: Value,
    attestation: gate_evidence::FileAttestation,
}

fn main() {
    let exit_code = match real_campaign_main() {
        Ok(true) => 0,
        Ok(false) => 1,
        Err(error) => {
            eprintln!("production A/C campaign preflight/publication error: {error}");
            2
        }
    };
    if exit_code != 0 {
        std::process::exit(exit_code);
    }
}

fn real_campaign_main() -> Result<bool, Box<dyn Error>> {
    if cfg!(debug_assertions) {
        return Err("production A/C campaign gate must be built with --release".into());
    }
    if !cfg!(target_os = "macos") {
        return Err("production A/C campaign gate requires macOS".into());
    }
    if std::env::var_os("APXINF_PERF").is_some() {
        return Err("production A/C campaign requires APXINF_PERF to be unset because its diagnostic output is inside GenerationProfile timing".into());
    }
    let args = parse_args_from(std::env::args_os())?;
    let embedded_candidate_commit = EMBEDDED_CANDIDATE_COMMIT
        .ok_or("release gate was not built with APXINF_CANDIDATE_COMMIT")?;
    if embedded_candidate_commit != args.candidate_commit {
        return Err(format!(
            "embedded candidate commit {embedded_candidate_commit} does not match requested {}",
            args.candidate_commit
        )
        .into());
    }

    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let workspace_dir = manifest_dir
        .parent()
        .and_then(Path::parent)
        .ok_or("apxinf-model manifest is not below the workspace root")?;
    let raw_target = capture_exact_campaign_target(
        &args.output,
        workspace_dir,
        RAW_RECEIPT_RELATIVE_PATH,
        "--output",
    )?;
    let sentinel_path = workspace_dir.join(CAMPAIGN_SENTINEL_RELATIVE_PATH);
    let sentinel_target = capture_exact_campaign_target(
        &sentinel_path,
        workspace_dir,
        CAMPAIGN_SENTINEL_RELATIVE_PATH,
        "campaign sentinel",
    )?;

    let git_start = git_custody(
        workspace_dir,
        &args.candidate_commit,
        GitCustodyPhase::CleanStart,
    )?;
    let custody = gate_evidence::GateCustody::capture_boundary_tail_head_v1(
        &args.model_dir,
        &args.source_lock,
        GATE_SOURCE_NAME,
        GATE_SOURCE_BYTES,
    )?;
    validate_source_lock(custody.source_lock_value())?;
    require_target_outside_model_dir(&raw_target, custody.model_dir(), "--output")?;
    require_target_outside_model_dir(&sentinel_target, custody.model_dir(), "campaign sentinel")?;
    let predeclaration = require_predeclaration_contract(workspace_dir)?;
    let source_custody_start = custody.receipt_json();
    let host_preflight = host_preflight(&args.candidate_commit);
    let schedule = fixed_schedule_json();
    let sentinel = json!({
        "format": CAMPAIGN_SENTINEL_FORMAT,
        "candidate_commit": args.candidate_commit,
        "embedded_candidate_commit": embedded_candidate_commit,
        "baseline_parent_commit": BASELINE_PARENT_COMMIT,
        "predeclaration": gate_evidence::attestation_json(&predeclaration.attestation),
        "binary": source_custody_start["binary"].clone(),
        "raw_output_path": raw_target.requested_path,
        "campaign_start_path": sentinel_target.requested_path,
        "fixed_schedule": schedule,
        "fresh_model_per_slot": true,
        "arbitrary_arm_or_order_cli_exposed": false,
        "retry": false,
        "resampling": false,
        "replacement_after_failure": false,
        "outlier_removal": false,
        "cell_median_substitution": false,
        "schedule_reordering": false,
        "git_custody_start": git_start,
        "source_custody_start": source_custody_start,
        "limitation": "without an external monotonic authority, the sentinel cannot prove that it was never deleted and restored or that the binary/worktree was never copied and executed elsewhere",
    });
    let sentinel_publication_failure = match publish_receipt_create_new(&sentinel_target, &sentinel)
    {
        Ok(()) => None,
        Err(error) => match fs::symlink_metadata(&sentinel_target.requested_path) {
            Ok(_) => Some(format!(
                "campaign sentinel publication reported an error after the fixed path was consumed: {error}"
            )),
            Err(inspect_error) if inspect_error.kind() == std::io::ErrorKind::NotFound => {
                return Err(error);
            }
            Err(inspect_error) => Some(format!(
                "campaign sentinel publication failed and sentinel existence could not be ruled out: {error}; inspection error: {inspect_error}"
            )),
        },
    };

    // From this point the v1 campaign is consumed. Every ordinary execution or
    // quality failure is captured into one partial raw receipt; no slot is
    // retried or replaced.
    let sentinel_attestation = gate_evidence::attest_file(
        &sentinel_target.requested_path,
        "production A/C campaign start sentinel",
        None,
    );
    let mut slots = Vec::with_capacity(14);
    let mut teacher_samples = Vec::with_capacity(4);
    let mut free_samples = Vec::with_capacity(8);
    let execution = if let Some(error) = sentinel_publication_failure {
        Err(error.into())
    } else if let Err(error) = sentinel_attestation.as_ref() {
        Err(format!("start sentinel attestation failed: {error}").into())
    } else {
        execute_fixed_campaign(
            &custody,
            &mut slots,
            &mut teacher_samples,
            &mut free_samples,
        )
    };
    if let Err(error) = execution.as_ref() {
        if slots.is_empty() {
            slots.push(slot_failure_json(
                0,
                "campaign_setup_before_cpu_teacher",
                None,
                &error.to_string(),
            ));
        }
    }

    let source_end = custody.verify_unchanged_receipt();
    let git_end = git_custody(
        workspace_dir,
        &args.candidate_commit,
        GitCustodyPhase::SentinelOnly,
    );
    let sentinel_end = sentinel_attestation.as_ref().map_or_else(
        |error| Err(format!("start sentinel attestation failed: {error}")),
        |attestation| {
            gate_evidence::verify_file_unchanged(
                &sentinel_target.requested_path,
                attestation,
                "production A/C campaign start sentinel",
            )
            .map(|()| gate_evidence::attestation_json(attestation))
            .map_err(|error| error.to_string())
        },
    );

    let (execution_receipt, execution_error, performance_passed) = match execution {
        Ok(value) => {
            let passed = value
                .pointer("/admission/all_performance_thresholds_passed")
                .and_then(Value::as_bool)
                == Some(true);
            (Some(value), None, passed)
        }
        Err(error) => (None, Some(error.to_string()), false),
    };
    let source_end_ok = source_end.is_ok();
    let git_end_ok = git_end.is_ok();
    let sentinel_end_ok = sentinel_end.is_ok();
    let attempted_slots = slots.len();
    let completed_slots = slots
        .iter()
        .filter(|slot| slot.get("passed").and_then(Value::as_bool) == Some(true))
        .count();
    let complete_schedule =
        attempted_slots == 14 && completed_slots == 14 && execution_error.is_none();
    let explicit_continuation_passed = complete_schedule
        && execution_error.is_none()
        && performance_passed
        && source_end_ok
        && git_end_ok
        && sentinel_end_ok;
    let formal_promotion_passed = explicit_continuation_passed
        && host_preflight
            .get("quiet_host_attested")
            .and_then(Value::as_bool)
            == Some(true);
    let sampled_attempt_failures = execution_error.iter().cloned().collect::<Vec<_>>();
    let raw = json!({
        "format": CAMPAIGN_FORMAT,
        "candidate_commit": args.candidate_commit,
        "embedded_candidate_commit": embedded_candidate_commit,
        "baseline_parent_commit": BASELINE_PARENT_COMMIT,
        "predeclaration": {
            "attestation": gate_evidence::attestation_json(&predeclaration.attestation),
            "contract_validated_before_campaign": true,
        },
        "primitive_authorization": predeclaration.value["primitive_authorization"].clone(),
        "cross_runtime_rule": predeclaration.value["cross_runtime_rule"].clone(),
        "campaign_start": sentinel_attestation
            .as_ref()
            .map(gate_evidence::attestation_json)
            .unwrap_or_else(|error| json!({"attestation_error": error.to_string()})),
        "campaign_start_sha256_chained": sentinel_attestation.is_ok(),
        "fixed_schedule": fixed_schedule_json(),
        "slots": slots,
        "attempted_slots": attempted_slots,
        "completed_slots": completed_slots,
        "complete_schedule": complete_schedule,
        "stopped_at_first_failure": execution_error.is_some(),
        "execution_error": execution_error,
        "sampled_attempt_failures": sampled_attempt_failures,
        "performance": execution_receipt,
        "host_preflight": host_preflight,
        "custody": {
            "source_start": custody.receipt_json(),
            "source_end": result_json(source_end),
            "git_end": result_json(git_end),
            "campaign_start_end": result_json(sentinel_end),
            "end_custody_attempted_before_raw_publication": true,
            "expected_campaign_mutation_at_end": CAMPAIGN_SENTINEL_RELATIVE_PATH,
        },
        "policy": {
            "single_release_binary_for_entire_campaign": true,
            "fresh_model_per_slot": true,
            "arbitrary_arm_or_order_cli_exposed": false,
            "retry": false,
            "resampling": false,
            "replacement_after_failure": false,
            "outlier_removal": false,
            "cell_median_substitution": false,
            "schedule_reordering": false,
        },
        "classification": {
            "explicit_full_path_continuation_passed": explicit_continuation_passed,
            "formal_promotion_passed": formal_promotion_passed,
            "default_activation_authorized": false,
        },
        "limitation": "without an external monotonic authority, the sentinel cannot prove that it was never deleted and restored or that the binary/worktree was never copied and executed elsewhere",
        "passed": explicit_continuation_passed,
    });
    publish_receipt_create_new(&raw_target, &raw)?;
    println!("{}", serde_json::to_string(&raw)?);
    Ok(explicit_continuation_passed)
}

fn execute_fixed_campaign(
    custody: &gate_evidence::GateCustody,
    slots: &mut Vec<Value>,
    teacher_samples: &mut Vec<TeacherSample>,
    free_samples: &mut Vec<FreeSample>,
) -> Result<Value, Box<dyn Error>> {
    let tokenizer = Tokenizer::from_bytes(
        custody.pinned_model_artifact_bytes("tokenizer.json")?,
        Some(custody.pinned_model_artifact_bytes("tokenizer_config.json")?),
        Some(custody.pinned_model_artifact_bytes("chat_template.jinja")?),
    )?;
    let formatted = tokenizer.apply_chat_template(&[ChatMessage::user(PROMPT)])?;
    let prompt_tokens = tokenizer.encode(&formatted)?;
    if prompt_tokens != PROMPT_TOKEN_IDS {
        return Err(format!(
            "frozen prompt token mismatch: expected {PROMPT_TOKEN_IDS:?}, got {prompt_tokens:?}"
        )
        .into());
    }
    let config_json = std::str::from_utf8(custody.pinned_model_artifact_bytes("config.json")?)?;
    let config = Qwen35Config::from_json_str(config_json)?;
    validate_official_schedule(&config.text.layer_types)?;
    let vocab_size = config.text.vocab_size;
    let checkpoint_started = Instant::now();
    let (tensors, _) = apxinf_loader::safetensors::load_native_file_filtered(
        custody.pinned_model_artifact_file(LOCKED_CHECKPOINT)?,
        |name| name.starts_with("model.language_model.") || name == "lm_head.weight",
    )?;
    custody.verify_pinned_model_handles_unchanged()?;
    let checkpoint_load_ms = checkpoint_started.elapsed().as_secs_f64() * 1_000.0;
    let max_context = prompt_tokens
        .len()
        .checked_add(STEPS + 1)
        .ok_or("context length overflow")?;

    let cpu_teacher_started = Instant::now();
    let mut cpu_teacher_model = match GeneralQwen35::from_weights(
        config.clone(),
        tensors.clone(),
        Device::Cpu,
        max_context,
    ) {
        Ok(model) => model,
        Err(error) => {
            slots.push(slot_failure_json(
                0,
                "cpu_teacher_oracle",
                None,
                &error.to_string(),
            ));
            return Err(error.into());
        }
    };
    let cpu_teacher_construct_ms = cpu_teacher_started.elapsed().as_secs_f64() * 1_000.0;
    let cpu_teacher_run_started = Instant::now();
    let (cpu_teacher_prefill, teacher_oracle) = match compute_same_process_teacher_oracle(
        &mut cpu_teacher_model,
        &prompt_tokens,
        vocab_size,
    ) {
        Ok(value) => value,
        Err(error) => {
            slots.push(slot_failure_json(
                0,
                "cpu_teacher_oracle",
                None,
                &error.to_string(),
            ));
            return Err(error);
        }
    };
    let cpu_teacher_run_ms = cpu_teacher_run_started.elapsed().as_secs_f64() * 1_000.0;
    drop(cpu_teacher_model);
    slots.push(json!({
        "slot": 0,
        "phase": "cpu_teacher_oracle",
        "arm": null,
        "fresh_model": true,
        "prompt_token_ids": prompt_tokens,
        "prefill_token": cpu_teacher_prefill,
        "teacher_input_ids": teacher_oracle.teacher_inputs,
        "cpu_expected_output_ids": teacher_oracle.expected_outputs,
        "comparisons": STEPS,
        "timing": {
            "construct_ms": cpu_teacher_construct_ms,
            "oracle_ms": cpu_teacher_run_ms,
            "classification": "CPU correctness oracle only; excluded from candidate performance admission",
        },
        "passed": true,
    }));

    let mut first_teacher_trace: Option<TeacherTrace> = None;
    for (offset, arm) in TEACHER_ORDER.into_iter().enumerate() {
        let slot = offset + 1;
        let result = run_candidate_teacher_slot(
            slot,
            arm,
            &config,
            &tensors,
            max_context,
            &prompt_tokens,
            vocab_size,
            cpu_teacher_prefill,
            &teacher_oracle,
        );
        let (mut receipt, trace, body_ns_per_call, mut passed) = match result {
            Ok(value) => value,
            Err(error) => {
                slots.push(slot_failure_json(
                    slot,
                    "candidate_teacher",
                    Some(arm),
                    &error.to_string(),
                ));
                return Err(error);
            }
        };
        let matches_first = first_teacher_trace
            .as_ref()
            .is_none_or(|expected| expected == &trace);
        if first_teacher_trace.is_none() {
            first_teacher_trace = Some(trace);
        }
        passed &= matches_first;
        receipt["all_teacher_token_top4_and_rerank_arrays_match_first"] = json!(matches_first);
        receipt["passed"] = json!(passed);
        slots.push(receipt);
        if !passed {
            return Err(format!(
                "teacher slot {slot} arm {} failed exactness or path admission",
                arm.short()
            )
            .into());
        }
        teacher_samples.push(TeacherSample {
            arm,
            body_ns_per_call,
        });
    }

    let cpu_free_slot = 5usize;
    let cpu_free_started = Instant::now();
    let mut cpu_free_model = match GeneralQwen35::from_weights(
        config.clone(),
        tensors.clone(),
        Device::Cpu,
        max_context,
    ) {
        Ok(model) => model,
        Err(error) => {
            slots.push(slot_failure_json(
                cpu_free_slot,
                "cpu_free_oracle",
                None,
                &error.to_string(),
            ));
            return Err(error.into());
        }
    };
    let cpu_free_construct_ms = cpu_free_started.elapsed().as_secs_f64() * 1_000.0;
    let cpu_free_run_started = Instant::now();
    let (cpu_free_oracle, cpu_free_profile) = match cpu_free_model.generate_streaming(
        LlmInput::text(&prompt_tokens),
        STEPS,
        |_| {},
        None,
    ) {
        Ok(value) => value,
        Err(error) => {
            slots.push(slot_failure_json(
                cpu_free_slot,
                "cpu_free_oracle",
                None,
                &error.to_string(),
            ));
            return Err(error.into());
        }
    };
    let cpu_free_harness_ms = cpu_free_run_started.elapsed().as_secs_f64() * 1_000.0;
    drop(cpu_free_model);
    if cpu_free_oracle.len() != STEPS {
        let error = format!(
            "CPU free oracle returned {} tokens, expected {STEPS}",
            cpu_free_oracle.len()
        );
        slots.push(slot_failure_json(
            cpu_free_slot,
            "cpu_free_oracle",
            None,
            &error,
        ));
        return Err(error.into());
    }
    slots.push(json!({
        "slot": cpu_free_slot,
        "phase": "cpu_free_oracle",
        "arm": null,
        "fresh_model": true,
        "generated_token_ids": cpu_free_oracle,
        "max_new_tokens": STEPS,
        "eos_stopping": false,
        "profile": generation_profile_json_campaign(
            &cpu_free_profile,
            cpu_free_harness_ms,
            json!({"construct_ms": cpu_free_construct_ms}),
            "CPU correctness oracle only; excluded from candidate performance admission",
        ),
        "passed": true,
    }));

    for (block, order, first_slot) in [
        (1usize, FREE_BLOCK_1_ORDER, 6usize),
        (2usize, FREE_BLOCK_2_ORDER, 10usize),
    ] {
        for (offset, arm) in order.into_iter().enumerate() {
            let slot = first_slot + offset;
            let result = run_candidate_free_slot(
                slot,
                block,
                arm,
                &config,
                &tensors,
                max_context,
                &prompt_tokens,
                &cpu_free_oracle,
            );
            let (receipt, tpot_ms, passed) = match result {
                Ok(value) => value,
                Err(error) => {
                    slots.push(slot_failure_json(
                        slot,
                        "candidate_free",
                        Some(arm),
                        &error.to_string(),
                    ));
                    return Err(error);
                }
            };
            slots.push(receipt);
            if !passed {
                return Err(format!(
                    "free slot {slot} block {block} arm {} failed exactness or path admission",
                    arm.short()
                )
                .into());
            }
            free_samples.push(FreeSample {
                arm,
                block,
                tpot_ms,
            });
        }
    }

    if slots.len() != 14 || teacher_samples.len() != 4 || free_samples.len() != 8 {
        return Err("fixed campaign completed with an impossible slot/sample count".into());
    }
    let performance = evaluate_performance(teacher_samples, free_samples)?;
    Ok(json!({
        "checkpoint_load_ms": checkpoint_load_ms,
        "teacher_body": performance["teacher_body"].clone(),
        "free_tpot": performance["free_tpot"].clone(),
        "admission": performance["admission"].clone(),
    }))
}

fn run_candidate_teacher_slot(
    slot: usize,
    arm: Arm,
    config: &Qwen35Config,
    tensors: &HashMap<String, Tensor>,
    max_context: usize,
    prompt_tokens: &[u32],
    vocab_size: usize,
    cpu_prefill_token: u32,
    oracle: &TeacherOracle,
) -> Result<(Value, TeacherTrace, f64, bool), Box<dyn Error>> {
    let construct_started = Instant::now();
    let mut model = construct_candidate_model(arm, config, tensors, max_context)?;
    let construct_ms = construct_started.elapsed().as_secs_f64() * 1_000.0;
    let prefill_started = Instant::now();
    let prefill_logits = model.prefill_for_generation(LlmInput::text(prompt_tokens))?;
    let prefill_ms = prefill_started.elapsed().as_secs_f64() * 1_000.0;
    let prefill_token = argmax(&prefill_logits, vocab_size)?;
    let aggregate = model
        .metal_w8_mlp_stack3_boundary_tail_head_v1_aggregate_ledger()
        .ok_or("candidate constructor omitted aggregate ledger")?;
    let prefill_stats = model
        .metal_w8_mlp_stack3_boundary_tail_head_v1_stats()
        .ok_or("candidate constructor omitted prefill stats")?;
    let prefill_generation = model
        .generation_path_receipt()
        .ok_or("candidate constructor omitted prefill generation receipt")?;
    let prefill_checks = boundary_tail_path_checks(
        arm,
        &prefill_stats,
        &aggregate,
        &prefill_generation,
        0,
        TailPhaseCounts::teacher(0),
    );

    let decode_started = Instant::now();
    let mut normalized_f32_tokens = Vec::with_capacity(STEPS);
    let mut top4_candidates = Vec::with_capacity(STEPS);
    let mut reranked_tokens = Vec::with_capacity(STEPS);
    let mut tail_transaction_elapsed_ns = Vec::with_capacity(STEPS);
    let mut direct_f32_rerank_elapsed_ns = Vec::with_capacity(STEPS);
    for step in 0..STEPS {
        let position = prompt_tokens
            .len()
            .checked_add(step)
            .ok_or("teacher position overflow")?;
        let comparison = model.teacher_forced_decode_candidates(
            oracle.teacher_inputs[step],
            u32::try_from(position)?,
        )?;
        normalized_f32_tokens.push(comparison.cpu_token);
        top4_candidates.push(comparison.w8_candidates);
        reranked_tokens.push(comparison.reranked_token);
        tail_transaction_elapsed_ns.push(comparison.topk_elapsed_ns);
        direct_f32_rerank_elapsed_ns.push(comparison.rerank_elapsed_ns);
    }
    let decode_ms = decode_started.elapsed().as_secs_f64() * 1_000.0;
    let evaluation = evaluate_teacher_candidate(
        oracle,
        &normalized_f32_tokens,
        &top4_candidates,
        &reranked_tokens,
        vocab_size,
    )?;
    let final_stats = model
        .metal_w8_mlp_stack3_boundary_tail_head_v1_stats()
        .ok_or("candidate constructor omitted final stats")?;
    let final_generation = model
        .generation_path_receipt()
        .ok_or("candidate constructor omitted final generation receipt")?;
    let final_checks = boundary_tail_path_checks(
        arm,
        &final_stats,
        &aggregate,
        &final_generation,
        STEPS,
        TailPhaseCounts::teacher(STEPS),
    );
    let body_total_ns = generation_body_elapsed_ns(&final_generation)?;
    let body_ns_per_call = body_total_ns as f64 / STEPS as f64;
    let prefill_token_exact = prefill_token == cpu_prefill_token;
    let passed = prefill_token_exact
        && evaluation.passed
        && prefill_checks.all_valid()
        && final_checks.all_valid();
    let trace = TeacherTrace {
        normalized_f32_tokens: normalized_f32_tokens.clone(),
        top4_candidates: top4_candidates.clone(),
        reranked_tokens: reranked_tokens.clone(),
    };
    let receipt = json!({
        "slot": slot,
        "phase": "candidate_teacher",
        "arm": arm.short(),
        "profile": arm.profile(),
        "gdn_function_chain": arm.function_chain(),
        "fresh_model": true,
        "prompt_token_ids": prompt_tokens,
        "prefill_token": prefill_token,
        "cpu_prefill_token": cpu_prefill_token,
        "prefill_token_exact": prefill_token_exact,
        "teacher_input_ids": oracle.teacher_inputs,
        "frozen_cpu_expected_output_ids": oracle.expected_outputs,
        "tail_normalized_hidden_f32_winner_ids": normalized_f32_tokens,
        "tail_top4_candidate_ids": top4_candidates,
        "direct_tied_f32_reranked_output_ids": reranked_tokens,
        "exactness": {
            "body_token_mismatches": evaluation.body_token_mismatches,
            "top4_mismatches": evaluation.top4_mismatches,
            "direct_rerank_mismatches": evaluation.rerank_mismatches,
            "end_to_end_mismatches": evaluation.end_to_end_mismatches,
            "hidden_tensor_exactness_claimed": false,
        },
        "prefill_generation_path_receipt": prefill_generation,
        "final_generation_path_receipt": final_generation,
        "aggregate_buffer_ledger": aggregate_ledger_json(arm, &aggregate),
        "path_checks": {
            "prefill": prefill_checks.receipt_json(),
            "final": final_checks.receipt_json(),
        },
        "performance": {
            "body_call_count": STEPS,
            "body_total_elapsed_ns": body_total_ns,
            "body_elapsed_ns_per_call": body_ns_per_call,
            "metric": "(initial_stack.block_elapsed_ns + sum(boundaries[*].block_elapsed_ns)) / body_call_count",
            "tail_transaction_elapsed_ns": tail_transaction_elapsed_ns,
            "direct_tied_f32_rerank_elapsed_ns": direct_f32_rerank_elapsed_ns,
            "tail_excluded_from_body_admission": true,
        },
        "timing": {
            "construct_ms": construct_ms,
            "prefill_ms": prefill_ms,
            "decode_ms": decode_ms,
        },
        "passed": passed,
    });
    Ok((receipt, trace, body_ns_per_call, passed))
}

fn run_candidate_free_slot(
    slot: usize,
    block: usize,
    arm: Arm,
    config: &Qwen35Config,
    tensors: &HashMap<String, Tensor>,
    max_context: usize,
    prompt_tokens: &[u32],
    cpu_expected: &[u32],
) -> Result<(Value, f64, bool), Box<dyn Error>> {
    let construct_started = Instant::now();
    let mut model = construct_candidate_model(arm, config, tensors, max_context)?;
    let construct_ms = construct_started.elapsed().as_secs_f64() * 1_000.0;
    let started = Instant::now();
    let (generated, profile) =
        model.generate_streaming(LlmInput::text(prompt_tokens), STEPS, |_| {}, None)?;
    let harness_elapsed_ms = started.elapsed().as_secs_f64() * 1_000.0;
    if generated.len() != STEPS {
        return Err(format!(
            "candidate free slot returned {} tokens, expected {STEPS}",
            generated.len()
        )
        .into());
    }
    let mismatches = evaluate_free_trajectory(cpu_expected, &generated)?;
    let stats = model
        .metal_w8_mlp_stack3_boundary_tail_head_v1_stats()
        .ok_or("candidate constructor omitted free stats")?;
    let aggregate = model
        .metal_w8_mlp_stack3_boundary_tail_head_v1_aggregate_ledger()
        .ok_or("candidate constructor omitted free aggregate ledger")?;
    let generation = model
        .generation_path_receipt()
        .ok_or("candidate constructor omitted free generation receipt")?;
    let body_tail_calls = STEPS - 1;
    let checks = boundary_tail_path_checks(
        arm,
        &stats,
        &aggregate,
        &generation,
        body_tail_calls,
        TailPhaseCounts::free(body_tail_calls),
    );
    let tpot_ms = profile
        .tpot_ms()
        .ok_or("candidate free slot profile omitted TPOT")?;
    if !tpot_ms.is_finite() || tpot_ms <= 0.0 {
        return Err(format!("candidate free slot reported invalid TPOT {tpot_ms}").into());
    }
    let passed = mismatches.is_empty() && checks.all_valid();
    let receipt = json!({
        "slot": slot,
        "phase": "candidate_free",
        "block": block,
        "arm": arm.short(),
        "profile_name": arm.profile(),
        "gdn_function_chain": arm.function_chain(),
        "fresh_model": true,
        "prompt_token_ids": prompt_tokens,
        "cpu_expected_token_ids": cpu_expected,
        "generated_token_ids": generated,
        "mismatches": mismatches,
        "exact_128_token_trajectory": mismatches.is_empty(),
        "final_generation_path_receipt": generation,
        "aggregate_buffer_ledger": aggregate_ledger_json(arm, &aggregate),
        "path_checks": checks.receipt_json(),
        "profile": generation_profile_json_campaign(
            &profile,
            harness_elapsed_ms,
            json!({"construct_ms": construct_ms}),
            "fixed single-process production A/C free slot; TPOT is admission evidence",
        ),
        "passed": passed,
    });
    Ok((receipt, tpot_ms, passed))
}

fn construct_candidate_model(
    arm: Arm,
    config: &Qwen35Config,
    tensors: &HashMap<String, Tensor>,
    max_context: usize,
) -> Result<GeneralQwen35, Box<dyn Error>> {
    let model = match arm {
        Arm::A => GeneralQwen35::from_weights_with_metal_w8_mlp_stack3_boundary_tail_head_v1(
            config.clone(),
            tensors.clone(),
            Device::Cpu,
            max_context,
        )?,
        Arm::C => GeneralQwen35::from_weights_with_metal_w8_mlp_stack3_boundary_tail_head_gdn_core_fused_v1(
            config.clone(),
            tensors.clone(),
            Device::Cpu,
            max_context,
        )?,
    };
    Ok(model)
}

fn generation_body_elapsed_ns(receipt: &Value) -> Result<u128, Box<dyn Error>> {
    let initial = receipt
        .pointer("/initial_stack/block_elapsed_ns")
        .and_then(Value::as_u64)
        .ok_or("generation receipt omitted initial_stack.block_elapsed_ns")?
        as u128;
    let boundaries = receipt
        .get("boundaries")
        .and_then(Value::as_array)
        .ok_or("generation receipt omitted boundaries")?;
    if boundaries.len() != BOUNDARY_REGIONS.len() {
        return Err("generation receipt does not contain five boundary timings".into());
    }
    boundaries.iter().try_fold(initial, |total, boundary| {
        let elapsed = boundary
            .get("block_elapsed_ns")
            .and_then(Value::as_u64)
            .ok_or("generation receipt boundary omitted block_elapsed_ns")?
            as u128;
        total
            .checked_add(elapsed)
            .ok_or_else(|| "body elapsed total overflow".into())
    })
}

fn evaluate_performance(
    teacher: &[TeacherSample],
    free: &[FreeSample],
) -> Result<Value, Box<dyn Error>> {
    let teacher_a = teacher
        .iter()
        .filter(|sample| sample.arm == Arm::A)
        .map(|sample| sample.body_ns_per_call)
        .collect::<Vec<_>>();
    let teacher_c = teacher
        .iter()
        .filter(|sample| sample.arm == Arm::C)
        .map(|sample| sample.body_ns_per_call)
        .collect::<Vec<_>>();
    if teacher_a.len() != 2 || teacher_c.len() != 2 {
        return Err(
            "teacher performance admission requires exactly two A and two C samples".into(),
        );
    }
    let body_a = even_median(&teacher_a)?;
    let body_c = even_median(&teacher_c)?;
    let body_improvement = improvement_percent(body_a, body_c)?;

    let pooled_a = free
        .iter()
        .filter(|sample| sample.arm == Arm::A)
        .map(|sample| sample.tpot_ms)
        .collect::<Vec<_>>();
    let pooled_c = free
        .iter()
        .filter(|sample| sample.arm == Arm::C)
        .map(|sample| sample.tpot_ms)
        .collect::<Vec<_>>();
    if pooled_a.len() != 4 || pooled_c.len() != 4 {
        return Err("free performance admission requires exactly four A and four C samples".into());
    }
    let pooled_a_median = even_median(&pooled_a)?;
    let pooled_c_median = even_median(&pooled_c)?;
    let pooled_improvement = improvement_percent(pooled_a_median, pooled_c_median)?;
    let mut blocks = Vec::with_capacity(2);
    let mut both_blocks_positive = true;
    for block in [1usize, 2usize] {
        let a = free
            .iter()
            .filter(|sample| sample.block == block && sample.arm == Arm::A)
            .map(|sample| sample.tpot_ms)
            .collect::<Vec<_>>();
        let c = free
            .iter()
            .filter(|sample| sample.block == block && sample.arm == Arm::C)
            .map(|sample| sample.tpot_ms)
            .collect::<Vec<_>>();
        if a.len() != 2 || c.len() != 2 {
            return Err(
                format!("free block {block} requires exactly two A and two C samples").into(),
            );
        }
        let a_median = even_median(&a)?;
        let c_median = even_median(&c)?;
        let improvement = improvement_percent(a_median, c_median)?;
        both_blocks_positive &= improvement > 0.0;
        blocks.push(json!({
            "block": block,
            "A_tpot_ms": a,
            "C_tpot_ms": c,
            "A_even_median_tpot_ms": a_median,
            "C_even_median_tpot_ms": c_median,
            "C_over_A_improvement_percent": improvement,
            "strictly_positive": improvement > 0.0,
        }));
    }
    let body_passed = body_improvement >= BODY_IMPROVEMENT_THRESHOLD_PERCENT;
    let pooled_passed = pooled_improvement >= POOLED_TPOT_IMPROVEMENT_THRESHOLD_PERCENT;
    Ok(json!({
        "teacher_body": {
            "metric": "per-slot (initial stack + five boundaries elapsed ns) / 128 teacher calls",
            "per_arm_statistic": "even median of two slots",
            "A_samples_ns_per_call": teacher_a,
            "C_samples_ns_per_call": teacher_c,
            "A_even_median_ns_per_call": body_a,
            "C_even_median_ns_per_call": body_c,
            "C_over_A_improvement_percent": body_improvement,
            "threshold_percent": BODY_IMPROVEMENT_THRESHOLD_PERCENT,
            "passed": body_passed,
        },
        "free_tpot": {
            "metric": "GenerationProfile.tpot_ms",
            "pooled_statistic": "even median of four slots per arm",
            "A_samples_ms": pooled_a,
            "C_samples_ms": pooled_c,
            "A_pooled_even_median_ms": pooled_a_median,
            "C_pooled_even_median_ms": pooled_c_median,
            "pooled_C_over_A_improvement_percent": pooled_improvement,
            "pooled_threshold_percent": POOLED_TPOT_IMPROVEMENT_THRESHOLD_PERCENT,
            "pooled_passed": pooled_passed,
            "blocks": blocks,
            "both_counterbalanced_blocks_strictly_positive": both_blocks_positive,
        },
        "admission": {
            "body_improvement_at_least_3_percent": body_passed,
            "pooled_tpot_improvement_at_least_1_5_percent": pooled_passed,
            "both_free_blocks_strictly_positive": both_blocks_positive,
            "all_performance_thresholds_passed": body_passed && pooled_passed && both_blocks_positive,
            "retry": false,
            "resampling": false,
            "outlier_removal": false,
            "cell_median_substitution": false,
        }
    }))
}

fn even_median(values: &[f64]) -> Result<f64, Box<dyn Error>> {
    if values.is_empty() || values.len() % 2 != 0 || values.iter().any(|value| !value.is_finite()) {
        return Err("even median requires a nonempty even count of finite values".into());
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let high = sorted.len() / 2;
    Ok((sorted[high - 1] + sorted[high]) / 2.0)
}

fn improvement_percent(a: f64, c: f64) -> Result<f64, Box<dyn Error>> {
    if !a.is_finite() || !c.is_finite() || a <= 0.0 || c <= 0.0 {
        return Err(format!("improvement requires positive finite inputs, got A={a} C={c}").into());
    }
    Ok(100.0 * (a - c) / a)
}

fn slot_failure_json(slot: usize, phase: &str, arm: Option<Arm>, error: &str) -> Value {
    json!({
        "slot": slot,
        "phase": phase,
        "arm": arm.map(Arm::short),
        "fresh_model": true,
        "completed": false,
        "error": error,
        "passed": false,
    })
}

fn generation_profile_json_campaign(
    profile: &apxinf_model::GenerationProfile,
    harness_elapsed_ms: f64,
    setup: Value,
    classification: &str,
) -> Value {
    json!({
        "setup": setup,
        "input_tokens": profile.input_tokens(),
        "output_tokens": profile.output_tokens(),
        "ttft_ms": profile.ttft_ms(),
        "tpot_ms": profile.tpot_ms(),
        "generation_tps": profile.generation_tps(),
        "generation_total_latency_ms": profile.total_latency_ms(),
        "harness_elapsed_ms": harness_elapsed_ms,
        "classification": classification,
    })
}

fn fixed_schedule_json() -> Value {
    json!({
        "cpu_teacher_oracles": 1,
        "teacher_order": TEACHER_ORDER.map(Arm::short),
        "cpu_free_oracles": 1,
        "free_block_1_order": FREE_BLOCK_1_ORDER.map(Arm::short),
        "free_block_2_order": FREE_BLOCK_2_ORDER.map(Arm::short),
        "candidate_teacher_slots": 4,
        "candidate_free_slots": 8,
        "total_slots_including_cpu_oracles": 14,
        "APXINF_PERF": "required-unset-before-sentinel",
    })
}

fn capture_exact_campaign_target(
    requested: &Path,
    workspace_dir: &Path,
    expected_relative: &str,
    label: &str,
) -> Result<PinnedOutputTarget, Box<dyn Error>> {
    let expected = workspace_dir.join(expected_relative);
    let target = PinnedOutputTarget::capture(requested)?;
    let expected_parent = fs::canonicalize(
        expected
            .parent()
            .ok_or("fixed campaign target omitted a parent")?,
    )?;
    let expected_name = expected
        .file_name()
        .ok_or("fixed campaign target omitted a file name")?;
    if requested != expected
        || target.canonical_parent != expected_parent
        || target
            .requested_path
            .file_name()
            .is_none_or(|name| name != expected_name)
    {
        return Err(format!(
            "{label} must be the fixed campaign path {}",
            expected.display()
        )
        .into());
    }
    Ok(target)
}

fn require_target_outside_model_dir(
    target: &PinnedOutputTarget,
    model_dir: &Path,
    label: &str,
) -> Result<(), Box<dyn Error>> {
    let canonical_model = fs::canonicalize(model_dir)?;
    if target.canonical_parent.starts_with(canonical_model) {
        return Err(format!("{label} must be outside the frozen model directory").into());
    }
    Ok(())
}

#[derive(Clone, Copy)]
enum GitCustodyPhase {
    CleanStart,
    SentinelOnly,
}

fn git_custody(
    workspace_dir: &Path,
    candidate_commit: &str,
    phase: GitCustodyPhase,
) -> Result<Value, Box<dyn Error>> {
    let git = |arguments: &[&str]| command_output_in("git", arguments, workspace_dir);
    let head = git(&["rev-parse", "HEAD"])?;
    let origin_main = git(&["rev-parse", "origin/main"])?;
    let origin_url = git(&["remote", "get-url", "origin"])?;
    let github_main_line = git(&["ls-remote", "--heads", "origin", "refs/heads/main"])?;
    let github_main = github_main_line
        .split_whitespace()
        .next()
        .ok_or("git ls-remote returned no origin main commit")?;
    let branch = git(&["symbolic-ref", "--short", "HEAD"])?;
    let commit_and_parents = git(&["rev-list", "--parents", "-n", "1", candidate_commit])?;
    let commit_and_parents = commit_and_parents.split_whitespace().collect::<Vec<_>>();
    if commit_and_parents != [candidate_commit, BASELINE_PARENT_COMMIT] {
        return Err(format!(
            "candidate must be one non-merge commit directly above baseline {BASELINE_PARENT_COMMIT}, observed {commit_and_parents:?}"
        )
        .into());
    }
    let changed_paths = git(&[
        "diff-tree",
        "--no-commit-id",
        "--name-only",
        "-r",
        candidate_commit,
    ])?;
    let mut changed_paths = changed_paths.lines().collect::<Vec<_>>();
    changed_paths.sort_unstable();
    if changed_paths != EXPECTED_CANDIDATE_CHANGED_PATHS {
        return Err(format!(
            "candidate changed-path custody mismatch: expected {:?}, observed {changed_paths:?}",
            EXPECTED_CANDIDATE_CHANGED_PATHS
        )
        .into());
    }
    if head != candidate_commit
        || origin_main != candidate_commit
        || github_main != candidate_commit
        || branch != "main"
        || origin_url != EXPECTED_ORIGIN_URL
    {
        return Err(format!(
            "git custody mismatch: head={head} origin/main={origin_main} GitHub/main={github_main} branch={branch} origin_url={origin_url} candidate={candidate_commit}"
        )
        .into());
    }
    let status = git(&["status", "--porcelain=v1", "--untracked-files=all"])?;
    let expected_status = match phase {
        GitCustodyPhase::CleanStart => String::new(),
        GitCustodyPhase::SentinelOnly => format!("?? {CAMPAIGN_SENTINEL_RELATIVE_PATH}"),
    };
    if status != expected_status {
        return Err(format!(
            "git worktree custody mismatch for campaign phase: expected {expected_status:?}, observed {status:?}"
        )
        .into());
    }
    Ok(json!({
        "head": head,
        "origin_main": origin_main,
        "github_main": github_main,
        "baseline_parent": BASELINE_PARENT_COMMIT,
        "candidate_is_one_non_merge_commit_above_baseline": true,
        "candidate_changed_paths": changed_paths,
        "origin_url": origin_url,
        "branch": branch,
        "worktree_state": match phase {
            GitCustodyPhase::CleanStart => "clean",
            GitCustodyPhase::SentinelOnly => "exactly-the-create-new-campaign-sentinel",
        },
    }))
}

fn require_predeclaration_contract(
    workspace_dir: &Path,
) -> Result<PredeclarationBinding, Box<dyn Error>> {
    let path = workspace_dir.join(PREDECLARATION_RELATIVE_PATH);
    let (value, attestation) =
        gate_evidence::read_attested_json(&path, "GDN core-fused production A/C predeclaration")?;
    let changed_paths = value
        .pointer("/custody/candidate_changed_path_whitelist")
        .and_then(Value::as_array)
        .ok_or("predeclaration omitted candidate changed-path whitelist")?
        .iter()
        .map(|entry| {
            entry
                .as_str()
                .ok_or("predeclaration changed path is not a string")
        })
        .collect::<Result<Vec<_>, _>>()?;
    let expected_cross_runtime_rule = json!({
        "frozen_comparison_is_unchanged_until_explicit_full_path_continuation_acceptance": true,
        "current_apxinf_metal_W8_tps": 66.33672849846097,
        "current_llama_cpp_metal_Q8_0_tps": 70.66394293407429,
        "current_llama_cpp_advantage_percent": 6.52310497300709,
        "llama_cpp_commit": "f280b26983ad0fdb705a0d9ebf0503e76f2899b0",
        "first_order_if_full_primitive_absolute_saving_transfers": {
            "projected_apxinf_tpot_ms": 14.47743831496063,
            "projected_apxinf_tps": 69.0729933186194,
            "projected_llama_cpp_advantage_percent": 2.303287491995849,
            "classification": "optimistic arithmetic projection only, never a measured result",
        },
        "after_acceptance": "refresh same-host same-prompt same-128-token greedy comparisons for llama.cpp, Ollama, and omnimind-ai/OmniInfer; OmniInfer reporting must separate llama.cpp core timing from gateway/orchestration wall time",
        "ascend_omni_ai_npu_project": "cross-hardware service metrics only, never a same-hardware engine claim",
    });
    if value.get("format").and_then(Value::as_str) != Some(PREDECLARATION_FORMAT)
        || value.get("baseline_parent_commit").and_then(Value::as_str)
        != Some(BASELINE_PARENT_COMMIT)
        || value.get("status").and_then(Value::as_str)
            != Some("PREDECLARED_BEFORE_PRODUCTION_TOPOLOGY_PERFORMANCE_SAMPLING")
        || value.get("cross_runtime_rule") != Some(&expected_cross_runtime_rule)
        || changed_paths != EXPECTED_CANDIDATE_CHANGED_PATHS
        || value.pointer("/single_process_fixed_campaign/cpu_teacher_oracles")
            .and_then(Value::as_u64)
            != Some(1)
        || value.pointer("/single_process_fixed_campaign/teacher_order")
            != Some(&json!(["C", "A", "A", "C"]))
        || value.pointer("/single_process_fixed_campaign/cpu_free_oracles")
            .and_then(Value::as_u64)
            != Some(1)
        || value.pointer("/single_process_fixed_campaign/free_block_1_order")
            != Some(&json!(["A", "C", "C", "A"]))
        || value.pointer("/single_process_fixed_campaign/free_block_2_order")
            != Some(&json!(["C", "A", "A", "C"]))
        || value.pointer("/single_process_fixed_campaign/candidate_teacher_slots")
            .and_then(Value::as_u64)
            != Some(4)
        || value.pointer("/single_process_fixed_campaign/candidate_free_slots")
            .and_then(Value::as_u64)
            != Some(8)
        || value.pointer("/single_process_fixed_campaign/total_slots_including_cpu_oracles")
            .and_then(Value::as_u64)
            != Some(14)
        || value.pointer("/single_process_fixed_campaign/fresh_model_per_slot")
            .and_then(Value::as_bool)
            != Some(true)
        || value.pointer("/single_process_fixed_campaign/arbitrary_arm_or_order_cli_exposed")
            .and_then(Value::as_bool)
            != Some(false)
        || value.pointer("/single_process_fixed_campaign/retry").and_then(Value::as_bool)
            != Some(false)
        || value.pointer("/single_process_fixed_campaign/resampling").and_then(Value::as_bool)
            != Some(false)
        || value.pointer("/single_process_fixed_campaign/replacement_after_failure")
            .and_then(Value::as_bool)
            != Some(false)
        || value.pointer("/single_process_fixed_campaign/outlier_removal").and_then(Value::as_bool)
            != Some(false)
        || value.pointer("/single_process_fixed_campaign/cell_median_substitution")
            .and_then(Value::as_bool)
            != Some(false)
        || value.pointer("/single_process_fixed_campaign/schedule_reordering")
            .and_then(Value::as_bool)
            != Some(false)
        || value.pointer("/single_process_fixed_campaign/APXINF_PERF_must_be_unset_before_sentinel")
            .and_then(Value::as_bool)
            != Some(true)
        || value.pointer("/scope/model").and_then(Value::as_str) != Some(REPO_ID)
        || value.pointer("/scope/revision").and_then(Value::as_str) != Some(LOCKED_REVISION)
        || value.pointer("/scope/prompt").and_then(Value::as_str) != Some(PROMPT)
        || value.pointer("/scope/prompt_token_ids") != Some(&json!(PROMPT_TOKEN_IDS))
        || value.pointer("/scope/output_tokens").and_then(Value::as_u64) != Some(STEPS as u64)
        || value.pointer("/scope/single_release_binary_for_entire_campaign")
            .and_then(Value::as_bool)
            != Some(true)
        || value.pointer("/arms/A/profile").and_then(Value::as_str) != Some(Arm::A.profile())
        || value.pointer("/arms/A/gdn_function_chain").and_then(Value::as_str)
            != Some(Arm::A.function_chain())
        || value.pointer("/arms/A/persistent_output_groups_per_row").and_then(Value::as_u64)
            != Some(64)
        || value.pointer("/arms/A/core_kernel_output_groups_per_row").and_then(Value::as_u64)
            != Some(Arm::A.core_kernel_output_groups_per_row() as u64)
        || value.pointer("/arms/A/constructor").and_then(Value::as_str)
            != Some("GeneralQwen35::from_weights_with_metal_w8_mlp_stack3_boundary_tail_head_v1")
        || value.pointer("/arms/C/profile").and_then(Value::as_str) != Some(Arm::C.profile())
        || value.pointer("/arms/C/gdn_function_chain").and_then(Value::as_str)
            != Some(Arm::C.function_chain())
        || value.pointer("/arms/C/persistent_output_groups_per_row").and_then(Value::as_u64)
            != Some(64)
        || value.pointer("/arms/C/core_kernel_output_groups_per_row").and_then(Value::as_u64)
            != Some(Arm::C.core_kernel_output_groups_per_row() as u64)
        || value.pointer("/arms/C/constructor").and_then(Value::as_str)
            != Some("GeneralQwen35::from_weights_with_metal_w8_mlp_stack3_boundary_tail_head_gdn_core_fused_v1")
        || value.pointer("/arms/C/threads_per_threadgroup").and_then(Value::as_u64)
            != Some(128)
        || value.pointer("/arms/C/threadgroups_per_seam").and_then(Value::as_u64) != Some(16)
        || value.pointer("/arms/C/internal_threadgroup_barrier_sites").and_then(Value::as_u64)
            != Some(4)
        || value.pointer("/arms/C/pipeline_static_threadgroup_memory_bytes").and_then(Value::as_u64)
            != Some(2_064)
        || value.pointer("/production_mapping/gdn_core_seams_per_decode").and_then(Value::as_u64)
            != Some(18)
        || value.pointer("/production_mapping/changed_transactions_per_decode").and_then(Value::as_u64)
            != Some(6)
        || value.pointer("/production_mapping/initial_stack_linear_layers")
            != Some(&json!([0, 1, 2]))
        || value.pointer("/production_mapping/five_boundary_stack_linear_layers")
            != Some(&json!([[4, 5, 6], [8, 9, 10], [12, 13, 14], [16, 17, 18], [20, 21, 22]]))
        || value.pointer("/precommitted_runtime_topology_per_decode/A/initial_stack/kernel_dispatches").and_then(Value::as_u64)
            != Some(Arm::A.initial_dispatches() as u64)
        || value.pointer("/precommitted_runtime_topology_per_decode/A/initial_stack/explicit_broad_buffer_barriers").and_then(Value::as_u64)
            != Some(Arm::A.initial_broad_barriers() as u64)
        || value.pointer("/precommitted_runtime_topology_per_decode/A/each_boundary/kernel_dispatches").and_then(Value::as_u64)
            != Some(Arm::A.boundary_dispatches() as u64)
        || value.pointer("/precommitted_runtime_topology_per_decode/A/each_boundary/explicit_broad_buffer_barriers").and_then(Value::as_u64)
            != Some(Arm::A.boundary_broad_barriers() as u64)
        || value.pointer("/precommitted_runtime_topology_per_decode/A/five_boundaries/kernel_dispatches").and_then(Value::as_u64)
            != Some((5 * Arm::A.boundary_dispatches()) as u64)
        || value.pointer("/precommitted_runtime_topology_per_decode/A/five_boundaries/explicit_broad_buffer_barriers").and_then(Value::as_u64)
            != Some((5 * Arm::A.boundary_broad_barriers()) as u64)
        || value.pointer("/precommitted_runtime_topology_per_decode/A/full_path/kernel_dispatches").and_then(Value::as_u64)
            != Some(Arm::A.full_dispatches() as u64)
        || value.pointer("/precommitted_runtime_topology_per_decode/A/full_path/explicit_broad_buffer_barriers").and_then(Value::as_u64)
            != Some(Arm::A.full_broad_barriers() as u64)
        || value.pointer("/precommitted_runtime_topology_per_decode/C/full_path/kernel_dispatches").and_then(Value::as_u64)
            != Some(Arm::C.full_dispatches() as u64)
        || value.pointer("/precommitted_runtime_topology_per_decode/C/full_path/explicit_broad_buffer_barriers").and_then(Value::as_u64)
            != Some(Arm::C.full_broad_barriers() as u64)
        || value.pointer("/precommitted_runtime_topology_per_decode/C/initial_stack/kernel_dispatches").and_then(Value::as_u64)
            != Some(Arm::C.initial_dispatches() as u64)
        || value.pointer("/precommitted_runtime_topology_per_decode/C/initial_stack/explicit_broad_buffer_barriers").and_then(Value::as_u64)
            != Some(Arm::C.initial_broad_barriers() as u64)
        || value.pointer("/precommitted_runtime_topology_per_decode/C/each_boundary/kernel_dispatches").and_then(Value::as_u64)
            != Some(Arm::C.boundary_dispatches() as u64)
        || value.pointer("/precommitted_runtime_topology_per_decode/C/each_boundary/explicit_broad_buffer_barriers").and_then(Value::as_u64)
            != Some(Arm::C.boundary_broad_barriers() as u64)
        || value.pointer("/precommitted_runtime_topology_per_decode/C/five_boundaries/kernel_dispatches").and_then(Value::as_u64)
            != Some((5 * Arm::C.boundary_dispatches()) as u64)
        || value.pointer("/precommitted_runtime_topology_per_decode/C/five_boundaries/explicit_broad_buffer_barriers").and_then(Value::as_u64)
            != Some((5 * Arm::C.boundary_broad_barriers()) as u64)
        || value.pointer("/precommitted_runtime_topology_per_decode/common_full_path/compute_encoders").and_then(Value::as_u64)
            != Some(24)
        || value.pointer("/precommitted_runtime_topology_per_decode/common_full_path/command_buffers").and_then(Value::as_u64)
            != Some(7)
        || value.pointer("/precommitted_runtime_topology_per_decode/common_full_path/commits").and_then(Value::as_u64)
            != Some(7)
        || value.pointer("/precommitted_runtime_topology_per_decode/common_full_path/waits").and_then(Value::as_u64)
            != Some(7)
        || value.pointer("/precommitted_runtime_topology_per_decode/common_full_path/host_to_device_bytes").and_then(Value::as_u64)
            != Some(28_672)
        || value.pointer("/precommitted_runtime_topology_per_decode/common_full_path/device_to_host_bytes").and_then(Value::as_u64)
            != Some(28_688)
        || value.pointer("/precommitted_performance_admission/changed_body_metric/C_over_A_improvement_percent_at_least")
            .and_then(Value::as_f64)
            != Some(BODY_IMPROVEMENT_THRESHOLD_PERCENT)
        || value.pointer("/precommitted_performance_admission/changed_body_metric/teacher_body_call_count")
            .and_then(Value::as_u64)
            != Some(STEPS as u64)
        || value.pointer("/precommitted_performance_admission/end_to_end_metric/pooled_C_over_A_improvement_percent_at_least")
            .and_then(Value::as_f64)
            != Some(POOLED_TPOT_IMPROVEMENT_THRESHOLD_PERCENT)
        || value.pointer("/precommitted_performance_admission/end_to_end_metric/both_counterbalanced_blocks_must_be_strictly_positive")
            .and_then(Value::as_bool)
            != Some(true)
        || value.pointer("/precommitted_performance_admission/sampled_attempt_failures_must_be_zero")
            .and_then(Value::as_bool)
            != Some(true)
        || value.pointer("/custody/campaign_start_sentinel/path").and_then(Value::as_str)
            != Some(CAMPAIGN_SENTINEL_RELATIVE_PATH)
        || value.pointer("/custody/raw_receipt/path").and_then(Value::as_str)
            != Some(RAW_RECEIPT_RELATIVE_PATH)
    {
        return Err("production A/C predeclaration contract mismatch".into());
    }
    require_bound_evidence(
        workspace_dir,
        &value,
        "accepted_summary",
        PRIMITIVE_ACCEPTED_SUMMARY_RELATIVE_PATH,
    )?;
    require_bound_evidence(
        workspace_dir,
        &value,
        "raw_receipt",
        PRIMITIVE_RAW_RELATIVE_PATH,
    )?;
    Ok(PredeclarationBinding { value, attestation })
}

fn require_bound_evidence(
    workspace_dir: &Path,
    predeclaration: &Value,
    key: &str,
    expected_path: &str,
) -> Result<(), Box<dyn Error>> {
    let prefix = format!("/primitive_authorization/{key}");
    let declared_path = predeclaration
        .pointer(&format!("{prefix}_path"))
        .and_then(Value::as_str)
        .ok_or("predeclaration omitted primitive evidence path")?;
    let declared_sha = predeclaration
        .pointer(&format!("{prefix}_sha256"))
        .and_then(Value::as_str)
        .ok_or("predeclaration omitted primitive evidence sha256")?;
    let declared_size = predeclaration
        .pointer(&format!("{prefix}_size_bytes"))
        .and_then(Value::as_u64)
        .ok_or("predeclaration omitted primitive evidence size")?;
    if declared_path != expected_path {
        return Err(format!("predeclaration primitive evidence path mismatch for {key}").into());
    }
    let (evidence, actual) = gate_evidence::read_attested_json(
        &workspace_dir.join(expected_path),
        "bound primitive evidence",
    )?;
    if actual.size != declared_size || actual.sha256 != declared_sha {
        return Err(format!("predeclaration primitive evidence SHA-256 mismatch for {key}").into());
    }
    let accepted = match key {
        "raw_receipt" => {
            evidence.get("format").and_then(Value::as_str)
                == Some("apxinf-qwen35-gdn-core-fused-primitive-abc-v1")
                && evidence.get("passed").and_then(Value::as_bool) == Some(true)
                && evidence
                    .get("primitive_continue_gate_passed")
                    .and_then(Value::as_bool)
                    == Some(true)
        }
        "accepted_summary" => {
            evidence.get("format").and_then(Value::as_str)
                == Some("apxinf-qwen35-gdn-core-fused-v1-accepted-diagnostic-summary-v1")
                && evidence.get("status").and_then(Value::as_str)
                    == Some("PASSED_PREDECLARED_PRIMITIVE_CONTINUE_GATE")
        }
        _ => false,
    };
    if !accepted {
        return Err(format!("bound primitive evidence is not accepted for {key}").into());
    }
    Ok(())
}

fn host_preflight(candidate_commit: &str) -> Value {
    let system = |program: &str, arguments: &[&str]| {
        command_output(program, arguments).unwrap_or_else(|error| format!("unavailable: {error}"))
    };
    let process_table = system("ps", &["-Ao", "pid=,pcpu=,comm=", "-r"]);
    let top_processes = process_table
        .lines()
        .take(12)
        .map(str::trim)
        .map(str::to_owned)
        .collect::<Vec<_>>();
    json!({
        "quiet_host_attested": false,
        "classification": "diagnostic unless a separately reviewed strict quiet-host attestation is present; one-shot schedule still cannot be rerun",
        "hardware_model": system("sysctl", &["-n", "hw.model"]),
        "cpu_brand": system("sysctl", &["-n", "machdep.cpu.brand_string"]),
        "os_build": system("sw_vers", &["-buildVersion"]),
        "os_version": system("sw_vers", &["-productVersion"]),
        "power": system("pmset", &["-g", "batt"]),
        "thermal": system("pmset", &["-g", "therm"]),
        "uptime": system("uptime", &[]),
        "top_processes_by_cpu": top_processes,
        "user_or_system_processes_terminated": false,
        "expected_release_build": format!("APXINF_CANDIDATE_COMMIT={candidate_commit} cargo build --release -p apxinf-model --features accelerate,metal-w8 --example qwen35_metal_w8_boundary_tail_head_v1_gate"),
    })
}

fn command_output(program: &str, arguments: &[&str]) -> Result<String, Box<dyn Error>> {
    command_output_in(program, arguments, Path::new("."))
}

fn command_output_in(
    program: &str,
    arguments: &[&str],
    directory: &Path,
) -> Result<String, Box<dyn Error>> {
    let output = Command::new(program)
        .args(arguments)
        .current_dir(directory)
        .output()?;
    if !output.status.success() {
        return Err(format!(
            "{program} {arguments:?} failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        )
        .into());
    }
    Ok(String::from_utf8(output.stdout)?.trim().to_string())
}

fn result_json<T>(result: Result<T, impl std::fmt::Display>) -> Value
where
    T: Into<Value>,
{
    match result {
        Ok(value) => json!({"passed": true, "receipt": value.into()}),
        Err(error) => json!({"passed": false, "error": error.to_string()}),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT_TEMP: AtomicU64 = AtomicU64::new(0);

    fn temp_output(label: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "apxinf-production-ac-campaign-{label}-{}-{}",
            std::process::id(),
            NEXT_TEMP.fetch_add(1, Ordering::Relaxed)
        ))
    }

    #[test]
    fn parser_exposes_only_campaign_identity_and_fixed_artifact_inputs() {
        let args = parse_args_from([
            "gate",
            "--model-dir",
            "/model",
            "--source-lock",
            "/source-lock.json",
            "--candidate-commit",
            "0123456789abcdef0123456789abcdef01234567",
            "--output",
            "/raw.json",
        ])
        .unwrap();
        assert_eq!(
            args.candidate_commit,
            "0123456789abcdef0123456789abcdef01234567"
        );
        for forbidden in ["--arm", "--order", "--mode", "--input-receipt"] {
            assert!(parse_args_from([
                "gate",
                "--model-dir",
                "/model",
                "--source-lock",
                "/source-lock.json",
                "--candidate-commit",
                "0123456789abcdef0123456789abcdef01234567",
                "--output",
                "/raw.json",
                forbidden,
                "A",
            ])
            .is_err());
        }
    }

    #[test]
    fn campaign_schedule_is_exactly_the_predeclared_fourteen_slots() {
        assert_eq!(TEACHER_ORDER.map(Arm::short), ["C", "A", "A", "C"]);
        assert_eq!(FREE_BLOCK_1_ORDER.map(Arm::short), ["A", "C", "C", "A"]);
        assert_eq!(FREE_BLOCK_2_ORDER.map(Arm::short), ["C", "A", "A", "C"]);
        let schedule = fixed_schedule_json();
        assert_eq!(schedule["total_slots_including_cpu_oracles"], 14);
        assert_eq!(schedule["cpu_teacher_oracles"], 1);
        assert_eq!(schedule["cpu_free_oracles"], 1);
        assert_eq!(schedule["APXINF_PERF"], "required-unset-before-sentinel");
    }

    #[test]
    fn campaign_sentinel_publication_is_create_new_and_consumes_the_path() {
        let output = temp_output("sentinel.json");
        let target = PinnedOutputTarget::capture(&output).unwrap();
        let sentinel = json!({"format": CAMPAIGN_SENTINEL_FORMAT});
        publish_receipt_create_new(&target, &sentinel).unwrap();
        let first = fs::read(&output).unwrap();
        assert!(publish_receipt_create_new(&target, &sentinel).is_err());
        assert_eq!(fs::read(&output).unwrap(), first);
        fs::remove_file(output).unwrap();
    }

    #[test]
    fn failed_slot_is_an_explicit_partial_campaign_record() {
        let failure = slot_failure_json(7, "candidate_free", Some(Arm::C), "injected");
        let partial = json!({
            "format": CAMPAIGN_FORMAT,
            "slots": [failure],
            "attempted_slots": 1,
            "completed_slots": 0,
            "complete_schedule": false,
            "stopped_at_first_failure": true,
            "end_custody_attempted_before_raw_publication": true,
            "passed": false,
        });
        assert_eq!(partial["slots"][0]["slot"], 7);
        assert_eq!(partial["slots"][0]["arm"], "C");
        assert_eq!(partial["slots"][0]["passed"], false);
        assert_eq!(partial["attempted_slots"], 1);
        assert_eq!(partial["completed_slots"], 0);
        assert_eq!(partial["complete_schedule"], false);
        assert_eq!(partial["stopped_at_first_failure"], true);
    }

    #[test]
    fn performance_admission_uses_even_medians_and_both_counterbalanced_blocks() {
        let teacher = [
            TeacherSample {
                arm: Arm::A,
                body_ns_per_call: 100.0,
            },
            TeacherSample {
                arm: Arm::C,
                body_ns_per_call: 94.0,
            },
            TeacherSample {
                arm: Arm::C,
                body_ns_per_call: 96.0,
            },
            TeacherSample {
                arm: Arm::A,
                body_ns_per_call: 102.0,
            },
        ];
        let free = [
            FreeSample {
                arm: Arm::A,
                block: 1,
                tpot_ms: 10.0,
            },
            FreeSample {
                arm: Arm::C,
                block: 1,
                tpot_ms: 9.7,
            },
            FreeSample {
                arm: Arm::C,
                block: 1,
                tpot_ms: 9.8,
            },
            FreeSample {
                arm: Arm::A,
                block: 1,
                tpot_ms: 10.1,
            },
            FreeSample {
                arm: Arm::C,
                block: 2,
                tpot_ms: 9.6,
            },
            FreeSample {
                arm: Arm::A,
                block: 2,
                tpot_ms: 10.2,
            },
            FreeSample {
                arm: Arm::A,
                block: 2,
                tpot_ms: 9.9,
            },
            FreeSample {
                arm: Arm::C,
                block: 2,
                tpot_ms: 9.7,
            },
        ];
        let receipt = evaluate_performance(&teacher, &free).unwrap();
        assert_eq!(
            receipt["admission"]["body_improvement_at_least_3_percent"],
            true
        );
        assert_eq!(
            receipt["admission"]["pooled_tpot_improvement_at_least_1_5_percent"],
            true
        );
        assert_eq!(
            receipt["admission"]["both_free_blocks_strictly_positive"],
            true
        );
        assert_eq!(
            receipt["admission"]["all_performance_thresholds_passed"],
            true
        );

        let mut regressed = free;
        regressed[4].tpot_ms = 10.5;
        regressed[7].tpot_ms = 10.6;
        let rejected = evaluate_performance(&teacher, &regressed).unwrap();
        assert_eq!(
            rejected["admission"]["both_free_blocks_strictly_positive"],
            false
        );
        assert_eq!(
            rejected["admission"]["all_performance_thresholds_passed"],
            false
        );
    }

    #[test]
    fn campaign_artifact_targets_are_exact_fixed_paths() {
        let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
        let workspace = manifest_dir.parent().and_then(Path::parent).unwrap();
        let expected = workspace.join(RAW_RECEIPT_RELATIVE_PATH);
        capture_exact_campaign_target(&expected, workspace, RAW_RECEIPT_RELATIVE_PATH, "--output")
            .unwrap();
        let wrong = expected.with_file_name("another-receipt.json");
        assert!(capture_exact_campaign_target(
            &wrong,
            workspace,
            RAW_RECEIPT_RELATIVE_PATH,
            "--output",
        )
        .is_err());
    }

    #[test]
    fn topology_constants_lock_a_and_c_full_path_counts() {
        assert_eq!(
            (Arm::A.full_dispatches(), Arm::A.full_broad_barriers()),
            (267, 243)
        );
        assert_eq!(
            (Arm::C.full_dispatches(), Arm::C.full_broad_barriers()),
            (213, 189)
        );
        assert_eq!(
            (Arm::A.initial_dispatches(), Arm::A.initial_broad_barriers()),
            (39, 36)
        );
        assert_eq!(
            (Arm::C.initial_dispatches(), Arm::C.initial_broad_barriers()),
            (30, 27)
        );
        assert_eq!(
            (
                Arm::A.boundary_dispatches(),
                Arm::A.boundary_broad_barriers()
            ),
            (44, 40)
        );
        assert_eq!(
            (
                Arm::C.boundary_dispatches(),
                Arm::C.boundary_broad_barriers()
            ),
            (35, 31)
        );
    }

    #[test]
    fn live_gdn_core_receipt_validator_locks_both_production_profiles() {
        for arm in [Arm::A, Arm::C] {
            let profile = arm.core_profile();
            let receipt = apxinf_metal::GdnCoreProductionReceiptV1 {
                profile,
                function_chain: arm.function_chain(),
                gdn_core_seams: 3,
                persistent_output_groups_per_row: 64,
                core_kernel_output_groups_per_row: arm.core_kernel_output_groups_per_row(),
                kernel_dispatches: profile.gdn_core_dispatches_for_seams(3),
                explicit_buffer_barriers: profile.gdn_core_dispatches_for_seams(3),
                recurrent_or_fused_threads_per_threadgroup: profile
                    .recurrent_threads_per_threadgroup(),
                threadgroups: profile.gdn_core_threadgroups_for_seams(3),
                launched_threads: profile.gdn_core_launched_threads_for_seams(3),
                pipeline_thread_execution_width: 32,
                source_declared_threadgroup_memory_bytes: profile
                    .source_declared_threadgroup_memory_bytes(),
                pipeline_static_threadgroup_memory_bytes: profile
                    .expected_pipeline_static_threadgroup_memory_bytes(),
                internal_threadgroup_barrier_sites_per_threadgroup: profile
                    .internal_threadgroup_barrier_sites_per_threadgroup(),
                fixed_shape_validated: true,
                rms_norm_eps_bits: 1.0e-6_f32.to_bits(),
            };
            assert!(production_receipt_struct_is_exact(Some(receipt), arm, 128));
            assert!(production_receipt_struct_is_exact(None, arm, 0));
            let receipt_json = json!({
                "profile": arm.profile(),
                "function_chain": arm.function_chain(),
                "gdn_core_seams": 3,
                "persistent_output_groups_per_row": 64,
                "core_kernel_output_groups_per_row": arm.core_kernel_output_groups_per_row(),
                "kernel_dispatches": profile.gdn_core_dispatches_for_seams(3),
                "explicit_buffer_barriers": profile.gdn_core_dispatches_for_seams(3),
                "recurrent_or_fused_threads_per_threadgroup": profile.recurrent_threads_per_threadgroup(),
                "threadgroups": profile.gdn_core_threadgroups_for_seams(3),
                "launched_threads": profile.gdn_core_launched_threads_for_seams(3),
                "pipeline_thread_execution_width": 32,
                "source_declared_threadgroup_memory_bytes": profile.source_declared_threadgroup_memory_bytes(),
                "pipeline_static_threadgroup_memory_bytes": profile.expected_pipeline_static_threadgroup_memory_bytes(),
                "internal_threadgroup_barrier_sites_per_threadgroup": profile.internal_threadgroup_barrier_sites_per_threadgroup(),
                "fixed_shape_validated": true,
                "rms_norm_eps_bits": 1.0e-6_f32.to_bits(),
            });
            assert!(production_receipt_json_is_exact(&receipt_json, arm, 128));
            let mut wrong_json = receipt_json;
            wrong_json["core_kernel_output_groups_per_row"] =
                json!(arm.core_kernel_output_groups_per_row() + 1);
            assert!(!production_receipt_json_is_exact(&wrong_json, arm, 128));
            let mut wrong = receipt;
            wrong.kernel_dispatches += 1;
            assert!(!production_receipt_struct_is_exact(Some(wrong), arm, 128));
            let mut wrong_groups = receipt;
            wrong_groups.core_kernel_output_groups_per_row += 1;
            assert!(!production_receipt_struct_is_exact(
                Some(wrong_groups),
                arm,
                128
            ));
            assert!(!production_receipt_struct_is_exact(Some(receipt), arm, 0));
        }
    }

    #[test]
    fn checked_in_predeclaration_matches_the_campaign_contract_and_bound_evidence() {
        let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
        let workspace = manifest_dir.parent().and_then(Path::parent).unwrap();
        require_predeclaration_contract(workspace).unwrap();
    }
}
