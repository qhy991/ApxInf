//! Same-binary four-mode gate for the explicit Qwen3.5 MLP→Stack3 boundary
//! body + fused tail-head v1 diagnostic. This example is not reachable from
//! CLI, AutoModel, registry, or any default constructor.

#[path = "support/qwen35_boundary_tail_head_v1_gate_evidence.rs"]
mod gate_evidence;

use std::ffi::OsString;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::{error::Error, time::Instant};

use serde_json::{json, Value};

use apxinf_core::{Device, Tensor};
#[cfg(test)]
use apxinf_model::qwen35::general::{
    Qwen35MetalW8LinearLayerStack3BufferLedger, Qwen35MetalW8MlpStack3BoundaryRegionBufferLedgerV1,
};
use apxinf_model::qwen35::general::{
    Qwen35MetalW8MlpStack3BoundaryTailHeadV1AggregateLedger,
    Qwen35MetalW8MlpStack3BoundaryTailHeadV1Stats,
};
use apxinf_model::{GeneralQwen35, LlmInput, LlmTrait, Qwen35Config, Qwen35LayerType};
use apxinf_tokenizer::{ChatMessage, Tokenizer};

const CPU_TEACHER_FORMAT: &str = "apxinf-qwen35-metal-w8-boundary-tail-head-v1-cpu-teacher-v1";
const CANDIDATE_TEACHER_FORMAT: &str =
    "apxinf-qwen35-metal-w8-boundary-tail-head-v1-teacher-gate-v1";
const CPU_FREE_FORMAT: &str = "apxinf-qwen35-metal-w8-boundary-tail-head-v1-cpu-free-run-v1";
const CANDIDATE_FREE_FORMAT: &str = "apxinf-qwen35-metal-w8-boundary-tail-head-v1-free-run-gate-v1";
const SOURCE_LOCK_FORMAT: &str = "apxinf-hf-source-lock-v1";
const REPO_ID: &str = "Qwen/Qwen3.5-0.8B";
const LOCKED_REVISION: &str = "2fc06364715b967f1860aea9cf38778875588b17";
const LOCKED_CHECKPOINT: &str = "model.safetensors-00001-of-00001.safetensors";
const LOCKED_CHECKPOINT_SHA256: &str =
    "04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696";
const LOCKED_CHECKPOINT_BYTES: u64 = 1_746_942_600;
const PROMPT: &str = "Hello";
const GATE_SOURCE_NAME: &str = "qwen35_metal_w8_boundary_tail_head_v1_gate.rs";
const GATE_SOURCE_BYTES: &[u8] = include_bytes!("qwen35_metal_w8_boundary_tail_head_v1_gate.rs");

struct RunResult {
    receipt: Value,
    passed: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mode {
    CpuTeacher,
    BoundaryTailV1Teacher,
    CpuFree,
    BoundaryTailV1Free,
}

impl Mode {
    fn parse(value: &str) -> Result<Self, String> {
        match value {
            "cpu-teacher" => Ok(Self::CpuTeacher),
            "boundary-tail-v1-teacher" => Ok(Self::BoundaryTailV1Teacher),
            "cpu-free" => Ok(Self::CpuFree),
            "boundary-tail-v1-free" => Ok(Self::BoundaryTailV1Free),
            other => Err(format!("invalid --mode {other:?}")),
        }
    }

    const fn label(self) -> &'static str {
        match self {
            Self::CpuTeacher => "cpu_teacher",
            Self::BoundaryTailV1Teacher => "metal_w8_boundary_tail_v1_teacher_forced",
            Self::CpuFree => "cpu_free_run",
            Self::BoundaryTailV1Free => "metal_w8_boundary_tail_v1_free_run",
        }
    }

    const fn receipt_format(self) -> &'static str {
        match self {
            Self::CpuTeacher => CPU_TEACHER_FORMAT,
            Self::BoundaryTailV1Teacher => CANDIDATE_TEACHER_FORMAT,
            Self::CpuFree => CPU_FREE_FORMAT,
            Self::BoundaryTailV1Free => CANDIDATE_FREE_FORMAT,
        }
    }

    const fn requires_input_receipt(self) -> bool {
        matches!(self, Self::BoundaryTailV1Teacher | Self::BoundaryTailV1Free)
    }

    const fn is_candidate(self) -> bool {
        self.requires_input_receipt()
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
    ledger: apxinf_metal::LinearLayerStack3BufferLedger,
) -> bool {
    ledger.allocated_buffers == 76
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
        && ledger.commits_per_decode == 1
        && ledger.waits_per_decode == 1
        && ledger.intermediate_host_finite_checks_per_decode == 0
        && ledger.final_output_finite_checks_per_decode == 1
}

fn official_boundary_ledger_is_exact(
    ledger: apxinf_metal::MlpStack3BoundaryBufferLedgerV1,
) -> bool {
    ledger.scope == "resident-mtlbuffer-only"
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
        && ledger.kernel_dispatches_per_decode == 44
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
    aggregate: &Qwen35MetalW8MlpStack3BoundaryTailHeadV1AggregateLedger,
) -> bool {
    if aggregate.scope != "resident-mtlbuffer-only"
        || aggregate.exclusions
            != "CPU F32 weights, host F32 tied embedding and exact four-candidate rerank, host allocations, Metal pipelines/libraries/queues, command objects, driver allocations, full attention/KV, model loader, and prefill CPU head"
        || !aggregate.includes_lm_head
        || aggregate.initial_stack.layer_indices != [0, 1, 2]
        || !official_initial_stack_ledger_is_exact(aggregate.initial_stack.ledger)
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
                    && official_boundary_ledger_is_exact(entry.ledger)
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
        .try_fold(3usize * 13, |total, entry| {
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
        && aggregate.kernel_dispatches_per_decode == 267
        && aggregate.commits_per_decode == 7
        && aggregate.waits_per_decode == 7
}

fn stack3_ledger_json(ledger: apxinf_metal::LinearLayerStack3BufferLedger) -> Value {
    json!({
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
        "commits_per_decode": ledger.commits_per_decode,
        "waits_per_decode": ledger.waits_per_decode,
        "intermediate_host_finite_checks_per_decode": ledger.intermediate_host_finite_checks_per_decode,
        "final_output_finite_checks_per_decode": ledger.final_output_finite_checks_per_decode,
    })
}

fn boundary_ledger_json(ledger: apxinf_metal::MlpStack3BoundaryBufferLedgerV1) -> Value {
    json!({
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
    ledger: &Qwen35MetalW8MlpStack3BoundaryTailHeadV1AggregateLedger,
) -> Value {
    json!({
        "scope": ledger.scope,
        "exclusions": ledger.exclusions,
        "includes_lm_head": ledger.includes_lm_head,
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
        "commits_per_decode": ledger.commits_per_decode,
        "waits_per_decode": ledger.waits_per_decode,
        "component_sum_recomputed_and_exact": official_aggregate_ledger_is_exact(ledger),
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
    receipt: &Value,
    body_tail_calls: usize,
    phase: TailPhaseCounts,
) -> bool {
    if phase.prefill != 0
        || phase.decode.checked_add(phase.teacher) != Some(body_tail_calls)
        || receipt.get("format").and_then(Value::as_str)
            != Some("apxinf-qwen35-mlp-stack3-boundary-tail-head-generation-path-v1")
        || receipt.get("mechanism").and_then(Value::as_str)
            != Some("metal-w8-mlp-stack3-boundary-tail-head-v1")
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
        && initial.get("mechanism").and_then(Value::as_str)
            == Some("metal-w8-linear-layer-stack3-v1")
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
                        == Some("metal-w8-mlp-stack3-boundary-v1")
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
            == Some(267)
        && aggregate.get("commits_per_decode").and_then(Value::as_u64) == Some(7)
        && aggregate.get("waits_per_decode").and_then(Value::as_u64) == Some(7);
    initial_valid && boundaries_valid && prefill_valid && tail_valid && aggregate_valid
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
        mechanism_and_precision_valid: stats.mechanism
            == "metal-w8-mlp-stack3-boundary-tail-head-v1"
            && stats.initial_stack.mechanism == "metal-w8-linear-layer-stack3-v1"
            && stats
                .initial_stack
                .quantization
                .iter()
                .copied()
                .all(quantization_profile_is_exact)
            && stats.boundaries.iter().all(|region| {
                region.mechanism == "metal-w8-mlp-stack3-boundary-v1"
                    && region
                        .quantization
                        .iter()
                        .copied()
                        .all(quantization_profile_is_exact)
            }),
        six_region_execution_valid: initial_valid && boundaries_valid,
        tail_execution_and_phase_valid: tail_valid,
        aggregate_ledger_valid: official_aggregate_ledger_is_exact(aggregate),
        generation_receipt_valid: boundary_tail_generation_receipt_is_exact(
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

fn json_u32_array(value: &Value, field: &str) -> Result<Vec<u32>, String> {
    value
        .get(field)
        .and_then(Value::as_array)
        .ok_or_else(|| format!("{field} must be an array"))?
        .iter()
        .map(|item| {
            item.as_u64()
                .and_then(|number| u32::try_from(number).ok())
                .ok_or_else(|| format!("{field} must contain only u32 values"))
        })
        .collect()
}

fn validate_cpu_teacher_receipt(
    receipt: &Value,
    expected_identity: &Value,
    prompt_tokens: &[u32],
    prefill_token: u32,
) -> Result<TeacherOracle, String> {
    if receipt.get("format").and_then(Value::as_str) != Some(CPU_TEACHER_FORMAT)
        || receipt.get("mode").and_then(Value::as_str) != Some(Mode::CpuTeacher.label())
        || receipt.get("passed").and_then(Value::as_bool) != Some(true)
        || receipt.get("identity") != Some(expected_identity)
        || receipt.get("comparisons").and_then(Value::as_u64) != Some(STEPS as u64)
        || receipt.get("prefill_token").and_then(Value::as_u64) != Some(prefill_token as u64)
        || json_u32_array(receipt, "prompt_token_ids")? != prompt_tokens
    {
        return Err("CPU teacher receipt does not match this frozen request".into());
    }
    let teacher_inputs = json_u32_array(receipt, "teacher_input_ids")?;
    let expected_outputs = json_u32_array(receipt, "cpu_expected_output_ids")?;
    if teacher_inputs.len() != STEPS
        || expected_outputs.len() != STEPS
        || teacher_inputs.first().copied() != Some(prefill_token)
        || teacher_inputs[1..] != expected_outputs[..STEPS - 1]
    {
        return Err("CPU teacher receipt does not contain the exact 128-step teacher chain".into());
    }
    Ok(TeacherOracle {
        teacher_inputs,
        expected_outputs,
    })
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

fn validate_cpu_free_receipt(
    receipt: &Value,
    expected_identity: &Value,
    prompt_tokens: &[u32],
) -> Result<Vec<u32>, String> {
    if receipt.get("format").and_then(Value::as_str) != Some(CPU_FREE_FORMAT)
        || receipt.get("mode").and_then(Value::as_str) != Some(Mode::CpuFree.label())
        || receipt.get("passed").and_then(Value::as_bool) != Some(true)
        || receipt.get("identity") != Some(expected_identity)
        || receipt.get("max_new_tokens").and_then(Value::as_u64) != Some(STEPS as u64)
        || receipt.get("eos_stopping").and_then(Value::as_bool) != Some(false)
        || json_u32_array(receipt, "prompt_token_ids")? != prompt_tokens
    {
        return Err("CPU free receipt does not match this frozen request".into());
    }
    let tokens = json_u32_array(receipt, "generated_token_ids")?;
    if tokens.len() != STEPS {
        return Err("CPU free receipt must contain exactly 128 generated tokens".into());
    }
    Ok(tokens)
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
    mode: Mode,
    input_receipt: Option<PathBuf>,
    output: PathBuf,
}

fn usage() -> &'static str {
    "Usage: qwen35_metal_w8_boundary_tail_head_v1_gate \
  --model-dir OFFICIAL_LOCAL_QWEN35_0_8B \
  --source-lock SOURCE_LOCK.json \
  --mode cpu-teacher|boundary-tail-v1-teacher|cpu-free|boundary-tail-v1-free \
  [--input-receipt CPU_RECEIPT.json] \
  --output NEW_RECEIPT.json"
}

fn parse_args_from<I, S>(args: I) -> Result<Args, String>
where
    I: IntoIterator<Item = S>,
    S: Into<OsString>,
{
    let mut model_dir = None;
    let mut source_lock = None;
    let mut mode = None;
    let mut input_receipt = None;
    let mut output = None;
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
            "--mode" => {
                let parsed = Mode::parse(&value.to_string_lossy())?;
                if mode.replace(parsed).is_some() {
                    return Err("duplicate --mode".into());
                }
            }
            "--input-receipt" => {
                if input_receipt.replace(PathBuf::from(value)).is_some() {
                    return Err("duplicate --input-receipt".into());
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
        mode: mode.ok_or_else(|| format!("--mode is required\n{}", usage()))?,
        input_receipt,
        output: output.ok_or_else(|| format!("--output is required\n{}", usage()))?,
    };
    if !args.model_dir.is_absolute()
        || !args.source_lock.is_absolute()
        || !args.output.is_absolute()
        || args
            .input_receipt
            .as_ref()
            .is_some_and(|path| !path.is_absolute())
    {
        return Err("all gate paths must be absolute".into());
    }
    if args.output.exists() {
        return Err("--output must not already exist".into());
    }
    if args.mode.requires_input_receipt() != args.input_receipt.is_some() {
        return Err(if args.mode.requires_input_receipt() {
            "candidate modes require --input-receipt".into()
        } else {
            "CPU modes reject --input-receipt".into()
        });
    }
    Ok(args)
}

fn publish_receipt_create_new(
    path: &Path,
    receipt: &Value,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut bytes = serde_json::to_vec(receipt)?;
    bytes.push(b'\n');
    let mut output = OpenOptions::new().write(true).create_new(true).open(path)?;
    output.write_all(&bytes)?;
    output.sync_all()?;
    Ok(())
}

fn validate_receipt_paths_outside_model_dir(
    model_dir: &Path,
    input_receipt: Option<&PathBuf>,
    output: &Path,
) -> Result<(), String> {
    let canonical_model = std::fs::canonicalize(model_dir)
        .map_err(|error| format!("cannot canonicalize model directory: {error}"))?;
    if let Some(input) = input_receipt {
        let canonical_input = std::fs::canonicalize(input)
            .map_err(|error| format!("cannot canonicalize input receipt: {error}"))?;
        if canonical_input.starts_with(&canonical_model) {
            return Err("--input-receipt must be outside the frozen model directory".into());
        }
    }
    let output_parent = output
        .parent()
        .ok_or_else(|| "--output must have a parent directory".to_string())?;
    let canonical_output_parent = std::fs::canonicalize(output_parent)
        .map_err(|error| format!("cannot canonicalize output parent directory: {error}"))?;
    if canonical_output_parent.starts_with(&canonical_model) {
        return Err("--output must be outside the frozen model directory".into());
    }
    Ok(())
}

fn finalize_run_with_end_custody<F>(
    mut result: RunResult,
    output: &Path,
    verify_end_custody: F,
) -> Result<RunResult, Box<dyn Error>>
where
    F: FnOnce() -> Result<Value, Box<dyn Error>>,
{
    let custody_end_verification = verify_end_custody()?;
    result
        .receipt
        .as_object_mut()
        .ok_or("gate receipt root must be an object")?
        .insert("custody_end_verification".into(), custody_end_verification);
    publish_receipt_create_new(output, &result.receipt)?;
    Ok(result)
}

fn generation_profile_json(
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

fn free_timing_classification(mode: Mode) -> &'static str {
    match mode {
        Mode::CpuFree => "CPU reference single-pass diagnostic timing only; never promotion evidence",
        Mode::BoundaryTailV1Free => {
            "candidate-only single pass under an uncontrolled host; never formal or promotion evidence"
        }
        Mode::CpuTeacher | Mode::BoundaryTailV1Teacher => {
            unreachable!("free timing classification is only defined for free modes")
        }
    }
}

fn main() {
    let exit_code = match real_main() {
        Ok(true) => 0,
        Ok(false) => 1,
        Err(error) => {
            eprintln!("boundary-tail v1 gate preflight/execution error: {error}");
            2
        }
    };
    if exit_code != 0 {
        std::process::exit(exit_code);
    }
}

fn real_main() -> Result<bool, Box<dyn Error>> {
    if cfg!(debug_assertions) {
        return Err(
            "qwen35_metal_w8_boundary_tail_head_v1_gate must be built with --release".into(),
        );
    }
    if !cfg!(target_os = "macos") {
        return Err("qwen35_metal_w8_boundary_tail_head_v1_gate requires macOS".into());
    }
    let args = parse_args_from(std::env::args_os())?;
    if args.output.exists() {
        return Err(format!(
            "refusing to replace existing receipt {}",
            args.output.display()
        )
        .into());
    }
    let custody = gate_evidence::GateCustody::capture_boundary_tail_head_v1(
        &args.model_dir,
        &args.source_lock,
        GATE_SOURCE_NAME,
        GATE_SOURCE_BYTES,
    )?;
    validate_source_lock(custody.source_lock_value())?;
    validate_receipt_paths_outside_model_dir(
        custody.model_dir(),
        args.input_receipt.as_ref(),
        &args.output,
    )?;
    let canonical_model_dir = custody.model_dir().to_path_buf();
    let canonical_source_lock = fs::canonicalize(&args.source_lock)?;
    let binary_path = fs::canonicalize(std::env::current_exe()?)?;

    let tokenizer = Tokenizer::from_file(canonical_model_dir.join("tokenizer.json"))?;
    let formatted = tokenizer.apply_chat_template(&[ChatMessage::user(PROMPT)])?;
    let prompt_tokens = tokenizer.encode(&formatted)?;
    let config = Qwen35Config::from_json_file(&canonical_model_dir.join("config.json"))?;
    validate_official_schedule(&config.text.layer_types)?;
    let vocab_size = config.text.vocab_size;

    let checkpoint_started = Instant::now();
    let (tensors, _) =
        apxinf_loader::safetensors::load_native_path_filtered(&canonical_model_dir, |name| {
            name.starts_with("model.language_model.") || name == "lm_head.weight"
        })?;
    let checkpoint_load_ms = checkpoint_started.elapsed().as_secs_f64() * 1_000.0;
    let max_context = prompt_tokens
        .len()
        .checked_add(STEPS + 1)
        .ok_or("context length overflow")?;
    let construct_started = Instant::now();
    let mut model = if args.mode.is_candidate() {
        GeneralQwen35::from_weights_with_metal_w8_mlp_stack3_boundary_tail_head_v1(
            config,
            tensors,
            Device::Cpu,
            max_context,
        )?
    } else {
        GeneralQwen35::from_weights(config, tensors, Device::Cpu, max_context)?
    };
    let model_construct_ms = construct_started.elapsed().as_secs_f64() * 1_000.0;
    let identity = json!({
        "repo_id": REPO_ID,
        "revision": LOCKED_REVISION,
        "checkpoint": LOCKED_CHECKPOINT,
        "checkpoint_sha256": LOCKED_CHECKPOINT_SHA256,
        "checkpoint_bytes": LOCKED_CHECKPOINT_BYTES,
        "model_dir": canonical_model_dir,
        "source_lock": canonical_source_lock,
        "binary_path": binary_path,
        "build_profile": "release",
        "matmul_feature": "accelerate",
        "metal_feature": "metal-w8",
        "cpu_reference_constructor": "GeneralQwen35::from_weights",
        "candidate_constructor": "GeneralQwen35::from_weights_with_metal_w8_mlp_stack3_boundary_tail_head_v1",
        "custody": custody.receipt_json(),
    });
    let setup = json!({
        "checkpoint_load_ms": checkpoint_load_ms,
        "model_construct_ms": model_construct_ms,
        "timing_classification": "single-pass diagnostic timing only; never formal promotion evidence",
    });
    let result = match args.mode {
        Mode::CpuTeacher | Mode::BoundaryTailV1Teacher => run_teacher(
            &args,
            &mut model,
            &prompt_tokens,
            vocab_size,
            identity,
            setup,
        )?,
        Mode::CpuFree | Mode::BoundaryTailV1Free => {
            run_free(&args, &mut model, &prompt_tokens, identity, setup)?
        }
    };
    let result =
        finalize_run_with_end_custody(result, &args.output, || custody.verify_unchanged_receipt())?;
    println!("{}", serde_json::to_string(&result.receipt)?);
    Ok(result.passed)
}

fn run_teacher(
    args: &Args,
    model: &mut GeneralQwen35,
    prompt_tokens: &[u32],
    vocab_size: usize,
    identity: Value,
    setup: Value,
) -> Result<RunResult, Box<dyn Error>> {
    let prefill_started = Instant::now();
    let prefill_logits = model.prefill_for_generation(LlmInput::text(prompt_tokens))?;
    let prefill_ms = prefill_started.elapsed().as_secs_f64() * 1_000.0;
    let prefill_token = argmax(&prefill_logits, vocab_size)?;

    if args.mode == Mode::CpuTeacher {
        let decode_started = Instant::now();
        let mut teacher = prefill_token;
        let mut teacher_inputs = Vec::with_capacity(STEPS);
        let mut expected_outputs = Vec::with_capacity(STEPS);
        for step in 0..STEPS {
            teacher_inputs.push(teacher);
            let position = prompt_tokens
                .len()
                .checked_add(step)
                .ok_or("teacher position overflow")?;
            let logits = model.forward(&[teacher], u32::try_from(position)?)?;
            teacher = argmax(&logits, vocab_size)?;
            expected_outputs.push(teacher);
        }
        let decode_ms = decode_started.elapsed().as_secs_f64() * 1_000.0;
        return Ok(RunResult {
            receipt: json!({
                "format": args.mode.receipt_format(),
                "mode": args.mode.label(),
                "identity": identity,
                "prompt": PROMPT,
                "prompt_token_ids": prompt_tokens,
                "official_layer_schedule_valid": true,
                "comparisons": STEPS,
                "prefill_token": prefill_token,
                "teacher_input_ids": teacher_inputs,
                "cpu_expected_output_ids": expected_outputs,
                "generation_path_contract": null,
                "timing": {
                    "setup": setup,
                    "prefill_ms": prefill_ms,
                    "decode_ms": decode_ms,
                    "decode_mean_ms": decode_ms / STEPS as f64,
                    "classification": "CPU reference single-pass diagnostic timing only; never formal or promotion evidence",
                },
                "passed": true,
            }),
            passed: true,
        });
    }

    let aggregate = model
        .metal_w8_mlp_stack3_boundary_tail_head_v1_aggregate_ledger()
        .ok_or("boundary-tail v1 constructor omitted the aggregate ledger")?;
    let prefill_stats = model
        .metal_w8_mlp_stack3_boundary_tail_head_v1_stats()
        .ok_or("boundary-tail v1 constructor omitted prefill stats")?;
    let prefill_generation = model
        .generation_path_receipt()
        .ok_or("boundary-tail v1 constructor omitted prefill generation receipt")?;
    let prefill_checks = boundary_tail_path_checks(
        &prefill_stats,
        &aggregate,
        &prefill_generation,
        0,
        TailPhaseCounts::teacher(0),
    );
    let input_path = args
        .input_receipt
        .as_ref()
        .ok_or("candidate teacher mode requires --input-receipt")?;
    let (cpu_receipt, input_attestation) =
        gate_evidence::read_attested_json(input_path, "boundary-tail v1 CPU teacher receipt")?;
    let oracle =
        validate_cpu_teacher_receipt(&cpu_receipt, &identity, prompt_tokens, prefill_token)?;

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
        &oracle,
        &normalized_f32_tokens,
        &top4_candidates,
        &reranked_tokens,
        vocab_size,
    )?;
    let final_stats = model
        .metal_w8_mlp_stack3_boundary_tail_head_v1_stats()
        .ok_or("boundary-tail v1 constructor omitted final stats")?;
    let final_generation = model
        .generation_path_receipt()
        .ok_or("boundary-tail v1 constructor omitted final generation receipt")?;
    let final_checks = boundary_tail_path_checks(
        &final_stats,
        &aggregate,
        &final_generation,
        STEPS,
        TailPhaseCounts::teacher(STEPS),
    );
    let passed = evaluation.passed && prefill_checks.all_valid() && final_checks.all_valid();
    gate_evidence::verify_file_unchanged(
        input_path,
        &input_attestation,
        "boundary-tail v1 CPU teacher receipt",
    )?;
    let tail_total_ns = tail_transaction_elapsed_ns.iter().copied().sum::<u128>();
    let rerank_total_ns = direct_f32_rerank_elapsed_ns.iter().copied().sum::<u128>();
    Ok(RunResult {
        receipt: json!({
            "format": args.mode.receipt_format(),
            "mode": args.mode.label(),
            "identity": identity,
            "input_receipt": gate_evidence::attestation_json(&input_attestation),
            "prompt": PROMPT,
            "prompt_token_ids": prompt_tokens,
            "official_layer_schedule_valid": true,
            "comparisons": STEPS,
            "prefill_token": prefill_token,
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
                "tail_normalized_hidden_f32_winner_matches_frozen_cpu_token": evaluation.body_token_mismatches.is_empty(),
                "top4_is_unique_in_range_and_contains_f32_winner": evaluation.top4_mismatches.is_empty(),
                "direct_tied_f32_rerank_matches_f32_winner": evaluation.rerank_mismatches.is_empty(),
                "composite_token_matches_frozen_cpu": evaluation.end_to_end_mismatches.is_empty(),
                "hidden_tensor_exactness_claimed": false,
            },
            "prefill_generation_path_receipt": prefill_generation,
            "final_generation_path_receipt": final_generation,
            "generation_path_contract": {
                "schema": "apxinf-qwen35-mlp-stack3-boundary-tail-head-generation-path-v1",
                "binds_initial_stack_five_boundaries_and_fused_tail": true,
                "cpu_f32_prefill": true,
                "tail_prefill_calls": 0,
                "tail_decode_calls": 0,
                "tail_teacher_calls": STEPS,
                "f32_rerank_input": "tail-normalized-hidden-direct",
                "hidden_tensor_exactness_claimed": false,
            },
            "aggregate_buffer_ledger": aggregate_ledger_json(&aggregate),
            "path_checks": {
                "prefill": prefill_checks.receipt_json(),
                "final": final_checks.receipt_json(),
            },
            "timing": {
                "setup": setup,
                "prefill_ms": prefill_ms,
                "decode_ms": decode_ms,
                "decode_mean_ms": decode_ms / STEPS as f64,
                "tail_transaction_elapsed_ns": tail_transaction_elapsed_ns,
                "direct_tied_f32_rerank_elapsed_ns": direct_f32_rerank_elapsed_ns,
                "tail_transaction_total_ms": tail_total_ns as f64 / 1_000_000.0,
                "direct_tied_f32_rerank_total_ms": rerank_total_ns as f64 / 1_000_000.0,
                "tail_transaction_semantics": "fused layer23 MLP + final RMS + W8 top4 accelerator candidate transaction",
                "classification": "candidate-only single pass under an uncontrolled host; never formal or promotion evidence",
            },
            "passed": passed,
        }),
        passed,
    })
}

fn run_free(
    args: &Args,
    model: &mut GeneralQwen35,
    prompt_tokens: &[u32],
    identity: Value,
    setup: Value,
) -> Result<RunResult, Box<dyn Error>> {
    let input = if args.mode == Mode::BoundaryTailV1Free {
        let input_path = args
            .input_receipt
            .as_ref()
            .ok_or("candidate free mode requires --input-receipt")?;
        let (receipt, attestation) =
            gate_evidence::read_attested_json(input_path, "boundary-tail v1 CPU free receipt")?;
        let expected = validate_cpu_free_receipt(&receipt, &identity, prompt_tokens)?;
        Some((input_path, attestation, expected))
    } else {
        None
    };
    let started = Instant::now();
    let (generated, profile) =
        model.generate_streaming(LlmInput::text(prompt_tokens), STEPS, |_| {}, None)?;
    let elapsed_ms = started.elapsed().as_secs_f64() * 1_000.0;
    if generated.len() != STEPS {
        return Err(format!(
            "shared free generation returned {} tokens, expected {STEPS}",
            generated.len()
        )
        .into());
    }
    if args.mode == Mode::CpuFree {
        return Ok(RunResult {
            receipt: json!({
                "format": args.mode.receipt_format(),
                "mode": args.mode.label(),
                "identity": identity,
                "prompt": PROMPT,
                "prompt_token_ids": prompt_tokens,
                "official_layer_schedule_valid": true,
                "max_new_tokens": STEPS,
                "eos_stopping": false,
                "generated_token_ids": generated,
                "generation_path_contract": null,
                "profile": generation_profile_json(
                    &profile,
                    elapsed_ms,
                    setup,
                    free_timing_classification(args.mode),
                ),
                "passed": true,
            }),
            passed: true,
        });
    }

    let (input_path, input_attestation, expected) = input.expect("candidate input checked above");
    let mismatches = evaluate_free_trajectory(&expected, &generated)?;
    let stats = model
        .metal_w8_mlp_stack3_boundary_tail_head_v1_stats()
        .ok_or("boundary-tail v1 constructor omitted final stats")?;
    let aggregate = model
        .metal_w8_mlp_stack3_boundary_tail_head_v1_aggregate_ledger()
        .ok_or("boundary-tail v1 constructor omitted aggregate ledger")?;
    let generation = model
        .generation_path_receipt()
        .ok_or("boundary-tail v1 constructor omitted final generation receipt")?;
    let body_tail_calls = STEPS - 1;
    let checks = boundary_tail_path_checks(
        &stats,
        &aggregate,
        &generation,
        body_tail_calls,
        TailPhaseCounts::free(body_tail_calls),
    );
    let passed = mismatches.is_empty() && checks.all_valid();
    gate_evidence::verify_file_unchanged(
        input_path,
        &input_attestation,
        "boundary-tail v1 CPU free receipt",
    )?;
    Ok(RunResult {
        receipt: json!({
            "format": args.mode.receipt_format(),
            "mode": args.mode.label(),
            "identity": identity,
            "input_receipt": gate_evidence::attestation_json(&input_attestation),
            "prompt": PROMPT,
            "prompt_token_ids": prompt_tokens,
            "official_layer_schedule_valid": true,
            "max_new_tokens": STEPS,
            "eos_stopping": false,
            "cpu_expected_token_ids": expected,
            "generated_token_ids": generated,
            "mismatches": mismatches,
            "exact_128_token_trajectory": mismatches.is_empty(),
            "final_generation_path_receipt": generation,
            "generation_path_contract": {
                "schema": "apxinf-qwen35-mlp-stack3-boundary-tail-head-generation-path-v1",
                "shared_generate_streaming": true,
                "binds_initial_stack_five_boundaries_and_fused_tail": true,
                "body_tail_calls": body_tail_calls,
                "cpu_f32_prefill_head_calls": 1,
                "tail_prefill_calls": 0,
                "tail_decode_calls": body_tail_calls,
                "tail_teacher_calls": 0,
                "f32_rerank_input": "tail-normalized-hidden-direct",
            },
            "aggregate_buffer_ledger": aggregate_ledger_json(&aggregate),
            "path_checks": checks.receipt_json(),
            "profile": generation_profile_json(
                &profile,
                elapsed_ms,
                setup,
                free_timing_classification(args.mode),
            ),
            "passed": passed,
        }),
        passed,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT_TEMP: AtomicU64 = AtomicU64::new(0);

    fn temp_output(label: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "apxinf-boundary-tail-head-v1-{label}-{}-{}",
            std::process::id(),
            NEXT_TEMP.fetch_add(1, Ordering::Relaxed)
        ))
    }

    fn official_aggregate_fixture() -> Qwen35MetalW8MlpStack3BoundaryTailHeadV1AggregateLedger {
        let initial_stack = Qwen35MetalW8LinearLayerStack3BufferLedger {
            layer_indices: [0, 1, 2],
            ledger: apxinf_metal::LinearLayerStack3BufferLedger {
                allocated_buffers: 76,
                shared_buffers: 68,
                private_buffers: 8,
                packed_weight_bytes: 64_585_728,
                packed_scale_bytes: 4_429_824,
                f32_parameter_bytes: 321_408,
                active_state_bytes: 3_440_640,
                scratch_state_bytes: 3_440_640,
                activation_bytes: 133_248,
                total_persistent_bytes: 76_351_488,
                host_input_bytes_per_decode: 4_096,
                host_output_bytes_per_decode: 4_096,
                state_host_transfer_bytes_per_decode: 0,
                command_buffers_per_decode: 1,
                compute_encoders_per_decode: 3,
                commits_per_decode: 1,
                waits_per_decode: 1,
                intermediate_host_finite_checks_per_decode: 0,
                final_output_finite_checks_per_decode: 1,
            },
        };
        let boundary = apxinf_metal::MlpStack3BoundaryBufferLedgerV1 {
            scope: "resident-mtlbuffer-only",
            exclusions: "CPU packed weights, host allocations, Metal pipelines/libraries/queues, command objects, driver allocations, attention/KV, model loader, and language-model head",
            abi_version: 1,
            stack_depth: 3,
            allocated_buffers: 81,
            shared_buffers: 73,
            private_buffers: 8,
            packed_weight_bytes: 75_595_776,
            packed_scale_bytes: 5_117_952,
            f32_parameter_bytes: 325_504,
            active_state_bytes: 3_440_640,
            scratch_state_bytes: 3_440_640,
            activation_bytes: 133_248,
            total_persistent_bytes: 88_053_760,
            host_input_bytes_per_decode: 4_096,
            host_output_bytes_per_decode: 4_096,
            state_host_transfer_bytes_per_decode: 0,
            command_buffers_per_decode: 1,
            compute_encoders_per_decode: 4,
            kernel_dispatches_per_decode: 44,
            commits_per_decode: 1,
            waits_per_decode: 1,
            intermediate_host_finite_checks_per_decode: 0,
            final_output_finite_checks_per_decode: 1,
        };
        Qwen35MetalW8MlpStack3BoundaryTailHeadV1AggregateLedger {
            scope: "resident-mtlbuffer-only",
            exclusions: "CPU F32 weights, host F32 tied embedding and exact four-candidate rerank, host allocations, Metal pipelines/libraries/queues, command objects, driver allocations, full attention/KV, model loader, and prefill CPU head",
            includes_lm_head: true,
            initial_stack,
            boundaries: BOUNDARY_REGIONS
                .map(|(boundary_mlp_layer_index, stack_layer_indices)| {
                    Qwen35MetalW8MlpStack3BoundaryRegionBufferLedgerV1 {
                        boundary_mlp_layer_index,
                        stack_layer_indices,
                        ledger: boundary,
                    }
                })
                .to_vec(),
            tail_layer_index: 23,
            tail: apxinf_metal::TailMlpHeadBufferLedgerV1::from_dimensions(
                1_024, 3_584, 248_320,
            )
            .unwrap(),
            total_persistent_mtlbuffer_bytes: 799_543_312,
            allocated_buffers: 494,
            shared_buffers: 443,
            private_buffers: 51,
            host_to_device_bytes_per_decode: 28_672,
            device_to_host_bytes_per_decode: 28_688,
            state_host_transfer_bytes_per_decode: 0,
            command_buffers_per_decode: 7,
            compute_encoders_per_decode: 24,
            kernel_dispatches_per_decode: 267,
            commits_per_decode: 7,
            waits_per_decode: 7,
        }
    }

    fn generation_receipt_fixture(calls: usize, phase: TailPhaseCounts) -> Value {
        let transfer = calls * 4_096;
        json!({
            "format": "apxinf-qwen35-mlp-stack3-boundary-tail-head-generation-path-v1",
            "mechanism": "metal-w8-mlp-stack3-boundary-tail-head-v1",
            "cpu_full_attention_and_kv": true,
            "cpu_prefill_all_24_layers": true,
            "metal_w8_initial_complete_linear_layer_stack3": true,
            "metal_w8_mlp_stack3_boundaries": true,
            "metal_w8_tail_layer23_mlp_final_rms_top4": true,
            "standalone_layer23_mlp": false,
            "standalone_metal_lm_head": false,
            "f32_tied_four_candidate_rerank": true,
            "initial_stack": {
                "layer_indices": [0, 1, 2],
                "mechanism": "metal-w8-linear-layer-stack3-v1",
                "prefill_seed_calls": [1, 1, 1],
                "decode_calls": calls,
                "successful_decodes": calls,
                "failed_decodes": 0,
                "command_buffers": calls,
                "compute_encoders": calls * 3,
                "commits": calls,
                "waits": calls,
                "host_to_device_bytes": transfer,
                "device_to_host_bytes": transfer,
                "state_commits": calls * 3,
                "last_state_commit_mask": if calls == 0 { 0 } else { 0b111 },
                "committed_stack_version": calls,
                "terminal_error": false,
            },
            "boundaries": BOUNDARY_REGIONS.map(|(boundary_mlp_layer_index, stack_layer_indices)| json!({
                "boundary_mlp_layer_index": boundary_mlp_layer_index,
                "stack_layer_indices": stack_layer_indices,
                "mechanism": "metal-w8-mlp-stack3-boundary-v1",
                "prefill_seed_calls": [1, 1, 1],
                "decode_calls": calls,
                "successful_decodes": calls,
                "failed_decodes": 0,
                "command_buffers": calls,
                "compute_encoders": calls * 4,
                "commits": calls,
                "waits": calls,
                "host_to_device_bytes": transfer,
                "device_to_host_bytes": transfer,
                "state_commits": calls * 3,
                "last_state_commit_mask": if calls == 0 { 0 } else { 0b111 },
                "committed_stack_version": calls,
                "terminal_error": false,
            })),
            "prefill_body_calls": 1,
            "prefill_head": {
                "mechanism": "cpu-f32-tied",
                "calls": 1,
                "tail_transactions": 0,
            },
            "decode_head": {
                "mechanism": "metal-w8-tail-v1",
                "layer_index": 23,
                "calls": phase.decode,
                "teacher_calls": phase.teacher,
                "tail_transactions": calls,
                "successful_transactions": calls,
                "failed_transactions": 0,
                "command_buffers": calls,
                "compute_encoders": calls,
                "kernel_dispatches": calls * 8,
                "commits": calls,
                "waits": calls,
                "host_to_device_bytes": transfer,
                "device_to_host_bytes": calls * 4_112,
                "output_commits": calls * 2,
                "last_output_commit_mask": if calls == 0 { 0 } else { 0b11 },
                "terminal_error": false,
            },
            "aggregate": {
                "scope": "resident-mtlbuffer-only",
                "includes_lm_head": true,
                "persistent_mtlbuffer_bytes": 799_543_312usize,
                "allocated_buffers": 494,
                "shared_buffers": 443,
                "private_buffers": 51,
                "host_to_device_bytes_per_decode": 28_672,
                "device_to_host_bytes_per_decode": 28_688,
                "state_host_transfer_bytes_per_decode": 0,
                "command_buffers_per_decode": 7,
                "compute_encoders_per_decode": 24,
                "kernel_dispatches_per_decode": 267,
                "commits_per_decode": 7,
                "waits_per_decode": 7,
            },
            "terminal_error": false,
        })
    }

    #[test]
    fn four_modes_have_distinct_versioned_receipt_formats() {
        let modes = [
            Mode::parse("cpu-teacher").unwrap(),
            Mode::parse("boundary-tail-v1-teacher").unwrap(),
            Mode::parse("cpu-free").unwrap(),
            Mode::parse("boundary-tail-v1-free").unwrap(),
        ];
        assert_eq!(
            modes.map(Mode::label),
            [
                "cpu_teacher",
                "metal_w8_boundary_tail_v1_teacher_forced",
                "cpu_free_run",
                "metal_w8_boundary_tail_v1_free_run",
            ]
        );
        let formats = modes.map(Mode::receipt_format);
        assert_eq!(
            formats
                .into_iter()
                .collect::<std::collections::BTreeSet<_>>()
                .len(),
            4
        );
        assert!(formats
            .iter()
            .all(|format| format.contains("boundary-tail-head-v1")));
        assert!(!modes[0].requires_input_receipt());
        assert!(modes[1].requires_input_receipt());
        assert!(!modes[2].requires_input_receipt());
        assert!(modes[3].requires_input_receipt());
    }

    #[test]
    fn parser_requires_absolute_paths_candidate_input_and_new_output() {
        let args = parse_args_from([
            "gate",
            "--model-dir",
            "/model",
            "--source-lock",
            "/source-lock.json",
            "--mode",
            "boundary-tail-v1-free",
            "--input-receipt",
            "/cpu-free.json",
            "--output",
            "/new-receipt.json",
        ])
        .unwrap();
        assert_eq!(args.mode, Mode::BoundaryTailV1Free);
        assert_eq!(
            args.input_receipt.unwrap(),
            std::path::PathBuf::from("/cpu-free.json")
        );
        assert!(parse_args_from([
            "gate",
            "--model-dir",
            "/model",
            "--source-lock",
            "/lock",
            "--mode",
            "boundary-tail-v1-teacher",
            "--output",
            "/new.json",
        ])
        .is_err());
        assert!(parse_args_from([
            "gate",
            "--model-dir",
            "/model",
            "--source-lock",
            "/lock",
            "--mode",
            "cpu-free",
            "--input-receipt",
            "/unexpected.json",
            "--output",
            "/new.json",
        ])
        .is_err());
        assert!(parse_args_from([
            "gate",
            "--model-dir",
            "relative",
            "--source-lock",
            "/lock",
            "--mode",
            "cpu-free",
            "--output",
            "/new.json",
        ])
        .is_err());
    }

    #[test]
    fn cpu_teacher_receipt_is_passed_identity_bound_and_chains_all_128_steps() {
        let expected = (100..228).collect::<Vec<u32>>();
        let mut inputs = vec![7];
        inputs.extend_from_slice(&expected[..127]);
        let identity = json!({"binary": "same-release"});
        let mut receipt = json!({
            "format": CPU_TEACHER_FORMAT,
            "mode": Mode::CpuTeacher.label(),
            "passed": true,
            "identity": identity,
            "prompt_token_ids": [1, 2],
            "prefill_token": 7,
            "comparisons": 128,
            "teacher_input_ids": inputs,
            "cpu_expected_output_ids": expected,
        });

        let oracle = validate_cpu_teacher_receipt(&receipt, &identity, &[1, 2], 7).unwrap();
        assert_eq!(oracle.teacher_inputs.len(), 128);
        assert_eq!(oracle.expected_outputs.len(), 128);
        receipt["passed"] = json!(false);
        assert!(validate_cpu_teacher_receipt(&receipt, &identity, &[1, 2], 7).is_err());
    }

    #[test]
    fn teacher_candidate_requires_valid_top4_direct_f32_rerank_and_exact_tokens() {
        let expected = (100..228).collect::<Vec<u32>>();
        let oracle = TeacherOracle {
            teacher_inputs: (0..128).collect(),
            expected_outputs: expected.clone(),
        };
        let top4 = expected
            .iter()
            .map(|token| [*token, token + 1, token + 2, token + 3])
            .collect::<Vec<_>>();
        let accepted =
            evaluate_teacher_candidate(&oracle, &expected, &top4, &expected, 500).unwrap();
        assert!(accepted.passed);
        assert!(accepted.top4_mismatches.is_empty());

        let mut duplicate = top4.clone();
        duplicate[3][1] = duplicate[3][0];
        assert!(
            !evaluate_teacher_candidate(&oracle, &expected, &duplicate, &expected, 500)
                .unwrap()
                .passed
        );
        let mut out_of_range = top4.clone();
        out_of_range[4][3] = 500;
        assert!(
            !evaluate_teacher_candidate(&oracle, &expected, &out_of_range, &expected, 500)
                .unwrap()
                .passed
        );
        let mut omitted = top4;
        omitted[5] = [1, 2, 3, 4];
        let rejected =
            evaluate_teacher_candidate(&oracle, &expected, &omitted, &expected, 500).unwrap();
        assert!(!rejected.passed);
        let mismatch = &rejected.top4_mismatches[0];
        assert_eq!(mismatch["step"], 5);
        assert_eq!(mismatch["input"], 5);
        assert_eq!(
            mismatch["expected"]["contains_tail_normalized_f32_winner"],
            105
        );
        assert_eq!(mismatch["actual"], json!([1, 2, 3, 4]));

        let mut wrong_rerank = expected.clone();
        wrong_rerank[9] += 1;
        let rejected =
            evaluate_teacher_candidate(&oracle, &expected, &duplicate, &wrong_rerank, 500).unwrap();
        let mismatch = &rejected.rerank_mismatches[0];
        assert_eq!(mismatch["step"], 9);
        assert_eq!(mismatch["input"], 9);
        assert_eq!(mismatch["expected"], 109);
        assert_eq!(mismatch["actual"], 110);
    }

    #[test]
    fn cpu_free_receipt_is_a_passed_identity_bound_128_token_oracle() {
        let identity = json!({"binary": "same-release"});
        let tokens = (100..228).collect::<Vec<u32>>();
        let mut receipt = json!({
            "format": CPU_FREE_FORMAT,
            "mode": Mode::CpuFree.label(),
            "passed": true,
            "identity": identity,
            "prompt_token_ids": [1, 2],
            "max_new_tokens": 128,
            "eos_stopping": false,
            "generated_token_ids": tokens,
        });
        assert_eq!(
            validate_cpu_free_receipt(&receipt, &identity, &[1, 2]).unwrap(),
            tokens
        );
        receipt["generated_token_ids"].as_array_mut().unwrap().pop();
        assert!(validate_cpu_free_receipt(&receipt, &identity, &[1, 2]).is_err());
    }

    #[test]
    fn free_candidate_compares_the_complete_128_token_trajectory() {
        let expected = (100..228).collect::<Vec<u32>>();
        assert!(evaluate_free_trajectory(&expected, &expected)
            .unwrap()
            .is_empty());

        let mut actual = expected.clone();
        actual[127] += 1;
        let mismatches = evaluate_free_trajectory(&expected, &actual).unwrap();
        assert_eq!(mismatches.len(), 1);
        assert_eq!(mismatches[0]["step"], 127);
        assert_eq!(mismatches[0]["expected"], 227);
        assert_eq!(mismatches[0]["actual"], 228);
        assert!(evaluate_free_trajectory(&expected[..127], &actual).is_err());
    }

    #[test]
    fn receipt_publication_uses_create_new_and_never_replaces() {
        let output = temp_output("no-replace.json");
        let first = json!({"formal": "first"});
        let second = json!({"formal": "second"});
        publish_receipt_create_new(&output, &first).unwrap();
        let first_bytes = std::fs::read(&output).unwrap();
        assert!(publish_receipt_create_new(&output, &second).is_err());
        assert_eq!(std::fs::read(&output).unwrap(), first_bytes);
        std::fs::remove_file(output).unwrap();
    }

    #[test]
    fn official_boundary_tail_schedule_is_exact() {
        let schedule = (0..24)
            .map(|index| {
                if FULL_ATTENTION_LAYER_INDICES.contains(&index) {
                    apxinf_model::Qwen35LayerType::FullAttention
                } else {
                    apxinf_model::Qwen35LayerType::LinearAttention
                }
            })
            .collect::<Vec<_>>();
        validate_official_schedule(&schedule).unwrap();
        let mut wrong = schedule.clone();
        wrong[19] = apxinf_model::Qwen35LayerType::LinearAttention;
        assert!(validate_official_schedule(&wrong).is_err());
    }

    #[test]
    fn component_ledgers_and_recomputed_aggregate_are_strictly_frozen() {
        let aggregate = official_aggregate_fixture();
        assert!(official_aggregate_ledger_is_exact(&aggregate));

        let mut wrong = aggregate.clone();
        wrong.boundaries[2].ledger.total_persistent_bytes -= 1;
        assert!(!official_aggregate_ledger_is_exact(&wrong));
        let mut wrong = aggregate.clone();
        wrong.tail.host_output_bytes_per_decode -= 1;
        assert!(!official_aggregate_ledger_is_exact(&wrong));
        let mut wrong = aggregate.clone();
        wrong.total_persistent_mtlbuffer_bytes -= 1;
        assert!(!official_aggregate_ledger_is_exact(&wrong));
        let mut wrong = aggregate;
        wrong.kernel_dispatches_per_decode -= 1;
        assert!(!official_aggregate_ledger_is_exact(&wrong));
    }

    #[test]
    fn generation_receipt_locks_six_regions_tail_phases_and_commit_masks() {
        let phase = TailPhaseCounts::teacher(7);
        let receipt = generation_receipt_fixture(7, phase);
        assert!(boundary_tail_generation_receipt_is_exact(
            &receipt, 7, phase
        ));

        let mut wrong = receipt.clone();
        wrong["initial_stack"]["last_state_commit_mask"] = json!(0b011);
        assert!(!boundary_tail_generation_receipt_is_exact(&wrong, 7, phase));
        let mut wrong = receipt.clone();
        wrong["boundaries"][3]["last_state_commit_mask"] = json!(0b110);
        assert!(!boundary_tail_generation_receipt_is_exact(&wrong, 7, phase));
        let mut wrong = receipt.clone();
        wrong["decode_head"]["teacher_calls"] = json!(6);
        assert!(!boundary_tail_generation_receipt_is_exact(&wrong, 7, phase));

        let mut wrong = receipt.clone();
        wrong["terminal_error"] = json!(true);
        assert!(!boundary_tail_generation_receipt_is_exact(&wrong, 7, phase));
        let mut wrong = receipt;
        wrong["decode_head"]["terminal_error"] = json!(true);
        assert!(!boundary_tail_generation_receipt_is_exact(&wrong, 7, phase));

        let free_phase = TailPhaseCounts::free(7);
        let free_receipt = generation_receipt_fixture(7, free_phase);
        assert!(boundary_tail_generation_receipt_is_exact(
            &free_receipt,
            7,
            free_phase
        ));
        let mut wrong = free_receipt;
        wrong["prefill_head"]["tail_transactions"] = json!(1);
        assert!(!boundary_tail_generation_receipt_is_exact(
            &wrong, 7, free_phase
        ));
    }

    #[test]
    fn receipt_paths_inside_the_canonical_model_directory_are_rejected_early() {
        let root = temp_output("receipt-paths");
        let model = root.join("model");
        let evidence = root.join("evidence");
        std::fs::create_dir_all(&model).unwrap();
        std::fs::create_dir_all(&evidence).unwrap();
        let model_input = model.join("cpu.json");
        std::fs::write(&model_input, b"{}\n").unwrap();

        assert!(validate_receipt_paths_outside_model_dir(
            &model,
            Some(&model_input),
            &evidence.join("candidate.json"),
        )
        .is_err());
        assert!(validate_receipt_paths_outside_model_dir(
            &model,
            None,
            &model.join("candidate.json"),
        )
        .is_err());
        validate_receipt_paths_outside_model_dir(&model, None, &evidence.join("candidate.json"))
            .unwrap();
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn fake_main_publishes_failed_quality_receipt_after_end_custody() {
        let output = temp_output("fake-main-failed-quality.json");
        let result = RunResult {
            receipt: json!({"passed": false, "mismatches": [{"step": 3}]}),
            passed: false,
        };
        let finalized =
            finalize_run_with_end_custody(result, &output, || Ok(json!({"verified_at_end": true})))
                .unwrap();
        assert!(!finalized.passed);
        let published: Value = serde_json::from_slice(&std::fs::read(&output).unwrap()).unwrap();
        assert_eq!(published["passed"], false);
        assert_eq!(
            published["custody_end_verification"]["verified_at_end"],
            true
        );
        std::fs::remove_file(output).unwrap();
    }

    #[test]
    fn finalize_rejects_same_size_input_receipt_drift_after_end_custody() {
        let input = temp_output("final-input-drift.json");
        let output = temp_output("final-input-drift-output.json");
        std::fs::write(&input, b"{\"passed\":true}\n").unwrap();
        let input_attestation = gate_evidence::attest_file(
            &input,
            "boundary-tail v1 CPU test receipt",
            None,
        )
        .unwrap();
        let result = RunResult {
            receipt: json!({"passed": true}),
            passed: true,
            input_receipt_guard: Some(InputReceiptGuard {
                path: input.clone(),
                attestation: input_attestation,
                label: "boundary-tail v1 CPU test receipt",
            }),
        };

        let finalized = finalize_run_with_end_custody(result, &output, || {
            std::fs::write(&input, b"{\"passed\":null}\n").unwrap();
            Ok(json!({"verified_at_end": true}))
        });
        assert!(finalized.is_err());
        assert!(!output.exists());

        std::fs::remove_file(input).unwrap();
    }
}
