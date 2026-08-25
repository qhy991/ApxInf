//! Same-binary four-mode gate for the explicit Qwen3.5 MLP→Stack3 boundary
//! body + fused tail-head v1 diagnostic. This example is not reachable from
//! CLI, AutoModel, registry, or any default constructor.

#[path = "support/qwen35_boundary_tail_head_v1_gate_evidence.rs"]
mod gate_evidence;

use std::ffi::{CString, OsString};
use std::fs::{self, File};
use std::io::Write;
use std::os::fd::{AsRawFd, FromRawFd};
use std::os::unix::ffi::OsStrExt;
use std::os::unix::fs::MetadataExt;
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

const CPU_TEACHER_FORMAT: &str = "apxinf-qwen35-metal-w8-boundary-tail-head-v1-cpu-teacher-v2";
const CANDIDATE_TEACHER_FORMAT: &str =
    "apxinf-qwen35-metal-w8-boundary-tail-head-v1-teacher-gate-v2";
const CPU_FREE_FORMAT: &str = "apxinf-qwen35-metal-w8-boundary-tail-head-v1-cpu-free-run-v2";
const CANDIDATE_FREE_FORMAT: &str = "apxinf-qwen35-metal-w8-boundary-tail-head-v1-free-run-gate-v2";
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
    input_receipt_guard: Option<InputReceiptGuard>,
}

struct InputReceiptGuard {
    path: PathBuf,
    attestation: gate_evidence::FileAttestation,
    label: &'static str,
    canonical_model_dir: PathBuf,
}

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
enum Mode {
    CpuTeacher,
    BoundaryTailV1Teacher,
    CpuFree,
    BoundaryTailV1Free,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BodyMlpEpilogue {
    Legacy,
    DownResidualFused,
}

impl BodyMlpEpilogue {
    fn parse(value: &str) -> Result<Self, String> {
        match value {
            "legacy" => Ok(Self::Legacy),
            "down-residual-fused" => Ok(Self::DownResidualFused),
            other => Err(format!("invalid --body-mlp-epilogue {other:?}")),
        }
    }

    const fn label(self) -> &'static str {
        match self {
            Self::Legacy => "legacy",
            Self::DownResidualFused => "down-residual-fused",
        }
    }

    const fn runtime_label(self) -> &'static str {
        match self {
            Self::Legacy => "legacy-separate",
            Self::DownResidualFused => "down-residual-fused",
        }
    }

    const fn mlp_down_function_name(self) -> &'static str {
        match self {
            Self::Legacy => "w8_mlp_down",
            Self::DownResidualFused => "w8_mlp_down_residual",
        }
    }

    const fn metal(self) -> apxinf_metal::MlpEpilogueV1 {
        match self {
            Self::Legacy => apxinf_metal::MlpEpilogueV1::LegacySeparate,
            Self::DownResidualFused => apxinf_metal::MlpEpilogueV1::DownResidualFused,
        }
    }

    const fn from_metal(value: apxinf_metal::MlpEpilogueV1) -> Self {
        match value {
            apxinf_metal::MlpEpilogueV1::LegacySeparate => Self::Legacy,
            apxinf_metal::MlpEpilogueV1::DownResidualFused => Self::DownResidualFused,
        }
    }

    const fn initial_dispatches(self) -> usize {
        match self {
            Self::Legacy => 39,
            Self::DownResidualFused => 36,
        }
    }

    const fn boundary_dispatches(self) -> usize {
        match self {
            Self::Legacy => 44,
            Self::DownResidualFused => 40,
        }
    }

    const fn aggregate_dispatches(self) -> usize {
        match self {
            Self::Legacy => 267,
            Self::DownResidualFused => 244,
        }
    }

    const fn initial_barriers(self) -> usize {
        match self {
            Self::Legacy => 36,
            Self::DownResidualFused => 33,
        }
    }

    const fn boundary_barriers(self) -> usize {
        match self {
            Self::Legacy => 40,
            Self::DownResidualFused => 36,
        }
    }

    const fn aggregate_barriers(self) -> usize {
        match self {
            Self::Legacy => 243,
            Self::DownResidualFused => 220,
        }
    }
}

fn mlp_epilogue_runtime_json_is_exact(
    receipt: Option<&Value>,
    body_mlp_epilogue: BodyMlpEpilogue,
    kernel_dispatches_per_decode: usize,
    buffer_barriers_per_decode: usize,
) -> bool {
    let Some(receipt) = receipt else {
        return false;
    };
    receipt.get("requested_profile").and_then(Value::as_str)
        == Some(body_mlp_epilogue.runtime_label())
        && receipt.get("observed_profile").and_then(Value::as_str)
            == Some(body_mlp_epilogue.runtime_label())
        && receipt
            .get("mlp_down_function_name")
            .and_then(Value::as_str)
            == Some(body_mlp_epilogue.mlp_down_function_name())
        && receipt
            .get("kernel_dispatches_per_decode")
            .and_then(Value::as_u64)
            == Some(kernel_dispatches_per_decode as u64)
        && receipt
            .get("buffer_barriers_per_decode")
            .and_then(Value::as_u64)
            == Some(buffer_barriers_per_decode as u64)
}

fn mlp_epilogue_runtime_is_exact(
    receipt: apxinf_metal::MlpEpilogueRuntimeReceiptV1,
    body_mlp_epilogue: BodyMlpEpilogue,
    kernel_dispatches_per_decode: usize,
    buffer_barriers_per_decode: usize,
) -> bool {
    receipt.requested_profile == body_mlp_epilogue.metal()
        && receipt.observed_profile == body_mlp_epilogue.metal()
        && receipt.mlp_down_function_name == body_mlp_epilogue.mlp_down_function_name()
        && receipt.kernel_dispatches_per_decode == kernel_dispatches_per_decode
        && receipt.buffer_barriers_per_decode == buffer_barriers_per_decode
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
    body_mlp_epilogue: BodyMlpEpilogue,
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
        && ledger.kernel_dispatches_per_decode == body_mlp_epilogue.initial_dispatches()
        && ledger.buffer_barriers_per_decode == body_mlp_epilogue.initial_barriers()
        && ledger.commits_per_decode == 1
        && ledger.waits_per_decode == 1
        && ledger.intermediate_host_finite_checks_per_decode == 0
        && ledger.final_output_finite_checks_per_decode == 1
}

fn official_boundary_ledger_is_exact(
    ledger: apxinf_metal::MlpStack3BoundaryBufferLedgerV1,
    body_mlp_epilogue: BodyMlpEpilogue,
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
        && ledger.kernel_dispatches_per_decode == body_mlp_epilogue.boundary_dispatches()
        && ledger.buffer_barriers_per_decode == body_mlp_epilogue.boundary_barriers()
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
    body_mlp_epilogue: BodyMlpEpilogue,
) -> bool {
    if aggregate.scope != "resident-mtlbuffer-only"
        || aggregate.exclusions
            != "CPU F32 weights, host F32 tied embedding and exact four-candidate rerank, host allocations, Metal pipelines/libraries/queues, command objects, driver allocations, full attention/KV, model loader, and prefill CPU head"
        || !aggregate.includes_lm_head
        || aggregate.initial_stack.layer_indices != [0, 1, 2]
        || aggregate.body_mlp_epilogue != body_mlp_epilogue.metal()
        || !official_initial_stack_ledger_is_exact(
            aggregate.initial_stack.ledger,
            body_mlp_epilogue,
        )
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
                    && official_boundary_ledger_is_exact(entry.ledger, body_mlp_epilogue)
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
        .try_fold(
            aggregate.initial_stack.ledger.kernel_dispatches_per_decode,
            |total, entry| total.checked_add(entry.ledger.kernel_dispatches_per_decode),
        )
        .and_then(|total| total.checked_add(tail.kernel_dispatches_per_decode));
    let recomputed_barriers = boundaries
        .iter()
        .try_fold(
            aggregate.initial_stack.ledger.buffer_barriers_per_decode,
            |total, entry| total.checked_add(entry.ledger.buffer_barriers_per_decode),
        )
        .and_then(|total| total.checked_add(tail.buffer_barriers_per_decode));
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
        && aggregate.buffer_barriers_per_decode == recomputed_barriers.unwrap_or(0)
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
        && aggregate.kernel_dispatches_per_decode == body_mlp_epilogue.aggregate_dispatches()
        && aggregate.buffer_barriers_per_decode == body_mlp_epilogue.aggregate_barriers()
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
        "kernel_dispatches_per_decode": ledger.kernel_dispatches_per_decode,
        "buffer_barriers_per_decode": ledger.buffer_barriers_per_decode,
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
        "buffer_barriers_per_decode": ledger.buffer_barriers_per_decode,
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
        "body_mlp_epilogue": match ledger.body_mlp_epilogue {
            apxinf_metal::MlpEpilogueV1::LegacySeparate => "legacy-separate",
            apxinf_metal::MlpEpilogueV1::DownResidualFused => "down-residual-fused",
        },
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
        "buffer_barriers_per_decode": ledger.buffer_barriers_per_decode,
        "commits_per_decode": ledger.commits_per_decode,
        "waits_per_decode": ledger.waits_per_decode,
        "component_sum_recomputed_and_exact": official_aggregate_ledger_is_exact(
            ledger,
            BodyMlpEpilogue::from_metal(ledger.body_mlp_epilogue),
        ),
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
    body_mlp_epilogue: BodyMlpEpilogue,
) -> bool {
    if phase.prefill != 0
        || phase.decode.checked_add(phase.teacher) != Some(body_tail_calls)
        || receipt.get("format").and_then(Value::as_str)
            != Some("apxinf-qwen35-mlp-stack3-boundary-tail-head-generation-path-v1")
        || receipt.get("mechanism").and_then(Value::as_str)
            != Some("metal-w8-mlp-stack3-boundary-tail-head-v1")
        || receipt.get("body_mlp_epilogue").and_then(Value::as_str)
            != Some(body_mlp_epilogue.runtime_label())
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
        && mlp_epilogue_runtime_json_is_exact(
            initial.get("mlp_epilogue_runtime"),
            body_mlp_epilogue,
            body_mlp_epilogue.initial_dispatches(),
            body_mlp_epilogue.initial_barriers(),
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
                        == Some("metal-w8-mlp-stack3-boundary-v1")
                    && mlp_epilogue_runtime_json_is_exact(
                        entry.get("mlp_epilogue_runtime"),
                        body_mlp_epilogue,
                        body_mlp_epilogue.boundary_dispatches(),
                        body_mlp_epilogue.boundary_barriers(),
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
        && aggregate.get("body_mlp_epilogue").and_then(Value::as_str)
            == Some(body_mlp_epilogue.runtime_label())
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
            == Some(body_mlp_epilogue.aggregate_dispatches() as u64)
        && aggregate
            .get("buffer_barriers_per_decode")
            .and_then(Value::as_u64)
            == Some(body_mlp_epilogue.aggregate_barriers() as u64)
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
    body_mlp_epilogue: BodyMlpEpilogue,
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
            && stats.body_mlp_epilogue == body_mlp_epilogue.metal()
            && stats.initial_stack.mechanism == "metal-w8-linear-layer-stack3-v1"
            && mlp_epilogue_runtime_is_exact(
                stats.initial_stack.mlp_epilogue_runtime,
                body_mlp_epilogue,
                body_mlp_epilogue.initial_dispatches(),
                body_mlp_epilogue.initial_barriers(),
            )
            && stats
                .initial_stack
                .quantization
                .iter()
                .copied()
                .all(quantization_profile_is_exact)
            && stats.boundaries.iter().all(|region| {
                region.mechanism == "metal-w8-mlp-stack3-boundary-v1"
                    && mlp_epilogue_runtime_is_exact(
                        region.mlp_epilogue_runtime,
                        body_mlp_epilogue,
                        body_mlp_epilogue.boundary_dispatches(),
                        body_mlp_epilogue.boundary_barriers(),
                    )
                    && region
                        .quantization
                        .iter()
                        .copied()
                        .all(quantization_profile_is_exact)
            }),
        six_region_execution_valid: initial_valid && boundaries_valid,
        tail_execution_and_phase_valid: tail_valid,
        aggregate_ledger_valid: official_aggregate_ledger_is_exact(aggregate, body_mlp_epilogue),
        generation_receipt_valid: boundary_tail_generation_receipt_is_exact(
            generation_receipt,
            body_tail_calls,
            phase,
            body_mlp_epilogue,
        ),
        terminal_clear: !stats.terminal_error
            && generation_receipt
                .get("terminal_error")
                .and_then(Value::as_bool)
                == Some(false),
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct TeacherOracle {
    teacher_inputs: Vec<u32>,
    expected_outputs: Vec<u32>,
}

enum SameProcessCpuOracle {
    Teacher {
        prefill_token: u32,
        oracle: TeacherOracle,
    },
    Free(Vec<u32>),
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

fn validate_finalized_cpu_receipt_custody(
    receipt: &Value,
    expected_identity: &Value,
) -> Result<(), String> {
    if receipt
        .get("input_receipt_verified_at_publication")
        .and_then(Value::as_bool)
        != Some(false)
        || receipt.get("generation_path_contract") != Some(&Value::Null)
        || receipt
            .get("official_layer_schedule_valid")
            .and_then(Value::as_bool)
            != Some(true)
        || receipt.get("prompt").and_then(Value::as_str) != Some(PROMPT)
    {
        return Err("CPU receipt is not a finalized standalone oracle receipt".into());
    }
    let start = expected_identity
        .get("custody")
        .ok_or_else(|| "current identity omitted start custody".to_string())?;
    let end = receipt
        .get("custody_end_verification")
        .ok_or_else(|| "CPU receipt omitted end custody".to_string())?;
    if end.get("verified_at_end").and_then(Value::as_bool) != Some(true)
        || end.get("source_set_id") != start.pointer("/sources/set_id")
        || end.get("source_set_coverage") != start.pointer("/sources/coverage")
        || start
            .pointer("/sources/binary_attestation_authoritative_for_full_executable")
            .and_then(Value::as_bool)
            != Some(true)
        || end.get("binary") != start.get("binary")
        || end.get("gate") != start.pointer("/sources/gate")
        || end.get("rust_and_bridge_sources") != start.pointer("/sources/rust_and_bridge_sources")
        || end.get("compiled_metal_shader_sources")
            != start.pointer("/sources/compiled_metal_shader_sources")
        || end.pointer("/model_dir/path") != start.pointer("/model_dir/path")
        || end.pointer("/model_dir/cache_present") != start.pointer("/model_dir/cache_present")
        || end.pointer("/model_dir/artifacts") != start.pointer("/model_dir/artifacts")
        || end
            .pointer("/model_dir/loaded_from_start_pinned_artifacts")
            .and_then(Value::as_bool)
            != Some(true)
        || end.pointer("/deployment_profile/path") != start.pointer("/profile/path")
        || end.pointer("/deployment_profile/size") != start.pointer("/profile/file_size")
        || end.pointer("/deployment_profile/sha256") != start.pointer("/profile/file_sha256")
        || end.pointer("/source_lock/path") != start.pointer("/source_lock/path")
        || end.pointer("/source_lock/size") != start.pointer("/source_lock/file_size")
        || end.pointer("/source_lock/sha256") != start.pointer("/source_lock/file_sha256")
    {
        return Err("CPU receipt end custody does not match its frozen start custody".into());
    }
    Ok(())
}

fn validate_cpu_teacher_receipt(
    receipt: &Value,
    expected_identity: &Value,
    prompt_tokens: &[u32],
    prefill_token: u32,
) -> Result<TeacherOracle, String> {
    validate_finalized_cpu_receipt_custody(receipt, expected_identity)?;
    if receipt.get("format").and_then(Value::as_str) != Some(CPU_TEACHER_FORMAT)
        || receipt.get("mode").and_then(Value::as_str) != Some(Mode::CpuTeacher.label())
        || receipt.get("body_mlp_epilogue").and_then(Value::as_str)
            != Some(BodyMlpEpilogue::Legacy.label())
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

fn compute_same_process_free_oracle(
    model: &mut GeneralQwen35,
    prompt_tokens: &[u32],
) -> Result<Vec<u32>, Box<dyn Error>> {
    let (generated, _) =
        model.generate_streaming(LlmInput::text(prompt_tokens), STEPS, |_| {}, None)?;
    if generated.len() != STEPS {
        return Err(format!(
            "same-process CPU free oracle returned {} tokens, expected {STEPS}",
            generated.len()
        )
        .into());
    }
    Ok(generated)
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
    validate_finalized_cpu_receipt_custody(receipt, expected_identity)?;
    if receipt.get("format").and_then(Value::as_str) != Some(CPU_FREE_FORMAT)
        || receipt.get("mode").and_then(Value::as_str) != Some(Mode::CpuFree.label())
        || receipt.get("body_mlp_epilogue").and_then(Value::as_str)
            != Some(BodyMlpEpilogue::Legacy.label())
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
    body_mlp_epilogue: BodyMlpEpilogue,
    input_receipt: Option<PathBuf>,
    output: PathBuf,
}

fn usage() -> &'static str {
    "Usage: qwen35_metal_w8_boundary_tail_head_v1_gate \
  --model-dir OFFICIAL_LOCAL_QWEN35_0_8B \
  --source-lock SOURCE_LOCK.json \
  --mode cpu-teacher|boundary-tail-v1-teacher|cpu-free|boundary-tail-v1-free \
  [--body-mlp-epilogue legacy|down-residual-fused] \
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
    let mut body_mlp_epilogue = None;
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
            "--body-mlp-epilogue" => {
                let parsed = BodyMlpEpilogue::parse(&value.to_string_lossy())?;
                if body_mlp_epilogue.replace(parsed).is_some() {
                    return Err("duplicate --body-mlp-epilogue".into());
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
        body_mlp_epilogue: body_mlp_epilogue.unwrap_or(BodyMlpEpilogue::Legacy),
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
    if !args.mode.is_candidate() && args.body_mlp_epilogue != BodyMlpEpilogue::Legacy {
        return Err("CPU modes require --body-mlp-epilogue legacy".into());
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

fn validate_receipt_paths_outside_model_dir(
    model_dir: &Path,
    input_receipt: Option<&PathBuf>,
    output: &Path,
) -> Result<PinnedOutputTarget, String> {
    let canonical_model = std::fs::canonicalize(model_dir)
        .map_err(|error| format!("cannot canonicalize model directory: {error}"))?;
    if let Some(input) = input_receipt {
        let canonical_input = std::fs::canonicalize(input)
            .map_err(|error| format!("cannot canonicalize input receipt: {error}"))?;
        if canonical_input.starts_with(&canonical_model) {
            return Err("--input-receipt must be outside the frozen model directory".into());
        }
    }
    let target = PinnedOutputTarget::capture(output)?;
    if target.canonical_parent.starts_with(&canonical_model) {
        return Err("--output must be outside the frozen model directory".into());
    }
    Ok(target)
}

fn finalize_run_with_end_custody<F>(
    mut result: RunResult,
    output: &PinnedOutputTarget,
    verify_end_custody: F,
) -> Result<RunResult, Box<dyn Error>>
where
    F: FnOnce() -> Result<Value, Box<dyn Error>>,
{
    let custody_end_verification = verify_end_custody()?;
    if let Some(guard) = result.input_receipt_guard.as_ref() {
        gate_evidence::verify_file_unchanged(&guard.path, &guard.attestation, guard.label)?;
        require_input_attestation_outside_model(&guard.attestation, &guard.canonical_model_dir)?;
    }
    output.verify_path_binding()?;
    result
        .receipt
        .as_object_mut()
        .ok_or("gate receipt root must be an object")?
        .insert("custody_end_verification".into(), custody_end_verification);
    result
        .receipt
        .as_object_mut()
        .expect("checked above")
        .insert(
            "input_receipt_verified_at_publication".into(),
            Value::Bool(result.input_receipt_guard.is_some()),
        );
    publish_receipt_create_new(output, &result.receipt)?;
    Ok(result)
}

fn require_input_attestation_outside_model(
    attestation: &gate_evidence::FileAttestation,
    canonical_model_dir: &Path,
) -> Result<(), String> {
    if attestation.path.starts_with(canonical_model_dir) {
        return Err("--input-receipt resolved inside the frozen model directory".into());
    }
    Ok(())
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
    let output_target = validate_receipt_paths_outside_model_dir(
        custody.model_dir(),
        args.input_receipt.as_ref(),
        &args.output,
    )?;
    let canonical_model_dir = custody.model_dir().to_path_buf();
    let canonical_source_lock = fs::canonicalize(&args.source_lock)?;
    let binary_path = fs::canonicalize(std::env::current_exe()?)?;

    let tokenizer = Tokenizer::from_bytes(
        custody.pinned_model_artifact_bytes("tokenizer.json")?,
        Some(custody.pinned_model_artifact_bytes("tokenizer_config.json")?),
        Some(custody.pinned_model_artifact_bytes("chat_template.jinja")?),
    )?;
    let formatted = tokenizer.apply_chat_template(&[ChatMessage::user(PROMPT)])?;
    let prompt_tokens = tokenizer.encode(&formatted)?;
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
    let same_process_oracle_started = Instant::now();
    let same_process_cpu_oracle = if args.mode.is_candidate() {
        let mut cpu_model =
            GeneralQwen35::from_weights(config.clone(), tensors.clone(), Device::Cpu, max_context)?;
        let oracle = match args.mode {
            Mode::BoundaryTailV1Teacher => {
                let (prefill_token, oracle) = compute_same_process_teacher_oracle(
                    &mut cpu_model,
                    &prompt_tokens,
                    vocab_size,
                )?;
                SameProcessCpuOracle::Teacher {
                    prefill_token,
                    oracle,
                }
            }
            Mode::BoundaryTailV1Free => SameProcessCpuOracle::Free(
                compute_same_process_free_oracle(&mut cpu_model, &prompt_tokens)?,
            ),
            Mode::CpuTeacher | Mode::CpuFree => unreachable!("candidate mode checked above"),
        };
        drop(cpu_model);
        Some(oracle)
    } else {
        None
    };
    let same_process_cpu_oracle_ms = same_process_oracle_started.elapsed().as_secs_f64() * 1_000.0;
    let construct_started = Instant::now();
    let mut model = if args.mode.is_candidate() {
        GeneralQwen35::from_weights_with_metal_w8_mlp_stack3_boundary_tail_head_v1_with_body_mlp_epilogue(
            config,
            tensors,
            Device::Cpu,
            max_context,
            args.body_mlp_epilogue.metal(),
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
        "model_load_binding": {
            "small_artifacts": "owned-bytes-retained-from-start-attested-file-descriptors",
            "checkpoint": "mmap-from-start-attested-pinned-file-descriptor",
            "checkpoint_index_role": "source-lock evidence only; the single frozen shard is loaded directly",
            "identity_fields": ["device", "inode", "size", "nlink", "ctime", "sha256"],
        },
        "cpu_reference_constructor": "GeneralQwen35::from_weights",
        "candidate_constructor": "GeneralQwen35::from_weights_with_metal_w8_mlp_stack3_boundary_tail_head_v1_with_body_mlp_epilogue",
        "custody": custody.receipt_json(),
    });
    let setup = json!({
        "checkpoint_load_ms": checkpoint_load_ms,
        "model_construct_ms": model_construct_ms,
        "same_process_cpu_oracle_ms": same_process_cpu_oracle_ms,
        "same_process_cpu_oracle": args.mode.is_candidate(),
        "timing_classification": "single-pass diagnostic timing only; never formal promotion evidence",
    });
    let result = match args.mode {
        Mode::CpuTeacher | Mode::BoundaryTailV1Teacher => run_teacher(
            &args,
            &mut model,
            &prompt_tokens,
            vocab_size,
            &canonical_model_dir,
            same_process_cpu_oracle.as_ref(),
            identity,
            setup,
        )?,
        Mode::CpuFree | Mode::BoundaryTailV1Free => run_free(
            &args,
            &mut model,
            &prompt_tokens,
            &canonical_model_dir,
            same_process_cpu_oracle.as_ref(),
            identity,
            setup,
        )?,
    };
    let result = finalize_run_with_end_custody(result, &output_target, || {
        custody.verify_unchanged_receipt()
    })?;
    println!("{}", serde_json::to_string(&result.receipt)?);
    Ok(result.passed)
}

fn run_teacher(
    args: &Args,
    model: &mut GeneralQwen35,
    prompt_tokens: &[u32],
    vocab_size: usize,
    canonical_model_dir: &Path,
    same_process_cpu_oracle: Option<&SameProcessCpuOracle>,
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
                "body_mlp_epilogue": args.body_mlp_epilogue.label(),
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
            input_receipt_guard: None,
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
        args.body_mlp_epilogue,
    );
    let input_path = args
        .input_receipt
        .as_ref()
        .ok_or("candidate teacher mode requires --input-receipt")?;
    let (cpu_receipt, input_attestation) =
        gate_evidence::read_attested_json(input_path, "boundary-tail v1 CPU teacher receipt")?;
    require_input_attestation_outside_model(&input_attestation, canonical_model_dir)?;
    let receipt_oracle =
        validate_cpu_teacher_receipt(&cpu_receipt, &identity, prompt_tokens, prefill_token)?;
    let (same_process_prefill, oracle) = match same_process_cpu_oracle {
        Some(SameProcessCpuOracle::Teacher {
            prefill_token,
            oracle,
        }) => (*prefill_token, oracle),
        _ => {
            return Err("candidate teacher mode requires an in-process CPU teacher oracle".into());
        }
    };
    if same_process_prefill != prefill_token || &receipt_oracle != oracle {
        return Err(
            "CPU teacher receipt does not match the same-process pinned-artifact oracle".into(),
        );
    }

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
        args.body_mlp_epilogue,
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
            "body_mlp_epilogue": args.body_mlp_epilogue.label(),
            "identity": identity,
            "oracle_source": "same-process-cpu-f32-from-pinned-artifacts",
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
                "body_mlp_epilogue": args.body_mlp_epilogue.runtime_label(),
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
        input_receipt_guard: Some(InputReceiptGuard {
            path: input_path.clone(),
            attestation: input_attestation,
            label: "boundary-tail v1 CPU teacher receipt",
            canonical_model_dir: canonical_model_dir.to_path_buf(),
        }),
    })
}

fn run_free(
    args: &Args,
    model: &mut GeneralQwen35,
    prompt_tokens: &[u32],
    canonical_model_dir: &Path,
    same_process_cpu_oracle: Option<&SameProcessCpuOracle>,
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
        require_input_attestation_outside_model(&attestation, canonical_model_dir)?;
        let receipt_expected = validate_cpu_free_receipt(&receipt, &identity, prompt_tokens)?;
        let expected = match same_process_cpu_oracle {
            Some(SameProcessCpuOracle::Free(expected)) => expected.clone(),
            _ => {
                return Err("candidate free mode requires an in-process CPU free oracle".into());
            }
        };
        if receipt_expected != expected {
            return Err(
                "CPU free receipt does not match the same-process pinned-artifact oracle".into(),
            );
        }
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
                "body_mlp_epilogue": args.body_mlp_epilogue.label(),
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
            input_receipt_guard: None,
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
        args.body_mlp_epilogue,
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
            "body_mlp_epilogue": args.body_mlp_epilogue.label(),
            "identity": identity,
            "oracle_source": "same-process-cpu-f32-from-pinned-artifacts",
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
                "body_mlp_epilogue": args.body_mlp_epilogue.runtime_label(),
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
        input_receipt_guard: Some(InputReceiptGuard {
            path: input_path.clone(),
            attestation: input_attestation,
            label: "boundary-tail v1 CPU free receipt",
            canonical_model_dir: canonical_model_dir.to_path_buf(),
        }),
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

    fn test_finalized_cpu_identity() -> Value {
        json!({
            "custody": {
                "profile": {"path": "/profile", "file_size": 11, "file_sha256": "profile"},
                "source_lock": {"path": "/source-lock", "file_size": 12, "file_sha256": "lock"},
                "model_dir": {"path": "/model", "cache_present": false, "artifacts": {}},
                "binary": {"sha256": "binary"},
                "sources": {
                    "set_id": "test-explicit-source-set",
                    "coverage": "explicit-non-transitive-source-set-v1",
                    "binary_attestation_authoritative_for_full_executable": true,
                    "gate": {"sha256": "gate"},
                    "rust_and_bridge_sources": {},
                    "compiled_metal_shader_sources": {},
                },
            }
        })
    }

    fn finalize_cpu_receipt_fixture(receipt: &mut Value, identity: &Value) {
        let custody = &identity["custody"];
        receipt["prompt"] = json!(PROMPT);
        receipt["official_layer_schedule_valid"] = json!(true);
        receipt["generation_path_contract"] = Value::Null;
        receipt["input_receipt_verified_at_publication"] = json!(false);
        receipt["custody_end_verification"] = json!({
            "verified_at_end": true,
            "source_set_id": custody["sources"]["set_id"],
            "source_set_coverage": custody["sources"]["coverage"],
            "deployment_profile": {
                "path": custody["profile"]["path"],
                "size": custody["profile"]["file_size"],
                "sha256": custody["profile"]["file_sha256"],
            },
            "source_lock": {
                "path": custody["source_lock"]["path"],
                "size": custody["source_lock"]["file_size"],
                "sha256": custody["source_lock"]["file_sha256"],
            },
            "model_dir": {
                "path": custody["model_dir"]["path"],
                "cache_present": custody["model_dir"]["cache_present"],
                "artifacts": custody["model_dir"]["artifacts"],
                "loaded_from_start_pinned_artifacts": true,
            },
            "binary": custody["binary"],
            "gate": custody["sources"]["gate"],
            "rust_and_bridge_sources": custody["sources"]["rust_and_bridge_sources"],
            "compiled_metal_shader_sources": custody["sources"]["compiled_metal_shader_sources"],
        });
    }

    fn official_aggregate_fixture(
        body_mlp_epilogue: BodyMlpEpilogue,
    ) -> Qwen35MetalW8MlpStack3BoundaryTailHeadV1AggregateLedger {
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
                kernel_dispatches_per_decode: body_mlp_epilogue.initial_dispatches(),
                buffer_barriers_per_decode: body_mlp_epilogue.initial_barriers(),
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
            kernel_dispatches_per_decode: body_mlp_epilogue.boundary_dispatches(),
            buffer_barriers_per_decode: body_mlp_epilogue.boundary_barriers(),
            commits_per_decode: 1,
            waits_per_decode: 1,
            intermediate_host_finite_checks_per_decode: 0,
            final_output_finite_checks_per_decode: 1,
        };
        Qwen35MetalW8MlpStack3BoundaryTailHeadV1AggregateLedger {
            scope: "resident-mtlbuffer-only",
            exclusions: "CPU F32 weights, host F32 tied embedding and exact four-candidate rerank, host allocations, Metal pipelines/libraries/queues, command objects, driver allocations, full attention/KV, model loader, and prefill CPU head",
            includes_lm_head: true,
            body_mlp_epilogue: body_mlp_epilogue.metal(),
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
            kernel_dispatches_per_decode: body_mlp_epilogue.aggregate_dispatches(),
            buffer_barriers_per_decode: body_mlp_epilogue.aggregate_barriers(),
            commits_per_decode: 7,
            waits_per_decode: 7,
        }
    }

    fn generation_receipt_fixture(
        calls: usize,
        phase: TailPhaseCounts,
        body_mlp_epilogue: BodyMlpEpilogue,
    ) -> Value {
        let transfer = calls * 4_096;
        json!({
            "format": "apxinf-qwen35-mlp-stack3-boundary-tail-head-generation-path-v1",
            "mechanism": "metal-w8-mlp-stack3-boundary-tail-head-v1",
            "body_mlp_epilogue": body_mlp_epilogue.runtime_label(),
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
                "mlp_epilogue_runtime": {
                    "requested_profile": body_mlp_epilogue.runtime_label(),
                    "observed_profile": body_mlp_epilogue.runtime_label(),
                    "mlp_down_function_name": body_mlp_epilogue.mlp_down_function_name(),
                    "kernel_dispatches_per_decode": body_mlp_epilogue.initial_dispatches(),
                    "buffer_barriers_per_decode": body_mlp_epilogue.initial_barriers(),
                },
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
                "mlp_epilogue_runtime": {
                    "requested_profile": body_mlp_epilogue.runtime_label(),
                    "observed_profile": body_mlp_epilogue.runtime_label(),
                    "mlp_down_function_name": body_mlp_epilogue.mlp_down_function_name(),
                    "kernel_dispatches_per_decode": body_mlp_epilogue.boundary_dispatches(),
                    "buffer_barriers_per_decode": body_mlp_epilogue.boundary_barriers(),
                },
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
                "body_mlp_epilogue": body_mlp_epilogue.runtime_label(),
                "persistent_mtlbuffer_bytes": 799_543_312usize,
                "allocated_buffers": 494,
                "shared_buffers": 443,
                "private_buffers": 51,
                "host_to_device_bytes_per_decode": 28_672,
                "device_to_host_bytes_per_decode": 28_688,
                "state_host_transfer_bytes_per_decode": 0,
                "command_buffers_per_decode": 7,
                "compute_encoders_per_decode": 24,
                "kernel_dispatches_per_decode": body_mlp_epilogue.aggregate_dispatches(),
                "buffer_barriers_per_decode": body_mlp_epilogue.aggregate_barriers(),
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
        assert_eq!(args.body_mlp_epilogue, BodyMlpEpilogue::Legacy);
        assert_eq!(
            args.input_receipt.unwrap(),
            std::path::PathBuf::from("/cpu-free.json")
        );
        let fused = parse_args_from([
            "gate",
            "--model-dir",
            "/model",
            "--source-lock",
            "/source-lock.json",
            "--mode",
            "boundary-tail-v1-free",
            "--body-mlp-epilogue",
            "down-residual-fused",
            "--input-receipt",
            "/cpu-free.json",
            "--output",
            "/new-fused-receipt.json",
        ])
        .unwrap();
        assert_eq!(fused.body_mlp_epilogue, BodyMlpEpilogue::DownResidualFused);
        assert!(parse_args_from([
            "gate",
            "--model-dir",
            "/model",
            "--source-lock",
            "/source-lock.json",
            "--mode",
            "cpu-free",
            "--body-mlp-epilogue",
            "down-residual-fused",
            "--output",
            "/new-cpu-receipt.json",
        ])
        .is_err());
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
        let identity = test_finalized_cpu_identity();
        let mut receipt = json!({
            "format": CPU_TEACHER_FORMAT,
            "mode": Mode::CpuTeacher.label(),
            "body_mlp_epilogue": BodyMlpEpilogue::Legacy.label(),
            "passed": true,
            "identity": identity,
            "prompt_token_ids": [1, 2],
            "prefill_token": 7,
            "comparisons": 128,
            "teacher_input_ids": inputs,
            "cpu_expected_output_ids": expected,
        });
        finalize_cpu_receipt_fixture(&mut receipt, &identity);

        let oracle = validate_cpu_teacher_receipt(&receipt, &identity, &[1, 2], 7).unwrap();
        assert_eq!(oracle.teacher_inputs.len(), 128);
        assert_eq!(oracle.expected_outputs.len(), 128);
        let mut missing_profile = receipt.clone();
        missing_profile
            .as_object_mut()
            .unwrap()
            .remove("body_mlp_epilogue");
        assert!(validate_cpu_teacher_receipt(&missing_profile, &identity, &[1, 2], 7).is_err());
        let mut wrong_profile = receipt.clone();
        wrong_profile["body_mlp_epilogue"] = json!(BodyMlpEpilogue::DownResidualFused.label());
        assert!(validate_cpu_teacher_receipt(&wrong_profile, &identity, &[1, 2], 7).is_err());
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
        let identity = test_finalized_cpu_identity();
        let tokens = (100..228).collect::<Vec<u32>>();
        let mut receipt = json!({
            "format": CPU_FREE_FORMAT,
            "mode": Mode::CpuFree.label(),
            "body_mlp_epilogue": BodyMlpEpilogue::Legacy.label(),
            "passed": true,
            "identity": identity,
            "prompt_token_ids": [1, 2],
            "max_new_tokens": 128,
            "eos_stopping": false,
            "generated_token_ids": tokens,
        });
        finalize_cpu_receipt_fixture(&mut receipt, &identity);
        assert_eq!(
            validate_cpu_free_receipt(&receipt, &identity, &[1, 2]).unwrap(),
            tokens
        );
        let mut missing_profile = receipt.clone();
        missing_profile
            .as_object_mut()
            .unwrap()
            .remove("body_mlp_epilogue");
        assert!(validate_cpu_free_receipt(&missing_profile, &identity, &[1, 2]).is_err());
        let mut wrong_profile = receipt.clone();
        wrong_profile["body_mlp_epilogue"] = json!(BodyMlpEpilogue::DownResidualFused.label());
        assert!(validate_cpu_free_receipt(&wrong_profile, &identity, &[1, 2]).is_err());
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
        let target = PinnedOutputTarget::capture(&output).unwrap();
        publish_receipt_create_new(&target, &first).unwrap();
        let first_bytes = std::fs::read(&output).unwrap();
        assert!(publish_receipt_create_new(&target, &second).is_err());
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
        let aggregate = official_aggregate_fixture(BodyMlpEpilogue::Legacy);
        assert!(official_aggregate_ledger_is_exact(
            &aggregate,
            BodyMlpEpilogue::Legacy,
        ));

        let fused = official_aggregate_fixture(BodyMlpEpilogue::DownResidualFused);
        assert!(official_aggregate_ledger_is_exact(
            &fused,
            BodyMlpEpilogue::DownResidualFused,
        ));
        assert!(!official_aggregate_ledger_is_exact(
            &fused,
            BodyMlpEpilogue::Legacy,
        ));

        let mut wrong = aggregate.clone();
        wrong.boundaries[2].ledger.total_persistent_bytes -= 1;
        assert!(!official_aggregate_ledger_is_exact(
            &wrong,
            BodyMlpEpilogue::Legacy,
        ));
        let mut wrong = aggregate.clone();
        wrong.tail.host_output_bytes_per_decode -= 1;
        assert!(!official_aggregate_ledger_is_exact(
            &wrong,
            BodyMlpEpilogue::Legacy,
        ));
        let mut wrong = aggregate.clone();
        wrong.total_persistent_mtlbuffer_bytes -= 1;
        assert!(!official_aggregate_ledger_is_exact(
            &wrong,
            BodyMlpEpilogue::Legacy,
        ));
        let mut wrong = aggregate;
        wrong.kernel_dispatches_per_decode -= 1;
        assert!(!official_aggregate_ledger_is_exact(
            &wrong,
            BodyMlpEpilogue::Legacy,
        ));
    }

    #[test]
    fn generation_receipt_locks_six_regions_tail_phases_and_commit_masks() {
        let phase = TailPhaseCounts::teacher(7);
        let receipt = generation_receipt_fixture(7, phase, BodyMlpEpilogue::Legacy);
        assert!(boundary_tail_generation_receipt_is_exact(
            &receipt,
            7,
            phase,
            BodyMlpEpilogue::Legacy,
        ));

        let mut wrong = receipt.clone();
        wrong["initial_stack"]["last_state_commit_mask"] = json!(0b011);
        assert!(!boundary_tail_generation_receipt_is_exact(
            &wrong,
            7,
            phase,
            BodyMlpEpilogue::Legacy,
        ));
        let mut wrong = receipt.clone();
        wrong["boundaries"][3]["last_state_commit_mask"] = json!(0b110);
        assert!(!boundary_tail_generation_receipt_is_exact(
            &wrong,
            7,
            phase,
            BodyMlpEpilogue::Legacy,
        ));
        let mut wrong = receipt.clone();
        wrong["decode_head"]["teacher_calls"] = json!(6);
        assert!(!boundary_tail_generation_receipt_is_exact(
            &wrong,
            7,
            phase,
            BodyMlpEpilogue::Legacy,
        ));

        let mut wrong = receipt.clone();
        wrong["terminal_error"] = json!(true);
        assert!(!boundary_tail_generation_receipt_is_exact(
            &wrong,
            7,
            phase,
            BodyMlpEpilogue::Legacy,
        ));
        let mut wrong = receipt;
        wrong["decode_head"]["terminal_error"] = json!(true);
        assert!(!boundary_tail_generation_receipt_is_exact(
            &wrong,
            7,
            phase,
            BodyMlpEpilogue::Legacy,
        ));

        let free_phase = TailPhaseCounts::free(7);
        let free_receipt =
            generation_receipt_fixture(7, free_phase, BodyMlpEpilogue::DownResidualFused);
        assert!(boundary_tail_generation_receipt_is_exact(
            &free_receipt,
            7,
            free_phase,
            BodyMlpEpilogue::DownResidualFused,
        ));
        let mut wrong = free_receipt;
        wrong["prefill_head"]["tail_transactions"] = json!(1);
        assert!(!boundary_tail_generation_receipt_is_exact(
            &wrong,
            7,
            free_phase,
            BodyMlpEpilogue::DownResidualFused,
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
            input_receipt_guard: None,
        };
        let target = PinnedOutputTarget::capture(&output).unwrap();
        let finalized =
            finalize_run_with_end_custody(result, &target, || Ok(json!({"verified_at_end": true})))
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
        let input_attestation =
            gate_evidence::attest_file(&input, "boundary-tail v1 CPU test receipt", None).unwrap();
        let result = RunResult {
            receipt: json!({"passed": true}),
            passed: true,
            input_receipt_guard: Some(InputReceiptGuard {
                path: input.clone(),
                attestation: input_attestation,
                label: "boundary-tail v1 CPU test receipt",
                canonical_model_dir: temp_output("model-root"),
            }),
        };

        let target = PinnedOutputTarget::capture(&output).unwrap();

        let finalized = finalize_run_with_end_custody(result, &target, || {
            std::fs::write(&input, b"{\"passed\":null}\n").unwrap();
            Ok(json!({"verified_at_end": true}))
        });
        assert!(finalized.is_err());
        assert!(!output.exists());

        std::fs::remove_file(input).unwrap();
    }

    #[test]
    fn finalize_rechecks_an_unchanged_input_receipt_immediately_before_publication() {
        let input = temp_output("final-input-stable.json");
        let output = temp_output("final-input-stable-output.json");
        std::fs::write(&input, b"{\"passed\":true}\n").unwrap();
        let input_attestation =
            gate_evidence::attest_file(&input, "stable CPU test receipt", None).unwrap();
        let result = RunResult {
            receipt: json!({"passed": true}),
            passed: true,
            input_receipt_guard: Some(InputReceiptGuard {
                path: input.clone(),
                attestation: input_attestation,
                label: "stable CPU test receipt",
                canonical_model_dir: temp_output("model-root"),
            }),
        };
        let target = PinnedOutputTarget::capture(&output).unwrap();
        finalize_run_with_end_custody(result, &target, || Ok(json!({"verified_at_end": true})))
            .unwrap();
        let published: Value = serde_json::from_slice(&std::fs::read(&output).unwrap()).unwrap();
        assert_eq!(published["input_receipt_verified_at_publication"], true);
        std::fs::remove_file(input).unwrap();
        std::fs::remove_file(output).unwrap();
    }

    #[test]
    fn pinned_output_parent_rejects_a_symlink_rebind_before_publication() {
        use std::os::unix::fs::symlink;

        let root = temp_output("output-parent-rebind");
        let model = root.join("model");
        let evidence = root.join("evidence");
        let alias = root.join("output-link");
        std::fs::create_dir_all(&model).unwrap();
        std::fs::create_dir_all(&evidence).unwrap();
        symlink(&evidence, &alias).unwrap();
        let output = alias.join("candidate.json");
        let target = validate_receipt_paths_outside_model_dir(&model, None, &output).unwrap();
        let result = RunResult {
            receipt: json!({"passed": false}),
            passed: false,
            input_receipt_guard: None,
        };
        let finalized = finalize_run_with_end_custody(result, &target, || {
            std::fs::remove_file(&alias).unwrap();
            symlink(&model, &alias).unwrap();
            Ok(json!({"verified_at_end": true}))
        });
        assert!(finalized.is_err());
        assert!(!model.join("candidate.json").exists());
        assert!(!evidence.join("candidate.json").exists());
        std::fs::remove_dir_all(root).unwrap();
    }
}
