//! Dedicated single-sample runner for the formal Qwen3.5 native v3 lanes.

#[path = "support/qwen35_boundary_tail_head_v1_gate_evidence.rs"]
#[allow(dead_code)]
mod gate_evidence;

use std::collections::HashMap;
use std::error::Error;
use std::ffi::{CStr, OsString};
use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

use apxinf_core::{Device, Tensor};
use apxinf_model::{GeneralQwen35, LlmInput, LlmTrait, Qwen35Config};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

const MAX_CONTEXT: usize = 256;
const STEPS: usize = 128;
const TEACHER_PREFILL_TOKENS: usize = 12;
const PROMPT_TOKEN_IDS: [u32; 13] = [
    248045, 846, 198, 9419, 248046, 198, 248045, 74455, 198, 248068, 271, 248069, 271,
];
const CANONICAL_FREE_TOKEN_IDS: [u32; 128] = [
    9419, 0, 2500, 628, 353, 1438, 488, 3242, 30, 25677, 232, 248046, 198, 248044, 248045, 2570,
    198, 9419, 248046, 198, 248045, 74455, 198, 248068, 271, 248069, 271, 9419, 0, 2500, 628, 353,
    7543, 488, 3242, 30, 25677, 232, 248046, 198, 248044, 248045, 2570, 198, 9419, 248046, 198,
    248045, 74455, 198, 248068, 271, 248069, 271, 9419, 0, 2500, 628, 353, 1438, 488, 3242, 30,
    25677, 232, 248046, 198, 248044, 248045, 2570, 198, 9419, 248046, 198, 248045, 74455, 198,
    248068, 271, 248069, 271, 9419, 0, 2500, 628, 353, 7543, 488, 3242, 30, 25677, 232, 248046,
    198, 248044, 248045, 2570, 198, 9419, 248046, 198, 248045, 74455, 198, 248068, 271, 248069,
    271, 9419, 0, 2500, 628, 353, 1438, 488, 3242, 30, 25677, 232, 248046, 198, 248044, 248045,
    2570, 198, 9419, 248046, 198,
];
const TEACHER_INPUT_TOKEN_IDS: [u32; 128] = [
    271, 9419, 0, 2500, 628, 353, 1438, 488, 3242, 30, 25677, 232, 248046, 198, 248044, 248045,
    2570, 198, 9419, 248046, 198, 248045, 74455, 198, 248068, 271, 248069, 271, 9419, 0, 2500, 628,
    353, 7543, 488, 3242, 30, 25677, 232, 248046, 198, 248044, 248045, 2570, 198, 9419, 248046,
    198, 248045, 74455, 198, 248068, 271, 248069, 271, 9419, 0, 2500, 628, 353, 1438, 488, 3242,
    30, 25677, 232, 248046, 198, 248044, 248045, 2570, 198, 9419, 248046, 198, 248045, 74455, 198,
    248068, 271, 248069, 271, 9419, 0, 2500, 628, 353, 7543, 488, 3242, 30, 25677, 232, 248046,
    198, 248044, 248045, 2570, 198, 9419, 248046, 198, 248045, 74455, 198, 248068, 271, 248069,
    271, 9419, 0, 2500, 628, 353, 1438, 488, 3242, 30, 25677, 232, 248046, 198, 248044, 248045,
    2570, 198, 9419, 248046,
];
const LOCKED_CHECKPOINT: &str = "model.safetensors-00001-of-00001.safetensors";
const RUNNER_SOURCE_NAME: &str = "qwen35_native_raw_token_runner_v3.rs";
const RUNNER_SOURCE_BYTES: &[u8] = include_bytes!("qwen35_native_raw_token_runner_v3.rs");
const CAMPAIGN_ID: &str = "qwen35-0.8b-cross-runtime-formal-v3-20260826";
const SUBCAMPAIGN_ID: &str = "qwen35-0.8b-native-apxinf-vs-llamacpp-formal-v3-20260826";
const EDGE_ID: &str = "NATIVE_A_VS_L";
const AN_CONFIGURATION_ID: &str = "ApxInf-native-hybrid-G32-G64-W8-CPU-F32-remainder-F32-KV-v3";
const EMBEDDED_CANDIDATE_COMMIT: Option<&str> = option_env!("APXINF_CANDIDATE_COMMIT");
const FUSED_PROFILE: &str = "gdn-core-fused-v1";
const FUSED_FUNCTION_CHAIN: &str = "gdn_core_fused_v1";
const FUSED_MECHANISM: &str = "metal-w8-mlp-stack3-boundary-tail-head-gdn-core-fused-v1";
const INITIAL_MECHANISM: &str = "metal-w8-linear-layer-stack3-gdn-core-fused-v1";
const BOUNDARY_MECHANISM: &str = "metal-w8-mlp-stack3-boundary-gdn-core-fused-v1";
const THREAD_OVERRIDE_ENVIRONMENT: [&str; 4] = [
    "VECLIB_MAXIMUM_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
];
const BOUNDARY_REGIONS: [(usize, [usize; 3]); 5] = [
    (3, [4, 5, 6]),
    (7, [8, 9, 10]),
    (11, [12, 13, 14]),
    (15, [16, 17, 18]),
    (19, [20, 21, 22]),
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RunMode {
    NativeV3Free,
    NativeV3Teacher,
}

impl RunMode {
    fn parse(value: &str) -> Result<Self, Box<dyn Error>> {
        match value {
            "native-v3-free" => Ok(Self::NativeV3Free),
            "native-v3-teacher" => Ok(Self::NativeV3Teacher),
            _ => Err(format!("unsupported native v3 mode: {value}").into()),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TeacherRole {
    Reference,
    Observed,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TailPhase {
    Teacher,
    Free,
}

impl TeacherRole {
    fn parse(value: &str) -> Result<Self, Box<dyn Error>> {
        match value {
            "reference" => Ok(Self::Reference),
            "observed" => Ok(Self::Observed),
            _ => Err(format!("unsupported native v3 teacher role: {value}").into()),
        }
    }
}

struct Args {
    mode: RunMode,
    teacher_role: Option<TeacherRole>,
    model_dir: PathBuf,
    source_lock: PathBuf,
}

fn argmax(logits: &Tensor, vocab_size: usize) -> Result<u32, Box<dyn Error>> {
    if logits.shape().dims() != [1, vocab_size] {
        return Err(format!("expected logits [1, {vocab_size}], got {}", logits.shape()).into());
    }
    let mut best_score = f32::NEG_INFINITY;
    let mut best_token = None;
    for (token, &score) in logits.as_f32()?.iter().enumerate() {
        if !score.is_finite() {
            return Err(format!("full-vocabulary argmax found non-finite logit at {token}").into());
        }
        if best_token.is_none() || score > best_score {
            best_score = score;
            best_token = Some(u32::try_from(token)?);
        }
    }
    best_token.ok_or_else(|| "full-vocabulary argmax received no logits".into())
}

fn load_model_inputs(
    custody: &gate_evidence::GateCustody,
) -> Result<(Qwen35Config, HashMap<String, Tensor>), Box<dyn Error>> {
    let config_json = std::str::from_utf8(custody.pinned_model_artifact_bytes("config.json")?)?;
    let config = Qwen35Config::from_json_str(config_json)?;
    let (tensors, _) = apxinf_loader::safetensors::load_native_file_filtered(
        custody.pinned_model_artifact_file(LOCKED_CHECKPOINT)?,
        |name| name.starts_with("model.language_model.") || name == "lm_head.weight",
    )?;
    custody.verify_pinned_model_handles_unchanged()?;
    Ok((config, tensors))
}

fn run_cpu_teacher_reference(
    config: Qwen35Config,
    tensors: HashMap<String, Tensor>,
) -> Result<Value, Box<dyn Error>> {
    let vocab_size = config.text.vocab_size;
    let mut model = GeneralQwen35::from_weights(config, tensors, Device::Cpu, MAX_CONTEXT)?;
    let _prefill = model
        .prefill_for_generation(LlmInput::text(&PROMPT_TOKEN_IDS[..TEACHER_PREFILL_TOKENS]))?;
    let mut reference_argmax_token_ids = Vec::with_capacity(STEPS);
    for (step, &teacher_token) in TEACHER_INPUT_TOKEN_IDS.iter().enumerate() {
        let position = u32::try_from(TEACHER_PREFILL_TOKENS + step)?;
        let logits = model.forward(&[teacher_token], position)?;
        reference_argmax_token_ids.push(argmax(&logits, vocab_size)?);
    }
    if reference_argmax_token_ids != CANONICAL_FREE_TOKEN_IDS {
        return Err("CPU/F32 teacher reference diverged from frozen free128".into());
    }
    Ok(json!({
        "format": "apxinf-qwen35-native-teacher-runtime-receipt-v3",
        "schema_version": 3,
        "mode": "native-v3-teacher",
        "teacher_role": "reference",
        "prefill_prompt_token_ids": &PROMPT_TOKEN_IDS[..TEACHER_PREFILL_TOKENS],
        "prefill_prompt_token_ids_count": TEACHER_PREFILL_TOKENS,
        "teacher_input_token_ids": &TEACHER_INPUT_TOKEN_IDS[..],
        "reference_argmax_token_ids": reference_argmax_token_ids,
        "eog_termination": false,
        "passed": true,
    }))
}

fn production_receipt_is_exact(
    receipt: Option<apxinf_metal::GdnCoreProductionReceiptV1>,
    calls: usize,
) -> bool {
    if calls == 0 {
        return receipt.is_none();
    }
    let Some(receipt) = receipt else {
        return false;
    };
    receipt.profile == apxinf_metal::GdnCoreProfileV1::Fused128
        && receipt.function_chain == FUSED_FUNCTION_CHAIN
        && receipt.gdn_core_seams == 3
        && receipt.kernel_dispatches == 3
        && receipt.explicit_buffer_barriers == 3
        && receipt.recurrent_or_fused_threads_per_threadgroup == 128
        && receipt.threadgroups == 48
        && receipt.launched_threads == 6_144
        && receipt.pipeline_thread_execution_width == 32
        && receipt.source_declared_threadgroup_memory_bytes == 2_060
        && receipt.pipeline_static_threadgroup_memory_bytes == 2_064
        && receipt.internal_threadgroup_barrier_sites_per_threadgroup == 4
        && receipt.fixed_shape_validated
        && receipt.rms_norm_eps_bits == 1.0e-6_f32.to_bits()
        && receipt.persistent_output_groups_per_row == 64
        && receipt.core_kernel_output_groups_per_row == 32
}

fn initial_ledger_is_exact(ledger: apxinf_metal::LinearLayerStack3BufferLedger) -> bool {
    ledger.gdn_core_profile == apxinf_metal::GdnCoreProfileV1::Fused128
        && ledger.gdn_function_chain == FUSED_FUNCTION_CHAIN
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
        && ledger.kernel_dispatches_per_decode == 30
        && ledger.explicit_buffer_barriers_per_decode == 27
        && ledger.gdn_core_seams_per_decode == 3
        && ledger.gdn_core_kernel_dispatches_per_decode == 3
        && ledger.gdn_core_explicit_buffer_barriers_per_decode == 3
        && ledger.gdn_core_recurrent_or_fused_threads_per_threadgroup == 128
        && ledger.gdn_core_threadgroups_per_decode == 48
        && ledger.gdn_core_launched_threads_per_decode == 6_144
        && ledger.gdn_core_source_declared_threadgroup_memory_bytes == 2_060
        && ledger.gdn_core_expected_pipeline_static_threadgroup_memory_bytes == 2_064
        && ledger.gdn_core_internal_threadgroup_barrier_sites_per_threadgroup == 4
        && ledger.commits_per_decode == 1
        && ledger.waits_per_decode == 1
        && ledger.intermediate_host_finite_checks_per_decode == 0
        && ledger.final_output_finite_checks_per_decode == 1
}

fn boundary_ledger_is_exact(ledger: apxinf_metal::MlpStack3BoundaryBufferLedgerV1) -> bool {
    ledger.gdn_core_profile == apxinf_metal::GdnCoreProfileV1::Fused128
        && ledger.gdn_function_chain == FUSED_FUNCTION_CHAIN
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
        && ledger.kernel_dispatches_per_decode == 35
        && ledger.explicit_buffer_barriers_per_decode == 31
        && ledger.gdn_core_seams_per_decode == 3
        && ledger.gdn_core_kernel_dispatches_per_decode == 3
        && ledger.gdn_core_explicit_buffer_barriers_per_decode == 3
        && ledger.gdn_core_recurrent_or_fused_threads_per_threadgroup == 128
        && ledger.gdn_core_threadgroups_per_decode == 48
        && ledger.gdn_core_launched_threads_per_decode == 6_144
        && ledger.gdn_core_source_declared_threadgroup_memory_bytes == 2_060
        && ledger.gdn_core_expected_pipeline_static_threadgroup_memory_bytes == 2_064
        && ledger.gdn_core_internal_threadgroup_barrier_sites_per_threadgroup == 4
        && ledger.commits_per_decode == 1
        && ledger.waits_per_decode == 1
        && ledger.intermediate_host_finite_checks_per_decode == 0
        && ledger.final_output_finite_checks_per_decode == 1
}

fn tail_ledger_is_exact(ledger: apxinf_metal::TailMlpHeadBufferLedgerV1) -> bool {
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

fn aggregate_ledger_is_exact(
    ledger: &apxinf_model::qwen35::general::Qwen35MetalW8MlpStack3BoundaryTailHeadV1AggregateLedger,
) -> bool {
    ledger.scope == "resident-mtlbuffer-only"
        && ledger.exclusions
            == "CPU F32 weights, host F32 tied embedding and exact four-candidate rerank, host allocations, Metal pipelines/libraries/queues, command objects, driver allocations, full attention/KV, model loader, and prefill CPU head"
        && ledger.includes_lm_head
        && ledger.gdn_core_profile == apxinf_metal::GdnCoreProfileV1::Fused128
        && ledger.gdn_function_chain == FUSED_FUNCTION_CHAIN
        && ledger.initial_stack.layer_indices == [0, 1, 2]
        && initial_ledger_is_exact(ledger.initial_stack.ledger)
        && ledger.boundaries.len() == BOUNDARY_REGIONS.len()
        && ledger
            .boundaries
            .iter()
            .zip(BOUNDARY_REGIONS)
            .all(|(entry, (boundary_layer, stack_layers))| {
                entry.boundary_mlp_layer_index == boundary_layer
                    && entry.stack_layer_indices == stack_layers
                    && boundary_ledger_is_exact(entry.ledger)
            })
        && ledger.tail_layer_index == 23
        && tail_ledger_is_exact(ledger.tail)
        && ledger.total_persistent_mtlbuffer_bytes == 799_543_312
        && ledger.allocated_buffers == 494
        && ledger.shared_buffers == 443
        && ledger.private_buffers == 51
        && ledger.host_to_device_bytes_per_decode == 28_672
        && ledger.device_to_host_bytes_per_decode == 28_688
        && ledger.state_host_transfer_bytes_per_decode == 0
        && ledger.command_buffers_per_decode == 7
        && ledger.compute_encoders_per_decode == 24
        && ledger.kernel_dispatches_per_decode == 213
        && ledger.explicit_buffer_barriers_per_decode == 189
        && ledger.commits_per_decode == 7
        && ledger.waits_per_decode == 7
}

fn aggregate_buffer_ledger_json(
    ledger: &apxinf_model::qwen35::general::Qwen35MetalW8MlpStack3BoundaryTailHeadV1AggregateLedger,
) -> Value {
    let initial = ledger.initial_stack.ledger;
    let boundaries = ledger
        .boundaries
        .iter()
        .map(|entry| {
            let region = entry.ledger;
            json!({
                "boundary_mlp_layer_index": entry.boundary_mlp_layer_index,
                "stack_layer_indices": entry.stack_layer_indices,
                "ledger": {
                    "gdn_core_profile": FUSED_PROFILE,
                    "gdn_function_chain": region.gdn_function_chain,
                    "scope": region.scope,
                    "exclusions": region.exclusions,
                    "abi_version": region.abi_version,
                    "stack_depth": region.stack_depth,
                    "allocated_buffers": region.allocated_buffers,
                    "shared_buffers": region.shared_buffers,
                    "private_buffers": region.private_buffers,
                    "packed_weight_bytes": region.packed_weight_bytes,
                    "packed_scale_bytes": region.packed_scale_bytes,
                    "f32_parameter_bytes": region.f32_parameter_bytes,
                    "active_state_bytes": region.active_state_bytes,
                    "scratch_state_bytes": region.scratch_state_bytes,
                    "activation_bytes": region.activation_bytes,
                    "total_persistent_bytes": region.total_persistent_bytes,
                    "host_input_bytes_per_decode": region.host_input_bytes_per_decode,
                    "host_output_bytes_per_decode": region.host_output_bytes_per_decode,
                    "state_host_transfer_bytes_per_decode": region.state_host_transfer_bytes_per_decode,
                    "command_buffers_per_decode": region.command_buffers_per_decode,
                    "compute_encoders_per_decode": region.compute_encoders_per_decode,
                    "kernel_dispatches_per_decode": region.kernel_dispatches_per_decode,
                    "explicit_buffer_barriers_per_decode": region.explicit_buffer_barriers_per_decode,
                    "gdn_core_seams_per_decode": region.gdn_core_seams_per_decode,
                    "gdn_core_kernel_dispatches_per_decode": region.gdn_core_kernel_dispatches_per_decode,
                    "gdn_core_explicit_buffer_barriers_per_decode": region.gdn_core_explicit_buffer_barriers_per_decode,
                    "gdn_core_recurrent_or_fused_threads_per_threadgroup": region.gdn_core_recurrent_or_fused_threads_per_threadgroup,
                    "gdn_core_threadgroups_per_decode": region.gdn_core_threadgroups_per_decode,
                    "gdn_core_launched_threads_per_decode": region.gdn_core_launched_threads_per_decode,
                    "gdn_core_source_declared_threadgroup_memory_bytes": region.gdn_core_source_declared_threadgroup_memory_bytes,
                    "gdn_core_expected_pipeline_static_threadgroup_memory_bytes": region.gdn_core_expected_pipeline_static_threadgroup_memory_bytes,
                    "gdn_core_internal_threadgroup_barrier_sites_per_threadgroup": region.gdn_core_internal_threadgroup_barrier_sites_per_threadgroup,
                    "commits_per_decode": region.commits_per_decode,
                    "waits_per_decode": region.waits_per_decode,
                    "intermediate_host_finite_checks_per_decode": region.intermediate_host_finite_checks_per_decode,
                    "final_output_finite_checks_per_decode": region.final_output_finite_checks_per_decode,
                },
            })
        })
        .collect::<Vec<_>>();
    let tail = ledger.tail;
    json!({
        "scope": ledger.scope,
        "exclusions": ledger.exclusions,
        "includes_lm_head": ledger.includes_lm_head,
        "gdn_core_profile": FUSED_PROFILE,
        "gdn_function_chain": ledger.gdn_function_chain,
        "initial_stack": {
            "layer_indices": ledger.initial_stack.layer_indices,
            "ledger": {
                "gdn_core_profile": FUSED_PROFILE,
                "gdn_function_chain": initial.gdn_function_chain,
                "allocated_buffers": initial.allocated_buffers,
                "shared_buffers": initial.shared_buffers,
                "private_buffers": initial.private_buffers,
                "packed_weight_bytes": initial.packed_weight_bytes,
                "packed_scale_bytes": initial.packed_scale_bytes,
                "f32_parameter_bytes": initial.f32_parameter_bytes,
                "active_state_bytes": initial.active_state_bytes,
                "scratch_state_bytes": initial.scratch_state_bytes,
                "activation_bytes": initial.activation_bytes,
                "total_persistent_bytes": initial.total_persistent_bytes,
                "host_input_bytes_per_decode": initial.host_input_bytes_per_decode,
                "host_output_bytes_per_decode": initial.host_output_bytes_per_decode,
                "state_host_transfer_bytes_per_decode": initial.state_host_transfer_bytes_per_decode,
                "command_buffers_per_decode": initial.command_buffers_per_decode,
                "compute_encoders_per_decode": initial.compute_encoders_per_decode,
                "kernel_dispatches_per_decode": initial.kernel_dispatches_per_decode,
                "explicit_buffer_barriers_per_decode": initial.explicit_buffer_barriers_per_decode,
                "gdn_core_seams_per_decode": initial.gdn_core_seams_per_decode,
                "gdn_core_kernel_dispatches_per_decode": initial.gdn_core_kernel_dispatches_per_decode,
                "gdn_core_explicit_buffer_barriers_per_decode": initial.gdn_core_explicit_buffer_barriers_per_decode,
                "gdn_core_recurrent_or_fused_threads_per_threadgroup": initial.gdn_core_recurrent_or_fused_threads_per_threadgroup,
                "gdn_core_threadgroups_per_decode": initial.gdn_core_threadgroups_per_decode,
                "gdn_core_launched_threads_per_decode": initial.gdn_core_launched_threads_per_decode,
                "gdn_core_source_declared_threadgroup_memory_bytes": initial.gdn_core_source_declared_threadgroup_memory_bytes,
                "gdn_core_expected_pipeline_static_threadgroup_memory_bytes": initial.gdn_core_expected_pipeline_static_threadgroup_memory_bytes,
                "gdn_core_internal_threadgroup_barrier_sites_per_threadgroup": initial.gdn_core_internal_threadgroup_barrier_sites_per_threadgroup,
                "commits_per_decode": initial.commits_per_decode,
                "waits_per_decode": initial.waits_per_decode,
                "intermediate_host_finite_checks_per_decode": initial.intermediate_host_finite_checks_per_decode,
                "final_output_finite_checks_per_decode": initial.final_output_finite_checks_per_decode,
            },
        },
        "boundaries": boundaries,
        "tail_layer_index": ledger.tail_layer_index,
        "tail": {
            "scope": tail.scope,
            "exclusions": tail.exclusions,
            "abi_version": tail.abi_version,
            "allocated_buffers": tail.allocated_buffers,
            "shared_buffers": tail.shared_buffers,
            "private_buffers": tail.private_buffers,
            "packed_weight_bytes": tail.packed_weight_bytes,
            "packed_scale_bytes": tail.packed_scale_bytes,
            "f32_parameter_bytes": tail.f32_parameter_bytes,
            "hidden_activation_bytes": tail.hidden_activation_bytes,
            "mlp_activation_bytes": tail.mlp_activation_bytes,
            "partial_topk_bytes": tail.partial_topk_bytes,
            "output_token_bytes": tail.output_token_bytes,
            "total_persistent_bytes": tail.total_persistent_bytes,
            "host_input_bytes_per_decode": tail.host_input_bytes_per_decode,
            "host_output_bytes_per_decode": tail.host_output_bytes_per_decode,
            "state_host_transfer_bytes_per_decode": tail.state_host_transfer_bytes_per_decode,
            "command_buffers_per_decode": tail.command_buffers_per_decode,
            "compute_encoders_per_decode": tail.compute_encoders_per_decode,
            "kernel_dispatches_per_decode": tail.kernel_dispatches_per_decode,
            "buffer_barriers_per_decode": tail.buffer_barriers_per_decode,
            "commits_per_decode": tail.commits_per_decode,
            "waits_per_decode": tail.waits_per_decode,
        },
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
        "component_sum_recomputed_and_exact": aggregate_ledger_is_exact(ledger),
    })
}

fn production_receipt_json_is_exact(receipt: &Value, calls: usize) -> bool {
    if calls == 0 {
        return receipt.is_null();
    }
    receipt.get("profile").and_then(Value::as_str) == Some(FUSED_PROFILE)
        && receipt.get("function_chain").and_then(Value::as_str) == Some(FUSED_FUNCTION_CHAIN)
        && receipt.get("gdn_core_seams").and_then(Value::as_u64) == Some(3)
        && receipt
            .get("persistent_output_groups_per_row")
            .and_then(Value::as_u64)
            == Some(64)
        && receipt
            .get("core_kernel_output_groups_per_row")
            .and_then(Value::as_u64)
            == Some(32)
        && receipt.get("kernel_dispatches").and_then(Value::as_u64) == Some(3)
        && receipt
            .get("explicit_buffer_barriers")
            .and_then(Value::as_u64)
            == Some(3)
        && receipt
            .get("recurrent_or_fused_threads_per_threadgroup")
            .and_then(Value::as_u64)
            == Some(128)
        && receipt.get("threadgroups").and_then(Value::as_u64) == Some(48)
        && receipt.get("launched_threads").and_then(Value::as_u64) == Some(6_144)
        && receipt
            .get("pipeline_thread_execution_width")
            .and_then(Value::as_u64)
            == Some(32)
        && receipt
            .get("source_declared_threadgroup_memory_bytes")
            .and_then(Value::as_u64)
            == Some(2_060)
        && receipt
            .get("pipeline_static_threadgroup_memory_bytes")
            .and_then(Value::as_u64)
            == Some(2_064)
        && receipt
            .get("internal_threadgroup_barrier_sites_per_threadgroup")
            .and_then(Value::as_u64)
            == Some(4)
        && receipt
            .get("fixed_shape_validated")
            .and_then(Value::as_bool)
            == Some(true)
        && receipt.get("rms_norm_eps_bits").and_then(Value::as_u64)
            == Some(1.0e-6_f32.to_bits() as u64)
}

fn generation_receipt_is_exact(
    receipt: &Value,
    calls: usize,
    expected_decode_calls: usize,
    expected_teacher_calls: usize,
) -> bool {
    if expected_decode_calls.checked_add(expected_teacher_calls) != Some(calls)
        || receipt.get("format").and_then(Value::as_str)
            != Some("apxinf-qwen35-mlp-stack3-boundary-tail-head-generation-path-v1")
        || receipt.get("mechanism").and_then(Value::as_str) != Some(FUSED_MECHANISM)
        || receipt.get("gdn_core_profile").and_then(Value::as_str) != Some(FUSED_PROFILE)
        || receipt.get("gdn_function_chain").and_then(Value::as_str) != Some(FUSED_FUNCTION_CHAIN)
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
    let calls_u64 = calls as u64;
    let Some(three_calls) = calls.checked_mul(3).map(|value| value as u64) else {
        return false;
    };
    let Some(four_calls) = calls.checked_mul(4).map(|value| value as u64) else {
        return false;
    };
    let Some(transfer) = calls.checked_mul(4_096).map(|value| value as u64) else {
        return false;
    };
    let Some(top4_transfer) = calls.checked_mul(4_112).map(|value| value as u64) else {
        return false;
    };
    let state_mask = if calls == 0 { 0 } else { 0b111 };
    let Some(initial) = receipt.get("initial_stack") else {
        return false;
    };
    let initial_valid = initial.get("layer_indices") == Some(&json!([0, 1, 2]))
        && initial.get("mechanism").and_then(Value::as_str) == Some(INITIAL_MECHANISM)
        && initial.get("gdn_core_profile").and_then(Value::as_str) == Some(FUSED_PROFILE)
        && initial.get("gdn_function_chain").and_then(Value::as_str) == Some(FUSED_FUNCTION_CHAIN)
        && initial
            .get("kernel_dispatches_per_decode")
            .and_then(Value::as_u64)
            == Some(30)
        && initial
            .get("explicit_buffer_barriers_per_decode")
            .and_then(Value::as_u64)
            == Some(27)
        && production_receipt_json_is_exact(
            initial.get("last_gdn_core_receipt").unwrap_or(&Value::Null),
            calls,
        )
        && initial.get("prefill_seed_calls") == Some(&json!([1, 1, 1]))
        && initial.get("decode_calls").and_then(Value::as_u64) == Some(calls_u64)
        && initial.get("successful_decodes").and_then(Value::as_u64) == Some(calls_u64)
        && initial.get("failed_decodes").and_then(Value::as_u64) == Some(0)
        && initial.get("command_buffers").and_then(Value::as_u64) == Some(calls_u64)
        && initial.get("compute_encoders").and_then(Value::as_u64) == Some(three_calls)
        && initial.get("commits").and_then(Value::as_u64) == Some(calls_u64)
        && initial.get("waits").and_then(Value::as_u64) == Some(calls_u64)
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
            == Some(calls_u64)
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
                    && entry.get("mechanism").and_then(Value::as_str) == Some(BOUNDARY_MECHANISM)
                    && entry.get("gdn_core_profile").and_then(Value::as_str) == Some(FUSED_PROFILE)
                    && entry.get("gdn_function_chain").and_then(Value::as_str)
                        == Some(FUSED_FUNCTION_CHAIN)
                    && entry
                        .get("kernel_dispatches_per_decode")
                        .and_then(Value::as_u64)
                        == Some(35)
                    && entry
                        .get("explicit_buffer_barriers_per_decode")
                        .and_then(Value::as_u64)
                        == Some(31)
                    && production_receipt_json_is_exact(
                        entry.get("last_gdn_core_receipt").unwrap_or(&Value::Null),
                        calls,
                    )
                    && entry.get("prefill_seed_calls") == Some(&json!([1, 1, 1]))
                    && entry.get("decode_calls").and_then(Value::as_u64) == Some(calls_u64)
                    && entry.get("successful_decodes").and_then(Value::as_u64) == Some(calls_u64)
                    && entry.get("failed_decodes").and_then(Value::as_u64) == Some(0)
                    && entry.get("command_buffers").and_then(Value::as_u64) == Some(calls_u64)
                    && entry.get("compute_encoders").and_then(Value::as_u64) == Some(four_calls)
                    && entry.get("commits").and_then(Value::as_u64) == Some(calls_u64)
                    && entry.get("waits").and_then(Value::as_u64) == Some(calls_u64)
                    && entry.get("host_to_device_bytes").and_then(Value::as_u64) == Some(transfer)
                    && entry.get("device_to_host_bytes").and_then(Value::as_u64) == Some(transfer)
                    && entry.get("state_commits").and_then(Value::as_u64) == Some(three_calls)
                    && entry.get("last_state_commit_mask").and_then(Value::as_u64)
                        == Some(state_mask)
                    && entry.get("committed_stack_version").and_then(Value::as_u64)
                        == Some(calls_u64)
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
            == Some(0);
    let Some(tail) = receipt.get("decode_head") else {
        return false;
    };
    let tail_valid = tail.get("mechanism").and_then(Value::as_str) == Some("metal-w8-tail-v1")
        && tail.get("layer_index").and_then(Value::as_u64) == Some(23)
        && tail.get("calls").and_then(Value::as_u64) == Some(expected_decode_calls as u64)
        && tail.get("teacher_calls").and_then(Value::as_u64) == Some(expected_teacher_calls as u64)
        && tail.get("tail_transactions").and_then(Value::as_u64) == Some(calls_u64)
        && tail.get("successful_transactions").and_then(Value::as_u64) == Some(calls_u64)
        && tail.get("failed_transactions").and_then(Value::as_u64) == Some(0)
        && tail.get("command_buffers").and_then(Value::as_u64) == Some(calls_u64)
        && tail.get("compute_encoders").and_then(Value::as_u64) == Some(calls_u64)
        && tail.get("kernel_dispatches").and_then(Value::as_u64)
            == calls.checked_mul(8).map(|value| value as u64)
        && tail.get("commits").and_then(Value::as_u64) == Some(calls_u64)
        && tail.get("waits").and_then(Value::as_u64) == Some(calls_u64)
        && tail.get("host_to_device_bytes").and_then(Value::as_u64) == Some(transfer)
        && tail.get("device_to_host_bytes").and_then(Value::as_u64) == Some(top4_transfer)
        && tail.get("output_commits").and_then(Value::as_u64)
            == calls.checked_mul(2).map(|value| value as u64)
        && tail.get("last_output_commit_mask").and_then(Value::as_u64)
            == Some(if calls == 0 { 0 } else { 0b11 })
        && tail.get("terminal_error").and_then(Value::as_bool) == Some(false);
    let Some(aggregate) = receipt.get("aggregate") else {
        return false;
    };
    let aggregate_valid = aggregate.get("scope").and_then(Value::as_str)
        == Some("resident-mtlbuffer-only")
        && aggregate.get("includes_lm_head").and_then(Value::as_bool) == Some(true)
        && aggregate.get("gdn_core_profile").and_then(Value::as_str) == Some(FUSED_PROFILE)
        && aggregate.get("gdn_function_chain").and_then(Value::as_str)
            == Some(FUSED_FUNCTION_CHAIN)
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
            == Some(213)
        && aggregate
            .get("explicit_buffer_barriers_per_decode")
            .and_then(Value::as_u64)
            == Some(189)
        && aggregate.get("commits_per_decode").and_then(Value::as_u64) == Some(7)
        && aggregate.get("waits_per_decode").and_then(Value::as_u64) == Some(7);
    initial_valid && boundaries_valid && prefill_valid && tail_valid && aggregate_valid
}

fn validate_fused_path_receipts(
    model: &GeneralQwen35,
    calls: usize,
    phase: TailPhase,
) -> Result<Value, Box<dyn Error>> {
    let stats = model
        .metal_w8_mlp_stack3_boundary_tail_head_v1_stats()
        .ok_or("fused-C constructor omitted live execution stats")?;
    let aggregate = model
        .metal_w8_mlp_stack3_boundary_tail_head_v1_aggregate_ledger()
        .ok_or("fused-C constructor omitted resident buffer ledger")?;
    let generation = model
        .generation_path_receipt()
        .ok_or("fused-C constructor omitted generation path receipt")?;
    let triple = calls
        .checked_mul(3)
        .ok_or("fused path call count overflow")?;
    let quadruple = calls
        .checked_mul(4)
        .ok_or("fused path call count overflow")?;
    let h4 = calls
        .checked_mul(4_096)
        .ok_or("fused path byte count overflow")?;
    let h4_top4 = calls
        .checked_mul(4_112)
        .ok_or("fused path byte count overflow")?;
    let state_mask = if calls == 0 { 0 } else { 0b111 };
    let initial = stats.initial_stack.execution;
    let schedule_valid = stats.initial_stack.layer_indices == [0, 1, 2]
        && stats
            .boundaries
            .iter()
            .map(|entry| (entry.boundary_mlp_layer_index, entry.stack_layer_indices))
            .eq(BOUNDARY_REGIONS)
        && stats.tail_layer_index == 23;
    let quantization_valid = stats
        .initial_stack
        .quantization
        .iter()
        .chain(
            stats
                .boundaries
                .iter()
                .flat_map(|region| region.quantization.iter()),
        )
        .all(|entry| {
            entry.gdn_input_group_size == apxinf_metal::W8GroupSize::G64
                && entry.gdn_output_group_size == apxinf_metal::W8GroupSize::G32
                && entry.mlp_gate_group_size == apxinf_metal::W8GroupSize::G64
                && entry.mlp_up_group_size == apxinf_metal::W8GroupSize::G64
                && entry.mlp_down_group_size == apxinf_metal::W8GroupSize::G64
        });
    let mechanism_and_precision_valid = stats.mechanism == FUSED_MECHANISM
        && stats.gdn_core_profile == apxinf_metal::GdnCoreProfileV1::Fused128
        && stats.gdn_function_chain == FUSED_FUNCTION_CHAIN
        && stats.initial_stack.mechanism == INITIAL_MECHANISM
        && stats.initial_stack.kernel_dispatches_per_decode == 30
        && stats.initial_stack.explicit_buffer_barriers_per_decode == 27
        && production_receipt_is_exact(stats.initial_stack.last_gdn_core_receipt, calls)
        && stats.boundaries.iter().all(|region| {
            region.mechanism == BOUNDARY_MECHANISM
                && region.gdn_core_profile == apxinf_metal::GdnCoreProfileV1::Fused128
                && region.gdn_function_chain == FUSED_FUNCTION_CHAIN
                && region.kernel_dispatches_per_decode == 35
                && region.explicit_buffer_barriers_per_decode == 31
                && production_receipt_is_exact(region.last_gdn_core_receipt, calls)
        })
        && quantization_valid;
    let initial_execution_valid = stats.initial_stack.prefill_seed_calls == [1, 1, 1]
        && initial.decode_calls == calls
        && initial.successful_decodes == calls
        && initial.failed_decodes == 0
        && initial.command_buffers == calls
        && initial.compute_encoders == triple
        && initial.commits == calls
        && initial.waits == calls
        && initial.host_to_device_bytes == h4
        && initial.device_to_host_bytes == h4
        && initial.state_commits == triple
        && initial.last_state_commit_mask == state_mask
        && initial.committed_stack_version == calls as u64
        && !initial.terminal_error
        && !stats.initial_stack.terminal_error;
    let boundaries_execution_valid = stats.boundaries.iter().all(|region| {
        let execution = region.execution;
        region.prefill_seed_calls == [1, 1, 1]
            && execution.decode_calls == calls
            && execution.successful_decodes == calls
            && execution.failed_decodes == 0
            && execution.command_buffers == calls
            && execution.compute_encoders == quadruple
            && execution.commits == calls
            && execution.waits == calls
            && execution.host_to_device_bytes == h4
            && execution.device_to_host_bytes == h4
            && execution.state_commits == triple
            && execution.last_state_commit_mask == state_mask
            && execution.committed_stack_version == calls as u64
            && !execution.terminal_error
            && !region.terminal_error
    });
    let (expected_decode_calls, expected_teacher_calls) = match phase {
        TailPhase::Teacher => (0, calls),
        TailPhase::Free => (calls, 0),
    };
    let tail = stats.tail;
    let tail_execution_and_phase_valid = stats.prefill_body_calls == 1
        && stats.prefill_cpu_head_calls == 1
        && stats.decode_calls == expected_decode_calls
        && stats.teacher_calls == expected_teacher_calls
        && tail.decode_calls == calls
        && tail.successful_decodes == calls
        && tail.failed_decodes == 0
        && tail.host_to_device_bytes == h4
        && tail.device_to_host_bytes == h4_top4
        && tail.command_buffers == calls
        && tail.compute_encoders == calls
        && tail.kernel_dispatches == calls.checked_mul(8).ok_or("tail count overflow")?
        && tail.buffer_barriers == calls.checked_mul(7).ok_or("tail count overflow")?
        && tail.commits == calls
        && tail.waits == calls
        && tail.output_commits == calls.checked_mul(2).ok_or("tail count overflow")?
        && tail.last_output_commit_mask == if calls == 0 { 0 } else { 0b11 }
        && !tail.terminal_error;
    let aggregate_ledger_valid = aggregate_ledger_is_exact(&aggregate);
    let generation_receipt_valid = generation_receipt_is_exact(
        &generation,
        calls,
        expected_decode_calls,
        expected_teacher_calls,
    );
    let terminal_clear = !stats.terminal_error && !tail.terminal_error;
    let all_valid = schedule_valid
        && mechanism_and_precision_valid
        && initial_execution_valid
        && boundaries_execution_valid
        && tail_execution_and_phase_valid
        && aggregate_ledger_valid
        && generation_receipt_valid
        && terminal_clear;
    let path_checks = json!({
        "schedule_valid": schedule_valid,
        "mechanism_and_precision_valid": mechanism_and_precision_valid,
        "six_region_execution_valid": initial_execution_valid && boundaries_execution_valid,
        "tail_execution_and_phase_valid": tail_execution_and_phase_valid,
        "aggregate_ledger_valid": aggregate_ledger_valid,
        "generation_receipt_valid": generation_receipt_valid,
        "terminal_clear": terminal_clear,
        "all_valid": all_valid,
    });
    if !all_valid {
        return Err(format!("fused-C live path receipt failed: {path_checks}").into());
    }
    Ok(json!({
        "generation_path_receipt": generation,
        "aggregate_buffer_ledger": aggregate_buffer_ledger_json(&aggregate),
        "path_checks": path_checks,
    }))
}

fn run_an_teacher_observed(
    config: Qwen35Config,
    tensors: HashMap<String, Tensor>,
) -> Result<Value, Box<dyn Error>> {
    let mut model =
        GeneralQwen35::from_weights_with_metal_w8_mlp_stack3_boundary_tail_head_gdn_core_fused_v1(
            config,
            tensors,
            Device::Cpu,
            MAX_CONTEXT,
        )?;
    let _prefill = model
        .prefill_for_generation(LlmInput::text(&PROMPT_TOKEN_IDS[..TEACHER_PREFILL_TOKENS]))?;
    let prefill_path = validate_fused_path_receipts(&model, 0, TailPhase::Teacher)?;
    let mut cpu_f32_argmax_token_ids = Vec::with_capacity(STEPS);
    let mut top4_candidate_token_ids = Vec::with_capacity(STEPS);
    let mut observed_argmax_token_ids = Vec::with_capacity(STEPS);
    let mut accelerator_candidate_elapsed_ns = Vec::with_capacity(STEPS);
    let mut f32_rerank_elapsed_ns = Vec::with_capacity(STEPS);
    let mut next_greedy_token_ready_elapsed_ns = Vec::with_capacity(STEPS);
    let started = Instant::now();
    for (step, &teacher_token) in TEACHER_INPUT_TOKEN_IDS.iter().enumerate() {
        let position = u32::try_from(TEACHER_PREFILL_TOKENS + step)?;
        let comparison = model.teacher_forced_decode_candidates(teacher_token, position)?;
        let token_ready_elapsed_ns = u64::try_from(started.elapsed().as_nanos())?;
        next_greedy_token_ready_elapsed_ns.push(token_ready_elapsed_ns);
        cpu_f32_argmax_token_ids.push(comparison.cpu_token);
        top4_candidate_token_ids.push(comparison.w8_candidates);
        observed_argmax_token_ids.push(comparison.reranked_token);
        accelerator_candidate_elapsed_ns.push(comparison.accelerator_candidate_elapsed_ns());
        f32_rerank_elapsed_ns.push(comparison.rerank_elapsed_ns);
    }
    if cpu_f32_argmax_token_ids != CANONICAL_FREE_TOKEN_IDS
        || observed_argmax_token_ids != CANONICAL_FREE_TOKEN_IDS
    {
        return Err("AN teacher argmax diverged from frozen CPU/F32 free128".into());
    }
    for (step, candidates) in top4_candidate_token_ids.iter().enumerate() {
        let expected = CANONICAL_FREE_TOKEN_IDS[step];
        let unique = candidates
            .iter()
            .copied()
            .collect::<std::collections::BTreeSet<_>>()
            .len()
            == candidates.len();
        if !unique || !candidates.contains(&expected) {
            return Err(
                format!("AN teacher top4 failed exact winner custody at step {step}").into(),
            );
        }
    }
    let final_path = validate_fused_path_receipts(&model, STEPS, TailPhase::Teacher)?;
    Ok(json!({
        "format": "apxinf-qwen35-native-teacher-runtime-receipt-v3",
        "schema_version": 3,
        "mode": "native-v3-teacher",
        "teacher_role": "observed",
        "prefill_prompt_token_ids": &PROMPT_TOKEN_IDS[..TEACHER_PREFILL_TOKENS],
        "prefill_prompt_token_ids_count": TEACHER_PREFILL_TOKENS,
        "teacher_input_token_ids": &TEACHER_INPUT_TOKEN_IDS[..],
        "reference_argmax_token_ids": &CANONICAL_FREE_TOKEN_IDS[..],
        "tail_normalized_hidden_f32_argmax_token_ids": cpu_f32_argmax_token_ids,
        "tail_top4_candidate_token_ids": top4_candidate_token_ids,
        "observed_argmax_token_ids": observed_argmax_token_ids,
        "mismatch_positions": [],
        "first_mismatch": null,
        "next_greedy_token_ready_elapsed_ns": next_greedy_token_ready_elapsed_ns,
        "accelerator_candidate_elapsed_ns": accelerator_candidate_elapsed_ns,
        "f32_tied_rerank_elapsed_ns": f32_rerank_elapsed_ns,
        "selection_work_included": true,
        "accelerator_completion_before_each_token_ready_timestamp": true,
        "eog_termination": false,
        "prefill_path": prefill_path,
        "final_path": final_path,
        "passed": true,
    }))
}

#[cfg(target_os = "macos")]
#[repr(C)]
struct Timespec {
    tv_sec: i64,
    tv_nsec: i64,
}

#[cfg(target_os = "macos")]
unsafe extern "C" {
    fn clock_gettime(clock_id: u32, time: *mut Timespec) -> std::os::raw::c_int;
    fn clock_getres(clock_id: u32, resolution: *mut Timespec) -> std::os::raw::c_int;
}

#[cfg(target_os = "macos")]
const CLOCK_MONOTONIC_RAW: u32 = 4;

#[cfg(target_os = "macos")]
fn timespec_ns(value: Timespec, label: &str) -> Result<u64, Box<dyn Error>> {
    if value.tv_sec < 0 || !(0..1_000_000_000).contains(&value.tv_nsec) {
        return Err(format!("{label} returned an invalid timespec").into());
    }
    let total = (value.tv_sec as u128)
        .checked_mul(1_000_000_000)
        .and_then(|seconds| seconds.checked_add(value.tv_nsec as u128))
        .ok_or_else(|| format!("{label} nanosecond value overflow"))?;
    Ok(u64::try_from(total)?)
}

#[cfg(target_os = "macos")]
fn monotonic_raw_ns() -> Result<u64, Box<dyn Error>> {
    let mut value = Timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    let status = unsafe { clock_gettime(CLOCK_MONOTONIC_RAW, &mut value) };
    if status != 0 {
        return Err(std::io::Error::last_os_error().into());
    }
    timespec_ns(value, "clock_gettime(CLOCK_MONOTONIC_RAW)")
}

#[cfg(target_os = "macos")]
fn monotonic_raw_resolution_ns() -> Result<u64, Box<dyn Error>> {
    let mut value = Timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    let status = unsafe { clock_getres(CLOCK_MONOTONIC_RAW, &mut value) };
    if status != 0 {
        return Err(std::io::Error::last_os_error().into());
    }
    let resolution = timespec_ns(value, "clock_getres(CLOCK_MONOTONIC_RAW)")?;
    if resolution == 0 {
        return Err("CLOCK_MONOTONIC_RAW reported zero resolution".into());
    }
    Ok(resolution)
}

#[cfg(not(target_os = "macos"))]
fn monotonic_raw_ns() -> Result<u64, Box<dyn Error>> {
    Err("native v3 runner requires macOS CLOCK_MONOTONIC_RAW".into())
}

#[cfg(not(target_os = "macos"))]
fn monotonic_raw_resolution_ns() -> Result<u64, Box<dyn Error>> {
    Err("native v3 runner requires macOS CLOCK_MONOTONIC_RAW".into())
}

fn run_an_free(
    config: Qwen35Config,
    tensors: HashMap<String, Tensor>,
) -> Result<Value, Box<dyn Error>> {
    let vocab_size = config.text.vocab_size;
    let mut model =
        GeneralQwen35::from_weights_with_metal_w8_mlp_stack3_boundary_tail_head_gdn_core_fused_v1(
            config,
            tensors,
            Device::Cpu,
            MAX_CONTEXT,
        )?;
    let clock_resolution_ns = monotonic_raw_resolution_ns()?;
    let mut token_ready_ns = Vec::with_capacity(STEPS);
    let mut generated_token_ids = Vec::with_capacity(STEPS);
    let prefill_start_ns = monotonic_raw_ns()?;
    let prefill_logits = model.prefill_for_generation(LlmInput::text(&PROMPT_TOKEN_IDS))?;
    let first_token = argmax(&prefill_logits, vocab_size)?;
    let first_token_ready_ns = monotonic_raw_ns()?;
    token_ready_ns.push(first_token_ready_ns);
    generated_token_ids.push(first_token);
    let mut previous_token = first_token;
    for step in 1..STEPS {
        let position = u32::try_from(PROMPT_TOKEN_IDS.len() + step - 1)?;
        previous_token = model
            .decode_token(previous_token, position)
            .ok_or("fused-C constructor did not claim decode_token")??;
        let token_ready = monotonic_raw_ns()?;
        token_ready_ns.push(token_ready);
        generated_token_ids.push(previous_token);
    }
    if generated_token_ids != CANONICAL_FREE_TOKEN_IDS {
        return Err("AN free trajectory diverged from frozen free128".into());
    }
    if token_ready_ns.len() != STEPS
        || !(prefill_start_ns < token_ready_ns[0]
            && token_ready_ns.windows(2).all(|pair| pair[0] < pair[1]))
    {
        return Err("next-greedy-token-ready timestamps are not strictly increasing".into());
    }
    let token_1_ready_ns = token_ready_ns[0];
    let token_128_ready_ns = token_ready_ns[STEPS - 1];
    let ttft_ns = token_1_ready_ns - prefill_start_ns;
    let total_ns = token_128_ready_ns - prefill_start_ns;
    let steady_ns = token_128_ready_ns - token_1_ready_ns;
    let final_path = validate_fused_path_receipts(&model, STEPS - 1, TailPhase::Free)?;
    let token_ready_elapsed_ns = token_ready_ns
        .iter()
        .map(|timestamp| timestamp - prefill_start_ns)
        .collect::<Vec<_>>();
    Ok(json!({
        "format": "apxinf-qwen35-native-free-sample-receipt-v3",
        "schema_version": 3,
        "mode": "native-v3-free",
        "workload": {
            "ingress_semantics": "raw-token-ids",
            "prompt_token_ids": PROMPT_TOKEN_IDS,
            "prefill_token_count": PROMPT_TOKEN_IDS.len(),
            "generated_token_ids": generated_token_ids,
            "generated_token_count": STEPS,
            "sampling": "unbiased-greedy-argmax",
            "temperature": 0,
            "eog_policy": "select-and-feed-back-eog-without-termination-and-without-eog-logit-suppression",
            "speculative_decoding": false,
            "continuous_batching": false,
            "sequence_count": 1,
            "requested_context_tokens": MAX_CONTEXT,
            "effective_context_tokens": MAX_CONTEXT,
            "requested_batch_tokens": PROMPT_TOKEN_IDS.len(),
            "effective_batch_tokens": PROMPT_TOKEN_IDS.len(),
            "requested_ubatch_tokens": PROMPT_TOKEN_IDS.len(),
            "effective_ubatch_tokens": PROMPT_TOKEN_IDS.len(),
            "empty_state_before_prefill": true,
            "prompt_cache_reused": false,
        },
        "timing": {
            "clock": "monotonic",
            "clock_identity": "Darwin CLOCK_MONOTONIC_RAW",
            "clock_resolution_ns": clock_resolution_ns,
            "start_boundary": "immediately-before-first-raw-token-prefill-dispatch",
            "common_token_ready_boundary": "next-greedy-token-ready",
            "end_boundary": "128th-next-greedy-token-ready",
            "selection_work_included": true,
            "accelerator_completion_before_each_token_ready_timestamp": true,
            "final_sampled_token_decoded_inside_timed_region": false,
            "prefill_start_ns": prefill_start_ns,
            "token_1_ready_ns": token_1_ready_ns,
            "token_128_ready_ns": token_128_ready_ns,
            "token_ready_ns": token_ready_ns,
            "next_greedy_token_ready_elapsed_ns": token_ready_elapsed_ns,
            "ttft_ms": ttft_ns as f64 / 1_000_000.0,
            "total_latency_ms": total_ns as f64 / 1_000_000.0,
            "tpot_ms": steady_ns as f64 / 127.0 / 1_000_000.0,
            "generation_tps": 127_000_000_000.0 / steady_ns as f64,
        },
        "final_path": final_path,
        "passed": true,
    }))
}

fn is_lower_hex(value: &str, length: usize) -> bool {
    value.len() == length
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn sha256_json(value: &Value) -> Result<String, Box<dyn Error>> {
    Ok(format!("{:x}", Sha256::digest(serde_json::to_vec(value)?)))
}

fn required_clone(value: &Value, pointer: &str) -> Result<Value, Box<dyn Error>> {
    value
        .pointer(pointer)
        .cloned()
        .ok_or_else(|| format!("custody receipt omitted {pointer}").into())
}

fn formal_file_identity(custody: &Value, pointer: &str) -> Result<Value, Box<dyn Error>> {
    let file = custody
        .pointer(pointer)
        .and_then(Value::as_object)
        .ok_or_else(|| format!("custody receipt omitted file {pointer}"))?;
    let path = file
        .get("path")
        .and_then(Value::as_str)
        .ok_or_else(|| format!("custody file {pointer} omitted path"))?;
    let size = file
        .get("size")
        .and_then(Value::as_u64)
        .ok_or_else(|| format!("custody file {pointer} omitted size"))?;
    let sha256 = file
        .get("sha256")
        .and_then(Value::as_str)
        .filter(|hash| is_lower_hex(hash, 64))
        .ok_or_else(|| format!("custody file {pointer} omitted SHA-256"))?;
    Ok(json!({
        "absolute_path": path,
        "size_bytes": size,
        "sha256": sha256,
    }))
}

#[cfg(target_os = "macos")]
unsafe extern "C" {
    fn _dyld_image_count() -> u32;
    fn _dyld_get_image_name(image_index: u32) -> *const std::os::raw::c_char;
}

fn is_system_library(path: &str) -> bool {
    path.starts_with("/System/")
        || path.starts_with("/usr/lib/")
        || path.starts_with("/Library/Apple/System/")
}

#[cfg(target_os = "macos")]
fn loaded_non_system_library_closure() -> Result<Value, Box<dyn Error>> {
    let current_exe = std::fs::canonicalize(std::env::current_exe()?)?;
    let count = unsafe { _dyld_image_count() };
    let mut paths = std::collections::BTreeSet::new();
    for image_index in 0..count {
        let raw = unsafe { _dyld_get_image_name(image_index) };
        if raw.is_null() {
            return Err(format!("dyld image {image_index} omitted its path").into());
        }
        let path = unsafe { CStr::from_ptr(raw) }
            .to_str()
            .map_err(|_| format!("dyld image {image_index} path is not UTF-8"))?;
        if is_system_library(path) {
            continue;
        }
        let canonical = std::fs::canonicalize(path)?;
        if canonical != current_exe {
            paths.insert(canonical);
        }
    }
    let entries = paths
        .iter()
        .map(|path| {
            let attestation = gate_evidence::attest_file(path, "loaded non-system library", None)?;
            Ok(json!({
                "absolute_path": attestation.path,
                "size_bytes": attestation.size,
                "sha256": attestation.sha256,
                "device": attestation.device,
                "inode": attestation.inode,
                "change_time_seconds": attestation.change_time_seconds,
                "change_time_nanoseconds": attestation.change_time_nanoseconds,
            }))
        })
        .collect::<Result<Vec<_>, Box<dyn Error>>>()?;
    Ok(Value::Array(entries))
}

#[cfg(not(target_os = "macos"))]
fn loaded_non_system_library_closure() -> Result<Value, Box<dyn Error>> {
    Err("native v3 runner requires macOS dyld custody".into())
}

fn an_deployment_receipt() -> Value {
    json!({
        "constructor_id": "from_weights_with_metal_w8_mlp_stack3_boundary_tail_head_gdn_core_fused_v1",
        "profile_id": FUSED_MECHANISM,
        "context_capacity_tokens": MAX_CONTEXT,
        "prefill_device": "CPU",
        "prefill_precision": "F32",
        "full_attention_device": "CPU",
        "full_attention_precision": "F32",
        "kv_key_dtype": "F32",
        "kv_value_dtype": "F32",
        "head": "F32 tied embedding top-4 exact rerank",
        "metal_build_input_count": 17,
        "exact_live_execution_ledger": true,
        "thread_policy": {
            "policy": "Accelerate OS-managed default",
            "fixed_worker_count_claimed": false,
            "VECLIB_MAXIMUM_THREADS_present": false,
            "OMP_NUM_THREADS_present": false,
            "OPENBLAS_NUM_THREADS_present": false,
            "MKL_NUM_THREADS_present": false,
        },
    })
}

fn cpu_reference_deployment_receipt() -> Value {
    json!({
        "constructor_id": "from_weights",
        "context_capacity_tokens": MAX_CONTEXT,
        "prefill_device": "CPU",
        "prefill_precision": "F32",
        "full_attention_device": "CPU",
        "full_attention_precision": "F32",
        "kv_key_dtype": "F32",
        "kv_value_dtype": "F32",
        "head": "CPU/F32 full-vocabulary tied argmax",
        "teacher_reference_only": true,
    })
}

fn formal_request_schedule_is_valid(object: &serde_json::Map<String, Value>) -> bool {
    let Some(sequence_index) = object
        .get("sequence_index")
        .and_then(Value::as_u64)
        .and_then(|value| usize::try_from(value).ok())
    else {
        return false;
    };
    match object.get("phase").and_then(Value::as_str) {
        Some("warmup") => {
            let Some(warmup_index) = object
                .get("warmup_index")
                .and_then(Value::as_u64)
                .and_then(|value| usize::try_from(value).ok())
            else {
                return false;
            };
            sequence_index == warmup_index
                && matches!(warmup_index, 0 | 3 | 4)
                && object.get("block_index") == Some(&Value::Null)
                && object.get("slot_index") == Some(&Value::Null)
        }
        Some("timed") => {
            let Some(block_index) = object
                .get("block_index")
                .and_then(Value::as_u64)
                .and_then(|value| usize::try_from(value).ok())
            else {
                return false;
            };
            let Some(slot_index) = object
                .get("slot_index")
                .and_then(Value::as_u64)
                .and_then(|value| usize::try_from(value).ok())
            else {
                return false;
            };
            let slot_is_an = if block_index % 2 == 0 {
                matches!(slot_index, 0 | 3)
            } else {
                matches!(slot_index, 1 | 2)
            };
            block_index < 8
                && slot_index < 4
                && slot_is_an
                && sequence_index == 6 + block_index * 4 + slot_index
                && object.get("warmup_index") == Some(&Value::Null)
        }
        _ => false,
    }
}

fn parse_formal_request() -> Result<Value, Box<dyn Error>> {
    let raw = std::env::var("APXINF_FORMAL_V3_REQUEST_JSON")
        .map_err(|_| "APXINF_FORMAL_V3_REQUEST_JSON is required for native-v3-free")?;
    let request: Value = serde_json::from_str(&raw)?;
    let object = request
        .as_object()
        .ok_or("formal v3 request must be a JSON object")?;
    let expected_fields = std::collections::BTreeSet::from([
        "nonce",
        "sequence_index",
        "phase",
        "warmup_index",
        "block_index",
        "slot_index",
        "role",
        "arm",
    ]);
    if object
        .keys()
        .map(String::as_str)
        .collect::<std::collections::BTreeSet<_>>()
        != expected_fields
    {
        return Err("formal v3 request fields differ from the frozen schedule binding".into());
    }
    let nonce = object
        .get("nonce")
        .and_then(Value::as_str)
        .ok_or("formal v3 request nonce is absent")?;
    if !is_lower_hex(nonce, 64)
        || object.get("arm").and_then(Value::as_str) != Some("AN")
        || object.get("role").and_then(Value::as_str) != Some("A")
    {
        return Err("formal v3 request is not bound to native arm AN/A".into());
    }
    if !formal_request_schedule_is_valid(object) {
        return Err("formal v3 request schedule tuple is invalid".into());
    }
    Ok(request)
}

fn capture_thread_policy_runtime() -> Result<Value, Box<dyn Error>> {
    for variable in THREAD_OVERRIDE_ENVIRONMENT {
        if std::env::var_os(variable).is_some() {
            return Err(format!(
                "{variable} must be absent for the Accelerate OS-managed thread policy"
            )
            .into());
        }
    }
    Ok(json!({
        "logical_cpu_count": std::thread::available_parallelism()?.get(),
        "logical_cpu_count_source": "std::thread::available_parallelism",
        "fixed_worker_count_claimed": false,
        "environment_overrides_absent": true,
        "absent_environment_overrides": THREAD_OVERRIDE_ENVIRONMENT,
    }))
}

fn packed_manifest(receipt: &Value) -> Option<&Value> {
    receipt
        .pointer("/final_path/aggregate_buffer_ledger")
        .or_else(|| receipt.pointer("/prefill_path/aggregate_buffer_ledger"))
}

fn real_main() -> Result<Value, Box<dyn Error>> {
    if cfg!(debug_assertions) {
        return Err("native v3 runner must be built with --release".into());
    }
    if !cfg!(target_os = "macos") {
        return Err("native v3 runner requires macOS".into());
    }
    if std::env::var_os("APXINF_PERF").is_some() {
        return Err("APXINF_PERF must be unset for the formal v3 runner".into());
    }
    let args = parse_args_from(std::env::args_os())?;
    let formal_request = if args.mode == RunMode::NativeV3Free {
        Some(parse_formal_request()?)
    } else {
        None
    };
    let thread_policy_runtime = capture_thread_policy_runtime()?;
    let runtime_source_commit = EMBEDDED_CANDIDATE_COMMIT
        .filter(|commit| is_lower_hex(commit, 40))
        .ok_or("release runner omitted a valid APXINF_CANDIDATE_COMMIT")?;
    let library_closure_start = loaded_non_system_library_closure()?;
    let custody = gate_evidence::GateCustody::capture_boundary_tail_head_v1(
        &args.model_dir,
        &args.source_lock,
        RUNNER_SOURCE_NAME,
        RUNNER_SOURCE_BYTES,
    )?;
    let source_custody_start = custody.receipt_json();
    let (config, tensors) = load_model_inputs(&custody)?;
    let mut receipt = match (args.mode, args.teacher_role) {
        (RunMode::NativeV3Free, None) => run_an_free(config, tensors)?,
        (RunMode::NativeV3Teacher, Some(TeacherRole::Reference)) => {
            run_cpu_teacher_reference(config, tensors)?
        }
        (RunMode::NativeV3Teacher, Some(TeacherRole::Observed)) => {
            run_an_teacher_observed(config, tensors)?
        }
        _ => return Err("native v3 mode/teacher role combination is impossible".into()),
    };
    let source_custody_end = custody.verify_unchanged_receipt()?;
    let library_closure_end = loaded_non_system_library_closure()?;
    let library_closure_start_sha256 = sha256_json(&library_closure_start)?;
    let library_closure_end_sha256 = sha256_json(&library_closure_end)?;
    if library_closure_start != library_closure_end
        || library_closure_start_sha256 != library_closure_end_sha256
    {
        return Err("loaded non-system library closure changed during the sample".into());
    }
    let packed_weight_and_resident_buffer_manifest_sha256 =
        packed_manifest(&receipt).map(sha256_json).transpose()?;
    let deployment = if args.teacher_role == Some(TeacherRole::Reference) {
        cpu_reference_deployment_receipt()
    } else {
        an_deployment_receipt()
    };
    let configuration_id = if args.teacher_role == Some(TeacherRole::Reference) {
        "ApxInf-native-CPU-F32-teacher-reference-v3"
    } else {
        AN_CONFIGURATION_ID
    };
    let custody_receipt = json!({
        "configuration_id": configuration_id,
        "runner": formal_file_identity(&source_custody_start, "/binary")?,
        "model": formal_file_identity(
            &source_custody_start,
            "/model_dir/artifacts/model.safetensors-00001-of-00001.safetensors",
        )?,
        "runtime_source_commit": runtime_source_commit,
        "loaded_non_system_library_closure_sha256": library_closure_end_sha256.clone(),
        "loaded_non_system_library_closure_start_sha256": library_closure_start_sha256,
        "loaded_non_system_library_closure_end_sha256": library_closure_end_sha256,
        "packed_weight_and_resident_buffer_manifest_sha256": packed_weight_and_resident_buffer_manifest_sha256,
        "deployment": deployment,
        "thread_policy_runtime": thread_policy_runtime,
        "fresh_process": true,
        "start_end_identity_equal": true,
        "loaded_non_system_library_closure": library_closure_end.clone(),
        "loaded_non_system_library_closure_start": library_closure_start,
        "loaded_non_system_library_closure_end": library_closure_end,
        "source_custody_start": source_custody_start,
        "source_custody_end": source_custody_end,
    });
    receipt["campaign_id"] = json!(CAMPAIGN_ID);
    receipt["subcampaign_id"] = json!(SUBCAMPAIGN_ID);
    receipt["edge_id"] = json!(EDGE_ID);
    if args.mode == RunMode::NativeV3Free {
        let request = formal_request.ok_or("native v3 free request binding vanished")?;
        let generated = required_clone(&receipt, "/workload/generated_token_ids")?;
        receipt["workload"]["generated_token_ids_sha256"] = json!(sha256_json(&generated)?);
        receipt["request"] = request;
    } else {
        receipt["arm"] = json!(if args.teacher_role == Some(TeacherRole::Observed) {
            "AN"
        } else {
            "CPU_REFERENCE"
        });
        receipt["teacher_input_token_ids_sha256"] =
            json!("c33b70a7626fbf3aaa9b8b09e03ce55b5d0e9a1b6ba7068d29067ccb6209a70d");
    }
    receipt["custody"] = custody_receipt;
    Ok(receipt)
}

fn parse_args_from<I>(values: I) -> Result<Args, Box<dyn Error>>
where
    I: IntoIterator<Item = OsString>,
{
    let mut values = values.into_iter();
    let _program = values.next().ok_or("argv omitted program name")?;
    let mut mode = None;
    let mut teacher_role = None;
    let mut model_dir = None;
    let mut source_lock = None;
    while let Some(argument) = values.next() {
        let argument = argument
            .to_str()
            .ok_or("native v3 arguments must be valid UTF-8")?;
        let value = values
            .next()
            .ok_or_else(|| format!("missing value for {argument}"))?;
        match argument {
            "--mode" => {
                if mode.is_some() {
                    return Err("--mode may be specified at most once".into());
                }
                mode = Some(RunMode::parse(
                    value.to_str().ok_or("--mode must be valid UTF-8")?,
                )?);
            }
            "--teacher-role" => {
                if teacher_role.is_some() {
                    return Err("--teacher-role may be specified at most once".into());
                }
                teacher_role = Some(TeacherRole::parse(
                    value.to_str().ok_or("--teacher-role must be valid UTF-8")?,
                )?);
            }
            "--model-dir" => {
                if model_dir.replace(PathBuf::from(value)).is_some() {
                    return Err("--model-dir may be specified at most once".into());
                }
            }
            "--source-lock" => {
                if source_lock.replace(PathBuf::from(value)).is_some() {
                    return Err("--source-lock may be specified at most once".into());
                }
            }
            _ => return Err(format!("unknown native v3 argument: {argument}").into()),
        }
    }
    let mode = mode.ok_or("--mode is required")?;
    match (mode, teacher_role) {
        (RunMode::NativeV3Teacher, None) => {
            return Err("teacher role is required for native-v3-teacher".into())
        }
        (RunMode::NativeV3Free, Some(_)) => {
            return Err("teacher role is invalid for native-v3-free".into())
        }
        _ => {}
    }
    Ok(Args {
        mode,
        teacher_role,
        model_dir: model_dir.ok_or("--model-dir is required")?,
        source_lock: source_lock.ok_or("--source-lock is required")?,
    })
}

fn main() {
    let (receipt, mut exit_code) = match real_main() {
        Ok(receipt) => (receipt, 0),
        Err(error) => (
            json!({
                "format": "apxinf-qwen35-native-runner-failure-v3",
                "schema_version": 3,
                "passed": false,
                "error": error.to_string(),
            }),
            2,
        ),
    };
    let mut line = match serde_json::to_vec(&receipt) {
        Ok(line) => line,
        Err(error) => {
            exit_code = 2;
            serde_json::to_vec(&json!({
                "format": "apxinf-qwen35-native-runner-failure-v3",
                "schema_version": 3,
                "passed": false,
                "error": format!("receipt serialization failed: {error}"),
            }))
            .expect("the fixed failure receipt is serializable")
        }
    };
    line.push(b'\n');
    let mut stdout = std::io::stdout().lock();
    if stdout.write_all(&line).is_err() || stdout.flush().is_err() {
        std::process::exit(3);
    }
    std::process::exit(exit_code);
}
