//! Same-release-binary real-checkpoint quality gate for one diagnostic Metal
//! W8 complete linear-attention layer. This example is deliberately absent from CLI/AutoModel and
//! all default construction paths.

#[path = "support/qwen35_gate_evidence.rs"]
mod gate_evidence;

use std::error::Error;
use std::ffi::OsString;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

use apxinf_core::{Device, Tensor};
use apxinf_model::qwen35::general::{
    Qwen35MetalW8AllLinearLayersPrecisionV2AggregateLedger,
    Qwen35MetalW8AllLinearLayersPrecisionV2Stats, Qwen35MetalW8LinearLayerPrecisionProfile,
    Qwen35MetalW8LinearLayerPrecisionV2Stats, Qwen35MetalW8LinearLayerStacksV1AggregateLedger,
    Qwen35MetalW8LinearLayerStacksV1Stats, Qwen35MetalW8LinearLayerStats,
};
#[cfg(test)]
use apxinf_model::qwen35::general::{
    Qwen35MetalW8LinearLayerBufferLedger, Qwen35MetalW8LinearLayerStack3BufferLedger,
    Qwen35MetalW8LinearLayerStack3V1Stats, Qwen35MetalW8MlpBlockBufferLedger,
};
use apxinf_model::{GeneralQwen35, LlmInput, LlmTrait, Qwen35Config, Qwen35LayerType};
use apxinf_tokenizer::{ChatMessage, Tokenizer};
use serde_json::{json, Value};

const CPU_TEACHER_FORMAT: &str = "apxinf-qwen35-metal-w8-linear-layer-cpu-teacher-v1";
const LINEAR_LAYER_TEACHER_FORMAT: &str = "apxinf-qwen35-metal-w8-linear-layer-teacher-gate-v1";
const LINEAR_LAYER_GDN_OUT_G32_V2_TEACHER_FORMAT: &str =
    "apxinf-qwen35-metal-w8-linear-layer-gdn-out-g32-v2-teacher-gate-v1";
const CPU_FREE_FORMAT: &str = "apxinf-qwen35-metal-w8-linear-layer-cpu-free-run-v1";
const LINEAR_LAYER_FREE_FORMAT: &str = "apxinf-qwen35-metal-w8-linear-layer-free-run-gate-v1";
const LINEAR_LAYER_GDN_OUT_G32_V2_FREE_FORMAT: &str =
    "apxinf-qwen35-metal-w8-linear-layer-gdn-out-g32-v2-free-run-gate-v1";
const ALL_LINEAR_LAYERS_GDN_OUT_G32_V2_TEACHER_FORMAT: &str =
    "apxinf-qwen35-metal-w8-all-linear-layers-gdn-out-g32-v2-teacher-gate-v1";
const ALL_LINEAR_LAYERS_GDN_OUT_G32_V2_FREE_FORMAT: &str =
    "apxinf-qwen35-metal-w8-all-linear-layers-gdn-out-g32-v2-free-run-gate-v1";
const ALL_LINEAR_LAYER_STACKS_V1_TEACHER_FORMAT: &str =
    "apxinf-qwen35-metal-w8-all-linear-layer-stacks-v1-teacher-gate-v1";
const ALL_LINEAR_LAYER_STACKS_V1_FREE_FORMAT: &str =
    "apxinf-qwen35-metal-w8-all-linear-layer-stacks-v1-free-run-gate-v1";
const SOURCE_LOCK_FORMAT: &str = "apxinf-hf-source-lock-v1";
const REPO_ID: &str = "Qwen/Qwen3.5-0.8B";
const LOCKED_REVISION: &str = "2fc06364715b967f1860aea9cf38778875588b17";
const LOCKED_CHECKPOINT: &str = "model.safetensors-00001-of-00001.safetensors";
const LOCKED_CHECKPOINT_SHA256: &str =
    "04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696";
const LOCKED_CHECKPOINT_BYTES: u64 = 1_746_942_600;
const STEPS: usize = 128;
const PROMPT: &str = "Hello";
const GATE_SOURCE_NAME: &str = "qwen35_metal_w8_linear_layer_gate.rs";
const GATE_SOURCE_BYTES: &[u8] = include_bytes!("qwen35_metal_w8_linear_layer_gate.rs");
const ALL_LINEAR_LAYER_INDICES: [usize; 18] = [
    0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 20, 21, 22,
];
const FULL_ATTENTION_LAYER_INDICES: [usize; 6] = [3, 7, 11, 15, 19, 23];
const LINEAR_LAYER_STACK3_INDICES: [[usize; 3]; 6] = [
    [0, 1, 2],
    [4, 5, 6],
    [8, 9, 10],
    [12, 13, 14],
    [16, 17, 18],
    [20, 21, 22],
];
const OFFICIAL_HIDDEN_BYTES: usize = 1_024 * std::mem::size_of::<f32>();
const OFFICIAL_LINEAR_LAYER_PERSISTENT_BYTES: usize = 25_539_328;
const OFFICIAL_MLP_BLOCK_PERSISTENT_BYTES: usize = 11_749_376;
const OFFICIAL_ALL_LINEAR_PERSISTENT_BYTES: usize = 530_204_160;
const OFFICIAL_STACK3_PERSISTENT_BYTES: usize = 76_351_488;
const OFFICIAL_STACK3_PACKED_WEIGHT_BYTES: usize = 64_585_728;
const OFFICIAL_STACK3_PACKED_SCALE_BYTES: usize = 4_429_824;
const OFFICIAL_STACK3_BODY_PERSISTENT_BYTES: usize = 528_605_184;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mode {
    CpuTeacher,
    LinearLayerTeacher,
    LinearLayerGdnOutG32V2Teacher,
    CpuFree,
    LinearLayerFree,
    LinearLayerGdnOutG32V2Free,
    AllLinearLayersGdnOutG32V2Teacher,
    AllLinearLayersGdnOutG32V2Free,
    AllLinearLayerStacksV1Teacher,
    AllLinearLayerStacksV1Free,
}

impl Mode {
    fn parse(value: &str) -> Result<Self, String> {
        match value {
            "cpu-teacher" => Ok(Self::CpuTeacher),
            "linear-layer-teacher" => Ok(Self::LinearLayerTeacher),
            "linear-layer-gdn-out-g32-v2-teacher" => Ok(Self::LinearLayerGdnOutG32V2Teacher),
            "cpu-free" => Ok(Self::CpuFree),
            "linear-layer-free" => Ok(Self::LinearLayerFree),
            "linear-layer-gdn-out-g32-v2-free" => Ok(Self::LinearLayerGdnOutG32V2Free),
            "all-linear-layers-gdn-out-g32-v2-teacher" => {
                Ok(Self::AllLinearLayersGdnOutG32V2Teacher)
            }
            "all-linear-layers-gdn-out-g32-v2-free" => Ok(Self::AllLinearLayersGdnOutG32V2Free),
            "all-linear-layer-stacks-v1-teacher" => Ok(Self::AllLinearLayerStacksV1Teacher),
            "all-linear-layer-stacks-v1-free" => Ok(Self::AllLinearLayerStacksV1Free),
            other => Err(format!("invalid --mode {other:?}")),
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::CpuTeacher => "linear_layer_cpu_teacher",
            Self::LinearLayerTeacher => "metal_w8_linear_layer_teacher_forced",
            Self::LinearLayerGdnOutG32V2Teacher => {
                "metal_w8_linear_layer_gdn_out_g32_v2_teacher_forced"
            }
            Self::CpuFree => "linear_layer_cpu_free_run",
            Self::LinearLayerFree => "metal_w8_linear_layer_free_run",
            Self::LinearLayerGdnOutG32V2Free => "metal_w8_linear_layer_gdn_out_g32_v2_free_run",
            Self::AllLinearLayersGdnOutG32V2Teacher => {
                "metal_w8_all_linear_layers_gdn_out_g32_v2_teacher_forced"
            }
            Self::AllLinearLayersGdnOutG32V2Free => {
                "metal_w8_all_linear_layers_gdn_out_g32_v2_free_run"
            }
            Self::AllLinearLayerStacksV1Teacher => {
                "metal_w8_all_linear_layer_stacks_v1_teacher_forced"
            }
            Self::AllLinearLayerStacksV1Free => "metal_w8_all_linear_layer_stacks_v1_free_run",
        }
    }

    fn is_linear_layer(self) -> bool {
        matches!(
            self,
            Self::LinearLayerTeacher
                | Self::LinearLayerGdnOutG32V2Teacher
                | Self::LinearLayerFree
                | Self::LinearLayerGdnOutG32V2Free
                | Self::AllLinearLayersGdnOutG32V2Teacher
                | Self::AllLinearLayersGdnOutG32V2Free
                | Self::AllLinearLayerStacksV1Teacher
                | Self::AllLinearLayerStacksV1Free
        )
    }

    fn is_precision_v2(self) -> bool {
        matches!(
            self,
            Self::LinearLayerGdnOutG32V2Teacher | Self::LinearLayerGdnOutG32V2Free
        )
    }

    fn is_all_linear_layers_precision_v2(self) -> bool {
        matches!(
            self,
            Self::AllLinearLayersGdnOutG32V2Teacher | Self::AllLinearLayersGdnOutG32V2Free
        )
    }

    fn is_all_linear_layer_stacks_v1(self) -> bool {
        matches!(
            self,
            Self::AllLinearLayerStacksV1Teacher | Self::AllLinearLayerStacksV1Free
        )
    }

    fn requires_input(self) -> bool {
        self.is_linear_layer()
    }
}

struct Args {
    model_dir: PathBuf,
    source_lock: PathBuf,
    mode: Mode,
    input_receipt: Option<PathBuf>,
    output: PathBuf,
    layer: usize,
}

struct RunResult {
    receipt: Value,
    passed: bool,
}

fn usage() -> &'static str {
    "Usage: qwen35_metal_w8_linear_layer_gate \
  --model-dir OFFICIAL_LOCAL_QWEN35_0_8B \
  --source-lock SOURCE_LOCK.json \
  --mode cpu-teacher|linear-layer-teacher|linear-layer-gdn-out-g32-v2-teacher|cpu-free|linear-layer-free|linear-layer-gdn-out-g32-v2-free|all-linear-layers-gdn-out-g32-v2-teacher|all-linear-layers-gdn-out-g32-v2-free|all-linear-layer-stacks-v1-teacher|all-linear-layer-stacks-v1-free \
  [--input-receipt CPU_RECEIPT.json] \
  --output NEW_RECEIPT.json \
  [--layer 0]\n\
The gate is frozen to prompt=Hello and 128 steps. linear-layer-* modes require the matching CPU receipt. Output publication uses create_new and never replaces an artifact."
}

fn main() -> Result<(), Box<dyn Error>> {
    if cfg!(debug_assertions) {
        return Err("qwen35_metal_w8_linear_layer_gate must be built with --release".into());
    }
    if !cfg!(target_os = "macos") {
        return Err("qwen35_metal_w8_linear_layer_gate requires macOS".into());
    }
    let args = parse_args()?;
    if args.output.exists() {
        return Err(format!(
            "refusing to replace existing receipt {}",
            args.output.display()
        )
        .into());
    }
    let custody = gate_evidence::GateCustody::capture_stack3(
        &args.model_dir,
        &args.source_lock,
        GATE_SOURCE_NAME,
        GATE_SOURCE_BYTES,
    )?;
    validate_source_lock(custody.source_lock_value())?;
    let canonical_model_dir = custody.model_dir().to_path_buf();
    let canonical_source_lock = fs::canonicalize(&args.source_lock)?;
    let binary_path = fs::canonicalize(std::env::current_exe()?)?;

    let tokenizer = Tokenizer::from_file(canonical_model_dir.join("tokenizer.json"))?;
    let formatted = tokenizer.apply_chat_template(&[ChatMessage::user(PROMPT)])?;
    let prompt_tokens = tokenizer.encode(&formatted)?;
    let config = Qwen35Config::from_json_file(&canonical_model_dir.join("config.json"))?;
    if args.mode.is_all_linear_layers_precision_v2() || args.mode.is_all_linear_layer_stacks_v1() {
        validate_official_all_linear_schedule(&config.text.layer_types)?;
    } else if config.text.layer_types.get(args.layer) != Some(&Qwen35LayerType::LinearAttention) {
        return Err(format!("selected layer {} is not linear attention", args.layer).into());
    }
    let vocab_size = config.text.vocab_size;

    let checkpoint_started = std::time::Instant::now();
    let (tensors, _) =
        apxinf_loader::safetensors::load_native_path_filtered(&canonical_model_dir, |name| {
            name.starts_with("model.language_model.") || name == "lm_head.weight"
        })?;
    let checkpoint_load_ms = checkpoint_started.elapsed().as_secs_f64() * 1000.0;
    let max_context = prompt_tokens
        .len()
        .checked_add(STEPS + 1)
        .ok_or("context length overflow")?;
    let construct_started = std::time::Instant::now();
    let mut model = if args.mode.is_all_linear_layer_stacks_v1() {
        GeneralQwen35::from_weights_with_metal_w8_all_linear_layer_stacks_v1(
            config,
            tensors,
            Device::Cpu,
            max_context,
        )?
    } else if args.mode.is_all_linear_layers_precision_v2() {
        GeneralQwen35::from_weights_with_metal_w8_all_linear_layers_precision_v2(
            config,
            tensors,
            Device::Cpu,
            max_context,
            Qwen35MetalW8LinearLayerPrecisionProfile::GdnOutG32V2,
        )?
    } else if args.mode.is_precision_v2() {
        GeneralQwen35::from_weights_with_metal_w8_linear_layer_precision_v2(
            config,
            tensors,
            Device::Cpu,
            max_context,
            args.layer,
            Qwen35MetalW8LinearLayerPrecisionProfile::GdnOutG32V2,
        )?
    } else if args.mode.is_linear_layer() {
        GeneralQwen35::from_weights_with_metal_w8_linear_layer(
            config,
            tensors,
            Device::Cpu,
            max_context,
            args.layer,
        )?
    } else {
        GeneralQwen35::from_weights(config, tensors, Device::Cpu, max_context)?
    };
    let model_construct_ms = construct_started.elapsed().as_secs_f64() * 1000.0;
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
        "custody": custody.receipt_json(),
    });
    let setup = json!({
        "checkpoint_load_ms": checkpoint_load_ms,
        "model_construct_ms": model_construct_ms,
        "timing_classification": "single-pass candidate timing only; not formal ABBA evidence",
    });

    let mut result = match args.mode {
        Mode::CpuTeacher | Mode::LinearLayerTeacher | Mode::LinearLayerGdnOutG32V2Teacher => {
            run_teacher(
                &args,
                &mut model,
                &prompt_tokens,
                vocab_size,
                identity,
                setup,
            )?
        }
        Mode::CpuFree | Mode::LinearLayerFree | Mode::LinearLayerGdnOutG32V2Free => run_free(
            &args,
            &mut model,
            &prompt_tokens,
            vocab_size,
            identity,
            setup,
        )?,
        Mode::AllLinearLayersGdnOutG32V2Teacher => run_all_linear_layers_teacher(
            &args,
            &mut model,
            &prompt_tokens,
            vocab_size,
            identity,
            setup,
        )?,
        Mode::AllLinearLayersGdnOutG32V2Free => run_all_linear_layers_free(
            &args,
            &mut model,
            &prompt_tokens,
            vocab_size,
            identity,
            setup,
        )?,
        Mode::AllLinearLayerStacksV1Teacher => run_all_linear_layer_stacks_v1_teacher(
            &args,
            &mut model,
            &prompt_tokens,
            vocab_size,
            identity,
            setup,
        )?,
        Mode::AllLinearLayerStacksV1Free => run_all_linear_layer_stacks_v1_free(
            &args,
            &mut model,
            &prompt_tokens,
            vocab_size,
            identity,
            setup,
        )?,
    };
    let custody_end_verification = custody.verify_unchanged_receipt()?;
    result
        .receipt
        .as_object_mut()
        .ok_or("gate receipt root must be an object")?
        .insert("custody_end_verification".into(), custody_end_verification);
    publish_no_replace(&args.output, &result.receipt)?;
    println!("{}", serde_json::to_string(&result.receipt)?);
    if !result.passed {
        std::process::exit(1);
    }
    Ok(())
}

fn run_teacher(
    args: &Args,
    model: &mut GeneralQwen35,
    prompt_tokens: &[u32],
    vocab_size: usize,
    identity: Value,
    setup: Value,
) -> Result<RunResult, Box<dyn Error>> {
    let prefill_started = std::time::Instant::now();
    let prefill = model.prefill_for_generation(LlmInput::text(prompt_tokens))?;
    let prefill_ms = prefill_started.elapsed().as_secs_f64() * 1000.0;
    let prefill_token = argmax(&prefill, vocab_size)?;
    let prefill_path_receipt = model
        .metal_w8_linear_layer_stats()
        .map(linear_layer_stats_json);
    let prefill_precision_v2_receipt = model
        .metal_w8_linear_layer_precision_v2_stats()
        .map(precision_v2_stats_json);
    let buffer_ledger = model.metal_w8_linear_layer_buffer_ledger();

    if args.mode == Mode::CpuTeacher {
        let decode_started = std::time::Instant::now();
        let mut teacher = prefill_token;
        let mut teacher_inputs = Vec::with_capacity(STEPS);
        let mut expected_outputs = Vec::with_capacity(STEPS);
        for step in 0..STEPS {
            teacher_inputs.push(teacher);
            let logits = model.forward(
                &[teacher],
                u32::try_from(
                    prompt_tokens
                        .len()
                        .checked_add(step)
                        .ok_or("position overflow")?,
                )?,
            )?;
            teacher = argmax(&logits, vocab_size)?;
            expected_outputs.push(teacher);
        }
        let decode_ms = decode_started.elapsed().as_secs_f64() * 1000.0;
        return Ok(RunResult {
            receipt: json!({
                "format": CPU_TEACHER_FORMAT,
                "mode": args.mode.label(),
                "identity": identity,
                "prompt": PROMPT,
                "prompt_token_ids": prompt_tokens,
                "comparisons": STEPS,
                "selected_layer": args.layer,
                "prefill_token": prefill_token,
                "teacher_input_ids": teacher_inputs,
                "cpu_expected_output_ids": expected_outputs,
                "path_receipt": null,
                "timing": {
                    "setup": setup,
                    "prefill_ms": prefill_ms,
                    "decode_ms": decode_ms,
                    "decode_mean_ms": decode_ms / STEPS as f64,
                },
                "passed": true,
            }),
            passed: true,
        });
    }

    let input_path = args
        .input_receipt
        .as_ref()
        .ok_or("--input-receipt is required in linear-layer-teacher mode")?;
    let (cpu_receipt, input_attestation) =
        gate_evidence::read_attested_json(input_path, "Metal linear-layer CPU teacher receipt")?;
    validate_cpu_receipt(
        &cpu_receipt,
        CPU_TEACHER_FORMAT,
        "linear_layer_cpu_teacher",
        &identity,
        prompt_tokens,
        args.layer,
        Some(prefill_token),
        "teacher_input_ids",
        "cpu_expected_output_ids",
    )?;
    let teacher_inputs = json_u32_array(&cpu_receipt, "teacher_input_ids")?;
    let expected_outputs = json_u32_array(&cpu_receipt, "cpu_expected_output_ids")?;
    let decode_started = std::time::Instant::now();
    let mut actual_outputs = Vec::with_capacity(STEPS);
    let mut mismatches = Vec::new();
    for step in 0..STEPS {
        let logits = model.forward(
            &[teacher_inputs[step]],
            u32::try_from(
                prompt_tokens
                    .len()
                    .checked_add(step)
                    .ok_or("position overflow")?,
            )?,
        )?;
        let actual = argmax(&logits, vocab_size)?;
        actual_outputs.push(actual);
        if actual != expected_outputs[step] {
            mismatches.push(json!({
                "step": step,
                "teacher_input": teacher_inputs[step],
                "cpu_expected": expected_outputs[step],
                "metal_w8_linear_layer_actual": actual,
            }));
        }
    }
    let decode_ms = decode_started.elapsed().as_secs_f64() * 1000.0;
    let final_stats = model
        .metal_w8_linear_layer_stats()
        .ok_or("diagnostic constructor omitted Metal W8 linear layer receipt")?;
    let final_precision_v2 = model.metal_w8_linear_layer_precision_v2_stats();
    let ledger =
        buffer_ledger.ok_or("diagnostic constructor omitted Metal W8 linear layer ledger")?;
    let prefill_hits_valid = prefill_path_receipt.as_ref().is_some_and(|stats| {
        stats["prefill_seed_calls"] == 1
            && stats["decode_calls"] == 0
            && stats["successful_decodes"] == 0
            && stats["failed_decodes"] == 0
            && stats["command_buffers"] == 0
            && stats["compute_encoders"] == 0
            && stats["commits"] == 0
            && stats["waits"] == 0
            && stats["host_to_device_bytes"] == 0
            && stats["device_to_host_bytes"] == 0
            && stats["committed_state_version"] == 0
            && stats["terminal_error"] == false
    });
    let decode_hits_valid = final_stats.layer_index == args.layer
        && final_stats.prefill_seed_calls == 1
        && final_stats.decode_calls == STEPS
        && final_stats.successful_decodes == STEPS
        && final_stats.failed_decodes == 0
        && final_stats.command_buffers == STEPS
        && final_stats.compute_encoders == STEPS
        && final_stats.commits == STEPS
        && final_stats.waits == STEPS
        && final_stats.host_to_device_bytes == ledger.host_input_bytes_per_decode * STEPS
        && final_stats.device_to_host_bytes == ledger.host_output_bytes_per_decode * STEPS
        && final_stats.committed_state_version == STEPS as u64
        && !final_stats.terminal_error;
    let ledger_valid = linear_layer_ledger_is_exact(ledger);
    let precision_v2_valid = if args.mode.is_precision_v2() {
        final_precision_v2.is_some_and(|stats| {
            stats.profile == Qwen35MetalW8LinearLayerPrecisionProfile::GdnOutG32V2
                && stats.mechanism == "metal-w8-linear-layer-precision-v2"
                && stats.quantization.gdn_input_group_size == apxinf_metal::W8GroupSize::G64
                && stats.quantization.gdn_output_group_size == apxinf_metal::W8GroupSize::G32
                && stats.quantization.mlp_gate_group_size == apxinf_metal::W8GroupSize::G64
                && stats.quantization.mlp_up_group_size == apxinf_metal::W8GroupSize::G64
                && stats.quantization.mlp_down_group_size == apxinf_metal::W8GroupSize::G64
                && stats.quantization.total_packed_scale_bytes == ledger.packed_scale_bytes
        })
    } else {
        final_precision_v2.is_none()
    };
    let passed = mismatches.is_empty()
        && prefill_hits_valid
        && decode_hits_valid
        && ledger_valid
        && precision_v2_valid;
    gate_evidence::verify_file_unchanged(
        input_path,
        &input_attestation,
        "Metal linear-layer CPU teacher receipt",
    )?;
    let mut receipt = json!({
        "format": if args.mode.is_precision_v2() {
            LINEAR_LAYER_GDN_OUT_G32_V2_TEACHER_FORMAT
        } else {
            LINEAR_LAYER_TEACHER_FORMAT
        },
        "mode": args.mode.label(),
        "identity": identity,
        "prompt": PROMPT,
        "prompt_token_ids": prompt_tokens,
        "comparisons": STEPS,
        "selected_layer": args.layer,
        "prefill_token": prefill_token,
        "linear_layer_cpu_teacher_receipt": gate_evidence::attestation_json(&input_attestation),
        "teacher_input_ids": teacher_inputs,
        "cpu_expected_output_ids": expected_outputs,
        "metal_w8_linear_layer_actual_output_ids": actual_outputs,
        "mismatches": mismatches,
        "prefill_path_receipt": prefill_path_receipt,
        "final_path_receipt": linear_layer_stats_json(final_stats),
        "buffer_ledger": linear_layer_ledger_json(ledger),
        "path_checks": {
            "prefill_hits_valid": prefill_hits_valid,
            "decode_hits_valid": decode_hits_valid,
            "ledger_valid": ledger_valid,
        },
        "timing": {
            "setup": setup,
            "prefill_ms": prefill_ms,
            "decode_ms": decode_ms,
            "decode_mean_ms": decode_ms / STEPS as f64,
            "metal_linear_layer_block_mean_us": final_stats.block_elapsed_ns as f64
                / final_stats.decode_calls.max(1) as f64 / 1000.0,
            "classification": "candidate-only single pass under an uncontrolled host; never promotion evidence",
        },
        "passed": passed,
    });
    attach_precision_v2_receipts(
        &mut receipt,
        prefill_precision_v2_receipt,
        final_precision_v2,
        precision_v2_valid,
    )?;
    Ok(RunResult { receipt, passed })
}

fn run_free(
    args: &Args,
    model: &mut GeneralQwen35,
    prompt_tokens: &[u32],
    vocab_size: usize,
    identity: Value,
    setup: Value,
) -> Result<RunResult, Box<dyn Error>> {
    let prefill_started = std::time::Instant::now();
    let prefill = model.prefill_for_generation(LlmInput::text(prompt_tokens))?;
    let prefill_ms = prefill_started.elapsed().as_secs_f64() * 1000.0;
    let mut current = argmax(&prefill, vocab_size)?;
    let prefill_path_receipt = model
        .metal_w8_linear_layer_stats()
        .map(linear_layer_stats_json);
    let prefill_precision_v2_receipt = model
        .metal_w8_linear_layer_precision_v2_stats()
        .map(precision_v2_stats_json);
    let buffer_ledger = model.metal_w8_linear_layer_buffer_ledger();
    let mut generated = Vec::with_capacity(STEPS);
    generated.push(current);
    let decode_started = std::time::Instant::now();
    for generated_index in 1..STEPS {
        let position = prompt_tokens
            .len()
            .checked_add(generated_index - 1)
            .ok_or("position overflow")?;
        let logits = model.forward(&[current], u32::try_from(position)?)?;
        current = argmax(&logits, vocab_size)?;
        generated.push(current);
    }
    let decode_ms = decode_started.elapsed().as_secs_f64() * 1000.0;

    if args.mode == Mode::CpuFree {
        return Ok(RunResult {
            receipt: json!({
                "format": CPU_FREE_FORMAT,
                "mode": args.mode.label(),
                "identity": identity,
                "prompt": PROMPT,
                "prompt_token_ids": prompt_tokens,
                "generated_tokens": STEPS,
                "selected_layer": args.layer,
                "generated_token_ids": generated,
                "path_receipt": null,
                "timing": {
                    "setup": setup,
                    "prefill_ms": prefill_ms,
                    "decode_ms": decode_ms,
                    "decode_mean_ms": decode_ms / (STEPS - 1) as f64,
                },
                "passed": true,
            }),
            passed: true,
        });
    }

    let input_path = args
        .input_receipt
        .as_ref()
        .ok_or("--input-receipt is required in linear-layer-free mode")?;
    let (cpu_receipt, input_attestation) =
        gate_evidence::read_attested_json(input_path, "Metal linear-layer CPU free-run receipt")?;
    validate_cpu_receipt(
        &cpu_receipt,
        CPU_FREE_FORMAT,
        "linear_layer_cpu_free_run",
        &identity,
        prompt_tokens,
        args.layer,
        None,
        "generated_token_ids",
        "generated_token_ids",
    )?;
    let cpu_generated = json_u32_array(&cpu_receipt, "generated_token_ids")?;
    let mismatches = cpu_generated
        .iter()
        .zip(&generated)
        .enumerate()
        .filter_map(|(step, (&cpu, &gdn))| {
            (cpu != gdn).then(|| json!({ "step": step, "cpu": cpu, "metal_w8_linear_layer": gdn }))
        })
        .collect::<Vec<_>>();
    let first_mismatch = mismatches
        .first()
        .and_then(|entry| entry.get("step"))
        .and_then(Value::as_u64);
    let final_stats = model
        .metal_w8_linear_layer_stats()
        .ok_or("diagnostic constructor omitted Metal W8 linear layer receipt")?;
    let final_precision_v2 = model.metal_w8_linear_layer_precision_v2_stats();
    let ledger =
        buffer_ledger.ok_or("diagnostic constructor omitted Metal W8 linear layer ledger")?;
    let prefill_hits_valid = prefill_path_receipt.as_ref().is_some_and(|stats| {
        stats["prefill_seed_calls"] == 1
            && stats["decode_calls"] == 0
            && stats["successful_decodes"] == 0
            && stats["failed_decodes"] == 0
            && stats["command_buffers"] == 0
            && stats["compute_encoders"] == 0
            && stats["commits"] == 0
            && stats["waits"] == 0
            && stats["host_to_device_bytes"] == 0
            && stats["device_to_host_bytes"] == 0
            && stats["committed_state_version"] == 0
            && stats["terminal_error"] == false
    });
    let expected_decode = STEPS - 1;
    let decode_hits_valid = final_stats.layer_index == args.layer
        && final_stats.prefill_seed_calls == 1
        && final_stats.decode_calls == expected_decode
        && final_stats.successful_decodes == expected_decode
        && final_stats.failed_decodes == 0
        && final_stats.command_buffers == expected_decode
        && final_stats.compute_encoders == expected_decode
        && final_stats.commits == expected_decode
        && final_stats.waits == expected_decode
        && final_stats.host_to_device_bytes == ledger.host_input_bytes_per_decode * expected_decode
        && final_stats.device_to_host_bytes
            == ledger.host_output_bytes_per_decode * expected_decode
        && final_stats.committed_state_version == expected_decode as u64
        && !final_stats.terminal_error;
    let ledger_valid = linear_layer_ledger_is_exact(ledger);
    let precision_v2_valid = if args.mode.is_precision_v2() {
        final_precision_v2.is_some_and(|stats| {
            stats.profile == Qwen35MetalW8LinearLayerPrecisionProfile::GdnOutG32V2
                && stats.mechanism == "metal-w8-linear-layer-precision-v2"
                && stats.quantization.gdn_input_group_size == apxinf_metal::W8GroupSize::G64
                && stats.quantization.gdn_output_group_size == apxinf_metal::W8GroupSize::G32
                && stats.quantization.mlp_gate_group_size == apxinf_metal::W8GroupSize::G64
                && stats.quantization.mlp_up_group_size == apxinf_metal::W8GroupSize::G64
                && stats.quantization.mlp_down_group_size == apxinf_metal::W8GroupSize::G64
                && stats.quantization.total_packed_scale_bytes == ledger.packed_scale_bytes
        })
    } else {
        final_precision_v2.is_none()
    };
    let passed = cpu_generated.len() == STEPS
        && generated.len() == STEPS
        && mismatches.is_empty()
        && prefill_hits_valid
        && decode_hits_valid
        && ledger_valid
        && precision_v2_valid;
    gate_evidence::verify_file_unchanged(
        input_path,
        &input_attestation,
        "Metal linear-layer CPU free-run receipt",
    )?;
    let mut receipt = json!({
        "format": if args.mode.is_precision_v2() {
            LINEAR_LAYER_GDN_OUT_G32_V2_FREE_FORMAT
        } else {
            LINEAR_LAYER_FREE_FORMAT
        },
        "mode": args.mode.label(),
        "identity": identity,
        "prompt": PROMPT,
        "prompt_token_ids": prompt_tokens,
        "generated_tokens": STEPS,
        "selected_layer": args.layer,
        "cpu_free_receipt": gate_evidence::attestation_json(&input_attestation),
        "cpu_generated_token_ids": cpu_generated,
        "metal_w8_linear_layer_generated_token_ids": generated,
        "mismatches": mismatches,
        "first_mismatch": first_mismatch,
        "prefill_path_receipt": prefill_path_receipt,
        "final_path_receipt": linear_layer_stats_json(final_stats),
        "buffer_ledger": linear_layer_ledger_json(ledger),
        "path_checks": {
            "prefill_hits_valid": prefill_hits_valid,
            "decode_hits_valid": decode_hits_valid,
            "ledger_valid": ledger_valid,
        },
        "timing": {
            "setup": setup,
            "prefill_ms": prefill_ms,
            "decode_ms": decode_ms,
            "decode_mean_ms": decode_ms / expected_decode as f64,
            "metal_linear_layer_block_mean_us": final_stats.block_elapsed_ns as f64
                / final_stats.decode_calls.max(1) as f64 / 1000.0,
            "classification": "candidate-only single pass under an uncontrolled host; never promotion evidence",
        },
        "passed": passed,
    });
    attach_precision_v2_receipts(
        &mut receipt,
        prefill_precision_v2_receipt,
        final_precision_v2,
        precision_v2_valid,
    )?;
    Ok(RunResult { receipt, passed })
}

fn run_all_linear_layers_teacher(
    args: &Args,
    model: &mut GeneralQwen35,
    prompt_tokens: &[u32],
    vocab_size: usize,
    identity: Value,
    setup: Value,
) -> Result<RunResult, Box<dyn Error>> {
    let prefill_started = std::time::Instant::now();
    let prefill = model.prefill_for_generation(LlmInput::text(prompt_tokens))?;
    let prefill_ms = prefill_started.elapsed().as_secs_f64() * 1000.0;
    let prefill_token = argmax(&prefill, vocab_size)?;
    let aggregate = model
        .metal_w8_all_linear_layers_precision_v2_aggregate_ledger()
        .ok_or("all-linear-layers constructor omitted aggregate buffer ledger")?;
    let prefill_stats = model
        .metal_w8_all_linear_layers_precision_v2_stats()
        .ok_or("all-linear-layers constructor omitted aggregate path receipt")?;
    let prefill_checks = all_linear_path_checks(&prefill_stats, &aggregate, 0);
    let prefill_generation_receipt = model
        .generation_path_receipt()
        .ok_or("all-linear-layers constructor omitted generation path receipt")?;
    let prefill_generation_valid =
        all_linear_generation_receipt_is_exact(&prefill_generation_receipt, 0);

    let input_path = args
        .input_receipt
        .as_ref()
        .ok_or("--input-receipt is required in all-linear-layers teacher mode")?;
    let (cpu_receipt, input_attestation) =
        gate_evidence::read_attested_json(input_path, "all-linear-layers CPU teacher receipt")?;
    validate_cpu_receipt(
        &cpu_receipt,
        CPU_TEACHER_FORMAT,
        "linear_layer_cpu_teacher",
        &identity,
        prompt_tokens,
        0,
        Some(prefill_token),
        "teacher_input_ids",
        "cpu_expected_output_ids",
    )?;
    let teacher_inputs = json_u32_array(&cpu_receipt, "teacher_input_ids")?;
    let expected_outputs = json_u32_array(&cpu_receipt, "cpu_expected_output_ids")?;
    let decode_started = std::time::Instant::now();
    let mut actual_outputs = Vec::with_capacity(STEPS);
    let mut mismatches = Vec::new();
    for step in 0..STEPS {
        let logits = model.forward(
            &[teacher_inputs[step]],
            u32::try_from(
                prompt_tokens
                    .len()
                    .checked_add(step)
                    .ok_or("position overflow")?,
            )?,
        )?;
        let actual = argmax(&logits, vocab_size)?;
        actual_outputs.push(actual);
        if actual != expected_outputs[step] {
            mismatches.push(json!({
                "step": step,
                "teacher_input": teacher_inputs[step],
                "cpu_expected": expected_outputs[step],
                "metal_w8_all_linear_layers_actual": actual,
            }));
        }
    }
    let decode_ms = decode_started.elapsed().as_secs_f64() * 1000.0;
    let final_stats = model
        .metal_w8_all_linear_layers_precision_v2_stats()
        .ok_or("all-linear-layers constructor omitted final aggregate path receipt")?;
    let final_checks = all_linear_path_checks(&final_stats, &aggregate, STEPS);
    let final_generation_receipt = model
        .generation_path_receipt()
        .ok_or("all-linear-layers constructor omitted final generation path receipt")?;
    let final_generation_valid =
        all_linear_generation_receipt_is_exact(&final_generation_receipt, STEPS);
    let passed = mismatches.is_empty()
        && prefill_checks.all_valid()
        && final_checks.all_valid()
        && prefill_generation_valid
        && final_generation_valid;
    gate_evidence::verify_file_unchanged(
        input_path,
        &input_attestation,
        "all-linear-layers CPU teacher receipt",
    )?;
    let linear_elapsed_ns = final_stats
        .linear_layers
        .iter()
        .map(|entry| entry.execution.block_elapsed_ns)
        .sum::<u128>();
    let full_mlp_elapsed_ns = final_stats
        .full_attention_mlp_layers
        .iter()
        .map(|entry| entry.block_elapsed_ns)
        .sum::<u128>();
    Ok(RunResult {
        receipt: json!({
            "format": ALL_LINEAR_LAYERS_GDN_OUT_G32_V2_TEACHER_FORMAT,
            "mode": args.mode.label(),
            "identity": identity,
            "prompt": PROMPT,
            "prompt_token_ids": prompt_tokens,
            "comparisons": STEPS,
            "prefill_token": prefill_token,
            "schedule": {
                "total_layers": 24,
                "linear_attention_complete_block_layers": ALL_LINEAR_LAYER_INDICES,
                "full_attention_cpu_attention_metal_mlp_layers": FULL_ATTENTION_LAYER_INDICES,
                "linear_layer_count": 18,
                "full_attention_layer_count": 6,
                "duplicate_mlp_execution": false,
            },
            "cpu_teacher_receipt": gate_evidence::attestation_json(&input_attestation),
            "teacher_input_ids": teacher_inputs,
            "cpu_expected_output_ids": expected_outputs,
            "metal_w8_all_linear_layers_actual_output_ids": actual_outputs,
            "mismatches": mismatches,
            "prefill_aggregate_path_receipt": all_linear_stats_json(&prefill_stats),
            "final_aggregate_path_receipt": all_linear_stats_json(&final_stats),
            "prefill_generation_path_receipt": prefill_generation_receipt,
            "final_generation_path_receipt": final_generation_receipt,
            "generation_path_contract": {
                "schema": "apxinf-qwen35-all-linear-layers-generation-path-v2",
                "legacy_v1_applicable": false,
                "binds_all_18_complete_layers_and_6_full_attention_mlp_layers": true,
            },
            "aggregate_buffer_ledger": all_linear_aggregate_ledger_json(&aggregate),
            "path_checks": {
                "prefill": prefill_checks.receipt_json(),
                "decode": final_checks.receipt_json(),
                "prefill_generation_receipt_valid": prefill_generation_valid,
                "decode_generation_receipt_valid": final_generation_valid,
                "exact_trajectory": mismatches.is_empty(),
            },
            "timing": {
                "setup": setup,
                "prefill_ms": prefill_ms,
                "decode_ms": decode_ms,
                "decode_mean_ms": decode_ms / STEPS as f64,
                "all_linear_complete_blocks_mean_us": linear_elapsed_ns as f64
                    / STEPS as f64 / 1000.0,
                "full_attention_mlp_blocks_mean_us": full_mlp_elapsed_ns as f64
                    / STEPS as f64 / 1000.0,
                "classification": "candidate-only single pass under an uncontrolled host; never promotion evidence",
            },
            "passed": passed,
        }),
        passed,
    })
}

fn run_all_linear_layers_free(
    args: &Args,
    model: &mut GeneralQwen35,
    prompt_tokens: &[u32],
    vocab_size: usize,
    identity: Value,
    setup: Value,
) -> Result<RunResult, Box<dyn Error>> {
    let input_path = args
        .input_receipt
        .as_ref()
        .ok_or("--input-receipt is required in all-linear-layers free mode")?;
    let (cpu_receipt, input_attestation) =
        gate_evidence::read_attested_json(input_path, "all-linear-layers CPU free-run receipt")?;
    validate_cpu_receipt(
        &cpu_receipt,
        CPU_FREE_FORMAT,
        "linear_layer_cpu_free_run",
        &identity,
        prompt_tokens,
        0,
        None,
        "generated_token_ids",
        "generated_token_ids",
    )?;
    let cpu_generated = json_u32_array(&cpu_receipt, "generated_token_ids")?;

    let prefill_started = std::time::Instant::now();
    let prefill = model.prefill_for_generation(LlmInput::text(prompt_tokens))?;
    let prefill_ms = prefill_started.elapsed().as_secs_f64() * 1000.0;
    let mut current = argmax(&prefill, vocab_size)?;
    let aggregate = model
        .metal_w8_all_linear_layers_precision_v2_aggregate_ledger()
        .ok_or("all-linear-layers constructor omitted aggregate buffer ledger")?;
    let prefill_stats = model
        .metal_w8_all_linear_layers_precision_v2_stats()
        .ok_or("all-linear-layers constructor omitted aggregate path receipt")?;
    let prefill_checks = all_linear_path_checks(&prefill_stats, &aggregate, 0);
    let prefill_generation_receipt = model
        .generation_path_receipt()
        .ok_or("all-linear-layers constructor omitted generation path receipt")?;
    let prefill_generation_valid =
        all_linear_generation_receipt_is_exact(&prefill_generation_receipt, 0);
    let mut generated = Vec::with_capacity(STEPS);
    generated.push(current);
    let decode_started = std::time::Instant::now();
    for generated_index in 1..STEPS {
        let position = prompt_tokens
            .len()
            .checked_add(generated_index - 1)
            .ok_or("position overflow")?;
        let logits = model.forward(&[current], u32::try_from(position)?)?;
        current = argmax(&logits, vocab_size)?;
        generated.push(current);
    }
    let decode_ms = decode_started.elapsed().as_secs_f64() * 1000.0;
    let mismatches = cpu_generated
        .iter()
        .zip(&generated)
        .enumerate()
        .filter_map(|(step, (&cpu, &actual))| {
            (cpu != actual).then(|| {
                json!({
                    "step": step,
                    "cpu": cpu,
                    "metal_w8_all_linear_layers": actual,
                })
            })
        })
        .collect::<Vec<_>>();
    let first_mismatch = mismatches
        .first()
        .and_then(|entry| entry.get("step"))
        .and_then(Value::as_u64);
    let expected_decode = STEPS - 1;
    let final_stats = model
        .metal_w8_all_linear_layers_precision_v2_stats()
        .ok_or("all-linear-layers constructor omitted final aggregate path receipt")?;
    let final_checks = all_linear_path_checks(&final_stats, &aggregate, expected_decode);
    let final_generation_receipt = model
        .generation_path_receipt()
        .ok_or("all-linear-layers constructor omitted final generation path receipt")?;
    let final_generation_valid =
        all_linear_generation_receipt_is_exact(&final_generation_receipt, expected_decode);
    let passed = cpu_generated.len() == STEPS
        && generated.len() == STEPS
        && mismatches.is_empty()
        && prefill_checks.all_valid()
        && final_checks.all_valid()
        && prefill_generation_valid
        && final_generation_valid;
    gate_evidence::verify_file_unchanged(
        input_path,
        &input_attestation,
        "all-linear-layers CPU free-run receipt",
    )?;
    let linear_elapsed_ns = final_stats
        .linear_layers
        .iter()
        .map(|entry| entry.execution.block_elapsed_ns)
        .sum::<u128>();
    let full_mlp_elapsed_ns = final_stats
        .full_attention_mlp_layers
        .iter()
        .map(|entry| entry.block_elapsed_ns)
        .sum::<u128>();
    Ok(RunResult {
        receipt: json!({
            "format": ALL_LINEAR_LAYERS_GDN_OUT_G32_V2_FREE_FORMAT,
            "mode": args.mode.label(),
            "identity": identity,
            "prompt": PROMPT,
            "prompt_token_ids": prompt_tokens,
            "generated_tokens": STEPS,
            "schedule": {
                "total_layers": 24,
                "linear_attention_complete_block_layers": ALL_LINEAR_LAYER_INDICES,
                "full_attention_cpu_attention_metal_mlp_layers": FULL_ATTENTION_LAYER_INDICES,
                "linear_layer_count": 18,
                "full_attention_layer_count": 6,
                "duplicate_mlp_execution": false,
            },
            "cpu_free_receipt": gate_evidence::attestation_json(&input_attestation),
            "cpu_generated_token_ids": cpu_generated,
            "metal_w8_all_linear_layers_generated_token_ids": generated,
            "mismatches": mismatches,
            "first_mismatch": first_mismatch,
            "prefill_aggregate_path_receipt": all_linear_stats_json(&prefill_stats),
            "final_aggregate_path_receipt": all_linear_stats_json(&final_stats),
            "prefill_generation_path_receipt": prefill_generation_receipt,
            "final_generation_path_receipt": final_generation_receipt,
            "generation_path_contract": {
                "schema": "apxinf-qwen35-all-linear-layers-generation-path-v2",
                "legacy_v1_applicable": false,
                "binds_all_18_complete_layers_and_6_full_attention_mlp_layers": true,
            },
            "aggregate_buffer_ledger": all_linear_aggregate_ledger_json(&aggregate),
            "path_checks": {
                "prefill": prefill_checks.receipt_json(),
                "decode": final_checks.receipt_json(),
                "prefill_generation_receipt_valid": prefill_generation_valid,
                "decode_generation_receipt_valid": final_generation_valid,
                "exact_trajectory": mismatches.is_empty(),
            },
            "timing": {
                "setup": setup,
                "prefill_ms": prefill_ms,
                "decode_ms": decode_ms,
                "decode_mean_ms": decode_ms / expected_decode as f64,
                "all_linear_complete_blocks_mean_us": linear_elapsed_ns as f64
                    / expected_decode as f64 / 1000.0,
                "full_attention_mlp_blocks_mean_us": full_mlp_elapsed_ns as f64
                    / expected_decode as f64 / 1000.0,
                "classification": "candidate-only single pass under an uncontrolled host; never promotion evidence",
            },
            "passed": passed,
        }),
        passed,
    })
}

fn run_all_linear_layer_stacks_v1_teacher(
    args: &Args,
    model: &mut GeneralQwen35,
    prompt_tokens: &[u32],
    vocab_size: usize,
    identity: Value,
    setup: Value,
) -> Result<RunResult, Box<dyn Error>> {
    let prefill_started = std::time::Instant::now();
    let prefill = model.prefill_for_generation(LlmInput::text(prompt_tokens))?;
    let prefill_ms = prefill_started.elapsed().as_secs_f64() * 1000.0;
    let prefill_token = argmax(&prefill, vocab_size)?;
    let aggregate = model
        .metal_w8_linear_layer_stacks_v1_aggregate_ledger()
        .ok_or("stack3-v1 constructor omitted aggregate buffer ledger")?;
    let prefill_stats = model
        .metal_w8_linear_layer_stacks_v1_stats()
        .ok_or("stack3-v1 constructor omitted aggregate path receipt")?;
    let prefill_checks = stack3_path_checks(&prefill_stats, &aggregate, 0);
    let prefill_generation_receipt = model
        .generation_path_receipt()
        .ok_or("stack3-v1 constructor omitted generation path receipt")?;
    let prefill_generation_valid =
        stack3_generation_receipt_is_exact(&prefill_generation_receipt, 0);

    let input_path = args
        .input_receipt
        .as_ref()
        .ok_or("--input-receipt is required in all-stack3-v1 teacher mode")?;
    let (cpu_receipt, input_attestation) =
        gate_evidence::read_attested_json(input_path, "stack3-v1 CPU teacher receipt")?;
    validate_cpu_receipt(
        &cpu_receipt,
        CPU_TEACHER_FORMAT,
        "linear_layer_cpu_teacher",
        &identity,
        prompt_tokens,
        0,
        Some(prefill_token),
        "teacher_input_ids",
        "cpu_expected_output_ids",
    )?;
    let teacher_inputs = json_u32_array(&cpu_receipt, "teacher_input_ids")?;
    let expected_outputs = json_u32_array(&cpu_receipt, "cpu_expected_output_ids")?;
    let decode_started = std::time::Instant::now();
    let mut actual_outputs = Vec::with_capacity(STEPS);
    let mut mismatches = Vec::new();
    for step in 0..STEPS {
        let logits = model.forward(
            &[teacher_inputs[step]],
            u32::try_from(
                prompt_tokens
                    .len()
                    .checked_add(step)
                    .ok_or("position overflow")?,
            )?,
        )?;
        let actual = argmax(&logits, vocab_size)?;
        actual_outputs.push(actual);
        if actual != expected_outputs[step] {
            mismatches.push(json!({
                "step": step,
                "teacher_input": teacher_inputs[step],
                "cpu_expected": expected_outputs[step],
                "metal_w8_all_linear_layer_stacks_v1_actual": actual,
            }));
        }
    }
    let decode_ms = decode_started.elapsed().as_secs_f64() * 1000.0;
    let final_stats = model
        .metal_w8_linear_layer_stacks_v1_stats()
        .ok_or("stack3-v1 constructor omitted final aggregate path receipt")?;
    let final_checks = stack3_path_checks(&final_stats, &aggregate, STEPS);
    let final_generation_receipt = model
        .generation_path_receipt()
        .ok_or("stack3-v1 constructor omitted final generation path receipt")?;
    let final_generation_valid =
        stack3_generation_receipt_is_exact(&final_generation_receipt, STEPS);
    let passed = mismatches.is_empty()
        && prefill_checks.all_valid()
        && final_checks.all_valid()
        && prefill_generation_valid
        && final_generation_valid;
    gate_evidence::verify_file_unchanged(
        input_path,
        &input_attestation,
        "stack3-v1 CPU teacher receipt",
    )?;
    let stack_elapsed_ns = final_stats
        .stacks
        .iter()
        .map(|entry| entry.block_elapsed_ns)
        .sum::<u128>();
    let full_mlp_elapsed_ns = final_stats
        .full_attention_mlp_layers
        .iter()
        .map(|entry| entry.block_elapsed_ns)
        .sum::<u128>();
    Ok(RunResult {
        receipt: json!({
            "format": ALL_LINEAR_LAYER_STACKS_V1_TEACHER_FORMAT,
            "mode": args.mode.label(),
            "identity": identity,
            "prompt": PROMPT,
            "prompt_token_ids": prompt_tokens,
            "comparisons": STEPS,
            "prefill_token": prefill_token,
            "schedule": {
                "total_layers": 24,
                "linear_attention_complete_layer_stacks": LINEAR_LAYER_STACK3_INDICES,
                "full_attention_cpu_attention_metal_mlp_layers": FULL_ATTENTION_LAYER_INDICES,
                "stack_count": 6,
                "layers_per_stack": 3,
                "linear_layer_count": 18,
                "full_attention_layer_count": 6,
                "duplicate_mlp_execution": false,
            },
            "per_stack_transaction_contract": {
                "command_buffers": 1,
                "compute_encoders": 3,
                "commits": 1,
                "waits": 1,
                "state_commits": 3,
                "state_commit_mask": 0b111,
                "intermediate_host_finite_checks": false,
                "final_output_finite_checks": true,
                "terminal_error": false,
            },
            "cpu_teacher_receipt": gate_evidence::attestation_json(&input_attestation),
            "teacher_input_ids": teacher_inputs,
            "cpu_expected_output_ids": expected_outputs,
            "metal_w8_all_linear_layer_stacks_v1_actual_output_ids": actual_outputs,
            "mismatches": mismatches,
            "prefill_aggregate_path_receipt": stack3_stats_json(&prefill_stats),
            "final_aggregate_path_receipt": stack3_stats_json(&final_stats),
            "prefill_generation_path_receipt": prefill_generation_receipt,
            "final_generation_path_receipt": final_generation_receipt,
            "generation_path_contract": {
                "schema": "apxinf-qwen35-linear-layer-stacks-generation-path-v1",
                "versioned_stack3_semantics": true,
                "binds_six_three_layer_stacks_and_six_full_attention_mlp_layers": true,
                "intermediate_host_finite_checks": false,
                "final_output_finite_checks": true,
            },
            "aggregate_buffer_ledger": stack3_aggregate_ledger_json(&aggregate),
            "path_checks": {
                "prefill": prefill_checks.receipt_json(),
                "decode": final_checks.receipt_json(),
                "prefill_generation_receipt_valid": prefill_generation_valid,
                "decode_generation_receipt_valid": final_generation_valid,
                "exact_trajectory": mismatches.is_empty(),
            },
            "timing": {
                "setup": setup,
                "prefill_ms": prefill_ms,
                "decode_ms": decode_ms,
                "decode_mean_ms": decode_ms / STEPS as f64,
                "six_stack3_blocks_mean_us": stack_elapsed_ns as f64
                    / STEPS as f64 / 1000.0,
                "full_attention_mlp_blocks_mean_us": full_mlp_elapsed_ns as f64
                    / STEPS as f64 / 1000.0,
                "classification": "candidate-only single pass under an uncontrolled host; never promotion evidence",
            },
            "passed": passed,
        }),
        passed,
    })
}

fn run_all_linear_layer_stacks_v1_free(
    args: &Args,
    model: &mut GeneralQwen35,
    prompt_tokens: &[u32],
    vocab_size: usize,
    identity: Value,
    setup: Value,
) -> Result<RunResult, Box<dyn Error>> {
    let input_path = args
        .input_receipt
        .as_ref()
        .ok_or("--input-receipt is required in all-stack3-v1 free mode")?;
    let (cpu_receipt, input_attestation) =
        gate_evidence::read_attested_json(input_path, "stack3-v1 CPU free-run receipt")?;
    validate_cpu_receipt(
        &cpu_receipt,
        CPU_FREE_FORMAT,
        "linear_layer_cpu_free_run",
        &identity,
        prompt_tokens,
        0,
        None,
        "generated_token_ids",
        "generated_token_ids",
    )?;
    let cpu_generated = json_u32_array(&cpu_receipt, "generated_token_ids")?;

    let prefill_started = std::time::Instant::now();
    let prefill = model.prefill_for_generation(LlmInput::text(prompt_tokens))?;
    let prefill_ms = prefill_started.elapsed().as_secs_f64() * 1000.0;
    let mut current = argmax(&prefill, vocab_size)?;
    let aggregate = model
        .metal_w8_linear_layer_stacks_v1_aggregate_ledger()
        .ok_or("stack3-v1 constructor omitted aggregate buffer ledger")?;
    let prefill_stats = model
        .metal_w8_linear_layer_stacks_v1_stats()
        .ok_or("stack3-v1 constructor omitted aggregate path receipt")?;
    let prefill_checks = stack3_path_checks(&prefill_stats, &aggregate, 0);
    let prefill_generation_receipt = model
        .generation_path_receipt()
        .ok_or("stack3-v1 constructor omitted generation path receipt")?;
    let prefill_generation_valid =
        stack3_generation_receipt_is_exact(&prefill_generation_receipt, 0);
    let mut generated = Vec::with_capacity(STEPS);
    generated.push(current);
    let decode_started = std::time::Instant::now();
    for generated_index in 1..STEPS {
        let position = prompt_tokens
            .len()
            .checked_add(generated_index - 1)
            .ok_or("position overflow")?;
        let logits = model.forward(&[current], u32::try_from(position)?)?;
        current = argmax(&logits, vocab_size)?;
        generated.push(current);
    }
    let decode_ms = decode_started.elapsed().as_secs_f64() * 1000.0;
    let mismatches = cpu_generated
        .iter()
        .zip(&generated)
        .enumerate()
        .filter_map(|(step, (&cpu, &actual))| {
            (cpu != actual).then(|| {
                json!({
                    "step": step,
                    "cpu": cpu,
                    "metal_w8_all_linear_layer_stacks_v1": actual,
                })
            })
        })
        .collect::<Vec<_>>();
    let first_mismatch = mismatches
        .first()
        .and_then(|entry| entry.get("step"))
        .and_then(Value::as_u64);
    let expected_decode = STEPS - 1;
    let final_stats = model
        .metal_w8_linear_layer_stacks_v1_stats()
        .ok_or("stack3-v1 constructor omitted final aggregate path receipt")?;
    let final_checks = stack3_path_checks(&final_stats, &aggregate, expected_decode);
    let final_generation_receipt = model
        .generation_path_receipt()
        .ok_or("stack3-v1 constructor omitted final generation path receipt")?;
    let final_generation_valid =
        stack3_generation_receipt_is_exact(&final_generation_receipt, expected_decode);
    let passed = cpu_generated.len() == STEPS
        && generated.len() == STEPS
        && mismatches.is_empty()
        && prefill_checks.all_valid()
        && final_checks.all_valid()
        && prefill_generation_valid
        && final_generation_valid;
    gate_evidence::verify_file_unchanged(
        input_path,
        &input_attestation,
        "stack3-v1 CPU free-run receipt",
    )?;
    let stack_elapsed_ns = final_stats
        .stacks
        .iter()
        .map(|entry| entry.block_elapsed_ns)
        .sum::<u128>();
    let full_mlp_elapsed_ns = final_stats
        .full_attention_mlp_layers
        .iter()
        .map(|entry| entry.block_elapsed_ns)
        .sum::<u128>();
    Ok(RunResult {
        receipt: json!({
            "format": ALL_LINEAR_LAYER_STACKS_V1_FREE_FORMAT,
            "mode": args.mode.label(),
            "identity": identity,
            "prompt": PROMPT,
            "prompt_token_ids": prompt_tokens,
            "generated_tokens": STEPS,
            "schedule": {
                "total_layers": 24,
                "linear_attention_complete_layer_stacks": LINEAR_LAYER_STACK3_INDICES,
                "full_attention_cpu_attention_metal_mlp_layers": FULL_ATTENTION_LAYER_INDICES,
                "stack_count": 6,
                "layers_per_stack": 3,
                "linear_layer_count": 18,
                "full_attention_layer_count": 6,
                "duplicate_mlp_execution": false,
            },
            "per_stack_transaction_contract": {
                "command_buffers": 1,
                "compute_encoders": 3,
                "commits": 1,
                "waits": 1,
                "state_commits": 3,
                "state_commit_mask": 0b111,
                "intermediate_host_finite_checks": false,
                "final_output_finite_checks": true,
                "terminal_error": false,
            },
            "cpu_free_receipt": gate_evidence::attestation_json(&input_attestation),
            "cpu_generated_token_ids": cpu_generated,
            "metal_w8_all_linear_layer_stacks_v1_generated_token_ids": generated,
            "mismatches": mismatches,
            "first_mismatch": first_mismatch,
            "prefill_aggregate_path_receipt": stack3_stats_json(&prefill_stats),
            "final_aggregate_path_receipt": stack3_stats_json(&final_stats),
            "prefill_generation_path_receipt": prefill_generation_receipt,
            "final_generation_path_receipt": final_generation_receipt,
            "generation_path_contract": {
                "schema": "apxinf-qwen35-linear-layer-stacks-generation-path-v1",
                "versioned_stack3_semantics": true,
                "binds_six_three_layer_stacks_and_six_full_attention_mlp_layers": true,
                "intermediate_host_finite_checks": false,
                "final_output_finite_checks": true,
            },
            "aggregate_buffer_ledger": stack3_aggregate_ledger_json(&aggregate),
            "path_checks": {
                "prefill": prefill_checks.receipt_json(),
                "decode": final_checks.receipt_json(),
                "prefill_generation_receipt_valid": prefill_generation_valid,
                "decode_generation_receipt_valid": final_generation_valid,
                "exact_trajectory": mismatches.is_empty(),
            },
            "timing": {
                "setup": setup,
                "prefill_ms": prefill_ms,
                "decode_ms": decode_ms,
                "decode_mean_ms": decode_ms / expected_decode as f64,
                "six_stack3_blocks_mean_us": stack_elapsed_ns as f64
                    / expected_decode as f64 / 1000.0,
                "full_attention_mlp_blocks_mean_us": full_mlp_elapsed_ns as f64
                    / expected_decode as f64 / 1000.0,
                "classification": "candidate-only single pass under an uncontrolled host; never promotion evidence",
            },
            "passed": passed,
        }),
        passed,
    })
}

fn parse_args() -> Result<Args, String> {
    let mut model_dir = None;
    let mut source_lock = None;
    let mut mode = None;
    let mut input_receipt = None;
    let mut output = None;
    let mut layer = 0usize;
    let mut layer_set = false;
    let mut iter = std::env::args_os().skip(1);
    while let Some(raw_flag) = iter.next() {
        let flag = raw_flag.to_string_lossy();
        let value = |iter: &mut dyn Iterator<Item = OsString>| {
            iter.next()
                .ok_or_else(|| format!("{flag} requires a value"))
        };
        match flag.as_ref() {
            "--model-dir" => model_dir = Some(PathBuf::from(value(&mut iter)?)),
            "--source-lock" => source_lock = Some(PathBuf::from(value(&mut iter)?)),
            "--mode" => mode = Some(Mode::parse(&value(&mut iter)?.to_string_lossy())?),
            "--input-receipt" => input_receipt = Some(PathBuf::from(value(&mut iter)?)),
            "--output" => output = Some(PathBuf::from(value(&mut iter)?)),
            "--layer" => {
                if layer_set {
                    return Err("--layer may be specified only once".into());
                }
                layer = value(&mut iter)?
                    .to_string_lossy()
                    .parse()
                    .map_err(|error| format!("invalid --layer: {error}"))?;
                layer_set = true;
            }
            "-h" | "--help" => {
                println!("{}", usage());
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument {other}\n{}", usage())),
        }
    }
    let mode = mode.ok_or_else(|| format!("--mode is required\n{}", usage()))?;
    validate_layer_argument(mode, layer_set)?;
    if mode.requires_input() != input_receipt.is_some() {
        return Err(if mode.requires_input() {
            "--input-receipt is required for linear-layer-* modes".into()
        } else {
            "--input-receipt is not accepted for cpu-* modes".into()
        });
    }
    Ok(Args {
        model_dir: model_dir.ok_or_else(|| format!("--model-dir is required\n{}", usage()))?,
        source_lock: source_lock
            .ok_or_else(|| format!("--source-lock is required\n{}", usage()))?,
        mode,
        input_receipt,
        output: output.ok_or_else(|| format!("--output is required\n{}", usage()))?,
        layer,
    })
}

fn validate_layer_argument(mode: Mode, explicitly_set: bool) -> Result<(), String> {
    if (mode.is_all_linear_layers_precision_v2() || mode.is_all_linear_layer_stacks_v1())
        && explicitly_set
    {
        return Err("--layer is not accepted for all-linear-layers-* modes".into());
    }
    Ok(())
}

fn validate_official_all_linear_schedule(layer_types: &[Qwen35LayerType]) -> Result<(), String> {
    if layer_types.len() != ALL_LINEAR_LAYER_INDICES.len() + FULL_ATTENTION_LAYER_INDICES.len() {
        return Err(format!(
            "all-linear-layers gate requires the official 24-layer schedule, got {} layers",
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
            "all-linear-layers gate requires linear={ALL_LINEAR_LAYER_INDICES:?} and full={FULL_ATTENTION_LAYER_INDICES:?}, got linear={linear:?}, full={full:?}"
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

#[allow(clippy::too_many_arguments)]
fn validate_cpu_receipt(
    receipt: &Value,
    format: &str,
    mode: &str,
    identity: &Value,
    prompt_tokens: &[u32],
    layer: usize,
    prefill_token: Option<u32>,
    first_array: &str,
    second_array: &str,
) -> Result<(), Box<dyn Error>> {
    if receipt.get("passed").and_then(Value::as_bool) != Some(true) {
        return Err("CPU receipt passed must be true".into());
    }
    let identity_matches = receipt
        .get("identity")
        .is_some_and(|candidate| candidate == identity);
    if receipt.get("format").and_then(Value::as_str) != Some(format)
        || receipt.get("mode").and_then(Value::as_str) != Some(mode)
        || !identity_matches
        || receipt.get("prompt").and_then(Value::as_str) != Some(PROMPT)
        || json_u32_array(receipt, "prompt_token_ids")? != prompt_tokens
        || receipt.get("selected_layer").and_then(Value::as_u64) != Some(layer as u64)
    {
        return Err("CPU receipt does not match this exact linear-layer gate request".into());
    }
    if let Some(expected) = prefill_token {
        if receipt.get("prefill_token").and_then(Value::as_u64) != Some(expected as u64)
            || receipt.get("comparisons").and_then(Value::as_u64) != Some(STEPS as u64)
        {
            return Err("CPU teacher prefill/comparison contract mismatch".into());
        }
    } else if receipt.get("generated_tokens").and_then(Value::as_u64) != Some(STEPS as u64) {
        return Err("CPU free-run length contract mismatch".into());
    }
    if json_u32_array(receipt, first_array)?.len() != STEPS
        || json_u32_array(receipt, second_array)?.len() != STEPS
    {
        return Err("CPU receipt trajectory length does not match the frozen 128-step gate".into());
    }
    Ok(())
}

fn json_u32_array(value: &Value, key: &str) -> Result<Vec<u32>, Box<dyn Error>> {
    value
        .get(key)
        .and_then(Value::as_array)
        .ok_or_else(|| format!("receipt field {key} must be an array"))?
        .iter()
        .enumerate()
        .map(|(index, value)| {
            value
                .as_u64()
                .ok_or_else(|| format!("receipt field {key}[{index}] must be unsigned"))
                .and_then(|value| {
                    u32::try_from(value)
                        .map_err(|_| format!("receipt field {key}[{index}] exceeds u32"))
                })
                .map_err(Into::into)
        })
        .collect()
}

fn linear_layer_stats_json(stats: Qwen35MetalW8LinearLayerStats) -> Value {
    json!({
        "layer_index": stats.layer_index,
        "prefill_seed_calls": stats.prefill_seed_calls,
        "decode_calls": stats.decode_calls,
        "successful_decodes": stats.successful_decodes,
        "failed_decodes": stats.failed_decodes,
        "command_buffers": stats.command_buffers,
        "compute_encoders": stats.compute_encoders,
        "commits": stats.commits,
        "waits": stats.waits,
        "host_to_device_bytes": stats.host_to_device_bytes,
        "device_to_host_bytes": stats.device_to_host_bytes,
        "committed_state_version": stats.committed_state_version,
        "terminal_error": stats.terminal_error,
        "block_elapsed_ns": stats.block_elapsed_ns,
    })
}

fn precision_v2_stats_json(stats: Qwen35MetalW8LinearLayerPrecisionV2Stats) -> Value {
    let quantization = stats.quantization;
    json!({
        "profile": stats.profile.as_str(),
        "mechanism": stats.mechanism,
        "quantization": {
            "gdn_input": {
                "group_size": quantization.gdn_input_group_size.columns(),
                "packed_weight_bytes": quantization.gdn_input_weight_bytes,
                "scale_bytes": quantization.gdn_input_scale_bytes,
            },
            "gdn_output": {
                "group_size": quantization.gdn_output_group_size.columns(),
                "packed_weight_bytes": quantization.gdn_output_weight_bytes,
                "scale_bytes": quantization.gdn_output_scale_bytes,
            },
            "mlp_gate": {
                "group_size": quantization.mlp_gate_group_size.columns(),
                "packed_weight_bytes": quantization.mlp_gate_weight_bytes,
                "scale_bytes": quantization.mlp_gate_scale_bytes,
            },
            "mlp_up": {
                "group_size": quantization.mlp_up_group_size.columns(),
                "packed_weight_bytes": quantization.mlp_up_weight_bytes,
                "scale_bytes": quantization.mlp_up_scale_bytes,
            },
            "mlp_down": {
                "group_size": quantization.mlp_down_group_size.columns(),
                "packed_weight_bytes": quantization.mlp_down_weight_bytes,
                "scale_bytes": quantization.mlp_down_scale_bytes,
            },
            "total_packed_weight_bytes": quantization.total_packed_weight_bytes,
            "total_packed_scale_bytes": quantization.total_packed_scale_bytes,
        },
        "execution": linear_layer_stats_json(stats.execution),
    })
}

fn attach_precision_v2_receipts(
    receipt: &mut Value,
    prefill: Option<Value>,
    final_stats: Option<Qwen35MetalW8LinearLayerPrecisionV2Stats>,
    precision_v2_valid: bool,
) -> Result<(), Box<dyn Error>> {
    match (prefill, final_stats) {
        (None, None) => Ok(()),
        (Some(prefill), Some(stats)) => {
            let object = receipt
                .as_object_mut()
                .ok_or("precision-v2 receipt root must be an object")?;
            object.insert("precision_v2_prefill_receipt".into(), prefill);
            object.insert(
                "precision_v2_receipt".into(),
                precision_v2_stats_json(stats),
            );
            receipt["path_checks"]["precision_v2_valid"] = json!(precision_v2_valid);
            Ok(())
        }
        _ => Err("precision-v2 prefill and final receipts must both be present".into()),
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct AllLinearPathChecks {
    schedule_valid: bool,
    precision_profile_valid: bool,
    linear_execution_valid: bool,
    full_attention_mlp_execution_valid: bool,
    no_duplicate_mlp: bool,
    aggregate_ledger_valid: bool,
    terminal_clear: bool,
}

impl AllLinearPathChecks {
    fn all_valid(self) -> bool {
        self.schedule_valid
            && self.precision_profile_valid
            && self.linear_execution_valid
            && self.full_attention_mlp_execution_valid
            && self.no_duplicate_mlp
            && self.aggregate_ledger_valid
            && self.terminal_clear
    }

    fn receipt_json(self) -> Value {
        json!({
            "schedule_valid": self.schedule_valid,
            "precision_profile_valid": self.precision_profile_valid,
            "linear_execution_valid": self.linear_execution_valid,
            "full_attention_mlp_execution_valid": self.full_attention_mlp_execution_valid,
            "no_duplicate_mlp": self.no_duplicate_mlp,
            "aggregate_ledger_valid": self.aggregate_ledger_valid,
            "terminal_clear": self.terminal_clear,
            "all_valid": self.all_valid(),
        })
    }
}

fn all_linear_path_checks(
    stats: &Qwen35MetalW8AllLinearLayersPrecisionV2Stats,
    aggregate: &Qwen35MetalW8AllLinearLayersPrecisionV2AggregateLedger,
    expected_decode_calls: usize,
) -> AllLinearPathChecks {
    let linear_stats_indices = stats
        .linear_layers
        .iter()
        .map(|entry| entry.execution.layer_index)
        .collect::<Vec<_>>();
    let full_stats_indices = stats
        .full_attention_mlp_layers
        .iter()
        .map(|entry| entry.layer_index)
        .collect::<Vec<_>>();
    let linear_ledger_indices = aggregate
        .linear_layers
        .iter()
        .map(|entry| entry.layer_index)
        .collect::<Vec<_>>();
    let full_ledger_indices = aggregate
        .full_attention_mlp_layers
        .iter()
        .map(|entry| entry.layer_index)
        .collect::<Vec<_>>();
    let schedule_valid = linear_stats_indices == ALL_LINEAR_LAYER_INDICES
        && full_stats_indices == FULL_ATTENTION_LAYER_INDICES
        && linear_ledger_indices == ALL_LINEAR_LAYER_INDICES
        && full_ledger_indices == FULL_ATTENTION_LAYER_INDICES;
    let precision_profile_valid = stats.profile
        == Qwen35MetalW8LinearLayerPrecisionProfile::GdnOutG32V2
        && stats.mechanism == "metal-w8-all-linear-layers-precision-v2"
        && stats.full_attention_mlp_mechanism == "metal-w8-mlp-block-g64"
        && stats.linear_layers.iter().all(|entry| {
            entry.profile == Qwen35MetalW8LinearLayerPrecisionProfile::GdnOutG32V2
                && entry.mechanism == "metal-w8-linear-layer-precision-v2"
                && entry.quantization.gdn_input_group_size == apxinf_metal::W8GroupSize::G64
                && entry.quantization.gdn_output_group_size == apxinf_metal::W8GroupSize::G32
                && entry.quantization.mlp_gate_group_size == apxinf_metal::W8GroupSize::G64
                && entry.quantization.mlp_up_group_size == apxinf_metal::W8GroupSize::G64
                && entry.quantization.mlp_down_group_size == apxinf_metal::W8GroupSize::G64
        });
    let linear_execution_valid = stats
        .linear_layers
        .iter()
        .zip(&aggregate.linear_layers)
        .all(|(stats, ledger)| {
            let execution = stats.execution;
            execution.layer_index == ledger.layer_index
                && execution.prefill_seed_calls == 1
                && execution.decode_calls == expected_decode_calls
                && execution.successful_decodes == expected_decode_calls
                && execution.failed_decodes == 0
                && execution.command_buffers == expected_decode_calls
                && execution.compute_encoders == expected_decode_calls
                && execution.commits == expected_decode_calls
                && execution.waits == expected_decode_calls
                && execution.host_to_device_bytes
                    == ledger.ledger.host_input_bytes_per_decode * expected_decode_calls
                && execution.device_to_host_bytes
                    == ledger.ledger.host_output_bytes_per_decode * expected_decode_calls
                && execution.committed_state_version == expected_decode_calls as u64
                && !execution.terminal_error
                && stats.quantization.total_packed_weight_bytes == ledger.ledger.packed_weight_bytes
                && stats.quantization.total_packed_scale_bytes == ledger.ledger.packed_scale_bytes
        });
    let full_attention_mlp_execution_valid = stats
        .full_attention_mlp_layers
        .iter()
        .all(|entry| entry.decode_calls == expected_decode_calls);
    let no_duplicate_mlp = linear_stats_indices
        .iter()
        .all(|index| !full_stats_indices.contains(index))
        && linear_stats_indices.len() + full_stats_indices.len() == 24;
    let linear_total = aggregate
        .linear_layers
        .iter()
        .map(|entry| entry.ledger.total_persistent_bytes)
        .sum::<usize>();
    let mlp_total = aggregate
        .full_attention_mlp_layers
        .iter()
        .map(|entry| entry.ledger.total_persistent_bytes)
        .sum::<usize>();
    let aggregate_ledger_valid = aggregate.scope == "resident-mtlbuffer-only"
        && !aggregate.exclusions.is_empty()
        && !aggregate.includes_lm_head
        && aggregate
            .linear_layers
            .iter()
            .all(|entry| official_linear_layer_ledger_is_exact(entry.ledger))
        && aggregate
            .full_attention_mlp_layers
            .iter()
            .all(|entry| official_mlp_block_ledger_is_exact(entry.ledger))
        && linear_total + mlp_total == OFFICIAL_ALL_LINEAR_PERSISTENT_BYTES
        && aggregate.total_persistent_mtlbuffer_bytes == OFFICIAL_ALL_LINEAR_PERSISTENT_BYTES
        && aggregate.allocated_buffers == 624
        && aggregate.shared_buffers == 468
        && aggregate.private_buffers == 156
        && aggregate.host_to_device_bytes_per_decode == OFFICIAL_HIDDEN_BYTES * 24
        && aggregate.device_to_host_bytes_per_decode == OFFICIAL_HIDDEN_BYTES * 24
        && aggregate.state_host_transfer_bytes_per_decode == 0
        && aggregate.command_buffers_per_decode == 24
        && aggregate.compute_encoders_per_decode == 36
        && aggregate.commits_per_decode == 24
        && aggregate.waits_per_decode == 24;
    AllLinearPathChecks {
        schedule_valid,
        precision_profile_valid,
        linear_execution_valid,
        full_attention_mlp_execution_valid,
        no_duplicate_mlp,
        aggregate_ledger_valid,
        terminal_clear: !stats.terminal_error,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Stack3PathChecks {
    schedule_valid: bool,
    precision_contract_valid: bool,
    stack_execution_valid: bool,
    full_attention_mlp_execution_valid: bool,
    no_duplicate_mlp: bool,
    aggregate_ledger_valid: bool,
    finite_check_contract_valid: bool,
    terminal_clear: bool,
}

impl Stack3PathChecks {
    fn all_valid(self) -> bool {
        self.schedule_valid
            && self.precision_contract_valid
            && self.stack_execution_valid
            && self.full_attention_mlp_execution_valid
            && self.no_duplicate_mlp
            && self.aggregate_ledger_valid
            && self.finite_check_contract_valid
            && self.terminal_clear
    }

    fn receipt_json(self) -> Value {
        json!({
            "schedule_valid": self.schedule_valid,
            "precision_contract_valid": self.precision_contract_valid,
            "stack_execution_valid": self.stack_execution_valid,
            "full_attention_mlp_execution_valid": self.full_attention_mlp_execution_valid,
            "no_duplicate_mlp": self.no_duplicate_mlp,
            "aggregate_ledger_valid": self.aggregate_ledger_valid,
            "intermediate_host_finite_checks": false,
            "final_output_finite_checks": true,
            "finite_check_contract_valid": self.finite_check_contract_valid,
            "terminal_clear": self.terminal_clear,
            "all_valid": self.all_valid(),
        })
    }
}

fn official_stack3_ledger_is_exact(ledger: apxinf_metal::LinearLayerStack3BufferLedger) -> bool {
    ledger.allocated_buffers == 76
        && ledger.shared_buffers == 68
        && ledger.private_buffers == 8
        && ledger.packed_weight_bytes == OFFICIAL_STACK3_PACKED_WEIGHT_BYTES
        && ledger.packed_scale_bytes == OFFICIAL_STACK3_PACKED_SCALE_BYTES
        && ledger.f32_parameter_bytes == 321_408
        && ledger.active_state_bytes == 3_440_640
        && ledger.scratch_state_bytes == 3_440_640
        && ledger.activation_bytes == 133_248
        && ledger.total_persistent_bytes == OFFICIAL_STACK3_PERSISTENT_BYTES
        && ledger.host_input_bytes_per_decode == OFFICIAL_HIDDEN_BYTES
        && ledger.host_output_bytes_per_decode == OFFICIAL_HIDDEN_BYTES
        && ledger.state_host_transfer_bytes_per_decode == 0
        && ledger.command_buffers_per_decode == 1
        && ledger.compute_encoders_per_decode == 3
        && ledger.commits_per_decode == 1
        && ledger.waits_per_decode == 1
        && ledger.intermediate_host_finite_checks_per_decode == 0
        && ledger.final_output_finite_checks_per_decode == 1
}

fn stack3_path_checks(
    stats: &Qwen35MetalW8LinearLayerStacksV1Stats,
    aggregate: &Qwen35MetalW8LinearLayerStacksV1AggregateLedger,
    expected_decode_calls: usize,
) -> Stack3PathChecks {
    let stats_stack_indices = stats
        .stacks
        .iter()
        .map(|entry| entry.layer_indices)
        .collect::<Vec<_>>();
    let stats_full_indices = stats
        .full_attention_mlp_layers
        .iter()
        .map(|entry| entry.layer_index)
        .collect::<Vec<_>>();
    let ledger_stack_indices = aggregate
        .stacks
        .iter()
        .map(|entry| entry.layer_indices)
        .collect::<Vec<_>>();
    let ledger_full_indices = aggregate
        .full_attention_mlp_layers
        .iter()
        .map(|entry| entry.layer_index)
        .collect::<Vec<_>>();
    let schedule_valid = stats_stack_indices == LINEAR_LAYER_STACK3_INDICES
        && stats_full_indices == FULL_ATTENTION_LAYER_INDICES
        && ledger_stack_indices == LINEAR_LAYER_STACK3_INDICES
        && ledger_full_indices == FULL_ATTENTION_LAYER_INDICES;
    let precision_contract_valid = stats.mechanism == "metal-w8-linear-layer-stack3-v1"
        && stats.full_attention_mlp_mechanism == "metal-w8-mlp-block-g64"
        && stats
            .stacks
            .iter()
            .zip(&aggregate.stacks)
            .all(|(stack, ledger)| {
                stack.mechanism == "metal-w8-linear-layer-stack3-v1"
                    && stack.quantization.iter().all(|quantization| {
                        quantization.gdn_input_group_size == apxinf_metal::W8GroupSize::G64
                            && quantization.gdn_output_group_size == apxinf_metal::W8GroupSize::G32
                            && quantization.mlp_gate_group_size == apxinf_metal::W8GroupSize::G64
                            && quantization.mlp_up_group_size == apxinf_metal::W8GroupSize::G64
                            && quantization.mlp_down_group_size == apxinf_metal::W8GroupSize::G64
                    })
                    && stack
                        .quantization
                        .iter()
                        .map(|entry| entry.total_packed_weight_bytes)
                        .sum::<usize>()
                        == ledger.ledger.packed_weight_bytes
                    && stack
                        .quantization
                        .iter()
                        .map(|entry| entry.total_packed_scale_bytes)
                        .sum::<usize>()
                        == ledger.ledger.packed_scale_bytes
            });
    let expected_state_commits = expected_decode_calls.checked_mul(3);
    let expected_encoders = expected_decode_calls.checked_mul(3);
    let expected_transfer = OFFICIAL_HIDDEN_BYTES.checked_mul(expected_decode_calls);
    let expected_mask = if expected_decode_calls == 0 { 0 } else { 0b111 };
    let stack_execution_valid = expected_state_commits.is_some()
        && expected_encoders.is_some()
        && expected_transfer.is_some()
        && stats.stacks.iter().all(|stack| {
            let execution = stack.execution;
            stack.prefill_seed_calls == [1, 1, 1]
                && execution.decode_calls == expected_decode_calls
                && execution.successful_decodes == expected_decode_calls
                && execution.failed_decodes == 0
                && execution.command_buffers == expected_decode_calls
                && execution.compute_encoders == expected_encoders.unwrap()
                && execution.commits == expected_decode_calls
                && execution.waits == expected_decode_calls
                && execution.host_to_device_bytes == expected_transfer.unwrap()
                && execution.device_to_host_bytes == expected_transfer.unwrap()
                && execution.state_commits == expected_state_commits.unwrap()
                && execution.last_state_commit_mask == expected_mask
                && execution.committed_stack_version == expected_decode_calls as u64
                && !execution.terminal_error
                && !stack.terminal_error
        });
    let full_attention_mlp_execution_valid = stats
        .full_attention_mlp_layers
        .iter()
        .all(|entry| entry.decode_calls == expected_decode_calls);
    let flattened_linear = stats_stack_indices
        .iter()
        .flatten()
        .copied()
        .collect::<Vec<_>>();
    let no_duplicate_mlp = flattened_linear == ALL_LINEAR_LAYER_INDICES
        && flattened_linear
            .iter()
            .all(|index| !stats_full_indices.contains(index))
        && flattened_linear.len() + stats_full_indices.len() == 24;
    let stack_total = aggregate
        .stacks
        .iter()
        .map(|entry| entry.ledger.total_persistent_bytes)
        .sum::<usize>();
    let full_mlp_total = aggregate
        .full_attention_mlp_layers
        .iter()
        .map(|entry| entry.ledger.total_persistent_bytes)
        .sum::<usize>();
    let aggregate_ledger_valid = aggregate.scope == "resident-mtlbuffer-only"
        && !aggregate.exclusions.is_empty()
        && !aggregate.includes_lm_head
        && aggregate
            .stacks
            .iter()
            .all(|entry| official_stack3_ledger_is_exact(entry.ledger))
        && aggregate
            .full_attention_mlp_layers
            .iter()
            .all(|entry| official_mlp_block_ledger_is_exact(entry.ledger))
        && stack_total + full_mlp_total == OFFICIAL_STACK3_BODY_PERSISTENT_BYTES
        && aggregate.total_persistent_mtlbuffer_bytes == OFFICIAL_STACK3_BODY_PERSISTENT_BYTES
        && aggregate.allocated_buffers == 504
        && aggregate.shared_buffers == 444
        && aggregate.private_buffers == 60
        && aggregate.host_to_device_bytes_per_decode == 49_152
        && aggregate.device_to_host_bytes_per_decode == 49_152
        && aggregate.state_host_transfer_bytes_per_decode == 0
        && aggregate.command_buffers_per_decode == 12
        && aggregate.compute_encoders_per_decode == 36
        && aggregate.commits_per_decode == 12
        && aggregate.waits_per_decode == 12
        && aggregate.intermediate_host_finite_checks_per_decode == 0
        && aggregate.final_output_finite_checks_per_decode == 6;
    let finite_check_contract_valid = stats.stacks.iter().all(|stack| {
        stack.intermediate_host_finite_checks_per_decode == 0
            && stack.final_output_finite_checks_per_decode == 1
    });
    Stack3PathChecks {
        schedule_valid,
        precision_contract_valid,
        stack_execution_valid,
        full_attention_mlp_execution_valid,
        no_duplicate_mlp,
        aggregate_ledger_valid,
        finite_check_contract_valid,
        terminal_clear: !stats.terminal_error,
    }
}

fn stack3_generation_receipt_is_exact(receipt: &Value, expected_decode_calls: usize) -> bool {
    if receipt.get("format").and_then(Value::as_str)
        != Some("apxinf-qwen35-linear-layer-stacks-generation-path-v1")
        || receipt.get("mechanism").and_then(Value::as_str)
            != Some("metal-w8-linear-layer-stack3-v1")
        || receipt
            .get("full_attention_mlp_mechanism")
            .and_then(Value::as_str)
            != Some("metal-w8-mlp-block-g64")
        || receipt
            .get("metal_w8_complete_linear_layer_stacks")
            .and_then(Value::as_bool)
            != Some(true)
        || receipt
            .get("metal_w8_full_attention_mlp_blocks")
            .and_then(Value::as_bool)
            != Some(true)
        || receipt.get("metal_w8_lm_head").and_then(Value::as_bool) != Some(false)
        || receipt
            .get("intermediate_host_finite_checks")
            .and_then(Value::as_bool)
            != Some(false)
        || receipt
            .get("final_output_finite_checks")
            .and_then(Value::as_bool)
            != Some(true)
        || receipt.get("terminal_error").and_then(Value::as_bool) != Some(false)
    {
        return false;
    }
    let Some(stacks) = receipt.get("stacks").and_then(Value::as_array) else {
        return false;
    };
    let Some(full) = receipt
        .get("full_attention_mlp_layers")
        .and_then(Value::as_array)
    else {
        return false;
    };
    if stacks.len() != LINEAR_LAYER_STACK3_INDICES.len()
        || full.len() != FULL_ATTENTION_LAYER_INDICES.len()
    {
        return false;
    }
    let expected_decode = expected_decode_calls as u64;
    let Some(expected_encoders) = expected_decode_calls
        .checked_mul(3)
        .map(|value| value as u64)
    else {
        return false;
    };
    let Some(expected_transfer) = OFFICIAL_HIDDEN_BYTES
        .checked_mul(expected_decode_calls)
        .map(|value| value as u64)
    else {
        return false;
    };
    let expected_mask = if expected_decode_calls == 0 { 0 } else { 0b111 };
    let stacks_valid =
        stacks
            .iter()
            .zip(LINEAR_LAYER_STACK3_INDICES)
            .all(|(entry, layer_indices)| {
                entry.get("layer_indices") == Some(&json!(layer_indices))
                    && entry.get("mechanism").and_then(Value::as_str)
                        == Some("metal-w8-linear-layer-stack3-v1")
                    && entry.get("gdn_output_group_sizes") == Some(&json!([32, 32, 32]))
                    && entry.get("prefill_seed_calls") == Some(&json!([1, 1, 1]))
                    && entry.get("decode_calls").and_then(Value::as_u64) == Some(expected_decode)
                    && entry.get("successful_decodes").and_then(Value::as_u64)
                        == Some(expected_decode)
                    && entry.get("failed_decodes").and_then(Value::as_u64) == Some(0)
                    && entry.get("command_buffers").and_then(Value::as_u64) == Some(expected_decode)
                    && entry.get("compute_encoders").and_then(Value::as_u64)
                        == Some(expected_encoders)
                    && entry.get("commits").and_then(Value::as_u64) == Some(expected_decode)
                    && entry.get("waits").and_then(Value::as_u64) == Some(expected_decode)
                    && entry.get("host_to_device_bytes").and_then(Value::as_u64)
                        == Some(expected_transfer)
                    && entry.get("device_to_host_bytes").and_then(Value::as_u64)
                        == Some(expected_transfer)
                    && entry.get("state_commits").and_then(Value::as_u64) == Some(expected_encoders)
                    && entry.get("last_state_commit_mask").and_then(Value::as_u64)
                        == Some(expected_mask)
                    && entry.get("committed_stack_version").and_then(Value::as_u64)
                        == Some(expected_decode)
                    && entry
                        .get("intermediate_host_finite_checks_per_decode")
                        .and_then(Value::as_u64)
                        == Some(0)
                    && entry
                        .get("final_output_finite_checks_per_decode")
                        .and_then(Value::as_u64)
                        == Some(1)
                    && entry.get("terminal_error").and_then(Value::as_bool) == Some(false)
            });
    let full_valid = full
        .iter()
        .zip(FULL_ATTENTION_LAYER_INDICES)
        .all(|(entry, layer_index)| {
            entry.get("layer_index").and_then(Value::as_u64) == Some(layer_index as u64)
                && entry.get("decode_calls").and_then(Value::as_u64) == Some(expected_decode)
        });
    stacks_valid && full_valid
}

fn all_linear_generation_receipt_is_exact(receipt: &Value, expected_decode_calls: usize) -> bool {
    if receipt.get("format").and_then(Value::as_str)
        != Some("apxinf-qwen35-all-linear-layers-generation-path-v2")
        || receipt.get("profile").and_then(Value::as_str) != Some("gdn-out-g32-v2")
        || receipt.get("mechanism").and_then(Value::as_str)
            != Some("metal-w8-all-linear-layers-precision-v2")
        || receipt
            .get("full_attention_mlp_mechanism")
            .and_then(Value::as_str)
            != Some("metal-w8-mlp-block-g64")
        || receipt
            .get("metal_w8_complete_linear_layers")
            .and_then(Value::as_bool)
            != Some(true)
        || receipt
            .get("metal_w8_full_attention_mlp_blocks")
            .and_then(Value::as_bool)
            != Some(true)
        || receipt.get("metal_w8_lm_head").and_then(Value::as_bool) != Some(false)
        || receipt.get("terminal_error").and_then(Value::as_bool) != Some(false)
    {
        return false;
    }
    let Some(linear) = receipt.get("linear_layers").and_then(Value::as_array) else {
        return false;
    };
    let Some(full) = receipt
        .get("full_attention_mlp_layers")
        .and_then(Value::as_array)
    else {
        return false;
    };
    if linear.len() != ALL_LINEAR_LAYER_INDICES.len()
        || full.len() != FULL_ATTENTION_LAYER_INDICES.len()
    {
        return false;
    }
    let expected_decode = expected_decode_calls as u64;
    let expected_transfer = (OFFICIAL_HIDDEN_BYTES * expected_decode_calls) as u64;
    let linear_valid = linear
        .iter()
        .zip(ALL_LINEAR_LAYER_INDICES)
        .all(|(entry, layer_index)| {
            entry.get("layer_index").and_then(Value::as_u64) == Some(layer_index as u64)
                && entry.get("profile").and_then(Value::as_str) == Some("gdn-out-g32-v2")
                && entry.get("mechanism").and_then(Value::as_str)
                    == Some("metal-w8-linear-layer-precision-v2")
                && entry.get("gdn_output_group_size").and_then(Value::as_u64) == Some(32)
                && entry.get("prefill_seed_calls").and_then(Value::as_u64) == Some(1)
                && entry.get("decode_calls").and_then(Value::as_u64) == Some(expected_decode)
                && entry.get("successful_decodes").and_then(Value::as_u64) == Some(expected_decode)
                && entry.get("failed_decodes").and_then(Value::as_u64) == Some(0)
                && entry.get("command_buffers").and_then(Value::as_u64) == Some(expected_decode)
                && entry.get("compute_encoders").and_then(Value::as_u64) == Some(expected_decode)
                && entry.get("commits").and_then(Value::as_u64) == Some(expected_decode)
                && entry.get("waits").and_then(Value::as_u64) == Some(expected_decode)
                && entry.get("host_to_device_bytes").and_then(Value::as_u64)
                    == Some(expected_transfer)
                && entry.get("device_to_host_bytes").and_then(Value::as_u64)
                    == Some(expected_transfer)
                && entry.get("committed_state_version").and_then(Value::as_u64)
                    == Some(expected_decode)
                && entry.get("terminal_error").and_then(Value::as_bool) == Some(false)
        });
    let full_valid = full
        .iter()
        .zip(FULL_ATTENTION_LAYER_INDICES)
        .all(|(entry, layer_index)| {
            entry.get("layer_index").and_then(Value::as_u64) == Some(layer_index as u64)
                && entry.get("decode_calls").and_then(Value::as_u64) == Some(expected_decode)
        });
    linear_valid && full_valid
}

fn official_linear_layer_ledger_is_exact(ledger: apxinf_metal::LinearLayerBufferLedger) -> bool {
    linear_layer_ledger_is_exact(ledger)
        && ledger.total_persistent_bytes == OFFICIAL_LINEAR_LAYER_PERSISTENT_BYTES
        && ledger.host_input_bytes_per_decode == OFFICIAL_HIDDEN_BYTES
        && ledger.host_output_bytes_per_decode == OFFICIAL_HIDDEN_BYTES
}

fn official_mlp_block_ledger_is_exact(ledger: apxinf_metal::MlpBlockBufferLedger) -> bool {
    ledger.scope == "resident-mtlbuffer-only"
        && ledger.allocated_buffers == 8
        && ledger.shared_buffers == 6
        && ledger.private_buffers == 2
        && ledger.total_persistent_bytes == OFFICIAL_MLP_BLOCK_PERSISTENT_BYTES
        && ledger.host_input_bytes_per_decode == OFFICIAL_HIDDEN_BYTES
        && ledger.host_output_bytes_per_decode == OFFICIAL_HIDDEN_BYTES
        && ledger.state_host_transfer_bytes_per_decode == 0
        && ledger.command_buffers_per_decode == 1
        && ledger.compute_encoders_per_decode == 3
        && ledger.commits_per_decode == 1
        && ledger.waits_per_decode == 1
}

fn all_linear_stats_json(stats: &Qwen35MetalW8AllLinearLayersPrecisionV2Stats) -> Value {
    json!({
        "profile": stats.profile.as_str(),
        "mechanism": stats.mechanism,
        "full_attention_mlp_mechanism": stats.full_attention_mlp_mechanism,
        "linear_layers": stats
            .linear_layers
            .iter()
            .copied()
            .map(precision_v2_stats_json)
            .collect::<Vec<_>>(),
        "full_attention_mlp_layers": stats
            .full_attention_mlp_layers
            .iter()
            .map(|entry| json!({
                "layer_index": entry.layer_index,
                "decode_calls": entry.decode_calls,
                "block_elapsed_ns": entry.block_elapsed_ns,
            }))
            .collect::<Vec<_>>(),
        "terminal_error": stats.terminal_error,
    })
}

fn stack3_quantization_json(quantization: apxinf_metal::LinearLayerQuantizationLedger) -> Value {
    json!({
        "gdn_input": {
            "group_size": quantization.gdn_input_group_size.columns(),
            "packed_weight_bytes": quantization.gdn_input_weight_bytes,
            "scale_bytes": quantization.gdn_input_scale_bytes,
        },
        "gdn_output": {
            "group_size": quantization.gdn_output_group_size.columns(),
            "packed_weight_bytes": quantization.gdn_output_weight_bytes,
            "scale_bytes": quantization.gdn_output_scale_bytes,
        },
        "mlp_gate": {
            "group_size": quantization.mlp_gate_group_size.columns(),
            "packed_weight_bytes": quantization.mlp_gate_weight_bytes,
            "scale_bytes": quantization.mlp_gate_scale_bytes,
        },
        "mlp_up": {
            "group_size": quantization.mlp_up_group_size.columns(),
            "packed_weight_bytes": quantization.mlp_up_weight_bytes,
            "scale_bytes": quantization.mlp_up_scale_bytes,
        },
        "mlp_down": {
            "group_size": quantization.mlp_down_group_size.columns(),
            "packed_weight_bytes": quantization.mlp_down_weight_bytes,
            "scale_bytes": quantization.mlp_down_scale_bytes,
        },
        "total_packed_weight_bytes": quantization.total_packed_weight_bytes,
        "total_packed_scale_bytes": quantization.total_packed_scale_bytes,
    })
}

fn stack3_stats_json(stats: &Qwen35MetalW8LinearLayerStacksV1Stats) -> Value {
    json!({
        "mechanism": stats.mechanism,
        "full_attention_mlp_mechanism": stats.full_attention_mlp_mechanism,
        "stacks": stats.stacks.iter().map(|stack| {
            let execution = stack.execution;
            json!({
                "layer_indices": stack.layer_indices,
                "mechanism": stack.mechanism,
                "quantization": stack.quantization.map(stack3_quantization_json),
                "prefill_seed_calls": stack.prefill_seed_calls,
                "execution": {
                    "decode_calls": execution.decode_calls,
                    "successful_decodes": execution.successful_decodes,
                    "failed_decodes": execution.failed_decodes,
                    "command_buffers": execution.command_buffers,
                    "compute_encoders": execution.compute_encoders,
                    "commits": execution.commits,
                    "waits": execution.waits,
                    "host_to_device_bytes": execution.host_to_device_bytes,
                    "device_to_host_bytes": execution.device_to_host_bytes,
                    "state_commits": execution.state_commits,
                    "last_state_commit_mask": execution.last_state_commit_mask,
                    "committed_stack_version": execution.committed_stack_version,
                    "terminal_error": execution.terminal_error,
                },
                "intermediate_host_finite_checks_per_decode": stack.intermediate_host_finite_checks_per_decode,
                "final_output_finite_checks_per_decode": stack.final_output_finite_checks_per_decode,
                "terminal_error": stack.terminal_error,
                "block_elapsed_ns": stack.block_elapsed_ns,
            })
        }).collect::<Vec<_>>(),
        "full_attention_mlp_layers": stats.full_attention_mlp_layers.iter().map(|entry| json!({
            "layer_index": entry.layer_index,
            "decode_calls": entry.decode_calls,
            "block_elapsed_ns": entry.block_elapsed_ns,
        })).collect::<Vec<_>>(),
        "terminal_error": stats.terminal_error,
    })
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

fn stack3_aggregate_ledger_json(ledger: &Qwen35MetalW8LinearLayerStacksV1AggregateLedger) -> Value {
    json!({
        "scope": ledger.scope,
        "exclusions": ledger.exclusions,
        "includes_lm_head": ledger.includes_lm_head,
        "stacks": ledger.stacks.iter().map(|entry| json!({
            "layer_indices": entry.layer_indices,
            "ledger": stack3_ledger_json(entry.ledger),
        })).collect::<Vec<_>>(),
        "full_attention_mlp_layers": ledger.full_attention_mlp_layers.iter().map(|entry| json!({
            "layer_index": entry.layer_index,
            "ledger": mlp_block_ledger_json(entry.ledger),
        })).collect::<Vec<_>>(),
        "total_persistent_mtlbuffer_bytes": ledger.total_persistent_mtlbuffer_bytes,
        "allocated_buffers": ledger.allocated_buffers,
        "shared_buffers": ledger.shared_buffers,
        "private_buffers": ledger.private_buffers,
        "host_to_device_bytes_per_decode": ledger.host_to_device_bytes_per_decode,
        "device_to_host_bytes_per_decode": ledger.device_to_host_bytes_per_decode,
        "state_host_transfer_bytes_per_decode": ledger.state_host_transfer_bytes_per_decode,
        "command_buffers_per_decode": ledger.command_buffers_per_decode,
        "compute_encoders_per_decode": ledger.compute_encoders_per_decode,
        "commits_per_decode": ledger.commits_per_decode,
        "waits_per_decode": ledger.waits_per_decode,
        "intermediate_host_finite_checks_per_decode": ledger.intermediate_host_finite_checks_per_decode,
        "final_output_finite_checks_per_decode": ledger.final_output_finite_checks_per_decode,
    })
}

fn mlp_block_ledger_json(ledger: apxinf_metal::MlpBlockBufferLedger) -> Value {
    json!({
        "scope": ledger.scope,
        "allocated_buffers": ledger.allocated_buffers,
        "shared_buffers": ledger.shared_buffers,
        "private_buffers": ledger.private_buffers,
        "packed_weight_bytes": ledger.packed_weight_bytes,
        "packed_scale_bytes": ledger.packed_scale_bytes,
        "activation_bytes": ledger.activation_bytes,
        "total_persistent_bytes": ledger.total_persistent_bytes,
        "host_input_bytes_per_decode": ledger.host_input_bytes_per_decode,
        "host_output_bytes_per_decode": ledger.host_output_bytes_per_decode,
        "state_host_transfer_bytes_per_decode": ledger.state_host_transfer_bytes_per_decode,
        "command_buffers_per_decode": ledger.command_buffers_per_decode,
        "compute_encoders_per_decode": ledger.compute_encoders_per_decode,
        "commits_per_decode": ledger.commits_per_decode,
        "waits_per_decode": ledger.waits_per_decode,
    })
}

fn all_linear_aggregate_ledger_json(
    ledger: &Qwen35MetalW8AllLinearLayersPrecisionV2AggregateLedger,
) -> Value {
    json!({
        "scope": ledger.scope,
        "exclusions": ledger.exclusions,
        "includes_lm_head": ledger.includes_lm_head,
        "linear_layers": ledger.linear_layers.iter().map(|entry| json!({
            "layer_index": entry.layer_index,
            "ledger": linear_layer_ledger_json(entry.ledger),
        })).collect::<Vec<_>>(),
        "full_attention_mlp_layers": ledger.full_attention_mlp_layers.iter().map(|entry| json!({
            "layer_index": entry.layer_index,
            "ledger": mlp_block_ledger_json(entry.ledger),
        })).collect::<Vec<_>>(),
        "total_persistent_mtlbuffer_bytes": ledger.total_persistent_mtlbuffer_bytes,
        "allocated_buffers": ledger.allocated_buffers,
        "shared_buffers": ledger.shared_buffers,
        "private_buffers": ledger.private_buffers,
        "host_to_device_bytes_per_decode": ledger.host_to_device_bytes_per_decode,
        "device_to_host_bytes_per_decode": ledger.device_to_host_bytes_per_decode,
        "state_host_transfer_bytes_per_decode": ledger.state_host_transfer_bytes_per_decode,
        "command_buffers_per_decode": ledger.command_buffers_per_decode,
        "compute_encoders_per_decode": ledger.compute_encoders_per_decode,
        "commits_per_decode": ledger.commits_per_decode,
        "waits_per_decode": ledger.waits_per_decode,
    })
}

fn linear_layer_ledger_is_exact(ledger: apxinf_metal::LinearLayerBufferLedger) -> bool {
    ledger.allocated_buffers == 32
        && ledger.shared_buffers == 24
        && ledger.private_buffers == 8
        && ledger.state_host_transfer_bytes_per_decode == 0
        && ledger.command_buffers_per_decode == 1
        && ledger.compute_encoders_per_decode == 1
        && ledger.commits_per_decode == 1
        && ledger.waits_per_decode == 1
}

fn linear_layer_ledger_json(ledger: apxinf_metal::LinearLayerBufferLedger) -> Value {
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
    })
}

fn publish_no_replace(path: &Path, receipt: &Value) -> Result<(), Box<dyn Error>> {
    let payload = serde_json::to_vec(receipt)?;
    let mut file = OpenOptions::new().write(true).create_new(true).open(path)?;
    file.write_all(&payload)?;
    file.write_all(b"\n")?;
    file.sync_all()?;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mode_parser_keeps_cpu_and_linear_layer_receipt_roles_separate() {
        assert_eq!(Mode::parse("cpu-teacher").unwrap(), Mode::CpuTeacher);
        assert!(!Mode::CpuFree.requires_input());
        assert!(Mode::LinearLayerTeacher.requires_input());
        let precision = Mode::parse("linear-layer-gdn-out-g32-v2-teacher").unwrap();
        assert_eq!(precision, Mode::LinearLayerGdnOutG32V2Teacher);
        assert!(precision.requires_input());
        assert!(precision.is_precision_v2());
        let precision_free = Mode::parse("linear-layer-gdn-out-g32-v2-free").unwrap();
        assert_eq!(precision_free, Mode::LinearLayerGdnOutG32V2Free);
        assert_eq!(
            precision_free.label(),
            "metal_w8_linear_layer_gdn_out_g32_v2_free_run"
        );
        assert!(precision_free.requires_input());
        assert!(precision_free.is_precision_v2());
        let all_teacher = Mode::parse("all-linear-layers-gdn-out-g32-v2-teacher").unwrap();
        assert_eq!(all_teacher, Mode::AllLinearLayersGdnOutG32V2Teacher);
        assert_eq!(
            all_teacher.label(),
            "metal_w8_all_linear_layers_gdn_out_g32_v2_teacher_forced"
        );
        assert!(all_teacher.requires_input());
        assert!(all_teacher.is_all_linear_layers_precision_v2());
        let all_free = Mode::parse("all-linear-layers-gdn-out-g32-v2-free").unwrap();
        assert_eq!(all_free, Mode::AllLinearLayersGdnOutG32V2Free);
        assert_eq!(
            all_free.label(),
            "metal_w8_all_linear_layers_gdn_out_g32_v2_free_run"
        );
        assert!(all_free.requires_input());
        assert!(all_free.is_all_linear_layers_precision_v2());
        let stack_teacher = Mode::parse("all-linear-layer-stacks-v1-teacher").unwrap();
        assert_eq!(stack_teacher, Mode::AllLinearLayerStacksV1Teacher);
        assert_eq!(
            stack_teacher.label(),
            "metal_w8_all_linear_layer_stacks_v1_teacher_forced"
        );
        assert!(stack_teacher.requires_input());
        assert!(stack_teacher.is_all_linear_layer_stacks_v1());
        let stack_free = Mode::parse("all-linear-layer-stacks-v1-free").unwrap();
        assert_eq!(stack_free, Mode::AllLinearLayerStacksV1Free);
        assert_eq!(
            stack_free.label(),
            "metal_w8_all_linear_layer_stacks_v1_free_run"
        );
        assert!(stack_free.requires_input());
        assert!(stack_free.is_all_linear_layer_stacks_v1());
        assert!(validate_layer_argument(all_teacher, true).is_err());
        assert!(validate_layer_argument(all_free, true).is_err());
        assert!(validate_layer_argument(stack_teacher, true).is_err());
        assert!(validate_layer_argument(stack_free, true).is_err());
        validate_layer_argument(all_teacher, false).unwrap();
        validate_layer_argument(stack_teacher, false).unwrap();
        validate_layer_argument(Mode::LinearLayerTeacher, true).unwrap();
        assert!(Mode::parse("combined").is_err());
    }

    #[test]
    fn all_linear_gate_schedule_is_frozen_to_official_18_plus_6_topology() {
        let schedule = (0..24)
            .map(|layer| {
                if layer % 4 == 3 {
                    Qwen35LayerType::FullAttention
                } else {
                    Qwen35LayerType::LinearAttention
                }
            })
            .collect::<Vec<_>>();
        validate_official_all_linear_schedule(&schedule).unwrap();

        let mut wrong = schedule;
        wrong[3] = Qwen35LayerType::LinearAttention;
        assert!(validate_official_all_linear_schedule(&wrong).is_err());
        assert!(validate_official_all_linear_schedule(&wrong[..23]).is_err());
    }

    #[test]
    fn all_linear_path_checks_bind_every_lane_and_exact_official_ledger() {
        let decode_calls = 3;
        let quantization = apxinf_metal::LinearLayerQuantizationLedger {
            gdn_input_group_size: apxinf_metal::W8GroupSize::G64,
            gdn_input_weight_bytes: 10,
            gdn_input_scale_bytes: 11,
            gdn_output_group_size: apxinf_metal::W8GroupSize::G32,
            gdn_output_weight_bytes: 20,
            gdn_output_scale_bytes: 22,
            mlp_gate_group_size: apxinf_metal::W8GroupSize::G64,
            mlp_gate_weight_bytes: 30,
            mlp_gate_scale_bytes: 31,
            mlp_up_group_size: apxinf_metal::W8GroupSize::G64,
            mlp_up_weight_bytes: 40,
            mlp_up_scale_bytes: 41,
            mlp_down_group_size: apxinf_metal::W8GroupSize::G64,
            mlp_down_weight_bytes: 50,
            mlp_down_scale_bytes: 51,
            total_packed_weight_bytes: 150,
            total_packed_scale_bytes: 156,
        };
        let linear_ledger = apxinf_metal::LinearLayerBufferLedger {
            allocated_buffers: 32,
            shared_buffers: 24,
            private_buffers: 8,
            packed_weight_bytes: 150,
            packed_scale_bytes: 156,
            f32_parameter_bytes: 0,
            active_state_bytes: 0,
            scratch_state_bytes: 0,
            activation_bytes: 0,
            total_persistent_bytes: OFFICIAL_LINEAR_LAYER_PERSISTENT_BYTES,
            host_input_bytes_per_decode: OFFICIAL_HIDDEN_BYTES,
            host_output_bytes_per_decode: OFFICIAL_HIDDEN_BYTES,
            state_host_transfer_bytes_per_decode: 0,
            command_buffers_per_decode: 1,
            compute_encoders_per_decode: 1,
            commits_per_decode: 1,
            waits_per_decode: 1,
        };
        let mlp_ledger = apxinf_metal::MlpBlockBufferLedger {
            scope: "resident-mtlbuffer-only",
            allocated_buffers: 8,
            shared_buffers: 6,
            private_buffers: 2,
            packed_weight_bytes: 0,
            packed_scale_bytes: 0,
            activation_bytes: 0,
            total_persistent_bytes: OFFICIAL_MLP_BLOCK_PERSISTENT_BYTES,
            host_input_bytes_per_decode: OFFICIAL_HIDDEN_BYTES,
            host_output_bytes_per_decode: OFFICIAL_HIDDEN_BYTES,
            state_host_transfer_bytes_per_decode: 0,
            command_buffers_per_decode: 1,
            compute_encoders_per_decode: 3,
            commits_per_decode: 1,
            waits_per_decode: 1,
        };
        let linear_layers = ALL_LINEAR_LAYER_INDICES
            .iter()
            .map(|&layer_index| Qwen35MetalW8LinearLayerPrecisionV2Stats {
                profile: Qwen35MetalW8LinearLayerPrecisionProfile::GdnOutG32V2,
                mechanism: "metal-w8-linear-layer-precision-v2",
                quantization,
                execution: Qwen35MetalW8LinearLayerStats {
                    layer_index,
                    prefill_seed_calls: 1,
                    decode_calls,
                    successful_decodes: decode_calls,
                    failed_decodes: 0,
                    command_buffers: decode_calls,
                    compute_encoders: decode_calls,
                    commits: decode_calls,
                    waits: decode_calls,
                    host_to_device_bytes: OFFICIAL_HIDDEN_BYTES * decode_calls,
                    device_to_host_bytes: OFFICIAL_HIDDEN_BYTES * decode_calls,
                    committed_state_version: decode_calls as u64,
                    terminal_error: false,
                    block_elapsed_ns: 1,
                },
            })
            .collect::<Vec<_>>();
        let full_attention_mlp_layers = FULL_ATTENTION_LAYER_INDICES
            .iter()
            .map(
                |&layer_index| apxinf_model::qwen35::general::Qwen35MetalW8MlpBlockStats {
                    layer_index,
                    decode_calls,
                    block_elapsed_ns: 1,
                },
            )
            .collect::<Vec<_>>();
        let stats = Qwen35MetalW8AllLinearLayersPrecisionV2Stats {
            profile: Qwen35MetalW8LinearLayerPrecisionProfile::GdnOutG32V2,
            mechanism: "metal-w8-all-linear-layers-precision-v2",
            full_attention_mlp_mechanism: "metal-w8-mlp-block-g64",
            linear_layers,
            full_attention_mlp_layers,
            terminal_error: false,
        };
        let ledger = Qwen35MetalW8AllLinearLayersPrecisionV2AggregateLedger {
            scope: "resident-mtlbuffer-only",
            exclusions: "CPU F32 weights, host Vec allocations, Metal pipelines/libraries/queues, driver allocations, KV cache, and lm_head",
            includes_lm_head: false,
            linear_layers: ALL_LINEAR_LAYER_INDICES
                .iter()
                .map(|&layer_index| Qwen35MetalW8LinearLayerBufferLedger {
                    layer_index,
                    ledger: linear_ledger,
                })
                .collect(),
            full_attention_mlp_layers: FULL_ATTENTION_LAYER_INDICES
                .iter()
                .map(|&layer_index| Qwen35MetalW8MlpBlockBufferLedger {
                    layer_index,
                    ledger: mlp_ledger,
                })
                .collect(),
            total_persistent_mtlbuffer_bytes: OFFICIAL_ALL_LINEAR_PERSISTENT_BYTES,
            allocated_buffers: 624,
            shared_buffers: 468,
            private_buffers: 156,
            host_to_device_bytes_per_decode: OFFICIAL_HIDDEN_BYTES * 24,
            device_to_host_bytes_per_decode: OFFICIAL_HIDDEN_BYTES * 24,
            state_host_transfer_bytes_per_decode: 0,
            command_buffers_per_decode: 24,
            compute_encoders_per_decode: 36,
            commits_per_decode: 24,
            waits_per_decode: 24,
        };

        let checks = all_linear_path_checks(&stats, &ledger, decode_calls);
        assert!(checks.all_valid());
        let generation_receipt = json!({
            "format": "apxinf-qwen35-all-linear-layers-generation-path-v2",
            "profile": "gdn-out-g32-v2",
            "mechanism": "metal-w8-all-linear-layers-precision-v2",
            "full_attention_mlp_mechanism": "metal-w8-mlp-block-g64",
            "metal_w8_complete_linear_layers": true,
            "metal_w8_full_attention_mlp_blocks": true,
            "metal_w8_lm_head": false,
            "linear_layers": stats.linear_layers.iter().map(|entry| {
                let execution = entry.execution;
                json!({
                    "layer_index": execution.layer_index,
                    "profile": "gdn-out-g32-v2",
                    "mechanism": "metal-w8-linear-layer-precision-v2",
                    "gdn_output_group_size": 32,
                    "prefill_seed_calls": execution.prefill_seed_calls,
                    "decode_calls": execution.decode_calls,
                    "successful_decodes": execution.successful_decodes,
                    "failed_decodes": execution.failed_decodes,
                    "command_buffers": execution.command_buffers,
                    "compute_encoders": execution.compute_encoders,
                    "commits": execution.commits,
                    "waits": execution.waits,
                    "host_to_device_bytes": execution.host_to_device_bytes,
                    "device_to_host_bytes": execution.device_to_host_bytes,
                    "committed_state_version": execution.committed_state_version,
                    "terminal_error": execution.terminal_error,
                })
            }).collect::<Vec<_>>(),
            "full_attention_mlp_layers": stats.full_attention_mlp_layers.iter().map(|entry| json!({
                "layer_index": entry.layer_index,
                "decode_calls": entry.decode_calls,
            })).collect::<Vec<_>>(),
            "terminal_error": false,
        });
        assert!(all_linear_generation_receipt_is_exact(
            &generation_receipt,
            decode_calls
        ));

        let mut wrong = stats;
        wrong.linear_layers[4].execution.waits -= 1;
        assert!(!all_linear_path_checks(&wrong, &ledger, decode_calls).all_valid());
    }

    #[test]
    fn stack3_path_checks_bind_six_stacks_six_full_layers_and_exact_transaction_ledger() {
        let decode_calls = 3;
        let quantization = apxinf_metal::LinearLayerQuantizationLedger {
            gdn_input_group_size: apxinf_metal::W8GroupSize::G64,
            gdn_input_weight_bytes: 1,
            gdn_input_scale_bytes: 2,
            gdn_output_group_size: apxinf_metal::W8GroupSize::G32,
            gdn_output_weight_bytes: 3,
            gdn_output_scale_bytes: 4,
            mlp_gate_group_size: apxinf_metal::W8GroupSize::G64,
            mlp_gate_weight_bytes: 5,
            mlp_gate_scale_bytes: 6,
            mlp_up_group_size: apxinf_metal::W8GroupSize::G64,
            mlp_up_weight_bytes: 7,
            mlp_up_scale_bytes: 8,
            mlp_down_group_size: apxinf_metal::W8GroupSize::G64,
            mlp_down_weight_bytes: 9,
            mlp_down_scale_bytes: 10,
            total_packed_weight_bytes: 21_528_576,
            total_packed_scale_bytes: 1_476_608,
        };
        let stack_ledger = apxinf_metal::LinearLayerStack3BufferLedger {
            gdn_core_profile: apxinf_metal::GdnCoreProfileV1::LegacyFourDispatch,
            gdn_function_chain: apxinf_metal::GdnCoreProfileV1::LegacyFourDispatch
                .expected_function_chain(),
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
            kernel_dispatches_per_decode: 39,
            explicit_buffer_barriers_per_decode: 36,
            gdn_core_seams_per_decode: 3,
            gdn_core_kernel_dispatches_per_decode: 12,
            gdn_core_explicit_buffer_barriers_per_decode: 12,
            gdn_core_recurrent_or_fused_threads_per_threadgroup: 256,
            gdn_core_threadgroups_per_decode: 126,
            gdn_core_launched_threads_per_decode: 30_864,
            gdn_core_source_declared_threadgroup_memory_bytes: 0,
            gdn_core_expected_pipeline_static_threadgroup_memory_bytes: 0,
            gdn_core_internal_threadgroup_barrier_sites_per_threadgroup: 0,
            commits_per_decode: 1,
            waits_per_decode: 1,
            intermediate_host_finite_checks_per_decode: 0,
            final_output_finite_checks_per_decode: 1,
        };
        let mlp_ledger = apxinf_metal::MlpBlockBufferLedger {
            scope: "resident-mtlbuffer-only",
            allocated_buffers: 8,
            shared_buffers: 6,
            private_buffers: 2,
            packed_weight_bytes: 11_010_048,
            packed_scale_bytes: 688_128,
            activation_bytes: 51_200,
            total_persistent_bytes: 11_749_376,
            host_input_bytes_per_decode: 4_096,
            host_output_bytes_per_decode: 4_096,
            state_host_transfer_bytes_per_decode: 0,
            command_buffers_per_decode: 1,
            compute_encoders_per_decode: 3,
            commits_per_decode: 1,
            waits_per_decode: 1,
        };
        let stacks = LINEAR_LAYER_STACK3_INDICES
            .iter()
            .map(|&layer_indices| Qwen35MetalW8LinearLayerStack3V1Stats {
                layer_indices,
                mechanism: "metal-w8-linear-layer-stack3-v1",
                gdn_core_profile: apxinf_metal::GdnCoreProfileV1::LegacyFourDispatch,
                gdn_function_chain: apxinf_metal::GdnCoreProfileV1::LegacyFourDispatch
                    .expected_function_chain(),
                quantization: [quantization; 3],
                prefill_seed_calls: [1, 1, 1],
                execution: apxinf_metal::LinearLayerStack3MetalStats {
                    decode_calls,
                    successful_decodes: decode_calls,
                    failed_decodes: 0,
                    command_buffers: decode_calls,
                    compute_encoders: 3 * decode_calls,
                    commits: decode_calls,
                    waits: decode_calls,
                    host_to_device_bytes: 4_096 * decode_calls,
                    device_to_host_bytes: 4_096 * decode_calls,
                    state_commits: 3 * decode_calls,
                    last_state_commit_mask: 0b111,
                    committed_stack_version: decode_calls as u64,
                    last_gdn_core_receipt: None,
                    terminal_error: false,
                },
                last_gdn_core_receipt: None,
                kernel_dispatches_per_decode: 39,
                explicit_buffer_barriers_per_decode: 36,
                intermediate_host_finite_checks_per_decode: 0,
                final_output_finite_checks_per_decode: 1,
                terminal_error: false,
                block_elapsed_ns: 1,
            })
            .collect::<Vec<_>>();
        let full_attention_mlp_layers = FULL_ATTENTION_LAYER_INDICES
            .iter()
            .map(
                |&layer_index| apxinf_model::qwen35::general::Qwen35MetalW8MlpBlockStats {
                    layer_index,
                    decode_calls,
                    block_elapsed_ns: 1,
                },
            )
            .collect::<Vec<_>>();
        let stats = Qwen35MetalW8LinearLayerStacksV1Stats {
            mechanism: "metal-w8-linear-layer-stack3-v1",
            full_attention_mlp_mechanism: "metal-w8-mlp-block-g64",
            stacks,
            full_attention_mlp_layers,
            terminal_error: false,
        };
        let aggregate = Qwen35MetalW8LinearLayerStacksV1AggregateLedger {
            scope: "resident-mtlbuffer-only",
            exclusions: "CPU F32 weights, host Vec allocations, Metal pipelines/libraries/queues, driver allocations, KV cache, and lm_head",
            includes_lm_head: false,
            stacks: LINEAR_LAYER_STACK3_INDICES
                .iter()
                .map(|&layer_indices| Qwen35MetalW8LinearLayerStack3BufferLedger {
                    layer_indices,
                    ledger: stack_ledger,
                })
                .collect(),
            full_attention_mlp_layers: FULL_ATTENTION_LAYER_INDICES
                .iter()
                .map(|&layer_index| Qwen35MetalW8MlpBlockBufferLedger {
                    layer_index,
                    ledger: mlp_ledger,
                })
                .collect(),
            total_persistent_mtlbuffer_bytes: 528_605_184,
            allocated_buffers: 504,
            shared_buffers: 444,
            private_buffers: 60,
            host_to_device_bytes_per_decode: 49_152,
            device_to_host_bytes_per_decode: 49_152,
            state_host_transfer_bytes_per_decode: 0,
            command_buffers_per_decode: 12,
            compute_encoders_per_decode: 36,
            commits_per_decode: 12,
            waits_per_decode: 12,
            intermediate_host_finite_checks_per_decode: 0,
            final_output_finite_checks_per_decode: 6,
        };

        let checks = stack3_path_checks(&stats, &aggregate, decode_calls);
        assert!(checks.all_valid());
        let stats_receipt = stack3_stats_json(&stats);
        assert_eq!(
            stats_receipt["stacks"][0]["layer_indices"],
            json!([0, 1, 2])
        );
        assert_eq!(
            stats_receipt["stacks"][0]["execution"]["compute_encoders"],
            9
        );
        assert_eq!(stats_receipt["stacks"][0]["execution"]["state_commits"], 9);
        assert_eq!(
            stats_receipt["stacks"][0]["execution"]["last_state_commit_mask"],
            0b111
        );
        let ledger_receipt = stack3_aggregate_ledger_json(&aggregate);
        assert_eq!(ledger_receipt["allocated_buffers"], 504);
        assert_eq!(ledger_receipt["shared_buffers"], 444);
        assert_eq!(ledger_receipt["private_buffers"], 60);
        assert_eq!(
            ledger_receipt["total_persistent_mtlbuffer_bytes"],
            528_605_184
        );
        assert_eq!(ledger_receipt["command_buffers_per_decode"], 12);
        assert_eq!(ledger_receipt["compute_encoders_per_decode"], 36);
        assert_eq!(ledger_receipt["host_to_device_bytes_per_decode"], 49_152);
        assert_eq!(ledger_receipt["device_to_host_bytes_per_decode"], 49_152);
        let generation_receipt = json!({
            "format": "apxinf-qwen35-linear-layer-stacks-generation-path-v1",
            "mechanism": "metal-w8-linear-layer-stack3-v1",
            "full_attention_mlp_mechanism": "metal-w8-mlp-block-g64",
            "metal_w8_complete_linear_layer_stacks": true,
            "metal_w8_full_attention_mlp_blocks": true,
            "metal_w8_lm_head": false,
            "intermediate_host_finite_checks": false,
            "final_output_finite_checks": true,
            "stacks": stats.stacks.iter().map(|entry| {
                let execution = entry.execution;
                json!({
                    "layer_indices": entry.layer_indices,
                    "mechanism": entry.mechanism,
                    "gdn_output_group_sizes": [32, 32, 32],
                    "prefill_seed_calls": entry.prefill_seed_calls,
                    "decode_calls": execution.decode_calls,
                    "successful_decodes": execution.successful_decodes,
                    "failed_decodes": execution.failed_decodes,
                    "command_buffers": execution.command_buffers,
                    "compute_encoders": execution.compute_encoders,
                    "commits": execution.commits,
                    "waits": execution.waits,
                    "host_to_device_bytes": execution.host_to_device_bytes,
                    "device_to_host_bytes": execution.device_to_host_bytes,
                    "state_commits": execution.state_commits,
                    "last_state_commit_mask": execution.last_state_commit_mask,
                    "committed_stack_version": execution.committed_stack_version,
                    "intermediate_host_finite_checks_per_decode": 0,
                    "final_output_finite_checks_per_decode": 1,
                    "terminal_error": false,
                    "block_elapsed_ns": 1,
                })
            }).collect::<Vec<_>>(),
            "full_attention_mlp_layers": stats.full_attention_mlp_layers.iter().map(|entry| json!({
                "layer_index": entry.layer_index,
                "decode_calls": entry.decode_calls,
                "block_elapsed_ns": entry.block_elapsed_ns,
            })).collect::<Vec<_>>(),
            "terminal_error": false,
        });
        assert!(stack3_generation_receipt_is_exact(
            &generation_receipt,
            decode_calls
        ));

        let mut wrong_stats = stats.clone();
        wrong_stats.stacks[2].execution.state_commits -= 1;
        assert!(!stack3_path_checks(&wrong_stats, &aggregate, decode_calls).all_valid());
        let mut wrong_receipt = generation_receipt;
        wrong_receipt["stacks"][4]["last_state_commit_mask"] = json!(0);
        assert!(!stack3_generation_receipt_is_exact(
            &wrong_receipt,
            decode_calls
        ));
    }

    #[test]
    fn precision_v2_receipt_names_the_versioned_mechanism_and_exact_packing() {
        let execution = Qwen35MetalW8LinearLayerStats {
            layer_index: 0,
            prefill_seed_calls: 1,
            decode_calls: 2,
            successful_decodes: 2,
            failed_decodes: 0,
            command_buffers: 2,
            compute_encoders: 2,
            commits: 2,
            waits: 2,
            host_to_device_bytes: 512,
            device_to_host_bytes: 512,
            committed_state_version: 2,
            terminal_error: false,
            block_elapsed_ns: 7,
        };
        let stats = Qwen35MetalW8LinearLayerPrecisionV2Stats {
            profile: Qwen35MetalW8LinearLayerPrecisionProfile::GdnOutG32V2,
            mechanism: "metal-w8-linear-layer-precision-v2",
            quantization: apxinf_metal::LinearLayerQuantizationLedger {
                gdn_input_group_size: apxinf_metal::W8GroupSize::G64,
                gdn_input_weight_bytes: 10,
                gdn_input_scale_bytes: 11,
                gdn_output_group_size: apxinf_metal::W8GroupSize::G32,
                gdn_output_weight_bytes: 20,
                gdn_output_scale_bytes: 22,
                mlp_gate_group_size: apxinf_metal::W8GroupSize::G64,
                mlp_gate_weight_bytes: 30,
                mlp_gate_scale_bytes: 31,
                mlp_up_group_size: apxinf_metal::W8GroupSize::G64,
                mlp_up_weight_bytes: 40,
                mlp_up_scale_bytes: 41,
                mlp_down_group_size: apxinf_metal::W8GroupSize::G64,
                mlp_down_weight_bytes: 50,
                mlp_down_scale_bytes: 51,
                total_packed_weight_bytes: 150,
                total_packed_scale_bytes: 156,
            },
            execution,
        };

        let receipt = precision_v2_stats_json(stats);

        assert_eq!(receipt["profile"], "gdn-out-g32-v2");
        assert_eq!(receipt["mechanism"], "metal-w8-linear-layer-precision-v2");
        assert_eq!(receipt["quantization"]["gdn_input"]["group_size"], 64);
        assert_eq!(receipt["quantization"]["gdn_output"]["group_size"], 32);
        assert_eq!(receipt["quantization"]["gdn_output"]["scale_bytes"], 22);
        assert_eq!(receipt["quantization"]["mlp_down"]["group_size"], 64);
        assert_eq!(receipt["execution"]["decode_calls"], 2);

        let prefill = json!({"execution": {"decode_calls": 0}});
        let mut candidate = json!({"path_checks": {}});
        attach_precision_v2_receipts(&mut candidate, Some(prefill.clone()), Some(stats), true)
            .unwrap();
        assert_eq!(candidate["precision_v2_prefill_receipt"], prefill);
        assert_eq!(
            candidate["precision_v2_receipt"]["profile"],
            "gdn-out-g32-v2"
        );
        assert_eq!(
            candidate["precision_v2_receipt"]["execution"]["decode_calls"],
            2
        );
        assert_eq!(candidate["path_checks"]["precision_v2_valid"], true);
    }

    #[test]
    fn source_lock_validation_is_frozen_to_the_official_checkpoint() {
        let valid = json!({
            "format": SOURCE_LOCK_FORMAT,
            "repo_id": REPO_ID,
            "resolved_commit": LOCKED_REVISION,
            "weights": { "files": [{
                "path": LOCKED_CHECKPOINT,
                "sha256": LOCKED_CHECKPOINT_SHA256,
                "size": LOCKED_CHECKPOINT_BYTES,
            }]},
        });
        validate_source_lock(&valid).unwrap();
        let mut invalid = valid;
        invalid["resolved_commit"] = json!("moving-main");
        assert!(validate_source_lock(&invalid).is_err());
    }

    #[test]
    fn cpu_receipt_must_have_passed_the_frozen_gate() {
        let identity = json!({"exact": "identity"});
        let prompt_tokens = vec![1, 2, 3];
        let receipt = json!({
            "format": CPU_TEACHER_FORMAT,
            "mode": "linear_layer_cpu_teacher",
            "identity": identity.clone(),
            "prompt": PROMPT,
            "prompt_token_ids": prompt_tokens,
            "selected_layer": 0,
            "prefill_token": 7,
            "comparisons": STEPS,
            "teacher_input_ids": vec![7; STEPS],
            "cpu_expected_output_ids": vec![8; STEPS],
            "passed": false,
        });

        let error = validate_cpu_receipt(
            &receipt,
            CPU_TEACHER_FORMAT,
            "linear_layer_cpu_teacher",
            &identity,
            &[1, 2, 3],
            0,
            Some(7),
            "teacher_input_ids",
            "cpu_expected_output_ids",
        )
        .unwrap_err();

        assert!(error.to_string().contains("passed"));
    }
}
