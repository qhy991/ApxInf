//! Same-release-binary real-checkpoint quality gate for one diagnostic CPU
//! packed-W8 complete linear-attention reference layer. This example is deliberately absent from CLI/AutoModel and
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
    Qwen35PackedW8LinearLayerReferenceProfile, Qwen35PackedW8LinearLayerReferenceStats,
};
use apxinf_model::{GeneralQwen35, LlmInput, LlmTrait, Qwen35Config, Qwen35LayerType};
use apxinf_tokenizer::{ChatMessage, Tokenizer};
use serde_json::{json, Value};

const CPU_TEACHER_FORMAT: &str = "apxinf-qwen35-packed-w8-linear-layer-cpu-teacher-v1";
const PACKED_REFERENCE_TEACHER_FORMAT: &str =
    "apxinf-qwen35-packed-w8-linear-layer-reference-teacher-gate-v1";
const CPU_FREE_FORMAT: &str = "apxinf-qwen35-packed-w8-linear-layer-cpu-free-run-v1";
const PACKED_REFERENCE_FREE_FORMAT: &str =
    "apxinf-qwen35-packed-w8-linear-layer-reference-free-run-gate-v1";
const SOURCE_LOCK_FORMAT: &str = "apxinf-hf-source-lock-v1";
const REPO_ID: &str = "Qwen/Qwen3.5-0.8B";
const LOCKED_REVISION: &str = "2fc06364715b967f1860aea9cf38778875588b17";
const LOCKED_CHECKPOINT: &str = "model.safetensors-00001-of-00001.safetensors";
const LOCKED_CHECKPOINT_SHA256: &str =
    "04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696";
const LOCKED_CHECKPOINT_BYTES: u64 = 1_746_942_600;
const STEPS: usize = 128;
const PROMPT: &str = "Hello";
const GATE_SOURCE_NAME: &str = "qwen35_packed_w8_linear_layer_reference_gate.rs";
const GATE_SOURCE_BYTES: &[u8] = include_bytes!("qwen35_packed_w8_linear_layer_reference_gate.rs");

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mode {
    CpuTeacher,
    PackedReferenceTeacher,
    CpuFree,
    PackedReferenceFree,
}

impl Mode {
    fn parse(value: &str) -> Result<Self, String> {
        match value {
            "cpu-teacher" => Ok(Self::CpuTeacher),
            "packed-reference-teacher" => Ok(Self::PackedReferenceTeacher),
            "cpu-free" => Ok(Self::CpuFree),
            "packed-reference-free" => Ok(Self::PackedReferenceFree),
            other => Err(format!("invalid --mode {other:?}")),
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::CpuTeacher => "packed_reference_cpu_teacher",
            Self::PackedReferenceTeacher => "packed_w8_linear_layer_reference_teacher_forced",
            Self::CpuFree => "packed_reference_cpu_free_run",
            Self::PackedReferenceFree => "packed_w8_linear_layer_reference_free_run",
        }
    }

    fn is_packed_reference(self) -> bool {
        matches!(
            self,
            Self::PackedReferenceTeacher | Self::PackedReferenceFree
        )
    }

    fn requires_input(self) -> bool {
        self.is_packed_reference()
    }
}

struct Args {
    model_dir: PathBuf,
    source_lock: PathBuf,
    mode: Mode,
    input_receipt: Option<PathBuf>,
    output: PathBuf,
    layer: usize,
    profile: Qwen35PackedW8LinearLayerReferenceProfile,
}

struct RunResult {
    receipt: Value,
    passed: bool,
}

fn usage() -> &'static str {
    "Usage: qwen35_packed_w8_linear_layer_reference_gate \
  --model-dir OFFICIAL_LOCAL_QWEN35_0_8B \
  --source-lock SOURCE_LOCK.json \
  --mode cpu-teacher|packed-reference-teacher|cpu-free|packed-reference-free \
  [--profile g64|gdn-out-g32|mlp-down-g32|gdn-out-and-mlp-down-g32] \
  [--input-receipt CPU_RECEIPT.json] \
  --output NEW_RECEIPT.json \
  [--layer 0]\n\
The gate is frozen to prompt=Hello and 128 steps. packed-reference-* modes require the matching CPU receipt. Output publication uses create_new and never replaces an artifact."
}

fn main() -> Result<(), Box<dyn Error>> {
    if cfg!(debug_assertions) {
        return Err(
            "qwen35_packed_w8_linear_layer_reference_gate must be built with --release".into(),
        );
    }
    if !cfg!(target_os = "macos") {
        return Err("qwen35_packed_w8_linear_layer_reference_gate requires macOS".into());
    }
    let args = parse_args()?;
    if args.output.exists() {
        return Err(format!(
            "refusing to replace existing receipt {}",
            args.output.display()
        )
        .into());
    }
    let custody = gate_evidence::GateCustody::capture(
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
    if config.text.layer_types.get(args.layer) != Some(&Qwen35LayerType::LinearAttention) {
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
    let mut model = if args.mode.is_packed_reference() {
        GeneralQwen35::from_weights_with_packed_w8_linear_layer_reference_profile(
            config,
            tensors,
            Device::Cpu,
            max_context,
            args.layer,
            args.profile,
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
        "execution_lane": "cpu-packed-w8-linear-layer-reference",
        "custody": custody.receipt_json(),
    });
    let setup = json!({
        "checkpoint_load_ms": checkpoint_load_ms,
        "model_construct_ms": model_construct_ms,
        "timing_classification": "single-pass candidate timing only; not formal ABBA evidence",
    });

    let result = match args.mode {
        Mode::CpuTeacher | Mode::PackedReferenceTeacher => run_teacher(
            &args,
            &mut model,
            &prompt_tokens,
            vocab_size,
            identity,
            setup,
        )?,
        Mode::CpuFree | Mode::PackedReferenceFree => run_free(
            &args,
            &mut model,
            &prompt_tokens,
            vocab_size,
            identity,
            setup,
        )?,
    };
    custody.verify_unchanged()?;
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
        .packed_w8_linear_layer_reference_stats()
        .map(packed_reference_stats_json);

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
                "precision_profile": null,
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
        .ok_or("--input-receipt is required in packed-reference-teacher mode")?;
    let (cpu_receipt, input_attestation) =
        gate_evidence::read_attested_json(input_path, "packed-reference CPU teacher receipt")?;
    validate_cpu_receipt(
        &cpu_receipt,
        CPU_TEACHER_FORMAT,
        "packed_reference_cpu_teacher",
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
                "packed_w8_linear_layer_reference_actual": actual,
            }));
        }
    }
    let decode_ms = decode_started.elapsed().as_secs_f64() * 1000.0;
    let final_stats = model
        .packed_w8_linear_layer_reference_stats()
        .ok_or("diagnostic constructor omitted packed W8 linear-layer reference receipt")?;
    let prefill_hits_valid = prefill_path_receipt.as_ref().is_some_and(|stats| {
        stats["profile"] == args.profile.as_str()
            && stats["prefill_seed_calls"] == 1
            && stats["decode_calls"] == 0
            && stats["successful_decodes"] == 0
            && stats["failed_decodes"] == 0
            && stats["committed_state_version"] == 0
            && stats["terminal_error"] == false
    });
    let decode_hits_valid = final_stats.layer_index == args.layer
        && final_stats.profile == args.profile
        && final_stats.prefill_seed_calls == 1
        && final_stats.decode_calls == STEPS
        && final_stats.successful_decodes == STEPS
        && final_stats.failed_decodes == 0
        && final_stats.committed_state_version == STEPS as u64
        && !final_stats.terminal_error;
    let passed = mismatches.is_empty() && prefill_hits_valid && decode_hits_valid;
    gate_evidence::verify_file_unchanged(
        input_path,
        &input_attestation,
        "packed-reference CPU teacher receipt",
    )?;
    Ok(RunResult {
        receipt: json!({
            "format": PACKED_REFERENCE_TEACHER_FORMAT,
            "mode": args.mode.label(),
            "identity": identity,
            "prompt": PROMPT,
            "prompt_token_ids": prompt_tokens,
            "comparisons": STEPS,
            "selected_layer": args.layer,
            "precision_profile": args.profile.as_str(),
            "prefill_token": prefill_token,
            "packed_reference_cpu_teacher_receipt": gate_evidence::attestation_json(&input_attestation),
            "teacher_input_ids": teacher_inputs,
            "cpu_expected_output_ids": expected_outputs,
            "packed_w8_linear_layer_reference_actual_output_ids": actual_outputs,
            "mismatches": mismatches,
            "prefill_path_receipt": prefill_path_receipt,
            "final_path_receipt": packed_reference_stats_json(final_stats),
            "path_checks": {
                "prefill_hits_valid": prefill_hits_valid,
                "decode_hits_valid": decode_hits_valid,
            },
            "timing": {
                "setup": setup,
                "prefill_ms": prefill_ms,
                "decode_ms": decode_ms,
                "decode_mean_ms": decode_ms / STEPS as f64,
                "packed_reference_block_mean_us": final_stats.block_elapsed_ns as f64
                    / final_stats.decode_calls.max(1) as f64 / 1000.0,
                "classification": "candidate-only single pass under an uncontrolled host; never promotion evidence",
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
    vocab_size: usize,
    identity: Value,
    setup: Value,
) -> Result<RunResult, Box<dyn Error>> {
    let prefill_started = std::time::Instant::now();
    let prefill = model.prefill_for_generation(LlmInput::text(prompt_tokens))?;
    let prefill_ms = prefill_started.elapsed().as_secs_f64() * 1000.0;
    let mut current = argmax(&prefill, vocab_size)?;
    let prefill_path_receipt = model
        .packed_w8_linear_layer_reference_stats()
        .map(packed_reference_stats_json);
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
                "precision_profile": null,
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
        .ok_or("--input-receipt is required in packed-reference-free mode")?;
    let (cpu_receipt, input_attestation) =
        gate_evidence::read_attested_json(input_path, "packed-reference CPU free-run receipt")?;
    validate_cpu_receipt(
        &cpu_receipt,
        CPU_FREE_FORMAT,
        "packed_reference_cpu_free_run",
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
            (cpu != gdn).then(
                || json!({ "step": step, "cpu": cpu, "packed_w8_linear_layer_reference": gdn }),
            )
        })
        .collect::<Vec<_>>();
    let first_mismatch = mismatches
        .first()
        .and_then(|entry| entry.get("step"))
        .and_then(Value::as_u64);
    let final_stats = model
        .packed_w8_linear_layer_reference_stats()
        .ok_or("diagnostic constructor omitted packed W8 linear-layer reference receipt")?;
    let prefill_hits_valid = prefill_path_receipt.as_ref().is_some_and(|stats| {
        stats["profile"] == args.profile.as_str()
            && stats["prefill_seed_calls"] == 1
            && stats["decode_calls"] == 0
            && stats["successful_decodes"] == 0
            && stats["failed_decodes"] == 0
            && stats["committed_state_version"] == 0
            && stats["terminal_error"] == false
    });
    let expected_decode = STEPS - 1;
    let decode_hits_valid = final_stats.layer_index == args.layer
        && final_stats.profile == args.profile
        && final_stats.prefill_seed_calls == 1
        && final_stats.decode_calls == expected_decode
        && final_stats.successful_decodes == expected_decode
        && final_stats.failed_decodes == 0
        && final_stats.committed_state_version == expected_decode as u64
        && !final_stats.terminal_error;
    let passed = cpu_generated.len() == STEPS
        && generated.len() == STEPS
        && mismatches.is_empty()
        && prefill_hits_valid
        && decode_hits_valid;
    gate_evidence::verify_file_unchanged(
        input_path,
        &input_attestation,
        "packed-reference CPU free-run receipt",
    )?;
    Ok(RunResult {
        receipt: json!({
            "format": PACKED_REFERENCE_FREE_FORMAT,
            "mode": args.mode.label(),
            "identity": identity,
            "prompt": PROMPT,
            "prompt_token_ids": prompt_tokens,
            "generated_tokens": STEPS,
            "selected_layer": args.layer,
            "precision_profile": args.profile.as_str(),
            "cpu_free_receipt": gate_evidence::attestation_json(&input_attestation),
            "cpu_generated_token_ids": cpu_generated,
            "packed_w8_linear_layer_reference_generated_token_ids": generated,
            "mismatches": mismatches,
            "first_mismatch": first_mismatch,
            "prefill_path_receipt": prefill_path_receipt,
            "final_path_receipt": packed_reference_stats_json(final_stats),
            "path_checks": {
                "prefill_hits_valid": prefill_hits_valid,
                "decode_hits_valid": decode_hits_valid,
            },
            "timing": {
                "setup": setup,
                "prefill_ms": prefill_ms,
                "decode_ms": decode_ms,
                "decode_mean_ms": decode_ms / expected_decode as f64,
                "packed_reference_block_mean_us": final_stats.block_elapsed_ns as f64
                    / final_stats.decode_calls.max(1) as f64 / 1000.0,
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
    let mut profile = Qwen35PackedW8LinearLayerReferenceProfile::G64;
    let mut profile_set = false;
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
            "--profile" => {
                if profile_set {
                    return Err("--profile may be specified only once".into());
                }
                profile = parse_profile(&value(&mut iter)?.to_string_lossy())?;
                profile_set = true;
            }
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
    if mode.requires_input() != input_receipt.is_some() {
        return Err(if mode.requires_input() {
            "--input-receipt is required for packed-reference-* modes".into()
        } else {
            "--input-receipt is not accepted for cpu-* modes".into()
        });
    }
    if !mode.is_packed_reference() && profile_set {
        return Err("--profile is accepted only for packed-reference-* modes".into());
    }
    Ok(Args {
        model_dir: model_dir.ok_or_else(|| format!("--model-dir is required\n{}", usage()))?,
        source_lock: source_lock
            .ok_or_else(|| format!("--source-lock is required\n{}", usage()))?,
        mode,
        input_receipt,
        output: output.ok_or_else(|| format!("--output is required\n{}", usage()))?,
        layer,
        profile,
    })
}

fn parse_profile(value: &str) -> Result<Qwen35PackedW8LinearLayerReferenceProfile, String> {
    match value {
        "g64" => Ok(Qwen35PackedW8LinearLayerReferenceProfile::G64),
        "gdn-out-g32" => Ok(Qwen35PackedW8LinearLayerReferenceProfile::GdnOutG32),
        "mlp-down-g32" => Ok(Qwen35PackedW8LinearLayerReferenceProfile::MlpDownG32),
        "gdn-out-and-mlp-down-g32" => {
            Ok(Qwen35PackedW8LinearLayerReferenceProfile::GdnOutAndMlpDownG32)
        }
        other => Err(format!("invalid --profile {other:?}")),
    }
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
        return Err("CPU receipt does not match this exact packed-reference gate request".into());
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

fn packed_reference_stats_json(stats: Qwen35PackedW8LinearLayerReferenceStats) -> Value {
    let quantization = stats.quantization;
    json!({
        "layer_index": stats.layer_index,
        "profile": stats.profile.as_str(),
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
        "prefill_seed_calls": stats.prefill_seed_calls,
        "decode_calls": stats.decode_calls,
        "successful_decodes": stats.successful_decodes,
        "failed_decodes": stats.failed_decodes,
        "committed_state_version": stats.committed_state_version,
        "terminal_error": stats.terminal_error,
        "block_elapsed_ns": stats.block_elapsed_ns,
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
    fn mode_parser_keeps_cpu_and_packed_reference_receipt_roles_separate() {
        assert_eq!(Mode::parse("cpu-teacher").unwrap(), Mode::CpuTeacher);
        assert!(!Mode::CpuFree.requires_input());
        assert!(Mode::PackedReferenceTeacher.requires_input());
        assert!(Mode::parse("combined").is_err());
    }

    #[test]
    fn precision_profile_parser_exposes_only_the_four_controlled_portfolio_variants() {
        assert_eq!(
            parse_profile("g64").unwrap(),
            Qwen35PackedW8LinearLayerReferenceProfile::G64
        );
        assert_eq!(
            parse_profile("gdn-out-g32").unwrap(),
            Qwen35PackedW8LinearLayerReferenceProfile::GdnOutG32
        );
        assert_eq!(
            parse_profile("mlp-down-g32").unwrap(),
            Qwen35PackedW8LinearLayerReferenceProfile::MlpDownG32
        );
        assert_eq!(
            parse_profile("gdn-out-and-mlp-down-g32").unwrap(),
            Qwen35PackedW8LinearLayerReferenceProfile::GdnOutAndMlpDownG32
        );
        assert!(parse_profile("all-g32").is_err());
    }

    #[test]
    fn path_receipt_records_exact_group_and_scale_bytes_for_every_matrix() {
        let stats = Qwen35PackedW8LinearLayerReferenceStats {
            layer_index: 0,
            profile: Qwen35PackedW8LinearLayerReferenceProfile::GdnOutAndMlpDownG32,
            quantization: apxinf_metal::LinearLayerQuantizationLedger {
                gdn_input_group_size: apxinf_metal::W8GroupSize::G64,
                gdn_input_weight_bytes: 101,
                gdn_input_scale_bytes: 102,
                gdn_output_group_size: apxinf_metal::W8GroupSize::G32,
                gdn_output_weight_bytes: 201,
                gdn_output_scale_bytes: 202,
                mlp_gate_group_size: apxinf_metal::W8GroupSize::G64,
                mlp_gate_weight_bytes: 301,
                mlp_gate_scale_bytes: 302,
                mlp_up_group_size: apxinf_metal::W8GroupSize::G64,
                mlp_up_weight_bytes: 401,
                mlp_up_scale_bytes: 402,
                mlp_down_group_size: apxinf_metal::W8GroupSize::G32,
                mlp_down_weight_bytes: 501,
                mlp_down_scale_bytes: 502,
                total_packed_weight_bytes: 1_505,
                total_packed_scale_bytes: 1_510,
            },
            prefill_seed_calls: 1,
            decode_calls: 2,
            successful_decodes: 2,
            failed_decodes: 0,
            committed_state_version: 2,
            terminal_error: false,
            block_elapsed_ns: 3,
        };

        let receipt = packed_reference_stats_json(stats);
        assert_eq!(receipt["profile"], "gdn-out-and-mlp-down-g32");
        assert_eq!(receipt["quantization"]["gdn_input"]["group_size"], 64);
        assert_eq!(receipt["quantization"]["gdn_output"]["group_size"], 32);
        assert_eq!(receipt["quantization"]["gdn_output"]["scale_bytes"], 202);
        assert_eq!(receipt["quantization"]["mlp_gate"]["scale_bytes"], 302);
        assert_eq!(receipt["quantization"]["mlp_up"]["scale_bytes"], 402);
        assert_eq!(receipt["quantization"]["mlp_down"]["group_size"], 32);
        assert_eq!(receipt["quantization"]["mlp_down"]["scale_bytes"], 502);
        assert_eq!(receipt["quantization"]["total_packed_scale_bytes"], 1_510);
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
            "mode": "packed_reference_cpu_teacher",
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
            "packed_reference_cpu_teacher",
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
