//! Same-release-binary real-checkpoint quality gate for one diagnostic Metal
//! W8 GDN layer. This example is deliberately absent from CLI/AutoModel and
//! all default construction paths.

use std::error::Error;
use std::ffi::OsString;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

use apxinf_core::{Device, Tensor};
use apxinf_model::qwen35::general::Qwen35MetalW8GdnStats;
use apxinf_model::{GeneralQwen35, LlmInput, LlmTrait, Qwen35Config, Qwen35LayerType};
use apxinf_tokenizer::{ChatMessage, Tokenizer};
use serde_json::{json, Value};

const CPU_TEACHER_FORMAT: &str = "apxinf-qwen35-metal-w8-gdn-cpu-teacher-v1";
const GDN_TEACHER_FORMAT: &str = "apxinf-qwen35-metal-w8-gdn-teacher-gate-v1";
const CPU_FREE_FORMAT: &str = "apxinf-qwen35-metal-w8-gdn-cpu-free-run-v1";
const GDN_FREE_FORMAT: &str = "apxinf-qwen35-metal-w8-gdn-free-run-gate-v1";
const SOURCE_LOCK_FORMAT: &str = "apxinf-hf-source-lock-v1";
const REPO_ID: &str = "Qwen/Qwen3.5-0.8B";
const LOCKED_REVISION: &str = "2fc06364715b967f1860aea9cf38778875588b17";
const LOCKED_CHECKPOINT: &str = "model.safetensors-00001-of-00001.safetensors";
const LOCKED_CHECKPOINT_SHA256: &str =
    "04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696";
const LOCKED_CHECKPOINT_BYTES: u64 = 1_746_942_600;
const STEPS: usize = 128;
const PROMPT: &str = "Hello";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mode {
    CpuTeacher,
    GdnTeacher,
    CpuFree,
    GdnFree,
}

impl Mode {
    fn parse(value: &str) -> Result<Self, String> {
        match value {
            "cpu-teacher" => Ok(Self::CpuTeacher),
            "gdn-teacher" => Ok(Self::GdnTeacher),
            "cpu-free" => Ok(Self::CpuFree),
            "gdn-free" => Ok(Self::GdnFree),
            other => Err(format!("invalid --mode {other:?}")),
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::CpuTeacher => "cpu_teacher",
            Self::GdnTeacher => "metal_w8_gdn_teacher_forced",
            Self::CpuFree => "cpu_free_run",
            Self::GdnFree => "metal_w8_gdn_free_run",
        }
    }

    fn is_gdn(self) -> bool {
        matches!(self, Self::GdnTeacher | Self::GdnFree)
    }

    fn requires_input(self) -> bool {
        self.is_gdn()
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
    "Usage: qwen35_metal_w8_gdn_gate \
  --model-dir OFFICIAL_LOCAL_QWEN35_0_8B \
  --source-lock SOURCE_LOCK.json \
  --mode cpu-teacher|gdn-teacher|cpu-free|gdn-free \
  [--input-receipt CPU_RECEIPT.json] \
  --output NEW_RECEIPT.json \
  [--layer 0]\n\
The gate is frozen to prompt=Hello and 128 steps. gdn-* modes require the matching CPU receipt. Output publication uses create_new and never replaces an artifact."
}

fn main() -> Result<(), Box<dyn Error>> {
    if cfg!(debug_assertions) {
        return Err("qwen35_metal_w8_gdn_gate must be built with --release".into());
    }
    if !cfg!(target_os = "macos") {
        return Err("qwen35_metal_w8_gdn_gate requires macOS".into());
    }
    let args = parse_args()?;
    if args.output.exists() {
        return Err(format!(
            "refusing to replace existing receipt {}",
            args.output.display()
        )
        .into());
    }
    let source_lock: Value = serde_json::from_slice(&fs::read(&args.source_lock)?)?;
    validate_source_lock(&source_lock)?;
    let canonical_model_dir = fs::canonicalize(&args.model_dir)?;
    let canonical_source_lock = fs::canonicalize(&args.source_lock)?;
    let binary_path = fs::canonicalize(std::env::current_exe()?)?;

    let tokenizer = Tokenizer::from_file(args.model_dir.join("tokenizer.json"))?;
    let formatted = tokenizer.apply_chat_template(&[ChatMessage::user(PROMPT)])?;
    let prompt_tokens = tokenizer.encode(&formatted)?;
    let config = Qwen35Config::from_json_file(&args.model_dir.join("config.json"))?;
    if config.text.layer_types.get(args.layer) != Some(&Qwen35LayerType::LinearAttention) {
        return Err(format!("selected layer {} is not linear attention", args.layer).into());
    }
    let vocab_size = config.text.vocab_size;

    let checkpoint_started = std::time::Instant::now();
    let (tensors, _) =
        apxinf_loader::safetensors::load_native_path_filtered(&args.model_dir, |name| {
            name.starts_with("model.language_model.") || name == "lm_head.weight"
        })?;
    let checkpoint_load_ms = checkpoint_started.elapsed().as_secs_f64() * 1000.0;
    let max_context = prompt_tokens
        .len()
        .checked_add(STEPS + 1)
        .ok_or("context length overflow")?;
    let construct_started = std::time::Instant::now();
    let mut model = if args.mode.is_gdn() {
        GeneralQwen35::from_weights_with_metal_w8_gdn_layer(
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
    });
    let setup = json!({
        "checkpoint_load_ms": checkpoint_load_ms,
        "model_construct_ms": model_construct_ms,
        "timing_classification": "single-pass candidate timing only; not formal ABBA evidence",
    });

    let result = match args.mode {
        Mode::CpuTeacher | Mode::GdnTeacher => run_teacher(
            &args,
            &mut model,
            &prompt_tokens,
            vocab_size,
            identity,
            setup,
        )?,
        Mode::CpuFree | Mode::GdnFree => run_free(
            &args,
            &mut model,
            &prompt_tokens,
            vocab_size,
            identity,
            setup,
        )?,
    };
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
    let prefill_path_receipt = model.metal_w8_gdn_stats().map(gdn_stats_json);

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
        .ok_or("--input-receipt is required in gdn-teacher mode")?;
    let cpu_receipt: Value = serde_json::from_slice(&fs::read(input_path)?)?;
    validate_cpu_receipt(
        &cpu_receipt,
        CPU_TEACHER_FORMAT,
        "cpu_teacher",
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
                "metal_w8_gdn_actual": actual,
            }));
        }
    }
    let decode_ms = decode_started.elapsed().as_secs_f64() * 1000.0;
    let final_stats = model
        .metal_w8_gdn_stats()
        .ok_or("diagnostic constructor omitted Metal W8 GDN receipt")?;
    let prefill_hits_valid = prefill_path_receipt.as_ref().is_some_and(|stats| {
        stats["prefill_seed_calls"] == 1
            && stats["decode_calls"] == 0
            && stats["command_buffers"] == 0
            && stats["waits"] == 0
            && stats["committed_state_version"] == 0
    });
    let decode_hits_valid = final_stats.layer_index == args.layer
        && final_stats.prefill_seed_calls == 1
        && final_stats.decode_calls == STEPS
        && final_stats.command_buffers == STEPS
        && final_stats.waits == STEPS
        && final_stats.committed_state_version == STEPS as u64;
    let passed = mismatches.is_empty() && prefill_hits_valid && decode_hits_valid;
    Ok(RunResult {
        receipt: json!({
            "format": GDN_TEACHER_FORMAT,
            "mode": args.mode.label(),
            "identity": identity,
            "prompt": PROMPT,
            "prompt_token_ids": prompt_tokens,
            "comparisons": STEPS,
            "selected_layer": args.layer,
            "prefill_token": prefill_token,
            "cpu_teacher_receipt": fs::canonicalize(input_path)?,
            "teacher_input_ids": teacher_inputs,
            "cpu_expected_output_ids": expected_outputs,
            "metal_w8_gdn_actual_output_ids": actual_outputs,
            "mismatches": mismatches,
            "prefill_path_receipt": prefill_path_receipt,
            "final_path_receipt": gdn_stats_json(final_stats),
            "path_checks": {
                "prefill_hits_valid": prefill_hits_valid,
                "decode_hits_valid": decode_hits_valid,
            },
            "timing": {
                "setup": setup,
                "prefill_ms": prefill_ms,
                "decode_ms": decode_ms,
                "decode_mean_ms": decode_ms / STEPS as f64,
                "metal_gdn_block_mean_us": final_stats.block_elapsed_ns as f64
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
    let prefill_path_receipt = model.metal_w8_gdn_stats().map(gdn_stats_json);
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
        .ok_or("--input-receipt is required in gdn-free mode")?;
    let cpu_receipt: Value = serde_json::from_slice(&fs::read(input_path)?)?;
    validate_cpu_receipt(
        &cpu_receipt,
        CPU_FREE_FORMAT,
        "cpu_free_run",
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
            (cpu != gdn).then(|| json!({ "step": step, "cpu": cpu, "metal_w8_gdn": gdn }))
        })
        .collect::<Vec<_>>();
    let first_mismatch = mismatches
        .first()
        .and_then(|entry| entry.get("step"))
        .and_then(Value::as_u64);
    let final_stats = model
        .metal_w8_gdn_stats()
        .ok_or("diagnostic constructor omitted Metal W8 GDN receipt")?;
    let prefill_hits_valid = prefill_path_receipt.as_ref().is_some_and(|stats| {
        stats["prefill_seed_calls"] == 1
            && stats["decode_calls"] == 0
            && stats["command_buffers"] == 0
            && stats["waits"] == 0
    });
    let expected_decode = STEPS - 1;
    let decode_hits_valid = final_stats.layer_index == args.layer
        && final_stats.prefill_seed_calls == 1
        && final_stats.decode_calls == expected_decode
        && final_stats.command_buffers == expected_decode
        && final_stats.waits == expected_decode
        && final_stats.committed_state_version == expected_decode as u64;
    let passed = cpu_generated.len() == STEPS
        && generated.len() == STEPS
        && mismatches.is_empty()
        && prefill_hits_valid
        && decode_hits_valid;
    Ok(RunResult {
        receipt: json!({
            "format": GDN_FREE_FORMAT,
            "mode": args.mode.label(),
            "identity": identity,
            "prompt": PROMPT,
            "prompt_token_ids": prompt_tokens,
            "generated_tokens": STEPS,
            "selected_layer": args.layer,
            "cpu_free_receipt": fs::canonicalize(input_path)?,
            "cpu_generated_token_ids": cpu_generated,
            "metal_w8_gdn_generated_token_ids": generated,
            "mismatches": mismatches,
            "first_mismatch": first_mismatch,
            "prefill_path_receipt": prefill_path_receipt,
            "final_path_receipt": gdn_stats_json(final_stats),
            "path_checks": {
                "prefill_hits_valid": prefill_hits_valid,
                "decode_hits_valid": decode_hits_valid,
            },
            "timing": {
                "setup": setup,
                "prefill_ms": prefill_ms,
                "decode_ms": decode_ms,
                "decode_mean_ms": decode_ms / expected_decode as f64,
                "metal_gdn_block_mean_us": final_stats.block_elapsed_ns as f64
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
    if mode.requires_input() != input_receipt.is_some() {
        return Err(if mode.requires_input() {
            "--input-receipt is required for gdn-* modes".into()
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
        return Err("CPU receipt does not match this exact GDN gate request".into());
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

fn gdn_stats_json(stats: Qwen35MetalW8GdnStats) -> Value {
    json!({
        "layer_index": stats.layer_index,
        "prefill_seed_calls": stats.prefill_seed_calls,
        "decode_calls": stats.decode_calls,
        "command_buffers": stats.command_buffers,
        "waits": stats.waits,
        "committed_state_version": stats.committed_state_version,
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
    fn mode_parser_keeps_cpu_and_gdn_receipt_roles_separate() {
        assert_eq!(Mode::parse("cpu-teacher").unwrap(), Mode::CpuTeacher);
        assert!(!Mode::CpuFree.requires_input());
        assert!(Mode::GdnTeacher.requires_input());
        assert!(Mode::parse("combined").is_err());
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
}
