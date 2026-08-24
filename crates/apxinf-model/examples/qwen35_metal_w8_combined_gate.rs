//! Same-binary CPU-teacher gate for the all-layer Metal W8 MLP + head tracer.

use std::error::Error;
use std::ffi::OsString;
use std::fs;
use std::path::{Path, PathBuf};

use apxinf_core::{Device, Tensor};
use apxinf_model::{GeneralQwen35, LlmInput, LlmTrait, Qwen35Config};
use apxinf_tokenizer::{ChatMessage, Tokenizer};
use serde_json::{json, Value};

const CPU_FORMAT: &str = "apxinf-qwen35-metal-w8-combined-cpu-teacher-v1";
const COMBINED_FORMAT: &str = "apxinf-qwen35-metal-w8-combined-teacher-gate-v1";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mode {
    Cpu,
    Combined,
}

struct Args {
    model_dir: PathBuf,
    mode: Mode,
    teacher_json: Option<PathBuf>,
    steps: usize,
    prompt: String,
}

fn usage() -> &'static str {
    "Usage: qwen35_metal_w8_combined_gate \
  --model-dir PATH \
  --mode cpu|combined \
  [--teacher-json CPU_RECEIPT.json] \
  [--steps 128] \
  [--prompt Hello]"
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = parse_args()?;
    let tokenizer = Tokenizer::from_file(args.model_dir.join("tokenizer.json"))?;
    let formatted = tokenizer.apply_chat_template(&[ChatMessage::user(&args.prompt)])?;
    let prompt_tokens = tokenizer.encode(&formatted)?;
    let config = Qwen35Config::from_json_file(&args.model_dir.join("config.json"))?;
    let vocab_size = config.text.vocab_size;
    let layer_count = config.text.n_layers;
    let (tensors, _) =
        apxinf_loader::safetensors::load_native_path_filtered(&args.model_dir, |name| {
            name.starts_with("model.language_model.") || name == "lm_head.weight"
        })?;
    let max_context = prompt_tokens
        .len()
        .checked_add(args.steps + 1)
        .ok_or("context length overflow")?;
    let mut model = match args.mode {
        Mode::Cpu => GeneralQwen35::from_weights(config, tensors, Device::Cpu, max_context)?,
        Mode::Combined => GeneralQwen35::from_weights_with_metal_w8_mlp_blocks_and_lm_head(
            config,
            tensors,
            Device::Cpu,
            max_context,
        )?,
    };
    // Deliberately use the ordinary CPU/F32 projection for teacher prefill.
    // The combined decode hook is exercised only after this state advance.
    let prefill = model.prefill_for_generation(LlmInput::text(&prompt_tokens))?;
    let prefill_token = argmax(&prefill, vocab_size)?;
    let canonical_model_dir = fs::canonicalize(&args.model_dir)?;

    match args.mode {
        Mode::Cpu => {
            let mut teacher = prefill_token;
            let mut teacher_inputs = Vec::with_capacity(args.steps);
            let mut expected_outputs = Vec::with_capacity(args.steps);
            for step in 0..args.steps {
                teacher_inputs.push(teacher);
                let logits =
                    model.forward(&[teacher], u32::try_from(prompt_tokens.len() + step)?)?;
                teacher = argmax(&logits, vocab_size)?;
                expected_outputs.push(teacher);
            }
            println!(
                "{}",
                serde_json::to_string(&json!({
                    "format": CPU_FORMAT,
                    "mode": "cpu_teacher",
                    "model_dir": canonical_model_dir,
                    "prompt": args.prompt,
                    "prompt_token_ids": prompt_tokens,
                    "prefill_token": prefill_token,
                    "comparisons": args.steps,
                    "teacher_input_ids": teacher_inputs,
                    "cpu_expected_output_ids": expected_outputs,
                }))?
            );
        }
        Mode::Combined => {
            let teacher_path = args
                .teacher_json
                .as_ref()
                .ok_or("--teacher-json is required in combined mode")?;
            let receipt: Value = serde_json::from_slice(&fs::read(teacher_path)?)?;
            validate_receipt(
                &receipt,
                &canonical_model_dir,
                &args.prompt,
                &prompt_tokens,
                args.steps,
                prefill_token,
            )?;
            let teacher_inputs = json_u32_array(&receipt, "teacher_input_ids")?;
            let expected_outputs = json_u32_array(&receipt, "cpu_expected_output_ids")?;
            let mut actual_outputs = Vec::with_capacity(args.steps);
            let mut mismatches = Vec::new();
            for step in 0..args.steps {
                let actual = model
                    .decode_token(
                        teacher_inputs[step],
                        u32::try_from(prompt_tokens.len() + step)?,
                    )
                    .ok_or("combined model did not claim the decode hook")??;
                actual_outputs.push(actual);
                if actual != expected_outputs[step] {
                    mismatches.push(json!({
                        "step": step,
                        "teacher_input": teacher_inputs[step],
                        "cpu_expected": expected_outputs[step],
                        "metal_combined_actual": actual,
                    }));
                }
            }
            let block_stats = model.metal_w8_mlp_block_layer_stats();
            let head_stats = model
                .metal_w8_lm_head_stats()
                .ok_or("combined model omitted Metal head stats")?;
            let path_layers = block_stats
                .iter()
                .map(|stats| stats.layer_index)
                .collect::<Vec<_>>();
            let expected_layers = (0..layer_count).collect::<Vec<_>>();
            let passed = mismatches.is_empty()
                && path_layers == expected_layers
                && block_stats
                    .iter()
                    .all(|stats| stats.decode_calls == args.steps)
                && head_stats.prefill_calls == 0
                && head_stats.decode_calls == args.steps
                && head_stats.teacher_calls == 0;
            println!(
                "{}",
                serde_json::to_string(&json!({
                    "format": COMBINED_FORMAT,
                    "mode": "metal_w8_mlp_blocks_and_lm_head_teacher_forced",
                    "model_dir": canonical_model_dir,
                    "prompt": args.prompt,
                    "prompt_token_ids": prompt_tokens,
                    "prefill_projection": "cpu_f32",
                    "prefill_token": prefill_token,
                    "comparisons": args.steps,
                    "selected_layers": expected_layers,
                    "teacher_receipt": fs::canonicalize(teacher_path)?,
                    "teacher_input_ids": teacher_inputs,
                    "cpu_expected_output_ids": expected_outputs,
                    "metal_combined_actual_output_ids": actual_outputs,
                    "mismatches": mismatches,
                    "mlp_block_path_receipt": block_stats.iter().map(|stats| json!({
                        "selected_layer": stats.layer_index,
                        "decode_calls": stats.decode_calls,
                        "block_elapsed_ns": stats.block_elapsed_ns,
                        "block_mean_us": stats.block_elapsed_ns as f64
                            / stats.decode_calls.max(1) as f64 / 1000.0,
                    })).collect::<Vec<_>>(),
                    "lm_head_path_receipt": {
                        "prefill_calls": head_stats.prefill_calls,
                        "decode_calls": head_stats.decode_calls,
                        "teacher_calls": head_stats.teacher_calls,
                        "topk_elapsed_ns": head_stats.topk_elapsed_ns,
                        "rerank_elapsed_ns": head_stats.rerank_elapsed_ns,
                    },
                    "passed": passed,
                }))?
            );
            if !passed {
                std::process::exit(1);
            }
        }
    }
    Ok(())
}

fn parse_args() -> Result<Args, String> {
    let mut model_dir = None;
    let mut mode = None;
    let mut teacher_json = None;
    let mut steps = 128usize;
    let mut prompt = "Hello".to_owned();
    let mut iter = std::env::args_os().skip(1);
    while let Some(raw_flag) = iter.next() {
        let flag = raw_flag.to_string_lossy();
        let value = |iter: &mut dyn Iterator<Item = OsString>| {
            iter.next()
                .ok_or_else(|| format!("{flag} requires a value"))
        };
        match flag.as_ref() {
            "--model-dir" => model_dir = Some(PathBuf::from(value(&mut iter)?)),
            "--mode" => {
                mode = Some(match value(&mut iter)?.to_string_lossy().as_ref() {
                    "cpu" => Mode::Cpu,
                    "combined" => Mode::Combined,
                    other => return Err(format!("invalid --mode {other:?}")),
                })
            }
            "--teacher-json" => teacher_json = Some(PathBuf::from(value(&mut iter)?)),
            "--steps" => {
                steps = value(&mut iter)?
                    .to_string_lossy()
                    .parse()
                    .map_err(|error| format!("invalid --steps: {error}"))?
            }
            "--prompt" => prompt = value(&mut iter)?.to_string_lossy().into_owned(),
            "-h" | "--help" => {
                println!("{}", usage());
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument {other}\n{}", usage())),
        }
    }
    if steps == 0 {
        return Err("--steps must be greater than zero".into());
    }
    Ok(Args {
        model_dir: model_dir.ok_or_else(|| format!("--model-dir is required\n{}", usage()))?,
        mode: mode.ok_or_else(|| format!("--mode is required\n{}", usage()))?,
        teacher_json,
        steps,
        prompt,
    })
}

fn validate_receipt(
    receipt: &Value,
    model_dir: &Path,
    prompt: &str,
    prompt_tokens: &[u32],
    steps: usize,
    prefill_token: u32,
) -> Result<(), Box<dyn Error>> {
    if receipt.get("format").and_then(Value::as_str) != Some(CPU_FORMAT)
        || receipt.get("mode").and_then(Value::as_str) != Some("cpu_teacher")
        || receipt.get("model_dir").and_then(Value::as_str) != model_dir.to_str()
        || receipt.get("prompt").and_then(Value::as_str) != Some(prompt)
        || receipt.get("comparisons").and_then(Value::as_u64) != Some(steps as u64)
        || receipt.get("prefill_token").and_then(Value::as_u64) != Some(prefill_token as u64)
        || json_u32_array(receipt, "prompt_token_ids")? != prompt_tokens
    {
        return Err("CPU teacher receipt does not match this combined request".into());
    }
    let inputs = json_u32_array(receipt, "teacher_input_ids")?;
    let outputs = json_u32_array(receipt, "cpu_expected_output_ids")?;
    if inputs.len() != steps || outputs.len() != steps {
        return Err("CPU teacher receipt length does not match --steps".into());
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

fn argmax(logits: &Tensor, vocab_size: usize) -> Result<u32, Box<dyn Error>> {
    if logits.shape().dims() != [1, vocab_size] {
        return Err(format!("expected logits [1, {vocab_size}], got {}", logits.shape()).into());
    }
    let mut best = f32::NEG_INFINITY;
    let mut best_token = 0u32;
    for (token, &score) in logits.as_f32()?.iter().enumerate() {
        if score > best {
            best = score;
            best_token = token as u32;
        }
    }
    Ok(best_token)
}
