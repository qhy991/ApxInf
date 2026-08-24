//! Same-binary free-run receipt for the all-layer Metal W8 MLP + head tracer.

use std::error::Error;
use std::ffi::OsString;
use std::path::PathBuf;

use apxinf_core::Device;
use apxinf_model::{GeneralQwen35, LlmInput, LlmTrait, Qwen35Config};
use apxinf_tokenizer::{ChatMessage, Tokenizer};
use serde_json::json;

const FORMAT: &str = "apxinf-qwen35-metal-w8-combined-free-run-v1";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mode {
    Cpu,
    Combined,
}

struct Args {
    model_dir: PathBuf,
    mode: Mode,
    max_new_tokens: usize,
    prompt: String,
}

fn usage() -> &'static str {
    "Usage: qwen35_metal_w8_combined_free_run \
  --model-dir PATH \
  --mode cpu|combined \
  [--max-new-tokens 128] \
  [--prompt Hello]"
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = parse_args()?;
    let tokenizer = Tokenizer::from_file(args.model_dir.join("tokenizer.json"))?;
    let formatted = tokenizer.apply_chat_template(&[ChatMessage::user(&args.prompt)])?;
    let prompt_tokens = tokenizer.encode(&formatted)?;
    let config = Qwen35Config::from_json_file(&args.model_dir.join("config.json"))?;
    let layer_count = config.text.n_layers;
    let (tensors, _) =
        apxinf_loader::safetensors::load_native_path_filtered(&args.model_dir, |name| {
            name.starts_with("model.language_model.") || name == "lm_head.weight"
        })?;
    let max_context = prompt_tokens
        .len()
        .checked_add(args.max_new_tokens)
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
    let started = std::time::Instant::now();
    let (generated, profile) = model.generate_streaming(
        LlmInput::text(&prompt_tokens),
        args.max_new_tokens,
        |_| {},
        None,
    )?;
    let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
    let block_stats = model.metal_w8_mlp_block_layer_stats();
    let head_stats = model.metal_w8_lm_head_stats();
    let expected_decode_calls = args.max_new_tokens.saturating_sub(1);
    let expected_layers = (0..layer_count).collect::<Vec<_>>();
    let path_passed = match args.mode {
        Mode::Cpu => block_stats.is_empty() && head_stats.is_none(),
        Mode::Combined => {
            block_stats
                .iter()
                .map(|stats| stats.layer_index)
                .eq(expected_layers.iter().copied())
                && block_stats
                    .iter()
                    .all(|stats| stats.decode_calls == expected_decode_calls)
                && head_stats.is_some_and(|stats| {
                    stats.prefill_calls == 1
                        && stats.decode_calls == expected_decode_calls
                        && stats.teacher_calls == 0
                })
        }
    };
    println!(
        "{}",
        serde_json::to_string(&json!({
            "format": FORMAT,
            "mode": match args.mode {
                Mode::Cpu => "cpu",
                Mode::Combined => "metal_w8_mlp_blocks_and_lm_head",
            },
            "model_dir": std::fs::canonicalize(&args.model_dir)?,
            "prompt": args.prompt,
            "prompt_token_ids": prompt_tokens,
            "max_new_tokens": args.max_new_tokens,
            "eos_stopping": false,
            "selected_layers": if args.mode == Mode::Combined {
                Some(&expected_layers)
            } else {
                None
            },
            "mlp_block_path_receipt": block_stats.iter().map(|stats| json!({
                "selected_layer": stats.layer_index,
                "decode_calls": stats.decode_calls,
                "block_elapsed_ns": stats.block_elapsed_ns,
                "block_mean_us": stats.block_elapsed_ns as f64
                    / stats.decode_calls.max(1) as f64 / 1000.0,
            })).collect::<Vec<_>>(),
            "lm_head_path_receipt": head_stats.map(|stats| json!({
                "prefill_calls": stats.prefill_calls,
                "decode_calls": stats.decode_calls,
                "teacher_calls": stats.teacher_calls,
                "topk_elapsed_ns": stats.topk_elapsed_ns,
                "rerank_elapsed_ns": stats.rerank_elapsed_ns,
                "topk_plus_rerank_mean_us": (stats.topk_elapsed_ns + stats.rerank_elapsed_ns)
                    as f64 / (stats.prefill_calls + stats.decode_calls + stats.teacher_calls)
                    .max(1) as f64 / 1000.0,
            })),
            "path_passed": path_passed,
            "generated_token_ids": generated,
            "profile": {
                "input_tokens": profile.input_tokens(),
                "output_tokens": profile.output_tokens(),
                "ttft_ms": profile.ttft_ms(),
                "tpot_ms": profile.tpot_ms(),
                "generation_tps": profile.generation_tps(),
                "generation_total_latency_ms": profile.total_latency_ms(),
                "harness_elapsed_ms": elapsed_ms,
            },
        }))?
    );
    if !path_passed {
        std::process::exit(1);
    }
    Ok(())
}

fn parse_args() -> Result<Args, String> {
    let mut model_dir = None;
    let mut mode = None;
    let mut max_new_tokens = 128usize;
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
            "--max-new-tokens" => {
                max_new_tokens = value(&mut iter)?
                    .to_string_lossy()
                    .parse()
                    .map_err(|error| format!("invalid --max-new-tokens: {error}"))?
            }
            "--prompt" => prompt = value(&mut iter)?.to_string_lossy().into_owned(),
            "-h" | "--help" => {
                println!("{}", usage());
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument {other}\n{}", usage())),
        }
    }
    if max_new_tokens == 0 {
        return Err("--max-new-tokens must be greater than zero".into());
    }
    Ok(Args {
        model_dir: model_dir.ok_or_else(|| format!("--model-dir is required\n{}", usage()))?,
        mode: mode.ok_or_else(|| format!("--mode is required\n{}", usage()))?,
        max_new_tokens,
        prompt,
    })
}
