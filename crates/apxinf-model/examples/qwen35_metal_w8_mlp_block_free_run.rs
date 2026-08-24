//! Direct shared-generation trajectory receipt for complete Metal W8 MLP blocks.

use std::error::Error;
use std::ffi::OsString;
use std::path::PathBuf;

use apxinf_core::Device;
use apxinf_model::{GeneralQwen35, LlmInput, LlmTrait, Qwen35Config};
use apxinf_tokenizer::{ChatMessage, Tokenizer};
use serde_json::json;

const FORMAT: &str = "apxinf-qwen35-metal-w8-mlp-block-free-run-v1";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mode {
    Cpu,
    Block,
}

struct Args {
    model_dir: PathBuf,
    mode: Mode,
    max_new_tokens: usize,
    prompt: String,
    layers: Option<Vec<usize>>,
    all_layers: bool,
}

fn usage() -> &'static str {
    "Usage: qwen35_metal_w8_mlp_block_free_run \
  --model-dir PATH \
  --mode cpu|block \
  [--max-new-tokens 128] \
  [--prompt Hello] \
  [--layer 0 | --layers 0,2,4 | --all-layers]"
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = parse_args()?;
    let tokenizer = Tokenizer::from_file(args.model_dir.join("tokenizer.json"))?;
    let formatted = tokenizer.apply_chat_template(&[ChatMessage::user(&args.prompt)])?;
    let prompt_tokens = tokenizer.encode(&formatted)?;
    let config = Qwen35Config::from_json_file(&args.model_dir.join("config.json"))?;
    let mut selected_layers = if args.all_layers {
        (0..config.text.n_layers).collect::<Vec<_>>()
    } else {
        args.layers.clone().unwrap_or_else(|| vec![0])
    };
    selected_layers.sort_unstable();
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
        Mode::Block => GeneralQwen35::from_weights_with_metal_w8_mlp_block_layers(
            config,
            tensors,
            Device::Cpu,
            max_context,
            &selected_layers,
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
    let path_receipt = model
        .metal_w8_mlp_block_layer_stats()
        .into_iter()
        .map(|stats| {
            json!({
                "selected_layer": stats.layer_index,
                "decode_calls": stats.decode_calls,
                "block_elapsed_ns": stats.block_elapsed_ns,
                "block_mean_us": stats.block_elapsed_ns as f64
                    / stats.decode_calls.max(1) as f64 / 1000.0,
            })
        })
        .collect::<Vec<_>>();
    println!(
        "{}",
        serde_json::to_string(&json!({
            "format": FORMAT,
            "mode": match args.mode { Mode::Cpu => "cpu", Mode::Block => "metal_w8_mlp_block" },
            "model_dir": std::fs::canonicalize(&args.model_dir)?,
            "prompt": args.prompt,
            "prompt_token_ids": prompt_tokens,
            "max_new_tokens": args.max_new_tokens,
            "eos_stopping": false,
            "selected_layers": if args.mode == Mode::Block { Some(&selected_layers) } else { None },
            "path_receipt": path_receipt,
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
    Ok(())
}

fn parse_args() -> Result<Args, String> {
    let mut model_dir = None;
    let mut mode = None;
    let mut max_new_tokens = 128usize;
    let mut prompt = "Hello".to_owned();
    let mut layers = None;
    let mut all_layers = false;
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
                    "block" => Mode::Block,
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
            "--layer" => {
                let layer = value(&mut iter)?
                    .to_string_lossy()
                    .parse()
                    .map_err(|error| format!("invalid --layer: {error}"))?;
                select_layers(&mut layers, &mut all_layers, vec![layer])?;
            }
            "--layers" => {
                let parsed = parse_layers(&value(&mut iter)?.to_string_lossy())?;
                select_layers(&mut layers, &mut all_layers, parsed)?;
            }
            "--all-layers" => {
                if layers.is_some() || all_layers {
                    return Err("select MLP block layers with exactly one layer flag".into());
                }
                all_layers = true;
            }
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
        layers,
        all_layers,
    })
}

fn select_layers(
    layers: &mut Option<Vec<usize>>,
    all_layers: &mut bool,
    selected: Vec<usize>,
) -> Result<(), String> {
    if layers.replace(selected).is_some() || *all_layers {
        return Err("select MLP block layers with exactly one layer flag".into());
    }
    Ok(())
}

fn parse_layers(value: &str) -> Result<Vec<usize>, String> {
    if value.is_empty() {
        return Err("--layers cannot be empty".into());
    }
    value
        .split(',')
        .enumerate()
        .map(|(index, value)| {
            value
                .parse::<usize>()
                .map_err(|error| format!("invalid --layers element {index}: {error}"))
        })
        .collect()
}
