//! Real-model diagnostic for the current Qwen3.5 boundary-tail execution lane.
//!
//! This is a noisy-host screening loop, not formal promotion evidence.

use std::error::Error;
use std::ffi::OsString;
use std::path::PathBuf;

use apxinf_core::Device;
use apxinf_metal::GdnCoreProfileV1;
use apxinf_model::{GeneralQwen35, LlmInput, LlmTrait, Qwen35Config};
use apxinf_tokenizer::{ChatMessage, Tokenizer};
use serde_json::json;

const FORMAT: &str = "apxinf-qwen35-short-sdpa-boundary-tail-screen-v1";

struct Args {
    model_dir: PathBuf,
    max_new_tokens: usize,
    prompt: String,
}

fn usage() -> &'static str {
    "Usage: qwen35_short_sdpa_boundary_tail_screen \
  --model-dir PATH \
  [--max-new-tokens 128] \
  [--prompt Hello]"
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = parse_args()?;
    let tokenizer = Tokenizer::from_file(args.model_dir.join("tokenizer.json"))?;
    let formatted = tokenizer.apply_chat_template(&[ChatMessage::user(&args.prompt)])?;
    let prompt_tokens = tokenizer.encode(&formatted)?;
    let config = Qwen35Config::from_json_file(&args.model_dir.join("config.json"))?;
    let (tensors, _) =
        apxinf_loader::safetensors::load_native_path_filtered(&args.model_dir, |name| {
            name.starts_with("model.language_model.") || name == "lm_head.weight"
        })?;
    let max_context = prompt_tokens
        .len()
        .checked_add(args.max_new_tokens)
        .ok_or("context length overflow")?;
    let mut model =
        GeneralQwen35::from_weights_with_metal_w8_mlp_stack3_boundary_tail_head_gdn_core_fused_v1(
            config,
            tensors,
            Device::Cpu,
            max_context,
        )?;

    let started = std::time::Instant::now();
    let (generated, profile) = model.generate_streaming(
        LlmInput::text(&prompt_tokens),
        args.max_new_tokens,
        |_| {},
        None,
    )?;
    let harness_elapsed_ms = started.elapsed().as_secs_f64() * 1_000.0;
    let stats = model
        .metal_w8_mlp_stack3_boundary_tail_head_v1_stats()
        .ok_or("boundary-tail screen constructor omitted stats")?;
    let generation = model
        .generation_path_receipt()
        .ok_or("boundary-tail screen constructor omitted generation receipt")?;
    let expected_decode_calls = args.max_new_tokens.saturating_sub(1);
    let initial_valid = stats.initial_stack.prefill_seed_calls == [1, 1, 1]
        && stats.initial_stack.execution.decode_calls == expected_decode_calls
        && stats.initial_stack.execution.successful_decodes == expected_decode_calls
        && !stats.initial_stack.terminal_error;
    let boundaries_valid = stats.boundaries.len() == 5
        && stats.boundaries.iter().all(|boundary| {
            boundary.prefill_seed_calls == [1, 1, 1]
                && boundary.execution.decode_calls == expected_decode_calls
                && boundary.execution.successful_decodes == expected_decode_calls
                && !boundary.terminal_error
        });
    let tail_valid = stats.tail.decode_calls == expected_decode_calls
        && stats.tail.successful_decodes == expected_decode_calls
        && stats.tail.failed_decodes == 0
        && !stats.tail.terminal_error;
    let path_passed = stats.gdn_core_profile == GdnCoreProfileV1::Fused128
        && stats.prefill_body_calls == 1
        && stats.prefill_cpu_head_calls == 1
        && stats.decode_calls == expected_decode_calls
        && stats.teacher_calls == 0
        && stats.tail_layer_index == 23
        && !stats.terminal_error
        && initial_valid
        && boundaries_valid
        && tail_valid
        && generation["terminal_error"] == false;

    println!(
        "{}",
        serde_json::to_string(&json!({
            "format": FORMAT,
            "qualification": "real-model noisy-host diagnostic; not formal evidence",
            "model_dir": std::fs::canonicalize(&args.model_dir)?,
            "prompt": args.prompt,
            "prompt_token_ids": prompt_tokens,
            "max_new_tokens": args.max_new_tokens,
            "generated_token_ids": generated,
            "path": {
                "mechanism": stats.mechanism,
                "gdn_core_profile": "gdn-core-fused-v1",
                "boundary_count": stats.boundaries.len(),
                "tail_layer_index": stats.tail_layer_index,
                "prefill_body_calls": stats.prefill_body_calls,
                "prefill_cpu_head_calls": stats.prefill_cpu_head_calls,
                "decode_calls": stats.decode_calls,
                "teacher_calls": stats.teacher_calls,
                "initial_valid": initial_valid,
                "boundaries_valid": boundaries_valid,
                "tail_valid": tail_valid,
                "terminal_error": stats.terminal_error,
                "passed": path_passed,
            },
            "profile": {
                "input_tokens": profile.input_tokens(),
                "output_tokens": profile.output_tokens(),
                "ttft_ms": profile.ttft_ms(),
                "tpot_ms": profile.tpot_ms(),
                "generation_tps": profile.generation_tps(),
                "generation_total_latency_ms": profile.total_latency_ms(),
                "harness_elapsed_ms": harness_elapsed_ms,
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
        max_new_tokens,
        prompt,
    })
}
