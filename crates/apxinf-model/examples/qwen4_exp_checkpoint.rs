//! Run a Qwen4-Exp text checkpoint through the correctness-first CPU runtime.

use std::collections::HashSet;
use std::path::PathBuf;
use std::time::Instant;

use apxinf_model::{GeneralQwen4Exp, LlmTrait, Qwen4ExpConfig, Qwen4ExpWeightSchema};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args_os().skip(1);
    let usage = "usage: qwen4_exp_checkpoint CONFIG CHECKPOINT [TOKENS]";
    let config_path = PathBuf::from(args.next().ok_or(usage)?);
    let checkpoint_path = PathBuf::from(args.next().ok_or(usage)?);
    let tokens = match args.next() {
        Some(value) => value
            .into_string()
            .map_err(|_| "TOKENS must be UTF-8")?
            .split(',')
            .map(str::parse::<u32>)
            .collect::<Result<Vec<_>, _>>()?,
        None => vec![1, 3, 5, 7, 9],
    };
    if args.next().is_some() || tokens.is_empty() {
        return Err(usage.into());
    }

    let config = Qwen4ExpConfig::from_json_file(&config_path)?;
    let schema = Qwen4ExpWeightSchema::new(&config)?;
    let runtime_names = schema
        .runtime_names()
        .iter()
        .cloned()
        .collect::<HashSet<_>>();
    let (tensors, _) =
        apxinf_loader::safetensors::load_native_path_mmap_filtered(&checkpoint_path, |name| {
            runtime_names.contains(name)
        })?;
    let mut model = GeneralQwen4Exp::from_tensors(config, tensors, tokens.len() + 8)?;
    let started = Instant::now();
    let logits = model.forward(&tokens, 0)?;
    let elapsed = started.elapsed();
    let values = logits.to_f32_vec()?;
    let checksum = values.iter().fold(0u64, |hash, value| {
        hash.wrapping_mul(1_099_511_628_211) ^ value.to_bits() as u64
    });
    println!(
        "{}",
        serde_json::json!({
            "format": "apxinf-qwen4-exp-checkpoint-run-v1",
            "tokens": tokens,
            "logits_shape": logits.shape().dims(),
            "logits": values,
            "elapsed_ms": elapsed.as_secs_f64() * 1000.0,
            "checksum": checksum,
            "generation_path": model.generation_path_receipt(),
        })
    );
    Ok(())
}
