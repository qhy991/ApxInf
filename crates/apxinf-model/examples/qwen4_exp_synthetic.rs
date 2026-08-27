//! Run the native Qwen4-Exp text path with deterministic synthetic weights.

use std::path::PathBuf;
use std::time::Instant;

use apxinf_model::{GeneralQwen4Exp, LlmTrait, Qwen4ExpConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args_os().skip(1);
    let config_path = PathBuf::from(
        args.next()
            .ok_or("usage: qwen4_exp_synthetic CONFIG [TOKENS]")?,
    );
    let tokens = match args.next() {
        Some(value) => value
            .into_string()
            .map_err(|_| "TOKENS must be UTF-8")?
            .split(',')
            .map(str::parse::<u32>)
            .collect::<Result<Vec<_>, _>>()?,
        None => vec![1, 3, 5, 7, 9, 11, 13, 15],
    };
    if args.next().is_some() || tokens.is_empty() {
        return Err("usage: qwen4_exp_synthetic CONFIG [TOKENS]".into());
    }

    let config = Qwen4ExpConfig::from_json_file(&config_path)?;
    let mut model = GeneralQwen4Exp::from_synthetic(config, 38, tokens.len() + 8)?;
    let started = Instant::now();
    let logits = model.forward(&tokens, 0)?;
    let elapsed = started.elapsed();
    let values = logits.as_f32()?;
    let checksum = values.iter().fold(0u64, |hash, value| {
        hash.wrapping_mul(1_099_511_628_211) ^ value.to_bits() as u64
    });
    println!(
        "{}",
        serde_json::json!({
            "format": "apxinf-qwen4-exp-synthetic-run-v1",
            "weights_loaded": false,
            "tokens": tokens.len(),
            "logits_shape": logits.shape().dims(),
            "elapsed_ms": elapsed.as_secs_f64() * 1000.0,
            "checksum": checksum,
            "generation_path": model.generation_path_receipt(),
        })
    );
    Ok(())
}
