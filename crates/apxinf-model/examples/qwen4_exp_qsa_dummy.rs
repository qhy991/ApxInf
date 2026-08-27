//! Checkpoint-free QSA selector benchmark using a real Qwen4-Exp config.

use std::hint::black_box;
use std::path::PathBuf;
use std::time::Instant;

use apxinf_model::{Qwen4ExpConfig, Qwen4ExpQsaSelector};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args_os().skip(1);
    let config_path = PathBuf::from(
        args.next()
            .ok_or("usage: qwen4_exp_qsa_dummy CONFIG CONTEXT ITERATIONS")?,
    );
    let context = parse_usize(
        args.next()
            .ok_or("usage: qwen4_exp_qsa_dummy CONFIG CONTEXT ITERATIONS")?,
        "CONTEXT",
    )?;
    let iterations = parse_usize(
        args.next()
            .ok_or("usage: qwen4_exp_qsa_dummy CONFIG CONTEXT ITERATIONS")?,
        "ITERATIONS",
    )?;
    if args.next().is_some() || context == 0 || iterations == 0 {
        return Err("usage: qwen4_exp_qsa_dummy CONFIG CONTEXT ITERATIONS".into());
    }

    let config = Qwen4ExpConfig::from_json_file(&config_path)?;
    let text = &config.text;
    let selector = Qwen4ExpQsaSelector::new(
        text.indexer_n_heads,
        text.indexer_head_dim,
        text.indexer_budget,
        text.indexer_compress_ratio,
        text.rotary_dim(),
        text.rope.theta,
        text.rms_norm_eps,
    )?;
    let mut rng = DeterministicRng::new(0x38f1_a5a5_2026_0827);
    let query = (0..text.indexer_n_heads * text.indexer_head_dim)
        .map(|_| rng.next_f32())
        .collect::<Vec<_>>();
    let raw_keys = (0..context * text.indexer_head_dim)
        .map(|_| rng.next_f32())
        .collect::<Vec<_>>();

    for _ in 0..2 {
        black_box(selector.select_unit_norm(black_box(&query), black_box(&raw_keys), context)?);
    }
    let mut samples_ns = Vec::with_capacity(iterations);
    let mut checksum = 0usize;
    let mut selected_tokens = 0usize;
    for _ in 0..iterations {
        let started = Instant::now();
        let selected =
            selector.select_unit_norm(black_box(&query), black_box(&raw_keys), context)?;
        samples_ns.push(started.elapsed().as_nanos());
        selected_tokens = selected.len();
        checksum ^= selected
            .iter()
            .fold(0usize, |sum, value| sum.wrapping_mul(16_777_619) ^ value);
        black_box(&selected);
    }
    samples_ns.sort_unstable();
    let median_ns = samples_ns[samples_ns.len() / 2];
    let p95_index = ((samples_ns.len() - 1) * 95).div_ceil(100);
    let p95_ns = samples_ns[p95_index];

    println!(
        "{}",
        serde_json::json!({
            "format": "apxinf-qwen4-exp-qsa-dummy-v1",
            "weights_loaded": false,
            "context_tokens": context,
            "query_heads": text.indexer_n_heads,
            "head_dim": text.indexer_head_dim,
            "compress_ratio": text.indexer_compress_ratio,
            "token_budget": text.indexer_budget,
            "complete_blocks": context / text.indexer_compress_ratio,
            "selected_tokens": selected_tokens,
            "iterations": iterations,
            "median_ms": median_ns as f64 / 1_000_000.0,
            "p95_ms": p95_ns as f64 / 1_000_000.0,
            "checksum": checksum,
        })
    );
    Ok(())
}

fn parse_usize(value: std::ffi::OsString, name: &str) -> Result<usize, String> {
    value
        .into_string()
        .map_err(|_| format!("{name} must be UTF-8"))?
        .parse()
        .map_err(|_| format!("{name} must be a positive integer"))
}

struct DeterministicRng(u64);

impl DeterministicRng {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_f32(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let unit = (self.0 >> 40) as f32 / (1u32 << 24) as f32;
        unit * 2.0 - 1.0
    }
}
