//! One complete Qwen3.5 decode token: embedding row -> 64 layers -> LM head -> argmax.

use std::cmp::Ordering;
use std::path::Path;
use std::time::Instant;

use apxinf_core::Tensor;
use apxinf_cuda::tuning::KERNEL_BUILD_ID;
use apxinf_cuda::{transfers, CudaContext};
use apxinf_loader::safetensors;
use apxinf_model::qwen35::{load_embedding_row, HybridUnit, HybridUnitMode, Qwen35LmHead};

const KV_HEADS: usize = 4;
const HEAD_DIM: usize = 256;
const PAIRS: usize = 5;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_dir = std::env::args()
        .nth(1)
        .ok_or("usage: qwen35_token_probe MODEL_DIR")?;
    let input_token = std::env::var("APXINF_INPUT_TOKEN")
        .unwrap_or_else(|_| "151644".into())
        .parse::<u32>()?;
    let kv_len = std::env::var("APXINF_TOKEN_KV_LEN")
        .unwrap_or_else(|_| "256".into())
        .parse::<usize>()?;
    if kv_len == 0 || kv_len > 32768 {
        return Err("APXINF_TOKEN_KV_LEN must be within 1..=32768".into());
    }
    let manifest = safetensors::inspect_path(Path::new(&model_dir))?;
    let context = CudaContext::new(0)?;
    let embedding = load_embedding_row(&manifest, input_token)?;
    let (key_cache, value_cache) = deterministic_caches(kv_len)?;
    let decoder = HybridUnit::load_all(&manifest, &context, kv_len)?;
    let lm_head = Qwen35LmHead::load(&manifest, &context)?;

    decoder.reset(&context, &embedding, &key_cache, &value_cache)?;
    decoder.forward(
        &context,
        HybridUnitMode::Native,
        kv_len,
        (kv_len - 1) as u32,
        false,
    )?;
    lm_head.forward(&context, decoder.normalized_output())?;
    let native_token = lm_head.argmax_cpu()?;
    let native_logits = transfers::to_cpu(lm_head.logits())?.to_f32_vec()?;

    decoder.reset(&context, &embedding, &key_cache, &value_cache)?;
    decoder.forward(
        &context,
        HybridUnitMode::ModelOptimized,
        kv_len,
        (kv_len - 1) as u32,
        false,
    )?;
    lm_head.forward(&context, decoder.normalized_output())?;
    let optimized_token = lm_head.argmax_cpu()?;
    let optimized_logits = transfers::to_cpu(lm_head.logits())?.to_f32_vec()?;
    let logit_metrics = metrics(&optimized_logits, &native_logits)?;
    if logit_metrics.0 < 0.98 || logit_metrics.1 > 0.20 {
        return Err(format!("complete-token logit gate failed: {logit_metrics:?}").into());
    }

    for mode in [HybridUnitMode::Native, HybridUnitMode::ModelOptimized] {
        decoder.reset(&context, &embedding, &key_cache, &value_cache)?;
        decoder.forward(&context, mode, kv_len, (kv_len - 1) as u32, false)?;
        lm_head.forward(&context, decoder.normalized_output())?;
        let _ = lm_head.argmax_cpu()?;
    }

    let mut records = Vec::with_capacity(2 * PAIRS);
    let mut native_samples = Vec::with_capacity(PAIRS);
    let mut optimized_samples = Vec::with_capacity(PAIRS);
    let mut emitted_tokens = Vec::with_capacity(2 * PAIRS);
    for pair in 0..PAIRS {
        let order = if pair % 2 == 0 {
            [HybridUnitMode::Native, HybridUnitMode::ModelOptimized]
        } else {
            [HybridUnitMode::ModelOptimized, HybridUnitMode::Native]
        };
        for (order_index, mode) in order.into_iter().enumerate() {
            decoder.reset(&context, &embedding, &key_cache, &value_cache)?;
            let start = Instant::now();
            decoder.forward(&context, mode, kv_len, (kv_len - 1) as u32, false)?;
            lm_head.forward(&context, decoder.normalized_output())?;
            let token = lm_head.argmax_cpu()?;
            let elapsed_us = start.elapsed().as_secs_f64() * 1.0e6;
            match mode {
                HybridUnitMode::Native => native_samples.push(elapsed_us),
                HybridUnitMode::ModelOptimized => optimized_samples.push(elapsed_us),
                HybridUnitMode::LayerOptimized => unreachable!("token probe uses model mode"),
            }
            emitted_tokens.push(token);
            records.push(serde_json::json!({
                "pair":pair,"order":if pair%2==0{"AB"}else{"BA"},
                "order_index":order_index,"mode":mode.as_str(),
                "elapsed_us":elapsed_us,"emitted_token":token,
            }));
        }
    }
    let paired_speedups = native_samples
        .iter()
        .zip(&optimized_samples)
        .map(|(native, optimized)| native / optimized)
        .collect::<Vec<_>>();
    let wins = native_samples
        .iter()
        .zip(&optimized_samples)
        .filter(|(native, optimized)| optimized < native)
        .count();

    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "schema":"apxinf.qwen35.token_probe.v1",
            "model_dir":model_dir,
            "contract":{
                "input_token":input_token,"batch":1,"new_tokens":1,
                "kv_len":kv_len,"kv_dtype":"bf16","sm":89,"eager":true,
                "embedding":"selective BF16 checkpoint row",
                "lm_head":"streaming per-row W8A16 conversion of BF16 checkpoint head",
                "timing":"64 layers + final RMSNorm + W8 LM head + D2H logits + CPU argmax; reset and weight load excluded",
            },
            "kernel_build_id":KERNEL_BUILD_ID,
            "correctness":{
                "native_token":native_token,"optimized_token":optimized_token,
                "exact_token_match":native_token==optimized_token,
                "logits":{"cosine":logit_metrics.0,"relative_l2":logit_metrics.1,"max_abs":logit_metrics.2},
                "threshold":{"cosine_min":0.98,"relative_l2_max":0.20},
                "pass":true,
            },
            "paired_timing":{
                "pairs":PAIRS,"records":records,
                "native_raw_us":native_samples,"optimized_raw_us":optimized_samples,
                "native_median_us":median(&native_samples),
                "optimized_median_us":median(&optimized_samples),
                "native_tokens_per_second":1.0e6/median(&native_samples),
                "optimized_tokens_per_second":1.0e6/median(&optimized_samples),
                "paired_speedups":paired_speedups,"median_speedup":median(&paired_speedups),
                "optimized_wins":wins,"all_emitted_tokens":emitted_tokens,
            },
            "evidence_level":"complete-one-token-offline","service_promoted":false,
        }))?
    );
    Ok(())
}

fn deterministic_caches(
    max_seq_len: usize,
) -> Result<(Tensor, Tensor), Box<dyn std::error::Error>> {
    let count = KV_HEADS * max_seq_len * HEAD_DIM;
    let mut key = vec![half::bf16::ZERO; count];
    let mut value = vec![half::bf16::ZERO; count];
    for head in 0..KV_HEADS {
        for token in 0..max_seq_len {
            let base = (head * max_seq_len + token) * HEAD_DIM;
            for dimension in 0..HEAD_DIM {
                let key_bits = (token * 17 + dimension * 13 + head * 7) & 255;
                let value_bits = (token * 29 + dimension * 11 + head * 19) & 255;
                key[base + dimension] = half::bf16::from_f32((key_bits as f32 - 128.0) / 1024.0);
                value[base + dimension] =
                    half::bf16::from_f32(0.1 + (value_bits as f32 - 128.0) / 2048.0);
            }
        }
    }
    Ok((
        Tensor::from_bf16(vec![KV_HEADS, max_seq_len, HEAD_DIM], &key)?,
        Tensor::from_bf16(vec![KV_HEADS, max_seq_len, HEAD_DIM], &value)?,
    ))
}

fn metrics(actual: &[f32], expected: &[f32]) -> Result<(f64, f64, f64), String> {
    if actual.len() != expected.len() || actual.is_empty() {
        return Err("length mismatch".into());
    }
    let (mut dot, mut aa, mut ee, mut error, mut maximum) = (0.0, 0.0, 0.0, 0.0, 0.0_f64);
    for (&actual, &expected) in actual.iter().zip(expected) {
        if !actual.is_finite() || !expected.is_finite() {
            return Err("non-finite token logits".into());
        }
        let (actual, expected) = (f64::from(actual), f64::from(expected));
        dot += actual * expected;
        aa += actual * actual;
        ee += expected * expected;
        error += (actual - expected).powi(2);
        maximum = maximum.max((actual - expected).abs());
    }
    Ok((dot / (aa.sqrt() * ee.sqrt()), (error / ee).sqrt(), maximum))
}

fn median(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(|left, right| left.partial_cmp(right).unwrap_or(Ordering::Equal));
    if sorted.len() % 2 == 1 {
        sorted[sorted.len() / 2]
    } else {
        (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) * 0.5
    }
}
