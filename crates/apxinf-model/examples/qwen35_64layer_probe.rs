//! Real 64-layer Qwen3.5 text-decoder trajectory before LM head.

use std::cmp::Ordering;
use std::path::Path;
use std::time::Instant;

use apxinf_core::Tensor;
use apxinf_cuda::tuning::KERNEL_BUILD_ID;
use apxinf_cuda::{transfers, CudaContext};
use apxinf_loader::safetensors;
use apxinf_model::qwen35::{HybridUnit, HybridUnitMode};

const HIDDEN: usize = 5120;
const KV_HEADS: usize = 4;
const HEAD_DIM: usize = 256;
const PAIRS: usize = 5;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_dir = std::env::args()
        .nth(1)
        .ok_or("usage: qwen35_64layer_probe MODEL_DIR")?;
    let kv_len = std::env::var("APXINF_64L_KV_LEN")
        .unwrap_or_else(|_| "256".into())
        .parse::<usize>()?;
    if kv_len == 0 || kv_len > 32768 {
        return Err("APXINF_64L_KV_LEN must be within 1..=32768".into());
    }
    let manifest = safetensors::inspect_path(Path::new(&model_dir))?;
    let context = CudaContext::new(0)?;
    if context.caps().sm != 89 {
        return Err(format!(
            "64-layer probe is frozen for SM89, got SM{}",
            context.caps().sm
        )
        .into());
    }
    let input_values = deterministic_hidden();
    let input = Tensor::from_bf16(vec![1, HIDDEN], &input_values)?;
    let (key_cache, value_cache) = deterministic_caches(kv_len)?;
    let decoder = HybridUnit::load_all(&manifest, &context, kv_len)?;
    if decoder.layer_count() != 64 {
        return Err(format!("expected 64 layers, loaded {}", decoder.layer_count()).into());
    }

    decoder.reset(&context, &input, &key_cache, &value_cache)?;
    decoder.forward(
        &context,
        HybridUnitMode::Native,
        kv_len,
        (kv_len - 1) as u32,
        false,
    )?;
    context.synchronize()?;
    let native_residual = transfers::to_cpu(decoder.output())?.to_f32_vec()?;
    let native_normalized = transfers::to_cpu(decoder.normalized_output())?.to_f32_vec()?;

    decoder.reset(&context, &input, &key_cache, &value_cache)?;
    decoder.forward(
        &context,
        HybridUnitMode::ModelOptimized,
        kv_len,
        (kv_len - 1) as u32,
        false,
    )?;
    context.synchronize()?;
    let optimized_residual = transfers::to_cpu(decoder.output())?.to_f32_vec()?;
    let optimized_normalized = transfers::to_cpu(decoder.normalized_output())?.to_f32_vec()?;
    let residual_metrics = metrics(&optimized_residual, &native_residual)?;
    let normalized_metrics = metrics(&optimized_normalized, &native_normalized)?;
    for (name, value) in [
        ("final_residual", residual_metrics),
        ("final_normalized", normalized_metrics),
    ] {
        if value.0 < 0.98 || value.1 > 0.20 {
            return Err(format!("64-layer {name} gate failed: {value:?}").into());
        }
    }

    for mode in [HybridUnitMode::Native, HybridUnitMode::ModelOptimized] {
        decoder.reset(&context, &input, &key_cache, &value_cache)?;
        decoder.forward(&context, mode, kv_len, (kv_len - 1) as u32, false)?;
        context.synchronize()?;
    }

    let mut records = Vec::with_capacity(2 * PAIRS);
    let mut native_samples = Vec::with_capacity(PAIRS);
    let mut optimized_samples = Vec::with_capacity(PAIRS);
    for pair in 0..PAIRS {
        let order = if pair % 2 == 0 {
            [HybridUnitMode::Native, HybridUnitMode::ModelOptimized]
        } else {
            [HybridUnitMode::ModelOptimized, HybridUnitMode::Native]
        };
        for (order_index, mode) in order.into_iter().enumerate() {
            decoder.reset(&context, &input, &key_cache, &value_cache)?;
            let start = Instant::now();
            decoder.forward(&context, mode, kv_len, (kv_len - 1) as u32, false)?;
            context.synchronize()?;
            let elapsed_us = start.elapsed().as_secs_f64() * 1.0e6;
            match mode {
                HybridUnitMode::Native => native_samples.push(elapsed_us),
                HybridUnitMode::ModelOptimized => optimized_samples.push(elapsed_us),
                HybridUnitMode::LayerOptimized => unreachable!("64-layer probe uses model mode"),
            }
            records.push(serde_json::json!({
                "pair":pair,"order":if pair%2==0{"AB"}else{"BA"},
                "order_index":order_index,"mode":mode.as_str(),"elapsed_us":elapsed_us,
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
            "schema":"apxinf.qwen35.64layer_probe.v1",
            "model_dir":model_dir,"layers":64,
            "contract":{
                "batch":1,"new_tokens":1,"kv_len":kv_len,"kv_dtype":"bf16",
                "hidden":HIDDEN,"sm":89,"eager":true,
                "boundary":"64 decoder layers through final offset RMSNorm; embedding and LM head excluded",
            },
            "kernel_build_id":KERNEL_BUILD_ID,
            "correctness":{
                "final_residual":metric_json(residual_metrics),
                "final_normalized":metric_json(normalized_metrics),
                "threshold":{"cosine_min":0.98,"relative_l2_max":0.20},
                "pass":true,
            },
            "paired_timing":{
                "pairs":PAIRS,"records":records,
                "native_raw_us":native_samples,"optimized_raw_us":optimized_samples,
                "native_median_us":median(&native_samples),
                "optimized_median_us":median(&optimized_samples),
                "paired_speedups":paired_speedups,"median_speedup":median(&paired_speedups),
                "optimized_wins":wins,
            },
            "evidence_level":"64-layer-module","token_trajectory_complete":false,
            "model_promoted":false,
        }))?
    );
    Ok(())
}

fn deterministic_hidden() -> Vec<half::bf16> {
    (0..HIDDEN)
        .map(|index| {
            let phase = index as f32 * 0.004_882_812_5 + (index % 37) as f32 * 0.001_953_125;
            half::bf16::from_f32((phase.sin() + 0.2 * phase.cos()) * 0.2)
        })
        .collect()
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
            return Err("non-finite 64-layer output".into());
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

fn metric_json(value: (f64, f64, f64)) -> serde_json::Value {
    serde_json::json!({"cosine":value.0,"relative_l2":value.1,"max_abs":value.2,"pass":true})
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
