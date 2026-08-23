//! Real layer-0..3 Qwen3.5 hybrid decoder-unit trajectory and paired timing.

use std::cmp::Ordering;
use std::path::Path;
use std::time::Instant;

use apxinf_core::Tensor;
use apxinf_cuda::kernels::qwen35_common;
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
        .ok_or("usage: qwen35_hybrid_unit_probe MODEL_DIR")?;
    let kv_len = std::env::var("APXINF_UNIT_KV_LEN")
        .unwrap_or_else(|_| "1024".into())
        .parse::<usize>()?;
    if kv_len == 0 || kv_len > 32768 {
        return Err("APXINF_UNIT_KV_LEN must be within 1..=32768".into());
    }
    let manifest = safetensors::inspect_path(Path::new(&model_dir))?;
    let context = CudaContext::new(0)?;
    if context.caps().sm != 89 {
        return Err(format!(
            "hybrid-unit probe is frozen for SM89, got SM{}",
            context.caps().sm
        )
        .into());
    }
    let input_values = deterministic_hidden();
    let input = Tensor::from_bf16(vec![1, HIDDEN], &input_values)?;
    let norm_correctness = validate_offset_norm(&manifest, &context, &input)?;
    let (key_cache, value_cache) = deterministic_caches(kv_len)?;
    let unit = HybridUnit::load_first(&manifest, &context, kv_len)?;

    unit.reset(&context, &input, &key_cache, &value_cache)?;
    unit.forward(
        &context,
        HybridUnitMode::Native,
        kv_len,
        (kv_len - 1) as u32,
        false,
    )?;
    context.synchronize()?;
    let native_output = transfers::to_cpu(unit.output())?.to_f32_vec()?;

    unit.reset(&context, &input, &key_cache, &value_cache)?;
    unit.forward(
        &context,
        HybridUnitMode::LayerOptimized,
        kv_len,
        (kv_len - 1) as u32,
        false,
    )?;
    context.synchronize()?;
    let optimized_output = transfers::to_cpu(unit.output())?.to_f32_vec()?;
    let correctness = metrics(&optimized_output, &native_output)?;
    if correctness.0 < 0.99 || correctness.1 > 0.15 {
        return Err(format!("hybrid-unit optimized/native gate failed: {correctness:?}").into());
    }

    for mode in [HybridUnitMode::Native, HybridUnitMode::LayerOptimized] {
        unit.reset(&context, &input, &key_cache, &value_cache)?;
        unit.forward(&context, mode, kv_len, (kv_len - 1) as u32, false)?;
        context.synchronize()?;
    }

    if std::env::var("APXINF_PROFILE").as_deref() == Ok("1") {
        let profile_mode = match std::env::var("APXINF_PROFILE_MODE")
            .unwrap_or_else(|_| "optimized".into())
            .as_str()
        {
            "native" => HybridUnitMode::Native,
            "optimized" => HybridUnitMode::LayerOptimized,
            value => {
                return Err(format!(
                    "APXINF_PROFILE_MODE must be native or optimized, got `{value}`"
                )
                .into())
            }
        };
        unit.reset(&context, &input, &key_cache, &value_cache)?;
        apxinf_cuda::profiler::start()?;
        unit.forward(&context, profile_mode, kv_len, (kv_len - 1) as u32, true)?;
        context.synchronize()?;
        apxinf_cuda::profiler::stop()?;
    }

    let mut records = Vec::with_capacity(2 * PAIRS);
    let mut native_samples = Vec::with_capacity(PAIRS);
    let mut optimized_samples = Vec::with_capacity(PAIRS);
    for pair in 0..PAIRS {
        let order = if pair % 2 == 0 {
            [HybridUnitMode::Native, HybridUnitMode::LayerOptimized]
        } else {
            [HybridUnitMode::LayerOptimized, HybridUnitMode::Native]
        };
        for (order_index, mode) in order.into_iter().enumerate() {
            unit.reset(&context, &input, &key_cache, &value_cache)?;
            let start = Instant::now();
            unit.forward(&context, mode, kv_len, (kv_len - 1) as u32, false)?;
            context.synchronize()?;
            let elapsed_us = start.elapsed().as_secs_f64() * 1.0e6;
            match mode {
                HybridUnitMode::Native => native_samples.push(elapsed_us),
                HybridUnitMode::LayerOptimized => optimized_samples.push(elapsed_us),
                HybridUnitMode::ModelOptimized => unreachable!("hybrid-unit probe uses layer mode"),
            }
            records.push(serde_json::json!({
                "pair":pair,
                "order":if pair%2==0{"AB"}else{"BA"},
                "order_index":order_index,
                "mode":mode.as_str(),
                "elapsed_us":elapsed_us,
            }));
        }
    }
    let native_median = median(&native_samples);
    let optimized_median = median(&optimized_samples);
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
            "schema":"apxinf.qwen35.hybrid_unit_probe.v1",
            "model_dir":model_dir,
            "layers":[0,1,2,3],
            "schedule":["linear_attention","linear_attention","linear_attention","full_attention"],
            "contract":{
                "batch":1,"new_tokens":1,"kv_len":kv_len,"kv_dtype":"bf16",
                "hidden":HIDDEN,"sm":89,"eager":true,
                "native":"checkpoint-native GDN out-proj (layer0 BF16, layers1/2 W4) + incumbent attention",
                "optimized":"layer0 W8A16 + checkpoint W4 layers1/2 + split16 attention for KV>=256",
                "timing":"four complete decoder layers including offset RMSNorm, residual, mixer, and MLP; reset excluded; stream synchronize included",
            },
            "kernel_build_id":KERNEL_BUILD_ID,
            "offset_norm_correctness":norm_correctness,
            "correctness":{
                "boundary":"optimized final layer-3 residual versus native final layer-3 residual from identical input and zero/reset state",
                "cosine":correctness.0,"relative_l2":correctness.1,"max_abs":correctness.2,
                "threshold":{"cosine_min":0.99,"relative_l2_max":0.15},
                "pass":true,
            },
            "paired_timing":{
                "pairs":PAIRS,"records":records,
                "native_raw_us":native_samples,"optimized_raw_us":optimized_samples,
                "native_median_us":native_median,"optimized_median_us":optimized_median,
                "paired_speedups":paired_speedups,"median_speedup":median(&paired_speedups),
                "optimized_wins":wins,
            },
            "evidence_level":"four-layer-module",
            "model_promoted":false,
        }))?
    );
    Ok(())
}

fn validate_offset_norm(
    manifest: &apxinf_loader::safetensors::CheckpointManifest,
    context: &CudaContext,
    input: &Tensor,
) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
    let weight_cpu = safetensors::load_manifest_tensor(
        manifest
            .tensor("model.language_model.layers.0.input_layernorm.weight")
            .ok_or("missing layer-0 input norm")?,
    )?;
    let weight = transfers::to_cuda(&weight_cpu, 0)?;
    let input_gpu = transfers::to_cuda(input, 0)?;
    let output = transfers::to_cuda(
        &Tensor::from_bf16(vec![1, HIDDEN], &vec![half::bf16::ZERO; HIDDEN])?,
        0,
    )?;
    qwen35_common::rmsnorm_offset_write(context, &input_gpu, &weight, &output, 1.0e-6)?;
    context.synchronize()?;
    let expected = cpu_offset_norm(input.as_bf16()?, weight_cpu.as_bf16()?);
    let direct = metrics(
        &transfers::to_cpu(&output)?.to_f32_vec()?,
        &bf16_f32(&expected),
    )?;

    let delta_values = (0..HIDDEN)
        .map(|index| half::bf16::from_f32(((index % 29) as f32 - 14.0) / 1024.0))
        .collect::<Vec<_>>();
    let delta_cpu = Tensor::from_bf16(vec![1, HIDDEN], &delta_values)?;
    let residual = transfers::to_cuda(input, 0)?;
    let delta = transfers::to_cuda(&delta_cpu, 0)?;
    let fused_output = transfers::to_cuda(
        &Tensor::from_bf16(vec![1, HIDDEN], &vec![half::bf16::ZERO; HIDDEN])?,
        0,
    )?;
    qwen35_common::residual_add_rmsnorm_offset_write(
        context,
        &residual,
        &delta,
        &weight,
        &fused_output,
        1.0e-6,
    )?;
    context.synchronize()?;
    let expected_residual = input
        .as_bf16()?
        .iter()
        .zip(&delta_values)
        .map(|(left, right)| half::bf16::from_f32(left.to_f32() + right.to_f32()))
        .collect::<Vec<_>>();
    let expected_fused = cpu_offset_norm(&expected_residual, weight_cpu.as_bf16()?);
    let residual_cpu = transfers::to_cpu(&residual)?;
    let residual_exact = residual_cpu.as_bf16()? == expected_residual.as_slice();
    let fused = metrics(
        &transfers::to_cpu(&fused_output)?.to_f32_vec()?,
        &bf16_f32(&expected_fused),
    )?;
    let pass = direct.0 >= 0.999999
        && direct.1 <= 1.0e-3
        && fused.0 >= 0.999999
        && fused.1 <= 1.0e-3
        && residual_exact;
    if !pass {
        return Err(format!(
            "Qwen3.5 offset norm gate failed: direct={direct:?}, fused={fused:?}, residual_exact={residual_exact}"
        )
        .into());
    }
    Ok(serde_json::json!({
        "direct":{"cosine":direct.0,"relative_l2":direct.1,"max_abs":direct.2},
        "residual_fused":{"cosine":fused.0,"relative_l2":fused.1,"max_abs":fused.2},
        "residual_bf16_seam_exact":residual_exact,
        "pass":true,
    }))
}

fn cpu_offset_norm(input: &[half::bf16], weight: &[half::bf16]) -> Vec<half::bf16> {
    let square_sum = input
        .iter()
        .map(|value| value.to_f32().powi(2))
        .sum::<f32>();
    let inverse = (square_sum / input.len() as f32 + 1.0e-6).sqrt().recip();
    input
        .iter()
        .zip(weight)
        .map(|(value, gamma)| {
            half::bf16::from_f32(value.to_f32() * inverse * (1.0 + gamma.to_f32()))
        })
        .collect()
}

fn bf16_f32(values: &[half::bf16]) -> Vec<f32> {
    values.iter().map(|value| value.to_f32()).collect()
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
            return Err("non-finite hybrid-unit output".into());
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
