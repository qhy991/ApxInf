//! Complete 64-token/64-layer Qwen3.8 M64 Marlin prefill admission probe.

use std::cmp::Ordering;
use std::path::Path;
use std::time::Instant;

use apxinf_core::Tensor;
use apxinf_cuda::tuning::KERNEL_BUILD_ID;
use apxinf_cuda::{transfers, CudaContext};
use apxinf_loader::safetensors::{self, CheckpointManifest};
use apxinf_model::qwen35::{
    load_embedding_row, HybridUnit, HybridUnitMode, Qwen35LmHead, Qwen35PrefillMode,
};

const ROWS: usize = 64;
const M8: usize = 8;
const HIDDEN: usize = 5120;
const PAIRS: usize = 5;

struct Snapshot {
    residual: Vec<f32>,
    normalized: Vec<f32>,
    first_logits: Vec<f32>,
    first_token: u32,
    next_normalized: Vec<f32>,
    second_logits: Vec<f32>,
    second_token: u32,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_dir = std::env::args()
        .nth(1)
        .ok_or("usage: qwen35_marlin_stack_probe MODEL_DIR")?;
    let manifest = safetensors::inspect_path(Path::new(&model_dir))?;
    let context = CudaContext::new(0)?;
    if context.caps().sm != 89 {
        return Err(format!(
            "Marlin stack probe is frozen for SM89, got SM{}",
            context.caps().sm
        )
        .into());
    }
    let decoder = HybridUnit::load_all_with_prefill_mode(
        &manifest,
        &context,
        128,
        Qwen35PrefillMode::MarlinM64,
    )?;
    if decoder.layer_count() != 64 || !decoder.has_marlin_prefill64() {
        return Err("expected a 64-layer decoder with Marlin M64 enabled".into());
    }
    let lm_head = Qwen35LmHead::load(&manifest, &context)?;
    let input = Tensor::from_bf16(vec![ROWS, HIDDEN], &deterministic_hidden())?;

    let baseline = correctness_baseline(&manifest, &context, &decoder, &lm_head, &input)?;
    let candidate = correctness_candidate(&manifest, &context, &decoder, &lm_head, &input)?;
    let residual = metrics(&candidate.residual, &baseline.residual)?;
    let normalized = metrics(&candidate.normalized, &baseline.normalized)?;
    let first_logits = metrics(&candidate.first_logits, &baseline.first_logits)?;
    let next_normalized = metrics(&candidate.next_normalized, &baseline.next_normalized)?;
    let second_logits = metrics(&candidate.second_logits, &baseline.second_logits)?;
    for (name, value) in [
        ("M64.first_logits", first_logits),
        ("M64.second_logits", second_logits),
    ] {
        if value.cosine < 0.999 || value.relative_l2 > 0.05 {
            return Err(format!("{name} numerical gate failed: {value:?}").into());
        }
    }
    if candidate.first_token != baseline.first_token
        || candidate.second_token != baseline.second_token
    {
        return Err(format!(
            "argmax mismatch: first={}/{} second={}/{}",
            baseline.first_token,
            candidate.first_token,
            baseline.second_token,
            candidate.second_token,
        )
        .into());
    }

    run_baseline(&context, &decoder, &input)?;
    run_candidate(&context, &decoder, &input)?;
    let mut baseline_samples = Vec::with_capacity(PAIRS);
    let mut candidate_samples = Vec::with_capacity(PAIRS);
    let mut records = Vec::with_capacity(2 * PAIRS);
    for pair in 0..PAIRS {
        let candidate_first = pair % 2 == 1;
        for order_index in 0..2 {
            let candidate_arm = (order_index == 0) == candidate_first;
            let elapsed_us = if candidate_arm {
                run_candidate(&context, &decoder, &input)?
            } else {
                run_baseline(&context, &decoder, &input)?
            };
            if candidate_arm {
                candidate_samples.push(elapsed_us);
            } else {
                baseline_samples.push(elapsed_us);
            }
            records.push(serde_json::json!({
                "pair":pair,"order":if candidate_first{"BA"}else{"AB"},
                "order_index":order_index,
                "arm":if candidate_arm{"marlin_m64_mlp"}else{"eight_m8"},
                "elapsed_us":elapsed_us,
            }));
        }
    }
    let speedups = baseline_samples
        .iter()
        .zip(&candidate_samples)
        .map(|(baseline, candidate)| baseline / candidate)
        .collect::<Vec<_>>();
    let wins = baseline_samples
        .iter()
        .zip(&candidate_samples)
        .filter(|(baseline, candidate)| candidate < baseline)
        .count();
    let speedup = median(&speedups);
    if speedup < 1.10 || wins < 4 {
        return Err(format!(
            "joined 64-layer Marlin gate failed: speedup={speedup}, wins={wins}/{PAIRS}"
        )
        .into());
    }

    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "schema":"apxinf.qwen38_27b.marlin_stack_probe.v1",
            "model_dir":model_dir,
            "contract":{
                "rows":ROWS,"layers":64,"max_seq_len":128,"sm":89,
                "baseline":"eight accepted M8 complete-stack tiles",
                "candidate":"M8 stateful mixers plus runtime-transform Marlin M64 MLP projections",
                "boundary":"input H2D through 64 layers and final offset RMSNorm; LM head excluded from timing",
                "decode":"unchanged ModelOptimized M1",
            },
            "kernel_build_id":KERNEL_BUILD_ID,
            "correctness":{
                "final_residual":metric_json(residual),
                "final_normalized":metric_json(normalized),
                "first_logits":metric_json(first_logits),
                "next_token_normalized":metric_json(next_normalized),
                "second_logits":metric_json(second_logits),
                "first_argmax":{"baseline":baseline.first_token,"candidate":candidate.first_token,"exact":true},
                "second_argmax":{"baseline":baseline.second_token,"candidate":candidate.second_token,"exact":baseline.second_token==candidate.second_token},
                "threshold":{"logit_cosine_min":0.999,"logit_relative_l2_max":0.05,"first_two_argmax_exact":true},
                "hidden_state_metrics_are_diagnostic":true,
                "pass":true,
            },
            "paired_timing":{
                "pairs":PAIRS,"records":records,
                "baseline_raw_us":baseline_samples,"candidate_raw_us":candidate_samples,
                "baseline_median_us":median(&baseline_samples),
                "candidate_median_us":median(&candidate_samples),
                "paired_speedups":speedups,"median_speedup":speedup,
                "candidate_wins":wins,
            },
            "evidence_level":"complete-64-token-64-layer-prefill-plus-next-token-state",
            "service_promoted":false,
        }))?
    );
    Ok(())
}

fn correctness_baseline(
    manifest: &CheckpointManifest,
    context: &CudaContext,
    decoder: &HybridUnit,
    lm_head: &Qwen35LmHead,
    input: &Tensor,
) -> Result<Snapshot, Box<dyn std::error::Error>> {
    reset_for_prefill(context, decoder, input)?;
    let mut residual = Vec::with_capacity(ROWS * HIDDEN);
    let mut normalized = Vec::with_capacity(ROWS * HIDDEN);
    for first in (0..ROWS).step_by(M8) {
        let tile = cpu_rows(input, first, M8)?;
        decoder.set_prefill8_input(context, &tile)?;
        decoder.forward_prefill8(context, first, false)?;
        context.synchronize()?;
        residual.extend(transfers::to_cpu(decoder.prefill_output())?.to_f32_vec()?);
        normalized.extend(transfers::to_cpu(decoder.prefill_normalized_output())?.to_f32_vec()?);
    }
    decoder.commit_prefill8_last(context)?;
    finish_snapshot(manifest, context, decoder, lm_head, residual, normalized)
}

fn correctness_candidate(
    manifest: &CheckpointManifest,
    context: &CudaContext,
    decoder: &HybridUnit,
    lm_head: &Qwen35LmHead,
    input: &Tensor,
) -> Result<Snapshot, Box<dyn std::error::Error>> {
    reset_for_prefill(context, decoder, input)?;
    decoder.set_marlin_prefill64_input(context, input)?;
    decoder.forward_marlin_prefill64(context, 0, false)?;
    context.synchronize()?;
    let residual = transfers::to_cpu(decoder.marlin_prefill64_output()?)?.to_f32_vec()?;
    let normalized =
        transfers::to_cpu(decoder.marlin_prefill64_normalized_output()?)?.to_f32_vec()?;
    decoder.commit_marlin_prefill64_last(context)?;
    finish_snapshot(manifest, context, decoder, lm_head, residual, normalized)
}

fn finish_snapshot(
    manifest: &CheckpointManifest,
    context: &CudaContext,
    decoder: &HybridUnit,
    lm_head: &Qwen35LmHead,
    residual: Vec<f32>,
    normalized: Vec<f32>,
) -> Result<Snapshot, Box<dyn std::error::Error>> {
    lm_head.forward(context, decoder.normalized_output())?;
    context.synchronize()?;
    let first_logits = transfers::to_cpu(lm_head.logits())?.to_f32_vec()?;
    let first_token = lm_head.argmax_cpu()?;
    let embedding = load_embedding_row(manifest, first_token)?;
    decoder.set_token_input(context, &embedding)?;
    decoder.forward(
        context,
        HybridUnitMode::ModelOptimized,
        128,
        ROWS as u32,
        false,
    )?;
    context.synchronize()?;
    let next_normalized = transfers::to_cpu(decoder.normalized_output())?.to_f32_vec()?;
    lm_head.forward(context, decoder.normalized_output())?;
    context.synchronize()?;
    let second_logits = transfers::to_cpu(lm_head.logits())?.to_f32_vec()?;
    let second_token = lm_head.argmax_cpu()?;
    Ok(Snapshot {
        residual,
        normalized,
        first_logits,
        first_token,
        next_normalized,
        second_logits,
        second_token,
    })
}

fn run_baseline(
    context: &CudaContext,
    decoder: &HybridUnit,
    input: &Tensor,
) -> Result<f64, Box<dyn std::error::Error>> {
    reset_for_prefill(context, decoder, input)?;
    let start = Instant::now();
    for first in (0..ROWS).step_by(M8) {
        decoder.set_prefill8_input(context, &cpu_rows(input, first, M8)?)?;
        decoder.forward_prefill8(context, first, false)?;
    }
    context.synchronize()?;
    Ok(start.elapsed().as_secs_f64() * 1.0e6)
}

fn run_candidate(
    context: &CudaContext,
    decoder: &HybridUnit,
    input: &Tensor,
) -> Result<f64, Box<dyn std::error::Error>> {
    reset_for_prefill(context, decoder, input)?;
    let start = Instant::now();
    decoder.set_marlin_prefill64_input(context, input)?;
    decoder.forward_marlin_prefill64(context, 0, false)?;
    context.synchronize()?;
    Ok(start.elapsed().as_secs_f64() * 1.0e6)
}

fn reset_for_prefill(
    context: &CudaContext,
    decoder: &HybridUnit,
    input: &Tensor,
) -> Result<(), Box<dyn std::error::Error>> {
    decoder.reset_text_request(context, &cpu_rows(input, 0, 1)?)?;
    Ok(())
}

fn cpu_rows(
    tensor: &Tensor,
    first: usize,
    rows: usize,
) -> Result<Tensor, Box<dyn std::error::Error>> {
    let values = tensor.as_bf16()?;
    let begin = first * HIDDEN;
    let end = begin + rows * HIDDEN;
    Ok(Tensor::from_bf16(vec![rows, HIDDEN], &values[begin..end])?)
}

fn deterministic_hidden() -> Vec<half::bf16> {
    (0..ROWS * HIDDEN)
        .map(|index| {
            let row = index / HIDDEN;
            let column = index % HIDDEN;
            let phase = column as f32 * 0.004_882_812_5
                + (column % 37) as f32 * 0.001_953_125
                + row as f32 * 0.013_671_875;
            half::bf16::from_f32((phase.sin() + 0.2 * phase.cos()) * 0.2)
        })
        .collect()
}

#[derive(Clone, Copy, Debug)]
struct Metrics {
    cosine: f64,
    relative_l2: f64,
    max_abs: f64,
}

fn metrics(actual: &[f32], expected: &[f32]) -> Result<Metrics, String> {
    if actual.len() != expected.len() || actual.is_empty() {
        return Err("length mismatch".into());
    }
    let (mut dot, mut aa, mut ee, mut error, mut maximum) = (0.0, 0.0, 0.0, 0.0, 0.0_f64);
    for (&actual, &expected) in actual.iter().zip(expected) {
        if !actual.is_finite() || !expected.is_finite() {
            return Err("non-finite Marlin stack output".into());
        }
        let (actual, expected) = (f64::from(actual), f64::from(expected));
        dot += actual * expected;
        aa += actual * actual;
        ee += expected * expected;
        error += (actual - expected).powi(2);
        maximum = maximum.max((actual - expected).abs());
    }
    Ok(Metrics {
        cosine: dot / (aa.sqrt() * ee.sqrt()),
        relative_l2: (error / ee).sqrt(),
        max_abs: maximum,
    })
}

fn metric_json(value: Metrics) -> serde_json::Value {
    serde_json::json!({
        "cosine":value.cosine,"relative_l2":value.relative_l2,
        "max_abs":value.max_abs,"finite":true,"pass":true,
    })
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
