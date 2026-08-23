//! Canonical four-layer Qwen3.5 M=8 hybrid-unit prefill trajectory on SM89.

use std::cmp::Ordering;
use std::path::Path;
use std::time::Instant;

use apxinf_core::{DType, Tensor};
use apxinf_cuda::tuning::KERNEL_BUILD_ID;
use apxinf_cuda::{transfers, CudaContext};
use apxinf_loader::safetensors;
use apxinf_model::qwen35::{HybridUnit, HybridUnitMode};

const TOKENS: usize = 8;
const HIDDEN: usize = 5120;
const KV_HEADS: usize = 4;
const HEAD_DIM: usize = 256;
const MAX_SEQ_LEN: usize = 32768;
const WARMUPS: usize = 2;
const PAIRS: usize = 5;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_dir = std::env::args()
        .nth(1)
        .ok_or("usage: qwen35_hybrid_prefill_probe MODEL_DIR")?;
    let start = std::env::var("APXINF_PREFILL_START")
        .unwrap_or_else(|_| "1024".into())
        .parse::<usize>()?;
    let stack_layers = std::env::var("APXINF_PREFILL_LAYERS")
        .unwrap_or_else(|_| "4".into())
        .parse::<usize>()?;
    if !matches!(stack_layers, 4 | 64) {
        return Err("APXINF_PREFILL_LAYERS must be 4 or 64".into());
    }
    if start + TOKENS > MAX_SEQ_LEN {
        return Err(format!("APXINF_PREFILL_START must be <= {}", MAX_SEQ_LEN - TOKENS).into());
    }
    let manifest = safetensors::inspect_path(Path::new(&model_dir))?;
    let context = CudaContext::new(0)?;
    if context.caps().sm != 89 {
        return Err(format!(
            "hybrid prefill probe is frozen for SM89, got SM{}",
            context.caps().sm
        )
        .into());
    }
    let values = deterministic_input();
    let tile = Tensor::from_bf16(vec![TOKENS, HIDDEN], &values)?;
    let rows = input_rows(&values)?;
    let zero_key = Tensor::zeros(vec![KV_HEADS, MAX_SEQ_LEN, HEAD_DIM], DType::BF16);
    let zero_value = Tensor::zeros(vec![KV_HEADS, MAX_SEQ_LEN, HEAD_DIM], DType::BF16);
    let unit = if stack_layers == 4 {
        HybridUnit::load_first(&manifest, &context, MAX_SEQ_LEN)?
    } else {
        HybridUnit::load_all(&manifest, &context, MAX_SEQ_LEN)?
    };

    unit.reset(&context, &rows[0], &zero_key, &zero_value)?;
    let (serial_residual, serial_normalized) = run_serial_capture(&context, &unit, &rows, start)?;
    unit.reset(&context, &rows[0], &zero_key, &zero_value)?;
    unit.set_prefill8_input(&context, &tile)?;
    unit.forward_prefill8(&context, start, false)?;
    context.synchronize()?;
    let candidate_residual = transfers::to_cpu(unit.prefill_output())?
        .as_bf16()?
        .to_vec();
    let candidate_normalized = transfers::to_cpu(unit.prefill_normalized_output())?
        .as_bf16()?
        .to_vec();
    let residual_different = different(&serial_residual, &candidate_residual);
    let normalized_different = different(&serial_normalized, &candidate_normalized);
    if residual_different != 0 || normalized_different != 0 {
        return Err(format!(
            "M8 hybrid prefill differs from serial M1: residual={residual_different}, normalized={normalized_different}"
        ).into());
    }

    for _ in 0..WARMUPS {
        unit.reset(&context, &rows[0], &zero_key, &zero_value)?;
        run_serial(&context, &unit, &rows, start)?;
        unit.reset(&context, &rows[0], &zero_key, &zero_value)?;
        unit.set_prefill8_input(&context, &tile)?;
        unit.forward_prefill8(&context, start, false)?;
        context.synchronize()?;
    }
    let mut serial_samples = Vec::with_capacity(PAIRS);
    let mut candidate_samples = Vec::with_capacity(PAIRS);
    let mut records = Vec::with_capacity(2 * PAIRS);
    for pair in 0..PAIRS {
        let candidate_first = pair % 2 == 1;
        for order_index in 0..2 {
            let candidate_arm = (order_index == 0) == candidate_first;
            unit.reset(&context, &rows[0], &zero_key, &zero_value)?;
            let begin = Instant::now();
            if candidate_arm {
                unit.set_prefill8_input(&context, &tile)?;
                unit.forward_prefill8(&context, start, false)?;
                context.synchronize()?;
            } else {
                run_serial(&context, &unit, &rows, start)?;
            }
            let elapsed = begin.elapsed().as_secs_f64() * 1.0e6;
            if candidate_arm {
                candidate_samples.push(elapsed)
            } else {
                serial_samples.push(elapsed)
            }
            records.push(serde_json::json!({
                "pair":pair,"order":if candidate_first{"BA"}else{"AB"},
                "order_index":order_index,"arm":if candidate_arm{"m8"}else{"serial_m1"},
                "us_per_8_tokens":elapsed,
            }));
        }
    }
    let speedups = serial_samples
        .iter()
        .zip(&candidate_samples)
        .map(|(serial, candidate)| serial / candidate)
        .collect::<Vec<_>>();
    let wins = serial_samples
        .iter()
        .zip(&candidate_samples)
        .filter(|(serial, candidate)| candidate < serial)
        .count();
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
        "schema":"apxinf.qwen35.hybrid_stack_prefill_probe.v1",
        "model_dir":model_dir,"layer_count":unit.layer_count(),
        "schedule":"repeating [linear_attention,linear_attention,linear_attention,full_attention]",
            "contract":{
                "tokens":TOKENS,"start_position":start,"end_position":start+TOKENS-1,
                "hidden":HIDDEN,"max_seq_len":MAX_SEQ_LEN,"kv_dtype":"bf16",
                "baseline":"eight canonical ModelOptimized one-token forwards",
                "candidate":"canonical M8 norms, W4 projections, stateful GDN scans, causal attention, residuals, and MLPs",
                "input_transfer":"CPU BF16 tile/rows to resident CUDA workspace included; state/cache reset excluded",
            },
            "device":{
                "name":context.caps().device_name,"sm":context.caps().sm,
                "multiprocessors":context.caps().multiprocessor_count,
                "cuda":context.library_versions().cuda,"cublas":context.library_versions().cublas,
            },
            "kernel_build_id":KERNEL_BUILD_ID,
            "correctness":{
                "final_residual_different_bf16":residual_different,
                "next_normalized_different_bf16":normalized_different,
                "comparison":"all 8 final residual and next-layer normalized rows bitwise identical",
                "pass":true,
            },
            "timing":{
            "boundary":format!("CPU input publication through {} complete layers and stream synchronize; state/cache reset excluded",unit.layer_count()),
                "warmups_per_arm":WARMUPS,"pairs":PAIRS,"records":records,
                "serial_raw_us":serial_samples,"candidate_raw_us":candidate_samples,
                "serial_median_us":median(&serial_samples),"candidate_median_us":median(&candidate_samples),
                "median_speedup":median(&speedups),"candidate_wins":wins,
                "candidate_tokens_per_second":TOKENS as f64*1.0e6/median(&candidate_samples),
            },
        "evidence_level":if stack_layers==4{"canonical-four-layer-prefill"}else{"canonical-64-layer-prefill"},
            "model_promoted":false,
        }))?
    );
    Ok(())
}

fn run_serial_capture(
    context: &CudaContext,
    unit: &HybridUnit,
    rows: &[Tensor],
    start: usize,
) -> Result<(Vec<half::bf16>, Vec<half::bf16>), Box<dyn std::error::Error>> {
    let mut residual = Vec::with_capacity(TOKENS * HIDDEN);
    let mut normalized = Vec::with_capacity(TOKENS * HIDDEN);
    for (token, input) in rows.iter().enumerate() {
        unit.set_token_input(context, input)?;
        unit.forward(
            context,
            HybridUnitMode::ModelOptimized,
            start + token + 1,
            (start + token) as u32,
            false,
        )?;
        context.synchronize()?;
        residual.extend_from_slice(transfers::to_cpu(unit.output())?.as_bf16()?);
        normalized.extend_from_slice(transfers::to_cpu(unit.normalized_output())?.as_bf16()?);
    }
    Ok((residual, normalized))
}

fn run_serial(
    context: &CudaContext,
    unit: &HybridUnit,
    rows: &[Tensor],
    start: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    for (token, input) in rows.iter().enumerate() {
        unit.set_token_input(context, input)?;
        unit.forward(
            context,
            HybridUnitMode::ModelOptimized,
            start + token + 1,
            (start + token) as u32,
            false,
        )?;
    }
    context.synchronize()?;
    Ok(())
}

fn input_rows(values: &[half::bf16]) -> Result<Vec<Tensor>, Box<dyn std::error::Error>> {
    (0..TOKENS)
        .map(|token| {
            Ok(Tensor::from_bf16(
                vec![1, HIDDEN],
                &values[token * HIDDEN..(token + 1) * HIDDEN],
            )?)
        })
        .collect()
}

fn deterministic_input() -> Vec<half::bf16> {
    (0..TOKENS * HIDDEN)
        .map(|index| {
            let token = index / HIDDEN;
            let column = index % HIDDEN;
            let phase = column as f32 * 0.004_882_812_5 + token as f32 * 0.101_562_5;
            half::bf16::from_f32((phase.sin() + 0.2 * phase.cos()) * 0.2)
        })
        .collect()
}

fn different(left: &[half::bf16], right: &[half::bf16]) -> usize {
    left.iter()
        .zip(right)
        .filter(|(left, right)| left.to_bits() != right.to_bits())
        .count()
}

fn median(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(|left, right| left.partial_cmp(right).unwrap_or(Ordering::Equal));
    if sorted.len() % 2 == 0 {
        (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) * 0.5
    } else {
        sorted[sorted.len() / 2]
    }
}
