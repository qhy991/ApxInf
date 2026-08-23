//! Qwen3.5 recurrent GDN core: 128-step state/output correctness and latency.

use std::cmp::Ordering;
use std::time::Instant;

use apxinf_core::{DType, Tensor};
use apxinf_cuda::kernels::gdn::{
    qwen35_recurrent_write, QWEN35_GDN_HEADS as HEADS, QWEN35_GDN_KEY_DIM as KEY_DIM,
    QWEN35_GDN_VALUE_DIM as VALUE_DIM,
};
use apxinf_cuda::tuning::KERNEL_BUILD_ID;
use apxinf_cuda::{transfers, CudaBuffer, CudaContext};

const CORRECTNESS_STEPS: usize = 128;
const WARMUPS: usize = 20;
const BLOCKS: usize = 30;
const CALLS_PER_BLOCK: usize = 200;
const COLD_BLOCKS: usize = 30;
const L2_EVICTION_BYTES: usize = 128 * 1024 * 1024;

#[derive(Clone, Copy, Debug)]
struct ErrorMetrics {
    cosine: f64,
    relative_l2: f64,
    max_abs: f64,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let context = CudaContext::new(0)?;
    if context.caps().sm != 89 {
        return Err(format!("probe is frozen for SM89, got SM{}", context.caps().sm).into());
    }
    let vector_shape = vec![HEADS, KEY_DIM];
    let scalar_shape = vec![HEADS];
    let state_shape = vec![HEADS, KEY_DIM, VALUE_DIM];
    let query_gpu = transfers::to_cuda(&Tensor::zeros(vector_shape.clone(), DType::BF16), 0)?;
    let key_gpu = transfers::to_cuda(&Tensor::zeros(vector_shape.clone(), DType::BF16), 0)?;
    let value_gpu = transfers::to_cuda(&Tensor::zeros(vector_shape.clone(), DType::BF16), 0)?;
    let g_gpu = transfers::to_cuda(&Tensor::zeros(scalar_shape.clone(), DType::F32), 0)?;
    let beta_gpu = transfers::to_cuda(&Tensor::zeros(scalar_shape.clone(), DType::F32), 0)?;
    let state_gpu = transfers::to_cuda(&Tensor::zeros(state_shape.clone(), DType::F32), 0)?;
    let output_gpu = transfers::to_cuda(&Tensor::zeros(vector_shape.clone(), DType::BF16), 0)?;
    let mut expected_state = vec![0.0_f32; HEADS * KEY_DIM * VALUE_DIM];
    let mut expected_output = vec![0.0_f32; HEADS * VALUE_DIM];
    let mut last_inputs = inputs(0)?;

    for step in 0..CORRECTNESS_STEPS {
        last_inputs = inputs(step)?;
        transfers::copy_cpu_to_cuda(&last_inputs.0, &query_gpu)?;
        transfers::copy_cpu_to_cuda(&last_inputs.1, &key_gpu)?;
        transfers::copy_cpu_to_cuda(&last_inputs.2, &value_gpu)?;
        transfers::copy_cpu_to_cuda(&last_inputs.3, &g_gpu)?;
        transfers::copy_cpu_to_cuda(&last_inputs.4, &beta_gpu)?;
        qwen35_recurrent_write(
            &context,
            &query_gpu,
            &key_gpu,
            &value_gpu,
            &g_gpu,
            &beta_gpu,
            &state_gpu,
            &output_gpu,
        )?;
        context.synchronize()?;
        cpu_step(&last_inputs, &mut expected_state, &mut expected_output)?;
    }

    let actual_output = transfers::to_cpu(&output_gpu)?.to_f32_vec()?;
    let actual_state = transfers::to_cpu(&state_gpu)?.to_f32_vec()?;
    let output_metrics = error_metrics(&actual_output, &expected_output)?;
    let state_metrics = error_metrics(&actual_state, &expected_state)?;
    if output_metrics.cosine < 0.9999 || state_metrics.relative_l2 > 0.001 {
        return Err(format!(
            "GDN correctness gate failed: output={output_metrics:?}, state={state_metrics:?}"
        )
        .into());
    }
    verify_input_immutability(
        &last_inputs,
        [&query_gpu, &key_gpu, &value_gpu, &g_gpu, &beta_gpu],
    )?;

    transfers::copy_cpu_to_cuda(&Tensor::zeros(state_shape, DType::F32), &state_gpu)?;
    for _ in 0..WARMUPS {
        launch(
            &context,
            [&query_gpu, &key_gpu, &value_gpu, &g_gpu, &beta_gpu],
            &state_gpu,
            &output_gpu,
        )?;
    }
    context.synchronize()?;
    let mut hot_samples_us = Vec::with_capacity(BLOCKS);
    for _ in 0..BLOCKS {
        let start = Instant::now();
        for _ in 0..CALLS_PER_BLOCK {
            launch(
                &context,
                [&query_gpu, &key_gpu, &value_gpu, &g_gpu, &beta_gpu],
                &state_gpu,
                &output_gpu,
            )?;
        }
        context.synchronize()?;
        hot_samples_us.push(start.elapsed().as_secs_f64() * 1.0e6 / CALLS_PER_BLOCK as f64);
    }
    let eviction = CudaBuffer::alloc(L2_EVICTION_BYTES, 0)?;
    let mut cold_samples_us = Vec::with_capacity(COLD_BLOCKS);
    for block in 0..COLD_BLOCKS {
        eviction.memset_async(block as u8, context.stream())?;
        context.synchronize()?;
        let start = Instant::now();
        launch(
            &context,
            [&query_gpu, &key_gpu, &value_gpu, &g_gpu, &beta_gpu],
            &state_gpu,
            &output_gpu,
        )?;
        context.synchronize()?;
        cold_samples_us.push(start.elapsed().as_secs_f64() * 1.0e6);
    }

    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "schema": "apxinf.qwen35.gdn_core_probe.v1",
            "contract": {
                "heads": HEADS,
                "key_dim": KEY_DIM,
                "value_dim": VALUE_DIM,
                "query_key_value_dtype": "bf16",
                "g_beta_state_dtype": "f32",
                "output_dtype": "bf16",
                "state_update": "in_place",
            },
            "device": {
                "name": context.caps().device_name,
                "sm": context.caps().sm,
                "multiprocessors": context.caps().multiprocessor_count,
                "cuda": context.library_versions().cuda,
                "cublas": context.library_versions().cublas,
            },
            "kernel_build_id": KERNEL_BUILD_ID,
            "correctness": {
                "steps": CORRECTNESS_STEPS,
                "output": metrics_json(output_metrics),
                "recurrent_state": metrics_json(state_metrics),
                "input_immutable": true,
                "finite": true,
                "pass": true,
            },
            "timing": {
                "hot_l2": timing_json(&hot_samples_us, Some((WARMUPS, CALLS_PER_BLOCK))),
                "cold_hbm_proxy": {
                    "eviction_bytes": L2_EVICTION_BYTES,
                    "boundary": "128 MiB stream memset and synchronize outside timing, then one host launch to stream synchronize",
                    "samples": timing_json(&cold_samples_us, None),
                },
            },
            "evidence_level": "operator-only",
            "promoted": false,
        }))?
    );
    Ok(())
}

type Inputs = (Tensor, Tensor, Tensor, Tensor, Tensor);

fn inputs(step: usize) -> Result<Inputs, Box<dyn std::error::Error>> {
    let count = HEADS * KEY_DIM;
    let make_vector = |salt: usize, scale: f32| {
        (0..count)
            .map(|index| {
                let phase = (index * (salt * 2 + 3) + step * (salt * 5 + 7)) as f32 * 0.001_953_125;
                half::bf16::from_f32((phase.sin() + 0.25 * phase.cos()) * scale)
            })
            .collect::<Vec<_>>()
    };
    let query = Tensor::from_bf16(vec![HEADS, KEY_DIM], &make_vector(1, 0.5))?;
    let key = Tensor::from_bf16(vec![HEADS, KEY_DIM], &make_vector(2, 0.5))?;
    let value = Tensor::from_bf16(vec![HEADS, VALUE_DIM], &make_vector(3, 0.25))?;
    let g = (0..HEADS)
        .map(|head| -0.01 - 0.0025 * ((head + step) % 7) as f32)
        .collect::<Vec<_>>();
    let beta = (0..HEADS)
        .map(|head| 0.2 + 0.1 * ((head * 3 + step) % 7) as f32)
        .collect::<Vec<_>>();
    Ok((
        query,
        key,
        value,
        Tensor::from_f32(vec![HEADS], &g)?,
        Tensor::from_f32(vec![HEADS], &beta)?,
    ))
}

fn cpu_step(
    inputs: &Inputs,
    state: &mut [f32],
    output: &mut [f32],
) -> Result<(), Box<dyn std::error::Error>> {
    let query = inputs.0.as_bf16()?;
    let key = inputs.1.as_bf16()?;
    let value = inputs.2.as_bf16()?;
    let g = inputs.3.as_f32()?;
    let beta = inputs.4.as_f32()?;
    let query_scale = 1.0_f32 / (KEY_DIM as f32).sqrt();
    for head in 0..HEADS {
        let offset = head * KEY_DIM;
        let query_sum = query[offset..offset + KEY_DIM]
            .iter()
            .map(|value| value.to_f32().powi(2))
            .sum::<f32>();
        let key_sum = key[offset..offset + KEY_DIM]
            .iter()
            .map(|value| value.to_f32().powi(2))
            .sum::<f32>();
        let query_normalizer = (query_sum + 1.0e-6).sqrt().recip() * query_scale;
        let key_normalizer = (key_sum + 1.0e-6).sqrt().recip();
        let normalized_query = query[offset..offset + KEY_DIM]
            .iter()
            .map(|value| value.to_f32() * query_normalizer)
            .collect::<Vec<_>>();
        let normalized_key = key[offset..offset + KEY_DIM]
            .iter()
            .map(|value| value.to_f32() * key_normalizer)
            .collect::<Vec<_>>();
        let qk = normalized_query
            .iter()
            .zip(&normalized_key)
            .map(|(q, k)| q * k)
            .sum::<f32>();
        let decay = g[head].exp();
        let state_base = head * KEY_DIM * VALUE_DIM;
        for value_dimension in 0..VALUE_DIM {
            let mut key_memory = 0.0_f32;
            let mut query_memory = 0.0_f32;
            for key_dimension in 0..KEY_DIM {
                let old = state[state_base + key_dimension * VALUE_DIM + value_dimension];
                key_memory = old.mul_add(normalized_key[key_dimension], key_memory);
                query_memory = old.mul_add(normalized_query[key_dimension], query_memory);
            }
            let delta =
                (value[offset + value_dimension].to_f32() - decay * key_memory) * beta[head];
            output[offset + value_dimension] =
                half::bf16::from_f32(decay * query_memory + delta * qk).to_f32();
            for key_dimension in 0..KEY_DIM {
                let index = state_base + key_dimension * VALUE_DIM + value_dimension;
                state[index] = normalized_key[key_dimension].mul_add(delta, state[index] * decay);
            }
        }
    }
    Ok(())
}

fn launch(
    context: &CudaContext,
    inputs: [&Tensor; 5],
    state: &Tensor,
    output: &Tensor,
) -> apxinf_core::Result<()> {
    qwen35_recurrent_write(
        context, inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], state, output,
    )
}

fn verify_input_immutability(
    expected: &Inputs,
    actual: [&Tensor; 5],
) -> Result<(), Box<dyn std::error::Error>> {
    let downloaded = actual
        .iter()
        .map(|tensor| transfers::to_cpu(tensor))
        .collect::<apxinf_core::Result<Vec<_>>>()?;
    if downloaded[0].as_bf16()? != expected.0.as_bf16()?
        || downloaded[1].as_bf16()? != expected.1.as_bf16()?
        || downloaded[2].as_bf16()? != expected.2.as_bf16()?
        || downloaded[3]
            .as_f32()?
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>()
            != expected
                .3
                .as_f32()?
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        || downloaded[4]
            .as_f32()?
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>()
            != expected
                .4
                .as_f32()?
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
    {
        return Err("GDN kernel mutated an input tensor".into());
    }
    Ok(())
}

fn error_metrics(actual: &[f32], expected: &[f32]) -> Result<ErrorMetrics, String> {
    if actual.len() != expected.len() || actual.is_empty() {
        return Err("comparison length mismatch".into());
    }
    let mut dot = 0.0_f64;
    let mut actual_squared = 0.0_f64;
    let mut expected_squared = 0.0_f64;
    let mut error_squared = 0.0_f64;
    let mut max_abs = 0.0_f64;
    for (&actual, &expected) in actual.iter().zip(expected) {
        if !actual.is_finite() || !expected.is_finite() {
            return Err("comparison contains non-finite values".into());
        }
        let actual = f64::from(actual);
        let expected = f64::from(expected);
        let error = actual - expected;
        dot += actual * expected;
        actual_squared += actual * actual;
        expected_squared += expected * expected;
        error_squared += error * error;
        max_abs = max_abs.max(error.abs());
    }
    Ok(ErrorMetrics {
        cosine: dot / (actual_squared.sqrt() * expected_squared.sqrt()),
        relative_l2: (error_squared / expected_squared).sqrt(),
        max_abs,
    })
}

fn metrics_json(metrics: ErrorMetrics) -> serde_json::Value {
    serde_json::json!({
        "cosine": metrics.cosine,
        "relative_l2": metrics.relative_l2,
        "max_abs": metrics.max_abs,
    })
}

fn timing_json(samples: &[f64], loop_contract: Option<(usize, usize)>) -> serde_json::Value {
    let mean = samples.iter().sum::<f64>() / samples.len() as f64;
    let standard_deviation = (samples
        .iter()
        .map(|sample| (sample - mean).powi(2))
        .sum::<f64>()
        / samples.len() as f64)
        .sqrt();
    let mut result = serde_json::json!({
        "raw_us_per_call": samples,
        "median_us": median(samples),
        "mean_us": mean,
        "standard_deviation_us": standard_deviation,
        "cv": standard_deviation / mean,
    });
    if let Some((warmups, calls)) = loop_contract {
        result["warmups"] = warmups.into();
        result["blocks"] = BLOCKS.into();
        result["calls_per_block"] = calls.into();
    }
    result
}

fn median(values: &[f64]) -> f64 {
    let mut values = values.to_vec();
    values.sort_by(|left, right| left.partial_cmp(right).unwrap_or(Ordering::Equal));
    if values.len() % 2 == 0 {
        (values[values.len() / 2 - 1] + values[values.len() / 2]) * 0.5
    } else {
        values[values.len() / 2]
    }
}
