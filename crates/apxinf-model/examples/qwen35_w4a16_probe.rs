//! Real-checkpoint correctness and latency probe for the native M=1 W4A16 GEMV.

use std::cmp::Ordering;
use std::path::Path;
use std::time::Instant;

use apxinf_core::{DType, Tensor};
use apxinf_cuda::kernels::gemm::{self, W4A16Layout, W4A16WeightView};
use apxinf_cuda::tuning::KERNEL_BUILD_ID;
use apxinf_cuda::{transfers, CudaBuffer, CudaContext};
use apxinf_loader::safetensors;

const DEFAULT_BASE: &str = "model.language_model.layers.0.mlp.gate_proj";
const WARMUPS: usize = 20;
const BLOCKS: usize = 30;
const CALLS_PER_BLOCK: usize = 100;
const COLD_BLOCKS: usize = 30;
const L2_EVICTION_BYTES: usize = 128 * 1024 * 1024;

#[derive(Clone, Copy, Debug)]
struct ErrorMetrics {
    cosine: f64,
    relative_l2: f64,
    max_abs: f64,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut arguments = std::env::args().skip(1);
    let model_dir = arguments
        .next()
        .ok_or("usage: qwen35_w4a16_probe MODEL_DIR [TENSOR_BASE]")?;
    let base = arguments.next().unwrap_or_else(|| DEFAULT_BASE.to_owned());
    if arguments.next().is_some() {
        return Err("usage: qwen35_w4a16_probe MODEL_DIR [TENSOR_BASE]".into());
    }
    let implementation = std::env::var("APXINF_W4A16_IMPL").unwrap_or_else(|_| "direct".into());
    if implementation != "direct" && implementation != "staged" {
        return Err(
            format!("APXINF_W4A16_IMPL must be direct or staged, got `{implementation}`").into(),
        );
    }
    let staged = implementation == "staged";

    let manifest = safetensors::inspect_path(Path::new(&model_dir))?;
    let packed = load(&manifest, &format!("{base}.weight_packed"))?;
    let scales = load(&manifest, &format!("{base}.weight_scale"))?;
    let zero_points = load(&manifest, &format!("{base}.weight_zero_point"))?;
    let logical_shape = safetensors::read_small_i64(
        manifest
            .tensor(&format!("{base}.weight_shape"))
            .ok_or_else(|| format!("missing {base}.weight_shape"))?,
    )?;
    if logical_shape.len() != 2 {
        return Err(format!("invalid logical shape {logical_shape:?}").into());
    }
    let output_dim = usize::try_from(logical_shape[0])?;
    let input_dim = usize::try_from(logical_shape[1])?;

    let activation_values = (0..input_dim)
        .map(|index| {
            let phase = index as f32 * 0.013_671 + (index % 17) as f32 * 0.007_812_5;
            half::bf16::from_f32(phase.sin() * 0.25 + phase.cos() * 0.031_25)
        })
        .collect::<Vec<_>>();
    let activation = Tensor::from_bf16(vec![1, input_dim], &activation_values)?;

    let context = CudaContext::new(0)?;
    if context.caps().sm != 89 {
        return Err(format!("probe is frozen for SM89, got SM{}", context.caps().sm).into());
    }
    let activation_gpu = transfers::to_cuda(&activation, 0)?;
    let packed_gpu = transfers::to_cuda(&packed, 0)?;
    let scales_gpu = transfers::to_cuda(&scales, 0)?;
    let zero_points_gpu = transfers::to_cuda(&zero_points, 0)?;
    let output_gpu = transfers::to_cuda(&Tensor::zeros(vec![1, output_dim], DType::BF16), 0)?;
    let weight = W4A16WeightView {
        packed_i32: &packed_gpu,
        scales_bf16: &scales_gpu,
        zero_points_i32: &zero_points_gpu,
        input_dim,
        output_dim,
        group_size: 32,
        layout: W4A16Layout::CompressedTensorsPackQuantized,
    };

    write(staged, &context, &activation_gpu, weight, &output_gpu)?;
    context.synchronize()?;
    let actual = transfers::to_cpu(&output_gpu)?.to_f32_vec()?;
    let expected = cpu_oracle(&activation_values, &packed, &scales, &zero_points)?;
    let metrics = error_metrics(&actual, &expected)?;
    if metrics.cosine < 0.999_99 || metrics.relative_l2 > 0.005 {
        return Err(format!("W4A16 correctness gate failed: {metrics:?}").into());
    }

    for _ in 0..WARMUPS {
        write(staged, &context, &activation_gpu, weight, &output_gpu)?;
    }
    context.synchronize()?;
    let mut samples_us = Vec::with_capacity(BLOCKS);
    for _ in 0..BLOCKS {
        let start = Instant::now();
        for _ in 0..CALLS_PER_BLOCK {
            write(staged, &context, &activation_gpu, weight, &output_gpu)?;
        }
        context.synchronize()?;
        samples_us.push(start.elapsed().as_secs_f64() * 1.0e6 / CALLS_PER_BLOCK as f64);
    }
    let eviction = CudaBuffer::alloc(L2_EVICTION_BYTES, 0)?;
    let mut cold_samples_us = Vec::with_capacity(COLD_BLOCKS);
    for block in 0..COLD_BLOCKS {
        eviction.memset_async(block as u8, context.stream())?;
        context.synchronize()?;
        let start = Instant::now();
        write(staged, &context, &activation_gpu, weight, &output_gpu)?;
        context.synchronize()?;
        cold_samples_us.push(start.elapsed().as_secs_f64() * 1.0e6);
    }
    let median_us = median(&samples_us);
    let mean_us = samples_us.iter().sum::<f64>() / samples_us.len() as f64;
    let standard_deviation_us = (samples_us
        .iter()
        .map(|sample| (sample - mean_us).powi(2))
        .sum::<f64>()
        / samples_us.len() as f64)
        .sqrt();
    let cold_median_us = median(&cold_samples_us);
    let cold_mean_us = cold_samples_us.iter().sum::<f64>() / cold_samples_us.len() as f64;
    let cold_standard_deviation_us = (cold_samples_us
        .iter()
        .map(|sample| (sample - cold_mean_us).powi(2))
        .sum::<f64>()
        / cold_samples_us.len() as f64)
        .sqrt();

    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "schema": "apxinf.qwen35.w4a16_probe.v1",
            "model_dir": model_dir,
            "tensor_base": base,
            "implementation": implementation,
            "logical_shape": [output_dim, input_dim],
            "physical_shapes": {
                "packed_i32": packed.shape().dims(),
                "scales_bf16": scales.shape().dims(),
                "zero_points_i32": zero_points.shape().dims(),
            },
            "contract": {
                "bits": 4,
                "group_size": 32,
                "symmetric": false,
                "activation_dtype": "bf16",
                "output_dtype": "bf16",
                "m": 1,
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
                "oracle": "compressed-tensors offset-binary unpack and asymmetric group dequant, CPU f32 dot, BF16 final round",
                "cosine": metrics.cosine,
                "relative_l2": metrics.relative_l2,
                "max_abs": metrics.max_abs,
                "pass": true,
            },
            "timing": {
                "hot_l2": {
                    "boundary": "host launches to stream synchronize, amortized per call, no profiler",
                    "warmups": WARMUPS,
                    "blocks": BLOCKS,
                    "calls_per_block": CALLS_PER_BLOCK,
                    "raw_us_per_call": samples_us,
                    "median_us": median_us,
                    "mean_us": mean_us,
                    "standard_deviation_us": standard_deviation_us,
                    "cv": standard_deviation_us / mean_us,
                },
                "cold_hbm_proxy": {
                    "boundary": "128 MiB stream memset and synchronize outside timing, then one host launch to stream synchronize",
                    "blocks": COLD_BLOCKS,
                    "eviction_bytes": L2_EVICTION_BYTES,
                    "raw_us_per_call": cold_samples_us,
                    "median_us": cold_median_us,
                    "mean_us": cold_mean_us,
                    "standard_deviation_us": cold_standard_deviation_us,
                    "cv": cold_standard_deviation_us / cold_mean_us,
                },
            },
            "evidence_level": "operator-only",
            "promoted": false,
        }))?
    );
    Ok(())
}

fn write(
    staged: bool,
    context: &CudaContext,
    activation: &Tensor,
    weight: W4A16WeightView<'_>,
    output: &Tensor,
) -> apxinf_core::Result<()> {
    if staged {
        gemm::w4a16_write(context, activation, weight, output)
    } else {
        gemm::w4a16_write_direct(context, activation, weight, output)
    }
}

fn load(
    manifest: &safetensors::CheckpointManifest,
    name: &str,
) -> Result<Tensor, Box<dyn std::error::Error>> {
    let entry = manifest
        .tensor(name)
        .ok_or_else(|| format!("missing checkpoint tensor `{name}`"))?;
    Ok(safetensors::load_manifest_tensor(entry)?)
}

fn cpu_oracle(
    activation: &[half::bf16],
    packed: &Tensor,
    scales: &Tensor,
    zero_points: &Tensor,
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let output_dim = packed.shape().dims()[0];
    let packed_cols = packed.shape().dims()[1];
    let input_dim = packed_cols * 8;
    let groups = input_dim / 32;
    let packed = packed.as_i32()?;
    let scales = scales.as_bf16()?;
    let zero_points = zero_points.as_i32()?;
    let mut output = vec![0.0_f32; output_dim];
    for row in 0..output_dim {
        let mut accumulator = 0.0_f32;
        for packed_col in 0..packed_cols {
            let weight_word = packed[row * packed_cols + packed_col] as u32;
            let group = packed_col / 4;
            let scale = scales[row * groups + group].to_f32();
            let zero_word = zero_points[(row / 8) * groups + group] as u32;
            let zero = (((zero_word >> ((row & 7) * 4)) & 0x0f) as i32) - 8;
            for index in 0..8 {
                let quantized = (((weight_word >> (index * 4)) & 0x0f) as i32) - 8;
                let weight = (quantized - zero) as f32 * scale;
                accumulator = activation[packed_col * 8 + index]
                    .to_f32()
                    .mul_add(weight, accumulator);
            }
        }
        output[row] = half::bf16::from_f32(accumulator).to_f32();
    }
    Ok(output)
}

fn error_metrics(actual: &[f32], expected: &[f32]) -> Result<ErrorMetrics, String> {
    if actual.len() != expected.len() || actual.is_empty() {
        return Err(format!(
            "comparison length mismatch: {} versus {}",
            actual.len(),
            expected.len()
        ));
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

fn median(values: &[f64]) -> f64 {
    let mut values = values.to_vec();
    values.sort_by(|left, right| left.partial_cmp(right).unwrap_or(Ordering::Equal));
    if values.len() % 2 == 0 {
        (values[values.len() / 2 - 1] + values[values.len() / 2]) * 0.5
    } else {
        values[values.len() / 2]
    }
}
