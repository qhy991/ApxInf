//! Real-checkpoint small-M W4A16 prefill projection and serial-M1 baseline.

use std::cmp::Ordering;
use std::path::Path;
use std::time::Instant;

use apxinf_core::{DType, Tensor};
use apxinf_cuda::kernels::gemm::{self, W4A16Layout, W4A16WeightView};
use apxinf_cuda::tuning::KERNEL_BUILD_ID;
use apxinf_cuda::{transfers, CudaBuffer, CudaContext};
use apxinf_loader::safetensors::{self, CheckpointManifest};

const INPUT: usize = 5120;
const OUTPUT: usize = 17408;
const BASE: &str = "model.language_model.layers.0.mlp.gate_proj";
const PAIRS: usize = 5;
const CALLS: usize = 3;
const L2_EVICTION_BYTES: usize = 128 * 1024 * 1024;

struct W4 {
    packed: Tensor,
    scales: Tensor,
    zero: Tensor,
}

impl W4 {
    fn view(&self) -> W4A16WeightView<'_> {
        W4A16WeightView {
            packed_i32: &self.packed,
            scales_bf16: &self.scales,
            zero_points_i32: &self.zero,
            input_dim: INPUT,
            output_dim: OUTPUT,
            group_size: 32,
            layout: W4A16Layout::CompressedTensorsPackQuantized,
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_dir = std::env::args()
        .nth(1)
        .ok_or("usage: qwen35_w4a16_prefill_probe MODEL_DIR")?;
    let tokens = std::env::var("APXINF_PREFILL_M")
        .unwrap_or_else(|_| "8".into())
        .parse::<usize>()?;
    if tokens == 0 || tokens > 8 {
        return Err("APXINF_PREFILL_M must be within 1..=8".into());
    }
    let manifest = safetensors::inspect_path(Path::new(&model_dir))?;
    let context = CudaContext::new(0)?;
    if context.caps().sm != 89 {
        return Err(format!(
            "prefill probe is frozen for SM89, got SM{}",
            context.caps().sm
        )
        .into());
    }
    let weight = load_w4(&manifest)?;
    let activation_values = deterministic_activations(tokens);
    let activation_cpu = Tensor::from_bf16(vec![tokens, INPUT], &activation_values)?;
    let activation = transfers::to_cuda(&activation_cpu, 0)?;
    let candidate = gpu_zeros(&[tokens, OUTPUT])?;
    let mut serial_inputs = Vec::with_capacity(tokens);
    let mut serial_outputs = Vec::with_capacity(tokens);
    for token in 0..tokens {
        serial_inputs.push(transfers::to_cuda(
            &Tensor::from_bf16(
                vec![1, INPUT],
                &activation_values[token * INPUT..(token + 1) * INPUT],
            )?,
            0,
        )?);
        serial_outputs.push(gpu_zeros(&[1, OUTPUT])?);
    }

    run_serial(&context, &serial_inputs, &serial_outputs, &weight)?;
    gemm::w4a16_m8_write(&context, &activation, weight.view(), &candidate)?;
    context.synchronize()?;
    let candidate_cpu = transfers::to_cpu(&candidate)?;
    let candidate_values = candidate_cpu.as_bf16()?;
    let mut different = 0usize;
    let mut max_abs = 0.0_f32;
    for token in 0..tokens {
        let serial = transfers::to_cpu(&serial_outputs[token])?;
        for (serial, candidate) in serial
            .as_bf16()?
            .iter()
            .zip(&candidate_values[token * OUTPUT..(token + 1) * OUTPUT])
        {
            if serial.to_bits() != candidate.to_bits() {
                different += 1;
                max_abs = max_abs.max((serial.to_f32() - candidate.to_f32()).abs());
            }
        }
    }
    if different != 0 {
        return Err(format!(
            "W4A16 M8 differs from serial M1 at {different} values, max_abs={max_abs}"
        )
        .into());
    }
    let input_immutable =
        transfers::to_cpu(&activation)?.as_bf16()? == activation_values.as_slice();
    if !input_immutable {
        return Err("W4A16 M8 modified its activation".into());
    }

    for _ in 0..3 {
        run_serial(&context, &serial_inputs, &serial_outputs, &weight)?;
        gemm::w4a16_m8_write(&context, &activation, weight.view(), &candidate)?;
    }
    context.synchronize()?;
    let mut records = Vec::with_capacity(2 * PAIRS);
    let mut serial_samples = Vec::with_capacity(PAIRS);
    let mut candidate_samples = Vec::with_capacity(PAIRS);
    for pair in 0..PAIRS {
        let candidate_first = pair % 2 == 1;
        for order_index in 0..2 {
            let candidate_arm = (order_index == 0) == candidate_first;
            let start = Instant::now();
            for _ in 0..CALLS {
                if candidate_arm {
                    gemm::w4a16_m8_write(&context, &activation, weight.view(), &candidate)?;
                } else {
                    run_serial(&context, &serial_inputs, &serial_outputs, &weight)?;
                }
            }
            context.synchronize()?;
            let elapsed = start.elapsed().as_secs_f64() * 1.0e6 / CALLS as f64;
            if candidate_arm {
                candidate_samples.push(elapsed);
            } else {
                serial_samples.push(elapsed);
            }
            records.push(serde_json::json!({
                "pair":pair,"order":if candidate_first{"BA"}else{"AB"},
                "order_index":order_index,
                "arm":if candidate_arm{"m8_reuse"}else{"serial_m1"},
                "us_per_tile":elapsed,
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
    let eviction = CudaBuffer::alloc(L2_EVICTION_BYTES, 0)?;
    let mut cold_serial = Vec::with_capacity(PAIRS);
    let mut cold_candidate = Vec::with_capacity(PAIRS);
    for pair in 0..PAIRS {
        let candidate_first = pair % 2 == 1;
        for order_index in 0..2 {
            let candidate_arm = (order_index == 0) == candidate_first;
            if candidate_arm {
                eviction.memset_async(pair as u8, context.stream())?;
                context.synchronize()?;
                let start = Instant::now();
                gemm::w4a16_m8_write(&context, &activation, weight.view(), &candidate)?;
                context.synchronize()?;
                cold_candidate.push(start.elapsed().as_secs_f64() * 1.0e6);
            } else {
                let mut total = 0.0_f64;
                for (token, (input, output)) in
                    serial_inputs.iter().zip(&serial_outputs).enumerate()
                {
                    eviction.memset_async((pair * tokens + token) as u8, context.stream())?;
                    context.synchronize()?;
                    let start = Instant::now();
                    gemm::w4a16_write(&context, input, weight.view(), output)?;
                    context.synchronize()?;
                    total += start.elapsed().as_secs_f64() * 1.0e6;
                }
                cold_serial.push(total);
            }
        }
    }
    let cold_speedups = cold_serial
        .iter()
        .zip(&cold_candidate)
        .map(|(serial, candidate)| serial / candidate)
        .collect::<Vec<_>>();
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "schema":"apxinf.qwen35.w4a16_prefill_probe.v1",
            "model_dir":model_dir,"tensor":BASE,
            "contract":{"tokens":tokens,"input":INPUT,"output":OUTPUT,"group_size":32,"dtype":"BF16xW4->BF16"},
            "kernel_build_id":KERNEL_BUILD_ID,
            "correctness":{"different_bf16_values":different,"max_abs":max_abs,"input_immutable":input_immutable,"pass":true},
            "timing":{"pairs":PAIRS,"calls_per_sample":CALLS,"records":records,
                "serial_raw_us":serial_samples,"candidate_raw_us":candidate_samples,
                "serial_median_us":median(&serial_samples),"candidate_median_us":median(&candidate_samples),
                "median_speedup":median(&speedups),"candidate_wins":wins,
                "candidate_tokens_per_second":tokens as f64*1.0e6/median(&candidate_samples),
                "cold_hbm_proxy":{"eviction_bytes":L2_EVICTION_BYTES,
                    "policy":"evict before each serial token; evict once before M-token candidate; eviction excluded from timing",
                    "serial_raw_us_per_tile":cold_serial,"candidate_raw_us_per_tile":cold_candidate,
                    "serial_median_us":median(&cold_serial),"candidate_median_us":median(&cold_candidate),
                    "median_speedup":median(&cold_speedups)}},
            "evidence_level":"operator-prefill","model_promoted":false,
        }))?
    );
    Ok(())
}

fn run_serial(
    context: &CudaContext,
    inputs: &[Tensor],
    outputs: &[Tensor],
    weight: &W4,
) -> apxinf_core::Result<()> {
    for (input, output) in inputs.iter().zip(outputs) {
        gemm::w4a16_write(context, input, weight.view(), output)?;
    }
    Ok(())
}

fn load_w4(manifest: &CheckpointManifest) -> Result<W4, Box<dyn std::error::Error>> {
    let load = |suffix: &str| {
        safetensors::load_manifest_tensor(
            manifest
                .tensor(&format!("{BASE}.{suffix}"))
                .ok_or_else(|| format!("missing {BASE}.{suffix}"))?,
        )
    };
    let shape = safetensors::read_small_i64(
        manifest
            .tensor(&format!("{BASE}.weight_shape"))
            .ok_or("missing logical shape")?,
    )?;
    if shape != [OUTPUT as i64, INPUT as i64] {
        return Err(format!("unexpected logical shape {shape:?}").into());
    }
    Ok(W4 {
        packed: transfers::to_cuda(&load("weight_packed")?, 0)?,
        scales: transfers::to_cuda(&load("weight_scale")?, 0)?,
        zero: transfers::to_cuda(&load("weight_zero_point")?, 0)?,
    })
}

fn deterministic_activations(tokens: usize) -> Vec<half::bf16> {
    (0..tokens * INPUT)
        .map(|index| {
            let token = index / INPUT;
            let column = index % INPUT;
            let phase = column as f32 * 0.005_859_375
                + token as f32 * 0.113_281_25
                + (column % 31) as f32 * 0.001_953_125;
            half::bf16::from_f32((phase.sin() + 0.25 * phase.cos()) * 0.25)
        })
        .collect()
}

fn gpu_zeros(shape: &[usize]) -> Result<Tensor, Box<dyn std::error::Error>> {
    Ok(transfers::to_cuda(
        &Tensor::zeros(shape.to_vec(), DType::BF16),
        0,
    )?)
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
