//! Real layer-0 MLP prefill tile using M<=8 W4 weight reuse.

use std::cmp::Ordering;
use std::path::Path;
use std::time::Instant;

use apxinf_core::{DType, Tensor};
use apxinf_cuda::kernels::activation;
use apxinf_cuda::kernels::gemm::{self, W4A16Layout, W4A16WeightView};
use apxinf_cuda::tuning::KERNEL_BUILD_ID;
use apxinf_cuda::{transfers, CudaBuffer, CudaContext};
use apxinf_loader::safetensors::{self, CheckpointManifest};

const HIDDEN: usize = 5120;
const INTERMEDIATE: usize = 17408;
const PREFIX: &str = "model.language_model.layers.0.mlp";
const TOKENS: usize = 8;
const PAIRS: usize = 5;
const CALLS: usize = 3;

struct W4 {
    packed: Tensor,
    scales: Tensor,
    zero: Tensor,
    input: usize,
    output: usize,
}

impl W4 {
    fn view(&self) -> W4A16WeightView<'_> {
        W4A16WeightView {
            packed_i32: &self.packed,
            scales_bf16: &self.scales,
            zero_points_i32: &self.zero,
            input_dim: self.input,
            output_dim: self.output,
            group_size: 32,
            layout: W4A16Layout::CompressedTensorsPackQuantized,
        }
    }
}

struct CandidateWorkspace {
    gate_up: Tensor,
    hidden: Tensor,
    output: Tensor,
}

struct SerialWorkspace {
    input: Vec<Tensor>,
    gate_up: Vec<Tensor>,
    hidden: Vec<Tensor>,
    output: Vec<Tensor>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_dir = std::env::args()
        .nth(1)
        .ok_or("usage: qwen35_mlp_prefill_probe MODEL_DIR")?;
    let manifest = safetensors::inspect_path(Path::new(&model_dir))?;
    let gate = load_w4_cpu(&manifest, &format!("{PREFIX}.gate_proj"))?;
    let up = load_w4_cpu(&manifest, &format!("{PREFIX}.up_proj"))?;
    let gate_up = combine_to_gpu(gate, up)?;
    let down = to_gpu(load_w4_cpu(&manifest, &format!("{PREFIX}.down_proj"))?)?;
    let context = CudaContext::new(0)?;
    if context.caps().sm != 89 {
        return Err(format!(
            "MLP prefill probe is frozen for SM89, got SM{}",
            context.caps().sm
        )
        .into());
    }
    let input_values = deterministic_input();
    let input = transfers::to_cuda(&Tensor::from_bf16(vec![TOKENS, HIDDEN], &input_values)?, 0)?;
    let candidate = CandidateWorkspace {
        gate_up: gpu_zeros(&[TOKENS, 2 * INTERMEDIATE])?,
        hidden: gpu_zeros(&[TOKENS, INTERMEDIATE])?,
        output: gpu_zeros(&[TOKENS, HIDDEN])?,
    };
    let serial = serial_workspace(&input_values)?;

    run_serial(&context, &serial, &gate_up, &down)?;
    run_candidate(&context, &input, &candidate, &gate_up, &down)?;
    context.synchronize()?;
    let correctness = compare_endpoints(&serial, &candidate)?;
    if correctness.iter().any(|(_, different)| *different != 0) {
        return Err(format!("M8 MLP differs from serial M1: {correctness:?}").into());
    }
    for _ in 0..3 {
        run_serial(&context, &serial, &gate_up, &down)?;
        run_candidate(&context, &input, &candidate, &gate_up, &down)?;
    }
    context.synchronize()?;
    let mut serial_samples = Vec::with_capacity(PAIRS);
    let mut candidate_samples = Vec::with_capacity(PAIRS);
    let mut records = Vec::with_capacity(2 * PAIRS);
    for pair in 0..PAIRS {
        let candidate_first = pair % 2 == 1;
        for order_index in 0..2 {
            let candidate_arm = (order_index == 0) == candidate_first;
            let start = Instant::now();
            for _ in 0..CALLS {
                if candidate_arm {
                    run_candidate(&context, &input, &candidate, &gate_up, &down)?;
                } else {
                    run_serial(&context, &serial, &gate_up, &down)?;
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
            "schema":"apxinf.qwen35.mlp_prefill_probe.v1",
            "model_dir":model_dir,"layer":0,
            "contract":{"tokens":TOKENS,"hidden":HIDDEN,"intermediate":INTERMEDIATE,
                "path":"gate/up M8 W4 -> eight existing SiLU*Mul views -> down M8 W4"},
            "kernel_build_id":KERNEL_BUILD_ID,
            "correctness":{"different_bf16_values":correctness,"pass":true},
            "timing":{"pairs":PAIRS,"calls_per_sample":CALLS,"records":records,
                "serial_raw_us":serial_samples,"candidate_raw_us":candidate_samples,
                "serial_median_us":median(&serial_samples),"candidate_median_us":median(&candidate_samples),
                "median_speedup":median(&speedups),"candidate_wins":wins,
                "candidate_tokens_per_second":TOKENS as f64*1.0e6/median(&candidate_samples)},
            "evidence_level":"mlp-prefill-layer","model_promoted":false,
        }))?
    );
    Ok(())
}

fn run_candidate(
    context: &CudaContext,
    input: &Tensor,
    workspace: &CandidateWorkspace,
    gate_up: &W4,
    down: &W4,
) -> apxinf_core::Result<()> {
    gemm::w4a16_m8_write(context, input, gate_up.view(), &workspace.gate_up)?;
    let gate_up_buffer =
        CudaBuffer::from_tensor(&workspace.gate_up).map_err(apxinf_core::Error::Cuda)?;
    let hidden_buffer =
        CudaBuffer::from_tensor(&workspace.hidden).map_err(apxinf_core::Error::Cuda)?;
    let gate_up_bytes = 2 * INTERMEDIATE * DType::BF16.size_in_bytes();
    let hidden_bytes = INTERMEDIATE * DType::BF16.size_in_bytes();
    for token in 0..TOKENS {
        activation::silu_mul_bf16_into(
            context,
            &gate_up_buffer
                .view(token * gate_up_bytes, gate_up_bytes)
                .map_err(apxinf_core::Error::Cuda)?,
            &hidden_buffer
                .view(token * hidden_bytes, hidden_bytes)
                .map_err(apxinf_core::Error::Cuda)?,
            INTERMEDIATE,
        )?;
    }
    gemm::w4a16_m8_write(context, &workspace.hidden, down.view(), &workspace.output)
}

fn run_serial(
    context: &CudaContext,
    workspace: &SerialWorkspace,
    gate_up: &W4,
    down: &W4,
) -> apxinf_core::Result<()> {
    for token in 0..TOKENS {
        gemm::w4a16_write(
            context,
            &workspace.input[token],
            gate_up.view(),
            &workspace.gate_up[token],
        )?;
        activation::silu_mul_bf16_into(
            context,
            &CudaBuffer::from_tensor(&workspace.gate_up[token])
                .map_err(apxinf_core::Error::Cuda)?,
            &CudaBuffer::from_tensor(&workspace.hidden[token]).map_err(apxinf_core::Error::Cuda)?,
            INTERMEDIATE,
        )?;
        gemm::w4a16_write(
            context,
            &workspace.hidden[token],
            down.view(),
            &workspace.output[token],
        )?;
    }
    Ok(())
}

fn compare_endpoints(
    serial: &SerialWorkspace,
    candidate: &CandidateWorkspace,
) -> Result<Vec<(String, usize)>, Box<dyn std::error::Error>> {
    let mut results = Vec::new();
    for (name, serial_values, candidate_tensor, width) in [
        (
            "gate_up",
            &serial.gate_up,
            &candidate.gate_up,
            2 * INTERMEDIATE,
        ),
        ("silu_mul", &serial.hidden, &candidate.hidden, INTERMEDIATE),
        ("output", &serial.output, &candidate.output, HIDDEN),
    ] {
        let candidate_cpu = transfers::to_cpu(candidate_tensor)?;
        let candidate_values = candidate_cpu.as_bf16()?;
        let mut different = 0usize;
        for token in 0..TOKENS {
            let serial_cpu = transfers::to_cpu(&serial_values[token])?;
            different += serial_cpu
                .as_bf16()?
                .iter()
                .zip(&candidate_values[token * width..(token + 1) * width])
                .filter(|(left, right)| left.to_bits() != right.to_bits())
                .count();
        }
        results.push((name.to_owned(), different));
    }
    Ok(results)
}

fn serial_workspace(values: &[half::bf16]) -> Result<SerialWorkspace, Box<dyn std::error::Error>> {
    let mut input = Vec::with_capacity(TOKENS);
    let mut gate_up = Vec::with_capacity(TOKENS);
    let mut hidden = Vec::with_capacity(TOKENS);
    let mut output = Vec::with_capacity(TOKENS);
    for token in 0..TOKENS {
        input.push(transfers::to_cuda(
            &Tensor::from_bf16(
                vec![1, HIDDEN],
                &values[token * HIDDEN..(token + 1) * HIDDEN],
            )?,
            0,
        )?);
        gate_up.push(gpu_zeros(&[1, 2 * INTERMEDIATE])?);
        hidden.push(gpu_zeros(&[1, INTERMEDIATE])?);
        output.push(gpu_zeros(&[1, HIDDEN])?);
    }
    Ok(SerialWorkspace {
        input,
        gate_up,
        hidden,
        output,
    })
}

struct CpuW4 {
    packed: Tensor,
    scales: Tensor,
    zero: Tensor,
    input: usize,
    output: usize,
}

fn load_w4_cpu(
    manifest: &CheckpointManifest,
    base: &str,
) -> Result<CpuW4, Box<dyn std::error::Error>> {
    let load = |suffix: &str| {
        safetensors::load_manifest_tensor(
            manifest
                .tensor(&format!("{base}.{suffix}"))
                .ok_or_else(|| format!("missing {base}.{suffix}"))?,
        )
    };
    let shape = safetensors::read_small_i64(
        manifest
            .tensor(&format!("{base}.weight_shape"))
            .ok_or("missing W4 shape")?,
    )?;
    Ok(CpuW4 {
        packed: load("weight_packed")?,
        scales: load("weight_scale")?,
        zero: load("weight_zero_point")?,
        input: shape[1] as usize,
        output: shape[0] as usize,
    })
}

fn to_gpu(weight: CpuW4) -> Result<W4, Box<dyn std::error::Error>> {
    Ok(W4 {
        packed: transfers::to_cuda(&weight.packed, 0)?,
        scales: transfers::to_cuda(&weight.scales, 0)?,
        zero: transfers::to_cuda(&weight.zero, 0)?,
        input: weight.input,
        output: weight.output,
    })
}

fn combine_to_gpu(first: CpuW4, second: CpuW4) -> Result<W4, Box<dyn std::error::Error>> {
    if first.input != second.input || first.output != second.output || first.output % 8 != 0 {
        return Err("gate/up W4 contract mismatch".into());
    }
    let mut packed = first.packed.as_i32()?.to_vec();
    packed.extend_from_slice(second.packed.as_i32()?);
    let mut scales = first.scales.as_bf16()?.to_vec();
    scales.extend_from_slice(second.scales.as_bf16()?);
    let mut zero = first.zero.as_i32()?.to_vec();
    zero.extend_from_slice(second.zero.as_i32()?);
    to_gpu(CpuW4 {
        packed: Tensor::from_i32(vec![2 * first.output, first.input / 8], &packed)?,
        scales: Tensor::from_bf16(vec![2 * first.output, first.input / 32], &scales)?,
        zero: Tensor::from_i32(vec![2 * first.output / 8, first.input / 32], &zero)?,
        input: first.input,
        output: 2 * first.output,
    })
}

fn deterministic_input() -> Vec<half::bf16> {
    (0..TOKENS * HIDDEN)
        .map(|index| {
            let token = index / HIDDEN;
            let column = index % HIDDEN;
            let phase = column as f32 * 0.005_859_375 + token as f32 * 0.117_187_5;
            half::bf16::from_f32((phase.sin() + 0.2 * phase.cos()) * 0.2)
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
    sorted[sorted.len() / 2]
}
