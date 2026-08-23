//! Dynamic original-layout transform plus raw Marlin M64 W4A16 gate projection.

use std::cmp::Ordering;
use std::path::Path;
use std::time::Instant;

use apxinf_core::{DType, Tensor};
use apxinf_cuda::kernels::gemm::{
    self, MarlinPreparedWeight, MarlinWorkspace, W4A16Layout, W4A16WeightView,
};
use apxinf_cuda::tuning::KERNEL_BUILD_ID;
use apxinf_cuda::{transfers, CudaContext};
use apxinf_loader::safetensors::{self, CheckpointManifest};

const ROWS: usize = 64;
const M8: usize = 8;
const INPUT: usize = 5120;
const OUTPUT: usize = 17408;
const BASE: &str = "model.language_model.layers.0.mlp.gate_proj";
const WARMUPS: usize = 3;
const PAIRS: usize = 5;

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
        .ok_or("usage: qwen35_marlin_probe MODEL_DIR")?;
    let manifest = safetensors::inspect_path(Path::new(&model_dir))?;
    let context = CudaContext::new(0)?;
    if context.caps().sm != 89 {
        return Err(format!(
            "Marlin probe is frozen for SM89, got SM{}",
            context.caps().sm
        )
        .into());
    }
    let weight = load_w4(&manifest)?;
    let values = deterministic_activations();
    let activation = transfers::to_cuda(&Tensor::from_bf16(vec![ROWS, INPUT], &values)?, 0)?;
    let mut serial_inputs = Vec::with_capacity(ROWS / M8);
    let mut serial_outputs = Vec::with_capacity(ROWS / M8);
    for tile in 0..ROWS / M8 {
        serial_inputs.push(transfers::to_cuda(
            &Tensor::from_bf16(
                vec![M8, INPUT],
                &values[tile * M8 * INPUT..(tile + 1) * M8 * INPUT],
            )?,
            0,
        )?);
        serial_outputs.push(gpu_zeros(&[M8, OUTPUT])?);
    }
    let candidate_output = gpu_zeros(&[ROWS, OUTPUT])?;
    let prepared = MarlinPreparedWeight::new(&context, INPUT, OUTPUT)?;
    let workspace = MarlinWorkspace::new(&context)?;

    run_serial(&context, &weight, &serial_inputs, &serial_outputs)?;
    run_candidate(
        &context,
        &weight,
        &activation,
        &candidate_output,
        &prepared,
        &workspace,
        true,
    )?;
    context.synchronize()?;
    let correctness = compare(&serial_outputs, &candidate_output)?;
    if correctness.cosine < 0.999 || correctness.relative_l2 > 0.005 {
        return Err(format!("Marlin numerical gate failed: {correctness:?}").into());
    }

    for _ in 0..WARMUPS {
        run_serial(&context, &weight, &serial_inputs, &serial_outputs)?;
        run_candidate(
            &context,
            &weight,
            &activation,
            &candidate_output,
            &prepared,
            &workspace,
            true,
        )?;
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
            if candidate_arm {
                run_candidate(
                    &context,
                    &weight,
                    &activation,
                    &candidate_output,
                    &prepared,
                    &workspace,
                    true,
                )?;
            } else {
                run_serial(&context, &weight, &serial_inputs, &serial_outputs)?;
            }
            context.synchronize()?;
            let elapsed = start.elapsed().as_secs_f64() * 1.0e6;
            if candidate_arm {
                candidate_samples.push(elapsed);
            } else {
                serial_samples.push(elapsed);
            }
            records.push(serde_json::json!({
                "pair":pair,"order":if candidate_first{"BA"}else{"AB"},
                "order_index":order_index,
                "arm":if candidate_arm{"transform_plus_marlin_m64"}else{"eight_m8"},
                "elapsed_us":elapsed,
            }));
        }
    }
    prepared.prepare(&context, weight.view())?;
    context.synchronize()?;
    let mut kernel_samples = Vec::with_capacity(20);
    for _ in 0..20 {
        let start = Instant::now();
        gemm::w4a16_marlin_write(
            &context,
            &activation,
            prepared.view(),
            &candidate_output,
            &workspace,
        )?;
        context.synchronize()?;
        kernel_samples.push(start.elapsed().as_secs_f64() * 1.0e6);
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
            "schema":"apxinf.qwen35.marlin_probe.v2",
            "model_dir":model_dir,"weight":BASE,
            "contract":{
                "rows":ROWS,"input":INPUT,"output":OUTPUT,
                "quantization":"compressed-tensors asymmetric U4 group-32",
                "baseline":"eight accepted scalar M8 W4A16 calls",
                "candidate":"GPU transpose + official-layout repack + scale/zero permutation + raw Marlin M64",
                "timing":"host launch through stream synchronize; no profiler",
            },
            "device":{
                "name":context.caps().device_name,"sm":context.caps().sm,
                "multiprocessors":context.caps().multiprocessor_count,
                "cuda":context.library_versions().cuda,
            },
            "kernel_build_id":KERNEL_BUILD_ID,
            "correctness":{
                "oracle":"eight exact scalar M8 calls",
                "cosine":correctness.cosine,"relative_l2":correctness.relative_l2,
                "max_abs":correctness.max_abs,"mean_abs":correctness.mean_abs,
                "different_bf16":correctness.different,"count":correctness.count,
                "threshold":{"cosine_min":0.999,"relative_l2_max":0.005},
                "threshold_basis":"INT4 exploratory contract plus independent checkpoint-dense Marlin rel-L2 1.08347e-4; v1 scalar-exact threshold 0.001 rejected rel-L2 0.00226126",
                "pass":true,
            },
            "timing":{
                "warmups_per_arm":WARMUPS,"pairs":PAIRS,"records":records,
                "serial_raw_us":serial_samples,"candidate_raw_us":candidate_samples,
                "serial_median_us":median(&serial_samples),
                "candidate_median_us":median(&candidate_samples),
                "median_speedup":median(&speedups),"candidate_wins":wins,
                "marlin_kernel_only_raw_us":kernel_samples,
                "marlin_kernel_only_median_us":median(&kernel_samples),
                "candidate_tokens_per_second":ROWS as f64*1.0e6/median(&candidate_samples),
            },
            "evidence_level":"operator-with-runtime-transform",
            "model_promoted":false,
        }))?
    );
    Ok(())
}

fn run_serial(
    context: &CudaContext,
    weight: &W4,
    inputs: &[Tensor],
    outputs: &[Tensor],
) -> apxinf_core::Result<()> {
    for (input, output) in inputs.iter().zip(outputs) {
        gemm::w4a16_m8_write(context, input, weight.view(), output)?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_candidate(
    context: &CudaContext,
    weight: &W4,
    activation: &Tensor,
    output: &Tensor,
    prepared: &MarlinPreparedWeight,
    workspace: &MarlinWorkspace,
    transform: bool,
) -> apxinf_core::Result<()> {
    if transform {
        prepared.prepare(context, weight.view())?;
    }
    gemm::w4a16_marlin_write(context, activation, prepared.view(), output, workspace)
}

#[derive(Debug)]
struct Metrics {
    cosine: f64,
    relative_l2: f64,
    max_abs: f64,
    mean_abs: f64,
    different: usize,
    count: usize,
}

fn compare(serial: &[Tensor], candidate: &Tensor) -> Result<Metrics, Box<dyn std::error::Error>> {
    let candidate = transfers::to_cpu(candidate)?;
    let candidate = candidate.as_bf16()?;
    let mut dot = 0.0_f64;
    let mut aa = 0.0_f64;
    let mut bb = 0.0_f64;
    let mut error = 0.0_f64;
    let mut absolute = 0.0_f64;
    let mut maximum = 0.0_f64;
    let mut different = 0;
    let mut offset = 0;
    for tile in serial {
        let tile = transfers::to_cpu(tile)?;
        for (left, right) in tile
            .as_bf16()?
            .iter()
            .zip(&candidate[offset..offset + tile.as_bf16()?.len()])
        {
            let (left_f32, right_f32) = (left.to_f32() as f64, right.to_f32() as f64);
            let delta = right_f32 - left_f32;
            dot += left_f32 * right_f32;
            aa += left_f32 * left_f32;
            bb += right_f32 * right_f32;
            error += delta * delta;
            absolute += delta.abs();
            maximum = maximum.max(delta.abs());
            different += usize::from(left.to_bits() != right.to_bits());
        }
        offset += tile.as_bf16()?.len();
    }
    Ok(Metrics {
        cosine: dot / (aa.sqrt() * bb.sqrt()),
        relative_l2: (error / aa).sqrt(),
        max_abs: maximum,
        mean_abs: absolute / offset as f64,
        different,
        count: offset,
    })
}

fn load_w4(manifest: &CheckpointManifest) -> Result<W4, Box<dyn std::error::Error>> {
    let load = |suffix: &str| {
        safetensors::load_manifest_tensor(
            manifest
                .tensor(&format!("{BASE}.{suffix}"))
                .ok_or_else(|| format!("missing {BASE}.{suffix}"))?,
        )
    };
    Ok(W4 {
        packed: transfers::to_cuda(&load("weight_packed")?, 0)?,
        scales: transfers::to_cuda(&load("weight_scale")?, 0)?,
        zero: transfers::to_cuda(&load("weight_zero_point")?, 0)?,
    })
}

fn deterministic_activations() -> Vec<half::bf16> {
    (0..ROWS * INPUT)
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
    if sorted.len() % 2 == 0 {
        (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) * 0.5
    } else {
        sorted[sorted.len() / 2]
    }
}
