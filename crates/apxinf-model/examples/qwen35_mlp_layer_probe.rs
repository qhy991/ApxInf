//! Real Qwen3.5 layer-0 MLP decode trajectory with packed gate/up W4.

use std::cmp::Ordering;
use std::path::Path;
use std::time::Instant;

use apxinf_core::{DType, Tensor};
use apxinf_cuda::kernels::gemm::{W4A16Layout, W4A16WeightView};
use apxinf_cuda::kernels::{activation, gemm};
use apxinf_cuda::tuning::KERNEL_BUILD_ID;
use apxinf_cuda::{transfers, CudaBuffer, CudaContext};
use apxinf_loader::safetensors::{self, CheckpointManifest};

const HIDDEN: usize = 5120;
const INTERMEDIATE: usize = 17408;
const PREFIX: &str = "model.language_model.layers.0.mlp";
const WARMUPS: usize = 5;
const BLOCKS: usize = 20;
const CALLS: usize = 10;

struct W4 {
    packed_cpu: Tensor,
    scales_cpu: Tensor,
    zero_cpu: Tensor,
    packed_gpu: Tensor,
    scales_gpu: Tensor,
    zero_gpu: Tensor,
    input: usize,
    output: usize,
}

impl W4 {
    fn view(&self) -> W4A16WeightView<'_> {
        W4A16WeightView {
            packed_i32: &self.packed_gpu,
            scales_bf16: &self.scales_gpu,
            zero_points_i32: &self.zero_gpu,
            input_dim: self.input,
            output_dim: self.output,
            group_size: 32,
            layout: W4A16Layout::CompressedTensorsPackQuantized,
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_dir = std::env::args()
        .nth(1)
        .ok_or("usage: qwen35_mlp_layer_probe MODEL_DIR")?;
    let manifest = safetensors::inspect_path(Path::new(&model_dir))?;
    let gate = load_w4(&manifest, &format!("{PREFIX}.gate_proj"))?;
    let up = load_w4(&manifest, &format!("{PREFIX}.up_proj"))?;
    let down = load_w4(&manifest, &format!("{PREFIX}.down_proj"))?;
    let gate_up = combine_w4(&gate, &up)?;
    let context = CudaContext::new(0)?;
    if context.caps().sm != 89 {
        return Err(format!("probe is frozen for SM89, got SM{}", context.caps().sm).into());
    }
    let hidden_values = (0..HIDDEN)
        .map(|index| {
            let phase = index as f32 * 0.005_859_375 + (index % 31) as f32 * 0.001_953_125;
            half::bf16::from_f32((phase.sin() + 0.25 * phase.cos()) * 0.25)
        })
        .collect::<Vec<_>>();
    let hidden_cpu = Tensor::from_bf16(vec![1, HIDDEN], &hidden_values)?;
    let hidden = transfers::to_cuda(&hidden_cpu, 0)?;
    let gate_up_output = gpu_zeros(&[1, 2 * INTERMEDIATE])?;
    let mlp_hidden = gpu_zeros(&[1, INTERMEDIATE])?;
    let output = gpu_zeros(&[1, HIDDEN])?;

    run(
        &context,
        &hidden,
        &gate_up,
        &down,
        &gate_up_output,
        &mlp_hidden,
        &output,
        false,
    )?;
    context.synchronize()?;
    let expected_gate_up = cpu_w4(hidden_cpu.as_bf16()?, &gate_up)?;
    let expected_hidden = cpu_swiglu(&expected_gate_up);
    let expected_output = cpu_w4(&expected_hidden, &down)?;
    let actual_gate_up = transfers::to_cpu(&gate_up_output)?.to_f32_vec()?;
    let actual_hidden = transfers::to_cpu(&mlp_hidden)?.to_f32_vec()?;
    let actual_output = transfers::to_cpu(&output)?.to_f32_vec()?;
    let gate_up_metrics = metrics(&actual_gate_up, &bf16_f32(&expected_gate_up))?;
    let hidden_metrics = metrics(&actual_hidden, &bf16_f32(&expected_hidden))?;
    let output_metrics = metrics(&actual_output, &bf16_f32(&expected_output))?;
    for (name, value) in [
        ("gate_up", gate_up_metrics),
        ("swiglu", hidden_metrics),
        ("output", output_metrics),
    ] {
        if value.0 < 0.999 || value.1 > 0.02 {
            return Err(format!("MLP endpoint {name} failed: {value:?}").into());
        }
    }

    if std::env::var("APXINF_PROFILE").as_deref() == Ok("1") {
        apxinf_cuda::profiler::start()?;
        run(
            &context,
            &hidden,
            &gate_up,
            &down,
            &gate_up_output,
            &mlp_hidden,
            &output,
            true,
        )?;
        context.synchronize()?;
        apxinf_cuda::profiler::stop()?;
    }
    for _ in 0..WARMUPS {
        run(
            &context,
            &hidden,
            &gate_up,
            &down,
            &gate_up_output,
            &mlp_hidden,
            &output,
            false,
        )?;
    }
    context.synchronize()?;
    let mut samples = Vec::with_capacity(BLOCKS);
    for _ in 0..BLOCKS {
        let start = Instant::now();
        for _ in 0..CALLS {
            run(
                &context,
                &hidden,
                &gate_up,
                &down,
                &gate_up_output,
                &mlp_hidden,
                &output,
                false,
            )?;
        }
        context.synchronize()?;
        samples.push(start.elapsed().as_secs_f64() * 1.0e6 / CALLS as f64);
    }
    let (median, mean, deviation) = summarize(&samples);
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "schema": "apxinf.qwen35.mlp_layer_probe.v1",
            "model_dir": model_dir,
            "layer": 0,
            "contract": {"hidden":HIDDEN,"intermediate":INTERMEDIATE,"gate_up_packed":true},
            "kernel_build_id": KERNEL_BUILD_ID,
            "correctness": {
                "gate_up": metric_json(gate_up_metrics),
                "swiglu": metric_json(hidden_metrics),
                "output": metric_json(output_metrics),
                "pass": true,
            },
            "timing": {
                "boundary":"combined gate/up W4 + BF16 SwiGLU + down W4 to stream synchronize; no profiler",
                "warmups":WARMUPS,"blocks":BLOCKS,"calls_per_block":CALLS,
                "raw_us_per_layer":samples,"median_us":median,"mean_us":mean,
                "standard_deviation_us":deviation,"cv":deviation/mean,
            },
            "evidence_level":"layer-module","model_promoted":false,
        }))?
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run(
    context: &CudaContext,
    hidden: &Tensor,
    gate_up: &W4,
    down: &W4,
    gate_up_output: &Tensor,
    mlp_hidden: &Tensor,
    output: &Tensor,
    profile: bool,
) -> apxinf_core::Result<()> {
    {
        let _range = profile.then(|| apxinf_cuda::nvtx::range("qwen35.mlp.gate_up_w4"));
        gemm::w4a16_write(context, hidden, gate_up.view(), gate_up_output)?;
    }
    {
        let _range = profile.then(|| apxinf_cuda::nvtx::range("qwen35.mlp.swiglu"));
        let input = CudaBuffer::from_tensor(gate_up_output).map_err(apxinf_core::Error::Cuda)?;
        let output_buffer =
            CudaBuffer::from_tensor(mlp_hidden).map_err(apxinf_core::Error::Cuda)?;
        activation::silu_mul_bf16_into(context, &input, &output_buffer, INTERMEDIATE)?;
    }
    {
        let _range = profile.then(|| apxinf_cuda::nvtx::range("qwen35.mlp.down_w4"));
        gemm::w4a16_write(context, mlp_hidden, down.view(), output)
    }
}

fn load_w4(manifest: &CheckpointManifest, base: &str) -> Result<W4, Box<dyn std::error::Error>> {
    let load = |suffix: &str| {
        safetensors::load_manifest_tensor(
            manifest
                .tensor(&format!("{base}.{suffix}"))
                .ok_or_else(|| format!("missing {base}.{suffix}"))?,
        )
    };
    let packed_cpu = load("weight_packed")?;
    let scales_cpu = load("weight_scale")?;
    let zero_cpu = load("weight_zero_point")?;
    let shape = safetensors::read_small_i64(
        manifest
            .tensor(&format!("{base}.weight_shape"))
            .ok_or("missing W4 shape")?,
    )?;
    make_w4(
        packed_cpu,
        scales_cpu,
        zero_cpu,
        shape[1] as usize,
        shape[0] as usize,
    )
}

fn make_w4(
    packed_cpu: Tensor,
    scales_cpu: Tensor,
    zero_cpu: Tensor,
    input: usize,
    output: usize,
) -> Result<W4, Box<dyn std::error::Error>> {
    Ok(W4 {
        packed_gpu: transfers::to_cuda(&packed_cpu, 0)?,
        scales_gpu: transfers::to_cuda(&scales_cpu, 0)?,
        zero_gpu: transfers::to_cuda(&zero_cpu, 0)?,
        packed_cpu,
        scales_cpu,
        zero_cpu,
        input,
        output,
    })
}

fn combine_w4(first: &W4, second: &W4) -> Result<W4, Box<dyn std::error::Error>> {
    if first.input != second.input || first.output != second.output || first.output % 8 != 0 {
        return Err("gate/up W4 contract mismatch".into());
    }
    let mut packed = first.packed_cpu.as_i32()?.to_vec();
    packed.extend_from_slice(second.packed_cpu.as_i32()?);
    let mut scales = first.scales_cpu.as_bf16()?.to_vec();
    scales.extend_from_slice(second.scales_cpu.as_bf16()?);
    let mut zero = first.zero_cpu.as_i32()?.to_vec();
    zero.extend_from_slice(second.zero_cpu.as_i32()?);
    make_w4(
        Tensor::from_i32(vec![2 * first.output, first.input / 8], &packed)?,
        Tensor::from_bf16(vec![2 * first.output, first.input / 32], &scales)?,
        Tensor::from_i32(vec![2 * first.output / 8, first.input / 32], &zero)?,
        first.input,
        2 * first.output,
    )
}

fn cpu_w4(
    input: &[half::bf16],
    weight: &W4,
) -> Result<Vec<half::bf16>, Box<dyn std::error::Error>> {
    let packed = weight.packed_cpu.as_i32()?;
    let scales = weight.scales_cpu.as_bf16()?;
    let zero = weight.zero_cpu.as_i32()?;
    let packed_cols = weight.input / 8;
    let groups = weight.input / 32;
    let mut output = vec![half::bf16::ZERO; weight.output];
    for row in 0..weight.output {
        let mut sum = 0.0_f32;
        for packed_col in 0..packed_cols {
            let word = packed[row * packed_cols + packed_col] as u32;
            let group = packed_col / 4;
            let scale = scales[row * groups + group].to_f32();
            let zero_word = zero[(row / 8) * groups + group] as u32;
            let zp = (((zero_word >> ((row & 7) * 4)) & 15) as i32) - 8;
            for index in 0..8 {
                let q = (((word >> (index * 4)) & 15) as i32) - 8;
                sum = input[packed_col * 8 + index]
                    .to_f32()
                    .mul_add((q - zp) as f32 * scale, sum);
            }
        }
        output[row] = half::bf16::from_f32(sum);
    }
    Ok(output)
}

fn cpu_swiglu(gate_up: &[half::bf16]) -> Vec<half::bf16> {
    (0..INTERMEDIATE)
        .map(|index| {
            let gate = gate_up[index].to_f32();
            let up = gate_up[index + INTERMEDIATE].to_f32();
            half::bf16::from_f32(gate / (1.0 + (-gate).exp()) * up)
        })
        .collect()
}

fn gpu_zeros(shape: &[usize]) -> Result<Tensor, Box<dyn std::error::Error>> {
    Ok(transfers::to_cuda(
        &Tensor::zeros(shape.to_vec(), DType::BF16),
        0,
    )?)
}

fn bf16_f32(values: &[half::bf16]) -> Vec<f32> {
    values.iter().map(|value| value.to_f32()).collect()
}

fn metrics(actual: &[f32], expected: &[f32]) -> Result<(f64, f64, f64), String> {
    if actual.len() != expected.len() || actual.is_empty() {
        return Err("length mismatch".into());
    }
    let (mut dot, mut aa, mut ee, mut error, mut max) = (0.0, 0.0, 0.0, 0.0, 0.0_f64);
    for (&a, &e) in actual.iter().zip(expected) {
        if !a.is_finite() || !e.is_finite() {
            return Err("non-finite".into());
        }
        let (a, e) = (f64::from(a), f64::from(e));
        dot += a * e;
        aa += a * a;
        ee += e * e;
        error += (a - e).powi(2);
        max = max.max((a - e).abs());
    }
    Ok((dot / (aa.sqrt() * ee.sqrt()), (error / ee).sqrt(), max))
}

fn metric_json(value: (f64, f64, f64)) -> serde_json::Value {
    serde_json::json!({"cosine":value.0,"relative_l2":value.1,"max_abs":value.2,"pass":value.0>=0.999&&value.1<=0.02})
}

fn summarize(samples: &[f64]) -> (f64, f64, f64) {
    let mut sorted = samples.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    let median = (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) * 0.5;
    let mean = samples.iter().sum::<f64>() / samples.len() as f64;
    let deviation =
        (samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / samples.len() as f64).sqrt();
    (median, mean, deviation)
}
