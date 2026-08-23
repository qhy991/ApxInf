//! Real layer-0 Qwen3.5 GDN decode trajectory and no-profiler timing.

use std::cmp::Ordering;
use std::path::Path;
use std::time::Instant;

use apxinf_core::{DType, Tensor};
use apxinf_cuda::kernels::gdn::{
    qwen35_conv4_prepare_write, qwen35_conv4_silu_write, qwen35_gated_rmsnorm_write,
    qwen35_prepare_write, qwen35_recurrent_write, QWEN35_GDN_CONV_DIM as CONV_DIM,
    QWEN35_GDN_CONV_KERNEL as CONV_KERNEL, QWEN35_GDN_HEADS as HEADS, QWEN35_GDN_KEY_DIM as DIM,
};
use apxinf_cuda::kernels::gemm::{
    self, W4A16Layout, W4A16WeightView, W8A16WeightView, W8A8Layout, W8A8ScaleMode, W8A8WeightView,
};
use apxinf_cuda::kernels::GraphWorkspace;
use apxinf_cuda::tuning::KERNEL_BUILD_ID;
use apxinf_cuda::{transfers, CublasTranspose, CudaBuffer, CudaContext};
use apxinf_loader::safetensors::{self, CheckpointManifest};

const HIDDEN: usize = 5120;
const KEY_HEADS: usize = 16;
const VALUE_WIDTH: usize = HEADS * DIM;
const KEY_WIDTH: usize = KEY_HEADS * DIM;
const RMS_EPSILON: f32 = 1.0e-6;
const WARMUPS: usize = 5;
const BLOCKS: usize = 20;
const CALLS_PER_BLOCK: usize = 10;
const PREFIX: &str = "model.language_model.layers.0.linear_attn";

struct W4Weight {
    packed_cpu: Tensor,
    scales_cpu: Tensor,
    zero_cpu: Tensor,
    packed_gpu: Tensor,
    scales_gpu: Tensor,
    zero_gpu: Tensor,
    input_dim: usize,
    output_dim: usize,
}

impl W4Weight {
    fn view(&self) -> W4A16WeightView<'_> {
        W4A16WeightView {
            packed_i32: &self.packed_gpu,
            scales_bf16: &self.scales_gpu,
            zero_points_i32: &self.zero_gpu,
            input_dim: self.input_dim,
            output_dim: self.output_dim,
            group_size: 32,
            layout: W4A16Layout::CompressedTensorsPackQuantized,
        }
    }
}

struct W8Weight {
    values_i8: CudaBuffer,
    scales_f32: Tensor,
    input_dim: usize,
    output_dim: usize,
}

impl W8Weight {
    fn view_w8a16(&self) -> W8A16WeightView<'_> {
        W8A16WeightView {
            values_i8: &self.values_i8,
            scales_f32: &self.scales_f32,
            input_dim: self.input_dim,
            output_dim: self.output_dim,
        }
    }

    fn view_w8a8(&self) -> W8A8WeightView<'_> {
        W8A8WeightView {
            values_i8: &self.values_i8,
            scales_f32: &self.scales_f32,
            input_dim: self.input_dim,
            output_dim: self.output_dim,
            scale_mode: W8A8ScaleMode::DynamicRowPerOutputChannel,
            layout: W8A8Layout::OutputMajor,
        }
    }
}

struct Layer {
    qkv: W4Weight,
    z: W4Weight,
    a_cpu: Tensor,
    b_cpu: Tensor,
    conv_cpu: Tensor,
    a_log_cpu: Tensor,
    dt_bias_cpu: Tensor,
    norm_cpu: Tensor,
    out_cpu: Tensor,
    out_w4: W4Weight,
    out_w8: W8Weight,
    workspace: GraphWorkspace,
    a: Tensor,
    b: Tensor,
    ab: Tensor,
    conv: Tensor,
    a_log: Tensor,
    dt_bias: Tensor,
    norm: Tensor,
    out: Tensor,
    qkv_output: Tensor,
    z_output: Tensor,
    a_output: Tensor,
    b_output: Tensor,
    ab_output: Tensor,
    conv_state: Tensor,
    conv_output: Tensor,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    g: Tensor,
    beta: Tensor,
    recurrent_state: Tensor,
    core_output: Tensor,
    norm_output: Tensor,
    layer_output: Tensor,
}

#[derive(Clone, Copy)]
struct Metrics {
    cosine: f64,
    relative_l2: f64,
    max_abs: f64,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_dir = std::env::args()
        .nth(1)
        .ok_or("usage: qwen35_gdn_layer_probe MODEL_DIR")?;
    let out_projection = std::env::var("APXINF_GDN_OUT_PROJ").unwrap_or_else(|_| "bf16".into());
    let fused_prepare = std::env::var("APXINF_GDN_FUSE_PREP").as_deref() == Ok("1");
    if out_projection != "bf16"
        && out_projection != "w4"
        && out_projection != "w8"
        && out_projection != "w8a8"
    {
        return Err(format!(
            "APXINF_GDN_OUT_PROJ must be bf16, w4, w8, or w8a8, got `{out_projection}`"
        )
        .into());
    }
    let manifest = safetensors::inspect_path(Path::new(&model_dir))?;
    let context = CudaContext::new(0)?;
    if context.caps().sm != 89 {
        return Err(format!("probe is frozen for SM89, got SM{}", context.caps().sm).into());
    }
    let hidden_values = (0..HIDDEN)
        .map(|index| {
            let phase = index as f32 * 0.007_812_5 + (index % 23) as f32 * 0.003_906_25;
            half::bf16::from_f32((phase.sin() + 0.125 * phase.cos()) * 0.25)
        })
        .collect::<Vec<_>>();
    let hidden_cpu = Tensor::from_bf16(vec![1, HIDDEN], &hidden_values)?;
    let hidden = transfers::to_cuda(&hidden_cpu, 0)?;
    let mut layer = load_layer(&manifest)?;

    run_layer(
        &context,
        &hidden,
        &layer,
        false,
        &out_projection,
        fused_prepare,
    )?;
    context.synchronize()?;
    let expected = cpu_layer(&hidden_cpu, &layer)?;
    let actual = download_endpoints(&layer, fused_prepare)?;
    let endpoint_metrics = compare_endpoints(&actual, &expected)?;
    let final_metrics = endpoint_metrics
        .iter()
        .find(|(name, _)| name == "layer_output")
        .map(|(_, metrics)| *metrics)
        .ok_or("missing layer_output metric")?;
    if final_metrics.cosine < 0.999 || final_metrics.relative_l2 > 0.02 {
        return Err(format!(
            "complete GDN layer gate failed: {}",
            metrics_text(final_metrics)
        )
        .into());
    }

    if std::env::var("APXINF_PROFILE").as_deref() == Ok("1") {
        reset_states(&mut layer)?;
        apxinf_cuda::profiler::start()?;
        {
            let _range = apxinf_cuda::nvtx::range("qwen35.gdn_layer.complete");
            run_layer(
                &context,
                &hidden,
                &layer,
                true,
                &out_projection,
                fused_prepare,
            )?;
            context.synchronize()?;
        }
        apxinf_cuda::profiler::stop()?;
    }

    reset_states(&mut layer)?;
    for _ in 0..WARMUPS {
        run_layer(
            &context,
            &hidden,
            &layer,
            false,
            &out_projection,
            fused_prepare,
        )?;
    }
    context.synchronize()?;
    let mut samples_us = Vec::with_capacity(BLOCKS);
    for _ in 0..BLOCKS {
        let start = Instant::now();
        for _ in 0..CALLS_PER_BLOCK {
            run_layer(
                &context,
                &hidden,
                &layer,
                false,
                &out_projection,
                fused_prepare,
            )?;
        }
        context.synchronize()?;
        samples_us.push(start.elapsed().as_secs_f64() * 1.0e6 / CALLS_PER_BLOCK as f64);
    }
    let timing = summarize(&samples_us);

    let endpoints = endpoint_metrics
        .iter()
        .map(|(name, metrics)| {
            (
                name.clone(),
                serde_json::json!({
                    "cosine": metrics.cosine,
                    "relative_l2": metrics.relative_l2,
                    "max_abs": metrics.max_abs,
                    "pass": metrics.cosine >= 0.999 && metrics.relative_l2 <= 0.02,
                }),
            )
        })
        .collect::<serde_json::Map<_, _>>();
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "schema": "apxinf.qwen35.gdn_layer_probe.v1",
            "model_dir": model_dir,
            "layer": 0,
            "out_projection": out_projection,
            "fused_prepare": fused_prepare,
            "contract": {
                "hidden": HIDDEN,
                "qkv_width": CONV_DIM,
                "key_heads": KEY_HEADS,
                "value_heads": HEADS,
                "head_dim": DIM,
                "conv_kernel": CONV_KERNEL,
                "rms_epsilon": RMS_EPSILON,
                "decode_tokens": 1,
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
                "oracle": "CPU compressed-tensors W4 + BF16 seam model of the complete layer-0 GDN path",
                "endpoints": endpoints,
                "finite": true,
                "pass": true,
            },
            "timing": {
                "boundary": "all projection, conv/state, prepare, recurrent/state, gated norm, out-proj launches to stream synchronize; no profiler",
                "warmups": WARMUPS,
                "blocks": BLOCKS,
                "calls_per_block": CALLS_PER_BLOCK,
                "raw_us_per_layer": samples_us,
                "median_us": timing.0,
                "mean_us": timing.1,
                "standard_deviation_us": timing.2,
                "cv": timing.2 / timing.1,
            },
            "evidence_level": "layer-module",
            "model_promoted": false,
        }))?
    );
    Ok(())
}

fn load_layer(manifest: &CheckpointManifest) -> Result<Layer, Box<dyn std::error::Error>> {
    let qkv = load_w4(manifest, &format!("{PREFIX}.in_proj_qkv"))?;
    let z = load_w4(manifest, &format!("{PREFIX}.in_proj_z"))?;
    let a_cpu = load_tensor(manifest, &format!("{PREFIX}.in_proj_a.weight"))?;
    let b_cpu = load_tensor(manifest, &format!("{PREFIX}.in_proj_b.weight"))?;
    let conv_cpu = load_tensor(manifest, &format!("{PREFIX}.conv1d.weight"))?;
    let a_log_cpu = load_tensor(manifest, &format!("{PREFIX}.A_log"))?;
    let dt_bias_cpu = load_tensor(manifest, &format!("{PREFIX}.dt_bias"))?;
    let norm_cpu = load_tensor(manifest, &format!("{PREFIX}.norm.weight"))?;
    let out_cpu = load_tensor(manifest, &format!("{PREFIX}.out_proj.weight"))?;
    let out_w4 = quantize_bf16_w4(&out_cpu)?;
    let out_w8 = quantize_bf16_w8(&out_cpu)?;
    let ab_cpu = concat_bf16_rows(&a_cpu, &b_cpu)?;
    Ok(Layer {
        qkv,
        z,
        a: transfers::to_cuda(&a_cpu, 0)?,
        b: transfers::to_cuda(&b_cpu, 0)?,
        ab: transfers::to_cuda(&ab_cpu, 0)?,
        conv: transfers::to_cuda(&conv_cpu, 0)?,
        a_log: transfers::to_cuda(&a_log_cpu, 0)?,
        dt_bias: transfers::to_cuda(&dt_bias_cpu, 0)?,
        norm: transfers::to_cuda(&norm_cpu, 0)?,
        out: transfers::to_cuda(&out_cpu, 0)?,
        out_w4,
        out_w8,
        workspace: GraphWorkspace::new(64 * 1024, 0)?,
        a_cpu,
        b_cpu,
        conv_cpu,
        a_log_cpu,
        dt_bias_cpu,
        norm_cpu,
        out_cpu,
        qkv_output: gpu_zeros(&[1, CONV_DIM], DType::BF16)?,
        z_output: gpu_zeros(&[1, VALUE_WIDTH], DType::BF16)?,
        a_output: gpu_zeros(&[1, HEADS], DType::BF16)?,
        b_output: gpu_zeros(&[1, HEADS], DType::BF16)?,
        ab_output: gpu_zeros(&[1, 2 * HEADS], DType::BF16)?,
        conv_state: gpu_zeros(&[CONV_DIM, CONV_KERNEL], DType::BF16)?,
        conv_output: gpu_zeros(&[CONV_DIM], DType::BF16)?,
        query: gpu_zeros(&[HEADS, DIM], DType::BF16)?,
        key: gpu_zeros(&[HEADS, DIM], DType::BF16)?,
        value: gpu_zeros(&[HEADS, DIM], DType::BF16)?,
        g: gpu_zeros(&[HEADS], DType::F32)?,
        beta: gpu_zeros(&[HEADS], DType::F32)?,
        recurrent_state: gpu_zeros(&[HEADS, DIM, DIM], DType::F32)?,
        core_output: gpu_zeros(&[HEADS, DIM], DType::BF16)?,
        norm_output: gpu_zeros(&[HEADS, DIM], DType::BF16)?,
        layer_output: gpu_zeros(&[1, HIDDEN], DType::BF16)?,
    })
}

fn load_w4(
    manifest: &CheckpointManifest,
    base: &str,
) -> Result<W4Weight, Box<dyn std::error::Error>> {
    let packed_cpu = load_tensor(manifest, &format!("{base}.weight_packed"))?;
    let scales_cpu = load_tensor(manifest, &format!("{base}.weight_scale"))?;
    let zero_cpu = load_tensor(manifest, &format!("{base}.weight_zero_point"))?;
    let shape = safetensors::read_small_i64(
        manifest
            .tensor(&format!("{base}.weight_shape"))
            .ok_or("missing W4 shape")?,
    )?;
    let output_dim = usize::try_from(shape[0])?;
    let input_dim = usize::try_from(shape[1])?;
    Ok(W4Weight {
        packed_gpu: transfers::to_cuda(&packed_cpu, 0)?,
        scales_gpu: transfers::to_cuda(&scales_cpu, 0)?,
        zero_gpu: transfers::to_cuda(&zero_cpu, 0)?,
        packed_cpu,
        scales_cpu,
        zero_cpu,
        input_dim,
        output_dim,
    })
}

fn load_tensor(
    manifest: &CheckpointManifest,
    name: &str,
) -> Result<Tensor, Box<dyn std::error::Error>> {
    Ok(safetensors::load_manifest_tensor(
        manifest
            .tensor(name)
            .ok_or_else(|| format!("missing `{name}`"))?,
    )?)
}

fn gpu_zeros(shape: &[usize], dtype: DType) -> Result<Tensor, Box<dyn std::error::Error>> {
    Ok(transfers::to_cuda(
        &Tensor::zeros(shape.to_vec(), dtype),
        0,
    )?)
}

fn run_layer(
    context: &CudaContext,
    hidden: &Tensor,
    layer: &Layer,
    profile: bool,
    out_projection: &str,
    fused_prepare: bool,
) -> apxinf_core::Result<()> {
    apxinf_cuda::kernels::with_workspace(&layer.workspace, || {
        run_layer_inner(
            context,
            hidden,
            layer,
            profile,
            out_projection,
            fused_prepare,
        )
    })
}

fn run_layer_inner(
    context: &CudaContext,
    hidden: &Tensor,
    layer: &Layer,
    profile: bool,
    out_projection: &str,
    fused_prepare: bool,
) -> apxinf_core::Result<()> {
    {
        let _range = profile.then(|| apxinf_cuda::nvtx::range("qwen35.gdn.qkv_w4"));
        gemm::w4a16_write(context, hidden, layer.qkv.view(), &layer.qkv_output)?;
    }
    {
        let _range = profile.then(|| apxinf_cuda::nvtx::range("qwen35.gdn.z_w4"));
        gemm::w4a16_write(context, hidden, layer.z.view(), &layer.z_output)?;
    }
    {
        let _range = profile.then(|| {
            apxinf_cuda::nvtx::range(if fused_prepare {
                "qwen35.gdn.ab_packed_bf16"
            } else {
                "qwen35.gdn.a_b_bf16"
            })
        });
        if fused_prepare {
            bf16_linear(
                context,
                hidden,
                &layer.ab,
                &layer.ab_output,
                HIDDEN,
                2 * HEADS,
            )?;
        } else {
            bf16_linear(context, hidden, &layer.a, &layer.a_output, HIDDEN, HEADS)?;
            bf16_linear(context, hidden, &layer.b, &layer.b_output, HIDDEN, HEADS)?;
        }
    }
    if fused_prepare {
        let _range = profile.then(|| apxinf_cuda::nvtx::range("qwen35.gdn.conv4_prepare_fused"));
        qwen35_conv4_prepare_write(
            context,
            &layer.qkv_output.reshape(vec![CONV_DIM])?,
            &layer.conv,
            &layer.conv_state,
            &layer.ab_output.reshape(vec![2 * HEADS])?,
            &layer.a_log,
            &layer.dt_bias,
            &layer.a_output.reshape(vec![HEADS])?,
            &layer.b_output.reshape(vec![HEADS])?,
            &layer.query,
            &layer.key,
            &layer.value,
            &layer.g,
            &layer.beta,
        )?;
    } else {
        let _range = profile.then(|| apxinf_cuda::nvtx::range("qwen35.gdn.conv4_silu"));
        qwen35_conv4_silu_write(
            context,
            &layer.qkv_output.reshape(vec![CONV_DIM])?,
            &layer.conv,
            &layer.conv_state,
            &layer.conv_output,
        )?;
        let _range = profile.then(|| apxinf_cuda::nvtx::range("qwen35.gdn.prepare"));
        qwen35_prepare_write(
            context,
            &layer.conv_output,
            &layer.a_output.reshape(vec![HEADS])?,
            &layer.b_output.reshape(vec![HEADS])?,
            &layer.a_log,
            &layer.dt_bias,
            &layer.query,
            &layer.key,
            &layer.value,
            &layer.g,
            &layer.beta,
        )?;
    }
    {
        let _range = profile.then(|| apxinf_cuda::nvtx::range("qwen35.gdn.recurrent"));
        qwen35_recurrent_write(
            context,
            &layer.query,
            &layer.key,
            &layer.value,
            &layer.g,
            &layer.beta,
            &layer.recurrent_state,
            &layer.core_output,
        )?;
    }
    {
        let _range = profile.then(|| apxinf_cuda::nvtx::range("qwen35.gdn.gated_norm"));
        qwen35_gated_rmsnorm_write(
            context,
            &layer.core_output,
            &layer.z_output.reshape(vec![HEADS, DIM])?,
            &layer.norm,
            &layer.norm_output,
            RMS_EPSILON,
        )?;
    }
    {
        let _range = profile.then(|| {
            apxinf_cuda::nvtx::range(match out_projection {
                "w4" => "qwen35.gdn.out_proj_w4",
                "w8" => "qwen35.gdn.out_proj_w8a16",
                "w8a8" => "qwen35.gdn.out_proj_w8a8",
                _ => "qwen35.gdn.out_proj_bf16",
            })
        });
        let activation = layer.norm_output.reshape(vec![1, VALUE_WIDTH])?;
        match out_projection {
            "w4" => gemm::w4a16_write(
                context,
                &activation,
                layer.out_w4.view(),
                &layer.layer_output,
            ),
            "w8" => gemm::w8a16_write(
                context,
                &activation,
                layer.out_w8.view_w8a16(),
                &layer.layer_output,
            ),
            "w8a8" => gemm::w8a8_write(
                context,
                &activation,
                layer.out_w8.view_w8a8(),
                &layer.layer_output,
            ),
            _ => bf16_linear(
                context,
                &activation,
                &layer.out,
                &layer.layer_output,
                VALUE_WIDTH,
                HIDDEN,
            ),
        }
    }
}

fn quantize_bf16_w8(weight: &Tensor) -> Result<W8Weight, Box<dyn std::error::Error>> {
    if weight.dtype() != DType::BF16 || weight.shape().dims().len() != 2 {
        return Err("W8 conversion expects a 2D BF16 weight".into());
    }
    let output_dim = weight.shape().dims()[0];
    let input_dim = weight.shape().dims()[1];
    let source = weight.as_bf16()?;
    let mut values = vec![0_u8; output_dim * input_dim];
    let mut scales = vec![0.0_f32; output_dim];
    for row in 0..output_dim {
        let row_values = &source[row * input_dim..(row + 1) * input_dim];
        let maximum = row_values
            .iter()
            .map(|value| value.to_f32().abs())
            .fold(0.0_f32, f32::max);
        let scale = (maximum / 127.0).max(1.0e-12);
        scales[row] = scale;
        for (column, value) in row_values.iter().enumerate() {
            let quantized = (value.to_f32() / scale).round().clamp(-128.0, 127.0) as i8;
            values[row * input_dim + column] = quantized as u8;
        }
    }
    let values_i8 = CudaBuffer::alloc(values.len(), 0)?;
    values_i8.copy_from_host(&values)?;
    let scales_f32 = transfers::to_cuda(&Tensor::from_f32(vec![output_dim], &scales)?, 0)?;
    Ok(W8Weight {
        values_i8,
        scales_f32,
        input_dim,
        output_dim,
    })
}

fn quantize_bf16_w4(weight: &Tensor) -> Result<W4Weight, Box<dyn std::error::Error>> {
    if weight.dtype() != DType::BF16 || weight.shape().dims().len() != 2 {
        return Err("W4 conversion expects a 2D BF16 weight".into());
    }
    let output_dim = weight.shape().dims()[0];
    let input_dim = weight.shape().dims()[1];
    if output_dim % 8 != 0 || input_dim % 32 != 0 {
        return Err(
            format!("W4 conversion shape [{output_dim},{input_dim}] is unsupported").into(),
        );
    }
    let groups = input_dim / 32;
    let packed_cols = input_dim / 8;
    let source = weight.as_bf16()?;
    let mut packed = vec![0_i32; output_dim * packed_cols];
    let mut scales = vec![half::bf16::ZERO; output_dim * groups];
    let mut zero_points = vec![0_i32; (output_dim / 8) * groups];
    for row in 0..output_dim {
        for group in 0..groups {
            let start = row * input_dim + group * 32;
            let values = &source[start..start + 32];
            let mut minimum = f32::INFINITY;
            let mut maximum = f32::NEG_INFINITY;
            for value in values {
                minimum = minimum.min(value.to_f32());
                maximum = maximum.max(value.to_f32());
            }
            let scale = half::bf16::from_f32(((maximum - minimum) / 15.0).max(1.0e-8));
            let scale_f32 = scale.to_f32();
            let zero_unsigned = (-minimum / scale_f32).round().clamp(0.0, 15.0) as u32;
            scales[row * groups + group] = scale;
            zero_points[(row / 8) * groups + group] |=
                ((zero_unsigned & 15) << ((row & 7) * 4)) as i32;
            for pack in 0..4 {
                let mut word = 0_u32;
                for nibble in 0..8 {
                    let value = values[pack * 8 + nibble].to_f32();
                    let quantized = (value / scale_f32).round() as i32 + zero_unsigned as i32;
                    let unsigned = quantized.clamp(0, 15) as u32;
                    word |= unsigned << (nibble * 4);
                }
                packed[row * packed_cols + group * 4 + pack] = word as i32;
            }
        }
    }
    let packed_cpu = Tensor::from_i32(vec![output_dim, packed_cols], &packed)?;
    let scales_cpu = Tensor::from_bf16(vec![output_dim, groups], &scales)?;
    let zero_cpu = Tensor::from_i32(vec![output_dim / 8, groups], &zero_points)?;
    Ok(W4Weight {
        packed_gpu: transfers::to_cuda(&packed_cpu, 0)?,
        scales_gpu: transfers::to_cuda(&scales_cpu, 0)?,
        zero_gpu: transfers::to_cuda(&zero_cpu, 0)?,
        packed_cpu,
        scales_cpu,
        zero_cpu,
        input_dim,
        output_dim,
    })
}

fn bf16_linear(
    context: &CudaContext,
    input: &Tensor,
    hf_weight: &Tensor,
    output: &Tensor,
    input_dim: usize,
    output_dim: usize,
) -> apxinf_core::Result<()> {
    let input_buffer = CudaBuffer::from_tensor(input).map_err(apxinf_core::Error::Cuda)?;
    let weight_buffer = CudaBuffer::from_tensor(hf_weight).map_err(apxinf_core::Error::Cuda)?;
    let output_buffer = CudaBuffer::from_tensor(output).map_err(apxinf_core::Error::Cuda)?;
    gemm::write_ex(
        context,
        DType::BF16,
        CublasTranspose::None,
        CublasTranspose::Transpose,
        1,
        output_dim,
        input_dim,
        1.0,
        &input_buffer,
        input_dim as i32,
        &weight_buffer,
        input_dim as i32,
        0.0,
        &output_buffer,
        output_dim as i32,
    )
}

struct Endpoints {
    values: Vec<(String, Vec<f32>)>,
}

fn download_endpoints(
    layer: &Layer,
    fused_prepare: bool,
) -> Result<Endpoints, Box<dyn std::error::Error>> {
    let mut tensors = vec![
        ("qkv_projection", &layer.qkv_output),
        ("z_projection", &layer.z_output),
        ("a_projection", &layer.a_output),
        ("b_projection", &layer.b_output),
        ("conv_state", &layer.conv_state),
        ("query", &layer.query),
        ("key", &layer.key),
        ("value", &layer.value),
        ("g", &layer.g),
        ("beta", &layer.beta),
        ("recurrent_state", &layer.recurrent_state),
        ("core_output", &layer.core_output),
        ("gated_norm", &layer.norm_output),
        ("layer_output", &layer.layer_output),
    ];
    if !fused_prepare {
        tensors.push(("conv_output", &layer.conv_output));
    }
    let mut values = Vec::with_capacity(tensors.len());
    for (name, tensor) in tensors {
        values.push((name.to_owned(), transfers::to_cpu(tensor)?.to_f32_vec()?));
    }
    Ok(Endpoints { values })
}

fn concat_bf16_rows(first: &Tensor, second: &Tensor) -> Result<Tensor, Box<dyn std::error::Error>> {
    if first.dtype() != DType::BF16
        || second.dtype() != DType::BF16
        || first.shape().dims() != [HEADS, HIDDEN]
        || second.shape().dims() != [HEADS, HIDDEN]
    {
        return Err("a/b packed weight contract mismatch".into());
    }
    let mut values = Vec::with_capacity(2 * HEADS * HIDDEN);
    values.extend_from_slice(first.as_bf16()?);
    values.extend_from_slice(second.as_bf16()?);
    Ok(Tensor::from_bf16(vec![2 * HEADS, HIDDEN], &values)?)
}

fn cpu_layer(hidden: &Tensor, layer: &Layer) -> Result<Endpoints, Box<dyn std::error::Error>> {
    let hidden = hidden.as_bf16()?;
    let qkv = cpu_w4(hidden, &layer.qkv)?;
    let z = cpu_w4(hidden, &layer.z)?;
    let a = cpu_bf16_linear(hidden, &layer.a_cpu)?;
    let b = cpu_bf16_linear(hidden, &layer.b_cpu)?;
    let mut conv_state = vec![half::bf16::ZERO; CONV_DIM * CONV_KERNEL];
    let conv = cpu_conv(&qkv, layer.conv_cpu.as_bf16()?, &mut conv_state);
    let (query, key, value, g, beta) = cpu_prepare(
        &conv,
        &a,
        &b,
        layer.a_log_cpu.as_bf16()?,
        layer.dt_bias_cpu.as_bf16()?,
    );
    let mut recurrent_state = vec![0.0_f32; HEADS * DIM * DIM];
    let core = cpu_recurrent(&query, &key, &value, &g, &beta, &mut recurrent_state);
    let norm = cpu_gated_norm(&core, &z, layer.norm_cpu.as_bf16()?);
    let output = cpu_bf16_linear(&norm, &layer.out_cpu)?;
    Ok(Endpoints {
        values: vec![
            ("qkv_projection".into(), bf16_to_f32(&qkv)),
            ("z_projection".into(), bf16_to_f32(&z)),
            ("a_projection".into(), bf16_to_f32(&a)),
            ("b_projection".into(), bf16_to_f32(&b)),
            ("conv_state".into(), bf16_to_f32(&conv_state)),
            ("conv_output".into(), bf16_to_f32(&conv)),
            ("query".into(), bf16_to_f32(&query)),
            ("key".into(), bf16_to_f32(&key)),
            ("value".into(), bf16_to_f32(&value)),
            ("g".into(), g.clone()),
            ("beta".into(), beta.clone()),
            ("recurrent_state".into(), recurrent_state),
            ("core_output".into(), bf16_to_f32(&core)),
            ("gated_norm".into(), bf16_to_f32(&norm)),
            ("layer_output".into(), bf16_to_f32(&output)),
        ],
    })
}

fn cpu_w4(
    input: &[half::bf16],
    weight: &W4Weight,
) -> Result<Vec<half::bf16>, Box<dyn std::error::Error>> {
    let packed = weight.packed_cpu.as_i32()?;
    let scales = weight.scales_cpu.as_bf16()?;
    let zero = weight.zero_cpu.as_i32()?;
    let packed_cols = weight.input_dim / 8;
    let groups = weight.input_dim / 32;
    let mut output = vec![half::bf16::ZERO; weight.output_dim];
    for row in 0..weight.output_dim {
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

fn cpu_bf16_linear(
    input: &[half::bf16],
    weight: &Tensor,
) -> Result<Vec<half::bf16>, Box<dyn std::error::Error>> {
    let dims = weight.shape().dims();
    let output_dim = dims[0];
    let input_dim = dims[1];
    let weight = weight.as_bf16()?;
    let mut output = vec![half::bf16::ZERO; output_dim];
    for row in 0..output_dim {
        let mut sum = 0.0_f32;
        for column in 0..input_dim {
            sum = input[column]
                .to_f32()
                .mul_add(weight[row * input_dim + column].to_f32(), sum);
        }
        output[row] = half::bf16::from_f32(sum);
    }
    Ok(output)
}

fn cpu_conv(
    input: &[half::bf16],
    weight: &[half::bf16],
    state: &mut [half::bf16],
) -> Vec<half::bf16> {
    let mut output = vec![half::bf16::ZERO; CONV_DIM];
    for channel in 0..CONV_DIM {
        let offset = channel * 4;
        state[offset] = state[offset + 1];
        state[offset + 1] = state[offset + 2];
        state[offset + 2] = state[offset + 3];
        state[offset + 3] = input[channel];
        let mut sum = 0.0_f32;
        for index in 0..4 {
            sum = state[offset + index]
                .to_f32()
                .mul_add(weight[offset + index].to_f32(), sum);
        }
        output[channel] = half::bf16::from_f32(sum / (1.0 + (-sum).exp()));
    }
    output
}

fn cpu_prepare(
    conv: &[half::bf16],
    a: &[half::bf16],
    b: &[half::bf16],
    a_log: &[half::bf16],
    dt_bias: &[half::bf16],
) -> (
    Vec<half::bf16>,
    Vec<half::bf16>,
    Vec<half::bf16>,
    Vec<f32>,
    Vec<f32>,
) {
    let mut query = vec![half::bf16::ZERO; VALUE_WIDTH];
    let mut key = query.clone();
    let mut value = query.clone();
    let mut g = vec![0.0_f32; HEADS];
    let mut beta = vec![0.0_f32; HEADS];
    for head in 0..HEADS {
        let source = head / 3;
        for dimension in 0..DIM {
            query[head * DIM + dimension] = conv[source * DIM + dimension];
            key[head * DIM + dimension] = conv[KEY_WIDTH + source * DIM + dimension];
            value[head * DIM + dimension] = conv[2 * KEY_WIDTH + head * DIM + dimension];
        }
        let dt = a[head].to_f32() + dt_bias[head].to_f32();
        let softplus = if dt > 20.0 { dt } else { dt.exp().ln_1p() };
        g[head] = -a_log[head].to_f32().exp() * softplus;
        beta[head] = 1.0 / (1.0 + (-b[head].to_f32()).exp());
    }
    (query, key, value, g, beta)
}

fn cpu_recurrent(
    query: &[half::bf16],
    key: &[half::bf16],
    value: &[half::bf16],
    g: &[f32],
    beta: &[f32],
    state: &mut [f32],
) -> Vec<half::bf16> {
    let mut output = vec![half::bf16::ZERO; VALUE_WIDTH];
    let query_scale = 1.0 / (DIM as f32).sqrt();
    for head in 0..HEADS {
        let offset = head * DIM;
        let qsum = query[offset..offset + DIM]
            .iter()
            .map(|x| x.to_f32().powi(2))
            .sum::<f32>();
        let ksum = key[offset..offset + DIM]
            .iter()
            .map(|x| x.to_f32().powi(2))
            .sum::<f32>();
        let qnorm = (qsum + 1e-6).sqrt().recip() * query_scale;
        let knorm = (ksum + 1e-6).sqrt().recip();
        let q = query[offset..offset + DIM]
            .iter()
            .map(|x| x.to_f32() * qnorm)
            .collect::<Vec<_>>();
        let k = key[offset..offset + DIM]
            .iter()
            .map(|x| x.to_f32() * knorm)
            .collect::<Vec<_>>();
        let qk = q.iter().zip(&k).map(|(q, k)| q * k).sum::<f32>();
        let decay = g[head].exp();
        let state_base = head * DIM * DIM;
        for vdim in 0..DIM {
            let mut kmem = 0.0;
            let mut qmem = 0.0;
            for kdim in 0..DIM {
                let old = state[state_base + kdim * DIM + vdim];
                kmem = old.mul_add(k[kdim], kmem);
                qmem = old.mul_add(q[kdim], qmem);
            }
            let delta = (value[offset + vdim].to_f32() - decay * kmem) * beta[head];
            output[offset + vdim] = half::bf16::from_f32(decay * qmem + delta * qk);
            for kdim in 0..DIM {
                let index = state_base + kdim * DIM + vdim;
                state[index] = k[kdim].mul_add(delta, state[index] * decay);
            }
        }
    }
    output
}

fn cpu_gated_norm(
    input: &[half::bf16],
    gate: &[half::bf16],
    weight: &[half::bf16],
) -> Vec<half::bf16> {
    let mut output = vec![half::bf16::ZERO; VALUE_WIDTH];
    for head in 0..HEADS {
        let offset = head * DIM;
        let variance = input[offset..offset + DIM]
            .iter()
            .map(|x| x.to_f32().powi(2))
            .sum::<f32>()
            / DIM as f32;
        let inverse = (variance + RMS_EPSILON).sqrt().recip();
        for dimension in 0..DIM {
            let normalized = half::bf16::from_f32(input[offset + dimension].to_f32() * inverse);
            let weighted = half::bf16::from_f32(normalized.to_f32() * weight[dimension].to_f32());
            let z = gate[offset + dimension].to_f32();
            output[offset + dimension] =
                half::bf16::from_f32(weighted.to_f32() * z / (1.0 + (-z).exp()));
        }
    }
    output
}

fn compare_endpoints(
    actual: &Endpoints,
    expected: &Endpoints,
) -> Result<Vec<(String, Metrics)>, String> {
    let mut results = Vec::new();
    for (name, actual) in &actual.values {
        let expected = expected
            .values
            .iter()
            .find(|(candidate, _)| candidate == name)
            .ok_or_else(|| format!("missing expected endpoint {name}"))?;
        let metrics = measure(actual, &expected.1)?;
        if metrics.cosine < 0.999 || metrics.relative_l2 > 0.02 {
            return Err(format!("endpoint {name} failed: {}", metrics_text(metrics)));
        }
        results.push((name.clone(), metrics));
    }
    Ok(results)
}

fn measure(actual: &[f32], expected: &[f32]) -> Result<Metrics, String> {
    if actual.len() != expected.len() || actual.is_empty() {
        return Err("comparison length mismatch".into());
    }
    let mut dot = 0.0;
    let mut aa = 0.0;
    let mut ee = 0.0;
    let mut error = 0.0;
    let mut max_abs = 0.0_f64;
    for (&a, &e) in actual.iter().zip(expected) {
        if !a.is_finite() || !e.is_finite() {
            return Err("non-finite endpoint".into());
        }
        let a = f64::from(a);
        let e = f64::from(e);
        dot += a * e;
        aa += a * a;
        ee += e * e;
        error += (a - e).powi(2);
        max_abs = max_abs.max((a - e).abs());
    }
    Ok(Metrics {
        cosine: if aa == 0.0 && ee == 0.0 {
            1.0
        } else {
            dot / (aa.sqrt() * ee.sqrt())
        },
        relative_l2: if ee == 0.0 {
            error.sqrt()
        } else {
            (error / ee).sqrt()
        },
        max_abs,
    })
}

fn reset_states(layer: &mut Layer) -> Result<(), Box<dyn std::error::Error>> {
    transfers::copy_cpu_to_cuda(
        &Tensor::zeros(vec![CONV_DIM, CONV_KERNEL], DType::BF16),
        &layer.conv_state,
    )?;
    transfers::copy_cpu_to_cuda(
        &Tensor::zeros(vec![HEADS, DIM, DIM], DType::F32),
        &layer.recurrent_state,
    )?;
    Ok(())
}

fn bf16_to_f32(values: &[half::bf16]) -> Vec<f32> {
    values.iter().map(|value| value.to_f32()).collect()
}

fn metrics_text(metrics: Metrics) -> String {
    format!(
        "cosine={}, relative_l2={}, max_abs={}",
        metrics.cosine, metrics.relative_l2, metrics.max_abs
    )
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
