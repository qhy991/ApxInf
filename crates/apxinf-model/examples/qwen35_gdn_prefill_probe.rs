//! Real layer-1 Qwen3.5 GDN M=8 prefill against eight serial decode layers.

use std::cmp::Ordering;
use std::path::Path;
use std::time::Instant;

use apxinf_core::{DType, Tensor};
use apxinf_cuda::kernels::gdn::{
    qwen35_conv4_prepare_m8_write, qwen35_conv4_prepare_write, qwen35_gated_rmsnorm_m8_write,
    qwen35_gated_rmsnorm_write, qwen35_recurrent_m8_write, qwen35_recurrent_write,
    QWEN35_GDN_CONV_DIM as CONV_DIM, QWEN35_GDN_CONV_KERNEL as CONV_KERNEL,
    QWEN35_GDN_HEADS as HEADS, QWEN35_GDN_KEY_DIM as DIM,
};
use apxinf_cuda::kernels::gemm::{self, W4A16Layout, W4A16WeightView};
use apxinf_cuda::kernels::{activation, qwen35_common};
use apxinf_cuda::tuning::KERNEL_BUILD_ID;
use apxinf_cuda::{transfers, CublasTranspose, CudaBuffer, CudaContext};
use apxinf_loader::safetensors::{self, CheckpointManifest};

const TOKENS: usize = 8;
const HIDDEN: usize = 5120;
const VALUE_WIDTH: usize = HEADS * DIM;
const AB_WIDTH: usize = 2 * HEADS;
const INTERMEDIATE: usize = 17408;
const PREFIX: &str = "model.language_model.layers.1.linear_attn";
const MLP_PREFIX: &str = "model.language_model.layers.1.mlp";
const RMS_EPSILON: f32 = 1.0e-6;
const WARMUPS: usize = 3;
const PAIRS: usize = 5;
const CALLS: usize = 1;

struct W4 {
    packed: Tensor,
    scales: Tensor,
    zero: Tensor,
    input: usize,
    output: usize,
}

struct CpuW4 {
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

struct Weights {
    qkv: W4,
    z: W4,
    out: W4,
    ab: Tensor,
    conv: Tensor,
    a_log: Tensor,
    dt_bias: Tensor,
    norm: Tensor,
    input_norm: Tensor,
    post_attention_norm: Tensor,
    next_input_norm: Tensor,
    mlp_gate_up: W4,
    mlp_down: W4,
}

struct Rows {
    input: Vec<Tensor>,
    residual: Vec<Tensor>,
    input_normalized: Vec<Tensor>,
    qkv: Vec<Tensor>,
    z: Vec<Tensor>,
    ab: Vec<Tensor>,
    a: Vec<Tensor>,
    b: Vec<Tensor>,
    query: Vec<Tensor>,
    key: Vec<Tensor>,
    value: Vec<Tensor>,
    g: Vec<Tensor>,
    beta: Vec<Tensor>,
    core: Vec<Tensor>,
    normalized: Vec<Tensor>,
    output: Vec<Tensor>,
    mlp_input: Vec<Tensor>,
    mlp_gate_up: Vec<Tensor>,
    mlp_hidden: Vec<Tensor>,
    mlp_delta: Vec<Tensor>,
    next_normalized: Vec<Tensor>,
    conv_state: Tensor,
    recurrent_state: Tensor,
}

struct Tile {
    input: Tensor,
    residual: Tensor,
    input_normalized: Tensor,
    qkv: Tensor,
    z: Tensor,
    ab: Tensor,
    a: Tensor,
    b: Tensor,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    g: Tensor,
    beta: Tensor,
    core: Tensor,
    normalized: Tensor,
    output: Tensor,
    mlp_input: Tensor,
    mlp_gate_up: Tensor,
    mlp_hidden: Tensor,
    mlp_delta: Tensor,
    next_normalized: Tensor,
    conv_state: Tensor,
    recurrent_state: Tensor,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_dir = std::env::args()
        .nth(1)
        .ok_or("usage: qwen35_gdn_prefill_probe MODEL_DIR")?;
    let manifest = safetensors::inspect_path(Path::new(&model_dir))?;
    let context = CudaContext::new(0)?;
    if context.caps().sm != 89 {
        return Err(format!(
            "GDN prefill probe is frozen for SM89, got SM{}",
            context.caps().sm
        )
        .into());
    }
    let weights = load_weights(&manifest)?;
    let input_values = deterministic_input();
    let serial = serial_workspace(&input_values)?;
    let candidate = candidate_workspace(&input_values)?;

    run_serial(&context, &weights, &serial)?;
    run_candidate(&context, &weights, &candidate)?;
    context.synchronize()?;
    let correctness = compare_all(&serial, &candidate)?;
    if correctness.iter().any(|(_, different)| *different != 0) {
        return Err(format!("M8 GDN layer differs from serial M1: {correctness:?}").into());
    }

    for _ in 0..WARMUPS {
        reset_serial(&serial, &input_values)?;
        run_serial(&context, &weights, &serial)?;
        reset_candidate(&candidate, &input_values)?;
        run_candidate(&context, &weights, &candidate)?;
    }
    context.synchronize()?;
    let mut serial_samples = Vec::with_capacity(PAIRS);
    let mut candidate_samples = Vec::with_capacity(PAIRS);
    let mut records = Vec::with_capacity(2 * PAIRS);
    for pair in 0..PAIRS {
        let candidate_first = pair % 2 == 1;
        for order_index in 0..2 {
            let candidate_arm = (order_index == 0) == candidate_first;
            if candidate_arm {
                reset_candidate(&candidate, &input_values)?;
            } else {
                reset_serial(&serial, &input_values)?;
            }
            let start = Instant::now();
            for _ in 0..CALLS {
                if candidate_arm {
                    run_candidate(&context, &weights, &candidate)?;
                } else {
                    run_serial(&context, &weights, &serial)?;
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
                "arm":if candidate_arm{"m8"}else{"serial_m1"},
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
            "schema":"apxinf.qwen35.gdn_block_prefill_probe.v1",
            "model_dir":model_dir,"layer":1,
            "contract":{
                "tokens":TOKENS,"hidden":HIDDEN,"qkv_width":CONV_DIM,"value_width":VALUE_WIDTH,
                "path":"input offset RMSNorm + complete stateful GDN + residual/post norm + complete MLP + residual/next norm",
                "gdn":"qkv/z/out M8 W4 + eight M1 a/b BF16 projections + causal conv/recurrent scan + gated norm",
                "mlp":"packed gate/up M8 W4 + eight SiLU*Mul row views + down M8 W4",
                "numerical_order":"a/b keeps the accepted M1 cuBLAS accumulation and BF16 seam; M8 cuBLAS is a separately rejected exactness arm",
                "oracle":"eight serial production-shape layer-1 GDN executions",
                "state_semantics":"one causal conv and recurrent state carried across all eight tokens",
            },
            "device":{
                "name":context.caps().device_name,"sm":context.caps().sm,
                "multiprocessors":context.caps().multiprocessor_count,
                "cuda":context.library_versions().cuda,"cublas":context.library_versions().cublas,
            },
            "kernel_build_id":KERNEL_BUILD_ID,
            "correctness":{
                "different_values":correctness,
                "comparison":"BF16 and FP32 bitwise identity at every norm, projection, state, residual, MLP, and next-layer endpoint",
                "pass":true,
            },
            "timing":{
                "boundary":"complete real layer-1 GDN decoder block through next-layer input norm and stream synchronize; state/residual reset excluded",
                "warmups_per_arm":WARMUPS,"pairs":PAIRS,"calls_per_sample":CALLS,
                "records":records,"serial_raw_us":serial_samples,"candidate_raw_us":candidate_samples,
                "serial_median_us":median(&serial_samples),
                "candidate_median_us":median(&candidate_samples),
                "median_speedup":median(&speedups),"candidate_wins":wins,
                "candidate_tokens_per_second":TOKENS as f64*1.0e6/median(&candidate_samples),
            },
            "evidence_level":"complete-stateful-gdn-decoder-block-prefill",
            "model_promoted":false,
        }))?
    );
    Ok(())
}

fn load_weights(manifest: &CheckpointManifest) -> Result<Weights, Box<dyn std::error::Error>> {
    let a = load_tensor(manifest, &format!("{PREFIX}.in_proj_a.weight"))?;
    let b = load_tensor(manifest, &format!("{PREFIX}.in_proj_b.weight"))?;
    if a.shape().dims() != [HEADS, HIDDEN] || b.shape().dims() != [HEADS, HIDDEN] {
        return Err("GDN a/b checkpoint shape mismatch".into());
    }
    let mut ab = a.as_bf16()?.to_vec();
    ab.extend_from_slice(b.as_bf16()?);
    Ok(Weights {
        qkv: load_w4(manifest, &format!("{PREFIX}.in_proj_qkv"))?,
        z: load_w4(manifest, &format!("{PREFIX}.in_proj_z"))?,
        out: load_w4(manifest, &format!("{PREFIX}.out_proj"))?,
        ab: to_gpu(Tensor::from_bf16(vec![AB_WIDTH, HIDDEN], &ab)?)?,
        conv: to_gpu(load_tensor(manifest, &format!("{PREFIX}.conv1d.weight"))?)?,
        a_log: to_gpu(load_tensor(manifest, &format!("{PREFIX}.A_log"))?)?,
        dt_bias: to_gpu(load_tensor(manifest, &format!("{PREFIX}.dt_bias"))?)?,
        norm: to_gpu(load_tensor(manifest, &format!("{PREFIX}.norm.weight"))?)?,
        input_norm: to_gpu(load_tensor(
            manifest,
            "model.language_model.layers.1.input_layernorm.weight",
        )?)?,
        post_attention_norm: to_gpu(load_tensor(
            manifest,
            "model.language_model.layers.1.post_attention_layernorm.weight",
        )?)?,
        next_input_norm: to_gpu(load_tensor(
            manifest,
            "model.language_model.layers.2.input_layernorm.weight",
        )?)?,
        mlp_gate_up: load_w4_pair(
            manifest,
            &format!("{MLP_PREFIX}.gate_proj"),
            &format!("{MLP_PREFIX}.up_proj"),
        )?,
        mlp_down: load_w4(manifest, &format!("{MLP_PREFIX}.down_proj"))?,
    })
}

fn load_w4(manifest: &CheckpointManifest, base: &str) -> Result<W4, Box<dyn std::error::Error>> {
    to_w4(load_w4_cpu(manifest, base)?)
}

fn load_w4_cpu(
    manifest: &CheckpointManifest,
    base: &str,
) -> Result<CpuW4, Box<dyn std::error::Error>> {
    let shape = safetensors::read_small_i64(
        manifest
            .tensor(&format!("{base}.weight_shape"))
            .ok_or_else(|| format!("missing `{base}.weight_shape`"))?,
    )?;
    Ok(CpuW4 {
        packed: load_tensor(manifest, &format!("{base}.weight_packed"))?,
        scales: load_tensor(manifest, &format!("{base}.weight_scale"))?,
        zero: load_tensor(manifest, &format!("{base}.weight_zero_point"))?,
        input: usize::try_from(shape[1])?,
        output: usize::try_from(shape[0])?,
    })
}

fn to_w4(weight: CpuW4) -> Result<W4, Box<dyn std::error::Error>> {
    Ok(W4 {
        packed: to_gpu(weight.packed)?,
        scales: to_gpu(weight.scales)?,
        zero: to_gpu(weight.zero)?,
        input: weight.input,
        output: weight.output,
    })
}

fn load_w4_pair(
    manifest: &CheckpointManifest,
    first: &str,
    second: &str,
) -> Result<W4, Box<dyn std::error::Error>> {
    let first = load_w4_cpu(manifest, first)?;
    let second = load_w4_cpu(manifest, second)?;
    if first.input != second.input || first.output != second.output {
        return Err("paired W4 projection shape mismatch".into());
    }
    let mut packed = first.packed.as_i32()?.to_vec();
    packed.extend_from_slice(second.packed.as_i32()?);
    let mut scales = first.scales.as_bf16()?.to_vec();
    scales.extend_from_slice(second.scales.as_bf16()?);
    let mut zero = first.zero.as_i32()?.to_vec();
    zero.extend_from_slice(second.zero.as_i32()?);
    to_w4(CpuW4 {
        packed: Tensor::from_i32(vec![2 * first.output, first.input / 8], &packed)?,
        scales: Tensor::from_bf16(vec![2 * first.output, first.input / 32], &scales)?,
        zero: Tensor::from_i32(vec![2 * first.output / 8, first.input / 32], &zero)?,
        input: first.input,
        output: 2 * first.output,
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

fn to_gpu(tensor: Tensor) -> Result<Tensor, Box<dyn std::error::Error>> {
    Ok(transfers::to_cuda(&tensor, 0)?)
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

fn serial_workspace(input: &[half::bf16]) -> Result<Rows, Box<dyn std::error::Error>> {
    let mut rows = Rows {
        input: Vec::with_capacity(TOKENS),
        residual: Vec::with_capacity(TOKENS),
        input_normalized: Vec::with_capacity(TOKENS),
        qkv: Vec::with_capacity(TOKENS),
        z: Vec::with_capacity(TOKENS),
        ab: Vec::with_capacity(TOKENS),
        a: Vec::with_capacity(TOKENS),
        b: Vec::with_capacity(TOKENS),
        query: Vec::with_capacity(TOKENS),
        key: Vec::with_capacity(TOKENS),
        value: Vec::with_capacity(TOKENS),
        g: Vec::with_capacity(TOKENS),
        beta: Vec::with_capacity(TOKENS),
        core: Vec::with_capacity(TOKENS),
        normalized: Vec::with_capacity(TOKENS),
        output: Vec::with_capacity(TOKENS),
        mlp_input: Vec::with_capacity(TOKENS),
        mlp_gate_up: Vec::with_capacity(TOKENS),
        mlp_hidden: Vec::with_capacity(TOKENS),
        mlp_delta: Vec::with_capacity(TOKENS),
        next_normalized: Vec::with_capacity(TOKENS),
        conv_state: gpu_zeros(&[CONV_DIM, CONV_KERNEL], DType::BF16)?,
        recurrent_state: gpu_zeros(&[HEADS, DIM, DIM], DType::F32)?,
    };
    for token in 0..TOKENS {
        rows.input.push(to_gpu(Tensor::from_bf16(
            vec![1, HIDDEN],
            &input[token * HIDDEN..(token + 1) * HIDDEN],
        )?)?);
        rows.residual.push(to_gpu(Tensor::from_bf16(
            vec![1, HIDDEN],
            &input[token * HIDDEN..(token + 1) * HIDDEN],
        )?)?);
        rows.input_normalized
            .push(gpu_zeros(&[1, HIDDEN], DType::BF16)?);
        rows.qkv.push(gpu_zeros(&[1, CONV_DIM], DType::BF16)?);
        rows.z.push(gpu_zeros(&[1, VALUE_WIDTH], DType::BF16)?);
        rows.ab.push(gpu_zeros(&[1, AB_WIDTH], DType::BF16)?);
        rows.a.push(gpu_zeros(&[HEADS], DType::BF16)?);
        rows.b.push(gpu_zeros(&[HEADS], DType::BF16)?);
        rows.query.push(gpu_zeros(&[HEADS, DIM], DType::BF16)?);
        rows.key.push(gpu_zeros(&[HEADS, DIM], DType::BF16)?);
        rows.value.push(gpu_zeros(&[HEADS, DIM], DType::BF16)?);
        rows.g.push(gpu_zeros(&[HEADS], DType::F32)?);
        rows.beta.push(gpu_zeros(&[HEADS], DType::F32)?);
        rows.core.push(gpu_zeros(&[HEADS, DIM], DType::BF16)?);
        rows.normalized.push(gpu_zeros(&[HEADS, DIM], DType::BF16)?);
        rows.output.push(gpu_zeros(&[1, HIDDEN], DType::BF16)?);
        rows.mlp_input.push(gpu_zeros(&[1, HIDDEN], DType::BF16)?);
        rows.mlp_gate_up
            .push(gpu_zeros(&[1, 2 * INTERMEDIATE], DType::BF16)?);
        rows.mlp_hidden
            .push(gpu_zeros(&[1, INTERMEDIATE], DType::BF16)?);
        rows.mlp_delta.push(gpu_zeros(&[1, HIDDEN], DType::BF16)?);
        rows.next_normalized
            .push(gpu_zeros(&[1, HIDDEN], DType::BF16)?);
    }
    Ok(rows)
}

fn candidate_workspace(input: &[half::bf16]) -> Result<Tile, Box<dyn std::error::Error>> {
    Ok(Tile {
        input: to_gpu(Tensor::from_bf16(vec![TOKENS, HIDDEN], input)?)?,
        residual: to_gpu(Tensor::from_bf16(vec![TOKENS, HIDDEN], input)?)?,
        input_normalized: gpu_zeros(&[TOKENS, HIDDEN], DType::BF16)?,
        qkv: gpu_zeros(&[TOKENS, CONV_DIM], DType::BF16)?,
        z: gpu_zeros(&[TOKENS, VALUE_WIDTH], DType::BF16)?,
        ab: gpu_zeros(&[TOKENS, AB_WIDTH], DType::BF16)?,
        a: gpu_zeros(&[TOKENS, HEADS], DType::BF16)?,
        b: gpu_zeros(&[TOKENS, HEADS], DType::BF16)?,
        query: gpu_zeros(&[TOKENS, HEADS, DIM], DType::BF16)?,
        key: gpu_zeros(&[TOKENS, HEADS, DIM], DType::BF16)?,
        value: gpu_zeros(&[TOKENS, HEADS, DIM], DType::BF16)?,
        g: gpu_zeros(&[TOKENS, HEADS], DType::F32)?,
        beta: gpu_zeros(&[TOKENS, HEADS], DType::F32)?,
        core: gpu_zeros(&[TOKENS, HEADS, DIM], DType::BF16)?,
        normalized: gpu_zeros(&[TOKENS, HEADS, DIM], DType::BF16)?,
        output: gpu_zeros(&[TOKENS, HIDDEN], DType::BF16)?,
        mlp_input: gpu_zeros(&[TOKENS, HIDDEN], DType::BF16)?,
        mlp_gate_up: gpu_zeros(&[TOKENS, 2 * INTERMEDIATE], DType::BF16)?,
        mlp_hidden: gpu_zeros(&[TOKENS, INTERMEDIATE], DType::BF16)?,
        mlp_delta: gpu_zeros(&[TOKENS, HIDDEN], DType::BF16)?,
        next_normalized: gpu_zeros(&[TOKENS, HIDDEN], DType::BF16)?,
        conv_state: gpu_zeros(&[CONV_DIM, CONV_KERNEL], DType::BF16)?,
        recurrent_state: gpu_zeros(&[HEADS, DIM, DIM], DType::F32)?,
    })
}

fn gpu_zeros(shape: &[usize], dtype: DType) -> Result<Tensor, Box<dyn std::error::Error>> {
    Ok(transfers::to_cuda(
        &Tensor::zeros(shape.to_vec(), dtype),
        0,
    )?)
}

fn reset_serial(rows: &Rows, input: &[half::bf16]) -> Result<(), Box<dyn std::error::Error>> {
    for token in 0..TOKENS {
        transfers::copy_cpu_to_cuda(
            &Tensor::from_bf16(
                vec![1, HIDDEN],
                &input[token * HIDDEN..(token + 1) * HIDDEN],
            )?,
            &rows.residual[token],
        )?;
    }
    transfers::copy_cpu_to_cuda(
        &Tensor::zeros(vec![CONV_DIM, CONV_KERNEL], DType::BF16),
        &rows.conv_state,
    )?;
    transfers::copy_cpu_to_cuda(
        &Tensor::zeros(vec![HEADS, DIM, DIM], DType::F32),
        &rows.recurrent_state,
    )?;
    Ok(())
}

fn reset_candidate(tile: &Tile, input: &[half::bf16]) -> Result<(), Box<dyn std::error::Error>> {
    transfers::copy_cpu_to_cuda(
        &Tensor::from_bf16(vec![TOKENS, HIDDEN], input)?,
        &tile.residual,
    )?;
    transfers::copy_cpu_to_cuda(
        &Tensor::zeros(vec![CONV_DIM, CONV_KERNEL], DType::BF16),
        &tile.conv_state,
    )?;
    transfers::copy_cpu_to_cuda(
        &Tensor::zeros(vec![HEADS, DIM, DIM], DType::F32),
        &tile.recurrent_state,
    )?;
    Ok(())
}

fn run_serial(context: &CudaContext, weights: &Weights, rows: &Rows) -> apxinf_core::Result<()> {
    for token in 0..TOKENS {
        qwen35_common::rmsnorm_offset_write(
            context,
            &rows.input[token],
            &weights.input_norm,
            &rows.input_normalized[token],
            RMS_EPSILON,
        )?;
        gemm::w4a16_write(
            context,
            &rows.input_normalized[token],
            weights.qkv.view(),
            &rows.qkv[token],
        )?;
        gemm::w4a16_write(
            context,
            &rows.input_normalized[token],
            weights.z.view(),
            &rows.z[token],
        )?;
        bf16_linear(
            context,
            &rows.input_normalized[token],
            &weights.ab,
            &rows.ab[token],
            1,
            HIDDEN,
            AB_WIDTH,
        )?;
        qwen35_conv4_prepare_write(
            context,
            &rows.qkv[token].reshape(vec![CONV_DIM])?,
            &weights.conv,
            &rows.conv_state,
            &rows.ab[token].reshape(vec![AB_WIDTH])?,
            &weights.a_log,
            &weights.dt_bias,
            &rows.a[token],
            &rows.b[token],
            &rows.query[token],
            &rows.key[token],
            &rows.value[token],
            &rows.g[token],
            &rows.beta[token],
        )?;
        qwen35_recurrent_write(
            context,
            &rows.query[token],
            &rows.key[token],
            &rows.value[token],
            &rows.g[token],
            &rows.beta[token],
            &rows.recurrent_state,
            &rows.core[token],
        )?;
        qwen35_gated_rmsnorm_write(
            context,
            &rows.core[token],
            &rows.z[token].reshape(vec![HEADS, DIM])?,
            &weights.norm,
            &rows.normalized[token],
            RMS_EPSILON,
        )?;
        gemm::w4a16_write(
            context,
            &rows.normalized[token].reshape(vec![1, VALUE_WIDTH])?,
            weights.out.view(),
            &rows.output[token],
        )?;
        qwen35_common::residual_add_rmsnorm_offset_write(
            context,
            &rows.residual[token],
            &rows.output[token],
            &weights.post_attention_norm,
            &rows.mlp_input[token],
            RMS_EPSILON,
        )?;
        gemm::w4a16_write(
            context,
            &rows.mlp_input[token],
            weights.mlp_gate_up.view(),
            &rows.mlp_gate_up[token],
        )?;
        activation::silu_mul_bf16_into(
            context,
            &CudaBuffer::from_tensor(&rows.mlp_gate_up[token]).map_err(apxinf_core::Error::Cuda)?,
            &CudaBuffer::from_tensor(&rows.mlp_hidden[token]).map_err(apxinf_core::Error::Cuda)?,
            INTERMEDIATE,
        )?;
        gemm::w4a16_write(
            context,
            &rows.mlp_hidden[token],
            weights.mlp_down.view(),
            &rows.mlp_delta[token],
        )?;
        qwen35_common::residual_add_rmsnorm_offset_write(
            context,
            &rows.residual[token],
            &rows.mlp_delta[token],
            &weights.next_input_norm,
            &rows.next_normalized[token],
            RMS_EPSILON,
        )?;
    }
    Ok(())
}

fn run_candidate(context: &CudaContext, weights: &Weights, tile: &Tile) -> apxinf_core::Result<()> {
    qwen35_common::rmsnorm_offset_write(
        context,
        &tile.input,
        &weights.input_norm,
        &tile.input_normalized,
        RMS_EPSILON,
    )?;
    gemm::w4a16_m8_write(
        context,
        &tile.input_normalized,
        weights.qkv.view(),
        &tile.qkv,
    )?;
    gemm::w4a16_m8_write(context, &tile.input_normalized, weights.z.view(), &tile.z)?;
    bf16_linear_serial_rows(
        context,
        &tile.input_normalized,
        &weights.ab,
        &tile.ab,
        TOKENS,
        HIDDEN,
        AB_WIDTH,
    )?;
    qwen35_conv4_prepare_m8_write(
        context,
        &tile.qkv,
        &weights.conv,
        &tile.conv_state,
        &tile.ab,
        &weights.a_log,
        &weights.dt_bias,
        &tile.a,
        &tile.b,
        &tile.query,
        &tile.key,
        &tile.value,
        &tile.g,
        &tile.beta,
    )?;
    qwen35_recurrent_m8_write(
        context,
        &tile.query,
        &tile.key,
        &tile.value,
        &tile.g,
        &tile.beta,
        &tile.recurrent_state,
        &tile.core,
    )?;
    qwen35_gated_rmsnorm_m8_write(
        context,
        &tile.core,
        &tile.z.reshape(vec![TOKENS, HEADS, DIM])?,
        &weights.norm,
        &tile.normalized,
        RMS_EPSILON,
    )?;
    gemm::w4a16_m8_write(
        context,
        &tile.normalized.reshape(vec![TOKENS, VALUE_WIDTH])?,
        weights.out.view(),
        &tile.output,
    )?;
    qwen35_common::residual_add_rmsnorm_offset_write(
        context,
        &tile.residual,
        &tile.output,
        &weights.post_attention_norm,
        &tile.mlp_input,
        RMS_EPSILON,
    )?;
    gemm::w4a16_m8_write(
        context,
        &tile.mlp_input,
        weights.mlp_gate_up.view(),
        &tile.mlp_gate_up,
    )?;
    let gate_up = CudaBuffer::from_tensor(&tile.mlp_gate_up).map_err(apxinf_core::Error::Cuda)?;
    let hidden = CudaBuffer::from_tensor(&tile.mlp_hidden).map_err(apxinf_core::Error::Cuda)?;
    let gate_up_bytes = 2 * INTERMEDIATE * DType::BF16.size_in_bytes();
    let hidden_bytes = INTERMEDIATE * DType::BF16.size_in_bytes();
    for token in 0..TOKENS {
        activation::silu_mul_bf16_into(
            context,
            &gate_up
                .view(token * gate_up_bytes, gate_up_bytes)
                .map_err(apxinf_core::Error::Cuda)?,
            &hidden
                .view(token * hidden_bytes, hidden_bytes)
                .map_err(apxinf_core::Error::Cuda)?,
            INTERMEDIATE,
        )?;
    }
    gemm::w4a16_m8_write(
        context,
        &tile.mlp_hidden,
        weights.mlp_down.view(),
        &tile.mlp_delta,
    )?;
    qwen35_common::residual_add_rmsnorm_offset_write(
        context,
        &tile.residual,
        &tile.mlp_delta,
        &weights.next_input_norm,
        &tile.next_normalized,
        RMS_EPSILON,
    )
}

fn bf16_linear_serial_rows(
    context: &CudaContext,
    input: &Tensor,
    weight: &Tensor,
    output: &Tensor,
    rows: usize,
    input_dim: usize,
    output_dim: usize,
) -> apxinf_core::Result<()> {
    let input = CudaBuffer::from_tensor(input).map_err(apxinf_core::Error::Cuda)?;
    let weight = CudaBuffer::from_tensor(weight).map_err(apxinf_core::Error::Cuda)?;
    let output = CudaBuffer::from_tensor(output).map_err(apxinf_core::Error::Cuda)?;
    let input_bytes = input_dim * DType::BF16.size_in_bytes();
    let output_bytes = output_dim * DType::BF16.size_in_bytes();
    for row in 0..rows {
        let input_row = input
            .view(row * input_bytes, input_bytes)
            .map_err(apxinf_core::Error::Cuda)?;
        let output_row = output
            .view(row * output_bytes, output_bytes)
            .map_err(apxinf_core::Error::Cuda)?;
        gemm::write_ex(
            context,
            DType::BF16,
            CublasTranspose::None,
            CublasTranspose::Transpose,
            1,
            output_dim,
            input_dim,
            1.0,
            &input_row,
            input_dim as i32,
            &weight,
            input_dim as i32,
            0.0,
            &output_row,
            output_dim as i32,
        )?;
    }
    Ok(())
}

fn bf16_linear(
    context: &CudaContext,
    input: &Tensor,
    weight: &Tensor,
    output: &Tensor,
    rows: usize,
    input_dim: usize,
    output_dim: usize,
) -> apxinf_core::Result<()> {
    gemm::write_ex(
        context,
        DType::BF16,
        CublasTranspose::None,
        CublasTranspose::Transpose,
        rows,
        output_dim,
        input_dim,
        1.0,
        &CudaBuffer::from_tensor(input).map_err(apxinf_core::Error::Cuda)?,
        input_dim as i32,
        &CudaBuffer::from_tensor(weight).map_err(apxinf_core::Error::Cuda)?,
        input_dim as i32,
        0.0,
        &CudaBuffer::from_tensor(output).map_err(apxinf_core::Error::Cuda)?,
        output_dim as i32,
    )
}

fn compare_all(
    rows: &Rows,
    tile: &Tile,
) -> Result<Vec<(String, usize)>, Box<dyn std::error::Error>> {
    let mut result = Vec::new();
    for (name, serial, candidate) in [
        (
            "input_normalized",
            &rows.input_normalized,
            &tile.input_normalized,
        ),
        ("qkv", &rows.qkv, &tile.qkv),
        ("z", &rows.z, &tile.z),
        ("ab", &rows.ab, &tile.ab),
        ("a", &rows.a, &tile.a),
        ("b", &rows.b, &tile.b),
        ("query", &rows.query, &tile.query),
        ("key", &rows.key, &tile.key),
        ("value", &rows.value, &tile.value),
        ("core", &rows.core, &tile.core),
        ("normalized", &rows.normalized, &tile.normalized),
        ("gdn_delta", &rows.output, &tile.output),
        ("mlp_input", &rows.mlp_input, &tile.mlp_input),
        ("mlp_gate_up", &rows.mlp_gate_up, &tile.mlp_gate_up),
        ("mlp_hidden", &rows.mlp_hidden, &tile.mlp_hidden),
        ("mlp_delta", &rows.mlp_delta, &tile.mlp_delta),
        (
            "next_normalized",
            &rows.next_normalized,
            &tile.next_normalized,
        ),
        ("final_residual", &rows.residual, &tile.residual),
    ] {
        result.push((name.into(), different_bf16_many(serial, candidate)?));
    }
    for (name, serial, candidate) in [("g", &rows.g, &tile.g), ("beta", &rows.beta, &tile.beta)] {
        result.push((name.into(), different_f32_many(serial, candidate)?));
    }
    result.push((
        "conv_state".into(),
        different_bf16(&rows.conv_state, &tile.conv_state)?,
    ));
    result.push((
        "recurrent_state".into(),
        different_f32(&rows.recurrent_state, &tile.recurrent_state)?,
    ));
    Ok(result)
}

fn different_bf16_many(
    rows: &[Tensor],
    tile: &Tensor,
) -> Result<usize, Box<dyn std::error::Error>> {
    let tile = transfers::to_cpu(tile)?;
    let tile = tile.as_bf16()?;
    let mut offset = 0;
    let mut different = 0;
    for row in rows {
        let row = transfers::to_cpu(row)?;
        let values = row.as_bf16()?;
        different += values
            .iter()
            .zip(&tile[offset..offset + values.len()])
            .filter(|(left, right)| left.to_bits() != right.to_bits())
            .count();
        offset += values.len();
    }
    Ok(different)
}

fn different_f32_many(rows: &[Tensor], tile: &Tensor) -> Result<usize, Box<dyn std::error::Error>> {
    let tile = transfers::to_cpu(tile)?;
    let tile = tile.as_f32()?;
    let mut offset = 0;
    let mut different = 0;
    for row in rows {
        let row = transfers::to_cpu(row)?;
        let values = row.as_f32()?;
        different += values
            .iter()
            .zip(&tile[offset..offset + values.len()])
            .filter(|(left, right)| left.to_bits() != right.to_bits())
            .count();
        offset += values.len();
    }
    Ok(different)
}

fn different_bf16(left: &Tensor, right: &Tensor) -> Result<usize, Box<dyn std::error::Error>> {
    let left = transfers::to_cpu(left)?;
    let right = transfers::to_cpu(right)?;
    Ok(left
        .as_bf16()?
        .iter()
        .zip(right.as_bf16()?)
        .filter(|(left, right)| left.to_bits() != right.to_bits())
        .count())
}

fn different_f32(left: &Tensor, right: &Tensor) -> Result<usize, Box<dyn std::error::Error>> {
    let left = transfers::to_cpu(left)?;
    let right = transfers::to_cpu(right)?;
    Ok(left
        .as_f32()?
        .iter()
        .zip(right.as_f32()?)
        .filter(|(left, right)| left.to_bits() != right.to_bits())
        .count())
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
