//! Qwen3.5 GDN M=8 stateful scan against eight serial decode launches.

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
use apxinf_cuda::tuning::KERNEL_BUILD_ID;
use apxinf_cuda::{transfers, CudaContext};
use apxinf_loader::safetensors::{self, CheckpointManifest};

const TOKENS: usize = 8;
const AB_WIDTH: usize = 2 * HEADS;
const VALUE_WIDTH: usize = HEADS * DIM;
const PREFIX: &str = "model.language_model.layers.1.linear_attn";
const RMS_EPSILON: f32 = 1.0e-6;
const WARMUPS: usize = 3;
const PAIRS: usize = 5;
const CALLS: usize = 3;

struct Parameters {
    conv: Tensor,
    a_log: Tensor,
    dt_bias: Tensor,
    norm: Tensor,
}

struct SerialWorkspace {
    projected_qkv: Vec<Tensor>,
    projected_ab: Vec<Tensor>,
    gate: Vec<Tensor>,
    a: Vec<Tensor>,
    b: Vec<Tensor>,
    query: Vec<Tensor>,
    key: Vec<Tensor>,
    value: Vec<Tensor>,
    g: Vec<Tensor>,
    beta: Vec<Tensor>,
    core: Vec<Tensor>,
    norm: Vec<Tensor>,
    conv_state: Tensor,
    recurrent_state: Tensor,
}

struct CandidateWorkspace {
    projected_qkv: Tensor,
    projected_ab: Tensor,
    gate: Tensor,
    a: Tensor,
    b: Tensor,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    g: Tensor,
    beta: Tensor,
    core: Tensor,
    norm: Tensor,
    conv_state: Tensor,
    recurrent_state: Tensor,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_dir = std::env::args()
        .nth(1)
        .ok_or("usage: qwen35_gdn_scan_prefill_probe MODEL_DIR")?;
    let manifest = safetensors::inspect_path(Path::new(&model_dir))?;
    let context = CudaContext::new(0)?;
    if context.caps().sm != 89 {
        return Err(format!(
            "GDN scan probe is frozen for SM89, got SM{}",
            context.caps().sm
        )
        .into());
    }
    let parameters = load_parameters(&manifest)?;
    let qkv_values = deterministic_bf16(TOKENS, CONV_DIM, 0.03125, 0.003_906_25);
    let ab_values = deterministic_bf16(TOKENS, AB_WIDTH, 0.125, 0.023_437_5);
    let gate_values = deterministic_bf16(TOKENS, VALUE_WIDTH, 0.25, 0.007_812_5);
    let serial = serial_workspace(&qkv_values, &ab_values, &gate_values)?;
    let candidate = candidate_workspace(&qkv_values, &ab_values, &gate_values)?;

    run_serial(&context, &parameters, &serial)?;
    run_candidate(&context, &parameters, &candidate)?;
    context.synchronize()?;
    let correctness = compare_all(&serial, &candidate)?;
    if correctness.iter().any(|(_, different)| *different != 0) {
        return Err(format!("M8 GDN scan differs from serial M1: {correctness:?}").into());
    }

    for _ in 0..WARMUPS {
        run_serial(&context, &parameters, &serial)?;
        run_candidate(&context, &parameters, &candidate)?;
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
                    run_candidate(&context, &parameters, &candidate)?;
                } else {
                    run_serial(&context, &parameters, &serial)?;
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
                "pair": pair,
                "order": if candidate_first { "BA" } else { "AB" },
                "order_index": order_index,
                "arm": if candidate_arm { "m8_scan" } else { "serial_m1" },
                "us_per_8_tokens": elapsed,
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
            "schema": "apxinf.qwen35.gdn_scan_prefill_probe.v1",
            "model_dir": model_dir,
            "layer": 1,
            "contract": {
                "tokens": TOKENS,
                "path": "fused conv4/prepare -> recurrent scan -> gated RMSNorm",
                "state_semantics": "one causal conv and recurrent state carried across all eight tokens",
                "oracle": "eight existing single-token production kernels",
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
                "different_values": correctness,
                "comparison": "BF16 and FP32 bitwise identity at every output and final state endpoint",
                "pass": true,
            },
            "timing": {
                "boundary": "all conv/prepare, recurrent, and gated-norm launches through stream synchronize",
                "warmups_per_arm": WARMUPS,
                "pairs": PAIRS,
                "calls_per_sample": CALLS,
                "records": records,
                "serial_raw_us": serial_samples,
                "candidate_raw_us": candidate_samples,
                "serial_median_us": median(&serial_samples),
                "candidate_median_us": median(&candidate_samples),
                "median_speedup": median(&speedups),
                "candidate_wins": wins,
                "candidate_tokens_per_second": TOKENS as f64 * 1.0e6 / median(&candidate_samples),
            },
            "evidence_level": "stateful-gdn-prefill-primitive",
            "model_promoted": false,
        }))?
    );
    Ok(())
}

fn load_parameters(
    manifest: &CheckpointManifest,
) -> Result<Parameters, Box<dyn std::error::Error>> {
    Ok(Parameters {
        conv: to_gpu(load_tensor(manifest, &format!("{PREFIX}.conv1d.weight"))?)?,
        a_log: to_gpu(load_tensor(manifest, &format!("{PREFIX}.A_log"))?)?,
        dt_bias: to_gpu(load_tensor(manifest, &format!("{PREFIX}.dt_bias"))?)?,
        norm: to_gpu(load_tensor(manifest, &format!("{PREFIX}.norm.weight"))?)?,
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

fn deterministic_bf16(rows: usize, columns: usize, scale: f32, token_step: f32) -> Vec<half::bf16> {
    (0..rows * columns)
        .map(|index| {
            let token = index / columns;
            let column = index % columns;
            let phase = column as f32 * 0.005_859_375 + token as f32 * token_step;
            half::bf16::from_f32((phase.sin() + 0.2 * phase.cos()) * scale)
        })
        .collect()
}

fn serial_workspace(
    qkv: &[half::bf16],
    ab: &[half::bf16],
    gate: &[half::bf16],
) -> Result<SerialWorkspace, Box<dyn std::error::Error>> {
    let mut projected_qkv = Vec::with_capacity(TOKENS);
    let mut projected_ab = Vec::with_capacity(TOKENS);
    let mut gates = Vec::with_capacity(TOKENS);
    let mut a = Vec::with_capacity(TOKENS);
    let mut b = Vec::with_capacity(TOKENS);
    let mut query = Vec::with_capacity(TOKENS);
    let mut key = Vec::with_capacity(TOKENS);
    let mut value = Vec::with_capacity(TOKENS);
    let mut g = Vec::with_capacity(TOKENS);
    let mut beta = Vec::with_capacity(TOKENS);
    let mut core = Vec::with_capacity(TOKENS);
    let mut norm = Vec::with_capacity(TOKENS);
    for token in 0..TOKENS {
        projected_qkv.push(to_gpu(Tensor::from_bf16(
            vec![CONV_DIM],
            &qkv[token * CONV_DIM..(token + 1) * CONV_DIM],
        )?)?);
        projected_ab.push(to_gpu(Tensor::from_bf16(
            vec![AB_WIDTH],
            &ab[token * AB_WIDTH..(token + 1) * AB_WIDTH],
        )?)?);
        gates.push(to_gpu(Tensor::from_bf16(
            vec![HEADS, DIM],
            &gate[token * VALUE_WIDTH..(token + 1) * VALUE_WIDTH],
        )?)?);
        a.push(gpu_zeros(&[HEADS], DType::BF16)?);
        b.push(gpu_zeros(&[HEADS], DType::BF16)?);
        query.push(gpu_zeros(&[HEADS, DIM], DType::BF16)?);
        key.push(gpu_zeros(&[HEADS, DIM], DType::BF16)?);
        value.push(gpu_zeros(&[HEADS, DIM], DType::BF16)?);
        g.push(gpu_zeros(&[HEADS], DType::F32)?);
        beta.push(gpu_zeros(&[HEADS], DType::F32)?);
        core.push(gpu_zeros(&[HEADS, DIM], DType::BF16)?);
        norm.push(gpu_zeros(&[HEADS, DIM], DType::BF16)?);
    }
    Ok(SerialWorkspace {
        projected_qkv,
        projected_ab,
        gate: gates,
        a,
        b,
        query,
        key,
        value,
        g,
        beta,
        core,
        norm,
        conv_state: gpu_zeros(&[CONV_DIM, CONV_KERNEL], DType::BF16)?,
        recurrent_state: gpu_zeros(&[HEADS, DIM, DIM], DType::F32)?,
    })
}

fn candidate_workspace(
    qkv: &[half::bf16],
    ab: &[half::bf16],
    gate: &[half::bf16],
) -> Result<CandidateWorkspace, Box<dyn std::error::Error>> {
    Ok(CandidateWorkspace {
        projected_qkv: to_gpu(Tensor::from_bf16(vec![TOKENS, CONV_DIM], qkv)?)?,
        projected_ab: to_gpu(Tensor::from_bf16(vec![TOKENS, AB_WIDTH], ab)?)?,
        gate: to_gpu(Tensor::from_bf16(vec![TOKENS, HEADS, DIM], gate)?)?,
        a: gpu_zeros(&[TOKENS, HEADS], DType::BF16)?,
        b: gpu_zeros(&[TOKENS, HEADS], DType::BF16)?,
        query: gpu_zeros(&[TOKENS, HEADS, DIM], DType::BF16)?,
        key: gpu_zeros(&[TOKENS, HEADS, DIM], DType::BF16)?,
        value: gpu_zeros(&[TOKENS, HEADS, DIM], DType::BF16)?,
        g: gpu_zeros(&[TOKENS, HEADS], DType::F32)?,
        beta: gpu_zeros(&[TOKENS, HEADS], DType::F32)?,
        core: gpu_zeros(&[TOKENS, HEADS, DIM], DType::BF16)?,
        norm: gpu_zeros(&[TOKENS, HEADS, DIM], DType::BF16)?,
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

fn run_serial(
    context: &CudaContext,
    parameters: &Parameters,
    workspace: &SerialWorkspace,
) -> apxinf_core::Result<()> {
    for token in 0..TOKENS {
        qwen35_conv4_prepare_write(
            context,
            &workspace.projected_qkv[token],
            &parameters.conv,
            &workspace.conv_state,
            &workspace.projected_ab[token],
            &parameters.a_log,
            &parameters.dt_bias,
            &workspace.a[token],
            &workspace.b[token],
            &workspace.query[token],
            &workspace.key[token],
            &workspace.value[token],
            &workspace.g[token],
            &workspace.beta[token],
        )?;
        qwen35_recurrent_write(
            context,
            &workspace.query[token],
            &workspace.key[token],
            &workspace.value[token],
            &workspace.g[token],
            &workspace.beta[token],
            &workspace.recurrent_state,
            &workspace.core[token],
        )?;
        qwen35_gated_rmsnorm_write(
            context,
            &workspace.core[token],
            &workspace.gate[token],
            &parameters.norm,
            &workspace.norm[token],
            RMS_EPSILON,
        )?;
    }
    Ok(())
}

fn run_candidate(
    context: &CudaContext,
    parameters: &Parameters,
    workspace: &CandidateWorkspace,
) -> apxinf_core::Result<()> {
    qwen35_conv4_prepare_m8_write(
        context,
        &workspace.projected_qkv,
        &parameters.conv,
        &workspace.conv_state,
        &workspace.projected_ab,
        &parameters.a_log,
        &parameters.dt_bias,
        &workspace.a,
        &workspace.b,
        &workspace.query,
        &workspace.key,
        &workspace.value,
        &workspace.g,
        &workspace.beta,
    )?;
    qwen35_recurrent_m8_write(
        context,
        &workspace.query,
        &workspace.key,
        &workspace.value,
        &workspace.g,
        &workspace.beta,
        &workspace.recurrent_state,
        &workspace.core,
    )?;
    qwen35_gated_rmsnorm_m8_write(
        context,
        &workspace.core,
        &workspace.gate,
        &parameters.norm,
        &workspace.norm,
        RMS_EPSILON,
    )
}

fn compare_all(
    serial: &SerialWorkspace,
    candidate: &CandidateWorkspace,
) -> Result<Vec<(String, usize)>, Box<dyn std::error::Error>> {
    let mut result = Vec::new();
    for (name, serial_tensors, candidate_tensor) in [
        ("a", &serial.a, &candidate.a),
        ("b", &serial.b, &candidate.b),
        ("query", &serial.query, &candidate.query),
        ("key", &serial.key, &candidate.key),
        ("value", &serial.value, &candidate.value),
        ("core", &serial.core, &candidate.core),
        ("norm", &serial.norm, &candidate.norm),
    ] {
        result.push((
            name.to_owned(),
            different_bf16_many(serial_tensors, candidate_tensor)?,
        ));
    }
    for (name, serial_tensors, candidate_tensor) in [
        ("g", &serial.g, &candidate.g),
        ("beta", &serial.beta, &candidate.beta),
    ] {
        result.push((
            name.to_owned(),
            different_f32_many(serial_tensors, candidate_tensor)?,
        ));
    }
    result.push((
        "conv_state".to_owned(),
        different_bf16(&serial.conv_state, &candidate.conv_state)?,
    ));
    result.push((
        "recurrent_state".to_owned(),
        different_f32(&serial.recurrent_state, &candidate.recurrent_state)?,
    ));
    Ok(result)
}

fn different_bf16_many(
    serial: &[Tensor],
    candidate: &Tensor,
) -> Result<usize, Box<dyn std::error::Error>> {
    let candidate_cpu = transfers::to_cpu(candidate)?;
    let candidate_values = candidate_cpu.as_bf16()?;
    let mut offset = 0usize;
    let mut different = 0usize;
    for tensor in serial {
        let serial_cpu = transfers::to_cpu(tensor)?;
        let serial_values = serial_cpu.as_bf16()?;
        different += serial_values
            .iter()
            .zip(&candidate_values[offset..offset + serial_values.len()])
            .filter(|(left, right)| left.to_bits() != right.to_bits())
            .count();
        offset += serial_values.len();
    }
    Ok(different)
}

fn different_f32_many(
    serial: &[Tensor],
    candidate: &Tensor,
) -> Result<usize, Box<dyn std::error::Error>> {
    let candidate_cpu = transfers::to_cpu(candidate)?;
    let candidate_values = candidate_cpu.as_f32()?;
    let mut offset = 0usize;
    let mut different = 0usize;
    for tensor in serial {
        let serial_cpu = transfers::to_cpu(tensor)?;
        let serial_values = serial_cpu.as_f32()?;
        different += serial_values
            .iter()
            .zip(&candidate_values[offset..offset + serial_values.len()])
            .filter(|(left, right)| left.to_bits() != right.to_bits())
            .count();
        offset += serial_values.len();
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
