//! Real Qwen3.5 layer-3 full-attention decode trajectory on SM89.
//!
//! The probe deliberately uses the checkpoint's compressed-tensors W4A16
//! projections and BF16 KV layout.  It validates producer/consumer endpoints
//! before reporting no-profiler module timing for fixed 1K/8K/32K KV cells.

use std::cmp::Ordering;
use std::path::Path;
use std::time::Instant;

use apxinf_core::{DType, Tensor};
use apxinf_cuda::kernels::gemm::{W4A16Layout, W4A16WeightView};
use apxinf_cuda::kernels::{attention, cache, gemm, qwen35_attention};
use apxinf_cuda::tuning::KERNEL_BUILD_ID;
use apxinf_cuda::{transfers, CudaBuffer, CudaContext, CudaDeviceAddress, HostMappedBuffer};
use apxinf_loader::safetensors::{self, CheckpointManifest};

const HIDDEN: usize = 5120;
const Q_HEADS: usize = 24;
const KV_HEADS: usize = 4;
const HEAD_DIM: usize = 256;
const GQA_RATIO: usize = Q_HEADS / KV_HEADS;
const WIDTH: usize = Q_HEADS * HEAD_DIM;
const KV_WIDTH: usize = KV_HEADS * HEAD_DIM;
const Q_PROJECTION: usize = 2 * WIDTH;
const MAX_SEQ_LEN: usize = 32768;
const PREFIX: &str = "model.language_model.layers.3.self_attn";
const WARMUPS: usize = 3;
const BLOCKS: usize = 10;
const CALLS: usize = 3;
const ATTENTION_SCALE: f32 = 1.0 / 16.0;

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

struct Prepared {
    query: Vec<half::bf16>,
    key: Vec<half::bf16>,
    value: Vec<half::bf16>,
    gate: Vec<half::bf16>,
}

#[allow(clippy::too_many_arguments)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_dir = std::env::args()
        .nth(1)
        .ok_or("usage: qwen35_attention_layer_probe MODEL_DIR")?;
    let kv_cells = requested_cells()?;
    let (implementation, configured_split_count, automatic) = requested_implementation()?;
    let manifest = safetensors::inspect_path(Path::new(&model_dir))?;
    let q_weight = load_w4(&manifest, &format!("{PREFIX}.q_proj"))?;
    let k_weight = load_w4(&manifest, &format!("{PREFIX}.k_proj"))?;
    let v_weight = load_w4(&manifest, &format!("{PREFIX}.v_proj"))?;
    let o_weight = load_w4(&manifest, &format!("{PREFIX}.o_proj"))?;
    require_w4_shape("q_proj", &q_weight, HIDDEN, Q_PROJECTION)?;
    require_w4_shape("k_proj", &k_weight, HIDDEN, KV_WIDTH)?;
    require_w4_shape("v_proj", &v_weight, HIDDEN, KV_WIDTH)?;
    require_w4_shape("o_proj", &o_weight, WIDTH, HIDDEN)?;
    let q_norm_cpu = load_tensor(&manifest, &format!("{PREFIX}.q_norm.weight"))?;
    let k_norm_cpu = load_tensor(&manifest, &format!("{PREFIX}.k_norm.weight"))?;
    require_bf16_shape("q_norm", &q_norm_cpu, &[HEAD_DIM])?;
    require_bf16_shape("k_norm", &k_norm_cpu, &[HEAD_DIM])?;

    let context = CudaContext::new(0)?;
    if context.caps().sm != 89 {
        return Err(format!("probe is frozen for SM89, got SM{}", context.caps().sm).into());
    }

    let hidden_values = deterministic_hidden();
    let hidden_cpu = Tensor::from_bf16(vec![1, HIDDEN], &hidden_values)?;
    let hidden = transfers::to_cuda(&hidden_cpu, 0)?;
    let q_norm = transfers::to_cuda(&q_norm_cpu, 0)?;
    let k_norm = transfers::to_cuda(&k_norm_cpu, 0)?;

    let q_projection = gpu_zeros(&[1, Q_PROJECTION])?;
    let k_projection = gpu_zeros(&[1, KV_WIDTH])?;
    let v_projection = gpu_zeros(&[1, KV_WIDTH])?;
    let query = gpu_zeros(&[Q_HEADS, HEAD_DIM])?;
    let key = gpu_zeros(&[KV_HEADS, HEAD_DIM])?;
    let value = gpu_zeros(&[KV_HEADS, HEAD_DIM])?;
    let gate = gpu_zeros(&[Q_HEADS, HEAD_DIM])?;
    let attention_output = gpu_zeros(&[Q_HEADS, HEAD_DIM])?;
    let gated_output = gpu_zeros(&[Q_HEADS, HEAD_DIM])?;
    let gated_flat = gated_output.reshape(vec![1, WIDTH])?;
    let output = gpu_zeros(&[1, HIDDEN])?;

    let (mut key_cache_values, mut value_cache_values) = deterministic_caches();
    let key_cache = transfers::to_cuda(
        &Tensor::from_bf16(vec![KV_HEADS, MAX_SEQ_LEN, HEAD_DIM], &key_cache_values)?,
        0,
    )?;
    let value_cache = transfers::to_cuda(
        &Tensor::from_bf16(vec![KV_HEADS, MAX_SEQ_LEN, HEAD_DIM], &value_cache_values)?,
        0,
    )?;
    let position = HostMappedBuffer::alloc(3 * 4, 0)?;
    let attention_workspace = qwen35_attention::SplitCtaWorkspace::new(&context)?;

    let expected_q_projection = cpu_w4(&hidden_values, &q_weight)?;
    let expected_k_projection = cpu_w4(&hidden_values, &k_weight)?;
    let expected_v_projection = cpu_w4(&hidden_values, &v_weight)?;
    let mut results = Vec::new();

    for &kv_len in &kv_cells {
        let split_count = if automatic {
            qwen35_attention::split_cta_candidate_for_bucket(kv_len)
        } else {
            configured_split_count
        };
        let cache_position = kv_len - 1;
        position.write_u32s(&[cache_position as u32; 3])?;
        let expected_prepared = cpu_prepare(
            &expected_q_projection,
            &expected_k_projection,
            &expected_v_projection,
            q_norm_cpu.as_bf16()?,
            k_norm_cpu.as_bf16()?,
            cache_position,
        );

        run(
            &context,
            &hidden,
            &q_weight,
            &k_weight,
            &v_weight,
            &o_weight,
            &q_projection,
            &k_projection,
            &v_projection,
            &q_norm,
            &k_norm,
            &query,
            &key,
            &value,
            &gate,
            &key_cache,
            &value_cache,
            &attention_output,
            &gated_output,
            &gated_flat,
            &output,
            &attention_workspace,
            split_count,
            kv_len,
            position.address(),
            false,
        )?;
        context.synchronize()?;

        // The cache's exact contract is producer-byte preservation.  Q/K
        // normalization reductions may legitimately round one BF16 value
        // differently from the scalar CPU oracle, so seed the cache oracle
        // from the actual GPU producer and compare that producer numerically
        // against the CPU endpoint below.
        let produced_key = transfers::to_cpu(&key)?.as_bf16()?.to_vec();
        let produced_value = transfers::to_cpu(&value)?.as_bf16()?.to_vec();
        write_cache_slot(&mut key_cache_values, cache_position, &produced_key);
        write_cache_slot(&mut value_cache_values, cache_position, &produced_value);
        let expected_attention = cpu_attention(
            &expected_prepared.query,
            &key_cache_values,
            &value_cache_values,
            kv_len,
        );
        let expected_gated = cpu_gate(&expected_attention, &expected_prepared.gate);
        let expected_output = cpu_w4(&expected_gated, &o_weight)?;

        let endpoints = validate_endpoints(
            &q_projection,
            &k_projection,
            &v_projection,
            &query,
            &key,
            &value,
            &gate,
            &attention_output,
            &gated_output,
            &output,
            &expected_q_projection,
            &expected_k_projection,
            &expected_v_projection,
            &expected_prepared,
            &expected_attention,
            &expected_gated,
            &expected_output,
        )?;
        let cache_contract = validate_cache(
            &key_cache,
            &value_cache,
            &key_cache_values,
            &value_cache_values,
            cache_position,
        )?;

        let profile_cell = std::env::var("APXINF_PROFILE_KV")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            == Some(kv_len);
        if profile_cell {
            apxinf_cuda::profiler::start()?;
            run(
                &context,
                &hidden,
                &q_weight,
                &k_weight,
                &v_weight,
                &o_weight,
                &q_projection,
                &k_projection,
                &v_projection,
                &q_norm,
                &k_norm,
                &query,
                &key,
                &value,
                &gate,
                &key_cache,
                &value_cache,
                &attention_output,
                &gated_output,
                &gated_flat,
                &output,
                &attention_workspace,
                split_count,
                kv_len,
                position.address(),
                true,
            )?;
            context.synchronize()?;
            apxinf_cuda::profiler::stop()?;
        }

        for _ in 0..WARMUPS {
            run(
                &context,
                &hidden,
                &q_weight,
                &k_weight,
                &v_weight,
                &o_weight,
                &q_projection,
                &k_projection,
                &v_projection,
                &q_norm,
                &k_norm,
                &query,
                &key,
                &value,
                &gate,
                &key_cache,
                &value_cache,
                &attention_output,
                &gated_output,
                &gated_flat,
                &output,
                &attention_workspace,
                split_count,
                kv_len,
                position.address(),
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
                    &q_weight,
                    &k_weight,
                    &v_weight,
                    &o_weight,
                    &q_projection,
                    &k_projection,
                    &v_projection,
                    &q_norm,
                    &k_norm,
                    &query,
                    &key,
                    &value,
                    &gate,
                    &key_cache,
                    &value_cache,
                    &attention_output,
                    &gated_output,
                    &gated_flat,
                    &output,
                    &attention_workspace,
                    split_count,
                    kv_len,
                    position.address(),
                    false,
                )?;
            }
            context.synchronize()?;
            samples.push(start.elapsed().as_secs_f64() * 1.0e6 / CALLS as f64);
        }
        let (median, mean, deviation) = summarize(&samples);
        results.push(serde_json::json!({
            "kv_len":kv_len,
            "cache_position":cache_position,
            "attention_path":if split_count.is_some(){"split_cta"}else{"incumbent"},
            "split_cta_count":split_count,
            "correctness":{"endpoints":endpoints,"cache":cache_contract,"pass":true},
            "timing":{
                "boundary":"q/k/v W4 + Q/K norm/partial RoPE + KV append + BF16 flash attention + output gate + o W4 to stream synchronize; no profiler",
                "warmups":WARMUPS,"blocks":BLOCKS,"calls_per_block":CALLS,
                "raw_us_per_layer":samples,"median_us":median,"mean_us":mean,
                "standard_deviation_us":deviation,"cv":deviation/mean,
            }
        }));
    }

    let hidden_after = transfers::to_cpu(&hidden)?;
    let hidden_immutable = hidden_after.as_bf16()? == hidden_values.as_slice();
    if !hidden_immutable {
        return Err("attention layer modified its hidden-state input".into());
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "schema":"apxinf.qwen35.attention_layer_probe.v1",
            "model_dir":model_dir,"layer":3,"device":"sm89",
            "contract":{
                "batch":1,"new_tokens":1,"hidden":HIDDEN,"q_heads":Q_HEADS,
                "kv_heads":KV_HEADS,"gqa_ratio":GQA_RATIO,"head_dim":HEAD_DIM,
                "partial_rope_dim":64,"rope_theta":10000000.0,"kv_dtype":"bf16",
                "max_seq_len":MAX_SEQ_LEN,"quantization":"W4A16 group-32 asymmetric",
                "attention_implementation":implementation,
                "configured_split_cta_count":configured_split_count,
                "auto_min_kv_bucket":if automatic {
                    Some(qwen35_attention::SPLIT_CTA_CANDIDATE_MIN_KV_BUCKET)
                } else { None },
            },
            "kernel_build_id":KERNEL_BUILD_ID,
            "input_immutable":hidden_immutable,
            "cells":results,
            "evidence_level":"layer-module",
            "model_promoted":false,
        }))?
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run(
    context: &CudaContext,
    hidden: &Tensor,
    q_weight: &W4,
    k_weight: &W4,
    v_weight: &W4,
    o_weight: &W4,
    q_projection: &Tensor,
    k_projection: &Tensor,
    v_projection: &Tensor,
    q_norm: &Tensor,
    k_norm: &Tensor,
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    gate: &Tensor,
    key_cache: &Tensor,
    value_cache: &Tensor,
    attention_output: &Tensor,
    gated_output: &Tensor,
    gated_flat: &Tensor,
    output: &Tensor,
    attention_workspace: &qwen35_attention::SplitCtaWorkspace,
    split_count: Option<usize>,
    kv_len: usize,
    position: CudaDeviceAddress,
    profile: bool,
) -> apxinf_core::Result<()> {
    let _complete = profile.then(|| apxinf_cuda::nvtx::range("qwen35.attention_layer.complete"));
    {
        let _range = profile.then(|| apxinf_cuda::nvtx::range("qwen35.attention.q_w4"));
        gemm::w4a16_write(context, hidden, q_weight.view(), q_projection)?;
    }
    {
        let _range = profile.then(|| apxinf_cuda::nvtx::range("qwen35.attention.k_w4"));
        gemm::w4a16_write(context, hidden, k_weight.view(), k_projection)?;
    }
    {
        let _range = profile.then(|| apxinf_cuda::nvtx::range("qwen35.attention.v_w4"));
        gemm::w4a16_write(context, hidden, v_weight.view(), v_projection)?;
    }
    {
        let _range = profile.then(|| apxinf_cuda::nvtx::range("qwen35.attention.prepare"));
        qwen35_attention::prepare_write(
            context,
            q_projection,
            k_projection,
            v_projection,
            q_norm,
            k_norm,
            query,
            key,
            value,
            gate,
            position,
        )?;
    }
    let key_buffer = CudaBuffer::from_tensor(key).map_err(apxinf_core::Error::Cuda)?;
    let value_buffer = CudaBuffer::from_tensor(value).map_err(apxinf_core::Error::Cuda)?;
    let key_cache_buffer = CudaBuffer::from_tensor(key_cache).map_err(apxinf_core::Error::Cuda)?;
    let value_cache_buffer =
        CudaBuffer::from_tensor(value_cache).map_err(apxinf_core::Error::Cuda)?;
    {
        let _range = profile.then(|| apxinf_cuda::nvtx::range("qwen35.attention.kv_append"));
        cache::append_at(
            context,
            DType::BF16,
            &key_cache_buffer,
            &key_buffer,
            KV_HEADS,
            HEAD_DIM,
            MAX_SEQ_LEN,
            position,
        )?;
        cache::append_at(
            context,
            DType::BF16,
            &value_cache_buffer,
            &value_buffer,
            KV_HEADS,
            HEAD_DIM,
            MAX_SEQ_LEN,
            position,
        )?;
    }
    {
        let range_name = if split_count.is_some() {
            "qwen35.attention.flash_split_cta"
        } else {
            "qwen35.attention.flash"
        };
        let _range = profile.then(|| apxinf_cuda::nvtx::range(range_name));
        if let Some(split_count) = split_count {
            qwen35_attention::flash_split_cta_write(
                context,
                query,
                &key_cache_buffer,
                &value_cache_buffer,
                attention_output,
                attention_workspace,
                split_count,
                kv_len,
                MAX_SEQ_LEN,
                ATTENTION_SCALE,
                position,
            )?;
        } else {
            attention::flash_bf16_into(
                context,
                &CudaBuffer::from_tensor(query).map_err(apxinf_core::Error::Cuda)?,
                &key_cache_buffer,
                &value_cache_buffer,
                &CudaBuffer::from_tensor(attention_output).map_err(apxinf_core::Error::Cuda)?,
                Q_HEADS,
                KV_HEADS,
                HEAD_DIM,
                kv_len,
                MAX_SEQ_LEN,
                ATTENTION_SCALE,
                position,
            )?;
        }
    }
    {
        let _range = profile.then(|| apxinf_cuda::nvtx::range("qwen35.attention.gate"));
        qwen35_attention::gate_write(context, attention_output, gate, gated_output)?;
    }
    {
        let _range = profile.then(|| apxinf_cuda::nvtx::range("qwen35.attention.o_w4"));
        gemm::w4a16_write(context, gated_flat, o_weight.view(), output)
    }
}

fn requested_cells() -> Result<Vec<usize>, Box<dyn std::error::Error>> {
    let raw = std::env::var("APXINF_KV_LENS").unwrap_or_else(|_| "1024,8192,32768".into());
    let mut cells = raw
        .split(',')
        .map(|value| value.trim().parse::<usize>())
        .collect::<Result<Vec<_>, _>>()?;
    cells.sort_unstable();
    cells.dedup();
    if cells.is_empty()
        || cells
            .iter()
            .any(|value| *value == 0 || *value > MAX_SEQ_LEN)
    {
        return Err(format!("APXINF_KV_LENS must be within 1..={MAX_SEQ_LEN}").into());
    }
    Ok(cells)
}

fn requested_implementation(
) -> Result<(&'static str, Option<usize>, bool), Box<dyn std::error::Error>> {
    let value = std::env::var("APXINF_ATTN_IMPL").unwrap_or_else(|_| "incumbent".into());
    match value.as_str() {
        "incumbent" => Ok(("incumbent", None, false)),
        "split2" => Ok(("split_cta", Some(2), false)),
        "split4" => Ok(("split_cta", Some(4), false)),
        "split8" => Ok(("split_cta", Some(8), false)),
        "split16" => Ok(("split_cta", Some(16), false)),
        "auto" => Ok(("auto", None, true)),
        _ => Err(format!(
            "APXINF_ATTN_IMPL must be incumbent, split2, split4, split8, split16, or auto; got {value}"
        )
        .into()),
    }
}

fn deterministic_hidden() -> Vec<half::bf16> {
    (0..HIDDEN)
        .map(|index| {
            let phase = index as f32 * 0.005_859_375 + (index % 31) as f32 * 0.001_953_125;
            half::bf16::from_f32((phase.sin() + 0.25 * phase.cos()) * 0.25)
        })
        .collect()
}

fn deterministic_caches() -> (Vec<half::bf16>, Vec<half::bf16>) {
    let count = KV_HEADS * MAX_SEQ_LEN * HEAD_DIM;
    let mut key = vec![half::bf16::ZERO; count];
    let mut value = vec![half::bf16::ZERO; count];
    for head in 0..KV_HEADS {
        for token in 0..MAX_SEQ_LEN {
            let base = (head * MAX_SEQ_LEN + token) * HEAD_DIM;
            for dimension in 0..HEAD_DIM {
                let key_bits = (token * 17 + dimension * 13 + head * 7) & 255;
                let value_bits = (token * 29 + dimension * 11 + head * 19) & 255;
                key[base + dimension] = half::bf16::from_f32((key_bits as f32 - 128.0) / 1024.0);
                value[base + dimension] =
                    half::bf16::from_f32(0.1 + (value_bits as f32 - 128.0) / 2048.0);
            }
        }
    }
    (key, value)
}

fn cpu_prepare(
    q_projection: &[half::bf16],
    k_projection: &[half::bf16],
    v_projection: &[half::bf16],
    q_norm: &[half::bf16],
    k_norm: &[half::bf16],
    position: usize,
) -> Prepared {
    let mut query = vec![half::bf16::ZERO; WIDTH];
    let mut key = vec![half::bf16::ZERO; KV_WIDTH];
    let mut gate = vec![half::bf16::ZERO; WIDTH];
    for head in 0..Q_HEADS {
        let source = &q_projection[head * 2 * HEAD_DIM..(head + 1) * 2 * HEAD_DIM];
        let prepared = cpu_norm_rope(&source[..HEAD_DIM], q_norm, position);
        query[head * HEAD_DIM..(head + 1) * HEAD_DIM].copy_from_slice(&prepared);
        gate[head * HEAD_DIM..(head + 1) * HEAD_DIM]
            .copy_from_slice(&source[HEAD_DIM..2 * HEAD_DIM]);
    }
    for head in 0..KV_HEADS {
        let source = &k_projection[head * HEAD_DIM..(head + 1) * HEAD_DIM];
        let prepared = cpu_norm_rope(source, k_norm, position);
        key[head * HEAD_DIM..(head + 1) * HEAD_DIM].copy_from_slice(&prepared);
    }
    Prepared {
        query,
        key,
        value: v_projection.to_vec(),
        gate,
    }
}

fn cpu_norm_rope(source: &[half::bf16], weight: &[half::bf16], position: usize) -> Vec<half::bf16> {
    let square_sum = source
        .iter()
        .map(|value| value.to_f32().powi(2))
        .sum::<f32>();
    let inverse = (square_sum / HEAD_DIM as f32 + 1.0e-6).sqrt().recip();
    let normalized = source
        .iter()
        .zip(weight)
        .map(|(value, gamma)| value.to_f32() * inverse * (1.0 + gamma.to_f32()))
        .collect::<Vec<_>>();
    (0..HEAD_DIM)
        .map(|dimension| {
            let mut prepared = normalized[dimension];
            if dimension < 64 {
                let frequency = dimension & 31;
                let angle = position as f32 * 10_000_000.0_f32.powf(-2.0 * frequency as f32 / 64.0);
                let (sine, cosine) = angle.sin_cos();
                let partner = if dimension < 32 {
                    dimension + 32
                } else {
                    dimension - 32
                };
                let rotated = if dimension < 32 {
                    -normalized[partner]
                } else {
                    normalized[partner]
                };
                prepared = prepared * cosine + rotated * sine;
            }
            half::bf16::from_f32(prepared)
        })
        .collect()
}

fn cpu_attention(
    query: &[half::bf16],
    key_cache: &[half::bf16],
    value_cache: &[half::bf16],
    kv_len: usize,
) -> Vec<half::bf16> {
    let mut output = vec![half::bf16::ZERO; WIDTH];
    let mut scores = vec![0.0_f32; kv_len];
    let mut accumulator = vec![0.0_f32; HEAD_DIM];
    for q_head in 0..Q_HEADS {
        let kv_head = q_head / GQA_RATIO;
        let query_row = &query[q_head * HEAD_DIM..(q_head + 1) * HEAD_DIM];
        let cache_base = kv_head * MAX_SEQ_LEN * HEAD_DIM;
        let mut maximum = f32::NEG_INFINITY;
        for token in 0..kv_len {
            let key_row =
                &key_cache[cache_base + token * HEAD_DIM..cache_base + (token + 1) * HEAD_DIM];
            let mut dot = 0.0_f32;
            for dimension in 0..HEAD_DIM {
                dot += query_row[dimension].to_f32() * key_row[dimension].to_f32();
            }
            scores[token] = dot * ATTENTION_SCALE;
            maximum = maximum.max(scores[token]);
        }
        accumulator.fill(0.0);
        let mut denominator = 0.0_f32;
        for token in 0..kv_len {
            let probability = (scores[token] - maximum).exp();
            denominator += probability;
            let value_row =
                &value_cache[cache_base + token * HEAD_DIM..cache_base + (token + 1) * HEAD_DIM];
            for dimension in 0..HEAD_DIM {
                accumulator[dimension] += probability * value_row[dimension].to_f32();
            }
        }
        let inverse = denominator.recip();
        for dimension in 0..HEAD_DIM {
            output[q_head * HEAD_DIM + dimension] =
                half::bf16::from_f32(accumulator[dimension] * inverse);
        }
    }
    output
}

fn cpu_gate(input: &[half::bf16], gate: &[half::bf16]) -> Vec<half::bf16> {
    input
        .iter()
        .zip(gate)
        .map(|(value, gate)| {
            let gate = gate.to_f32();
            half::bf16::from_f32(value.to_f32() / (1.0 + (-gate).exp()))
        })
        .collect()
}

fn write_cache_slot(cache: &mut [half::bf16], position: usize, row: &[half::bf16]) {
    for head in 0..KV_HEADS {
        let destination = (head * MAX_SEQ_LEN + position) * HEAD_DIM;
        cache[destination..destination + HEAD_DIM]
            .copy_from_slice(&row[head * HEAD_DIM..(head + 1) * HEAD_DIM]);
    }
}

#[allow(clippy::too_many_arguments)]
fn validate_endpoints(
    q_projection: &Tensor,
    k_projection: &Tensor,
    v_projection: &Tensor,
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    gate: &Tensor,
    attention: &Tensor,
    gated: &Tensor,
    output: &Tensor,
    expected_q_projection: &[half::bf16],
    expected_k_projection: &[half::bf16],
    expected_v_projection: &[half::bf16],
    expected_prepared: &Prepared,
    expected_attention: &[half::bf16],
    expected_gated: &[half::bf16],
    expected_output: &[half::bf16],
) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
    let definitions = [
        ("q_projection", q_projection, expected_q_projection, true),
        ("k_projection", k_projection, expected_k_projection, true),
        ("v_projection", v_projection, expected_v_projection, true),
        (
            "prepared_query",
            query,
            expected_prepared.query.as_slice(),
            true,
        ),
        ("prepared_key", key, expected_prepared.key.as_slice(), true),
        (
            "prepared_value",
            value,
            expected_prepared.value.as_slice(),
            true,
        ),
        ("output_gate", gate, expected_prepared.gate.as_slice(), true),
        ("attention_output", attention, expected_attention, false),
        ("gated_output", gated, expected_gated, false),
        ("output_projection", output, expected_output, false),
    ];
    let mut values = serde_json::Map::new();
    for (name, actual, expected, strict) in definitions {
        let actual = transfers::to_cpu(actual)?.to_f32_vec()?;
        let expected = bf16_f32(expected);
        let metric = metrics(&actual, &expected)?;
        let pass = if strict {
            metric.0 >= 0.999 && metric.1 <= 0.02
        } else {
            metric.0 >= 0.995 && metric.1 <= 0.05
        };
        if !pass {
            return Err(format!("attention endpoint {name} failed: {metric:?}").into());
        }
        values.insert(name.into(), metric_json(metric, pass));
    }
    Ok(serde_json::Value::Object(values))
}

fn validate_cache(
    key_cache: &Tensor,
    value_cache: &Tensor,
    expected_key: &[half::bf16],
    expected_value: &[half::bf16],
    position: usize,
) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
    let actual_key = transfers::to_cpu(key_cache)?;
    let actual_value = transfers::to_cpu(value_cache)?;
    let actual_key = actual_key.as_bf16()?;
    let actual_value = actual_value.as_bf16()?;
    let mut slot_values = 0usize;
    let mut sentinel_values = 0usize;
    for head in 0..KV_HEADS {
        let slot = (head * MAX_SEQ_LEN + position) * HEAD_DIM;
        for dimension in 0..HEAD_DIM {
            if actual_key[slot + dimension].to_bits() != expected_key[slot + dimension].to_bits()
                || actual_value[slot + dimension].to_bits()
                    != expected_value[slot + dimension].to_bits()
            {
                return Err(format!("KV append mismatch at head={head}, dim={dimension}").into());
            }
            slot_values += 2;
        }
        for token in [0, position / 2, position.saturating_sub(1), MAX_SEQ_LEN - 1] {
            if token == position {
                continue;
            }
            for dimension in [0, 63, 127, 191, 255] {
                let index = (head * MAX_SEQ_LEN + token) * HEAD_DIM + dimension;
                if actual_key[index].to_bits() != expected_key[index].to_bits()
                    || actual_value[index].to_bits() != expected_value[index].to_bits()
                {
                    return Err(format!(
                        "KV sentinel changed at head={head}, token={token}, dim={dimension}"
                    )
                    .into());
                }
                sentinel_values += 2;
            }
        }
    }
    Ok(serde_json::json!({
        "slot_exact":true,"slot_values_checked":slot_values,
        "sentinels_exact":true,"sentinel_values_checked":sentinel_values,
    }))
}

fn load_w4(manifest: &CheckpointManifest, base: &str) -> Result<W4, Box<dyn std::error::Error>> {
    let packed_cpu = load_tensor(manifest, &format!("{base}.weight_packed"))?;
    let scales_cpu = load_tensor(manifest, &format!("{base}.weight_scale"))?;
    let zero_cpu = load_tensor(manifest, &format!("{base}.weight_zero_point"))?;
    let shape = safetensors::read_small_i64(
        manifest
            .tensor(&format!("{base}.weight_shape"))
            .ok_or_else(|| format!("missing {base}.weight_shape"))?,
    )?;
    if shape.len() != 2 || shape.iter().any(|dimension| *dimension <= 0) {
        return Err(format!("invalid W4 logical shape for {base}: {shape:?}").into());
    }
    Ok(W4 {
        packed_gpu: transfers::to_cuda(&packed_cpu, 0)?,
        scales_gpu: transfers::to_cuda(&scales_cpu, 0)?,
        zero_gpu: transfers::to_cuda(&zero_cpu, 0)?,
        packed_cpu,
        scales_cpu,
        zero_cpu,
        input: shape[1] as usize,
        output: shape[0] as usize,
    })
}

fn load_tensor(
    manifest: &CheckpointManifest,
    name: &str,
) -> Result<Tensor, Box<dyn std::error::Error>> {
    Ok(safetensors::load_manifest_tensor(
        manifest
            .tensor(name)
            .ok_or_else(|| format!("missing {name}"))?,
    )?)
}

fn require_w4_shape(
    name: &str,
    weight: &W4,
    input: usize,
    output: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    if weight.input != input || weight.output != output {
        return Err(format!(
            "{name} logical shape is [{},{}], expected [{output},{input}]",
            weight.output, weight.input
        )
        .into());
    }
    Ok(())
}

fn require_bf16_shape(
    name: &str,
    tensor: &Tensor,
    shape: &[usize],
) -> Result<(), Box<dyn std::error::Error>> {
    if tensor.dtype() != DType::BF16 || tensor.shape().dims() != shape {
        return Err(format!(
            "{name} must be BF16 {shape:?}, got {} {:?}",
            tensor.dtype(),
            tensor.shape().dims()
        )
        .into());
    }
    Ok(())
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
            let zero_point = (((zero_word >> ((row & 7) * 4)) & 15) as i32) - 8;
            for index in 0..8 {
                let quantized = (((word >> (index * 4)) & 15) as i32) - 8;
                sum = input[packed_col * 8 + index]
                    .to_f32()
                    .mul_add((quantized - zero_point) as f32 * scale, sum);
            }
        }
        output[row] = half::bf16::from_f32(sum);
    }
    Ok(output)
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
    for (&actual, &expected) in actual.iter().zip(expected) {
        if !actual.is_finite() || !expected.is_finite() {
            return Err("non-finite".into());
        }
        let (actual, expected) = (f64::from(actual), f64::from(expected));
        dot += actual * expected;
        aa += actual * actual;
        ee += expected * expected;
        error += (actual - expected).powi(2);
        max = max.max((actual - expected).abs());
    }
    if aa == 0.0 || ee == 0.0 {
        return Err("zero-norm endpoint".into());
    }
    Ok((dot / (aa.sqrt() * ee.sqrt()), (error / ee).sqrt(), max))
}

fn metric_json(value: (f64, f64, f64), pass: bool) -> serde_json::Value {
    serde_json::json!({
        "cosine":value.0,"relative_l2":value.1,"max_abs":value.2,"pass":pass
    })
}

fn summarize(samples: &[f64]) -> (f64, f64, f64) {
    let mut sorted = samples.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    let median = (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) * 0.5;
    let mean = samples.iter().sum::<f64>() / samples.len() as f64;
    let deviation = (samples
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / samples.len() as f64)
        .sqrt();
    (median, mean, deviation)
}
