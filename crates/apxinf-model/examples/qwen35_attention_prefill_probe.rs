//! Real layer-3 Qwen3.5 full-attention M=8 causal prefill tile on SM89.

use std::cmp::Ordering;
use std::path::Path;
use std::time::Instant;

use apxinf_core::{DType, Tensor};
use apxinf_cuda::kernels::gemm::{W4A16Layout, W4A16WeightView};
use apxinf_cuda::kernels::{attention, cache, gemm, qwen35_attention};
use apxinf_cuda::tuning::KERNEL_BUILD_ID;
use apxinf_cuda::{transfers, CudaBuffer, CudaContext, HostMappedBuffer};
use apxinf_loader::safetensors::{self, CheckpointManifest};

const TOKENS: usize = 8;
const HIDDEN: usize = 5120;
const Q_HEADS: usize = 24;
const KV_HEADS: usize = 4;
const HEAD_DIM: usize = 256;
const WIDTH: usize = Q_HEADS * HEAD_DIM;
const KV_WIDTH: usize = KV_HEADS * HEAD_DIM;
const Q_PROJECTION: usize = 2 * WIDTH;
const MAX_SEQ_LEN: usize = 32768;
const PREFIX: &str = "model.language_model.layers.3.self_attn";
const ATTENTION_SCALE: f32 = 1.0 / 16.0;
const WARMUPS: usize = 3;
const PAIRS: usize = 5;

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

struct Weights {
    q: W4,
    k: W4,
    v: W4,
    o: W4,
    q_norm: Tensor,
    k_norm: Tensor,
}

struct Rows {
    input: Vec<Tensor>,
    q_projection: Vec<Tensor>,
    k_projection: Vec<Tensor>,
    v_projection: Vec<Tensor>,
    query: Vec<Tensor>,
    key: Vec<Tensor>,
    value: Vec<Tensor>,
    gate: Vec<Tensor>,
    attended: Vec<Tensor>,
    gated: Vec<Tensor>,
    output: Vec<Tensor>,
    key_cache: Tensor,
    value_cache: Tensor,
}

struct Tile {
    input: Tensor,
    q_projection: Tensor,
    k_projection: Tensor,
    v_projection: Tensor,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    gate: Tensor,
    attended: Tensor,
    gated: Tensor,
    output: Tensor,
    key_cache: Tensor,
    value_cache: Tensor,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_dir = std::env::args()
        .nth(1)
        .ok_or("usage: qwen35_attention_prefill_probe MODEL_DIR")?;
    let start = std::env::var("APXINF_PREFILL_START")
        .unwrap_or_else(|_| "1024".into())
        .parse::<usize>()?;
    if start + TOKENS > MAX_SEQ_LEN {
        return Err(format!("APXINF_PREFILL_START must be <= {}", MAX_SEQ_LEN - TOKENS).into());
    }
    let manifest = safetensors::inspect_path(Path::new(&model_dir))?;
    let context = CudaContext::new(0)?;
    if context.caps().sm != 89 {
        return Err(format!(
            "attention prefill probe is frozen for SM89, got SM{}",
            context.caps().sm
        )
        .into());
    }
    let weights = load_weights(&manifest)?;
    let input_values = deterministic_input();
    let serial = serial_workspace(&input_values)?;
    let candidate = candidate_workspace(&input_values)?;
    let cache_positions = HostMappedBuffer::alloc(TOKENS * 4, 0)?;
    let rope_positions = HostMappedBuffer::alloc(TOKENS * 3 * 4, 0)?;
    let attention_workspace = qwen35_attention::SplitCtaWorkspace::new(&context)?;
    let positions = (start..start + TOKENS)
        .map(|position| position as u32)
        .collect::<Vec<_>>();
    cache_positions.write_u32s(&positions)?;
    rope_positions.write_u32s(
        &positions
            .iter()
            .flat_map(|position| [*position, *position, *position])
            .collect::<Vec<_>>(),
    )?;

    run_serial(
        &context,
        &weights,
        &serial,
        &cache_positions,
        &rope_positions,
        start,
        &attention_workspace,
    )?;
    run_candidate(
        &context,
        &weights,
        &candidate,
        &cache_positions,
        &rope_positions,
        start,
        &attention_workspace,
    )?;
    context.synchronize()?;
    let correctness = compare_all(&serial, &candidate)?;
    if correctness.iter().any(|(_, different)| *different != 0) {
        return Err(format!("M8 attention prefill differs from serial M1: {correctness:?}").into());
    }

    for _ in 0..WARMUPS {
        reset_caches(&context, &serial.key_cache, &serial.value_cache)?;
        run_serial(
            &context,
            &weights,
            &serial,
            &cache_positions,
            &rope_positions,
            start,
            &attention_workspace,
        )?;
        reset_caches(&context, &candidate.key_cache, &candidate.value_cache)?;
        run_candidate(
            &context,
            &weights,
            &candidate,
            &cache_positions,
            &rope_positions,
            start,
            &attention_workspace,
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
            if candidate_arm {
                reset_caches(&context, &candidate.key_cache, &candidate.value_cache)?;
            } else {
                reset_caches(&context, &serial.key_cache, &serial.value_cache)?;
            }
            let begin = Instant::now();
            if candidate_arm {
                run_candidate(
                    &context,
                    &weights,
                    &candidate,
                    &cache_positions,
                    &rope_positions,
                    start,
                    &attention_workspace,
                )?;
            } else {
                run_serial(
                    &context,
                    &weights,
                    &serial,
                    &cache_positions,
                    &rope_positions,
                    start,
                    &attention_workspace,
                )?;
            }
            context.synchronize()?;
            let elapsed = begin.elapsed().as_secs_f64() * 1.0e6;
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
            "schema":"apxinf.qwen35.attention_prefill_probe.v1",
            "model_dir":model_dir,"layer":3,
            "contract":{
                "tokens":TOKENS,"start_position":start,"end_position":start+TOKENS-1,
                "hidden":HIDDEN,"q_heads":Q_HEADS,"kv_heads":KV_HEADS,"head_dim":HEAD_DIM,
                "max_seq_len":MAX_SEQ_LEN,"kv_dtype":"bf16",
            "path":format!("q/k/v M8 W4 + M8 QK norm/partial RoPE + one M8 KV append + eight causal {} attention calls + M8 gate + o M8 W4",
                if start+1>=qwen35_attention::SPLIT_CTA_CANDIDATE_MIN_KV_BUCKET{"split16"}else{"incumbent"}),
                "oracle":"eight serial production-shape full-attention executions",
            "attention_path":if start+1>=qwen35_attention::SPLIT_CTA_CANDIDATE_MIN_KV_BUCKET{
                "split16 candidate"
            }else{"incumbent one-CTA-per-query-head"},
            },
            "device":{
                "name":context.caps().device_name,"sm":context.caps().sm,
                "multiprocessors":context.caps().multiprocessor_count,
                "cuda":context.library_versions().cuda,"cublas":context.library_versions().cublas,
            },
            "kernel_build_id":KERNEL_BUILD_ID,
            "correctness":{
                "different_values":correctness,
                "comparison":"BF16 bitwise identity at all projection, prepare, causal attention, cache, gate, and output endpoints",
                "pass":true,
            },
            "timing":{
                "boundary":"complete real layer-3 attention tile launches through stream synchronize; cache reset excluded",
                "warmups_per_arm":WARMUPS,"pairs":PAIRS,"records":records,
                "serial_raw_us":serial_samples,"candidate_raw_us":candidate_samples,
                "serial_median_us":median(&serial_samples),"candidate_median_us":median(&candidate_samples),
                "median_speedup":median(&speedups),"candidate_wins":wins,
                "candidate_tokens_per_second":TOKENS as f64*1.0e6/median(&candidate_samples),
            },
            "evidence_level":"complete-causal-attention-layer-prefill",
            "model_promoted":false,
        }))?
    );
    Ok(())
}

fn load_weights(manifest: &CheckpointManifest) -> Result<Weights, Box<dyn std::error::Error>> {
    Ok(Weights {
        q: load_w4(manifest, &format!("{PREFIX}.q_proj"))?,
        k: load_w4(manifest, &format!("{PREFIX}.k_proj"))?,
        v: load_w4(manifest, &format!("{PREFIX}.v_proj"))?,
        o: load_w4(manifest, &format!("{PREFIX}.o_proj"))?,
        q_norm: to_gpu(load_tensor(manifest, &format!("{PREFIX}.q_norm.weight"))?)?,
        k_norm: to_gpu(load_tensor(manifest, &format!("{PREFIX}.k_norm.weight"))?)?,
    })
}

fn load_w4(manifest: &CheckpointManifest, base: &str) -> Result<W4, Box<dyn std::error::Error>> {
    let shape = safetensors::read_small_i64(
        manifest
            .tensor(&format!("{base}.weight_shape"))
            .ok_or_else(|| format!("missing `{base}.weight_shape`"))?,
    )?;
    Ok(W4 {
        packed: to_gpu(load_tensor(manifest, &format!("{base}.weight_packed"))?)?,
        scales: to_gpu(load_tensor(manifest, &format!("{base}.weight_scale"))?)?,
        zero: to_gpu(load_tensor(manifest, &format!("{base}.weight_zero_point"))?)?,
        input: usize::try_from(shape[1])?,
        output: usize::try_from(shape[0])?,
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
            let phase = column as f32 * 0.004_882_812_5 + token as f32 * 0.109_375;
            half::bf16::from_f32((phase.sin() + 0.2 * phase.cos()) * 0.2)
        })
        .collect()
}

fn serial_workspace(input: &[half::bf16]) -> Result<Rows, Box<dyn std::error::Error>> {
    let mut rows = Rows {
        input: Vec::with_capacity(TOKENS),
        q_projection: Vec::with_capacity(TOKENS),
        k_projection: Vec::with_capacity(TOKENS),
        v_projection: Vec::with_capacity(TOKENS),
        query: Vec::with_capacity(TOKENS),
        key: Vec::with_capacity(TOKENS),
        value: Vec::with_capacity(TOKENS),
        gate: Vec::with_capacity(TOKENS),
        attended: Vec::with_capacity(TOKENS),
        gated: Vec::with_capacity(TOKENS),
        output: Vec::with_capacity(TOKENS),
        key_cache: gpu_zeros(&[KV_HEADS, MAX_SEQ_LEN, HEAD_DIM])?,
        value_cache: gpu_zeros(&[KV_HEADS, MAX_SEQ_LEN, HEAD_DIM])?,
    };
    for token in 0..TOKENS {
        rows.input.push(to_gpu(Tensor::from_bf16(
            vec![1, HIDDEN],
            &input[token * HIDDEN..(token + 1) * HIDDEN],
        )?)?);
        rows.q_projection.push(gpu_zeros(&[1, Q_PROJECTION])?);
        rows.k_projection.push(gpu_zeros(&[1, KV_WIDTH])?);
        rows.v_projection.push(gpu_zeros(&[1, KV_WIDTH])?);
        rows.query.push(gpu_zeros(&[Q_HEADS, HEAD_DIM])?);
        rows.key.push(gpu_zeros(&[KV_HEADS, HEAD_DIM])?);
        rows.value.push(gpu_zeros(&[KV_HEADS, HEAD_DIM])?);
        rows.gate.push(gpu_zeros(&[Q_HEADS, HEAD_DIM])?);
        rows.attended.push(gpu_zeros(&[Q_HEADS, HEAD_DIM])?);
        rows.gated.push(gpu_zeros(&[Q_HEADS, HEAD_DIM])?);
        rows.output.push(gpu_zeros(&[1, HIDDEN])?);
    }
    Ok(rows)
}

fn candidate_workspace(input: &[half::bf16]) -> Result<Tile, Box<dyn std::error::Error>> {
    Ok(Tile {
        input: to_gpu(Tensor::from_bf16(vec![TOKENS, HIDDEN], input)?)?,
        q_projection: gpu_zeros(&[TOKENS, Q_PROJECTION])?,
        k_projection: gpu_zeros(&[TOKENS, KV_WIDTH])?,
        v_projection: gpu_zeros(&[TOKENS, KV_WIDTH])?,
        query: gpu_zeros(&[TOKENS, Q_HEADS, HEAD_DIM])?,
        key: gpu_zeros(&[TOKENS, KV_HEADS, HEAD_DIM])?,
        value: gpu_zeros(&[TOKENS, KV_HEADS, HEAD_DIM])?,
        gate: gpu_zeros(&[TOKENS, Q_HEADS, HEAD_DIM])?,
        attended: gpu_zeros(&[TOKENS, Q_HEADS, HEAD_DIM])?,
        gated: gpu_zeros(&[TOKENS, Q_HEADS, HEAD_DIM])?,
        output: gpu_zeros(&[TOKENS, HIDDEN])?,
        key_cache: gpu_zeros(&[KV_HEADS, MAX_SEQ_LEN, HEAD_DIM])?,
        value_cache: gpu_zeros(&[KV_HEADS, MAX_SEQ_LEN, HEAD_DIM])?,
    })
}

fn gpu_zeros(shape: &[usize]) -> Result<Tensor, Box<dyn std::error::Error>> {
    Ok(transfers::to_cuda(
        &Tensor::zeros(shape.to_vec(), DType::BF16),
        0,
    )?)
}

fn reset_caches(context: &CudaContext, key: &Tensor, value: &Tensor) -> apxinf_core::Result<()> {
    CudaBuffer::from_tensor(key)
        .map_err(apxinf_core::Error::Cuda)?
        .memset_async(0, context.stream())
        .map_err(apxinf_core::Error::Cuda)?;
    CudaBuffer::from_tensor(value)
        .map_err(apxinf_core::Error::Cuda)?
        .memset_async(0, context.stream())
        .map_err(apxinf_core::Error::Cuda)?;
    context.synchronize().map_err(apxinf_core::Error::Cuda)
}

fn run_serial(
    context: &CudaContext,
    weights: &Weights,
    rows: &Rows,
    cache_positions: &HostMappedBuffer,
    rope_positions: &HostMappedBuffer,
    start: usize,
    attention_workspace: &qwen35_attention::SplitCtaWorkspace,
) -> apxinf_core::Result<()> {
    let key_cache = CudaBuffer::from_tensor(&rows.key_cache).map_err(apxinf_core::Error::Cuda)?;
    let value_cache =
        CudaBuffer::from_tensor(&rows.value_cache).map_err(apxinf_core::Error::Cuda)?;
    for token in 0..TOKENS {
        let cache_position = cache_positions
            .address_at(token * 4, 4)
            .map_err(apxinf_core::Error::Cuda)?;
        let rope_position = rope_positions
            .address_at(token * 3 * 4, 3 * 4)
            .map_err(apxinf_core::Error::Cuda)?;
        gemm::w4a16_write(
            context,
            &rows.input[token],
            weights.q.view(),
            &rows.q_projection[token],
        )?;
        gemm::w4a16_write(
            context,
            &rows.input[token],
            weights.k.view(),
            &rows.k_projection[token],
        )?;
        gemm::w4a16_write(
            context,
            &rows.input[token],
            weights.v.view(),
            &rows.v_projection[token],
        )?;
        qwen35_attention::prepare_write(
            context,
            &rows.q_projection[token],
            &rows.k_projection[token],
            &rows.v_projection[token],
            &weights.q_norm,
            &weights.k_norm,
            &rows.query[token],
            &rows.key[token],
            &rows.value[token],
            &rows.gate[token],
            rope_position,
        )?;
        cache::append_at(
            context,
            DType::BF16,
            &key_cache,
            &CudaBuffer::from_tensor(&rows.key[token]).map_err(apxinf_core::Error::Cuda)?,
            KV_HEADS,
            HEAD_DIM,
            MAX_SEQ_LEN,
            cache_position,
        )?;
        cache::append_at(
            context,
            DType::BF16,
            &value_cache,
            &CudaBuffer::from_tensor(&rows.value[token]).map_err(apxinf_core::Error::Cuda)?,
            KV_HEADS,
            HEAD_DIM,
            MAX_SEQ_LEN,
            cache_position,
        )?;
        run_attention(
            context,
            &CudaBuffer::from_tensor(&rows.query[token]).map_err(apxinf_core::Error::Cuda)?,
            &key_cache,
            &value_cache,
            &CudaBuffer::from_tensor(&rows.attended[token]).map_err(apxinf_core::Error::Cuda)?,
            start + token + 1,
            MAX_SEQ_LEN,
            cache_position,
            attention_workspace,
        )?;
        qwen35_attention::gate_write(
            context,
            &rows.attended[token],
            &rows.gate[token],
            &rows.gated[token],
        )?;
        gemm::w4a16_write(
            context,
            &rows.gated[token].reshape(vec![1, WIDTH])?,
            weights.o.view(),
            &rows.output[token],
        )?;
    }
    Ok(())
}

fn run_candidate(
    context: &CudaContext,
    weights: &Weights,
    tile: &Tile,
    cache_positions: &HostMappedBuffer,
    rope_positions: &HostMappedBuffer,
    start: usize,
    attention_workspace: &qwen35_attention::SplitCtaWorkspace,
) -> apxinf_core::Result<()> {
    gemm::w4a16_m8_write(context, &tile.input, weights.q.view(), &tile.q_projection)?;
    gemm::w4a16_m8_write(context, &tile.input, weights.k.view(), &tile.k_projection)?;
    gemm::w4a16_m8_write(context, &tile.input, weights.v.view(), &tile.v_projection)?;
    qwen35_attention::prepare_m8_write(
        context,
        &tile.q_projection,
        &tile.k_projection,
        &tile.v_projection,
        &weights.q_norm,
        &weights.k_norm,
        &tile.query,
        &tile.key,
        &tile.value,
        &tile.gate,
        rope_positions.address(),
    )?;
    let key_cache = CudaBuffer::from_tensor(&tile.key_cache).map_err(apxinf_core::Error::Cuda)?;
    let value_cache =
        CudaBuffer::from_tensor(&tile.value_cache).map_err(apxinf_core::Error::Cuda)?;
    cache::append(
        context,
        &key_cache,
        &tile.key,
        KV_HEADS,
        HEAD_DIM,
        MAX_SEQ_LEN,
        start,
        TOKENS,
    )?;
    cache::append(
        context,
        &value_cache,
        &tile.value,
        KV_HEADS,
        HEAD_DIM,
        MAX_SEQ_LEN,
        start,
        TOKENS,
    )?;
    let query = CudaBuffer::from_tensor(&tile.query).map_err(apxinf_core::Error::Cuda)?;
    let attended = CudaBuffer::from_tensor(&tile.attended).map_err(apxinf_core::Error::Cuda)?;
    let row_bytes = WIDTH * DType::BF16.size_in_bytes();
    for token in 0..TOKENS {
        let cache_position = cache_positions
            .address_at(token * 4, 4)
            .map_err(apxinf_core::Error::Cuda)?;
        run_attention(
            context,
            &query
                .view(token * row_bytes, row_bytes)
                .map_err(apxinf_core::Error::Cuda)?,
            &key_cache,
            &value_cache,
            &attended
                .view(token * row_bytes, row_bytes)
                .map_err(apxinf_core::Error::Cuda)?,
            start + token + 1,
            MAX_SEQ_LEN,
            cache_position,
            attention_workspace,
        )?;
    }
    qwen35_attention::gate_m8_write(context, &tile.attended, &tile.gate, &tile.gated)?;
    gemm::w4a16_m8_write(
        context,
        &tile.gated.reshape(vec![TOKENS, WIDTH])?,
        weights.o.view(),
        &tile.output,
    )
}

#[allow(clippy::too_many_arguments)]
fn run_attention(
    context: &CudaContext,
    query: &CudaBuffer,
    key_cache: &CudaBuffer,
    value_cache: &CudaBuffer,
    output: &CudaBuffer,
    kv_len: usize,
    max_seq_len: usize,
    position: apxinf_cuda::CudaDeviceAddress,
    workspace: &qwen35_attention::SplitCtaWorkspace,
) -> apxinf_core::Result<()> {
    if let Some(split_count) = qwen35_attention::split_cta_candidate_for_bucket(kv_len) {
        qwen35_attention::flash_split_cta_buffer_write(
            context,
            query,
            key_cache,
            value_cache,
            output,
            workspace,
            split_count,
            kv_len,
            max_seq_len,
            ATTENTION_SCALE,
            position,
        )
    } else {
        attention::flash_bf16_into(
            context,
            query,
            key_cache,
            value_cache,
            output,
            Q_HEADS,
            KV_HEADS,
            HEAD_DIM,
            kv_len,
            max_seq_len,
            ATTENTION_SCALE,
            position,
        )
    }
}

fn compare_all(
    rows: &Rows,
    tile: &Tile,
) -> Result<Vec<(String, usize)>, Box<dyn std::error::Error>> {
    let mut result = Vec::new();
    for (name, serial, candidate) in [
        ("q_projection", &rows.q_projection, &tile.q_projection),
        ("k_projection", &rows.k_projection, &tile.k_projection),
        ("v_projection", &rows.v_projection, &tile.v_projection),
        ("query", &rows.query, &tile.query),
        ("key", &rows.key, &tile.key),
        ("value", &rows.value, &tile.value),
        ("gate", &rows.gate, &tile.gate),
        ("attended", &rows.attended, &tile.attended),
        ("gated", &rows.gated, &tile.gated),
        ("output", &rows.output, &tile.output),
    ] {
        result.push((name.into(), different_many(serial, candidate)?));
    }
    result.push((
        "key_cache".into(),
        different(&rows.key_cache, &tile.key_cache)?,
    ));
    result.push((
        "value_cache".into(),
        different(&rows.value_cache, &tile.value_cache)?,
    ));
    Ok(result)
}

fn different_many(rows: &[Tensor], tile: &Tensor) -> Result<usize, Box<dyn std::error::Error>> {
    let tile = transfers::to_cpu(tile)?;
    let tile = tile.as_bf16()?;
    let mut offset = 0;
    let mut count = 0;
    for row in rows {
        let row = transfers::to_cpu(row)?;
        let values = row.as_bf16()?;
        count += values
            .iter()
            .zip(&tile[offset..offset + values.len()])
            .filter(|(left, right)| left.to_bits() != right.to_bits())
            .count();
        offset += values.len();
    }
    Ok(count)
}

fn different(left: &Tensor, right: &Tensor) -> Result<usize, Box<dyn std::error::Error>> {
    let left = transfers::to_cpu(left)?;
    let right = transfers::to_cpu(right)?;
    Ok(left
        .as_bf16()?
        .iter()
        .zip(right.as_bf16()?)
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
