//! Stateful multi-token Qwen3.5 decode trajectory on the native ApxInf graph.

use std::path::Path;
use std::time::Instant;

use apxinf_core::Tensor;
use apxinf_cuda::tuning::KERNEL_BUILD_ID;
use apxinf_cuda::CudaContext;
use apxinf_loader::safetensors::{self, CheckpointManifest};
use apxinf_model::qwen35::{load_embedding_row, HybridUnit, HybridUnitMode, Qwen35LmHead};

const KV_HEADS: usize = 4;
const HEAD_DIM: usize = 256;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_dir = std::env::args()
        .nth(1)
        .ok_or("usage: qwen35_trajectory_probe MODEL_DIR")?;
    let initial_token = std::env::var("APXINF_INPUT_TOKEN")
        .unwrap_or_else(|_| "151644".into())
        .parse::<u32>()?;
    let prefix_len = std::env::var("APXINF_TRAJECTORY_PREFIX")
        .unwrap_or_else(|_| "256".into())
        .parse::<usize>()?;
    let steps = std::env::var("APXINF_TRAJECTORY_STEPS")
        .unwrap_or_else(|_| "16".into())
        .parse::<usize>()?;
    if prefix_len == 0 || steps == 0 || prefix_len + steps > 32768 {
        return Err("trajectory requires prefix>0, steps>0, and prefix+steps<=32768".into());
    }
    let max_seq_len = (prefix_len + steps).next_power_of_two();
    let manifest = safetensors::inspect_path(Path::new(&model_dir))?;
    let context = CudaContext::new(0)?;
    let initial_embedding = load_embedding_row(&manifest, initial_token)?;
    let (key_cache, value_cache) = deterministic_caches(max_seq_len)?;
    let decoder = HybridUnit::load_all(&manifest, &context, max_seq_len)?;
    let lm_head = Qwen35LmHead::load(&manifest, &context)?;

    // Prepare cuBLAS plans and kernels for both selectors, then reset all
    // mutable state inside the measured trajectories below.
    for mode in [HybridUnitMode::Native, HybridUnitMode::ModelOptimized] {
        let _ = run_trajectory(
            &manifest,
            &context,
            &decoder,
            &lm_head,
            mode,
            initial_token,
            &initial_embedding,
            &key_cache,
            &value_cache,
            prefix_len,
            1,
            max_seq_len,
        )?;
    }

    let native = run_trajectory(
        &manifest,
        &context,
        &decoder,
        &lm_head,
        HybridUnitMode::Native,
        initial_token,
        &initial_embedding,
        &key_cache,
        &value_cache,
        prefix_len,
        steps,
        max_seq_len,
    )?;
    let optimized = run_trajectory(
        &manifest,
        &context,
        &decoder,
        &lm_head,
        HybridUnitMode::ModelOptimized,
        initial_token,
        &initial_embedding,
        &key_cache,
        &value_cache,
        prefix_len,
        steps,
        max_seq_len,
    )?;
    let exact_steps = native
        .tokens
        .iter()
        .zip(&optimized.tokens)
        .filter(|(left, right)| left == right)
        .count();
    let exact_trajectory = native.tokens == optimized.tokens;

    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "schema":"apxinf.qwen35.trajectory_probe.v1",
            "model_dir":model_dir,
            "contract":{
                "initial_token":initial_token,"prefix_len":prefix_len,"steps":steps,
                "max_seq_len":max_seq_len,"kv_dtype":"bf16","batch":1,"sm":89,
                "state":"48 recurrent + 48 conv + 16 KV caches mutate continuously; one reset per trajectory",
                "timing":"embedding-row H2D + input norm + 64 layers + final norm + W8 LM head + D2H logits + CPU argmax; disk row read excluded",
                "stopping_reason":"fixed_steps",
            },
            "kernel_build_id":KERNEL_BUILD_ID,
            "correctness":{
                "native_tokens":native.tokens,"optimized_tokens":optimized.tokens,
                "exact_steps":exact_steps,"match_rate":exact_steps as f64/steps as f64,
                "exact_trajectory":exact_trajectory,
                "pass":exact_steps>0,
            },
            "performance":{
                "native_raw_us":native.times_us,"optimized_raw_us":optimized.times_us,
                "native_total_us":native.total_us,"optimized_total_us":optimized.total_us,
                "native_tokens_per_second":steps as f64*1.0e6/native.total_us,
                "optimized_tokens_per_second":steps as f64*1.0e6/optimized.total_us,
                "speedup":native.total_us/optimized.total_us,
            },
            "evidence_level":"stateful-multi-token-offline","service_promoted":false,
        }))?
    );
    Ok(())
}

struct Trajectory {
    tokens: Vec<u32>,
    times_us: Vec<f64>,
    total_us: f64,
}

#[allow(clippy::too_many_arguments)]
fn run_trajectory(
    manifest: &CheckpointManifest,
    context: &CudaContext,
    decoder: &HybridUnit,
    lm_head: &Qwen35LmHead,
    mode: HybridUnitMode,
    initial_token: u32,
    initial_embedding: &Tensor,
    key_cache: &Tensor,
    value_cache: &Tensor,
    prefix_len: usize,
    steps: usize,
    max_seq_len: usize,
) -> Result<Trajectory, Box<dyn std::error::Error>> {
    decoder.reset(context, initial_embedding, key_cache, value_cache)?;
    let mut current_token = initial_token;
    let mut tokens = Vec::with_capacity(steps);
    let mut times_us = Vec::with_capacity(steps);
    for step in 0..steps {
        let embedding = load_embedding_row(manifest, current_token)?;
        let start = Instant::now();
        decoder.set_token_input(context, &embedding)?;
        decoder.forward(
            context,
            mode,
            max_seq_len,
            (prefix_len - 1 + step) as u32,
            false,
        )?;
        lm_head.forward(context, decoder.normalized_output())?;
        current_token = lm_head.argmax_cpu()?;
        times_us.push(start.elapsed().as_secs_f64() * 1.0e6);
        tokens.push(current_token);
    }
    let total_us = times_us.iter().sum();
    Ok(Trajectory {
        tokens,
        times_us,
        total_us,
    })
}

fn deterministic_caches(
    max_seq_len: usize,
) -> Result<(Tensor, Tensor), Box<dyn std::error::Error>> {
    let count = KV_HEADS * max_seq_len * HEAD_DIM;
    let mut key = vec![half::bf16::ZERO; count];
    let mut value = vec![half::bf16::ZERO; count];
    for head in 0..KV_HEADS {
        for token in 0..max_seq_len {
            let base = (head * max_seq_len + token) * HEAD_DIM;
            for dimension in 0..HEAD_DIM {
                let key_bits = (token * 17 + dimension * 13 + head * 7) & 255;
                let value_bits = (token * 29 + dimension * 11 + head * 19) & 255;
                key[base + dimension] = half::bf16::from_f32((key_bits as f32 - 128.0) / 1024.0);
                value[base + dimension] =
                    half::bf16::from_f32(0.1 + (value_bits as f32 - 128.0) / 2048.0);
            }
        }
    }
    Ok((
        Tensor::from_bf16(vec![KV_HEADS, max_seq_len, HEAD_DIM], &key)?,
        Tensor::from_bf16(vec![KV_HEADS, max_seq_len, HEAD_DIM], &value)?,
    ))
}
