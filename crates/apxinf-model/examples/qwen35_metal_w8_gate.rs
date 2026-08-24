//! Real-checkpoint teacher-forced gate for the decode-only Metal W8 lm_head.

use std::error::Error;
use std::path::PathBuf;

use apxinf_core::{Device, Tensor};
use apxinf_model::{GeneralQwen35, LlmInput, LlmTrait, Qwen35Config};
use apxinf_tokenizer::{ChatMessage, Tokenizer};

fn main() -> Result<(), Box<dyn Error>> {
    let mut arguments = std::env::args_os().skip(1);
    let model_dir = PathBuf::from(
        arguments
            .next()
            .ok_or("usage: qwen35_metal_w8_gate MODEL_DIR [STEPS] [PROMPT]")?,
    );
    let steps = arguments
        .next()
        .map(|value| value.to_string_lossy().parse::<usize>())
        .transpose()?
        .unwrap_or(128);
    let prompt = arguments
        .next()
        .map(|value| value.to_string_lossy().into_owned())
        .unwrap_or_else(|| "Hello".to_owned());
    if steps == 0 {
        return Err("STEPS must be greater than zero".into());
    }

    let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json"))?;
    let formatted = tokenizer.apply_chat_template(&[ChatMessage::user(&prompt)])?;
    let prompt_tokens = tokenizer.encode(&formatted)?;
    let config = Qwen35Config::from_json_file(&model_dir.join("config.json"))?;
    let (tensors, _) = apxinf_loader::safetensors::load_native_path_filtered(&model_dir, |name| {
        name.starts_with("model.language_model.") || name == "lm_head.weight"
    })?;
    let mut model = GeneralQwen35::from_weights_with_metal_w8(
        config,
        tensors,
        Device::Cpu,
        prompt_tokens.len() + steps + 1,
    )?;

    let prefill_logits = model.prefill_for_generation(LlmInput::text(&prompt_tokens))?;
    let mut teacher_token = argmax(&prefill_logits, model.vocab_size())?;
    let started = std::time::Instant::now();
    let mut raw_w8_mismatches = Vec::new();
    let mut reranked_mismatches = Vec::new();
    let mut rerank_changes = Vec::new();
    let mut topk_elapsed_ns = 0u128;
    let mut rerank_elapsed_ns = 0u128;
    for step in 0..steps {
        let position = (prompt_tokens.len() + step) as u32;
        let comparison = model.teacher_forced_decode_candidates(teacher_token, position)?;
        let cpu_token = comparison.cpu_token;
        let candidates = comparison.w8_candidates;
        let reranked_token = comparison.reranked_token;
        topk_elapsed_ns += comparison.topk_elapsed_ns;
        rerank_elapsed_ns += comparison.rerank_elapsed_ns;
        let raw_w8_token = candidates[0];
        if cpu_token != raw_w8_token {
            raw_w8_mismatches.push(serde_json::json!({
                "step": step,
                "input_token": teacher_token,
                "cpu_token": cpu_token,
                "metal_w8_top1_token": raw_w8_token,
                "metal_w8_top4": candidates,
            }));
        }
        if cpu_token != reranked_token {
            reranked_mismatches.push(serde_json::json!({
                "step": step,
                "input_token": teacher_token,
                "cpu_token": cpu_token,
                "reranked_token": reranked_token,
                "metal_w8_top4": candidates,
            }));
        }
        if raw_w8_token != reranked_token {
            rerank_changes.push(serde_json::json!({
                "step": step,
                "input_token": teacher_token,
                "metal_w8_top1_token": raw_w8_token,
                "reranked_token": reranked_token,
                "cpu_token": cpu_token,
                "metal_w8_top4": candidates,
            }));
        }
        teacher_token = cpu_token;
    }

    // Exercise the exact production generation loop after reset. The
    // teacher-forced loop above intentionally drives prefill and decode
    // manually, so it cannot by itself catch a fast-hook wiring regression.
    model.reset();
    let production_steps = steps.min(10);
    let (production_tokens, production_profile) = model.generate_streaming(
        LlmInput::text(&prompt_tokens),
        production_steps,
        |_| {},
        None,
    )?;

    println!(
        "{}",
        serde_json::to_string(&serde_json::json!({
            "format": "apxinf-qwen35-metal-w8-top4-teacher-gate-v2",
            "prompt": prompt,
            "prompt_token_count": prompt_tokens.len(),
            "comparisons": steps,
            "raw_w8_top1": {
                "matches": steps - raw_w8_mismatches.len(),
                "match_rate": (steps - raw_w8_mismatches.len()) as f64 / steps as f64,
                "mismatches": raw_w8_mismatches,
            },
            "f32_reranked": {
                "matches": steps - reranked_mismatches.len(),
                "match_rate": (steps - reranked_mismatches.len()) as f64 / steps as f64,
                "mismatches": reranked_mismatches,
            },
            "rerank_changes": rerank_changes,
            "production_generation": {
                "comparisons": production_steps,
                "generated_token_ids": production_tokens,
                "ttft_ms": production_profile.ttft_ms(),
                "generation_tps": production_profile.generation_tps(),
            },
            "isolated_timing": {
                "metal_topk4_total_ms": topk_elapsed_ns as f64 / 1_000_000.0,
                "metal_topk4_mean_us": topk_elapsed_ns as f64 / steps as f64 / 1_000.0,
                "f32_rerank_total_ms": rerank_elapsed_ns as f64 / 1_000_000.0,
                "f32_rerank_mean_us": rerank_elapsed_ns as f64 / steps as f64 / 1_000.0,
                "rerank_share_of_topk4_plus_rerank": rerank_elapsed_ns as f64
                    / (topk_elapsed_ns + rerank_elapsed_ns) as f64,
            },
            "elapsed_ms": started.elapsed().as_secs_f64() * 1000.0,
            "quantization": {
                "layout": "hf-row-major",
                "scheme": "symmetric-int8-per-row-group",
                "group_size": 64,
                "scale_dtype": "f32",
            },
        }))?
    );
    Ok(())
}

fn argmax(logits: &Tensor, vocab_size: usize) -> Result<u32, Box<dyn Error>> {
    if logits.shape().dims() != [1, vocab_size] {
        return Err(format!(
            "expected prefill logits [1, {vocab_size}], got {}",
            logits.shape()
        )
        .into());
    }
    let mut best = f32::NEG_INFINITY;
    let mut best_token = 0u32;
    for (token, &score) in logits.as_f32()?.iter().enumerate() {
        if score > best {
            best = score;
            best_token = token as u32;
        }
    }
    Ok(best_token)
}
