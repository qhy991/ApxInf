//! Common LLM trait for all model implementations.

use std::collections::HashMap;

use apxinf_core::{Device, Error, Result, Tensor};
use apxinf_loader::ModelConfig;

use crate::profiling::GenerationProfile;

/// Processor output for one or more images in a generation prompt.
///
/// `pixel_values` is deliberately borrowed: creating a text-only request does
/// not allocate, clone a tensor, or alter the decode hot path. Models define
/// the exact tensor layout they accept. `grid_thw` contains one entry per
/// image represented by the (possibly concatenated) tensor.
#[derive(Clone, Copy, Debug)]
pub struct ImageInput<'a> {
    pub pixel_values: &'a Tensor,
    pub grid_thw: &'a [[u32; 3]],
}

/// Processor output for one or more audio clips in a generation prompt.
///
/// Features use the model-owned `[frames, mel_bins]` layout. The mask is the
/// processor-produced frame mask, `feature_lengths` identifies the valid
/// frames in each group, and `token_counts` maps each group to the expanded
/// audio-placeholder run in `token_ids`. The model validates all four views
/// before executing the audio tower.
#[derive(Clone, Copy, Debug)]
pub struct AudioInput<'a> {
    pub input_features: &'a Tensor,
    pub attention_mask: &'a Tensor,
    pub feature_lengths: &'a [u32],
    pub token_counts: &'a [u32],
}

impl<'a> AudioInput<'a> {
    pub const fn new(
        input_features: &'a Tensor,
        attention_mask: &'a Tensor,
        feature_lengths: &'a [u32],
        token_counts: &'a [u32],
    ) -> Self {
        Self {
            input_features,
            attention_mask,
            feature_lengths,
            token_counts,
        }
    }
}

impl<'a> ImageInput<'a> {
    pub const fn new(pixel_values: &'a Tensor, grid_thw: &'a [[u32; 3]]) -> Self {
        Self {
            pixel_values,
            grid_thw,
        }
    }
}

/// Unified prompt input for text and vision-language generation.
///
/// Media is attached to the prompt and consumed during prefill. Autoregressive
/// decode continues to use token-only [`LlmTrait::forward`], so text and VLM
/// models share the same generation loop without a modality check per token.
#[derive(Clone, Copy, Debug)]
pub struct LlmInput<'a> {
    pub token_ids: &'a [u32],
    pub image: Option<ImageInput<'a>>,
    pub audio: Option<AudioInput<'a>>,
}

impl<'a> LlmInput<'a> {
    pub const fn text(token_ids: &'a [u32]) -> Self {
        Self {
            token_ids,
            image: None,
            audio: None,
        }
    }

    pub const fn with_image(token_ids: &'a [u32], image: ImageInput<'a>) -> Self {
        Self {
            token_ids,
            image: Some(image),
            audio: None,
        }
    }

    pub const fn with_audio(token_ids: &'a [u32], audio: AudioInput<'a>) -> Self {
        Self {
            token_ids,
            image: None,
            audio: Some(audio),
        }
    }

    pub const fn with_media(
        token_ids: &'a [u32],
        image: Option<ImageInput<'a>>,
        audio: Option<AudioInput<'a>>,
    ) -> Self {
        Self {
            token_ids,
            image,
            audio,
        }
    }
}

/// Input modalities accepted by an LLM implementation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LlmCapabilities {
    pub image: bool,
    pub audio: bool,
}

impl LlmCapabilities {
    pub const TEXT_ONLY: Self = Self {
        image: false,
        audio: false,
    };
    pub const VISION: Self = Self {
        image: true,
        audio: false,
    };
    pub const OMNI: Self = Self {
        image: true,
        audio: true,
    };
}

/// Common interface for all LLM implementations.
pub trait LlmTrait {
    /// Load model weights and configure for the given device.
    fn load(config: ModelConfig, weights: HashMap<String, Tensor>, device: Device) -> Result<Self>
    where
        Self: Sized;

    /// Token-level forward pass.
    /// Returns logits of shape `[seq_len, vocab_size]`.
    fn forward(&mut self, token_ids: &[u32], start_pos: u32) -> Result<Tensor>;

    /// Modalities accepted by [`Self::prefill`]. Text is always supported.
    fn capabilities(&self) -> LlmCapabilities {
        LlmCapabilities::TEXT_ONLY
    }

    /// Process a complete prompt and return its logits.
    ///
    /// Text-only models inherit this implementation. It rejects image input
    /// explicitly instead of silently ignoring it. Vision-language models
    /// override this one request-level hook to encode and merge image features.
    fn prefill(&mut self, input: LlmInput<'_>) -> Result<Tensor> {
        if input.image.is_some() {
            return Err(Error::Other(
                "this model does not support image input".into(),
            ));
        }
        if input.audio.is_some() {
            return Err(Error::Other(
                "this model does not support audio input".into(),
            ));
        }
        self.forward(input.token_ids, 0)
    }

    /// Reset state for a new generation.
    fn reset(&mut self);

    /// Optional hook called once before prefill, with the prompt length and
    /// the number of tokens that will be generated. Models with a CUDA
    /// decode graph use it to pre-capture every bucket they'll hit so the
    /// per-token TPOT stays at pure graph-replay cost. Default: no-op.
    fn prewarm_decode(&mut self, _prompt_len: usize, _max_new_tokens: usize) {}

    /// Optional hard context capacity. The shared loop rejects combined
    /// prompt+completion overflow before prewarm or cache mutation.
    fn max_context_len(&self) -> Option<usize> {
        None
    }

    /// Optional per-request generation limit owned by a deployed model.
    fn max_new_tokens_limit(&self) -> Option<usize> {
        None
    }

    /// Greedy-decode one token directly to its id, skipping the full-logits
    /// D2H + CPU argmax. Returns `None` if the model has no GPU-argmax fast
    /// path (caller falls back to `forward` + `argmax_last_row`).
    fn decode_token(&mut self, _token: u32, _pos: u32) -> Option<Result<u32>> {
        None
    }

    /// Vocabulary size (used by default generate_streaming for argmax).
    fn vocab_size(&self) -> usize;

    /// Ergonomic, statically typed streaming entrypoint. Models that replace
    /// the shared greedy algorithm should override `generate_streaming_dyn`
    /// so the same behavior is visible through `AutoModel`.
    fn generate_streaming(
        &mut self,
        input: LlmInput<'_>,
        max_new_tokens: usize,
        on_token: impl FnMut(u32),
        eos_token_id: Option<u32>,
    ) -> Result<(Vec<u32>, GenerationProfile)>
    where
        Self: Sized,
    {
        generate_streaming(self, input, max_new_tokens, on_token, eos_token_id)
    }

    /// Object-safe entry used by `AutoModel`. The vtable dispatch happens
    /// once for the complete request; the concrete model then owns the whole
    /// prefill/decode loop rather than paying model dispatch per token.
    fn generate_streaming_dyn(
        &mut self,
        input: LlmInput<'_>,
        max_new_tokens: usize,
        on_token: &mut dyn FnMut(u32),
        eos_token_id: Option<u32>,
    ) -> Result<(Vec<u32>, GenerationProfile)> {
        generate_streaming(self, input, max_new_tokens, on_token, eos_token_id)
    }
}

/// Run the shared greedy generation loop for a concrete model or
/// `dyn LlmTrait`. Most callers use [`LlmTrait::generate_streaming`] or
/// [`crate::LoadedModel::generate_streaming`]; this function contains the one
/// canonical generation algorithm.
pub fn generate_streaming<M, F>(
    model: &mut M,
    input: LlmInput<'_>,
    max_new_tokens: usize,
    mut on_token: F,
    eos_token_id: Option<u32>,
) -> Result<(Vec<u32>, GenerationProfile)>
where
    M: LlmTrait + ?Sized,
    F: FnMut(u32),
{
    let prompt_tokens = input.token_ids;
    validate_generation_limits(
        prompt_tokens.len(),
        max_new_tokens,
        model.max_new_tokens_limit(),
        model.max_context_len(),
    )?;
    // Reject unsupported media before graph prewarm or any model forward.
    if input.image.is_some() && !model.capabilities().image {
        return Err(Error::Other(
            "this model does not support image input".into(),
        ));
    }
    if input.audio.is_some() && !model.capabilities().audio {
        return Err(Error::Other(
            "this model does not support audio input".into(),
        ));
    }

    let mut profile = GenerationProfile::new();
    let mut generated = Vec::with_capacity(max_new_tokens);
    let vocab_size = model.vocab_size();

    // Pre-capture any decode graphs (CUDA) BEFORE prefill so the per-token
    // TPOT below is pure graph replay — keeps capture/instantiate cost in
    // setup (TTFT bucket), not in the steady-state TPOT measurement.
    model.prewarm_decode(prompt_tokens.len(), max_new_tokens);

    // Prefill: process the entire prompt
    let logits = model.prefill(input)?;
    profile.record_first_token();

    let next_token = argmax_last_row(&logits, prompt_tokens.len(), vocab_size)?;
    generated.push(next_token);
    on_token(next_token);

    if eos_token_id == Some(next_token) {
        profile.finalize(prompt_tokens.len(), generated.len());
        return Ok((generated, profile));
    }

    // Decode: one token at a time
    let prompt_len = prompt_tokens.len();
    let mut current_token = next_token;
    let perf = std::env::var("APXINF_PERF")
        .map(|v| !v.is_empty())
        .unwrap_or(false);
    let mut t_fwd = std::time::Duration::ZERO;
    let mut t_am = std::time::Duration::ZERO;
    let mut t_cb = std::time::Duration::ZERO;
    for i in 0..max_new_tokens.saturating_sub(1) {
        let pos = (prompt_len + i) as u32;
        // GPU-argmax fast path: skip full-logits D2H + CPU scan.
        if let Some(Ok(tok)) = model.decode_token(current_token, pos) {
            current_token = tok;
            generated.push(current_token);
            on_token(current_token);
            if eos_token_id == Some(current_token) {
                break;
            }
            continue;
        }
        let t0 = std::time::Instant::now();
        let logits = model.forward(&[current_token], pos)?;
        t_fwd += t0.elapsed();
        let t1 = std::time::Instant::now();
        current_token = argmax_last_row(&logits, 1, vocab_size)?;
        t_am += t1.elapsed();
        generated.push(current_token);
        let t2 = std::time::Instant::now();
        on_token(current_token);
        t_cb += t2.elapsed();
        if eos_token_id == Some(current_token) {
            break;
        }
    }
    if perf {
        let n = generated.len().saturating_sub(1).max(1) as f32;
        eprintln!(
            "[loop] fwd={:.2}ms am={:.3}ms cb={:.3}ms (per-tok)",
            t_fwd.as_secs_f32() * 1000.0 / n,
            t_am.as_secs_f32() * 1000.0 / n,
            t_cb.as_secs_f32() * 1000.0 / n
        );
    }

    profile.finalize(prompt_len, generated.len());
    Ok((generated, profile))
}

/// Validate the one canonical autoregressive capacity contract.
///
/// Prompt and completion limits are not independent allocation promises: a
/// request must satisfy both the completion cap and `prompt + completion <=
/// context`. The checked addition makes oversized input fail before model,
/// cache, or backend work.
pub fn validate_generation_limits(
    prompt_tokens: usize,
    max_new_tokens: usize,
    max_new_tokens_limit: Option<usize>,
    context_limit: Option<usize>,
) -> Result<()> {
    if prompt_tokens == 0 {
        return Err(Error::Other("generate_streaming: empty prompt".into()));
    }
    if max_new_tokens == 0 {
        return Err(Error::Other(
            "generate_streaming: max_new_tokens must be positive".into(),
        ));
    }
    if let Some(limit) = max_new_tokens_limit {
        if max_new_tokens > limit {
            return Err(Error::Other(format!(
                "generate_streaming: max_new_tokens {max_new_tokens} exceeds model limit {limit}"
            )));
        }
    }
    if let Some(limit) = context_limit {
        let required = prompt_tokens.checked_add(max_new_tokens).ok_or_else(|| {
            Error::Other("generate_streaming: combined context length overflow".into())
        })?;
        if required > limit {
            return Err(Error::Other(format!(
                "generate_streaming: prompt {prompt_tokens} + completion {max_new_tokens} exceeds context {limit}"
            )));
        }
    }
    Ok(())
}

/// Extract logits for the last row and return its argmax token.
/// `logits` shape: `[seq_len, vocab_size]`. Logits may live on any device — this
/// helper moves to CPU if needed (callers' responsibility for now).
fn argmax_last_row(logits: &Tensor, seq_len: usize, vocab_size: usize) -> Result<u32> {
    let dims = logits.shape().dims();
    if dims.len() != 2 || dims[1] != vocab_size || dims[0] == 0 {
        return Err(Error::Other(format!(
            "argmax logits shape {dims:?}, expected [rows, {vocab_size}]"
        )));
    }
    let _ = seq_len;
    let last_row_offset = (dims[0] - 1) * vocab_size;
    // Fast path: scan bf16 directly (the decode graph returns a bf16 row).
    // Manual loop with `>` beats the iterator + partial_cmp (no NaN handling
    // overhead; logits don't contain NaN in practice).
    if let Ok(data) = logits.as_bf16() {
        let row = &data[last_row_offset..last_row_offset + vocab_size];
        let mut best = half::bf16::from_f32(f32::NEG_INFINITY);
        let mut best_i: u32 = 0;
        for (i, &v) in row.iter().enumerate() {
            if v > best {
                best = v;
                best_i = i as u32;
            }
        }
        return Ok(best_i);
    }
    // Fallback: f32 path for non-bf16 tensors (prefill, CPU models).
    let data = logits.to_f32_vec()?;
    let row = &data[last_row_offset..last_row_offset + vocab_size];
    let mut best = f32::NEG_INFINITY;
    let mut best_i: u32 = 0;
    for (i, &v) in row.iter().enumerate() {
        if v > best {
            best = v;
            best_i = i as u32;
        }
    }
    Ok(best_i)
}

#[cfg(test)]
mod tests {
    use super::validate_generation_limits;

    #[test]
    fn combined_context_contract_is_checked_and_fail_closed() {
        assert!(validate_generation_limits(32_640, 128, Some(128), Some(32_768)).is_ok());
        assert!(validate_generation_limits(32_767, 1, Some(128), Some(32_768)).is_ok());
        assert!(validate_generation_limits(32_768, 1, Some(128), Some(32_768)).is_err());
        assert!(validate_generation_limits(1, 129, Some(128), Some(32_768)).is_err());
        assert!(validate_generation_limits(usize::MAX, 1, Some(128), Some(usize::MAX)).is_err());
        assert!(validate_generation_limits(0, 1, Some(128), Some(32_768)).is_err());
        assert!(validate_generation_limits(1, 0, Some(128), Some(32_768)).is_err());
    }
}
