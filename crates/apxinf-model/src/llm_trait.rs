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
}

impl<'a> LlmInput<'a> {
    pub const fn text(token_ids: &'a [u32]) -> Self {
        Self {
            token_ids,
            image: None,
        }
    }

    pub const fn with_image(token_ids: &'a [u32], image: ImageInput<'a>) -> Self {
        Self {
            token_ids,
            image: Some(image),
        }
    }
}

/// Input modalities accepted by an LLM implementation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LlmCapabilities {
    pub image: bool,
}

impl LlmCapabilities {
    pub const TEXT_ONLY: Self = Self { image: false };
    pub const VISION: Self = Self { image: true };
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
        self.forward(input.token_ids, 0)
    }

    /// Process a complete prompt for autoregressive generation.
    ///
    /// The returned tensor may contain either every prompt row or only the
    /// final row; the shared generation loop derives the row count from the
    /// tensor itself. Models can override this hook when producing
    /// `[prompt_len, vocab_size]` logits would be needlessly expensive. The
    /// ordinary [`Self::prefill`] and [`Self::forward`] contracts remain
    /// unchanged.
    fn prefill_for_generation(&mut self, input: LlmInput<'_>) -> Result<Tensor> {
        self.prefill(input)
    }

    /// Greedy-decode the first generated token directly from prompt prefill.
    ///
    /// Models with an output-head argmax fast path can override this hook to
    /// avoid materializing a full vocabulary row at the end of prefill. A
    /// returned error is terminal: the prompt may already have advanced model
    /// state, so the shared loop must never fall through and prefill twice.
    fn prefill_token_for_generation(&mut self, _input: LlmInput<'_>) -> Option<Result<u32>> {
        None
    }

    /// Validate request-wide generation limits before any prewarm or prefill.
    ///
    /// This object-safe hook lets models reject a request without partially
    /// mutating caches. Models without a fixed request budget inherit the
    /// no-op implementation.
    fn validate_generation_budget(&self, _prompt_len: usize, _max_new_tokens: usize) -> Result<()> {
        Ok(())
    }

    /// Reset state for a new generation.
    fn reset(&mut self);

    /// Reset state for a new generation and surface backend failures.
    ///
    /// Implementations with fallible cache or accelerator state clearing
    /// should override this hook. The default preserves compatibility with
    /// models whose existing reset is infallible.
    fn reset_checked(&mut self) -> Result<()> {
        self.reset();
        Ok(())
    }

    /// Optional hook called once before prefill, with the prompt length and
    /// the number of tokens that will be generated. Models with a CUDA
    /// decode graph use it to pre-capture every bucket they'll hit so the
    /// per-token TPOT stays at pure graph-replay cost. Default: no-op.
    fn prewarm_decode(&mut self, _prompt_len: usize, _max_new_tokens: usize) {}

    /// Greedy-decode one token directly to its id, skipping the full-logits
    /// D2H + CPU argmax. Returns `None` if the model has no GPU-argmax fast
    /// path (caller falls back to `forward` + `argmax_last_row`).
    fn decode_token(&mut self, _token: u32, _pos: u32) -> Option<Result<u32>> {
        None
    }

    /// Optional machine-readable receipt for explicitly selected generation
    /// paths and their observed hit counts. Implementations should return
    /// `None` when they have no diagnostic runtime path to report.
    fn generation_path_receipt(&self) -> Option<serde_json::Value> {
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
    if prompt_tokens.is_empty() {
        return Err(Error::Other("generate_streaming: empty prompt".into()));
    }
    // Reject unsupported media before graph prewarm or any model forward.
    if input.image.is_some() && !model.capabilities().image {
        return Err(Error::Other(
            "this model does not support image input".into(),
        ));
    }

    let mut profile = GenerationProfile::new();
    if max_new_tokens == 0 {
        profile.finalize(prompt_tokens.len(), 0);
        return Ok((Vec::new(), profile));
    }

    let mut generated = Vec::with_capacity(max_new_tokens);
    let vocab_size = model.vocab_size();

    model.validate_generation_budget(prompt_tokens.len(), max_new_tokens)?;

    // Pre-capture any decode graphs (CUDA) BEFORE prefill so the per-token
    // TPOT below is pure graph replay — keeps capture/instantiate cost in
    // setup (TTFT bucket), not in the steady-state TPOT measurement.
    model.prewarm_decode(prompt_tokens.len(), max_new_tokens);

    // Prefill: process the entire prompt. A direct-token fast path is
    // authoritative once selected because it may already have advanced the
    // model's prompt state.
    let next_token = match model.prefill_token_for_generation(input) {
        Some(result) => result?,
        None => {
            let logits = model.prefill_for_generation(input)?;
            argmax_last_row(&logits, vocab_size)?
        }
    };
    profile.record_first_token();
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
        match model.decode_token(current_token, pos) {
            Some(Ok(tok)) => {
                current_token = tok;
                generated.push(current_token);
                on_token(current_token);
                if eos_token_id == Some(current_token) {
                    break;
                }
                continue;
            }
            Some(Err(error)) => return Err(error),
            None => {}
        }
        let t0 = std::time::Instant::now();
        let logits = model.forward(&[current_token], pos)?;
        t_fwd += t0.elapsed();
        let t1 = std::time::Instant::now();
        current_token = argmax_last_row(&logits, vocab_size)?;
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

/// Extract logits for the last row and return its argmax token.
/// `logits` shape: `[seq_len, vocab_size]`. Logits may live on any device — this
/// helper moves to CPU if needed (callers' responsibility for now).
fn argmax_last_row(logits: &Tensor, vocab_size: usize) -> Result<u32> {
    let dims = logits.shape().dims();
    if dims.len() != 2 || dims[0] == 0 || dims[1] != vocab_size {
        return Err(Error::ShapeMismatch {
            expected: format!("[non-zero rows, {vocab_size}]"),
            got: logits.shape().to_string(),
        });
    }
    let seq_len = dims[0];
    let last_row_offset = (seq_len - 1) * vocab_size;
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
    use super::*;

    struct FailingDecodeModel {
        position: usize,
        fallback_forwards: usize,
        decode_calls: usize,
    }

    struct DirectPrefillModel {
        position: usize,
        direct_prefill_calls: usize,
        logits_prefill_calls: usize,
        fail_direct_prefill: bool,
    }

    impl LlmTrait for DirectPrefillModel {
        fn load(
            _config: ModelConfig,
            _weights: HashMap<String, Tensor>,
            _device: Device,
        ) -> Result<Self> {
            unreachable!("test model is constructed directly")
        }

        fn forward(&mut self, _token_ids: &[u32], _start_pos: u32) -> Result<Tensor> {
            unreachable!("one-token test must not decode")
        }

        fn prefill_for_generation(&mut self, _input: LlmInput<'_>) -> Result<Tensor> {
            self.logits_prefill_calls += 1;
            Err(Error::Other("logits prefill must not run".into()))
        }

        fn prefill_token_for_generation(&mut self, input: LlmInput<'_>) -> Option<Result<u32>> {
            self.direct_prefill_calls += 1;
            self.position += input.token_ids.len();
            Some(if self.fail_direct_prefill {
                Err(Error::Other("injected direct prefill failure".into()))
            } else {
                Ok(3)
            })
        }

        fn reset(&mut self) {
            self.position = 0;
        }

        fn vocab_size(&self) -> usize {
            4
        }
    }

    impl LlmTrait for FailingDecodeModel {
        fn load(
            _config: ModelConfig,
            _weights: HashMap<String, Tensor>,
            _device: Device,
        ) -> Result<Self> {
            unreachable!("test model is constructed directly")
        }

        fn forward(&mut self, _token_ids: &[u32], _start_pos: u32) -> Result<Tensor> {
            self.fallback_forwards += 1;
            self.position += 1;
            Tensor::from_f32(vec![1, 4], &[0.0, 0.0, 1.0, 0.0])
        }

        fn prefill_for_generation(&mut self, input: LlmInput<'_>) -> Result<Tensor> {
            self.position += input.token_ids.len();
            Tensor::from_f32(vec![1, 4], &[0.0, 1.0, 0.0, 0.0])
        }

        fn decode_token(&mut self, _token: u32, _pos: u32) -> Option<Result<u32>> {
            self.decode_calls += 1;
            self.position += 1;
            Some(Err(Error::Other("injected decode failure".into())))
        }

        fn reset(&mut self) {
            self.position = 0;
        }

        fn vocab_size(&self) -> usize {
            4
        }
    }

    #[test]
    fn decode_fast_path_error_does_not_fall_through_or_advance_twice() {
        let mut model = FailingDecodeModel {
            position: 0,
            fallback_forwards: 0,
            decode_calls: 0,
        };
        let result = generate_streaming(&mut model, LlmInput::text(&[3, 2, 1]), 2, |_| {}, None);
        let error = match result {
            Err(error) => error,
            Ok(_) => panic!("decode failure must terminate generation"),
        };
        assert!(error.to_string().contains("injected decode failure"));
        assert_eq!(model.decode_calls, 1);
        assert_eq!(model.fallback_forwards, 0);
        assert_eq!(model.position, 4);
    }

    #[test]
    fn direct_prefill_token_skips_logits_projection_and_advances_once() {
        let mut model = DirectPrefillModel {
            position: 0,
            direct_prefill_calls: 0,
            logits_prefill_calls: 0,
            fail_direct_prefill: false,
        };
        let mut observed = Vec::new();
        let (generated, profile) = generate_streaming(
            &mut model,
            LlmInput::text(&[2, 1, 0]),
            1,
            |token| observed.push(token),
            None,
        )
        .unwrap();

        assert_eq!(generated, vec![3]);
        assert_eq!(observed, generated);
        assert_eq!(model.position, 3);
        assert_eq!(model.direct_prefill_calls, 1);
        assert_eq!(model.logits_prefill_calls, 0);
        assert_eq!(profile.input_tokens(), 3);
        assert_eq!(profile.output_tokens(), 1);
    }

    #[test]
    fn direct_prefill_error_is_terminal_after_state_advancement() {
        let mut model = DirectPrefillModel {
            position: 0,
            direct_prefill_calls: 0,
            logits_prefill_calls: 0,
            fail_direct_prefill: true,
        };
        let result = generate_streaming(&mut model, LlmInput::text(&[2, 1, 0]), 1, |_| {}, None);
        let error = match result {
            Err(error) => error,
            Ok(_) => panic!("direct prefill failure must terminate generation"),
        };

        assert!(error
            .to_string()
            .contains("injected direct prefill failure"));
        assert_eq!(model.position, 3);
        assert_eq!(model.direct_prefill_calls, 1);
        assert_eq!(model.logits_prefill_calls, 0);
    }
}
