//! Hybrid autoregressive state for the Qwen3.5 text stack.
//!
//! Full-attention layers share one compact KV cache indexed by their ordinal
//! (six entries for the 24-layer 0.8B model). Linear-attention layers keep the
//! three canonical `[kernel_size, channels]` convolution histories and the
//! canonical FP32 `[value_heads, key_dim, value_dim]` Gated DeltaNet matrix.

use apxinf_core::{Backend, Error, KvCache, Result, Tensor};

use super::config::{Qwen35LayerType, Qwen35TextConfig};

/// Mutable state for one recurrent linear-attention layer.
#[derive(Default)]
pub struct Qwen35LinearState {
    pub(super) query_conv: Option<Tensor>,
    pub(super) key_conv: Option<Tensor>,
    pub(super) value_conv: Option<Tensor>,
    pub(super) recurrent: Option<Tensor>,
}

impl Qwen35LinearState {
    fn clear(&mut self) {
        self.query_conv = None;
        self.key_conv = None;
        self.value_conv = None;
        self.recurrent = None;
    }

    /// Current canonical `[value_heads, key_dim, value_dim]` recurrent matrix.
    pub fn recurrent(&self) -> Option<&Tensor> {
        self.recurrent.as_ref()
    }

    /// Canonical `[kernel_size, channels]` raw-input histories consumed by the
    /// next causal convolution call, ordered query/key/value.
    pub fn convolution_suffixes(&self) -> [Option<&Tensor>; 3] {
        [
            self.query_conv.as_ref(),
            self.key_conv.as_ref(),
            self.value_conv.as_ref(),
        ]
    }
}

/// Cache and recurrent state for one single-request Qwen3.5 stream.
pub struct Qwen35HybridState {
    pub(super) linear: Vec<Option<Qwen35LinearState>>,
    pub(super) full_cache_indices: Vec<Option<usize>>,
    pub(super) kv: Box<dyn KvCache>,
    position: usize,
    max_context: usize,
}

impl Qwen35HybridState {
    pub fn new(
        config: &Qwen35TextConfig,
        backend: &dyn Backend,
        requested_max_context: usize,
    ) -> Result<Self> {
        if requested_max_context == 0 {
            return Err(Error::Other(
                "qwen3.5: max context must be greater than zero".into(),
            ));
        }
        let max_context = requested_max_context.min(config.max_position_embeddings);
        let mut linear = Vec::with_capacity(config.n_layers);
        let mut full_cache_indices = Vec::with_capacity(config.n_layers);
        let mut full_count = 0usize;
        for layer_type in &config.layer_types {
            match layer_type {
                Qwen35LayerType::LinearAttention => {
                    linear.push(Some(Qwen35LinearState::default()));
                    full_cache_indices.push(None);
                }
                Qwen35LayerType::FullAttention => {
                    linear.push(None);
                    full_cache_indices.push(Some(full_count));
                    full_count += 1;
                }
            }
        }
        let kv =
            backend.create_kv_cache(full_count, config.n_kv_heads, config.head_dim, max_context);
        Ok(Self {
            linear,
            full_cache_indices,
            kv,
            position: 0,
            max_context,
        })
    }

    pub fn position(&self) -> usize {
        self.position
    }

    pub fn max_context(&self) -> usize {
        self.max_context
    }

    pub fn full_attention_layers(&self) -> usize {
        self.kv.n_layers()
    }

    pub fn linear_state(&self, layer_index: usize) -> Option<&Qwen35LinearState> {
        self.linear.get(layer_index).and_then(Option::as_ref)
    }

    pub(super) fn linear_state_mut(
        &mut self,
        layer_index: usize,
    ) -> Result<&mut Qwen35LinearState> {
        self.linear
            .get_mut(layer_index)
            .and_then(Option::as_mut)
            .ok_or_else(|| {
                Error::Other(format!(
                    "qwen3.5: layer {layer_index} has no linear-attention state"
                ))
            })
    }

    pub(super) fn full_cache_index(&self, layer_index: usize) -> Result<usize> {
        self.full_cache_indices
            .get(layer_index)
            .copied()
            .flatten()
            .ok_or_else(|| {
                Error::Other(format!(
                    "qwen3.5: layer {layer_index} has no full-attention KV cache"
                ))
            })
    }

    pub(super) fn validate_forward(&self, start_pos: u32, sequence_length: usize) -> Result<()> {
        let start_pos = usize::try_from(start_pos)
            .map_err(|_| Error::Other("qwen3.5: start_pos exceeds usize".into()))?;
        if start_pos != self.position {
            return Err(Error::Other(format!(
                "qwen3.5: start_pos {start_pos} does not match cached position {}",
                self.position
            )));
        }
        if self.kv.seq_len() != self.position {
            return Err(Error::Other(format!(
                "qwen3.5: KV position {} disagrees with hybrid position {}",
                self.kv.seq_len(),
                self.position
            )));
        }
        let end = start_pos
            .checked_add(sequence_length)
            .ok_or_else(|| Error::Other("qwen3.5: context length overflow".into()))?;
        if end > self.max_context {
            return Err(Error::Other(format!(
                "qwen3.5: context end {end} exceeds configured maximum {}",
                self.max_context
            )));
        }
        Ok(())
    }

    pub(super) fn advance(&mut self, sequence_length: usize) {
        self.kv.advance(sequence_length);
        self.position += sequence_length;
    }

    pub fn reset(&mut self) -> Result<()> {
        self.kv.clear()?;
        for state in self.linear.iter_mut().flatten() {
            state.clear();
        }
        self.position = 0;
        Ok(())
    }
}
