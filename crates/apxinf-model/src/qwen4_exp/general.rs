//! Correctness-first single-request Qwen4-Exp text runtime.
//!
//! This first slice accepts deterministic downsized synthetic weights only.
//! It executes the released architecture and cache semantics; it is not a
//! production checkpoint loader or performance path.

use std::collections::HashMap;
use std::sync::Arc;

use apxinf_core::{Backend, CpuBackend, DType, Device, Error, Result, Tensor};
use apxinf_loader::ModelConfig;
use rayon::prelude::*;

use super::config::{Qwen4ExpConfig, Qwen4ExpLayerType, Qwen4ExpTextConfig};
use super::qsa::{apply_partial_mrope, rms_norm_zero_centered_into, Qwen4ExpQsaSelector};
use super::weights::{metadata_from_tensors, ple_vocab_layout, Qwen4ExpWeightSchema};
use crate::llm_trait::{LlmCapabilities, LlmInput, LlmTrait};
use crate::qwen3vl::config::{Qwen3VLConfig, Qwen3VLTextConfig, Qwen3VLVisionConfig};
use crate::qwen3vl::Qwen3VLVisionWeights;

pub struct GeneralQwen4Exp {
    config: Qwen4ExpConfig,
    weights: RuntimeWeights,
    state: RuntimeState,
    backend: Arc<dyn Backend>,
    max_context: usize,
    weight_source: &'static str,
    checkpoint_payloads_mmap: bool,
    vision: Option<Qwen4ExpVisionRuntime>,
    rope_delta: i64,
}

struct Qwen4ExpVisionRuntime {
    config: Qwen3VLConfig,
    weights: Qwen3VLVisionWeights,
}

pub fn encode_qwen4_exp_vision(
    config: &Qwen4ExpConfig,
    tensors: HashMap<String, Tensor>,
    pixel_values: &Tensor,
    grid_thw: &[[u32; 3]],
) -> Result<Tensor> {
    let vision = build_vision_runtime(config, tensors)?;
    Ok(crate::qwen3vl::vision::forward(
        &vision.config,
        &vision.weights,
        &CpuBackend,
        pixel_values,
        grid_thw,
    )?
    .primary)
}

impl GeneralQwen4Exp {
    pub fn from_tensors(
        config: Qwen4ExpConfig,
        tensors: HashMap<String, Tensor>,
        requested_max_context: usize,
    ) -> Result<Self> {
        Self::from_tensors_with_backend(
            config,
            tensors,
            requested_max_context,
            Arc::new(CpuBackend),
        )
    }

    pub(crate) fn from_tensors_with_backend(
        config: Qwen4ExpConfig,
        tensors: HashMap<String, Tensor>,
        requested_max_context: usize,
        backend: Arc<dyn Backend>,
    ) -> Result<Self> {
        if backend.device() != Device::Cpu {
            return Err(Error::Other(
                "qwen4-exp checkpoint runtime currently supports CPU only".into(),
            ));
        }
        if requested_max_context == 0 {
            return Err(Error::Other(
                "qwen4-exp checkpoint max_context must be positive".into(),
            ));
        }
        let schema = Qwen4ExpWeightSchema::new(&config)?;
        schema.validate_runtime_metadata(&metadata_from_tensors(&tensors))?;
        let checkpoint_payloads_mmap = tensors
            .values()
            .all(|tensor| matches!(tensor.storage(), apxinf_core::Storage::CpuMmap { .. }));
        let weights = RuntimeWeights::from_tensors(&config.text, tensors)?;
        let state = RuntimeState::new(&config.text);
        let max_context = requested_max_context.min(config.text.max_position_embeddings);
        Ok(Self {
            config,
            weights,
            state,
            backend,
            max_context,
            weight_source: "checkpoint",
            checkpoint_payloads_mmap,
            vision: None,
            rope_delta: 0,
        })
    }

    pub fn from_synthetic(
        config: Qwen4ExpConfig,
        seed: u64,
        requested_max_context: usize,
    ) -> Result<Self> {
        Self::from_synthetic_with_backend(config, seed, requested_max_context, Arc::new(CpuBackend))
    }

    pub(crate) fn from_synthetic_with_backend(
        config: Qwen4ExpConfig,
        seed: u64,
        requested_max_context: usize,
        backend: Arc<dyn Backend>,
    ) -> Result<Self> {
        validate_synthetic_size(&config.text)?;
        if backend.device() != Device::Cpu {
            return Err(Error::Other(
                "qwen4-exp synthetic runtime currently supports CPU only".into(),
            ));
        }
        if requested_max_context == 0 {
            return Err(Error::Other(
                "qwen4-exp synthetic max_context must be positive".into(),
            ));
        }
        let max_context = requested_max_context.min(config.text.max_position_embeddings);
        let weights = RuntimeWeights::synthetic(&config.text, seed)?;
        let state = RuntimeState::new(&config.text);
        Ok(Self {
            config,
            weights,
            state,
            backend,
            max_context,
            weight_source: "deterministic-synthetic",
            checkpoint_payloads_mmap: false,
            vision: None,
            rope_delta: 0,
        })
    }

    pub(crate) fn with_vision_tensors(mut self, tensors: HashMap<String, Tensor>) -> Result<Self> {
        self.vision = Some(build_vision_runtime(&self.config, tensors)?);
        Ok(self)
    }

    pub fn encode_image(&self, pixel_values: &Tensor, grid_thw: &[[u32; 3]]) -> Result<Tensor> {
        let vision = self
            .vision
            .as_ref()
            .ok_or_else(|| Error::Other("qwen4-exp vision weights are not loaded".into()))?;
        let pixels = if pixel_values.dtype() == DType::F32 {
            pixel_values.clone()
        } else {
            Tensor::from_f32(
                pixel_values.shape().dims().to_vec(),
                &pixel_values.to_f32_vec()?,
            )?
        };
        Ok(crate::qwen3vl::vision::forward(
            &vision.config,
            &vision.weights,
            &*self.backend,
            &pixels,
            grid_thw,
        )?
        .primary)
    }

    fn advance_one(
        &mut self,
        token: u32,
        position: [u32; 3],
        embedding_override: Option<&[f32]>,
    ) -> Result<Vec<f32>> {
        let text = &self.config.text;
        if token as usize >= text.vocab_size {
            return Err(Error::Other(format!(
                "qwen4-exp token {token} is outside vocabulary {}",
                text.vocab_size
            )));
        }
        let embedding = match embedding_override {
            Some(embedding) if embedding.len() == text.hidden_size => embedding.to_vec(),
            Some(embedding) => {
                return Err(Error::ShapeMismatch {
                    expected: format!("[{}] visual embedding", text.hidden_size),
                    got: format!("[{}]", embedding.len()),
                })
            }
            None => self
                .weights
                .token_embedding
                .row(token as usize, text.hidden_size)?,
        };
        let mut hyper = Vec::with_capacity(text.hc_count * text.hidden_size);
        for _ in 0..text.hc_count {
            hyper.extend_from_slice(&embedding);
        }

        for layer_index in 0..text.n_layers {
            let weights = &self.weights.layers[layer_index];
            let state = &mut self.state.layers[layer_index];
            if let (Some(ple_weights), Some(ple_state)) = (&weights.ple, &mut state.ple) {
                let ple = run_ple(text, &hyper, token, ple_weights, ple_state)?;
                add_assign(&mut hyper, &ple);
            }

            let attention_mix =
                run_gated_residual(text, &hyper, &weights.attention_connection, true)?;
            let attention_output = match (&weights.attention, &mut state.attention) {
                (AttentionWeights::Linear(weights), AttentionState::Linear(state)) => {
                    run_gdn(&*self.backend, text, &attention_mix.mixed, weights, state)?
                }
                (AttentionWeights::Qsa(weights), AttentionState::Qsa(state)) => {
                    run_qsa(text, &attention_mix.mixed, position, weights, state)?
                }
                _ => {
                    return Err(Error::Other(format!(
                        "qwen4-exp attention state mismatch at layer {layer_index}"
                    )))
                }
            };
            hyper = inject(&hyper, &attention_output, &attention_mix.injection, text);

            let mlp_mix = run_gated_residual(text, &hyper, &weights.mlp_connection, true)?;
            let mlp_output = run_moe(text, &mlp_mix.mixed, &weights.moe)?;
            hyper = inject(&hyper, &mlp_output, &mlp_mix.injection, text);
        }

        let hidden = run_gated_residual(text, &hyper, &self.weights.final_connection, false)?.mixed;
        Ok(hidden)
    }

    fn project_hidden(&self, hidden: &[f32]) -> Result<Vec<f32>> {
        self.weights.lm_head.apply(hidden)
    }

    fn forward_rows(
        &mut self,
        token_ids: &[u32],
        start_pos: u32,
        project_all: bool,
    ) -> Result<Tensor> {
        if token_ids.is_empty() {
            return Err(Error::Other("qwen4-exp forward requires tokens".into()));
        }
        let start_pos = usize::try_from(start_pos)
            .map_err(|_| Error::Other("qwen4-exp start_pos exceeds usize".into()))?;
        if start_pos != self.state.position {
            return Err(Error::Other(format!(
                "qwen4-exp start_pos {start_pos} does not match cached position {}",
                self.state.position
            )));
        }
        let end = start_pos
            .checked_add(token_ids.len())
            .ok_or_else(|| Error::Other("qwen4-exp context length overflow".into()))?;
        if end > self.max_context {
            return Err(Error::Other(format!(
                "qwen4-exp context end {end} exceeds maximum {}",
                self.max_context
            )));
        }
        let output_rows = if project_all { token_ids.len() } else { 1 };
        let mut logits = Vec::with_capacity(output_rows * self.config.text.vocab_size);
        for (offset, &token) in token_ids.iter().enumerate() {
            let position = (start_pos + offset) as i64 + self.rope_delta;
            let position = u32::try_from(position).map_err(|_| {
                Error::Other(format!(
                    "qwen4-exp effective position {position} is outside u32"
                ))
            })?;
            let hidden = self.advance_one(token, [position; 3], None)?;
            if project_all || offset + 1 == token_ids.len() {
                logits.extend(self.project_hidden(&hidden)?);
            }
            self.state.position += 1;
        }
        Tensor::from_f32(vec![output_rows, self.config.text.vocab_size], &logits)
    }
}

fn build_vision_runtime(
    config: &Qwen4ExpConfig,
    mut tensors: HashMap<String, Tensor>,
) -> Result<Qwen4ExpVisionRuntime> {
    for tensor in tensors.values_mut() {
        if tensor.dtype() != DType::F32 {
            *tensor = Tensor::from_f32(tensor.shape().dims().to_vec(), &tensor.to_f32_vec()?)?;
        }
    }
    let config = qwen3vl_vision_adapter(config);
    let weights = Qwen3VLVisionWeights::from_map(&config, tensors)?;
    Ok(Qwen4ExpVisionRuntime { config, weights })
}

fn multimodal_positions(
    token_ids: &[u32],
    image_token_id: u32,
    grids: &[[u32; 3]],
    merge: usize,
) -> Result<(Vec<[u32; 3]>, i64)> {
    let mut positions = Vec::with_capacity(token_ids.len());
    let mut current = 0u32;
    let mut token_index = 0usize;
    let mut grid_index = 0usize;
    while token_index < token_ids.len() {
        let is_image = token_ids[token_index] == image_token_id;
        let end = token_ids[token_index..]
            .iter()
            .position(|token| (*token == image_token_id) != is_image)
            .map_or(token_ids.len(), |offset| token_index + offset);
        if !is_image {
            for offset in 0..end - token_index {
                positions.push([current + offset as u32; 3]);
            }
            current += (end - token_index) as u32;
        } else {
            let grid = grids.get(grid_index).ok_or_else(|| {
                Error::Other("qwen4-exp image token group has no grid_thw entry".into())
            })?;
            grid_index += 1;
            let (time, height, width) = (
                grid[0] as usize,
                grid[1] as usize / merge,
                grid[2] as usize / merge,
            );
            let expected = time * height * width;
            if end - token_index != expected {
                return Err(Error::Other(format!(
                    "qwen4-exp image placeholder group has {} tokens, grid requires {expected}",
                    end - token_index
                )));
            }
            for temporal in 0..time {
                for row in 0..height {
                    for column in 0..width {
                        positions.push([
                            current + temporal as u32,
                            current + row as u32,
                            current + column as u32,
                        ]);
                    }
                }
            }
            current += height.max(width) as u32;
        }
        token_index = end;
    }
    if grid_index != grids.len() {
        return Err(Error::Other(format!(
            "qwen4-exp received {} image grids for {grid_index} token groups",
            grids.len()
        )));
    }
    let max_position = positions
        .iter()
        .flat_map(|position| position.iter())
        .copied()
        .max()
        .unwrap_or(0);
    let delta = max_position as i64 + 1 - token_ids.len() as i64;
    Ok((positions, delta))
}

impl GeneralQwen4Exp {
    fn prefill_rows(&mut self, input: LlmInput<'_>, project_all: bool) -> Result<Tensor> {
        let (pixel_values, encoder_grids, placeholder_token, position_grids, media_label) =
            match (input.image, input.video) {
                (None, None) => return self.forward_rows(input.token_ids, 0, project_all),
                (Some(_), Some(_)) => {
                    return Err(Error::Other(
                        "qwen4-exp accepts image or video in one request, not both".into(),
                    ))
                }
                (Some(image), None) => (
                    image.pixel_values,
                    image.grid_thw,
                    self.config.image_token_id,
                    image.grid_thw.to_vec(),
                    "image",
                ),
                (None, Some(video)) => {
                    let mut frame_grids = Vec::new();
                    for &[time, height, width] in video.grid_thw {
                        frame_grids.extend((0..time).map(|_| [1, height, width]));
                    }
                    (
                        video.pixel_values,
                        video.grid_thw,
                        self.config.video_token_id,
                        frame_grids,
                        "video",
                    )
                }
            };
        if self.state.position != 0 {
            return Err(Error::Other(
                "qwen4-exp multimodal prefill requires empty state".into(),
            ));
        }
        if input.token_ids.is_empty() || input.token_ids.len() > self.max_context {
            return Err(Error::Other(
                "qwen4-exp multimodal prompt length is invalid".into(),
            ));
        }
        let visual = self.encode_image(pixel_values, encoder_grids)?;
        let visual_dims = visual.shape().dims();
        if visual_dims.len() != 2 || visual_dims[1] != self.config.text.hidden_size {
            return Err(Error::ShapeMismatch {
                expected: format!("[image_tokens, {}]", self.config.text.hidden_size),
                got: visual.shape().to_string(),
            });
        }
        let placeholder_tokens = input
            .token_ids
            .iter()
            .filter(|token| **token == placeholder_token)
            .count();
        if placeholder_tokens != visual_dims[0] {
            return Err(Error::Other(format!(
                "qwen4-exp {media_label} features {} != placeholder tokens {placeholder_tokens}",
                visual_dims[0]
            )));
        }
        let (positions, rope_delta) = multimodal_positions(
            input.token_ids,
            placeholder_token,
            &position_grids,
            self.config.vision.spatial_merge_size,
        )?;
        self.rope_delta = rope_delta;
        let visual = visual.as_f32()?;
        let mut visual_row = 0usize;
        let output_rows = if project_all {
            input.token_ids.len()
        } else {
            1
        };
        let mut logits = Vec::with_capacity(output_rows * self.config.text.vocab_size);
        for (offset, (&token, position)) in input.token_ids.iter().zip(positions).enumerate() {
            let embedding = if token == placeholder_token {
                let start = visual_row * self.config.text.hidden_size;
                visual_row += 1;
                Some(&visual[start..start + self.config.text.hidden_size])
            } else {
                None
            };
            let hidden = self.advance_one(token, position, embedding)?;
            if project_all || offset + 1 == input.token_ids.len() {
                logits.extend(self.project_hidden(&hidden)?);
            }
            self.state.position += 1;
        }
        Tensor::from_f32(vec![output_rows, self.config.text.vocab_size], &logits)
    }
}

impl LlmTrait for GeneralQwen4Exp {
    fn load(
        _config: ModelConfig,
        _weights: HashMap<String, Tensor>,
        _device: Device,
    ) -> Result<Self> {
        Err(Error::Other(
            "Qwen4-Exp uses a nested config; use AutoModel or from_synthetic".into(),
        ))
    }

    fn forward(&mut self, token_ids: &[u32], start_pos: u32) -> Result<Tensor> {
        self.forward_rows(token_ids, start_pos, true)
    }

    fn backend(&self) -> &dyn Backend {
        &*self.backend
    }

    fn capabilities(&self) -> LlmCapabilities {
        LlmCapabilities {
            image: self.vision.is_some(),
            video: self.vision.is_some(),
        }
    }

    fn prefill(&mut self, input: LlmInput<'_>) -> Result<Tensor> {
        self.prefill_rows(input, true)
    }

    fn prefill_for_generation(&mut self, input: LlmInput<'_>) -> Result<Tensor> {
        self.prefill_rows(input, false)
    }

    fn reset(&mut self) {
        self.state = RuntimeState::new(&self.config.text);
        self.rope_delta = 0;
    }

    fn generation_path_receipt(&self) -> Option<serde_json::Value> {
        Some(serde_json::json!({
            "format": "apxinf-qwen4-exp-synthetic-text-v1",
            "weights": self.weight_source,
            "checkpoint_payloads_mmap": self.checkpoint_payloads_mmap,
            "device": "cpu-f32",
            "position": self.state.position,
            "layers": self.config.text.n_layers,
            "qsa": true,
            "gated_delta_net": true,
            "gated_residual_streams": self.config.text.hc_count,
            "routed_moe": true,
            "ple_layers": self.config.text.ple_layer_ids,
            "vision": self.vision.is_some(),
            "video": self.vision.is_some(),
            "vision_encoder_loaded": self.vision.is_some(),
            "multimodal_prefill": self.vision.is_some(),
            "generation_prefill_logits_rows": 1,
            "rope_delta": self.rope_delta,
            "mtp": false,
        }))
    }

    fn vocab_size(&self) -> usize {
        self.config.text.vocab_size
    }
}

struct RuntimeWeights {
    token_embedding: ResidentEmbedding,
    layers: Vec<LayerWeights>,
    final_connection: GatedResidualWeights,
    lm_head: ResidentMatrix,
}

struct LayerWeights {
    attention_connection: GatedResidualWeights,
    attention: AttentionWeights,
    mlp_connection: GatedResidualWeights,
    moe: MoeWeights,
    ple: Option<PleWeights>,
}

struct GatedResidualWeights {
    norm: Vec<f32>,
    down: ResidentMatrix,
    up: ResidentMatrix,
    inject: Option<ResidentMatrix>,
}

enum AttentionWeights {
    Linear(Box<GdnWeights>),
    Qsa(Box<QsaWeights>),
}

struct GdnWeights {
    in_qkv: ResidentMatrix,
    in_z: ResidentMatrix,
    in_a: ResidentMatrix,
    in_b: ResidentMatrix,
    conv: Tensor,
    a_log: Tensor,
    dt_bias: Tensor,
    norm: Tensor,
    out: ResidentMatrix,
}

struct QsaWeights {
    query_gate: QueryGateWeights,
    k: ResidentMatrix,
    v: ResidentMatrix,
    out: ResidentMatrix,
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
    index_qk: ResidentMatrix,
    index_q_norm: Vec<f32>,
    index_k_norm: Vec<f32>,
    selector: Qwen4ExpQsaSelector,
}

struct MoeWeights {
    router: ResidentMatrix,
    experts: ExpertWeights,
    shared_gate: ResidentMatrix,
    shared_up: ResidentMatrix,
    shared_down: ResidentMatrix,
    shared_expert_gate: ResidentMatrix,
}

enum ExpertWeights {
    Synthetic {
        gate_up: Vec<Vec<f32>>,
        down: Vec<Vec<f32>>,
    },
    Checkpoint {
        gate_up: Tensor,
        down: Tensor,
    },
}

struct PleWeights {
    embedding: PleEmbedding,
    head_vocab_sizes: Vec<usize>,
    head_offsets: Vec<usize>,
    multipliers: Vec<i64>,
    head_dim: usize,
    key: ResidentMatrix,
    value: ResidentMatrix,
    norm_key: Vec<f32>,
    norm_query: Vec<f32>,
    norm_conv: Vec<f32>,
    conv: Vec<f32>,
}

enum ResidentEmbedding {
    Synthetic(Vec<f32>),
    Checkpoint(Tensor),
}

enum ResidentMatrix {
    Synthetic {
        values: Vec<f32>,
        output_size: usize,
    },
    Checkpoint(Tensor),
}

enum QueryGateWeights {
    Synthetic {
        query: ResidentMatrix,
        gate: ResidentMatrix,
    },
    Checkpoint(Tensor),
}

enum PleEmbedding {
    Synthetic(Vec<f32>),
    Checkpoint {
        shards: Vec<Tensor>,
        row_ends: Vec<usize>,
    },
}

impl ResidentEmbedding {
    fn row(&self, row: usize, columns: usize) -> Result<Vec<f32>> {
        let mut output = Vec::with_capacity(columns);
        match self {
            Self::Synthetic(values) => {
                output.extend_from_slice(&values[row * columns..(row + 1) * columns]);
            }
            Self::Checkpoint(tensor) => {
                append_tensor_values(tensor, row * columns, (row + 1) * columns, &mut output)?;
            }
        }
        Ok(output)
    }
}

impl ResidentMatrix {
    fn synthetic(values: Vec<f32>, output_size: usize) -> Self {
        Self::Synthetic {
            values,
            output_size,
        }
    }

    fn apply(&self, input: &[f32]) -> Result<Vec<f32>> {
        match self {
            Self::Synthetic {
                values,
                output_size,
            } => Ok(linear(input, values, *output_size)),
            Self::Checkpoint(tensor) => {
                let shape = tensor.shape().dims();
                linear_checkpoint_matrix(input, tensor, 0, shape[0], shape[1])
            }
        }
    }

    fn bf16_checkpoint_numel(&self) -> Option<usize> {
        match self {
            Self::Checkpoint(tensor) if tensor.dtype() == DType::BF16 => Some(tensor.numel()),
            _ => None,
        }
    }

    fn is_large_bf16_checkpoint(&self) -> bool {
        self.bf16_checkpoint_numel()
            .is_some_and(|numel| numel >= BF16_GEMV_PAR_MIN_ELEMENTS)
    }
}

impl QueryGateWeights {
    fn apply(&self, config: &Qwen4ExpTextConfig, hidden: &[f32]) -> Result<(Vec<f32>, Vec<f32>)> {
        match self {
            Self::Synthetic { query, gate } => Ok((query.apply(hidden)?, gate.apply(hidden)?)),
            Self::Checkpoint(tensor) => {
                let packed = linear_checkpoint_matrix(
                    hidden,
                    tensor,
                    0,
                    config.full_q_projection_width(),
                    config.hidden_size,
                )?;
                let mut query = Vec::with_capacity(config.full_query_width());
                let mut gate = Vec::with_capacity(config.full_query_width());
                for head in 0..config.n_attention_heads {
                    let start = head * 2 * config.head_dim;
                    query.extend_from_slice(&packed[start..start + config.head_dim]);
                    gate.extend_from_slice(
                        &packed[start + config.head_dim..start + 2 * config.head_dim],
                    );
                }
                Ok((query, gate))
            }
        }
    }
}

impl ExpertWeights {
    fn apply(
        &self,
        config: &Qwen4ExpTextConfig,
        hidden: &[f32],
        expert: usize,
    ) -> Result<Vec<f32>> {
        let gate_up = self.gate_up(config, hidden, expert)?;
        let activated = activate_expert(&gate_up, config.moe_intermediate_size);
        self.down(config, &activated, expert)
    }

    fn is_bf16_checkpoint(&self) -> bool {
        matches!(
            self,
            Self::Checkpoint { gate_up, down }
                if gate_up.dtype() == DType::BF16 && down.dtype() == DType::BF16
        )
    }

    fn apply_bf16_serial(
        &self,
        config: &Qwen4ExpTextConfig,
        hidden: &[f32],
        expert: usize,
    ) -> Result<Vec<f32>> {
        let Self::Checkpoint { gate_up, down } = self else {
            return Err(Error::Other(
                "qwen4-exp serial BF16 expert requires checkpoint weights".into(),
            ));
        };
        let gate_up = linear_checkpoint_bf16_serial(
            hidden,
            gate_up,
            expert * 2 * config.moe_intermediate_size * config.hidden_size,
            2 * config.moe_intermediate_size,
            config.hidden_size,
        )?;
        let activated = activate_expert(&gate_up, config.moe_intermediate_size);
        linear_checkpoint_bf16_serial(
            &activated,
            down,
            expert * config.hidden_size * config.moe_intermediate_size,
            config.hidden_size,
            config.moe_intermediate_size,
        )
    }

    fn gate_up(
        &self,
        config: &Qwen4ExpTextConfig,
        hidden: &[f32],
        expert: usize,
    ) -> Result<Vec<f32>> {
        match self {
            Self::Synthetic { gate_up, .. } => Ok(linear(
                hidden,
                &gate_up[expert],
                2 * config.moe_intermediate_size,
            )),
            Self::Checkpoint { gate_up, .. } => linear_checkpoint_matrix(
                hidden,
                gate_up,
                expert * 2 * config.moe_intermediate_size * config.hidden_size,
                2 * config.moe_intermediate_size,
                config.hidden_size,
            ),
        }
    }

    fn down(&self, config: &Qwen4ExpTextConfig, hidden: &[f32], expert: usize) -> Result<Vec<f32>> {
        match self {
            Self::Synthetic { down, .. } => Ok(linear(hidden, &down[expert], config.hidden_size)),
            Self::Checkpoint { down, .. } => linear_checkpoint_matrix(
                hidden,
                down,
                expert * config.hidden_size * config.moe_intermediate_size,
                config.hidden_size,
                config.moe_intermediate_size,
            ),
        }
    }
}

fn activate_expert(gate_up: &[f32], intermediate_size: usize) -> Vec<f32> {
    (0..intermediate_size)
        .map(|index| silu(gate_up[index]) * gate_up[intermediate_size + index])
        .collect()
}

impl PleEmbedding {
    fn extend_row(&self, row: usize, columns: usize, output: &mut Vec<f32>) -> Result<()> {
        match self {
            Self::Synthetic(values) => {
                output.extend_from_slice(&values[row * columns..(row + 1) * columns]);
            }
            Self::Checkpoint { shards, row_ends } => {
                let shard_index = row_ends.partition_point(|end| row >= *end);
                let previous_end = shard_index
                    .checked_sub(1)
                    .map_or(0, |index| row_ends[index]);
                let local_row = row - previous_end;
                let tensor = shards.get(shard_index).ok_or_else(|| {
                    Error::Other(format!("qwen4-exp PLE row {row} is outside shards"))
                })?;
                append_tensor_values(
                    tensor,
                    local_row * columns,
                    local_row * columns + columns,
                    output,
                )?;
            }
        }
        Ok(())
    }
}

struct RuntimeState {
    layers: Vec<LayerState>,
    position: usize,
}

struct LayerState {
    attention: AttentionState,
    ple: Option<PleState>,
}

enum AttentionState {
    Linear(GdnState),
    Qsa(QsaState),
}

#[derive(Default)]
struct GdnState {
    conv: Option<Tensor>,
    recurrent: Option<Tensor>,
}

#[derive(Default)]
struct QsaState {
    raw_keys: Vec<f32>,
    keys: Vec<f32>,
    values: Vec<f32>,
    positions: Vec<[u32; 3]>,
}

struct PleState {
    context: Vec<u32>,
    conv_history: Vec<f32>,
}

impl RuntimeState {
    fn new(config: &Qwen4ExpTextConfig) -> Self {
        let mut layers = Vec::with_capacity(config.n_layers);
        for (index, layer_type) in config.layer_types.iter().enumerate() {
            let attention = match layer_type {
                Qwen4ExpLayerType::LinearAttention => AttentionState::Linear(GdnState::default()),
                Qwen4ExpLayerType::QwenSparseAttention => AttentionState::Qsa(QsaState::default()),
            };
            let ple = config
                .ple_layer_ids
                .contains(&(index + 1))
                .then(|| PleState {
                    context: vec![config.eos_token_id; config.ngram_size - 1],
                    conv_history: vec![
                        0.0;
                        (config.ple_conv_kernel_size - 1)
                            * config.ngram_size
                            * config.hc_count
                            * config.hidden_size
                    ],
                });
            layers.push(LayerState { attention, ple });
        }
        Self {
            layers,
            position: 0,
        }
    }
}

impl RuntimeWeights {
    fn from_tensors(
        config: &Qwen4ExpTextConfig,
        mut tensors: HashMap<String, Tensor>,
    ) -> Result<Self> {
        let token_embedding = ResidentEmbedding::Checkpoint(take_tensor(
            &mut tensors,
            "model.language_model.embed_tokens.weight",
        )?);
        let lm_head = ResidentMatrix::Checkpoint(take_tensor(&mut tensors, "lm_head.weight")?);
        let final_connection = GatedResidualWeights::from_tensors(
            config,
            &mut tensors,
            "model.language_model.hyper_connection_mixer",
            false,
        )?;
        let mut layers = Vec::with_capacity(config.n_layers);
        for (layer_index, layer_type) in config.layer_types.iter().copied().enumerate() {
            let prefix = format!("model.language_model.layers.{layer_index}");
            let attention_connection = GatedResidualWeights::from_tensors(
                config,
                &mut tensors,
                &format!("{prefix}.attn_hyper_connection"),
                true,
            )?;
            let attention = match layer_type {
                Qwen4ExpLayerType::LinearAttention => AttentionWeights::Linear(Box::new(
                    GdnWeights::from_tensors(config, &mut tensors, &prefix)?,
                )),
                Qwen4ExpLayerType::QwenSparseAttention => AttentionWeights::Qsa(Box::new(
                    QsaWeights::from_tensors(config, &mut tensors, &prefix)?,
                )),
            };
            let mlp_connection = GatedResidualWeights::from_tensors(
                config,
                &mut tensors,
                &format!("{prefix}.mlp_hyper_connection"),
                true,
            )?;
            let moe = MoeWeights::from_tensors(config, &mut tensors, &prefix)?;
            let ple = if config.ple_layer_ids.contains(&(layer_index + 1)) {
                Some(PleWeights::from_tensors(
                    config,
                    &mut tensors,
                    &prefix,
                    layer_index,
                )?)
            } else {
                None
            };
            layers.push(LayerWeights {
                attention_connection,
                attention,
                mlp_connection,
                moe,
                ple,
            });
        }
        if !tensors.is_empty() {
            let mut names = tensors.into_keys().collect::<Vec<_>>();
            names.sort_unstable();
            return Err(Error::Other(format!(
                "qwen4-exp checkpoint packer left unconsumed tensor {}",
                names[0]
            )));
        }
        Ok(Self {
            token_embedding,
            layers,
            final_connection,
            lm_head,
        })
    }

    fn synthetic(config: &Qwen4ExpTextConfig, seed: u64) -> Result<Self> {
        let mut rng = SyntheticRng::new(seed);
        let hidden = config.hidden_size;
        let token_embedding = ResidentEmbedding::Synthetic(rng.matrix(config.vocab_size, hidden));
        let lm_head =
            ResidentMatrix::synthetic(rng.matrix(hidden, config.vocab_size), config.vocab_size);
        let mut layers = Vec::with_capacity(config.n_layers);
        for (index, layer_type) in config.layer_types.iter().enumerate() {
            let attention = match layer_type {
                Qwen4ExpLayerType::LinearAttention => {
                    let qkv = 2 * config.linear_key_width() + config.linear_value_width();
                    let mut conv = vec![0.0; qkv * config.linear_conv_kernel_dim];
                    for channel in 0..qkv {
                        conv[channel * config.linear_conv_kernel_dim
                            + config.linear_conv_kernel_dim
                            - 1] = 1.0;
                    }
                    AttentionWeights::Linear(Box::new(GdnWeights {
                        in_qkv: ResidentMatrix::synthetic(rng.matrix(hidden, qkv), qkv),
                        in_z: ResidentMatrix::synthetic(
                            rng.matrix(hidden, config.linear_value_width()),
                            config.linear_value_width(),
                        ),
                        in_a: ResidentMatrix::synthetic(
                            rng.matrix(hidden, config.linear_num_value_heads),
                            config.linear_num_value_heads,
                        ),
                        in_b: ResidentMatrix::synthetic(
                            rng.matrix(hidden, config.linear_num_value_heads),
                            config.linear_num_value_heads,
                        ),
                        conv: Tensor::from_f32(vec![qkv, config.linear_conv_kernel_dim], &conv)?,
                        a_log: Tensor::from_f32(
                            vec![config.linear_num_value_heads],
                            &vec![-0.7; config.linear_num_value_heads],
                        )?,
                        dt_bias: Tensor::from_f32(
                            vec![config.linear_num_value_heads],
                            &vec![0.0; config.linear_num_value_heads],
                        )?,
                        norm: Tensor::from_f32(
                            vec![config.linear_value_head_dim],
                            &vec![1.0; config.linear_value_head_dim],
                        )?,
                        out: ResidentMatrix::synthetic(
                            rng.matrix(config.linear_value_width(), hidden),
                            hidden,
                        ),
                    }))
                }
                Qwen4ExpLayerType::QwenSparseAttention => {
                    AttentionWeights::Qsa(Box::new(QsaWeights {
                        query_gate: QueryGateWeights::Synthetic {
                            query: ResidentMatrix::synthetic(
                                rng.matrix(hidden, config.full_query_width()),
                                config.full_query_width(),
                            ),
                            gate: ResidentMatrix::synthetic(
                                rng.matrix(hidden, config.full_query_width()),
                                config.full_query_width(),
                            ),
                        },
                        k: ResidentMatrix::synthetic(
                            rng.matrix(hidden, config.full_kv_width()),
                            config.full_kv_width(),
                        ),
                        v: ResidentMatrix::synthetic(
                            rng.matrix(hidden, config.full_kv_width()),
                            config.full_kv_width(),
                        ),
                        out: ResidentMatrix::synthetic(
                            rng.matrix(config.full_query_width(), hidden),
                            hidden,
                        ),
                        q_norm: vec![0.0; config.head_dim],
                        k_norm: vec![0.0; config.head_dim],
                        index_qk: ResidentMatrix::synthetic(
                            rng.matrix(
                                hidden,
                                (config.indexer_n_heads + config.indexer_kv_heads)
                                    * config.indexer_head_dim,
                            ),
                            (config.indexer_n_heads + config.indexer_kv_heads)
                                * config.indexer_head_dim,
                        ),
                        index_q_norm: vec![0.0; config.indexer_head_dim],
                        index_k_norm: vec![0.0; config.indexer_head_dim],
                        selector: Qwen4ExpQsaSelector::new(
                            config.indexer_n_heads,
                            config.indexer_head_dim,
                            config.indexer_budget,
                            config.indexer_compress_ratio,
                            config.rotary_dim(),
                            config.rope.theta,
                            config.rms_norm_eps,
                        )?,
                    }))
                }
            };
            let ple = if config.ple_layer_ids.contains(&(index + 1)) {
                Some(PleWeights::synthetic(config, &mut rng, index)?)
            } else {
                None
            };
            layers.push(LayerWeights {
                attention_connection: GatedResidualWeights::synthetic(config, &mut rng, true),
                attention,
                mlp_connection: GatedResidualWeights::synthetic(config, &mut rng, true),
                moe: MoeWeights::synthetic(config, &mut rng),
                ple,
            });
        }
        Ok(Self {
            token_embedding,
            layers,
            final_connection: GatedResidualWeights::synthetic(config, &mut rng, false),
            lm_head,
        })
    }
}

impl GatedResidualWeights {
    fn from_tensors(
        _config: &Qwen4ExpTextConfig,
        tensors: &mut HashMap<String, Tensor>,
        prefix: &str,
        combine: bool,
    ) -> Result<Self> {
        Ok(Self {
            norm: take_f32(tensors, &format!("{prefix}.hc_norm.weight"))?,
            down: ResidentMatrix::Checkpoint(take_tensor(
                tensors,
                &format!("{prefix}.input_mix_weight_down.weight"),
            )?),
            up: ResidentMatrix::Checkpoint(take_tensor(
                tensors,
                &format!("{prefix}.input_mix_weight_up.weight"),
            )?),
            inject: combine
                .then(|| {
                    take_tensor(tensors, &format!("{prefix}.block_inject_weight.weight"))
                        .map(ResidentMatrix::Checkpoint)
                })
                .transpose()?,
        })
    }

    fn synthetic(config: &Qwen4ExpTextConfig, rng: &mut SyntheticRng, combine: bool) -> Self {
        let hc_hidden = config.hc_count * config.hidden_size;
        Self {
            norm: vec![0.0; hc_hidden],
            down: ResidentMatrix::synthetic(
                rng.matrix(hc_hidden, config.hc_lowrank),
                config.hc_lowrank,
            ),
            up: ResidentMatrix::synthetic(rng.matrix(config.hc_lowrank, hc_hidden), hc_hidden),
            inject: combine.then(|| {
                ResidentMatrix::synthetic(rng.matrix(hc_hidden, config.hc_count), config.hc_count)
            }),
        }
    }
}

impl GdnWeights {
    fn from_tensors(
        config: &Qwen4ExpTextConfig,
        tensors: &mut HashMap<String, Tensor>,
        layer_prefix: &str,
    ) -> Result<Self> {
        let prefix = format!("{layer_prefix}.linear_attn");
        let qkv = 2 * config.linear_key_width() + config.linear_value_width();
        let conv = take_tensor(tensors, &format!("{prefix}.conv1d.weight"))?;
        let conv = Tensor::from_f32(
            vec![qkv, config.linear_conv_kernel_dim],
            &conv.to_f32_vec()?,
        )?;
        Ok(Self {
            in_qkv: ResidentMatrix::Checkpoint(take_tensor(
                tensors,
                &format!("{prefix}.in_proj_qkv.weight"),
            )?),
            in_z: ResidentMatrix::Checkpoint(take_tensor(
                tensors,
                &format!("{prefix}.in_proj_z.weight"),
            )?),
            in_a: ResidentMatrix::Checkpoint(take_tensor(
                tensors,
                &format!("{prefix}.in_proj_a.weight"),
            )?),
            in_b: ResidentMatrix::Checkpoint(take_tensor(
                tensors,
                &format!("{prefix}.in_proj_b.weight"),
            )?),
            conv,
            a_log: take_f32_tensor(tensors, &format!("{prefix}.A_log"))?,
            dt_bias: take_f32_tensor(tensors, &format!("{prefix}.dt_bias"))?,
            norm: take_f32_tensor(tensors, &format!("{prefix}.norm.weight"))?,
            out: ResidentMatrix::Checkpoint(take_tensor(
                tensors,
                &format!("{prefix}.out_proj.weight"),
            )?),
        })
    }
}

impl QsaWeights {
    fn from_tensors(
        config: &Qwen4ExpTextConfig,
        tensors: &mut HashMap<String, Tensor>,
        layer_prefix: &str,
    ) -> Result<Self> {
        let prefix = format!("{layer_prefix}.self_attn");
        Ok(Self {
            query_gate: QueryGateWeights::Checkpoint(take_tensor(
                tensors,
                &format!("{prefix}.q_proj.weight"),
            )?),
            k: ResidentMatrix::Checkpoint(take_tensor(
                tensors,
                &format!("{prefix}.k_proj.weight"),
            )?),
            v: ResidentMatrix::Checkpoint(take_tensor(
                tensors,
                &format!("{prefix}.v_proj.weight"),
            )?),
            out: ResidentMatrix::Checkpoint(take_tensor(
                tensors,
                &format!("{prefix}.o_proj.weight"),
            )?),
            q_norm: take_f32(tensors, &format!("{prefix}.q_norm.weight"))?,
            k_norm: take_f32(tensors, &format!("{prefix}.k_norm.weight"))?,
            index_qk: ResidentMatrix::Checkpoint(take_tensor(
                tensors,
                &format!("{prefix}.indexer.index_qk_proj.weight"),
            )?),
            index_q_norm: take_f32(tensors, &format!("{prefix}.indexer.q_layernorm.weight"))?,
            index_k_norm: take_f32(tensors, &format!("{prefix}.indexer.k_layernorm.weight"))?,
            selector: Qwen4ExpQsaSelector::new(
                config.indexer_n_heads,
                config.indexer_head_dim,
                config.indexer_budget,
                config.indexer_compress_ratio,
                config.rotary_dim(),
                config.rope.theta,
                config.rms_norm_eps,
            )?,
        })
    }
}

impl MoeWeights {
    fn from_tensors(
        _config: &Qwen4ExpTextConfig,
        tensors: &mut HashMap<String, Tensor>,
        layer_prefix: &str,
    ) -> Result<Self> {
        let prefix = format!("{layer_prefix}.mlp");
        Ok(Self {
            router: ResidentMatrix::Checkpoint(take_tensor(
                tensors,
                &format!("{prefix}.gate.weight"),
            )?),
            experts: ExpertWeights::Checkpoint {
                gate_up: take_tensor(tensors, &format!("{prefix}.experts.gate_up_proj"))?,
                down: take_tensor(tensors, &format!("{prefix}.experts.down_proj"))?,
            },
            shared_gate: ResidentMatrix::Checkpoint(take_tensor(
                tensors,
                &format!("{prefix}.shared_expert.gate_proj.weight"),
            )?),
            shared_up: ResidentMatrix::Checkpoint(take_tensor(
                tensors,
                &format!("{prefix}.shared_expert.up_proj.weight"),
            )?),
            shared_down: ResidentMatrix::Checkpoint(take_tensor(
                tensors,
                &format!("{prefix}.shared_expert.down_proj.weight"),
            )?),
            shared_expert_gate: ResidentMatrix::Checkpoint(take_tensor(
                tensors,
                &format!("{prefix}.shared_expert_gate.weight"),
            )?),
        })
    }

    fn synthetic(config: &Qwen4ExpTextConfig, rng: &mut SyntheticRng) -> Self {
        Self {
            router: ResidentMatrix::synthetic(
                rng.matrix(config.hidden_size, config.num_experts),
                config.num_experts,
            ),
            experts: ExpertWeights::Synthetic {
                gate_up: (0..config.num_experts)
                    .map(|_| rng.matrix(config.hidden_size, 2 * config.moe_intermediate_size))
                    .collect(),
                down: (0..config.num_experts)
                    .map(|_| rng.matrix(config.moe_intermediate_size, config.hidden_size))
                    .collect(),
            },
            shared_gate: ResidentMatrix::synthetic(
                rng.matrix(config.hidden_size, config.shared_expert_intermediate_size),
                config.shared_expert_intermediate_size,
            ),
            shared_up: ResidentMatrix::synthetic(
                rng.matrix(config.hidden_size, config.shared_expert_intermediate_size),
                config.shared_expert_intermediate_size,
            ),
            shared_down: ResidentMatrix::synthetic(
                rng.matrix(config.shared_expert_intermediate_size, config.hidden_size),
                config.hidden_size,
            ),
            shared_expert_gate: ResidentMatrix::synthetic(rng.matrix(config.hidden_size, 1), 1),
        }
    }
}

impl PleWeights {
    fn from_tensors(
        config: &Qwen4ExpTextConfig,
        tensors: &mut HashMap<String, Tensor>,
        layer_prefix: &str,
        layer_index: usize,
    ) -> Result<Self> {
        let prefix = format!("{layer_prefix}.ple");
        let ordinal = config
            .ple_layer_ids
            .iter()
            .position(|id| *id == layer_index + 1)
            .ok_or_else(|| Error::Other("qwen4-exp PLE ordinal missing".into()))?;
        let ngram_heads = (config.ngram_size - 1) * config.heads_per_ngram;
        let head_dim = config.ple_embed_dim / ngram_heads;
        let (head_vocab_sizes, head_offsets, padded) = ple_vocab_layout(config, ordinal);
        let mut shards = Vec::with_capacity(config.split_ngram_parts);
        let mut row_ends = Vec::with_capacity(config.split_ngram_parts);
        let mut rows = 0usize;
        for shard in 0..config.split_ngram_parts {
            let tensor = take_tensor(
                tensors,
                &format!("{prefix}.ple_embedding.ngram_embedding.shard_{shard}.weight"),
            )?;
            rows += tensor.shape().dims()[0];
            row_ends.push(rows);
            shards.push(tensor);
        }
        if rows != padded {
            return Err(Error::Other(format!(
                "qwen4-exp {prefix} embedding has {rows} rows, expected {padded}"
            )));
        }
        let conv = take_f32(tensors, &format!("{prefix}.conv1d.weight"))?;
        Ok(Self {
            embedding: PleEmbedding::Checkpoint { shards, row_ends },
            head_vocab_sizes,
            head_offsets,
            multipliers: build_multipliers(
                config.vocab_size,
                config.ngram_size,
                ordinal,
                config.seed,
            ),
            head_dim,
            key: ResidentMatrix::Checkpoint(take_tensor(
                tensors,
                &format!("{prefix}.key_proj.weight"),
            )?),
            value: ResidentMatrix::Checkpoint(take_tensor(
                tensors,
                &format!("{prefix}.value_proj.weight"),
            )?),
            norm_key: take_f32(tensors, &format!("{prefix}.norm_key.weight"))?,
            norm_query: take_f32(tensors, &format!("{prefix}.norm_query.weight"))?,
            norm_conv: take_f32(tensors, &format!("{prefix}.norm_conv.weight"))?,
            conv,
        })
    }

    fn synthetic(
        config: &Qwen4ExpTextConfig,
        rng: &mut SyntheticRng,
        layer_index: usize,
    ) -> Result<Self> {
        let ngram_heads = (config.ngram_size - 1) * config.heads_per_ngram;
        let head_dim = config.ple_embed_dim / ngram_heads;
        let ordinal = config
            .ple_layer_ids
            .iter()
            .position(|id| *id == layer_index + 1)
            .ok_or_else(|| Error::Other("qwen4-exp PLE ordinal missing".into()))?;
        let (head_vocab_sizes, head_offsets, padded) = ple_vocab_layout(config, ordinal);
        Ok(Self {
            embedding: PleEmbedding::Synthetic(rng.matrix(padded, head_dim)),
            head_vocab_sizes,
            head_offsets,
            multipliers: build_multipliers(
                config.vocab_size,
                config.ngram_size,
                ordinal,
                config.seed,
            ),
            head_dim,
            key: ResidentMatrix::synthetic(
                rng.matrix(config.ple_embed_dim, config.hc_count * config.hidden_size),
                config.hc_count * config.hidden_size,
            ),
            value: ResidentMatrix::synthetic(
                rng.matrix(config.ple_embed_dim, config.hidden_size),
                config.hidden_size,
            ),
            norm_key: vec![0.0; config.hc_count * config.hidden_size],
            norm_query: vec![0.0; config.hc_count * config.hidden_size],
            norm_conv: vec![0.0; config.hc_count * config.hidden_size],
            conv: rng.matrix(
                config.hc_count * config.hidden_size,
                config.ple_conv_kernel_size,
            ),
        })
    }
}

struct ResidualMix {
    mixed: Vec<f32>,
    injection: Vec<f32>,
}

fn run_gated_residual(
    config: &Qwen4ExpTextConfig,
    hyper: &[f32],
    weights: &GatedResidualWeights,
    combine: bool,
) -> Result<ResidualMix> {
    let hc_hidden = config.hc_count * config.hidden_size;
    if hyper.len() != hc_hidden {
        return Err(Error::Other("qwen4-exp hyper input shape mismatch".into()));
    }
    let normalized = grouped_rms_norm(
        hyper,
        &weights.norm,
        config.hc_count,
        config.hidden_size,
        config.rms_norm_eps,
    );
    let down = weights
        .down
        .apply(&normalized)?
        .into_iter()
        .map(|value| silu(value / config.hc_count as f32))
        .collect::<Vec<_>>();
    let mix_weights = weights
        .up
        .apply(&down)?
        .into_iter()
        .map(sigmoid)
        .collect::<Vec<_>>();
    let mut mixed = vec![0.0; config.hidden_size];
    for stream in 0..config.hc_count {
        for (column, mixed) in mixed.iter_mut().enumerate() {
            let index = stream * config.hidden_size + column;
            *mixed += mix_weights[index] * normalized[index] / config.hc_count as f32;
        }
    }
    let injection = if combine {
        weights
            .inject
            .as_ref()
            .ok_or_else(|| Error::Other("qwen4-exp injection weights missing".into()))?
            .apply(&normalized)?
            .into_iter()
            .map(|value| 2.0 * sigmoid(value / config.hc_count as f32))
            .collect()
    } else {
        Vec::new()
    };
    Ok(ResidualMix { mixed, injection })
}

fn inject(
    hyper: &[f32],
    block: &[f32],
    injection: &[f32],
    config: &Qwen4ExpTextConfig,
) -> Vec<f32> {
    let mut output = hyper.to_vec();
    for stream in 0..config.hc_count {
        for column in 0..config.hidden_size {
            output[stream * config.hidden_size + column] += injection[stream] * block[column];
        }
    }
    output
}

fn run_gdn(
    backend: &dyn Backend,
    config: &Qwen4ExpTextConfig,
    hidden: &[f32],
    weights: &GdnWeights,
    state: &mut GdnState,
) -> Result<Vec<f32>> {
    let key_width = config.linear_key_width();
    let value_width = config.linear_value_width();
    let qkv_width = 2 * key_width + value_width;
    let (qkv, prefetched_z) =
        if weights.in_qkv.is_large_bf16_checkpoint() && weights.in_z.is_large_bf16_checkpoint() {
            let (qkv, z) = rayon::join(
                || weights.in_qkv.apply(hidden),
                || weights.in_z.apply(hidden),
            );
            (qkv?, Some(z?))
        } else {
            (weights.in_qkv.apply(hidden)?, None)
        };
    let qkv = Tensor::from_f32(vec![1, qkv_width], &qkv)?;
    let (qkv, next_conv) =
        backend.causal_depthwise_conv1d(&qkv, &weights.conv, None, state.conv.as_ref())?;
    let qkv = backend.silu(&qkv)?.to_f32_vec()?;
    let mut q = Tensor::from_f32(
        vec![1, config.linear_num_key_heads, config.linear_key_head_dim],
        &qkv[..key_width],
    )?;
    let mut k = Tensor::from_f32(
        vec![1, config.linear_num_key_heads, config.linear_key_head_dim],
        &qkv[key_width..2 * key_width],
    )?;
    q = backend.l2_normalize(&q, -1, 1.0e-6)?;
    q = backend.scale(&q, 1.0 / (config.linear_key_head_dim as f32).sqrt())?;
    k = backend.l2_normalize(&k, -1, 1.0e-6)?;
    let v = Tensor::from_f32(
        vec![
            1,
            config.linear_num_value_heads,
            config.linear_value_head_dim,
        ],
        &qkv[2 * key_width..],
    )?;
    let a = Tensor::from_f32(
        vec![1, config.linear_num_value_heads],
        &weights.in_a.apply(hidden)?,
    )?;
    let b = Tensor::from_f32(
        vec![1, config.linear_num_value_heads],
        &weights.in_b.apply(hidden)?,
    )?;
    let (core, recurrent) = backend.gated_delta_recurrent(
        &q,
        &k,
        &v,
        &a,
        &b,
        &weights.a_log,
        &weights.dt_bias,
        state.recurrent.as_ref(),
    )?;
    let core = core.reshape(vec![
        config.linear_num_value_heads,
        config.linear_value_head_dim,
    ])?;
    let core = backend
        .rms_norm(&core, &weights.norm, config.rms_norm_eps)?
        .to_f32_vec()?;
    let z = prefetched_z
        .map_or_else(|| weights.in_z.apply(hidden), Ok)?
        .into_iter()
        .map(|value| match config.output_gate_type.as_str() {
            "sigmoid" => sigmoid(value),
            _ => silu(value),
        });
    let gated = core
        .into_iter()
        .zip(z)
        .map(|(core, gate)| core * gate)
        .collect::<Vec<_>>();
    state.conv = Some(next_conv);
    state.recurrent = Some(recurrent);
    weights.out.apply(&gated)
}

fn run_qsa(
    config: &Qwen4ExpTextConfig,
    hidden: &[f32],
    position: [u32; 3],
    weights: &QsaWeights,
    state: &mut QsaState,
) -> Result<Vec<f32>> {
    let (mut query, gate) = weights.query_gate.apply(config, hidden)?;
    let gate = gate.into_iter().map(sigmoid).collect::<Vec<_>>();
    for head in query.chunks_exact_mut(config.head_dim) {
        let source = head.to_vec();
        rms_norm_zero_centered_into(&source, &weights.q_norm, config.rms_norm_eps, head);
        apply_partial_mrope(
            head,
            config.rotary_dim(),
            config.rope.theta,
            position,
            config.rope.mrope_section,
        );
    }
    let kvi_elements = weights
        .k
        .bf16_checkpoint_numel()
        .zip(weights.v.bf16_checkpoint_numel())
        .zip(weights.index_qk.bf16_checkpoint_numel())
        .map(|((k, v), index)| k.saturating_add(v).saturating_add(index));
    let (mut key, value, index_qk) =
        if kvi_elements.is_some_and(|numel| numel >= BF16_GEMV_PAR_MIN_ELEMENTS) {
            let (key, (value, index_qk)) = rayon::join(
                || weights.k.apply(hidden),
                || {
                    rayon::join(
                        || weights.v.apply(hidden),
                        || weights.index_qk.apply(hidden),
                    )
                },
            );
            (key?, value?, index_qk?)
        } else {
            (
                weights.k.apply(hidden)?,
                weights.v.apply(hidden)?,
                weights.index_qk.apply(hidden)?,
            )
        };
    for head in key.chunks_exact_mut(config.head_dim) {
        let source = head.to_vec();
        rms_norm_zero_centered_into(&source, &weights.k_norm, config.rms_norm_eps, head);
        apply_partial_mrope(
            head,
            config.rotary_dim(),
            config.rope.theta,
            position,
            config.rope.mrope_section,
        );
    }
    let query_width = config.indexer_n_heads * config.indexer_head_dim;
    let index_query = &index_qk[..query_width];
    state.raw_keys.extend_from_slice(&index_qk[query_width..]);
    state.keys.extend_from_slice(&key);
    state.values.extend_from_slice(&value);
    state.positions.push(position);
    let selected = weights.selector.select_with_positions(
        index_query,
        &state.raw_keys,
        &state.positions,
        config.rope.mrope_section,
        &weights.index_q_norm,
        &weights.index_k_norm,
    )?;

    let mut attention = sparse_attention(
        &query,
        &state.keys,
        &state.values,
        &selected,
        config.n_attention_heads,
        config.n_kv_heads,
        config.head_dim,
    );
    for (attention, gate) in attention.iter_mut().zip(gate) {
        *attention *= gate;
    }
    weights.out.apply(&attention)
}

fn sparse_attention(
    query: &[f32],
    keys: &[f32],
    values: &[f32],
    selected: &[usize],
    query_heads: usize,
    kv_heads: usize,
    head_dim: usize,
) -> Vec<f32> {
    debug_assert_eq!(query.len(), query_heads * head_dim);
    let queries_per_kv = query_heads / kv_heads;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut attention = vec![0.0f32; query.len()];
    let work = query_heads
        .saturating_mul(selected.len())
        .saturating_mul(head_dim);
    if work >= SPARSE_ATTENTION_PAR_MIN_WORK {
        attention
            .par_chunks_exact_mut(head_dim)
            .enumerate()
            .for_each(|(query_head, output)| {
                let kv_head = query_head / queries_per_kv;
                let query_row = &query[query_head * head_dim..(query_head + 1) * head_dim];
                sparse_attention_head(
                    query_row, keys, values, selected, kv_head, kv_heads, head_dim, scale, output,
                );
            });
    } else {
        for (query_head, output) in attention.chunks_exact_mut(head_dim).enumerate() {
            let kv_head = query_head / queries_per_kv;
            let query_row = &query[query_head * head_dim..(query_head + 1) * head_dim];
            sparse_attention_head(
                query_row, keys, values, selected, kv_head, kv_heads, head_dim, scale, output,
            );
        }
    }
    attention
}

const SPARSE_ATTENTION_PAR_MIN_WORK: usize = 1 << 19;

#[allow(clippy::too_many_arguments)]
fn sparse_attention_head(
    query: &[f32],
    keys: &[f32],
    values: &[f32],
    selected: &[usize],
    kv_head: usize,
    kv_heads: usize,
    head_dim: usize,
    scale: f32,
    output: &mut [f32],
) {
    let mut scores = vec![0.0f32; selected.len()];
    for (slot, &token) in selected.iter().enumerate() {
        let key_start = (token * kv_heads + kv_head) * head_dim;
        scores[slot] = dot(query, &keys[key_start..key_start + head_dim]) * scale;
    }
    softmax_in_place(&mut scores);
    for (slot, &token) in selected.iter().enumerate() {
        let value_start = (token * kv_heads + kv_head) * head_dim;
        for column in 0..head_dim {
            output[column] += scores[slot] * values[value_start + column];
        }
    }
}

fn run_moe(config: &Qwen4ExpTextConfig, hidden: &[f32], weights: &MoeWeights) -> Result<Vec<f32>> {
    let mut probabilities = weights.router.apply(hidden)?;
    softmax_in_place(&mut probabilities);
    let mut ranked = rank_top_experts(&probabilities, config.num_experts_per_tok);
    if config.norm_topk_prob {
        let sum = ranked.iter().map(|(_, value)| value).sum::<f32>();
        for (_, value) in &mut ranked {
            *value /= sum;
        }
    }
    let (mut output, shared) =
        if should_parallelize_selected_experts(config, &weights.experts, ranked.len()) {
            let (routed, shared) = rayon::join(
                || run_selected_experts(config, hidden, &weights.experts, &ranked),
                || run_shared_expert(hidden, weights),
            );
            (routed?, shared?)
        } else {
            (
                run_selected_experts(config, hidden, &weights.experts, &ranked)?,
                run_shared_expert(hidden, weights)?,
            )
        };
    for (output, shared) in output.iter_mut().zip(shared) {
        *output += shared;
    }
    Ok(output)
}

fn rank_top_experts(probabilities: &[f32], topk: usize) -> Vec<(usize, f32)> {
    let mut ranked = probabilities
        .iter()
        .copied()
        .enumerate()
        .collect::<Vec<_>>();
    let compare = |(left_index, left): &(usize, f32), (right_index, right): &(usize, f32)| {
        right
            .total_cmp(left)
            .then_with(|| left_index.cmp(right_index))
    };
    let keep = topk.min(ranked.len());
    if keep < ranked.len() {
        ranked.select_nth_unstable_by(keep, &compare);
    }
    ranked.truncate(keep);
    ranked.sort_unstable_by(compare);
    ranked
}

fn run_shared_expert(hidden: &[f32], weights: &MoeWeights) -> Result<Vec<f32>> {
    let shared_gate = weights.shared_gate.apply(hidden)?;
    let shared_up = weights.shared_up.apply(hidden)?;
    let shared = shared_gate
        .into_iter()
        .zip(shared_up)
        .map(|(gate, up)| silu(gate) * up)
        .collect::<Vec<_>>();
    let mut shared = weights.shared_down.apply(&shared)?;
    let gate = sigmoid(weights.shared_expert_gate.apply(hidden)?[0]);
    for value in &mut shared {
        *value *= gate;
    }
    Ok(shared)
}

fn run_selected_experts(
    config: &Qwen4ExpTextConfig,
    hidden: &[f32],
    experts: &ExpertWeights,
    ranked: &[(usize, f32)],
) -> Result<Vec<f32>> {
    let parallel = should_parallelize_selected_experts(config, experts, ranked.len());
    let expert_outputs = if parallel {
        // Indexed parallel collection preserves router rank order. Keeping the
        // inner GEMVs serial avoids nested Rayon dispatch while experts own the
        // available row-level parallelism.
        ranked
            .par_iter()
            .map(|(expert, probability)| {
                experts
                    .apply_bf16_serial(config, hidden, *expert)
                    .map(|output| (*probability, output))
            })
            .collect::<Result<Vec<_>>>()?
    } else {
        ranked
            .iter()
            .map(|(expert, probability)| {
                experts
                    .apply(config, hidden, *expert)
                    .map(|output| (*probability, output))
            })
            .collect::<Result<Vec<_>>>()?
    };
    let mut output = vec![0.0; config.hidden_size];
    for (probability, expert_output) in expert_outputs {
        for (output, value) in output.iter_mut().zip(expert_output) {
            *output += probability * value;
        }
    }
    Ok(output)
}

fn should_parallelize_selected_experts(
    config: &Qwen4ExpTextConfig,
    experts: &ExpertWeights,
    selected: usize,
) -> bool {
    let per_expert_elements = config
        .hidden_size
        .saturating_mul(config.moe_intermediate_size)
        .saturating_mul(3);
    selected > 1
        && per_expert_elements >= BF16_GEMV_PAR_MIN_ELEMENTS
        && experts.is_bf16_checkpoint()
}

fn run_ple(
    config: &Qwen4ExpTextConfig,
    hyper: &[f32],
    token: u32,
    weights: &PleWeights,
    state: &mut PleState,
) -> Result<Vec<f32>> {
    let mut recent = Vec::with_capacity(config.ngram_size);
    recent.push(token);
    recent.extend(state.context.iter().rev().copied());
    let mut embedding = Vec::with_capacity(config.ple_embed_dim);
    let mut head = 0usize;
    for ngram in 2..=config.ngram_size {
        let mut mixed = (token as i64).wrapping_mul(weights.multipliers[0]);
        for (position, &recent_token) in recent.iter().take(ngram).enumerate().skip(1) {
            mixed ^= (recent_token as i64).wrapping_mul(weights.multipliers[position]);
        }
        for _ in 0..config.heads_per_ngram {
            let row = mixed.rem_euclid(weights.head_vocab_sizes[head] as i64) as usize
                + weights.head_offsets[head];
            weights
                .embedding
                .extend_row(row, weights.head_dim, &mut embedding)?;
            head += 1;
        }
    }
    let key = weights.key.apply(&embedding)?;
    let value = weights.value.apply(&embedding)?;
    let key = grouped_rms_norm(
        &key,
        &weights.norm_key,
        config.hc_count,
        config.hidden_size,
        config.rms_norm_eps,
    );
    let query = grouped_rms_norm(
        hyper,
        &weights.norm_query,
        config.hc_count,
        config.hidden_size,
        config.rms_norm_eps,
    );
    let mut gated = vec![0.0; config.hc_count * config.hidden_size];
    for stream in 0..config.hc_count {
        let start = stream * config.hidden_size;
        let raw_gate = dot(
            &key[start..start + config.hidden_size],
            &query[start..start + config.hidden_size],
        ) / (config.hidden_size as f32).sqrt();
        let transformed = raw_gate.signum() * raw_gate.abs().max(1.0e-6).sqrt();
        let gate = sigmoid(transformed);
        for column in 0..config.hidden_size {
            gated[start + column] = gate * value[column];
        }
    }
    let normalized = grouped_rms_norm(
        &gated,
        &weights.norm_conv,
        config.hc_count,
        config.hidden_size,
        config.rms_norm_eps,
    );
    let channels = config.hc_count * config.hidden_size;
    let history_len = (config.ple_conv_kernel_size - 1) * config.ngram_size;
    let mut output = gated;
    for channel in 0..channels {
        let mut convolution = weights.conv
            [channel * config.ple_conv_kernel_size + config.ple_conv_kernel_size - 1]
            * normalized[channel];
        for tap in 0..config.ple_conv_kernel_size - 1 {
            let history_index =
                history_len - config.ngram_size * (config.ple_conv_kernel_size - 1 - tap);
            convolution += weights.conv[channel * config.ple_conv_kernel_size + tap]
                * state.conv_history[history_index * channels + channel];
        }
        output[channel] += silu(convolution);
    }
    if history_len > 0 {
        state.conv_history.copy_within(channels.., 0);
        let end = state.conv_history.len();
        state.conv_history[end - channels..].copy_from_slice(&normalized);
    }
    if token == config.eos_token_id {
        state.context.fill(config.eos_token_id);
    } else if !state.context.is_empty() {
        state.context.rotate_left(1);
        *state.context.last_mut().unwrap() = token;
    }
    Ok(output)
}

fn take_tensor(tensors: &mut HashMap<String, Tensor>, name: &str) -> Result<Tensor> {
    tensors
        .remove(name)
        .ok_or_else(|| Error::Other(format!("qwen4-exp checkpoint: missing {name}")))
}

fn take_f32(tensors: &mut HashMap<String, Tensor>, name: &str) -> Result<Vec<f32>> {
    take_tensor(tensors, name)?.to_f32_vec()
}

fn take_f32_tensor(tensors: &mut HashMap<String, Tensor>, name: &str) -> Result<Tensor> {
    let tensor = take_tensor(tensors, name)?;
    Tensor::from_f32(tensor.shape().dims().to_vec(), &tensor.to_f32_vec()?)
}

fn append_tensor_values(
    tensor: &Tensor,
    start: usize,
    end: usize,
    output: &mut Vec<f32>,
) -> Result<()> {
    match tensor.dtype() {
        DType::F32 => output.extend_from_slice(&tensor.as_f32()?[start..end]),
        DType::BF16 => output.extend(
            tensor.as_bf16()?[start..end]
                .iter()
                .map(|value| value.to_f32()),
        ),
        DType::F16 => output.extend(
            tensor.as_f16()?[start..end]
                .iter()
                .map(|value| value.to_f32()),
        ),
        dtype => {
            return Err(Error::Other(format!(
                "qwen4-exp resident tensor dtype {dtype} is unsupported"
            )))
        }
    }
    Ok(())
}

fn linear_checkpoint_matrix(
    input: &[f32],
    tensor: &Tensor,
    offset: usize,
    output_size: usize,
    input_size: usize,
) -> Result<Vec<f32>> {
    debug_assert_eq!(input.len(), input_size);
    let mut output = vec![0.0f32; output_size];
    match tensor.dtype() {
        DType::F32 => {
            let weights = tensor.as_f32()?;
            for (row, output) in output.iter_mut().enumerate() {
                let start = offset + row * input_size;
                *output = dot(input, &weights[start..start + input_size]);
            }
        }
        DType::BF16 => {
            let weights = tensor.as_bf16()?;
            let end = offset + output_size * input_size;
            let matrix = &weights[offset..end];
            if matrix.len() >= BF16_GEMV_PAR_MIN_ELEMENTS {
                output
                    .par_iter_mut()
                    .zip(matrix.par_chunks_exact(input_size))
                    .for_each(|(output, row)| *output = dot_bf16(input, row));
            } else {
                for (output, row) in output.iter_mut().zip(matrix.chunks_exact(input_size)) {
                    *output = dot_bf16(input, row);
                }
            }
        }
        DType::F16 => {
            let weights = tensor.as_f16()?;
            for (row, output) in output.iter_mut().enumerate() {
                let start = offset + row * input_size;
                *output = input
                    .iter()
                    .zip(&weights[start..start + input_size])
                    .map(|(input, weight)| input * weight.to_f32())
                    .sum();
            }
        }
        dtype => {
            return Err(Error::Other(format!(
                "qwen4-exp resident matrix dtype {dtype} is unsupported"
            )))
        }
    }
    Ok(output)
}

fn linear_checkpoint_bf16_serial(
    input: &[f32],
    tensor: &Tensor,
    offset: usize,
    output_size: usize,
    input_size: usize,
) -> Result<Vec<f32>> {
    debug_assert_eq!(input.len(), input_size);
    let weights = tensor.as_bf16()?;
    let end = offset + output_size * input_size;
    Ok(weights[offset..end]
        .chunks_exact(input_size)
        .map(|row| dot_bf16(input, row))
        .collect())
}

const BF16_GEMV_PAR_MIN_ELEMENTS: usize = 1 << 21;

fn grouped_rms_norm(
    input: &[f32],
    weight: &[f32],
    groups: usize,
    group_size: usize,
    eps: f32,
) -> Vec<f32> {
    let mut output = vec![0.0; input.len()];
    for group in 0..groups {
        let start = group * group_size;
        rms_norm_zero_centered_into(
            &input[start..start + group_size],
            &weight[start..start + group_size],
            eps,
            &mut output[start..start + group_size],
        );
    }
    output
}

fn linear(input: &[f32], weight: &[f32], output_size: usize) -> Vec<f32> {
    debug_assert_eq!(weight.len(), input.len() * output_size);
    let mut output = vec![0.0; output_size];
    for (input_index, value) in input.iter().copied().enumerate() {
        let row = &weight[input_index * output_size..(input_index + 1) * output_size];
        for (output, weight) in output.iter_mut().zip(row) {
            *output += value * weight;
        }
    }
    output
}

fn softmax_in_place(values: &mut [f32]) {
    let maximum = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0;
    for value in values.iter_mut() {
        *value = (*value - maximum).exp();
        sum += *value;
    }
    for value in values {
        *value /= sum;
    }
}

#[inline]
fn dot(left: &[f32], right: &[f32]) -> f32 {
    debug_assert_eq!(left.len(), right.len());
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON is mandatory in AArch64, and the helper bounds every
        // vector load before reading it.
        unsafe { dot_f32_neon(left, right) }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        dot_f32_scalar(left, right)
    }
}

#[cfg(any(not(target_arch = "aarch64"), test))]
#[inline]
fn dot_f32_scalar(left: &[f32], right: &[f32]) -> f32 {
    let mut sums = [0.0f32; 8];
    let mut index = 0usize;
    while index + 8 <= left.len() {
        for lane in 0..8 {
            sums[lane] += left[index + lane] * right[index + lane];
        }
        index += 8;
    }
    let mut sum =
        (sums[0] + sums[1]) + (sums[2] + sums[3]) + (sums[4] + sums[5]) + (sums[6] + sums[7]);
    while index < left.len() {
        sum += left[index] * right[index];
        index += 1;
    }
    sum
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn dot_f32_neon(left: &[f32], right: &[f32]) -> f32 {
    use std::arch::aarch64::*;

    let mut sums = [vdupq_n_f32(0.0); 4];
    let mut index = 0usize;
    while index + 16 <= left.len() {
        for lane in 0..4 {
            let offset = index + lane * 4;
            sums[lane] = vfmaq_f32(
                sums[lane],
                unsafe { vld1q_f32(left.as_ptr().add(offset)) },
                unsafe { vld1q_f32(right.as_ptr().add(offset)) },
            );
        }
        index += 16;
    }
    let paired = vaddq_f32(vaddq_f32(sums[0], sums[1]), vaddq_f32(sums[2], sums[3]));
    let mut sum = vaddvq_f32(paired);
    while index < left.len() {
        sum += left[index] * right[index];
        index += 1;
    }
    sum
}

#[inline]
fn dot_bf16(left: &[f32], right: &[half::bf16]) -> f32 {
    debug_assert_eq!(left.len(), right.len());
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON is mandatory in AArch64, and the helper bounds every
        // vector load before reading it.
        unsafe { dot_bf16_neon(left, right) }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        dot_bf16_scalar(left, right)
    }
}

#[cfg(any(not(target_arch = "aarch64"), test))]
#[inline]
fn dot_bf16_scalar(left: &[f32], right: &[half::bf16]) -> f32 {
    let mut sums = [0.0f32; 8];
    let mut index = 0usize;
    while index + 8 <= left.len() {
        sums[0] += left[index] * right[index].to_f32();
        sums[1] += left[index + 1] * right[index + 1].to_f32();
        sums[2] += left[index + 2] * right[index + 2].to_f32();
        sums[3] += left[index + 3] * right[index + 3].to_f32();
        sums[4] += left[index + 4] * right[index + 4].to_f32();
        sums[5] += left[index + 5] * right[index + 5].to_f32();
        sums[6] += left[index + 6] * right[index + 6].to_f32();
        sums[7] += left[index + 7] * right[index + 7].to_f32();
        index += 8;
    }
    let mut sum =
        (sums[0] + sums[1]) + (sums[2] + sums[3]) + (sums[4] + sums[5]) + (sums[6] + sums[7]);
    while index < left.len() {
        sum += left[index] * right[index].to_f32();
        index += 1;
    }
    sum
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn dot_bf16_neon(left: &[f32], right: &[half::bf16]) -> f32 {
    use std::arch::aarch64::*;

    #[inline]
    unsafe fn load_bf16x4(pointer: *const half::bf16) -> float32x4_t {
        let bits = unsafe { vld1_u16(pointer.cast::<u16>()) };
        vreinterpretq_f32_u32(vshlq_n_u32::<16>(vmovl_u16(bits)))
    }

    let mut sums = [vdupq_n_f32(0.0); 4];
    let mut index = 0usize;
    while index + 16 <= left.len() {
        let input = left.as_ptr();
        let weights = right.as_ptr();
        sums[0] = vfmaq_f32(sums[0], unsafe { vld1q_f32(input.add(index)) }, unsafe {
            load_bf16x4(weights.add(index))
        });
        sums[1] = vfmaq_f32(
            sums[1],
            unsafe { vld1q_f32(input.add(index + 4)) },
            unsafe { load_bf16x4(weights.add(index + 4)) },
        );
        sums[2] = vfmaq_f32(
            sums[2],
            unsafe { vld1q_f32(input.add(index + 8)) },
            unsafe { load_bf16x4(weights.add(index + 8)) },
        );
        sums[3] = vfmaq_f32(
            sums[3],
            unsafe { vld1q_f32(input.add(index + 12)) },
            unsafe { load_bf16x4(weights.add(index + 12)) },
        );
        index += 16;
    }
    let paired = vaddq_f32(vaddq_f32(sums[0], sums[1]), vaddq_f32(sums[2], sums[3]));
    let mut sum = vaddvq_f32(paired);
    while index < left.len() {
        sum += left[index] * right[index].to_f32();
        index += 1;
    }
    sum
}

fn add_assign(left: &mut [f32], right: &[f32]) {
    for (left, right) in left.iter_mut().zip(right) {
        *left += *right;
    }
}

fn sigmoid(value: f32) -> f32 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exp = value.exp();
        exp / (1.0 + exp)
    }
}

fn silu(value: f32) -> f32 {
    value * sigmoid(value)
}

fn qwen3vl_vision_adapter(config: &Qwen4ExpConfig) -> Qwen3VLConfig {
    Qwen3VLConfig {
        text: Qwen3VLTextConfig {
            hidden_size: config.text.hidden_size,
            intermediate_size: config.text.moe_intermediate_size,
            n_layers: config.text.n_layers,
            n_heads: config.text.n_attention_heads,
            n_kv_heads: config.text.n_kv_heads,
            head_dim: config.text.head_dim,
            vocab_size: config.text.vocab_size,
            max_position_embeddings: config.text.max_position_embeddings,
            rms_norm_eps: config.text.rms_norm_eps,
            rope_theta: config.text.rope.theta,
            mrope_section: config.text.rope.mrope_section,
            mrope_interleaved: config.text.rope.mrope_interleaved,
            tie_word_embeddings: config.text.tie_word_embeddings,
        },
        vision: Qwen3VLVisionConfig {
            depth: config.vision.depth,
            hidden_size: config.vision.hidden_size,
            intermediate_size: config.vision.intermediate_size,
            num_heads: config.vision.num_heads,
            head_dim: config.vision.hidden_size / config.vision.num_heads,
            patch_size: config.vision.patch_size,
            temporal_patch_size: config.vision.temporal_patch_size,
            in_channels: config.vision.in_channels,
            spatial_merge_size: config.vision.spatial_merge_size,
            num_position_embeddings: config.vision.num_position_embeddings,
            out_hidden_size: config.vision.out_hidden_size,
            deepstack_visual_indexes: config.vision.deepstack_visual_indexes.clone(),
        },
        image_token_id: config.image_token_id,
        video_token_id: config.video_token_id,
        vision_start_token_id: config.vision_start_token_id,
        vision_end_token_id: config.vision_end_token_id,
    }
}

fn validate_synthetic_size(config: &Qwen4ExpTextConfig) -> Result<()> {
    if config.hidden_size > 256
        || config.vocab_size > 65_536
        || config.n_layers > 64
        || config.num_experts > 64
        || config.moe_intermediate_size > 512
        || config.ple_embed_dim > 512
        || config.ngram_vocab_size_base > 1_000_000
    {
        return Err(Error::Other(
            "qwen4-exp synthetic runtime requires a downsized architecture config; the official checkpoint must not be materialized as random weights"
                .into(),
        ));
    }
    Ok(())
}

struct SyntheticRng(u64);

impl SyntheticRng {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn matrix(&mut self, rows: usize, columns: usize) -> Vec<f32> {
        let scale = 0.08 / (rows.max(1) as f32).sqrt();
        (0..rows * columns)
            .map(|_| {
                self.0 = self
                    .0
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                let unit = (self.0 >> 40) as f32 / (1u32 << 24) as f32;
                (unit * 2.0 - 1.0) * scale
            })
            .collect()
    }
}

const SPLITMIX_GAMMA: u64 = 0x9E3779B97F4A7C15;
const SPLITMIX_M1: u64 = 0xBF58476D1CE4E5B9;
const SPLITMIX_M2: u64 = 0x94D049BB133111EB;

fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(SPLITMIX_GAMMA);
    value = (value ^ (value >> 30)).wrapping_mul(SPLITMIX_M1);
    value = (value ^ (value >> 27)).wrapping_mul(SPLITMIX_M2);
    value ^ (value >> 31)
}

fn build_multipliers(vocab: usize, ngram: usize, ordinal: usize, seed: u64) -> Vec<i64> {
    let multiplier_max = i64::MAX as u64 / vocab.max(1) as u64;
    let half_bound = (multiplier_max / 2).max(1);
    let base_seed = seed.wrapping_add(10_007 * ordinal as u64);
    (0..ngram)
        .map(|index| {
            let value = base_seed.wrapping_add(SPLITMIX_GAMMA.wrapping_mul(index as u64 + 1));
            (2 * (splitmix64(value) % half_bound) + 1) as i64
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, HashSet};
    use std::io::Write;
    use std::sync::atomic::{AtomicU64, Ordering};

    use super::*;
    use crate::qwen4_exp::config::tests::MINI_CONFIG;
    use crate::{AutoModel, LoadOptions, SyntheticWeights};

    fn model() -> GeneralQwen4Exp {
        let config = Qwen4ExpConfig::from_json_str(MINI_CONFIG).unwrap();
        GeneralQwen4Exp::from_synthetic(config, 38, 64).unwrap()
    }

    fn benchmark_bf16_gemv_pair(
        input: &[f32],
        matrix: &[half::bf16],
        rows: usize,
        columns: usize,
        samples: usize,
    ) -> (f64, f64) {
        debug_assert_eq!(input.len(), columns);
        debug_assert_eq!(matrix.len(), rows * columns);
        let mut sequential_samples = Vec::with_capacity(samples);
        let mut parallel_samples = Vec::with_capacity(samples);
        for sample in 0..=samples {
            let start = std::time::Instant::now();
            let sequential = matrix
                .chunks_exact(columns)
                .map(|row| dot_bf16(input, row))
                .collect::<Vec<_>>();
            if sample > 0 {
                sequential_samples.push(start.elapsed().as_secs_f64() * 1_000.0);
            }
            std::hint::black_box(sequential);

            let start = std::time::Instant::now();
            let mut parallel = vec![0.0f32; rows];
            parallel
                .par_iter_mut()
                .zip(matrix.par_chunks_exact(columns))
                .for_each(|(output, row)| *output = dot_bf16(input, row));
            if sample > 0 {
                parallel_samples.push(start.elapsed().as_secs_f64() * 1_000.0);
            }
            std::hint::black_box(parallel);
        }
        sequential_samples.sort_by(f64::total_cmp);
        parallel_samples.sort_by(f64::total_cmp);
        (
            sequential_samples[samples / 2],
            parallel_samples[samples / 2],
        )
    }

    fn sparse_attention_sequential_reference(
        query: &[f32],
        keys: &[f32],
        values: &[f32],
        selected: &[usize],
        query_heads: usize,
        kv_heads: usize,
        head_dim: usize,
    ) -> Vec<f32> {
        let queries_per_kv = query_heads / kv_heads;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut attention = vec![0.0f32; query.len()];
        for (query_head, output) in attention.chunks_exact_mut(head_dim).enumerate() {
            let query_row = &query[query_head * head_dim..(query_head + 1) * head_dim];
            sparse_attention_head(
                query_row,
                keys,
                values,
                selected,
                query_head / queries_per_kv,
                kv_heads,
                head_dim,
                scale,
                output,
            );
        }
        attention
    }

    fn write_safetensors(path: &std::path::Path, tensors: &HashMap<String, Tensor>) {
        let mut entries = BTreeMap::new();
        let mut offset = 0usize;
        let mut names = tensors.keys().collect::<Vec<_>>();
        names.sort_unstable();
        for name in &names {
            let tensor = &tensors[*name];
            let bytes = tensor.shape().numel() * std::mem::size_of::<f32>();
            entries.insert(
                (*name).clone(),
                serde_json::json!({
                    "dtype": "F32",
                    "shape": tensor.shape().dims(),
                    "data_offsets": [offset, offset + bytes],
                }),
            );
            offset += bytes;
        }
        let mut header = serde_json::to_vec(&entries).unwrap();
        while header.len() % 8 != 0 {
            header.push(b' ');
        }
        let mut file = std::fs::File::create(path).unwrap();
        file.write_all(&(header.len() as u64).to_le_bytes())
            .unwrap();
        file.write_all(&header).unwrap();
        for name in names {
            for value in tensors[name].as_f32().unwrap() {
                file.write_all(&value.to_le_bytes()).unwrap();
            }
        }
    }

    #[test]
    fn synthetic_prefill_and_decode_are_finite_and_stateful() {
        let mut model = model();
        let prefill = model.forward(&[1, 3, 5, 7], 0).unwrap();
        assert_eq!(prefill.shape().dims(), [4, 32]);
        assert!(prefill
            .as_f32()
            .unwrap()
            .iter()
            .all(|value| value.is_finite()));
        let decode = model.forward(&[9], 4).unwrap();
        assert_eq!(decode.shape().dims(), [1, 32]);
        assert!(decode
            .as_f32()
            .unwrap()
            .iter()
            .all(|value| value.is_finite()));
        assert_eq!(model.generation_path_receipt().unwrap()["position"], 5);
    }

    #[test]
    fn generation_prefill_projects_only_the_final_row() {
        let tokens = [1, 3, 5, 7, 9];
        let mut full = model();
        let full_logits = full.forward(&tokens, 0).unwrap();
        let vocab = full.config.text.vocab_size;
        let expected = &full_logits.as_f32().unwrap()[(tokens.len() - 1) * vocab..];

        let mut optimized = model();
        let actual = optimized
            .prefill_for_generation(LlmInput::text(&tokens))
            .unwrap();
        assert_eq!(actual.shape().dims(), [1, vocab]);
        assert_eq!(actual.as_f32().unwrap(), expected);
        assert_eq!(optimized.state.position, tokens.len());
        assert_eq!(
            optimized.generation_path_receipt().unwrap()["generation_prefill_logits_rows"],
            1
        );

        let expected_decode = full.forward(&[11], tokens.len() as u32).unwrap();
        let actual_decode = optimized.forward(&[11], tokens.len() as u32).unwrap();
        assert_eq!(
            actual_decode.as_f32().unwrap(),
            expected_decode.as_f32().unwrap()
        );
    }

    #[test]
    fn one_shot_and_tokenwise_execution_match() {
        let mut one_shot = model();
        let expected = one_shot.forward(&[1, 3, 5, 7, 9], 0).unwrap();
        let mut tokenwise = model();
        let mut actual = Vec::new();
        for (position, token) in [1, 3, 5, 7, 9].into_iter().enumerate() {
            actual.extend(
                tokenwise
                    .forward(&[token], position as u32)
                    .unwrap()
                    .to_f32_vec()
                    .unwrap(),
            );
        }
        for (left, right) in expected.as_f32().unwrap().iter().zip(actual) {
            assert!((left - right).abs() <= 1.0e-6, "{left} != {right}");
        }
    }

    #[test]
    fn reset_replays_the_same_logits() {
        let mut model = model();
        let first = model.forward(&[2, 4, 6], 0).unwrap().to_f32_vec().unwrap();
        model.reset();
        let second = model.forward(&[2, 4, 6], 0).unwrap().to_f32_vec().unwrap();
        assert_eq!(first, second);
    }

    #[test]
    fn ple_multipliers_match_the_frozen_transformers_algorithm() {
        assert_eq!(
            build_multipliers(32, 3, 0, 1234),
            vec![
                256_496_738_022_509_279,
                251_627_398_771_002_343,
                55_314_113_489_879_221,
            ]
        );
    }

    #[test]
    fn multimodal_positions_match_official_grouping_and_delta() {
        let (positions, delta) =
            multimodal_positions(&[1, 28, 28, 28, 28, 3], 28, &[[1, 4, 4]], 2).unwrap();
        assert_eq!(
            positions,
            vec![
                [0, 0, 0],
                [1, 1, 1],
                [1, 1, 2],
                [1, 2, 1],
                [1, 2, 2],
                [3, 3, 3],
            ]
        );
        assert_eq!(delta, -2);
        assert!(multimodal_positions(&[1, 28, 28, 3], 28, &[[1, 4, 4]], 2).is_err());
    }

    #[test]
    fn consumes_a_complete_toy_checkpoint_map() {
        let (config, tensors) = crate::qwen4_exp::weights::tests::zero_runtime_tensors();
        let mut model = GeneralQwen4Exp::from_tensors(config, tensors, 64).unwrap();
        let logits = model.forward(&[1, 3, 5, 7, 9], 0).unwrap();
        assert_eq!(logits.shape().dims(), [5, 32]);
        assert!(logits
            .as_f32()
            .unwrap()
            .iter()
            .all(|value| value.is_finite()));
    }

    #[test]
    fn keeps_routed_experts_and_ple_shards_bf16_resident() {
        let (config, tensors) = crate::qwen4_exp::weights::tests::zero_bf16_runtime_tensors();
        let mut model = GeneralQwen4Exp::from_tensors(config, tensors, 64).unwrap();
        let assert_matrix_bf16 = |matrix: &ResidentMatrix| match matrix {
            ResidentMatrix::Checkpoint(tensor) => assert_eq!(tensor.dtype(), DType::BF16),
            ResidentMatrix::Synthetic { .. } => panic!("checkpoint matrix was expanded"),
        };
        match &model.weights.token_embedding {
            ResidentEmbedding::Checkpoint(tensor) => assert_eq!(tensor.dtype(), DType::BF16),
            ResidentEmbedding::Synthetic(_) => panic!("checkpoint embedding was expanded"),
        }
        assert_matrix_bf16(&model.weights.lm_head);
        assert_matrix_bf16(&model.weights.layers[0].attention_connection.down);
        assert_matrix_bf16(&model.weights.layers[0].attention_connection.up);
        assert_matrix_bf16(
            model.weights.layers[0]
                .attention_connection
                .inject
                .as_ref()
                .unwrap(),
        );
        match &model.weights.layers[0].attention {
            AttentionWeights::Linear(weights) => {
                assert_matrix_bf16(&weights.in_qkv);
                assert_matrix_bf16(&weights.in_z);
                assert_matrix_bf16(&weights.out);
            }
            AttentionWeights::Qsa(_) => panic!("layer zero must be GDN"),
        }
        match &model.weights.layers[3].attention {
            AttentionWeights::Qsa(weights) => {
                match &weights.query_gate {
                    QueryGateWeights::Checkpoint(tensor) => {
                        assert_eq!(tensor.dtype(), DType::BF16)
                    }
                    QueryGateWeights::Synthetic { .. } => {
                        panic!("checkpoint query gate was expanded")
                    }
                }
                assert_matrix_bf16(&weights.k);
                assert_matrix_bf16(&weights.v);
                assert_matrix_bf16(&weights.out);
                assert_matrix_bf16(&weights.index_qk);
            }
            AttentionWeights::Linear(_) => panic!("layer three must be QSA"),
        }
        match &model.weights.layers[0].moe.experts {
            ExpertWeights::Checkpoint { gate_up, down } => {
                assert_eq!(gate_up.dtype(), DType::BF16);
                assert_eq!(down.dtype(), DType::BF16);
            }
            ExpertWeights::Synthetic { .. } => panic!("checkpoint experts were expanded"),
        }
        assert_matrix_bf16(&model.weights.layers[0].moe.router);
        assert_matrix_bf16(&model.weights.layers[0].moe.shared_gate);
        assert_matrix_bf16(&model.weights.layers[0].moe.shared_down);
        match &model.weights.layers[1].ple.as_ref().unwrap().embedding {
            PleEmbedding::Checkpoint { shards, row_ends } => {
                assert_eq!(shards.len(), 2);
                assert!(shards.iter().all(|tensor| tensor.dtype() == DType::BF16));
                assert_eq!(row_ends.last().copied(), Some(168));
            }
            PleEmbedding::Synthetic(_) => panic!("checkpoint PLE was concatenated"),
        }
        assert_matrix_bf16(&model.weights.layers[1].ple.as_ref().unwrap().key);
        assert_matrix_bf16(&model.weights.layers[1].ple.as_ref().unwrap().value);
        let logits = model.forward(&[1, 3, 5], 0).unwrap();
        assert!(logits
            .as_f32()
            .unwrap()
            .iter()
            .all(|value| value.is_finite()));
    }

    #[test]
    #[ignore = "manual performance benchmark"]
    fn benchmark_checkpoint_bf16_gemv() {
        const ROWS: usize = 4096;
        const COLUMNS: usize = 4096;
        const SAMPLES: usize = 9;

        let input = (0..COLUMNS)
            .map(|index| ((index % 251) as f32 - 125.0) / 251.0)
            .collect::<Vec<_>>();
        let weights = (0..ROWS * COLUMNS)
            .map(|index| half::bf16::from_f32(((index % 509) as f32 - 254.0) / 509.0))
            .collect::<Vec<_>>();
        let tensor = Tensor::from_bf16(vec![ROWS, COLUMNS], &weights).unwrap();

        let warmup = linear_checkpoint_matrix(&input, &tensor, 0, ROWS, COLUMNS).unwrap();
        let reference = weights
            .chunks_exact(COLUMNS)
            .map(|row| {
                input
                    .iter()
                    .zip(row)
                    .map(|(input, weight)| input * weight.to_f32())
                    .sum::<f32>()
            })
            .collect::<Vec<_>>();
        let max_abs = warmup
            .iter()
            .zip(&reference)
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0f32, f32::max);
        std::hint::black_box(&warmup);
        let mut samples = Vec::with_capacity(SAMPLES);
        let mut checksum = 0.0f64;
        for _ in 0..SAMPLES {
            let start = std::time::Instant::now();
            let output = linear_checkpoint_matrix(&input, &tensor, 0, ROWS, COLUMNS).unwrap();
            samples.push(start.elapsed().as_secs_f64() * 1_000.0);
            checksum = output.iter().map(|value| *value as f64).sum();
            std::hint::black_box(output);
        }
        samples.sort_by(f64::total_cmp);
        eprintln!(
            "qwen4_exp_bf16_gemv rows={ROWS} columns={COLUMNS} median_ms={:.6} checksum={checksum:.9} reference_max_abs={max_abs:.9}",
            samples[SAMPLES / 2]
        );

        for rows in [16usize, 32, 64, 128, 256, 512] {
            let matrix = &weights[..rows * COLUMNS];
            let (sequential, parallel) =
                benchmark_bf16_gemv_pair(&input, matrix, rows, COLUMNS, SAMPLES);
            eprintln!(
                "qwen4_exp_bf16_gemv_crossover rows={rows} columns={COLUMNS} elements={} sequential_median_ms={:.6} parallel_median_ms={:.6}",
                rows * COLUMNS,
                sequential,
                parallel
            );
        }

        for (name, rows, columns) in [
            ("router", 512usize, 2560usize),
            ("moe_down", 2560, 640),
            ("moe_gate_up", 1280, 2560),
        ] {
            let official_input = &input[..columns];
            let matrix = &weights[..rows * columns];
            let (sequential, parallel) =
                benchmark_bf16_gemv_pair(official_input, matrix, rows, columns, SAMPLES);
            eprintln!(
                "qwen4_exp_bf16_gemv_official_shape name={name} rows={rows} columns={columns} elements={} sequential_median_ms={sequential:.6} parallel_median_ms={parallel:.6}",
                rows * columns
            );
        }

        drop(tensor);
        drop(weights);

        const TOPK: usize = 10;
        const OFFICIAL_HIDDEN: usize = 2560;
        const OFFICIAL_INTERMEDIATE: usize = 640;
        let mut moe_config = Qwen4ExpConfig::from_json_str(MINI_CONFIG).unwrap().text;
        moe_config.hidden_size = OFFICIAL_HIDDEN;
        moe_config.moe_intermediate_size = OFFICIAL_INTERMEDIATE;
        moe_config.num_experts = TOPK;
        moe_config.num_experts_per_tok = TOPK;
        let hidden = (0..OFFICIAL_HIDDEN)
            .map(|index| ((index % 251) as f32 - 125.0) / 251.0)
            .collect::<Vec<_>>();

        let gate_up_numel = TOPK * 2 * OFFICIAL_INTERMEDIATE * OFFICIAL_HIDDEN;
        let packed_gate_up = (0..gate_up_numel)
            .map(|index| {
                half::bf16::from_f32(((index.wrapping_mul(13) % 509) as f32 - 254.0) / 509.0)
            })
            .collect::<Vec<_>>();
        let gate_up = Tensor::from_bf16(
            vec![TOPK, 2 * OFFICIAL_INTERMEDIATE, OFFICIAL_HIDDEN],
            &packed_gate_up,
        )
        .unwrap();
        drop(packed_gate_up);
        let down_numel = TOPK * OFFICIAL_HIDDEN * OFFICIAL_INTERMEDIATE;
        let packed_down = (0..down_numel)
            .map(|index| {
                half::bf16::from_f32(((index.wrapping_mul(29) % 509) as f32 - 254.0) / 509.0)
            })
            .collect::<Vec<_>>();
        let down = Tensor::from_bf16(
            vec![TOPK, OFFICIAL_HIDDEN, OFFICIAL_INTERMEDIATE],
            &packed_down,
        )
        .unwrap();
        drop(packed_down);
        let experts = ExpertWeights::Checkpoint { gate_up, down };

        let apply_serial = |expert: usize| {
            experts
                .apply_bf16_serial(&moe_config, &hidden, expert)
                .unwrap()
        };

        let mut sequential_samples = Vec::with_capacity(SAMPLES);
        let mut parallel_samples = Vec::with_capacity(SAMPLES);
        let mut outer_only_samples = Vec::with_capacity(SAMPLES);
        let mut reference = Vec::new();
        let mut candidate = Vec::new();
        for sample in 0..=SAMPLES {
            let start = std::time::Instant::now();
            let sequential = (0..TOPK)
                .map(|expert| experts.apply(&moe_config, &hidden, expert).unwrap())
                .collect::<Vec<_>>();
            if sample > 0 {
                sequential_samples.push(start.elapsed().as_secs_f64() * 1_000.0);
            }

            let start = std::time::Instant::now();
            let parallel = (0..TOPK)
                .into_par_iter()
                .map(|expert| experts.apply(&moe_config, &hidden, expert))
                .collect::<Result<Vec<_>>>()
                .unwrap();
            if sample > 0 {
                parallel_samples.push(start.elapsed().as_secs_f64() * 1_000.0);
            }

            let start = std::time::Instant::now();
            let outer_only = (0..TOPK)
                .into_par_iter()
                .map(&apply_serial)
                .collect::<Vec<_>>();
            if sample > 0 {
                outer_only_samples.push(start.elapsed().as_secs_f64() * 1_000.0);
            }
            reference = sequential;
            candidate = outer_only;
            std::hint::black_box(parallel);
        }
        sequential_samples.sort_by(f64::total_cmp);
        parallel_samples.sort_by(f64::total_cmp);
        outer_only_samples.sort_by(f64::total_cmp);
        let max_abs = reference
            .iter()
            .flatten()
            .zip(candidate.iter().flatten())
            .map(|(expected, actual)| (expected - actual).abs())
            .fold(0.0f32, f32::max);
        std::hint::black_box((reference, candidate));
        eprintln!(
            "qwen4_exp_bf16_moe_experts topk={TOPK} hidden={OFFICIAL_HIDDEN} intermediate={OFFICIAL_INTERMEDIATE} sequential_median_ms={:.6} nested_parallel_median_ms={:.6} outer_only_parallel_median_ms={:.6} max_abs={max_abs:.9}",
            sequential_samples[SAMPLES / 2],
            parallel_samples[SAMPLES / 2],
            outer_only_samples[SAMPLES / 2]
        );

        let checkpoint_matrix = |rows: usize, columns: usize, multiplier: usize| {
            let values = (0..rows * columns)
                .map(|index| {
                    half::bf16::from_f32(
                        ((index.wrapping_mul(multiplier) % 509) as f32 - 254.0) / 509.0,
                    )
                })
                .collect::<Vec<_>>();
            ResidentMatrix::Checkpoint(Tensor::from_bf16(vec![rows, columns], &values).unwrap())
        };
        let moe_weights = MoeWeights {
            router: ResidentMatrix::synthetic(vec![0.0; OFFICIAL_HIDDEN], 1),
            experts,
            shared_gate: checkpoint_matrix(OFFICIAL_INTERMEDIATE, OFFICIAL_HIDDEN, 31),
            shared_up: checkpoint_matrix(OFFICIAL_INTERMEDIATE, OFFICIAL_HIDDEN, 37),
            shared_down: checkpoint_matrix(OFFICIAL_HIDDEN, OFFICIAL_INTERMEDIATE, 41),
            shared_expert_gate: checkpoint_matrix(1, OFFICIAL_HIDDEN, 43),
        };
        let ranked = (0..TOPK)
            .map(|expert| (expert, 1.0 / TOPK as f32))
            .collect::<Vec<_>>();
        let mut staged_samples = Vec::with_capacity(SAMPLES);
        let mut overlapped_samples = Vec::with_capacity(SAMPLES);
        let mut staged_output = Vec::new();
        let mut overlapped_output = Vec::new();
        for sample in 0..=SAMPLES {
            let start = std::time::Instant::now();
            let mut routed =
                run_selected_experts(&moe_config, &hidden, &moe_weights.experts, &ranked).unwrap();
            let shared = run_shared_expert(&hidden, &moe_weights).unwrap();
            for (output, shared) in routed.iter_mut().zip(shared) {
                *output += shared;
            }
            if sample > 0 {
                staged_samples.push(start.elapsed().as_secs_f64() * 1_000.0);
            }
            staged_output = routed;

            let start = std::time::Instant::now();
            let (routed, shared) = rayon::join(
                || run_selected_experts(&moe_config, &hidden, &moe_weights.experts, &ranked),
                || run_shared_expert(&hidden, &moe_weights),
            );
            let mut routed = routed.unwrap();
            for (output, shared) in routed.iter_mut().zip(shared.unwrap()) {
                *output += shared;
            }
            if sample > 0 {
                overlapped_samples.push(start.elapsed().as_secs_f64() * 1_000.0);
            }
            overlapped_output = routed;
        }
        staged_samples.sort_by(f64::total_cmp);
        overlapped_samples.sort_by(f64::total_cmp);
        let max_abs = staged_output
            .iter()
            .zip(&overlapped_output)
            .map(|(expected, actual)| (expected - actual).abs())
            .fold(0.0f32, f32::max);
        std::hint::black_box((staged_output, overlapped_output));
        eprintln!(
            "qwen4_exp_bf16_moe_full staged_median_ms={:.6} overlapped_median_ms={:.6} max_abs={max_abs:.9}",
            staged_samples[SAMPLES / 2],
            overlapped_samples[SAMPLES / 2]
        );

        const ROUTER_EXPERTS: usize = 512;
        const ROUTER_TOPK: usize = 10;
        const ROUTER_ITERATIONS: usize = 20_000;
        let probabilities = (0..ROUTER_EXPERTS)
            .map(|index| {
                let mixed = index.wrapping_mul(73) % 997;
                (mixed as f32 - 498.0) / 997.0
            })
            .collect::<Vec<_>>();
        let compare = |(left_index, left): &(usize, f32), (right_index, right): &(usize, f32)| {
            right
                .total_cmp(left)
                .then_with(|| left_index.cmp(right_index))
        };
        let start = std::time::Instant::now();
        let mut full_sort = Vec::new();
        for _ in 0..ROUTER_ITERATIONS {
            let mut ranked = probabilities
                .iter()
                .copied()
                .enumerate()
                .collect::<Vec<_>>();
            ranked.sort_unstable_by(&compare);
            ranked.truncate(ROUTER_TOPK);
            full_sort = ranked;
            std::hint::black_box(&full_sort);
        }
        let full_sort_ms = start.elapsed().as_secs_f64() * 1_000.0;
        let start = std::time::Instant::now();
        let mut linear_partition = Vec::new();
        for _ in 0..ROUTER_ITERATIONS {
            linear_partition = rank_top_experts(&probabilities, ROUTER_TOPK);
            std::hint::black_box(&linear_partition);
        }
        let linear_partition_ms = start.elapsed().as_secs_f64() * 1_000.0;
        assert_eq!(linear_partition, full_sort);
        eprintln!(
            "qwen4_exp_moe_router_topk experts={ROUTER_EXPERTS} topk={ROUTER_TOPK} iterations={ROUTER_ITERATIONS} full_sort_ms={full_sort_ms:.6} linear_partition_ms={linear_partition_ms:.6}"
        );
    }

    #[test]
    #[ignore = "manual performance benchmark"]
    fn benchmark_checkpoint_bf16_gdn() {
        const HIDDEN: usize = 2560;
        const KEY_HEADS: usize = 16;
        const VALUE_HEADS: usize = 48;
        const HEAD_DIM: usize = 128;
        const CONV_KERNEL: usize = 4;
        const SAMPLES: usize = 9;

        let mut config = Qwen4ExpConfig::from_json_str(MINI_CONFIG).unwrap().text;
        config.hidden_size = HIDDEN;
        config.linear_num_key_heads = KEY_HEADS;
        config.linear_key_head_dim = HEAD_DIM;
        config.linear_num_value_heads = VALUE_HEADS;
        config.linear_value_head_dim = HEAD_DIM;
        config.linear_conv_kernel_dim = CONV_KERNEL;
        config.output_gate_type = "sigmoid".into();
        let key_width = config.linear_key_width();
        let value_width = config.linear_value_width();
        let qkv_width = 2 * key_width + value_width;
        let checkpoint_matrix = |rows: usize, columns: usize, multiplier: usize| {
            let values = (0..rows * columns)
                .map(|index| {
                    half::bf16::from_f32(
                        ((index.wrapping_mul(multiplier) % 509) as f32 - 254.0) / 509.0,
                    )
                })
                .collect::<Vec<_>>();
            ResidentMatrix::Checkpoint(Tensor::from_bf16(vec![rows, columns], &values).unwrap())
        };
        let mut conv = vec![0.0f32; qkv_width * CONV_KERNEL];
        for channel in 0..qkv_width {
            conv[channel * CONV_KERNEL + CONV_KERNEL - 1] = 1.0;
        }
        let weights = GdnWeights {
            in_qkv: checkpoint_matrix(qkv_width, HIDDEN, 13),
            in_z: checkpoint_matrix(value_width, HIDDEN, 17),
            in_a: checkpoint_matrix(VALUE_HEADS, HIDDEN, 19),
            in_b: checkpoint_matrix(VALUE_HEADS, HIDDEN, 23),
            conv: Tensor::from_f32(vec![qkv_width, CONV_KERNEL], &conv).unwrap(),
            a_log: Tensor::from_f32(vec![VALUE_HEADS], &vec![-0.7; VALUE_HEADS]).unwrap(),
            dt_bias: Tensor::from_f32(vec![VALUE_HEADS], &vec![0.0; VALUE_HEADS]).unwrap(),
            norm: Tensor::from_f32(vec![HEAD_DIM], &vec![1.0; HEAD_DIM]).unwrap(),
            out: checkpoint_matrix(HIDDEN, value_width, 29),
        };
        let hidden = (0..HIDDEN)
            .map(|index| ((index % 251) as f32 - 125.0) / 251.0)
            .collect::<Vec<_>>();
        let backend = CpuBackend;
        let mut sequential_projection_samples = Vec::with_capacity(SAMPLES);
        let mut parallel_projection_samples = Vec::with_capacity(SAMPLES);
        let mut sequential_projection = (Vec::new(), Vec::new());
        let mut parallel_projection = (Vec::new(), Vec::new());
        for sample in 0..=SAMPLES {
            let start = std::time::Instant::now();
            let sequential_qkv = weights.in_qkv.apply(&hidden).unwrap();
            let sequential_z = weights.in_z.apply(&hidden).unwrap();
            if sample > 0 {
                sequential_projection_samples.push(start.elapsed().as_secs_f64() * 1_000.0);
            }

            let start = std::time::Instant::now();
            let (parallel_qkv, parallel_z) = rayon::join(
                || weights.in_qkv.apply(&hidden),
                || weights.in_z.apply(&hidden),
            );
            if sample > 0 {
                parallel_projection_samples.push(start.elapsed().as_secs_f64() * 1_000.0);
            }
            sequential_projection = (sequential_qkv, sequential_z);
            parallel_projection = (parallel_qkv.unwrap(), parallel_z.unwrap());
        }
        sequential_projection_samples.sort_by(f64::total_cmp);
        parallel_projection_samples.sort_by(f64::total_cmp);
        assert_eq!(sequential_projection, parallel_projection);
        std::hint::black_box((sequential_projection, parallel_projection));
        eprintln!(
            "qwen4_exp_bf16_gdn_input_projections sequential_median_ms={:.6} parallel_median_ms={:.6}",
            sequential_projection_samples[SAMPLES / 2],
            parallel_projection_samples[SAMPLES / 2]
        );

        let mut state = GdnState::default();
        std::hint::black_box(run_gdn(&backend, &config, &hidden, &weights, &mut state).unwrap());
        let mut samples = Vec::with_capacity(SAMPLES);
        let mut checksum = 0.0f64;
        for _ in 0..SAMPLES {
            let start = std::time::Instant::now();
            let output = run_gdn(&backend, &config, &hidden, &weights, &mut state).unwrap();
            samples.push(start.elapsed().as_secs_f64() * 1_000.0);
            checksum = output.iter().map(|value| *value as f64).sum();
            std::hint::black_box(output);
        }
        samples.sort_by(f64::total_cmp);
        eprintln!(
            "qwen4_exp_bf16_gdn hidden={HIDDEN} key_heads={KEY_HEADS} value_heads={VALUE_HEADS} head_dim={HEAD_DIM} median_ms={:.6} checksum={checksum:.9}",
            samples[SAMPLES / 2]
        );
    }

    #[test]
    #[ignore = "manual performance benchmark"]
    fn benchmark_checkpoint_bf16_qsa_kvi() {
        const HIDDEN: usize = 2560;
        const KV_WIDTH: usize = 512;
        const INDEX_WIDTH: usize = 640;
        const SAMPLES: usize = 21;

        let checkpoint_matrix = |rows: usize, multiplier: usize| {
            let values = (0..rows * HIDDEN)
                .map(|index| {
                    half::bf16::from_f32(
                        ((index.wrapping_mul(multiplier) % 509) as f32 - 254.0) / 509.0,
                    )
                })
                .collect::<Vec<_>>();
            ResidentMatrix::Checkpoint(Tensor::from_bf16(vec![rows, HIDDEN], &values).unwrap())
        };
        let k = checkpoint_matrix(KV_WIDTH, 13);
        let v = checkpoint_matrix(KV_WIDTH, 17);
        let index_qk = checkpoint_matrix(INDEX_WIDTH, 19);
        let hidden = (0..HIDDEN)
            .map(|index| ((index % 251) as f32 - 125.0) / 251.0)
            .collect::<Vec<_>>();
        let mut sequential_samples = Vec::with_capacity(SAMPLES);
        let mut parallel_samples = Vec::with_capacity(SAMPLES);
        let mut sequential = (Vec::new(), Vec::new(), Vec::new());
        let mut parallel = (Vec::new(), Vec::new(), Vec::new());
        for sample in 0..=SAMPLES {
            let start = std::time::Instant::now();
            let sequential_k = k.apply(&hidden).unwrap();
            let sequential_v = v.apply(&hidden).unwrap();
            let sequential_index = index_qk.apply(&hidden).unwrap();
            if sample > 0 {
                sequential_samples.push(start.elapsed().as_secs_f64() * 1_000.0);
            }

            let start = std::time::Instant::now();
            let (parallel_k, (parallel_v, parallel_index)) = rayon::join(
                || k.apply(&hidden),
                || rayon::join(|| v.apply(&hidden), || index_qk.apply(&hidden)),
            );
            if sample > 0 {
                parallel_samples.push(start.elapsed().as_secs_f64() * 1_000.0);
            }
            sequential = (sequential_k, sequential_v, sequential_index);
            parallel = (
                parallel_k.unwrap(),
                parallel_v.unwrap(),
                parallel_index.unwrap(),
            );
        }
        sequential_samples.sort_by(f64::total_cmp);
        parallel_samples.sort_by(f64::total_cmp);
        assert_eq!(sequential, parallel);
        std::hint::black_box((sequential, parallel));
        eprintln!(
            "qwen4_exp_bf16_qsa_kvi hidden={HIDDEN} kv_width={KV_WIDTH} index_width={INDEX_WIDTH} sequential_median_ms={:.6} parallel_median_ms={:.6}",
            sequential_samples[SAMPLES / 2],
            parallel_samples[SAMPLES / 2]
        );
    }

    #[test]
    #[ignore = "manual performance benchmark"]
    fn benchmark_qsa_sparse_attention() {
        const QUERY_HEADS: usize = 24;
        const KV_HEADS: usize = 2;
        const HEAD_DIM: usize = 256;
        const MAX_SELECTED: usize = 2048;
        const SAMPLES: usize = 9;

        let values = |length: usize, multiplier: usize| {
            (0..length)
                .map(|index| ((index.wrapping_mul(multiplier) % 509) as f32 - 254.0) / 509.0)
                .collect::<Vec<_>>()
        };
        let query = values(QUERY_HEADS * HEAD_DIM, 13);
        let keys = values(MAX_SELECTED * KV_HEADS * HEAD_DIM, 17);
        let values = values(MAX_SELECTED * KV_HEADS * HEAD_DIM, 19);
        for selected_len in [1usize, 8, 32, 64, 128, 512, 2048] {
            let selected = (0..selected_len).collect::<Vec<_>>();
            let mut sequential_samples = Vec::with_capacity(SAMPLES);
            let mut parallel_samples = Vec::with_capacity(SAMPLES);
            let mut sequential = Vec::new();
            let mut parallel = Vec::new();
            for _ in 0..SAMPLES {
                let start = std::time::Instant::now();
                sequential = sparse_attention_sequential_reference(
                    &query,
                    &keys,
                    &values,
                    &selected,
                    QUERY_HEADS,
                    KV_HEADS,
                    HEAD_DIM,
                );
                sequential_samples.push(start.elapsed().as_secs_f64() * 1_000.0);

                let start = std::time::Instant::now();
                parallel = sparse_attention(
                    &query,
                    &keys,
                    &values,
                    &selected,
                    QUERY_HEADS,
                    KV_HEADS,
                    HEAD_DIM,
                );
                parallel_samples.push(start.elapsed().as_secs_f64() * 1_000.0);
            }
            sequential_samples.sort_by(f64::total_cmp);
            parallel_samples.sort_by(f64::total_cmp);
            assert_eq!(sequential, parallel);
            let checksum = parallel.iter().map(|value| *value as f64).sum::<f64>();
            std::hint::black_box((sequential, parallel));
            eprintln!(
                "qwen4_exp_qsa_sparse_attention selected={selected_len} sequential_median_ms={:.6} parallel_median_ms={:.6} checksum={checksum:.9}",
                sequential_samples[SAMPLES / 2],
                parallel_samples[SAMPLES / 2]
            );
        }
    }

    #[test]
    fn f32_dot_matches_sequential_reference() {
        for length in [0usize, 1, 7, 8, 9, 15, 16, 17, 127, 256, 4096, 4103] {
            let left = (0..length)
                .map(|index| ((index % 251) as f32 - 125.0) / 251.0)
                .collect::<Vec<_>>();
            let right = (0..length)
                .map(|index| ((index.wrapping_mul(17) % 509) as f32 - 254.0) / 509.0)
                .collect::<Vec<_>>();
            let expected = left
                .iter()
                .zip(&right)
                .map(|(left, right)| left * right)
                .sum::<f32>();
            let actual = dot(&left, &right);
            let scalar = dot_f32_scalar(&left, &right);
            let tolerance = 5.0e-4 + 5.0e-5 * expected.abs();
            assert!(
                (actual - expected).abs() <= tolerance,
                "length={length} actual={actual} expected={expected} tolerance={tolerance}"
            );
            assert!(
                (scalar - expected).abs() <= tolerance,
                "length={length} scalar={scalar} expected={expected} tolerance={tolerance}"
            );
        }
    }

    #[test]
    fn qsa_sparse_attention_parallel_matches_sequential_heads() {
        const QUERY_HEADS: usize = 4;
        const KV_HEADS: usize = 2;
        const HEAD_DIM: usize = 256;
        const SELECTED: usize = 512;

        let values = |length: usize, multiplier: usize| {
            (0..length)
                .map(|index| ((index.wrapping_mul(multiplier) % 509) as f32 - 254.0) / 509.0)
                .collect::<Vec<_>>()
        };
        let query = values(QUERY_HEADS * HEAD_DIM, 13);
        let keys = values(SELECTED * KV_HEADS * HEAD_DIM, 17);
        let values = values(SELECTED * KV_HEADS * HEAD_DIM, 19);
        let selected = (0..SELECTED).collect::<Vec<_>>();
        let expected = sparse_attention_sequential_reference(
            &query,
            &keys,
            &values,
            &selected,
            QUERY_HEADS,
            KV_HEADS,
            HEAD_DIM,
        );
        let actual = sparse_attention(
            &query,
            &keys,
            &values,
            &selected,
            QUERY_HEADS,
            KV_HEADS,
            HEAD_DIM,
        );
        assert_eq!(actual, expected);
    }

    #[test]
    fn bf16_dot_matches_sequential_reference() {
        for length in [0usize, 1, 7, 8, 9, 15, 16, 17, 127, 4096, 4103] {
            let left = (0..length)
                .map(|index| ((index % 251) as f32 - 125.0) / 251.0)
                .collect::<Vec<_>>();
            let right = (0..length)
                .map(|index| {
                    half::bf16::from_f32(((index.wrapping_mul(17) % 509) as f32 - 254.0) / 509.0)
                })
                .collect::<Vec<_>>();
            let expected = left
                .iter()
                .zip(&right)
                .map(|(left, right)| left * right.to_f32())
                .sum::<f32>();
            let actual = dot_bf16(&left, &right);
            let scalar = dot_bf16_scalar(&left, &right);
            let tolerance = 5.0e-4 + 5.0e-5 * expected.abs();
            assert!(
                (actual - expected).abs() <= tolerance,
                "length={length} actual={actual} expected={expected} tolerance={tolerance}"
            );
            assert!(
                (scalar - expected).abs() <= tolerance,
                "length={length} scalar={scalar} expected={expected} tolerance={tolerance}"
            );
        }
    }

    #[test]
    fn moe_router_partition_matches_full_ranking_and_tie_breaks() {
        let probabilities = (0usize..512)
            .map(|index| ((index.wrapping_mul(73) % 97) as f32 - 48.0) / 97.0)
            .collect::<Vec<_>>();
        for topk in [0usize, 1, 10, 511, 512, 600] {
            let mut expected = probabilities
                .iter()
                .copied()
                .enumerate()
                .collect::<Vec<_>>();
            expected.sort_unstable_by(|(left_index, left), (right_index, right)| {
                right
                    .total_cmp(left)
                    .then_with(|| left_index.cmp(right_index))
            });
            expected.truncate(topk.min(expected.len()));
            assert_eq!(rank_top_experts(&probabilities, topk), expected);
        }
    }

    #[test]
    fn large_bf16_experts_parallelize_without_changing_router_order() {
        const EXPERTS: usize = 2;
        const HIDDEN: usize = 1024;
        const INTERMEDIATE: usize = 704;

        let mut config = Qwen4ExpConfig::from_json_str(MINI_CONFIG).unwrap().text;
        config.hidden_size = HIDDEN;
        config.moe_intermediate_size = INTERMEDIATE;
        config.num_experts = EXPERTS;
        config.num_experts_per_tok = EXPERTS;
        let hidden = (0..HIDDEN)
            .map(|index| ((index % 251) as f32 - 125.0) / 251.0)
            .collect::<Vec<_>>();
        let make_weights = |length: usize, multiplier: usize| {
            (0..length)
                .map(|index| {
                    half::bf16::from_f32(
                        ((index.wrapping_mul(multiplier) % 509) as f32 - 254.0) / 509.0,
                    )
                })
                .collect::<Vec<_>>()
        };
        let packed_gate_up = make_weights(EXPERTS * 2 * INTERMEDIATE * HIDDEN, 13);
        let gate_up =
            Tensor::from_bf16(vec![EXPERTS, 2 * INTERMEDIATE, HIDDEN], &packed_gate_up).unwrap();
        let packed_down = make_weights(EXPERTS * HIDDEN * INTERMEDIATE, 29);
        let down = Tensor::from_bf16(vec![EXPERTS, HIDDEN, INTERMEDIATE], &packed_down).unwrap();
        let experts = ExpertWeights::Checkpoint { gate_up, down };
        let ranked = [(1usize, 0.625f32), (0usize, 0.375f32)];

        let actual = run_selected_experts(&config, &hidden, &experts, &ranked).unwrap();
        let mut expected = vec![0.0f32; HIDDEN];
        for (expert, probability) in ranked {
            let expert_output = experts.apply_bf16_serial(&config, &hidden, expert).unwrap();
            for (output, value) in expected.iter_mut().zip(expert_output) {
                *output += probability * value;
            }
        }
        assert_eq!(actual, expected);

        config.shared_expert_intermediate_size = INTERMEDIATE;
        let zero_matrix = |input: usize, output: usize| {
            ResidentMatrix::synthetic(vec![0.0; input * output], output)
        };
        let weights = MoeWeights {
            router: zero_matrix(HIDDEN, EXPERTS),
            experts,
            shared_gate: zero_matrix(HIDDEN, INTERMEDIATE),
            shared_up: zero_matrix(HIDDEN, INTERMEDIATE),
            shared_down: zero_matrix(INTERMEDIATE, HIDDEN),
            shared_expert_gate: zero_matrix(HIDDEN, 1),
        };
        let actual = run_moe(&config, &hidden, &weights).unwrap();
        let uniform_ranked = [(0usize, 0.5f32), (1usize, 0.5f32)];
        let mut expected =
            run_selected_experts(&config, &hidden, &weights.experts, &uniform_ranked).unwrap();
        let shared = run_shared_expert(&hidden, &weights).unwrap();
        for (output, shared) in expected.iter_mut().zip(shared) {
            *output += shared;
        }
        assert_eq!(actual, expected);
    }

    #[test]
    fn loads_a_complete_toy_safetensors_file() {
        static NEXT_FILE: AtomicU64 = AtomicU64::new(0);
        let (config, tensors) = crate::qwen4_exp::weights::tests::zero_runtime_tensors();
        let directory = std::env::temp_dir().join(format!(
            "apxinf-qwen4-exp-safetensors-{}-{}",
            std::process::id(),
            NEXT_FILE.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&directory).unwrap();
        let checkpoint = directory.join("model.safetensors");
        write_safetensors(&checkpoint, &tensors);
        let schema = Qwen4ExpWeightSchema::new(&config).unwrap();
        let runtime_names = schema
            .runtime_names()
            .iter()
            .cloned()
            .collect::<HashSet<_>>();
        let (loaded, _) =
            apxinf_loader::safetensors::load_native_path_mmap_filtered(&checkpoint, |name| {
                runtime_names.contains(name)
            })
            .unwrap();
        assert!(matches!(
            loaded["model.language_model.embed_tokens.weight"].storage(),
            apxinf_core::Storage::CpuMmap { .. }
        ));
        let mut model = GeneralQwen4Exp::from_tensors(config, loaded, 64).unwrap();
        let logits = model.forward(&[1, 3, 5], 0).unwrap();
        assert_eq!(logits.shape().dims(), [3, 32]);
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn checkpoint_query_gate_is_deinterleaved_per_head() {
        let config = Qwen4ExpConfig::from_json_str(MINI_CONFIG).unwrap();
        let text = &config.text;
        let packed = (0..text.full_q_projection_width() * text.hidden_size)
            .map(|value| value as f32)
            .collect::<Vec<_>>();
        let weights = QueryGateWeights::Checkpoint(
            Tensor::from_f32(
                vec![text.full_q_projection_width(), text.hidden_size],
                &packed,
            )
            .unwrap(),
        );
        for input in 0..text.hidden_size {
            let mut hidden = vec![0.0; text.hidden_size];
            hidden[input] = 1.0;
            let (query, gate) = weights.apply(text, &hidden).unwrap();
            for head in 0..text.n_attention_heads {
                for column in 0..text.head_dim {
                    let output = head * text.head_dim + column;
                    let packed_query = head * 2 * text.head_dim + column;
                    assert_eq!(
                        query[output],
                        packed[packed_query * text.hidden_size + input]
                    );
                    assert_eq!(
                        gate[output],
                        packed[(packed_query + text.head_dim) * text.hidden_size + input]
                    );
                }
            }
        }
    }

    #[test]
    fn auto_model_registers_the_explicit_synthetic_path() {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let directory = std::env::temp_dir().join(format!(
            "apxinf-qwen4-exp-synthetic-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&directory).unwrap();
        let mut text_only: serde_json::Value = serde_json::from_str(MINI_CONFIG).unwrap();
        text_only["language_model_only"] = true.into();
        std::fs::write(directory.join("config.json"), text_only.to_string()).unwrap();
        let options = LoadOptions {
            synthetic: Some(SyntheticWeights { seed: 38 }),
            max_context: Some(64),
            ..LoadOptions::default()
        };
        let mut loaded = AutoModel::load_model(Device::Cpu, &directory, &options).unwrap();
        let logits = loaded.forward(&[1, 2, 3], 0).unwrap();
        assert_eq!(logits.shape().dims(), [3, 32]);
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn auto_model_loads_the_real_toy_checkpoint_path() {
        static NEXT_REAL: AtomicU64 = AtomicU64::new(0);
        let (_config, tensors) = crate::qwen4_exp::weights::tests::zero_runtime_tensors();
        let directory = std::env::temp_dir().join(format!(
            "apxinf-qwen4-exp-auto-real-{}-{}",
            std::process::id(),
            NEXT_REAL.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&directory).unwrap();
        let mut text_only: serde_json::Value = serde_json::from_str(MINI_CONFIG).unwrap();
        text_only["language_model_only"] = true.into();
        std::fs::write(directory.join("config.json"), text_only.to_string()).unwrap();
        write_safetensors(&directory.join("model.safetensors"), &tensors);
        let options = LoadOptions {
            max_context: Some(64),
            ..LoadOptions::default()
        };
        let mut loaded = AutoModel::load_model(Device::Cpu, &directory, &options).unwrap();
        let logits = loaded.forward(&[1, 2, 3], 0).unwrap();
        assert_eq!(logits.shape().dims(), [3, 32]);
        std::fs::remove_dir_all(directory).unwrap();
    }
}
