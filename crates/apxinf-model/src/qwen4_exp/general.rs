//! Correctness-first single-request Qwen4-Exp text runtime.
//!
//! This first slice accepts deterministic downsized synthetic weights only.
//! It executes the released architecture and cache semantics; it is not a
//! production checkpoint loader or performance path.

use std::collections::HashMap;
use std::sync::Arc;

use apxinf_core::{Backend, CpuBackend, DType, Device, Error, Result, Tensor};
use apxinf_loader::ModelConfig;

use super::config::{Qwen4ExpConfig, Qwen4ExpLayerType, Qwen4ExpTextConfig};
use super::qsa::{apply_partial_rope, rms_norm_zero_centered_into, Qwen4ExpQsaSelector};
use super::weights::{metadata_from_tensors, ple_vocab_layout, Qwen4ExpWeightSchema};
use crate::llm_trait::LlmTrait;

pub struct GeneralQwen4Exp {
    config: Qwen4ExpConfig,
    weights: RuntimeWeights,
    state: RuntimeState,
    backend: Arc<dyn Backend>,
    max_context: usize,
    weight_source: &'static str,
    checkpoint_payloads_mmap: bool,
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
        })
    }

    fn forward_one(&mut self, token: u32, position: usize) -> Result<Vec<f32>> {
        let text = &self.config.text;
        if token as usize >= text.vocab_size {
            return Err(Error::Other(format!(
                "qwen4-exp token {token} is outside vocabulary {}",
                text.vocab_size
            )));
        }
        let embedding = self
            .weights
            .token_embedding
            .row(token as usize, text.hidden_size)?;
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
        self.weights.lm_head.apply(&hidden)
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
        let mut logits = Vec::with_capacity(token_ids.len() * self.config.text.vocab_size);
        for (offset, &token) in token_ids.iter().enumerate() {
            logits.extend(self.forward_one(token, start_pos + offset)?);
            self.state.position += 1;
        }
        Tensor::from_f32(vec![token_ids.len(), self.config.text.vocab_size], &logits)
    }

    fn reset(&mut self) {
        self.state = RuntimeState::new(&self.config.text);
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
            "vision": false,
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
    let qkv = Tensor::from_f32(vec![1, qkv_width], &weights.in_qkv.apply(hidden)?)?;
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
    let z = weights.in_z.apply(hidden)?.into_iter().map(|value| {
        match config.output_gate_type.as_str() {
            "sigmoid" => sigmoid(value),
            _ => silu(value),
        }
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
    position: usize,
    weights: &QsaWeights,
    state: &mut QsaState,
) -> Result<Vec<f32>> {
    let (mut query, gate) = weights.query_gate.apply(config, hidden)?;
    let gate = gate.into_iter().map(sigmoid).collect::<Vec<_>>();
    for head in query.chunks_exact_mut(config.head_dim) {
        let source = head.to_vec();
        rms_norm_zero_centered_into(&source, &weights.q_norm, config.rms_norm_eps, head);
        apply_partial_rope(head, config.rotary_dim(), config.rope.theta, position);
    }
    let mut key = weights.k.apply(hidden)?;
    for head in key.chunks_exact_mut(config.head_dim) {
        let source = head.to_vec();
        rms_norm_zero_centered_into(&source, &weights.k_norm, config.rms_norm_eps, head);
        apply_partial_rope(head, config.rotary_dim(), config.rope.theta, position);
    }
    let value = weights.v.apply(hidden)?;
    let index_qk = weights.index_qk.apply(hidden)?;
    let query_width = config.indexer_n_heads * config.indexer_head_dim;
    let index_query = &index_qk[..query_width];
    state.raw_keys.extend_from_slice(&index_qk[query_width..]);
    state.keys.extend_from_slice(&key);
    state.values.extend_from_slice(&value);
    let visible = state.raw_keys.len() / config.indexer_head_dim;
    let selected = weights.selector.select(
        index_query,
        &state.raw_keys,
        visible,
        &weights.index_q_norm,
        &weights.index_k_norm,
    )?;

    let mut attention = vec![0.0; config.full_query_width()];
    let queries_per_kv = config.n_attention_heads / config.n_kv_heads;
    let scale = 1.0 / (config.head_dim as f32).sqrt();
    let mut scores = vec![0.0; selected.len()];
    for query_head in 0..config.n_attention_heads {
        let kv_head = query_head / queries_per_kv;
        let query_row = &query[query_head * config.head_dim..(query_head + 1) * config.head_dim];
        for (slot, &token) in selected.iter().enumerate() {
            let key_start = (token * config.n_kv_heads + kv_head) * config.head_dim;
            scores[slot] = dot(
                query_row,
                &state.keys[key_start..key_start + config.head_dim],
            ) * scale;
        }
        softmax_in_place(&mut scores);
        let output_start = query_head * config.head_dim;
        for (slot, &token) in selected.iter().enumerate() {
            let value_start = (token * config.n_kv_heads + kv_head) * config.head_dim;
            for column in 0..config.head_dim {
                attention[output_start + column] +=
                    scores[slot] * state.values[value_start + column];
            }
        }
    }
    for (attention, gate) in attention.iter_mut().zip(gate) {
        *attention *= gate;
    }
    weights.out.apply(&attention)
}

fn run_moe(config: &Qwen4ExpTextConfig, hidden: &[f32], weights: &MoeWeights) -> Result<Vec<f32>> {
    let mut probabilities = weights.router.apply(hidden)?;
    softmax_in_place(&mut probabilities);
    let mut ranked = probabilities
        .iter()
        .copied()
        .enumerate()
        .collect::<Vec<_>>();
    ranked.sort_unstable_by(|(left_index, left), (right_index, right)| {
        right
            .total_cmp(left)
            .then_with(|| left_index.cmp(right_index))
    });
    ranked.truncate(config.num_experts_per_tok);
    if config.norm_topk_prob {
        let sum = ranked.iter().map(|(_, value)| value).sum::<f32>();
        for (_, value) in &mut ranked {
            *value /= sum;
        }
    }
    let mut output = vec![0.0; config.hidden_size];
    for (expert, probability) in ranked {
        let gate_up = weights.experts.gate_up(config, hidden, expert)?;
        let activated = (0..config.moe_intermediate_size)
            .map(|index| silu(gate_up[index]) * gate_up[config.moe_intermediate_size + index])
            .collect::<Vec<_>>();
        let expert_output = weights.experts.down(config, &activated, expert)?;
        for (output, value) in output.iter_mut().zip(expert_output) {
            *output += probability * value;
        }
    }
    let shared_gate = weights.shared_gate.apply(hidden)?;
    let shared_up = weights.shared_up.apply(hidden)?;
    let shared = shared_gate
        .into_iter()
        .zip(shared_up)
        .map(|(gate, up)| silu(gate) * up)
        .collect::<Vec<_>>();
    let shared = weights.shared_down.apply(&shared)?;
    let gate = sigmoid(weights.shared_expert_gate.apply(hidden)?[0]);
    for (output, shared) in output.iter_mut().zip(shared) {
        *output += gate * shared;
    }
    Ok(output)
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
            for (row, output) in output.iter_mut().enumerate() {
                let start = offset + row * input_size;
                *output = input
                    .iter()
                    .zip(&weights[start..start + input_size])
                    .map(|(input, weight)| input * weight.to_f32())
                    .sum();
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

fn dot(left: &[f32], right: &[f32]) -> f32 {
    left.iter()
        .zip(right)
        .map(|(left, right)| left * right)
        .sum()
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
        std::fs::write(directory.join("config.json"), MINI_CONFIG).unwrap();
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
        std::fs::write(directory.join("config.json"), MINI_CONFIG).unwrap();
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
