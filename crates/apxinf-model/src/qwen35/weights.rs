//! Native Qwen3.5 text weight schema and loader.
//!
//! Tensors stay in their Hugging Face `[out, in]` checkpoint layout here.
//! Backend-specific packing belongs in the runtime so CPU/Accelerate and a
//! future Metal backend can choose different representations without changing
//! checkpoint validation.

use std::collections::{HashMap, HashSet};

use apxinf_core::{DType, Error, Result, Tensor};

use super::config::{Qwen35Config, Qwen35LayerType};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Qwen35WeightMetadata {
    pub shape: Vec<usize>,
    pub dtype: DType,
}

impl Qwen35WeightMetadata {
    pub fn new(shape: impl Into<Vec<usize>>, dtype: DType) -> Self {
        Self {
            shape: shape.into(),
            dtype,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Qwen35WeightSpec {
    pub name: String,
    pub shape: Vec<usize>,
    /// Dtype used by the unmodified HF checkpoint. Runtime validation accepts
    /// other floating dtypes so a CPU loader may upcast before constructing
    /// `Qwen35TextWeights`.
    pub checkpoint_dtype: DType,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Qwen35WeightValidation {
    pub text_tensors: usize,
    pub ignored_vision_tensors: usize,
    pub ignored_mtp_tensors: usize,
}

/// Shape contract for the Qwen3.5 base autoregressive text stack.
#[derive(Clone, Debug)]
pub struct Qwen35WeightSchema {
    specs: Vec<Qwen35WeightSpec>,
}

impl Qwen35WeightSchema {
    pub fn new(config: &Qwen35Config) -> Result<Self> {
        config.text.validate()?;
        let text_dtype = checkpoint_dtype(&config.text.dtype)?;
        let state_dtype = checkpoint_dtype(&config.text.recurrent_state_dtype)?;
        let text = &config.text;
        let mut specs = Vec::new();

        push_spec(
            &mut specs,
            "model.language_model.embed_tokens.weight",
            [text.vocab_size, text.hidden_size],
            text_dtype,
        );

        for (index, layer_type) in text.layer_types.iter().copied().enumerate() {
            let prefix = format!("model.language_model.layers.{index}");
            push_spec(
                &mut specs,
                format!("{prefix}.input_layernorm.weight"),
                [text.hidden_size],
                text_dtype,
            );

            match layer_type {
                Qwen35LayerType::LinearAttention => {
                    let attention = format!("{prefix}.linear_attn");
                    push_spec(
                        &mut specs,
                        format!("{attention}.A_log"),
                        [text.linear_num_value_heads],
                        state_dtype,
                    );
                    push_spec(
                        &mut specs,
                        format!("{attention}.conv1d.weight"),
                        [text.linear_qkv_width(), 1, text.linear_conv_kernel_dim],
                        text_dtype,
                    );
                    push_spec(
                        &mut specs,
                        format!("{attention}.dt_bias"),
                        [text.linear_num_value_heads],
                        text_dtype,
                    );
                    push_spec(
                        &mut specs,
                        format!("{attention}.in_proj_a.weight"),
                        [text.linear_num_value_heads, text.hidden_size],
                        text_dtype,
                    );
                    push_spec(
                        &mut specs,
                        format!("{attention}.in_proj_b.weight"),
                        [text.linear_num_value_heads, text.hidden_size],
                        text_dtype,
                    );
                    push_spec(
                        &mut specs,
                        format!("{attention}.in_proj_qkv.weight"),
                        [text.linear_qkv_width(), text.hidden_size],
                        text_dtype,
                    );
                    push_spec(
                        &mut specs,
                        format!("{attention}.in_proj_z.weight"),
                        [text.linear_value_width(), text.hidden_size],
                        text_dtype,
                    );
                    push_spec(
                        &mut specs,
                        format!("{attention}.norm.weight"),
                        [text.linear_value_head_dim],
                        state_dtype,
                    );
                    push_spec(
                        &mut specs,
                        format!("{attention}.out_proj.weight"),
                        [text.hidden_size, text.linear_value_width()],
                        text_dtype,
                    );
                }
                Qwen35LayerType::FullAttention => {
                    let attention = format!("{prefix}.self_attn");
                    push_spec(
                        &mut specs,
                        format!("{attention}.q_proj.weight"),
                        [text.full_q_projection_width(), text.hidden_size],
                        text_dtype,
                    );
                    push_spec(
                        &mut specs,
                        format!("{attention}.k_proj.weight"),
                        [text.full_kv_width(), text.hidden_size],
                        text_dtype,
                    );
                    push_spec(
                        &mut specs,
                        format!("{attention}.v_proj.weight"),
                        [text.full_kv_width(), text.hidden_size],
                        text_dtype,
                    );
                    push_spec(
                        &mut specs,
                        format!("{attention}.o_proj.weight"),
                        [text.hidden_size, text.full_query_width()],
                        text_dtype,
                    );
                    push_spec(
                        &mut specs,
                        format!("{attention}.q_norm.weight"),
                        [text.head_dim],
                        text_dtype,
                    );
                    push_spec(
                        &mut specs,
                        format!("{attention}.k_norm.weight"),
                        [text.head_dim],
                        text_dtype,
                    );
                }
            }

            push_spec(
                &mut specs,
                format!("{prefix}.post_attention_layernorm.weight"),
                [text.hidden_size],
                text_dtype,
            );
            push_spec(
                &mut specs,
                format!("{prefix}.mlp.gate_proj.weight"),
                [text.intermediate_size, text.hidden_size],
                text_dtype,
            );
            push_spec(
                &mut specs,
                format!("{prefix}.mlp.up_proj.weight"),
                [text.intermediate_size, text.hidden_size],
                text_dtype,
            );
            push_spec(
                &mut specs,
                format!("{prefix}.mlp.down_proj.weight"),
                [text.hidden_size, text.intermediate_size],
                text_dtype,
            );
        }

        push_spec(
            &mut specs,
            "model.language_model.norm.weight",
            [text.hidden_size],
            text_dtype,
        );
        if !text.tie_word_embeddings {
            push_spec(
                &mut specs,
                "lm_head.weight",
                [text.vocab_size, text.hidden_size],
                text_dtype,
            );
        }
        Ok(Self { specs })
    }

    pub fn specs(&self) -> &[Qwen35WeightSpec] {
        &self.specs
    }

    /// Validate shapes and floating dtypes after any loader-side upcast.
    pub fn validate_metadata(
        &self,
        metadata: &HashMap<String, Qwen35WeightMetadata>,
    ) -> Result<Qwen35WeightValidation> {
        self.validate(metadata, false)
    }

    /// Validate the exact storage dtypes declared by the original HF config.
    /// This is intended for checkpoint intake/auditing, not a runtime map that
    /// has already been upcast for a CPU backend.
    pub fn validate_checkpoint_metadata(
        &self,
        metadata: &HashMap<String, Qwen35WeightMetadata>,
    ) -> Result<Qwen35WeightValidation> {
        self.validate(metadata, true)
    }

    fn validate(
        &self,
        metadata: &HashMap<String, Qwen35WeightMetadata>,
        exact_checkpoint_dtype: bool,
    ) -> Result<Qwen35WeightValidation> {
        for spec in &self.specs {
            let actual = metadata.get(&spec.name).ok_or_else(|| {
                Error::Other(format!("qwen3.5 checkpoint: missing {}", spec.name))
            })?;
            if actual.shape != spec.shape {
                return Err(Error::Other(format!(
                    "qwen3.5 checkpoint {}: expected shape {:?}, got {:?}",
                    spec.name, spec.shape, actual.shape
                )));
            }
            if !matches!(actual.dtype, DType::F16 | DType::BF16 | DType::F32) {
                return Err(Error::Other(format!(
                    "qwen3.5 checkpoint {}: unsupported dtype {}",
                    spec.name, actual.dtype
                )));
            }
            if exact_checkpoint_dtype && actual.dtype != spec.checkpoint_dtype {
                return Err(Error::Other(format!(
                    "qwen3.5 checkpoint {}: expected checkpoint dtype {}, got {}",
                    spec.name, spec.checkpoint_dtype, actual.dtype
                )));
            }
        }

        let expected = self
            .specs
            .iter()
            .map(|spec| spec.name.as_str())
            .collect::<HashSet<_>>();
        let mut unknown = metadata
            .keys()
            .filter(|name| {
                !expected.contains(name.as_str())
                    && !name.starts_with("model.visual.")
                    && !name.starts_with("mtp.")
            })
            .cloned()
            .collect::<Vec<_>>();
        unknown.sort_unstable();
        if let Some(name) = unknown.first() {
            return Err(Error::Other(format!(
                "qwen3.5 checkpoint: unexpected tensor {name}; only model.visual.* and mtp.* \
                 are explicitly ignored by the text-only loader"
            )));
        }

        Ok(Qwen35WeightValidation {
            text_tensors: self.specs.len(),
            ignored_vision_tensors: metadata
                .keys()
                .filter(|name| name.starts_with("model.visual."))
                .count(),
            ignored_mtp_tensors: metadata
                .keys()
                .filter(|name| name.starts_with("mtp."))
                .count(),
        })
    }
}

pub fn metadata_from_tensors(
    tensors: &HashMap<String, Tensor>,
) -> HashMap<String, Qwen35WeightMetadata> {
    tensors
        .iter()
        .map(|(name, tensor)| {
            (
                name.clone(),
                Qwen35WeightMetadata {
                    shape: tensor.shape().dims().to_vec(),
                    dtype: tensor.dtype(),
                },
            )
        })
        .collect()
}

pub struct Qwen35TextWeights {
    /// `[vocab_size, hidden_size]`; also the output weight when embeddings are tied.
    pub token_embedding: Tensor,
    pub layers: Vec<Qwen35LayerWeights>,
    pub output_norm_weight: Tensor,
    /// Present only for checkpoints with untied word embeddings.
    pub lm_head_weight: Option<Tensor>,
}

pub struct Qwen35LayerWeights {
    pub input_norm_weight: Tensor,
    pub attention: Qwen35AttentionWeights,
    pub post_attention_norm_weight: Tensor,
    pub mlp: Qwen35MlpWeights,
}

pub enum Qwen35AttentionWeights {
    Linear(Qwen35LinearAttentionWeights),
    Full(Qwen35FullAttentionWeights),
}

pub struct Qwen35LinearAttentionWeights {
    pub a_log: Tensor,
    pub conv1d_weight: Tensor,
    pub dt_bias: Tensor,
    pub in_proj_a_weight: Tensor,
    pub in_proj_b_weight: Tensor,
    pub in_proj_qkv_weight: Tensor,
    pub in_proj_z_weight: Tensor,
    pub norm_weight: Tensor,
    pub out_proj_weight: Tensor,
}

pub struct Qwen35FullAttentionWeights {
    /// `[2 * num_heads * head_dim, hidden]` when `attn_output_gate=true`.
    pub q_proj_weight: Tensor,
    pub k_proj_weight: Tensor,
    pub v_proj_weight: Tensor,
    pub o_proj_weight: Tensor,
    pub q_norm_weight: Tensor,
    pub k_norm_weight: Tensor,
}

pub struct Qwen35MlpWeights {
    pub gate_proj_weight: Tensor,
    pub up_proj_weight: Tensor,
    pub down_proj_weight: Tensor,
}

impl Qwen35TextWeights {
    /// Consume the base text stack. Vision and MTP tensors are accepted only
    /// under their explicit namespaces and then dropped for the text-only MVP.
    pub fn from_map(config: &Qwen35Config, mut tensors: HashMap<String, Tensor>) -> Result<Self> {
        let schema = Qwen35WeightSchema::new(config)?;
        schema.validate_metadata(&metadata_from_tensors(&tensors))?;

        let token_embedding = take(&mut tensors, "model.language_model.embed_tokens.weight")?;
        let mut layers = Vec::with_capacity(config.text.n_layers);
        for (index, layer_type) in config.text.layer_types.iter().copied().enumerate() {
            let prefix = format!("model.language_model.layers.{index}");
            let input_norm_weight =
                take(&mut tensors, &format!("{prefix}.input_layernorm.weight"))?;
            let attention = match layer_type {
                Qwen35LayerType::LinearAttention => {
                    let attention = format!("{prefix}.linear_attn");
                    Qwen35AttentionWeights::Linear(Qwen35LinearAttentionWeights {
                        a_log: take(&mut tensors, &format!("{attention}.A_log"))?,
                        conv1d_weight: take(&mut tensors, &format!("{attention}.conv1d.weight"))?,
                        dt_bias: take(&mut tensors, &format!("{attention}.dt_bias"))?,
                        in_proj_a_weight: take(
                            &mut tensors,
                            &format!("{attention}.in_proj_a.weight"),
                        )?,
                        in_proj_b_weight: take(
                            &mut tensors,
                            &format!("{attention}.in_proj_b.weight"),
                        )?,
                        in_proj_qkv_weight: take(
                            &mut tensors,
                            &format!("{attention}.in_proj_qkv.weight"),
                        )?,
                        in_proj_z_weight: take(
                            &mut tensors,
                            &format!("{attention}.in_proj_z.weight"),
                        )?,
                        norm_weight: take(&mut tensors, &format!("{attention}.norm.weight"))?,
                        out_proj_weight: take(
                            &mut tensors,
                            &format!("{attention}.out_proj.weight"),
                        )?,
                    })
                }
                Qwen35LayerType::FullAttention => {
                    let attention = format!("{prefix}.self_attn");
                    Qwen35AttentionWeights::Full(Qwen35FullAttentionWeights {
                        q_proj_weight: take(&mut tensors, &format!("{attention}.q_proj.weight"))?,
                        k_proj_weight: take(&mut tensors, &format!("{attention}.k_proj.weight"))?,
                        v_proj_weight: take(&mut tensors, &format!("{attention}.v_proj.weight"))?,
                        o_proj_weight: take(&mut tensors, &format!("{attention}.o_proj.weight"))?,
                        q_norm_weight: take(&mut tensors, &format!("{attention}.q_norm.weight"))?,
                        k_norm_weight: take(&mut tensors, &format!("{attention}.k_norm.weight"))?,
                    })
                }
            };
            let post_attention_norm_weight = take(
                &mut tensors,
                &format!("{prefix}.post_attention_layernorm.weight"),
            )?;
            let mlp = Qwen35MlpWeights {
                gate_proj_weight: take(&mut tensors, &format!("{prefix}.mlp.gate_proj.weight"))?,
                up_proj_weight: take(&mut tensors, &format!("{prefix}.mlp.up_proj.weight"))?,
                down_proj_weight: take(&mut tensors, &format!("{prefix}.mlp.down_proj.weight"))?,
            };
            layers.push(Qwen35LayerWeights {
                input_norm_weight,
                attention,
                post_attention_norm_weight,
                mlp,
            });
        }

        let output_norm_weight = take(&mut tensors, "model.language_model.norm.weight")?;
        let lm_head_weight = if config.text.tie_word_embeddings {
            None
        } else {
            Some(take(&mut tensors, "lm_head.weight")?)
        };
        Ok(Self {
            token_embedding,
            layers,
            output_norm_weight,
            lm_head_weight,
        })
    }
}

fn checkpoint_dtype(name: &str) -> Result<DType> {
    match name {
        "float32" => Ok(DType::F32),
        "float16" => Ok(DType::F16),
        "bfloat16" => Ok(DType::BF16),
        other => Err(Error::Other(format!(
            "qwen3.5 checkpoint: unsupported configured dtype `{other}`"
        ))),
    }
}

fn push_spec<const N: usize>(
    specs: &mut Vec<Qwen35WeightSpec>,
    name: impl Into<String>,
    shape: [usize; N],
    checkpoint_dtype: DType,
) {
    specs.push(Qwen35WeightSpec {
        name: name.into(),
        shape: shape.to_vec(),
        checkpoint_dtype,
    });
}

fn take(tensors: &mut HashMap<String, Tensor>, name: &str) -> Result<Tensor> {
    tensors
        .remove(name)
        .ok_or_else(|| Error::Other(format!("qwen3.5 checkpoint: missing {name}")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::qwen35::config::tests::MINI_CONFIG;

    fn fixture() -> (Qwen35Config, Qwen35WeightSchema) {
        let config = Qwen35Config::from_json_str(MINI_CONFIG).unwrap();
        let schema = Qwen35WeightSchema::new(&config).unwrap();
        (config, schema)
    }

    fn synthetic_metadata(schema: &Qwen35WeightSchema) -> HashMap<String, Qwen35WeightMetadata> {
        schema
            .specs()
            .iter()
            .map(|spec| {
                (
                    spec.name.clone(),
                    Qwen35WeightMetadata::new(spec.shape.clone(), spec.checkpoint_dtype),
                )
            })
            .collect()
    }

    #[test]
    fn schema_matches_linear_and_gated_full_attention_shapes() {
        let (_, schema) = fixture();
        let specs = schema
            .specs()
            .iter()
            .map(|spec| (spec.name.as_str(), spec))
            .collect::<HashMap<_, _>>();

        assert_eq!(
            specs["model.language_model.layers.0.linear_attn.in_proj_qkv.weight"].shape,
            vec![24, 8]
        );
        assert_eq!(
            specs["model.language_model.layers.0.linear_attn.conv1d.weight"].shape,
            vec![24, 1, 4]
        );
        assert_eq!(
            specs["model.language_model.layers.0.linear_attn.norm.weight"].checkpoint_dtype,
            DType::F32
        );
        assert_eq!(
            specs["model.language_model.layers.3.self_attn.q_proj.weight"].shape,
            vec![32, 8]
        );
        assert_eq!(
            specs["model.language_model.layers.3.self_attn.o_proj.weight"].shape,
            vec![8, 16]
        );
    }

    #[test]
    fn validates_synthetic_metadata_and_reports_explicitly_ignored_towers() {
        let (_, schema) = fixture();
        let mut metadata = synthetic_metadata(&schema);
        metadata.insert(
            "model.visual.patch_embed.proj.weight".into(),
            Qwen35WeightMetadata::new(vec![8, 3, 2, 2, 2], DType::BF16),
        );
        metadata.insert(
            "mtp.layers.0.self_attn.q_proj.weight".into(),
            Qwen35WeightMetadata::new(vec![32, 8], DType::BF16),
        );
        let report = schema.validate_checkpoint_metadata(&metadata).unwrap();
        assert_eq!(report.text_tensors, schema.specs().len());
        assert_eq!(report.ignored_vision_tensors, 1);
        assert_eq!(report.ignored_mtp_tensors, 1);
    }

    #[test]
    fn shape_error_names_the_exact_checkpoint_tensor() {
        let (_, schema) = fixture();
        let mut metadata = synthetic_metadata(&schema);
        let name = "model.language_model.layers.0.linear_attn.conv1d.weight";
        metadata.get_mut(name).unwrap().shape = vec![24, 4];
        let error = schema.validate_metadata(&metadata).unwrap_err();
        assert!(error.to_string().contains(name));
        assert!(error.to_string().contains("[24, 1, 4]"));
        assert!(error.to_string().contains("[24, 4]"));
    }

    #[test]
    fn missing_and_unexpected_text_weights_fail_closed() {
        let (_, schema) = fixture();
        let mut metadata = synthetic_metadata(&schema);
        metadata.remove("model.language_model.layers.0.linear_attn.A_log");
        let error = schema.validate_metadata(&metadata).unwrap_err();
        assert!(error.to_string().contains("linear_attn.A_log"));

        let mut metadata = synthetic_metadata(&schema);
        metadata.insert(
            "model.language_model.layers.0.linear_attn.new_projection.weight".into(),
            Qwen35WeightMetadata::new(vec![1], DType::BF16),
        );
        let error = schema.validate_metadata(&metadata).unwrap_err();
        assert!(error.to_string().contains("unexpected tensor"));
        assert!(error.to_string().contains("new_projection.weight"));
    }

    #[test]
    fn runtime_validation_allows_cpu_upcast_but_checkpoint_audit_is_exact() {
        let (_, schema) = fixture();
        let mut metadata = synthetic_metadata(&schema);
        for value in metadata.values_mut() {
            value.dtype = DType::F32;
        }
        schema.validate_metadata(&metadata).unwrap();
        let error = schema.validate_checkpoint_metadata(&metadata).unwrap_err();
        assert!(error.to_string().contains("expected checkpoint dtype bf16"));
    }

    #[test]
    fn loader_consumes_a_small_synthetic_text_checkpoint() {
        let (config, schema) = fixture();
        let tensors = schema
            .specs()
            .iter()
            .map(|spec| {
                (
                    spec.name.clone(),
                    Tensor::zeros(spec.shape.clone(), spec.checkpoint_dtype),
                )
            })
            .collect();
        let weights = Qwen35TextWeights::from_map(&config, tensors).unwrap();
        assert_eq!(weights.layers.len(), 4);
        assert!(matches!(
            &weights.layers[0].attention,
            Qwen35AttentionWeights::Linear(_)
        ));
        assert!(matches!(
            &weights.layers[3].attention,
            Qwen35AttentionWeights::Full(_)
        ));
        assert!(weights.lm_head_weight.is_none());
    }
}
