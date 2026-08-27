//! Qwen3.8-Flash-Next text checkpoint-name contract.

use std::collections::{HashMap, HashSet};

use apxinf_core::{DType, Error, Result, Tensor};

use super::config::{Qwen4ExpConfig, Qwen4ExpLayerType, Qwen4ExpTextConfig};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Qwen4ExpWeightIndexValidation {
    pub text_tensors: usize,
    pub runtime_tensors: usize,
    pub ignored_vision_tensors: usize,
    pub ignored_mtp_tensors: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Qwen4ExpWeightMetadata {
    pub shape: Vec<usize>,
    pub dtype: DType,
}

impl Qwen4ExpWeightMetadata {
    pub fn new(shape: impl Into<Vec<usize>>, dtype: DType) -> Self {
        Self {
            shape: shape.into(),
            dtype,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Qwen4ExpRuntimeWeightValidation {
    pub tensors: usize,
    pub ple_shards: usize,
}

#[derive(Clone, Debug)]
enum RuntimeShape {
    Exact(Vec<usize>),
    PleShard {
        group: String,
        columns: usize,
        total_rows: usize,
    },
}

#[derive(Clone, Debug)]
pub struct Qwen4ExpWeightSchema {
    expected_names: Vec<String>,
    runtime_names: Vec<String>,
    runtime_shapes: HashMap<String, RuntimeShape>,
}

impl Qwen4ExpWeightSchema {
    pub fn new(config: &Qwen4ExpConfig) -> Result<Self> {
        let text = &config.text;
        let mut expected_names = vec![
            "model.language_model.embed_tokens.weight".to_owned(),
            "model.language_model.hyper_connection_mixer.hc_norm.weight".to_owned(),
            "model.language_model.hyper_connection_mixer.input_mix_weight_down.weight".to_owned(),
            "model.language_model.hyper_connection_mixer.input_mix_weight_up.weight".to_owned(),
            "lm_head.weight".to_owned(),
        ];
        let mut derived_names = HashSet::new();

        for (layer_index, layer_type) in text.layer_types.iter().copied().enumerate() {
            let prefix = format!("model.language_model.layers.{layer_index}");
            for connection in ["attn_hyper_connection", "mlp_hyper_connection"] {
                for suffix in [
                    "block_inject_weight.weight",
                    "hc_norm.weight",
                    "input_mix_weight_down.weight",
                    "input_mix_weight_up.weight",
                ] {
                    expected_names.push(format!("{prefix}.{connection}.{suffix}"));
                }
            }

            match layer_type {
                super::config::Qwen4ExpLayerType::LinearAttention => {
                    for suffix in [
                        "A_log",
                        "conv1d.weight",
                        "dt_bias",
                        "in_proj_a.weight",
                        "in_proj_b.weight",
                        "in_proj_qkv.weight",
                        "in_proj_z.weight",
                        "norm.weight",
                        "out_proj.weight",
                    ] {
                        expected_names.push(format!("{prefix}.linear_attn.{suffix}"));
                    }
                }
                super::config::Qwen4ExpLayerType::QwenSparseAttention => {
                    for suffix in [
                        "indexer.index_qk_proj.weight",
                        "indexer.k_layernorm.weight",
                        "indexer.q_layernorm.weight",
                        "k_norm.weight",
                        "k_proj.weight",
                        "o_proj.weight",
                        "q_norm.weight",
                        "q_proj.weight",
                        "v_proj.weight",
                    ] {
                        expected_names.push(format!("{prefix}.self_attn.{suffix}"));
                    }
                }
            }

            for suffix in [
                "experts.down_proj",
                "experts.gate_up_proj",
                "gate.weight",
                "shared_expert.down_proj.weight",
                "shared_expert.gate_proj.weight",
                "shared_expert.up_proj.weight",
                "shared_expert_gate.weight",
            ] {
                expected_names.push(format!("{prefix}.mlp.{suffix}"));
            }

            if text.ple_layer_ids.contains(&(layer_index + 1)) {
                for suffix in [
                    "conv1d.weight",
                    "key_proj.weight",
                    "norm_conv.weight",
                    "norm_key.weight",
                    "norm_query.weight",
                    "value_proj.weight",
                ] {
                    expected_names.push(format!("{prefix}.ple.{suffix}"));
                }
                for suffix in [
                    "layer_multipliers",
                    "ngram_heads_offsets",
                    "ngram_heads_vocab_sizes",
                ] {
                    let name = format!("{prefix}.ple.ple_embedding.{suffix}");
                    derived_names.insert(name.clone());
                    expected_names.push(name);
                }
                for shard in 0..text.split_ngram_parts {
                    expected_names.push(format!(
                        "{prefix}.ple.ple_embedding.ngram_embedding.shard_{shard}.weight"
                    ));
                }
            }
        }

        expected_names.sort_unstable();
        let original_len = expected_names.len();
        expected_names.dedup();
        if expected_names.len() != original_len {
            return Err(Error::Other(
                "qwen4-exp weight schema generated duplicate names".into(),
            ));
        }
        let runtime_names = expected_names
            .iter()
            .filter(|name| !derived_names.contains(name.as_str()))
            .cloned()
            .collect::<Vec<_>>();
        let runtime_shapes = runtime_names
            .iter()
            .map(|name| Ok((name.clone(), runtime_shape(text, name)?)))
            .collect::<Result<HashMap<_, _>>>()?;
        Ok(Self {
            expected_names,
            runtime_names,
            runtime_shapes,
        })
    }

    pub fn expected_names(&self) -> &[String] {
        &self.expected_names
    }

    pub fn runtime_names(&self) -> &[String] {
        &self.runtime_names
    }

    pub fn runtime_f32_bytes(&self) -> Result<u64> {
        let mut elements = 0u64;
        let mut ple_groups = HashSet::new();
        for shape in self.runtime_shapes.values() {
            let contribution = match shape {
                RuntimeShape::Exact(shape) => {
                    shape.iter().try_fold(1u64, |product, &dimension| {
                        product.checked_mul(dimension as u64).ok_or_else(|| {
                            Error::Other("qwen4-exp runtime parameter count overflow".into())
                        })
                    })?
                }
                RuntimeShape::PleShard {
                    group,
                    columns,
                    total_rows,
                } if ple_groups.insert(group) => (*columns as u64)
                    .checked_mul(*total_rows as u64)
                    .ok_or_else(|| Error::Other("qwen4-exp PLE parameter count overflow".into()))?,
                RuntimeShape::PleShard { .. } => 0,
            };
            elements = elements
                .checked_add(contribution)
                .ok_or_else(|| Error::Other("qwen4-exp runtime parameter count overflow".into()))?;
        }
        elements
            .checked_mul(4)
            .ok_or_else(|| Error::Other("qwen4-exp F32 runtime byte count overflow".into()))
    }

    pub fn runtime_hybrid_weight_bytes(&self) -> Result<u64> {
        let mut bytes = 0u64;
        let mut ple_groups = HashSet::new();
        for name in &self.runtime_names {
            let (elements, bytes_per_element) = match &self.runtime_shapes[name] {
                RuntimeShape::Exact(shape) => {
                    let elements = shape.iter().try_fold(1u64, |product, &dimension| {
                        product.checked_mul(dimension as u64).ok_or_else(|| {
                            Error::Other("qwen4-exp hybrid parameter count overflow".into())
                        })
                    })?;
                    let checkpoint_resident = shape.len() == 2 || name.contains(".mlp.experts.");
                    let bytes = if checkpoint_resident { 2 } else { 4 };
                    (elements, bytes)
                }
                RuntimeShape::PleShard {
                    group,
                    columns,
                    total_rows,
                } if ple_groups.insert(group) => (
                    (*columns as u64)
                        .checked_mul(*total_rows as u64)
                        .ok_or_else(|| {
                            Error::Other("qwen4-exp hybrid PLE count overflow".into())
                        })?,
                    2,
                ),
                RuntimeShape::PleShard { .. } => (0, 2),
            };
            bytes =
                bytes
                    .checked_add(elements.checked_mul(bytes_per_element).ok_or_else(|| {
                        Error::Other("qwen4-exp hybrid byte count overflow".into())
                    })?)
                    .ok_or_else(|| Error::Other("qwen4-exp hybrid byte count overflow".into()))?;
        }
        Ok(bytes)
    }

    pub fn validate_index_json_str(&self, raw: &str) -> Result<Qwen4ExpWeightIndexValidation> {
        let root: serde_json::Value = serde_json::from_str(raw)
            .map_err(|error| Error::Other(format!("qwen4-exp weight index json: {error}")))?;
        let weight_map = root
            .get("weight_map")
            .and_then(serde_json::Value::as_object)
            .ok_or_else(|| {
                Error::Other("qwen4-exp weight index: missing object weight_map".into())
            })?;
        if weight_map.is_empty() {
            return Err(Error::Other(
                "qwen4-exp weight index: weight_map is empty".into(),
            ));
        }
        if let Some((name, _)) = weight_map
            .iter()
            .find(|(_, shard)| shard.as_str().is_none_or(str::is_empty))
        {
            return Err(Error::Other(format!(
                "qwen4-exp weight index: tensor {name} has an invalid shard name"
            )));
        }

        let actual_names = weight_map
            .keys()
            .map(String::as_str)
            .collect::<HashSet<_>>();
        if let Some(missing) = self
            .expected_names
            .iter()
            .find(|name| !actual_names.contains(name.as_str()))
        {
            return Err(Error::Other(format!(
                "qwen4-exp weight index: missing text tensor {missing}"
            )));
        }

        let expected_names = self
            .expected_names
            .iter()
            .map(String::as_str)
            .collect::<HashSet<_>>();
        let mut unexpected = actual_names
            .iter()
            .copied()
            .filter(|name| {
                !expected_names.contains(name)
                    && !name.starts_with("model.visual.")
                    && !name.starts_with("mtp.")
            })
            .collect::<Vec<_>>();
        unexpected.sort_unstable();
        if let Some(name) = unexpected.first() {
            return Err(Error::Other(format!(
                "qwen4-exp weight index: unexpected tensor {name}; only model.visual.* and mtp.* are ignored by the text path"
            )));
        }

        Ok(Qwen4ExpWeightIndexValidation {
            text_tensors: self.expected_names.len(),
            runtime_tensors: self.runtime_names.len(),
            ignored_vision_tensors: actual_names
                .iter()
                .filter(|name| name.starts_with("model.visual."))
                .count(),
            ignored_mtp_tensors: actual_names
                .iter()
                .filter(|name| name.starts_with("mtp."))
                .count(),
        })
    }

    pub fn validate_runtime_metadata(
        &self,
        metadata: &HashMap<String, Qwen4ExpWeightMetadata>,
    ) -> Result<Qwen4ExpRuntimeWeightValidation> {
        let expected = self
            .runtime_names
            .iter()
            .map(String::as_str)
            .collect::<HashSet<_>>();
        if let Some(missing) = self
            .runtime_names
            .iter()
            .find(|name| !metadata.contains_key(name.as_str()))
        {
            return Err(Error::Other(format!(
                "qwen4-exp checkpoint: missing runtime tensor {missing}"
            )));
        }
        let mut unexpected = metadata
            .keys()
            .filter(|name| !expected.contains(name.as_str()))
            .collect::<Vec<_>>();
        unexpected.sort_unstable();
        if let Some(name) = unexpected.first() {
            return Err(Error::Other(format!(
                "qwen4-exp checkpoint: unexpected runtime tensor {name}"
            )));
        }

        let mut ple_rows = HashMap::<&str, (usize, usize)>::new();
        let mut ple_shards = 0usize;
        for name in &self.runtime_names {
            let actual = &metadata[name];
            if !matches!(actual.dtype, DType::F16 | DType::BF16 | DType::F32) {
                return Err(Error::Other(format!(
                    "qwen4-exp checkpoint {name}: unsupported dtype {}",
                    actual.dtype
                )));
            }
            match &self.runtime_shapes[name] {
                RuntimeShape::Exact(expected) => {
                    if actual.shape != *expected {
                        return Err(Error::Other(format!(
                            "qwen4-exp checkpoint {name}: expected shape {expected:?}, got {:?}",
                            actual.shape
                        )));
                    }
                }
                RuntimeShape::PleShard {
                    group,
                    columns,
                    total_rows,
                } => {
                    ple_shards += 1;
                    if actual.shape.len() != 2
                        || actual.shape[0] == 0
                        || actual.shape[1] != *columns
                    {
                        return Err(Error::Other(format!(
                            "qwen4-exp checkpoint {name}: expected non-empty [rows, {columns}], got {:?}",
                            actual.shape
                        )));
                    }
                    let entry = ple_rows.entry(group).or_insert((0, *total_rows));
                    entry.0 = entry.0.checked_add(actual.shape[0]).ok_or_else(|| {
                        Error::Other(format!("qwen4-exp checkpoint {group}: PLE rows overflow"))
                    })?;
                }
            }
        }
        for (group, (actual, expected)) in ple_rows {
            if actual != expected {
                return Err(Error::Other(format!(
                    "qwen4-exp checkpoint {group}: PLE shards contain {actual} rows, expected {expected}"
                )));
            }
        }
        Ok(Qwen4ExpRuntimeWeightValidation {
            tensors: self.runtime_names.len(),
            ple_shards,
        })
    }
}

pub fn metadata_from_tensors(
    tensors: &HashMap<String, Tensor>,
) -> HashMap<String, Qwen4ExpWeightMetadata> {
    tensors
        .iter()
        .map(|(name, tensor)| {
            (
                name.clone(),
                Qwen4ExpWeightMetadata::new(tensor.shape().dims().to_vec(), tensor.dtype()),
            )
        })
        .collect()
}

fn runtime_shape(config: &Qwen4ExpTextConfig, name: &str) -> Result<RuntimeShape> {
    let hidden = config.hidden_size;
    let hc_hidden = config.hc_count * hidden;
    let exact = |shape: Vec<usize>| Ok(RuntimeShape::Exact(shape));
    match name {
        "model.language_model.embed_tokens.weight" => {
            return exact(vec![config.vocab_size, hidden]);
        }
        "model.language_model.hyper_connection_mixer.hc_norm.weight" => {
            return exact(vec![hc_hidden]);
        }
        "model.language_model.hyper_connection_mixer.input_mix_weight_down.weight" => {
            return exact(vec![config.hc_lowrank, hc_hidden]);
        }
        "model.language_model.hyper_connection_mixer.input_mix_weight_up.weight" => {
            return exact(vec![hc_hidden, config.hc_lowrank]);
        }
        "lm_head.weight" => return exact(vec![config.vocab_size, hidden]),
        _ => {}
    }

    let rest = name
        .strip_prefix("model.language_model.layers.")
        .ok_or_else(|| Error::Other(format!("qwen4-exp shape owner missing for {name}")))?;
    let (layer, suffix) = rest
        .split_once('.')
        .ok_or_else(|| Error::Other(format!("qwen4-exp invalid layer tensor {name}")))?;
    let layer_index = layer
        .parse::<usize>()
        .map_err(|_| Error::Other(format!("qwen4-exp invalid layer index in {name}")))?;
    let layer_type = config
        .layer_types
        .get(layer_index)
        .ok_or_else(|| Error::Other(format!("qwen4-exp layer {layer_index} is outside config")))?;

    for connection in ["attn_hyper_connection", "mlp_hyper_connection"] {
        if let Some(field) = suffix.strip_prefix(&format!("{connection}.")) {
            return match field {
                "block_inject_weight.weight" => exact(vec![config.hc_count, hc_hidden]),
                "hc_norm.weight" => exact(vec![hc_hidden]),
                "input_mix_weight_down.weight" => exact(vec![config.hc_lowrank, hc_hidden]),
                "input_mix_weight_up.weight" => exact(vec![hc_hidden, config.hc_lowrank]),
                _ => Err(Error::Other(format!(
                    "qwen4-exp unknown hyper-connection tensor {name}"
                ))),
            };
        }
    }

    if let Some(field) = suffix.strip_prefix("linear_attn.") {
        if *layer_type != Qwen4ExpLayerType::LinearAttention {
            return Err(Error::Other(format!(
                "qwen4-exp linear tensor appears on QSA layer {layer_index}"
            )));
        }
        let qkv = 2 * config.linear_key_width() + config.linear_value_width();
        return match field {
            "A_log" | "dt_bias" => exact(vec![config.linear_num_value_heads]),
            "conv1d.weight" => exact(vec![qkv, 1, config.linear_conv_kernel_dim]),
            "in_proj_a.weight" | "in_proj_b.weight" => {
                exact(vec![config.linear_num_value_heads, hidden])
            }
            "in_proj_qkv.weight" => exact(vec![qkv, hidden]),
            "in_proj_z.weight" => exact(vec![config.linear_value_width(), hidden]),
            "norm.weight" => exact(vec![config.linear_value_head_dim]),
            "out_proj.weight" => exact(vec![hidden, config.linear_value_width()]),
            _ => Err(Error::Other(format!("qwen4-exp unknown GDN tensor {name}"))),
        };
    }

    if let Some(field) = suffix.strip_prefix("self_attn.") {
        if *layer_type != Qwen4ExpLayerType::QwenSparseAttention {
            return Err(Error::Other(format!(
                "qwen4-exp QSA tensor appears on linear layer {layer_index}"
            )));
        }
        return match field {
            "q_proj.weight" => exact(vec![config.full_q_projection_width(), hidden]),
            "k_proj.weight" | "v_proj.weight" => exact(vec![config.full_kv_width(), hidden]),
            "o_proj.weight" => exact(vec![hidden, config.full_query_width()]),
            "q_norm.weight" | "k_norm.weight" => exact(vec![config.head_dim]),
            "indexer.index_qk_proj.weight" => exact(vec![
                (config.indexer_n_heads + config.indexer_kv_heads) * config.indexer_head_dim,
                hidden,
            ]),
            "indexer.q_layernorm.weight" | "indexer.k_layernorm.weight" => {
                exact(vec![config.indexer_head_dim])
            }
            _ => Err(Error::Other(format!("qwen4-exp unknown QSA tensor {name}"))),
        };
    }

    if let Some(field) = suffix.strip_prefix("mlp.") {
        return match field {
            "experts.down_proj" => exact(vec![
                config.num_experts,
                hidden,
                config.moe_intermediate_size,
            ]),
            "experts.gate_up_proj" => exact(vec![
                config.num_experts,
                2 * config.moe_intermediate_size,
                hidden,
            ]),
            "gate.weight" => exact(vec![config.num_experts, hidden]),
            "shared_expert.down_proj.weight" => {
                exact(vec![hidden, config.shared_expert_intermediate_size])
            }
            "shared_expert.gate_proj.weight" | "shared_expert.up_proj.weight" => {
                exact(vec![config.shared_expert_intermediate_size, hidden])
            }
            "shared_expert_gate.weight" => exact(vec![1, hidden]),
            _ => Err(Error::Other(format!("qwen4-exp unknown MoE tensor {name}"))),
        };
    }

    if let Some(field) = suffix.strip_prefix("ple.") {
        let ordinal = config
            .ple_layer_ids
            .iter()
            .position(|id| *id == layer_index + 1)
            .ok_or_else(|| Error::Other(format!("qwen4-exp PLE tensor on layer {layer_index}")))?;
        let ngram_heads = (config.ngram_size - 1) * config.heads_per_ngram;
        let head_dim = config.ple_embed_dim / ngram_heads;
        return match field {
            "conv1d.weight" => exact(vec![hc_hidden, 1, config.ple_conv_kernel_size]),
            "key_proj.weight" => exact(vec![hc_hidden, config.ple_embed_dim]),
            "value_proj.weight" => exact(vec![hidden, config.ple_embed_dim]),
            "norm_conv.weight" | "norm_key.weight" | "norm_query.weight" => exact(vec![hc_hidden]),
            field
                if field.starts_with("ple_embedding.ngram_embedding.shard_")
                    && field.ends_with(".weight") =>
            {
                let (_, _, total_rows) = ple_vocab_layout(config, ordinal);
                Ok(RuntimeShape::PleShard {
                    group: format!(
                        "model.language_model.layers.{layer_index}.ple.ple_embedding.ngram_embedding"
                    ),
                    columns: head_dim,
                    total_rows,
                })
            }
            _ => Err(Error::Other(format!("qwen4-exp unknown PLE tensor {name}"))),
        };
    }
    Err(Error::Other(format!(
        "qwen4-exp shape owner missing for {name}"
    )))
}

pub(super) fn ple_vocab_layout(
    config: &Qwen4ExpTextConfig,
    ordinal: usize,
) -> (Vec<usize>, Vec<usize>, usize) {
    let ngram_heads = (config.ngram_size - 1) * config.heads_per_ngram;
    let mut sizes = Vec::with_capacity(ngram_heads);
    let mut offsets = Vec::with_capacity(ngram_heads);
    let mut total = 0usize;
    for head in 0..ngram_heads {
        let size = find_nth_prime_after(
            config.ngram_vocab_size_base - 1,
            ordinal * ngram_heads + head + 1,
        );
        sizes.push(size);
        offsets.push(total);
        total += size;
    }
    let divisor = config.make_ngram_vocab_size_divisible_by;
    let padded = total.div_ceil(divisor) * divisor;
    (sizes, offsets, padded)
}

fn find_nth_prime_after(start: usize, count: usize) -> usize {
    let mut prime = start;
    for _ in 0..count {
        prime += 1;
        while !is_prime(prime) {
            prime += 1;
        }
    }
    prime
}

fn is_prime(value: usize) -> bool {
    if value < 2 {
        return false;
    }
    if value.is_multiple_of(2) {
        return value == 2;
    }
    let mut divisor = 3;
    while divisor * divisor <= value {
        if value.is_multiple_of(divisor) {
            return false;
        }
        divisor += 2;
    }
    true
}

#[cfg(test)]
pub(crate) mod tests {
    use std::collections::{BTreeMap, HashMap};

    use super::*;
    use crate::qwen4_exp::config::tests::MINI_CONFIG;

    fn schema() -> Qwen4ExpWeightSchema {
        let config = Qwen4ExpConfig::from_json_str(MINI_CONFIG).unwrap();
        Qwen4ExpWeightSchema::new(&config).unwrap()
    }

    fn runtime_metadata(schema: &Qwen4ExpWeightSchema) -> HashMap<String, Qwen4ExpWeightMetadata> {
        schema
            .runtime_names()
            .iter()
            .map(|name| {
                let shape = match &schema.runtime_shapes[name] {
                    RuntimeShape::Exact(shape) => shape.clone(),
                    RuntimeShape::PleShard {
                        columns,
                        total_rows,
                        ..
                    } => {
                        let shard = name
                            .rsplit_once("shard_")
                            .unwrap()
                            .1
                            .strip_suffix(".weight")
                            .unwrap()
                            .parse::<usize>()
                            .unwrap();
                        let parts = 2usize;
                        let rows = total_rows / parts + usize::from(shard < total_rows % parts);
                        vec![rows, *columns]
                    }
                };
                (
                    name.clone(),
                    Qwen4ExpWeightMetadata::new(shape, DType::BF16),
                )
            })
            .collect()
    }

    pub(crate) fn zero_runtime_tensors() -> (Qwen4ExpConfig, HashMap<String, Tensor>) {
        let config = Qwen4ExpConfig::from_json_str(MINI_CONFIG).unwrap();
        let schema = Qwen4ExpWeightSchema::new(&config).unwrap();
        let metadata = runtime_metadata(&schema);
        let tensors = metadata
            .into_iter()
            .map(|(name, metadata)| {
                let elements = metadata.shape.iter().product();
                let tensor = Tensor::from_f32(metadata.shape, &vec![0.0; elements]).unwrap();
                (name, tensor)
            })
            .collect();
        (config, tensors)
    }

    pub(crate) fn zero_bf16_runtime_tensors() -> (Qwen4ExpConfig, HashMap<String, Tensor>) {
        let config = Qwen4ExpConfig::from_json_str(MINI_CONFIG).unwrap();
        let schema = Qwen4ExpWeightSchema::new(&config).unwrap();
        let metadata = runtime_metadata(&schema);
        let tensors = metadata
            .into_iter()
            .map(|(name, metadata)| {
                let elements = metadata.shape.iter().product();
                let tensor =
                    Tensor::from_bf16(metadata.shape, &vec![half::bf16::ZERO; elements]).unwrap();
                (name, tensor)
            })
            .collect();
        (config, tensors)
    }

    fn index_json(schema: &Qwen4ExpWeightSchema) -> String {
        let mut weights = schema
            .expected_names()
            .iter()
            .cloned()
            .map(|name| (name, "model-00001.safetensors".to_owned()))
            .collect::<BTreeMap<_, _>>();
        weights.insert(
            "model.visual.patch_embed.proj.weight".into(),
            "model-00002.safetensors".into(),
        );
        weights.insert(
            "mtp.fc_hidden.weight".into(),
            "model-00002.safetensors".into(),
        );
        serde_json::json!({"weight_map": weights}).to_string()
    }

    #[test]
    fn mini_schema_matches_released_layer_and_ple_name_counts() {
        let schema = schema();
        assert_eq!(schema.expected_names().len(), 112);
        assert_eq!(schema.runtime_names().len(), 109);
        assert!(schema.expected_names().contains(
            &"model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight"
                .into()
        ));
        assert!(schema.expected_names().contains(
            &"model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_1.weight"
                .into()
        ));
        assert!(!schema
            .expected_names()
            .iter()
            .any(|name| name.contains("shard_2.weight")));
        assert!(schema.runtime_f32_bytes().unwrap() > 0);
        assert!(
            schema.runtime_hybrid_weight_bytes().unwrap() < schema.runtime_f32_bytes().unwrap()
        );
    }

    #[test]
    fn accepts_exact_text_names_and_explicitly_ignored_towers() {
        let schema = schema();
        let validation = schema
            .validate_index_json_str(&index_json(&schema))
            .unwrap();
        assert_eq!(validation.text_tensors, 112);
        assert_eq!(validation.runtime_tensors, 109);
        assert_eq!(validation.ignored_vision_tensors, 1);
        assert_eq!(validation.ignored_mtp_tensors, 1);
    }

    #[test]
    fn rejects_missing_text_tensor() {
        let schema = schema();
        let mut value: serde_json::Value = serde_json::from_str(&index_json(&schema)).unwrap();
        value["weight_map"]
            .as_object_mut()
            .unwrap()
            .remove("model.language_model.layers.0.linear_attn.A_log");
        let error = schema
            .validate_index_json_str(&value.to_string())
            .unwrap_err();
        assert!(error.to_string().contains("missing"));
        assert!(error.to_string().contains("linear_attn.A_log"));
    }

    #[test]
    fn rejects_unknown_text_tensor() {
        let schema = schema();
        let mut value: serde_json::Value = serde_json::from_str(&index_json(&schema)).unwrap();
        value["weight_map"].as_object_mut().unwrap().insert(
            "model.language_model.layers.0.parallel_fallback.weight".into(),
            "model-00001.safetensors".into(),
        );
        let error = schema
            .validate_index_json_str(&value.to_string())
            .unwrap_err();
        assert!(error.to_string().contains("unexpected"));
        assert!(error.to_string().contains("parallel_fallback"));
    }

    #[test]
    fn validates_every_runtime_shape_and_ple_shard_total() {
        let schema = schema();
        let validation = schema
            .validate_runtime_metadata(&runtime_metadata(&schema))
            .unwrap();
        assert_eq!(validation.tensors, 109);
        assert_eq!(validation.ple_shards, 2);
    }

    #[test]
    fn runtime_shape_error_names_the_checkpoint_tensor() {
        let schema = schema();
        let mut metadata = runtime_metadata(&schema);
        let name = "model.language_model.layers.3.self_attn.q_proj.weight";
        metadata.get_mut(name).unwrap().shape[0] += 1;
        let error = schema.validate_runtime_metadata(&metadata).unwrap_err();
        assert!(error.to_string().contains(name));
        assert!(error.to_string().contains("expected shape"));
    }

    #[test]
    fn rejects_ple_shards_with_wrong_total_rows() {
        let schema = schema();
        let mut metadata = runtime_metadata(&schema);
        let name = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_1.weight";
        metadata.get_mut(name).unwrap().shape[0] -= 1;
        let error = schema.validate_runtime_metadata(&metadata).unwrap_err();
        assert!(error.to_string().contains("PLE shards contain"));
    }
}
