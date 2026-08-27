//! Qwen3.8-Flash-Next text checkpoint-name contract.

use std::collections::HashSet;

use apxinf_core::{Error, Result};

use super::config::Qwen4ExpConfig;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Qwen4ExpWeightIndexValidation {
    pub text_tensors: usize,
    pub runtime_tensors: usize,
    pub ignored_vision_tensors: usize,
    pub ignored_mtp_tensors: usize,
}

#[derive(Clone, Debug)]
pub struct Qwen4ExpWeightSchema {
    expected_names: Vec<String>,
    runtime_names: Vec<String>,
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
            .collect();
        Ok(Self {
            expected_names,
            runtime_names,
        })
    }

    pub fn expected_names(&self) -> &[String] {
        &self.expected_names
    }

    pub fn runtime_names(&self) -> &[String] {
        &self.runtime_names
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
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;
    use crate::qwen4_exp::config::tests::MINI_CONFIG;

    fn schema() -> Qwen4ExpWeightSchema {
        let config = Qwen4ExpConfig::from_json_str(MINI_CONFIG).unwrap();
        Qwen4ExpWeightSchema::new(&config).unwrap()
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
}
