//! Check a Qwen3.8-Flash-Next config and SafeTensors index without weights.

use std::collections::HashMap;
use std::path::PathBuf;

use apxinf_core::DType;
use apxinf_model::{Qwen4ExpConfig, Qwen4ExpWeightMetadata, Qwen4ExpWeightSchema};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args_os().skip(1);
    let usage = "usage: qwen4_exp_contract CONFIG INDEX [HEADER_MANIFEST]";
    let config_path = PathBuf::from(args.next().ok_or(usage)?);
    let index_path = PathBuf::from(args.next().ok_or(usage)?);
    let header_manifest = args.next().map(PathBuf::from);
    if args.next().is_some() {
        return Err(usage.into());
    }

    let config = Qwen4ExpConfig::from_json_file(&config_path)?;
    let schema = Qwen4ExpWeightSchema::new(&config)?;
    let index = std::fs::read_to_string(&index_path)?;
    let validation = schema.validate_index_json_str(&index)?;
    let runtime_f32_bytes = schema.runtime_f32_bytes()?;
    let runtime_hybrid_resident_bytes = schema.runtime_hybrid_resident_bytes()?;
    let runtime_validation = header_manifest
        .map(|path| validate_header_manifest(&schema, &path))
        .transpose()?;
    let linear_layers = config
        .text
        .layer_types
        .iter()
        .filter(|layer| matches!(layer, apxinf_model::Qwen4ExpLayerType::LinearAttention))
        .count();
    let qsa_layers = config.text.n_layers - linear_layers;

    println!(
        "{}",
        serde_json::json!({
            "format": "apxinf-qwen4-exp-contract-v1",
            "model_type": "qwen4_exp",
            "text": {
                "hidden_size": config.text.hidden_size,
                "layers": config.text.n_layers,
                "linear_attention_layers": linear_layers,
                "qsa_layers": qsa_layers,
                "residual_streams": config.text.hc_count,
                "experts": config.text.num_experts,
                "experts_per_token": config.text.num_experts_per_tok,
                "ple_layers": config.text.ple_layer_ids,
                "ngram_embedding_shards_per_ple_layer": config.text.split_ngram_parts,
            },
            "checkpoint": {
                "text_tensors": validation.text_tensors,
                "runtime_tensors": validation.runtime_tensors,
                "runtime_f32_bytes": runtime_f32_bytes,
                "runtime_hybrid_resident_bytes": runtime_hybrid_resident_bytes,
                "ignored_vision_tensors": validation.ignored_vision_tensors,
                "ignored_mtp_tensors": validation.ignored_mtp_tensors,
                "runtime_metadata": runtime_validation.map(|validation| serde_json::json!({
                    "tensors": validation.tensors,
                    "ple_shards": validation.ple_shards,
                })),
            },
            "weights_loaded": false,
        })
    );
    Ok(())
}

fn validate_header_manifest(
    schema: &Qwen4ExpWeightSchema,
    path: &std::path::Path,
) -> Result<apxinf_model::Qwen4ExpRuntimeWeightValidation, Box<dyn std::error::Error>> {
    let raw = std::fs::read_to_string(path)?;
    let root: serde_json::Value = serde_json::from_str(&raw)?;
    let entries = root
        .get("metadata")
        .and_then(serde_json::Value::as_object)
        .ok_or("header manifest is missing metadata object")?;
    let mut metadata = HashMap::with_capacity(entries.len());
    for (name, entry) in entries {
        let shape = entry
            .get("shape")
            .and_then(serde_json::Value::as_array)
            .ok_or("header manifest tensor is missing shape")?
            .iter()
            .map(|value| {
                value
                    .as_u64()
                    .and_then(|value| usize::try_from(value).ok())
                    .ok_or("header manifest shape must contain usize integers")
            })
            .collect::<Result<Vec<_>, _>>()?;
        let dtype = match entry.get("dtype").and_then(serde_json::Value::as_str) {
            Some("F32") => DType::F32,
            Some("F16") => DType::F16,
            Some("BF16") => DType::BF16,
            Some(other) => return Err(format!("unsupported manifest dtype {other}").into()),
            None => return Err("header manifest tensor is missing dtype".into()),
        };
        metadata.insert(name.clone(), Qwen4ExpWeightMetadata::new(shape, dtype));
    }
    Ok(schema.validate_runtime_metadata(&metadata)?)
}
