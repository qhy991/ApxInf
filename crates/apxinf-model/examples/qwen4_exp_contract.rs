//! Check a Qwen3.8-Flash-Next config and SafeTensors index without weights.

use std::path::PathBuf;

use apxinf_model::{Qwen4ExpConfig, Qwen4ExpWeightSchema};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args_os().skip(1);
    let config_path = PathBuf::from(
        args.next()
            .ok_or("usage: qwen4_exp_contract CONFIG INDEX")?,
    );
    let index_path = PathBuf::from(
        args.next()
            .ok_or("usage: qwen4_exp_contract CONFIG INDEX")?,
    );
    if args.next().is_some() {
        return Err("usage: qwen4_exp_contract CONFIG INDEX".into());
    }

    let config = Qwen4ExpConfig::from_json_file(&config_path)?;
    let schema = Qwen4ExpWeightSchema::new(&config)?;
    let index = std::fs::read_to_string(&index_path)?;
    let validation = schema.validate_index_json_str(&index)?;
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
                "ignored_vision_tensors": validation.ignored_vision_tensors,
                "ignored_mtp_tensors": validation.ignored_mtp_tensors,
            },
            "weights_loaded": false,
        })
    );
    Ok(())
}
