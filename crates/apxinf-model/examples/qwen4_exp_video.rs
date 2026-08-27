//! Run a fixed tiny Qwen4-Exp video-text prefill and one decode step.

use std::path::PathBuf;

use apxinf_core::{Device, Tensor};
use apxinf_model::{AutoModel, LlmInput, LoadOptions, VideoInput};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args_os().skip(1);
    let usage = "usage: qwen4_exp_video MODEL_DIR";
    let model_dir = PathBuf::from(args.next().ok_or(usage)?);
    if args.next().is_some() {
        return Err(usage.into());
    }
    let tokens = [1u32, 29, 29, 29, 29, 2, 29, 29, 29, 29, 3];
    let grid = [[2u32, 4, 4]];
    let patches = 32usize;
    let patch_width = 3 * 2 * 2 * 2;
    let pixels = (0..patches * patch_width)
        .map(|index| index as f32 / 100.0)
        .collect::<Vec<_>>();
    let pixels = Tensor::from_f32(vec![patches, patch_width], &pixels)?;
    let options = LoadOptions {
        max_context: Some(64),
        ..LoadOptions::default()
    };
    let mut model = AutoModel::load_model(Device::Cpu, &model_dir, &options)?;
    let prefill = model.text_mut()?.prefill(LlmInput::with_video(
        &tokens,
        VideoInput::new(&pixels, &grid),
    ))?;
    let decode_token = 4u32;
    let decode = model.forward(&[decode_token], tokens.len() as u32)?;
    let mut logits = prefill.to_f32_vec()?;
    logits.extend(decode.to_f32_vec()?);
    println!(
        "{}",
        serde_json::json!({
            "format": "apxinf-qwen4-exp-video-run-v1",
            "tokens": tokens,
            "decode_token": decode_token,
            "grid_thw": grid,
            "logits_shape": [tokens.len() + 1, prefill.shape().dims()[1]],
            "logits": logits,
            "generation_path": model.generation_path_receipt()?,
        })
    );
    Ok(())
}
