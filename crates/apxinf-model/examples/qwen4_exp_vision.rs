//! Run the Qwen4-Exp vision encoder against a published-layout checkpoint.

use std::path::PathBuf;

use apxinf_core::Tensor;
use apxinf_model::{encode_qwen4_exp_vision, Qwen4ExpConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args_os().skip(1);
    let usage = "usage: qwen4_exp_vision CONFIG CHECKPOINT";
    let config_path = PathBuf::from(args.next().ok_or(usage)?);
    let checkpoint_path = PathBuf::from(args.next().ok_or(usage)?);
    if args.next().is_some() {
        return Err(usage.into());
    }
    let config = Qwen4ExpConfig::from_json_file(&config_path)?;
    let (tensors, _) =
        apxinf_loader::safetensors::load_native_path_mmap_filtered(&checkpoint_path, |name| {
            name.starts_with("model.visual.")
        })?;
    let grid = [[1u32, 2, 2]];
    let patches = 4usize;
    let patch_width = config.vision.in_channels
        * config.vision.temporal_patch_size
        * config.vision.patch_size
        * config.vision.patch_size;
    let pixels = (0..patches * patch_width)
        .map(|index| index as f32 / 100.0)
        .collect::<Vec<_>>();
    let pixels = Tensor::from_f32(vec![patches, patch_width], &pixels)?;
    let output = encode_qwen4_exp_vision(&config, tensors, &pixels, &grid)?;
    let values = output.to_f32_vec()?;
    println!(
        "{}",
        serde_json::json!({
            "format": "apxinf-qwen4-exp-vision-run-v1",
            "grid_thw": grid,
            "output_shape": output.shape().dims(),
            "output": values,
        })
    );
    Ok(())
}
