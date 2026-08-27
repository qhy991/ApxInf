//! Run the Qwen4-Exp vision encoder against a published-layout checkpoint.

use std::path::PathBuf;

use apxinf_core::Tensor;
use apxinf_model::{encode_qwen4_exp_vision, Qwen4ExpConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args_os().skip(1);
    let usage = "usage: qwen4_exp_vision CONFIG CHECKPOINT [T,H,W]";
    let config_path = PathBuf::from(args.next().ok_or(usage)?);
    let checkpoint_path = PathBuf::from(args.next().ok_or(usage)?);
    let grid = match args.next() {
        Some(value) => {
            let values = value
                .into_string()
                .map_err(|_| "grid must be UTF-8")?
                .split(',')
                .map(str::parse::<u32>)
                .collect::<Result<Vec<_>, _>>()?;
            if values.len() != 3 {
                return Err(usage.into());
            }
            [[values[0], values[1], values[2]]]
        }
        None => [[1, 2, 2]],
    };
    if args.next().is_some() {
        return Err(usage.into());
    }
    let config = Qwen4ExpConfig::from_json_file(&config_path)?;
    let (tensors, _) =
        apxinf_loader::safetensors::load_native_path_mmap_filtered(&checkpoint_path, |name| {
            name.starts_with("model.visual.")
        })?;
    let patches = grid[0]
        .iter()
        .map(|value| *value as usize)
        .product::<usize>();
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
