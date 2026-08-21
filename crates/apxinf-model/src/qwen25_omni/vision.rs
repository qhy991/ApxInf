//! Native Qwen2.5-Omni vision encoder.

use std::collections::HashMap;

use apxinf_core::{Backend, Error, Result, Tensor};

use super::config::Qwen25OmniConfig;
use super::weights::{flatten_and_transpose, transpose_2d};

pub struct Qwen25OmniVisionWeights {
    patch_embed: Tensor,
    blocks: Vec<VisionBlock>,
    merger_norm: Tensor,
    merger_fc1: Tensor,
    merger_fc1_bias: Tensor,
    merger_fc2: Tensor,
    merger_fc2_bias: Tensor,
}

struct VisionBlock {
    norm1: Tensor,
    wq: Tensor,
    bq: Tensor,
    wk: Tensor,
    bk: Tensor,
    wv: Tensor,
    bv: Tensor,
    wo: Tensor,
    bo: Tensor,
    norm2: Tensor,
    w_gate: Tensor,
    b_gate: Tensor,
    w_up: Tensor,
    b_up: Tensor,
    w_down: Tensor,
    b_down: Tensor,
}

impl Qwen25OmniVisionWeights {
    pub fn from_map(
        config: &Qwen25OmniConfig,
        tensors: &mut HashMap<String, Tensor>,
    ) -> Result<Self> {
        let take = |name: &str, map: &mut HashMap<String, Tensor>| {
            map.remove(name)
                .ok_or_else(|| Error::Other(format!("missing {name}")))
        };
        let vision = &config.vision;
        let patch_width =
            vision.in_channels * vision.temporal_patch_size * vision.patch_size * vision.patch_size;
        let patch = take("thinker.visual.patch_embed.proj.weight", tensors)?;
        let patch_embed = flatten_and_transpose(&patch, vision.hidden_size, patch_width)?;
        let mut blocks = Vec::with_capacity(vision.depth);
        for index in 0..vision.depth {
            let prefix = format!("thinker.visual.blocks.{index}");
            blocks.push(VisionBlock {
                norm1: take(&format!("{prefix}.norm1.weight"), tensors)?,
                wq: transpose_2d(&take(&format!("{prefix}.attn.q.weight"), tensors)?)?,
                bq: take(&format!("{prefix}.attn.q.bias"), tensors)?,
                wk: transpose_2d(&take(&format!("{prefix}.attn.k.weight"), tensors)?)?,
                bk: take(&format!("{prefix}.attn.k.bias"), tensors)?,
                wv: transpose_2d(&take(&format!("{prefix}.attn.v.weight"), tensors)?)?,
                bv: take(&format!("{prefix}.attn.v.bias"), tensors)?,
                wo: transpose_2d(&take(&format!("{prefix}.attn.proj.weight"), tensors)?)?,
                bo: take(&format!("{prefix}.attn.proj.bias"), tensors)?,
                norm2: take(&format!("{prefix}.norm2.weight"), tensors)?,
                w_gate: transpose_2d(&take(&format!("{prefix}.mlp.gate_proj.weight"), tensors)?)?,
                b_gate: take(&format!("{prefix}.mlp.gate_proj.bias"), tensors)?,
                w_up: transpose_2d(&take(&format!("{prefix}.mlp.up_proj.weight"), tensors)?)?,
                b_up: take(&format!("{prefix}.mlp.up_proj.bias"), tensors)?,
                w_down: transpose_2d(&take(&format!("{prefix}.mlp.down_proj.weight"), tensors)?)?,
                b_down: take(&format!("{prefix}.mlp.down_proj.bias"), tensors)?,
            });
        }
        Ok(Self {
            patch_embed,
            blocks,
            merger_norm: take("thinker.visual.merger.ln_q.weight", tensors)?,
            merger_fc1: transpose_2d(&take("thinker.visual.merger.mlp.0.weight", tensors)?)?,
            merger_fc1_bias: take("thinker.visual.merger.mlp.0.bias", tensors)?,
            merger_fc2: transpose_2d(&take("thinker.visual.merger.mlp.2.weight", tensors)?)?,
            merger_fc2_bias: take("thinker.visual.merger.mlp.2.bias", tensors)?,
        })
    }

    pub fn to_device(self, backend: &dyn Backend) -> Result<Self> {
        let blocks = self
            .blocks
            .into_iter()
            .map(|block| {
                Ok(VisionBlock {
                    norm1: backend.to_device(&block.norm1)?,
                    wq: backend.to_device(&block.wq)?,
                    bq: backend.to_device(&block.bq)?,
                    wk: backend.to_device(&block.wk)?,
                    bk: backend.to_device(&block.bk)?,
                    wv: backend.to_device(&block.wv)?,
                    bv: backend.to_device(&block.bv)?,
                    wo: backend.to_device(&block.wo)?,
                    bo: backend.to_device(&block.bo)?,
                    norm2: backend.to_device(&block.norm2)?,
                    w_gate: backend.to_device(&block.w_gate)?,
                    b_gate: backend.to_device(&block.b_gate)?,
                    w_up: backend.to_device(&block.w_up)?,
                    b_up: backend.to_device(&block.b_up)?,
                    w_down: backend.to_device(&block.w_down)?,
                    b_down: backend.to_device(&block.b_down)?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            patch_embed: backend.to_device(&self.patch_embed)?,
            blocks,
            merger_norm: backend.to_device(&self.merger_norm)?,
            merger_fc1: backend.to_device(&self.merger_fc1)?,
            merger_fc1_bias: backend.to_device(&self.merger_fc1_bias)?,
            merger_fc2: backend.to_device(&self.merger_fc2)?,
            merger_fc2_bias: backend.to_device(&self.merger_fc2_bias)?,
        })
    }
}

pub fn forward(
    config: &Qwen25OmniConfig,
    weights: &Qwen25OmniVisionWeights,
    backend: &dyn Backend,
    pixel_values: &Tensor,
    grid_thw: &[[u32; 3]],
) -> Result<Tensor> {
    let vision = &config.vision;
    let raw_tokens = validate_input(config, pixel_values, grid_thw)?;
    let uploaded = if pixel_values.device() != backend.device() {
        Some(backend.to_device(pixel_values)?)
    } else {
        None
    };
    let pixels = uploaded.as_ref().unwrap_or(pixel_values);
    let mut hidden = backend.matmul(pixels, &weights.patch_embed)?;
    let positions = vision_positions(grid_thw, vision.spatial_merge_size);
    let groups = vision_window_groups(config, grid_thw)?;
    for (index, block) in weights.blocks.iter().enumerate() {
        let normalized = backend.rms_norm(&hidden, &block.norm1, 1e-6)?;
        let q = backend.add_bias(&backend.matmul(&normalized, &block.wq)?, &block.bq)?;
        let k = backend.add_bias(&backend.matmul(&normalized, &block.wk)?, &block.bk)?;
        let v = backend.add_bias(&backend.matmul(&normalized, &block.wv)?, &block.bv)?;
        let q = q.reshape(vec![raw_tokens, vision.n_heads, vision.head_dim])?;
        let k = k.reshape(vec![raw_tokens, vision.n_heads, vision.head_dim])?;
        let v = v.reshape(vec![raw_tokens, vision.n_heads, vision.head_dim])?;
        let q =
            backend.rope_vision_2d(&q, vision.n_heads, vision.head_dim, 10_000.0, &positions)?;
        let k =
            backend.rope_vision_2d(&k, vision.n_heads, vision.head_dim, 10_000.0, &positions)?;
        let attention = if vision.full_attention_blocks.contains(&index) {
            backend.vision_sdpa(&q, &k, &v, raw_tokens, vision.n_heads, vision.head_dim)?
        } else {
            backend.grouped_sdpa(
                &q,
                &k,
                &v,
                raw_tokens,
                vision.n_heads,
                vision.head_dim,
                &groups,
            )?
        };
        let attention = backend.add_bias(&backend.matmul(&attention, &block.wo)?, &block.bo)?;
        hidden = backend.add(&hidden, &attention)?;
        let normalized = backend.rms_norm(&hidden, &block.norm2, 1e-6)?;
        let gate = backend.add_bias(&backend.matmul(&normalized, &block.w_gate)?, &block.b_gate)?;
        let gate = backend.silu(&gate)?;
        let up = backend.add_bias(&backend.matmul(&normalized, &block.w_up)?, &block.b_up)?;
        let mlp = backend.mul(&gate, &up)?;
        let mlp = backend.add_bias(&backend.matmul(&mlp, &block.w_down)?, &block.b_down)?;
        hidden = backend.add(&hidden, &mlp)?;
    }

    let normalized = backend.rms_norm(&hidden, &weights.merger_norm, 1e-6)?;
    let merged_tokens = merged_token_count(grid_thw, vision.spatial_merge_size)?;
    let merged_width = vision.hidden_size * vision.spatial_merge_size.pow(2);
    let merged = normalized.reshape(vec![merged_tokens, merged_width])?;
    let merged = backend.add_bias(
        &backend.matmul(&merged, &weights.merger_fc1)?,
        &weights.merger_fc1_bias,
    )?;
    let merged = backend.gelu_tanh(&merged)?;
    backend.add_bias(
        &backend.matmul(&merged, &weights.merger_fc2)?,
        &weights.merger_fc2_bias,
    )
}

/// Validate processor-owned image views without executing a backend operator.
pub fn validate_input(
    config: &Qwen25OmniConfig,
    pixel_values: &Tensor,
    grid_thw: &[[u32; 3]],
) -> Result<usize> {
    if grid_thw.len() != 1 {
        return Err(Error::Other(
            "qwen2.5-omni first deployment slice accepts exactly one image grid".into(),
        ));
    }
    let vision = &config.vision;
    let patch_width =
        vision.in_channels * vision.temporal_patch_size * vision.patch_size * vision.patch_size;
    let raw_tokens = validate_grids(grid_thw, vision.spatial_merge_size)?;
    if pixel_values.shape().dims() != [raw_tokens, patch_width] {
        return Err(Error::Other(format!(
            "qwen2.5-omni pixel_values shape {:?}, expected [{raw_tokens}, {patch_width}]",
            pixel_values.shape().dims()
        )));
    }
    Ok(raw_tokens)
}

pub fn merged_token_count(grid_thw: &[[u32; 3]], merge: usize) -> Result<usize> {
    let merge = u32::try_from(merge)
        .map_err(|_| Error::Other("qwen2.5-omni merge size exceeds u32".into()))?;
    grid_thw
        .iter()
        .try_fold(0usize, |total, &[time, height, width]| {
            let count = (time as usize)
                .checked_mul((height / merge) as usize)
                .and_then(|value| value.checked_mul((width / merge) as usize))
                .ok_or_else(|| Error::Other("qwen2.5-omni merged image token overflow".into()))?;
            total
                .checked_add(count)
                .ok_or_else(|| Error::Other("qwen2.5-omni merged image total overflow".into()))
        })
}

fn validate_grids(grid_thw: &[[u32; 3]], merge: usize) -> Result<usize> {
    let merge = u32::try_from(merge)
        .map_err(|_| Error::Other("qwen2.5-omni merge size exceeds u32".into()))?;
    if merge == 0 {
        return Err(Error::Other("qwen2.5-omni merge size is zero".into()));
    }
    grid_thw
        .iter()
        .try_fold(0usize, |total, &[time, height, width]| {
            if time == 0 || height == 0 || width == 0 || height % merge != 0 || width % merge != 0 {
                return Err(Error::Other(format!(
                    "qwen2.5-omni invalid image grid [{time}, {height}, {width}] for merge {merge}"
                )));
            }
            let count = (time as usize)
                .checked_mul(height as usize)
                .and_then(|value| value.checked_mul(width as usize))
                .ok_or_else(|| Error::Other("qwen2.5-omni image token overflow".into()))?;
            total
                .checked_add(count)
                .ok_or_else(|| Error::Other("qwen2.5-omni image token total overflow".into()))
        })
}

fn vision_positions(grid_thw: &[[u32; 3]], merge: usize) -> Vec<u32> {
    let mut positions = Vec::new();
    for &[time, height, width] in grid_thw {
        for _ in 0..time {
            for merged_row in 0..height as usize / merge {
                for merged_col in 0..width as usize / merge {
                    for inner_row in 0..merge {
                        for inner_col in 0..merge {
                            positions.extend_from_slice(&[
                                (merged_row * merge + inner_row) as u32,
                                (merged_col * merge + inner_col) as u32,
                            ]);
                        }
                    }
                }
            }
        }
    }
    positions
}

fn vision_window_groups(config: &Qwen25OmniConfig, grid_thw: &[[u32; 3]]) -> Result<Vec<u32>> {
    let patches_per_window = config.vision.window_size / config.vision.patch_size;
    if patches_per_window == 0 {
        return Err(Error::Other(
            "qwen2.5-omni vision window is smaller than a patch".into(),
        ));
    }
    let mut groups = Vec::new();
    let mut group_base = 0_u32;
    for &[time, height, width] in grid_thw {
        let window_columns = (width as usize).div_ceil(patches_per_window);
        let window_rows = (height as usize).div_ceil(patches_per_window);
        let merge = config.vision.spatial_merge_size;
        for temporal in 0..time as usize {
            for merged_row in 0..height as usize / merge {
                for merged_col in 0..width as usize / merge {
                    for inner_row in 0..merge {
                        for inner_col in 0..merge {
                            let row = merged_row * merge + inner_row;
                            let col = merged_col * merge + inner_col;
                            let group = temporal * window_rows * window_columns
                                + (row / patches_per_window) * window_columns
                                + col / patches_per_window;
                            groups.push(group_base + group as u32);
                        }
                    }
                }
            }
        }
        group_base = group_base
            .checked_add((time as usize * window_rows * window_columns) as u32)
            .ok_or_else(|| Error::Other("qwen2.5-omni vision group id overflow".into()))?;
    }
    Ok(groups)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validates_merged_grid_contract() {
        assert_eq!(merged_token_count(&[[1, 4, 6]], 2).unwrap(), 6);
        assert!(validate_grids(&[[1, 3, 4]], 2).is_err());
    }

    #[test]
    fn validates_one_processor_image_before_backend_work() {
        let raw = include_str!("../../tests/data/qwen25_omni_config_minimal.json");
        let config = Qwen25OmniConfig::from_json_str(raw).unwrap();
        let pixels = Tensor::from_f32(vec![16, 1176], &vec![0.0; 16 * 1176]).unwrap();
        assert_eq!(validate_input(&config, &pixels, &[[1, 4, 4]]).unwrap(), 16);
        assert!(validate_input(&config, &pixels, &[]).is_err());
        assert!(validate_input(&config, &pixels, &[[1, 4, 4], [1, 4, 4]]).is_err());
        let wrong = Tensor::from_f32(vec![15, 1176], &vec![0.0; 15 * 1176]).unwrap();
        assert!(validate_input(&config, &wrong, &[[1, 4, 4]]).is_err());
    }
}
