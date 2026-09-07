//! Shared Qwen vision-transformer runtime used by Qwen3-VL and Qwen4-Exp.

mod forward;
mod weights;

pub use forward::{forward, forward_debug, VisionOutput};
pub use weights::{transfer_vision_weights, VisionBlock, VisionMerger, VisionWeights};

#[derive(Clone, Debug)]
pub struct VisionConfig {
    pub depth: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub patch_size: usize,
    pub temporal_patch_size: usize,
    pub in_channels: usize,
    pub spatial_merge_size: usize,
    pub num_position_embeddings: usize,
    pub out_hidden_size: usize,
    pub deepstack_visual_indexes: Vec<usize>,
}

impl VisionConfig {
    pub fn head_dim(&self) -> usize {
        if self.head_dim != 0 {
            self.head_dim
        } else {
            self.hidden_size / self.num_heads
        }
    }
}
