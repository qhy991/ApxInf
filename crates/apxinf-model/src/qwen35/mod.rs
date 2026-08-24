//! Qwen3.5 hybrid text/VLM architecture.
//!
//! This module currently owns the strict Hugging Face config and text-weight
//! contracts. Execution is added separately because Qwen3.5 requires a
//! recurrent Gated DeltaNet state in addition to conventional KV caches.

pub mod config;
pub mod general;
pub mod state;
pub mod weights;

pub use config::{
    Qwen35Config, Qwen35LayerType, Qwen35RopeConfig, Qwen35TextConfig, Qwen35VisionConfig,
};
pub use general::GeneralQwen35;
pub use state::{Qwen35HybridState, Qwen35LinearState};
pub use weights::{
    metadata_from_tensors, Qwen35AttentionWeights, Qwen35FullAttentionWeights, Qwen35LayerWeights,
    Qwen35LinearAttentionWeights, Qwen35MlpWeights, Qwen35TextWeights, Qwen35WeightMetadata,
    Qwen35WeightSchema, Qwen35WeightSpec, Qwen35WeightValidation,
};
