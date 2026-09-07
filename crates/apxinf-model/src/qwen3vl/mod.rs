//! Qwen3-VL-2B-Instruct model implementation.
//!
//! Text stack + vision tower today; multimodal wiring in Phase 5. All CUDA
//! and bf16-first.

pub mod config;
pub mod weights;
pub mod general;
#[cfg(feature = "cuda")]
pub mod decode_graph;

pub use config::Qwen3VLConfig;
pub use weights::Qwen3VLTextWeights;
pub use crate::qwen_vl_vision as vision;
pub use crate::qwen_vl_vision::{VisionOutput, VisionWeights as Qwen3VLVisionWeights};
pub use general::GeneralQwen3VL;
#[cfg(feature = "cuda")]
pub use decode_graph::{Qwen3VLDecodeGraph, Qwen3VLDecodeGraphConfig,
                       Qwen3VLDecodeGraphWeights, Qwen3VLDecodeLayerWeights};
