//! Qwen3.8-Flash-Next (`qwen4_exp`) architecture.

pub mod config;
pub mod general;
pub mod qsa;
pub mod weights;

pub use config::{
    Qwen4ExpConfig, Qwen4ExpLayerType, Qwen4ExpRopeConfig, Qwen4ExpTextConfig, Qwen4ExpVisionConfig,
};
pub use general::{encode_qwen4_exp_vision, GeneralQwen4Exp};
pub use qsa::Qwen4ExpQsaSelector;
pub use weights::{
    metadata_from_tensors, Qwen4ExpRuntimeWeightValidation, Qwen4ExpWeightIndexValidation,
    Qwen4ExpWeightMetadata, Qwen4ExpWeightSchema,
};
