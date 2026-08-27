//! Qwen3.8-Flash-Next (`qwen4_exp`) architecture.

pub mod config;
pub mod general;
pub mod qsa;
pub mod weights;

pub use config::{
    Qwen4ExpConfig, Qwen4ExpLayerType, Qwen4ExpRopeConfig, Qwen4ExpTextConfig, Qwen4ExpVisionConfig,
};
pub use general::GeneralQwen4Exp;
pub use qsa::Qwen4ExpQsaSelector;
pub use weights::{Qwen4ExpWeightIndexValidation, Qwen4ExpWeightSchema};
