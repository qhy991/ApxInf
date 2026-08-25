//! Native Qwen2.5-Omni Thinker (text/image/audio to text).

pub mod audio;
pub mod checkpoint;
pub mod config;
#[cfg(feature = "cuda")]
pub mod decode_graph;
pub mod general;
pub mod vision;
pub mod weights;

pub use checkpoint::Qwen25OmniCheckpointReport;
pub use config::{
    Qwen25OmniAudioConfig, Qwen25OmniConfig, Qwen25OmniProcessorConfig, Qwen25OmniTextConfig,
    Qwen25OmniVisionConfig,
};
pub use general::GeneralQwen25Omni;
pub use weights::Qwen25OmniTextWeights;

#[cfg(feature = "cuda")]
pub(crate) fn parse_binary_env(name: &str) -> std::result::Result<bool, String> {
    match std::env::var(name) {
        Err(std::env::VarError::NotPresent) => Ok(false),
        Ok(value) if value == "0" => Ok(false),
        Ok(value) if value == "1" => Ok(true),
        Ok(value) => Err(format!("{name} must be 0 or 1, got `{value}`")),
        Err(std::env::VarError::NotUnicode(_)) => Err(format!("{name} must be UTF-8")),
    }
}
