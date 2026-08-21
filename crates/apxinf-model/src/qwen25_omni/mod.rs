//! Native Qwen2.5-Omni Thinker (text/image/audio to text).

pub mod audio;
pub mod checkpoint;
pub mod config;
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
