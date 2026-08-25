//! LLM model architectures and abstractions.

mod accelerator;
pub mod builtin;
pub mod debug;
pub mod llama;
pub mod llm_trait;
pub mod qwen25_omni;
pub mod registry;
pub mod auto;
pub mod profiling;
pub mod pi05;
pub mod qwen3vl;
pub mod vla;

pub use auto::{AutoModel, LoadOptions, LoadedModel, ModelPrecision, SyntheticWeights};
pub use builtin::{register_builtin_models, validate_qwen25_omni_load_options};
pub use debug::{DebugCapture, DebugConfig};
#[cfg(feature = "cuda")]
pub use llama::{DecodeGraph, DecodeGraphConfig, DecodeGraphWeights, DecodeLayerWeights};
pub use llama::{GeneralLlama, KVCache, LlamaModel, LlamaWeights, TransformerLayer};
pub use llm_trait::{
    generate_streaming, AudioInput, ImageInput, LlmCapabilities, LlmInput, LlmTrait,
};
pub use pi05::{Pi05Config, Pi05PerformanceProfile};
pub use profiling::GenerationProfile;
pub use qwen25_omni::{Qwen25OmniCheckpointReport, Qwen25OmniConfig};
pub use registry::{get, list, register};
pub use qwen3vl::{GeneralQwen3VL, Qwen3VLConfig, Qwen3VLTextWeights};
pub use vla::{
    Action, ImageLayout, InferenceSpec, Observation, PreparedInference, VisionObservation,
    VlaRuntime,
};
