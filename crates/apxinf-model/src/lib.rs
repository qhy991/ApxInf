//! LLM model architectures and abstractions.

mod accelerator;
pub mod auto;
pub mod builtin;
pub mod debug;
mod generation_config;
pub mod llama;
pub mod llm_trait;
pub mod pi05;
pub mod profiling;
pub mod qwen25_omni;
pub mod qwen3vl;
pub mod registry;
pub mod vla;

pub use auto::{AutoModel, LoadOptions, LoadedModel, ModelPrecision, SyntheticWeights};
pub use builtin::{register_builtin_models, validate_qwen25_omni_load_options};
pub use debug::{DebugCapture, DebugConfig};
pub use generation_config::{GenerationConfigSource, GenerationOptions, SamplingMode};
#[cfg(feature = "cuda")]
pub use llama::{DecodeGraph, DecodeGraphConfig, DecodeGraphWeights, DecodeLayerWeights};
pub use llama::{GeneralLlama, KVCache, LlamaModel, LlamaWeights, TransformerLayer};
pub use llm_trait::{
    generate_streaming, generate_streaming_with_options, AudioInput, GeneratedToken,
    GenerationOutput, GenerationRequest, ImageInput, LlmCapabilities, LlmInput, LlmTrait,
};
pub use pi05::{Pi05Config, Pi05PerformanceProfile};
pub use profiling::GenerationProfile;
pub use qwen25_omni::{Qwen25OmniCheckpointReport, Qwen25OmniConfig};
pub use qwen3vl::{GeneralQwen3VL, Qwen3VLConfig, Qwen3VLTextWeights};
pub use registry::{get, list, register};
pub use vla::{
    Action, ImageLayout, InferenceSpec, InitialLatent, Observation, PreparedInference,
    VisionObservation, VlaRequest, VlaRuntime,
};
