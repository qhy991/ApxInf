//! Built-in model registrations used by [`crate::AutoModel`].

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use apxinf_core::{Backend, DType, Device, Error, Result, Tensor};

use crate::auto::{LoadOptions, LoadedModel, ModelPrecision};
use crate::llama::{GeneralLlama, LlamaWeights};
use crate::qwen25_omni::{checkpoint, GeneralQwen25Omni, Qwen25OmniConfig};
use crate::qwen3vl::{GeneralQwen3VL, Qwen3VLConfig};
use crate::registry;

/// Register every implementation shipped in this crate. Re-registering is
/// harmless and keeps `AutoModel::load_model` self-contained for users.
pub fn register_builtin_models() {
    registry::register("llama", load_llama);
    registry::register("qwen3_vl", load_qwen3vl);
    registry::register("qwen3vl", load_qwen3vl);
    registry::register("qwen2_5_omni", load_qwen25_omni);

    #[cfg(feature = "cuda")]
    crate::pi05::register_builtin();
}

fn load_qwen25_omni(
    path: &Path,
    device: Device,
    backend: Arc<dyn Backend>,
    options: &LoadOptions,
) -> Result<LoadedModel> {
    validate_qwen25_omni_load_options(device, options)?;
    #[cfg(feature = "cuda")]
    {
        let chunk_tactics = qwen25_omni_chunk_tactics_enabled()?;
        let m1_packed_mlp = qwen25_omni_m1_packed_mlp_enabled()?;
        if chunk_tactics || m1_packed_mlp {
            use crate::accelerator::cuda::{downcast, kernels};
            let cuda = downcast(&*backend)
                .ok_or_else(|| Error::Other("Qwen2.5-Omni tactics require CudaBackend".into()))?;
            if cuda.context().caps().sm != 89
                || cuda.context().caps().device_name != "NVIDIA GeForce RTX 4090"
            {
                return Err(Error::Other(format!(
                    "Qwen2.5-Omni chunk tactics require RTX 4090 SM89, got {} SM{}",
                    cuda.context().caps().device_name,
                    cuda.context().caps().sm
                )));
            }
            use kernels::gemm::Bf16CublasLtTactic as Tactic;
            let mut tactics = Vec::new();
            if chunk_tactics {
                tactics.extend([
                    Tactic {
                        m: 256,
                        n: 2560,
                        k: 2048,
                        heuristic_rank: 2,
                        milliseconds: 0.029792001470923424,
                    },
                    Tactic {
                        m: 256,
                        n: 11008,
                        k: 2048,
                        heuristic_rank: 1,
                        milliseconds: 0.08908800780773163,
                    },
                    Tactic {
                        m: 256,
                        n: 2048,
                        k: 11008,
                        heuristic_rank: 2,
                        milliseconds: 0.09156159311532974,
                    },
                    Tactic {
                        m: 1024,
                        n: 2560,
                        k: 2048,
                        heuristic_rank: 2,
                        milliseconds: 0.07874559611082077,
                    },
                    Tactic {
                        m: 1024,
                        n: 11008,
                        k: 2048,
                        heuristic_rank: 1,
                        milliseconds: 0.2900064289569855,
                    },
                ]);
            }
            if m1_packed_mlp {
                tactics.push(Tactic {
                    m: 1,
                    n: 22016,
                    k: 2048,
                    heuristic_rank: 1,
                    milliseconds: 0.10788639634847641,
                });
            }
            kernels::gemm::install_cublaslt_bf16_tactics(cuda.context(), &tactics)?;
        }
    }
    let model_dir = if path.is_dir() {
        path
    } else {
        path.parent().unwrap_or_else(|| Path::new("."))
    };
    let config = Qwen25OmniConfig::from_model_dir(model_dir)?;
    let (tensors, _report) = checkpoint::load_required_tensors(model_dir, &config)?;
    let model = GeneralQwen25Omni::from_selected_weights(config, tensors, backend)?;
    Ok(LoadedModel::Text(Box::new(model)))
}

#[cfg(feature = "cuda")]
fn qwen25_omni_chunk_tactics_enabled() -> Result<bool> {
    crate::qwen25_omni::parse_binary_env("APXINF_QWEN25_BF16_CHUNK_TACTICS").map_err(Error::Other)
}

#[cfg(feature = "cuda")]
fn qwen25_omni_m1_packed_mlp_enabled() -> Result<bool> {
    crate::qwen25_omni::parse_binary_env("APXINF_QWEN25_M1_PACKED_MLP").map_err(Error::Other)
}

pub fn validate_qwen25_omni_load_options(device: Device, options: &LoadOptions) -> Result<()> {
    if !matches!(device, Device::Cuda(_)) {
        return Err(Error::Other(
            "qwen2.5-omni BF16 deployment requires a CUDA device".into(),
        ));
    }
    if !matches!(
        options.precision,
        ModelPrecision::Auto | ModelPrecision::Bf16
    ) {
        return Err(Error::Other(format!(
            "qwen2.5-omni deployment supports only checkpoint-native BF16, not {:?}",
            options.precision
        )));
    }
    if options
        .text_weight_dtype
        .is_some_and(|dtype| dtype != DType::BF16)
    {
        return Err(Error::Other(
            "qwen2.5-omni text_weight_dtype must be BF16 when specified".into(),
        ));
    }
    if options.calibration_path.is_some()
        || options.tuning_path.is_some()
        || options.config.is_some()
        || options.synthetic.is_some()
        || options.uniform_fp8_scale.is_some()
    {
        return Err(Error::Other(
            "qwen2.5-omni rejects calibration, tuning, config overrides, and synthetic weights"
                .into(),
        ));
    }
    Ok(())
}
fn load_llama(
    path: &Path,
    device: Device,
    backend: Arc<dyn Backend>,
    options: &LoadOptions,
) -> Result<LoadedModel> {
    let (mut tensors, metadata) = apxinf_loader::safetensors::load_native_path(path)
        .map_err(|error| Error::Other(format!("load {}: {error}", path.display())))?;
    if let Some(dtype) = options.text_weight_dtype {
        if !matches!(dtype, DType::F32 | DType::BF16) {
            return Err(Error::Other(format!(
                "Llama text weights support f32 or bf16, not {dtype}"
            )));
        }
    }
    if matches!(device, Device::Cpu) || options.text_weight_dtype == Some(DType::F32) {
        upcast_bf16_weights(&mut tensors)?;
    }
    let config = apxinf_loader::safetensors::config_from_metadata(&metadata);
    let weights = LlamaWeights::from_map(&config, tensors)?;
    Ok(LoadedModel::Text(Box::new(GeneralLlama::new(
        config, weights, backend,
    )?)))
}

fn load_qwen3vl(
    path: &Path,
    _device: Device,
    backend: Arc<dyn Backend>,
    _options: &LoadOptions,
) -> Result<LoadedModel> {
    let model_dir = if path.is_dir() {
        path
    } else {
        path.parent().unwrap_or_else(|| Path::new("."))
    };
    let config = Qwen3VLConfig::from_json_file(&model_dir.join("config.json"))?;
    let (tensors, _) = apxinf_loader::safetensors::load_native_path(path)
        .map_err(|error| Error::Other(format!("load {}: {error}", path.display())))?;
    let model = GeneralQwen3VL::from_weights_with_backend(config, tensors, backend)?;
    Ok(LoadedModel::Text(Box::new(model)))
}

fn upcast_bf16_weights(tensors: &mut HashMap<String, Tensor>) -> Result<()> {
    for tensor in tensors.values_mut() {
        if tensor.dtype() != DType::BF16 {
            continue;
        }
        let shape = tensor.shape().dims().to_vec();
        *tensor = Tensor::from_f32(shape, &tensor.to_f32_vec()?)?;
    }
    Ok(())
}
