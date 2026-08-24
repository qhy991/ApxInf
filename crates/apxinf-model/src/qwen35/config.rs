//! Qwen3.5 configuration parsed from a Hugging Face `config.json`.
//!
//! Qwen3.5 is not a conventional decoder-only Transformer: its text stack
//! alternates recurrent Gated DeltaNet layers with full-attention layers.
//! Keeping the layer schedule and both sets of head dimensions explicit here
//! prevents a checkpoint from being accidentally loaded through the Qwen3-VL
//! or Llama implementations.

use std::path::Path;

use apxinf_core::{Error, Result};
use serde_json::Value;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Qwen35LayerType {
    LinearAttention,
    FullAttention,
}

impl Qwen35LayerType {
    fn parse(value: &str, index: usize) -> Result<Self> {
        match value {
            "linear_attention" => Ok(Self::LinearAttention),
            "full_attention" => Ok(Self::FullAttention),
            other => Err(Error::Other(format!(
                "qwen3.5 config: unsupported layer_types[{index}] `{other}`"
            ))),
        }
    }
}

#[derive(Clone, Debug)]
pub struct Qwen35RopeConfig {
    pub theta: f32,
    pub partial_rotary_factor: f32,
    /// `[T, H, W]` sections measured in rotary frequency pairs.
    pub mrope_section: [usize; 3],
    pub mrope_interleaved: bool,
}

#[derive(Clone, Debug)]
pub struct Qwen35TextConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub n_layers: usize,
    pub n_attention_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f32,
    pub hidden_act: String,
    pub eos_token_id: u32,
    pub tie_word_embeddings: bool,
    pub attention_bias: bool,
    pub attn_output_gate: bool,
    pub full_attention_interval: usize,
    pub layer_types: Vec<Qwen35LayerType>,
    pub linear_conv_kernel_dim: usize,
    pub linear_key_head_dim: usize,
    pub linear_num_key_heads: usize,
    pub linear_value_head_dim: usize,
    pub linear_num_value_heads: usize,
    pub mtp_num_hidden_layers: usize,
    pub mtp_use_dedicated_embeddings: bool,
    pub rope: Qwen35RopeConfig,
    /// HF checkpoint storage dtype (`bfloat16`, `float16`, or `float32`).
    pub dtype: String,
    /// Recurrent-state parameter dtype. Qwen3.5 checkpoints require float32.
    pub recurrent_state_dtype: String,
}

impl Qwen35TextConfig {
    pub fn full_query_width(&self) -> usize {
        self.n_attention_heads * self.head_dim
    }

    pub fn full_q_projection_width(&self) -> usize {
        self.full_query_width() * if self.attn_output_gate { 2 } else { 1 }
    }

    pub fn full_kv_width(&self) -> usize {
        self.n_kv_heads * self.head_dim
    }

    pub fn linear_key_width(&self) -> usize {
        self.linear_num_key_heads * self.linear_key_head_dim
    }

    pub fn linear_value_width(&self) -> usize {
        self.linear_num_value_heads * self.linear_value_head_dim
    }

    pub fn linear_qkv_width(&self) -> usize {
        self.linear_key_width() * 2 + self.linear_value_width()
    }

    pub fn rotary_dim(&self) -> usize {
        (self.head_dim as f32 * self.rope.partial_rotary_factor).round() as usize
    }

    pub fn validate(&self) -> Result<()> {
        let positive = [
            ("hidden_size", self.hidden_size),
            ("intermediate_size", self.intermediate_size),
            ("num_hidden_layers", self.n_layers),
            ("num_attention_heads", self.n_attention_heads),
            ("num_key_value_heads", self.n_kv_heads),
            ("head_dim", self.head_dim),
            ("vocab_size", self.vocab_size),
            ("max_position_embeddings", self.max_position_embeddings),
            ("full_attention_interval", self.full_attention_interval),
            ("linear_conv_kernel_dim", self.linear_conv_kernel_dim),
            ("linear_key_head_dim", self.linear_key_head_dim),
            ("linear_num_key_heads", self.linear_num_key_heads),
            ("linear_value_head_dim", self.linear_value_head_dim),
            ("linear_num_value_heads", self.linear_num_value_heads),
        ];
        for (name, value) in positive {
            if value == 0 {
                return Err(Error::Other(format!(
                    "qwen3.5 config: {name} must be positive"
                )));
            }
        }
        if !self.rms_norm_eps.is_finite() || self.rms_norm_eps <= 0.0 {
            return Err(Error::Other(
                "qwen3.5 config: rms_norm_eps must be finite and positive".into(),
            ));
        }
        if self.layer_types.len() != self.n_layers {
            return Err(Error::Other(format!(
                "qwen3.5 config: layer_types has {} entries, expected {}",
                self.layer_types.len(),
                self.n_layers
            )));
        }
        for (index, layer_type) in self.layer_types.iter().enumerate() {
            let expected = if (index + 1) % self.full_attention_interval == 0 {
                Qwen35LayerType::FullAttention
            } else {
                Qwen35LayerType::LinearAttention
            };
            if *layer_type != expected {
                return Err(Error::Other(format!(
                    "qwen3.5 config: layer {index} is {layer_type:?}, but \
                     full_attention_interval={} requires {expected:?}",
                    self.full_attention_interval
                )));
            }
        }
        if self.attention_bias {
            return Err(Error::Other(
                "qwen3.5 config: attention_bias=true is not supported by the native weight schema"
                    .into(),
            ));
        }
        if self.hidden_act != "silu" {
            return Err(Error::Other(format!(
                "qwen3.5 config: unsupported hidden_act `{}`",
                self.hidden_act
            )));
        }
        if self.n_attention_heads % self.n_kv_heads != 0 {
            return Err(Error::Other(format!(
                "qwen3.5 config: num_attention_heads {} is not divisible by num_key_value_heads {}",
                self.n_attention_heads, self.n_kv_heads
            )));
        }
        if self.linear_num_value_heads % self.linear_num_key_heads != 0 {
            return Err(Error::Other(format!(
                "qwen3.5 config: linear_num_value_heads {} is not divisible by linear_num_key_heads {}",
                self.linear_num_value_heads, self.linear_num_key_heads
            )));
        }
        if !matches!(self.dtype.as_str(), "bfloat16" | "float16" | "float32") {
            return Err(Error::Other(format!(
                "qwen3.5 config: unsupported text dtype `{}`",
                self.dtype
            )));
        }
        if self.recurrent_state_dtype != "float32" {
            return Err(Error::Other(format!(
                "qwen3.5 config: mamba_ssm_dtype must be float32, got `{}`",
                self.recurrent_state_dtype
            )));
        }
        if !self.rope.theta.is_finite() || self.rope.theta <= 0.0 {
            return Err(Error::Other(
                "qwen3.5 config: rope_theta must be finite and positive".into(),
            ));
        }
        if !self.rope.partial_rotary_factor.is_finite()
            || self.rope.partial_rotary_factor <= 0.0
            || self.rope.partial_rotary_factor > 1.0
        {
            return Err(Error::Other(
                "qwen3.5 config: partial_rotary_factor must be in (0, 1]".into(),
            ));
        }
        let exact_rotary_dim = self.head_dim as f32 * self.rope.partial_rotary_factor;
        if (exact_rotary_dim - exact_rotary_dim.round()).abs() > 1e-5 {
            return Err(Error::Other(format!(
                "qwen3.5 config: head_dim {} * partial_rotary_factor {} is not integral",
                self.head_dim, self.rope.partial_rotary_factor
            )));
        }
        let section_rotary_dim = 2 * self.rope.mrope_section.iter().sum::<usize>();
        if section_rotary_dim != self.rotary_dim() {
            return Err(Error::Other(format!(
                "qwen3.5 config: 2 * sum(mrope_section) is {section_rotary_dim}, \
                 but partial rotary dimension is {}",
                self.rotary_dim()
            )));
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct Qwen35VisionConfig {
    pub depth: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_heads: usize,
    pub in_channels: usize,
    pub patch_size: usize,
    pub temporal_patch_size: usize,
    pub spatial_merge_size: usize,
    pub num_position_embeddings: usize,
    pub out_hidden_size: usize,
    pub deepstack_visual_indexes: Vec<usize>,
}

#[derive(Clone, Debug)]
pub struct Qwen35Config {
    pub text: Qwen35TextConfig,
    pub vision: Qwen35VisionConfig,
    pub image_token_id: u32,
    pub video_token_id: u32,
    pub vision_start_token_id: u32,
    pub vision_end_token_id: u32,
}

impl Qwen35Config {
    pub fn from_json_file(path: &Path) -> Result<Self> {
        let raw = std::fs::read_to_string(path)
            .map_err(|error| Error::Other(format!("read {}: {error}", path.display())))?;
        Self::from_json_str(&raw)
    }

    pub fn from_json_str(raw: &str) -> Result<Self> {
        let root: Value = serde_json::from_str(raw)
            .map_err(|error| Error::Other(format!("qwen3.5 config json: {error}")))?;
        let model_type = required_str(&root, "model_type", "root")?;
        if model_type != "qwen3_5" {
            return Err(Error::Other(format!(
                "qwen3.5 config: expected model_type `qwen3_5`, got `{model_type}`"
            )));
        }

        let tc = required_object(&root, "text_config", "root")?;
        let text_model_type = required_str(tc, "model_type", "text_config")?;
        if text_model_type != "qwen3_5_text" {
            return Err(Error::Other(format!(
                "qwen3.5 config: expected text_config.model_type `qwen3_5_text`, got `{text_model_type}`"
            )));
        }
        let rope = required_object(tc, "rope_parameters", "text_config")?;
        let rope_type = required_str(rope, "rope_type", "text_config.rope_parameters")?;
        if rope_type != "default" {
            return Err(Error::Other(format!(
                "qwen3.5 config: unsupported rope_type `{rope_type}`"
            )));
        }
        let section = required_array(rope, "mrope_section", "text_config.rope_parameters")?;
        if section.len() != 3 {
            return Err(Error::Other(format!(
                "qwen3.5 config: mrope_section must have 3 entries, got {}",
                section.len()
            )));
        }
        let mrope_section = [
            value_to_usize(&section[0], "text_config.rope_parameters.mrope_section[0]")?,
            value_to_usize(&section[1], "text_config.rope_parameters.mrope_section[1]")?,
            value_to_usize(&section[2], "text_config.rope_parameters.mrope_section[2]")?,
        ];

        let raw_layer_types = required_array(tc, "layer_types", "text_config")?;
        let mut layer_types = Vec::with_capacity(raw_layer_types.len());
        for (index, value) in raw_layer_types.iter().enumerate() {
            let name = value.as_str().ok_or_else(|| {
                Error::Other(format!(
                    "qwen3.5 config: text_config.layer_types[{index}] must be a string"
                ))
            })?;
            layer_types.push(Qwen35LayerType::parse(name, index)?);
        }

        let mlp_only_layers = optional_usize_array(tc, "mlp_only_layers", "text_config")?;
        if !mlp_only_layers.is_empty() {
            return Err(Error::Other(format!(
                "qwen3.5 config: mlp_only_layers is not supported, got {mlp_only_layers:?}"
            )));
        }

        let text = Qwen35TextConfig {
            hidden_size: required_usize(tc, "hidden_size", "text_config")?,
            intermediate_size: required_usize(tc, "intermediate_size", "text_config")?,
            n_layers: required_usize(tc, "num_hidden_layers", "text_config")?,
            n_attention_heads: required_usize(tc, "num_attention_heads", "text_config")?,
            n_kv_heads: required_usize(tc, "num_key_value_heads", "text_config")?,
            head_dim: required_usize(tc, "head_dim", "text_config")?,
            vocab_size: required_usize(tc, "vocab_size", "text_config")?,
            max_position_embeddings: required_usize(tc, "max_position_embeddings", "text_config")?,
            rms_norm_eps: required_f32(tc, "rms_norm_eps", "text_config")?,
            hidden_act: required_str(tc, "hidden_act", "text_config")?.to_owned(),
            eos_token_id: required_u32(tc, "eos_token_id", "text_config")?,
            tie_word_embeddings: required_bool(tc, "tie_word_embeddings", "text_config")?,
            attention_bias: required_bool(tc, "attention_bias", "text_config")?,
            attn_output_gate: required_bool(tc, "attn_output_gate", "text_config")?,
            full_attention_interval: required_usize(tc, "full_attention_interval", "text_config")?,
            layer_types,
            linear_conv_kernel_dim: required_usize(tc, "linear_conv_kernel_dim", "text_config")?,
            linear_key_head_dim: required_usize(tc, "linear_key_head_dim", "text_config")?,
            linear_num_key_heads: required_usize(tc, "linear_num_key_heads", "text_config")?,
            linear_value_head_dim: required_usize(tc, "linear_value_head_dim", "text_config")?,
            linear_num_value_heads: required_usize(tc, "linear_num_value_heads", "text_config")?,
            mtp_num_hidden_layers: required_usize(tc, "mtp_num_hidden_layers", "text_config")?,
            mtp_use_dedicated_embeddings: required_bool(
                tc,
                "mtp_use_dedicated_embeddings",
                "text_config",
            )?,
            rope: Qwen35RopeConfig {
                theta: required_f32(rope, "rope_theta", "text_config.rope_parameters")?,
                partial_rotary_factor: required_f32(
                    rope,
                    "partial_rotary_factor",
                    "text_config.rope_parameters",
                )?,
                mrope_section,
                mrope_interleaved: required_bool(
                    rope,
                    "mrope_interleaved",
                    "text_config.rope_parameters",
                )?,
            },
            dtype: required_str(tc, "dtype", "text_config")?.to_owned(),
            recurrent_state_dtype: required_str(tc, "mamba_ssm_dtype", "text_config")?.to_owned(),
        };
        text.validate()?;

        let vc = required_object(&root, "vision_config", "root")?;
        let vision = Qwen35VisionConfig {
            depth: required_usize(vc, "depth", "vision_config")?,
            hidden_size: required_usize(vc, "hidden_size", "vision_config")?,
            intermediate_size: required_usize(vc, "intermediate_size", "vision_config")?,
            num_heads: required_usize(vc, "num_heads", "vision_config")?,
            in_channels: required_usize(vc, "in_channels", "vision_config")?,
            patch_size: required_usize(vc, "patch_size", "vision_config")?,
            temporal_patch_size: required_usize(vc, "temporal_patch_size", "vision_config")?,
            spatial_merge_size: required_usize(vc, "spatial_merge_size", "vision_config")?,
            num_position_embeddings: required_usize(
                vc,
                "num_position_embeddings",
                "vision_config",
            )?,
            out_hidden_size: required_usize(vc, "out_hidden_size", "vision_config")?,
            deepstack_visual_indexes: optional_usize_array(
                vc,
                "deepstack_visual_indexes",
                "vision_config",
            )?,
        };
        validate_vision(&vision, text.hidden_size)?;

        Ok(Self {
            text,
            vision,
            image_token_id: required_u32(&root, "image_token_id", "root")?,
            video_token_id: required_u32(&root, "video_token_id", "root")?,
            vision_start_token_id: required_u32(&root, "vision_start_token_id", "root")?,
            vision_end_token_id: required_u32(&root, "vision_end_token_id", "root")?,
        })
    }
}

fn validate_vision(config: &Qwen35VisionConfig, text_hidden_size: usize) -> Result<()> {
    let positive = [
        ("depth", config.depth),
        ("hidden_size", config.hidden_size),
        ("intermediate_size", config.intermediate_size),
        ("num_heads", config.num_heads),
        ("in_channels", config.in_channels),
        ("patch_size", config.patch_size),
        ("temporal_patch_size", config.temporal_patch_size),
        ("spatial_merge_size", config.spatial_merge_size),
        ("num_position_embeddings", config.num_position_embeddings),
        ("out_hidden_size", config.out_hidden_size),
    ];
    for (name, value) in positive {
        if value == 0 {
            return Err(Error::Other(format!(
                "qwen3.5 config: vision_config.{name} must be positive"
            )));
        }
    }
    if config.hidden_size % config.num_heads != 0 {
        return Err(Error::Other(format!(
            "qwen3.5 config: vision hidden_size {} is not divisible by num_heads {}",
            config.hidden_size, config.num_heads
        )));
    }
    if config.out_hidden_size != text_hidden_size {
        return Err(Error::Other(format!(
            "qwen3.5 config: vision out_hidden_size {} does not match text hidden_size {text_hidden_size}",
            config.out_hidden_size
        )));
    }
    if let Some(index) = config
        .deepstack_visual_indexes
        .iter()
        .copied()
        .find(|index| *index >= config.depth)
    {
        return Err(Error::Other(format!(
            "qwen3.5 config: deepstack visual layer {index} is outside depth {}",
            config.depth
        )));
    }
    Ok(())
}

fn required_object<'a>(value: &'a Value, key: &str, context: &str) -> Result<&'a Value> {
    value
        .get(key)
        .filter(|value| value.is_object())
        .ok_or_else(|| Error::Other(format!("qwen3.5 config: missing object {context}.{key}")))
}

fn required_array<'a>(value: &'a Value, key: &str, context: &str) -> Result<&'a [Value]> {
    value
        .get(key)
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .ok_or_else(|| Error::Other(format!("qwen3.5 config: missing array {context}.{key}")))
}

fn required_str<'a>(value: &'a Value, key: &str, context: &str) -> Result<&'a str> {
    value
        .get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| Error::Other(format!("qwen3.5 config: missing string {context}.{key}")))
}

fn required_bool(value: &Value, key: &str, context: &str) -> Result<bool> {
    value
        .get(key)
        .and_then(Value::as_bool)
        .ok_or_else(|| Error::Other(format!("qwen3.5 config: missing boolean {context}.{key}")))
}

fn required_usize(value: &Value, key: &str, context: &str) -> Result<usize> {
    let value = value
        .get(key)
        .ok_or_else(|| Error::Other(format!("qwen3.5 config: missing integer {context}.{key}")))?;
    value_to_usize(value, &format!("{context}.{key}"))
}

fn required_u32(value: &Value, key: &str, context: &str) -> Result<u32> {
    let number = value
        .get(key)
        .and_then(Value::as_u64)
        .ok_or_else(|| Error::Other(format!("qwen3.5 config: missing integer {context}.{key}")))?;
    u32::try_from(number).map_err(|_| {
        Error::Other(format!(
            "qwen3.5 config: {context}.{key}={number} exceeds u32"
        ))
    })
}

fn required_f32(value: &Value, key: &str, context: &str) -> Result<f32> {
    let number = value
        .get(key)
        .and_then(Value::as_f64)
        .ok_or_else(|| Error::Other(format!("qwen3.5 config: missing number {context}.{key}")))?;
    if !number.is_finite() || number < -(f32::MAX as f64) || number > f32::MAX as f64 {
        return Err(Error::Other(format!(
            "qwen3.5 config: {context}.{key} is outside f32 range"
        )));
    }
    Ok(number as f32)
}

fn value_to_usize(value: &Value, path: &str) -> Result<usize> {
    let number = value.as_u64().ok_or_else(|| {
        Error::Other(format!(
            "qwen3.5 config: {path} must be an unsigned integer"
        ))
    })?;
    usize::try_from(number)
        .map_err(|_| Error::Other(format!("qwen3.5 config: {path} exceeds usize")))
}

fn optional_usize_array(value: &Value, key: &str, context: &str) -> Result<Vec<usize>> {
    let Some(raw) = value.get(key) else {
        return Ok(Vec::new());
    };
    let raw = raw
        .as_array()
        .ok_or_else(|| Error::Other(format!("qwen3.5 config: {context}.{key} must be an array")))?;
    raw.iter()
        .enumerate()
        .map(|(index, value)| value_to_usize(value, &format!("{context}.{key}[{index}]")))
        .collect()
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    pub(crate) const MINI_CONFIG: &str = r#"{
        "model_type": "qwen3_5",
        "image_token_id": 101,
        "video_token_id": 102,
        "vision_start_token_id": 103,
        "vision_end_token_id": 104,
        "text_config": {
            "attention_bias": false,
            "attn_output_gate": true,
            "dtype": "bfloat16",
            "eos_token_id": 2,
            "full_attention_interval": 4,
            "head_dim": 8,
            "hidden_act": "silu",
            "hidden_size": 8,
            "intermediate_size": 12,
            "layer_types": [
                "linear_attention", "linear_attention", "linear_attention", "full_attention"
            ],
            "linear_conv_kernel_dim": 4,
            "linear_key_head_dim": 4,
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 2,
            "linear_value_head_dim": 4,
            "max_position_embeddings": 128,
            "mlp_only_layers": [],
            "model_type": "qwen3_5_text",
            "mtp_num_hidden_layers": 1,
            "mtp_use_dedicated_embeddings": false,
            "num_attention_heads": 2,
            "num_hidden_layers": 4,
            "num_key_value_heads": 1,
            "rms_norm_eps": 1e-6,
            "tie_word_embeddings": true,
            "vocab_size": 32,
            "mamba_ssm_dtype": "float32",
            "rope_parameters": {
                "mrope_interleaved": true,
                "mrope_section": [1, 1, 1],
                "rope_type": "default",
                "rope_theta": 10000000,
                "partial_rotary_factor": 0.75
            }
        },
        "vision_config": {
            "deepstack_visual_indexes": [],
            "depth": 2,
            "hidden_size": 8,
            "in_channels": 3,
            "intermediate_size": 16,
            "num_heads": 2,
            "num_position_embeddings": 16,
            "out_hidden_size": 8,
            "patch_size": 2,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2
        }
    }"#;

    #[test]
    fn parses_hybrid_schedule_and_derived_dimensions() {
        let config = Qwen35Config::from_json_str(MINI_CONFIG).unwrap();
        assert_eq!(config.text.layer_types.len(), 4);
        assert_eq!(
            config.text.layer_types,
            vec![
                Qwen35LayerType::LinearAttention,
                Qwen35LayerType::LinearAttention,
                Qwen35LayerType::LinearAttention,
                Qwen35LayerType::FullAttention,
            ]
        );
        assert_eq!(config.text.full_query_width(), 16);
        assert_eq!(config.text.full_q_projection_width(), 32);
        assert_eq!(config.text.full_kv_width(), 8);
        assert_eq!(config.text.linear_qkv_width(), 24);
        assert_eq!(config.text.rotary_dim(), 6);
        assert_eq!(config.vision.out_hidden_size, config.text.hidden_size);
    }

    #[test]
    fn rejects_schedule_that_disagrees_with_interval() {
        let raw = MINI_CONFIG.replacen(
            "\"linear_attention\", \"linear_attention\", \"linear_attention\", \"full_attention\"",
            "\"full_attention\", \"linear_attention\", \"linear_attention\", \"full_attention\"",
            1,
        );
        let error = Qwen35Config::from_json_str(&raw).unwrap_err();
        assert!(error.to_string().contains("layer 0"));
        assert!(error.to_string().contains("full_attention_interval"));
    }

    #[test]
    fn rejects_mrope_partition_that_does_not_cover_rotary_dims() {
        let raw = MINI_CONFIG.replacen("[1, 1, 1]", "[1, 1, 2]", 1);
        let error = Qwen35Config::from_json_str(&raw).unwrap_err();
        assert!(error.to_string().contains("sum(mrope_section)"));
    }

    #[test]
    fn rejects_wrong_model_family() {
        let raw = MINI_CONFIG.replacen("\"qwen3_5\"", "\"qwen3_vl\"", 1);
        let error = Qwen35Config::from_json_str(&raw).unwrap_err();
        assert!(error.to_string().contains("expected model_type `qwen3_5`"));
    }
}
