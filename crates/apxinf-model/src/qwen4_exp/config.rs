//! Strict Hugging Face configuration contract for Qwen3.8-Flash-Next.

use std::path::Path;

use apxinf_core::{Error, Result};
use serde_json::Value;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Qwen4ExpLayerType {
    LinearAttention,
    QwenSparseAttention,
}

#[derive(Clone, Debug)]
pub struct Qwen4ExpRopeConfig {
    pub theta: f32,
    pub partial_rotary_factor: f32,
    pub mrope_section: [usize; 3],
    pub mrope_interleaved: bool,
}

#[derive(Clone, Debug)]
pub struct Qwen4ExpTextConfig {
    pub hidden_size: usize,
    pub n_layers: usize,
    pub n_attention_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f32,
    pub hidden_act: String,
    pub output_gate_type: String,
    pub eos_token_id: u32,
    pub tie_word_embeddings: bool,
    pub attention_bias: bool,
    pub full_attention_interval: usize,
    pub layer_types: Vec<Qwen4ExpLayerType>,
    pub linear_conv_kernel_dim: usize,
    pub linear_key_head_dim: usize,
    pub linear_num_key_heads: usize,
    pub linear_value_head_dim: usize,
    pub linear_num_value_heads: usize,
    pub moe_intermediate_size: usize,
    pub shared_expert_intermediate_size: usize,
    pub num_experts: usize,
    pub num_experts_per_tok: usize,
    pub hc_count: usize,
    pub hc_lowrank: usize,
    pub ple_layer_ids: Vec<usize>,
    pub ple_embed_dim: usize,
    pub ple_conv_kernel_size: usize,
    pub ngram_size: usize,
    pub heads_per_ngram: usize,
    pub ngram_vocab_size_base: usize,
    pub make_ngram_vocab_size_divisible_by: usize,
    pub split_ngram_parts: usize,
    pub seed: u64,
    pub indexer_n_heads: usize,
    pub indexer_kv_heads: usize,
    pub indexer_head_dim: usize,
    pub indexer_budget: usize,
    pub indexer_compress_ratio: usize,
    pub mtp_num_hidden_layers: usize,
    pub mtp_use_dedicated_embeddings: bool,
    pub norm_topk_prob: bool,
    pub rope: Qwen4ExpRopeConfig,
    pub dtype: String,
    pub recurrent_state_dtype: String,
}

impl Qwen4ExpTextConfig {
    pub fn full_query_width(&self) -> usize {
        self.n_attention_heads * self.head_dim
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

    pub fn rotary_dim(&self) -> usize {
        (self.head_dim as f32 * self.rope.partial_rotary_factor).round() as usize
    }
}

#[derive(Clone, Debug)]
pub struct Qwen4ExpVisionConfig {
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
pub struct Qwen4ExpConfig {
    pub text: Qwen4ExpTextConfig,
    pub vision: Qwen4ExpVisionConfig,
    pub language_model_only: bool,
    pub image_token_id: u32,
    pub video_token_id: u32,
    pub vision_start_token_id: u32,
    pub vision_end_token_id: u32,
}

impl Qwen4ExpConfig {
    pub fn from_json_file(path: &Path) -> Result<Self> {
        let raw = std::fs::read_to_string(path)
            .map_err(|error| Error::Other(format!("read {}: {error}", path.display())))?;
        Self::from_json_str(&raw)
    }

    pub fn from_json_str(raw: &str) -> Result<Self> {
        let root: Value = serde_json::from_str(raw)
            .map_err(|error| Error::Other(format!("qwen4-exp config json: {error}")))?;
        let model_type = required_str(&root, "model_type", "root")?;
        if model_type != "qwen4_exp" {
            return Err(config_error(format!(
                "expected model_type `qwen4_exp`, got `{model_type}`"
            )));
        }

        let tc = required_object(&root, "text_config", "root")?;
        let text_model_type = required_str(tc, "model_type", "text_config")?;
        if text_model_type != "qwen4_exp_text" {
            return Err(config_error(format!(
                "expected text_config.model_type `qwen4_exp_text`, got `{text_model_type}`"
            )));
        }

        let rope_value = required_object(tc, "rope_parameters", "text_config")?;
        let rope_type = required_str(rope_value, "rope_type", "text_config.rope_parameters")?;
        if rope_type != "default" {
            return Err(config_error(format!("unsupported rope_type `{rope_type}`")));
        }
        let section = required_array(rope_value, "mrope_section", "text_config.rope_parameters")?;
        if section.len() != 3 {
            return Err(config_error(format!(
                "mrope_section must have 3 entries, got {}",
                section.len()
            )));
        }
        let rope = Qwen4ExpRopeConfig {
            theta: required_f32(rope_value, "rope_theta", "text_config.rope_parameters")?,
            partial_rotary_factor: required_f32(
                rope_value,
                "partial_rotary_factor",
                "text_config.rope_parameters",
            )?,
            mrope_section: [
                value_to_usize(&section[0], "text_config.rope_parameters.mrope_section[0]")?,
                value_to_usize(&section[1], "text_config.rope_parameters.mrope_section[1]")?,
                value_to_usize(&section[2], "text_config.rope_parameters.mrope_section[2]")?,
            ],
            mrope_interleaved: required_bool(
                rope_value,
                "mrope_interleaved",
                "text_config.rope_parameters",
            )?,
        };

        let raw_layer_types = required_array(tc, "layer_types", "text_config")?;
        let mut layer_types = Vec::with_capacity(raw_layer_types.len());
        for (index, value) in raw_layer_types.iter().enumerate() {
            let name = value.as_str().ok_or_else(|| {
                config_error(format!("text_config.layer_types[{index}] must be a string"))
            })?;
            let layer_type = match name {
                "linear_attention" => Qwen4ExpLayerType::LinearAttention,
                "full_attention" | "qwen_sparse_attention" => {
                    Qwen4ExpLayerType::QwenSparseAttention
                }
                other => {
                    return Err(config_error(format!(
                        "unsupported layer_types[{index}] `{other}`"
                    )))
                }
            };
            layer_types.push(layer_type);
        }

        let text = Qwen4ExpTextConfig {
            hidden_size: required_usize(tc, "hidden_size", "text_config")?,
            n_layers: required_usize(tc, "num_hidden_layers", "text_config")?,
            n_attention_heads: required_usize(tc, "num_attention_heads", "text_config")?,
            n_kv_heads: required_usize(tc, "num_key_value_heads", "text_config")?,
            head_dim: required_usize(tc, "head_dim", "text_config")?,
            vocab_size: required_usize(tc, "vocab_size", "text_config")?,
            max_position_embeddings: required_usize(tc, "max_position_embeddings", "text_config")?,
            rms_norm_eps: required_f32(tc, "rms_norm_eps", "text_config")?,
            hidden_act: required_str(tc, "hidden_act", "text_config")?.to_owned(),
            output_gate_type: required_str(tc, "output_gate_type", "text_config")?.to_owned(),
            eos_token_id: required_u32(tc, "eos_token_id", "text_config")?,
            tie_word_embeddings: required_bool(tc, "tie_word_embeddings", "text_config")?,
            attention_bias: required_bool(tc, "attention_bias", "text_config")?,
            full_attention_interval: required_usize(tc, "full_attention_interval", "text_config")?,
            layer_types,
            linear_conv_kernel_dim: required_usize(tc, "linear_conv_kernel_dim", "text_config")?,
            linear_key_head_dim: required_usize(tc, "linear_key_head_dim", "text_config")?,
            linear_num_key_heads: required_usize(tc, "linear_num_key_heads", "text_config")?,
            linear_value_head_dim: required_usize(tc, "linear_value_head_dim", "text_config")?,
            linear_num_value_heads: required_usize(tc, "linear_num_value_heads", "text_config")?,
            moe_intermediate_size: required_usize(tc, "moe_intermediate_size", "text_config")?,
            shared_expert_intermediate_size: required_usize(
                tc,
                "shared_expert_intermediate_size",
                "text_config",
            )?,
            num_experts: required_usize(tc, "num_experts", "text_config")?,
            num_experts_per_tok: required_usize(tc, "num_experts_per_tok", "text_config")?,
            hc_count: required_usize(tc, "hc_count", "text_config")?,
            hc_lowrank: required_usize(tc, "hc_lowrank", "text_config")?,
            ple_layer_ids: required_usize_array(tc, "ple_layer_ids", "text_config")?,
            ple_embed_dim: required_usize(tc, "ple_embed_dim", "text_config")?,
            ple_conv_kernel_size: required_usize(tc, "ple_conv_kernel_size", "text_config")?,
            ngram_size: required_usize(tc, "ngram_size", "text_config")?,
            heads_per_ngram: required_usize(tc, "heads_per_ngram", "text_config")?,
            ngram_vocab_size_base: required_usize(tc, "ngram_vocab_size_base", "text_config")?,
            make_ngram_vocab_size_divisible_by: required_usize(
                tc,
                "make_ngram_vocab_size_divisible_by",
                "text_config",
            )?,
            split_ngram_parts: required_usize(tc, "split_ngram_parts", "text_config")?,
            seed: optional_u64(tc, "seed", "text_config")?.unwrap_or(1234),
            indexer_n_heads: required_usize(tc, "indexer_n_heads", "text_config")?,
            indexer_kv_heads: required_usize(tc, "indexer_kv_heads", "text_config")?,
            indexer_head_dim: required_usize(tc, "indexer_head_dim", "text_config")?,
            indexer_budget: required_usize(tc, "indexer_budget", "text_config")?,
            indexer_compress_ratio: required_usize(tc, "indexer_compress_ratio", "text_config")?,
            mtp_num_hidden_layers: required_usize(tc, "mtp_num_hidden_layers", "text_config")?,
            mtp_use_dedicated_embeddings: required_bool(
                tc,
                "mtp_use_dedicated_embeddings",
                "text_config",
            )?,
            norm_topk_prob: optional_bool(tc, "norm_topk_prob", "text_config")?.unwrap_or(true),
            rope,
            dtype: required_str(tc, "dtype", "text_config")?.to_owned(),
            recurrent_state_dtype: required_str(tc, "mamba_ssm_dtype", "text_config")?.to_owned(),
        };
        text.validate()?;

        let vc = required_object(&root, "vision_config", "root")?;
        let vision = Qwen4ExpVisionConfig {
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
            deepstack_visual_indexes: required_usize_array(
                vc,
                "deepstack_visual_indexes",
                "vision_config",
            )?,
        };
        validate_vision(&vision, text.hidden_size)?;

        Ok(Self {
            text,
            vision,
            language_model_only: required_bool(&root, "language_model_only", "root")?,
            image_token_id: required_u32(&root, "image_token_id", "root")?,
            video_token_id: required_u32(&root, "video_token_id", "root")?,
            vision_start_token_id: required_u32(&root, "vision_start_token_id", "root")?,
            vision_end_token_id: required_u32(&root, "vision_end_token_id", "root")?,
        })
    }
}

impl Qwen4ExpTextConfig {
    fn validate(&self) -> Result<()> {
        let positive = [
            ("hidden_size", self.hidden_size),
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
            ("moe_intermediate_size", self.moe_intermediate_size),
            (
                "shared_expert_intermediate_size",
                self.shared_expert_intermediate_size,
            ),
            ("num_experts", self.num_experts),
            ("num_experts_per_tok", self.num_experts_per_tok),
            ("hc_lowrank", self.hc_lowrank),
            ("ple_embed_dim", self.ple_embed_dim),
            ("ple_conv_kernel_size", self.ple_conv_kernel_size),
            ("ngram_size", self.ngram_size),
            ("heads_per_ngram", self.heads_per_ngram),
            ("ngram_vocab_size_base", self.ngram_vocab_size_base),
            (
                "make_ngram_vocab_size_divisible_by",
                self.make_ngram_vocab_size_divisible_by,
            ),
            ("split_ngram_parts", self.split_ngram_parts),
            ("indexer_n_heads", self.indexer_n_heads),
            ("indexer_kv_heads", self.indexer_kv_heads),
            ("indexer_head_dim", self.indexer_head_dim),
            ("indexer_budget", self.indexer_budget),
            ("indexer_compress_ratio", self.indexer_compress_ratio),
        ];
        for (name, value) in positive {
            if value == 0 {
                return Err(config_error(format!("{name} must be positive")));
            }
        }
        if self.hc_count <= 1 {
            return Err(config_error(format!(
                "hc_count must be greater than one, got {}",
                self.hc_count
            )));
        }
        if self.num_experts_per_tok > self.num_experts {
            return Err(config_error(format!(
                "num_experts_per_tok {} exceeds num_experts {}",
                self.num_experts_per_tok, self.num_experts
            )));
        }
        if self.layer_types.len() != self.n_layers {
            return Err(config_error(format!(
                "layer_types has {} entries, expected {}",
                self.layer_types.len(),
                self.n_layers
            )));
        }
        for (index, layer_type) in self.layer_types.iter().enumerate() {
            let expected = if (index + 1) % self.full_attention_interval == 0 {
                Qwen4ExpLayerType::QwenSparseAttention
            } else {
                Qwen4ExpLayerType::LinearAttention
            };
            if *layer_type != expected {
                return Err(config_error(format!(
                    "layer {index} is {layer_type:?}, but full_attention_interval={} requires {expected:?}",
                    self.full_attention_interval
                )));
            }
        }
        if self.attention_bias {
            return Err(config_error(
                "attention_bias=true is not supported by the released weight schema",
            ));
        }
        if self.tie_word_embeddings {
            return Err(config_error(
                "tie_word_embeddings=true is not supported by the released weight schema",
            ));
        }
        if self.hidden_act != "silu" {
            return Err(config_error(format!(
                "unsupported hidden_act `{}`",
                self.hidden_act
            )));
        }
        if !matches!(self.output_gate_type.as_str(), "sigmoid" | "silu") {
            return Err(config_error(format!(
                "unsupported output_gate_type `{}`",
                self.output_gate_type
            )));
        }
        if !self.n_attention_heads.is_multiple_of(self.n_kv_heads) {
            return Err(config_error(format!(
                "num_attention_heads {} is not divisible by num_key_value_heads {}",
                self.n_attention_heads, self.n_kv_heads
            )));
        }
        if !self
            .linear_num_value_heads
            .is_multiple_of(self.linear_num_key_heads)
        {
            return Err(config_error(format!(
                "linear_num_value_heads {} is not divisible by linear_num_key_heads {}",
                self.linear_num_value_heads, self.linear_num_key_heads
            )));
        }
        if self.indexer_kv_heads != 1 {
            return Err(config_error(format!(
                "QSA requires indexer_kv_heads=1, got {}",
                self.indexer_kv_heads
            )));
        }
        if !self
            .indexer_budget
            .is_multiple_of(self.indexer_compress_ratio)
        {
            return Err(config_error(format!(
                "indexer_budget {} is not divisible by indexer_compress_ratio {}",
                self.indexer_budget, self.indexer_compress_ratio
            )));
        }
        if self.rotary_dim() > self.indexer_head_dim {
            return Err(config_error(format!(
                "rotary dimension {} exceeds indexer_head_dim {}",
                self.rotary_dim(),
                self.indexer_head_dim
            )));
        }
        if !self.rms_norm_eps.is_finite() || self.rms_norm_eps <= 0.0 {
            return Err(config_error("rms_norm_eps must be finite and positive"));
        }
        if !matches!(self.dtype.as_str(), "bfloat16" | "float16" | "float32") {
            return Err(config_error(format!(
                "unsupported text dtype `{}`",
                self.dtype
            )));
        }
        if self.recurrent_state_dtype != "float32" {
            return Err(config_error(format!(
                "mamba_ssm_dtype must be float32, got `{}`",
                self.recurrent_state_dtype
            )));
        }
        if !self.rope.theta.is_finite() || self.rope.theta <= 0.0 {
            return Err(config_error("rope_theta must be finite and positive"));
        }
        if !self.rope.partial_rotary_factor.is_finite()
            || self.rope.partial_rotary_factor <= 0.0
            || self.rope.partial_rotary_factor > 1.0
        {
            return Err(config_error("partial_rotary_factor must be in (0, 1]"));
        }
        let exact_rotary_dim = self.head_dim as f32 * self.rope.partial_rotary_factor;
        if (exact_rotary_dim - exact_rotary_dim.round()).abs() > 1e-5 {
            return Err(config_error(format!(
                "head_dim {} * partial_rotary_factor {} is not integral",
                self.head_dim, self.rope.partial_rotary_factor
            )));
        }
        let section_rotary_dim = 2 * self.rope.mrope_section.iter().sum::<usize>();
        if section_rotary_dim != self.rotary_dim() {
            return Err(config_error(format!(
                "2 * sum(mrope_section) is {section_rotary_dim}, but rotary dimension is {}",
                self.rotary_dim()
            )));
        }

        let ngram_heads = self
            .ngram_size
            .checked_sub(1)
            .and_then(|orders| orders.checked_mul(self.heads_per_ngram))
            .ok_or_else(|| config_error("n-gram head count overflowed"))?;
        if ngram_heads == 0 || !self.ple_embed_dim.is_multiple_of(ngram_heads) {
            return Err(config_error(format!(
                "ple_embed_dim {} is not divisible by n-gram head count {ngram_heads}",
                self.ple_embed_dim
            )));
        }
        let mut previous_layer = None;
        for &layer_id in &self.ple_layer_ids {
            if !(1..=self.n_layers).contains(&layer_id) {
                return Err(config_error(format!(
                    "PLE layer {layer_id} is outside 1..={}",
                    self.n_layers
                )));
            }
            if previous_layer.is_some_and(|previous| layer_id <= previous) {
                return Err(config_error(
                    "ple_layer_ids must be strictly increasing and unique",
                ));
            }
            if self.layer_types[layer_id - 1] != Qwen4ExpLayerType::LinearAttention {
                return Err(config_error(format!(
                    "PLE layer {layer_id} must be a linear-attention layer"
                )));
            }
            previous_layer = Some(layer_id);
        }
        Ok(())
    }
}

fn validate_vision(config: &Qwen4ExpVisionConfig, text_hidden_size: usize) -> Result<()> {
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
            return Err(config_error(format!(
                "vision_config.{name} must be positive"
            )));
        }
    }
    if !config.hidden_size.is_multiple_of(config.num_heads) {
        return Err(config_error(format!(
            "vision hidden_size {} is not divisible by num_heads {}",
            config.hidden_size, config.num_heads
        )));
    }
    if config.out_hidden_size != text_hidden_size {
        return Err(config_error(format!(
            "vision out_hidden_size {} does not match text hidden_size {text_hidden_size}",
            config.out_hidden_size
        )));
    }
    if let Some(index) = config
        .deepstack_visual_indexes
        .iter()
        .copied()
        .find(|index| *index >= config.depth)
    {
        return Err(config_error(format!(
            "deepstack visual layer {index} is outside depth {}",
            config.depth
        )));
    }
    Ok(())
}

fn config_error(message: impl Into<String>) -> Error {
    Error::Other(format!("qwen4-exp config: {}", message.into()))
}

fn required_object<'a>(value: &'a Value, key: &str, context: &str) -> Result<&'a Value> {
    value
        .get(key)
        .filter(|value| value.is_object())
        .ok_or_else(|| config_error(format!("missing object {context}.{key}")))
}

fn required_array<'a>(value: &'a Value, key: &str, context: &str) -> Result<&'a [Value]> {
    value
        .get(key)
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .ok_or_else(|| config_error(format!("missing array {context}.{key}")))
}

fn required_str<'a>(value: &'a Value, key: &str, context: &str) -> Result<&'a str> {
    value
        .get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| config_error(format!("missing string {context}.{key}")))
}

fn required_bool(value: &Value, key: &str, context: &str) -> Result<bool> {
    value
        .get(key)
        .and_then(Value::as_bool)
        .ok_or_else(|| config_error(format!("missing boolean {context}.{key}")))
}

fn optional_bool(value: &Value, key: &str, context: &str) -> Result<Option<bool>> {
    match value.get(key) {
        None => Ok(None),
        Some(value) => value
            .as_bool()
            .map(Some)
            .ok_or_else(|| config_error(format!("{context}.{key} must be a boolean"))),
    }
}

fn optional_u64(value: &Value, key: &str, context: &str) -> Result<Option<u64>> {
    match value.get(key) {
        None => Ok(None),
        Some(value) => value
            .as_u64()
            .map(Some)
            .ok_or_else(|| config_error(format!("{context}.{key} must be an unsigned integer"))),
    }
}

fn required_usize(value: &Value, key: &str, context: &str) -> Result<usize> {
    let value = value
        .get(key)
        .ok_or_else(|| config_error(format!("missing integer {context}.{key}")))?;
    value_to_usize(value, &format!("{context}.{key}"))
}

fn required_u32(value: &Value, key: &str, context: &str) -> Result<u32> {
    let number = value
        .get(key)
        .and_then(Value::as_u64)
        .ok_or_else(|| config_error(format!("missing integer {context}.{key}")))?;
    u32::try_from(number).map_err(|_| config_error(format!("{context}.{key}={number} exceeds u32")))
}

fn required_f32(value: &Value, key: &str, context: &str) -> Result<f32> {
    let number = value
        .get(key)
        .and_then(Value::as_f64)
        .ok_or_else(|| config_error(format!("missing number {context}.{key}")))?;
    if !number.is_finite() || number < -(f32::MAX as f64) || number > f32::MAX as f64 {
        return Err(config_error(format!(
            "{context}.{key} is outside f32 range"
        )));
    }
    Ok(number as f32)
}

fn value_to_usize(value: &Value, path: &str) -> Result<usize> {
    let number = value
        .as_u64()
        .ok_or_else(|| config_error(format!("{path} must be an unsigned integer")))?;
    usize::try_from(number).map_err(|_| config_error(format!("{path} exceeds usize")))
}

fn required_usize_array(value: &Value, key: &str, context: &str) -> Result<Vec<usize>> {
    required_array(value, key, context)?
        .iter()
        .enumerate()
        .map(|(index, value)| value_to_usize(value, &format!("{context}.{key}[{index}]")))
        .collect()
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    pub(crate) const MINI_CONFIG: &str =
        include_str!("../../../../configs/qwen4-exp/synthetic-tiny.json");

    #[test]
    fn parses_released_topology_and_normalizes_qsa_name() {
        let config = Qwen4ExpConfig::from_json_str(MINI_CONFIG).unwrap();
        assert_eq!(config.text.hidden_size, 16);
        assert_eq!(config.text.full_query_width(), 16);
        assert_eq!(config.text.full_kv_width(), 8);
        assert_eq!(config.text.linear_key_width(), 8);
        assert_eq!(config.text.linear_value_width(), 16);
        assert_eq!(config.text.rotary_dim(), 2);
        assert_eq!(config.text.ple_layer_ids, vec![2]);
        assert_eq!(
            config.text.layer_types[3],
            Qwen4ExpLayerType::QwenSparseAttention
        );
    }

    #[test]
    fn rejects_single_residual_stream() {
        let raw = MINI_CONFIG.replacen("\"hc_count\": 4", "\"hc_count\": 1", 1);
        let error = Qwen4ExpConfig::from_json_str(&raw).unwrap_err();
        assert!(error.to_string().contains("hc_count"));
    }

    #[test]
    fn rejects_ple_on_qsa_layer() {
        let raw = MINI_CONFIG.replacen("\"ple_layer_ids\": [2]", "\"ple_layer_ids\": [4]", 1);
        let error = Qwen4ExpConfig::from_json_str(&raw).unwrap_err();
        assert!(error.to_string().contains("PLE"));
        assert!(error.to_string().contains("layer 4"));
    }

    #[test]
    fn rejects_moe_topk_larger_than_expert_count() {
        let raw = MINI_CONFIG.replacen(
            "\"num_experts_per_tok\": 2",
            "\"num_experts_per_tok\": 5",
            1,
        );
        let error = Qwen4ExpConfig::from_json_str(&raw).unwrap_err();
        assert!(error.to_string().contains("num_experts_per_tok"));
    }

    #[test]
    fn rejects_schedule_that_disagrees_with_interval() {
        let mut raw: Value = serde_json::from_str(MINI_CONFIG).unwrap();
        raw["text_config"]["layer_types"][0] = "full_attention".into();
        let error = Qwen4ExpConfig::from_json_str(&raw.to_string()).unwrap_err();
        assert!(error.to_string().contains("layer 0"));
        assert!(error.to_string().contains("full_attention_interval"));
    }

    #[test]
    fn rejects_incomplete_qsa_contract() {
        let raw = MINI_CONFIG.replacen("\"indexer_kv_heads\": 1,", "", 1);
        let error = Qwen4ExpConfig::from_json_str(&raw).unwrap_err();
        assert!(error.to_string().contains("indexer_kv_heads"));
    }
}
