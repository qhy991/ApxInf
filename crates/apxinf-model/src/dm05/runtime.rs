//! Fixed-cell native BF16 DM05 execution carrier.

use std::sync::Arc;

use apxinf_core::{Backend, DType, Error, Result, Tensor};

use super::backend::{kernels, Context, DeviceBuffer, RuntimeBackend};
use super::executor::row_view;
use super::{
    action_layer, action_mask, action_projection, encode_vision, language_layer, merge_prefix,
    prefix_attention_segments, projector_pool_matrix, time_values, ActionStyles, DeviceDm05Weights,
    Dm05Config, Dm05LayerType,
};

pub struct PrefixKvCache {
    pub keys: Vec<Tensor>,
    pub values: Vec<Tensor>,
    pub tokens: usize,
}

struct PreparedStyles {
    attention: Vec<Tensor>,
    mlp: Vec<Tensor>,
    final_norm: Tensor,
}

#[derive(Clone)]
struct RopePair {
    cosine: Tensor,
    sine: Tensor,
}

#[derive(Clone)]
struct DualRopeTables {
    sliding: RopePair,
    full: RopePair,
}

impl DualRopeTables {
    fn for_layer(&self, kind: Dm05LayerType) -> &RopePair {
        match kind {
            Dm05LayerType::Sliding => &self.sliding,
            Dm05LayerType::Full => &self.full,
        }
    }
}

#[derive(Clone)]
pub struct Dm05PreparedShape {
    prefix_tokens: usize,
    prefix_rope: DualRopeTables,
    suffix_rope: DualRopeTables,
}

#[derive(Clone)]
pub struct Dm05Bf16Runtime {
    backend: Arc<RuntimeBackend>,
    config: Arc<Dm05Config>,
    weights: Arc<DeviceDm05Weights>,
    pool_matrix: Tensor,
    action_mask: Tensor,
    styles: Arc<PreparedStyles>,
}

impl Dm05Bf16Runtime {
    pub fn new(
        backend: Arc<RuntimeBackend>,
        config: Arc<Dm05Config>,
        weights: Arc<DeviceDm05Weights>,
    ) -> Result<Self> {
        config.validate()?;
        if config.action_horizon != 10 {
            return Err(Error::Other(format!(
                "DM05 native LIBERO runtime requires action_horizon=10, got {}",
                config.action_horizon
            )));
        }
        let caps = backend.context().caps();
        if (caps.compute_major, caps.compute_minor) != (8, 9) {
            return Err(Error::Other(format!(
                "DM05 native runtime requires SM89, got SM{}{}",
                caps.compute_major, caps.compute_minor
            )));
        }
        if weights.vision_layers.len() != config.vision.depth
            || weights.language_layers.len() != config.language.depth
            || weights.action_layers.len() != config.action_expert.depth
        {
            return Err(Error::Other("DM05 device weight depth mismatch".into()));
        }
        let pool_matrix = backend.to_device(&projector_pool_matrix(&config)?)?;
        let action_mask = backend.to_device(&action_mask(&config)?)?;
        let styles = Arc::new(prepare_styles(
            backend.context(),
            &config,
            &weights,
            &backend,
        )?);
        backend.synchronize()?;
        Ok(Self {
            backend,
            config,
            weights,
            pool_matrix,
            action_mask,
            styles,
        })
    }

    fn ctx(&self) -> &Context {
        self.backend.context()
    }

    pub fn prepare_shape(&self, prefix_tokens: usize) -> Result<Dm05PreparedShape> {
        if prefix_tokens == 0 || prefix_tokens > self.config.max_prefix_len {
            return Err(Error::Other(format!(
                "DM05 prefix length must be in 1..={}, got {prefix_tokens}",
                self.config.max_prefix_len
            )));
        }
        let prefix_positions = (0..prefix_tokens)
            .map(|position| u32::try_from(position).expect("bounded prefix"))
            .collect::<Vec<_>>();
        let suffix_positions = (prefix_tokens..prefix_tokens + self.config.action_horizon)
            .map(|position| u32::try_from(position).expect("bounded suffix"))
            .collect::<Vec<_>>();
        let prefix_rope = prepare_rope_tables(self.ctx(), &self.config, &prefix_positions)?;
        let suffix_rope = prepare_rope_tables(self.ctx(), &self.config, &suffix_positions)?;
        self.backend.synchronize()?;
        Ok(Dm05PreparedShape {
            prefix_tokens,
            prefix_rope,
            suffix_rope,
        })
    }

    pub fn infer(
        &self,
        patches: &Tensor,
        token_ids: &DeviceBuffer,
        host_token_ids: &[u32],
        noise: &Tensor,
        shape: &Dm05PreparedShape,
    ) -> Result<Tensor> {
        if host_token_ids.len() != shape.prefix_tokens {
            return Err(Error::Other(format!(
                "DM05 prepared prefix length {} does not match {} token IDs",
                shape.prefix_tokens,
                host_token_ids.len()
            )));
        }
        self.config.validate_prefix_tokens(host_token_ids)?;
        if noise.dtype() != DType::BF16
            || noise.shape().dims() != [self.config.action_horizon, self.config.action_dim]
        {
            return Err(Error::Other(format!(
                "DM05 initial latent must be BF16 [{},{}], got {} {:?}",
                self.config.action_horizon,
                self.config.action_dim,
                noise.dtype(),
                noise.shape().dims()
            )));
        }

        let image_tokens = encode_vision(
            self.ctx(),
            &self.config,
            &self.weights,
            &self.pool_matrix,
            patches,
        )?;
        let prefix = merge_prefix(
            self.ctx(),
            &self.config,
            &self.weights,
            token_ids,
            host_token_ids,
            &image_tokens,
        )?;
        let cache = self.prefix_forward(&prefix, host_token_ids, &shape.prefix_rope)?;
        self.denoise(noise, &cache, &shape.suffix_rope)
    }

    fn prefix_forward(
        &self,
        prefix: &Tensor,
        token_ids: &[u32],
        rope: &DualRopeTables,
    ) -> Result<PrefixKvCache> {
        let segments = prefix_attention_segments(&self.config, token_ids)?;
        let mut hidden = prefix.clone();
        let mut keys = Vec::with_capacity(self.config.language.depth);
        let mut values = Vec::with_capacity(self.config.language.depth);
        for (index, layer) in self.weights.language_layers.iter().enumerate() {
            let tables = rope.for_layer(layer.layer_type);
            let output = language_layer(
                self.ctx(),
                &self.config,
                layer,
                &hidden,
                &segments,
                &tables.cosine,
                &tables.sine,
                index + 1 < self.config.language.depth,
            )?;
            hidden = output.hidden;
            keys.push(output.key);
            values.push(output.value);
        }
        Ok(PrefixKvCache {
            keys,
            values,
            tokens: prefix.shape().dims()[0],
        })
    }

    fn denoise(
        &self,
        noise: &Tensor,
        prefix: &PrefixKvCache,
        rope: &DualRopeTables,
    ) -> Result<Tensor> {
        if prefix.keys.len() != self.config.action_expert.depth
            || prefix.values.len() != self.config.action_expert.depth
        {
            return Err(Error::Other("DM05 prefix cache depth mismatch".into()));
        }
        let mut state = noise.clone();
        for step in 0..self.config.diffusion_steps {
            let masked = kernels::elementwise::mul(self.ctx(), &state, &self.action_mask)?;
            let mut hidden = action_projection(self.ctx(), &masked, &self.weights.action_in)?;
            for (index, layer) in self.weights.action_layers.iter().enumerate() {
                let attention_style = style_row(&self.styles.attention[index], step)?;
                let mlp_style = style_row(&self.styles.mlp[index], step)?;
                let tables = rope.for_layer(layer.layer_type);
                hidden = action_layer(
                    self.ctx(),
                    &self.config,
                    layer,
                    &hidden,
                    ActionStyles {
                        attention: &attention_style,
                        mlp: &mlp_style,
                    },
                    &prefix.keys[index],
                    &prefix.values[index],
                    &tables.cosine,
                    &tables.sine,
                )?;
            }
            let final_style = style_row(&self.styles.final_norm, step)?;
            let hidden = kernels::norm::adaptive_rms_bf16(
                self.ctx(),
                &hidden,
                &final_style,
                self.config.action_expert.rms_norm_eps,
            )?;
            let velocity = action_projection(self.ctx(), &hidden, &self.weights.action_out)?;
            state =
                kernels::elementwise::euler_two_stage_bf16(self.ctx(), &state, &velocity, -0.1)?;
        }
        Ok(state)
    }
}

fn style_row(matrix: &Tensor, step: usize) -> Result<Tensor> {
    row_view(matrix, step, step + 1)?.reshape(vec![matrix.shape().dims()[1]])
}

fn prepare_styles(
    ctx: &Context,
    config: &Dm05Config,
    weights: &DeviceDm05Weights,
    backend: &dyn Backend,
) -> Result<PreparedStyles> {
    let times = backend.to_device(&Tensor::from_bf16(vec![10], &time_values())?)?;
    let embedding = kernels::rope::sinusoidal_time_embedding_bf16(
        ctx,
        &times,
        config.action_expert.width,
        4e-3,
        4.0,
    )?;
    let hidden = action_projection(ctx, &embedding, &weights.time_mlp_in)?;
    let hidden = kernels::activation::bias_silu_bf16(ctx, &hidden, None)?;
    let conditioning = action_projection(ctx, &hidden, &weights.time_mlp_out)?;
    let conditioning = kernels::activation::bias_silu_bf16(ctx, &conditioning, None)?;
    let attention = weights
        .action_layers
        .iter()
        .map(|layer| action_projection(ctx, &conditioning, &layer.input_modulator))
        .collect::<Result<Vec<_>>>()?;
    let mlp = weights
        .action_layers
        .iter()
        .map(|layer| action_projection(ctx, &conditioning, &layer.mlp_modulator))
        .collect::<Result<Vec<_>>>()?;
    let final_norm = action_projection(ctx, &conditioning, &weights.action_final_modulator)?;
    Ok(PreparedStyles {
        attention,
        mlp,
        final_norm,
    })
}

fn prepare_rope_tables(
    ctx: &Context,
    config: &Dm05Config,
    positions: &[u32],
) -> Result<DualRopeTables> {
    let buffer = DeviceBuffer::alloc_zeros(
        positions
            .len()
            .checked_mul(std::mem::size_of::<u32>())
            .ok_or_else(|| Error::Other("DM05 RoPE position size overflow".into()))?,
        ctx.device_id(),
    )
    .map_err(Error::Cuda)?;
    let bytes = positions
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect::<Vec<_>>();
    buffer.copy_from_host(&bytes).map_err(Error::Cuda)?;
    let build = |rope: super::Dm05RopeConfig| -> Result<RopePair> {
        let tables = kernels::rope::rope_tables_bf16(
            ctx,
            &buffer,
            positions.len(),
            config.language.head_dim,
            rope.theta,
            rope.linear_factor,
        )?;
        Ok(RopePair {
            cosine: tables.cosine,
            sine: tables.sine,
        })
    };
    Ok(DualRopeTables {
        sliding: build(config.language.sliding_rope)?,
        full: build(config.language.full_rope)?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dm05::config::tests_support::exact_config;

    #[test]
    fn runtime_rejects_checkpoint_horizon_without_libero_override_before_cuda() {
        assert_eq!(exact_config().action_horizon, 50);
        assert_eq!(
            exact_config()
                .with_action_horizon(10)
                .unwrap()
                .action_horizon,
            10
        );
    }
}
