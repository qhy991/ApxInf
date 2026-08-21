//! Native Qwen2.5-Omni Thinker orchestration.

use std::collections::HashMap;
use std::sync::Arc;

use apxinf_core::{Backend, Device, Error, KvCache, Result, Tensor};

use crate::llm_trait::{LlmCapabilities, LlmInput, LlmTrait};

use super::audio::{self, Qwen25OmniAudioWeights};
use super::config::Qwen25OmniConfig;
use super::vision::{self, Qwen25OmniVisionWeights};
use super::weights::Qwen25OmniTextWeights;

pub struct GeneralQwen25Omni {
    config: Qwen25OmniConfig,
    text: Qwen25OmniTextWeights,
    vision: Qwen25OmniVisionWeights,
    audio: Qwen25OmniAudioWeights,
    backend: Arc<dyn Backend>,
    kv: Box<dyn KvCache>,
    rope_delta: i64,
}

impl GeneralQwen25Omni {
    pub(crate) fn from_selected_weights(
        config: Qwen25OmniConfig,
        mut tensors: HashMap<String, Tensor>,
        backend: Arc<dyn Backend>,
    ) -> Result<Self> {
        let text = Qwen25OmniTextWeights::from_map(&config, &mut tensors)?.to_device(&*backend)?;
        let vision =
            Qwen25OmniVisionWeights::from_map(&config, &mut tensors)?.to_device(&*backend)?;
        let audio =
            Qwen25OmniAudioWeights::from_map(&config, &mut tensors)?.to_device(&*backend)?;
        if !tensors.is_empty() {
            let mut names = tensors.keys().cloned().collect::<Vec<_>>();
            names.sort();
            return Err(Error::Other(format!(
                "qwen2.5-omni selected loader left unowned tensors: {}",
                names.into_iter().take(8).collect::<Vec<_>>().join(", ")
            )));
        }
        let kv = backend.create_kv_cache(
            config.text.n_layers,
            config.text.n_kv_heads,
            config.text.head_dim,
            config.text.max_position_embeddings,
        );
        Ok(Self {
            config,
            text,
            vision,
            audio,
            backend,
            kv,
            rope_delta: 0,
        })
    }

    pub fn config(&self) -> &Qwen25OmniConfig {
        &self.config
    }

    fn forward_inner(&mut self, token_ids: &[u32], start_pos: u32) -> Result<Tensor> {
        if token_ids.is_empty() {
            return Err(Error::Other("qwen2.5-omni forward: empty token_ids".into()));
        }
        let expected_start = self.kv.seq_len();
        if start_pos as usize != expected_start {
            return Err(Error::Other(format!(
                "qwen2.5-omni cache position mismatch: start_pos={start_pos}, cache={expected_start}"
            )));
        }
        if start_pos as usize + token_ids.len() > self.config.text.max_position_embeddings {
            return Err(Error::Other(
                "qwen2.5-omni forward exceeds context capacity".into(),
            ));
        }
        reject_video(token_ids, self.config.video_token_id)?;
        let mut hidden = self
            .backend
            .embedding(&self.text.token_embedding, token_ids)?;
        let positions = linear_positions(token_ids.len(), start_pos, self.rope_delta)?;
        for index in 0..self.config.text.n_layers {
            hidden = self.forward_layer(&hidden, index, &positions)?;
        }
        self.kv.advance(token_ids.len());
        self.logits_last_row(&hidden)
    }

    fn prefill_inner(&mut self, input: LlmInput<'_>) -> Result<Tensor> {
        if self.kv.seq_len() != 0 {
            return Err(Error::Other(
                "qwen2.5-omni prefill requires reset state".into(),
            ));
        }
        if input.token_ids.is_empty() {
            return Err(Error::Other("qwen2.5-omni prefill: empty token_ids".into()));
        }
        if input.token_ids.len() > self.config.text.max_position_embeddings {
            return Err(Error::Other(
                "qwen2.5-omni prompt exceeds context capacity".into(),
            ));
        }
        reject_unsupported_media_combination(input)?;
        reject_video(input.token_ids, self.config.video_token_id)?;
        let image_positions = if let Some(image) = input.image {
            vision::validate_input(&self.config, image.pixel_values, image.grid_thw)?;
            let expected =
                vision::merged_token_count(image.grid_thw, self.config.vision.spatial_merge_size)?;
            let positions = token_positions(input.token_ids, self.config.image_token_id);
            if positions.len() != expected {
                return Err(Error::Other(format!(
                    "qwen2.5-omni image placeholders {} != encoded tokens {expected}",
                    positions.len()
                )));
            }
            Some(positions)
        } else {
            if input.token_ids.contains(&self.config.image_token_id) {
                return Err(Error::Other(
                    "qwen2.5-omni image placeholders require image input".into(),
                ));
            }
            None
        };
        let audio_positions = if let Some(audio_input) = input.audio {
            let expected = audio::validate_input(&self.config, audio_input)?;
            let positions = token_positions(input.token_ids, self.config.audio_token_id);
            if positions.len() != expected {
                return Err(Error::Other(format!(
                    "qwen2.5-omni audio placeholders {} != encoded tokens {expected}",
                    positions.len()
                )));
            }
            let boundaries = audio_boundary_positions(
                input.token_ids,
                self.config.audio_start_token_id,
                self.config.audio_token_id,
                self.config.audio_end_token_id,
                expected,
            )?;
            Some((positions, boundaries))
        } else {
            if input.token_ids.contains(&self.config.audio_token_id)
                || input.token_ids.contains(&self.config.audio_start_token_id)
                || input.token_ids.contains(&self.config.audio_end_token_id)
            {
                return Err(Error::Other(
                    "qwen2.5-omni audio markers require audio input".into(),
                ));
            }
            None
        };

        // Every processor shape, placeholder count, and modality marker is
        // validated above so malformed media fails before any backend work.
        let mut hidden = self
            .backend
            .embedding(&self.text.token_embedding, input.token_ids)?;

        if let (Some(image), Some(positions)) = (input.image, image_positions.as_ref()) {
            let encoded = vision::forward(
                &self.config,
                &self.vision,
                &*self.backend,
                image.pixel_values,
                image.grid_thw,
            )?;
            hidden = scatter_replace(&hidden, &positions, &encoded, &*self.backend)?;
        }

        if let (Some(audio_input), Some((positions, boundaries))) =
            (input.audio, audio_positions.as_ref())
        {
            let encoded = audio::forward(&self.config, &self.audio, &*self.backend, audio_input)?;
            hidden = scatter_replace(&hidden, &positions, &encoded, &*self.backend)?;
            hidden = scatter_replace(
                &hidden,
                boundaries,
                self.audio.boundary_embeddings(),
                &*self.backend,
            )?;
        }

        let positions = multimodal_positions(&self.config, input)?;
        let max_position = positions
            .chunks_exact(3)
            .map(|position| position[0].max(position[1]).max(position[2]))
            .max()
            .unwrap_or(0) as i64;
        self.rope_delta = max_position + 1 - input.token_ids.len() as i64;
        for index in 0..self.config.text.n_layers {
            hidden = self.forward_layer(&hidden, index, &positions)?;
        }
        self.kv.advance(input.token_ids.len());
        self.logits_last_row(&hidden)
    }

    fn forward_layer(
        &mut self,
        hidden: &Tensor,
        index: usize,
        positions: &[u32],
    ) -> Result<Tensor> {
        let text = &self.config.text;
        let sequence = hidden.shape().dims()[0];
        let layer = &self.text.layers[index];
        let normalized = self
            .backend
            .rms_norm(hidden, &layer.attn_norm, text.rms_norm_eps)?;
        let q = self
            .backend
            .add_bias(&self.backend.matmul(&normalized, &layer.wq)?, &layer.bq)?;
        let k = self
            .backend
            .add_bias(&self.backend.matmul(&normalized, &layer.wk)?, &layer.bk)?;
        let v = self
            .backend
            .add_bias(&self.backend.matmul(&normalized, &layer.wv)?, &layer.bv)?;
        let q = q.reshape(vec![sequence, text.n_heads, text.head_dim])?;
        let k = k.reshape(vec![sequence, text.n_kv_heads, text.head_dim])?;
        let v = v.reshape(vec![sequence, text.n_kv_heads, text.head_dim])?;
        let q = self.backend.rope_tmrope(
            &q,
            text.n_heads,
            text.head_dim,
            text.rope_theta,
            text.mrope_section,
            positions,
        )?;
        let k = self.backend.rope_tmrope(
            &k,
            text.n_kv_heads,
            text.head_dim,
            text.rope_theta,
            text.mrope_section,
            positions,
        )?;
        self.backend
            .kv_append(&mut *self.kv, index, &k, &v, sequence)?;
        let kv_len = self.kv.seq_len() + sequence;
        let attention = if sequence == 1 {
            self.backend.sdpa_decode(
                &q,
                &mut *self.kv,
                index,
                text.n_heads,
                text.n_kv_heads,
                text.head_dim,
                kv_len,
                text.max_position_embeddings,
            )?
        } else {
            self.backend.sdpa_prefill(
                &q,
                &mut *self.kv,
                index,
                text.n_heads,
                text.n_kv_heads,
                text.head_dim,
                kv_len,
                text.max_position_embeddings,
            )?
        };
        let attention = self.backend.matmul(&attention, &layer.wo)?;
        let residual = self.backend.add(hidden, &attention)?;
        let normalized = self
            .backend
            .rms_norm(&residual, &layer.ffn_norm, text.rms_norm_eps)?;
        let gate = self
            .backend
            .silu(&self.backend.matmul(&normalized, &layer.w_gate)?)?;
        let up = self.backend.matmul(&normalized, &layer.w_up)?;
        let mlp = self
            .backend
            .matmul(&self.backend.mul(&gate, &up)?, &layer.w_down)?;
        self.backend.add(&residual, &mlp)
    }

    fn logits_last_row(&self, hidden: &Tensor) -> Result<Tensor> {
        let hidden = self.backend.rms_norm(
            hidden,
            &self.text.output_norm,
            self.config.text.rms_norm_eps,
        )?;
        // Avoid materializing `[prompt, vocab]`: only the last position owns
        // the next-token distribution. The small hidden row is copied, then
        // returned to the model device for the separate native LM head.
        let cpu = self.backend.to_cpu(&hidden)?;
        let rows = cpu.shape().dims()[0];
        let width = self.config.text.hidden_size;
        let values = cpu.to_f32_vec()?;
        let last = &values[(rows - 1) * width..rows * width];
        let row = match cpu.dtype() {
            apxinf_core::DType::BF16 => Tensor::from_bf16(
                vec![1, width],
                &last
                    .iter()
                    .copied()
                    .map(half::bf16::from_f32)
                    .collect::<Vec<_>>(),
            )?,
            apxinf_core::DType::F32 => Tensor::from_f32(vec![1, width], last)?,
            dtype => {
                return Err(Error::Other(format!(
                    "qwen2.5-omni final hidden dtype {dtype} is unsupported"
                )))
            }
        };
        let row = self.backend.to_device(&row)?;
        let logits = self.backend.matmul(&row, &self.text.lm_head)?;
        self.backend.synchronize()?;
        let logits = self.backend.to_cpu(&logits)?;
        Tensor::from_f32(vec![1, self.config.text.vocab_size], &logits.to_f32_vec()?)
    }

    fn clear_state(&mut self) {
        let _ = self.kv.clear();
        self.rope_delta = 0;
    }
}

impl LlmTrait for GeneralQwen25Omni {
    fn load(
        _config: apxinf_loader::ModelConfig,
        _weights: HashMap<String, Tensor>,
        _device: Device,
    ) -> Result<Self>
    where
        Self: Sized,
    {
        Err(Error::Other(
            "GeneralQwen25Omni owns a nested config; load through AutoModel".into(),
        ))
    }

    fn forward(&mut self, token_ids: &[u32], start_pos: u32) -> Result<Tensor> {
        let result = self.forward_inner(token_ids, start_pos);
        if result.is_err() {
            self.clear_state();
        }
        result
    }

    fn capabilities(&self) -> LlmCapabilities {
        LlmCapabilities::OMNI
    }

    fn prefill(&mut self, input: LlmInput<'_>) -> Result<Tensor> {
        let result = self.prefill_inner(input);
        if result.is_err() {
            self.clear_state();
        }
        result
    }

    fn reset(&mut self) {
        self.clear_state();
    }

    fn vocab_size(&self) -> usize {
        self.config.text.vocab_size
    }

    fn max_context_len(&self) -> Option<usize> {
        Some(self.config.text.max_position_embeddings)
    }

    fn max_new_tokens_limit(&self) -> Option<usize> {
        Some(128)
    }
}

fn reject_video(token_ids: &[u32], video_token_id: u32) -> Result<()> {
    if token_ids.contains(&video_token_id) {
        return Err(Error::Other(
            "qwen2.5-omni video input is outside the deployed capability".into(),
        ));
    }
    Ok(())
}

fn reject_unsupported_media_combination(input: LlmInput<'_>) -> Result<()> {
    if input.image.is_some() && input.audio.is_some() {
        return Err(Error::Other(
            "qwen2.5-omni simultaneous image and audio input is outside the deployed capability"
                .into(),
        ));
    }
    Ok(())
}

fn token_positions(token_ids: &[u32], token: u32) -> Vec<usize> {
    token_ids
        .iter()
        .enumerate()
        .filter_map(|(index, value)| (*value == token).then_some(index))
        .collect()
}

fn audio_boundary_positions(
    token_ids: &[u32],
    start_token: u32,
    audio_token: u32,
    end_token: u32,
    audio_count: usize,
) -> Result<Vec<usize>> {
    let starts = token_positions(token_ids, start_token);
    let ends = token_positions(token_ids, end_token);
    if starts.len() != 1 || ends.len() != 1 {
        return Err(Error::Other(format!(
            "qwen2.5-omni one audio clip requires exactly one start/end marker, got {}/{}",
            starts.len(),
            ends.len()
        )));
    }
    let start = starts[0];
    let end = ends[0];
    if end != start + audio_count + 1
        || token_ids[start + 1..end]
            .iter()
            .any(|token| *token != audio_token)
    {
        return Err(Error::Other(
            "qwen2.5-omni audio markers must enclose one contiguous placeholder run".into(),
        ));
    }
    Ok(vec![start, end])
}

fn linear_positions(length: usize, start: u32, delta: i64) -> Result<Vec<u32>> {
    let first = i64::from(start) + delta;
    if first < 0 {
        return Err(Error::Other(
            "qwen2.5-omni negative TMRoPE decode position".into(),
        ));
    }
    let mut positions = Vec::with_capacity(length * 3);
    for offset in 0..length {
        let position = u32::try_from(first + offset as i64)
            .map_err(|_| Error::Other("qwen2.5-omni TMRoPE position overflow".into()))?;
        positions.extend_from_slice(&[position, position, position]);
    }
    Ok(positions)
}

fn multimodal_positions(config: &Qwen25OmniConfig, input: LlmInput<'_>) -> Result<Vec<u32>> {
    let image_counts = input
        .image
        .map(|image| {
            image
                .grid_thw
                .iter()
                .map(|&[time, height, width]| {
                    (time as usize)
                        * (height as usize / config.vision.spatial_merge_size)
                        * (width as usize / config.vision.spatial_merge_size)
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let audio_counts = input
        .audio
        .map(|audio| {
            audio
                .token_counts
                .iter()
                .map(|count| *count as usize)
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let grids = input.image.map(|image| image.grid_thw).unwrap_or(&[]);
    let mut output = Vec::with_capacity(input.token_ids.len() * 3);
    let mut index = 0;
    let mut image = 0;
    let mut audio = 0;
    let mut next = 0_u32;
    while index < input.token_ids.len() {
        if input.token_ids[index] == config.image_token_id {
            let count = *image_counts
                .get(image)
                .ok_or_else(|| Error::Other("qwen2.5-omni image placeholder has no grid".into()))?;
            if input.token_ids[index..]
                .iter()
                .take(count)
                .any(|token| *token != config.image_token_id)
            {
                return Err(Error::Other(
                    "qwen2.5-omni image placeholders are not contiguous".into(),
                ));
            }
            let [time, height, width] = grids[image];
            let height = height / config.vision.spatial_merge_size as u32;
            let width = width / config.vision.spatial_merge_size as u32;
            for temporal in 0..time {
                for row in 0..height {
                    for col in 0..width {
                        output.extend_from_slice(&[next + temporal, next + row, next + col]);
                    }
                }
            }
            next += time.max(height).max(width);
            index += count;
            image += 1;
        } else if input.token_ids[index] == config.audio_token_id {
            let count = *audio_counts.get(audio).ok_or_else(|| {
                Error::Other("qwen2.5-omni audio placeholder has no feature group".into())
            })?;
            if input.token_ids[index..]
                .iter()
                .take(count)
                .any(|token| *token != config.audio_token_id)
            {
                return Err(Error::Other(
                    "qwen2.5-omni audio placeholders are not contiguous".into(),
                ));
            }
            for temporal in 0..count as u32 {
                output.extend_from_slice(&[next + temporal, next, next]);
            }
            next += count as u32;
            index += count;
            audio += 1;
        } else {
            output.extend_from_slice(&[next, next, next]);
            next += 1;
            index += 1;
        }
    }
    if image != image_counts.len() || audio != audio_counts.len() {
        return Err(Error::Other(
            "qwen2.5-omni unused media group after TMRoPE construction".into(),
        ));
    }
    if output.len() != input.token_ids.len() * 3 {
        return Err(Error::Other(
            "qwen2.5-omni TMRoPE position length drift".into(),
        ));
    }
    Ok(output)
}

fn scatter_replace(
    hidden: &Tensor,
    positions: &[usize],
    replacement: &Tensor,
    backend: &dyn Backend,
) -> Result<Tensor> {
    let hidden_cpu = backend.to_cpu(hidden)?;
    let replacement_cpu = backend.to_cpu(replacement)?;
    let width = *hidden_cpu
        .shape()
        .dims()
        .last()
        .ok_or_else(|| Error::Other("qwen2.5-omni scatter hidden is scalar".into()))?;
    if replacement_cpu.shape().dims() != [positions.len(), width] {
        return Err(Error::Other(format!(
            "qwen2.5-omni replacement shape {:?}, expected [{}, {width}]",
            replacement_cpu.shape().dims(),
            positions.len()
        )));
    }
    let mut values = hidden_cpu.to_f32_vec()?;
    let replacement = replacement_cpu.to_f32_vec()?;
    for (source, position) in positions.iter().enumerate() {
        values[*position * width..(*position + 1) * width]
            .copy_from_slice(&replacement[source * width..(source + 1) * width]);
    }
    let output = match hidden_cpu.dtype() {
        apxinf_core::DType::BF16 => Tensor::from_bf16(
            hidden_cpu.shape().dims().to_vec(),
            &values
                .into_iter()
                .map(half::bf16::from_f32)
                .collect::<Vec<_>>(),
        )?,
        apxinf_core::DType::F32 => Tensor::from_f32(hidden_cpu.shape().dims().to_vec(), &values)?,
        dtype => {
            return Err(Error::Other(format!(
                "qwen2.5-omni scatter does not support {dtype}"
            )))
        }
    };
    backend.to_device(&output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm_trait::{AudioInput, ImageInput};

    fn config() -> Qwen25OmniConfig {
        let raw = include_str!("../../tests/data/qwen25_omni_config_minimal.json");
        let mut config = Qwen25OmniConfig::from_json_str(raw).unwrap();
        config.processor = super::super::config::Qwen25OmniProcessorConfig {
            sampling_rate: 16000,
            n_fft: 400,
            hop_length: 160,
            feature_size: 128,
        };
        config
    }

    #[test]
    fn rejects_combined_media_and_video() {
        let config = config();
        let pixels = Tensor::from_f32(vec![16, 1176], &vec![0.0; 16 * 1176]).unwrap();
        let grids = [[1, 4, 4]];
        let features = Tensor::from_f32(vec![4, 128], &vec![0.0; 4 * 128]).unwrap();
        let mask = Tensor::from_f32(vec![4], &[1.0; 4]).unwrap();
        let lengths = [4];
        let audio_counts = [2];
        let tokens = [1, 151655, 151655, 151655, 151655, 2, 151646, 151646, 3];
        let input = LlmInput::with_media(
            &tokens,
            Some(ImageInput::new(&pixels, &grids)),
            Some(AudioInput::new(&features, &mask, &lengths, &audio_counts)),
        );
        assert!(reject_unsupported_media_combination(input).is_err());
        assert!(reject_video(&[config.video_token_id], config.video_token_id).is_err());
    }

    #[test]
    fn validates_audio_boundaries_and_multimodal_positions() {
        let config = config();
        let audio_tokens = [10, 151647, 151646, 151646, 151646, 151648, 11];
        assert_eq!(
            audio_boundary_positions(&audio_tokens, 151647, 151646, 151648, 3).unwrap(),
            [1, 5]
        );
        assert!(audio_boundary_positions(
            &[10, 151647, 151646, 12, 151646, 151648],
            151647,
            151646,
            151648,
            3
        )
        .is_err());

        let pixels = Tensor::from_f32(vec![16, 1176], &vec![0.0; 16 * 1176]).unwrap();
        let image_grid = [[1, 4, 4]];
        let image_tokens = [10, 151655, 151655, 151655, 151655, 11];
        let image_positions = multimodal_positions(
            &config,
            LlmInput::with_image(&image_tokens, ImageInput::new(&pixels, &image_grid)),
        )
        .unwrap();
        assert_eq!(
            image_positions,
            [0, 0, 0, 1, 1, 1, 1, 1, 2, 1, 2, 1, 1, 2, 2, 3, 3, 3]
        );

        let features = Tensor::from_f32(vec![7, 128], &vec![0.0; 7 * 128]).unwrap();
        let mask = Tensor::from_f32(vec![7], &[1.0; 7]).unwrap();
        let lengths = [7];
        let counts = [2];
        let positioned_audio = [10, 151647, 151646, 151646, 151648, 11];
        let audio_positions = multimodal_positions(
            &config,
            LlmInput::with_audio(
                &positioned_audio,
                AudioInput::new(&features, &mask, &lengths, &counts),
            ),
        )
        .unwrap();
        assert_eq!(
            audio_positions,
            [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 2, 2, 4, 4, 4, 5, 5, 5]
        );
    }
}
