//! CPU reference selector for Qwen Sparse Attention compressed blocks.

use apxinf_core::{Error, Result};
use rayon::prelude::*;

#[derive(Clone, Debug)]
pub struct Qwen4ExpQsaSelector {
    query_heads: usize,
    head_dim: usize,
    token_budget: usize,
    compress_ratio: usize,
    rotary_dim: usize,
    theta: f32,
    rms_norm_eps: f32,
}

impl Qwen4ExpQsaSelector {
    pub fn new(
        query_heads: usize,
        head_dim: usize,
        token_budget: usize,
        compress_ratio: usize,
        rotary_dim: usize,
        theta: f32,
        rms_norm_eps: f32,
    ) -> Result<Self> {
        if query_heads == 0
            || head_dim == 0
            || token_budget == 0
            || compress_ratio == 0
            || !token_budget.is_multiple_of(compress_ratio)
            || rotary_dim > head_dim
            || !rotary_dim.is_multiple_of(2)
            || !theta.is_finite()
            || theta <= 0.0
            || !rms_norm_eps.is_finite()
            || rms_norm_eps <= 0.0
        {
            return Err(Error::Other("invalid QSA selector dimensions".into()));
        }
        Ok(Self {
            query_heads,
            head_dim,
            token_budget,
            compress_ratio,
            rotary_dim,
            theta,
            rms_norm_eps,
        })
    }

    pub fn select_unit_norm(
        &self,
        query: &[f32],
        raw_keys: &[f32],
        visible_tokens: usize,
    ) -> Result<Vec<usize>> {
        self.select(
            query,
            raw_keys,
            visible_tokens,
            &vec![0.0; self.head_dim],
            &vec![0.0; self.head_dim],
        )
    }

    /// Select the same causal token mask as the official eager QSA indexer.
    /// Norm weights use Qwen's zero-centred convention (`scale = 1 + weight`).
    pub fn select(
        &self,
        query: &[f32],
        raw_keys: &[f32],
        visible_tokens: usize,
        query_norm_weight: &[f32],
        key_norm_weight: &[f32],
    ) -> Result<Vec<usize>> {
        let positions = (0..visible_tokens)
            .map(|position| [position as u32; 3])
            .collect::<Vec<_>>();
        self.select_with_positions(
            query,
            raw_keys,
            &positions,
            [self.rotary_dim / 2, 0, 0],
            query_norm_weight,
            key_norm_weight,
        )
    }

    pub fn select_with_positions(
        &self,
        query: &[f32],
        raw_keys: &[f32],
        positions: &[[u32; 3]],
        mrope_sections: [usize; 3],
        query_norm_weight: &[f32],
        key_norm_weight: &[f32],
    ) -> Result<Vec<usize>> {
        let visible_tokens = positions.len();
        let query_len = self
            .query_heads
            .checked_mul(self.head_dim)
            .ok_or_else(|| Error::Other("QSA query dimensions overflow".into()))?;
        if query.len() != query_len {
            return Err(Error::Other(format!(
                "QSA query has {} elements, expected {query_len}",
                query.len()
            )));
        }
        let raw_key_len = visible_tokens
            .checked_mul(self.head_dim)
            .ok_or_else(|| Error::Other("QSA raw-key dimensions overflow".into()))?;
        if raw_keys.len() != raw_key_len {
            return Err(Error::Other(format!(
                "QSA raw_keys has {} elements, expected {raw_key_len}",
                raw_keys.len()
            )));
        }
        if visible_tokens == 0 {
            return Err(Error::Other(
                "QSA requires at least the current visible token".into(),
            ));
        }
        if mrope_sections.iter().sum::<usize>() != self.rotary_dim / 2 {
            return Err(Error::Other(format!(
                "QSA mRoPE sections {mrope_sections:?} do not cover rotary pairs"
            )));
        }
        if query_norm_weight.len() != self.head_dim || key_norm_weight.len() != self.head_dim {
            return Err(Error::Other(format!(
                "QSA norm weights must both have {} elements",
                self.head_dim
            )));
        }
        if query
            .iter()
            .chain(raw_keys)
            .chain(query_norm_weight)
            .chain(key_norm_weight)
            .any(|value| !value.is_finite())
        {
            return Err(Error::Other("QSA inputs must be finite".into()));
        }

        let query_position = positions[visible_tokens - 1];
        let mut normalized_query = Vec::with_capacity(query_len);
        for head in 0..self.query_heads {
            let start = head * self.head_dim;
            let mut values = rms_norm_zero_centered(
                &query[start..start + self.head_dim],
                query_norm_weight,
                self.rms_norm_eps,
            );
            apply_partial_mrope(
                &mut values,
                self.rotary_dim,
                self.theta,
                query_position,
                mrope_sections,
            );
            normalized_query.extend(values);
        }

        let complete_blocks = visible_tokens / self.compress_ratio;
        let score_block = |block: usize, pooled_key: &mut [f32], normalized_key: &mut [f32]| {
            pooled_key.fill(0.0);
            let block_start = block * self.compress_ratio;
            for token in block_start..block_start + self.compress_ratio {
                let key = &raw_keys[token * self.head_dim..(token + 1) * self.head_dim];
                for (pooled, value) in pooled_key.iter_mut().zip(key) {
                    *pooled += *value;
                }
            }
            let reciprocal = 1.0 / self.compress_ratio as f32;
            for pooled in pooled_key.iter_mut() {
                *pooled *= reciprocal;
            }
            rms_norm_zero_centered_into(
                pooled_key,
                key_norm_weight,
                self.rms_norm_eps,
                normalized_key,
            );
            apply_partial_mrope(
                normalized_key,
                self.rotary_dim,
                self.theta,
                positions[block_start],
                mrope_sections,
            );

            let mut score = 0.0f32;
            for query_head in normalized_query.chunks_exact(self.head_dim) {
                let dot = dot_f32(query_head, &normalized_key);
                score += dot.max(0.0);
            }
            score /= (self.head_dim as f32).sqrt();
            (block, score)
        };
        let mut ranked_blocks = if complete_blocks >= QSA_SELECTOR_PAR_MIN_BLOCKS {
            (0..complete_blocks)
                .into_par_iter()
                .map_init(
                    || (vec![0.0f32; self.head_dim], vec![0.0f32; self.head_dim]),
                    |(pooled_key, normalized_key), block| {
                        score_block(block, pooled_key, normalized_key)
                    },
                )
                .collect::<Vec<_>>()
        } else {
            let mut pooled_key = vec![0.0f32; self.head_dim];
            let mut normalized_key = vec![0.0f32; self.head_dim];
            (0..complete_blocks)
                .map(|block| score_block(block, &mut pooled_key, &mut normalized_key))
                .collect::<Vec<_>>()
        };
        let selected_block_count = (self.token_budget / self.compress_ratio).min(complete_blocks);
        let rank = |(left_block, left_score): &(usize, f32),
                    (right_block, right_score): &(usize, f32)| {
            right_score
                .total_cmp(left_score)
                .then_with(|| left_block.cmp(right_block))
        };
        if selected_block_count < ranked_blocks.len() {
            ranked_blocks.select_nth_unstable_by(selected_block_count, rank);
        }
        let mut selected = Vec::with_capacity(
            selected_block_count * self.compress_ratio
                + visible_tokens.saturating_sub(complete_blocks * self.compress_ratio),
        );
        for &(block, _) in ranked_blocks.iter().take(selected_block_count) {
            selected.extend(block * self.compress_ratio..(block + 1) * self.compress_ratio);
        }
        selected.extend(complete_blocks * self.compress_ratio..visible_tokens);
        // QSA consumes this result as a mask, so token order carries no
        // semantics. Canonical ordering makes receipts and tests deterministic.
        selected.sort_unstable();
        Ok(selected)
    }
}

const QSA_SELECTOR_PAR_MIN_BLOCKS: usize = 256;

#[inline]
pub(super) fn dot_f32(left: &[f32], right: &[f32]) -> f32 {
    debug_assert_eq!(left.len(), right.len());
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON is mandatory in AArch64, and the helper bounds every
        // vector load before reading it.
        unsafe { dot_f32_neon(left, right) }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        dot_f32_scalar(left, right)
    }
}

#[cfg(any(not(target_arch = "aarch64"), test))]
#[inline]
pub(super) fn dot_f32_scalar(left: &[f32], right: &[f32]) -> f32 {
    let mut sums = [0.0f32; 8];
    let mut index = 0usize;
    while index + 8 <= left.len() {
        for lane in 0..8 {
            sums[lane] += left[index + lane] * right[index + lane];
        }
        index += 8;
    }
    let mut sum =
        (sums[0] + sums[1]) + (sums[2] + sums[3]) + (sums[4] + sums[5]) + (sums[6] + sums[7]);
    while index < left.len() {
        sum += left[index] * right[index];
        index += 1;
    }
    sum
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn dot_f32_neon(left: &[f32], right: &[f32]) -> f32 {
    use std::arch::aarch64::*;

    let mut sums = [vdupq_n_f32(0.0); 4];
    let mut index = 0usize;
    while index + 16 <= left.len() {
        for lane in 0..4 {
            let offset = index + lane * 4;
            sums[lane] = vfmaq_f32(
                sums[lane],
                unsafe { vld1q_f32(left.as_ptr().add(offset)) },
                unsafe { vld1q_f32(right.as_ptr().add(offset)) },
            );
        }
        index += 16;
    }
    let paired = vaddq_f32(vaddq_f32(sums[0], sums[1]), vaddq_f32(sums[2], sums[3]));
    let mut sum = vaddvq_f32(paired);
    while index < left.len() {
        sum += left[index] * right[index];
        index += 1;
    }
    sum
}

fn rms_norm_zero_centered(input: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    let mut output = vec![0.0; input.len()];
    rms_norm_zero_centered_into(input, weight, eps, &mut output);
    output
}

pub(super) fn rms_norm_zero_centered_into(
    input: &[f32],
    weight: &[f32],
    eps: f32,
    output: &mut [f32],
) {
    let mean_square =
        input.iter().fold(0.0f32, |sum, value| sum + value * value) / input.len() as f32;
    let inverse_rms = 1.0 / (mean_square + eps).sqrt();
    for ((output, value), scale) in output.iter_mut().zip(input).zip(weight) {
        *output = value * inverse_rms * (1.0 + scale);
    }
}

pub(super) fn apply_partial_mrope(
    values: &mut [f32],
    rotary_dim: usize,
    theta: f32,
    position: [u32; 3],
    sections: [usize; 3],
) {
    let half = rotary_dim / 2;
    for pair in 0..half {
        let axis = if pair % 3 == 1 && pair < sections[1] * 3 {
            1
        } else if pair % 3 == 2 && pair < sections[2] * 3 {
            2
        } else {
            0
        };
        let inverse_frequency = theta.powf(-((2 * pair) as f32) / rotary_dim as f32);
        let angle = position[axis] as f32 * inverse_frequency;
        let (sin, cos) = angle.sin_cos();
        let first = values[pair];
        let second = values[half + pair];
        values[pair] = first * cos - second * sin;
        values[half + pair] = second * cos + first * sin;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn selector() -> Qwen4ExpQsaSelector {
        Qwen4ExpQsaSelector::new(1, 2, 2, 2, 0, 10_000.0, 1.0e-6).unwrap()
    }

    #[test]
    fn selects_best_complete_block_and_always_keeps_tail() {
        let query = [1.0, 0.0];
        let raw_keys = [
            1.0, 0.0, 1.0, 0.0, // block 0: aligned
            0.0, 1.0, 0.0, 1.0, // block 1: orthogonal
            -1.0, 0.0, -1.0, 0.0, // block 2: negative
            0.5, 0.5, // incomplete tail
        ];
        assert_eq!(
            selector().select_unit_norm(&query, &raw_keys, 7).unwrap(),
            vec![0, 1, 6]
        );
    }

    #[test]
    fn returns_visible_tail_when_no_block_is_complete() {
        let selector = Qwen4ExpQsaSelector::new(2, 2, 4, 4, 0, 10_000.0, 1.0e-6).unwrap();
        assert_eq!(
            selector
                .select_unit_norm(&[1.0, 0.0, 0.0, 1.0], &[0.25, -0.5], 1)
                .unwrap(),
            vec![0]
        );
    }

    #[test]
    fn rejects_query_or_key_shape_drift() {
        let error = selector()
            .select_unit_norm(&[1.0], &[1.0, 0.0], 1)
            .unwrap_err();
        assert!(error.to_string().contains("query"));
        let error = selector()
            .select_unit_norm(&[1.0, 0.0], &[1.0], 1)
            .unwrap_err();
        assert!(error.to_string().contains("raw_keys"));
    }

    #[test]
    fn partial_rope_keeps_selection_finite() {
        let selector = Qwen4ExpQsaSelector::new(2, 4, 4, 2, 4, 10_000.0, 1.0e-6).unwrap();
        let query = [1.0, 2.0, 3.0, 4.0, -2.0, 1.0, 0.5, 3.0];
        let raw_keys = (0..24)
            .map(|index| (index as f32 * 0.17).sin())
            .collect::<Vec<_>>();
        let selected = selector.select_unit_norm(&query, &raw_keys, 6).unwrap();
        assert_eq!(selected.len(), 4);
        assert!(selected.iter().all(|index| *index < 6));
    }
}
