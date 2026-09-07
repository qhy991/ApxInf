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

/// Request-owned index keys. Completed blocks are immutable; only the raw
/// incomplete tail remains until enough appended tokens complete it. This
/// private cache is always used with one runtime's fixed selector and weights.
#[derive(Default)]
pub(super) struct QsaKeyCache {
    keys: Vec<f32>,
    tail: Vec<f32>,
    tail_position: [u32; 3],
    visible_tokens: usize,
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
        self.validate_inputs(
            query,
            raw_keys,
            positions,
            mrope_sections,
            query_norm_weight,
            key_norm_weight,
        )?;
        let visible_tokens = positions.len();
        let complete_blocks = visible_tokens / self.compress_ratio;
        if complete_blocks <= self.token_budget / self.compress_ratio {
            return Ok((0..visible_tokens).collect());
        }
        let mut keys = vec![0.0; complete_blocks * self.head_dim];
        let compress = |block: usize, output: &mut [f32], pooled: &mut [f32]| {
            let start = block * self.compress_ratio;
            self.compress_key(
                &raw_keys[start * self.head_dim..(start + self.compress_ratio) * self.head_dim],
                positions[start],
                mrope_sections,
                key_norm_weight,
                pooled,
                output,
            );
        };
        if complete_blocks >= QSA_SELECTOR_PAR_MIN_BLOCKS {
            keys.par_chunks_exact_mut(self.head_dim)
                .enumerate()
                .for_each_init(
                    || vec![0.0; self.head_dim],
                    |pooled, (block, output)| compress(block, output, pooled),
                );
        } else {
            let mut pooled = vec![0.0; self.head_dim];
            for (block, output) in keys.chunks_exact_mut(self.head_dim).enumerate() {
                compress(block, output, &mut pooled);
            }
        }
        self.select_compressed(
            query,
            &keys,
            visible_tokens,
            positions[visible_tokens - 1],
            mrope_sections,
            query_norm_weight,
        )
    }

    /// Append new contiguous visible tokens and select for the final query.
    /// The runtime resets this cache together with its attention KV state.
    pub(super) fn append_and_select(
        &self,
        query: &[f32],
        raw_keys: &[f32],
        positions: &[[u32; 3]],
        mrope_sections: [usize; 3],
        query_norm_weight: &[f32],
        key_norm_weight: &[f32],
        cache: &mut QsaKeyCache,
    ) -> Result<Vec<usize>> {
        // Validate only new keys: prior keys were validated on append and are
        // owned by this cache, so decode never rereads the full raw prefix.
        self.validate_inputs(
            query,
            raw_keys,
            positions,
            mrope_sections,
            query_norm_weight,
            key_norm_weight,
        )?;
        let visible_tokens = cache
            .visible_tokens
            .checked_add(positions.len())
            .ok_or_else(|| Error::Other("QSA cached context length overflow".into()))?;
        let mut pooled = Vec::new();
        for (key, &position) in raw_keys.chunks_exact(self.head_dim).zip(positions) {
            if cache.tail.is_empty() {
                cache.tail_position = position;
            }
            cache.tail.extend_from_slice(key);
            if cache.tail.len() / self.head_dim == self.compress_ratio {
                pooled.resize(self.head_dim, 0.0);
                let start = cache.keys.len();
                cache.keys.resize(start + self.head_dim, 0.0);
                self.compress_key(
                    &cache.tail,
                    cache.tail_position,
                    mrope_sections,
                    key_norm_weight,
                    &mut pooled,
                    &mut cache.keys[start..],
                );
                cache.tail.clear();
            }
        }
        cache.visible_tokens = visible_tokens;
        self.select_compressed(
            query,
            &cache.keys,
            visible_tokens,
            positions[positions.len() - 1],
            mrope_sections,
            query_norm_weight,
        )
    }

    fn validate_inputs(
        &self,
        query: &[f32],
        raw_keys: &[f32],
        positions: &[[u32; 3]],
        mrope_sections: [usize; 3],
        query_norm_weight: &[f32],
        key_norm_weight: &[f32],
    ) -> Result<()> {
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

        Ok(())
    }

    fn compress_key(
        &self,
        raw_keys: &[f32],
        position: [u32; 3],
        mrope_sections: [usize; 3],
        key_norm_weight: &[f32],
        pooled: &mut [f32],
        output: &mut [f32],
    ) {
        pooled.fill(0.0);
        for key in raw_keys.chunks_exact(self.head_dim) {
            for (pooled, value) in pooled.iter_mut().zip(key) {
                *pooled += *value;
            }
        }
        let reciprocal = 1.0 / self.compress_ratio as f32;
        for pooled in pooled.iter_mut() {
            *pooled *= reciprocal;
        }
        rms_norm_zero_centered_into(pooled, key_norm_weight, self.rms_norm_eps, output);
        apply_partial_mrope(
            output,
            self.rotary_dim,
            self.theta,
            position,
            mrope_sections,
        );
    }

    fn select_compressed(
        &self,
        query: &[f32],
        keys: &[f32],
        visible_tokens: usize,
        query_position: [u32; 3],
        mrope_sections: [usize; 3],
        query_norm_weight: &[f32],
    ) -> Result<Vec<usize>> {
        let complete_blocks = visible_tokens / self.compress_ratio;
        if complete_blocks <= self.token_budget / self.compress_ratio {
            // Every completed block and the unfinished tail are selected,
            // independently of query scores (including ties).
            return Ok((0..visible_tokens).collect());
        }
        let mut normalized_query = Vec::with_capacity(query.len());
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

        let score_block = |(block, normalized_key): (usize, &[f32])| {
            let mut score = 0.0f32;
            for query_head in normalized_query.chunks_exact(self.head_dim) {
                let dot = dot_f32(query_head, normalized_key);
                score += dot.max(0.0);
            }
            score /= (self.head_dim as f32).sqrt();
            (block, score)
        };
        let mut ranked_blocks = if complete_blocks >= QSA_SELECTOR_PAR_MIN_BLOCKS {
            keys.par_chunks_exact(self.head_dim)
                .enumerate()
                .map(score_block)
                .collect::<Vec<_>>()
        } else {
            keys.chunks_exact(self.head_dim)
                .enumerate()
                .map(score_block)
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

    #[test]
    fn cached_append_matches_stateless_across_chunks_and_mrope_positions() {
        let selector = Qwen4ExpQsaSelector::new(3, 12, 8, 4, 8, 10_000.0, 1.0e-6).unwrap();
        let keys = (0..1033 * 12)
            .map(|i| (i as f32 * 0.173).sin())
            .collect::<Vec<_>>();
        // Repeated and nonmonotonic spatial positions cannot be inferred from
        // the cache length. Chunks intentionally split compressed blocks.
        let positions = (0..1033)
            .map(|i| [i / 7, (i % 11) + 2, (i % 3) + 5])
            .collect::<Vec<_>>();
        let q_weight = (0..12).map(|i| i as f32 * 0.013).collect::<Vec<_>>();
        let k_weight = (0..12).map(|i| i as f32 * -0.021).collect::<Vec<_>>();
        for chunk_size in [1, 3, 17, 257] {
            let mut cache = QsaKeyCache::default();
            let mut start = 0;
            while start < positions.len() {
                let end = (start + chunk_size).min(positions.len());
                let query = (0..36)
                    .map(|i| ((i + end) as f32 * 0.31).cos())
                    .collect::<Vec<_>>();
                let expected = selector
                    .select_with_positions(
                        &query,
                        &keys[..end * 12],
                        &positions[..end],
                        [2, 1, 1],
                        &q_weight,
                        &k_weight,
                    )
                    .unwrap();
                let completed_before = cache.keys.clone();
                let actual = selector
                    .append_and_select(
                        &query,
                        &keys[start * 12..end * 12],
                        &positions[start..end],
                        [2, 1, 1],
                        &q_weight,
                        &k_weight,
                        &mut cache,
                    )
                    .unwrap();
                assert_eq!(actual, expected, "chunk={chunk_size}, end={end}");
                assert_eq!(&cache.keys[..completed_before.len()], completed_before);
                assert_eq!(cache.keys.len(), end / 4 * 12);
                assert_eq!(cache.tail, keys[end / 4 * 4 * 12..end * 12]);
                start = end;
            }
        }
    }

    #[test]
    fn cached_topk_keeps_canonical_ties_and_recomputes_query_scores() {
        let selector = Qwen4ExpQsaSelector::new(1, 2, 4, 2, 0, 10_000.0, 1.0e-6).unwrap();
        let keys = [
            1.0, 0.0, 1.0, 0.0, // tied aligned blocks
            1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, -1.0, 0.0, -1.0,
        ];
        let positions = (0..10).map(|p| [p; 3]).collect::<Vec<_>>();
        let mut cache = QsaKeyCache::default();
        for end in 1..=10 {
            let query = if end < 9 { [1.0, 0.0] } else { [0.0, 1.0] };
            let actual = selector
                .append_and_select(
                    &query,
                    &keys[(end - 1) * 2..end * 2],
                    &positions[end - 1..end],
                    [0; 3],
                    &[0.0; 2],
                    &[0.0; 2],
                    &mut cache,
                )
                .unwrap();
            assert_eq!(
                actual,
                selector
                    .select_unit_norm(&query, &keys[..end * 2], end)
                    .unwrap()
            );
            match end {
                5 => assert_eq!(actual, vec![0, 1, 2, 3, 4]), // all blocks + tail
                6 => assert_eq!(actual, vec![0, 1, 2, 3]),    // top-k tie boundary
                9 => assert_eq!(actual, vec![0, 1, 6, 7, 8]), // changed query + tail
                _ => {}
            }
        }
    }

    #[test]
    fn invalid_append_leaves_cache_unchanged_even_when_all_blocks_are_selected() {
        let selector = selector();
        let mut cache = QsaKeyCache::default();
        for (keys, positions) in [
            (vec![f32::NAN, 0.0], vec![[0; 3]]),
            (vec![0.0], vec![[0; 3]]),
            (vec![], vec![]),
        ] {
            assert!(selector
                .append_and_select(
                    &[1.0, 0.0],
                    &keys,
                    &positions,
                    [0; 3],
                    &[0.0; 2],
                    &[0.0; 2],
                    &mut cache,
                )
                .is_err());
            assert_eq!(cache.visible_tokens, 0);
            assert!(cache.keys.is_empty() && cache.tail.is_empty());
        }
        assert!(selector
            .select_unit_norm(&[f32::NAN, 0.0], &[0.0; 2], 1)
            .is_err());
    }
}
