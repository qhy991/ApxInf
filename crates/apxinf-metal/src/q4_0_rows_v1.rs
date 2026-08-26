//! Versioned, CPU-only Q4_0 row packing and candidate-selection oracle.
//!
//! This module deliberately has no Metal handle or production decode wiring.
//! Its block representation follows llama.cpp Q4_0: 32 consecutive row
//! values, one FP16 scale selected from the signed absolute maximum, and 16
//! bytes whose low/high nibbles encode the first/second half of the block.

use std::error::Error;
use std::fmt::{Display, Formatter};

use half::f16;

pub const Q4_0_BLOCK_SIZE_V1: usize = 32;
pub const Q4_0_QUANT_BYTES_PER_BLOCK_V1: usize = Q4_0_BLOCK_SIZE_V1 / 2;
pub const Q4_0_PACKED_BYTES_PER_BLOCK_V1: usize =
    std::mem::size_of::<u16>() + Q4_0_QUANT_BYTES_PER_BLOCK_V1;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Q4_0RowsErrorV1(String);

impl Q4_0RowsErrorV1 {
    fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl Display for Q4_0RowsErrorV1 {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl Error for Q4_0RowsErrorV1 {}

/// One canonical logical Q4_0 block.
///
/// The scale bits are exposed separately so evidence writers can hash the
/// exact little-endian block stream without relying on host struct padding.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PackedQ4_0BlockV1 {
    scale_f16_bits: u16,
    quant_nibbles: [u8; Q4_0_QUANT_BYTES_PER_BLOCK_V1],
}

impl PackedQ4_0BlockV1 {
    pub const fn scale_f16_bits(&self) -> u16 {
        self.scale_f16_bits
    }

    pub fn scale_f32(&self) -> f32 {
        f16::from_bits(self.scale_f16_bits).to_f32()
    }

    pub const fn quant_nibbles(&self) -> &[u8; Q4_0_QUANT_BYTES_PER_BLOCK_V1] {
        &self.quant_nibbles
    }
}

/// Row-major Q4_0 block32 data for a CPU correctness/coverage oracle.
#[derive(Clone, Debug)]
pub struct PackedQ4_0RowsV1 {
    rows: usize,
    columns: usize,
    blocks: Vec<PackedQ4_0BlockV1>,
}

impl PackedQ4_0RowsV1 {
    /// Pack F32 `[rows, columns]` data with llama.cpp-compatible Q4_0 math.
    pub fn pack_f32(source: &[f32], rows: usize, columns: usize) -> Result<Self, Q4_0RowsErrorV1> {
        validate_shape(source, rows, columns)?;
        let blocks_per_row = columns / Q4_0_BLOCK_SIZE_V1;
        let block_count = rows
            .checked_mul(blocks_per_row)
            .ok_or_else(|| Q4_0RowsErrorV1::new("Q4_0 v1 block dimensions overflow"))?;
        let mut blocks = Vec::with_capacity(block_count);

        for (block_index, values) in source.chunks_exact(Q4_0_BLOCK_SIZE_V1).enumerate() {
            let mut absolute_maximum = 0.0f32;
            let mut signed_maximum = 0.0f32;
            for &value in values {
                let absolute = value.abs();
                // The strict comparison intentionally preserves the first
                // signed value in an equal-absolute-magnitude tie.
                if absolute_maximum < absolute {
                    absolute_maximum = absolute;
                    signed_maximum = value;
                }
            }
            let scale = signed_maximum / -8.0;
            let inverse_scale = if scale == 0.0 { 0.0 } else { scale.recip() };
            let stored_scale = f16::from_f32(scale);
            if !stored_scale.is_finite() {
                return Err(Q4_0RowsErrorV1::new(format!(
                    "Q4_0 v1 scale for block {block_index} is not finite after FP16 storage"
                )));
            }
            let mut quant_nibbles = [0u8; Q4_0_QUANT_BYTES_PER_BLOCK_V1];
            for pair in 0..Q4_0_QUANT_BYTES_PER_BLOCK_V1 {
                let low = quantize_nibble(values[pair], inverse_scale);
                let high =
                    quantize_nibble(values[pair + Q4_0_QUANT_BYTES_PER_BLOCK_V1], inverse_scale);
                quant_nibbles[pair] = low | (high << 4);
            }
            blocks.push(PackedQ4_0BlockV1 {
                // Q4_0 quant selection uses the original F32 scale, while
                // scoring uses the scale after canonical FP16 storage.
                scale_f16_bits: stored_scale.to_bits(),
                quant_nibbles,
            });
        }

        Ok(Self {
            rows,
            columns,
            blocks,
        })
    }

    pub const fn rows(&self) -> usize {
        self.rows
    }

    pub const fn columns(&self) -> usize {
        self.columns
    }

    pub fn blocks(&self) -> &[PackedQ4_0BlockV1] {
        &self.blocks
    }

    /// Dequantize the exact stored Q4_0 blocks to row-major F32 values.
    pub fn dequantize_f32(&self) -> Vec<f32> {
        let mut output = vec![0.0f32; self.rows * self.columns];
        for (block_index, block) in self.blocks.iter().enumerate() {
            let output_start = block_index * Q4_0_BLOCK_SIZE_V1;
            let scale = block.scale_f32();
            for pair in 0..Q4_0_QUANT_BYTES_PER_BLOCK_V1 {
                let packed = block.quant_nibbles[pair];
                output[output_start + pair] = ((packed & 0x0f) as i32 - 8) as f32 * scale;
                output[output_start + pair + Q4_0_QUANT_BYTES_PER_BLOCK_V1] =
                    ((packed >> 4) as i32 - 8) as f32 * scale;
            }
        }
        output
    }

    /// Scalar CPU score oracle for one hidden row.
    pub fn scores(&self, input: &[f32]) -> Result<Vec<f32>, Q4_0RowsErrorV1> {
        validate_input(input, self.columns)?;
        let blocks_per_row = self.columns / Q4_0_BLOCK_SIZE_V1;
        let mut output = vec![0.0f32; self.rows];
        for (row, score) in output.iter_mut().enumerate() {
            let mut sum = 0.0f32;
            for block_in_row in 0..blocks_per_row {
                let block = self.blocks[row * blocks_per_row + block_in_row];
                let input_start = block_in_row * Q4_0_BLOCK_SIZE_V1;
                let scale = block.scale_f32();
                // Preserve monotonically increasing column accumulation for
                // deterministic agreement with `dequantize_f32` dot products.
                for column in 0..Q4_0_BLOCK_SIZE_V1 {
                    let packed = block.quant_nibbles[column % Q4_0_QUANT_BYTES_PER_BLOCK_V1];
                    let quant = if column < Q4_0_QUANT_BYTES_PER_BLOCK_V1 {
                        packed & 0x0f
                    } else {
                        packed >> 4
                    };
                    sum += (quant as i32 - 8) as f32 * scale * input[input_start + column];
                }
            }
            if !sum.is_finite() {
                return Err(Q4_0RowsErrorV1::new(format!(
                    "Q4_0 v1 score for row {row} is non-finite"
                )));
            }
            *score = sum;
        }
        Ok(output)
    }

    /// Select deterministic quantized candidates after request-scoped masking.
    /// Exact score ties are ordered by the lowest token ID.
    pub fn topk_excluding(
        &self,
        input: &[f32],
        k: usize,
        excluded_tokens: &[u32],
    ) -> Result<Vec<u32>, Q4_0RowsErrorV1> {
        let scores = self.scores(input)?;
        self.topk_scores_excluding(&scores, k, excluded_tokens)
    }

    /// Select candidates from externally computed scores of these exact rows.
    /// This supports correctness gates that batch the dequantized matrix in a
    /// CPU BLAS call without changing the packing or ranking contract.
    pub fn topk_scores_excluding(
        &self,
        scores: &[f32],
        k: usize,
        excluded_tokens: &[u32],
    ) -> Result<Vec<u32>, Q4_0RowsErrorV1> {
        if scores.len() != self.rows {
            return Err(Q4_0RowsErrorV1::new(format!(
                "Q4_0 v1 score row count is {}, expected {}",
                scores.len(),
                self.rows
            )));
        }
        select_topk_scores_excluding(scores, k, excluded_tokens)
    }
}

fn validate_shape(source: &[f32], rows: usize, columns: usize) -> Result<(), Q4_0RowsErrorV1> {
    if rows == 0 || rows > u32::MAX as usize || columns == 0 {
        return Err(Q4_0RowsErrorV1::new(
            "Q4_0 v1 requires 1..=u32::MAX rows and non-zero columns",
        ));
    }
    if columns % Q4_0_BLOCK_SIZE_V1 != 0 {
        return Err(Q4_0RowsErrorV1::new(format!(
            "Q4_0 v1 columns must be divisible by block size {Q4_0_BLOCK_SIZE_V1}, got {columns}"
        )));
    }
    let elements = rows
        .checked_mul(columns)
        .ok_or_else(|| Q4_0RowsErrorV1::new("Q4_0 v1 dimensions overflow"))?;
    if source.len() != elements {
        return Err(Q4_0RowsErrorV1::new(format!(
            "Q4_0 v1 source has {} elements, expected {elements}",
            source.len()
        )));
    }
    if let Some(index) = source.iter().position(|value| !value.is_finite()) {
        return Err(Q4_0RowsErrorV1::new(format!(
            "Q4_0 v1 source contains a non-finite value at element {index}"
        )));
    }
    Ok(())
}

fn validate_input(input: &[f32], columns: usize) -> Result<(), Q4_0RowsErrorV1> {
    if input.len() != columns {
        return Err(Q4_0RowsErrorV1::new(format!(
            "Q4_0 v1 input has {} elements, expected {columns}",
            input.len()
        )));
    }
    if let Some(index) = input.iter().position(|value| !value.is_finite()) {
        return Err(Q4_0RowsErrorV1::new(format!(
            "Q4_0 v1 input contains a non-finite value at element {index}"
        )));
    }
    Ok(())
}

fn quantize_nibble(value: f32, inverse_scale: f32) -> u8 {
    // llama.cpp uses a C cast after adding 8.5, i.e. truncation toward zero,
    // and caps the representable unsigned nibble at 15.
    ((value * inverse_scale + 8.5).trunc() as i32).clamp(0, 15) as u8
}

fn select_topk_scores_excluding(
    scores: &[f32],
    k: usize,
    excluded_tokens: &[u32],
) -> Result<Vec<u32>, Q4_0RowsErrorV1> {
    if scores.is_empty() || scores.len() > u32::MAX as usize || k == 0 {
        return Err(Q4_0RowsErrorV1::new(
            "Q4_0 v1 top-k requires 1..=u32::MAX scores and non-zero k",
        ));
    }
    for (index, &token) in excluded_tokens.iter().enumerate() {
        if token as usize >= scores.len() {
            return Err(Q4_0RowsErrorV1::new(format!(
                "Q4_0 v1 exclusion token {token} is outside vocabulary {}",
                scores.len()
            )));
        }
        if excluded_tokens[..index].contains(&token) {
            return Err(Q4_0RowsErrorV1::new(format!(
                "Q4_0 v1 exclusion token {token} is duplicated"
            )));
        }
    }
    if scores.len().saturating_sub(excluded_tokens.len()) < k {
        return Err(Q4_0RowsErrorV1::new(format!(
            "Q4_0 v1 exclusions leave fewer than requested top-{k} rows"
        )));
    }
    if let Some(token) = scores.iter().position(|score| !score.is_finite()) {
        return Err(Q4_0RowsErrorV1::new(format!(
            "Q4_0 v1 candidate score for token {token} is non-finite"
        )));
    }

    let mut best_scores = vec![f32::NEG_INFINITY; k];
    let mut best_tokens = vec![u32::MAX; k];
    for (token, &score) in scores.iter().enumerate() {
        let token = token as u32;
        if excluded_tokens.contains(&token)
            || !candidate_better(score, token, best_scores[k - 1], best_tokens[k - 1])
        {
            continue;
        }
        let mut position = k - 1;
        while position > 0
            && candidate_better(
                score,
                token,
                best_scores[position - 1],
                best_tokens[position - 1],
            )
        {
            best_scores[position] = best_scores[position - 1];
            best_tokens[position] = best_tokens[position - 1];
            position -= 1;
        }
        best_scores[position] = score;
        best_tokens[position] = token;
    }
    Ok(best_tokens)
}

fn candidate_better(score: f32, token: u32, current_score: f32, current_token: u32) -> bool {
    score > current_score || (score == current_score && token < current_token)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_block32_layout_round_trips_exact_integer_levels() {
        let source = (-8..=7)
            .chain(-8..=7)
            .map(|value| value as f32)
            .collect::<Vec<_>>();
        let packed = PackedQ4_0RowsV1::pack_f32(&source, 1, 32).unwrap();

        assert_eq!(std::mem::size_of::<PackedQ4_0BlockV1>(), 18);
        assert_eq!(Q4_0_PACKED_BYTES_PER_BLOCK_V1, 18);
        assert_eq!(packed.blocks().len(), 1);
        assert_eq!(
            packed.blocks()[0].scale_f16_bits(),
            f16::from_f32(1.0).to_bits()
        );
        assert_eq!(
            packed.blocks()[0].quant_nibbles(),
            &std::array::from_fn::<_, 16, _>(|index| index as u8 | ((index as u8) << 4))
        );
        assert_eq!(packed.dequantize_f32(), source);
    }

    #[test]
    fn equal_absolute_extrema_keep_the_first_signed_scale() {
        let mut positive_first = [0.0f32; 32];
        positive_first[0] = 8.0;
        positive_first[1] = -8.0;
        let positive = PackedQ4_0RowsV1::pack_f32(&positive_first, 1, 32).unwrap();
        assert_eq!(
            positive.blocks()[0].scale_f16_bits(),
            f16::from_f32(-1.0).to_bits()
        );
        assert_eq!(&positive.dequantize_f32()[..2], &[8.0, -7.0]);

        let mut negative_first = [0.0f32; 32];
        negative_first[0] = -8.0;
        negative_first[1] = 8.0;
        let negative = PackedQ4_0RowsV1::pack_f32(&negative_first, 1, 32).unwrap();
        assert_eq!(
            negative.blocks()[0].scale_f16_bits(),
            f16::from_f32(1.0).to_bits()
        );
        assert_eq!(&negative.dequantize_f32()[..2], &[-8.0, 7.0]);
    }

    #[test]
    fn scoring_uses_stored_fp16_scale_and_preserves_order() {
        let mut source = vec![1.0f32; 32];
        source.extend(vec![-1.0f32; 32]);
        source.extend(vec![0.0f32; 32]);
        let packed = PackedQ4_0RowsV1::pack_f32(&source, 3, 32).unwrap();
        let scores = packed.scores(&[1.0; 32]).unwrap();

        assert_eq!(scores, vec![32.0, -32.0, 0.0]);
        assert_eq!(
            packed.topk_excluding(&[1.0; 32], 3, &[]).unwrap(),
            [0, 2, 1]
        );
    }

    #[test]
    fn topk_exclusion_and_exact_ties_choose_low_tokens() {
        let packed = PackedQ4_0RowsV1::pack_f32(&vec![0.0; 20 * 32], 20, 32).unwrap();
        assert_eq!(
            packed.topk_excluding(&[0.0; 32], 4, &[0, 2]).unwrap(),
            [1, 3, 4, 5]
        );
        assert_eq!(
            packed
                .topk_scores_excluding(&[0.0; 20], 8, &[0, 2])
                .unwrap(),
            [1, 3, 4, 5, 6, 7, 8, 9]
        );
    }

    #[test]
    fn validation_rejects_malformed_shapes_values_masks_and_k() {
        assert!(PackedQ4_0RowsV1::pack_f32(&[0.0; 31], 1, 31).is_err());
        assert!(PackedQ4_0RowsV1::pack_f32(&[0.0; 32], 2, 32).is_err());
        let mut non_finite = [0.0; 32];
        non_finite[7] = f32::NAN;
        assert!(PackedQ4_0RowsV1::pack_f32(&non_finite, 1, 32).is_err());
        assert!(PackedQ4_0RowsV1::pack_f32(&[f32::MAX; 32], 1, 32).is_err());

        let packed = PackedQ4_0RowsV1::pack_f32(&vec![0.0; 8 * 32], 8, 32).unwrap();
        assert!(packed.topk_excluding(&[0.0; 31], 4, &[]).is_err());
        assert!(packed.topk_excluding(&[0.0; 32], 0, &[]).is_err());
        assert!(packed.topk_excluding(&[0.0; 32], 4, &[1, 1]).is_err());
        assert!(packed.topk_excluding(&[0.0; 32], 4, &[8]).is_err());
        assert!(packed.topk_excluding(&[0.0; 32], 7, &[0, 1]).is_err());
        assert!(packed.topk_scores_excluding(&[0.0; 7], 4, &[]).is_err());
        let mut bad_scores = [0.0; 8];
        bad_scores[3] = f32::INFINITY;
        assert!(packed.topk_scores_excluding(&bad_scores, 4, &[]).is_err());
    }
}
