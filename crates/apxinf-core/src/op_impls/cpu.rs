//! CPU backend implementation.

use crate::{
    Backend, Device, Error, Graph, NormalGenerator, Result, SamplingBackend,
    Tensor, TokenSampler, TokenSamplingSpec,
};
use crate::kv_cache::{CpuKVCache, KvCache};
use crate::sampling::{CpuNormalGenerator, CpuTokenSampler};

/// CPU backend — all ops execute synchronously on the host.
pub struct CpuBackend;

impl SamplingBackend for CpuBackend {
    fn create_token_sampler(&self, spec: TokenSamplingSpec) -> Result<Box<dyn TokenSampler>> {
        Ok(Box::new(CpuTokenSampler::new(spec)?))
    }

    fn create_normal_generator(&self, output: Tensor) -> Result<Box<dyn NormalGenerator>> {
        Ok(Box::new(CpuNormalGenerator::new(output)?))
    }
}

impl Backend for CpuBackend {
    fn rms_norm(&self, input: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
        let data = input.as_f32()?;
        let w = weight.as_f32()?;
        let dims = input.shape().dims();
        let seq_len = dims[0];
        let hidden = dims[1];

        let mut out = vec![0.0f32; data.len()];
        for s in 0..seq_len {
            let row_start = s * hidden;
            let row = &data[row_start..row_start + hidden];
            let mean_sq: f32 = row.iter().map(|v| v * v).sum::<f32>() / hidden as f32;
            let rms = (mean_sq + eps).sqrt();
            for (i, &val) in row.iter().enumerate() {
                out[row_start + i] = (val / rms) * w[i];
            }
        }
        Tensor::from_f32(dims.to_vec(), &out)
    }

    fn silu(&self, x: &Tensor) -> Result<Tensor> {
        let data = x.as_f32()?;
        let out: Vec<f32> = data.iter().map(|&v| v / (1.0 + (-v).exp())).collect();
        Tensor::from_f32(x.shape().dims().to_vec(), &out)
    }

    fn add(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        let a_data = a.as_f32()?;
        let b_data = b.as_f32()?;
        let out: Vec<f32> = a_data.iter().zip(b_data.iter()).map(|(a, b)| a + b).collect();
        Tensor::from_f32(a.shape().dims().to_vec(), &out)
    }

    fn mul(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        let a_data = a.as_f32()?;
        let b_data = b.as_f32()?;
        let out: Vec<f32> = a_data.iter().zip(b_data.iter()).map(|(a, b)| a * b).collect();
        Tensor::from_f32(a.shape().dims().to_vec(), &out)
    }

    fn scale(&self, input: &Tensor, factor: f32) -> Result<Tensor> {
        let data = input.as_f32()?;
        let out: Vec<f32> = data.iter().map(|&v| v * factor).collect();
        Tensor::from_f32(input.shape().dims().to_vec(), &out)
    }

    fn matmul(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        a.matmul_cpu(b)
    }

    fn rope(&self, input: &Tensor, n_heads: usize, head_dim: usize,
            theta: f32, pos_offset: u32) -> Result<Tensor> {
        let data = input.as_f32()?;
        let dims = input.shape().dims();
        let seq_len = if dims.len() == 2 { 1 } else { dims[0] };
        let half_dim = head_dim / 2;

        let freqs: Vec<f32> = (0..half_dim)
            .map(|i| 1.0 / theta.powf(2.0 * i as f32 / head_dim as f32))
            .collect();

        let mut out = vec![0.0f32; data.len()];
        for s in 0..seq_len {
            let pos = pos_offset as usize + s;
            for h in 0..n_heads {
                let base = s * n_heads * head_dim + h * head_dim;
                for i in 0..half_dim {
                    let angle = pos as f32 * freqs[i];
                    let cos_v = angle.cos();
                    let sin_v = angle.sin();
                    let x1 = data[base + i];
                    let x2 = data[base + half_dim + i];
                    out[base + i] = x1 * cos_v - x2 * sin_v;
                    out[base + half_dim + i] = x1 * sin_v + x2 * cos_v;
                }
            }
        }
        Tensor::from_f32(dims.to_vec(), &out)
    }

    fn layer_norm(
        &self,
        input: &Tensor,
        weight: &Tensor,
        bias: &Tensor,
        eps: f32,
    ) -> Result<Tensor> {
        let dims = input.shape().dims();
        let hidden = dims.last().copied().ok_or_else(|| {
            Error::Other("layer_norm: input must have at least one dimension".into())
        })?;
        if hidden == 0 || weight.shape().dims() != [hidden] || bias.shape().dims() != [hidden] {
            return Err(Error::ShapeMismatch {
                expected: format!("input [..., {hidden}], weight/bias [{hidden}]"),
                got: format!(
                    "input {}, weight {}, bias {}",
                    input.shape(),
                    weight.shape(),
                    bias.shape()
                ),
            });
        }
        let data = input.as_f32()?;
        let weight = weight.as_f32()?;
        let bias = bias.as_f32()?;
        let mut output = vec![0.0; data.len()];
        for (row, output) in data
            .chunks_exact(hidden)
            .zip(output.chunks_exact_mut(hidden))
        {
            let mean = row.iter().sum::<f32>() / hidden as f32;
            let variance = row
                .iter()
                .map(|value| {
                    let centered = value - mean;
                    centered * centered
                })
                .sum::<f32>()
                / hidden as f32;
            let inverse_std = (variance + eps).sqrt().recip();
            for column in 0..hidden {
                output[column] = (row[column] - mean) * inverse_std * weight[column] + bias[column];
            }
        }
        Tensor::from_f32_vec(dims.to_vec(), output)
    }

    fn gelu_tanh(&self, input: &Tensor) -> Result<Tensor> {
        let coefficient = (2.0f32 / std::f32::consts::PI).sqrt();
        let output = input
            .as_f32()?
            .iter()
            .map(|&value| {
                0.5 * value * (1.0 + (coefficient * (value + 0.044_715 * value.powi(3))).tanh())
            })
            .collect();
        Tensor::from_f32_vec(input.shape().dims().to_vec(), output)
    }

    fn add_bias(&self, input: &Tensor, bias: &Tensor) -> Result<Tensor> {
        let columns = input.shape().dims().last().copied().ok_or_else(|| {
            Error::Other("add_bias: input must have at least one dimension".into())
        })?;
        if bias.shape().dims() != [columns] {
            return Err(Error::ShapeMismatch {
                expected: format!("[{columns}]"),
                got: bias.shape().to_string(),
            });
        }
        let bias = bias.as_f32()?;
        let output = input
            .as_f32()?
            .iter()
            .enumerate()
            .map(|(index, value)| value + bias[index % columns])
            .collect();
        Tensor::from_f32_vec(input.shape().dims().to_vec(), output)
    }

    fn rope_vision_2d(
        &self,
        input: &Tensor,
        n_heads: usize,
        head_dim: usize,
        theta: f32,
        pos_ids: &[u32],
    ) -> Result<Tensor> {
        let seq_len = rope_seq_len(input, n_heads, head_dim, "rope_vision_2d")?;
        if head_dim == 0 || !head_dim.is_multiple_of(4) || pos_ids.len() != seq_len * 2 {
            return Err(Error::Other(
                "rope_vision_2d: head_dim must be divisible by 4 and pos_ids len must be seq*2"
                    .into(),
            ));
        }
        let half = head_dim / 2;
        let axis_pairs = half / 2;
        let data = input.as_f32()?;
        let mut output = data.to_vec();
        for seq in 0..seq_len {
            for head in 0..n_heads {
                let base = (seq * n_heads + head) * head_dim;
                for pair in 0..half {
                    let axis = usize::from(pair >= axis_pairs);
                    let pair_in_axis = pair % axis_pairs;
                    let inverse_frequency = theta.powf(-(2.0 * pair_in_axis as f32 / half as f32));
                    let angle = pos_ids[seq * 2 + axis] as f32 * inverse_frequency;
                    let (sin, cos) = angle.sin_cos();
                    let first = data[base + pair];
                    let second = data[base + half + pair];
                    output[base + pair] = first * cos - second * sin;
                    output[base + half + pair] = first * sin + second * cos;
                }
            }
        }
        Tensor::from_f32_vec(input.shape().dims().to_vec(), output)
    }

    fn vision_sdpa(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        seq_len: usize,
        n_heads: usize,
        head_dim: usize,
    ) -> Result<Tensor> {
        let expected = [seq_len, n_heads, head_dim];
        if q.shape().dims() != expected
            || k.shape().dims() != expected
            || v.shape().dims() != expected
        {
            return Err(Error::ShapeMismatch {
                expected: format!("[{seq_len}, {n_heads}, {head_dim}] for q/k/v"),
                got: format!("q={}, k={}, v={}", q.shape(), k.shape(), v.shape()),
            });
        }
        let q = q.as_f32()?;
        let k = k.as_f32()?;
        let v = v.as_f32()?;
        let scale = (head_dim as f32).sqrt().recip();
        let mut output = vec![0.0; q.len()];
        let mut scores = vec![0.0; seq_len];
        for head in 0..n_heads {
            for query in 0..seq_len {
                let query_start = (query * n_heads + head) * head_dim;
                for (key, score) in scores.iter_mut().enumerate() {
                    let key_start = (key * n_heads + head) * head_dim;
                    *score = q[query_start..query_start + head_dim]
                        .iter()
                        .zip(&k[key_start..key_start + head_dim])
                        .map(|(query, key)| query * key)
                        .sum::<f32>()
                        * scale;
                }
                let maximum = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let sum = scores
                    .iter_mut()
                    .map(|score| {
                        *score = (*score - maximum).exp();
                        *score
                    })
                    .sum::<f32>();
                for score in &mut scores {
                    *score /= sum;
                }
                for column in 0..head_dim {
                    output[query_start + column] = (0..seq_len)
                        .map(|key| scores[key] * v[(key * n_heads + head) * head_dim + column])
                        .sum();
                }
            }
        }
        Tensor::from_f32_vec(vec![seq_len, n_heads * head_dim], output)
    }

    fn embedding(&self, table: &Tensor, ids: &[u32]) -> Result<Tensor> {
        let table_data = table.as_f32()?;
        let embed_dim = table.shape().dims()[1];
        let seq_len = ids.len();

        let mut out = vec![0.0f32; seq_len * embed_dim];
        for (i, &tid) in ids.iter().enumerate() {
            let src_offset = tid as usize * embed_dim;
            let dst_offset = i * embed_dim;
            out[dst_offset..dst_offset + embed_dim]
                .copy_from_slice(&table_data[src_offset..src_offset + embed_dim]);
        }
        Tensor::from_f32(vec![seq_len, embed_dim], &out)
    }

    fn sdpa_decode(&self, q: &Tensor, kv: &mut dyn KvCache,
                   layer_idx: usize, n_heads: usize, n_kv_heads: usize,
                   head_dim: usize, kv_len: usize, max_seq_len: usize) -> Result<Tensor> {
        let _ = max_seq_len;
        let q_data = q.as_f32()?;
        let cache = kv.as_any_mut().downcast_mut::<CpuKVCache>()
            .ok_or_else(|| Error::Other("expected CpuKVCache".into()))?;
        let (k_cached, v_cached) = cache.get_kv(layer_idx);

        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut output = vec![0.0f32; n_heads * head_dim];

        for h in 0..n_heads {
            let kv_h = h * n_kv_heads / n_heads;
            let mut scores = vec![0.0f32; kv_len];
            for t in 0..kv_len {
                for d in 0..head_dim {
                    scores[t] += q_data[h * head_dim + d] * k_cached[kv_h][t][d];
                }
                scores[t] *= scale;
            }
            let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exp_sum: f32 = scores.iter().map(|&s| (s - max_score).exp()).sum();
            for t in 0..kv_len {
                scores[t] = (scores[t] - max_score).exp() / exp_sum;
            }
            for t in 0..kv_len {
                for d in 0..head_dim {
                    output[h * head_dim + d] += scores[t] * v_cached[kv_h][t][d];
                }
            }
        }
        Tensor::from_f32(vec![1, n_heads * head_dim], &output)
    }

    fn sdpa_prefill(&self, q: &Tensor, kv: &mut dyn KvCache,
                    layer_idx: usize, n_heads: usize, n_kv_heads: usize,
                    head_dim: usize, kv_len: usize, max_seq_len: usize) -> Result<Tensor> {
        let _ = max_seq_len;
        let q_data = q.as_f32()?;
        let seq_len = q.shape().dims()[0];
        let cache = kv.as_any_mut().downcast_mut::<CpuKVCache>()
            .ok_or_else(|| Error::Other("expected CpuKVCache".into()))?;
        let (k_cached, v_cached) = cache.get_kv(layer_idx);

        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut output = vec![0.0f32; seq_len * n_heads * head_dim];

        for s in 0..seq_len {
            for h in 0..n_heads {
                let kv_h = h * n_kv_heads / n_heads;
                let valid_len = kv_len.min(s + 1 + kv_len - seq_len);
                let mut scores = vec![0.0f32; kv_len];
                for t in 0..valid_len {
                    for d in 0..head_dim {
                        scores[t] += q_data[s * n_heads * head_dim + h * head_dim + d]
                            * k_cached[kv_h][t][d];
                    }
                    scores[t] *= scale;
                }
                let max_score = scores[..valid_len].iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let exp_sum: f32 = scores[..valid_len].iter().map(|&s| (s - max_score).exp()).sum();
                for t in 0..valid_len {
                    scores[t] = (scores[t] - max_score).exp() / exp_sum;
                }
                for t in 0..valid_len {
                    for d in 0..head_dim {
                        output[s * n_heads * head_dim + h * head_dim + d]
                            += scores[t] * v_cached[kv_h][t][d];
                    }
                }
            }
        }
        Tensor::from_f32(vec![seq_len, n_heads * head_dim], &output)
    }

    fn create_kv_cache(&self, n_layers: usize, n_kv_heads: usize,
                       head_dim: usize, max_seq_len: usize) -> Box<dyn KvCache> {
        Box::new(CpuKVCache::new(n_layers, n_kv_heads, head_dim, max_seq_len))
    }

    fn kv_append(&self, kv: &mut dyn KvCache, layer_idx: usize,
                 k: &Tensor, v: &Tensor, append_len: usize) -> Result<()> {
        kv.append(layer_idx, k, v, append_len)
    }

    fn synchronize(&self) -> Result<()> { Ok(()) }

    fn begin_capture(&self) -> Result<()> { Ok(()) }

    fn end_capture(&self) -> Result<Box<dyn Graph>> {
        Ok(Box::new(NoopGraph))
    }

    fn device(&self) -> Device { Device::Cpu }

    fn to_device(&self, tensor: &Tensor) -> Result<Tensor> {
        Ok(tensor.clone())
    }

    fn to_cpu(&self, tensor: &Tensor) -> Result<Tensor> {
        Ok(tensor.clone())
    }

    fn as_any(&self) -> &dyn std::any::Any { self }
}

/// No-op graph (CPU has nothing to capture).
struct NoopGraph;

impl Graph for NoopGraph {
    fn replay(&self) -> Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
        assert_eq!(actual.len(), expected.len());
        for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
            let error = (actual - expected).abs();
            assert!(
                error <= tolerance,
                "element {index}: actual={actual}, expected={expected}, error={error}, tolerance={tolerance}"
            );
        }
    }

    #[test]
    fn vision_elementwise_primitives_match_scalar_references() {
        let backend = CpuBackend;
        let input = Tensor::from_f32(vec![2, 2], &[1.0, 3.0, -2.0, 2.0]).unwrap();
        let weight = Tensor::from_f32(vec![2], &[2.0, 0.5]).unwrap();
        let bias = Tensor::from_f32(vec![2], &[0.25, -0.75]).unwrap();
        let normalized = backend.layer_norm(&input, &weight, &bias, 1.0e-6).unwrap();
        assert_close(
            normalized.as_f32().unwrap(),
            &[-1.75, -0.25, -1.75, -0.25],
            2.0e-6,
        );

        let biased = backend.add_bias(&input, &bias).unwrap();
        assert_close(biased.as_f32().unwrap(), &[1.25, 2.25, -1.75, 1.25], 0.0);
        let gelu = backend
            .gelu_tanh(&Tensor::from_f32(vec![3], &[-1.0, 0.0, 1.0]).unwrap())
            .unwrap();
        assert_close(
            gelu.as_f32().unwrap(),
            &[-0.158_808, 0.0, 0.841_192],
            2.0e-6,
        );
    }

    #[test]
    fn vision_rope_and_attention_match_simple_references() {
        let backend = CpuBackend;
        let input = (0..16).map(|value| value as f32 * 0.1).collect::<Vec<_>>();
        let tensor = Tensor::from_f32(vec![2, 1, 8], &input).unwrap();
        let rope = backend
            .rope_vision_2d(&tensor, 1, 8, 10_000.0, &[0, 0, 1, 2])
            .unwrap();
        assert_eq!(&rope.as_f32().unwrap()[..8], &input[..8]);
        assert!(rope.as_f32().unwrap()[8..]
            .iter()
            .all(|value| value.is_finite()));

        let zeros = Tensor::from_f32(vec![2, 1, 2], &[0.0; 4]).unwrap();
        let values = Tensor::from_f32(vec![2, 1, 2], &[1.0, 3.0, 5.0, 7.0]).unwrap();
        let attention = backend
            .vision_sdpa(&zeros, &zeros, &values, 2, 1, 2)
            .unwrap();
        assert_eq!(attention.shape().dims(), [2, 2]);
        assert_close(attention.as_f32().unwrap(), &[3.0, 5.0, 3.0, 5.0], 0.0);
    }

    #[test]
    fn sigmoid_and_softplus_are_stable_in_both_tails() {
        let backend = CpuBackend;
        let input = Tensor::from_f32(vec![5], &[-100.0, -1.0, 0.0, 1.0, 100.0]).unwrap();

        let sigmoid = backend.sigmoid(&input).unwrap();
        let sigmoid = sigmoid.as_f32().unwrap();
        assert!(sigmoid[0].is_finite() && sigmoid[0] < 1.0e-40);
        assert_close(&sigmoid[1..4], &[0.268_941_43, 0.5, 0.731_058_6], 1.0e-6);
        assert_eq!(sigmoid[4], 1.0);

        let softplus = backend.softplus(&input).unwrap();
        let softplus = softplus.as_f32().unwrap();
        assert!(softplus[0].is_finite() && softplus[0] < 1.0e-40);
        assert_close(
            &softplus[1..4],
            &[0.313_261_7, std::f32::consts::LN_2, 1.313_261_6],
            1.0e-6,
        );
        assert_eq!(softplus[4], 100.0);
    }

    #[test]
    fn l2_normalize_uses_fla_epsilon_and_arbitrary_axis() {
        let backend = CpuBackend;
        let input =
            Tensor::from_f32(vec![2, 2, 2], &[3.0, 0.0, 4.0, 0.0, 0.0, 5.0, 12.0, 0.0]).unwrap();
        let output = backend.l2_normalize(&input, 1, 1.0e-6).unwrap();

        let norm_5 = (25.0f32 + 1.0e-6).sqrt();
        let norm_12 = (144.0f32 + 1.0e-6).sqrt();
        assert_close(
            output.as_f32().unwrap(),
            &[
                3.0 / norm_5,
                0.0,
                4.0 / norm_5,
                0.0,
                0.0,
                5.0 / norm_5,
                12.0 / norm_12,
                0.0,
            ],
            1.0e-6,
        );

        let last_dim = backend.l2_normalize(&input, -1, 1.0e-6).unwrap();
        assert_eq!(last_dim.shape(), input.shape());
        assert!(backend.l2_normalize(&input, 3, 1.0e-6).is_err());
        assert!(backend.l2_normalize(&input, -4, 1.0e-6).is_err());
    }

    #[test]
    fn rms_norm_offset_applies_qwen35_zero_centered_scale() {
        let backend = CpuBackend;
        let input = Tensor::from_f32(vec![2, 2], &[3.0, 4.0, 0.0, 2.0]).unwrap();
        let weight = Tensor::from_f32(vec![2], &[0.0, 1.0]).unwrap();
        let output = backend
            .rms_norm_offset(&input, &weight, 1.0e-6, 1.0)
            .unwrap();

        let first_rms = (12.5f32 + 1.0e-6).sqrt();
        let second_rms = (2.0f32 + 1.0e-6).sqrt();
        assert_close(
            output.as_f32().unwrap(),
            &[3.0 / first_rms, 8.0 / first_rms, 0.0, 4.0 / second_rms],
            1.0e-6,
        );
    }

    #[test]
    fn matmul_rhs_transposed_handles_non_square_matrices() {
        let backend = CpuBackend;
        let a = Tensor::from_f32(vec![2, 3], &[1.0, 2.0, 3.0, -1.0, 0.5, 4.0]).unwrap();
        let b = Tensor::from_f32(
            vec![4, 3],
            &[
                1.0, 0.0, 2.0, // 7, 7
                -1.0, 3.0, 0.5, // 6.5, 4.5
                2.0, -2.0, 1.0, // 1, 3
                0.0, 4.0, -1.0, // 5, -2
            ],
        )
        .unwrap();

        let output = backend.matmul_rhs_transposed(&a, &b).unwrap();
        assert_eq!(output.shape().dims(), &[2, 4]);
        assert_close(
            output.as_f32().unwrap(),
            &[7.0, 6.5, 1.0, 5.0, 7.0, 4.5, 1.0, -2.0],
            1.0e-6,
        );
    }

    #[test]
    fn matmul_rhs_transposed_rejects_incompatible_shapes() {
        let backend = CpuBackend;
        let a = Tensor::from_f32(vec![2, 3], &[1.0; 6]).unwrap();
        let wrong_k = Tensor::from_f32(vec![4, 2], &[1.0; 8]).unwrap();
        let rank_three = Tensor::from_f32(vec![1, 2, 3], &[1.0; 6]).unwrap();

        assert!(backend.matmul_rhs_transposed(&a, &wrong_k).is_err());
        assert!(backend.matmul_rhs_transposed(&rank_three, &a).is_err());
    }

    #[test]
    fn long_context_gqa_decode_matches_scalar_reference() {
        let backend = CpuBackend;
        let sequence_length = GQA_BLAS_MIN_KV_LEN;
        let n_heads = 4;
        let n_kv_heads = 2;
        let head_dim = 4;

        let q_values: Vec<f32> = (0..n_heads * head_dim)
            .map(|index| ((index as f32 + 1.0) * 0.17).sin())
            .collect();
        let k_values: Vec<f32> = (0..sequence_length * n_kv_heads * head_dim)
            .map(|index| ((index as f32 + 3.0) * 0.013).cos())
            .collect();
        let v_values: Vec<f32> = (0..sequence_length * n_kv_heads * head_dim)
            .map(|index| ((index as f32 + 5.0) * 0.019).sin())
            .collect();
        let query = Tensor::from_f32(vec![1, n_heads, head_dim], &q_values).unwrap();
        let key = Tensor::from_f32(vec![sequence_length, n_kv_heads, head_dim], &k_values).unwrap();
        let value =
            Tensor::from_f32(vec![sequence_length, n_kv_heads, head_dim], &v_values).unwrap();
        let mut cache = backend.create_kv_cache(1, n_kv_heads, head_dim, sequence_length);
        backend
            .kv_append(cache.as_mut(), 0, &key, &value, sequence_length)
            .unwrap();
        cache.advance(sequence_length);

        let actual = backend
            .sdpa_decode(
                &query,
                cache.as_mut(),
                0,
                n_heads,
                n_kv_heads,
                head_dim,
                sequence_length,
                sequence_length,
            )
            .unwrap();

        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut expected = vec![0.0f32; n_heads * head_dim];
        for head in 0..n_heads {
            let kv_head = head * n_kv_heads / n_heads;
            let mut scores = vec![0.0f32; sequence_length];
            for time in 0..sequence_length {
                let key_start = (time * n_kv_heads + kv_head) * head_dim;
                for dim in 0..head_dim {
                    scores[time] += q_values[head * head_dim + dim] * k_values[key_start + dim];
                }
                scores[time] *= scale;
            }
            let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let denominator: f32 = scores.iter().map(|score| (*score - max_score).exp()).sum();
            for time in 0..sequence_length {
                let probability = (scores[time] - max_score).exp() / denominator;
                let value_start = (time * n_kv_heads + kv_head) * head_dim;
                for dim in 0..head_dim {
                    expected[head * head_dim + dim] += probability * v_values[value_start + dim];
                }
            }
        }
        assert_close(actual.as_f32().unwrap(), &expected, 2.0e-5);
    }

    #[test]
    fn long_context_gqa_prefill_matches_causal_scalar_reference() {
        let backend = CpuBackend;
        let sequence_length = GQA_BLAS_MIN_KV_LEN;
        let n_heads = 4;
        let n_kv_heads = 2;
        let head_dim = 4;

        let q_values: Vec<f32> = (0..sequence_length * n_heads * head_dim)
            .map(|index| ((index as f32 + 1.0) * 0.011).sin())
            .collect();
        let k_values: Vec<f32> = (0..sequence_length * n_kv_heads * head_dim)
            .map(|index| ((index as f32 + 3.0) * 0.013).cos())
            .collect();
        let v_values: Vec<f32> = (0..sequence_length * n_kv_heads * head_dim)
            .map(|index| ((index as f32 + 5.0) * 0.019).sin())
            .collect();
        let query = Tensor::from_f32(vec![sequence_length, n_heads, head_dim], &q_values).unwrap();
        let key = Tensor::from_f32(vec![sequence_length, n_kv_heads, head_dim], &k_values).unwrap();
        let value =
            Tensor::from_f32(vec![sequence_length, n_kv_heads, head_dim], &v_values).unwrap();
        let mut cache = backend.create_kv_cache(1, n_kv_heads, head_dim, sequence_length);
        backend
            .kv_append(cache.as_mut(), 0, &key, &value, sequence_length)
            .unwrap();

        let actual = backend
            .sdpa_prefill(
                &query,
                cache.as_mut(),
                0,
                n_heads,
                n_kv_heads,
                head_dim,
                sequence_length,
                sequence_length,
            )
            .unwrap();

        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut expected = vec![0.0f32; sequence_length * n_heads * head_dim];
        for sequence in 0..sequence_length {
            let valid_length = sequence + 1;
            for head in 0..n_heads {
                let kv_head = head * n_kv_heads / n_heads;
                let mut scores = vec![0.0f32; valid_length];
                for time in 0..valid_length {
                    let key_start = (time * n_kv_heads + kv_head) * head_dim;
                    for dim in 0..head_dim {
                        scores[time] += q_values[(sequence * n_heads + head) * head_dim + dim]
                            * k_values[key_start + dim];
                    }
                    scores[time] *= scale;
                }
                let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let denominator: f32 = scores.iter().map(|score| (*score - max_score).exp()).sum();
                for time in 0..valid_length {
                    let probability = (scores[time] - max_score).exp() / denominator;
                    let value_start = (time * n_kv_heads + kv_head) * head_dim;
                    for dim in 0..head_dim {
                        expected[(sequence * n_heads + head) * head_dim + dim] +=
                            probability * v_values[value_start + dim];
                    }
                }
            }
        }
        assert_close(actual.as_f32().unwrap(), &expected, 2.0e-5);
    }

    #[test]
    fn partial_rope_rotates_prefix_and_preserves_tail() {
        let backend = CpuBackend;
        let input = Tensor::from_f32(vec![1, 1, 6], &[1.0, 2.0, 3.0, 4.0, 9.0, 10.0]).unwrap();
        let output = backend
            .rope_partial(&input, 1, 6, 4, 100.0, 1, false)
            .unwrap();
        let (sin_0, cos_0) = 1.0f32.sin_cos();
        let (sin_1, cos_1) = 0.1f32.sin_cos();
        assert_close(
            output.as_f32().unwrap(),
            &[
                1.0 * cos_0 - 3.0 * sin_0,
                2.0 * cos_1 - 4.0 * sin_1,
                1.0 * sin_0 + 3.0 * cos_0,
                2.0 * sin_1 + 4.0 * cos_1,
                9.0,
                10.0,
            ],
            1.0e-6,
        );

        let adjacent = backend
            .rope_partial(&input, 1, 6, 4, 100.0, 1, true)
            .unwrap();
        assert_close(
            adjacent.as_f32().unwrap(),
            &[
                1.0 * cos_0 - 2.0 * sin_0,
                1.0 * sin_0 + 2.0 * cos_0,
                3.0 * cos_1 - 4.0 * sin_1,
                3.0 * sin_1 + 4.0 * cos_1,
                9.0,
                10.0,
            ],
            1.0e-6,
        );
    }

    #[test]
    fn partial_mrope_interleaves_temporal_height_and_width_axes() {
        let backend = CpuBackend;
        let input =
            Tensor::from_f32(vec![1, 1, 8], &[1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 9.0, 10.0]).unwrap();
        let output = backend
            .rope_mrope_partial(&input, 1, 8, 6, 64.0, [1, 1, 1], &[1, 2, 3])
            .unwrap();

        let angle_t = 1.0f32;
        let angle_h = 2.0 / 64.0f32.powf(2.0 / 6.0);
        let angle_w = 3.0 / 64.0f32.powf(4.0 / 6.0);
        let (sin_t, cos_t) = angle_t.sin_cos();
        let (sin_h, cos_h) = angle_h.sin_cos();
        let (sin_w, cos_w) = angle_w.sin_cos();
        assert_close(
            output.as_f32().unwrap(),
            &[cos_t, cos_h, cos_w, sin_t, sin_h, sin_w, 9.0, 10.0],
            1.0e-6,
        );

        // With identical T/H/W positions, mRoPE must degenerate to scalar RoPE.
        let scalar = backend
            .rope_partial(&input, 1, 8, 6, 64.0, 7, false)
            .unwrap();
        let text_mrope = backend
            .rope_mrope_partial(&input, 1, 8, 6, 64.0, [1, 1, 1], &[7, 7, 7])
            .unwrap();
        assert_close(
            scalar.as_f32().unwrap(),
            text_mrope.as_f32().unwrap(),
            1.0e-6,
        );
    }

    #[test]
    fn partial_rope_tables_are_reused_identically_across_heads() {
        let backend = CpuBackend;
        let mut values = Vec::new();
        for head_values in [
            [1.0, 2.0, 3.0, 4.0, 9.0, 10.0],
            [-1.0, 0.5, 2.0, -3.0, 11.0, 12.0],
        ] {
            for _ in 0..3 {
                values.extend(head_values);
            }
        }
        let input = Tensor::from_f32(vec![2, 3, 6], &values).unwrap();

        let scalar = backend
            .rope_partial(&input, 3, 6, 4, 10_000.0, 17, true)
            .unwrap();
        for token in scalar.as_f32().unwrap().chunks_exact(3 * 6) {
            assert_eq!(&token[..6], &token[6..12]);
            assert_eq!(&token[..6], &token[12..18]);
        }

        let mrope = backend
            .rope_mrope_partial(
                &input,
                3,
                6,
                4,
                10_000.0,
                [1, 1, 0],
                &[17, 23, 29, 31, 37, 41],
            )
            .unwrap();
        for token in mrope.as_f32().unwrap().chunks_exact(3 * 6) {
            assert_eq!(&token[..6], &token[6..12]);
            assert_eq!(&token[..6], &token[12..18]);
        }
    }

    #[test]
    fn causal_depthwise_conv1d_matches_cross_correlation_and_returns_suffix() {
        let backend = CpuBackend;
        let input = Tensor::from_f32(vec![3, 2], &[1.0, 10.0, 2.0, 20.0, 3.0, 30.0]).unwrap();
        let weight = Tensor::from_f32(vec![2, 3], &[1.0, 2.0, 3.0, -1.0, 0.0, 1.0]).unwrap();
        let (output, state) = backend
            .causal_depthwise_conv1d(&input, &weight, None, None)
            .unwrap();

        assert_eq!(output.shape().dims(), &[3, 2]);
        assert_eq!(
            output.as_f32().unwrap(),
            &[3.0, 10.0, 8.0, 20.0, 14.0, 20.0]
        );
        assert_eq!(state.shape().dims(), &[3, 2]);
        assert_eq!(state.as_f32().unwrap(), &[1.0, 10.0, 2.0, 20.0, 3.0, 30.0]);
    }

    #[test]
    fn causal_depthwise_conv1d_incremental_matches_prefill() {
        let backend = CpuBackend;
        let input_values = [1.0, -1.0, 2.0, -2.0, 4.0, -4.0, 8.0, -8.0];
        let input = Tensor::from_f32(vec![4, 2], &input_values).unwrap();
        let weight = Tensor::from_f32(vec![2, 1, 3], &[0.25, 0.5, 1.0, 1.0, -0.5, 0.25]).unwrap();
        let bias = Tensor::from_f32(vec![2], &[0.1, -0.2]).unwrap();
        let (prefill, prefill_state) = backend
            .causal_depthwise_conv1d(&input, &weight, Some(&bias), None)
            .unwrap();

        let mut incremental_output = Vec::new();
        let mut state = None;
        for time_idx in 0..4 {
            let token = Tensor::from_f32(vec![1, 2], &input_values[time_idx * 2..time_idx * 2 + 2])
                .unwrap();
            let (output, next_state) = backend
                .causal_depthwise_conv1d(&token, &weight, Some(&bias), state.as_ref())
                .unwrap();
            incremental_output.extend_from_slice(output.as_f32().unwrap());
            state = Some(next_state);
        }

        assert_close(prefill.as_f32().unwrap(), &incremental_output, 1.0e-6);
        assert_eq!(
            prefill_state.as_f32().unwrap(),
            state.unwrap().as_f32().unwrap()
        );
    }

    #[test]
    fn gated_delta_recurrent_matches_simple_two_head_update() {
        let backend = CpuBackend;
        let q = Tensor::from_f32(vec![2, 1, 2], &[1.0, 0.0, 1.0, 1.0]).unwrap();
        let k = Tensor::from_f32(vec![2, 1, 2], &[1.0, 0.0, 0.0, 1.0]).unwrap();
        let v = Tensor::from_f32(vec![2, 2, 1], &[2.0, 3.0, 4.0, 5.0]).unwrap();
        let a = Tensor::from_f32(vec![2, 2], &[0.0; 4]).unwrap();
        let b = Tensor::from_f32(vec![2, 2], &[f32::INFINITY; 4]).unwrap();
        let a_log = Tensor::from_f32(vec![2], &[f32::NEG_INFINITY; 2]).unwrap();
        let dt_bias = Tensor::from_f32(vec![2], &[0.0, 0.0]).unwrap();

        let (output, state) = backend
            .gated_delta_recurrent(&q, &k, &v, &a, &b, &a_log, &dt_bias, None)
            .unwrap();

        assert_eq!(output.as_f32().unwrap(), &[2.0, 3.0, 6.0, 8.0]);
        assert_eq!(state.shape().dims(), &[2, 2, 1]);
        assert_eq!(state.as_f32().unwrap(), &[2.0, 4.0, 3.0, 5.0]);
    }

    #[test]
    fn gated_delta_state_uses_key_major_value_minor_layout() {
        let backend = CpuBackend;
        let q = Tensor::from_f32(vec![1, 1, 2], &[1.0, 0.0]).unwrap();
        let k = Tensor::from_f32(vec![1, 1, 2], &[1.0, 0.0]).unwrap();
        let v = Tensor::from_f32(vec![1, 1, 3], &[2.0, 3.0, 4.0]).unwrap();
        let a = Tensor::from_f32(vec![1, 1], &[0.0]).unwrap();
        let b = Tensor::from_f32(vec![1, 1], &[f32::INFINITY]).unwrap();
        let a_log = Tensor::from_f32(vec![1], &[f32::NEG_INFINITY]).unwrap();
        let dt_bias = Tensor::from_f32(vec![1], &[0.0]).unwrap();

        let (output, state) = backend
            .gated_delta_recurrent(&q, &k, &v, &a, &b, &a_log, &dt_bias, None)
            .unwrap();

        assert_eq!(output.as_f32().unwrap(), &[2.0, 3.0, 4.0]);
        assert_eq!(state.shape().dims(), &[1, 2, 3]);
        assert_eq!(state.as_f32().unwrap(), &[2.0, 3.0, 4.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn gated_delta_incremental_matches_prefill_with_finite_gates() {
        let backend = CpuBackend;
        let q_values = [0.5, -0.25, 0.1, 0.9, -0.4, 0.3];
        let k_values = [0.2, 0.7, -0.6, 0.4, 0.8, -0.1];
        let v_values = [
            1.0, -1.0, 2.0, 0.5, 0.2, 0.4, -0.3, 1.5, 2.0, 0.1, 0.8, -0.7,
        ];
        let a_values = [-0.5, 0.3, 0.2, -0.8, 0.7, 0.1];
        let b_values = [0.1, -0.2, 0.5, 0.9, -0.4, 0.6];
        let q = Tensor::from_f32(vec![3, 1, 2], &q_values).unwrap();
        let k = Tensor::from_f32(vec![3, 1, 2], &k_values).unwrap();
        let v = Tensor::from_f32(vec![3, 2, 2], &v_values).unwrap();
        let a = Tensor::from_f32(vec![3, 2], &a_values).unwrap();
        let b = Tensor::from_f32(vec![3, 2], &b_values).unwrap();
        let a_log = Tensor::from_f32(vec![2], &[-1.0, -0.25]).unwrap();
        let dt_bias = Tensor::from_f32(vec![2], &[0.2, 0.4]).unwrap();
        let (prefill, final_state) = backend
            .gated_delta_recurrent(&q, &k, &v, &a, &b, &a_log, &dt_bias, None)
            .unwrap();

        let mut incremental_output = Vec::new();
        let mut state = None;
        for time_idx in 0..3 {
            let q_token =
                Tensor::from_f32(vec![1, 1, 2], &q_values[time_idx * 2..time_idx * 2 + 2]).unwrap();
            let k_token =
                Tensor::from_f32(vec![1, 1, 2], &k_values[time_idx * 2..time_idx * 2 + 2]).unwrap();
            let v_token =
                Tensor::from_f32(vec![1, 2, 2], &v_values[time_idx * 4..time_idx * 4 + 4]).unwrap();
            let a_token =
                Tensor::from_f32(vec![1, 2], &a_values[time_idx * 2..time_idx * 2 + 2]).unwrap();
            let b_token =
                Tensor::from_f32(vec![1, 2], &b_values[time_idx * 2..time_idx * 2 + 2]).unwrap();
            let (output, next_state) = backend
                .gated_delta_recurrent(
                    &q_token,
                    &k_token,
                    &v_token,
                    &a_token,
                    &b_token,
                    &a_log,
                    &dt_bias,
                    state.as_ref(),
                )
                .unwrap();
            incremental_output.extend_from_slice(output.as_f32().unwrap());
            state = Some(next_state);
        }

        assert_close(prefill.as_f32().unwrap(), &incremental_output, 1.0e-6);
        assert_close(
            final_state.as_f32().unwrap(),
            state.unwrap().as_f32().unwrap(),
            1.0e-6,
        );
    }

    #[test]
    fn qwen35_primitives_reject_incompatible_shapes() {
        let backend = CpuBackend;
        let input = Tensor::from_f32(vec![2, 2], &[1.0; 4]).unwrap();
        let odd_rope = backend.rope_partial(
            &input.reshape(vec![1, 1, 4]).unwrap(),
            1,
            4,
            3,
            10_000.0,
            0,
            false,
        );
        assert!(odd_rope.is_err());

        let weight = Tensor::from_f32(vec![3, 2], &[1.0; 6]).unwrap();
        assert!(backend
            .causal_depthwise_conv1d(&input, &weight, None, None)
            .is_err());

        let q = Tensor::from_f32(vec![1, 2, 1], &[1.0; 2]).unwrap();
        let k = Tensor::from_f32(vec![1, 2, 1], &[1.0; 2]).unwrap();
        let v = Tensor::from_f32(vec![1, 3, 1], &[1.0; 3]).unwrap();
        let gate = Tensor::from_f32(vec![1, 3], &[0.0; 3]).unwrap();
        let head = Tensor::from_f32(vec![3], &[0.0; 3]).unwrap();
        assert!(backend
            .gated_delta_recurrent(&q, &k, &v, &gate, &gate, &head, &head, None)
            .is_err());
    }
}
