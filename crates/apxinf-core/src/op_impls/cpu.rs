//! CPU backend implementation.

use crate::kv_cache::{CpuKVCache, KvCache};
use crate::{Backend, Device, Error, Graph, Result, Tensor};

/// Below this length, small BLAS calls cost more than the scalar GQA loop.
/// Long-context decode groups every set of query heads sharing a KV head.
const GQA_BLAS_MIN_KV_LEN: usize = 128;

/// CPU backend — all ops execute synchronously on the host.
pub struct CpuBackend;

#[inline]
fn sigmoid_scalar(x: f32) -> f32 {
    // This branch form avoids overflow in exp(-x) for large negative inputs.
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let exp_x = x.exp();
        exp_x / (1.0 + exp_x)
    }
}

#[inline]
fn softplus_scalar(x: f32) -> f32 {
    // Match PyTorch's default threshold while keeping both tails finite.
    if x > 20.0 {
        x
    } else if x < -20.0 {
        x.exp()
    } else {
        x.exp().ln_1p()
    }
}

fn resolve_axis(ndim: usize, dim: isize, op: &str) -> Result<usize> {
    let axis = if dim < 0 { ndim as isize + dim } else { dim };
    if axis < 0 || axis as usize >= ndim {
        return Err(Error::Other(format!(
            "{op}: invalid dim {dim} for {ndim}D tensor"
        )));
    }
    Ok(axis as usize)
}

fn rope_seq_len(input: &Tensor, n_heads: usize, head_dim: usize, op: &str) -> Result<usize> {
    let dims = input.shape().dims();
    match dims {
        [heads, dim] if *heads == n_heads && *dim == head_dim => Ok(1),
        [seq, heads, dim] if *heads == n_heads && *dim == head_dim => Ok(*seq),
        _ => Err(Error::ShapeMismatch {
            expected: format!("[{n_heads}, {head_dim}] or [seq, {n_heads}, {head_dim}]"),
            got: format!("{op} input {}", input.shape()),
        }),
    }
}

fn validate_partial_rope(
    input: &Tensor,
    n_heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    theta: f32,
    op: &str,
) -> Result<usize> {
    let seq_len = rope_seq_len(input, n_heads, head_dim, op)?;
    if rotary_dim == 0 || rotary_dim > head_dim || rotary_dim % 2 != 0 {
        return Err(Error::Other(format!(
            "{op}: rotary_dim must be non-zero, even, and <= head_dim; got {rotary_dim} and {head_dim}"
        )));
    }
    if !theta.is_finite() || theta <= 0.0 {
        return Err(Error::Other(format!(
            "{op}: theta must be finite and positive, got {theta}"
        )));
    }
    Ok(seq_len)
}

fn partial_rope_inv_freq(rotary_dim: usize, theta: f32) -> Vec<f32> {
    (0..rotary_dim / 2)
        .map(|pair_idx| 1.0f32 / theta.powf(2.0 * pair_idx as f32 / rotary_dim as f32))
        .collect()
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
        Tensor::from_f32_vec(dims.to_vec(), out)
    }

    fn silu(&self, x: &Tensor) -> Result<Tensor> {
        let data = x.as_f32()?;
        let out: Vec<f32> = data.iter().map(|&v| v / (1.0 + (-v).exp())).collect();
        Tensor::from_f32_vec(x.shape().dims().to_vec(), out)
    }

    fn sigmoid(&self, input: &Tensor) -> Result<Tensor> {
        let data = input.as_f32()?;
        let out: Vec<f32> = data.iter().copied().map(sigmoid_scalar).collect();
        Tensor::from_f32_vec(input.shape().dims().to_vec(), out)
    }

    fn softplus(&self, input: &Tensor) -> Result<Tensor> {
        let data = input.as_f32()?;
        let out: Vec<f32> = data.iter().copied().map(softplus_scalar).collect();
        Tensor::from_f32_vec(input.shape().dims().to_vec(), out)
    }

    fn l2_normalize(&self, input: &Tensor, dim: isize, eps: f32) -> Result<Tensor> {
        let dims = input.shape().dims();
        let axis = resolve_axis(dims.len(), dim, "l2_normalize")?;
        if !eps.is_finite() || eps < 0.0 {
            return Err(Error::Other(format!(
                "l2_normalize: eps must be finite and non-negative, got {eps}"
            )));
        }

        let axis_len = dims[axis];
        if axis_len == 0 {
            return Tensor::from_f32(dims.to_vec(), &[]);
        }
        let outer: usize = dims[..axis].iter().product();
        let inner: usize = dims[axis + 1..].iter().product();
        let data = input.as_f32()?;
        let mut out = vec![0.0f32; data.len()];

        for outer_idx in 0..outer {
            for inner_idx in 0..inner {
                let mut sum_sq = 0.0f32;
                for axis_idx in 0..axis_len {
                    let idx = (outer_idx * axis_len + axis_idx) * inner + inner_idx;
                    sum_sq += data[idx] * data[idx];
                }
                let inv_norm = (sum_sq + eps).sqrt().recip();
                for axis_idx in 0..axis_len {
                    let idx = (outer_idx * axis_len + axis_idx) * inner + inner_idx;
                    out[idx] = data[idx] * inv_norm;
                }
            }
        }

        Tensor::from_f32_vec(dims.to_vec(), out)
    }

    fn rms_norm_offset(
        &self,
        input: &Tensor,
        weight: &Tensor,
        eps: f32,
        weight_offset: f32,
    ) -> Result<Tensor> {
        let dims = input.shape().dims();
        let hidden = dims.last().copied().ok_or_else(|| {
            Error::Other("rms_norm_offset: input must have at least one dimension".into())
        })?;
        if hidden == 0 {
            return Err(Error::Other(
                "rms_norm_offset: last dimension must be non-zero".into(),
            ));
        }
        if weight.shape().dims() != [hidden] {
            return Err(Error::ShapeMismatch {
                expected: format!("[{hidden}]"),
                got: weight.shape().to_string(),
            });
        }
        if !eps.is_finite() || eps < 0.0 {
            return Err(Error::Other(format!(
                "rms_norm_offset: eps must be finite and non-negative, got {eps}"
            )));
        }

        let data = input.as_f32()?;
        let scales = weight.as_f32()?;
        let mut out = vec![0.0f32; data.len()];
        for (row, out_row) in data.chunks_exact(hidden).zip(out.chunks_exact_mut(hidden)) {
            let mean_sq = row.iter().map(|value| value * value).sum::<f32>() / hidden as f32;
            let inv_rms = (mean_sq + eps).sqrt().recip();
            for column in 0..hidden {
                out_row[column] = row[column] * inv_rms * (scales[column] + weight_offset);
            }
        }
        Tensor::from_f32_vec(dims.to_vec(), out)
    }

    fn causal_depthwise_conv1d(
        &self,
        input: &Tensor,
        weight: &Tensor,
        bias: Option<&Tensor>,
        state: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor)> {
        let input_dims = input.shape().dims();
        if input_dims.len() < 2 {
            return Err(Error::Other(format!(
                "causal_depthwise_conv1d: input must be [..., seq, channels], got {}",
                input.shape()
            )));
        }
        let seq_len = input_dims[input_dims.len() - 2];
        let channels = input_dims[input_dims.len() - 1];
        if channels == 0 {
            return Err(Error::Other(
                "causal_depthwise_conv1d: channels must be non-zero".into(),
            ));
        }
        let batch: usize = input_dims[..input_dims.len() - 2].iter().product();
        let batch = batch.max(1);

        let weight_dims = weight.shape().dims();
        let kernel_size = match weight_dims {
            [weight_channels, kernel] if *weight_channels == channels => *kernel,
            [weight_channels, one, kernel] if *weight_channels == channels && *one == 1 => *kernel,
            _ => {
                return Err(Error::ShapeMismatch {
                    expected: format!("[{channels}, kernel] or [{channels}, 1, kernel]"),
                    got: weight.shape().to_string(),
                });
            }
        };
        if kernel_size == 0 {
            return Err(Error::Other(
                "causal_depthwise_conv1d: kernel_size must be non-zero".into(),
            ));
        }
        // Qwen3.5 keeps K raw inputs (rather than the mathematical minimum
        // K-1) in its cache. The oldest slot is skipped for the first output
        // of this chunk, then the current token enters at the right edge.
        let state_len = kernel_size;

        let bias_data = match bias {
            Some(bias) => {
                if bias.shape().dims() != [channels] {
                    return Err(Error::ShapeMismatch {
                        expected: format!("[{channels}]"),
                        got: bias.shape().to_string(),
                    });
                }
                Some(bias.as_f32()?)
            }
            None => None,
        };

        let mut state_dims = input_dims[..input_dims.len() - 2].to_vec();
        state_dims.extend([state_len, channels]);
        let state_data = match state {
            Some(state) => {
                if state.shape().dims() != state_dims.as_slice() {
                    return Err(Error::ShapeMismatch {
                        expected: format!("{}", crate::Shape::new(state_dims.clone())),
                        got: state.shape().to_string(),
                    });
                }
                state.as_f32()?.to_vec()
            }
            None => vec![0.0f32; batch * state_len * channels],
        };

        let input_data = input.as_f32()?;
        let weight_data = weight.as_f32()?;
        let mut output = vec![0.0f32; input_data.len()];

        // Cross-correlation over [previous state, this chunk]. At output t the
        // K-wide window begins at extended-sequence index t+1 because the
        // cached state contains K samples and its oldest one is retired.
        for batch_idx in 0..batch {
            for time_idx in 0..seq_len {
                for channel_idx in 0..channels {
                    let mut value = bias_data.map_or(0.0, |values| values[channel_idx]);
                    for kernel_idx in 0..kernel_size {
                        let extended_idx = time_idx + kernel_idx + 1;
                        let sample = if extended_idx < state_len {
                            state_data
                                [(batch_idx * state_len + extended_idx) * channels + channel_idx]
                        } else {
                            let input_time = extended_idx - state_len;
                            input_data[(batch_idx * seq_len + input_time) * channels + channel_idx]
                        };
                        value += sample * weight_data[channel_idx * kernel_size + kernel_idx];
                    }
                    output[(batch_idx * seq_len + time_idx) * channels + channel_idx] = value;
                }
            }
        }

        let mut next_state = vec![0.0f32; batch * state_len * channels];
        for batch_idx in 0..batch {
            for state_idx in 0..state_len {
                // The retained suffix starts at `seq_len` in the conceptual
                // concatenation [old_state (K), input].
                let extended_idx = seq_len + state_idx;
                for channel_idx in 0..channels {
                    next_state[(batch_idx * state_len + state_idx) * channels + channel_idx] =
                        if extended_idx < state_len {
                            state_data
                                [(batch_idx * state_len + extended_idx) * channels + channel_idx]
                        } else {
                            let input_time = extended_idx - state_len;
                            input_data[(batch_idx * seq_len + input_time) * channels + channel_idx]
                        };
                }
            }
        }

        Ok((
            Tensor::from_f32_vec(input_dims.to_vec(), output)?,
            Tensor::from_f32_vec(state_dims, next_state)?,
        ))
    }

    fn gated_delta_recurrent(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        a: &Tensor,
        b: &Tensor,
        a_log: &Tensor,
        dt_bias: &Tensor,
        state: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor)> {
        let q_dims = q.shape().dims();
        let k_dims = k.shape().dims();
        let v_dims = v.shape().dims();
        if q_dims.len() != 3 || k_dims.len() != 3 || v_dims.len() != 3 {
            return Err(Error::Other(format!(
                "gated_delta_recurrent: q/k/v must be 3D, got {}, {}, {}",
                q.shape(),
                k.shape(),
                v.shape()
            )));
        }
        let (seq_len, key_heads, key_dim) = (q_dims[0], q_dims[1], q_dims[2]);
        if k_dims != q_dims {
            return Err(Error::ShapeMismatch {
                expected: q.shape().to_string(),
                got: k.shape().to_string(),
            });
        }
        let (value_seq_len, value_heads, value_dim) = (v_dims[0], v_dims[1], v_dims[2]);
        if key_heads == 0 || key_dim == 0 || value_heads == 0 || value_dim == 0 {
            return Err(Error::Other(
                "gated_delta_recurrent: head counts and dimensions must be non-zero".into(),
            ));
        }
        if value_seq_len != seq_len {
            return Err(Error::ShapeMismatch {
                expected: format!("[{seq_len}, Hv, Dv]"),
                got: v.shape().to_string(),
            });
        }
        if value_heads % key_heads != 0 {
            return Err(Error::Other(format!(
                "gated_delta_recurrent: Hv ({value_heads}) must be divisible by Hk ({key_heads})"
            )));
        }
        let gate_shape = [seq_len, value_heads];
        for (name, tensor) in [("a", a), ("b", b)] {
            if tensor.shape().dims() != gate_shape {
                return Err(Error::ShapeMismatch {
                    expected: format!("[{seq_len}, {value_heads}] for {name}"),
                    got: tensor.shape().to_string(),
                });
            }
        }
        for (name, tensor) in [("A_log", a_log), ("dt_bias", dt_bias)] {
            if tensor.shape().dims() != [value_heads] {
                return Err(Error::ShapeMismatch {
                    expected: format!("[{value_heads}] for {name}"),
                    got: tensor.shape().to_string(),
                });
            }
        }

        let state_dims = vec![value_heads, key_dim, value_dim];
        let mut state_data = match state {
            Some(state) => {
                if state.shape().dims() != state_dims.as_slice() {
                    return Err(Error::ShapeMismatch {
                        expected: crate::Shape::new(state_dims.clone()).to_string(),
                        got: state.shape().to_string(),
                    });
                }
                state.as_f32()?.to_vec()
            }
            None => vec![0.0f32; value_heads * value_dim * key_dim],
        };

        let q_data = q.as_f32()?;
        let k_data = k.as_f32()?;
        let v_data = v.as_f32()?;
        let a_data = a.as_f32()?;
        let b_data = b.as_f32()?;
        let a_log_data = a_log.as_f32()?;
        let dt_bias_data = dt_bias.as_f32()?;
        let repeat_factor = value_heads / key_heads;
        let mut output = vec![0.0f32; seq_len * value_heads * value_dim];
        // One scratch row is enough because heads and timesteps are advanced
        // serially.  It first holds the retrieved value (`S^T k`) and is then
        // rewritten in place as the gated delta.  Keeping V as the inner loop
        // follows the canonical `[H, K, V]` state layout, so every state row is
        // streamed contiguously instead of touching one cache line per K item.
        let mut delta = vec![0.0f32; value_dim];

        for time_idx in 0..seq_len {
            for value_head in 0..value_heads {
                let key_head = value_head / repeat_factor;
                let gate_idx = time_idx * value_heads + value_head;
                let beta = sigmoid_scalar(b_data[gate_idx]);
                let decay = (-a_log_data[value_head].exp()
                    * softplus_scalar(a_data[gate_idx] + dt_bias_data[value_head]))
                .exp();
                let q_offset = (time_idx * key_heads + key_head) * key_dim;
                let k_offset = q_offset;
                let state_offset = value_head * key_dim * value_dim;
                let value_offset = (time_idx * value_heads + value_head) * value_dim;

                // S <- decay * S.  This is elementwise, so moving it ahead of
                // the independent V-column reductions preserves the scalar
                // arithmetic while making the state walk contiguous.
                for state_value in &mut state_data[state_offset..state_offset + key_dim * value_dim]
                {
                    *state_value *= decay;
                }

                delta.fill(0.0);
                for key_idx in 0..key_dim {
                    let state_row = state_offset + key_idx * value_dim;
                    let key_value = k_data[k_offset + key_idx];
                    for value_idx in 0..value_dim {
                        delta[value_idx] += state_data[state_row + value_idx] * key_value;
                    }
                }

                for value_idx in 0..value_dim {
                    delta[value_idx] = (v_data[value_offset + value_idx] - delta[value_idx]) * beta;
                }

                for key_idx in 0..key_dim {
                    let state_row = state_offset + key_idx * value_dim;
                    let key_value = k_data[k_offset + key_idx];
                    let query_value = q_data[q_offset + key_idx];
                    for value_idx in 0..value_dim {
                        let state_idx = state_row + value_idx;
                        state_data[state_idx] += key_value * delta[value_idx];
                        output[value_offset + value_idx] += state_data[state_idx] * query_value;
                    }
                }
            }
        }

        Ok((
            Tensor::from_f32_vec(vec![seq_len, value_heads, value_dim], output)?,
            Tensor::from_f32_vec(state_dims, state_data)?,
        ))
    }

    fn add(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        let a_data = a.as_f32()?;
        let b_data = b.as_f32()?;
        let out: Vec<f32> = a_data
            .iter()
            .zip(b_data.iter())
            .map(|(a, b)| a + b)
            .collect();
        Tensor::from_f32_vec(a.shape().dims().to_vec(), out)
    }

    fn mul(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        let a_data = a.as_f32()?;
        let b_data = b.as_f32()?;
        let out: Vec<f32> = a_data
            .iter()
            .zip(b_data.iter())
            .map(|(a, b)| a * b)
            .collect();
        Tensor::from_f32_vec(a.shape().dims().to_vec(), out)
    }

    fn scale(&self, input: &Tensor, factor: f32) -> Result<Tensor> {
        let data = input.as_f32()?;
        let out: Vec<f32> = data.iter().map(|&v| v * factor).collect();
        Tensor::from_f32_vec(input.shape().dims().to_vec(), out)
    }

    fn matmul(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        a.matmul_cpu(b)
    }

    fn matmul_rhs_transposed(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        let a_dims = a.shape().dims();
        let b_dims = b.shape().dims();
        if a_dims.len() != 2 || b_dims.len() != 2 || a_dims[1] != b_dims[1] {
            return Err(Error::ShapeMismatch {
                expected: "a=[m, k], b=[n, k] with matching k".into(),
                got: format!("a={}, b={}", a.shape(), b.shape()),
            });
        }

        // `as_f32` borrows the existing buffers. In particular, B can be the
        // multi-gigabyte token-embedding table without a per-call clone.
        let a_data = a.as_f32()?;
        let b_data = b.as_f32()?;
        let (m, k, n) = (a_dims[0], a_dims[1], b_dims[0]);
        let mut output = vec![0.0f32; m * n];
        crate::ops::sgemm_rhs_transposed(m, k, n, a_data, b_data, &mut output);
        Tensor::from_f32_vec(vec![m, n], output)
    }

    fn rope(
        &self,
        input: &Tensor,
        n_heads: usize,
        head_dim: usize,
        theta: f32,
        pos_offset: u32,
    ) -> Result<Tensor> {
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
        Tensor::from_f32_vec(dims.to_vec(), out)
    }

    fn rope_partial(
        &self,
        input: &Tensor,
        n_heads: usize,
        head_dim: usize,
        rotary_dim: usize,
        theta: f32,
        pos_offset: u32,
        interleaved: bool,
    ) -> Result<Tensor> {
        let seq_len =
            validate_partial_rope(input, n_heads, head_dim, rotary_dim, theta, "rope_partial")?;
        let data = input.as_f32()?;
        let pair_count = rotary_dim / 2;
        let inv_freq = partial_rope_inv_freq(rotary_dim, theta);
        let mut trig_table = vec![(0.0f32, 0.0f32); seq_len * pair_count];
        for seq_idx in 0..seq_len {
            let position = (pos_offset as u64 + seq_idx as u64) as f32;
            for pair_idx in 0..pair_count {
                trig_table[seq_idx * pair_count + pair_idx] =
                    (position * inv_freq[pair_idx]).sin_cos();
            }
        }
        let mut out = data.to_vec();

        for seq_idx in 0..seq_len {
            for head_idx in 0..n_heads {
                let base = (seq_idx * n_heads + head_idx) * head_dim;
                for pair_idx in 0..pair_count {
                    let (sin, cos) = trig_table[seq_idx * pair_count + pair_idx];
                    let (first_idx, second_idx) = if interleaved {
                        (base + 2 * pair_idx, base + 2 * pair_idx + 1)
                    } else {
                        (base + pair_idx, base + pair_count + pair_idx)
                    };
                    let first = data[first_idx];
                    let second = data[second_idx];
                    out[first_idx] = first * cos - second * sin;
                    out[second_idx] = first * sin + second * cos;
                }
            }
        }

        Tensor::from_f32_vec(input.shape().dims().to_vec(), out)
    }

    fn rope_mrope_partial(
        &self,
        input: &Tensor,
        n_heads: usize,
        head_dim: usize,
        rotary_dim: usize,
        theta: f32,
        sections: [usize; 3],
        pos_ids: &[u32],
    ) -> Result<Tensor> {
        let seq_len = validate_partial_rope(
            input,
            n_heads,
            head_dim,
            rotary_dim,
            theta,
            "rope_mrope_partial",
        )?;
        let pair_count = rotary_dim / 2;
        if sections.iter().sum::<usize>() != pair_count {
            return Err(Error::Other(format!(
                "rope_mrope_partial: sections {:?} must sum to rotary_dim / 2 ({pair_count})",
                sections
            )));
        }
        if pos_ids.len() != seq_len * 3 {
            return Err(Error::Other(format!(
                "rope_mrope_partial: pos_ids len {} != seq_len {} * 3",
                pos_ids.len(),
                seq_len
            )));
        }

        let data = input.as_f32()?;
        let inv_freq = partial_rope_inv_freq(rotary_dim, theta);
        let mut trig_table = vec![(0.0f32, 0.0f32); seq_len * pair_count];
        for seq_idx in 0..seq_len {
            for pair_idx in 0..pair_count {
                // Hugging Face starts with the temporal frequency and
                // overwrites every third H/W slot up to that axis' section.
                let axis = if pair_idx % 3 == 1 && pair_idx < sections[1] * 3 {
                    1
                } else if pair_idx % 3 == 2 && pair_idx < sections[2] * 3 {
                    2
                } else {
                    0
                };
                let position = pos_ids[seq_idx * 3 + axis] as f32;
                trig_table[seq_idx * pair_count + pair_idx] =
                    (position * inv_freq[pair_idx]).sin_cos();
            }
        }
        let mut out = data.to_vec();
        for seq_idx in 0..seq_len {
            for head_idx in 0..n_heads {
                let base = (seq_idx * n_heads + head_idx) * head_dim;
                for pair_idx in 0..pair_count {
                    let (sin, cos) = trig_table[seq_idx * pair_count + pair_idx];
                    let first_idx = base + pair_idx;
                    let second_idx = base + pair_count + pair_idx;
                    let first = data[first_idx];
                    let second = data[second_idx];
                    out[first_idx] = first * cos - second * sin;
                    out[second_idx] = first * sin + second * cos;
                }
            }
        }

        Tensor::from_f32_vec(input.shape().dims().to_vec(), out)
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
        Tensor::from_f32_vec(vec![seq_len, embed_dim], out)
    }

    fn sdpa_decode(
        &self,
        q: &Tensor,
        kv: &mut dyn KvCache,
        layer_idx: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        kv_len: usize,
        max_seq_len: usize,
    ) -> Result<Tensor> {
        let _ = max_seq_len;
        let q_data = q.as_f32()?;
        let cache = kv
            .as_any_mut()
            .downcast_mut::<CpuKVCache>()
            .ok_or_else(|| Error::Other("expected CpuKVCache".into()))?;
        let (k_cached, v_cached) = cache.get_kv(layer_idx);

        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut output = vec![0.0f32; n_heads * head_dim];

        if cfg!(any(feature = "accelerate", feature = "openblas"))
            && n_kv_heads > 0
            && n_heads % n_kv_heads == 0
            && kv_len >= GQA_BLAS_MIN_KV_LEN
        {
            let queries_per_kv = n_heads / n_kv_heads;
            let mut grouped_scores = vec![0.0f32; n_heads * kv_len];
            for kv_h in 0..n_kv_heads {
                let query_head = kv_h * queries_per_kv;
                let query_start = query_head * head_dim;
                let query_end = query_start + queries_per_kv * head_dim;
                let score_start = query_head * kv_len;
                let score_end = score_start + queries_per_kv * kv_len;
                let cache_start = cache.row_offset(kv_h, 0);

                crate::ops::sgemm_rhs_transposed(
                    queries_per_kv,
                    head_dim,
                    kv_len,
                    &q_data[query_start..query_end],
                    &k_cached[cache_start..],
                    &mut grouped_scores[score_start..score_end],
                );

                for query in 0..queries_per_kv {
                    let row_start = score_start + query * kv_len;
                    let row = &mut grouped_scores[row_start..row_start + kv_len];
                    for score in row.iter_mut() {
                        *score *= scale;
                    }
                    let max_score = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    let exp_sum: f32 = row.iter().map(|score| (*score - max_score).exp()).sum();
                    for score in row.iter_mut() {
                        *score = (*score - max_score).exp() / exp_sum;
                    }
                }

                crate::ops::sgemm(
                    queries_per_kv,
                    kv_len,
                    head_dim,
                    &grouped_scores[score_start..score_end],
                    &v_cached[cache_start..],
                    &mut output[query_start..query_end],
                );
            }
            return Tensor::from_f32_vec(vec![1, n_heads * head_dim], output);
        }

        let mut scores = vec![0.0f32; kv_len];

        for h in 0..n_heads {
            let kv_h = h * n_kv_heads / n_heads;
            scores.fill(0.0);
            for t in 0..kv_len {
                let cache_row = cache.row_offset(kv_h, t);
                for d in 0..head_dim {
                    scores[t] += q_data[h * head_dim + d] * k_cached[cache_row + d];
                }
                scores[t] *= scale;
            }
            let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exp_sum: f32 = scores.iter().map(|&s| (s - max_score).exp()).sum();
            for t in 0..kv_len {
                scores[t] = (scores[t] - max_score).exp() / exp_sum;
            }
            for t in 0..kv_len {
                let cache_row = cache.row_offset(kv_h, t);
                for d in 0..head_dim {
                    output[h * head_dim + d] += scores[t] * v_cached[cache_row + d];
                }
            }
        }
        Tensor::from_f32_vec(vec![1, n_heads * head_dim], output)
    }

    fn sdpa_prefill(
        &self,
        q: &Tensor,
        kv: &mut dyn KvCache,
        layer_idx: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        kv_len: usize,
        max_seq_len: usize,
    ) -> Result<Tensor> {
        let _ = max_seq_len;
        let q_data = q.as_f32()?;
        let seq_len = q.shape().dims()[0];
        let cache = kv
            .as_any_mut()
            .downcast_mut::<CpuKVCache>()
            .ok_or_else(|| Error::Other("expected CpuKVCache".into()))?;
        let (k_cached, v_cached) = cache.get_kv(layer_idx);

        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut output = vec![0.0f32; seq_len * n_heads * head_dim];
        let mut scores = vec![0.0f32; kv_len];
        let grouped = cfg!(any(feature = "accelerate", feature = "openblas"))
            && n_kv_heads > 0
            && n_heads % n_kv_heads == 0;
        let queries_per_kv = if grouped { n_heads / n_kv_heads } else { 0 };
        let mut grouped_scores = vec![0.0f32; queries_per_kv * kv_len];

        for s in 0..seq_len {
            let valid_len = kv_len.min(s + 1 + kv_len - seq_len);
            if grouped && valid_len >= GQA_BLAS_MIN_KV_LEN {
                for kv_h in 0..n_kv_heads {
                    let query_head = kv_h * queries_per_kv;
                    let query_start = s * n_heads * head_dim + query_head * head_dim;
                    let query_end = query_start + queries_per_kv * head_dim;
                    let score_len = queries_per_kv * valid_len;
                    let cache_start = cache.row_offset(kv_h, 0);

                    crate::ops::sgemm_rhs_transposed(
                        queries_per_kv,
                        head_dim,
                        valid_len,
                        &q_data[query_start..query_end],
                        &k_cached[cache_start..],
                        &mut grouped_scores[..score_len],
                    );
                    for query in 0..queries_per_kv {
                        let row_start = query * valid_len;
                        let row = &mut grouped_scores[row_start..row_start + valid_len];
                        for score in row.iter_mut() {
                            *score *= scale;
                        }
                        let max_score = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                        let exp_sum: f32 = row.iter().map(|score| (*score - max_score).exp()).sum();
                        for score in row.iter_mut() {
                            *score = (*score - max_score).exp() / exp_sum;
                        }
                    }
                    crate::ops::sgemm(
                        queries_per_kv,
                        valid_len,
                        head_dim,
                        &grouped_scores[..score_len],
                        &v_cached[cache_start..],
                        &mut output[query_start..query_end],
                    );
                }
                continue;
            }

            for h in 0..n_heads {
                let kv_h = h * n_kv_heads / n_heads;
                scores[..valid_len].fill(0.0);
                for t in 0..valid_len {
                    let cache_row = cache.row_offset(kv_h, t);
                    for d in 0..head_dim {
                        scores[t] += q_data[s * n_heads * head_dim + h * head_dim + d]
                            * k_cached[cache_row + d];
                    }
                    scores[t] *= scale;
                }
                let max_score = scores[..valid_len]
                    .iter()
                    .cloned()
                    .fold(f32::NEG_INFINITY, f32::max);
                let exp_sum: f32 = scores[..valid_len]
                    .iter()
                    .map(|&s| (s - max_score).exp())
                    .sum();
                for t in 0..valid_len {
                    scores[t] = (scores[t] - max_score).exp() / exp_sum;
                }
                for t in 0..valid_len {
                    let cache_row = cache.row_offset(kv_h, t);
                    for d in 0..head_dim {
                        output[s * n_heads * head_dim + h * head_dim + d] +=
                            scores[t] * v_cached[cache_row + d];
                    }
                }
            }
        }
        Tensor::from_f32_vec(vec![seq_len, n_heads * head_dim], output)
    }

    fn create_kv_cache(
        &self,
        n_layers: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
    ) -> Box<dyn KvCache> {
        Box::new(CpuKVCache::new(n_layers, n_kv_heads, head_dim, max_seq_len))
    }

    fn kv_append(
        &self,
        kv: &mut dyn KvCache,
        layer_idx: usize,
        k: &Tensor,
        v: &Tensor,
        append_len: usize,
    ) -> Result<()> {
        kv.append(layer_idx, k, v, append_len)
    }

    fn synchronize(&self) -> Result<()> {
        Ok(())
    }

    fn begin_capture(&self) -> Result<()> {
        Ok(())
    }

    fn end_capture(&self) -> Result<Box<dyn Graph>> {
        Ok(Box::new(NoopGraph))
    }

    fn device(&self) -> Device {
        Device::Cpu
    }

    fn to_device(&self, tensor: &Tensor) -> Result<Tensor> {
        Ok(tensor.clone())
    }

    fn to_cpu(&self, tensor: &Tensor) -> Result<Tensor> {
        Ok(tensor.clone())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
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
