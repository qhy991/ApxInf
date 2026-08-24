use std::hint::black_box;
use std::time::Instant;

use apxinf_core::{Backend, CpuBackend, CpuKVCache, KvCache, Tensor};

fn scalar_prefill(
    q: &[f32],
    cache: &CpuKVCache,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    seq_len: usize,
    kv_len: usize,
) -> Vec<f32> {
    let (keys, values) = cache.get_kv(0);
    let mut output = vec![0.0; seq_len * n_heads * head_dim];
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut scores = vec![0.0; kv_len];

    for sequence in 0..seq_len {
        let valid_len = kv_len.min(sequence + 1 + kv_len - seq_len);
        for head in 0..n_heads {
            let kv_head = head * n_kv_heads / n_heads;
            scores[..valid_len].fill(0.0);
            for time in 0..valid_len {
                let cache_row = cache.row_offset(kv_head, time);
                for dim in 0..head_dim {
                    scores[time] +=
                        q[(sequence * n_heads + head) * head_dim + dim] * keys[cache_row + dim];
                }
                scores[time] *= scale;
            }
            let max_score = scores[..valid_len]
                .iter()
                .copied()
                .fold(f32::NEG_INFINITY, f32::max);
            let denominator: f32 = scores[..valid_len]
                .iter()
                .map(|score| (*score - max_score).exp())
                .sum();
            for time in 0..valid_len {
                scores[time] = (scores[time] - max_score).exp() / denominator;
                let cache_row = cache.row_offset(kv_head, time);
                for dim in 0..head_dim {
                    output[(sequence * n_heads + head) * head_dim + dim] +=
                        scores[time] * values[cache_row + dim];
                }
            }
        }
    }
    output
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    let middle = values.len() / 2;
    if values.len() % 2 == 0 {
        (values[middle - 1] + values[middle]) * 0.5
    } else {
        values[middle]
    }
}

#[allow(clippy::too_many_arguments)]
fn measure_grouped(
    backend: &CpuBackend,
    q: &Tensor,
    cache: &mut CpuKVCache,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    kv_len: usize,
    reps: usize,
) -> f64 {
    let start = Instant::now();
    for _ in 0..reps {
        black_box(
            backend
                .sdpa_prefill(q, cache, 0, n_heads, n_kv_heads, head_dim, kv_len, kv_len)
                .unwrap(),
        );
    }
    start.elapsed().as_nanos() as f64 / reps as f64
}

#[allow(clippy::too_many_arguments)]
fn measure_scalar(
    q: &[f32],
    cache: &CpuKVCache,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    seq_len: usize,
    kv_len: usize,
    reps: usize,
) -> f64 {
    let start = Instant::now();
    for _ in 0..reps {
        black_box(scalar_prefill(
            q, cache, n_heads, n_kv_heads, head_dim, seq_len, kv_len,
        ));
    }
    start.elapsed().as_nanos() as f64 / reps as f64
}

fn main() {
    let backend = CpuBackend;
    let n_heads = 8;
    let n_kv_heads = 2;
    let head_dim = 256;

    for seq_len in [128, 256, 512] {
        let kv_len = seq_len;
        let q_values: Vec<f32> = (0..seq_len * n_heads * head_dim)
            .map(|index| ((index as f32 + 1.0) * 0.0017).sin())
            .collect();
        let k_values: Vec<f32> = (0..kv_len * n_kv_heads * head_dim)
            .map(|index| ((index as f32 + 3.0) * 0.0013).cos())
            .collect();
        let v_values: Vec<f32> = (0..kv_len * n_kv_heads * head_dim)
            .map(|index| ((index as f32 + 5.0) * 0.0019).sin())
            .collect();
        let q = Tensor::from_f32(vec![seq_len, n_heads, head_dim], &q_values).unwrap();
        let k = Tensor::from_f32(vec![kv_len, n_kv_heads, head_dim], &k_values).unwrap();
        let v = Tensor::from_f32(vec![kv_len, n_kv_heads, head_dim], &v_values).unwrap();
        let mut cache = CpuKVCache::new(1, n_kv_heads, head_dim, kv_len);
        cache.append(0, &k, &v, kv_len).unwrap();

        let grouped_reference = backend
            .sdpa_prefill(
                &q, &mut cache, 0, n_heads, n_kv_heads, head_dim, kv_len, kv_len,
            )
            .unwrap();
        let scalar_reference = scalar_prefill(
            &q_values, &cache, n_heads, n_kv_heads, head_dim, seq_len, kv_len,
        );
        let max_abs_error = grouped_reference
            .as_f32()
            .unwrap()
            .iter()
            .zip(&scalar_reference)
            .map(|(grouped, scalar)| (grouped - scalar).abs())
            .fold(0.0f32, f32::max);
        assert!(max_abs_error <= 2.0e-5, "max_abs_error={max_abs_error}");

        let reps = match seq_len {
            128 => 8,
            256 => 3,
            _ => 1,
        };
        black_box(
            backend
                .sdpa_prefill(
                    &q, &mut cache, 0, n_heads, n_kv_heads, head_dim, kv_len, kv_len,
                )
                .unwrap(),
        );
        black_box(scalar_prefill(
            &q_values, &cache, n_heads, n_kv_heads, head_dim, seq_len, kv_len,
        ));

        let mut grouped_ns = Vec::new();
        let mut scalar_ns = Vec::new();
        for grouped_first in [true, false, false, true] {
            if grouped_first {
                grouped_ns.push(measure_grouped(
                    &backend, &q, &mut cache, n_heads, n_kv_heads, head_dim, kv_len, reps,
                ));
                scalar_ns.push(measure_scalar(
                    &q_values, &cache, n_heads, n_kv_heads, head_dim, seq_len, kv_len, reps,
                ));
            } else {
                scalar_ns.push(measure_scalar(
                    &q_values, &cache, n_heads, n_kv_heads, head_dim, seq_len, kv_len, reps,
                ));
                grouped_ns.push(measure_grouped(
                    &backend, &q, &mut cache, n_heads, n_kv_heads, head_dim, kv_len, reps,
                ));
            }
        }
        let grouped = median(&mut grouped_ns);
        let scalar = median(&mut scalar_ns);
        println!(
            "seq={seq_len} grouped_ms={:.3} scalar_ms={:.3} speedup={:.3}x max_abs={:.3e}",
            grouped / 1_000_000.0,
            scalar / 1_000_000.0,
            scalar / grouped,
            max_abs_error,
        );
    }
}
