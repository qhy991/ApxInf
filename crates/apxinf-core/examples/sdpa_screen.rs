use std::hint::black_box;
use std::time::Instant;

use apxinf_core::{Backend, CpuBackend, CpuKVCache, KvCache, Tensor};

const MAX_CONTEXT: usize = 256;

fn scalar(
    q: &[f32],
    cache: &CpuKVCache,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    kv_len: usize,
) -> Vec<f32> {
    let (keys, values) = cache.get_kv(0);
    let mut output = vec![0.0; n_heads * head_dim];
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut scores = vec![0.0; kv_len];
    for head in 0..n_heads {
        let kv_head = head * n_kv_heads / n_heads;
        scores.fill(0.0);
        for time in 0..kv_len {
            let row = cache.row_offset(kv_head, time);
            for dim in 0..head_dim {
                scores[time] += q[head * head_dim + dim] * keys[row + dim];
            }
            scores[time] *= scale;
        }
        let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let denominator: f32 = scores.iter().map(|score| (*score - max_score).exp()).sum();
        for time in 0..kv_len {
            scores[time] = (scores[time] - max_score).exp() / denominator;
        }
        for time in 0..kv_len {
            let row = cache.row_offset(kv_head, time);
            for dim in 0..head_dim {
                output[head * head_dim + dim] += scores[time] * values[row + dim];
            }
        }
    }
    output
}

fn main() {
    assert!(
        cfg!(feature = "accelerate"),
        "sdpa_screen requires --features accelerate"
    );
    let backend = CpuBackend;
    let n_heads = 8;
    let n_kv_heads = 2;
    let head_dim = 256;
    for kv_len in [
        1, 4, 8, 13, 14, 16, 24, 31, 32, 48, 64, 76, 77, 96, 112, 126, 127, 128, 140, 256,
    ] {
        let q_values: Vec<f32> = (0..n_heads * head_dim)
            .map(|i| ((i as f32 + 1.0) * 0.017).sin())
            .collect();
        let k_values: Vec<f32> = (0..kv_len * n_kv_heads * head_dim)
            .map(|i| ((i as f32 + 3.0) * 0.0013).cos())
            .collect();
        let v_values: Vec<f32> = (0..kv_len * n_kv_heads * head_dim)
            .map(|i| ((i as f32 + 5.0) * 0.0019).sin())
            .collect();
        let q = Tensor::from_f32(vec![1, n_heads, head_dim], &q_values).unwrap();
        let k = Tensor::from_f32(vec![kv_len, n_kv_heads, head_dim], &k_values).unwrap();
        let v = Tensor::from_f32(vec![kv_len, n_kv_heads, head_dim], &v_values).unwrap();
        let mut cache = CpuKVCache::new(1, n_kv_heads, head_dim, MAX_CONTEXT);
        cache.append(0, &k, &v, kv_len).unwrap();
        cache.advance(kv_len);
        let grouped_reference = backend
            .sdpa_decode(
                &q,
                &mut cache,
                0,
                n_heads,
                n_kv_heads,
                head_dim,
                kv_len,
                MAX_CONTEXT,
            )
            .unwrap();
        let scalar_reference = scalar(&q_values, &cache, n_heads, n_kv_heads, head_dim, kv_len);
        let max_abs_error = grouped_reference
            .as_f32()
            .unwrap()
            .iter()
            .zip(&scalar_reference)
            .map(|(grouped, scalar)| (grouped - scalar).abs())
            .fold(0.0f32, f32::max);
        assert!(max_abs_error <= 2.0e-5, "max_abs_error={max_abs_error}");
        let reps = if kv_len <= 32 {
            1_000
        } else if kv_len <= 128 {
            500
        } else {
            250
        };
        for _ in 0..5 {
            black_box(
                backend
                    .sdpa_decode(
                        &q,
                        &mut cache,
                        0,
                        n_heads,
                        n_kv_heads,
                        head_dim,
                        kv_len,
                        MAX_CONTEXT,
                    )
                    .unwrap(),
            );
            black_box(scalar(
                &q_values, &cache, n_heads, n_kv_heads, head_dim, kv_len,
            ));
        }
        let mut grouped_ns = Vec::new();
        let mut scalar_ns = Vec::new();
        for order in [0, 1, 1, 0, 0, 1] {
            if order == 0 {
                let start = Instant::now();
                for _ in 0..reps {
                    black_box(
                        backend
                            .sdpa_decode(
                                &q,
                                &mut cache,
                                0,
                                n_heads,
                                n_kv_heads,
                                head_dim,
                                kv_len,
                                MAX_CONTEXT,
                            )
                            .unwrap(),
                    );
                }
                grouped_ns.push(start.elapsed().as_nanos() as f64 / reps as f64);
                let start = Instant::now();
                for _ in 0..reps {
                    black_box(scalar(
                        &q_values, &cache, n_heads, n_kv_heads, head_dim, kv_len,
                    ));
                }
                scalar_ns.push(start.elapsed().as_nanos() as f64 / reps as f64);
            } else {
                let start = Instant::now();
                for _ in 0..reps {
                    black_box(scalar(
                        &q_values, &cache, n_heads, n_kv_heads, head_dim, kv_len,
                    ));
                }
                scalar_ns.push(start.elapsed().as_nanos() as f64 / reps as f64);
                let start = Instant::now();
                for _ in 0..reps {
                    black_box(
                        backend
                            .sdpa_decode(
                                &q,
                                &mut cache,
                                0,
                                n_heads,
                                n_kv_heads,
                                head_dim,
                                kv_len,
                                MAX_CONTEXT,
                            )
                            .unwrap(),
                    );
                }
                grouped_ns.push(start.elapsed().as_nanos() as f64 / reps as f64);
            }
        }
        grouped_ns.sort_by(f64::total_cmp);
        scalar_ns.sort_by(f64::total_cmp);
        let grouped = (grouped_ns[2] + grouped_ns[3]) * 0.5;
        let scalar = (scalar_ns[2] + scalar_ns[3]) * 0.5;
        println!(
            "kv={kv_len} grouped_us={:.3} scalar_us={:.3} speedup={:.3}x max_abs={:.3e}",
            grouped / 1000.0,
            scalar / 1000.0,
            scalar / grouped,
            max_abs_error,
        );
    }
}
