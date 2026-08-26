use std::hint::black_box;
use std::time::Instant;

use apxinf_core::{Backend, CpuBackend, CpuKVCache, KvCache, Tensor};

const N_LAYERS: usize = 6;
const N_HEADS: usize = 8;
const N_KV_HEADS: usize = 2;
const HEAD_DIM: usize = 256;
const MAX_CONTEXT: usize = 256;
const FIRST_DECODE_KV_LEN: usize = 14;
const LAST_DECODE_KV_LEN: usize = 140;
const REPS_PER_BATCH: usize = 8;
const BATCH_ORDER: [usize; 6] = [0, 1, 1, 0, 0, 1];

fn build_fixture() -> (Vec<Tensor>, CpuKVCache) {
    let mut cache = CpuKVCache::new(N_LAYERS, N_KV_HEADS, HEAD_DIM, MAX_CONTEXT);
    let mut queries = Vec::with_capacity(N_LAYERS);
    for layer in 0..N_LAYERS {
        let q: Vec<f32> = (0..N_HEADS * HEAD_DIM)
            .map(|index| (((layer * N_HEADS * HEAD_DIM + index + 1) as f32) * 0.017).sin())
            .collect();
        let k: Vec<f32> = (0..LAST_DECODE_KV_LEN * N_KV_HEADS * HEAD_DIM)
            .map(|index| {
                (((layer * LAST_DECODE_KV_LEN * N_KV_HEADS * HEAD_DIM + index + 3) as f32) * 0.0013)
                    .cos()
            })
            .collect();
        let v: Vec<f32> = (0..LAST_DECODE_KV_LEN * N_KV_HEADS * HEAD_DIM)
            .map(|index| {
                (((layer * LAST_DECODE_KV_LEN * N_KV_HEADS * HEAD_DIM + index + 5) as f32) * 0.0019)
                    .sin()
            })
            .collect();
        queries.push(Tensor::from_f32(vec![1, N_HEADS, HEAD_DIM], &q).unwrap());
        let k = Tensor::from_f32(vec![LAST_DECODE_KV_LEN, N_KV_HEADS, HEAD_DIM], &k).unwrap();
        let v = Tensor::from_f32(vec![LAST_DECODE_KV_LEN, N_KV_HEADS, HEAD_DIM], &v).unwrap();
        cache.append(layer, &k, &v, LAST_DECODE_KV_LEN).unwrap();
    }
    cache.advance(LAST_DECODE_KV_LEN);
    (queries, cache)
}

fn trajectory(backend: &CpuBackend, queries: &[Tensor], cache: &mut CpuKVCache) {
    for kv_len in FIRST_DECODE_KV_LEN..=LAST_DECODE_KV_LEN {
        for (layer, query) in queries.iter().enumerate() {
            black_box(
                backend
                    .sdpa_decode(
                        query,
                        cache,
                        layer,
                        N_HEADS,
                        N_KV_HEADS,
                        HEAD_DIM,
                        kv_len,
                        MAX_CONTEXT,
                    )
                    .unwrap(),
            );
        }
    }
}

fn output_digest(backend: &CpuBackend, queries: &[Tensor], cache: &mut CpuKVCache) -> u64 {
    let mut digest = 0xcbf29ce484222325u64;
    for kv_len in FIRST_DECODE_KV_LEN..=LAST_DECODE_KV_LEN {
        for (layer, query) in queries.iter().enumerate() {
            let output = backend
                .sdpa_decode(
                    query,
                    cache,
                    layer,
                    N_HEADS,
                    N_KV_HEADS,
                    HEAD_DIM,
                    kv_len,
                    MAX_CONTEXT,
                )
                .unwrap();
            for value in output.as_f32().unwrap() {
                for byte in value.to_bits().to_le_bytes() {
                    digest ^= u64::from(byte);
                    digest = digest.wrapping_mul(0x100000001b3);
                }
            }
        }
    }
    digest
}

fn even_median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    (values[values.len() / 2 - 1] + values[values.len() / 2]) * 0.5
}

fn main() {
    assert!(
        cfg!(feature = "accelerate"),
        "qwen35_sdpa_decode_trajectory_screen requires --features accelerate"
    );
    let backend = CpuBackend;
    let (queries, mut cache) = build_fixture();
    let digest = output_digest(&backend, &queries, &mut cache);

    trajectory(&backend, &queries, &mut cache);
    let calls_per_trajectory = (LAST_DECODE_KV_LEN - FIRST_DECODE_KV_LEN + 1) * N_LAYERS;
    let mut batch_ns_per_call = Vec::with_capacity(BATCH_ORDER.len());
    for _ in BATCH_ORDER {
        let started = Instant::now();
        for _ in 0..REPS_PER_BATCH {
            trajectory(&backend, &queries, &mut cache);
        }
        batch_ns_per_call.push(
            started.elapsed().as_nanos() as f64 / (REPS_PER_BATCH * calls_per_trajectory) as f64,
        );
    }
    let median_ns_per_call = even_median(&mut batch_ns_per_call);
    println!(
        "calls_per_trajectory={calls_per_trajectory} reps_per_batch={REPS_PER_BATCH} \
         median_ns_per_call={median_ns_per_call:.3} output_fnv1a64={digest:016x}"
    );
}
