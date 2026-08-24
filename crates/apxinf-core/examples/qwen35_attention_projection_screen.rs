//! Checkpoint-free short screen of the CPU F32 projection bundles that remain
//! in the native Qwen3.5-0.8B decode path.
//!
//! The shapes and call boundaries match the current runtime, but the values
//! are deterministic synthetic data.  This is diagnostic, not formal model
//! performance evidence.

use std::hint::black_box;
use std::time::Instant;

use apxinf_core::{Backend, CpuBackend, Tensor};

const HIDDEN: usize = 1024;
const GDN_WIDTH: usize = 2048;
const GDN_GATE_WIDTH: usize = 16;
const FULL_Q_WIDTH: usize = 2048;
const FULL_KV_WIDTH: usize = 512;
const LINEAR_HEADS: usize = 16;
const LINEAR_HEAD_DIM: usize = 128;

fn values(elements: usize, multiplier: usize, modulus: usize) -> Vec<f32> {
    (0..elements)
        .map(|index| {
            ((index.wrapping_mul(multiplier) % modulus) as f32 - (modulus / 2) as f32)
                / modulus as f32
        })
        .collect()
}

fn matrix(columns: usize, rows: usize, seed: usize) -> Tensor {
    Tensor::from_f32(
        vec![columns, rows],
        &values(columns * rows, seed * 2 + 1, 251),
    )
    .expect("create synthetic projection")
}

fn median_us(mut samples: Vec<u128>) -> f64 {
    samples.sort_unstable();
    samples[samples.len() / 2] as f64 / 1_000.0
}

fn main() {
    let iterations = std::env::var("APXINF_CPU_SCREEN_ITERATIONS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(9);
    assert!(iterations > 0, "iterations must be positive");

    let backend = CpuBackend;
    let hidden = Tensor::from_f32(vec![1, HIDDEN], &values(HIDDEN, 7, 127)).unwrap();
    let gdn_mid = Tensor::from_f32(vec![1, GDN_WIDTH], &values(GDN_WIDTH, 11, 127)).unwrap();
    let full_mid = Tensor::from_f32(vec![1, FULL_Q_WIDTH], &values(FULL_Q_WIDTH, 13, 127)).unwrap();

    let gdn_wide = (0..4)
        .map(|seed| matrix(HIDDEN, GDN_WIDTH, seed + 1))
        .collect::<Vec<_>>();
    let gdn_gate_a = matrix(HIDDEN, GDN_GATE_WIDTH, 7);
    let gdn_gate_b = matrix(HIDDEN, GDN_GATE_WIDTH, 8);
    let gdn_output = matrix(GDN_WIDTH, HIDDEN, 9);

    let full_query = matrix(HIDDEN, FULL_Q_WIDTH, 10);
    let full_gate = matrix(HIDDEN, FULL_Q_WIDTH, 11);
    let full_key = matrix(HIDDEN, FULL_KV_WIDTH, 12);
    let full_value = matrix(HIDDEN, FULL_KV_WIDTH, 13);
    let full_output = matrix(FULL_Q_WIDTH, HIDDEN, 14);

    let recurrent_q = Tensor::from_f32(
        vec![1, LINEAR_HEADS, LINEAR_HEAD_DIM],
        &values(LINEAR_HEADS * LINEAR_HEAD_DIM, 31, 127),
    )
    .unwrap();
    let recurrent_k = Tensor::from_f32(
        vec![1, LINEAR_HEADS, LINEAR_HEAD_DIM],
        &values(LINEAR_HEADS * LINEAR_HEAD_DIM, 37, 127),
    )
    .unwrap();
    let recurrent_v = Tensor::from_f32(
        vec![1, LINEAR_HEADS, LINEAR_HEAD_DIM],
        &values(LINEAR_HEADS * LINEAR_HEAD_DIM, 41, 127),
    )
    .unwrap();
    let recurrent_a =
        Tensor::from_f32(vec![1, LINEAR_HEADS], &values(LINEAR_HEADS, 43, 127)).unwrap();
    let recurrent_b =
        Tensor::from_f32(vec![1, LINEAR_HEADS], &values(LINEAR_HEADS, 47, 127)).unwrap();
    let recurrent_a_log =
        Tensor::from_f32(vec![LINEAR_HEADS], &values(LINEAR_HEADS, 53, 127)).unwrap();
    let recurrent_dt =
        Tensor::from_f32(vec![LINEAR_HEADS], &values(LINEAR_HEADS, 59, 127)).unwrap();
    let recurrent_state = Tensor::from_f32(
        vec![LINEAR_HEADS, LINEAR_HEAD_DIM, LINEAR_HEAD_DIM],
        &values(LINEAR_HEADS * LINEAR_HEAD_DIM * LINEAR_HEAD_DIM, 61, 127),
    )
    .unwrap();

    let run_gdn = || {
        for weight in &gdn_wide {
            black_box(
                backend
                    .matmul(&hidden, weight)
                    .expect("GDN input projection"),
            );
        }
        black_box(
            backend
                .matmul(&hidden, &gdn_gate_a)
                .expect("GDN a projection"),
        );
        black_box(
            backend
                .matmul(&hidden, &gdn_gate_b)
                .expect("GDN b projection"),
        );
        black_box(
            backend
                .matmul(&gdn_mid, &gdn_output)
                .expect("GDN output projection"),
        );
    };
    let run_full = || {
        black_box(
            backend
                .matmul(&hidden, &full_query)
                .expect("full query projection"),
        );
        black_box(
            backend
                .matmul(&hidden, &full_gate)
                .expect("full gate projection"),
        );
        black_box(
            backend
                .matmul(&hidden, &full_key)
                .expect("full key projection"),
        );
        black_box(
            backend
                .matmul(&hidden, &full_value)
                .expect("full value projection"),
        );
        black_box(
            backend
                .matmul(&full_mid, &full_output)
                .expect("full output projection"),
        );
    };
    let run_recurrent = || {
        black_box(
            backend
                .gated_delta_recurrent(
                    &recurrent_q,
                    &recurrent_k,
                    &recurrent_v,
                    &recurrent_a,
                    &recurrent_b,
                    &recurrent_a_log,
                    &recurrent_dt,
                    Some(&recurrent_state),
                )
                .expect("GDN recurrent core"),
        );
    };

    for _ in 0..2 {
        run_gdn();
        run_full();
        run_recurrent();
    }
    let mut gdn_samples = Vec::with_capacity(iterations);
    let mut full_samples = Vec::with_capacity(iterations);
    let mut recurrent_samples = Vec::with_capacity(iterations);
    for order in 0..iterations {
        if order % 2 == 0 {
            let started = Instant::now();
            run_gdn();
            gdn_samples.push(started.elapsed().as_nanos());
            let started = Instant::now();
            run_full();
            full_samples.push(started.elapsed().as_nanos());
            let started = Instant::now();
            run_recurrent();
            recurrent_samples.push(started.elapsed().as_nanos());
        } else {
            let started = Instant::now();
            run_recurrent();
            recurrent_samples.push(started.elapsed().as_nanos());
            let started = Instant::now();
            run_full();
            full_samples.push(started.elapsed().as_nanos());
            let started = Instant::now();
            run_gdn();
            gdn_samples.push(started.elapsed().as_nanos());
        }
    }

    println!("{{");
    println!("  \"format\": \"apxinf-qwen35-cpu-attention-projection-synthetic-screen-v1\",");
    println!("  \"qualification\": \"checkpoint-free synthetic screen; not formal performance evidence\",");
    println!("  \"iterations\": {iterations},");
    println!("  \"gdn_one_layer\": {{\"f32_weight_bytes\": 42074112, \"matmul_calls\": 7, \"median_us\": {:.3}}},", median_us(gdn_samples));
    println!("  \"gdn_recurrent_core_one_layer\": {{\"persistent_state_bytes\": 1048576, \"median_us\": {:.3}}},", median_us(recurrent_samples));
    println!("  \"full_attention_one_layer\": {{\"f32_weight_bytes\": 29360128, \"matmul_calls\": 5, \"median_us\": {:.3}}}", median_us(full_samples));
    println!("}}");
}
