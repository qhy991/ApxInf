//! Synthetic same-binary candidate screen for Qwen3.5 decode MLP gate+up.
//!
//! This is a primitive screen, not an end-to-end or formal admission result.

use std::error::Error;
use std::hint::black_box;
use std::time::Instant;

use apxinf_core::{Backend, CpuBackend, Tensor};
use apxinf_metal::{MetalW8MatVec, PackedW8Rows, W8_GROUP_SIZE};
use serde_json::json;

const HIDDEN: usize = 1024;
const INTERMEDIATE: usize = 3584;
const OUTPUT: usize = 2 * INTERMEDIATE;

fn main() -> Result<(), Box<dyn Error>> {
    let iterations = std::env::args()
        .nth(1)
        .map(|value| value.parse::<usize>())
        .transpose()?
        .unwrap_or(20);
    if iterations == 0 {
        return Err("iterations must be greater than zero".into());
    }

    let input: Vec<f32> = (0..HIDDEN)
        .map(|index| (((index * 29 + 5) % 101) as f32 - 50.0) * 0.006)
        .collect();
    let hf_rows: Vec<f32> = (0..OUTPUT * HIDDEN)
        .map(|index| {
            let value = ((index * 37 + index / 11 + 17) % 251) as f32 - 125.0;
            value * 0.00017
        })
        .collect();
    let packed = PackedW8Rows::pack_f32(&hf_rows, OUTPUT, HIDDEN)?;
    let quantized_oracle = packed.scores(&input)?;
    let mut metal = MetalW8MatVec::from_packed(&packed)?;

    let gate = Tensor::from_f32(
        vec![HIDDEN, INTERMEDIATE],
        &transpose_rows(&hf_rows[..INTERMEDIATE * HIDDEN], INTERMEDIATE, HIDDEN),
    )?;
    let up = Tensor::from_f32(
        vec![HIDDEN, INTERMEDIATE],
        &transpose_rows(&hf_rows[INTERMEDIATE * HIDDEN..], INTERMEDIATE, HIDDEN),
    )?;
    let input_tensor = Tensor::from_f32(vec![1, HIDDEN], &input)?;
    let backend = CpuBackend;

    let metal_output = metal.multiply(&input)?;
    let metal_vs_w8 = error_metrics(&metal_output, &quantized_oracle);
    let cpu_output = cpu_gate_up(&backend, &input_tensor, &gate, &up)?;
    let metal_vs_f32 = error_metrics(&metal_output, &cpu_output);

    for _ in 0..3 {
        black_box(cpu_gate_up(&backend, &input_tensor, &gate, &up)?);
        black_box(metal.multiply(&input)?);
    }

    let orders = [
        "cpu", "metal", "metal", "cpu", "metal", "cpu", "cpu", "metal",
    ];
    let mut cpu_ns = Vec::new();
    let mut metal_ns = Vec::new();
    let mut blocks = Vec::new();
    for implementation in orders {
        let started = Instant::now();
        let mut checksum = 0.0f32;
        for _ in 0..iterations {
            let output = if implementation == "cpu" {
                cpu_gate_up(&backend, &input_tensor, &gate, &up)?
            } else {
                metal.multiply(&input)?
            };
            checksum += output[0] + output[INTERMEDIATE] + output[OUTPUT - 1];
            black_box(&output);
        }
        let elapsed_ns = started.elapsed().as_nanos();
        let mean_ns = elapsed_ns as f64 / iterations as f64;
        if implementation == "cpu" {
            cpu_ns.push(mean_ns);
        } else {
            metal_ns.push(mean_ns);
        }
        blocks.push(json!({
            "implementation": implementation,
            "iterations": iterations,
            "elapsed_ns": elapsed_ns,
            "mean_ns": mean_ns,
            "checksum": checksum,
        }));
    }
    let cpu_median_ns = median(&mut cpu_ns);
    let metal_median_ns = median(&mut metal_ns);
    println!(
        "{}",
        serde_json::to_string(&json!({
            "format": "apxinf-qwen35-metal-w8-body-primitive-screen-v1",
            "qualification": "synthetic same-binary candidate screen only; not formal or end-to-end evidence",
            "shape": {
                "input": [1, HIDDEN],
                "gate_up_weights": [OUTPUT, HIDDEN],
                "output": [1, OUTPUT],
            },
            "quantization": {
                "scheme": "symmetric-int8-per-row-group",
                "group_size": W8_GROUP_SIZE,
            },
            "transfer_per_call_bytes": {
                "host_to_metal": HIDDEN * std::mem::size_of::<f32>(),
                "metal_to_host": OUTPUT * std::mem::size_of::<f32>(),
            },
            "persistent_metal_bytes": OUTPUT * HIDDEN
                + OUTPUT * (HIDDEN / W8_GROUP_SIZE) * std::mem::size_of::<f32>()
                + HIDDEN * std::mem::size_of::<f32>()
                + OUTPUT * std::mem::size_of::<f32>(),
            "correctness": {
                "metal_vs_w8_cpu_oracle": metal_vs_w8,
                "metal_w8_vs_f32_accelerate": metal_vs_f32,
            },
            "alternating_blocks": blocks,
            "summary": {
                "cpu_f32_accelerate_median_ns": cpu_median_ns,
                "metal_w8_median_ns": metal_median_ns,
                "speedup": cpu_median_ns / metal_median_ns,
            }
        }))?
    );
    Ok(())
}

fn transpose_rows(source: &[f32], rows: usize, columns: usize) -> Vec<f32> {
    let mut output = vec![0.0f32; source.len()];
    for row in 0..rows {
        for column in 0..columns {
            output[column * rows + row] = source[row * columns + column];
        }
    }
    output
}

fn cpu_gate_up(
    backend: &CpuBackend,
    input: &Tensor,
    gate: &Tensor,
    up: &Tensor,
) -> Result<Vec<f32>, Box<dyn Error>> {
    let gate_output = backend.matmul(input, gate)?;
    let up_output = backend.matmul(input, up)?;
    let mut output = Vec::with_capacity(OUTPUT);
    output.extend_from_slice(gate_output.as_f32()?);
    output.extend_from_slice(up_output.as_f32()?);
    Ok(output)
}

fn error_metrics(actual: &[f32], expected: &[f32]) -> serde_json::Value {
    let mut max_abs = 0.0f32;
    let mut squared_error = 0.0f64;
    let mut squared_reference = 0.0f64;
    for (&actual, &expected) in actual.iter().zip(expected) {
        let error = (actual - expected).abs();
        max_abs = max_abs.max(error);
        squared_error += (actual as f64 - expected as f64).powi(2);
        squared_reference += (expected as f64).powi(2);
    }
    json!({
        "max_abs": max_abs,
        "nrmse": (squared_error / squared_reference.max(f64::MIN_POSITIVE)).sqrt(),
    })
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    (values[(values.len() - 1) / 2] + values[values.len() / 2]) * 0.5
}
