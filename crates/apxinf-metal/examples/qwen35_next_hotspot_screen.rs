//! Checkpoint-free, short Metal screen for the next Qwen3.5-0.8B decode slice.
//!
//! This intentionally constructs deterministic synthetic matrices with the
//! production projection shapes.  It is not a model benchmark and must not be
//! used as formal performance evidence.

use std::time::Instant;

use apxinf_metal::{MetalW8MatVec, MetalW8MlpBlock, PackedW8Rows};

const HIDDEN: usize = 1024;
const GDN_INPUT_ROWS: usize = 8_224;
const FULL_INPUT_ROWS: usize = 5_120;
const ATTENTION_WIDTH: usize = 2_048;

fn synthetic_weights(elements: usize) -> Vec<f32> {
    (0..elements)
        .map(|index| ((index.wrapping_mul(17) % 255) as f32 - 127.0) / 256.0)
        .collect()
}

fn synthetic_input(columns: usize) -> Vec<f32> {
    (0..columns)
        .map(|index| ((index.wrapping_mul(29) % 127) as f32 - 63.0) / 128.0)
        .collect()
}

fn percentile(sorted: &[u128], numerator: usize, denominator: usize) -> u128 {
    let index = (sorted.len() - 1) * numerator / denominator;
    sorted[index]
}

struct Screen {
    label: &'static str,
    rows: usize,
    columns: usize,
    packed_bytes: usize,
    h2d_bytes: usize,
    d2h_bytes: usize,
    iterations: usize,
    p10_us: f64,
    median_us: f64,
    p90_us: f64,
    min_us: f64,
    max_us: f64,
}

fn screen(label: &'static str, rows: usize, columns: usize, iterations: usize) -> Screen {
    let source = synthetic_weights(rows * columns);
    let packed = PackedW8Rows::pack_f32(&source, rows, columns).expect("pack synthetic W8 rows");
    drop(source);
    let packed_bytes = packed.values().len() + packed.scales().len() * size_of::<f32>();
    let mut projection = MetalW8MatVec::from_packed(&packed).expect("create synthetic Metal lane");
    drop(packed);
    let input = synthetic_input(columns);

    for _ in 0..3 {
        std::hint::black_box(projection.multiply(&input).expect("warmup Metal matvec"));
    }
    let mut samples_ns = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let started = Instant::now();
        std::hint::black_box(projection.multiply(&input).expect("screen Metal matvec"));
        samples_ns.push(started.elapsed().as_nanos());
    }
    samples_ns.sort_unstable();
    Screen {
        label,
        rows,
        columns,
        packed_bytes,
        h2d_bytes: columns * size_of::<f32>(),
        d2h_bytes: rows * size_of::<f32>(),
        iterations,
        p10_us: percentile(&samples_ns, 1, 10) as f64 / 1_000.0,
        median_us: percentile(&samples_ns, 1, 2) as f64 / 1_000.0,
        p90_us: percentile(&samples_ns, 9, 10) as f64 / 1_000.0,
        min_us: samples_ns[0] as f64 / 1_000.0,
        max_us: samples_ns[samples_ns.len() - 1] as f64 / 1_000.0,
    }
}

fn screen_mlp_dispatch_floor(iterations: usize) -> Screen {
    const WIDTH: usize = 64;
    let gate = synthetic_weights(WIDTH * WIDTH);
    let up = synthetic_weights(WIDTH * WIDTH);
    let down = synthetic_weights(WIDTH * WIDTH);
    let mut mlp = MetalW8MlpBlock::from_f32_weights(&gate, &up, &down, WIDTH, WIDTH)
        .expect("create tiny synthetic Metal MLP");
    let input = synthetic_input(WIDTH);
    for _ in 0..3 {
        std::hint::black_box(mlp.forward(&input).expect("warmup tiny Metal MLP"));
    }
    let mut samples_ns = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let started = Instant::now();
        std::hint::black_box(mlp.forward(&input).expect("screen tiny Metal MLP"));
        samples_ns.push(started.elapsed().as_nanos());
    }
    samples_ns.sort_unstable();
    Screen {
        label: "three_encoder_mlp_dispatch_floor_64x64",
        rows: WIDTH,
        columns: WIDTH,
        packed_bytes: 3 * WIDTH * WIDTH + 3 * WIDTH * size_of::<f32>(),
        h2d_bytes: WIDTH * size_of::<f32>(),
        d2h_bytes: WIDTH * size_of::<f32>(),
        iterations,
        p10_us: percentile(&samples_ns, 1, 10) as f64 / 1_000.0,
        median_us: percentile(&samples_ns, 1, 2) as f64 / 1_000.0,
        p90_us: percentile(&samples_ns, 9, 10) as f64 / 1_000.0,
        min_us: samples_ns[0] as f64 / 1_000.0,
        max_us: samples_ns[samples_ns.len() - 1] as f64 / 1_000.0,
    }
}

fn main() {
    let iterations = std::env::var("APXINF_METAL_SCREEN_ITERATIONS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(15);
    assert!(iterations > 0, "iterations must be positive");

    let cases = vec![
        screen("dispatch_floor_64x64", 64, 64, iterations),
        screen_mlp_dispatch_floor(iterations),
        screen(
            "gdn_fused_input_qkv_z_a_b",
            GDN_INPUT_ROWS,
            HIDDEN,
            iterations,
        ),
        screen("gdn_output_projection", HIDDEN, ATTENTION_WIDTH, iterations),
        screen(
            "full_attention_fused_input_q_gate_k_v",
            FULL_INPUT_ROWS,
            HIDDEN,
            iterations,
        ),
        screen(
            "full_attention_output_projection",
            HIDDEN,
            ATTENTION_WIDTH,
            iterations,
        ),
    ];
    println!("{{");
    println!("  \"format\": \"apxinf-qwen35-next-hotspot-synthetic-screen-v1\",");
    println!("  \"qualification\": \"checkpoint-free synthetic screen; not formal performance evidence\",");
    println!("  \"production_shapes\": {{\"hidden_size\": {HIDDEN}, \"gdn_fused_input_rows\": {GDN_INPUT_ROWS}, \"full_attention_fused_input_rows\": {FULL_INPUT_ROWS}, \"attention_output_input_width\": {ATTENTION_WIDTH}}},");
    println!("  \"cases\": [");
    for (index, case) in cases.iter().enumerate() {
        println!(
            "    {{\"label\": \"{}\", \"rows\": {}, \"columns\": {}, \"packed_weight_and_scale_bytes\": {}, \"h2d_bytes_per_call\": {}, \"d2h_bytes_per_call\": {}, \"iterations\": {}, \"p10_us\": {:.3}, \"median_us\": {:.3}, \"p90_us\": {:.3}, \"min_us\": {:.3}, \"max_us\": {:.3}}}{}",
            case.label,
            case.rows,
            case.columns,
            case.packed_bytes,
            case.h2d_bytes,
            case.d2h_bytes,
            case.iterations,
            case.p10_us,
            case.median_us,
            case.p90_us,
            case.min_us,
            case.max_us,
            if index + 1 == cases.len() { "" } else { "," }
        );
    }
    println!("  ]");
    println!("}}");
}
