#[cfg(target_os = "macos")]
use apxinf_metal::MetalW8GdnBlock;
use apxinf_metal::{GdnDecodeState, GdnDimensions, GdnF32Weights, PackedW8GdnBlock};

fn values(elements: usize, multiplier: usize, modulus: usize) -> Vec<f32> {
    (0..elements)
        .map(|index| {
            ((index.wrapping_mul(multiplier) % modulus) as f32 - (modulus / 2) as f32)
                / modulus as f32
        })
        .collect()
}

fn fixture() -> (GdnDimensions, Vec<f32>, PackedW8GdnBlock) {
    let dims = GdnDimensions {
        hidden_size: 64,
        key_heads: 2,
        value_heads: 2,
        key_dim: 32,
        value_dim: 32,
        conv_kernel_size: 4,
        rms_norm_eps: 1.0e-6,
    };
    let input_rows = dims.input_projection_rows();
    let input_projection = values(input_rows * dims.hidden_size, 17, 251);
    let output_projection = values(dims.hidden_size * dims.value_width(), 19, 241);
    let conv_weight = values(dims.qkv_width() * dims.conv_kernel_size, 23, 127);
    let a_log = values(dims.value_heads, 29, 97);
    let dt_bias = values(dims.value_heads, 31, 89);
    let norm_weight = values(dims.value_dim, 37, 83);
    let packed = PackedW8GdnBlock::pack_f32(
        dims,
        GdnF32Weights {
            input_projection: &input_projection,
            output_projection: &output_projection,
            conv_weight: &conv_weight,
            a_log: &a_log,
            dt_bias: &dt_bias,
            norm_weight: &norm_weight,
        },
    )
    .unwrap();
    let input = values(dims.hidden_size, 41, 79);
    (dims, input, packed)
}

#[test]
fn packed_gdn_oracle_advances_canonical_state_without_mutating_its_input() {
    let (dims, input, packed) = fixture();
    let initial = GdnDecodeState::zeroed(dims).unwrap();
    let preserved = initial.clone();

    let first = packed.decode_reference(&input, &initial).unwrap();
    assert_eq!(initial, preserved);
    assert_eq!(first.output.len(), dims.hidden_size);
    assert!(first.output.iter().all(|value| value.is_finite()));
    assert_ne!(first.state, initial);

    let second = packed.decode_reference(&input, &first.state).unwrap();
    assert_ne!(second.output, first.output);
    assert_ne!(second.state, first.state);
}

#[test]
fn overflowing_gdn_dimensions_fail_closed_instead_of_panicking() {
    let dims = GdnDimensions {
        hidden_size: 64,
        key_heads: usize::MAX,
        value_heads: usize::MAX,
        key_dim: 2,
        value_dim: 64,
        conv_kernel_size: 4,
        rms_norm_eps: 1.0e-6,
    };
    assert!(GdnDecodeState::zeroed(dims).is_err());
}

#[cfg(target_os = "macos")]
#[test]
fn metal_gdn_decode_matches_the_packed_oracle_and_commits_once() {
    let (dims, input, packed) = fixture();
    let initial = GdnDecodeState::zeroed(dims).unwrap();
    let expected = packed.decode_reference(&input, &initial).unwrap();

    let mut metal = MetalW8GdnBlock::from_packed(&packed).unwrap();
    metal.seed_decode_state(&initial).unwrap();
    let actual = metal.decode(&input).unwrap().to_vec();
    assert_close(&actual, &expected.output, 3.0e-4, "output");
    let actual_state = metal.state_snapshot().unwrap();
    assert_close(
        actual_state.query_conv(),
        expected.state.query_conv(),
        1.0e-6,
        "query conv state",
    );
    assert_close(
        actual_state.key_conv(),
        expected.state.key_conv(),
        1.0e-6,
        "key conv state",
    );
    assert_close(
        actual_state.value_conv(),
        expected.state.value_conv(),
        1.0e-6,
        "value conv state",
    );
    assert_close(
        actual_state.recurrent(),
        expected.state.recurrent(),
        3.0e-4,
        "recurrent state",
    );
    let stats = metal.stats();
    assert_eq!(stats.decode_calls, 1);
    assert_eq!(stats.command_buffers, 1);
    assert_eq!(stats.waits, 1);
    assert_eq!(stats.committed_state_version, 1);
}

#[cfg(target_os = "macos")]
#[test]
fn metal_gdn_preserves_nonzero_prefill_state_across_two_decode_steps() {
    let (dims, input, packed) = fixture();
    let initial = GdnDecodeState::from_parts(
        dims,
        values(dims.conv_kernel_size * dims.key_width(), 43, 257),
        values(dims.conv_kernel_size * dims.key_width(), 47, 263),
        values(dims.conv_kernel_size * dims.value_width(), 53, 269),
        values(dims.value_heads * dims.key_dim * dims.value_dim, 59, 271),
    )
    .unwrap();
    let expected_first = packed.decode_reference(&input, &initial).unwrap();
    let second_input = values(dims.hidden_size, 61, 73);
    let expected_second = packed
        .decode_reference(&second_input, &expected_first.state)
        .unwrap();

    let mut metal = MetalW8GdnBlock::from_packed(&packed).unwrap();
    metal.seed_decode_state(&initial).unwrap();
    assert_close(
        metal.decode(&input).unwrap(),
        &expected_first.output,
        5.0e-4,
        "first output",
    );
    assert_close(
        metal.decode(&second_input).unwrap(),
        &expected_second.output,
        7.0e-4,
        "second output",
    );
    let state = metal.state_snapshot().unwrap();
    assert_close(
        state.query_conv(),
        expected_second.state.query_conv(),
        1.0e-6,
        "second query conv",
    );
    assert_close(
        state.recurrent(),
        expected_second.state.recurrent(),
        7.0e-4,
        "second recurrent",
    );
    assert_eq!(metal.stats().decode_calls, 2);
    assert_eq!(metal.stats().command_buffers, 2);
    assert_eq!(metal.stats().waits, 2);
    assert_eq!(metal.stats().committed_state_version, 2);
}

#[cfg(target_os = "macos")]
#[test]
fn metal_gdn_clear_requires_a_fresh_prefill_seed_and_resets_receipts() {
    let (dims, input, packed) = fixture();
    let initial = GdnDecodeState::zeroed(dims).unwrap();
    let mut metal = MetalW8GdnBlock::from_packed(&packed).unwrap();
    metal.seed_decode_state(&initial).unwrap();
    metal.decode(&input).unwrap();

    metal.clear_decode_state().unwrap();

    assert_eq!(metal.stats(), apxinf_metal::GdnMetalStats::default());
    assert!(metal.state_snapshot().is_err());
    assert!(metal.decode(&input).is_err());
    metal.seed_decode_state(&initial).unwrap();
    assert!(metal.decode(&input).is_ok());
}

#[cfg(all(target_os = "macos", debug_assertions))]
#[test]
fn exposed_diagnostic_failure_executes_scratch_without_committing() {
    let (dims, input, packed) = fixture();
    let initial = GdnDecodeState::zeroed(dims).unwrap();
    let mut metal = MetalW8GdnBlock::from_packed(&packed).unwrap();
    metal.seed_decode_state(&initial).unwrap();
    let before = metal.state_snapshot().unwrap();

    let error = metal
        .inject_failure_after_scratch_execution_for_testing(&input)
        .unwrap_err();

    assert!(error.to_string().contains("injected"));
    assert_eq!(metal.state_snapshot().unwrap(), before);
    assert_eq!(metal.stats(), apxinf_metal::GdnMetalStats::default());
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "explicit checkpoint-free production-shape ABI gate"]
fn production_shape_metal_gdn_matches_the_packed_oracle_for_one_decode() {
    let dims = GdnDimensions {
        hidden_size: 1024,
        key_heads: 16,
        value_heads: 16,
        key_dim: 128,
        value_dim: 128,
        conv_kernel_size: 4,
        rms_norm_eps: 1.0e-6,
    };
    let scaled = |elements, multiplier, modulus, scale: f32| {
        values(elements, multiplier, modulus)
            .into_iter()
            .map(|value| value * scale)
            .collect::<Vec<_>>()
    };
    let input_projection = scaled(
        dims.input_projection_rows() * dims.hidden_size,
        17,
        251,
        0.02,
    );
    let output_projection = scaled(dims.hidden_size * dims.value_width(), 19, 241, 0.02);
    let conv_weight = scaled(dims.qkv_width() * dims.conv_kernel_size, 23, 127, 0.05);
    let a_log = scaled(dims.value_heads, 29, 97, 0.1);
    let dt_bias = scaled(dims.value_heads, 31, 89, 0.1);
    let norm_weight = scaled(dims.value_dim, 37, 83, 0.2);
    let packed = PackedW8GdnBlock::pack_f32(
        dims,
        GdnF32Weights {
            input_projection: &input_projection,
            output_projection: &output_projection,
            conv_weight: &conv_weight,
            a_log: &a_log,
            dt_bias: &dt_bias,
            norm_weight: &norm_weight,
        },
    )
    .unwrap();
    let input = scaled(dims.hidden_size, 41, 79, 0.5);
    let initial = GdnDecodeState::zeroed(dims).unwrap();
    let expected = packed.decode_reference(&input, &initial).unwrap();

    let mut metal = MetalW8GdnBlock::from_packed(&packed).unwrap();
    metal.seed_decode_state(&initial).unwrap();
    let actual = metal.decode(&input).unwrap();
    assert_close(actual, &expected.output, 3.0e-4, "production output");
    let state = metal.state_snapshot().unwrap();
    assert_close(
        state.recurrent(),
        expected.state.recurrent(),
        1.0e-3,
        "production recurrent",
    );
    assert_eq!(metal.stats().command_buffers, 1);
    assert_eq!(metal.stats().waits, 1);
}

#[cfg(target_os = "macos")]
#[test]
fn metal_gdn_rejects_unseeded_or_invalid_decode_without_state_change() {
    let (dims, input, packed) = fixture();
    let initial = GdnDecodeState::zeroed(dims).unwrap();
    let mut metal = MetalW8GdnBlock::from_packed(&packed).unwrap();
    assert!(metal
        .decode(&input)
        .unwrap_err()
        .to_string()
        .contains("seeded"));
    assert_eq!(metal.stats().decode_calls, 0);

    metal.seed_decode_state(&initial).unwrap();
    let before = metal.state_snapshot().unwrap();
    assert!(metal.decode(&input[..input.len() - 1]).is_err());
    let mut non_finite = input.clone();
    non_finite[17] = f32::NAN;
    assert!(metal.decode(&non_finite).is_err());
    assert_eq!(metal.state_snapshot().unwrap(), before);
    assert_eq!(metal.stats().decode_calls, 0);
    assert_eq!(metal.stats().committed_state_version, 0);
}

#[cfg(target_os = "macos")]
fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32, label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label} length");
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() <= tolerance,
            "{label}[{index}] Metal={actual} CPU={expected} tolerance={tolerance}"
        );
    }
}
