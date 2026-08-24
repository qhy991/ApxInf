use apxinf_metal::{
    GdnDecodeState, GdnDimensions, GdnF32Weights, PackedW8GdnBlock, PackedW8LinearLayerBlock,
    PackedW8MlpBlock, W8GroupSize,
};
#[cfg(target_os = "macos")]
use apxinf_metal::{MetalW8GdnBlock, MetalW8LinearLayerBlock, MetalW8MlpBlock};

fn values(elements: usize, multiplier: usize, modulus: usize, scale: f32) -> Vec<f32> {
    (0..elements)
        .map(|index| {
            (((index.wrapping_mul(multiplier) % modulus) as f32 - (modulus / 2) as f32)
                / modulus as f32)
                * scale
        })
        .collect()
}

fn rms_norm(input: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    let mean_square = input.iter().map(|value| value * value).sum::<f32>() / input.len() as f32;
    let inverse_rms = (mean_square + eps).sqrt().recip();
    input
        .iter()
        .zip(weight)
        .map(|(&value, &weight)| value * inverse_rms * weight)
        .collect()
}

fn fixture() -> (
    GdnDimensions,
    Vec<f32>,
    Vec<f32>,
    Vec<f32>,
    PackedW8GdnBlock,
    PackedW8MlpBlock,
) {
    let dims = GdnDimensions {
        hidden_size: 64,
        key_heads: 2,
        value_heads: 2,
        key_dim: 32,
        value_dim: 32,
        conv_kernel_size: 4,
        rms_norm_eps: 1.0e-6,
    };
    let input_projection = values(
        dims.input_projection_rows() * dims.hidden_size,
        17,
        251,
        0.04,
    );
    let output_projection = values(dims.hidden_size * dims.value_width(), 19, 241, 0.04);
    let conv_weight = values(dims.qkv_width() * dims.conv_kernel_size, 23, 127, 0.1);
    let a_log = values(dims.value_heads, 29, 97, 0.2);
    let dt_bias = values(dims.value_heads, 31, 89, 0.2);
    let gdn_norm_weight = values(dims.value_dim, 37, 83, 0.2);
    let gdn = PackedW8GdnBlock::pack_f32(
        dims,
        GdnF32Weights {
            input_projection: &input_projection,
            output_projection: &output_projection,
            conv_weight: &conv_weight,
            a_log: &a_log,
            dt_bias: &dt_bias,
            norm_weight: &gdn_norm_weight,
        },
    )
    .unwrap();

    let intermediate_size = 64;
    let gate = values(dims.hidden_size * intermediate_size, 41, 233, 0.04);
    let up = values(dims.hidden_size * intermediate_size, 43, 229, 0.04);
    let down = values(dims.hidden_size * intermediate_size, 47, 227, 0.04);
    let mlp =
        PackedW8MlpBlock::pack_f32(&gate, &up, &down, dims.hidden_size, intermediate_size).unwrap();

    let hidden = values(dims.hidden_size, 53, 211, 0.8);
    let input_rms_weight = values(dims.hidden_size, 59, 199, 0.2)
        .into_iter()
        .map(|value| 1.0 + value)
        .collect();
    let post_attention_rms_weight = values(dims.hidden_size, 61, 197, 0.2)
        .into_iter()
        .map(|value| 1.0 + value)
        .collect();
    (
        dims,
        hidden,
        input_rms_weight,
        post_attention_rms_weight,
        gdn,
        mlp,
    )
}

#[test]
fn packed_linear_layer_oracle_applies_both_residuals_and_advances_state() {
    let (dims, hidden, input_norm, post_norm, gdn, mlp) = fixture();
    let initial = GdnDecodeState::zeroed(dims).unwrap();
    let normalized = rms_norm(&hidden, &input_norm, 1.0e-6);
    let attention = gdn.decode_reference(&normalized, &initial).unwrap();
    let post_attention = hidden
        .iter()
        .zip(&attention.output)
        .map(|(&residual, &update)| residual + update)
        .collect::<Vec<_>>();
    let normalized_post_attention = rms_norm(&post_attention, &post_norm, 1.0e-6);
    let mlp_output = mlp.forward(&normalized_post_attention).unwrap();
    let expected = post_attention
        .iter()
        .zip(&mlp_output)
        .map(|(&residual, &update)| residual + update)
        .collect::<Vec<_>>();

    let packed = PackedW8LinearLayerBlock::new(gdn, mlp, &input_norm, &post_norm, 1.0e-6).unwrap();
    let actual = packed.decode_reference(&hidden, &initial).unwrap();

    assert_eq!(actual.output, expected);
    assert_eq!(actual.state, attention.state);
    assert_ne!(actual.state, initial);
}

#[cfg(target_os = "macos")]
#[test]
fn cpu_precision_screen_ledger_is_exact_and_the_legacy_metal_layer_rejects_it() {
    let dims = GdnDimensions {
        hidden_size: 64,
        key_heads: 2,
        value_heads: 2,
        key_dim: 32,
        value_dim: 32,
        conv_kernel_size: 4,
        rms_norm_eps: 1.0e-6,
    };
    let input_projection = values(
        dims.input_projection_rows() * dims.hidden_size,
        17,
        251,
        0.04,
    );
    let output_projection = values(dims.hidden_size * dims.value_width(), 19, 241, 0.04);
    let conv_weight = values(dims.qkv_width() * dims.conv_kernel_size, 23, 127, 0.1);
    let a_log = values(dims.value_heads, 29, 97, 0.2);
    let dt_bias = values(dims.value_heads, 31, 89, 0.2);
    let norm_weight = values(dims.value_dim, 37, 83, 0.2);
    let gdn = PackedW8GdnBlock::pack_f32_with_output_group_size(
        dims,
        GdnF32Weights {
            input_projection: &input_projection,
            output_projection: &output_projection,
            conv_weight: &conv_weight,
            a_log: &a_log,
            dt_bias: &dt_bias,
            norm_weight: &norm_weight,
        },
        W8GroupSize::G32,
    )
    .unwrap();
    let intermediate_size = 64;
    let elements = dims.hidden_size * intermediate_size;
    let gate = values(elements, 41, 233, 0.04);
    let up = values(elements, 43, 229, 0.04);
    let down = values(elements, 47, 227, 0.04);
    let mlp = PackedW8MlpBlock::pack_f32_with_down_group_size(
        &gate,
        &up,
        &down,
        dims.hidden_size,
        intermediate_size,
        W8GroupSize::G32,
    )
    .unwrap();
    let block = PackedW8LinearLayerBlock::new(
        gdn,
        mlp,
        &vec![1.0; dims.hidden_size],
        &vec![1.0; dims.hidden_size],
        dims.rms_norm_eps,
    )
    .unwrap();

    let ledger = block.quantization_ledger().unwrap();
    assert_eq!(ledger.gdn_input_group_size, W8GroupSize::G64);
    assert_eq!(ledger.gdn_output_group_size, W8GroupSize::G32);
    assert_eq!(ledger.mlp_gate_group_size, W8GroupSize::G64);
    assert_eq!(ledger.mlp_up_group_size, W8GroupSize::G64);
    assert_eq!(ledger.mlp_down_group_size, W8GroupSize::G32);
    assert_eq!(
        ledger.gdn_input_scale_bytes,
        dims.input_projection_rows() * 4
    );
    assert_eq!(ledger.gdn_output_scale_bytes, dims.hidden_size * 2 * 4);
    assert_eq!(ledger.mlp_gate_scale_bytes, intermediate_size * 4);
    assert_eq!(ledger.mlp_up_scale_bytes, intermediate_size * 4);
    assert_eq!(ledger.mlp_down_scale_bytes, dims.hidden_size * 2 * 4);
    assert_eq!(
        ledger.total_packed_scale_bytes,
        ledger.gdn_input_scale_bytes
            + ledger.gdn_output_scale_bytes
            + ledger.mlp_gate_scale_bytes
            + ledger.mlp_up_scale_bytes
            + ledger.mlp_down_scale_bytes
    );
    let error = MetalW8LinearLayerBlock::from_packed(&block)
        .err()
        .expect("legacy complete-layer Metal ABI must reject g32");
    assert!(error.to_string().contains("group size 64"));
}

#[cfg(target_os = "macos")]
#[test]
fn metal_gdn_out_g32_v2_matches_the_exact_packed_cpu_oracle_in_one_transaction() {
    let dims = GdnDimensions {
        hidden_size: 64,
        key_heads: 2,
        value_heads: 2,
        key_dim: 32,
        value_dim: 32,
        conv_kernel_size: 4,
        rms_norm_eps: 1.0e-6,
    };
    let input_projection = values(
        dims.input_projection_rows() * dims.hidden_size,
        17,
        251,
        0.04,
    );
    let output_projection = values(dims.hidden_size * dims.value_width(), 19, 241, 0.04);
    let conv_weight = values(dims.qkv_width() * dims.conv_kernel_size, 23, 127, 0.1);
    let a_log = values(dims.value_heads, 29, 97, 0.2);
    let dt_bias = values(dims.value_heads, 31, 89, 0.2);
    let norm_weight = values(dims.value_dim, 37, 83, 0.2);
    let gdn = PackedW8GdnBlock::pack_f32_with_output_group_size(
        dims,
        GdnF32Weights {
            input_projection: &input_projection,
            output_projection: &output_projection,
            conv_weight: &conv_weight,
            a_log: &a_log,
            dt_bias: &dt_bias,
            norm_weight: &norm_weight,
        },
        W8GroupSize::G32,
    )
    .unwrap();
    let intermediate_size = 64;
    let elements = dims.hidden_size * intermediate_size;
    let mlp = PackedW8MlpBlock::pack_f32(
        &values(elements, 41, 233, 0.04),
        &values(elements, 43, 229, 0.04),
        &values(elements, 47, 227, 0.04),
        dims.hidden_size,
        intermediate_size,
    )
    .unwrap();
    let input_norm = vec![1.0; dims.hidden_size];
    let post_norm = vec![1.0; dims.hidden_size];
    let packed = PackedW8LinearLayerBlock::new(gdn, mlp, &input_norm, &post_norm, 1.0e-6).unwrap();
    let initial = GdnDecodeState::zeroed(dims).unwrap();
    let hidden = values(dims.hidden_size, 53, 211, 0.8);
    let expected = packed.decode_reference(&hidden, &initial).unwrap();

    let mut metal = MetalW8LinearLayerBlock::from_packed_gdn_out_g32(&packed).unwrap();
    assert_eq!(
        metal.buffer_ledger().packed_scale_bytes,
        packed
            .quantization_ledger()
            .unwrap()
            .total_packed_scale_bytes
    );
    metal.seed_decode_state(&initial).unwrap();
    let actual = metal.decode(&hidden).unwrap().to_vec();

    assert_close(&actual, &expected.output, 1.0e-3, "GDN-out-G32 v2 output");
    let actual_state = metal.state_snapshot().unwrap();
    assert_close(
        actual_state.recurrent(),
        expected.state.recurrent(),
        1.0e-3,
        "GDN-out-G32 v2 recurrent",
    );
    let stats = metal.stats();
    assert_eq!(stats.decode_calls, 1);
    assert_eq!(stats.successful_decodes, 1);
    assert_eq!(stats.failed_decodes, 0);
    assert_eq!(stats.command_buffers, 1);
    assert_eq!(stats.compute_encoders, 1);
    assert_eq!(stats.commits, 1);
    assert_eq!(stats.waits, 1);
    assert_eq!(stats.committed_state_version, 1);
}

#[cfg(target_os = "macos")]
#[test]
fn metal_gdn_out_g32_v2_rejects_the_legacy_all_g64_packing() {
    let (_, _, input_norm, post_norm, gdn, mlp) = fixture();
    let packed = PackedW8LinearLayerBlock::new(gdn, mlp, &input_norm, &post_norm, 1.0e-6).unwrap();

    let error = MetalW8LinearLayerBlock::from_packed_gdn_out_g32(&packed)
        .err()
        .expect("precision-v2 must not silently accept the legacy all-G64 profile");

    assert!(error.to_string().contains("GDN output projection"));
    assert!(error.to_string().contains("requires group size 32"));
}

#[cfg(target_os = "macos")]
#[test]
fn metal_linear_layer_matches_the_packed_oracle_in_one_state_transaction() {
    let (dims, hidden, input_norm, post_norm, gdn, mlp) = fixture();
    let packed = PackedW8LinearLayerBlock::new(gdn, mlp, &input_norm, &post_norm, 1.0e-6).unwrap();
    let initial = GdnDecodeState::zeroed(dims).unwrap();
    let expected = packed.decode_reference(&hidden, &initial).unwrap();

    let mut metal = MetalW8LinearLayerBlock::from_packed(&packed).unwrap();
    let ledger = metal.buffer_ledger();
    assert_eq!(ledger.allocated_buffers, 32);
    assert_eq!(ledger.shared_buffers, 24);
    assert_eq!(ledger.private_buffers, 8);
    let expected_state_elements = 2 * dims.key_width() * dims.conv_kernel_size
        + dims.value_width() * dims.conv_kernel_size
        + dims.value_heads * dims.key_dim * dims.value_dim;
    assert_eq!(ledger.active_state_bytes, expected_state_elements * 4);
    assert_eq!(ledger.scratch_state_bytes, expected_state_elements * 4);
    assert_eq!(
        ledger.total_persistent_bytes,
        ledger.packed_weight_bytes
            + ledger.packed_scale_bytes
            + ledger.f32_parameter_bytes
            + ledger.active_state_bytes
            + ledger.scratch_state_bytes
            + ledger.activation_bytes
    );
    assert_eq!(ledger.host_input_bytes_per_decode, dims.hidden_size * 4);
    assert_eq!(ledger.host_output_bytes_per_decode, dims.hidden_size * 4);
    assert_eq!(ledger.state_host_transfer_bytes_per_decode, 0);
    assert_eq!(ledger.command_buffers_per_decode, 1);
    assert_eq!(ledger.compute_encoders_per_decode, 1);
    assert_eq!(ledger.commits_per_decode, 1);
    assert_eq!(ledger.waits_per_decode, 1);

    metal.seed_decode_state(&initial).unwrap();
    let actual = metal.decode(&hidden).unwrap().to_vec();
    assert_close(&actual, &expected.output, 1.0e-3, "layer output");
    let actual_state = metal.state_snapshot().unwrap();
    assert_close(
        actual_state.query_conv(),
        expected.state.query_conv(),
        1.0e-5,
        "query conv",
    );
    assert_close(
        actual_state.key_conv(),
        expected.state.key_conv(),
        1.0e-5,
        "key conv",
    );
    assert_close(
        actual_state.value_conv(),
        expected.state.value_conv(),
        1.0e-5,
        "value conv",
    );
    assert_close(
        actual_state.recurrent(),
        expected.state.recurrent(),
        1.0e-3,
        "recurrent",
    );
    let stats = metal.stats();
    assert_eq!(stats.decode_calls, 1);
    assert_eq!(stats.successful_decodes, 1);
    assert_eq!(stats.failed_decodes, 0);
    assert_eq!(stats.command_buffers, 1);
    assert_eq!(stats.compute_encoders, 1);
    assert_eq!(stats.commits, 1);
    assert_eq!(stats.waits, 1);
    assert_eq!(stats.host_to_device_bytes, dims.hidden_size * 4);
    assert_eq!(stats.device_to_host_bytes, dims.hidden_size * 4);
    assert_eq!(stats.committed_state_version, 1);
}

#[cfg(target_os = "macos")]
#[test]
fn fused_layer_matches_the_same_packed_gdn_and_mlp_run_as_staged_primitives() {
    let (dims, hidden, input_norm, post_norm, gdn, mlp) = fixture();
    let initial = GdnDecodeState::zeroed(dims).unwrap();

    let mut staged_gdn = MetalW8GdnBlock::from_packed(&gdn).unwrap();
    staged_gdn.seed_decode_state(&initial).unwrap();
    let normalized = rms_norm(&hidden, &input_norm, 1.0e-6);
    let attention = staged_gdn.decode(&normalized).unwrap().to_vec();
    let staged_state = staged_gdn.state_snapshot().unwrap();
    let post_attention = hidden
        .iter()
        .zip(attention)
        .map(|(&residual, update)| residual + update)
        .collect::<Vec<_>>();
    let normalized_post_attention = rms_norm(&post_attention, &post_norm, 1.0e-6);
    let mut staged_mlp = MetalW8MlpBlock::from_packed(&mlp).unwrap();
    let mlp_output = staged_mlp
        .forward(&normalized_post_attention)
        .unwrap()
        .to_vec();
    let staged_output = post_attention
        .into_iter()
        .zip(mlp_output)
        .map(|(residual, update)| residual + update)
        .collect::<Vec<_>>();

    let packed = PackedW8LinearLayerBlock::new(gdn, mlp, &input_norm, &post_norm, 1.0e-6).unwrap();
    let mut fused = MetalW8LinearLayerBlock::from_packed(&packed).unwrap();
    fused.seed_decode_state(&initial).unwrap();
    let fused_output = fused.decode(&hidden).unwrap().to_vec();
    let fused_state = fused.state_snapshot().unwrap();

    let output_max_abs = max_abs_difference(&fused_output, &staged_output);
    let recurrent_max_abs = max_abs_difference(fused_state.recurrent(), staged_state.recurrent());
    eprintln!(
        "fused-vs-staged output_max_abs={output_max_abs:e} recurrent_max_abs={recurrent_max_abs:e}"
    );
    assert_close(&fused_output, &staged_output, 1.0e-5, "fused/staged output");
    assert_close(
        fused_state.query_conv(),
        staged_state.query_conv(),
        1.0e-5,
        "fused/staged query conv",
    );
    assert_close(
        fused_state.key_conv(),
        staged_state.key_conv(),
        1.0e-5,
        "fused/staged key conv",
    );
    assert_close(
        fused_state.value_conv(),
        staged_state.value_conv(),
        1.0e-5,
        "fused/staged value conv",
    );
    assert_close(
        fused_state.recurrent(),
        staged_state.recurrent(),
        1.0e-5,
        "fused/staged recurrent",
    );
}

#[cfg(all(target_os = "macos", debug_assertions))]
#[test]
fn metal_linear_layer_failure_records_work_but_does_not_commit_state_or_output() {
    let (dims, hidden, input_norm, post_norm, gdn, mlp) = fixture();
    let packed = PackedW8LinearLayerBlock::new(gdn, mlp, &input_norm, &post_norm, 1.0e-6).unwrap();
    let initial = GdnDecodeState::zeroed(dims).unwrap();
    let expected = packed.decode_reference(&hidden, &initial).unwrap();
    let mut metal = MetalW8LinearLayerBlock::from_packed(&packed).unwrap();
    metal.seed_decode_state(&initial).unwrap();

    let error = metal
        .inject_failure_after_scratch_execution_for_testing(&hidden)
        .unwrap_err();

    assert!(error.to_string().contains("injected"));
    assert_eq!(metal.state_snapshot().unwrap(), initial);
    let failed = metal.stats();
    assert_eq!(failed.decode_calls, 1);
    assert_eq!(failed.successful_decodes, 0);
    assert_eq!(failed.failed_decodes, 1);
    assert_eq!(failed.command_buffers, 1);
    assert_eq!(failed.compute_encoders, 1);
    assert_eq!(failed.commits, 1);
    assert_eq!(failed.waits, 1);
    assert_eq!(failed.host_to_device_bytes, dims.hidden_size * 4);
    assert_eq!(failed.device_to_host_bytes, 0);
    assert_eq!(failed.committed_state_version, 0);

    let actual = metal.decode(&hidden).unwrap().to_vec();
    assert_close(&actual, &expected.output, 1.0e-3, "retry output");
    let committed = metal.stats();
    assert_eq!(committed.decode_calls, 2);
    assert_eq!(committed.successful_decodes, 1);
    assert_eq!(committed.failed_decodes, 1);
    assert_eq!(committed.command_buffers, 2);
    assert_eq!(committed.waits, 2);
    assert_eq!(committed.host_to_device_bytes, dims.hidden_size * 8);
    assert_eq!(committed.device_to_host_bytes, dims.hidden_size * 4);
    assert_eq!(committed.committed_state_version, 1);
}

#[cfg(target_os = "macos")]
#[test]
fn metal_linear_layer_reset_is_fail_closed_until_a_fresh_seed() {
    let (dims, hidden, input_norm, post_norm, gdn, mlp) = fixture();
    let packed = PackedW8LinearLayerBlock::new(gdn, mlp, &input_norm, &post_norm, 1.0e-6).unwrap();
    let initial = GdnDecodeState::zeroed(dims).unwrap();
    let expected = packed.decode_reference(&hidden, &initial).unwrap();
    let mut metal = MetalW8LinearLayerBlock::from_packed(&packed).unwrap();
    metal.seed_decode_state(&initial).unwrap();
    metal.decode(&hidden).unwrap();

    metal.clear_decode_state().unwrap();

    assert_eq!(
        metal.stats(),
        apxinf_metal::LinearLayerMetalStats::default()
    );
    assert!(metal.state_snapshot().is_err());
    let error = metal.decode(&hidden).unwrap_err();
    assert!(error.to_string().contains("seeded"));
    assert_eq!(
        metal.stats(),
        apxinf_metal::LinearLayerMetalStats::default()
    );

    metal.seed_decode_state(&initial).unwrap();
    assert_close(
        metal.decode(&hidden).unwrap(),
        &expected.output,
        1.0e-3,
        "reset output",
    );
    assert_eq!(metal.stats().successful_decodes, 1);
    assert_eq!(metal.stats().committed_state_version, 1);
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

#[cfg(target_os = "macos")]
fn max_abs_difference(actual: &[f32], expected: &[f32]) -> f32 {
    actual
        .iter()
        .zip(expected)
        .map(|(&actual, &expected)| (actual - expected).abs())
        .fold(0.0, f32::max)
}
