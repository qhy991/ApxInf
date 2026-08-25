use apxinf_metal::{
    GdnDecodeState, GdnDimensions, GdnF32Weights, MetalW8MlpStack3BoundaryV1, PackedW8GdnBlock,
    PackedW8LinearLayerBlock, PackedW8MlpBlock, PackedW8MlpStack3BoundaryV1, W8GroupSize,
};

fn values(elements: usize, multiplier: usize, modulus: usize, scale: f32) -> Vec<f32> {
    (0..elements)
        .map(|index| {
            (((index.wrapping_mul(multiplier) % modulus) as f32 - (modulus / 2) as f32)
                / modulus as f32)
                * scale
        })
        .collect()
}

fn layer_fixture(seed: usize) -> (GdnDimensions, PackedW8LinearLayerBlock) {
    layer_fixture_with_output_group(seed, W8GroupSize::G32)
}

fn layer_fixture_with_output_group(
    seed: usize,
    output_group_size: W8GroupSize,
) -> (GdnDimensions, PackedW8LinearLayerBlock) {
    let dims = GdnDimensions {
        hidden_size: 64,
        key_heads: 2,
        value_heads: 2,
        key_dim: 32,
        value_dim: 32,
        conv_kernel_size: 4,
        rms_norm_eps: 1.0e-6,
    };
    let gdn = PackedW8GdnBlock::pack_f32_with_output_group_size(
        dims,
        GdnF32Weights {
            input_projection: &values(
                dims.input_projection_rows() * dims.hidden_size,
                17 + seed * 2,
                251,
                0.04,
            ),
            output_projection: &values(
                dims.hidden_size * dims.value_width(),
                19 + seed * 2,
                241,
                0.04,
            ),
            conv_weight: &values(
                dims.qkv_width() * dims.conv_kernel_size,
                23 + seed * 2,
                127,
                0.1,
            ),
            a_log: &values(dims.value_heads, 29 + seed * 2, 97, 0.2),
            dt_bias: &values(dims.value_heads, 31 + seed * 2, 89, 0.2),
            norm_weight: &values(dims.value_dim, 37 + seed * 2, 83, 0.2),
        },
        output_group_size,
    )
    .unwrap();
    let projection_elements = dims.hidden_size * 64;
    let mlp = PackedW8MlpBlock::pack_f32(
        &values(projection_elements, 41 + seed * 2, 233, 0.04),
        &values(projection_elements, 43 + seed * 2, 229, 0.04),
        &values(projection_elements, 47 + seed * 2, 227, 0.04),
        dims.hidden_size,
        64,
    )
    .unwrap();
    let input_norm = values(dims.hidden_size, 59 + seed * 2, 199, 0.2)
        .into_iter()
        .map(|value| 1.0 + value)
        .collect::<Vec<_>>();
    let post_norm = values(dims.hidden_size, 61 + seed * 2, 197, 0.2)
        .into_iter()
        .map(|value| 1.0 + value)
        .collect::<Vec<_>>();
    (
        dims,
        PackedW8LinearLayerBlock::new(gdn, mlp, &input_norm, &post_norm, 1.0e-6).unwrap(),
    )
}

fn boundary_mlp(hidden_size: usize, intermediate_size: usize) -> PackedW8MlpBlock {
    let elements = hidden_size * intermediate_size;
    PackedW8MlpBlock::pack_f32(
        &values(elements, 83, 223, 0.04),
        &values(elements, 89, 211, 0.04),
        &values(elements, 97, 199, 0.04),
        hidden_size,
        intermediate_size,
    )
    .unwrap()
}

fn nonzero_state(dims: GdnDimensions, seed: usize) -> GdnDecodeState {
    GdnDecodeState::from_parts(
        dims,
        values(
            dims.conv_kernel_size * dims.key_width(),
            67 + seed * 2,
            193,
            0.03,
        ),
        values(
            dims.conv_kernel_size * dims.key_width(),
            71 + seed * 2,
            191,
            0.03,
        ),
        values(
            dims.conv_kernel_size * dims.value_width(),
            73 + seed * 2,
            181,
            0.03,
        ),
        values(
            dims.value_heads * dims.key_dim * dims.value_dim,
            79 + seed * 2,
            179,
            0.03,
        ),
    )
    .unwrap()
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

#[test]
fn packed_boundary_v1_matches_the_manual_four_stage_cpu_oracle() {
    let (dims, layer0) = layer_fixture(0);
    let (_, layer1) = layer_fixture(1);
    let (_, layer2) = layer_fixture(2);
    let boundary_mlp = boundary_mlp(dims.hidden_size, 128);
    let boundary_norm = values(dims.hidden_size, 101, 197, 0.2)
        .into_iter()
        .map(|value| 1.0 + value)
        .collect::<Vec<_>>();
    let boundary = PackedW8MlpStack3BoundaryV1::new(
        boundary_mlp.clone(),
        &boundary_norm,
        1.0e-6,
        [layer0.clone(), layer1.clone(), layer2.clone()],
    )
    .unwrap();
    let input = values(dims.hidden_size, 103, 193, 0.8);
    let states = std::array::from_fn(|slot| nonzero_state(dims, slot));

    let normalized = rms_norm(&input, &boundary_norm, 1.0e-6);
    let boundary_update = boundary_mlp.forward(&normalized).unwrap();
    let mut expected = input
        .iter()
        .zip(boundary_update)
        .map(|(&residual, update)| residual + update)
        .collect::<Vec<_>>();
    let mut expected_states = states.clone();
    for (slot, layer) in [&layer0, &layer1, &layer2].into_iter().enumerate() {
        let result = layer
            .decode_reference(&expected, &expected_states[slot])
            .unwrap();
        expected = result.output;
        expected_states[slot] = result.state;
    }

    let actual = boundary.decode_reference(&input, &states).unwrap();

    assert_close(&actual.output, &expected, 0.0, "packed output");
    assert_eq!(actual.states, expected_states);
}

#[cfg(target_os = "macos")]
#[test]
fn metal_boundary_v1_matches_packed_output_and_all_three_states_in_one_transaction() {
    let (dims, layer0) = layer_fixture(0);
    let (_, layer1) = layer_fixture(1);
    let (_, layer2) = layer_fixture(2);
    let boundary_norm = values(dims.hidden_size, 101, 197, 0.2)
        .into_iter()
        .map(|value| 1.0 + value)
        .collect::<Vec<_>>();
    let packed = PackedW8MlpStack3BoundaryV1::new(
        boundary_mlp(dims.hidden_size, 128),
        &boundary_norm,
        1.0e-6,
        [layer0, layer1, layer2],
    )
    .unwrap();
    let input = values(dims.hidden_size, 103, 193, 0.8);
    let states = std::array::from_fn(|slot| nonzero_state(dims, slot));
    let expected = packed.decode_reference(&input, &states).unwrap();
    let expected_ledger = packed.buffer_ledger().unwrap();

    let mut metal = MetalW8MlpStack3BoundaryV1::from_packed(&packed).unwrap();
    assert_eq!(metal.buffer_ledger(), expected_ledger);
    metal.seed_decode_states(&states).unwrap();
    let actual = metal.decode(&input).unwrap().to_vec();

    assert_close(&actual, &expected.output, 3.0e-3, "Metal boundary output");
    let actual_states = metal.state_snapshots().unwrap();
    for slot in 0..3 {
        assert_close(
            actual_states[slot].query_conv(),
            expected.states[slot].query_conv(),
            1.0e-5,
            "Metal boundary query state",
        );
        assert_close(
            actual_states[slot].key_conv(),
            expected.states[slot].key_conv(),
            1.0e-5,
            "Metal boundary key state",
        );
        assert_close(
            actual_states[slot].value_conv(),
            expected.states[slot].value_conv(),
            1.0e-5,
            "Metal boundary value state",
        );
        assert_close(
            actual_states[slot].recurrent(),
            expected.states[slot].recurrent(),
            1.0e-3,
            "Metal boundary recurrent state",
        );
    }
    let stats = metal.stats();
    assert_eq!(stats.decode_calls, 1);
    assert_eq!(stats.successful_decodes, 1);
    assert_eq!(stats.failed_decodes, 0);
    assert_eq!(stats.command_buffers, 1);
    assert_eq!(stats.compute_encoders, 4);
    assert_eq!(stats.commits, 1);
    assert_eq!(stats.waits, 1);
    assert_eq!(stats.state_commits, 3);
    assert_eq!(stats.last_state_commit_mask, 0b111);
    assert_eq!(stats.committed_stack_version, 1);
    assert!(!stats.terminal_error);
}

#[test]
fn boundary_v1_ledger_is_exact_and_counts_only_outer_hidden_transfers() {
    let (dims, layer0) = layer_fixture(0);
    let (_, layer1) = layer_fixture(1);
    let (_, layer2) = layer_fixture(2);
    let boundary_mlp = boundary_mlp(dims.hidden_size, 128);
    let boundary_mlp_ledger = boundary_mlp.buffer_ledger().unwrap();
    let layer_ledgers = [&layer0, &layer1, &layer2].map(|layer| layer.buffer_ledger().unwrap());
    let boundary_norm = vec![1.0; dims.hidden_size];
    let packed = PackedW8MlpStack3BoundaryV1::new(
        boundary_mlp,
        &boundary_norm,
        1.0e-6,
        [layer0, layer1, layer2],
    )
    .unwrap();

    let ledger = packed.buffer_ledger().unwrap();

    assert_eq!(ledger.scope, "resident-mtlbuffer-only");
    assert_eq!(ledger.abi_version, 1);
    assert_eq!(ledger.stack_depth, 3);
    assert_eq!(ledger.allocated_buffers, 81);
    assert_eq!(ledger.shared_buffers, 73);
    assert_eq!(ledger.private_buffers, 8);
    assert_eq!(
        ledger.packed_weight_bytes,
        boundary_mlp_ledger.packed_weight_bytes
            + layer_ledgers
                .iter()
                .map(|item| item.packed_weight_bytes)
                .sum::<usize>()
    );
    assert_eq!(
        ledger.packed_scale_bytes,
        boundary_mlp_ledger.packed_scale_bytes
            + layer_ledgers
                .iter()
                .map(|item| item.packed_scale_bytes)
                .sum::<usize>()
    );
    assert_eq!(
        ledger.f32_parameter_bytes,
        dims.hidden_size * 4
            + layer_ledgers
                .iter()
                .map(|item| item.f32_parameter_bytes)
                .sum::<usize>()
    );
    assert_eq!(
        ledger.active_state_bytes,
        layer_ledgers
            .iter()
            .map(|item| item.active_state_bytes)
            .sum::<usize>()
    );
    assert_eq!(ledger.scratch_state_bytes, ledger.active_state_bytes);
    let activation_elements = 4 * dims.hidden_size
        + dims.input_projection_rows()
        + dims.qkv_width()
        + 2 * dims.value_width()
        + 3 * 128;
    assert_eq!(ledger.activation_bytes, activation_elements * 4);
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
    assert_eq!(ledger.compute_encoders_per_decode, 4);
    assert_eq!(ledger.kernel_dispatches_per_decode, 44);
    assert_eq!(ledger.commits_per_decode, 1);
    assert_eq!(ledger.waits_per_decode, 1);
    assert_eq!(ledger.intermediate_host_finite_checks_per_decode, 0);
    assert_eq!(ledger.final_output_finite_checks_per_decode, 1);
}

#[cfg(all(target_os = "macos", debug_assertions))]
#[test]
fn boundary_v1_scratch_failure_publishes_no_state_or_output_and_reset_recovers() {
    let (dims, layer0) = layer_fixture(0);
    let (_, layer1) = layer_fixture(1);
    let (_, layer2) = layer_fixture(2);
    let packed = PackedW8MlpStack3BoundaryV1::new(
        boundary_mlp(dims.hidden_size, 128),
        &vec![1.0; dims.hidden_size],
        1.0e-6,
        [layer0, layer1, layer2],
    )
    .unwrap();
    let input = values(dims.hidden_size, 103, 193, 0.8);
    let initial = std::array::from_fn(|slot| nonzero_state(dims, slot));
    let mut metal = MetalW8MlpStack3BoundaryV1::from_packed(&packed).unwrap();
    metal.seed_decode_states(&initial).unwrap();

    let error = metal
        .inject_failure_after_scratch_execution_for_testing(&input)
        .unwrap_err();

    assert!(error.to_string().contains("injected"));
    assert_eq!(metal.state_snapshots().unwrap(), initial);
    let failed = metal.stats();
    assert_eq!(failed.decode_calls, 1);
    assert_eq!(failed.successful_decodes, 0);
    assert_eq!(failed.failed_decodes, 1);
    assert_eq!(failed.command_buffers, 1);
    assert_eq!(failed.compute_encoders, 4);
    assert_eq!(failed.commits, 1);
    assert_eq!(failed.waits, 1);
    assert_eq!(failed.host_to_device_bytes, dims.hidden_size * 4);
    assert_eq!(failed.device_to_host_bytes, 0);
    assert_eq!(failed.state_commits, 0);
    assert_eq!(failed.last_state_commit_mask, 0);
    assert_eq!(failed.committed_stack_version, 0);
    assert!(failed.terminal_error);

    let retry = metal.decode(&input).unwrap_err();
    assert!(retry.to_string().contains("terminal"));
    assert_eq!(metal.stats(), failed, "terminal retry must submit no work");

    metal.clear_decode_states().unwrap();
    assert_eq!(metal.stats(), Default::default());
    assert!(metal
        .decode(&input)
        .unwrap_err()
        .to_string()
        .contains("seeded"));
    metal.seed_decode_states(&initial).unwrap();
    metal.decode(&input).unwrap();
    assert_eq!(metal.stats().state_commits, 3);
    assert_eq!(metal.stats().last_state_commit_mask, 0b111);
}

#[test]
fn boundary_v1_rejects_incompatible_shapes_and_group_contracts_before_metal() {
    let (dims, layer0) = layer_fixture(0);
    let (_, layer1) = layer_fixture(1);
    let (_, layer2) = layer_fixture(2);
    let boundary_norm = vec![1.0; dims.hidden_size];

    let wrong_hidden = PackedW8MlpStack3BoundaryV1::new(
        boundary_mlp(128, 128),
        &boundary_norm,
        1.0e-6,
        [layer0.clone(), layer1.clone(), layer2.clone()],
    )
    .unwrap_err();
    assert!(wrong_hidden.to_string().contains("hidden sizes differ"));

    let elements = dims.hidden_size * 128;
    let boundary_down_g32 = PackedW8MlpBlock::pack_f32_with_down_group_size(
        &values(elements, 83, 223, 0.04),
        &values(elements, 89, 211, 0.04),
        &values(elements, 97, 199, 0.04),
        dims.hidden_size,
        128,
        W8GroupSize::G32,
    )
    .unwrap();
    let wrong_boundary_group = PackedW8MlpStack3BoundaryV1::new(
        boundary_down_g32,
        &boundary_norm,
        1.0e-6,
        [layer0.clone(), layer1.clone(), layer2.clone()],
    )
    .unwrap_err();
    assert!(wrong_boundary_group
        .to_string()
        .contains("boundary MLP down projection requires group size 64, got 32"));

    let (_, layer0_g64_output) = layer_fixture_with_output_group(0, W8GroupSize::G64);
    let wrong_stack_group = PackedW8MlpStack3BoundaryV1::new(
        boundary_mlp(dims.hidden_size, 128),
        &boundary_norm,
        1.0e-6,
        [layer0_g64_output, layer1, layer2],
    )
    .unwrap_err();
    assert!(wrong_stack_group
        .to_string()
        .contains("layer 0 GDN output projection requires group size 32, got 64"));
}

fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32, label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label} length");
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() <= tolerance,
            "{label}[{index}] actual={actual} expected={expected} tolerance={tolerance}"
        );
    }
}
