use apxinf_metal::{
    BodyInputStagingV1, GdnDecodeState, GdnDimensions, GdnF32Weights, MetalW8LinearLayerStack3,
    PackedW8GdnBlock, PackedW8LinearLayerBlock, PackedW8MlpBlock, W8GroupSize,
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

fn fixture(seed: usize) -> (GdnDimensions, PackedW8LinearLayerBlock) {
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
        W8GroupSize::G32,
    )
    .unwrap();
    let intermediate_size = 64;
    let elements = dims.hidden_size * intermediate_size;
    let mlp = PackedW8MlpBlock::pack_f32(
        &values(elements, 41 + seed * 2, 233, 0.04),
        &values(elements, 43 + seed * 2, 229, 0.04),
        &values(elements, 47 + seed * 2, 227, 0.04),
        dims.hidden_size,
        intermediate_size,
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
    let packed = PackedW8LinearLayerBlock::new(gdn, mlp, &input_norm, &post_norm, 1.0e-6).unwrap();
    (dims, packed)
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

#[test]
fn body_input_staging_v1_selector_and_function_name_contract_is_stable() {
    assert_eq!(
        BodyInputStagingV1::default(),
        BodyInputStagingV1::LegacyDevice
    );
    assert_eq!(BodyInputStagingV1::LegacyDevice.selector(), 0);
    assert_eq!(BodyInputStagingV1::ThreadgroupShared.selector(), 1);
    assert_eq!(
        BodyInputStagingV1::try_from(0).unwrap(),
        BodyInputStagingV1::LegacyDevice
    );
    assert_eq!(
        BodyInputStagingV1::try_from(1).unwrap(),
        BodyInputStagingV1::ThreadgroupShared
    );
    assert!(BodyInputStagingV1::try_from(2).is_err());
    assert!(BodyInputStagingV1::try_from(u32::MAX).is_err());

    assert_eq!(
        [
            BodyInputStagingV1::LegacyDevice.expected_gdn_input_function_name(),
            BodyInputStagingV1::LegacyDevice.expected_gdn_output_function_name(),
            BodyInputStagingV1::LegacyDevice.expected_mlp_gate_up_function_name(),
            BodyInputStagingV1::LegacyDevice.expected_mlp_down_function_name(),
        ],
        [
            "gdn_w8_input_projection",
            "gdn_w8_output_projection_g32",
            "w8_mlp_gate_up",
            "w8_mlp_down",
        ]
    );
    assert_eq!(
        [
            BodyInputStagingV1::ThreadgroupShared.expected_gdn_input_function_name(),
            BodyInputStagingV1::ThreadgroupShared.expected_gdn_output_function_name(),
            BodyInputStagingV1::ThreadgroupShared.expected_mlp_gate_up_function_name(),
            BodyInputStagingV1::ThreadgroupShared.expected_mlp_down_function_name(),
        ],
        [
            "gdn_w8_input_projection_tg_shared",
            "gdn_w8_output_projection_g32_tg_shared",
            "w8_mlp_gate_up_tg_shared",
            "w8_mlp_down_tg_shared",
        ]
    );
}

#[cfg(target_os = "macos")]
#[test]
fn stack3_v1_matches_three_sequential_packed_cpu_layers_in_one_transaction() {
    let (dims, layer0) = fixture(0);
    let (_, layer1) = fixture(1);
    let (_, layer2) = fixture(2);
    let layers = [&layer0, &layer1, &layer2];
    let initial = std::array::from_fn(|slot| nonzero_state(dims, slot));
    let hidden = values(dims.hidden_size, 53, 211, 0.8);

    let mut expected_hidden = hidden.clone();
    let mut expected_states = initial.clone();
    for index in 0..3 {
        let result = layers[index]
            .decode_reference(&expected_hidden, &expected_states[index])
            .unwrap();
        expected_hidden = result.output;
        expected_states[index] = result.state;
    }

    let mut stack = MetalW8LinearLayerStack3::from_packed_gdn_out_g32_v1(layers).unwrap();
    stack.seed_decode_states(&initial).unwrap();
    let actual = stack.decode(&hidden).unwrap().to_vec();

    assert_close(&actual, &expected_hidden, 3.0e-3, "stack output");
    let actual_states = stack.state_snapshots().unwrap();
    for index in 0..3 {
        assert_close(
            actual_states[index].query_conv(),
            expected_states[index].query_conv(),
            1.0e-5,
            "stack query conv",
        );
        assert_close(
            actual_states[index].key_conv(),
            expected_states[index].key_conv(),
            1.0e-5,
            "stack key conv",
        );
        assert_close(
            actual_states[index].value_conv(),
            expected_states[index].value_conv(),
            1.0e-5,
            "stack value conv",
        );
        assert_close(
            actual_states[index].recurrent(),
            expected_states[index].recurrent(),
            1.0e-3,
            "stack recurrent",
        );
    }
    let stats = stack.stats();
    assert_eq!(stats.decode_calls, 1);
    assert_eq!(stats.successful_decodes, 1);
    assert_eq!(stats.failed_decodes, 0);
    assert_eq!(stats.command_buffers, 1);
    assert_eq!(stats.compute_encoders, 3);
    assert_eq!(stats.commits, 1);
    assert_eq!(stats.waits, 1);
    assert_eq!(stats.state_commits, 3);
    assert_eq!(stats.last_state_commit_mask, 0b111);
    assert_eq!(stats.committed_stack_version, 1);
}

#[cfg(target_os = "macos")]
#[test]
fn stack3_v1_threadgroup_input_staging_is_bit_exact_state_exact_and_live_receipted() {
    let (dims, layer0) = fixture(0);
    let (_, layer1) = fixture(1);
    let (_, layer2) = fixture(2);
    let layers = [&layer0, &layer1, &layer2];
    let initial = std::array::from_fn(|slot| nonzero_state(dims, slot));
    let mut hidden = values(dims.hidden_size, 53, 211, 0.8);

    let mut legacy = MetalW8LinearLayerStack3::from_packed_gdn_out_g32_v1(layers).unwrap();
    let mut candidate =
        MetalW8LinearLayerStack3::from_packed_gdn_out_g32_with_body_input_staging_v1(
            layers,
            BodyInputStagingV1::ThreadgroupShared,
        )
        .unwrap();

    let legacy_receipt = legacy.body_input_staging_runtime_receipt_v1();
    assert_eq!(
        legacy_receipt.requested_profile,
        BodyInputStagingV1::LegacyDevice
    );
    assert_eq!(
        legacy_receipt.observed_profile,
        legacy_receipt.requested_profile
    );
    assert_eq!(legacy_receipt.threads_per_threadgroup, 256);
    assert_eq!(
        [
            legacy_receipt.gdn_input_threadgroup_bytes,
            legacy_receipt.gdn_output_threadgroup_bytes,
            legacy_receipt.mlp_gate_up_threadgroup_bytes,
            legacy_receipt.mlp_down_threadgroup_bytes,
        ],
        [0, 0, 0, 0]
    );
    assert_eq!(legacy_receipt.pipeline_thread_execution_width, 32);
    assert_eq!(legacy_receipt.threadgroup_barriers_per_projection, 0);
    assert_eq!(
        legacy_receipt.gdn_input_function_name,
        "gdn_w8_input_projection"
    );
    assert_eq!(
        legacy_receipt.gdn_output_function_name,
        "gdn_w8_output_projection_g32"
    );
    assert_eq!(legacy_receipt.mlp_gate_up_function_name, "w8_mlp_gate_up");
    assert_eq!(legacy_receipt.mlp_down_function_name, "w8_mlp_down");

    let candidate_receipt = candidate.body_input_staging_runtime_receipt_v1();
    assert_eq!(
        candidate_receipt.requested_profile,
        BodyInputStagingV1::ThreadgroupShared
    );
    assert_eq!(
        candidate_receipt.observed_profile,
        candidate_receipt.requested_profile
    );
    assert_eq!(candidate_receipt.threads_per_threadgroup, 256);
    assert_eq!(
        [
            candidate_receipt.gdn_input_threadgroup_bytes,
            candidate_receipt.gdn_output_threadgroup_bytes,
            candidate_receipt.mlp_gate_up_threadgroup_bytes,
            candidate_receipt.mlp_down_threadgroup_bytes,
        ],
        [256, 256, 256, 256]
    );
    assert_eq!(candidate_receipt.pipeline_thread_execution_width, 32);
    assert_eq!(candidate_receipt.threadgroup_barriers_per_projection, 1);
    assert_eq!(
        candidate_receipt.gdn_input_function_name,
        "gdn_w8_input_projection_tg_shared"
    );
    assert_eq!(
        candidate_receipt.gdn_output_function_name,
        "gdn_w8_output_projection_g32_tg_shared"
    );
    assert_eq!(
        candidate_receipt.mlp_gate_up_function_name,
        "w8_mlp_gate_up_tg_shared"
    );
    assert_eq!(
        candidate_receipt.mlp_down_function_name,
        "w8_mlp_down_tg_shared"
    );

    assert_eq!(candidate.buffer_ledger(), legacy.buffer_ledger());
    legacy.seed_decode_states(&initial).unwrap();
    candidate.seed_decode_states(&initial).unwrap();
    for _ in 0..2 {
        let legacy_output = legacy.decode(&hidden).unwrap().to_vec();
        let candidate_output = candidate.decode(&hidden).unwrap().to_vec();
        assert_eq!(
            legacy_output
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            candidate_output
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        );
        assert_eq!(
            legacy.state_snapshots().unwrap(),
            candidate.state_snapshots().unwrap()
        );
        assert_eq!(legacy.stats(), candidate.stats());
        hidden = legacy_output;
    }
}

#[cfg(all(target_os = "macos", debug_assertions))]
#[test]
fn stack3_v1_fault_is_atomic_and_terminal_until_clear() {
    let (dims, layer0) = fixture(0);
    let (_, layer1) = fixture(1);
    let (_, layer2) = fixture(2);
    let layers = [&layer0, &layer1, &layer2];
    let initial = std::array::from_fn(|_| GdnDecodeState::zeroed(dims).unwrap());
    let hidden = values(dims.hidden_size, 53, 211, 0.8);
    for profile in [
        BodyInputStagingV1::LegacyDevice,
        BodyInputStagingV1::ThreadgroupShared,
    ] {
        let mut stack =
            MetalW8LinearLayerStack3::from_packed_gdn_out_g32_with_body_input_staging_v1(
                layers, profile,
            )
            .unwrap();
        stack.seed_decode_states(&initial).unwrap();

        let error = stack
            .inject_failure_after_scratch_execution_for_testing(&hidden)
            .unwrap_err();

        assert!(error.to_string().contains("injected"));
        assert_eq!(stack.state_snapshots().unwrap(), initial);
        let failed = stack.stats();
        assert_eq!(failed.decode_calls, 1);
        assert_eq!(failed.successful_decodes, 0);
        assert_eq!(failed.failed_decodes, 1);
        assert_eq!(failed.command_buffers, 1);
        assert_eq!(failed.compute_encoders, 3);
        assert_eq!(failed.commits, 1);
        assert_eq!(failed.waits, 1);
        assert_eq!(failed.state_commits, 0);
        assert_eq!(failed.last_state_commit_mask, 0);
        assert_eq!(failed.committed_stack_version, 0);
        assert!(failed.terminal_error);

        let retry = stack.decode(&hidden).unwrap_err();
        assert!(retry.to_string().contains("terminal"));
        assert_eq!(stack.stats(), failed, "terminal retry must submit no work");
        stack.clear_decode_states().unwrap();
        assert_eq!(stack.stats(), Default::default());
        assert!(stack
            .decode(&hidden)
            .unwrap_err()
            .to_string()
            .contains("seeded"));
        stack.seed_decode_states(&initial).unwrap();
        stack.decode(&hidden).unwrap();
        assert_eq!(stack.stats().state_commits, 3);
        assert_eq!(stack.stats().last_state_commit_mask, 0b111);
    }
}

#[cfg(target_os = "macos")]
#[test]
fn stack3_v1_ledger_counts_only_the_shared_resident_resources_and_boundary_transfers() {
    let (dims, layer0) = fixture(0);
    let (_, layer1) = fixture(1);
    let (_, layer2) = fixture(2);
    let layers = [&layer0, &layer1, &layer2];
    let individual = layers.map(|layer| layer.buffer_ledger().unwrap());
    let stack = MetalW8LinearLayerStack3::from_packed_gdn_out_g32_v1(layers).unwrap();
    let ledger = stack.buffer_ledger();

    assert_eq!(ledger.allocated_buffers, 76);
    assert_eq!(ledger.shared_buffers, 68);
    assert_eq!(ledger.private_buffers, 8);
    assert_eq!(
        ledger.packed_weight_bytes,
        individual.iter().map(|item| item.packed_weight_bytes).sum()
    );
    assert_eq!(
        ledger.packed_scale_bytes,
        individual.iter().map(|item| item.packed_scale_bytes).sum()
    );
    assert_eq!(
        ledger.f32_parameter_bytes,
        individual.iter().map(|item| item.f32_parameter_bytes).sum()
    );
    assert_eq!(
        ledger.active_state_bytes,
        individual.iter().map(|item| item.active_state_bytes).sum()
    );
    assert_eq!(
        ledger.scratch_state_bytes,
        individual.iter().map(|item| item.scratch_state_bytes).sum()
    );
    assert_eq!(ledger.activation_bytes, individual[0].activation_bytes);
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
    assert_eq!(ledger.compute_encoders_per_decode, 3);
    assert_eq!(ledger.commits_per_decode, 1);
    assert_eq!(ledger.waits_per_decode, 1);
    assert_eq!(ledger.intermediate_host_finite_checks_per_decode, 0);
    assert_eq!(ledger.final_output_finite_checks_per_decode, 1);
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
