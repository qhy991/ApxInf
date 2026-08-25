use apxinf_metal::{
    MetalW8TailMlpHeadV1, PackedW8MlpBlock, PackedW8Rows, PackedW8TailMlpHeadV1,
    TailMlpHeadBufferLedgerV1, TailMlpHeadRowsKernelV1,
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
fn packed_tail_v1_matches_the_manual_oracle_and_breaks_head_ties_by_lowest_token() {
    let hidden_size = 64;
    let intermediate_size = 64;
    let vocab_size = 8usize;
    let projection_elements = hidden_size * intermediate_size;
    let mlp = PackedW8MlpBlock::pack_f32(
        &values(projection_elements, 17, 251, 0.04),
        &values(projection_elements, 19, 241, 0.04),
        &values(projection_elements, 23, 239, 0.04),
        hidden_size,
        intermediate_size,
    )
    .unwrap();
    let post_attention_rms_weight = values(hidden_size, 29, 233, 0.2)
        .into_iter()
        .map(|value| 1.0 + value)
        .collect::<Vec<_>>();
    let final_rms_weight = values(hidden_size, 31, 229, 0.2)
        .into_iter()
        .map(|value| 1.0 + value)
        .collect::<Vec<_>>();
    let vocab = PackedW8Rows::pack_f32(
        &vec![0.0; vocab_size * hidden_size],
        vocab_size,
        hidden_size,
    )
    .unwrap();
    let packed = PackedW8TailMlpHeadV1::new(
        mlp.clone(),
        &post_attention_rms_weight,
        &final_rms_weight,
        1.0e-6,
        vocab,
    )
    .unwrap();
    let input = values(hidden_size, 37, 223, 0.8);

    let normalized = rms_norm(&input, &post_attention_rms_weight, 1.0e-6);
    let update = mlp.forward(&normalized).unwrap();
    let residual = input
        .iter()
        .zip(update)
        .map(|(&residual, update)| residual + update)
        .collect::<Vec<_>>();
    let expected_hidden = rms_norm(&residual, &final_rms_weight, 1.0e-6);

    let actual = packed.decode_reference(&input).unwrap();

    assert_eq!(actual.normalized_hidden, expected_hidden);
    assert_eq!(actual.candidate_token_ids, [0, 1, 2, 3]);
}

#[test]
fn tail_v1_ledger_is_exact_for_tiny_and_qwen35_official_dimensions() {
    let hidden_size = 64;
    let intermediate_size = 128;
    let vocab_size = 8;
    let ledger =
        TailMlpHeadBufferLedgerV1::from_dimensions(hidden_size, intermediate_size, vocab_size)
            .unwrap();

    assert_eq!(ledger.scope, "resident-mtlbuffer-only");
    assert_eq!(ledger.abi_version, 1);
    assert_eq!(ledger.allocated_buffers, 13);
    assert_eq!(ledger.shared_buffers, 10);
    assert_eq!(ledger.private_buffers, 3);
    assert_eq!(
        ledger.packed_weight_bytes,
        3 * hidden_size * intermediate_size + vocab_size * hidden_size
    );
    assert_eq!(
        ledger.packed_scale_bytes,
        (3 * hidden_size * intermediate_size / 64 + vocab_size * hidden_size / 64) * 4
    );
    assert_eq!(ledger.f32_parameter_bytes, 2 * hidden_size * 4);
    assert_eq!(ledger.hidden_activation_bytes, 2 * hidden_size * 4);
    assert_eq!(ledger.mlp_activation_bytes, 3 * intermediate_size * 4);
    assert_eq!(ledger.partial_topk_bytes, vocab_size.div_ceil(8) * 4 * 8);
    assert_eq!(ledger.output_token_bytes, 4 * 4);
    assert_eq!(
        ledger.total_persistent_bytes,
        ledger.packed_weight_bytes
            + ledger.packed_scale_bytes
            + ledger.f32_parameter_bytes
            + ledger.hidden_activation_bytes
            + ledger.mlp_activation_bytes
            + ledger.partial_topk_bytes
            + ledger.output_token_bytes
    );
    assert_eq!(ledger.host_input_bytes_per_decode, hidden_size * 4);
    assert_eq!(ledger.host_output_bytes_per_decode, hidden_size * 4 + 4 * 4);
    assert_eq!(ledger.state_host_transfer_bytes_per_decode, 0);
    assert_eq!(ledger.command_buffers_per_decode, 1);
    assert_eq!(ledger.compute_encoders_per_decode, 1);
    assert_eq!(ledger.kernel_dispatches_per_decode, 8);
    assert_eq!(ledger.buffer_barriers_per_decode, 7);
    assert_eq!(ledger.commits_per_decode, 1);
    assert_eq!(ledger.waits_per_decode, 1);

    let qwen = TailMlpHeadBufferLedgerV1::from_dimensions(1_024, 3_584, 248_320).unwrap();
    assert_eq!(qwen.total_persistent_bytes, 282_923_024);
    assert_eq!(qwen.host_input_bytes_per_decode, 4_096);
    assert_eq!(qwen.host_output_bytes_per_decode, 4_112);
}

#[test]
fn tail_v1_rows_kernel_selector_is_explicit_and_legacy_by_default() {
    assert_eq!(
        TailMlpHeadRowsKernelV1::default(),
        TailMlpHeadRowsKernelV1::LegacyR8Sg8
    );
    assert_eq!(
        TailMlpHeadRowsKernelV1::LegacyR8Sg8.receipt_label(),
        "w8_rows_topk4"
    );
    assert_eq!(
        TailMlpHeadRowsKernelV1::Sg16R16.receipt_label(),
        "w8_rows_topk4_sg16"
    );
    assert_eq!(
        TailMlpHeadRowsKernelV1::LegacyR8Sg8.execution_shape(),
        (8, 1, 8, 256, false)
    );
    assert_eq!(
        TailMlpHeadRowsKernelV1::Sg16R16.execution_shape(),
        (16, 1, 16, 512, false)
    );
    let legacy = TailMlpHeadBufferLedgerV1::from_dimensions_with_rows_kernel(
        1_024,
        3_584,
        248_320,
        TailMlpHeadRowsKernelV1::LegacyR8Sg8,
    )
    .unwrap();
    let sg16 = TailMlpHeadBufferLedgerV1::from_dimensions_with_rows_kernel(
        1_024,
        3_584,
        248_320,
        TailMlpHeadRowsKernelV1::Sg16R16,
    )
    .unwrap();
    assert_eq!(
        legacy,
        TailMlpHeadBufferLedgerV1::from_dimensions(1_024, 3_584, 248_320).unwrap()
    );
    assert_eq!(legacy.partial_topk_bytes, 993_280);
    assert_eq!(sg16.partial_topk_bytes, 496_640);
    assert_eq!(sg16.total_persistent_bytes, 282_426_384);
}

#[cfg(target_os = "macos")]
#[test]
fn metal_tail_v1_matches_packed_hidden_and_top4_in_one_transaction() {
    let hidden_size = 64;
    let intermediate_size = 64;
    let vocab_size = 8;
    let projection_elements = hidden_size * intermediate_size;
    let mlp = PackedW8MlpBlock::pack_f32(
        &values(projection_elements, 41, 211, 0.04),
        &values(projection_elements, 43, 199, 0.04),
        &values(projection_elements, 47, 197, 0.04),
        hidden_size,
        intermediate_size,
    )
    .unwrap();
    let post_attention_rms_weight = values(hidden_size, 53, 193, 0.2)
        .into_iter()
        .map(|value| 1.0 + value)
        .collect::<Vec<_>>();
    let final_rms_weight = values(hidden_size, 59, 191, 0.2)
        .into_iter()
        .map(|value| 1.0 + value)
        .collect::<Vec<_>>();
    let vocab = PackedW8Rows::pack_f32(
        &values(vocab_size * hidden_size, 61, 181, 0.1),
        vocab_size,
        hidden_size,
    )
    .unwrap();
    let packed = PackedW8TailMlpHeadV1::new(
        mlp,
        &post_attention_rms_weight,
        &final_rms_weight,
        1.0e-6,
        vocab,
    )
    .unwrap();
    let input = values(hidden_size, 67, 179, 0.8);
    let expected = packed.decode_reference(&input).unwrap();
    let expected_ledger = packed.buffer_ledger().unwrap();
    let mut metal = MetalW8TailMlpHeadV1::from_packed(&packed).unwrap();

    assert_eq!(metal.buffer_ledger(), expected_ledger);
    let actual = metal.decode(&input).unwrap();
    assert_close(
        actual.normalized_hidden,
        &expected.normalized_hidden,
        3.0e-3,
        "Metal tail normalized hidden",
    );
    assert_eq!(actual.candidate_token_ids, expected.candidate_token_ids);
    let stats = metal.stats();
    assert_eq!(stats.decode_calls, 1);
    assert_eq!(stats.successful_decodes, 1);
    assert_eq!(stats.failed_decodes, 0);
    assert_eq!(stats.host_to_device_bytes, hidden_size * 4);
    assert_eq!(stats.device_to_host_bytes, hidden_size * 4 + 16);
    assert_eq!(stats.command_buffers, 1);
    assert_eq!(stats.compute_encoders, 1);
    assert_eq!(stats.kernel_dispatches, 8);
    assert_eq!(stats.buffer_barriers, 7);
    assert_eq!(stats.commits, 1);
    assert_eq!(stats.waits, 1);
    assert_eq!(stats.output_commits, 2);
    assert_eq!(stats.last_output_commit_mask, 0b11);
    assert!(!stats.terminal_error);
}

#[cfg(target_os = "macos")]
#[test]
fn metal_tail_sg16_matches_legacy_and_cpu_for_an_incomplete_final_row_group() {
    let hidden_size = 64;
    let intermediate_size = 64;
    let vocab_size = 19;
    let projection_elements = hidden_size * intermediate_size;
    let packed = PackedW8TailMlpHeadV1::new(
        PackedW8MlpBlock::pack_f32(
            &values(projection_elements, 149, 257, 0.04),
            &values(projection_elements, 151, 251, 0.04),
            &values(projection_elements, 157, 241, 0.04),
            hidden_size,
            intermediate_size,
        )
        .unwrap(),
        &values(hidden_size, 163, 239, 0.2)
            .into_iter()
            .map(|value| 1.0 + value)
            .collect::<Vec<_>>(),
        &values(hidden_size, 167, 233, 0.2)
            .into_iter()
            .map(|value| 1.0 + value)
            .collect::<Vec<_>>(),
        1.0e-6,
        PackedW8Rows::pack_f32(
            &values(vocab_size * hidden_size, 173, 229, 0.1),
            vocab_size,
            hidden_size,
        )
        .unwrap(),
    )
    .unwrap();
    let input = values(hidden_size, 179, 227, 0.8);
    let expected = packed.decode_reference(&input).unwrap();
    let mut legacy = MetalW8TailMlpHeadV1::from_packed(&packed).unwrap();
    let mut sg16 = MetalW8TailMlpHeadV1::from_packed_with_rows_kernel(
        &packed,
        TailMlpHeadRowsKernelV1::Sg16R16,
    )
    .unwrap();

    assert_eq!(legacy.rows_kernel(), TailMlpHeadRowsKernelV1::LegacyR8Sg8);
    assert_eq!(sg16.rows_kernel(), TailMlpHeadRowsKernelV1::Sg16R16);
    let legacy_output = legacy.decode(&input).unwrap();
    let legacy_hidden = legacy_output.normalized_hidden.to_vec();
    let legacy_candidates = legacy_output.candidate_token_ids;
    let sg16_output = sg16.decode(&input).unwrap();
    assert_close(
        &legacy_hidden,
        &expected.normalized_hidden,
        3.0e-3,
        "legacy normalized hidden",
    );
    assert_close(
        sg16_output.normalized_hidden,
        &expected.normalized_hidden,
        3.0e-3,
        "sg16 normalized hidden",
    );
    assert_eq!(legacy_candidates, expected.candidate_token_ids);
    assert_eq!(
        sg16_output.candidate_token_ids,
        expected.candidate_token_ids
    );
}

#[cfg(target_os = "macos")]
#[test]
fn metal_tail_sg16_preserves_lowest_token_ties_with_invalid_tail_rows() {
    let hidden_size = 64;
    let intermediate_size = 64;
    let vocab_size = 17;
    let projection_elements = hidden_size * intermediate_size;
    let packed = PackedW8TailMlpHeadV1::new(
        PackedW8MlpBlock::pack_f32(
            &values(projection_elements, 181, 223, 0.04),
            &values(projection_elements, 191, 211, 0.04),
            &values(projection_elements, 193, 199, 0.04),
            hidden_size,
            intermediate_size,
        )
        .unwrap(),
        &vec![1.0; hidden_size],
        &vec![1.0; hidden_size],
        1.0e-6,
        PackedW8Rows::pack_f32(
            &vec![0.0; vocab_size * hidden_size],
            vocab_size,
            hidden_size,
        )
        .unwrap(),
    )
    .unwrap();
    let input = values(hidden_size, 197, 197, 0.8);
    let expected = packed.decode_reference(&input).unwrap();
    let mut sg16 = MetalW8TailMlpHeadV1::from_packed_with_rows_kernel(
        &packed,
        TailMlpHeadRowsKernelV1::Sg16R16,
    )
    .unwrap();

    assert_eq!(expected.candidate_token_ids, [0, 1, 2, 3]);
    assert_eq!(
        sg16.decode(&input).unwrap().candidate_token_ids,
        [0, 1, 2, 3]
    );
}

#[cfg(all(target_os = "macos", debug_assertions))]
#[test]
fn tail_v1_post_gpu_failure_publishes_nothing_is_terminal_and_reset_recovers() {
    let hidden_size = 64;
    let intermediate_size = 64;
    let vocab_size = 8;
    let projection_elements = hidden_size * intermediate_size;
    let packed = PackedW8TailMlpHeadV1::new(
        PackedW8MlpBlock::pack_f32(
            &values(projection_elements, 71, 173, 0.04),
            &values(projection_elements, 73, 167, 0.04),
            &values(projection_elements, 79, 163, 0.04),
            hidden_size,
            intermediate_size,
        )
        .unwrap(),
        &vec![1.0; hidden_size],
        &vec![1.0; hidden_size],
        1.0e-6,
        PackedW8Rows::pack_f32(
            &values(vocab_size * hidden_size, 83, 157, 0.1),
            vocab_size,
            hidden_size,
        )
        .unwrap(),
    )
    .unwrap();
    let input = values(hidden_size, 89, 151, 0.8);
    let mut metal = MetalW8TailMlpHeadV1::from_packed(&packed).unwrap();

    let error = metal
        .inject_failure_after_gpu_execution_for_testing(&input)
        .unwrap_err();

    assert!(error.to_string().contains("injected"));
    let failed = metal.stats();
    assert_eq!(failed.decode_calls, 1);
    assert_eq!(failed.successful_decodes, 0);
    assert_eq!(failed.failed_decodes, 1);
    assert_eq!(failed.host_to_device_bytes, hidden_size * 4);
    assert_eq!(failed.device_to_host_bytes, 0);
    assert_eq!(failed.command_buffers, 1);
    assert_eq!(failed.compute_encoders, 1);
    assert_eq!(failed.kernel_dispatches, 8);
    assert_eq!(failed.buffer_barriers, 7);
    assert_eq!(failed.commits, 1);
    assert_eq!(failed.waits, 1);
    assert_eq!(failed.output_commits, 0);
    assert_eq!(failed.last_output_commit_mask, 0);
    assert!(failed.terminal_error);

    let retry = metal.decode(&input).unwrap_err();
    assert!(retry.to_string().contains("terminal"));
    assert_eq!(metal.stats(), failed, "terminal retry must submit no work");

    metal.reset().unwrap();
    assert_eq!(metal.stats(), Default::default());
    metal.decode(&input).unwrap();
    assert_eq!(metal.stats().output_commits, 2);
    assert_eq!(metal.stats().last_output_commit_mask, 0b11);
}

#[cfg(all(target_os = "macos", debug_assertions))]
#[test]
fn tail_v1_rejects_malformed_gpu_outputs_before_atomic_publication() {
    let hidden_size = 64;
    let intermediate_size = 64;
    let vocab_size = 8;
    let projection_elements = hidden_size * intermediate_size;
    let packed = PackedW8TailMlpHeadV1::new(
        PackedW8MlpBlock::pack_f32(
            &values(projection_elements, 97, 149, 0.04),
            &values(projection_elements, 101, 139, 0.04),
            &values(projection_elements, 103, 137, 0.04),
            hidden_size,
            intermediate_size,
        )
        .unwrap(),
        &vec![1.0; hidden_size],
        &vec![1.0; hidden_size],
        1.0e-6,
        PackedW8Rows::pack_f32(
            &values(vocab_size * hidden_size, 107, 131, 0.1),
            vocab_size,
            hidden_size,
        )
        .unwrap(),
    )
    .unwrap();
    let input = values(hidden_size, 109, 127, 0.8);

    for (label, inject) in [
        (
            "non-finite normalized hidden",
            MetalW8TailMlpHeadV1::inject_nonfinite_normalized_output_for_testing
                as fn(&mut MetalW8TailMlpHeadV1, &[f32]) -> Result<(), apxinf_metal::MetalW8Error>,
        ),
        (
            "duplicate candidate",
            MetalW8TailMlpHeadV1::inject_duplicate_candidate_output_for_testing,
        ),
        (
            "out-of-range candidate",
            MetalW8TailMlpHeadV1::inject_out_of_range_candidate_output_for_testing,
        ),
    ] {
        let mut metal = MetalW8TailMlpHeadV1::from_packed(&packed).unwrap();
        let error = inject(&mut metal, &input).unwrap_err();
        assert!(
            error.to_string().contains("GPU output failed validation"),
            "{label}: {error}"
        );
        let stats = metal.stats();
        assert_eq!(stats.failed_decodes, 1, "{label}");
        assert_eq!(stats.device_to_host_bytes, 0, "{label}");
        assert_eq!(stats.output_commits, 0, "{label}");
        assert_eq!(stats.last_output_commit_mask, 0, "{label}");
        assert!(stats.terminal_error, "{label}");
    }
}

#[cfg(all(target_os = "macos", debug_assertions))]
fn assert_tail_v1_rust_staging_validation_failure(
    expected_error: &str,
    inject: fn(&mut MetalW8TailMlpHeadV1, &[f32]) -> Result<(), apxinf_metal::MetalW8Error>,
) {
    let hidden_size = 64;
    let intermediate_size = 64;
    let vocab_size = 8;
    let projection_elements = hidden_size * intermediate_size;
    let packed = PackedW8TailMlpHeadV1::new(
        PackedW8MlpBlock::pack_f32(
            &values(projection_elements, 113, 127, 0.04),
            &values(projection_elements, 109, 131, 0.04),
            &values(projection_elements, 107, 137, 0.04),
            hidden_size,
            intermediate_size,
        )
        .unwrap(),
        &vec![1.0; hidden_size],
        &vec![1.0; hidden_size],
        1.0e-6,
        PackedW8Rows::pack_f32(
            &values(vocab_size * hidden_size, 103, 139, 0.1),
            vocab_size,
            hidden_size,
        )
        .unwrap(),
    )
    .unwrap();
    let input = values(hidden_size, 101, 149, 0.8);
    let mut metal = MetalW8TailMlpHeadV1::from_packed(&packed).unwrap();

    let error = inject(&mut metal, &input).unwrap_err();

    assert!(error.to_string().contains(expected_error), "{error}");
    let failed = metal.stats();
    assert_eq!(failed.decode_calls, 1);
    assert_eq!(failed.successful_decodes, 0);
    assert_eq!(failed.failed_decodes, 1);
    assert_eq!(failed.host_to_device_bytes, hidden_size * 4);
    assert_eq!(failed.device_to_host_bytes, hidden_size * 4 + 16);
    assert_eq!(failed.command_buffers, 1);
    assert_eq!(failed.compute_encoders, 1);
    assert_eq!(failed.kernel_dispatches, 8);
    assert_eq!(failed.buffer_barriers, 7);
    assert_eq!(failed.commits, 1);
    assert_eq!(failed.waits, 1);
    assert_eq!(failed.output_commits, 2);
    assert_eq!(failed.last_output_commit_mask, 0b11);
    assert!(failed.terminal_error);

    assert!(metal
        .decode(&input)
        .unwrap_err()
        .to_string()
        .contains("terminal"));
    assert_eq!(metal.stats(), failed, "terminal retry must submit no work");

    metal.reset().unwrap();
    assert_eq!(metal.stats(), Default::default());
    metal.decode(&input).unwrap();
    assert_eq!(metal.stats().successful_decodes, 1);
}

#[cfg(all(target_os = "macos", debug_assertions))]
#[test]
fn tail_v1_rust_rejects_nonfinite_staging_after_bridge_success_as_terminal_failure() {
    assert_tail_v1_rust_staging_validation_failure(
        "non-finite",
        MetalW8TailMlpHeadV1::inject_nonfinite_staging_after_bridge_success_for_testing,
    );
}

#[cfg(all(target_os = "macos", debug_assertions))]
#[test]
fn tail_v1_rust_rejects_duplicate_staging_after_bridge_success_as_terminal_failure() {
    assert_tail_v1_rust_staging_validation_failure(
        "duplicate candidate",
        MetalW8TailMlpHeadV1::inject_duplicate_staging_after_bridge_success_for_testing,
    );
}

#[cfg(all(target_os = "macos", debug_assertions))]
#[test]
fn tail_v1_rust_rejects_out_of_range_staging_after_bridge_success_as_terminal_failure() {
    assert_tail_v1_rust_staging_validation_failure(
        "outside vocabulary",
        MetalW8TailMlpHeadV1::inject_out_of_range_staging_after_bridge_success_for_testing,
    );
}

#[test]
fn tail_v1_fails_closed_on_shapes_groups_nonfinite_values_and_ledger_overflow() {
    let hidden_size = 64;
    let intermediate_size = 64;
    let projection_elements = hidden_size * intermediate_size;
    let gate = values(projection_elements, 113, 127, 0.04);
    let up = values(projection_elements, 127, 113, 0.04);
    let down = values(projection_elements, 109, 107, 0.04);
    let mlp =
        PackedW8MlpBlock::pack_f32(&gate, &up, &down, hidden_size, intermediate_size).unwrap();
    let vocab_values = values(8 * hidden_size, 103, 101, 0.1);
    let vocab = PackedW8Rows::pack_f32(&vocab_values, 8, hidden_size).unwrap();

    let bad_eps = PackedW8TailMlpHeadV1::new(
        mlp.clone(),
        &vec![1.0; hidden_size],
        &vec![1.0; hidden_size],
        f32::NAN,
        vocab.clone(),
    )
    .unwrap_err();
    assert!(bad_eps.to_string().contains("RMS epsilon"));
    let negative_eps = PackedW8TailMlpHeadV1::new(
        mlp.clone(),
        &vec![1.0; hidden_size],
        &vec![1.0; hidden_size],
        -1.0e-6,
        vocab.clone(),
    )
    .unwrap_err();
    assert!(negative_eps.to_string().contains("RMS epsilon"));

    let mut bad_post = vec![1.0; hidden_size];
    bad_post[7] = f32::INFINITY;
    let bad_rms = PackedW8TailMlpHeadV1::new(
        mlp.clone(),
        &bad_post,
        &vec![1.0; hidden_size],
        1.0e-6,
        vocab.clone(),
    )
    .unwrap_err();
    assert!(bad_rms.to_string().contains("non-finite"));
    let mut bad_final = vec![1.0; hidden_size];
    bad_final[11] = f32::NEG_INFINITY;
    let bad_final_rms = PackedW8TailMlpHeadV1::new(
        mlp.clone(),
        &vec![1.0; hidden_size],
        &bad_final,
        1.0e-6,
        vocab.clone(),
    )
    .unwrap_err();
    assert!(bad_final_rms.to_string().contains("non-finite"));

    let too_small_vocab = PackedW8TailMlpHeadV1::new(
        mlp.clone(),
        &vec![1.0; hidden_size],
        &vec![1.0; hidden_size],
        1.0e-6,
        PackedW8Rows::pack_f32(&vec![0.0; 3 * hidden_size], 3, hidden_size).unwrap(),
    )
    .unwrap_err();
    assert!(too_small_vocab.to_string().contains("at least four"));

    let g32_down = PackedW8MlpBlock::pack_f32_with_down_group_size(
        &gate,
        &up,
        &down,
        hidden_size,
        intermediate_size,
        apxinf_metal::W8GroupSize::G32,
    )
    .unwrap();
    let wrong_mlp_group = PackedW8TailMlpHeadV1::new(
        g32_down,
        &vec![1.0; hidden_size],
        &vec![1.0; hidden_size],
        1.0e-6,
        vocab.clone(),
    )
    .unwrap_err();
    assert!(wrong_mlp_group.to_string().contains("group size 64"));

    let wrong_vocab_group = PackedW8TailMlpHeadV1::new(
        mlp.clone(),
        &vec![1.0; hidden_size],
        &vec![1.0; hidden_size],
        1.0e-6,
        PackedW8Rows::pack_f32_g32(&vocab_values, 8, hidden_size).unwrap(),
    )
    .unwrap_err();
    assert!(wrong_vocab_group.to_string().contains("group size 64"));

    let wrong_shape = PackedW8TailMlpHeadV1::new(
        mlp.clone(),
        &vec![1.0; hidden_size - 1],
        &vec![1.0; hidden_size],
        1.0e-6,
        vocab.clone(),
    )
    .unwrap_err();
    assert!(wrong_shape.to_string().contains("hidden shapes differ"));

    assert!(
        TailMlpHeadBufferLedgerV1::from_dimensions(u32::MAX as usize + 1, 64, 8)
            .unwrap_err()
            .to_string()
            .contains("u32 ABI")
    );
    let largest_g64_u32 = u32::MAX as usize - (u32::MAX as usize % 64);
    let largest_g64_intermediate = u32::MAX as usize / 2 - (u32::MAX as usize / 2 % 64);
    assert!(TailMlpHeadBufferLedgerV1::from_dimensions(
        largest_g64_u32,
        largest_g64_intermediate,
        4,
    )
    .unwrap_err()
    .to_string()
    .contains("overflow"));

    let packed = PackedW8TailMlpHeadV1::new(
        mlp,
        &vec![1.0; hidden_size],
        &vec![1.0; hidden_size],
        1.0e-6,
        vocab,
    )
    .unwrap();
    let mut nonfinite_input = vec![0.0; hidden_size];
    nonfinite_input[5] = f32::NAN;
    assert!(packed
        .decode_reference(&nonfinite_input)
        .unwrap_err()
        .to_string()
        .contains("non-finite"));

    #[cfg(target_os = "macos")]
    {
        let mut metal = MetalW8TailMlpHeadV1::from_packed(&packed).unwrap();
        assert!(metal
            .decode(&nonfinite_input)
            .unwrap_err()
            .to_string()
            .contains("non-finite"));
        assert_eq!(
            metal.stats(),
            Default::default(),
            "invalid host input submits no work"
        );
    }
}

#[test]
fn tail_v1_bridge_shape_and_shader_custody_match_the_public_contract() {
    let bridge = include_str!("../src/metal_w8_tail_mlp_head_v1_bridge.mm");
    for symbol in [
        "apxinf_metal_w8_tail_mlp_head_create_v1(",
        "apxinf_metal_w8_tail_mlp_head_decode_v1(",
        "apxinf_metal_w8_tail_mlp_head_reset_v1(",
        "apxinf_metal_w8_tail_mlp_head_destroy_v1(",
    ] {
        assert!(bridge.contains(symbol), "missing bridge symbol {symbol}");
    }
    let handle_buffers = bridge
        .split("struct ApxinfMetalW8TailMlpHeadHandleV1 {")
        .nth(1)
        .unwrap()
        .split("LinearLayerParams layer_params;")
        .next()
        .unwrap();
    assert_eq!(handle_buffers.matches("id<MTLBuffer>").count(), 13);
    let allocation = bridge
        .split("const MTLResourceOptions shared")
        .nth(1)
        .unwrap()
        .split("handle->layer_params =")
        .next()
        .unwrap();
    assert_eq!(allocation.matches("options:shared").count(), 10);
    assert_eq!(allocation.matches("options:private_storage").count(), 3);
    let encoder = bridge
        .split("void encode_tail(")
        .nth(1)
        .unwrap()
        .split("}  // namespace")
        .next()
        .unwrap();
    assert_eq!(encoder.matches("dispatchThread").count(), 8);
    assert_eq!(encoder.matches("buffer_barrier(encoder);").count(), 7);
    assert_eq!(bridge.matches("[handle->queue commandBuffer]").count(), 1);
    assert_eq!(bridge.matches("[command computeCommandEncoder]").count(), 1);
    assert_eq!(bridge.matches("[command commit]").count(), 1);
    assert_eq!(bridge.matches("[command waitUntilCompleted]").count(), 1);
    assert!(!bridge.contains("kernel void w8_mlp_gate_up("));
    assert!(!bridge.contains("kernel void w8_rows_topk4("));
    assert!(bridge.contains("@\"w8_rows_topk4_sg16\""));
    assert!(bridge.contains("kHeadSg16Threads"));
    assert!(bridge.contains("apxinf_metal_w8_tail_mlp_head_create_with_rows_kernel_v1("));
    assert!(bridge.contains("uint32_t rows_kernel"));
    assert!(bridge.contains("rows_kernel > kRowsKernelSg16R16"));

    let shader = include_str!("../src/metal_w8.metal");
    assert_eq!(shader.matches("kernel void w8_rows_topk4_sg16(").count(), 1);
    assert!(shader.contains("constexpr uint rows_per_threadgroup = 16;"));
    assert!(bridge.contains("kHeadSg16RowsPerThreadgroup = 16"));
    assert!(bridge.contains("maxTotalThreadsPerThreadgroup"));
    let standalone_bridge = include_str!("../src/metal_w8_bridge.mm");
    assert!(!standalone_bridge.contains("w8_rows_topk4_sg16"));

    let build = include_str!("../build.rs");
    assert!(build.contains("format!(\"{mlp_shader}\\n{linear_layer_shader}\\n{shader}\")"));
    assert!(build.contains("metal_w8_tail_mlp_head_v1_source.inc"));
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
