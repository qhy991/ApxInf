#[cfg(target_os = "macos")]
use std::sync::{Mutex, MutexGuard, OnceLock};

#[cfg(target_os = "macos")]
use apxinf_metal::{
    FullAttentionLayerF32WeightsV1, FullAttentionStack6RuntimeReceiptV1,
    MetalW8FullAttentionStack6V1, PackedW8FullAttentionStack6V1, QWEN35_FULL_ATTENTION_HEAD_DIM_V1,
    QWEN35_FULL_ATTENTION_HIDDEN_SIZE_V1, QWEN35_FULL_ATTENTION_KV_HEADS_V1,
    QWEN35_FULL_ATTENTION_KV_WIDTH_V1, QWEN35_FULL_ATTENTION_LAYER_SLOTS_V1,
    QWEN35_FULL_ATTENTION_QUERY_HEADS_V1, QWEN35_FULL_ATTENTION_QUERY_WIDTH_V1,
    QWEN35_FULL_ATTENTION_ROTARY_DIM_V1, W8_GROUP_SIZE,
};

#[cfg(target_os = "macos")]
fn serial_metal_fixture() -> MutexGuard<'static, ()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

#[cfg(target_os = "macos")]
fn deterministic_values(
    elements: usize,
    multiplier: usize,
    modulus: usize,
    scale: f32,
) -> Vec<f32> {
    (0..elements)
        .map(|index| {
            let bucket = index.wrapping_mul(multiplier).wrapping_add(17) % modulus;
            (bucket as f32 / (modulus - 1) as f32 * 2.0 - 1.0) * scale
        })
        .collect()
}

/// All five projections deliberately borrow the same largest F32 allocation,
/// and all six layer descriptors borrow those same slices. The packer consumes
/// one layer at a time, so the fixture never retains six F32 checkpoint copies.
#[cfg(target_os = "macos")]
fn packed_shared_fixture(projection_scale: f32, norm_scale: f32) -> PackedW8FullAttentionStack6V1 {
    let projection = deterministic_values(
        QWEN35_FULL_ATTENTION_QUERY_WIDTH_V1 * QWEN35_FULL_ATTENTION_HIDDEN_SIZE_V1,
        73,
        65_521,
        projection_scale,
    );
    let norm = deterministic_values(QWEN35_FULL_ATTENTION_HIDDEN_SIZE_V1, 29, 4_093, norm_scale);
    let kv_projection_elements =
        QWEN35_FULL_ATTENTION_KV_WIDTH_V1 * QWEN35_FULL_ATTENTION_HIDDEN_SIZE_V1;
    let layer = FullAttentionLayerF32WeightsV1 {
        input_rms_weight: &norm,
        query_rows: &projection,
        gate_rows: &projection,
        key_rows: &projection[..kv_projection_elements],
        value_rows: &projection[..kv_projection_elements],
        query_norm_weight: &norm[..QWEN35_FULL_ATTENTION_HEAD_DIM_V1],
        key_norm_weight: &norm[..QWEN35_FULL_ATTENTION_HEAD_DIM_V1],
        output_rows: &projection,
    };
    PackedW8FullAttentionStack6V1::pack_f32(&[layer; QWEN35_FULL_ATTENTION_LAYER_SLOTS_V1]).unwrap()
}

#[cfg(target_os = "macos")]
fn assert_fixed_topology(
    receipt: FullAttentionStack6RuntimeReceiptV1,
    max_context: usize,
    successful_decodes: u64,
) {
    assert_eq!(
        receipt.layer_slots,
        QWEN35_FULL_ATTENTION_LAYER_SLOTS_V1 as u32
    );
    assert_eq!(
        receipt.hidden_size,
        QWEN35_FULL_ATTENTION_HIDDEN_SIZE_V1 as u32
    );
    assert_eq!(
        receipt.query_heads,
        QWEN35_FULL_ATTENTION_QUERY_HEADS_V1 as u32
    );
    assert_eq!(receipt.kv_heads, QWEN35_FULL_ATTENTION_KV_HEADS_V1 as u32);
    assert_eq!(receipt.head_dim, QWEN35_FULL_ATTENTION_HEAD_DIM_V1 as u32);
    assert_eq!(
        receipt.rotary_dim,
        QWEN35_FULL_ATTENTION_ROTARY_DIM_V1 as u32
    );
    assert_eq!(receipt.max_context, max_context as u32);
    assert_eq!(receipt.group_size, W8_GROUP_SIZE as u32);
    assert_eq!(receipt.command_buffers_per_decode, 1);
    assert_eq!(receipt.compute_encoders_per_decode, 1);
    assert_eq!(receipt.kernel_dispatches_per_decode, 5);
    assert_eq!(receipt.explicit_buffer_barriers_per_decode, 4);
    assert_eq!(receipt.commits_per_decode, 1);
    assert_eq!(receipt.waits_per_decode, 1);
    assert!(receipt.fixed_shape_validated);
    assert_eq!(receipt.successful_decodes, successful_decodes);
    let observed = u32::from(successful_decodes != 0);
    assert_eq!(receipt.last_observed_command_buffers, observed);
    assert_eq!(receipt.last_observed_compute_encoders, observed);
    assert_eq!(receipt.last_observed_kernel_dispatches, observed * 5);
    assert_eq!(receipt.last_observed_explicit_buffer_barriers, observed * 4);
    assert_eq!(receipt.last_observed_commits, observed);
    assert_eq!(receipt.last_observed_waits, observed);
}

#[cfg(target_os = "macos")]
fn error_metrics(actual: &[f32], expected: &[f32]) -> (f32, f32) {
    assert_eq!(actual.len(), expected.len());
    let mut max_abs = 0.0f32;
    let mut squared_error = 0.0f64;
    let mut squared_reference = 0.0f64;
    for (&actual, &expected) in actual.iter().zip(expected) {
        let error = (actual - expected).abs();
        max_abs = max_abs.max(error);
        squared_error += f64::from(error) * f64::from(error);
        squared_reference += f64::from(expected) * f64::from(expected);
    }
    let rmse = (squared_error / actual.len() as f64).sqrt();
    let reference_rms = (squared_reference / actual.len() as f64).sqrt();
    (max_abs, (rmse / reference_rms.max(1.0e-12)) as f32)
}

#[cfg(target_os = "macos")]
fn assert_error_metrics(
    label: &str,
    actual: &[f32],
    expected: &[f32],
    max_abs_limit: f32,
    nrmse_limit: f32,
) {
    let (max_abs, nrmse) = error_metrics(actual, expected);
    assert!(
        max_abs <= max_abs_limit && nrmse <= nrmse_limit,
        "{label}: max_abs={max_abs:e} (limit {max_abs_limit:e}), NRMSE={nrmse:e} (limit {nrmse_limit:e})"
    );
}

#[cfg(target_os = "macos")]
#[test]
fn zero_projection_is_exact_residual_zero_kv_and_reports_the_fixed_stack_topology() {
    let _serial = serial_metal_fixture();
    const MAX_CONTEXT: usize = 16;
    let packed = packed_shared_fixture(0.0, 0.0);
    let input = deterministic_values(QWEN35_FULL_ATTENTION_HIDDEN_SIZE_V1, 37, 2_039, 0.75);
    let mut metal = MetalW8FullAttentionStack6V1::from_packed(&packed, MAX_CONTEXT).unwrap();

    let initial = metal.runtime_receipt().unwrap();
    assert_fixed_topology(initial, MAX_CONTEXT, 0);

    for layer_slot in 0..QWEN35_FULL_ATTENTION_LAYER_SLOTS_V1 {
        let residual = metal.decode(layer_slot, &input, 0).unwrap();
        assert_eq!(
            residual, input,
            "zero projection must preserve the residual bit-for-bit at layer slot {layer_slot}"
        );
        let (key, value) = metal.snapshot_cache_row(layer_slot, 0).unwrap();
        assert!(key.iter().all(|&element| element == 0.0));
        assert!(value.iter().all(|&element| element == 0.0));
    }

    let receipt = metal.runtime_receipt().unwrap();
    assert_fixed_topology(receipt, MAX_CONTEXT, 6);
    assert_eq!(receipt.last_layer_slot, 5);
    assert_eq!(receipt.last_start_pos, 0);
    assert_eq!(receipt.last_kv_length, 1);
}

#[cfg(target_os = "macos")]
#[test]
fn nonzero_prefix_matches_packed_oracle_and_validation_failures_preserve_retryability() {
    let _serial = serial_metal_fixture();
    const START_POS: u32 = 13;
    const MAX_CONTEXT: usize = START_POS as usize + 1;
    const LAYER_SLOT: usize = 4;

    let packed = packed_shared_fixture(1.5e-3, 0.125);
    let input = deterministic_values(QWEN35_FULL_ATTENTION_HIDDEN_SIZE_V1, 43, 2_033, 0.8);
    let prefix_elements = START_POS as usize * QWEN35_FULL_ATTENTION_KV_WIDTH_V1;
    let prefix_keys = deterministic_values(prefix_elements, 47, 8_191, 0.04);
    let prefix_values = deterministic_values(prefix_elements, 53, 8_209, 0.035);
    let expected = packed
        .decode_with_prefix(LAYER_SLOT, &input, START_POS, &prefix_keys, &prefix_values)
        .unwrap();

    let mut metal = MetalW8FullAttentionStack6V1::from_packed(&packed, MAX_CONTEXT).unwrap();
    metal
        .seed_cache(LAYER_SLOT, START_POS, &prefix_keys, &prefix_values)
        .unwrap();
    let initial_receipt = metal.runtime_receipt().unwrap();
    assert_fixed_topology(initial_receipt, MAX_CONTEXT, 0);

    let mut nonfinite_input = input.clone();
    nonfinite_input[127] = f32::NAN;
    let nonfinite_error = metal
        .decode(LAYER_SLOT, &nonfinite_input, START_POS)
        .unwrap_err();
    assert!(
        nonfinite_error.to_string().contains("non-finite"),
        "{nonfinite_error}"
    );
    assert_eq!(metal.runtime_receipt().unwrap(), initial_receipt);

    let range_error = metal
        .decode(LAYER_SLOT, &input, MAX_CONTEXT as u32)
        .unwrap_err();
    assert!(range_error.to_string().contains("outside max_context"));
    assert_eq!(metal.runtime_receipt().unwrap(), initial_receipt);

    // The valid retry uses the same position as the rejected NaN call. Neither
    // validation failure submitted work or advanced the successful counter.
    let residual = metal
        .decode(LAYER_SLOT, &input, START_POS)
        .unwrap()
        .to_vec();
    let (appended_key, appended_value) = metal.snapshot_cache_row(LAYER_SLOT, START_POS).unwrap();

    assert_error_metrics("residual", &residual, &expected.residual, 1.0e-5, 1.0e-6);
    assert_error_metrics("appended key", &appended_key, &expected.key, 1.0e-5, 2.0e-6);
    assert_error_metrics(
        "appended value",
        &appended_value,
        &expected.value,
        2.0e-6,
        2.0e-6,
    );

    let receipt = metal.runtime_receipt().unwrap();
    assert_fixed_topology(receipt, MAX_CONTEXT, 1);
    assert_eq!(receipt.last_layer_slot, LAYER_SLOT as u32);
    assert_eq!(receipt.last_start_pos, START_POS);
    assert_eq!(receipt.last_kv_length, START_POS + 1);
}
