use super::{
    checked_sum, f32_bytes, GdnDecodeState, GdnDimensions, MetalW8Error, PackedW8LinearLayerBlock,
    W8GroupSize,
};
use crate::{body_scale_load_receipt, W8BodyScaleLoadRuntimeReceiptV1, W8ScaleLoadProfileV1};

const STACK_DEPTH: usize = 3;

/// Persistent-memory and per-decode transaction contract for the versioned
/// three-layer diagnostic stack.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LinearLayerStack3BufferLedger {
    pub allocated_buffers: usize,
    pub shared_buffers: usize,
    pub private_buffers: usize,
    pub packed_weight_bytes: usize,
    pub packed_scale_bytes: usize,
    pub f32_parameter_bytes: usize,
    pub active_state_bytes: usize,
    pub scratch_state_bytes: usize,
    pub activation_bytes: usize,
    pub total_persistent_bytes: usize,
    pub host_input_bytes_per_decode: usize,
    pub host_output_bytes_per_decode: usize,
    pub state_host_transfer_bytes_per_decode: usize,
    pub command_buffers_per_decode: usize,
    pub compute_encoders_per_decode: usize,
    pub commits_per_decode: usize,
    pub waits_per_decode: usize,
    /// The v1 stack keeps intermediate rows on device. Only the final row is
    /// checked for finiteness on the host-visible Metal buffer.
    pub intermediate_host_finite_checks_per_decode: usize,
    pub final_output_finite_checks_per_decode: usize,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LinearLayerStack3MetalStats {
    pub decode_calls: usize,
    pub successful_decodes: usize,
    pub failed_decodes: usize,
    pub command_buffers: usize,
    pub compute_encoders: usize,
    pub commits: usize,
    pub waits: usize,
    pub host_to_device_bytes: usize,
    pub device_to_host_bytes: usize,
    pub state_commits: usize,
    pub last_state_commit_mask: u32,
    pub committed_stack_version: u64,
    pub terminal_error: bool,
}

/// Versioned, diagnostic-only stack of exactly three consecutive complete
/// linear-attention layers. One decode uploads/fetches one hidden row and runs
/// one command buffer with three encoders. Intermediate rows stay on device;
/// unlike three host-staged v2 blocks, v1 checks only the final output for
/// finiteness. No default runtime constructs this type.
pub struct MetalW8LinearLayerStack3 {
    dims: GdnDimensions,
    inner: platform::LinearLayerStack3Handle,
    output: Vec<f32>,
    seeded: bool,
    terminal_error: bool,
    stats: LinearLayerStack3MetalStats,
    buffer_ledger: LinearLayerStack3BufferLedger,
    scale_load_receipt: W8BodyScaleLoadRuntimeReceiptV1,
}

impl MetalW8LinearLayerStack3 {
    /// Fixed v1 precision contract: G64 GDN input, G32 GDN output, and G64 MLP.
    /// All three layers must have exactly equal dimensions and RMS epsilons.
    pub fn from_packed_gdn_out_g32_v1(
        weights: [&PackedW8LinearLayerBlock; STACK_DEPTH],
    ) -> Result<Self, MetalW8Error> {
        Self::from_packed_gdn_out_g32_with_scale_load_profile_v1(
            weights,
            W8ScaleLoadProfileV1::LegacyPerLane,
        )
    }

    /// Additive diagnostic selector for broadcasting identical per-group W8
    /// scales within a SIMD group. The legacy constructor above always keeps
    /// the original per-lane scale loads.
    pub fn from_packed_gdn_out_g32_with_scale_load_profile_v1(
        weights: [&PackedW8LinearLayerBlock; STACK_DEPTH],
        scale_load_profile: W8ScaleLoadProfileV1,
    ) -> Result<Self, MetalW8Error> {
        for (index, weights) in weights.iter().enumerate() {
            validate_precision_v1(index, weights)?;
        }
        let dims = weights[0].gdn.dimensions();
        let intermediate_size = weights[0].intermediate_size();
        let layer_rms_norm_eps = weights[0].rms_norm_eps;
        for (index, candidate) in weights.iter().enumerate().skip(1) {
            if candidate.gdn.dimensions() != dims
                || candidate.intermediate_size() != intermediate_size
                || candidate.rms_norm_eps != layer_rms_norm_eps
            {
                return Err(MetalW8Error::new(format!(
                    "Metal W8 stack3 v1 layer {index} dimensions or RMS epsilons differ from layer 0"
                )));
            }
        }
        for (label, value) in [
            ("hidden_size", dims.hidden_size),
            ("key_heads", dims.key_heads),
            ("value_heads", dims.value_heads),
            ("key_dim", dims.key_dim),
            ("value_dim", dims.value_dim),
            ("conv_kernel_size", dims.conv_kernel_size),
            ("input_rows", dims.input_projection_rows()),
            ("intermediate_size", intermediate_size),
        ] {
            if value > u32::MAX as usize {
                return Err(MetalW8Error::new(format!(
                    "Metal W8 stack3 v1 {label} exceeds the u32 ABI"
                )));
            }
        }
        validate_stack3_state_abi(dims)?;
        let buffer_ledger = stack_buffer_ledger(weights)?;
        let inner = platform::LinearLayerStack3Handle::new(weights, scale_load_profile)?;
        let observed_profile = inner.observed_scale_load_profile()?;
        if observed_profile != scale_load_profile {
            return Err(MetalW8Error::new(
                "Metal W8 stack3 v1 observed scale-load profile differs from its request",
            ));
        }
        Ok(Self {
            dims,
            inner,
            output: vec![0.0; dims.hidden_size],
            seeded: false,
            terminal_error: false,
            stats: LinearLayerStack3MetalStats::default(),
            buffer_ledger,
            scale_load_receipt: body_scale_load_receipt(scale_load_profile, observed_profile),
        })
    }

    pub fn seed_decode_states(
        &mut self,
        states: &[GdnDecodeState; STACK_DEPTH],
    ) -> Result<(), MetalW8Error> {
        if self.terminal_error {
            return Err(MetalW8Error::new(
                "Metal W8 stack3 v1 is terminal after a decode failure; clear before reseeding",
            ));
        }
        validate_states(states, self.dims)?;
        self.inner.seed(states)?;
        self.seeded = true;
        self.stats = LinearLayerStack3MetalStats::default();
        Ok(())
    }

    pub fn clear_decode_states(&mut self) -> Result<(), MetalW8Error> {
        let cleared = std::array::from_fn(|_| GdnDecodeState::zeroed(self.dims).unwrap());
        self.inner.seed(&cleared)?;
        self.output.fill(0.0);
        self.seeded = false;
        self.terminal_error = false;
        self.stats = LinearLayerStack3MetalStats::default();
        Ok(())
    }

    pub fn decode(&mut self, hidden: &[f32]) -> Result<&[f32], MetalW8Error> {
        self.validate_decode_input(hidden)?;
        let execution = self.inner.decode(hidden, &mut self.output, false);
        self.record_execution(&execution);
        if let Err(error) = execution.result {
            self.terminal_error = true;
            self.stats.terminal_error = true;
            return Err(error);
        }
        Ok(&self.output)
    }

    pub fn state_snapshots(&self) -> Result<[GdnDecodeState; STACK_DEPTH], MetalW8Error> {
        if !self.seeded {
            return Err(MetalW8Error::new(
                "Metal W8 stack3 v1 states must be seeded before snapshot",
            ));
        }
        let snapshots = (0..STACK_DEPTH)
            .map(|slot| self.inner.snapshot(slot, self.dims))
            .collect::<Result<Vec<_>, _>>()?;
        snapshots
            .try_into()
            .map_err(|_| MetalW8Error::new("Metal W8 stack3 v1 snapshot depth changed"))
    }

    pub fn stats(&self) -> LinearLayerStack3MetalStats {
        self.stats
    }

    pub fn buffer_ledger(&self) -> LinearLayerStack3BufferLedger {
        self.buffer_ledger
    }

    pub fn scale_load_runtime_receipt_v1(&self) -> W8BodyScaleLoadRuntimeReceiptV1 {
        self.scale_load_receipt
    }

    fn validate_decode_input(&self, hidden: &[f32]) -> Result<(), MetalW8Error> {
        if self.terminal_error {
            return Err(MetalW8Error::new(
                "Metal W8 stack3 v1 is terminal after a decode failure; clear before retry",
            ));
        }
        if !self.seeded {
            return Err(MetalW8Error::new(
                "Metal W8 stack3 v1 decode states must be seeded after CPU prefill",
            ));
        }
        if hidden.len() != self.dims.hidden_size {
            return Err(MetalW8Error::new(format!(
                "Metal W8 stack3 v1 input has {} elements, expected {}",
                hidden.len(),
                self.dims.hidden_size
            )));
        }
        if let Some(index) = hidden.iter().position(|value| !value.is_finite()) {
            return Err(MetalW8Error::new(format!(
                "Metal W8 stack3 v1 input contains a non-finite value at element {index}"
            )));
        }
        Ok(())
    }

    fn record_execution(&mut self, execution: &platform::Stack3Execution) {
        self.stats.decode_calls += 1;
        if execution.result.is_ok() {
            self.stats.successful_decodes += 1;
        } else {
            self.stats.failed_decodes += 1;
        }
        self.stats.command_buffers += execution.receipt.command_buffers as usize;
        self.stats.compute_encoders += execution.receipt.compute_encoders as usize;
        self.stats.commits += execution.receipt.commits as usize;
        self.stats.waits += execution.receipt.waits as usize;
        self.stats.host_to_device_bytes += execution.receipt.host_to_device_bytes as usize;
        self.stats.device_to_host_bytes += execution.receipt.device_to_host_bytes as usize;
        self.stats.state_commits += execution.receipt.state_commits as usize;
        self.stats.last_state_commit_mask = execution.receipt.state_commit_mask;
        if execution.receipt.state_commits == STACK_DEPTH as u32
            && execution.receipt.state_commit_mask == 0b111
        {
            self.stats.committed_stack_version += 1;
        }
    }

    /// Diagnostic-only fault injection after GPU completion but before any of
    /// the three active states or the host output are published.
    #[cfg(any(test, debug_assertions))]
    #[doc(hidden)]
    pub fn inject_failure_after_scratch_execution_for_testing(
        &mut self,
        hidden: &[f32],
    ) -> Result<(), MetalW8Error> {
        self.validate_decode_input(hidden)?;
        let execution = self.inner.decode(hidden, &mut self.output, true);
        self.record_execution(&execution);
        if execution.result.is_err() {
            self.terminal_error = true;
            self.stats.terminal_error = true;
        }
        execution.result
    }
}

fn validate_precision_v1(
    index: usize,
    weights: &PackedW8LinearLayerBlock,
) -> Result<(), MetalW8Error> {
    for (label, actual, expected) in [
        (
            "GDN input projection",
            weights.gdn.input_projection.group_size(),
            W8GroupSize::G64,
        ),
        (
            "GDN output projection",
            weights.gdn.output_projection.group_size(),
            W8GroupSize::G32,
        ),
        (
            "MLP gate/up projection",
            weights.mlp.gate_up.group_size(),
            W8GroupSize::G64,
        ),
        (
            "MLP down projection",
            weights.mlp.down.group_size(),
            W8GroupSize::G64,
        ),
    ] {
        if actual != expected {
            return Err(MetalW8Error::new(format!(
                "Metal W8 stack3 v1 layer {index} {label} requires group size {}, got {}",
                expected.columns(),
                actual.columns()
            )));
        }
    }
    Ok(())
}

fn validate_states(
    states: &[GdnDecodeState; STACK_DEPTH],
    dims: GdnDimensions,
) -> Result<(), MetalW8Error> {
    let expected_query = dims.conv_kernel_size * dims.key_width();
    let expected_value = dims.conv_kernel_size * dims.value_width();
    let expected_recurrent = dims.value_heads * dims.key_dim * dims.value_dim;
    for (slot, state) in states.iter().enumerate() {
        for (label, actual, expected) in [
            ("query", state.query_conv().len(), expected_query),
            ("key", state.key_conv().len(), expected_query),
            ("value", state.value_conv().len(), expected_value),
            ("recurrent", state.recurrent().len(), expected_recurrent),
        ] {
            if actual != expected {
                return Err(MetalW8Error::new(format!(
                    "Metal W8 stack3 v1 state slot {slot} {label} has {actual} elements, expected {expected}"
                )));
            }
        }
        for (label, values) in [
            ("query", state.query_conv()),
            ("key", state.key_conv()),
            ("value", state.value_conv()),
            ("recurrent", state.recurrent()),
        ] {
            if let Some(element) = values.iter().position(|value| !value.is_finite()) {
                return Err(MetalW8Error::new(format!(
                    "Metal W8 stack3 v1 state slot {slot} {label} contains a non-finite value at element {element}"
                )));
            }
        }
    }
    Ok(())
}

fn validate_stack3_state_abi(dims: GdnDimensions) -> Result<(), MetalW8Error> {
    fn count_and_bytes(label: &str, factors: &[usize]) -> Result<(), MetalW8Error> {
        let count = factors.iter().try_fold(1usize, |product, &factor| {
            product.checked_mul(factor).ok_or_else(|| {
                MetalW8Error::new(format!("Metal W8 stack3 v1 {label} element count overflow"))
            })
        })?;
        if count > u32::MAX as usize {
            return Err(MetalW8Error::new(format!(
                "Metal W8 stack3 v1 {label} element count {count} exceeds the u32 ABI"
            )));
        }
        count
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| {
                MetalW8Error::new(format!("Metal W8 stack3 v1 {label} byte count overflow"))
            })?;
        Ok(())
    }

    count_and_bytes(
        "query state",
        &[dims.key_heads, dims.key_dim, dims.conv_kernel_size],
    )?;
    count_and_bytes(
        "value state",
        &[dims.value_heads, dims.value_dim, dims.conv_kernel_size],
    )?;
    count_and_bytes(
        "recurrent state",
        &[dims.value_heads, dims.key_dim, dims.value_dim],
    )?;
    Ok(())
}

fn stack_buffer_ledger(
    weights: [&PackedW8LinearLayerBlock; STACK_DEPTH],
) -> Result<LinearLayerStack3BufferLedger, MetalW8Error> {
    let layer_ledgers = weights
        .iter()
        .map(|weights| weights.buffer_ledger())
        .collect::<Result<Vec<_>, _>>()?;
    let packed_weight_bytes = checked_sum(
        &layer_ledgers
            .iter()
            .map(|ledger| ledger.packed_weight_bytes)
            .collect::<Vec<_>>(),
        "stack3 packed weight byte ledger",
    )?;
    let packed_scale_bytes = checked_sum(
        &layer_ledgers
            .iter()
            .map(|ledger| ledger.packed_scale_bytes)
            .collect::<Vec<_>>(),
        "stack3 packed scale byte ledger",
    )?;
    let f32_parameter_bytes = checked_sum(
        &layer_ledgers
            .iter()
            .map(|ledger| ledger.f32_parameter_bytes)
            .collect::<Vec<_>>(),
        "stack3 F32 parameter byte ledger",
    )?;
    let active_state_bytes = checked_sum(
        &layer_ledgers
            .iter()
            .map(|ledger| ledger.active_state_bytes)
            .collect::<Vec<_>>(),
        "stack3 active state byte ledger",
    )?;
    let scratch_state_bytes = checked_sum(
        &layer_ledgers
            .iter()
            .map(|ledger| ledger.scratch_state_bytes)
            .collect::<Vec<_>>(),
        "stack3 scratch state byte ledger",
    )?;
    // Equal dimensions let the stack reuse one private activation set and two
    // shared hidden rows across all three encoders.
    let activation_bytes = layer_ledgers[0].activation_bytes;
    let total_persistent_bytes = checked_sum(
        &[
            packed_weight_bytes,
            packed_scale_bytes,
            f32_parameter_bytes,
            active_state_bytes,
            scratch_state_bytes,
            activation_bytes,
        ],
        "stack3 total persistent byte ledger",
    )?;
    let hidden_bytes = f32_bytes(
        weights[0].hidden_size(),
        "stack3 hidden transfer byte ledger",
    )?;
    Ok(LinearLayerStack3BufferLedger {
        allocated_buffers: 76,
        shared_buffers: 68,
        private_buffers: 8,
        packed_weight_bytes,
        packed_scale_bytes,
        f32_parameter_bytes,
        active_state_bytes,
        scratch_state_bytes,
        activation_bytes,
        total_persistent_bytes,
        host_input_bytes_per_decode: hidden_bytes,
        host_output_bytes_per_decode: hidden_bytes,
        state_host_transfer_bytes_per_decode: 0,
        command_buffers_per_decode: 1,
        compute_encoders_per_decode: 3,
        commits_per_decode: 1,
        waits_per_decode: 1,
        intermediate_host_finite_checks_per_decode: 0,
        final_output_finite_checks_per_decode: 1,
    })
}

#[cfg(test)]
mod tests {
    use super::{validate_stack3_state_abi, GdnDimensions};

    #[test]
    fn stack3_v1_bridge_source_matches_the_versioned_transaction_and_buffer_ledger() {
        let bridge = include_str!("../metal_w8_linear_layer_stack3_bridge.mm");
        assert!(bridge.contains("apxinf_metal_w8_linear_layer_stack3_create_gdn_out_g32_v1("));
        assert!(bridge.contains(
            "apxinf_metal_w8_linear_layer_stack3_create_gdn_out_g32_with_scale_load_profile_v1("
        ));
        assert!(
            bridge.contains("apxinf_metal_w8_linear_layer_stack3_observed_scale_load_profile_v1(")
        );
        assert!(bridge.contains("apxinf_metal_w8_linear_layer_stack3_seed_states_v1("));
        assert!(bridge.contains("apxinf_metal_w8_linear_layer_stack3_decode_v1("));
        assert!(bridge.contains("apxinf_metal_w8_linear_layer_stack3_snapshot_state_v1("));
        assert!(bridge.contains("apxinf_metal_w8_linear_layer_stack3_destroy_v1("));

        let layer_buffers = bridge
            .split("struct Stack3Layer {")
            .nth(1)
            .unwrap()
            .split("GdnParams gdn_params;")
            .next()
            .unwrap();
        assert_eq!(layer_buffers.matches("id<MTLBuffer>").count(), 22);
        let shared_scratch_buffers = bridge
            .split("struct ApxinfMetalW8LinearLayerStack3HandleV1 {")
            .nth(1)
            .unwrap()
            .split("uint32_t seeded_mask;")
            .next()
            .unwrap()
            .split("Stack3Layer layers[kStackDepth];")
            .nth(1)
            .unwrap();
        assert_eq!(shared_scratch_buffers.matches("id<MTLBuffer>").count(), 10);
        assert_eq!(bridge.matches("[handle->queue commandBuffer]").count(), 1);
        assert_eq!(bridge.matches("[command computeCommandEncoder]").count(), 1);
        assert_eq!(bridge.matches("[command commit]").count(), 1);
        assert_eq!(bridge.matches("[command waitUntilCompleted]").count(), 1);
        assert!(bridge.contains("for (uint32_t slot = 0; slot < kStackDepth; ++slot)"));
        assert!(bridge.contains("receipt->state_commits = kStackDepth;"));
        assert!(bridge.contains("receipt->state_commit_mask = kAllSeededMask;"));
        assert!(bridge.contains("Only the final hidden_b row is checked"));

        let candidate = include_str!("../metal_w8_body_scale_broadcast.metal");
        for function in [
            "kernel void gdn_w8_input_projection_scale_broadcast(",
            "kernel void gdn_w8_output_projection_g32_scale_broadcast(",
            "kernel void w8_mlp_gate_up_scale_broadcast(",
            "kernel void w8_mlp_down_scale_broadcast(",
        ] {
            assert!(
                candidate.contains(function),
                "missing candidate function {function}"
            );
        }
        for legacy in [
            include_str!("../metal_w8_gdn.metal"),
            include_str!("../metal_w8_gdn_out_g32.metal"),
            include_str!("../metal_w8_mlp.metal"),
            include_str!("../metal_w8_linear_layer.metal"),
        ] {
            assert!(!legacy.contains("scale_broadcast"));
        }
        let build = include_str!("../../build.rs");
        assert!(build.contains(
            "{gdn_shader}\\n{mlp_shader}\\n{linear_layer_shader}\\n{gdn_out_g32_shader}\\n{body_scale_broadcast_shader}"
        ));
    }

    #[test]
    fn stack3_v1_rejects_state_element_counts_above_the_u32_abi_before_create() {
        let too_many_key_heads = u32::MAX as usize / 32 + 1;
        let dims = GdnDimensions {
            hidden_size: 64,
            key_heads: too_many_key_heads,
            value_heads: too_many_key_heads,
            key_dim: 32,
            value_dim: 32,
            conv_kernel_size: 1,
            rms_norm_eps: 1.0e-6,
        };

        let error = validate_stack3_state_abi(dims).unwrap_err();

        assert!(error.to_string().contains("query state element count"));
        assert!(error.to_string().contains("u32 ABI"));
    }
}

#[cfg(target_os = "macos")]
mod platform {
    use super::{
        GdnDecodeState, GdnDimensions, MetalW8Error, PackedW8LinearLayerBlock,
        W8ScaleLoadProfileV1, STACK_DEPTH,
    };
    use std::ffi::{c_char, c_int, c_void, CStr};
    use std::ptr::NonNull;

    const ERROR_CAPACITY: usize = 1024;

    #[repr(C)]
    struct Stack3LayerDescriptorV1 {
        gdn_input_weights: *const i8,
        gdn_input_scales: *const f32,
        gdn_output_weights: *const i8,
        gdn_output_scales: *const f32,
        conv_weight: *const f32,
        a_log: *const f32,
        dt_bias: *const f32,
        gdn_norm_weight: *const f32,
        mlp_gate_up_weights: *const i8,
        mlp_gate_up_scales: *const f32,
        mlp_down_weights: *const i8,
        mlp_down_scales: *const f32,
        input_rms_weight: *const f32,
        post_attention_rms_weight: *const f32,
        hidden_size: u32,
        key_heads: u32,
        value_heads: u32,
        key_dim: u32,
        value_dim: u32,
        conv_kernel_size: u32,
        gdn_rms_norm_eps: f32,
        intermediate_size: u32,
        layer_rms_norm_eps: f32,
    }

    impl Stack3LayerDescriptorV1 {
        fn from_packed(weights: &PackedW8LinearLayerBlock) -> Self {
            let dims = weights.gdn.dimensions();
            Self {
                gdn_input_weights: weights.gdn.input_projection.values().as_ptr(),
                gdn_input_scales: weights.gdn.input_projection.scales().as_ptr(),
                gdn_output_weights: weights.gdn.output_projection.values().as_ptr(),
                gdn_output_scales: weights.gdn.output_projection.scales().as_ptr(),
                conv_weight: weights.gdn.conv_weight.as_ptr(),
                a_log: weights.gdn.a_log.as_ptr(),
                dt_bias: weights.gdn.dt_bias.as_ptr(),
                gdn_norm_weight: weights.gdn.norm_weight.as_ptr(),
                mlp_gate_up_weights: weights.mlp.gate_up.values().as_ptr(),
                mlp_gate_up_scales: weights.mlp.gate_up.scales().as_ptr(),
                mlp_down_weights: weights.mlp.down.values().as_ptr(),
                mlp_down_scales: weights.mlp.down.scales().as_ptr(),
                input_rms_weight: weights.input_rms_weight.as_ptr(),
                post_attention_rms_weight: weights.post_attention_rms_weight.as_ptr(),
                hidden_size: dims.hidden_size as u32,
                key_heads: dims.key_heads as u32,
                value_heads: dims.value_heads as u32,
                key_dim: dims.key_dim as u32,
                value_dim: dims.value_dim as u32,
                conv_kernel_size: dims.conv_kernel_size as u32,
                gdn_rms_norm_eps: dims.rms_norm_eps,
                intermediate_size: weights.intermediate_size() as u32,
                layer_rms_norm_eps: weights.rms_norm_eps,
            }
        }
    }

    #[repr(C)]
    struct Stack3StateDescriptorV1 {
        query_state: *const f32,
        query_count: u32,
        key_state: *const f32,
        key_count: u32,
        value_state: *const f32,
        value_count: u32,
        recurrent_state: *const f32,
        recurrent_count: u32,
    }

    #[repr(C)]
    struct Stack3MutableStateDescriptorV1 {
        query_state: *mut f32,
        query_count: u32,
        key_state: *mut f32,
        key_count: u32,
        value_state: *mut f32,
        value_count: u32,
        recurrent_state: *mut f32,
        recurrent_count: u32,
    }

    #[repr(C)]
    #[derive(Clone, Copy, Debug, Default)]
    pub(super) struct Stack3ExecutionReceipt {
        pub(super) host_to_device_bytes: u64,
        pub(super) device_to_host_bytes: u64,
        pub(super) command_buffers: u32,
        pub(super) compute_encoders: u32,
        pub(super) commits: u32,
        pub(super) waits: u32,
        pub(super) state_commits: u32,
        pub(super) state_commit_mask: u32,
    }

    pub(super) struct Stack3Execution {
        pub(super) receipt: Stack3ExecutionReceipt,
        pub(super) result: Result<(), MetalW8Error>,
    }

    extern "C" {
        fn apxinf_metal_w8_linear_layer_stack3_create_gdn_out_g32_with_scale_load_profile_v1(
            layers: *const Stack3LayerDescriptorV1,
            layer_count: u32,
            scale_load_profile: u32,
            output: *mut *mut c_void,
            error: *mut c_char,
            error_capacity: usize,
        ) -> c_int;
        fn apxinf_metal_w8_linear_layer_stack3_observed_scale_load_profile_v1(
            handle: *mut c_void,
            profile: *mut u32,
            error: *mut c_char,
            error_capacity: usize,
        ) -> c_int;
        fn apxinf_metal_w8_linear_layer_stack3_create_gdn_out_g32_v1(
            layers: *const Stack3LayerDescriptorV1,
            layer_count: u32,
            output: *mut *mut c_void,
            error: *mut c_char,
            error_capacity: usize,
        ) -> c_int;
        fn apxinf_metal_w8_linear_layer_stack3_seed_states_v1(
            handle: *mut c_void,
            states: *const Stack3StateDescriptorV1,
            state_count: u32,
            error: *mut c_char,
            error_capacity: usize,
        ) -> c_int;
        fn apxinf_metal_w8_linear_layer_stack3_decode_v1(
            handle: *mut c_void,
            input: *const f32,
            input_count: u32,
            output: *mut f32,
            output_count: u32,
            inject_failure_after_execution: u8,
            receipt: *mut Stack3ExecutionReceipt,
            error: *mut c_char,
            error_capacity: usize,
        ) -> c_int;
        fn apxinf_metal_w8_linear_layer_stack3_snapshot_state_v1(
            handle: *mut c_void,
            slot: u32,
            state: *mut Stack3MutableStateDescriptorV1,
            error: *mut c_char,
            error_capacity: usize,
        ) -> c_int;
        fn apxinf_metal_w8_linear_layer_stack3_destroy_v1(handle: *mut c_void);
    }

    pub(super) struct LinearLayerStack3Handle(NonNull<c_void>);

    impl LinearLayerStack3Handle {
        pub(super) fn new(
            weights: [&PackedW8LinearLayerBlock; STACK_DEPTH],
            scale_load_profile: W8ScaleLoadProfileV1,
        ) -> Result<Self, MetalW8Error> {
            let descriptors = weights.map(Stack3LayerDescriptorV1::from_packed);
            let mut output = std::ptr::null_mut();
            let mut error = [0 as c_char; ERROR_CAPACITY];
            let status = unsafe {
                match scale_load_profile {
                    W8ScaleLoadProfileV1::LegacyPerLane => {
                        apxinf_metal_w8_linear_layer_stack3_create_gdn_out_g32_v1(
                            descriptors.as_ptr(),
                            descriptors.len() as u32,
                            &mut output,
                            error.as_mut_ptr(),
                            error.len(),
                        )
                    }
                    W8ScaleLoadProfileV1::SimdBroadcast => {
                        apxinf_metal_w8_linear_layer_stack3_create_gdn_out_g32_with_scale_load_profile_v1(
                            descriptors.as_ptr(),
                            descriptors.len() as u32,
                            scale_load_profile.selector(),
                            &mut output,
                            error.as_mut_ptr(),
                            error.len(),
                        )
                    }
                }
            };
            if status != 0 {
                return Err(bridge_error("create Metal W8 stack3 v1", &error));
            }
            NonNull::new(output).map(Self).ok_or_else(|| {
                MetalW8Error::new("create Metal W8 stack3 v1 returned a null handle")
            })
        }

        pub(super) fn observed_scale_load_profile(
            &self,
        ) -> Result<W8ScaleLoadProfileV1, MetalW8Error> {
            let mut profile = u32::MAX;
            let mut error = [0 as c_char; ERROR_CAPACITY];
            let status = unsafe {
                apxinf_metal_w8_linear_layer_stack3_observed_scale_load_profile_v1(
                    self.0.as_ptr(),
                    &mut profile,
                    error.as_mut_ptr(),
                    error.len(),
                )
            };
            if status != 0 {
                return Err(bridge_error(
                    "read Metal W8 stack3 v1 scale-load profile",
                    &error,
                ));
            }
            W8ScaleLoadProfileV1::try_from(profile)
        }

        pub(super) fn seed(
            &mut self,
            states: &[GdnDecodeState; STACK_DEPTH],
        ) -> Result<(), MetalW8Error> {
            let descriptors = states.each_ref().map(|state| Stack3StateDescriptorV1 {
                query_state: state.query_conv().as_ptr(),
                query_count: state.query_conv().len() as u32,
                key_state: state.key_conv().as_ptr(),
                key_count: state.key_conv().len() as u32,
                value_state: state.value_conv().as_ptr(),
                value_count: state.value_conv().len() as u32,
                recurrent_state: state.recurrent().as_ptr(),
                recurrent_count: state.recurrent().len() as u32,
            });
            let mut error = [0 as c_char; ERROR_CAPACITY];
            let status = unsafe {
                apxinf_metal_w8_linear_layer_stack3_seed_states_v1(
                    self.0.as_ptr(),
                    descriptors.as_ptr(),
                    descriptors.len() as u32,
                    error.as_mut_ptr(),
                    error.len(),
                )
            };
            if status == 0 {
                Ok(())
            } else {
                Err(bridge_error("seed Metal W8 stack3 v1 states", &error))
            }
        }

        pub(super) fn decode(
            &mut self,
            input: &[f32],
            output: &mut [f32],
            inject_failure: bool,
        ) -> Stack3Execution {
            let mut receipt = Stack3ExecutionReceipt::default();
            let mut error = [0 as c_char; ERROR_CAPACITY];
            let status = unsafe {
                apxinf_metal_w8_linear_layer_stack3_decode_v1(
                    self.0.as_ptr(),
                    input.as_ptr(),
                    input.len() as u32,
                    output.as_mut_ptr(),
                    output.len() as u32,
                    u8::from(inject_failure),
                    &mut receipt,
                    error.as_mut_ptr(),
                    error.len(),
                )
            };
            let result = if status == 0 {
                Ok(())
            } else {
                Err(bridge_error("run Metal W8 stack3 v1 decode", &error))
            };
            Stack3Execution { receipt, result }
        }

        pub(super) fn snapshot(
            &self,
            slot: usize,
            dims: GdnDimensions,
        ) -> Result<GdnDecodeState, MetalW8Error> {
            let mut query = vec![0.0f32; dims.conv_kernel_size * dims.key_width()];
            let mut key = vec![0.0f32; dims.conv_kernel_size * dims.key_width()];
            let mut value = vec![0.0f32; dims.conv_kernel_size * dims.value_width()];
            let mut recurrent = vec![0.0f32; dims.value_heads * dims.key_dim * dims.value_dim];
            let mut descriptor = Stack3MutableStateDescriptorV1 {
                query_state: query.as_mut_ptr(),
                query_count: query.len() as u32,
                key_state: key.as_mut_ptr(),
                key_count: key.len() as u32,
                value_state: value.as_mut_ptr(),
                value_count: value.len() as u32,
                recurrent_state: recurrent.as_mut_ptr(),
                recurrent_count: recurrent.len() as u32,
            };
            let mut error = [0 as c_char; ERROR_CAPACITY];
            let status = unsafe {
                apxinf_metal_w8_linear_layer_stack3_snapshot_state_v1(
                    self.0.as_ptr(),
                    slot as u32,
                    &mut descriptor,
                    error.as_mut_ptr(),
                    error.len(),
                )
            };
            if status != 0 {
                return Err(bridge_error("snapshot Metal W8 stack3 v1 state", &error));
            }
            GdnDecodeState::from_parts(dims, query, key, value, recurrent)
        }
    }

    impl Drop for LinearLayerStack3Handle {
        fn drop(&mut self) {
            unsafe { apxinf_metal_w8_linear_layer_stack3_destroy_v1(self.0.as_ptr()) };
        }
    }

    fn bridge_error(context: &str, buffer: &[c_char]) -> MetalW8Error {
        let detail = unsafe { CStr::from_ptr(buffer.as_ptr()) }
            .to_string_lossy()
            .into_owned();
        if detail.is_empty() {
            MetalW8Error::new(context)
        } else {
            MetalW8Error::new(format!("{context}: {detail}"))
        }
    }
}

#[cfg(not(target_os = "macos"))]
mod platform {
    use super::{
        GdnDecodeState, GdnDimensions, MetalW8Error, PackedW8LinearLayerBlock,
        W8ScaleLoadProfileV1, STACK_DEPTH,
    };

    #[derive(Clone, Copy, Debug, Default)]
    pub(super) struct Stack3ExecutionReceipt {
        pub(super) host_to_device_bytes: u64,
        pub(super) device_to_host_bytes: u64,
        pub(super) command_buffers: u32,
        pub(super) compute_encoders: u32,
        pub(super) commits: u32,
        pub(super) waits: u32,
        pub(super) state_commits: u32,
        pub(super) state_commit_mask: u32,
    }

    pub(super) struct Stack3Execution {
        pub(super) receipt: Stack3ExecutionReceipt,
        pub(super) result: Result<(), MetalW8Error>,
    }

    pub(super) struct LinearLayerStack3Handle;

    impl LinearLayerStack3Handle {
        pub(super) fn new(
            _weights: [&PackedW8LinearLayerBlock; STACK_DEPTH],
            _scale_load_profile: W8ScaleLoadProfileV1,
        ) -> Result<Self, MetalW8Error> {
            Err(MetalW8Error::new("Metal W8 stack3 v1 requires macOS"))
        }

        pub(super) fn observed_scale_load_profile(
            &self,
        ) -> Result<W8ScaleLoadProfileV1, MetalW8Error> {
            Err(MetalW8Error::new("Metal W8 stack3 v1 requires macOS"))
        }

        pub(super) fn seed(
            &mut self,
            _states: &[GdnDecodeState; STACK_DEPTH],
        ) -> Result<(), MetalW8Error> {
            Err(MetalW8Error::new("Metal W8 stack3 v1 requires macOS"))
        }

        pub(super) fn decode(
            &mut self,
            _input: &[f32],
            _output: &mut [f32],
            _inject_failure: bool,
        ) -> Stack3Execution {
            Stack3Execution {
                receipt: Stack3ExecutionReceipt::default(),
                result: Err(MetalW8Error::new("Metal W8 stack3 v1 requires macOS")),
            }
        }

        pub(super) fn snapshot(
            &self,
            _slot: usize,
            _dims: GdnDimensions,
        ) -> Result<GdnDecodeState, MetalW8Error> {
            Err(MetalW8Error::new("Metal W8 stack3 v1 requires macOS"))
        }
    }
}
