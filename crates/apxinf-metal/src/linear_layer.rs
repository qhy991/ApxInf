use crate::{
    GdnDecodeResult, GdnDecodeState, GdnDimensions, MetalW8Error, PackedW8GdnBlock,
    PackedW8MlpBlock, W8GroupSize,
};

mod mlp_stack3_boundary;
mod stack3;

pub use mlp_stack3_boundary::{
    MetalW8MlpStack3BoundaryV1, MlpStack3BoundaryBufferLedgerV1, MlpStack3BoundaryDecodeResultV1,
    MlpStack3BoundaryMetalStatsV1, PackedW8MlpStack3BoundaryV1,
};
pub use stack3::{
    LinearLayerStack3BufferLedger, LinearLayerStack3MetalStats, MetalW8LinearLayerStack3,
};

#[derive(Clone, Debug)]
pub struct PackedW8LinearLayerBlock {
    gdn: PackedW8GdnBlock,
    mlp: PackedW8MlpBlock,
    input_rms_weight: Vec<f32>,
    post_attention_rms_weight: Vec<f32>,
    rms_norm_eps: f32,
}

pub struct LinearLayerDecodeResult {
    pub output: Vec<f32>,
    pub state: GdnDecodeState,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LinearLayerBufferLedger {
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
}

/// Exact CPU packed-weight precision receipt for one complete linear layer.
/// Gate and up are stored in one buffer but reported separately here.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LinearLayerQuantizationLedger {
    pub gdn_input_group_size: W8GroupSize,
    pub gdn_input_weight_bytes: usize,
    pub gdn_input_scale_bytes: usize,
    pub gdn_output_group_size: W8GroupSize,
    pub gdn_output_weight_bytes: usize,
    pub gdn_output_scale_bytes: usize,
    pub mlp_gate_group_size: W8GroupSize,
    pub mlp_gate_weight_bytes: usize,
    pub mlp_gate_scale_bytes: usize,
    pub mlp_up_group_size: W8GroupSize,
    pub mlp_up_weight_bytes: usize,
    pub mlp_up_scale_bytes: usize,
    pub mlp_down_group_size: W8GroupSize,
    pub mlp_down_weight_bytes: usize,
    pub mlp_down_scale_bytes: usize,
    pub total_packed_weight_bytes: usize,
    pub total_packed_scale_bytes: usize,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LinearLayerMetalStats {
    pub decode_calls: usize,
    pub successful_decodes: usize,
    pub failed_decodes: usize,
    pub command_buffers: usize,
    pub compute_encoders: usize,
    pub commits: usize,
    pub waits: usize,
    pub host_to_device_bytes: usize,
    pub device_to_host_bytes: usize,
    pub committed_state_version: u64,
}

/// Independent, decode-only linear-attention layer tracer. It owns one Metal
/// handle for both residual branches; no model loader or default runtime
/// constructs this type.
pub struct MetalW8LinearLayerBlock {
    dims: GdnDimensions,
    inner: platform::LinearLayerHandle,
    output: Vec<f32>,
    seeded: bool,
    stats: LinearLayerMetalStats,
    buffer_ledger: LinearLayerBufferLedger,
}

impl PackedW8LinearLayerBlock {
    pub fn new(
        gdn: PackedW8GdnBlock,
        mlp: PackedW8MlpBlock,
        input_rms_weight: &[f32],
        post_attention_rms_weight: &[f32],
        rms_norm_eps: f32,
    ) -> Result<Self, MetalW8Error> {
        let hidden_size = gdn.dimensions().hidden_size;
        if mlp.down.rows != hidden_size || mlp.gate_up.columns != hidden_size {
            return Err(MetalW8Error::new(
                "Metal W8 linear layer GDN and MLP hidden sizes differ",
            ));
        }
        for (label, weight) in [
            ("input RMSNorm", input_rms_weight),
            ("post-attention RMSNorm", post_attention_rms_weight),
        ] {
            if weight.len() != hidden_size {
                return Err(MetalW8Error::new(format!(
                    "Metal W8 linear layer {label} weight has {} elements, expected {hidden_size}",
                    weight.len()
                )));
            }
            if let Some(index) = weight.iter().position(|value| !value.is_finite()) {
                return Err(MetalW8Error::new(format!(
                    "Metal W8 linear layer {label} weight contains a non-finite value at element {index}"
                )));
            }
        }
        if !rms_norm_eps.is_finite() || rms_norm_eps < 0.0 {
            return Err(MetalW8Error::new(
                "Metal W8 linear layer RMS epsilon must be finite and non-negative",
            ));
        }
        Ok(Self {
            gdn,
            mlp,
            input_rms_weight: input_rms_weight.to_vec(),
            post_attention_rms_weight: post_attention_rms_weight.to_vec(),
            rms_norm_eps,
        })
    }

    pub fn hidden_size(&self) -> usize {
        self.gdn.dimensions().hidden_size
    }

    pub fn intermediate_size(&self) -> usize {
        self.mlp.down.columns
    }

    pub fn quantization_ledger(&self) -> Result<LinearLayerQuantizationLedger, MetalW8Error> {
        if self.mlp.gate_up.values().len() % 2 != 0 || self.mlp.gate_up.scales().len() % 2 != 0 {
            return Err(MetalW8Error::new(
                "Metal W8 linear layer gate/up packing cannot be split evenly",
            ));
        }
        let gdn_input_weight_bytes = self.gdn.input_projection.values().len();
        let gdn_output_weight_bytes = self.gdn.output_projection.values().len();
        let mlp_gate_weight_bytes = self.mlp.gate_up.values().len() / 2;
        let mlp_up_weight_bytes = mlp_gate_weight_bytes;
        let mlp_down_weight_bytes = self.mlp.down.values().len();
        let gdn_input_scale_bytes = f32_bytes(
            self.gdn.input_projection.scales().len(),
            "GDN input scale byte ledger",
        )?;
        let gdn_output_scale_bytes = f32_bytes(
            self.gdn.output_projection.scales().len(),
            "GDN output scale byte ledger",
        )?;
        let mlp_gate_scale_bytes = f32_bytes(
            self.mlp.gate_up.scales().len() / 2,
            "MLP gate scale byte ledger",
        )?;
        let mlp_up_scale_bytes = mlp_gate_scale_bytes;
        let mlp_down_scale_bytes =
            f32_bytes(self.mlp.down.scales().len(), "MLP down scale byte ledger")?;
        let total_packed_weight_bytes = checked_sum(
            &[
                gdn_input_weight_bytes,
                gdn_output_weight_bytes,
                mlp_gate_weight_bytes,
                mlp_up_weight_bytes,
                mlp_down_weight_bytes,
            ],
            "total packed weight byte ledger",
        )?;
        let total_packed_scale_bytes = checked_sum(
            &[
                gdn_input_scale_bytes,
                gdn_output_scale_bytes,
                mlp_gate_scale_bytes,
                mlp_up_scale_bytes,
                mlp_down_scale_bytes,
            ],
            "total packed scale byte ledger",
        )?;
        Ok(LinearLayerQuantizationLedger {
            gdn_input_group_size: self.gdn.input_projection.group_size(),
            gdn_input_weight_bytes,
            gdn_input_scale_bytes,
            gdn_output_group_size: self.gdn.output_projection.group_size(),
            gdn_output_weight_bytes,
            gdn_output_scale_bytes,
            mlp_gate_group_size: self.mlp.gate_up.group_size(),
            mlp_gate_weight_bytes,
            mlp_gate_scale_bytes,
            mlp_up_group_size: self.mlp.gate_up.group_size(),
            mlp_up_weight_bytes,
            mlp_up_scale_bytes,
            mlp_down_group_size: self.mlp.down.group_size(),
            mlp_down_weight_bytes,
            mlp_down_scale_bytes,
            total_packed_weight_bytes,
            total_packed_scale_bytes,
        })
    }

    pub fn buffer_ledger(&self) -> Result<LinearLayerBufferLedger, MetalW8Error> {
        let dims = self.gdn.dimensions();
        let intermediate_size = self.intermediate_size();
        let quantization = self.quantization_ledger()?;
        let packed_weight_bytes = quantization.total_packed_weight_bytes;
        let packed_scale_bytes = quantization.total_packed_scale_bytes;
        let parameter_elements = checked_sum(
            &[
                self.gdn.conv_weight.len(),
                self.gdn.a_log.len(),
                self.gdn.dt_bias.len(),
                self.gdn.norm_weight.len(),
                self.input_rms_weight.len(),
                self.post_attention_rms_weight.len(),
            ],
            "F32 parameter ledger",
        )?;
        let f32_parameter_bytes = f32_bytes(parameter_elements, "F32 parameter byte ledger")?;
        let key_conv_elements = dims
            .key_width()
            .checked_mul(dims.conv_kernel_size)
            .ok_or_else(|| MetalW8Error::new("Metal W8 linear layer state ledger overflow"))?;
        let value_conv_elements = dims
            .value_width()
            .checked_mul(dims.conv_kernel_size)
            .ok_or_else(|| MetalW8Error::new("Metal W8 linear layer state ledger overflow"))?;
        let recurrent_elements = dims
            .value_heads
            .checked_mul(dims.key_dim)
            .and_then(|value| value.checked_mul(dims.value_dim))
            .ok_or_else(|| MetalW8Error::new("Metal W8 linear layer state ledger overflow"))?;
        let state_elements = checked_sum(
            &[
                key_conv_elements,
                key_conv_elements,
                value_conv_elements,
                recurrent_elements,
            ],
            "state ledger",
        )?;
        let active_state_bytes = f32_bytes(state_elements, "active state byte ledger")?;
        let scratch_state_bytes = active_state_bytes;

        // Two shared H rows (input/output), one reusable normalized H row, one
        // reusable branch-output H row, and the resident GDN/MLP intermediates.
        let activation_elements = checked_sum(
            &[
                dims.hidden_size
                    .checked_mul(4)
                    .ok_or_else(|| MetalW8Error::new("activation ledger overflow"))?,
                dims.input_projection_rows(),
                dims.qkv_width(),
                dims.value_width(),
                dims.value_width(),
                intermediate_size
                    .checked_mul(3)
                    .ok_or_else(|| MetalW8Error::new("activation ledger overflow"))?,
            ],
            "activation ledger",
        )?;
        let activation_bytes = f32_bytes(activation_elements, "activation byte ledger")?;
        let total_persistent_bytes = checked_sum(
            &[
                packed_weight_bytes,
                packed_scale_bytes,
                f32_parameter_bytes,
                active_state_bytes,
                scratch_state_bytes,
                activation_bytes,
            ],
            "total persistent byte ledger",
        )?;
        let hidden_bytes = f32_bytes(dims.hidden_size, "hidden transfer byte ledger")?;
        Ok(LinearLayerBufferLedger {
            allocated_buffers: 32,
            shared_buffers: 24,
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
            compute_encoders_per_decode: 1,
            commits_per_decode: 1,
            waits_per_decode: 1,
        })
    }

    pub fn decode_reference(
        &self,
        hidden: &[f32],
        state: &GdnDecodeState,
    ) -> Result<LinearLayerDecodeResult, MetalW8Error> {
        self.validate_hidden(hidden)?;
        let normalized = rms_norm(hidden, &self.input_rms_weight, self.rms_norm_eps);
        let GdnDecodeResult {
            output: attention,
            state,
        } = self.gdn.decode_reference(&normalized, state)?;
        let post_attention = hidden
            .iter()
            .zip(attention)
            .map(|(&residual, update)| residual + update)
            .collect::<Vec<_>>();
        let normalized_post_attention = rms_norm(
            &post_attention,
            &self.post_attention_rms_weight,
            self.rms_norm_eps,
        );
        let mlp = self.mlp.forward(&normalized_post_attention)?;
        let output = post_attention
            .into_iter()
            .zip(mlp)
            .map(|(residual, update)| residual + update)
            .collect();
        Ok(LinearLayerDecodeResult { output, state })
    }

    fn validate_hidden(&self, hidden: &[f32]) -> Result<(), MetalW8Error> {
        if hidden.len() != self.hidden_size() {
            return Err(MetalW8Error::new(format!(
                "Metal W8 linear layer input has {} elements, expected {}",
                hidden.len(),
                self.hidden_size()
            )));
        }
        if let Some(index) = hidden.iter().position(|value| !value.is_finite()) {
            return Err(MetalW8Error::new(format!(
                "Metal W8 linear layer input contains a non-finite value at element {index}"
            )));
        }
        Ok(())
    }
}

impl MetalW8LinearLayerBlock {
    pub fn from_packed(weights: &PackedW8LinearLayerBlock) -> Result<Self, MetalW8Error> {
        weights
            .gdn
            .input_projection
            .require_metal_g64("linear layer GDN input projection")?;
        weights
            .gdn
            .output_projection
            .require_metal_g64("linear layer GDN output projection")?;
        weights
            .mlp
            .gate_up
            .require_metal_g64("linear layer MLP gate/up projection")?;
        weights
            .mlp
            .down
            .require_metal_g64("linear layer MLP down projection")?;
        let dims = weights.gdn.dimensions();
        let intermediate_size = weights.intermediate_size();
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
                    "Metal W8 linear layer {label} exceeds the u32 ABI"
                )));
            }
        }
        let buffer_ledger = weights.buffer_ledger()?;
        Ok(Self {
            dims,
            inner: platform::LinearLayerHandle::new(weights)?,
            output: vec![0.0; dims.hidden_size],
            seeded: false,
            stats: LinearLayerMetalStats::default(),
            buffer_ledger,
        })
    }

    /// Versioned precision-v2 constructor. Its fixed contract is G64 for the
    /// GDN input and all MLP projections, and G32 only for the GDN output.
    /// The legacy constructor remains all-G64 and rejects this packing.
    pub fn from_packed_gdn_out_g32(
        weights: &PackedW8LinearLayerBlock,
    ) -> Result<Self, MetalW8Error> {
        for (label, actual, expected) in [
            (
                "linear layer GDN input projection",
                weights.gdn.input_projection.group_size(),
                W8GroupSize::G64,
            ),
            (
                "linear layer GDN output projection",
                weights.gdn.output_projection.group_size(),
                W8GroupSize::G32,
            ),
            (
                "linear layer MLP gate/up projection",
                weights.mlp.gate_up.group_size(),
                W8GroupSize::G64,
            ),
            (
                "linear layer MLP down projection",
                weights.mlp.down.group_size(),
                W8GroupSize::G64,
            ),
        ] {
            if actual != expected {
                return Err(MetalW8Error::new(format!(
                    "Metal W8 precision-v2 {label} requires group size {}, got {}",
                    expected.columns(),
                    actual.columns()
                )));
            }
        }
        let dims = weights.gdn.dimensions();
        let intermediate_size = weights.intermediate_size();
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
                    "Metal W8 precision-v2 linear layer {label} exceeds the u32 ABI"
                )));
            }
        }
        let buffer_ledger = weights.buffer_ledger()?;
        Ok(Self {
            dims,
            inner: platform::LinearLayerHandle::new_gdn_out_g32(weights)?,
            output: vec![0.0; dims.hidden_size],
            seeded: false,
            stats: LinearLayerMetalStats::default(),
            buffer_ledger,
        })
    }

    pub fn seed_decode_state(&mut self, state: &GdnDecodeState) -> Result<(), MetalW8Error> {
        self.inner.seed(state)?;
        self.seeded = true;
        self.stats = LinearLayerMetalStats::default();
        Ok(())
    }

    pub fn clear_decode_state(&mut self) -> Result<(), MetalW8Error> {
        let cleared = GdnDecodeState::zeroed(self.dims)?;
        self.inner.seed(&cleared)?;
        self.output.fill(0.0);
        self.seeded = false;
        self.stats = LinearLayerMetalStats::default();
        Ok(())
    }

    pub fn decode(&mut self, hidden: &[f32]) -> Result<&[f32], MetalW8Error> {
        self.validate_decode_input(hidden)?;
        let execution = self.inner.decode(hidden, &mut self.output, false);
        self.record_execution(&execution);
        execution.result?;
        Ok(&self.output)
    }

    pub fn state_snapshot(&self) -> Result<GdnDecodeState, MetalW8Error> {
        if !self.seeded {
            return Err(MetalW8Error::new(
                "Metal W8 linear layer state must be seeded before snapshot",
            ));
        }
        self.inner.snapshot(self.dims)
    }

    pub fn stats(&self) -> LinearLayerMetalStats {
        self.stats
    }

    pub fn buffer_ledger(&self) -> LinearLayerBufferLedger {
        self.buffer_ledger
    }

    fn validate_decode_input(&self, hidden: &[f32]) -> Result<(), MetalW8Error> {
        if !self.seeded {
            return Err(MetalW8Error::new(
                "Metal W8 linear layer decode state must be seeded after CPU prefill",
            ));
        }
        if hidden.len() != self.dims.hidden_size {
            return Err(MetalW8Error::new(format!(
                "Metal W8 linear layer input has {} elements, expected {}",
                hidden.len(),
                self.dims.hidden_size
            )));
        }
        if let Some(index) = hidden.iter().position(|value| !value.is_finite()) {
            return Err(MetalW8Error::new(format!(
                "Metal W8 linear layer input contains a non-finite value at element {index}"
            )));
        }
        Ok(())
    }

    fn record_execution(&mut self, execution: &platform::LinearLayerExecution) {
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
        self.stats.committed_state_version += execution.receipt.state_commits as u64;
    }

    /// Diagnostic-only fault injection. The complete command executes into
    /// scratch state, but the active/scratch swap and host output copy do not
    /// occur.
    #[cfg(any(test, debug_assertions))]
    #[doc(hidden)]
    pub fn inject_failure_after_scratch_execution_for_testing(
        &mut self,
        hidden: &[f32],
    ) -> Result<(), MetalW8Error> {
        self.validate_decode_input(hidden)?;
        let execution = self.inner.decode(hidden, &mut self.output, true);
        self.record_execution(&execution);
        execution.result
    }
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

fn checked_sum(values: &[usize], label: &str) -> Result<usize, MetalW8Error> {
    values.iter().try_fold(0usize, |total, &value| {
        total
            .checked_add(value)
            .ok_or_else(|| MetalW8Error::new(format!("Metal W8 linear layer {label} overflow")))
    })
}

fn f32_bytes(elements: usize, label: &str) -> Result<usize, MetalW8Error> {
    elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| MetalW8Error::new(format!("Metal W8 linear layer {label} overflow")))
}

#[cfg(test)]
mod tests {
    use sha2::{Digest, Sha256};

    #[test]
    fn bridge_source_has_one_encoder_buffer_commit_wait_and_exact_buffer_count() {
        let bridge = include_str!("metal_w8_linear_layer_bridge.mm");
        assert_eq!(bridge.matches("[handle->queue commandBuffer]").count(), 1);
        assert_eq!(bridge.matches("[command computeCommandEncoder]").count(), 1);
        assert_eq!(bridge.matches("[encoder endEncoding]").count(), 1);
        assert_eq!(bridge.matches("[command commit]").count(), 1);
        assert_eq!(bridge.matches("[command waitUntilCompleted]").count(), 1);

        let handle_buffers = bridge
            .split("struct ApxinfMetalW8LinearLayerHandle {")
            .nth(1)
            .unwrap()
            .split("GdnParams gdn_params;")
            .next()
            .unwrap();
        assert_eq!(handle_buffers.matches("id<MTLBuffer>").count(), 32);
        for kernel in [
            "linear_layer_rms_norm",
            "gdn_w8_input_projection",
            "gdn_depthwise_preprocess",
            "gdn_normalize_qk",
            "gdn_recurrent_update",
            "gdn_norm_gate",
            "gdn_w8_output_projection",
            "linear_layer_residual_add",
            "w8_mlp_gate_up",
            "w8_mlp_silu_mul",
            "w8_mlp_down",
        ] {
            assert!(bridge.contains(&format!("@\"{kernel}\"")));
        }
    }

    #[test]
    fn precision_v2_is_additive_and_preserves_the_legacy_gdn_kernel_bytes_and_abi() {
        let legacy_gdn = include_bytes!("metal_w8_gdn.metal");
        assert_eq!(
            format!("{:x}", Sha256::digest(legacy_gdn)),
            "e79b20f0ee630c2876fda87b658714362b26d3f391d499087cf075351c74fa55"
        );
        let g32 = include_str!("metal_w8_gdn_out_g32.metal");
        for binding in 0..=4 {
            assert!(g32.contains(&format!("[[buffer({binding})]]")));
        }
        assert!(g32.contains("kernel void gdn_w8_output_projection_g32("));
        assert!(g32.contains("constexpr uint rows_per_threadgroup = 8;"));
        assert!(g32.contains("constexpr uint float4_per_group = 8;"));

        let bridge = include_str!("metal_w8_linear_layer_bridge.mm");
        assert!(bridge.contains("extern \"C\" int apxinf_metal_w8_linear_layer_create("));
        assert!(
            bridge.contains("extern \"C\" int apxinf_metal_w8_linear_layer_gdn_out_g32_create(")
        );
        assert!(bridge.contains("@\"gdn_w8_output_projection\""));
        assert!(bridge.contains("@\"gdn_w8_output_projection_g32\""));
    }
}

#[cfg(target_os = "macos")]
mod platform {
    use super::{GdnDecodeState, GdnDimensions, MetalW8Error, PackedW8LinearLayerBlock};
    use crate::W8_GROUP_SIZE;
    use std::ffi::{c_char, c_int, c_void, CStr};
    use std::ptr::NonNull;

    const ERROR_CAPACITY: usize = 1024;

    #[repr(C)]
    #[derive(Clone, Copy, Debug, Default)]
    pub(super) struct LinearLayerExecutionReceipt {
        pub(super) host_to_device_bytes: u64,
        pub(super) device_to_host_bytes: u64,
        pub(super) command_buffers: u32,
        pub(super) compute_encoders: u32,
        pub(super) commits: u32,
        pub(super) waits: u32,
        pub(super) state_commits: u32,
        reserved: u32,
    }

    pub(super) struct LinearLayerExecution {
        pub(super) receipt: LinearLayerExecutionReceipt,
        pub(super) result: Result<(), MetalW8Error>,
    }

    extern "C" {
        fn apxinf_metal_w8_linear_layer_create(
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
            group_size: u32,
            output: *mut *mut c_void,
            error: *mut c_char,
            error_capacity: usize,
        ) -> c_int;
        fn apxinf_metal_w8_linear_layer_gdn_out_g32_create(
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
            output: *mut *mut c_void,
            error: *mut c_char,
            error_capacity: usize,
        ) -> c_int;
        fn apxinf_metal_w8_linear_layer_seed_state(
            handle: *mut c_void,
            query_state: *const f32,
            query_count: u32,
            key_state: *const f32,
            key_count: u32,
            value_state: *const f32,
            value_count: u32,
            recurrent_state: *const f32,
            recurrent_count: u32,
            error: *mut c_char,
            error_capacity: usize,
        ) -> c_int;
        fn apxinf_metal_w8_linear_layer_decode(
            handle: *mut c_void,
            input: *const f32,
            input_count: u32,
            output: *mut f32,
            output_count: u32,
            inject_failure_after_execution: u8,
            receipt: *mut LinearLayerExecutionReceipt,
            error: *mut c_char,
            error_capacity: usize,
        ) -> c_int;
        fn apxinf_metal_w8_linear_layer_snapshot_state(
            handle: *mut c_void,
            query_state: *mut f32,
            query_count: u32,
            key_state: *mut f32,
            key_count: u32,
            value_state: *mut f32,
            value_count: u32,
            recurrent_state: *mut f32,
            recurrent_count: u32,
            error: *mut c_char,
            error_capacity: usize,
        ) -> c_int;
        fn apxinf_metal_w8_linear_layer_destroy(handle: *mut c_void);
    }

    pub(super) struct LinearLayerHandle(NonNull<c_void>);

    impl LinearLayerHandle {
        pub(super) fn new(weights: &PackedW8LinearLayerBlock) -> Result<Self, MetalW8Error> {
            let dims = weights.gdn.dimensions();
            let mut output = std::ptr::null_mut();
            let mut error = [0 as c_char; ERROR_CAPACITY];
            let status = unsafe {
                apxinf_metal_w8_linear_layer_create(
                    weights.gdn.input_projection.values().as_ptr(),
                    weights.gdn.input_projection.scales().as_ptr(),
                    weights.gdn.output_projection.values().as_ptr(),
                    weights.gdn.output_projection.scales().as_ptr(),
                    weights.gdn.conv_weight.as_ptr(),
                    weights.gdn.a_log.as_ptr(),
                    weights.gdn.dt_bias.as_ptr(),
                    weights.gdn.norm_weight.as_ptr(),
                    weights.mlp.gate_up.values().as_ptr(),
                    weights.mlp.gate_up.scales().as_ptr(),
                    weights.mlp.down.values().as_ptr(),
                    weights.mlp.down.scales().as_ptr(),
                    weights.input_rms_weight.as_ptr(),
                    weights.post_attention_rms_weight.as_ptr(),
                    dims.hidden_size as u32,
                    dims.key_heads as u32,
                    dims.value_heads as u32,
                    dims.key_dim as u32,
                    dims.value_dim as u32,
                    dims.conv_kernel_size as u32,
                    dims.rms_norm_eps,
                    weights.intermediate_size() as u32,
                    weights.rms_norm_eps,
                    W8_GROUP_SIZE as u32,
                    &mut output,
                    error.as_mut_ptr(),
                    error.len(),
                )
            };
            if status != 0 {
                return Err(bridge_error("create Metal W8 linear layer", &error));
            }
            NonNull::new(output).map(Self).ok_or_else(|| {
                MetalW8Error::new("create Metal W8 linear layer returned a null handle")
            })
        }

        pub(super) fn new_gdn_out_g32(
            weights: &PackedW8LinearLayerBlock,
        ) -> Result<Self, MetalW8Error> {
            let dims = weights.gdn.dimensions();
            let mut output = std::ptr::null_mut();
            let mut error = [0 as c_char; ERROR_CAPACITY];
            let status = unsafe {
                apxinf_metal_w8_linear_layer_gdn_out_g32_create(
                    weights.gdn.input_projection.values().as_ptr(),
                    weights.gdn.input_projection.scales().as_ptr(),
                    weights.gdn.output_projection.values().as_ptr(),
                    weights.gdn.output_projection.scales().as_ptr(),
                    weights.gdn.conv_weight.as_ptr(),
                    weights.gdn.a_log.as_ptr(),
                    weights.gdn.dt_bias.as_ptr(),
                    weights.gdn.norm_weight.as_ptr(),
                    weights.mlp.gate_up.values().as_ptr(),
                    weights.mlp.gate_up.scales().as_ptr(),
                    weights.mlp.down.values().as_ptr(),
                    weights.mlp.down.scales().as_ptr(),
                    weights.input_rms_weight.as_ptr(),
                    weights.post_attention_rms_weight.as_ptr(),
                    dims.hidden_size as u32,
                    dims.key_heads as u32,
                    dims.value_heads as u32,
                    dims.key_dim as u32,
                    dims.value_dim as u32,
                    dims.conv_kernel_size as u32,
                    dims.rms_norm_eps,
                    weights.intermediate_size() as u32,
                    weights.rms_norm_eps,
                    &mut output,
                    error.as_mut_ptr(),
                    error.len(),
                )
            };
            if status != 0 {
                return Err(bridge_error(
                    "create Metal W8 precision-v2 GDN-out-G32 linear layer",
                    &error,
                ));
            }
            NonNull::new(output).map(Self).ok_or_else(|| {
                MetalW8Error::new(
                    "create Metal W8 precision-v2 GDN-out-G32 linear layer returned a null handle",
                )
            })
        }

        pub(super) fn seed(&mut self, state: &GdnDecodeState) -> Result<(), MetalW8Error> {
            let mut error = [0 as c_char; ERROR_CAPACITY];
            let status = unsafe {
                apxinf_metal_w8_linear_layer_seed_state(
                    self.0.as_ptr(),
                    state.query_conv().as_ptr(),
                    state.query_conv().len() as u32,
                    state.key_conv().as_ptr(),
                    state.key_conv().len() as u32,
                    state.value_conv().as_ptr(),
                    state.value_conv().len() as u32,
                    state.recurrent().as_ptr(),
                    state.recurrent().len() as u32,
                    error.as_mut_ptr(),
                    error.len(),
                )
            };
            if status != 0 {
                return Err(bridge_error("seed Metal W8 linear layer state", &error));
            }
            Ok(())
        }

        pub(super) fn decode(
            &mut self,
            input: &[f32],
            output: &mut [f32],
            inject_failure: bool,
        ) -> LinearLayerExecution {
            let mut receipt = LinearLayerExecutionReceipt::default();
            let mut error = [0 as c_char; ERROR_CAPACITY];
            let status = unsafe {
                apxinf_metal_w8_linear_layer_decode(
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
                Err(bridge_error("run Metal W8 linear layer decode", &error))
            };
            LinearLayerExecution { receipt, result }
        }

        pub(super) fn snapshot(&self, dims: GdnDimensions) -> Result<GdnDecodeState, MetalW8Error> {
            let mut query = vec![0.0f32; dims.conv_kernel_size * dims.key_width()];
            let mut key = vec![0.0f32; dims.conv_kernel_size * dims.key_width()];
            let mut value = vec![0.0f32; dims.conv_kernel_size * dims.value_width()];
            let mut recurrent = vec![0.0f32; dims.value_heads * dims.key_dim * dims.value_dim];
            let mut error = [0 as c_char; ERROR_CAPACITY];
            let status = unsafe {
                apxinf_metal_w8_linear_layer_snapshot_state(
                    self.0.as_ptr(),
                    query.as_mut_ptr(),
                    query.len() as u32,
                    key.as_mut_ptr(),
                    key.len() as u32,
                    value.as_mut_ptr(),
                    value.len() as u32,
                    recurrent.as_mut_ptr(),
                    recurrent.len() as u32,
                    error.as_mut_ptr(),
                    error.len(),
                )
            };
            if status != 0 {
                return Err(bridge_error("snapshot Metal W8 linear layer state", &error));
            }
            GdnDecodeState::from_parts(dims, query, key, value, recurrent)
        }
    }

    impl Drop for LinearLayerHandle {
        fn drop(&mut self) {
            unsafe { apxinf_metal_w8_linear_layer_destroy(self.0.as_ptr()) };
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
    use super::{GdnDecodeState, GdnDimensions, MetalW8Error, PackedW8LinearLayerBlock};

    #[derive(Clone, Copy, Debug, Default)]
    pub(super) struct LinearLayerExecutionReceipt {
        pub(super) host_to_device_bytes: u64,
        pub(super) device_to_host_bytes: u64,
        pub(super) command_buffers: u32,
        pub(super) compute_encoders: u32,
        pub(super) commits: u32,
        pub(super) waits: u32,
        pub(super) state_commits: u32,
    }

    pub(super) struct LinearLayerExecution {
        pub(super) receipt: LinearLayerExecutionReceipt,
        pub(super) result: Result<(), MetalW8Error>,
    }

    pub(super) struct LinearLayerHandle;

    impl LinearLayerHandle {
        pub(super) fn new(_weights: &PackedW8LinearLayerBlock) -> Result<Self, MetalW8Error> {
            Err(MetalW8Error::new("Metal W8 linear layer requires macOS"))
        }

        pub(super) fn new_gdn_out_g32(
            _weights: &PackedW8LinearLayerBlock,
        ) -> Result<Self, MetalW8Error> {
            Err(MetalW8Error::new(
                "Metal W8 precision-v2 GDN-out-G32 linear layer requires macOS",
            ))
        }

        pub(super) fn seed(&mut self, _state: &GdnDecodeState) -> Result<(), MetalW8Error> {
            Err(MetalW8Error::new("Metal W8 linear layer requires macOS"))
        }

        pub(super) fn decode(
            &mut self,
            _input: &[f32],
            _output: &mut [f32],
            _inject_failure: bool,
        ) -> LinearLayerExecution {
            LinearLayerExecution {
                receipt: LinearLayerExecutionReceipt::default(),
                result: Err(MetalW8Error::new("Metal W8 linear layer requires macOS")),
            }
        }

        pub(super) fn snapshot(
            &self,
            _dims: GdnDimensions,
        ) -> Result<GdnDecodeState, MetalW8Error> {
            Err(MetalW8Error::new("Metal W8 linear layer requires macOS"))
        }
    }
}
