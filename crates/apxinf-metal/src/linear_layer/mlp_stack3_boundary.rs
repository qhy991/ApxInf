use super::{
    checked_sum, f32_bytes, rms_norm, GdnDecodeState, GdnDimensions, MetalW8Error,
    PackedW8LinearLayerBlock,
};
use crate::gdn_core_fused_profile_v1::{
    validate_production_profile_v1, validate_production_receipt_v1, GdnCoreProductionObservedV1,
};
use crate::{GdnCoreProductionReceiptV1, GdnCoreProfileV1};
use crate::{PackedW8MlpBlock, W8GroupSize};

const STACK_DEPTH: usize = 3;

/// CPU packed-weight result for one full-attention MLP boundary followed by
/// exactly three complete linear-attention layers.
pub struct MlpStack3BoundaryDecodeResultV1 {
    pub output: Vec<f32>,
    pub states: [GdnDecodeState; STACK_DEPTH],
}

/// Exact resident-buffer and per-decode transaction contract for boundary v1.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MlpStack3BoundaryBufferLedgerV1 {
    pub gdn_core_profile: GdnCoreProfileV1,
    pub gdn_function_chain: &'static str,
    pub scope: &'static str,
    pub exclusions: &'static str,
    pub abi_version: u32,
    pub stack_depth: usize,
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
    pub kernel_dispatches_per_decode: usize,
    pub explicit_buffer_barriers_per_decode: usize,
    pub gdn_core_seams_per_decode: usize,
    pub gdn_core_kernel_dispatches_per_decode: usize,
    pub gdn_core_explicit_buffer_barriers_per_decode: usize,
    pub gdn_core_recurrent_or_fused_threads_per_threadgroup: usize,
    pub gdn_core_threadgroups_per_decode: usize,
    pub gdn_core_launched_threads_per_decode: usize,
    pub gdn_core_source_declared_threadgroup_memory_bytes: usize,
    pub gdn_core_expected_pipeline_static_threadgroup_memory_bytes: usize,
    pub gdn_core_internal_threadgroup_barrier_sites_per_threadgroup: usize,
    pub commits_per_decode: usize,
    pub waits_per_decode: usize,
    pub intermediate_host_finite_checks_per_decode: usize,
    pub final_output_finite_checks_per_decode: usize,
}

/// Versioned packed-weight oracle for the synthetic-only MLP→Stack3 boundary.
/// It is independent of model loading and every default runtime selector.
#[derive(Clone, Debug)]
pub struct PackedW8MlpStack3BoundaryV1 {
    boundary_mlp: PackedW8MlpBlock,
    boundary_post_attention_rms_weight: Vec<f32>,
    boundary_rms_norm_eps: f32,
    stack_layers: [PackedW8LinearLayerBlock; STACK_DEPTH],
    dims: GdnDimensions,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MlpStack3BoundaryMetalStatsV1 {
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
    pub last_gdn_core_receipt: Option<GdnCoreProductionReceiptV1>,
    pub terminal_error: bool,
}

/// Synthetic-only Metal primitive for the full-attention MLP→Stack3 boundary.
/// No model loader or default runtime constructs this versioned type.
pub struct MetalW8MlpStack3BoundaryV1 {
    dims: GdnDimensions,
    inner: platform::BoundaryHandleV1,
    output: Vec<f32>,
    seeded: bool,
    terminal_error: bool,
    stats: MlpStack3BoundaryMetalStatsV1,
    buffer_ledger: MlpStack3BoundaryBufferLedgerV1,
    gdn_core_profile: GdnCoreProfileV1,
    production_receipt_enabled: bool,
    last_gdn_core_receipt: Option<GdnCoreProductionReceiptV1>,
}

impl PackedW8MlpStack3BoundaryV1 {
    pub fn new(
        boundary_mlp: PackedW8MlpBlock,
        boundary_post_attention_rms_weight: &[f32],
        boundary_rms_norm_eps: f32,
        stack_layers: [PackedW8LinearLayerBlock; STACK_DEPTH],
    ) -> Result<Self, MetalW8Error> {
        validate_precision_contract(&boundary_mlp, &stack_layers)?;
        let dims = stack_layers[0].gdn.dimensions();
        let hidden_size = dims.hidden_size;
        if boundary_mlp.down.rows != hidden_size || boundary_mlp.gate_up.columns != hidden_size {
            return Err(MetalW8Error::new(
                "Metal W8 MLP→Stack3 boundary v1 hidden sizes differ",
            ));
        }
        let stack_intermediate_size = stack_layers[0].intermediate_size();
        let stack_rms_norm_eps = stack_layers[0].rms_norm_eps;
        for (slot, layer) in stack_layers.iter().enumerate() {
            if layer.gdn.dimensions() != dims
                || layer.intermediate_size() != stack_intermediate_size
                || layer.rms_norm_eps != stack_rms_norm_eps
            {
                return Err(MetalW8Error::new(format!(
                    "Metal W8 MLP→Stack3 boundary v1 layer {slot} dimensions or RMS epsilons differ from layer 0"
                )));
            }
        }
        if boundary_post_attention_rms_weight.len() != hidden_size {
            return Err(MetalW8Error::new(format!(
                "Metal W8 MLP→Stack3 boundary v1 post-attention RMS weight has {} elements, expected {hidden_size}",
                boundary_post_attention_rms_weight.len()
            )));
        }
        if let Some(index) = boundary_post_attention_rms_weight
            .iter()
            .position(|value| !value.is_finite())
        {
            return Err(MetalW8Error::new(format!(
                "Metal W8 MLP→Stack3 boundary v1 post-attention RMS weight contains a non-finite value at element {index}"
            )));
        }
        if !boundary_rms_norm_eps.is_finite() || boundary_rms_norm_eps < 0.0 {
            return Err(MetalW8Error::new(
                "Metal W8 MLP→Stack3 boundary v1 RMS epsilon must be finite and non-negative",
            ));
        }
        Ok(Self {
            boundary_mlp,
            boundary_post_attention_rms_weight: boundary_post_attention_rms_weight.to_vec(),
            boundary_rms_norm_eps,
            stack_layers,
            dims,
        })
    }

    pub fn hidden_size(&self) -> usize {
        self.dims.hidden_size
    }

    pub fn boundary_intermediate_size(&self) -> usize {
        self.boundary_mlp.down.columns
    }

    pub fn buffer_ledger(&self) -> Result<MlpStack3BoundaryBufferLedgerV1, MetalW8Error> {
        self.buffer_ledger_for_profile(GdnCoreProfileV1::LegacyFourDispatch)
    }

    fn buffer_ledger_for_profile(
        &self,
        profile: GdnCoreProfileV1,
    ) -> Result<MlpStack3BoundaryBufferLedgerV1, MetalW8Error> {
        let layer_ledgers = self
            .stack_layers
            .iter()
            .map(PackedW8LinearLayerBlock::buffer_ledger)
            .collect::<Result<Vec<_>, _>>()?;
        let boundary_mlp_ledger = self.boundary_mlp.buffer_ledger()?;
        let packed_weight_bytes = boundary_mlp_ledger
            .packed_weight_bytes
            .checked_add(checked_sum(
                &layer_ledgers
                    .iter()
                    .map(|ledger| ledger.packed_weight_bytes)
                    .collect::<Vec<_>>(),
                "MLP→Stack3 boundary v1 packed weight byte ledger",
            )?)
            .ok_or_else(|| {
                MetalW8Error::new(
                    "Metal W8 MLP→Stack3 boundary v1 packed weight byte ledger overflow",
                )
            })?;
        let packed_scale_bytes = boundary_mlp_ledger
            .packed_scale_bytes
            .checked_add(checked_sum(
                &layer_ledgers
                    .iter()
                    .map(|ledger| ledger.packed_scale_bytes)
                    .collect::<Vec<_>>(),
                "MLP→Stack3 boundary v1 packed scale byte ledger",
            )?)
            .ok_or_else(|| {
                MetalW8Error::new(
                    "Metal W8 MLP→Stack3 boundary v1 packed scale byte ledger overflow",
                )
            })?;
        let f32_parameter_bytes = f32_bytes(
            self.hidden_size(),
            "MLP→Stack3 boundary v1 prefix RMS byte ledger",
        )?
        .checked_add(checked_sum(
            &layer_ledgers
                .iter()
                .map(|ledger| ledger.f32_parameter_bytes)
                .collect::<Vec<_>>(),
            "MLP→Stack3 boundary v1 F32 parameter byte ledger",
        )?)
        .ok_or_else(|| {
            MetalW8Error::new("Metal W8 MLP→Stack3 boundary v1 F32 parameter byte ledger overflow")
        })?;
        let active_state_bytes = checked_sum(
            &layer_ledgers
                .iter()
                .map(|ledger| ledger.active_state_bytes)
                .collect::<Vec<_>>(),
            "MLP→Stack3 boundary v1 active state byte ledger",
        )?;
        let scratch_state_bytes = checked_sum(
            &layer_ledgers
                .iter()
                .map(|ledger| ledger.scratch_state_bytes)
                .collect::<Vec<_>>(),
            "MLP→Stack3 boundary v1 scratch state byte ledger",
        )?;
        let dims = self.dims;
        let maximum_intermediate_size = self
            .boundary_intermediate_size()
            .max(self.stack_layers[0].intermediate_size());
        let activation_elements = checked_sum(
            &[
                dims.hidden_size.checked_mul(4).ok_or_else(|| {
                    MetalW8Error::new(
                        "Metal W8 MLP→Stack3 boundary v1 hidden activation ledger overflow",
                    )
                })?,
                dims.input_projection_rows(),
                dims.qkv_width(),
                dims.value_width().checked_mul(2).ok_or_else(|| {
                    MetalW8Error::new(
                        "Metal W8 MLP→Stack3 boundary v1 GDN activation ledger overflow",
                    )
                })?,
                maximum_intermediate_size.checked_mul(3).ok_or_else(|| {
                    MetalW8Error::new(
                        "Metal W8 MLP→Stack3 boundary v1 MLP activation ledger overflow",
                    )
                })?,
            ],
            "MLP→Stack3 boundary v1 activation element ledger",
        )?;
        let activation_bytes = f32_bytes(
            activation_elements,
            "MLP→Stack3 boundary v1 activation byte ledger",
        )?;
        let total_persistent_bytes = checked_sum(
            &[
                packed_weight_bytes,
                packed_scale_bytes,
                f32_parameter_bytes,
                active_state_bytes,
                scratch_state_bytes,
                activation_bytes,
            ],
            "MLP→Stack3 boundary v1 total persistent byte ledger",
        )?;
        let hidden_bytes = f32_bytes(
            dims.hidden_size,
            "MLP→Stack3 boundary v1 hidden transfer byte ledger",
        )?;
        let gdn_core_seams = STACK_DEPTH as u32;
        let gdn_core_dispatches = profile.gdn_core_dispatches_for_seams(gdn_core_seams);
        let (gdn_core_threadgroups, gdn_core_launched_threads) =
            if profile == GdnCoreProfileV1::Fused128 {
                (
                    profile.gdn_core_threadgroups_for_seams(gdn_core_seams) as usize,
                    profile.gdn_core_launched_threads_for_seams(gdn_core_seams) as usize,
                )
            } else {
                let depthwise_threadgroups = self.dims.qkv_width().div_ceil(256);
                let per_seam_threadgroups = depthwise_threadgroups + 1 + self.dims.value_heads + 1;
                let per_seam_launched_threads = self.dims.qkv_width()
                    + 2 * self.dims.key_heads
                    + self.dims.value_heads * 256
                    + self.dims.value_heads;
                (
                    STACK_DEPTH * per_seam_threadgroups,
                    STACK_DEPTH * per_seam_launched_threads,
                )
            };
        Ok(MlpStack3BoundaryBufferLedgerV1 {
            gdn_core_profile: profile,
            gdn_function_chain: profile.expected_function_chain(),
            scope: "resident-mtlbuffer-only",
            exclusions: "CPU packed weights, host allocations, Metal pipelines/libraries/queues, command objects, driver allocations, attention/KV, model loader, and language-model head",
            abi_version: 1,
            stack_depth: STACK_DEPTH,
            allocated_buffers: 81,
            shared_buffers: 73,
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
            compute_encoders_per_decode: 4,
            kernel_dispatches_per_decode: if profile == GdnCoreProfileV1::Fused128 {
                35
            } else {
                44
            },
            explicit_buffer_barriers_per_decode: if profile
                == GdnCoreProfileV1::Fused128
            {
                31
            } else {
                40
            },
            gdn_core_seams_per_decode: gdn_core_seams as usize,
            gdn_core_kernel_dispatches_per_decode: gdn_core_dispatches as usize,
            gdn_core_explicit_buffer_barriers_per_decode: gdn_core_dispatches as usize,
            gdn_core_recurrent_or_fused_threads_per_threadgroup: profile
                .recurrent_threads_per_threadgroup()
                as usize,
            gdn_core_threadgroups_per_decode: gdn_core_threadgroups,
            gdn_core_launched_threads_per_decode: gdn_core_launched_threads,
            gdn_core_source_declared_threadgroup_memory_bytes: profile
                .source_declared_threadgroup_memory_bytes()
                as usize,
            gdn_core_expected_pipeline_static_threadgroup_memory_bytes: profile
                .expected_pipeline_static_threadgroup_memory_bytes()
                as usize,
            gdn_core_internal_threadgroup_barrier_sites_per_threadgroup: profile
                .internal_threadgroup_barrier_sites_per_threadgroup()
                as usize,
            commits_per_decode: 1,
            waits_per_decode: 1,
            intermediate_host_finite_checks_per_decode: 0,
            final_output_finite_checks_per_decode: 1,
        })
    }

    pub fn decode_reference(
        &self,
        hidden: &[f32],
        states: &[GdnDecodeState; STACK_DEPTH],
    ) -> Result<MlpStack3BoundaryDecodeResultV1, MetalW8Error> {
        if hidden.len() != self.hidden_size() {
            return Err(MetalW8Error::new(format!(
                "Metal W8 MLP→Stack3 boundary v1 input has {} elements, expected {}",
                hidden.len(),
                self.hidden_size()
            )));
        }
        if let Some(index) = hidden.iter().position(|value| !value.is_finite()) {
            return Err(MetalW8Error::new(format!(
                "Metal W8 MLP→Stack3 boundary v1 input contains a non-finite value at element {index}"
            )));
        }
        let normalized = rms_norm(
            hidden,
            &self.boundary_post_attention_rms_weight,
            self.boundary_rms_norm_eps,
        );
        let boundary_update = self.boundary_mlp.forward(&normalized)?;
        let mut output = hidden
            .iter()
            .zip(boundary_update)
            .map(|(&residual, update)| residual + update)
            .collect::<Vec<_>>();
        let mut next_states = states.clone();
        for (slot, layer) in self.stack_layers.iter().enumerate() {
            let result = layer.decode_reference(&output, &next_states[slot])?;
            output = result.output;
            next_states[slot] = result.state;
        }
        Ok(MlpStack3BoundaryDecodeResultV1 {
            output,
            states: next_states,
        })
    }
}

impl MetalW8MlpStack3BoundaryV1 {
    pub fn from_packed(weights: &PackedW8MlpStack3BoundaryV1) -> Result<Self, MetalW8Error> {
        Self::from_packed_with_profile_impl(weights, GdnCoreProfileV1::LegacyFourDispatch, false)
    }

    /// Explicit fixed-shape production continuation. The legacy arm is used
    /// for same-binary A/C custody; the Q/K-staged diagnostic arm is rejected.
    pub fn from_packed_with_gdn_core_profile_v1(
        weights: &PackedW8MlpStack3BoundaryV1,
        profile: GdnCoreProfileV1,
    ) -> Result<Self, MetalW8Error> {
        Self::from_packed_with_profile_impl(weights, profile, true)
    }

    fn from_packed_with_profile_impl(
        weights: &PackedW8MlpStack3BoundaryV1,
        profile: GdnCoreProfileV1,
        production_receipt_enabled: bool,
    ) -> Result<Self, MetalW8Error> {
        validate_u32_contract(weights)?;
        if production_receipt_enabled {
            validate_production_profile_v1(profile, weights.dims)?;
        } else if profile != GdnCoreProfileV1::LegacyFourDispatch {
            return Err(MetalW8Error::new(
                "ordinary Metal W8 MLP-to-Stack3 constructors must use the legacy GDN core profile",
            ));
        }
        let buffer_ledger = weights.buffer_ledger_for_profile(profile)?;
        Ok(Self {
            dims: weights.dims,
            inner: platform::BoundaryHandleV1::new(weights, profile, production_receipt_enabled)?,
            output: vec![0.0; weights.hidden_size()],
            seeded: false,
            terminal_error: false,
            stats: MlpStack3BoundaryMetalStatsV1::default(),
            buffer_ledger,
            gdn_core_profile: profile,
            production_receipt_enabled,
            last_gdn_core_receipt: None,
        })
    }

    pub fn seed_decode_states(
        &mut self,
        states: &[GdnDecodeState; STACK_DEPTH],
    ) -> Result<(), MetalW8Error> {
        if self.terminal_error {
            return Err(MetalW8Error::new(
                "Metal W8 MLP→Stack3 boundary v1 is terminal after a decode failure; clear before reseeding",
            ));
        }
        validate_states(states, self.dims)?;
        self.inner.seed(states)?;
        self.seeded = true;
        self.stats = MlpStack3BoundaryMetalStatsV1::default();
        self.last_gdn_core_receipt = None;
        Ok(())
    }

    pub fn clear_decode_states(&mut self) -> Result<(), MetalW8Error> {
        let cleared = std::array::from_fn(|_| GdnDecodeState::zeroed(self.dims).unwrap());
        self.inner.seed(&cleared)?;
        self.output.fill(0.0);
        self.seeded = false;
        self.terminal_error = false;
        self.stats = MlpStack3BoundaryMetalStatsV1::default();
        self.last_gdn_core_receipt = None;
        Ok(())
    }

    pub fn decode(&mut self, hidden: &[f32]) -> Result<&[f32], MetalW8Error> {
        self.validate_decode_input(hidden)?;
        self.last_gdn_core_receipt = None;
        self.stats.last_gdn_core_receipt = None;
        let execution = self.inner.decode(hidden, &mut self.output, false);
        self.record_execution(&execution);
        if let Err(error) = execution.result {
            self.terminal_error = true;
            self.stats.terminal_error = true;
            return Err(error);
        }
        if self.production_receipt_enabled {
            let receipt = match platform::validate_execution_receipt(
                &execution.receipt,
                self.gdn_core_profile,
            ) {
                Ok(receipt) => receipt,
                Err(error) => {
                    self.terminal_error = true;
                    self.stats.terminal_error = true;
                    self.stats.successful_decodes -= 1;
                    self.stats.failed_decodes += 1;
                    return Err(error);
                }
            };
            self.last_gdn_core_receipt = Some(receipt);
            self.stats.last_gdn_core_receipt = Some(receipt);
        }
        Ok(&self.output)
    }

    pub fn state_snapshots(&self) -> Result<[GdnDecodeState; STACK_DEPTH], MetalW8Error> {
        if !self.seeded {
            return Err(MetalW8Error::new(
                "Metal W8 MLP→Stack3 boundary v1 states must be seeded before snapshot",
            ));
        }
        let snapshots = (0..STACK_DEPTH)
            .map(|slot| self.inner.snapshot(slot, self.dims))
            .collect::<Result<Vec<_>, _>>()?;
        snapshots.try_into().map_err(|_| {
            MetalW8Error::new("Metal W8 MLP→Stack3 boundary v1 snapshot depth changed")
        })
    }

    pub fn stats(&self) -> MlpStack3BoundaryMetalStatsV1 {
        self.stats
    }

    pub fn buffer_ledger(&self) -> MlpStack3BoundaryBufferLedgerV1 {
        self.buffer_ledger
    }

    pub fn last_gdn_core_receipt(&self) -> Option<GdnCoreProductionReceiptV1> {
        self.last_gdn_core_receipt
    }

    fn validate_decode_input(&self, hidden: &[f32]) -> Result<(), MetalW8Error> {
        if self.terminal_error {
            return Err(MetalW8Error::new(
                "Metal W8 MLP→Stack3 boundary v1 is terminal after a decode failure; clear before retry",
            ));
        }
        if !self.seeded {
            return Err(MetalW8Error::new(
                "Metal W8 MLP→Stack3 boundary v1 states must be seeded after CPU prefill",
            ));
        }
        if hidden.len() != self.dims.hidden_size {
            return Err(MetalW8Error::new(format!(
                "Metal W8 MLP→Stack3 boundary v1 input has {} elements, expected {}",
                hidden.len(),
                self.dims.hidden_size
            )));
        }
        if let Some(index) = hidden.iter().position(|value| !value.is_finite()) {
            return Err(MetalW8Error::new(format!(
                "Metal W8 MLP→Stack3 boundary v1 input contains a non-finite value at element {index}"
            )));
        }
        Ok(())
    }

    fn record_execution(&mut self, execution: &platform::BoundaryExecutionV1) {
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

    #[cfg(any(test, debug_assertions))]
    #[doc(hidden)]
    pub fn inject_failure_after_scratch_execution_for_testing(
        &mut self,
        hidden: &[f32],
    ) -> Result<(), MetalW8Error> {
        self.validate_decode_input(hidden)?;
        self.last_gdn_core_receipt = None;
        self.stats.last_gdn_core_receipt = None;
        let execution = self.inner.decode(hidden, &mut self.output, true);
        self.record_execution(&execution);
        if execution.result.is_err() {
            self.terminal_error = true;
            self.stats.terminal_error = true;
        }
        execution.result
    }
}

fn validate_precision_contract(
    boundary_mlp: &PackedW8MlpBlock,
    stack_layers: &[PackedW8LinearLayerBlock; STACK_DEPTH],
) -> Result<(), MetalW8Error> {
    for (label, actual, expected) in [
        (
            "boundary MLP gate/up projection",
            boundary_mlp.gate_up.group_size(),
            W8GroupSize::G64,
        ),
        (
            "boundary MLP down projection",
            boundary_mlp.down.group_size(),
            W8GroupSize::G64,
        ),
    ] {
        if actual != expected {
            return Err(MetalW8Error::new(format!(
                "Metal W8 MLP→Stack3 boundary v1 {label} requires group size {}, got {}",
                expected.columns(),
                actual.columns()
            )));
        }
    }
    for (slot, layer) in stack_layers.iter().enumerate() {
        for (label, actual, expected) in [
            (
                "GDN input projection",
                layer.gdn.input_projection.group_size(),
                W8GroupSize::G64,
            ),
            (
                "GDN output projection",
                layer.gdn.output_projection.group_size(),
                W8GroupSize::G32,
            ),
            (
                "MLP gate/up projection",
                layer.mlp.gate_up.group_size(),
                W8GroupSize::G64,
            ),
            (
                "MLP down projection",
                layer.mlp.down.group_size(),
                W8GroupSize::G64,
            ),
        ] {
            if actual != expected {
                return Err(MetalW8Error::new(format!(
                    "Metal W8 MLP→Stack3 boundary v1 layer {slot} {label} requires group size {}, got {}",
                    expected.columns(),
                    actual.columns()
                )));
            }
        }
    }
    Ok(())
}

fn validate_u32_contract(weights: &PackedW8MlpStack3BoundaryV1) -> Result<(), MetalW8Error> {
    validate_raw_abi_contract(
        weights.dims,
        weights.boundary_intermediate_size(),
        weights.stack_layers[0].intermediate_size(),
    )
}

fn validate_raw_abi_contract(
    dims: GdnDimensions,
    boundary_intermediate_size: usize,
    stack_intermediate_size: usize,
) -> Result<(), MetalW8Error> {
    for (label, value) in [
        ("hidden_size", dims.hidden_size),
        ("key_heads", dims.key_heads),
        ("value_heads", dims.value_heads),
        ("key_dim", dims.key_dim),
        ("value_dim", dims.value_dim),
        ("conv_kernel_size", dims.conv_kernel_size),
        ("boundary_intermediate_size", boundary_intermediate_size),
        ("stack_intermediate_size", stack_intermediate_size),
    ] {
        if value > u32::MAX as usize {
            return Err(MetalW8Error::new(format!(
                "Metal W8 MLP→Stack3 boundary v1 {label} exceeds the u32 ABI"
            )));
        }
    }
    for (label, intermediate_size) in [
        ("boundary", boundary_intermediate_size),
        ("stack", stack_intermediate_size),
    ] {
        if intermediate_size > u32::MAX as usize / 2 {
            return Err(MetalW8Error::new(format!(
                "Metal W8 MLP→Stack3 boundary v1 {label} gate/up row count exceeds the u32 ABI"
            )));
        }
    }
    let key_width = dims
        .key_heads
        .checked_mul(dims.key_dim)
        .ok_or_else(|| MetalW8Error::new("Metal W8 MLP→Stack3 boundary v1 key width overflow"))?;
    let value_width = dims
        .value_heads
        .checked_mul(dims.value_dim)
        .ok_or_else(|| MetalW8Error::new("Metal W8 MLP→Stack3 boundary v1 value width overflow"))?;
    let qkv_width = key_width
        .checked_mul(2)
        .and_then(|count| count.checked_add(value_width))
        .ok_or_else(|| MetalW8Error::new("Metal W8 MLP→Stack3 boundary v1 QKV width overflow"))?;
    let input_rows = qkv_width
        .checked_add(value_width)
        .and_then(|count| count.checked_add(dims.value_heads.checked_mul(2)?))
        .ok_or_else(|| {
            MetalW8Error::new("Metal W8 MLP→Stack3 boundary v1 input row count overflow")
        })?;
    for (label, value) in [
        ("key width", key_width),
        ("value width", value_width),
        ("QKV width", qkv_width),
        ("input row count", input_rows),
    ] {
        if value > u32::MAX as usize {
            return Err(MetalW8Error::new(format!(
                "Metal W8 MLP→Stack3 boundary v1 {label} exceeds the u32 ABI"
            )));
        }
    }
    validate_state_abi(dims)
}

fn validate_states(
    states: &[GdnDecodeState; STACK_DEPTH],
    dims: GdnDimensions,
) -> Result<(), MetalW8Error> {
    let expected_query = dims
        .conv_kernel_size
        .checked_mul(dims.key_width())
        .ok_or_else(|| MetalW8Error::new("Metal W8 MLP→Stack3 boundary v1 state overflow"))?;
    let expected_value = dims
        .conv_kernel_size
        .checked_mul(dims.value_width())
        .ok_or_else(|| MetalW8Error::new("Metal W8 MLP→Stack3 boundary v1 state overflow"))?;
    let expected_recurrent = dims
        .value_heads
        .checked_mul(dims.key_dim)
        .and_then(|count| count.checked_mul(dims.value_dim))
        .ok_or_else(|| MetalW8Error::new("Metal W8 MLP→Stack3 boundary v1 state overflow"))?;
    for (slot, state) in states.iter().enumerate() {
        for (label, actual, expected) in [
            ("query", state.query_conv().len(), expected_query),
            ("key", state.key_conv().len(), expected_query),
            ("value", state.value_conv().len(), expected_value),
            ("recurrent", state.recurrent().len(), expected_recurrent),
        ] {
            if actual != expected {
                return Err(MetalW8Error::new(format!(
                    "Metal W8 MLP→Stack3 boundary v1 state slot {slot} {label} has {actual} elements, expected {expected}"
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
                    "Metal W8 MLP→Stack3 boundary v1 state slot {slot} {label} contains a non-finite value at element {element}"
                )));
            }
        }
    }
    Ok(())
}

fn validate_state_abi(dims: GdnDimensions) -> Result<(), MetalW8Error> {
    fn check(label: &str, factors: &[usize]) -> Result<(), MetalW8Error> {
        let count = factors.iter().try_fold(1usize, |product, &factor| {
            product.checked_mul(factor).ok_or_else(|| {
                MetalW8Error::new(format!(
                    "Metal W8 MLP→Stack3 boundary v1 {label} element count overflow"
                ))
            })
        })?;
        if count > u32::MAX as usize {
            return Err(MetalW8Error::new(format!(
                "Metal W8 MLP→Stack3 boundary v1 {label} element count {count} exceeds the u32 ABI"
            )));
        }
        count
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| {
                MetalW8Error::new(format!(
                    "Metal W8 MLP→Stack3 boundary v1 {label} byte count overflow"
                ))
            })?;
        Ok(())
    }
    check(
        "query state",
        &[dims.key_heads, dims.key_dim, dims.conv_kernel_size],
    )?;
    check(
        "value state",
        &[dims.value_heads, dims.value_dim, dims.conv_kernel_size],
    )?;
    check(
        "recurrent state",
        &[dims.value_heads, dims.key_dim, dims.value_dim],
    )
}

#[cfg(test)]
mod tests {
    use super::{validate_raw_abi_contract, GdnDimensions};

    #[test]
    fn boundary_v1_bridge_symbols_and_resident_buffer_shape_match_the_public_ledger() {
        let bridge = include_str!("../metal_w8_mlp_stack3_boundary_v1_bridge.mm");
        for symbol in [
            "apxinf_metal_w8_mlp_stack3_boundary_create_gdn_out_g32_v1(",
            "apxinf_metal_w8_mlp_stack3_boundary_seed_states_v1(",
            "apxinf_metal_w8_mlp_stack3_boundary_decode_v1(",
            "apxinf_metal_w8_mlp_stack3_boundary_snapshot_state_v1(",
            "apxinf_metal_w8_mlp_stack3_boundary_destroy_v1(",
        ] {
            assert!(bridge.contains(symbol));
        }
        let layer_buffers = bridge
            .split("struct BoundaryStackLayer {")
            .nth(1)
            .unwrap()
            .split("GdnParams gdn_params;")
            .next()
            .unwrap();
        assert_eq!(layer_buffers.matches("id<MTLBuffer>").count(), 22);
        let handle_buffers = bridge
            .split("struct ApxinfMetalW8MlpStack3BoundaryHandleV1 {")
            .nth(1)
            .unwrap()
            .split("uint32_t seeded_mask;")
            .next()
            .unwrap();
        assert_eq!(handle_buffers.matches("id<MTLBuffer>").count(), 15);
        assert_eq!(22 * 3 + 15, 81);
        assert_eq!(bridge.matches("[handle->queue commandBuffer]").count(), 1);
        assert_eq!(bridge.matches("[command commit]").count(), 1);
        assert_eq!(bridge.matches("[command waitUntilCompleted]").count(), 1);
        assert!(bridge.contains("receipt->compute_encoders = 1;"));
        assert!(bridge.contains("receipt->compute_encoders += 1;"));
        assert!(bridge.contains("Only final hidden_a is checked"));
        let prefix_encoder = bridge
            .split("void encode_boundary_mlp(")
            .nth(1)
            .unwrap()
            .split("void encode_layer(")
            .next()
            .unwrap();
        assert_eq!(
            prefix_encoder.matches("dispatchThread").count(),
            5,
            "boundary prefix dispatch count"
        );
        let stack_layer_encoder = bridge
            .split("void encode_layer(")
            .nth(1)
            .unwrap()
            .split("}  // namespace")
            .next()
            .unwrap();
        assert_eq!(
            stack_layer_encoder.matches("dispatchThread").count(),
            14,
            "source contains thirteen legacy dispatches plus one mutually exclusive fused dispatch"
        );
        assert!(stack_layer_encoder.contains("if (handle->gdn_core_profile == kFusedProfile)"));
        assert!(stack_layer_encoder.contains("handle->gdn_core_fused_pipeline"));
        assert!(stack_layer_encoder.contains("GdnParams fused_core_params = layer.gdn_params;"));
        assert!(stack_layer_encoder.contains("fused_core_params.output_groups_per_row = 32;"));
        assert!(bridge.contains("params.output_groups_per_row == 64"));
        assert!(bridge.contains("expected_transaction_dispatches"));
        assert!(bridge.contains("return profile == kFusedProfile ? 35 : 44;"));
        assert!(bridge.contains("return profile == kFusedProfile ? 31 : 40;"));
        assert!(bridge.contains("gdn_core_profile == kFusedProfile && require_fixed_shape != 1"));
    }

    #[test]
    fn boundary_v1_rejects_gate_up_row_and_state_counts_outside_the_u32_abi() {
        let dims = GdnDimensions {
            hidden_size: 64,
            key_heads: 2,
            value_heads: 2,
            key_dim: 32,
            value_dim: 32,
            conv_kernel_size: 4,
            rms_norm_eps: 1.0e-6,
        };
        let boundary_rows =
            validate_raw_abi_contract(dims, u32::MAX as usize / 2 + 1, 64).unwrap_err();
        assert!(boundary_rows
            .to_string()
            .contains("boundary gate/up row count"));
        assert!(boundary_rows.to_string().contains("u32 ABI"));

        let too_many_key_heads = u32::MAX as usize / 32 + 1;
        let state_dims = GdnDimensions {
            key_heads: too_many_key_heads,
            value_heads: too_many_key_heads,
            ..dims
        };
        let state = validate_raw_abi_contract(state_dims, 64, 64).unwrap_err();
        assert!(state.to_string().contains("u32 ABI"));
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn boundary_production_execution_receipt_abi_is_exact() {
        assert_eq!(
            std::mem::size_of::<super::platform::BoundaryExecutionReceiptV1>(),
            368
        );
    }
}

#[cfg(target_os = "macos")]
mod platform {
    use super::{
        validate_production_receipt_v1, GdnCoreProductionObservedV1, GdnCoreProductionReceiptV1,
        GdnCoreProfileV1, GdnDecodeState, GdnDimensions, MetalW8Error, PackedW8LinearLayerBlock,
        PackedW8MlpStack3BoundaryV1, STACK_DEPTH,
    };
    use std::ffi::{c_char, c_int, c_void, CStr};
    use std::ptr::NonNull;

    const ERROR_CAPACITY: usize = 1024;

    #[repr(C)]
    struct BoundaryMlpDescriptorV1 {
        gate_up_weights: *const i8,
        gate_up_scales: *const f32,
        down_weights: *const i8,
        down_scales: *const f32,
        post_attention_rms_weight: *const f32,
        hidden_size: u32,
        intermediate_size: u32,
        rms_norm_eps: f32,
    }

    #[repr(C)]
    struct BoundaryStackLayerDescriptorV1 {
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

    impl BoundaryStackLayerDescriptorV1 {
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
    struct BoundaryStateDescriptorV1 {
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
    struct BoundaryMutableStateDescriptorV1 {
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
    #[derive(Clone, Copy, Debug)]
    pub(super) struct BoundaryExecutionReceiptV1 {
        pub(super) host_to_device_bytes: u64,
        pub(super) device_to_host_bytes: u64,
        pub(super) command_buffers: u32,
        pub(super) compute_encoders: u32,
        pub(super) commits: u32,
        pub(super) waits: u32,
        pub(super) state_commits: u32,
        pub(super) state_commit_mask: u32,
        requested_profile: u32,
        observed_profile: u32,
        kernel_dispatches: u32,
        explicit_buffer_barriers: u32,
        gdn_core_seams: u32,
        gdn_core_kernel_dispatches: u32,
        gdn_core_explicit_buffer_barriers: u32,
        threads_per_threadgroup: u32,
        gdn_core_threadgroups: u32,
        gdn_core_launched_threads: u32,
        pipeline_thread_execution_width: u32,
        source_declared_threadgroup_memory_bytes: u32,
        pipeline_static_threadgroup_memory_bytes: u32,
        internal_threadgroup_barrier_sites_per_threadgroup: u32,
        fixed_shape_validated: u32,
        rms_norm_eps_bits: u32,
        persistent_output_groups_per_row: u32,
        core_kernel_output_groups_per_row: u32,
        observed_function_chain: [c_char; 256],
    }

    impl Default for BoundaryExecutionReceiptV1 {
        fn default() -> Self {
            Self {
                host_to_device_bytes: 0,
                device_to_host_bytes: 0,
                command_buffers: 0,
                compute_encoders: 0,
                commits: 0,
                waits: 0,
                state_commits: 0,
                state_commit_mask: 0,
                requested_profile: u32::MAX,
                observed_profile: u32::MAX,
                kernel_dispatches: 0,
                explicit_buffer_barriers: 0,
                gdn_core_seams: 0,
                gdn_core_kernel_dispatches: 0,
                gdn_core_explicit_buffer_barriers: 0,
                threads_per_threadgroup: 0,
                gdn_core_threadgroups: 0,
                gdn_core_launched_threads: 0,
                pipeline_thread_execution_width: 0,
                source_declared_threadgroup_memory_bytes: 0,
                pipeline_static_threadgroup_memory_bytes: 0,
                internal_threadgroup_barrier_sites_per_threadgroup: 0,
                fixed_shape_validated: 0,
                rms_norm_eps_bits: 0,
                persistent_output_groups_per_row: 0,
                core_kernel_output_groups_per_row: 0,
                observed_function_chain: [0; 256],
            }
        }
    }

    pub(super) struct BoundaryExecutionV1 {
        pub(super) receipt: BoundaryExecutionReceiptV1,
        pub(super) result: Result<(), MetalW8Error>,
    }

    extern "C" {
        fn apxinf_metal_w8_mlp_stack3_boundary_create_gdn_out_g32_v1(
            boundary: *const BoundaryMlpDescriptorV1,
            layers: *const BoundaryStackLayerDescriptorV1,
            layer_count: u32,
            gdn_core_profile: u32,
            require_fixed_shape: u8,
            output: *mut *mut c_void,
            error: *mut c_char,
            error_capacity: usize,
        ) -> c_int;
        fn apxinf_metal_w8_mlp_stack3_boundary_seed_states_v1(
            handle: *mut c_void,
            states: *const BoundaryStateDescriptorV1,
            state_count: u32,
            error: *mut c_char,
            error_capacity: usize,
        ) -> c_int;
        fn apxinf_metal_w8_mlp_stack3_boundary_decode_v1(
            handle: *mut c_void,
            input: *const f32,
            input_count: u32,
            output: *mut f32,
            output_count: u32,
            inject_failure_after_execution: u8,
            receipt: *mut BoundaryExecutionReceiptV1,
            error: *mut c_char,
            error_capacity: usize,
        ) -> c_int;
        fn apxinf_metal_w8_mlp_stack3_boundary_snapshot_state_v1(
            handle: *mut c_void,
            slot: u32,
            state: *mut BoundaryMutableStateDescriptorV1,
            error: *mut c_char,
            error_capacity: usize,
        ) -> c_int;
        fn apxinf_metal_w8_mlp_stack3_boundary_destroy_v1(handle: *mut c_void);
    }

    pub(super) struct BoundaryHandleV1(NonNull<c_void>);

    impl BoundaryHandleV1 {
        pub(super) fn new(
            weights: &PackedW8MlpStack3BoundaryV1,
            profile: GdnCoreProfileV1,
            require_fixed_shape: bool,
        ) -> Result<Self, MetalW8Error> {
            let boundary = BoundaryMlpDescriptorV1 {
                gate_up_weights: weights.boundary_mlp.gate_up.values().as_ptr(),
                gate_up_scales: weights.boundary_mlp.gate_up.scales().as_ptr(),
                down_weights: weights.boundary_mlp.down.values().as_ptr(),
                down_scales: weights.boundary_mlp.down.scales().as_ptr(),
                post_attention_rms_weight: weights.boundary_post_attention_rms_weight.as_ptr(),
                hidden_size: weights.hidden_size() as u32,
                intermediate_size: weights.boundary_intermediate_size() as u32,
                rms_norm_eps: weights.boundary_rms_norm_eps,
            };
            let layers = weights
                .stack_layers
                .each_ref()
                .map(BoundaryStackLayerDescriptorV1::from_packed);
            let mut output = std::ptr::null_mut();
            let mut error = [0 as c_char; ERROR_CAPACITY];
            let status = unsafe {
                apxinf_metal_w8_mlp_stack3_boundary_create_gdn_out_g32_v1(
                    &boundary,
                    layers.as_ptr(),
                    layers.len() as u32,
                    profile.selector(),
                    u8::from(require_fixed_shape),
                    &mut output,
                    error.as_mut_ptr(),
                    error.len(),
                )
            };
            if status != 0 {
                return Err(bridge_error(
                    "create Metal W8 MLP→Stack3 boundary v1",
                    &error,
                ));
            }
            NonNull::new(output).map(Self).ok_or_else(|| {
                MetalW8Error::new("create Metal W8 MLP→Stack3 boundary v1 returned a null handle")
            })
        }

        pub(super) fn seed(
            &mut self,
            states: &[GdnDecodeState; STACK_DEPTH],
        ) -> Result<(), MetalW8Error> {
            let descriptors = states.each_ref().map(|state| BoundaryStateDescriptorV1 {
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
                apxinf_metal_w8_mlp_stack3_boundary_seed_states_v1(
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
                Err(bridge_error(
                    "seed Metal W8 MLP→Stack3 boundary v1 states",
                    &error,
                ))
            }
        }

        pub(super) fn decode(
            &mut self,
            input: &[f32],
            output: &mut [f32],
            inject_failure: bool,
        ) -> BoundaryExecutionV1 {
            let mut receipt = BoundaryExecutionReceiptV1::default();
            let mut error = [0 as c_char; ERROR_CAPACITY];
            let status = unsafe {
                apxinf_metal_w8_mlp_stack3_boundary_decode_v1(
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
                Err(bridge_error(
                    "run Metal W8 MLP→Stack3 boundary v1 decode",
                    &error,
                ))
            };
            BoundaryExecutionV1 { receipt, result }
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
            let mut descriptor = BoundaryMutableStateDescriptorV1 {
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
                apxinf_metal_w8_mlp_stack3_boundary_snapshot_state_v1(
                    self.0.as_ptr(),
                    slot as u32,
                    &mut descriptor,
                    error.as_mut_ptr(),
                    error.len(),
                )
            };
            if status != 0 {
                return Err(bridge_error(
                    "snapshot Metal W8 MLP→Stack3 boundary v1 state",
                    &error,
                ));
            }
            GdnDecodeState::from_parts(dims, query, key, value, recurrent)
        }
    }

    pub(super) fn validate_execution_receipt(
        raw: &BoundaryExecutionReceiptV1,
        profile: GdnCoreProfileV1,
    ) -> Result<GdnCoreProductionReceiptV1, MetalW8Error> {
        let expected_dispatches = if profile == GdnCoreProfileV1::Fused128 {
            35
        } else {
            44
        };
        let expected_barriers = if profile == GdnCoreProfileV1::Fused128 {
            31
        } else {
            40
        };
        if raw.host_to_device_bytes != 4096
            || raw.device_to_host_bytes != 4096
            || raw.command_buffers != 1
            || raw.compute_encoders != 4
            || raw.kernel_dispatches != expected_dispatches
            || raw.explicit_buffer_barriers != expected_barriers
            || raw.commits != 1
            || raw.waits != 1
            || raw.state_commits != STACK_DEPTH as u32
            || raw.state_commit_mask != 0b111
        {
            return Err(MetalW8Error::new(
                "invalid live Metal W8 MLP-to-Stack3 production transaction receipt",
            ));
        }
        let nul = raw
            .observed_function_chain
            .iter()
            .position(|&value| value == 0)
            .ok_or_else(|| {
                MetalW8Error::new(
                    "Metal W8 MLP-to-Stack3 production function chain is not NUL-terminated",
                )
            })?;
        let bytes = raw.observed_function_chain[..nul]
            .iter()
            .map(|&value| value as u8)
            .collect::<Vec<_>>();
        let function_chain = std::str::from_utf8(&bytes).map_err(|_| {
            MetalW8Error::new("Metal W8 MLP-to-Stack3 production function chain is not valid UTF-8")
        })?;
        validate_production_receipt_v1(
            profile,
            GdnCoreProductionObservedV1 {
                requested_profile: raw.requested_profile,
                observed_profile: raw.observed_profile,
                gdn_core_seams: raw.gdn_core_seams,
                kernel_dispatches: raw.gdn_core_kernel_dispatches,
                explicit_buffer_barriers: raw.gdn_core_explicit_buffer_barriers,
                recurrent_or_fused_threads_per_threadgroup: raw.threads_per_threadgroup,
                threadgroups: raw.gdn_core_threadgroups,
                launched_threads: raw.gdn_core_launched_threads,
                pipeline_thread_execution_width: raw.pipeline_thread_execution_width,
                source_declared_threadgroup_memory_bytes: raw
                    .source_declared_threadgroup_memory_bytes,
                pipeline_static_threadgroup_memory_bytes: raw
                    .pipeline_static_threadgroup_memory_bytes,
                internal_threadgroup_barrier_sites_per_threadgroup: raw
                    .internal_threadgroup_barrier_sites_per_threadgroup,
                fixed_shape_validated: raw.fixed_shape_validated,
                rms_norm_eps_bits: raw.rms_norm_eps_bits,
                persistent_output_groups_per_row: raw.persistent_output_groups_per_row,
                core_kernel_output_groups_per_row: raw.core_kernel_output_groups_per_row,
            },
            function_chain,
        )
    }

    impl Drop for BoundaryHandleV1 {
        fn drop(&mut self) {
            unsafe { apxinf_metal_w8_mlp_stack3_boundary_destroy_v1(self.0.as_ptr()) };
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
        GdnCoreProductionReceiptV1, GdnCoreProfileV1, GdnDecodeState, GdnDimensions, MetalW8Error,
        PackedW8MlpStack3BoundaryV1, STACK_DEPTH,
    };

    #[derive(Clone, Copy, Debug, Default)]
    pub(super) struct BoundaryExecutionReceiptV1 {
        pub(super) host_to_device_bytes: u64,
        pub(super) device_to_host_bytes: u64,
        pub(super) command_buffers: u32,
        pub(super) compute_encoders: u32,
        pub(super) commits: u32,
        pub(super) waits: u32,
        pub(super) state_commits: u32,
        pub(super) state_commit_mask: u32,
    }

    pub(super) struct BoundaryExecutionV1 {
        pub(super) receipt: BoundaryExecutionReceiptV1,
        pub(super) result: Result<(), MetalW8Error>,
    }

    pub(super) struct BoundaryHandleV1;

    impl BoundaryHandleV1 {
        pub(super) fn new(
            _weights: &PackedW8MlpStack3BoundaryV1,
            _profile: GdnCoreProfileV1,
            _require_fixed_shape: bool,
        ) -> Result<Self, MetalW8Error> {
            Err(MetalW8Error::new(
                "Metal W8 MLP→Stack3 boundary v1 requires macOS",
            ))
        }

        pub(super) fn seed(
            &mut self,
            _states: &[GdnDecodeState; STACK_DEPTH],
        ) -> Result<(), MetalW8Error> {
            Err(MetalW8Error::new(
                "Metal W8 MLP→Stack3 boundary v1 requires macOS",
            ))
        }

        pub(super) fn decode(
            &mut self,
            _input: &[f32],
            _output: &mut [f32],
            _inject_failure: bool,
        ) -> BoundaryExecutionV1 {
            BoundaryExecutionV1 {
                receipt: BoundaryExecutionReceiptV1::default(),
                result: Err(MetalW8Error::new(
                    "Metal W8 MLP→Stack3 boundary v1 requires macOS",
                )),
            }
        }

        pub(super) fn snapshot(
            &self,
            _slot: usize,
            _dims: GdnDimensions,
        ) -> Result<GdnDecodeState, MetalW8Error> {
            Err(MetalW8Error::new(
                "Metal W8 MLP→Stack3 boundary v1 requires macOS",
            ))
        }
    }

    pub(super) fn validate_execution_receipt(
        _raw: &BoundaryExecutionReceiptV1,
        _profile: GdnCoreProfileV1,
    ) -> Result<GdnCoreProductionReceiptV1, MetalW8Error> {
        Err(MetalW8Error::new(
            "Metal W8 MLP-to-Stack3 boundary v1 requires macOS",
        ))
    }
}
