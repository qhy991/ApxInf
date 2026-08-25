use crate::{MetalW8Error, PackedW8MlpBlock, PackedW8Rows, W8GroupSize, W8_TOP_K};

/// Explicit first-stage vocabulary-row kernel for the diagnostic tail lane.
/// The legacy selector remains the default; alternatives require opt-in.
#[repr(u32)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum TailMlpHeadRowsKernelV1 {
    #[default]
    LegacyR8Sg8 = 0,
    Pair2R16Sg8 = 1,
}

impl TailMlpHeadRowsKernelV1 {
    pub const fn receipt_label(self) -> &'static str {
        match self {
            Self::LegacyR8Sg8 => "w8_rows_topk4",
            Self::Pair2R16Sg8 => "w8_rows_topk4_pair2_r16_sg8",
        }
    }

    /// `(rows/TG, rows/SIMD-group, SIMD-groups/TG, threads/TG, cooperative)`.
    pub const fn execution_shape(self) -> (u32, u32, u32, u32, bool) {
        match self {
            Self::LegacyR8Sg8 => (8, 1, 8, 256, false),
            Self::Pair2R16Sg8 => (16, 2, 8, 256, false),
        }
    }

    const fn abi_value(self) -> u32 {
        self as u32
    }

    fn from_abi_value(value: u32) -> Option<Self> {
        match value {
            0 => Some(Self::LegacyR8Sg8),
            1 => Some(Self::Pair2R16Sg8),
            _ => None,
        }
    }
}

/// Runtime-observed Metal pipeline, dispatch, and scratch allocation identity.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TailMlpHeadRowsKernelReceiptV1 {
    pub kernel: TailMlpHeadRowsKernelV1,
    pub rows_per_threadgroup: u32,
    pub rows_per_simdgroup: u32,
    pub simdgroups_per_threadgroup: u32,
    pub threads_per_threadgroup: u32,
    pub partial_count: u32,
    pub partial_topk_bytes: usize,
    pub pipeline_max_total_threads_per_threadgroup: u32,
    pub pipeline_thread_execution_width: u32,
}

/// Exact resident-buffer and per-decode transaction contract for tail v1.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TailMlpHeadBufferLedgerV1 {
    pub scope: &'static str,
    pub exclusions: &'static str,
    pub abi_version: u32,
    pub allocated_buffers: usize,
    pub shared_buffers: usize,
    pub private_buffers: usize,
    pub packed_weight_bytes: usize,
    pub packed_scale_bytes: usize,
    pub f32_parameter_bytes: usize,
    pub hidden_activation_bytes: usize,
    pub mlp_activation_bytes: usize,
    pub partial_topk_bytes: usize,
    pub output_token_bytes: usize,
    pub total_persistent_bytes: usize,
    pub host_input_bytes_per_decode: usize,
    pub host_output_bytes_per_decode: usize,
    pub state_host_transfer_bytes_per_decode: usize,
    pub command_buffers_per_decode: usize,
    pub compute_encoders_per_decode: usize,
    pub kernel_dispatches_per_decode: usize,
    pub buffer_barriers_per_decode: usize,
    pub commits_per_decode: usize,
    pub waits_per_decode: usize,
}

impl TailMlpHeadBufferLedgerV1 {
    pub fn from_dimensions(
        hidden_size: usize,
        intermediate_size: usize,
        vocab_size: usize,
    ) -> Result<Self, MetalW8Error> {
        Self::from_dimensions_with_rows_kernel(
            hidden_size,
            intermediate_size,
            vocab_size,
            TailMlpHeadRowsKernelV1::default(),
        )
    }

    pub fn from_dimensions_with_rows_kernel(
        hidden_size: usize,
        intermediate_size: usize,
        vocab_size: usize,
        rows_kernel: TailMlpHeadRowsKernelV1,
    ) -> Result<Self, MetalW8Error> {
        if hidden_size == 0
            || intermediate_size == 0
            || vocab_size < W8_TOP_K
            || hidden_size % 64 != 0
            || intermediate_size % 64 != 0
        {
            return Err(MetalW8Error::new(
                "Metal W8 tail MLP+head v1 ledger requires non-zero G64 shapes and at least four vocabulary rows",
            ));
        }
        if hidden_size > u32::MAX as usize
            || intermediate_size > u32::MAX as usize / 2
            || vocab_size > u32::MAX as usize
        {
            return Err(MetalW8Error::new(
                "Metal W8 tail MLP+head v1 ledger dimensions exceed the u32 ABI",
            ));
        }
        let hidden_intermediate = hidden_size
            .checked_mul(intermediate_size)
            .ok_or_else(|| MetalW8Error::new("Metal W8 tail MLP+head v1 ledger overflow"))?;
        let packed_weight_bytes = hidden_intermediate
            .checked_mul(3)
            .and_then(|bytes| vocab_size.checked_mul(hidden_size)?.checked_add(bytes))
            .ok_or_else(|| MetalW8Error::new("Metal W8 tail MLP+head v1 weight ledger overflow"))?;
        let packed_scale_bytes = hidden_intermediate
            .checked_mul(3)
            .and_then(|count| vocab_size.checked_mul(hidden_size)?.checked_add(count))
            .and_then(|count| count.checked_div(64))
            .and_then(|count| count.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| MetalW8Error::new("Metal W8 tail MLP+head v1 scale ledger overflow"))?;
        let f32_parameter_bytes = hidden_size
            .checked_mul(2)
            .and_then(|count| count.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| MetalW8Error::new("Metal W8 tail MLP+head v1 RMS ledger overflow"))?;
        let hidden_activation_bytes = f32_parameter_bytes;
        let mlp_activation_bytes = intermediate_size
            .checked_mul(3)
            .and_then(|count| count.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| {
                MetalW8Error::new("Metal W8 tail MLP+head v1 activation ledger overflow")
            })?;
        let rows_per_threadgroup = rows_kernel.execution_shape().0 as usize;
        let partial_topk_bytes = vocab_size
            .checked_add(rows_per_threadgroup - 1)
            .map(|rows| rows / rows_per_threadgroup)
            .and_then(|count| count.checked_mul(W8_TOP_K))
            .and_then(|count| count.checked_mul(8))
            .ok_or_else(|| MetalW8Error::new("Metal W8 tail MLP+head v1 top-4 ledger overflow"))?;
        let output_token_bytes = W8_TOP_K * std::mem::size_of::<u32>();
        let total_persistent_bytes = [
            packed_weight_bytes,
            packed_scale_bytes,
            f32_parameter_bytes,
            hidden_activation_bytes,
            mlp_activation_bytes,
            partial_topk_bytes,
            output_token_bytes,
        ]
        .into_iter()
        .try_fold(0usize, |total, bytes| total.checked_add(bytes))
        .ok_or_else(|| MetalW8Error::new("Metal W8 tail MLP+head v1 total ledger overflow"))?;
        let host_input_bytes_per_decode = hidden_size
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| {
                MetalW8Error::new("Metal W8 tail MLP+head v1 transfer ledger overflow")
            })?;
        let host_output_bytes_per_decode = host_input_bytes_per_decode
            .checked_add(output_token_bytes)
            .ok_or_else(|| {
                MetalW8Error::new("Metal W8 tail MLP+head v1 transfer ledger overflow")
            })?;
        Ok(Self {
            scope: "resident-mtlbuffer-only",
            exclusions: "CPU packed weights, host F32 tied embedding and exact four-candidate rerank, host allocations, Metal pipelines/libraries/queues, command objects, driver allocations, attention/KV, model loader, and all earlier model layers",
            abi_version: 1,
            allocated_buffers: 13,
            shared_buffers: 10,
            private_buffers: 3,
            packed_weight_bytes,
            packed_scale_bytes,
            f32_parameter_bytes,
            hidden_activation_bytes,
            mlp_activation_bytes,
            partial_topk_bytes,
            output_token_bytes,
            total_persistent_bytes,
            host_input_bytes_per_decode,
            host_output_bytes_per_decode,
            state_host_transfer_bytes_per_decode: 0,
            command_buffers_per_decode: 1,
            compute_encoders_per_decode: 1,
            kernel_dispatches_per_decode: 8,
            buffer_barriers_per_decode: 7,
            commits_per_decode: 1,
            waits_per_decode: 1,
        })
    }
}

/// CPU packed-weight result for the synthetic-only versioned tail primitive.
#[derive(Clone, Debug, PartialEq)]
pub struct TailMlpHeadDecodeResultV1 {
    pub normalized_hidden: Vec<f32>,
    pub candidate_token_ids: [u32; W8_TOP_K],
}

/// Packed CPU oracle for layer 23 post-attention RMS → MLP → residual,
/// followed by final RMS and the existing two-stage W8 vocabulary top-4.
/// It is independent of model loading and every default runtime selector.
#[derive(Clone, Debug)]
pub struct PackedW8TailMlpHeadV1 {
    mlp: PackedW8MlpBlock,
    post_attention_rms_weight: Vec<f32>,
    final_rms_weight: Vec<f32>,
    rms_norm_eps: f32,
    vocab: PackedW8Rows,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TailMlpHeadMetalStatsV1 {
    pub decode_calls: usize,
    pub successful_decodes: usize,
    pub failed_decodes: usize,
    pub host_to_device_bytes: usize,
    pub device_to_host_bytes: usize,
    pub command_buffers: usize,
    pub compute_encoders: usize,
    pub kernel_dispatches: usize,
    pub buffer_barriers: usize,
    pub commits: usize,
    pub waits: usize,
    pub output_commits: usize,
    pub last_output_commit_mask: u32,
    pub terminal_error: bool,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TailMlpHeadDecodeViewV1<'a> {
    pub normalized_hidden: &'a [f32],
    pub candidate_token_ids: [u32; W8_TOP_K],
}

/// Synthetic-only Metal primitive for the versioned layer-23 MLP + head
/// transaction. No model loader or default runtime constructs this type.
pub struct MetalW8TailMlpHeadV1 {
    inner: platform::TailHandleV1,
    rows_kernel_receipt: TailMlpHeadRowsKernelReceiptV1,
    normalized_hidden: Vec<f32>,
    candidate_token_ids: [u32; W8_TOP_K],
    terminal_error: bool,
    stats: TailMlpHeadMetalStatsV1,
    buffer_ledger: TailMlpHeadBufferLedgerV1,
}

impl PackedW8TailMlpHeadV1 {
    pub fn new(
        mlp: PackedW8MlpBlock,
        post_attention_rms_weight: &[f32],
        final_rms_weight: &[f32],
        rms_norm_eps: f32,
        vocab: PackedW8Rows,
    ) -> Result<Self, MetalW8Error> {
        if !rms_norm_eps.is_finite() || rms_norm_eps < 0.0 {
            return Err(MetalW8Error::new(
                "Metal W8 tail MLP+head v1 RMS epsilon must be finite and non-negative",
            ));
        }
        let hidden_size = mlp.down.rows;
        let intermediate_size = mlp.down.columns;
        if mlp.gate_up.columns != hidden_size
            || mlp.gate_up.rows
                != intermediate_size.checked_mul(2).ok_or_else(|| {
                    MetalW8Error::new("Metal W8 tail MLP+head v1 gate/up row count overflow")
                })?
            || post_attention_rms_weight.len() != hidden_size
            || final_rms_weight.len() != hidden_size
            || vocab.columns != hidden_size
        {
            return Err(MetalW8Error::new(
                "Metal W8 tail MLP+head v1 hidden shapes differ",
            ));
        }
        if vocab.rows < W8_TOP_K {
            return Err(MetalW8Error::new(
                "Metal W8 tail MLP+head v1 requires at least four vocabulary rows",
            ));
        }
        for (label, values) in [
            ("post-attention RMS weight", post_attention_rms_weight),
            ("final RMS weight", final_rms_weight),
        ] {
            if let Some(index) = values.iter().position(|value| !value.is_finite()) {
                return Err(MetalW8Error::new(format!(
                    "Metal W8 tail MLP+head v1 {label} contains a non-finite value at element {index}"
                )));
            }
        }
        for (label, group_size) in [
            ("MLP gate/up projection", mlp.gate_up.group_size()),
            ("MLP down projection", mlp.down.group_size()),
            ("vocabulary projection", vocab.group_size()),
        ] {
            if group_size != W8GroupSize::G64 {
                return Err(MetalW8Error::new(format!(
                    "Metal W8 tail MLP+head v1 {label} requires group size 64, got {}",
                    group_size.columns()
                )));
            }
        }
        TailMlpHeadBufferLedgerV1::from_dimensions(hidden_size, intermediate_size, vocab.rows)?;
        Ok(Self {
            mlp,
            post_attention_rms_weight: post_attention_rms_weight.to_vec(),
            final_rms_weight: final_rms_weight.to_vec(),
            rms_norm_eps,
            vocab,
        })
    }

    pub fn hidden_size(&self) -> usize {
        self.mlp.down.rows
    }

    pub fn intermediate_size(&self) -> usize {
        self.mlp.down.columns
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab.rows
    }

    pub fn buffer_ledger(&self) -> Result<TailMlpHeadBufferLedgerV1, MetalW8Error> {
        TailMlpHeadBufferLedgerV1::from_dimensions(
            self.hidden_size(),
            self.intermediate_size(),
            self.vocab_size(),
        )
    }

    pub fn decode_reference(
        &self,
        full_attention_residual: &[f32],
    ) -> Result<TailMlpHeadDecodeResultV1, MetalW8Error> {
        if full_attention_residual.len() != self.hidden_size() {
            return Err(MetalW8Error::new(format!(
                "Metal W8 tail MLP+head v1 input has {} elements, expected {}",
                full_attention_residual.len(),
                self.hidden_size()
            )));
        }
        require_finite(full_attention_residual, "Metal W8 tail MLP+head v1 input")?;
        let normalized = rms_norm(
            full_attention_residual,
            &self.post_attention_rms_weight,
            self.rms_norm_eps,
        );
        require_finite(
            &normalized,
            "Metal W8 tail MLP+head v1 post-attention normalized row",
        )?;
        let update = self.mlp.forward(&normalized)?;
        require_finite(&update, "Metal W8 tail MLP+head v1 MLP update")?;
        let residual = full_attention_residual
            .iter()
            .zip(update)
            .map(|(&residual, update)| residual + update)
            .collect::<Vec<_>>();
        require_finite(&residual, "Metal W8 tail MLP+head v1 residual row")?;
        let normalized_hidden = rms_norm(&residual, &self.final_rms_weight, self.rms_norm_eps);
        require_finite(
            &normalized_hidden,
            "Metal W8 tail MLP+head v1 final normalized row",
        )?;
        let candidate_token_ids = self.vocab.topk4(&normalized_hidden)?;
        Ok(TailMlpHeadDecodeResultV1 {
            normalized_hidden,
            candidate_token_ids,
        })
    }
}

fn require_finite(values: &[f32], label: &str) -> Result<(), MetalW8Error> {
    if let Some(index) = values.iter().position(|value| !value.is_finite()) {
        return Err(MetalW8Error::new(format!(
            "{label} contains a non-finite value at element {index}"
        )));
    }
    Ok(())
}

impl MetalW8TailMlpHeadV1 {
    pub fn from_packed(weights: &PackedW8TailMlpHeadV1) -> Result<Self, MetalW8Error> {
        Self::from_packed_with_rows_kernel(weights, TailMlpHeadRowsKernelV1::default())
    }

    pub fn from_packed_with_rows_kernel(
        weights: &PackedW8TailMlpHeadV1,
        rows_kernel: TailMlpHeadRowsKernelV1,
    ) -> Result<Self, MetalW8Error> {
        let buffer_ledger = TailMlpHeadBufferLedgerV1::from_dimensions_with_rows_kernel(
            weights.hidden_size(),
            weights.intermediate_size(),
            weights.vocab_size(),
            rows_kernel,
        )?;
        let inner = platform::TailHandleV1::new(weights, rows_kernel, buffer_ledger)?;
        let rows_kernel_receipt = inner.rows_kernel_receipt();
        Ok(Self {
            inner,
            rows_kernel_receipt,
            normalized_hidden: vec![0.0; weights.hidden_size()],
            candidate_token_ids: [u32::MAX; W8_TOP_K],
            terminal_error: false,
            stats: TailMlpHeadMetalStatsV1::default(),
            buffer_ledger,
        })
    }

    pub fn decode(
        &mut self,
        full_attention_residual: &[f32],
    ) -> Result<TailMlpHeadDecodeViewV1<'_>, MetalW8Error> {
        self.validate_decode_input(full_attention_residual)?;
        let execution = self.inner.decode(
            full_attention_residual,
            &mut self.normalized_hidden,
            &mut self.candidate_token_ids,
            0,
        );
        self.record_execution(&execution);
        if let Err(error) = execution.result {
            self.terminal_error = true;
            self.stats.terminal_error = true;
            return Err(error);
        }
        self.validate_staged_output()?;
        Ok(TailMlpHeadDecodeViewV1 {
            normalized_hidden: &self.normalized_hidden,
            candidate_token_ids: self.candidate_token_ids,
        })
    }

    pub fn reset(&mut self) -> Result<(), MetalW8Error> {
        self.inner.reset()?;
        self.normalized_hidden.fill(0.0);
        self.candidate_token_ids.fill(u32::MAX);
        self.terminal_error = false;
        self.stats = TailMlpHeadMetalStatsV1::default();
        Ok(())
    }

    pub fn stats(&self) -> TailMlpHeadMetalStatsV1 {
        self.stats
    }

    pub fn buffer_ledger(&self) -> TailMlpHeadBufferLedgerV1 {
        self.buffer_ledger
    }

    pub fn rows_kernel(&self) -> TailMlpHeadRowsKernelV1 {
        self.rows_kernel_receipt.kernel
    }

    pub fn rows_kernel_receipt(&self) -> TailMlpHeadRowsKernelReceiptV1 {
        self.rows_kernel_receipt
    }

    #[cfg(any(test, debug_assertions))]
    #[doc(hidden)]
    pub fn inject_failure_after_gpu_execution_for_testing(
        &mut self,
        full_attention_residual: &[f32],
    ) -> Result<(), MetalW8Error> {
        self.validate_decode_input(full_attention_residual)?;
        let execution = self.inner.decode(
            full_attention_residual,
            &mut self.normalized_hidden,
            &mut self.candidate_token_ids,
            1,
        );
        self.record_execution(&execution);
        if execution.result.is_err() {
            self.terminal_error = true;
            self.stats.terminal_error = true;
        }
        execution.result
    }

    #[cfg(any(test, debug_assertions))]
    #[doc(hidden)]
    pub fn inject_nonfinite_staging_after_bridge_success_for_testing(
        &mut self,
        input: &[f32],
    ) -> Result<(), MetalW8Error> {
        self.execute_bridge_success_for_staging_fault_for_testing(input)?;
        self.normalized_hidden[0] = f32::NAN;
        self.validate_staged_output()
    }

    #[cfg(any(test, debug_assertions))]
    #[doc(hidden)]
    pub fn inject_duplicate_staging_after_bridge_success_for_testing(
        &mut self,
        input: &[f32],
    ) -> Result<(), MetalW8Error> {
        self.execute_bridge_success_for_staging_fault_for_testing(input)?;
        self.candidate_token_ids[1] = self.candidate_token_ids[0];
        self.validate_staged_output()
    }

    #[cfg(any(test, debug_assertions))]
    #[doc(hidden)]
    pub fn inject_out_of_range_staging_after_bridge_success_for_testing(
        &mut self,
        input: &[f32],
    ) -> Result<(), MetalW8Error> {
        self.execute_bridge_success_for_staging_fault_for_testing(input)?;
        self.candidate_token_ids[0] = self.inner.vocab_size() as u32;
        self.validate_staged_output()
    }

    #[cfg(any(test, debug_assertions))]
    fn execute_bridge_success_for_staging_fault_for_testing(
        &mut self,
        input: &[f32],
    ) -> Result<(), MetalW8Error> {
        self.validate_decode_input(input)?;
        let execution = self.inner.decode(
            input,
            &mut self.normalized_hidden,
            &mut self.candidate_token_ids,
            0,
        );
        self.record_execution(&execution);
        if let Err(error) = execution.result {
            self.terminal_error = true;
            self.stats.terminal_error = true;
            return Err(error);
        }
        Ok(())
    }

    #[cfg(any(test, debug_assertions))]
    #[doc(hidden)]
    pub fn inject_nonfinite_normalized_output_for_testing(
        &mut self,
        input: &[f32],
    ) -> Result<(), MetalW8Error> {
        self.inject_malformed_output_for_testing(input, 2)
    }

    #[cfg(any(test, debug_assertions))]
    #[doc(hidden)]
    pub fn inject_duplicate_candidate_output_for_testing(
        &mut self,
        input: &[f32],
    ) -> Result<(), MetalW8Error> {
        self.inject_malformed_output_for_testing(input, 3)
    }

    #[cfg(any(test, debug_assertions))]
    #[doc(hidden)]
    pub fn inject_out_of_range_candidate_output_for_testing(
        &mut self,
        input: &[f32],
    ) -> Result<(), MetalW8Error> {
        self.inject_malformed_output_for_testing(input, 4)
    }

    #[cfg(any(test, debug_assertions))]
    fn inject_malformed_output_for_testing(
        &mut self,
        input: &[f32],
        fault_mode: u32,
    ) -> Result<(), MetalW8Error> {
        self.validate_decode_input(input)?;
        let execution = self.inner.decode(
            input,
            &mut self.normalized_hidden,
            &mut self.candidate_token_ids,
            fault_mode,
        );
        self.record_execution(&execution);
        if execution.result.is_err() {
            self.terminal_error = true;
            self.stats.terminal_error = true;
        }
        execution.result
    }

    fn validate_decode_input(&self, input: &[f32]) -> Result<(), MetalW8Error> {
        if self.terminal_error {
            return Err(MetalW8Error::new(
                "Metal W8 tail MLP+head v1 is terminal after a decode failure; reset before retry",
            ));
        }
        if input.len() != self.normalized_hidden.len() {
            return Err(MetalW8Error::new(format!(
                "Metal W8 tail MLP+head v1 input has {} elements, expected {}",
                input.len(),
                self.normalized_hidden.len()
            )));
        }
        if let Some(index) = input.iter().position(|value| !value.is_finite()) {
            return Err(MetalW8Error::new(format!(
                "Metal W8 tail MLP+head v1 input contains a non-finite value at element {index}"
            )));
        }
        Ok(())
    }

    fn record_execution(&mut self, execution: &platform::TailExecutionV1) {
        self.stats.decode_calls += 1;
        if execution.result.is_ok() {
            self.stats.successful_decodes += 1;
        } else {
            self.stats.failed_decodes += 1;
        }
        let receipt = execution.receipt;
        self.stats.host_to_device_bytes += receipt.host_to_device_bytes as usize;
        self.stats.device_to_host_bytes += receipt.device_to_host_bytes as usize;
        self.stats.command_buffers += receipt.command_buffers as usize;
        self.stats.compute_encoders += receipt.compute_encoders as usize;
        self.stats.kernel_dispatches += receipt.kernel_dispatches as usize;
        self.stats.buffer_barriers += receipt.buffer_barriers as usize;
        self.stats.commits += receipt.commits as usize;
        self.stats.waits += receipt.waits as usize;
        self.stats.output_commits += receipt.output_commits as usize;
        self.stats.last_output_commit_mask = receipt.output_commit_mask;
    }

    fn validate_staged_output(&mut self) -> Result<(), MetalW8Error> {
        if let Err(error) = validate_published_output(
            &self.normalized_hidden,
            self.candidate_token_ids,
            self.inner.vocab_size(),
        ) {
            debug_assert!(self.stats.successful_decodes > 0);
            self.stats.successful_decodes -= 1;
            self.stats.failed_decodes += 1;
            self.terminal_error = true;
            self.stats.terminal_error = true;
            return Err(error);
        }
        Ok(())
    }
}

fn validate_published_output(
    normalized_hidden: &[f32],
    candidate_token_ids: [u32; W8_TOP_K],
    vocab_size: usize,
) -> Result<(), MetalW8Error> {
    if let Some(index) = normalized_hidden
        .iter()
        .position(|value| !value.is_finite())
    {
        return Err(MetalW8Error::new(format!(
            "Metal W8 tail MLP+head v1 normalized output contains a non-finite value at element {index}"
        )));
    }
    for (index, &token) in candidate_token_ids.iter().enumerate() {
        if token as usize >= vocab_size {
            return Err(MetalW8Error::new(format!(
                "Metal W8 tail MLP+head v1 candidate {index} token {token} is outside vocabulary {vocab_size}"
            )));
        }
        if candidate_token_ids[..index].contains(&token) {
            return Err(MetalW8Error::new(format!(
                "Metal W8 tail MLP+head v1 returned duplicate candidate token {token}"
            )));
        }
    }
    Ok(())
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

#[cfg(target_os = "macos")]
mod platform {
    use super::{
        MetalW8Error, PackedW8TailMlpHeadV1, TailMlpHeadBufferLedgerV1,
        TailMlpHeadRowsKernelReceiptV1, TailMlpHeadRowsKernelV1, W8_TOP_K,
    };
    use std::ffi::{c_char, c_int, c_void, CStr};
    use std::ptr::NonNull;

    const ERROR_CAPACITY: usize = 1024;

    #[repr(C)]
    struct TailDescriptorV1 {
        gate_up_weights: *const i8,
        gate_up_scales: *const f32,
        down_weights: *const i8,
        down_scales: *const f32,
        post_attention_rms_weight: *const f32,
        final_rms_weight: *const f32,
        vocab_weights: *const i8,
        vocab_scales: *const f32,
        hidden_size: u32,
        intermediate_size: u32,
        vocab_size: u32,
        rms_norm_eps: f32,
    }

    #[repr(C)]
    #[derive(Clone, Copy, Debug, Default)]
    pub(super) struct TailExecutionReceiptV1 {
        pub(super) host_to_device_bytes: u64,
        pub(super) device_to_host_bytes: u64,
        pub(super) command_buffers: u32,
        pub(super) compute_encoders: u32,
        pub(super) kernel_dispatches: u32,
        pub(super) buffer_barriers: u32,
        pub(super) commits: u32,
        pub(super) waits: u32,
        pub(super) output_commits: u32,
        pub(super) output_commit_mask: u32,
    }

    #[repr(C)]
    #[derive(Clone, Copy, Debug, Default)]
    struct RawTailRowsKernelReceiptV1 {
        rows_kernel: u32,
        rows_per_threadgroup: u32,
        rows_per_simdgroup: u32,
        simdgroups_per_threadgroup: u32,
        threads_per_threadgroup: u32,
        partial_count: u32,
        partial_topk_bytes: u64,
        pipeline_max_total_threads_per_threadgroup: u32,
        pipeline_thread_execution_width: u32,
    }

    pub(super) struct TailExecutionV1 {
        pub(super) receipt: TailExecutionReceiptV1,
        pub(super) result: Result<(), MetalW8Error>,
    }

    extern "C" {
        fn apxinf_metal_w8_tail_mlp_head_create_v1(
            descriptor: *const TailDescriptorV1,
            output: *mut *mut c_void,
            error: *mut c_char,
            error_capacity: usize,
        ) -> c_int;
        fn apxinf_metal_w8_tail_mlp_head_create_with_rows_kernel_v1(
            descriptor: *const TailDescriptorV1,
            rows_kernel: u32,
            output: *mut *mut c_void,
            error: *mut c_char,
            error_capacity: usize,
        ) -> c_int;
        fn apxinf_metal_w8_tail_mlp_head_rows_kernel_receipt_v1(
            handle: *mut c_void,
            receipt: *mut RawTailRowsKernelReceiptV1,
            error: *mut c_char,
            error_capacity: usize,
        ) -> c_int;
        fn apxinf_metal_w8_tail_mlp_head_decode_v1(
            handle: *mut c_void,
            input: *const f32,
            input_count: u32,
            normalized_hidden: *mut f32,
            normalized_hidden_count: u32,
            candidate_token_ids: *mut u32,
            candidate_count: u32,
            fault_mode: u32,
            receipt: *mut TailExecutionReceiptV1,
            error: *mut c_char,
            error_capacity: usize,
        ) -> c_int;
        fn apxinf_metal_w8_tail_mlp_head_reset_v1(
            handle: *mut c_void,
            error: *mut c_char,
            error_capacity: usize,
        ) -> c_int;
        fn apxinf_metal_w8_tail_mlp_head_destroy_v1(handle: *mut c_void);
    }

    pub(super) struct TailHandleV1 {
        handle: NonNull<c_void>,
        vocab_size: usize,
        rows_kernel_receipt: TailMlpHeadRowsKernelReceiptV1,
    }

    impl TailHandleV1 {
        pub(super) fn new(
            weights: &PackedW8TailMlpHeadV1,
            rows_kernel: TailMlpHeadRowsKernelV1,
            buffer_ledger: TailMlpHeadBufferLedgerV1,
        ) -> Result<Self, MetalW8Error> {
            let descriptor = TailDescriptorV1 {
                gate_up_weights: weights.mlp.gate_up.values().as_ptr(),
                gate_up_scales: weights.mlp.gate_up.scales().as_ptr(),
                down_weights: weights.mlp.down.values().as_ptr(),
                down_scales: weights.mlp.down.scales().as_ptr(),
                post_attention_rms_weight: weights.post_attention_rms_weight.as_ptr(),
                final_rms_weight: weights.final_rms_weight.as_ptr(),
                vocab_weights: weights.vocab.values().as_ptr(),
                vocab_scales: weights.vocab.scales().as_ptr(),
                hidden_size: weights.hidden_size() as u32,
                intermediate_size: weights.intermediate_size() as u32,
                vocab_size: weights.vocab_size() as u32,
                rms_norm_eps: weights.rms_norm_eps,
            };
            let mut output = std::ptr::null_mut();
            let mut error = [0 as c_char; ERROR_CAPACITY];
            let status = unsafe {
                match rows_kernel {
                    TailMlpHeadRowsKernelV1::LegacyR8Sg8 => {
                        apxinf_metal_w8_tail_mlp_head_create_v1(
                            &descriptor,
                            &mut output,
                            error.as_mut_ptr(),
                            error.len(),
                        )
                    }
                    TailMlpHeadRowsKernelV1::Pair2R16Sg8 => {
                        apxinf_metal_w8_tail_mlp_head_create_with_rows_kernel_v1(
                            &descriptor,
                            rows_kernel.abi_value(),
                            &mut output,
                            error.as_mut_ptr(),
                            error.len(),
                        )
                    }
                }
            };
            if status != 0 {
                return Err(bridge_error("create Metal W8 tail MLP+head v1", &error));
            }
            let handle = NonNull::new(output).ok_or_else(|| {
                MetalW8Error::new("create Metal W8 tail MLP+head v1 returned a null handle")
            })?;
            let mut raw_receipt = RawTailRowsKernelReceiptV1::default();
            error.fill(0);
            let receipt_status = unsafe {
                apxinf_metal_w8_tail_mlp_head_rows_kernel_receipt_v1(
                    handle.as_ptr(),
                    &mut raw_receipt,
                    error.as_mut_ptr(),
                    error.len(),
                )
            };
            if receipt_status != 0 {
                unsafe { apxinf_metal_w8_tail_mlp_head_destroy_v1(handle.as_ptr()) };
                return Err(bridge_error(
                    "read Metal W8 tail MLP+head v1 rows-kernel receipt",
                    &error,
                ));
            }
            let observed_kernel = TailMlpHeadRowsKernelV1::from_abi_value(raw_receipt.rows_kernel)
                .ok_or_else(|| {
                    unsafe { apxinf_metal_w8_tail_mlp_head_destroy_v1(handle.as_ptr()) };
                    MetalW8Error::new(
                        "Metal W8 tail MLP+head v1 returned an unknown rows-kernel selector",
                    )
                })?;
            let expected_shape = rows_kernel.execution_shape();
            let expected_partial_count =
                weights.vocab_size().div_ceil(expected_shape.0 as usize) as u32;
            let observed_partial_topk_bytes = usize::try_from(raw_receipt.partial_topk_bytes)
                .map_err(|_| {
                    unsafe { apxinf_metal_w8_tail_mlp_head_destroy_v1(handle.as_ptr()) };
                    MetalW8Error::new("Metal W8 tail MLP+head v1 rows-kernel receipt exceeds usize")
                })?;
            let receipt_matches = observed_kernel == rows_kernel
                && raw_receipt.rows_per_threadgroup == expected_shape.0
                && raw_receipt.rows_per_simdgroup == expected_shape.1
                && raw_receipt.simdgroups_per_threadgroup == expected_shape.2
                && raw_receipt.threads_per_threadgroup == expected_shape.3
                && raw_receipt.partial_count == expected_partial_count
                && observed_partial_topk_bytes == buffer_ledger.partial_topk_bytes
                && raw_receipt.pipeline_max_total_threads_per_threadgroup >= expected_shape.3
                && raw_receipt.pipeline_thread_execution_width == 32
                && raw_receipt
                    .pipeline_thread_execution_width
                    .checked_mul(raw_receipt.simdgroups_per_threadgroup)
                    == Some(raw_receipt.threads_per_threadgroup);
            if !receipt_matches {
                unsafe { apxinf_metal_w8_tail_mlp_head_destroy_v1(handle.as_ptr()) };
                return Err(MetalW8Error::new(
                    "Metal W8 tail MLP+head v1 rows-kernel runtime receipt drifted from the requested selector and ledger",
                ));
            }
            let rows_kernel_receipt = TailMlpHeadRowsKernelReceiptV1 {
                kernel: observed_kernel,
                rows_per_threadgroup: raw_receipt.rows_per_threadgroup,
                rows_per_simdgroup: raw_receipt.rows_per_simdgroup,
                simdgroups_per_threadgroup: raw_receipt.simdgroups_per_threadgroup,
                threads_per_threadgroup: raw_receipt.threads_per_threadgroup,
                partial_count: raw_receipt.partial_count,
                partial_topk_bytes: observed_partial_topk_bytes,
                pipeline_max_total_threads_per_threadgroup: raw_receipt
                    .pipeline_max_total_threads_per_threadgroup,
                pipeline_thread_execution_width: raw_receipt.pipeline_thread_execution_width,
            };
            Ok(Self {
                handle,
                vocab_size: weights.vocab_size(),
                rows_kernel_receipt,
            })
        }

        pub(super) fn rows_kernel_receipt(&self) -> TailMlpHeadRowsKernelReceiptV1 {
            self.rows_kernel_receipt
        }

        pub(super) fn decode(
            &mut self,
            input: &[f32],
            normalized_hidden: &mut [f32],
            candidate_token_ids: &mut [u32; W8_TOP_K],
            fault_mode: u32,
        ) -> TailExecutionV1 {
            let mut receipt = TailExecutionReceiptV1::default();
            let mut error = [0 as c_char; ERROR_CAPACITY];
            let status = unsafe {
                apxinf_metal_w8_tail_mlp_head_decode_v1(
                    self.handle.as_ptr(),
                    input.as_ptr(),
                    input.len() as u32,
                    normalized_hidden.as_mut_ptr(),
                    normalized_hidden.len() as u32,
                    candidate_token_ids.as_mut_ptr(),
                    W8_TOP_K as u32,
                    fault_mode,
                    &mut receipt,
                    error.as_mut_ptr(),
                    error.len(),
                )
            };
            let result = if status == 0 {
                Ok(())
            } else {
                Err(bridge_error("run Metal W8 tail MLP+head v1 decode", &error))
            };
            TailExecutionV1 { receipt, result }
        }

        pub(super) fn reset(&mut self) -> Result<(), MetalW8Error> {
            let mut error = [0 as c_char; ERROR_CAPACITY];
            let status = unsafe {
                apxinf_metal_w8_tail_mlp_head_reset_v1(
                    self.handle.as_ptr(),
                    error.as_mut_ptr(),
                    error.len(),
                )
            };
            if status == 0 {
                Ok(())
            } else {
                Err(bridge_error("reset Metal W8 tail MLP+head v1", &error))
            }
        }

        pub(super) fn vocab_size(&self) -> usize {
            self.vocab_size
        }
    }

    impl Drop for TailHandleV1 {
        fn drop(&mut self) {
            unsafe { apxinf_metal_w8_tail_mlp_head_destroy_v1(self.handle.as_ptr()) };
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
        MetalW8Error, PackedW8TailMlpHeadV1, TailMlpHeadBufferLedgerV1,
        TailMlpHeadRowsKernelReceiptV1, TailMlpHeadRowsKernelV1, W8_TOP_K,
    };

    #[derive(Clone, Copy, Debug, Default)]
    pub(super) struct TailExecutionReceiptV1 {
        pub(super) host_to_device_bytes: u64,
        pub(super) device_to_host_bytes: u64,
        pub(super) command_buffers: u32,
        pub(super) compute_encoders: u32,
        pub(super) kernel_dispatches: u32,
        pub(super) buffer_barriers: u32,
        pub(super) commits: u32,
        pub(super) waits: u32,
        pub(super) output_commits: u32,
        pub(super) output_commit_mask: u32,
    }

    pub(super) struct TailExecutionV1 {
        pub(super) receipt: TailExecutionReceiptV1,
        pub(super) result: Result<(), MetalW8Error>,
    }

    pub(super) struct TailHandleV1;

    impl TailHandleV1 {
        pub(super) fn new(
            _weights: &PackedW8TailMlpHeadV1,
            _rows_kernel: TailMlpHeadRowsKernelV1,
            _buffer_ledger: TailMlpHeadBufferLedgerV1,
        ) -> Result<Self, MetalW8Error> {
            Err(MetalW8Error::new(
                "Metal W8 tail MLP+head v1 requires macOS",
            ))
        }

        pub(super) fn rows_kernel_receipt(&self) -> TailMlpHeadRowsKernelReceiptV1 {
            unreachable!("non-macOS tail handle cannot be constructed")
        }

        pub(super) fn decode(
            &mut self,
            _input: &[f32],
            _normalized_hidden: &mut [f32],
            _candidate_token_ids: &mut [u32; W8_TOP_K],
            _fault_mode: u32,
        ) -> TailExecutionV1 {
            TailExecutionV1 {
                receipt: TailExecutionReceiptV1::default(),
                result: Err(MetalW8Error::new(
                    "Metal W8 tail MLP+head v1 requires macOS",
                )),
            }
        }

        pub(super) fn reset(&mut self) -> Result<(), MetalW8Error> {
            Err(MetalW8Error::new(
                "Metal W8 tail MLP+head v1 requires macOS",
            ))
        }

        pub(super) fn vocab_size(&self) -> usize {
            0
        }
    }
}
