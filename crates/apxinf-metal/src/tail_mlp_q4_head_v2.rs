use crate::{
    validate_topk4_exclusions, MetalW8Error, PackedQ4_0RowsV1, PackedW8MlpBlock, W8GroupSize,
    Q4_0_BLOCK_SIZE_V1, Q4_0_PACKED_BYTES_PER_BLOCK_V1, W8_TOP_K,
};

pub const W8_Q4_TAIL_ABI_VERSION_V2: u32 = 2;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TailMlpQ4HeadBufferLedgerV2 {
    pub scope: &'static str,
    pub exclusions: &'static str,
    pub abi_version: u32,
    pub allocated_buffers: usize,
    pub shared_buffers: usize,
    pub private_buffers: usize,
    pub w8_mlp_weight_bytes: usize,
    pub w8_mlp_scale_bytes: usize,
    pub q4_vocab_weight_bytes: usize,
    pub w8_vocab_weight_bytes: usize,
    pub w8_vocab_scale_bytes: usize,
    pub f32_parameter_bytes: usize,
    pub hidden_activation_bytes: usize,
    pub mlp_activation_bytes: usize,
    pub partial_topk_bytes: usize,
    pub full_score_scratch_bytes: usize,
    pub output_token_bytes: usize,
    pub status_bytes: usize,
    pub total_persistent_bytes: usize,
    pub host_to_device_bytes_per_decode: usize,
    pub device_to_host_bytes_per_decode: usize,
    pub state_host_transfer_bytes_per_decode: usize,
    pub command_buffers_per_decode: usize,
    pub compute_encoders_per_decode: usize,
    pub kernel_dispatches_per_decode: usize,
    pub buffer_barriers_per_decode: usize,
    pub commits_per_decode: usize,
    pub waits_per_decode: usize,
}

impl TailMlpQ4HeadBufferLedgerV2 {
    pub fn from_dimensions(
        hidden_size: usize,
        intermediate_size: usize,
        vocab_size: usize,
    ) -> Result<Self, MetalW8Error> {
        if hidden_size == 0
            || intermediate_size == 0
            || vocab_size < W8_TOP_K
            || hidden_size % 64 != 0
            || hidden_size % Q4_0_BLOCK_SIZE_V1 != 0
            || intermediate_size % 64 != 0
        {
            return Err(MetalW8Error::new(
                "Metal W8+Q4_0 tail v2 ledger requires non-zero W8-G64/Q4-block32 shapes and at least four vocabulary rows",
            ));
        }
        if hidden_size > u32::MAX as usize
            || intermediate_size > u32::MAX as usize / 2
            || vocab_size > u32::MAX as usize
        {
            return Err(MetalW8Error::new(
                "Metal W8+Q4_0 tail v2 ledger dimensions exceed the u32 ABI",
            ));
        }
        let hidden_intermediate = hidden_size
            .checked_mul(intermediate_size)
            .ok_or_else(|| MetalW8Error::new("Metal W8+Q4_0 tail v2 ledger overflow"))?;
        let w8_mlp_weight_bytes = hidden_intermediate
            .checked_mul(3)
            .ok_or_else(|| MetalW8Error::new("Metal W8+Q4_0 tail v2 W8 weight overflow"))?;
        let w8_mlp_scale_bytes = hidden_intermediate
            .checked_mul(3)
            .and_then(|count| count.checked_div(64))
            .and_then(|count| count.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| MetalW8Error::new("Metal W8+Q4_0 tail v2 W8 scale overflow"))?;
        let q4_vocab_weight_bytes = vocab_size
            .checked_mul(hidden_size / Q4_0_BLOCK_SIZE_V1)
            .and_then(|blocks| blocks.checked_mul(Q4_0_PACKED_BYTES_PER_BLOCK_V1))
            .ok_or_else(|| MetalW8Error::new("Metal W8+Q4_0 tail v2 Q4 weight overflow"))?;
        let f32_parameter_bytes = hidden_size
            .checked_mul(2)
            .and_then(|count| count.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| MetalW8Error::new("Metal W8+Q4_0 tail v2 RMS overflow"))?;
        let hidden_activation_bytes = f32_parameter_bytes;
        let mlp_activation_bytes = intermediate_size
            .checked_mul(3)
            .and_then(|count| count.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| MetalW8Error::new("Metal W8+Q4_0 tail v2 activation overflow"))?;
        let partial_topk_bytes = vocab_size
            .checked_add(7)
            .map(|rows| rows / 8)
            .and_then(|count| count.checked_mul(W8_TOP_K))
            .and_then(|count| count.checked_mul(2 * std::mem::size_of::<u32>()))
            .ok_or_else(|| MetalW8Error::new("Metal W8+Q4_0 tail v2 partial top-k overflow"))?;
        let output_token_bytes = W8_TOP_K * std::mem::size_of::<u32>();
        let status_bytes = std::mem::size_of::<u32>();
        let total_persistent_bytes = [
            w8_mlp_weight_bytes,
            w8_mlp_scale_bytes,
            q4_vocab_weight_bytes,
            f32_parameter_bytes,
            hidden_activation_bytes,
            mlp_activation_bytes,
            partial_topk_bytes,
            output_token_bytes,
            status_bytes,
        ]
        .into_iter()
        .try_fold(0usize, |total, bytes| total.checked_add(bytes))
        .ok_or_else(|| MetalW8Error::new("Metal W8+Q4_0 tail v2 total ledger overflow"))?;
        let host_to_device_bytes_per_decode = hidden_size
            .checked_mul(std::mem::size_of::<f32>())
            .and_then(|bytes| bytes.checked_add(output_token_bytes))
            .and_then(|bytes| bytes.checked_add(status_bytes))
            .ok_or_else(|| MetalW8Error::new("Metal W8+Q4_0 tail v2 H2D ledger overflow"))?;
        let device_to_host_bytes_per_decode = host_to_device_bytes_per_decode;
        Ok(Self {
            scope: "resident-mtlbuffer-only-plus-explicit-control-transfers-v2",
            exclusions: "CPU packed weights, host F32 tied embedding and exact four-candidate rerank, host allocations, Metal pipelines/libraries/queues/commands, driver allocations, attention/KV, model loader, and all earlier model layers",
            abi_version: W8_Q4_TAIL_ABI_VERSION_V2,
            allocated_buffers: 13,
            shared_buffers: 10,
            private_buffers: 3,
            w8_mlp_weight_bytes,
            w8_mlp_scale_bytes,
            q4_vocab_weight_bytes,
            w8_vocab_weight_bytes: 0,
            w8_vocab_scale_bytes: 0,
            f32_parameter_bytes,
            hidden_activation_bytes,
            mlp_activation_bytes,
            partial_topk_bytes,
            full_score_scratch_bytes: 0,
            output_token_bytes,
            status_bytes,
            total_persistent_bytes,
            host_to_device_bytes_per_decode,
            device_to_host_bytes_per_decode,
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

#[derive(Clone, Debug, PartialEq)]
pub struct TailMlpQ4HeadDecodeResultV2 {
    pub normalized_hidden: Vec<f32>,
    pub candidate_token_ids: [u32; W8_TOP_K],
}

#[derive(Clone, Debug)]
pub struct PackedW8TailMlpQ4HeadV2 {
    pub(crate) mlp: PackedW8MlpBlock,
    pub(crate) post_attention_rms_weight: Vec<f32>,
    pub(crate) final_rms_weight: Vec<f32>,
    pub(crate) rms_norm_eps: f32,
    pub(crate) q4_vocab: PackedQ4_0RowsV1,
}

impl PackedW8TailMlpQ4HeadV2 {
    pub fn new(
        mlp: PackedW8MlpBlock,
        post_attention_rms_weight: &[f32],
        final_rms_weight: &[f32],
        rms_norm_eps: f32,
        q4_vocab: PackedQ4_0RowsV1,
    ) -> Result<Self, MetalW8Error> {
        if !rms_norm_eps.is_finite() || rms_norm_eps < 0.0 {
            return Err(MetalW8Error::new(
                "Metal W8+Q4_0 tail v2 RMS epsilon must be finite and non-negative",
            ));
        }
        let hidden_size = mlp.down.rows;
        let intermediate_size = mlp.down.columns;
        if mlp.gate_up.columns != hidden_size
            || mlp.gate_up.rows
                != intermediate_size.checked_mul(2).ok_or_else(|| {
                    MetalW8Error::new("Metal W8+Q4_0 tail v2 gate/up row count overflow")
                })?
            || post_attention_rms_weight.len() != hidden_size
            || final_rms_weight.len() != hidden_size
            || q4_vocab.columns() != hidden_size
            || q4_vocab.rows() < W8_TOP_K
        {
            return Err(MetalW8Error::new(
                "Metal W8+Q4_0 tail v2 hidden or vocabulary shapes differ",
            ));
        }
        for (label, values) in [
            ("post-attention RMS weight", post_attention_rms_weight),
            ("final RMS weight", final_rms_weight),
        ] {
            if let Some(index) = values.iter().position(|value| !value.is_finite()) {
                return Err(MetalW8Error::new(format!(
                    "Metal W8+Q4_0 tail v2 {label} contains a non-finite value at element {index}"
                )));
            }
        }
        for (label, group_size) in [
            ("MLP gate/up projection", mlp.gate_up.group_size()),
            ("MLP down projection", mlp.down.group_size()),
        ] {
            if group_size != W8GroupSize::G64 {
                return Err(MetalW8Error::new(format!(
                    "Metal W8+Q4_0 tail v2 {label} requires group size 64, got {}",
                    group_size.columns()
                )));
            }
        }
        TailMlpQ4HeadBufferLedgerV2::from_dimensions(
            hidden_size,
            intermediate_size,
            q4_vocab.rows(),
        )?;
        Ok(Self {
            mlp,
            post_attention_rms_weight: post_attention_rms_weight.to_vec(),
            final_rms_weight: final_rms_weight.to_vec(),
            rms_norm_eps,
            q4_vocab,
        })
    }

    pub fn hidden_size(&self) -> usize {
        self.mlp.down.rows
    }

    pub fn intermediate_size(&self) -> usize {
        self.mlp.down.columns
    }

    pub fn vocab_size(&self) -> usize {
        self.q4_vocab.rows()
    }

    pub fn buffer_ledger(&self) -> Result<TailMlpQ4HeadBufferLedgerV2, MetalW8Error> {
        TailMlpQ4HeadBufferLedgerV2::from_dimensions(
            self.hidden_size(),
            self.intermediate_size(),
            self.vocab_size(),
        )
    }

    pub fn decode_reference_excluding(
        &self,
        full_attention_residual: &[f32],
        excluded_tokens: &[u32],
    ) -> Result<TailMlpQ4HeadDecodeResultV2, MetalW8Error> {
        if full_attention_residual.len() != self.hidden_size() {
            return Err(MetalW8Error::new(format!(
                "Metal W8+Q4_0 tail v2 input has {} elements, expected {}",
                full_attention_residual.len(),
                self.hidden_size()
            )));
        }
        require_finite_v2(full_attention_residual, "Metal W8+Q4_0 tail v2 input")?;
        validate_topk4_exclusions(excluded_tokens, self.vocab_size())?;
        let normalized = rms_norm_v2(
            full_attention_residual,
            &self.post_attention_rms_weight,
            self.rms_norm_eps,
        );
        let update = self.mlp.forward(&normalized)?;
        let residual = full_attention_residual
            .iter()
            .zip(update)
            .map(|(&residual, update)| residual + update)
            .collect::<Vec<_>>();
        require_finite_v2(&residual, "Metal W8+Q4_0 tail v2 residual row")?;
        let normalized_hidden = rms_norm_v2(&residual, &self.final_rms_weight, self.rms_norm_eps);
        require_finite_v2(
            &normalized_hidden,
            "Metal W8+Q4_0 tail v2 final normalized row",
        )?;
        let candidates = self
            .q4_vocab
            .topk_excluding(&normalized_hidden, W8_TOP_K, excluded_tokens)
            .map_err(|error| {
                MetalW8Error::new(format!("Q4_0 tail v2 candidate oracle: {error}"))
            })?;
        let candidate_token_ids = candidates.try_into().map_err(|_| {
            MetalW8Error::new("Metal W8+Q4_0 tail v2 oracle returned a non-top4 result")
        })?;
        Ok(TailMlpQ4HeadDecodeResultV2 {
            normalized_hidden,
            candidate_token_ids,
        })
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TailMlpQ4HeadMetalStatsV2 {
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
pub struct TailMlpQ4HeadDecodeViewV2<'a> {
    pub normalized_hidden: &'a [f32],
    pub candidate_token_ids: [u32; W8_TOP_K],
}

pub struct MetalW8TailMlpQ4HeadV2 {
    inner: platform::TailHandleV2,
    normalized_hidden: Vec<f32>,
    candidate_token_ids: [u32; W8_TOP_K],
    terminal_error: bool,
    stats: TailMlpQ4HeadMetalStatsV2,
    buffer_ledger: TailMlpQ4HeadBufferLedgerV2,
}

impl MetalW8TailMlpQ4HeadV2 {
    pub fn from_packed(weights: &PackedW8TailMlpQ4HeadV2) -> Result<Self, MetalW8Error> {
        let buffer_ledger = weights.buffer_ledger()?;
        Ok(Self {
            inner: platform::TailHandleV2::new(weights)?,
            normalized_hidden: vec![0.0; weights.hidden_size()],
            candidate_token_ids: [u32::MAX; W8_TOP_K],
            terminal_error: false,
            stats: TailMlpQ4HeadMetalStatsV2::default(),
            buffer_ledger,
        })
    }

    pub fn decode(&mut self, input: &[f32]) -> Result<TailMlpQ4HeadDecodeViewV2<'_>, MetalW8Error> {
        self.decode_excluding(input, &[])
    }

    pub fn decode_excluding(
        &mut self,
        input: &[f32],
        excluded_tokens: &[u32],
    ) -> Result<TailMlpQ4HeadDecodeViewV2<'_>, MetalW8Error> {
        self.validate_decode_input(input)?;
        validate_topk4_exclusions(excluded_tokens, self.inner.vocab_size())?;
        let execution = self.inner.decode_excluding(
            input,
            &mut self.normalized_hidden,
            &mut self.candidate_token_ids,
            excluded_tokens,
            0,
        );
        self.record_execution(&execution);
        if let Err(error) = execution.result {
            self.latch_terminal();
            return Err(error);
        }
        self.validate_staged_output(excluded_tokens)?;
        Ok(TailMlpQ4HeadDecodeViewV2 {
            normalized_hidden: &self.normalized_hidden,
            candidate_token_ids: self.candidate_token_ids,
        })
    }

    pub fn reset(&mut self) -> Result<(), MetalW8Error> {
        self.inner.reset()?;
        self.normalized_hidden.fill(0.0);
        self.candidate_token_ids.fill(u32::MAX);
        self.terminal_error = false;
        self.stats = TailMlpQ4HeadMetalStatsV2::default();
        Ok(())
    }

    pub fn stats(&self) -> TailMlpQ4HeadMetalStatsV2 {
        self.stats
    }

    pub fn buffer_ledger(&self) -> TailMlpQ4HeadBufferLedgerV2 {
        self.buffer_ledger
    }

    #[cfg(any(test, debug_assertions))]
    #[doc(hidden)]
    pub fn inject_failure_after_gpu_execution_for_testing(
        &mut self,
        input: &[f32],
    ) -> Result<(), MetalW8Error> {
        self.inject_fault(input, 1)
    }

    #[cfg(any(test, debug_assertions))]
    #[doc(hidden)]
    pub fn inject_nonfinite_normalized_output_for_testing(
        &mut self,
        input: &[f32],
    ) -> Result<(), MetalW8Error> {
        self.inject_fault(input, 2)
    }

    #[cfg(any(test, debug_assertions))]
    #[doc(hidden)]
    pub fn inject_duplicate_candidate_output_for_testing(
        &mut self,
        input: &[f32],
    ) -> Result<(), MetalW8Error> {
        self.inject_fault(input, 3)
    }

    #[cfg(any(test, debug_assertions))]
    #[doc(hidden)]
    pub fn inject_out_of_range_candidate_output_for_testing(
        &mut self,
        input: &[f32],
    ) -> Result<(), MetalW8Error> {
        self.inject_fault(input, 4)
    }

    #[cfg(any(test, debug_assertions))]
    fn inject_fault(&mut self, input: &[f32], fault_mode: u32) -> Result<(), MetalW8Error> {
        self.validate_decode_input(input)?;
        let execution = self.inner.decode_excluding(
            input,
            &mut self.normalized_hidden,
            &mut self.candidate_token_ids,
            &[],
            fault_mode,
        );
        self.record_execution(&execution);
        if execution.result.is_err() {
            self.latch_terminal();
        }
        execution.result
    }

    fn validate_decode_input(&self, input: &[f32]) -> Result<(), MetalW8Error> {
        if self.terminal_error {
            return Err(MetalW8Error::new(
                "Metal W8+Q4_0 tail v2 is terminal after a decode failure; reset before retry",
            ));
        }
        if input.len() != self.normalized_hidden.len() {
            return Err(MetalW8Error::new(format!(
                "Metal W8+Q4_0 tail v2 input has {} elements, expected {}",
                input.len(),
                self.normalized_hidden.len()
            )));
        }
        require_finite_v2(input, "Metal W8+Q4_0 tail v2 input")
    }

    fn record_execution(&mut self, execution: &platform::TailExecutionV2) {
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

    fn validate_staged_output(&mut self, excluded_tokens: &[u32]) -> Result<(), MetalW8Error> {
        if let Err(error) = validate_published_output_v2(
            &self.normalized_hidden,
            self.candidate_token_ids,
            self.inner.vocab_size(),
            excluded_tokens,
        ) {
            debug_assert!(self.stats.successful_decodes > 0);
            self.stats.successful_decodes -= 1;
            self.stats.failed_decodes += 1;
            self.latch_terminal();
            return Err(error);
        }
        Ok(())
    }

    fn latch_terminal(&mut self) {
        self.terminal_error = true;
        self.stats.terminal_error = true;
    }
}

fn require_finite_v2(values: &[f32], label: &str) -> Result<(), MetalW8Error> {
    if let Some(index) = values.iter().position(|value| !value.is_finite()) {
        return Err(MetalW8Error::new(format!(
            "{label} contains a non-finite value at element {index}"
        )));
    }
    Ok(())
}

fn validate_published_output_v2(
    normalized_hidden: &[f32],
    candidate_token_ids: [u32; W8_TOP_K],
    vocab_size: usize,
    excluded_tokens: &[u32],
) -> Result<(), MetalW8Error> {
    require_finite_v2(normalized_hidden, "Metal W8+Q4_0 tail v2 normalized output")?;
    for (index, &token) in candidate_token_ids.iter().enumerate() {
        if token as usize >= vocab_size {
            return Err(MetalW8Error::new(format!(
                "Metal W8+Q4_0 tail v2 candidate {index} token {token} is outside vocabulary {vocab_size}"
            )));
        }
        if candidate_token_ids[..index].contains(&token) {
            return Err(MetalW8Error::new(format!(
                "Metal W8+Q4_0 tail v2 returned duplicate candidate token {token}"
            )));
        }
        if excluded_tokens.contains(&token) {
            return Err(MetalW8Error::new(format!(
                "Metal W8+Q4_0 tail v2 returned excluded candidate token {token}"
            )));
        }
    }
    Ok(())
}

fn rms_norm_v2(input: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
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
    use super::{MetalW8Error, PackedW8TailMlpQ4HeadV2, W8_Q4_TAIL_ABI_VERSION_V2, W8_TOP_K};
    use std::ffi::{c_char, c_int, c_void, CStr};
    use std::ptr::NonNull;

    const ERROR_CAPACITY: usize = 1024;

    #[repr(C)]
    struct Q4TailDescriptorV2 {
        gate_up_weights: *const i8,
        gate_up_scales: *const f32,
        down_weights: *const i8,
        down_scales: *const f32,
        post_attention_rms_weight: *const f32,
        final_rms_weight: *const f32,
        q4_vocab_blocks: *const u8,
        q4_vocab_byte_count: usize,
        hidden_size: u32,
        intermediate_size: u32,
        vocab_size: u32,
        rms_norm_eps: f32,
        abi_version: u32,
    }

    #[repr(C)]
    #[derive(Clone, Copy, Debug, Default)]
    pub(super) struct TailExecutionReceiptV2 {
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

    pub(super) struct TailExecutionV2 {
        pub(super) receipt: TailExecutionReceiptV2,
        pub(super) result: Result<(), MetalW8Error>,
    }

    extern "C" {
        fn apxinf_metal_w8_q4_tail_mlp_head_create_v2(
            descriptor: *const Q4TailDescriptorV2,
            output: *mut *mut c_void,
            error: *mut c_char,
            error_capacity: usize,
        ) -> c_int;
        fn apxinf_metal_w8_q4_tail_mlp_head_decode_excluding_v2(
            handle: *mut c_void,
            input: *const f32,
            input_count: u32,
            normalized_hidden: *mut f32,
            normalized_hidden_count: u32,
            candidate_token_ids: *mut u32,
            candidate_count: u32,
            excluded_tokens: *const u32,
            excluded_count: u32,
            fault_mode: u32,
            receipt: *mut TailExecutionReceiptV2,
            error: *mut c_char,
            error_capacity: usize,
        ) -> c_int;
        fn apxinf_metal_w8_q4_tail_mlp_head_reset_v2(
            handle: *mut c_void,
            error: *mut c_char,
            error_capacity: usize,
        ) -> c_int;
        fn apxinf_metal_w8_q4_tail_mlp_head_destroy_v2(handle: *mut c_void);
    }

    pub(super) struct TailHandleV2 {
        handle: NonNull<c_void>,
        vocab_size: usize,
    }

    impl TailHandleV2 {
        pub(super) fn new(weights: &PackedW8TailMlpQ4HeadV2) -> Result<Self, MetalW8Error> {
            let q4_bytes = weights.q4_vocab.canonical_bytes_le();
            let descriptor = Q4TailDescriptorV2 {
                gate_up_weights: weights.mlp.gate_up.values().as_ptr(),
                gate_up_scales: weights.mlp.gate_up.scales().as_ptr(),
                down_weights: weights.mlp.down.values().as_ptr(),
                down_scales: weights.mlp.down.scales().as_ptr(),
                post_attention_rms_weight: weights.post_attention_rms_weight.as_ptr(),
                final_rms_weight: weights.final_rms_weight.as_ptr(),
                q4_vocab_blocks: q4_bytes.as_ptr(),
                q4_vocab_byte_count: q4_bytes.len(),
                hidden_size: weights.hidden_size() as u32,
                intermediate_size: weights.intermediate_size() as u32,
                vocab_size: weights.vocab_size() as u32,
                rms_norm_eps: weights.rms_norm_eps,
                abi_version: W8_Q4_TAIL_ABI_VERSION_V2,
            };
            let mut output = std::ptr::null_mut();
            let mut error = [0 as c_char; ERROR_CAPACITY];
            let status = unsafe {
                apxinf_metal_w8_q4_tail_mlp_head_create_v2(
                    &descriptor,
                    &mut output,
                    error.as_mut_ptr(),
                    error.len(),
                )
            };
            if status != 0 {
                return Err(bridge_error("create Metal W8+Q4_0 tail v2", &error));
            }
            let handle = NonNull::new(output).ok_or_else(|| {
                MetalW8Error::new("create Metal W8+Q4_0 tail v2 returned a null handle")
            })?;
            Ok(Self {
                handle,
                vocab_size: weights.vocab_size(),
            })
        }

        pub(super) fn decode_excluding(
            &mut self,
            input: &[f32],
            normalized_hidden: &mut [f32],
            candidate_token_ids: &mut [u32; W8_TOP_K],
            excluded_tokens: &[u32],
            fault_mode: u32,
        ) -> TailExecutionV2 {
            let mut receipt = TailExecutionReceiptV2::default();
            let mut error = [0 as c_char; ERROR_CAPACITY];
            let status = unsafe {
                apxinf_metal_w8_q4_tail_mlp_head_decode_excluding_v2(
                    self.handle.as_ptr(),
                    input.as_ptr(),
                    input.len() as u32,
                    normalized_hidden.as_mut_ptr(),
                    normalized_hidden.len() as u32,
                    candidate_token_ids.as_mut_ptr(),
                    W8_TOP_K as u32,
                    excluded_tokens.as_ptr(),
                    excluded_tokens.len() as u32,
                    fault_mode,
                    &mut receipt,
                    error.as_mut_ptr(),
                    error.len(),
                )
            };
            let result = if status == 0 {
                Ok(())
            } else {
                Err(bridge_error("run Metal W8+Q4_0 tail v2 decode", &error))
            };
            TailExecutionV2 { receipt, result }
        }

        pub(super) fn reset(&mut self) -> Result<(), MetalW8Error> {
            let mut error = [0 as c_char; ERROR_CAPACITY];
            let status = unsafe {
                apxinf_metal_w8_q4_tail_mlp_head_reset_v2(
                    self.handle.as_ptr(),
                    error.as_mut_ptr(),
                    error.len(),
                )
            };
            if status == 0 {
                Ok(())
            } else {
                Err(bridge_error("reset Metal W8+Q4_0 tail v2", &error))
            }
        }

        pub(super) fn vocab_size(&self) -> usize {
            self.vocab_size
        }
    }

    impl Drop for TailHandleV2 {
        fn drop(&mut self) {
            unsafe { apxinf_metal_w8_q4_tail_mlp_head_destroy_v2(self.handle.as_ptr()) };
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
    use super::{MetalW8Error, PackedW8TailMlpQ4HeadV2, W8_TOP_K};

    #[derive(Clone, Copy, Debug, Default)]
    pub(super) struct TailExecutionReceiptV2 {
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

    pub(super) struct TailExecutionV2 {
        pub(super) receipt: TailExecutionReceiptV2,
        pub(super) result: Result<(), MetalW8Error>,
    }

    pub(super) struct TailHandleV2;

    impl TailHandleV2 {
        pub(super) fn new(_weights: &PackedW8TailMlpQ4HeadV2) -> Result<Self, MetalW8Error> {
            Err(MetalW8Error::new("Metal W8+Q4_0 tail v2 requires macOS"))
        }

        pub(super) fn decode_excluding(
            &mut self,
            _input: &[f32],
            _normalized_hidden: &mut [f32],
            _candidate_token_ids: &mut [u32; W8_TOP_K],
            _excluded_tokens: &[u32],
            _fault_mode: u32,
        ) -> TailExecutionV2 {
            TailExecutionV2 {
                receipt: TailExecutionReceiptV2::default(),
                result: Err(MetalW8Error::new("Metal W8+Q4_0 tail v2 requires macOS")),
            }
        }

        pub(super) fn reset(&mut self) -> Result<(), MetalW8Error> {
            Err(MetalW8Error::new("Metal W8+Q4_0 tail v2 requires macOS"))
        }

        pub(super) fn vocab_size(&self) -> usize {
            0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synthetic_weights() -> PackedW8TailMlpQ4HeadV2 {
        let hidden = 64;
        let intermediate = 64;
        let vocab = 8;
        let projection = hidden * intermediate;
        let gate = vec![0.0; projection];
        let up = vec![0.0; projection];
        let down = vec![0.0; projection];
        let mlp = PackedW8MlpBlock::pack_f32(&gate, &up, &down, hidden, intermediate).unwrap();
        let mut embedding = vec![0.0; vocab * hidden];
        for row in 0..vocab {
            embedding[row * hidden + row] = 1.0 + row as f32;
        }
        let q4 = PackedQ4_0RowsV1::pack_f32(&embedding, vocab, hidden).unwrap();
        PackedW8TailMlpQ4HeadV2::new(mlp, &vec![1.0; hidden], &vec![1.0; hidden], 1.0e-6, q4)
            .unwrap()
    }

    #[test]
    fn ledger_forbids_w8_vocab_and_full_score_scratch() {
        let ledger = TailMlpQ4HeadBufferLedgerV2::from_dimensions(1_024, 3_584, 248_320).unwrap();
        assert_eq!(ledger.abi_version, 2);
        assert_eq!(ledger.q4_vocab_weight_bytes, 143_032_320);
        assert_eq!(ledger.w8_vocab_weight_bytes, 0);
        assert_eq!(ledger.w8_vocab_scale_bytes, 0);
        assert_eq!(ledger.full_score_scratch_bytes, 0);
        assert_eq!(ledger.command_buffers_per_decode, 1);
        assert_eq!(ledger.compute_encoders_per_decode, 1);
        assert_eq!(ledger.commits_per_decode, 1);
        assert_eq!(ledger.waits_per_decode, 1);
    }

    #[test]
    fn cpu_reference_masks_before_q4_candidate_selection() {
        let packed = synthetic_weights();
        let input = (0..64).map(|index| index as f32 + 1.0).collect::<Vec<_>>();
        let unmasked = packed.decode_reference_excluding(&input, &[]).unwrap();
        let masked = packed
            .decode_reference_excluding(&input, &[unmasked.candidate_token_ids[0]])
            .unwrap();
        assert!(!masked
            .candidate_token_ids
            .contains(&unmasked.candidate_token_ids[0]));
    }

    #[test]
    fn invalid_exclusions_fail_before_reference_work() {
        let packed = synthetic_weights();
        let input = vec![1.0; 64];
        let error = packed
            .decode_reference_excluding(&input, &[1, 1])
            .unwrap_err();
        assert!(error.to_string().contains("duplicate"));
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_v2_matches_oracle_masks_and_resets_terminal_state() {
        let packed = synthetic_weights();
        let input = (0..64).map(|index| index as f32 + 1.0).collect::<Vec<_>>();
        let expected = packed.decode_reference_excluding(&input, &[]).unwrap();
        let mut metal = MetalW8TailMlpQ4HeadV2::from_packed(&packed).unwrap();

        let invalid = metal.decode_excluding(&input, &[1, 1]).unwrap_err();
        assert!(invalid.to_string().contains("duplicate"));
        assert_eq!(metal.stats(), TailMlpQ4HeadMetalStatsV2::default());

        let actual = metal.decode(&input).unwrap();
        assert_eq!(actual.candidate_token_ids, expected.candidate_token_ids);
        for (&actual, &expected) in actual
            .normalized_hidden
            .iter()
            .zip(&expected.normalized_hidden)
        {
            assert!((actual - expected).abs() <= 1.0e-5);
        }
        let first = actual.candidate_token_ids[0];
        let masked_expected = packed.decode_reference_excluding(&input, &[first]).unwrap();
        let masked_actual = metal.decode_excluding(&input, &[first]).unwrap();
        assert_eq!(
            masked_actual.candidate_token_ids,
            masked_expected.candidate_token_ids
        );
        assert!(!masked_actual.candidate_token_ids.contains(&first));
        let before_fault = metal.stats();
        assert_eq!(before_fault.command_buffers, 2);
        assert_eq!(before_fault.compute_encoders, 2);
        assert_eq!(before_fault.commits, 2);
        assert_eq!(before_fault.waits, 2);

        assert!(metal
            .inject_failure_after_gpu_execution_for_testing(&input)
            .is_err());
        let failed = metal.stats();
        assert!(failed.terminal_error);
        assert_eq!(failed.command_buffers, 3);
        assert_eq!(failed.commits, 3);
        assert_eq!(failed.waits, 3);
        assert!(metal.decode(&input).is_err());
        assert_eq!(metal.stats(), failed);

        metal.reset().unwrap();
        assert_eq!(metal.stats(), TailMlpQ4HeadMetalStatsV2::default());
        assert_eq!(
            metal.decode(&input).unwrap().candidate_token_ids,
            expected.candidate_token_ids
        );
    }
}
