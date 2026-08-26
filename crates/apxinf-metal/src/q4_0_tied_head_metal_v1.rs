//! Isolated Metal Q4_0 tied-head correctness primitive.
//!
//! This explicitly versioned slice is not wired into production decoding. It
//! consumes [`PackedQ4_0RowsV1`](crate::PackedQ4_0RowsV1), computes one full
//! vocabulary score row on Metal, and returns deterministic top-4 candidate
//! token IDs after request-scoped exclusions. The original F32 tied embedding
//! and exact candidate rerank remain outside this primitive.
//!
//! Production decode integration is forbidden until a separate real-
//! checkpoint gate proves exact CPU-Q4/Metal candidate equality at every one
//! of the 128 frozen suppressed-free-run teacher hiddens. Synthetic coverage
//! in this module is necessary but is not that admission gate.

use std::error::Error;
use std::fmt::{Display, Formatter};

use crate::{PackedQ4_0RowsV1, Q4_0_BLOCK_SIZE_V1, Q4_0_PACKED_BYTES_PER_BLOCK_V1};

/// Stable ABI discriminator passed through the isolated bridge.
pub const Q4_0_TIED_HEAD_ABI_VERSION_V1: u32 = 1;
/// Number of candidate token IDs returned by the v1 primitive.
pub const Q4_0_TIED_HEAD_TOP_K_V1: usize = 4;
/// Maximum number of distinct token IDs masked before candidate selection.
pub const Q4_0_TIED_HEAD_MAX_EXCLUDED_TOKENS_V1: usize = 5;

const ROWS_PER_THREADGROUP_V1: usize = 8;
const CANDIDATE_BYTES_V1: usize = 2 * std::mem::size_of::<u32>();
const STATUS_BYTES_V1: usize = std::mem::size_of::<u32>();

/// Fail-closed error from the isolated Metal Q4_0 tied-head v1 contract.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Q4_0TiedHeadErrorV1(String);

impl Q4_0TiedHeadErrorV1 {
    fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl Display for Q4_0TiedHeadErrorV1 {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl Error for Q4_0TiedHeadErrorV1 {}

/// Exact MTLBuffer, transfer, dispatch, and wait contract for v1.
///
/// The score and partial-candidate buffers are persistent private scratch.
/// [`MetalQ4_0TiedHeadV1::scores`] allocates one temporary shared readback
/// buffer; that correctness-only allocation is reported separately and is not
/// part of the candidate-call persistent footprint.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Q4_0TiedHeadBufferLedgerV1 {
    /// Ledger scope; CPU copies, model-body weights, and driver allocations are excluded.
    pub scope: &'static str,
    /// Functionality deliberately outside this isolated primitive.
    pub exclusions: &'static str,
    /// Stable ABI version.
    pub abi_version: u32,
    /// Number of persistent MTLBuffer allocations.
    pub allocated_buffers: usize,
    /// Persistent shared-storage buffers.
    pub shared_buffers: usize,
    /// Persistent private-storage buffers.
    pub private_buffers: usize,
    /// Canonical scale-plus-nibble Q4_0 bytes resident on Metal.
    pub packed_weight_bytes: usize,
    /// Shared F32 hidden-row bytes.
    pub hidden_bytes: usize,
    /// Private full-vocabulary F32 score scratch.
    pub score_scratch_bytes: usize,
    /// Private per-eight-row top-4 candidate scratch.
    pub partial_topk_scratch_bytes: usize,
    /// Shared four-token result bytes.
    pub output_token_bytes: usize,
    /// Shared non-finite-status control bytes.
    pub status_bytes: usize,
    /// Sum of private score and partial-candidate scratch.
    pub persistent_scratch_bytes: usize,
    /// Sum of all persistent MTLBuffer bytes.
    pub total_persistent_bytes: usize,
    /// Temporary shared score readback allocated only by `scores`.
    pub transient_score_readback_bytes_per_score_call: usize,
    /// Hidden payload plus status reset and token poison writes.
    pub host_to_device_bytes_per_candidate_call: usize,
    /// Four token IDs plus the status read.
    pub device_to_host_bytes_per_candidate_call: usize,
    /// Hidden payload plus status reset for a correctness score readback.
    pub host_to_device_bytes_per_score_call: usize,
    /// Full scores plus the status read for a correctness score readback.
    pub device_to_host_bytes_per_score_call: usize,
    /// Candidate-path command buffers.
    pub command_buffers_per_candidate_call: usize,
    /// Candidate-path compute encoders.
    pub compute_encoders_per_candidate_call: usize,
    /// Candidate-path kernel dispatches.
    pub kernel_dispatches_per_candidate_call: usize,
    /// Candidate-path blit encoders.
    pub blit_encoders_per_candidate_call: usize,
    /// Candidate-path commits.
    pub commits_per_candidate_call: usize,
    /// Candidate-path CPU waits.
    pub waits_per_candidate_call: usize,
    /// Correctness score-readback command buffers.
    pub command_buffers_per_score_call: usize,
    /// Correctness score-readback compute encoders.
    pub compute_encoders_per_score_call: usize,
    /// Correctness score-readback kernel dispatches.
    pub kernel_dispatches_per_score_call: usize,
    /// Correctness score-readback blit encoders.
    pub blit_encoders_per_score_call: usize,
    /// Correctness score-readback commits.
    pub commits_per_score_call: usize,
    /// Correctness score-readback CPU waits.
    pub waits_per_score_call: usize,
}

impl Q4_0TiedHeadBufferLedgerV1 {
    /// Close the v1 ledger from a row-major `[vocab, hidden]` shape without
    /// allocating the packed matrix or creating a Metal device.
    pub fn from_dimensions(rows: usize, columns: usize) -> Result<Self, Q4_0TiedHeadErrorV1> {
        validate_dimensions(rows, columns)?;
        let blocks_per_row = columns / Q4_0_BLOCK_SIZE_V1;
        let packed_weight_bytes = rows
            .checked_mul(blocks_per_row)
            .and_then(|blocks| blocks.checked_mul(Q4_0_PACKED_BYTES_PER_BLOCK_V1))
            .ok_or_else(|| {
                Q4_0TiedHeadErrorV1::new("Metal Q4_0 tied-head v1 packed-byte ledger overflow")
            })?;
        let hidden_bytes = columns
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| {
                Q4_0TiedHeadErrorV1::new("Metal Q4_0 tied-head v1 hidden-byte ledger overflow")
            })?;
        let score_scratch_bytes =
            rows.checked_mul(std::mem::size_of::<f32>())
                .ok_or_else(|| {
                    Q4_0TiedHeadErrorV1::new("Metal Q4_0 tied-head v1 score-byte ledger overflow")
                })?;
        let partial_count = rows
            .checked_add(ROWS_PER_THREADGROUP_V1 - 1)
            .map(|count| count / ROWS_PER_THREADGROUP_V1)
            .ok_or_else(|| {
                Q4_0TiedHeadErrorV1::new("Metal Q4_0 tied-head v1 partial-count ledger overflow")
            })?;
        let partial_topk_scratch_bytes = partial_count
            .checked_mul(Q4_0_TIED_HEAD_TOP_K_V1)
            .and_then(|count| count.checked_mul(CANDIDATE_BYTES_V1))
            .ok_or_else(|| {
                Q4_0TiedHeadErrorV1::new("Metal Q4_0 tied-head v1 partial-byte ledger overflow")
            })?;
        let output_token_bytes = Q4_0_TIED_HEAD_TOP_K_V1 * std::mem::size_of::<u32>();
        let persistent_scratch_bytes = score_scratch_bytes
            .checked_add(partial_topk_scratch_bytes)
            .ok_or_else(|| {
                Q4_0TiedHeadErrorV1::new("Metal Q4_0 tied-head v1 scratch ledger overflow")
            })?;
        let total_persistent_bytes = [
            packed_weight_bytes,
            hidden_bytes,
            score_scratch_bytes,
            partial_topk_scratch_bytes,
            output_token_bytes,
            STATUS_BYTES_V1,
        ]
        .into_iter()
        .try_fold(0usize, |total, bytes| total.checked_add(bytes))
        .ok_or_else(|| Q4_0TiedHeadErrorV1::new("Metal Q4_0 tied-head v1 total ledger overflow"))?;
        let host_to_device_bytes_per_candidate_call = hidden_bytes
            .checked_add(output_token_bytes)
            .and_then(|bytes| bytes.checked_add(STATUS_BYTES_V1))
            .ok_or_else(|| {
                Q4_0TiedHeadErrorV1::new("Metal Q4_0 tied-head v1 candidate H2D ledger overflow")
            })?;
        let device_to_host_bytes_per_candidate_call = output_token_bytes
            .checked_add(STATUS_BYTES_V1)
            .ok_or_else(|| {
                Q4_0TiedHeadErrorV1::new("Metal Q4_0 tied-head v1 candidate D2H ledger overflow")
            })?;
        let host_to_device_bytes_per_score_call =
            hidden_bytes.checked_add(STATUS_BYTES_V1).ok_or_else(|| {
                Q4_0TiedHeadErrorV1::new("Metal Q4_0 tied-head v1 score H2D ledger overflow")
            })?;
        let device_to_host_bytes_per_score_call = score_scratch_bytes
            .checked_add(STATUS_BYTES_V1)
            .ok_or_else(|| {
                Q4_0TiedHeadErrorV1::new("Metal Q4_0 tied-head v1 score D2H ledger overflow")
            })?;

        Ok(Self {
            scope: "persistent-mtlbuffer-only-plus-explicit-per-call-transfers-v1",
            exclusions: "CPU PackedQ4_0RowsV1 storage and canonical-byte staging, temporary score-readback buffer, F32 tied embedding/exact rerank, Metal library/pipelines/queue/commands, model body, and KV cache",
            abi_version: Q4_0_TIED_HEAD_ABI_VERSION_V1,
            allocated_buffers: 6,
            shared_buffers: 4,
            private_buffers: 2,
            packed_weight_bytes,
            hidden_bytes,
            score_scratch_bytes,
            partial_topk_scratch_bytes,
            output_token_bytes,
            status_bytes: STATUS_BYTES_V1,
            persistent_scratch_bytes,
            total_persistent_bytes,
            transient_score_readback_bytes_per_score_call: score_scratch_bytes,
            host_to_device_bytes_per_candidate_call,
            device_to_host_bytes_per_candidate_call,
            host_to_device_bytes_per_score_call,
            device_to_host_bytes_per_score_call,
            command_buffers_per_candidate_call: 1,
            compute_encoders_per_candidate_call: 2,
            kernel_dispatches_per_candidate_call: 2,
            blit_encoders_per_candidate_call: 0,
            commits_per_candidate_call: 1,
            waits_per_candidate_call: 1,
            command_buffers_per_score_call: 1,
            compute_encoders_per_score_call: 1,
            kernel_dispatches_per_score_call: 1,
            blit_encoders_per_score_call: 1,
            commits_per_score_call: 1,
            waits_per_score_call: 1,
        })
    }
}

/// Persistent Metal resources for the isolated Q4_0 tied-head v1 slice.
///
/// This type is deliberately absent from `GeneralQwen35` and all production
/// decode dispatch. Construction uploads the canonical Q4_0 block bytes once.
pub struct MetalQ4_0TiedHeadV1 {
    inner: platform::Handle,
    rows: usize,
    columns: usize,
    ledger: Q4_0TiedHeadBufferLedgerV1,
}

impl MetalQ4_0TiedHeadV1 {
    /// Create the isolated Metal handle from the exact CPU Q4_0 oracle rows.
    pub fn from_packed(weights: &PackedQ4_0RowsV1) -> Result<Self, Q4_0TiedHeadErrorV1> {
        let ledger =
            Q4_0TiedHeadBufferLedgerV1::from_dimensions(weights.rows(), weights.columns())?;
        let canonical_bytes = weights.canonical_bytes_le();
        if canonical_bytes.len() != ledger.packed_weight_bytes {
            return Err(Q4_0TiedHeadErrorV1::new(format!(
                "Metal Q4_0 tied-head v1 canonical stream has {} bytes, expected {}",
                canonical_bytes.len(),
                ledger.packed_weight_bytes
            )));
        }
        let inner = platform::Handle::new(&canonical_bytes, weights.rows(), weights.columns())?;
        Ok(Self {
            inner,
            rows: weights.rows(),
            columns: weights.columns(),
            ledger,
        })
    }

    /// Vocabulary row count fixed at construction.
    pub const fn rows(&self) -> usize {
        self.rows
    }

    /// Hidden width fixed at construction.
    pub const fn columns(&self) -> usize {
        self.columns
    }

    /// Return the exact persistent-buffer and call-transaction ledger.
    pub const fn buffer_ledger(&self) -> Q4_0TiedHeadBufferLedgerV1 {
        self.ledger
    }

    /// Correctness-only readback of all Q4_0 scores for one hidden row.
    ///
    /// The candidate path does not read these logits back. This method exists
    /// so synthetic and checkpoint gates can compare the Metal score surface
    /// with [`PackedQ4_0RowsV1::scores`].
    pub fn scores(&mut self, hidden: &[f32]) -> Result<Vec<f32>, Q4_0TiedHeadErrorV1> {
        validate_hidden(hidden, self.columns)?;
        let scores = self.inner.scores(hidden, self.rows)?;
        if scores.len() != self.rows {
            return Err(Q4_0TiedHeadErrorV1::new(format!(
                "Metal Q4_0 tied-head v1 returned {} scores, expected {}",
                scores.len(),
                self.rows
            )));
        }
        if let Some(token) = scores.iter().position(|score| !score.is_finite()) {
            return Err(Q4_0TiedHeadErrorV1::new(format!(
                "Metal Q4_0 tied-head v1 returned a non-finite score for token {token}"
            )));
        }
        Ok(scores)
    }

    /// Select the deterministic four highest quantized scores.
    pub fn topk4(
        &mut self,
        hidden: &[f32],
    ) -> Result<[u32; Q4_0_TIED_HEAD_TOP_K_V1], Q4_0TiedHeadErrorV1> {
        self.topk4_excluding(hidden, &[])
    }

    /// Select deterministic top-4 candidates after removing at most five
    /// distinct vocabulary rows before either reduction stage.
    pub fn topk4_excluding(
        &mut self,
        hidden: &[f32],
        excluded_tokens: &[u32],
    ) -> Result<[u32; Q4_0_TIED_HEAD_TOP_K_V1], Q4_0TiedHeadErrorV1> {
        // Both validations intentionally precede the bridge call, which is the
        // first operation able to mutate persistent buffers or dispatch work.
        validate_hidden(hidden, self.columns)?;
        validate_exclusions(excluded_tokens, self.rows)?;
        let tokens = self.inner.topk4_excluding(hidden, excluded_tokens)?;
        validate_returned_candidates(&tokens, excluded_tokens, self.rows)?;
        Ok(tokens)
    }
}

fn validate_dimensions(rows: usize, columns: usize) -> Result<(), Q4_0TiedHeadErrorV1> {
    if rows < Q4_0_TIED_HEAD_TOP_K_V1 {
        return Err(Q4_0TiedHeadErrorV1::new(format!(
            "Metal Q4_0 tied-head v1 requires at least {Q4_0_TIED_HEAD_TOP_K_V1} vocabulary rows"
        )));
    }
    if columns == 0 || columns % Q4_0_BLOCK_SIZE_V1 != 0 {
        return Err(Q4_0TiedHeadErrorV1::new(format!(
            "Metal Q4_0 tied-head v1 columns must be non-zero and divisible by {Q4_0_BLOCK_SIZE_V1}, got {columns}"
        )));
    }
    if rows > u32::MAX as usize || columns > u32::MAX as usize {
        return Err(Q4_0TiedHeadErrorV1::new(
            "Metal Q4_0 tied-head v1 dimensions exceed the u32 ABI",
        ));
    }
    Ok(())
}

fn validate_hidden(hidden: &[f32], columns: usize) -> Result<(), Q4_0TiedHeadErrorV1> {
    if hidden.len() != columns {
        return Err(Q4_0TiedHeadErrorV1::new(format!(
            "Metal Q4_0 tied-head v1 hidden row has {} elements, expected {columns}",
            hidden.len()
        )));
    }
    if let Some(index) = hidden.iter().position(|value| !value.is_finite()) {
        return Err(Q4_0TiedHeadErrorV1::new(format!(
            "Metal Q4_0 tied-head v1 hidden row contains a non-finite value at element {index}"
        )));
    }
    Ok(())
}

fn validate_exclusions(excluded_tokens: &[u32], rows: usize) -> Result<(), Q4_0TiedHeadErrorV1> {
    if excluded_tokens.len() > Q4_0_TIED_HEAD_MAX_EXCLUDED_TOKENS_V1 {
        return Err(Q4_0TiedHeadErrorV1::new(format!(
            "Metal Q4_0 tied-head v1 accepts at most {Q4_0_TIED_HEAD_MAX_EXCLUDED_TOKENS_V1} excluded tokens"
        )));
    }
    for (index, &token) in excluded_tokens.iter().enumerate() {
        if token as usize >= rows {
            return Err(Q4_0TiedHeadErrorV1::new(format!(
                "Metal Q4_0 tied-head v1 exclusion token {token} is outside vocabulary {rows}"
            )));
        }
        if excluded_tokens[..index].contains(&token) {
            return Err(Q4_0TiedHeadErrorV1::new(format!(
                "Metal Q4_0 tied-head v1 exclusion token {token} is duplicated"
            )));
        }
    }
    if rows.saturating_sub(excluded_tokens.len()) < Q4_0_TIED_HEAD_TOP_K_V1 {
        return Err(Q4_0TiedHeadErrorV1::new(format!(
            "Metal Q4_0 tied-head v1 exclusions leave fewer than {Q4_0_TIED_HEAD_TOP_K_V1} vocabulary rows"
        )));
    }
    Ok(())
}

fn validate_returned_candidates(
    tokens: &[u32; Q4_0_TIED_HEAD_TOP_K_V1],
    excluded_tokens: &[u32],
    rows: usize,
) -> Result<(), Q4_0TiedHeadErrorV1> {
    for (index, &token) in tokens.iter().enumerate() {
        if token as usize >= rows {
            return Err(Q4_0TiedHeadErrorV1::new(format!(
                "Metal Q4_0 tied-head v1 returned candidate {index} token {token} outside vocabulary {rows}"
            )));
        }
        if tokens[..index].contains(&token) {
            return Err(Q4_0TiedHeadErrorV1::new(format!(
                "Metal Q4_0 tied-head v1 returned duplicate candidate token {token}"
            )));
        }
        if excluded_tokens.contains(&token) {
            return Err(Q4_0TiedHeadErrorV1::new(format!(
                "Metal Q4_0 tied-head v1 returned excluded candidate token {token}"
            )));
        }
    }
    Ok(())
}

#[cfg(target_os = "macos")]
mod platform {
    use super::{Q4_0TiedHeadErrorV1, Q4_0_TIED_HEAD_ABI_VERSION_V1, Q4_0_TIED_HEAD_TOP_K_V1};
    use std::ffi::{c_char, c_int, c_void, CStr};
    use std::ptr::NonNull;

    const ERROR_CAPACITY: usize = 1024;

    extern "C" {
        fn apxinf_metal_q4_0_tied_head_v1_create(
            packed_bytes: *const u8,
            packed_byte_count: usize,
            rows: u32,
            columns: u32,
            abi_version: u32,
            output: *mut *mut c_void,
            error: *mut c_char,
            error_capacity: usize,
        ) -> c_int;
        fn apxinf_metal_q4_0_tied_head_v1_scores(
            handle: *mut c_void,
            hidden: *const f32,
            hidden_count: u32,
            output_scores: *mut f32,
            output_count: u32,
            error: *mut c_char,
            error_capacity: usize,
        ) -> c_int;
        fn apxinf_metal_q4_0_tied_head_v1_topk4_excluding(
            handle: *mut c_void,
            hidden: *const f32,
            hidden_count: u32,
            excluded_tokens: *const u32,
            excluded_count: u32,
            output_tokens: *mut u32,
            error: *mut c_char,
            error_capacity: usize,
        ) -> c_int;
        fn apxinf_metal_q4_0_tied_head_v1_destroy(handle: *mut c_void);
    }

    pub(super) struct Handle(NonNull<c_void>);

    impl Handle {
        pub(super) fn new(
            packed_bytes: &[u8],
            rows: usize,
            columns: usize,
        ) -> Result<Self, Q4_0TiedHeadErrorV1> {
            let mut output = std::ptr::null_mut();
            let mut error = [0 as c_char; ERROR_CAPACITY];
            let status = unsafe {
                apxinf_metal_q4_0_tied_head_v1_create(
                    packed_bytes.as_ptr(),
                    packed_bytes.len(),
                    rows as u32,
                    columns as u32,
                    Q4_0_TIED_HEAD_ABI_VERSION_V1,
                    &mut output,
                    error.as_mut_ptr(),
                    error.len(),
                )
            };
            if status != 0 {
                return Err(bridge_error("create Metal Q4_0 tied-head v1", &error));
            }
            NonNull::new(output).map(Self).ok_or_else(|| {
                Q4_0TiedHeadErrorV1::new("create Metal Q4_0 tied-head v1 returned a null handle")
            })
        }

        pub(super) fn scores(
            &mut self,
            hidden: &[f32],
            rows: usize,
        ) -> Result<Vec<f32>, Q4_0TiedHeadErrorV1> {
            let mut output = vec![0.0f32; rows];
            let mut error = [0 as c_char; ERROR_CAPACITY];
            let status = unsafe {
                apxinf_metal_q4_0_tied_head_v1_scores(
                    self.0.as_ptr(),
                    hidden.as_ptr(),
                    hidden.len() as u32,
                    output.as_mut_ptr(),
                    output.len() as u32,
                    error.as_mut_ptr(),
                    error.len(),
                )
            };
            if status != 0 {
                return Err(bridge_error("read Metal Q4_0 tied-head v1 scores", &error));
            }
            Ok(output)
        }

        pub(super) fn topk4_excluding(
            &mut self,
            hidden: &[f32],
            excluded_tokens: &[u32],
        ) -> Result<[u32; Q4_0_TIED_HEAD_TOP_K_V1], Q4_0TiedHeadErrorV1> {
            let mut output = [0u32; Q4_0_TIED_HEAD_TOP_K_V1];
            let mut error = [0 as c_char; ERROR_CAPACITY];
            let status = unsafe {
                apxinf_metal_q4_0_tied_head_v1_topk4_excluding(
                    self.0.as_ptr(),
                    hidden.as_ptr(),
                    hidden.len() as u32,
                    excluded_tokens.as_ptr(),
                    excluded_tokens.len() as u32,
                    output.as_mut_ptr(),
                    error.as_mut_ptr(),
                    error.len(),
                )
            };
            if status != 0 {
                return Err(bridge_error("run Metal Q4_0 tied-head v1 top-4", &error));
            }
            Ok(output)
        }
    }

    impl Drop for Handle {
        fn drop(&mut self) {
            unsafe { apxinf_metal_q4_0_tied_head_v1_destroy(self.0.as_ptr()) };
        }
    }

    fn bridge_error(context: &str, buffer: &[c_char]) -> Q4_0TiedHeadErrorV1 {
        let detail = unsafe { CStr::from_ptr(buffer.as_ptr()) }
            .to_string_lossy()
            .into_owned();
        if detail.is_empty() {
            Q4_0TiedHeadErrorV1::new(context)
        } else {
            Q4_0TiedHeadErrorV1::new(format!("{context}: {detail}"))
        }
    }
}

#[cfg(not(target_os = "macos"))]
mod platform {
    use super::{Q4_0TiedHeadErrorV1, Q4_0_TIED_HEAD_TOP_K_V1};

    pub(super) struct Handle;

    impl Handle {
        pub(super) fn new(
            _packed_bytes: &[u8],
            _rows: usize,
            _columns: usize,
        ) -> Result<Self, Q4_0TiedHeadErrorV1> {
            Err(Q4_0TiedHeadErrorV1::new(
                "Metal Q4_0 tied-head v1 requires macOS",
            ))
        }

        pub(super) fn scores(
            &mut self,
            _hidden: &[f32],
            _rows: usize,
        ) -> Result<Vec<f32>, Q4_0TiedHeadErrorV1> {
            Err(Q4_0TiedHeadErrorV1::new(
                "Metal Q4_0 tied-head v1 requires macOS",
            ))
        }

        pub(super) fn topk4_excluding(
            &mut self,
            _hidden: &[f32],
            _excluded_tokens: &[u32],
        ) -> Result<[u32; Q4_0_TIED_HEAD_TOP_K_V1], Q4_0TiedHeadErrorV1> {
            Err(Q4_0TiedHeadErrorV1::new(
                "Metal Q4_0 tied-head v1 requires macOS",
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture(rows: usize, columns: usize) -> (Vec<f32>, Vec<f32>) {
        let weights = (0..rows * columns)
            .map(|index| {
                let value = ((index * 43 + index / 13 + 19) % 257) as f32 - 128.0;
                value * 0.0013
            })
            .collect::<Vec<_>>();
        let hidden = (0..columns)
            .map(|index| (((index * 29 + 7) % 113) as f32 - 56.0) * 0.0041)
            .collect::<Vec<_>>();
        (weights, hidden)
    }

    #[test]
    fn qwen35_ledger_closes_to_the_official_tied_head_shape() {
        let ledger = Q4_0TiedHeadBufferLedgerV1::from_dimensions(248_320, 1_024).unwrap();

        assert_eq!(ledger.abi_version, 1);
        assert_eq!(ledger.allocated_buffers, 6);
        assert_eq!(ledger.shared_buffers, 4);
        assert_eq!(ledger.private_buffers, 2);
        assert_eq!(ledger.packed_weight_bytes, 143_032_320);
        assert_eq!(ledger.hidden_bytes, 4_096);
        assert_eq!(ledger.score_scratch_bytes, 993_280);
        assert_eq!(ledger.partial_topk_scratch_bytes, 993_280);
        assert_eq!(ledger.output_token_bytes, 16);
        assert_eq!(ledger.status_bytes, 4);
        assert_eq!(ledger.persistent_scratch_bytes, 1_986_560);
        assert_eq!(ledger.total_persistent_bytes, 145_022_996);
        assert_eq!(
            ledger.transient_score_readback_bytes_per_score_call,
            993_280
        );
        assert_eq!(ledger.host_to_device_bytes_per_candidate_call, 4_116);
        assert_eq!(ledger.device_to_host_bytes_per_candidate_call, 20);
        assert_eq!(ledger.host_to_device_bytes_per_score_call, 4_100);
        assert_eq!(ledger.device_to_host_bytes_per_score_call, 993_284);
        assert_eq!(ledger.kernel_dispatches_per_candidate_call, 2);
        assert_eq!(ledger.waits_per_candidate_call, 1);
        assert_eq!(ledger.kernel_dispatches_per_score_call, 1);
        assert_eq!(ledger.blit_encoders_per_score_call, 1);
        assert_eq!(ledger.waits_per_score_call, 1);
    }

    #[test]
    fn ledger_rejects_shapes_outside_the_versioned_abi() {
        assert!(Q4_0TiedHeadBufferLedgerV1::from_dimensions(3, 32).is_err());
        assert!(Q4_0TiedHeadBufferLedgerV1::from_dimensions(4, 0).is_err());
        assert!(Q4_0TiedHeadBufferLedgerV1::from_dimensions(4, 33).is_err());
    }

    #[test]
    fn shader_and_bridge_are_isolated_and_discoverable() {
        let shader = include_str!("metal_q4_0_tied_head_v1.metal");
        let bridge = include_str!("metal_q4_0_tied_head_v1_bridge.mm");

        assert!(shader.contains("kernel void q4_0_tied_head_rows_v1("));
        assert!(shader.contains("kernel void q4_0_tied_head_final_topk4_v1("));
        assert!(shader.contains("q4_0_token_is_excluded_v1(row, params)"));
        assert!(bridge.contains("#include \"metal_q4_0_tied_head_v1_source.inc\""));
        assert!(bridge.contains("apxinf_metal_q4_0_tied_head_v1_create("));
        assert!(bridge.contains("apxinf_metal_q4_0_tied_head_v1_topk4_excluding("));
        assert!(!bridge.contains("kernel void q4_0_tied_head_rows_v1("));
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_scores_and_top4_match_the_q4_0_cpu_oracle() {
        let rows = 257;
        let columns = 128;
        let (mut weights, hidden) = fixture(rows, columns);
        for (column, value) in hidden.iter().enumerate() {
            weights[173 * columns + column] += value * 4.0;
        }
        let packed = PackedQ4_0RowsV1::pack_f32(&weights, rows, columns).unwrap();
        let expected_scores = packed.scores(&hidden).unwrap();
        let expected_tokens = packed.topk_excluding(&hidden, 4, &[]).unwrap();
        let mut head = MetalQ4_0TiedHeadV1::from_packed(&packed).unwrap();
        let actual_scores = head.scores(&hidden).unwrap();

        for (row, (&actual, &expected)) in actual_scores.iter().zip(&expected_scores).enumerate() {
            let tolerance = 4.0e-5f32.max(expected.abs() * 2.0e-5);
            assert!(
                (actual - expected).abs() <= tolerance,
                "row {row}: Metal={actual}, CPU Q4_0={expected}, tolerance={tolerance}"
            );
        }
        assert_eq!(head.topk4(&hidden).unwrap().as_slice(), expected_tokens);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_exclusions_match_cpu_and_are_checked_before_reuse() {
        let rows = 17;
        let columns = 64;
        let hidden = vec![1.0f32; columns];
        let mut weights = vec![0.0f32; rows * columns];
        for row in 0..rows {
            weights[row * columns..(row + 1) * columns].fill((rows - row) as f32 / rows as f32);
        }
        let packed = PackedQ4_0RowsV1::pack_f32(&weights, rows, columns).unwrap();
        let excluded = [0, 1, 2, 3, 4];
        let expected = packed.topk_excluding(&hidden, 4, &excluded).unwrap();
        let mut head = MetalQ4_0TiedHeadV1::from_packed(&packed).unwrap();

        assert!(head.topk4_excluding(&hidden, &[0, 1, 2, 3, 4, 5]).is_err());
        assert!(head.topk4_excluding(&hidden, &[2, 2]).is_err());
        assert!(head.topk4_excluding(&hidden, &[17]).is_err());
        assert!(head.topk4_excluding(&hidden[..63], &[]).is_err());
        let mut non_finite = hidden.clone();
        non_finite[9] = f32::NAN;
        assert!(head.topk4_excluding(&non_finite, &[]).is_err());

        assert_eq!(
            head.topk4_excluding(&hidden, &excluded).unwrap().as_slice(),
            expected
        );
        assert_eq!(expected, [5, 6, 7, 8]);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_exact_ties_choose_the_lowest_allowed_tokens() {
        let rows = 19;
        let columns = 64;
        let weights = vec![0.0f32; rows * columns];
        let hidden = vec![0.25f32; columns];
        let packed = PackedQ4_0RowsV1::pack_f32(&weights, rows, columns).unwrap();
        let mut head = MetalQ4_0TiedHeadV1::from_packed(&packed).unwrap();

        assert_eq!(head.topk4(&hidden).unwrap(), [0, 1, 2, 3]);
        assert_eq!(
            head.topk4_excluding(&hidden, &[0, 2, 4, 6, 8]).unwrap(),
            [1, 3, 5, 7]
        );
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn final_reducer_spans_multiple_stride_rounds_and_partial_groups() {
        // 4,097 rows produce 513 partial lists and 2,052 candidates. That is
        // more than eight rounds of the final kernel's `index += 256` loop,
        // while the non-multiple-of-eight tail also exercises invalid rows.
        let rows = 4_097;
        let columns = 64;
        let hidden = vec![1.0f32; columns];
        let mut weights = vec![0.0f32; rows * columns];
        let ranked_rows = [
            (3usize, 0.95f32),
            (1_025, 0.90),
            (2_048, 0.85),
            (3_073, 0.80),
            (4_096, 0.75),
            (511, 0.70),
            (1_537, 0.65),
            (2_561, 0.60),
            (3_585, 0.55),
        ];
        for &(row, value) in &ranked_rows {
            weights[row * columns..(row + 1) * columns].fill(value);
        }
        let packed = PackedQ4_0RowsV1::pack_f32(&weights, rows, columns).unwrap();
        let excluded = [3, 1_025, 2_048, 3_073, 4_096];
        let expected = packed.topk_excluding(&hidden, 4, &[]).unwrap();
        let expected_excluding = packed.topk_excluding(&hidden, 4, &excluded).unwrap();
        let mut head = MetalQ4_0TiedHeadV1::from_packed(&packed).unwrap();

        assert_eq!(expected, [3, 1_025, 2_048, 3_073]);
        assert_eq!(expected_excluding, [511, 1_537, 2_561, 3_585]);
        assert_eq!(head.topk4(&hidden).unwrap().as_slice(), expected);
        assert_eq!(
            head.topk4_excluding(&hidden, &excluded).unwrap().as_slice(),
            expected_excluding
        );
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn non_finite_score_poison_fails_closed_and_status_resets() {
        let rows = 9;
        let columns = 64;
        let weights = vec![1.0f32; rows * columns];
        let packed = PackedQ4_0RowsV1::pack_f32(&weights, rows, columns).unwrap();
        let poison = vec![f32::MAX; columns];
        let valid = vec![1.0f32; columns];
        assert!(packed.scores(&poison).is_err());
        let expected = packed.topk_excluding(&valid, 4, &[]).unwrap();
        let mut head = MetalQ4_0TiedHeadV1::from_packed(&packed).unwrap();

        assert!(head
            .topk4(&poison)
            .unwrap_err()
            .to_string()
            .contains("non-finite"));
        assert_eq!(head.topk4(&valid).unwrap().as_slice(), expected);
        assert!(head
            .scores(&poison)
            .unwrap_err()
            .to_string()
            .contains("non-finite"));
        assert_eq!(head.scores(&valid).unwrap(), packed.scores(&valid).unwrap());
    }
}
