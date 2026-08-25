use crate::MetalW8Error;

pub const QWEN35_RESIDUAL_RMS_HIDDEN_SIZE_V1: usize = 1024;
pub const QWEN35_RESIDUAL_RMS_SEAMS_PER_DECODE_V1: usize = 18;

/// Explicit selector for the bounded post-attention residual→RMS mechanism
/// screen. Existing production constructors remain bound to
/// [`Self::LegacySeparate`].
#[repr(u32)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ResidualRmsProfileV1 {
    #[default]
    LegacySeparate = 0,
    FusedExact = 1,
}

impl ResidualRmsProfileV1 {
    pub const fn selector(self) -> u32 {
        self as u32
    }

    pub const fn expected_primary_function_name(self) -> &'static str {
        match self {
            Self::LegacySeparate => "linear_layer_residual_add",
            Self::FusedExact => "linear_layer_residual_rms_norm_fused_exact_v1",
        }
    }

    pub const fn expected_secondary_function_name(self) -> &'static str {
        match self {
            Self::LegacySeparate => "linear_layer_rms_norm",
            Self::FusedExact => "",
        }
    }

    pub const fn kernel_dispatches_per_run(self) -> u32 {
        match self {
            Self::LegacySeparate => 36,
            Self::FusedExact => 18,
        }
    }

    pub const fn pair_local_raw_barriers_per_run(self) -> u32 {
        match self {
            Self::LegacySeparate => 18,
            Self::FusedExact => 0,
        }
    }

    pub const fn common_consumer_barriers_per_run(self) -> u32 {
        QWEN35_RESIDUAL_RMS_SEAMS_PER_DECODE_V1 as u32
    }

    pub const fn explicit_buffer_barriers_per_run(self) -> u32 {
        self.pair_local_raw_barriers_per_run() + self.common_consumer_barriers_per_run()
    }

    const fn from_selector(selector: u32) -> Option<Self> {
        match selector {
            0 => Some(Self::LegacySeparate),
            1 => Some(Self::FusedExact),
            _ => None,
        }
    }
}

impl TryFrom<u32> for ResidualRmsProfileV1 {
    type Error = MetalW8Error;

    fn try_from(selector: u32) -> Result<Self, Self::Error> {
        Self::from_selector(selector).ok_or_else(|| {
            MetalW8Error::new(format!(
                "Metal residual-RMS profile {selector} is invalid; expected 0 or 1"
            ))
        })
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ResidualRmsCount18SnapshotV1 {
    pub materialized_residual_rows: Vec<f32>,
    pub normalized_rows: Vec<f32>,
}

/// Live function identity and actual call-site topology for one arm of the
/// additive count-18 residual→RMS primitive. Internal barrier and source-read
/// counts are source-derived; `last_observed_*` values are incremented at the
/// bridge call sites and published only after successful GPU completion.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResidualRmsCount18RuntimeReceiptV1 {
    pub requested_profile: ResidualRmsProfileV1,
    pub observed_profile: ResidualRmsProfileV1,
    pub requested_primary_function_name: String,
    pub observed_primary_function_name: String,
    pub requested_secondary_function_name: String,
    pub observed_secondary_function_name: String,
    pub hidden_size: u32,
    pub seams_per_run: u32,
    pub threads_per_threadgroup: u32,
    pub simdgroups_per_threadgroup: u32,
    pub primary_pipeline_max_total_threads_per_threadgroup: u32,
    pub primary_pipeline_thread_execution_width: u32,
    pub primary_static_threadgroup_memory_bytes: u32,
    pub secondary_pipeline_max_total_threads_per_threadgroup: u32,
    pub secondary_pipeline_thread_execution_width: u32,
    pub secondary_static_threadgroup_memory_bytes: u32,
    pub dynamic_threadgroup_memory_bytes: u32,
    pub internal_threadgroup_barriers_per_seam: u32,
    pub internal_threadgroup_barriers_per_run: u32,
    pub command_buffers_per_run: u32,
    pub compute_encoders_per_run: u32,
    pub kernel_dispatches_per_run: u32,
    pub explicit_buffer_barriers_per_run: u32,
    pub pair_local_raw_barriers_per_run: u32,
    pub common_consumer_barriers_per_run: u32,
    pub commits_per_run: u32,
    pub waits_per_run: u32,
    pub host_to_device_bytes_per_run: u64,
    pub device_to_host_bytes_per_run: u64,
    pub successful_runs: u64,
    pub last_observed_command_buffers: u32,
    pub last_observed_compute_encoders: u32,
    pub last_observed_kernel_dispatches: u32,
    pub last_observed_explicit_buffer_barriers: u32,
    pub last_observed_pair_local_raw_barriers: u32,
    pub last_observed_common_consumer_barriers: u32,
    pub last_observed_commits: u32,
    pub last_observed_waits: u32,
}

/// Same-binary H=1024 aggregate primitive. A run submits the 18 dependent
/// same-encoder seams represented in the current Qwen3.5-0.8B decode path.
/// Fixture staging and full trace snapshots are separate calls so the timed
/// run contains no explicit bridge memcpy.
pub struct MetalResidualRmsCount18PrimitiveV1 {
    inner: platform::Handle,
}

impl MetalResidualRmsCount18PrimitiveV1 {
    pub fn new(weights: &[f32], rms_norm_eps: f32) -> Result<Self, MetalW8Error> {
        validate_trace("weights", weights)?;
        if !rms_norm_eps.is_finite() || rms_norm_eps < 0.0 {
            return Err(MetalW8Error::new(
                "Metal residual-RMS epsilon must be finite and non-negative",
            ));
        }
        Ok(Self {
            inner: platform::Handle::new(weights, rms_norm_eps)?,
        })
    }

    pub fn stage_fixture(&mut self, seed: &[f32], updates: &[f32]) -> Result<(), MetalW8Error> {
        validate_row("seed", seed)?;
        validate_trace("updates", updates)?;
        self.inner.stage_fixture(seed, updates)
    }

    /// Fill both shared output traces with NaNs outside the timed path. The
    /// correctness gate calls this before every arm so an unwritten element
    /// cannot inherit a value from the preceding run.
    pub fn poison_traces_for_correctness(&mut self) -> Result<(), MetalW8Error> {
        self.inner.poison_traces_for_correctness()
    }

    pub fn run(&mut self, profile: ResidualRmsProfileV1) -> Result<(), MetalW8Error> {
        self.inner.run(profile)
    }

    pub fn snapshot(&self) -> Result<ResidualRmsCount18SnapshotV1, MetalW8Error> {
        self.inner.snapshot()
    }

    pub fn runtime_receipt(
        &self,
        profile: ResidualRmsProfileV1,
    ) -> Result<ResidualRmsCount18RuntimeReceiptV1, MetalW8Error> {
        self.inner.runtime_receipt(profile)
    }

    /// Exercise invalid raw selectors and prove that the rejected calls leave
    /// both arm receipts and the most recent correctness snapshot unchanged.
    pub fn verify_invalid_raw_selector_fail_closed(&self) -> Result<(), MetalW8Error> {
        if self
            .inner
            .invalid_raw_selectors_are_rejected_without_mutation()
        {
            Ok(())
        } else {
            Err(MetalW8Error::new(
                "invalid raw Metal residual-RMS selector mutated observable state",
            ))
        }
    }
}

fn validate_row(label: &str, values: &[f32]) -> Result<(), MetalW8Error> {
    if values.len() != QWEN35_RESIDUAL_RMS_HIDDEN_SIZE_V1 {
        return Err(MetalW8Error::new(format!(
            "Metal residual-RMS {label} has {} elements, expected {}",
            values.len(),
            QWEN35_RESIDUAL_RMS_HIDDEN_SIZE_V1
        )));
    }
    if let Some(index) = values.iter().position(|value| !value.is_finite()) {
        return Err(MetalW8Error::new(format!(
            "Metal residual-RMS {label} contains a non-finite value at element {index}"
        )));
    }
    Ok(())
}

fn validate_trace(label: &str, values: &[f32]) -> Result<(), MetalW8Error> {
    let expected = QWEN35_RESIDUAL_RMS_HIDDEN_SIZE_V1 * QWEN35_RESIDUAL_RMS_SEAMS_PER_DECODE_V1;
    if values.len() != expected {
        return Err(MetalW8Error::new(format!(
            "Metal residual-RMS {label} has {} elements, expected {expected}",
            values.len()
        )));
    }
    if let Some(index) = values.iter().position(|value| !value.is_finite()) {
        return Err(MetalW8Error::new(format!(
            "Metal residual-RMS {label} contains a non-finite value at element {index}"
        )));
    }
    Ok(())
}

#[cfg(target_os = "macos")]
mod platform {
    use super::{
        MetalW8Error, ResidualRmsCount18RuntimeReceiptV1, ResidualRmsCount18SnapshotV1,
        ResidualRmsProfileV1, QWEN35_RESIDUAL_RMS_HIDDEN_SIZE_V1,
        QWEN35_RESIDUAL_RMS_SEAMS_PER_DECODE_V1,
    };
    use std::ffi::{c_char, c_int, c_void, CStr};
    use std::ptr::NonNull;

    const ERROR_CAPACITY: usize = 1024;
    const FUNCTION_NAME_CAPACITY: usize = 64;

    #[repr(C)]
    #[derive(Clone, Copy)]
    struct RawRuntimeReceiptV1 {
        requested_profile: u32,
        observed_profile: u32,
        hidden_size: u32,
        seams_per_run: u32,
        threads_per_threadgroup: u32,
        simdgroups_per_threadgroup: u32,
        primary_pipeline_max_total_threads_per_threadgroup: u32,
        primary_pipeline_thread_execution_width: u32,
        primary_static_threadgroup_memory_bytes: u32,
        secondary_pipeline_max_total_threads_per_threadgroup: u32,
        secondary_pipeline_thread_execution_width: u32,
        secondary_static_threadgroup_memory_bytes: u32,
        dynamic_threadgroup_memory_bytes: u32,
        internal_threadgroup_barriers_per_seam: u32,
        internal_threadgroup_barriers_per_run: u32,
        command_buffers_per_run: u32,
        compute_encoders_per_run: u32,
        kernel_dispatches_per_run: u32,
        explicit_buffer_barriers_per_run: u32,
        pair_local_raw_barriers_per_run: u32,
        common_consumer_barriers_per_run: u32,
        commits_per_run: u32,
        waits_per_run: u32,
        reserved_alignment: u32,
        host_to_device_bytes_per_run: u64,
        device_to_host_bytes_per_run: u64,
        successful_runs: u64,
        last_observed_command_buffers: u32,
        last_observed_compute_encoders: u32,
        last_observed_kernel_dispatches: u32,
        last_observed_explicit_buffer_barriers: u32,
        last_observed_pair_local_raw_barriers: u32,
        last_observed_common_consumer_barriers: u32,
        last_observed_commits: u32,
        last_observed_waits: u32,
        requested_primary_function_name: [c_char; FUNCTION_NAME_CAPACITY],
        observed_primary_function_name: [c_char; FUNCTION_NAME_CAPACITY],
        requested_secondary_function_name: [c_char; FUNCTION_NAME_CAPACITY],
        observed_secondary_function_name: [c_char; FUNCTION_NAME_CAPACITY],
    }

    impl Default for RawRuntimeReceiptV1 {
        fn default() -> Self {
            Self {
                requested_profile: u32::MAX,
                observed_profile: u32::MAX,
                hidden_size: 0,
                seams_per_run: 0,
                threads_per_threadgroup: 0,
                simdgroups_per_threadgroup: 0,
                primary_pipeline_max_total_threads_per_threadgroup: 0,
                primary_pipeline_thread_execution_width: 0,
                primary_static_threadgroup_memory_bytes: 0,
                secondary_pipeline_max_total_threads_per_threadgroup: 0,
                secondary_pipeline_thread_execution_width: 0,
                secondary_static_threadgroup_memory_bytes: 0,
                dynamic_threadgroup_memory_bytes: 0,
                internal_threadgroup_barriers_per_seam: 0,
                internal_threadgroup_barriers_per_run: 0,
                command_buffers_per_run: 0,
                compute_encoders_per_run: 0,
                kernel_dispatches_per_run: 0,
                explicit_buffer_barriers_per_run: 0,
                pair_local_raw_barriers_per_run: 0,
                common_consumer_barriers_per_run: 0,
                commits_per_run: 0,
                waits_per_run: 0,
                reserved_alignment: 0,
                host_to_device_bytes_per_run: 0,
                device_to_host_bytes_per_run: 0,
                successful_runs: 0,
                last_observed_command_buffers: 0,
                last_observed_compute_encoders: 0,
                last_observed_kernel_dispatches: 0,
                last_observed_explicit_buffer_barriers: 0,
                last_observed_pair_local_raw_barriers: 0,
                last_observed_common_consumer_barriers: 0,
                last_observed_commits: 0,
                last_observed_waits: 0,
                requested_primary_function_name: [0; FUNCTION_NAME_CAPACITY],
                observed_primary_function_name: [0; FUNCTION_NAME_CAPACITY],
                requested_secondary_function_name: [0; FUNCTION_NAME_CAPACITY],
                observed_secondary_function_name: [0; FUNCTION_NAME_CAPACITY],
            }
        }
    }

    extern "C" {
        fn apxinf_metal_residual_rms_count18_profile_v1_create(
            weight: *const f32,
            weight_count: u32,
            rms_norm_eps: f32,
            output: *mut *mut c_void,
            error: *mut c_char,
            error_capacity: usize,
        ) -> c_int;
        fn apxinf_metal_residual_rms_count18_profile_v1_stage_fixture(
            handle: *mut c_void,
            seed: *const f32,
            seed_count: u32,
            updates: *const f32,
            update_count: u32,
            error: *mut c_char,
            error_capacity: usize,
        ) -> c_int;
        fn apxinf_metal_residual_rms_count18_profile_v1_poison_traces(
            handle: *mut c_void,
            error: *mut c_char,
            error_capacity: usize,
        ) -> c_int;
        fn apxinf_metal_residual_rms_count18_profile_v1_run(
            handle: *mut c_void,
            profile: u32,
            error: *mut c_char,
            error_capacity: usize,
        ) -> c_int;
        fn apxinf_metal_residual_rms_count18_profile_v1_snapshot(
            handle: *mut c_void,
            residual_output: *mut f32,
            residual_count: u32,
            normalized_output: *mut f32,
            normalized_count: u32,
            error: *mut c_char,
            error_capacity: usize,
        ) -> c_int;
        fn apxinf_metal_residual_rms_count18_profile_v1_receipt(
            handle: *mut c_void,
            profile: u32,
            receipt: *mut RawRuntimeReceiptV1,
            error: *mut c_char,
            error_capacity: usize,
        ) -> c_int;
        fn apxinf_metal_residual_rms_count18_profile_v1_destroy(handle: *mut c_void);
    }

    pub(super) struct Handle(NonNull<c_void>);

    impl Handle {
        pub(super) fn new(weights: &[f32], rms_norm_eps: f32) -> Result<Self, MetalW8Error> {
            let mut output = std::ptr::null_mut();
            let mut error = [0 as c_char; ERROR_CAPACITY];
            let status = unsafe {
                apxinf_metal_residual_rms_count18_profile_v1_create(
                    weights.as_ptr(),
                    weights.len() as u32,
                    rms_norm_eps,
                    &mut output,
                    error.as_mut_ptr(),
                    error.len(),
                )
            };
            if status != 0 {
                return Err(bridge_error(
                    "create Metal residual-RMS count18 primitive",
                    &error,
                ));
            }
            let handle = Self(NonNull::new(output).ok_or_else(|| {
                MetalW8Error::new(
                    "create Metal residual-RMS count18 primitive returned a null handle",
                )
            })?);
            for profile in [
                ResidualRmsProfileV1::LegacySeparate,
                ResidualRmsProfileV1::FusedExact,
            ] {
                let receipt = handle.runtime_receipt(profile)?;
                if receipt.successful_runs != 0 {
                    return Err(MetalW8Error::new(
                        "new Metal residual-RMS primitive reported successful runs",
                    ));
                }
            }
            Ok(handle)
        }

        pub(super) fn stage_fixture(
            &mut self,
            seed: &[f32],
            updates: &[f32],
        ) -> Result<(), MetalW8Error> {
            let mut error = [0 as c_char; ERROR_CAPACITY];
            let status = unsafe {
                apxinf_metal_residual_rms_count18_profile_v1_stage_fixture(
                    self.0.as_ptr(),
                    seed.as_ptr(),
                    seed.len() as u32,
                    updates.as_ptr(),
                    updates.len() as u32,
                    error.as_mut_ptr(),
                    error.len(),
                )
            };
            if status != 0 {
                return Err(bridge_error("stage Metal residual-RMS fixture", &error));
            }
            Ok(())
        }

        pub(super) fn poison_traces_for_correctness(&mut self) -> Result<(), MetalW8Error> {
            let mut error = [0 as c_char; ERROR_CAPACITY];
            let status = unsafe {
                apxinf_metal_residual_rms_count18_profile_v1_poison_traces(
                    self.0.as_ptr(),
                    error.as_mut_ptr(),
                    error.len(),
                )
            };
            if status != 0 {
                return Err(bridge_error("poison Metal residual-RMS traces", &error));
            }
            Ok(())
        }

        pub(super) fn run(&mut self, profile: ResidualRmsProfileV1) -> Result<(), MetalW8Error> {
            let mut error = [0 as c_char; ERROR_CAPACITY];
            let status = unsafe {
                apxinf_metal_residual_rms_count18_profile_v1_run(
                    self.0.as_ptr(),
                    profile.selector(),
                    error.as_mut_ptr(),
                    error.len(),
                )
            };
            if status != 0 {
                return Err(bridge_error("run Metal residual-RMS primitive", &error));
            }
            Ok(())
        }

        pub(super) fn snapshot(&self) -> Result<ResidualRmsCount18SnapshotV1, MetalW8Error> {
            let count =
                QWEN35_RESIDUAL_RMS_HIDDEN_SIZE_V1 * QWEN35_RESIDUAL_RMS_SEAMS_PER_DECODE_V1;
            let mut materialized_residual_rows = vec![0.0f32; count];
            let mut normalized_rows = vec![0.0f32; count];
            let mut error = [0 as c_char; ERROR_CAPACITY];
            let status = unsafe {
                apxinf_metal_residual_rms_count18_profile_v1_snapshot(
                    self.0.as_ptr(),
                    materialized_residual_rows.as_mut_ptr(),
                    materialized_residual_rows.len() as u32,
                    normalized_rows.as_mut_ptr(),
                    normalized_rows.len() as u32,
                    error.as_mut_ptr(),
                    error.len(),
                )
            };
            if status != 0 {
                return Err(bridge_error("snapshot Metal residual-RMS trace", &error));
            }
            Ok(ResidualRmsCount18SnapshotV1 {
                materialized_residual_rows,
                normalized_rows,
            })
        }

        pub(super) fn runtime_receipt(
            &self,
            expected_profile: ResidualRmsProfileV1,
        ) -> Result<ResidualRmsCount18RuntimeReceiptV1, MetalW8Error> {
            let mut raw = RawRuntimeReceiptV1::default();
            let mut error = [0 as c_char; ERROR_CAPACITY];
            let status = unsafe {
                apxinf_metal_residual_rms_count18_profile_v1_receipt(
                    self.0.as_ptr(),
                    expected_profile.selector(),
                    &mut raw,
                    error.as_mut_ptr(),
                    error.len(),
                )
            };
            if status != 0 {
                return Err(bridge_error("read Metal residual-RMS receipt", &error));
            }
            convert_and_validate_receipt(raw, expected_profile)
        }
    }

    impl Drop for Handle {
        fn drop(&mut self) {
            unsafe { apxinf_metal_residual_rms_count18_profile_v1_destroy(self.0.as_ptr()) };
        }
    }

    fn c_string(raw: &[c_char; FUNCTION_NAME_CAPACITY]) -> String {
        unsafe { CStr::from_ptr(raw.as_ptr()) }
            .to_string_lossy()
            .into_owned()
    }

    fn convert_and_validate_receipt(
        raw: RawRuntimeReceiptV1,
        expected: ResidualRmsProfileV1,
    ) -> Result<ResidualRmsCount18RuntimeReceiptV1, MetalW8Error> {
        let requested_profile = ResidualRmsProfileV1::try_from(raw.requested_profile)?;
        let observed_profile = ResidualRmsProfileV1::try_from(raw.observed_profile)?;
        let requested_primary_function_name = c_string(&raw.requested_primary_function_name);
        let observed_primary_function_name = c_string(&raw.observed_primary_function_name);
        let requested_secondary_function_name = c_string(&raw.requested_secondary_function_name);
        let observed_secondary_function_name = c_string(&raw.observed_secondary_function_name);
        let expected_last = u32::from(raw.successful_runs != 0);
        let secondary_valid = match expected {
            ResidualRmsProfileV1::LegacySeparate => {
                raw.secondary_pipeline_max_total_threads_per_threadgroup >= 256
                    && raw.secondary_pipeline_thread_execution_width == 32
                    && raw.secondary_static_threadgroup_memory_bytes == 1024
            }
            ResidualRmsProfileV1::FusedExact => {
                raw.secondary_pipeline_max_total_threads_per_threadgroup == 0
                    && raw.secondary_pipeline_thread_execution_width == 0
                    && raw.secondary_static_threadgroup_memory_bytes == 0
            }
        };
        let expected_primary_static = match expected {
            ResidualRmsProfileV1::LegacySeparate => 0,
            ResidualRmsProfileV1::FusedExact => 1024,
        };
        if requested_profile != expected
            || observed_profile != expected
            || requested_primary_function_name != expected.expected_primary_function_name()
            || observed_primary_function_name != expected.expected_primary_function_name()
            || requested_secondary_function_name != expected.expected_secondary_function_name()
            || observed_secondary_function_name != expected.expected_secondary_function_name()
            || raw.hidden_size != QWEN35_RESIDUAL_RMS_HIDDEN_SIZE_V1 as u32
            || raw.seams_per_run != QWEN35_RESIDUAL_RMS_SEAMS_PER_DECODE_V1 as u32
            || raw.threads_per_threadgroup != 256
            || raw.simdgroups_per_threadgroup != 8
            || raw.primary_pipeline_max_total_threads_per_threadgroup < 256
            || raw.primary_pipeline_thread_execution_width != 32
            || raw.primary_static_threadgroup_memory_bytes != expected_primary_static
            || !secondary_valid
            || raw.dynamic_threadgroup_memory_bytes != 0
            || raw.internal_threadgroup_barriers_per_seam != 9
            || raw.internal_threadgroup_barriers_per_run != 162
            || raw.command_buffers_per_run != 1
            || raw.compute_encoders_per_run != 1
            || raw.kernel_dispatches_per_run != expected.kernel_dispatches_per_run()
            || raw.explicit_buffer_barriers_per_run != expected.explicit_buffer_barriers_per_run()
            || raw.pair_local_raw_barriers_per_run != expected.pair_local_raw_barriers_per_run()
            || raw.common_consumer_barriers_per_run != expected.common_consumer_barriers_per_run()
            || raw.commits_per_run != 1
            || raw.waits_per_run != 1
            || raw.host_to_device_bytes_per_run != 0
            || raw.device_to_host_bytes_per_run != 0
            || raw.last_observed_command_buffers != expected_last
            || raw.last_observed_compute_encoders != expected_last
            || raw.last_observed_kernel_dispatches
                != expected_last * expected.kernel_dispatches_per_run()
            || raw.last_observed_explicit_buffer_barriers
                != expected_last * expected.explicit_buffer_barriers_per_run()
            || raw.last_observed_pair_local_raw_barriers
                != expected_last * expected.pair_local_raw_barriers_per_run()
            || raw.last_observed_common_consumer_barriers
                != expected_last * expected.common_consumer_barriers_per_run()
            || raw.last_observed_commits != expected_last
            || raw.last_observed_waits != expected_last
        {
            return Err(MetalW8Error::new(format!(
                "invalid live Metal residual-RMS count18 receipt for {expected:?}"
            )));
        }
        Ok(ResidualRmsCount18RuntimeReceiptV1 {
            requested_profile,
            observed_profile,
            requested_primary_function_name,
            observed_primary_function_name,
            requested_secondary_function_name,
            observed_secondary_function_name,
            hidden_size: raw.hidden_size,
            seams_per_run: raw.seams_per_run,
            threads_per_threadgroup: raw.threads_per_threadgroup,
            simdgroups_per_threadgroup: raw.simdgroups_per_threadgroup,
            primary_pipeline_max_total_threads_per_threadgroup: raw
                .primary_pipeline_max_total_threads_per_threadgroup,
            primary_pipeline_thread_execution_width: raw.primary_pipeline_thread_execution_width,
            primary_static_threadgroup_memory_bytes: raw.primary_static_threadgroup_memory_bytes,
            secondary_pipeline_max_total_threads_per_threadgroup: raw
                .secondary_pipeline_max_total_threads_per_threadgroup,
            secondary_pipeline_thread_execution_width: raw
                .secondary_pipeline_thread_execution_width,
            secondary_static_threadgroup_memory_bytes: raw
                .secondary_static_threadgroup_memory_bytes,
            dynamic_threadgroup_memory_bytes: raw.dynamic_threadgroup_memory_bytes,
            internal_threadgroup_barriers_per_seam: raw.internal_threadgroup_barriers_per_seam,
            internal_threadgroup_barriers_per_run: raw.internal_threadgroup_barriers_per_run,
            command_buffers_per_run: raw.command_buffers_per_run,
            compute_encoders_per_run: raw.compute_encoders_per_run,
            kernel_dispatches_per_run: raw.kernel_dispatches_per_run,
            explicit_buffer_barriers_per_run: raw.explicit_buffer_barriers_per_run,
            pair_local_raw_barriers_per_run: raw.pair_local_raw_barriers_per_run,
            common_consumer_barriers_per_run: raw.common_consumer_barriers_per_run,
            commits_per_run: raw.commits_per_run,
            waits_per_run: raw.waits_per_run,
            host_to_device_bytes_per_run: raw.host_to_device_bytes_per_run,
            device_to_host_bytes_per_run: raw.device_to_host_bytes_per_run,
            successful_runs: raw.successful_runs,
            last_observed_command_buffers: raw.last_observed_command_buffers,
            last_observed_compute_encoders: raw.last_observed_compute_encoders,
            last_observed_kernel_dispatches: raw.last_observed_kernel_dispatches,
            last_observed_explicit_buffer_barriers: raw.last_observed_explicit_buffer_barriers,
            last_observed_pair_local_raw_barriers: raw.last_observed_pair_local_raw_barriers,
            last_observed_common_consumer_barriers: raw.last_observed_common_consumer_barriers,
            last_observed_commits: raw.last_observed_commits,
            last_observed_waits: raw.last_observed_waits,
        })
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

    impl Handle {
        pub(super) fn invalid_raw_selectors_are_rejected_without_mutation(&self) -> bool {
            let before_a = self
                .runtime_receipt(ResidualRmsProfileV1::LegacySeparate)
                .ok();
            let before_b = self.runtime_receipt(ResidualRmsProfileV1::FusedExact).ok();
            let before_snapshot = self.snapshot().ok();
            let mut raw = RawRuntimeReceiptV1::default();
            let mut error = [0 as c_char; ERROR_CAPACITY];
            let receipt_rejected = unsafe {
                apxinf_metal_residual_rms_count18_profile_v1_receipt(
                    self.0.as_ptr(),
                    99,
                    &mut raw,
                    error.as_mut_ptr(),
                    error.len(),
                ) != 0
            };
            error.fill(0);
            let run_rejected = unsafe {
                apxinf_metal_residual_rms_count18_profile_v1_run(
                    self.0.as_ptr(),
                    99,
                    error.as_mut_ptr(),
                    error.len(),
                ) != 0
            };
            let after_a = self
                .runtime_receipt(ResidualRmsProfileV1::LegacySeparate)
                .ok();
            let after_b = self.runtime_receipt(ResidualRmsProfileV1::FusedExact).ok();
            let after_snapshot = self.snapshot().ok();
            receipt_rejected
                && run_rejected
                && before_a.is_some()
                && before_b.is_some()
                && before_a == after_a
                && before_b == after_b
                && before_snapshot == after_snapshot
        }
    }

    #[cfg(test)]
    pub(super) fn raw_receipt_size() -> usize {
        std::mem::size_of::<RawRuntimeReceiptV1>()
    }
}

#[cfg(not(target_os = "macos"))]
mod platform {
    use super::{
        MetalW8Error, ResidualRmsCount18RuntimeReceiptV1, ResidualRmsCount18SnapshotV1,
        ResidualRmsProfileV1,
    };

    pub(super) struct Handle;

    impl Handle {
        pub(super) fn new(_weights: &[f32], _rms_norm_eps: f32) -> Result<Self, MetalW8Error> {
            Err(MetalW8Error::new(
                "Metal residual-RMS count18 primitive requires macOS",
            ))
        }

        pub(super) fn stage_fixture(
            &mut self,
            _seed: &[f32],
            _updates: &[f32],
        ) -> Result<(), MetalW8Error> {
            Err(MetalW8Error::new(
                "Metal residual-RMS count18 primitive requires macOS",
            ))
        }

        pub(super) fn poison_traces_for_correctness(&mut self) -> Result<(), MetalW8Error> {
            Err(MetalW8Error::new(
                "Metal residual-RMS count18 primitive requires macOS",
            ))
        }

        pub(super) fn run(&mut self, _profile: ResidualRmsProfileV1) -> Result<(), MetalW8Error> {
            Err(MetalW8Error::new(
                "Metal residual-RMS count18 primitive requires macOS",
            ))
        }

        pub(super) fn snapshot(&self) -> Result<ResidualRmsCount18SnapshotV1, MetalW8Error> {
            Err(MetalW8Error::new(
                "Metal residual-RMS count18 primitive requires macOS",
            ))
        }

        pub(super) fn runtime_receipt(
            &self,
            _profile: ResidualRmsProfileV1,
        ) -> Result<ResidualRmsCount18RuntimeReceiptV1, MetalW8Error> {
            Err(MetalW8Error::new(
                "Metal residual-RMS count18 primitive requires macOS",
            ))
        }

        pub(super) fn invalid_raw_selectors_are_rejected_without_mutation(&self) -> bool {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn candidate_is_additive_and_production_bridges_remain_legacy() {
        let shader = include_str!("metal_w8_linear_layer.metal");
        assert!(shader.contains("kernel void linear_layer_rms_norm("));
        assert!(shader.contains("kernel void linear_layer_residual_add("));
        assert!(shader.contains("kernel void linear_layer_residual_rms_norm_fused_exact_v1("));
        let candidate = shader
            .split("kernel void linear_layer_residual_rms_norm_fused_exact_v1(")
            .nth(1)
            .unwrap()
            .split("kernel void linear_layer_residual_add(")
            .next()
            .unwrap();
        assert_eq!(candidate.matches("threadgroup_barrier").count(), 2);
        assert!(candidate.contains("for (uint stride = 128; stride != 0; stride >>= 1)"));
        assert_eq!(
            candidate
                .matches("#pragma clang fp reassociate(off)")
                .count(),
            1
        );
        assert!(!shader.contains("#pragma clang fp reassociate(on)"));
        assert!(candidate.contains("float4 retained_values"));
        assert!(candidate.contains("const float value = residual[index] + update[index]"));
        assert!(candidate.contains("materialized_residual[index] = value"));
        assert!(candidate.contains("retained_values[retained_index] * inverse_rms * weight[index]"));
        assert!(!candidate.contains("simd_"));

        for bridge in [
            include_str!("metal_w8_linear_layer_bridge.mm"),
            include_str!("metal_w8_linear_layer_stack3_bridge.mm"),
            include_str!("metal_w8_mlp_stack3_boundary_v1_bridge.mm"),
            include_str!("metal_w8_tail_mlp_head_v1_bridge.mm"),
        ] {
            assert!(bridge.contains("@\"linear_layer_rms_norm\""));
            assert!(bridge.contains("@\"linear_layer_residual_add\""));
            assert!(!bridge.contains("linear_layer_residual_rms_norm_fused_exact_v1"));
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn count18_fusion_matches_both_legacy_traces_to_bits() {
        let trace_count =
            QWEN35_RESIDUAL_RMS_HIDDEN_SIZE_V1 * QWEN35_RESIDUAL_RMS_SEAMS_PER_DECODE_V1;
        let weights = (0..trace_count)
            .map(|index| 0.75 + ((index * 37 + 11) % 257) as f32 / 1024.0)
            .collect::<Vec<_>>();
        let updates = (0..trace_count)
            .map(|index| (((index * 19 + 3) % 63) as f32 - 31.0) / 4096.0)
            .collect::<Vec<_>>();
        let seed = (0..QWEN35_RESIDUAL_RMS_HIDDEN_SIZE_V1)
            .map(|index| (((index * 29 + index / 7 + 5) % 401) as f32 - 200.0) / 256.0)
            .collect::<Vec<_>>();
        let mut primitive = MetalResidualRmsCount18PrimitiveV1::new(&weights, 1.0e-6).unwrap();
        primitive.stage_fixture(&seed, &updates).unwrap();
        assert!(primitive.snapshot().is_err());
        primitive.poison_traces_for_correctness().unwrap();
        primitive.run(ResidualRmsProfileV1::LegacySeparate).unwrap();
        let legacy = primitive.snapshot().unwrap();
        primitive.poison_traces_for_correctness().unwrap();
        primitive.run(ResidualRmsProfileV1::FusedExact).unwrap();
        let fused = primitive.snapshot().unwrap();
        let expected_count =
            QWEN35_RESIDUAL_RMS_HIDDEN_SIZE_V1 * QWEN35_RESIDUAL_RMS_SEAMS_PER_DECODE_V1;
        assert_eq!(legacy.materialized_residual_rows.len(), expected_count);
        assert_eq!(legacy.normalized_rows.len(), expected_count);
        for (label, expected, actual) in [
            (
                "materialized_residual",
                &legacy.materialized_residual_rows,
                &fused.materialized_residual_rows,
            ),
            (
                "normalized",
                &legacy.normalized_rows,
                &fused.normalized_rows,
            ),
        ] {
            for (index, (&left, &right)) in expected.iter().zip(actual).enumerate() {
                assert!(
                    left.is_finite() && right.is_finite(),
                    "{label} produced a non-finite element at {index}"
                );
                assert_eq!(
                    left.to_bits(),
                    right.to_bits(),
                    "{label} mismatch at trace element {index}"
                );
            }
        }
        assert_eq!(platform::raw_receipt_size(), 408);
        primitive.verify_invalid_raw_selector_fail_closed().unwrap();
        assert_eq!(
            primitive
                .runtime_receipt(ResidualRmsProfileV1::LegacySeparate)
                .unwrap()
                .successful_runs,
            1
        );
        assert_eq!(
            primitive
                .runtime_receipt(ResidualRmsProfileV1::FusedExact)
                .unwrap()
                .successful_runs,
            1
        );
    }
}
