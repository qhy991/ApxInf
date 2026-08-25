use crate::MetalW8Error;

pub const QWEN35_GDN_RECURRENT_SEAMS_PER_DECODE_V1: usize = 18;
pub const QWEN35_GDN_KEY_HEADS_V1: usize = 16;
pub const QWEN35_GDN_VALUE_HEADS_V1: usize = 16;
pub const QWEN35_GDN_KEY_DIM_V1: usize = 128;
pub const QWEN35_GDN_VALUE_DIM_V1: usize = 128;
pub const QWEN35_GDN_PROCESSED_ELEMENTS_PER_SEAM_V1: usize =
    2 * QWEN35_GDN_KEY_HEADS_V1 * QWEN35_GDN_KEY_DIM_V1
        + QWEN35_GDN_VALUE_HEADS_V1 * QWEN35_GDN_VALUE_DIM_V1;
pub const QWEN35_GDN_PROJECTED_ELEMENTS_PER_SEAM_V1: usize =
    QWEN35_GDN_PROCESSED_ELEMENTS_PER_SEAM_V1
        + QWEN35_GDN_VALUE_HEADS_V1 * QWEN35_GDN_VALUE_DIM_V1
        + 2 * QWEN35_GDN_VALUE_HEADS_V1;
pub const QWEN35_GDN_RECURRENT_ELEMENTS_PER_SEAM_V1: usize =
    QWEN35_GDN_VALUE_HEADS_V1 * QWEN35_GDN_KEY_DIM_V1 * QWEN35_GDN_VALUE_DIM_V1;
pub const QWEN35_GDN_CORE_ELEMENTS_PER_SEAM_V1: usize =
    QWEN35_GDN_VALUE_HEADS_V1 * QWEN35_GDN_VALUE_DIM_V1;

const PROCESSED_TRACE_ELEMENTS: usize =
    QWEN35_GDN_RECURRENT_SEAMS_PER_DECODE_V1 * QWEN35_GDN_PROCESSED_ELEMENTS_PER_SEAM_V1;
const PROJECTED_TRACE_ELEMENTS: usize =
    QWEN35_GDN_RECURRENT_SEAMS_PER_DECODE_V1 * QWEN35_GDN_PROJECTED_ELEMENTS_PER_SEAM_V1;
const HEAD_SCALAR_TRACE_ELEMENTS: usize =
    QWEN35_GDN_RECURRENT_SEAMS_PER_DECODE_V1 * QWEN35_GDN_VALUE_HEADS_V1;
const RECURRENT_TRACE_ELEMENTS: usize =
    QWEN35_GDN_RECURRENT_SEAMS_PER_DECODE_V1 * QWEN35_GDN_RECURRENT_ELEMENTS_PER_SEAM_V1;
const CORE_TRACE_ELEMENTS: usize =
    QWEN35_GDN_RECURRENT_SEAMS_PER_DECODE_V1 * QWEN35_GDN_CORE_ELEMENTS_PER_SEAM_V1;

/// Explicit selector for the bounded count-18 GDN recurrent mechanism screen.
/// Existing production constructors remain bound to [`Self::Legacy256`].
#[repr(u32)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum GdnRecurrentProfileV1 {
    #[default]
    Legacy256 = 0,
    LeaderBroadcast128 = 1,
    QkStaged128 = 2,
}

impl GdnRecurrentProfileV1 {
    pub const ALL: [Self; 3] = [Self::Legacy256, Self::LeaderBroadcast128, Self::QkStaged128];

    pub const fn selector(self) -> u32 {
        self as u32
    }

    pub const fn expected_function_name(self) -> &'static str {
        match self {
            Self::Legacy256 => "gdn_recurrent_update",
            Self::LeaderBroadcast128 => "gdn_recurrent_update_leader_broadcast_v1",
            Self::QkStaged128 => "gdn_recurrent_update_qk_staged_v1",
        }
    }

    pub const fn threads_per_threadgroup(self) -> u32 {
        match self {
            Self::Legacy256 => 256,
            Self::LeaderBroadcast128 | Self::QkStaged128 => 128,
        }
    }

    pub const fn source_declared_threadgroup_memory_bytes(self) -> u32 {
        match self {
            Self::Legacy256 => 0,
            Self::LeaderBroadcast128 => 8,
            Self::QkStaged128 => 1032,
        }
    }

    pub const fn internal_threadgroup_barrier_sites(self) -> u32 {
        match self {
            Self::Legacy256 => 0,
            Self::LeaderBroadcast128 | Self::QkStaged128 => 1,
        }
    }

    fn from_selector(selector: u32) -> Option<Self> {
        match selector {
            0 => Some(Self::Legacy256),
            1 => Some(Self::LeaderBroadcast128),
            2 => Some(Self::QkStaged128),
            _ => None,
        }
    }
}

impl TryFrom<u32> for GdnRecurrentProfileV1 {
    type Error = MetalW8Error;

    fn try_from(selector: u32) -> Result<Self, Self::Error> {
        Self::from_selector(selector).ok_or_else(|| {
            MetalW8Error::new(format!(
                "Metal GDN recurrent profile {selector} is invalid; expected 0, 1, or 2"
            ))
        })
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct GdnRecurrentCount18SnapshotV1 {
    pub next_state: Vec<f32>,
    pub core: Vec<f32>,
}

/// Live function identity and observed topology for one arm of the additive
/// count-18 recurrent primitive. Source-declared memory and internal barrier
/// counts remain explicitly qualified rather than presented as hardware
/// counters.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GdnRecurrentCount18RuntimeReceiptV1 {
    pub requested_profile: GdnRecurrentProfileV1,
    pub observed_profile: GdnRecurrentProfileV1,
    pub requested_function_name: String,
    pub observed_function_name: String,
    pub seams_per_run: u32,
    pub key_heads: u32,
    pub value_heads: u32,
    pub key_dim: u32,
    pub value_dim: u32,
    pub processed_elements_per_seam: u32,
    pub projected_elements_per_seam: u32,
    pub recurrent_elements_per_seam: u32,
    pub core_elements_per_seam: u32,
    pub threads_per_threadgroup: u32,
    pub simdgroups_per_threadgroup: u32,
    pub pipeline_max_total_threads_per_threadgroup: u32,
    pub pipeline_thread_execution_width: u32,
    pub pipeline_static_threadgroup_memory_bytes: u32,
    pub source_declared_threadgroup_memory_bytes: u32,
    pub dynamic_threadgroup_memory_bytes: u32,
    pub internal_threadgroup_barrier_sites_per_threadgroup: u32,
    pub source_derived_internal_barrier_executions_per_run: u32,
    pub launched_threads_per_run: u32,
    pub active_value_threads_per_run: u32,
    pub idle_threads_per_run: u32,
    pub command_buffers_per_run: u32,
    pub compute_encoders_per_run: u32,
    pub kernel_dispatches_per_run: u32,
    pub threadgroups_per_run: u32,
    pub explicit_buffer_barriers_per_run: u32,
    pub commits_per_run: u32,
    pub waits_per_run: u32,
    pub fixed_shape_host_validated: bool,
    pub input_output_buffers_non_overlapping: bool,
    pub host_to_device_bytes_per_run: u64,
    pub device_to_host_bytes_per_run: u64,
    pub processed_buffer_bytes: u64,
    pub projected_buffer_bytes: u64,
    pub a_log_buffer_bytes: u64,
    pub dt_bias_buffer_bytes: u64,
    pub state_buffer_bytes: u64,
    pub next_state_buffer_bytes: u64,
    pub core_buffer_bytes: u64,
    pub persistent_buffer_bytes_total: u64,
    pub successful_runs: u64,
    pub last_observed_command_buffers: u32,
    pub last_observed_compute_encoders: u32,
    pub last_observed_kernel_dispatches: u32,
    pub last_observed_threadgroups: u32,
    pub last_observed_explicit_buffer_barriers: u32,
    pub last_observed_launched_threads: u32,
    pub last_observed_active_value_threads: u32,
    pub last_observed_idle_threads: u32,
    pub last_observed_commits: u32,
    pub last_observed_waits: u32,
}

/// Same-binary fixed-shape aggregate primitive. Staging, output poisoning,
/// and snapshots are separate calls, so a timed run performs no explicit
/// bridge memcpy.
pub struct MetalGdnRecurrentCount18PrimitiveV1 {
    inner: platform::Handle,
}

impl MetalGdnRecurrentCount18PrimitiveV1 {
    pub fn new() -> Result<Self, MetalW8Error> {
        Ok(Self {
            inner: platform::Handle::new()?,
        })
    }

    pub fn stage_fixture(
        &mut self,
        processed: &[f32],
        projected: &[f32],
        a_log: &[f32],
        dt_bias: &[f32],
        state: &[f32],
    ) -> Result<(), MetalW8Error> {
        validate_finite("processed", processed, PROCESSED_TRACE_ELEMENTS)?;
        validate_finite("projected", projected, PROJECTED_TRACE_ELEMENTS)?;
        validate_finite("A_log", a_log, HEAD_SCALAR_TRACE_ELEMENTS)?;
        validate_finite("dt_bias", dt_bias, HEAD_SCALAR_TRACE_ELEMENTS)?;
        validate_finite("state", state, RECURRENT_TRACE_ELEMENTS)?;
        self.inner
            .stage_fixture(processed, projected, a_log, dt_bias, state)
    }

    pub fn poison_outputs_for_correctness(&mut self) -> Result<(), MetalW8Error> {
        self.inner.poison_outputs_for_correctness()
    }

    pub fn verify_staged_fixture_unchanged(
        &self,
        processed: &[f32],
        projected: &[f32],
        a_log: &[f32],
        dt_bias: &[f32],
        state: &[f32],
    ) -> Result<(), MetalW8Error> {
        validate_finite("processed", processed, PROCESSED_TRACE_ELEMENTS)?;
        validate_finite("projected", projected, PROJECTED_TRACE_ELEMENTS)?;
        validate_finite("A_log", a_log, HEAD_SCALAR_TRACE_ELEMENTS)?;
        validate_finite("dt_bias", dt_bias, HEAD_SCALAR_TRACE_ELEMENTS)?;
        validate_finite("state", state, RECURRENT_TRACE_ELEMENTS)?;
        self.inner
            .verify_staged_fixture_unchanged(processed, projected, a_log, dt_bias, state)
    }

    pub fn run(&mut self, profile: GdnRecurrentProfileV1) -> Result<(), MetalW8Error> {
        self.inner.run(profile)
    }

    pub fn snapshot(&self) -> Result<GdnRecurrentCount18SnapshotV1, MetalW8Error> {
        self.inner.snapshot()
    }

    pub fn runtime_receipt(
        &self,
        profile: GdnRecurrentProfileV1,
    ) -> Result<GdnRecurrentCount18RuntimeReceiptV1, MetalW8Error> {
        self.inner.runtime_receipt(profile)
    }

    pub fn verify_invalid_raw_selector_fail_closed(&self) -> Result<(), MetalW8Error> {
        if self
            .inner
            .invalid_raw_selectors_are_rejected_without_mutation()
        {
            Ok(())
        } else {
            Err(MetalW8Error::new(
                "invalid raw Metal GDN recurrent selector mutated observable state",
            ))
        }
    }
}

fn validate_finite(label: &str, values: &[f32], expected: usize) -> Result<(), MetalW8Error> {
    if values.len() != expected {
        return Err(MetalW8Error::new(format!(
            "Metal GDN recurrent {label} has {} elements, expected {expected}",
            values.len()
        )));
    }
    if let Some(index) = values.iter().position(|value| !value.is_finite()) {
        return Err(MetalW8Error::new(format!(
            "Metal GDN recurrent {label} contains a non-finite value at element {index}"
        )));
    }
    Ok(())
}

fn optional_snapshots_match_to_bits(
    left: &Option<GdnRecurrentCount18SnapshotV1>,
    right: &Option<GdnRecurrentCount18SnapshotV1>,
) -> bool {
    match (left, right) {
        (None, None) => true,
        (Some(left), Some(right)) => {
            left.next_state.len() == right.next_state.len()
                && left.core.len() == right.core.len()
                && left
                    .next_state
                    .iter()
                    .zip(&right.next_state)
                    .all(|(left, right)| left.to_bits() == right.to_bits())
                && left
                    .core
                    .iter()
                    .zip(&right.core)
                    .all(|(left, right)| left.to_bits() == right.to_bits())
        }
        _ => false,
    }
}

#[cfg(target_os = "macos")]
mod platform {
    use super::*;
    use std::ffi::{c_char, c_int, c_void, CStr};
    use std::ptr::NonNull;

    const ERROR_CAPACITY: usize = 1024;
    const FUNCTION_NAME_CAPACITY: usize = 64;

    #[repr(C)]
    #[derive(Clone, Copy)]
    struct RawRuntimeReceiptV1 {
        requested_profile: u32,
        observed_profile: u32,
        seams_per_run: u32,
        key_heads: u32,
        value_heads: u32,
        key_dim: u32,
        value_dim: u32,
        processed_elements_per_seam: u32,
        projected_elements_per_seam: u32,
        recurrent_elements_per_seam: u32,
        core_elements_per_seam: u32,
        threads_per_threadgroup: u32,
        simdgroups_per_threadgroup: u32,
        pipeline_max_total_threads_per_threadgroup: u32,
        pipeline_thread_execution_width: u32,
        pipeline_static_threadgroup_memory_bytes: u32,
        source_declared_threadgroup_memory_bytes: u32,
        dynamic_threadgroup_memory_bytes: u32,
        internal_threadgroup_barrier_sites_per_threadgroup: u32,
        source_derived_internal_barrier_executions_per_run: u32,
        launched_threads_per_run: u32,
        active_value_threads_per_run: u32,
        idle_threads_per_run: u32,
        command_buffers_per_run: u32,
        compute_encoders_per_run: u32,
        kernel_dispatches_per_run: u32,
        threadgroups_per_run: u32,
        explicit_buffer_barriers_per_run: u32,
        commits_per_run: u32,
        waits_per_run: u32,
        fixed_shape_host_validated: u32,
        input_output_buffers_non_overlapping: u32,
        host_to_device_bytes_per_run: u64,
        device_to_host_bytes_per_run: u64,
        processed_buffer_bytes: u64,
        projected_buffer_bytes: u64,
        a_log_buffer_bytes: u64,
        dt_bias_buffer_bytes: u64,
        state_buffer_bytes: u64,
        next_state_buffer_bytes: u64,
        core_buffer_bytes: u64,
        persistent_buffer_bytes_total: u64,
        successful_runs: u64,
        last_observed_command_buffers: u32,
        last_observed_compute_encoders: u32,
        last_observed_kernel_dispatches: u32,
        last_observed_threadgroups: u32,
        last_observed_explicit_buffer_barriers: u32,
        last_observed_launched_threads: u32,
        last_observed_active_value_threads: u32,
        last_observed_idle_threads: u32,
        last_observed_commits: u32,
        last_observed_waits: u32,
        requested_function_name: [c_char; FUNCTION_NAME_CAPACITY],
        observed_function_name: [c_char; FUNCTION_NAME_CAPACITY],
    }

    impl Default for RawRuntimeReceiptV1 {
        fn default() -> Self {
            Self {
                requested_profile: u32::MAX,
                observed_profile: u32::MAX,
                seams_per_run: 0,
                key_heads: 0,
                value_heads: 0,
                key_dim: 0,
                value_dim: 0,
                processed_elements_per_seam: 0,
                projected_elements_per_seam: 0,
                recurrent_elements_per_seam: 0,
                core_elements_per_seam: 0,
                threads_per_threadgroup: 0,
                simdgroups_per_threadgroup: 0,
                pipeline_max_total_threads_per_threadgroup: 0,
                pipeline_thread_execution_width: 0,
                pipeline_static_threadgroup_memory_bytes: 0,
                source_declared_threadgroup_memory_bytes: 0,
                dynamic_threadgroup_memory_bytes: 0,
                internal_threadgroup_barrier_sites_per_threadgroup: 0,
                source_derived_internal_barrier_executions_per_run: 0,
                launched_threads_per_run: 0,
                active_value_threads_per_run: 0,
                idle_threads_per_run: 0,
                command_buffers_per_run: 0,
                compute_encoders_per_run: 0,
                kernel_dispatches_per_run: 0,
                threadgroups_per_run: 0,
                explicit_buffer_barriers_per_run: 0,
                commits_per_run: 0,
                waits_per_run: 0,
                fixed_shape_host_validated: 0,
                input_output_buffers_non_overlapping: 0,
                host_to_device_bytes_per_run: 0,
                device_to_host_bytes_per_run: 0,
                processed_buffer_bytes: 0,
                projected_buffer_bytes: 0,
                a_log_buffer_bytes: 0,
                dt_bias_buffer_bytes: 0,
                state_buffer_bytes: 0,
                next_state_buffer_bytes: 0,
                core_buffer_bytes: 0,
                persistent_buffer_bytes_total: 0,
                successful_runs: 0,
                last_observed_command_buffers: 0,
                last_observed_compute_encoders: 0,
                last_observed_kernel_dispatches: 0,
                last_observed_threadgroups: 0,
                last_observed_explicit_buffer_barriers: 0,
                last_observed_launched_threads: 0,
                last_observed_active_value_threads: 0,
                last_observed_idle_threads: 0,
                last_observed_commits: 0,
                last_observed_waits: 0,
                requested_function_name: [0; FUNCTION_NAME_CAPACITY],
                observed_function_name: [0; FUNCTION_NAME_CAPACITY],
            }
        }
    }

    extern "C" {
        fn apxinf_metal_gdn_recurrent_count18_profile_v1_create(
            output: *mut *mut c_void,
            error: *mut c_char,
            error_capacity: usize,
        ) -> c_int;
        fn apxinf_metal_gdn_recurrent_count18_profile_v1_stage_fixture(
            handle: *mut c_void,
            processed: *const f32,
            processed_count: u32,
            projected: *const f32,
            projected_count: u32,
            a_log: *const f32,
            a_log_count: u32,
            dt_bias: *const f32,
            dt_bias_count: u32,
            state: *const f32,
            state_count: u32,
            error: *mut c_char,
            error_capacity: usize,
        ) -> c_int;
        fn apxinf_metal_gdn_recurrent_count18_profile_v1_verify_fixture_unchanged(
            handle: *mut c_void,
            processed: *const f32,
            processed_count: u32,
            projected: *const f32,
            projected_count: u32,
            a_log: *const f32,
            a_log_count: u32,
            dt_bias: *const f32,
            dt_bias_count: u32,
            state: *const f32,
            state_count: u32,
            error: *mut c_char,
            error_capacity: usize,
        ) -> c_int;
        fn apxinf_metal_gdn_recurrent_count18_profile_v1_poison_outputs(
            handle: *mut c_void,
            error: *mut c_char,
            error_capacity: usize,
        ) -> c_int;
        fn apxinf_metal_gdn_recurrent_count18_profile_v1_run(
            handle: *mut c_void,
            profile: u32,
            error: *mut c_char,
            error_capacity: usize,
        ) -> c_int;
        fn apxinf_metal_gdn_recurrent_count18_profile_v1_snapshot(
            handle: *mut c_void,
            next_state_output: *mut f32,
            next_state_count: u32,
            core_output: *mut f32,
            core_count: u32,
            error: *mut c_char,
            error_capacity: usize,
        ) -> c_int;
        fn apxinf_metal_gdn_recurrent_count18_profile_v1_receipt(
            handle: *mut c_void,
            profile: u32,
            receipt: *mut RawRuntimeReceiptV1,
            error: *mut c_char,
            error_capacity: usize,
        ) -> c_int;
        fn apxinf_metal_gdn_recurrent_count18_profile_v1_destroy(handle: *mut c_void);
    }

    pub(super) struct Handle(NonNull<c_void>);

    impl Handle {
        pub(super) fn new() -> Result<Self, MetalW8Error> {
            let mut output = std::ptr::null_mut();
            let mut error = [0 as c_char; ERROR_CAPACITY];
            let status = unsafe {
                apxinf_metal_gdn_recurrent_count18_profile_v1_create(
                    &mut output,
                    error.as_mut_ptr(),
                    error.len(),
                )
            };
            if status != 0 {
                return Err(bridge_error("create Metal GDN recurrent primitive", &error));
            }
            let handle = Self(NonNull::new(output).ok_or_else(|| {
                MetalW8Error::new("create Metal GDN recurrent primitive returned a null handle")
            })?);
            for profile in GdnRecurrentProfileV1::ALL {
                let receipt = handle.runtime_receipt(profile)?;
                if receipt.successful_runs != 0 {
                    return Err(MetalW8Error::new(
                        "new Metal GDN recurrent primitive reported successful runs",
                    ));
                }
            }
            Ok(handle)
        }

        pub(super) fn stage_fixture(
            &mut self,
            processed: &[f32],
            projected: &[f32],
            a_log: &[f32],
            dt_bias: &[f32],
            state: &[f32],
        ) -> Result<(), MetalW8Error> {
            let mut error = [0 as c_char; ERROR_CAPACITY];
            let status = unsafe {
                apxinf_metal_gdn_recurrent_count18_profile_v1_stage_fixture(
                    self.0.as_ptr(),
                    processed.as_ptr(),
                    processed.len() as u32,
                    projected.as_ptr(),
                    projected.len() as u32,
                    a_log.as_ptr(),
                    a_log.len() as u32,
                    dt_bias.as_ptr(),
                    dt_bias.len() as u32,
                    state.as_ptr(),
                    state.len() as u32,
                    error.as_mut_ptr(),
                    error.len(),
                )
            };
            if status != 0 {
                return Err(bridge_error("stage Metal GDN recurrent fixture", &error));
            }
            Ok(())
        }

        pub(super) fn poison_outputs_for_correctness(&mut self) -> Result<(), MetalW8Error> {
            let mut error = [0 as c_char; ERROR_CAPACITY];
            let status = unsafe {
                apxinf_metal_gdn_recurrent_count18_profile_v1_poison_outputs(
                    self.0.as_ptr(),
                    error.as_mut_ptr(),
                    error.len(),
                )
            };
            if status != 0 {
                return Err(bridge_error("poison Metal GDN recurrent outputs", &error));
            }
            Ok(())
        }

        pub(super) fn verify_staged_fixture_unchanged(
            &self,
            processed: &[f32],
            projected: &[f32],
            a_log: &[f32],
            dt_bias: &[f32],
            state: &[f32],
        ) -> Result<(), MetalW8Error> {
            let mut error = [0 as c_char; ERROR_CAPACITY];
            let status = unsafe {
                apxinf_metal_gdn_recurrent_count18_profile_v1_verify_fixture_unchanged(
                    self.0.as_ptr(),
                    processed.as_ptr(),
                    processed.len() as u32,
                    projected.as_ptr(),
                    projected.len() as u32,
                    a_log.as_ptr(),
                    a_log.len() as u32,
                    dt_bias.as_ptr(),
                    dt_bias.len() as u32,
                    state.as_ptr(),
                    state.len() as u32,
                    error.as_mut_ptr(),
                    error.len(),
                )
            };
            if status != 0 {
                return Err(bridge_error(
                    "verify staged Metal GDN recurrent fixture",
                    &error,
                ));
            }
            Ok(())
        }

        pub(super) fn run(&mut self, profile: GdnRecurrentProfileV1) -> Result<(), MetalW8Error> {
            let mut error = [0 as c_char; ERROR_CAPACITY];
            let status = unsafe {
                apxinf_metal_gdn_recurrent_count18_profile_v1_run(
                    self.0.as_ptr(),
                    profile.selector(),
                    error.as_mut_ptr(),
                    error.len(),
                )
            };
            if status != 0 {
                return Err(bridge_error("run Metal GDN recurrent primitive", &error));
            }
            Ok(())
        }

        pub(super) fn snapshot(&self) -> Result<GdnRecurrentCount18SnapshotV1, MetalW8Error> {
            let mut next_state = vec![0.0f32; RECURRENT_TRACE_ELEMENTS];
            let mut core = vec![0.0f32; CORE_TRACE_ELEMENTS];
            let mut error = [0 as c_char; ERROR_CAPACITY];
            let status = unsafe {
                apxinf_metal_gdn_recurrent_count18_profile_v1_snapshot(
                    self.0.as_ptr(),
                    next_state.as_mut_ptr(),
                    next_state.len() as u32,
                    core.as_mut_ptr(),
                    core.len() as u32,
                    error.as_mut_ptr(),
                    error.len(),
                )
            };
            if status != 0 {
                return Err(bridge_error("snapshot Metal GDN recurrent outputs", &error));
            }
            Ok(GdnRecurrentCount18SnapshotV1 { next_state, core })
        }

        pub(super) fn runtime_receipt(
            &self,
            expected: GdnRecurrentProfileV1,
        ) -> Result<GdnRecurrentCount18RuntimeReceiptV1, MetalW8Error> {
            let mut raw = RawRuntimeReceiptV1::default();
            let mut error = [0 as c_char; ERROR_CAPACITY];
            let status = unsafe {
                apxinf_metal_gdn_recurrent_count18_profile_v1_receipt(
                    self.0.as_ptr(),
                    expected.selector(),
                    &mut raw,
                    error.as_mut_ptr(),
                    error.len(),
                )
            };
            if status != 0 {
                return Err(bridge_error("read Metal GDN recurrent receipt", &error));
            }
            convert_and_validate_receipt(raw, expected)
        }

        pub(super) fn invalid_raw_selectors_are_rejected_without_mutation(&self) -> bool {
            let before =
                GdnRecurrentProfileV1::ALL.map(|profile| self.runtime_receipt(profile).ok());
            let before_snapshot = self.snapshot().ok();
            let mut raw = RawRuntimeReceiptV1::default();
            let mut error = [0 as c_char; ERROR_CAPACITY];
            let receipt_rejected = unsafe {
                apxinf_metal_gdn_recurrent_count18_profile_v1_receipt(
                    self.0.as_ptr(),
                    99,
                    &mut raw,
                    error.as_mut_ptr(),
                    error.len(),
                ) != 0
            };
            error.fill(0);
            let run_rejected = unsafe {
                apxinf_metal_gdn_recurrent_count18_profile_v1_run(
                    self.0.as_ptr(),
                    99,
                    error.as_mut_ptr(),
                    error.len(),
                ) != 0
            };
            let after =
                GdnRecurrentProfileV1::ALL.map(|profile| self.runtime_receipt(profile).ok());
            let after_snapshot = self.snapshot().ok();
            receipt_rejected
                && run_rejected
                && before.iter().all(Option::is_some)
                && before == after
                && optional_snapshots_match_to_bits(&before_snapshot, &after_snapshot)
        }
    }

    impl Drop for Handle {
        fn drop(&mut self) {
            unsafe { apxinf_metal_gdn_recurrent_count18_profile_v1_destroy(self.0.as_ptr()) };
        }
    }

    fn convert_and_validate_receipt(
        raw: RawRuntimeReceiptV1,
        expected: GdnRecurrentProfileV1,
    ) -> Result<GdnRecurrentCount18RuntimeReceiptV1, MetalW8Error> {
        let requested_profile = GdnRecurrentProfileV1::try_from(raw.requested_profile)?;
        let observed_profile = GdnRecurrentProfileV1::try_from(raw.observed_profile)?;
        let requested_function_name = c_string(&raw.requested_function_name);
        let observed_function_name = c_string(&raw.observed_function_name);
        let expected_last = u32::from(raw.successful_runs != 0);
        let static_valid = if expected == GdnRecurrentProfileV1::Legacy256 {
            raw.pipeline_static_threadgroup_memory_bytes == 0
        } else {
            raw.pipeline_static_threadgroup_memory_bytes
                >= expected.source_declared_threadgroup_memory_bytes()
                && raw.pipeline_static_threadgroup_memory_bytes <= 32768
        };
        let expected_launched = QWEN35_GDN_RECURRENT_SEAMS_PER_DECODE_V1 as u32
            * QWEN35_GDN_VALUE_HEADS_V1 as u32
            * expected.threads_per_threadgroup();
        let expected_active = QWEN35_GDN_RECURRENT_SEAMS_PER_DECODE_V1 as u32
            * QWEN35_GDN_VALUE_HEADS_V1 as u32
            * QWEN35_GDN_VALUE_DIM_V1 as u32;
        let expected_idle = expected_launched - expected_active;
        let processed_bytes = (PROCESSED_TRACE_ELEMENTS * std::mem::size_of::<f32>()) as u64;
        let projected_bytes = (PROJECTED_TRACE_ELEMENTS * std::mem::size_of::<f32>()) as u64;
        let head_scalar_bytes = (HEAD_SCALAR_TRACE_ELEMENTS * std::mem::size_of::<f32>()) as u64;
        let recurrent_bytes = (RECURRENT_TRACE_ELEMENTS * std::mem::size_of::<f32>()) as u64;
        let core_bytes = (CORE_TRACE_ELEMENTS * std::mem::size_of::<f32>()) as u64;
        let persistent_bytes = processed_bytes
            + projected_bytes
            + 2 * head_scalar_bytes
            + 2 * recurrent_bytes
            + core_bytes;
        if requested_profile != expected
            || observed_profile != expected
            || requested_function_name != expected.expected_function_name()
            || observed_function_name != expected.expected_function_name()
            || raw.seams_per_run != QWEN35_GDN_RECURRENT_SEAMS_PER_DECODE_V1 as u32
            || raw.key_heads != QWEN35_GDN_KEY_HEADS_V1 as u32
            || raw.value_heads != QWEN35_GDN_VALUE_HEADS_V1 as u32
            || raw.key_dim != QWEN35_GDN_KEY_DIM_V1 as u32
            || raw.value_dim != QWEN35_GDN_VALUE_DIM_V1 as u32
            || raw.processed_elements_per_seam != QWEN35_GDN_PROCESSED_ELEMENTS_PER_SEAM_V1 as u32
            || raw.projected_elements_per_seam != QWEN35_GDN_PROJECTED_ELEMENTS_PER_SEAM_V1 as u32
            || raw.recurrent_elements_per_seam != QWEN35_GDN_RECURRENT_ELEMENTS_PER_SEAM_V1 as u32
            || raw.core_elements_per_seam != QWEN35_GDN_CORE_ELEMENTS_PER_SEAM_V1 as u32
            || raw.threads_per_threadgroup != expected.threads_per_threadgroup()
            || raw.simdgroups_per_threadgroup != expected.threads_per_threadgroup() / 32
            || raw.pipeline_max_total_threads_per_threadgroup < expected.threads_per_threadgroup()
            || raw.pipeline_thread_execution_width != 32
            || !static_valid
            || raw.source_declared_threadgroup_memory_bytes
                != expected.source_declared_threadgroup_memory_bytes()
            || raw.dynamic_threadgroup_memory_bytes != 0
            || raw.internal_threadgroup_barrier_sites_per_threadgroup
                != expected.internal_threadgroup_barrier_sites()
            || raw.source_derived_internal_barrier_executions_per_run
                != expected.internal_threadgroup_barrier_sites()
                    * QWEN35_GDN_RECURRENT_SEAMS_PER_DECODE_V1 as u32
                    * QWEN35_GDN_VALUE_HEADS_V1 as u32
            || raw.launched_threads_per_run != expected_launched
            || raw.active_value_threads_per_run != expected_active
            || raw.idle_threads_per_run != expected_idle
            || raw.command_buffers_per_run != 1
            || raw.compute_encoders_per_run != 1
            || raw.kernel_dispatches_per_run != 18
            || raw.threadgroups_per_run != 288
            || raw.explicit_buffer_barriers_per_run != 18
            || raw.commits_per_run != 1
            || raw.waits_per_run != 1
            || raw.fixed_shape_host_validated != 1
            || raw.input_output_buffers_non_overlapping != 1
            || raw.host_to_device_bytes_per_run != 0
            || raw.device_to_host_bytes_per_run != 0
            || raw.processed_buffer_bytes != processed_bytes
            || raw.projected_buffer_bytes != projected_bytes
            || raw.a_log_buffer_bytes != head_scalar_bytes
            || raw.dt_bias_buffer_bytes != head_scalar_bytes
            || raw.state_buffer_bytes != recurrent_bytes
            || raw.next_state_buffer_bytes != recurrent_bytes
            || raw.core_buffer_bytes != core_bytes
            || raw.persistent_buffer_bytes_total != persistent_bytes
            || raw.last_observed_command_buffers != expected_last
            || raw.last_observed_compute_encoders != expected_last
            || raw.last_observed_kernel_dispatches != expected_last * 18
            || raw.last_observed_threadgroups != expected_last * 288
            || raw.last_observed_explicit_buffer_barriers != expected_last * 18
            || raw.last_observed_launched_threads != expected_last * expected_launched
            || raw.last_observed_active_value_threads != expected_last * expected_active
            || raw.last_observed_idle_threads != expected_last * expected_idle
            || raw.last_observed_commits != expected_last
            || raw.last_observed_waits != expected_last
        {
            return Err(MetalW8Error::new(format!(
                "invalid live Metal GDN recurrent count18 receipt for {expected:?}"
            )));
        }
        Ok(GdnRecurrentCount18RuntimeReceiptV1 {
            requested_profile,
            observed_profile,
            requested_function_name,
            observed_function_name,
            seams_per_run: raw.seams_per_run,
            key_heads: raw.key_heads,
            value_heads: raw.value_heads,
            key_dim: raw.key_dim,
            value_dim: raw.value_dim,
            processed_elements_per_seam: raw.processed_elements_per_seam,
            projected_elements_per_seam: raw.projected_elements_per_seam,
            recurrent_elements_per_seam: raw.recurrent_elements_per_seam,
            core_elements_per_seam: raw.core_elements_per_seam,
            threads_per_threadgroup: raw.threads_per_threadgroup,
            simdgroups_per_threadgroup: raw.simdgroups_per_threadgroup,
            pipeline_max_total_threads_per_threadgroup: raw
                .pipeline_max_total_threads_per_threadgroup,
            pipeline_thread_execution_width: raw.pipeline_thread_execution_width,
            pipeline_static_threadgroup_memory_bytes: raw.pipeline_static_threadgroup_memory_bytes,
            source_declared_threadgroup_memory_bytes: raw.source_declared_threadgroup_memory_bytes,
            dynamic_threadgroup_memory_bytes: raw.dynamic_threadgroup_memory_bytes,
            internal_threadgroup_barrier_sites_per_threadgroup: raw
                .internal_threadgroup_barrier_sites_per_threadgroup,
            source_derived_internal_barrier_executions_per_run: raw
                .source_derived_internal_barrier_executions_per_run,
            launched_threads_per_run: raw.launched_threads_per_run,
            active_value_threads_per_run: raw.active_value_threads_per_run,
            idle_threads_per_run: raw.idle_threads_per_run,
            command_buffers_per_run: raw.command_buffers_per_run,
            compute_encoders_per_run: raw.compute_encoders_per_run,
            kernel_dispatches_per_run: raw.kernel_dispatches_per_run,
            threadgroups_per_run: raw.threadgroups_per_run,
            explicit_buffer_barriers_per_run: raw.explicit_buffer_barriers_per_run,
            commits_per_run: raw.commits_per_run,
            waits_per_run: raw.waits_per_run,
            fixed_shape_host_validated: raw.fixed_shape_host_validated == 1,
            input_output_buffers_non_overlapping: raw.input_output_buffers_non_overlapping == 1,
            host_to_device_bytes_per_run: raw.host_to_device_bytes_per_run,
            device_to_host_bytes_per_run: raw.device_to_host_bytes_per_run,
            processed_buffer_bytes: raw.processed_buffer_bytes,
            projected_buffer_bytes: raw.projected_buffer_bytes,
            a_log_buffer_bytes: raw.a_log_buffer_bytes,
            dt_bias_buffer_bytes: raw.dt_bias_buffer_bytes,
            state_buffer_bytes: raw.state_buffer_bytes,
            next_state_buffer_bytes: raw.next_state_buffer_bytes,
            core_buffer_bytes: raw.core_buffer_bytes,
            persistent_buffer_bytes_total: raw.persistent_buffer_bytes_total,
            successful_runs: raw.successful_runs,
            last_observed_command_buffers: raw.last_observed_command_buffers,
            last_observed_compute_encoders: raw.last_observed_compute_encoders,
            last_observed_kernel_dispatches: raw.last_observed_kernel_dispatches,
            last_observed_threadgroups: raw.last_observed_threadgroups,
            last_observed_explicit_buffer_barriers: raw.last_observed_explicit_buffer_barriers,
            last_observed_launched_threads: raw.last_observed_launched_threads,
            last_observed_active_value_threads: raw.last_observed_active_value_threads,
            last_observed_idle_threads: raw.last_observed_idle_threads,
            last_observed_commits: raw.last_observed_commits,
            last_observed_waits: raw.last_observed_waits,
        })
    }

    fn c_string(raw: &[c_char; FUNCTION_NAME_CAPACITY]) -> String {
        unsafe { CStr::from_ptr(raw.as_ptr()) }
            .to_string_lossy()
            .into_owned()
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

    #[cfg(test)]
    pub(super) fn raw_receipt_size() -> usize {
        std::mem::size_of::<RawRuntimeReceiptV1>()
    }
}

#[cfg(not(target_os = "macos"))]
mod platform {
    use super::*;

    pub(super) struct Handle;

    impl Handle {
        pub(super) fn new() -> Result<Self, MetalW8Error> {
            Err(MetalW8Error::new(
                "Metal GDN recurrent count18 primitive requires macOS",
            ))
        }

        pub(super) fn stage_fixture(
            &mut self,
            _processed: &[f32],
            _projected: &[f32],
            _a_log: &[f32],
            _dt_bias: &[f32],
            _state: &[f32],
        ) -> Result<(), MetalW8Error> {
            Err(MetalW8Error::new(
                "Metal GDN recurrent count18 primitive requires macOS",
            ))
        }

        pub(super) fn poison_outputs_for_correctness(&mut self) -> Result<(), MetalW8Error> {
            Err(MetalW8Error::new(
                "Metal GDN recurrent count18 primitive requires macOS",
            ))
        }

        pub(super) fn verify_staged_fixture_unchanged(
            &self,
            _processed: &[f32],
            _projected: &[f32],
            _a_log: &[f32],
            _dt_bias: &[f32],
            _state: &[f32],
        ) -> Result<(), MetalW8Error> {
            Err(MetalW8Error::new(
                "Metal GDN recurrent count18 primitive requires macOS",
            ))
        }

        pub(super) fn run(&mut self, _profile: GdnRecurrentProfileV1) -> Result<(), MetalW8Error> {
            Err(MetalW8Error::new(
                "Metal GDN recurrent count18 primitive requires macOS",
            ))
        }

        pub(super) fn snapshot(&self) -> Result<GdnRecurrentCount18SnapshotV1, MetalW8Error> {
            Err(MetalW8Error::new(
                "Metal GDN recurrent count18 primitive requires macOS",
            ))
        }

        pub(super) fn runtime_receipt(
            &self,
            _profile: GdnRecurrentProfileV1,
        ) -> Result<GdnRecurrentCount18RuntimeReceiptV1, MetalW8Error> {
            Err(MetalW8Error::new(
                "Metal GDN recurrent count18 primitive requires macOS",
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
    fn invalid_selector_snapshot_custody_distinguishes_signed_zero_bits() {
        let positive = Some(GdnRecurrentCount18SnapshotV1 {
            next_state: vec![0.0],
            core: vec![-0.0],
        });
        let negative = Some(GdnRecurrentCount18SnapshotV1 {
            next_state: vec![-0.0],
            core: vec![0.0],
        });
        assert!(!optional_snapshots_match_to_bits(&positive, &negative));
        assert!(optional_snapshots_match_to_bits(&positive, &positive));
        assert!(optional_snapshots_match_to_bits(&None, &None));
    }

    #[test]
    fn candidate_is_additive_and_production_bridges_remain_legacy() {
        let shader = include_str!("metal_w8_gdn.metal");
        assert!(shader.contains("kernel void gdn_recurrent_update("));
        assert!(shader.contains("kernel void gdn_recurrent_update_leader_broadcast_v1("));
        assert!(shader.contains("kernel void gdn_recurrent_update_qk_staged_v1("));
        let leader = shader
            .split("kernel void gdn_recurrent_update_leader_broadcast_v1(")
            .nth(1)
            .unwrap()
            .split("kernel void gdn_recurrent_update_qk_staged_v1(")
            .next()
            .unwrap();
        let staged = shader
            .split("kernel void gdn_recurrent_update_qk_staged_v1(")
            .nth(1)
            .unwrap()
            .split("kernel void gdn_core_fused_v1(")
            .next()
            .unwrap();
        assert_eq!(leader.matches("threadgroup_barrier").count(), 1);
        assert_eq!(staged.matches("threadgroup_barrier").count(), 1);
        assert!(leader.contains("threadgroup float shared_scalars[2]"));
        assert!(staged.contains("threadgroup float shared_query[128]"));
        assert!(staged.contains("threadgroup float shared_key[128]"));
        assert!(!leader.contains("params.key_dim !="));
        assert!(!staged.contains("params.key_dim !="));
        assert!(!leader.contains("params.value_dim !="));
        assert!(!staged.contains("params.value_dim !="));
        assert!(!leader.contains("thread_count.x !="));
        assert!(!staged.contains("thread_count.x !="));
        for bridge in [
            include_str!("metal_w8_gdn_bridge.mm"),
            include_str!("metal_w8_linear_layer_bridge.mm"),
            include_str!("metal_w8_linear_layer_stack3_bridge.mm"),
            include_str!("metal_w8_mlp_stack3_boundary_v1_bridge.mm"),
        ] {
            assert!(bridge.contains("@\"gdn_recurrent_update\""));
            assert!(!bridge.contains("gdn_recurrent_update_leader_broadcast_v1"));
            assert!(!bridge.contains("gdn_recurrent_update_qk_staged_v1"));
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn count18_candidates_match_legacy_core_and_state_to_bits() {
        fn values(count: usize, seed: u64, scale: f32) -> Vec<f32> {
            let mut state = seed;
            (0..count)
                .map(|_| {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    let signed = ((state >> 32) % 2001) as i32 - 1000;
                    signed as f32 * scale
                })
                .collect()
        }
        let processed = values(
            PROCESSED_TRACE_ELEMENTS,
            0x243f_6a88_85a3_08d3,
            1.0 / 65536.0,
        );
        let projected = values(
            PROJECTED_TRACE_ELEMENTS,
            0x1319_8a2e_0370_7344,
            1.0 / 1024.0,
        );
        let a_log = values(
            HEAD_SCALAR_TRACE_ELEMENTS,
            0xa409_3822_299f_31d0,
            1.0 / 4096.0,
        );
        let dt_bias = values(
            HEAD_SCALAR_TRACE_ELEMENTS,
            0x082e_fa98_ec4e_6c89,
            1.0 / 4096.0,
        );
        let state = values(
            RECURRENT_TRACE_ELEMENTS,
            0x4528_21e6_38d0_1377,
            1.0 / 65536.0,
        );
        let mut primitive = MetalGdnRecurrentCount18PrimitiveV1::new().unwrap();
        primitive.verify_invalid_raw_selector_fail_closed().unwrap();
        primitive
            .stage_fixture(&processed, &projected, &a_log, &dt_bias, &state)
            .unwrap();
        primitive.verify_invalid_raw_selector_fail_closed().unwrap();
        assert!(primitive.snapshot().is_err());
        let mut snapshots = Vec::new();
        for profile in GdnRecurrentProfileV1::ALL {
            primitive.poison_outputs_for_correctness().unwrap();
            primitive.run(profile).unwrap();
            snapshots.push(primitive.snapshot().unwrap());
        }
        for (profile_index, actual) in snapshots.iter().enumerate().skip(1) {
            for (label, expected, actual) in [
                ("next_state", &snapshots[0].next_state, &actual.next_state),
                ("core", &snapshots[0].core, &actual.core),
            ] {
                for (index, (&left, &right)) in expected.iter().zip(actual).enumerate() {
                    assert!(
                        left.is_finite() && right.is_finite(),
                        "{label} non-finite for profile {profile_index} at {index}"
                    );
                    assert_eq!(
                        left.to_bits(),
                        right.to_bits(),
                        "{label} mismatch for profile {profile_index} at {index}"
                    );
                }
            }
        }
        primitive
            .verify_staged_fixture_unchanged(&processed, &projected, &a_log, &dt_bias, &state)
            .unwrap();
        assert_eq!(platform::raw_receipt_size(), 384);
        primitive.verify_invalid_raw_selector_fail_closed().unwrap();
        for profile in GdnRecurrentProfileV1::ALL {
            let receipt = primitive.runtime_receipt(profile).unwrap();
            assert_eq!(receipt.successful_runs, 1);
        }
    }
}
