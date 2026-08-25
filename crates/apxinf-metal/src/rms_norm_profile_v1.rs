use crate::MetalW8Error;

pub const QWEN35_RMS_HIDDEN_SIZE_V1: usize = 1024;
pub const QWEN35_RMS_CALLS_PER_DECODE_V1: usize = 43;

/// Explicit selector for the count-matched Qwen3.5 RMSNorm mechanism screen.
/// Existing production constructors remain bound to [`Self::LegacySharedTree`].
#[repr(u32)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum RmsNormReductionProfileV1 {
    #[default]
    LegacySharedTree = 0,
    ExactRedundantSimdTail = 1,
}

impl RmsNormReductionProfileV1 {
    pub const fn selector(self) -> u32 {
        self as u32
    }

    pub const fn expected_function_name(self) -> &'static str {
        match self {
            Self::LegacySharedTree => "linear_layer_rms_norm",
            Self::ExactRedundantSimdTail => "linear_layer_rms_norm_simd_tail_exact_v1",
        }
    }

    pub const fn internal_threadgroup_barriers_per_dispatch(self) -> u32 {
        match self {
            Self::LegacySharedTree => 9,
            Self::ExactRedundantSimdTail => 4,
        }
    }

    const fn from_selector(selector: u32) -> Option<Self> {
        match selector {
            0 => Some(Self::LegacySharedTree),
            1 => Some(Self::ExactRedundantSimdTail),
            _ => None,
        }
    }
}

impl TryFrom<u32> for RmsNormReductionProfileV1 {
    type Error = MetalW8Error;

    fn try_from(selector: u32) -> Result<Self, Self::Error> {
        Self::from_selector(selector).ok_or_else(|| {
            MetalW8Error::new(format!(
                "Metal RMSNorm reduction profile {selector} is invalid; expected 0 or 1"
            ))
        })
    }
}

/// Live identity and actual call-site topology for one arm of the additive
/// count-43 RMSNorm screen. Internal barrier counts are source-derived; the
/// `last_observed_*` fields are counters incremented at bridge call sites and
/// published only after successful GPU completion.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RmsNormCount43RuntimeReceiptV1 {
    pub requested_profile: RmsNormReductionProfileV1,
    pub observed_profile: RmsNormReductionProfileV1,
    pub requested_function_name: String,
    pub observed_function_name: String,
    pub hidden_size: u32,
    pub rms_calls_per_run: u32,
    pub threads_per_threadgroup: u32,
    pub simdgroups_per_threadgroup: u32,
    pub pipeline_max_total_threads_per_threadgroup: u32,
    pub pipeline_thread_execution_width: u32,
    pub static_threadgroup_memory_bytes: u32,
    pub dynamic_threadgroup_memory_bytes: u32,
    pub internal_threadgroup_barriers_per_dispatch: u32,
    pub internal_threadgroup_barriers_per_run: u32,
    pub command_buffers_per_run: u32,
    pub compute_encoders_per_run: u32,
    pub kernel_dispatches_per_run: u32,
    pub explicit_buffer_barriers_per_run: u32,
    pub commits_per_run: u32,
    pub waits_per_run: u32,
    pub host_to_device_bytes_per_run: u64,
    pub device_to_host_bytes_per_run: u64,
    pub successful_runs: u64,
    pub last_observed_command_buffers: u32,
    pub last_observed_compute_encoders: u32,
    pub last_observed_kernel_dispatches: u32,
    pub last_observed_explicit_buffer_barriers: u32,
    pub last_observed_commits: u32,
    pub last_observed_waits: u32,
}

/// Synthetic production-H primitive that owns both RMS kernels in one Metal
/// library. Each `run` submits exactly 43 dependent RMS dispatches. Input
/// staging and correctness snapshots are explicit and remain outside timing.
pub struct MetalRmsNormCount43PrimitiveV1 {
    inner: platform::Handle,
}

impl MetalRmsNormCount43PrimitiveV1 {
    pub fn new(weight: &[f32], rms_norm_eps: f32) -> Result<Self, MetalW8Error> {
        if weight.len() != QWEN35_RMS_HIDDEN_SIZE_V1 {
            return Err(MetalW8Error::new(format!(
                "Metal RMSNorm count43 weight has {} elements, expected {}",
                weight.len(),
                QWEN35_RMS_HIDDEN_SIZE_V1
            )));
        }
        if let Some(index) = weight.iter().position(|value| !value.is_finite()) {
            return Err(MetalW8Error::new(format!(
                "Metal RMSNorm count43 weight contains a non-finite value at element {index}"
            )));
        }
        if !rms_norm_eps.is_finite() || rms_norm_eps < 0.0 {
            return Err(MetalW8Error::new(
                "Metal RMSNorm count43 epsilon must be finite and non-negative",
            ));
        }
        Ok(Self {
            inner: platform::Handle::new(weight, rms_norm_eps)?,
        })
    }

    pub fn stage_input(&mut self, input: &[f32]) -> Result<(), MetalW8Error> {
        if input.len() != QWEN35_RMS_HIDDEN_SIZE_V1 {
            return Err(MetalW8Error::new(format!(
                "Metal RMSNorm count43 input has {} elements, expected {}",
                input.len(),
                QWEN35_RMS_HIDDEN_SIZE_V1
            )));
        }
        if let Some(index) = input.iter().position(|value| !value.is_finite()) {
            return Err(MetalW8Error::new(format!(
                "Metal RMSNorm count43 input contains a non-finite value at element {index}"
            )));
        }
        self.inner.stage_input(input)
    }

    pub fn run(&mut self, profile: RmsNormReductionProfileV1) -> Result<(), MetalW8Error> {
        self.inner.run(profile)
    }

    pub fn snapshot_chain(&self) -> Result<Vec<f32>, MetalW8Error> {
        self.inner.snapshot_chain()
    }

    pub fn runtime_receipt(
        &self,
        profile: RmsNormReductionProfileV1,
    ) -> Result<RmsNormCount43RuntimeReceiptV1, MetalW8Error> {
        self.inner.runtime_receipt(profile)
    }
}

#[cfg(target_os = "macos")]
mod platform {
    use super::{
        MetalW8Error, RmsNormCount43RuntimeReceiptV1, RmsNormReductionProfileV1,
        QWEN35_RMS_CALLS_PER_DECODE_V1, QWEN35_RMS_HIDDEN_SIZE_V1,
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
        rms_calls_per_run: u32,
        threads_per_threadgroup: u32,
        simdgroups_per_threadgroup: u32,
        pipeline_max_total_threads_per_threadgroup: u32,
        pipeline_thread_execution_width: u32,
        static_threadgroup_memory_bytes: u32,
        dynamic_threadgroup_memory_bytes: u32,
        internal_threadgroup_barriers_per_dispatch: u32,
        command_buffers_per_run: u32,
        compute_encoders_per_run: u32,
        kernel_dispatches_per_run: u32,
        explicit_buffer_barriers_per_run: u32,
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
                hidden_size: 0,
                rms_calls_per_run: 0,
                threads_per_threadgroup: 0,
                simdgroups_per_threadgroup: 0,
                pipeline_max_total_threads_per_threadgroup: 0,
                pipeline_thread_execution_width: 0,
                static_threadgroup_memory_bytes: 0,
                dynamic_threadgroup_memory_bytes: 0,
                internal_threadgroup_barriers_per_dispatch: 0,
                command_buffers_per_run: 0,
                compute_encoders_per_run: 0,
                kernel_dispatches_per_run: 0,
                explicit_buffer_barriers_per_run: 0,
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
                last_observed_commits: 0,
                last_observed_waits: 0,
                requested_function_name: [0; FUNCTION_NAME_CAPACITY],
                observed_function_name: [0; FUNCTION_NAME_CAPACITY],
            }
        }
    }

    extern "C" {
        fn apxinf_metal_rms_norm_count43_profile_v1_create(
            weight: *const f32,
            weight_count: u32,
            rms_norm_eps: f32,
            output: *mut *mut c_void,
            error: *mut c_char,
            error_capacity: usize,
        ) -> c_int;
        fn apxinf_metal_rms_norm_count43_profile_v1_stage_input(
            handle: *mut c_void,
            input: *const f32,
            input_count: u32,
            error: *mut c_char,
            error_capacity: usize,
        ) -> c_int;
        fn apxinf_metal_rms_norm_count43_profile_v1_run(
            handle: *mut c_void,
            profile: u32,
            error: *mut c_char,
            error_capacity: usize,
        ) -> c_int;
        fn apxinf_metal_rms_norm_count43_profile_v1_snapshot_chain(
            handle: *mut c_void,
            output: *mut f32,
            output_count: u32,
            error: *mut c_char,
            error_capacity: usize,
        ) -> c_int;
        fn apxinf_metal_rms_norm_count43_profile_v1_receipt(
            handle: *mut c_void,
            profile: u32,
            receipt: *mut RawRuntimeReceiptV1,
            error: *mut c_char,
            error_capacity: usize,
        ) -> c_int;
        fn apxinf_metal_rms_norm_count43_profile_v1_destroy(handle: *mut c_void);
    }

    pub(super) struct Handle(NonNull<c_void>);

    impl Handle {
        pub(super) fn new(weight: &[f32], rms_norm_eps: f32) -> Result<Self, MetalW8Error> {
            let mut output = std::ptr::null_mut();
            let mut error = [0 as c_char; ERROR_CAPACITY];
            let status = unsafe {
                apxinf_metal_rms_norm_count43_profile_v1_create(
                    weight.as_ptr(),
                    weight.len() as u32,
                    rms_norm_eps,
                    &mut output,
                    error.as_mut_ptr(),
                    error.len(),
                )
            };
            if status != 0 {
                return Err(bridge_error(
                    "create Metal RMSNorm count43 primitive",
                    &error,
                ));
            }
            let handle = Self(NonNull::new(output).ok_or_else(|| {
                MetalW8Error::new("create Metal RMSNorm count43 primitive returned a null handle")
            })?);
            for profile in [
                RmsNormReductionProfileV1::LegacySharedTree,
                RmsNormReductionProfileV1::ExactRedundantSimdTail,
            ] {
                let receipt = handle.runtime_receipt(profile)?;
                if receipt.successful_runs != 0 {
                    return Err(MetalW8Error::new(
                        "new Metal RMSNorm count43 primitive reported successful runs",
                    ));
                }
            }
            Ok(handle)
        }

        pub(super) fn stage_input(&mut self, input: &[f32]) -> Result<(), MetalW8Error> {
            let mut error = [0 as c_char; ERROR_CAPACITY];
            let status = unsafe {
                apxinf_metal_rms_norm_count43_profile_v1_stage_input(
                    self.0.as_ptr(),
                    input.as_ptr(),
                    input.len() as u32,
                    error.as_mut_ptr(),
                    error.len(),
                )
            };
            if status != 0 {
                return Err(bridge_error("stage Metal RMSNorm count43 input", &error));
            }
            Ok(())
        }

        pub(super) fn run(
            &mut self,
            profile: RmsNormReductionProfileV1,
        ) -> Result<(), MetalW8Error> {
            let mut error = [0 as c_char; ERROR_CAPACITY];
            let status = unsafe {
                apxinf_metal_rms_norm_count43_profile_v1_run(
                    self.0.as_ptr(),
                    profile.selector(),
                    error.as_mut_ptr(),
                    error.len(),
                )
            };
            if status != 0 {
                return Err(bridge_error("run Metal RMSNorm count43 primitive", &error));
            }
            Ok(())
        }

        pub(super) fn snapshot_chain(&self) -> Result<Vec<f32>, MetalW8Error> {
            let mut output =
                vec![0.0f32; QWEN35_RMS_HIDDEN_SIZE_V1 * QWEN35_RMS_CALLS_PER_DECODE_V1];
            let mut error = [0 as c_char; ERROR_CAPACITY];
            let status = unsafe {
                apxinf_metal_rms_norm_count43_profile_v1_snapshot_chain(
                    self.0.as_ptr(),
                    output.as_mut_ptr(),
                    output.len() as u32,
                    error.as_mut_ptr(),
                    error.len(),
                )
            };
            if status != 0 {
                return Err(bridge_error("snapshot Metal RMSNorm count43 chain", &error));
            }
            Ok(output)
        }

        pub(super) fn runtime_receipt(
            &self,
            expected_profile: RmsNormReductionProfileV1,
        ) -> Result<RmsNormCount43RuntimeReceiptV1, MetalW8Error> {
            let mut raw = RawRuntimeReceiptV1::default();
            let mut error = [0 as c_char; ERROR_CAPACITY];
            let status = unsafe {
                apxinf_metal_rms_norm_count43_profile_v1_receipt(
                    self.0.as_ptr(),
                    expected_profile.selector(),
                    &mut raw,
                    error.as_mut_ptr(),
                    error.len(),
                )
            };
            if status != 0 {
                return Err(bridge_error("read Metal RMSNorm count43 receipt", &error));
            }
            convert_and_validate_receipt(raw, expected_profile)
        }
    }

    impl Drop for Handle {
        fn drop(&mut self) {
            unsafe { apxinf_metal_rms_norm_count43_profile_v1_destroy(self.0.as_ptr()) };
        }
    }

    fn convert_and_validate_receipt(
        raw: RawRuntimeReceiptV1,
        expected: RmsNormReductionProfileV1,
    ) -> Result<RmsNormCount43RuntimeReceiptV1, MetalW8Error> {
        let requested_profile = RmsNormReductionProfileV1::try_from(raw.requested_profile)?;
        let observed_profile = RmsNormReductionProfileV1::try_from(raw.observed_profile)?;
        let requested_function_name =
            unsafe { CStr::from_ptr(raw.requested_function_name.as_ptr()) }
                .to_string_lossy()
                .into_owned();
        let observed_function_name = unsafe { CStr::from_ptr(raw.observed_function_name.as_ptr()) }
            .to_string_lossy()
            .into_owned();
        let expected_name = expected.expected_function_name();
        let expected_last = if raw.successful_runs == 0 { 0 } else { 1 };
        if requested_profile != expected
            || observed_profile != expected
            || requested_function_name != expected_name
            || observed_function_name != expected_name
            || raw.hidden_size != QWEN35_RMS_HIDDEN_SIZE_V1 as u32
            || raw.rms_calls_per_run != QWEN35_RMS_CALLS_PER_DECODE_V1 as u32
            || raw.threads_per_threadgroup != 256
            || raw.simdgroups_per_threadgroup != 8
            || raw.pipeline_thread_execution_width != 32
            || raw.pipeline_max_total_threads_per_threadgroup < 256
            || raw.static_threadgroup_memory_bytes != 1024
            || raw.dynamic_threadgroup_memory_bytes != 0
            || raw.internal_threadgroup_barriers_per_dispatch
                != expected.internal_threadgroup_barriers_per_dispatch()
            || raw.command_buffers_per_run != 1
            || raw.compute_encoders_per_run != 1
            || raw.kernel_dispatches_per_run != QWEN35_RMS_CALLS_PER_DECODE_V1 as u32
            || raw.explicit_buffer_barriers_per_run != QWEN35_RMS_CALLS_PER_DECODE_V1 as u32
            || raw.commits_per_run != 1
            || raw.waits_per_run != 1
            || raw.host_to_device_bytes_per_run != 0
            || raw.device_to_host_bytes_per_run != 0
            || raw.last_observed_command_buffers != expected_last
            || raw.last_observed_compute_encoders != expected_last
            || raw.last_observed_kernel_dispatches
                != expected_last * QWEN35_RMS_CALLS_PER_DECODE_V1 as u32
            || raw.last_observed_explicit_buffer_barriers
                != expected_last * QWEN35_RMS_CALLS_PER_DECODE_V1 as u32
            || raw.last_observed_commits != expected_last
            || raw.last_observed_waits != expected_last
        {
            return Err(MetalW8Error::new(format!(
                "invalid live Metal RMSNorm count43 receipt for {expected:?}"
            )));
        }
        Ok(RmsNormCount43RuntimeReceiptV1 {
            requested_profile,
            observed_profile,
            requested_function_name,
            observed_function_name,
            hidden_size: raw.hidden_size,
            rms_calls_per_run: raw.rms_calls_per_run,
            threads_per_threadgroup: raw.threads_per_threadgroup,
            simdgroups_per_threadgroup: raw.simdgroups_per_threadgroup,
            pipeline_max_total_threads_per_threadgroup: raw
                .pipeline_max_total_threads_per_threadgroup,
            pipeline_thread_execution_width: raw.pipeline_thread_execution_width,
            static_threadgroup_memory_bytes: raw.static_threadgroup_memory_bytes,
            dynamic_threadgroup_memory_bytes: raw.dynamic_threadgroup_memory_bytes,
            internal_threadgroup_barriers_per_dispatch: raw
                .internal_threadgroup_barriers_per_dispatch,
            internal_threadgroup_barriers_per_run: raw.internal_threadgroup_barriers_per_dispatch
                * raw.rms_calls_per_run,
            command_buffers_per_run: raw.command_buffers_per_run,
            compute_encoders_per_run: raw.compute_encoders_per_run,
            kernel_dispatches_per_run: raw.kernel_dispatches_per_run,
            explicit_buffer_barriers_per_run: raw.explicit_buffer_barriers_per_run,
            commits_per_run: raw.commits_per_run,
            waits_per_run: raw.waits_per_run,
            host_to_device_bytes_per_run: raw.host_to_device_bytes_per_run,
            device_to_host_bytes_per_run: raw.device_to_host_bytes_per_run,
            successful_runs: raw.successful_runs,
            last_observed_command_buffers: raw.last_observed_command_buffers,
            last_observed_compute_encoders: raw.last_observed_compute_encoders,
            last_observed_kernel_dispatches: raw.last_observed_kernel_dispatches,
            last_observed_explicit_buffer_barriers: raw.last_observed_explicit_buffer_barriers,
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

    #[cfg(test)]
    pub(super) fn invalid_raw_selectors_are_rejected_without_submission(handle: &Handle) -> bool {
        let before_a = handle
            .runtime_receipt(RmsNormReductionProfileV1::LegacySharedTree)
            .ok();
        let before_b = handle
            .runtime_receipt(RmsNormReductionProfileV1::ExactRedundantSimdTail)
            .ok();
        let mut raw = RawRuntimeReceiptV1::default();
        let mut error = [0 as c_char; ERROR_CAPACITY];
        let receipt_rejected = unsafe {
            apxinf_metal_rms_norm_count43_profile_v1_receipt(
                handle.0.as_ptr(),
                99,
                &mut raw,
                error.as_mut_ptr(),
                error.len(),
            ) != 0
        };
        error.fill(0);
        let run_rejected = unsafe {
            apxinf_metal_rms_norm_count43_profile_v1_run(
                handle.0.as_ptr(),
                99,
                error.as_mut_ptr(),
                error.len(),
            ) != 0
        };
        let after_a = handle
            .runtime_receipt(RmsNormReductionProfileV1::LegacySharedTree)
            .ok();
        let after_b = handle
            .runtime_receipt(RmsNormReductionProfileV1::ExactRedundantSimdTail)
            .ok();
        receipt_rejected
            && run_rejected
            && before_a.is_some()
            && before_b.is_some()
            && before_a == after_a
            && before_b == after_b
    }

    #[cfg(test)]
    pub(super) fn raw_receipt_size() -> usize {
        std::mem::size_of::<RawRuntimeReceiptV1>()
    }
}

#[cfg(not(target_os = "macos"))]
mod platform {
    use super::{MetalW8Error, RmsNormCount43RuntimeReceiptV1, RmsNormReductionProfileV1};

    pub(super) struct Handle;

    impl Handle {
        pub(super) fn new(_weight: &[f32], _rms_norm_eps: f32) -> Result<Self, MetalW8Error> {
            Err(MetalW8Error::new(
                "Metal RMSNorm count43 primitive requires macOS",
            ))
        }

        pub(super) fn stage_input(&mut self, _input: &[f32]) -> Result<(), MetalW8Error> {
            Err(MetalW8Error::new(
                "Metal RMSNorm count43 primitive requires macOS",
            ))
        }

        pub(super) fn run(
            &mut self,
            _profile: RmsNormReductionProfileV1,
        ) -> Result<(), MetalW8Error> {
            Err(MetalW8Error::new(
                "Metal RMSNorm count43 primitive requires macOS",
            ))
        }

        pub(super) fn snapshot_chain(&self) -> Result<Vec<f32>, MetalW8Error> {
            Err(MetalW8Error::new(
                "Metal RMSNorm count43 primitive requires macOS",
            ))
        }

        pub(super) fn runtime_receipt(
            &self,
            _profile: RmsNormReductionProfileV1,
        ) -> Result<RmsNormCount43RuntimeReceiptV1, MetalW8Error> {
            Err(MetalW8Error::new(
                "Metal RMSNorm count43 primitive requires macOS",
            ))
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
        assert!(shader.contains("kernel void linear_layer_rms_norm_simd_tail_exact_v1("));
        let legacy = shader
            .split("kernel void linear_layer_rms_norm(")
            .nth(1)
            .unwrap()
            .split("kernel void linear_layer_rms_norm_simd_tail_exact_v1(")
            .next()
            .unwrap();
        assert_eq!(legacy.matches("threadgroup_barrier").count(), 2);
        assert!(legacy.contains("for (uint stride = 128; stride != 0; stride >>= 1)"));
        let candidate = shader
            .split("kernel void linear_layer_rms_norm_simd_tail_exact_v1(")
            .nth(1)
            .unwrap()
            .split("kernel void linear_layer_residual_add(")
            .next()
            .unwrap();
        assert_eq!(candidate.matches("threadgroup_barrier").count(), 2);
        assert!(candidate.contains("for (uint stride = 128; stride >= 32; stride >>= 1)"));
        assert_eq!(candidate.matches("simd_shuffle_down").count(), 5);
        assert!(candidate.contains("simd_broadcast(simd_sum, 0)"));

        for bridge in [
            include_str!("metal_w8_linear_layer_bridge.mm"),
            include_str!("metal_w8_linear_layer_stack3_bridge.mm"),
            include_str!("metal_w8_mlp_stack3_boundary_v1_bridge.mm"),
            include_str!("metal_w8_tail_mlp_head_v1_bridge.mm"),
        ] {
            assert!(bridge.contains("@\"linear_layer_rms_norm\""));
            assert!(!bridge.contains("linear_layer_rms_norm_simd_tail_exact_v1"));
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn count43_candidate_matches_every_legacy_intermediate_to_bits() {
        let weight = (0..QWEN35_RMS_HIDDEN_SIZE_V1)
            .map(|index| 0.75 + ((index * 37 + 11) % 257) as f32 / 1024.0)
            .collect::<Vec<_>>();
        let input = (0..QWEN35_RMS_HIDDEN_SIZE_V1)
            .map(|index| (((index * 29 + index / 7 + 5) % 401) as f32 - 200.0) / 256.0)
            .collect::<Vec<_>>();
        let mut primitive = MetalRmsNormCount43PrimitiveV1::new(&weight, 1.0e-6).unwrap();
        primitive.stage_input(&input).unwrap();
        assert!(primitive.snapshot_chain().is_err());
        assert!(platform::invalid_raw_selectors_are_rejected_without_submission(&primitive.inner));
        assert!(primitive.snapshot_chain().is_err());
        assert_eq!(platform::raw_receipt_size(), 248);
        primitive
            .run(RmsNormReductionProfileV1::LegacySharedTree)
            .unwrap();
        let legacy = primitive.snapshot_chain().unwrap();
        primitive
            .run(RmsNormReductionProfileV1::ExactRedundantSimdTail)
            .unwrap();
        let candidate = primitive.snapshot_chain().unwrap();
        assert_eq!(legacy.len(), QWEN35_RMS_HIDDEN_SIZE_V1 * 43);
        for (index, (&expected, &actual)) in legacy.iter().zip(&candidate).enumerate() {
            assert!(
                expected.is_finite() && actual.is_finite(),
                "count43 RMSNorm produced a non-finite value at chain element {index}"
            );
            assert_eq!(
                expected.to_bits(),
                actual.to_bits(),
                "count43 RMSNorm mismatch at chain element {index}"
            );
        }
        assert_eq!(
            primitive
                .runtime_receipt(RmsNormReductionProfileV1::LegacySharedTree)
                .unwrap()
                .successful_runs,
            1
        );
        assert_eq!(
            primitive
                .runtime_receipt(RmsNormReductionProfileV1::ExactRedundantSimdTail)
                .unwrap()
                .successful_runs,
            1
        );
    }
}
