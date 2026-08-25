use crate::{GdnDimensions, MetalW8Error};
use std::ffi::c_char;

pub const QWEN35_GDN_CORE_SEAMS_PER_DECODE_V1: usize = 18;
pub const QWEN35_GDN_CORE_HIDDEN_SIZE_V1: usize = 1024;
pub const QWEN35_GDN_CORE_KEY_HEADS_V1: usize = 16;
pub const QWEN35_GDN_CORE_VALUE_HEADS_V1: usize = 16;
pub const QWEN35_GDN_CORE_KEY_DIM_V1: usize = 128;
pub const QWEN35_GDN_CORE_VALUE_DIM_V1: usize = 128;
pub const QWEN35_GDN_CORE_CONV_KERNEL_SIZE_V1: usize = 4;
pub const QWEN35_GDN_CORE_KEY_WIDTH_V1: usize =
    QWEN35_GDN_CORE_KEY_HEADS_V1 * QWEN35_GDN_CORE_KEY_DIM_V1;
pub const QWEN35_GDN_CORE_VALUE_WIDTH_V1: usize =
    QWEN35_GDN_CORE_VALUE_HEADS_V1 * QWEN35_GDN_CORE_VALUE_DIM_V1;
pub const QWEN35_GDN_CORE_QKV_WIDTH_V1: usize =
    2 * QWEN35_GDN_CORE_KEY_WIDTH_V1 + QWEN35_GDN_CORE_VALUE_WIDTH_V1;
pub const QWEN35_GDN_CORE_PROJECTED_ELEMENTS_PER_SEAM_V1: usize = QWEN35_GDN_CORE_QKV_WIDTH_V1
    + QWEN35_GDN_CORE_VALUE_WIDTH_V1
    + 2 * QWEN35_GDN_CORE_VALUE_HEADS_V1;
pub const QWEN35_GDN_CORE_CONV_WEIGHT_ELEMENTS_PER_SEAM_V1: usize =
    QWEN35_GDN_CORE_QKV_WIDTH_V1 * QWEN35_GDN_CORE_CONV_KERNEL_SIZE_V1;
pub const QWEN35_GDN_CORE_QUERY_STATE_ELEMENTS_PER_SEAM_V1: usize =
    QWEN35_GDN_CORE_KEY_WIDTH_V1 * QWEN35_GDN_CORE_CONV_KERNEL_SIZE_V1;
pub const QWEN35_GDN_CORE_KEY_STATE_ELEMENTS_PER_SEAM_V1: usize =
    QWEN35_GDN_CORE_KEY_WIDTH_V1 * QWEN35_GDN_CORE_CONV_KERNEL_SIZE_V1;
pub const QWEN35_GDN_CORE_VALUE_STATE_ELEMENTS_PER_SEAM_V1: usize =
    QWEN35_GDN_CORE_VALUE_WIDTH_V1 * QWEN35_GDN_CORE_CONV_KERNEL_SIZE_V1;
pub const QWEN35_GDN_CORE_HEAD_SCALAR_ELEMENTS_PER_SEAM_V1: usize = QWEN35_GDN_CORE_VALUE_HEADS_V1;
pub const QWEN35_GDN_CORE_RECURRENT_ELEMENTS_PER_SEAM_V1: usize =
    QWEN35_GDN_CORE_VALUE_HEADS_V1 * QWEN35_GDN_CORE_KEY_DIM_V1 * QWEN35_GDN_CORE_VALUE_DIM_V1;
pub const QWEN35_GDN_CORE_NORM_WEIGHT_ELEMENTS_PER_SEAM_V1: usize = QWEN35_GDN_CORE_VALUE_DIM_V1;
pub const QWEN35_GDN_CORE_GATED_ELEMENTS_PER_SEAM_V1: usize = QWEN35_GDN_CORE_VALUE_WIDTH_V1;

pub const QWEN35_GDN_CORE_PROJECTED_TRACE_ELEMENTS_V1: usize =
    QWEN35_GDN_CORE_SEAMS_PER_DECODE_V1 * QWEN35_GDN_CORE_PROJECTED_ELEMENTS_PER_SEAM_V1;
pub const QWEN35_GDN_CORE_CONV_WEIGHT_TRACE_ELEMENTS_V1: usize =
    QWEN35_GDN_CORE_SEAMS_PER_DECODE_V1 * QWEN35_GDN_CORE_CONV_WEIGHT_ELEMENTS_PER_SEAM_V1;
pub const QWEN35_GDN_CORE_QUERY_STATE_TRACE_ELEMENTS_V1: usize =
    QWEN35_GDN_CORE_SEAMS_PER_DECODE_V1 * QWEN35_GDN_CORE_QUERY_STATE_ELEMENTS_PER_SEAM_V1;
pub const QWEN35_GDN_CORE_KEY_STATE_TRACE_ELEMENTS_V1: usize =
    QWEN35_GDN_CORE_SEAMS_PER_DECODE_V1 * QWEN35_GDN_CORE_KEY_STATE_ELEMENTS_PER_SEAM_V1;
pub const QWEN35_GDN_CORE_VALUE_STATE_TRACE_ELEMENTS_V1: usize =
    QWEN35_GDN_CORE_SEAMS_PER_DECODE_V1 * QWEN35_GDN_CORE_VALUE_STATE_ELEMENTS_PER_SEAM_V1;
pub const QWEN35_GDN_CORE_HEAD_SCALAR_TRACE_ELEMENTS_V1: usize =
    QWEN35_GDN_CORE_SEAMS_PER_DECODE_V1 * QWEN35_GDN_CORE_HEAD_SCALAR_ELEMENTS_PER_SEAM_V1;
pub const QWEN35_GDN_CORE_RECURRENT_TRACE_ELEMENTS_V1: usize =
    QWEN35_GDN_CORE_SEAMS_PER_DECODE_V1 * QWEN35_GDN_CORE_RECURRENT_ELEMENTS_PER_SEAM_V1;
pub const QWEN35_GDN_CORE_NORM_WEIGHT_TRACE_ELEMENTS_V1: usize =
    QWEN35_GDN_CORE_SEAMS_PER_DECODE_V1 * QWEN35_GDN_CORE_NORM_WEIGHT_ELEMENTS_PER_SEAM_V1;
pub const QWEN35_GDN_CORE_GATED_TRACE_ELEMENTS_V1: usize =
    QWEN35_GDN_CORE_SEAMS_PER_DECODE_V1 * QWEN35_GDN_CORE_GATED_ELEMENTS_PER_SEAM_V1;
pub const QWEN35_GDN_CORE_RMS_NORM_EPS_BITS_V1: u32 = 1.0e-6f32.to_bits();
pub const QWEN35_GDN_CORE_PIPELINE_THREAD_EXECUTION_WIDTH_V1: u32 = 32;
pub const QWEN35_GDN_CORE_FUSED_PIPELINE_STATIC_THREADGROUP_MEMORY_BYTES_V1: u32 = 2_064;

// Stable aggregate aliases used by the public crate surface. The `_V2`
// suffixes distinguish the wider core-fusion fixture from the earlier
// recurrent-only primitive's public shape constants.
pub const QWEN35_GDN_SEAMS_PER_DECODE_V1: usize = QWEN35_GDN_CORE_SEAMS_PER_DECODE_V1;
pub const QWEN35_GDN_PROJECTED_ELEMENTS_PER_SEAM_V2: usize =
    QWEN35_GDN_CORE_PROJECTED_ELEMENTS_PER_SEAM_V1;
pub const QWEN35_GDN_CONV_ELEMENTS_PER_SEAM_V1: usize = QWEN35_GDN_CORE_QKV_WIDTH_V1;
pub const QWEN35_GDN_CONV_STATE_ELEMENTS_PER_SEAM_V1: usize =
    QWEN35_GDN_CORE_QUERY_STATE_ELEMENTS_PER_SEAM_V1
        + QWEN35_GDN_CORE_KEY_STATE_ELEMENTS_PER_SEAM_V1
        + QWEN35_GDN_CORE_VALUE_STATE_ELEMENTS_PER_SEAM_V1;
pub const QWEN35_GDN_RECURRENT_ELEMENTS_PER_SEAM_V2: usize =
    QWEN35_GDN_CORE_RECURRENT_ELEMENTS_PER_SEAM_V1;
pub const QWEN35_GDN_NORM_WEIGHT_ELEMENTS_PER_SEAM_V1: usize =
    QWEN35_GDN_CORE_NORM_WEIGHT_ELEMENTS_PER_SEAM_V1;
pub const QWEN35_GDN_GATED_ELEMENTS_PER_SEAM_V1: usize = QWEN35_GDN_CORE_GATED_ELEMENTS_PER_SEAM_V1;

const FUNCTION_CHAIN_CAPACITY: usize = 256;
const RAW_RUNTIME_RECEIPT_SIZE: usize = 368;

/// Explicit selector shared by the fixed-shape count-18 screen and the
/// opt-in production continuation. Ordinary constructors remain legacy;
/// production constructors reject the Q/K-staged diagnostic arm.
#[repr(u32)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum GdnCoreProfileV1 {
    #[default]
    LegacyFourDispatch = 0,
    QkStagedFourDispatch = 1,
    Fused128 = 2,
}

impl GdnCoreProfileV1 {
    pub const ALL: [Self; 3] = [
        Self::LegacyFourDispatch,
        Self::QkStagedFourDispatch,
        Self::Fused128,
    ];

    pub const fn selector(self) -> u32 {
        self as u32
    }

    pub const fn expected_function_chain(self) -> &'static str {
        match self {
            Self::LegacyFourDispatch => {
                "gdn_depthwise_preprocess|gdn_normalize_qk|gdn_recurrent_update|gdn_norm_gate"
            }
            Self::QkStagedFourDispatch => "gdn_depthwise_preprocess|gdn_normalize_qk|gdn_recurrent_update_qk_staged_v1|gdn_norm_gate",
            Self::Fused128 => "gdn_core_fused_v1",
        }
    }

    pub const fn kernel_dispatches_per_run(self) -> u32 {
        match self {
            Self::LegacyFourDispatch | Self::QkStagedFourDispatch => 72,
            Self::Fused128 => 18,
        }
    }

    pub const fn explicit_buffer_barriers_per_run(self) -> u32 {
        match self {
            Self::LegacyFourDispatch | Self::QkStagedFourDispatch => 72,
            Self::Fused128 => 18,
        }
    }

    pub const fn launched_threads_per_run(self) -> u32 {
        match self {
            Self::LegacyFourDispatch => 185_184,
            Self::QkStagedFourDispatch => 148_320,
            Self::Fused128 => 36_864,
        }
    }

    pub const fn threadgroups_per_run(self) -> u32 {
        match self {
            Self::LegacyFourDispatch | Self::QkStagedFourDispatch => 756,
            Self::Fused128 => 288,
        }
    }

    pub const fn recurrent_threads_per_threadgroup(self) -> u32 {
        match self {
            Self::LegacyFourDispatch => 256,
            Self::QkStagedFourDispatch | Self::Fused128 => 128,
        }
    }

    pub const fn source_declared_threadgroup_memory_bytes(self) -> u32 {
        match self {
            Self::LegacyFourDispatch => 0,
            Self::QkStagedFourDispatch => 1_032,
            Self::Fused128 => 2_060,
        }
    }

    pub const fn internal_threadgroup_barrier_sites_per_threadgroup(self) -> u32 {
        match self {
            Self::LegacyFourDispatch => 0,
            Self::QkStagedFourDispatch => 1,
            Self::Fused128 => 4,
        }
    }

    pub const fn is_production_profile(self) -> bool {
        matches!(self, Self::LegacyFourDispatch | Self::Fused128)
    }

    pub const fn expected_pipeline_static_threadgroup_memory_bytes(self) -> u32 {
        match self {
            Self::LegacyFourDispatch => 0,
            Self::QkStagedFourDispatch => 1_040,
            Self::Fused128 => QWEN35_GDN_CORE_FUSED_PIPELINE_STATIC_THREADGROUP_MEMORY_BYTES_V1,
        }
    }

    pub const fn gdn_core_dispatches_for_seams(self, seams: u32) -> u32 {
        seams * if matches!(self, Self::Fused128) { 1 } else { 4 }
    }

    pub const fn gdn_core_launched_threads_for_seams(self, seams: u32) -> u32 {
        let per_seam = match self {
            Self::LegacyFourDispatch => 10_288,
            Self::QkStagedFourDispatch => 8_240,
            Self::Fused128 => 2_048,
        };
        seams * per_seam
    }

    pub const fn gdn_core_threadgroups_for_seams(self, seams: u32) -> u32 {
        seams
            * match self {
                Self::LegacyFourDispatch | Self::QkStagedFourDispatch => 42,
                Self::Fused128 => 16,
            }
    }

    fn from_selector(selector: u32) -> Option<Self> {
        match selector {
            0 => Some(Self::LegacyFourDispatch),
            1 => Some(Self::QkStagedFourDispatch),
            2 => Some(Self::Fused128),
            _ => None,
        }
    }
}

/// Live production-path identity and topology for the three GDN cores inside
/// one Stack3 or MLP-to-Stack3 transaction. The dispatch/barrier/thread counts
/// intentionally exclude the surrounding projections, residuals, and MLPs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GdnCoreProductionReceiptV1 {
    pub profile: GdnCoreProfileV1,
    pub function_chain: &'static str,
    pub gdn_core_seams: u32,
    pub kernel_dispatches: u32,
    pub explicit_buffer_barriers: u32,
    /// Threads in the selected recurrent kernel for A, or fused core kernel
    /// for C. Other dispatches in the legacy four-function chain differ.
    pub recurrent_or_fused_threads_per_threadgroup: u32,
    pub threadgroups: u32,
    pub launched_threads: u32,
    pub pipeline_thread_execution_width: u32,
    pub source_declared_threadgroup_memory_bytes: u32,
    pub pipeline_static_threadgroup_memory_bytes: u32,
    pub internal_threadgroup_barrier_sites_per_threadgroup: u32,
    pub fixed_shape_validated: bool,
    pub rms_norm_eps_bits: u32,
    /// Group count retained by the following G32 output projection.
    pub persistent_output_groups_per_row: u32,
    /// Group count passed to the selected core kernel. The fused kernel's
    /// accepted shape guard is 32; it never indexes output weights or scales.
    pub core_kernel_output_groups_per_row: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct GdnCoreProductionObservedV1 {
    pub requested_profile: u32,
    pub observed_profile: u32,
    pub gdn_core_seams: u32,
    pub kernel_dispatches: u32,
    pub explicit_buffer_barriers: u32,
    pub recurrent_or_fused_threads_per_threadgroup: u32,
    pub threadgroups: u32,
    pub launched_threads: u32,
    pub pipeline_thread_execution_width: u32,
    pub source_declared_threadgroup_memory_bytes: u32,
    pub pipeline_static_threadgroup_memory_bytes: u32,
    pub internal_threadgroup_barrier_sites_per_threadgroup: u32,
    pub fixed_shape_validated: u32,
    pub rms_norm_eps_bits: u32,
    pub persistent_output_groups_per_row: u32,
    pub core_kernel_output_groups_per_row: u32,
}

pub(crate) fn validate_production_profile_v1(
    profile: GdnCoreProfileV1,
    dims: GdnDimensions,
) -> Result<(), MetalW8Error> {
    if !profile.is_production_profile() {
        return Err(MetalW8Error::new(
            "the Q/K-staged GDN core profile is diagnostic-only and cannot be selected by a production bridge",
        ));
    }
    if dims.hidden_size != QWEN35_GDN_CORE_HIDDEN_SIZE_V1
        || dims.key_heads != QWEN35_GDN_CORE_KEY_HEADS_V1
        || dims.value_heads != QWEN35_GDN_CORE_VALUE_HEADS_V1
        || dims.value_heads != dims.key_heads
        || dims.key_dim != QWEN35_GDN_CORE_KEY_DIM_V1
        || dims.value_dim != QWEN35_GDN_CORE_VALUE_DIM_V1
        || dims.conv_kernel_size != QWEN35_GDN_CORE_CONV_KERNEL_SIZE_V1
        || dims.key_width() != QWEN35_GDN_CORE_KEY_WIDTH_V1
        || dims.value_width() != QWEN35_GDN_CORE_VALUE_WIDTH_V1
        || dims.qkv_width() != QWEN35_GDN_CORE_QKV_WIDTH_V1
        || dims.input_projection_rows() != QWEN35_GDN_CORE_PROJECTED_ELEMENTS_PER_SEAM_V1
        || dims.rms_norm_eps.to_bits() != QWEN35_GDN_CORE_RMS_NORM_EPS_BITS_V1
    {
        return Err(MetalW8Error::new(
            "the fixed-shape production GDN core requires H=1024, KH=VH=16, KD=VD=128, conv=4, input_rows=8224, input_groups=16, and rms_norm_eps bits equal to 1e-6f32; production G32 output projection params retain 64 output groups, while the fused core-local shape-guard copy uses 32 and is verified by its live receipt",
        ));
    }
    Ok(())
}

pub(crate) fn validate_production_receipt_v1(
    expected_profile: GdnCoreProfileV1,
    observed: GdnCoreProductionObservedV1,
    observed_function_chain: &str,
) -> Result<GdnCoreProductionReceiptV1, MetalW8Error> {
    if !expected_profile.is_production_profile()
        || GdnCoreProfileV1::try_from(observed.requested_profile)? != expected_profile
        || GdnCoreProfileV1::try_from(observed.observed_profile)? != expected_profile
    {
        return Err(MetalW8Error::new(
            "production GDN core receipt profile identity mismatch",
        ));
    }
    if observed.gdn_core_seams != 3 {
        return Err(MetalW8Error::new(
            "production GDN core receipt must contain exactly three seams",
        ));
    }
    let expected_function_chain = expected_profile.expected_function_chain();
    let expected_dispatches =
        expected_profile.gdn_core_dispatches_for_seams(observed.gdn_core_seams);
    let expected_threadgroups =
        expected_profile.gdn_core_threadgroups_for_seams(observed.gdn_core_seams);
    let expected_launched =
        expected_profile.gdn_core_launched_threads_for_seams(observed.gdn_core_seams);
    if observed_function_chain != expected_function_chain
        || observed.kernel_dispatches != expected_dispatches
        || observed.explicit_buffer_barriers != expected_dispatches
        || observed.recurrent_or_fused_threads_per_threadgroup
            != expected_profile.recurrent_threads_per_threadgroup()
        || observed.threadgroups != expected_threadgroups
        || observed.launched_threads != expected_launched
        || observed.pipeline_thread_execution_width
            != QWEN35_GDN_CORE_PIPELINE_THREAD_EXECUTION_WIDTH_V1
        || observed.source_declared_threadgroup_memory_bytes
            != expected_profile.source_declared_threadgroup_memory_bytes()
        || observed.pipeline_static_threadgroup_memory_bytes
            != expected_profile.expected_pipeline_static_threadgroup_memory_bytes()
        || observed.internal_threadgroup_barrier_sites_per_threadgroup
            != expected_profile.internal_threadgroup_barrier_sites_per_threadgroup()
        || observed.fixed_shape_validated != 1
        || observed.rms_norm_eps_bits != QWEN35_GDN_CORE_RMS_NORM_EPS_BITS_V1
        || observed.persistent_output_groups_per_row != 64
        || observed.core_kernel_output_groups_per_row
            != if expected_profile == GdnCoreProfileV1::Fused128 {
                32
            } else {
                64
            }
    {
        return Err(MetalW8Error::new(format!(
            "invalid live production GDN core receipt for {expected_profile:?}"
        )));
    }
    Ok(GdnCoreProductionReceiptV1 {
        profile: expected_profile,
        function_chain: expected_function_chain,
        gdn_core_seams: observed.gdn_core_seams,
        kernel_dispatches: observed.kernel_dispatches,
        explicit_buffer_barriers: observed.explicit_buffer_barriers,
        recurrent_or_fused_threads_per_threadgroup: observed
            .recurrent_or_fused_threads_per_threadgroup,
        threadgroups: observed.threadgroups,
        launched_threads: observed.launched_threads,
        pipeline_thread_execution_width: observed.pipeline_thread_execution_width,
        source_declared_threadgroup_memory_bytes: observed.source_declared_threadgroup_memory_bytes,
        pipeline_static_threadgroup_memory_bytes: observed.pipeline_static_threadgroup_memory_bytes,
        internal_threadgroup_barrier_sites_per_threadgroup: observed
            .internal_threadgroup_barrier_sites_per_threadgroup,
        fixed_shape_validated: true,
        rms_norm_eps_bits: observed.rms_norm_eps_bits,
        persistent_output_groups_per_row: observed.persistent_output_groups_per_row,
        core_kernel_output_groups_per_row: observed.core_kernel_output_groups_per_row,
    })
}

impl TryFrom<u32> for GdnCoreProfileV1 {
    type Error = MetalW8Error;

    fn try_from(selector: u32) -> Result<Self, Self::Error> {
        Self::from_selector(selector).ok_or_else(|| {
            MetalW8Error::new(format!(
                "Metal GDN core profile {selector} is invalid; expected 0, 1, or 2"
            ))
        })
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct GdnCoreFusedCount18SnapshotV1 {
    pub next_query_state: Vec<f32>,
    pub next_key_state: Vec<f32>,
    pub next_value_state: Vec<f32>,
    pub next_recurrent_state: Vec<f32>,
    pub gated: Vec<f32>,
}

/// Validated live identity and topology for one arm of the count-18 screen.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GdnCoreFusedCount18RuntimeReceiptV1 {
    pub requested_profile: GdnCoreProfileV1,
    pub observed_profile: GdnCoreProfileV1,
    pub seams_per_run: u32,
    pub kernel_dispatches_per_run: u32,
    pub explicit_buffer_barriers_per_run: u32,
    pub launched_threads_per_run: u32,
    pub threadgroups_per_run: u32,
    pub recurrent_threads_per_threadgroup: u32,
    pub pipeline_thread_execution_width: u32,
    pub pipeline_static_threadgroup_memory_bytes: u32,
    pub source_declared_threadgroup_memory_bytes: u32,
    pub internal_threadgroup_barrier_sites_per_threadgroup: u32,
    pub fixed_shape_host_validated: bool,
    pub input_output_buffers_non_overlapping: bool,
    pub command_buffers_per_run: u32,
    pub compute_encoders_per_run: u32,
    pub commits_per_run: u32,
    pub waits_per_run: u32,
    pub last_observed_kernel_dispatches: u32,
    pub last_observed_explicit_buffer_barriers: u32,
    pub last_observed_launched_threads: u32,
    pub last_observed_threadgroups: u32,
    pub last_observed_command_buffers: u32,
    pub last_observed_compute_encoders: u32,
    pub last_observed_commits: u32,
    pub last_observed_waits: u32,
    pub successful_runs: u64,
    pub observed_function_chain: String,
}

pub type GdnCoreCount18SnapshotV1 = GdnCoreFusedCount18SnapshotV1;
pub type GdnCoreCount18RuntimeReceiptV1 = GdnCoreFusedCount18RuntimeReceiptV1;

/// Same-binary fixed-shape aggregate primitive. Fixture staging, immutable
/// input verification, output poisoning, and snapshots stay outside `run`.
pub struct MetalGdnCoreFusedCount18PrimitiveV1 {
    inner: platform::Handle,
}

impl MetalGdnCoreFusedCount18PrimitiveV1 {
    pub fn new() -> Result<Self, MetalW8Error> {
        Ok(Self {
            inner: platform::Handle::new()?,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn stage_fixture(
        &mut self,
        projected: &[f32],
        conv_weight: &[f32],
        query_state: &[f32],
        key_state: &[f32],
        value_state: &[f32],
        a_log: &[f32],
        dt_bias: &[f32],
        recurrent_state: &[f32],
        norm_weight: &[f32],
    ) -> Result<(), MetalW8Error> {
        validate_fixture(
            projected,
            conv_weight,
            query_state,
            key_state,
            value_state,
            a_log,
            dt_bias,
            recurrent_state,
            norm_weight,
        )?;
        self.inner.stage_fixture(
            projected,
            conv_weight,
            query_state,
            key_state,
            value_state,
            a_log,
            dt_bias,
            recurrent_state,
            norm_weight,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn verify_fixture_unchanged(
        &self,
        projected: &[f32],
        conv_weight: &[f32],
        query_state: &[f32],
        key_state: &[f32],
        value_state: &[f32],
        a_log: &[f32],
        dt_bias: &[f32],
        recurrent_state: &[f32],
        norm_weight: &[f32],
    ) -> Result<(), MetalW8Error> {
        validate_fixture(
            projected,
            conv_weight,
            query_state,
            key_state,
            value_state,
            a_log,
            dt_bias,
            recurrent_state,
            norm_weight,
        )?;
        self.inner.verify_fixture_unchanged(
            projected,
            conv_weight,
            query_state,
            key_state,
            value_state,
            a_log,
            dt_bias,
            recurrent_state,
            norm_weight,
        )
    }

    pub fn poison_outputs(&mut self) -> Result<(), MetalW8Error> {
        self.inner.poison_outputs()
    }

    pub fn run(&mut self, profile: GdnCoreProfileV1) -> Result<(), MetalW8Error> {
        self.inner.run(profile)
    }

    pub fn snapshot(&self) -> Result<GdnCoreFusedCount18SnapshotV1, MetalW8Error> {
        self.inner.snapshot()
    }

    pub fn runtime_receipt(
        &self,
        profile: GdnCoreProfileV1,
    ) -> Result<GdnCoreFusedCount18RuntimeReceiptV1, MetalW8Error> {
        self.inner.runtime_receipt(profile)
    }

    pub fn verify_invalid_raw_selectors_fail_closed(&self) -> Result<(), MetalW8Error> {
        if self
            .inner
            .invalid_raw_selectors_are_rejected_without_mutation()
        {
            Ok(())
        } else {
            Err(MetalW8Error::new(
                "invalid raw Metal GDN core selector mutated a receipt or snapshot",
            ))
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn validate_fixture(
    projected: &[f32],
    conv_weight: &[f32],
    query_state: &[f32],
    key_state: &[f32],
    value_state: &[f32],
    a_log: &[f32],
    dt_bias: &[f32],
    recurrent_state: &[f32],
    norm_weight: &[f32],
) -> Result<(), MetalW8Error> {
    validate_finite(
        "projected",
        projected,
        QWEN35_GDN_CORE_PROJECTED_TRACE_ELEMENTS_V1,
    )?;
    validate_finite(
        "conv_weight",
        conv_weight,
        QWEN35_GDN_CORE_CONV_WEIGHT_TRACE_ELEMENTS_V1,
    )?;
    validate_finite(
        "query_state",
        query_state,
        QWEN35_GDN_CORE_QUERY_STATE_TRACE_ELEMENTS_V1,
    )?;
    validate_finite(
        "key_state",
        key_state,
        QWEN35_GDN_CORE_KEY_STATE_TRACE_ELEMENTS_V1,
    )?;
    validate_finite(
        "value_state",
        value_state,
        QWEN35_GDN_CORE_VALUE_STATE_TRACE_ELEMENTS_V1,
    )?;
    validate_finite(
        "A_log",
        a_log,
        QWEN35_GDN_CORE_HEAD_SCALAR_TRACE_ELEMENTS_V1,
    )?;
    validate_finite(
        "dt_bias",
        dt_bias,
        QWEN35_GDN_CORE_HEAD_SCALAR_TRACE_ELEMENTS_V1,
    )?;
    validate_finite(
        "recurrent_state",
        recurrent_state,
        QWEN35_GDN_CORE_RECURRENT_TRACE_ELEMENTS_V1,
    )?;
    validate_finite(
        "norm_weight",
        norm_weight,
        QWEN35_GDN_CORE_NORM_WEIGHT_TRACE_ELEMENTS_V1,
    )
}

fn validate_finite(label: &str, values: &[f32], expected: usize) -> Result<(), MetalW8Error> {
    if values.len() != expected {
        return Err(MetalW8Error::new(format!(
            "Metal GDN core {label} has {} elements, expected {expected}",
            values.len()
        )));
    }
    if let Some(index) = values.iter().position(|value| !value.is_finite()) {
        return Err(MetalW8Error::new(format!(
            "Metal GDN core {label} contains a non-finite value at element {index}"
        )));
    }
    Ok(())
}

fn slices_match_to_bits(left: &[f32], right: &[f32]) -> bool {
    left.len() == right.len()
        && left
            .iter()
            .zip(right)
            .all(|(left, right)| left.to_bits() == right.to_bits())
}

fn snapshots_match_to_bits(
    left: &GdnCoreFusedCount18SnapshotV1,
    right: &GdnCoreFusedCount18SnapshotV1,
) -> bool {
    slices_match_to_bits(&left.next_query_state, &right.next_query_state)
        && slices_match_to_bits(&left.next_key_state, &right.next_key_state)
        && slices_match_to_bits(&left.next_value_state, &right.next_value_state)
        && slices_match_to_bits(&left.next_recurrent_state, &right.next_recurrent_state)
        && slices_match_to_bits(&left.gated, &right.gated)
}

fn optional_snapshots_match_to_bits(
    left: &Option<GdnCoreFusedCount18SnapshotV1>,
    right: &Option<GdnCoreFusedCount18SnapshotV1>,
) -> bool {
    match (left, right) {
        (None, None) => true,
        (Some(left), Some(right)) => snapshots_match_to_bits(left, right),
        _ => false,
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct RawRuntimeReceiptV1 {
    requested_profile: u32,
    observed_profile: u32,
    seams_per_run: u32,
    kernel_dispatches_per_run: u32,
    explicit_buffer_barriers_per_run: u32,
    launched_threads_per_run: u32,
    threadgroups_per_run: u32,
    recurrent_threads_per_threadgroup: u32,
    pipeline_thread_execution_width: u32,
    pipeline_static_threadgroup_memory_bytes: u32,
    source_declared_threadgroup_memory_bytes: u32,
    internal_threadgroup_barrier_sites_per_threadgroup: u32,
    fixed_shape_host_validated: u32,
    input_output_buffers_non_overlapping: u32,
    command_buffers_per_run: u32,
    compute_encoders_per_run: u32,
    commits_per_run: u32,
    waits_per_run: u32,
    last_observed_kernel_dispatches: u32,
    last_observed_explicit_buffer_barriers: u32,
    last_observed_launched_threads: u32,
    last_observed_threadgroups: u32,
    last_observed_command_buffers: u32,
    last_observed_compute_encoders: u32,
    last_observed_commits: u32,
    last_observed_waits: u32,
    successful_runs: u64,
    observed_function_chain: [c_char; FUNCTION_CHAIN_CAPACITY],
}

impl Default for RawRuntimeReceiptV1 {
    fn default() -> Self {
        Self {
            requested_profile: u32::MAX,
            observed_profile: u32::MAX,
            seams_per_run: 0,
            kernel_dispatches_per_run: 0,
            explicit_buffer_barriers_per_run: 0,
            launched_threads_per_run: 0,
            threadgroups_per_run: 0,
            recurrent_threads_per_threadgroup: 0,
            pipeline_thread_execution_width: 0,
            pipeline_static_threadgroup_memory_bytes: 0,
            source_declared_threadgroup_memory_bytes: 0,
            internal_threadgroup_barrier_sites_per_threadgroup: 0,
            fixed_shape_host_validated: 0,
            input_output_buffers_non_overlapping: 0,
            command_buffers_per_run: 0,
            compute_encoders_per_run: 0,
            commits_per_run: 0,
            waits_per_run: 0,
            last_observed_kernel_dispatches: 0,
            last_observed_explicit_buffer_barriers: 0,
            last_observed_launched_threads: 0,
            last_observed_threadgroups: 0,
            last_observed_command_buffers: 0,
            last_observed_compute_encoders: 0,
            last_observed_commits: 0,
            last_observed_waits: 0,
            successful_runs: 0,
            observed_function_chain: [0; FUNCTION_CHAIN_CAPACITY],
        }
    }
}

const _: [(); RAW_RUNTIME_RECEIPT_SIZE] = [(); std::mem::size_of::<RawRuntimeReceiptV1>()];

#[cfg(target_os = "macos")]
mod platform {
    use super::*;
    use std::ffi::{c_int, c_void};
    use std::ptr::NonNull;

    const ERROR_CAPACITY: usize = 1024;

    extern "C" {
        fn apxinf_metal_gdn_core_fused_count18_profile_v1_create(
            output: *mut *mut c_void,
            error: *mut c_char,
            error_capacity: usize,
        ) -> c_int;
        fn apxinf_metal_gdn_core_fused_count18_profile_v1_stage_fixture(
            handle: *mut c_void,
            projected: *const f32,
            projected_count: u32,
            conv_weight: *const f32,
            conv_weight_count: u32,
            query_state: *const f32,
            query_state_count: u32,
            key_state: *const f32,
            key_state_count: u32,
            value_state: *const f32,
            value_state_count: u32,
            a_log: *const f32,
            a_log_count: u32,
            dt_bias: *const f32,
            dt_bias_count: u32,
            recurrent_state: *const f32,
            recurrent_state_count: u32,
            norm_weight: *const f32,
            norm_weight_count: u32,
            error: *mut c_char,
            error_capacity: usize,
        ) -> c_int;
        fn apxinf_metal_gdn_core_fused_count18_profile_v1_verify_fixture_unchanged(
            handle: *mut c_void,
            projected: *const f32,
            projected_count: u32,
            conv_weight: *const f32,
            conv_weight_count: u32,
            query_state: *const f32,
            query_state_count: u32,
            key_state: *const f32,
            key_state_count: u32,
            value_state: *const f32,
            value_state_count: u32,
            a_log: *const f32,
            a_log_count: u32,
            dt_bias: *const f32,
            dt_bias_count: u32,
            recurrent_state: *const f32,
            recurrent_state_count: u32,
            norm_weight: *const f32,
            norm_weight_count: u32,
            error: *mut c_char,
            error_capacity: usize,
        ) -> c_int;
        fn apxinf_metal_gdn_core_fused_count18_profile_v1_poison_outputs(
            handle: *mut c_void,
            error: *mut c_char,
            error_capacity: usize,
        ) -> c_int;
        fn apxinf_metal_gdn_core_fused_count18_profile_v1_run(
            handle: *mut c_void,
            profile: u32,
            error: *mut c_char,
            error_capacity: usize,
        ) -> c_int;
        fn apxinf_metal_gdn_core_fused_count18_profile_v1_snapshot(
            handle: *mut c_void,
            next_query_state: *mut f32,
            next_query_state_count: u32,
            next_key_state: *mut f32,
            next_key_state_count: u32,
            next_value_state: *mut f32,
            next_value_state_count: u32,
            next_recurrent_state: *mut f32,
            next_recurrent_state_count: u32,
            gated: *mut f32,
            gated_count: u32,
            error: *mut c_char,
            error_capacity: usize,
        ) -> c_int;
        fn apxinf_metal_gdn_core_fused_count18_profile_v1_receipt(
            handle: *mut c_void,
            profile: u32,
            receipt: *mut RawRuntimeReceiptV1,
            error: *mut c_char,
            error_capacity: usize,
        ) -> c_int;
        fn apxinf_metal_gdn_core_fused_count18_profile_v1_destroy(handle: *mut c_void);
    }

    pub(super) struct Handle(NonNull<c_void>);

    impl Handle {
        pub(super) fn new() -> Result<Self, MetalW8Error> {
            let mut output = std::ptr::null_mut();
            let mut error = [0 as c_char; ERROR_CAPACITY];
            let status = unsafe {
                apxinf_metal_gdn_core_fused_count18_profile_v1_create(
                    &mut output,
                    error.as_mut_ptr(),
                    error.len(),
                )
            };
            if status != 0 {
                return Err(bridge_error(
                    "create Metal GDN core fused primitive",
                    &error,
                ));
            }
            let handle = Self(NonNull::new(output).ok_or_else(|| {
                MetalW8Error::new("create Metal GDN core fused primitive returned a null handle")
            })?);
            for profile in GdnCoreProfileV1::ALL {
                let receipt = handle.runtime_receipt(profile)?;
                if receipt.successful_runs != 0 {
                    return Err(MetalW8Error::new(
                        "new Metal GDN core fused primitive reported successful runs",
                    ));
                }
            }
            Ok(handle)
        }

        #[allow(clippy::too_many_arguments)]
        pub(super) fn stage_fixture(
            &mut self,
            projected: &[f32],
            conv_weight: &[f32],
            query_state: &[f32],
            key_state: &[f32],
            value_state: &[f32],
            a_log: &[f32],
            dt_bias: &[f32],
            recurrent_state: &[f32],
            norm_weight: &[f32],
        ) -> Result<(), MetalW8Error> {
            let mut error = [0 as c_char; ERROR_CAPACITY];
            let status = unsafe {
                apxinf_metal_gdn_core_fused_count18_profile_v1_stage_fixture(
                    self.0.as_ptr(),
                    projected.as_ptr(),
                    projected.len() as u32,
                    conv_weight.as_ptr(),
                    conv_weight.len() as u32,
                    query_state.as_ptr(),
                    query_state.len() as u32,
                    key_state.as_ptr(),
                    key_state.len() as u32,
                    value_state.as_ptr(),
                    value_state.len() as u32,
                    a_log.as_ptr(),
                    a_log.len() as u32,
                    dt_bias.as_ptr(),
                    dt_bias.len() as u32,
                    recurrent_state.as_ptr(),
                    recurrent_state.len() as u32,
                    norm_weight.as_ptr(),
                    norm_weight.len() as u32,
                    error.as_mut_ptr(),
                    error.len(),
                )
            };
            if status != 0 {
                return Err(bridge_error("stage Metal GDN core fixture", &error));
            }
            Ok(())
        }

        #[allow(clippy::too_many_arguments)]
        pub(super) fn verify_fixture_unchanged(
            &self,
            projected: &[f32],
            conv_weight: &[f32],
            query_state: &[f32],
            key_state: &[f32],
            value_state: &[f32],
            a_log: &[f32],
            dt_bias: &[f32],
            recurrent_state: &[f32],
            norm_weight: &[f32],
        ) -> Result<(), MetalW8Error> {
            let mut error = [0 as c_char; ERROR_CAPACITY];
            let status = unsafe {
                apxinf_metal_gdn_core_fused_count18_profile_v1_verify_fixture_unchanged(
                    self.0.as_ptr(),
                    projected.as_ptr(),
                    projected.len() as u32,
                    conv_weight.as_ptr(),
                    conv_weight.len() as u32,
                    query_state.as_ptr(),
                    query_state.len() as u32,
                    key_state.as_ptr(),
                    key_state.len() as u32,
                    value_state.as_ptr(),
                    value_state.len() as u32,
                    a_log.as_ptr(),
                    a_log.len() as u32,
                    dt_bias.as_ptr(),
                    dt_bias.len() as u32,
                    recurrent_state.as_ptr(),
                    recurrent_state.len() as u32,
                    norm_weight.as_ptr(),
                    norm_weight.len() as u32,
                    error.as_mut_ptr(),
                    error.len(),
                )
            };
            if status != 0 {
                return Err(bridge_error("verify staged Metal GDN core fixture", &error));
            }
            Ok(())
        }

        pub(super) fn poison_outputs(&mut self) -> Result<(), MetalW8Error> {
            let mut error = [0 as c_char; ERROR_CAPACITY];
            let status = unsafe {
                apxinf_metal_gdn_core_fused_count18_profile_v1_poison_outputs(
                    self.0.as_ptr(),
                    error.as_mut_ptr(),
                    error.len(),
                )
            };
            if status != 0 {
                return Err(bridge_error("poison Metal GDN core outputs", &error));
            }
            Ok(())
        }

        pub(super) fn run(&mut self, profile: GdnCoreProfileV1) -> Result<(), MetalW8Error> {
            let mut error = [0 as c_char; ERROR_CAPACITY];
            let status = unsafe {
                apxinf_metal_gdn_core_fused_count18_profile_v1_run(
                    self.0.as_ptr(),
                    profile.selector(),
                    error.as_mut_ptr(),
                    error.len(),
                )
            };
            if status != 0 {
                return Err(bridge_error("run Metal GDN core fused primitive", &error));
            }
            Ok(())
        }

        pub(super) fn snapshot(&self) -> Result<GdnCoreFusedCount18SnapshotV1, MetalW8Error> {
            let mut next_query_state = vec![0.0f32; QWEN35_GDN_CORE_QUERY_STATE_TRACE_ELEMENTS_V1];
            let mut next_key_state = vec![0.0f32; QWEN35_GDN_CORE_KEY_STATE_TRACE_ELEMENTS_V1];
            let mut next_value_state = vec![0.0f32; QWEN35_GDN_CORE_VALUE_STATE_TRACE_ELEMENTS_V1];
            let mut next_recurrent_state =
                vec![0.0f32; QWEN35_GDN_CORE_RECURRENT_TRACE_ELEMENTS_V1];
            let mut gated = vec![0.0f32; QWEN35_GDN_CORE_GATED_TRACE_ELEMENTS_V1];
            let mut error = [0 as c_char; ERROR_CAPACITY];
            let status = unsafe {
                apxinf_metal_gdn_core_fused_count18_profile_v1_snapshot(
                    self.0.as_ptr(),
                    next_query_state.as_mut_ptr(),
                    next_query_state.len() as u32,
                    next_key_state.as_mut_ptr(),
                    next_key_state.len() as u32,
                    next_value_state.as_mut_ptr(),
                    next_value_state.len() as u32,
                    next_recurrent_state.as_mut_ptr(),
                    next_recurrent_state.len() as u32,
                    gated.as_mut_ptr(),
                    gated.len() as u32,
                    error.as_mut_ptr(),
                    error.len(),
                )
            };
            if status != 0 {
                return Err(bridge_error("snapshot Metal GDN core outputs", &error));
            }
            Ok(GdnCoreFusedCount18SnapshotV1 {
                next_query_state,
                next_key_state,
                next_value_state,
                next_recurrent_state,
                gated,
            })
        }

        pub(super) fn runtime_receipt(
            &self,
            expected: GdnCoreProfileV1,
        ) -> Result<GdnCoreFusedCount18RuntimeReceiptV1, MetalW8Error> {
            let mut raw = RawRuntimeReceiptV1::default();
            let mut error = [0 as c_char; ERROR_CAPACITY];
            let status = unsafe {
                apxinf_metal_gdn_core_fused_count18_profile_v1_receipt(
                    self.0.as_ptr(),
                    expected.selector(),
                    &mut raw,
                    error.as_mut_ptr(),
                    error.len(),
                )
            };
            if status != 0 {
                return Err(bridge_error("read Metal GDN core receipt", &error));
            }
            convert_and_validate_receipt(raw, expected)
        }

        pub(super) fn invalid_raw_selectors_are_rejected_without_mutation(&self) -> bool {
            let before = GdnCoreProfileV1::ALL.map(|profile| self.runtime_receipt(profile).ok());
            if !before.iter().all(Option::is_some) {
                return false;
            }
            let before_snapshot = self.snapshot().ok();
            for selector in [3, u32::MAX] {
                let mut raw = RawRuntimeReceiptV1::default();
                let raw_before = raw;
                let mut error = [0 as c_char; ERROR_CAPACITY];
                let receipt_rejected = unsafe {
                    apxinf_metal_gdn_core_fused_count18_profile_v1_receipt(
                        self.0.as_ptr(),
                        selector,
                        &mut raw,
                        error.as_mut_ptr(),
                        error.len(),
                    ) != 0
                };
                error.fill(0);
                let run_rejected = unsafe {
                    apxinf_metal_gdn_core_fused_count18_profile_v1_run(
                        self.0.as_ptr(),
                        selector,
                        error.as_mut_ptr(),
                        error.len(),
                    ) != 0
                };
                let after = GdnCoreProfileV1::ALL.map(|profile| self.runtime_receipt(profile).ok());
                let after_snapshot = self.snapshot().ok();
                if !receipt_rejected
                    || !run_rejected
                    || raw != raw_before
                    || before != after
                    || !optional_snapshots_match_to_bits(&before_snapshot, &after_snapshot)
                {
                    return false;
                }
            }
            true
        }
    }

    impl Drop for Handle {
        fn drop(&mut self) {
            unsafe { apxinf_metal_gdn_core_fused_count18_profile_v1_destroy(self.0.as_ptr()) };
        }
    }

    fn convert_and_validate_receipt(
        raw: RawRuntimeReceiptV1,
        expected: GdnCoreProfileV1,
    ) -> Result<GdnCoreFusedCount18RuntimeReceiptV1, MetalW8Error> {
        let requested_profile = GdnCoreProfileV1::try_from(raw.requested_profile)?;
        let observed_profile = GdnCoreProfileV1::try_from(raw.observed_profile)?;
        let observed_function_chain = function_chain(&raw.observed_function_chain)?;
        let expected_last = u32::from(raw.successful_runs != 0);
        let source_memory = expected.source_declared_threadgroup_memory_bytes();
        let pipeline_static_memory_valid = if source_memory == 0 {
            raw.pipeline_static_threadgroup_memory_bytes == 0
        } else {
            raw.pipeline_static_threadgroup_memory_bytes >= source_memory
                && raw.pipeline_static_threadgroup_memory_bytes <= 32_768
        };
        if requested_profile != expected
            || observed_profile != expected
            || observed_function_chain != expected.expected_function_chain()
            || raw.seams_per_run != QWEN35_GDN_CORE_SEAMS_PER_DECODE_V1 as u32
            || raw.kernel_dispatches_per_run != expected.kernel_dispatches_per_run()
            || raw.explicit_buffer_barriers_per_run != expected.explicit_buffer_barriers_per_run()
            || raw.launched_threads_per_run != expected.launched_threads_per_run()
            || raw.threadgroups_per_run != expected.threadgroups_per_run()
            || raw.recurrent_threads_per_threadgroup != expected.recurrent_threads_per_threadgroup()
            || raw.pipeline_thread_execution_width != 32
            || !pipeline_static_memory_valid
            || raw.source_declared_threadgroup_memory_bytes != source_memory
            || raw.internal_threadgroup_barrier_sites_per_threadgroup
                != expected.internal_threadgroup_barrier_sites_per_threadgroup()
            || raw.fixed_shape_host_validated != 1
            || raw.input_output_buffers_non_overlapping != 1
            || raw.command_buffers_per_run != 1
            || raw.compute_encoders_per_run != 1
            || raw.commits_per_run != 1
            || raw.waits_per_run != 1
            || raw.last_observed_kernel_dispatches
                != expected_last * expected.kernel_dispatches_per_run()
            || raw.last_observed_explicit_buffer_barriers
                != expected_last * expected.explicit_buffer_barriers_per_run()
            || raw.last_observed_launched_threads
                != expected_last * expected.launched_threads_per_run()
            || raw.last_observed_threadgroups != expected_last * expected.threadgroups_per_run()
            || raw.last_observed_command_buffers != expected_last
            || raw.last_observed_compute_encoders != expected_last
            || raw.last_observed_commits != expected_last
            || raw.last_observed_waits != expected_last
        {
            return Err(MetalW8Error::new(format!(
                "invalid live Metal GDN core count18 receipt for {expected:?}"
            )));
        }
        Ok(GdnCoreFusedCount18RuntimeReceiptV1 {
            requested_profile,
            observed_profile,
            seams_per_run: raw.seams_per_run,
            kernel_dispatches_per_run: raw.kernel_dispatches_per_run,
            explicit_buffer_barriers_per_run: raw.explicit_buffer_barriers_per_run,
            launched_threads_per_run: raw.launched_threads_per_run,
            threadgroups_per_run: raw.threadgroups_per_run,
            recurrent_threads_per_threadgroup: raw.recurrent_threads_per_threadgroup,
            pipeline_thread_execution_width: raw.pipeline_thread_execution_width,
            pipeline_static_threadgroup_memory_bytes: raw.pipeline_static_threadgroup_memory_bytes,
            source_declared_threadgroup_memory_bytes: raw.source_declared_threadgroup_memory_bytes,
            internal_threadgroup_barrier_sites_per_threadgroup: raw
                .internal_threadgroup_barrier_sites_per_threadgroup,
            fixed_shape_host_validated: true,
            input_output_buffers_non_overlapping: true,
            command_buffers_per_run: raw.command_buffers_per_run,
            compute_encoders_per_run: raw.compute_encoders_per_run,
            commits_per_run: raw.commits_per_run,
            waits_per_run: raw.waits_per_run,
            last_observed_kernel_dispatches: raw.last_observed_kernel_dispatches,
            last_observed_explicit_buffer_barriers: raw.last_observed_explicit_buffer_barriers,
            last_observed_launched_threads: raw.last_observed_launched_threads,
            last_observed_threadgroups: raw.last_observed_threadgroups,
            last_observed_command_buffers: raw.last_observed_command_buffers,
            last_observed_compute_encoders: raw.last_observed_compute_encoders,
            last_observed_commits: raw.last_observed_commits,
            last_observed_waits: raw.last_observed_waits,
            successful_runs: raw.successful_runs,
            observed_function_chain,
        })
    }

    fn function_chain(raw: &[c_char; FUNCTION_CHAIN_CAPACITY]) -> Result<String, MetalW8Error> {
        let end = raw.iter().position(|byte| *byte == 0).ok_or_else(|| {
            MetalW8Error::new("Metal GDN core receipt function chain is not NUL-terminated")
        })?;
        let bytes = raw[..end]
            .iter()
            .map(|byte| *byte as u8)
            .collect::<Vec<_>>();
        String::from_utf8(bytes).map_err(|_| {
            MetalW8Error::new("Metal GDN core receipt function chain is not valid UTF-8")
        })
    }

    fn bridge_error(context: &str, buffer: &[c_char]) -> MetalW8Error {
        let end = buffer
            .iter()
            .position(|byte| *byte == 0)
            .unwrap_or(buffer.len());
        let bytes = buffer[..end]
            .iter()
            .map(|byte| *byte as u8)
            .collect::<Vec<_>>();
        let detail = String::from_utf8_lossy(&bytes);
        if detail.is_empty() {
            MetalW8Error::new(context)
        } else {
            MetalW8Error::new(format!("{context}: {detail}"))
        }
    }
}

#[cfg(not(target_os = "macos"))]
mod platform {
    use super::*;

    pub(super) struct Handle;

    impl Handle {
        pub(super) fn new() -> Result<Self, MetalW8Error> {
            Err(unsupported())
        }

        #[allow(clippy::too_many_arguments)]
        pub(super) fn stage_fixture(
            &mut self,
            _projected: &[f32],
            _conv_weight: &[f32],
            _query_state: &[f32],
            _key_state: &[f32],
            _value_state: &[f32],
            _a_log: &[f32],
            _dt_bias: &[f32],
            _recurrent_state: &[f32],
            _norm_weight: &[f32],
        ) -> Result<(), MetalW8Error> {
            Err(unsupported())
        }

        #[allow(clippy::too_many_arguments)]
        pub(super) fn verify_fixture_unchanged(
            &self,
            _projected: &[f32],
            _conv_weight: &[f32],
            _query_state: &[f32],
            _key_state: &[f32],
            _value_state: &[f32],
            _a_log: &[f32],
            _dt_bias: &[f32],
            _recurrent_state: &[f32],
            _norm_weight: &[f32],
        ) -> Result<(), MetalW8Error> {
            Err(unsupported())
        }

        pub(super) fn poison_outputs(&mut self) -> Result<(), MetalW8Error> {
            Err(unsupported())
        }

        pub(super) fn run(&mut self, _profile: GdnCoreProfileV1) -> Result<(), MetalW8Error> {
            Err(unsupported())
        }

        pub(super) fn snapshot(&self) -> Result<GdnCoreFusedCount18SnapshotV1, MetalW8Error> {
            Err(unsupported())
        }

        pub(super) fn runtime_receipt(
            &self,
            _profile: GdnCoreProfileV1,
        ) -> Result<GdnCoreFusedCount18RuntimeReceiptV1, MetalW8Error> {
            Err(unsupported())
        }

        pub(super) fn invalid_raw_selectors_are_rejected_without_mutation(&self) -> bool {
            false
        }
    }

    fn unsupported() -> MetalW8Error {
        MetalW8Error::new("Metal GDN core fused count18 primitive requires macOS")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selectors_and_fixed_topology_are_exact() {
        assert_eq!(GdnCoreProfileV1::LegacyFourDispatch.selector(), 0);
        assert_eq!(GdnCoreProfileV1::QkStagedFourDispatch.selector(), 1);
        assert_eq!(GdnCoreProfileV1::Fused128.selector(), 2);
        assert!(GdnCoreProfileV1::try_from(3).is_err());
        assert!(GdnCoreProfileV1::try_from(u32::MAX).is_err());
        assert_eq!(QWEN35_GDN_CORE_HIDDEN_SIZE_V1, 1024);
        assert_eq!(QWEN35_GDN_CORE_KEY_HEADS_V1, 16);
        assert_eq!(QWEN35_GDN_CORE_VALUE_HEADS_V1, 16);
        assert_eq!(QWEN35_GDN_CORE_KEY_DIM_V1, 128);
        assert_eq!(QWEN35_GDN_CORE_VALUE_DIM_V1, 128);
        assert_eq!(QWEN35_GDN_CORE_CONV_KERNEL_SIZE_V1, 4);
        assert_eq!(QWEN35_GDN_CORE_SEAMS_PER_DECODE_V1, 18);
        let observed = GdnCoreProfileV1::ALL.map(|profile| {
            (
                profile.kernel_dispatches_per_run(),
                profile.explicit_buffer_barriers_per_run(),
                profile.launched_threads_per_run(),
                profile.threadgroups_per_run(),
                profile.recurrent_threads_per_threadgroup(),
                profile.source_declared_threadgroup_memory_bytes(),
                profile.internal_threadgroup_barrier_sites_per_threadgroup(),
            )
        });
        assert_eq!(
            observed,
            [
                (72, 72, 185_184, 756, 256, 0, 0),
                (72, 72, 148_320, 756, 128, 1_032, 1),
                (18, 18, 36_864, 288, 128, 2_060, 4),
            ]
        );
        assert_eq!(
            GdnCoreProfileV1::LegacyFourDispatch.expected_function_chain(),
            "gdn_depthwise_preprocess|gdn_normalize_qk|gdn_recurrent_update|gdn_norm_gate"
        );
        assert_eq!(
            GdnCoreProfileV1::QkStagedFourDispatch.expected_function_chain(),
            "gdn_depthwise_preprocess|gdn_normalize_qk|gdn_recurrent_update_qk_staged_v1|gdn_norm_gate"
        );
        assert_eq!(
            GdnCoreProfileV1::Fused128.expected_function_chain(),
            "gdn_core_fused_v1"
        );
    }

    #[test]
    fn production_profiles_lock_shape_epsilon_and_live_topology() {
        let dims = GdnDimensions {
            hidden_size: 1024,
            key_heads: 16,
            value_heads: 16,
            key_dim: 128,
            value_dim: 128,
            conv_kernel_size: 4,
            rms_norm_eps: 1.0e-6,
        };
        for profile in [
            GdnCoreProfileV1::LegacyFourDispatch,
            GdnCoreProfileV1::Fused128,
        ] {
            validate_production_profile_v1(profile, dims).unwrap();
            let seams = 3;
            let receipt = validate_production_receipt_v1(
                profile,
                GdnCoreProductionObservedV1 {
                    requested_profile: profile.selector(),
                    observed_profile: profile.selector(),
                    gdn_core_seams: seams,
                    kernel_dispatches: profile.gdn_core_dispatches_for_seams(seams),
                    explicit_buffer_barriers: profile.gdn_core_dispatches_for_seams(seams),
                    recurrent_or_fused_threads_per_threadgroup: profile
                        .recurrent_threads_per_threadgroup(),
                    threadgroups: profile.gdn_core_threadgroups_for_seams(seams),
                    launched_threads: profile.gdn_core_launched_threads_for_seams(seams),
                    pipeline_thread_execution_width:
                        QWEN35_GDN_CORE_PIPELINE_THREAD_EXECUTION_WIDTH_V1,
                    source_declared_threadgroup_memory_bytes: profile
                        .source_declared_threadgroup_memory_bytes(),
                    pipeline_static_threadgroup_memory_bytes: profile
                        .expected_pipeline_static_threadgroup_memory_bytes(),
                    internal_threadgroup_barrier_sites_per_threadgroup: profile
                        .internal_threadgroup_barrier_sites_per_threadgroup(),
                    fixed_shape_validated: 1,
                    rms_norm_eps_bits: QWEN35_GDN_CORE_RMS_NORM_EPS_BITS_V1,
                    persistent_output_groups_per_row: 64,
                    core_kernel_output_groups_per_row: if profile == GdnCoreProfileV1::Fused128 {
                        32
                    } else {
                        64
                    },
                },
                profile.expected_function_chain(),
            )
            .unwrap();
            assert_eq!(receipt.profile, profile);
            assert_eq!(receipt.gdn_core_seams, 3);
        }

        assert!(
            validate_production_profile_v1(GdnCoreProfileV1::QkStagedFourDispatch, dims).is_err()
        );
        let wrong_eps = GdnDimensions {
            rms_norm_eps: f32::from_bits(QWEN35_GDN_CORE_RMS_NORM_EPS_BITS_V1 + 1),
            ..dims
        };
        assert!(validate_production_profile_v1(GdnCoreProfileV1::Fused128, wrong_eps).is_err());
    }

    #[test]
    fn raw_runtime_receipt_abi_is_exactly_368_bytes() {
        assert_eq!(std::mem::size_of::<RawRuntimeReceiptV1>(), 368);
        assert_eq!(std::mem::align_of::<RawRuntimeReceiptV1>(), 8);
    }

    #[test]
    fn invalid_selector_snapshot_custody_distinguishes_signed_zero_bits() {
        let positive = Some(GdnCoreFusedCount18SnapshotV1 {
            next_query_state: vec![0.0],
            next_key_state: vec![-0.0],
            next_value_state: vec![0.0],
            next_recurrent_state: vec![-0.0],
            gated: vec![0.0],
        });
        let negative = Some(GdnCoreFusedCount18SnapshotV1 {
            next_query_state: vec![-0.0],
            next_key_state: vec![0.0],
            next_value_state: vec![-0.0],
            next_recurrent_state: vec![0.0],
            gated: vec![-0.0],
        });
        assert!(!optional_snapshots_match_to_bits(&positive, &negative));
        assert!(optional_snapshots_match_to_bits(&positive, &positive));
        assert!(optional_snapshots_match_to_bits(&None, &None));
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn count18_qk_control_and_fused_outputs_match_legacy_to_bits() {
        fn values(count: usize, seed: u64, scale: f32) -> Vec<f32> {
            let mut state = seed;
            let mut values = (0..count)
                .map(|_| {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    let signed = ((state >> 32) % 2001) as i32 - 1000;
                    signed as f32 * scale
                })
                .collect::<Vec<_>>();
            if values.len() >= 2 {
                values[0] = 0.0;
                values[1] = -0.0;
            }
            values
        }

        fn assert_snapshot_bits(
            profile: GdnCoreProfileV1,
            expected: &GdnCoreFusedCount18SnapshotV1,
            actual: &GdnCoreFusedCount18SnapshotV1,
        ) {
            for (label, expected, actual) in [
                (
                    "next_query_state",
                    &expected.next_query_state,
                    &actual.next_query_state,
                ),
                (
                    "next_key_state",
                    &expected.next_key_state,
                    &actual.next_key_state,
                ),
                (
                    "next_value_state",
                    &expected.next_value_state,
                    &actual.next_value_state,
                ),
                (
                    "next_recurrent_state",
                    &expected.next_recurrent_state,
                    &actual.next_recurrent_state,
                ),
                ("gated", &expected.gated, &actual.gated),
            ] {
                assert_eq!(
                    expected.len(),
                    actual.len(),
                    "{label} length for {profile:?}"
                );
                for (index, (&left, &right)) in expected.iter().zip(actual).enumerate() {
                    assert!(
                        left.is_finite() && right.is_finite(),
                        "{label} non-finite for {profile:?} at {index}"
                    );
                    assert_eq!(
                        left.to_bits(),
                        right.to_bits(),
                        "{label} mismatch for {profile:?} at {index}"
                    );
                }
            }
        }

        let projected = values(
            QWEN35_GDN_CORE_PROJECTED_TRACE_ELEMENTS_V1,
            0x243f_6a88_85a3_08d3,
            1.0 / 4096.0,
        );
        let conv_weight = values(
            QWEN35_GDN_CORE_CONV_WEIGHT_TRACE_ELEMENTS_V1,
            0x1319_8a2e_0370_7344,
            1.0 / 4096.0,
        );
        let query_state = values(
            QWEN35_GDN_CORE_QUERY_STATE_TRACE_ELEMENTS_V1,
            0xa409_3822_299f_31d0,
            1.0 / 4096.0,
        );
        let key_state = values(
            QWEN35_GDN_CORE_KEY_STATE_TRACE_ELEMENTS_V1,
            0x082e_fa98_ec4e_6c89,
            1.0 / 4096.0,
        );
        let value_state = values(
            QWEN35_GDN_CORE_VALUE_STATE_TRACE_ELEMENTS_V1,
            0x4528_21e6_38d0_1377,
            1.0 / 4096.0,
        );
        let a_log = values(
            QWEN35_GDN_CORE_HEAD_SCALAR_TRACE_ELEMENTS_V1,
            0xbe54_66cf_34e9_0c6c,
            1.0 / 4096.0,
        );
        let dt_bias = values(
            QWEN35_GDN_CORE_HEAD_SCALAR_TRACE_ELEMENTS_V1,
            0xc0ac_29b7_c97c_50dd,
            1.0 / 4096.0,
        );
        let recurrent_state = values(
            QWEN35_GDN_CORE_RECURRENT_TRACE_ELEMENTS_V1,
            0x3f84_d5b5_b547_0917,
            1.0 / 65_536.0,
        );
        let norm_weight = values(
            QWEN35_GDN_CORE_NORM_WEIGHT_TRACE_ELEMENTS_V1,
            0x9216_d5d9_8979_fb1b,
            1.0 / 4096.0,
        );

        let mut primitive = MetalGdnCoreFusedCount18PrimitiveV1::new().unwrap();
        primitive
            .verify_invalid_raw_selectors_fail_closed()
            .unwrap();
        primitive
            .stage_fixture(
                &projected,
                &conv_weight,
                &query_state,
                &key_state,
                &value_state,
                &a_log,
                &dt_bias,
                &recurrent_state,
                &norm_weight,
            )
            .unwrap();
        primitive
            .verify_invalid_raw_selectors_fail_closed()
            .unwrap();
        assert!(primitive.snapshot().is_err());

        let mut snapshots = Vec::new();
        for profile in GdnCoreProfileV1::ALL {
            primitive.poison_outputs().unwrap();
            primitive.run(profile).unwrap();
            let snapshot = primitive.snapshot().unwrap();
            primitive
                .verify_fixture_unchanged(
                    &projected,
                    &conv_weight,
                    &query_state,
                    &key_state,
                    &value_state,
                    &a_log,
                    &dt_bias,
                    &recurrent_state,
                    &norm_weight,
                )
                .unwrap();
            snapshots.push(snapshot);
        }

        for (profile, snapshot) in GdnCoreProfileV1::ALL.into_iter().zip(&snapshots).skip(1) {
            assert_snapshot_bits(profile, &snapshots[0], snapshot);
        }
        primitive
            .verify_invalid_raw_selectors_fail_closed()
            .unwrap();
        for profile in GdnCoreProfileV1::ALL {
            let receipt = primitive.runtime_receipt(profile).unwrap();
            assert_eq!(receipt.requested_profile, profile);
            assert_eq!(receipt.observed_profile, profile);
            assert_eq!(
                receipt.observed_function_chain,
                profile.expected_function_chain()
            );
            assert_eq!(receipt.successful_runs, 1);
        }
    }
}
