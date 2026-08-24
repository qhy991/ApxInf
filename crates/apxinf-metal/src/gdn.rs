use super::{MetalW8Error, PackedW8Rows, W8GroupSize, W8_GROUP_SIZE};

const L2_NORM_EPS: f32 = 1.0e-6;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GdnDimensions {
    pub hidden_size: usize,
    pub key_heads: usize,
    pub value_heads: usize,
    pub key_dim: usize,
    pub value_dim: usize,
    pub conv_kernel_size: usize,
    pub rms_norm_eps: f32,
}

impl GdnDimensions {
    pub fn key_width(self) -> usize {
        self.key_heads.saturating_mul(self.key_dim)
    }

    pub fn value_width(self) -> usize {
        self.value_heads.saturating_mul(self.value_dim)
    }

    pub fn qkv_width(self) -> usize {
        self.key_width()
            .saturating_mul(2)
            .saturating_add(self.value_width())
    }

    pub fn input_projection_rows(self) -> usize {
        self.qkv_width()
            .saturating_add(self.value_width())
            .saturating_add(self.value_heads.saturating_mul(2))
    }

    fn validate(self) -> Result<(), MetalW8Error> {
        for (label, value) in [
            ("hidden_size", self.hidden_size),
            ("key_heads", self.key_heads),
            ("value_heads", self.value_heads),
            ("key_dim", self.key_dim),
            ("value_dim", self.value_dim),
            ("conv_kernel_size", self.conv_kernel_size),
        ] {
            if value == 0 {
                return Err(MetalW8Error::new(format!(
                    "Metal W8 GDN {label} must be positive"
                )));
            }
        }
        if self.value_heads % self.key_heads != 0 {
            return Err(MetalW8Error::new(format!(
                "Metal W8 GDN value_heads {} must be divisible by key_heads {}",
                self.value_heads, self.key_heads
            )));
        }
        let key_width = self
            .key_heads
            .checked_mul(self.key_dim)
            .ok_or_else(|| MetalW8Error::new("Metal W8 GDN dimensions overflow"))?;
        let value_width = self
            .value_heads
            .checked_mul(self.value_dim)
            .ok_or_else(|| MetalW8Error::new("Metal W8 GDN dimensions overflow"))?;
        let qkv_width = key_width
            .checked_mul(2)
            .and_then(|value| value.checked_add(value_width))
            .ok_or_else(|| MetalW8Error::new("Metal W8 GDN dimensions overflow"))?;
        let input_rows = qkv_width
            .checked_add(value_width)
            .and_then(|value| value.checked_add(self.value_heads.checked_mul(2)?))
            .ok_or_else(|| MetalW8Error::new("Metal W8 GDN dimensions overflow"))?;
        if self.hidden_size % W8_GROUP_SIZE != 0 || value_width % W8_GROUP_SIZE != 0 {
            return Err(MetalW8Error::new(format!(
                "Metal W8 GDN hidden_size and value width must be divisible by {W8_GROUP_SIZE}"
            )));
        }
        if !self.rms_norm_eps.is_finite() || self.rms_norm_eps < 0.0 {
            return Err(MetalW8Error::new(
                "Metal W8 GDN rms_norm_eps must be finite and non-negative",
            ));
        }
        input_rows
            .checked_mul(self.hidden_size)
            .and_then(|_| self.hidden_size.checked_mul(value_width))
            .and_then(|_| qkv_width.checked_mul(self.conv_kernel_size))
            .and_then(|_| key_width.checked_mul(self.conv_kernel_size))
            .and_then(|_| value_width.checked_mul(self.conv_kernel_size))
            .and_then(|_| self.value_heads.checked_mul(self.key_dim))
            .and_then(|value| value.checked_mul(self.value_dim))
            .ok_or_else(|| MetalW8Error::new("Metal W8 GDN dimensions overflow"))?;
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct GdnDecodeState {
    query_conv: Vec<f32>,
    key_conv: Vec<f32>,
    value_conv: Vec<f32>,
    recurrent: Vec<f32>,
}

impl GdnDecodeState {
    pub fn zeroed(dims: GdnDimensions) -> Result<Self, MetalW8Error> {
        dims.validate()?;
        Ok(Self {
            query_conv: vec![0.0; dims.conv_kernel_size * dims.key_width()],
            key_conv: vec![0.0; dims.conv_kernel_size * dims.key_width()],
            value_conv: vec![0.0; dims.conv_kernel_size * dims.value_width()],
            recurrent: vec![0.0; dims.value_heads * dims.key_dim * dims.value_dim],
        })
    }

    pub fn from_parts(
        dims: GdnDimensions,
        query_conv: Vec<f32>,
        key_conv: Vec<f32>,
        value_conv: Vec<f32>,
        recurrent: Vec<f32>,
    ) -> Result<Self, MetalW8Error> {
        let state = Self {
            query_conv,
            key_conv,
            value_conv,
            recurrent,
        };
        state.validate(dims)?;
        Ok(state)
    }

    pub fn query_conv(&self) -> &[f32] {
        &self.query_conv
    }

    pub fn key_conv(&self) -> &[f32] {
        &self.key_conv
    }

    pub fn value_conv(&self) -> &[f32] {
        &self.value_conv
    }

    pub fn recurrent(&self) -> &[f32] {
        &self.recurrent
    }

    fn validate(&self, dims: GdnDimensions) -> Result<(), MetalW8Error> {
        dims.validate()?;
        let expected = [
            ("query conv", dims.conv_kernel_size * dims.key_width()),
            ("key conv", dims.conv_kernel_size * dims.key_width()),
            ("value conv", dims.conv_kernel_size * dims.value_width()),
            (
                "recurrent",
                dims.value_heads * dims.key_dim * dims.value_dim,
            ),
        ];
        for ((label, expected), actual) in expected.into_iter().zip([
            self.query_conv.len(),
            self.key_conv.len(),
            self.value_conv.len(),
            self.recurrent.len(),
        ]) {
            if actual != expected {
                return Err(MetalW8Error::new(format!(
                    "Metal W8 GDN {label} state has {actual} elements, expected {expected}"
                )));
            }
        }
        for (label, values) in [
            ("query conv", self.query_conv.as_slice()),
            ("key conv", self.key_conv.as_slice()),
            ("value conv", self.value_conv.as_slice()),
            ("recurrent", self.recurrent.as_slice()),
        ] {
            if let Some(index) = values.iter().position(|value| !value.is_finite()) {
                return Err(MetalW8Error::new(format!(
                    "Metal W8 GDN {label} state contains a non-finite value at element {index}"
                )));
            }
        }
        Ok(())
    }
}

pub struct GdnF32Weights<'a> {
    /// Canonical output-row order: `q, k, v, z, a, b`.
    pub input_projection: &'a [f32],
    pub output_projection: &'a [f32],
    /// Canonical channel-row order: `q, k, v`, each row containing K taps.
    pub conv_weight: &'a [f32],
    pub a_log: &'a [f32],
    pub dt_bias: &'a [f32],
    pub norm_weight: &'a [f32],
}

#[derive(Clone, Debug)]
pub struct PackedW8GdnBlock {
    pub(crate) dims: GdnDimensions,
    pub(crate) input_projection: PackedW8Rows,
    pub(crate) output_projection: PackedW8Rows,
    pub(crate) conv_weight: Vec<f32>,
    pub(crate) a_log: Vec<f32>,
    pub(crate) dt_bias: Vec<f32>,
    pub(crate) norm_weight: Vec<f32>,
}

pub struct GdnDecodeResult {
    pub output: Vec<f32>,
    pub state: GdnDecodeState,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GdnMetalStats {
    pub decode_calls: usize,
    pub command_buffers: usize,
    pub waits: usize,
    pub committed_state_version: u64,
}

/// Independent, decode-only GDN attention tracer. Nothing in the model loader
/// or default runtime constructs this type.
pub struct MetalW8GdnBlock {
    dims: GdnDimensions,
    inner: platform::GdnHandle,
    output: Vec<f32>,
    seeded: bool,
    stats: GdnMetalStats,
}

impl PackedW8GdnBlock {
    pub fn pack_f32(dims: GdnDimensions, weights: GdnF32Weights<'_>) -> Result<Self, MetalW8Error> {
        Self::pack_f32_with_output_group_size(dims, weights, W8GroupSize::G64)
    }

    /// Precision-screen packer. The input projection remains canonical G64;
    /// only the output projection may opt into CPU-only G32.
    pub fn pack_f32_with_output_group_size(
        dims: GdnDimensions,
        weights: GdnF32Weights<'_>,
        output_group_size: W8GroupSize,
    ) -> Result<Self, MetalW8Error> {
        dims.validate()?;
        let expected = [
            (
                "input projection",
                dims.input_projection_rows() * dims.hidden_size,
                weights.input_projection,
            ),
            (
                "output projection",
                dims.hidden_size * dims.value_width(),
                weights.output_projection,
            ),
            (
                "convolution",
                dims.qkv_width() * dims.conv_kernel_size,
                weights.conv_weight,
            ),
            ("A_log", dims.value_heads, weights.a_log),
            ("dt_bias", dims.value_heads, weights.dt_bias),
            ("norm", dims.value_dim, weights.norm_weight),
        ];
        for (label, expected, values) in expected {
            if values.len() != expected {
                return Err(MetalW8Error::new(format!(
                    "Metal W8 GDN {label} has {} elements, expected {expected}",
                    values.len()
                )));
            }
            if let Some(index) = values.iter().position(|value| !value.is_finite()) {
                return Err(MetalW8Error::new(format!(
                    "Metal W8 GDN {label} contains a non-finite value at element {index}"
                )));
            }
        }
        Ok(Self {
            dims,
            input_projection: PackedW8Rows::pack_f32(
                weights.input_projection,
                dims.input_projection_rows(),
                dims.hidden_size,
            )?,
            output_projection: PackedW8Rows::pack_f32_with_group_size(
                weights.output_projection,
                dims.hidden_size,
                dims.value_width(),
                output_group_size,
            )?,
            conv_weight: weights.conv_weight.to_vec(),
            a_log: weights.a_log.to_vec(),
            dt_bias: weights.dt_bias.to_vec(),
            norm_weight: weights.norm_weight.to_vec(),
        })
    }

    pub fn dimensions(&self) -> GdnDimensions {
        self.dims
    }

    pub fn input_projection_group_size(&self) -> W8GroupSize {
        self.input_projection.group_size()
    }

    pub fn output_projection_group_size(&self) -> W8GroupSize {
        self.output_projection.group_size()
    }

    pub fn input_projection_scale_bytes(&self) -> usize {
        self.input_projection.scales().len() * std::mem::size_of::<f32>()
    }

    pub fn output_projection_scale_bytes(&self) -> usize {
        self.output_projection.scales().len() * std::mem::size_of::<f32>()
    }

    pub fn decode_reference(
        &self,
        input: &[f32],
        state: &GdnDecodeState,
    ) -> Result<GdnDecodeResult, MetalW8Error> {
        if input.len() != self.dims.hidden_size {
            return Err(MetalW8Error::new(format!(
                "Metal W8 GDN input has {} elements, expected {}",
                input.len(),
                self.dims.hidden_size
            )));
        }
        if let Some(index) = input.iter().position(|value| !value.is_finite()) {
            return Err(MetalW8Error::new(format!(
                "Metal W8 GDN input contains a non-finite value at element {index}"
            )));
        }
        state.validate(self.dims)?;

        let projected = self.input_projection.scores(input)?;
        let key_width = self.dims.key_width();
        let value_width = self.dims.value_width();
        let q_offset = 0;
        let k_offset = q_offset + key_width;
        let v_offset = k_offset + key_width;
        let z_offset = v_offset + value_width;
        let a_offset = z_offset + value_width;
        let b_offset = a_offset + self.dims.value_heads;

        let (query, query_conv) = depthwise_decode(
            &projected[q_offset..k_offset],
            &self.conv_weight[..key_width * self.dims.conv_kernel_size],
            &state.query_conv,
            self.dims.conv_kernel_size,
        );
        let key_conv_weight = key_width * self.dims.conv_kernel_size;
        let (key, key_conv) = depthwise_decode(
            &projected[k_offset..v_offset],
            &self.conv_weight[key_conv_weight..2 * key_conv_weight],
            &state.key_conv,
            self.dims.conv_kernel_size,
        );
        let (value, value_conv) = depthwise_decode(
            &projected[v_offset..z_offset],
            &self.conv_weight[2 * key_conv_weight..],
            &state.value_conv,
            self.dims.conv_kernel_size,
        );

        let mut query = query.into_iter().map(silu).collect::<Vec<_>>();
        let mut key = key.into_iter().map(silu).collect::<Vec<_>>();
        let value = value.into_iter().map(silu).collect::<Vec<_>>();
        normalize_heads(&mut query, self.dims.key_dim, L2_NORM_EPS);
        let query_scale = (self.dims.key_dim as f32).sqrt().recip();
        for value in &mut query {
            *value *= query_scale;
        }
        normalize_heads(&mut key, self.dims.key_dim, L2_NORM_EPS);

        let mut recurrent = state.recurrent.clone();
        let mut core = vec![0.0f32; value_width];
        let repeat_factor = self.dims.value_heads / self.dims.key_heads;
        let mut delta = vec![0.0f32; self.dims.value_dim];
        for value_head in 0..self.dims.value_heads {
            let key_head = value_head / repeat_factor;
            let beta = sigmoid(projected[b_offset + value_head]);
            let decay = (-self.a_log[value_head].exp()
                * softplus(projected[a_offset + value_head] + self.dt_bias[value_head]))
            .exp();
            let q_base = key_head * self.dims.key_dim;
            let state_base = value_head * self.dims.key_dim * self.dims.value_dim;
            let value_base = value_head * self.dims.value_dim;
            for state_value in
                &mut recurrent[state_base..state_base + self.dims.key_dim * self.dims.value_dim]
            {
                *state_value *= decay;
            }
            delta.fill(0.0);
            for key_index in 0..self.dims.key_dim {
                let row = state_base + key_index * self.dims.value_dim;
                let key_value = key[q_base + key_index];
                for value_index in 0..self.dims.value_dim {
                    delta[value_index] += recurrent[row + value_index] * key_value;
                }
            }
            for value_index in 0..self.dims.value_dim {
                delta[value_index] = (value[value_base + value_index] - delta[value_index]) * beta;
            }
            for key_index in 0..self.dims.key_dim {
                let row = state_base + key_index * self.dims.value_dim;
                let key_value = key[q_base + key_index];
                let query_value = query[q_base + key_index];
                for value_index in 0..self.dims.value_dim {
                    let state_index = row + value_index;
                    recurrent[state_index] += key_value * delta[value_index];
                    core[value_base + value_index] += recurrent[state_index] * query_value;
                }
            }
        }

        for head in 0..self.dims.value_heads {
            let row = &mut core[head * self.dims.value_dim..(head + 1) * self.dims.value_dim];
            let mean_square =
                row.iter().map(|value| value * value).sum::<f32>() / self.dims.value_dim as f32;
            let inverse_rms = (mean_square + self.dims.rms_norm_eps).sqrt().recip();
            for (value, weight) in row.iter_mut().zip(&self.norm_weight) {
                *value *= inverse_rms * weight;
            }
        }
        for index in 0..value_width {
            core[index] *= silu(projected[z_offset + index]);
        }
        let output = self.output_projection.scores(&core)?;
        Ok(GdnDecodeResult {
            output,
            state: GdnDecodeState {
                query_conv,
                key_conv,
                value_conv,
                recurrent,
            },
        })
    }
}

impl MetalW8GdnBlock {
    pub fn from_packed(weights: &PackedW8GdnBlock) -> Result<Self, MetalW8Error> {
        weights
            .input_projection
            .require_metal_g64("GDN input projection")?;
        weights
            .output_projection
            .require_metal_g64("GDN output projection")?;
        for (label, value) in [
            ("hidden_size", weights.dims.hidden_size),
            ("key_heads", weights.dims.key_heads),
            ("value_heads", weights.dims.value_heads),
            ("key_dim", weights.dims.key_dim),
            ("value_dim", weights.dims.value_dim),
            ("conv_kernel_size", weights.dims.conv_kernel_size),
            ("input_rows", weights.dims.input_projection_rows()),
        ] {
            if value > u32::MAX as usize {
                return Err(MetalW8Error::new(format!(
                    "Metal W8 GDN {label} exceeds the u32 ABI"
                )));
            }
        }
        Ok(Self {
            dims: weights.dims,
            inner: platform::GdnHandle::new(weights)?,
            output: vec![0.0; weights.dims.hidden_size],
            seeded: false,
            stats: GdnMetalStats::default(),
        })
    }

    pub fn seed_decode_state(&mut self, state: &GdnDecodeState) -> Result<(), MetalW8Error> {
        state.validate(self.dims)?;
        self.inner.seed(state)?;
        self.seeded = true;
        self.stats = GdnMetalStats::default();
        Ok(())
    }

    /// Clear a stream between requests. Decode remains fail-closed until the
    /// next CPU prefill supplies an exact state seed.
    pub fn clear_decode_state(&mut self) -> Result<(), MetalW8Error> {
        let cleared = GdnDecodeState::zeroed(self.dims)?;
        self.inner.seed(&cleared)?;
        self.output.fill(0.0);
        self.seeded = false;
        self.stats = GdnMetalStats::default();
        Ok(())
    }

    pub fn decode(&mut self, normalized_hidden: &[f32]) -> Result<&[f32], MetalW8Error> {
        self.validate_decode_input(normalized_hidden)?;
        self.inner
            .decode(normalized_hidden, &mut self.output, false)?;
        self.stats.decode_calls += 1;
        self.stats.command_buffers += 1;
        self.stats.waits += 1;
        self.stats.committed_state_version += 1;
        Ok(&self.output)
    }

    pub fn state_snapshot(&self) -> Result<GdnDecodeState, MetalW8Error> {
        if !self.seeded {
            return Err(MetalW8Error::new(
                "Metal W8 GDN state must be seeded before snapshot",
            ));
        }
        self.inner.snapshot(self.dims)
    }

    pub fn stats(&self) -> GdnMetalStats {
        self.stats
    }

    fn validate_decode_input(&self, input: &[f32]) -> Result<(), MetalW8Error> {
        if !self.seeded {
            return Err(MetalW8Error::new(
                "Metal W8 GDN decode state must be seeded after CPU prefill",
            ));
        }
        if input.len() != self.dims.hidden_size {
            return Err(MetalW8Error::new(format!(
                "Metal W8 GDN input has {} elements, expected {}",
                input.len(),
                self.dims.hidden_size
            )));
        }
        if let Some(index) = input.iter().position(|value| !value.is_finite()) {
            return Err(MetalW8Error::new(format!(
                "Metal W8 GDN input contains a non-finite value at element {index}"
            )));
        }
        Ok(())
    }

    /// Diagnostic-only fault injection used to verify state transactions in a
    /// higher-level debug build. It executes into scratch buffers and returns
    /// an error before the active/scratch swap.
    #[cfg(any(test, debug_assertions))]
    #[doc(hidden)]
    pub fn inject_failure_after_scratch_execution_for_testing(
        &mut self,
        normalized_hidden: &[f32],
    ) -> Result<(), MetalW8Error> {
        self.validate_decode_input(normalized_hidden)?;
        self.inner.decode(normalized_hidden, &mut self.output, true)
    }
}

fn depthwise_decode(
    input: &[f32],
    weight: &[f32],
    state: &[f32],
    kernel_size: usize,
) -> (Vec<f32>, Vec<f32>) {
    let channels = input.len();
    let mut output = vec![0.0f32; channels];
    for channel in 0..channels {
        let mut sum = 0.0f32;
        for kernel in 0..kernel_size {
            let sample = if kernel + 1 < kernel_size {
                state[(kernel + 1) * channels + channel]
            } else {
                input[channel]
            };
            sum += sample * weight[channel * kernel_size + kernel];
        }
        output[channel] = sum;
    }
    let mut next = vec![0.0f32; state.len()];
    for time in 0..kernel_size {
        let destination = time * channels;
        if time + 1 < kernel_size {
            let source = (time + 1) * channels;
            next[destination..destination + channels]
                .copy_from_slice(&state[source..source + channels]);
        } else {
            next[destination..destination + channels].copy_from_slice(input);
        }
    }
    (output, next)
}

fn normalize_heads(values: &mut [f32], head_dim: usize, eps: f32) {
    for row in values.chunks_exact_mut(head_dim) {
        let inverse_norm = (row.iter().map(|value| value * value).sum::<f32>() + eps)
            .sqrt()
            .recip();
        for value in row {
            *value *= inverse_norm;
        }
    }
}

fn silu(value: f32) -> f32 {
    value / (1.0 + (-value).exp())
}

fn sigmoid(value: f32) -> f32 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exp = value.exp();
        exp / (1.0 + exp)
    }
}

fn softplus(value: f32) -> f32 {
    if value > 20.0 {
        value
    } else if value < -20.0 {
        value.exp()
    } else {
        value.exp().ln_1p()
    }
}

#[cfg(target_os = "macos")]
mod platform {
    use super::{GdnDecodeState, GdnDimensions, MetalW8Error, PackedW8GdnBlock, W8_GROUP_SIZE};
    use std::ffi::{c_char, c_int, c_void, CStr};
    use std::ptr::NonNull;

    const ERROR_CAPACITY: usize = 1024;

    extern "C" {
        fn apxinf_metal_w8_gdn_create(
            input_weights: *const i8,
            input_scales: *const f32,
            output_weights: *const i8,
            output_scales: *const f32,
            conv_weight: *const f32,
            a_log: *const f32,
            dt_bias: *const f32,
            norm_weight: *const f32,
            hidden_size: u32,
            key_heads: u32,
            value_heads: u32,
            key_dim: u32,
            value_dim: u32,
            conv_kernel_size: u32,
            rms_norm_eps: f32,
            group_size: u32,
            output: *mut *mut c_void,
            error: *mut c_char,
            error_capacity: usize,
        ) -> c_int;
        fn apxinf_metal_w8_gdn_seed_state(
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
        fn apxinf_metal_w8_gdn_decode(
            handle: *mut c_void,
            input: *const f32,
            input_count: u32,
            output: *mut f32,
            output_count: u32,
            inject_failure_after_execution: u8,
            error: *mut c_char,
            error_capacity: usize,
        ) -> c_int;
        fn apxinf_metal_w8_gdn_snapshot_state(
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
        fn apxinf_metal_w8_gdn_destroy(handle: *mut c_void);
    }

    pub(super) struct GdnHandle(NonNull<c_void>);

    impl GdnHandle {
        pub(super) fn new(weights: &PackedW8GdnBlock) -> Result<Self, MetalW8Error> {
            let dims = weights.dims;
            let mut output = std::ptr::null_mut();
            let mut error = [0 as c_char; ERROR_CAPACITY];
            let status = unsafe {
                apxinf_metal_w8_gdn_create(
                    weights.input_projection.values.as_ptr(),
                    weights.input_projection.scales.as_ptr(),
                    weights.output_projection.values.as_ptr(),
                    weights.output_projection.scales.as_ptr(),
                    weights.conv_weight.as_ptr(),
                    weights.a_log.as_ptr(),
                    weights.dt_bias.as_ptr(),
                    weights.norm_weight.as_ptr(),
                    dims.hidden_size as u32,
                    dims.key_heads as u32,
                    dims.value_heads as u32,
                    dims.key_dim as u32,
                    dims.value_dim as u32,
                    dims.conv_kernel_size as u32,
                    dims.rms_norm_eps,
                    W8_GROUP_SIZE as u32,
                    &mut output,
                    error.as_mut_ptr(),
                    error.len(),
                )
            };
            if status != 0 {
                return Err(bridge_error("create Metal W8 GDN block", &error));
            }
            NonNull::new(output).map(Self).ok_or_else(|| {
                MetalW8Error::new("create Metal W8 GDN block returned a null handle")
            })
        }

        pub(super) fn seed(&mut self, state: &GdnDecodeState) -> Result<(), MetalW8Error> {
            let mut error = [0 as c_char; ERROR_CAPACITY];
            let status = unsafe {
                apxinf_metal_w8_gdn_seed_state(
                    self.0.as_ptr(),
                    state.query_conv.as_ptr(),
                    state.query_conv.len() as u32,
                    state.key_conv.as_ptr(),
                    state.key_conv.len() as u32,
                    state.value_conv.as_ptr(),
                    state.value_conv.len() as u32,
                    state.recurrent.as_ptr(),
                    state.recurrent.len() as u32,
                    error.as_mut_ptr(),
                    error.len(),
                )
            };
            if status != 0 {
                return Err(bridge_error("seed Metal W8 GDN state", &error));
            }
            Ok(())
        }

        pub(super) fn decode(
            &mut self,
            input: &[f32],
            output: &mut [f32],
            inject_failure: bool,
        ) -> Result<(), MetalW8Error> {
            let mut error = [0 as c_char; ERROR_CAPACITY];
            let status = unsafe {
                apxinf_metal_w8_gdn_decode(
                    self.0.as_ptr(),
                    input.as_ptr(),
                    input.len() as u32,
                    output.as_mut_ptr(),
                    output.len() as u32,
                    u8::from(inject_failure),
                    error.as_mut_ptr(),
                    error.len(),
                )
            };
            if status != 0 {
                return Err(bridge_error("run Metal W8 GDN decode", &error));
            }
            Ok(())
        }

        pub(super) fn snapshot(&self, dims: GdnDimensions) -> Result<GdnDecodeState, MetalW8Error> {
            let mut state = GdnDecodeState::zeroed(dims)?;
            let mut error = [0 as c_char; ERROR_CAPACITY];
            let status = unsafe {
                apxinf_metal_w8_gdn_snapshot_state(
                    self.0.as_ptr(),
                    state.query_conv.as_mut_ptr(),
                    state.query_conv.len() as u32,
                    state.key_conv.as_mut_ptr(),
                    state.key_conv.len() as u32,
                    state.value_conv.as_mut_ptr(),
                    state.value_conv.len() as u32,
                    state.recurrent.as_mut_ptr(),
                    state.recurrent.len() as u32,
                    error.as_mut_ptr(),
                    error.len(),
                )
            };
            if status != 0 {
                return Err(bridge_error("snapshot Metal W8 GDN state", &error));
            }
            Ok(state)
        }
    }

    impl Drop for GdnHandle {
        fn drop(&mut self) {
            unsafe { apxinf_metal_w8_gdn_destroy(self.0.as_ptr()) };
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

#[cfg(test)]
mod tests {
    use super::*;

    fn values(elements: usize, multiplier: usize, modulus: usize) -> Vec<f32> {
        (0..elements)
            .map(|index| {
                ((index.wrapping_mul(multiplier) % modulus) as f32 - (modulus / 2) as f32)
                    / modulus as f32
            })
            .collect()
    }

    fn fixture() -> (GdnDimensions, Vec<f32>, PackedW8GdnBlock) {
        let dims = GdnDimensions {
            hidden_size: 64,
            key_heads: 2,
            value_heads: 2,
            key_dim: 32,
            value_dim: 32,
            conv_kernel_size: 4,
            rms_norm_eps: 1.0e-6,
        };
        let input_projection = values(dims.input_projection_rows() * dims.hidden_size, 17, 251);
        let output_projection = values(dims.hidden_size * dims.value_width(), 19, 241);
        let conv_weight = values(dims.qkv_width() * dims.conv_kernel_size, 23, 127);
        let a_log = values(dims.value_heads, 29, 97);
        let dt_bias = values(dims.value_heads, 31, 89);
        let norm_weight = values(dims.value_dim, 37, 83);
        let packed = PackedW8GdnBlock::pack_f32(
            dims,
            GdnF32Weights {
                input_projection: &input_projection,
                output_projection: &output_projection,
                conv_weight: &conv_weight,
                a_log: &a_log,
                dt_bias: &dt_bias,
                norm_weight: &norm_weight,
            },
        )
        .unwrap();
        (dims, values(dims.hidden_size, 41, 79), packed)
    }

    #[test]
    fn bridge_source_has_one_command_buffer_one_commit_and_one_wait() {
        let bridge = include_str!("metal_w8_gdn_bridge.mm");
        let shader = include_str!("metal_w8_gdn.metal");
        assert_eq!(bridge.matches("[handle->queue commandBuffer]").count(), 1);
        assert_eq!(bridge.matches("[command commit]").count(), 1);
        assert_eq!(bridge.matches("[command waitUntilCompleted]").count(), 1);
        for kernel in [
            "gdn_w8_input_projection",
            "gdn_depthwise_preprocess",
            "gdn_normalize_qk",
            "gdn_recurrent_update",
            "gdn_norm_gate",
            "gdn_w8_output_projection",
        ] {
            assert!(shader.contains(&format!("kernel void {kernel}(")));
        }
    }

    #[test]
    fn precision_screen_only_changes_the_gdn_output_projection() {
        let (dims, _, legacy) = fixture();
        let input_projection = values(dims.input_projection_rows() * dims.hidden_size, 17, 251);
        let output_projection = values(dims.hidden_size * dims.value_width(), 19, 241);
        let conv_weight = values(dims.qkv_width() * dims.conv_kernel_size, 23, 127);
        let a_log = values(dims.value_heads, 29, 97);
        let dt_bias = values(dims.value_heads, 31, 89);
        let norm_weight = values(dims.value_dim, 37, 83);
        let screened = PackedW8GdnBlock::pack_f32_with_output_group_size(
            dims,
            GdnF32Weights {
                input_projection: &input_projection,
                output_projection: &output_projection,
                conv_weight: &conv_weight,
                a_log: &a_log,
                dt_bias: &dt_bias,
                norm_weight: &norm_weight,
            },
            crate::W8GroupSize::G32,
        )
        .unwrap();

        assert_eq!(
            screened.input_projection_group_size(),
            crate::W8GroupSize::G64
        );
        assert_eq!(
            screened.output_projection_group_size(),
            crate::W8GroupSize::G32
        );
        assert_eq!(
            screened.output_projection_scale_bytes(),
            legacy.output_projection_scale_bytes() * 2
        );
        let error = MetalW8GdnBlock::from_packed(&screened)
            .err()
            .expect("legacy Metal GDN must reject CPU-only g32 weights");
        assert!(error.to_string().contains("group size 64"));
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn injected_failure_after_scratch_execution_does_not_commit_state() {
        let (dims, input, packed) = fixture();
        let initial = GdnDecodeState::zeroed(dims).unwrap();
        let mut metal = MetalW8GdnBlock::from_packed(&packed).unwrap();
        metal.seed_decode_state(&initial).unwrap();
        let before = metal.state_snapshot().unwrap();
        let error = metal
            .inject_failure_after_scratch_execution_for_testing(&input)
            .unwrap_err();
        assert!(error.to_string().contains("injected"));
        assert_eq!(metal.state_snapshot().unwrap(), before);
        assert_eq!(metal.stats(), GdnMetalStats::default());

        let expected = packed.decode_reference(&input, &initial).unwrap();
        let actual = metal.decode(&input).unwrap();
        for (&actual, &expected) in actual.iter().zip(&expected.output) {
            assert!((actual - expected).abs() <= 3.0e-4);
        }
    }
}

#[cfg(not(target_os = "macos"))]
mod platform {
    use super::{GdnDecodeState, GdnDimensions, MetalW8Error, PackedW8GdnBlock};

    pub(super) struct GdnHandle;

    impl GdnHandle {
        pub(super) fn new(_weights: &PackedW8GdnBlock) -> Result<Self, MetalW8Error> {
            Err(MetalW8Error::new("Metal W8 GDN block requires macOS"))
        }

        pub(super) fn seed(&mut self, _state: &GdnDecodeState) -> Result<(), MetalW8Error> {
            Err(MetalW8Error::new("Metal W8 GDN block requires macOS"))
        }

        pub(super) fn decode(
            &mut self,
            _input: &[f32],
            _output: &mut [f32],
            _inject_failure: bool,
        ) -> Result<(), MetalW8Error> {
            Err(MetalW8Error::new("Metal W8 GDN block requires macOS"))
        }

        pub(super) fn snapshot(
            &self,
            _dims: GdnDimensions,
        ) -> Result<GdnDecodeState, MetalW8Error> {
            Err(MetalW8Error::new("Metal W8 GDN block requires macOS"))
        }
    }
}
