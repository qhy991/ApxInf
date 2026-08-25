//! Fixed-shape Qwen3.5 full-attention decode boundary for the six full layers.
//!
//! The primitive deliberately has no internal sequence cursor. `start_pos` is
//! supplied on every seed/decode call by the model-wide cache authority. A
//! failed decode may have dirtied the physical KV row at `start_pos`, but that
//! row remains logically unreachable and the same `start_pos` retry overwrites
//! it. Packed Q/G/K/V rows are canonical and already de-interleaved by the
//! caller: `[Q rows | gate rows | K rows | V rows]` for each layer.

use crate::{MetalW8Error, PackedW8Rows, W8_GROUP_SIZE};

pub const QWEN35_FULL_ATTENTION_LAYER_SLOTS_V1: usize = 6;
pub const QWEN35_FULL_ATTENTION_HIDDEN_SIZE_V1: usize = 1024;
pub const QWEN35_FULL_ATTENTION_QUERY_HEADS_V1: usize = 8;
pub const QWEN35_FULL_ATTENTION_KV_HEADS_V1: usize = 2;
pub const QWEN35_FULL_ATTENTION_HEAD_DIM_V1: usize = 256;
pub const QWEN35_FULL_ATTENTION_ROTARY_DIM_V1: usize = 64;
pub const QWEN35_FULL_ATTENTION_QUERY_WIDTH_V1: usize =
    QWEN35_FULL_ATTENTION_QUERY_HEADS_V1 * QWEN35_FULL_ATTENTION_HEAD_DIM_V1;
pub const QWEN35_FULL_ATTENTION_KV_WIDTH_V1: usize =
    QWEN35_FULL_ATTENTION_KV_HEADS_V1 * QWEN35_FULL_ATTENTION_HEAD_DIM_V1;
pub const QWEN35_FULL_ATTENTION_QGKV_ROWS_PER_LAYER_V1: usize =
    2 * QWEN35_FULL_ATTENTION_QUERY_WIDTH_V1 + 2 * QWEN35_FULL_ATTENTION_KV_WIDTH_V1;
pub const QWEN35_FULL_ATTENTION_ROPE_THETA_V1: f32 = 10_000_000.0;
pub const QWEN35_FULL_ATTENTION_RMS_NORM_EPS_V1: f32 = 1.0e-6;

const Q_ROW_OFFSET: usize = 0;
const GATE_ROW_OFFSET: usize = Q_ROW_OFFSET + QWEN35_FULL_ATTENTION_QUERY_WIDTH_V1;
const KEY_ROW_OFFSET: usize = GATE_ROW_OFFSET + QWEN35_FULL_ATTENTION_QUERY_WIDTH_V1;
const VALUE_ROW_OFFSET: usize = KEY_ROW_OFFSET + QWEN35_FULL_ATTENTION_KV_WIDTH_V1;

/// Borrowed checkpoint rows for one fixed-shape full-attention layer.
///
/// Every projection is row-major `[output, input]`. `query_rows` and
/// `gate_rows` must already have been separated per head from the checkpoint's
/// interleaved Q/gate representation.
#[derive(Clone, Copy, Debug)]
pub struct FullAttentionLayerF32WeightsV1<'a> {
    pub input_rms_weight: &'a [f32],
    pub query_rows: &'a [f32],
    pub gate_rows: &'a [f32],
    pub key_rows: &'a [f32],
    pub value_rows: &'a [f32],
    pub query_norm_weight: &'a [f32],
    pub key_norm_weight: &'a [f32],
    pub output_rows: &'a [f32],
}

/// Six resident layers of canonical G64 W8 attention weights.
///
/// Q/G/K/V are intentionally one packed matrix so the Metal path can issue a
/// single input-projection dispatch per layer. Rows are layer-major, then
/// `[Q | gate | K | V]` within a layer. The output matrix is also layer-major.
#[derive(Clone, Debug)]
pub struct PackedW8FullAttentionStack6V1 {
    qgkv: PackedW8Rows,
    output: PackedW8Rows,
    input_rms_weight: Vec<f32>,
    query_norm_weight: Vec<f32>,
    key_norm_weight: Vec<f32>,
}

impl PackedW8FullAttentionStack6V1 {
    pub fn pack_f32(layers: &[FullAttentionLayerF32WeightsV1<'_>]) -> Result<Self, MetalW8Error> {
        if layers.len() != QWEN35_FULL_ATTENTION_LAYER_SLOTS_V1 {
            return Err(MetalW8Error::new(format!(
                "Qwen3.5 full-attention stack requires exactly {} layers, got {}",
                QWEN35_FULL_ATTENTION_LAYER_SLOTS_V1,
                layers.len()
            )));
        }
        for (layer_slot, layer) in layers.iter().enumerate() {
            validate_layer_f32_shapes(layer_slot, layer)?;
        }

        let qgkv_value_capacity = QWEN35_FULL_ATTENTION_LAYER_SLOTS_V1
            .checked_mul(QWEN35_FULL_ATTENTION_QGKV_ROWS_PER_LAYER_V1)
            .and_then(|rows| rows.checked_mul(QWEN35_FULL_ATTENTION_HIDDEN_SIZE_V1))
            .ok_or_else(|| MetalW8Error::new("full-attention QGKV size overflow"))?;
        let qgkv_scale_capacity = qgkv_value_capacity / W8_GROUP_SIZE;
        let output_value_capacity = QWEN35_FULL_ATTENTION_LAYER_SLOTS_V1
            .checked_mul(QWEN35_FULL_ATTENTION_HIDDEN_SIZE_V1)
            .and_then(|rows| rows.checked_mul(QWEN35_FULL_ATTENTION_QUERY_WIDTH_V1))
            .ok_or_else(|| MetalW8Error::new("full-attention output size overflow"))?;
        let output_scale_capacity = output_value_capacity / W8_GROUP_SIZE;

        let mut qgkv_values = Vec::with_capacity(qgkv_value_capacity);
        let mut qgkv_scales = Vec::with_capacity(qgkv_scale_capacity);
        let mut output_values = Vec::with_capacity(output_value_capacity);
        let mut output_scales = Vec::with_capacity(output_scale_capacity);
        let mut input_rms_weight = Vec::with_capacity(
            QWEN35_FULL_ATTENTION_LAYER_SLOTS_V1 * QWEN35_FULL_ATTENTION_HIDDEN_SIZE_V1,
        );
        let mut query_norm_weight = Vec::with_capacity(
            QWEN35_FULL_ATTENTION_LAYER_SLOTS_V1 * QWEN35_FULL_ATTENTION_HEAD_DIM_V1,
        );
        let mut key_norm_weight = Vec::with_capacity(
            QWEN35_FULL_ATTENTION_LAYER_SLOTS_V1 * QWEN35_FULL_ATTENTION_HEAD_DIM_V1,
        );

        // Quantize one layer at a time so packing does not need a second
        // stack-sized F32 staging allocation.
        for layer in layers {
            let mut qgkv_f32 = Vec::with_capacity(
                QWEN35_FULL_ATTENTION_QGKV_ROWS_PER_LAYER_V1 * QWEN35_FULL_ATTENTION_HIDDEN_SIZE_V1,
            );
            qgkv_f32.extend_from_slice(layer.query_rows);
            qgkv_f32.extend_from_slice(layer.gate_rows);
            qgkv_f32.extend_from_slice(layer.key_rows);
            qgkv_f32.extend_from_slice(layer.value_rows);
            let packed_qgkv = PackedW8Rows::pack_f32(
                &qgkv_f32,
                QWEN35_FULL_ATTENTION_QGKV_ROWS_PER_LAYER_V1,
                QWEN35_FULL_ATTENTION_HIDDEN_SIZE_V1,
            )?;
            qgkv_values.extend_from_slice(packed_qgkv.values());
            qgkv_scales.extend_from_slice(packed_qgkv.scales());

            let packed_output = PackedW8Rows::pack_f32(
                layer.output_rows,
                QWEN35_FULL_ATTENTION_HIDDEN_SIZE_V1,
                QWEN35_FULL_ATTENTION_QUERY_WIDTH_V1,
            )?;
            output_values.extend_from_slice(packed_output.values());
            output_scales.extend_from_slice(packed_output.scales());
            input_rms_weight.extend_from_slice(layer.input_rms_weight);
            query_norm_weight.extend_from_slice(layer.query_norm_weight);
            key_norm_weight.extend_from_slice(layer.key_norm_weight);
        }

        debug_assert_eq!(qgkv_values.len(), qgkv_value_capacity);
        debug_assert_eq!(qgkv_scales.len(), qgkv_scale_capacity);
        debug_assert_eq!(output_values.len(), output_value_capacity);
        debug_assert_eq!(output_scales.len(), output_scale_capacity);
        Ok(Self {
            qgkv: PackedW8Rows {
                rows: QWEN35_FULL_ATTENTION_LAYER_SLOTS_V1
                    * QWEN35_FULL_ATTENTION_QGKV_ROWS_PER_LAYER_V1,
                columns: QWEN35_FULL_ATTENTION_HIDDEN_SIZE_V1,
                group_size: crate::W8GroupSize::G64,
                values: qgkv_values,
                scales: qgkv_scales,
            },
            output: PackedW8Rows {
                rows: QWEN35_FULL_ATTENTION_LAYER_SLOTS_V1 * QWEN35_FULL_ATTENTION_HIDDEN_SIZE_V1,
                columns: QWEN35_FULL_ATTENTION_QUERY_WIDTH_V1,
                group_size: crate::W8GroupSize::G64,
                values: output_values,
                scales: output_scales,
            },
            input_rms_weight,
            query_norm_weight,
            key_norm_weight,
        })
    }

    pub fn qgkv_rows(&self) -> &PackedW8Rows {
        &self.qgkv
    }

    pub fn output_rows(&self) -> &PackedW8Rows {
        &self.output
    }

    pub fn input_rms_weight(&self) -> &[f32] {
        &self.input_rms_weight
    }

    pub fn query_norm_weight(&self) -> &[f32] {
        &self.query_norm_weight
    }

    pub fn key_norm_weight(&self) -> &[f32] {
        &self.key_norm_weight
    }

    /// Packed-weight CPU oracle using an explicit, read-only prefix.
    ///
    /// Prefix K/V are sequence-major `[start_pos, KVH, D]`. The returned
    /// `key` and `value` are the new row for `start_pos`; `residual` is the
    /// complete outer residual (`input + attention_output`).
    pub fn decode_with_prefix(
        &self,
        layer_slot: usize,
        input: &[f32],
        start_pos: u32,
        prefix_keys: &[f32],
        prefix_values: &[f32],
    ) -> Result<FullAttentionDecodeOracleV1, MetalW8Error> {
        validate_layer_slot(layer_slot)?;
        validate_finite_exact(
            "full-attention input",
            input,
            QWEN35_FULL_ATTENTION_HIDDEN_SIZE_V1,
        )?;
        let prefix_elements = prefix_elements(start_pos)?;
        validate_finite_exact("full-attention prefix keys", prefix_keys, prefix_elements)?;
        validate_finite_exact(
            "full-attention prefix values",
            prefix_values,
            prefix_elements,
        )?;

        let input_norm = layer_slice(
            &self.input_rms_weight,
            layer_slot,
            QWEN35_FULL_ATTENTION_HIDDEN_SIZE_V1,
        );
        let normalized = rms_norm_offset(input, input_norm, QWEN35_FULL_ATTENTION_RMS_NORM_EPS_V1)?;
        let qgkv_layer_row = layer_slot * QWEN35_FULL_ATTENTION_QGKV_ROWS_PER_LAYER_V1;
        let mut query = packed_scores_range(
            &self.qgkv,
            qgkv_layer_row + Q_ROW_OFFSET,
            QWEN35_FULL_ATTENTION_QUERY_WIDTH_V1,
            &normalized,
        )?;
        let gate = packed_scores_range(
            &self.qgkv,
            qgkv_layer_row + GATE_ROW_OFFSET,
            QWEN35_FULL_ATTENTION_QUERY_WIDTH_V1,
            &normalized,
        )?;
        let mut key = packed_scores_range(
            &self.qgkv,
            qgkv_layer_row + KEY_ROW_OFFSET,
            QWEN35_FULL_ATTENTION_KV_WIDTH_V1,
            &normalized,
        )?;
        let value = packed_scores_range(
            &self.qgkv,
            qgkv_layer_row + VALUE_ROW_OFFSET,
            QWEN35_FULL_ATTENTION_KV_WIDTH_V1,
            &normalized,
        )?;

        let query_norm = layer_slice(
            &self.query_norm_weight,
            layer_slot,
            QWEN35_FULL_ATTENTION_HEAD_DIM_V1,
        );
        let key_norm = layer_slice(
            &self.key_norm_weight,
            layer_slot,
            QWEN35_FULL_ATTENTION_HEAD_DIM_V1,
        );
        rms_norm_offset_in_place(
            &mut query,
            QWEN35_FULL_ATTENTION_HEAD_DIM_V1,
            query_norm,
            QWEN35_FULL_ATTENTION_RMS_NORM_EPS_V1,
        )?;
        rms_norm_offset_in_place(
            &mut key,
            QWEN35_FULL_ATTENTION_HEAD_DIM_V1,
            key_norm,
            QWEN35_FULL_ATTENTION_RMS_NORM_EPS_V1,
        )?;
        apply_partial_rope(
            &mut query,
            QWEN35_FULL_ATTENTION_QUERY_HEADS_V1,
            QWEN35_FULL_ATTENTION_HEAD_DIM_V1,
            QWEN35_FULL_ATTENTION_ROTARY_DIM_V1,
            QWEN35_FULL_ATTENTION_ROPE_THETA_V1,
            start_pos,
        )?;
        apply_partial_rope(
            &mut key,
            QWEN35_FULL_ATTENTION_KV_HEADS_V1,
            QWEN35_FULL_ATTENTION_HEAD_DIM_V1,
            QWEN35_FULL_ATTENTION_ROTARY_DIM_V1,
            QWEN35_FULL_ATTENTION_ROPE_THETA_V1,
            start_pos,
        )?;
        ensure_finite("full-attention query", &query)?;
        ensure_finite("full-attention appended key", &key)?;
        ensure_finite("full-attention appended value", &value)?;

        let mut attention = sdpa_decode_seq_major(
            &query,
            prefix_keys,
            prefix_values,
            &key,
            &value,
            start_pos as usize,
            QWEN35_FULL_ATTENTION_QUERY_HEADS_V1,
            QWEN35_FULL_ATTENTION_KV_HEADS_V1,
            QWEN35_FULL_ATTENTION_HEAD_DIM_V1,
        )?;
        for (attention_value, gate_value) in attention.iter_mut().zip(gate) {
            *attention_value *= sigmoid(gate_value);
        }
        ensure_finite("full-attention gated SDPA output", &attention)?;

        let output_layer_row = layer_slot * QWEN35_FULL_ATTENTION_HIDDEN_SIZE_V1;
        let projected = packed_scores_range(
            &self.output,
            output_layer_row,
            QWEN35_FULL_ATTENTION_HIDDEN_SIZE_V1,
            &attention,
        )?;
        let residual: Vec<f32> = input
            .iter()
            .zip(projected)
            .map(|(&input_value, attention_value)| input_value + attention_value)
            .collect();
        ensure_finite("full-attention residual", &residual)?;
        Ok(FullAttentionDecodeOracleV1 {
            residual,
            key,
            value,
        })
    }

    /// CPU oracle that reads exactly `0..start_pos` from `state`, then commits
    /// the new KV row only after the complete residual has been validated.
    /// Existing rows at and after `start_pos` are truncated/overwritten, which
    /// makes retries and model-wide rollback obey the explicit cursor.
    pub fn decode_with_state(
        &self,
        layer_slot: usize,
        input: &[f32],
        start_pos: u32,
        state: &mut FullAttentionKvStateV1,
    ) -> Result<FullAttentionDecodeOracleV1, MetalW8Error> {
        state.validate_decode_position(layer_slot, start_pos)?;
        let prefix_elements = prefix_elements(start_pos)?;
        let (keys, values) = state.layer_cache(layer_slot)?;
        let result = self.decode_with_prefix(
            layer_slot,
            input,
            start_pos,
            &keys[..prefix_elements],
            &values[..prefix_elements],
        )?;
        state.commit_token(layer_slot, start_pos, &result.key, &result.value)?;
        Ok(result)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct FullAttentionDecodeOracleV1 {
    pub residual: Vec<f32>,
    pub key: Vec<f32>,
    pub value: Vec<f32>,
}

#[derive(Clone, Debug, Default, PartialEq)]
struct FullAttentionLayerKvV1 {
    keys: Vec<f32>,
    values: Vec<f32>,
}

/// Sparse host-side oracle state. Construction does not allocate the maximum
/// cache; each layer grows only as it is seeded or decoded.
#[derive(Clone, Debug, PartialEq)]
pub struct FullAttentionKvStateV1 {
    max_context: usize,
    layers: [FullAttentionLayerKvV1; QWEN35_FULL_ATTENTION_LAYER_SLOTS_V1],
}

impl FullAttentionKvStateV1 {
    pub fn new(max_context: usize) -> Result<Self, MetalW8Error> {
        validate_max_context(max_context)?;
        Ok(Self {
            max_context,
            layers: std::array::from_fn(|_| FullAttentionLayerKvV1::default()),
        })
    }

    pub fn max_context(&self) -> usize {
        self.max_context
    }

    /// Replace one layer with a complete sequence-major prefix of exactly
    /// `start_pos` tokens. `start_pos == 0` and empty slices are valid.
    pub fn seed_cache(
        &mut self,
        layer_slot: usize,
        start_pos: u32,
        keys: &[f32],
        values: &[f32],
    ) -> Result<(), MetalW8Error> {
        validate_layer_slot(layer_slot)?;
        if start_pos as usize > self.max_context {
            return Err(MetalW8Error::new(format!(
                "full-attention seed position {start_pos} exceeds max_context {}",
                self.max_context
            )));
        }
        let expected = prefix_elements(start_pos)?;
        validate_finite_exact("full-attention seed keys", keys, expected)?;
        validate_finite_exact("full-attention seed values", values, expected)?;
        self.layers[layer_slot].keys.clear();
        self.layers[layer_slot].keys.extend_from_slice(keys);
        self.layers[layer_slot].values.clear();
        self.layers[layer_slot].values.extend_from_slice(values);
        Ok(())
    }

    pub fn layer_cache(&self, layer_slot: usize) -> Result<(&[f32], &[f32]), MetalW8Error> {
        validate_layer_slot(layer_slot)?;
        let layer = &self.layers[layer_slot];
        Ok((&layer.keys, &layer.values))
    }

    pub fn cached_tokens(&self, layer_slot: usize) -> Result<usize, MetalW8Error> {
        let (keys, values) = self.layer_cache(layer_slot)?;
        if keys.len() != values.len() || keys.len() % QWEN35_FULL_ATTENTION_KV_WIDTH_V1 != 0 {
            return Err(MetalW8Error::new(
                "full-attention host KV state is internally inconsistent",
            ));
        }
        Ok(keys.len() / QWEN35_FULL_ATTENTION_KV_WIDTH_V1)
    }

    fn validate_decode_position(
        &self,
        layer_slot: usize,
        start_pos: u32,
    ) -> Result<(), MetalW8Error> {
        validate_layer_slot(layer_slot)?;
        if start_pos as usize >= self.max_context {
            return Err(MetalW8Error::new(format!(
                "full-attention decode position {start_pos} is outside max_context {}",
                self.max_context
            )));
        }
        let needed = prefix_elements(start_pos)?;
        let layer = &self.layers[layer_slot];
        if layer.keys.len() < needed || layer.values.len() < needed {
            return Err(MetalW8Error::new(format!(
                "full-attention layer {layer_slot} has only {} key and {} value elements, but start_pos {start_pos} requires {needed}",
                layer.keys.len(),
                layer.values.len()
            )));
        }
        Ok(())
    }

    fn commit_token(
        &mut self,
        layer_slot: usize,
        start_pos: u32,
        key: &[f32],
        value: &[f32],
    ) -> Result<(), MetalW8Error> {
        self.validate_decode_position(layer_slot, start_pos)?;
        validate_finite_exact(
            "full-attention committed key",
            key,
            QWEN35_FULL_ATTENTION_KV_WIDTH_V1,
        )?;
        validate_finite_exact(
            "full-attention committed value",
            value,
            QWEN35_FULL_ATTENTION_KV_WIDTH_V1,
        )?;
        let prefix = prefix_elements(start_pos)?;
        let layer = &mut self.layers[layer_slot];
        layer.keys.truncate(prefix);
        layer.values.truncate(prefix);
        layer.keys.extend_from_slice(key);
        layer.values.extend_from_slice(value);
        Ok(())
    }
}

/// Live topology and last-success identity returned by the Metal bridge.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FullAttentionStack6RuntimeReceiptV1 {
    pub layer_slots: u32,
    pub hidden_size: u32,
    pub query_heads: u32,
    pub kv_heads: u32,
    pub head_dim: u32,
    pub rotary_dim: u32,
    pub max_context: u32,
    pub group_size: u32,
    pub command_buffers_per_decode: u32,
    pub compute_encoders_per_decode: u32,
    pub kernel_dispatches_per_decode: u32,
    pub explicit_buffer_barriers_per_decode: u32,
    pub commits_per_decode: u32,
    pub waits_per_decode: u32,
    pub fixed_shape_validated: bool,
    pub successful_decodes: u64,
    pub last_layer_slot: u32,
    pub last_start_pos: u32,
    pub last_kv_length: u32,
    pub last_observed_command_buffers: u32,
    pub last_observed_compute_encoders: u32,
    pub last_observed_kernel_dispatches: u32,
    pub last_observed_explicit_buffer_barriers: u32,
    pub last_observed_commits: u32,
    pub last_observed_waits: u32,
}

/// Persistent Metal resources for the six Qwen3.5 full-attention layer slots.
pub struct MetalW8FullAttentionStack6V1 {
    inner: platform::Handle,
    max_context: usize,
    output: Vec<f32>,
}

impl MetalW8FullAttentionStack6V1 {
    pub fn from_packed(
        weights: &PackedW8FullAttentionStack6V1,
        max_context: usize,
    ) -> Result<Self, MetalW8Error> {
        validate_packed_stack(weights)?;
        validate_max_context(max_context)?;
        Ok(Self {
            inner: platform::Handle::new(weights, max_context)?,
            max_context,
            output: vec![0.0; QWEN35_FULL_ATTENTION_HIDDEN_SIZE_V1],
        })
    }

    pub fn from_f32_layers(
        layers: &[FullAttentionLayerF32WeightsV1<'_>],
        max_context: usize,
    ) -> Result<Self, MetalW8Error> {
        let packed = PackedW8FullAttentionStack6V1::pack_f32(layers)?;
        Self::from_packed(&packed, max_context)
    }

    pub fn max_context(&self) -> usize {
        self.max_context
    }

    /// Replace one layer's complete prefix. The prefix is sequence-major
    /// `[start_pos, KVH, D]`; an empty prefix at `start_pos == 0` is valid.
    pub fn seed_cache(
        &mut self,
        layer_slot: usize,
        start_pos: u32,
        keys: &[f32],
        values: &[f32],
    ) -> Result<(), MetalW8Error> {
        validate_layer_slot(layer_slot)?;
        if start_pos as usize > self.max_context {
            return Err(MetalW8Error::new(format!(
                "full-attention seed position {start_pos} exceeds max_context {}",
                self.max_context
            )));
        }
        let expected = prefix_elements(start_pos)?;
        validate_finite_exact("full-attention seed keys", keys, expected)?;
        validate_finite_exact("full-attention seed values", values, expected)?;
        self.inner.seed_cache(layer_slot, start_pos, keys, values)
    }

    /// Decode one layer at the explicit global `start_pos` and return the
    /// complete residual row. No internal cursor is read or advanced.
    pub fn decode(
        &mut self,
        layer_slot: usize,
        input: &[f32],
        start_pos: u32,
    ) -> Result<&[f32], MetalW8Error> {
        validate_layer_slot(layer_slot)?;
        if start_pos as usize >= self.max_context {
            return Err(MetalW8Error::new(format!(
                "full-attention decode position {start_pos} is outside max_context {}",
                self.max_context
            )));
        }
        validate_finite_exact(
            "full-attention input",
            input,
            QWEN35_FULL_ATTENTION_HIDDEN_SIZE_V1,
        )?;
        self.inner
            .decode(layer_slot, input, start_pos, &mut self.output)?;
        ensure_finite("Metal full-attention residual", &self.output)?;
        // A successful call must also produce the predeclared live topology.
        self.inner.runtime_receipt(self.max_context)?;
        Ok(&self.output)
    }

    pub fn runtime_receipt(&self) -> Result<FullAttentionStack6RuntimeReceiptV1, MetalW8Error> {
        self.inner.runtime_receipt(self.max_context)
    }

    /// Copy one physical sequence-major KV row back for correctness checks.
    /// This is diagnostic custody, not cursor state: rows at or beyond the
    /// model's global cursor may contain unreachable data from a failed call.
    pub fn snapshot_cache_row(
        &self,
        layer_slot: usize,
        position: u32,
    ) -> Result<(Vec<f32>, Vec<f32>), MetalW8Error> {
        validate_layer_slot(layer_slot)?;
        if position as usize >= self.max_context {
            return Err(MetalW8Error::new(format!(
                "full-attention cache snapshot position {position} is outside max_context {}",
                self.max_context
            )));
        }
        let (keys, values) = self.inner.snapshot_cache_row(layer_slot, position)?;
        validate_finite_exact(
            "Metal full-attention cache snapshot key",
            &keys,
            QWEN35_FULL_ATTENTION_KV_WIDTH_V1,
        )?;
        validate_finite_exact(
            "Metal full-attention cache snapshot value",
            &values,
            QWEN35_FULL_ATTENTION_KV_WIDTH_V1,
        )?;
        Ok((keys, values))
    }
}

fn validate_layer_f32_shapes(
    layer_slot: usize,
    layer: &FullAttentionLayerF32WeightsV1<'_>,
) -> Result<(), MetalW8Error> {
    for (label, values, expected) in [
        (
            "input RMS weight",
            layer.input_rms_weight,
            QWEN35_FULL_ATTENTION_HIDDEN_SIZE_V1,
        ),
        (
            "query rows",
            layer.query_rows,
            QWEN35_FULL_ATTENTION_QUERY_WIDTH_V1 * QWEN35_FULL_ATTENTION_HIDDEN_SIZE_V1,
        ),
        (
            "gate rows",
            layer.gate_rows,
            QWEN35_FULL_ATTENTION_QUERY_WIDTH_V1 * QWEN35_FULL_ATTENTION_HIDDEN_SIZE_V1,
        ),
        (
            "key rows",
            layer.key_rows,
            QWEN35_FULL_ATTENTION_KV_WIDTH_V1 * QWEN35_FULL_ATTENTION_HIDDEN_SIZE_V1,
        ),
        (
            "value rows",
            layer.value_rows,
            QWEN35_FULL_ATTENTION_KV_WIDTH_V1 * QWEN35_FULL_ATTENTION_HIDDEN_SIZE_V1,
        ),
        (
            "query norm weight",
            layer.query_norm_weight,
            QWEN35_FULL_ATTENTION_HEAD_DIM_V1,
        ),
        (
            "key norm weight",
            layer.key_norm_weight,
            QWEN35_FULL_ATTENTION_HEAD_DIM_V1,
        ),
        (
            "output rows",
            layer.output_rows,
            QWEN35_FULL_ATTENTION_HIDDEN_SIZE_V1 * QWEN35_FULL_ATTENTION_QUERY_WIDTH_V1,
        ),
    ] {
        validate_finite_exact(
            &format!("full-attention layer {layer_slot} {label}"),
            values,
            expected,
        )?;
    }
    Ok(())
}

fn validate_packed_stack(weights: &PackedW8FullAttentionStack6V1) -> Result<(), MetalW8Error> {
    weights.qgkv.require_metal_g64("full-attention QGKV")?;
    weights.output.require_metal_g64("full-attention output")?;
    if weights.qgkv.rows()
        != QWEN35_FULL_ATTENTION_LAYER_SLOTS_V1 * QWEN35_FULL_ATTENTION_QGKV_ROWS_PER_LAYER_V1
        || weights.qgkv.columns() != QWEN35_FULL_ATTENTION_HIDDEN_SIZE_V1
        || weights.output.rows()
            != QWEN35_FULL_ATTENTION_LAYER_SLOTS_V1 * QWEN35_FULL_ATTENTION_HIDDEN_SIZE_V1
        || weights.output.columns() != QWEN35_FULL_ATTENTION_QUERY_WIDTH_V1
        || weights.input_rms_weight.len()
            != QWEN35_FULL_ATTENTION_LAYER_SLOTS_V1 * QWEN35_FULL_ATTENTION_HIDDEN_SIZE_V1
        || weights.query_norm_weight.len()
            != QWEN35_FULL_ATTENTION_LAYER_SLOTS_V1 * QWEN35_FULL_ATTENTION_HEAD_DIM_V1
        || weights.key_norm_weight.len()
            != QWEN35_FULL_ATTENTION_LAYER_SLOTS_V1 * QWEN35_FULL_ATTENTION_HEAD_DIM_V1
    {
        return Err(MetalW8Error::new(
            "fixed-shape full-attention packed stack has incompatible shapes",
        ));
    }
    ensure_finite(
        "full-attention input RMS weights",
        &weights.input_rms_weight,
    )?;
    ensure_finite(
        "full-attention query norm weights",
        &weights.query_norm_weight,
    )?;
    ensure_finite("full-attention key norm weights", &weights.key_norm_weight)
}

fn validate_max_context(max_context: usize) -> Result<(), MetalW8Error> {
    let ffi_count_limit = (u32::MAX as usize) / QWEN35_FULL_ATTENTION_KV_WIDTH_V1;
    if max_context == 0 || max_context > ffi_count_limit {
        return Err(MetalW8Error::new(format!(
            "full-attention max_context must be in 1..={ffi_count_limit} so seed element counts fit the u32 ABI, got {max_context}"
        )));
    }
    QWEN35_FULL_ATTENTION_LAYER_SLOTS_V1
        .checked_mul(max_context)
        .and_then(|tokens| tokens.checked_mul(QWEN35_FULL_ATTENTION_KV_WIDTH_V1))
        .and_then(|elements| elements.checked_mul(2))
        .and_then(|elements| elements.checked_mul(std::mem::size_of::<f32>()))
        .ok_or_else(|| MetalW8Error::new("full-attention KV allocation size overflow"))?;
    Ok(())
}

fn validate_layer_slot(layer_slot: usize) -> Result<(), MetalW8Error> {
    if layer_slot >= QWEN35_FULL_ATTENTION_LAYER_SLOTS_V1 {
        return Err(MetalW8Error::new(format!(
            "full-attention layer slot {layer_slot} is outside 0..{}",
            QWEN35_FULL_ATTENTION_LAYER_SLOTS_V1
        )));
    }
    Ok(())
}

fn prefix_elements(start_pos: u32) -> Result<usize, MetalW8Error> {
    (start_pos as usize)
        .checked_mul(QWEN35_FULL_ATTENTION_KV_WIDTH_V1)
        .ok_or_else(|| MetalW8Error::new("full-attention prefix size overflow"))
}

fn validate_finite_exact(label: &str, values: &[f32], expected: usize) -> Result<(), MetalW8Error> {
    if values.len() != expected {
        return Err(MetalW8Error::new(format!(
            "{label} has {} elements, expected {expected}",
            values.len()
        )));
    }
    ensure_finite(label, values)
}

fn ensure_finite(label: &str, values: &[f32]) -> Result<(), MetalW8Error> {
    if let Some(index) = values.iter().position(|value| !value.is_finite()) {
        return Err(MetalW8Error::new(format!(
            "{label} contains a non-finite value at element {index}"
        )));
    }
    Ok(())
}

fn layer_slice(values: &[f32], layer_slot: usize, width: usize) -> &[f32] {
    &values[layer_slot * width..(layer_slot + 1) * width]
}

fn packed_scores_range(
    weights: &PackedW8Rows,
    row_start: usize,
    row_count: usize,
    input: &[f32],
) -> Result<Vec<f32>, MetalW8Error> {
    if input.len() != weights.columns {
        return Err(MetalW8Error::new(format!(
            "packed projection input has {} elements, expected {}",
            input.len(),
            weights.columns
        )));
    }
    ensure_finite("packed projection input", input)?;
    let row_end = row_start
        .checked_add(row_count)
        .ok_or_else(|| MetalW8Error::new("packed projection row range overflow"))?;
    if row_end > weights.rows {
        return Err(MetalW8Error::new(format!(
            "packed projection rows {row_start}..{row_end} exceed {}",
            weights.rows
        )));
    }
    let group_columns = weights.group_size.columns();
    let groups_per_row = weights.columns / group_columns;
    let mut output = vec![0.0f32; row_count];
    for (local_row, score) in output.iter_mut().enumerate() {
        let row = row_start + local_row;
        let value_base = row * weights.columns;
        let scale_base = row * groups_per_row;
        let mut sum = 0.0f32;
        for column in 0..weights.columns {
            sum += weights.values[value_base + column] as f32
                * weights.scales[scale_base + column / group_columns]
                * input[column];
        }
        *score = sum;
    }
    ensure_finite("packed projection output", &output)?;
    Ok(output)
}

fn rms_norm_offset(input: &[f32], weight: &[f32], eps: f32) -> Result<Vec<f32>, MetalW8Error> {
    let mut output = input.to_vec();
    rms_norm_offset_in_place(&mut output, input.len(), weight, eps)?;
    Ok(output)
}

fn rms_norm_offset_in_place(
    values: &mut [f32],
    row_width: usize,
    weight: &[f32],
    eps: f32,
) -> Result<(), MetalW8Error> {
    if row_width == 0 || values.len() % row_width != 0 || weight.len() != row_width {
        return Err(MetalW8Error::new(
            "full-attention RMS norm received incompatible shapes",
        ));
    }
    if !eps.is_finite() || eps < 0.0 {
        return Err(MetalW8Error::new(
            "full-attention RMS norm epsilon must be finite and non-negative",
        ));
    }
    ensure_finite("full-attention RMS norm input", values)?;
    ensure_finite("full-attention RMS norm weight", weight)?;
    for row in values.chunks_exact_mut(row_width) {
        let mean_square = row.iter().map(|value| value * value).sum::<f32>() / row_width as f32;
        let inverse_rms = (mean_square + eps).sqrt().recip();
        for (value, &scale) in row.iter_mut().zip(weight) {
            *value = *value * inverse_rms * (scale + 1.0);
        }
    }
    ensure_finite("full-attention RMS norm output", values)
}

fn apply_partial_rope(
    values: &mut [f32],
    heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    theta: f32,
    start_pos: u32,
) -> Result<(), MetalW8Error> {
    if values.len() != heads.saturating_mul(head_dim)
        || rotary_dim == 0
        || rotary_dim > head_dim
        || rotary_dim % 2 != 0
        || !theta.is_finite()
        || theta <= 0.0
    {
        return Err(MetalW8Error::new(
            "full-attention partial RoPE received an invalid shape or parameter",
        ));
    }
    ensure_finite("full-attention partial RoPE input", values)?;
    let pair_count = rotary_dim / 2;
    for pair in 0..pair_count {
        let inverse_frequency = 1.0f32 / theta.powf(2.0 * pair as f32 / rotary_dim as f32);
        let angle = start_pos as f32 * inverse_frequency;
        let (sin, cos) = angle.sin_cos();
        for head in 0..heads {
            let base = head * head_dim;
            let first_index = base + pair;
            let second_index = base + pair_count + pair;
            let first = values[first_index];
            let second = values[second_index];
            values[first_index] = first * cos - second * sin;
            values[second_index] = first * sin + second * cos;
        }
    }
    ensure_finite("full-attention partial RoPE output", values)
}

#[allow(clippy::too_many_arguments)]
fn sdpa_decode_seq_major(
    query: &[f32],
    prefix_keys: &[f32],
    prefix_values: &[f32],
    appended_key: &[f32],
    appended_value: &[f32],
    prefix_tokens: usize,
    query_heads: usize,
    kv_heads: usize,
    head_dim: usize,
) -> Result<Vec<f32>, MetalW8Error> {
    if query_heads == 0
        || kv_heads == 0
        || query_heads % kv_heads != 0
        || head_dim == 0
        || query.len() != query_heads.saturating_mul(head_dim)
        || appended_key.len() != kv_heads.saturating_mul(head_dim)
        || appended_value.len() != kv_heads.saturating_mul(head_dim)
        || prefix_keys.len()
            != prefix_tokens
                .saturating_mul(kv_heads)
                .saturating_mul(head_dim)
        || prefix_values.len() != prefix_keys.len()
    {
        return Err(MetalW8Error::new(
            "full-attention SDPA received incompatible shapes",
        ));
    }
    for (label, values) in [
        ("SDPA query", query),
        ("SDPA prefix keys", prefix_keys),
        ("SDPA prefix values", prefix_values),
        ("SDPA appended key", appended_key),
        ("SDPA appended value", appended_value),
    ] {
        ensure_finite(label, values)?;
    }
    let kv_length = prefix_tokens
        .checked_add(1)
        .ok_or_else(|| MetalW8Error::new("full-attention SDPA length overflow"))?;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut scores = vec![0.0f32; kv_length];
    let mut output = vec![0.0f32; query_heads * head_dim];
    for query_head in 0..query_heads {
        let kv_head = query_head * kv_heads / query_heads;
        let query_base = query_head * head_dim;
        for (token, score) in scores.iter_mut().enumerate() {
            let key_row = if token < prefix_tokens {
                let base = (token * kv_heads + kv_head) * head_dim;
                &prefix_keys[base..base + head_dim]
            } else {
                let base = kv_head * head_dim;
                &appended_key[base..base + head_dim]
            };
            let mut dot = 0.0f32;
            for dimension in 0..head_dim {
                dot += query[query_base + dimension] * key_row[dimension];
            }
            *score = dot * scale;
        }
        let maximum = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exponential_sum: f32 = scores.iter().map(|score| (*score - maximum).exp()).sum();
        for score in &mut scores {
            *score = (*score - maximum).exp() / exponential_sum;
        }
        for (token, &probability) in scores.iter().enumerate() {
            let value_row = if token < prefix_tokens {
                let base = (token * kv_heads + kv_head) * head_dim;
                &prefix_values[base..base + head_dim]
            } else {
                let base = kv_head * head_dim;
                &appended_value[base..base + head_dim]
            };
            for dimension in 0..head_dim {
                output[query_base + dimension] += probability * value_row[dimension];
            }
        }
    }
    ensure_finite("full-attention SDPA output", &output)?;
    Ok(output)
}

fn sigmoid(value: f32) -> f32 {
    1.0 / (1.0 + (-value).exp())
}

#[cfg(target_os = "macos")]
mod platform {
    use super::*;
    use std::ffi::{c_char, c_int, c_void, CStr};
    use std::ptr::NonNull;

    const ERROR_CAPACITY: usize = 1024;

    #[repr(C)]
    #[derive(Clone, Copy, Debug, Default)]
    struct RawRuntimeReceiptV1 {
        layer_slots: u32,
        hidden_size: u32,
        query_heads: u32,
        kv_heads: u32,
        head_dim: u32,
        rotary_dim: u32,
        max_context: u32,
        group_size: u32,
        command_buffers_per_decode: u32,
        compute_encoders_per_decode: u32,
        kernel_dispatches_per_decode: u32,
        explicit_buffer_barriers_per_decode: u32,
        commits_per_decode: u32,
        waits_per_decode: u32,
        fixed_shape_validated: u32,
        reserved: u32,
        successful_decodes: u64,
        last_layer_slot: u32,
        last_start_pos: u32,
        last_kv_length: u32,
        last_observed_command_buffers: u32,
        last_observed_compute_encoders: u32,
        last_observed_kernel_dispatches: u32,
        last_observed_explicit_buffer_barriers: u32,
        last_observed_commits: u32,
        last_observed_waits: u32,
    }

    extern "C" {
        fn apxinf_metal_w8_full_attention_stack6_v1_create(
            qgkv_weights: *const i8,
            qgkv_scales: *const f32,
            output_weights: *const i8,
            output_scales: *const f32,
            input_rms_weight: *const f32,
            query_norm_weight: *const f32,
            key_norm_weight: *const f32,
            max_context: u32,
            group_size: u32,
            output: *mut *mut c_void,
            error: *mut c_char,
            error_capacity: usize,
        ) -> c_int;
        fn apxinf_metal_w8_full_attention_stack6_v1_seed_cache(
            handle: *mut c_void,
            layer_slot: u32,
            start_pos: u32,
            keys: *const f32,
            key_count: u32,
            values: *const f32,
            value_count: u32,
            error: *mut c_char,
            error_capacity: usize,
        ) -> c_int;
        fn apxinf_metal_w8_full_attention_stack6_v1_decode(
            handle: *mut c_void,
            layer_slot: u32,
            input: *const f32,
            input_count: u32,
            start_pos: u32,
            output: *mut f32,
            output_count: u32,
            error: *mut c_char,
            error_capacity: usize,
        ) -> c_int;
        fn apxinf_metal_w8_full_attention_stack6_v1_receipt(
            handle: *mut c_void,
            receipt: *mut RawRuntimeReceiptV1,
            error: *mut c_char,
            error_capacity: usize,
        ) -> c_int;
        fn apxinf_metal_w8_full_attention_stack6_v1_snapshot_cache_row(
            handle: *mut c_void,
            layer_slot: u32,
            position: u32,
            key_output: *mut f32,
            key_count: u32,
            value_output: *mut f32,
            value_count: u32,
            error: *mut c_char,
            error_capacity: usize,
        ) -> c_int;
        fn apxinf_metal_w8_full_attention_stack6_v1_destroy(handle: *mut c_void);
    }

    pub(super) struct Handle(NonNull<c_void>);

    impl Handle {
        pub(super) fn new(
            weights: &PackedW8FullAttentionStack6V1,
            max_context: usize,
        ) -> Result<Self, MetalW8Error> {
            let mut output = std::ptr::null_mut();
            let mut error = [0 as c_char; ERROR_CAPACITY];
            let status = unsafe {
                apxinf_metal_w8_full_attention_stack6_v1_create(
                    weights.qgkv.values().as_ptr(),
                    weights.qgkv.scales().as_ptr(),
                    weights.output.values().as_ptr(),
                    weights.output.scales().as_ptr(),
                    weights.input_rms_weight.as_ptr(),
                    weights.query_norm_weight.as_ptr(),
                    weights.key_norm_weight.as_ptr(),
                    max_context as u32,
                    W8_GROUP_SIZE as u32,
                    &mut output,
                    error.as_mut_ptr(),
                    error.len(),
                )
            };
            if status != 0 {
                return Err(bridge_error(
                    "create Metal W8 full-attention stack6",
                    &error,
                ));
            }
            NonNull::new(output).map(Self).ok_or_else(|| {
                MetalW8Error::new("create Metal W8 full-attention stack6 returned a null handle")
            })
        }

        pub(super) fn seed_cache(
            &mut self,
            layer_slot: usize,
            start_pos: u32,
            keys: &[f32],
            values: &[f32],
        ) -> Result<(), MetalW8Error> {
            let mut error = [0 as c_char; ERROR_CAPACITY];
            let key_pointer = if keys.is_empty() {
                std::ptr::null()
            } else {
                keys.as_ptr()
            };
            let value_pointer = if values.is_empty() {
                std::ptr::null()
            } else {
                values.as_ptr()
            };
            let status = unsafe {
                apxinf_metal_w8_full_attention_stack6_v1_seed_cache(
                    self.0.as_ptr(),
                    layer_slot as u32,
                    start_pos,
                    key_pointer,
                    keys.len() as u32,
                    value_pointer,
                    values.len() as u32,
                    error.as_mut_ptr(),
                    error.len(),
                )
            };
            if status != 0 {
                return Err(bridge_error("seed Metal W8 full-attention cache", &error));
            }
            Ok(())
        }

        pub(super) fn decode(
            &mut self,
            layer_slot: usize,
            input: &[f32],
            start_pos: u32,
            output: &mut [f32],
        ) -> Result<(), MetalW8Error> {
            let mut error = [0 as c_char; ERROR_CAPACITY];
            let status = unsafe {
                apxinf_metal_w8_full_attention_stack6_v1_decode(
                    self.0.as_ptr(),
                    layer_slot as u32,
                    input.as_ptr(),
                    input.len() as u32,
                    start_pos,
                    output.as_mut_ptr(),
                    output.len() as u32,
                    error.as_mut_ptr(),
                    error.len(),
                )
            };
            if status != 0 {
                return Err(bridge_error("run Metal W8 full-attention decode", &error));
            }
            Ok(())
        }

        pub(super) fn runtime_receipt(
            &self,
            max_context: usize,
        ) -> Result<FullAttentionStack6RuntimeReceiptV1, MetalW8Error> {
            let mut raw = RawRuntimeReceiptV1::default();
            let mut error = [0 as c_char; ERROR_CAPACITY];
            let status = unsafe {
                apxinf_metal_w8_full_attention_stack6_v1_receipt(
                    self.0.as_ptr(),
                    &mut raw,
                    error.as_mut_ptr(),
                    error.len(),
                )
            };
            if status != 0 {
                return Err(bridge_error("read Metal W8 full-attention receipt", &error));
            }
            convert_and_validate_receipt(raw, max_context)
        }

        pub(super) fn snapshot_cache_row(
            &self,
            layer_slot: usize,
            position: u32,
        ) -> Result<(Vec<f32>, Vec<f32>), MetalW8Error> {
            let mut keys = vec![0.0f32; QWEN35_FULL_ATTENTION_KV_WIDTH_V1];
            let mut values = vec![0.0f32; QWEN35_FULL_ATTENTION_KV_WIDTH_V1];
            let mut error = [0 as c_char; ERROR_CAPACITY];
            let status = unsafe {
                apxinf_metal_w8_full_attention_stack6_v1_snapshot_cache_row(
                    self.0.as_ptr(),
                    layer_slot as u32,
                    position,
                    keys.as_mut_ptr(),
                    keys.len() as u32,
                    values.as_mut_ptr(),
                    values.len() as u32,
                    error.as_mut_ptr(),
                    error.len(),
                )
            };
            if status != 0 {
                return Err(bridge_error(
                    "snapshot Metal W8 full-attention cache row",
                    &error,
                ));
            }
            Ok((keys, values))
        }
    }

    impl Drop for Handle {
        fn drop(&mut self) {
            unsafe { apxinf_metal_w8_full_attention_stack6_v1_destroy(self.0.as_ptr()) };
        }
    }

    fn convert_and_validate_receipt(
        raw: RawRuntimeReceiptV1,
        max_context: usize,
    ) -> Result<FullAttentionStack6RuntimeReceiptV1, MetalW8Error> {
        let observed = u32::from(raw.successful_decodes != 0);
        let last_position_valid = if raw.successful_decodes == 0 {
            raw.last_layer_slot == u32::MAX
                && raw.last_start_pos == u32::MAX
                && raw.last_kv_length == 0
        } else {
            raw.last_layer_slot < QWEN35_FULL_ATTENTION_LAYER_SLOTS_V1 as u32
                && raw.last_start_pos < max_context as u32
                && raw.last_kv_length == raw.last_start_pos + 1
        };
        if raw.layer_slots != QWEN35_FULL_ATTENTION_LAYER_SLOTS_V1 as u32
            || raw.hidden_size != QWEN35_FULL_ATTENTION_HIDDEN_SIZE_V1 as u32
            || raw.query_heads != QWEN35_FULL_ATTENTION_QUERY_HEADS_V1 as u32
            || raw.kv_heads != QWEN35_FULL_ATTENTION_KV_HEADS_V1 as u32
            || raw.head_dim != QWEN35_FULL_ATTENTION_HEAD_DIM_V1 as u32
            || raw.rotary_dim != QWEN35_FULL_ATTENTION_ROTARY_DIM_V1 as u32
            || raw.max_context != max_context as u32
            || raw.group_size != W8_GROUP_SIZE as u32
            || raw.command_buffers_per_decode != 1
            || raw.compute_encoders_per_decode != 1
            || raw.kernel_dispatches_per_decode != 5
            || raw.explicit_buffer_barriers_per_decode != 4
            || raw.commits_per_decode != 1
            || raw.waits_per_decode != 1
            || raw.fixed_shape_validated != 1
            || raw.reserved != 0
            || !last_position_valid
            || raw.last_observed_command_buffers != observed
            || raw.last_observed_compute_encoders != observed
            || raw.last_observed_kernel_dispatches != observed * 5
            || raw.last_observed_explicit_buffer_barriers != observed * 4
            || raw.last_observed_commits != observed
            || raw.last_observed_waits != observed
        {
            return Err(MetalW8Error::new(
                "invalid live Metal W8 full-attention stack6 receipt",
            ));
        }
        Ok(FullAttentionStack6RuntimeReceiptV1 {
            layer_slots: raw.layer_slots,
            hidden_size: raw.hidden_size,
            query_heads: raw.query_heads,
            kv_heads: raw.kv_heads,
            head_dim: raw.head_dim,
            rotary_dim: raw.rotary_dim,
            max_context: raw.max_context,
            group_size: raw.group_size,
            command_buffers_per_decode: raw.command_buffers_per_decode,
            compute_encoders_per_decode: raw.compute_encoders_per_decode,
            kernel_dispatches_per_decode: raw.kernel_dispatches_per_decode,
            explicit_buffer_barriers_per_decode: raw.explicit_buffer_barriers_per_decode,
            commits_per_decode: raw.commits_per_decode,
            waits_per_decode: raw.waits_per_decode,
            fixed_shape_validated: true,
            successful_decodes: raw.successful_decodes,
            last_layer_slot: raw.last_layer_slot,
            last_start_pos: raw.last_start_pos,
            last_kv_length: raw.last_kv_length,
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
    pub(super) fn raw_receipt_size() -> usize {
        std::mem::size_of::<RawRuntimeReceiptV1>()
    }
}

#[cfg(not(target_os = "macos"))]
mod platform {
    use super::*;

    pub(super) struct Handle;

    impl Handle {
        pub(super) fn new(
            _weights: &PackedW8FullAttentionStack6V1,
            _max_context: usize,
        ) -> Result<Self, MetalW8Error> {
            Err(MetalW8Error::new(
                "Metal W8 full-attention stack6 requires macOS",
            ))
        }

        pub(super) fn seed_cache(
            &mut self,
            _layer_slot: usize,
            _start_pos: u32,
            _keys: &[f32],
            _values: &[f32],
        ) -> Result<(), MetalW8Error> {
            Err(MetalW8Error::new(
                "Metal W8 full-attention stack6 requires macOS",
            ))
        }

        pub(super) fn decode(
            &mut self,
            _layer_slot: usize,
            _input: &[f32],
            _start_pos: u32,
            _output: &mut [f32],
        ) -> Result<(), MetalW8Error> {
            Err(MetalW8Error::new(
                "Metal W8 full-attention stack6 requires macOS",
            ))
        }

        pub(super) fn runtime_receipt(
            &self,
            _max_context: usize,
        ) -> Result<FullAttentionStack6RuntimeReceiptV1, MetalW8Error> {
            Err(MetalW8Error::new(
                "Metal W8 full-attention stack6 requires macOS",
            ))
        }

        pub(super) fn snapshot_cache_row(
            &self,
            _layer_slot: usize,
            _position: u32,
        ) -> Result<(Vec<f32>, Vec<f32>), MetalW8Error> {
            Err(MetalW8Error::new(
                "Metal W8 full-attention stack6 requires macOS",
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rms_norm_offset_is_zero_centered_and_row_local() {
        let mut values = vec![3.0, 4.0, 0.0, -2.0];
        rms_norm_offset_in_place(&mut values, 2, &[0.0, 1.0], 0.0).unwrap();
        assert!((values[0] - 3.0 / 3.535_534).abs() < 1.0e-6);
        assert!((values[1] - 2.0 * 4.0 / 3.535_534).abs() < 1.0e-6);
        assert_eq!(values[2], 0.0);
        assert!((values[3] + 2.0 * std::f32::consts::SQRT_2).abs() < 1.0e-6);
    }

    #[test]
    fn partial_rope_uses_rotate_half_and_preserves_tail() {
        let mut values = vec![1.0, 2.0, 7.0, 8.0, 3.0, 4.0, 9.0, 10.0];
        apply_partial_rope(&mut values, 1, 8, 4, 10_000.0, 1).unwrap();
        let (sin0, cos0) = 1.0f32.sin_cos();
        let (sin1, cos1) = 0.01f32.sin_cos();
        assert!((values[0] - (1.0 * cos0 - 7.0 * sin0)).abs() < 1.0e-6);
        assert!((values[2] - (1.0 * sin0 + 7.0 * cos0)).abs() < 1.0e-6);
        assert!((values[1] - (2.0 * cos1 - 8.0 * sin1)).abs() < 1.0e-6);
        assert!((values[3] - (2.0 * sin1 + 8.0 * cos1)).abs() < 1.0e-6);
        assert_eq!(&values[4..], &[3.0, 4.0, 9.0, 10.0]);
    }

    #[test]
    fn sdpa_maps_grouped_query_heads_and_appends_current_token() {
        // Q heads 0/1 share KV head 0; Q heads 2/3 share KV head 1.
        let query = [1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0];
        let prefix_keys = [1.0, 0.0, 0.0, 1.0];
        let prefix_values = [2.0, 3.0, 5.0, 7.0];
        let appended_key = [1.0, 0.0, 0.0, 1.0];
        let appended_value = [11.0, 13.0, 17.0, 19.0];
        let output = sdpa_decode_seq_major(
            &query,
            &prefix_keys,
            &prefix_values,
            &appended_key,
            &appended_value,
            1,
            4,
            2,
            2,
        )
        .unwrap();
        assert_eq!(output, vec![6.5, 8.0, 6.5, 8.0, 11.0, 13.0, 11.0, 13.0]);
    }

    #[test]
    fn explicit_cursor_can_overwrite_a_later_host_state() {
        let mut state = FullAttentionKvStateV1::new(4).unwrap();
        let two_tokens = vec![1.0; 2 * QWEN35_FULL_ATTENTION_KV_WIDTH_V1];
        state.seed_cache(3, 2, &two_tokens, &two_tokens).unwrap();
        let replacement_key = vec![2.0; QWEN35_FULL_ATTENTION_KV_WIDTH_V1];
        let replacement_value = vec![3.0; QWEN35_FULL_ATTENTION_KV_WIDTH_V1];
        state
            .commit_token(3, 1, &replacement_key, &replacement_value)
            .unwrap();
        assert_eq!(state.cached_tokens(3).unwrap(), 2);
        let (keys, values) = state.layer_cache(3).unwrap();
        assert!(keys[..QWEN35_FULL_ATTENTION_KV_WIDTH_V1]
            .iter()
            .all(|&value| value == 1.0));
        assert!(keys[QWEN35_FULL_ATTENTION_KV_WIDTH_V1..]
            .iter()
            .all(|&value| value == 2.0));
        assert!(values[QWEN35_FULL_ATTENTION_KV_WIDTH_V1..]
            .iter()
            .all(|&value| value == 3.0));
    }

    #[test]
    fn empty_seed_is_valid_but_decode_bounds_are_strict() {
        let mut state = FullAttentionKvStateV1::new(2).unwrap();
        state.seed_cache(0, 0, &[], &[]).unwrap();
        assert_eq!(state.cached_tokens(0).unwrap(), 0);
        assert!(state.validate_decode_position(0, 0).is_ok());
        assert!(state.validate_decode_position(0, 1).is_err());
        assert!(state.validate_decode_position(0, 2).is_err());
        assert!(state.seed_cache(6, 0, &[], &[]).is_err());
    }

    #[test]
    fn fixed_shape_pack_rejects_before_large_allocation() {
        let empty = FullAttentionLayerF32WeightsV1 {
            input_rms_weight: &[],
            query_rows: &[],
            gate_rows: &[],
            key_rows: &[],
            value_rows: &[],
            query_norm_weight: &[],
            key_norm_weight: &[],
            output_rows: &[],
        };
        let error = PackedW8FullAttentionStack6V1::pack_f32(&[empty; 6]).unwrap_err();
        assert!(error
            .to_string()
            .contains("input RMS weight has 0 elements"));
        assert!(PackedW8FullAttentionStack6V1::pack_f32(&[empty; 5]).is_err());
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn runtime_receipt_abi_is_exact() {
        assert_eq!(platform::raw_receipt_size(), 112);
    }
}
