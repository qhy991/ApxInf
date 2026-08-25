#include <metal_stdlib>
using namespace metal;

// Fixed Qwen3.5-0.8B full-attention decode shape. Six non-contiguous model
// layers share one resident weight/cache handle, but each decode call executes
// exactly one layer transaction (one command buffer and one wait).
constant uint kHiddenSizeV1 = 1024;
constant uint kQueryHeadsV1 = 8;
constant uint kKvHeadsV1 = 2;
constant uint kHeadDimV1 = 256;
constant uint kRotaryDimV1 = 64;
constant uint kQueryWidthV1 = kQueryHeadsV1 * kHeadDimV1;
constant uint kKvWidthV1 = kKvHeadsV1 * kHeadDimV1;
constant uint kQOffsetV1 = 0;
constant uint kGateOffsetV1 = kQueryWidthV1;
constant uint kKOffsetV1 = 2 * kQueryWidthV1;
constant uint kVOffsetV1 = 2 * kQueryWidthV1 + kKvWidthV1;
constant uint kQGKVRowsV1 = 2 * kQueryWidthV1 + 2 * kKvWidthV1;
constant uint kRowsPerThreadgroupV1 = 8;
constant uint kFloat4PerGroupV1 = 16;

struct FullAttentionParamsV1 {
    uint max_context;
    uint position;
    uint layer_slot;
    uint reserved;
    float rms_norm_eps;
    float rope_theta;
};

// Zero-centred input RMSNorm. One SIMD-group processes the complete H=1024
// row so the normalized row stays resident for every following projection.
kernel void full_attention_input_rms_v1(
    device const float *input [[buffer(0)]],
    device const float *input_norm_weight [[buffer(1)]],
    device float *normalized [[buffer(2)]],
    constant FullAttentionParamsV1& params [[buffer(3)]],
    uint lane [[thread_index_in_simdgroup]]) {
    float sum_sq = 0.0f;
    for (uint column = lane; column < kHiddenSizeV1; column += 32) {
        const float value = input[column];
        sum_sq += value * value;
    }
    sum_sq = simd_sum(sum_sq);
    const float inv_rms = rsqrt(sum_sq / float(kHiddenSizeV1) + params.rms_norm_eps);
    const uint weight_base = params.layer_slot * kHiddenSizeV1;
    for (uint column = lane; column < kHiddenSizeV1; column += 32) {
        normalized[column] =
            input[column] * inv_rms * (input_norm_weight[weight_base + column] + 1.0f);
    }
}

// Q/Gate/K/V are concatenated by row. One SIMD-group evaluates one G64 row;
// eight rows share a threadgroup, matching the established W8 matvec topology.
kernel void full_attention_qgkv_v1(
    device const char4 *weights [[buffer(0)]],
    device const float *scales [[buffer(1)]],
    device const float4 *normalized [[buffer(2)]],
    device float *projected [[buffer(3)]],
    constant FullAttentionParamsV1& params [[buffer(4)]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint group [[threadgroup_position_in_grid]]) {
    const uint row = group * kRowsPerThreadgroupV1 + simdgroup;
    if (row >= kQGKVRowsV1) {
        return;
    }
    const uint columns4 = kHiddenSizeV1 >> 2;
    const uint groups_per_row = kHiddenSizeV1 >> 6;
    const uint global_row = params.layer_slot * kQGKVRowsV1 + row;
    const uint weight_base = global_row * columns4;
    const uint scale_base = global_row * groups_per_row;
    float sum = 0.0f;
    for (uint index = lane; index < columns4; index += 32) {
        const char4 quantized = weights[weight_base + index];
        const float scale = scales[scale_base + index / kFloat4PerGroupV1];
        sum += dot(float4(quantized), normalized[index]) * scale;
    }
    sum = simd_sum(sum);
    if (lane == 0) {
        projected[row] = sum;
    }
}

// Ten SIMD-groups independently normalize/rotate eight Q and two K heads.
// K/V are written into the explicit uncommitted position. The primitive owns
// no cursor: the model's global hybrid-state cursor remains authoritative, so
// a later failure leaves this row logically unreachable and retry overwrites it.
kernel void full_attention_prepare_qkv_v1(
    device const float *projected [[buffer(0)]],
    device const float *query_norm_weight [[buffer(1)]],
    device const float *key_norm_weight [[buffer(2)]],
    device float *query [[buffer(3)]],
    device float *key_cache [[buffer(4)]],
    device float *value_cache [[buffer(5)]],
    constant FullAttentionParamsV1& params [[buffer(6)]],
    uint lane [[thread_index_in_simdgroup]],
    uint head_group [[threadgroup_position_in_grid]]) {
    const bool is_query = head_group < kQueryHeadsV1;
    const uint head = is_query ? head_group : head_group - kQueryHeadsV1;
    const uint source_base =
        (is_query ? kQOffsetV1 : kKOffsetV1) + head * kHeadDimV1;
    float sum_sq = 0.0f;
    for (uint dim = lane; dim < kHeadDimV1; dim += 32) {
        const float value = projected[source_base + dim];
        sum_sq += value * value;
    }
    sum_sq = simd_sum(sum_sq);
    const float inv_rms = rsqrt(sum_sq / float(kHeadDimV1) + params.rms_norm_eps);
    const uint norm_base = params.layer_slot * kHeadDimV1;
    device const float *norm_weight = is_query ? query_norm_weight : key_norm_weight;

    const ulong cache_base =
        ((ulong(params.layer_slot) * ulong(kKvHeadsV1) + ulong(head)) *
             ulong(params.max_context) +
         ulong(params.position)) *
        ulong(kHeadDimV1);
    for (uint pair = lane; pair < kRotaryDimV1 / 2; pair += 32) {
        const uint second = pair + kRotaryDimV1 / 2;
        const float first_value = projected[source_base + pair] * inv_rms *
            (norm_weight[norm_base + pair] + 1.0f);
        const float second_value = projected[source_base + second] * inv_rms *
            (norm_weight[norm_base + second] + 1.0f);
        const float frequency =
            pow(params.rope_theta, -2.0f * float(pair) / float(kRotaryDimV1));
        const float angle = float(params.position) * frequency;
        const float cosine = cos(angle);
        const float sine = sin(angle);
        const float rotated_first = first_value * cosine - second_value * sine;
        const float rotated_second = first_value * sine + second_value * cosine;
        if (is_query) {
            const uint output_base = head * kHeadDimV1;
            query[output_base + pair] = rotated_first;
            query[output_base + second] = rotated_second;
        } else {
            key_cache[cache_base + pair] = rotated_first;
            key_cache[cache_base + second] = rotated_second;
        }
    }
    for (uint dim = kRotaryDimV1 + lane; dim < kHeadDimV1; dim += 32) {
        const float value = projected[source_base + dim] * inv_rms *
            (norm_weight[norm_base + dim] + 1.0f);
        if (is_query) {
            query[head * kHeadDimV1 + dim] = value;
        } else {
            key_cache[cache_base + dim] = value;
        }
    }
    if (!is_query) {
        const uint value_base = kVOffsetV1 + head * kHeadDimV1;
        for (uint dim = lane; dim < kHeadDimV1; dim += 32) {
            value_cache[cache_base + dim] = projected[value_base + dim];
        }
    }
}

// Online two-pass decode attention avoids a max-context-sized threadgroup
// score array. One SIMD-group owns one query head; each lane holds eight output
// dimensions. This supports dynamic context capacity without hidden scratch.
kernel void full_attention_sdpa_gate_v1(
    device const float *query [[buffer(0)]],
    device const float *projected [[buffer(1)]],
    device const float *key_cache [[buffer(2)]],
    device const float *value_cache [[buffer(3)]],
    device float *gated_attention [[buffer(4)]],
    constant FullAttentionParamsV1& params [[buffer(5)]],
    uint lane [[thread_index_in_simdgroup]],
    uint query_head [[threadgroup_position_in_grid]]) {
    const uint kv_head = query_head * kKvHeadsV1 / kQueryHeadsV1;
    const uint query_base = query_head * kHeadDimV1;
    const ulong cache_head_base =
        (ulong(params.layer_slot) * ulong(kKvHeadsV1) + ulong(kv_head)) *
        ulong(params.max_context) * ulong(kHeadDimV1);
    const uint valid_length = params.position + 1;
    constexpr float attention_scale = 0.0625f;  // 1 / sqrt(256)

    float maximum = -INFINITY;
    for (uint token = 0; token < valid_length; ++token) {
        const ulong cache_base = cache_head_base + ulong(token) * ulong(kHeadDimV1);
        float partial = 0.0f;
        for (uint dim = lane; dim < kHeadDimV1; dim += 32) {
            partial += query[query_base + dim] * key_cache[cache_base + dim];
        }
        const float score = simd_sum(partial) * attention_scale;
        maximum = max(maximum, score);
    }

    float denominator = 0.0f;
    float accumulators[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (uint token = 0; token < valid_length; ++token) {
        const ulong cache_base = cache_head_base + ulong(token) * ulong(kHeadDimV1);
        float partial = 0.0f;
        for (uint dim = lane; dim < kHeadDimV1; dim += 32) {
            partial += query[query_base + dim] * key_cache[cache_base + dim];
        }
        const float probability_numerator =
            exp(simd_sum(partial) * attention_scale - maximum);
        denominator += probability_numerator;
        for (uint item = 0; item < 8; ++item) {
            const uint dim = lane + item * 32;
            accumulators[item] += probability_numerator * value_cache[cache_base + dim];
        }
    }

    const uint gate_base = kGateOffsetV1 + query_base;
    for (uint item = 0; item < 8; ++item) {
        const uint dim = lane + item * 32;
        const float gate = projected[gate_base + dim];
        const float sigmoid = gate >= 0.0f
            ? 1.0f / (1.0f + exp(-gate))
            : exp(gate) / (1.0f + exp(gate));
        gated_attention[query_base + dim] = accumulators[item] / denominator * sigmoid;
    }
}

// Final W8 O projection and residual add. Only the residual-width row crosses
// back to the host after command completion.
kernel void full_attention_output_residual_v1(
    device const char4 *weights [[buffer(0)]],
    device const float *scales [[buffer(1)]],
    device const float4 *gated_attention [[buffer(2)]],
    device const float *residual_input [[buffer(3)]],
    device float *residual_output [[buffer(4)]],
    constant FullAttentionParamsV1& params [[buffer(5)]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint group [[threadgroup_position_in_grid]]) {
    const uint row = group * kRowsPerThreadgroupV1 + simdgroup;
    if (row >= kHiddenSizeV1) {
        return;
    }
    const uint columns4 = kQueryWidthV1 >> 2;
    const uint groups_per_row = kQueryWidthV1 >> 6;
    const uint global_row = params.layer_slot * kHiddenSizeV1 + row;
    const uint weight_base = global_row * columns4;
    const uint scale_base = global_row * groups_per_row;
    float sum = 0.0f;
    for (uint index = lane; index < columns4; index += 32) {
        const char4 quantized = weights[weight_base + index];
        const float scale = scales[scale_base + index / kFloat4PerGroupV1];
        sum += dot(float4(quantized), gated_attention[index]) * scale;
    }
    sum = simd_sum(sum);
    if (lane == 0) {
        residual_output[row] = residual_input[row] + sum;
    }
}
