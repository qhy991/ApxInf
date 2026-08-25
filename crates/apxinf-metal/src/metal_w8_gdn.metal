#include <metal_stdlib>
using namespace metal;

struct GdnParams {
    uint hidden_size;
    uint key_heads;
    uint value_heads;
    uint key_dim;
    uint value_dim;
    uint conv_kernel_size;
    uint key_width;
    uint value_width;
    uint qkv_width;
    uint input_rows;
    uint input_groups_per_row;
    uint output_groups_per_row;
    float rms_norm_eps;
};

kernel void gdn_w8_input_projection(
    device const char4 *weights [[buffer(0)]],
    device const float *scales [[buffer(1)]],
    device const float4 *input [[buffer(2)]],
    device float *projected [[buffer(3)]],
    constant GdnParams& params [[buffer(4)]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint group [[threadgroup_position_in_grid]]) {
    constexpr uint rows_per_threadgroup = 8;
    constexpr uint float4_per_group = 16;
    const uint row = group * rows_per_threadgroup + simdgroup;
    if (row >= params.input_rows) {
        return;
    }
    const uint columns4 = params.hidden_size >> 2;
    const uint weight_base = row * columns4;
    const uint scale_base = row * params.input_groups_per_row;
    float sum = 0.0f;
    for (uint index = lane; index < columns4; index += 32) {
        sum += dot(float4(weights[weight_base + index]), input[index]) *
               scales[scale_base + index / float4_per_group];
    }
    sum = simd_sum(sum);
    if (lane == 0) {
        projected[row] = sum;
    }
}

kernel void gdn_depthwise_preprocess(
    device const float *projected [[buffer(0)]],
    device const float *conv_weight [[buffer(1)]],
    device const float *query_state [[buffer(2)]],
    device const float *key_state [[buffer(3)]],
    device const float *value_state [[buffer(4)]],
    device float *next_query_state [[buffer(5)]],
    device float *next_key_state [[buffer(6)]],
    device float *next_value_state [[buffer(7)]],
    device float *processed [[buffer(8)]],
    constant GdnParams& params [[buffer(9)]],
    uint channel [[thread_position_in_grid]]) {
    if (channel >= params.qkv_width) {
        return;
    }
    device const float *state;
    device float *next_state;
    uint local_channel;
    uint channels;
    if (channel < params.key_width) {
        state = query_state;
        next_state = next_query_state;
        local_channel = channel;
        channels = params.key_width;
    } else if (channel < 2 * params.key_width) {
        state = key_state;
        next_state = next_key_state;
        local_channel = channel - params.key_width;
        channels = params.key_width;
    } else {
        state = value_state;
        next_state = next_value_state;
        local_channel = channel - 2 * params.key_width;
        channels = params.value_width;
    }

    float sum = 0.0f;
    for (uint tap = 0; tap < params.conv_kernel_size; ++tap) {
        const float sample = tap + 1 < params.conv_kernel_size
            ? state[(tap + 1) * channels + local_channel]
            : projected[channel];
        sum += sample * conv_weight[channel * params.conv_kernel_size + tap];
    }
    processed[channel] = sum / (1.0f + exp(-sum));
    for (uint time = 0; time < params.conv_kernel_size; ++time) {
        next_state[time * channels + local_channel] = time + 1 < params.conv_kernel_size
            ? state[(time + 1) * channels + local_channel]
            : projected[channel];
    }
}

kernel void gdn_normalize_qk(
    device float *processed [[buffer(0)]],
    constant GdnParams& params [[buffer(1)]],
    uint head_index [[thread_position_in_grid]]) {
    if (head_index >= 2 * params.key_heads) {
        return;
    }
    const bool query = head_index < params.key_heads;
    const uint head = query ? head_index : head_index - params.key_heads;
    const uint base = (query ? 0 : params.key_width) + head * params.key_dim;
    float sum_square = 0.0f;
    for (uint index = 0; index < params.key_dim; ++index) {
        const float value = processed[base + index];
        sum_square += value * value;
    }
    float scale = rsqrt(sum_square + 1.0e-6f);
    if (query) {
        scale *= rsqrt(float(params.key_dim));
    }
    for (uint index = 0; index < params.key_dim; ++index) {
        processed[base + index] *= scale;
    }
}

kernel void gdn_recurrent_update(
    device const float *processed [[buffer(0)]],
    device const float *projected [[buffer(1)]],
    device const float *a_log [[buffer(2)]],
    device const float *dt_bias [[buffer(3)]],
    device const float *state [[buffer(4)]],
    device float *next_state [[buffer(5)]],
    device float *core [[buffer(6)]],
    constant GdnParams& params [[buffer(7)]],
    uint value_index [[thread_index_in_threadgroup]],
    uint3 group_position [[threadgroup_position_in_grid]],
    uint3 thread_count [[threads_per_threadgroup]]) {
    const uint value_head = group_position.x;
    if (value_head >= params.value_heads) {
        return;
    }
    const uint repeat_factor = params.value_heads / params.key_heads;
    const uint key_head = value_head / repeat_factor;
    const uint query_base = key_head * params.key_dim;
    const uint value_base = 2 * params.key_width + value_head * params.value_dim;
    const uint a_base = params.qkv_width + params.value_width;
    const uint b_base = a_base + params.value_heads;
    const float b = projected[b_base + value_head];
    const float beta = b >= 0.0f ? 1.0f / (1.0f + exp(-b))
                                 : exp(b) / (1.0f + exp(b));
    const float gate = projected[a_base + value_head] + dt_bias[value_head];
    const float softplus = gate > 20.0f ? gate
                         : gate < -20.0f ? exp(gate)
                                         : log(1.0f + exp(gate));
    const float decay = exp(-exp(a_log[value_head]) * softplus);
    const uint state_base = value_head * params.key_dim * params.value_dim;

    for (uint v = value_index; v < params.value_dim; v += thread_count.x) {
        float delta = 0.0f;
        for (uint key = 0; key < params.key_dim; ++key) {
            const uint index = state_base + key * params.value_dim + v;
            delta += state[index] * decay * processed[params.key_width + query_base + key];
        }
        delta = (processed[value_base + v] - delta) * beta;
        float output = 0.0f;
        for (uint key = 0; key < params.key_dim; ++key) {
            const uint index = state_base + key * params.value_dim + v;
            const float updated = state[index] * decay +
                                  processed[params.key_width + query_base + key] * delta;
            next_state[index] = updated;
            output += updated * processed[query_base + key];
        }
        core[value_head * params.value_dim + v] = output;
    }
}

// Additive diagnostic profiles for the fixed Qwen3.5-0.8B recurrent shape.
// Existing production bridges intentionally continue to select only
// gdn_recurrent_update.  These profiles preserve the per-value key loop and
// state-update order while removing source-level work that is uniform across
// a value-head threadgroup.
kernel void gdn_recurrent_update_leader_broadcast_v1(
    device const float *processed [[buffer(0)]],
    device const float *projected [[buffer(1)]],
    device const float *a_log [[buffer(2)]],
    device const float *dt_bias [[buffer(3)]],
    device const float *state [[buffer(4)]],
    device float *next_state [[buffer(5)]],
    device float *core [[buffer(6)]],
    constant GdnParams& params [[buffer(7)]],
    uint value_index [[thread_index_in_threadgroup]],
    uint3 group_position [[threadgroup_position_in_grid]],
    uint3 thread_count [[threads_per_threadgroup]]) {
    threadgroup float shared_scalars[2];
    const uint value_head = group_position.x;
    if (value_head >= params.value_heads) {
        return;
    }
    const uint repeat_factor = params.value_heads / params.key_heads;
    const uint key_head = value_head / repeat_factor;
    const uint query_base = key_head * params.key_dim;
    const uint value_base = 2 * params.key_width + value_head * params.value_dim;
    const uint a_base = params.qkv_width + params.value_width;
    const uint b_base = a_base + params.value_heads;
    if (value_index == 0) {
        const float b = projected[b_base + value_head];
        const float beta = b >= 0.0f ? 1.0f / (1.0f + exp(-b))
                                     : exp(b) / (1.0f + exp(b));
        const float gate = projected[a_base + value_head] + dt_bias[value_head];
        const float softplus = gate > 20.0f ? gate
                             : gate < -20.0f ? exp(gate)
                                             : log(1.0f + exp(gate));
        shared_scalars[0] = beta;
        shared_scalars[1] = exp(-exp(a_log[value_head]) * softplus);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float beta = shared_scalars[0];
    const float decay = shared_scalars[1];
    const uint state_base = value_head * params.key_dim * params.value_dim;

    for (uint v = value_index; v < params.value_dim; v += thread_count.x) {
        float delta = 0.0f;
        for (uint key = 0; key < params.key_dim; ++key) {
            const uint index = state_base + key * params.value_dim + v;
            delta += state[index] * decay * processed[params.key_width + query_base + key];
        }
        delta = (processed[value_base + v] - delta) * beta;
        float output = 0.0f;
        for (uint key = 0; key < params.key_dim; ++key) {
            const uint index = state_base + key * params.value_dim + v;
            const float updated = state[index] * decay +
                                  processed[params.key_width + query_base + key] * delta;
            next_state[index] = updated;
            output += updated * processed[query_base + key];
        }
        core[value_head * params.value_dim + v] = output;
    }
}

kernel void gdn_recurrent_update_qk_staged_v1(
    device const float *processed [[buffer(0)]],
    device const float *projected [[buffer(1)]],
    device const float *a_log [[buffer(2)]],
    device const float *dt_bias [[buffer(3)]],
    device const float *state [[buffer(4)]],
    device float *next_state [[buffer(5)]],
    device float *core [[buffer(6)]],
    constant GdnParams& params [[buffer(7)]],
    uint value_index [[thread_index_in_threadgroup]],
    uint3 group_position [[threadgroup_position_in_grid]],
    uint3 thread_count [[threads_per_threadgroup]]) {
    threadgroup float shared_scalars[2];
    threadgroup float shared_query[128];
    threadgroup float shared_key[128];
    const uint value_head = group_position.x;
    if (value_head >= params.value_heads) {
        return;
    }
    const uint repeat_factor = params.value_heads / params.key_heads;
    const uint key_head = value_head / repeat_factor;
    const uint query_base = key_head * params.key_dim;
    const uint value_base = 2 * params.key_width + value_head * params.value_dim;
    const uint a_base = params.qkv_width + params.value_width;
    const uint b_base = a_base + params.value_heads;
    if (value_index == 0) {
        const float b = projected[b_base + value_head];
        const float beta = b >= 0.0f ? 1.0f / (1.0f + exp(-b))
                                     : exp(b) / (1.0f + exp(b));
        const float gate = projected[a_base + value_head] + dt_bias[value_head];
        const float softplus = gate > 20.0f ? gate
                             : gate < -20.0f ? exp(gate)
                                             : log(1.0f + exp(gate));
        shared_scalars[0] = beta;
        shared_scalars[1] = exp(-exp(a_log[value_head]) * softplus);
    }
    if (value_index < 128) {
        shared_query[value_index] = processed[query_base + value_index];
        shared_key[value_index] = processed[params.key_width + query_base + value_index];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float beta = shared_scalars[0];
    const float decay = shared_scalars[1];
    const uint state_base = value_head * params.key_dim * params.value_dim;

    for (uint v = value_index; v < params.value_dim; v += thread_count.x) {
        float delta = 0.0f;
        for (uint key = 0; key < params.key_dim; ++key) {
            const uint index = state_base + key * params.value_dim + v;
            delta += state[index] * decay * shared_key[key];
        }
        delta = (processed[value_base + v] - delta) * beta;
        float output = 0.0f;
        for (uint key = 0; key < params.key_dim; ++key) {
            const uint index = state_base + key * params.value_dim + v;
            const float updated = state[index] * decay + shared_key[key] * delta;
            next_state[index] = updated;
            output += updated * shared_query[key];
        }
        core[value_head * params.value_dim + v] = output;
    }
}

kernel void gdn_norm_gate(
    device const float *core [[buffer(0)]],
    device const float *projected [[buffer(1)]],
    device const float *norm_weight [[buffer(2)]],
    device float *gated [[buffer(3)]],
    constant GdnParams& params [[buffer(4)]],
    uint value_head [[thread_position_in_grid]]) {
    if (value_head >= params.value_heads) {
        return;
    }
    const uint base = value_head * params.value_dim;
    float mean_square = 0.0f;
    for (uint index = 0; index < params.value_dim; ++index) {
        const float value = core[base + index];
        mean_square += value * value;
    }
    const float inverse_rms = rsqrt(mean_square / float(params.value_dim) + params.rms_norm_eps);
    const uint z_base = params.qkv_width;
    for (uint index = 0; index < params.value_dim; ++index) {
        const float z = projected[z_base + base + index];
        const float silu_z = z / (1.0f + exp(-z));
        gated[base + index] = core[base + index] * inverse_rms *
                              norm_weight[index] * silu_z;
    }
}

kernel void gdn_w8_output_projection(
    device const char4 *weights [[buffer(0)]],
    device const float *scales [[buffer(1)]],
    device const float4 *input [[buffer(2)]],
    device float *output [[buffer(3)]],
    constant GdnParams& params [[buffer(4)]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint group [[threadgroup_position_in_grid]]) {
    constexpr uint rows_per_threadgroup = 8;
    constexpr uint float4_per_group = 16;
    const uint row = group * rows_per_threadgroup + simdgroup;
    if (row >= params.hidden_size) {
        return;
    }
    const uint columns4 = params.value_width >> 2;
    const uint weight_base = row * columns4;
    const uint scale_base = row * params.output_groups_per_row;
    float sum = 0.0f;
    for (uint index = lane; index < columns4; index += 32) {
        sum += dot(float4(weights[weight_base + index]), input[index]) *
               scales[scale_base + index / float4_per_group];
    }
    sum = simd_sum(sum);
    if (lane == 0) {
        output[row] = sum;
    }
}
