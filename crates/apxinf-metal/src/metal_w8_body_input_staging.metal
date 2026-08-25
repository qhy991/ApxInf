// Diagnostic body-only candidates. This file is appended only to the complete
// linear-layer library, so the tail library and every legacy entry point keep
// their original source and ABI.

kernel void gdn_w8_input_projection_tg_shared(
    device const char4 *weights [[buffer(0)]],
    device const float *scales [[buffer(1)]],
    device const float4 *input [[buffer(2)]],
    device float *projected [[buffer(3)]],
    constant GdnParams& params [[buffer(4)]],
    threadgroup float4 *input_tile [[threadgroup(0)]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint group [[threadgroup_position_in_grid]]) {
    constexpr uint threads_per_threadgroup = 256;
    constexpr uint rows_per_threadgroup = 8;
    constexpr uint float4_per_group = 16;
    const uint columns4 = params.hidden_size >> 2;
    for (uint index = tid; index < columns4;
         index += threads_per_threadgroup) {
        input_tile[index] = input[index];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint row = group * rows_per_threadgroup + simdgroup;
    if (row >= params.input_rows) {
        return;
    }
    const uint weight_base = row * columns4;
    const uint scale_base = row * params.input_groups_per_row;
    float sum = 0.0f;
    for (uint index = lane; index < columns4; index += 32) {
        sum += dot(float4(weights[weight_base + index]), input_tile[index]) *
               scales[scale_base + index / float4_per_group];
    }
    sum = simd_sum(sum);
    if (lane == 0) {
        projected[row] = sum;
    }
}

kernel void gdn_w8_output_projection_g32_tg_shared(
    device const char4 *weights [[buffer(0)]],
    device const float *scales [[buffer(1)]],
    device const float4 *input [[buffer(2)]],
    device float *output [[buffer(3)]],
    constant GdnParams& params [[buffer(4)]],
    threadgroup float4 *input_tile [[threadgroup(0)]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint group [[threadgroup_position_in_grid]]) {
    constexpr uint threads_per_threadgroup = 256;
    constexpr uint rows_per_threadgroup = 8;
    constexpr uint float4_per_group = 8;
    const uint columns4 = params.value_width >> 2;
    for (uint index = tid; index < columns4;
         index += threads_per_threadgroup) {
        input_tile[index] = input[index];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint row = group * rows_per_threadgroup + simdgroup;
    if (row >= params.hidden_size) {
        return;
    }
    const uint weight_base = row * columns4;
    const uint scale_base = row * params.output_groups_per_row;
    float sum = 0.0f;
    for (uint index = lane; index < columns4; index += 32) {
        sum += dot(float4(weights[weight_base + index]), input_tile[index]) *
               scales[scale_base + index / float4_per_group];
    }
    sum = simd_sum(sum);
    if (lane == 0) {
        output[row] = sum;
    }
}

kernel void w8_mlp_gate_up_tg_shared(
    device const char4 *weights [[buffer(0)]],
    device const float *scales [[buffer(1)]],
    device const float4 *input [[buffer(2)]],
    device float *gate_up [[buffer(3)]],
    constant MlpParams& params [[buffer(4)]],
    threadgroup float4 *input_tile [[threadgroup(0)]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint group [[threadgroup_position_in_grid]]) {
    constexpr uint threads_per_threadgroup = 256;
    constexpr uint rows_per_threadgroup = 8;
    constexpr uint float4_per_group = 16;
    const uint columns4 = params.hidden_size >> 2;
    for (uint index = tid; index < columns4;
         index += threads_per_threadgroup) {
        input_tile[index] = input[index];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint row = group * rows_per_threadgroup + simdgroup;
    const uint rows = params.intermediate_size * 2;
    if (row >= rows) {
        return;
    }
    const uint weight_base = row * columns4;
    const uint scale_base = row * params.gate_up_groups_per_row;
    float sum = 0.0f;
    for (uint index = lane; index < columns4; index += 32) {
        const char4 quantized = weights[weight_base + index];
        const float scale = scales[scale_base + index / float4_per_group];
        sum += dot(float4(quantized), input_tile[index]) * scale;
    }
    sum = simd_sum(sum);
    if (lane == 0) {
        gate_up[row] = sum;
    }
}

kernel void w8_mlp_down_tg_shared(
    device const char4 *weights [[buffer(0)]],
    device const float *scales [[buffer(1)]],
    device const float4 *activated [[buffer(2)]],
    device float *output [[buffer(3)]],
    constant MlpParams& params [[buffer(4)]],
    threadgroup float4 *input_tile [[threadgroup(0)]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint group [[threadgroup_position_in_grid]]) {
    constexpr uint threads_per_threadgroup = 256;
    constexpr uint rows_per_threadgroup = 8;
    constexpr uint float4_per_group = 16;
    const uint columns4 = params.intermediate_size >> 2;
    for (uint index = tid; index < columns4;
         index += threads_per_threadgroup) {
        input_tile[index] = activated[index];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint row = group * rows_per_threadgroup + simdgroup;
    if (row >= params.hidden_size) {
        return;
    }
    const uint weight_base = row * columns4;
    const uint scale_base = row * params.down_groups_per_row;
    float sum = 0.0f;
    for (uint index = lane; index < columns4; index += 32) {
        const char4 quantized = weights[weight_base + index];
        const float scale = scales[scale_base + index / float4_per_group];
        sum += dot(float4(quantized), input_tile[index]) * scale;
    }
    sum = simd_sum(sum);
    if (lane == 0) {
        output[row] = sum;
    }
}
