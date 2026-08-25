// Diagnostic-only scale-load variants for the complete linear-layer library.
// This fragment is appended after the legacy GDN, MLP, linear-layer, and G32
// sources. Standalone GDN/MLP libraries and every legacy function retain their
// original source bytes and ABI.

inline float w8_scale_broadcast_g64(
    device const float *scales,
    uint scale_base,
    uint chunk,
    uint columns4,
    uint lane) {
    const float owned0 = lane == 0
        ? scales[scale_base + chunk / 16]
        : 0.0f;
    const float owned16 = lane == 16 && chunk + 16 < columns4
        ? scales[scale_base + chunk / 16 + 1]
        : 0.0f;
    const float scale0 = simd_broadcast(owned0, ushort(0));
    const float scale16 = simd_broadcast(owned16, ushort(16));
    return lane < 16 ? scale0 : scale16;
}

inline float w8_scale_broadcast_g32(
    device const float *scales,
    uint scale_base,
    uint chunk,
    uint columns4,
    uint lane) {
    const uint group_base = chunk / 8;
    const float owned0 = lane == 0
        ? scales[scale_base + group_base]
        : 0.0f;
    const float owned8 = lane == 8 && chunk + 8 < columns4
        ? scales[scale_base + group_base + 1]
        : 0.0f;
    const float owned16 = lane == 16 && chunk + 16 < columns4
        ? scales[scale_base + group_base + 2]
        : 0.0f;
    const float owned24 = lane == 24 && chunk + 24 < columns4
        ? scales[scale_base + group_base + 3]
        : 0.0f;
    const float scale0 = simd_broadcast(owned0, ushort(0));
    const float scale8 = simd_broadcast(owned8, ushort(8));
    const float scale16 = simd_broadcast(owned16, ushort(16));
    const float scale24 = simd_broadcast(owned24, ushort(24));
    return lane < 8 ? scale0
         : lane < 16 ? scale8
         : lane < 24 ? scale16
                     : scale24;
}

kernel void gdn_w8_input_projection_scale_broadcast(
    device const char4 *weights [[buffer(0)]],
    device const float *scales [[buffer(1)]],
    device const float4 *input [[buffer(2)]],
    device float *projected [[buffer(3)]],
    constant GdnParams& params [[buffer(4)]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint group [[threadgroup_position_in_grid]]) {
    constexpr uint rows_per_threadgroup = 8;
    const uint row = group * rows_per_threadgroup + simdgroup;
    if (row >= params.input_rows) {
        return;
    }
    const uint columns4 = params.hidden_size >> 2;
    const uint weight_base = row * columns4;
    const uint scale_base = row * params.input_groups_per_row;
    float sum = 0.0f;
    for (uint index = lane, chunk = 0; chunk < columns4;
         index += 32, chunk += 32) {
        const float scale =
            w8_scale_broadcast_g64(scales, scale_base, chunk, columns4, lane);
        if (index < columns4) {
            sum += dot(float4(weights[weight_base + index]), input[index]) * scale;
        }
    }
    sum = simd_sum(sum);
    if (lane == 0) {
        projected[row] = sum;
    }
}

kernel void gdn_w8_output_projection_g32_scale_broadcast(
    device const char4 *weights [[buffer(0)]],
    device const float *scales [[buffer(1)]],
    device const float4 *input [[buffer(2)]],
    device float *output [[buffer(3)]],
    constant GdnParams& params [[buffer(4)]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint group [[threadgroup_position_in_grid]]) {
    constexpr uint rows_per_threadgroup = 8;
    const uint row = group * rows_per_threadgroup + simdgroup;
    if (row >= params.hidden_size) {
        return;
    }
    const uint columns4 = params.value_width >> 2;
    const uint weight_base = row * columns4;
    const uint scale_base = row * params.output_groups_per_row;
    float sum = 0.0f;
    for (uint index = lane, chunk = 0; chunk < columns4;
         index += 32, chunk += 32) {
        const float scale =
            w8_scale_broadcast_g32(scales, scale_base, chunk, columns4, lane);
        if (index < columns4) {
            sum += dot(float4(weights[weight_base + index]), input[index]) * scale;
        }
    }
    sum = simd_sum(sum);
    if (lane == 0) {
        output[row] = sum;
    }
}

kernel void w8_mlp_gate_up_scale_broadcast(
    device const char4 *weights [[buffer(0)]],
    device const float *scales [[buffer(1)]],
    device const float4 *input [[buffer(2)]],
    device float *gate_up [[buffer(3)]],
    constant MlpParams& params [[buffer(4)]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint group [[threadgroup_position_in_grid]]) {
    constexpr uint rows_per_threadgroup = 8;
    const uint row = group * rows_per_threadgroup + simdgroup;
    const uint rows = params.intermediate_size * 2;
    if (row >= rows) {
        return;
    }
    const uint columns4 = params.hidden_size >> 2;
    const uint weight_base = row * columns4;
    const uint scale_base = row * params.gate_up_groups_per_row;
    float sum = 0.0f;
    for (uint index = lane, chunk = 0; chunk < columns4;
         index += 32, chunk += 32) {
        const float scale =
            w8_scale_broadcast_g64(scales, scale_base, chunk, columns4, lane);
        if (index < columns4) {
            const char4 quantized = weights[weight_base + index];
            sum += dot(float4(quantized), input[index]) * scale;
        }
    }
    sum = simd_sum(sum);
    if (lane == 0) {
        gate_up[row] = sum;
    }
}

kernel void w8_mlp_down_scale_broadcast(
    device const char4 *weights [[buffer(0)]],
    device const float *scales [[buffer(1)]],
    device const float4 *activated [[buffer(2)]],
    device float *output [[buffer(3)]],
    constant MlpParams& params [[buffer(4)]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint group [[threadgroup_position_in_grid]]) {
    constexpr uint rows_per_threadgroup = 8;
    const uint row = group * rows_per_threadgroup + simdgroup;
    if (row >= params.hidden_size) {
        return;
    }
    const uint columns4 = params.intermediate_size >> 2;
    const uint weight_base = row * columns4;
    const uint scale_base = row * params.down_groups_per_row;
    float sum = 0.0f;
    for (uint index = lane, chunk = 0; chunk < columns4;
         index += 32, chunk += 32) {
        const float scale =
            w8_scale_broadcast_g64(scales, scale_base, chunk, columns4, lane);
        if (index < columns4) {
            const char4 quantized = weights[weight_base + index];
            sum += dot(float4(quantized), activated[index]) * scale;
        }
    }
    sum = simd_sum(sum);
    if (lane == 0) {
        output[row] = sum;
    }
}
