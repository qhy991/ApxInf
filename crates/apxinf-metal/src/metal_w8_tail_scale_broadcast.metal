// Diagnostic-only scale-load variants for the fused tail MLP + top-4 library.
// This fragment is appended after the legacy MLP, linear-layer, and head
// sources. Every legacy function retains its original source bytes and ABI.

inline float tail_w8_scale_broadcast_g64(
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
            tail_w8_scale_broadcast_g64(scales, scale_base, chunk, columns4, lane);
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
            tail_w8_scale_broadcast_g64(scales, scale_base, chunk, columns4, lane);
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

kernel void w8_rows_topk4_scale_broadcast(
    device const char4 *weights [[buffer(0)]],
    device const float *scales [[buffer(1)]],
    device const float4 *hidden [[buffer(2)]],
    device Candidate *partial [[buffer(3)]],
    constant KernelParams& params [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint group [[threadgroup_position_in_grid]]) {
    constexpr uint rows_per_threadgroup = 8;
    threadgroup float row_scores[rows_per_threadgroup];
    threadgroup uint row_tokens[rows_per_threadgroup];

    const uint row = group * rows_per_threadgroup + simdgroup;
    const uint columns4 = params.columns >> 2;
    float sum = 0.0f;
    if (row < params.rows) {
        const uint weight_base = row * columns4;
        const uint scale_base = row * params.groups_per_row;
        for (uint index = lane, chunk = 0; chunk < columns4;
             index += 32, chunk += 32) {
            const float scale = tail_w8_scale_broadcast_g64(
                scales, scale_base, chunk, columns4, lane);
            if (index < columns4) {
                const char4 quantized = weights[weight_base + index];
                sum += dot(float4(quantized), hidden[index]) * scale;
            }
        }
    }
    sum = simd_sum(sum);
    if (lane == 0) {
        row_scores[simdgroup] = row < params.rows && !isnan(sum) ? sum : -INFINITY;
        row_tokens[simdgroup] = row < params.rows ? row : UINT_MAX;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        float scores[top_k] = {-INFINITY, -INFINITY, -INFINITY, -INFINITY};
        uint tokens[top_k] = {UINT_MAX, UINT_MAX, UINT_MAX, UINT_MAX};
        for (uint index = 0; index < rows_per_threadgroup; ++index) {
            insert_candidate(scores, tokens, row_scores[index], row_tokens[index]);
        }
        const uint base = group * top_k;
        for (uint index = 0; index < top_k; ++index) {
            partial[base + index] = Candidate{scores[index], tokens[index]};
        }
    }
}
