// Precision-v2 GDN output projection. This source is appended only to the
// complete linear-layer library; standalone GDN and every legacy G64 kernel
// retain their original source bytes and ABI.
kernel void gdn_w8_output_projection_g32(
    device const char4 *weights [[buffer(0)]],
    device const float *scales [[buffer(1)]],
    device const float4 *input [[buffer(2)]],
    device float *output [[buffer(3)]],
    constant GdnParams& params [[buffer(4)]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint group [[threadgroup_position_in_grid]]) {
    constexpr uint rows_per_threadgroup = 8;
    constexpr uint float4_per_group = 8;
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
