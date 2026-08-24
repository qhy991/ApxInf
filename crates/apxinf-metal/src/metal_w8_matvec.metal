#include <metal_stdlib>
using namespace metal;

struct KernelParams {
    uint columns;
    uint rows;
    uint groups_per_row;
    uint partial_count;
};

// Generic decode M=1 projection. One SIMD-group evaluates one output row and
// eight rows share a threadgroup. The complete F32 output is copied back only
// after this command buffer finishes; packed weights and scales stay resident.
kernel void w8_rows_matvec(
    device const char4 *weights [[buffer(0)]],
    device const float *scales [[buffer(1)]],
    device const float4 *input [[buffer(2)]],
    device float *output [[buffer(3)]],
    constant KernelParams& params [[buffer(4)]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint group [[threadgroup_position_in_grid]]) {
    constexpr uint rows_per_threadgroup = 8;
    constexpr uint float4_per_group = 16;
    const uint row = group * rows_per_threadgroup + simdgroup;
    if (row >= params.rows) {
        return;
    }
    const uint columns4 = params.columns >> 2;
    const uint weight_base = row * columns4;
    const uint scale_base = row * params.groups_per_row;
    float sum = 0.0f;
    for (uint index = lane; index < columns4; index += 32) {
        const char4 quantized = weights[weight_base + index];
        const float scale = scales[scale_base + index / float4_per_group];
        sum += dot(float4(quantized), input[index]) * scale;
    }
    sum = simd_sum(sum);
    if (lane == 0) {
        output[row] = sum;
    }
}
