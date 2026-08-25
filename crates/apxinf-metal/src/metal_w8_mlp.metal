#include <metal_stdlib>
using namespace metal;

struct MlpParams {
    uint hidden_size;
    uint intermediate_size;
    uint gate_up_groups_per_row;
    uint down_groups_per_row;
};

// First stage of the complete decode MLP. One SIMD-group evaluates one gate
// or up row and eight rows share a threadgroup.
kernel void w8_mlp_gate_up(
    device const char4 *weights [[buffer(0)]],
    device const float *scales [[buffer(1)]],
    device const float4 *input [[buffer(2)]],
    device float *gate_up [[buffer(3)]],
    constant MlpParams& params [[buffer(4)]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint group [[threadgroup_position_in_grid]]) {
    constexpr uint rows_per_threadgroup = 8;
    constexpr uint float4_per_group = 16;
    const uint row = group * rows_per_threadgroup + simdgroup;
    const uint rows = params.intermediate_size * 2;
    if (row >= rows) {
        return;
    }
    const uint columns4 = params.hidden_size >> 2;
    const uint weight_base = row * columns4;
    const uint scale_base = row * params.gate_up_groups_per_row;
    float sum = 0.0f;
    for (uint index = lane; index < columns4; index += 32) {
        const char4 quantized = weights[weight_base + index];
        const float scale = scales[scale_base + index / float4_per_group];
        sum += dot(float4(quantized), input[index]) * scale;
    }
    sum = simd_sum(sum);
    if (lane == 0) {
        gate_up[row] = sum;
    }
}

// The full intermediate row remains GPU-resident between both projections.
kernel void w8_mlp_silu_mul(
    device const float *gate_up [[buffer(0)]],
    device float *activated [[buffer(1)]],
    constant MlpParams& params [[buffer(2)]],
    uint index [[thread_position_in_grid]]) {
    if (index >= params.intermediate_size) {
        return;
    }
    const float gate = gate_up[index];
    const float up = gate_up[index + params.intermediate_size];
    activated[index] = (gate / (1.0f + exp(-gate))) * up;
}

// Final W8 down projection. Only this hidden-width row is copied to the CPU.
kernel void w8_mlp_down(
    device const char4 *weights [[buffer(0)]],
    device const float *scales [[buffer(1)]],
    device const float4 *activated [[buffer(2)]],
    device float *output [[buffer(3)]],
    constant MlpParams& params [[buffer(4)]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint group [[threadgroup_position_in_grid]]) {
    constexpr uint rows_per_threadgroup = 8;
    constexpr uint float4_per_group = 16;
    const uint row = group * rows_per_threadgroup + simdgroup;
    if (row >= params.hidden_size) {
        return;
    }
    const uint columns4 = params.intermediate_size >> 2;
    const uint weight_base = row * columns4;
    const uint scale_base = row * params.down_groups_per_row;
    float sum = 0.0f;
    for (uint index = lane; index < columns4; index += 32) {
        const char4 quantized = weights[weight_base + index];
        const float scale = scales[scale_base + index / float4_per_group];
        sum += dot(float4(quantized), activated[index]) * scale;
    }
    sum = simd_sum(sum);
    if (lane == 0) {
        output[row] = sum;
    }
}
