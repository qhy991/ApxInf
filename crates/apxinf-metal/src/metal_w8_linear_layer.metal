#include <metal_stdlib>
using namespace metal;

struct LinearLayerParams {
    uint hidden_size;
    float rms_norm_eps;
};

// One threadgroup normalizes a single decode row. The fixed 256-lane
// reduction is deterministic for a given Metal implementation and leaves the
// normalized row resident for the following packed projection.
kernel void linear_layer_rms_norm(
    device const float *input [[buffer(0)]],
    device const float *weight [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant LinearLayerParams& params [[buffer(3)]],
    uint lane [[thread_index_in_threadgroup]]) {
    threadgroup float partial[256];
    float sum = 0.0f;
    for (uint index = lane; index < params.hidden_size; index += 256) {
        const float value = input[index];
        sum += value * value;
    }
    partial[lane] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128; stride != 0; stride >>= 1) {
        if (lane < stride) {
            partial[lane] += partial[lane + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float inverse_rms = rsqrt(partial[0] / float(params.hidden_size) +
                                    params.rms_norm_eps);
    for (uint index = lane; index < params.hidden_size; index += 256) {
        output[index] = input[index] * inverse_rms * weight[index];
    }
}

// Additive, opt-in exact-tree screen for 32-lane Apple SIMDgroups. The first
// three shared-memory stages preserve the legacy 256-lane reduction. Every
// SIMDgroup then redundantly loads the same 32 partials and completes the
// legacy 16..1 operand tree locally, which avoids five threadgroup barriers
// without reducing the 256 output writers.
kernel void linear_layer_rms_norm_simd_tail_exact_v1(
    device const float *input [[buffer(0)]],
    device const float *weight [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant LinearLayerParams& params [[buffer(3)]],
    uint lane [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]]) {
    threadgroup float partial[256];
    float sum = 0.0f;
    for (uint index = lane; index < params.hidden_size; index += 256) {
        const float value = input[index];
        sum += value * value;
    }
    partial[lane] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128; stride >= 32; stride >>= 1) {
        if (lane < stride) {
            partial[lane] += partial[lane + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float simd_sum = partial[simd_lane];
    float shuffled = simd_shuffle_down(simd_sum, 16);
    if (simd_lane < 16) {
        simd_sum += shuffled;
    }
    shuffled = simd_shuffle_down(simd_sum, 8);
    if (simd_lane < 8) {
        simd_sum += shuffled;
    }
    shuffled = simd_shuffle_down(simd_sum, 4);
    if (simd_lane < 4) {
        simd_sum += shuffled;
    }
    shuffled = simd_shuffle_down(simd_sum, 2);
    if (simd_lane < 2) {
        simd_sum += shuffled;
    }
    shuffled = simd_shuffle_down(simd_sum, 1);
    if (simd_lane < 1) {
        simd_sum += shuffled;
    }
    const float total = simd_broadcast(simd_sum, 0);
    const float inverse_rms = rsqrt(total / float(params.hidden_size) +
                                    params.rms_norm_eps);
    for (uint index = lane; index < params.hidden_size; index += 256) {
        output[index] = input[index] * inverse_rms * weight[index];
    }
}

kernel void linear_layer_residual_add(
    device const float *residual [[buffer(0)]],
    device const float *update [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant LinearLayerParams& params [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
    if (index < params.hidden_size) {
        output[index] = residual[index] + update[index];
    }
}
