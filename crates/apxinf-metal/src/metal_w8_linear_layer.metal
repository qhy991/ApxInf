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
