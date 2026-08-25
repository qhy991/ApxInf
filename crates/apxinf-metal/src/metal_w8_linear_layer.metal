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

// Diagnostic H=1024 primitive only. This fuses the attention residual add
// into the following RMSNorm while still materializing the residual row. The
// four values owned by each of the 256 lanes remain live across the unchanged
// legacy reduction tree, avoiding the two source-level residual-row reads in
// the standalone RMS kernel. Reassociation is disabled only for the final
// retained-value product because default fast math otherwise changes its
// left-associated result by one ULP for some elements.
// Production bridges do not select this function.
kernel void linear_layer_residual_rms_norm_fused_exact_v1(
    device const float *residual [[buffer(0)]],
    device const float *update [[buffer(1)]],
    device const float *weight [[buffer(2)]],
    device float *materialized_residual [[buffer(3)]],
    device float *normalized [[buffer(4)]],
    constant LinearLayerParams& params [[buffer(5)]],
    uint lane [[thread_index_in_threadgroup]]) {
    threadgroup float partial[256];

    float4 retained_values;
    float sum = 0.0f;
    uint retained_index = 0;
    for (uint index = lane; index < params.hidden_size; index += 256) {
        const float value = residual[index] + update[index];
        materialized_residual[index] = value;
        retained_values[retained_index] = value;
        retained_index += 1;
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
    {
#pragma clang fp reassociate(off)
        retained_index = 0;
        for (uint index = lane; index < params.hidden_size; index += 256) {
            normalized[index] =
                retained_values[retained_index] * inverse_rms * weight[index];
            retained_index += 1;
        }
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
