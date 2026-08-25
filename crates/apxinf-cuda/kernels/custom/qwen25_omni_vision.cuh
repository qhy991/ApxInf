#pragma once

// Qwen2.5-Omni vision Q/K/V projection epilogue. The projection output is
// first bias-added and rounded to BF16, then Q/K apply the incumbent 2-D RoPE
// pair order while V publishes the rounded value directly.
template <int kHeads, int kHeadDim>
__global__ void qwen25_omni_vision_qkv_bias_rope_bf16_kernel(
    const __nv_bfloat16* query, const __nv_bfloat16* key,
    const __nv_bfloat16* value, const __nv_bfloat16* query_bias,
    const __nv_bfloat16* key_bias, const __nv_bfloat16* value_bias,
    __nv_bfloat16* query_output, __nv_bfloat16* key_output,
    __nv_bfloat16* value_output, int sequence, float theta,
    const uint32_t* positions) {
  constexpr int kHidden = kHeads * kHeadDim;
  constexpr int kHalf = kHeadDim / 2;
  const int item = blockIdx.x * blockDim.x + threadIdx.x;
  const int token = blockIdx.y;
  const int projection = blockIdx.z;
  if (token >= sequence) return;

  if (projection == 2) {
    if (item >= kHidden) return;
    const int64_t index = static_cast<int64_t>(token) * kHidden + item;
    value_output[index] = __float2bfloat16(
        __bfloat162float(value[index]) +
        __bfloat162float(value_bias[item]));
    return;
  }

  if (item >= kHeads * kHalf) return;
  const int head = item / kHalf;
  const int pair = item - head * kHalf;
  const int base = token * kHidden + head * kHeadDim;
  const int index0 = base + pair;
  const int index1 = base + kHalf + pair;
  const __nv_bfloat16* input = projection == 0 ? query : key;
  const __nv_bfloat16* bias = projection == 0 ? query_bias : key_bias;
  __nv_bfloat16* output = projection == 0 ? query_output : key_output;
  const __nv_bfloat16 rounded0 = __float2bfloat16(
      __bfloat162float(input[index0]) +
      __bfloat162float(bias[head * kHeadDim + pair]));
  const __nv_bfloat16 rounded1 = __float2bfloat16(
      __bfloat162float(input[index1]) +
      __bfloat162float(bias[head * kHeadDim + kHalf + pair]));
  const int axis = pair < kHalf / 2 ? 0 : 1;
  const int pair_in_axis = pair < kHalf / 2 ? pair : pair - kHalf / 2;
  const uint32_t position = positions[token * 2 + axis];
  const float frequency =
      1.0f / powf(theta, 2.0f * static_cast<float>(pair_in_axis) / kHalf);
  const float angle = static_cast<float>(position) * frequency;
  const float cosine = cosf(angle);
  const float sine = sinf(angle);
  const float value0 = __bfloat162float(rounded0);
  const float value1 = __bfloat162float(rounded1);
  output[index0] = __float2bfloat16(value0 * cosine - value1 * sine);
  output[index1] = __float2bfloat16(value0 * sine + value1 * cosine);
}
