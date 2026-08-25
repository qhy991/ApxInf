#include <metal_stdlib>
using namespace metal;

struct KernelParams {
    uint columns;
    uint rows;
    uint groups_per_row;
    uint partial_count;
};

struct Candidate {
    float score;
    uint token;
};

constant uint top_k = 4;

inline bool candidate_better(float score, uint token, float current_score,
                             uint current_token) {
    return score > current_score || (score == current_score && token < current_token);
}

inline void insert_candidate(thread float *scores, thread uint *tokens,
                             float score, uint token) {
    if (!candidate_better(score, token, scores[top_k - 1], tokens[top_k - 1])) {
        return;
    }
    uint position = top_k - 1;
    while (position != 0 &&
           candidate_better(score, token, scores[position - 1], tokens[position - 1])) {
        scores[position] = scores[position - 1];
        tokens[position] = tokens[position - 1];
        --position;
    }
    scores[position] = score;
    tokens[position] = token;
}

// One SIMD-group evaluates one vocabulary row. Eight rows share a threadgroup.
// Keep four candidates from every group: reducing each group to top-1 would be
// incorrect when several global winners happen to occupy the same group.
kernel void w8_rows_topk4(
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
    constexpr uint float4_per_group = 16;
    threadgroup float row_scores[rows_per_threadgroup];
    threadgroup uint row_tokens[rows_per_threadgroup];

    const uint row = group * rows_per_threadgroup + simdgroup;
    const uint columns4 = params.columns >> 2;
    float sum = 0.0f;
    if (row < params.rows) {
        const uint weight_base = row * columns4;
        const uint scale_base = row * params.groups_per_row;
        for (uint index = lane; index < columns4; index += 32) {
            const char4 quantized = weights[weight_base + index];
            const float scale = scales[scale_base + index / float4_per_group];
            sum += dot(float4(quantized), hidden[index]) * scale;
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

// Sixteen SIMD-groups evaluate thirty-two vocabulary rows per threadgroup. Each
// SIMD-group handles a consecutive pair and reuses the hidden float4 load for
// both rows. Per-row lane assignment and simd_sum order remain identical to
// w8_rows_topk4, while the first-stage partial list is quartered.
kernel void w8_rows_topk4_pair2_r32_sg16(
    device const char4 *weights [[buffer(0)]],
    device const float *scales [[buffer(1)]],
    device const float4 *hidden [[buffer(2)]],
    device Candidate *partial [[buffer(3)]],
    constant KernelParams& params [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint group [[threadgroup_position_in_grid]]) {
    constexpr uint rows_per_simdgroup = 2;
    constexpr uint simdgroups_per_threadgroup = 16;
    constexpr uint rows_per_threadgroup =
        rows_per_simdgroup * simdgroups_per_threadgroup;
    constexpr uint float4_per_group = 16;
    threadgroup float row_scores[rows_per_threadgroup];
    threadgroup uint row_tokens[rows_per_threadgroup];

    const uint row0 =
        group * rows_per_threadgroup + simdgroup * rows_per_simdgroup;
    const uint row1 = row0 + 1;
    const uint columns4 = params.columns >> 2;
    float sum0 = 0.0f;
    float sum1 = 0.0f;
    for (uint index = lane; index < columns4; index += 32) {
        const float4 hidden_value = hidden[index];
        const uint scale_group = index / float4_per_group;
        if (row0 < params.rows) {
            const char4 quantized = weights[row0 * columns4 + index];
            const float scale =
                scales[row0 * params.groups_per_row + scale_group];
            sum0 += dot(float4(quantized), hidden_value) * scale;
        }
        if (row1 < params.rows) {
            const char4 quantized = weights[row1 * columns4 + index];
            const float scale =
                scales[row1 * params.groups_per_row + scale_group];
            sum1 += dot(float4(quantized), hidden_value) * scale;
        }
    }
    sum0 = simd_sum(sum0);
    sum1 = simd_sum(sum1);
    if (lane == 0) {
        const uint slot = simdgroup * rows_per_simdgroup;
        row_scores[slot] =
            row0 < params.rows && !isnan(sum0) ? sum0 : -INFINITY;
        row_tokens[slot] = row0 < params.rows ? row0 : UINT_MAX;
        row_scores[slot + 1] =
            row1 < params.rows && !isnan(sum1) ? sum1 : -INFINITY;
        row_tokens[slot + 1] = row1 < params.rows ? row1 : UINT_MAX;
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

// A single threadgroup folds every per-group top-4 list into the global top-4.
// Each thread first produces a sorted local list. Thread zero then merges the
// 256 small lists deterministically and writes only four token IDs.
kernel void w8_final_topk4(
    device const Candidate *partial [[buffer(0)]],
    device uint *output_tokens [[buffer(1)]],
    constant KernelParams& params [[buffer(2)]],
    uint tid [[thread_index_in_threadgroup]]) {
    constexpr uint threads = 256;
    constexpr uint local_candidate_count = threads * top_k;
    threadgroup float local_scores[local_candidate_count];
    threadgroup uint local_tokens[local_candidate_count];
    float scores[top_k] = {-INFINITY, -INFINITY, -INFINITY, -INFINITY};
    uint tokens[top_k] = {UINT_MAX, UINT_MAX, UINT_MAX, UINT_MAX};
    const uint candidate_count = params.partial_count * top_k;
    for (uint index = tid; index < candidate_count; index += threads) {
        const Candidate candidate = partial[index];
        insert_candidate(scores, tokens, candidate.score, candidate.token);
    }
    const uint local_base = tid * top_k;
    for (uint index = 0; index < top_k; ++index) {
        local_scores[local_base + index] = scores[index];
        local_tokens[local_base + index] = tokens[index];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        float global_scores[top_k] = {-INFINITY, -INFINITY, -INFINITY, -INFINITY};
        uint global_tokens[top_k] = {UINT_MAX, UINT_MAX, UINT_MAX, UINT_MAX};
        for (uint index = 0; index < local_candidate_count; ++index) {
            insert_candidate(global_scores, global_tokens,
                             local_scores[index], local_tokens[index]);
        }
        for (uint index = 0; index < top_k; ++index) {
            output_tokens[index] = global_tokens[index];
        }
    }
}
