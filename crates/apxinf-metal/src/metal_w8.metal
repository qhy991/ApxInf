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

// Scalar form used by the hierarchical final reduction. Keeping every field
// explicit gives the compiler a fixed register-only top-4 and avoids indexing
// a private array during the SIMD shuffle/merge stages.
struct Top4 {
    float score0;
    float score1;
    float score2;
    float score3;
    uint token0;
    uint token1;
    uint token2;
    uint token3;
};

inline Top4 empty_top4() {
    return Top4{-INFINITY, -INFINITY, -INFINITY, -INFINITY,
                UINT_MAX, UINT_MAX, UINT_MAX, UINT_MAX};
}

inline void insert_top4(thread Top4& top, float score, uint token) {
    if (!candidate_better(score, token, top.score3, top.token3)) {
        return;
    }
    if (candidate_better(score, token, top.score0, top.token0)) {
        top.score3 = top.score2;
        top.token3 = top.token2;
        top.score2 = top.score1;
        top.token2 = top.token1;
        top.score1 = top.score0;
        top.token1 = top.token0;
        top.score0 = score;
        top.token0 = token;
    } else if (candidate_better(score, token, top.score1, top.token1)) {
        top.score3 = top.score2;
        top.token3 = top.token2;
        top.score2 = top.score1;
        top.token2 = top.token1;
        top.score1 = score;
        top.token1 = token;
    } else if (candidate_better(score, token, top.score2, top.token2)) {
        top.score3 = top.score2;
        top.token3 = top.token2;
        top.score2 = score;
        top.token2 = token;
    } else {
        top.score3 = score;
        top.token3 = token;
    }
}

inline void merge_top4(thread Top4& target, Top4 source) {
    insert_top4(target, source.score0, source.token0);
    insert_top4(target, source.score1, source.token1);
    insert_top4(target, source.score2, source.token2);
    insert_top4(target, source.score3, source.token3);
}

// Every lane executes every shuffle. Only the left receiver of each disjoint
// pair merges, and that decision happens after all collective reads.
inline void merge_top4_shuffle_down(thread Top4& top, uint lane, ushort offset) {
    const Top4 other = Top4{
        simd_shuffle_down(top.score0, offset),
        simd_shuffle_down(top.score1, offset),
        simd_shuffle_down(top.score2, offset),
        simd_shuffle_down(top.score3, offset),
        simd_shuffle_down(top.token0, offset),
        simd_shuffle_down(top.token1, offset),
        simd_shuffle_down(top.token2, offset),
        simd_shuffle_down(top.token3, offset),
    };
    const uint span = uint(offset) * 2;
    if (lane % span == 0 && lane + uint(offset) < 32) {
        merge_top4(top, other);
    }
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

// The top-4 operator is exact for a total score-descending/token-ascending
// order: top4(A union B) == top4(top4(A) union top4(B)). Each thread first
// scans its strided subset, each SIMD-group then reduces its 32 register-only
// lists, and SIMD-group zero reduces the eight leaders after one TG barrier.
kernel void w8_final_topk4_simd_hierarchical(
    device const Candidate *partial [[buffer(0)]],
    device uint *output_tokens [[buffer(1)]],
    constant KernelParams& params [[buffer(2)]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]]) {
    constexpr uint threads = 256;
    constexpr uint simdgroups = 8;
    threadgroup float leader_scores[simdgroups * top_k];
    threadgroup uint leader_tokens[simdgroups * top_k];

    Top4 local = empty_top4();
    const uint candidate_count = params.partial_count * top_k;
    for (uint index = tid; index < candidate_count; index += threads) {
        const Candidate candidate = partial[index];
        const bool valid = candidate.token < params.rows && !isnan(candidate.score);
        insert_top4(local,
                    valid ? candidate.score : -INFINITY,
                    valid ? candidate.token : UINT_MAX);
    }

    merge_top4_shuffle_down(local, lane, 1);
    merge_top4_shuffle_down(local, lane, 2);
    merge_top4_shuffle_down(local, lane, 4);
    merge_top4_shuffle_down(local, lane, 8);
    merge_top4_shuffle_down(local, lane, 16);

    if (lane == 0) {
        const uint base = simdgroup * top_k;
        leader_scores[base] = local.score0;
        leader_scores[base + 1] = local.score1;
        leader_scores[base + 2] = local.score2;
        leader_scores[base + 3] = local.score3;
        leader_tokens[base] = local.token0;
        leader_tokens[base + 1] = local.token1;
        leader_tokens[base + 2] = local.token2;
        leader_tokens[base + 3] = local.token3;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simdgroup == 0) {
        Top4 leader = empty_top4();
        if (lane < simdgroups) {
            const uint base = lane * top_k;
            leader = Top4{
                leader_scores[base],
                leader_scores[base + 1],
                leader_scores[base + 2],
                leader_scores[base + 3],
                leader_tokens[base],
                leader_tokens[base + 1],
                leader_tokens[base + 2],
                leader_tokens[base + 3],
            };
        }
        merge_top4_shuffle_down(leader, lane, 1);
        merge_top4_shuffle_down(leader, lane, 2);
        merge_top4_shuffle_down(leader, lane, 4);
        if (lane == 0) {
            output_tokens[0] = leader.token0;
            output_tokens[1] = leader.token1;
            output_tokens[2] = leader.token2;
            output_tokens[3] = leader.token3;
        }
    }
}
