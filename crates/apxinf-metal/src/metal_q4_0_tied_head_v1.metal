#include <metal_stdlib>
using namespace metal;

struct Q4_0TiedHeadParamsV1 {
    uint columns;
    uint rows;
    uint blocks_per_row;
    uint partial_count;
    uint excluded_tokens[5];
    uint excluded_count;
};

struct Q4_0CandidateV1 {
    float score;
    uint token;
};

constant uint q4_0_top_k_v1 = 4;
constant uint q4_0_rows_per_threadgroup_v1 = 8;
constant uint q4_0_block_size_v1 = 32;
constant uint q4_0_block_bytes_v1 = 18;
constant uint q4_0_max_excluded_tokens_v1 = 5;

inline bool q4_0_token_is_excluded_v1(
    uint token,
    constant Q4_0TiedHeadParamsV1& params) {
    for (uint index = 0;
         index < params.excluded_count && index < q4_0_max_excluded_tokens_v1;
         ++index) {
        if (params.excluded_tokens[index] == token) {
            return true;
        }
    }
    return false;
}

inline bool q4_0_candidate_better_v1(
    float score,
    uint token,
    float current_score,
    uint current_token) {
    return score > current_score ||
           (score == current_score && token < current_token);
}

inline void q4_0_insert_candidate_v1(
    thread float *scores,
    thread uint *tokens,
    float score,
    uint token) {
    if (!q4_0_candidate_better_v1(
            score,
            token,
            scores[q4_0_top_k_v1 - 1],
            tokens[q4_0_top_k_v1 - 1])) {
        return;
    }
    uint position = q4_0_top_k_v1 - 1;
    while (position != 0 &&
           q4_0_candidate_better_v1(
               score,
               token,
               scores[position - 1],
               tokens[position - 1])) {
        scores[position] = scores[position - 1];
        tokens[position] = tokens[position - 1];
        --position;
    }
    scores[position] = score;
    tokens[position] = token;
}

// One SIMD-group evaluates one vocabulary row. Each lane owns one column in
// every canonical Q4_0 block32, reconstructing the little-endian FP16 scale
// and the llama.cpp low/high nibble halves directly from the 18-byte stream.
// Eight rows share a threadgroup and publish both the full score surface and
// one sorted partial top-4 list.
kernel void q4_0_tied_head_rows_v1(
    device const uchar *packed [[buffer(0)]],
    device const float *hidden [[buffer(1)]],
    device float *full_scores [[buffer(2)]],
    device Q4_0CandidateV1 *partial [[buffer(3)]],
    constant Q4_0TiedHeadParamsV1& params [[buffer(4)]],
    device atomic_uint *status [[buffer(5)]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint group [[threadgroup_position_in_grid]]) {
    threadgroup float row_scores[q4_0_rows_per_threadgroup_v1];
    threadgroup uint row_tokens[q4_0_rows_per_threadgroup_v1];

    const uint row = group * q4_0_rows_per_threadgroup_v1 + simdgroup;
    float sum = 0.0f;
    if (row < params.rows) {
        const ulong row_block_base = ulong(row) * ulong(params.blocks_per_row);
        for (uint block = 0; block < params.blocks_per_row; ++block) {
            const ulong block_byte_base =
                (row_block_base + ulong(block)) * ulong(q4_0_block_bytes_v1);
            const ushort scale_bits =
                ushort(packed[block_byte_base]) |
                (ushort(packed[block_byte_base + 1]) << 8);
            const float scale = float(as_type<half>(scale_bits));
            const uchar nibbles =
                packed[block_byte_base + 2 + (lane & 15)];
            const int quantized = lane < 16
                ? int(nibbles & 15)
                : int(nibbles >> 4);
            sum += float(quantized - 8) * scale *
                   hidden[block * q4_0_block_size_v1 + lane];
        }
    }
    sum = simd_sum(sum);

    if (lane == 0) {
        bool score_is_finite = row < params.rows && isfinite(sum);
        if (row < params.rows) {
            full_scores[row] = score_is_finite ? sum : -INFINITY;
            if (!score_is_finite) {
                atomic_store_explicit(status, 1u, memory_order_relaxed);
            }
        }
        const bool row_is_allowed =
            score_is_finite && !q4_0_token_is_excluded_v1(row, params);
        row_scores[simdgroup] = row_is_allowed ? sum : -INFINITY;
        row_tokens[simdgroup] = row_is_allowed ? row : UINT_MAX;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        float scores[q4_0_top_k_v1] = {
            -INFINITY, -INFINITY, -INFINITY, -INFINITY};
        uint tokens[q4_0_top_k_v1] = {
            UINT_MAX, UINT_MAX, UINT_MAX, UINT_MAX};
        for (uint index = 0;
             index < q4_0_rows_per_threadgroup_v1;
             ++index) {
            q4_0_insert_candidate_v1(
                scores,
                tokens,
                row_scores[index],
                row_tokens[index]);
        }
        const uint base = group * q4_0_top_k_v1;
        for (uint index = 0; index < q4_0_top_k_v1; ++index) {
            partial[base + index] =
                Q4_0CandidateV1{scores[index], tokens[index]};
        }
    }
}

// Fold every per-eight-row list into one deterministic global top-4. The
// score/token ordering is total for finite scores: descending score, then the
// lowest token ID for exact ties.
kernel void q4_0_tied_head_final_topk4_v1(
    device const Q4_0CandidateV1 *partial [[buffer(0)]],
    device uint *output_tokens [[buffer(1)]],
    constant Q4_0TiedHeadParamsV1& params [[buffer(2)]],
    uint tid [[thread_index_in_threadgroup]]) {
    constexpr uint threads = 256;
    constexpr uint local_candidate_count = threads * q4_0_top_k_v1;
    threadgroup float local_scores[local_candidate_count];
    threadgroup uint local_tokens[local_candidate_count];
    float scores[q4_0_top_k_v1] = {
        -INFINITY, -INFINITY, -INFINITY, -INFINITY};
    uint tokens[q4_0_top_k_v1] = {
        UINT_MAX, UINT_MAX, UINT_MAX, UINT_MAX};
    const uint candidate_count = params.partial_count * q4_0_top_k_v1;
    for (uint index = tid; index < candidate_count; index += threads) {
        const Q4_0CandidateV1 candidate = partial[index];
        q4_0_insert_candidate_v1(
            scores,
            tokens,
            candidate.score,
            candidate.token);
    }
    const uint local_base = tid * q4_0_top_k_v1;
    for (uint index = 0; index < q4_0_top_k_v1; ++index) {
        local_scores[local_base + index] = scores[index];
        local_tokens[local_base + index] = tokens[index];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        float global_scores[q4_0_top_k_v1] = {
            -INFINITY, -INFINITY, -INFINITY, -INFINITY};
        uint global_tokens[q4_0_top_k_v1] = {
            UINT_MAX, UINT_MAX, UINT_MAX, UINT_MAX};
        for (uint index = 0; index < local_candidate_count; ++index) {
            q4_0_insert_candidate_v1(
                global_scores,
                global_tokens,
                local_scores[index],
                local_tokens[index]);
        }
        for (uint index = 0; index < q4_0_top_k_v1; ++index) {
            output_tokens[index] = global_tokens[index];
        }
    }
}
