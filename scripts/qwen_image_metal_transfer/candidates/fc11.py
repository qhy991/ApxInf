"""Private BF16 projections with SwiGLU and gated residuals in GEMM epilogues."""

import mlx.core as mx
from mflux.models.qwen21.model.qwen21_transformer.qwen21_fused_kernels import fused_qk_norm_rope


# out[M, N] = lhs[M, K] @ rhs[N, K]^T with FP32 accumulation; M, N multiples of 64, K of 32.
_GEMM_SOURCE = """
    constexpr uint BM = 64;
    constexpr uint BN = 64;
    constexpr uint BK = 32;
    constexpr uint LD = BK + 8;
    // uint4 storage keeps the 16-byte tile stores aligned.
    threadgroup uint4 lhs_storage[BM * LD / 8];
    threadgroup uint4 rhs_storage[BN * LD / 8];
    threadgroup bfloat16_t* lhs_tile = (threadgroup bfloat16_t*)lhs_storage;
    threadgroup bfloat16_t* rhs_tile = (threadgroup bfloat16_t*)rhs_storage;

    // Groups of 8 M-tiles walk N together so concurrent threadgroups share k-slabs.
    const uint tiles_m = M / BM;
    const uint tiles_n = N / BN;
    const uint tile_id = threadgroup_position_in_grid.x;
    const uint group_width = 8 * tiles_n;
    const uint group_id = tile_id / group_width;
    const uint first_m = group_id * 8;
    const uint group_rows = metal::min(tiles_m - first_m, 8u);
    const uint in_group = tile_id - group_id * group_width;
    const uint row0 = (first_m + in_group % group_rows) * BM;
    const uint col0 = (in_group / group_rows) * BN;

    const uint tid = thread_index_in_threadgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;

    // Each thread copies two 8-element chunks of each operand per k-slab.
    const uint load_row = tid / 4;
    const uint load_col = (tid % 4) * 8;
    const device bfloat16_t* lhs_src = lhs + (row0 + load_row) * K + load_col;
    const device bfloat16_t* rhs_src = rhs + (col0 + load_row) * K + load_col;
    threadgroup bfloat16_t* lhs_dst = lhs_tile + load_row * LD + load_col;
    threadgroup bfloat16_t* rhs_dst = rhs_tile + load_row * LD + load_col;

    // Lane position inside an 8x8 SIMD-group matrix, as in MLX steel mma.h.
    const uint qid = lane / 4;
    const uint fm = (qid & 4) + (lane / 2) % 4;
    const uint fn = (qid & 2) * 2 + (lane % 2) * 2;
    const uint sg_row = (sg / 2) * 32;
    const uint sg_col = (sg % 2) * 32;

    metal::simdgroup_float8x8 acc[4][4];
    #pragma clang loop unroll(full)
    for (uint i = 0; i < 4; ++i) {
        #pragma clang loop unroll(full)
        for (uint j = 0; j < 4; ++j) {
            acc[i][j] = metal::make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        }
    }

    uint4 lhs_next0 = *(const device uint4*)(lhs_src);
    uint4 lhs_next1 = *(const device uint4*)(lhs_src + 32 * K);
    uint4 rhs_next0 = *(const device uint4*)(rhs_src);
    uint4 rhs_next1 = *(const device uint4*)(rhs_src + 32 * K);

    for (uint k0 = 0; k0 < K; k0 += BK) {
        metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
        *(threadgroup uint4*)(lhs_dst) = lhs_next0;
        *(threadgroup uint4*)(lhs_dst + 32 * LD) = lhs_next1;
        *(threadgroup uint4*)(rhs_dst) = rhs_next0;
        *(threadgroup uint4*)(rhs_dst + 32 * LD) = rhs_next1;
        metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);

        if (k0 + BK < K) {
            lhs_next0 = *(const device uint4*)(lhs_src + k0 + BK);
            lhs_next1 = *(const device uint4*)(lhs_src + 32 * K + k0 + BK);
            rhs_next0 = *(const device uint4*)(rhs_src + k0 + BK);
            rhs_next1 = *(const device uint4*)(rhs_src + 32 * K + k0 + BK);
        }

        #pragma clang loop unroll(full)
        for (uint kk = 0; kk < BK; kk += 8) {
            metal::simdgroup_float8x8 lhs_frag[4];
            metal::simdgroup_float8x8 rhs_frag[4];
            #pragma clang loop unroll(full)
            for (uint i = 0; i < 4; ++i) {
                const threadgroup bfloat16_t* p = lhs_tile + (sg_row + i * 8 + fm) * LD + kk + fn;
                lhs_frag[i].thread_elements()[0] = static_cast<float>(p[0]);
                lhs_frag[i].thread_elements()[1] = static_cast<float>(p[1]);
            }
            // rhs rows are output columns, so the fragment is read transposed.
            #pragma clang loop unroll(full)
            for (uint j = 0; j < 4; ++j) {
                const threadgroup bfloat16_t* p = rhs_tile + (sg_col + j * 8 + fn) * LD + kk + fm;
                rhs_frag[j].thread_elements()[0] = static_cast<float>(p[0]);
                rhs_frag[j].thread_elements()[1] = static_cast<float>(p[LD]);
            }
            #pragma clang loop unroll(full)
            for (uint i = 0; i < 4; ++i) {
                #pragma clang loop unroll(full)
                for (uint j = 0; j < 4; ++j) {
                    metal::simdgroup_multiply_accumulate(acc[i][j], lhs_frag[i], rhs_frag[j], acc[i][j]);
                }
            }
        }
    }
"""

_GEMM_STORE = """
    #pragma clang loop unroll(full)
    for (uint i = 0; i < 4; ++i) {
        #pragma clang loop unroll(full)
        for (uint j = 0; j < 4; ++j) {
            device bfloat16_t* dst = out + (row0 + sg_row + i * 8 + fm) * N + col0 + sg_col + j * 8 + fn;
            dst[0] = static_cast<bfloat16_t>(acc[i][j].thread_elements()[0]);
            dst[1] = static_cast<bfloat16_t>(acc[i][j].thread_elements()[1]);
        }
    }
"""

_GATED_STORE = """
    #pragma clang loop unroll(full)
    for (uint j = 0; j < 4; ++j) {
        const uint col = col0 + sg_col + j * 8 + fn;
        const bfloat16_t g0 = static_cast<bfloat16_t>(metal::precise::tanh(static_cast<float>(gate[col])));
        const bfloat16_t g1 = static_cast<bfloat16_t>(metal::precise::tanh(static_cast<float>(gate[col + 1])));
        #pragma clang loop unroll(full)
        for (uint i = 0; i < 4; ++i) {
            const uint index = (row0 + sg_row + i * 8 + fm) * N + col;
            const bfloat16_t u0 = static_cast<bfloat16_t>(acc[i][j].thread_elements()[0]);
            const bfloat16_t u1 = static_cast<bfloat16_t>(acc[i][j].thread_elements()[1]);
            const bfloat16_t v0 = static_cast<bfloat16_t>(static_cast<float>(g0) * static_cast<float>(u0));
            const bfloat16_t v1 = static_cast<bfloat16_t>(static_cast<float>(g1) * static_cast<float>(u1));
            out[index] = static_cast<bfloat16_t>(static_cast<float>(residual[index]) + static_cast<float>(v0));
            out[index + 1] = static_cast<bfloat16_t>(static_cast<float>(residual[index + 1]) + static_cast<float>(v1));
        }
    }
"""

_SWIGLU_HEADER = """
inline float bf16_rounded(float x) {
    return static_cast<float>(static_cast<bfloat16_t>(x));
}
"""

# Weight rows are interleaved in setup so fragment columns j and j + 2 hold gate and up of the same features.
_SWIGLU_STORE = """
    #pragma clang loop unroll(full)
    for (uint i = 0; i < 4; ++i) {
        #pragma clang loop unroll(full)
        for (uint j = 0; j < 2; ++j) {
            device bfloat16_t* dst = out + (row0 + sg_row + i * 8 + fm) * (N / 2) + (col0 + sg_col) / 2 + j * 8 + fn;
            #pragma clang loop unroll(full)
            for (uint e = 0; e < 2; ++e) {
                // BF16 after every op, as in the compiled silu(gate) * up graph and MLX's Sigmoid functor.
                const float g = bf16_rounded(acc[i][j].thread_elements()[e]);
                const float u = bf16_rounded(acc[i][j + 2].thread_elements()[e]);
                const float y = bf16_rounded(1.0f / bf16_rounded(1.0f + bf16_rounded(metal::precise::exp(metal::abs(g)))));
                const float s = g < 0.0f ? y : bf16_rounded(1.0f - y);
                dst[e] = static_cast<bfloat16_t>(bf16_rounded(g * s) * u);
            }
        }
    }
"""

_gemm_kernel = mx.fast.metal_kernel(
    name="bf16_nt_gemm_64x64x32",
    input_names=["lhs", "rhs"],
    output_names=["out"],
    source=_GEMM_SOURCE + _GEMM_STORE,
)

_gated_gemm_kernel = mx.fast.metal_kernel(
    name="bf16_nt_gemm_gated_residual_64x64x32",
    input_names=["lhs", "rhs", "residual", "gate"],
    output_names=["out"],
    source=_GEMM_SOURCE + _GATED_STORE,
)

_swiglu_gemm_kernel = mx.fast.metal_kernel(
    name="bf16_nt_gemm_swiglu_64x64x32",
    input_names=["lhs", "rhs"],
    output_names=["out"],
    source=_GEMM_SOURCE + _SWIGLU_STORE,
    header=_SWIGLU_HEADER,
)


def _linear(x, weight):
    m, k = x.shape[1], x.shape[2]
    n = weight.shape[0]
    (out,) = _gemm_kernel(
        inputs=[mx.reshape(x, (m, k)), weight],
        template=[("M", m), ("N", n), ("K", k)],
        grid=((m // 64) * (n // 64) * 128, 1, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(m, n)],
        output_dtypes=[mx.bfloat16],
    )
    return mx.reshape(out, (1, m, n))


def _linear_gated(x, weight, residual, gate):
    m, k = x.shape[1], x.shape[2]
    n = weight.shape[0]
    (out,) = _gated_gemm_kernel(
        inputs=[mx.reshape(x, (m, k)), weight, mx.reshape(residual, (m, n)), gate[0, 0, :]],
        template=[("M", m), ("N", n), ("K", k)],
        grid=((m // 64) * (n // 64) * 128, 1, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(m, n)],
        output_dtypes=[mx.bfloat16],
    )
    return mx.reshape(out, (1, m, n))


def _linear_swiglu(x, weight):
    m, k = x.shape[1], x.shape[2]
    n = weight.shape[0]
    (out,) = _swiglu_gemm_kernel(
        inputs=[mx.reshape(x, (m, k)), weight],
        template=[("M", m), ("N", n), ("K", k)],
        grid=((m // 64) * (n // 64) * 128, 1, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(m, n // 2)],
        output_dtypes=[mx.bfloat16],
    )
    return mx.reshape(out, (1, m, n // 2))


@mx.compile
def _modulate(normed, scale):
    return normed * (1 + scale)


def create(block):
    mlp = block.img_mlp
    attn = block.attn
    if mlp.compute_precision.dtype is not None or attn.compute_precision.dtype is not None:
        raise ValueError("this candidate requires native BF16 projection computation")
    if attn.num_heads != 32 or attn.head_dim != 128:
        raise ValueError("expected 32 attention heads of width 128")

    def prepare(layer):
        if layer.bits != 4 or layer.group_size != 64 or layer.mode != "affine" or "bias" in layer:
            raise ValueError("expected affine Q4/group64 projection without linear bias")
        # Match the QMM tile loader's FP32 affine decode and BF16 narrowing.
        weight = mx.dequantize(
            layer.weight,
            layer.scales.astype(mx.float32),
            layer.biases.astype(mx.float32),
            group_size=64,
            bits=4,
            mode="affine",
        ).astype(mx.bfloat16)
        mx.eval(weight)
        return weight

    gate_weight = prepare(mlp.gate_layer)
    up_weight = prepare(mlp.proj)
    # Rows per 64-row tile: gate f..f+15, up f..f+15, gate f+16..f+31, up f+16..f+31.
    features, width = gate_weight.shape
    blocks = (features // 32, 2, 16, width)
    gate_up_weight = mx.reshape(
        mx.stack([mx.reshape(gate_weight, blocks), mx.reshape(up_weight, blocks)], axis=2),
        (2 * features, width),
    )
    mx.eval(gate_up_weight)
    del gate_weight, up_weight
    down_weight = prepare(mlp.out)
    query_weight = prepare(attn.to_q)
    key_weight = prepare(attn.to_k)
    value_weight = prepare(attn.to_v)
    qkv_weight = mx.concatenate([query_weight, key_weight, value_weight], axis=0)
    mx.eval(qkv_weight)
    del query_weight, key_weight, value_weight
    output_weight = prepare(attn.to_out[0])

    def image_attention(hidden_states, key_text, value_text, cos, sin, residual, gate):
        qkv = _linear(hidden_states, qkv_weight)
        query, key, value = mx.split(qkv, 3, axis=-1)
        head_shape = (*hidden_states.shape[:-1], 32, 128)
        fused = None
        if attn.use_fused_prologue and attn.norm_q.eps == attn.norm_k.eps == 1e-6:
            fused = fused_qk_norm_rope(
                query, key, attn.norm_q.weight, attn.norm_k.weight, cos, sin, 32, 128
            )
        if fused is not None:
            query, key = fused
        else:
            query = attn.norm_q(mx.reshape(query, head_shape))
            key = attn.norm_k(mx.reshape(key, head_shape))
            query = mx.transpose(attn._apply_rope(query, cos, sin), (0, 2, 1, 3))
            key = mx.transpose(attn._apply_rope(key, cos, sin), (0, 2, 1, 3))
        value = mx.transpose(mx.reshape(value, head_shape), (0, 2, 1, 3))
        key = mx.concatenate([key_text, key], axis=2)
        value = mx.concatenate([value_text, value], axis=2)
        output = mx.fast.scaled_dot_product_attention(query, key, value, scale=128**-0.5)
        output = mx.transpose(output, (0, 2, 1, 3))
        output = mx.reshape(output, (*hidden_states.shape[:-1], 4096))
        return _linear_gated(output, output_weight, residual, gate)

    def run(x, k, v, mod1, mod2, cos, sin):
        t = x.shape[1]
        if x.shape != (1, t, 4096) or t not in (1024, 2304) or k.shape[2] not in (128, 512):
            raise ValueError("unsupported task shape; no fallback")
        if (
            k.shape != (1, 32, k.shape[2], 128)
            or v.shape != k.shape
            or mod1.shape != (1, t, 8192)
            or mod2.shape != mod1.shape
            or cos.shape != (t, 64)
            or sin.shape != cos.shape
        ):
            raise ValueError("input ABI mismatch")
        if (
            any(a.dtype != mx.bfloat16 for a in (x, k, v, mod1, mod2))
            or cos.dtype != mx.float32
            or sin.dtype != mx.float32
        ):
            raise ValueError("input dtype mismatch")

        scale1, gate1 = mx.split(mod1, 2, axis=-1)
        scale2, gate2 = mx.split(mod2, 2, axis=-1)
        attn_input = _modulate(block.img_norm1(x), scale1)
        hidden = image_attention(attn_input, k, v, cos, sin, x, gate1)
        mlp_input = _modulate(block.img_norm2(hidden), scale2)
        return _linear_gated(_linear_swiglu(mlp_input, gate_up_weight), down_weight, hidden, gate2)

    return run
