"""Private BF16 projection weights, compiled elementwise glue, native QK norm/RoPE and SDPA."""

import mlx.core as mx
from mlx import nn
from mflux.models.qwen21.model.qwen21_transformer.qwen21_fused_kernels import fused_qk_norm_rope


@mx.compile
def _modulate(normed, scale):
    return normed * (1 + scale)


@mx.compile
def _gated_residual(residual, gate, update):
    return residual + nn.tanh(gate) * update


@mx.compile
def _swiglu(gate, up):
    return nn.silu(gate) * up


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
    down_weight = prepare(mlp.out)
    query_weight = prepare(attn.to_q)
    key_weight = prepare(attn.to_k)
    value_weight = prepare(attn.to_v)
    output_weight = prepare(attn.to_out[0])

    def image_attention(hidden_states, key_text, value_text, cos, sin):
        query = hidden_states @ query_weight.T
        key = hidden_states @ key_weight.T
        value = hidden_states @ value_weight.T
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
        return output @ output_weight.T

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
        hidden = _gated_residual(x, gate1, image_attention(attn_input, k, v, cos, sin))
        mlp_input = _modulate(block.img_norm2(hidden), scale2)
        gate = mlp_input @ gate_weight.T
        up = mlp_input @ up_weight.T
        mlp_output = _swiglu(gate, up) @ down_weight.T
        return _gated_residual(hidden, gate2, mlp_output)

    return run
