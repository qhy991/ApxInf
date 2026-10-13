"""Q4 candidate A: compressed QKV and gate/up concatenation plus compiled elementwise seams.

Weights stay affine Q4/group64: packed uint32 rows and their bf16 scales/biases are
concatenated along the output axis only, and every projection runs through
mx.quantized_matmul. The caller's block and checkpoint tensors are never mutated.
"""
def create(block):
    # One setup per worker, before timed calls. No preprocessing of case inputs.
    import mlx.core as mx
    from mlx import nn
    from mflux.models.qwen21.model.qwen21_transformer.qwen21_fused_kernels import fused_qk_norm_rope

    attn, mlp = block.attn, block.img_mlp

    def packed(*layers):
        for layer in layers:
            if (not isinstance(layer, nn.QuantizedLinear) or layer.mode != 'affine' or layer.bits != 4
                    or layer.group_size != 64 or 'bias' in layer or layer.get('biases') is None
                    or layer.weight.dtype != mx.uint32 or layer.weight.shape[1] != 4096 * 4 // 32):
                raise ValueError('expected affine Q4/group64 bias-free projections over 4096 inputs')
        return tuple(mx.concatenate([layer[name] for layer in layers], axis=0)
                     for name in ('weight', 'scales', 'biases'))

    qkv = packed(attn.to_q, attn.to_k, attn.to_v)
    gate_up = packed(mlp.gate_layer, mlp.proj)
    mx.eval(qkv, gate_up)
    if not attn.use_fused_prologue or attn.norm_q.eps != 1e-6 or attn.norm_k.eps != 1e-6:
        raise ValueError('native fused Q/K norm+RoPE prologue unavailable; no fallback')
    if attn.compute_precision.dtype is not None or mlp.compute_precision.dtype is not None:
        raise ValueError('native compute precision must be the stream dtype')

    def q4(x, weights):
        w, scales, biases = weights
        return mx.quantized_matmul(x, w, scales=scales, biases=biases, transpose=True,
                                   group_size=64, bits=4, mode='affine')

    # Same expressions and operation order as the native block; each fused
    # intermediate keeps the op's bf16 output type, so cast seams are unchanged.
    modulate = mx.compile(lambda normed, scale: normed * (1 + scale))
    residual = mx.compile(lambda hidden, gate, update: hidden + mx.tanh(gate) * update)
    swiglu = mx.compile(lambda gate, proj: (gate * mx.sigmoid(gate)) * proj)

    def run(x,k,v,mod1,mod2,cos,sin):
        t=x.shape[1]
        if x.shape!=(1,t,4096) or t not in (1024,2304) or k.shape[2] not in (128,512):
            raise ValueError('unsupported task shape; no fallback')
        if k.shape!=(1,32,k.shape[2],128) or v.shape!=k.shape or mod1.shape!=(1,t,8192) or mod2.shape!=mod1.shape or cos.shape!=(t,64) or sin.shape!=cos.shape:
            raise ValueError('input ABI mismatch')
        if any(a.dtype!=mx.bfloat16 for a in (x,k,v,mod1,mod2)) or cos.dtype!=mx.float32 or sin.dtype!=mx.float32:
            raise ValueError('input dtype mismatch')
        scale1, gate1 = mx.split(mod1, 2, axis=-1)
        scale2, gate2 = mx.split(mod2, 2, axis=-1)

        projected = q4(modulate(block.img_norm1(x), scale1), qkv)
        fused = fused_qk_norm_rope(projected[:, :, :4096], projected[:, :, 4096:8192],
                                   attn.norm_q.weight, attn.norm_k.weight, cos, sin, 32, 128)
        if fused is None:
            raise ValueError('native fused Q/K norm+RoPE rejected the projection views')
        query, key = fused
        value = mx.transpose(mx.reshape(projected, (1, t, 3, 32, 128))[:, :, 2], (0, 2, 1, 3))
        key = mx.concatenate([k, key], axis=2)
        value = mx.concatenate([v, value], axis=2)
        attended = mx.fast.scaled_dot_product_attention(query, key, value, scale=128**-0.5)
        x = residual(x, gate1, attn._unproject(attended))

        hidden = q4(modulate(block.img_norm2(x), scale2), gate_up)
        return residual(x, gate2, mlp.out(swiglu(hidden[:, :, :12288], hidden[:, :, 12288:])))
    return run
