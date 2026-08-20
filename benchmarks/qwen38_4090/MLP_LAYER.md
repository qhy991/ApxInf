# Native Qwen3.8 MLP layer on SM89

This layer/module probe uses the real layer-0 checkpoint tensors:

```text
hidden [1,5120]
  -> combined gate/up W4 [34816,5120]
  -> BF16 SiLU(gate) * up [1,17408]
  -> down W4 [5120,17408]
  -> output [1,5120]
```

The checkpoint stores gate and up separately. At load time their packed I32,
BF16 scales, and packed zero-point rows are concatenated into one W4 view. This
does not requantize weights and removes one activation staging plus one kernel
launch. The existing caller-owned ApxInf BF16 SiLU-multiply kernel is reused.

## Correctness

Against a CPU compressed-tensors/BF16-seam oracle:

| Endpoint | Cosine | Relative L2 | Max abs |
|---|---:|---:|---:|
| combined gate/up | 0.999999999507 | 3.138e-5 | 9.766e-4 |
| SwiGLU | 0.999999999996 | 2.995e-6 | 7.629e-6 |
| final output | 0.999999999610 | 2.794e-5 | 6.104e-5 |

All values are finite and every endpoint passes the layer gate.

## Performance

```text
median        179.8626 us
mean          180.5161 us
CV              1.597%  (one 193.07-us outlier; remaining samples ~179.7-180.1)
boundary      combined gate/up W4 + BF16 SwiGLU + down W4 through stream sync
```

Remote raw receipt:

```text
benchmarks/qwen38_4090/results/native_qwen35_mlp_layer0.json
SHA-256 1299870c941430a623396d95460ab123a99f23dbbee912afdada8e3a7f539f35
```

Probe binary:

```text
/root/apxinf-target-sm89/release/examples/qwen35_mlp_layer_probe
Build ID fbb3c8839dd9552fff96e7c446bdbc4d7974bd1f
SHA-256 95db685d6a4452607989054a83abf4caed8db91fb4b783f2b1c16d7bfd57ee20
```

## Boundary

This closes the MLP sublayer but does not include its input RMSNorm, residual,
or a preceding attention/GDN sublayer. The next useful composition is a complete
GDN decoder layer and then the repeating `GDN,GDN,GDN,full-attention` unit. It
must time cross-layer weight streaming rather than sum isolated layer medians.
