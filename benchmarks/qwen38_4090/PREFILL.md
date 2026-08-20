# Qwen3.8 M<=8 W4A16 prefill projection on SM89

The resident service proves that decode beats vLLM while serial prompt
processing is about 50x slower. This record starts the M>1 path at its largest
repeated primitive: checkpoint W4A16 projection.

## Contract

```text
GPU             RTX 4090 / SM89
checkpoint      layer-0 MLP gate_proj
logical weight  [17408,5120]
quantization    compressed-tensors W4A16 group-32 asymmetric
activation      BF16 [M,5120], M in {2,4,8}
output          BF16 [M,17408]
baseline        M calls to accepted staged M=1 GEMV
candidate       one M-token kernel, M<=8
correctness     every BF16 output bit equal; input immutable
screen          five alternating AB/BA pairs
```

One CTA owns eight output rows and one warp owns each row. The candidate keeps
eight FP32 accumulators per lane, streams each packed weight/scale/zero once,
and applies it to up to eight token activations. M*K activations stay in
global/L2 because the full tile does not fit shared memory.

Binary resource use:

```text
registers  45/thread
shared     0
stack      0
local      0
spill      0
```

## Correctness

M=2,4,8 all produce zero different BF16 values versus the serial M=1 path;
max absolute difference is zero and the activation tensor remains bit-exact.

## Performance

### Hot-L2 repeated-weight proxy

| M | Serial M1 tile | M-token candidate | Speedup | Wins |
|---:|---:|---:|---:|---:|
| 2 | 60.093 us | 63.054 us | 0.952x | candidate loses |
| 4 | 118.507 us | 103.147 us | 1.149x | 5/5 |
| 8 | 234.867 us | 182.304 us | 1.289x | 5/5 |

M=2 is rejected for the hot repeated-weight cell. It may not replace M1 when
the same projection remains in L2.

### Cold-HBM cross-model proxy

A 128 MiB stream memset and synchronize evicts L2 outside timing. Serial M1 is
evicted before every token, matching the current whole-model-per-token prompt
path; the M-token candidate is evicted once before the tile.

| M | Serial cold tile | Candidate cold tile | Speedup |
|---:|---:|---:|---:|
| 2 | 169.450 us | 113.450 us | 1.491x |
| 4 | 339.101 us | 141.780 us | 2.392x |
| 8 | 678.060 us | 220.641 us | 3.079x |

The M8 candidate recovers only part of the ideal 8x weight reuse because it
still performs the same dequantization/FMA work and carries eight accumulator
chains. It is nevertheless large enough to justify layer integration.

## Real M8 MLP layer

The first joined layer boundary reuses the M8 W4 primitive for both sides of
the real layer-0 MLP:

```text
M8 hidden [8,5120]
  -> packed gate/up W4 [34816,5120]
  -> eight existing BF16 SiLU*Mul row views
  -> down W4 [5120,17408]
  -> BF16 output [8,5120]
```

Gate/up, post-SwiGLU, and final output are bitwise identical to eight serial
M1 executions. Five alternating AB/BA pairs all win:

| Boundary | Serial M1 tile | M8 candidate | Speedup | Wins | Candidate throughput |
|---|---:|---:|---:|---:|---:|
| Real layer-0 MLP | 1438.779 us | 609.620 us | 2.360x | 5/5 | 13,122.9 tok/s |

This is a real-checkpoint layer boundary but excludes input RMSNorm, residual,
and the surrounding attention/GDN module.

## Stateful M8 GDN scan

The causal part of a Qwen3.8 GDN layer cannot treat prompt tokens as an
ordinary batch: the depthwise-convolution and recurrent states must advance in
token order. The M8 candidate therefore composes three state-aware kernels:

```text
fused conv4 + prepare scan
  -> recurrent scan
  -> gated RMSNorm over 8 x 48 heads
```

It uses the real layer-1 `conv1d`, `A_log`, `dt_bias`, and gated-norm weights.
The oracle is eight calls to the accepted production M1 kernels from identical
initial states. Every BF16 and FP32 endpoint is bitwise identical: `a`, `b`,
Q/K/V, `g`, `beta`, every core and norm output, the final convolution state,
and the final recurrent state all have zero differing values.

| Boundary | Serial M1 tile | M8 candidate | Speedup | Wins | Candidate throughput |
|---|---:|---:|---:|---:|---:|
| Stateful GDN scan | 121.000 us | 87.280 us | 1.383x | 5/5 | 91,659.0 tok/s |

Binary resources for `(conv4_prepare_m8, recurrent_m8, gated_rmsnorm_m8)` are
respectively `(29, 40, 15)` registers/thread, `(0, 1056, 32)` bytes shared
memory, and zero stack/local memory. This is resource-safe on SM89; the gain is
primarily launch reduction while preserving the true sequential state
transition.

## Promotion boundary

The M8 path has now crossed the complete layer, hybrid-unit, 64-layer, and
service boundaries.

### Complete GDN layer and decoder block

The real layer-1 GDN first keeps the accepted M1 cuBLAS accumulation order for
the small 96-output `a/b` projection. A direct M8 BF16 GEMM arm changed 350
BF16 projection values and was rejected for the bit-exact path. All large W4
projections and the stateful scan remain batched.

| Boundary | Serial M1 tile | M8 candidate | Speedup | Wins | Exact endpoints |
|---|---:|---:|---:|---:|---:|
| Complete stateful GDN layer | 539.457 us | 373.878 us | 1.438x | 5/5 | all projections, outputs, conv/recurrent state |
| GDN decoder block through next norm | 2673.781 us | 1071.115 us | 2.497x | 5/5 | norms, residuals, GDN, MLP, next normalized |

The decoder-block boundary contains input offset RMSNorm, complete GDN,
residual/post-attention norm, packed gate/up + SwiGLU + down MLP, the second
BF16 residual seam, and the following layer's input norm.

### Causal full-attention layer

The layer-3 candidate batches q/k/v and output W4 projections, Q/K norm plus
partial RoPE, KV append, and output gate. The eight causal attention rows keep
their individual positions and use the accepted split16 path for KV >= 256.
Future rows may already exist in the cache, but each attention launch exposes
only `position + 1`, preserving causality. Every projection, prepared Q/K/V,
attention result, complete KV cache, gate, and output is BF16 bitwise exact.

| Start position | Serial M1 tile | M8 candidate | Speedup | Wins |
|---:|---:|---:|---:|---:|
| 1K | 752.155 us | 475.837 us | 1.583x | 5/5 |
| 8K | 1380.011 us | 788.795 us | 1.750x | 5/5 |
| 32K-8 | 2864.212 us | 2435.304 us | 1.177x | 5/5 |

### Canonical hybrid and 64-layer stack

`HybridUnit` now owns the M8 workspaces, position vector, strict eight-token
entry, state/KV lifetime, and final-row publication to the M1 decode workspace.
The candidate is not a probe-only composition.

| Boundary | KV cell | Serial M1 tile | M8 candidate | Speedup | Wins | Final differences |
|---|---:|---:|---:|---:|---:|---:|
| Layers 0..3 | 1K | 10.468 ms | 4.466 ms | 2.340x | 5/5 | 0 |
| Layers 0..3 | 8K | 11.043 ms | 4.899 ms | 2.255x | 5/5 | 0 |
| Layers 0..3 | 32K | 12.523 ms | 6.543 ms | 1.914x | 5/5 | 0 |
| All 64 layers | 1K | 159.637 ms | 64.552 ms | 2.474x | 5/5 | 0 |

For every row, both the final residual and the final/next normalized hidden
state are bitwise identical. The 64-layer M8 model-body throughput is 123.93
token/s at the 1K cell, up from 50.11 token/s.

### Service admission

The resident server now tiles every full group of eight prompt tokens and uses
the accepted M1 path only for the tail. Both a 53-token mixed tile/tail prompt
and an exact 56-token prompt pass; the latter validates final-row publication
before the first LM-head call.

| HTTP cell | Serial TTFT | M8 TTFT | Speedup | Effective prefill | Functional |
|---|---:|---:|---:|---:|---:|
| 1,064 prompt / 128 output | 21.3588 s | 8.9065 s median | 2.398x | 119.46 tok/s | 5/5 |
| 8,232 prompt / 128 output | 167.9669 s | 71.9498 s | 2.334x | 114.41 tok/s | 1/1 |

The 1K M8 TTFT CV is 9.30% because one of five samples is a 10.795 s outlier;
the other four lie in 8.781..8.928 s. Decode remains independent: 1K TPOT is
23.889 ms and throughput is 41.86 token/s.

```text
decision           M8 canonical prompt path admitted to the resident service
model default      ON for every full eight-token prompt tile
rollback           accepted staged M1 GEMV
next boundary      Marlin-class Tensor Core W4A16 for larger prefill tiles
not proven         vLLM-class TTFT, M>8, formal 32K service, multimodal
```

The candidate must not be wired as a generic M=2 hot selector. Model
integration must choose it only when token tiling actually prevents repeated
weight streaming across the full graph.

## Evidence

| Artifact | SHA-256 |
|---|---|
| `native_qwen35_w4a16_prefill_m2.json` | `c0d0b9f5fef619e472da5d2733a241a93540c520ba3c3c50df97187cb5f0b10b` |
| `native_qwen35_w4a16_prefill_m4.json` | `917bf5b3ed6384e8af650bdc97d2d624c44055a7d0f5e922cdf80ec149c19faf` |
| `native_qwen35_w4a16_prefill_m8.json` | `3eeed8c0bbdc3ec5a64cacd7d3e46e211b0a023bdbb226cac529fe56f09ae686` |
| `native_qwen35_mlp_prefill_m8.json` | `4f352e9d3c580c20a63697bfca2bba84b34365ddc5c62a4682085cce46467547` |
| `native_qwen35_gdn_scan_prefill_m8.json` | `246aeeea712dc317269482d4417edbadb1f4818568e8fb90826b47ef57e12a63` |
| `native_qwen35_gdn_prefill_m8.json` | `10a9d1100ce65a45953627b2bbdf12f9577267ea010f370e27998932cd7c45e6` |
| `native_qwen35_gdn_block_prefill_m8.json` | `1b12f60f24db8bd7c4b6c545d963d1894628d9c0439ad1d0e6d0858d970d57da` |
| `native_qwen35_attention_prefill_m8_1k_split16.json` | `3ca99398469c4ea12cbd6afa478a47e0d5f0afbfb7ece7d1016d9ffec918b33d` |
| `native_qwen35_attention_prefill_m8_8k_split16.json` | `e1cbf5a478d77fe04c5daa2d05a2369978b054b672270290fb697833e0aea77d` |
| `native_qwen35_attention_prefill_m8_32k_split16.json` | `d266ae30a64440cc71818a38e41154b9eeb44f18ed9508824c19ce171dacb046` |
| `native_qwen35_hybrid_prefill_m8_1k.json` | `190b816b9c919f2242eed625645e1d0b4a509871753fcb58669d9dca4c2d408e` |
| `native_qwen35_hybrid_prefill_m8_8k.json` | `bd4383579de0f05f7a82717d4458d36a1c167ea9d9aedb8a3eac40b322666ad1` |
| `native_qwen35_hybrid_prefill_m8_32k.json` | `67c74ccf1f1bca83af4de988c143f49754047ad12a84a336656f02931debb855` |
| `native_qwen35_64layer_prefill_m8_1k.json` | `36060a0f3534307e8c3afd2d7fc23475a87cf7c90ef478f41a4750c69a6c101d` |

Release example build ID `c3919800a2d985ccb9e06fa356d156b2468fcd79`,
SHA-256 `0f83a8b452796ae6219da4f35d9158f5a22d754bfc392534f1ad53dac03e5b8d`.

The GDN scan probe build ID is
`kb1-4e9f6bf88b93b7c8d954d65e2aca2027`; its release binary SHA-256 is
`c0e23279e39e29c2f142682567fad59f362a8e7591bdf148a2b105f7b2f84ff0`.
