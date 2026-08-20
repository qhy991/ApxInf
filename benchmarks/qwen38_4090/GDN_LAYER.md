# Native Qwen3.8 complete GDN layer on SM89

This record closes one real layer-0 linear-attention/GDN module from normalized
hidden input to the `[1,5120]` sublayer output. It is a layer/module result, not
a 64-layer model or serving result.

## Frozen boundary

```text
BF16 hidden [1,5120]
  -> checkpoint W4 in_proj_qkv [10240,5120]
  -> checkpoint W4 in_proj_z [6144,5120]
  -> BF16 in_proj_a and in_proj_b [48,5120]
  -> conv4 state update + SiLU
  -> 16 -> 48 Q/K head mapping and 48-head V
  -> FP32 g and beta
  -> recurrent GDN core and FP32 recurrent-state mutation
  -> per-head gated RMSNorm
  -> output projection [5120,6144]
  -> BF16 layer output [1,5120]
```

All weights are real tensors from layer 0 of model revision
`63768c10df38c0395e12ef49edac1bd539eaeeea`. W4 projections consume the packed
checkpoint directly. The BF16 baseline out-proj consumes the HF `[out,in]`
layout through a cuBLAS transpose view, without a physical transpose copy.

Named endpoints include QKV/Z/a/b projections, conv state and output, Q/K/V,
g/beta, recurrent state, core output, gated norm, and final layer output.

## BF16 baseline

All 15 endpoints passed the CPU compressed-tensors/BF16-seam oracle. Final
layer output:

```text
cosine       0.9999999987781727
relative L2  4.943547003536391e-5
max abs      1.9073486328125e-6
```

No-profiler layer timing:

```text
median       126.6465 us
mean         126.6638 us
CV           0.0926%
```

Nsight Systems single-step kernel sum was 126.144 us, closing the timing
boundary. Kernel attribution:

| Boundary | GPU time | Layer share |
|---|---:|---:|
| BF16 out-proj | 70.944 us | 56.2% |
| QKV + Z W4 | 32.160 us | 25.5% |
| recurrent core | 10.176 us | 8.1% |
| a + b BF16 GEMV | 8.672 us | 6.9% |
| conv + prepare + gated norm | 4.192 us | 3.3% |

This profile selected the out-proj representation as the only boundary with a
large enough end-to-end ceiling.

## Rejected candidates

The checkpoint deliberately leaves the GDN out-proj BF16. Two explicit lossy
counterfactuals failed the predeclared layer-output gate
`cosine >= 0.999 && relative-L2 <= 0.02`:

| Candidate | Cosine | Relative L2 | Decision |
|---|---:|---:|---|
| group-32 asymmetric W4A16 | 0.992481 | 0.123186 | reject before timing |
| per-channel W8A8 | 0.996364 | 0.086032 | reject before timing |

W8A8 failed because it added single-row activation quantization. The gate was
not relaxed after observing either result.

## Accepted opt-in: W8A16 plus graph rewrite

Weight-only per-output-channel W8A16 kept activation BF16. It passed correctness:

```text
cosine       0.9998731093090095
relative L2  0.015931741546564054
max abs      0.0000762939453125
```

The first W8A16 layer measured 105.348 us and therefore failed the separately
predeclared `<=100 us` performance gate. Systems measured its out-proj kernel at
27.008 us; further isolated kernel tuning had insufficient low-risk ceiling.

The admitted graph rewrite then made two orthogonal changes while preserving
the W8A16 representation:

1. Physically pack `in_proj_a` and `in_proj_b` into one `[96,5120]` BF16 GEMV.
2. Fuse conv4 state update, SiLU, Q/K repeat, V write, and g/beta preparation.
   This deletes one GEMV launch, one prepare launch, and the 20 KiB conv-output
   materialization.

The fused kernel uses 22 registers with zero spill/local/shared. W8A16 uses 31
registers, 12,320 bytes dynamic shared, and zero spill/local.

One detailed no-profiler run:

```text
median       98.46915 us
mean         98.47097 us
CV           0.1156%
```

Five alternating BF16/candidate process pairs:

```text
baseline median        126.7789 us
candidate median        98.4772 us
median speedup            1.2873940808x
candidate wins            5/5
correctness pass          true
decision                  promote-layer-opt-in
model promoted            false
```

The candidate is opt-in because its W8A16 out-proj is not the checkpoint's
declared precision and complete model token/quality evidence does not yet exist.
The BF16 layer remains the accepted semantic baseline and rollback.

## Evidence

Remote files under `benchmarks/qwen38_4090/results/`:

| File | SHA-256 |
|---|---|
| `native_qwen35_gdn_layer0.json` | `401ef8eace5a9bdd7933e441af2f206cd2d14a27cc52fcebe41fac4da0029134` |
| `native_qwen35_gdn_layer0_w8a16_candidate.json` | `e42eece0a7c7e8b99fd0ce246a6ac5c300560da57395807157bff774245881bb` |
| `native_qwen35_gdn_layer0_w8a16_fused_candidate.json` | `3ef97825b4dabb8d53d71f49d36ed8fc4e8557ec673c7853237c53115bef48f6` |
| `native_qwen35_gdn_layer0_screen_ab.json` | `90ac205cd115cff7abf5f240c0783fe75f4d927ec6bd3df1b2d9d70b3d3c82d0` |
| `native_qwen35_gdn_layer0_nsys.nsys-rep` | `d6c6bad9edaa504b1234e7c846eb50226242889e3f1bb45a04f381b18be1c440` |
| `native_qwen35_gdn_layer0_w8a16_nsys.nsys-rep` | `5fb229dc59508866b21a41ad092fc1d1814385d3b4fc0b2e06e83034ef6904ff` |

Final probe binary:

```text
/root/apxinf-target-sm89/release/examples/qwen35_gdn_layer_probe
Build ID a053cc0e112d71e4121be2e4f5e6af5c4a7a7fa6
SHA-256 56e0c40768d60bcc5ca50e9a17d068f9b3008a9a8ad43384011deb4bc657e959
kernel build id kb1-4b3650d143930db74c17b6823f4ad8c1
```

The binary hash predates only Rust-level receipt/report generation; all named
CUDA kernels and the final candidate selector are contained in that build.

## Next promotion boundary

The GDN layer is now sufficient for model integration, but the model still
needs the common RMSNorm/residual/MLP path, all 16 full-attention layers, 64-layer
weight/state ownership, logits, tokenizer trajectory, and service scheduler.

Next work should implement one real MLP layer and one real full-attention decode
layer, then compose the four-layer repeating unit:

```text
GDN -> GDN -> GDN -> full attention
```

Only a complete repeated unit can reveal L2/HBM interaction between successive
layer weights and determine whether the 1.287x isolated GDN layer win survives.
