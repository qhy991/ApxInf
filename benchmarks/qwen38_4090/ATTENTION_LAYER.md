# Native Qwen3.8 full-attention layer on SM89

This record closes the real layer-3 one-token full-attention module and its
first long-context scheduling optimization. It is layer/module evidence, not a
64-layer token-trajectory or serving promotion.

## Frozen contract

```text
model                 cyankiwi/Qwen3.8-27B-AWQ-INT4
model revision        63768c10df38c0395e12ef49edac1bd539eaeeea
repository base       ea3a4eb1057a1eff127b8187d4d844f12c29fff9 + dirty source
GPU                    one RTX 4090 / SM89
layer                  model.language_model.layers.3.self_attn
request                batch=1, new tokens=1, eager, one CUDA stream
hidden                 BF16 [1,5120]
Q / KV heads           24 / 4, GQA ratio 6
head dimension         256
partial RoPE           first 64 dimensions, theta=10,000,000
KV                     BF16 [4,32768,256] K and V, head-major
projections            compressed-tensors W4A16, group 32, asymmetric
positions tested       1, 64, 128, 256, 512, 1K, 8K, 32K
official timing        no-profiler module wall time through stream synchronize
screen                 five alternating AB/BA process pairs
```

The timed module is:

```text
hidden [1,5120]
  -> q/k/v W4 projections
  -> offset Q/K RMSNorm and text-position partial RoPE
  -> exact K/V append
  -> BF16 GQA decode attention
  -> sigmoid output gate
  -> o_proj W4
  -> output [1,5120]
```

Input RMSNorm, residual, post-attention RMSNorm, and MLP are outside this
module. They belong to the next complete decoder-layer/four-layer-unit proof.

## Named correctness endpoints

The probe uses the real layer-3 checkpoint bundles and compares ten named
producer/consumer boundaries against a scalar CPU compressed-tensors/BF16
oracle:

```text
q_projection, k_projection, v_projection
prepared_query, prepared_key, prepared_value, output_gate
attention_output, gated_output, output_projection
```

Projection/preparation endpoints require cosine at least `0.999` and relative
L2 at most `0.02`. Attention and downstream endpoints require cosine at least
`0.995` and relative L2 at most `0.05`. These thresholds were frozen before
the split-CTA timing screen.

The KV side-effect contract is stronger: the appended slot must be bit-exact to
the actual GPU K/V producer, while sentinel locations before, inside, and after
the valid interval remain bit-exact. Each cell checks 2,048 appended BF16
values and 120 or 160 sentinel values. The hidden input must remain bit-exact.

At 32K, the admitted split16 candidate reports:

| Endpoint | Cosine | Relative L2 | Max abs |
|---|---:|---:|---:|
| attention output | 0.999999946 | 3.291e-4 | 4.883e-4 |
| final output projection | 0.999999724 | 7.426e-4 | 9.766e-4 |
| KV appended slot | bit-exact | 0 | 0 |
| KV sentinels | bit-exact | 0 | 0 |
| hidden input | bit-exact | 0 | 0 |

All declared cells pass. The split changes the FP32 reduction tree, so it is
not claimed byte-identical to the incumbent attention output.

## Incumbent bottleneck

The original head-dim-256 kernel launches one 512-thread CTA per Q head. That
is only 24 CTAs on a 128-SM RTX 4090, so a long sequence is processed by a
small fraction of the GPU while each CTA loops over the entire KV interval.

At 32K, Nsight Systems attributes the incumbent module GPU envelope as:

| Node | GPU time | Share |
|---|---:|---:|
| one-CTA-per-head attention | 2,309.577 us | 96.3% |
| four W4 projections | 76.480 us | 3.2% |
| prepare + two appends + gate | 12.865 us | 0.5% |
| complete module GPU projection | 2,412.554 us | 100% |

The problem is exposed parallelism, not projection math or launch count.

## Split-CTA rewrite

The explicit candidate partitions each Q head's valid sequence across 16 CTAs:

```text
24 Q heads x 16 sequence splits = 384 stage CTAs
  -> per-split FP32 online-softmax {max, sum, accumulator[256]}
  -> 24 small FP32 merge CTAs
  -> BF16 attention output
```

The workspace is caller-owned and allocated once:

```text
partial max            1,536 B
partial sum            1,536 B
partial accumulator  393,216 B
total                 396,288 B (387 KiB)
```

The incumbent remains available. The candidate has no automatic fallback and
fails closed unless the device is SM89, tensors are BF16 with the exact
24/4/256 Qwen shape, split count is one of 2/4/8/16, buffers are on the same
device, and sequence/cache bounds are valid.

The layer-screened opt-in selector is:

```text
KV bucket < 256   -> incumbent
KV bucket >= 256  -> split16
```

It remains model-default OFF until complete-token E2E evidence exists.

## Binary resources

`cuobjdump --dump-resource-usage` on the SM89 release binary reports:

| Kernel | Registers | Static shared | Stack/local | Spill |
|---|---:|---:|---:|---:|
| incumbent head-dim-256 attention | 54 | 17,536 B | 0 | 0 |
| split16 stage | 40 | 9,280 B | 0 | 0 |
| split16 FP32 merge | 38 | 80 B | 0 | 0 |

The stage exposes 384 CTAs without creating a residency cliff; the merge is a
small second launch rather than a device-wide in-kernel barrier.

## No-profiler performance

### Main decode cells

| KV | Incumbent module | split16 module | Speedup | Latency reduction |
|---:|---:|---:|---:|---:|
| 1K | 124.8065 us | 82.9212 us | 1.505x | 33.56% |
| 8K | 677.5905 us | 167.9660 us | 4.034x | 75.21% |
| 32K | 2,403.9992 us | 353.3378 us | 6.804x | 85.30% |

### 32K split portfolio

| Arm | Median module time | Correctness |
|---|---:|---|
| split2 | 1,770.9862 us | pass |
| split4 | 950.4573 us | pass |
| split8 | 547.1900 us | pass |
| split16 | 353.3378 us | pass |

The monotonic improvement through 16 splits confirms that the incumbent was
starved for CTA-level parallelism.

### Selector boundary

| KV | Incumbent | split16 | Result |
|---:|---:|---:|---|
| 1 | 67.2180 us | 74.9328 us | split16 loses 10.3% |
| 64 | 69.6913 us | 79.6647 us | split16 loses 12.5% |
| 128 | 73.4747 us | 76.5930 us | split16 loses 4.1% |
| 256 | 85.0828 us | 77.8997 us | split16 wins 1.093x, 5/5 pairs |
| 512 | 98.6545 us | 79.4213 us | split16 wins 1.242x, 5/5 pairs |

The auto-selector smoke proves the actual selected path in JSON: incumbent at
128 and split16 at 256/512/32768, with every correctness endpoint passing.

### Alternating 32K screen

Five process-level AB/BA pairs preserve raw per-block samples and order:

```text
incumbent median       2,404.1670 us
split16 median           353.1917 us
median paired speedup       6.8031x
paired wins                  5 / 5
all correctness              pass
```

This promotes split16 as an SM89 layer opt-in for KV buckets at least 256. It
does not promote a model or service.

## Profile attribution after the rewrite

At 32K, candidate Systems attribution is:

| Node | GPU time | Share |
|---|---:|---:|
| split16 stage, 384 CTAs | 258.625 us | 73.9% |
| four W4 projections | 76.095 us | 21.8% |
| FP32 merge | 2.721 us | 0.8% |
| prepare + append + gate | 12.385 us | 3.5% |
| complete module GPU projection | 390.496 us | 100% |

The changed critical interval shrank from 2,309.577 to 262.049 us including
merge. The bottleneck has migrated: attention remains first, but W4 projections
now account for about one fifth of the module rather than three percent.

Nsight Compute was invoked only for the split stage after Systems identified
it. The driver rejected counter access with `ERR_NVGPUCTRPERM`; no report was
created and no utilization percentages are invented. The raw failure receipt
is retained as `native_qwen35_attention_layer3_split16_32k_ncu_gap.txt`.

## SubCUDA transfer decision

Primary classification: **source/runtime graph**. CUDA implements a guarded
sequence-ownership and reduction rewrite; PTX is not the performance carrier.

- Accepted analog: `fi-single-decode-cuda-translation-sm100` shows that
  production-shape long-context decode can benefit strongly from guarded CUDA,
  but it is B200 operator evidence and not copied mechanically.
- Rejected counterexample: `omoe-qwen35-tp2-d045-fmha-multicta-oracle` shows
  that changing split count changes the BF16 reduction tree and that GMEM
  partials can lose when the incumbent already exposes roughly one CTA per SM.
- Migration boundary: `fi-batch-decode-cuda-translation-sm100` shows that a
  single-decode mechanism and geometry must not be generalized to batch decode.

This case differs materially from the rejected multi-CTA oracle: the SM89
incumbent exposes only 24 CTAs for 128 SMs, the user accepts a declared numeric
tolerance rather than byte identity, and the exact joined layer boundary wins
by a large no-profiler margin after paying the GMEM merge cost.

## Reproduction and evidence

```bash
APXINF_KV_LENS=1024,8192,32768 \
APXINF_ATTN_IMPL=incumbent \
/root/apxinf-target-sm89/release/examples/qwen35_attention_layer_probe \
  /mnt/user_dir/hanjinchen/models/cyankiwi/Qwen3.8-27B-AWQ-INT4

APXINF_KV_LENS=1024,8192,32768 \
APXINF_ATTN_IMPL=split16 \
/root/apxinf-target-sm89/release/examples/qwen35_attention_layer_probe \
  /mnt/user_dir/hanjinchen/models/cyankiwi/Qwen3.8-27B-AWQ-INT4

APXINF_KV_LENS=128,256,512,32768 \
APXINF_ATTN_IMPL=auto \
/root/apxinf-target-sm89/release/examples/qwen35_attention_layer_probe \
  /mnt/user_dir/hanjinchen/models/cyankiwi/Qwen3.8-27B-AWQ-INT4
```

| Artifact | SHA-256 |
|---|---|
| `native_qwen35_attention_layer3.json` | `92743e348c802f476fc67f83909d95bc9e31669846036232c47a3c7563a8bfad` |
| `native_qwen35_attention_layer3_split16.json` | `a8adc9432d01b43113b136f9030d0819b6595b4a10f9b0bc5dc639e7a9473ee6` |
| `native_qwen35_attention_layer3_split_portfolio.json` | `a83a6c3b67ba6320ce0e3841061763e48f3c332a1f416b26315dc27caa2f8e19` |
| `native_qwen35_attention_layer3_split16_256_screen_ab.json` | `189e6894967288785c8b2fa66ba40fbc8bdb34dac4ccb7563f97ab28ee2f5076` |
| `native_qwen35_attention_layer3_split16_512_screen_ab.json` | `863428e512fb82224ec80009189017a2f3a50ed3b71b7d6b45bd046e64cca996` |
| `native_qwen35_attention_layer3_split16_screen_ab.json` | `cd1fa5146db1854a5b139e94082619cd8a145eb4e491951cf5cbb2926318ab51` |
| incumbent 32K Systems | `58bf19d5fa83c1ab59db2245a2e897193c42741ff8d62a56296ab71143dc9028` |
| split16 32K Systems | `67e9b45aa9a2270744c21134e2e0187728b1fc3dc7bdc9cf7e651383a4077c74` |
| NCU permission-gap receipt | `4856a337bc93e5dbde7a8ca6c3e9f50d4f0f79689e6ec90583a8246fe379dd57` |

Latest release example:

```text
ELF build ID  42f440a0a5499d23907979b7c7edadfc305494ac
SHA-256      627b5594b844df418eb76649a5443ce1739f28f964e7f793cf3fec5d04a6a634
```

Raw JSON is mirrored under the ignored local `results/` directory. Systems
reports remain on the remote persistent volume.

## Decision and next boundary

```text
decision             promote as SM89 layer opt-in; model decision continues
default state        incumbent / model optimized path still OFF
rollback             incumbent one-CTA-per-Q-head kernel
covered              batch=1 M=1 eager BF16 KV, exact Qwen 24/4/256 shape
unsupported          M>1 prefill, batch decode, CUDA Graph replay, other GPUs
model E2E            not yet tested
```

The next bounded proof composes one repeating hybrid unit:

```text
GDN layer -> GDN layer -> GDN layer -> full-attention layer
```

with real RMSNorm, residual, MLP, recurrent state, and KV ownership. Only that
larger boundary can determine whether the GDN and attention layer wins survive
composition on the path toward all 64 layers and complete tokens.
