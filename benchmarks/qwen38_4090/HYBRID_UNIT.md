# Native Qwen3.8 four-layer hybrid unit on SM89

This record composes the first real repeating decoder unit:

```text
layer 0 GDN -> layer 1 GDN -> layer 2 GDN -> layer 3 full attention
```

Each layer includes input RMSNorm, mixer, residual, post-attention RMSNorm,
MLP, and the second residual. It is the first multi-layer ApxInf result for the
model, but it is not yet a 64-layer token trajectory or serving result.

## Frozen contract

```text
model                 cyankiwi/Qwen3.8-27B-AWQ-INT4
model revision        63768c10df38c0395e12ef49edac1bd539eaeeea
repository base       ea3a4eb1057a1eff127b8187d4d844f12c29fff9 + dirty source
GPU                    one RTX 4090 / SM89
layers                 0,1,2,3
schedule               GDN,GDN,GDN,full-attention
request                batch=1, new tokens=1, eager, one CUDA stream
hidden                 BF16 [1,5120]
state                  three independent conv/recurrent states
KV                     one BF16 K/V cache, 4 heads x KV x 256
timing boundary        four complete layers through stream synchronize
timing exclusions      weight load, state/cache/input reset, CPU oracle
screen                 five in-process alternating AB/BA pairs
```

The implementation uses one shared execution workspace across all four layers;
only persistent weights, three GDN states, and the layer-3 KV cache are owned
per layer. This is the same ownership model needed to scale from one unit to 16
units without allocating per-layer activation workspaces.

## Checkpoint dtype discovery

The unit load found a fact that the layer-0 probe could not establish:

```text
layer 0 GDN out_proj   BF16 checkpoint weight (ignored by quantizer)
layer 1 GDN out_proj   checkpoint W4A16 bundle
layer 2 GDN out_proj   checkpoint W4A16 bundle
```

Only layer 0 is listed in `quantization_config.ignore` for this projection.
The runtime therefore dispatches from the manifest instead of assuming one
GDN out-projection dtype for all 48 GDN layers.

The two measured arms are:

```text
native:
  layer0 BF16 GDN out_proj
  layer1/2 checkpoint W4 GDN out_proj
  incumbent layer3 attention

optimized:
  layer0 W8A16 GDN out_proj
  layer1/2 checkpoint W4 GDN out_proj
  split16 layer3 attention for KV bucket >=256
```

## Qwen offset RMSNorm

The authoritative Transformers implementation computes decoder RMSNorm as:

```text
BF16( normalize_f32(x) * (1 + checkpoint_weight_f32) )
```

The generic ApxInf/Llama RMSNorm multiplies by `weight` directly, so it cannot
serve this model. The unit adds two narrow caller-owned kernels:

```text
offset RMSNorm
BF16 residual add -> offset RMSNorm, with residual updated in place
```

The residual fusion explicitly rounds the residual sum to BF16 before using it
for RMSNorm, matching the materialized PyTorch seam rather than normalizing an
unrounded FP32 sum.

Real layer-0 input-norm correctness:

| Endpoint | Cosine | Relative L2 | Max abs |
|---|---:|---:|---:|
| direct offset RMSNorm | 0.999999987 | 1.613e-4 | 0.0078125 |
| fused residual + offset RMSNorm | 0.999999974 | 2.282e-4 | 0.0078125 |
| updated residual BF16 seam | bit-exact | 0 | 0 |

Binary resources:

| Kernel | Registers | Static shared | Dynamic shared | Spill/local |
|---|---:|---:|---:|---:|
| direct offset RMSNorm | 16 | 32 B | 0 | 0 |
| residual + offset RMSNorm | 16 | 32 B | 20,480 B | 0 |

## Four-layer correctness

Every arm begins from the same deterministic hidden state, deterministic BF16
KV prefix, zero convolution states, and zero FP32 recurrent states. Native runs
first, all mutable inputs are reset, and optimized runs second.

The final layer-3 residual is compared as the complete four-layer endpoint.
The preregistered bring-up gate is cosine at least `0.99` and relative L2 at
most `0.15`, reflecting the explicitly lossy layer-0 W8A16 arm. Observed error
is much smaller:

| KV | Cosine | Relative L2 | Max abs | Result |
|---:|---:|---:|---:|---|
| 256 | 0.999992405 | 0.007050 | 0.5 | pass |
| 1K | 0.999992224 | 0.007168 | recorded pass |
| 8K | 0.999991663 | 0.007337 | recorded pass |
| 32K | 0.999991470 | 0.007413 | recorded pass |

The small monotonic change with KV is consistent with the split-attention
reduction tree; the dominant difference remains the admitted layer-0 W8A16
projection.

## No-profiler paired performance

| KV | Native median | Optimized median | Median paired speedup | Wins |
|---:|---:|---:|---:|---:|
| 256 | 1,298.733 us | 1,266.553 us | 1.0256x | 5/5 |
| 1K | 1,354.043 us | 1,270.603 us | 1.0649x | 4/5 |
| 8K | 1,868.540 us | 1,332.952 us | 1.4032x | 5/5 |
| 32K | 3,597.850 us | 1,518.062 us | 2.3704x | 5/5 |

This proves that the attention win survives a real joined graph. It also shows
why isolated-kernel ratios cannot be projected directly: at KV=256 the
attention and one eligible GDN out-projection save only about 2.6% of the
complete four-layer boundary.

## Systems attribution at 32K

### Native

| Range/node | GPU projection |
|---|---:|
| complete four-layer unit | 3,648.258 us |
| layer 0 | 409.185 us |
| layer 1 | 311.969 us |
| layer 2 | 312.544 us |
| layer 3 | 2,612.545 us |
| incumbent attention tail | 2,335.905 us |
| layer-0 BF16 GDN out-proj | 68.192 us |

The one-CTA-per-Q-head attention kernel alone is 2,311.425 us and 64.8% of all
summed unit kernel time.

### Optimized

| Range/node | GPU projection |
|---|---:|
| complete four-layer unit | 1,533.696 us |
| layer 0 | 339.872 us |
| layer 1 | 312.320 us |
| layer 2 | 312.928 us |
| layer 3 | 566.592 us |
| split16 attention tail | 288.320 us |
| layer-0 W8A16 GDN out-proj | 36.799 us |

The critical interval moves exactly where predicted. After the rewrite, the
20 checkpoint W4 projection launches total 1,008.160 us and account for 67.7%
of summed optimized kernel time. Split attention is now 260.960 us plus a
2.688-us merge. Eight residual/offset-norm kernels total 92.065 us.

The bottleneck has therefore migrated from long-context attention to the
projection graph, especially the four MLPs. The next optimization should work
at that joined projection/activation boundary or at the full 64-layer runtime;
further tuning the 2.7-us merge has no material ceiling.

## Evidence

| Artifact | SHA-256 |
|---|---|
| `native_qwen35_hybrid_unit_256.json` | `7bff003dada32b0b41d0082ac6a49eaffb8d067af1f2223fa85be6a347ae3dd0` |
| `native_qwen35_hybrid_unit_1024.json` | `48639bea91bf7d75b0700794712908117a984d5d2f867763de9d922ac7bafe0c` |
| `native_qwen35_hybrid_unit_8192.json` | `d895ad35b7163a64bf0e05c5f7cd8245aa78555c3657cbc559a372cc4bd4e83d` |
| `native_qwen35_hybrid_unit_32768.json` | `5a1bdf1bc33ed54a8f39d5bc8757f75e92f4ec383e963c95895a6fadabfe94b6` |
| native 32K Systems | `871d4ed650a6440a0419516175e63e3831d1e25123cc5adbdefbafae2fbfd6ef` |
| optimized 32K Systems | `c242e27329ed5a85949c84431d6e409bb9ea47bc1e8d65dc7d7666a6b21495bc` |

Latest release example:

```text
ELF build ID  3c737726fe50f1c3e717225f88a4ee093021b311
SHA-256      4149752d943d82aa5f29717cae21e39c6bd0d46821b833c554aadd3cb449fdb7
```

Reproduction:

```bash
APXINF_UNIT_KV_LEN=32768 \
/root/apxinf-target-sm89/release/examples/qwen35_hybrid_unit_probe \
  /mnt/user_dir/hanjinchen/models/cyankiwi/Qwen3.8-27B-AWQ-INT4
```

Raw JSON is mirrored under the ignored local `results/` directory. Systems
reports remain on the persistent remote volume.

## Decision and next boundary

```text
decision             promote as four-layer opt-in; model decision continues
covered              layer 0..3, batch=1, M=1, eager, BF16 KV, SM89
default state        native model path still unavailable / optimized OFF
rollback             checkpoint-native out-proj + incumbent attention
unsupported          M>1 prefill, FP8 KV, CUDA Graph, batch decode, other GPUs
model E2E            not yet tested
```

The next boundary is all 16 repeating units with one shared activation
workspace, 48 recurrent-state owners, 16 KV owners, final offset RMSNorm, tied
LM head, and a complete token/logit trajectory. Memory admission must use FP8
KV or an equivalently frozen cache policy before comparison with vLLM.
