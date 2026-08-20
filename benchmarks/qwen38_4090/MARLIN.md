# SM89 Marlin-class W4A16 prefill branch

This record begins the larger-tile Tensor Core branch after the canonical M8
service reduced TTFT by 2.33..2.40x but remained about 20x behind vLLM. It does
not promote Marlin into the model or service yet.

## Frozen contract

```text
GPU                 RTX 4090 / SM89
checkpoint weight   layer-0 MLP gate_proj [17408,5120]
quantization        asymmetric U4, group size 32, BF16 scale
activation/output   BF16
source authority    vLLM tag v0.27.1, commit 6e448d0ea9bf3d88d898b65449ca6dc2aec170ac
upstream boundary   repack + permuted scales/zero points + Marlin BF16/U4 GEMM
ApxInf carrier       raw-pointer C ABI, default off, SM89/shape fail-closed
correctness          raw C ABI bitwise equal to official Marlin output
performance         CUDA-event kernel interval; no server claim
```

The upstream source is the Apache-2.0 Marlin directory at
<https://github.com/vllm-project/vllm/tree/v0.27.1/csrc/libtorch_stable/quantization/marlin>.
The current vLLM quantization documentation identifies Marlin as supporting
AWQ on Ada GPUs:
<https://github.com/vllm-project/vllm/blob/v0.27.1/docs/features/quantization/README.md>.

## Selected SubCUDA evidence

```text
accepted mechanism   fi-single-decode-cuda-translation-sm100
                     maintainable CUDA may carry a PTX-discovered mechanism
counterexamples      omoe-qwen35-tp2-gdn-portfolio20
                     cache/schedule PTX edits did not clear the operator floor
                     omoe-qwen35-tp2-d005-state-cpasync-smem
                     staging can regress through shared memory and barriers
migration case       omoe-qwen35-tp2-r95-gdn-cuda-counterfactual
                     do not claim PTX exclusivity when CUDA recovers the win
```

Primary classification is **PTX/SASS-directed CUDA**. Marlin internally uses
inline PTX and Tensor Core MMA, but the ApxInf integration boundary is CUDA
source plus a raw C ABI, not authored standalone PTX.

## Official-kernel screen

The installed frozen vLLM 0.27.1 environment loads the real compressed-tensors
checkpoint, transposes and repacks the W4 matrix, permutes group scales and
packed zero points, then invokes its official `marlin_gemm` operator.

| M | Official Marlin median | Per token |
|---:|---:|---:|
| 8 | 75.776 us | 9.472 us |
| 16 | 76.800 us | 4.800 us |
| 32 | 93.152 us | 2.911 us |
| 64 | 132.064 us | 2.063 us |
| 128 | 204.800 us | 1.600 us |

The current ApxInf M8 scalar W4 kernel takes 182.304 us for the same logical
gate projection, or 22.788 us/token. At M64, official Marlin is about 11x
lower per token. This is material enough to justify the port; another M8-only
fusion is not.

## Numerical gate

For the exact deterministic M8 activation used by the ApxInf W4 probe, official
Marlin is compared with a checkpoint-derived BF16 dense-matmul oracle:

```text
cosine           1.000000119
relative L2      1.08347e-4
max absolute     0.001953125
mean absolute    1.73856e-7
different BF16  95 / 139264
finite           true
```

This is not bitwise equal to the scalar-FMA seam, so the eventual model gate
must use a declared numerical threshold plus complete token trajectories. The
original request permits non-BF16-exact INT4 exploration, but no model-level
accuracy conclusion is inferred from this operator result.

## Raw ApxInf C ABI

The PyTorch tensor/dispatcher boundary is removed. A raw-pointer adapter
instantiates the BF16/U4/BF16, group-block-2 template with the official M64
schedule `(threads=256, M blocks=4, N blocks=16, K blocks=4, stages=4)`.

Against the official vLLM operator on the same transformed tensors:

```text
outputs compared       1,114,112 BF16
different values       0
max absolute           0
relative L2            0
raw C ABI median        99.328 us
official Python median 132.064 us in the preceding screen
```

The raw carrier therefore preserves the exact Marlin kernel result and removes
the Python/dispatcher interval. Vendored sources retain their upstream
license/copyright headers; `kernels/marlin/README.md` records provenance.

## Runtime transformation boundary

ApxInf must retain the original compressed-tensors layout for its faster M1
decode path. Storing a second repacked copy of every weight would exceed 24 GiB.
The intended prompt-only graph is therefore:

```text
resident original W4
  -> reusable GPU transpose/repack workspace
  -> reusable scale/zero-point permutation workspace
  -> Marlin GEMM over a large prompt tile
  -> workspace reused by the next projection
```

The production candidate now includes a tiled I32 transpose, the official
repack core, a direct original-layout BF16 scale permutation, and a direct
packed zero-point permutation. Against the official transformed tensors:

```text
repacked weight    0 / 11,141,120 I32 different
permuted scales    0 / 2,785,280 BF16 different
packed zero points 0 / 348,160 I32 different
combined transform median 242.880 us
```

This replaces the earlier 439.296 us transpose/repack/scale prototype and also
includes the previously missing zero-point transform.

## ApxInf Rust operator admission

The release `qwen35_marlin_probe` compares the complete dynamic transform plus
raw M64 GEMM against eight accepted scalar M8 calls on the deterministic gate
input.

The first v1 rule (`rel-L2 <= 0.001` versus scalar M8) rejected the candidate:

```text
cosine       0.999997443
relative L2  0.002261256
max absolute 0.00390625
```

That rejection is retained. The v2 exploratory INT4 rule is `cosine >= 0.999`
and `rel-L2 <= 0.005`, grounded by the independent checkpoint-dense Marlin
result (`rel-L2 1.08347e-4`). Passing v2 does not replace the future model
trajectory gate.

| Arm | Median | Raw paired range | Wins |
|---|---:|---:|---:|
| Eight scalar M8 calls | 1485.320 us | 1449.460..1487.360 us | — |
| Dynamic transform + Marlin M64 | 281.148 us | 279.399..328.027 us | 5/5 |
| Marlin kernel only | 89.699 us | 89.219..90.249 us | — |

The admitted operator boundary is 5.156x faster and processes 227,638
token-rows/s for this projection. Two candidate samples at 325..328 us show an
order/cache effect; the other three are 279..281 us. Formal model timing must
retain AB/BA order rather than report only the fastest subgroup.

Binary evidence:

```text
kernel build ID  kb1-3f9407cde3f53f70a86a56b9288d5cf3
ELF build ID     27d87e974b96c1f43e446c9087b92f60ac571658
ELF SHA-256      c9a245a59527de724ccb6f5dfef3e3fb32b7ca148221acdc5e22bbe095833d1e
result SHA-256   e9810aeba8db5034457c7f5dfc944b180b91b456eb8ac9240c4e3cf33b862921
```

The M64 instantiation uses 255 registers/thread, 32 bytes stack, no reported
local memory, and dynamic shared memory selected from the SM89 opt-in limit.
That is a one-CTA-per-SM resource regime by design; any register cap is a new
candidate and requires its own SASS, correctness, and paired timing evidence.

## Decision

```text
decision           continue
operator mechanism accepted as material and numerically bounded
raw C ABI           Cargo/AOT, transform correctness, and Rust timing passed
model default       canonical M8 remains ON; Marlin remains OFF
rollback            M8 W4A16 + M1 tail
next boundary       M64 MLP, stateful GDN, causal attention, then 64 layers
promotion blocker   no model trajectory and no Marlin service TTFT data
```
