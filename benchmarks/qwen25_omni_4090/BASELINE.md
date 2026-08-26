# Native Qwen2.5-Omni RTX 4090 baseline

This file records the current accepted service, not the optimization diary.
Generated trials and profiler exports remain on the experiment host; the PR
keeps only the aggregate evidence listed below.

## Frozen contract

- Model: `Qwen/Qwen2.5-Omni-3B`
- Revision: `f75b40e3da2003cdd6e1829b1f420ca70797c34e`
- Hardware: one NVIDIA GeForce RTX 4090, SM89, 24 GiB
- Precision and scheduling: BF16, one request, greedy, non-streaming,
  `ignore_eos=true`
- Output scope: thinker text output; real PNG and WAV inputs are covered, but
  video and speech generation are not
- Maximum request: 32,768 combined prompt and output tokens
- Timing authority: service-emitted TTFT/TPOT plus client wall time, without a
  profiler attached

## Accepted artifact

- Binary SHA-256:
  `0a56c36754c9003bbd14dad75ec0041a34b03910c86579222f9bb1f540b59054`
- Immediate rollback SHA-256:
  `43bfa1993c6b507db44d05d8b7f6a8a02fe061b8bdd52d8b43851f223c2e6ad7`
- Service owner: runit plus `agent-gpu-broker`
- Desired state after validation: stopped

Build the accepted SM89 artifact with:

```bash
CARGO_TARGET_DIR=target/qwen25-omni-sm89 \
  benchmarks/qwen25_omni_4090/build_sm89.sh
```

The checked-in runit definition under `service/` is the launch and environment
authority. Every optional optimized path is explicit and fails closed when its
shape, architecture, build feature, or prerequisite does not match.

## Current implementation

The accepted implementation combines:

- a persistent image/audio processor with typed malformed-media recovery;
- exact causal chunked prefill and request-scoped FlashAttention-2 scheduling;
- packed QKV, cached TMRoPE positions, fused TMRoPE/KV publication, GPU token
  selection, and short-context decode graphs;
- an SM89-only long-decode split-CTA attention path and dedicated graph for
  positions 32,760 through 32,767;
- a graph-only M=1 packed Gate/Up projection and bit-exact fused SiLU/multiply;
- exact SM89 cuBLASLt tactics for the one-token WO and Down projections;
- BF16-exact residual-add/RMSNorm fusion in the short one-token decode graph;
- a 32-warp split-K attention geometry in the SM89 short decode graph;
- one packed-QKV bias/TMRoPE/KV-publish owner in the short decode graph;
- aligned pack8 global I/O for the exact H=2,048 residual/RMSNorm nodes;
- exact fused score scaling and numerator-cache softmax for short text and
  multimodal prefill;
- grouped variable-length FA2 for windowed vision attention and FA2 for the
  four full-attention vision blocks;
- one BF16-exact Q/K/V bias and Q/K 2-D RoPE owner in each vision block;
- one BF16-exact SiLU/multiply owner in each vision MLP;
- two BF16-exact projection-bias/residual owners in each vision block;
- one BF16-exact Gate/Up-bias and SiLU/multiply owner in each vision MLP;
- direct grouped-QKV producer layout for the 28 windowed vision blocks, without
  an intermediate grouped-pack kernel or three transient packed allocations;
- one packed Q/K/V projection GEMM per vision block, with a direct epilogue
  consumer and no split materialization.

The latest packed-MLP refinement removes 72 graph nodes per generated token.
Eager decode and prefill retain their previous projection ownership and GEMM
tactics. Unsupported shapes do not silently select an unqualified path.

The latest M1 GEMV refinement lowers the frozen 128-token decode wall median
from 1.03749 s to 1.01698 s across five alternating pairs. All five pairs win,
the paired wall speedup median is `1.0203x`, and the slowest pair is `1.0188x`.
The 12K guard remains within 0.22% in every pair; the 32K wall cell wins 5/5.

The latest short-decode residual/RMSNorm refinement lowers the 128-token wall
median from 1.01910 s to 1.00918 s across five alternating pairs. All five
pairs win, the paired wall speedup median is `1.0105x`, and the slowest pair is
`1.0050x`. TPOT falls from 7.871 ms to 7.792 ms. The 12K and 32K graphs do not
install this selector; their worst paired wall regressions remain bounded at
0.402% and 0.145%, respectively.

The latest short-attention refinement lowers the frozen 1,024+128 wall median
from 1.21039 s to 1.13270 s and TPOT from 8.994 ms to 8.369 ms. All five
alternating pairs win; paired wall speedup median is `1.0688x` and the slowest
pair is `1.0664x`. The 128+128 guard also wins 5/5 at about `1.011x`. The 12K
and post-32K graphs do not install W32; their wall medians remain within 0.08%.

The latest QKV-prelude refinement lowers the frozen 1,024+128 wall median from
1.12922 s to 1.11967 s and TPOT from 8.358 ms to 8.280 ms. All five alternating
pairs win; paired wall speedup median is `1.0077x` and the slowest pair is
`1.0062x`. The 128+128 guard improves about `1.43%`. The 12K/post-32K paths do
not install the candidate; their wall medians remain within 0.19%. One guard
sample had a simultaneous long-prefill TTFT health outlier and is retained in
the raw evidence rather than discarded.

The latest pack8 residual/RMSNorm refinement lowers the frozen 1,024+128 wall
median from 1.11748 s to 1.09324 s and TPOT from 8.260 ms to 8.078 ms. All five
alternating pairs win; paired wall speedup median is `1.0227x` and the slowest
pair is `1.0184x`. The 128+128 guard improves `2.60%`. The 12K/post-32K paths do
not install the candidate; their wall medians remain within `0.08%` and
`0.03%`, respectively. Systems attributes 0.192 ms of per-step GPU-busy saving
to the intended residual/RMSNorm nodes while kernel count remains unchanged.

The latest scaled exp-cache refinement lowers the real-PNG wall median from
0.58104 s to 0.57302 s and TTFT from 251.81 ms to 244.21 ms. All five paired
wall and TTFT samples win; the paired wall speedup median is `1.0149x` and the
slowest pair is `1.0068x`. The frozen 1K text TTFT also wins 5/5 and falls
`1.65%`. Systems removes exactly 36 scale nodes before the first image token,
reduces GPU busy by 7.38 ms and reduces the complete first-token GPU envelope
by 5.18 ms. Request-scoped 12K FA2, 32K, and decode trajectories remain exact.

The latest vision-QKV refinement lowers the same real-PNG wall median from
0.56924 s to 0.56559 s and TTFT from 243.87 ms to 239.05 ms. TTFT wins all five
alternating pairs and complete wall time wins four of five; paired TTFT speedup
is at least `1.0193x`. Systems replaces 160 Q/K/V-bias and Q/K-RoPE nodes with
32 fused nodes, reduces first-token GPU busy by 2.92 ms, and shortens the
first-token envelope by 5.88 ms. The complete PNG trajectory is exact and the
text, audio, 12K, 32K, HTTP-error, and malformed-media guards remain unchanged.

The latest vision-MLP refinement lowers real-PNG wall median from 0.56543 s to
0.56269 s and TTFT from 239.24 ms to 235.36 ms. TTFT wins all five alternating
pairs and complete wall time wins four of five; paired TTFT speedup is at least
`1.0105x`. Systems removes exactly 32 SiLU nodes, reduces the target boundary
from 8.294 ms to 4.884 ms, reduces first-token GPU busy by 3.79 ms, and shortens
the first-token envelope by 7.31 ms. Text, audio, 12K, 32K, HTTP-error, and
malformed-media guards retain their accepted trajectories.

The latest vision residual refinement lowers real-PNG wall median from 0.56570
s to 0.55984 s and TTFT from 238.54 ms to 234.08 ms. TTFT wins all five
alternating pairs and complete wall time wins four of five; paired TTFT speedup
is at least `1.0033x`. Systems replaces 64 bias plus 64 residual-add nodes with
64 exact two-round nodes, reduces the target GPU boundary from 3.900 ms to
2.756 ms, and reduces first-token GPU busy by 0.91 ms. The larger profile
envelope change includes a host-side gap and is not attributed to the kernel.

The latest vision Gate/Up refinement lowers real-PNG wall median from 0.56073 s
to 0.55675 s and TTFT from 234.61 ms to 227.45 ms. All five alternating wall
and TTFT pairs win; the slowest paired wall and TTFT speedups are `1.0067x` and
`1.0230x`. Systems replaces 64 Gate/Up bias nodes plus 32 SiLU/multiply nodes
with 32 exact owners, reducing the target boundary from 10.926 ms to 4.817 ms,
first-token GPU busy by 6.67 ms, and the complete envelope by 6.07 ms.

The latest grouped-QKV layout refinement lowers real-PNG TTFT from 227.58 ms
to 225.61 ms. All five alternating TTFT pairs win and the slowest paired
speedup is `1.0076x`. Complete wall p50 falls from 0.55401 s to 0.55232 s, but
only three of five wall pairs win, so no stable wall improvement is claimed.
Systems removes exactly 28 grouped-pack nodes, saves 1.96 ms at the changed
producer/pack boundary, and reduces first-token GPU union by 1.75 ms. The four
full-attention blocks and final restore-to-token-order step remain unchanged.

The latest packed vision-QKV refinement lowers real-PNG TTFT from 225.90 ms to
224.70 ms. All five alternating TTFT pairs win and the slowest paired speedup
is `1.0024x`. Complete wall p50 changes from 0.55113 s to 0.55149 s and only
three of five wall pairs win, so no stable wall improvement is claimed.
Systems replaces 96 projection GEMMs with 32 packed GEMMs, removes exactly 64
kernels, saves 1.54 ms across the projection-plus-epilogue boundary, and
reduces first-token GPU union by 1.71 ms.

## Accepted measurements

| Workload | ApxInf TTFT | ApxInf TPOT | vLLM-Omni 0.26.0 TPOT | Result |
|---|---:|---:|---:|---|
| 1,024 + 32 | 65.091 ms | 8.088 ms | 22.617 ms | ApxInf TPOT `2.796x` |
| 1,024 + 128 | 65.668 ms | 8.078 ms | — | paired wall `1.0227x` |
| 128 + 128 | 16.613 ms | 7.425 ms | 22.681 ms | ApxInf TPOT `3.055x` |
| 8,192 + 8 | 406.548 ms | 10.694 ms | 19.111 ms | ApxInf TPOT `1.787x` |
| 12,288 + 8 | 655.240 ms | 13.075 ms | 19.094 ms | ApxInf TPOT `1.460x` |
| 32,760 + 8 | 2,596.522 ms | 10.242 ms | 17.577 ms | ApxInf TPOT `1.716x` |

For the latest packed-MLP candidate, five fixed-parent AB/BA pairs at
32,760+8 all favored the candidate. Baseline and candidate TPOT medians were
10.432 ms and 10.242 ms; the paired speedup median was `1.0250x` and the
minimum pair was `1.0172x`. The 12K eager guard was neutral at about `1.001x`.
All compared text trajectories were exact.

For real PNG 1,760+16, ApxInf wall p50 is 0.551 s versus 0.565 s for
vLLM-Omni. On the frozen matched records ApxInf is lower on wall p50, TTFT
(`224.70` versus `231.96` ms), and `2.446x` lower TPOT. The internal
ApxInf-to-ApxInf wall admission remains explicitly negative because only three
of five alternating wall pairs win. For real WAV 52+16, the final acceptance
wall is 0.172 s versus vLLM-Omni 0.619 s.
Both media cases preserve the accepted complete output-token trajectories.

The legal 32,760+8 boundary passes. The final acceptance sample records
15,805 MiB peak memory and at least 8,759 MiB headroom. Requests beyond the
combined context limit are rejected as typed HTTP 400 responses rather than
being admitted to OOM.

## Correctness and regression status

- model CPU tests: 68 passed;
- benchmark-script tests: 15 passed;
- CUDA tests: 94 passed, with two known FP8 cuBLAS status-15 controls outside
  the frozen BF16 Omni contract;
- pack8 residual/RMSNorm CUDA regression: 1 passed;
- scaled exp-cache CUDA regression: 1 passed;
- vision QKV bias/RoPE CUDA regression: 1 passed;
- vision fused SiLU/multiply CUDA regression: 1 passed;
- vision exact bias/residual CUDA regression: 1 passed;
- vision Gate/Up bias SiLU/multiply CUDA regression: 1 passed;
- vision grouped-QKV producer-layout CUDA regression: 1 passed;
- grouped FA2 prepacked-input CUDA regression: 1 passed;
- vision packed-QKV direct-consumer CUDA regression: 1 passed;
- exact text trajectories: 1K, 128-token decode, 4K, 8K, 12K, and 32K cells;
- exact media trajectories: real PNG and WAV;
- typed contract and malformed-media recovery: passed.

## Checked-in evidence

- `results/promotion-m1-packed-mlp.json`: current 32K text promotion,
  correctness, graph attribution, binaries, and rollback identity;
- `results/promotion-m1-gemv-tactics.json`: current 128-token decode promotion,
  guard cells, operator gate, paired wall timing, and profile attribution;
- `results/promotion-short-exact-residual-norm.json`: current short-decode
  residual/RMSNorm promotion, long-context guards, and node-level attribution;
- `results/promotion-short-w32-attention.json`: current 1K short-decode
  attention geometry promotion, 128/12K/32K guards, and Systems attribution;
- `results/promotion-fused-qkv-prelude.json`: current short-decode packed-QKV
  producer promotion, long-path guards, cache correctness, and node attribution;
- `results/promotion-pack8-residual-rmsnorm.json`: current exact H=2,048
  residual/RMSNorm I/O promotion, long-path guards, and Systems attribution;
- `results/promotion-scaled-exp-cache-prefill.json`: current real-image and
  short-text prefill promotion, FA2 precedence gate, and Systems attribution;
- `results/promotion-vision-qkv-bias-rope.json`: current real-image vision
  projection-epilogue promotion, strict model gate, and Systems attribution;
- `results/promotion-vision-fused-silu-mul.json`: current real-image vision MLP
  activation promotion, exact primitive gate, and Systems attribution;
- `results/promotion-vision-bias-residual.json`: current real-image vision
  projection/residual promotion, two-round BF16 gate, and Systems attribution;
- `results/promotion-vision-gate-up-bias-silu-mul.json`: current real-image
  vision Gate/Up activation promotion, four-seam BF16 gate, and attribution;
- `results/promotion-vision-prepacked-qkv.json`: current real-image TTFT and
  GPU-path promotion, direct grouped producer layout, and explicit wall
  non-claim;
- `results/promotion-vision-packed-qkv.json`: current real-image TTFT and
  GPU-path promotion, unique packed projection ownership, and explicit wall
  non-claim;
- `results/promotion-grouped-varlen-fa2.json`: current real-image promotion and
  complete multimodal controls;
- `results/omni-packed-mlp-acceptance-summary.json`: final endpoint acceptance
  matrix;
- `results/apxinf-vs-vllm-omni-0.26.0.json`: matched ApxInf/vLLM-Omni text,
  context, media, capacity, MFU, and BWU summary.

The promotion summaries retain hashes and provenance names for raw trials and
profiler reports. Those artifacts are deliberately not part of the source
contract. Re-run the benchmark scripts when new raw evidence is needed.

## Claim limits

These results do not claim multi-request throughput, continuous batching,
non-SM89 portability, video input, or speech output. MFU and BWU are derived
from an explicit dense-BF16 peak convention and algorithmic byte lower bounds;
they are diagnostic estimates, not measured HBM transactions.
