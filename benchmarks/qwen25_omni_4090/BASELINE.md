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
  `b1a6252930d0312000cceda77c449988e22c17072fc62e930e196b3ad292083c`
- Immediate rollback SHA-256:
  `f77d1cea2ba62a048405e45221c133ebf0a837d24c79345fbc935b45e6158ac0`
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
- grouped variable-length FA2 for windowed vision attention and FA2 for the
  four full-attention vision blocks.

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

## Accepted measurements

| Workload | ApxInf TTFT | ApxInf TPOT | vLLM-Omni 0.26.0 TPOT | Result |
|---|---:|---:|---:|---|
| 1,024 + 32 | 64.924 ms | 9.358 ms | 22.617 ms | ApxInf TPOT `2.417x` |
| 128 + 128 | 17.281 ms | 7.792 ms | 22.681 ms | ApxInf TPOT `2.911x` |
| 8,192 + 8 | 406.548 ms | 10.694 ms | 19.111 ms | ApxInf TPOT `1.787x` |
| 12,288 + 8 | 655.240 ms | 13.075 ms | 19.094 ms | ApxInf TPOT `1.460x` |
| 32,760 + 8 | 2,596.522 ms | 10.242 ms | 17.577 ms | ApxInf TPOT `1.716x` |

For the latest packed-MLP candidate, five fixed-parent AB/BA pairs at
32,760+8 all favored the candidate. Baseline and candidate TPOT medians were
10.432 ms and 10.242 ms; the paired speedup median was `1.0250x` and the
minimum pair was `1.0172x`. The 12K eager guard was neutral at about `1.001x`.
All compared text trajectories were exact.

For real PNG 1,760+16, ApxInf wall p50 is 0.581 s versus 0.565 s for
vLLM-Omni: near parity, with vLLM retaining lower TTFT and ApxInf retaining
`2.110x` lower TPOT. For real WAV 52+16, ApxInf wall is 0.159 s versus 0.619 s.
Both media cases preserve the accepted complete output-token trajectories.

The legal 32,760+8 boundary passes. The final acceptance sample records
15,993 MiB peak memory and at least 8,571 MiB headroom. Requests beyond the
combined context limit are rejected as typed HTTP 400 responses rather than
being admitted to OOM.

## Correctness and regression status

- model CPU tests: 66 passed;
- benchmark-script tests: 15 passed;
- CUDA tests: 94 passed, with two known FP8 cuBLAS status-15 controls outside
  the frozen BF16 Omni contract;
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
