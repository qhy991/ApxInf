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

- Source commit: `2e09096bd73eb41767bc1ab06eacb09454ce6ef3`
- Binary SHA-256:
  `af0330a74746e36972a2fd24187d7b73f9d7cf491d644b657590e0ffae39a7f1`
- Immediate rollback SHA-256:
  `c322a8bb97635f5efaeb79bbbcab88505d53f273b24531d994f55bd7ab4e20be`
- Service owner: runit plus `agent-gpu-broker`
- Desired state after validation: stopped

Build the accepted SM89 artifact with:

```bash
CARGO_TARGET_DIR=/opt/apxinf/qwen25-omni-sm89-target \
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
- grouped variable-length FA2 for windowed vision attention and FA2 for the
  four full-attention vision blocks.

The latest packed-MLP refinement removes 72 graph nodes per generated token.
Eager decode and prefill retain their previous projection ownership and GEMM
tactics. Unsupported shapes do not silently select an unqualified path.

## Accepted measurements

| Workload | ApxInf TTFT | ApxInf TPOT | vLLM-Omni 0.26.0 TPOT | Result |
|---|---:|---:|---:|---|
| 1,024 + 32 | 64.924 ms | 9.358 ms | 22.617 ms | ApxInf TPOT `2.417x` |
| 128 + 128 | 15.000 ms | 8.255 ms | 22.681 ms | ApxInf TPOT `2.748x` |
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
