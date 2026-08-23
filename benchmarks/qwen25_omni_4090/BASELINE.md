# Native Qwen2.5-Omni RTX 4090 baseline

## Current accepted service

- Model/revision:
  `Qwen/Qwen2.5-Omni-3B@f75b40e3da2003cdd6e1829b1f420ca70797c34e`
- GPU: NVIDIA GeForce RTX 4090, 24,564 MiB, single request, BF16
- Deployed binary SHA-256:
  `432ac73ef573f36fa47b8c2112abc7f1b5b561a79816ed60f57c16f0d06adb18`
- Immediate rollback SHA-256:
  `2296e9b3010d8174902f7ec6b2ffb22f7a046121bb4beb21f364d67eb1468374`
- Deployment owner: runit launches the checked-in Broker service definition;
  the service is stopped when unused so other queued work can own the GPU.

The current path composes the prior exact softmax, chunking, CUDA Graph,
GPU-token-selection and cache optimizations with flattened long-prefill GQA
and uninitialized allocation for outputs that GEMM, softmax, pointwise,
normalization, RoPE, embedding or QKV-split producers provably overwrite in
full. Exact SM89 chunk-shape GEMMs use load-time cuBLASLt tactics from the
same immutable TacticStore; all unmatched shapes retain vendor cuBLAS.
Long flattened-GQA softmax with at most 256 query tokens scales its
single-consumer score buffer once and normalizes that storage in place; larger
query chunks retain the non-mutating path.
The shared-memory FP32 numerator cache uses a parallel exact maximum while
preserving its ordered FP32 sum and normalization.
The long-decode global FP32 numerator cache uses the same parallel exact
maximum while preserving its existing ordered global-cache sum.
Separate Gate SiLU and elementwise multiply launches are replaced by one
backend primitive that deliberately rounds the SiLU intermediate to BF16
before multiplication, preserving the removed tensor bit for bit.
Windowed vision attention caches an ascending group-to-key plan and skips
masked K/V rows while preserving every original key index, max/sum lane and
accumulation order. The four full-attention vision blocks remain unchanged.
For BF16 multi-token attention above 4,096 cached tokens, sequence and GQA
rows sharing one K/V head are packed into large score and value GEMMs. The
existing KV-cache representation is unchanged, and the output is restored to
the model-owned row layout before its next consumer.

| Workload | Repeats | TTFT p50 | Prior TTFT p50 | Change | TPOT p50 |
|---|---:|---:|---:|---:|---:|
| 1,024 + 32 | 5 | 66.02 ms | 67.18 ms | 1.7% lower | 9.368 ms |
| 8,192 + 8 | 3 | 0.816 s | 0.822 s | 0.7% lower | 10.582 ms |
| 12,288 + 8 | 10 paired | 1.374 s | 1.386 s | 0.9% lower | 13.063 ms |
| 32,760 + 8 | 3 | 5.719 s | 5.759 s | 0.7% lower | 24.370 ms |

The real PNG gate improves independently of text execution:

| Workload | Prior TTFT | Current TTFT p50 | Change | Prior wall | Current wall p50 |
|---|---:|---:|---:|---:|---:|
| PNG, 1,760 + 16 | 33.382 s | 6.733 s | 79.8% lower | 39.555 s | 12.845 s |

All repeated text cases preserve their accepted complete trajectory hashes.
The exact 32,768-token contract passes with 9,591 MiB minimum sampled memory
headroom. The 50-test CUDA operator regression, typed invalid-request recovery,
and real PNG/WAV exact-token gates pass. See `PROFILE.md` and the
`candidate-vision-grouped-sparse-*` and
`deployed-vision-grouped-sparse-*` records for the latest promotion evidence.

## Original frozen deployment

- Model: `Qwen/Qwen2.5-Omni-3B`
- Model revision: `f75b40e3da2003cdd6e1829b1f420ca70797c34e`
- ApxInf candidate: `b8292a30dc0c898918f118c5b1af094f3a26b094`
- Remote binary SHA-256: `27f8edab296731137a76edfd43af720a5bdaaad1b503e90c89e5c9c332d9c7f0`
- GPU: NVIDIA GeForce RTX 4090, 24,564 MiB
- Service: runit + `gpu-run`, exclusive GPU 0, concurrency one
- Processor: Transformers 4.56.0, huggingface-hub 0.34.4, tokenizers 0.22.0,
  slow Qwen2VL image processor, offline local snapshot
- API: `/v1/evaluations/generate`, greedy, non-streaming, pre-tokenized,
  `ignore_eos=true`

KerSor v6 completed with `deployment_verified=true`; its independent run
verifier passed with no completion gaps. The service also passed real
text/image/audio HTTP requests with `fallback_active=false`.

## Fixed-workload timing

| Workload | Repeats | TTFT p50 | TPOT p50 | Decode rate p50 | Client wall p50 | CV |
|---|---:|---:|---:|---:|---:|---:|
| 1,024 prompt + 32 output | 3 | 2.052 s | 18.90 ms | 52.91 tok/s | 2.746 s | <0.8% |
| 128 prompt + 128 output | 3 | 183.5 ms | 17.57 ms | 56.93 tok/s | 2.520 s | <0.8% |

Every repeated workload produced one stable trajectory hash. TTFT is the
service's first-token interval; the reported prefill tokens/s value is only a
proxy because TTFT includes first-token work.

## Context gradient and capacity

Each context case requests eight output tokens. Exploratory points use one
trial because the current quadratic path takes several minutes at the upper
end.

| Prompt | Result | TTFT | TPOT | Peak VRAM | Mean GPU util | Mean power |
|---:|---|---:|---:|---:|---:|---:|
| 1,024 | pass | 2.059 s | 18.94 ms | 12,395 MiB | 75.6% | 117.4 W |
| 2,048 | pass | 4.195 s | 24.85 ms | 12,705 MiB | 79.0% | 237.0 W |
| 4,096 | pass after grid fix | 18.190 s | 28.78 ms | 13,899 MiB | 96.4% | 357.6 W |
| 6,144 | pass | 51.311 s | 48.13 ms | 15,865 MiB | 95.8% | 405.4 W |
| 8,192 | pass | 114.142 s | 61.00 ms | 18,593 MiB | 97.7% | 429.6 W |
| 10,240 | pass | 216.863 s | 78.29 ms | 22,095 MiB | 97.7% | 430.7 W |
| 10,752 | pass | 250.451 s | 79.61 ms | 23,089 MiB | 99.1% | 430.6 W |
| 11,264 | CUDA OOM | n/a | n/a | allocation failed | n/a | n/a |
| 12,288 | CUDA OOM | n/a | n/a | allocation failed | n/a | n/a |

The declared model capacity is 32,768 total tokens, but the current native
implementation's proven 24 GiB operating range is 10,752 prompt + 8 output.
The first proven failing point is 11,264 prompt + 8 output. Both OOM probes
were fail-closed; runit relaunched the service and Broker ownership, resident
VRAM, and `/health` recovered.

## Closed launch-boundary defect

Before commit `b8292a3`, 4,096 tokens failed immediately with CUDA error 9.
The attention-score tensor has `seq_len * heads = 4,096 * 16 = 65,536` rows,
while the fused softmax launch placed every logical row in `grid.y`, whose
limit is 65,535. A minimal 65,536×1 BF16 regression reproduced the same error
with negligible memory. The fix maps logical rows across `grid.y × grid.z`.

Evidence gates:

- red boundary test: CUDA error 9, exit 101;
- green boundary and existing small-reference tests: exit 0;
- 4,096-token endpoint: changed from error 9 to a complete eight-token
  trajectory;
- service health and Broker ownership remained recoverable.

## Accepted successor

The accepted path composes sequential-order softmax with a shape-specialized
FP32 numerator cache, strided-batched GQA prefill, stream-ordered transient
allocation, in-place KV reset, decode TMRoPE position caching and a
single-owner packed QKV layout, plus prefill TMRoPE position reuse and a
short-KV CUDA Graph with exact
two-stage GPU token selection and fused TMRoPE K/V cache publication. Text-only
prompts longer than 1,024 tokens are evaluated as exact causal chunks against
the same KV owner: 512 tokens through the measured 12,288-token crossover and
1,024 tokens above it, with a 256-token specialization from 4,096 through
6,144. Image and audio prefill remain unchunked.
Decode beyond the 11,264-column exp-cache gate uses an explicit exact scalar
fallback; leaving that selector unset still fails closed.

The graph and selection path remain restricted to SM89 one-token BF16 decode
with `start_pos < 3072`; longer-KV decode keeps ordinary compute but uses the
same exact two-stage GPU selection. The current
deployed binary SHA-256 is
`321e8a6db1e932e5138170070ae4bd27183e42cd3143128c75d477f10ccaba5e`.
The prior accepted fused-KV and chunked binaries remain archived by their
`dcccfb7b`, `f6ba8836`, `c8e06b41`, `778066a3`, `ac8e2436` and `0fc97f78`
prefixes, with `881491b0` as the immediate rollback.

Relative to the graph/token-selection service, packed QKV improves 1K+32 TPOT
from 9.707 ms to 9.485 ms (1.023×) and 128+128 TPOT from 8.582 ms to
8.363 ms (1.026×). Fused TMRoPE K/V publication then reaches 9.384 and
8.252 ms (another 1.011× and 1.013×). The
2,048 and 2,560-token TPOT rows additionally improve by 1.081× and 1.044×
after extending the graph to its measured crossover; packed ordinary decode
then improves the 3,072 and 3,584 rows by 1.030× and 1.033×. Against the
original baseline, deployed decode TPOT improves from 17.567 ms to 8.256 ms
(2.128×). Decode is statistically unchanged by chunked prefill. At 2,048,
2,560, 3,072 and 4,096 prompt tokens, chunking lowers TTFT by 12.7%, 14.8%,
16.5% and 21.6%, respectively. At 10,752+8, TTFT falls from 7.930 s to
5.525 s (1.435×). See `PROFILE.md` and the structured raw results for the
complete promotion record.

The adaptive 512-token refinement then lowers the already-chunked TTFT by a
further 5.85% at 2,048 tokens, 2.68% at 4,096, 3.20% at 8,192 and 1.41% at
11,264. A constant 512-token policy began regressing at 16K; the 12,288-token
crossover restores the accepted 1,024-token path above that point. Decode and
1K prefill remain unchanged.

Removing output normalization, hidden D2H, LM-head projection and logits D2H
from every non-final chunk lowers the adaptive TTFT by another 3.98% at 2K,
5.51% at 4K, 3.90% at 8K, 3.87% at 11K and 3.44% at both 16K and 32K. Only
the final chunk produces logits; KV semantics and complete trajectories remain
unchanged.

A final chunk-size screen found that 256-token chunks regress 2K/3K and 11K,
but improve a contiguous 4K–6K interval. Restricting them to 4,096–6,144
lowers deployed 4K TTFT from 539.121 to 532.223 ms (1.28%) while 2K, 8K,
11K and the 32K capacity path retain their accepted chunk sizes.

Reusing one device TMRoPE position array per prefill slice then lowers 1K TTFT
from 80.439 to 76.877 ms (4.43%), 4K from 532.223 to 506.916 ms (4.75%),
11K from 5.152 to 5.106 s (0.88%) and 32K from 43.784 to 43.491 s (0.67%).
It preserves the full context boundary and exact text/image/audio trajectories.

Taking a zero-copy GPU view of the final hidden row before output normalization
then lowers the 10-sample 1K TTFT median from 76.871 to 75.802 ms (1.39%). It
removes the final whole-slice D2H and hidden-row H2D transfers; 4K/11K timing,
decode TPOT, 32K capacity and all trajectories remain unchanged.

Extending exact GPU argmax to graph-ineligible eager decode then lowers TPOT
by 1.80% at 4K, 1.47% at 8K and 1.07% at 11K. It removes one full-logit D2H
and CPU scan per generated token while retaining the ordinary long-KV compute
path and complete trajectories.

For decode beyond the 11,264-column shared numerator-cache limit, an exact
global FP32 cache lowers scalar-path TPOT by about 14.6% at 11K, 15.0% at 12K
and 8.0% at 32K. One CTA kernel performs fill, ordered sum and normalize with
block-local synchronization. It preserves scalar max/sum order, all output
trajectories and the 32K memory boundary.

The former 10,752-token memory ceiling is now historical. Exact trajectories
pass at 11,264, 12,288, 16,384, 24,576 and 32,760 prompt tokens; the last case
requests eight outputs and exactly fills the declared 32,768-token contract.
Its current TTFT is 43.591 s. No legal single request now OOMs in
the tested gradient. A request exceeding the combined context, a 129-token completion,
nonzero temperature and evaluation streaming all return typed HTTP 400
`invalid_request` responses, and `/health` remains ready afterwards.

The actual promoted-binary profile records 127 `cudaGraphLaunch` calls and no
decode-step logits D2H. The 128-block partial argmax and one-block final
argmax take 2.69 and 2.40 microseconds in the observable eager prewarm; the
fused TMRoPE K/V node takes 3.66 microseconds and the complete request profile
reports 8.276 ms average stream synchronization.
Gate/Up packing, one-block GPU argmax and combined Q/K/V TMRoPE were retained
as null or sub-threshold results. Remaining single-request decode latency is
dominated by GPU graph compute: the BF16 text-weight read lower bound is
6.172 GB/token, equivalent to about 747.6 GB/s or 74.17% of the RTX 4090's
1,008 GB/s peak at the accepted 8.256 ms TPOT. In the 11,264-token request
profile, long-KV scalar softmax accounts for about 1.638 s and small-N GEMV for
about 2.056 s of summed GPU kernel time, so long-prefill softmax is the next
bounded target. The evidence still does not include a vLLM baseline or
multi-request serving; the bandwidth figure is a weight-only lower-bound
estimate, not a memory-transaction counter.
