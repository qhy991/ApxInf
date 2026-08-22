# Native Qwen2.5-Omni RTX 4090 baseline

## Frozen deployment

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
allocation, in-place KV reset, decode-only TMRoPE position caching and a
single-owner packed QKV layout, plus a short-KV CUDA Graph with exact
two-stage GPU token selection and fused TMRoPE K/V cache publication. The graph and
selection path are selected only for SM89 one-token BF16 decode with
`start_pos < 3072`; longer-KV decode and all prefill retain the accepted
ordinary path. It preserves complete trajectories through the 10,752-token
passing boundary, keeps the first failure at 11,264 tokens, and recovers the
immediately following request after OOM. The deployed SM89 binary SHA-256 is
`dcccfb7bc7c9ca8d634a09f2128028e9f86d770a3e1d6aa83fe238fba8bea4e2`.

Relative to the graph/token-selection service, packed QKV improves 1K+32 TPOT
from 9.707 ms to 9.485 ms (1.023×) and 128+128 TPOT from 8.582 ms to
8.363 ms (1.026×). Fused TMRoPE K/V publication then reaches 9.384 and
8.252 ms (another 1.011× and 1.013×). The
2,048 and 2,560-token TPOT rows additionally improve by 1.081× and 1.044×
after extending the graph to its measured crossover; packed ordinary decode
then improves the 3,072 and 3,584 rows by 1.030× and 1.033×. Against the
original baseline, decode TPOT improves from 17.567 ms to 8.252 ms (2.129×).
See `PROFILE.md` and the structured raw results for the promotion record.

The actual promoted-binary profile records 127 `cudaGraphLaunch` calls and no
decode-step logits D2H. The 128-block partial argmax and one-block final
argmax take 2.69 and 2.40 microseconds in the observable eager prewarm; the
fused TMRoPE K/V node takes 3.66 microseconds and the complete request profile
reports 8.276 ms average stream synchronization.
Gate/Up packing and one-block GPU argmax were separately retained as null and
sub-threshold results. Remaining single-request decode latency is dominated
by GPU graph compute: the BF16 text-weight read lower bound is 6.172 GB/token,
equivalent to about 748 GB/s or 74.2% of the RTX 4090's 1,008 GB/s peak at the
accepted 8.252 ms TPOT. The evidence still does not include a vLLM baseline
or multi-request serving; the bandwidth figure is a weight-only lower-bound
estimate, not a memory-transaction counter.
