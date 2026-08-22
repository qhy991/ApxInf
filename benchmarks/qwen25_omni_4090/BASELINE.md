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

The accepted path composes the sequential-order one-CTA-per-row softmax with
strided-batched GQA prefill. It preserves complete trajectories through the
10,752-token passing boundary, keeps the first failure at 11,264 tokens, and
improves paired 4K TTFT from 18.3407 s to 1.9661 s (9.329×). Decode TPOT is
unchanged within 0.05%. See `PROFILE.md` and the structured raw results for the
promotion record.

The actual candidate profile confirms that batching removes the launch
explosion: `cudaLaunchKernel` calls fall from 892,860 to 8,268. The next
bottleneck is 7,096 request-local `cudaMalloc`/`cudaFree` pairs, consuming
about 2.11 s of profiled host API time. Request-local H2D remains negligible.
The evidence still does not include a vLLM baseline, multi-request serving, or
an MFU/BWU estimate.
