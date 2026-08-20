# Approximate Marlin-M64 serving contract on RTX 4090

Status: screening passed on 2026-08-18; retain as opt-in `continue`, not the
accuracy-sensitive default.

## Naming boundary

The checkpoint's authoritative Hugging Face identity is
`model_type=qwen3_5` and `Qwen3_5ForConditionalGeneration`. Architecture-level
config parsing therefore keeps the Qwen3.5 name; renaming it to `qwen3` would
incorrectly imply compatibility with the distinct Qwen3 implementation.

This campaign is narrower than the full Qwen3.5 architecture. The native CUDA
path is currently specialized to the Qwen3.8-27B tensor contract
(`hidden=5120`, `intermediate=17408`, 64 layers, 24 query heads, 4 KV heads)
and SM89. New experiment schemas therefore use `qwen38_27b`, while the
benchmark directory remains `qwen38_4090`. Existing `qwen35` source symbols,
probe names, and archived evidence are retained during performance screening
to avoid an unrelated migration changing the accepted binary or breaking
replay. A future source migration should separate `qwen3_5` architecture
parsing from an explicitly gated `qwen38_27b_sm89` execution specialization.

## Why this is a separate candidate

The exact-M8-alignment Marlin candidate was rejected because 64-layer hidden
residuals diverged (`cosine=0.97014`, `relative-L2=0.24472`) even though the
same deterministic run retained first/second logit cosine above 0.99988 and
both greedy argmax token IDs exactly. The exact M16 CUDA alternative then
proved bitwise correct but delivered no speedup.

The user's declared bring-up contract explicitly lists BF16 numerical
equivalence as a non-goal and asks only that inference have no obvious quality
or capability deviation. This document therefore defines a new
**approximate-serving** candidate whose authority is final logits, greedy token
trajectories, functional task results, and service timing. It does not relax,
rewrite, or retroactively pass the rejected hidden-state experiment. The M8
path remains the accuracy-sensitive default and rollback.

## Frozen deployment and execution contract

```text
model                 cyankiwi/Qwen3.8-27B-AWQ-INT4
model revision        63768c10df38c0395e12ef49edac1bd539eaeeea
source                 715a0ed790f2d10d82fab53fbeac3da3075adf26 + dirty optimization overlay
GPU                    one RTX 4090 / SM89 / 24 GiB
parallelism            TP=PP=DP=1
quantization           compressed-tensors asymmetric U4 group-32
activation/output      BF16
KV                     BF16, 32K resident
execution              eager, one stream, single request, greedy argmax
baseline               accepted M8 prefill plus M1 tail
candidate              M64 Marlin MLP gate_up/down; M8 stateful mixer; explicit M8/M1 tail
decode                 identical accepted ModelOptimized M1 path
API                    one resident process; per-request explicit selector
```

The service is started with `--enable-experimental-marlin-m64`, but every
request defaults to M8. Only a request carrying
`"apxinf_prefill_mode":"marlin-m64"` enters the candidate. Requests shorter
than 64 tokens fail closed in candidate mode. Responses and logs must prove the
selected mode and M64/M8/M1 tile counts; no silent fallback is allowed.

## Quality and correctness gates

The rejected hidden-state metrics remain visible diagnostics. Admission uses
these predeclared user-facing endpoints:

1. **Synthetic 64-token stack:** first- and second-step full-vocabulary logit
   cosine at least 0.999, relative L2 at most 0.05, all finite, and both greedy
   argmax IDs exact against M8.
2. **Needle capability:** run all six frozen
   `text-niah-{1024,8192}-{p10,p50,p90}` cases with normalized-exact expected
   answers. Candidate functional pass rate must be no lower than M8 and it may
   not turn any M8 pass into a failure. Both modes must return a non-empty valid
   response; no retry after a failure. Existing M8 capability failures, if any,
   are recorded rather than incorrectly attributed to the candidate.
3. **Performance prompts:** 1K and 8K requests must be non-empty, valid SSE,
   end with usage and `[DONE]`, contain only valid token IDs, and leave the
   service healthy.
4. **Trajectory:** record the exact token-ID match rate and common-prefix length
   for every paired performance request. Require the first greedy token exact
   in every pair and at least 90% aggregate token-ID match. Functional success
   is mandatory even if later greedy trajectories diverge.
5. **State/reset:** repeat a mixed M64+M8/M1 request after another request and
   require the same token trajectory, proving request-local state reset.

This contract is not a general accuracy, perplexity, benchmark, or BF16
equivalence claim. Passing only means no obvious regression on the declared
deterministic text cells.

## Performance gates

Official timing is client-observed, no-profiler wall time in one resident
process. The paired schedule alternates AB/BA, where A is M8 and B is
Marlin-M64. Preserve raw order and GPU health samples.

- 1K: one warmup per arm, five measured pairs; candidate wins at least 4/5 and
  TTFT median speedup is at least 1.50x.
- 8K: two measured balanced pairs after one warmup per arm; candidate wins 2/2
  and median TTFT speedup is at least 1.50x.
- Decode guard: paired TPOT median may not regress by more than 5% in either
  target cell.
- Reliability: no OOM, API failure, invalid path marker, foreign GPU process,
  Xid, hardware slowdown, or less than 256 MiB measured memory headroom.

An end-to-end win remains **opt-in approximate** until a formal 25-pair run or
separately frozen equivalent confidence rule and a complete Nsight Systems
trace explain the changed critical-path interval. It cannot replace M8 as the
accuracy-sensitive default from screening evidence alone.

## Screening result

The candidate passed the frozen operator, path, reliability, and 1K/8K
screening gates in one resident process. It remains explicit opt-in because the
25-pair confidence run and Nsight Systems causal trace are still pending.

| Cell | M8 TTFT median | M64 TTFT median | Speedup | Wins | TPOT ratio | Token trajectory |
|---|---:|---:|---:|---:|---:|---:|
| 1K / 128 output | 9.0216 s | 5.9725 s | 1.5105x | 5/5 | 1.0006 | 640/640 exact |
| 8K / 128 output | 71.6709 s | 47.1258 s | 1.5208x | 2/2 | 1.0066 | 248/256 exact; first 104 exact in each pair |

The complete 64-token, 64-layer stack probe measured 512.160 ms for eight M8
tiles versus 265.538 ms for M64, a 1.9277x median speedup with 5/5 wins. First-
and second-step argmax IDs were exact; first/second logit cosine was
0.999780/0.999882 and relative L2 was 0.02432/0.01535. Hidden-state divergence
remains diagnostic and is not reclassified as exact equivalence.

All six frozen NIAH comparisons admitted the candidate because it did not
regress M8 and both modes produced identical trajectories. Neither M8 nor M64
returned the expected key on any of the six cases, so this is a non-regression
result, not a capability pass. Repeated 1K and 8K requests each produced one
stable trajectory per mode, satisfying the request-state reset check.

Peak measured memory use was 18,373 MiB with 6,191 MiB minimum headroom. The
candidate arm averaged about 81.1% sampled GPU utilization; sampled memory-
controller utilization averaged 41.6% at 1K and 25.3% at 8K. The service
remained healthy after all requests with no OOM or API failure.

Evidence:

```text
source overlay               75c3583a6b3610273589d28cb8ba4772797d21cbb66a454e24d1286c8c4ce590
candidate apxinf             2febdbf9ccb4afc9aaaaca234f1df1b589ad4fa128638ceb482a232de11a04dd
stack probe                  57bd38004e641513243fd93edee53855726ad38c4d75a282132441ed0e58284b
stack result                 24c32efdd7a62a3c45ab89fbba835a26545b22bca8d499e33453d59fecdff021
1K result                    b8609e667110ec23a5707b0ad256a30d4a49b5f81a323d58aefffecf2619ac09
8K result                    178d67b2419ab65cc3d5424b775059f9e99083de1bd94f3c23e65d00b9c5eee5
```

## Hypothesis and stop rule

Runtime transforms plus Marlin M64 replace eight M8 reads of the two large MLP
weights per layer. The 64-token/16-layer joined probe showed up to 1.885x before
the exact hidden-state gate rejected the earlier contract, so the service
hypothesis predicts at least 1.50x TTFT improvement while decode stays fixed.

Reject and restore M8 if any quality/path/reliability gate fails, if 1K misses
the performance floor, or if 8K reverses direction. If it wins but exposes one
specific removable transform or launch loss, retain it as `continue` and allow
one causally targeted iteration; do not tune thresholds or select new prompts
after seeing results.
