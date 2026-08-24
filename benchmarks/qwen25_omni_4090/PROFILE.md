# Qwen2.5-Omni RTX 4090 optimization evidence

`BASELINE.md` owns the current accepted deployment summary. This file retains
the causal profiles, rejected branches and promotion evidence for each stage;
the current promotion record is **Promoted global-cache parallel maximum**
below.

## Contract

- Model/revision: `Qwen/Qwen2.5-Omni-3B@f75b40e3da2003cdd6e1829b1f420ca70797c34e`
- Baseline source: `7093b119620c80802348d4a31664106bc57638db`
- Binary: `27f8edab296731137a76edfd43af720a5bdaaad1b503e90c89e5c9c332d9c7f0`
- GPU: RTX 4090, SM89, one request, BF16
- Request: 4,096 pre-tokenized prompt IDs + 8 greedy output IDs,
  non-streaming, `ignore_eos=true`
- Performance authority: no-profiler TTFT 18.190 s, TPOT 28.78 ms,
  wall 18.525 s, complete trajectory SHA-256
  `edc940b80ff945971996ddf2b30534773258bb41d3d1812c38f36ba06eaabcbd`
- Profile-only observation: TTFT 22.724 s and wall 23.099 s; the profiler
  overhead is not used for admission.

The Nsight Systems report is 74,384,370 bytes and remains on the owning GPU
host at `/var/lib/agent-gpu-broker/profiles/omni-4k-prefill.nsys-rep`; its
SHA-256 is
`be596e7aa2226f25b7952299c0eb1addb4bbcaa063c132c20f018589c905e588`.
Small CSV summaries are stored in `profiles/`.

## Causal finding

The GPU-kernel summary attributes 83.4% of summed GPU kernel time to 288
`attention_softmax_bf16_kernel` launches: 12.423 s total. The two dominant
WMMA GEMM families account for 6.8% and 5.4%. Host-to-device copies account
for 2.108 s of GPU memory-operation time.

The baseline softmax launch has one logical row in `grid.y` and
`ceil(cols/256)` blocks in `grid.x`. Every output thread independently scans
all valid columns twice, once for max and once for sum. Thus every logical row
repeats the same reductions across multiple blocks and across all output
threads. At 4K, this redundant work dominates the critical path.

The API summary also reports 892,860 `cudaLaunchKernel` calls. This is a
separate control-path problem: the proposed softmax change keeps the 288
softmax launches and only removes redundant CTA/thread work inside them. No
launch-count reduction is claimed for this candidate.

## First candidate and stop rule

Primary classification: **source/runtime graph**. The candidate uses one
256-thread CTA per logical attention row. Threads cooperatively reduce max and
sum through 1 KiB shared memory, then write strided output columns. Logical
rows remain split over `grid.y × grid.z`, preserving support beyond 65,535
rows.

Promotion requires all of the following:

1. the small BF16 FP32-reference test and the 65,536-row launch-boundary test
   pass on SM89;
2. exact complete token trajectories match for 1K/32, 128/128 and 4K/8;
3. real text/image/audio HTTP smoke remains successful with no fallback;
4. repeated no-profiler 4K TTFT and wall improve materially;
5. decode regression is not both material and repeatable in paired A/B;
6. binary resource audit shows no local memory or stack spill.

Initial cooperative-candidate evidence shows 20 registers, 1,024 bytes static
shared memory, zero local memory and zero stack for the BF16 kernel; operator
tests pass; exact trajectories match; and the first candidate-only 4K run
reduces TTFT p50 from 18.190 s to 5.867 s. A paired binary A/B/B/A health
window was then executed.

## Post-candidate attribution

The candidate Nsight Systems report remains on the GPU host at
`/var/lib/agent-gpu-broker/profiles/omni-4k-cooperative.nsys-rep`, size
74,537,241 bytes, SHA-256
`13c9ce6798c43705ae2d00aa0cc2c858e52ff3f9a7b809a6e3bc578ff15d2c3c`.
It records the same output IDs as the baseline profile.

The changed softmax kernel falls from 12.423 s to 34.193 ms of summed GPU
kernel time, a 363× reduction, and from 83.4% to 1.4% of kernel time. The
dominant kernel families become the two WMMA GEMMs at 40.5% and 32.1%, plus
split-K reduction at 15.0%. This is the predicted bottleneck migration.

The candidate does not change the 892,860 `cudaLaunchKernel` calls. In the
candidate profile, CUDA launch API time remains 4.218 s and H2D memcpy time
remains about 2.1 s. These are now the next bounded control/data-path
opportunities; they are not part of the cooperative-softmax gain.

## Cooperative-candidate decision

**Decision: continue, do not promote.** Paired 4K timing was a material win:
TTFT p50 18.294 s to 5.967 s and client wall p50 18.599 s to 6.205 s, with
the exact 4K trajectory preserved. Paired 128/128 decode did not regress.
Real text/image/audio HTTP smoke also passed.

However, the 10,752-token trajectory differed from the frozen baseline. One
quick run performed during an unclean binary switch also returned all-zero
tokens; after waiting for Broker release, a clean cold restart produced 5/5
correct trajectories, proving the all-zero event was a switch-ownership race,
not accepted steady-state evidence. The long-context mismatch remains and is
enough to reject cooperative summation under the exact-trajectory contract.

The bounded next iteration keeps one CTA per row but lets lane 0 calculate max
and sum in the original sequential order and uses the original division
expression. Other lanes only write strided output columns. This preserves the
baseline arithmetic order while still removing duplicate scans from every
output thread and every `grid.x` block.

## Promoted sequential-order candidate

The second candidate keeps one 256-thread CTA per logical row but makes lane 0
perform the maximum and exponential-sum loops in exactly the baseline order.
Two FP32 scalars are published through static shared memory, and all threads
then write strided output columns using the unchanged division expression. The
logical-row mapping over `grid.y × grid.z` is unchanged. This removes duplicate
row scans while preserving the arithmetic order that determines long-context
token trajectories.

The deployed SM89 binary SHA-256 is
`b0d9dc0d13ecf68c0e5f9cd6bd6847eda222edc9ded0e4068ee03817624c29aa`.
`cuobjdump --dump-resource-usage` reports 36 registers, 16 bytes of static
shared memory, zero local memory and zero stack for the BF16 kernel. The small
FP32-reference test and the 65,536-row boundary test both pass.

### Complete-trajectory correctness

| Workload | Result | Candidate trajectory SHA-256 | Baseline agreement |
|---|---:|---|---:|
| 1,024 prompt + 32 output | 5/5 stable | `bf1da0a151446e7ff757474de26ceb13ced3cf9a422fcc17b1f4699fb89d38ea` | exact |
| 128 prompt + 128 output | 7/7 stable in final screen | `a892eb11d69ed4a408e680432ebee0de04a202950ba7c5072731a001c753f039` | exact |
| 4,096 prompt + 8 output | 3/3 stable | `edc940b80ff945971996ddf2b30534773258bb41d3d1812c38f36ba06eaabcbd` | exact |
| 8,192 prompt + 8 output | 1/1 exploratory | `490c84bc9f905195eeeb560ed9b64d55f5e10430cb12f146d672491d860229cf` | exact |
| 10,752 prompt + 8 output | 1/1 exploratory | `19478a6e232ab7479a2a8026f01096ab68a16f72d922be9695d807928125b02d` | exact |

Real HTTP image and audio requests also pass without fallback. The image case
reads the title from a real PNG chart, and the audio case identifies the input
WAV as a continuous sine wave. The deployed service still declares text,
image and audio inputs and text-only output; video and speech output remain
unsupported by contract.

### No-profiler end-to-end result

The 1K, decode and 4K rows below were measured after clean Broker-owned binary
switches in the same health window. Context-limit points are single-trial
capacity evidence and are not used as repeatability claims.

| Workload | Metric | Baseline | Sequential | Ratio / change |
|---|---|---:|---:|---:|
| 1,024 + 32 | TTFT p50 | 2.0755 s | 1.8733 s | 1.108× faster |
| 1,024 + 32 | wall p50 | 2.7737 s | 2.5589 s | 1.084× faster |
| 128 + 128 | TPOT p50 | 17.567 ms | 17.612 ms | 0.26% slower |
| 128 + 128 | wall p50 | 2.5267 s | 2.5296 s | 0.11% slower |
| 4,096 + 8 | TTFT p50 | 18.3407 s | 5.9641 s | 3.075× faster |
| 4,096 + 8 | wall p50 | 18.6458 s | 6.2266 s | 2.995× faster |
| 8,192 + 8 | TTFT, one trial | 114.1419 s | 12.2866 s | 9.290× faster |
| 10,752 + 8 | TTFT, one trial | 250.4515 s | 20.3175 s | 12.327× faster |

The final 4K baseline and candidate TTFT CVs are 1.32% and 0.19%. One
three-sample decode batch immediately after a service restart was noisy
(19.174 ms TPOT p50, 4.83% CV). It did not reproduce: after two warm-ups, the
final seven-sample screen measured 17.612 ms p50 with 0.18% CV, consistent
with the earlier candidate screen at 17.543 ms and the 17.567 ms baseline.

### Sequential-candidate attribution

The actual promoted-candidate report remains on the GPU host at
`/var/lib/agent-gpu-broker/profiles/omni-4k-sequential.nsys-rep`, size
74,147,569 bytes, SHA-256
`672f7d48065124618eda4c4778f6654cb9493c12af70338cec9aa0ab8fa5e0fd`.
The profiled request preserves the 4K baseline trajectory. Its timing includes
profiler overhead and is not used for admission.

The BF16 attention softmax falls from 12.423 s in the baseline profile to
246.974 ms across the same 288 launches, a 50.3× reduction. Its share of
summed GPU kernel time falls from 83.4% to 9.2%. The two dominant WMMA families
now account for 37.3% and 29.6%, and split-K reduction accounts for 13.8%.
The expected bottleneck migration therefore appears in the actual promoted
binary, not only in the rejected cooperative prototype.

The full capture includes model loading, so its aggregate 2.070 s of H2D time
is not a request-path cost. Filtering to the 4K request's observed GPU interval
(`71.4 s..82.2 s`) leaves only 3.594 MB and 0.475 ms of H2D work. The request
still issues 892,860 `cudaLaunchKernel` calls, consuming 4.295 s of host API
time. It also records 1,774,188 stream-capture queries (0.852 s), 7,096 each of
`cudaMalloc` and `cudaFree` (0.622 s and 1.023 s), and 295,704 event records
(0.262 s).

The grid-aware summary makes the launch explosion concrete. Each of the two
dominant 16×16 WMMA kernel families has 294,912 instances, organized as 36
bursts of 8,192 launches; the split-K reduction family has about 295 thousand
instances. This is an observed cuBLAS execution shape, not yet proof of which
model GEMM selects it. Exact GEMM-shape logging and matched cuBLASLt tactics are
therefore the next bounded attribution step.

## Promotion decision

**Decision: promote for the tested single-request BF16 cells.** The candidate
passes the operator, launch-boundary, binary-resource, real multimodal,
complete-trajectory, repeated no-profiler E2E and causal-profile gates. The
Broker-owned service on the RTX 4090 runs the promoted binary, and the Qwen3.8
resident service remains down.

This decision does not claim multi-request or continuous-batching performance,
video or speech generation, vLLM parity, a larger OOM boundary, or MFU/BWU.
The first bounded follow-up is the now-dominant cuBLAS launch/control path at
4K. Request-local H2D is already negligible. The WMMA families should only be
replaced after exact model GEMM shapes are tied to the 8,192-launch bursts and
a matched cuBLASLt counterfactual passes the complete trajectory gate.

## Linear-GEMM counterfactual

The first launch-count hypothesis was wrong: direct cold-L2 comparisons of the
four 4K text-linear shapes found only 1.003× for Q/O, 1.009× for K/V, 1.005×
for gate/up and 1.098× for down with the best of eight cuBLASLt heuristics.
Those isolated shapes cannot explain the 8,192-launch bursts, so no tuning
database or production-path change was made. The probe and its raw JSON are
retained as negative evidence.

Source inspection then identified the real owner in `attention::sdpa`. For
every layer, scalar GQA launched one score GEMM for each of 4,096 query tokens
and two KV heads, followed by the same 8,192-call loop for values. Across 36
layers this produces the two 294,912-instance WMMA families; approximately
295 thousand split-K reductions are secondary work from those calls.

## Promoted strided-batched GQA candidate

Primary classification: **source/runtime graph**. The candidate keeps the
existing score tensor, scale, sequential-order causal softmax and value output
layout. It only rewrites each per-token GEMM loop as one
`cublasGemmStridedBatchedEx` call per KV head. K/V use zero batch stride for
read-only broadcast; query, attention and output retain their original
sequence strides. Each layer therefore uses two score and two value API calls
instead of 16,384 scalar calls.

The candidate is explicit and fail-closed. `APXINF_BATCHED_GQA_PREFILL=1`
enables only multi-token GQA; unset or `0` preserves the scalar path, and any
other value is rejected. Decode (`seq_len=1`) remains on the scalar path. A
CUDA operator test compares scalar and batched BF16 GQA on the same query and
KV cache and passes. The deployed SM89 binary SHA-256 is
`8855ca4e7266e585dc06a5d1639e3bc241b1c2d44b66a0b32501dd066a1d274d`.

### Correctness and capability

| Workload | Result | Trajectory SHA-256 | Prior accepted agreement |
|---|---:|---|---:|
| 1,024 prompt + 32 output | 3/3 stable | `bf1da0a151446e7ff757474de26ceb13ced3cf9a422fcc17b1f4699fb89d38ea` | exact |
| 128 prompt + 128 output | 3/3 stable | `a892eb11d69ed4a408e680432ebee0de04a202950ba7c5072731a001c753f039` | exact |
| 4,096 prompt + 8 output | 3/3 stable | `edc940b80ff945971996ddf2b30534773258bb41d3d1812c38f36ba06eaabcbd` | exact |
| 8,192 prompt + 8 output | 1/1 exploratory | `490c84bc9f905195eeeb560ed9b64d55f5e10430cb12f146d672491d860229cf` | exact |
| 10,752 prompt + 8 output | 1/1 exploratory | `19478a6e232ab7479a2a8026f01096ab68a16f72d922be9695d807928125b02d` | exact |
| 11,264 prompt + 8 output | CUDA OOM | n/a | same first failure |

Real image and audio HTTP requests reproduce the exact prior output-token
sequences with `fallback_active=false`: the PNG chart title is read correctly
and the WAV is described as a continuous sine wave. The 11,264-token probe
returns an explicit CUDA OOM as HTTP 503; the same Broker-owned service process
remains healthy afterward. The proven capacity interval therefore remains
10,752 pass / 11,264 fail.

### No-profiler end-to-end result

| Workload | Metric | Sequential-order | Batched GQA | Change |
|---|---|---:|---:|---:|
| 1,024 + 32 | TTFT p50 | 1.8733 s | 0.3459 s | 5.415× faster |
| 1,024 + 32 | wall p50 | 2.5589 s | 1.0330 s | 2.477× faster |
| 128 + 128 | TTFT p50 | 0.1837 s | 0.0829 s | 2.216× faster |
| 128 + 128 | TPOT p50 | 17.612 ms | 17.621 ms | 0.05% slower |
| 4,096 + 8 | TTFT p50 | 5.9641 s | 1.9661 s | 3.034× faster |
| 4,096 + 8 | wall p50 | 6.2266 s | 2.2440 s | 2.775× faster |
| 8,192 + 8 | TTFT, one trial | 12.2866 s | 6.3336 s | 1.940× faster |
| 10,752 + 8 | TTFT, one trial | 20.3175 s | 13.4895 s | 1.506× faster |

Against the original paired scalar-softmax baseline, 4K TTFT improves from
18.3407 s to 1.9661 s, or 9.329×. The batched candidate's repeated 4K TTFT CV
is 0.38%. A clean runit deployment repeats the 1K result at 0.3468 s TTFT and
the exact baseline trajectory, proving the accepted service is not the
default-off scalar path.

### Batched-candidate attribution

The actual candidate report remains on the GPU host at
`/var/lib/agent-gpu-broker/profiles/omni-4k-batched-gqa.nsys-rep`, size
1,611,010 bytes, SHA-256
`9e9944ebb860bf4aa41342afab17b1c5f298b0ef721ff3667fa772a348386191`.
The profiled request preserves the 4K trajectory; profiler timing is not used
for admission.

`cudaLaunchKernel` calls fall from 892,860 to 8,268 (108× fewer), and their
host API time falls from 4.295 s to 53.1 ms. Stream-capture queries fall from
1,774,188 to 4,716, and event records fall from 295,704 to 792. The split-K
reduction family falls from about 295 thousand instances to 756. This is the
predicted causal movement from batching the score/value calls.

The next bottleneck is now explicit. In the request-filtered profile,
`cudaFree` consumes 1.525 s across 7,096 calls and `cudaMalloc` consumes
0.585 s across the same count. GPU kernel time is led by the unchanged
sequential softmax at 248.3 ms (32.2%), followed by the batched GEMM families.
Request-local H2D remains only 0.477 ms. The next bounded hypothesis is a
request-lifetime allocator/workspace that removes these transient
allocations; it must preserve aliasing, lifetimes, OOM behavior and the exact
trajectory before it can replace this candidate.

## Batched-GQA promotion decision

**Decision: promote strided-batched GQA for the tested single-request BF16
cells.** It passes operator, exact complete-trajectory, repeated no-profiler,
real multimodal, long-context boundary, explicit-path and causal-profile
gates. The Broker-owned service uses the archived candidate binary and the
checked-in runit reference with `APXINF_BATCHED_GQA_PREFILL=1`; the Qwen3.8
resident service remains down.

This result does not claim multi-request or continuous-batching performance,
video or speech generation, vLLM parity, a larger OOM boundary, or MFU/BWU.

## Initial chunked-prefill promotion and full 32K contract

Primary classification: **source/runtime graph**. The previous full-prompt
prefill materialized quadratic score/output work for every token at once and
first OOMed at 11,264 prompt + 8 output tokens. The promoted path applies only
to reset, text-only requests longer than 1,024 tokens. It executes contiguous
1,024-token causal chunks against one KV owner and returns logits from the
final chunk. The existing prefill attention contract derives the causal offset
from accumulated KV length, so no mask, position or cache ownership rule is
changed. Image/audio requests retain the accepted processor-owned multimodal
path.

`APXINF_QWEN25_CHUNKED_PREFILL=1` selects the path; unset or `0` retains the
full prompt and invalid values fail closed.
`APXINF_SOFTMAX_EXP_CACHE_LONG_FALLBACK=1` explicitly selects exact scalar
softmax once decode KV exceeds the tested 11,264-column numerator-cache limit.
Without the flag, that shape returns an error instead of silently changing
kernels. The deployed service records both selectors in
`service/apxinf-qwen25-omni-broker.run`.

### Correctness, capacity and interface gates

| Workload | Result | Trajectory SHA-256 | Prior agreement |
|---|---:|---|---:|
| 1,024 prompt + 32 output | 7/7 candidate; 3/3 final binary | `bf1da0a151446e7ff757474de26ceb13ced3cf9a422fcc17b1f4699fb89d38ea` | exact |
| 128 prompt + 128 output | 7/7 candidate | `a892eb11d69ed4a408e680432ebee0de04a202950ba7c5072731a001c753f039` | exact |
| 2,048–4,096 prompt + 32 output | four stable repeated cells | four frozen hashes | exact |
| 4,096 / 8,192 / 10,752 prompt + 8 output | all pass | three frozen hashes | exact |
| 11,264 / 12,288 / 16,384 prompt + 8 output | all pass | three stable hashes | new capacity |
| 24,576 prompt + 8 output | pass | `c06cdd72bea947c852bb91661789c17a59792f6eeb70675262cecdc363cf4eac` | new capacity |
| 32,760 prompt + 8 output | candidate and deployed pass | `f5ef60ededd5770627b7963e24ff339aef60d63d061cafa37b7ee4e4b0598cb9` | exact contract limit |

Real PNG and WAV inputs reproduce the complete accepted token sequences with
`fallback_active=false`; the chunk selector is ineligible for both media
paths. The reusable gate and record are `benchmark_multimodal.py` and
`results/candidate-chunked-prefill-final-multimodal.json`.

The new `benchmark_contract.py` gate exposed one interface defect: a combined
context overflow was rejected before model work but initially surfaced as HTTP
503 `runtime_error`. `results/deployed-chunked-prefill-contract-pre-fix.json`
retains that failure. Chat and evaluation generation errors now share one
mapping owner; the final report proves combined-context overflow,
completion-limit overflow, non-greedy sampling and evaluation streaming all
return HTTP 400 `invalid_request`. The service remains healthy after all four
probes.

### No-profiler end-to-end result

| Workload | Metric | Fused-KV baseline | Chunked prefill | Change |
|---|---|---:|---:|---:|
| 2,048 + 32 | TTFT p50 | 232.765 ms | 203.196 ms | 12.7% lower |
| 2,560 + 32 | TTFT p50 | 326.419 ms | 278.258 ms | 14.8% lower |
| 3,072 + 32 | TTFT p50 | 440.834 ms | 368.269 ms | 16.5% lower |
| 4,096 + 32 | TTFT p50 | 748.543 ms | 586.865 ms | 21.6% lower |
| 10,752 + 8 | TTFT | 7.930 s | 5.525 s | 1.435× faster |
| 32,760 + 8 | TTFT | OOM / untested | 45.426 s deployed | full contract |

The four repeated short-context TTFT CVs are at most 0.82%, every trajectory
is stable, and TPOT is unchanged. The final deployed binary measures
9.377 ms TPOT at 1K+32 and the same complete trajectory. Its 128+128 screen
measures 8.255 ms TPOT, versus 8.254 ms before chunking and 17.567 ms in the
original baseline. The weight-only effective bandwidth remains about
747.6 GB/s, or 74.17% of the declared 1,008 GB/s peak.

### Causal profile and migrated bottleneck

The exact 11,264+8 profile uses the same inference implementation as the final
binary and reports 5.457 s TTFT; profiler timing is explanatory only. The
report remains on the GPU host at
`/var/lib/agent-gpu-broker/profiles/omni-11264-chunked-prefill.nsys-rep`, size
2,581,652 bytes, SHA-256
`dde5c03bda788202e01f57121dd005152c700cec6751c0bd3c469189b5b4c81a`.
Its source binary SHA-256 is
`f6ba88369c36fbe2eeff6b6aa780bc061ae93e54044a833ba58d8dc2d9f9138c`;
the promoted `c8e06b41` binary differs only by the HTTP error-mapping fix and
repeats the valid quick trajectory.

The 8.221-second idle gap between graph prewarm and the first request kernel
gives an unambiguous request window. That window spans 5.610 seconds and
contains 18 embedding launches: 11 prefill chunks plus seven decode steps.
Long-KV scalar softmax contributes 504 launches and 1.638 seconds; cached-exp
softmax contributes 144 launches and 150 ms. Small-N GEMV contributes 504
launches and 2.056 seconds. Synchronous D2H contains eleven 4 MiB final-hidden
copies—one per prefill chunk—and eighteen 303,872-byte logits copies. The
first ten chunk logits are dead outputs, but their removable GPU copy/head
work is only about 10 ms, below 0.2% of this request; it is not the next
priority. The next bounded hypothesis is extending or replacing the exact
long-prefill softmax without exceeding SM89 dynamic-shared-memory limits.

The checked-in full-capture CSV hashes for CUDA API, GPU kernel and memory
time are respectively
`20fbeb837d2724572e0e01569afa9213e395c9f4e1782bec320e275ca3a8be88`,
`2aaf70c01d703b5efd627e7b569a1ea1447a380320e567c5c91a72f0bca3e654`
and `706b9e5db3a9cfeb6ae1bdb641154088bc930ec2d5a41de4f4e6306cd991960e`.

## Initial chunked-prefill decision

**Decision at that stage: promote text-only 1,024-token chunked prefill and
the explicit exact long-decode softmax fallback for the tested single-request
SM89 BF16 cells.** The candidate passes exact complete trajectories, repeated
no-profiler timing, the complete 32,768-token service boundary, real image and
audio requests, typed invalid-request handling, Broker configuration, binary
custody and causal profile gates. The deployed binary SHA-256 is
`c8e06b416c040505a837e5b07c50dc1177ab40189a31fa247d4ddee18196dc90`.
Qwen3.8 remains down.

This result does not claim multi-request or continuous-batching performance,
video or speech generation, vLLM parity, exact softmax acceleration above
4,096 prefill columns, or transaction-counter MFU/BWU. The old 11,264 OOM is
historical evidence for full-prompt prefill, not the current service limit.

### Rejected long-prefill exp-cache extension

The next bounded candidate raised the exact FP32 numerator-cache prefill gate
from 4,096 to 11,264 columns. Its SM89 operator test compared the full BF16
output at 11,264 columns byte-for-byte with scalar softmax and passed. Quick
and decode trajectories also remained exact. Candidate binary SHA-256 was
`5b886f7b487883f145322870e90ecacc852aeb02e5f63491082760cec754f563`.

Matched no-profiler measurements exposed a shared-memory residency crossover:

| Prompt + 8 output | Scalar TTFT p50 | Extended cache TTFT p50 | Change |
|---:|---:|---:|---:|
| 5,120 | 1.0250 s | 1.0099 s | 1.48% faster |
| 6,144 | 1.5470 s | 1.5530 s | 0.39% slower |
| 7,168 | 2.1517 s | 2.1910 s | 1.82% slower |
| 8,192 | 2.8461 s | 2.9062 s | 2.11% slower |
| 11,264 | 5.4246 s | 5.9987 s | 10.58% slower |

The 5,120-only gain removes about 15 ms but does not survive the standard 8K
cell, while dynamic shared memory grows with KV length and progressively
reduces CTA residency. **Decision: revert.** The experimental selector,
policy helper and long-shape test were removed from the production source;
the structured `baseline-long-prefill-exp-cache-*` and
`candidate-long-prefill-exp-cache-*` records remain to close this branch.
The accepted `c8e06b41` service is restored and healthy; Qwen3.8 remains down.

### Promoted adaptive chunk size

Primary classification: **source/runtime graph**. A causal chunk of width
`C` computes scores for future positions inside that same chunk and masks them
afterward. Reducing `C` therefore removes deterministic attention work, but it
also increases chunk dispatches, synchronizing final-row copies and dead
intermediate LM-head evaluations. The candidate first changed the fixed chunk
from 1,024 to 512 without adding a new public selector.

The constant-512 ladder preserved every complete trajectory and improved the
target range, but found a clear length crossover:

| Prompt | 1,024-chunk TTFT p50 | 512-chunk TTFT p50 | Change |
|---:|---:|---:|---:|
| 2,048 | 203.196 ms | 191.307 ms | 5.85% lower |
| 4,096 | 586.865 ms | 571.149 ms | 2.68% lower |
| 8,192 | 2.8461 s | 2.7549 s | 3.20% lower |
| 11,264 | 5.4246 s | 5.3479 s | 1.41% lower |
| 12,288 | 6.4414 s | 6.3842 s | 0.89% lower |
| 16,384 | 11.4064 s | 11.4780 s | 0.63% higher |
| 24,576 | 25.3042 s | 25.5571 s | 1.00% higher |
| 32,760 | 45.4262 s deployed | 45.9741 s | 1.21% higher |

The final rule is one static partial evaluation of prompt length: text-only
prompts through 12,288 tokens use 512-token chunks, while longer prompts keep
the accepted 1,024-token chunks. It introduces no second runtime mode or
environment variable. The final binary repeats 5.357 s TTFT at 11,264 with
0.24% CV, 45.345 s at 32,760, 8.265 ms decode TPOT and the exact frozen
trajectories. Real PNG/WAV requests and all four typed contract probes pass.

The actual adaptive-binary report remains on the GPU host at
`/var/lib/agent-gpu-broker/profiles/omni-11264-adaptive-chunk.nsys-rep`, size
3,825,650 bytes, SHA-256
`05b710996ec9968b29769e563912e25a8fc933fc1cf93a752d8bc0f0dfb6afb6`.
Profiler timing is not admission evidence: tracing 23,811 request kernels
raises the measured TTFT to 7.446 s. Request-window kernel sums provide the
causal comparison against the 1,024-chunk profile:

- total summed kernel time falls from 5,141.495 to 5,070.062 ms;
- small-N GEMV falls by 141.323 ms, from 2,055.659 to 1,914.336 ms;
- scalar softmax grows by 27.532 ms and LM head grows by 7.507 ms because
  there are 22 prefill chunks instead of 11;
- the net 71.433 ms kernel reduction agrees with the 76.743 ms no-profiler
  11,264-token TTFT reduction.

The checked-in full-capture CUDA API, GPU-kernel and memory-time CSV hashes
are respectively
`6114a598536241b012dcd1ae73d2915ee68c2686cbcc27de30738a2bead865d5`,
`703c62d3381ccc958ec44e267dbdbfc80cae4e5d3d2db84483ecbb935b6ef237`
and `f420680e7639328a4d3be8b7a9e6e37df926a6937aee54a546f69182257e9241`.

**Decision: promote the 512/1,024 adaptive chunk policy for the tested
single-request SM89 BF16 text cells.** It passes exact trajectory, repeated
no-profiler, 32K capacity, multimodal isolation, typed interface, binary
custody and causal profile gates. The deployed binary SHA-256 is
`778066a34e7df2c5c300db6bc4286f546d8c877ffa07987ae8ecc89460c08178`;
`c8e06b41` is retained for rollback. Qwen3.8 remains down when unused.

### Promoted final-chunk-only logits

Primary classification: **source/runtime graph**. Every non-final chunk used
to run output normalization, synchronously copy its complete final hidden
tensor to CPU, upload the selected hidden row, run the LM head and copy logits
back to CPU. No consumer observed those logits; only the last chunk determines
the first generated token. The promoted path now ends intermediate chunks
immediately after their KV state is published. The final chunk still calls the
ordinary validated forward path, including the existing one-token graph rule
for a one-token tail, so cache, position and final-logit semantics are
unchanged.

| Workload | Adaptive TTFT p50 | Final-only TTFT p50 | Change |
|---:|---:|---:|---:|
| 2,048 + 32 | 191.307 ms | 183.692 ms | 3.98% lower |
| 4,096 + 32 | 571.149 ms | 539.671 ms | 5.51% lower |
| 8,192 + 8 | 2.7551 s | 2.6477 s | 3.90% lower |
| 11,264 + 8 | 5.3570 s | 5.1495 s deployed | 3.87% lower |
| 16,384 + 8 | 11.4457 s | 11.0517 s | 3.44% lower |
| 32,760 + 8 | 45.3449 s | 43.7842 s | 3.44% lower |

The 11,264 candidate stability screen passes 5/5 with 0.16% TTFT CV; the
deployed screen passes 3/3 with 0.09% CV. Every trajectory is byte-identical
to the accepted adaptive path. The 1K path remains 0.0801 s TTFT and
9.390 ms TPOT; deployed decode remains 8.276 ms TPOT. Real PNG/WAV requests
and all typed contract probes pass.

The first 75-second profile budget expired six seconds into the request and
is retained as `candidate-chunk-final-logits-11264-profile-timeout.json`; it
is a measurement-window failure, not model evidence. The valid 120-second
report remains at
`/var/lib/agent-gpu-broker/profiles/omni-11264-chunk-final-logits-v2.nsys-rep`,
size 3,775,568 bytes, SHA-256
`4ac44472c9e9c2b00ee5e78bf87d828fe677227109ddaaa669f9a4925304f984`.
Profiler timing is explanatory only.

At 11,264 tokens, the request-window profile changes exactly the predicted
edges:

- final-hidden D2H copies fall from 22 to 1 and logits D2H copies from 29 to
  8; corresponding hidden-row H2D copies also fall from 29 to 8;
- output-normalization and LM-head kernels fall from 29 to 8; LM-head kernel
  time falls from 19.775 to 5.385 ms;
- stream synchronizations fall from 30 to 9 and synchronous `cudaMemcpy`
  calls from 1,699 to 1,636;
- total summed GPU kernel time falls by 24.935 ms, from 5,070.062 to
  5,045.126 ms. The larger 207 ms no-profiler TTFT reduction includes the
  removed host synchronization/control gaps; profiler span is not used to
  quantify that effect.

The checked-in CUDA API, GPU-kernel and memory-time CSV hashes are respectively
`dc4fe602886949a2e98af500a334636cedbde0b2a03d4a653aa31687df3df51b`,
`182179afbbc1e9a603d134e363d882270141fb37ee8d908bfc5189189bee75ff`
and `37b149a20704e888ddbac206aa50f7207148399e5fb26f86518f12c8c89ee74e`.

**Decision: promote final-chunk-only logits for all chunked text-prefill
cells.** It passes complete-trajectory, repeated no-profiler, 32K capacity,
multimodal isolation, interface, binary-custody and causal-profile gates. The
deployed binary SHA-256 is
`ac8e2436d8aca552b713de5b0cb7ed1a12b320e510a4c5f1aae63d116a33cc17`;
`778066a3` is retained for rollback. Qwen3.8 remains down when unused.

### Promoted 4K–6K chunk specialization

After final-chunk-only logits removed the dominant per-chunk synchronization,
the chunk crossover was remeasured. A global 256-token candidate preserved all
trajectories but was not generally better:

| Prompt | Accepted TTFT p50 | Global 256 TTFT p50 | Change |
|---:|---:|---:|---:|
| 2,048 | 183.692 ms | 205.542 ms | 11.89% slower |
| 3,072 | 339.291 ms | 351.480 ms | 3.59% slower |
| 4,096 | 539.671 ms | 531.772 ms | 1.46% faster |
| 5,120 | 893.739 ms | 862.307 ms | 3.52% faster |
| 6,144 | 1.3988 s | 1.3756 s | 1.66% faster |
| 8,192 | 2.6477 s | 2.6455 s | unchanged |
| 11,264 | 5.1495 s | 5.2033 s | 1.04% slower |

The promoted static policy therefore uses 256-token chunks only for prompt
lengths 4,096 through 6,144, 512 for the other eligible prompts through
12,288, and 1,024 above 12,288. This is one closed shape policy, not a public
mode. Deployed 4K+8 TTFT is 532.223 ms with 0.06% CV, down 1.28% from the
accepted 539.121 ms. Deployed 2K, 8K and 11K checks repeat the accepted paths,
all trajectories remain exact, and typed contract probes pass.

Matched actual-binary 4K profiles are retained at
`/var/lib/agent-gpu-broker/profiles/omni-4096-chunk512-final-logits.nsys-rep`
(2,237,942 bytes, SHA-256
`1feb9151af6b9d9b2653bb001b51b9f22337d2c50f405208f29db882a140af2a`)
and `/var/lib/agent-gpu-broker/profiles/omni-4096-tiered-chunk.nsys-rep`
(3,227,581 bytes, SHA-256
`735b22918c182bb13a2ed4d1cb5e618cd35d41076d3c33ab1430ca5da8cdde85`).
Profiler timing is explanatory only.

The 256-token path raises request kernel count from 12,667 to 19,011, but
reduces summed kernel time from 569.149 to 547.199 ms. Cached-softmax time
falls by 35.892 ms, from 155.197 to 119.305 ms. Several GEMM algorithms change
at the smaller row shape and recover part of that saving; the final
no-profiler service result, not the profiler span, decides promotion.

The checked-in baseline CUDA API/kernel/memory CSV hashes are
`7d2f09aa5a162af278baf3697630ad70e7615848cf680bd612d681a6880987e2`,
`444deaedae6901add2ca3252df05154f545d452629e0be91cb7ab50856e6b4c8`
and `1b562114a40ba5bc22da6360adb3476bcb3aa76d931c9b3328a1cf2d7a9ad056`.
The candidate hashes are
`4272d3d14e5b49dc4f5bc2b4c87e3d00cfbcd2641d20f5e105765e9edf4d5ab0`,
`0f5dcb2a31f872db119377a020bd0bfe8966846ae347b205f0f7079440f08e31`
and `c18a0099704cf909d0932047db91ee6d17d9378795486bd8597c93054a02a367`.

**Decision: promote the 4K–6K 256-token specialization.** It passes exact
trajectory, repeated no-profiler, unchanged-cell, interface, binary-custody
and matched causal-profile gates. The deployed binary SHA-256 is
`0fc97f78403cc8e0974f2c48f2267fc5d57dbe8e459ab3af48dc86a6a0c4460a`;
`ac8e2436` is retained for rollback. Qwen3.8 remains down when unused.

#### Rejected 384-token midrange counterfactual

One final bounded search replaced the 256-token midrange chunk with 384.
Candidate binary SHA-256 was
`2b3cf381f977f68270924234a426da480b9c3223c843e30916538797dc5799b8`.
All trajectories remained exact. Relative to the promoted 256 policy, 384
changed 4K/5K/6K TTFT from 532.223/862.307/1,375.587 ms to
533.143/854.242/1,360.461 ms: 0.17% slower, 0.94% faster and 1.10% faster.

**Decision: revert.** It missed the preregistered all-cell direction gate and
does not improve the public 4K gradient point. Splitting another policy range
only for nonstandard 5K/6K cells would add more selection authority than the
measured benefit justifies. The no-profiler record is retained as
`candidate-chunk384-midrange.json`; production source remains unchanged.

#### Rejected long-context 512-token retry

After dead intermediate logits were removed, the earlier long-context chunk
crossover was retested rather than assumed. Candidate binary SHA-256 was
`90fd0627766addfcd253e110a95af8fcdf445a8378574f312e2dbef29f799192`.
The 16,384-token trajectory remained exact, but TTFT regressed from the
accepted 11.052 s to 12.346 s (11.7%). A 24,576-token request also completed
at 24.672 s, but the 16K regression already triggered the stopping rule, so a
32K trial was not spent.

**Decision: revert.** Removing dead logits did not make 512-token chunks
appropriate above the measured 12,288-token crossover. The structured record
is `candidate-long-chunk512-16k-24k.json`; production retains 1,024-token
chunks for longer prompts.

### Promoted prefill TMRoPE position cache

Primary classification: **source/runtime graph**. Within one prefill slice,
all 36 layers and both Q/K rotations consume the identical TMRoPE position
array. The accepted backend nevertheless allocated and synchronously uploaded
that array for every call. `APXINF_TMROPE_POSITION_CACHE_PREFILL=1` extends the
existing content-keyed decode cache to multi-token slices: one stream-ordered
allocation and asynchronous upload owns each distinct slice, and all 72
rotations reuse it. Unset or `0` retains the accepted per-call path; invalid
values fail closed. The base `APXINF_TMROPE_POSITION_CACHE=1` selector remains
the single parent gate.

This does not revive the rejected pre-chunk full-prompt cache blindly. Text
chunks bound the cached input to 1,024 tokens (12 KiB of positions), while the
real multimodal path is separately validated. Replaced buffers enqueue their
free on the same stream after all prior consumers.

| Workload | Tiered baseline TTFT p50 | Prefill cache TTFT p50 | Change |
|---:|---:|---:|---:|
| 1,024 + 32 | 80.439 ms | 76.877 ms deployed | 4.43% lower |
| 4,096 + 8 | 532.223 ms | 506.916 ms deployed | 4.75% lower |
| 11,264 + 8 | 5.152 s | 5.106 s deployed | 0.88% lower |
| 32,760 + 8 | 43.784 s | 43.491 s | 0.67% lower |

The candidate 11K stability screen passes 5/5 with 0.03% TTFT CV, and the
deployed 4K/11K screens pass with at most 0.04% CV. Decode TPOT remains
8.268 ms. Every text trajectory, the real PNG/WAV pair, the exact 32,768-token
service boundary and all typed contract probes pass.

The actual candidate profile remains at
`/var/lib/agent-gpu-broker/profiles/omni-4096-prefill-tmrope-cache-v2.nsys-rep`,
size 3,173,006 bytes, SHA-256
`dd32aec3da2351c8252b6875522e7b1e126255a64cac68e04624fe0b831c4fb6`.
Compared with the matched tiered-binary profile, request kernels remain
19,011 and summed kernel time remains effectively unchanged. The control/data
path changes exactly as predicted:

- 3,072-byte position H2D copies fall from 1,152 to 16;
- synchronous `cudaMalloc/cudaFree` calls fall from 1,182/1,183 to 30/31;
- synchronous `cudaMemcpy` calls fall from 1,198 to 46;
- 16 stream-ordered allocations and asynchronous uploads replace those
  per-chunk position transfers.

Profiler span is not admission evidence: traced `cudaMallocAsync` time is
noisy and larger in the candidate capture. The repeated no-profiler service
timing decides promotion. Candidate CUDA API/kernel/memory CSV hashes are
`773842af394e7afdf41e34c335756138fca40b4ac43b9909f826f90eb701a8f0`,
`75c1c9113edda6d43973f0843d0f1ee79c74f59531000e8f3501a2bfb81a4913`
and `bf9653b785030a3842ce3b8869c25b88c552a2fc371bdbf106987659efafc091`.

**Decision: promote prefill TMRoPE position reuse for tested text, image and
audio BF16 cells.** It passes complete-trajectory, repeated no-profiler,
32K-capacity, prior-OOM, multimodal, interface, binary-custody and causal
profile gates. The deployed binary SHA-256 is
`55283606f3ea88508e0bd9682c80c48edb13fa76fb638dad9ae1879d439e72ea`;
`0fc97f78` is retained for rollback. Qwen3.8 remains down when unused.

### Rejected stride-aware causal FA2 GQA prefill

Primary classification: **source/runtime graph with a vendored CUDA
operator**. The candidate added one bottom-right causal FA2 entry for
`head_dim=128`, GQA 16/2 heads and the existing head-major KV-cache strides.
It bypassed score materialization, exact-order softmax and the value GEMM under
`APXINF_FA2_GQA_PREFILL=1`; unsupported builds failed closed.

The first ordinary build, SHA-256
`1e927031894e1ed32710894b03327cbfa0a9e94b21a6e0f4372b7d96c1c9cc72`,
correctly rejected the selector because that target did not contain the SM89
FA2 conditional. An isolated explicit-SM89 target then produced binary
`6c28e72668404ead5c8014e53792c9d0a7684ee5fa5ef6cbcb92ead91a0eb1bd`.
The generic causal template was stopped after 15 minutes of compilation; a
single SM89 64x64/no-dropout/causal instance completed, while the existing
build still spent substantial time on unrelated FP16 FA2 instances. This
build cost is part of the candidate's maintenance evidence.

The explicit binary was fast but failed the predeclared complete-trajectory
gate at the first token:

| Workload | Accepted TTFT / hash | FA2 TTFT / hash | Result |
|---|---|---|---|
| 1,024 + 32 | 76.877 ms / `bf1da0…` | 55.761 ms / `ce1397…` | reject |
| 4,096 + 8 | 506.916 ms / `edc940…` | 298.365 ms / `00774e…` | reject |

The 1K first token changed from the frozen sequence's `1004` to `1003`; the
4K first token changed from `1016` to `1015`. This is consistent with the
known reduction-order sensitivity already exposed by the rejected cooperative
softmax, regardless of the large 27%/41% TTFT reductions.

**Decision: revert before long-context or profiler budget.** Exact model
behavior is the promotion boundary, so a faster stable but different
trajectory is not admissible. The unavailable-build and explicit-SM89 smoke
records are retained under `candidate-fa2-gqa-prefill*`; the C ABI, kernel
instance, selector and runtime branch were removed from production source.

### Promoted GPU final-hidden-row view

Primary classification: **source/runtime graph**. The final-logit path used to
normalize every row in the last prefill slice, synchronously copy that whole
hidden tensor to CPU, convert all rows to F32, select one row, rebuild BF16 and
upload the row to GPU before the LM head. `APXINF_QWEN25_GPU_LAST_ROW=1` now
creates a bounds-checked view of the final BF16 GPU row first, then runs the
same per-row RMSNorm and LM head. Logits still synchronize and return through
the accepted CPU boundary, so greedy selection semantics are unchanged. The
flag is CUDA-only, default-off and fail-closed.

The main 10-sample paired screens are exact and stable:

| Workload | Accepted TTFT p50 | GPU-row TTFT p50 | Change |
|---:|---:|---:|---:|
| 1,024 + 32 | 76.871 ms | 75.802 ms | 1.39% lower |
| 4,096 + 8 | 506.916 ms | 507.212 ms deployed | unchanged |
| 11,264 + 8 | 5.106 s | 5.110 s | unchanged |
| 32,760 + 8 | 43.491 s | 43.569 s hot retry | unchanged |

The first 32K candidate observation was a 44.863-second outlier; the immediate
same-service retry measured 43.569 seconds with the exact frozen trajectory.
The outlier and retry are both retained. Deployed 1K TTFT is 75.677 ms and
decode TPOT is 8.272 ms. Real PNG/WAV requests and all typed contract probes
pass exactly.

The actual candidate report remains at
`/var/lib/agent-gpu-broker/profiles/omni-4096-gpu-last-row.nsys-rep`, size
3,178,376 bytes, SHA-256
`1a748772d09fda64796e60090c2ed70b56b3aff414ccdd79cc5e10ea922e9625`.
Against the matched prefill-position-cache profile, request kernel count stays
19,011 and summed kernel time stays effectively unchanged. The data edges
change exactly as intended:

- one 1 MiB final-hidden D2H disappears;
- eight 4 KiB hidden-row H2D copies disappear, covering prefill plus seven
  graph-ineligible eager decode steps;
- seven 4 KiB eager hidden-row D2H copies also disappear;
- the eight full-logit D2H copies remain, preserving the accepted CPU greedy
  boundary for this path.

The candidate CUDA API/kernel/memory CSV hashes are
`21fc1c1067e27af9ddf56ef85ab721f52312563d6bd17a51998ecc7ac09fb669`,
`4e1e2590424d1e96ff0cc7bb4cd1959d81bae17485000babf859eb40df088d1e`
and `f5e5016aa832d697697535a30459525e192d058ec7ed5fc9b6c3b853732bdeba`.

**Decision: promote the GPU final-row view for tested BF16 text/image/audio
cells.** It passes exact trajectory, paired 1K materiality, unchanged long
cells, 32K capacity, multimodal, interface, binary-custody and causal-profile
gates. The deployed binary SHA-256 is
`bbf1b0b29396a546a55b3cb586fd4f4215438859790c805b322f78897159a201`;
`55283606` is retained for rollback. Qwen3.8 remains down when unused.

### Promoted eager GPU argmax

Primary classification: **source/runtime graph**. GPU token selection was
previously coupled to CUDA Graph eligibility, so positions at or above 3,072
fell back to ordinary forward, a 303,872-byte logits D2H and CPU greedy scan.
`APXINF_QWEN25_EAGER_GPU_ARGMAX=1` keeps the accepted ordinary long-KV layer
compute, creates logits from the GPU final-row view, and reuses the exact
128-block partial plus one-block final selector and mapped result already
owned by the decode workspace. It requires the existing GPU argmax and GPU
last-row selectors; unsupported combinations fail model load.

Matched no-profiler 32-output screens preserve complete trajectories:

| Prompt | CPU-selection TPOT p50 | Eager GPU TPOT p50 | Change |
|---:|---:|---:|---:|
| 4,096 | 12.616 ms | 12.390 ms | 1.80% lower |
| 8,192 | 14.778 ms | 14.562 ms | 1.47% lower |
| 11,264 | 20.967 ms | 20.742 ms | 1.07% lower |

The deployed service independently repeats 12.395/14.544 ms at 4K/8K; short
graph decode remains 8.245 ms. The 4K, 8K and 11K 32-token hashes match the
accepted binary exactly. Prefill, KV ownership, multimodal processing and the
combined-context contract are unchanged.

The actual 4K+8 candidate report remains at
`/var/lib/agent-gpu-broker/profiles/omni-4096-eager-gpu-argmax.nsys-rep`, size
3,178,646 bytes, SHA-256
`aaeadc34d938078d003ed0a430c3d2d02c1409902e882583b85d4d4786776609`.
Compared with the matched GPU-last-row baseline, seven decode-step full-logit
D2H copies disappear; only the prefill logits copy remains. Fourteen argmax
kernels are added—seven partial and seven final—and together consume only
0.036 ms. CUPTI kernel sum and span fluctuate upward under tracing, so the
matched no-profiler TPOT is the admission authority.

Candidate CUDA API/kernel/memory CSV hashes are
`67e7c936f8aa8163f8ce5ae28939088822ea2d43deead7214ecf2121e1f87b94`,
`240b8384a287ef0ebf9421b2cd57a22068cb929589b349e31ff19c8c33bd37fa`
and `c48631f7853b24465743a6e4eba96a91c30bbfd4aaddb87522d8112ee654545e`.

**Decision: promote eager exact GPU selection for graph-ineligible SM89 BF16
decode.** It passes exact complete trajectories, repeated no-profiler
materiality at three KV lengths, unchanged short decode, interface,
binary-custody and causal-profile gates. The deployed binary SHA-256 is
`767c36ad6b7d1d3f65b372dd5e47ba27a9fdac8a76c2cbce536109b7878d7d37`;
`bbf1b0b2` is retained for rollback. Qwen3.8 remains down when unused.

#### Rejected decode exp-cache shared-memory boundary

The historical 11,264-column decode numerator-cache limit was tested against
the apparent SM89 48 KiB shared-memory ceiling. Bit-exact operator candidates
at 12,286, 12,284 and 12,280 columns requested 49,144, 49,136 and 49,120 bytes
of dynamic shared memory. All three failed before arithmetic with CUDA error 1
(`invalid argument`), including attempts that reserved 16 and 32 bytes for the
compiler-aligned static reduction region.

**Decision: revert without model timing.** Source-visible arrays are not a
sufficient contract for the runtime's effective dynamic-shared allowance, and
further blind bisection was outside the stopping rule. Production retains the
verified 11,264 limit. The three Broker job receipts and requested byte counts
are recorded in `candidate-softmax-exp-cache-decode-boundary.json`.

### Promoted global exact numerator cache for long decode

Primary classification: **source/runtime graph with a custom CUDA operator**.
Shared-memory boundary
tests proved that the existing exact numerator cache could not simply grow
beyond 11,264 columns. `APXINF_SOFTMAX_GLOBAL_EXP_CACHE=1` therefore selects a
decode-only two-kernel path above that limit: the first kernel preserves the
scalar max order and writes each FP32 exponential once to a bounded global
workspace; the second preserves scalar column-order summation and normalizes
to BF16. Disabled mode retains the explicit scalar fallback.

The SM89 operator gate compares the complete BF16 output against scalar
softmax at 32,768 columns and passes byte-for-byte. Its Broker receipt is
`gpuq-b8f5e05cdb75`, recorded in
`candidate-global-exp-cache-operator.json`.

| Prompt / output | Scalar TPOT p50 | Global-cache TPOT p50 | Change |
|---|---:|---:|---:|
| 11,264 + 32 | 20.742 ms candidate reference | 19.422 ms deployed | 6.3% lower |
| 12,288 + 32 | 21.724 ms | 20.304 ms deployed | 6.5% lower |
| 32,760 + 8 | 41.075 ms | 37.892 ms | 7.8% lower |

All complete trajectories match exactly. The 32K candidate peaks at 16,061
MiB—identical to the accepted service—with 8,503 MiB headroom. Short decode
remains 8.268 ms and all typed contract probes pass.

The actual 11K+8 candidate profile remains at
`/var/lib/agent-gpu-broker/profiles/omni-11264-global-exp-cache.nsys-rep`, size
3,677,984 bytes, SHA-256
`93c7ad5ee80d51c33301399bfcf6431f65506d9bc1f2fce8deb96ac941ed65bc`.
Across seven decode steps, 252 scalar-softmax launches are replaced by 252
global-fill and 252 global-normalize launches. Relative to the closest
accepted scalar profile, that decode contribution falls from about 70.6 to
61.0 ms, or roughly 1.37 ms per generated token, matching the no-profiler
TPOT movement. Total profiled kernel span remains noisy and is not admission
evidence.

Candidate CUDA API/kernel/memory CSV hashes are
`aac0112e78d5215c76ac4e1df7357c1e48aee16b3730623e0d4964c6125f8b5c`,
`08d19177c56e497123f0bfda68a60a9403aa457f826af88b011262469e306568`
and `8cc3f464ed89481fd8977dc1bfa618ad0ce01ad8ea0035470ffefb446fa98cda`.

**Decision: promote the global exact numerator cache for tested long-decode
BF16 cells.** It passes 32K bit-exact operator, complete-trajectory, repeated
no-profiler, capacity/memory, interface, binary-custody and causal-profile
gates. The deployed binary SHA-256 is
`881491b0de93a73c7e77b050c11a83436cdb5a0beb8b99b72d7871c777c5c035`;
`767c36ad` is retained for rollback. Qwen3.8 remains down when unused.

### Promoted single-kernel global exp cache

Primary classification: **source/runtime graph with a custom CUDA operator**.
The first global-cache implementation launched a fill kernel and a separate
normalize kernel even though each decode row is owned by one CTA. The promoted
kernel performs parallel numerator fill, block synchronization, scalar-order
sum and parallel normalization in one CTA. It keeps the same bounded FP32
workspace and arithmetic contract while deleting one launch and preserving
numerator locality per layer.

The fused production-header operator independently passes the same 32,768-column
byte-exact scalar comparison under Broker job `gpuq-99792400416e`.

| Prompt / output | Two-kernel TPOT p50 | Fused TPOT p50 | Change |
|---|---:|---:|---:|
| 11,264 + 32 | 19.422 ms | 17.722 ms deployed | 8.8% lower |
| 12,288 + 32 | 20.304 ms | 18.475 ms deployed | 9.0% lower |
| 32,760 + 8 | 37.892 ms | 37.773 ms | 0.3% lower |

All trajectories and the 32K memory boundary remain exact. Short decode is
8.256 ms and typed contract probes pass.

The actual 11K+8 candidate report remains at
`/var/lib/agent-gpu-broker/profiles/omni-11264-fused-global-exp-cache.nsys-rep`,
size 3,682,215 bytes, SHA-256
`836fe2b93e15afb227d0642c83bbf859fde5f7743c355ba203a33489081757db`.
The request loses 252 kernel launches. The two global kernels' combined
60.975 ms becomes 49.604 ms in the fused kernel, while total summed kernel
time falls by 14.890 ms and the traced request window by about 123 ms.
No-profiler TPOT remains the admission authority.

Candidate CUDA API/kernel/memory CSV hashes are
`0c62da0853a6c285c6d003d94602ced16d3211989d8016643441e4d4d23a7e2c`,
`c6dbf0e802196492fd03d469a639162c5ec9d7b7f43421237fe56006b563ac17`
and `ed30f77d4eaa70d383e4e86d5e0dac496a441f6f4ed150965883378450e5b8a4`.

**Decision: promote the single-kernel global exact numerator cache.** It
passes fused-operator exactness, complete trajectories, repeated no-profiler
materiality, 32K capacity/memory, interface, binary-custody and causal-profile
gates. The deployed binary SHA-256 is
`321e8a6db1e932e5138170070ae4bd27183e42cd3143128c75d477f10ccaba5e`;
`881491b0` is retained for rollback. Qwen3.8 remains down when unused.

#### Rejected persistent global-exp workspace

The fused kernel still allocates one numerator buffer per layer/token. A
bounded candidate reused one thread-local, device/stream-keyed power-of-two
buffer (at most 2 MiB for this service) under
`APXINF_SOFTMAX_GLOBAL_EXP_WORKSPACE=1`. Candidate binary SHA-256 was
`1457cba891a015533247e727b2904f57c76454c349e1760de955bb09ae2a55f8`.

All trajectories remained exact, but 11K/12K TPOT moved only from
17.734/18.456 to 17.704/18.396 ms: 0.17% and 0.32%, below the 1% materiality
gate. **Decision: revert without profiler budget.** The added thread-local
state and cross-request lifetime were not justified by the saving. The raw
record is `candidate-global-exp-workspace-context.json`; production keeps
stream-ordered request-local workspaces.

#### Rejected 512-thread global-cache block

A final geometry candidate doubled the fused global-cache block from 256 to
512 threads. Binary SHA-256 was
`723fa0d70d19579c92ee799f6ef875c77c80f39bc93f2f3b0572073149af6bc9`,
and the 32K operator remained byte-exact under Broker job
`gpuq-b0154308b191`.

The 11K/12K TPOT medians moved from 17.722/18.475 to 17.657/18.348 ms:
0.36% and 0.69%, below the 1% materiality gate. **Decision: revert without
profiler budget.** Production keeps the 256-thread geometry; the structured
record is `candidate-global-exp-cache-512t-context.json`.

#### Promoted ordered lane-0 preload ILP

The late-stage admission rule now retains a small change when correctness and
interfaces are unchanged and repeated no-profiler measurements show a stable
positive direction; it no longer requires a 1% minimum. The candidate keeps
the fused global-cache block at 256 threads and preserves the exact scalar
maximum and sum order. Lane 0 only preloads four adjacent BF16 scores or FP32
numerators before applying the original four `fmaxf` or addition operations in
sequence. It adds no flag, workspace, lifetime or fallback state.

Candidate binary SHA-256 is
`0723831f1adb81e98232f75fa445818b9307a88af0eedbd30244b4fdc61f55b5`.
The 32K operator remained byte-exact under Broker job
`gpuq-d78f7ac200a7`.

| Prompt / output | Accepted TPOT p50 | Candidate repeat TPOT p50 | Change |
|---|---:|---:|---:|
| 11,264 + 32 | 17.727 ms | 17.670 ms | 0.32% lower |
| 12,288 + 32 | 18.457 ms | 18.420 ms | 0.20% lower |
| 32,760 + 8 | 37.773 ms historical | 37.719 ms | 0.14% lower, one trial |

The two candidate screens contain ten measurements per long-context cell in
total. Every candidate TPOT sample is below every sample in the adjacent
five-trial accepted repeat for both 11K and 12K. Candidate-repeat TPOT CV is
0.052% and 0.103%; all trajectories are exact. The 32K model trajectory keeps
SHA-256 `f5ef60ededd5770627b7963e24ff339aef60d63d061cafa37b7ee4e4b0598cb9`,
peaks at 15,965 MiB and retains 8,599 MiB headroom.

The attempted candidate and accepted 12K Nsight wrappers both omitted the
global-cache kernel that the prior formal runit capture proves is present in
the production path. Those wrapper captures are therefore path-mismatched and
are not used as attribution or admission evidence. This promotion relies on
the exact changed-operator gate, the prior production-path profile and the
repeated formal-runit no-profiler measurements; it does not claim a new
candidate-specific Systems attribution.

**Decision: promote the ordered lane-0 preloads under the late-stage stable-win
rule.** Raw evidence is
`baseline-fused-global-exp-cache-repeat-context.json`,
`candidate-global-exp-cache-lane0-ilp-context.json`,
`candidate-global-exp-cache-lane0-ilp-repeat-context.json` and
`candidate-global-exp-cache-lane0-ilp-32760.json`. Qwen3.8 and Omni remain down
when unused.

#### Rejected warp-staged ordered reductions

The next candidate used the first warp to coalesce groups of 32 global loads
into shared memory, while lane 0 consumed every value in the original scalar
order. It therefore preserved exact arithmetic but added two warp barriers and
shared-memory traffic per group. Binary SHA-256 was
`6b6b2b441f0242f2f7b9dec1c1e94ccdb2ddeda810e4b67c67f56a99097a88bf`;
the 32K operator was byte-exact under Broker job `gpuq-48fb826e90b9`.

The synchronization cost dominated: 11K TPOT moved from 17.670 to 17.890 ms
and 12K from 18.420 to 18.652 ms, regressions of about 1.25% and 1.26%.
**Decision: revert without profiler budget.** The raw record is
`candidate-global-exp-cache-warp-stage-context.json`; production retains the
ordered lane-0 preload implementation.

#### Rejected eight-value lane-0 preload

Doubling the ordered preload group from four values to eight preserved the
same scalar arithmetic and passed the 32K byte-exact operator under Broker job
`gpuq-05bf9838ea83`. Candidate binary SHA-256 was
`827ab7cd266ac9da8d6143e8456d5cc83e875ce349628b9225a2a1a0cac00cf1`.
The longer live range regressed 11K TPOT from 17.670 to 18.021 ms and 12K from
18.420 to 18.813 ms, about 1.99% and 2.14%. **Decision: revert without profiler
budget.** The raw record is `candidate-global-exp-cache-lane0-ilp8-context.json`;
production keeps four-value preloads.

#### Rejected two-value lane-0 preload

Reducing the ordered preload group from four values to two also passed the 32K
byte-exact operator, under Broker job `gpuq-69d136b04c7d`. Candidate binary
SHA-256 was
`59d5dadd809b494de2b5e95ed959966e92f82c7004f74679ac82db4db5afb937`.
Insufficient memory-level parallelism regressed 11K TPOT from 17.670 to
19.055 ms and 12K from 18.420 to 19.872 ms, about 7.85% and 7.89%.
**Decision: revert and close the preload-depth search.** The two-, four- and
eight-value variants establish four as the stable tested optimum. The raw
record is `candidate-global-exp-cache-lane0-ilp2-context.json`.

#### Promoted aligned pair-load reduction schedule

The next candidate retains the accepted four-value scalar arithmetic but gives
the compiler an explicit aligned access contract. On aligned rows it reads
BF16 scores through two `__nv_bfloat162` views and FP32 numerators through two
`float2` views, then applies `fmaxf` and addition in the original element
order. Row alignment is checked once; unsupported alignment takes the prior
four-value path, so shape coverage and failure semantics do not change.

Candidate binary SHA-256 is
`8c157d4193979246160e62ba1aa34f0cc6251d2d8b1c20930ed828e119281241`.
The 32K operator remained byte-exact under Broker job
`gpuq-dbdf0ed0791d`.

| Prompt / output | Four-value TPOT p50 | Aligned candidate repeat p50 | Change |
|---|---:|---:|---:|
| 11,264 + 32 | 17.670 ms | 16.960 ms | 4.02% lower |
| 12,288 + 32 | 18.420 ms | 17.642 ms | 4.22% lower |
| 32,760 + 8 | 37.719 ms | 36.452 ms | 3.36% lower, one trial |

Both five-trial candidate screens are trajectory exact and directionally
consistent. The repeat TPOT CV is 0.081% at 11K and 0.152% at 12K. The 32K
trajectory remains
`f5ef60ededd5770627b7963e24ff339aef60d63d061cafa37b7ee4e4b0598cb9`,
with the unchanged 16,061 MiB peak and 8,503 MiB headroom. The 1K, 128-token
decode, typed interface and real PNG/WAV gates all pass; the media outputs
match the frozen token references exactly.

`cuobjdump` reports 32 registers, zero stack, zero local memory and 16 bytes of
shared memory for both the accepted and candidate kernels. The embedded
`sm_52` SASS scalarizes the BF16 pair view but uses a different aligned load and
address schedule; it does not prove a native wide SM89 instruction. The fatbin
audit also shows that current production artifacts contain `sm_52` cubins plus
PTX and rely on driver JIT on the RTX 4090. Earlier uses of “SM89 binary” in
this document mean “validated on SM89 hardware”, not “contains a native
`sm_89` cubin”. An explicit `sm_89` build is the next separate candidate.

**Decision: promote the aligned pair-load schedule.** Raw evidence uses the
`candidate-global-exp-cache-aligned-vector-*` prefix. Qwen3.8 and Omni remain
down when unused.

#### Promoted native SM89 build contract

Primary classification: **ptxas configuration and deployment contract**. The
accepted source was rebuilt in the independent target
`/opt/apxinf/qwen25-omni-sm89-target-20260823a` with
`APXINF_CUDA_ARCH=sm_89`. `cuobjdump` verifies thirteen native `sm_89` cubins;
the prior artifact contained six `sm_52` cubins plus PTX and depended on driver
JIT on the RTX 4090. Candidate binary SHA-256 is
`0ab51980fe0544e11817cac1a11177a24ad9a021b577df87a7394aaf8d637a04`.
The native 32K softmax test remained byte-exact under Broker job
`gpuq-bed7a85bea43`.

| Prompt / output | PTX-JIT TPOT p50 | Native SM89 repeat p50 | Change |
|---|---:|---:|---:|
| 1,024 + 32 | 9.368 ms | 9.355 ms | 0.14% lower |
| 128 + 128 | 8.266 ms | 8.251 ms | 0.19% lower |
| 11,264 + 32 | 16.960 ms | 16.896 ms | 0.37% lower |
| 12,288 + 32 | 17.642 ms | 17.556 ms | 0.49% lower |
| 32,760 + 8 | 36.452 ms | 36.394 ms | 0.16% lower, one trial |

Both long-context candidate screens preserve exact trajectories and remain
directionally positive. Repeat TPOT CV is 0.095% at 11K and 0.056% at 12K.
The 32K trajectory remains
`f5ef60ededd5770627b7963e24ff339aef60d63d061cafa37b7ee4e4b0598cb9`,
with a 15,997 MiB peak and 8,567 MiB headroom. Typed HTTP probes and the frozen
real PNG/WAV token references also pass, covering the compile-time FA2 side
effect of the SM89 build.

The deployment gain has explicit costs. The first clean build took 18 minutes
6 seconds because architecture selection also compiled FA2, INT8 and Marlin;
the subsequent service link took 24.84 seconds. Binary size increased from
15,109,000 to 47,965,440 bytes. This architecture/optional-operator coupling
is a build-system limitation, not a runtime speedup claim. The checked-in
`build_sm89.sh` is now the reproducible 4090 build owner.

**Decision: promote the native SM89 artifact under the late-stage stable-win
rule.** Raw evidence uses the `candidate-native-sm89-*` prefix. The formal
binary is installed but Qwen3.8 and Omni remain down when unused.

#### Promoted SM89 core operator set

Primary classification: **build/runtime-graph specialization**. Explicit
`sm_89` previously implied compiling FA2, INT8 and Marlin and also emitted
their Rust cfg paths, even though this BF16 Omni service does not select those
operators. The build owner now accepts
`APXINF_CUDA_OPERATOR_SET=core|core-fa2|full`. Unset remains `full` for
compatibility. This stage used `core`; the later vision-FA2 promotion moves the
Omni build script to `core-fa2`. The operator set participates in the kernel
build ID.

The first clean core candidate SHA-256 is
`8f8b58d101f326f5da04fe7ae031cb7940f81f5a0cee1d7506e08a8c1bda9be2`.
The final rebuild from the formatted committed build owner is
`6b26aebcae3ac0d653cd0f60527a0636513e852bf5c69147ab4f9cecf40614fc`;
it repeats the exact quick trajectory at 9.353 ms TPOT and passes the typed
contract. The artifact contains six `sm_89` cubins and no FA2, Marlin or
CUTLASS-INT8 symbols.
The full and core global-softmax kernels have identical normalized machine
opcode hash `fa710f54d797463cae0cf8a76dc15763d92730347207df4820b100d5df4f33c5`
and identical 40-register, zero-local/stack, 16-byte-shared resource usage.
The 32K operator is byte-exact under Broker job `gpuq-3a6fe4fb5b59`.

| Prompt / output | Full SM89 TPOT p50 | Core SM89 TPOT p50 | Change |
|---|---:|---:|---:|
| 1,024 + 32 | 9.355 ms | 9.358 ms | 0.03% slower |
| 128 + 128 | 8.251 ms | 8.250 ms | unchanged |
| 11,264 + 32 | 16.896 ms | 16.934 ms | 0.23% slower |
| 12,288 + 32 | 17.556 ms | 17.558 ms | unchanged |
| 32,760 + 8 | 36.394 ms | 36.267 ms | 0.35% lower, one trial |

Every trajectory, typed HTTP probe and real PNG/WAV reference passes. The
largest repeated runtime movement is 0.23%, below the predeclared 0.5%
non-regression threshold. The 32K sample retains 9,079 MiB measured headroom.

The clean build improves from 1,086 to 61 seconds and the binary shrinks from
47,965,440 to 15,278,120 bytes. **Decision: promote the core operator set as
the Omni 4090 build contract.** Raw evidence uses the
`candidate-native-sm89-core-*` prefix. The `full` default remains available to
Qwen3.8 and other consumers that need the optional operators; Qwen3.8 and Omni
remain down when unused.

#### Rejected reciprocal normalization

A normalization candidate replaced each exact FP32 division with one lane-0
reciprocal and parallel multiplication, without adding a launch or barrier.
Candidate SHA-256 was
`47952d3fa391b78e4113fa3108df6a74ac20eee77b7a5608459b1ddde52d426b`.
It passed the original 32K sine gate under `gpuq-23a9db431b7c`, but a new
deterministic pseudorandom gate over 11,265, 12,288, 16,385 and 32,767 columns
found a BF16 mismatch at 32,767 columns, index 7,655: candidate
`5.47152e-9` versus scalar `5.500624e-9`.

**Decision: revert before end-to-end timing.** The restored exact-division
path passes the expanded gate under `gpuq-3a80edb0d2d5`. The concise
multi-boundary regression test remains in the suite; the structured record is
`candidate-global-exp-cache-reciprocal.json`.

#### Rejected two-way exact normalization

The next candidate preserved exact division but manually issued two
independent normalization elements per thread to expose division ILP. It
passed the expanded long-boundary exactness gate under `gpuq-8355b758eeed`;
candidate SHA-256 was
`2d54dd57a8506c4074b1ef984bcc1ce0ca17ec2449bbedc250c913271cdcaf40`.
At 11K TPOT moved from 16.934 to 16.916 ms, but 12K regressed from 17.558 to
17.614 ms, about 0.32%, above the candidate's measurement variation.

**Decision: revert without a post-hoc shape selector.** The uncertain 11K
gain does not justify another policy boundary. The raw record is
`candidate-global-exp-cache-normalize-ilp2-context.json`.

#### Rejected forced 32-bit score loads

An aligned candidate loaded two packed `uint32_t` values and reconstructed
their four BF16 elements, intending to replace four 16-bit score loads with
two 32-bit loads. Candidate SHA-256 was
`e049b9b905bb5e7685cee7a1899c34a87fd1621015236e08d4d64cf5133fa875`.
Native SM89 SASS retained exactly 66 `LDG.E.U16` and 82 32-bit `LDG.E`
instructions in both accepted and candidate kernels, with unchanged resources.
The detailed branch diff shows that the accepted aligned path already uses two
32-bit loads; the candidate only changed the unpack sequence. The U16 loads
belong to fallback and other phases, not scalarization of the aligned path.

**Decision: revert at the build/SASS gate without correctness or end-to-end
budget.** The intended load-width reduction was already present, so the Inline
PTX added no mechanism. The structured record is
`candidate-global-exp-cache-forced-u32-load.json`.

#### Promoted exact parallel prefill maximum

Primary classification: **source/runtime graph with a custom CUDA operator**.
The plain causal softmax used after the 4K shared-cache boundary left 255
threads idle while lane 0 scanned every score twice. The promoted kernel has
all 256 threads compute strided local maxima and combines them through a 1 KiB
shared-memory `fmaxf` tree. Lane 0 still evaluates every exponential and adds
them in the original order; final exponentials and divisions are unchanged.
For finite BF16 model scores, maximum selection introduces no rounding or
arithmetic-order difference.

Candidate SHA-256 is
`139ce5b622d3db62b494816233fba5347e470269009d57f8dabad040d939307c`.
The original exact comparison passes under `gpuq-91038615cb53`; a new
4,096-column pseudorandom prefill boundary comparison passes under
`gpuq-0b89eb90df2d`.

| Prompt / output | Accepted TTFT p50 | Parallel-max TTFT p50 | Change |
|---|---:|---:|---:|
| 4,096 + 8 | path-control only | 0.504 s | expected unchanged path |
| 11,264 + 32 | 5.100 s | 4.429 s | 13.16% lower |
| 12,288 + 32 | 6.115 s | 5.276 s | 13.73% lower |
| 32,760 + 8 | 43.405 s | 37.181 s | 14.34% lower, one trial |

The five-trial 11K/12K screens have TTFT CV 0.13% and 0.01%. TPOT remains
unchanged, every text trajectory is exact, typed interface probes pass and
the real PNG/WAV outputs match their frozen references. The 32K sample keeps
8,567 MiB measured headroom.

Interactive Nsight capture stopped and flushed collection before terminating
the service, so both reports include the 252 global-decode softmax launches.
At 12K the plain prefill softmax falls from 1.946 to 1.110 seconds, 42.98%
lower; profiler TTFT falls from 6.128 to 5.289 seconds, 13.69%, matching the
no-profiler result. The baseline and candidate reports remain on the host as
`omni-12288-current-sm89-core-interactive.nsys-rep` and
`omni-12288-prefill-parallel-max-interactive.nsys-rep`, SHA-256
`7d8b845fa2b67b0fef54c7317908808cf565bbd9cddd63a0520e02efc2a824d4`
and `4a57db47acbee2ccaa626c6dd8bf76cde75fb5b128227d22e3b38b8bf6a3a0ad`.
The Broker wrappers exit 143 only after the successful collection stop and are
not timing evidence.

**Decision: promote exact parallel maximum reduction for finite BF16 model
scores.** Raw request evidence uses the `candidate-prefill-parallel-max-*`
prefix; small profiler exports are checked in under `profiles/`. Qwen3.8 and
Omni remain down when unused.

#### Promoted tiled exact exponentials with ordered summation

The next candidate reuses the same 1 KiB shared array in 256-column tiles.
Every CTA lane computes one `expf` in parallel, then lane 0 consumes the FP32
values in original column order. This preserves the exact maximum, exponential,
summation and division contracts while eliminating serial transcendental work;
two CTA barriers are paid per tile.

The first binary, SHA-256
`285f683cf8b693452a9e408b46755108aebf1f047d7ffe0c45d3a930d3df2a97`,
missed a barrier between reading `max_values[0]` and reusing the array. One of
five 12K trials emitted all-zero token IDs. `compute-sanitizer racecheck`
reported 24 hazards under `gpuq-0f390357b1c6`. Adding the missing CTA barrier
produced final SHA-256
`bc05d88e0c04061b4562b7e599578576bcad018f1d579922039c8f65a1d3fea0`;
racecheck then reports zero hazards under `gpuq-c29bdaaf002c`.

| Prompt / output | Parallel-max TTFT p50 | Fixed tiled-exp TTFT p50 | Change |
|---|---:|---:|---:|
| 11,264 + 32 | 4.429 s | 3.788 s | 14.47% lower |
| 12,288 + 32 | 5.276 s | 4.497 s | 14.75% lower |
| 32,760 + 8 | 37.181 s | 29.968 s | 19.40% lower, one trial |

The fixed 12K binary produces one exact trajectory across ten consecutive
trials; 11K is exact across five. Short text, typed interface, real PNG/WAV and
32K gates all pass. The 32K sample retains 8,567 MiB measured headroom.
Relative to the native-core prefill state before both exact rewrites, TTFT is
now 25.73% lower at 11K, 26.46% lower at 12K and 30.96% lower at 32K.

Interactive profile attribution compares the fixed candidate against the
already-promoted parallel-max profile. Plain prefill softmax falls from 1.110
seconds to 342.3 ms, 69.15% lower; profiler TTFT falls from 5.289 to 4.525
seconds, 14.43%. The fixed report remains at
`/var/lib/agent-gpu-broker/profiles/omni-12288-prefill-tiled-exp-fixed-interactive.nsys-rep`,
size 3,666,730 bytes, SHA-256
`12eb8e789402559e5e13ac28433e5157f45ee1afe3162716edfe166b574540dc`.

**Decision: promote the race-free tiled exponential schedule.** The unsafe
iteration remains explicit in `candidate-prefill-tiled-exp-sum-racecheck.json`;
fixed raw evidence uses the `candidate-prefill-tiled-exp-sum-fixed-*` prefix.
Qwen3.8 and Omni remain down when unused.

#### Promoted post-softmax chunk retuning

After the exact softmax rewrites, small-N GEMM became the primary prefill
cost, so the old chunk crossover was remeasured. A 1,024-token candidate for
prompts through 12K reduced launches but increased masked attention work:
8K, 11K and 12K TTFT regressed by about 6%, 4% and 3%. Its SHA-256 was
`69e8ea618c549681b1eda351140bcdee809cf590649627ab10c5cf997a8f2690`;
the raw record is `candidate-chunk1024-retune-context.json`.

A global 256-token arm then improved 8K–12K but regressed the short range.
The final closed policy therefore uses 512-token chunks below 4,096 prompt
tokens, 256 from 4,096 through 12,288, and 1,024 above 12,288. Final SHA-256
is `7c52ad66e7dcfc4de27c1e46019facfee4533bbe72e712c83998dcf052be4b00`.

| Prompt / output | 512-chunk TTFT p50 | Final policy TTFT p50 | Change |
|---|---:|---:|---:|
| 2,048 + 8 | retained 512 path | 0.174 s | path control |
| 3,072 + 8 | retained 512 path | 0.325 s | path control |
| 8,192 + 8 | 2.023 s | 1.969 s | 2.65% lower |
| 11,264 + 32 | 3.788 s | 3.745 s | 1.15% lower |
| 12,288 + 32 | 4.497 s | 4.448 s | 1.10% lower |

Every cell has one exact stable trajectory; final 11K/12K TTFT CV is 0.18%
and 0.03%. Prompts above 12K retain the already-validated 1,024 path.

At 12K, interactive profile changes the leading small-N GEMM family from
2.342 seconds over 1,080 launches to 2.264 seconds over 2,088 launches; the
average launch duration roughly halves. Plain softmax grows from 342.3 to
363.5 ms because chunk count doubles, and other GEMM algorithms recover part
of the saving. Complete profile TTFT still falls from 4.525 to 4.497 seconds;
no-profiler timing remains the admission authority. The final report remains
at `/var/lib/agent-gpu-broker/profiles/omni-12288-chunk256-4k12k-interactive.nsys-rep`,
size 6,787,116 bytes, SHA-256
`d3c6b9c5f7d11e768c8ca4a59dc60e365a3ea9ae4cee1d0bab1e2bcff5c2e0ad`.

**Decision: promote the 512/256/1,024 post-softmax chunk policy.** Raw evidence
is `candidate-chunk256-retune-context.json`,
`candidate-chunk256-4k12k-context.json` and
`candidate-chunk256-4k12k-11k-12k.json`; small profiler exports are checked in.
Qwen3.8 and Omni remain down when unused.

#### Rejected 128-token chunk floor

The final chunk-size arm halved the 4K–12K chunk from 256 to 128. Candidate
SHA-256 was
`0c08eee73cfad643939805aa45da3b0f1603521a74ee9e9cf54dd5d8de5dc567`.
Exact trajectories remained stable, but TTFT regressed from 0.505 to 0.613
seconds at 4K, 1.969 to 2.114 seconds at 8K, 3.745 to 3.927 seconds at 11K
and 4.448 to 4.640 seconds at 12K.

**Decision: revert and close the chunk-size axis.** Across 128, 256, 512 and
1,024, the final 512/256/1,024 policy is the tested optimum. The raw record is
`candidate-chunk128-retune-context.json`.

#### Rejected explicit cuBLAS Tensor-Op default

The first library-selector counterfactual changed only BF16
`cublasGemmStridedBatchedEx` from `CUBLAS_GEMM_DEFAULT` (-1) to
`CUBLAS_GEMM_DEFAULT_TENSOR_OP` (99). Candidate SHA-256 was
`283cc91598113e4b4f3268058c9cd320521444205a73cd5b47e8401fb1cc3ee9`.
Trajectories stayed exact, but 8K TTFT regressed about 0.33% while 11K/12K
moved only about 0.10%/0.04% in the positive direction.

**Decision: revert as a mixed null result.** cuBLAS automatic selection remains
the owner; the raw record is `candidate-batched-algo99-context.json`.

#### Closed cuBLAS batched-algorithm enumeration

A production-shape ignored probe measured score GEMM with `m=8`, `k=128`,
`batch=256`, broadcast K and the real query/output strides. Algorithms -1,
99 and 100–115 were screened at 4K/8K/12K, then apparent windows were repeated
across 5K–12K and every 256 tokens around 7K. Initial 8K and 7,168-token
signals of 8% and 5% disappeared on the matched repeats; the final boundary
sweep differed by less than 0.02% between default and algorithm 105.

**Decision: remove the temporary algorithm parameter and ignored probe.** No
algorithm has a reproducible continuous KV-length advantage, so another
runtime selector is not justified. The structured evidence is
`candidate-cublas-batched-algorithm-probe.json`.

#### Promoted exact fused attention scaling

Primary classification: **source/runtime graph with a custom CUDA operator**.
For BF16 multi-token attention beyond the 4K shared-cache boundary, the score
scale tensor had no consumer other than plain softmax. The promoted path passes
the scale into that kernel and, on every score read, explicitly computes
`BF16(BF16_score * scale)` before converting back to FP32. This reproduces the
removed scale kernel's intermediate BF16 rounding exactly. F32, decode and
≤4K paths retain the separate operation.

Candidate SHA-256 is
`5ad5c4985d2b1cafdccf6afe4f94b83c0cea0efd52cb3f360f122358378533bf`.
Separate and fused paths match every BF16 output at 4,097, 8,192 and 12,288
columns under `gpuq-1ed1152d9bd9`.

| Prompt / output | Separate-scale TTFT p50 | Fused-scale TTFT p50 | Change |
|---|---:|---:|---:|
| 4,096 + 8 | 0.505 s | 0.505 s | unchanged control |
| 8,192 + 8 | 1.969 s | 1.887 s | 4.21% lower |
| 11,264 + 32 | 3.745 s | 3.551 s | 5.18% lower |
| 12,288 + 32 | 4.448 s | 4.214 s | 5.26% lower |
| 32,760 + 8 | 29.968 s | 27.989 s | 6.60% lower, one trial |

All text and real PNG/WAV trajectories are exact. At 12K the independent
scale kernel falls from 1,980 launches and 186.7 ms to 828 launches and 11.3
ms, removing 1,152 long-prefill materializations. Plain softmax remains nearly
flat, 363.5 to 362.2 ms, despite recomputing the exact scale on each read.
Profiler TTFT falls from 4.497 to 4.255 seconds, 5.37%, matching no-profiler
timing. The report remains at
`/var/lib/agent-gpu-broker/profiles/omni-12288-prefill-fused-scale-interactive.nsys-rep`,
size 6,658,585 bytes, SHA-256
`b5c3319499ba7897ee998ca65cc583534524ee389a6782837d2d164e0a4a0184`.

**Decision: promote exact scale-softmax fusion for BF16 long prefill.** Raw
evidence uses the `candidate-prefill-fused-scale-*` prefix; small profiler
exports are checked in. Qwen3.8 and Omni remain down when unused.

### Rejected dual-stream score overlap

An optimistic ceiling probe placed two independent production-shape GQA score
GEMMs on separate CUDA streams and handles. Across seven repeated trials, the
median dual-stream improvement was 1.87% at 4K, 0.20% at 8K and 0.13% at
12K. The long-context ceiling is smaller than the event, ownership and buffer
lifetime costs required by a production path.

**Decision: remove the temporary dual-stream probe and retain one inference
stream.** Structured evidence is in
`candidate-gqa-score-dual-stream-ceiling.json`.

## Promoted flattened long-prefill GQA

Primary classification: **source/runtime graph and data layout**. The former
prefill path submitted a strided batch of 256 independent `m=8`, `k=128`
score GEMMs for every KV head, even though every batch member shares the same
K matrix. At 12K, `gemmSN_TN` accounted for 54.7% of summed GPU kernel time.
The promoted path packs the query rows by KV head, flattens sequence and GQA
rows into one GEMM, keeps score and softmax rows KV-head contiguous, performs
the value GEMM in the same layout, then restores the model-owned output
layout with bounded 2-D device copies. KV-cache ownership and layout do
not change.

The path is deliberately restricted to BF16 multi-token prefill with
`kv_len > 4096` under the existing `APXINF_BATCHED_GQA_PREFILL=1` selector.
Decode, F32, short prefill and the selector-off reference retain their prior
implementations. Packed causal softmax derives each sequence position from
the packed row index; a direct CUDA test proves its unpacked BF16 output is
bit-exact to the standard row layout.

The repeated hot-cache operator ceiling was stable:

| KV length | Batched score | Flattened score | Speedup | Batched value | Flattened value | Speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 4,096 | 0.1168 ms | 0.0213 ms | 5.49× | 0.1185 ms | 0.0217 ms | 5.46× |
| 8,192 | 1.0723 ms | 0.0378 ms | 28.38× | 0.2339 ms | 0.0339 ms | 6.89× |
| 12,288 | 1.6064 ms | 0.0537 ms | 29.92× | 0.3565 ms | 0.0472 ms | 7.55× |

No-profiler endpoint measurements include the layout-copy cost:

| Prompt / output | Fused-scale TTFT p50 | Flattened-GQA TTFT p50 | Change | Candidate repeats |
|---|---:|---:|---:|---:|
| 8,192 + 8 | 1.887 s | 0.982 s | 47.9% lower | 5 |
| 11,264 + 8 | 3.541 s | 1.455 s | 58.9% lower | 3 |
| 12,288 + 8 | 4.211 s | 1.633 s | 61.2% lower | 5 |
| 32,760 + 8 | 27.989 s | 6.658 s | 76.2% lower | 3 |

TTFT CV is 0.11% at 12K and 3.49% at 32K. Decode TPOT remains within the
accepted path's variation: 17.57 ms at 12K and 36.21 ms at 32K. Every text
case reproduces its previously accepted complete trajectory hash, including
`f5ef60ededd5770627b7963e24ff339aef60d63d061cafa37b7ee4e4b0598cb9`
at the exact 32,768-token contract. Peak sampled 32K
memory remains 14,973 MiB with 9,591 MiB headroom. Typed invalid requests
recover cleanly, and the real PNG and WAV cases reproduce their complete
reference token sequences without fallback.

The candidate binary SHA-256 is
`23ec923e386425e69a5455517e16f9ac4c5378aa1a78c5d9eeadb2a288aa8d5e`.
Raw evidence uses the `candidate-flattened-gqa-*` prefix; the operator ceiling
is `candidate-gqa-flattened-gemm-ceiling.json`.

**Decision: promote flattened GQA for exact BF16 long prefill.** It removes
the dominant migrated bottleneck without a second KV representation, a new
runtime selector or a decode-path change.

## Promoted full-write output allocation

Primary classification: **source/runtime graph and data materialization**.
After flattened GQA, the 12K profile contained 41,615 device memset operations
using 144.7 ms of GPU time. The largest avoidable groups were the causal
attention outputs above 32 MiB (1,152 operations, 70.7 ms) and GEMM outputs
with `beta=0` (including 6,912 operations of 5,636,096 bytes, 21.9 ms).
These buffers were allocated through the zero-filling output helper and then
fully overwritten before any consumer could read them.

The promoted change selects the existing uninitialized stream-ordered helper
only at contracts with a complete producer write: general and causal softmax,
scaled long-prefill softmax, packed-GQA softmax, global-cache softmax, and
GEMM outputs whose call fixes `beta=0`. Buffer ownership, shape, lifetime,
CUDA stream, GEMM arithmetic and every kernel instruction remain unchanged.
Outputs with partial-write or accumulator semantics retain zero initialization.

No-profiler results against the flattened-GQA deployment are:

| Prompt / output | Zero-filled TTFT p50 | Full-write TTFT p50 | Change | Repeats |
|---|---:|---:|---:|---:|
| 1,024 + 32 | 76.36 ms | 74.18 ms | 2.85% lower | 5 |
| 8,192 + 8 | 0.982 s | 0.927 s | 5.64% lower | 5 |
| 12,288 + 8 | 1.633 s | 1.523 s | 6.75% lower | 5 |
| 32,760 + 8 | 6.658 s | 5.969 s | 10.35% lower | 3 |

The 128+128 decode TPOT is 8.260 ms and therefore unchanged. All cells retain
their complete accepted trajectory hashes; 12K TTFT CV is 0.13% and 32K is
0.19%. The exact 32,768-token contract still has 9,591 MiB minimum sampled
headroom. The 45-test CUDA operator suite, typed invalid-request recovery and
real PNG/WAV complete-token gates pass.

The matched Systems profile reports device memset time falling from 144.7 ms
to 49.6 ms and operations from 41,615 to 29,727. Profile TTFT falls from
1.693 s to 1.584 s (6.43%), agreeing with the no-profiler result. The baseline
report is
`/var/lib/agent-gpu-broker/profiles/omni-12288-flattened-gqa-interactive.nsys-rep`
(SHA-256 `bc55ee19384929e604414794b0b51f5aafe9759bc85599d2e6621844c7d73302`);
the candidate report is
`/var/lib/agent-gpu-broker/profiles/omni-12288-nozero-fullwrite-interactive.nsys-rep`
(SHA-256 `def09f7f22f319dd82838bf9603c6af9400a1429b824aaf54e676b46faeffbc9`).
Small CSV exports are checked in.

The deployed binary SHA-256 is
`fbad7e2359f0e34cfb95112f01dad00fffdc36ad1ea6c17dc63e2ed8217291f8`;
`23ec923e...88d5e` remains the immediate rollback artifact.

**Decision: promote full-write output allocation.** It removes redundant
memory traffic with no new selector, representation or execution path.

## Promoted pointwise full-write allocation

The next bounded pass applies the same ownership proof to ordinary pointwise
and layout operators. SiLU, GELU, add, bias-add, multiply, scale, RMSNorm,
LayerNorm, RoPE/TMRoPE/MRoPE, embedding and QKV-split kernels write every
legal output element before its first consumer. Their eager outputs now use
the existing uninitialized stream-ordered allocator. KV-cache, prefix reserve,
partial-write and accumulator contracts remain zero-initialized.

Against the first full-write deployment:

| Prompt / output | Prior TTFT p50 | Pointwise TTFT p50 | Change | Repeats |
|---|---:|---:|---:|---:|
| 1,024 + 32 | 74.18 ms | 71.49 ms | 3.63% lower | 5 |
| 8,192 + 8 | 0.927 s | 0.891 s | 3.84% lower | 5 |
| 12,288 + 8 | 1.523 s | 1.479 s | 2.90% lower | 5 |
| 32,760 + 8 | 5.969 s | 5.907 s | 1.03% lower | 3 |

The 32K improvement is larger than its 0.21% TTFT CV. The 128+128 decode TPOT
is unchanged at 8.258 ms, while long eager decode improves with the removed
pointwise clears. Every complete trajectory remains exact; the 45-test CUDA
operator suite, typed request recovery and real PNG/WAV gates pass. The 32K
capacity run retains 9,591 MiB minimum sampled memory headroom.

The matched 12K profile reduces device memset operations from 29,727 to 7,056
and memset GPU time from 49.6 ms to 12.4 ms. Profile TTFT falls from 1.584 s
to 1.526 s (3.67%), agreeing with no-profiler timing. The candidate Systems
report remains at
`/var/lib/agent-gpu-broker/profiles/omni-12288-nozero-pointwise-interactive.nsys-rep`,
SHA-256
`ccae673f613793d9b9d782bc8650d6c8519819912a75c9f6d5bbac9d613478fa`;
small CSV summaries are checked in.

The deployed binary SHA-256 is
`b07642c15372ed769bf2a6cde443df157cf6191c927e27bec474f4c4150e140b`;
`fbad7e23...217291f8` remains the immediate rollback artifact.

**Decision: promote pointwise full-write allocation.** It provides stable
positive gains across every measured context without adding a public mode.

## Promoted exact SM89 BF16 chunk tactics

Primary classification: **source/runtime graph and library tactic selection**.
The earlier 4K tuning probe did not match the promoted chunked workload. The
current probe adds the packed-QKV shape and measures the actual 256-row and
1,024-row chunks with cold L2. Repeated selections agree on rank 2 for packed
QKV and rank 1 for Gate/Up; the 256-row Down projection additionally selects
rank 2. Unsupported shapes, devices and the selector-off path stay on vendor
cuBLAS.

The model installs five exact records into the existing immutable TacticStore
before weight execution or graph capture. The CUDA mechanism prepares all
cuBLASLt plans and one fixed 32 MiB workspace at load time. The hot path only
does an exact key lookup; it never autotunes or mutates policy. The selector is
`APXINF_QWEN25_BF16_CHUNK_TACTICS=1`, accepts only `0` or `1`, and fails closed
unless the runtime is an RTX 4090 SM89.

The direct cold-L2 probes show these relevant operator signals:

| Shape | Selected rank | Speedup over vendor |
|---|---:|---:|
| M256 packed-QKV, N2560 K2048 | 2 | 1.079× |
| M256 Gate/Up, N11008 K2048 | 1 | 1.115× |
| M256 Down, N2048 K11008 | 2 | 1.034× |
| M1024 packed-QKV, N2560 K2048 | 2 | 1.047× |
| M1024 Gate/Up, N11008 K2048 | 1 | 1.052× |

Final-SHA no-profiler results against the pointwise deployment are:

| Prompt / output | Vendor TTFT p50 | Tactic TTFT p50 | Change | Evidence |
|---|---:|---:|---:|---|
| 1,024 + 32 | 71.49 ms | 70.59 ms | 1.25% lower | 3 formal repeats |
| 8,192 + 8 | 0.891 s | 0.882 s | 1.03% lower | 5 formal repeats |
| 12,288 + 8 | 1.479 s | 1.465 s | 0.94% lower | adjacent A/B, 5+5 repeats |
| 32,760 + 8 | 5.907 s | 5.869 s | 0.64% lower | 5 formal repeats |

The adjacent 12K candidate CV is 0.022%; every complete trajectory is exact.
The 128+128 decode path is unchanged at 8.260 ms TPOT. The 45-test CUDA
operator suite, typed invalid-request recovery, real PNG/WAV final-binary gate
and exact 32K capacity boundary pass.

In the 12K profile, the two dominant selected GEMM grids change from
`32x22x2` and `32x4x7` to `32x22x1` and `32x4x4`. Their aggregate leading
kernel family falls from 516.5 ms to 509.3 ms, while profile TTFT falls from
1.526 s to 1.504 s. The candidate report remains at
`/var/lib/agent-gpu-broker/profiles/omni-12288-bf16-chunk-tactics-interactive.nsys-rep`,
SHA-256
`b2495bce678d08eae08d28ab4b987d786ac61413bfd0f6f8d1f03ef351f85d3b`.

The deployed binary SHA-256 is
`44d8e31699faec2f5856c2799d26e725e8d1190e3a602f50a9be1505c4084680`;
`b07642c1...150e140b` is the immediate rollback artifact.

**Decision: promote exact SM89 BF16 chunk tactics.** The gain is small but
stable under the late-optimization rule, and the selector remains explicit,
bounded and reproducible.

## Promoted bounded in-place scaled softmax

Primary classification: **source/runtime graph and data ownership**. The
flattened-GQA score tensor has one consumer. For query chunks of at most 256
tokens, the promoted kernel scales each score once during the existing maximum
scan, stores the exact BF16-rounded value back into the owned score buffer,
and normalizes that same storage. Subsequent phases therefore avoid two
repeated scale-and-BF16 conversions, and the runtime removes one large
attention-output allocation/free per layer and chunk.

The direct operator gate compares the ordinary and in-place paths at 4,097,
8,192 and 12,288 columns and requires every BF16 output to match exactly. The
selector `APXINF_SOFTMAX_INPLACE_SCALE=1` is default-off and accepts only `0`
or `1`; production additionally requires flattened BF16 prefill,
`kv_len > 4096`, and `query_tokens <= 256`.

The first unrestricted iteration proved why the query-token gate is required:
12K improved, but the 1,024-token 32K chunk regressed from 5.869 s to 6.059 s
(3.2%). The bounded version restores 32K to the accepted non-mutating path.

Adjacent no-profiler evidence for the selected cells is:

| Prompt / output | Baseline TTFT p50 | In-place TTFT p50 | Change | Candidate CV |
|---|---:|---:|---:|---:|
| 8,192 + 8 | 0.88396 s | 0.87803 s | 0.67% lower | 0.083% |
| 12,288 + 8 | 1.46502 s | 1.43879 s | 1.79% lower | 0.152% |
| 32,760 + 8 | 5.86917 s | 5.85931 s | unchanged control | 0.294% |

Every complete trajectory is exact. The 46-test CUDA operator suite, 128+128
decode, typed invalid-request recovery, real PNG/WAV and exact 32K capacity
gates pass.

Systems shows 1,152 fewer `cudaMallocAsync` and `cudaFreeAsync` calls. The
in-place softmax kernel itself is 4.7 ms slower because of the score write,
but the two downstream 64x128 GEMM families fall from 105.9 ms to 79.3 ms,
and profile TTFT falls from 1.504 s to 1.473 s. The candidate report remains
at
`/var/lib/agent-gpu-broker/profiles/omni-12288-inplace-softmax-interactive.nsys-rep`,
SHA-256
`317948dda30b0a57e456efc35908f0bbd26314cf1969ebdb53cdd931f31e4f1a`.

The deployed binary SHA-256 is
`2a70b977ac4222634569dbee2406128adcfe93a48e7a09d3525aa8199ee649a8`;
`44d8e316...c4084680` is the immediate rollback artifact.

**Decision: promote the bounded in-place path.** It composes a stable 8K/12K
gain while excluding the measured 32K regression.

## Promoted shared-cache parallel maximum

Primary classification: **PTX/SASS-directed CUDA**. The shared-memory exact
numerator-cache softmax still assigned its maximum scan to lane 0 even though
all 256 threads were resident. The promoted kernel gives each lane strided
columns and reduces 256 local maxima through a fixed shared-memory tree. The
maximum value is unchanged for finite BF16 inputs; numerator generation,
lane-0 FP32 summation order, division and output layout are untouched.

The existing `APXINF_SOFTMAX_EXP_CACHE=1` selector remains the owner. No new
flag or dispatch branch is added. Direct tests require bit-exact output at the
257-column small case, the 4,096-column prefill boundary and the 11,264-column
decode limit.

No-profiler results against bounded in-place softmax are:

| Prompt / output | Prior TTFT p50 | Parallel-max TTFT p50 | Change | Repeats |
|---|---:|---:|---:|---:|
| 1,024 + 32 | 70.09 ms | 67.18 ms | 4.15% lower | 5 |
| 8,192 + 8 | 0.878 s | 0.822 s | 6.39% lower | 5 |
| 12,288 + 8 | 1.439 s | 1.386 s | 3.69% lower | 5 |
| 32,760 + 8 | 5.859 s | 5.765 s | 1.62% lower | 3 |

At 8K, decode TPOT additionally falls from 13.689 ms to 10.697 ms because the
shared cache remains active below its 11,264-column limit. The 128+128 and 32K
decode controls remain near 8.254 ms and 35.220 ms. All trajectories are exact;
the 47-test CUDA operator suite, typed request recovery, real PNG/WAV and full
32K capacity gates pass.

The 12K profile attributes the gain directly: 576 shared-cache launches fall
from 92.740 ms to 32.772 ms (64.7% lower), while the in-place long-softmax and
leading GEMM families are unchanged. Profile TTFT falls from 1.473 s to 1.414
s. The report remains at
`/var/lib/agent-gpu-broker/profiles/omni-12288-shared-cache-parallel-interactive.nsys-rep`,
SHA-256
`051459cdf75f93bb750dc6a1ac2247b99db4567924298918bb96eca10a9c67e2`.

The deployed binary SHA-256 is
`487915bac1df5c81bb9c754ba42944b0735f7b029c146b9177348ca731ea4af8`;
`2a70b977...e649a8` is the immediate rollback artifact.

**Decision: promote shared-cache parallel maximum.** It is exact, removes a
serial critical interval and improves every measured context cell.

## Promoted global-cache parallel maximum

Primary classification: **PTX/SASS-directed CUDA**. Above the 11,264-column
shared-memory limit, decode uses an exact global FP32 numerator cache. Its max
scan was still lane-0 sequential. The promoted version gives every lane
strided BF16 scores and reduces the maximum through the same 256-thread tree
as the shared cache. Numerator generation, aligned four-value lane-0 sum,
division and output layout are unchanged.

The existing `APXINF_SOFTMAX_GLOBAL_EXP_CACHE=1` selector remains the only
owner. Direct tests at long decode boundaries and 32K require exact BF16
agreement with the scalar reference.

Adjacent 12K+32 measurements isolate decode:

| Metric | Sequential max | Parallel max | Change |
|---|---:|---:|---:|
| TTFT p50 | 1.38236 s | 1.38211 s | unchanged |
| TPOT p50 | 16.581 ms | 12.941 ms | 21.95% lower |
| TPOT CV | 0.190% | 0.116% | stable |

At 32K+8, TPOT falls from 35.220 ms to 24.441 ms (30.61%) over three stable
candidate trials. The 11,264-column shared-cache control remains 12.633 ms.
All complete trajectories are exact; contract, real PNG/WAV and the 47-test
CUDA suite pass.

The 12K profile attributes the change directly: 252 global-cache launches
fall from 48.988 ms to 22.197 ms (54.69% lower), and profiled TPOT falls from
17.241 ms to 13.428 ms. The report remains at
`/var/lib/agent-gpu-broker/profiles/omni-12288-global-cache-parallel-interactive.nsys-rep`,
SHA-256
`ae7b53fb0a5676d630c810c9fa66dfdac1e48e9ddab7487e33d407acca26b4a6`.

The deployed binary SHA-256 is
`8feb945de67c7f5566dd5843cddbda559090424e0dfae073c403971af3435ae0`;
`487915ba...31ea4af8` is the immediate rollback artifact.

**Decision: promote global-cache parallel maximum.** It preserves the exact
decode contract and materially reduces long-context TPOT.

### Rejected flattened-GQA cuBLASLt tactics

The BF16 tuner was extended with the real flattened score/value shapes at
`M=2048`, `KV=12288`. The best cuBLASLt score tactic regressed 2.2%, while the
best value tactic improved only 0.95%. This is below a credible end-to-end
ceiling, so vendor cuBLAS remains the owner. Evidence is
`candidate-flat-gemm-cublaslt-rejected.json`.

### Rejected in-place normalization ILP2

A manual two-element normalization loop kept the in-place kernel at 28
registers with zero local/stack bytes and remained bit-exact, but regressed
8K/12K TTFT by 5.91%/6.55%. The compiler's original scalar-stride schedule is
therefore retained. Structured evidence is
`candidate-inplace-softmax-ilp2-rejected.json`.

### Rejected split-buffer FP32 numerator cache

A bit-exact follow-up split each FP32 numerator's raw bits across the consumed
BF16 score buffer and the BF16 output buffer. This avoided the failed 192 MiB
FP32 workspace and kept ordered summation in shared memory, but the added two
16-bit stores and two 16-bit loads were still more expensive than recomputing
`expf`. At 12K, TTFT regressed from 1.439 s to 1.514 s (5.23%) over five stable
trials.

**Decision: revert before profiling.** Structured evidence is
`candidate-split-exp-cache-rejected.json`; the recompute-based bounded
in-place path remains deployed.

### Rejected long-prefill global numerator cache

The next candidate stored every exact scaled FP32 softmax numerator so the
long-prefill kernel could avoid a second score read and `expf`. Direct tests at
4,097, 8,192 and 12,288 columns were bit-exact, but the 12K chunk required a
192 MiB workspace plus one FP32 write and two FP32 reads per valid element.
No-profiler 12K TTFT regressed from 1.523 s to 1.870 s (22.79%), with candidate
CV rising to 9.9%.

**Decision: revert before profiling.** The end-to-end rejection gate failed
decisively; production retains the recompute-based exact softmax. Structured
evidence is `candidate-prefill-global-exp-cache-rejected.json`.

## Promoted exact Gate/Up SiLU-Mul fusion

Primary classification: **higher-level graph rewrite plus exact CUDA fusion**.
The Qwen2.5-Omni MLP formerly materialized a BF16 SiLU tensor and launched a
second elementwise multiply. The promoted backend primitive consumes the
separate Gate and Up tensors and produces their product in one complete-write
kernel. Its CUDA implementation explicitly rounds SiLU to BF16 before
multiplication, matching the removed intermediate tensor bit for bit. Other
backends retain the ordinary `silu` then `mul` composition.

`APXINF_QWEN25_FUSED_SILU_MUL=1` selects the model path. Unset or `0` keeps
the prior composition and invalid values fail closed. A direct operator test
requires bit-exact agreement, and the complete 48-test CUDA suite passes.
Contract recovery, real PNG/WAV inputs and the full 32,760+8 boundary also
pass with the accepted trajectory hashes.

An ABBA test used the same candidate binary with only the selector changed,
five measured requests per service phase and one warmup per phase:

| 12,288 prompt + 8 output | Separate kernels | Fused kernel | Change |
|---|---:|---:|---:|
| TTFT mean | 1.38644 s | 1.37438 s | 0.87% lower |
| Client wall mean | 1.48111 s | 1.46961 s | 0.78% lower |
| TPOT mean | 13.021 ms | 13.063 ms | 0.32% higher |

The short eight-token TPOT delta is below the admission claim. An independent
128+128 control improves TPOT from 8.254 ms to 8.237 ms, while the admission
claim is the repeated prefill and complete-request reduction. Every phase has
one identical complete trajectory hash.

The 12K profile supplies causal attribution. Across 1,980 MLP invocations,
separate SiLU and multiply take 13.791 ms and 12.696 ms, or 26.487 ms total.
The fused kernel takes 16.060 ms, 39.37% less, removes 1,980 launches and
reduces total profiled GPU kernel time from 1,435.536 ms to 1,418.467 ms
(1.19%). The report remains at
`/var/lib/agent-gpu-broker/profiles/omni-12288-fused-silu-mul-interactive.nsys-rep`,
SHA-256
`7da3531fe366f3e090ca18a2ce41df2929b9178ebfe1b2c41c9b157efcfcce45`.

The deployed binary SHA-256 is
`2296e9b3010d8174902f7ec6b2ffb22f7a046121bb4beb21f364d67eb1468374`;
`8feb945d...3435ae0` is the immediate rollback artifact.

**Decision: promote exact Gate/Up SiLU-Mul fusion.** The profile proves the
removed work, ABBA proves a stable end-to-end prefill gain, decode does not
regress in the longer control, and all correctness/capacity gates pass.

The subsequent same-GPU external baseline is documented in
`VLLM_OMNI_BASELINE.md`. It changes the next optimization priority: ApxInf
wins short single-request decode, while vLLM-Omni wins long prefill and the
real image path. Long-prefill attention and the vision tower therefore outrank
another sub-percent pointwise candidate.

## Promoted indexed window vision attention

Primary classification: **source/runtime graph**. The frozen PNG processor
produces grid `[1,64,108]`, 6,912 raw patch tokens and 1,728 merged text
placeholders. The vision encoder has 32 attention blocks: four full-attention
blocks and 28 windowed blocks over at most 8×8 patches.

The prior grouped kernel assigned one CTA to every query/head and checked the
query/key group before the dot product, but its output phase still loaded and
multiplied V for all 6,912 keys. The window contract therefore retained nearly
full quadratic traffic. The promoted path builds ascending original key-index
lists for every processor-owned group, uploads them once through a CUDA-backend
cache keyed by the complete group vector, and visits only those keys.

`APXINF_VISION_GROUPED_SPARSE=1` selects the path. Unset or `0` retains the
full-scan grouped reference; invalid values and malformed plans fail closed.
The plan deliberately stores original key indices rather than a reordered
layout. The kernel preserves the old `key % 32` max/sum lane assignment, fixed
warp-reduction tree, and increasing-key V accumulation. A direct interleaved
group test at head dimension 80 is bit-exact, and the complete 50-test CUDA
operator suite passes.

Five no-profiler candidate requests after one smoke observation are stable and
retain the exact 16-token PNG trajectory:

| Real PNG, 1,760 + 16 | Prior accepted | Indexed windows | Change |
|---|---:|---:|---:|
| TTFT | 33.382 s | 6.733 s p50 | 79.83% lower / 4.96× |
| Client wall | 39.555 s | 12.845 s p50 | 67.53% lower / 3.08× |
| TPOT | 10.332 ms | 10.383 ms p50 | unchanged |

Candidate TTFT CV is 0.096% and wall CV is 0.587%. Real WAV output, 1K+32,
128+128, 12K+8, the full 32,760+8 boundary, typed request recovery and all
complete text trajectories remain unchanged. The final formal PNG observation
is 6.773 s TTFT and 12.864 s wall with exact tokens.

The causal profiles are decisive. In the accepted baseline, 32 vision
attention launches consume 32.933 s and 99.4% of GPU kernel time. With indexed
windows, the 28 grouped launches consume 0.503 s total (17.746 ms median per
launch), while the four unchanged full-attention launches consume 5.975 s.
The matched full launches imply grouped attention falls from 26.958 s to
0.503 s, a 98.13% reduction or 53.58×. Complete profiled GPU kernel time falls
from 33.146 s to 6.692 s (79.81%). The reports remain at:

- `/var/lib/agent-gpu-broker/profiles/omni-vision-png-1760-baseline-interactive.nsys-rep`,
  SHA-256 `f64f18205128280cd0781fd439608721609a2a7daf5d990c550891ee6cf74c10`;
- `/var/lib/agent-gpu-broker/profiles/omni-vision-grouped-sparse-1760-interactive.nsys-rep`,
  SHA-256 `f74c0b929ffb40f540cb06a6fdad3e68cb194c6438cccfdcfe6d50f18ffcf954`.

The deployed binary SHA-256 is
`432ac73ef573f36fa47b8c2112abc7f1b5b561a79816ed60f57c16f0d06adb18`;
`2296e9b3...1468374` is the immediate rollback artifact.

**Decision: promote indexed window vision attention.** It removes masked K/V
work without changing arithmetic order, produces a large repeated end-to-end
gain, and passes all text, multimodal, capacity and recovery gates. The four
full-attention vision blocks now own 89.3% of candidate GPU kernel time and are
the next bounded vision target.

## Promoted vision-only BF16 FlashAttention-2

Primary classification: **source/runtime graph with a vendored FA2 CUDA
operator**. The repository already vendored a non-causal SM80-family FA2
HeadDim96 instance capable of serving the vision encoder's actual head
dimension 80, but the Qwen2.5-Omni path explicitly excluded 80. A previous
causal text-FA2 candidate was fast but changed the first output token, so this
candidate is restricted to the four full-attention vision blocks behind a new
selector. Text attention and the 28 indexed window blocks remain unchanged.

`APXINF_VISION_FULL_FA2=1` is default-off and accepts only `0` or `1`. It
requires an SM80-family build carrying FA2 and otherwise fails closed. The
operator output passes the existing head-dim-80 CPU-reference tolerance; no
bit-exact intermediate claim is made because FA2 changes reduction order. The
promotion boundary remains stricter at model level: every repeated PNG, WAV,
text and 32K request must preserve its complete accepted token trajectory.
The final `core-fa2` carrier also keeps head dimension 128 on the custom
reference path and passes the complete 51-test CUDA operator suite.

The first mechanism screen used the existing `full` SM89 operator set. Five
repeated no-profiler PNG requests pass exact tokens:

| Real PNG, 1,760 + 16 | Indexed windows | Full vision FA2 | Change |
|---|---:|---:|---:|
| TTFT | 6.733 s | 0.762 s p50 | 88.67% lower / 8.82× |
| Client wall | 12.845 s | 6.864 s p50 | 46.34% lower / 1.86× |
| TPOT | 10.383 ms | 10.380 ms p50 | unchanged |

TTFT and wall CV are 0.533% and 0.562%. A three-repeat final-carrier screen
then reaches 0.763 s TTFT and 6.893 s wall with exact tokens. The final
isolated deployment observes 0.758 s TTFT and 6.959 s wall. WAV, 1K+32,
128+128, the
full 32,760+8 boundary and typed recovery remain exact and unchanged.

The formal carrier profile confirms four FA2 launches consume 7.696 ms total
(1.898 ms median), replacing 5.975 s of scalar full attention: a 99.87%
reduction or 776.37×. Complete GPU kernel time falls from 6.692 s after the
indexed-window promotion to 0.726 s (89.15% lower). The 28 indexed-window
launches are unchanged at 0.504 s and now own 69.4% of GPU kernel time. The
formal report remains at
`/var/lib/agent-gpu-broker/profiles/omni-vision-core-fa2-1760-interactive.nsys-rep`,
SHA-256
`fccdb95f0b490f8c0e609d8196ba375543d1844d8c32a290354c59d5b5a00518`.
The final vision-only-cfg rebuild has the same normalized FA2 plus indexed
window SASS hash, `75e5252b01135c8489d3088685d4a6321d6ed02eaddca97028d42305eb040d1f`,
as the profiled carrier. The subsequent head-dim-128 fallback addition is a
Rust-only dispatch/test rebuild and reuses those CUDA objects unchanged.

The original `full` build proved the mechanism but took 17 minutes 43 seconds
and produced a 48,191,288-byte binary because it also compiled FP16 FA2,
HeadDim256, INT8 and Marlin. The promoted `core-fa2` operator set retains only
the BF16 HeadDim96 instance. Its clean build takes 3 minutes 49 seconds; the
final vision-only-cfg rebuild takes 3 minutes 43 seconds; the final dispatch
and head128 regression rebuild produces a 21,603,888-byte binary. Other static
MQA/MHA entry points retain their native
build path, while unsupported vision FA2 configurations fail closed. The prior
`core` text-only artifact remains available but cannot enable full vision FA2.

The deployed binary SHA-256 is
`116242bac3452be382a3c0f2487aa31d331153fdac74022d92f0f486ef8fc1be`;
`b61731d6...2e3d5433` is the immediate rollback artifact.

**Decision: promote vision-only FA2 with the minimal `core-fa2` carrier.** It
passes the model-level trajectory gate that rejected text FA2, removes the
profiled critical path, and retains all text/capacity/recovery behavior. The
service wall is now dominated by the per-request external Python processor;
indexed window attention is the remaining GPU vision leader.

## Promoted persistent Omni processor

Primary classification: **control/runtime graph**. The accepted chat path
started a fresh Python interpreter, imported NumPy, SciPy, SoundFile, PIL,
Torch and Transformers, and rebuilt the same `AutoProcessor` for every image
or audio request. The preceding complete profile already isolated this
boundary: the final PNG service wall was 6.959 seconds while its model
TTFT plus decode interval was about 0.915 seconds. Another pointwise CUDA
change could not materially remove the remaining six seconds.

`APXINF_OMNI_PERSISTENT_PROCESSOR=1` now starts one CPU-only worker during
service initialization, waits for an explicit JSON ready handshake, and
reuses the pinned processor through a newline-delimited request/response
protocol. Request-owned media tensors still cross the same `.npy` boundary;
the model inputs, tokenization and CUDA path are unchanged. Unset or `0`
retains per-request worker construction, unsupported values fail closed, and
`/health` reports `processor_mode=persistent` only while the child remains
alive. The server is serialized, so this candidate deliberately adds no
parallel worker pool or scheduling state.

One warmup followed by five no-profiler requests produced the following
complete-trajectory result:

| Workload | Prior wall | Persistent wall p50 | Speedup | Model TTFT p50 | Model TPOT p50 | Wall CV |
|---|---:|---:|---:|---:|---:|---:|
| PNG, 1,760 + 16 | 6.959 s | 1.074 s | 6.48× | 0.752 s | 10.382 ms | 0.77% |
| WAV, 52 + 16 | 5.937 s | 0.151 s | 39.24× | 20.17 ms | 8.113 ms | 0.46% |

Every PNG and WAV request reproduced its full accepted token sequence and one
stable trajectory. The estimated non-model interval falls from 6.044 to 0.166
seconds for PNG and from 5.850 to 0.066 seconds for WAV, while the model
interval remains unchanged. This removes 97.25% and 98.87% of the respective
non-model overhead. The same worker PID was observed before and after the
request series.

Failure semantics are explicit. A syntactically valid request containing a
corrupt PNG returns HTTP 422 `unprocessable_media`; the worker remains
healthy, and the immediately following real PNG reproduces the complete
accepted trajectory. The recovery authority is
`results/candidate-persistent-processor-recovery.json`; raw timing and the
promotion decision are indexed by
`results/promotion-persistent-processor.json`.

The promoted SM89 binary SHA-256 is
`20165551d65748aa9fba00c06fe5d40d5126b88a69cb070b9c9230d5730b153d`.
The immediate rollback binary is
`116242bac3452be382a3c0f2487aa31d331153fdac74022d92f0f486ef8fc1be`.
No new Nsight capture was needed: the prior complete Systems report owns the
GPU/model interval, this Rust-only candidate reuses identical CUDA objects,
and no-profiler timing closes precisely the isolated host-side gap.

**Decision: promote the persistent processor for the tested serialized image
and audio chat cells.** It passes exact output and bad-media recovery gates and
delivers a large stable service-wall gain without changing model timing. The
formal binary and runit definition are updated but remain stopped while
Qwen3.8 owns the GPU. This result does not claim multi-request throughput,
video input, speech output or a native-Rust media processor.

## Promoted grouped variable-length vision FA2

Primary classification: **source/runtime graph**. After the persistent
processor and full-attention FA2 promotions, the 28 windowed vision blocks
still spent about 503.1 ms in the exact indexed one-warp kernel. The frozen
real PNG has 6,912 raw tokens partitioned into 112 nonempty windows: 104
windows of 64 tokens and eight edge windows of 32. The vendored HeadDim96 FA2
kernel already supports cumulative variable-length sequence offsets, so the
remaining work did not require a new attention algorithm or PTX carrier.
No authoritative SubCUDA checkout was configured, so no external case-library
transfer is claimed.

`APXINF_VISION_GROUPED_FA2=1` requires the accepted
`APXINF_VISION_GROUPED_SPARSE=1` plan, BF16 Q/K/V, head dimension at most 96,
nonempty contiguous groups and an SM80-family FA2 build. Unsupported settings
fail closed. The candidate uses the existing stable `group_indices`
permutation to pack Q/K/V, passes the cached cumulative offsets to one varlen
FA2 call per block, and scatters the output back to the model-owned token
order. Unset or `0` retains the exact indexed kernel. The four full-attention
blocks, audio encoder and text path are unchanged.

The minimal `core-fa2` build completed in 230 seconds. Its 21,698,600-byte
binary SHA-256 is
`942eea1b67eb173b0494dd6ec83d8b7559fc081612b0694de168de224bcda269`.
The immediate rollback is
`20165551d65748aa9fba00c06fe5d40d5126b88a69cb070b9c9230d5730b153d`.
The authoritative BF16/general operator filter passes 52/52, including the
new unequal/interleaved-group varlen FA2 comparison. A broader 86-test run
passes 84 and observes two unrelated FP8 GEMM failures because `core-fa2`
deliberately excludes that FP8 carrier; neither failure enters the BF16 Omni
contract.

One warmup plus five clean no-profiler real-PNG requests all reproduce the
complete accepted trajectory:

| Real PNG, 1,760 + 16 | Indexed window baseline | Grouped varlen FA2 | Change |
|---|---:|---:|---:|
| Model TTFT p50 | 0.752 s | 0.257 s | 2.92× faster |
| Client wall p50 | 1.074 s | 0.581 s | 1.85× faster |
| Model TPOT p50 | 10.382 ms | 10.388 ms | unchanged |
| Wall CV | 0.77% | 1.43% | stable |

WAV remains a non-target control and reproduces its exact trajectory with
20.12 ms TTFT and 8.09 ms TPOT. The 1K+32 text control is stable at 64.92 ms
TTFT and 9.37 ms TPOT; 32,760+8, the typed protocol gate and malformed-media
recovery also pass. A preliminary run overlapped an externally owned root
Rust build and is retained under `candidate-grouped-varlen-fa2-*`, but is not
the timing authority. The clean formal files use the
`formal-grouped-varlen-fa2-*` prefix.

The complete Nsight Systems report remains on the GPU host at
`/var/lib/agent-gpu-broker/profiles/omni-vision-grouped-varlen-fa2-1760-interactive.nsys-rep`,
size 744,425 bytes, SHA-256
`c6ed2dc9e120397573423f3b7ff8f42eff1a97dde93ba95f6beb6f3e967d2a9c`.
No indexed grouped kernel appears. Across 28 blocks, Q/K/V packing consumes
2.421 ms, varlen FA2 consumes 1.655 ms, and output restoration consumes 0.327
ms: 4.403 ms total versus the prior 503.099 ms, a 114.3× interval reduction.
The unchanged four full-image FA2 launches consume 8.013 ms. This critical
path movement explains the approximately 495 ms TTFT reduction.

Against vLLM-Omni 0.26.0 on the same real PNG, ApxInf now records 0.581 s wall
versus 0.565 s and 0.257 s TTFT versus 0.232 s. The 2.8% wall difference is
near parity rather than a general winner; ApxInf retains a 2.11× TPOT
advantage on this cell.

**Decision: promote grouped variable-length FA2 for the tested single-image
SM89 BF16 cell.** The strict selector, operator suite, complete trajectories,
repeated E2E gain, controls, recovery and causal profile all pass. The formal
binary and runit selector are updated while the service remains stopped. This
does not claim other image geometries, multi-request throughput, video or
speech output.

## Promoted short-KV CUDA Graph decode candidate

Primary classification: **source/runtime graph**. The accepted Qwen2.5-Omni
BF16 decode layer loop is captured once at model load and replayed with
caller-owned workspaces, mapped token/TMRoPE/cache-position controls and the
existing KV cache. It deliberately preserves the accepted separate Q/K/V
biases, TMRoPE arithmetic, Gate and Up projections, SiLU, multiply and
residual order. No weights are packed or duplicated.

`APXINF_QWEN25_DECODE_GRAPH=1` is default-off and accepts only `0` or `1`.
Enabling it on a non-CUDA backend or failing graph construction fails model
load closed. The model selects graph replay only for one-token decode with
`start_pos < 2048`; all prefill and longer-KV decode use the accepted ordinary
path. The service remains single-request, BF16 and greedy under the frozen
contract.

### Rejected unrestricted graph and resulting selector

The first graph candidate, binary SHA-256
`fd1b29d4ea089cc013cee1325038e273efb53d21d481eb14dc60da23390d5820`,
used graph replay at every decode position. It preserved every tested token
trajectory but regressed long-KV TPOT even though TTFT remained unchanged:

| Workload | Accepted TPOT | Unrestricted graph TPOT | Change |
|---|---:|---:|---:|
| 4,096 + 8 | 13.432 ms | 14.319 ms | 6.6% slower |
| 8,192 + 8 | 15.620 ms | 19.649 ms | 25.8% slower |
| 10,752 + 8 | 17.023 ms | 23.055 ms | 35.4% slower |

This disproved a general graph policy. The bounded candidate therefore keeps
graph replay only below position 2,048. Raw unrestricted results use the
`candidate-decode-graph-preliminary-*` prefix; the first bounded screens and
same-window pair use `candidate-decode-graph-short-*` and
`paired-decode-graph-*`. They are retained as negative and iteration evidence,
not presented as the promotion binary.

### Clean build and binary custody

The final source was rebuilt from zero under the independent target
`/opt/apxinf/qwen25-omni-decode-graph-target-20260822a`. Before the build, the
remote CUDA source was synchronized to the locally reverted tree and searched
for the rejected exact-order packed-SwiGLU implementation. As a binary check,
`strings` finds 50 `swiglu_bf16_exact` occurrences in the preliminary linked
image and zero in the clean image. The promoted binary SHA-256 is
`a8cb1be3e697a96642d0500e096b0d6749eb1adb482bf65f8f6f8793e81aa217`.
The previous accepted binary is retained on the host as
`apxinf-accepted-softmax-5a131f72` for rollback.

### Complete-trajectory and failure-semantics gates

| Workload | Result | Trajectory SHA-256 | Accepted agreement |
|---|---:|---|---:|
| 1,024 prompt + 32 output | 5/5 stable | `bf1da0a151446e7ff757474de26ceb13ced3cf9a422fcc17b1f4699fb89d38ea` | exact |
| 128 prompt + 128 output | 5/5 stable | `a892eb11d69ed4a408e680432ebee0de04a202950ba7c5072731a001c753f039` | exact |
| 4,096 prompt + 8 output | 7/7 stable | `edc940b80ff945971996ddf2b30534773258bb41d3d1812c38f36ba06eaabcbd` | exact |
| 10,752 prompt + 8 output | 1/1 capacity probe | `19478a6e232ab7479a2a8026f01096ab68a16f72d922be9695d807928125b02d` | exact |
| 11,264 prompt + 8 output | HTTP 503 CUDA OOM | n/a | same first failure |
| Post-OOM 1,024 + 32 | 3/3 stable | `bf1da0a151446e7ff757474de26ceb13ced3cf9a422fcc17b1f4699fb89d38ea` | exact |

After the expected OOM, resident memory returns to approximately 12.34 GiB,
`/health` stays successful and all three immediate recovery requests pass.
The graph workspace does not reduce the proven 10,752-token capacity or move
the first failure.

Real multimodal input is also covered by complete output IDs. A 1,760-token
PNG chart request produces the exact 16-token baseline sequence and a
52-token WAV request produces the exact nine-token baseline sequence; both
report `fallback_active=false`. Their single-observation TPOT changes from
12.934 to 10.846 ms and from 10.813 to 8.721 ms respectively. These are
correctness and path-coverage observations, not independent performance
admission samples. The structured record is
`results/candidate-decode-graph-clean-multimodal.json`.

### No-profiler end-to-end result

The final B/A/B window used the clean graph binary, then the previous accepted
binary, then the deployed clean binary. The table compares five-sample clean
and accepted screens; the deployed three-sample repeat independently matches
the clean result.

| Workload | Metric | Previous accepted | Clean graph | Ratio / change |
|---|---|---:|---:|---:|
| 1,024 + 32 | wall p50 | 0.4408 s | 0.3894 s | 1.132× faster |
| 1,024 + 32 | TPOT p50 | 11.605 ms | 9.922 ms | 1.170× faster |
| 128 + 128 | wall p50 | 1.3935 s | 1.1381 s | 1.224× faster |
| 128 + 128 | TPOT p50 | 10.809 ms | 8.799 ms | 1.228× faster |
| 4,096 + 8 | TTFT p50 | 0.7297 s prior stable | 0.7296 s | unchanged |
| 4,096 + 8 | TPOT p50 | 13.432 ms prior stable | 13.445 ms | unchanged |
| 10,752 + 8 | TTFT, one trial | 7.8803 s | 7.8869 s | unchanged |

Clean quick and decode wall-time CVs are 0.38% and 0.20%; TPOT CVs are 0.09%
and 0.18%. The seven-sample graph-ineligible 4K screen has 0.64% TTFT CV. The
deployed run repeats 9.913 ms quick TPOT and 8.800 ms decode TPOT with exact
trajectories. Against the original baseline, decode TPOT improves from
17.567 ms to 8.799 ms (1.997×), 4K TTFT improves from 18.3407 s to 0.7296 s
(25.138×), and 1K wall time improves from 2.7737 s to 0.3894 s (7.123×).

### Actual clean-binary attribution

The actual promoted binary was profiled under Broker ownership. The exact
128+128 trajectory passes under profiling at 8.874 ms TPOT; profiler timing is
not used for admission. The report remains on the GPU host at
`/var/lib/agent-gpu-broker/profiles/omni-decode128-graph-clean-v2.nsys-rep`,
size 718,472 bytes, SHA-256
`4a6c4ece0139605e71eea53bf3cce706dc06050632be6f552ff838cc7f71b141`.

The CUDA API summary contains 127 `cudaGraphLaunch` calls, one for every
decode step after the first token produced by prefill. They consume 8.628 ms
of host API time in total, with a 57.2 microsecond median. The 131
`cudaStreamSynchronize` calls consume 1.091 s in total, averaging 8.330 ms;
the remaining critical interval is therefore GPU graph execution and its
per-token synchronization, not graph-launch dispatch. Graph instantiation is
a one-time 2.096 ms load cost. The 2,709 ordinary `cudaLaunchKernel` calls are
from model load, prewarm and prefill; CUPTI kernel summaries undercount work
replayed inside the graph, so they are not used to rank graph nodes. The
checked-in API, kernel and memory CSV hashes are respectively
`4482a2039711e9706622be532d41a55f78e70b8df755a44d300939c94f145058`,
`d085136dd97530efd04e2b2047d61938bb57b4642bfd1af715dd3673849256bd`
and `4f68de1366aa11d696100b8af128992b26117071a987d7138a66c3300aace4bc`.

## Short-KV decode-graph decision

**Decision: promote the start-position-bounded CUDA Graph, composed with the
accepted softmax, batched GQA, stream-ordered allocation and decode TMRoPE
cache, for the tested single-request BF16 cells.** It passes clean-build and
binary-custody checks, exact complete trajectories, repeated no-profiler
timing, long-context capacity, real image/audio input, OOM recovery,
explicit-service configuration and actual-clean-binary profile gates. The
Broker-owned RTX 4090 service runs the promoted SHA-256; Qwen3.8 remains down.

This result does not claim a graph win at or above position 2,048,
multi-request or continuous-batching performance, video or speech generation,
vLLM parity, a larger OOM boundary, or MFU/BWU. The following experiment
resolves its bounded logit-readback and CPU-selection question.

## Promoted two-stage GPU token selection

Primary classification: **source/runtime graph**. The graph-only service
copied one 151,936-element BF16 logit row (303,872 bytes) to the CPU after
every short-KV decode step, converted it to F32, cloned it once more and ran a
strict-`>` greedy scan. Existing `APXINF_PERF=1` instrumentation measured
8.68 ms/token in forward, 0.109 ms/token in CPU argmax and effectively zero in
the callback. In the actual graph-only Nsight request window, 126 observable
D2H transfers total 38,287,872 bytes, each `cudaMemcpy` call averages
49.2 microseconds, graph launch averages 67.9 microseconds and stream
synchronization averages 8.581 ms. This bounded the removable host selection
surface at about 0.23 ms/token.

`APXINF_QWEN25_GPU_ARGMAX=1` now selects an exact SM89-only path and requires
`APXINF_QWEN25_DECODE_GRAPH=1`; missing flags preserve the CPU path, invalid
values fail closed, and a non-SM89 request fails model load. The selector is
used only for one-token decode with `start_pos < 2048`. It performs 128
independent CTA reductions into a 1 KiB persistent workspace, then one final
CTA writes the lowest-index maximum to a four-byte host-mapped result. Equal
values, signed-zero ties and NaNs follow the canonical CPU scan. A fast-path
error is propagated instead of silently falling back to another forward pass.

### Rejected one-block iteration

The first end-to-end candidate used one 256-thread CTA for the full vocabulary.
Its binary SHA-256 was
`149b2b268bd10e99151a47896530fe54665ec926905511e6666a2f9024b58aa8`.
It preserved every quick, decode, 2,040-boundary and 4K trajectory and removed
decode-step D2H, but failed the predeclared 1% materiality gate in the main
decode cell:

| Workload | Graph-only TPOT | One-block TPOT | Improvement |
|---|---:|---:|---:|
| 1,024 + 32 | 9.943 ms | 9.844 ms | 1.00% |
| 128 + 128 | 8.794 ms | 8.716 ms | 0.89% |

Its actual report remains on the GPU host at
`/var/lib/agent-gpu-broker/profiles/omni-decode128-gpu-argmax-oneblock.nsys-rep`,
size 706,140 bytes, SHA-256
`b55c55144e90df655a683eed7b631cf5a8c68d3c1cc61251c605c1b62b0698a0`.
The observable eager argmax kernel takes 143.1 microseconds and decode stream
synchronization averages 8.737 ms. **Decision: continue, do not promote the
one-block implementation.**

The production-header microbenchmark then compared both exact kernels over
1,000 full-vocabulary iterations with the same host-mapped output. One block
takes 83.855 microseconds; the two-stage path takes 5.607 microseconds
(14.96× faster), and both select the expected lowest index 17. This operator
evidence justified one final end-to-end iteration but did not itself promote
it.

### Correctness, boundary and failure gates

| Workload | Result | Trajectory SHA-256 | Graph-only agreement |
|---|---:|---|---:|
| GPU operator: ties, signed zero, NaN, full vocab | pass | n/a | exact CPU contract |
| 1,024 prompt + 32 output | 10/10 stable | `bf1da0a151446e7ff757474de26ceb13ced3cf9a422fcc17b1f4699fb89d38ea` | exact |
| 128 prompt + 128 output | 12/12 stable | `a892eb11d69ed4a408e680432ebee0de04a202950ba7c5072731a001c753f039` | exact |
| 2,040 prompt + 16 output | 3/3 stable | `50857ada86058d751e96a27b9f943025c41da1b2b8c856812e45f1ab00024498` | exact |
| 4,096 prompt + 8 output | 3/3 stable | `edc940b80ff945971996ddf2b30534773258bb41d3d1812c38f36ba06eaabcbd` | exact |
| 10,752 prompt + 8 output | 1/1 capacity probe | `19478a6e232ab7479a2a8026f01096ab68a16f72d922be9695d807928125b02d` | exact |
| 11,264 prompt + 8 output | HTTP 503 CUDA OOM | n/a | same first failure |
| Post-OOM 1,024 + 32 | 3/3 stable | `bf1da0a151446e7ff757474de26ceb13ced3cf9a422fcc17b1f4699fb89d38ea` | exact |

The 2,040 case crosses the selector boundary during generation: positions
below 2,048 use graph selection and later positions use the ordinary path.
Its exact trajectory proves mixed-path KV ownership. After the expected OOM,
the service remains healthy, resident memory returns to about 12.3 GiB and
the next three requests pass. A real 1,760-token PNG request and 52-token WAV
request reproduce the complete graph-only output IDs with no fallback. The
structured multimodal record is
`results/candidate-gpu-argmax-two-stage-multimodal.json`.

### No-profiler end-to-end result

The stable candidate screens are compared with the nearest stable graph-only
screens; all performance rows are client/service timing without a profiler.

| Workload | Metric | Graph only | Two stage | Ratio / change |
|---|---|---:|---:|---:|
| 1,024 + 32 | wall p50 | 0.3893 s | 0.3820 s | 1.019× faster |
| 1,024 + 32 | TPOT p50 | 9.915 ms | 9.699 ms | 1.022× faster |
| 128 + 128 | wall p50 | 1.1385 s | 1.1122 s | 1.024× faster |
| 128 + 128 | TPOT p50 | 8.794 ms | 8.589 ms | 1.024× faster |
| 2,040 + 16 | TPOT p50 | 11.632 ms | 11.549 ms | 0.7% faster |
| 4,096 + 8 | TTFT p50 | 0.7497 s | 0.7490 s | unchanged |
| 4,096 + 8 | TPOT p50 | 13.438 ms | 13.429 ms | unchanged |

Candidate quick and decode TPOT CVs are 0.19% and 0.05%. The deployed service
repeats exact trajectories at 9.706 ms quick TPOT and 8.580 ms decode TPOT.
Against the original baseline, decode TPOT improves from 17.567 ms to
8.589 ms (2.045×); the accepted 4K TTFT improvement remains 25.138×.

### Actual promoted-binary attribution

The deployed binary SHA-256 is
`ba1d82933b68506832c75ed63bf7c95d07f865329402bfe97293f84240577944`.
The actual-candidate report remains at
`/var/lib/agent-gpu-broker/profiles/omni-decode128-gpu-argmax-two-stage.nsys-rep`,
size 706,500 bytes, SHA-256
`cb0c61cbc927606c4192256dbd683450a40b3062adc672c149330e3a7fdf9e96`.
Its exact profiled trajectory reports 8.676 ms TPOT; profiler timing is not an
admission result.

The request window contains 127 graph launches and zero logits D2H transfers.
Graph-launch API time averages 64.1 microseconds and the 127 stream
synchronizations average 8.609 ms. CUPTI undercounts graph-replayed nodes, but
the eager prewarm records the partial and final argmax kernels at 2.688 and
2.400 microseconds. The checked-in API, kernel and memory CSV hashes are
respectively
`133df8f574feec8b6d657e8187f730601a577c0cec46d67fe5e3880f9048ea73`,
`1f1ea33f02f591c8700fd8ad2cdb0d1a849db1de6135136a3006c9191137da4e`
and `e571f3aa0676c78ad08b8cc6ff43bb70536c9026df8e6e4e037f16a7c68e5794`.

## GPU token-selection decision

**Decision: promote two-stage exact GPU token selection, composed with the
accepted short-KV CUDA Graph, softmax, batched GQA, allocator and TMRoPE
cache, for the tested SM89 single-request BF16 cells.** It passes exact
operator semantics, complete trajectories, mixed-boundary behavior, repeated
no-profiler materiality, long-context/OOM recovery, real image/audio input,
explicit Broker configuration and actual-binary profile gates. Qwen3.8
remains down.

This result does not claim benefit at or above position 2,048, multi-request
or continuous-batching performance, non-SM89 portability, video or speech
generation, vLLM parity, a larger OOM boundary, or MFU/BWU. The serial greedy
loop now has no full-logit D2H during short decode; the next bounded target
must come from GPU graph compute rather than host token selection.

## Promoted 3,072-position graph crossover

The initial 2,048 selector was a conservative boundary chosen after the
unrestricted graph regressed 4K and longer decode. GPU token selection changed
the short-graph cost, so the crossover was remeasured rather than assumed.
The graph-only deployment was compared with a 4,096 candidate over four
32-output context cells:

| Prompt | Ordinary TPOT | Graph TPOT | Graph change |
|---:|---:|---:|---:|
| 2,048 | 11.959 ms | 11.069 ms | 1.080× faster |
| 2,560 | 12.250 ms | 11.729 ms | 1.044× faster |
| 3,072 | 12.484 ms | 12.387 ms | 0.8% faster |
| 3,584 | 12.793 ms | 13.096 ms | 2.4% slower |

The 4,096 selector was rejected because it admitted a repeatable 3,584-token
regression. The final constant is 3,072: positions below it use graph replay
and exact GPU selection, while positions 3,072 and above use the ordinary
path. The deployed result reproduces all four reference trajectories:

| Prompt | Deployed TPOT | Relative to ordinary | Path |
|---:|---:|---:|---|
| 2,048 | 11.058 ms | 1.081× faster | graph |
| 2,560 | 11.737 ms | 1.044× faster | graph |
| 3,072 | 12.501 ms | unchanged | ordinary |
| 3,584 | 12.798 ms | unchanged | ordinary |

A 3,064 prompt + 16 output request crosses the new boundary within one
generation and reproduces the ordinary reference trajectory SHA-256
`a23d632e4073ed3d8f11890a011e3e4be520644827d54a25044d82c96f224675`.
The final binary also repeats the accepted 1K and 128+128 trajectories at
9.707 and 8.582 ms TPOT. Its SHA-256 is
`e29b62bfd035c280cf8342e77e0efe1f45e0194696e55ac4f9a0e3cf90daad93`;
the prior 2,048 binary is retained as `apxinf-accepted-gpu-argmax-ba1d8293`.

**Decision: promote the 3,072 crossover.** It adds no API mode, workspace,
kernel or weight representation; it only widens the already accepted graph
selector through the last cell with material benefit and excludes the first
sub-threshold and regressing cells.

### Post-promotion GPU compute attribution

To expose graph-replayed nodes, the deployed arithmetic was profiled once
with `APXINF_NO_GRAPH=1` for seven decode steps. This preserves weights,
workspace, KV, kernels and two-stage selection while paying ordinary launch
overhead; it is attribution evidence only. The exact request trajectory is
`649905ec73afb193907d2f0439834dd96975d7af84958d5f3fa85a0a84bbba63`.
The report remains at
`/var/lib/agent-gpu-broker/profiles/omni-decode8-eager-attribution.nsys-rep`,
size 990,741 bytes, SHA-256
`c978720395669116f9cfd253d06a4e6683efd80735e9533d0393d03832e7250b`.

Six complete device-visible steps average 8.905 ms of GPU kernel time:

| Node group | Time / token | Share / interpretation |
|---|---:|---|
| Gate projection | 1.886 ms | large BF16 weight stream |
| Up projection | 1.885 ms | large BF16 weight stream |
| Down projection | 1.914 ms | large BF16 weight stream |
| LM head | 0.664 ms | 151,936-vocab weight stream |
| Q + O projections | 0.820 ms | two dense attention projections |
| K + V projections | 0.243 ms | small, under-utilized GEMVs |
| Attention | 0.330 ms | not the dominant short-KV node |
| Norm/RoPE/cache/bias/residual/activation | 1.153 ms | remaining elementwise state path |
| Two-stage argmax | 0.005 ms | no longer material |

The exact BF16 text-weight lower bound is 6.172 GB/token from the frozen model
shapes. At the accepted 8.589 ms no-profiler TPOT this is about 719 GB/s, or
71.3% of the RTX 4090's 1,008 GB/s peak. This is a weight-only lower-bound BWU
estimate, not a measured memory-transaction ratio. Gate/Up/Down already imply
about 850 GB/s each, consistent with the rejected Gate/Up packing result. The
next bounded exact-semantics candidate is asymmetric Q/K/V packing: it may
amortize the two poorly utilized K/V GEMVs without reducing precision or
revisiting the closed MLP branch.

The checked-in eager-attribution API, kernel and memory CSV hashes are
respectively
`bfae6eb0407e66cd038191bcf6ba157d6e220b11c68cd437c22544a0e498cdd4`,
`1b6d1242ea1890e301cdb047dd84e2a658c8d45d8966d816fc722bab6fcc9754`
and `1bde091904f872d8cadd6f13b91ae1c2d4e077006cfdab61cdc34434b56d567e`.

This result does not claim graph benefit at or above position 3,072,
multi-request or continuous-batching performance, non-SM89 portability,
video or speech generation, vLLM parity, a larger OOM boundary, or a
transaction-counter MFU/BWU measurement.

## Promoted single-owner packed QKV

The expanded decode DAG showed 0.411 ms/token in Q, 0.121 ms in K and
0.122 ms in V. The two small K/V GEMVs were inefficient enough that an
asymmetric packed projection had a measurable exact-semantics upper bound.
`APXINF_QWEN25_PACKED_QKV=1` now chooses one packed `[hidden, hidden+2*kv]`
weight and bias owner per layer. Short graph decode uses one packed GEMV and
one packed bias operation with zero-copy buffer views. Ordinary decode and
prefill use the same packed GEMM, followed by one model-neutral unequal-width
BF16 GQA split/bias kernel. Unset or `0` retains separate Q/K/V weights;
invalid values, non-SM89 use and use without the graph fail closed.

### Ownership and migration ladder

The first graph-only upper-bound probe deliberately retained both separate and
packed weights. Candidate SHA-256
`dd0fabf90248b151203303633cf9c79979faac44c7e1680a9f3e1825484e2756`
raised resident memory from about 12,274 to 12,636 MiB. It preserved exact
quick/decode trajectories and measured 9.504 ms quick TPOT and 8.366 ms
decode TPOT. **Decision: useful upper bound, forbidden from promotion because
weight authority was duplicated.**

The first single-owner migration used three strided submatrix GEMMs against
the packed row layout. It reduced resident memory to 12,276 MiB and kept the
short win, but regressed ordinary 3K–4K TPOT by roughly 10–14%. Candidate
SHA-256 was
`751005ae2418d84d0e445a1a054bfa090f7b1c15830ca29372abb9e5a824b407`.
**Decision: reject the strided adapter, retain the single-owner requirement.**

The final migration consumes each separate layer while creating one packed
layer, synchronizes the two packing copies before releasing their sources,
and stores exactly one enum variant. Its resident 12,276 MiB proves that the
362 MiB probe duplicate is gone. The deployed binary SHA-256 is
`93e3a9bed77bc55eb341580798439e34283fe68e1acd9100f013d4e64f31e37b`;
the prior accepted binary is retained as `apxinf-accepted-cutoff3072-e29b62bf`.

### Correctness, capacity and multimodal gates

| Workload | Result | Trajectory SHA-256 | Prior agreement |
|---|---:|---|---:|
| Unequal-width GQA split/bias operator | pass | n/a | BF16 reference |
| 1,024 prompt + 32 output | 7/7 stable | `bf1da0a151446e7ff757474de26ceb13ced3cf9a422fcc17b1f4699fb89d38ea` | exact |
| 128 prompt + 128 output | 7/7 stable | `a892eb11d69ed4a408e680432ebee0de04a202950ba7c5072731a001c753f039` | exact |
| 2,048–4,096 prompt + 32 output | all stable | four frozen hashes | exact |
| 10,752 prompt + 8 output | 1/1 capacity probe | `19478a6e232ab7479a2a8026f01096ab68a16f72d922be9695d807928125b02d` | exact |
| 11,264 prompt + 8 output | HTTP 503 CUDA OOM | n/a | same first failure |
| Post-OOM 1,024 + 32 | 3/3 stable | `bf1da0a151446e7ff757474de26ceb13ced3cf9a422fcc17b1f4699fb89d38ea` | exact |

Packed prefill's extra temporary projection does not move the context
boundary, and recovery returns to the healthy 12.3 GiB resident state. Real
PNG and WAV inputs reproduce the complete accepted token sequences with
`fallback_active=false`; their structured record is
`results/candidate-packed-qkv-fused-split-multimodal.json`.

### No-profiler end-to-end result

| Workload | Metric | Prior accepted | Packed QKV | Ratio / change |
|---|---|---:|---:|---:|
| 1,024 + 32 | TPOT p50 | 9.707 ms | 9.485 ms | 1.023× faster |
| 128 + 128 | TPOT p50 | 8.582 ms | 8.363 ms | 1.026× faster |
| 2,560 + 32 | TPOT p50 | 11.737 ms | 11.523 ms | 1.019× faster |
| 3,072 + 32 | TPOT p50 | 12.501 ms | 12.138 ms | 1.030× faster |
| 3,584 + 32 | TPOT p50 | 12.798 ms | 12.386 ms | 1.033× faster |
| 10,752 + 8 | TPOT, one trial | 17.148 ms | 16.719 ms | 1.026× faster |

The stable short candidate screens have exact trajectories and sub-0.2% TPOT
CV. The deployed service independently repeats 9.485 and 8.361 ms TPOT. The
4K TTFT screen remains approximately 0.73 s. Against the original baseline,
decode TPOT improves from 17.567 to 8.363 ms (2.101×). The updated weight-only
effective bandwidth is about 738 GB/s, or 73.2% of the declared 1,008 GB/s
peak.

### Actual promoted-binary attribution

The exact profiled 128+128 trajectory reports 8.453 ms TPOT; profiler timing
is not used for admission. The report remains at
`/var/lib/agent-gpu-broker/profiles/omni-decode128-packed-qkv-fused-split.nsys-rep`,
size 697,561 bytes, SHA-256
`21c0b13c2972cd9cbeaec1e47d46e97b7befa0da0c35e63059967b7d30c6082e`.

The request window contains 127 graph launches, zero logits D2H and 8.396 ms
average stream synchronization, down from 8.609 ms before QKV packing. In the
observable eager prewarm, one packed QKV GEMV takes roughly 17.7 microseconds
per layer versus about 19.7 microseconds for separate Q/K/V; one packed bias
takes about 1.2 microseconds versus about 3.5 microseconds for three biases.
The checked-in API, kernel and memory CSV hashes are respectively
`3a9323f59c2f61c182a27eea5cbbbf4919858872186ab4289def9822ad0d6c8d`,
`a2c540ab6d19e3e2b2f16c51e9bcfafb68b773fefc91881a8bab4b8d5ac34b51`
and `ef9f3d18af3e358c3d7884210a90638ac1ca67893984bf780328c7cf50dff790`.

## Packed-QKV decision

**Decision: promote the single-owner packed-QKV layout and fused unequal GQA
split/bias adapter for the tested SM89 single-request BF16 cells.** It passes
operator, ownership, exact trajectory, repeated no-profiler, graph/ordinary,
long-context/OOM recovery, real multimodal, explicit Broker configuration and
actual-binary profile gates. Qwen3.8 remains down.

This result does not claim multi-request or continuous-batching performance,
non-SM89 portability, video or speech generation, vLLM parity, a larger OOM
boundary, or transaction-counter MFU/BWU. Remaining latency is dominated by
the three near-bandwidth-roofline MLP projections; the previously rejected
Gate/Up packing branch remains closed.

## Promoted fused TMRoPE K/V publication

`APXINF_QWEN25_FUSED_TMROPE_KV=1` replaces three short-graph nodes per layer:
K TMRoPE materialization, rotated-K cache append and unchanged-V cache append.
One decode-only kernel evaluates the identical per-dimension T/H/W rotation
and writes rotated K plus BF16 V directly into the caller-owned cache slot. Q
keeps its independent TMRoPE output because attention consumes it. Unset or
`0` preserves the three-node path; invalid values, non-SM89 use and use
without the graph fail closed.

The direct CUDA operator test compares the complete K and V cache allocations
byte-for-byte against separate TMRoPE plus two cache appends, including
non-equal T/H/W positions and a nonzero cache slot. It passes exactly. The
model screens preserve all frozen trajectories:

| Workload | Result | Trajectory SHA-256 | Prior agreement |
|---|---:|---|---:|
| 1,024 prompt + 32 output | 10/10 stable | `bf1da0a151446e7ff757474de26ceb13ced3cf9a422fcc17b1f4699fb89d38ea` | exact |
| 128 prompt + 128 output | 12/12 stable | `a892eb11d69ed4a408e680432ebee0de04a202950ba7c5072731a001c753f039` | exact |
| 2,048–4,096 prompt + 32 output | all stable | four frozen hashes | exact |
| 10,752 prompt + 8 output | retry passes | `19478a6e232ab7479a2a8026f01096ab68a16f72d922be9695d807928125b02d` | exact |
| 11,264 prompt + 8 output | HTTP 503 CUDA OOM | n/a | same first failure |
| Post-OOM 1,024 + 32 | 3/3 stable | `bf1da0a151446e7ff757474de26ceb13ced3cf9a422fcc17b1f4699fb89d38ea` | exact |

The first 10,752 observation had 8.458 s TTFT despite being graph-ineligible;
an immediate independent retry measured 7.930 s and did not reproduce the
regression. Capacity, OOM recovery and resident memory remain unchanged. Real
PNG and WAV requests reproduce their complete accepted token IDs with no
fallback; the structured record is
`results/candidate-fused-tmrope-kv-multimodal.json`.

### No-profiler result

| Workload | Metric | Packed QKV | Fused KV write | Ratio / change |
|---|---|---:|---:|---:|
| 1,024 + 32 | TPOT p50 | 9.485 ms | 9.384 ms | 1.011× faster |
| 128 + 128 | TPOT p50 | 8.363 ms | 8.252 ms | 1.013× faster |
| 2,048 + 32 | TPOT p50 | 10.863 ms | 10.754 ms | 1.010× faster |
| 2,560 + 32 | TPOT p50 | 11.503 ms | 11.426 ms | 0.7% faster |
| 3,072 + 32 | TPOT p50 | 12.178 ms | 12.155 ms | unchanged ordinary path |
| 4,096 + 32 | TPOT p50 | 12.691 ms | 12.701 ms | unchanged ordinary path |

Quick and decode TPOT CVs are 0.06% and 0.10%. The deployed service repeats
9.383 and 8.254 ms TPOT with exact trajectories. Against the original
baseline, decode TPOT improves from 17.567 to 8.252 ms (2.129×). The weight
lower-bound effective bandwidth is now about 748 GB/s, or 74.2% of the
declared RTX 4090 peak.

### Actual promoted-binary attribution

The deployed binary SHA-256 is
`dcccfb7bc7c9ca8d634a09f2128028e9f86d770a3e1d6aa83fe238fba8bea4e2`.
The exact profiled request reports 8.327 ms TPOT; profiler timing is not used
for admission. Its report remains at
`/var/lib/agent-gpu-broker/profiles/omni-decode128-fused-tmrope-kv.nsys-rep`,
size 692,357 bytes, SHA-256
`7845d92901bcb2cfa91db84703f75ecb6b0b2434bb1ba05ab4fb5f4cd4d7cd80`.

The request window contains 127 graph launches, zero logits D2H and 8.276 ms
average stream synchronization, down from 8.396 ms. In the observable eager
prewarm, separate K TMRoPE plus K/V appends total about 9.4 microseconds per
layer; the fused node takes 3.66 microseconds. Kernel accounting changes from
one Q and one K TMRoPE plus two cache appends to one Q TMRoPE plus one fused
K/V publication node. The checked-in API, kernel and memory CSV hashes are
respectively
`7792c917c4c0534d32b2c9dc5899fd4a369fcc10815eef86c7ae2877cfd66411`,
`4e94b20accc2d62cd88e02a7af520457e140feefd41349d3bee77933a741625f`
and `ebb0c8b4238d43733322114fdbcdea9277ca95cedf58c6b25559dcdd82143e9d`.

## Fused TMRoPE K/V decision

**Decision: promote direct TMRoPE K/V cache publication, composed with the
accepted single-owner packed QKV, graph crossover and GPU token selection, for
the tested SM89 single-request BF16 cells.** It passes exact operator/cache,
complete trajectory, no-profiler materiality, graph/ordinary isolation,
long-context/OOM recovery, real multimodal, explicit Broker configuration and
actual-binary profile gates. Qwen3.8 remains down.

This result does not claim multi-request or continuous-batching performance,
non-SM89 portability, video or speech generation, vLLM parity, a larger OOM
boundary, or transaction-counter MFU/BWU.

## Rejected combined Q/K/V TMRoPE launch

The next candidate placed the 16 Q-head TMRoPE blocks and two fused K/V cache
blocks in one launch. Candidate binary SHA-256 was
`bee2ec7b4e1a37fd285e508f99d23066db4ead542ff00637c60396c98996d5a6`.
A direct CUDA test compared Q output plus complete K/V cache allocations
byte-for-byte with the two accepted nodes and passed; quick and decode token
trajectories were also exact and stable.

| Workload | Accepted TPOT | Combined launch TPOT | Improvement |
|---|---:|---:|---:|
| 1,024 + 32 | 9.383 ms | 9.321 ms | 0.66% |
| 128 + 128 | 8.254 ms | 8.188 ms | 0.80% |

Both changes exceed measurement noise but miss the predeclared 1% materiality
gate. Long-context, OOM, multimodal and profiler budgets were therefore not
spent. **Decision: revert and remove the combined kernel/selector.** The raw
no-profiler evidence is retained; direct K/V publication remains the accepted
boundary.

## Promoted shape-specialized softmax exp cache

Primary classification: **source/runtime graph**. The exact-order softmax
previously evaluated `exp(score - max)` twice: once for the sequential sum and
again for the output numerator. The candidate computes each FP32 numerator
once in parallel, caches it in dynamic shared memory, lets lane 0 add those
unchanged FP32 values in the original column order, then performs the original
parallel division. Max order, sum order and BF16 output conversion therefore
remain unchanged.

`APXINF_SOFTMAX_EXP_CACHE=1` selects two explicit shape regimes. Multi-token
prefill uses the shared cache only through 4,096 columns; longer prefill uses
the accepted sequential kernel because larger per-CTA shared memory reduces
occupancy. Single-token decode uses the shared cache through the tested 11,264
column limit. Invalid flag values and unsupported BF16 cache shapes fail
closed. The deployed SM89 binary SHA-256 is
`5a131f72e6d04a439545bec8ab8c7893da7c499b0cacf8378bc9594a58063f68`.

The shape split was selected from negative evidence, not post-hoc omission.
Enabling the cache for every prefill length kept exact trajectories and
improved 4K, but regressed 10,752 TTFT from about 7.97 s to 8.88 s. The final
hybrid retains the 4K and decode wins while its independent 8K/10,752 retries
match or slightly improve the accepted path. Both the full-cache and hybrid
raw results are retained.

### Operator, resource and correctness gates

The candidate output is bit-exact to the scalar BF16 kernel in a direct CUDA
test; a second candidate-specific test crosses 65,535 logical rows. The final
binary reports 36 registers, 16 bytes static shared memory, zero local memory
and zero stack. Dynamic shared memory is 16,384 bytes at 4K and 43,040 bytes at
the longest generated 10,760-column decode step.

| Workload | Result | Trajectory SHA-256 | Prior accepted agreement |
|---|---:|---|---:|
| 1,024 prompt + 32 output | 5/5 stable | `bf1da0a151446e7ff757474de26ceb13ced3cf9a422fcc17b1f4699fb89d38ea` | exact |
| 128 prompt + 128 output | 5/5 stable | `a892eb11d69ed4a408e680432ebee0de04a202950ba7c5072731a001c753f039` | exact |
| 4,096 prompt + 8 output | 3/3 stable | `edc940b80ff945971996ddf2b30534773258bb41d3d1812c38f36ba06eaabcbd` | exact |
| 8,192 prompt + 8 output | 1/1 retry | `490c84bc9f905195eeeb560ed9b64d55f5e10430cb12f146d672491d860229cf` | exact |
| 10,752 prompt + 8 output | 1/1 retry | `19478a6e232ab7479a2a8026f01096ab68a16f72d922be9695d807928125b02d` | exact |
| 11,264 prompt + 8 output | CUDA OOM | n/a | same first failure |

The immediately following three 1K requests recover with one stable
trajectory and 12.31 GiB resident memory. Real PNG and WAV requests reproduce
their prior exact output-token sequences with no fallback. The model and API
capability contract is unchanged.

### No-profiler end-to-end result

| Workload | Metric | Decode-cache baseline | Hybrid exp cache | Change |
|---|---|---:|---:|---:|
| 1,024 + 32 | TTFT p50 | 0.0845 s | 0.0794 s | 1.064× faster |
| 1,024 + 32 | TPOT p50 | 12.048 ms | 11.613 ms | 1.037× faster |
| 1,024 + 32 | wall p50 | 0.4595 s | 0.4411 s | 1.042× faster |
| 128 + 128 | TPOT p50 | 10.891 ms | 10.822 ms | 0.6% faster |
| 128 + 128 | wall p50 | 1.4041 s | 1.3952 s | 0.6% faster |
| 4,096 + 8 | TTFT p50 | 0.8031 s | 0.7297 s | 1.100× faster |
| 4,096 + 8 | wall p50 | 0.9106 s | 0.8262 s | 1.102× faster |
| 8,192 + 8 | TTFT, retry | 4.6836 s | 4.6286 s | 1.012× faster |
| 10,752 + 8 | TTFT, retry | 7.9655 s | 7.8803 s | 1.011× faster |

Hybrid 4K TTFT CV is 0.03% and decode TPOT CV is 0.07%. Against the original
paired baseline, 4K TTFT improves from 18.3407 s to 0.7297 s (25.133×) and
decode TPOT from 17.567 ms to 10.822 ms (1.623×). The Broker-owned deployment
repeats the exact 1K trajectory with 0.0793 s TTFT and 11.613 ms TPOT.

### Causal profile

The actual hybrid-candidate report remains on the GPU host at
`/var/lib/agent-gpu-broker/profiles/omni-4k-softmax-exp-cache.nsys-rep`, size
1,487,341 bytes, SHA-256
`1d0957cf164fa6a35d282eff742550f479202425d870d7636af7adf8f7676da2`.
The profiled request preserves the 4K trajectory; profiler timing is not used
for admission.

Across the same 288 softmax launches, summed kernel time falls from
250.759 ms to 184.393 ms (26.5% lower, 1.360× faster). The softmax share falls
from 32.3% to 25.9%. The two leading packed-GEMM candidates remain unchanged
at about 162.4 ms and 132.1 ms, and launch count remains 8,268. This is the
predicted causal result of deleting one exponential evaluation per valid
score, not a launch or model-path change.

## Softmax decision at that stage

**Decision: promote the shape-specialized FP32 numerator cache, composed with
decode TMRoPE caching, stream-ordered allocation, batched GQA and exact-order
softmax, for the tested single-request BF16 cells.** It passes bit-exact and
launch-boundary operator gates, complete trajectories, repeated no-profiler
timing, long-context capacity, real multimodal, OOM recovery, explicit-path
and actual-candidate profile gates. The RTX 4090 service runs this binary under
Broker ownership; Qwen3.8 remains down.

This result does not claim multi-request or continuous-batching performance,
video or speech generation, vLLM parity, a larger OOM boundary, or MFU/BWU.

## Rejected load-time Gate/Up packing

Primary classification: **source/runtime graph**. The candidate selected one
of two weight layouts at load time: either the accepted separate Gate and Up
matrices, or one row-major `[hidden, 2*intermediate]` matrix. It never kept both
GPU representations. Packed mode replaced two GEMMs, SiLU and Mul with one
wider GEMM and one SwiGLU kernel. The fused kernel explicitly rounded the SiLU
intermediate to BF16 before multiplication, and its CUDA operator test was
bit-exact to the two-kernel reference. Resident VRAM was 12,190 MiB versus the
accepted service's approximately 12,309 MiB, confirming there was no hidden
weight duplicate. Candidate binary SHA-256 was
`4940704818480a7a457fa71ae6350795a0ee218317fc2cd6fbe26bdbc96970cb`.

Complete trajectories remained exact for all admission screens, but no
end-to-end win existed:

| Workload | Metric | Accepted | Packed MLP | Change |
|---|---|---:|---:|---:|
| 1,024 + 32 | wall p50 | 0.4411 s | 0.4428 s | 0.4% slower |
| 1,024 + 32 | TPOT p50 | 11.613 ms | 11.730 ms | 1.0% slower |
| 128 + 128 | wall p50 | 1.3952 s | 1.4085 s | 1.0% slower |
| 128 + 128 | TPOT p50 | 10.822 ms | 10.936 ms | 1.1% slower |
| 4,096 + 8 | TTFT p50 | 0.7297 s | 0.7354 s | 0.8% slower |
| 4,096 + 8 | wall p50 | 0.8262 s | 0.8311 s | 0.6% slower |

**Decision: revert.** The wide GEMM saved launches but did not shorten any
tested service envelope, so long-context and profiler budgets were not spent.
The implementation was removed; the bit-exact operator, layout and structured
E2E evidence are retained to close this branch. The next bounded direction is
a Qwen2.5-Omni decode graph that preserves the accepted kernels but removes the
roughly one thousand host launches per generated token.

## Promoted decode TMRoPE position cache

Primary classification: **source/runtime graph**. Each Qwen2.5-Omni layer
applies identical TMRoPE positions to Q and K. Before this candidate, the CUDA
backend synchronously allocated and uploaded the same position array twice per
layer. During decode that means 72 allocation/upload/free sequences for every
generated token.

`APXINF_TMROPE_POSITION_CACHE=1` now caches one decode-only GPU position
buffer keyed by the complete `Vec<u32>` contents. The cache never uses host
pointer identity. A changed decode position replaces the buffer with a
stream-ordered allocation and copy whose source bytes stay alive until the
stream reaches them; all 36 Q/K layer pairs for that token reuse it. Multi-token
prefill deliberately retains the accepted v4 path. Unset or `0` preserves the
uncached path, and invalid values fail closed. The deployed SM89 binary SHA-256
is `342cb4bd2a4b4ab58102866e61656b4ab071fae8e9ceee9d0f54c19387526e95`.

The decode-only boundary is the result of a negative iteration ladder. A
full-prefill cache improved decode but made the 10,752-token passing point OOM.
Moving its small buffer to the async pool did not recover capacity. Draining
all 36 layers restored the trajectory but regressed 10,752 TTFT from 7.92 s to
16.67 s; draining only layer 0 still OOMed. These branches are retained under
the `candidate-tmrope-cache-*` result prefixes. They were rejected rather than
hidden by a shorter-context score.

### Correctness and coverage

| Workload | Result | Trajectory SHA-256 | v4 agreement |
|---|---:|---|---:|
| 1,024 prompt + 32 output | 5/5 stable | `bf1da0a151446e7ff757474de26ceb13ced3cf9a422fcc17b1f4699fb89d38ea` | exact |
| 128 prompt + 128 output | 5/5 stable | `a892eb11d69ed4a408e680432ebee0de04a202950ba7c5072731a001c753f039` | exact |
| 4,096 prompt + 8 output | 3/3 stable | `edc940b80ff945971996ddf2b30534773258bb41d3d1812c38f36ba06eaabcbd` | exact |
| 8,192 prompt + 8 output | 1/1 exploratory | `490c84bc9f905195eeeb560ed9b64d55f5e10430cb12f146d672491d860229cf` | exact |
| 10,752 prompt + 8 output | 1/1 exploratory | `19478a6e232ab7479a2a8026f01096ab68a16f72d922be9695d807928125b02d` | exact |
| 11,264 prompt + 8 output | CUDA OOM | n/a | same first failure |

After the 11,264 OOM, resident memory is 12.31 GiB and the immediately
following three 1K requests all pass with one stable trajectory. Real PNG and
WAV inputs reproduce the prior exact output-token sequences and no fallback;
their decode TPOT also improves. The model/API capability contract is
unchanged.

### No-profiler end-to-end result

| Workload | Metric | Allocator v4 | Decode cache | Change |
|---|---|---:|---:|---:|
| 1,024 + 32 | TTFT p50 | 0.0855 s | 0.0845 s | unchanged |
| 1,024 + 32 | TPOT p50 | 13.783 ms | 12.048 ms | 1.144× faster |
| 1,024 + 32 | wall p50 | 0.5150 s | 0.4595 s | 1.121× faster |
| 128 + 128 | TTFT p50 | 0.0195 s | 0.0191 s | unchanged |
| 128 + 128 | TPOT p50 | 13.033 ms | 10.891 ms | 1.197× faster |
| 128 + 128 | wall p50 | 1.6768 s | 1.4041 s | 1.194× faster |
| 4,096 + 8 | TTFT p50 | 0.8011 s | 0.8031 s | 0.24% slower |
| 4,096 + 8 | wall p50 | 0.9215 s | 0.9106 s | 1.012× faster |
| 8,192 + 8 | TTFT, one trial | 4.6841 s | 4.6836 s | unchanged |
| 10,752 + 8 | TTFT, one trial | 7.9156 s | 7.9655 s | 0.63% slower |

Decode-cache TPOT CV is 0.05% in the 128+128 screen. Against the original
paired baseline, decode TPOT improves from 17.567 ms to 10.891 ms (1.613×),
and 4K TTFT remains 22.838× faster. The Broker-owned deployment repeats the
128+128 result at 10.883 ms TPOT with the exact trajectory.

### Decode-cache attribution

The actual candidate report remains on the GPU host at
`/var/lib/agent-gpu-broker/profiles/omni-4k-tmrope-decode-cache.nsys-rep`, size
1,492,354 bytes, SHA-256
`a5b4d3e6303775cb514a4a01c6fbdc2e9c237dba42cd1b82675f5e3268522290`.
The profiled 4K request preserves its trajectory; profiler timing is not used
for admission.

Across seven decode steps, synchronous `cudaMalloc`/`cudaFree` falls from
518/518 in allocator v4 to 14/14. H2D operations fall from 532 to 21, totaling
only 9 microseconds of GPU copy time. Prefill API counts remain essentially
unchanged, proving the optimized gate is decode-only. Kernel count stays at
8,268 and the leading prefill GPU kernel remains sequential softmax.

## Decode-cache decision at that stage

**Decision: promote decode-only TMRoPE position caching, composed with the
accepted allocator, batched GQA and exact-order softmax, for the tested
single-request BF16 cells.** It passes complete trajectories, repeated
no-profiler decode and prefill, long-context capacity, real multimodal, OOM
recovery, explicit-path and actual-candidate profile gates. The RTX 4090
service runs this binary under Broker ownership; Qwen3.8 remains down.

This result does not claim multi-request or continuous-batching performance,
video or speech generation, vLLM parity, a larger OOM boundary, or MFU/BWU.

## Earlier promoted stream-ordered allocation candidate

Primary classification: **source/runtime graph**. The candidate adds an
explicit `APXINF_STREAM_ORDERED_ALLOC=1` mode to the existing transient output
buffer owner. Model-hot temporary buffers use matched
`cudaMallocAsync`/`cudaFreeAsync` operations on the same inference stream;
unset or `0` preserves synchronous allocation, and invalid values fail closed.
Persistent weights, KV storage and host-upload buffers remain synchronously
allocated. CUDA Graph workspaces keep their existing arena path.

The final candidate also makes KV reset reuse and zero its existing buffers
instead of reallocating all 72 K/V allocations. On a failed request, the model
drains the backend stream before clearing state, and context synchronization
consumes the thread-local CUDA error after preserving the original error for
the current request. These are required failure semantics, not performance
fallbacks. The deployed SM89 binary SHA-256 is
`cee5ef02d5cf89b47ca4bd160103619037dfa7f2dba270254ff519c4b65a8f43`.
Raw result prefixes record the iteration ladder: unqualified
`candidate-streamalloc-*` is the first prototype, `recovery-*` adds a stream
drain, `kvclear-*` adds in-place KV reset, and `v4-*` is the final sticky-error
fix and promotion candidate.

### Correctness, stability and recovery

The final binary preserves every previously frozen trajectory:

| Workload | Result | Trajectory SHA-256 | Prior accepted agreement |
|---|---:|---|---:|
| 1,024 prompt + 32 output | 5/5 stable | `bf1da0a151446e7ff757474de26ceb13ced3cf9a422fcc17b1f4699fb89d38ea` | exact |
| 128 prompt + 128 output | 5/5 stable | `a892eb11d69ed4a408e680432ebee0de04a202950ba7c5072731a001c753f039` | exact |
| 4,096 prompt + 8 output | 3/3 stable | `edc940b80ff945971996ddf2b30534773258bb41d3d1812c38f36ba06eaabcbd` | exact |
| 8,192 prompt + 8 output | 1/1 exploratory | `490c84bc9f905195eeeb560ed9b64d55f5e10430cb12f146d672491d860229cf` | exact |
| 10,752 prompt + 8 output | 1/1 exploratory | `19478a6e232ab7479a2a8026f01096ab68a16f72d922be9695d807928125b02d` | exact |

Real PNG and WAV requests also reproduce the prior output-token sequences with
no fallback. Ten consecutive 1K requests on the first allocator prototype were
stable and returned resident memory to about 12.3 GiB.

The first allocator prototype failed the OOM recovery gate: after the expected
11,264-token OOM, pool memory remained near 20.2 GiB and the next 1K request
also failed. A stream drain recovered memory but still left one sticky CUDA
error; reallocating KV storage could also fail partway. The final candidate
closes both issues. It reports 11,264 OOM as HTTP 503, immediately returns to
12.27 GiB, and the next three 1K requests all pass with the exact trajectory.
The raw negative and recovery screens are retained; only the final candidate
is deployed.

### No-profiler end-to-end result

| Workload | Metric | Batched GQA | Final allocator | Change |
|---|---|---:|---:|---:|
| 1,024 + 32 | TTFT p50 | 0.3459 s | 0.0855 s | 4.047× faster |
| 1,024 + 32 | wall p50 | 1.0330 s | 0.5150 s | 2.006× faster |
| 128 + 128 | TTFT p50 | 0.0829 s | 0.0195 s | 4.259× faster |
| 128 + 128 | TPOT p50 | 17.621 ms | 13.033 ms | 1.352× faster |
| 4,096 + 8 | TTFT p50 | 1.9661 s | 0.8011 s | 2.454× faster |
| 4,096 + 8 | wall p50 | 2.2440 s | 0.9215 s | 2.435× faster |
| 8,192 + 8 | TTFT, one trial | 6.3336 s | 4.6841 s | 1.352× faster |
| 10,752 + 8 | TTFT, one trial | 13.4895 s | 7.9156 s | 1.704× faster |

Against the original paired baseline, 4K TTFT improves from 18.3407 s to
0.8011 s (22.894×), while 1K TTFT improves from 2.0755 s to 0.0855 s
(24.281×). Final 4K TTFT CV is 0.22%; final decode TPOT CV is 0.08%. The
Broker-owned deployment repeats the 1K trajectory at 0.0860 s TTFT and returns
to 12.27 GiB resident memory.

### Final allocator attribution

The actual final-candidate report remains on the GPU host at
`/var/lib/agent-gpu-broker/profiles/omni-4k-streamalloc-v4.nsys-rep`, size
1,534,333 bytes, SHA-256
`ef6552ba918d175f610de05fe149b20c852b82603caa7c75978480717bc894c5`.
The profiled request preserves the 4K trajectory; profiler timing is not used
for admission.

In prefill, synchronous `cudaMalloc`/`cudaFree` calls fall from 941/939 in the
batched-GQA profile to 74/74; 795 transient allocations move to the ordered
pool. Across seven decode steps, synchronous allocation/free falls from about
6.1 thousand each to 518 each, with about 5.6 thousand operations becoming
stream ordered. Launch count remains 8,268. Request-local H2D is 0.295 ms.

The large 0.620 s profiled `cudaMemcpy` API interval is primarily the final
synchronizing D2H copy waiting for already-enqueued GPU work, not 0.620 s of
data movement; GPU H2D/D2H operation time is only about 8.8 ms. The leading GPU
kernel remains sequential softmax at about 249 ms, followed by the two batched
GEMM families. The next bounded opportunity is to upload TMRoPE positions once
per forward instead of once per Q/K layer call, then reassess whether the
remaining softmax arithmetic justifies a semantics-preserving kernel change.

## Allocator decision at that stage

**Decision: promote stream-ordered transient allocation plus in-place KV
reset, composed with strided-batched GQA and sequential-order softmax, for the
tested single-request BF16 cells.** The final binary passes complete trajectory,
repeated no-profiler, multimodal, long-context, OOM recovery, explicit-path and
actual-candidate profile gates. Qwen3.8 remains down, and all GPU ownership is
through the Broker.

This result does not claim multi-request or continuous-batching performance,
video or speech generation, vLLM parity, a larger OOM boundary, or MFU/BWU.

## Deferred long-prefill CUDA Graph hypothesis

The 12,288-token accepted Systems export contains 45,553 kernel instances over
a 1.6195-second first-to-last device span. Merging the device intervals leaves
79.67 ms of all positive inter-kernel gaps, so even impossible removal of every
gap has a 1.052× speedup ceiling. Only 14.42 ms lies in gaps above 100 us. The
220.24 ms summed `cudaLaunchKernel` API time is not a wall-clock quantity because
host submission overlaps device execution.

The matched no-profiler ApxInf/vLLM-Omni TTFT gap at 12K is 541.42 ms. Perfect
removal of the profiled device gaps could explain at most 14.7% of that deficit
and would still leave ApxInf about 1.56× slower. A full prefill graph executor is
therefore deferred: launch/control overhead is real but not the primary migrated
bottleneck. The timeline hashes, gap calculation and comparison are frozen in
`candidate-prefill-cuda-graph-upper-bound.json`.

An archived causal-FA2 long-only gate was also attempted under Broker 0.5.1,
but the old binary remained in model-loading/NFS I/O with only 398 MiB resident
VRAM and never opened the service endpoint. It was cancelled after 229.9 seconds
to release an already queued Qwen3.8 qualification job. This is an invalid
infrastructure attempt, not correctness or performance evidence; the 8K token
gate remains unexecuted.

## Promoted long-only causal FA2 prefill

Primary classification: **model/runtime graph rewrite plus shape-bounded CUDA
specialization**. `APXINF_FA2_GQA_PREFILL=1` composes with the accepted
batched-GQA selector only for BF16 suffix-prefill calls whose accumulated KV
length is at least 4,097 and whose QH/KVH/D shape is exactly 16/2/128. Decode,
at-most-4K prefill, multimodal attention and selector-off execution remain on
their accepted paths. Unavailable kernels, non-suffix requests and shape
mismatches fail explicitly instead of silently falling back.

The `core-fa2` carrier adds one SM89 64x64, no-dropout, causal head-128
specialization beside the accepted non-causal vision specialization. It reads
the existing head-major KV cache through explicit row/head strides and returns
the ordinary flattened model-owned output. The direct 4,097-boundary operator
contract passes against the scalar implementation.

### Exactness and no-profiler admission

The predeclared 8K correctness-only request passed before 12K or 32K was run.
All later complete trajectories stayed exact, including 30 formal ABBA
requests and 40 measured boundary requests. The frozen long hashes are:

| Workload | Complete trajectory SHA-256 |
|---|---|
| 8,192 + 8 | `490c84bc9f905195eeeb560ed9b64d55f5e10430cb12f146d672491d860229cf` |
| 12,288 + 8 | `57c5d6ea1879e2f718dc40d47409b0a6aee31afdbd668c255d97409e4661f832` |
| 32,760 + 8 | `f5ef60ededd5770627b7963e24ff339aef60d63d061cafa37b7ee4e4b0598cb9` |

Five pairs alternated `AB / BA / AB / BA / AB`. A and B used the same binary
SHA-256
`d28373c62dd6e0adae899ef856ea3461d40a279982a2757394babefcaea4848a`;
only the process-start selector changed. Each service had one warmup before one
measured request per length.

| Workload | Selector-off TTFT median | FA2 TTFT median | Paired TTFT speedup median | FA2 wall median | Paired wall speedup median |
|---|---:|---:|---:|---:|---:|
| 8,192 + 8 | 0.8154 s | 0.7076 s | 1.153× | 0.7848 s | 1.138× |
| 12,288 + 8 | 1.3730 s | 1.0871 s | 1.263× | 1.1813 s | 1.243× |
| 32,760 + 8 | 5.6934 s | 2.6084 s | 2.181× | 2.7869 s | 2.105× |

One fifth-pair 8K candidate request took 1.0082 s and lost that pair; it is
retained in the paired statistics. The other four 8K pairs won, and three
additional exact candidate requests measured 0.7104 s mean TTFT with 0.092%
CV. This is recorded as an observed transient outlier, not deleted evidence.
All five 12K and 32K pairs won; their worst paired TTFT speedups were 1.258×
and 2.180×.

The selector's lower boundary was screened separately with one warmup and five
requests per mode:

| Prompt + 8 | Selector-off TTFT p50 | FA2 TTFT p50 | Speedup | Trajectory |
|---:|---:|---:|---:|---|
| 4,352 | 0.4179 s | 0.4137 s | 1.010× | exact/stable |
| 5,120 | 0.4876 s | 0.4682 s | 1.041× | exact/stable |
| 6,144 | 0.5901 s | 0.5455 s | 1.082× | exact/stable |
| 7,168 | 0.7004 s | 0.6266 s | 1.118× | exact/stable |

This measured crossover admits the 4,097 threshold without an untested
4K-to-8K performance region.

### Matched Systems attribution

Only after the repeated end-to-end win, matched 32K selector-off/on profiles
captured one warmed request each. Profiler timing is not admission timing. The
selector-off report contains 29,301 kernels and 5.7507 s summed GPU kernel
time; FA2 contains 25,269 kernels and 2.7359 s, a 2.102× reduction in summed
kernel work.

The selector-off path spends 2.6155 s in 1,008 scalar softmax kernels. FA2
replaces them and the associated materialized QK/PV work with 1,008 fused
kernels totaling 1.0490 s. It also removes 4,032 kernel launches, all 4,032
device-to-device copies, and 3,024 async allocation/free pairs. Summed CUDA API
durations are not interpreted as wall time.

The reports remain on the GPU host:

- selector off: `omni-32760-long-fa2-baseline-interactive.nsys-rep`, 3,704,930
  bytes, SHA-256
  `2213331d3e3f9ef705df5e0ac7e614ee9690809c5f8fb3e11892760b7dae037e`;
- FA2: `omni-32760-long-fa2-candidate-interactive.nsys-rep`, 2,949,923 bytes,
  SHA-256
  `87b6bc72a46ce3ce20e5fb57b8a8634b1333c4a7be9955bae83cef57d0b1b1aa`.

### Regression and decision

The selector-on service passes repeated 1K, 128+128 decode, 4K boundary and
8K stability trajectories, real PNG, real WAV and malformed-media recovery.
The CUDA binary passes 85 non-FP8 tests, including the new FA2 contract. Two
FP8 GEMM tests return cuBLAS status 15 on RTX 4090; the accepted control binary
fails the same tests with the same status, so they are preserved as known
hardware/test-applicability failures rather than misreported as candidate
successes or regressions.

The promoted binary SHA-256 is
`d28373c62dd6e0adae899ef856ea3461d40a279982a2757394babefcaea4848a`;
the immediate rollback is
`942eea1b67eb173b0494dd6ec83d8b7559fc081612b0694de168de224bcda269`.
A runit-owned 12K deployment smoke reproduced the exact trajectory at 1.093 s
TTFT. The service was then returned to its declared down state and the Broker
reported the GPU idle.

**Decision: promote the long-only causal FA2 selector for the tested
single-request BF16 RTX 4090 service.** The structured decision and raw-file
index are `promotion-causal-fa2-long-prefill.json`. This does not claim
multi-request performance, non-SM89 portability, training, or speech/video
generation.

## Promoted FA2-aware 8K–12K chunk-count retune

Primary classification: **execution-graph scheduling rewrite**. After causal
FA2 promotion, a warmed 12,288+8 Systems capture moved the leading cost to the
four projection/MLP GEMMs repeated for every 256-token chunk: 6,912 launches
totaled 505.7 ms, or 44.2% of summed GPU kernel time. FA2 itself totaled
274.1 ms over 1,152 launches. The 48-chunk schedule therefore exposed more
removable low-M GEMM, launch and allocation work than an FA2 tile change. The
source report is
`omni-12288-long-fa2-current-interactive.nsys-rep`, size 3,673,046 bytes,
SHA-256
`1e20fa669f49436294d8601d9ae401e3d42047ba471b55be56a3736d1d865306`.

The earlier 1,024-token experiment is not reused as evidence: it ran before
FA2 and regressed because larger chunks increased materialized masked
attention. The migrated runtime no longer materializes that path above 4,096
accumulated KV tokens, so its causal mechanism changed.

`APXINF_QWEN25_FA2_CHUNK1024=1` changes only reset, text-only prompts from
8,192 through 12,288 tokens from 256- to 1,024-token chunks. It requires both
`APXINF_QWEN25_CHUNKED_PREFILL=1` and `APXINF_FA2_GQA_PREFILL=1`; a negative
service gate proves model initialization fails explicitly when causal FA2 is
disabled. Shorter prompts, prompts above 12,288, decode, multimodal inputs and
selector-off execution retain the accepted policy. A pure policy test freezes
the boundaries, and a service log marker proves the selected path.

The first 12K candidate request reproduced the accepted complete trajectory at
0.8692 s TTFT. Formal screening then alternated `AB / BA / AB / BA / AB` with
the same binary SHA-256
`71db60c7a545647c5a2f6e9cd1967e402d1188f5098dc5b27853605cc4f1fba1`;
only the selector changed. Every one of the 30 measured trajectories was exact
and stable.

| Workload | Selector-off TTFT median | 1,024-chunk TTFT median | Paired TTFT speedup | Paired wins | Candidate wall median |
|---|---:|---:|---:|---:|---:|
| 8,192 + 8 | 0.7081 s | 0.6145 s | 1.153× | 5/5 | 0.6917 s |
| 12,288 + 8 | 1.0877 s | 0.8660 s | 1.256× | 5/5 | 0.9608 s |
| 32,760 + 8 control | 2.6100 s | 2.6125 s | 1.000× | unchanged path | 2.7905 s |

The worst paired TTFT speedups are 1.151× at 8K and 1.252× at 12K. Candidate
TTFT CV is 0.097% and 0.084%; the 32K control changes by only 0.009%. The
eight-output TPOT deltas are below the prefill admission claim.

### Matched Systems attribution

Only after the repeated win, one warmed 12K request per mode was profiled.
Summed GPU kernel time falls from 1.1411 to 0.9295 seconds, an 18.54%
reduction, while kernel launches fall from 36,661 to 13,729. Async allocation
and free calls each fall from 35,694 to 12,726. FA2 calls fall from 1,152 to
288 and their total time from 273.2 to 137.6 ms; the leading 6,912-launch
M=256 GEMM family becomes a set of larger-M algorithms, with only 216 calls
remaining in that exact family. This 211.6 ms GPU-work reduction explains the
approximately 221.7 ms no-profiler TTFT reduction. CUDA API sums are not used
as wall time.

The matched reports remain on the GPU host:

- selector off: `omni-12288-fa2-chunk1024-baseline-interactive.nsys-rep`,
  3,669,493 bytes, SHA-256
  `32c17ebd05e48c0cff9a127b1b322d182fed3f9cd8b259439c3533e6a91c0fe7`;
- 1,024 chunks: `omni-12288-fa2-chunk1024-candidate-interactive.nsys-rep`,
  1,708,695 bytes, SHA-256
  `38afda2147712f5bb6964915fe6eea4c3cc7fdeefaa15918b6f946ab97c4865f`.

### Regression and decision

The candidate passes 63 model CPU tests, 13 benchmark tests, repeated quick
and decode trajectories, 4K/7168/8192 boundaries, real PNG/WAV and malformed
media recovery. The current CUDA suite passes 93 non-FP8 tests, including all
Omni/FA2 and newly imported SM89 GeGLU routing tests. The same two RTX 4090
FP8 cuBLAS status-15 tests fail as in the accepted-control binary and remain
explicit known failures.

Relative to the frozen vLLM-Omni 0.26.0 result, ApxInf is now within 4.3% TTFT
and effectively tied on wall time at 12K; ApxInf wins 32K TTFT by 1.115× and
wall by 1.088×, while vLLM still wins 8K TTFT by 1.200×.

The promoted binary SHA-256 is
`71db60c7a545647c5a2f6e9cd1967e402d1188f5098dc5b27853605cc4f1fba1`;
the immediate rollback is
`d28373c62dd6e0adae899ef856ea3461d40a279982a2757394babefcaea4848a`.
A runit-owned 8K/12K deployment smoke reproduced both exact trajectories at
0.622 and 0.861 s TTFT. The service was then returned down and the Broker
reported the GPU idle.

**Decision: promote FA2-aware 1,024-token chunks for the tested 8K–12K
single-request BF16 RTX 4090 cells.** Structured evidence and the raw-file
index are `promotion-fa2-chunk1024.json`. This does not claim multi-request
performance, non-SM89 performance, training, or speech/video generation.

## Rejected post-promotion 2,048-token chunks

The promoted 8K profile contains 11,421 kernels and 666.4 ms summed GPU kernel
time. FA2 is only 52.8 ms; four M=1,024 GEMM families total about 411.8 ms.
An exact cold-L2 tuning probe first closed the lower-risk tactic branch:
M1024 Down rank 0 measures 0.28728 ms versus 0.28724 ms vendor, Q/O and K/V
are also null, while the installed packed-QKV rank 2 and Gate/Up rank 1 wins
reproduce. No sixth tactic is justified. The current 8K report remains at
`/var/lib/agent-gpu-broker/profiles/omni-8192-fa2-chunk1024-current-interactive.nsys-rep`,
size 1,443,341 bytes, SHA-256
`75ee162fca9c902bf611fa37766e9e1a164099ce8dc55aeb1e1636157041cb91`.

A bounded probe on branch `codex/omni-fa2-chunk2048-probe` then halved the
schedule from eight to four chunks. Its one permitted 8K request reproduced
the accepted trajectory and logged `chunk=2048`, but TTFT regressed from the
accepted 0.6145 s median to 0.6900 s (12.28%); wall rose from 0.6917 to
0.7687 s. The larger first scalar-attention chunks erase the launch-count
opportunity before FA2 owns the later half.

**Decision: reject and close the 2,048-token chunk axis without 12K, AB/BA or
profiler spending.** Raw evidence is `omni-fa2-chunk2048-smoke-8k.json`; the
profile/tuning inputs use the `omni-8192-fa2-chunk1024-current` and
`omni-8192-m1024-bf16-tuning` names. Production remains on 1,024-token chunks.

## Prepared all-chunks causal FA2 upper-bound probe

Status: **exploratory source only; not a promotable request gate**. The 8K
profile still spends about 49.2 ms in early-chunk scalar softmax plus 24.8 ms
in score scaling and associated score/value GEMMs. A one-line compile-time
probe lowers the causal-FA2 KV threshold from 4,097 to 1 so every 1,024-token
chunk uses FA2. This estimates whether removing the early materialized path is
worth introducing an explicit request-scoped model/backend boundary.

The probe is deliberately not valid for deployment: the earlier all-prefill
FA2 experiment changed frozen 1K and 4K tokens. The only budget is one 8K
request. It must reproduce the accepted trajectory
`490c84bc9f905195eeeb560ed9b64d55f5e10430cb12f146d672491d860229cf`
and beat the accepted 0.6145 s TTFT median by at least 2%. A mismatch or null
result closes the boundary immediately. An exact material win only authorizes
designing a request-scoped strict gate; it cannot itself be promoted or
profiled.
