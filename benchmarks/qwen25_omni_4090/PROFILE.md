# Qwen2.5-Omni RTX 4090 optimization evidence

`BASELINE.md` owns the current accepted deployment summary. This file retains
the causal profiles, rejected branches and promotion evidence for each stage;
the current promotion record is **Promoted fused TMRoPE K/V publication**
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
