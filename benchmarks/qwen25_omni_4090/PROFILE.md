# 4K prefill critical-path profile

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

## Newest promotion decision

**Decision: promote the shape-specialized FP32 numerator cache, composed with
decode TMRoPE caching, stream-ordered allocation, batched GQA and exact-order
softmax, for the tested single-request BF16 cells.** It passes bit-exact and
launch-boundary operator gates, complete trajectories, repeated no-profiler
timing, long-context capacity, real multimodal, OOM recovery, explicit-path
and actual-candidate profile gates. The RTX 4090 service runs this binary under
Broker ownership; Qwen3.8 remains down.

The next bounded opportunity is load-time Gate/Up packing: replace two GEMMs
with one wider GEMM and a fused SwiGLU without duplicating GPU weights. This
result does not claim multi-request or continuous-batching performance, video
or speech generation, vLLM parity, a larger OOM boundary, or MFU/BWU.

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
