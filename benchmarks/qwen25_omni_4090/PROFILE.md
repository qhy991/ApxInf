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
