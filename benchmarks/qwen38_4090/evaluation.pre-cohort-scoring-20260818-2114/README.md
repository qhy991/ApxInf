# ApxInf Qwen3.8-27B evaluation v1

This directory freezes the public, machine-readable part of the summer-camp
assignment. The primary leaderboard workload is deliberately narrow:

- one RTX 4090;
- one text request at a time (`parallel_requests=1`, `max_num_seqs=1`);
- the pinned Qwen3.8-27B AWQ INT4 checkpoint;
- deterministic greedy decode;
- exact pre-tokenized prompt IDs and exact output budgets;
- client-observed TTFT and TPOT with no profiler attached.

The current ApxInf implementation and the hidden cases are teacher-side
artifacts. Students submit a PR against the public ApxInf interfaces. The
public evaluator and schema let them self-check; the same contract with a
hidden correctness split is used by the midterm leaderboard.

Multi-request scheduling is intentionally outside v1. It changes the problem
from single-request execution-graph optimization to admission, scheduling,
continuous batching, KV ownership, fairness, and tail-latency control. It can
be added later as a separate bonus track after those interfaces exist; it must
not be mixed into the v1 score.

## Files and authoritative roles

- `contract-v1.json`: workload, hardware constants, score weights, anchors,
  gates, and course-grade split. This is the scoring source of truth.
- `submission-schema-v1.json`: exchange format between a runner and scorer.
- `score_submission.py`: deterministic stdlib-only scorer.
- `compute_efficiency.py`: MFU/BWU proxy calculator and optional profiler
  counter calculator.
- `fixtures/`: teacher-run calibration inputs for the current Marlin-M64
  candidate and vLLM.

Do not duplicate weights or anchors in a runner. Read them from the contract.

## Score

The automated leaderboard is 100 points:

| Section | Points | Meaning |
|---|---:|---|
| correctness | 25 | protocol, public/hidden cases, token trajectory |
| TTFT | 30 | 1K, 2K, 4K, 8K, and 16K single-request cells |
| TPOT | 20 | 1K and 8K decode cells |
| maximum context | 15 | logarithmic progress from 1K to 32K |
| reliability | 10 | success rate and hard health/fallback checks |

For each latency cell:

```text
points = weight * min(1, anchor_seconds / observed_seconds)
```

A latency cell receives zero when prompt tokens, output tokens, or success rate
do not exactly match the contract. Correctness and reliability are eligibility
gates: an ineligible run still receives a `diagnostic_score`, but its
`leaderboard_score` is `null`.

The course grade is 80% automated leaderboard plus 20% PR review. The PR review
allocates 8 points to tests and negative controls, and 4 points each to
interface/error handling, reproducibility, and analysis/decision quality.

Run the frozen calibration examples:

```bash
python3 benchmarks/qwen38_4090/evaluation/score_submission.py \
  --submission benchmarks/qwen38_4090/evaluation/fixtures/vllm-reference.json

python3 benchmarks/qwen38_4090/evaluation/score_submission.py \
  --submission benchmarks/qwen38_4090/evaluation/fixtures/apxinf-marlin-current.json
```

The public profile does not invent a hidden score. The midterm runner must
populate `correctness.hidden_pass_rate` and use:

```bash
python3 benchmarks/qwen38_4090/evaluation/score_submission.py \
  --profile midterm_leaderboard --submission submission.json
```

The frozen teacher calibration currently produces:

| Implementation | Public calibration | TTFT | TPOT | Context | Status |
|---|---:|---:|---:|---:|---|
| ApxInf Marlin-M64 | 65.005 | 1.005/30 | 20/20 | 9/15 | eligible, provisional one-repeat performance cells |
| vLLM reference | 86.312 | 30/30 | 6.312/20 | 15/15 | eligible, frozen three-repeat baseline |

The public correctness set is the three 1K NIAH cases declared in the
contract. The token-trajectory gate is exact token-ID agreement on the 1K and
8K performance cells. The current ApxInf candidate passes 3/3 public cases and
256/256 declared trajectory tokens when both backends receive the same exact
pre-tokenized prompts.

## MFU and BWU

For W4A16 there is no honest single hardware-operation peak that represents the
whole model: the path mixes quantized weight loads/dequantization, BF16 Tensor
Core MMA, recurrent work, attention, and elementwise kernels. Therefore v1
reports two explicitly named wall-time proxies:

1. `estimated_mfu_bf16_equivalent_pct` uses the frozen dense-equivalent model
   FLOP estimate and the RTX 4090 dense BF16 Tensor Core peak with FP32
   accumulation. The INT4 TOPS value is shown only as a hardware reference.
2. `minimum_model_bwu_pct` uses a frozen minimum model/checkpoint byte estimate.
   It is a lower-bound model-byte proxy, not measured HBM traffic.

Neither proxy is a promotion gate and values are not clipped. A value above
100% is evidence that the proxy or timing boundary is invalid for that run, not
permission to report impossible utilization.

```bash
python3 benchmarks/qwen38_4090/evaluation/compute_efficiency.py \
  --submission submission.json --output efficiency.json
```

When targeted Nsight Compute collection is available, put phase-scoped
`kernel_elapsed_s`, `dram_read_bytes`, and `dram_write_bytes` counters under
`cells.<id>.profile.prefill` or `.decode`. The script then computes:

```text
profiled_bwu = (dram_read_bytes + dram_write_bytes)
               / kernel_elapsed_s / peak_dram_bandwidth
```

`tensor_pipe_active_pct`, when present, is reported separately. It is not
renamed to MFU. `nvidia-smi utilization.memory` is likewise telemetry only and
must not be multiplied by nominal GB/s.

## PR and leaderboard flow

1. The student implementation changes only public ApxInf extension points and
   includes focused tests, failure handling, a reproducible command, and a
   short result analysis.
2. Public CI checks the schema, public functional cases, exact token budgets,
   the fixed performance cells, and scorer unit tests.
3. The teacher runner checks out the PR SHA on a clean RTX 4090 image, runs the
   pinned checkpoint and hidden data, records environment/provenance, and
   publishes the machine score.
4. A PR is ranked only when every eligibility gate passes. Missing performance
   cells lose points but do not become fabricated numbers.

Public calibration accepts one measured repeat so that interface bring-up can
be scored quickly. The midterm profile requires at least one warm-up and three
measured repeats per latency cell; the submitted values are medians. A
single-repeat public result must never be presented as a formal leaderboard
number.

Before the assignment is published, the public runner still needs one canonical
pre-tokenized inference entry point, machine-readable health/model metadata,
and a clean adapter that emits `leaderboard_submission.v1` directly. Those are
P0 release gates, not student optimization work.
