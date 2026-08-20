# Exact M16 W4A16 prefill candidate on RTX 4090

Status: operator contract frozen before implementation on 2026-08-18.

## Objective and scope

Reduce Qwen3.8 single-request text-prefill TTFT by extending the accepted,
bit-exact small-M W4A16 kernel from eight to sixteen prompt rows. The first
gate is the real layer-0 `mlp.gate_proj` shape `[17408,5120]`; model promotion
must subsequently cover every W4 projection shape, stateful 64-layer execution,
M8/M1 tails, and the 1K/8K service cells.

This is PTX/SASS-directed CUDA: the performance carrier remains maintainable
CUDA source, while exact target-entry register, spill, stack, and local-memory
facts decide whether the larger live accumulator set is viable.

## Frozen contract

```text
model                 cyankiwi/Qwen3.8-27B-AWQ-INT4
model revision        63768c10df38c0395e12ef49edac1bd539eaeeea
source                 715a0ed790f2d10d82fab53fbeac3da3075adf26 + accepted dirty overlay
hardware               one RTX 4090 / SM89 / 128 SMs
CUDA                   12.3
weight                 compressed-tensors asymmetric U4 group-32
activation/output      BF16 [16,K] -> BF16 [16,N]
operator baseline      two accepted M8 calls
candidate              one M16 call
execution              one CUDA stream, eager, caller-owned output
correctness            every BF16 output bit exact; input immutable
timing boundary        host launch through stream synchronize, no profiler
```

The CUDA thread/warp ownership and per-token FMA/reduction order remain the
same as M8. Only the number of independent FP32 accumulator chains per lane
changes from eight to sixteen, so the intended semantic result is byte exact.

## Runtime graph and hypothesis

```text
two M8 calls:
  stream packed weight/scales/zero -> update 8 token accumulators
  stream the same weight again     -> update next 8 token accumulators

one M16 call:
  stream packed weight/scales/zero once -> update 16 token accumulators
```

The candidate removes one launch and one complete read of each projection's
packed weight, scales, and zero points. It doubles accumulator live ranges and
may lose through register pressure, occupancy, spills, instruction scheduling,
or activation-cache pressure. The closest SubCUDA positive case is
`flashinfer-silu-r32-sm100`, where crossing a register-residency boundary won;
the governing counterexample is `omoe-qwen35-tp2-d010-eight-lane`, where a
doubled live set reached 126 registers and regressed despite no spill.
`omoe-qwen35-tp2-d037-activation-register-budget` prevents treating a lower or
higher register count as a monotonic objective.

## Gates and stopping rule

Operator admission requires all of:

1. M16 output is bitwise identical to two M8 calls and sixteen M1 calls for
   deterministic and edge-pattern inputs; activation remains bitwise unchanged.
2. The exact `sm_89` target entry has no spill stores/loads, no stack frame, and
   no local memory. Record registers/thread; reject above 96 registers unless a
   measured occupancy/resource explanation justifies it before timing.
3. Five alternating AB/BA pairs: M16 wins at least 4/5, median speedup at least
   1.15x versus two M8 calls in both the hot repeated-weight and cold-HBM proxy.
4. Baseline repeatability is sufficient to distinguish the declared 15% floor.

If operator admission passes, the next boundary is one complete 16-token,
64-layer stack with M16 projections and exact residual/normalized/state/KV
endpoints. Model screening requires at least 1.20x joined speedup and 5/5 exact
endpoint pairs. Only then may the default-off service candidate be built.

The service screen uses one resident model, explicit `m16` versus accepted `m8`
selection, five adjacent balanced 1K pairs, at least two balanced 8K pairs, path
proof, unchanged decode, and the existing functional/token trajectory gates.
Promotion requires at least 25 stable pairs or a separately frozen equivalent
confidence rule plus a complete Nsight Systems causal trace.

M16 is rejected without model integration if it misses an operator gate. If it
passes locally but the 64-layer boundary misses its materiality gate, retain the
negative evidence and revert rather than adding an automatic fallback.

## Result and decision

The exact SM89 build produced 54 registers/thread for M16 versus 45 for M8,
with zero stack, local, shared, and spill memory. Correctness passed with zero
BF16 differences against both sixteen M1 calls and two accepted M8 calls; the
activation remained bitwise unchanged.

The performance gate failed:

| Cell | Two M8 median | M16 median | Speedup | Wins |
|---|---:|---:|---:|---:|
| hot repeated-weight | 363.889 us | 363.836 us | 1.00059x | 3/5 |
| cold-HBM proxy | 431.739 us | 436.119 us | 0.98891x | 0/5 |

Probe build ID: `c5c91f062196ee42655c2161ba9a85c0cc283706`.
Probe SHA-256:
`1184409d08f07b7d4c2ee1e58cc795ca22a80bb1c9eb2475dbb8c31f4e0ca3fa`.
Kernel build ID: `kb1-25db95c6fd404a401c43dde9c56e02cb`.

Decision: **revert before model integration**. M8 is the scalar/FP32-FMA
resource optimum for this shape. M32/M64 accumulator expansion is closed unless
a new implementation changes ownership or execution units; merely increasing
the template token count cannot recover the required ceiling.
