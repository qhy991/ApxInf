# Qwen3.8 Marlin-M64 end-to-end promotion contract

Status: frozen before model-path implementation on 2026-08-18.

## Objective

Reduce single-request text-prefill TTFT for the 1,064-token and 8,232-token
service cells on one RTX 4090 by replacing the repeated M8 MLP W4A16
projections inside each decoder layer with a 64-row Marlin scheduling boundary.
The primary metric is no-profiler client TTFT. TPOT is a regression guard, not
the optimization target.

## Invariants

```text
model                 cyankiwi/Qwen3.8-27B-AWQ-INT4
model revision        63768c10df38c0395e12ef49edac1bd539eaeeea
source baseline       715a0ed790f2d10d82fab53fbeac3da3075adf26 + dirty optimization overlay
GPU                    one RTX 4090 / SM89 / 24 GiB
parallelism            TP=PP=DP=1
quantization           compressed-tensors asymmetric U4 group-32
activation/output      BF16
KV                     BF16, 32K resident capacity
execution              eager, one stream, one request, no continuous batching
sampling               greedy argmax, unchanged tokenizer/chat template
decode                 accepted stateful M1 graph, unchanged
baseline prefill       M8 tiles plus M1 tail
candidate prefill      M64 MLP tiles, then explicit M8/M1 tail
timing                 client send to first non-empty SSE output
```

The candidate may change INT4 accumulation order only within the already
declared exploratory numerical contract. It must preserve recurrent state,
convolution state, KV writes, causal positions, residual mutation, final-row
publication, prompt/output lengths, request reset, and every decode operation.

## Non-goals

- multimodal execution, concurrency, cancellation, continuous batching, CUDA
  Graph, prefix caching, and contexts above 32K;
- M64 mixer projections in the first candidate; GDN and attention mixers retain
  their accepted M8 decomposition;
- claiming a general vLLM win from an operator, layer, or 1K-only result;
- prepacking every checkpoint weight resident at once, which would duplicate
  the INT4 model and exceed the 24 GiB memory contract;
- Direct PTX authorship or a PTX-exclusive performance claim.

## Sources of truth

- runtime source: `crates/apxinf-model/src/qwen35/decode.rs` and
  `src/qwen35_server.rs`;
- Marlin transform/kernel: `crates/apxinf-cuda/src/kernels/gemm/marlin.rs` and
  `crates/apxinf-cuda/adapters/marlin_adapter.cu`;
- workload/spec: `benchmarks/qwen38_4090/spec.json` and generated dataset
  manifest SHA-256
  `c5d48c9f7d42823141f59b066094e93d589dfb96a7d68166ffb8574cf9f52eff`;
- accepted service baseline: `benchmarks/qwen38_4090/SERVICE.md` and the
  `upstream_715a0ed` evidence bundles;
- operator evidence: `native_qwen35_marlin_m64.json` plus a fresh build probe;
- upstream Marlin provenance: `crates/apxinf-cuda/kernels/marlin/README.apxinf.md`.

## Minimum primitives

1. `M8` and `marlin-m64` request modes under one resident model and one
   explicit, default-off service capability switch.
2. A fail-closed selector: requesting `marlin-m64` without an enabled SM89
   workspace, a prompt of at least 64 tokens, or exact supported shapes is an
   error. The declared M8/M1 remainder is a visible decomposition, not fallback.
3. One reusable prepared-weight buffer for each MLP shape, rather than one
   duplicate prepared copy per layer.
4. One reusable M64 activation/workspace and zero-copy M8 row views for the
   stateful mixer and residual seams.
5. Path proof in every response/log: selected mode and M64/M8/M1 tile counts.
6. A 64-token complete-stack oracle and an alternating service A/B harness.
7. The accepted M8 binary/source tree remains runnable until promotion.

## Runtime DAG and hypothesis

Baseline, per layer and per eight M8 tiles:

```text
8 x [M8 mixer -> residual/norm -> M8 gate_up -> 8 activation launches
     -> M8 down -> residual/norm]
```

Candidate, per layer and one M64 tile:

```text
8 x [M8 mixer -> residual/norm]
-> transform gate_up + Marlin M64
-> 64 row-strided activation launches (first proof)
-> transform down + Marlin M64
-> 8 x residual/norm
```

The first candidate removes fourteen large MLP weight-streaming launches per
layer and reuses each transformed weight across 64 rows. It introduces two
runtime transforms per layer and initially leaves activation launch count
unchanged. The isolated gate projection measured 1484.884 us for eight M8 calls
versus 280.924 us for transform plus Marlin M64, so the hypothesis predicts a
material TTFT reduction even without M64 mixer projections.

Reject the hypothesis if the joined 64-layer boundary is not at least 10%
faster, if 1K candidate TTFT does not improve by at least 25%, if it wins fewer
than four of five screening pairs, if TPOT regresses by more than 5%, or if any
correctness/state/path gate fails. A larger boundary that is slower gets one
causal profile and at most one targeted iteration before revert.

## Correctness endpoints

- `M64.final_residual`: all 64 rows after layer 63 and final residual update;
- `M64.final_normalized`: all 64 rows after the final Qwen offset RMSNorm;
- `M64.next_token_normalized`: the following M1 token after committing the last
  prefill row, covering recurrent state and full-attention KV state;
- `service.token_trajectory`: exact greedy token IDs for every emitted step;
- `service.functional`: existing dataset checker, finite output, valid SSE
  termination, no API error, and unchanged usage accounting.

For the three tensor endpoints require 100% finite values, cosine at least
0.999, and relative L2 at most 0.05 against eight accepted M8 tiles. Require an
exact next-token argmax for the frozen 64-token oracle. Record full service
token exact-match rate, but under the user's exploratory INT4 contract the
promotion gate is functional correctness rather than complete token identity.

## Measurement and evidence

1. Build for `sm_89`; record source status, binary SHA/build-id, CUDA version,
   GPU state, and path marker.
2. Re-run the Marlin operator oracle with balanced AB/BA samples.
3. Run the complete 64-token stack oracle, M64+M8/M1 tail smoke, and repeated
   request reset.
4. Run five adjacent alternating M8/candidate 1K pairs in one resident process,
   balanced AB/BA, with raw TTFT/TPOT/E2E/token/path records.
5. If admitted, run at least two balanced 8K pairs as a screen. A formal service
   promotion requires 25 stable pairs per target cell or a separately frozen
   equivalent confidence rule.
6. If the candidate wins, capture one complete Nsight Systems trace to show
   whether the MLP projection interval actually shrank in the joined DAG. Use
   Nsight Compute only for a specific remaining kernel question.

Decision values are `promote`, `continue`, or `revert`. Operator-only or
64-layer-only success remains `continue`; it cannot replace the service path.

## Frozen targeted iteration after the first rejection

The first complete candidate applied Marlin to both `gate_up` and `down`. It
passed at 1, 4, and 16 layers but failed the predeclared tensor gate at 32 and
64 layers:

| Layers | Final residual cosine | Final residual relative L2 | Decision |
|---:|---:|---:|---|
| 1 | 0.99999859 | 0.0016787 | pass |
| 4 | 0.99999605 | 0.0028189 | pass |
| 16 | 0.99998749 | 0.0050173 | pass |
| 32 | 0.99930399 | 0.0373076 | final-normalized gate failed |
| 64 | 0.97014074 | 0.2447246 | reject |

The exact first and second argmax at 1/4/16 layers and exact M8 mixer state
path rule out row-view, position, recurrent-state, and KV scheduling as the
first cause. The bounded second candidate therefore applies Marlin only to the
larger `gate_up` projection and restores the accepted M8 accumulation order for
`down`, which writes directly into the residual edge. This is a new registered
mechanism, not a relaxed correctness threshold. All original thresholds and
the 1.10x joined/full-stack stop rule remain unchanged. If the 64-layer gate or
speed rule fails, the M64-MLP branch is closed and reverted without service A/B.
