# Qwen3.5 macOS bring-up

This document fixes the scope and acceptance contract for the native
`Qwen/Qwen3.5-0.8B` port. The first executable slice is intentionally
text-only and correctness-first:

- model revision: `2fc06364715b967f1860aea9cf38778875588b17`;
- device: Apple Silicon CPU with Accelerate-backed matrix multiplication;
- arithmetic: F32 runtime weights and recurrent state;
- context: user-capped, 4096 tokens by default;
- generation: single request, greedy decode;
- excluded from the initial CPU tracer: vision, video, MTP, quantization, and
  Metal. Later sections document separately gated MLX and Metal lanes; they do
  not expand the native text-only model claim.

The Hugging Face config and the official Transformers implementation are the
semantic authority. OMoE is used as performance and verification evidence.
Its current main and later Mac branches do not carry a root license; an older
`origin/public` snapshot is MIT and contains the same Qwen3.5 Python-model
blob. To keep provenance unambiguous, this port still reimplements the public
model equations and verifies them independently in ApxInf instead of copying
the later branch code.

## Hybrid execution contract

Qwen3.5-0.8B has 24 text layers: 18 Gated DeltaNet layers and six full
attention layers, in a repeating `linear, linear, linear, full` schedule.

The runtime therefore owns two state families:

1. six ordinary K/V caches, indexed by the ordinal of a full-attention layer;
2. one convolution suffix and one F32 recurrent matrix for each linear layer.

The following invariants are fail-closed:

- `start_pos` must equal both the hybrid-state position and K/V-cache length;
- `start_pos + sequence_length` must not exceed the configured context cap;
- only `model.language_model.*` and an optional `lm_head.weight` are loaded;
- `model.visual.*` and `mtp.*` are excluded by the loader, not silently used;
- reset clears both state families and restores deterministic first-token
  behavior.

## OMoE lessons incorporated

The local OMoE snapshot is `e2d2bdf15270170451d07e1c230344618a12558d`.
The following ideas affect this port.

### Correctness substrate

- Keep ordinary Qwen3.5 RMSNorm zero-centered: normalize in F32 and multiply
  by `(1 + weight)`. The GDN-local gated norm instead multiplies by `weight`
  and then by `SiLU(z)`.
- Keep partial RoPE rotate-half semantics and preserve the non-rotary tail.
  For text, T/H/W positions are equal; multimodal interleaving is a later
  milestone.
- Normalize Q and K explicitly before GDN recurrence and scale Q by
  `1 / sqrt(key_head_dim)`. Do not hide normalization inside a substituted
  kernel until its numerical path has passed the model oracle.
- Treat external GDN state layouts as an adapter boundary. A backend may use
  a batch dimension, but the canonical per-request layout is `[H,K,V]` and
  conversion to an external `[B,H,K,V]` layout must be explicit and tested.
- Match reduction dtype, cast position, multiply/add order, and cache state at
  the first divergent seam before considering a fused Metal kernel.

Relevant OMoE evidence:

- `recipes/gdn-prefill-flashinfer-conventions.md`
- `recipes/cross-backend-norm-rope-cast-parity.md`
- `omoe/model_qwen35.py`
- `omoe/ops/gdn.py`

### Performance sequence after correctness

1. Cache F32 norm weights and RoPE tables by context bucket.
2. Batch token-independent GDN/attention projections around the strictly
   ordered recurrent or K/V update. Never batch a state transition itself.
3. Fuse gate/up weights for decode only after the exact target shape passes
   the token oracle. Keep a split path for prefill when contiguous norm inputs
   are more valuable than fewer launches.
4. Implement a Metal Q/K norm plus partial-RoPE kernel for Qwen3.5 geometry
   (`Q=8`, `KV=2`, `head_dim=256`, `rotary_dim=64`), with a kill switch and a
   CPU oracle per shape.
5. Implement recurrent GDN state residency on Metal only after profiling shows
   state traffic is the accepted-tip bottleneck.
6. Consider M=1 weight-only quantization only for roles whose complete greedy
   trajectory passes. Prefill and decode use separate dispatch policies.

Relevant OMoE evidence:

- `recipes/batch-token-independent-projections-around-causal-state.md`
- `recipes/hybrid-shared-register-state-residency.md`
- `recipes/hoist-weight-preparation-across-causal-prefill-tiles.md`
- `recipes/projection-weight-fusion.md`
- `recipes/weight-only-w8-scoped-by-m-metal.md`
- the branch `target/qwen3-06b/m1pro/b1-p512-d128-mlx`

The Qwen3-0.6B Metal kernel in that branch is not directly reusable: it assumes
16/8 heads, head dimension 128, and full 128-dimensional RoPE. This port must
use Qwen3.5's gated query layout and partial multimodal RoPE geometry.

The executable Qwen3.5/MLX files on OMoE `main` remain byte-identical to the
MIT `origin/public` snapshot. The later August 22-23 work is primarily recipes
and experiment evidence, so it is treated as design input rather than copied
runtime source.

### Latest M2 campaign evidence

The Qwen3.5 M2 campaign, first audited on
`origin/codex/qwen35-m2-omoe-results`, is now merged into OMoE `main`. Its
frozen result evidence measures Qwen3.5-2B Q4_K_M through a pinned llama.cpp/Metal
adapter on a fanless 16 GiB Apple M2. Two exact, correctness-preserving
candidates did not clear the preregistered end-to-end gate:

- exact single-token Q4 projection kernels produced a balanced median decode
  ratio of `1.026709x`, with only `3/6` positive blocks;
- the standalone F32 `float4` SwiGLU path produced `0.999523x`, also with only
  `3/6` positive blocks.

These results are not directly comparable to ApxInf's 0.8B F32/Accelerate
runtime, but they tighten its optimization policy: an isolated operator win is
only eligibility evidence. Promotion requires a same-binary fallback, exact
semantic trajectory, prompt guardrail, alternating-order repeats, and an
end-to-end threshold on the target Mac. In particular, the branch does not
justify adding a Q4 or vectorized SwiGLU path to ApxInf before its own profile
shows sufficient product-level headroom.

The latest OMoE `main` also records role-selective APXInf wins on RTX 4090:
batch token-independent mixer/gate/up projections around the causal state, but
not blanket replacement of every projection family. The transferable lesson
is to keep state transitions ordered and optimize only roles whose complete
trajectory passes; the CUDA kernels and their absolute timings do not transfer
to Apple Silicon.

OMoE `main` also added a conditional full-context FP8 KV-cache recipe. Its
per-row scaling halves the KV payload and passed a layer-level consumer gate
on RTX 4090, but it is not evidence for this F32 CPU runtime or for a complete
128K deployment. It is retained only as a future Metal/long-context candidate:
direct-append storage, full-model token/quality parity, capacity headroom, and
target-Mac measurements remain mandatory before adoption.

## Current macOS tracer bullet

The native CPU/Accelerate slice now includes:

- strict nested `qwen3_5` config and exact 320-tensor text schema;
- filtered SafeTensors loading that does not materialize vision or MTP tensors;
- canonical K-slot convolution caches and F32 `[H,K,V]` recurrent state;
- the 18 Gated DeltaNet plus six full-attention execution schedule;
- Python-compatible Hugging Face chat-template methods without whitespace
  rewriting; `Hello` and `你好` match Transformers byte-for-byte and token-for-token;
- F32 GEMM operands borrowed without per-call weight copies;
- CPU tensor storage shared with copy-on-write semantics, so reshape and
  same-device transfer no longer duplicate activation or weight buffers;
- flat contiguous K/V storage with bulk row appends and reused attention-score
  scratch, replacing pointer-heavy nested vectors and per-head allocations;
- Apple Accelerate decode now uses grouped GQA BLAS from the first cached
  token for Qwen3.5-0.8B's exact `8Q/2KV/256D` geometry; prefill, other
  geometries, and OpenBLAS-only builds retain the conservative
  128-token cutoff. The target-shape primitive and noisy-host real-model paired
  screen are recorded in
  [`qwen35-short-gqa-sdpa-accelerate-diagnostic-20260826.json`](./qwen35-short-gqa-sdpa-accelerate-diagnostic-20260826.json);
- a follow-up attempt to retain each grouped-softmax exponential in place was
  bit-identical but failed its all-positive paired trajectory gate, so the
  runtime change was rejected; see
  [`qwen35-sdpa-single-exp-rejected-diagnostic-20260826.json`](./qwen35-sdpa-single-exp-rejected-diagnostic-20260826.json);
- tied embedding/output projection sharing through a transpose-B GEMM, avoiding
  an approximately 0.95 GiB duplicate for the official checkpoint.

This slice has now passed the pinned real-checkpoint parity gates below. It is
still only a text-only CPU/Accelerate tracer: vision, MTP, a full Metal model
body, and a service API remain outside the claim. A separately gated Metal W8
output-head sidecar is described below.

## MLX provider screen on the target Mac

The same pinned Hugging Face bundle now also has a request-level macOS screen
through official `mlx-lm` 0.31.3 and `mlx`/`mlx-metal` 0.32.1. This is a
provider path, not evidence that the ApxInf tensor backend has become Metal:
the worker loads only a local directory, receives raw token IDs over a strict
one-line JSON contract, clears ambient credentials and proxies, requests
Hugging Face local-files/offline behavior, and disables remote code. This is a
trusted, pinned-worker policy rather than an OS-level network sandbox. The
native CPU implementation remains the independent correctness oracle and
fallback.

The provider is now connected to `apxinf generate --provider mlx` through a
Rust process boundary that clears the ambient environment, sets the trusted
worker's offline/local-files policy, starts a fresh process group, bounds both
output streams and cleanup time, rejects schema drift, and cross-checks prompt
tokens, `config.json`, interpreter, runner, and request fields. The isolated
toolchain uses a direct (non-symlink) Python executable and eight fixed package
versions so the boundary does not silently escape or drift from its verified
environment. A real invocation through the ApxInf CLI reproduced the frozen
ten tokens at 54.07 token/s.

MLX-converted bundles exposed another portability detail: recent converters
store the chat template in the standard standalone `chat_template.jinja`
rather than embedding it in `tokenizer_config.json`. The ApxInf tokenizer now
supports both forms. The W8 and W4 bundles consequently render the same
13-token `Hello` chat prompt as the source checkpoint instead of incorrectly
treating it as one raw token.

Directly loading the original mixed-dtype checkpoint preserved the frozen
ten-token trajectory exactly. Across six warmed, forced 100-token runs it
measured a median 52.37 decode token/s, with 52.21-54.13 token/s observed,
about 2.96 times the current 17.72 token/s native median. MLX reported roughly
1.56 GB peak allocation and the process reached about 1.92 GB RSS with zero
swaps. This BF16 provider is therefore the default-eligible fast Mac route.

Affine group-64 quantization was also screened, but remains explicit rather
than silent:

- W8 measured a 92.63 token/s median (`1.77x` over MLX BF16). On the exact
  cached 128-step teacher-forced path it matched BF16 top-1 at 126/128 steps
  and changed the free-running tenth token. Reprojecting its hidden states with
  the original BF16 head still matched only 126/128, proving that output-head
  reranking cannot repair the remaining body drift.
- W4 measured a 154.31 token/s median. A later state-aligned attribution found
  only 105/128 self-consistent BF16 top-1 matches (superseding the earlier
  114/128 screen); its hidden-state cosine fell to about 0.951. It is retained
  only as a speed-first experiment.

A newly rebuilt, source-locked W4/G64 bundle also failed the stricter
four-domain production-semantics gate. Both repeats were deterministic, but
the exact BF16 prefix was only 2/64 English tokens, 1/64 Chinese tokens, 0/32
Python tokens, and 10/32 structured-math tokens; position-wise agreement was
6.25%, 3.125%, 0%, and 31.25%, respectively. W4 is therefore an optimization
search baseline, not a deployable default. The immutable comparison is in
[`qwen35-w4-multi-prompt-quality-v1.json`](./qwen35-w4-multi-prompt-quality-v1.json),
and the deliberately narrow gate contract is documented in
[`mlx-multi-prompt-quality-gate-v1.md`](./mlx-multi-prompt-quality-gate-v1.md).

An additional conversion audit found that the ordinary `mlx_lm.convert`
dtype path casts the 18 Gated DeltaNet gated-norm tensors (2,304 values) from
their checkpoint F32 dtype to BF16. Direct checkpoint loading does not. Saving
the already-sanitized model without that blanket cast restores the exact BF16
100-token trajectory; quantized bundles should use the same mixed-dtype
procedure. All artifact hashes, raw samples, memory values, trajectory hashes,
quality metrics, and limitations are frozen in
[`mlx-provider-screen-20260824.json`](./mlx-provider-screen-20260824.json).
The repository also contains a deterministic bundle builder that preserves
the mixed source dtypes, validates every transformed tensor, pins the eight
verified inference-critical packages, and publishes with no-replace semantics.
No exploratory bundle is promoted merely because it loads or benchmarks
quickly.

The pinned MLX worker also has a separate persistent-process and exact-append
session slice. Ordinary generation remains cache-free. Session requests must
bind the loaded model, greedy strategy, session ID, and the exact complete
token prefix; only a non-empty suffix is reused. Arbitrary prefix trimming and
branching are rejected because Qwen3.5's 18 recurrent Gated DeltaNet caches
cannot be rolled back safely. A real two-turn cached generation matched a
fresh full-prompt generation token for token. The versioned contract, bounded
LRU policy, failure invalidation rule, and noisy-host limitations are recorded
in [`mlx-session-prefix-cache-20260824.md`](./mlx-session-prefix-cache-20260824.md)
and
[`mlx-session-prefix-cache-evidence-20260824.json`](./mlx-session-prefix-cache-evidence-20260824.json).

That persistent service is exposed as a local application boundary through
`apxinf mlx-serve`. It accepts raw token-ID operations over bounded stdin/stdout
JSONL and wraps only Rust-validated service receipts. It deliberately does not
open an HTTP/OpenAI/SSE endpoint. The command and complete request examples
are documented in
[`mlx-serve-cli-20260824.md`](./mlx-serve-cli-20260824.md).

## Native long-context GQA screen

The CPU K/V cache now uses one flat F32 allocation in
`[layer, kv_head, position, head_dim]` order. At 128 tokens and above, decode
groups the four Qwen query heads sharing each KV head and computes QK and PV
with Accelerate SGEMM; shorter sequences keep the scalar path. A same-process
alternating-order primitive screen at the exact `Q=8`, `KV=2`, `D=256`
geometry measured `13.01x`, `15.81x`, `17.09x`, and `11.18x` speedups at
contexts 128, 512, 1024, and 4096. Maximum absolute error against the retained
scalar reference stayed below `1.2e-7`.

This is an operator screen, not a token-throughput result. Full-model oracle
and quiet-host long-context ABBA/BAAB gates remain mandatory. The exact source
hashes, orders, repetitions, and measurements are in
[`native-gqa-sdpa-screen-20260824.json`](./native-gqa-sdpa-screen-20260824.json).

## Native Metal W8 output-head slice

The first native Metal slice keeps the 24-layer Qwen3.5 body on
CPU/Accelerate but replaces the first-token and decode-time tied vocabulary
projection—the largest single native hotspot—with a persistent group-64 W8
Metal kernel. The GPU returns the global top four and the CPU recomputes just
those scores from the original F32 embedding. Each step transfers one 4 KiB
normalized hidden row to Metal and reads back 16 bytes; full 248,320-way logits
are not materialized on the CPU. It is feature gated and remains explicit
opt-in through `--metal-w8-lm-head`.

The frozen v1 top-one binary matched the native F32 head on the 128-step
teacher gate, and all 28 recorded 100-token trajectories were identical. Its
accepted same-binary ABBA/BAAB set measured 17.24 versus 22.89 token/s median
(`1.3277x`), reducing median generation latency by 24.45%, with about 4.7 GB
peak RSS and zero swaps. Six blocks were same-direction after retaining the
contaminated original block 6 and using a predeclared replacement; no raw
sample was deleted.

The current v2 source adds globally correct deterministic GPU top-four and
exact native-F32 row reranking. Adversarial Metal tests, the 128-step native
teacher gate, and the production 100-token trajectory pass. The shared
generation loop also uses this path for the first prompt result, avoiding a
full CPU vocabulary projection; a real ten-token smoke kept the frozen IDs and
reported a directional 95.90 ms TTFT. V2's formal ABBA/BAAB performance gate
is deliberately deferred because the host was contaminated by desktop agents
and existing swap. V1 performance is not silently reassigned to the new
binary.

Neither version is claimed as BF16/Hugging Face exact: one observed BF16
boundary selects a different token from native F32 even before quantization.
Frozen v1 evidence is in
[`../../crates/apxinf-metal/evidence/qwen35-0.8b-m4-lm-head-20260824.json`](../../crates/apxinf-metal/evidence/qwen35-0.8b-m4-lm-head-20260824.json),
and the current correctness/pending-performance record is in
[`../../crates/apxinf-metal/evidence/qwen35-0.8b-m4-lm-head-20260824-v2.json`](../../crates/apxinf-metal/evidence/qwen35-0.8b-m4-lm-head-20260824-v2.json).

## Native complete-MLP and combined Metal W8 slices

The next explicit native slice moves each decode-time MLP as a complete block:
W8 gate+up, fused SiLU-times-up, and W8 down execute in one command buffer per
layer. Attention, recurrent state, residual addition, and prefill remain
CPU/F32. `--metal-w8-mlp-block` selects all 24 blocks; combining it with
`--metal-w8-lm-head` selects the already-verified full MLP plus v2-head tracer
in one model construction. Neither flag changes the default path.

The frozen pre-CLI combined binary passed its CPU teacher 128/128 and matched
the direct 128-token CPU trajectory. Each free run recorded 24 layers times
127 decode hits plus one head prefill and 127 head decode hits, stayed below
6 GiB RSS, and reported zero process swaps. Its noisy-host 3+3 screen measured
18.9539 versus 32.0277 token/s median (+68.98%, 3/3 pairs positive), so it is
promising but still awaits quiet-host formal admission. Strict F32 parity was
not restored; this remains an explicit quantized-quality lane.

The product CLI now fails closed for feature-missing builds, MLX, CUDA, BF16,
and non-Qwen3.5 requests. `--json` reports both selected flags and actual MLP
and head hit counts. This final CLI/loader wiring was covered by synthetic and
tiny-model tests only; the real checkpoint was deliberately not rerun after
the wiring-only change. The frozen model evidence remains
[`../../crates/apxinf-metal/evidence/combined/qwen35-0.8b-m4-combined-summary-20260824.json`](../../crates/apxinf-metal/evidence/combined/qwen35-0.8b-m4-combined-summary-20260824.json).

## Acceptance ladder

Each optimization starts from the most recently accepted implementation.

1. **Primitive gate:** compare CPU kernels with small F32 fixtures, including
   one-shot versus incremental convolution and recurrence.
2. **State gate:** compare full-prompt forward with prompt-plus-decode and
   verify reset reproducibility.
3. **Teacher-forced gate:** compare selected hidden states and logits with a
   pinned Transformers oracle.
4. **Greedy gate:** require the first ten generated token IDs to match exactly,
   then expand to a frozen prompt matrix.
5. **Product gate:** measure TTFT, token latency, peak resident memory, and swap
   use on the target Mac. An operator microbenchmark cannot promote a runtime
   change by itself.

Metal work begins only after gates 1-4 pass for the CPU implementation.

## Pinned checkpoint result on the target Mac

The first target-machine run used the official shard with SHA-256
`04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696` and
the runtime combination PyTorch 2.13.0, Transformers 5.15.1, SafeTensors
0.8.0, Rust release mode, and Apple Accelerate.

The frozen oracle currently passes all of the following:

- all prefill rows: maximum logit error below `1.5e-4`, normalized RMSE below
  `4.5e-6`;
- cached decode: maximum logit error below `4.3e-5`, normalized RMSE below
  `1.9e-6`;
- 108 convolution-state comparisons and 36 recurrent-state comparisons;
- first-step and cached-probe top-20 overlap `20/20`, with identical top-1;
- cached decode versus fresh one-shot execution;
- the ten-token greedy trajectory
  `[9419, 0, 2500, 628, 353, 1438, 488, 3242, 30, 25677]`.

The verifier has frozen, category-specific limits and exits non-zero on a
failure; threshold overrides are intentionally unsupported.

For the 13-token `Hello` chat prompt, the first accepted CPU implementation
measured roughly 0.20-0.48 seconds TTFT and 55-69 ms per decode token (about
14.5-18.2 token/s).

An OMoE-guided target-machine sample then identified the scalar Gated DeltaNet
recurrence as the largest non-BLAS decode cost: 367 of 3,062 top-of-stack
samples, versus 1,569 samples in Accelerate/BLAS. The accepted CPU candidate
keeps the canonical `[H,K,V]` state and the same per-value reduction order, but
walks V contiguously inside each K row. Its complete Transformers/state/top-20
and exact ten-token oracle remained green. In a balanced 12-run sequence with
100 generated tokens per run, the six control runs had median 16.479 token/s
and the six candidate runs had median 18.057 token/s (`1.0958x`); the slowest
candidate exceeded the fastest control. A follow-up sample reduced recurrence
top-of-stack observations from 367 to 88. This is the first ApxInf-native Mac
optimization promoted from the OMoE-derived state-layout hypothesis.

A second balanced 6-control/6-candidate run is preserved in
[`gdn-contiguous-benchmark-20260824.json`](./gdn-contiguous-benchmark-20260824.json).
It measured 16.663 versus 17.976 token/s median (`1.0788x`); the slowest
candidate run was still faster than the fastest control run, and all twelve
100-token trajectories were identical. This receipt records every sample,
binary identity, command contract, and the remaining environmental limits.

The next OMoE-derived hypothesis was screened and deliberately rejected rather
than promoted. A real-shape Apple Accelerate comparison of batch-one SGEMV
against the existing SGEMM dispatch covered every major Qwen3.5 projection and
the tied vocabulary head. Four independent, nine-pair screens projected only
about 1.2% median weighted improvement, well below the 5% candidate threshold;
the complete output vectors were identical. The shape/call contract and raw
weighted ratios are preserved in
[`accelerate-sgemv-screen-20260824.json`](./accelerate-sgemv-screen-20260824.json).
No ApxInf dispatch code was changed. This leaves decode-only W8 and, for longer
contexts, flat/grouped KV attention as the next evidence-backed candidates.

The current release binary is 8,163,904 bytes with SHA-256
`d9cb4de44b236b5b3f216a81079b11102220939a2b179cbc2678442ff947803b`.

The earlier live v2 onboarding run published the model at a new absolute path, then
passed both the generation gate and the independent Seatbelt memory smoke.
Its deployment lock is
`.apxinf/deployments/qwen35-0.8b-macos-cpu/deployment-lock-staged-v3.json`
(content SHA-256
`c7e14b676fb42567e973495939f662412a280bea6857a9a7604870bcbedee3c2`).
The memory smoke measured peak RSS `4,691,607,552` bytes, zero child swaps,
about 218 ms TTFT, and 18.23 token/s. A second fully offline, existing-only run
re-hashed all 1,759,828,853 bundle bytes without network access and produced
`deployment-lock-reused-offline-v4.json` (content SHA-256
`4e88dd1d90b2de3e7de82a2cfcd4ee9d9f583e9192581c35220ba85426448f3b`).
Those two locks remain immutable historical evidence for the previous binary;
the contiguous-state release was revalidated fully offline as
`.apxinf/deployments/qwen35-0.8b-macos-cpu/deployment-lock-gdn-contiguous-v5.json`
(content SHA-256
`c4089eb6a6f181ac4fb7aa9087beebd9e92e8c0f391e2dae52c92b44139622e0`).
That run re-hashed all bundle bytes, generated the exact trajectory, measured
17.66 token/s with about 170 ms TTFT, used 4,722,819,072 bytes peak RSS, and
recorded zero child swaps.

The staged run downloaded 12,875,591 remaining metadata/tokenizer bytes in
three real HTTPS requests. Before that run, the raw downloader had fetched and
strictly resumed a 40,894,464-byte checkpoint prefix; to finish the end-to-end
test in bounded time, that partial was replaced by an APFS clone of the
already-local checkpoint after the prefix and complete checkpoint hashes were
verified. The receipt therefore correctly reports the full checkpoint as
`resumed`, not as bytes downloaded by that invocation. A complete 1.76 GB
fresh network transfer remains a throughput test, not a correctness gap.

These numbers are a bring-up baseline, not a final benchmark contract; a
dedicated benchmark run must control temperature, background load, prompt
set, and cache state.

## llama.cpp diagnostic comparison

The pinned-source raw-token comparison with llama.cpp is documented in
[`qwen35-apxinf-vs-llamacpp-diagnostic-comparison-v2.md`](./qwen35-apxinf-vs-llamacpp-diagnostic-comparison-v2.md).
Its two single-observation lanes reproduce the exact same frozen 128-token
free-run trajectory, while remaining explicitly non-formal because the host,
thread-policy, repetition, and teacher-forced cross-runtime gates are not yet
closed.
