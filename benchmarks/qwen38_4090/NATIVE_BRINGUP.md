# Native ApxInf bring-up and optimization record

This record tracks the native path separately from the accepted vLLM reference
baseline. A local operator result, a valid checkpoint, or a successful build is
not promoted to a native serving claim until the complete request and token
trajectory pass through ApxInf.

## Frozen promotion contract

The model revision, RTX 4090, TP/PP/DP=1, request manifests, output budgets,
sampling, FP8 KV policy, and client timing boundaries remain those in
`spec.json`, `BASELINE.md`, and `CONTEXT_LIMIT.md`. The reference and candidate
must use the same input hashes. Admission timing is no-profiler wall time.

The initial candidate cells are:

| Cell | Prompt/output | Primary metric | Correctness |
|---|---:|---|---|
| decode | 1K/128 | TPOT, decode tok/s | complete greedy token trajectory |
| balanced | 8K/128 | TTFT and TPOT | complete greedy token trajectory |
| long prefill | 32K/128 | TTFT, prefill tok/s, completion | complete greedy token trajectory |
| multimodal | `mm-shape-count` | TTFT and completion | exact deterministic answer |

Unsupported configuration must fail before allocation. The native and optimized
paths will remain explicit selectors; optimized mode may not silently fall back.
The first screen requires five alternating pairs with at least four wins and a
lower median. Formal promotion requires 25 stable-machine pairs or an equivalent
predeclared confidence rule, plus causal Systems/Compute evidence.

## 2026-08-17 loader/config slice

The first native slice establishes the authoritative model boundary without
allocating the 21 GB checkpoint:

1. `DType` and `Tensor` preserve SafeTensors I32/I64 bits. Integer tensors do
   not implicitly convert to floating point or enter plain cuBLAS GEMM.
2. `safetensors::inspect_path` reads only each shard header, validates safe shard
   paths, exact index ownership, unique names, dtype/shape byte lengths, file
   bounds, and the index `total_size`.
3. `Qwen35Config` strictly requires the hybrid multimodal identity, the declared
   3-GDN/1-full-attention schedule, mRoPE coverage, and compressed-tensors
   pack-quantized W4A16 group-32 asymmetric contract.
4. Every packed linear must have an exact four-tensor bundle:
   `weight_packed:I32`, `weight_scale:BF16`, `weight_zero_point:I32`, and
   `weight_shape:I64`. The logical shape is read from the 16-byte shape tensor
   and checked against all physical packed axes.
5. CLI dispatch recognizes `qwen3_5`. `generate`, including image input, fails
   closed until execution kernels exist; it can no longer fall through to
   `GeneralLlama` or the Qwen3-VL implementation.

Remote SM89 command:

```bash
/root/apxinf-target-sm89/release/apxinf inspect \
  --model /mnt/user_dir/hanjinchen/models/cyankiwi/Qwen3.8-27B-AWQ-INT4 \
  --json
```

Observed result in 0.437 seconds:

```text
status                    validated
shards                    5
tensors                   2396
validated tensor bytes    21017689808
dtype counts              BF16=1199, I32=798, I64=399
quantized linears         399
text layers               48 GDN + 16 full attention
vision depth              27
native_execution_ready    false
```

The exact machine-readable receipt is `native_contract.json`. The SM89 release
binary has build ID `70fadb0f85b20576fd8b46512cd88f535e1743c4` and SHA-256
`4d83eef615bff6861002f516c68477dc8eb55239a0c9dc2c2f45b86ca84ece8d`.

## 2026-08-17 first native W4A16 operator

The real layer-0 gate projection now runs through ApxInf without materializing a
BF16 weight. Compile-time group-32 specialization reduced the initial hot-L2
control from 43.7551 to 31.6466 microseconds. A shared-staging counterfactual
then passed its frozen cold/hot gates and all five alternating AB/BA pairs:

```text
direct cold / hot    90.0995 / 31.6466 us
staged cold / hot    84.2945 / 29.1354 us
median pair speedup  1.0693x cold, 1.0962x hot
correctness          cosine 0.999999999993, relative L2 3.76e-6
resources            40 registers, 13,440 B dynamic shared, zero spill/local
decision             staged operator default; direct rollback; no E2E claim
```

See `W4A16.md` for the contract, raw evidence hashes, SASS boundary, NCU
permission blocker, and promotion decision.

## 2026-08-17 recurrent GDN core

The one-token recurrent gated-delta core now passes a 128-step mutated-state
oracle on the RTX 4090:

```text
output cosine / relative L2  0.99999999785 / 6.56e-5
state cosine / relative L2   0.9999999999997 / 4.35e-7
input immutability           byte-exact
hot / cold proxy             11.0015 / 26.090 us
resources                    28 regs, 1,056 B shared, zero spill/local
```

This is the recurrent core only. `GDN_CORE.md` freezes its equations, named
endpoints, resource/timing evidence, and the required complete-layer boundary.

## 2026-08-17 complete GDN layer

Layer 0 now runs from `[1,5120]` hidden input through real QKV/Z/a/b/conv/norm/
out-proj weights and both mutable states. The BF16 semantic baseline passes all
15 endpoints at 126.6465 microseconds. Systems attributes 56.2% to the 60 MiB
BF16 out-proj.

W4A16 and W8A8 out-proj counterfactuals failed the frozen accuracy gate.
Weight-only W8A16 passed accuracy but initially missed the 100-microsecond
performance gate. Packing a/b into one GEMV and fusing conv4 through prepare
reduced the admitted opt-in to 98.4772 microseconds across five alternating
pairs, a median 1.2874x layer speedup with 5/5 wins.

This is a layer opt-in, not a model promotion. See `GDN_LAYER.md` for all
endpoints, negative candidates, Systems attribution, raw hashes, and the next
four-layer repeating-unit boundary.

## 2026-08-17 real MLP sublayer

The layer-0 gate/up checkpoint bundles are now concatenated without
requantization and executed as one `[34816,5120]` W4 projection, followed by the
existing BF16 SwiGLU kernel and real down-projection W4. All three endpoints
pass; final output cosine is `0.999999999610` and the no-profiler sublayer median
is `179.8626 us`. See `MLP_LAYER.md`.

## 2026-08-18 real full-attention layer and SM89 split scheduling

Layer 3 now runs the real one-token full-attention module from hidden input
through q/k/v W4, offset Q/K RMSNorm, 64-dimension partial RoPE, exact BF16 KV
append, 24/4-head GQA, sigmoid output gate, and o-proj W4. Ten named endpoints,
the appended slot, cache sentinels, and hidden-input immutability pass at 1K,
8K, and 32K.

Systems found that the incumbent assigns only 24 CTAs to the 128-SM 4090. At
32K, its attention kernel consumes 2,309.577 microseconds and 96.3% of module
GPU time. A caller-workspace split/reduction rewrite exposes 24x16=384 stage
CTAs and a 24-CTA FP32 merge. The admitted stage uses 40 registers and 9,280 B
shared; merge uses 38 registers and 80 B shared; both have zero spill/local.

Five 32K AB/BA pairs pass correctness and improve the complete attention module
from a 2,404.1670-microsecond median to 353.1917 microseconds, a 6.8031x median
speedup with 5/5 wins. The screened opt-in selector keeps the incumbent below
KV bucket 256 and uses split16 at or above 256; KV=256 and 512 also win 5/5
pairs. See `ATTENTION_LAYER.md`.

This layer proof uses BF16 KV. The frozen vLLM comparison uses FP8 KV, so the
candidate is not yet memory-equivalent or eligible for server-level promotion.
FP8 KV storage/consumption and the complete token trajectory remain required.

## 2026-08-18 real four-layer hybrid unit

The first production-shaped scheduling boundary now composes real layers 0–3:

```text
GDN -> GDN -> GDN -> full attention
```

Every layer includes Qwen's `(1+weight)` input/post-attention RMSNorm, both
BF16 residual seams, the real mixer, and the real W4 MLP. Three recurrent/conv
states and one KV cache have independent ownership while all four layers reuse
one activation workspace.

The manifest revealed that only layer 0 GDN out-proj is BF16; layer 1/2 use
checkpoint W4 bundles. Native/optimized final residuals stay within relative
L2 `0.00705..0.00741` across KV 256..32K. At 32K, five alternating pairs
improve the complete four-layer median from 3,597.850 to 1,518.062
microseconds, a 2.3704x median speedup with 5/5 wins. Systems moves the unit GPU
envelope from 3,648.258 to 1,533.696 microseconds and identifies 20 W4
projections as the new 67.7% kernel-time bottleneck. See `HYBRID_UNIT.md`.

## 2026-08-18 complete native text decode

All 64 layers now share one activation workspace while owning 48 recurrent
states, 48 conv states, and 16 KV caches. A selectively loaded checkpoint
embedding row, final offset RMSNorm, streaming W8A16 LM head, BF16 logits, and
CPU argmax close the token boundary.

The model-level selector rejects the layer-0 W8 GDN candidate because it grows
to relative L2 `0.166` after 64 layers for under 0.2% model-time ceiling. The
admitted model arm preserves checkpoint-native GDN weights and changes only
attention. Complete-token optimized throughput is 46.77 token/s at 1K, 44.85
at 8K, and 39.75 at 32K. Native/optimized emit the same token at every cell.

Three continuously mutated 16-token trajectories at prefix 256, 8,192, and
32,752 match 16/16 tokens. The 32K complete-token process uses about 18,061 MiB
and leaves about 6,021 MiB free. `apxinf generate` now accepts a real prompt;
the `Hello` smoke applies the official 53-token chat template and emits
`The user said "` at 56.48 decode token/s. See `TEXT_DECODE.md`.

The resident single-request service now exposes OpenAI-compatible non-stream
and SSE chat completions. Under the unchanged client, the 1K formal median is
23.619 ms TPOT / 42.338 decode token/s with 5/5 success, about 3.6x the frozen
vLLM decode throughput. The same run has 21.359 s TTFT because prompt tokens
still execute serially at about 49.8 token/s. The 8K smoke confirms 28.335 ms
TPOT / 35.293 token/s but 167.967 s TTFT. See `SERVICE.md`.

The first M>1 operator now reuses each real gate-projection W4 value across up
to eight tokens. M=2/4/8 are bit-exact to serial M1. M8 improves the hot tile
1.289x and the cold-HBM whole-model proxy 3.079x; M2 loses the hot cell and is
not generally selected. See `PREFILL.md`.

Local validation passed `cargo check --workspace`, 16 core tests, 14 loader
tests, 38 model tests, and a non-CUDA CLI build. The full local workspace test
cannot link the CUDA test binary on the ARM64 Mac without CUDA libraries; the
same source passed remote workspace check and an SM89 release build.

## Execution primitives and order

The minimum native graph is deliberately small:

```text
validated checkpoint
  -> resident packed-linear views and bounded workspaces
  -> W4A16 projection primitive
  -> GDN recurrent-state primitive / full-attention KV primitive
  -> hybrid text layer executor
  -> output head and complete token trajectory
  -> vision encoder / merger / media-token injection
  -> streaming benchmark adapter
```

The packed projection proof covers M=1 and the first M<=8 weight-reuse tile at
K=5120, N=17408. Other projection shapes and complete M>1 layer/model prefill
remain open and are separate cells.

The recurrent core, real GDN/MLP/full-attention modules, repeating unit, all 64
layers, final norm, LM head, and greedy token trajectory are now proven. The
remaining text bottleneck is M>1 prefill; the current functional CLI reuses the
M=1 graph serially across prompt tokens. Multimodal execution remains separate.

## Case-guided optimization policy

The fixed SubCUDA case source is `qhy991/SubCUDA@d1db18fbc46f873d827bc7d276988d5cef3199ab`.
The relevant accepted, attribution, and negative evidence gives these rules:

- Start with CUDA vector I/O and producer-to-consumer layout/fusion. In the GDN
  three-arm case, CUDA recovered about 90% of the Direct-PTX end-to-end saving;
  PTX is not the first implementation choice on SM89.
- Count boundary frequency before tuning. A 33% one-per-step fusion was rejected
  because its projected end-to-end ceiling was only 0.0226%.
- Do not promote an isolated launch/materialization win. A byte-exact fusion
  saved 1.327 microseconds locally and still lost its joined TP2 graph.
- Avoid speculative cache hints/prefetch. Twenty GDN mutations produced no
  admitted winner; several cache and prefetch arms regressed.
- Reduced weight bytes do not guarantee a faster decode. A pure-PTX FP8 QKV
  path halved DRAM reads but lost matched event time after unpack/conversion.
- Any later PTX candidate must prove exact source fingerprint, final SASS,
  registers, spills, selected runtime entry, matched operator timing, and an
  end-to-end win. An equivalent CUDA counterfactual is mandatory.

## Current gate and next bounded action

The loader/config, all decode operators, first hybrid unit, complete 64-layer
stack, final norm, LM head, exact multi-token trajectory, and CLI text-
generation and resident single-request service gates are complete. Native
**text decode/service** execution is ready. Native multimodal and M>1 prefill
remain false; concurrency and online tail/goodput are untested.

The next bounded actions are:

```text
M>1 prompt path for W4 projections + GDN scan + full attention
  -> matched TTFT retest on the existing resident worker
  -> concurrency / graph / tail-latency service work
```

The frozen vLLM baseline uses FP8 KV while the current ApxInf text path uses
BF16 KV and a W8-converted LM head. Those differences must remain explicit in
the comparison contract; API-level superiority is not promoted until the same
client workload runs through the resident ApxInf service.
