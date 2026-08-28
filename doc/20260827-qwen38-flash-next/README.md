# Qwen3.8-Flash-Next native bring-up

Date: 2026-08-27
Status: active, synthetic-first

## Contract

Goal: add one native ApxInf text path for `Qwen/Qwen3.8-Flash-Next`
(`model_type=qwen4_exp`) and optimize it from reproducible measurements.

The canonical runtime owner is `crates/apxinf-model/src/qwen4_exp/`. The
shared `AutoModel` registry remains the only public loading path. A synthetic
fixture may reduce dimensions and checkpoint storage, but it must execute the
same architecture implementation as a real checkpoint; there is no parallel
dummy model.

Current non-goals are MTP serving, distributed loading, and a real-checkpoint
quality or end-to-end throughput claim. The released checkpoint is about
360 GB before runtime state, while this host has less than 45 GiB free. No
weight shard is downloaded for this bring-up.

## Invariants

- Parse the released nested `qwen4_exp` / `qwen4_exp_text` schema strictly and
  reject unsupported or internally inconsistent topology.
- Preserve the released 3:1 Gated DeltaNet/QSA layer schedule, four residual
  streams, top-10-of-512 routed MoE plus shared expert, one-indexed PLE layer,
  and distinct recurrent, QSA, PLE-convolution, and n-gram states.
- Keep Hugging Face weight-name translation inside the model folder. Shared
  backend primitives must not import model types.
- Fail closed when a requested feature is not implemented. Do not silently
  run Qwen3.8-Flash-Next through the Qwen3.5 path.
- Treat synthetic latency as architecture/dispatch evidence only. It is not a
  correctness or production-performance claim for the 180B checkpoint.

## Acceptance evidence

The first complete slice requires:

1. the released config and weight index pass strict schema validation;
2. malformed schedule, QSA, hyper-connection, PLE, and MoE configs fail;
3. deterministic small synthetic weights run prefill and cached decode through
   the registered `AutoModel` path;
4. the portable workspace checks pass; and
5. a repeatable dummy benchmark names the first measured hotspot before any
   performance change is attempted.

Real support is complete only after a separately provisioned machine loads the
official checkpoint and matches an independent Transformers reference on
per-layer activations and greedy tokens.

## Iteration log

- The official config and 1,658-entry SafeTensors index pass the no-weight
  contract checker. The text path owns 1,294 names; 1,291 are floating runtime
  tensors, while three deterministic PLE integer buffers are reconstructed.
- The CPU QSA selector now implements compressed-block pooling, zero-centred
  RMSNorm, partial RoPE, top-k block selection, and mandatory incomplete-tail
  retention. Its initial dummy optimization reuses per-block scratch and uses
  linear top-k partitioning. See `qsa-selector-dummy-v1.json`; the result is
  selector-only synthetic evidence, not an end-to-end performance claim.
- The registered `AutoModel` `qwen4_exp` route now runs a downsized native
  CPU/F32 text model with deterministic synthetic weights. Its one-shot and
  cached tokenwise paths agree within `1e-6`; reset reproduces logits exactly.
  The checked-in `configs/qwen4-exp/synthetic-tiny.json` fixture exercises one
  PLE layer, three GDN layers, one QSA layer, four residual streams, and
  top-2-of-4 routed MoE.
- The real SafeTensors text route now validates and consumes the published
  tensor layout, including QSA's interleaved per-head query/output gate and
  PLE shard concatenation. All 1,291 BF16 runtime tensor headers across the
  official 131 shards pass without downloading payloads. A frozen official
  Transformers toy checkpoint matches all ApxInf logits to `1.49e-8` max
  absolute error with 5/5 top-1 agreement. See
  `source-contract-and-toy-oracle-v1.json`.
- The official text path contains 176,943,899,520 parameters and the current
  all-F32 pack needed at least 659 GiB. Routed experts now remain in their packed
  checkpoint BF16 tensors and PLE performs row lookup across the original 128
  BF16 shards, without whole-table concatenation. All other checkpoint matrices
  retain their published BF16 `[out,in]` layout and multiply without a transpose
  copy. SafeTensors payloads are read-only mmap ranges, so startup copies zero
  payload bytes and physical RSS is demand-paged; about 330 GiB is the logical
  mapped text weight size, within roughly 5 MiB of the all-BF16 floor. Small
  norm/conv weights and arithmetic remain F32. Shard files are an immutable
  runtime input for the lifetime of the mappings. The 180B checkpoint remains
  unstaged and unexecuted on this host because local storage is insufficient;
  CUDA, distributed execution, and MTP remain explicit errors or non-goals.
- The Qwen4 vision tower now reuses the generalized Qwen3-VL block/merger
  primitive with zero deepstack mergers. CPU reference implementations cover
  LayerNorm, GELU-tanh, bias, 2D RoPE, and non-causal SDPA. A frozen BF16 toy
  oracle matches the official Transformers pooler output to `9.34e-7` max
  absolute error; see the encoder-only `vision-oracle-v1.json`.
- The complete image-text prefill and one cached decode step now match the
  frozen official conditional-generation model to `2.99e-6` max absolute
  error with 7/7 top-1 agreement. The gate covers image-placeholder injection,
  official modality-group position IDs, main-attention and QSA-indexer mRoPE,
  and `rope_delta=-2` decode continuation; see `multimodal-oracle-v1.json`.
  MTP, CUDA, and distributed execution remain open.
- Video input is now a separate `VideoInput` capability. The encoder attends
  within each frame, while text mRoPE expands `T` into timestamp-separated
  `[1,H,W]` groups exactly like Transformers. A two-frame BF16 prefill plus
  cached decode matches to `3.19e-6` max absolute error with 12/12 top-1 and
  `rope_delta=-4`; see `video-oracle-v1.json`. MTP, CUDA, and distributed
  execution remain open.
- The mmap-backed BF16 decode GEMV now uses dependency-free accumulation,
  row-level parallelism above a fixed work threshold, and AArch64 NEON widening
  plus FMA. A 4096x4096 checkpoint-free dummy projection improved from
  `15.454 ms` to a repeat median of `0.365 ms` (42.35x, 85.63 GiB/s) without an
  F32 weight copy. The official top-10 MoE active shape additionally improves
  from a repeat median of `2.805 ms` to `1.058 ms` (2.65x) by parallelizing
  experts while keeping their inner GEMVs serial. Overlapping the independent
  shared expert then reduces the complete MoE slice from `1.425 ms` to
  `1.130 ms` (1.26x); both changes preserve bit-identical output.
  Router 512-to-10 selection replaces a full sort with linear partition plus a
  10-item sort, improving its repeated dummy median by 4.18x with identical
  expert order (about `0.216 ms/token` across 48 layers on this host).
  All four text/vision/multimodal/video oracles remain within their frozen
  thresholds; see `bf16-gemv-dummy-v1.json`.

## Reproduce the no-weight gates

The header fetcher refuses non-`206` responses and never requests tensor
payload ranges:

```bash
python3 scripts/fetch_qwen4_exp_headers.py > /tmp/qwen4-exp-headers.json
cargo run -p apxinf-model --example qwen4_exp_contract -- \
  /path/to/config.json /path/to/model.safetensors.index.json \
  /tmp/qwen4-exp-headers.json
```

Build the ignored local oracle environment from the frozen Transformers commit,
compile `qwen4_exp_checkpoint`, then run:

```bash
.apxinf/toolchains/qwen4-oracle/bin/python \
  scripts/qwen4_exp_toy_oracle.py

cargo test --release -p apxinf-model \
  benchmark_checkpoint_bf16_gemv -- --ignored --nocapture
```

## Frozen upstream evidence

- Model config SHA-256:
  `889658f2508e8c61d409b02e70e0d78d8d4452ec65aaafbe129805d213d2e74b`
- Model weight index SHA-256:
  `99e815241ef03325536b0aaa4441deea45174c17fae31e10f0bb456410c590de`
- Transformers source commit:
  `19876312341f42cf49467bb24d67271cf28cb599`
- `modular_qwen4_exp.py` SHA-256 at that commit:
  `54b9e147c1e1b95169419a4258c0ea4cefeaf7c37b9af2456e57ebc92a4ecc56`

Primary sources:

- <https://huggingface.co/Qwen/Qwen3.8-Flash-Next>
- <https://github.com/QwenLM/Qwen3.8-Flash-Next>
- <https://github.com/huggingface/transformers/tree/19876312341f42cf49467bb24d67271cf28cb599/src/transformers/models/qwen4_exp>
