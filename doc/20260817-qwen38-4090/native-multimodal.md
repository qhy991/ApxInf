# Qwen3.8-27B INT4 native image inference on RTX 4090

Status: native single-image execution is available; public capability passes,
hidden promotion remains one case short.

## Frozen scope

- Model: `cyankiwi/Qwen3.8-27B-AWQ-INT4`, revision
  `63768c10df38c0395e12ef49edac1bd539eaeeea`.
- Hardware: one RTX 4090, SM89.
- Request: one PNG data URL plus one user text part, greedy decode,
  `enable_thinking=false`, non-streaming response.
- Runtime: pinned Hugging Face processor for media preprocessing; ApxInf owns
  the visual encoder, merger, embedding injection, hybrid language model, KV,
  mRoPE, LM head, and decode. There is no vLLM or Transformers model fallback.
- Non-goals for this slice: video, more than one image, image batching,
  streaming image responses, and visual performance optimization.

The semantic promotion gate was frozen before implementation: public 4/4,
hidden 8/8, all requests successful, no fallback, and healthy recovery. A
candidate that misses hidden 8/8 may remain available for development but may
not claim `multimodal-ready`.

## Why the former server could not execute images

The checkpoint already contained a real 27-layer visual tower, but the native
service stopped at the text boundary:

1. `qwen35_server.rs` rejected every image content part and advertised
   `multimodal=false`.
2. The reusable Qwen3-VL loader hard-coded hidden size 1024, patch width 1536,
   and three deepstack mergers. Qwen3.8 uses hidden size 1152, patch width
   1536, and zero deepstack mergers.
3. The CUDA vision SDPA primitive accepted head dimension 64 only; Qwen3.8 is
   `1152 / 16 = 72`.
4. `HybridUnit` used one scalar position for both KV ownership and rotary
   position. Qwen3.8 image prefill needs the true sequential position for KV
   writes and independent T/H/W positions for interleaved mRoPE.
5. There was no processor-to-runtime media contract, no image-token embedding
   replacement, and no post-image decode delta.

Changing only the health flag or injecting visual embeddings without the
position rewrite would therefore have produced a false-positive implementation.

## Implemented execution graph

```text
PNG data URL
  -> pinned Qwen3VLProcessor
  -> pixel_values BF16 [N,1536] + grid_thw + input_ids + modality mask
  -> native ApxInf visual patch embed / pos embed
  -> 27 visual blocks
  -> native FA2 non-causal vision attention, head_dim=72
  -> primary merger [N/4,5120]
  -> replace image-token rows in the language embedding stream
  -> compute exact T/H/W mRoPE and decode delta
  -> Qwen3.8 64-layer GDN/full-attention W4A16 prefill
  -> LM head and greedy decode
```

For a representative 448×448 image, the processor emits grid `[1,28,28]`,
784 patch rows, and 196 image tokens. With `enable_thinking=false`, the public
OCR request has 230 total prompt tokens. The full-attention CUDA prepare kernel
selects the rotary axis as `frequency_pair % 3`, matching the model's
interleaved `[11,11,10]` mRoPE sections. KV writes continue to use the true
token index. Decode uses `cache_position + mrope_delta` on all three rotary
axes.

The feature is behind `--enable-multimodal`; without the flag, the accepted
text path and its memory footprint remain available. With the flag, startup
validates `APXINF_PROCESSOR_PYTHON`, loads the native visual weights, and only
then advertises `multimodal=true`.

## Named endpoint evidence

Reference endpoint: output of the official Qwen3.5 visual primary merger,
captured in BF16 and compared as FP32.

For the hidden bar-chart case that exposes the remaining tail:

| Candidate | MAE | RMSE | Cosine | Max abs |
|---|---:|---:|---:|---:|
| original single-warp vision SDPA | 0.010219 | 0.035425 | 0.999192 | 17.0 |
| native vendored FA2 | 0.009004 | 0.030839 | 0.999389 | 16.0 |

Patch embedding is nearly exact (`MAE 2.47e-6`). Small BF16 GEMM differences
appear in block 0 and accumulate through 27 residual blocks. The primary merger
normalizes most of the tail, but the remaining difference crosses one final
token decision boundary in the hidden suite.

An isolation run injected the official visual primary into the same ApxInf
INT4 language runtime. The failed answer changed from `3` to the expected `2`.
This proves the processor, image-token placement, mRoPE, hybrid language stack,
and decode are correctly connected; the open discrepancy is visual numerical
accumulation.

## End-to-end result

| Implementation | Public | Hidden | Median public E2E | Median hidden E2E | Badge |
|---|---:|---:|---:|---:|---|
| vLLM 0.27.1 control | 4/4 | 8/8 | 0.304 s | 0.300 s | `multimodal-ready` |
| ApxInf native FA2 | 4/4 | 7/8 | 8.403 s | 8.256 s | `multimodal-public-pass` |

All 12 ApxInf requests completed successfully, without fallback, and the
service remained healthy. The only failed validator is
`hidden-mm-bar-arithmetic-02`: expected `2`, deterministic output `3`.

After the image suite, the text path still produced the previous 1K first token
and passed the public 8K multi-hop case with exact answer `MH-521240`. Resident
VRAM with visual weights is approximately 19,334 MiB, versus approximately
18,364 MiB for the text-only service.

## Rejected branches

- Reference-feature override: useful only to localize the fault; removed from
  final source and binary.
- Post-hoc feature scaling: values such as 0.8 or 1.1 flip the failed answer,
  but they change model semantics and were rejected.
- cuBLASLt heuristic rank changes: no numerical change.
- Pedantic accumulation: slightly changed endpoint metrics but did not change
  the failed final token.
- Explicit default Tensor-Op GEMM: identical endpoint to the normal path.
- Relaxing the hidden gate from 8/8 to 7/8: rejected.

## Current candidate and decision

```text
binary:
/root/apxinf-target-sm89-715a0ed/release/apxinf-course-native-mm-fa2-final-20260819

sha256:
4b1f1231c051e67522f66ab942d164ee784d2934e20004a44b5c89937889d007
```

Decision: **continue**, not full promotion. Keep the native candidate available
because it provides real image inference and passes the public contract, while
preserving the previous text-only binary for rollback. Do not label it
`multimodal-ready` until the visual numerical path reaches hidden 8/8 without
reference features, calibration hacks, or external inference fallback.

Primary next hypothesis: remove CPU materialization and align the producer and
consumer layouts inside the visual block while preserving BF16 operation order;
then re-run the named visual endpoints and the complete 12-case trajectory.
