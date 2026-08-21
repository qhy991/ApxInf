# Qwen2.5-Omni native implementation record

Node: `r3_implement_native`

Mission: `apxinf-qwen25-omni-3b-v3`

Workspace: `/Users/haiyan-infiniai/ApxInf-omni-4090`

## Result and boundary

The bounded native implementation node is complete at the local source and CPU-verification boundary. The workspace contains a strict native ApxInf Thinker path for text, one image, or one WAV audio clip to text. This node closed the remaining first-slice semantic gap by rejecting simultaneous image-plus-audio input in the model before embedding/backend work and at the HTTP boundary before processor launch. It did not load the pinned checkpoint, execute a model forward, create a CUDA context, access the network or remote target, invoke KerSor or either frozen Host command, operate a service, publish Git state, or claim deployment verification. Evidence: `/Users/haiyan-infiniai/ApxInf-omni-4090/crates/apxinf-model/src/qwen25_omni/general.rs`, `/Users/haiyan-infiniai/ApxInf-omni-4090/src/qwen25_omni_server.rs`, and `/Users/haiyan-infiniai/ApxInf-omni-4090/AGENTS.md`.

## Consumed contract and custody

- The immutable request is `/Users/haiyan-infiniai/ApxInf-omni-4090-kersor/apxinf-qwen25-omni-3b-v3/deployment-request.json`, SHA-256 `6174b88215063dce0149132e3b2011136daa0ccda179367d55c718099d28b738`. It pins revision `f75b40e3da2003cdd6e1829b1f420ca70797c34e`, BF16, HTTP, concurrency one, text/image/audio input, text-only output, no video or speech output, a 32,768 prompt limit, and a 128 output limit. Evidence: the request path.
- The canonical guide was read before implementation work; `/Users/haiyan-infiniai/ApxInf-omni-4090/doc/adding-a-new-model.md` has SHA-256 `6b217d3764f5d81d2ef043c4ea2e5c8a9ef6b4d7bb42857d5e71dab621e7ea51`. Evidence: that guide path.
- `input_artifact:model_manifest`, `input_artifact:deployment_plan`, and `input_artifact:compatibility_report` were consumed. The Host-provided condition `model_identity_verified=true` was accepted as input; this node produced no fact and did not rerun identity inspection.
- The existing Hugging Face reference oracle `/Users/haiyan-infiniai/ApxInf-omni-4090/scripts/hf_reference_dump.py` remains SHA-256 `48bbaea4173e8242c925194022932b9c3be6dc9cd40369f490dc0d8e7c840659`. The external verifier was neither read nor modified. Evidence: the reference script and immutable request.

## Native implementation surface

The native source baseline is Git commit `bbc84b63745a1d213c61b63bc0366d11805b6a8a` (`feat: add native Qwen2.5 Omni thinker path`). It owns the following implementation surfaces:

- Model-owned strict nested config, required-tensor header validation, selective Thinker loading, BF16-preserving projection transforms, 36-layer GQA text execution, KV/cache position state, final FP32 logits, TMRoPE, image/audio placeholder fusion, and reset-on-error behavior: `/Users/haiyan-infiniai/ApxInf-omni-4090/crates/apxinf-model/src/qwen25_omni/`.
- Exact `qwen2_5_omni` registry dispatch and rejection of unsupported devices, precision/quantization, calibration, tuning, config overrides, and synthetic weights before checkpoint payload loading: `/Users/haiyan-infiniai/ApxInf-omni-4090/crates/apxinf-model/src/builtin.rs`.
- Canonical borrowed audio/image request inputs and one shared prefill-to-decode generation path: `/Users/haiyan-infiniai/ApxInf-omni-4090/crates/apxinf-model/src/llm_trait.rs`.
- Reusable grouped non-causal attention, im2col, average pooling, LayerNorm/GELU support, and contiguous-section TMRoPE across CPU reference and CUDA implementations: `/Users/haiyan-infiniai/ApxInf-omni-4090/crates/apxinf-core/src/backend.rs`, `/Users/haiyan-infiniai/ApxInf-omni-4090/crates/apxinf-core/src/op_impls/cpu.rs`, and `/Users/haiyan-infiniai/ApxInf-omni-4090/crates/apxinf-cuda/`.
- Strict inspect/generate/serve wiring, mutually exclusive CLI image/audio options, local-files-only processor use without model forward, greedy-only HTTP semantics, local base64 PNG/WAV media, text-only responses, exact identity/health reporting, concurrency-one serialization, and `fallback_active=false`: `/Users/haiyan-infiniai/ApxInf-omni-4090/src/main.rs` and `/Users/haiyan-infiniai/ApxInf-omni-4090/src/qwen25_omni_server.rs`.

The node-specific source delta is exactly two files:

1. `/Users/haiyan-infiniai/ApxInf-omni-4090/crates/apxinf-model/src/qwen25_omni/general.rs` rejects an input containing both image and audio before embedding, modality-tower execution, or KV mutation and replaces the former mixed-media test with a fail-closed regression test. Final SHA-256: `89000aaec1e52c2cdf9e4799cf133f929b24a3a7bfbec13dedc3ca76e9849bff`.
2. `/Users/haiyan-infiniai/ApxInf-omni-4090/src/qwen25_omni_server.rs` rejects the same unsupported combination after structural request validation and before local processor launch, with a focused HTTP-boundary unit test. Final SHA-256: `55930330b9873dd9e803b5bfe9f2d64f25ca42f6757fbdbbe6d29cab9498c73a`.

The binary source-only patch over the baseline commit for those two files has SHA-256 `42b89bd4a1c539432d9f304911e0c6a0ed762397ec781e182b23c73bb3721bb3`. The later Host gate must bind both the baseline commit and this patch digest; no new commit was created by this node. Evidence: `/Users/haiyan-infiniai/ApxInf-omni-4090` Git worktree and the two source files above.

## Final local verification

Every Cargo command set `CARGO_NET_OFFLINE=true` and ran from `/Users/haiyan-infiniai/ApxInf-omni-4090`. No test loaded model weights or created a CUDA context.

1. `cargo test -p apxinf-core omni_operator_tests` — exit 0; 3 passed, 0 failed. It covers grouped-attention isolation, im2col/average-pool reference values, and contiguous-section TMRoPE axis ownership. Evidence: `/Users/haiyan-infiniai/ApxInf-omni-4090/crates/apxinf-core/src/op_impls/cpu.rs`.
2. `cargo test -p apxinf-model qwen25_omni --lib` — exit 0; 8 passed, 0 failed. It covers strict nested config/identity drift, required Thinker tensor ownership, pre-payload shape/dtype failure, BF16 transpose, audio convolution length, vision merge grids, combined-media rejection, and video rejection. Evidence: `/Users/haiyan-infiniai/ApxInf-omni-4090/crates/apxinf-model/src/qwen25_omni/`.
3. `cargo test -p apxinf-model --test unified_llm_input` — exit 0; 10 passed, 0 failed. It covers shared generation, media-once prefill, pre-forward rejection, exact registry ownership, and BF16 load policy. Evidence: `/Users/haiyan-infiniai/ApxInf-omni-4090/crates/apxinf-model/tests/unified_llm_input.rs`.
4. `cargo test -p apxinf --bin apxinf --features cuda qwen25_omni_server::tests` — exit 0; 2 passed, 0 failed. It covers greedy/unknown-field rejection and combined image/audio rejection before preprocessing. Build output explicitly reported `CUDA not found — building without GPU support`; this was CPU-only validation of feature-gated Rust source. Evidence: `/Users/haiyan-infiniai/ApxInf-omni-4090/src/qwen25_omni_server.rs` and `/Users/haiyan-infiniai/ApxInf-omni-4090/crates/apxinf-cuda/build.rs`.
5. `cargo check -p apxinf --bin apxinf` — exit 0. Evidence: `/Users/haiyan-infiniai/ApxInf-omni-4090/src/main.rs`.
6. `cargo check -p apxinf --tests --features cuda` — exit 0; build output again reported CUDA unavailable, so this is Rust compile evidence only, not NVCC or device evidence. Evidence: `/Users/haiyan-infiniai/ApxInf-omni-4090/Cargo.toml` and `/Users/haiyan-infiniai/ApxInf-omni-4090/crates/apxinf-cuda/build.rs`.
7. Scoped `rustfmt --edition 2021 --check` over the two node-specific Rust files — exit 0. `git diff --check` — exit 0. Evidence: `/Users/haiyan-infiniai/ApxInf-omni-4090` and the two node-specific source files.

The builds emitted pre-existing warnings, including `CpuKVCache.max_seq_len` dead code and CUDA-stub unused items; they did not fail a named gate. Evidence: `/Users/haiyan-infiniai/ApxInf-omni-4090/crates/apxinf-core/src/kv_cache.rs` and `/Users/haiyan-infiniai/ApxInf-omni-4090/crates/apxinf-cuda/`.

## Remaining limitations and Host frontier

This node cannot establish deployment correctness. The pinned snapshot was not materialized or loaded; checkpoint headers/payloads were not exercised locally; no Hugging Face BF16 reference activations or first-ten-token trajectories were generated; CUDA kernels were not compiled by NVCC or executed; and text/image/audio token equality, full trajectories, state recovery on a live runtime, capacity/OOM behavior, HTTP observations, streaming, concurrency, memory, TTFT/TPOT, throughput, utilization, service replacement/recovery, and the immutable acceptance command remain unverified. Evidence: `/Users/haiyan-infiniai/ApxInf-omni-4090/doc/adding-a-new-model.md`, `/Users/haiyan-infiniai/ApxInf-omni-4090/doc/20260821-qwen25-omni-4090/mission.md`, `input_artifact:deployment_plan`, and the immutable request.

The next authorized Host node must consume this record and the two-part implementation revision, then run only its frozen command. This record is implementation evidence only and does not declare the Mission complete.
