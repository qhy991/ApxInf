# Qwen2.5-Omni native implementation record

Node: `implement_pinned_omni_native_r3`

Mission: `apxinf-qwen25-omni-3b-v6`

Workspace: `/Users/haiyan-infiniai/ApxInf-omni-4090`

## Result and authority boundary

This node completed the local source and CPU-verification boundary for the pinned text/one-image/one-WAV-to-text slice. The implementation remains a native ApxInf Thinker path: model registration and checkpoint loading terminate in `GeneralQwen25Omni`, while the only Python subprocess is a local-files-only `AutoProcessor` that emits tensors and never constructs or executes a model. Unsupported sampling, video, speech-oriented fields, remote media, mixed image-plus-audio, malformed processor views, alternate devices/precision/quantization, and unknown generation fields fail explicitly. Evidence: `/Users/haiyan-infiniai/ApxInf-omni-4090/crates/apxinf-model/src/builtin.rs`, `/Users/haiyan-infiniai/ApxInf-omni-4090/crates/apxinf-model/src/qwen25_omni/general.rs`, `/Users/haiyan-infiniai/ApxInf-omni-4090/src/qwen25_omni_server.rs`, and `/Users/haiyan-infiniai/ApxInf-omni-4090/src/main.rs`.

This node did not create a CUDA context, load checkpoint payloads, run either frozen Host command, access the network or remote target, start or replace a service, invoke KerSor, commit or push Git, or claim deployment verification. Those effects remain outside `implement_native`. Evidence: `/Users/haiyan-infiniai/ApxInf-omni-4090/AGENTS.md`, `/Users/haiyan-infiniai/ApxInf-omni-4090/doc/20260821-qwen25-omni-4090/mission.md`, and `/Users/haiyan-infiniai/ApxInf-omni-4090-kersor/apxinf-qwen25-omni-3b-v6/deployment-request.json`.

## Consumed contract and custody

- The immutable request is `/Users/haiyan-infiniai/ApxInf-omni-4090-kersor/apxinf-qwen25-omni-3b-v6/deployment-request.json`, current SHA-256 `31b6c7240f4962514d29daa56af32232c052924e516f85def5c8ac74b5c28528`. It pins `Qwen/Qwen2.5-Omni-3B@f75b40e3da2003cdd6e1829b1f420ca70797c34e`, BF16, one HTTP request at a time, text/image/audio input, text-only output, no speech/video, 32,768 prompt tokens, and at most 128 output tokens. Evidence: that request path.
- The canonical guide was read completely before source work; `/Users/haiyan-infiniai/ApxInf-omni-4090/doc/adding-a-new-model.md` has SHA-256 `6b217d3764f5d81d2ef043c4ea2e5c8a9ef6b4d7bb42857d5e71dab621e7ea51`. Evidence: that guide path.
- `input_artifact:model_manifest`, `input_artifact:deployment_plan`, and `input_artifact:compatibility_report` were consumed. The Host-provided `model_identity_verified=true` was treated as an input condition; this capability produces no fact.
- The HF reference generator remains `/Users/haiyan-infiniai/ApxInf-omni-4090/scripts/hf_reference_dump.py`, SHA-256 `34f2088a6be6c0b416804a36b661966fb1a3d936464bbd33bd78e3632d2bd010`. It was not executed or edited by this node. Evidence: that script path and `/Users/haiyan-infiniai/ApxInf-omni-4090/tests/qwen25_omni_reference/README.md`.

## Native implementation surface

The candidate is the tracked native implementation at Git HEAD `1b3c3fdb7fa1e6688c977bcd187614e95d97c398` plus this node's reviewable uncommitted delta. Existing tracked surfaces provide exact registry routing, header-first required Thinker tensor validation and selective loading, BF16-preserving weight transforms, 36-layer text GQA/SwiGLU with KV state, vision and audio towers, modality fusion, TMRoPE and rope-delta decode, greedy generation, CLI inspection/generation/serve wiring, and a serialized HTTP boundary. Evidence: `/Users/haiyan-infiniai/ApxInf-omni-4090/crates/apxinf-model/src/builtin.rs`, `/Users/haiyan-infiniai/ApxInf-omni-4090/crates/apxinf-model/src/qwen25_omni/`, `/Users/haiyan-infiniai/ApxInf-omni-4090/crates/apxinf-model/src/llm_trait.rs`, `/Users/haiyan-infiniai/ApxInf-omni-4090/src/main.rs`, and `/Users/haiyan-infiniai/ApxInf-omni-4090/src/qwen25_omni_server.rs`.

This node changed exactly seven files, including this record:

1. `/Users/haiyan-infiniai/ApxInf-omni-4090/crates/apxinf-model/src/qwen25_omni/config.rs` now requires the nested Thinker/text/vision/audio model identities and the activation, cache, sliding-window, dropout, geometry, source-length, and timing semantics assumed by the hard-coded operators. Source SHA-256 before this record was written: `abd4c29e1b91fe1d87e9d2f0e0786db7c0319b8adc6ad63fcb01f1ee9c4ec8be`.
2. `/Users/haiyan-infiniai/ApxInf-omni-4090/crates/apxinf-model/src/qwen25_omni/vision.rs` exposes CPU-side processor-view validation, requires exactly one image grid, and validates grid/pixel shape before model backend work. SHA-256: `c0fdda967ed0109ae1a7777e0a9c785436d894860b6071d9a54a5a20cdd4e450`.
3. `/Users/haiyan-infiniai/ApxInf-omni-4090/crates/apxinf-model/src/qwen25_omni/audio.rs` exposes the selected two-row learned audio boundary embedding to model-owned fusion. SHA-256: `8df93ff47486521a760dd5cdbe5325d3d4b5213a307dfddeb8495a60489da1ce`.
4. `/Users/haiyan-infiniai/ApxInf-omni-4090/crates/apxinf-model/src/qwen25_omni/general.rs` validates all media shapes, placeholder counts, and exactly one contiguous audio marker span before embedding/backend execution; it scatters encoded audio into `<|AUDIO|>` positions and the learned boundary rows into the enclosing audio start/end positions. It adds direct image/audio TMRoPE position and boundary-marker regressions. SHA-256: `70bd200e3ac58dfe7423452b41b0eb10204de7e7d326d02edec3be26fa810a1d`.
5. `/Users/haiyan-infiniai/ApxInf-omni-4090/crates/apxinf-model/tests/data/qwen25_omni_config_minimal.json` now carries the exact nested semantic fields required by the strict parser. SHA-256: `75fb3d712007755186a03016b4227dcfe46fbfcc235324c4e66183730061805c`.
6. `/Users/haiyan-infiniai/ApxInf-omni-4090/src/qwen25_omni_server.rs` extends local validation regressions across wrong model, zero/oversized completion, non-neutral sampling/penalties/count, unknown fields, remote image URL, video, malformed audio format, unsupported role, and mixed image/audio. SHA-256: `271a8db088f5b5bcbc9114dcba757ead3b7a96557129271d4604e1e9ca9c04b9`.
7. `/Users/haiyan-infiniai/ApxInf-omni-4090/doc/20260821-qwen25-omni-4090/implementation-record.md` replaces the obsolete v3 record with this v6 node record. Evidence: this file.

## Final local verification

All final Cargo commands used `--offline`, ran in `/Users/haiyan-infiniai/ApxInf-omni-4090`, and were CPU-side. No command loaded weights or created a CUDA context.

1. `cargo test --offline -p apxinf-core omni_operator_tests` — exit 0; 3 passed, 0 failed. Evidence: `/Users/haiyan-infiniai/ApxInf-omni-4090/crates/apxinf-core/src/op_impls/cpu.rs`.
2. `cargo test --offline -p apxinf-model qwen25_omni --lib` — exit 0; 10 passed, 0 failed. It covers strict nested config drift, required Thinker ownership/shape/dtype, BF16 transpose, audio convolution length, one-image processor shapes, mixed-media/video rejection, audio boundary markers, and image/audio TMRoPE construction. Evidence: `/Users/haiyan-infiniai/ApxInf-omni-4090/crates/apxinf-model/src/qwen25_omni/`.
3. `cargo test --offline -p apxinf-model --test unified_llm_input` — exit 0; 10 passed, 0 failed. Evidence: `/Users/haiyan-infiniai/ApxInf-omni-4090/crates/apxinf-model/tests/unified_llm_input.rs`.
4. `cargo test --offline -p apxinf --bin apxinf --features cuda qwen25_omni_server::tests` — exit 0; 2 passed, 0 failed. The build explicitly reported `CUDA not found — building without GPU support`; this validates feature-gated Rust and request semantics only. Evidence: `/Users/haiyan-infiniai/ApxInf-omni-4090/src/qwen25_omni_server.rs` and `/Users/haiyan-infiniai/ApxInf-omni-4090/crates/apxinf-cuda/build.rs`.
5. `cargo check --offline -p apxinf --bin apxinf` — exit 0. Evidence: `/Users/haiyan-infiniai/ApxInf-omni-4090/src/main.rs`.
6. `cargo check --offline -p apxinf --tests --features cuda` — exit 0; it again reported CUDA unavailable, so this is not NVCC or device evidence. Evidence: `/Users/haiyan-infiniai/ApxInf-omni-4090/Cargo.toml` and `/Users/haiyan-infiniai/ApxInf-omni-4090/crates/apxinf-cuda/build.rs`.
7. `rustfmt --edition 2021` over the five edited Rust files — exit 0. `git diff --check` — exit 0. Evidence: `/Users/haiyan-infiniai/ApxInf-omni-4090`.

The builds emitted pre-existing warnings from `CpuKVCache.max_seq_len` and CUDA-stub unused items; no named gate failed. Evidence: `/Users/haiyan-infiniai/ApxInf-omni-4090/crates/apxinf-core/src/kv_cache.rs` and `/Users/haiyan-infiniai/ApxInf-omni-4090/crates/apxinf-cuda/`.

## Remaining limitations and frozen Host frontier

Local source tests do not establish numerical model correctness. The pinned snapshot and three required NPZ references are absent from this node's evidence; checkpoint headers/payloads were not loaded; the learned audio-boundary ordering and all intermediate operators still require the frozen HF/native equality gate; CUDA was neither compiled by NVCC nor executed; and exact first-ten-token text/image/audio trajectories remain unverified. Evidence: `/Users/haiyan-infiniai/ApxInf-omni-4090/tests/qwen25_omni_reference/README.md`, `/Users/haiyan-infiniai/ApxInf-omni-4090/doc/adding-a-new-model.md`, and `input_artifact:deployment_plan`.

No live HTTP response, recovery behavior, capacity/OOM boundary, GPU memory, TTFT/TPOT, throughput/utilization, service replacement, baseline restoration, immutable verifier result, or deployment fact was produced. Those checks remain exclusively with later frozen Host nodes; this implementation node does not declare the Mission complete. Evidence: `/Users/haiyan-infiniai/ApxInf-omni-4090/doc/20260821-qwen25-omni-4090/mission.md`, `/Users/haiyan-infiniai/ApxInf-omni-4090/AGENTS.md`, and `/Users/haiyan-infiniai/ApxInf-omni-4090-kersor/apxinf-qwen25-omni-3b-v6/deployment-request.json`.
