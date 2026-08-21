# Qwen2.5-Omni native implementation record

Node: `implement_native_path_r3`

Mission: `apxinf-qwen25-omni-3b`

Workspace: `/Users/haiyan-infiniai/ApxInf-omni-4090`

## Scope and result

This node implemented the local native ApxInf source path for the pinned Qwen2.5-Omni Thinker deployment slice. It did not execute a model forward, create a CUDA context, access a remote target or network, replace a service, run either frozen Host command, publish Git state, or claim deployment verification. The later Host gates remain responsible for pinned-checkpoint loading, Hugging Face reference comparison, CUDA execution, capacity, performance, HTTP smoke, service replacement/recovery, and binary acceptance. Evidence: `/Users/haiyan-infiniai/ApxInf-omni-4090/crates/apxinf-model/src/qwen25_omni/`, `/Users/haiyan-infiniai/ApxInf-omni-4090/AGENTS.md`.

## Consumed immutable contract

- Deployment request: `/Users/haiyan-infiniai/ApxInf-omni-4090-kersor/apxinf-qwen25-omni-3b-v2/deployment-request.json`, SHA-256 `2287ede3c5963711735be83c58c8e260e6081cc124d381df4fb230bc532f3531`. It pins `Qwen/Qwen2.5-Omni-3B`, revision `f75b40e3da2003cdd6e1829b1f420ca70797c34e`, BF16, HTTP, concurrency one, text/image/audio input, text-only output, no video, prompt cap 32,768, and output cap 128. Evidence: the request path above.
- Canonical guide: `/Users/haiyan-infiniai/ApxInf-omni-4090/doc/adding-a-new-model.md`, SHA-256 `6b217d3764f5d81d2ef043c4ea2e5c8a9ef6b4d7bb42857d5e71dab621e7ea51`. Evidence: that guide path.
- Host model manifest: consumed as `input_artifact:model_manifest`; schema `apxinf.hf_model_manifest.v1`, root config SHA-256 `20790f362c37a1718a3e764f597ae33dfc20177399762018dffaabf8321d4dd1`, index SHA-256 `5b7629198e2ef80e37612a491d9bfd71639d2f212632d36d8ab086922e74e129`, preprocessor SHA-256 `b47055ce61463ce143e9aab741d55c0aa520801a0a5d63be73c5b17cecb6bc69`, tokenizer-config SHA-256 `569aa7a9171e36dfff80f0a7550ab0c9c09e46ac840fea5030641417394fb0d2`.
- The consumed `input_artifact:deployment_plan` and `input_artifact:compatibility_report` define the bounded implementation and the still-pending Host gates. No successor node was executed.

## Implemented ownership surface

### Strict identity, config, checkpoint, and precision gates

- Added a model-owned nested config parser requiring the exact root architecture, Thinker text/vision/audio dimensions, BF16, tokens, TMRoPE sections, processor sampling rate, FFT, hop, and mel feature count. Missing or drifted identity-critical fields fail rather than default. Evidence: `/Users/haiyan-infiniai/ApxInf-omni-4090/crates/apxinf-model/src/qwen25_omni/config.rs`.
- Added a header-first tensor contract for every deployed `thinker.model.*`, `thinker.lm_head.*`, `thinker.visual.*`, and `thinker.audio_tower.*` tensor. Required shape/dtype mismatches fail before payload reads. The selected loader reads only those validated tensors and never materializes `talker.*` or `token2wav.*`. Evidence: `/Users/haiyan-infiniai/ApxInf-omni-4090/crates/apxinf-model/src/qwen25_omni/checkpoint.rs` and `/Users/haiyan-infiniai/ApxInf-omni-4090/crates/apxinf-loader/src/safetensors.rs`.
- Registered exactly `qwen2_5_omni`; Auto/BF16 are accepted and FP8/W8A8/F32, calibration, tuning, config overrides, and synthetic weights fail before checkpoint payload loading. Evidence: `/Users/haiyan-infiniai/ApxInf-omni-4090/crates/apxinf-model/src/builtin.rs`.

### Native Thinker execution

- Added physical BF16-preserving HF linear-weight transposition, explicit Q/K/V biases, a separate LM head, 36-layer GQA decoder execution, 32,768-position KV cache, cache-position checks, reset-on-error, final FP32 logits, and last-row-only LM-head projection for the next-token distribution. Evidence: `/Users/haiyan-infiniai/ApxInf-omni-4090/crates/apxinf-model/src/qwen25_omni/weights.rs` and `/Users/haiyan-infiniai/ApxInf-omni-4090/crates/apxinf-model/src/qwen25_omni/general.rs`.
- Added the 32-layer native vision path with patch projection, merge-layout 2D positions, window-grouped versus declared full attention, RMSNorm, biased Q/K/V/output projections, biased SwiGLU, merger, and exact placeholder replacement. Evidence: `/Users/haiyan-infiniai/ApxInf-omni-4090/crates/apxinf-model/src/qwen25_omni/vision.rs`.
- Added the 32-layer native audio path with processor-feature validation, stride-one/stride-two convolution lowering, positional embedding, LayerNorm-with-bias, non-causal attention, GELU MLP, post norm, average pooling, projection, and exact placeholder-count validation. Evidence: `/Users/haiyan-infiniai/ApxInf-omni-4090/crates/apxinf-model/src/qwen25_omni/audio.rs`.
- Added Qwen2.5-Omni contiguous-section TMRoPE separately from Qwen3-VL's interleaved mRoPE, mixed image/audio position construction, decode delta, media-once prefill, token-only decode, video rejection, combined context checks, maximum 128 output tokens, and reset between requests. Evidence: `/Users/haiyan-infiniai/ApxInf-omni-4090/crates/apxinf-model/src/qwen25_omni/general.rs`, `/Users/haiyan-infiniai/ApxInf-omni-4090/crates/apxinf-core/src/backend.rs`, and `/Users/haiyan-infiniai/ApxInf-omni-4090/crates/apxinf-cuda/kernels/custom/rope.cuh`.

### Shared modalities, reusable operators, CLI, and HTTP

- Extended the canonical borrowed request with `AudioInput` features, attention mask, feature lengths, and token counts; added audio capability; and generalized pre-prewarm rejection while retaining the single shared greedy prefill/decode loop. Evidence: `/Users/haiyan-infiniai/ApxInf-omni-4090/crates/apxinf-model/src/llm_trait.rs`.
- Added reusable CPU/CUDA primitives for grouped non-causal attention, one-dimensional im2col, average pooling, and contiguous-section TMRoPE. Added CPU reference oracles and CUDA operator tests, including head dimension 80. Evidence: `/Users/haiyan-infiniai/ApxInf-omni-4090/crates/apxinf-core/src/op_impls/cpu.rs`, `/Users/haiyan-infiniai/ApxInf-omni-4090/crates/apxinf-cuda/src/tests/operators.rs`.
- Extended `inspect` for strict Omni identity/tensor reporting and `generate` with mutually exclusive local image/WAV paths. The processor subprocesses use `local_files_only=True` and do preprocessing/tokenization only; model forward always uses native ApxInf. Evidence: `/Users/haiyan-infiniai/ApxInf-omni-4090/src/main.rs`.
- Added serialized Omni dispatch behind the existing `serve` command with health, model list, chat completions, deterministic token evaluation, streaming/non-streaming text responses, local data-URL media only, greedy-only request validation, 400/413/422/503 failure classes, `fallback_active=false`, talker disabled, and no video/speech output. Evidence: `/Users/haiyan-infiniai/ApxInf-omni-4090/src/qwen25_omni_server.rs`.

## Changed files and SHA-256

The implementation source surface, excluding this self-describing record, is:

```text
b9e7e2422f0d7b7ab9a86c19b32e3f25bb2c3e3f387294a0feb0743fb79c8552  crates/apxinf-core/src/backend.rs
52e23bfecfb753179f34fa5aec3da31e61448e3d9228aec9aa80683493f0adcb  crates/apxinf-core/src/op_impls/cpu.rs
d1d10cee762004b57b29053285a3e17897af7186446f31225afddb5049341861  crates/apxinf-cuda/adapters/core_kernels_adapter.cu
ced922239777c8dc02434b01b5b6512c226e7042b91c3ced83f98e54c9bb9aa3  crates/apxinf-cuda/adapters/custom_kernels.cu
9ae80022f43b0fcccb13e95408ca8634fcefaba4119855a4f5cd018499915d92  crates/apxinf-cuda/kernels/custom/attention.cuh
22d8bc892b04be665e03d3003695d167bae288e5bb9f179c9797a5fc7a8899de  crates/apxinf-cuda/kernels/custom/preprocess.cuh
693ba7ecc6f51f94ea845251ea557ba115b354e44943d24408bcf9f15121dd05  crates/apxinf-cuda/kernels/custom/rope.cuh
9892d59a0b9de2c04360a0f3b0b0388ae984bb22be0189c115604cd05e5805aa  crates/apxinf-cuda/src/backend.rs
26ce34f51d33f2b26b2e8b5ca127e0bbe2fcbc77d538f90b5cf37f10cc582655  crates/apxinf-cuda/src/ffi/custom.rs
157fd2a01de5cacc72d712763ea932f6f7349675f6b0c9b04e21d20b301ee06c  crates/apxinf-cuda/src/kernels/attention.rs
63800058cfd6e29e418aaf5d79623ad9eefa813eae56ea8091acf85565c8d04d  crates/apxinf-cuda/src/kernels/preprocess.rs
23e81a8ad5e7ec3a87cd7de7f483f9e7f4496e3b5484c86ac974e98595be19df  crates/apxinf-cuda/src/kernels/rope.rs
540cd7532e69aa82b68bbcf599ca430ed606e91aa26b5e2b99a90f41d2c8fd17  crates/apxinf-cuda/src/tests/operators.rs
cde63e8419c2b19a9477437aca2d56eb61bf6ebce87204f6fd3af9d87a40bc36  crates/apxinf-model/src/builtin.rs
0ab82bc5f8fcdd55d8037d72cb8deea501f09d31c4a210b0d0608f6d96345ea2  crates/apxinf-model/src/lib.rs
2d685bff22df6fef5a15131ef28ebf1b804c2b00a42fd7e0fb65655926a63b6f  crates/apxinf-model/src/llm_trait.rs
885c5f199bdc0e0b34f63b2719160141facfee561b6b136caa578ae06116feaa  crates/apxinf-model/src/qwen25_omni/audio.rs
dace284d102fe3e960d4b05c6d54022d8d20559e0e0db740fd9f2948c3978c61  crates/apxinf-model/src/qwen25_omni/checkpoint.rs
7ecbdd4054ba691c9a0190c951df535c6fad2bed60f438217a037601c0d87661  crates/apxinf-model/src/qwen25_omni/config.rs
3ee8914dbba4d0ceb44c03882733195f50e62dd9117f340b5aae3e01c9840ffa  crates/apxinf-model/src/qwen25_omni/general.rs
d7b0a7f7e87410ce73eaa8ed565bafe25a0e59f67595c00b42dae847fe8c7c9a  crates/apxinf-model/src/qwen25_omni/mod.rs
3bdd39a72b719c35d53ff18c55a1f85094ee71e9d77e4fffccd62e61a97016e8  crates/apxinf-model/src/qwen25_omni/vision.rs
7e72498fd399526b846ebbeec0c76bd59b2d2258c5193c0273396a969aa51514  crates/apxinf-model/src/qwen25_omni/weights.rs
9c116d3062414a48110bf4b6066aa86602c54bde9c85ab62d0535280a85459a8  crates/apxinf-model/tests/data/qwen25_omni_config_minimal.json
fc8e4771d5ab92c4ef75bbc39a30ee26593d1d7566b4d9b7bb27207a1ef148a8  crates/apxinf-model/tests/unified_llm_input.rs
b1114b4a7929b15cac2e7ba5f21b7edb0d66bf1ea518d998ba261585a4111ecb  src/main.rs
18e8fc0788efe1460c81f90628d2a2a1eefc7eb23173148ca11b62e5f1696707  src/qwen25_omni_server.rs
```

Evidence: the listed files under `/Users/haiyan-infiniai/ApxInf-omni-4090` and the final local `shasum -a 256` audit.

## Final local CPU-safe verification

All final commands ran from `/Users/haiyan-infiniai/ApxInf-omni-4090` without model loading or a CUDA context:

1. `cargo test -p apxinf-core omni_operator_tests` — exit 0; 3 passed, 0 failed. This covers grouped attention isolation, im2col/average-pool reference values, and contiguous-section TMRoPE axis ownership. Evidence: `/Users/haiyan-infiniai/ApxInf-omni-4090/crates/apxinf-core/src/op_impls/cpu.rs`.
2. `cargo test -p apxinf-model qwen25_omni --lib` — exit 0; 8 passed, 0 failed. This covers strict nested config, identity drift, BF16 transpose, complete required-tensor ownership, pre-payload shape/dtype rejection, audio downsampling length, vision merge grids, mixed positions, and video rejection. Evidence: `/Users/haiyan-infiniai/ApxInf-omni-4090/crates/apxinf-model/src/qwen25_omni/`.
3. `cargo test -p apxinf-model --test unified_llm_input` — exit 0; 10 passed, 0 failed. Existing text/image behavior stayed green; added audio pre-forward rejection, zero-token rejection, exact registry ownership, and BF16 load-policy rejection passed. Evidence: `/Users/haiyan-infiniai/ApxInf-omni-4090/crates/apxinf-model/tests/unified_llm_input.rs`.
4. `cargo check -p apxinf --bin apxinf` — exit 0. Evidence: `/Users/haiyan-infiniai/ApxInf-omni-4090/src/main.rs`.
5. `cargo check -p apxinf --tests --features cuda` — exit 0 for Rust CUDA-feature source and test compilation; build output explicitly reported `CUDA not found — building without GPU support`, so this is not NVCC or device evidence. Evidence: `/Users/haiyan-infiniai/ApxInf-omni-4090/crates/apxinf-cuda/build.rs` and the CUDA source paths listed above.
6. Scoped `rustfmt --edition 2021 --check` over `src/main.rs`, `src/qwen25_omni_server.rs`, and all new `qwen25_omni/*.rs` files — exit 0. Evidence: those paths.
7. `git diff --check` — exit 0. Evidence: `/Users/haiyan-infiniai/ApxInf-omni-4090` worktree diff.

Diagnostic attempts retained for truthfulness:

- The first `cargo test -p apxinf-model --lib --test unified_llm_input` exited 101 because two newly added tests used `unwrap_err()` on a success type containing non-`Debug` `GenerationProfile`; the tests were corrected to explicit matches and the final targeted suites above pass. Evidence: `/Users/haiyan-infiniai/ApxInf-omni-4090/crates/apxinf-model/tests/unified_llm_input.rs`.
- A subsequent broad invocation of that same command was interrupted with exit 130 after two unrelated pre-existing Pi0.5 synthetic-weight tests exceeded 60 seconds; all 48 completed tests had passed before the interruption. The bounded Omni and unified suites above are the final results. Evidence: `/Users/haiyan-infiniai/ApxInf-omni-4090/crates/apxinf-model/src/pi05/weights.rs` and the final test sources above.
- `cargo check --workspace --all-targets` exited 101 on pre-existing examples that import optional `apxinf_cuda` without declaring required features. The bounded default binary and CUDA-feature checks above both pass. Evidence: `/Users/haiyan-infiniai/ApxInf-omni-4090/crates/apxinf-model/examples/qwen35_gdn_prefill_probe.rs` and `/Users/haiyan-infiniai/ApxInf-omni-4090/crates/apxinf-model/Cargo.toml`.
- `cargo fmt --all -- --check` exited 1 because the pre-existing workspace is not globally rustfmt-clean and would require broad unrelated rewrites. Formatting was applied and checked only on the new model/service and touched CLI surfaces. Evidence: `/Users/haiyan-infiniai/ApxInf-omni-4090` worktree and the scoped formatting command above.

## Custody and negative-surface audit

- The immutable deployment request remains SHA-256 `2287ede3c5963711735be83c58c8e260e6081cc124d381df4fb230bc532f3531`. Evidence: `/Users/haiyan-infiniai/ApxInf-omni-4090-kersor/apxinf-qwen25-omni-3b-v2/deployment-request.json`.
- `scripts/hf_reference_dump.py` was not modified and is SHA-256 `48bbaea4173e8242c925194022932b9c3be6dc9cd40369f490dc0d8e7c840659`. Existing unified-input assertions were retained; new assertions were added rather than tolerances weakened. Evidence: `/Users/haiyan-infiniai/ApxInf-omni-4090/scripts/hf_reference_dump.py` and `/Users/haiyan-infiniai/ApxInf-omni-4090/crates/apxinf-model/tests/unified_llm_input.rs`.
- The external frozen verifier named by the request was neither read nor modified. No request, source, or service path contains an inference fallback; Transformers is invoked only as a local-files-only processor for tokenization/media tensors. Evidence: `/Users/haiyan-infiniai/ApxInf-omni-4090/src/main.rs`, `/Users/haiyan-infiniai/ApxInf-omni-4090/src/qwen25_omni_server.rs`, and the immutable request.
- No Git commit/push, network/model download, remote access, GPU broker job, CUDA context, service stop/start, or service replacement was performed. Evidence: this bounded implementation record and `/Users/haiyan-infiniai/ApxInf-omni-4090/AGENTS.md`.

## Remaining Host-gate limitations

These limitations are intentionally unresolved in this node and prevent any `deployment_verified` claim:

1. The pinned checkpoint and processor were not locally materialized or loaded, so the manifest contract has not yet been exercised against real shard headers/payloads. Required/excluded resident bytes are therefore not measured. Evidence: `input_artifact:model_manifest` is inspection evidence; no local model path exists in this node's results.
2. No Hugging Face BF16 activations, processor tensors, layer checkpoints, logits, first ten greedy tokens, complete 128-token-or-EOS trajectories, stop positions, decoded texts, or reset comparisons were generated for text/image/audio/mixed fixtures. Numerical and semantic equivalence remains unknown until the frozen Host reference gate. Evidence: `/Users/haiyan-infiniai/ApxInf-omni-4090/doc/adding-a-new-model.md` and `input_artifact:deployment_plan`.
3. The local host has no CUDA toolkit, so the new CUDA kernels were neither compiled by NVCC nor executed. The authored CUDA operator oracles in `crates/apxinf-cuda/src/tests/operators.rs` remain pending under the authorized GPU Host phase. Evidence: `/Users/haiyan-infiniai/ApxInf-omni-4090/crates/apxinf-cuda/build.rs` and `/Users/haiyan-infiniai/ApxInf-omni-4090/crates/apxinf-cuda/src/tests/operators.rs`.
4. Capacity, cold-load/steady/peak memory, full-context behavior, TTFT, TPOT, progress timeout, HTTP endpoint bodies, streaming behavior, negative pre-forward cache invariants on the live runtime, and concurrency-one service behavior were not measured. Evidence: `/Users/haiyan-infiniai/ApxInf-omni-4090/AGENTS.md` and `input_artifact:deployment_plan`.
5. Neither frozen Host command ran, no service was replaced, and the Qwen3.8 rollback service was not touched. Binary Host Completion and Mission Completion remain undecided. Evidence: `/Users/haiyan-infiniai/ApxInf-omni-4090-kersor/apxinf-qwen25-omni-3b-v2/deployment-request.json` and `/Users/haiyan-infiniai/ApxInf-omni-4090/AGENTS.md`.

This record is local implementation evidence only; it is not deployment verification, model-identity verification beyond the consumed Host fact, or a Mission-completion declaration.
