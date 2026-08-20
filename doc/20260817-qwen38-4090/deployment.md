# Qwen3.8-27B AWQ deployment boundary

## Repository and model authorities

```text
ApxInf repository: team-mz/rusin-dev
ApxInf revision:   ea3a4eb1057a1eff127b8187d4d844f12c29fff9
ApxInf branch:     master

Model repository:  cyankiwi/Qwen3.8-27B-AWQ-INT4
Model revision:    63768c10df38c0395e12ef49edac1bd539eaeeea
Model type:        qwen3_5
Architecture:      Qwen3_5ForConditionalGeneration
Quantization:      compressed-tensors W4A16, group size 32, asymmetric
```

## Original native support audit

The pinned ApxInf revision could not load or execute this model natively before
the uncommitted 2026-08-17 bring-up changes described below.

1. `src/main.rs` dispatches only `model_type == "qwen3_vl"` to the Qwen path.
   Every other model type falls through to `GeneralLlama`.
2. `crates/apxinf-model` contains Llama, Qwen3-VL, and PI0.5 implementations,
   but no `qwen3_5`/Qwen3.8 implementation.
3. The target text stack alternates Gated DeltaNet and full-attention layers.
   ApxInf has no Gated DeltaNet state/update implementation for this model.
4. `crates/apxinf-loader/src/safetensors.rs` accepts F32, F16, BF16, and
   F8_E4M3. The target checkpoint also contains 798 I32 and 399 I64 tensors used
   by packed INT4 weights and quantization metadata.
5. The target quantization config is compressed-tensors `pack-quantized`; the
   existing PI0.5 W8A8 path is a different format and cannot be reused by name.
6. The existing Qwen3-VL configuration, tensor names, vision encoder, mRoPE, and
   KV cache contracts do not match Qwen3.8.
7. The CLI is not an OpenAI-compatible streaming service. The existing
   websocket service is PI0.5/OpenPI-specific.

Silently routing Qwen3.8 through Llama or Qwen3-VL would load the wrong tensor
contract and is forbidden.

## Reference deployment

Until native support exists, the repository carries a reproducible reference
deployment under `benchmarks/qwen38_4090/`.

Remote authorities used by the launcher:

```text
Model directory:
/mnt/user_dir/hanjinchen/models/cyankiwi/Qwen3.8-27B-AWQ-INT4

Isolated runtime:
/root/venvs/qwen38-vllm-0.27.1
```

Launch from the ApxInf repository root:

```bash
cd benchmarks/qwen38_4090
./serve_vllm_reference.sh
```

Frozen service contract:

```text
vLLM:                   0.27.1
PyTorch:                2.13.0+cu130
Transformers:           5.15.0
compressed-tensors:     0.17.0
FlashInfer:             0.6.16.post3
TP / PP / DP:           1 / 1 / 1
weights:                compressed-tensors W4A16
KV cache:               FP8 E4M3
execution:              eager
chunked prefill:        enabled
max sequences:          1
gpu memory utilization: 0.97
```

The base launcher uses the accepted 33,792-token service limit. The follow-up
capacity service was launched with `MAX_MODEL_LEN=43298`; see
`CONTEXT_LIMIT.md` before changing the limit or output budget.

## Required native primitives

The minimum complete native implementation is:

1. A Qwen3.8-owned nested config parser and strict `model_type=qwen3_5` gate.
2. SafeTensors support for I32/I64 packed tensors without lossy reinterpretation.
3. A compressed-tensors W4A16 loader that validates group size, zero points,
   scales, packing order, ignored/unquantized modules, and tensor ownership.
4. SM89 W4A16 GEMM kernels or a proven CUTLASS/Marlin-equivalent backend.
5. The 64-layer hybrid text runtime: Gated DeltaNet layers, full attention
   layers, RMSNorm/residual order, mRoPE, MTP/output head, and sampling.
6. Explicit hybrid recurrent-state and paged-KV ownership with bounded memory,
   request reset, and long-context allocation gates.
7. The 27-layer vision encoder, merger, media token mapping, and multi-image
   processor contract.
8. A streaming service adapter exposing the same request semantics used by
   `run_benchmark.py`, or a narrow benchmark adapter that preserves all timing
   boundaries and complete token trajectories.

Each primitive must fail closed when its configuration does not match. Native
promotion requires the existing generated manifests to pass through the ApxInf
path without changing weights, requests, output budgets, or metric definitions.

## Deployment acceptance sequence

```text
repository/build smoke
  -> loader/config smoke
  -> 1K text exact-key smoke
  -> deterministic image smoke
  -> 1K–32K fixed-output staircase
  -> 18-position long-context retrieval
  -> 42K capacity probe under the declared memory contract
  -> no-profiler repeated performance
  -> Nsight Systems causal profile
  -> selected Nsight Compute kernels
```

The current working tree now satisfies the repository/build and strict
loader/config inspection layers for this model; the reference deployment still
satisfies the serving capability/performance layers.

## Remote deployment record

The local authoritative checkout was copied without overwriting the earlier
research directory:

```text
Local checkout:  /Users/haiyan-infiniai/rusin-dev
Remote checkout: /mnt/user_dir/hanjinchen/apxinf
Revision:        ea3a4eb1057a1eff127b8187d4d844f12c29fff9
Origin:          git@github.com:team-mz/rusin-dev.git
```

The remote repository now contains uncommitted tracked-source changes for the
strict Qwen3.5/Qwen3.8 loader/config slice, plus untracked benchmark,
documentation, and `qwen35` module files. Nothing has been committed or pushed.

### ApxInf build verification

Rust was installed with the minimal stable profile through rsproxy:

```text
rustc 1.97.1 (8bab26f4f 2026-07-14)
cargo 1.97.1 (c980f4866 2026-06-30)
CUDA 12.3
APXINF_CUDA_ARCH=sm_89
```

Because the repository lock file omits the `apxinf-py` workspace dependencies,
the build used a disposable system-disk source copy so Cargo could update only
that copy:

```text
Build source: /root/apxinf-build-src
Target dir:   /root/apxinf-target-sm89
```

Commands:

```bash
PATH=/root/.cargo/bin:$PATH \
CARGO_REGISTRIES_CRATES_IO_INDEX='sparse+https://rsproxy.cn/index/' \
CARGO_TARGET_DIR=/root/apxinf-target-sm89 \
cargo check --workspace

PATH=/root/.cargo/bin:$PATH \
CUDA_PATH=/usr/local/cuda \
APXINF_CUDA_ARCH=sm_89 \
CARGO_TARGET_DIR=/root/apxinf-target-sm89 \
cargo build --release --features cuda
```

Both commands passed. The release build took 9m11s and produced:

```text
/root/apxinf-target-sm89/release/apxinf
ELF x86-64 PIE, 28 MiB
Build ID: 3e34afdc32daac3621e31a985294703db6f76257
```

`ldd` resolved CUDA driver/runtime, cuBLAS, cuBLASLt, NVTX, libstdc++, and
libgcc without a missing library. `apxinf --help` completed successfully.

### Reference model service from the new checkout

The previous service was stopped only after the new checkout and manifests were
ready. The replacement service was launched from:

```text
/mnt/user_dir/hanjinchen/apxinf/benchmarks/qwen38_4090
```

Active tmux session:

```text
apxinf-qwen38-server
```

The service uses the already-validated `MAX_MODEL_LEN=43298` reference contract.
After startup, `/health` returned HTTP 200 and two deployment smokes passed:

```text
results/20260817-205828/  text-niah-1024-p50: 1/1 request, exact answer
results/20260817-205836/  mm-shape-count:     1/1 request, exact answer
```

These smokes prove that the copied repository's deployment assets can start and
exercise the model. They remain vLLM reference-runtime results, not native
Qwen3.8 execution through the Rust `apxinf` binary.

### Native negative smoke

Before the native loader/config slice, the built Rust CLI was pointed at the
same model directory with CPU execution and a one-token output budget. It loaded
the tokenizer and then followed the generic Llama/single-file path:

```text
Loading model from ".../Qwen3.8-27B-AWQ-INT4/model.safetensors"...
Failed to load model: No such file or directory
```

The target checkpoint is a five-shard indexed AWQ repository, so this is the
expected historical failure and confirmed the original audit.

### Native loader/config checkpoint

The current uncommitted working tree adds I32/I64 tensor storage, header-only
indexed SafeTensors inspection, a strict `qwen3_5` nested-config parser, exact
compressed-tensors W4 group-32 asymmetric bundle validation, and fail-closed CLI
dispatch. The new SM89 release build produced:

```text
Build ID: 70fadb0f85b20576fd8b46512cd88f535e1743c4
SHA-256:  4d83eef615bff6861002f516c68477dc8eb55239a0c9dc2c2f45b86ca84ece8d
```

The real checkpoint inspection completed in 0.437 seconds and validated all 5
shards, 2,396 tensors, 21,017,689,808 tensor bytes, and 399 quantized-linear
bundles. The exact receipt is
`../../benchmarks/qwen38_4090/native_contract.json`; the design and promotion
boundary are in `../../benchmarks/qwen38_4090/NATIVE_BRINGUP.md`.

`generate` now recognizes `qwen3_5` and fails before tokenizer/weight allocation
with an explicit “native execution is not implemented yet” message. This is the
required fail-closed state. Native execution remains false until W4A16, GDN,
full-attention, vision, and service paths pass the named endpoint and complete
trajectory gates.
