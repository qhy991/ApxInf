# llama.cpp raw-token diagnostic runner

This is a deliberately narrow llama.cpp C-API runner for comparing the same
Qwen3.5 prompt schedule with ApxInf. It bypasses tokenization and chat-template
logic, submits these exact 13 token IDs, uses greedy argmax, ignores EOG, and
samples exactly 128 output tokens:

```text
248045,846,198,9419,248046,198,248045,74455,198,248068,271,248069,271
```

The runner always requests `n_ctx=142`, `n_batch=n_ubatch=13`, and
`n_seq_max=1`. Some llama.cpp/model combinations round the effective context
upward (the pinned Qwen3.5 build currently reports 256); the JSON preserves both
the requested and effective values and rejects an effective context smaller
than 142. Batch and sequence values must remain exact. The first
`token_ready_elapsed_ns` value includes prompt decode and sampling. Later values
are cumulative from the same origin. The 128th sampled token is intentionally
not decoded, so llama.cpp's `n_eval` counter is normally 127 while
`output.token_ids` contains 128 IDs.

The model is opened once with `O_RDONLY|O_NOFOLLOW|O_CLOEXEC`, must be a
single-link regular file, and is passed to llama.cpp through
`llama_model_load_from_file_ptr`. The runner deliberately disables mmap so the
load cannot fall back to reopening a path. It records and rechecks the same
descriptor's device, inode, size, link count, and ctime after loading and again
before publishing JSON. Vocabulary size `248320`, strictly increasing token
times, and performance counts `n_p_eval=13`, `n_eval=127`, `n_reused=126`,
`n_sample=128` are fail-closed contract checks. The CPU lane uses F32 K/V cache
and disables KQV and operation offload. The Metal lane uses F16 K/V cache.

The executable never calls llama.cpp's dynamic backend discovery functions. It
uses only the backends linked into the static executable, rejects
`GGML_BACKEND_PATH`, and does not accept a backend-directory option. A GPU run
must name one registered GPU explicitly. Success additionally requires all 24
transformer layers and the output layer to be assigned to that device, with
nonzero GPU model, context, and compute allocations. The receipt separately
discloses the input-embedding CPU fallback and the full memory breakdown. After
all token timing and performance counters are captured, one excluded proof
decode uses llama.cpp's scheduler callback to require completion of the input
embedding on CPU and all 24 layer endpoints plus the output head on Metal. The
CPU lane inversely requires all 26 sentinels on CPU.

## Build against a pinned checkout

Use a clean build directory outside the source tree. The llama.cpp checkout is
embedded with `add_subdirectory`, so no install step or edit to llama.cpp is
needed. The CMake project freezes the audited Release, static Metal/Accelerate
configuration, disables every other accelerator backend, and rejects other
build types. The placement proof intentionally uses llama.cpp experimental
internal APIs and is therefore coupled to the exact pinned source commit; a
source upgrade must rebuild and re-audit the runner.

```sh
cmake -S benchmarks/llama_cpp -B /tmp/apxinf-llama-runner-build \
  -DLLAMA_CPP_SOURCE_DIR=/absolute/path/to/pinned/llama.cpp \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=OFF \
  -DGGML_BACKEND_DL=OFF \
  -DGGML_METAL=ON \
  -DGGML_METAL_EMBED_LIBRARY=ON \
  -DGGML_ACCELERATE=ON
cmake --build /tmp/apxinf-llama-runner-build \
  --target apxinf-llama-cpp-raw-token-runner -j
```

For a checkout without Git metadata, pass its audited identifier explicitly:

```sh
-DLLAMA_CPP_SOURCE_ID=<source-tree-sha256-or-release-id>
```

If omitted, CMake embeds the checkout's Git `HEAD` only when the whole checkout
(including untracked files) is clean. A dirty checkout is marked with a
`-dirty` suffix, and a tree with no resolvable commit reports `unavailable`.
The JSON also records how the source ID was obtained. Hash the final runner,
llama/ggml libraries, and GGUF separately when creating a benchmark receipt.

## Run

The only successful stdout record is one JSON object. llama.cpp diagnostic logs
remain on stderr, so keep the streams separate.

```sh
/tmp/apxinf-llama-runner-build/bin/apxinf-llama-cpp-raw-token-runner \
  --model /absolute/path/to/model.gguf \
  --gpu-layers -1 \
  --gpu-device MTL0 \
  --threads 4 \
  > run.json 2> run.stderr
```

For the CPU-only F32 lane, use `--gpu-layers 0` and omit `--gpu-device`. The JSON
captures the raw input/output IDs, every cumulative token-ready time, llama.cpp
performance counters, requested and effective context parameters, registered
and selected devices, model/context/compute placement, model metadata, system
information, and the embedded source/build configuration.

This runner is diagnostic infrastructure, not by itself a formal benchmark
harness. Host quiescence, warmups, ABBA/BAAB ordering, process/RSS/swap custody,
artifact hashes, and cross-engine quality checks belong in the outer harness.
