# Qwen2.5-Omni RTX 4090 service baseline

This directory owns the no-profiler performance and context-limit baseline for
the native `Qwen/Qwen2.5-Omni-3B` service. Run the benchmark on the GPU host
while the service itself is owned by `gpu-run`; the client does not create a
second CUDA context.

The same-GPU external comparison is owned by `VLLM_OMNI_BASELINE.md`. It pins
vLLM/vLLM-Omni 0.26.0, the thinker-only BF16 pipeline, exact token-count text
loads and the same real PNG/WAV assets. Keep those results separate from the
native ApxInf acceptance baseline.

The timing authority is client-observed wall time plus the service-emitted
TTFT/TPOT from `/v1/evaluations/generate`. `nvidia-smi` samples are explanatory
hardware evidence, not a replacement for endpoint timing. The prefill rate is
named a proxy because TTFT includes first-token work.

Build the accepted RTX 4090 artifact with an explicit native architecture:

```bash
CARGO_TARGET_DIR=/opt/apxinf/qwen25-omni-sm89-target \
  benchmarks/qwen25_omni_4090/build_sm89.sh
```

The script fixes `APXINF_CUDA_ARCH=sm_89` and defaults
`APXINF_CUDA_OPERATOR_SET=core-fa2`, builds the release CUDA service and prints its
SHA-256. A generic x86 CUDA build currently emits `sm_52` cubins plus PTX and
relies on driver JIT; it is not the accepted 4090 build contract. The measured
clean `core-fa2` build takes 229 seconds and produces a 21.6 MB binary. It adds
only the BF16 HeadDim96 non-causal FA2 instance needed by the head-dim-80 Omni
vision encoder. Setting the operator set to `core` retains the roughly
61-second, 15.5 MB text-only artifact but makes the full-vision selector fail
closed.
The build-system default remains `APXINF_CUDA_OPERATOR_SET=full` for backward
compatibility; that setting also compiles architecture-coupled FA2, INT8 and
Marlin objects and took 17 minutes 43 seconds with a 48.2 MB binary. Reusing
the same target directory makes subsequent links faster.

```bash
python3 benchmarks/qwen25_omni_4090/benchmark_service.py \
  --suite quick --warmups 1 --repeats 3

python3 benchmarks/qwen25_omni_4090/benchmark_service.py \
  --suite context \
  --lengths 1024,2048,4096,8192,12288,16384,24576,32760 \
  --context-output-tokens 8 --warmups 1 --repeats 3 --timeout 300

python3 benchmarks/qwen25_omni_4090/benchmark_contract.py \
  --output benchmarks/qwen25_omni_4090/results/contract.json

python3 benchmarks/qwen25_omni_4090/benchmark_multimodal.py \
  --image scripts/roofline_decode_throughput.png \
  --audio /var/lib/agent-gpu-broker/apxinf-omni-tone.wav \
  --output benchmarks/qwen25_omni_4090/results/multimodal.json

python3 benchmarks/qwen25_omni_4090/decode_roofline.py \
  --tpot-ms 8.255564196850394 --kv-len 128 \
  --peak-bandwidth-gbps 1008
```

`decode_roofline.py` reports algorithmic weight/KV byte lower bounds and an
effective BWU estimate. MFU is emitted only when the caller also supplies an
explicit dense-peak convention through `--peak-tflops`; neither estimate is a
replacement for profiler memory transactions or no-profiler endpoint timing.

Every request is greedy, non-streaming, `ignore_eos=true`, concurrency one,
and uses deterministic pre-tokenized IDs. Raw trials retain output-trajectory
hashes, client wall time, TTFT, TPOT, throughput proxies, peak memory, GPU and
memory-controller utilization, clocks, and power. Context probing stops at the
first failed case; service recovery is an operator action and must be recorded
separately rather than hidden by the benchmark.

The promoted text-only path uses 512-token causal chunks below 4,096 prompt
tokens, 256-token chunks from 4,096 through 12,288, and 1,024-token chunks
above that measured crossover. It reaches the complete
service contract at 32,760 prompt + 8 output tokens on the 24 GiB RTX 4090.
Image and audio
requests deliberately retain their processor-owned, unchunked path. Only the
final text chunk runs output normalization and the LM head; earlier chunks
publish KV state directly without synchronizing unused logits through CPU.
`benchmark_contract.py` verifies that over-context, over-completion,
non-greedy and streaming evaluation requests fail as typed HTTP 400 errors
without poisoning the service. `benchmark_multimodal.py` compares complete
image/audio output-token sequences against a frozen accepted report; its
single observations are correctness coverage, not timing admission samples.

The accepted deployment keeps all optimized paths explicit through
`APXINF_BATCHED_GQA_PREFILL=1`, `APXINF_STREAM_ORDERED_ALLOC=1` and
`APXINF_TMROPE_POSITION_CACHE=1`,
`APXINF_TMROPE_POSITION_CACHE_PREFILL=1`, `APXINF_SOFTMAX_EXP_CACHE=1`,
`APXINF_SOFTMAX_GLOBAL_EXP_CACHE=1`,
`APXINF_SOFTMAX_EXP_CACHE_LONG_FALLBACK=1`,
`APXINF_QWEN25_CHUNKED_PREFILL=1` and `APXINF_QWEN25_DECODE_GRAPH=1`, plus
`APXINF_QWEN25_GPU_ARGMAX=1`, `APXINF_QWEN25_EAGER_GPU_ARGMAX=1` and
`APXINF_QWEN25_GPU_LAST_ROW=1`. The decode
graph and exact two-stage GPU token selection are deliberately restricted to
SM89 one-token decode with `start_pos < 3072`; prefill and longer-KV decode
keep the accepted ordinary path. Decode beyond the tested exp-cache range
uses the explicitly selected exact scalar softmax; without that selector it
fails closed. `APXINF_QWEN25_PACKED_QKV=1` selects one
packed QKV owner shared by both paths. `APXINF_QWEN25_FUSED_TMROPE_KV=1`
publishes rotated K and unchanged V directly to their caches during graph
decode. `APXINF_QWEN25_FUSED_SILU_MUL=1` replaces separate Gate SiLU and
multiply launches with a complete-write backend primitive while retaining the
old BF16 intermediate rounding exactly. `APXINF_VISION_GROUPED_SPARSE=1`
caches the processor-owned window group plan and lets the 28 grouped vision
blocks visit only ascending in-window key indices; unset or `0` retains the
full-scan reference, and malformed plans fail closed.
`APXINF_VISION_FULL_FA2=1` selects the bundled BF16 HeadDim96 FA2 kernel for
the four full-attention vision blocks at actual head dimension 80. It requires
an SM80-family `core-fa2` or `full` build and otherwise fails closed; it does
not alter text attention. The prefill position-cache selector uploads one
TMRoPE position array
per text or multimodal prefill slice instead of once per Q/K layer call. The
batched-GQA selector additionally packs BF16 prefill query rows by KV head
above 4,096 cached tokens, flattens sequence and GQA rows into large score and
value GEMMs, and restores the existing output layout without changing the KV
cache. GEMM, softmax, pointwise, normalization, RoPE, embedding and QKV-split
outputs that are fully overwritten use uninitialized stream-ordered storage,
avoiding a redundant device memset; cache, prefix, partial-write and
accumulator contracts retain zero initialization. Short prefill and decode
retain their accepted paths. The
`APXINF_QWEN25_BF16_CHUNK_TACTICS=1` selector installs five exact RTX 4090
SM89 cuBLASLt records for the promoted 256- and 1,024-row packed-QKV and MLP
shapes before execution; unmatched shapes remain on vendor cuBLAS. The
`APXINF_SOFTMAX_INPLACE_SCALE=1` selector lets flattened BF16 prefill chunks
of at most 256 query tokens scale and normalize their single-consumer score
buffer in place. Larger chunks retain the accepted non-mutating softmax. The
shared-memory numerator cache parallelizes only the exact maximum; its FP32
sum remains in original column order. The long-decode global numerator cache
does the same beyond 11,264 columns. The
GPU last-row selector creates a view of the final hidden row before output
normalization, avoiding a whole-slice D2H and row H2D round trip. The
eager argmax selector applies the same exact two-stage GPU selection to
graph-ineligible decode positions, leaving only the prefill logits on CPU. The
global exp-cache selector preserves scalar max/sum order beyond the shared
numerator-cache limit by staging FP32 numerators in a bounded decode workspace.
The Broker-owned runit reference is checked in at
`service/apxinf-qwen25-omni-broker.run`; it is the environment and launch
authority for reproducing the promoted service. Unset or `0` preserves the
corresponding native path, while invalid values fail closed.
