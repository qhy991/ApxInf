# Qwen2.5-Omni RTX 4090 service baseline

This directory owns the no-profiler performance and context-limit baseline for
the native `Qwen/Qwen2.5-Omni-3B` service. Run the benchmark on the GPU host
while the service itself is owned by `gpu-run`; the client does not create a
second CUDA context.

The same-GPU external comparison is owned by `VLLM_OMNI_BASELINE.md`. It pins
vLLM/vLLM-Omni 0.26.0, the thinker-only BF16 pipeline, exact token-count text
loads and the same real PNG/WAV assets. The external scripts also accept an
engine name, version endpoint and audio content schema so the SGLang
Transformers compatibility path can be evaluated without changing the frozen
workload. Keep those results separate from the native ApxInf acceptance
baseline. Current stable SGLang 0.5.18 fails the frozen checkpoint at
`AutoModel.from_config(Qwen2_5OmniConfig)` before weight loading, so it is a
capability result rather than a performance row; the reproduction probe lives
under `comparators/`.

The timing authority is client-observed wall time plus the service-emitted
TTFT/TPOT from `/v1/evaluations/generate`. `nvidia-smi` samples are explanatory
hardware evidence, not a replacement for endpoint timing. The prefill rate is
named a proxy because TTFT includes first-token work.

Build the accepted RTX 4090 artifact with an explicit native architecture:

```bash
CARGO_TARGET_DIR=target/qwen25-omni-sm89 \
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
compatibility. Reusing the same target directory makes subsequent links
faster.

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
  --audio "$APXINF_OMNI_AUDIO" \
  --reference "$APXINF_OMNI_REFERENCE" \
  --output benchmarks/qwen25_omni_4090/results/multimodal.json

python3 benchmarks/qwen25_omni_4090/benchmark_processor_recovery.py \
  --binary-path "$APXINF_BINARY" \
  --reference "$APXINF_OMNI_REFERENCE" \
  --output benchmarks/qwen25_omni_4090/results/processor-recovery.json

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

Raw trials and profiler exports are generated evidence, not repository API.
Keep them on the experiment host. This directory checks in only the current
aggregate promotion, acceptance, multimodal and external-engine comparison
records listed in `BASELINE.md`; `.gitignore` prevents new per-run JSON/CSV
files from accidentally expanding the PR.

The promoted text-only path uses 512-token causal chunks below 4,096 prompt
tokens, 256-token chunks from 4,096 through 8,191, and 1,024-token chunks from
8,192 upward. The 8K crossover is admitted only with causal FA2 enabled; it
reaches the complete
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
`benchmark_processor_recovery.py` additionally requires a malformed PNG to
return typed HTTP 422, confirms the persistent worker remains healthy, and
then requires a valid PNG to reproduce the frozen complete token sequence.

The accepted deployment keeps all optimized paths explicit through
`APXINF_OMNI_PERSISTENT_PROCESSOR=1`, `APXINF_BATCHED_GQA_PREFILL=1`,
`APXINF_FA2_GQA_PREFILL=1`,
`APXINF_QWEN25_FA2_CHUNK1024=1`,
`APXINF_QWEN25_FA2_ALL_CHUNKS=1`,
`APXINF_QWEN25_LONG_DECODE_SPLIT_CTA=1`,
`APXINF_QWEN25_LONG_DECODE_GRAPH=1`,
`APXINF_STREAM_ORDERED_ALLOC=1` and
`APXINF_TMROPE_POSITION_CACHE=1`,
`APXINF_TMROPE_POSITION_CACHE_PREFILL=1`, `APXINF_SOFTMAX_EXP_CACHE=1`,
`APXINF_SOFTMAX_GLOBAL_EXP_CACHE=1`,
`APXINF_SOFTMAX_EXP_CACHE_LONG_FALLBACK=1`,
`APXINF_QWEN25_CHUNKED_PREFILL=1` and `APXINF_QWEN25_DECODE_GRAPH=1`, plus
`APXINF_QWEN25_GPU_ARGMAX=1`, `APXINF_QWEN25_EAGER_GPU_ARGMAX=1` and
`APXINF_QWEN25_GPU_LAST_ROW=1`, plus
`APXINF_QWEN25_M1_PACKED_MLP=1` and
`APXINF_QWEN25_M1_GEMV_TACTICS=1`, plus
`APXINF_QWEN25_SHORT_DECODE_EXACT_RESIDUAL_NORM=1` and
`APXINF_QWEN25_SHORT_DECODE_W32_ATTENTION=1`, plus
`APXINF_QWEN25_SHORT_DECODE_FUSED_QKV_PRELUDE=1`. The decode
graph and exact two-stage GPU token selection are deliberately restricted to
SM89 one-token decode with `start_pos < 3072`; prefill and longer-KV decode
keep the accepted ordinary path except for the explicit long-decode selector.
That selector allocates one persistent 520 KiB workspace and uses grouped
four-query-head split-64 online-softmax attention only for SM89 one-token decode
at KV 32,761--32,767, QH/KVH/D=16/2/128 and max context 32,768. It requires the cached TMRoPE
position owner; unsupported shapes or an invalid selector composition fail
model load, while every shorter, prefill and multimodal call retains the
ordinary path. The long-decode graph selector additionally captures this exact
post-32K path and submits one graph replay per token at positions
32,760--32,767. It requires the ordinary decode graph, GPU argmax, packed QKV,
fused TMRoPE/KV and long split-CTA selectors; invalid compositions fail model
load instead of falling back. Decode beyond the tested exp-cache range
uses the explicitly selected exact scalar softmax; without that selector it
fails closed. The M1 packed-MLP selector requires the decode graph, concatenates
Gate/Up weights at model load, installs one exact RTX 4090 cuBLASLt tactic for
`[1,2048] @ [2048,22016]`, and captures one packed projection plus a bit-exact
SiLU/multiply node. Eager decode and prefill keep separate Gate/Up weights and
their prior tactics. The M1 GEMV selector installs exact RTX 4090 cuBLASLt
tactics for the one-token WO `[1,2048] @ [2048,2048]` and Down
`[1,11008] @ [11008,2048]` projections. Exact keys prevent those tactics from
changing prefill or unmatched shapes; unset or `0` retains vendor selection.
The short-decode exact residual selector replaces each BF16 residual add and
following RMSNorm with one graph node. It explicitly rounds the updated
residual to BF16 before the RMS reduction, so both the residual state and norm
output remain bit-exact. The selector is admitted only for the pinned SM89
Qwen2.5-Omni-3B composition and only changes the graph used below position
3,072; eager, prefill, 12K, and the dedicated post-32K graph retain their prior
nodes. Invalid dependencies or model shapes fail model load.
The short W32 attention selector captures the same online-softmax kernel with
32 split-K warps instead of the Thor-oriented 16-warp geometry. It is restricted
to the pinned SM89 short graph and requires the exact residual selector, so
eager, prefill, 12K, and post-32K execution keep W16 or their existing dedicated
attention owners. The changed merge order is not byte-exact for every synthetic
activation, so promotion requires complete text, image, and audio token
trajectories rather than an operator-only claim.
The fused QKV prelude selector replaces packed-QKV bias, Q TMRoPE, and K
TMRoPE/KV publication with one short-graph node. Projection+bias is explicitly
rounded to BF16 before TMRoPE, Q is written directly in the attention layout,
and K/V publish directly to their persistent cache slot. It requires W32 and
the pinned SM89 Qwen2.5-Omni-3B graph; prefill, eager, 12K, and post-32K paths
retain their existing owners.
The pack8 residual/RMSNorm selector keeps the same exact fused graph boundary
but specializes its aligned one-row H=2,048 implementation. Eight BF16 values
share each 128-bit global transaction while the square-sum is reconstructed in
the incumbent thread order, preserving both the rounded residual and normalized
output bit-for-bit. It requires the fused QKV prelude and the pinned SM89 short
graph; invalid epsilon, alignment, shape, or selector composition fails closed.
`APXINF_QWEN25_PACKED_QKV=1` selects one
packed QKV owner shared by both paths. `APXINF_QWEN25_FUSED_TMROPE_KV=1`
publishes rotated K and unchanged V directly to their caches during graph
decode. `APXINF_QWEN25_FUSED_SILU_MUL=1` replaces separate Gate SiLU and
multiply launches with a complete-write backend primitive while retaining the
old BF16 intermediate rounding exactly. `APXINF_VISION_GROUPED_SPARSE=1`
caches the processor-owned window group plan and lets the 28 grouped vision
blocks visit only ascending in-window key indices; unset or `0` retains the
full-scan reference, and malformed plans fail closed.
`APXINF_VISION_GROUPED_FA2=1` composes with that plan: it packs Q/K/V into
stable group order, runs the vendored variable-length BF16 HeadDim96 FA2
kernel, and restores the original token order. It requires nonempty groups,
an FA2 build and head dimension at most 96; otherwise it fails closed.
`APXINF_VISION_FULL_FA2=1` selects the bundled BF16 HeadDim96 FA2 kernel for
the four full-attention vision blocks at actual head dimension 80. It requires
an SM80-family `core-fa2` or `full` build and otherwise fails closed; it does
not alter text attention. The prefill position-cache selector uploads one
TMRoPE position array
per text or multimodal prefill slice instead of once per Q/K layer call. The
batched-GQA selector additionally packs BF16 prefill query rows by KV head
above 4,096 cached tokens, flattens sequence and GQA rows into large score and
value GEMMs, and restores the existing output layout without changing the KV
cache. For BF16 suffix prefill with accumulated KV length at least 4,097 and
the exact QH/KVH/D=16/2/128 shape, the causal-FA2 selector replaces that
materialized score/value path with the SM89 HeadDim128 specialization. It
consumes the same cache through explicit strides and returns the same flattened
layout. Decode, at-most-4K, multimodal and unmatched shapes retain their named
paths; unavailable or malformed admitted calls fail explicitly. GEMM, softmax,
pointwise, normalization, RoPE, embedding and QKV-split
outputs that are fully overwritten use uninitialized stream-ordered storage,
avoiding a redundant device memset; cache, prefix, partial-write and
accumulator contracts retain zero initialization. Short prefill and decode
retain their accepted paths. The
FA2-aware chunk selector raises only reset text prompts from 8,192 through
12,288 tokens from 256- to 1,024-token chunks. It requires both chunked
prefill and causal FA2 at model initialization; selector-off, shorter, longer,
decode and multimodal paths keep their accepted policy. The request-scoped
all-chunk selector additionally routes the first four chunks in exactly that
8K–12K cell through the validated causal FA2 capability below its default
long-KV policy threshold. The model passes this decision explicitly through
the layer boundary; no request state is stored globally. The
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
The persistent-processor selector starts one CPU-only Python child after model
load, waits for an explicit ready handshake, reuses the exact pinned
`AutoProcessor`, and exposes its liveness and mode through `/health`. The
serialized service has one request owner, so this optimization does not imply
a parallel processor protocol or scheduler.
The Broker-owned runit reference is checked in at
`service/apxinf-qwen25-omni-broker.run`; it is the environment and launch
authority for reproducing the promoted service. It requires
`agent-gpu-broker>=0.5.0` so a resident service can declare its advisory ETA as
`unknown` without fabricating a year-long completion time. The checked-in empty
`service/down` file is the desired-state authority: the service remains stopped
after supervisor or host restart and is started explicitly for a named workload.
Unset or `0` preserves the corresponding native path, while invalid values fail
closed.
