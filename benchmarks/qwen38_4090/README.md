# Qwen3.8 AWQ INT4 / RTX 4090 bring-up benchmark

This directory defines a reproducible, backend-neutral baseline for bringing up
`cyankiwi/Qwen3.8-27B-AWQ-INT4` on one RTX 4090. `spec.json` is the source of
truth only for the historical 2026-08-17 bring-up experiment; its datasets,
request manifests, raw samples, and summaries are derived from it. The summer-
camp leaderboard has a different, later authority:
`evaluation/contract-v1.json`. In particular, the course contract adds public
and hidden correctness, dynamic cohort scoring, context beyond 32K, and a
multi-request bonus. Historical `spec.json` non-goals must not be interpreted
as leaderboard exclusions.

The current `rusin-dev` source tree validates the native `qwen3_5` hybrid
config/checkpoint and executes the complete 64-layer text decoder with real
W4A16 weights, stateful GDN, full attention, MLP, final norm, and LM head. A
resident ApxInf process exposes OpenAI-compatible health, model-list, and
streaming/non-streaming chat-completions endpoints. Its decode path already
beats the frozen vLLM reference, while prompt prefill is still tiled through a
canonical M8 prompt tiles with an M1 tail. This cuts matched 1K/8K service TTFT
by 2.40x/2.33x while preserving the decode path, but prefill remains roughly
20x behind vLLM. The complete operator-to-service evidence and next
Marlin-class Tensor Core boundary are recorded in `PREFILL.md`.

## Frozen scope

- Hardware: one NVIDIA GeForce RTX 4090, compute capability 8.9, 24,564 MiB.
- Model revision: `63768c10df38c0395e12ef49edac1bd539eaeeea`.
- Quantization: compressed-tensors W4A16, group size 32, asymmetric zero point.
- KV cache: FP8 for the initial maximum-context search.
- Execution: eager mode for the first baseline. CUDA Graph needs about 0.40 GiB
  on this model and would reduce the measured KV budget below the 32K cell.
- Parallelism: TP=1, PP=1, DP=1, one request at a time.
- Text contexts: 1K, 2K, 4K, 8K, 16K, and 32K prompt tokens.
- Server sequence limit: 33,792 tokens, so the 32K prompt cell still has its
  declared 128-token decode budget and chat-template margin.
- Multimodal: deterministic counting, OCR, chart, spatial, and two-image cases.
- Non-goals: BF16 agreement, concurrency/goodput, and contexts above 32K.

Functional acceptance means HTTP success, non-empty output, no server crash or
NaN/OOM, and the exact deterministic answer where one is defined. It does not
mean BF16 numerical equivalence. Performance acceptance for this first run is a
repeatable baseline plus anomaly flags; there is no previous accepted apxinf
baseline from which a speedup or regression can honestly be claimed.

## Bring-up sequence

Download the pinned model through the mirror:

```bash
HF_ENDPOINT=https://hf-mirror.com \
huggingface-cli download cyankiwi/Qwen3.8-27B-AWQ-INT4 \
  --revision 63768c10df38c0395e12ef49edac1bd539eaeeea \
  --local-dir /mnt/user_dir/hanjinchen/models/cyankiwi/Qwen3.8-27B-AWQ-INT4
```

Create an isolated serving environment and start the reference server:

```bash
/root/venvs/qwen38-vllm-0.27.1/bin/python generate_dataset.py
./serve_vllm_reference.sh
```

Run one short text smoke and one image smoke before the full matrix:

```bash
/root/venvs/qwen38-vllm-0.27.1/bin/python run_benchmark.py \
  --case-id text-niah-1024-p50 --warmups 0 --repeats 1

/root/venvs/qwen38-vllm-0.27.1/bin/python run_benchmark.py \
  --case-id mm-shape-count --warmups 0 --repeats 1
```

Run the capability suites:

```bash
/root/venvs/qwen38-vllm-0.27.1/bin/python run_benchmark.py \
  --suite text-capability

/root/venvs/qwen38-vllm-0.27.1/bin/python run_benchmark.py \
  --suite multimodal
```

Run the controlled performance/context staircase:

```bash
/root/venvs/qwen38-vllm-0.27.1/bin/python run_benchmark.py \
  --suite text-performance
```

If the 32K-configured server cannot start, reduce `MAX_MODEL_LEN` in a fresh
server process and run only matching points. Never compare runs from different
server contracts as if only the context changed. Suggested diagnostic ladder:

```bash
MAX_MODEL_LEN=16384 ./serve_vllm_reference.sh
MAX_MODEL_LEN=8192  ./serve_vllm_reference.sh
```

The reported maximum context is the largest target for which every measured
request succeeds and the server remains healthy. A single lucky request is not
enough.

## Dataset design

`generate_dataset.py` creates:

- `data/text.jsonl`: three exact-retrieval needle positions for every context,
  plus one fixed-output performance case per context;
- `data/multimodal.jsonl`: cases with programmatic image ground truth;
- `data/images/*.png`: deterministic raster inputs;
- `data/manifest.json`: hashes of the spec and both JSONL manifests.

The long-context prompts are tokenizer-calibrated after applying the chat
template. Each row records both target and actual prompt-token counts. Inputs
carry SHA-256 hashes so results cannot silently mix different prompts.

## Metrics

The client records every raw request before computing summaries.

| Metric | Definition |
|---|---|
| TTFT | Client send to first non-empty streamed model output. Includes queue, request preprocessing, tokenization, prefill, and first decode. |
| E2E | Client send to stream completion. |
| TPOT | `(E2E - TTFT) / (completion_tokens - 1)`. |
| Decode tok/s | `(completion_tokens - 1) / (E2E - TTFT)`. |
| Effective prefill tok/s | `prompt_tokens / TTFT`; this is service-path throughput, not isolated kernel throughput. |
| Chunk ITL | Client-observed interval between non-empty SSE chunks. A chunk may contain more than one token, so server per-request ITL is preferred when available. |
| GPU utilization | Mean/max `nvidia-smi utilization.gpu` in the request window. |
| Memory-controller utilization | Mean/max `nvidia-smi utilization.memory`; a bandwidth-activity proxy, not GB/s. |
| Peak VRAM/headroom | Peak `memory.used` and total minus peak in the request window. |
| Power/energy | Mean/max power and `mean_power * request_duration`. |

Formal latency is always the no-profiler client timing. Use Nsight Systems to
explain one representative 8K prefill and one 8K/128-token decode trajectory.
Use Nsight Compute only for selected critical kernels and report
`dram__bytes.sum.per_second` or equivalent when claiming actual DRAM bandwidth;
do not convert `nvidia-smi utilization.memory` directly into GB/s.

For a later online-serving phase, add fixed concurrency and arrival-rate cells,
then report P50/P95/P99 TTFT, TPOT, ITL, E2E, failure rate, queue time, throughput,
and goodput. Those are deliberately not folded into this single-request
bring-up baseline.

## Artifacts

Each invocation writes a timestamped directory under `results/`:

```text
metadata.json  exact model/spec/dataset/run identity
raw.jsonl      warm-up and measured request records, outputs, errors, hardware
summary.json   per-case medians, means, CVs, pass counts, maximum context, flags
```

Keep the raw samples and their order. A median without the underlying requests,
model revision, dataset hash, and server command is not an acceptable baseline.

## Python 3.10 environment note

The isolated vLLM 0.27.1 environment resolved `flashinfer-python==0.6.16.post3`.
That package imports `array.array[int]`, which is evaluated immediately and is
not supported by the server's Python 3.10. The isolated environment therefore
adds `from __future__ import annotations` after the module docstring in
`flashinfer/comm/fd_exchange.py`. The original file remains beside it as
`fd_exchange.py.py310.orig`. `environment_patch.json` records the exact package
versions, file paths, original/patched SHA-256 values, and import validation.

This is an environment-only compatibility patch. It does not alter model
weights, vLLM scheduling, request semantics, or the benchmark code path.

The follow-up context-limit search and its exact one-token boundary are recorded
in `CONTEXT_LIMIT.md`. Its derived inputs and raw evidence live under
`limit_*`/`edge_*`/`final_edge_*` directories and remain excluded from Git.

The native bring-up contract, real-checkpoint inspection receipt, selected
SubCUDA evidence, and next W4A16/GDN gate are recorded in `NATIVE_BRINGUP.md`
and `native_contract.json`. The first real SM89 decode projection, its A/B
screen, SASS/resource audit, and operator-only promotion are in `W4A16.md`.
The 128-step recurrent GDN state/output gate and its complete-layer boundary are
in `GDN_CORE.md`.
The real layer-0 trajectory, BF16 profile, rejected out-proj candidates, and
1.287x W8A16/fused layer opt-in are in `GDN_LAYER.md`.
The real combined gate/up, SwiGLU, and down-projection MLP path is in
`MLP_LAYER.md`.
The real layer-3 Q/K/V/KV-cache/gated-attention/output path, 32K Systems
attribution, split-CTA portfolio, 256-token crossover, and 6.803x 32K layer
screen are in `ATTENTION_LAYER.md`.
The real layer-0..3 `GDN,GDN,GDN,full-attention` unit, Qwen offset RMSNorm
contract, mixed GDN out-projection dtype discovery, complete residual/MLP
composition, and 2.370x 32K joined-graph screen are in `HYBRID_UNIT.md`.
The complete 64-layer decoder, W8 LM head, 256/1K/8K/32K complete-token
curve, three exact 16-token mutated-state trajectories, memory admission, and
functional `apxinf generate` smoke are in `TEXT_DECODE.md`.
The resident OpenAI-compatible text server, stream/non-stream API smokes,
formal five-request 1K client run, 8K client smoke, hardware telemetry, and the
decode-win/prefill-regression decision are in `SERVICE.md`.
The first M<=8 weight-reuse prefill projection, exact BF16 gate, hot/cold
selector boundary, and 3.079x cold M8 operator screen are in `PREFILL.md`.
The larger-tile Tensor Core branch, official-vLLM Marlin screen, numerical
gate, bit-exact raw C ABI proof, vendoring provenance, and remaining runtime
repack boundary are in `MARLIN.md`.
