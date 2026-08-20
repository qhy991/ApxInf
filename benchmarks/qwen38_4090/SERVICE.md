# Native ApxInf Qwen3.8 resident service on RTX 4090

This record closes the first online boundary for the native text runtime. The
CUDA release binary keeps the 64-layer decoder, states, KV, W8 LM head, and
tokenizer resident and exposes:

```text
GET  /health
GET  /v1/models
POST /v1/chat/completions  stream=false or stream=true
```

The implementation is deliberately single-request and text-only. It is enough
to run the frozen client timing protocol and separate the decode win from the
remaining prefill regression.

## Service contract

```text
GPU                    one RTX 4090 / SM89
model                  cyankiwi/Qwen3.8-27B-AWQ-INT4
model revision         63768c10df38c0395e12ef49edac1bd539eaeeea
parallelism            TP=PP=DP=1
worker                 one resident Rust process, one request at a time
API                    OpenAI-compatible chat completions and SSE
decoder                model-optimized attention selector, native GDN weights
KV                     BF16, resident capacity 8K or 32K by server launch
LM head                resident W8A16 conversion
prefill                canonical M8 tiles plus M1 tail, exact BF16 seams
decode                 stateful M=1 graph
sampling               greedy argmax
```

Request reset clears the 48 GDN recurrent and 48 conv states. It does not clear
the full KV pool: causal valid length cannot read stale positions, and prompt
processing overwrites each visible position before use. At 32K this removes
roughly 2 GiB of unnecessary request-start H2D writes.

## API smoke

Health and model-list endpoints return HTTP 200. A non-streaming `Hello`
request returns:

```json
{
  "choices": [
    {
      "message": {"role": "assistant", "content": "The user said \""},
      "finish_reason": "stop"
    }
  ],
  "usage": {"prompt_tokens": 53, "completion_tokens": 4, "total_tokens": 57}
}
```

The streaming request emits four valid `chat.completion.chunk` events with
`The`, ` user`, ` said`, ` "`, then a final usage chunk and `data: [DONE]`.

## Serial baseline client results

The existing `run_benchmark.py` client is used unchanged except for
`--base-url http://127.0.0.1:8001` and the native model ID.

### 1K formal run

Protocol: one warmup plus five measured requests, 1,064 actual prompt tokens,
128 completion tokens.

| Metric | ApxInf native service |
|---|---:|
| successes / functional passes | 5 / 5 |
| TTFT median | 21.3588 s |
| TTFT CV | 0.457% |
| TPOT median | 23.6194 ms |
| TPOT CV | 0.272% |
| decode throughput median | 42.338 token/s |
| serial effective prefill | 49.815 token/s |
| E2E median | 24.3568 s |
| mean GPU utilization | 97.79% |
| mean memory-controller proxy | 91.43% |
| peak VRAM | 16,531 MiB |
| mean power | 384.87 W |

All five measured TPOT values are between 23.556 and 23.732 ms. The model is
resident throughout; these are not load/JIT samples.

### 8K smoke

Protocol: zero warmup, one measured request, 8,232 actual prompt tokens, 128
completion tokens, 32K resident service.

| Metric | ApxInf native service |
|---|---:|
| success / functional pass | 1 / 1 |
| TTFT | 167.9669 s |
| TPOT | 28.3346 ms |
| decode throughput | 35.293 token/s |
| serial effective prefill | 49.010 token/s |
| E2E | 171.5654 s |
| mean GPU utilization | 99.35% |
| mean memory-controller proxy | 92.52% |
| peak VRAM | 18,067 MiB |

## Matched interpretation against vLLM

The frozen vLLM service baseline reports:

```text
1K TTFT     0.431 s
8K TTFT     3.184 s
TPOT        about 83.8..87.1 ms
decode      about 11.49..11.93 token/s
prefill     about 2.36K..2.57K token/s
```

ApxInf service decode is approximately 3.6x faster at 1K and 3.0x faster at
8K. This is a real API/client result, not a kernel projection. The decode
objective has crossed the vLLM baseline for the measured single-request cells.

The serial baseline TTFT is about 50x slower because it executes one complete
64-layer decode graph for every prompt token. Its prefill throughput gap is
also about 50x (`~49 token/s` versus `~2.5K token/s`). Therefore the overall
service is **not promoted as generally faster than vLLM**. Its decode path is
faster; its prefill path is the dominant regression.

## M8 service results

The canonical M8 prompt path is now enabled for every full eight-token tile.
The service retains the M1 path for the final 0..7-token tail and copies the
last M8 normalized row into the resident decode workspace when no tail exists.

### 1K formal run

One warmup plus five measured requests, 1,064 prompt and 128 completion tokens:

| Metric | M8 ApxInf service | Serial ApxInf | Change |
|---|---:|---:|---:|
| successes / functional | 5 / 5 | 5 / 5 | unchanged |
| TTFT median | 8.9065 s | 21.3588 s | 2.398x faster |
| effective prefill | 119.46 token/s | 49.82 token/s | 2.398x |
| TPOT median | 23.889 ms | 23.619 ms | decode preserved |
| decode throughput | 41.86 token/s | 42.34 token/s | within 1.2% |
| E2E median | 11.9293 s | 24.3568 s | 2.042x faster |
| mean GPU utilization | 96.47% | 97.79% | telemetry only |
| peak VRAM | 18,069 MiB | 16,531 MiB | different 32K resident contract |

One measured TTFT is 10.795 s; the other four are 8.781..8.928 s, producing a
9.30% CV. The median is stable enough for admission but requires more formal
pairs before any small next-step claim.

### 8K smoke

The 8,232-prompt/128-output cell succeeds functionally:

| Metric | M8 ApxInf service | Serial ApxInf | Change |
|---|---:|---:|---:|
| TTFT | 71.9498 s | 167.9669 s | 2.334x faster |
| effective prefill | 114.41 token/s | 49.01 token/s | 2.334x |
| TPOT | 30.132 ms | 28.335 ms | one-sample decode noise/regression |
| decode throughput | 33.19 token/s | 35.29 token/s | still 2.89x vLLM |
| E2E | 75.7766 s | 171.5654 s | 2.264x faster |

The M8 result materially improves TTFT but remains about 20.7x slower than
vLLM at 1K and 22.6x at 8K. The next material boundary is therefore the W4A16
GEMM itself, not HTTP, tokenizer, launch-count, or another M8-only fusion.

## Hardware interpretation

GPU and memory-controller activity are both above 90% during the long client
windows, so the serial prefill is not waiting on the host. It repeatedly
streams the model weights once per prompt token. M>1 projection/scan/attention
must reuse weights and parallelize tokens; HTTP or scheduler micro-optimization
cannot repair a 50x TTFT gap.

The completed prefill proof now covers:

```text
M>1 checkpoint W4 projection
  -> M-token MLP gate/up/down
  -> per-layer GDN recurrent scan over M tokens
  -> full-attention prefill with KV write
  -> all 64 layers + first token
```

The next branch is a larger-tile, Tensor Core W4A16 path with an explicit
Marlin-compatible repack/scale/zero-point contract. Scalar CUDA expansion to
M16/M32 is not expected to close a 20x TTFT gap and is not the primary branch.

## Evidence

### API smoke

| Artifact | SHA-256 |
|---|---|
| non-stream response | `c6534cd77b5a935de43be14504e14291e9c6f50b4a1ba1fe8fc2d0e5e3047f05` |
| SSE stream transcript | `1f59038f12671f068eff6ccaa0968d73cc65e6af7565b78e00d56f3e294d5b40` |

### 1K formal client bundle

Remote directory: `native_service_formal/20260818-122654/`.

| Artifact | SHA-256 |
|---|---|
| metadata | `3031befb906d6e685fb432993894c2eaf365f8704d4e8fc73ad1804e52981339` |
| raw samples | `e56171402053dd34cf5cc0a99830d1743f1b9ff331bdc99a75679744db1ae88f` |
| summary | `43b3f838bbc5ac9d3f8c167c5f0d646e4b781b3f341f4a265e40b7e9df78ce6b` |

### 8K client bundle

Remote directory: `native_service_8k/20260818-123335/`.

| Artifact | SHA-256 |
|---|---|
| metadata | `aaffc24671113b610710b046aa8e84a361f747b5ef440f08a60bbe6e6327a348` |
| raw sample | `7efbb560b27984ee6d9c5f3401432a7b1387c279a6101066cdae342549ea7d9c` |
| summary | `58b01d59dae0ff4b811866c2e124ad6434c60debeecc2ee36737dad0b63021ee` |

### M8 1K formal client bundle

Remote directory: `results/native_service_m8_1k/20260818-142503/`.

| Artifact | SHA-256 |
|---|---|
| metadata | `241496f831a87506b79a61b6d493ed1d7328ab9f4974dfa5695d9a5eef232fe9` |
| raw samples | `6f5e5f54fc1aa6b3a636ac9675e61861d03fd5b6bbd84b2b86185de587b6d2bf` |
| summary | `b20e630276741f051c6423c885e5ef40bfbadd2f26ba78b055a87cb6697d4422` |

### M8 8K client bundle

Remote directory: `results/native_service_m8_8k/20260818-142701/`.

| Artifact | SHA-256 |
|---|---|
| metadata | `35f3e7bdab5fed73afa9402b6769a1c0ee776c312d4bd780c3e9e8165405869c` |
| raw sample | `b27ad3f3c5c73af73ea4468fcf43c25971c0d05e4d24c07c522e1efe38483925` |
| summary | `d0ba374cd2e0673dde42f2304ef488fa4d2bf20c165f08c8f1dc13f4d6bca3f4` |

Latest measured M8 CUDA CLI/service binary:

```text
ELF build ID  7f63912a1aba132ff4fc12722281c12131152717
SHA-256      4cec4945b4f51cf57e0b6c3ffe5a6471e2c0a975b256f9bd49ab43e29f470f53
```

The formal M8 service bundles use this binary. Historical serial bundles retain
their earlier binary identities.

## Upstream unified-interface update (`715a0ed`)

The 2026-08-18 deployment was rebased from `ea3a4eb` onto upstream commit
`715a0ed790f2d10d82fab53fbeac3da3075adf26` (`feat(model): unify LLM/VLM
interface and loading (#11)`). The upstream change adds `LlmInput`, image input
metadata, model capabilities, and `AutoModel::load_model` auto-detection. The
generic LLM/VLM path keeps that new interface as the authority.

Qwen3.8 remains an explicit native text fast path. The generic generation loop
currently assumes prompt-wide logits, while the 32K native path deliberately
retains only the final logits row. Adapting it directly would materialize a
prohibitively large `[prompt_len, 248320]` tensor, so claiming unified-input
support would be incorrect. `inspect --json` therefore reports
`unified_llm_input=false`, `multimodal=false`, `m_gt_1_prefill=true`, and
`stateful_decode=true`. Unifying this path requires first changing the shared
prefill/logits contract, not an adapter or compatibility shim.

### Build and regression gates

```text
source HEAD       715a0ed790f2d10d82fab53fbeac3da3075adf26
CUDA target       sm_89
release build     9m19s
ELF build ID      8cecfeaf4de9df61311a9a249a52a0619fa2e580
binary SHA-256    ca88525122b6507663ad889d5920e1bbeb40668d158eb919bef2cf5f090cd5a5
model load        59.309s
resident VRAM     18,063..18,069 MiB
```

The remote regression gates passed: unified-input tests 6/6, Qwen3.8
checkpoint/config tests 4/4, tokenizer tests 2/2, model inspection, health,
model listing, non-streaming chat, SSE chat including final usage and `[DONE]`,
M8 prompt tiles, and the M1 prompt tail. The live Marlin M64 probe also passed
its numerical gate (`cosine=0.9999974434`, `relative-L2=0.0022612562`) and
measured 280.924 us including runtime transform versus 1484.884 us for eight
M8 calls (5.175x, 5/5 wins; kernel-only median 89.956 us).

### Matched service comparison

The 1K run uses one warmup plus five measured requests. The accepted 8K value
is the immediate matched rerun after the first single sample was observed as an
outlier. Both use the unchanged dataset/spec hashes, 128 output tokens, the
same model revision, and a 32K resident server.

| Cell | Metric | pre-update M8 | `715a0ed` | Change |
|---|---|---:|---:|---:|
| 1K, 5/5 | TTFT median | 8.9065 s | 9.0226 s | 1.30% slower |
| 1K, 5/5 | TPOT median | 23.889 ms | 23.600 ms | 1.21% faster |
| 1K, 5/5 | decode | 41.86 token/s | 42.37 token/s | 1.23% faster |
| 1K, 5/5 | E2E median | 11.9293 s | 12.0198 s | 0.76% slower |
| 8K, 1/1 rerun | TTFT | 71.9498 s | 71.7461 s | 0.28% faster |
| 8K, 1/1 rerun | TPOT | 30.132 ms | 28.151 ms | 6.58% faster |
| 8K, 1/1 rerun | decode | 33.19 token/s | 35.52 token/s | 7.03% faster |
| 8K, 1/1 rerun | E2E | 75.7766 s | 75.3213 s | 0.60% faster |

The first post-update 8K sample was slower (TTFT 76.444 s, TPOT 35.700 ms).
It did not reproduce: the matched rerun returned to or improved on the frozen
baseline while the GPU held 2.58..2.66 GHz at at most 70 C with no thermal
slowdown. It is retained as an outlier rather than deleted or silently averaged.

Evidence bundles:

| Cell | Remote directory | Metadata SHA-256 | Raw SHA-256 | Summary SHA-256 |
|---|---|---|---|---|
| 1K | `results/upstream_715a0ed_1k/20260818-161648/` | `9abbc147648042dfb7810e74446bf97c6c00fafd8de88fe6f3610387f77ebc4e` | `f2757af1e14a5c85254acb202d11f55d25ed7a84ae89fd866d8030f8553c50e7` | `52fccb5cc70b1baae44a8f247a73a8065d25a3e8d3aa4f699cdcab6a3e51d0e1` |
| 8K outlier | `results/upstream_715a0ed_8k/20260818-161820/` | `4a552649d2a6e2c86d596a84e425dbca6eca8a485e1cf485fa429adb21064ddc` | `974665b813028ec6186b7b430526d991d8b019729570929ccc854b1e0fe3af2c` | `003ebac8d1520837e215c0ea670491eb5a4dde9f9a9f16b2ac01252e430faf3e` |
| 8K rerun | `results/upstream_715a0ed_8k_rerun/20260818-162036/` | `6e6a8d4a12f36c1525e2738131b30c9a6f1f9053b82da639d30d734431b3c9b5` | `8204453cf4459104ed5a37ed638dfae3313c0b2b3bef7ec321eec851232d0400` | `944a48363db576836116dd2ab1326450dbf7b101216c85cbc187db35c632d4dc` |

The verified service runs in tmux session `apxinf-qwen38-715a0ed` on port 8001
with 32K capacity. The standard source path is
`/mnt/user_dir/hanjinchen/apxinf`; the complete pre-update tree is retained at
`/mnt/user_dir/hanjinchen/apxinf-ea3a4eb-m8-backup-20260818-1622` for rollback.
The vLLM reference remains stopped to avoid GPU-memory overlap, and its frozen
restart command remains documented in `deployment.md`.

## Decision

```text
decode decision       promote for 1K and 8K single-request API cells
prefill decision      M8 admitted; 2.33..2.40x service win, larger GEMM needed
overall service       continue, not generally promoted over vLLM
covered API           health, models, chat stream/non-stream, one request
unsupported           concurrency, cancellation, multimodal, M>8/Tensor Core,
                      CUDA Graph, online tail/goodput
next action           port and gate Marlin-class W4A16 for larger prompt tiles
```
