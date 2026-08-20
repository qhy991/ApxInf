# RTX 4090 context-capacity boundary — 2026-08-17

## Outcome

Under the full multimodal Qwen3.8 AWQ INT4 service contract, the largest prompt
that completes all 128 requested output tokens is **42,209 prompt tokens**. The
returned sequence is 42,337 tokens. Adding one prompt token causes the stream to
produce 127 output tokens and then stop making progress because no additional KV
slot can be scheduled.

No CUDA OOM was observed. vLLM's bounded KV allocator converts the memory limit
into a deterministic capacity stall, and the API/engine remain healthy after the
client disconnects.

## Frozen contract

- Model revision: `63768c10df38c0395e12ef49edac1bd539eaeeea`.
- One RTX 4090, driver 580.95.05.
- vLLM 0.27.1, eager execution, TP/PP/DP=1/1/1.
- Compressed-tensors W4A16 weights; FP8 E4M3 KV.
- `gpu_memory_utilization=0.97`, `max_num_seqs=1`.
- Server `max_model_len=43,298`.
- Fixed decode request: 128 tokens, temperature 0, `ignore_eos=true`.
- Thinking and preserved thinking disabled.

Engine initialization reported:

```text
Available KV memory:          1.52 GiB
GPU KV cache size:            43,298 tokens
Configured maximum sequence:  43,298 tokens
Maximum concurrency:          1.00x
Hybrid attention page size:   1,568 tokens
```

## Successful staircase

| Prompt tokens | Requested output | Returned sequence | TTFT (s) | TPOT (ms) | Decode (tok/s) | Effective prefill (tok/s) | Peak VRAM (MiB) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 36,000 | 128 | 36,128 | 15.450 | 82.64 | 12.10 | 2,330 | 22,755 |
| 40,000 | 128 | 40,128 | 17.455 | 84.54 | 11.83 | 2,292 | 22,755 |
| 41,000 | 128 | 41,128 | 18.022 | 82.25 | 12.16 | 2,275 | 22,755 |
| 42,000 | 128 | 42,128 | 18.476 | 84.79 | 11.79 | 2,273 | 22,755 |
| 42,208 | 128 | 42,336 | 18.556 | 84.54 | 11.83 | 2,275 | 22,755 |
| **42,209** | **128** | **42,337** | **18.531** | **82.25** | **12.16** | **2,278** | **22,755** |

All successful cases returned exactly 128 completion tokens. There is no abrupt
prefill or decode performance collapse near the boundary.

## Boundary-plus-one failure

For 42,210 prompt tokens and 128 requested output tokens:

```text
HTTP status:          200 (stream opened)
TTFT:                 18.548 s
Output tokens seen:   127
Output token 128:     never scheduled
Client end-to-end:    59.046 s
Client error:         ConnectionError: Read timed out
Server health after:  HTTP 200
Running/waiting after disconnect: 0 / 0
Idle VRAM after:      22,755 MiB
```

The KV state needed to emit 128 tokens is `prompt + output - 1` because the last
emitted token does not need to be cached for another decode step:

```text
42,209 + 128 - 1 = 42,336 KV slots  -> succeeds
42,210 + 128 - 1 = 42,337 KV slots  -> cannot schedule token 128
```

42,336 is exactly 27 pages of 1,568 tokens. The next KV slot would require a 28th
page under this hybrid cache layout. This explains why the operational full-output
limit is lower than the nominal 43,298-token pool/configuration limit.

A 43,000-token prompt was also tested. It opened an HTTP 200 stream but never
produced the first token within 60 seconds; server metrics showed
`waiting_by_reason="capacity"`, zero running requests, and zero active KV use.
The client timeout canceled the request cleanly.

## Evidence

Successful staircase:

```text
limit_results/20260817-200406/
raw.jsonl SHA-256:
a7feec680adb7c4c2ccf5d099e5b0493b8bc0f62d61bbd3bd6fafd0de9b64b68
```

Additional 42,209 success:

```text
limit_results/20260817-200657/
raw.jsonl SHA-256:
1a93eb0192d3fd4ac20945232d0f094662154c1e685608527162ac6c24cdcb5a
```

Self-contained final edge pair:

```text
final_edge_results/20260817-201339/  # 42,209 + 128, succeeds
raw SHA-256: 84f9479113671416b94d5c807ad4151f46d371b2c5404b2ea0b29817adf82d3c

final_edge_results/20260817-201442/  # 42,210 + 128, 127 tokens then timeout
raw SHA-256: 7689f7e4516ef4bfd8ab0c87aca8afaee7f4e2d57626ba47b4d3a0284ee87935
```

43,000 capacity-wait evidence:

```text
edge_results/20260817-201004/
raw SHA-256: 30f9920f26b3d0ab49861339e6e8f77457bfedd17738289d92840aecbdbc47db
```

The interrupted exploratory run is retained at
`limit_results/20260817-195712/` as negative evidence; it contains flushed raw
samples but no summary because the permanently waiting 43K request was manually
canceled.

## Interpretation

For this exact multimodal/eager/FP8-KV contract:

```text
Maximum prompt with a full 128-token completion: 42,209
Maximum returned sequence in that request:        42,337
First failing prompt:                              42,210
Failure mode: 127 tokens then KV-capacity stall
CUDA OOM:                                           not observed
Service crash:                                      not observed
```

Forcing an allocator bypass merely to obtain a CUDA OOM would replace a precise,
recoverable boundary with a crash and would not improve the capacity estimate.
To move the limit, the next experiments should explicitly reclaim memory from the
16,384-token vision encoder cache, run a text-only service without the vision
stack, quantize/offload currently uncompressed components, or reduce the hybrid
KV/page allocation cost. Each is a different deployment contract and must be
measured separately.
