# Pretokenized inference interface v1

Only this contract and its tests need to be public. A reference implementation,
checkpoint access, private kernels, and hidden evaluation data are not part of
the student handout.

## Health and identity

`GET /health` returns HTTP 200 only when the service can accept a request:

```json
{
  "status": "ok",
  "evaluation_contract": "apxinf.qwen38_27b.inference_interface.v1",
  "model_revision": "63768c10df38c0395e12ef49edac1bd539eaeeea",
  "max_model_len": 32768,
  "parallel_requests": 1,
  "capabilities": {
    "pretokenized_input_ids": true,
    "token_id_output": true,
    "multimodal": false
  }
}
```

The evaluator rejects a model revision or contract identifier mismatch. In v1,
`parallel_requests` must be exactly one.

## Generate

`POST /v1/evaluations/generate` accepts:

```json
{
  "input_ids": [151644, 872, 198, 9707, 151645, 198, 151644, 77091, 198],
  "max_new_tokens": 128,
  "temperature": 0.0,
  "ignore_eos": true,
  "stream": true
}
```

The requirements are intentionally small:

- `input_ids` is a non-empty array of unsigned 32-bit token IDs;
- `max_new_tokens` is positive and the full request must fit
  `max_model_len`;
- v1 supports greedy sampling only (`temperature=0`);
- performance cells set `ignore_eos=true` so every eligible run produces the
  exact output budget;
- exactly one request may execute at a time.

For a streaming request, the response content type is `text/event-stream`.
Each generated token is one event:

```text
data: {"type":"token","request_id":"eval-apxinf-1","index":0,"token_id":198}
```

The zero-based indexes must be contiguous. The final event is:

```text
data: {"type":"done","request_id":"eval-apxinf-1","usage":{"prompt_tokens":9,"completion_tokens":128,"total_tokens":137}}

data: [DONE]
```

The evaluator measures TTFT from client send to the first token event and TPOT
as `(last token arrival - first token arrival) / (completion_tokens - 1)`.
Server timing fields are diagnostic only and cannot replace client timing.

A non-streaming request returns `type=result`, `request_id`, `output_ids`, and
the same `usage` object. Invalid JSON, IDs, budgets, or sampling parameters
return HTTP 400 with a JSON `error`. Context exhaustion must be recoverable:
the next `/health` call and a small generation request must still succeed.

Text rendering, chat templates, and tokenizer selection are deliberately
outside this boundary. The teacher evaluator owns tokenization once and passes
the exact same IDs to every implementation. That removes the 40-token template
drift observed between the current ApxInf service and the vLLM reference.
