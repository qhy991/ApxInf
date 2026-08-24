# `apxinf mlx-serve`: local persistent MLX application boundary

`apxinf mlx-serve` starts one already-local MLX model once, then accepts
strict JSONL requests on stdin and writes strict JSONL receipts on stdout. It
does not listen on a network socket, download files, or expose the Python
worker protocol directly. Every nested service ready/response receipt has
already passed the Rust identity and schema checks in `MlxService`.

## Start the boundary

All three paths must be absolute, existing, direct paths. Symlinks are
rejected.

```text
/absolute/path/apxinf mlx-serve \
  --model /absolute/path/Qwen3.5-0.8B-mlx \
  --mlx-python /absolute/path/mlx-env/bin/python3.14 \
  --mlx-runner /absolute/path/ApxInf/scripts/apxinf_mlx_serve.py \
  --timeout-seconds 120
```

The first stdout line is `apxinf-mlx-cli-ready-v1`. Its
`validated_service_ready` field freezes the model bundle, Python, runner,
generation helper, package versions, strategy, and session-cache policy
validated by Rust.

## Raw token-ID requests

Each request is exactly one newline-terminated JSON object. Unknown fields,
duplicate object keys, duplicate request IDs, partial lines, and oversized
lines stop the application boundary.

Ordinary generation does not use a session cache:

```json
{"format":"apxinf-mlx-cli-request-v1","request_id":"g1","operation":"generate","prompt_token_ids":[248045,846,198,9419,248046,198,248045,74455,198,248068,271,248069,271],"max_tokens":10,"stop_on_eos":true}
```

Create a cache-bound session with the same canonical prompt:

```json
{"format":"apxinf-mlx-cli-request-v1","request_id":"s1-create","operation":"session_generate","session_id":"chat-1","prompt_token_ids":[248045,846,198,9419,248046,198,248045,74455,198,248068,271,248069,271],"max_tokens":9,"stop_on_eos":true}
```

The frozen first response yields
`[9419,0,2500,628,353,1438,488,3242,30]`. Those nine IDs are part of the
committed cache prefix. An append must resend every committed token followed
by at least one new token. This complete 38-token example adds a second user
turn and requests ten more tokens:

```json
{"format":"apxinf-mlx-cli-request-v1","request_id":"s1-append","operation":"session_generate","session_id":"chat-1","prompt_token_ids":[248045,846,198,9419,248046,198,248045,74455,198,248068,271,248069,271,9419,0,2500,628,353,1438,488,3242,30,248046,198,248045,846,198,22791,13,248046,198,248045,74455,198,248068,271,248069,271],"max_tokens":10,"stop_on_eos":true}
```

Dropping, changing, or branching from an old token returns a recoverable
`invalid_request`; no fresh-generation fallback occurs. A model failure after
cache mutation returns a recoverable model error and invalidates that session,
so the same ID can subsequently be created from a new prompt.

Reset and clean shutdown are explicit:

```json
{"format":"apxinf-mlx-cli-request-v1","request_id":"s1-reset","operation":"session_reset","session_id":"chat-1"}
{"format":"apxinf-mlx-cli-request-v1","request_id":"stop","operation":"shutdown"}
```

Successful operations use `apxinf-mlx-cli-response-v1` and contain a
`validated_service_receipt`. Recoverable request/model failures use
`apxinf-mlx-cli-response-error-v1`. Recovery is an allowlist, not a default:
ordinary generation permits only `generation_failed` and `invalid_model`;
session generation additionally permits `session_cache_failed` and
`session_cache_limit`, after invalidating the affected session. Any unknown
worker code is fatal. The allowlists are enforced inside `MlxService` as well
as at the outer CLI, so direct Rust callers cannot continue after an unknown
worker error.

Explicit shutdown is the only path that sends the worker shutdown control. A
validated acknowledgement is followed by an unconditional sweep of the
original worker process group before its leader is reaped, so descendants
cannot keep inherited JSONL pipes open. It then emits
`apxinf-mlx-cli-shutdown-v1` and exits zero. EOF before shutdown or an outer
protocol violation instead aborts the process group immediately, independent
of `--timeout-seconds`; it never waits for a graceful worker acknowledgement.
The boundary emits one `apxinf-mlx-cli-fatal-error-v1` line on stderr and exits
nonzero.

Input lines are capped at 1 MiB and output lines at 4 MiB. The service remains
serial, local, and process-bound. It applies the pinned dependencies'
local-files/offline policy, but it is not an OS-level network sandbox. HTTP
routing, OpenAI-compatible schemas, SSE streaming, cancellation, and network
authentication are future application layers; none are implied by this
command.
