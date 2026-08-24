# MLX exact-append session cache

This slice adds an explicit, versioned session protocol to the existing
single-model MLX process. It does not change ordinary generation: an
`apxinf-mlx-service-request-v1` request still calls `generate_step` without a
`prompt_cache` argument.

## Supported reuse

Only one reuse shape is accepted:

1. `create` evaluates a complete prompt and commits that prompt plus every
   token yielded by generation.
2. `append` supplies the complete new prompt, an exact count and SHA-256 for
   the previously committed prefix, and at least one new token.
3. The service checks every old token before evaluating only the suffix.

Changing, dropping, or branching from any committed token is rejected. A
missing or evicted session is also rejected; the service never falls back to
fresh generation under a session request.

This restriction is required for Qwen3.5. Its MLX cache contains 18
`ArraysCache` instances for Gated DeltaNet convolution/recurrent state and six
ordinary `KVCache` instances. The recurrent caches are not trimmable, so the
generic longest-common-prefix trimming path cannot implement safe branching.

## Cache-position contract

The pinned `mlx-lm==0.31.3` `generate_step` first evaluates the complete
prompt. Before yielding each generated token, it evaluates that token to
prepare the next logits. Therefore, after a token has been yielded and MLX is
synchronized, the cache contains exactly:

`complete prompt + every yielded generated token`

It does not contain the next predicted token. A fake cache records this
position directly, and the real two-turn smoke compares an appended cached
request with a fresh full-prompt request token for token.

## Identity, failure, and memory rules

Every request binds the session ID, exact prefix token hash, model config hash,
complete local-bundle manifest hash, greedy strategy, and cache policy. The
Rust caller independently validates those fields in the ready message and
every response.

Live session caches are serial and in-process, with deterministic LRU limits
of four sessions and 512 MiB according to the MLX cache objects' `nbytes`.
Eviction IDs and byte accounting are returned and checked. This is a bound on
live logical session-cache storage, not a hard bound on the MLX allocator's
reserved memory or process RSS.

Reset requires the exact current prefix identity. Generation mutates MLX cache
objects in place, and Qwen3.5 recurrent state cannot be rolled back atomically.
Consequently, any error after an append begins invalidates that whole session;
the partially mutated cache is never committed or reused. A killed worker
loses all sessions.

## Current scope

The feature is a local JSONL/Rust library boundary. It does not add HTTP, CLI
surface, persistence across process restarts, prefix branching, automatic
common-prefix lookup, or cross-model reuse. The real run is a correctness
smoke under a noisy desktop, not a throughput admission benchmark. Frozen
details are in
[`mlx-session-prefix-cache-evidence-20260824.json`](./mlx-session-prefix-cache-evidence-20260824.json).
