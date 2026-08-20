# Qwen3.8-27B AWQ INT4 on RTX 4090

This record freezes the 2026-08-17 bring-up result for
`cyankiwi/Qwen3.8-27B-AWQ-INT4` on one RTX 4090 and states the boundary between
the current ApxInf source tree and the reference runtime used to obtain the
measurements.

## Outcome

- The pinned AWQ repository revision is
  `63768c10df38c0395e12ef49edac1bd539eaeeea`.
- All five SafeTensors shard SHA-256 values match the Hugging Face LFS manifest.
- The model runs on one 24,564 MiB RTX 4090 through the pinned vLLM reference
  environment.
- Deterministic multimodal tests pass 15/15 measured requests.
- Text performance cells from 1K through 32K pass 18/18 measured requests.
- Needle retrieval at 10%, 50%, and 90% of every 1K–32K context passes 18/18.
- With 128 requested output tokens, the largest prompt that completes all output
  tokens is 42,209 tokens. At 42,210 prompt tokens, the stream emits 127 tokens
  and then stalls on KV capacity without crashing the service.

The canonical detailed records are:

- [`../../benchmarks/qwen38_4090/BASELINE.md`](../../benchmarks/qwen38_4090/BASELINE.md)
- [`../../benchmarks/qwen38_4090/CONTEXT_LIMIT.md`](../../benchmarks/qwen38_4090/CONTEXT_LIMIT.md)
- [`deployment.md`](deployment.md)
- [`../../benchmarks/qwen38_4090/NATIVE_BRINGUP.md`](../../benchmarks/qwen38_4090/NATIVE_BRINGUP.md)
- [`../../benchmarks/qwen38_4090/W4A16.md`](../../benchmarks/qwen38_4090/W4A16.md)
- [`../../benchmarks/qwen38_4090/GDN_CORE.md`](../../benchmarks/qwen38_4090/GDN_CORE.md)
- [`../../benchmarks/qwen38_4090/GDN_LAYER.md`](../../benchmarks/qwen38_4090/GDN_LAYER.md)
- [`../../benchmarks/qwen38_4090/MLP_LAYER.md`](../../benchmarks/qwen38_4090/MLP_LAYER.md)

The generated datasets and raw result directories are intentionally not tracked;
the generator, frozen specs, result schema, and evidence hashes are tracked under
`benchmarks/qwen38_4090/`.

## Status terminology

`reference deployed` means the model is running through the frozen vLLM service
contract from this repository's deployment assets. It does **not** mean the
current Rust ApxInf engine natively implements Qwen3.8 or its AWQ format.

The first native loader/config gate is now validated against the real five-shard
checkpoint; see `NATIVE_BRINGUP.md` and `native_contract.json`.

`native ApxInf deployed` remains false until the architecture, packed-weight
loader, hybrid KV/state model, multimodal processor, and serving boundary listed
in `deployment.md` exist and pass the same workload manifests.
