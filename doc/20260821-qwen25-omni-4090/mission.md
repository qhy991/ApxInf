# Qwen2.5-Omni-3B on RTX 4090

## Frozen deployment slice

- Model: `Qwen/Qwen2.5-Omni-3B`
- Revision: `f75b40e3da2003cdd6e1829b1f420ca70797c34e`
- Target: one RTX 4090 (`sm_89`, 24 GiB)
- Input: text, one image, or one WAV audio clip
- Output: text only
- Deliberate exclusions: Talker/speech output and video
- Concurrency: one request
- Precision: checkpoint-native BF16; FP32 reductions and logits where required

The first accepted deployment must execute the Qwen Thinker natively through
ApxInf. Transformers, vLLM, subprocess inference, and model-family fallback are
reference tools only and cannot satisfy native Completion.

## Repository structure

Follow `doc/adding-a-new-model.md`, `doc/model-organization.md`, and
`doc/design.md`. The model implementation belongs in:

```text
crates/apxinf-model/src/qwen25_omni/
```

Use the shared `AutoModel` registry with exact
`config.json:model_type=qwen2_5_omni`. Extend the shared request boundary with
an optional borrowed `AudioInput`, parallel to `ImageInput`, and expose audio
support through `LlmCapabilities`. Architecture, modality injection, TMRoPE,
cache/state, and fusion orchestration remain in the model layer. Backend crates
may add only reusable device or single-kernel primitives.

## Minimum public interface

The existing commands remain stable. The shared generation command gains one
optional audio path:

```text
apxinf inspect --model <snapshot> --json
apxinf generate --model <snapshot> --prompt <text> [--image <file> | --audio <wav>]
                --device cuda --dtype bf16 --max-tokens <n>
apxinf serve --model <snapshot> --host <host> --port <port>
             --max-model-len <n> --enable-multimodal
```

`--image` and `--audio` are mutually exclusive in this first slice. Unsupported
video or speech-output requests fail explicitly. `inspect` must report the
model identity, Thinker architecture, shard inventory, dtype, enabled input
modalities, disabled Talker, and native readiness without loading weights onto
the GPU.

The HTTP service keeps the existing OpenAI-compatible chat-completions route.
It accepts ordinary text plus one `image_url` or one `input_audio` content item
and returns a complete text token trajectory with TTFT/TPOT fields. No request
may be routed to the Qwen3.8, Qwen3-VL, Transformers, or vLLM implementation.

## Acceptance order

1. Strict config, shard-index, tensor-ownership, and no-fallback inspection.
2. Unit tests for config parsing, modality validation, processor shapes, and
   TMRoPE position construction.
3. Hugging Face BF16 reference for text: per-layer checkpoints and exact first
   ten greedy token IDs.
4. Native CUDA text generation on the RTX 4090.
5. Deterministic real-image input and deterministic WAV input, both producing
   nonempty coherent text without a fallback process.
6. HTTP health plus text/image/audio requests.
7. Prompt-capacity staircase and post-failure recovery; record the largest
   successful prompt and first OOM.
8. Repeated TTFT, TPOT, prefill throughput, decode throughput, peak GPU memory,
   and GPU utilization under concurrency one.

The existing Qwen3.8 service is the rollback baseline. Build and model download
must use isolated paths. It may be stopped only when the frozen deployment
request grants service replacement, and must be restored if the Omni service
does not pass its health and modality smokes.
