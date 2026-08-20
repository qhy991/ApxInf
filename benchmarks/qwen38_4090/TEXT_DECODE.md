# Native Qwen3.8 64-layer text decode on SM89

This record advances ApxInf from layer/module evidence to a real text token
path:

```text
checkpoint embedding row
  -> 64 hybrid decoder layers
  -> final Qwen offset RMSNorm
  -> W8A16 LM head
  -> BF16 logits
  -> CPU argmax
  -> emitted token
```

It also records stateful 16-token trajectories and the first successful
`apxinf generate` prompt. It is offline/native text evidence; an
OpenAI-compatible service and M>1 prefill are still missing.

## Frozen runtime contract

```text
model                 cyankiwi/Qwen3.8-27B-AWQ-INT4
model revision        63768c10df38c0395e12ef49edac1bd539eaeeea
repository base       ea3a4eb1057a1eff127b8187d4d844f12c29fff9 + dirty source
GPU                    one RTX 4090 / SM89
decoder                64 layers = 48 GDN + 16 full attention
parallelism            TP=PP=DP=1
request                batch=1, one token per step, eager, one CUDA stream
checkpoint linears     compressed-tensors W4A16 group-32 asymmetric
KV                     BF16, 16 independent full-attention cache owners
GDN state              48 FP32 recurrent + 48 BF16 conv state owners
workspace              one shared activation workspace across all 64 layers
embedding              one selectively loaded BF16 checkpoint row
LM head                BF16 checkpoint streamed row-wise into W8A16 once
sampling               greedy argmax
```

The embedding table and vision tower are not resident on GPU. Prompt embedding
rows are read selectively; this keeps the text runtime within 4090 memory but
is not the future high-throughput embedding implementation.

## Model-level selector decision

The four-layer experiment admitted a lossy W8A16 replacement for the only BF16
GDN out-projection, layer 0. At 64 layers that difference amplified to final
residual relative L2 `0.166`, while saving only about 31 microseconds out of a
roughly 20-millisecond step. That branch was rejected at the model boundary.

The model candidate therefore preserves every checkpoint-native GDN
out-projection and changes only full attention:

```text
native            one CTA per Q head
model optimized   incumbent below KV bucket 256, split16 at/above 256
```

At KV=256, native and model-optimized final residual/final RMSNorm become
byte-identical. This is a direct example of a valid layer opt-in being rejected
from the larger model graph when its error/cost tradeoff stops transferring.

## 64-layer decoder body

The boundary below includes all 64 layers and final RMSNorm but excludes LM
head and argmax.

| KV | Native | Model optimized | Speedup | Wins | Final norm rel L2 |
|---:|---:|---:|---:|---:|---:|
| 256 | 19.9831 ms | 19.7090 ms | 1.0018x | 4/5 | 0 (exact) |
| 1K | 20.5875 ms | 19.7813 ms | 1.0426x | 5/5 | 0.02731 |
| 8K | 28.4690 ms | 20.8392 ms | 1.3647x | 5/5 | 0.02286 |
| 32K | 55.1344 ms | 23.4897 ms | 2.3479x | 5/5 | 0.02507 |

This is the first evidence that all 48 recurrent states and all 16 attention
layers execute in one ApxInf model graph.

## Complete one-token boundary

The complete-token timing includes 64 layers, final RMSNorm, W8 LM head, BF16
logit D2H, CPU argmax, and the resulting token ID. Weight loading, cache/state
reset, and embedding-row disk I/O are excluded.

| KV | Native | Model optimized | Optimized tok/s | Speedup | Token match | Logit rel L2 |
|---:|---:|---:|---:|---:|---|---:|
| 256 | 21.5877 ms | 21.2793 ms | 46.99 | 1.0058x | 10748 = 10748 | 0.01281 |
| 1K | 22.2343 ms | 21.3821 ms | 46.77 | 1.0397x | 10748 = 10748 | 0.01384 |
| 8K | 30.1030 ms | 22.2978 ms | 44.85 | 1.3505x | 10748 = 10748 | 0.01313 |
| 32K | 56.8616 ms | 25.1563 ms | 39.75 | 2.2594x | 10748 = 10748 | 0.04175 |

Every measured invocation emits the same native/optimized token. The W8 LM
head is shared by both arms; logit differences originate from attention
reduction order propagated through the model.

## Stateful multi-token trajectories

Each trajectory resets once, then continuously mutates all 48 recurrent
states, all 48 conv states, and all 16 KV caches. Every emitted token is loaded
from the real checkpoint embedding row and fed into the next step.

| Prefix / steps | Exact native/optimized tokens | Native tok/s | Optimized tok/s | Speedup |
|---:|---:|---:|---:|---:|
| 256 / 16 | 16/16 | 46.56 | 46.74 | 1.0040x |
| 8,192 / 16 | 16/16 | 33.20 | 44.79 | 1.3492x |
| 32,752 / 16 | 16/16 | 17.58 | 39.62 | 2.2534x |

The maximum trajectory fills positions 32,752 through 32,767 and remains
exact between selectors. This is a complete token-ID trajectory, not a cosine
comparison over categorical labels.

## Memory admission

During the 32K complete-token run, `nvidia-smi` observed:

```text
memory.used  18,061 MiB
memory.free   6,021 MiB
```

This includes the 64-layer text weights, BF16 KV for 16 attention layers,
all recurrent/conv states, shared workspaces, W8 LM head, and logits. It omits
the vision tower and full embedding table. The standing vLLM reference used
22,477 MiB under its broader FP8-KV service contract, so the numbers are not a
like-for-like memory claim; they do prove substantial headroom for the current
ApxInf text runtime.

## Current comparison with frozen vLLM

The frozen vLLM service reports roughly `83.8..87.1 ms` TPOT and
`11.49..11.93 token/s` across the declared context cells. The ApxInf offline
complete-token path reports `21.38 ms / 46.77 token/s` at 1K and
`25.16 ms / 39.75 token/s` at 32K.

That is approximately 3.9x the vLLM decode throughput at 1K and 3.5x at 32K,
but **it is not yet an end-to-end service promotion** because:

- ApxInf is measured offline without HTTP queue/scheduler overhead;
- ApxInf uses BF16 KV while the reference service uses FP8 KV;
- ApxInf uses a W8-converted LM head;
- prompt prefill is not an M>1 kernel path;
- concurrency, CUDA Graph, cancellation, EOS scheduling, and tail latency are
  not implemented in the ApxInf service boundary.

The result proves a large decoder-compute advantage worth integrating into a
service; it does not yet prove API-level TTFT/TPOT superiority.

## Functional CLI generation

The CUDA release CLI now routes `model_type=qwen3_5` through the native text
runtime. It uses the real tokenizer and official Qwen chat template, including
Python-compatible `startswith`/`endswith` method handling.

```bash
/root/apxinf-target-sm89/release/apxinf generate \
  --model /mnt/user_dir/hanjinchen/models/cyankiwi/Qwen3.8-27B-AWQ-INT4 \
  --prompt 'Hello' --max-tokens 4 --device cuda --no-eos-stop
```

Observed:

```text
prompt tokens    53
model load       50.899 s
serial prefill   1.111 s, 47.7 token/s
decode           4 tokens in 0.071 s, 56.48 token/s, 17.70 ms/token
output prefix    The user said "
```

Serial reuse of the M=1 graph makes this a correct functional prefill, not an
accepted TTFT implementation. The next performance boundary is M>1 W4 GEMM and
prefill attention/GDN scanning.

## Evidence

### 64-layer body

| Artifact | SHA-256 |
|---|---|
| `native_qwen35_64layer_256.json` | `43624651adc45500bfe7d0eba4dcccb8dcda504a0e8df7d9aca8fd093b683cc6` |
| `native_qwen35_64layer_1024.json` | `c1c25b3f29cc8b5526f4c01a896d578d158f039ef66f7b32f247f8aeb5cc53a6` |
| `native_qwen35_64layer_8192.json` | `27f743b429ea4509097447483da647da468b62eb46ce2f6744d3a1e26325fd58` |
| `native_qwen35_64layer_32768.json` | `8cabf61d1e9c93619d68cf6082e1ab1df09055b0fd08b490cfad347781575966` |

64-layer example build ID `8e11aed377d3865ce93420b816e6a3a6da6eb124`, SHA-256
`19c1565d3303117e014c4ab4cc6870e833b005fb4458feac1b7baafe42f5ab6e`.

### Complete tokens

| Artifact | SHA-256 |
|---|---|
| `native_qwen35_token_256.json` | `d19e200f698add6f1793cf6e2214c86eb84114c3a43494a5e047b3c4c155506e` |
| `native_qwen35_token_1024.json` | `7f2ec66460d0d52e6a9657a87ad2b46b34346cb94b8f36716be546058ca6cfd9` |
| `native_qwen35_token_8192.json` | `153213ab3aab7db5b76e1cc4765176e8f307828961d7d9cf467bf764f471e7da` |
| `native_qwen35_token_32768.json` | `860a69bf19169acf4327f0131ac14d88d1bd4c909b42082eccbe56d17fe63211` |

Complete-token example build ID `11e9e7f3c7672c32ca38bf60de3adce6247ea560`, SHA-256
`3b60bee4466285efc2632601ea8682399c68c5fbfe8ce23b2fbbfa3bc4e48f62`.

### Stateful trajectories and CLI

| Artifact | SHA-256 |
|---|---|
| `native_qwen35_trajectory_256_16.json` | `bf02a0b05b9192eedad68d765e8f450d88f9746bd4e9a1294fd842ab5a378485` |
| `native_qwen35_trajectory_8192_16.json` | `88214c8d891bc3abc409539c8cc3974b62e38e22831d3b36eb3b1fbcbd35fd4f` |
| `native_qwen35_trajectory_32752_16.json` | `7e2cff3a08e356d6e3b6bdfb7a462f000213c7382da433d32e95f747b6c25ad2` |
| `native_qwen35_cli_hello.log` | `9c2ec09988e37f48ff0cecc81e1ac9aae87d5229bbe331a04dfd8309d98119ea` |

Trajectory example SHA-256
`e07898d89e7f9bd2be0f5ba290336d31878fb61c9c0cc1f245f8157f4e8d7c81`.
The final CLI/service binary has build ID
`d26a94463a33116149f3d429a9ac97b425cec5b3` and SHA-256
`c307d99f2345dea903a543643d746492f3d651871df8343666ede03018e62f3b`.
Its final inspect receipt has SHA-256
`6e64f46e298bcb78803cf73660966067ec252d0d5c22bd44333533d9c4de7880`.

## Decision and next boundary

```text
decision             promote native offline text generation; service continues
covered              tokenizer -> serial prompt -> 64 layers -> logits -> greedy tokens
decode contexts       256, 1K, 8K, 32K
stateful trajectories 256, 8K, 32K; 16 exact steps each
default model arm     checkpoint-native GDN + guarded split-attention
rejected model arm    layer-0 W8 GDN out-proj due 64-layer error amplification
unsupported           multimodal, M>1 prefill, OpenAI service, concurrency, graph
```

The next complete promotion boundary is a resident ApxInf worker with an
OpenAI-compatible endpoint, then matched client-side TTFT/TPOT against the
frozen vLLM service. M>1 prefill is the next major kernel/system bottleneck.
