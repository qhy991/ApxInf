# ApxInf versus vLLM-Omni on RTX 4090

## Decision

The external baseline is **vLLM-Omni 0.26.0**, not SGLang-Omni. The current
[vLLM-Omni model table](https://docs.vllm.ai/projects/vllm-omni/en/latest/models/supported_models/)
explicitly lists both `Qwen/Qwen2.5-Omni-3B` and the 7B checkpoint. The current
[SGLang-Omni service list](https://github.com/sgl-project/sglang-omni) centers
on Qwen3-Omni and does not publish a Qwen2.5-Omni deployment contract. Using
vLLM-Omni therefore gives a maintained, same-model comparison instead of a
nearby but structurally different model.

The result is not a single winner:

- ApxInf now wins TTFT, TPOT and wall time in every frozen single-request text
  cell, including the 32K boundary.
- The real-image path is within 3% on client wall after ApxInf's
  grouped-varlen FA2 promotion.
- Both reach the complete 32,760 prompt + 8 output contract after vLLM's KV
  reservation is aligned to the declared single-request workload.
- Both understand the frozen real PNG and WAV. ApxInf now wins real-audio
  service wall time as well as decode TPOT; vLLM still wins the image path.

The structured authority is
`results/apxinf-vs-vllm-omni-0.26.0.json`; this document is its human-readable
interpretation.

## Frozen comparison contract

| Dimension | ApxInf | vLLM-Omni |
|---|---|---|
| Model/revision | `Qwen/Qwen2.5-Omni-3B@f75b40e...` | same local snapshot |
| Precision | BF16 | BF16 |
| GPU | one RTX 4090 | same RTX 4090, run serially through Broker |
| Model stages | Thinker only, Talker disabled | `qwen2_5_omni_thinker_only` |
| Requests | concurrency 1 | `max_num_seqs: 1` |
| Context | 32,768 total tokens | 32,768 total tokens |
| Sampling | greedy, EOS ignored | greedy, EOS ignored |
| Prefix cache | none | disabled |
| Output | text | text |

Text requests have exact prompt and output token counts. vLLM receives the
OpenAI chat message `"x " * (prompt_tokens - 20)`; its `/tokenize` and final
usage record must both equal the requested chat-template length. ApxInf uses
its deterministic pre-tokenized evaluation endpoint. ApxInf TTFT/TPOT are
model-runtime metrics emitted by the service; vLLM TTFT/TPOT are observed from
its true SSE token stream. Wall time is client-observed in both cases, though
the vLLM chat path includes a small tokenizer/API cost that the ApxInf
evaluation endpoint excludes.

Cross-engine token identity is not an admission condition. Each engine must
produce one stable trajectory under its own frozen request. Multimodal output
uses bounded semantic checks: the PNG must contain the visible chart title and
the WAV must be identified as a sine wave.

## Fixed text workloads

All numbers are p50 over five measured requests after one warmup.

| Workload | Metric | ApxInf | vLLM-Omni | Result |
|---|---:|---:|---:|---|
| 1,024 + 32 | TTFT | 64.92 ms | 68.30 ms | ApxInf 1.05× |
| 1,024 + 32 | TPOT | 9.36 ms | 22.62 ms | ApxInf 2.42× |
| 1,024 + 32 | wall | 0.357 s | 0.770 s | ApxInf 2.16× |
| 128 + 128 | TTFT | 15.00 ms | 53.09 ms | ApxInf 3.54× |
| 128 + 128 | TPOT | 8.25 ms | 22.68 ms | ApxInf 2.75× |
| 128 + 128 | wall | 1.065 s | 2.934 s | ApxInf 2.75× |

The 128+128 algorithmic roofline makes the short-decode difference explicit.
With NVIDIA's dense BF16 Tensor-Core convention of 165.2 TFLOP/s and 1,008
GB/s memory bandwidth, ApxInf reaches a 74.23% minimum BWU lower bound versus
27.02% for vLLM. Linear-only MFU is 0.453% versus 0.165%; both are low because
batch-one decode is weight-bandwidth-bound. These are model-byte/FLOP lower
bounds, not Nsight memory-transaction counters.

## Context gradient

Each row requests eight outputs. Current ApxInf numbers are medians from five
paired measurements; vLLM numbers retain their frozen p50 after one warmup.

| Prompt | ApxInf TTFT | vLLM TTFT | TTFT winner | ApxInf TPOT | vLLM TPOT | Wall winner |
|---:|---:|---:|---:|---:|---:|---|
| 8,192 | 0.407 s | 0.512 s | ApxInf 1.259× | 10.69 ms | 19.11 ms | ApxInf 1.337× |
| 12,288 | 0.655 s | 0.830 s | ApxInf 1.267× | 13.07 ms | 19.09 ms | ApxInf 1.284× |
| 32,760 | 2.597 s | 2.912 s | ApxInf 1.122× | 10.24 ms | 17.58 ms | ApxInf 1.134× |

ApxInf's causal FA2 path, FA2-aware 1,024-token chunks and request-scoped early
FA2 remove the former prefill deficit across all frozen lengths. At the 32K
decode boundary, grouped-GQA split-64 attention plus the dedicated long decode
CUDA Graph and packed M1 MLP lower ApxInf TPOT from 24.35 to 10.24 ms and
reverse the last vLLM text advantage: ApxInf is 1.716× faster. The minimum-BWU
lower bound at 32K is 71.48% for ApxInf and 41.65% for vLLM.

Both engines pass 32,760 + 8. vLLM's throughput-oriented automatic policy
reserved 11.45 GiB for 333,552 KV tokens—10.18 simultaneous 32K requests—and
then OOMed on the first 24K MLP prefill. The final single-request baseline
fixes KV storage at 2 GiB, still enough for 58,240 tokens or 1.78 complete 32K
requests. This is workload alignment, not a kernel modification. With that
configuration, sampled peak memory is 16,565 MiB for vLLM versus 15,069 MiB
for ApxInf at 32K.

## Real multimodal inputs

| Input | Result | ApxInf | vLLM-Omni | Interpretation |
|---|---|---:|---:|---|
| PNG, 1,760 + 16 | both read `TinyLlama-1.1B Decode Throughput vs Position` | TTFT 0.257 s; TPOT 10.39 ms; wall 0.581 s | TTFT 0.232 s; TPOT 21.92 ms; wall 0.565 s | vLLM TTFT 1.11× and wall 1.03× lower; ApxInf TPOT 2.11× lower |
| WAV, 52 + 16 | both identify a sine wave | model TTFT 20.12 ms; TPOT 8.09 ms; wall 0.159 s | client TTFT 290.40 ms; TPOT 21.89 ms; wall 0.619 s | ApxInf TTFT 14.43×, TPOT 2.71× and wall 3.89× lower |

Indexed group plans plus vision-only FA2 reduce ApxInf PNG TTFT by 44.0× and
the persistent processor reduces its remaining service overhead, and grouped
variable-length FA2 then lowers PNG TTFT another 2.92× and wall another 1.85×.
WAV remains a non-target control. The model
TTFT/TPOT intervals remain effectively unchanged, while estimated non-model
overhead from the processor promotion stays removed. On this one PNG the final
2.8% wall difference should be called near parity, not a general winner.

## SGLang compatibility status

This exact checkpoint is not currently a native SGLang-Omni model. Its public
model list names Qwen3-Omni, while current stable SGLang 0.5.18 resolves
`Qwen2_5OmniForConditionalGeneration` through `--model-impl transformers`.
Its generic Transformers multimodal processor loads images, but the
audio-loading branch is an explicit TODO.

The exact BF16, 32K, concurrency-one contract was attempted on both 0.5.17 and
0.5.18 after fixing only their environment boundaries. Version 0.5.18 also
ships mutually incompatible Torch 2.13.0 and Torchaudio 2.11.0 CUDA wheels;
removing the unavailable optional audio package makes SGLang import and
explicitly marks audio unavailable. The server still fails before weight
loading: its Transformers wrapper calls `AutoModel.from_config` on the
checkpoint's full `Qwen2_5OmniConfig`, which Transformers 5.12.1 does not
register for `AutoModel`. Consequently no text, image, context or performance
number is valid for SGLang. It is recorded as a capability failure, while
vLLM-Omni remains the maintained same-model baseline. See
`results/sglang-0.5.18-compatibility.json` and the checked-in reproduction
probe; no checkpoint rewrite or synthetic thinker extraction was used.

## vLLM deployment and observed defects

The isolated environment is frozen in
`results/vllm-omni-0.26.0-environment.json`. It uses Python 3.12.14,
vLLM/vLLM-Omni 0.26.0, Torch 2.11.0+cu130, Transformers 5.15.1 and the same
model files. The system vLLM 0.6.2 installation was not changed.

Three integration facts matter:

1. The GPU Broker user inherited root cache paths. vLLM, FlashInfer, Torch,
   Triton and speaker storage therefore have explicit Broker-owned cache
   roots in the checked-in service definition.
2. Even the thinker-only server initializes speech-upload storage. Its public
   `SPEAKER_SAMPLES_DIR` boundary must be writable although Talker and
   Code2Wav are absent.
3. `/v1/completions` returns HTTP 500 in this release because
   `OmniRequestOutput` lacks `ec_transfer_params`. The supported and measured
   authority is `/v1/chat/completions`, which passes text, image and audio.

The final service reserves about 16.56 GiB after initialization. It is stopped
after measurement, leaving the Broker GPU idle.

## Reproduction

The official version pair follows the
[vLLM-Omni GPU installation guide](https://docs.vllm.ai/projects/vllm-omni/en/latest/getting_started/installation/gpu/).
On this host, the isolated setup was:

```bash
python3 -m pip install --prefix /opt/apxinf/vllm-omni-bootstrap uv==0.12.5

UV_PYTHON_INSTALL_DIR=/opt/apxinf/vllm-omni-python \
UV_CACHE_DIR=/opt/apxinf/vllm-omni-uv-cache \
  /opt/apxinf/vllm-omni-bootstrap/local/bin/uv python install 3.12

UV_PYTHON_INSTALL_DIR=/opt/apxinf/vllm-omni-python \
UV_CACHE_DIR=/opt/apxinf/vllm-omni-uv-cache \
  /opt/apxinf/vllm-omni-bootstrap/local/bin/uv venv \
  --python 3.12 --seed /opt/apxinf/vllm-omni-v0.26.0

UV_CACHE_DIR=/opt/apxinf/vllm-omni-uv-cache \
  /opt/apxinf/vllm-omni-bootstrap/local/bin/uv pip install \
  --python /opt/apxinf/vllm-omni-v0.26.0/bin/python \
  --torch-backend=auto vllm==0.26.0

SETUPTOOLS_USE_DISTUTILS=local \
UV_CACHE_DIR=/opt/apxinf/vllm-omni-uv-cache \
  /opt/apxinf/vllm-omni-bootstrap/local/bin/uv pip install \
  --python /opt/apxinf/vllm-omni-v0.26.0/bin/python \
  vllm-omni==0.26.0
```

Start the external service through the same Broker used by ApxInf:

```bash
benchmarks/qwen25_omni_4090/service/vllm-qwen25-omni-broker.run
```

Run the frozen measurements from a second shell on the GPU host:

```bash
python3 benchmarks/qwen25_omni_4090/benchmark_vllm_omni.py \
  --suite quick --warmups 1 --repeats 5 \
  --output benchmarks/qwen25_omni_4090/results/vllm-quick.json

python3 benchmarks/qwen25_omni_4090/benchmark_vllm_omni.py \
  --suite decode --warmups 1 --repeats 5 \
  --output benchmarks/qwen25_omni_4090/results/vllm-decode.json

python3 benchmarks/qwen25_omni_4090/benchmark_vllm_omni.py \
  --suite context \
  --lengths 1024,2048,4096,8192,12288,16384,24576,32760 \
  --context-output-tokens 8 --warmups 1 --repeats 3 \
  --output benchmarks/qwen25_omni_4090/results/vllm-context.json

python3 benchmarks/qwen25_omni_4090/benchmark_vllm_omni_multimodal.py \
  --image scripts/roofline_decode_throughput.png \
  --audio /var/lib/agent-gpu-broker/apxinf-omni-tone.wav \
  --warmups 1 --repeats 3 \
  --output benchmarks/qwen25_omni_4090/results/vllm-multimodal.json
```

## ApxInf priorities implied by the comparison

The next high-leverage work is not another sub-percent pointwise kernel:

1. Re-profile the post-request-scoped-FA2 8K path before selecting another
   prefill rewrite; the prior scalar-attention materialization path is closed.
2. Replace the resident Python processor with native Rust only if CPU/RSS or
   future multi-request evidence makes that boundary material.
3. Re-profile the post-FA2 image path before selecting another vision change;
   the prior indexed-window bottleneck is closed.

Only after these boundaries move should smaller decode-kernel fusions again
become the primary optimization target.
