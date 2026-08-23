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

- ApxInf is the stronger short-to-mid-context, single-request text decoder.
- vLLM-Omni is substantially stronger in long-context prefill and remains
  stronger on the real image path after ApxInf's indexed-window promotion.
- Both reach the complete 32,760 prompt + 8 output contract after vLLM's KV
  reservation is aligned to the declared single-request workload.
- Both understand the frozen real PNG and WAV. vLLM has much lower
  service-level multimodal latency; ApxInf retains lower text decode TPOT.

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

Each row requests eight outputs. Numbers are p50 over three measured requests
after one warmup.

| Prompt | ApxInf TTFT | vLLM TTFT | TTFT winner | ApxInf TPOT | vLLM TPOT | Wall winner |
|---:|---:|---:|---:|---:|---:|---|
| 8,192 | 0.816 s | 0.512 s | vLLM 1.59× | 10.58 ms | 19.11 ms | vLLM 1.38× |
| 12,288 | 1.372 s | 0.830 s | vLLM 1.65× | 13.03 ms | 19.09 ms | vLLM 1.52× |
| 32,760 | 5.685 s | 2.912 s | vLLM 1.95× | 24.31 ms | 17.58 ms | vLLM 1.93× |

vLLM's tiled FlashAttention path wins prefill increasingly with context. ApxInf
keeps the faster decoder through 12K, but its long-KV decode crosses over by
32K. The minimum-BWU lower bound at 32K is 30.11% for ApxInf and 41.65% for
vLLM.

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
| PNG, 1,760 + 16 | both read `TinyLlama-1.1B Decode Throughput vs Position` | TTFT 0.758 s; wall 6.959 s | TTFT 0.232 s; wall 0.565 s | vLLM TTFT 3.27× and wall 12.32× lower |
| WAV, 52 + 16 | both identify a sine wave | model TTFT 22.15 ms; TPOT 8.16 ms; wall 5.937 s | client TTFT 290.40 ms; TPOT 21.89 ms; wall 0.619 s | ApxInf decode is faster, but vLLM wall is 9.59× lower |

Indexed group plans plus vision-only FA2 reduce ApxInf PNG TTFT by 44.0× and
wall time by 5.68× relative to its original external-baseline measurement.
The remaining 12× service-wall gap is now primarily the per-request external
Python image processor rather than GPU full attention. The WAV split identifies
the same boundary: ApxInf's model execution is fast, but external processing
dominates service wall time. vLLM keeps processing in its resident service.

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

1. Move image/audio processing into the resident service and remove
   per-request Python startup, especially for audio.
2. Pack or reorder the 28 indexed vision windows so a batched tiled attention
   path can replace their remaining 0.504-second GPU interval.
3. Replace the remaining long-prefill text attention algorithm with a tiled,
   FlashAttention-class path while preserving the complete trajectory gate.

Only after these boundaries move should smaller decode-kernel fusions again
become the primary optimization target.
