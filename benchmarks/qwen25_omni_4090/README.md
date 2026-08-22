# Qwen2.5-Omni RTX 4090 service baseline

This directory owns the no-profiler performance and context-limit baseline for
the native `Qwen/Qwen2.5-Omni-3B` service. Run the benchmark on the GPU host
while the service itself is owned by `gpu-run`; the client does not create a
second CUDA context.

The timing authority is client-observed wall time plus the service-emitted
TTFT/TPOT from `/v1/evaluations/generate`. `nvidia-smi` samples are explanatory
hardware evidence, not a replacement for endpoint timing. The prefill rate is
named a proxy because TTFT includes first-token work.

```bash
python3 benchmarks/qwen25_omni_4090/benchmark_service.py \
  --suite quick --warmups 1 --repeats 3

python3 benchmarks/qwen25_omni_4090/benchmark_service.py \
  --suite context \
  --lengths 1024,2048,4096,8192,12288,16384,24576,32760 \
  --context-output-tokens 8 --warmups 1 --repeats 3 --timeout 300

python3 benchmarks/qwen25_omni_4090/benchmark_contract.py \
  --output benchmarks/qwen25_omni_4090/results/contract.json

python3 benchmarks/qwen25_omni_4090/benchmark_multimodal.py \
  --image scripts/roofline_decode_throughput.png \
  --audio /var/lib/agent-gpu-broker/apxinf-omni-tone.wav \
  --output benchmarks/qwen25_omni_4090/results/multimodal.json

python3 benchmarks/qwen25_omni_4090/decode_roofline.py \
  --tpot-ms 8.276138574803148 --kv-len 128 \
  --peak-bandwidth-gbps 1008
```

`decode_roofline.py` reports algorithmic weight/KV byte lower bounds and an
effective BWU estimate. MFU is emitted only when the caller also supplies an
explicit dense-peak convention through `--peak-tflops`; neither estimate is a
replacement for profiler memory transactions or no-profiler endpoint timing.

Every request is greedy, non-streaming, `ignore_eos=true`, concurrency one,
and uses deterministic pre-tokenized IDs. Raw trials retain output-trajectory
hashes, client wall time, TTFT, TPOT, throughput proxies, peak memory, GPU and
memory-controller utilization, clocks, and power. Context probing stops at the
first failed case; service recovery is an operator action and must be recorded
separately rather than hidden by the benchmark.

The promoted text-only path uses 256-token causal chunks from 4,096 through
6,144 prompt tokens, 512-token chunks for the other prompts through 12,288,
and 1,024-token chunks above that measured crossover. It reaches the complete
service contract at 32,760 prompt + 8 output tokens on the 24 GiB RTX 4090.
Image and audio
requests deliberately retain their processor-owned, unchunked path. Only the
final text chunk runs output normalization and the LM head; earlier chunks
publish KV state directly without synchronizing unused logits through CPU.
`benchmark_contract.py` verifies that over-context, over-completion,
non-greedy and streaming evaluation requests fail as typed HTTP 400 errors
without poisoning the service. `benchmark_multimodal.py` compares complete
image/audio output-token sequences against a frozen accepted report; its
single observations are correctness coverage, not timing admission samples.

The accepted deployment keeps all optimized paths explicit through
`APXINF_BATCHED_GQA_PREFILL=1`, `APXINF_STREAM_ORDERED_ALLOC=1` and
`APXINF_TMROPE_POSITION_CACHE=1`, `APXINF_SOFTMAX_EXP_CACHE=1`,
`APXINF_SOFTMAX_EXP_CACHE_LONG_FALLBACK=1`,
`APXINF_QWEN25_CHUNKED_PREFILL=1` and `APXINF_QWEN25_DECODE_GRAPH=1`, plus
`APXINF_QWEN25_GPU_ARGMAX=1`. The decode
graph and exact two-stage GPU token selection are deliberately restricted to
SM89 one-token decode with `start_pos < 3072`; prefill and longer-KV decode
keep the accepted ordinary path. Decode beyond the tested exp-cache range
uses the explicitly selected exact scalar softmax; without that selector it
fails closed. `APXINF_QWEN25_PACKED_QKV=1` selects one
packed QKV owner shared by both paths. `APXINF_QWEN25_FUSED_TMROPE_KV=1`
publishes rotated K and unchanged V directly to their caches during graph
decode. The Broker-owned runit reference is checked in at
`service/apxinf-qwen25-omni-broker.run`; it is the environment and launch
authority for reproducing the promoted service. Unset or `0` preserves the
corresponding native path, while invalid values fail closed.
