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
```

Every request is greedy, non-streaming, `ignore_eos=true`, concurrency one,
and uses deterministic pre-tokenized IDs. Raw trials retain output-trajectory
hashes, client wall time, TTFT, TPOT, throughput proxies, peak memory, GPU and
memory-controller utilization, clocks, and power. Context probing stops at the
first failed case; service recovery is an operator action and must be recorded
separately rather than hidden by the benchmark.
