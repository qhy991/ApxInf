# RTX 4090 bring-up baseline — 2026-08-17

## Decision

`cyankiwi/Qwen3.8-27B-AWQ-INT4` runs successfully on one RTX 4090 with the
frozen eager-mode vLLM reference contract. Text prompts from 1K through 32K,
128-token decode at every context, deterministic retrieval at three positions,
single-image tasks, and a two-image task all complete without request failures
or functional failures.

This is a bring-up baseline, not an apxinf optimization result. No apxinf binary,
package, source directory, container, or service was present on the machine.

## Contract

- GPU: NVIDIA GeForce RTX 4090, 24,564 MiB, compute capability 8.9.
- Driver: 580.95.05.
- Model revision: `63768c10df38c0395e12ef49edac1bd539eaeeea`.
- Runtime: vLLM 0.27.1, PyTorch 2.13.0+cu130, Transformers 5.15.0.
- Quantization: compressed-tensors W4A16, group size 32, asymmetric.
- TP/PP/DP: 1/1/1; max sequences: 1.
- Sequence limit: 33,792 tokens.
- KV cache: FP8 E4M3.
- Execution: eager, chunked prefill enabled.
- GPU memory utilization target: 0.97.
- Sampling: temperature 0, thinking disabled, preserved thinking disabled.

Engine initialization reported 19.23 GiB for model loading, 19.41 GiB consumed
for weights plus non-Torch memory, 1.88 GiB peak activation, 0 GiB CUDA Graph,
and 1.52 GiB KV cache. The allocated KV capacity is 41,902 tokens, or 1.24x one
33,792-token request.

## Text performance staircase

Every cell contains one warm-up followed by three measured requests with exactly
the declared prompt length and 128 generated tokens.

| Prompt | TTFT median (s) | Effective prefill (tok/s) | TPOT median (ms) | Decode (tok/s) | GPU util mean | Memory-controller mean | Peak VRAM (MiB) | Mean power (W) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1,024 | 0.431 | 2,375 | 83.83 | 11.93 | 35.7% | 28.9% | 22,479 | 143.8 |
| 2,048 | 0.827 | 2,476 | 87.06 | 11.49 | 34.4% | 28.3% | 22,791 | 151.9 |
| 4,096 | 1.602 | 2,557 | 85.29 | 11.72 | 40.5% | 30.9% | 22,791 | 170.6 |
| 8,192 | 3.184 | 2,573 | 84.59 | 11.82 | 44.4% | 33.3% | 22,791 | 204.9 |
| 16,384 | 6.525 | 2,511 | 86.96 | 11.50 | 55.2% | 37.5% | 22,791 | 254.3 |
| 32,768 | 13.906 | 2,356 | 84.46 | 11.84 | 68.7% | 42.0% | 22,791 | 314.1 |

Result: 18/18 measured requests succeeded. TTFT and TPOT CV were below 1.5% in
every cell, and the automatic adjacent-context decoder regression rule emitted
no warning. The largest confirmed prompt is 32,768 tokens plus 128 decoded
tokens.

Evidence: `results/20260817-185028/`.

## Long-context retrieval

One exact-key case was run at 10%, 50%, and 90% of every target context. All
18/18 requests returned the exact key. The three 32K TTFT values were
13.913s, 13.920s, and 13.930s; TPOT was 81.4–83.5ms.

Evidence: `results/20260817-185705/`.

## Multimodal

Each deterministic case contains one warm-up and three measured requests.

| Case | Expected / observed | Passes | TTFT median (ms) | TPOT median (ms) |
|---|---|---:|---:|---:|
| Shape counting | `3` / `3` | 3/3 | 227.7 | 85.88 |
| OCR | `APX-4090-Q38-1729` / same | 3/3 | 230.9 | 87.04 |
| Bar chart | `D` / `D` | 3/3 | 234.6 | 87.98 |
| Spatial relation | `blue square` / same | 3/3 | 231.8 | 88.21 |
| Two-image difference | `2` / `2` | 3/3 | 265.2 | 83.05 |

Result: 15/15 measured requests succeeded and passed deterministic validation.

Evidence: `results/20260817-184945/`.

## Hardware interpretation and limitations

`nvidia-smi utilization.memory` is reported only as a memory-controller activity
proxy. It is not measured DRAM GB/s and must not be multiplied by the nominal
1,008 GB/s bandwidth. Actual critical-kernel bandwidth still requires a targeted
Nsight Compute collection; critical-path causality still requires Nsight Systems.

The service runs eager because CUDA Graph profiling reserved about 0.40 GiB and
left only 1.12 GiB for KV, below the 1.20 GiB required by the 33,792-token
contract. Eager recovered 1.52 GiB KV. A future graph candidate must first reduce
the default 16,384-token encoder cache budget or otherwise make memory ownership
explicit, then compare graph with graph and eager with eager.

The standing service has processed 460,945 prompt tokens and 3,465 generated
tokens, remained healthy after the full matrix, and idles at 22,791 MiB VRAM.
