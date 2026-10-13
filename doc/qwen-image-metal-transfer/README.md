# Qwen Image block0 transfer references

This branch continues `codex/apxinf-mlx-models` at `d5185c3a3151b95e933a719acee8254fbc0e590c`.
It preserves reusable Qwen Image 2.1 block0 implementations and their evidence boundaries for a future ApxInf Rust Metal task.
The existing Rust MLX backend remains unchanged. Qwen Image is not registered as an ApxInf model by this commit.

## Start here

Use `scripts/qwen_image_metal_transfer/candidates/q4-a-concat-compile-v1.py` for the compressed Q4 lane.
Use `fc09.py` as the historical dense BF16 GEMM reference when the deployment contract explicitly permits resident dense weights.
Use other candidates only for the mechanism and coverage listed below.
Run `python3 scripts/qwen_image_metal_transfer/verify_bundle.py` to verify the retained bytes without importing MLX or loading a model.
Read [the manifest](MANIFEST.json) for exact origin commits, source hashes, original archive paths and recorded checks.
Read [the next-task guide](NEXT-TASK.md) before starting new hardware work.

## Candidate selection

| Candidate | Reusable mechanism | Original evidence boundary |
| --- | --- | --- |
| Q4 A | Packed QKV and gate/up plus local compiled seams | Four-cell isolated Q4 protocol passed; 0.98%-1.19% recorded reductions; no independent GPU rerun or full-image qualification |
| fc03 | Dense preparation and compiled glue | Historical complete-layer pass; component effects were not isolated |
| fc05 | Dense gate/up packing | Historical complete-layer pass; different storage policy from Q4 |
| fc06 | Dense QKV and gate/up packing | Historical complete-layer pass; no physical dispatch attribution |
| fc07 | Modulation and residual cast boundaries | Four-cell numerical checks passed; fourth-cell native timing control failed |
| fc08 | Compiled normalization/modulation | Four-cell numerical checks passed; fourth-cell native timing control failed |
| fc09 | 64x64x32 BF16 NT GEMM | Historical selected resident layer; approximately 10% versus its native arm; no full-image qualification |
| fc10 | GEMM gated-residual epilogues | Historical complete-layer pass; no paired incremental proof against fc09 |
| fc11 | GEMM SwiGLU epilogue and interleaved gate/up | Only first-cell numerical checks; no candidate timings |
| fc12 | 128x64 BF16 GEMM and gated epilogue | Four-cell numerical checks passed; fourth-cell native timing control failed |

The BF16 candidates prepare full dense weights from an affine Q4 checkpoint.
Their warm layer timing excludes preparation. Their results do not qualify compressed Q4 execution, per-call decoding or full-model capacity.
A previous all32 attempt added 13 GiB logical private weights on a 16 GiB M4 and stopped before producing an image. OOM was not proven.
A timing-control failure leaves performance unknown. It does not prove a candidate slower, equal or faster.

## Retained contracts

The historical layer uses real block0 weights and synthetic activations for image tokens 1024 and 2304, each with text tokens 128 and 512, at batch 1.
Activations and residuals are BF16. RoPE is FP32. Preserve affine scales, biases, group size, bit order, row mapping and cast order.
Recorded two-seed, changed-live-input and prior-output-survival checks establish only their archived operator coverage.
The old numerical tolerances do not establish image quality.

The Python candidates are exact historical source references. They depend on MLX and the named mflux fused helper.
They are not standalone Rust or MSL implementations. MLX supplies kernel signatures, template specializations and launch metadata.
A Rust port must bind array ownership, strides, temporary buffers, synchronization and rounding explicitly.
Follow [MLX backend ownership](../mlx-backend.md) and [model-layer ownership](../model-layer-architecture.md).

## Profiling and provenance

Native and candidate-A Metal captures exist in the original Q4 archive. Source binding was reviewed on CPU.
The two captures total 2,221,909,370 bytes, exceeding the original combined 2 GiB bound. No exported attribution or counter conclusion is available.
The retained captures are diagnostic evidence, not latency measurements. Trace buffers, checkpoints, calibration tensors and credentials are excluded from this branch.
Raw results and controller history remain in their original experiment repositories. Curated origin hashes and limitations remain in MANIFEST.json.

This branch does not renew either historical clock, change a judge or launch a campaign.
