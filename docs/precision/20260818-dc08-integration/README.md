# dc08 CUDA and tuning integration on Thor

Date: 2026-08-18
Host: NVIDIA Thor at 10.30.14.158
Integration branch: codex/20260818-master-dc08-integration
Worktree: /home/wwxq/Projects/users/wwxq/rusin-master-dc08-integration-20260818
Base: origin/master at ea3a4eb1057a1eff127b8187d4d844f12c29fff9
Semantic source: dc08ba067d614aa89db3a961d7725f8464c8c1f7

## Scope and merge policy

The dc08 CUDA and offline-tuning functionality was ported semantically onto current master. The historical branch was not merged because master already contains a parallel repository refactor and the Rusin-to-ApxInf rename. Benchmark prose and old tactic files were deliberately excluded; tactics were regenerated from the integrated binary on the target Thor.

The only source conflict was crates/apxinf-cuda/src/kernels/gemm/mod.rs. It was resolved by preserving the current master physical-operator module layout and adding the dc08 BF16/FP8 tuning exports. No old rusin, RUSIN, or Rusin identifiers remain in the touched CUDA/model trees.

## Integrated functionality

- BF16 cuBLASLt exact-shape offline tuning with vendor fallback.
- Cold-L2 eviction and CUDA-event-only GEMM timing.
- FP8 CUTLASS/cuBLASLt exact-shape offline tuning.
- Persisted exact tactic lookup, validation, and startup installation.
- FP8 GEMM plus bias epilogue.
- Packed vision QKV attention path.
- F16 FA2 sources for SM100-family head dimensions 96 and 256.
- Two-view and three-view PI0.5 schedules for token counts 10, 21, 50, and 200.
- Runtime initialization-time validation of APXINF_CUDA_VISION_QKV_LAYOUT. The per-layer hot path now receives a bound boolean instead of reading the environment.
- A persisted-BF16-tactic numerical test that loads the generated database and follows the production install, lookup, prepare, and launch path.

## Generated tactics

| Precision | File | Exact shapes | Backend distribution | Workload coverage |
|---|---|---:|---|---|
| BF16 | configs/pi05/thor_sm110_bf16_v2_v3_h10_tactics.json | 52 | cuBLASLt 27, vendor 25 | 2/3 views x T10/T21/T50/T200 |
| FP8 | configs/pi05/thor_sm110_fp8_native_v2_v3_h10_tactics.json | 52 | cuBLASLt 47, CUTLASS 5 | 2/3 views x T10/T21/T50/T200 |

Both files report:

- schema: apxinf.cuda.tuning.v1
- kernel_build_id: kb1-72b398f332ec581e84b8489c4d336ddf
- device: NVIDIA Thor, SM 110
- CUDA/cuBLAS: 13.0/13.0
- warmup iterations: 10
- benchmark iterations: 30
- cache policy: cold L2

FP8 records use a 128 MiB eviction buffer for the reported 32 MiB L2 and exclude eviction from CUDA-event timing.

SHA-256:

- BF16: 4b3ffc032c060e472f787a69862d5657ceeac1e8c7b5c2325aa3bc563fd9fe37
- FP8: 1745591f583c29b2a3f926899394f1bc98aad12172daf73101d240d5fd427b19

## Verification results

| Gate | Result | Evidence |
|---|---|---|
| CUDA model examples compile for sm_110/sm_110a | PASS | evidence/cargo-check-cuda-examples.log |
| Release BF16 and FP8 tuners build | PASS | evidence/cargo-build-release-tuners.log |
| PI0.5 schedule unit tests | PASS, 3/3 | evidence/test-schedule.log |
| Full apxinf-model library tests | PASS, 35/35 | evidence/test-apxinf-model-lib.log |
| Full apxinf-cuda tests, serial under GPU lock | PASS, 62/62 | evidence/test-apxinf-cuda-full.log |
| Kernel architecture rules | PASS | evidence/check-kernel-architecture.log |
| Persisted BF16 cuBLASLt tactic vs vendor | PASS, max_abs=0, RMSE=0 | evidence/test-bf16-cublaslt-vendor.log |
| FA2 language MQA vs cuBLAS reference | PASS, 1/1 | evidence/test-fa2-mqa.log |
| Packed vision QKV vs split path | PASS, 1/1 | evidence/test-packed-qkv.log |
| Fused FP8 bias vs decomposed path | PASS, 1/1 | evidence/test-fused-fp8-bias.log |
| Whitespace/conflict audit | PASS | evidence/git-diff-check.log |

All GPU commands first checked the compute-process list and ran under flock on /tmp/rusin-thor-gpu.lock. Only Xorg and gnome-shell graphics processes were present before tuning.

## Failure investigated

The first direct BF16 test launch returned CUBLAS_STATUS_NOT_INITIALIZED because the test bypassed prepare_bf16_gemm. The production runtime already prepares native resources before the first launch. The final test was changed to load the regenerated tactics database and exercise the complete production install/lookup/prepare/launch lifecycle. The original failure log is preserved as evidence/test-bf16-cublaslt-vendor-preprepare-failure.log.

This integration validates compilation, scheduling, tactic generation, database installation, and operator-level numerical equivalence. It does not claim a new end-to-end PI0.5 latency or LIBERO success-rate result. No nsys or ncu profile was needed for this integration gate; the tuning evidence uses the ported cold-L2 CUDA-event methodology. Any later performance promotion should run the full no-profiler benchmark first and attach profiler evidence only for the selected kernel experiment.
