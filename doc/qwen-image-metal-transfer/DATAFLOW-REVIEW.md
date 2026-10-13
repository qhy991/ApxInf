# Root review of the first CPU dataflow research

Recorded 2026-10-12 22:50 UTC. Root read the complete LAYER-DATAFLOW.md and MEGAKERNEL-SOURCES.md. Both roles have now recorded the limited CPU research acknowledgment. This review permits continuation of the already released reading/design phase only.

Root independently read pinned scaled_dot_product_attention.cpp:1010-1061, custom_kernel.cpp:1-105, the metal_kernel.cpp default-contiguity declaration, and the current fc13 custom-kernel construction and attention wiring. Full-attention output allocation explicitly uses strides [T*C,D,C,1] and row_contiguous=false. Transposing to token-major order supports a contiguous flattening view. The custom input check copies a non-row-contiguous array when ensure_row_contiguous is enabled; fc13 does not supply an override. These source regions support the documented conditional counterexample to fc13's copy-removal premise.

Keep the conclusion conditional on the actual installed execution path. This review does not establish wheel/C++ binary identity, evaluated runtime strides, actual copy dispatches, latency regression, or numerical failure. No model, source test, compilation or GPU probe ran. Preserve the unsubmitted fc13 proposal without repair or admission.

Continue with personally read scheduler/attention/GEMM sources and at least two materially different designs as already authorized. Distinguish logical payload from measured traffic and retain unresolved ownership, synchronization and runtime facts. Do not mark a direction closed from the map. GPU calibration remains unusable; no optimization, calibration retry or E2E execution is released. Research cutoff 00:38:25 UTC; hard end 00:58:25 UTC.
