# Prior experience: inputs, not qualification

The prior experiment optimized a Q4 checkpoint using complete dense BF16 weights. That mechanism is excluded here. Its approximately10% block0 result is not a Q4-execution gain, not a new baseline and not an E2E qualification.

Read the transferred dataflow map for shapes, cast seams, residual lifetimes and conditional SDPA physical layout. Preserve its installed-binary/runtime limitations. Read scale-correction: FLOP share, tile count and nominal bandwidth do not prove wall-time bottlenecks, a hardware ceiling, small wave tails or a1% floor. Copied reading records are another author's records, not your own source reading.

All32 dense preparation added13GiB logical private weights on an actual16GiB M4; the first candidate terminated at0/40 without an image. OOM was not proven, and the exit code was lost because the controller failed during a transient resource-tracker cleanup check. New controllers must record exit code before cleanup, preserve failures and reap only owned children.

Reuse the pinned native Q4 quantized_matmul and fused QK/RoPE first. A new hypothesis may concatenate compressed Q/K/V and matching scales/biases or compressed gate/up weights; do not copy the old dense concatenation path or infer its incremental gain. No complete dense weight cache.

v4's two one-sided control tests had inflated family alarm probability under conditional ideal assumptions. v5 uses independent prospective ABBA/BAAB four-call blocks and8-direction Bonferroni .00625; all4 candidate positive p<=.05, no effect floor. Prior v5 calibration was unusable. This fresh explicitly user-authorized experiment must run one new calibration and freeze its own native data; it cannot borrow old PASS or retry a failed calibration until it passes. Physical DVFS/cache/scheduler causes remain unknown.

Primary source versions: MLX0.32.3 64ea011cb65f14d9ce2737e60db9a4ae91ed7441; HazyResearch/Megakernels7309cec801537b61fea3b50d7dfe454a6cde578e; Mirage mpkf9eb70c254acefc9f3667b2a973d0dcf25471fce. CUDA scheduling assumptions need actual Metal synchronization/progress evidence. No profiling/model call outside the trusted service.
