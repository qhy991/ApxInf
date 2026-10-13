# Continue Qwen Image transfer work

Start from this branch, which inherits the existing Rust native MLX backend. Compile a new bounded task before source edits or model execution.
The preparation commit is a reusable reference package, not an active task contract.

1. Choose one storage contract. Prefer compressed affine Q4/group64 for the current 16 GiB M4 target. A dense BF16 experiment needs its own explicit lifetime and capacity budget.
2. Read the exact Q4-A source, current native Rust Array and quantized matmul APIs, and the retained dataflow review. Keep checkpoint mapping and projection packing in the model layer. Keep reusable kernel operations in apxinf-mlx.
3. First port a complete block0 callable using existing safe Rust MLX operations. Preserve the native rollback and match outputs before introducing custom MSL.
4. Prepare source-bound reference inputs and outputs for the new task. Historical fixtures can guide case selection but cannot qualify the new Rust execution.
5. Register correctness, live-input/output lifetime checks, workload shapes, cold preparation, warm latency, peak memory and weight residency. Include transfers, casts, preparation or synchronization when the actual deployed path performs them.
6. Define migration acceptance prospectively. A correct implementation can be useful without acceleration if it removes a concrete deployment dependency or establishes native ownership. Specify a permitted latency regression and a paired confidence decision before collecting data. No numerical allowance is selected by this package. Failure to detect a speedup is not a noninferiority result.
7. Review or collect bounded Metal evidence to establish the next bottleneck. Fix the combined capture-size accounting before reusing the old capture adapter. Shader source geometry or nominal bandwidth cannot establish hardware attribution.
8. Use fc09/fc10/fc11/fc12 only when their dense-weight policy matches the new task. Treat fc11 as incomplete until all target cells pass. Do not combine measurements from different candidates into one claimed implementation.
9. Independently verify the exact Rust candidate. Add full-model real-output and capacity qualification before presenting it as an image deployment path.

Suggested first bounded action: read the current Rust quantized-matmul interface and specify the Q4-A packing and complete-block API without launching a model.
Keep the current experiment clock and receipts intact. This guide grants no GPU work, retries or renewed time window.
