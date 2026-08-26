# CPU Q/G/K/V projection fusion rejection

Date: 2026-08-26

Qualification: checkpoint-free, production-shape synthetic screening on a
noisy host. This is a rejection record, not formal model-performance evidence.

## Hypothesis

The six Qwen3.5 full-attention layers each execute separate CPU F32 Q, gate,
K, and V input projections. Packing them could reduce the complete projection
path from five matmuls per layer (four inputs plus output) to two without
increasing persistent weight payload.

The current `Tensor` has no offset/narrow view, so a fused `[Q,G,K,V]` result
must be copied back into four tensors. At the 0.8B shape this adds 20,480 bytes
of payload copy per layer, or 122,880 bytes and six temporary allocations per
decode token. Weight traffic and FLOPs do not change.

## Paired screen

The existing projection screen was extended in an isolated, discarded
worktree to include the real split cost. Each run used 101 rotated-order
iterations. Speedup percentages versus the current five-matmul path were:

| Variant | Matmuls/layer | Split bytes/layer | Seven run results (%) |
|---|---:|---:|---|
| fused Q+gate | 4 | 16,384 | -2.237, -2.456, -3.232, -1.632, -3.648, -3.173, -1.829 |
| fused K+V | 4 | 4,096 | +0.537, -1.107, -0.599, +1.727, -0.362, -0.313, +1.052 |
| fused Q+gate and K+V | 3 | 20,480 | -2.156, -2.768, -3.060, -2.591, -4.000, -3.610, -2.246 |
| fused Q/G/K/V | 2 | 20,480 | +0.597, +1.027, -0.336, -0.933, +0.048, +0.285, +0.455 |

The full fusion produced only five positive runs out of seven and stayed
inside approximately `-0.93%..+1.03%`. It did not clear the predeclared
stable-positive admission rule. Q+gate and the two-bundle variant regressed
every run.

## Decision

Do not add a packed CPU Q/G/K/V runtime path. Reducing 18 CBLAS entries per
token does not reliably repay the mandatory output split. Keeping both packed
and separate weights as a fallback would additionally duplicate about 120 MiB
of persistent model payload.

Reopen only if a safe contiguous Tensor subview removes the split copies, or
if a complete fused full-attention primitive consumes packed offsets without
round-tripping through four host tensors. Any reopened candidate must again
pass a rotated paired screen and an end-to-end target-path gate.
