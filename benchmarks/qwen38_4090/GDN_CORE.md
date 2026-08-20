# Native Qwen3.8 recurrent GDN core on SM89

This record covers only the one-token recurrent gated-delta rule. It excludes
input projections, causal convolution, g/beta preparation, gated RMSNorm, output
projection, residual, MLP, and full model execution.

## Semantic contract

The implementation is derived from the installed Transformers 5.15.0
`Qwen3_5GatedDeltaNet` and `torch_recurrent_gated_delta_rule` used by the frozen
reference environment.

```text
heads:             48
key dimension:     128
value dimension:   128
Q/K/V:             BF16 [48,128]
g, beta:           FP32 [48]
recurrent state:   FP32 [48,128,128], updated in place
core output:       BF16 [48,128]
Q/K normalization: L2, epsilon 1e-6
query scaling:     1/sqrt(128)
```

For every head and token:

```text
q = l2norm(q) / sqrt(128)
k = l2norm(k)
decay = exp(g)
state_decayed = state_old * decay
key_memory = state_decayed^T * k
delta = (v - key_memory) * beta
state_new = state_decayed + outer(k, delta)
output = state_new^T * q
```

The kernel avoids materializing `state_decayed` and avoids a third state read:

```text
output = decay * (state_old^T * q) + delta * dot(k,q)
state_new = decay * state_old + outer(k,delta)
```

One 128-thread CTA owns one head and one thread owns one value column. Q/K are
normalized once into shared memory. State layout keeps the value dimension
contiguous, so each key-row access is coalesced across the CTA.

## Named correctness endpoints

The deterministic probe executes 128 different recurrent steps and checks:

1. `core_output`: final BF16 `[48,128]` after step 128;
2. `recurrent_state`: complete FP32 `[48,128,128]` after all mutations;
3. `input_immutability`: Q/K/V BF16 bits and g/beta FP32 bits after execution;
4. `finite`: output and state contain no NaN or infinity.

The thresholds were frozen before GPU execution:

```text
output cosine >= 0.9999
state relative L2 <= 1e-3
input tensors byte-identical
all endpoints finite
```

Observed:

```text
output cosine        0.9999999978480585
output relative L2   0.00006561208714676978
output max abs       0.00006103515625
state cosine         0.9999999999997108
state relative L2    4.3547682225341065e-7
state max abs        4.842877388000488e-8
input immutable      true
finite               true
decision             pass, continue to complete GDN layer
```

## Performance and resources

```text
hot L2:       11.00148 us median, 0.0582% CV, 30 x 200 calls
cold proxy:   26.09000 us median, 2.303% CV, 30 calls
resources:    28 registers, 1,056 B static shared, zero spill/local/stack
```

The cold proxy uses the same 128 MiB eviction method documented in `W4A16.md`.
The state payload is 3 MiB; two state reads and one state write imply about 9
MiB of state traffic before smaller Q/K/V/output traffic. The cold host-wall
number includes launch and synchronization overhead and is not an NCU bandwidth
claim. GPU counters remain unavailable because the machine returns
`ERR_NVGPUCTRPERM`.

Remote raw receipt:

```text
benchmarks/qwen38_4090/results/native_qwen35_gdn_core.json
SHA-256 2f9a2589be93547a48bf75712ba54f17c0f9b3f7bbf630ccfa576daec8f715a4
```

Probe binary:

```text
/root/apxinf-target-sm89/release/examples/qwen35_gdn_core_probe
Build ID 68bb1430f77463b3f457380d19f8debc59edbf69
SHA-256 480f3601adc4656d39ba109ce53dde268c11c371ec25ae286d33e46ef5492fed
kernel build id kb1-3c897b599de43b61bf7b1453a0af138b
```

## Promotion boundary and next action

The result is accepted as an operator correctness/performance baseline, not as
a full GDN layer and not as an end-to-end optimization.

The next complete layer boundary is:

```text
RMS-normalized hidden [1,5120]
  -> W4 in_proj_qkv [10240,5120]
  -> depthwise causal conv4 + SiLU and conv-state mutation
  -> Q/K repeat 16 -> 48 heads, V 48 heads
  -> BF16 in_proj_a/in_proj_b -> FP32 g/beta
  -> recurrent GDN core + recurrent-state mutation
  -> W4 in_proj_z
  -> gated RMSNorm per 128-value head
  -> BF16 out_proj [5120,6144]
  -> layer output [1,5120]
```

That layer probe must compare every projection output, conv state, g/beta,
core output, recurrent state, gated norm output, and final layer output against
the same installed reference. Only after it passes may the 48 GDN layers enter a
model trajectory.
