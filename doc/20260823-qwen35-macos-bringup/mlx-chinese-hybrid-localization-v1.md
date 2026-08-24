# Qwen3.5-0.8B Chinese hybrid localization v1

## Current result

This is a diagnostic-only vertical slice. It has not loaded either model and
has not built a candidate bundle.

The already-certified multi-prompt envelope fixes the first Chinese free-run
divergence at zero-based step 46 (the 47th generated token):

| Lane | Token ID | Decoded token |
|---|---:|---|
| BF16 reference | 100745 | `强度` |
| hybrid W8/BF16 | 110926 | `光的` |

The candidate's following 17 tokens equal reference tokens 46 through 62. In
the fixed 64-token window this is one candidate insertion followed by the
reference tail token being truncated:

```text
BF16:   ……即散射强度与波长的四次方成反比……颜色越蓝
hybrid: ……即散射光的强度与波长的四次方成反比……颜色越
```

This explains the recorded 46-token exact prefix and 0.71875 positional match
ratio. It is a token-trajectory mismatch. The diagnostic deliberately records
`semantic_stability.assessed=false` and makes no semantic-equivalence claim;
the similar wording does not relax the existing free-run gate or establish
general parity.

The evidence-only path is runnable now and does not import MLX or open model
weights:

```bash
/usr/bin/python3 -I -B \
  /Users/haiyan-mini/Agent4Kernel/ApxInf/scripts/diagnose_mlx_chinese_hybrid.py \
  --quality-evidence /Users/haiyan-mini/Agent4Kernel/ApxInf/doc/20260823-qwen35-macos-bringup/qwen35-hybrid-w8-bf16-g64-multi-prompt-quality-v1.json \
  --inspect-trajectory-only
```

## Pinned MLX-LM interface finding

The audited `mlx-lm==0.31.3` Qwen3.5 public call surface is insufficient for
trusted per-layer or per-module capture:

- `Model.__call__`, `TextModel.__call__`, and `Qwen3_5TextModel.__call__`
  return logits/final hidden output only; none accepts `output_hidden_states`.
- `generate_step` yields token IDs and log probabilities, not intermediate
  residuals or module inputs.
- `mlx.nn.Module.named_modules()` can enumerate modules, but the pinned MLX
  `Module` has no public forward-hook registration API.

The audit is bound to these installed source hashes:

| Source | SHA-256 |
|---|---|
| `mlx_lm/models/qwen3_5.py` | `f0daa30bba5cb521c8bdfa7093101a544c6a37bbba09bca582288219cb04ae3a` |
| `mlx_lm/models/qwen3_next.py` | `3c572fe3fbb36721efab4d80d1bb6af11beb4ad1caae18deefc9fc84cbcd9b79` |
| `mlx/nn/layers/base.py` | `ec749e1d50fd1a5e57e0aedc8e6eb13fc697e630f59333a0e24aee62a8dc7f0f` |
| `mlx_lm/generate.py` | `270778ad53eaca55a8533d82e6752660fe5d2605c4aa0879b48a50a91f69345f` |

Consequently a hook-based implementation remains unavailable. The independent
`mlx_qwen35_state_aligned_capture.py` backend now implements a source-bound
manual forward instead. Importing the backend still does not import MLX or open
weights; `open_pair` first verifies the exact eight-package lock, all pinned
source hashes, both complete bundle manifests, and the frozen bundle configs.
Any mismatch fails before model loading.

## Minimal read-only custom forward wrapper

The script freezes a small backend interface rather than permitting monkey
patches or module replacement:

1. Load the certified BF16 and hybrid bundles as two independent models in one
   process and bind their handles to the existing bundle manifest hashes.
2. Give each model its own normal Qwen3.5 cache. Cache advancement is allowed;
   weight writes, module replacement, and cross-model cache reuse are not.
3. Drive both models with the fixed Chinese prompt followed only by BF16
   reference tokens. Repeat the complete 64-step teacher-forced capture twice.
4. Mirror the pinned Qwen3.5 forward order explicitly, using the model's
   existing embedding, norms, decoder layers, masks, caches, final norm, and
   head. At every stateless quantized weight module, evaluate the BF16 and W8
   module on the same BF16 input and retain deterministic aggregate metrics
   across all 64 teacher-forced predictor positions.
5. Cover the exact 184-module W8 portfolio inferred from the frozen architecture
   and hybrid policy. The three already-BF16 GDN `out_proj` paths at layers 12,
   14, and 20 are excluded. Qwen3.5-0.8B ties its embedding and output head, so
   the single W8 embedding path accumulates both the input lookup and output
   `as_linear` calls; every other W8 path is called exactly once.
6. At every step, evaluate full-vocabulary logits in float32 and record:
   BF16 top-1 token, BF16 top-1/top-2 margin, candidate top-1 token, and the
   candidate margin of the BF16 reference token versus its best alternative.
7. Require both capture repeats to be byte-identical, then close both handles.

The production implementation additionally compares the manual BF16 logits to
the installed model's official logits with a bit-exact shape/finite/value gate
on every repeat. This gate is implemented but has not yet been exercised on
the real bundles. Until that controlled run succeeds, the wrapper is
fake/tiny-tested only and no real module-ranking claim exists.

For the fixed Chinese input, the manual capture uses 25 prompt tokens plus 63
BF16 teacher-prefix tokens (88 input positions) and scores all 64 reference
tokens. On this 16 GiB Apple M4, the two bundle files contain about 1.448 GiB
of BF16 weights and 0.781 GiB of hybrid weights. A planning estimate is
3.0–4.0 GiB peak process RSS, with 4.5 GiB reserved as a conservative ceiling,
and 20–60 seconds wall time for the required two repeats (30–90 seconds under
desktop noise). These are pre-run resource budgets, not measured performance
evidence; the controlled run must record zero swap and must not promote
throughput.

This gives two intentionally separate diagnostics:

- `trajectory_exact`: the unchanged free-run token comparison;
- `teacher_forced_stability`: reference-token top-1 agreement and margin under
  a BF16 teacher prefix.

`semantic_equivalence_assessed` always remains false. Teacher-forced stability
can show that a one-token free-run rewrite recovers under the reference prefix,
but it cannot turn a free-run failure into a passing gate.

## Receipt and candidate boundary

The fake-tested receipt format is
`apxinf-mlx-chinese-hybrid-diagnostic-receipt-v1`. It binds the certified
quality envelope, contract, bundle manifests, same-process pair identity,
pinned source audit, exact prompt and teacher trajectory, 184-module portfolio,
two identical captures, and capture self-hash.

Module ranking uses only same-BF16-input relative L1 output error. Global
top-1 margin is recorded as context and explicitly marked non-causal at module
level. The receipt returns at most three W8 paths as BF16 restoration
candidates. Those paths are diagnostic hypotheses only: no weight is swapped
in place and no model is declared improved. Each selected path must next be
materialized as its own independent no-replace bundle and rerun through the
unchanged quality gate.

Sixteen fake/tiny tests cover certified trajectory binding, exact insertion
localization, strict read-only capabilities, lazy production-backend loading,
two independent same-process handles, fresh independent caches, deterministic
teacher-forced repeats, the tied-embedding double invocation, BF16
reference-token top-1 and margin aggregation, complete 184-module coverage,
top-three ranking, start/end custody, cleanup, and no-replace publication. The
synthetic top-three paths in the fake test are not real localization results.
