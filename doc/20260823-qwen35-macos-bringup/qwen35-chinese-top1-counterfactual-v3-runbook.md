# Qwen3.5-0.8B Chinese top-1 counterfactual v3 runbook

## Status and scope

**Effective status: `REAL_REJECTED_CANONICAL_STEP9`.** The real v3 build was
attempted and rejected by the mandatory in-memory canonical teacher-forced
gate before save. The candidate target was not published, the four-prompt gate
was not run, and the certified v2 bundle remains unchanged and unreplaced.

The frozen policy artifact is retained byte-for-byte as the historical input
to that attempt. Its embedded `unvalidated-candidate` value records the state
when the experiment was proposed; it is not the current operational status and
is not build authorization. The authoritative closure record is
[`qwen35-chinese-top3-o-proj-counterfactual-rejection-v1.json`](./qwen35-chinese-top3-o-proj-counterfactual-rejection-v1.json).

The selected v3 path was a trusted diagnostic trigger, not a causal
attribution:

```text
language_model.model.layers.19.self_attn.o_proj
```

The profile keeps global affine W8/group-64 quantization and retains exactly
four BF16 paths:

```text
language_model.model.layers.12.linear_attn.out_proj
language_model.model.layers.14.linear_attn.out_proj
language_model.model.layers.19.self_attn.o_proj
language_model.model.layers.20.linear_attn.out_proj
```

It is bound to:

- source revision `2fc06364715b967f1860aea9cf38778875588b17`;
- source manifest `436821ae50e981b9176784ac6ff9548742a865d60d726c58d3bfa9f76d86b500`;
- parent v2 manifest `5f65527db17cd4fb7a1a46ca6ab3940a8bcc8225c620ca10d3016265d2bf8553`;
- BF16 reference manifest `fdce8bac86b1bbc888cac0139065f0291a9a57ce7f448e591b748f4baaad5dea`;
- diagnostic artifact SHA-256 `1b30a3a7f6d609a8265112bde3189b7638a9072561530b852ec86dbc4794b73d`;
- diagnostic content SHA-256 `e0c207bf46a62b643e3aeadc9398aea0d983426585d9b13ce25d21ce35d21a7f`;
- counterfactual policy SHA-256 `7030fe5a7c4dd55cbf158750e9da3a67c7f8e65944b8f8835c75b1093e12eec9`.

The frozen policy ledger predicted 183 quantized modules and four retained BF16
modules. Estimated parameter bytes rise from `805,788,352` to `807,754,432`,
an exact accounting increase of `1,966,080` bytes (1.875 MiB). The complete
serialized-bundle increase was never measured because the candidate failed
before save.

## Real canonical rejection

The real build entered the production teacher-forced harness in memory and
failed during repeat 1 at zero-based step 9:

- expected BF16 teacher token: `25677`;
- independently reproduced v3 token: `248046` (the builder error itself
  reported the mismatch index, not the actual token ID);
- mismatch indices: `[9]` only in the first ten screened positions;
- builder exit code: `2`;
- save started: no;
- target published: no;
- observed elapsed time: `4.14` seconds;
- observed maximum RSS: `1,948,205,056` bytes;
- observed swaps: `0`.

The existing v2 parent manifest
`5f65527db17cd4fb7a1a46ca6ab3940a8bcc8225c620ca10d3016265d2bf8553`
provided one passing observation with the same first-ten teacher harness. No
harness baseline failure was observed; parent-control repeatability is not
claimed. Since canonical admission requires every teacher token to be exact,
the step-9 mismatch is sufficient to reject v3; the remaining canonical steps
and free-run gate cannot rescue it.

## Bounded top-3 closure screen

All seven non-empty subsets of the diagnostic top-3 `self_attn.o_proj` paths
were screened independently. Every case used a fresh process, global affine
W8/group-64, the three certified BF16 GDN output paths, the same production
teacher harness, one independent first-ten observation, and no save. Rank 1
also has the separate real-build repeat-1 mismatch observation, for two rank-1
observations total:

| Restored `self_attn.o_proj` subset | Observed runs | Result |
| --- | ---: | --- |
| layer 19 | 2 | step 9 mismatch; independent reproduction observed `25677` -> `248046` |
| layer 11 | 1 | step 9: `25677` -> `248046` |
| layer 7 | 1 | step 9: `25677` -> `248046` |
| layers 19 + 11 | 1 | step 9: `25677` -> `248046` |
| layers 19 + 7 | 1 | step 9: `25677` -> `248046` |
| layers 11 + 7 | 1 | step 9: `25677` -> `248046` |
| layers 19 + 11 + 7 | 1 | step 9: `25677` -> `248046` |

These are operator-supervised in-memory screen summaries, not
machine-generated formal acceptance receipts. No raw screen sidecar exists.
The common token pattern was observed across all seven cases; it is not a claim
of repeated-identical runs for every case. The observations establish rejection
at a required exact prefix, but do not establish per-case repeatability,
module-causal attribution, general parity, or a formal performance claim.

## Final decision

- v3 top-1 candidate: **REJECT** (`REAL_REJECTED_CANONICAL_STEP9`);
- top3 `self_attn.o_proj` restore family: **STOP**;
- diagnostic top-3 non-empty subset search: **EXHAUSTED**;
- top-4 screen: **DO NOT RUN**;
- further combinations: **DO NOT RUN**;
- parent v2 replacement: **NO**;
- four-prompt quality gate: **NOT RUN**, because canonical admission failed.

The rejected candidate target remains absent:

```text
/Users/haiyan-mini/Agent4Kernel/ApxInf/.apxinf/models/Qwen3.5-0.8B-mlx-w8-g64-gdn3-l19-o-proj-chinese-counterfactual-v3
```

The planned four-prompt evidence target also remains absent. All former build,
verify, and four-prompt commands have been retired from this runbook and must
not be reconstructed or run for this rejected profile.
