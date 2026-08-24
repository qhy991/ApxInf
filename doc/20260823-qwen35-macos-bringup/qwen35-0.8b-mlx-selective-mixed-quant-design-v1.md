# Qwen3.5-0.8B MLX selective W4/W8/BF16 design v1

## Status and claim boundary

This change establishes the offline policy, search, bundle-build, and static
verification contracts.  No Qwen checkpoint was loaded and no throughput was
measured while implementing it.  In particular, this document does **not**
claim that W4, or any selective policy, is exact or faster than the existing W8
v2 bundle (about 95 token/s under its non-formal local measurement).

Every `exact` status in this design means exact only on the one frozen
canonical 13-token/128-step trajectory.  It is not general model parity, broad
prompt coverage, or a default-ready claim.  Publication admission can add
independent prompt gates later without weakening or reinterpreting this frozen
single-trajectory receipt.

A real bundle is publishable only if both its in-memory pre-save runtime gate
and a fresh load from the staged saved bundle reproduce the frozen 128-token
BF16 teacher trajectory twice under the canonical 13-token chat prompt and
`mlx-generate-step-argmax-v1`.  Static `--verify-only` never re-runs those gates
and explicitly reports that limitation.

## MLX-LM 0.31.3 capability audit

The pinned local installation supports the required per-module decision:

- `.apxinf/toolchains/mlx-lm-0.31.3/lib/python3.14/site-packages/mlx_lm/utils.py`
  defines `quantize_model(..., quant_predicate=...)`.  Its wrapped predicate
  accepts `False`, `True`, or a parameter dictionary.  A dictionary is recorded
  under the exact module path in `config["quantization"]` before `nn.quantize`.
- The loader in the same file checks an exact module path in
  `config["quantization"]` and returns that path's dictionary to
  `nn.quantize`; otherwise it uses the global W4 defaults when the packed
  `scales` tensor exists.  This makes the saved configuration reloadable.
- `mlx_lm/convert.py` ships `mixed_quant_predicate_builder`, which uses the same
  per-module dictionary contract.

The exact persisted shape locked by the fake round-trip test is:

```json
{
  "bits": 4,
  "group_size": 64,
  "mode": "affine",
  "language_model.model.layers.N.some_proj": {
    "bits": 8,
    "group_size": 64,
    "mode": "affine"
  }
}
```

W4 paths use the global entry, W8 paths have exact overrides, and BF16 paths do
not appear in the MLX quantization object.  A separate ApxInf manifest binds
the BF16 path set, exact policy-document and search-receipt hashes, certified
source revision/config/schema/tensor count/full artifact manifest, candidate
hash, trace hash, and byte ledger.  `quantization` and
`quantization_config` must be exactly equal.

The production source identity is derived from the existing jointly validated
Qwen profile and source lock: source-lock semantic hash
`021209cc96e398db4aac6d126890f7bb5a5a3b5fce7204fed0328f544cbb7500`
and canonical six-artifact manifest
`436821ae50e981b9176784ac6ff9548742a865d60d726c58d3bfa9f76d86b500`.
Both the current source and the policy must name that exact full manifest, so a
same-schema checkpoint with rehashed weights cannot claim the frozen revision.

The fake round-trip additionally checks packed tensor schemas independently:

- W4: `U32`, packed input dimension `original * 4 / 32`, plus BF16 group-64
  scales and biases;
- W8: `U32`, packed input dimension `original * 8 / 32`, plus BF16 group-64
  scales and biases;
- BF16: original dtype and shape, with no scales or biases.

MLX-LM 0.31.3 declares MIT in its installed distribution metadata and includes
the corresponding license file.  The new ApxInf implementation was written
against the public API behavior; no MLX-LM source was copied.  ApxInf still has
no repository-root license, so redistribution terms for the combined repo need
an explicit project decision.

## Frozen policy and bounded search

`scripts/plan_mlx_mixed_quant.py init` scans only local JSON, tokenizer files,
safetensors headers, and hashes.  It does not import MLX or read tensor payloads.
The resulting generation-zero policy fixes:

- Hugging Face repo ID and immutable source revision;
- source `config.json` SHA-256, canonical language-schema SHA-256, and tensor
  count;
- the full sorted candidate module list and its SHA-256;
- W4 affine/group-64 as the default and an initially empty override list;
- the exact prompt, 128 teacher IDs and hash, production generation semantics,
  and repeat count;
- parent policy, observation, transition, and receipt hashes.

An observation must contain two complete state-aligned teacher-forced runs and
two complete asynchronous free runs.  A divergent observation may change only
one uniquely selected frozen candidate by one tier:

```text
W4 -> W8 -> BF16 -> STOP
```

Localization is bound to a separate runner receipt and fixes a 32-step screen,
128-step gate, `(layer, family)` grouping, hidden-error plus top-1-margin
ranking, and single-module/no-combination scope.  Non-deterministic repeats and
an already-BF16 sensitive module return an evidence-bound terminal `STOP`
without policy mutation.  An unbound path or partial/internally inconsistent
trajectory fails closed.  These unsigned planner hashes provide structural
integrity for observations from a trusted runner; they cannot authenticate a
fully fabricated observation that an adversary rehashes.  Such a document
cannot by itself authorize publication: the builder independently gates the
actual weights before and after save/reload.

An exact all-W4 result may be labeled `exact-pareto` within this three-tier
policy family.  An exact policy with any W8/BF16 override is only an
`exact-candidate`; reverse ablation is still required before a Pareto/minimality
claim.

Policy outputs use same-filesystem atomic no-replace publication.  Existing
policy files and model bundles are never overwritten.

## Offline commands

Create an all-W4 policy from a frozen local source and a 128-step trace file:

```bash
python scripts/plan_mlx_mixed_quant.py init \
  --source-dir /absolute/frozen/Qwen3.5-0.8B \
  --repo-id Qwen/Qwen3.5-0.8B \
  --revision 2fc06364715b967f1860aea9cf38778875588b17 \
  --trace-contract /absolute/trace-contract.json \
  --output /absolute/policies/all-w4.json
```

Advance one evidence-bound tier:

```bash
python scripts/plan_mlx_mixed_quant.py advance \
  --policy /absolute/policies/all-w4.json \
  --observation /absolute/observations/generation-0.json \
  --output /absolute/policies/generation-1.json
```

After real-model execution is explicitly approved, build a candidate.  The
builder rejects before `save` if one of the 128 tokens differs, and rejects the
staged bundle if a fresh reload differs:

```bash
python scripts/build_mlx_bundle.py \
  --source-dir /absolute/frozen/Qwen3.5-0.8B \
  --output-dir /absolute/new/no-replace-bundle \
  --mode affine-w4-g64 \
  --mixed-policy /absolute/policies/generation-N.json \
  --source-revision 2fc06364715b967f1860aea9cf38778875588b17
```

## Real-run admission sequence (not yet executed)

1. Confirm that no other large model process is occupying memory, report that
   check, and wait for explicit approval.
2. Run the all-W4 candidate first.  Do not save a divergent bundle.
3. If it diverges, load BF16/W4 in one process only if module replacement and
   state capture can be demonstrated safe.  Rank `(layer, family)` groups using
   state-aligned hidden error and top-1 margin over 32 steps.
4. Promote one module at a time, first to W8 and then to BF16.  If safe dynamic
   replacement cannot be proven, use small ledgered disk batches rather than a
   187-way bundle sweep.
5. Require the full 128-step teacher/asynchronous gate for every promoted
   candidate.  Once exact, run reverse ablation before claiming Pareto.
6. Publish only the final exact policy through the no-replace builder, then
   verify it from a separate process.  Throughput measurements remain
   non-formal under system noise unless a later benchmark protocol promotes
   them.

## Controlled search-runner contract (static/fake phase)

`scripts/run_mlx_mixed_quant_search.py` now defines the evidence-first
orchestration for exactly one search generation.  Its local certification path
authenticates the frozen six-file BF16 source manifest, revision, schema,
candidate set, policy document/artifact/search-receipt hashes, and a canonical
quality-suite file before a backend may be created.  It rechecks the source,
policy, suite, and W4 baseline evidence after evaluation and publishes only an
observation JSON through a same-filesystem no-replace rename.  It never
publishes a model bundle.

The quality suite additionally freezes the real deterministic W4 four-prompt
failure at
`qwen35-w4-multi-prompt-quality-v1.json`: contract content SHA-256
`d52a79e62827913a34e8f3961233aea6b49d91cc317ab6e4a69405b80d9a311f`
and evidence content SHA-256
`04f40f00cb3031a56c53d6e6bbb861f98ba6cbcd272a2e28a4f7185f7bd8373d`.
That evidence is a search baseline, not an admission pass.  It records exact
prefixes English 2, Chinese 1, code 0, and math/structured 10, with position
matches 6.25%, 3.125%, 0%, and 31.25%.  A final candidate must rerun and pass
the independent fixed multi-prompt suite; canonical-chat exactness alone still
means only one frozen trajectory and never means general parity or
default-ready.

The orchestration contract requires separate BF16/current-candidate handles,
a 32-step BF16-teacher-prefix-aligned screen, then two 128-step teacher-forced
and two 128-step asynchronous runs.  A deterministic divergent generation may
evaluate only the uniquely ranked `(layer, family)` group, one independent
saved/static-verified/reloaded counterfactual at a time.  A selected module is
rematerialized and must strictly improve the full 128-step double gate before
the observation can request exactly one `W4 -> W8` or `W8 -> BF16`
transition.  Teacher-forced and async lanes must each avoid both mismatch-count
and first-divergence-prefix regression; an aggregate improvement cannot hide a
regression in either lane.  Ties and nondeterministic repeats produce
evidence-bound STOP; gate, custody, or input failures produce no observation.

The rematerialized bundle used by the selected 128-step gate must have the
same manifest SHA-256 as the bundle that won its 32-step screen.  A different
manifest is a failed deterministic rebuild, not evidence for an upgrade.
Bundle cleanup is best-effort across every live handle: one `close()` failure
cannot skip the remaining handles, and cleanup errors do not replace the
original evaluation failure.

The runner receipt binds all raw trajectories, input/policy/suite hashes,
module and transition, program files, pinned offline runtime, and bundle
manifest hashes.  Dynamic in-memory module replacement is forbidden.  The
current phase provides and tests this orchestration and certification boundary
with fake backends only.  An audited MLX counterfactual materializer and
state-capture adapter are intentionally not implemented yet.  Likewise, HF
offline environment variables alone are not called a network sandbox: the
default runtime receipt honestly sets `network_blocked=false`, which the
production evaluation boundary rejects.  The real search must therefore remain
blocked until the next backend slice supplies a verifiable process-level
network-denial boundary, rather than silently falling back to unsafe in-place
replacement or a self-asserted isolation flag.

## Offline verification completed

The offline targeted suite covers initial all-W4 policy creation, single-tier
W8 and BF16 promotion, terminal STOP/Pareto boundaries, false-exact rejection,
atomic policy writes, exact policy/search-receipt binding, exact mixed config
persistence, strict safetensors byte/range validation, W4/W8/BF16 tensor
packing, pre-save and post-save-reload 128-step divergence rejection, static
verify-only drift/race detection, tensor-to-shard ownership, final directory
re-scan, full source-manifest provenance, and all pre-existing
BF16/W8/W4/hybrid builder behavior.  The complete Python suite also passes:
The controlled runner adds fake coverage for exact, unique W4-to-W8 and
W8-to-BF16 selection, teacher/async disagreement, tied and nondeterministic
STOP, failed selected gates, program/policy hash binding, cleanup/no-publication,
and ingestion of the real W4 multi-prompt failed-comparison evidence.  These
were offline fake/tiny-fixture tests; the runner implementation did not load a
Qwen checkpoint.
