# Qwen3.5 Stack3 + tied lm_head v2 formal benchmark contract v1

Date: 2026-08-24

Status: harness committed for dry planning and synthetic/static verification. No
formal model campaign has been executed by this work. The current live source
tree is intentionally post-freeze and therefore fails the frozen custody check.

## Scope and isolation

This contract applies only to the explicit diagnostic route containing:

- six complete three-layer `metal-w8-linear-layer-stack3-v1` stacks over linear
  attention layers `0..2`, `4..6`, `8..10`, `12..14`, `16..18`, and `20..22`;
- six `metal-w8-mlp-block-g64` full-attention MLPs at layers
  `3, 7, 11, 15, 19, 23` with no duplicate MLP execution; and
- the tied lm_head v2 route `metal-w8-top4-f32-rerank`.

It does not change or reuse the result format of the Stack3 v1 or all18 formal
harness. It does not authorize the main, Auto, registry, or default path.

The versioned wrapper is
`scripts/run_qwen35_stack3_head_v2_formal_benchmark.py`. It imports the audited
generic campaign engine from `scripts/run_qwen35_all18_formal_benchmark.py` and
pins that direct file to SHA-256
`971f8ee9778f64c3e68470ff05e5bc71481239054560ec971be0e74811337bfe`.
The wrapper and base engine are rehashed as direct, single-link files when
inputs are frozen, before measurement, and after measurement. Their starting
records are embedded in the dry plan and formal lane identity.

## Frozen correctness oracle

The only admitted archive summary is:

`crates/apxinf-metal/evidence/next-hotspot/qwen35-stack3-head-v2-real-gate-summary-v1-20260824.json`

- format:
  `apxinf-qwen35-metal-w8-stack3-head-v2-real-gate-summary-v1`
- size: `25,937` bytes
- SHA-256:
  `7aa07e1dd7beda066fa7c7048bfcbb0505b793c1caf43aac5f104c4a45177727`
- binary SHA-256:
  `0e70fe6589a77c78c79aa5071741eae27ae184b863b7a49adf47228f86ea1812`

The four archived direct, single-link receipts are pinned through the summary:

| Role | SHA-256 | Size |
| --- | --- | ---: |
| `cpu_teacher128` | `8a75fcc81c7c5aa6427263fe265de28936efc2e36eaab95264dc743b4b3b0cd7` | 14,775 |
| `candidate_teacher128` | `aceeac13bd4d3fad03b26e307f6f3858861fa719239fa50c91519f33b8f784d5` | 37,421 |
| `cpu_free128` | `8c61ad71dc8f8b73755abaf168d772da71e7150a21799142a7f83310e5e765d2` | 14,316 |
| `candidate_free128` | `fac7f7230589650b8bd8924443012982df296301fcdc02e8cca4f1b4b14f9b22` | 28,710 |

All four identities and end-custody receipts must be byte-for-byte equivalent
as parsed JSON. The teacher chain must contain exactly 128 inputs and outputs,
with `teacher_input_ids[0] == prefill_token` and every later input equal to the
previous CPU output. Candidate-hidden CPU/F32 winners, F32-reranked winners, and
the frozen CPU outputs must all agree at every step; each winner must occur in
its four-token Metal candidate set.

The frozen `cpu_free128.generated_token_ids` array is the immutable
`frozen-reference-oracle`. Every timed A receipt and every timed B receipt must
independently reproduce all 128 token IDs. Every B receipt must embed the exact
direct-file record of that CPU-free reference through `--input-receipt`.

## Required v2 path and ledger

Every timed candidate receipt must use free-run format
`apxinf-qwen35-metal-w8-stack3-head-v2-free-run-gate-v1`, mode
`metal_w8_stack3_head_v2_free_run`, and generation schema
`apxinf-qwen35-stack3-lm-head-generation-path-v2` through the shared production
`generate_streaming` path.

For a 128-token free run, the six Stack3 lanes and six full-attention Metal MLPs
must each report 127 body decode calls. The tied head must report exactly one
prefill call, 127 decode calls, and zero teacher calls. All mechanisms,
transaction counters, finite-check fields, terminal state, layer indices, and
path-check booleans are binding; unknown or missing fields are rejected.

The resident-MTLBuffer ledger is exact:

- body: 504 buffers = 444 shared + 60 private, 528,605,184 bytes;
- tied head: 5 buffers = 4 shared + 1 private, 271,169,552 bytes;
- composite: 509 buffers = 448 shared + 61 private, 799,774,736 bytes;
- per composite call: 13 command buffers, 38 compute encoders, 13 commits,
  13 waits, 53,248 bytes H2D, and 49,168 bytes D2H.

Every nested Stack3, full-attention MLP, head, body, and composite ledger field,
including scope and exclusions, is checked exactly. Host F32 tied embeddings,
the exact four-candidate rerank, other CPU weights, host allocations, Metal
pipeline/library/queue and command objects, driver allocations, and KV cache
remain excluded from resident MTLBuffer accounting.

## Formal schedule and promotion gates

The schedule is fixed to `ABBA, BAAB, ABBA, BAAB, ABBA, BAAB`, producing exactly
12 CPU-free A samples and 12 Stack3+lm_head v2 B samples. No warmup or replacement
run is admitted into the campaign.

All of the following are required:

- all 24 receipts reproduce the frozen 128-token trajectory and frozen custody;
- all six block medians point in the candidate-faster direction;
- median candidate throughput speedup is at least `1.10x`;
- median candidate/CPU TTFT ratio is at most `1.10`;
- process-group peak RSS is positive and strictly below 6 GiB for every run;
- `/usr/bin/time -l` reports zero child swaps for every run;
- system swap is unchanged from quiet preflight through all blocks;
- each run has complete quiet start, online, and end samples with no thermal
  throttling or unowned process contamination;
- each child uses the fixed 600-second timeout and 4 MiB cap for each output
  stream; and
- binary, complete source/shader closure, profile, source lock, exact model
  artifact closure, CPU oracle, wrapper, and base engine rehash identically at
  campaign start and end.

The archived single-pass free observation had a throughput ratio near `2.768x`
but a TTFT ratio near `1.5457`. It is diagnostic only. The formal harness must
reject that TTFT regression even if decode throughput passes by a wide margin;
the `1.10` TTFT limit is not relaxed.

## Safety and publication

Without `--execute`, the wrapper can only validate custody and emit a plan with
`execution_started=false`. `--execute` additionally requires an absolute output
directory outside the frozen model directory. Commands are argument arrays, not
shell strings. The audited engine performs quiet preflight before creating the
campaign directory, supervises a private process group, preserves all samples,
and publishes receipts and the final report with no-replace direct-file
semantics. Failure and interruption reports set `formal_accepted=false`; an
interrupted or contaminated run is never silently replaced.

The formal result format is
`apxinf-qwen35-stack3-head-v2-formal-benchmark-v1`. Passing correctness evidence
alone never implies performance promotion.

## Current post-freeze blocker

After the archive was frozen, another optimization slice changed
`crates/apxinf-metal/src/lib.rs`. The archived receipt pins that file to SHA-256
`48776e1f8a85e8d53bb3906a4f515996573b484d8e68a72dd7cb7d480daf47d1`.
The formal wrapper intentionally does not update the summary or source pin to
follow the live tree. Consequently the current real dry-plan and any attempted
execution fail closed during source custody admission, before quiet preflight,
output creation, or model startup. A future formal campaign requires restoring
the exact frozen closure or producing a separately reviewed, newly versioned
correctness archive and harness contract.
