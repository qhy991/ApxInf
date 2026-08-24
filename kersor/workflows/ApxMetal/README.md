# ApxInf Metal W8 KerSor workflow

This repository-local workflow gives KerSor 3.0.1 one exact, feasible route for
`apxinf_qwen35_metal_w8_head`. It targets Apple M4, Qwen3.5-0.8B, symmetric W8
group-64 tied `lm_head`, deterministic top-4 Metal candidates, and production
native-F32 reranking. It does not claim BF16 or Hugging Face parity.

The workflow is intentionally narrow. An inner agent is read-only and may only
return complete candidate bytes for:

```text
crates/apxinf-metal/src/metal_w8.metal
```

The fixed Host evaluator copies one current repository snapshot into KerSor's
private `command-v1` scratch area, forks isolated A/B trees, and changes only
that shader in B. Thus both variants include the same current Rust/Objective-C++
Host implementation, including `prefill_token_for_generation`; an agent cannot
change the bridge, traits, general model path, CLI, build script, tests, or
evaluator. The candidate is returned to outer KerSor only after acceptance; the
workflow itself never installs it in the source tree.

## Files and authority

- `manifest.yaml` is the catalog routing declaration.
- `apxinf-metal-w8-head-optimization.js` owns one read-only proposal call and
  one fixed Host evaluation call.
- `host_contract.json` is the machine-readable scope and measurement contract.
- `host_evaluator.py` is the only compilation, oracle, path-hit, and timing
  authority. It requires `/usr/bin/python3 -I -B` inside a KerSor
  `command-v1` read-only/network-denied sandbox and writes only its private
  ephemeral `TMPDIR`.
- `fixtures/workflow_host_smoke.mjs` compiles and executes the workflow through
  the real Workflow Host with a fake broker and fake deterministic evaluator.
  It starts no Agent and performs no benchmark.

The current Workflow Host can express this safely for the Codex runtime through
`evaluate(command-v1)`. KerSor's DSH static adapter currently rejects
`evaluate()`, so this workflow is Codex-only until DSH exposes the same
journaled primitive with exact argv, read-only filesystem, denied network,
private temporary writes, bounded output, timeout, and `stdout_json` evidence.
Do not replace it with an agent-owned shell harness.

## Required dispatch inputs

`kernel_path` must be the absolute, direct path to the canonical shader.
`model_path` is also a required catalog argument and must be absolute. When it
is absent or relative, the workflow
returns `needs_model_path` before calling an agent. The directory must be an
absolute, non-symlink local Qwen3.5 SafeTensors bundle containing direct regular
`config.json`, `tokenizer.json`, and at least one `*.safetensors` shard.

The model directory is external Host input. It is read during the real
teacher-forced and generation gates and is never copied into the repository or
made writable. Network access remains denied, and Cargo always runs
`--offline --locked` against the adjacent `.apxinf-toolchains` cache.

## Admission protocol

Correctness is fail-closed and ordered:

1. `cargo test --offline --locked -p apxinf-metal` (including adversarial Metal
   tests).
2. Qwen3.5 library tests with `accelerate,metal-w8`.
3. A real-checkpoint 128-step teacher gate requiring 128/128 native-F32 rerank
   matches; its production sub-gate also exercises the current prefill-first
   token and decode path for ten tokens.
4. Exact equality of a no-EOS-stop 100-token baseline/candidate trajectory.
5. Execution-path evidence: the release binary contains the exact candidate
   shader bytes, the positive generation receipt reports
   `build.metal_w8_lm_head=true`, and the same candidate binary without the flag
   reports `false` while producing the same first native-F32 token.

Only then does formal performance begin. The primary metric is 100-token
`generation_tps`. The complete predeclared order is:

```text
ABBA, BAAB, ABBA, BAAB, ABBA, BAAB
```

All 24 processes must reproduce the accepted trajectory and report zero child
swaps. Before each block, one quiet-host gate requires one-minute load no higher
than 0.5 per logical CPU and no other process at or above 25% CPU. System swap
must not grow. A contaminated or incomplete block is retained verbatim and the
receipt returns `replacement_required`; no sample is deleted, relabelled, or
promoted. Acceptance requires at least 1.10x median generation throughput and a
candidate median win in all six blocks. Secondary non-regression guardrails are
TTFT at most 1.05x baseline and median peak RSS growth at most 64 MiB.

This is a single-proposal, reference-only workflow rather than an evolutionary
search loop. If host noise requires a replacement schedule, the result retains
`retry_candidate_source`, its SHA-256, and its strategy id. Supplying those
three values on the next invocation skips the Agent and remeasures exactly the
same bytes. Measurement feedback is never used to generate a different shader.

The Host additionally hashes every direct model file at evaluation start and
end, freezes source manifests at start/after-gates/end, rechecks Cargo and Rustc
binary hashes at the end, and measures system swap across the complete formal
schedule. The Workflow accepts only a receipt whose five gates, two builds, six
blocks, 24 samples, custody identities, and command-v1 enforcement all match the
pinned contract and evaluator hashes.

## Catalog use

Catalog roots are frozen into a session at setup. The existing historical
session `.kersor/20260824-102135` points at `AKW-kersor-runtime`, so its recorded
`feasible_count=0` must not be hand-edited. Create a fresh session with
`KERSOR_WORKFLOW_DIR` set to this repository's `kersor/workflows` directory
before the KerSor setup command is minted:

```text
/Users/haiyan-mini/Agent4Kernel/ApxInf/kersor/workflows
```

With backend `metal`, language `metal`, and integration pattern exactly
`apxinf_qwen35_metal_w8_head`, the official 3.0.1 catalog generator projects one
eligible workflow named `apxinf-metal-w8-head-optimization`.

For a no-Agent local check, run the repository test module with bytecode writes
disabled:

```bash
PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 -B -m unittest \
  tests.python.test_kersor_apx_metal_workflow
```

Those tests invoke the installed official catalog generator, validate the Host
contract and reducers, verify isolated shader-only snapshots, and exercise the
Workflow Host fixture. They do not call a real agent, load the model, compile
the production workspace, or run the formal 24-sample schedule.

## Evidence status

This authoring change establishes a validated catalog/workflow/evaluator
design. It contains no measured candidate speedup and makes no performance
claim. A future real run is authoritative only if the final Host receipt has
`accepted=true`, all five correctness gates passed, execution-path evidence
passed, exactly 24 preserved samples, six same-direction block wins, and a
measured speedup of at least 1.10x.
