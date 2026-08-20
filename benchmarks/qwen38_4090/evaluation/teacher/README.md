# Teacher-only cohort execution

`orchestrate_cohort.py` is the official single-GPU lifecycle boundary. It does
not accept student-generated aggregate JSON. For each plan entry it:

1. resolves a full 40-character commit in the named local Git repository;
2. creates a detached clean worktree at exactly that commit;
3. runs argv-form build and service commands without a shell;
4. waits for the configured health endpoint;
5. invokes the separately pinned teacher evaluator with a stable run ID;
6. stops the entire service process group before starting the next entry;
7. invokes the cohort scorer once with exactly one vLLM control; and
8. hashes the contract, evaluator, scorer, submissions, logs, snapshot, Git
   trees, and GPU-environment record into `provenance.json`.

The plan itself remains teacher-only because it contains hidden dataset paths,
model locations, and service commands. Fetch each PR commit into the local
repository before starting the round; the orchestrator intentionally performs
no network fetch and never resolves a mutable branch name.

## Minimal plan shape

```json
{
  "schema": "apxinf.qwen38_27b.cohort_plan.v1",
  "round_id": "midterm-2026-v1",
  "profile": "midterm_leaderboard",
  "entries": [
    {
      "name": "vllm-control",
      "role": "control",
      "repository": "/absolute/path/to/local/ApxInf",
      "checkout_revision": "FULL_40_CHARACTER_COMMIT_SHA",
      "build_argv": ["/usr/bin/true"],
      "serve_argv": ["/absolute/path/to/start-vllm"],
      "health_url": "http://127.0.0.1:8000/health",
      "startup_timeout_s": 300,
      "runner_argv": [
        "{python}",
        "{evaluator_root}/run_evaluation.py",
        "--dataset", "/teacher/public-v1",
        "--hidden-dataset", "/teacher/hidden-v1",
        "--model-dir", "/teacher/model",
        "--base-url", "http://127.0.0.1:8000",
        "--api-mode", "vllm-completions",
        "--served-model-name", "qwen3.8-27b-awq-int4",
        "--implementation-name", "vllm-control",
        "--implementation-revision", "vllm-version-and-image-digest",
        "--backend", "vllm",
        "--profile", "midterm_leaderboard",
        "--trajectory-reference", "/teacher/reference.json"
      ]
    },
    {
      "name": "pr-001",
      "role": "candidate",
      "repository": "/absolute/path/to/local/ApxInf",
      "checkout_revision": "FULL_40_CHARACTER_PR_COMMIT_SHA",
      "build_argv": ["cargo", "build", "--release", "--features", "cuda"],
      "serve_argv": ["{checkout}/target/release/apxinf", "serve", "--model", "/teacher/model"],
      "health_url": "http://127.0.0.1:8001/health",
      "runner_argv": ["{python}", "{evaluator_root}/run_evaluation.py", "... frozen teacher arguments ..."]
    }
  ]
}
```

Only these placeholders are supported: `{checkout}`, `{entry_artifacts}`,
`{round_artifacts}`, `{evaluator_root}`, and `{python}`. The evaluator appends
its own `--output-dir` and stable `--run-id`; plans must not supply either.

Validate commit availability and plan shape without building or using the GPU:

```bash
python teacher/orchestrate_cohort.py \
  --plan /teacher/cohort-plan.json \
  --evaluator-root /teacher/frozen-evaluator \
  --artifacts-root /teacher/rounds \
  --validate-only
```

The GPU worker used for official grading must still be isolated from untrusted
networks and credentials. A clean worktree controls source provenance; it is
not a security sandbox for arbitrary PR build code.

## Independent multimodal capability run

The image-input overlay is deliberately not folded into the v1 cohort score.
Before course release, create a random teacher seed file with mode `0600`, then
freeze eight hidden image cases:

```bash
python teacher/generate_hidden_multimodal_cases.py \
  --seed-file /teacher/secrets/multimodal-seed \
  --output-dir /teacher/release-v1/multimodal-hidden
```

For every cohort entry, run `run_multimodal.py` first on the public suite and
then on the teacher-only suite. `score_multimodal.py` assigns
`declared-unsupported`, `multimodal-public-pass`, or `multimodal-ready`; it
always emits `leaderboard_points=0`. Archive both reports and their manifest
hashes beside the normal cohort evidence. Never copy the hidden seed, PNGs,
prompts, answers, manifest, or report into the student release.
