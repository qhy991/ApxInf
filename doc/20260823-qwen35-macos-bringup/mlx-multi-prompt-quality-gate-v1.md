# Qwen3.5-0.8B MLX multi-prompt quality gate v1

## Status and purpose

This is an offline admission contract for uniform, selective mixed, and the
certified hybrid W8/BF16 candidate. It prevents one exact `Hello` or
canonical-chat trajectory from being relabelled as broad parity. It does not
replace or modify the current builder, policy, planner, runner, or published
evidence.

The trusted producer has now run the full fixed suite against the local
certified BF16 reference and uniform affine W4 group-64 candidate. Both lanes
were deterministic across their required repeats, and the producer published
the complete custody envelope as
[`qwen35-w4-multi-prompt-quality-v1.json`](./qwen35-w4-multi-prompt-quality-v1.json).
The structurally valid result is `failed_comparison` (`accepted=false`, process
exit `1`), so it does **not** admit the W4 bundle and does not establish general
parity.

| Prompt ID | Exact prefix | Position match | Exact |
|---|---:|---:|---|
| `english-explanation` | 2 / 64 | 0.0625 | no |
| `chinese-explanation` | 1 / 64 | 0.03125 | no |
| `python-code` | 0 / 32 | 0.0 | no |
| `math-structured-json` | 10 / 32 | 0.3125 | no |

The receipt reports `exact fixed-suite mismatch` for all four prompts. This is
real negative evidence: BF16 and W4 were executed successfully under the frozen
offline contract, while W4 failed the requested fixed-suite exact-parity gate.

### Certified hybrid W8/BF16 v2 result

The producer also ran the existing certified canonical-chat v2 bundle under
the new, explicit `hybrid-w8-bf16-g64` profile. This profile admits only exact
global affine W8 group-64 `quantization` and `quantization_config` objects and
the frozen `apxinf_hybrid_preset`: preset name, source revision, policy hash,
complete weight ledger, and exactly the layer 12/14/20 GDN `out_proj` BF16
retentions must match. A selective-policy manifest or any additional hybrid
manifest field is rejected before MLX loads.

Both BF16 and hybrid lanes completed two identical runs for each prompt. The
trusted no-replace envelope is
[`qwen35-hybrid-w8-bf16-g64-multi-prompt-quality-v1.json`](./qwen35-hybrid-w8-bf16-g64-multi-prompt-quality-v1.json).
The requested `fixed-suite-threshold-match` result is a structurally valid
`failed_comparison` (`accepted=false`, process exit `1`):

| Prompt ID | Exact prefix | Position match | Exact | Threshold |
|---|---:|---:|---|---|
| `english-explanation` | 44 / 64 | 0.75 | no | pass |
| `chinese-explanation` | 46 / 64 | 0.71875 | no | **fail** |
| `python-code` | 32 / 32 | 1.0 | yes | pass |
| `math-structured-json` | 32 / 32 | 1.0 | yes | pass |

The single failing Chinese ratio is below the frozen 0.75 floor, so the bundle
is not admitted even though its earlier one-prompt canonical-chat trajectory
was exact. The envelope content self-hash is
`16d7fd7ff43c56ba9d8992f39efe32caca2d1b2790499c78e7ecd5e60a460b0c`;
the serialized file SHA-256 is
`a82cb9b1d8372d67d4d36952961e1bbc0a03e7ba9a41fcb9cd60a644521934a7`.
Its start/end runtime and both full bundle snapshots are identical. The
quality-run child recorded zero swaps and a 2,593,128,448-byte maximum resident
set; these are resource observations, not throughput evidence or a performance
promotion.

## Frozen prompt suite

The versioned contract is
`configs/qwen35-0.8b-mlx-multi-prompt-quality-v1.json`. It binds the immutable
Qwen revision and tokenizer SHA-256, and stores raw token IDs rather than
retokenizing text at admission time. The IDs were prepared for these four
fixed prompts:

| Prompt ID | Domain | Prompt | BF16/candidate steps |
|---|---|---|---:|
| `english-explanation` | English | Explain why the sky is blue in three concise sentences. | 64 |
| `chinese-explanation` | Chinese | 请用中文解释为什么天空是蓝色的，并给出三点理由。 | 64 |
| `python-code` | Code | Write a Python function that returns the first n Fibonacci numbers. Return code only. | 32 |
| `math-structured-json` | Math/structured | Solve `2x^2 - 5x - 3 = 0`. Return JSON with keys roots and verification. | 32 |

Each raw-token prompt includes the same Qwen user/assistant chat control tokens
used by the production canonical-chat path. The validator requires all four
records in this exact order; a one-prompt receipt is structurally invalid.

## Production generation contract

Both BF16 references and candidates must declare and use the same frozen
execution semantics:

- API: `mlx_lm.generate.generate_step`
- strategy: `mlx-generate-step-argmax-v1`
- sampler: explicit `mx.argmax(logprobs,axis=-1)`
- asynchronous MLX evaluation with native prompt chunking
- fixed-step generation with no EOS stop
- exactly two runs per prompt

For every prompt, the two BF16 runs must be byte-for-byte identical and the
two candidate runs must also be identical. Token arrays, lengths, hashes,
prompt IDs, raw prompt tokens, precision labels, and contract hash are all
validated before comparison.

## Admission modes and claim boundary

`fixed-suite-exact-parity` requires every candidate token on every prompt to
equal its BF16 reference. The name is intentionally scoped: even a passing
receipt means exact parity only for this frozen four-prompt suite.

`fixed-suite-threshold-match` is a weaker regression tier. Every prompt must
have an exact prefix of at least 8 tokens and position-wise agreement of at
least 0.75 across its complete 32/64-token trajectory. All four prompts must
pass. This metric is explicit and reproducible, but it is not a semantic
quality evaluation and it is never called parity.

The contract and receipt always set `claims_general_parity=false`. Claims such
as `general-parity`, `universal-parity`, `model-parity`, and
`all-prompts-parity` are forbidden. A single canonical prompt cannot satisfy
either mode.

## Trusted local evidence producer

`scripts/run_mlx_multi_prompt_quality.py` turns the frozen contract into a
local evidence run. Its four path inputs must be canonical absolute paths:
the contract must be a direct regular file; reference and candidate must be
distinct direct directories; the output parent must already exist; and the
output itself must not exist. Publication is atomic and no-replace.

Both bundles must use the builder's controlled flat layout: `README.md`,
`chat_template.jinja`, `config.json`, `model.safetensors.index.json`,
`tokenizer.json`, `tokenizer_config.json`, plus either one
`model.safetensors` or a complete `model-00001-of-000NN.safetensors` shard
sequence. Nested paths, symlinks, hard links, custom Python, missing shards,
unknown files, index drift, a tokenizer hash outside the frozen contract, and
Qwen/config/precision-profile drift fail before MLX is imported. Every file is
stream-hashed before generation and again afterwards.

Supported candidate labels are `w4-g64`, `w8-g64`,
`mixed-w4-w8-bf16`, and `hybrid-w8-bf16-g64`; `bf16` remains available for
controlled comparisons. The hybrid label is deliberately distinct from both
uniform W8 and the selective W4/W8/BF16 manifest format, so neither weaker
config gate can be used to relabel this certified preset.

The production process requires CPython 3.14.3 and exactly these eight
packages: `mlx==0.32.1`, `mlx-metal==0.32.1`, `mlx-lm==0.31.3`,
`transformers==5.15.1`, `safetensors==0.8.0`, `tokenizers==0.22.2`,
`huggingface-hub==1.28.0`, and `numpy==2.5.2`. Their identities are checked
before MLX import and after both lanes. The producer forces Hugging Face's
offline flags, removes ambient tokens and proxy variables, accepts no Python
files in a bundle, rejects remote/custom-code config keys recursively, and
loads each local bundle with `trust_remote_code=false`.

The BF16 bundle is loaded first and released before the candidate is loaded.
For each bundle, all four raw-token prompts run twice through
`mlx_lm.generate.generate_step`. The supplied sampler calls
`mx.argmax(logprobs, axis=-1)` explicitly; each produced token is evaluated,
and EOS never shortens a frozen 32/64-step trajectory. The producer then
rechecks both bundles, the contract, its own source, the validator source, and
the eight-package runtime lock before constructing evidence and invoking the
existing independent validator.

Run it from the pinned interpreter after replacing the three `/ABSOLUTE/...`
paths with direct local paths. The output parent must already exist and the
output filename must still be absent:

```bash
/Users/haiyan-mini/Agent4Kernel/ApxInf/.apxinf/toolchains/mlx-lm-0.31.3-copies/bin/python3.14 -I -B \
  /Users/haiyan-mini/Agent4Kernel/ApxInf/scripts/run_mlx_multi_prompt_quality.py \
  --contract /Users/haiyan-mini/Agent4Kernel/ApxInf/configs/qwen35-0.8b-mlx-multi-prompt-quality-v1.json \
  --reference-bundle /ABSOLUTE/DIRECT/Qwen3.5-0.8B-mlx-bf16 \
  --candidate-bundle /ABSOLUTE/DIRECT/Qwen3.5-0.8B-mlx-mixed-w4-w8-bf16 \
  --candidate-id qwen35-0.8b-mixed-w4-w8-bf16-candidate-v1 \
  --precision-profile mixed-w4-w8-bf16 \
  --requested-claim fixed-suite-exact-parity \
  --output /ABSOLUTE/EXISTING-DIRECTORY/qwen35-multi-prompt-quality-v1.json
```

The command emits one summary line. Exit `0` means an accepted receipt was
published. Exit `1` means both lanes were deterministic and fully valid but
the candidate missed its requested fixed-suite comparison; that explicit
`failed_comparison` envelope is also published for diagnosis. Exit `2` means
the run, custody, runtime, or validator contract was invalid, and no output is
published. The producer never promotes either outcome to general parity.

## Standalone offline validator

Validate the published producer envelope without importing MLX or opening model
weights:

```bash
/usr/bin/python3 -I -B \
  /Users/haiyan-mini/Agent4Kernel/ApxInf/scripts/validate_mlx_multi_prompt_quality.py \
  --contract /Users/haiyan-mini/Agent4Kernel/ApxInf/configs/qwen35-0.8b-mlx-multi-prompt-quality-v1.json \
  --evidence /Users/haiyan-mini/Agent4Kernel/ApxInf/doc/20260823-qwen35-macos-bringup/qwen35-w4-multi-prompt-quality-v1.json
```

For the checked-in W4 envelope this emits exactly its recomputed inner receipt
and exits `1`: `accepted=false`, `precision_profile="w4-g64"`, four prompt
records, and the four-prompt exact-mismatch problem above. Exit `1` is the
expected valid comparison failure, not a validator error. Exit status is `0`
for admission and `2` for malformed, unbound, custody-drifted,
non-deterministic, or semantics-drifted input.

The CLI also remains compatible with a bare
`apxinf-mlx-multi-prompt-quality-evidence-v1` object. When given the producer's
`apxinf-mlx-multi-prompt-quality-run-v1` envelope, it first verifies the exact
outer schema, content self-hash, frozen policy, contract/runtime/source/bundle
custody and bundle manifests. It then independently recomputes the inner
receipt, requires an exact embedded-receipt match, binds candidate precision to
bundle custody, and checks that `status` agrees with the recomputed result.

The validator is a deterministic structural and token-comparison boundary. It
cannot authenticate fabricated input by itself. A production integration must
have a trusted Host run BF16 and candidate `generate_step` twice per prompt,
retain those raw trajectories, and bind that evidence to the candidate bundle
before this receipt can authorize any fixed-suite claim.
