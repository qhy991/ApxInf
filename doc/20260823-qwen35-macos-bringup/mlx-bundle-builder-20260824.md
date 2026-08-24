# Deterministic offline MLX bundle builder (2026-08-24)

## Outcome

`scripts/build_mlx_bundle.py` defines the reproducible HF-local-checkpoint to
MLX-bundle boundary for the macOS workflow. It accepts one explicit local
source directory, one exclusive absolute output directory, and exactly one of
these modes:

| Mode | Meaning | Admission claim |
|---|---|---|
| `mixed-bf16` | Preserve the Qwen3.5 text checkpoint's retained BF16/F32 tensor schema | Parity candidate; still requires the generation oracle |
| `affine-w8-g64` | MLX affine 8-bit, group size 64 | Explicit speed/quality tier, **not** a parity claim |
| `affine-w4-g64` | MLX affine 4-bit, group size 64 | Experimental speed-first tier, **not** a parity claim |

The optional preset
`qwen35-0.8b-affine-w8-g64-gdn-outproj-parity-v1` is a narrower contract over
`affine-w8-g64`. It is certified only for the frozen raw `Hello` 128-step
teacher trajectory described below. That manual full-prompt scope proved too
narrow for the production generator and is now legacy auxiliary evidence. The
production preset is
`qwen35-0.8b-affine-w8-g64-gdn-outproj-async-chat-parity-v2`; it is admitted
only by the exact canonical 13-token chat prompt under MLX-LM's production
`generate_step` path with the explicit `mlx-generate-step-argmax-v1` sampler.
Neither preset changes the generic W8 mode or turns generic quantization into a
parity claim.

The builder does not use `mlx_lm.convert` and does not apply a blanket dtype
cast. It loads the source with `mlx_lm.utils.load`; quantized modes call only
`mlx_lm.utils.quantize_model` with the exact mode parameters; all modes publish
through `mlx_lm.utils.save`. The accepted runtime closure is pinned to MLX and
MLX-Metal 0.32.1, MLX-LM 0.31.3, Transformers 5.15.1, safetensors 0.8.0,
tokenizers 0.22.2, huggingface-hub 1.28.0, and NumPy 2.5.2.

## Security and reproducibility boundary

- Source and destination must be absolute, non-overlapping paths. The source
  and destination parent must be real, current-UID directories without
  symlink components.
- The source is a flat Qwen3.5 bundle. Missing config/tokenizer/template/index
  files, unexpected files or directories, remote-code configuration, symlinks,
  hardlinks, invalid safetensors headers, and index/header disagreement fail
  closed.
- HF and Transformers offline flags are forced, ambient proxies and HF tokens
  are removed during MLX execution, private cache paths are used, and Python
  socket connection entrypoints are blocked.
- MLX may serialize tokenizer files differently. After MLX saves into the
  private staging area, `tokenizer.json`, `tokenizer_config.json`, and
  `chat_template.jinja` are recreated from the source bytes and verified for
  exact equality. This keeps ApxInf prompt construction byte-stable.
- Every source and output file receives an exact byte size and SHA-256 in one
  canonical, single-line JSON receipt. The receipt also records the exact
  Python, MLX, MLX-LM, and Transformers versions.
- Publication is a same-filesystem atomic rename with a kernel-level
  no-replace flag (`RENAME_EXCL` on macOS, `RENAME_NOREPLACE` on Linux). A
  concurrent or pre-existing output is never replaced.
- `--verify-only` validates and hashes an existing bundle in place without
  loading MLX or copying weights. It is the safe path for reusing a previously
  constructed large artifact.
- Mixed output must map every one of the 320 source language tensors to its
  exact MLX name, dtype, and shape. The sole permitted layout adaptation is the
  pinned Qwen3.5 depthwise-convolution transform from
  `[channels, 1, kernel]` to `[channels, kernel, 1]`.
- Quantized output is also checked tensor by tensor. Every eligible two-
  dimensional BF16 weight with a group-64 input dimension must become the
  exact U32 packed shape plus BF16 `scales` and `biases`; every ineligible
  parameter must retain its canonical name, dtype, and shape. Missing,
  partially quantized, over-quantized, or unexpected tensors fail closed.
- Each hybrid preset is additionally bound to revision
  `2fc06364715b967f1860aea9cf38778875588b17`, the exact source config SHA-256,
  the canonical 320-tensor language schema SHA-256, and the canonical policy
  hash. Its exact retained BF16 paths and complete logical-weight/byte ledger
  are embedded in `config.json` and repeated in the build receipt. The builder
  recomputes every value during build and `--verify-only`; preset, policy,
  source, output-schema, or ledger drift fails closed.
- Before a v2 payload can be saved or published, the builder calls
  `mlx_lm.generate.generate_step` twice on the quantized in-memory model with
  the canonical prompt, `max_tokens=128`, and an explicit sampler equivalent
  to `mx.argmax(logprobs, axis=-1)`. Both complete trajectories must equal the
  frozen BF16 teacher, both first-100 hashes must match, and the repeats must be
  identical. A mismatch fails before `mlx_lm.utils.save`, so no output bundle
  is published. `--verify-only` remains a static, no-MLX-load check of the
  already hash- and schema-bound artifact; it does not silently rerun inference.

The filesystem checks protect against accidental path substitution and
cooperating-process races. As elsewhere in this local workflow, the trust
boundary is the current UID; they do not claim to resist a malicious process
running as that same user.

## Real Qwen3.5 source preflight

The production source was inspected read-only; no additional 1.7 GB bundle was
created.

```text
source: /Users/haiyan-mini/Agent4Kernel/.apxinf-models/Qwen3.5-0.8B-2fc063647-staged
model_type: qwen3_5
source artifacts: 6
source manifest SHA-256: 436821ae50e981b9176784ac6ff9548742a865d60d726c58d3bfa9f76d86b500
safetensors tensors: 488 (452 BF16, 36 F32)
retained language tensors expected in mixed output: 320
runtime: CPython 3.14.3, MLX 0.32.1, MLX-LM 0.31.3, Transformers 5.15.1
```

The six source file hashes emitted by the preflight match the pinned HF source
lock, including the 1,746,942,600-byte weight shard with SHA-256
`04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696`.

The earlier exploratory `.apxinf/models/Qwen3.5-0.8B-mlx-mixed` directory was
also checked with `--verify-only`. It was correctly rejected because its
MLX-reserialized `tokenizer.json` is not byte-identical to the source. That
directory remains useful as benchmark evidence, but it is not a certified
output of this stricter builder and was not modified.

As a separate read-only schema check, the retained tensor trees in all three
exploratory bundles matched the stricter invariants: mixed contained exactly
320 retained tensors; W8 and W4 each contained 694 output tensors produced by
exactly 187 affine group-64 transformations. This validates the schema rules
against real MLX output without constructing another large bundle. It does not
override the byte-identical tokenizer requirement or certify those earlier
directories.

## Legacy v1 raw-prompt selective-W8 preset

The checked policy and evidence live in
`doc/20260823-qwen35-macos-bringup/qwen35-0.8b-mlx-w8-g64-parity-policy-evidence-v1.json`.
Only its canonical `policy` object is hashed, so adding immutable build and
parity evidence does not change the admitted policy. The policy SHA-256 is
`560f9b3df77a650603d91ff2ed60c0a56761f2d3fc408296be0a87a2f13e65cf`.

The preset keeps only these GDN output projections in BF16:

```text
language_model.model.layers.12.linear_attn.out_proj
language_model.model.layers.13.linear_attn.out_proj
language_model.model.layers.14.linear_attn.out_proj
language_model.model.layers.17.linear_attn.out_proj
language_model.model.layers.18.linear_attn.out_proj
language_model.model.layers.20.linear_attn.out_proj
language_model.model.layers.21.linear_attn.out_proj
```

The remaining 180 eligible modules are affine W8 group-64. The exact ledger is
737,214,464 W8 logical weights and 14,680,064 retained BF16 logical weights;
the estimated parameter payload is 813,652,672 bytes across 680 tensors.

The certified bundle was published with atomic no-replace semantics at
`.apxinf/models/Qwen3.5-0.8B-mlx-w8-g64-gdn-outproj-parity-v1` and then checked
with `--verify-only`. Its manifest SHA-256 is
`8685f2ed5246379287761eebd026c711ed23ccdbe68882f0e96a48cdad9a49f1`.
Against the frozen mixed-BF16 raw-`Hello` teacher trajectory it matched all
128/128 next-token decisions twice. Its canonical token-ID hashes are:

```text
teacher 128: 35c9912981f47d93414d42fc8bd627e4e500a64535193790d488f913f82b2606
free-run 10: 6dc93669ffde4e6905f9818ef89a2b82f9f155bbe3462fd5a2c0c3e7866f8430
free-run 100: bb7789354b4936d0c83286cdf7ca40e71874d9a3e543aa1c138667fa64c0d5c7
```

Performance samples taken during diagnosis are deliberately not part of the
admission contract.

This v1 result is not a production canonical-chat claim. When the same bundle
was exercised with MLX-LM's native `generate_step` prompt prefill and
asynchronous decode path on the canonical 13-token chat prompt, it first
diverged at step 9 and matched only 14/128 BF16 tokens. The old evidence file is
retained unchanged, but v1 is superseded for production use by v2.

## Certified v2 canonical-chat async preset

The v2 policy and evidence live in
`doc/20260823-qwen35-macos-bringup/qwen35-0.8b-mlx-w8-g64-async-chat-parity-policy-evidence-v2.json`.
Its canonical policy SHA-256 is
`64a2ba1741fd5a76a7e72580ce9188d1554e1488ce6504b20054bf42479eaf8f`.
The exact canonical prompt token IDs are:

```text
[248045,846,198,9419,248046,198,248045,74455,198,248068,271,248069,271]
```

Only three GDN output projections remain BF16:

```text
language_model.model.layers.12.linear_attn.out_proj
language_model.model.layers.14.linear_attn.out_proj
language_model.model.layers.20.linear_attn.out_proj
```

The other 184 eligible modules remain affine W8 group-64. The precise ledger is
745,603,072 W8 logical weights and 6,291,456 retained BF16 logical weights,
giving an estimated 805,788,352-byte parameter payload across 688 tensors.
Thus 99.1632528545% of eligible logical weights remain W8. Compared with the
799,890,112-byte all-W8 parameter estimate, the hybrid adds 5,898,240 bytes;
the simple parameter-traffic ratio is 0.9926801622. This is an accounting model,
not a promoted throughput result or guarantee.

The real no-replace bundle is
`.apxinf/models/Qwen3.5-0.8B-mlx-w8-g64-gdn-outproj-async-chat-parity-v2`.
Its manifest SHA-256 is
`5f65527db17cd4fb7a1a46ca6ab3940a8bcc8225c620ca10d3016265d2bf8553`,
and its `model.safetensors` SHA-256 is
`c70b8b7efa75380008eaa449b1cdddf945e80e40f0b7970baa6ab445ea878bc7`.
It passed static `--verify-only`, the in-builder pre-publication gate twice,
and two independent post-publication disk-model runs with 128/128 exact tokens.
The admitted hashes are:

```text
first 10:  a3382db9d7097d3702e935cd53a1589256a8d123e614504f5bdf646cec36d200
first 100: 79e574ba6c3c81eda6f78c9d0412f2f91e6b3f6137e7cb7c146c7714d1a96578
all 128:   2042d5522ed5e768b938ac3fd9e19d3936dfc6e685fc157d41a8e8132c5b42fe
```

The legacy raw-`Hello` check is retained as an explicitly non-admission
auxiliary result. This smaller v2 keep-set reaches only 72/128 under the raw
async trajectory (identically on two runs); it is not represented as a raw
prompt compatibility claim. The production canonical-chat gate is the sole v2
quality admission contract.

## Usage

Build a new, exclusive output:

```sh
.apxinf/toolchains/mlx-lm-0.31.3/bin/python scripts/build_mlx_bundle.py \
  --source-dir /absolute/path/to/pinned-hf-bundle \
  --output-dir /absolute/path/to/new-mlx-bundle \
  --mode mixed-bf16
```

Build the exact selective-W8 preset from the frozen source:

```sh
.apxinf/toolchains/mlx-lm-0.31.3-copies/bin/python3.14 \
  scripts/build_mlx_bundle.py \
  --source-dir /absolute/path/to/Qwen3.5-0.8B-2fc063647-staged \
  --output-dir /absolute/path/to/new-selective-w8-bundle \
  --mode affine-w8-g64 \
  --preset qwen35-0.8b-affine-w8-g64-gdn-outproj-parity-v1 \
  --source-revision 2fc06364715b967f1860aea9cf38778875588b17
```

Build the production canonical-chat v2 preset into a new exclusive directory:

```sh
.apxinf/toolchains/mlx-lm-0.31.3-copies/bin/python3.14 \
  scripts/build_mlx_bundle.py \
  --source-dir /absolute/path/to/Qwen3.5-0.8B-2fc063647-staged \
  --output-dir /absolute/path/to/new-v2-selective-w8-bundle \
  --mode affine-w8-g64 \
  --preset qwen35-0.8b-affine-w8-g64-gdn-outproj-async-chat-parity-v2 \
  --source-revision 2fc06364715b967f1860aea9cf38778875588b17
```

Verify an existing output without loading or copying model weights:

```sh
.apxinf/toolchains/mlx-lm-0.31.3/bin/python scripts/build_mlx_bundle.py \
  --source-dir /absolute/path/to/pinned-hf-bundle \
  --output-dir /absolute/path/to/existing-mlx-bundle \
  --mode affine-w8-g64 \
  --verify-only
```

Verifying the hybrid bundle requires the same `--preset` and
`--source-revision` arguments. Omitting either cannot silently downgrade it to
a generic W8 artifact: the embedded hybrid manifest is rejected.

Successful stdout is exactly one canonical JSON receipt line. Failures produce
exactly one JSON error line on stderr and a non-zero exit status.

## Verification performed

```text
python3 -m unittest tests.python.test_build_mlx_bundle -v
26 tests passed

python3 -m unittest discover -s tests/python -v
245 tests discovered; 244 passed; one unrelated Kersor subprocess wall-clock
assertion failed at about 2.04 s versus its 1.5 s threshold

same discovery with that one named timing test explicitly excluded
244 tests passed

ruff check scripts/build_mlx_bundle.py tests/python/test_build_mlx_bundle.py
All checks passed

python3 -m py_compile scripts/build_mlx_bundle.py tests/python/test_build_mlx_bundle.py
passed

git diff --check -- scripts/build_mlx_bundle.py tests/python/test_build_mlx_bundle.py
passed
```

The single full-suite failure is
`test_host_stops_a_command_as_soon_as_a_stream_exceeds_its_bound` in the
untracked Kersor Metal workflow tests, outside the authorized files for this
change. It launches a child that writes a bounded stream and sleeps for two
seconds, then asserts the host returns in under 1.5 seconds. Two isolated runs
observed about 2.04 seconds. This is recorded rather than hidden or modified;
all 244 remaining tests pass, and the v2 builder suite is 26/26. It does not
change the v2 correctness admission, and no timing result is promoted.

The tests use tiny synthetic safetensors headers and a mocked pinned MLX API;
they perform no network access and create no large model copy. Coverage includes
mixed-dtype preservation, exact tokenizer restoration, W8/W4 parameters,
runtime-closure pin enforcement, offline environment isolation, socket blocking,
existing-output rejection, no-replace publication races, source mutation,
verify-only reuse, quantized packing drift, remote-code rejection, and
symlink/hardlink/unexpected-path hazards. Hybrid coverage additionally includes
successful v1 and v2 selective construction; v2's explicit argmax sampler and
two production-semantic repeats; divergence before save/publication; deleted,
added, and renamed policy/source/output layers; prompt and sampler-semantics
drift; config and revision drift; non-Qwen sources; exact ledger and policy-
manifest validation; and verify-only reuse without MLX import.
