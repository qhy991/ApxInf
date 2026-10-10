# Dedicated native binary comparison

This comparison uses the fixed local Qwen3.5-0.8B assets listed in [assets.json](assets.json).
It compares three build selections from one source snapshot.

| Selection | Entry | Compiled scope |
|---|---|---|
| full | Existing ApxInf CLI | All model families and legacy Metal features |
| reference | Dedicated Qwen3.5 entry | General model, Metal, loader, and tokenizer features |
| minimal | Identical dedicated entry | Required Qwen3.5, Metal, loader, and tokenizer features |

Only reference and minimal share the dedicated timing boundaries and asset checks.
The full CLI supplies output-token parity and prompt-length checks.
Its existing result format does not expose prompt token IDs.
It does not supply a controlled runtime comparison with the dedicated entry.

## Reproduce

Use macOS on Apple Silicon, a compatible Rust toolchain, and Python 3.10 or later.
Set `MODEL` to the directory containing the six files in `assets.json`.
Keep their exact filenames and bytes.
The general CLI needs the shard index. The dedicated executable needs the other five files.
The manifest records existing local hashes. It does not check the remote revision again or download weights.

Set `SNAPSHOT` and `OUTPUT` to new absolute directory paths.
Do not reuse a build directory for a clean-build measurement.
Select an SDK compatible with the installed compiler.
The historical measurement used MacOSX15.4.sdk.

```sh
MODEL=/absolute/path/to/fixed-model-assets
SNAPSHOT=/absolute/path/to/new-source-snapshot
OUTPUT=/absolute/path/to/new-experiment-output
export SDKROOT="$(xcrun --show-sdk-path)"

# Populate the local Cargo cache before the offline build phase.
cargo fetch --locked

python3 benchmarks/specialization/v2/compare_native.py prepare \
  --source "$PWD" --snapshot "$SNAPSHOT" --output "$OUTPUT" --model "$MODEL"

python3 benchmarks/specialization/v2/compare_native.py build \
  --source "$PWD" --snapshot "$SNAPSHOT" --output "$OUTPUT"

python3 benchmarks/specialization/v2/compare_native.py runtime \
  --source "$PWD" --snapshot "$SNAPSHOT" --output "$OUTPUT" \
  --metal-lock "$HOME/.cache/enginetailor/metal-measure.lock" --lock-wait-seconds 60

python3 benchmarks/specialization/v2/compare_native.py summarize \
  --source "$PWD" --snapshot "$SNAPSHOT" --output "$OUTPUT"
```

The prepare phase checks the complete asset manifest and saves a source snapshot.
Later phases use the model path recorded by prepare.
An optional `--model` in those phases must select the same directory.
The build phase uses six independent target directories and the locked dependencies.
It measures clean, unchanged, and comment-only rebuilds separately.

The runtime phase holds one advisory file lock around all measured processes.
The default lock is `$HOME/.cache/enginetailor/metal-measure.lock`.
`--metal-lock` selects another shared path. The script creates a missing parent directory and lock file.
Configure every cooperating Metal test to use the same path.
An advisory lock cannot exclude programs that ignore it.
Do not remove or replace the lock file while measurements run.
Lock waiting occurs before process timing starts. A timeout starts no measured process.
The comparison does not require serving scripts or a previous experiment directory.

## Evidence

The experiment contract is `contract.json`. The fixed corpus is `cases.json`.
`inputs.json` binds assets, source files, configuration, and measurement scripts by SHA-256.
`builds.json` contains build commands, durations, binary identities, dependency records, native objects, and symbols.
`runtime.json` contains process state, raw output identities, parsed records, and request labels.
`summary.json` checks the sample grid, source exclusions, identities, and corpus equivalence.

The two dedicated selections each run twice, in reference/minimal/minimal/reference order.
Each process runs two warmup rounds and three measured rounds over the complete corpus.
The corpus includes short Chinese and English prompts, arithmetic, and a longer English response.
Every request resets model state.
One separate plain-output request checks actual streaming text against complete decoding.

The runtime records swap and thermal state around every process.
Active swapping makes attribution of latency changes inconclusive.
These measurements do not establish formal performance qualification.
The source snapshot excludes optional CUDA vendor files because this experiment does not build CUDA.
It includes declared crate tests because Cargo checks their paths during binary builds.

## Historical results and current harness

[RESULTS.md](RESULTS.md) describes the historical r3 experiment.
The public reproduction harness adds explicit model selection, a repository asset manifest, and internal lock coordination.
Its bytes differ from the historical r3 harness. Its contract also differs from the historical contract.
We did not repeat the full performance experiment with these reproduction changes.
Do not use the historical script digest to identify the current script.
Keep historical evidence and new experiment outputs separate.

Earlier setup attempts did not contribute to the r3 measurements.
One attempt omitted declared test files from its snapshot.
Another required configuration EOS 248044 to equal tokenizer EOS 248046 and stopped before generation.
The final executable preserves tokenizer EOS 248046, matching the general CLI.

Run the portable-input and lock tests without building or starting a model:

```sh
python3 -m unittest discover -s benchmarks/specialization/v2 -p 'test_*.py'
```
