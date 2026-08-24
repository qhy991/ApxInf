# Qwen3.5 all18 formal benchmark contract v1

The archived all18 correctness summary with SHA-256
`3695081d1a5fc1bd54fc041a91596b46fbf0d7d7d2d4befd9bce0efc1b1b9e0d`
is historical frozen evidence. Its source custody binds `general.rs` at
`fb4993e5b27e4842837e6aefe8be985b2f96bd5f44f9911e2458403b1869ab27`.
Later stack3 work changed that source, so the harness correctly refuses to use
the old summary for a formal result. Do not overwrite or relabel the archived
summary or its four receipts. A formal run now requires four new correctness
receipts and a new no-replace summary for the new binary/source/model identity.

The harness defaults to a dry 24-run plan. Real execution requires explicit
`--execute` and an absolute new output directory. The harness completes all
expensive live-custody hashing before its final admission gate. That gate
samples the host five times and rejects any throttled pages, swap movement,
excess load, or non-allowlisted process above 5% CPU before creating output or
starting a child. Its final swap value is the immutable campaign baseline.

Each admitted run is process-group supervised with a 600-second timeout, a
strict less-than-6-GiB group RSS limit, independent 4-MiB stdout/stderr limits,
and zero child swaps. Every run also requires a start sample, at least one
online sample at intervals no greater than one second, and an end sample.
Throughout those samples, non-owned process CPU above 5%, nonzero throttling,
or any swap drift fails the campaign and prevents later runs. Global load is
recorded during a run but is an admission-only gate: it includes the owned
workload and has a one-minute lag, so it cannot prove external contamination.

The fixed order is `ABBA, BAAB, ABBA, BAAB, ABBA, BAAB`. Formal acceptance
requires exact 128-token trajectories, unchanged path/ledger/custody evidence,
all six candidate block medians faster, overall median throughput speedup of at
least 1.10x, and candidate median TTFT no more than 1.10x baseline. Final files
are atomically published without replacement; any mid-campaign failure can
publish only an explicit `formal_accepted: false` result. Interrupts attempt the
same nonaccepted publication before propagating, and cleanup errors cannot
replace the primary failure.
