# Qwen3.5 Stack3 formal benchmark contract v1

This harness admits exactly the Stack3 correctness summary at SHA-256
`6e4e6551336b55b7ce4131fc94f2ff2820d62bd98cf2951ee327fca488d926c0`.
That summary binds the release gate binary at SHA-256
`882441d89c820031bd61afebd2ffdf12de49817f16f73a9e6c48556bc7f55007`
and the frozen CPU-free reference receipt at SHA-256
`63c8dc6282fc617d5453d60a8825b52b0b6c293c329fa717593837bb688ddfbb`.
The four correctness receipts, binary, complete Stack3 Rust/bridge/shader source
closure, profile, source lock, and exact cache-free model closure are rehashed
before admission. The binary/source/model closure and frozen reference receipt
are rehashed again at campaign completion.

The frozen CPU-free receipt is a reference oracle, not a timed sample and not a
same-physical-run pairing. Every timed A receipt must independently reproduce
its exact 128-token trajectory. Every timed B command receives that exact
reference through `--input-receipt`; its embedded receipt record, embedded CPU
trajectory, and Stack3 trajectory must all match the frozen oracle byte-for-byte
at the receipt-field level. This preserves a pure 12-A/12-B campaign without an
untimed or hidden model run.

The fixed physical order is `ABBA, BAAB, ABBA, BAAB, ABBA, BAAB`. A is
`cpu-free`; B is `all-linear-layer-stacks-v1-free`. Formal acceptance requires
all 24 samples, 12 A and 12 B samples, all six B block medians faster, overall
median throughput speedup of at least 1.10x, and candidate median TTFT no more
than 1.10x baseline. Every sample must retain the exact 128-token oracle,
identity custody, and passing path/ledger evidence.

Each B receipt must prove six three-layer Stack3 transactions and six
full-attention Metal MLP blocks. Both prefill/final aggregate and generation
receipts must name `metal-w8-mlp-block-g64`, and the versioned generation-path
contract must bind all six stacks and all six MLP layers. Each stack has 127 successful free-run decode
calls, 127 command buffers/commits/waits, 381 compute encoders and state
commits, commit mask 7, version 127, zero failures, and a clear terminal state.
The admitted aggregate ledger is 504 buffers (444 shared and 60 private),
528,605,184 persistent MTLBuffer bytes, 12 command buffers/commits/waits, 36
compute encoders, and 49,152 bytes in each transfer direction per decode.
Intermediate host finite checks are intentionally disabled only under this
versioned contract; six final-output finite checks remain required.

The harness defaults to a dry plan. Real execution requires explicit
`--execute` and an absolute, nonexistent output directory outside the frozen
model directory. It delegates process
and publication safety to the audited all18 formal engine: five-sample quiet
admission; no non-allowlisted process above 5% CPU; zero throttled pages; stable
system swap; per-run start, online, and end custody samples; a 600-second
process-group timeout; strictly less than 6 GiB group RSS; independent 4 MiB
stdout/stderr limits; and zero child swaps. Runtime load is recorded but is not
an external-noise gate because it includes the owned workload.

Receipts and the final result are published without replacement. Any failure
may publish only `formal_accepted: false`; interrupts attempt the same
nonaccepted publication before propagating, and staging cleanup cannot replace
the primary error. The dry plan and every Stack3 formal result record direct-file
SHA-256 custody for this wrapper and the audited all18 base engine; both files
are rehashed at campaign start and end, and drift forces a nonaccepted result.
No formal performance result exists until a quiet-host
`--execute` campaign completes this entire contract.
