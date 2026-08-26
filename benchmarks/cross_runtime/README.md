# Cross-runtime benchmark drivers

## ApxInf versus OmniInfer resident HTTP diagnostic

`apxinf_vs_omniinfer_http_driver_v1.py` runs a strict paired diagnostic between
the already-resident ApxInf benchmark adapter (arm `B`) and an already-resident
OmniInfer gateway (arm `G`). It is always `NON_FORMAL`: even a quiet, stable run
cannot be used as frozen evidence or as an engine-ranking claim.

The ApxInf server must be started with
`--expected-generation-requests 68`. The driver uses its one ApxInf connection
for health, state, every checked reset, every generation, and final state. It
opens one persistent OmniInfer generation connection and one explicitly named
clear connection. Four fixed warmup pairs precede 16 balanced four-pair
`BG`/`GB` blocks. There is no retry, resampling, or outlier removal.

Run zero-network checks and unit tests first:

```sh
python3 benchmarks/cross_runtime/apxinf_vs_omniinfer_http_driver_v1.py self-test
python3 -m unittest discover -s tests/python \
  -p 'test_apxinf_vs_omniinfer_http_driver_v1.py' -v
```

For OmniInfer's gateway-owned cache-clear endpoint:

```sh
python3 benchmarks/cross_runtime/apxinf_vs_omniinfer_http_driver_v1.py run \
  --apx-endpoint http://127.0.0.1:9100 \
  --omni-endpoint http://127.0.0.1:9000 \
  --omni-clear-endpoint http://127.0.0.1:9000 \
  --omni-clear-path /omni/cache/clear \
  --omni-clear-contract omni-gateway \
  --omni-clear-body empty-object \
  --quiet-host-status not-evaluated \
  --output /absolute/absent/path/apxinf-vs-omniinfer-raw-v1.json
```

If cache clear must address the resident llama.cpp backend directly, use its
explicit endpoint and exact slot-erase contract:

```sh
  --omni-clear-endpoint http://127.0.0.1:51090 \
  --omni-clear-path '/slots/0?action=erase' \
  --omni-clear-contract llama-slot-erase \
  --omni-clear-body empty
```

The output path must be absolute and absent. It is reserved with exclusive
create before any request; terminal failures are written to that same raw JSON
file when possible. Cache clear is outside the primary interval. The primary
wall starts immediately before one `sendall` of the complete pre-serialized
HTTP/1.1 request and ends immediately after the complete response body is read;
strict JSON parsing and all semantic checks happen afterward.

`--quiet-host-status passed` or `failed` additionally requires an absolute
`--quiet-host-receipt` JSON file whose top-level `passed` boolean agrees with
the declared status. `not-evaluated` must not carry a receipt. This is a
diagnostic gate receipt only and never promotes the campaign to formal status.

Both arms receive the exact 383-byte canonical request (SHA-256
`7773f5337693843f1e8cf3017b98868517cbddd3bc32649e550d8f2fec1d5cf6`).
The driver verifies 128 returned raw tokens, length termination, 13/128/141
usage, absence of the five EOG tokens, and equivalent five-token `-inf` policy
receipts. Each runtime must be deterministic independently. Cross-runtime
trajectory equality is neither required nor reported.

The selected OmniInfer clear endpoint is bound during preflight: gateway clear
must use the generation gateway itself, while direct slot erase must use the
resident backend endpoint reported by `/omni/state`. Every OmniInfer generation
must additionally receipt slot 0, zero cached prompt tokens, native `cache_n=0`,
the exact 140-token post-generation KV ledger, and `truncated=false`; otherwise
the run fails terminally.
