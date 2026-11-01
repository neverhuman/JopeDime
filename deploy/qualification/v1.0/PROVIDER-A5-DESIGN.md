# REL-07 A5 named-client provider

## Evidence contract

A5 accepts a full, non-development candidate only when its clean source pin and
full artifact SHA match a fresh named-client transcript package. The package
uses `jopedime.named-client-transcripts/v1`, adds
`candidate_artifact_sha256` to the existing `source` pin, and inventories every
sidecar by byte count and SHA-256. Each client must provide an execution record
and a semantic review receipt bound to the same source, candidate bytes, client
version and client executable hash. The three admitted identities are Claude
Code, the owner-approved Antigravity Gemini client, and Codex CLI.

The provider verifies the complete sidecar inventory, then runs the existing
`verify_capture.py --check-run-retry` validator for the paired wire and ordered
run-inventory checks. That validator explicitly does not qualify the A5 full
path, so its `a5_status` field is never treated as a PASS. The provider also
replays each raw `.mcp.jsonl` stream with the verifier's JSON-RPC pairing parser
and requires exactly 16 successful calls with the specified tool histogram.
Those raw results must show the claimed TRAIN registration and inspection,
plan/create/retry path, completed evaluation, six contiguous events, two
completed-model bundle exports, and prediction matching the semantic receipt.
Sidecar booleans and counts alone cannot establish a flow. A single missing
client, failed call, incomplete raw flow, stale source/artifact, symlink,
non-file, changed sidecar or inconsistent receipt refuses a result. The result
counts three client flows; it does not count tool calls as separate gate
executions.

Before running the provider, capture all three actual named clients against the
candidate executable and place their raw captures and verifier receipts beside
the manifest. Set `JOPE_REL07_A5_MANIFEST` to that absolute manifest path and
`JOPE_REL07_A5_MANIFEST_SHA256` to its reviewed full SHA-256. The five standard
`JOPE_REL07_*` gate bindings select candidate, source and result. Preserve the
manifest hash in the outer gate-command record as an input pin.
The provider does not launch clients or claim that the historical package at
`deploy/qualification/agents/transcripts/20261101T021633Z/` qualifies the
current candidate: its source and executable SHA are older. That package is
useful for offline controls only. A new full candidate-bound run is required.

No client configuration, credentials, sealed or holdout data, GPU, CI or
signing action is needed. The provider neither certifies model parity nor closes
any gate beyond A5's named-client interoperability evidence.

## Data-free controls

Run the provider controls with:

```sh
nice -n 10 env CARGO_BUILD_JOBS=8 CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=line-tables-only uv run --project python --locked --no-sync pytest -q deploy/qualification/v1.0/test_provider_a5
```

Synthetic manifests and child-process results exercise admission and refusal
logic only; they are not named-client runs or release-gate evidence.
