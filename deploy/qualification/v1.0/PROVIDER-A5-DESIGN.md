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
`verify_capture.py --check-run-retry` validator. It independently checks all
three semantic receipts for complete 16-call AG1 flows, clean client/server
exits, exact same-key run retry with inventory counts `0 -> 1 -> 1`, completed
CPU evaluation, six contiguous events, export retry and prediction output. A
single missing client, failed call, stale source/artifact, symlink, non-file,
changed sidecar or incomplete receipt refuses a result. The result counts three
client flows; it does not count tool calls as separate gate executions.

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
