# MCP-05 A5 run preparation

This directory prepares repeatable capture review without claiming a client run. The existing
`20261031T015221Z/` capture is the baseline: its two live clients stopped at the real
`dataset_register` capability error, Gemini was unavailable, and A5 remains `NOT_QUALIFIED`.

## Design

Once the named clients and complete AG1 server routes are available, make a fresh run directory
under `../` and retain each client's exact version, invocation, stdout/stderr, MCP wire frames,
engine receipt, and SHA-256/byte inventory. Reuse the captured train-only input and do not substitute
mock replies. A5 needs three complete client transcripts and a successful same-key `run_start`
retry that returns the same run ID while the persisted run count remains one.

This prep adds two bounded pieces: `capture_receipt.template.json` is an explicitly `NOT_RUN`
starting receipt, and `verify_capture.py` checks that every sidecar declared by an existing capture
manifest is a regular file with the declared byte count and SHA-256. The verifier reports the
manifest's A5 label as unverified metadata; it does not qualify A5. A hash inventory establishes
artifact integrity only, not that a client made the calls or that the gate passed.

The alternative of replaying clients now is rejected because the merged baseline already records
real capability failures and Gemini's absence. Editing backend routes is outside MCP-05's owned
path. The primary risk is mistaking intact partial transcripts for successful qualification; the
verifier deliberately keeps integrity and qualification as separate results.

## Use

From the repository root:

```sh
python3 deploy/qualification/agents/transcripts/run-prep/verify_capture.py \
  deploy/qualification/agents/transcripts/20261031T015221Z/manifest.json
python3 deploy/qualification/agents/transcripts/run-prep/verify_capture.py \
  path/to/new-run/manifest.json --root path/to/new-run
```

Exit 0 means only that declared sidecars match the manifest. `a5_status` in its JSON output is
reported as `UNVERIFIED:<manifest value>`. Missing files, symlinks, path traversal, size mismatch,
or hash mismatch fail closed. Keep credentials out of transcripts and receipts.
