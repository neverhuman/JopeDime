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

## Captured run-retry check

The optional retry check inspects an actual capture's JSON-RPC wire evidence for
each named client. It requires a trusted manifest SHA-256 supplied by the caller,
the recorded client version, and hash-bound execution and wire sidecars. Obtain
the pin from the reviewed capture receipt; hashing an untrusted replacement
manifest does not authenticate it.

```sh
python3 deploy/qualification/agents/transcripts/run-prep/verify_capture.py \
  path/to/new-run/manifest.json --check-run-retry \
  --manifest-sha256 REVIEWED_MANIFEST_SHA256
python3 deploy/qualification/agents/transcripts/run-prep/verify_capture.py --self-test
```

For each of `claude_code`, `gemini_cli` and `codex_cli`, retain
`<client>.execution.json` and `<client>.mcp.jsonl` in the manifest's file inventory,
using the existing hyphenated filenames (`claude-code`, `gemini-cli`, `codex-cli`).
Execution records must identify that client, have integer-zero `exit_code` and
`engine_exit`, and `timed_out:false`. The wire format is the existing transparent
relay's JSONL: `direction` (`client_to_server` or `server_to_client`) and `message`
(the complete JSON-RPC object).

The verifier pairs requests and replies, including lifecycle traffic. RPC string
and integer IDs remain distinct; reuse after a reply is allowed. A Boolean ID,
unmatched reply, overlapping reused ID or unanswered request refuses. Duplicate
JSON keys, nonfinite numbers and invalid UTF-8 also refuse.

There must be exactly two successful non-dry `run_start` calls with identical
JSON arguments, a nonblank idempotency key and the same canonical run ID. Capture
three complete `run_status` calls **without `run_id`**, so they list runs:

1. A completed list before the first creation request.
2. A list requested after the first creation reply and completed before retry.
3. A list requested after the retry reply.

Each selected list must contain unique valid run IDs and explicit
`next_after:null`; paginated inventories cannot prove a complete count here.
The returned run must be absent initially. The first creation must add exactly
that ID while preserving existing runs, and the retry must leave the ID set
unchanged. Equal counts alone are insufficient. Dry-run plans cannot stand in for
a created run. The original partial capture therefore passes integrity-only
verification but fails this stronger check.

Success adds `run_retry_contract` with each client's returned run ID, two
successful attempts and the three inventory counts. **A5 remains
`UNVERIFIED:<manifest value>`, even when the manifest claims `QUALIFIED`.** This
checks a necessary captured subpath, not full AG1 or a client's provenance. It
does not prove on-disk persistence, response-loss injection, engine isolation or
complete API schemas. Independent review still needs real client/version/source
and input custody, the dry-run plan, polling/events, export/prediction and the
persisted-store inventory. Synthetic self-tests are verifier controls only;
they are never named-client runs or qualification evidence.
