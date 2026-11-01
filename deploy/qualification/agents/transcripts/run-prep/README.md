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

### Candidate-bound three-client custody package

`capture_clients.py` packages already captured, fresh sidecars for exactly
`claude_code`, `gemini_cli` and `codex_cli`. It does not start clients, an engine,
or inspect dataset contents. The caller supplies a clean checkout SHA, the exact
candidate file and SHA-256, a DSC01 handoff JSON, client identity metadata, and a
directory containing each client's `.execution.json`, `.verification.json` and
`.mcp.jsonl` files. The handoff is content-pinned and must identify `TRAIN` (the
`split` field may be omitted only when the handoff itself is the approved TRAIN
identity). The output directory must be new.

```sh
nice -n 10 python3 -B deploy/qualification/agents/transcripts/run-prep/capture_clients.py \
  --repo "$PWD" --source-sha "$SOURCE_SHA" \
  --candidate "$CANDIDATE" --candidate-sha256 "$CANDIDATE_SHA256" \
  --input-handoff "$DSC01_TRAIN_HANDOFF.json" \
  --input-handoff-sha256 "$HANDOFF_SHA256" \
  --client-metadata "$CLIENT_METADATA.json" \
  --capture-root "$FRESH_CAPTURE_SIDECARS" --out-dir "$PRIVATE_CUSTODY/FRESH-package"
```

The tool copies only the nine required sidecars into a private new directory and
writes `capture-manifest.json` plus `trusted-manifest.sha256`. Recompute and
independently review that pin after the manifest is finalized; the pin is a
content identity, not a signature or an authority by itself. The custody schema
is deliberately separate from the A5 provider manifest. It records
`PREPARED_UNVERIFIED` / `NOT_QUALIFIED` and must not be passed as an A5 PASS
receipt. Semantic review, provider-ready manifest construction, and the actual
candidate/handoff remain separate gates. Never fabricate client results to make
this package qualify. It verifies the source, candidate and client identity fields
inside each execution and verification JSON receipt before copying; the MCP wire
sidecar is bound by its inventory hash. Outputs are refused inside the checkout
or under `/tmp`, and no existing output is overwritten.

The synthetic controls use a disposable empty Git repo and fixture bytes only:

```sh
nice -n 10 python3 -B deploy/qualification/agents/transcripts/run-prep/capture_clients.py --self-test
pytest deploy/qualification/agents/transcripts/run-prep/test_capture_clients.py -q
```

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

## Reproducible stdio capture

The historical capture producer lived only in host-local evidence. The repository
now supplies `capture_stdio.py` for a fresh, bounded recording on a POSIX host
with `poll`, `waitid` and `WNOWAIT` (tested on Linux). Configure a named client's
temporary MCP server entry to run the equivalent of:

```sh
nice -n 10 python3 -B deploy/qualification/agents/transcripts/run-prep/capture_stdio.py \
  --out-dir path/to/FRESH-client-capture --seconds 180 --max-bytes 16777216 \
  -- /absolute/path/to/jopedime mcp
python3 -B deploy/qualification/agents/transcripts/run-prep/capture_stdio.py --self-test
```

Use the admitted client invocation and isolated `JOPEDIME_HOME`; this recorder
does not edit shared configuration, select a model, launch the named client or
create an engine store. Server arguments are passed directly, without a shell.
Keep credentials out of command arguments and the recorded MCP traffic. No
environment dump is retained.

The output directory must not exist. It is created privately (0700, with 0600
files), and no old directory or evidence is deleted or overwritten:

- `client.raw` and `server.raw` retain the received stream bytes, including invalid
  or partial frames on a failed capture.
- `wire.jsonl` contains only observed JSON-RPC frames with the existing
  `at`/`direction`/`message` envelope. Forwarded valid frames retain their original
  bytes and whitespace. No replies or exit-status frames are invented. Recording
  enforces UTF-8 JSON-line framing and the `2.0` marker; request/reply semantics
  remain the verifier's responsibility.
- `server.stderr` is separate from protocol stdout.
- `capture.json` records producer/parser/executable hashes, exact server argv,
  time, actual server/relay exits, frame counts, stream hashes and limits.

Bounds cover retained stream files (default 16 MiB), individual frames (1 MiB)
and monotonic wall time (default 180 seconds). A failed/limited capture can retain
only a prefix; its receipt reports `complete:false` and a nonzero relay exit.
The fixed metadata receipt is additional to the stream-byte bound. Invalid
framing is retained in raw files and refused, rather than forwarded as a tool
reply. Normal nonzero server exits are preserved. Timeout, interrupt, broken
consumer or capture failure terminates only the fresh server process group;
the group leader's PID stays reserved until cleanup, including after normal
leader exit. Descendants that deliberately leave that group are outside this
recorder's cleanup guarantee.

The recorder's receipt is **not** `<client>.execution.json`: it cannot observe
the parent named client's final exit and records `client_exit_code:null`.
Retain the actual client command/version/stdout/stderr and execution receipt
separately. Package the wire as `<client>.mcp.jsonl` with its hash and retain the
capture receipt/raw files as additional sidecars. `complete:true` describes
transport capture, not tool success or AG1 completion. A5 is always `UNVERIFIED`.
Self-tests use disposable synthetic child processes only, never a real engine
or a named-client qualification run.

### Child environment

The server receives an explicit environment, rather than inheriting every caller
variable. Defaults retain `PATH`, `HOME`, `LANG`, `LC_ALL`, `LC_CTYPE`, `TZ`,
`TMPDIR`, `TMP`, `TEMP`, the four XDG config/data/state/cache home paths,
`JOPEDIME_HOME`, `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`,
`RAYON_NUM_THREADS` and `CUDA_VISIBLE_DEVICES`. Empty values are preserved,
including an empty CUDA visibility mask; the isolated engine-home and resource
controls therefore remain effective. When `PATH` is absent, the OS default
executable path is used for both lookup and child execution.

Other variables, including ambient GitHub/AWS/API credentials, proxy settings,
Python paths and loader options, are omitted by default. For additional required
configuration, name each variable explicitly; its value comes from the caller's
environment and is never placed in the command arguments:

```sh
# JOPE_DATASETS_URL must already exist in the caller's environment.
nice -n 10 python3 -B deploy/qualification/agents/transcripts/run-prep/capture_stdio.py \
  --out-dir path/to/FRESH-capture --env-allow JOPE_DATASETS_URL \
  -- /absolute/path/to/jopedime mcp
```

`--env-allow NAME` is repeatable. Missing variables or names containing an
assignment are refused before outputs or a child are created. There is no prefix
or wildcard allowance. `capture.json` lists the selected `environment_names` and
explicit `extra_environment_names`, without their values.

Explicit opt-ins can still contain sensitive values, and a server can echo them
into its output. Argv, wire traffic and stderr remain unredacted; keep secrets out
of those channels. Environment selection does not sandbox filesystem access:
the server retains the caller's normal file permissions, including the selected
home/configuration paths. These controls limit ambient propagation and do not
establish client provenance or qualify A5.

## Bounded system/dataset prefix probe

`probe_prefix.py` is a deterministic protocol test client for preparation while the
run lifecycle is unavailable. It reuses the unchanged recorder above and identifies
itself as `jope-prefix-probe`, not Claude Code, Gemini CLI or Codex CLI. It launches
no named client or LLM. Its only tool calls are `system_info`, `dataset_inspect`
(list/profile), and `dataset_register` with an identical-key registration retry.
Initialization is protocol lifecycle traffic. The send path rejects every other
method/tool before writing; it never invokes `run_*`, `model_*` or control tools.

Use a source-bound CPU executable and an approved, hash-pinned training-only CSV.
Both source SHAs below are full 40-character values: the executing harness commit
and the actual executable's build source. Keep a scoped compatibility comparison
when these differ; a passing prefix is not proof that the executable was built at
the harness commit. Binary/input digests are lowercase SHA256 values.

```sh
nice -n 10 python3 -B deploy/qualification/agents/transcripts/run-prep/probe_prefix.py \
  --binary "$CPU_BINARY" --binary-sha256 "$BINARY_SHA256" \
  --binary-source-sha "$BUILD_SOURCE_SHA" --harness-source-sha "$HARNESS_SHA" \
  --csv "$APPROVED_TRAIN_CSV" --csv-sha256 "$TRAIN_SHA256" \
  --out-dir "$PRIVATE_CUSTODY/FRESH-prefix" --seconds 45
python3 -B deploy/qualification/agents/transcripts/run-prep/probe_prefix.py --self-test
```

The output must be new, outside Git and `/tmp`. Input is limited to 1 MiB. The
probe uses private HOME/XDG/engine storage, a cleared explicit child environment,
one-thread settings and an empty CUDA mask. It reserves at most two allowed CPUs
under the host's 85% ceiling; no existing engine/store is reused. It terminates
only its own fresh process groups, reserving each leader PID through cleanup.
MCP capture is limited to 1 MiB and the supplied monotonic deadline (at most 120s);
owned-process cleanup has additional bounded grace periods. Engine diagnostics
retain at most 64 KiB and explicitly report truncation. Retain underlying build
receipts and input custody separately; source/digest fields are caller-supplied
pins checked against executable/input bytes, not build attestations.

`capture/` contains the real recorder's raw/wire bytes, stderr and receipt.
`prefix.json` records actual per-step replies, source/digest identities, exits,
transport completeness and semantic success separately. Exit 0 requires all six
prefix steps plus the repeated dataset ID; exit 2 retains partial/refused tool
outcomes, and 125 reports a preflight/start failure. Listing errors do not prevent
independent registration probes. Profile inspection is `not_run` if registration
returned no successful ID; no ID is invented. A same-ID registration retry does
not prove persisted uniqueness or successful run-start idempotency.

Every result reports zero named-client runs and `a5_status: NOT_QUALIFIED`.
Synthetic embedded controls are client/cleanup tests only. Full AG1, actual pinned
named clients, run retries/inventory, export/prediction and A5 remain outstanding.
The private fixture/store/raw transcripts are never committed by this probe.

### CPU run and retry replay

Add `--run-target TARGET_COLUMN` to explicitly select the next AG1 subpath. The
same source/input pins, fresh private store, recorder and deadline apply. Default
prefix mode still refuses every run tool. Selected mode additionally admits only
`run_start`, `run_status` and `run_events`; model/control/fork tools remain refused.

```sh
nice -n 10 python3 -B deploy/qualification/agents/transcripts/run-prep/probe_prefix.py \
 --binary "$CPU_BINARY" --binary-sha256 "$BINARY_SHA256" \
 --binary-source-sha "$BUILD_SOURCE_SHA" --harness-source-sha "$HARNESS_SHA" \
 --csv "$APPROVED_TRAIN_CSV" --csv-sha256 "$TRAIN_SHA256" \
 --out-dir "$PRIVATE_CUSTODY/FRESH-run-replay" --seconds 90 --run-target target
```

After successful registration/retry, the probe requests a dry-run CPU regression
plan, then creates one run and retries the identical body/key exactly once. Its
fixed spec uses heuristic guidance, no landmarks, one evaluation and five nominal
wall seconds. The plan must select one CPU worker. Native wave/finalization time
may exceed the nominal search budget; the outer monotonic deadline and owned
process-group cleanup remain in force. Failed plans never proceed to a start.

The unchanged wire verifier checks the three ordered, complete run inventories:
before creation, after creation and after retry. It requires the exact ID-set
transition, including preservation of existing runs, rather than equal counts
alone. Polling and paged events must retain the returned run ID and contiguous
sequences through `run.finished` at a completed snapshot's watermark. Status uses
non-waiting requests with 250 ms between polls; this does not qualify long-poll
latency or the full AG1 wait-seconds path. Failed/cancelled runs,
different retry IDs, missing/paginated inventories, event gaps and capability
errors cannot produce replay success. Existing inspection failures remain visible
in the prefix results; AG1's run subpath does not require `dataset_inspect`.

`run-replay.json` reports `run_replay_success`, the wire retry contract, terminal
watermark and actual per-step replies separately from `prefix_success` and
`transport_complete`. A verified retry subpath is retained even if later polling
fails. Semantic refusals close recorder input and drain the capture normally, so
a complete error exchange remains distinct from incomplete transport.
Exit 0 requires completed transport, a successful run and
the verified retry contract; exit 2 retains an incomplete/refused replay. Capture
integrity alone cannot satisfy these conditions. The server must have implemented
the selected operations; current-main and candidate build identities are distinct.

This deterministic protocol probe is not a named-client invocation. Every receipt
retains zero named-client/LLM calls and `a5_status: NOT_QUALIFIED`. It does not
inject response loss, restart the engine to prove on-disk uniqueness, or invoke
export/prediction. Those steps and all three pinned actual clients remain required
for full AG1/A5. Embedded positive/negative controls are synthetic harness checks.

### Completed-run retry across an engine restart

Add `--restart-retry --run-target TARGET_COLUMN` to the bounded CPU command
above. This opt-in first completes the existing one-evaluation run/retry replay,
closes its recorder and stops/reaps its owned engine. A new engine process opens
**the same private engine home**, and a second captured MCP session lists the
persisted run, repeats the exact original mutation arguments/key, lists again,
and reads completed status and durable events. It never re-registers the dataset
or creates another store during recovery.

The new `run-restart.json` and outer exit code require all of the following:

- Both recorder captures complete successfully and their four streams match the
  declared byte lengths/hashes and admitted executable digest.
- The first complete inventories contain exactly 0, 1, 1 runs; the restarted
  inventories contain exactly 1, 1 with the same actual run ID. Equal counts with
  substituted IDs, failed/paginated lists and changed request types/keys fail.
  All three successful mutation replies must retain the entire original outcome,
  not only its run ID.
- The completed snapshot watermark and full ordered durable events match across
  restart, including the terminal `run.finished` event.
- The first engine has been stopped/reaped and a distinct owned engine PID opens
  the same home. Both phases share one monotonic deadline (`--seconds`, max120),
  with bounded owned-process cleanup. The initial host CPU capacity is retained
  for admission after the probe restricts itself to two affinity slots.

The original `run-replay.json` remains a first-stage component result; it does
not establish successful restart. A failed first stage never starts recovery.
The new result retains zero named-client/LLM calls and A5 `NOT_QUALIFIED`.
This tests completed-run persistence and idempotent replay on the recorded CPU
build. SIGKILL at internal cuts, response loss between commit and delivery,
actual named clients, full AG1, export and prediction remain separate acceptance
requirements. Source/build identities and a passing subset must stay explicit.
