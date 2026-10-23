# Model sessions, versions and rollback

A session is an append-only history over [immutable model objects](model-objects.md).
Each save retains a new object ID. Recording that ID in a session makes it the active
selection. Rollback appends a selection of an earlier version. Every previous candidate
and selection remains available. Session operations never fit a model or rewrite its bytes.

## Save, select, continue

Use one explicit absolute store root per application. Read each command's JSON response;
`head` is the exact token required by the next update, initially `empty`.

```sh
# Fit version A; read outputs[0].path and its sha256-* basename from fit-a.json.
jopedime fit --train train.csv --target y --method ridge --model-root /data/models --json > fit-a.json
jopedime model session record --root /data/models --name experiment-1 \
  --id "$MODEL_A_ID" --expected-head empty --label initial > version-a.json

# Continue the search/training in its owning worker, then save candidate B.
# Read the returned object ID into MODEL_B_ID and version-a.json head into HEAD_A.
jopedime model session record --root /data/models --name experiment-1 \
  --id "$MODEL_B_ID" --expected-head "$HEAD_A" --label candidate-b > version-b.json
jopedime model session show --root /data/models --name experiment-1

# Read version-b.json head into HEAD_B. Version numbers are 1-based.
jopedime model session rollback --root /data/models --name experiment-1 \
  --version 1 --expected-head "$HEAD_B" --label restore-a > rollback.json
# rollback.json model_path identifies the original A object for predict.
# A later candidate C is recorded against rollback.json head as version 4.
```

`fit` always persists its model by default; session recording is a separate explicit command.
If training succeeds but recording fails, retain the fit receipt and object ID, inspect the
session, and retry recording with the head you deliberately choose. Do not refit or delete
the saved object. A failed/stale expected head returns output failure (exit 7) and cannot
replace the active selection. A caller must read and make a new decision before retrying.

`show` returns `schema`, `session`, `head`, `versions` and `model_path`. An empty/nonexistent
session has `head: "empty"`, no versions and null model path. The latest entry selects the
active model. A history entry contains `version`, `previous`, `model_id`, `selected_from`,
`label`, `schema` and `session`. Ordinary records have null `selected_from`; rollback names
a previous entry and must select exactly its model ID. Labels are descriptive, at most
256 UTF-8 bytes, and do not encode authoritative metrics or comparison identity.

Prediction should pin `model_path` or `model_id` from one history response for the whole
request. Do not reread the active selection between batches of one inference run. The
selected object is verified again when loaded. `show` validates the history chain but does
not load all referenced payloads; a missing selected model will fail at inference. Recording
or rollback validates the selected object before committing. Unknown-codec checkpoints can
be recorded for retention but require their own compatible reader before use.

## Training continuation is a separate capability

An inference object preserves a fitted predictor. It does not by itself preserve the state
of a continuing search or optimizer. Exact continuation may require the entire candidate
set, archive, random generator state and algorithm/version, optimizer and scheduler,
worker progress, counters, budgets/deadline, operator vocabulary, data/fold/cut identities,
evaluator version, admission history and comparison receipts. Restoring only the best
predictor can start another search, but cannot promise the same subsequent trajectory.

An owning training system can place a versioned checkpoint in this same envelope, list every
required buffer and dependency, and record its object ID in a session. It must register and
qualify a checkpoint decoder before claiming resume support. These commands do not add a
training worker resume implementation. Save predictor and checkpoint references together in
a future declared compound codec if they must be restored as one unit; two independent
session updates are not a transaction. Enforce limits and deadline when continuing, including
after rollback. Do not reset consumed budgets because an earlier predictor was selected.

Each recorded candidate should retain authoritative lineage/comparison metadata in its own
codec: parent object ID, dataset hash, cut/fold identity, evaluator, vocabulary, protocol and
source version, plus generation/step where meaningful. The envelope leaves these fields to
the owning codec rather than inventing universal search semantics. A rollback selects a
retained artifact; it does not retract historical evaluation evidence or change sealed cuts.

## Session wire format and concurrency

Session names are 1–64 lowercase ASCII letters, digits or hyphens, starting with a letter
or digit; Windows device names are refused. The store path is
`models/sessions/<name>/<20-digit-version>.json`, starting with
`00000000000000000001.json`. Each event is UTF-8 JSON with schema
`jope.model-session/v1`; see [`Version`](../crates/jope-runtime/src/model_session.rs).
`previous` is `empty` for version 1 or `sha256-` plus SHA-256 of the exact previous event bytes.
The response `head` hashes the latest event. All keys are unique and unknown fields are
refused. The v1 reader permits at most 10,000 events, 4 KiB per event. Event versions are
small exact integers across the supported language interfaces.

There is no mutable HEAD file or lock to leave stale after a crash. A writer reads and
validates the contiguous hash chain, compares `expected_head`, then atomically creates the
next numbered event with a hard link. Exactly one competing writer can commit that slot.
A loser reports failure, including when both requests proposed identical bytes. A timeout
may occur after a successful commit: read history to reconcile before another update.
A crash before the link leaves only an ignored `.jope-write-*` file. A crash after it leaves
a complete event. Directory sync follows commit on Unix. A reader can observe an older
valid prefix during concurrent publication; retry a transient scan failure without repairing
or rewriting history. Sessions require the same trusted local filesystem assumptions as
model publication. Cross-host/network filesystem coordination is not qualified.

Rollback itself creates a version. After A, B, rollback-to-A, C the history contains four
entries. Repeated rollbacks preserve that history. Branch into a different session by
recording the desired retained object there with `expected_head empty`. Session names are
local conveniences, not global identities or authorization boundaries.

Back up the complete session directory **and every referenced model object**. Copying an
object alone does not copy session history. No garbage collector or automatic pruning is
provided. Deleting a superseded object can break rollback; do not reclaim it while a session,
checkpoint, receipt or external consumer references it. If a history entry is missing or
corrupt, the reader refuses the chain. Preserve it and restore a verified backup. The hash
chain detects internal inconsistency, but cannot detect an attacker rewriting the whole
history or truncating a valid suffix without an independently retained head token.

## MCP and language adapters

The same configured-root MCP server exposes `model_session_show {name}`,
`model_session_record {name,id,expected_head,label?}` and
`model_session_rollback {name,version,expected_head,label?}`. Updates require requests with
IDs; notifications never record state. MCP's 8 MiB message/response limit applies, including
large histories; use CLI for histories beyond that transport budget. Object IDs, session
names and version numbers are validated in the shared native implementation. The server
never accepts arbitrary destination paths from tool arguments.

Every CLI-capable language can use these JSON commands. The executable
[Node acceptance example](../examples/model-objects/check.mjs) demonstrates multiple retained
objects, explicit rollback and disk inference. Keeping session events separate from model
bytes lets languages share the storage and version policy without sharing an in-memory
object representation.
