# W8 — Publish reusable intermediate learning progress to S3

Status: planned. Owner: learning persistence/publisher. Requires W3 decision semantics,
W6 complete checkpoints and W7 immutable worker results. xbabe2 is the sole publisher
initially. Apply [the execution contract](EXECUTION-CONTRACT.md).

## Storage boundary and sources

Extend `crates/jope-learning/src/ledger.rs`, `mirror.rs`, `hwm.rs`, `state.rs`, existing
blob/synchronization code, and `crates/jope-engine/src/native_persistence.rs`,
`corpus_publication.rs`, `checkpoint_recovery.rs`. Inspect actual conditional write
primitives and existing `mirror_lost_update`/`mirror_reconcile` tests before adding code.

Use the learning bucket only through explicit `--store`/`JOPE_LEARNING_URL`. Live
qualification uses a separate run-scoped prefix. Historical `staging/` stays historical;
renaming its objects does not promote them. The available administrator AWS profile
must not enter workers or substitute for a narrowly scoped publisher credential.
Credential provisioning/IAM changes are outside this work order's authorization.

## Ordered tasks

### W8-T01 — Define the durable artifact and event inventory

- [ ] Specify immutable run manifests, scheduled/attempt/retry/failure/completion events,
  candidate evidence, qualification/HWM decisions, corpus manifests, exposure/update
  records, complete checkpoints and rebuildable summaries.
- [ ] Bind logical work identity separately from delivery/attempt identity, source and
  parent dependencies. Define finite canonical encodings, schema limits and hash domains.
- [ ] Acceptance: one run's complete planned census reconciles to terminal dispositions;
  a failed or retried attempt cannot disappear or become a second logical improvement.

### W8-T02 — Persist locally before acknowledging completion

Depends on T01.

- [ ] Implement/reuse a durable outbox with fsync/transaction boundaries, sequence IDs,
  dependency references and acknowledged delivery state. Worker results arrive as verified
  immutable artifacts; workers cannot publish run pointers themselves.
- [ ] Specify bounded local capacity and approved high-water behavior. At capacity,
  pause new work at a committed boundary; preserve pending evidence and failure accounting.
- [ ] Acceptance: crash after local append but before acknowledgment/delivery recovers
  exactly once logically. A full outbox pauses rather than dropping evidence or growing unbounded.

### W8-T03 — Implement dependency-first conditional publication

Depends on T02 and W3/W6 committed manifests.

- [ ] Upload content-addressed dependencies first, immutable manifests second, then
  conditionally advance the committed run pointer from its expected predecessor/version.
  Reuse conditional primitives; unsupported preconditions fail without unconditional fallback.
- [ ] Verify referenced objects before marking remotely published corpus/checkpoint
  reusable. Preserve conflicts/divergent histories for reconciliation rather than overwriting.
- [ ] Acceptance: two writers racing the same predecessor produce one winner and one
  explicit conflict. Missing/corrupt dependency prevents pointer publication or reusable status.

### W8-T04 — Make retries and summaries rebuildable

Depends on T03.

- [ ] Retry object upload idempotently; bind acknowledgments to exact bytes and logical
  delivery. Handle response loss after successful remote commit without duplicating events.
- [ ] Reconstruct secondary indexes, best-RMSE views, candidate yield, refusals,
  unique teachers and optimizer exposure from committed records. Report freshness and coverage.
- [ ] Acceptance: deleting only a disposable derived index and rebuilding it yields the
  same scoped view; stale/corrupt pointers and divergent branches are visible failures.
  Source evidence is never deleted as part of this proof.

### W8-T05 — Apply cadence and retention classes

Depends on T02/T04.

- [ ] Flush compact progress batches every 60 seconds or 1 MiB, whichever comes first.
  Request immediate flush after HWM changes, corpus freezes, checkpoint commits and terminal
  failures. A failed flush remains pending and cannot falsely acknowledge remote durability.
- [ ] Checkpoint at committed updates, initially every 100 updates or five minutes,
  plus controlled shutdown. Retain full fitted artifacts for winners, admitted teachers
  and declared diagnostic samples; compact evidence for other attempts.
- [ ] Acceptance: deterministic clock/size tests prove each trigger and outage behavior.
  No automatic deletion, retention-policy change or hidden dropping of nonwinner outcomes.

### W8-T06 — Execute offline fault-injection matrix

Depends on T02–T05.

- [ ] Inject disconnect, partial upload, duplicate delivery, conditional conflict,
  missing dependency, corrupt blob, stale pointer, response loss and process crashes
  on both sides of each acknowledgment/commit boundary.
- [ ] Reconstruct a fresh local state from the simulated remote committed graph.
  Explicitly distinguish remotely acknowledged progress from local-only progress lost
  before upload; neither may be misreported as fully recoverable.
- [ ] Acceptance: committed reusable state is complete, deterministic and authentic;
  uncommitted/torn state is refused or quarantined and every pending delivery is accounted for.

### W8-T07 — Admit scoped live publisher access and storage budget

Depends on T06 and the credential owner; no IAM mutation task is implied.

- [ ] Obtain the narrowly scoped publisher credential through the owner process; verify
  actual account, allowed run prefix and necessary read/write/conditional capabilities.
  Keep credentials only in publisher memory/environment, out of receipts and workers.
- [ ] Freeze the run prefix, intended object classes, request accounting, byte accounting,
  restoration destinations and proposed 20 GiB/20,000-write ceilings. Include conflict tests,
  retries and failed requests in the approved accounting definition.
- [ ] Acceptance: access outside the admitted scope is refused and the publisher proves
  its valid path. Missing scoped credentials blocks live proof, not offline implementation.

### W8-T08 — Qualify live conflict, outage and restoration

Depends on T07 and W7 managed execution.

- [ ] Through repository code, produce immutable progress, one conditional-write winner
  and one conflict at the same expected predecessor. Interrupt publication and reconcile
  without overriding the winner or losing durable local events.
- [ ] Restore a committed corpus/checkpoint and its dependency graph from S3 into fresh
  custody outside the source tree. Independently hash/validate every required object,
  then resume bounded computation through repository code within the original window.
- [ ] Acceptance: restored state yields the expected next authorized work/update; missing,
  corrupted or stale objects fail. Retain actual request/byte totals, conflict response,
  restore manifest and resumed computation receipt. No credentials or direct SSH scheduler.

## Verification and delivery

```sh
cargo test --locked -p jope-learning --test mirror_lost_update -- --list
cargo test --locked -p jope-learning --test mirror_reconcile -- --list
cargo test --locked -p jope-learning --test ledger_tamper -- --list
```

Proposed slices: (A) inventory/outbox T01–T02; (B) conditional publication T03;
(C) replay/cadence/offline faults T04–T06; (D) separately admitted live T07–T08.
Handoff to W10/W12: reconstructable committed graph, outbox/retry accounting, full
restore and resume witness. “GP target improved,” “policy updated,” and “transfer
improved” remain separate event types with separate evidence requirements.

## Original finding routes

Primary register entries: `S01-G7`, `S01-K7`, `S08-G03`, `S08-K03`.

See [the complete crosswalk](FINDINGS-COVERAGE.md) and original clauses. This primary
routing is not a waiver of cross-cutting acceptance or later live integration.
