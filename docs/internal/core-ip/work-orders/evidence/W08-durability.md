# W8-T02 onward — Durable outbox, conditional publication and offline faults

Owner: Claude implementation subagent in the lead's integration lane `.agent/worktrees/bench`
(branch `feat/regression-remediation`). This file records W8 work that builds on the
[T01 inventory](W08-inventory.md).

All observations are **fixture-proven** only, against local-filesystem (`file://`) and
in-memory (`memory:///`) `object_store` backends. Faults are injected by a test wrapper.
None of this is `CI-verified` or `live-qualified`, and no real S3 or AWS credential was used.
Register entries `S01-G7`, `S01-K7`, `S08-G03` and `S08-K03` stay open.

## W8-T02 — Persist locally before acknowledging completion

This is a **new capability**: `jope-learning` had no outbox. Local state was written first
and then mirrored directly (`hwm`/`scores` `--store`), with no durable pending-delivery
state. So there was no existing receiving path to reproduce a defect against.

**Source.**

- `crates/jope-learning/src/progress/outbox.rs` — the outbox.
- `crates/jope-learning/src/progress/remote.rs` — the remote trait, implemented by
  `Mirror`.
- `inventory.rs` — `Dependency` is now serializable and has `check`/`verify`.

### Layout

The outbox lives at `<state>/progress/outbox/`:

- `LOCK` — a non-blocking exclusive lock. A second writer is refused.
- `log/<seq:020>.json` — append records `jope.progress.outbox-record/v1`. Each record holds:
  - a canonical JSON body;
  - `prev_record_sha256`, which chains it to the previous record;
  - the item (object or blob), its byte length and sha256;
  - the sorted dependencies the item names.
- `objects/<kind>/<digest>.json` — object payloads.
- `acks/<seq:020>.json` — delivery acknowledgements `jope.progress.outbox-ack/v1`. Each binds
  the sequence number, the key, the record file sha256, the payload sha256, the remote URL
  and the outcome (`created` or `existed`).

Blob payloads use the existing state store at `blobs/sha256/<hex>`, via `State::put_blob`
and `State::read_blob`.

### Boundaries

Each durable step is `ledger::atomic_write`: temporary file, fsync, rename, then a
directory fsync.

1. Payload.
2. Record — **the commit point**.
3. Remote create-only write.
4. Ack.

Guarantees:

- **Crash after step 1.** This leaves an uncommitted orphan payload, which is reported and
  then reused.
- **Crash after step 2.** The entry is committed and pending. A producer replay finds the
  item by key and returns its original sequence number (`Appended::Existing`), so the item is
  recorded logically once.
- **Crash after step 3.** Re-delivery gets `Exists` with identical bytes and acknowledges the
  entry once as `existed`. No second object is created.
- **Failed remote write.** The entry stays pending and is never acknowledged.
- **Remote object with other bytes.** It is refused: the entry is not acknowledged and the
  object is not overwritten.

### Opening

`Outbox::open` verifies every committed file. It fails closed on:

- a sequence gap;
- a broken chain or a non-canonical record;
- a duplicate item;
- a dependency named before it was recorded;
- a missing or altered payload;
- an ack for another record, for other bytes, or for no record;
- any unknown file.

`atomic_write` temporary files are reported as `uncommitted` and are never read.

### Capacity

`Limits` sets the pause thresholds and the maxima for pending entries and pending bytes.
The proposed defaults are **awaiting owner approval**: pause at 7,500 entries or 768 MiB,
and refuse everything at 10,000 entries or 1 GiB.

- **At the pause threshold**, appends that open new work are refused with
  `Refused::Paused`. These are `scheduled` and `retry` events. The check uses the durable
  pending counts, so the pause lands at a committed boundary.
- **Between pause and maximum**, outcomes of in-flight work are still admitted: failures,
  completions, results and evidence.
- **At the maximum**, every append is refused with `Refused::Full`. Nothing is written, and
  the item remains the caller's.

### Admission

The outbox admits only content-addressed items:

- an object whose bytes decode canonically to its domain digest and name exactly the offered
  dependencies;
- a worker result that equals the `BlobRef` its completion will carry.

The run pointer is not such an item, so a producer or worker cannot publish it.

Every dependency must already be recorded. Sequence order is therefore dependency order,
and delivery runs in that order.

### Tests (`progress::tests::outbox`)

- **`crash_after_local_append_before_delivery_recovers_exactly_once`.** It crashes at
  `Payload` and at `Record`, on an object, on a blob and on a retry event, and
  plants a torn temporary file. It then:
  1. reopens;
  2. replays every produced item;
  3. checks that earlier items return their original sequence numbers;
  4. delivers.

  The remote then holds exactly the 10 produced keys. A second delivery attempts nothing.
  After reopening, all 10 are acknowledged.

  It also crashes at `RemotePut` (the 5th item is re-acknowledged as `existed`, with no
  duplicate) and at `Ack`. Every ack's `bytes_sha256` equals the sha256 of the bytes
  the remote holds.
- **`a_full_outbox_pauses_new_work_and_never_drops_or_grows`.** Pause is 4 entries and the
  maximum is 6.
  - While paused, `scheduled` is refused and nothing is written.
  - A failure and a result are still admitted.
  - At 6 entries, a completion, a retry and a blob are refused and nothing is written.
  - Delivery drains the outbox and the refused items are appended again.
  - Pending entries never exceed 6, and the peak is 6.
  - All 11 items end up on the remote exactly once. A census rebuilt from the remote events
    still counts the timeout recorded while paused.
  - The byte bound behaves the same way (100-byte maximum, 64-byte pause).
- **`a_failed_delivery_stays_pending_and_is_never_acknowledged`.**
  - A disconnect on the 3rd write: 2 are acknowledged and the rest stay pending.
  - Response loss after the store applied the write: no ack.
  - After a restart that entry is acknowledged once as `existed`.
  - A remote object with other bytes under the key is neither acknowledged nor overwritten.
- **`dependencies_are_recorded_and_delivered_first`.**
  - An event before its manifest, a completion before its result blob, and a retry before
    its failure are all refused, with nothing written.
  - Recorded dependencies equal what the bytes name.
  - The remote write order, as logged by the fault wrapper, puts every dependency before its
    dependent.
- **`only_verified_content_addressed_items_enter`.** A forged result, a tampered object,
  another kind's domain, hidden dependencies, a non-canonical spelling and a run pointer
  offered as a commit are all refused, with nothing written to the outbox or the blob store.
  A positive control is admitted.
- **`committed_state_is_verified_on_reopen`.** A second concurrent writer is refused. Each of
  9 tampers makes open fail without writing anything:
  1. a record removed;
  2. a record rewritten;
  3. a record replaced;
  4. an object payload altered;
  5. a blob payload lost;
  6. an ack for another record;
  7. an ack for other bytes;
  8. an ack for no record;
  9. an unknown file.

  An untouched copy opens with 10 records, 2 acknowledged and 8 pending.

### Evidence

The evidence root is
`/home/ubuntu/jope-rescue-20260921/regression-remediation-20260926/claude-w0-w12/runs/`.

- **Pre-state.** `w8-baseline-20260926T062059Z/` is the pre-state of this new capability:
  there is no outbox code.
- **`w8-t02-mutations-20260926T064748Z/`** holds the mutation controls: `mutations.txt`,
  and `rerun/` for three mutations whose first spelling did not compile. Each mutation is
  applied to the finished source and restored byte-identically afterwards (sha256 over
  `src/**/*.rs`). Each is caught by at least one test:

  | Mutation | What it breaks | Caught by |
  |---|---|---|
  | `pause` | Opening work ignores the pause | `a_full_outbox_…` |
  | `dedup` | A re-append is not recognised | 5 of 6 outbox tests |
  | `ack-on-failure` | A failed remote write is acknowledged | `a_failed_delivery_…` |
  | `no-max` | The maximum is not enforced | `a_full_outbox_…` |
  | `deps-unordered` | Dependencies are not required first | `dependencies_are_…` |
  | `ack-bytes-unchecked` | Open does not bind an ack to the record's bytes | `committed_state_…` |
- **`w8-t02-postfix-20260926T065338Z/`** holds the full run and supporting files:
  - `nextest.log` — the full `jope-learning` run;
  - `doctest.log`;
  - `check.log` — jope-motifs, jope-cli and jope-bench, all targets;
  - `diff.patch` with its sha256, and `diff-check.txt`, which is empty.

  Clippy reports no warnings in the W8 files.

### Not established here (open)

- **Nothing in the product appends to the outbox yet.** No CLI verb, and no hookup to the
  `hwm`/`scores` mirror push or to jope-engine producers.
- **Retention.** Acknowledged entries are kept, and nothing is compacted or deleted.
  Retention classes are W8-T05. Automatic deletion is out of scope.
- **Remote truth.** Acks are local assertions. Checking them against the remote graph is
  W8-T04/T06 reconstruction.
- **Durability.** Directory fsync is best-effort, as in `atomic_write`. A newly created
  parent directory's own entry is not fsynced.
- **Limits.** The `Limits` defaults are proposals, not approved high-water values.
