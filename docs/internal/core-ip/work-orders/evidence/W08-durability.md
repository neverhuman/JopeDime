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

## W8-T03 — Dependency-first conditional publication

This is a **new capability**. The existing mirror push already orders content-addressed
objects before a conditional `HEAD.json` move, for the `hwm`/`scores` ledgers (`sync.rs`,
`mirror_lost_update`). No run-level committed graph or publisher existed, so there was no
receiving path to reproduce a defect against. The existing `put_immutable` and
`put_conditional` primitives are reused through `Remote`, not duplicated.

**Source.**

- `crates/jope-learning/src/progress/publish.rs`.
- `Outbox::redeliver` in `outbox.rs`.

### Publication order

`prepare` then `advance` (together, `publish`):

1. **Deliver.**
   - Refuse, writing nothing, when the caller is not the run manifest's `publisher`, or when
     the commit belongs to another run or manifest.
   - Append the commit to the outbox. It is refused if it names anything unrecorded.
   - Deliver: dependencies first, then the objects naming them, then the commit. All writes
     are create-only.
   - A delivery failure gives `Publication::Blocked`.
2. **Verify.** `verify_closure` reads the commit's whole closure back from the remote.
   - Every object must be present, canonical and match its domain digest. Every blob must be
     present and match its sha256.
   - `check_rules` must then hold:
     - one run;
     - commit heights count up by one on a single manifest;
     - exposure update `n` starts from the checkpoint of update `n−1` (or from nothing at 0);
     - checkpoints commit exactly their exposure and corpus, contiguous with their parent;
     - HWM decisions and candidate evidence come only from the unit's `completed` attempt.
   - Every failure is a typed `Defect` (`missing`, `corrupt` or `invalid`).
   - A missing object the local outbox can back is re-uploaded, create-only, from its verified
     payload (`Plan.reuploaded`). Anything else gives `Blocked`.
3. **Advance.** Read `progress/runs/<run>/HEAD.json` once, with its version.
   - It names this commit: `AlreadyPublished`.
   - It is not the commit's parent: `Conflict`, and nothing is written.
   - It is invalid, belongs to another run, or misstates the parent's height: `Blocked`, and
     the pointer is left as it is.
   - Otherwise `advance` writes the new `Head` with `put_pointer`, conditional on that
     version.
4. **After a failed pointer write**, the pointer is re-read. Only the pointer can say what
   happened:
   - it names this commit: `Advanced` (the response was lost);
   - it still names the parent: an error, since nothing was written and the publication can be
     retried;
   - anything else: `Conflict`.

   There is no unconditional fallback.

**Losing commits** stay on the remote as unreferenced immutable objects, and in the outbox,
for reconciliation. The winner is never displaced.

**`reusable(remote, run)`** is the only route to reusable corpora and checkpoints. It
verifies the pointer, then its commit's height and run, then the whole closure. Any missing
or corrupt object, broken rule or dangling pointer is an error, never a partial result.

### Tests (`progress::tests::publish`)

- **`racing_publishers_at_one_predecessor_get_one_winner_and_one_conflict`.** It runs on both
  `memory:///`, which exercises the server-side version precondition, and `file://`, which
  exercises the lock-and-revalidate path.
  1. Two hosts, each with its own outbox, prepare against the absent pointer: one `Advanced`,
     one `Conflict{expected: None, found: winner}`. The loser's commit object remains.
  2. Two commits prepared on the same parent: one winner at height 1, one explicit conflict.
  3. The winner then advances to height 2. A retry gives `AlreadyPublished`.
  4. The stale loser retried later is still a `Conflict`, and the pointer stays at height 2.
- **`a_missing_or_corrupt_dependency_blocks_the_pointer`.**
  - A partial or foreign object under a dependency's immutable key blocks delivery. The
    pointer is absent, the object is untouched, and no dependent was uploaded.
  - An object lost remotely after acknowledgement is re-uploaded from the verified local
    payload, and `prepare` itself never writes the pointer.
  - A remotely corrupted object gives `Blocked Corrupt`: not overwritten, pointer unchanged.
  - A blob lost both remotely and locally gives `Blocked Missing`, pointer unchanged.
- **`graph_rules_are_checked_before_the_pointer_moves`.** Each of these gives `Blocked
  Invalid` with the pointer unchanged:
  - a commit height that skips;
  - an HWM decision drawn from a failed attempt;
  - a checkpoint at update 1 with no parent.

  A commit naming an unrecorded object is refused before any remote write (remote snapshot
  unchanged). A contiguous checkpoint is the positive control and advances.
- **`the_pointer_moves_only_from_its_expected_predecessor`.**
  - The wrong publisher and the wrong run are refused with no local or remote write.
  - An injected pointer disconnect gives an error: exactly one conditional pointer request,
    no unconditional write of the key, and the pointer unchanged.
  - An injected pointer response loss gives `Advanced`, from one request.
  - A stale parent gives `Conflict`, with nothing written.
  - A garbage pointer, another run's pointer and a wrong-height pointer are each left
    byte-identical.
  - Once the pointer is restored, publication advances to height 2.
- **`only_a_verified_closure_is_reusable`.**
  - Two commits add a corpus, two exposures of the same corpus (repeated consumption stays
    visible as two exposures) and two contiguous checkpoints.
  - `reusable` returns the height-2 head, 1 corpus, 2 checkpoints and 2 exposures.
  - Each of these makes `reusable` fail; each restore makes it succeed again:
    - checkpoint state corrupted;
    - a corpus member lost;
    - the parent checkpoint lost;
    - the first commit lost;
    - a dangling pointer.

### Evidence

- **`w8-t03-mutations-20260926T070146Z/`** holds the mutation controls (`mutations.txt`,
  plus `rerun/` for two first spellings that did not compile). Each mutation is restored
  byte-identically afterwards and is caught:

  | Mutation | What it breaks | Caught by |
  |---|---|---|
  | `no-predecessor-check` | The pointer moves from whatever it holds | race and pointer tests |
  | `skip-verify` | No closure verification | missing/corrupt and rules tests |
  | `rules-off` | No graph rules | rules test |
  | `corrupt-accepted` | A corrupt object is accepted | missing/corrupt and reusable tests |
  | `lost-response-is-failure` | A lost response is not resolved from the pointer | pointer test |
  | `no-reupload` | A lost object is never re-uploaded | missing/corrupt test |
- **`w8-t03-postfix-20260926T070525Z/`** holds the full run:
  - `nextest.log` — the full `jope-learning` run;
  - `doctest.log` — 1 of 1;
  - `check.log` — jope-motifs, jope-cli and jope-bench, all targets;
  - `diff-check.txt`, which is empty.

  Clippy reported one test-only `manual_inspect` lint in `tests/publish.rs`, which was fixed
  afterwards. The `recheck-*` logs hold fmt-check, clippy and the `progress` tests on the
  final tree, and `diff-final.patch` is that final tree.

### Not established here (open)

- **No product caller.** Nothing creates commits yet: no CLI verb, and no engine publisher
  wiring. `native_persistence.rs`, `corpus_publication.rs` and `checkpoint_recovery.rs` are
  unchanged and are follow-ups.
- **HWM decisions are not re-checked against the ledger.** A decision's link is checked for
  graph consistency only, not against the local HWM ledger (`HwmDecision::verify_against`).
  Wiring that into publication needs the publisher's ledger.
- **Conflicts are detected, not reconciled.** Losing commits are kept, but there is no
  reconciliation or rebase procedure.
- **One publisher per run.** The run manifest's `publisher` names it; the publisher's
  identity is a local string, not an authenticated credential.
- **No live proof.** Live S3 conditional-write behaviour (`If-Match` on the real bucket) is
  W8-T07/T08 and needs a scoped credential.

## W8-T04/T06 — Rebuildable views, crash boundaries and the offline fault matrix

### Two defects in existing code, reproduced first

**1. A crash between a ledger batch and its `HEAD.json` wedged the ledger permanently.**

- **Mechanism.** `Ledger::append` renames the batch, then renames `HEAD.json`. A crash between
  the two leaves a sealed batch that the head does not name. Readers (`events()`) already
  count it.
- **Consequences.**
  - `require_intact` reported `head-mismatch` and refused for ever. Every later
    `hwm ingest`, `scores append`, eligibility decision and `freeze` failed until the tree was
    repaired by hand.
  - An `append` that skipped `require_intact` chained onto the stale head and **forked** the
    chain, producing a permanent `prev-link` break.
- **Fix** (`ledger.rs`).
  - `Ledger::lagging_head` is read-only. It finds the head that an exact, sealed continuation
    of the chain implies. It refuses in each of these cases:
    - a suffix that is not readable, not sealed under its own name, or not for this ledger;
    - a suffix whose event count or hash is wrong, or that is not linked by id and file hash;
    - a head that does not match the batch it names.
  - `Ledger::recover` rolls such a head forward under the ledger lock. It writes nothing, and
    takes no lock, when the head does not lag.
  - `append` and `require_intact` both recover first.
  - The documented limitation stands: a removed tail with `HEAD.json` rewritten to match is
    still undetectable locally. There is no suffix to adopt, and the `ledger_tamper` test for
    it is unchanged.

**2. A mirror `current.json` that failed after the mirror's `HEAD.json` moved was never recorded
as owed.**

- **Mechanism.** `sync::plan` derives the push window from the mirror's `HEAD.json` alone.
  Once `HEAD.json` names a batch, the pointers that batch moved were never offered again.
- **Consequence.** Every later push reported success while the mirror's ledger accepted a
  record at a pointer the mirror did not hold. This lasted until a later batch happened to
  accept at that exact pointer. The old doc comment described this as tolerable staleness.
- **Fix** (`hwm.rs`, `mirror.rs`).
  - `push_mirror` records the pointers moved by the batches about to be published in
    `<state>/progress/mirror-owed/hwm-<sha256(url)>.json` (`jope.hwm-mirror-owed/v1`). It does
    so durably, *before* anything goes up.
  - Every push, including one with nothing new, settles the owed set under the unchanged
    ancestor rule. A pointer leaves the set only once the mirror holds exactly the local
    bytes.
  - Unsettled pointers stay owed and are counted in `MirrorReport.pointers_owed`.
  - A divergent mirror pointer still fails the push, as before. It now keeps failing until it
    is reconciled, instead of being forgotten after one attempt.

**Reproductions.** Each was run pre-fix on unchanged `ledger.rs` and `hwm.rs`, and fails for
the defect's reason:

- `ledger::tests::a_crash_between_a_batch_and_its_head_is_rolled_forward_not_a_wedge` —
  `hwm ledger chain broken … head-mismatch`.
- `ledger::tests::an_append_after_a_lagging_head_never_forks_the_chain` — a `prev-link` fork.
- `ledger::tests::only_a_sealed_continuation_of_the_head_is_rolled_forward` — pre-fix, only
  its positive control fails (a first append that died before any head). Its three negative
  controls already refused and still refuse, writing nothing:
  - an altered batch;
  - a gapped suffix;
  - a forged head.
- `progress::tests::faults::a_crash_between_an_hwm_batch_and_its_head_does_not_wedge_ingest`
  — `hwm::ingest` was wedged. Post-fix, the durable 0.4 acceptance is the HWM, a 0.45
  candidate is `not-improved`, and the pointer rolls forward.
- `mirror_reconcile::a_pointer_write_lost_after_the_head_moved_is_still_owed` — pre-fix the
  owed pointer was `None` after a push with nothing new. The test also covers a push of an
  unrelated cut, and then requires every local pointer to be on the mirror byte for byte.

### Derived views (T04)

This is a **new capability**, in `progress/rebuild.rs`.

- **`summarize`** rebuilds `Summary` from the verified committed closure. `Summary` is now
  extended with:
  - candidate yield (evidenced completions over completions);
  - exposures and consumed rows, with repeated consumption counted each time;
  - corpora and unique teachers;
  - checkpoints and the latest checkpoint;
  - the census counts it already carried, including coverage, failures by class, decisions
    (refusals included) and duplicates.
- **`publish_summary`** stores the summary object create-only. It points the disposable index
  `progress/runs/<run>/summary.json` at it, replacing the index conditionally.
- **`read_summary`** returns `Fresh`, `Stale{derived vs head}` or `Missing`. A corrupt,
  non-canonical or inconsistent index, or a missing summary object, is an error.

Tests (`progress::tests::rebuild`):

- **`a_deleted_derived_index_rebuilds_to_the_same_view`.**
  - Expected values: coverage 2/3 (an unscheduled unit is visible, not hidden), yield 1/2,
    2 exposures of one corpus (6 rows), 3 unique teachers, 2 checkpoints.
  - Deleting the index gives `Missing`; rebuilding gives a byte-identical index.
  - Deleting the summary object as well rebuilds byte-identical bytes.
  - An index whose object is gone is an error until rebuilt.
  - A snapshot of all source evidence is unchanged throughout.
- **`stale_corrupt_or_dangling_state_is_a_visible_failure`.**
  - Covers stale after a new commit, then `Fresh` after a rebuild.
  - Garbage, inconsistent and non-canonical indexes are refused. The rebuild replaces only
    the index.
  - A dangling pointer makes both rebuild and reconstruct fail, and leaves the index
    untouched.
  - A divergent losing commit is visible in the reconstruction and absent from the view.
- **`retries_and_lost_responses_never_duplicate_committed_events`.** Every third immutable
  write loses its response, and the pointer write loses its response. Plain retries converge
  to a summary byte-identical to a clean run, with 0 duplicate deliveries and the same number
  of event objects.

### Reconstruction, restore and the fault matrix (T06)

- **`reconstruct`** classifies a run's keys into these sets:
  - `committed`: the verified closure of the pointer;
  - `uncommitted`, with `divergent` commits;
  - `quarantined`: objects that do not verify;
  - `local_only`: the remote does not hold these exact bytes;
  - `unacknowledged`;
  - `acknowledged_missing`.

  A present-but-invalid pointer, or a closure that does not verify, is an error.
- **`restore`** rebuilds a fresh outbox from the committed closure alone, in dependency
  order. It re-verifies everything and acknowledges it all against the remote.

`progress::tests::faults::every_fault_leaves_committed_state_restorable_and_every_delivery_accounted_for`
runs 18 cases. Each starts from committed c0 and a pending c1, as follows:

1. It injects the fault.
2. It reconstructs, checking that every outbox entry is accounted for (pending = local-only +
   unacknowledged; local-only is never committed).
3. It restores two fresh hosts. They must be identical (deterministic), hold exactly the
   committed closure and none of the local-only items, and a second restore appends nothing.
4. It recovers.
5. It requires a clean reconstruction.
6. A freshly restored host then **resumes the computation**: it publishes unit 2's work on
   the tip at height + 1.

Faults: disconnect before and mid delivery; response loss mid delivery; partial upload
(quarantined, never overwritten); duplicate delivery; conditional conflict (reconciled by
adopting the winner through `restore` and re-committing); missing dependency (remote only, and
everywhere); corrupt blob; stale (rolled-back) pointer; dangling pointer; crash after the
commit's payload, its record, a remote put, or an ack; crash before the pointer write; pointer
disconnect; and pointer response loss. Nextest discards the output of passing tests, so the
per-case lines are recorded separately, by a `--no-capture` run, in `matrix.log` in the final
evidence directory.

### Evidence

- **`w8-t04-prefix-ledger-mirror-20260926T071309Z/`** — the pre-fix run on unchanged
  `ledger.rs` and `hwm.rs`: 5 of 8 selected tests failed, each for the defect's reason
  (`nextest.log`, `diff.patch`, `argv.txt`).
- **`w8-t04-postfix-ledger-mirror-20260926T071515Z/`** — the same reproductions plus the
  existing ledger, hwm, mirror, ledger_tamper, champions_fixture and score_import tests after
  the fixes: 21 of 21 pass.
- **`w8-t04-mutations-20260926T072657Z/`** — 7 mutations of the finished source, each caught
  and restored byte-identically:

  | Mutation | What it breaks | Caught by |
  |---|---|---|
  | `no-rollforward-in-append` | `append` does not recover | the fork test |
  | `no-recover-in-require-intact` | `require_intact` does not recover | 3 wedge tests |
  | `adopt-unlinked-suffix` | An unlinked suffix is adopted | the sealed-continuation test |
  | `owed-not-recorded` | Owed pointers are not recorded | the mirror test |
  | `latest-is-earliest` | The wrong latest checkpoint | the rebuild test |
  | `shadowed-counts-as-delivered` | A partial upload counts as delivered | the fault matrix |
  | `no-divergence` | Divergence is not detected | the matrix and the stale-state test |

  `owed-record/` holds one more mutation, `owed-record-stale`: a stale on-disk owed record
  after a clean push. It is caught by the mirror test's final assertion.
- **`w8-t04-postfix-20260926T073522Z/`** — the full `jope-learning` run: 148 of 148 pass
  with 0 skipped, doctest 1 of 1, and `cargo check -p jope-motifs -p jope-cli -p jope-bench
  --all-targets` passes. The public API changed: `MirrorReport.pointers_owed`, the new
  `Ledger` methods, the `Summary` fields, and `Encoded::from_bytes`. Its `diff-check.txt`
  flagged one trailing blank line at the end of this document, fixed afterwards.
- **`w8-t04-final-20260926T091302Z/`** — the same checks, repeated at lane HEAD `6004e829a`
  (recorded in `HEAD.txt`) on the final tree:
  - fmt-check is clean, and clippy reports no warnings in W8 files;
  - `nextest.log`: 148 of 148 pass with 0 skipped;
  - doctest 1 of 1;
  - `check.log`: jope-motifs, jope-cli and jope-bench, all targets;
  - `diff-check.txt` is empty;
  - `matrix.log` holds the `--no-capture` run of the fault-matrix and rebuild tests, with
    one line per fault case, and `matrix-argv.txt` holds its command line.

### Not established here (open)

- **Best-RMSE views are not derived.** Progress objects deliberately carry no floating-point
  scores. The authoritative best per comparison remains the W3 HWM pointer and ledger
  (`hwm::show`). Joining a summary to it needs the publisher's ledger and is left open.
- **The owed-pointer record is local.** Losing `<state>/progress/mirror-owed/` loses the
  debt list. Rebuilding it needs one GET per accepted pointer: the cost the design avoids.
  `hwm verify` against the mirror would show the disagreement.
- **A divergent mirror `current.json` now fails every push until it is reconciled.** It is
  no longer forgotten after one failed attempt. This is consistent with the existing
  `HEAD.json` divergence rule, but it is a behaviour change operators need to know about.
- **Reconciliation of conflicts is a test procedure only.** The procedure is: adopt the
  winner with `restore`, then re-commit the same content-addressed work on top. There is no
  CLI verb for it.
- **No live proof.** Everything here uses `file://` and `memory:///` stores with injected
  faults. Live S3 (T07/T08) needs a scoped credential and an approved budget.
