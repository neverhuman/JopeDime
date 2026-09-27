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
The proposed defaults were **awaiting owner approval** when this section was written, and
are now `adopted-by-owner-delegation-2026-09-26` (OWNER-DECISIONS.md D6): pause at 7,500
entries or 768 MiB, and refuse everything at 10,000 entries or 1 GiB.

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
- **Limits.** The `Limits` defaults were proposals, not approved high-water values, when this
  section was written. They are now `adopted-by-owner-delegation-2026-09-26`
  (OWNER-DECISIONS.md D6) and remain explicit configuration.

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

## W8-T05 — Cadence and retention classes

This is a **new capability**. `jope-learning` had no flush cadence, checkpoint cadence or
retention class, so there was no receiving path to reproduce a defect against.

**Source.** `crates/jope-learning/src/progress/cadence.rs`. All time comes from an injected
`Clock`.

### Flush cadence (`Cadence`)

A flush is due when anything is pending and any of these holds:

- an urgent request is outstanding:
  - `hwm-change`;
  - `corpus-freeze`;
  - `checkpoint-commit`;
  - `terminal-failure`;
  - `shutdown`;
  - `restart`: `Cadence::start` finds pending entries left in the outbox;
- at least 1 MiB is pending;
- at least 60 s have passed since the last flush that left nothing pending.

Only such a complete flush clears urgency and restarts the interval. A failed, partial or
response-lost flush keeps its entries pending and keeps its urgency.

**Durable watermark.** `durable_through(&Outbox)` is the highest `n` such that records
`1..=n` are acknowledged. It is derived only from durable acks, so it cannot be advanced by
memory, by a failure, or by a restart.

### Checkpoint cadence (`Checkpoints`)

A checkpoint is due at a committed update:

- when 100 updates are uncheckpointed;
- or when 5 minutes have passed since the last checkpoint;
- and at controlled shutdown, when any committed update is uncheckpointed.

It is never due without an update to checkpoint.

### Retention (`retention(Role)`)

- **Full** (fitted artifacts retained): winners, admitted teachers and declared diagnostic
  samples.
- **Compact**: every other completed attempt.

There is no class that retains nothing. Nothing here deletes anything or changes a retention
policy.

### Tests (`progress::tests::cadence`)

- **`each_flush_trigger_fires_exactly_when_due`.**
  - Nothing is due at 59.999 s; the interval fires at 60 s.
  - Size does not fire at 1 MiB minus 1 byte and fires at 1 MiB. The interval then restarts.
  - Each of the five urgent events flushes immediately.
  - An urgent request with nothing pending is satisfied without a flush.
- **`a_failed_flush_stays_pending_and_never_claims_durability`.**
  - An outage keeps the watermark at 0, 4 entries pending, and the urgency set.
  - A partial flush moves the watermark to exactly the acknowledged prefix (2). A lost
    response does not move it.
  - After a simulated crash and reopen, the watermark is still 2.
  - `start` requests `restart` and flushes to 4.
  - Every outbox file present before the outage is still present, with the same bytes.
- **`each_checkpoint_trigger_fires_at_committed_updates_and_shutdown`.** Covers 100 updates,
  5 minutes measured at the crossing update (299.999 s does not fire; 300 s does), shutdown
  with 2 uncheckpointed, and no shutdown checkpoint with nothing uncheckpointed.
- **`non_winner_outcomes_keep_compact_evidence_never_nothing`.**
  - A winner's full evidence (with artifacts) and a non-winner's compact evidence (none) are
    both delivered.
  - The census still counts the non-winner's timeout and completion.

### Evidence

- **Pre-state.** `w8-baseline-20260926T062059Z/` is the pre-state of this new capability:
  there is no cadence code.
- **`w8-t05-mutations-20260926T091939Z/`** holds 7 mutations of the finished source, each
  caught and restored byte-identically:

  | Mutation | What it breaks | Caught by |
  |---|---|---|
  | `failed-flush-clears-urgency` | A failed flush counts as flushed | the outage test |
  | `watermark-counts-attempts` | The watermark counts records, not acks | outage and trigger tests |
  | `no-size-trigger` | The size trigger never fires | the trigger test |
  | `interval-off-by-one` | The interval fires one tick late (`>` instead of `>=`) | the trigger test |
  | `no-restart-urgency` | A restart does not flush pending entries | the outage test |
  | `checkpoint-without-update` | Shutdown checkpoints with nothing uncheckpointed | the checkpoint test |
  | `non-winner-dropped` | Non-winners are retained in full | the retention test |
- **`w8-t05-postfix-20260926T092104Z/`** — at lane HEAD `7906af67c`:
  - fmt-check is clean, and clippy reports no warnings in W8 files;
  - `nextest.log`: 152 of 152 `jope-learning` tests pass with 0 skipped;
  - doctest 1 of 1;
  - `check.log`: jope-motifs, jope-cli and jope-bench, all targets, pass;
  - `diff-check.txt` is empty;
  - `diff-final.patch` is the final tree.

### Not established here (open)

- **Proposed values.** `Policy::PROPOSED` (60 s, 1 MiB, 100 updates, 5 min) and the outbox
  `Limits` were proposals awaiting owner approval when this section was written. They are now
  `Policy::ADOPTED` and `Limits::ADOPTED`, `adopted-by-owner-delegation-2026-09-26`
  (OWNER-DECISIONS.md D6).
- **No product scheduler.** Nothing calls `Cadence::tick` or `Checkpoints` yet.
  - The engine's training loop (`checkpoint_recovery.rs`) and the corpus freeze
    (`corpus_publication.rs`) would request `CheckpointCommit` and `CorpusFreeze`.
  - `hwm::ingest` would request `HwmChange`.
  - This wiring is engine-side and CLI follow-up work.
- **Role is not recorded.** The role that decides retention is supplied by the caller;
  `CandidateEvidence` does not record it. Nothing yet checks that an admitted teacher's
  evidence is `full`.
- **No compaction.** Acknowledged outbox entries are kept indefinitely. Any future compaction
  must preserve the committed graph and is not implemented.

## W8 integration (rr2/integration-a) — receiving-path rules at publication

Lane `.agent/worktrees/bench`, branch `rr2/integration-a` from `main` `e4f1b6c62`. Two gaps the
T01/T03 records left open are closed at the receiving path, `publish::prepare`, before the run
pointer moves. Both were reproduced first through real publication.

**1. Consecutive update indexes.** `check_rules` checked every exposure and checkpoint against
the checkpoint before it, but not that the updates form one chain. Pre-fix, each of these
reached the pointer (`Advanced`):

- a second update 0 committed later (a fork from nothing);
- two updates 1 from the same checkpoint in one commit;
- update 1 committed again after update 1 was committed;
- a second checkpoint of update 0 committing the same exposure.

**Fix.** `check_rules` now requires exactly one exposure and at most one checkpoint per update
index, with the indexes 0, 1, 2, … and no gap. Each case above is now `Blocked Invalid` with the
pointer byte-identical. The positive control publishes updates 1 and 2 on top of update 0.

**2. HWM decisions against the local ledger.** A decision's link was graph-checked only. Pre-fix,
each of these reached the pointer:

- a decision naming a batch this host never wrote;
- an initialization claimed as an improvement;
- a decision only another host's ledger holds.

**Fix.** After the closure verifies, every `HwmDecision` the commit *adds* must pass
`HwmDecision::verify_against` on the HWM ledger of the publishing host (the state tree of its
outbox: `Outbox::state`). Otherwise the result is `Blocked` and the pointer is unchanged.
Decisions an ancestor commit already added are not re-checked, so a host restored from the
remote publishes on top of them; a new decision that host cannot back is refused. The positive
control is a decision built from a real `hwm::ingest` batch.

**Tests.** `progress::tests::receiving` (lib target; no new test target):
`a_forked_or_repeated_update_index_never_reaches_the_pointer` and
`hwm_decisions_are_checked_against_the_local_hwm_ledger_at_publication`.

**Evidence** (under the evidence root):

- `w8-integ-receiving-pre-20260926T194140Z/` — pre-fix, with the two tests only: both fail. The
  log lists all seven accepted defect cases.
- `w8-integ-receiving-post-20260926T194229Z/` — post-fix, the full `jope-learning` run: 154 of
  154 pass, 0 skipped. The existing `graph_rules_are_checked_before_the_pointer_moves` still
  reports its failed-attempt decision as `Invalid`, because the graph rules run first.

**Not established here.** Offline `file://` and `memory:///` stores only. Publisher identity is
still a local string (T03).

## W8 integration (rr2/integration-a) — the product path

**Pre-state.** Nothing in the product appended to the outbox, published through the
conditional publisher or called the cadence. `w8-integ-product-pre-20260926T194952Z/` records
this: a source search at HEAD `65f69edce` for non-test callers of `Outbox::open`,
`append_object`/`append_blob`, `Cadence::start`, `Checkpoints::new` and `publish` outside
`progress/` finds only unrelated code (jope-arena and jope-models publication, the jainnode
control outbox). No engine module names W8 progress, and `jope-learning --help` has no
progress verb. This is a new capability, so there was no receiving path to reproduce a defect
against.

**Why the mapping lives in jope-learning.** As with the champion and genebank sources, the
learning store maps the engine's *published bytes* at its own receiving path. The engine
(`jope-parallel`) is unchanged and keeps no dependency on the learning store. No in-process
producer claim is trusted: every mapping re-verifies the bytes it reads. `jope-parallel` is only
a dev-dependency, so the tests can run the engine's own verifiers on their fixtures.
`Cargo.lock` gains that one edge.

### Mapping (`progress/engine.rs`)

Each mapping reads each file once, checks those bytes and publishes the same bytes.

| Engine event | Engine source | W8 objects |
|---|---|---|
| no-replace publication | `native_persistence::publish_staged` (`{path, sha256}`) | the verified result blob of a `completed` event |
| published corpus | `corpus_publication::build` (`parallel.teaching_corpus.v1`) | a `CorpusManifest` whose six members are the payload files |
| published checkpoint | `checkpoint_commit::commit`, named by `latest.json` (`parallel.checkpoint.v1`) | an `Exposure` and a `Checkpoint` at the next consecutive update index |
| checkpoint recovery | `checkpoint_recovery::reconcile` (`parallel.checkpoint_recovery.v1`) | the attempt's retryable `worker-lost` failure, with the receipt sha256 in its detail |

Refused, with nothing appended:

- a corpus still in staging, a retired corpus, a diagnostic corpus (`qualification: false`), and
  one whose payload, receipt, record length or publication request does not bind together as
  `corpus_publication::verify` requires;
- a checkpoint that `latest.json` does not name (unpublished or quarantined), and one whose
  manifest or objects differ from their digests;
- a checkpoint bound to another corpus (`corpus_sha256` is the corpus `index.json` sha256), and
  one whose step does not advance past the last mapped checkpoint;
- an incomplete recovery receipt, and events for unplanned work;
- a symlink, a non-regular file or a `..` path anywhere on a path that is read.

**The update index.** One W8 update is the window of optimizer steps between two published W8
checkpoints: `update_index` counts those windows 0, 1, 2, …, and `consumed_rows` is
`(step − previous step) × global_batch`. The graph rules need exactly one checkpoint per update,
because an exposure starts from the checkpoint of the update before it. So an engine checkpoint
the cadence does not select is not an update; its steps are counted in the next selected one.
This is a mapping decision recorded for review.

### Publisher (`progress/product.rs`, `Driver`)

`Driver` is the one product path. For one run it:

1. appends mapped items to the outbox, blobs first. A mapping is admitted whole or not at all:
   if its new items would pass the outbox maximum, it is refused (`Refused::Full`) before
   anything is written;
2. commits staged objects on top of the committed tip through `publish`, which applies
   commit 1's receiving rules;
3. drives the cadence:
   - a corpus freeze, a checkpoint commit, an initializing or improving HWM decision and a
     shutdown request an urgent flush;
   - otherwise the flush follows the size and interval triggers;
   - the engine's newly committed optimizer steps are counted by `Checkpoints` (new:
     `committed_updates(n)`). A W8 checkpoint is mapped only when that count says one is due,
     or at controlled shutdown;
   - the checkpoint cadence restarts only after its commit is durable.

**Configuration.** The outbox limits and the cadence are explicit `Config` values
(`jope.progress.config/v1`). An explicit file must name every value; unknown fields,
missing values, zero cadence values and a pause above the maximum are refused. The defaults
are `Limits::ADOPTED` and `Policy::ADOPTED` (named `PROPOSED` when this section was written).
Every report of the defaults carries `approval: "adopted-by-owner-delegation-2026-09-26"`
(OWNER-DECISIONS.md D6; it read `owner-pending` when this section was written), and no file
can change that: a file's own values report `operator-configured-not-the-adopted-defaults`.

**Restart.** Everything is re-derived from the outbox and the remote's verified committed
closure:

- the manifest;
- what is committed;
- what is appended but not committed, which is flushed at once as a restart;
- the last W8 checkpoint, and its engine step, read back from its own `state.json` blob.

A new run's own manifest is not a restart (`Cadence::new`).

**CLI.** `jope-learning progress config|init|corpus|checkpoint|schedule|retry|recovery|result|
decision|flush|follow|status`, with one capability-matrix row each (`learning-subcommand`).

- Every verb except `config` refuses without `--store`/`JOPE_LEARNING_URL`, and writes nothing
  then.
- `status` is read-only.
- `follow` is the long-running product loop. It observes an engine training run's published
  checkpoints and ticks the flush cadence on the wall clock, and it shuts down (checkpoint and
  flush) when the run writes `complete.json`.

**Tests.**

- `progress::tests::product` (lib):
  - `engine_events_reach_the_committed_graph_through_the_product_driver` — the positive
    control, end to end. The engine's own `corpus_publication::verify` accepts the fixture, and
    the engine's own `checkpoint_recovery::reconcile` writes the receipt. It covers:
    - steps 40 and 80 are deferred;
    - step 100 maps update 0 on the update trigger;
    - an interval flush happens at exactly 60 s;
    - step 150 maps update 1 on the 5-minute trigger;
    - step 180 maps update 2 at shutdown;
    - consumed rows are 800, 400 and 240;
    - the census has one completed unit (attempt 2), one retry and one worker-lost failure;
    - the quarantined step-130 checkpoint never reaches the remote;
    - a restarted driver maps nothing twice.
  - `engine_events_that_do_not_verify_append_nothing_and_move_nothing` — 17 refusals, each
    with the local state tree and the remote byte-identical and the head unchanged. An
    unpublished checkpoint maps to nothing, and the positive control then maps update 1.
  - `a_failed_flush_keeps_progress_owed_and_claims_nothing` — a disconnect mid-delivery gives
    `Blocked`, keeps the staged objects and the urgency, and the next tick commits. A restart
    with uncommitted progress commits it at once.
  - `outbox_limits_and_cadence_are_explicit_adopted_configuration` (named
    `outbox_limits_and_cadence_are_explicit_owner_pending_configuration` before D6 was adopted).
  - `hwm_decisions_link_through_this_hosts_ledger_and_flush_at_once`.
- `tests/progress_cli.rs` (new target; coverage.json and test-inventory.json updated):
  - `progress_verbs_publish_engine_events_to_an_explicit_store_only` exercises the shipped verbs
    against a `file://` store: the refusal without a store (no local state is created), the
    configuration report, init, corpus, a deferred then a shutdown checkpoint, schedule and
    flush, a read-only status (store byte-identical), and a wrong publisher;
  - `follow_drives_the_cadence_until_the_run_completes`: a bounded follow of a running run
    publishes nothing below the cadence, and the follow of the completed run maps the counted
    steps at shutdown and exits with nothing owed.

**Post-change run.** `w8-integ-product-post-20260926T200708Z/` holds the full `jope-learning`
run: 161 of 161 pass, 0 skipped. That is 154 before, plus 5 product tests and the 2 CLI tests.
The pre-commit gate re-ran them on the committed tree, with the same result.

**Mutation controls.** `w8-integ-product-mutations-20260926T200052Z/` holds six mutations of
the finished source. Each is caught, and each file is restored byte-identically (sha256
checked):

| Mutation | What it breaks | Caught by |
|---|---|---|
| `staging-accepted` | A staging or retired corpus directory is mapped | the refusal test |
| `corpus-binding-unchecked` | A checkpoint of another corpus is mapped | the refusal test |
| `update-index-not-consecutive` | Every W8 checkpoint is update 0 | the end-to-end and refusal tests |
| `cadence-ignored` | Every published engine checkpoint is mapped | the end-to-end test |
| `flush-not-due` | Every tick commits | the end-to-end test |
| `partial-mapping-admitted` | A mapping past the outbox maximum is half-appended | the configuration test |

**Not established here (open).**

- **No engine-side trigger.** The engine is unchanged. The publisher reads what the engine
  has durably published, through the verbs, or through `follow` running beside a training
  run. No engine or W7 supervisor code invokes it, and the native loop's live path is W7-T07.
- **One update per W8 checkpoint.** A W8 update is a checkpoint window, not an optimizer step
  (see the update index above). Cadence "updates" count optimizer steps.
- **Retention.** Retention is not applied to engine events: engine results are appended whole,
  and nothing is compacted. The role is still not recorded (W8-T05).
- **Publisher identity** is still a local string.
- **Approval.** The configuration values are `adopted-by-owner-delegation-2026-09-26`
  (OWNER-DECISIONS.md D6); they were owner-pending when this section was written.
- **Live stores.** There is no live proof: `file://` and `memory:///` only. Live S3 is W8-T07/T08
  and needs a scoped credential and an approved budget.

## W8-T04 follow-up (rr2/integration-a): the `progress reconcile` verb

**Reproduced first** (`w3w8-verbs-pre-20260926T201229Z/`):
`recovery_cli::progress_reconcile_is_read_only_by_default_and_rebases_only_with_apply` fails,
because no reconcile verb existed. Before this change, a publisher that lost a race, or that
found the run committed at a commit it never recorded, was stuck:

- its next commit named a stale parent (a conflict for ever), or named a commit its outbox had
  not recorded (an error for ever);
- the only way out was the test procedure in the fault matrix.

The driver now refuses that second case up front, with a message naming the verb and nothing
written.

**`progress::reconcile`.**

- `plan` is read-only. It refuses when there is no local outbox rather than create one, and it
  requires the run's named publisher. It reports:
  - the committed head (the winner);
  - this host's commits outside the committed chain (the races it lost);
  - the committed items to adopt;
  - this host's work outside the committed closure (`recommit`);
  - whether the rebased commit would pass `check_rules` (one consecutive update chain included)
    and the HWM-ledger check, and why not.
- `apply` performs an admissible plan, and only that. It runs `rebuild::restore` to adopt the
  winner's closure (every item re-verified), then commits the remaining work through
  `Driver::commit` and the ordinary receiving path. An inadmissible plan is returned unapplied,
  with nothing written.

**CLI.** `jope-learning progress reconcile [--apply]`. Exit 0 when admissible (or there is
nothing to do) and, with `--apply`, when the pointer advanced; exit 1 otherwise. It has a
capability-matrix row.

**Tests.**

- `progress::tests::reconcile::a_lost_race_is_rebased_onto_the_winner_only_on_apply`: a true
  conditional-write race.
  - The loser's driver cannot commit.
  - The plan writes nothing (remote and local snapshots unchanged) and names the winner, the
    one losing commit, the one event to re-commit and the two items to adopt.
  - Apply advances to height 1 from the winner, and both events are committed exactly once.
  - The losing commit stays on the remote as divergent history.
  - A second plan has nothing to do, and a second apply writes nothing.
- `progress::tests::reconcile::a_rebase_the_receiving_checks_would_refuse_writes_nothing`:
  both hosts trained update 0 on different corpora. The plan refuses (`update 0` is already
  committed), and apply leaves remote and local byte-identical.
- `recovery_cli::progress_reconcile_is_read_only_by_default_and_rebases_only_with_apply`: the
  shipped verbs on a `file://` store. It covers:
  - host B's flushes refuse, naming the verb;
  - the plan leaves the store and B's state byte-identical;
  - apply advances from A's commit;
  - status shows both units in flight;
  - a third host that froze another census for the run is refused by both the plan and apply
    (exit 1, graph rules), with nothing written.

**Post-change run.** `w3w8-verbs-post-20260926T201520Z/`: the full `jope-learning` run, 166 of
166 passing, 0 skipped.

**Not established here (open).** Reconciliation adopts the winner and carries this host's own
work. It does not merge two diverged histories whose work conflicts (for example, two
update-0 checkpoints), which it refuses with the reason. Publisher identity is still
unauthenticated. Live S3 conditional writes are W8-T07/T08.

## Owner decisions of 2026-09-26 (recorded on rr2/integration-a)

The owner delegated the pending W8 policy choices and adopted the conservative options. This
section supersedes the "owner-pending" and "proposed" wording of the earlier sections, which
stays as the history of what was reviewed. The canonical record of each decision is
`docs/internal/core-ip/work-orders/OWNER-DECISIONS.md`, under the D-number given below.

**Outbox limits and cadence: adopted (D6).**

- Values:
  - pause new work at 7,500 entries or 768 MiB pending;
  - refuse every append at 10,000 entries or 1 GiB;
  - flush every 60 s or at 1 MiB pending;
  - checkpoint every 100 updates or 5 minutes, and at controlled shutdown.
- Source:
  - `Limits::ADOPTED` and `Policy::ADOPTED` (renamed from `PROPOSED`);
  - `Config::adopted()`;
  - the report marker `APPROVAL = "adopted-by-owner-delegation-2026-09-26"` (was
    `owner-pending`).
- They stay explicit configuration. A `jope.progress.config/v1` file can set other values, and
  every report then says `adopted_defaults: false` and
  `approval: "operator-configured-not-the-adopted-defaults"`. A file cannot mark its own values
  adopted.
- Tests:
  - `progress::tests::product::outbox_limits_and_cadence_are_explicit_adopted_configuration`;
  - `progress_cli::progress_verbs_publish_engine_events_to_an_explicit_store_only`.

**One W8 update is one checkpoint window: adopted (D17).** `update_index` counts the windows of
optimizer steps between two published W8 checkpoints, and `consumed_rows` is
`(step − previous step) × global_batch`. See "The update index" above.

**The `jope-parallel` dev-dependency of `jope-learning`: accepted (D18).** It is used only by the
tests, to check their engine fixtures with the engine's own `corpus_publication::verify` and
`checkpoint_recovery::reconcile`. The library does not link the engine.

**Live S3 phase: authorized once the owner creates a scoped credential (D20).** No credential was
used here. The product path reaches an `s3://` store by configuration alone, with no code
change.

The environment it reads (`progress::remote::S3_ENVIRONMENT`, and `jope-learning progress
--help`):

| Variable | Needed for |
|---|---|
| `JOPE_LEARNING_URL=s3://<bucket>/<run-scoped prefix>` (or `--store`) | the store. A bare bucket is refused |
| `JOPE_AWS_PROFILE=<scoped publisher profile>` (or `AWS_PROFILE`) | the credential, evaluated through `aws configure export-credentials` (`jope_datasets::credentials`) |
| instead: `AWS_ACCESS_KEY_ID` + `AWS_SECRET_ACCESS_KEY` (+ `AWS_SESSION_TOKEN`) | a static or STS key |
| `AWS_REGION` | the bucket region; `us-east-2` when unset |
| `AWS_CONDITIONAL_PUT` | unset or `etag`. object_store 0.14 defaults to `etag`: `If-None-Match: *` for immutable objects, `If-Match` for the run pointer |
| `JOPE_STATE` (or `--state`) | the local state tree: outbox, HWM ledger and revocations |

`progress::remote::preflight_publishing_store` refuses each of these before any store is
opened, with nothing written:

- a bare bucket;
- `JOPE_DATASETS_ANON=1` (unsigned requests);
- `AWS_CONDITIONAL_PUT` other than `etag`.

With no credential source, opening fails fast and by name, before any request.

`progress_cli::an_s3_store_is_configuration_only_and_misconfiguration_is_refused_first` runs the
shipped binary with an empty environment and a temporary `HOME`, so no credential of this host
can reach it. It shows:

- the three refusals;
- that a well-formed `s3://` store passes the checks and stops at the credential step.

The live phase itself (W8-T07/T08) is not run here.

## W12-T05 at W8 publication (rr2/integration-a)

A commit whose closure, ancestors included, names revoked evidence never moves the run pointer,
and nothing of it is delivered. The product driver refuses to append a revoked engine
publication at all. Revocations are imported from the claims and revocation graph into the
state tree's `revocations` ledger (`jope-learning revocations import`).

Evidence of revoked items committed earlier stays on the remote: nothing is deleted. Every
later commit of that run is then refused, because it names the revoked evidence through its
ancestors. Continuing needs a new run whose closure excludes it.

The reproduction, source, tests, evidence and mutation controls are in `W03-ingress-map.md`,
"W12-T05 at the live consumers".

## PR #114 review round 1 (rr2/gap-closure)

The independent review of PR #114 (review 5328374415, REQUEST_CHANGES; receipts in
`independent-review-pr114/` beside the runs root) found one W8 defect and made one non-blocking
observation.

### P2: the S3 preflight reads the URL the store opens

**Defect.** `preflight_publishing_store` split the raw string at `://` and `/`, while
`Store::open_with` parses it with `url::Url` and builds its object prefix from the normalized
path. The two disagreed:

- `S3://example-bucket` returned early at the case-sensitive scheme check, so neither the bucket
  root nor `JOPE_DATASETS_ANON=1` nor `AWS_CONDITIONAL_PUT=disabled` was refused; the store's
  parser reads it as S3;
- `s3://example-bucket/run/..` passed, but normalizes to `/`, so the store would write at the
  bucket root.

The reviewer reproduced both with pure URL parsing (`s3-preflight-probe.rs`,
`s3-preflight-probe.log`); nothing was opened and no credential or network was used.

**Fix.** One reading of a store URL, shared by the store and the preflight:
`jope_datasets::store::StoreLocation::parse`. It does exactly what `Store::open_with` did (a bare
path becomes `file://`; `url::Url` parses and normalizes, lower-casing the scheme and removing
`.` and `..` segments, percent-encoded ones included), and for `s3://`/`s3a://` it names the bucket
and the effective object prefix (`ObjPath::parse` of the trimmed path, the bucket root when that
does not parse, as before). `Store::open_with` now takes its URL, scheme and prefix from it, with
no change in behaviour. The preflight refuses, before any store is opened, every S3 spelling
whose effective prefix is empty (or whose bucket is), and applies the unsigned-request and
conditional-write checks to every spelling the store reads as S3. The refusal names where the
store would have written.

**Regression test** (the reviewer's probe, asserting refusal):
`progress::remote::tests::every_s3_spelling_the_store_opens_goes_through_the_preflight`. No
store is opened; no credential or network is used.

- Refused as the bucket root, with a signed environment and with `JOPE_DATASETS_ANON=1` plus
  `AWS_CONDITIONAL_PUT=disabled`: `s3://b`, `S3://b`, `s3://b/`, `s3://b//`, `s3://b/.`,
  `s3://b/run/..`, `s3://b/run/%2e%2e`, `s3://b/%2E`, `S3A://b/run/./../`, `s3://b/run//1` (an
  empty segment `object_store` cannot parse, so the store writes at the root) and `s3:///run-1`
  (no bucket).
- With a run-scoped prefix, in any spelling (`s3://b/run-1`, `S3://b/run-1`,
  `s3a://b/runs/../run-1`, `S3://b/run-1/./`): accepted when signed, refused under
  `JOPE_DATASETS_ANON=1` and under `AWS_CONDITIONAL_PUT=disabled`, and the store's own reading
  gives the prefix `run-1`.
- `memory:///`, a `file://` URL and a bare path need nothing.

**Evidence** (under the evidence root; commit `dde575db81ce`):

- `pr114-r1-s3-preflight-pre-20260927T020605Z/`: the reproduction. With the test but not the
  fix, it fails and lists 20 bypasses. The upper-case `S3` and `S3A` spellings skipped every
  check, `JOPE_DATASETS_ANON=1` and `AWS_CONDITIONAL_PUT=disabled` included. The dot,
  percent-encoded dot and empty-segment spellings passed with a signed environment although the
  store would write at the bucket root.
- `pr114-r1-s3-preflight-post-20260927T020655Z/`: `jope-learning` and `jope-datasets`, 268 of
  268.
- `pr114-r1-s3-preflight-gate-20260927T021139Z/`: the commit gate. fmt, the diff check,
  coverage, the test inventory and the workspace check pass; the fence suites 85 of 85.

### Observation: where the low-level publisher refuses

The review observed that `publish::prepare` uploads a commit's immutable objects before it
verifies the closure and checks the graph rules and HWM decisions, and only then blocks the
pointer. That is correct, and some earlier wording claimed more. The precise statement, for the
low-level publisher (`publish::prepare`, `publish::publish`):

- **Refused before anything is appended or delivered:** the wrong publisher or run; a commit
  naming anything the outbox has not recorded; a commit whose closure, ancestors included, names
  revoked evidence (W12-T05).
- **Refused before the pointer advances; immutable objects may already be uploaded:** a closure
  that does not verify on the remote (missing, corrupt), a broken graph rule (forked or repeated
  update indexes, skipped heights, a checkpoint without its parent, an HWM decision from a failed
  attempt), and an HWM decision this host's ledger did not record. The objects of a refused
  commit are create-only and content-addressed, and no run pointer names them; `reusable` and
  every reader start from the pointer, so they never count as progress.
- The product driver (`progress::product::Driver::decision`) checks HWM decisions against the
  local ledger before it stages anything, so on the product path such a decision is refused
  before delivery.

Validating before upload was considered and not done here. The closure verification and the
graph rules are defined over the remote closure: they must also see ancestors a host restored
from the remote holds only there, and objects lost remotely are re-uploaded from the outbox during
verification. Moving them ahead of delivery needs a reader that overlays undelivered outbox
records on the remote, which changes the remote read pattern the offline fault matrix
(`progress::tests::faults`) is built on. That is not a contained change for a review round. The
W8-T03 task-status text now says "before the pointer advances" where it said "before delivery".

## PR #114 review round 2 (rr2/gap-closure)

The second independent review of PR #114 (review 5328684141, REQUEST_CHANGES; receipts in
`independent-review-pr114-round2/` beside the runs root) confirmed the round-1 preflight fix
across 192 `s3`/`s3a` spellings and found one W8 defect beyond them.

### P2: HTTPS URLs object_store opens on S3 passed the preflight

**Defect.** `StoreLocation::parse` named a location S3 only for the `s3` and `s3a` schemes.
`Store::open_with` sends every other URL to `object_store::parse_url`, and the locked
object_store 0.14.2 classifies AWS virtual-host
(`https://<bucket>.s3.<region>.amazonaws.com/<prefix>`) and path-style
(`https://s3.<region>.amazonaws.com/<bucket>/<prefix>`) HTTPS URLs and R2 URLs
(`https://<account>.r2.cloudflarestorage.com/<bucket>/<prefix>`) as AmazonS3, at the prefix after
the bucket. The preflight returned `Ok(())` for them as for any store that is not S3, so
`https://example-bucket.s3.us-east-2.amazonaws.com` (and its `/run/..` spelling) would have
published at the bucket root, and neither the unsigned-request nor the conditional-write check
applied. It was the round-1 failure mode (the preflight and the store reading a URL differently)
outside the two schemes round 1 covered. The reviewer reproduced it with pure URL parsing
(`https-s3-probe.rs`, `https-s3-probe.log`, six URLs); nothing was opened.

**Fix** (commit `90acc312fd38`), fail-closed in two parts:

1. **An allowlist of publishing stores.** `preflight_publishing_store` admits only `s3://` and
   `s3a://` (in any case) at a run-scoped prefix, which then go through the unsigned-request and
   conditional-write checks, and `file://` (or a bare path) and `memory:///`, which need nothing.
   It refuses every other URL before any store is opened: HTTPS that object_store opens on S3,
   Azure over HTTPS, plain `http(s)://`, `gs://`, `az://`, `abfss://` and any other backend
   (`"... storage through <scheme>://, which is not a publishing store: progress publishes only
   to ..."`). A URL object_store does not recognise (`ftp://`, `sftp://`, `file://<host>/`,
   `memory://<host>/`) is refused as not a store URL.
2. **One classification, object_store's own.** `StoreLocation` now carries the route
   `Store::open_with` takes (`StoreRoute::S3` for `s3`/`s3a`, unchanged: the S3 builder
   configured from the environment; `StoreRoute::ParseUrl` for everything else), the backend
   (`object_store::ObjectStoreScheme`, re-exported from `jope_datasets::store`) and the prefix
   the store writes under. On the `ParseUrl` route the backend and prefix are exactly
   `ObjectStoreScheme::parse` and then `Path::parse`, the two steps `parse_url` takes before it
   builds the store, and a URL the classifier does not recognise is not a store location. Every
   location on S3, by either route, reports `s3` with its bucket and prefix, and the preflight
   refuses any S3 location at the bucket root before it applies the allowlist (belt and braces):
   the four HTTPS root URLs are refused as the bucket root. `Store::open_with` takes the
   location's route, backend and prefix, so reads are unchanged for existing callers (HTTPS URLs
   still open through `parse_url`); on the `ParseUrl` route it refuses to open a store if
   `parse_url` ever names a prefix other than the location's.

No existing caller or test opens a store over HTTPS: in `jope-datasets`, `jope-learning`,
`jope-models`, `jope-cli` and `jope-arena`, `https://` appears only in metadata and source URLs.
`jope-arena`'s own `publish` opens stores with `parse_url` directly; it is not the W8 path.

**Regression tests.** Pure URL parsing: no store is opened, and no credential or network is used.

- `progress::remote::tests::only_the_admitted_store_kinds_pass_the_publishing_preflight`
  (`jope-learning`), each case under a signed environment and under `JOPE_DATASETS_ANON=1` plus
  `AWS_CONDITIONAL_PUT=disabled`:
  - the reviewer's four root URLs (virtual-host, path-style, the normalized `/run/..` spelling,
    R2) are refused as the bucket root;
  - the reviewer's two prefixed URLs, a prefixed R2 URL, Azure over HTTPS, plain HTTPS and HTTP,
    `gs://`, `az://` and `abfss://` are refused as not a publishing store;
  - `ftp://`, `sftp://`, `file://remote-host/` and `memory://remote/` are refused as not a store
    URL;
  - positive controls: `s3://b/run-1`, `S3A://b/run-1`, `memory:///`, a `file://` URL and a bare
    path pass.
- `store::tests::a_store_location_names_the_backend_and_prefix_object_store_opens`
  (`jope-datasets`): for 20 URLs (`s3`, `s3a`, `S3` with `..`; AWS virtual-host and path-style
  HTTPS at the root, normalized to the root and with a prefix; R2; `gs`; `az`; `abfss`; Azure over
  HTTPS; HTTP(S); `file`, percent-encoded included; `memory`), the location's backend and prefix
  equal object_store's classifier, the independent oracle, and a stated expectation; the route
  is S3 exactly for `s3`/`s3a`; exactly the AmazonS3 locations report `s3`, with bucket
  `example-bucket` and the same prefix. A bare path is a local store at its absolute path. The
  four unrecognised URLs are refused by both the classifier and `StoreLocation::parse`. HTTPS
  URLs still parse for the read path.
- The round-1 test `every_s3_spelling_the_store_opens_goes_through_the_preflight` passes
  unchanged.

**Evidence** (under the evidence root; commit `90acc312fd38`):

- `pr114-r2-https-s3-pre-20260927T033307Z/`: the reproduction at `09b9676b3`, with the new
  `jope-learning` test and not the fix. It fails and lists 34 bypasses, 17 URLs under each of
  the two environments, every one `Ok(())` ("passed the preflight"), the reviewer's six
  included. The first attempt, `pr114-r2-https-s3-pre-20260927T032956Z/`, is superseded: its
  list included `unix:/run/jope.sock`, which has no `://`, so the store reads it as a bare local
  path (a directory `unix:` under the working directory) and `Ok` is right for it. The test uses
  `sftp://example.com/run-1` instead.
- `pr114-r2-https-s3-post-20260927T033330Z/`: the fmt check, then `jope-learning` and
  `jope-datasets`, 270 of 270.
- `pr114-r2-https-s3-mutation-20260927T033846Z/`: the allowlist arm replaced by
  `_ => return Ok(())`. The test fails with 18 bypasses: the 9 prefixed remote URLs under both
  environments. The four HTTPS bucket-root URLs stay refused as the bucket root, because the
  location reports S3 for them (the second part holds on its own).
- `pr114-r2-https-s3-gate-20260927T033904Z/`: the commit gate. fmt, the diff checks, coverage,
  the test inventory and the workspace check pass; the fence suites 85 of 85.
- `pr114-r2-https-s3-extra-20260927T034427Z/`: the reviewer's own probes at the fix, appended
  to a working copy only and not committed. `extra-s3-probes.rs`, unchanged (192 `s3`/`s3a`
  URL variants, each signed, unsigned and unconditional, and the round-1 originals), passes, so
  the S3 route's policy is unchanged. `https-s3-probe.rs`, with only its import of
  `ObjectStoreScheme` taken through `jope_datasets::store` (`jope-learning` has no direct
  `object_store` dependency), now fails at its first assertion, `location.s3.is_none()`: the
  location reports S3 for the HTTPS URLs.

What stays open: nothing for this finding. Live S3 publication remains W8-T07/T08; no store was
opened, and no credential or network was used.
