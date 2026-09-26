# W8-T01 — Durable artifact and event inventory

Task: W8-T01 (slice A, first half). Owner: Claude implementation subagent in the lead's
integration lane `.agent/worktrees/bench` (branch `feat/regression-remediation`).
Observed state: **implemented**, **fixture-proven** locally. This is a **new capability**:
no durable progress inventory existed in `jope-learning` before this slice, so there was no
receiving path to reproduce a defect against. Nothing here is `CI-verified` or
`live-qualified`, touches a remote store, or closes a register entry
(`S01-G7`, `S01-K7`, `S08-G03`, `S08-K03` stay open).

Source: `crates/jope-learning/src/progress.rs`, `progress/inventory.rs`, `progress/census.rs`.
Tests: `progress/tests/{census,encoding,fixture}.rs` (lib target, no new test target).

## Inventory

Every durable object is one `Kind`. The schema string is also the hash domain:
`digest = sha256(schema ‖ 0x00 ‖ canonical bytes)`. The store key is
`progress/objects/<dir>/<digest>.json`.

| Kind | Schema / hash domain | Class | Names (dependencies) |
|---|---|---|---|
| `RunManifest` | `jope.progress.run-manifest/v1` | authority | nothing |
| `Event` | `jope.progress.event/v1` | authority | run manifest; a retry's failure event; a completion's result blob |
| `CandidateEvidence` | `jope.progress.candidate-evidence/v1` | evidence | completion event, record blob, retained artifacts |
| `HwmDecision` | `jope.progress.hwm-decision/v1` | evidence (link to the W3 ledger) | completion event, record blob |
| `CorpusManifest` | `jope.progress.corpus-manifest/v1` | authority | member blobs, parent corpus, admitting HWM decisions |
| `Exposure` | `jope.progress.exposure/v1` | authority | consumed corpus, starting checkpoint |
| `Checkpoint` | `jope.progress.checkpoint/v1` | authority | exposure, corpus, parent checkpoint, state blobs |
| `Commit` | `jope.progress.commit/v1` | authority | run manifest, parent commit, every added object |
| `Summary` | `jope.progress.summary/v1` | derived (rebuildable, never authority) | the commit it was derived from |

Two further items are not content-addressed objects:

- `Head` (`jope.progress.head/v1`) is the run's one mutable pointer, at
  `progress/runs/<run_id>/HEAD.json`. It is canonical too. W8-T03 advances it conditionally.
- Blobs keep the existing `blobs/sha256/<hex>` layout (`State::blob_rel`).

**Event bodies.** An event is one of:

- `scheduled` — attempt 1 only, recorded before execution.
- `retry` — attempt ≥ 2. It names the digest of the prior attempt's `failed` event.
- `started` — a worker id.
- `failed` — a class (`timeout`, `preflight-refusal`, `parse-rejection`,
  `validation-rejection`, `worker-error`, `worker-lost`, `missing-input`, `cancelled`), a
  `retryable` flag and a detail.
- `completed` — the verified immutable result blob.
- `excluded` — attempt 0, with a reason.

**Separate event types.** "GP target improved" is an `HwmDecision`. "Policy updated" is an
`Exposure` plus its `Checkpoint`. Transfer improvement has no kind (W10), so it cannot be
recorded as either.

## Identities

- **Logical work.** `WorkSpec::work_id` is the domain digest (`jope.progress.work-id/v1`)
  of `{run_id, spec}`. The spec is stage, subject and sorted input digests. It never
  depends on an attempt. The manifest refuses a unit whose `work_id` its own spec does not
  produce, and it refuses the same logical work planned twice.
- **Attempt.** `(work_id, attempt)`.
- **Delivery.** The object digest. The same bytes delivered twice are one object.

## Canonical encoding and limits

- **Encoding.** Compact JSON with sorted keys (`canon::canonical_json`), and no floating-point
  fields at all. Every optional field is written as an explicit `null`. Every set is sorted
  and duplicate-free.
- **Decoding.** `decode` re-encodes and requires byte equality. It therefore refuses:
  whitespace, key order, unknown fields (top-level and in the event body), an omitted `null`,
  a float in an integer field, and unsorted or duplicated sets.
- **Limits.**
  - Identifiers: plain path segments of at most 128 bytes.
  - Digests: lowercase sha256.
  - `source_commit`: 40 or 64 lowercase hex.
  - Free text: 1–1024 bytes without control characters.
  - Lists: at most 4096 entries, and at most 65,536 planned units.
  - Attempts: at most 64. `max_attempts` must be in 1..=64.
  - Whole object: at most 4 MiB.
- **Retention.** `full` must carry artifacts. `compact` carries none: a smaller record, never
  a dropped one.
- **Commits.** A commit has a parent exactly when its height is above 0. It never adds a
  commit, summary or run manifest.

## Census reconciliation (acceptance)

`census::reconcile(manifest, events, decisions)` is pure and independent of arrival order.

**Dispositions.** Each planned unit ends up in exactly one of these.

- Terminal:
  - `completed` — exactly one completion.
  - `failed` — the last attempt failed, and either it was not retryable or the budget is
    exhausted.
  - `excluded`.
- Not terminal:
  - `in-flight`
  - `awaiting-retry`
  - `unscheduled`

The census is `reconciled()` only when every unit is terminal and there are no violations.

**Violations.** Each is reported and counted, never resolved by picking a winner:

- `foreign-run`
- `unplanned-work`
- `attempt-without-schedule`
- `conflicting-schedules`
- `retry-without-failure`
- `retry-after-terminal-failure`
- `attempt-budget-exceeded`
- `conflicting-starts`
- `conflicting-outcomes`
- `second-completion`
- `work-after-completion`
- `conflicting-exclusions`
- `excluded-work-was-scheduled`
- `decision-without-completion`
- `second-decision`

**Counts** (`Counts`, also the body of a derived `Summary`):

- planned, completed, failed, excluded, in-flight, awaiting-retry and unscheduled units;
- attempts and retries;
- failures by class, including failures that were later retried;
- exclusions by reason;
- counted decisions by transition, and improvements by logical unit;
- deliveries and duplicate deliveries;
- violations;
- `coverage` (terminal / planned) and `completion` (completed / (planned − excluded)).

A `Fraction` with an empty denominator has no rate (`rate() == None`).

**W3 link.** `HwmDecision::verify_against(&Ledger)` requires all of the following:

- the HWM chain verifies (`require_intact`);
- the batch is in the chain, with the recorded file hash and id;
- the event at `seq` names the linked record, identity, transition and qualification digest.

## Tests (`cargo nextest run -p jope-learning --lib progress`)

`progress::tests::census`:

- **`planned_census_reconciles_to_terminal_dispositions`** (positive control). Five planned
  units, a budget of 3 attempts:
  - completed on the first attempt;
  - a timeout, then a retry that completed;
  - a preflight refusal (terminal);
  - excluded for a missing paired arm;
  - three retryable failures, exhausting the budget.

  Results: 7 attempts, 3 retries, 5 failures by class, 1 improvement plus 1 initialization,
  coverage 5/5, completion 2/4. Reversed arrival order gives an identical census.
- **`a_failed_or_retried_attempt_cannot_disappear`**. Each case drops or forges one or more
  deliveries. It must leave the census unreconciled, and must not raise coverage or
  completions:
  - dropping a retried failure gives `retry-without-failure`;
  - dropping the retry gives `attempt-without-schedule`;
  - dropping a terminal failure, the last exhausted failure or a whole trailing attempt
    leaves the unit `in-flight`;
  - dropping a whole unit leaves it `unscheduled`;
  - dropping an exclusion restores the completion denominator;
  - a retry that borrows another unit's failure is refused;
  - retrying a non-retryable failure gives `retry-after-terminal-failure`;
  - exceeding the budget is refused;
  - events from a foreign run or for unplanned work change no planned unit.
- **`a_retried_attempt_cannot_become_a_second_logical_improvement`**:
  - a failed attempt that also "completed" gives `conflicting-outcomes`,
    `second-completion` and `work-after-completion`;
  - a decision drawn from a failed attempt gives `decision-without-completion` and
    `second-decision`;
  - the same completion decided twice gives `second-decision`;
  - improvements stay at 1 throughout.
- **`transport_replay_has_no_effect_but_a_new_attempt_counts`**:
  - delivering everything twice changes only `deliveries` and `duplicate_deliveries`;
  - an explicit retry is a new counted attempt.
- **`empty_denominators_are_unavailable_not_rates`**. An exclusion cannot hide a unit that
  was scheduled.

`progress::tests::encoding`:

- **`logical_work_identity_is_separate_from_attempt_and_delivery_identity`**.
- **`canonical_encoding_is_unique_and_bounded`**.
- **`hash_domains_separate_kinds_and_blobs`**.
- **`every_kind_round_trips_and_names_its_dependencies`**. It covers all nine kinds, and a
  one-bit corruption of each is refused.
- **`hwm_decision_link_is_checked_against_the_real_hwm_ledger`**. It runs two real
  `hwm::ingest` batches (initialized, then improved). Seven forged claims are refused:
  - an initialization claimed as an improvement;
  - another record;
  - another identity;
  - a missing event;
  - another batch's bytes;
  - an unrelied qualification;
  - a batch outside the chain.

  Refusals write nothing. A tampered batch verifies no decision.

## Evidence

The evidence root is
`/home/ubuntu/jope-rescue-20260921/regression-remediation-20260926/claude-w0-w12/runs/`.

- **`w8-baseline-20260926T062059Z/`** — HEAD `c4f60b4c7`, no W8 change. `nextest.log` shows
  118 of 118 `jope-learning` tests passing, with no progress tests (the pre-state of a new
  capability).
- **`w8-t01-mutations-20260926T063358Z/`** — mutation controls. Each mutation was applied to
  the finished source, and the files were restored byte-identically afterwards (sha256
  checked). Each must fail at least one test:
  - `a` counts every bound decision. It fails
    `a_retried_attempt_cannot_become_a_second_logical_improvement`.
  - `c` treats a last attempt with no outcome as terminally failed. It fails three census
    tests, including `a_failed_or_retried_attempt_cannot_disappear`.
  - `d` drops the canonical re-encode check in `decode`. It fails
    `canonical_encoding_is_unique_and_bounded`.
- **`w8-t01-postfix-20260926T063508Z/`** — HEAD `fc1f7ec4d` plus this slice's working tree.
  - `diff.patch` holds the tracked `lib.rs` change and every new file, with its sha256 in
    `diff.sha256`. `diff-check.txt` is empty.
  - `nextest.log` shows 128 of 128 `jope-learning` tests passing with 0 skipped: 118
    existing plus 10 new.
  - `doctest.log` and `check.log` (`cargo check -p jope-motifs -p jope-cli -p jope-bench
    --all-targets`) are alongside.

## Not established here (open)

- **Durability.** No durable outbox, delivery or publication yet (W8-T02/T03). The inventory
  is a contract, and `reconcile` a pure check. Nothing is persisted or uploaded by this slice.
- **Producers.** Nothing in the product emits these objects yet.
  - The engine's `native_persistence.rs`, `corpus_publication.rs` and
    `checkpoint_recovery.rs` are unchanged.
  - Mapping their manifests and checkpoints onto `CorpusManifest`, `Checkpoint` and
    `Exposure`, including real blob closure, is an engine-side follow-up.
- **Worker results.** `Completed.result` is a digest the receiver must have verified. Worker
  result verification (W7) is not implemented here.
- **Exposure contiguity.** `Exposure` and `Checkpoint` do not yet enforce contiguous
  `update_index` across parents. Checking that needs the parent objects (W8-T03 closure
  verification).
- **Live proof.** No live S3, credential or budget. W8-T07/T08 need a scoped publisher
  credential and an approved storage budget.
