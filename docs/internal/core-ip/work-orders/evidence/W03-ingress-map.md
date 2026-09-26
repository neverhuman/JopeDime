# W3-T01 — Teaching, HWM and corpus ingress map

Task: W3-T01, with the W3-T02/T03 receiving-gate change on branch `rr/w3-provenance`
(base `354962c5c`, change `d63dd2c35`). Owner: Claude implementation subagent (lane `.agent/worktrees/honesty`).
Observed states: jope-learning route **reproduced**, **implemented**, **fixture-proven**
locally. Nothing here is `CI-verified`, `live-qualified` or a closure of any register entry.
Every other route below is **mapped only**. Its enforcement is unchanged by this branch.

Line anchors are for this branch's working tree. Anchors under `crates/jope-learning` are
post-change. Anchors under `crates/jope-engine`, `crates/jope-cli`, `crates/jope-motifs`
and `crates/jope-bench` are as read on this branch.

## Definitions

- **Authority-bearing state.** Any of the following:
  - a stored qualification `eligibility/qualified/<comparison>/<record>.json`;
  - an accepted HWM event plus its `current.json` pointer;
  - membership in `freeze_teaching`'s manifest;
  - membership in a training corpus, or a training/optimizer input derived from it.
- **Not authority.** These never make a teacher eligible:
  - `rejected` decisions in the `eligibility` and `hwm` ledgers (the submitted-record
    inventory);
  - content-addressed blob custody (`State::put_blob`);
  - lock files.
- **Context provenance.** A retained `jope.encoder_receipt/v1` bound to:
  - the consumer's comparison identity;
  - the frozen fitting/forbidden membership;
  - the preregistered encoder (artifact, configuration, seed, product);
  - the retained fitted state and final context bytes;
  - a retained parent-receipt chain inside the same scope.

## A. jope-learning (the route changed by this branch)

| Entry point | Role | Last validation before first authority mutation | Context provenance before → after |
|---|---|---|---|
| `freeze_optimization` / `freeze_inner_optimization`, `eligibility.rs:440`, `:443`; `inner.rs:157` | Preoptimization freeze, controller-owned | `task()` `:300`, label role, authenticated rows, fold/group checks. Then writes the burn receipt `:527`, freeze file `:556` and ledger `frozen` `:558`. | Before: none. After: an optional `FreezePlan.context` (`ContextPlan`: source family and encoder pins) is validated at `:449-451` and frozen into `FrozenPolicy.context`. A freeze without it is diagnostic-only for teaching. |
| `register_inner_candidate`, `inner.rs:187` | Ordering marker before outer scoring | `load_freeze` plus required blobs. Then writes the candidate freeze `:232` and ledger `:233`. | Not a teaching decision, so no context is required there. |
| `qualify`, `eligibility.rs:1042` | **Live admission** | `measured(.., live=true)` `:1046`. The new `context::verify` `:744` runs right after the freeze/guard load and before experiment, inner-bind, fold or blob-retention work. The first authority write is `immutable_write(qualification_path)` `:1058`. The decision ledger append `:1064` records `qualified` or `rejected` (inventory). | Before: **none**, so any receipt was ignored and a missing receipt passed (reproduced below). After: **mandatory**. See `context.rs:134-170`. Missing, altered or wrong-scope provenance is `rejected` and inventoried; no qualification file is written. |
| `hwm::ingest`, `hwm.rs:197` | HWM ingestion | For `train_oof` comparisons, `require_qualified` `hwm.rs:265`. That is `eligibility.rs:1074` recomputing `measured()` from the *stored* evidence, which includes `context::verify`. Then comes the ledger append `hwm.rs:309` and pointer write `:320`. | Before: none. After: re-verified on every ingest. A missing, corrupt or lost context blob gives `teaching-ineligible` with no pointer (warm/retry revalidation). |
| CLI `jope-learning hwm ingest`, `main.rs:195-243` (champions, genebank, snapshot) | Historical import | Source parsers, then `hwm::ingest`. Every source writes `comparison: None` (`sources/champions.rs:81`, `snapshot.rs:220`, `genebank.rs:53`). | Legacy diagnostic pointer space. It can never enter `freeze_teaching` (`eligibility.rs:1161-1165` refuses a legacy comparison). Unchanged. |
| `freeze_teaching`, `eligibility.rs:1137` | **Teaching freeze** (read-only; consumers persist its manifest) | Per pointer: exact accepted event, stored qualification, then `measured(.., false)` `:1177`. | Before: none. After: re-verified. A legacy or missing-provenance entry is an `exclusions` row (reason `MissingEncoderProvenance: ...`), never an entry. |
| `revalidate_replay`, `eligibility.rs:1102` | Historical replay | Accepted history plus `revalidate_qualification` `:1118` → `measured` `:1026`. | Re-verified. A legacy qualification fails replay. |
| `revalidate_qualification` / `require_qualified`, `:1023` / `:1074` | Resume/verification | `measured(.., false)`. | Re-verified from retained bytes. |
| `hwm::freeze`, `hwm.rs:1069` | Diagnostic HWM archive | `require_intact`. | `teaching_eligible: false`. Not a teaching path. Unchanged. |

**Receiver derivation** (`eligibility/context.rs`). The submission supplies only
`ContextEvidence{receipt_sha256, fitted_state_sha256, output_sha256}`. From that:

1. The receiver reads the scope from the frozen policy (`scope`, `:45`):
   - For an inner partition, the fitting rows and groups are `fitting_rows`/`fitting_groups`,
     `hold_rows` are **forbidden**, and `outer_fold` is `Some`.
   - For a whole-development freeze, the fitting rows are the frozen targets, forbidden is
     empty, and `outer_fold` is `None` (new `encoder_receipt::Scope` semantics).
2. It binds `EncoderCapability` with the **preregistered** encoder.
3. It calls the existing `encoder_receipt::check_teaching_binding` with:
   - consumer comparison = `record.identity_sha256`;
   - program = the record's program;
   - product = `TeachingContext2048`.
4. It checks that the retained fitted and output bytes rehash exactly.
5. It then resolves parents (`parents`, `:97`):
   - at most 8 levels and 64 receipts, with a visited set;
   - each parent must be canonical, retained, and in the same scope and comparison.

The receipt must be canonical content (`canonical_sha256(receipt) == receipt_sha256`).

## B. Consumers of the jope-learning manifest (already enforced, unchanged)

- **`jope-cli` `train corpus prepare`** (`train/corpus.rs:504`).
  - It reads `freeze_teaching` (`:530`).
  - It requires an inner-OOF partition (`:391-395`) and one `CpuProof` per admitted
    comparison (`:565-569`).
  - It verifies every proof before the first write (`:603-607`; first write `directory`
    `:612`) through `verify_context` (`:381`), which covers exact fitting rows and values,
    `load_qualified` and encoder identity.
  - **This is an existing corpus-loading safeguard.**
  - **Gap (open):** it does not require the proof's context bytes to equal the admitted
    `Qualification.context_receipt_sha256` output. Admission and corpus contexts are linked
    only by the comparison digest.
- **`jope-cli` `train.rs` `admission`** (`:634`).
  - It calls `freeze_teaching` `:660`, or `revalidate_replay` `:693` on compare/replay.
  - Contexts go through `qualified_cpu::verify` `:1289`.
  - Tolerances reported by the map (not changed here):
    - the projection digest is waived on replay;
    - legacy split-level contexts are allowed when `inner_partition` is null;
    - `synthetic_fixture` categories skip context verification;
    - the execution window is reserved (`:901-908`) before `admission()`.
- **`jope-motifs` `learning.rs` / `teaching.rs`.** Read-only consumers of `freeze_teaching`.
  `teaching.rs:417-457` refuses an entry without a comparison context.
- **`jope-bench` `panel/teaching.rs` (live driver).**
  - `preregister` `:158` freezes with `context: None` (`:220`), because this route has no
    preregistered context encoder.
  - `admit_one` `:304` submits `context: None` (`:534`) to `qualify` `:536`.
  - **Result after this branch:** every admission on this route is refused as
    `MissingEncoderProvenance` and is kept as non-terminal `teaching-inventory`
    (`admit_dataset` `:564`). It is not bypassed.
  - **Open.** It needs an authenticated W2 context producer to run *before* admission, and a
    freeze that preregisters its encoder.

## C. jope-engine (package `jope-parallel`) corpus and teaching-store routes — missing enforcement, not changed

None of these routes calls jope-learning admission. Their context check is at most a
self-declared sidecar. This is where the recurring CRITICAL finding still applies.

| Entry point | Role | Last validation before first durable mutation | Context provenance |
|---|---|---|---|
| `teaching_store::publish`, `teaching_store.rs:273` | Legacy publication | Only `ensure!(qualification)` `:284-287`, then `publish_impl` with no admission manifest (`reference_v1`, `midrank_v2`, `frozen_z_v1`). | Only the ContextRow file hash `:457` and `Encoding.encoder_spec_sha256 == ci.assembly_sha256` `:462-463`. With `qualification=true` the context-index flag is ignored (`:388-392`). |
| `teaching_store::publish_admitted` → `publish_impl`, `:302` / `:327` | Label-admitted publication | Store dir and lock at `:376-381` come before any authority check. The sqlite binding row at `:505` comes before per-row validation. Rows go through `label_admission::verify_inputs` / `validate_receipt` / `validate_preverified` (`:593-612`), then the transaction commits at `:841`. | `verify_inputs` never checks the Encoding sidecar or the assembly. Failing rows are quarantined, not aborted. |
| `pack_group` reuse, `teaching_store.rs:143-147` | Resume | Only the `verify_shard` content hash. | None. |
| `teaching_store::load` / `load_resident` / `read_index`, `:1074` / `:1040` / `:1079` | Training loader | Legacy `parallel.teaching_corpus.v1` branch `:1089-1109` has no provenance. The sharded branch checks internal hashes only. | Contexts get a partition check only (`:1141-1144`). |
| `teaching_store::export`, `:1283` | Compatibility copy | Output must not exist, then `load`. It copies shard labels without checking `labels_sha256` (`:1298-1310`). | None. |
| `corpus::build_impl` / `build_observed_cached`, `corpus.rs:677` / `:653` | Reference corpus build | Only `ensure!(qualification)` `:663`/`:688`, then the context loop `:723-744`. | Same sidecar-only check. `scan_ledger` `:547-550` skips the provisional-row check when `qualification=true`. |
| `corpus_publication::build` / `verify`, `corpus_publication.rs:67` / `:14` | Publication and resume | Existing provenance returns early at `:109-111`. Staging is reused at `:117-121` then `:166`. | Hash consistency only. |
| `training::rank_impl`, `training.rs:760` | Corpus consumer | Qualification flag gate `:815-820`. Writes `initialization.json` `:1210-1212` **before** the context re-hash `:1213-1234`. | Partition and hash only. No encoder or admission check. |
| `champions::record_inner` / `import` / `freeze_snapshot_inner`, `champions.rs:420` / `:747` / `:532` | Teacher seeds | Identity format and score, then a pending write. | None. The snapshot rewrites `evaluator_sha256` (`:687-689`). |
| `continuation_corpus::build`, `corpus_projection::project` | Derived corpora | Render/validate after `create_dir_all`. Projection validates **after** its rename (`:231` then `:233-240`). | New context indexes drop `assembly_sha256`. |

This table is from a read-only source trace (Explore subagent, 2026-09-26). No reproduction
has been run for these routes on this branch.

## Reproductions and evidence

The evidence root is
`/home/ubuntu/jope-rescue-20260921/regression-remediation-20260926/claude-w0-w12/runs/`.

- **Pre-fix** `w3-prefix-missing-provenance-20260926T051459Z/`: HEAD `354962c5c`, `diff.patch`,
  untracked test source, `nextest.log`.
  - `eligibility::tests::provenance::missing_context_provenance_is_refused_at_real_teaching_ingress`
    **FAILED**.
  - A teacher with no context provenance produced a qualification file, an HWM pointer, an
    accepted event and a teaching-manifest entry. The snapshot changed from empty to one of
    each.
  - Pre-fix, altered or wrong-scope receipts could not even be submitted: the ingress had no
    provenance input.
- **Post-fix** `w3-postfix-mandatory-provenance-20260926T052124Z/`: HEAD `354962c5c` plus
  `diff.patch`. That patch is byte-identical to the staged content of commit `d63dd2c35`. `nextest.log` shows 85 of 85
  `jope-learning` tests passing with 0 skipped. `doctest.log` shows 1 of 1 passing.
  `check.log` shows `cargo check -p jope-bench -p jope-motifs --all-targets` passing. It includes these tests in
  `crates/jope-learning/src/eligibility/tests/provenance.rs`:
  - `missing_context_provenance_is_refused_at_real_teaching_ingress`: the negative case plus
    an authorized positive control on the same state.
  - `wrong_scope_consumer_encoder_or_altered_context_is_refused_before_authority`. It runs
    17 negatives, each with an unchanged authority snapshot, an inventoried `rejected`
    decision and an expected reason:
    - other cut;
    - other dataset;
    - reordered, missing or extra-forbidden rows;
    - family;
    - encoder artifact, seed and product;
    - altered output and fitted state;
    - noncanonical, unretained or path-shaped receipt;
    - unretained output;
    - unresolved or out-of-scope parent.

    It then runs the positive control.
  - `freeze_without_preregistered_context_encoder_is_diagnostic_only`.
  - `retained_context_is_revalidated_by_every_receiving_gate`: corrupt, restore and delete a
    retained context blob, checked across `hwm::ingest`, `require_qualified`,
    `revalidate_qualification` and `freeze_teaching`.
  - `legacy_qualification_without_provenance_stays_quarantined_and_is_not_upgraded`:
    pre-change bytes are excluded, but the pointer and event stay readable. Valid evidence
    cannot upgrade them in place, because `immutable_write` refuses.
  - `inner_partition_scope_forbids_outer_hold_rows`: helper-level for the inner route. The
    full inner `qualify` path lacks a fixture.

## Migration and quarantine policy (T03)

- **Stored `StoredQualification` written before this branch.** It has no
  `OofEvidence.context` and no `Qualification.context_receipt_sha256`.
  - It still deserializes (serde default), so it is readable.
  - It fails `measured()`, so every gate excludes or refuses it.
  - No default value is invented.
- **Freezes written before this branch.** They have no `context`, so they are diagnostic-only
  for teaching. Their `plan_sha256` is unchanged: an absent field is not serialized.
- **Requalification.** Not implemented.
  - A retained legacy decision cannot be overwritten, because the qualification path is
    immutable per (comparison, record).
  - Requalifying needs a reviewed, versioned procedure that consumes original context
    evidence and keeps the legacy file. This is open.
- **Legacy incumbent risk.** A legacy HWM pointer without provenance still holds its
  comparison's pointer. A new qualified candidate must strictly beat it: this is
  conservative, not permissive. Deciding whether a quarantined incumbent should stop
  defining the bar is W3-T05/T06 scope.

## Related: W3-T05 explicit HWM transitions (fixture-proven, same branch)

Eligibility stays separate from HWM state. `qualify` writes no HWM event, and `hwm::ingest`
now records exactly one `Transition` per candidate:

- `initialized`
- `improved`
- `tie`
- `not-improved`
- `noncomparable`
- `ineligible`
- `invalid`

`IngestReport.transitions` counts them. `hwm::verify` rederives `initialized` and `improved`
from ledger order, independent of any recorded flag, and reports `transition-mismatch`.
The existing `decide` rule, guard and pointer movement are unchanged.

Evidence:

- **Pre-fix** `w3-t05-prefix-transitions-20260926T052543Z/`: both tests FAILED. The events
  carried no transition (`[Null, ...]`), so a bootstrap was indistinguishable from an
  improvement.
- **Post-fix** `w3-t05-postfix-transitions-20260926T052648Z/`: `jope-learning` 87 of 87 pass,
  with 0 skipped.

Tests:

- `hwm::tests::transitions_separate_initialization_ties_non_improvement_and_improvement`
- `eligibility::tests::transitions::eligibility_alone_and_redelivery_never_count_as_improvement`

Not done at T05 (the first three were later addressed, see T04 and T06 below):

- per-fold adverse-tail diagnostics;
- idempotent delivery identity: a redelivered record is a `tie`, not a distinct `duplicate`;
- race and crash reconciliation (T06);
- whether a quarantined legacy incumbent should stop defining the bar.

## W3-T04 recomputed metrics and diagnostics (fixture-proven)

`measured()` already recomputed every fold loss and the train-only baseline from retained
target and prediction bytes. It also already required exact row order, five frozen folds in
fold order, and a clean bound worker execution whose predictions and fitted state are among
its outputs. It refused a forged fold score (`result.primary`) and a forged aggregate
(`record.score.value`).

The pre-fix run established two missing pieces:

- **Score units were not bound.** A regression record relabelled `normalized_rmse` or
  `dimensionless` was qualified, and because the qualification is keyed only by record blob,
  a relabelled record could ride a stored qualification into the HWM.
  - Now `record.score.units` must be `original_target_units` for RMSE and `dimensionless`
    otherwise, and it is re-checked on every revalidation.
- **No separately named diagnostics existed.** `Qualification.diagnostics`
  (`jope.teaching-diagnostics/v1`) now carries:
  - per-fold scored rows;
  - candidate minus baseline loss per fold, adverse-fold count and worst fold delta (the tail);
  - for regression only: row-weighted pooled RMSE, pooled baseline RMSE, and pooled R² about
    the pooled held-out mean. It is `None` when that mean has zero spread.

  Admission still reads only `fold_losses` under the unchanged guard: five folds, four
  improved, the same margins, and mean-fold loss.

Evidence:

- **Pre-fix** `w3-t04-prefix-metrics-20260926T054825Z/`: 46 of 48 lib tests passed and 2
  failed:
  - `forged_units_cannot_relabel_a_measured_score`: the relabelled record was qualified.
  - `pooled_rmse_r2_and_tail_are_separately_named_diagnostics`: `diagnostics` was `Null`.

  In the same run, `independent_evaluator_agrees_with_mean_fold_admission_on_uneven_folds` and
  `prediction_evidence_substitutions_fail_before_authority` **passed**. They document existing
  safeguards that are now under regression protection.
- **Post-fix** `w3-t04-postfix-metrics-20260926T055008Z/`: `jope-learning` 91 of 91 pass with
  0 skipped. It also holds `doctest.log` and a `check.log` for jope-motifs, jope-cli and
  jope-bench across all targets.

Test detail (`crates/jope-learning/src/eligibility/tests/metrics.rs`):

- **Independent evaluator.** Written apart from `score_predictions`, `baseline` and
  `feat_math`. On uneven folds (2/4/3/5/2 rows, heterogeneous residuals, one adverse fold) it
  reproduces the per-fold losses, the mean-fold aggregates, the four-of-five count and the
  pooled figures. It also shows that the pooled RMSE differs from the admitted mean-fold loss.
- **Substitution suite.** Each case is refused with an exact reason and no change to
  authority-bearing state, and the suite ends with a positive control. The cases are:
  - swapped fold order;
  - a duplicated fold;
  - a missing fold and a sixth fold;
  - a duplicated row ID;
  - another fold's fitted state, both naming only and a coherent rebind;
  - missing predictions;
  - a forged fold score and a forged aggregate;
  - a nonfinite (`null`) prediction.

Fixture changes, which supply valid values rather than weaken any check:

- Records carry correct units.
- Each fold has its own fitted state.
- `fixture_spec` adds uneven folds and custom predictions.

Not done:

- **Fold-specific fitted state.** A fitted artifact is not bound to its fold's training rows.
  Both artifact schemas carry no row identity, and the worker envelope `input_sha256` is not
  interpretable in jope-learning. A producer that reuses one all-row fit for every fold
  cannot be detected here.
- **Classification metrics.** Validity outcomes and class-permutation tests for binary and
  multiclass are deferred with the register scope.

## W3-T06 recoverable decision linkage (fixture-proven)

Two crash windows existed between durable writes. Both were reproduced by removing exactly
the later step's durable effect: the appended batch plus its `HEAD.json`, or the pointer
write.

- **`qualify`.** The immutable qualification file is written before its `qualified` decision
  event. After a crash in between, the file alone qualified the teacher:
  - `hwm::ingest` accepted it;
  - `freeze_teaching` would have listed it.
- **`hwm::ingest`.** The batch is appended before pointers move, and later ingests decided
  against whatever pointer was on disk. After a crash in between:
  - a candidate weaker than the ledger's HWM was accepted against the stale pointer;
  - `freeze_teaching` presented a teacher the ledger had already superseded;
  - an acceptance whose first pointer write was lost silently vanished from the manifest;
  - a pointer holding a record the ledger never accepted was used as the incumbent.

Behaviour now:

- **Qualification linkage.**
  - The `jope.teaching-decision/v1` event carries `qualification_sha256`, the digest of the
    stored qualification bytes.
  - `require_qualified`, `revalidate_qualification`, `revalidate_replay` and
    `freeze_teaching` require a durable `qualified` decision for (comparison, record,
    digest). Until one exists, a qualification file is pending.
  - Redelivering `qualify` completes a pending link. Redelivering an already durable decision
    is idempotent and appends no second event.
- **HWM linkage.**
  - An HWM event records the `qualification_sha256` its train-side decision relied on.
  - `freeze_teaching` requires the ledger tip's digest to equal the stored qualification's.
- **Ledger-resolved incumbents.** `hwm::ingest` resolves every touched pointer against the
  ledger's accepted history before writing anything.
  - The incumbent is the ledger tip.
  - A lagging pointer (missing, or an earlier accepted record) is rolled forward in the
    pointer phase and counted in `IngestReport.pointers_recovered`.
  - A pointer holding a record the ledger never accepted at that pointer fails the ingest
    before any ledger or pointer write.
- **Recovery.** `hwm::recover` rolls every lagging exact-identity pointer forward. It reports
  unaccepted pointers and leaves them untouched.
- **Explicit duplicates.** A record already accepted at the pointer is a `duplicate`
  transition and is never accepted again. The ledger `reason` stays `tie` or `not-better`.
- **Visible incompleteness.** `freeze_teaching` admits a pointer only when it equals its
  ledger tip. It reports as exclusions both a pointer that lags its tip and an accepted
  teaching tip that has no pointer.

Evidence:

- **Pre-fix** `w3-t06-prefix-linkage-20260926T060249Z/`: 6 of 55 lib tests failed:
  - `qualification_without_a_durable_decision_is_not_eligible_until_qualify_resumes`
  - `crash_between_hwm_append_and_pointer_advance_never_lists_a_superseded_teacher`
  - `first_acceptance_lost_before_its_pointer_is_visible_not_silently_absent`
  - `hwm::tests::a_pointer_behind_the_ledger_is_rolled_forward_before_any_decision`
  - `hwm::tests::a_pointer_the_ledger_never_accepted_is_refused_before_any_write`
  - `hwm::tests::redelivery_of_an_accepted_record_is_a_duplicate_not_a_tie`

  `stronger_and_weaker_teachers_race_in_both_orders_to_the_same_state` **passed** pre-fix. It
  covers both sequential orders and a two-thread race, and records an existing serialization
  safeguard.
- **Post-fix** `w3-t06-postfix-linkage-20260926T060641Z/`: `jope-learning` 99 of 99 pass with
  0 skipped. It also holds `doctest.log` and `check.log` (jope-motifs, jope-cli and
  jope-bench, all targets).

Test changes:

- The T05 redelivery expectations changed from `tie` to `duplicate`. This is intended: the
  ledger reason is unchanged.
- The eligibility fixture gained `Fixture::candidate` for a second teacher on the same
  frozen comparison.

Not done:

- There is no CLI verb for `hwm::recover`.
- **Rejected-decision redelivery is not deduplicated.** Every submission stays in the
  rejected inventory.
- **Mirror recovery is out of scope.** A mirror that holds a lagging pointer catches up only
  when a later batch accepts at that pointer; remote durability is W8.
- **Corpus linkage beyond `freeze_teaching` is not covered.** That includes the jope-cli
  corpus writer and the jope-engine teaching store; both are outside this crate.

## W3-T07 ingress adversarial matrix (fixture-proven, test-only)

`crates/jope-learning/src/eligibility/tests/matrix.rs` crosses the five jope-learning teaching
ingresses with 16 submission mutation classes. Each class gets its own fresh fixture.

**Routes.** `qualify`, `revalidate_qualification`, `hwm::ingest`, `freeze_teaching` and
`revalidate_replay`.

**Stored-route attack.** The stored routes are attacked with a forged but internally
consistent success claim, appended through the real ledger API:

- the mutated evidence, stored with the valid submission's own measurement;
- a `qualified` decision bound to that file's digest;
- an accepted HWM event carrying that digest, plus its pointer.

Every route must recompute and refuse without trusting those recorded flags.

**Checks per class.**

- `qualify` refuses with the class's expected reason. The rejection is inventoried, and no
  qualification, pointer, accepted event or teaching entry changes.
- `revalidate_qualification` refuses the forged claim.
- `hwm::ingest` rejects it as `teaching-ineligible`, with no authority change.
- `freeze_teaching` excludes the forged pointer and ledger tip.
- `revalidate_replay` refuses the forged claim.
- The independent replay returns nothing, even though the forged flags stand in both ledgers
  and on disk.

**Mutation classes.**

| # | Class | What is mutated |
|---|---|---|
| 1 | `missing-context` | context receipt omitted |
| 2 | `wrong-consumer-context` | receipt bound to another consumer dataset |
| 3 | `altered-context` | context output bytes changed |
| 4 | `unresolved-parent` | parent receipt missing; this stands in for revocation, since no revocation source exists |
| 5 | `missing-fold` | one fold's predictions absent |
| 6 | `repeated-fold` | one fold's predictions duplicated |
| 7 | `wrong-fold-order` | folds out of order |
| 8 | `duplicated-row` | a row ID repeated |
| 9 | `nonfinite-prediction` | a `null` prediction |
| 10 | `wrong-fitted-subject` | another fold's fitted state, coherently rebound |
| 11 | `forged-fold-score` | a fold score forged |
| 12 | `forged-aggregate-score` | the aggregate score forged |
| 13 | `relabelled-units` | score units relabelled |
| 14 | `wrong-comparison-identity` | the record's comparison identity changed |
| 15 | `substituted-freeze` | a different optimization freeze substituted |
| 16 | `wrong-program` | the record's program changed |

**Counts.** 16 classes × 5 routes = 80 refusals.
`every_mutation_class_is_run_through_every_route` asserts that every declared class has a
test and that class names are unique, so nothing is skipped silently.

**Positive controls.** `authorized_submission_passes_every_route_for_every_task` admits the
authorized submission through all five routes for regression, binary and multiclass (15 route
executions).

**Independent replay.** It is written apart from `hwm::decide`, `hwm::verify` and
`freeze_teaching`. It replays the original HWM events in ledger order and ignores every
recorded outcome, transition, displaced link, improvement counter and pointer. For each
comparison it keeps the earliest record with the lowest recomputed mean fold loss. It agrees
with the manifest:

- on each authorized control;
- in `independent_replay_reproduces_the_winner_in_either_arrival_order`, which delivers a
  weaker teacher, a stronger teacher and an unqualified candidate in both arrival orders;
- in every attack, where both return nothing.

Evidence is in `w3-t07-matrix-20260926T061453Z/`, run at HEAD `0f4d3a253`. Only test code
changed, so this run is both the pre-change observation of unchanged source and the
post-change proof:

- `matrix.log` holds 19 of 19 passing tests, run with `--no-capture`. It prints one refusal
  reason per cell: 80 lines, 16 per route. It also prints three authorized-task lines and the
  matrix summary. The diff it ran against is `diff-matrix-run.patch`, which is before
  rustfmt.
- `nextest.log` holds the full `jope-learning` run after formatting: 118 of 118 pass with
  0 skipped. `diff.patch` is byte-identical to the working tree.

No route defect was found.

Finding (policy decision, not changed here): an ineligible ledger tip still defines the HWM
bar.

- `freeze_teaching` excludes it, so it grants no authority.
- `hwm::ingest` still compares later candidates against it. An eligible teacher that does
  not strictly beat it is `not-improved`, and that comparison then has no teacher.
- An ineligible tip can arise from:
  - a legacy pre-provenance acceptance;
  - later loss or corruption of retained evidence;
  - forged ledger state.
- The independent replay's eligibility-first rule would pick the best eligible teacher
  instead. This is exactly the open item "whether a quarantined legacy incumbent should stop
  defining the bar". It needs an owner decision and, if changed, a new policy version.

## Not established here (open)

- **Producer authentication.** The receipt is a producer claim: nothing executes or signs
  the encoder run. W2 execution proofs (`CpuProof` / `load_qualified`) are not linked to
  `EncoderReceipt`.
- **Input digests.** `Inputs.features_sha256` and `labels_sha256` are not recomputed by the
  receiver. That needs the W2 projection convention.
- **Revocation.** Revocation lookup for receipts and parents does not exist (W12).
- **Section C routes.** The jope-engine corpus and teaching-store routes, champions, derived
  corpora and training loaders are not changed.
- **Live driver.** The `jope-bench` live driver can no longer admit teachers until a context
  producer runs before admission.
- **Live evidence.** No live, three-node or remote (W8) evidence exists.
