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

## W3-T06 follow-ups (rr2/integration-a): the `hwm recover` verb and rejected redeliveries

Lane `.agent/worktrees/bench`, branch `rr2/integration-a`.

**Reproduced first** in `w3w8-verbs-pre-20260926T201229Z/`, HEAD `532fc81b0` plus the new
tests:

- `hwm::tests::redelivery_of_a_rejected_record_is_a_duplicate_not_a_second_decision` fails:
  a worse record delivered a second time was decided again as `not-improved` (expected
  `duplicate`).
- `recovery_cli::hwm_recover_is_read_only_by_default_and_rolls_forward_only_with_apply` fails
  with `error: unrecognized subcommand 'recover'`. `hwm::recover` had no product verb, although
  `freeze_teaching`'s refusals tell the operator to "run hwm recovery".

### Rejected redeliveries (`hwm::ingest`)

The ledger history now also yields, for each pointer, the records it rejected there
(`decided_events`).

- A record rejected at a pointer before, and rejected again, is recorded with transition
  `duplicate` and its current reason. It is not a second `not-improved`, `tie`, `invalid` or
  `ineligible` decision. This applies in a later batch and twice in one batch.
- A redelivery that the ledger now *accepts* is a new decision and is unaffected. The case is
  a train-side OOF record rejected as `teaching-ineligible` before its qualification was
  durable. The existing
  `eligibility::tests::linkage::qualification_without_a_durable_decision_is_not_eligible_until_qualify_resumes`
  is the positive control and still passes.
- Another record with the same worse score is still its own decision (the positive control in
  the new test).
- Old ledgers keep their meaning. Events already written are never reinterpreted, and
  `hwm verify` accepts the new events.

### `hwm recover` (read-only by default)

- **Library.** `hwm::recover_plan` reports, without writing anything or taking any lock:
  - the ledger batch a lagging `HEAD.json` would roll forward to;
  - the pointers that lag their ledger tip;
  - the pointers it would refuse.

  A chain broken in any other way is refused, as `require_intact` refuses it. `hwm::recover`
  now also rolls a lagging ledger head forward, and reports it (`ledger_head`, `applied`).
- **CLI.** `jope-learning hwm recover [--apply]` exits 1 when a pointer holds a record the
  ledger never accepted there; neither the plan nor apply touches that pointer. It has a
  capability-matrix row.
- **Test.** `recovery_cli::hwm_recover_is_read_only_by_default_and_rolls_forward_only_with_apply`
  (new target) covers:
  - a crash that left both `HEAD.json` and a pointer lagging;
  - the plan is exit 0 with the whole state tree byte-identical, and no lock file;
  - apply rolls both forward, after which `hwm verify` passes;
  - a second plan finds nothing to do;
  - a forged pointer gives exit 1 and stays byte-identical under both the plan and apply.

**Post-change run.** `w3w8-verbs-post-20260926T201520Z/` holds the full `jope-learning` run:
166 of 166 pass, 0 skipped.

## Owner decision of 2026-09-26: the HWM bar (recorded on rr2/integration-a)

**Adopted: an ineligible ledger tip still sets the HWM bar.** This is the conservative option of
the W3-T07 finding above, and the behaviour is unchanged.

- `hwm::ingest` compares a later candidate against the ledger's tip even when that tip cannot
  teach: a legacy pre-provenance acceptance, or a tip whose retained evidence is lost.
- `freeze_teaching` still excludes such a tip, so it grants no authority.
- A comparison whose tip is ineligible therefore has no teacher until a qualified candidate
  strictly beats that tip.
- Any stricter rule needs a new, versioned policy.

## W12-T05 at the live consumers (rr2/integration-a): revocation lookup

The revocation authority is the claims and revocation graph (`jope.core-ip-claims/v1`, verified
by `jope-records verify-claims`). Revoking a node there revokes exactly its descendants. Before
this change no consumer read it.

**Reproduced first** in `w12-t05-consumers-pre-20260926T223206Z/`: three temporary tests, whose
source is in that run's `diff.patch`, fail. In each, `jope_records::claims::verify` reports the
node as revoked, and yet:

- a teacher whose context parent receipt is revoked (the context receipt with it, as its
  descendant) is admitted: `qualify` succeeds, the HWM initializes, and `freeze_teaching` lists
  it;
- a commit naming a revoked corpus member moves the W8 run pointer (`Advanced`, height 1);
- a record revoked in the graph initializes its HWM pointer.

**Source.**

- `jope_records::claims::revoked_evidence` returns the revocation closure of a verified claims
  record:
  - every revoked node, with the digest its bytes were pinned at;
  - the revocation that reaches it (the lowest sequence);
  - the sha256 of the exact bytes verified. A record that changes while it is verified is
    refused.

  A record `verify` refuses yields nothing.
- `jope_learning::revocation`:
  - `import` appends newly revoked nodes to the state tree's append-only, hash-chained
    `revocations` ledger. It is read-only unless applied.
  - Nothing is ever un-revoked: a later document that no longer revokes a node is reported, and
    the node stays revoked.
  - A document that re-pins a revoked node under other bytes is refused, with nothing written.
  - `Revocations::refuse` gives the `RevokedEvidence` error, naming the node, the revocation
    (issuer, time, sequence, the ancestor that reaches it) and the requalification rule.
- Consumers:
  - **W3 teacher admission.** `eligibility::measured`, which `qualify`, `hwm::ingest`,
    `freeze_teaching`, `revalidate_qualification` and `revalidate_replay` all recompute
    through, refuses a candidate when any of these is revoked:
    - the record, freeze, experiment, window, frozen targets or panel;
    - the inner-partition evidence;
    - any fold's predictions, fitted state, request, job, result, receipts or baseline;
    - the context receipt, its fitted state or output;
    - any parent receipt, or a parent's fitted state or output. The consumer walks the parents
      itself, so a revoked parent refuses its descendant even when the graph does not name the
      descendant.
  - **HWM ingest.** A candidate whose record is revoked is rejected with the new reason and
    transition `revoked`, before any comparison. It never initializes or improves, and a
    redelivery of it is a `duplicate`. Consistent with the adopted HWM-bar decision, a tip
    revoked after it was accepted still sets the bar, and every teaching gate excludes it.
  - **W8 publication.** `publish::prepare` checks the commit's whole closure, ancestors
    included, as the outbox recorded it: each item's reference digest and the sha256 of its
    bytes. It does so before anything is appended or delivered, and a match is `Blocked` with
    nothing delivered. The product driver refuses to append a revoked engine publication.
- CLI: `jope-learning revocations import --claims <record> [--root <repo>] [--apply]` and
  `revocations status`, with capability-matrix rows.

**Tests.**

- `eligibility::tests::revocation`, with `assert_refused` meaning "no authority changed":
  - `a_revoked_parent_or_its_descendant_is_refused_at_teacher_admission` covers:
    - a revoked parent (the graph also revokes the context receipt);
    - the context receipt alone;
    - the parent alone, with no descendant in the graph;
    - one fold's fitted state;
    - the frozen targets;
    - an unrelated revocation as the positive control.
  - `a_revocation_after_admission_withdraws_the_teacher_and_is_never_undone`:
    - the teacher is admitted, then revoked;
    - `freeze_teaching` excludes it and `revalidate_qualification` refuses it, with qualified
      files, pointers and accepted events unchanged;
    - a re-import appends nothing;
    - an edited document un-revokes nothing;
    - a re-pin is refused, with the ledger unchanged.
- `hwm::tests::a_revoked_record_is_rejected_and_never_initializes_or_improves`.
- `progress::tests::receiving::a_commit_whose_closure_names_revoked_evidence_never_moves_the_pointer`
  covers:
  - a new revoked corpus member: `Blocked`, with the remote byte-identical;
  - evidence committed earlier and revoked later: the descendant commit is refused;
  - an object matched by its bytes' sha256;
  - an unrelated revocation as the positive control.
- `progress::tests::product::a_revoked_engine_publication_is_never_appended`.
- `recovery_cli::revocations_import_is_read_only_by_default_and_refuses_the_revoked_record_once_applied`:
  the plan writes nothing; apply leads `hwm ingest` to reject the record as `revoked`; a
  tampered evidence file (the claims verifier refuses it) imports nothing, with exit 2.
- `jope-records` `claims::revoked_evidence_gives_consumers_digests_attributed_to_their_revocation`.

**Evidence** (under the evidence root):

- `w12-t05-consumers-pre-20260926T223206Z/` — the three reproductions fail. This is also the
  run that adds the `jope-records` dependency; the diff is in the run.
- `w12-t05-consumers-post-20260926T223745Z/` — `jope-learning` and `jope-records`: 258 of 258
  pass, 0 skipped.
- `w12-t05-consumers-mutations-20260926T223927Z/` — eight mutations of the finished source,
  each caught and restored byte-identically:

  | Mutation | What it breaks | Caught by |
  |---|---|---|
  | `candidate-closure-unchecked` | The candidate's record, freeze, targets, … are not checked | the teacher-admission test (revoked targets) |
  | `fold-fitted-unchecked` | A fold's fitted state is not checked | the teacher-admission test (revoked fold output) |
  | `context-receipt-unchecked` | The context receipt is not checked | the teacher-admission test (receipt alone) |
  | `parents-unchecked` | The consumer does not walk the parents | the teacher-admission test (parent alone) |
  | `hwm-unchecked` | HWM ingest ignores revocations | the HWM test |
  | `publish-unchecked` | Publication ignores revocations | the publication test |
  | `driver-unchecked` | The driver appends revoked publications | the product test (a later block is not enough: nothing may be appended) |
  | `repin-accepted` | A re-pinned revoked node is imported | the after-admission test |

**Not established here (open).**

- **Revocation freshness.** A host sees only the revocations it imported. There is no
  subscription, and nothing proves this host holds the latest claims record. `revocations
  status` reports the ledger head and count, not completeness.
- **Custody evidence.** An evidence node whose custody bytes are absent on this host is still
  imported by its pinned digest, unverified here, as `jope-records verify-claims` allows without
  `--require-custody`.
- **Live stores.** Nothing here reaches a live store.

## W3-T02 / W2-T07 at teacher admission (rr2/integration-a): producer authentication and the W2 handoff

This closes the "Producer authentication" item under "Not established here (open)" above, to the
extent stated below. Before this change, teacher admission trusted the encoder receipt's own
claims. Nothing named a W2 producer, linked a W2 execution proof, or consumed the W2
`ContextHandoff` (`jope.cpu-context-handoff/v1`, consumer `teacher_admission`), although W2 issues
one for teacher admission (W02 §8.5).

**Reproduced first** in `w3-t02-handoff-pre-20260926T224712Z/`. A temporary test, whose source is in that run's `diff.patch`,
fails. It shows that the submission's context evidence was three digests the candidate chose
(receipt, fitted state, output), and that `qualify` admitted it.

**Source** (`crates/jope-learning/src/eligibility/context.rs`, `context::verify_handoff` and
`verify_execution`, run by `eligibility::measured` after the receipt and parent checks, so by
every route: `qualify`, `hwm::ingest`, `freeze_teaching`, `revalidate_qualification` and
`revalidate_replay`):

- **The freeze preregisters the producer.** `ContextPlan.producer` (`ProducerPlan`) holds:
  - the W2 release identity, field for field as `ContextRelease` serializes: the eight-digest
    encoder closure, source commit, producer executable, verification policy and reference
    manifest;
  - the typed transformation chain;
  - the W2 qualification the producer passed.

  A submission never supplies these.
- **The submission carries the handoff.** `ContextEvidence.handoff` (`HandoffEvidence`) holds the
  retained handoff record's digest and the retained pool receipts of the execution that produced
  its artifact: the job, the supervisor outcome, and the context job's experiment and original
  window.
- **Teacher admission checks the handoff:**
  - it parses field for field and refuses unknown fields;
  - the schema is `jope.cpu-context-handoff/v1` and the consumer is `teacher_admission`;
  - the release, chain and W2 qualification equal the preregistered ones
    (`ProducerNotAuthenticated`, `ContextHandoffMismatch`);
  - the handed-off `context_sha256` equals the receipt's retained output digest. This is the
    sha256 of the context's little-endian f32 bytes, the same bytes, and it is pinned on the W2
    side by a new `jope-policy` test;
  - the training scope's dataset equals the comparison's, and every other scope digest is
    well-formed;
  - no revoked evidence (W12-T05) is named: the handoff, the job, outcome, experiment and
    window, the artifact or the W2 qualification.
- **Teacher admission re-verifies the execution** from the retained receipts, making the checks
  `cpu_context_bridge::execution_attestation` makes on live files:
  - the job ran the preregistered producer executable;
  - it completed cleanly under its own job and request identity (exit 0, no signal);
  - the handoff's input is a job input, and its artifact is a declared job output that is
    retained;
  - it ran inside its original window, which opened no earlier than the freeze that
    preregistered the producer;
  - its accounting is finite, within the pool fit limit, under the per-fit memory cap.
- **The qualification records the handoff** it was admitted under
  (`Qualification.context_handoff_sha256`).

**Old formats keep their meaning.**

- Every new field is optional and absent fields are not serialized. So a freeze plan with no
  producer, context evidence with no handoff and a qualification with no handoff digest all parse,
  and re-serialize byte-identically. Freeze and evidence digests are unchanged.
- Following owner decision (d) of 2026-09-26, what they admit is quarantined:
  - A freeze with a context encoder but no producer refuses with `PreHandoffProvenance`.
  - A submission without a handoff refuses with `MissingContextHandoff`.
  - Both messages end with the requalification rule: nothing is upgraded in place, and
    requalification happens only by producing a new, fully provenanced qualification.
- A stored qualification written before this change is excluded by `freeze_teaching` with that
  message, and it is refused by both revalidations. Presenting handed-off evidence later cannot
  upgrade it, because the stored file is immutable.

**Tests.**

- `eligibility::tests::handoff` (new):
  - `teacher_admission_consumes_an_authenticated_w2_handoff`. It runs 16 negatives, each through
    `qualify` and `hwm::ingest`, inventoried, with no authority change:
    - another consumer;
    - another producer executable, encoder closure or W2 qualification;
    - another transformation chain;
    - a handoff of another context, or scoped to another dataset;
    - an unknown handoff field;
    - an execution of another executable;
    - a failed execution;
    - an artifact the job never wrote, or an input it never read;
    - no memory cap;
    - an execution window opened before the freeze;
    - no handoff;
    - a revoked handoff.

    It then runs the positive control for regression, binary and multiclass. The qualification
    records the handoff digest.
  - `a_freeze_without_a_preregistered_producer_stays_quarantined`.
  - `pre_handoff_formats_keep_their_meaning_and_stay_quarantined`.
- The W3-T07 matrix gains two classes, `missing-handoff` and `unauthenticated-producer`, each
  through all five routes, with forged success flags.
- `jope-policy`
  `context_producer::admission::handoff_tests::a_teacher_admission_handoff_serializes_as_the_record_its_consumer_parses`
  (feature `context-producer`) pins the handoff record the consumer parses: keys at every level,
  the eight encoder fields, `cut` as an integer, the consumer spelling, and the context digest
  over little-endian bytes.
- Fixtures: every fixture context is now handed off by the preregistered synthetic producer.
  Nothing is encoded or executed.

**Evidence** (under the evidence root):

- `w3-t02-handoff-pre-20260926T224712Z/` — the reproduction fails: the unauthenticated
  context was admitted.
- `w3-t02-handoff-post-20260926T225441Z/`:
  - `jope-learning` and `jope-motifs`: 279 of 279 pass, 0 skipped;
  - `jope-policy` `handoff_tests` (features `testing,context-producer`): 2 of 2 pass.
- `w3-t02-handoff-mutations-20260926T225603Z/` — fifteen mutations of the finished source, each
  caught and restored byte-identically:

  | Mutation | What it breaks | Caught by |
  |---|---|---|
  | `gate-unwired` | The handoff check is not enforced | every handoff test and both new matrix classes |
  | `consumer-unchecked` | Another consumer's handoff is accepted | the handoff test |
  | `release-unchecked` | Another release identity is accepted | the handoff test |
  | `w2-qualification-unchecked` | Another W2 qualification is accepted | the handoff test |
  | `chain-unchecked` | Another transformation chain is accepted | the handoff test |
  | `context-digest-unchecked` | A handoff of another context is accepted | the handoff test |
  | `dataset-unchecked` | A handoff for another dataset is accepted | the handoff test |
  | `executable-unchecked` | A job of another executable is accepted | the handoff test and the `unauthenticated-producer` matrix class |
  | `exit-code-unchecked` | A failed execution is accepted | the handoff test |
  | `input-unchecked` | An input the job never read is accepted | the handoff test |
  | `artifact-unchecked` | An artifact the job never wrote is accepted | the handoff test |
  | `window-unchecked` | A context executed before the freeze is accepted | the handoff test |
  | `memory-unchecked` | An execution without the pool memory cap is accepted | the handoff test |
  | `handoff-revocation-unchecked` | A revoked handoff is accepted | the handoff test |
  | `handoff-not-recorded` | The qualification does not record the handoff | the handoff test's positive control |

**Not established here (open).**

- **No signature.** The handoff and the pool receipts are content-addressed consistency
  evidence. This is the same trust level as the W3 fold worker receipts. Nothing in the product
  signs a pool execution, so internally consistent receipts fabricated for the preregistered
  producer are not detected.
- **No W2 re-verification at W3.** Teacher admission does not re-run W2 artifact verification
  (`cpu_context_bridge::load_qualified`, feature `qualified-context`, which needs `jope-policy`
  and the native reference). It authenticates the handoff that verification issues, and the
  execution behind it.
- **Row scopes are not joined.** The handoff's training-row digests are over W2's prepared-row
  encoding, not the W3 receipt's, so only the dataset is joined. A shared row-identity encoding
  would be needed (see W3-T04 below).
- **Live routes.** The live `jope-bench` driver (§B) still freezes with `context: None` and
  preregisters no producer, so it cannot admit teachers. No live W2 producer has run for teacher
  admission.
- **Controller.** No product controller issues `ProducerPlan` yet. It is supplied with the freeze
  plan, as `ContextPlan` already was.
