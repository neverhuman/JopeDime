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
  **Decided 2026-09-26:** keep the tip as the bar (OWNER-DECISIONS.md D1; "Owner decision of
  2026-09-26: the HWM bar" below).

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

**Adopted: an ineligible ledger tip still sets the HWM bar** (OWNER-DECISIONS.md D1). This is
the conservative option of the W3-T07 finding above, and the behaviour is unchanged.

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
- Following owner decision D4 of 2026-09-26 (recorded on rr2/integration-a as decision (d)), what
  they admit is quarantined:
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

## W3-T03 / W3-T07 (rr2/integration-a): requalification is a refusal, the matrix covers every task, and the engine routes

### Requalification (owner decision D4, 2026-09-26; recorded on rr2/integration-a as decision (d))

**Adopted.** Pre-provenance qualifications stay quarantined. Requalification happens only by
producing a new, fully provenanced qualification. It is a refusal with a clear message, not a
procedure, and no route upgrades or re-measures a retained qualification into eligibility.
This replaces "Requalification. Not implemented … open" in the T03 migration policy above.

**Reproduced first** in `w3-t03-requalify-pre-20260926T230527Z/`. The new requalification
test runs on the unchanged source (its source is in the run's `diff.patch`) and fails. The `freeze_teaching` exclusion of a
pre-provenance qualification said only
`MissingEncoderProvenance: submission carries no context receipt`. Requalifying that record in
place was refused only as `immutable eligibility evidence already differs`. Neither message
named the quarantine or the way back.

**Source** (`eligibility.rs`, `eligibility/context.rs`):

- `eligibility::quarantined(reason)` appends one rule to every pre-provenance refusal:
  `MissingEncoderProvenance` (in both forms), and W3-T02's `PreHandoffProvenance` and
  `MissingContextHandoff`. The rule reads: "Quarantined: nothing upgrades it in place, and no
  route re-admits it. Requalification happens only by producing a new, fully provenanced
  qualification: a new record, qualified with evidence carrying this provenance under a freeze
  that preregisters it (freezes are immutable, so a comparison frozen without it needs a new
  comparison identity)".
- `qualify` refuses in-place requalification before writing, with `RequalificationRefused`. This
  covers a record whose retained qualification differs from the new measurement. The refusal
  names the record and comparison and carries the same rule. It is inventoried as a `rejected`
  decision, and the retained file is untouched.
- Unchanged: a byte-identical redelivery (a pending file gaining its decision event) is still
  idempotent.

**Tests.** `eligibility::tests::provenance::requalification_is_refused_in_place_and_happens_only_through_a_new_qualification`
takes a teacher admitted with full provenance and rewrites its retained bytes as pre-provenance.
Then:

- the `freeze_teaching` exclusion and the revalidation refusal both carry the rule;
- `qualify` of the same record with full evidence is `RequalificationRefused`: inventoried, with
  authority and the retained bytes unchanged;
- **positive control:** a new record with complete provenance qualifies. Because it strictly
  beats the quarantined tip's bar (the adopted HWM-bar decision), it becomes the teacher. The
  quarantined file stays byte-identical.

### Adversarial ingress matrix for binary and multiclass (W3-T07)

**Before:** `attack()` ran every mutation class on a regression fixture only.
`authorized_submission_passes_every_route_for_every_task` was the only binary or multiclass
coverage, and it was positive only.

**Now** every class runs through all five routes (`qualify`, `revalidate`, `ingest`, `freeze`,
`replay`) for regression, binary (AUROC) and multiclass (log-loss), each with forged success
flags.

- The `nonfinite-prediction` class writes a `null` into the first value of whatever the task
  stores: `values[0]` for regression, `probabilities.values[0][0]` for classification.
- Every other class is task-independent. Each must refuse with the same named reason for every
  task.
- The totals are 18 classes × 3 tasks × 5 routes = 270 refusals.

**Finding from the pre-change run.** Every classification cell refused, and no admission gap was
found. One cell refused for another reason than the class names. On a binary fixture with
exact predictions (AUROC 1.0), `forged-fold-score` forged 1.1, which result validation refuses
first (`AUROC exceeds one`), so the recomputation was never reached. The class now forges
within the metric's range for every task (0.9 for AUROC; 0.1 for RMSE and log-loss at 0).
Every task therefore reaches, and is refused by, the recomputation
(`execution score differs from stored predictions`). The out-of-range forgery remains refused
by result validation.

Evidence:

- `w3-t03-requalify-pre-20260926T230527Z/` — before the source change: the requalification
  test fails (no rule in the message), and the all-task matrix has 17 of 18 classes passing.
  The one failure is the binary `forged-fold-score` described above.
- `w3-t03-requalify-post-20260926T230650Z/` — `jope-learning` and `jope-motifs`: 280 of 280
  pass, 0 skipped.
- `w3-t03-requalify-mutations-20260926T230840Z/` — four mutations of the finished source, each
  caught and restored byte-identically:

  | Mutation | What it breaks | Caught by |
  |---|---|---|
  | `requalification-message-dropped` | In-place requalification reverts to the bare `already differs` | the requalification test |
  | `requalification-check-removed` | `qualify` no longer refuses before writing | the requalification test |
  | `receipt-quarantine-rule-dropped` | A missing context receipt no longer names the rule | the requalification test (exclusion message) |
  | `freeze-quarantine-rule-dropped` | A freeze without a context encoder no longer names the rule | `freeze_without_preregistered_context_encoder_is_diagnostic_only` |

### jope-engine corpus and teaching-store routes (§C): not wired, and why

These routes are not wired to the W3 checks: `teaching_store::publish`, `publish_admitted`,
`load` / `load_resident` / `read_index`, `export`, `corpus::build_impl`,
`corpus_publication::build` / `verify`, `training::rank_impl`, `champions`, and derived corpora.
Wiring them in this lane is not containable, for three reasons:

1. **There is no identity to join on.** The W3 checks key on a comparison (`HwmIdentity`) and
   a record (`record_sha256`), resolved to a retained `StoredQualification`. None of the
   engine's formats names either one. `ContextRow`, the `Encoding` sidecar, teaching shards,
   the `parallel.teaching_corpus.v1` index and champion events carry only engine-local hashes
   and a `qualification: bool`. The engine format needs a new version that names the W3
   qualification, and every writer must supply it.
2. **The dependency direction and the fence.** `jope-engine` (package `jope-parallel`) does not
   depend on `jope-learning`. `jope-learning` has `jope-parallel` as a dev-dependency (the
   owner-accepted W8 one, OWNER-DECISIONS.md D18), so the reverse edge would make a
   dev-dependency cycle. The fence
   suites (`fit_boundary`, `view_contracts`, `gp_contracts`, `gp::fit_contract_tests`) pin that
   crate.
3. **Enforcement would stop engine training, which is the owner's call.** Every engine corpus
   today is pre-provenance by construction (reason 1). Applying the requalification rule would
   quarantine all of them and stop engine training until a W2 producer and new engine formats
   exist.

**Status (reconciled with OWNER-DECISIONS.md, 2026-09-26).** This is a format follow-up, not a
policy default. The engine formats must first carry a W3 comparison and record identity to join
on (reason 1); enforcing the rule without that format change would quarantine every existing
engine corpus (reason 3). No entry in OWNER-DECISIONS.md decides it.

The CLI route is already covered. `jope-cli train corpus prepare` and `train admission` read
`freeze_teaching` / `revalidate_replay`, which recompute `measured`. So they now also enforce
the W2 handoff, W12 revocation and the requalification rule.

Still open on that route:

- The corpus `CpuProof`'s context is not joined to the admitted handoff's `context_sha256`. The
  check itself is a few lines (`jope_policy::context_sha256(token.context())` against the
  retained handoff). But it cannot be proven here: its positive path needs a fully admitted W3
  state plus a qualified W2 `CpuProof` fixture (feature `qualified-context`), and neither
  `jope-cli` nor `jope-learning` has one. It is recorded rather than added unproven.

## W3-T04 follow-up (rr2/integration-a): fitted artifacts bound to their fold's fitting input

This closes the "Fold-specific fitted state" item under "Not done" in the W3-T04 section above,
for the Rust fitted artifact on the train-side OOF route.

**Reproduced first** in `w3-t04-fitted-pre-20260926T231509Z/`. A temporary test, whose source is in the run's `diff.patch`,
fails. Fold 2's fitted artifact was re-stored stating that it was fitted on another input (one
all-row fit, held-out rows included), with every digest link kept coherent. `qualify` admitted
it: admission read only the artifact's schema, recipe and genome.

**Design.** The fit worker is deliberately given no row identities ("Scoring labels, row
identities … are deliberately absent", `RustFitRequest`), so it cannot state which rows it was
fitted on. It can digest the exact input it fitted, and the receiver can recompute that input
from the authenticated prepared development bytes.

- **`jope-pipeline` `prepared_data::fitting_input_sha256(table, targets)`** is the digest of
  `{schema: "jope.fitting-input/v1", table, targets}`: the training table and targets in row
  order, exactly as the fit worker receives them. It is serialized as a struct, so the byte
  order never depends on serde_json features. `DataPartition::fitting_input_sha256` applies it
  to a fold's training partition.
- **`jope-bench`**:
  - New schema `jope.panel-fitted-rust/v2`: `FittedRustArtifact` gains `fitting_input_sha256`,
    present exactly on v2.
  - `fit_rust` now writes v2, with the digest of the request it fitted.
  - v1 still validates, loads and predicts unchanged. The two forms cannot be mixed.
  - The private/public export boundary test gains the v2 schema string and the field name as
    canaries.
- **`jope-learning` eligibility** (`measured`, so every route):
  - Admission accepts Rust v1 or v2.
  - An artifact names a fitting input exactly when it is v2 (`FittedInputMismatch` otherwise).
  - On the train-side OOF route, a v2 artifact's input must equal
    `DevelopmentData::cut(fold).train.fitting_input_sha256()`, recomputed from the frozen
    dataset's authenticated development bytes (`FittedInputMismatch`). The bytes are parsed
    only when a v2 artifact is present, so a v1-only qualification never needs them.
  - `Qualification.fitted_inputs_bound` is `Some(true)` when all five folds are bound. It is
    absent otherwise: for any v1 fold, and for the inner route.

**Old formats keep their meaning.**

- A v1 artifact is unbound, as every artifact before v2, and is admitted as before.
- A v1 qualification serializes without the new field. Qualifications stored before this
  change therefore recompute byte-identically and still revalidate.

**Tests.**

- `eligibility::tests::fitted` (new):
  - `v2_fitted_artifacts_bind_every_fold_to_its_own_training_input`: for regression, binary and
    multiclass, admitted, bound, revalidated.
  - `a_fitted_artifact_naming_another_input_is_refused`. The negatives are:
    - another fold's input;
    - one all-row fit reused for a fold;
    - v2 naming no input;
    - v1 naming an input.

    Each runs on its own fixture, because the cases rewrite worker outputs on disk. Each goes
    through `qualify` and ingest, inventoried, with no authority change. Each is followed by a
    positive control on the same state: its properly bound submission admits with
    `fitted_inputs_bound == Some(true)`. The test ends with an unbound v1 control.
  - `v1_artifacts_stay_unbound_and_their_qualifications_keep_their_bytes`, including a v1/v2
    mix.
- `jope-bench`
  `panel::methods::tests::fitted_artifacts_name_their_fitting_input_and_v1_stays_loadable`
  covers all three tasks. A fit names the digest of its own request, another input is another
  digest, v1 round-trips and predicts identically, and mixed forms are refused.
- `jope-pipeline` `prepared_data` unit test: the digest is the fold's training input and
  changes with any row, order or target.

**Evidence** (under the evidence root):

- `w3-t04-fitted-pre-20260926T231509Z/` — the reproduction fails: the artifact naming an
  all-row input was admitted.
- `w3-t04-fitted-post-20260926T231923Z/`:
  - `jope-learning`, `jope-motifs` and `jope-bench` (default features): 387 of 387 pass, 0
    skipped;
  - `jope-pipeline` `prepared_data`: 7 of 7 pass.
- `w3-t04-fitted-mutations-20260926T232056Z/` — nine mutations of the finished source across
  the three crates, each caught and restored byte-identically:

  | Mutation | What it breaks | Caught by |
  |---|---|---|
  | `binding-unchecked` | A v2 artifact's input is not compared | the refusal test |
  | `form-unchecked` | v2 without an input, or v1 with one, is accepted | the refusal test |
  | `v2-not-accepted` | Admission accepts only v1 | all three `fitted` tests |
  | `wrong-fold-input` | The input is recomputed for another fold | all three `fitted` tests |
  | `bound-flag-forced` | `fitted_inputs_bound` is always set | the v1 test (old bytes must not change) |
  | `bound-flag-never` | `fitted_inputs_bound` is never set | the v2 and refusal tests |
  | `bench-v1-named-accepted` | A v1 artifact may name an input | the `jope-bench` methods test |
  | `bench-fit-names-other-input` | `fit_rust` names an input it did not fit | the `jope-bench` methods test |
  | `pipeline-digest-ignores-targets` | The digest omits the targets | the `prepared_data` test |

**Not established here (open).**

- **Producer claim.** The fitting-input digest is computed by the fit worker from its own
  input. It is a producer attestation, not a re-execution: a fitter that fits one input and
  names another is not detected here.
- **Other routes.** The inner route (`jope.panel-fitted-synthesized/v1` and inner Rust fits) is
  not bound, because its fitting input is the inner partition and not `cut(fold)`. It stays
  unbound, as does any v1 artifact.
- **Live producers.** Live producers now write v2 through `fit_rust`. Nothing requires v2 yet:
  consumers can require `fitted_inputs_bound == Some(true)` once v2 artifacts are the norm.
  That requirement is a policy change for the owner.
- **W2 handoff scope.** The handoff's training-row digests (W3-T02 section above) remain
  unjoined. That needs W2 to adopt the same fitting-input digest.
- **`asm`-feature tests.** `jope-bench` code behind the `asm` feature (the live teaching driver)
  compiles unchanged: no fitted-artifact literal or schema string lives there. Its tests were
  not run here; the default-feature tests were.

## W3-T02 producer side (rr2/gap-closure): the product preregisters the W2 producer and retains its handoff

The W3-T02 / W2-T07 section above made teacher admission require a W2 handoff from a producer
the freeze preregisters. It left two items open ("Live routes", "Controller"): nothing in the
product issued `ProducerPlan`, and the live `jope-bench` driver froze with `context: None`. So no
real teacher could be admitted, and nothing in the product could change that. This section
closes both items in source, to the extent stated below. The admission checks are unchanged; one
refusal message gains a clearer cause.

**Reproduced first** in `w3-t02-producer-pre-20260927T001342Z/`. A temporary test, whose source
is in that run's `diff.patch`, fails. It submits a teacher exactly as the live driver did (a
freeze with `context: None`, a submission with `context: None`), and with the most any product
caller could preregister (a context encoder without a producer, since nothing issued a
`ProducerPlan`). Both are refused: `MissingEncoderProvenance` and `PreHandoffProvenance`. The
run also records the source facts: the only product `FreezePlan` is `panel/teaching.rs` with
`context: None`, every `ProducerPlan` construction is test code, and nothing emits a
`TeacherAdmission` handoff.

### The traced path

1. **Freeze.**
   - The only product entry that freezes a teaching run is the live `jope-bench` driver.
     `panel::teaching::preregister` runs during clock-free plan construction (`optimize_cli`, and
     `origin_proof` for its single dataset). It freezes each dataset with
     `eligibility::freeze_inner_optimization` and pins the freeze in the experiment's checkpoints.
     Every ASM entry (standalone optimize, the campaign and demo drivers) reaches teaching only
     through `preregister`, then `search_dataset` and `admit_dataset`.
   - No `jope-learning` verb freezes a teaching run. The verbs are `hwm`, `scores`, `schema`,
     `progress` and `revocations`; `hwm freeze` builds the teaching manifest, which is a
     consumer.
   - No campaign or controller in `jope-campaigns`, `jope-cli` or `jope-engine` calls
     `freeze_optimization`, `freeze_inner_optimization` or `qualify`.
2. **W2 producer run.**
   - The W2 CPU context is produced by `jope-policy` `context_producer::CpuProducer`, run as the
     `jope-context-worker` executable.
   - The live driver runs it as a pool job inside the neural arm's window, in
     `panel::optimize::prepare_regression_context`. That function is called by `prepare_neural`
     (the accepted proposal) and by `experimental_proposal::prepare` (regression). It builds the
     `QualifiedContextBundle` from the actual pool receipts and verifies it.
   - `verify_qualified` (through `QualifiedContextBundle::verify`) is the only way to obtain a
     `VerifiedCpuContext`, and `VerifiedCpuContext::handoff` is the only way to obtain a
     `ContextHandoff`.
   - The neural search path (`jope-pipeline` `search_context::ContextFiles::QualifiedCpu`,
     `experimental_producer`) and `jope-cli train` re-admit the same files with
     `cpu_context_bridge::load_qualified`. They hand the context to the accepted policy proposal
     worker (`admit_for_request`), never to teacher admission. Native (GPU) contexts carry no W2
     handoff.
3. **Handoff record.** Before this change, no product code emitted a `teacher_admission`
   handoff. Now `prepare_regression_context` also emits one: it replays the frozen assembly of
   the context it just verified and writes `handoff(TeacherAdmission)` as
   `teacher-admission-handoff.json`, beside the run's `context-files.json` in
   `asm-contexts/<neural arm>/`. The bytes are `serde_json::to_vec` of the handoff, so their
   digest is `ContextHandoff::sha256`.
4. **Retained pool receipts.** `panel::teaching::admit_one` now reads that run and passes it to
   `eligibility::retain_handed_off_context`. It passes the handoff bytes, the context job
   (canonical JSON), the supervisor outcome and the window (the pool's own files, re-read
   against their pins), the experiment, the artifact the job wrote and the context bytes.
   `retain_handed_off_context` retains them in the teaching state, where teacher admission
   looks for them.
5. **Teacher admission** (`qualify` and every revalidating route) is unchanged. It authenticates
   the handoff against the preregistered producer, re-verifies the execution from the retained
   receipts and records `context_handoff_sha256`.

### Source

- **The W2 side registers the producer** (`jope-policy`
  `context_producer::ProducerRegistration::from_qualification`). The registration is read from
  the exact bytes of the W2 qualification the producer passed: the release identity (encoder
  closure, source, producer executable, verification policy, reference manifest), the typed
  transformation chain and the qualification digest.
  - Only the accepted encoder, qualified under the current verification policy, registers.
  - It serializes field for field as `ProducerPlan`, and cannot be deserialized.
  - `jope-pipeline` `CompletedQualification::producer_registration` is the loader the freeze
    uses. It re-admits the retained qualification with its existing `verify` (the eight executed
    cases, their pool receipts and the rebuilt qualification), then reads the registration from
    the qualification bytes its pin authenticates. No identity is re-derived by hand.
- **The freeze preregisters it** (`jope-bench` `panel::teaching`). `producer_plan` loads the
  configuration's W2 qualification once per plan. `teaching_context` then decides per dataset:
  - A regression dataset, whose configuration's producer registered, is frozen with a
    `ContextPlan`. Its encoder is the producer executable under its asset configuration (seed 0,
    `teaching_context_2048`), its source family is `jope.cpu-context/v1`, and its producer is
    the registration, converted field for field.
  - Every other dataset is frozen with `context: None`, as before: a classification dataset (the
    accepted W2 CPU context is regression-only), no configured qualification, one its loader
    refuses, or an unavailable producer runtime. Teacher admission then refuses its teachers
    with the existing `MissingEncoderProvenance` message.
  - The registry records the choice either way: `Registration::Frozen.context` is
    `producer { qualification_sha256 }` or `no_context { reason }`. The field is optional and
    absent on registries written before this change, which keep their bytes.
- **The run emits the handoff** (`panel::optimize::prepare_regression_context`), as in item 3 of
  the traced path. It replays the same frozen assembly that the policy proposal's own admission
  replays (`load_qualified`, `admit_for_request`), so a context the proposal can use can also be
  handed off.
- **The controller retains the run and issues the receipt** (`jope-learning`
  `eligibility::retain_handed_off_context`, `ContextRun`).
  - It refuses before retaining anything when the freeze preregistered no context encoder
    (`MissingEncoderProvenance`) or no W2 producer (`PreHandoffProvenance`), or when the handoff
    is not a `jope.cpu-context-handoff/v1` record.
  - It issues the context's `jope.encoder_receipt/v1` under the freeze's encoder and scope (for
    the inner route, the inner partition's fitting rows, with the outer hold forbidden). The
    fitted state is the artifact the job wrote. The input closure is the handoff's own input
    (`scope.input_sha256` and `scope.training.training_sha256`). The receipt has no parents.
  - It decides nothing and changes no authority state.
- **The driver submits it** (`panel::teaching::admit_one` and `handed_off_context`). When the
  freeze preregistered a producer, the submission carries the retained context.
  - If the neural arm emitted no handoff (no context was produced), the submission carries none,
    and teacher admission refuses (`MissingEncoderProvenance: submission carries no context
    receipt`).
  - A handoff whose run files are unreadable is an admission error: retryable inventory, never a
    context.
- **Receiving path.** `verify_execution` now names a missing or unreadable retained pool receipt
  (`UnprovenContextExecution: a retained pool receipt of the context job is missing or
  unreadable`) instead of the generic blob message, which it keeps as the cause. The
  preregistered-plan, producer and handoff-parse checks are factored out for the issuer, with
  their messages unchanged.

**Operator verb: none.** The plan is never issued by hand. The only product freeze of a teaching
run issues it from its own frozen configuration, so there is no capability-matrix row.

**Old formats keep their meaning** (owner decision (d)). Freezes, registries and qualifications
made before this change keep their bytes and stay quarantined. A comparison already frozen with
`context: None` cannot be re-frozen with a producer, because freezes are immutable ("optimization
freeze is already pinned to different inputs"). Its dataset's registration is then `unavailable`
with that reason, and requalification needs a new comparison identity.

### Tests

- `jope-learning` `eligibility::tests::producer` (new, 5 tests). The producer run is a CPU
  fixture: synthetic pool receipts and a synthetic context, shaped as the live driver retains
  them.
  - `a_teacher_is_admitted_through_the_product_path` (positive control):
    - the freeze preregisters the producer parsed from its registration record;
    - `retain_handed_off_context` retains every byte admission reads, with no authority change;
    - `qualify` admits the teacher, recording the handoff and receipt digests;
    - `hwm::ingest` accepts it, and `freeze_teaching` and `revalidate_qualification` agree;
    - reissuing the run is idempotent.
  - `the_product_path_refuses_an_unauthenticated_or_unproven_context`, 9 negatives, each
    through `qualify` and `hwm::ingest`, inventoried, with no authority change:
    - a run of a producer the freeze did not preregister;
    - a handoff under another W2 qualification;
    - a handoff for another consumer;
    - a substituted context;
    - a run that exited after its original window;
    - a run in a window opened before the freeze;
    - missing pool receipts: the supervisor outcome, the context job, and the artifact the job
      wrote.
  - `without_a_preregistered_producer_nothing_is_issued_and_no_teacher_is_admitted`. The
    no-context freeze and a freeze with an encoder but no producer: nothing is issued or
    retained, and the submission without a context is refused with the existing messages.
  - `a_pre_handoff_qualification_is_not_upgraded_by_the_product_path` (owner decision (d)):
    `RequalificationRefused`, stored bytes and authority unchanged, the teaching manifest still
    excludes it.
  - `the_issued_context_is_bound_to_the_inner_scope_the_live_driver_freezes`: the receipt
    carries the inner scope and verifies for the inner comparison only.
- `jope-policy` `context_producer::admission::handoff_tests::the_preregistered_producer_is_the_producer_that_hands_off_to_teacher_admission`
  (new). A context verified under the qualification bytes hands off to teacher admission under
  exactly the registered release, chain and qualification. The registration's keys are exactly
  `ProducerPlan`'s. It refuses bytes other than the pin, another protocol or schema, another
  encoder, a malformed producer digest and an unknown field.
- `jope-bench` `panel::teaching::tests` (3 new, feature `asm`):
  - `a_context_is_preregistered_only_from_a_registered_w2_producer`:
    - a registration is converted field for field into `ProducerPlan`, crossing both crates;
    - the regression plan's producer, encoder and source family;
    - `no_context` with its reason for binary and multiclass, a missing qualification, one the
      loader refuses, and an unavailable producer runtime.
  - `registrations_record_the_teaching_context_and_old_ones_keep_their_bytes`.
  - `a_run_without_a_teacher_admission_handoff_yields_no_context`.
- Fixture change: `HandoffParts` carries the artifact bytes, and `retain_handoff` retains them
  (instead of `handoff_parts`), so a run can be built without retaining anything.

**Evidence** (under the evidence root):

- `w3-t02-producer-pre-20260927T001342Z/`: the reproduction fails. The live driver's teacher is
  refused with `MissingEncoderProvenance`, and the best a caller could preregister is refused
  with `PreHandoffProvenance`.
- `w3-t02-producer-gate1-20260927T002409Z/` and `w3-t02-producer-gate2-20260927T002812Z/` are
  the pre-commit gates of the `jope-policy`/`jope-pipeline` commit and the `jope-learning`
  commit. Both are green. Each ran:
  - fmt and the diff check;
  - the workspace check, with and without `jope-cli/asm`;
  - the coverage validation and the test inventory;
  - the fence suites, 85 of 85;
  - the suites of the commit's own crates.
- `w3-t02-producer-post-20260927T003118Z/` is the post-fix run and the gate of the `jope-bench`
  commit. Every step exits 0:

  | Step | Result |
  |---|---|
  | The new and changed tests | 15 of 15 |
  | fmt, diff check, workspace check (with and without `jope-cli/asm`), coverage, test inventory, parity matrix | pass |
  | Fence suites | 85 of 85 |
  | `jope-learning` | 187 of 187 |
  | `jope-motifs` | 101 of 101 |
  | `jope-bench` (default features) | 106 of 106 |
  | `jope-bench` (feature `asm`) | 172 pass, 3 skipped |
  | `jope-policy` (default features) | 41 pass, 2 skipped |
  | `jope-policy` (`testing,context-producer`) | 106 pass, 8 skipped |
  | `jope-policy` doctests | 10 of 10 |
  | `jope-pipeline` (default features) | 95 pass, 2 skipped |
  | `jope-pipeline` (`qualified-context`) | 145 pass, 2 skipped |
  | clippy on the four crates with their live features | exit 0; no warning on a changed line |

  The skipped tests are the crates' existing ignored tests.
- `w3-t02-producer-mutations-20260927T003823Z/`: mutations of the committed source. Each was
  applied with an exact-match replacement, run against its tests, and then restored
  byte-identically:

  | Mutation | What it breaks | Result |
  |---|---|---|
  | `issuer-outcome-not-retained` | The supervisor outcome is not retained | caught by the positive control, the negatives, the inner-scope and quarantine tests |
  | `issuer-labels-unbound` | The receipt drops its label input | caught by the positive control |
  | `issuer-skips-producer-check` | A freeze without a producer is issued | caught by the no-producer test |
  | `receipts-message-dropped` | A missing receipt loses its named cause | caught by the missing-receipt negatives |
  | `bench-classification-gets-producer` | A classification dataset is preregistered a producer | caught by the `jope-bench` context test |
  | `bench-unavailable-runtime-ignored` | An unavailable producer runtime is preregistered | caught by the `jope-bench` context test |
  | `bench-encoder-not-the-producer-v2` | The encoder names another executable | caught by the `jope-bench` context test |
  | `policy-encoder-not-held` | Another encoder registers | caught by the `jope-policy` registration test |
  | `policy-chain-stale` | The registered chain is not the handed-off chain | caught by the `jope-policy` registration test |
  | `policy-pin-unchecked` | Bytes other than the pin register | caught by the `jope-policy` registration test |
  | `bench-no-context-not-recorded` | `preregister` records no context choice | **not caught**: no test runs `preregister` (see below) |

  The first form of the encoder mutation also matched the test's own assertion, so it was not
  applied. It was rerun with a pattern that occurs only in the source, as `-v2`.

### Not established here (open)

- **No live run.** No W2 producer has run for teacher admission, and no real teacher has been
  admitted. The live run needs all of the following:
  - a completed W2 CPU qualification (`Configuration.qualification`): eight cases executed
    through the pool against the native reference manifest (pinned registry `22b5de5f…`, index
    `6c81da15…`), with the accepted encoder assets (Hyperion weights and manifest, component
    bundle, frozen transforms) and the `jope-context-worker` built at the configuration's
    source. W2-T07 records this as not run.
  - an `ExecutionPool` with cgroup v2 delegation (`/sys/fs/cgroup/system.slice/fit-<job>`) for
    the context job and the teacher's folds;
  - a teaching configuration: a durable state root outside checkouts, the prepared root and the
    approved frozen panel;
  - a new comparison identity for any dataset already frozen with `context: None`.

  All of it runs on CPU (the producer is Candle CPU FP32), but it needs the live assets and a
  delegated pool, which this lane does not have.
- **The `jope-bench` wiring is not exercised end to end.** `preregister` and `admit_one` need a
  frozen campaign plan, the approved panel and pool journals, so no test runs them. The mutation
  `bench-no-context-not-recorded` is therefore not caught. The pieces they call are tested: the
  registration, its conversion, `teaching_context`, the registry, `handed_off_context`, the
  issuer and admission. The one seam no single test crosses is minting a real
  `VerifiedCpuContext`, which needs the pinned native reference bytes. The two sides of that seam
  are pinned against each other, in `jope-policy` and in the `jope-bench` conversion test.
- **Unchanged from the section above:**
  - the handoff and pool receipts are unsigned, content-addressed evidence;
  - teacher admission does not re-run W2 `load_qualified` (the live driver's emission comes
    from the bundle it just verified);
  - handoff row scopes are not joined (dataset only).

## PR #114 review round 1 (rr2/gap-closure): the handed-off context and its window are the producer's own

The independent review of PR #114 (review 5328374415, REQUEST_CHANGES; receipts in
`independent-review-pr114/` beside the runs root) reproduced two acceptances at teacher admission
with a target-only probe harness (`adversarial-probes.rs`, `probes-pinned.log`). Both are fixed at
the receiving path, `eligibility/context.rs` (`verify_handoff`, `verify_execution`), each in its
own commit, and each probe is now a permanent regression test asserting refusal. Every refusal
below goes through `qualify` and `hwm::ingest`, is inventoried, and changes no authority state
(no qualification, no HWM record, no pointer, no teaching-manifest entry). The positive control
(`a_teacher_is_admitted_through_the_product_path`) still admits.

### P1: the consumed context is the one the producer artifact contains

**Defect.** Admission compared two digests the submission controls. `retain_handed_off_context`
issued a receipt for whatever `run.context` it was given, and `verify_execution` only checked that
`handoff.artifact_sha256` names a retained output of the context job. Nothing decoded the context
from those artifact bytes, and `admission_pins_sha256` was checked for its form only. The
reviewer kept the producer artifact, the job and the supervisor outcome byte-identical, replaced
only the submitted 2048-f32 vector (finite, clipped values) and `handoff.context_sha256`, and
`qualify` admitted the teacher and `hwm::ingest` accepted it.

**Fix** (in `verify_handoff`, after the execution proof):

- **The context comes from the artifact.** `produced_context` reads the retained artifact
  (`State::read_blob` refuses bytes that do not hash to their address), bounded at 16 MiB as
  `verify_qualified` bounds it, and decodes it as the W2 producer does:
  - the bytes parse as `CpuContextArtifact`, field for field (`ProducedArtifact`; unknown fields
    are refused);
  - the envelope is W2's: its schema and backend, `qualification: "unqualified"`, no final test
    targets;
  - it was produced under the handoff's release (source, producer executable, encoder closure)
    and input scope (input, training identity and its fitting rows, preprocessing, prepared
    rows);
  - its `vector` is 2048 finite values inside the frozen clip (|v| <= 32).

  The consumed vector (the receipt's retained output) must then be exactly that vector's
  little-endian bytes, and `handoff.context_sha256` their digest. Otherwise:
  `ContextHandoffMismatch: the handed-off context … is not the context the producer artifact
  contains (…)`.
- **The admission pins are rebuilt, not read.** `admission_pins_sha256` must be the digest of the
  controller's `AdmissionPins` for this run. Teacher admission rebuilds them field for field
  (`AdmissionPinsRecord`):
  - from the freeze's preregistration: the release (source, producer, reference manifest,
    verification policy), the W2 qualification digest, and the controller's pin of each of the
    eight qualification case executions;
  - from the authenticated run: the artifact digest, the input, training identity and
    preprocessing, and `context_execution_sha256`. That is the digest of the controller's
    attestation of the context execution (`ExecutionAttestationRecord`), which
    `verify_execution` now derives from the verified receipts as `ExecutionProof::attest` does:
    the supervisor receipt is the retained outcome file's digest, then the program, input and
    output, the per-fit memory allocation, swap, the wall time and the per-fit allowance.

  Otherwise: `ProducerNotAuthenticated: the handoff's admission pins … are not the controller's
  admission pins … for this run`.
- **The case-execution pins are preregistered.** The W2 qualification fixes them, so the
  registration records them:
  - `jope-policy` `ProducerRegistration` gains `qualification_execution_sha256`: eight hex
    digests, one per qualification case. Incomplete or malformed pins do not register.
  - `jope-pipeline` `CompletedQualification::case_execution_attestations` derives them from the
    case pool receipts. `producer_registration` (the freeze) and `jope-bench`
    `prepare_regression_context` (every producer run) both take them from it, so the two cannot
    disagree.
  - `ProducerPlan` gains the same field, optional. A freeze made without it keeps its bytes, and
    nothing under it is issued or admitted: `PreHandoffProvenance: the freeze's W2 producer
    registration records no controller admission pins …`, quarantined as owner decision D4
    requires.
- **Both sides are pinned.** `jope-policy`
  `handoff_tests::the_records_teacher_admission_rebuilds_are_pinned` and `jope-learning`
  `eligibility::tests::producer::the_w2_records_teacher_admission_rebuilds_are_pinned_field_for_field`
  pin the same bytes for `AdmissionPins` and `ExecutionAttestation` (placeholder values, exact
  serialization order), and the same `CpuContextArtifact` field set, schema and backend.
  `jope-pipeline` `cpu_context_bridge::tests::the_attestation_is_derivable_from_the_retained_receipts`
  pins, on the producer side, every attestation field teacher admission derives.

**Tests** (`eligibility::tests::producer`, new):

- `a_substituted_context_with_a_matching_handoff_digest_is_refused`: the reviewer's probe. The
  artifact, job and outcome are asserted byte-identical to the producer run's.
- `an_artifact_that_decodes_to_another_context_is_refused`: the job wrote, and the handoff and
  pins name, an artifact whose vector differs in one finite, clipped value.
- `a_truncated_or_corrupt_artifact_is_refused`: a truncated artifact, an unknown field, another
  schema, final test targets used, another input, a value outside the clip, a short context, and
  an artifact corrupted on disk.
- `admission_pins_the_controller_did_not_issue_are_refused`: pins under other case executions,
  pins over another record of the context execution, a digest that names no pins, a freeze that
  preregistered other case pins, and a registration without pins (nothing issued or retained).
- `the_w2_records_teacher_admission_rebuilds_are_pinned_field_for_field` (above).

The fixture's artifact is now a `jope.cpu-context/v1` artifact holding the handed-off context,
and its handoff names the controller's pins, written out by the fixture independently of the
receiver. Fixture contexts stay inside the frozen clip.

### P2: the context execution is the teacher's own authorized execution

**Defect.** The only link between the context execution and the teacher's freeze was a lower
bound: the context window had to open after the freeze. The experiment and window loaded from
`context.handoff.execution` were never joined to the teacher's `OofEvidence.experiment_blob_sha256`
and `window_sha256`. The reviewer moved the context window and its execution 120,000 ms later,
leaving the teacher's original 60,000-ms window and all its evidence unchanged: `qualify`
admitted the teacher and `hwm::ingest` accepted it. A context produced entirely after the
teaching deadline acquired teaching authority.

**Fix.** The freeze already models the teacher's authorized execution: `qualify` holds the
teacher's experiment and original window to the freeze (the experiment pins the freeze and
vocabulary, and the window opens after the freeze and belongs to the experiment). So the context
execution is joined to exactly that, and no freeze field is added.

- `qualify` reads the teacher's experiment and window before the context check and passes them
  in (`context::Teaching`). The context check still runs before the inner binding, the folds and
  any blob retention.
- `verify_execution` requires the context job's experiment identity and window (experiment,
  start and deadline) to be the teacher's. It runs after the existing lower bound:
  `UnprovenContextExecution: the context was produced in experiment … over [start, deadline] ms,
  not in the teacher's authorized experiment … and window […] ms`.
- With the existing timing check (the job exits by the window's deadline less its reporting
  allowance), a context produced after the teaching deadline never acquires teaching authority.
- In the live driver the context job and the teacher's folds run in the same campaign experiment
  and journal: `journal/window.json` is both the teacher's `window_sha256` and the context proof's
  window. So the join holds there by construction.

**Tests** (`eligibility::tests::producer`, new):

- `a_context_produced_after_the_teaching_window_is_refused`: the reviewer's +120,000 ms probe.
- `a_context_from_another_experiment_is_refused`: a context from another experiment, over the
  teacher's window times.
- `a_context_window_that_ends_after_the_teaching_deadline_is_refused`: a context from an
  experiment with a longer allowance, whose window ends after the teaching deadline and whose job
  exits after it.

The tests that call `context::verify` directly pass the teacher's window: the inner-scope tests
pass the window that opened with the inner freeze. The existing before-the-freeze and
escaped-window refusals keep their messages.

### Evidence (under the evidence root)

- **P1** (commit `d36cdab6aee9`):
  - `pr114-r1-handoff-context-pre-20260927T013224Z/`: the reproduction. With the new fixture
    and tests but not the fix, the four refusal tests fail because `qualify` admits (the log
    shows each returned `Qualification`); the other 62 eligibility tests pass on the new
    fixture.
  - `pr114-r1-handoff-context-post-20260927T014011Z/`: `jope-learning`, 192 of 192.
  - `pr114-r1-handoff-context-gate-20260927T014544Z/`: the commit gate. fmt, the diff check,
    coverage, the test inventory and the workspace check pass; the fence suites 85 of 85; the
    feature receivers (`jope-policy` `testing,context-producer`, `jope-pipeline`
    `qualified-context`, `jope-bench` `asm`) 425 pass, 13 skipped (their existing ignored
    tests).
- **P2** (commit `a5f71dfeef74`):
  - `pr114-r1-context-window-pre-20260927T015558Z/`: the reproduction. The three refusal tests
    fail because `qualify` admits; the other 67 eligibility tests pass. An earlier run,
    `-015432Z`, held both experiment cases in one test, so only its first case ran; they are
    now two tests.
  - `pr114-r1-context-window-post-20260927T015750Z/`: `jope-learning`, 195 of 195.
  - `pr114-r1-context-window-gate-20260927T020227Z/`: the commit gate. fmt, the diff check,
    coverage, the test inventory and the workspace check pass; the fence suites 85 of 85.
- The re-authoring of the branch before these fixes is recorded in
  `pr114-r1-reauthor-20260927T012953Z/`.

### Not established here (open)

- **Unchanged from the sections above:** the handoff and pool receipts are unsigned,
  content-addressed evidence. A handoff can no longer be rewritten beside its producer run, but a
  submitter who fabricates a whole consistent run (artifact, job, outcome) is bounded only by the
  preregistered release, qualification and case-execution pins, not by a signature. Teacher
  admission decodes the artifact and holds it to its envelope and layout; it does not re-run W2
  `load_qualified` (the frozen assembly is not replayed from the artifact's components). Handoff
  row scopes are not joined (dataset only).
- **No live run.** As in the section above: no W2 producer has run for teacher admission, and no
  real teacher has been admitted.
