# W10 — Demonstrate regression learning and measure research hypotheses

Status: current per-task state lives only in the generated [STATUS.md](STATUS.md); the
checkboxes below are the original acceptance plan, not a progress record.
Owner: scientific controller/evaluation. Operational loop requires
the applicable W1–W9 controls and live gates. Studies require their own costed frozen
manifests. Follow [the execution contract](EXECUTION-CONTRACT.md) and preserve
[the regression transfer contract](../../transfer-measurement.md).

## Sources and outcome distinctions

Inspect `crates/jope-engine/src/native_loop.rs`, `native_observer/science.rs`,
`training_exposure.rs`, `crates/jope-bench/src/panel/transfer.rs`, and associated
optimization/export paths. Existing transfer-plan/report commands perform zero fits;
they do not supply an isolated executor, grants or historical exposure truth.

Operational loop completion, actual active-parameter updates, GP target improvement,
matched search outcome, transfer improvement and serving promotion are separate
milestones. Negative or inconclusive measurements can complete a study. Missing
coverage, broken isolation or unknown required exposure cannot become a complete score.

## Ordered tasks

### W10-T01 — Freeze the smallest complete operational witness

- [ ] Pin the regression release, parent policy, encoder bundle, teacher policy,
  controller/topology, corpus and checkpoint formats, export backends and S3 destination.
  Enumerate candidate/fold/update/export/restore work and all terminal outcomes in advance.
- [ ] Include a no-update control and matched parent/challenger search settings.
  Admit the proposed 60-minute global smoke window only if the complete work census fits.
- [ ] Acceptance: every prerequisite is independently verified; insufficient budget
  refuses the phase rather than omitting refits, failures, restoration or controls.

### W10-T02 — Produce candidates, qualified teachers and a frozen corpus

Depends on T01 and W3/W5.

- [ ] Run real regression search through the managed controller, retaining original and
  executable candidates, fold predictions/losses, fitted subjects and complete attempt ledgers.
- [ ] Admit teachers through mandatory provenance and metric gates; record each HWM
  disposition separately. Freeze and publish the exact qualified corpus via W8.
- [ ] Acceptance: independent replay reproduces eligibility and HWM outcomes. If no
  eligible teacher is found, report the honest terminal outcome; do not weaken admission
  to manufacture the planned learning witness or claim this chain complete.

### W10-T03 — Execute actual updates and restore the challenger

Depends on T02 and W6/W7/W8.

- [ ] Run native regression updates; retain exact consumed records, objective terms,
  supervised positions, effective weights, gradient/update outcomes and active parameter deltas.
- [ ] Commit a complete checkpoint, publish it, restore it into fresh custody and verify
  model/optimizer/scheduler/sampler state. Compare against the no-update control.
- [ ] Acceptance: active serving-relevant parameters change through real optimizer work;
  serialization-only or inactive-tensor changes fail the learning-update claim.

### W10-T04 — Complete matched search, export and S3 replay

Depends on T03 and W5/W9.

- [ ] Run matched parent/challenger/no-update searches under identical admitted logical
  budgets. Preserve every refusal, failure and timeout. Export a supported selected
  fitted predictor and verify its shipping bytes without refitting.
- [ ] Restore the complete reusable evidence graph from S3 and independently replay
  the selection, teacher, corpus, update and export links.
- [ ] Acceptance: one joined real witness closes the operational chain irrespective
  of search benefit. A missing link stays incomplete; export success alone cannot imply transfer.

### W10-T05 — Preregister conditioning and ranker studies

Depends on T04; new study identity/budget required.

- [ ] Compare correct context with diagnostic shuffled/block-perturbed contexts while
  freezing other factors. Separately train a proper program-only control at matched exposure;
  zeroing context at inference is only a diagnostic, not a fair trained control.
- [ ] Freeze a pre-gate proposal pool including rejected candidates. Evaluate ranker
  shadow ordering against declared controls without letting its ranking change pool membership.
- [ ] Acceptance: report per-stage attrition and measured losses; distinguish generator
  conditioning from ranker effects and from GP/fitting improvements. Retain negative results.

### W10-T06 — Measure reward architecture, search and curriculum hypotheses

Depends on T05; separate one-factor manifests.

- [ ] Retain order-blind reward collisions; train/evaluate an order-aware challenger
  under a new architecture identity. Do not alter accepted weights or objective silently.
- [ ] Separately compare generation-zero selection, additional fresh proposals and
  evolution; current versus diversity-aware curriculum at matched exposure; objective
  ablations/runtime-label usefulness; and duplicate-aware allocation under equal feasibility rules.
- [ ] Acceptance: each study changes one causal factor and reconciles exposures, logical
  budgets, quality, failures and tails. No omnibus run claims causal attribution for all changes.

### W10-T07 — Admit complete transfer coverage and exposure evidence

Depends on T04 and W1 history/family records.

- [ ] Preserve five outer folds and five independent audit-fold procedures plus one
  outer-training procedure per outer fold:30 complete searches per arm/dataset/seed.
  Every search independently constructs context, selects candidates and refits in scope.
- [ ] Budget at least `30 × arms × datasets × seeds` searches; with candidate cap C,
  account for up to `30 × arms × datasets × seeds × (5C + 1)` inner/final fits, plus
  encoding, search, I/O and export overhead. Record actual early stops separately.
- [ ] Verify separate checkpoint/encoder/evaluation exposure and family identities.
  Unknown required exposure blocks the current complete transfer score. Adaptive panel
  relabeling, aliases or re-encoding cannot make it untouched; development studies need a new claim.

### W10-T08 — Commit the full round before score release

Depends on T07 and W1 scorer separation.

- [ ] Retain a prediction or explicit failed/unavailable terminal outcome for every
  planned search/arm. Commit the entire round before any audit/outer score release;
  inner selection scores remain permitted within their declared search scope.
- [ ] Bind release authority to independently verified isolation/chronology and the
  exact plan/freeze, not a receipt digest recomputed from an unreviewed bundle.
- [ ] Acceptance: partial freezes do not release convenient score subsets; a sentinel
  proves held labels and prior audit/outer results cannot feed later discovery in the same round.

### W10-T09 — Independently calculate transfer and uncertainty

Depends on T08.

- [ ] For each dataset/outer-fold/seed unit, compute normalized audit and outer gains
  against each permitted training-mean baseline, take their minimum per unit, then aggregate
  units within datasets, datasets within families and families equally.
- [ ] Retain absolute skill, adverse tails, coverage, failures, both component gains,
  denominators and declared uncertainty method. Do not treat overlapping folds/seeds as
  independent replicates or the descriptive minimum score as a confidence bound.
- [ ] Acceptance: an independent scorer reproduces the report; zero/nonfinite baselines,
  missing pairs and invalid exposure block the complete headline without becoming zero gains.

### W10-T10 — Record measurement and promotion decisions separately

Depends on each completed study, not on a positive outcome.

- [ ] Publish preregistered method, actual coverage, findings, negative/inconclusive
  results and limitations. Keep operational success separate from scientific benefit.
- [ ] Apply only existing or separately preregistered promotion/tail-risk rules through
  the authorized decision owner. A checkpoint, S3 pointer or study completion cannot promote serving.
- [ ] Acceptance: every decision cites its comparison identity and evidence; historical
  unknowns remain blocked. Hand W12 the complete measured and unmeasured claim inventory.

## Verification and delivery

```sh
cargo test --locked -p jope-bench --lib panel::transfer -- --list
cargo test --locked -p jope-parallel --lib native_loop -- --list
```

Proposed slices/phases: (A) protocol/executor controls T01; (B) admitted operational
witness T02–T04; (C) separately costed causal studies T05–T06; (D) transfer admission,
isolation and scoring T07–T09; (E) decision/reporting T10. Existing reporting fixtures
do not replace actual30-search execution. Deliver complete search/update/export/restore
inventories and signed/scoped decisions, including valid negative findings.

## Original finding routes

Primary register entries: `S01-G4`, `S01-K4`, `S02-I6`, `S03-G09`, `S03-K09`, `S04-G06`, `S04-G07`, `S04-G10`, `S04-I06`, `S04-I07`, `S04-I10`, `S05-G8`, `S05-I8`, `S06-G10`, `S06-G11`, `S06-G12`, `S06-K07`, `S06-K11`, `S06-K12`, `S08-G07`, `S08-K07`, `S09-G7`, `S09-I7`, `S10-G9`, `S10-K8`.

See [the complete crosswalk](FINDINGS-COVERAGE.md) and original clauses. This primary
routing is not a waiver of cross-cutting acceptance or later live integration.
