# W1 — Seal regression inputs and enforce permitted access

Status: current per-task state lives only in the generated [STATUS.md](STATUS.md); the
checkboxes below are the original acceptance plan, not a progress record.
Accountable subsystem: preparation/data admission; consumers:
context production, fitting, scorers and learning workers. Read the
[execution contract](EXECUTION-CONTRACT.md). W0's applicable boundary/fixture tests
must execute honestly before this package can claim fixture qualification.

## Starting points and invariants

Inspect `crates/jope-engine/src/prepare.rs`, `crates/jope-engine/src/families.rs`,
`crates/jope-engine/src/label_admission.rs`, `crates/jope-engine/src/native_input_cache.rs`,
`crates/jope-policy/src/context_producer/input.rs`, and
`crates/jope-bench/src/panel/inference_benchmark.rs`. These are candidate modification
sites, not proof that every receiving path is already enforced. Trace wrappers,
worker arguments, caches and resumes before choosing the actual patch boundaries.

Preserve original labels, original thirty-dataset panel and historical reports.
Derive a new immutable ten-regression release. Acquisition is outside the learning
worker. A missing exact object fails; no local cache, alternate version, `latest`
pointer or network fallback may replace it. Adaptive development evidence and
fold-pure evaluation of the whole discovery procedure remain different claims.

## Ordered tasks

### W1-T01 — Inventory input consumers and release sources

- [ ] Trace each panel entry from catalog/raw object through preparation, encoder,
  GP/fitter, scorer, teacher and trainer. List files opened, parent receipts, cache
  keys, retry resolution and environment-derived roots at each receiver.
- [ ] Identify the exact ten regression members and prove membership against the
  original panel; record every classification/time-series exclusion without changing
  the old panel. Produce `consumer-map` and `release-source-inventory` receipts.
- [ ] Acceptance: every numerical input has an accountable acquiring component,
  permitted consumer and byte identity; unexplained access routes block T03/T05.

### W1-T02 — Freeze the regression release and family registry

Depends on T01.

- [ ] Bind raw/prepared bytes, target identity, ordered row IDs, row/target alignment,
  schema, groups, folds, preparation source/configuration and original label role.
  Specify canonical encoding and unknown-field/version handling before serialization.
- [ ] Version aliases, alternate targets, resampling, shared entities and unresolved
  family relationships. Track encoder exposure and checkpoint exposure separately;
  permit `unknown` but never equate it with `unseen`.
- [ ] Acceptance: changed source, target, rows, schema or family mapping produces a
  new release identity or rejection. Reopening the old release reproduces its exact
  bytes. Evidence includes membership comparison and mutation decisions.

### W1-T03 — Define preparation semantics explicitly

Depends on T02.

- [ ] State prediction-time feature availability, join keys/cardinality, target
  aliases, categorical missingness, numeric missing values, type coercion, row order
  and group handling. Bind learned preparation state to fitting membership.
- [ ] Reject duplicate-expanding joins, undocumented target-derived features,
  category recoding and row/label misalignment before context construction or fit.
- [ ] Acceptance: valid preparation round-trips with original labels; reordered rows,
  duplicate joins, changed categories and target aliases fail or receive a separately
  declared valid transformation. Retain transformed-row and schema comparisons.

### W1-T04 — Separate fit inputs, scoring labels and result release

Depends on T02/T03.

- [ ] Project fitting-only artifacts before worker launch. Scorers receive only their
  declared scoring labels and an authenticated fitted subject/prediction identity.
  A separate release authority controls audit/outer result visibility.
- [ ] Define task-level development search versus inner-fold-pure context construction.
  Do not test invariance of a scorer when its authorized scoring labels change;
  test the upstream producers that are forbidden to read those labels.
- [ ] Acceptance: access traces and capabilities establish both authorized scoring
  and denied upstream access. Retain a valid score and a refused label-read receipt.

### W1-T05 — Enforce an immutable worker input closure

Depends on T04; live integration depends on W7.

- [ ] Stage exact verified objects with bounded physical paths; reject symlink escapes,
  unlisted siblings, stale cache entries, hardlink/replacement races where applicable,
  and undeclared ancestor dependencies. Consume the validated bytes/handles.
- [ ] Remove broad AWS credentials, home-directory access and network fallback from
  the learning worker. Acquisition failures remain visible at the acquiring component.
- [ ] Acceptance: denied storage reads and a missing required object stop the request;
  an authorized pre-staged object succeeds without a network fetch. Test cache hit,
  cache miss and resume, not only the cold constructor.

### W1-T06 — Build counterfactual receiving-path proofs

Depends on T03–T05.

- [ ] Freeze permitted inputs, source, random streams and execution settings. Perturb
  only forbidden targets/features/parents and compare restricted intermediate outputs,
  proposals and learning inputs; instrument attempted reads.
- [ ] Add authorized-input perturbations that must change the intended intermediate
  result. A test that ignores both allowed and forbidden input is not a boundary proof.
- [ ] Cover target shuffles, row reorderings, group aliases, changed source bytes,
  duplicate joins, category changes, denied reads, symlink escapes and stale caches.
  Retain one receipt per mutation and explain comparison tolerances.

### W1-T07 — Resolve historical staging without rewriting it

Depends on T01; corrected execution depends on T05/T06 and W11 admission.

- [ ] Reconcile retained evidence showing outer development labels were parsed before
  projection and the run failed before CUDA qualification. Record affected descendants,
  actual execution extent and unknown outcomes; do not infer completed contamination.
- [ ] Give the corrected fitting-only planner/run a new identity and original window.
  Keep old failed evidence accessible and excluded from new headline measurements.
- [ ] Acceptance: historical and corrected manifests cannot be substituted; a replay
  verifier rejects the old input closure for the corrected protocol.

## Verification and delivery

Existing collection entrypoints, to run only within a claimed build/test budget:

```sh
cargo test --locked -p jope-parallel --test data_contracts -- --list
cargo test --locked -p jope-parallel --test families -- --list
cargo test --locked -p jope-parallel --test view_contracts -- --list
```

Proposed slices: (A) release/family schemas and preparation semantics T01–T03;
(B) projection and worker closure T04–T05; (C) counterfactual harness and historical
disposition T06–T07. Each slice needs exact new receiving-path test names and receipts.

Handoff to W2/W3/W5/W7: frozen regression release, family/exposure registry, producer
scope rules, worker closure and scorer-release contract. Fixture completion requires
all T06 cases plus valid controls; live isolation stays open until managed workers
exercise the same closure. Unresolved exposure blocks untouched-transfer claims,
not a separately identified permitted development experiment.

## Original finding routes

Primary register entries: `S02-G2`, `S02-G3`, `S02-G4`, `S02-G5`, `S02-G6`, `S02-G7`, `S02-G8`, `S02-I2`, `S02-I3`, `S02-I4`, `S02-I5`, `S02-I7`, `S02-I8`, `S03-G02`, `S03-K04`, `S10-G2`, `S10-K2`.

See [the complete crosswalk](FINDINGS-COVERAGE.md) and original clauses. This primary
routing is not a waiver of cross-cutting acceptance or later live integration.
