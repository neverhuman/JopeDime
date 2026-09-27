# W5 — Close GP, fitting, selection and ancestry contracts

Status: current per-task state lives only in the generated [STATUS.md](STATUS.md); the
checkboxes below are the original acceptance plan, not a progress record.
Owner: engine search/fitting. Requires W1 inputs and W4 proposal
semantics; W2 supplies model/context identities. W0 must repair test authority through
legitimate issuers. Follow [the execution contract](EXECUTION-CONTRACT.md).

## Source anchors and receiving boundaries

Inspect `crates/jope-engine/src/gp.rs`, `fit_boundary.rs`, `proposal_population.rs`,
`native_genome.rs`, `eval_registry/authority.rs`, `eval_registry/native_scq.rs`,
`crates/jope-bench/src/panel/optimize_fit.rs`, and `fitted_export.rs`.
Legacy `namespace_activate` intentionally refuses while its verifiers are absent.
Do not fix tests by exposing test-only activation or weakening that production fence.

Keep original proposal, input-stage adaptation, executable recipe, variation child,
fold fit, selected recipe, final refit and exported fitted subject distinct. Explicit
transformations link them; requiring one hash everywhere would reject valid refits.

## Ordered tasks

### W5-T01 — Define the immutable candidate lineage envelope

- [ ] Bind original tokens/program, adaptation version, executable recipe, variation
  parents/events, fit scopes, context/release, evaluator and origin. Specify parent-child
  transformations and prohibit a second implicit adaptation.
- [ ] Bind each fold fitted artifact and final refit to its input identity; make parent
  fitted state immutable and child variation copy-on-write or equivalent.
- [ ] Acceptance: child changes cannot alter parent predictions; parent swaps, omitted
  variation events, recipe substitutions and double adaptation fail lineage verification.

### W5-T02 — Freeze request objectives and paired-arm differences

Depends on T01.

- [ ] Freeze objective direction, metric reduction, fold completeness, tie rule,
  evaluation scope, generation-zero selection, stop policy, RNG streams and arm differences.
  Specify permitted adaptation before any candidates are evaluated.
- [ ] Use the same feasibility/accounting policy in both arms. Separate number of
  draws, unique candidates, fitted attempts and completed folds.
- [ ] Acceptance: independent inspection can identify the sole intended arm change;
  changing seed/objective/folds/ties on resume rejects rather than silently starting anew.

### W5-T03 — Separate fit caches from score caches

Depends on T01/T02.

- [ ] Derive fit keys from executable program, input/preparation bytes, fitting scope,
  fit seed and numerical implementation. Derive score keys additionally from evaluation
  membership, evaluator, metric protocol and exact fitted subject.
- [ ] Retain parent provenance and revocation dependencies in cache entries; validate
  cold, warm, retry and resume paths against the same request.
- [ ] Acceptance: changed recipe/scope/evaluator cannot inherit stale scores; legitimate
  identical fitting work may be reused without inventing a new physical fit or hiding exposure.

### W5-T04 — Implement conservative feasibility classification

Depends on T01/T02.

- [ ] Inspect supported operators, shape propagation, intermediate widths, dtype and
  checked arithmetic. Return proven feasible, proven outside limits or unknown, with reasons.
  Unknown is not a soundness proof or automatic authorization for unbounded execution.
- [ ] Bound arithmetic overflow, one-hot expansion, Cartesian joins and matrix growth
  before allocation where provable; define bounded runtime handling for unknown cases.
- [ ] Acceptance: adversarial dimensions/overflow reject safely; valid edge-width programs
  succeed. Both arms receive identical rules; record feasibility false-refusal diagnostics.

### W5-T05 — Reserve work before dispatch and supervise termination

Depends on T02/T04; distributed proof joins W7.

- [ ] Atomically reserve candidate/fold capacity before dispatch. Track regeneration
  and retries independently from fitted-candidate attempts and retained logical reuse.
- [ ] Enforce original deadline, CPU/RAM and child-process drain through existing
  supervisor/SmartCluster components. Late results cannot reopen exhausted requests.
- [ ] Acceptance: blocked fitter, OOM, cancellation, dispatch race and delayed result
  cannot exceed admitted work or publish incomplete winners. Retain actual allocation,
  attempt ledger, terminal disposition and no-orphan process census.

### W5-T06 — Build an independent selector and stopping replay

Depends on T02/T03/T05.

- [ ] Retain every candidate's fold outcomes, measured losses, failures, timing and
  eligibility for selection. Keep predicted neural scores in distinct fields.
- [ ] Implement replay from the complete ledger using the frozen objective/tie/stop
  contract without invoking the searcher's winner-selection implementation.
- [ ] Acceptance: replay reproduces winners and stop decisions; missing folds, NaN/Inf,
  delayed outcomes and proposal reward cannot become measured winners. Ties are deterministic.

### W5-T07 — Seal selection, final refit and export handoff

Depends on T01/T06; W9 verifies serving packages.

- [ ] Record the selected executable recipe, authorized final-fit scope, complete fitted
  state and explicit link from evaluation fits to final refit. Keep held labels outside fit.
- [ ] An unsupported export returns that status while preserving the selected fitted
  predictor. It must not substitute a more exportable program or refit to imitate the winner.
- [ ] Acceptance: wrong fit-state references, missing parent events and same-recipe
  different-state swaps fail independently of prediction parity. The supported valid path succeeds.

## Verification and delivery

```sh
cargo test --locked -p jope-parallel --test gp_contracts -- --list
cargo test --locked -p jope-parallel --test fit_boundary -- --list
cargo test --locked -p jope-parallel --test gp_admission -- --list
cargo test --locked -p jope-parallel --test view_contracts -- --list
```

Proposed slices: (A) candidate/request contracts T01–T02; (B) caches and feasibility
T03–T04; (C) reservation/supervision T05; (D) selector and fitted handoff T06–T07.
Deliver ancestry/attempt ledgers, an independent replay executable, immutable fitted
subjects, failure-drain receipts and preserved unsupported exports. W10 later measures
evolution's benefit separately from extra generation-zero or fresh-proposal selection.

## Original finding routes

Primary register entries: `S01-G3`, `S01-K3`, `S04-G08`, `S04-I08`, `S05-G1`, `S05-G2`, `S05-G3`, `S05-G4`, `S05-G5`, `S05-G6`, `S05-G7`, `S05-I1`, `S05-I2`, `S05-I3`, `S05-I4`, `S05-I5`, `S05-I6`, `S05-I7`.

See [the complete crosswalk](FINDINGS-COVERAGE.md) and original clauses. This primary
routing is not a waiver of cross-cutting acceptance or later live integration.
