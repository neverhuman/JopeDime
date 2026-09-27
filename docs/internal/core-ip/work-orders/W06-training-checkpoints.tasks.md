# W6 — Bind training consumption, objectives and exact continuation

Status: current per-task state lives only in the generated [STATUS.md](STATUS.md); the
checkboxes below are the original acceptance plan, not a progress record.
Owner: native trainer/corpus. Requires W2 semantic identities,
W3 qualified teachers and W5 program/fitted ancestry. Use native lineage-B as the
initial regression route; the experimental Python architecture is not a substitute.
Apply [the execution contract](EXECUTION-CONTRACT.md).

## Source anchors and continuation modes

Inspect `crates/jope-engine/src/training.rs`, `training_exposure.rs`, `corpus.rs`,
`continuation_corpus.rs`, `corpus_projection.rs`, `checkpoint_recovery.rs`,
`auxiliary_training.rs`, `crates/jope-policy/src/model.rs`, and native runtime code.
Retain accepted objective coefficients unless a separately identified experiment changes them.

Four explicit modes are required: cold initialization; weight warm start; explicit
new-corpus transition; exact interrupted resume. Weight recovery alone does not restore
optimizer trajectory. A topology, corpus, objective or sampling-partition change is
not exact resume. Rank-specific semantics are finalized with W7.

## Ordered tasks

### W6-T01 — Freeze resolved training inputs

- [ ] Resolve and pin the corpus's actual bytes, qualified record set, programs,
  comparison groups, contexts, vocabulary and every transitive dependency for the block.
  Validate qualification/revocation at freeze and load; no mutable path substitution.
- [ ] Separate available corpus records from unique programs and sampled exposures.
  Specify allowed repeated exposure without treating it as new data.
- [ ] Acceptance: missing/stale/revoked dependencies and changed resolved bytes fail
  before the first update; valid cold and cached loading produce identical authorized tensors.

### W6-T02 — Specify the complete objective by term

Depends on T01.

- [ ] Trace all imitation, reward/preference/listwise, runtime, stability, ablation and
  regularization terms actually used. Record masks, supervised positions, target source,
  smoothing, reductions, grouping, units, coefficient and active parameter consumers.
- [ ] Define the numerator and denominator of each reduction, including empty support
  and unequal groups. List frozen/inactive tensors and prevent them being cited as learning.
- [ ] Acceptance: independently computed tiny fixtures reproduce each scalar loss and
  intended gradients. No undocumented term or default coefficient remains in the update path.

### W6-T03 — Validate comparisons and runtime observations

Depends on T02.

- [ ] Keep complete comparable preference groups together; bind task/data/evaluator,
  fitted subject, loss direction and label provenance. Reject invalid pairings and
  distinguish list membership/permutation from sequence order.
- [ ] Represent missing, failed, censored and incomparable runtime observations explicitly,
  including device/profile and timing boundary. Missing or timed-out runtime is never zero.
- [ ] Acceptance: pair reversal has the specified effect; list permutation, duplication,
  masks, missing labels and censoring follow the declared objective without invented comparisons.

### W6-T04 — Record actual consumption and committed updates

Depends on T01–T03.

- [ ] Retain sampled record/program IDs, comparison groups, positions, effective weights,
  sampler cursor/RNG identity, microbatches, attempted updates and committed global updates.
- [ ] Define commit semantics for accumulation, rejected/nonfinite updates and interruption.
  Reconcile physical computation, repeated exposure and committed optimizer steps separately.
- [ ] Acceptance: reconstruct the next sample sequence and actual supervision from the
  receipt; a skipped/rejected update cannot increase committed steps or claim changed parameters.

### W6-T05 — Commit complete checkpoints atomically

Depends on T04.

- [ ] Publish model, optimizer, scheduler, RNG/sampler state, counters, corpus, objective,
  numerical partition and parentage as one versioned committed manifest. Write dependencies
  before the manifest; expose no checkpoint as complete while a required object is missing.
- [ ] Bind rank-local states and global committed update; define compatible software/runtime
  versions and loader behavior. W8 supplies conditional remote publication and restoration.
- [ ] Acceptance: torn checkpoints, missing optimizer/RNG, stale corpus and inconsistent
  counters fail before updates; a complete checkpoint reloads the qualified state.

### W6-T06 — Enforce all four continuation modes

Depends on T05.

- [ ] Make each mode explicit in request/receipt. Cold init creates a new lineage;
  weight warm start declares optimizer reset; new-corpus transition records changed
  sampling/exposure policy; exact resume requires the complete same-topology state.
- [ ] Define allowed transition fields and reject an undeclared objective, partition,
  vocabulary or topology change. Preserve original execution limits during recovery.
- [ ] Acceptance: interrupted/uninterrupted deterministic runs consume the same next
  records and reach the same committed state for the qualified topology. Transition
  results are separately named and cannot inherit exact-resume claims.

### W6-T07 — Reconcile accepted ancestry and exposure limits

Depends on T01/T04/T06.

- [ ] Reconstruct accepted parent/recovery edges from original retained objects and
  receipts. Identify irrecoverable or unknown edges, including optimizer-state gaps.
- [ ] Derive defensible exposure totals only from known consumed records and committed
  updates; retain availability, possible exposure and exact exposure separately.
- [ ] Acceptance: the genealogy can be independently checked and refuses unsupported
  continuity claims. A successful new run does not manufacture missing historical evidence.

## Verification and delivery

```sh
cargo test --locked -p jope-parallel --test training -- --list
cargo test --locked -p jope-parallel --test replay_training -- --list
cargo test --locked -p jope-parallel --test corpus_projection -- --list
cargo test --locked -p jope-parallel --features neural --lib -- --list
```

The last command requires the qualified native SDK environment and a bounded build;
it must not silently fall back to a different trainer when unavailable. Proposed
slices: (A) corpus/objective T01–T03; (B) consumption accounting T04; (C) committed
checkpoint/modes T05–T06; (D) ancestry reconciliation T07. Deliver objective fixtures,
exposure ledger, checkpoint format, loader refusal matrix and deterministic resume
receipts. Actual GPU and multi-rank parameter changes remain W7/W10 gates.

## Original finding routes

Primary register entries: `S06-G05`, `S06-G06`, `S06-G07`, `S06-G08`, `S06-G09`, `S06-K04`, `S06-K05`, `S06-K06`, `S06-K09`, `S06-K10`, `S09-G3`, `S09-G4`, `S09-G5`, `S09-I3`, `S09-I4`, `S09-I5`.

See [the complete crosswalk](FINDINGS-COVERAGE.md) and original clauses. This primary
routing is not a waiver of cross-cutting acceptance or later live integration.
