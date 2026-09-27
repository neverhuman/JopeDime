# W4 — Qualify neural generation and ranking

Status: current per-task state lives only in the generated [STATUS.md](STATUS.md); the
checkboxes below are the original acceptance plan, not a progress record.
Owner: policy inference. Requires W2 semantic bundle and context
ABI. W5 receives executable candidates; W10 measures utility. Apply the
[execution contract](EXECUTION-CONTRACT.md). Numerical agreement, identical sampling
decisions, valid syntax and useful proposals are separate claims.

## Source anchors and constraints

Inspect `crates/jope-policy/src/model.rs`, `decode.rs`, `grammar.rs`, `ranking.rs`,
`crates/jope-engine/src/training.rs`, `proposals.rs`, `ranked_proposals.rs`,
`grammar.rs`, and `crates/jope-engine/native/runtime.cpp`. Trace the actual native
and Candle computations. The full-source planning pass found a program-dependent
reward branch with order-blind mean pooling; do not assert program blindness from
the excerpt-only concern. An order-aware architecture is a W10 challenger.

Keep accepted tensor/calculation semantics, fallback origin labels and current
qualified numerical policy. Predicted reward/runtime values never become measured
fitted loss. Every retry consumes the original request's allowance.

## Ordered tasks

### W4-T01 — Write the operation-level forward contract

- [ ] Trace context encoding, prefix construction, embedding scale, token/position
  indexing, decoder layers, norms, tied output embeddings and active heads in both
  implementations. Record dtype, inference/dropout mode and intermediate shapes.
- [ ] Specify target shifting, BOS/EOS placement, causal mask, PAD participation and
  supported sequence limits. Map each operation to W2 accepted tensors/configuration.
- [ ] Acceptance: an independent reader can reconstruct the executed forward pass
  without guessing from names; undocumented backend differences block parity claims.

### W4-T02 — Prove autoregressive and batching invariants

Depends on T01.

- [ ] Compare teacher-forced and incremental predictions at equivalent logical
  histories. Exercise shortest, typical and maximum supported histories, with and
  without implemented caches.
- [ ] Change future suffixes, append supported padding and permute batch membership.
  Earlier next-token outputs and valid-token scores must obey the frozen semantics.
- [ ] Acceptance: no suffix leakage, off-by-one target shift, PAD contribution or
  accidental training mode; unsupported layouts reject explicitly. Retain intermediate
  comparisons and an independent reference, not only successful parser output.

### W4-T03 — Make decoding outcomes total and bounded

Depends on T01 and W2 identity checks.

- [ ] Define explicit terminal outcomes: completed neural proposal, incomplete sequence,
  invalid numeric input, parser rejection, duplicate, bounded fallback and exhausted budget.
  Bind original draw/request identity, tokens, grammar state and origin to each outcome.
- [ ] Bound token steps, regeneration, grammar repair and fallback separately from
  the later fitted-candidate allowance. Reserve/charge retries before executing them.
- [ ] Acceptance: invalid temperature, NaN/Inf logits, empty support and maximum-length
  exhaustion terminate deterministically without unmasked sampling or unbounded retries.

### W4-T04 — Differentially test grammar and typed parser

Depends on T03.

- [ ] Freeze the legal token vocabulary and grammar version. Enumerate a bounded
  state domain plus adversarial sequences; compare masking/acceptance with the production
  typed parser, including EOS in accepting and nonaccepting states.
- [ ] Test extreme finite logits, ties, invalid IDs, dead-end legal prefixes, empty
  support and exactly-at/over-length boundaries. Document any completeness pruning
  as a changed proposal policy requiring its own identity.
- [ ] Acceptance: every accepted sequence parses as the intended executable language;
  every selected token was in the actual legal support. Keep legal positive examples.

### W4-T05 — Establish scorer dependencies and architectural limits

Depends on T01.

- [ ] Trace program/context inputs to reward, runtime, stability and ablation heads.
  Perturb heads in disposable challenger copies and verify declared consumers respond.
  Do not use a single equal-output example to infer structural blindness.
- [ ] Retain a token-multiset collision fixture demonstrating order-blind mean pooling;
  also demonstrate nonsaturated program edits that change program-dependent outputs.
- [ ] Acceptance: the certificate distinguishes program sensitivity from order
  sensitivity and active from inactive heads. No reward redesign enters accepted inference.

### W4-T06 — Freeze sampler replay separately from backend parity

Depends on T02–T05.

- [ ] Record fixed logits, logical random draws, supported sampling configuration,
  grammar states, ties, token choices and terminal outcomes. Replay cold/cache/resume
  routes using the same logical random stream.
- [ ] Compare native/Candle tensors under their declared threshold; then separately
  measure token/selection agreement. Small floating-point differences near a sampling
  boundary may change decisions and must not be hidden by tensor parity.
- [ ] Acceptance: fixed-logit replay reproduces tokens and failures exactly; numerical
  parity reports its own scope and all divergent decisions.

### W4-T07 — Preserve proposal identity and complete denominators

Depends on T03/T06; joins W5-T01.

- [ ] Carry original tokens/program, neural/fallback origin, predicted scores, parser
  result, canonical identity and explicit adaptation edges into the candidate ledger.
- [ ] Reconcile scheduled draws, complete sequences, valid programs, unique programs,
  refusals, rank survivors and handed-off candidates. Keep rejected pre-gate candidates
  for W10 shadow ranking studies without granting them fitting authority.
- [ ] Acceptance: no duplicate/fallback inflation, dropped failures, double adaptation
  or score attachment to a different executable subject; every scheduled attempt terminates.

## Verification and delivery

```sh
cargo test --locked -p jope-policy --features testing --test jope_dime -- --list
cargo test --locked -p jope-parallel --test grammar -- --list
cargo test --locked -p jope-parallel --test reward_sequence -- --list
cargo test --locked -p jope-parallel --test proposal_ranking -- --list
```

Proposed slices: (A) forward specification and causality T01–T02; (B) total decode
and grammar T03–T04; (C) scorer/replay T05–T06; (D) candidate handoff T07.
Deliver golden traces, outcome schema, parser differential inventory, sampler capsule,
dependency certificate and per-request denominator reconciliation. GPU parity requires
its own admitted runtime and hardware receipts; CPU fixtures alone do not qualify it.

## Original finding routes

Primary register entries: `S04-G02`, `S04-G03`, `S04-G04`, `S04-G05`, `S04-G09`, `S04-I02`, `S04-I03`, `S04-I04`, `S04-I05`, `S04-I09`.

See [the complete crosswalk](FINDINGS-COVERAGE.md) and original clauses. This primary
routing is not a waiver of cross-cutting acceptance or later live integration.
