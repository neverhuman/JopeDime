# W3 — Mandatory provenance, authenticated metrics and HWM decisions

Status: current per-task state lives only in the generated [STATUS.md](STATUS.md); the
checkboxes below are the original acceptance plan, not a progress record.
Owner: learning admission/storage. Requires W1 permitted scopes and
W2 authenticated context producers. [Shared execution rules](EXECUTION-CONTRACT.md)
apply. Existing arithmetic or receipt helpers are not evidence of mandatory ingress.

## Source anchors and preserved policy

Start with `crates/jope-learning/src/eligibility.rs`, `eligibility/inner.rs`,
`encoder_receipt.rs`, `hwm.rs`, `ledger.rs`, `record.rs`, and the teaching/corpus callers
in `crates/jope-engine/src/teaching_store.rs` and `corpus_publication.rs`.
The current receipt module declares that no existing HWM/teaching ingress calls it.
Trace all entrypoints before deciding where the mandatory capability belongs.

Preserve five-fold/four-improved-fold admission and existing margins. Preserve
mean-fold RMSE where selection/admission currently specifies it. Pooled RMSE and R²
are separately named diagnostics. Fit the trivial baseline on training rows only.
Eligibility, HWM initialization, comparable non-improvement and committed improvement
are distinct outcomes. Fold-tail reporting must not silently change admission.

## Ordered tasks

### W3-T01 — Map every authority-bearing ingress

- [ ] Inventory live admission, HWM ingestion, teaching freeze, historical replay,
  corpus loading and resume, including compatibility/import paths and duplicate delivery.
  For each, identify the last validation before durable mutation.
- [ ] Add pre-fix missing-context/altered-context reproductions through real ingress,
  with snapshots of ledgers, pointers and corpus membership before/after.
- [ ] Acceptance: the call map and reproductions distinguish missing enforcement from
  already-implemented safeguards. No helper-only test closes the ingress finding.

### W3-T02 — Resolve trusted transitive provenance

Depends on T01 and W1/W2.

- [ ] Resolve preprocessing, encoders, contexts, cached derivatives and candidate
  ancestors against expected identities from the frozen plan/admitted producer.
  Define root trust, cycle rejection, bounded graph traversal and revocation lookup.
- [ ] Bind consumer, scope, release, final context bytes and parent chain to the
  capability used for admission. Authenticate the producer, not just a self-consistent sidecar.
- [ ] Acceptance: missing/revoked parents, cycles, wrong scope/consumer, replayed receipts
  and altered context fail before any mutation; a complete authorized chain succeeds.

### W3-T03 — Version records and enforce every receiving gate

Depends on T02.

- [ ] Require validated provenance at every T01 ingress. Preserve diagnostic imports
  separately; old records without original proof cannot become qualified by defaults.
- [ ] Define migration inventory, quarantined states and explicit original-evidence
  requalification procedure. Ensure retries and warm caches rerun applicable validation.
- [ ] Acceptance: legacy/missing records remain readable diagnostically but cannot
  produce eligible teachers or optimizer corpus membership. Test valid migration and refusal.

### W3-T04 — Recompute metrics from exact authenticated predictions

Depends on T03.

- [ ] Bind predictions to fitted subject, executable recipe, row IDs, fold inventory,
  scorer/runtime and expected label role. Require unique complete fold membership.
- [ ] Recompute RMSE and training-only baseline from trusted bytes; retain reduction,
  units, direction and numerical policy. Expose pooled RMSE/R²/tail diagnostics separately.
- [ ] Acceptance: swapped fitted states, wrong folds, duplicated rows/folds, missing
  predictions, nonfinite values and forged scalar scores fail. An independent evaluator
  agrees with the declared selection arithmetic on valid uneven-fold fixtures.

### W3-T05 — Separate eligibility from HWM transition outcomes

Depends on T04.

- [ ] Preserve comparison identity `(task,dataset_hash,cut_id,evaluator,vocabulary,protocol)`.
  Emit explicit eligible/rejected, initialized, noncomparable, comparable-not-improved,
  tie and committed-improvement outcomes; define which ones can enter the corpus.
- [ ] Add per-fold adverse-tail diagnostics without changing the four-of-five rule or
  its margins. Specify any future stricter policy under a different version only.
- [ ] Acceptance: eligibility alone never increments improvement counters. Equal scores,
  mismatched identities and initialization cannot masquerade as new improvements.

### W3-T06 — Make decisions recoverably link to corpus eligibility

Depends on T05; remote recovery qualified by W8.

- [ ] Reuse durable ledger and conditional pointer primitives. Link qualification,
  predecessor, HWM decision and corpus eligibility by stable event identities.
  Define idempotent delivery and the recoverable state between each durable write.
- [ ] Race stronger/weaker comparable candidates in both orders, duplicate deliveries,
  and crash before/after decision append, pointer advance and corpus linking.
- [ ] Acceptance: replay reconstructs the winning comparable state with no duplicate
  improvement; incomplete linkage stays visibly incomplete until recovery succeeds.

### W3-T07 — Run the full ingress adversarial matrix

Depends on T03–T06.

- [ ] Exercise every T01 route with missing, wrong-fold, wrong-consumer, altered-context,
  revoked-parent, repeated-fold, nonfinite and wrong-fitted-subject cases.
- [ ] For each refusal assert unchanged authority-bearing state, then execute the
  authorized equivalent successfully. Retain counts per ingress and mutation; no unexplained skips.
- [ ] Acceptance: an independent replay consumes original decisions and reproduces
  eligibility/HWM/corpus disposition without trusting a producer's final success flag.

## Verification and delivery

```sh
cargo test --locked -p jope-learning --lib eligibility -- --list
cargo test --locked -p jope-learning --lib encoder_receipt -- --list
cargo test --locked -p jope-learning --test ledger_tamper -- --list
cargo test --locked -p jope-learning --test nonfinite_scores -- --list
```

Proposed slices: (A) ingress map/trusted capability T01–T02; (B) persisted versions and
mandatory consumers T03; (C) metric/HWM outcomes T04–T05; (D) crash/race reconciliation
T06–T07. Deliver receiver coverage, original/changed-state evidence, migration inventory,
metric specification and decision ledger. Hand validated records to W6 and W8; W12
must not present these fixtures as live operational admission until live proof exists.

## Original finding routes

Primary register entries: `S01-G1`, `S01-K1`, `S02-G1`, `S02-I1`, `S03-G01`, `S03-K01`, `S06-G01`, `S06-G02`, `S06-G03`, `S06-G04`, `S06-K01`, `S06-K02`, `S06-K03`, `S06-K08`, `S08-G02`, `S08-G04`, `S08-G05`, `S08-G06`, `S08-G08`, `S08-K02`, `S08-K04`, `S08-K05`, `S08-K06`, `S08-K08`, `S10-G1`, `S10-K1`.

See [the complete crosswalk](FINDINGS-COVERAGE.md) and original clauses. This primary
routing is not a waiver of cross-cutting acceptance or later live integration.
