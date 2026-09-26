# W11 — Qualify GPU measurement and evaluate one optimization at a time

Status: planned. Owner: benchmark/controller. Requires W1 corrected fitting-only
inputs, W2/W4 semantics, W5 search accounting, W7 admission and applicable W9 export.
Use [the execution contract](EXECUTION-CONTRACT.md). A fast failed request is not a
qualified latency result; every planned request retains a terminal disposition.

## Sources and historical boundary

Inspect `crates/jope-bench/src/panel/inference_benchmark.rs`, `inference_cuda.rs`,
`inference_gp.rs`, `inference_export.rs`, the adjacent prediction/report modules,
and `docs/internal/core-ip/README.md`. Preserve the historical failed study and
exhausted window. A corrected protocol receives a new immutable manifest and admission.
Do not revive its old deadline or count the failed staging run as CUDA qualification.

## Ordered tasks

### W11-T01 — Reconcile prior study and freeze the corrected census

- [ ] Retain original source/input/journal identities, staging deviation, failed
  qualification and unexecuted work. State which outcomes are known versus unavailable.
- [ ] Freeze new fitting-only inputs, label-free assignments, dataset/request/seed
  census, warmups, measured requests, fresh-process repetitions and GP attempts. Pin all
  required backends and request-level success criteria before execution.
- [ ] Acceptance: each original/new request has a distinct identity; no replacement
  retry overwrites a failed slot or imports an old qualification result.

### W11-T02 — Qualify runtime/device/source profiles independently

Depends on T01 and W7.

- [ ] Verify source, executable, accepted bundle, input closure, device UUID, CUDA/driver,
  compiler and runtime dependencies. Qualify xbabe2 first, then each other hardware profile.
- [ ] Execute relevant semantic tests through the actual GPU path before headline
  measurement. CPU fallback, unavailable native dependencies and incomplete qualification
  suppress headline latency rather than receive an attractive timing.
- [ ] Acceptance: each profile has explicit qualified/unqualified status and its own
  receipts; the RTX4090 cannot inherit RTX3090 evidence merely because both execute CUDA.

### W11-T03 — Instrument synchronized whole-request boundaries

Depends on T02.

- [ ] Specify monotonic timing boundaries and GPU synchronization. Include required
  transfers, encoding/decoding, fitting, emitted compilation and first prediction in
  whole-request wall time where the declared request requires them.
- [ ] Keep stage diagnostics, standalone backend timings and cumulative backend
  milestones separate. Record warm/cold model/cache/process state and compilation reuse.
- [ ] Acceptance: injected delays in transfer, fit, compilation and prediction appear
  in the correct enclosing measurements and not unrelated exclusive stage totals.

### W11-T04 — Retain complete measured and failed outcomes

Depends on T03.

- [ ] Persist scheduled/start/finish/failure/timeout/cancel events for all warmups,
  measured requests, fresh processes and GP attempts. Retain actual retry work and
  censoring thresholds; failures do not vanish from latency denominators.
- [ ] Bind timing samples to qualified profile/request/result identity. Reconcile the
  planned census, journal, controller allocations and report inputs independently.
- [ ] Acceptance: missing slots and incomplete predictions prevent complete headlines;
  summaries show failure rate, coverage and uncertainty alongside qualified timing.

### W11-T05 — Profile separately and choose a measured bottleneck

Depends on T04; profiling has its own identity/resource allowance.

- [ ] Measure candidate bottlenecks: model residency, context recomputation, transfers,
  decoding, compatible batching, compilation reuse or deterministic repeated fitting.
  Record overhead added by profilers and keep those samples out of headline measurements.
- [ ] Select one optimization with an explicit hypothesis, semantic invariants,
  admissible reuse scope and preregistered comparison. Reject optimizations that depend
  on forbidden label access or cross-scope cached products.
- [ ] Acceptance: the proposed change addresses an observed bottleneck with a reviewable
  workload and decision rule, not an assumed GPU advantage or a post-hoc favorable slice.

### W11-T06 — Replay identical work before adaptive scheduling experiments

Depends on T05.

- [ ] Run parent/optimized implementations on the same frozen logical work, inputs,
  seeds, candidate identities and required backends. Preserve all fit/selection/export
  semantics and original resource accounting.
- [ ] Test a scope-safe execution DAG for permitted reuse/parallelism. Keep logical
  candidate accounting distinct from physical shared work and charge retries accurately.
- [ ] Acceptance: semantic qualification and complete outcomes remain intact. Any
  changed adaptive scheduling/search trajectory requires a different matched-budget
  protocol; it cannot inherit an identical-work acceleration claim.

### W11-T07 — Evaluate benefit under the frozen rule

Depends on T06.

- [ ] Analyze paired outcomes, latency distribution/tails, throughput where specified,
  warm/cold conditions, failure rate and resource use under the preregistered rule.
  Explain noise/uncertainty and retain negative or inconclusive results.
- [ ] Report benefit only if the comparison supports it without dropping difficult
  requests, changing required work or weakening semantic thresholds.
- [ ] Acceptance: an independent replay regenerates the comparison from retained
  journals; failed qualification suppresses headline claims even when timings exist.

### W11-T08 — Hand off bounded performance claims

Depends on T07.

- [ ] Bind each claim to hardware/runtime/input/protocol identities, denominators,
  confidence/uncertainty scope, exclusions and original measurement source.
- [ ] Keep broader regression stress, unseen operators and alternate hardware as
  separately scoped extensions; never broaden the original study by changing its label.
- [ ] Acceptance: W12 can revoke a performance claim when a profile or semantic
  prerequisite fails without deleting historical samples or rewriting the old result.

## Verification and delivery

```sh
cargo test --locked -p jope-bench --lib inference_benchmark -- --list
cargo test --locked -p jope-bench --features asm --lib inference_benchmark::reporting
```

Proposed slices/phases: (A) corrected manifest/qualification T01–T02; (B) boundaries
and complete journals T03–T04; (C) profiling and one optimization T05–T06; (D) admitted
comparison/report T07–T08. Existing report tests are fixtures; GPU execution requires
the approved new manifest and grants. Deliver all planned terminal outcomes and
whole-request timing receipts, not only selected successful samples.

## Original finding routes

Primary register entries: `S10-G3`, `S10-G4`, `S10-G5`, `S10-G6`, `S10-G7`, `S10-G8`, `S10-K3`, `S10-K4`, `S10-K5`, `S10-K6`, `S10-K7`, `S10-K9`.

See [the complete crosswalk](FINDINGS-COVERAGE.md) and original clauses. This primary
routing is not a waiver of cross-cutting acceptance or later live integration.
