# W9 — Bind fitted export to the released package

Status: planned. Owner: fitted exporters/package loader. Requires W1 raw-input
semantics, W2 applicable identities and W5 selected fitted subject. Follow
[the execution contract](EXECUTION-CONTRACT.md). Extend the existing fitted export
path; do not replace it with recipe refitting or an alternate selected model.

## Sources and supported scope

Inspect `crates/jope-bench/src/panel/fitted_export.rs`, `fitted_export_run.rs`,
`inference_export.rs`, `crates/jope-engine/src/policy_export.rs`, the corresponding
fitted predictor serializers and backend emitters reached from those paths.
Start from actual supported regression compositions for Python, Rust and C++17.
Preserve C++17's declared Python typed-table host dependency. Native-only C++ serving
and broader compositions are separate extensions, not implicit requirements here.

Preserve existing exact-bit backend parity. Source generated, runtime ready,
numerically qualified and released are distinct states. Backend-specific failure
cannot be hidden by another backend's success.

## Ordered tasks

### W9-T01 — Inventory fitted operations and dependency profiles

- [ ] Enumerate supported composed operations, fitted-state versions, source emitter,
  wrapper/runtime dependencies and qualification entrypoints per backend. Distinguish
  parse support, emission support, runtime support and numerical support.
- [ ] Trace any path that can refit, silently default missing state, substitute a
  selected predictor or lower a different input adaptation. Preserve explicit unsupported results.
- [ ] Acceptance: each supported composition has a real qualification route and each
  unsupported one preserves the original selected fitted subject with a terminal reason.

### W9-T02 — Freeze a complete inference capsule

Depends on T01 and W5-T07.

- [ ] Bind original/executable recipe relationships, preprocessing state, fitted
  parameters, operation order, raw-input schema, output units/cardinality and state versions.
  Separate selected evaluation evidence from the authorized final refit identity.
- [ ] Define read-only inference-only access. Remove fitting capabilities and training
  labels from emission and serving; never reconstruct missing learned state with defaults.
- [ ] Acceptance: two fitted states for one recipe remain distinct; state/receipt swaps
  and missing stages fail even when convenient test rows happen to produce equal predictions.

### W9-T03 — Emit and build in a bounded credential-free environment

Depends on T02.

- [ ] Bind emitted source, constants/state, wrapper, compiler/toolchain, runtime
  dependencies and native reference. Treat category names/hostile strings as data using
  backend-appropriate escaping/serialization, never command interpolation.
- [ ] Run generated-code compilation and first prediction with network and credentials
  unavailable, bounded CPU/RAM/time, and explicit child cleanup. Instrument fit entrypoints.
- [ ] Acceptance: valid builds/predictions succeed with zero fit calls; hostile strings
  do not execute, private canaries do not enter public artifacts, and timeouts leave no workers.

### W9-T04 — Qualify raw-input and exact-bit behavior

Depends on T03.

- [ ] Compare against independently retained native predictions from the exact fitted
  subject, not a native model reconstructed from emitted state. Check preprocessing
  intermediates before final outputs on diagnostic fixtures.
- [ ] Cover missing values, unseen categories, numeric/string confusion, schema/order
  errors, all-missing columns, duplicate/extra columns, one-hot boundaries, empty input
  where supported, and exact output cardinality.
- [ ] Acceptance: supported valid inputs retain existing exact-bit parity on all three
  backends; malformed inputs reject. Empty/truncated predictions cannot pass by zip truncation.

### W9-T05 — Bind qualification to installation/load

Depends on T04.

- [ ] Produce backend-specific package manifests that bind source, executable/wrapper,
  state, dependency profile, native reference and qualification record. Define required
  versus optional backends in the frozen request.
- [ ] At install/load verify actual shipping bytes and required dependencies again.
  A changed file, stale reference or different runtime profile cannot inherit prior qualification.
- [ ] Acceptance: corrupt constants, substituted executables, missing wrappers/stages,
  changed dependencies and wrong-backend receipts fail before serving; intact packages load.

### W9-T06 — Separate determinism and bounded translation checks

Depends on T02–T05.

- [ ] Re-emit identical capsules and compare source identities independently from
  binary reproducibility under a pinned toolchain/environment.
- [ ] Add bounded structural checks for supported fitted operations: stage order,
  constants/shapes, transformation links, input/output ABI and absence of fitting calls.
  State the finite supported language and limits of these checks.
- [ ] Acceptance: deliberate omitted/reordered stages and corrupted constants are
  detected; do not claim universal program equivalence or reproducible binaries from source equality.

### W9-T07 — Release one supported real selected subject

Depends on T05/T06; real loop subject supplied by W10.

- [ ] Join the selected final-fit identity, package qualification, release decision and
  installed bytes. Preserve any backend partial/failure status and all runtime dependencies.
- [ ] Acceptance: independent replay follows the same fitted subject from selection to
  installed predictions with no refit or replacement. Export qualification does not promote
  the proposal policy or imply every operator/backend is supported.

## Verification and delivery

```sh
cargo test --locked -p jope-bench --lib fitted_export -- --list
cargo test --locked -p jope-parallel --test gp_contracts -- --list
```

Proposed slices: (A) support inventory/capsule T01–T02; (B) bounded emit/build and
raw-input qualification T03–T04; (C) install/load and translation/determinism T05–T06;
(D) real fitted-subject release T07 integrated into W10. Deliver capsule format,
support matrix, three backend qualification records, package loader checks and
tamper/private-canary receipts. Unsupported export never changes the selected predictor.

## Original finding routes

Primary register entries: `S01-G6`, `S01-K6`, `S07-G01`, `S07-G02`, `S07-G03`, `S07-G04`, `S07-G05`, `S07-G06`, `S07-G07`, `S07-G08`, `S07-G09`, `S07-G10`, `S07-K01`, `S07-K02`, `S07-K03`, `S07-K04`, `S07-K05`, `S07-K06`, `S07-K07`, `S07-K08`, `S07-K09`, `S07-K10`.

See [the complete crosswalk](FINDINGS-COVERAGE.md) and original clauses. This primary
routing is not a waiver of cross-cutting acceptance or later live integration.
