# Opt-in scalar producer envelopes

The B05–B07 producer API selects inverse links (Func 19–20), stable exponentials
(Func 21–22), or numerical policy v2 (asinh and stable row reductions). All
selectors are Experimental. Existing profiles, defaults, function IDs, and
numerical policy v1 remain unchanged.

`scalar_producer_profiles(carrier, width)` returns explicit selectors for Row or
Elementwise input, at widths 1–512. Each declaration embeds the accepted
`kernel2-baseline` revision-3 profile and its SHA. Shapes remain bounded to 15
nodes and depth 4. The separate opt-in draw has probability 1/4; disabled choices
retain their slots, and rejected proposals retain the parent without another draw.
The API reuses the existing expression grow and mutation operations.

`ScalarProducerProfile::fit_training` admits the program and declared input scope
before using the generic scalar fit route. It fits only the supplied training
rows and labels. Prediction reads immutable fitted state. Rust, Python, and
C++17 recipes retain the existing fitter and numerical helpers; their header and
sidecar bind the declaration SHA. Fitted C++ source and codec bind the full model.

Unbound scalar models retain their exact format-2/min-reader-2 serialization.
Producer-bound models use format 3/min-reader 3, carrying the canonical producer
declaration and SHA. A reader checks the version pair and producer identity before
interpreting program or fitted state. Unknown identities, scope mismatches, and
invalid state are refused. The generic `VersionedModel::from_json` delegates
scalar ingress to this reader.

Deploy readers that support format 3 before distributing producer-bound models.
Older readers must reject the new envelope. Keep unbound format-2 models for
callers that need their existing reader contract. Disabling a family's generation
preserves its catalog, parser, and saved-program admission; its new declaration
SHA records the changed selection configuration.

The reproducible utility example uses eight pinned TRAIN-only datasets, five seeds,
equal 72-attempt budgets, and fixed caps. It evaluates the primary Penalize policy
and reports Legacy separately. Manual old-formula references share the training
folds and objective but do not enter the search population or consume its budget.
Reporting rows are a separate internal training cut, accessed after selection and
refit. Neither outer-test nor sealed data enters the worker.

Run the native measurements with:

```sh
cargo run --locked -p feat-math --example scalar_utility_v2 -- \
  crates/feat-math/tests/fixtures/scalar_v2/utility/manifest.json \
  /path/to/new/evidence-directory FULL_SOURCE_SHA Penalize
```

Repeat with `Legacy` and a distinct evidence directory. The example writes actual
work counts, artifact hashes and token costs, per-dataset metrics, and explicit
promotion-gate results. Native measurements leave compiled parity pending.
Unreachable literal 95%-of-final-best thresholds remain null. Zero-noise ties
do not count as improvements. Unknown partial work cannot qualify soundness.

These APIs do not authorize a Baseline/default flip. Promotion requires the W §17
quality, parity, soundness, cost, and elite-use receipts plus the owner decision.
Failing rungs stay Experimental or decline while their parser IDs remain reserved.
Reviewed merges, exact-head CI, compiled EEC/utility replay, and final card-suite
receipts are separate from native example output.
