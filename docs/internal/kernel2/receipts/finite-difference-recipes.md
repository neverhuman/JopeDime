# Ordered difference recipes and fitted export

This is the second focused PR of the B08 held remainder, after native routing
#182. Wire 59 now has Rust, Python and C++17 train-and-predict recipes through
`VersionedProgram::emit_ordered_recipe`, plus caller-owned fitted C++17 through
`VersionedModel::emit_ordered_fitted_cpp17`. The admitted downstream shape is
the same plain numeric Ridge shape as the native fitter. Stored unsupported
shapes still decode losslessly and receive an explicit export refusal. Fixed
metadata must also fit a minimum native model and fitted codec before export
admission; the emitted training APIs enforce the final model size.

The recipes call the existing emitters and the unchanged integer-stencil
helpers. They fit exactly the declared training prefix, admit only finite
labels, preserve group-local history across the cut, and carry the native
ordered input identity. Each language reads and writes the native raw-word
state codec, including signed zero and NaN payloads. Prediction commits a
batch only after every row succeeds. Empty prediction batches preserve state;
width, order, timestamp and malformed-state refusals preserve continuation.

The fitted artifact captures a validated native checkpoint. Each caller owns
its continuation, and the downstream fitted predictor remains immutable. Its
codec declares format/reader version 2, feature 64 and the feature-specific
profile, then binds the native checkpoint, program identity and source hash.
The reader regenerates the source from the validated checkpoint and rejects
forged metadata, unknown fields, unsupported versions and oversized codecs.
Legacy emitter identities, recipes and model formats are unchanged.

Acceptance commands:

```sh
cargo test -p feat-math --test finite_difference_v2 recipes::
cargo test -p feat-math --test finite_difference_v2
cargo clippy -p feat-math --all-targets -- -D warnings
NEXTEST_TEST_THREADS=16 just test-pr
```

The compiled fixture proof covers orders 1–4 through all three recipes and
fitted C++17, using the existing v2 raw boundary fixture. It compares exact
prediction words, fault sets and continuation bytes with native execution.
The input identity includes quotes, backslashes, a tab and U+2028. Negative
checks cover prefix-length/label overflow, premature finish, empty batches,
late invalid rows, truncated replacement state and independent fitted callers.
Codec tests bind the feature/profile/source/program/checkpoint fields and
reject tampering. Compiler and sanitizer portability qualification, full card
after-suite and hosted exact-head evidence are recorded separately; a missing
or pending gate is not a pass.

Evidence lives under
`/home/ubuntu/jope-evidence/kernel2/bt/B08_REMAINDER/recipes-implementation-01/`.
Pre-registration compiler-harness and actual-API proofs live under
`B08_REMAINDER/recipe-prep/`; they are preparation rather than shipping gates.

Per-op definition of done from plan §3: this slice provides runtime/native,
compiled and emitted twins; guarded NaN/Inf/zero/fault behavior; bounded arity
and deterministic width; train-only fit; saturation faults; three-language
recipes and fitted C++; append-only IDs/helpers; version rejection and old
freeze checks; IP-scrub admission; and no new feat-math dependency. The kernel
and native codec proofs remain in the earlier B08 receipts.

Still held for the third focused remainder PR: shortest-existing-tree
comparison, bounded Experimental generation profile/prior, EEC create/change/
remove with lossless round trips, token/artifact cost and at least six training
datasets × five seeds at equal evaluation budget. W §17 G-A…G-E qualification
is required before Baseline promotion. Generation remains disabled in this
slice, and B08 is not yet card-complete.
