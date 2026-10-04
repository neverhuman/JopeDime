# Feature-position polynomial FIR (B12 / wire 58)

The opt-in `feat-math.feature-row.experimental.v2` profile uses program subtype
3, required feature bit 32, and the separate `feat-math.feature-row-model` JSON
envelope. Legacy program/model readers reject these envelopes. The frozen
registry, exact scalar codec, default grammar, and existing artifact bytes are
unchanged. Source additions to the shared planner/emitters mint an announced
evaluator source identity; only the two announced identity metadata fixtures receive a hash-bound successor
(the lineage fixture changes its generation commit only).

`savgol` accepts `Axis::FeatureWithinRow`, an odd window 3..129, polynomial
degree `p <= min(6,w-1)`, derivative `r <= p`, and `boundary="shrink"`.
Feature spacing is one. It reads positions within a single observation and
preserves width. SampleTime, timestamp/group state, offline smoothing, and
cross-observation filtering are refused. The whole pipeline binds input width
1..512 and at most eight stages, then uses the existing ridge head.

For output position j, clip `[j-w/2,j+w/2]` to the row. Fit degree
`min(p,number_of_samples-1)` on those positions, evaluated at j. If r exceeds
that degree, return +0 with stage-local `BOUNDARY_DEGREE` bit 18. An empty row
is valid for the standalone kernel, yielding empty values/faults; a complete
model refuses it. Empty prediction tables yield empty predictions.

The geometry setup uses scaled monomials and two modified Gram–Schmidt passes,
then solves the triangular derivative system. Reject a singular setup. Freeze
the resulting FIR coefficients as u64 bits, including every clipped window.
Their bit patterns are embedded in all three recipes and the fitted C++17
predictor. This is geometry-only setup at plan time: no observations or labels
enter coefficient fitting. Restore verifies canonical geometry and bits.
Polynomial FIR derivative coefficients and evaluation position follow the
usual least-squares definition ([SciPy coefficient reference](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.savgol_coeffs.html));
the clipped, degree-reducing boundary rule here is an explicit v2 contract.

Dot products start at +0, read coefficients in increasing feature position,
and perform a separate multiply followed by add. Replace nonfinite inputs by
+0 with `NONFINITE_INPUT`. Clamp finite inputs and final values to ±1e12,
raising `MAGNITUDE_CLAMPED` at either clamp. A nonfinite final value becomes +0
with `NONFINITE_SANITIZED`. Input signed zeros survive normalization; the
specified dot product determines the output zero sign. C++ builds require
`-ffp-contract=off -fno-fast-math`. Prediction fault words aggregate row-stage
faults; they do not retroactively change the historical ridge fault contract.

The named profile supports seeded insert, parameter edit, and remove. Its
mutation scheduler gates Experimental moves with prior <=0.25. Disabling the
profile prevents generation while decoding existing programs remains available.
Generation is bounded to the row program; it never modifies the scalar AST or
the default legacy search. Canonical text and binary round trips are lossless.

The focused acceptance test is `cargo nextest run -P pr -p feat-math --locked --test
kernel2_closure -E 'test(savgol_v2::)'`. It checks polynomial derivatives (including a nonzero sixth
derivative), an independent analytic five-tap stencil, short/empty/max-width
rows, faults, envelope rejection, exact coefficient/head serialization, and
train-only fit. It compiles and executes the boundary corpus and GP-generated
insert/edit/remove programs in Rust, Python, C++17 recipes, and fitted C++17.
`JOPE_ROW_CPP_QUALIFY=1` adds clang++, C++20, and ASan/UBSan execution for
boundary helpers, GP recipes, and fitted predictors.
The owning fixtures live in `tests/fixtures/math_boundary/v2/savgol*.json`.

The fixture utility lane uses six pinned polynomial signal tasks × five seeds,
with 64 identical rows/evaluations per task/seed for this filter and the old
two-neighbor derivative. It writes per-task MSE receipts and token/artifact
deltas. It is synthetic numerical utility evidence. Real-campaign equal-budget
G-A..G-E qualification, outer-fold generalization, and Baseline promotion remain
held; this PR does not claim those milestones. A fixed-width expansion needs a
distinct neighbor-weighted expression for every output feature, whereas one
bounded stage handles variable widths and clipped windows; `row_diff_stats`
emits summaries rather than the required derivative vector.

No feat-math dependency is added. JavaScript is explicitly unsupported for the
new profile. Historical compatibility, strict emitter, PR/main test inventories,
clippy ratchet, lane audit, exact-head CI and independent native approval are
recorded in the card's external receipt.
