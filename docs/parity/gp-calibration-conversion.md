# Persisted scalar calibration conversion proof

This tests-only slice depends on the B09 `LinearScaledWinner` admission in
PR #178. It is staged locally until that dependency lands. It does not qualify
full B09 shipping search, utility promotion, VarPro, or arbitrary-fit accuracy.

Acceptance:

```sh
cargo test -p feat-math --locked --test conversion_calibration_winner -- --nocapture
```

The [test](../../crates/feat-math/tests/conversion_calibration_winner.rs) fits
five winners using only their training tables. It writes their binary envelopes
to disk, drops the original fits, loads the persisted winners, checks lossless
text/binary replay, and emits the loaded coefficients and expression. No held
rows or labels enter calibration.

Each winner executes 73 held rows: the 66 two-column rows from the unchanged
`compat_freeze/mathexpr_values.v1.json` fixture plus seven explicit witnesses.
The corpus includes signed zeros, subnormals, finite extremes, NaNs, infinities,
out-of-training-range values, and wrong widths. Native persisted predictions are
checked against the separately compiled expression evaluator followed by two
binary64 affine operations. The raw expression's documented protected root
fallback is retained. Invalid inputs and nonfinite affine outputs must refuse.

Real emitted code is separately compiled and executed in four configurations:
Rust O0/O3 and g++ C++17 O2/O3 with `-ffp-contract=off -fno-fast-math`. All
1,460 output-word/refusal comparisons must match exactly. The drivers check
output cardinality and empty-input behavior. Persisted bytes remain unchanged.
Analytical coefficient witnesses use symmetric raw values or balanced Step
classes; they do not claim exact coefficients for arbitrary training tables.

Silent fallback checks include a prediction of 307 outside the legacy clip,
varying nonzero outputs, and an extreme-slope affine overflow refusal. Local
red-team probes changed emitted text only: Rust and C++ constant-intercept
fallbacks, Rust legacy clipping, and C++ omission of calibration. All four
compiled successfully and failed the exact-output assertion. The original test
source was restored; no mutation enters the commit.

Source-bound logs, compiler commands, mutation receipts, fixture hashes, and
the final local qualifier are retained outside the checkout under
`/home/ubuntu/jope-evidence/kernel2/bt/GP_CALIBRATION_CONVERSION/`.
