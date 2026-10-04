# Opt-in scalar calibration and persisted winner

Status: first scalar library slice; full B09 search/serving admission remains held.
Current combined base: `87b81928deeed097099ed1e0d4d718d0bc26ceaf` (#179 CI retention; #176 included).
Initial standalone proof base: `b036848460f3a2887fcdc6120db97152a3c01490`.
Profile: `feat-math.scalar-linear-scaling.v2`.

`LinearScaledWinner::fit_training` computes slope/intercept from the explicitly
supplied training fold, then rescores the persisted winner on the same rows.
Exactly constant expression values use slope zero and target mean; tiny
nonconstant values are not replaced by an epsilon fallback. Rectangular finite
features, aligned finite targets and a bounded row expression are required.
Unrepresentable coefficients, loss or affine predictions refuse cleanly.

The base expression keeps legacy protected arithmetic; inference includes its
hard-error zero fallback. The new affine epilogue performs separately rounded
`a*f+b` without the legacy model's [-60,60] prediction clip. Ordinary means use
ordered compensated sums, with magnitude-normalized fallback if the sum
overflows. Nonconstant calibration translates by an input anchor before
power-of-two normalization and compensated centering; overflowing target
differences use a zero anchor instead. Coefficients and squared loss are
rescaled by exact powers of two, avoiding premature overflow or underflow.
Positive loss that cannot be represented refuses instead of scoring zero. No
correct-rounding guarantee for arbitrary-magnitude fitted coefficients is made.

The additive program envelope uses FMKPRG version 2, variant 2, minimum reader
2, required-feature bit 16, width, exact coefficient bits and an embedded typed
raw symbolic program. Generic versioned/lossless ingress preserves the same
bytes. Historical v1 readers, scalar-only readers and legacy model/export routes
reject the wrapper. The typed inner expression retains the v1 numerical policy.
No Func, Var or wire ID is added, and no existing envelope bytes are changed.

Export reuses the existing symbolic planner and helpers, appending the explicit
fitted library function `scaled_predict(rows)` in Rust, Python and C++17.
Coefficient constants are reconstructed from their persisted binary64 words.
Use a Rust library, an imported Python module, or the C++17 library function;
the inherited raw recipe/CSV APIs are not the fitted-winner entry point.
Inference accepts no labels and cannot refit or replace coefficients.

Acceptance:

```sh
CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=line-tables-only \
TMPDIR=/home/ubuntu/.cache/jope-test-tmp/bt-implementation \
cargo test --locked -p feat-math --test linear_scaling_v2
```

Nine tests cover exact `y=3*c0+7`, constant fallback, malformed shape/targets,
tiny features, overflow refusal, malformed/truncated/unknown envelopes,
signed coefficient bits, deterministic widths 1..16, poisoned outer folds,
and stored-winner Rust/Python/C++17 output-bit parity. Seven formula families
and an extreme-finite winner execute 53 predictions per backend (159 exact
comparisons in the initial proof); generated functions also refuse malformed/nonfinite inputs.
Correction-01 includes Step: eight formula families plus the extreme winner
execute 60 predictions per backend (180 comparisons). Only the new v2 Rust
library binds its scalar result explicitly as f64; legacy emitted bytes stay frozen.
C++17 uses `-O2 -ffp-contract=off -fno-fast-math`; Rust uses `rustc -O`.

Durable command logs, baseline/after inventories, frozen-input checks and audit:
`/home/ubuntu/jope-evidence/kernel2/bt/B09_LINEAR_SCALING/`.
Baseline reuse follows standing gate 1 at the identical freshly fetched SHA,
after verification of 38 immutable evidence hashes and 100 protected inputs.
Full after-gates and hosted lifecycle outcomes must be read from that receipt.

Remaining B09 gates: shipping profile/selection/persistence integration, full
shipping training-recipe integration and EEC/serving envelope qualification, and
at least six fixed datasets × five seeds under equal-budget G-A..G-E utility
gates before Baseline promotion. No VarPro or dataset campaign is included.
Legacy scoring, defaults, RNG streams, models, emitters, historical fixtures
and evaluator identity inputs remain byte-identical. PR #176 is merged in the current base; #177 remains parked and unchanged.
The owner authorized inclusion of the qualified local training-library follow-up
in PR #178, after a fresh-main baseline, on 2026-10-04.

## Combined PR #178 qualification

The preceding full baseline at `2e75f00cc8abe2323fa6e3c5e2d0965c35b68bc3` passed PR2108/29 and
main4190/66, Python231/2, golden16/16 and fitted18/18 in both matrices. The
combined winner plus explicit training-library changes and correction-01 add
fourteen nonignored tests; expected after counts are PR2122/29 and main4204/66. Named old
selected/skipped/excluded sets, all 100 protected inputs and Clippy ratchets
are rechecked. Actual `just required` remains exit101 NOTPASS on the accepted
unchanged held-design floor. New-head results, dedicated recipe sanitizer
and C++20 runs, lane audit, hosted CI and fresh independent approval are
recorded in `/home/ubuntu/jope-evidence/kernel2/bt/B09_LINEAR_SCALING/rebase-after-176-01`.
These outcomes must be read back; this source receipt does not assert pending gates passed.

Correction-01 addresses independent review 5407908412: adjacent-feature
centering, representable subnormal loss and accepted Step Rust compilation.
The native exact-affine family covers 11 signed offsets, lengths 2..32 and
widths 1/2/7/32/512 (1,705 fits plus extrapolation), with an independent
power-of-two slope/intercept oracle. Loss at 2/100/8192 rows must retain word
0x14; genuinely unrepresentable positive loss refuses. Normalization by powers
of two preserves representable translated variation; compensated summation
reduces accumulation error, but covariance cancellation and rounded products
still preclude a uniform relative-error guarantee for arbitrary ill-conditioned
least squares. The separately rounded inference epilogue is unchanged.

Correction-01 rebases on the reviewed CI-retention merge #179; the prior numerical baseline remains independently preserved. Current-main shared baseline and full correction after-gates are pending under the owner heavy lock. No pending outcome is claimed passed.
