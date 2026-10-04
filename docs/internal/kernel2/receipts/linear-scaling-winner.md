# Opt-in scalar calibration and persisted winner

Status: first scalar library slice; full B09 search/serving admission remains held.
Base: `b036848460f3a2887fcdc6120db97152a3c01490`.
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
overflows; centered covariance/variance use magnitude-normalized terms. No
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
comparisons); generated functions also refuse malformed/nonfinite inputs.
C++17 uses `-O2 -ffp-contract=off -fno-fast-math`; Rust uses `rustc -O`.

Durable command logs, baseline/after inventories, frozen-input checks and audit:
`/home/ubuntu/jope-evidence/kernel2/bt/B09_LINEAR_SCALING/`.
Baseline reuse follows standing gate 1 at the identical freshly fetched SHA,
after verification of 38 immutable evidence hashes and 100 protected inputs.
Full after-gates and hosted lifecycle outcomes must be read from that receipt.

Remaining B09 gates: shipping profile/selection/persistence integration, full
three-language training recipes and EEC/serving envelope qualification, and
at least six fixed datasets × five seeds under equal-budget G-A..G-E utility
gates before Baseline promotion. No VarPro or dataset campaign is included.
Legacy scoring, defaults, RNG streams, models, emitters, historical fixtures
and evaluator identity inputs remain byte-identical. Parked #176/#177 are
unchanged; the owner authorized this independent B09 slice on 2026-10-04.
