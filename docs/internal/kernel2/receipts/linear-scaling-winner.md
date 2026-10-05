# Opt-in scalar calibration and persisted winner

Status: first scalar library slice; full B09 search/serving admission remains held.
Current correction base: `79a31f7650b5199e636524ac85ec58af7e09ae43` (merged B14); shared baseline PR2234/33 skipped, main4316/70 skipped. The candidate adds21 nonignored tests:13 winner,7 recipe,1 private evaluator unit. Expected final selection PR2255/33 and main4337/70; current-main execution comes from the shared baseline, not this correction. Historical source-bound counts below describe earlier attempts.
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

The initial nine-test proof covered exact `y=3*c0+7`, constant fallback, malformed shape/targets,
tiny features, overflow refusal, malformed/truncated/unknown envelopes,
signed coefficient bits, deterministic widths 1..16, poisoned outer folds,
and stored-winner Rust/Python/C++17 output-bit parity. Seven formula families
and an extreme-finite winner execute 53 predictions per backend (159 exact
comparisons in the initial proof); generated functions also refuse malformed/nonfinite inputs.
Correction-01 includes Step and five nested method-receiver shapes: thirteen
formula families plus the extreme winner execute 95 predictions per backend
(285 comparisons). Only the new v2 Rust library types its scalar binding and
Step literals as f64; legacy emitted bytes stay frozen.
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

## Historical combined PR #178 qualification

The shared repaired-main baseline at `6df43201750b50ffcee908333e2f90dee8672829` passed PR2108/29 and
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

Correction-01 rebases on reviewed fixture prerequisite #181. Shared full
baseline DONE at this base qualifies all standing commands and the accepted
unchanged Clippy floor; the prior failed-main proof is preserved. Final
correction after-gates and fresh hosted CI/review remain pending. Lane D
advisory R3 nested-Step repro is preserved and covered in both export routes.
Lane A completes #178 then #177 and hands off idle; B10/B11 belong to Lane D.


Unmerged correction-02 shares the private signed-zero/power adapters with the
training recipe. Stored winners retain exact coefficients/state and never refit
at prediction; the new composition regression covers 30 trees/15,900 rows and
all three emitted library entry points. Independent R4–R6 are pending fresh
exact-head qualification and native review, not yet claimed resolved. See
[training recipe correction](linear-scaling-recipes.md) and external
`B09_LINEAR_SCALING/correction-02-signed-zero` evidence. Legacy emitter bytes,
raw evaluator/codec and historical identities remain unchanged.


Correction-03 explicitly freezes row-extrema zero ties for this unmerged v2
profile. Fit and prediction use the same private mixed-zero binding; neither
mutates the original tree or serialized winner. All other protected arithmetic,
hard-error fallback and old callers remain on the unchanged evaluator. The
analytical width1..512 regression passes in optimized builds after reproducing
the defect on the prior source. Final default/release focused17, PR/compat/lint,
affected recipe matrices, audit and fresh exact-head hosted/native review are
tracked under `B09_LINEAR_SCALING/correction-03-extrema`; no merged/full B09 claim.


## Correction-04: runtime power operands

Independent review 5414110872 on `f4f11e7829ed33cab6adcf373446da564beaf77e`
reproduced LLVM constant-base power lowering despite an opaque exponent.
`8^c0` and `1024^c0` changed prediction words, fitted coefficients, training
MSE and complete persisted state in optimized Rust. The expanded regression
fails on that source; its command and failure remain immutable.

Only the private v2 scalar library now hides **both** operands at the actual
power call: Rust uses `black_box` for base and exponent, C++ uses volatile
operands. Frozen emitters, evaluator, codec and identity inputs are unchanged.
Both stored-winner and independently trained/restored routes retain all prior
30 signed-zero corpora and add 14 constant/nested-base and fractional-training
corpora. Together 44 corpora per route exercise 19,220 held-out rows per route
across widths 1/2/512: 115,320 exact backend prediction-word comparisons,
plus coefficients, MSE, state and refusal checks. This is a same-host strict
compiler parity contract; no cross-platform libm guarantee is claimed.

Current default/release focused17, recipe ASan/UBSan6 and C++20 six-test
matrices, PR/compat/lint/required ratchets, protected inputs and lane audit
are tracked in `B09_LINEAR_SCALING/correction-04-power`. Final qualification,
exact-head hosted CI and fresh independent native approval remain required;
old source-bound passing gates do not certify this correction. Full B09
shipping search/EEC/utility/serving and VarPro stay outside this library slice.


## Correction-05: Clip bound ties and observable refusals

Lane D advisory5994937113 and independent exact-head COMMENT5415170521
identified signed-zero Clip bounds still using optimizer-dependent Rust min/max
on `ff7ecde112030420919ca78ac5d9010d8dc12df6`. The actual stored-winner export
mismatched native in four of five initial probes; the new repository regression
fails on that source. Only the private v2 Rust library rewrites the Clip bounds
to NaN-aware ordered selects with right-operand ties, and refuses unexpected
helper text. Frozen emitters and evaluator identity remain unchanged. Training
recipe admission continues to refuse every Clip tree.

Twenty stored-winner Clip cohorts cover the five reported shapes and five finite
nested variants at widths4/512, both zero-bound signs/orders, positive/negative
inputs, minimum subnormals and finite extremes. The finite nested variants keep
zero signs observable without relying on a constant-feature fit. They add640
held-out rows to the existing stored-winner composition proof. The shared Pow
corpus also pins the independently divergent base2 and base8/1024 exponents.
Together both routes execute117384 exact backend prediction-word comparisons
(native vs Rust/Python/C++), plus fit coefficients/MSE/state and refusals.

A private evaluator unit checks Owned binding for mixed zeros at widths4/512
and Borrowed ordinary/single-sign/nonzero-extrema controls. This makes the
binding guard observable in unoptimized CI as well as release execution. Native
and all three independent training recipes explicitly refuse the cancellation
case whose nonzero slope is unrepresentable: c0 rows0,-1e12,1e12,1e-300 with
targets0,0,0,1. Stored-export C++ drivers now honor the same strict sanitizer
and C++20 matrix selectors as the training recipes.

Source-bound failed attempts and actual exits are preserved under
`B09_LINEAR_SCALING/correction-04-power`; new final-head light gates, release
proof and affected matrices belong to `B09_LINEAR_SCALING/correction-05-clip`.
No pending qualification, CI or native approval is asserted passed here. Full
shipping search/profile/EEC/utility/serving and VarPro remain held.
