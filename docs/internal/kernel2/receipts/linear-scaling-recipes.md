# Explicit scalar calibration training libraries

Qualified local continuation now included in PR #178 on
`kernel2/bt09-linear-scaling`, rebased onto remote main
`4183d24ae5832c114d0194d4a6a7128647027782` after #187.
The rebase preserves merged ordered-fit/model and row-vector dispatch arms;
the three calibration/export corrections remain confined to the opt-in library.
Original dependency `e5266782121dc0cf131aa9e7ff4ee6a496e7d4a5` and local
follow-up `fbba3ec6247167465edd924d2aa7c0e748ccab6e` remain preserved with
their dated local proof. The owner lifted the publication hold after the Team
upgrade; no artifacts or repository data are deleted.

`emit_training_recipe(expression, input_width)` adds Rust, Python and C++17
library sources. Call `scaled_train(training_rows, training_targets)` after
structural scoring, then the returned model's `predict` without inference
labels. Rows must be finite, rectangular and the declared width, training must
be nonempty, and target length must match exactly. No outer fold/callback/RNG
is accepted. This preserves the explicit opt-in profile and legacy defaults.

The exact constant fallback, finite representability checks and training MSE
rescore retain the explicit calibration contract. Correction-01 translates
before power-of-two normalization, compensates centered products and rescales
coefficients/loss without premature overflow or per-row underflow. It fixes the
unmerged centering and false-zero-loss defects. Native fitting and generated Rust share one
source body; Python/C++ twins freeze the same operation order. This is not a
correctly rounded least-squares guarantee for arbitrary extreme data.

The trained model stores the exact variant-2 winner bytes. Generated readers
are bound to their expression and width; header/body drift, truncation, trailing
data and nonfinite coefficients refuse. The host versioned reader remains the
arbitrary-winner loader. Prediction does not mutate the stored coefficient
words and has no inherited legacy model's prediction clip.

Recipe admission refuses `clip` anywhere in the AST, since the old helper's
zero fallback does not preserve native fitting's typed invalid-bounds error.
The existing fitted winner API still supports those trees. Legacy emitters,
helpers, codec, frozen fixtures and evaluator identity inputs stay unchanged.

Focused acceptance:

```sh
cargo test -p feat-math --locked --test linear_scaling_recipes --test linear_scaling_v2
JOPE_FITTED_CPP_SANITIZER=address,undefined cargo test -p feat-math --locked --test linear_scaling_recipes
JOPE_CPP_STD=20 cargo test -p feat-math --locked --test linear_scaling_recipes
```

Five new nonignored tests cover admission, native extraction compatibility,
the 1,705-fit adjacent-feature oracle, representable subnormal loss and
nineteen independent train-and-predict corpora. Every generated model is restored
before predicting. Comparisons include two coefficient words, MSE, complete
winner bytes and 59 prediction words per backend; total 348 scalar word
comparisons and 57 complete-envelope comparisons. Finite extremes, constant
fallback and tiny nonconstant features are included. Every backend also rejects
invalid/nonfinite tables and labels, unrepresentable training loss, all state
prefix truncations, corrupt headers/body, trailing bytes and NaN coefficients.
Compile generated Rust with `rustc --edition=2021 -O`; C++ uses
`-O2 -ffp-contract=off -fno-fast-math`, with sanitizer/C++20 variants above.
The new Rust library types both the scalar binding and Step branch literals.
Five nested Step shapes (abs, min, powers as base/exponent, and sqrt of a
negated Step) compile and execute in all three languages. The frozen legacy
emitter and its source bytes remain unchanged.

Exact-source baseline reuse: completed #178 local gates at the dependency SHA,
39 immutable evidence files and 100 protected inputs reverified. PR2108/29 and
main4190/66, Python231/2, golden16/16, fitted18/18 in both matrices, zero-new
Clippy/audit. Actual `just required` is 101 NOTPASS at the owner-accepted
unchanged held-design floor. Original local after counts qualified PR2111/29, main4193/66;
immutable logs and results are outside the checkout in
`/home/ubuntu/jope-evidence/kernel2/bt/B09_SCALING_RECIPES_LOCAL`.

This is a training **library** recipe slice. The inherited raw CSV main is not
the new entry point. Standalone calibrated CSV wrappers, shipping selection and
CLI integration, full profile generation/EEC, equal-budget six-dataset utility
and Baseline/serving promotion remain held. No VarPro or later-card work.

Combined requalification compares the preceding baseline PR2108/29, main4190/66
and all fourteen new tests from winner plus recipes: expected PR2122/29 and
main4204/66. All normal after gates, dedicated recipe ASan/UBSan and C++20,
protected-input/skip/Clippy/audit comparisons and exact-head hosted CI plus
fresh independent approval are recorded separately in
`/home/ubuntu/jope-evidence/kernel2/bt/B09_LINEAR_SCALING/rebase-after-176-01`.
The inherited raw CSV entry point remains distinct from `scaled_train`.

Correction-01 rebases on reviewed fixture prerequisite #181. Shared full
baseline DONE at this base qualifies all standing commands and the accepted
unchanged Clippy floor; the prior failed-main proof is preserved. Final
correction after-gates and fresh hosted CI/review remain pending. Lane D
advisory R3 nested-Step repro is preserved and covered in both export routes.
Lane A completes #178 then #177 and hands off idle; B10/B11 belong to Lane D.


Correction-02 addresses independent review R4–R6 on unmerged head
`f301a672d8cd75820ca2d4c166c34654dc8cad1c`. Only these opt-in libraries adapt
legacy generated helpers: row sum/mean start at negative zero; floating min/max
use explicit ordered comparisons with right-operand zero ties and one-NaN
handling; Python floor and negative rounding retain the resulting zero sign.
Generated Rust/C++ power calls retain a runtime exponent, preventing constant
reciprocal rewrites from changing the evaluator's separately rounded result.
All frozen emitter/MathExpr/identity inputs remain unchanged. This is a correction
of the unmerged version-2 library, with no legacy emission transition.

Two additional nonignored regressions execute 30 compositions per export route,
including reductions, nested min/max/NaN, floor/round, sqrt/log1p/trigonometric
zero producers and reciprocal-to-final-finite-guard compositions. Widths 2 and
512 cover every signed-zero prefix in both orders, both all-zero signs,
minimum subnormals and rounding boundaries: 15,900 held-out rows per route,
95,400 backend prediction-word comparisons across stored winner and independent
training/restoration. Recipes additionally compare coefficient/MSE words and
complete serialized winner bytes. Test drivers compress repeated input words
by their bits and expand at runtime without removing cases or assertions.
Both initial reproducing failures and the incomplete oversized compile probe
are preserved in external correction evidence; neither is labeled passed.

Owner D3 correction gates are focused tests plus final PR tier, affected
compatibility, owned lint/held-floor and lane audit; affected training recipes
also run ASan/UBSan and C++20. The completed full card run remains bound to
`50245d4fcab6997a258a06f9c117c5f8d527ade2`, and is not relabeled as execution of
these corrected helpers. Current main's shared baseline is reused. Qualification,
actual exits, named test/skip inventories and exact-head CI/review are recorded
under `/home/ubuntu/jope-evidence/kernel2/bt/B09_LINEAR_SCALING/correction-02-signed-zero`.


Correction-03 found that the frozen Rust row-extrema fold itself has different
signed-zero ties in default and optimized builds (release analytical repro fails
at width4). The unmerged v2 scalar profile now freezes right-operand row-extrema
ties in both fitting and prediction. Only mixed-sign-zero rows whose min/max is
zero need a private binding of row_min/row_max to finite constant words before
executing the unchanged bounded tree. Ordinary rows borrow the compiled tree.
The original expression, serialized state and evaluator identity inputs stay
unchanged; legacy callers keep their old evaluator. There is no new historical
transition or change to old operator IDs. The private exported helpers freeze the
same order, independently of optimization.

A third new regression analytically checks 7 versus10 after reciprocal/finite
guards for row_min/row_max, every width1..512, both zero orders/signs, fitting,
prediction and byte-stable state. Total new nonignored tests are now17 (11 winner,
6 recipes). Full focused tests must pass in both default and release builds;
ASan/UBSan and C++20 train/restore matrices remain required. Earlier33d debug16,
PR2199/30 and audit passed, but its release failure blocks it: those receipts are
retained as superseded evidence. Active correction evidence is
`/home/ubuntu/jope-evidence/kernel2/bt/B09_LINEAR_SCALING/correction-03-extrema`.


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
