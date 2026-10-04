# Explicit scalar calibration training libraries

Qualified local continuation now included in PR #178 on
`kernel2/bt09-linear-scaling`, rebased onto remote main
`2e75f00cc8abe2323fa6e3c5e2d0965c35b68bc3` after #176.
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

The ordered compensated means, normalized covariance/variance, exact constant
fallback, finite representability checks and training MSE rescore preserve the
preceding scalar fit arithmetic. Native fitting and generated Rust share one
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

Three new nonignored tests cover admission, native extraction compatibility and
eleven independent train-and-predict corpora. Every generated model is restored
before predicting. Comparisons include two coefficient words, MSE, complete
winner bytes and 33 prediction words per backend; total 198 scalar word
comparisons and 33 complete-envelope comparisons. Finite extremes, constant
fallback and tiny nonconstant features are included. Every backend also rejects
invalid/nonfinite tables and labels, unrepresentable training loss, all state
prefix truncations, corrupt headers/body, trailing bytes and NaN coefficients.
Compile generated Rust with `rustc --edition=2021 -O`; C++ uses
`-O2 -ffp-contract=off -fno-fast-math`, with sanitizer/C++20 variants above.

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

Combined requalification uses the fresh-main baseline PR2108/29, main4190/66
and all twelve new tests from winner plus recipes: expected PR2120/29 and
main4202/66. All normal after gates, dedicated recipe ASan/UBSan and C++20,
protected-input/skip/Clippy/audit comparisons and exact-head hosted CI plus
fresh independent approval are recorded separately in
`/home/ubuntu/jope-evidence/kernel2/bt/B09_LINEAR_SCALING/rebase-after-176-01`.
The inherited raw CSV entry point remains distinct from `scaled_train`.
