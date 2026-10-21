# GPT6p1_Sol_Max

The algorithm is a regularized linear predictor followed by a histogram tree
expansion of its residuals. All numerical code is original safe Rust, composed
from scalar arithmetic, arrays, comparisons, ordering, and deterministic integer
pseudorandomness. It calls no external fitting routine. One frozen procedure
handles every dataset independently.

## Learning procedure

First validate every supplied cell, training dimensions, and target position.
Targets must be finite; features may be finite or NaN. Training column `C-1` is
the target and the other `C-1` columns are features. Uniform reservoir sampling
retains at most 24,576 training rows, then shuffles them using the supplied seed.
This bound is an explicit statistical/computational tradeoff.

When at least 64 sampled rows and one feature are available, reserve the larger
of 16 rows and one fifth of the sample for internal validation. All feature
screening, scaling, target normalization, and quantile thresholds for that
comparison are fitted exclusively on its internal training rows.

Retain all features when there are at most 96. Otherwise sample up to 512
candidate columns, score marginal value/target and missingness/target squared
correlations on up to 512 internal training rows, retain 72 leaders and 24
randomly selected remaining candidates. This preserves some opportunities for
interactions without marginal effects; it cannot preserve every useful feature.

For each retained column, divide by its largest absolute observed training
value, center and standardize using observed training values, and map missing
values to zero. A centered missing indicator becomes a second linear coordinate.
Tiny variances are treated as constant; standardized values are clipped to
[-20,20]. Quantiles of the original finite training values separately define at
most 64 ordered observed bins, with a distinct missing bin. The linear and tree
representations therefore use the same input columns but different coordinates.

Target normalization divides by the largest absolute training target before
computing its centered second moment. This avoids directly squaring extreme
original-unit labels. The zero and constant cases are explicitly defined.

Fit ridge predictors with penalties 0.0001, 0.01, 1, and 100 by coordinate
descent. The objective is squared residual loss plus `n * penalty` times the
squared coefficient norm; the intercept is unpenalized. Per-coordinate updates
solve the scalar quadratic, followed by a fixed coefficient guard. A deterministic
shape-based pass budget is at most 48, with a small-update stopping rule. Select
the penalty with the lowest internal validation squared error.

Starting from that linear predictor, repeatedly fit a tree to training residuals.
Each tree uses an independently sampled 85% of training rows and a randomly
selected subset of retained columns. At each node, histogram residual sums and
counts determine the split maximizing regularized squared-mean improvement.
Missing values are tried on both sides; endpoint thresholds also support a pure
missing/observed split. Leaves return a residual mean shrunk by a count penalty
of 5 and learning rate 0.06. Trees have depth 4 for fewer than 2,048 fitting rows
and depth 6 otherwise. Minimum leaf count is at least three.

Check every eight trees. On validation rows, analytically minimize squared error
over the scalar blend between the linear baseline and the accumulated tree
correction, constrained to [0,1]. Retain the best tree count and blend, including
the zero-tree baseline. Stop after a fixed no-improvement window or the
deterministic shape-based tree limit, at most 640. The limit approximates work;
it is not a measured runtime guarantee.

After all internal decisions, refit preprocessing, the selected ridge penalty,
and the selected tree count on the complete sampled outer-training rows. This
includes the former inner validation rows, as permitted. Reuse the selected
blend. No outer held-out targets or prediction rows participate in any decision.

If a meaningful validation split is unavailable, use a fixed conservative ridge
penalty and, for at least 12 rows with features, 96 trees with blend 0.5. With no
features or constant targets, the tree correction is disabled and prediction is
the fitted target mean.

## Prediction and state

Prediction validates all supplied features and the fitted width, applies frozen
per-column transforms, evaluates the linear predictor and trees, and restores
target units. An explicit extreme-value guard bounds normalized output; overflow
on restoration saturates to a finite f64 endpoint. This guarantees defined finite
outputs on successful calls but may sacrifice extrapolation for extreme values.

Each prediction depends only on its row and fitted state. Batching and row order
cannot change normalization, model selection, or randomness. Prediction accepts
features only: it never discards another final column. Zero-feature predictions
use the explicit row count with an empty feature slice.

Construction is unfitted. Every fit invalidates earlier state before validation
and restarts from the original seed. An unsuccessful refit therefore cannot
expose stale predictions. There is no global mutable state, file access, clock,
external entropy, thread, subprocess, or cross-dataset cache in the source.

## Expected strengths and limitations

The linear component supplies shrinkage and limited extrapolation; the tree
correction supplies threshold interactions and local nonlinearities. Validation
can suppress a harmful correction rather than requiring it on every task. Explicit
missingness coordinates and split direction cover missing inputs. These are
mathematical expectations, not measured predictive claims.

Feature and row caps can discard rare signals, especially in wide tables.
Marginal screening misses pure interactions. Quantile bins lose fine resolution;
small validation partitions yield noisy choices. Coordinate descent and finite
tree depth constrain approximation quality. Arbitrary numerical category codes
are treated as numerical values. There is no categorical vocabulary, specialized
time-series handling, kernel model, or pretrained representation. The fixed work
limits do not prove compliance with 600 seconds on every input/hardware pair.

The manifest specifies the dominant complexity bounds, including source-side
input validation, preprocessing, all four internal ridge fits, residual tree
selection, and final refitting. Input memory and transport must also be charged
by an organizer. The product same-cut integration does not itself provide the
challenge's isolated runtime supervisor.

## Authorship and checks

The source was written once and frozen at SHA-256
`fe5bb21ed62edb3aebc1bd141d54834dd3f42b97cb9b0366f2e6b24bc99f75a8`
before repository-code inspection or numerical execution. It compiled on the
first post-freeze attempt. Subsequent tests exercise contract correctness only;
no dataset corpus, predictive comparison, tuning experiment, or Elo evaluation
was used. The source bytes remain unchanged after those checks.

File-writing and integration tools were used. This conversation already contained
prior project context, so the authorship record explicitly declines pristine
fresh-session challenge eligibility. `compiled_during_design=false` describes
the frozen-source authoring phase, not the later organizer-style build check.
