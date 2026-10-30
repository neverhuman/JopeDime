# H09: DSL-preserving conditional residual (development experiment)

Freeze this protocol, the implementation and input roster with `run.py plan` before fitting.
This executes the previously queued H09 idea; it does not recreate an unavailable historical
implementation or establish independent generalization on the repeatedly inspected DL-Val.

Use only the catalog-joined regression contexts in the H49 allowlisted packet, whose source
manifest excludes DL-Holdout and outer rows. Reject sealed-name exclusions and BeyondArena
metadata before opening pools/tables. Group connected catalog-family and packet-family aliases
as one unit, and reject any component crossing DL-Train/DL-Val. Every admitted dataset needs a
recorded catalog family key and matching canonical dataset digest. No fallback family IDs.

Deterministically shuffle sorted training components with seed 53009 into five folds: folds
0–2 fit the tree; fold 3 fits residual coefficients; fold 4 selects regularization. These roles
never overlap. Do not refit either scorer after selection. This deliberately sacrifices some
training breadth to avoid in-sample residual targets or indirect leakage through cross-fit
models trained on a selection family's labels.

Sample at most 64 candidates per context by sorted program SHA, independent of cost/gates.
The baseline is HistGradientBoostingRegressor (128 iterations, 31 leaves, learning rate .05,
L2=1, no early stopping, seed 53009). Inputs retain all 412 DSL columns, plus eight training-table
context summaries: log1p(rows), log1p(columns), missing fraction, mean/std of log1p(column std),
mean absolute column skew, mean absolute column-target correlation, and log1p(target std).
Impute nonfinite table entries with column medians. Context summaries use the delivered
train-only table, never outer/test rows. Training targets are raw constrained fitness centered
within each sampled context. Weight contexts equally within components and components equally.

Residual: Ridge with intercept and alpha in [1, 10, 100], trained on the calibration group's
centered fitness minus the frozen baseline prediction. Retain all standardized 412 DSL main
effects. Add a 16-by-8 bilinear interaction between a fixed seed-53009 Gaussian projection of
standardized DSL features and standardized context summaries; standardizers use calibration
inputs only. This is a non-neural contextual residual; DOPE is unused and unchanged. Choose the
alpha with the lowest mean selected constrained fitness over selection components, with smaller
alpha breaking exact ties. Rank the complete archived pool in each context; program SHA order
breaks score ties. No gate or measured-cost filtering at inference.

Primary contrast: baseline selected constrained fitness minus residual selected constrained
fitness (positive is lift), averaging contexts within connected families and then families.
Terminal DL-Val evaluation occurs once after training/selection and model hashes are frozen.
Use 20,000 paired whole-component bootstrap draws (seed 53010), central 97.5% percentile CI
[.0125, .9875]. No outcome-dependent changes, extra arms, seed searches or repeated DL-Val runs.
Report all context selections and family differences, including failures/harms. An entirely
negative CI is evidence of harm, not a reason to omit the study. A CI excluding zero on the
positive side triggers a handoff to HWV-01 for actual Golden45 regression Elo. No proxy Elo or
binary/multiclass claim. This archive-selection experiment performs zero new native GP fits.

CPU only: nice 10, at most eight math/build threads, host CPU <=85%, check load before run,
30-minute wall limit per train/score process. Outputs under ~/jope-data/hwv-04; no bulk inputs
inside Git. Verify the copied checkpoint's DOPE parameter+normalizer composite c51a56d0 before
and after each phase, and its full file hash. No neural forward/backward or weight writes,
GPU, live-flywheel changes, remote storage writes, holdout/sealed or BeyondArena payload reads.
