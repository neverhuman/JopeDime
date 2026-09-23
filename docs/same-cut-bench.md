# Same-cut benchmark (`jope-bench`)

One cut for everyone. `jope-bench` scores every tier-1 baseline, every program the caller names,
and the program the search finds, on **identical held-out rows** with **one evaluator**, then
follows the honest screen-then-deepen protocol. It is the number the README's benchmark-valid
ladder will be replaced by: reproducible in this tree, from committed data, with a cut identity on
every score. It covers three tasks: regression, binary classification and multiclass
classification.

## What it does

For every dataset directory under `--datasets` (a directory holding `metadata.json` or
`manifest.json` plus a headerless numeric `train.csv`, target last; `datasets/test/` has ten
regression datasets):

1. **Cuts.** Fold membership is `feat_math::fold_of("<dataset_hash>:<seed>", row, k)`, the same key
   and function the product pipeline uses, so a cut here is the pipeline's cut. Each cut gets a
   `jope.cut/v1` identity: the SHA-256 of a canonical record naming the dataset hash, the
   `train.csv` bytes, the key, `k`, the fold and the held-out indices. Only `train.csv` rows are
   cut; the catalog's sealed `test.csv` is never opened. Cuts are not class-stratified: a binary
   cut whose held-out rows happen to be one class has no AUC on that cut (its other metrics and
   its rows are kept), and the dataset is refused up front unless each class has at least two
   rows.
2. **Methods, per task.** Every head the task's grammar accepts as a single-head program:
   regression `baseline.ridge` = `(pipe X (ridge))` and so on for `ridge_gcv`, `elasticnet`,
   `histgbm`, `histgbm2`, `ksmooth`, `ensemble`, `krr`, `gpr`, `stack`; binary `logit_ridge`,
   `logistic`, `tab_boost`, `extra_trees`, `gam_boost`; multiclass `multiclass_ridge`,
   `multiclass_ovr_logistic`, `multiclass_diagonal_lda`, `multiclass_nearest_centroid`; binary
   also `portfolio`, which for a binary task is an ensemble over the other heads that fits with
   no members named. Heads named with `--heads` are checked against the task of every discovered
   dataset before any report is written. Plus
   `--program NAME=<dsl>` or `NAME=@file` for any program you want on the ladder (a program whose
   head does not suit the task is a rejection with the grammar's reason, ranked last), plus
   `--search`, which runs the lite race on the **screen cut's training rows only** and enters its
   best program as `afm.lite-race.v1`. The search is regression-only; with `--search`, datasets
   of other tasks are skipped with a message.
3. **One evaluator.** Parse, fit on the cut's training rows, predict the held-out rows, score.
   Regression programs run on the frozen kernel and score RMSE and R². Binary and multiclass
   programs run on the schema-bound v3 table API over a synthetic all-numeric schema (numeric
   cells rendered shortest-round-trip, so the encoded block equals the matrix bit for bit) and
   score ROC AUC, log loss and Brier, or macro-F1 and accuracy. Parse, fit and predict all run
   under `catch_unwind`; panics and errors are rejections with a reason; non-finite predictions,
   probabilities outside `[0, 1]`, non-integer class labels and non-finite scores are rejections.
   The evaluator is one function in `crates/jope-bench/src/eval.rs`, deliberately not shared with
   the pipeline's private one, so a change there cannot silently move a benchmark.
4. **Screen, then deepen.** Every method is scored on `--screen-fold` (default 0) and ranked by
   the task's primary score there (R², AUC or macro-F1), **raw** (R² uncapped, so a degenerate
   fold still orders methods instead of tying them; two methods far below −1 keep their order).
   On a binary screen cut whose held-out rows are one class no method has an AUC, so the screen
   ranks by the negated log loss instead and the finalists are still the least-bad methods rather
   than an alphabetical pick. The `--top N` (default 5) are re-scored on every cut; the committed number is the pooled
   out-of-fold score over all rows with the campaign's variance penalty: regression
   `max(oof_r2, -1) - 0.25 * min(sd(fold_r2), 1)` (`robust_fitness`, the exact-CV campaign's);
   binary the same on the Gini scale `2·AUC − 1` (`robust_fitness_binary`, the binary campaign
   lane's; folds without an AUC are left out of the penalty); multiclass
   `oof_macro_f1 - 0.25 * min(sd(fold_macro_f1), 1)` (the benchmark's own definition; no campaign
   lane defines one). Method ids must be unique; a duplicate is an error, not two rows.
5. **Reports.** One `jope.bench-report/v2` per dataset: task, score metric, ladder metric and its
   direction, plan, every cut id, every method's screen and deep scores (a primary `score` plus a
   `metrics` map) and timings, host cores and worker count, git head when known.
6. **Ladders.** `jope-bench ladder` turns a directory of reports (one task per directory) into two
   local `jope-arena` tracks under protocol `same-cut` — `<track>-screen` (every method, one cut)
   and `<track>-deep` (finalists, every cut; everyone else `missing`) — so the same Bradley–Terry
   code that ranks the published benchmark ranks this one. The track ranks by RMSE (minimize) for
   regression, ROC AUC or macro-F1 (maximize) for classification; `jope-arena`'s official ladder
   accepts both directions (the published regression fixtures are unchanged).

```sh
cargo run --release -p jope-bench -- run --datasets datasets/test --out target/bench/reports \
  --top 5 --search --search-pop 32 --search-gens 8 --workers 16
cargo run --release -p jope-bench -- ladder --reports target/bench/reports --track-root target/bench/track
cargo run --release -p jope-arena -- elo --track jope-same-cut-v1-deep --root target/bench/track
```

## Committed evidence

`assets/same-cut/v1/` holds the first committed run: the ten `datasets/test` regression datasets,
every default head plus the searched program with `--resample-search` (32 programs × 8 rounds),
k=5 on the pipeline's seed, top 5 deepened, with the tracks, both ladders and their snapshots
(`SOURCES.md` there records the exact commands, the recording tree and host). The run is
deterministic apart from timings. The ten datasets are the `test-10` collection, chosen from the
published ladder where the historical engine ranked first (`docs/datasets.md`): a favourable
selection, used because it is the only committed regression data. Identical cuts make the
comparison between methods fair on those ten; the evidence says nothing about a broader panel, and
the README block says so. The experiment contract's frozen 30-dataset PMLB panel (`docs/asm.md`)
excludes `test-10` on purpose; a same-cut run over that panel is the next evidence step. `crates/jope-bench/tests/committed_evidence.rs` re-derives the
tracks from the reports and both ladders from the tracks byte for byte, and re-evaluates one
dataset's baselines on the same cuts bit for bit; `deploy/ci/readme-benchmarks.py` renders the
README's same-cut block from these files and refuses when they disagree. Regenerate with the
commands in `SOURCES.md` and commit the whole directory together; a changed evaluator, head set
or search budget is a new evidence directory, not an edit.

## Reading the result

- A method's deep row is the honest one; its screen row is what got it there. A method that is
  strong on the screen cut and weak out of fold is exactly what the deep stage exists to catch.
- `afm.lite-race.v1` is fitted and scored like any baseline. Its program was chosen using the
  screen cut's training rows only (the race's inner folds live inside them). Without
  `--resample-search`, its deep score re-fits that one fixed program on every cut: an honest
  number for *the program*, but a biased one for the search, because the program's selection saw
  rows that the other cuts hold out (upward in expectation; on any one dataset the fixed program
  can still land below the resampled number). With `--resample-search`, the deep stage re-runs
  the search on each cut's training rows and scores that cut's own program (per-fold programs are
  recorded in the report); that measures the *procedure* and is the number to quote against
  baselines. It is entered under its own id, `afm.lite-race.v1.resampled`, so a ladder built from
  mixed reports never merges the two numbers. The per-cut search time is recorded as
  `produce_seconds`; a cut whose search fails is a rejected fold naming the cause.
- A dataset whose screen fold is degenerate (a near-constant held-out target gives wildly
  negative R² for every method) is still ordered by raw screen R², so the finalists are the
  least-bad methods rather than an alphabetical tie. If that recurs, screen on two folds.
- No classification dataset is committed yet; `crates/jope-bench/tests/classification.rs` builds
  catalog-layout fixtures in a temporary directory to exercise both tasks end to end. The
  committed regression data stays the only benchmark input in this tree.
- Tier-2 external baselines (the model-zoo report adapters) are a later step: they accept the same
  headerless CSV layout but score themselves from the held-out labels and return metrics only, so
  their rows will be marked as self-reported.

## Files

| path | what |
| --- | --- |
| `crates/jope-bench/src/task.rs` | the three tasks: primary score, ladder metric and direction, default heads, v3 task spec |
| `crates/jope-bench/src/cuts.rs` | fold plan, cuts, `jope.cut/v1` identity |
| `crates/jope-bench/src/eval.rs` | the evaluator (frozen kernel for regression, v3 table API for classification), the three variance-penalty rules, deep (pooled out-of-fold) scoring |
| `crates/jope-bench/src/load.rs` | catalog loading, target checks per task |
| `crates/jope-bench/src/methods.rs` | baselines checked against the task's grammar, caller programs, the search entry |
| `crates/jope-bench/src/protocol.rs` | screen → top N → deep |
| `crates/jope-bench/src/report.rs` | `jope.bench-report/v2` |
| `crates/jope-bench/src/emit.rs` | `scores.csv` rows and track/registry files for `jope-arena`, metric and direction per task |
| `crates/jope-bench/tests/pwlinear.rs` | offline end-to-end on the committed `pmlb-229-pwlinear`, including bit-for-bit agreement of the deep ridge score with `feat_math::cv_r2` on the same key |
| `crates/jope-bench/tests/classification.rs` | offline end-to-end for binary and multiclass on generated fixtures, through the maximize ladder |
