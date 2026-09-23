# Same-cut benchmark (`jope-bench`)

One cut for everyone. `jope-bench` scores every tier-1 baseline, every program the caller names,
and the program the search finds, on **identical held-out rows** with **one evaluator**, then
follows the honest screen-then-deepen protocol. It is the number the README's benchmark-valid
ladder will be replaced by: reproducible in this tree, from committed data, with a cut identity on
every score.

## What it does

For every dataset directory under `--datasets` (a directory holding `metadata.json` or
`manifest.json` plus a headerless numeric `train.csv`, target last; `datasets/test/` has ten):

1. **Cuts.** Fold membership is `feat_math::fold_of("<dataset_hash>:<seed>", row, k)`, the same key
   and function the product pipeline uses, so a cut here is the pipeline's cut. Each cut gets a
   `jope.cut/v1` identity: the SHA-256 of a canonical record naming the dataset hash, the
   `train.csv` bytes, the key, `k`, the fold and the held-out indices. Only `train.csv` rows are
   cut; the catalog's sealed `test.csv` is never opened.
2. **Methods.** Every distinct head of the registry as a single-head program
   (`baseline.ridge` = `(pipe X (ridge))`, and so on for `ridge_gcv`, `elasticnet`, `histgbm`,
   `histgbm2`, `ksmooth`, `ensemble`, `krr`, `gpr`, `stack`), plus `--program NAME=<dsl>` or
   `NAME=@file` for any program you want on the ladder, plus `--search`, which runs the lite race
   on the **screen cut's training rows only** and enters its best program as `afm.lite-race.v1`.
3. **One evaluator.** Parse, fit on the cut's training rows, predict the held-out rows, score
   RMSE and R². Parse, fit and predict all run under `catch_unwind`; panics and errors are
   rejections with a reason, and non-finite predictions or scores are rejections. The evaluator
   is one small function in `crates/jope-bench/src/eval.rs`, deliberately not shared with the
   pipeline's private one, so a change there cannot silently move a benchmark.
4. **Screen, then deepen.** Every method is scored on `--screen-fold` (default 0) and ranked by
   its raw held-out R² there (uncapped, so a degenerate fold still orders methods instead of
   tying them). The `--top N` (default 5) are re-scored on every cut; the committed number is the
   pooled out-of-fold RMSE/R² over all rows with the campaign's variance-penalised score
   `max(oof_r2, -1) - 0.25 * min(sd(fold_r2), 1)` (`robust_fitness`). Method ids must be unique;
   a duplicate is an error, not two rows.
5. **Reports.** One `jope.bench-report/v1` per dataset: plan, every cut id, every method's screen
   and deep scores and timings, host cores and worker count, git head when known.
6. **Ladders.** `jope-bench ladder` turns a directory of reports into two local `jope-arena`
   tracks under protocol `same-cut` — `<track>-screen` (every method, one cut) and
   `<track>-deep` (finalists, every cut; everyone else `missing`) — so the same Bradley–Terry code
   that ranks the published benchmark ranks this one.

```sh
cargo run --release -p jope-bench -- run --datasets datasets/test --out target/bench/reports \
  --top 5 --search --search-pop 32 --search-gens 8 --workers 16
cargo run --release -p jope-bench -- ladder --reports target/bench/reports --track-root target/bench/track
cargo run --release -p jope-arena -- elo --track jope-same-cut-v1-deep --root target/bench/track
```

## Reading the result

- A method's deep row is the honest one; its screen row is what got it there. A method that is
  strong on the screen cut and weak out of fold is exactly what the deep stage exists to catch.
- `afm.lite-race.v1` is fitted and scored like any baseline. Its program was chosen using the
  screen cut's training rows only (the race's inner folds live inside them). Without
  `--resample-search`, its deep score re-fits that one fixed program on every cut: an honest
  number for *the program*, but an **optimistic** one for the search, because the program's
  selection saw rows that the other cuts hold out. With `--resample-search`, the deep stage
  re-runs the search on each cut's training rows and scores that cut's own program (per-fold
  programs are recorded in the report); that measures the *procedure* and is the number to quote
  against baselines. The per-cut search time is recorded as `produce_seconds`.
- A dataset whose screen fold is degenerate (a near-constant held-out target gives wildly
  negative R² for every method) is still ordered by raw screen R², so the finalists are the
  least-bad methods rather than an alphabetical tie. If that recurs, screen on two folds.
- Regression only in this version. Binary, multiclass and timeseries datasets are skipped with a
  message; they go through the v3 table API in a later step.
- Tier-2 external baselines (the model-zoo report adapters) are a later step: they accept the same
  headerless CSV layout but score themselves from the held-out labels and return metrics only, so
  their rows will be marked as self-reported.

## Files

| path | what |
| --- | --- |
| `crates/jope-bench/src/cuts.rs` | fold plan, cuts, `jope.cut/v1` identity |
| `crates/jope-bench/src/eval.rs` | the evaluator, `robust_fitness`, deep (pooled out-of-fold) scoring |
| `crates/jope-bench/src/methods.rs` | baselines, caller programs, the search entry |
| `crates/jope-bench/src/protocol.rs` | screen → top N → deep |
| `crates/jope-bench/src/report.rs` | `jope.bench-report/v1` |
| `crates/jope-bench/src/emit.rs` | `scores.csv` rows and track/registry files for `jope-arena` |
| `crates/jope-bench/tests/pwlinear.rs` | offline end-to-end on the committed `pmlb-229-pwlinear`, including bit-for-bit agreement of the deep ridge score with `feat_math::cv_r2` on the same key |
