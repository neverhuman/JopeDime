# assets/same-cut/v1 — provenance

The first committed same-cut evidence: every tier-1 baseline and the program this repository's
search wrote, scored on identical held-out cuts of the ten committed regression datasets under
`datasets/test/`, by a release build of `jope-bench` from the commit each report names in
`git_sha` (the last commit before the one that adds this directory; its `crates/jope-bench` source
is the one that ships with the evidence). Nothing here was edited by hand. The ten datasets are the `test-10` collection, chosen from the published ladder
where the historical engine ranked first (`docs/datasets.md`): a favourable selection, used because
it is the only committed regression data. The comparison between methods on identical cuts of these
ten is fair; nothing here generalises to a broader panel. `crates/jope-bench/tests/committed_evidence.rs` re-derives the tracks and both
ladders from the reports byte for byte and re-evaluates the committed baselines of one dataset on
the same cuts; `deploy/ci/readme-benchmarks.py` renders the README block from these files and
refuses when they disagree with each other.

## How it was produced

From the repository root, checked out at the commit each report names in `git_sha`, with the
release build of `jope-bench` and `jope-arena` from that commit
(`cargo build --release -p jope-bench -p jope-arena --locked`):

```sh
jope-bench run --datasets datasets/test --out assets/same-cut/v1/reports \
  --top 5 --search --search-pop 32 --search-gens 8 --resample-search --workers 16 \
  --git-sha "$(git rev-parse HEAD)"
jope-bench ladder --reports assets/same-cut/v1/reports --track-root assets/same-cut/v1/track \
  --track jope-same-cut-v1
jope-arena elo --track jope-same-cut-v1-deep --root assets/same-cut/v1/track \
  --out assets/same-cut/v1/leaderboard-jope-same-cut-v1-deep.csv \
  --json assets/same-cut/v1/elo-jope-same-cut-v1-deep.json
jope-arena elo --track jope-same-cut-v1-screen --root assets/same-cut/v1/track \
  --out assets/same-cut/v1/leaderboard-jope-same-cut-v1-screen.csv \
  --json assets/same-cut/v1/elo-jope-same-cut-v1-screen.json
```

The run is deterministic: three full runs on this host (two throwaway runs before this one, then
this one) produced reports identical in every field except the timings, `generated_at_unix`,
`host` and `git_sha`. The fold plan is the product pipeline's (`fold_of("<dataset_hash>:20260907",
row, 5)`); the screen cut is fold 0; the top 5 screened methods are re-scored on every cut; the
search runs with a 32-program pool for 8 rounds on the screen cut's training rows, then again on
each cut's training rows in the deep stage (`--resample-search`, method id
`afm.lite-race.v1.resampled`), so its deep row measures the procedure rather than one program
chosen with knowledge of the other cuts' held-out rows.

## Files

| path | what |
| --- | --- |
| `reports/<dataset_hash>.bench-report.json` | one `jope.bench-report/v2` per dataset: task, metrics, plan, every cut id, every method's screen and deep scores and timings, host stamp, git head |
| `track/` | the `jope-arena` root `jope-bench ladder` wrote: `tracks/jope-same-cut-v1-{screen,deep}/track.json`, `registry/`, `results/jope-same-cut-v1-{screen,deep}/scores.csv` (protocol `same-cut`, metric `rmse`, minimize) |
| `leaderboard-jope-same-cut-v1-deep.csv`, `elo-jope-same-cut-v1-deep.json` | the deep ladder (finalists on every cut, pooled out-of-fold RMSE) and its `jope.arena-elo-snapshot/v2`; the snapshot's `leaderboard_sha256` is the CSV's hash |
| `leaderboard-jope-same-cut-v1-screen.csv`, `elo-jope-same-cut-v1-screen.json` | the screen ladder (every method, one cut) and its snapshot |

Timings in the reports are wall seconds on the recording host under whatever else it was running
(the host block records cores and workers); they are recorded, not controlled. The search seconds
per dataset are the number the README's seconds-per-program line summarises.

## Reproduce and check

```sh
cargo run --release -p jope-bench -- run --datasets datasets/test --out target/same-cut/reports \
  --top 5 --search --search-pop 32 --search-gens 8 --resample-search --workers 16
cargo run --release -p jope-bench -- ladder --reports target/same-cut/reports \
  --track-root target/same-cut/track --track jope-same-cut-v1
cargo run -p jope-arena -- elo --track jope-same-cut-v1-deep --root target/same-cut/track \
  --check --committed assets/same-cut/v1/leaderboard-jope-same-cut-v1-deep.csv
cargo run -p jope-arena -- elo --track jope-same-cut-v1-deep --root assets/same-cut/v1/track \
  --check --committed assets/same-cut/v1/leaderboard-jope-same-cut-v1-deep.csv
```

The first `--check` proves a fresh run reproduces the committed ladder; the second proves the
committed track re-ranks to the committed ladder without running anything.
