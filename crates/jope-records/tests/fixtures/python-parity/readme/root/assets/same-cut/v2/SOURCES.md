# assets/same-cut/v2 — provenance

The second committed same-cut evidence: the ten committed regression datasets under
`datasets/test/`, every tier-1 head, the program this repository's search wrote, and **four
external engines** run as subprocesses on the same cuts (XGBoost, LightGBM, CatBoost and a
scikit-learn random forest, each with its library defaults, seed 0 and one thread, handed a cut's
training rows and its held-out features only, and scored by `jope-bench` from the predictions it
returned), by a release build of `jope-bench` from the commit each report names in `git_sha` (the
commit that carries this directory's bench source; after the rebase merge it stays reachable
through the pull request's head ref rather than `main`). Nothing here was edited by hand. v1
(`../v1`) is the heads-only run and stays committed as its own record.

The ten datasets are the `test-10` collection, chosen from the published ladder where the
historical engine ranked first (`docs/datasets.md`): a favourable selection, used because it is the
only committed regression data. The comparison between methods on identical cuts of these ten is
fair; nothing here generalises to a broader panel.

## How it was produced

From the repository root, checked out at the commit each report names in `git_sha`, with the
release build of `jope-bench` and `jope-arena` from that commit
(`cargo build --release -p jope-bench -p jope-arena --locked`), on Python 3.12.3 with
xgboost 2.1.4, lightgbm 4.6.0, catboost 1.2.10 and scikit-learn 1.7.2 (the versions each adapter
printed for `--version` are recorded on its method in every report):

```sh
jope-bench run --datasets datasets/test --out assets/same-cut/v2/reports \
  --adapter "xgboost=python3 crates/jope-bench/adapters/xgboost_adapter.py" \
  --adapter "lightgbm=python3 crates/jope-bench/adapters/lightgbm_adapter.py" \
  --adapter "catboost=python3 crates/jope-bench/adapters/catboost_adapter.py" \
  --adapter "rf=python3 crates/jope-bench/adapters/sklearn_rf_adapter.py" \
  --top 5 --search --search-pop 32 --search-gens 8 --resample-search --workers 16 \
  --git-sha "$(git rev-parse HEAD)"
jope-bench ladder --reports assets/same-cut/v2/reports --track-root assets/same-cut/v2/track \
  --track jope-same-cut-v1
jope-arena elo --track jope-same-cut-v1-deep --root assets/same-cut/v2/track \
  --out assets/same-cut/v2/leaderboard-jope-same-cut-v1-deep.csv \
  --json assets/same-cut/v2/elo-jope-same-cut-v1-deep.json
jope-arena elo --track jope-same-cut-v1-screen --root assets/same-cut/v2/track \
  --out assets/same-cut/v2/leaderboard-jope-same-cut-v1-screen.csv \
  --json assets/same-cut/v2/elo-jope-same-cut-v1-screen.json
```

The run is deterministic: two full runs on this host produced reports identical in every field
except the timings, `generated_at_unix`, `host` and `git_sha`; the first is what is committed. The
fold plan is the product pipeline's (`fold_of("<dataset_hash>:20260907", row, 5)`); the screen cut
is fold 0; the top 5 screened methods are re-scored on every cut; the search runs with a
32-program pool for 8 rounds on the screen cut's training rows, then again on each cut's training
rows in the deep stage (`--resample-search`, method id `afm.lite-race.v1.resampled`). Each engine
is fitted once per cut on that cut's training rows (a fresh process each time); its wall time per
cut is recorded as its fit time.

Reproducing the engine rows needs the same library versions; the heads, the search and the ladder
re-derivation do not. The heads-only v1 run remains the reproduction target that needs no Python
library.

## Files

| path | what |
| --- | --- |
| `reports/<dataset_hash>.bench-report.json` | one `jope.bench-report/v2` per dataset: task, metrics, plan, every cut id, every method's screen and deep scores and timings (adapter methods carry their `adapter` spec with the recorded engine version), host stamp, git head |
| `track/` | the `jope-arena` root `jope-bench ladder` wrote: `tracks/jope-same-cut-v1-{screen,deep}/track.json`, `registry/`, `results/jope-same-cut-v1-{screen,deep}/scores.csv` (protocol `same-cut`, metric `rmse`, minimize; engine rows are noted `external adapter <name> (<version>), controller-scored`) |
| `leaderboard-jope-same-cut-v1-deep.csv`, `elo-jope-same-cut-v1-deep.json` | the deep ladder (finalists on every cut, pooled out-of-fold RMSE) and its `jope.arena-elo-snapshot/v2`; the snapshot's `leaderboard_sha256` is the CSV's hash |
| `leaderboard-jope-same-cut-v1-screen.csv`, `elo-jope-same-cut-v1-screen.json` | the screen ladder (every method, one cut) and its snapshot |

Timings in the reports are wall seconds on the recording host under whatever else it was running;
they are recorded, not controlled.

## Reproduce and check

```sh
cargo run --release -p jope-bench -- run --datasets datasets/test --out target/same-cut/reports \
  --adapter "xgboost=python3 crates/jope-bench/adapters/xgboost_adapter.py" \
  --adapter "lightgbm=python3 crates/jope-bench/adapters/lightgbm_adapter.py" \
  --adapter "catboost=python3 crates/jope-bench/adapters/catboost_adapter.py" \
  --adapter "rf=python3 crates/jope-bench/adapters/sklearn_rf_adapter.py" \
  --top 5 --search --search-pop 32 --search-gens 8 --resample-search --workers 16
cargo run --release -p jope-bench -- ladder --reports target/same-cut/reports \
  --track-root target/same-cut/track --track jope-same-cut-v1
cargo run -p jope-arena -- elo --track jope-same-cut-v1-deep --root target/same-cut/track \
  --check --committed assets/same-cut/v2/leaderboard-jope-same-cut-v1-deep.csv
cargo run -p jope-arena -- elo --track jope-same-cut-v1-deep --root assets/same-cut/v2/track \
  --check --committed assets/same-cut/v2/leaderboard-jope-same-cut-v1-deep.csv
```

The first `--check` proves a fresh run (same library versions) reproduces the committed ladder; the
second proves the committed track re-ranks to the committed ladder without running anything.
