# Search ablation protocol

Internal protocol for kernel2 work item WP-0.4. It says how to compare two configurations of the
shipping search at an equal evaluation budget, what each reported number means, how the
promotion gates are evaluated, and which data the comparison may never touch. The harness is
`jope-bench ablation` (`crates/jope-bench/src/ablation.rs`).

## What is compared

A *profile* is a named configuration of the shipping in-process search,
`jope_pipeline::search::run_cuts_with`, the search `jope pipeline run` uses. Profile A is the
reference and profile B the candidate. Both run on the same datasets, the same cuts, the same
seeds and the same evaluation budget. Profile B is always executed, never copied from profile A,
so an A=A comparison runs every search twice.

| Profile | What it runs |
|---|---|
| `legacy` | The shipping search unchanged: `SearchOptions::new(budget, seed)`, one worker, one island, candidates from the v1 canonical-pipeline wrappers of `feat_math::evolve` (12 candidates per round, the best 4 carried with their scores, at most 4 transforms). |

Later kernel2 work adds profiles as new `Profile` variants. The plan's name for the current
search, `legacy-v2`, is this `legacy` profile: the shipping search calls the v1 wrappers, not the
`_v2` ones.

## How to run it

```sh
cargo run --release -p jope-bench -- ablation \
  --profile-a legacy --profile-b legacy \
  --datasets datasets/test \
  --seeds 1,2,3,4,5 --evaluations 64 --cuts 1 \
  --jobs 8 --out <evidence dir>/ablation.json
```

- `--datasets` takes a dataset directory (`metadata.json` or `manifest.json` next to `train.csv`)
  or a directory of them, and may be repeated. Every dataset must be a regression dataset; a
  dataset listed twice is refused.
- `--seeds` are the search's random seeds. The fold seed (`--fold-seed`, default 20260907, the
  pipeline's) stays fixed, so every seed searches the same cuts and the spread over seeds is the
  search's own variability.
- `--evaluations` is the budget of every cut's search, 1 to 1000. The seconds bound is always 600
  seconds, the largest the search accepts, so the evaluation budget ends every phase.
- `--cuts` (1 to 5) searches that many folds of the five-fold partition of `train.csv`. With more
  than one cut the search spends half of each cut's budget on its own exploration and the rest on
  comparing the best programs across cuts.
- `--jobs` runs searches concurrently. The body does not depend on it.
- `--scratch` keeps every search's evidence (`prepared.json`, `search.json`,
  `observations.json`). Without it the evidence goes to a temporary directory under `TMPDIR` and is
  removed. Keep evidence outside the checkout.

The default dataset set is the ten committed regression datasets of `datasets/test` (collection
`test-10`, `datasets/manifest.json`): the real-data half of fixture set D5 in `MASTER_PLAN.md`,
already used by `deploy/ci/check.sh`, `emitters.sh` and `e2e-smoke.sh`. Record the document's
`body.datasets` (id, dataset hash, `train.csv` sha256, rows, features, partition sha256) with any
result.

## Data that may never be used

- **Outer test folds.** The harness reads `train.csv` only, through `jope_bench::load`, which
  verifies its sha256 against the catalog row and never opens `test.csv`. Every score is measured
  on a held-out fold of the training rows.
- **Sealed meta-test families.** Do not point `--datasets` at a sealed release (for example
  `crates/jope-datasets/releases/regression-10-v1`) or at any family kept outside the learning
  worker's input closure (`AGENTS.md`, "Direction and evidence").
- The synthetic D5 fixtures (`datasets/synthetic/*`) are not in the catalog layout the loader
  reads, and each row carries its own split tag, so they are not used.

## Determinism

The same inputs give byte-identical `body` bytes on one host and toolchain. The harness makes
that hold and refuses runs where it cannot:

- Every search runs with one worker, one island and the 600-second bound, so its decisions are a
  function of the rows, the cuts, the budget and the seed.
- A search in which the seconds bound ended any phase, or whose active time overran the bound, is
  refused rather than reported. Lower `--evaluations` if that happens.
- The body holds no wall time, no digest of a file that holds wall time (the search's own
  `search.json` and `observations.json` do), and no path.
- `body_sha256` is the sha256 of the compact body bytes the harness serializes.

`timing` (M10 and the wall-time clause of G-D) and `provenance` (dataset paths, jobs, time) are
outside the digest.

## Output

One `jope.bench-ablation/v1` JSON document:

| Field | Content |
|---|---|
| `body.protocol` | Definitions version, both profiles, seeds, budget, cuts, folds and fold seed |
| `body.datasets` | One record per dataset |
| `body.runs` | One record per (profile slot, dataset, seed): the metrics and each cut's accounting (generations, evaluations, stop reasons) |
| `body.aggregates` | Per profile: per-dataset means over seeds with the seed standard deviation of M1, and the mean over datasets |
| `body.gates` | G-A to G-E, each with a status, whether every clause could be evaluated, and its clauses |
| `body.verdict` | `no difference` when every profile-B run record equals its profile-A record, otherwise `difference` |
| `timing` | Wall seconds, per-run M10, mean M10 per profile and G-D's wall-time clause |
| `provenance` | Tool version, jobs, dataset paths, time |

## Metrics

Definitions version `kernel2-ablation-metrics.v1`. "The ranking" is the pipeline's own selection
ranking (`jope_pipeline::select::ranking`): programs measured on every cut, lowest mean held-out
RMSE first, ties broken on the program text. "Exploration" is a cut's own candidates, in
evaluation order; the cross-cut comparison re-measurements are left out.

| Metric | Definition |
|---|---|
| M1 | The first program of the ranking, the one `jope pipeline run` would emit: its mean held-out RMSE and R², the number of cuts it was measured on, its text and sha256 |
| M2 | Median held-out RMSE and median R² of the first 10 programs of the ranking (fewer when fewer were measured) |
| M3 | Per cut, the 1-based number of exploration evaluations after which the running best held-out RMSE has closed 95% of the gap between the first measured candidate and the final best; a cut that never improves on its first measured candidate reports that candidate's position. Reported per cut and as the mean |
| M4 | Fault rate: `null` until fault telemetry exists |
| M5 | Hard-reject rate: `null` until fault telemetry exists |
| M6 | Exploration candidates whose status is `parse_failed`, `fit_failed`, `score_failed` or `panicked`, over exploration candidates attempted, with the count per status |
| M7 | Over the first 4 programs of the ranking (the search's carried set size): mean stage count, the number of `map` and `derive` stages, and the mean and maximum node count and depth of their expressions (`null` when there are none) |
| M8 | For the M1 program: emitted Python and Rust artifact bytes (`feat_math::transpile::emit`) and token cost (`feat_math::tokens::try_estimate_tokens`); `unsupported` names the reason when the emitters refuse it |
| M9 | Share of new tokens among the best programs: `null` until a profile adds tokens |
| M10 | Wall seconds per evaluation of the search. In `timing`, never in the body |
| M11 | Measured exploration candidates derived from parents (the search's `mutate` and `crossover` origins), per exploration evaluation. A cut never fits a program twice, so every one is unique |

## Gates

The gates of the workplan's promotion ladder (§17.3), evaluated as data with A the reference and
B the candidate. A clause is `pass`, `fail`, `no-difference` (A and B are equal) or
`not-evaluated` (its metric does not exist yet, or a run lacks its value). A gate fails on any failing clause, reports
`no-difference` when every evaluated clause does, and is `not-evaluated` when no clause could be
evaluated; `complete` is false when any clause was not evaluated.

| Gate | Rule as evaluated |
|---|---|
| G-A quality | Per dataset, delta = mean M1 RMSE of B minus A, and sigma = the pooled seed standard deviation of A and B. Pass when no dataset has delta above 0.25 sigma, and either at least two datasets have delta at or below minus one sigma or the mean M3 of B is at most 90% of A's. `not-evaluated` with fewer than two seeds |
| G-B parity | The parity lanes (`crates/feat-math/tests/parity_bits.rs`) must be green with the profile's new elements in the corpus. The search does not evaluate it: `no-difference` when both profiles are the same, otherwise `not-evaluated` |
| G-C soundness | M4 increase at most 0.02 and M5 increase at most 0.01 (`not-evaluated` until fault telemetry exists); M6 increase at most 0.01 |
| G-D cost | M7 mean stages and mean expression nodes non-increasing; M8 token cost and artifact bytes increase at most 5%. An M8 clause is `not-evaluated` when any run of either profile has no M8 value (the emitters or the token estimator refused its best program), also when both profiles lack it; the expression-node clause is `no-difference` when no best program of either profile has an expression. The M10 clause (increase at most 10%) is in `timing` |
| G-E use | M9 above 0: `not-evaluated` until a profile adds tokens |

With A=A every run record of B equals A's, so the verdict is `no difference` and every gate is
`no-difference` or `not-evaluated`.

## Tests

- `ablation::tests::ablation_a_equals_a_replays_byte_for_byte_and_reports_no_difference` (PR tier,
  seconds): A=A on the two narrowest D5 datasets (`rdatasets-agridat-bancroft-peanut-uniformity`
  and `pmlb-first-principles-supernovae-zg`), seeds 5 and 6 and 16 evaluations (the twelve
  first-round candidates and four derived ones), run twice with different job counts. The two
  bodies are byte-identical and every gate says there is no difference. Those seeds were chosen
  because their candidates fit in seconds in a debug build; the search's candidate streams are
  frozen, so the choice stays cheap.
- `ablation::tests::ablation_heavy_a_equals_a_on_every_d5_regression_dataset` (ignored, and it
  refuses to run without `JOPE_ABLATION_HEAVY=1`): A=A on all ten datasets and five seeds at 64
  evaluations, twice. Unoptimised builds fit some candidates slowly, so run it with the release
  profile:

  ```sh
  JOPE_ABLATION_HEAVY=1 cargo nextest run --cargo-profile release -p jope-bench \
    --run-ignored only \
    -E 'test(=ablation::tests::ablation_heavy_a_equals_a_on_every_d5_regression_dataset)'
  ```

## Adding a profile

Append a variant to `Profile` with its id, its record and its call into the search. Never change
what `legacy` runs: it is the reference every later comparison is made against. Record the
profile's id and its definition's sha256 in any receipt that uses it.
