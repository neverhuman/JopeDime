# Search ablation protocol

Internal protocol for kernel2 work item WP-0.4. It says how to compare two configurations of the
search at an equal evaluation budget, what each reported number means, how the
promotion gates are evaluated, and which data the comparison may never touch. The harness is
`jope-bench ablation` (`crates/jope-bench/src/ablation.rs`).

## What is compared

A *profile* is a named configuration of an in-process search that `jope pipeline run` runs:
the shipping search `jope_pipeline::search::run_cuts_with`, or the opt-in kernel2 route
`jope_pipeline::kernel2::search::run_cut`. Profile A is the reference and profile B the
candidate. Both run on the same datasets, the same cuts, the same seeds and the same evaluation
budget. Profile B is always executed, never copied from profile A, so an A=A comparison runs
every search twice.

| Profile | What it runs |
|---|---|
| `legacy` | The shipping search unchanged: `SearchOptions::new(budget, seed)`, one worker, one island, candidates from the v1 canonical-pipeline wrappers of `feat_math::evolve` (12 candidates per round, the best 4 carried with their scores, at most 4 transforms). |
| `kernel2-baseline` | The kernel2 route as `jope pipeline run --search-profile kernel2-baseline` runs it: `run_cut` with the shipped revision-3 profile on one cut, one thread, one island, fault policy `reject-hard` (the route's default) and the comparison's refinement allowance (`--refine-top-k`, `--refine-trials`). |
| `kernel2-experimental` | The same route with the shipped `kernel2-experimental` profile. |

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
  removed; with neither `--scratch` nor `TMPDIR` the harness refuses to run rather than use the
  system temporary directory. Keep evidence outside the checkout.

The default dataset set is the ten committed regression datasets of `datasets/test` (collection
`test-10`, `datasets/manifest.json`): the real-data half of fixture set D5 in `MASTER_PLAN.md`,
already used by `deploy/ci/check.sh`, `emitters.sh` and `e2e-smoke.sh`. Record the document's
`body.datasets` (id, dataset hash, `train.csv` sha256, rows, features, partition sha256) with any
result.

## Paired kernel2 ablation

A kernel2 comparison, two kernel2 configurations or a kernel2 profile against `legacy`, runs both
arms on the same datasets, fold seed, seeds and `--evaluations`, on one cut (the route searches
one cut, as `jope pipeline run` does), with one thread, one island and the route's default fault
policy, and with one explicit refinement allowance for every kernel2 search:

```sh
cargo run --release -p jope-bench -- ablation \
  --profile-a kernel2-baseline --profile-b kernel2-baseline \
  --refine-top-k 1 --refine-trials 4 \
  --datasets datasets/test --seeds 1,2,3,4,5 --evaluations 64 --cuts 1 \
  --jobs 8 --out <evidence dir>/ablation.json
```

`--refine-top-k` (0 to 4) and `--refine-trials` (at most `--evaluations`) are required when either
profile is a kernel2 profile and refused otherwise, and `--cuts` other than 1 is refused, as
`jope pipeline run` refuses them. These refusals, and the checks of `--seeds`, `--evaluations`,
`--cuts` and `--jobs`, are made before the scratch directory is created. Refinement trials are
charged to the same per-cut budget as measurements. Each kernel2 profile record (`body.protocol.profile_a.kernel2`) states the shipped
profile's revision and sha256, the fault policy, the allowance and the selection rule. A kernel2
search that the seconds bound ended, or whose active time overran it, is refused as a legacy one
is. With `--scratch`, each kernel2 search also keeps the route's own evidence
(`search/cut-0/{search,observations,refinements}.json`). A scratch directory that already holds a
kernel2 search of the same slot, dataset and seed (anything at its path, a symbolic link
included) is refused before any partition is written or any search starts, so a rerun never
replaces that evidence; choose a new `--scratch`.

**A kernel2 comparison is not accuracy evidence unless it is reported as paired wins, ties and
losses beside both medians.** The harness writes them to `body.paired` (definitions
`kernel2-ablation-paired.v1`). Runs pair one to one by (dataset, seed) on
`body.runs[].metrics.M1`; two runs of one slot with the same dataset and seed, a run without its
partner in the other slot, and a run on a dataset the comparison does not list are refused. B wins
a pair when its M1 RMSE is lower, loses when it is higher, and ties only when the two are
bit-equal. For the comparison overall and for each dataset, `body.paired` gives:
- W/T/L;
- the median M1 RMSE of A and of B;
- the median and mean paired difference (B minus A);
- the pair with the largest gain, and the mean difference without it;
- a two-sided exact sign test of wins against losses, ties left out (1 when every pair ties);
- how many pairs keep the same M1 program (`program_sha256`).

Every report of the result (PR body, commit message, receipt, ledger) quotes:
- the overall and per-dataset W/T/L with both medians;
- the sign test;
- the same-winner count;
- each slot's wall seconds from `timing.paired` (and `timing.runs[].search_seconds` per run).

An equal evaluation budget is not equal compute: a refinement trial and a measurement take
different time. No median is reported without its W/T/L. G-A's per-dataset means against
σ_seed stay in `body.gates`; they do not replace the paired counts. Results on synthetic or
fixture data are labelled as such and qualify nothing. With A=A every pair ties and the body
replays byte for byte. A legacy-only comparison has neither `body.paired` nor `timing.paired`,
and its document is unchanged.

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
| `body.runs` | One record per (profile slot, dataset, seed): the metrics and each cut's accounting (generations, evaluations, stop reasons). A kernel2 cut also has `kernel2`: measurements, objective and rejected refinement trials, and arm draws; its one phase's stop reason is `exploration_stopped_by`, and `comparison_stopped_by` is `not_requested` |
| `body.aggregates` | Per profile: per-dataset means over seeds with the seed standard deviation of M1, and the mean over datasets |
| `body.gates` | G-A to G-E, each with a status, whether every clause could be evaluated, and its clauses |
| `body.verdict` | `no difference` when every profile-B run record equals its profile-A record, otherwise `difference` |
| `body.paired` | Kernel2 comparisons only: the paired comparison of M1, overall and per dataset (see "Paired kernel2 ablation") |
| `timing` | Wall seconds, per-run M10, mean M10 per profile and G-D's wall-time clause; for a kernel2 comparison also `paired`: each slot's total and median search seconds and the B/A ratio |
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

A kernel2 run fills the same fields. Its ranking is the route's own: the fault-aware validation
score, with each program once. So M1 is the program the route publishes, measured on the one cut.
The route has no comparison phase, so M3, M6 and M11 count every measurement of the cut,
refinement children included. M11's derived origins stay `mutate` and `crossover`.

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

The tests put their scratch directories under `TMPDIR`, or under the workspace's `target/tmp`
when it is not set; never under the system temporary directory.

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
- `ablation::tests::ablation_kernel2_a_equals_a_replays_byte_for_byte` (PR tier, seconds):
  kernel2-baseline A=A on the same two datasets with seed 6 at 16 evaluations, with
  `--refine-top-k 1 --refine-trials 2`, twice with different job counts. The bodies are
  byte-identical. M1 is the route's own winner. The budget accounting adds up, and the paired
  comparison is all ties with p = 1. Seed 5 is left out because it draws 1000-round boosted-tree
  heads, which take 13 to 20 seconds a search in a debug build.
- `ablation::tests::ablation_legacy_document_is_unchanged_by_the_kernel2_arms`: the legacy
  profile record and cut record bytes, and the key sets of a legacy-only run's body, runs and
  timing, are what they were before the kernel2 arms.
- `ablation::tests::ablation_kernel2_refuses_what_the_pipeline_refuses`: the refusals listed
  above, from the library and from the command line.
- `ablation::tests::ablation_execute_refuses_before_creating_scratch`: through `execute`, each
  refused combination of options leaves neither the `--scratch` directory nor `--out` behind.
- `ablation::tests::ablation_kernel2_refuses_a_scratch_that_holds_its_searches`: a rerun into the
  same scratch directory is refused, with every byte and modification time of the first run's
  entries unchanged and nothing added; so is a planned output that is a dangling symbolic link.
- `ablation::tests::ablation_paired_rejects_duplicate_a_and_unpaired_b`,
  `ablation_paired_rejects_unlisted_unpaired_a` and
  `ablation_paired_rejects_a_repeated_dataset_or_an_unknown_slot`: the pairing refuses anything
  but one-to-one pairs on listed datasets.
- `ablation::tests::ablation_paired_summary_arithmetic` and
  `ablation::tests::ablation_sign_test_matches_the_exact_binomial_tail`: the paired arithmetic on
  a synthetic input, including ties, an all-tie input and the bit-equality rule, and the sign
  test against the exact binomial tail.
- `ablation::tests::ablation_kernel2_text_passes_the_shipped_text_lint`: the new option names,
  help, ids and document keys pass `lint_shipped_text`.

## Adding a profile

Append a variant to `Profile` with its id and its record, and route it in `run_one`. A kernel2
configuration adds a variant that `kernel2_options` builds. Never change what `legacy` runs: it
is the reference every later comparison is made against. Record the profile's id and its
definition's sha256 in any receipt that uses it.
