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
  system temporary directory. Keep evidence outside the checkout. One comparison at a time holds
  a scratch directory (`.ablation-lock`), and `--out` is checked before it is written; see
  "Reservation, marker and `--out` admission".

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
replaces that evidence; choose a new `--scratch`. **Each comparison with a kernel2 arm takes its
own `--scratch`.** Every search of a dataset shares its partition,
`prepared/<dataset hash>/prepared.json`, and a later comparison into the same scratch would
rewrite it, so the `prepared_sha256` of an earlier document would no longer name the file. A
comparison with a kernel2 profile therefore needs an unused scratch directory: absent (it is
created) or an empty directory that is not a symbolic link. Anything else, also a scratch that
holds only an earlier comparison with other seeds and the same fold seed, a stray file, a symbolic
link, a regular file or a path that cannot be inspected, is refused before any evidence is
written; at most the comparison's own lock file is created in the scratch and removed again.
`--scratch` given on the command line is created empty when it is absent, and a run without
`--scratch` uses a fresh temporary directory.

**Reservation, marker and `--out` admission: what is enforced, and what is not.**

- *Reservation.* Every comparison, legacy-only included, creates its scratch directory when it is
  absent and then creates `<scratch>/.ablation-lock` atomically (it fails if anything is already
  at that path). It holds the lock while it checks the scratch, partitions the datasets and runs
  the searches, and removes it, and nothing else, when the comparison returns, also after a
  refusal or an error. A second comparison on the same scratch, in the same process or another,
  is refused while the lock file exists (`another comparison holds this --scratch`). The lock file
  records the holder's process id and start time; it holds no evidence.
- *Kernel2 marker.* Under the reservation, a comparison with a kernel2 profile requires the
  scratch to hold nothing but its own lock file, then creates the permanent marker
  `<scratch>/.ablation-kernel2` before any other write. The marker is never removed. Every later
  comparison, legacy-only included, refuses a marked scratch as its `--scratch` before it writes
  any evidence (`holds a kernel2 comparison's evidence; use a new --scratch`).
- *`--out` admission.* `jope-bench ablation` checks `--out` before it creates the scratch or
  computes anything. After the reservation is released, it creates `--out`'s directory if it is
  missing, checks `--out` again, and only then writes the document. The write is rename-based:
  - the bytes go to a new temporary file in `--out`'s directory (`.<name>.tmp-<pid>-<n>`, created
    only if nothing is at that path), which is flushed and synced;
  - that file is then renamed over `--out`, and `--out`'s directory is synced after the rename,
    so on success the rename has been written as far as the filesystem honours a directory sync
    (no power-loss test was made). A failure to sync the directory fails the command with the
    document already at `--out`;
  - an existing `--out` is replaced, never truncated, so another hard link to it (for example
    into a kernel2 scratch) keeps its bytes, and a hard-linked `--out` is admitted;
  - the replacement is a new file: it has the default mode (from the umask) and owner of a new
    file, not the old file's permissions, owner, ACLs or extended attributes, and a read-only
    `--out` is replaced when its directory is writable;
  - on an error before the rename only the temporary file is removed (a failure to remove it
    leaves it behind).

  It refuses (`--out refused (<reason>)`):
  - a `--out` whose file name is `.ablation-lock` or `.ablation-kernel2`, for every comparison:
    those names are reserved, since a document there would itself lock or mark its directory;
  - a `--out` that is a symbolic link, or exists and is not a regular file;
  - one that, with its symbolic links and `..` resolved, lies in a directory that holds
    `.ablation-kernel2` or `.ablation-lock`, checked from the directory that would hold the
    document up to the filesystem root;
  - one whose path has a `..` after a directory that does not exist yet;
  - for a comparison with a kernel2 profile, one inside its own scratch.

  A legacy-only comparison writes any other `--out`, one inside its own legacy scratch included,
  with the same document bytes as before. "As before" here and in the limits means the same
  document bytes and scratch layout, not the same file: a replaced `--out` is a new file, with
  the mode, owner and sync described above.
- *Limits.*
  - Writers outside the harness are not prevented. Neither is a program that calls the library's
    `write_document` itself, which writes in place; only `jope-bench ablation` checks `--out` and
    writes by rename.
  - The checks and the rename are separate steps. A change made to `--out`'s directory between
    the second `--out` check and the rename is not prevented.
  - A crash can leave `.ablation-lock` behind. Remove it by hand, and only when no comparison is
    running.
  - A legacy-only comparison on a scratch that only legacy-only comparisons used behaves as
    before (same document bytes and scratch entries). It writes no marker, and a later
    legacy-only comparison into the same scratch is still admitted and rewrites
    `prepared.json`, so a legacy-only document's `prepared_sha256` is not protected. The
    scratch holds the same entries afterwards, since the lock file is gone.
  - Of the harness's own writes, only the document is followed by a directory sync. No
    crash-durability claim is made for the lock file, the marker or the scratch's evidence.

**A kernel2 comparison is not accuracy evidence unless it is reported as paired wins, ties and
losses beside both medians.** The harness writes them to `body.paired` (definitions
`kernel2-ablation-paired.v1`). Runs pair one to one by (dataset, seed) on
`body.runs[].metrics.M1`; two runs of one slot with the same dataset and seed, a run without its
partner in the other slot, and a run on a dataset the comparison does not list are refused. So is
an M1 RMSE that is not finite (NaN or an infinity, on either side), before any comparison, and a
summary number that would leave the range of an f64; every number in `body.paired` is finite. B
wins a pair when its M1 RMSE is lower, loses when it is higher, and ties only when the two are
bit-equal. For the comparison overall and for each dataset, `body.paired` gives:
- W/T/L;
- the median M1 RMSE of A and of B;
- the median and mean paired difference (B minus A);
- the pair with the largest gain, and the mean difference without it;
- a two-sided exact sign test of wins against losses, ties left out (1 when every pair ties);
- how many pairs keep the same M1 program (`program_sha256`), whatever their scores: a tie with
  different programs does not count, a win or loss with the same program does.

The sign test is exact as a test; its p is an f64. A unanimous split of n pairs gives exactly
2^(1-n) down to the smallest positive f64, 2^-1074 at n = 1075; from n = 1076 on the exact value
is at most half of that and the p is 0. A p under 2^-1022 is subnormal and carries fewer
significant bits.

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
- `ablation::tests::ablation_kernel2_refuses_a_used_scratch_with_a_new_fold_seed`: the review's
  reproducer, a second comparison into the same scratch directory with search seed 7 and fold
  seed 20260908, is refused with every entry unchanged, also with a legacy slot A, and the first
  document's `prepared_sha256` still names `prepared.json`.
- `ablation::tests::ablation_kernel2_refuses_a_used_scratch_with_the_same_fold_seed`: a new seed
  with the same fold seed is refused too; each comparison with a kernel2 arm takes its own
  `--scratch`.
- `ablation::tests::ablation_kernel2_refuses_a_scratch_that_is_not_unused`: a scratch holding only
  `prepared/`, one holding a stray file, a symbolic link to an empty directory and a regular file
  are refused with nothing changed.
- `ablation::tests::ablation_kernel2_admits_an_unused_scratch`: an existing empty directory, an
  absent path and the directory `execute` creates for `--scratch` are admitted.
- `ablation::tests::ablation_legacy_refuses_a_scratch_a_kernel2_comparison_used` and
  `review03_legacy_followup_preserves_kernel2_preparation`: a legacy-only comparison after a
  kernel2 one in the same scratch is refused with every entry unchanged, and the first document's
  `prepared_sha256` still names `prepared.json`.
- `ablation::tests::ablation_kernel2_refuses_a_scratch_a_legacy_comparison_used` and
  `ablation_legacy_reruns_into_a_legacy_scratch_as_before`: a kernel2 comparison after a
  legacy-only one is refused; a legacy-only one after a legacy-only one is admitted with the
  body bytes of a fresh scratch.
- `ablation::tests::ablation_concurrent_comparisons_on_one_scratch_admit_exactly_one` and
  `review03_concurrent_library_runs_preserve_both_preparations`: comparisons started together on
  one scratch admit exactly one; the scratch then holds the winner's evidence alone.
- `ablation::tests::ablation_a_lock_file_refuses_and_a_comparison_removes_its_own`,
  `ablation_reservation_is_released_while_unwinding` and
  `ablation_kernel2_marker_persists_and_legacy_writes_none`: an existing (stale) lock file is
  refused and named, a comparison removes its own lock after success, after a refusal and while
  unwinding, and only a kernel2 comparison leaves the marker.
- `ablation::tests::ablation_kernel2_refuses_a_symlinked_scratch_ending_in_a_slash`: a symbolic
  link spelled with a trailing `/` is refused as a kernel2 scratch.
- `ablation::tests::review04_execute_output_preserves_marked_kernel2_evidence` and
  `ablation_execute_refuses_an_out_in_a_kernel2_scratch`: a later comparison's `--out` in a
  kernel2 comparison's scratch, directly, through a symbolic link or through a `..`, is refused
  with nothing written, and the first document's `prepared_sha256` still names `prepared.json`.
- `ablation::tests::ablation_execute_refuses_an_out_inside_its_own_kernel2_scratch`,
  `ablation_execute_refuses_an_out_under_a_lock_file` and
  `ablation_execute_refuses_an_out_that_is_not_a_regular_file`: the self-collision, a `--out`
  under `.ablation-lock`, and a `--out` that is a symbolic link or a directory are refused before
  anything is created.
- `ablation::tests::ablation_execute_writes_a_legacy_out_as_before`: a legacy-only `--out` in an
  ordinary directory, or inside its own legacy scratch, is written with the same document bytes
  as before.
- `ablation::tests::ablation_execute_refuses_a_reserved_out_name`: a `--out` named
  `.ablation-lock` or `.ablation-kernel2` is refused before anything is created, for a
  legacy-only comparison inside its own legacy scratch (absent or empty) or in an ordinary
  directory (also with a trailing `/`), and for a kernel2 comparison; `.ablation-lock.json` is
  admitted.
- `ablation::tests::ablation_execute_replaces_an_out_hard_linked_to_kernel2_evidence` and
  `ablation_execute_replaces_an_out_hard_linked_to_an_ordinary_file`: a `--out` that is a hard
  link to a kernel2 scratch's `prepared.json` or `search.json`, or to an ordinary file, receives
  the document as a new inode; the other link keeps its bytes and no temporary file is left.
- `ablation::tests::ablation_execute_overwrites_an_existing_legacy_out`,
  `ablation_document_replacement_writes_the_same_bytes` and
  `ablation_document_replacement_removes_its_temporary_file_on_failure`: an existing `--out` is
  replaced by exactly one document; the rename-based write produces `write_document`'s bytes for a
  legacy and a kernel2 document; and a failing rename leaves no temporary file.
- `ablation::tests::ablation_document_replacement_syncs_its_directory_and_keeps_the_bytes`: with
  the directory sync after the rename, the document still has `write_document`'s bytes, in a new
  nested directory and over its own earlier document, with nothing else left in the directory;
  a replaced read-only `--out` gets the mode of a new file. Durability itself is not tested.
- `ablation::tests::ablation_paired_refuses_a_nonfinite_m1`,
  `ablation_paired_refuses_a_summary_that_is_not_finite` and
  `review02_public_paired_refuses_nonfinite_m1`: equal NaNs, equal infinities, an infinity or NaN
  against a finite value, and a summary that would overflow are refused.
- `ablation::tests::ablation_paired_rejects_duplicate_a_and_unpaired_b`,
  `ablation_paired_rejects_unlisted_unpaired_a` and
  `ablation_paired_rejects_a_repeated_dataset_or_an_unknown_slot`: the pairing refuses anything
  but one-to-one pairs on listed datasets.
- `ablation::tests::ablation_paired_summary_arithmetic` and
  `ablation::tests::ablation_sign_test_matches_the_exact_binomial_tail`: the paired arithmetic on
  a synthetic input, including ties, an all-tie input and the bit-equality rule, and the sign
  test against the exact binomial tail.
- `ablation::tests::review05_same_winner_means_program_identity_including_score_ties`: a tie with
  different programs is not the same winner and a win with the same program is (in the fixture
  above every tie also shares its program, so it cannot tell the two rules apart).
- `ablation::tests::ablation_sign_test_is_exact_down_to_the_smallest_positive_f64`: a unanimous
  split of 1074, 1075 and 1076 pairs gives 2^-1073, 2^-1074 and 0 bit for bit, and every split up
  to 120 pairs matches an independent integer oracle.
- `ablation::tests::ablation_kernel2_text_passes_the_shipped_text_lint`: the new option names,
  help, ids and document keys pass `lint_shipped_text`. Only that new text is checked. Text that
  predates the kernel2 arms (some of the command's help and the legacy record's keys) is not, and
  not all of it passes.

## Adding a profile

Append a variant to `Profile` with its id and its record, and route it in `run_one`. A kernel2
configuration adds a variant that `kernel2_options` builds. Never change what `legacy` runs: it
is the reference every later comparison is made against. Record the profile's id and its
definition's sha256 in any receipt that uses it.
