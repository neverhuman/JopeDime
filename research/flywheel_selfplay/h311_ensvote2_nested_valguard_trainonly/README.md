# H311 — nested TRAIN-only VALGUARD

**VALGUARD_NOT_ADVANCED**. Median invalid counts A/B/N:
**15/14/16**.
Paired B−A family-equal median gain **+0.003778 [+0.000502, +0.005077], LCB95 +0.000680**.
B beats N on invalid counts in **7/9** cells.
Failed gates: **invalid_cut_median_ge2**. Shadow research only; no serving or flywheel change.

## Hypothesis and setup

H295's frozen validity-guard recipe may reduce invalid picks when applied to
H303 NESTED, without losing gain, and outperform a shuffled-label guard.
A is H303's mean-objective nested chooser, with H308 P1/P2 partitions added.
B is A plus the H295 guard; N uses the same procedure with validity labels
permuted only within each fitting subset. All nine cells use the same
1,219 TRAIN contexts / 205 families, P0/P1/P2 and root seeds 30001–30003.
Null seeds 311201–311203. Full rules in [PREREG.md](PREREG.md), published
**37c01836b003f759332cbc64116544bcda33c345** before any new fit; implementation
**41f1eb3c32607bea89e99ed54ac6c1fd87da8dc1** frozen before guard fitting/scoring. The prereg
publication verification receipt precedes regression launch in S3 custody.

H308 TRAIN-prefix loader is byte-exact SHA
65a86b6d636461761eaca9bdba413ba5dec34140cd49d4e0ca020aa849a26c6d.
Dependencies and S3 references are in [dependency_pins.json](dependency_pins.json)
and [input_receipt.json](input_receipt.json). Archived refit models and picks
were reused, with all recovered A outer picks exactly matching H303/H308.
No selector refit beyond the mandated fresh P0 regression.

The frozen H295 hybrid averages shallow LightGBM invalid probabilities and
standardized L2 logistic regression (C=.05,max_iter500). Raw 568 and augmented 652
features, fallback candidate order, .055 slot0 escape floor and .015 alternative
risk cap remain fixed. H303's chosen agreement a=2 **or 3** is retained for A;
the wrapper does not silently substitute H295's old two-vote primary choice.

Guard tau in .030/.035/.040/.045/.050/.060 is selected on three family-grouped
inner folds of outer_train. H295's lexicographic feasible/invalid/harm/gain
objective is retained, with owner-required exact ties choosing larger tau.
True training outcomes select thresholds in both B and N. Thresholds and
seed streams are preserved in all 36 fit locks and [metrics.json](metrics.json).

The selector used during threshold tuning is this outer fold's **frozen refit**
scored on outer_train. Its predictions are in-sample; guard probabilities are
OOF. This conditional tuning is a declared limitation. Using other outer-fold
OOF choices instead could import the current outer-test labels through their
selector models. Neither method creates a fresh population.

## Regression and isolation evidence

[regression_receipt.json](regression_receipt.json): fresh P0/fold0/root30001
reproduced choice(.05,5,2), inner gain .09612510193329017, complete inner table,
member model hashes and pick SHA
18fdb2538ceabd9a9e5d07ac4a80cf9ff0eaf7597d9dc5fd1645acf0566bd3aa.
H295's ENS_VOTE2 reference reproduces exact published utility vectors, gains
.053903/.059932/.062204 and invalid 14/13/14. This reference is H287 ENS_VOTE2,
not H295's guarded outcome.

Feature-only reconstruction equals all H295 TRAIN features byte-for-byte
([feature_regression.json](feature_regression.json)). It reads program features
and predicted costs, **zero validity/utility members**, and only TRAIN cache
prefixes: 2,811,012 feature rows, 0 DL-Val feature rows. IDs/offset metadata
identify that prefix. Poisoned outcome-member and non-TRAIN-tail tests enforce
the boundaries.

Unlike historical H308 regression packet construction, the guard fit parent
contains no global label array. Each worker's scoped accessor loads outcomes
only for outer_train and refuses outer-test indices. All 36 task counters are
zero; 144 final B/N model files and taus are fsynced and hash-inventoried before
any guarded outer-test score. Score mode forbids fitting, verifies locks/models,
and reads outer outcomes only after the inventory is complete.
The separate mandated regression process follows the historical all-TRAIN
packet build, and is not counted as zero physical label reads.

## Outcomes

**reference = r882 slot0 (DL-Train OOF); r3175-anchored Elo unmeasured**.
All intervals use 10,000 whole-family draws, RNG 302150, shared across arms,
seeds and partitions. The pooled statistic is the median across nine dependent
cell statistics in each draw. It is not a bootstrap of nine independent datasets.
CI95 is 2.5/97.5 percentiles; LCB95 is 5th percentile. Family-equal means average
contexts within family before averaging families. Ties do not count as wins.

| Arm | Median family-equal gain vs slot0 [CI], LCB | Median invalid | Median harmed | Drop-top5 [CI], LCB | MM BT Elo [CI] |
|---|---|---|---|---|---|
| A | +0.048215 [+0.025157, +0.076452], LCB95 +0.028158 | 15 | 19 | +0.021223 [+0.009900, +0.038235], LCB95 +0.011708 | 1717.59 [1668.23, 1767.57] |
| B | +0.049717 [+0.028512, +0.077442], LCB95 +0.031529 | 14 | 19 | +0.024425 [+0.013271, +0.041543], LCB95 +0.014956 | 1727.10 [1677.28, 1778.25] |
| N | +0.045860 [+0.024350, +0.080068], LCB95 +0.027066 | 16 | 17 | +0.021939 [+0.009343, +0.037143], LCB95 +0.011053 | 1721.92 [1670.36, 1768.50] |

| Partition:seed | Invalid A/B/N | Harmed A/B/N | Paired B−A [CI], LCB | Paired B−N [CI], LCB |
|---|---|---|---|---|
| P0:30001 | 13/12/12 | 13/12/13 | +0.001365 [-0.000215, +0.004068], LCB95 -0.000115 | -0.000106 [-0.004427, +0.003913], LCB95 -0.003575 |
| P0:30002 | 14/12/14 | 18/17/17 | +0.006043 [-0.000283, +0.014756], LCB95 +0.000448 | -0.001905 [-0.022539, +0.012503], LCB95 -0.018638 |
| P0:30003 | 11/10/11 | 19/20/17 | +0.001407 [-0.002101, +0.005288], LCB95 -0.001549 | +0.004536 [-0.001072, +0.011231], LCB95 -0.000204 |
| P1:30001 | 17/16/18 | 23/19/23 | +0.004683 [+0.000969, +0.009501], LCB95 +0.001562 | +0.007038 [+0.001777, +0.013974], LCB95 +0.002333 |
| P1:30002 | 15/15/15 | 21/20/19 | +0.001059 [+0.000099, +0.002506], LCB95 +0.000128 | +0.002426 [+0.000616, +0.004743], LCB95 +0.000830 |
| P1:30003 | 16/14/16 | 17/16/15 | +0.004975 [-0.002122, +0.016984], LCB95 -0.001446 | +0.007259 [-0.002066, +0.020583], LCB95 -0.000995 |
| P2:30001 | 16/14/16 | 22/21/19 | +0.004688 [+0.000363, +0.011790], LCB95 +0.000623 | -0.000992 [-0.017220, +0.010789], LCB95 -0.014148 |
| P2:30002 | 17/16/17 | 16/15/15 | +0.003778 [+0.000329, +0.009442], LCB95 +0.000424 | +0.003046 [-0.000891, +0.008363], LCB95 -0.000356 |
| P2:30003 | 15/15/16 | 22/20/24 | -0.000822 [-0.003194, +0.000763], LCB95 -0.002737 | +0.010481 [-0.000951, +0.029930], LCB95 -0.000179 |

Pooled W/T/L componentwise medians (A/B/N): A=[113, 57, 33], B=[117, 59, 31], N=[112, 57, 32]. These describe cell counts; they are not a new pooled sample.
The median of paired B−A gains (+.003778) differs from the difference of the
two arm medians (+.001503); the preregistered decision uses the former.

Pooled B−N gain: **+0.003046 [+0.001004, +0.005994], LCB95 +0.001315**. Its median per-cell invalid delta is
**-1**. Slot0 invalid count is
17 for every cell. Full per-cell and
per-partition three-seed summaries include W/T/L, invalid excess, harmed counts,
family medians, drop-top5, Elo and paired invalid deltas in metrics.
Drop-top5 removes each arm's five highest family gains and re-trims in each
bootstrap draw. Paired drop5 is a difference of arm trims, not a trim of paired
differences. MM two-arm Bradley–Terry uses symmetric .01 decisive counts.

## Decision and interpretation

The frozen rule requires median invalid reduction>=2; B invalid<N on>=6/9;
paired gain median>=−.002 with pooled LCB95>=−.005; and harmed(B)<=A on>=6/9.
Failure of the null clause has priority and yields VALGUARD_NULL_EQUIVALENT;
otherwise unmet gates yield VALGUARD_NOT_ADVANCED. All gates pass only for
VALGUARD_NESTED_ADVANCE. The recorded verdict is **VALGUARD_NOT_ADVANCED**. Only the median invalid-cut
gate fails: 15 to 14 is a reduction of one, not the required two. B beats N on
invalids in 7/9 cells and has no more harmed families than A in 8/9. Both gain
gates pass. Retain the unguarded H303/H308 chooser; no transfer run is launched.

Absolute null gain mostly inherits the shared A chooser; it does not alone
prove the incremental guard is noise. This wave tests incremental gain and
invalid contrast. Historical H287/H295/H303 design exposure remains. P0's
near-duplicate seeds and all cells' shared families limit effective replication.
H310's dev91 transfer is a separate experiment: no dev91 files are accessed or
its result borrowed as H311 validation. The positive conditional paired gain
and null contrast do not override the preregistered invalid-cut failure. No
post-outcome threshold or decision relaxation is applied.

## Verification, resources and deviations

18 focused tests pass: label-access denial and authorized reads; poisoned labels
and non-TRAIN feature tails; partition salts; fallback risk ordering, a=3 and
slot0 escape; conservative tau ties; family weighting; all decision boundaries;
dependent-cell bootstrap and top5 trimming. Source lint/compilation, lock/model
review, independent result/decision/bootstrap recomputation and exact publication
verification pass. Hosted/full repository CI is NOT_RUN under the owner freeze.

xbabe2 only, fit/score cores 52,56,58,62,74,75 (earlier preparation 52,56,57,58,59,62), nice 19, numerical threads 1, at most 6 workers,
systemd MemoryMax 8 GiB, aggregate RSS guard 8 GiB, host CPU <=85%, MemAvailable >=20 GiB,
scratch <=3 GB; H310 cores/jobs untouched. Fit peak aggregate RSS
4,459,036,672 bytes, observed hostCPU peak 26.2%;
no resource guard trigger. Stop 11:15Z, close 11:30Z Nov 4. GPU/neural updates/
neural forwards=0; DOPE c51a56d0 and live flywheel untouched.

The historical regression helper initially re-exported nine byte-exact H287
source files to its default closed H303 scratch source directory before its
task switched to H311 scratch. No model fit ran there. The executed source
snapshot and export hashes are retained; the replay wrapper now redirects that
setup export into H311 scratch. This operational path correction changes no
algorithm or outcome. Original regression source remains separately archived.
The initial checks launcher could not find ruff on the systemd PATH; using its
absolute path resolved the operational failure, and all 18 tests passed.
No seed, partition or requested estimand was cut; no post-outcome tuning.

## Custody and replay

Create-only S3 prefix
`s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H311/astra-20261104T085652Z/`.
[manifest.json](manifest.json) pins source, inputs and evidence archive;
[s3_manifest.json](s3_manifest.json) pins its exact uploaded version. Models,
locks, predictions, family contrasts/bootstrap draws, launch snapshots and
resource logs are archived. Exact-version GET plus archive and every-member
SHA verification precedes deletion of own scratch. Final publication/cleanup
records are in lifecycle.tar.gz and local custody
`/home/ubuntu/jope-rescue-20260921/h311-astra`. No models are in Git.

For an independently authorized replay, restore source/ and vendor/ from the
archive, with operational receipts/*.py at its root. Run prepare_inputs.py,
regression.py, run_h311.py features, verify_features.py, tests, then
run_h311.py fit; review all 36 locks before run_h311.py score and evaluate_h311.py.
report_summaries.py adds descriptive count medians without changing any outcome
or decision; its raw pre-annotation metrics are retained.
Publication receipt pins are prerequisites. The operational supervisor records
exact commands, environment, source snapshots and fixed deadlines. A later
replay needs a new resource authorization; do not silently reset this campaign.
