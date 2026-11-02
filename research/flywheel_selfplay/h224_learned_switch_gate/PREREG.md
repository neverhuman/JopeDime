# H224 — learned FIT-only switch gate

Prospective protocol under `hw_t1325_astra_h224.md`. No H224 data fit or treatment
outcome has been observed. Merge this preregistration before native91, smoke,
or full execution. H217 is the pinned reference; H221 is closed NO_ADVANCE.

## Hypothesis and arms

A logistic gate using several outcome-free switch features may retain more of
the BLEND_025 gain than H217's scalar zgap threshold, while controlling harm.
Arms: frozen R882; **LGATE** primary; **GATED_025** reference; **LGATE_SHUF**
placebo. Report paired LGATE minus GATED_025, without replacing the primary.
No adapter or DOPE parameter trains; zero adapter optimizer updates.

GATED_025 preserves H217's `blend`, `zgap`, `tau_grid`, `gated`, and `choose_tau`
function bodies exactly, checked by AST comparison. The HistGBM recipe is the
pinned H217/H124 implementation: 300 iterations, learning rate .08, 63 leaves,
L2=1; 192 unrevealed candidate samples per TRAIN context, fresh sample seed224140.
Inputs are 412 program features plus the existing 48+629 context features;
centered utility targets and FIT-only positive-slope calibration stay unchanged.
The fixed blend is r882 + .25*(calibrated HistGBM cost - r882), float32.

## Cohort and partitions

Use the SHA-pinned H124 packet: 1,219 real-S3 DL-Train contexts /205 families,
and the original 91 DL-Val contexts /19 disjoint families. [provenance.json](provenance.json)
records historical real-S3 custody and dataset identities. Labels are not changed.
No sealed, holdout, meta-test or original outer-test payloads; no live loop writes.

Seeds224101–224103, fold seeds224141–224143. For each seed independently,
permute sorted families with NumPy default_rng and array_split into five groups
of41. For cell k: A=group k, B=group(k+1)%5, FIT=the other123 families. B is not
used to train or choose the learned gate. [fold_manifest.json](fold_manifest.json)
records every partition and overlap against H203/H210/H217/H218/H223. Best-match
means the average across the five new folds of each fold's maximum Jaccard with
an old fold; all old/new partition comparisons must be <=.20. Observed metadata
maximum:0.1927922014576975. These are reused development families, not independent
unseen-family or sealed validation.

Every cell uses one HistGBM trained on its full FIT and three inner HistGBMs,
each omitting one of its three FIT family groups. Gate labels/features use
only that held group's inner predictions: the HistGBM producing a gate-training
row never trained on its family. All five outer exclusions and ten distinct
inner exclusions are fitted per seed. Thus60 logical cell model uses require
45 unique HistGBM fits; identical exclusion/seed fits are reused by hash, as in
H217. No retraining on A, B or DL-Val. A family may be FIT in another cell;
this is cross-fitting, not a claim that A labels remain globally unopened.

## Gate training, C selection and threshold

On each cell's inner-OOF FIT predictions, use H217's FIT-only calibration to
convert HistGBM predictions to costs. Compute the blend's first-argmin choice b
and r882's first-argmin choice a. For contexts where b differs from a, the binary
label is 1 iff true native utility(b) > utility(a), otherwise0; exact ties are0.
Native utility is negative fitness for valid candidates, otherwise -2.

The five gate features contain costs/identities only:

1. H217 zgap, capped at1e6 to encode the constant-reference infinity case.
2. r882's second-smallest cost minus its smallest cost.
3. Stable ascending rank of a under blend costs, divided by pool_size-1.
4. Calibrated HistGBM cost(a) minus cost(b), a predicted utility delta.
5. Candidate-pool size.

Train only switching contexts. StandardScaler is fitted on each gate-training
subset with equal total weight per represented family; LogisticRegression uses
L2/lbfgs, C in [.01,.1,1,10], max_iter1000, tol1e-8, intercept enabled. Its sample
weights give each represented family equal total weight, normalized to sum n.
An empty training subset predicts0; a one-class subset predicts that class's
constant0/1 probability. Reaching1000 iterations is a failure, not a new solver.

Choose C using three inner FIT-family groups: train the logistic gate on two,
predict the third, and minimize equal-family weighted binary log loss over
switching contexts; exact ties choose smaller C. Scaling is fitted only on the
two gate-training groups. This CV is conditional on the fixed, FIT-calibrated
OOF HistGBM feature table; it is not an independently nested estimate of the
whole calibration pipeline. All model/threshold choices remain inside FIT.

For the chosen C, use **gate-OOF probabilities** for threshold selection. Grid:
closed(+infinity), linear probability quantiles0,.1,.25,.5,.75,.9,1 among FIT
switches, and open(-infinity). Open iff probability is strictly greater than
threshold and the picks differ. Maximize equal-family true native FIT gain
subject to <=5% of all123 FIT families losing more than .01. Exact objective
ties choose the larger, more conservative threshold; closed is always feasible.
Fit the final logistic weights on all switching FIT contexts at the chosen C.
No switching contexts means closed. No threshold is changed after scoring.

LGATE_SHUF permutes those binary labels within each FIT switching family using
a fixed H224 seed/cell stream. It repeats the same C CV and OOF threshold rule;
threshold gain/harm still use true FIT utilities. The underlying HistGBMs,
calibration and features are shared. The true dataset/pool labels are untouched.
This placebo and GATED_025 are diagnostic arms, not additional decision arms.

## Baseline, smoke and global stage lock

Before any HistGBM or logistic fit, recompute unchanged r882 on all91 DL-Val
contexts: every ordered float64 native utility and mean1.1377941898164525 must
match exactly. Use the pinned H124/H139 CPU baseline and a SHA-verified r882 COPY.
DOPE composite c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814
is verified before/after baseline and after the study; retain and compare the
full frozen model state hash. No GPU initialization or qualification lease.

Smoke is seed224101/cell0's four registered HistGBMs plus its FIT-only gate fit.
Require all four fit receipts, finite predictions, calibration and gate choices;
score no A or trained DL-Val utility. Reuse these exact four SHA-verified fit
artifacts in the full45-fit plan. Smoke and full run share one attempt and one
absolute deadline. The repeated small logistic gate fit is deterministic; no
extra HistGBM fit, model search or budget reset follows the smoke.

Compute all15 cells' FIT calibrations, final gate weights, C choices and
thresholds. Fsync the global stage1 lock and exact-version S3-recover it before
any A or trained DL-Val scoring. Prediction forwards alone may precede the lock;
outcome scoring may not. Hash-check locked fit artifacts and lock on readout.
Native91 baseline is the expressly authorized pre-lock DL-Val exception.

## Readout and decisions

For each seed, pooled A covers all1,219 contexts once. Compute paired native gain
within each family, then equal-family mean per seed; report the median of three
means. Use10,000 paired-family bootstrap draws shared across seeds, seed224150,
and two-sided95% percentile CI of the median. **CANDIDATE** iff lower CI>0,
all3 seed means>0, and <=10% of families have three-seed mean loss<-.01.
Otherwise NO_ADVANCE. Exact ties fail. H217's p versus .001 is report-only and
explicitly labelled; it does not replace the registered zero-bound criterion.

Prospective numerical correction: H217 computes its point gain with NumPy mean;
a synthetic constant .002 difference over91 contexts rounded above .002 and
incorrectly passed an exact tie. H224 uses math.fsum(delta)/n for the point gain.
The retained failing test and adjacent-representable-value controls bind this
fix before outcomes. Reference gate function bodies and bootstrap are unchanged.

Co-primary DL-Val: each of15 locked cells scores91 contexts; average utilities
across cells per context, then context mean. **PLATEAU_WIN** iff gain versus the
native91 baseline is strictly >.002 and10,000 family-cluster bootstrap lower95%
CI>0 (seed224152; sampled families contribute all their contexts). Otherwise
NO_PLATEAU_WIN. Also report per-seed DL-Val results, without selecting a seed.

Report each arm's paired gain/CI/harm/signs/adoption, LGATE-GATED_025 paired,
and H217's family-grouped Elo anchored r882=1500 (1,000 draws, seed224151).
No AUC, speed verdict, U4 measurement, serving promotion or across-campaign
multiplicity claim. A primary or plateau success does not authorize deployment.

## Resources, publication and custody

xbabe2 CPU only, nice19; monitor and worker inherit the same <=4 logical CPU
kernel affinity. HistGBM OpenMP<=4; BLAS/MKL1; native torch1. Monitor the complete
process tree every .25s: RSS strictly<4GiB, load5<=60, free disk>=300GB,
MemAvailable>=20GiB. Swap-free is recorded. Own run output<600MB. Guards apply
during native baseline, fits, scoring and transport. Stop compute at18:00Z,
leaving30min to close by18:30Z on2026-11-02. One create-only attempt, no retry.

Exact merged source is loaded from Git objects; no new physical checkout/lane
is required. Targeted CPU tests, Ruff, source pins/links and five records gates,
exact-head review and squash merge precede launch. Full manual CI is not claimed
under the owner-directed targeted-check lifecycle. Both indexes receive a
PREREGISTERED row, then the terminal result.

All run artifacts use create-only S3 PUT and exact VersionId GET/SHA256 recovery
under the H224 prefix in [manifest.json](manifest.json). Bulk stays outside Git.
Closeout preserves partial failures as INCOMPLETE, without zero-filled outcomes,
and publishes README, small metrics, both index rows, custody and reviewed PR.

## Prelaunch amendment A1 — transient child sampling

Before any H224 launch or claim, a negative mock reproduced psutil.NoSuchProcess
when a short-lived Git child exits between process-tree discovery and RSS/thread
sampling. Ignore only those already-exited children/threads; retain all live
RSS/affinity/host/deadline stops. No scientific source, inputs, seeds, choices or
budget changes. Execute the exact merged prelaunch-amendment source. The failure
and corrected control are retained in custody. No attempt has been consumed.
