# H181: frozen-bank covariate-adjusted selector audit

Authority: `~/nudges/hwave_t0600_h181_tokcodex.md`; owner gpu_tokbench_codex.
No H181 scientific outcomes were opened before this preregistration. H175 was the
motivation; its reused bank makes this exploratory estimator development, not an
independent validation or evidence to overturn a closed study. No serving promotion.

## Hypothesis and fixed estimand

Pre-treatment baseline difficulty predicts counterfactual family gain. Cross-fit
CUPED may lower the minimum detectable effect without changing the +0.001 margin,
LCB95 > 0 requirement or tie rejection. The estimand is the paired, equal-family
mean gain of the SAME policy on the fixed 205-family TRAIN cohort. B is a randomized
41-family sample of that cohort. Its known 205-family pre-treatment covariate means
and stratum masses are fixed before any policy outcomes are used. This is a
finite-cohort estimator study, not an assertion that new-family population moments
are known. Fitted nuisance-coefficient uncertainty is not added to the conditional
bootstrap interval; calibration and measured bias therefore gate any recommendation.

Group 205 sorted TRAIN family IDs into five shuffled groups with fold_seed 181140.
For outer k: A=group k; B=group(k+1)%5; FIT=the other 123 families. FIT coefficients
exclude both A and B outputs. The evaluated B policy's outputs never enter its own
coefficient fit or normalizers. A is unused by H181 selection. This is the H175-style
single-candidate B41 selector (m=1), not the H177 pooled-205, best-of-four/Holm selector.
The comparisons share policies, folds and paired bootstrap draws across estimators.

## Frozen inputs and streams

Use the exact H175 real-S3 TRAIN-derived counterfactual bank (1219 contexts, 205
families), hash-pinned in manifest.json, including all 56 policy cells and original
candidate cost/utility payload. No synthetic DATASETS; simulated policy activation
is explicit. Underlying candidate utility is true gate-passed negative fitness,
otherwise exactly -2.0. DL-Val 91 contexts/19 disjoint families are baseline-only.
No sealed, holdout or meta-test access, no new network training, optimizer or backward.

The existing three source Monte Carlo panels use H175 seeds 175101–175103, 200 draws
per cell, and its 24-candidate slates (first four stored reveals + 20 seeded others)
and full pools. These bank policies are retained exactly, not relabeled as newly
independent model/data seeds. They map one-to-one to NEW bootstrap/activation streams
181101–181103. Full-pool baseline covariates are identical across source seeds;
24-pool covariates may differ because the original slates differ. boot_seed 181141;
all reported paired family CIs use 10000 shared family draws, median of 3 seeds
reduced on axis=0. Fresh 181 streams generate the additional target-power policies.

## Estimators

STRICT: unchanged paired B-family mean and paired-family percentile bootstrap.
CUPED1: g_f - theta (X_f - mean_205 X), X=baseline family utility.
CUPED_K: same formula with four baseline-only covariates: family mean r882 utility,
COUNT of its exact -2.0 contexts, mean top-two predicted COST margin, and log context
count. Predictions always come from frozen r882; no candidate output defines X,
its normalization, strata or target moments. A permutation-invariance test enforces
this API. Each cell/draw's coefficients are centered OLS on FIT123, with FIT-only
standardization, zero coefficient for std<=1e-12 and minimum-norm pinv rcond=1e-10.
These statistical nuisance fits are not network training. Cross-fitting does not
make finite-population bias impossible; measure it rather than assume it is zero.

POSTSTRAT: rank baseline utility into five 41-family strata, stable family-ID tie
break. Weight B observations within each stratum to its known 1/5 population mass.
Use the SAME B-family bootstrap samples and recompute stratum weights per draw.
A missing stratum falls back to that draw's STRICT mean; record fallback counts.
It is descriptive and not eligible for ADOPT under the owner criterion.

All selectors admit only when point gain > 0.001 + 1e-12 and paired-family 95% LCB
> 0. Ties reject. Selection uses 10000 B-family bootstrap draws per seed/fold,
common across estimators/cells; H175 used 1000, so do not claim exact historical
admission replication. Every comparison in H181 uses the same 10000-draw reference.
Adjusted gains are estimators only: selected policy utility and Elo ALWAYS use the
original, unadjusted policy outcome panel. Elo uses context-pairwise wins/ties
(1e-4 tie tolerance) before equal-family aggregation and two-policy BT prior .01,
with r882 fixed at 1500. Family-mean gain signs cannot reconstruct this Elo.

## Simulation, calibration, power and bias

Primary bank grid: slate 24/full × activation context/family × oracle/random_nonoracle
× f in [0,.005,.01,.02,.05,.1,.2], 3 retained source streams × 200 policy draws.
Random_nonoracle is the H175 harmful-mirror mechanism; a bad baseline can sometimes
improve under this mechanism. Conservatively gate ALL such cells regardless of the
sign of their realized population gain. Separately report the admission proportion
among draws whose true 205-family gain <=0, with its actual denominator.

CP calibration is exact one-sided 95% beta.ppf(.95,k+1,n-k), n=200 policy draws PER
seed PER fold. Do not pool overlapping folds into 1000 Bernoulli trials. Gate the
maximum upper bound over all three seeds, five folds, all f=0 and mirror cells.
Even 0/200 has a positive upper bound (~.01487). These are marginal per-cell bounds,
not a simultaneous 95% confidence statement or pooled/Holm error qualification.

Report per-seed values, median and 10000-family-bootstrap conditional 95% CIs for
power, bias and empirical estimator-error variance ratio vs STRICT. Power assigns
each fixed fold's 200-draw admission rate to its 41 B families, then resamples
families; also give 5-fold-block sensitivity intervals. This interval describes the
fixed bank/folds; it does not estimate unseen-cohort generalization. Variance ratio
uses squared/first estimator error moments across policies and fixed B folds,
subtracting each same-policy TRUE 205-family mean gain. Zero/zero variance is 1;
positive/zero is undefined and rejected. Conditional selector-bootstrap variance
ratio is secondary. Bias is the adjusted point estimate minus STRICT, aggregated
equally over five B folds. Gate MAX absolute per-seed cell bias, without canceling
bias across cells/slates/seeds. Include all 56 grid and 12 target-power cells.

MDE80: first tested oracle fraction with median admission power >=.80, in units of
its true finite-205-family utility gain. Report context and family activation for
both slates, no interpolation/extrapolation, explicit right-censoring. Recompute the
first crossing within each of 10000 shared-family bootstrap draws. Censored draws
stay +infinity; null CI endpoint means beyond the tested grid, never dropping draws.

Additional power targets: +.016, +.028, +.042, each slate and activation unit. The
fixed frozen-bank oracle headroom h defines activation probability f=target/h for
each seed. New independent 181 activation streams produce 200 policies, common
uniform variates across targets. Thus EXPECTED finite-family gain is the requested
target; realized draws differ. Report both. Reject unattainable f>1; never scale
utility values to fabricate a desired effect. Oracle policy construction uses real
outcomes as a simulation device and is not a deployable learned policy.

## Preregistered decision and closed-wave diagnostics

ADOPT_CANDIDATE_ESTIMATOR iff CUPED1 or CUPED_K reduces family-activation MDE80 by
>=20% in BOTH slate panels, maximum null/mirror CP upper <=.05, and maximum absolute
per-seed cell bias <.001 (strict epsilon tie rejection). Otherwise NO_GAIN. For a
right-censored STRICT MDE, use the last tested gain as a conservative denominator
bound; two censored MDEs cannot qualify. Report each component even on failure.
This means candidate estimator for a FUTURE fresh-fold preregistered wave, NOT lift,
a closed-wave re-verdict, or a live rule change. Flag a qualifying candidate to the
flywheel/H178 owners; only a later owner ruling can change the live selector.

After the primary run, read H170/H171/H177 closed-wave TRAIN step-40 panels and
report only descriptive STRICT and cross-fit CUPED_K family gains/CIs. No re-verdict,
no posthoc admission claims. H171 CTRL_EX4 uses logical10=actual40, not its logical40
(actual160) panel. H177 uses pooled_b raw four candidate columns, not selected
fallback/primary panels. Their source seeds and correlation with the motivating
cohort are disclosed; they do not influence H181's decision.

## Execution and release

Exact fresh native91 baseline must PASS first: utility bytes, mean exactly
1.1377941898164525, and all cached candidate cost bytes against the restored gold.
All network parameters require_grad=False, DOPE c51a56d0 and FULL state SHA checked
before/after. Optimizer construction, Tensor.backward and autograd.backward raise.
Hash pins may be checked as bytes before the gate; TRAIN outcomes are decoded only
after it. Read-only audit hook restricts input paths; only DL-Val pools are allowed.

One exclusive H181 scientific attempt from merged reviewed source. CPU xbabe2
nice19; math threads1, aggregate worker+monitor threads<=4, RSS<6GB. Terminal stop:
host CPU>85%, MemAvailable<16GiB, disk user-visible >=93.8% OR free<40GB, plus load
>85% online cores. No GPU, xbabe0/3, optimizer updates or live flywheel changes.
Metadata-only constructor smoke has zero model loads/scores. Data-free tests cover
candidate invariance, FIT no-self-use, antithetic synthetic null, scalar OLS and
bootstrap agreement, heterogeneous-seed axis=0, CP/gate boundaries and target support.
Targeted pytest/Ruff + five jope-records gates, one rd_argon cross-review <=15min,
self squash-merge (update branch/admin if only behind), then one guarded run.

Closeout: README hypothesis/setup/outcome/median+CI/verdict/next; compact metrics
<100KB, own overnight index row, versioned create-only S3 raw custody with actual
restore/decode verification, self-merge and preserve/remove ONLY owned worktree.
