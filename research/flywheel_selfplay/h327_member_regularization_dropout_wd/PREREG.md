# H327 — fixed member regularization (TRAIN only)

Authority: hw_t1700_astra_h327_reg.md, 2026-11-04. H324 is closed at
55c77aa56ccbccfae364a0a84b1c90aa56efe7a7. One fixed REG arm, no tuning.
Hypothesis: stronger ranker regularization reduces concentration of utility gains.
This is LightGBM L2 leaf regularization and feature subsampling, not neural
weight decay or activation dropout; larger minimum leaf populations restrict fits.

## Exact treatment

Pinned H287 export commit 213b30ad52e1ce48927bba960ed609bd98ab4dfe:
`topk.py:fit_ranker` and `bagged.py:fit_ranker_n` both explicitly specify:

| Parameter (native alias) | BASE | REG |
|---|---:|---:|
| reg_lambda (lambda_l2) | 1.0 | max(10*1.0,1.0) = 10.0 |
| colsample_bytree (feature_fraction) | 0.8 | 0.8*0.8 = 0.6400000000000001 |
| min_child_samples (min_data_in_leaf) | 20 | 2*20 = 40 |

Use the literal Python product for feature fraction; do not introduce a rounded
alternative. Exactly these three keyword constants change at each of two
LGBMRanker constructors. Replacing them with BASE values must restore identical
function ASTs. No sample weights. Calls include inner rankers, final ranker refits,
and all eight bag rankers. Each outer-fold/root cell has 819 ranker fits and 195
unchanged gate fits; 36 cells total 29,484 ranker fits, 7,020 gate fits.

Everything else stays BASE: objective=lambdarank, metric=ndcg,
lambdarank_truncation_level=8, boosting_type=gbdt, max_depth=3,num_leaves=7,
subsample=.8,subsample_freq=1,n_jobs=1,verbosity=-1,deterministic=True,
force_row_wise=True; learning_rate=float(lr), topk n_estimators=200,
bag n_estimators=int(selected_n); random_state=int(existing stream seed).
No explicit feature_fraction_seed is present: keep existing random_state and
LightGBM's seed derivation. No explicit label_gain override is present: keep
the pinned installed LightGBM defaults and unchanged ordinal labels.
Selection grids for learning rate/rounds/margin remain unchanged; selected values
may differ as a treatment effect. Veto classifiers retain identical code/params
and no weights; verify all 108 final gate hashes exactly equal BASE.
Raw utility remains np.where(gate,-fitness,-2.0); failure sentinel is -2.0.

## Protocol and isolation

BASE is unchanged H303 NESTED, H308 partitions P0/P1/P2, roots30001–30003,
four original-family outer folds, 1,219 TRAIN contexts/205 families.
Exact-version GET archived H318 BASE, verify object/member SHA, then replay
36 fold/root pick hashes, nine pick/raw-utility panels and three exact medians:
P0 .0643163334285782; P1 .04821480030677507; P2 .04360142642936269.
Any mismatch is REGRESSION_FAIL and stops experimental fitting.
H324 harness is reused only for isolation/custody; its context weights are absent.

Keep original member folds/seeds/features, nested 18-config TAU/k/agreement grid,
inner mean family gain, tie rules and switching rules. Fitting workers read only
their outer-TRAIN pools/cost members/cache byte ranges; historical outer outcomes
are outside their input closure. No outer packets prebuilt. All 36 fsynced locks,
model/config/input audits and 108 gate identity checks must finish before a
separate once-only 36-fold/root REG score. BASE frozen panel reused read-only.
No partial score, no hidden retries or extra arms.

## Frozen metrics and decision

Raw utility relative to r882 slot0. Family-equal means per cell, median across
nine cells; report literal family medians separately. Whole-family paired
bootstrap: 10,000 shared resamples of 205 families, RNG327191, same draws for both
arms and all cells. Trim each arm's top10 families separately, subtract REG-BASE,
then median over nine cells. CI .025/.975; one-sided LCB95 .05.

REG_ADVANCE requires ALL:
1. Paired drop10 median >=+.002 and LCB95>0.
2. REG median gain >=.90*BASE median gain.
3. REG median harmed-family rate <=BASE.
4. Positive paired drop10 median over three partitions for >=2/3 root seeds.

Otherwise REG_NOT_ADVANCED. Zero LCB/root differences do not qualify. Report
drop1/5/10, LOFO minimum, W/T/L tolerance1e-4, harm gain<-.01, invalid/switch counts,
config agreement, seed Jaccard and existing Elo recipe. Exact Elo label:
**reference = r882 slot0 (DL-Train outer); r3175 not measured on TRAIN**.
CONFIG_INSENSITIVE if >=30/36 configs match is diagnostic only, not a decision gate.

Overfit diagnostic, frozen before results: for each of 108 final members per arm,
score its own outer-TRAIN contexts with its final ranker at selected_n, unchanged
gate and selected_tau, selecting with raw_margin (the same margin whose selected
inner-OOF gain is stored in the member receipt). Compute family-equal train gain
minus stored inner_fit_family_gain. Report both gains and gaps by arm/cell and
aggregate, with paired differences. This compares final resubstitution to selected
inner-OOF performance, not untouched validation or a same-model unbiased error
estimate. No bag/ensemble gap is implied, no extra BASE fits, and no outcome-based
selection uses this diagnostic. Inputs are outer-TRAIN only, in a separate
post-fit audit process; no outer score is consumed.

These are dependent cells and conditional bootstrap intervals on an adaptive,
reused TRAIN cohort; not adjusted for the wider wave search. If ADVANCE, separate
one-shot dev91 preregistration is next; no dev91 is part of H327.

## Resources and publication

xbabe2 CPU only; cores4,5,6,8,33,41 and fresh idle/SMT/peer check, <=6workers,
nice19, numeric threads1, systemd MemoryMax6GiB/MemorySwapMax0, aggregateRSS6GiB,
hostCPU<=85%,load1<=85,MemAvailable>=20GiB. Own scratch<=2,000,000,000bytes at
/home/ubuntu/jope-data/h327-astra. Compute stop18:45Z and close19:00Z Nov4;
resume does not reset deadlines. Forecast exceeding stop => INCOMPLETE, no cuts.
Create-only S3 experiments/rd_breakthrough_20261030/H327/astra-20261104T170100Z/;
exact-version GET/SHA/member verification before deleting own scratch.

Prereg published nonforce v1/next before any fit. Private index; own study and
one row each index only. No PR/main/fastlane/CI under freeze/merge hold. No GPU,
qualification.lock, live writers, xbabe0/xbabe3, foreign jobs, DL-Val/dev91,
DL-Test, sealed/holdout/meta-test/fresh cohorts. DOPE:
`{"composite_sha256_prefix":"c51a56d0","loaded":false,"scope":"CPU selection layer only"}`.
Existing permitted cached TRAIN features may be read; DOPE weights unchanged.
