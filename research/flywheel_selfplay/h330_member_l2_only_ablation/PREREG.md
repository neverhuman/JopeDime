# H330 — member L2-only ablation

Authority: hw_t1825_astra_h330_l2only.md, 2026-11-04. H327 is closed at
9b8e5076ed80359d7bc49890c64a32205fbadf10. One fixed L2ONLY arm, no tuning.
Hypothesis: L2 smoothing accounts for H327's gain, while its feature/leaf changes
contributed to harm. H330 isolates the L2 change versus BASE; comparing with the
three-change H327 recipe remains descriptive and cannot identify interactions.

## Exact treatment

Pinned H287 export 213b30ad52e1ce48927bba960ed609bd98ab4dfe:
`topk.fit_ranker` and `bagged.fit_ranker_n` use reg_lambda1.0,
colsample_bytree.8,min_child_samples20. L2ONLY changes **reg_lambda to10.0 only**.
Feature fraction stays .8; minimum leaf population stays20. Restoring the one
keyword constant must restore identical original function ASTs. All inner,
final and eight bag rankers use the change; veto code and parameters remain BASE
(including gate reg_lambda2.0). No sample weights or label_gain override.

Keep LambdaRank, ordinal labels/default label_gain, max_depth3,num_leaves7,
subsample.8/subsample_freq1, learning-rate/round/margin selection procedure,
n_estimators200 for topk and selected rounds for bags, seeds/random_state,
feature packets/family folds, nested18-config TAU/k/agreement grid, mean-family
inner objective, conservative ties and switching rules. Selected hyperparameters
may differ through the unchanged selection algorithm. Numeric threads1,
deterministic=True,force_row_wise=True remain. Failure utility is still
np.where(gate,-fitness,-2.0), sentinel **-2.0**. This is GBDT leaf L2 regularization,
not a neural optimizer update or activation dropout.

## BASE gate, fitting and scoring

TRAIN only:1,219 contexts/205 original families, H308 P0/P1/P2,
roots30001–30003, four original-family outer folds. Exact-version GET the pinned
H318 BASE archive read-only, verify object/member hashes, then reproduce all
36fold/root pick hashes,9pick/raw-utility panels and three exact medians:
P0 .0643163334285782; P1 .04821480030677507; P2 .04360142642936269.
Mismatch => REGRESSION_FAIL, stop before experimental fits. No BASE refits.

Each of36 cells has819 ranker fits and195 unchanged gate fits:29,484 ranker
records and7,020 gate fits total. Verify all108 final gate hashes equal BASE,
the native L2 override and retained feature/leaf values in972 final rankers,
all fitting input hashes and original config-selection rules.

Fitting workers read only their own outer-TRAIN pools/cost members and declared
cache byte ranges. Historical BASE outcomes and the H327 comparison metrics are
outside worker input closure. No outer packets prebuilt. All36 fsynced locks
and the full audit precede a separate once-only36 fold/root L2ONLY score.
BASE frozen outcomes reused read-only. No partial scoring or hidden restart.

## Decision, equality cases and metrics

Raw utility relative to r882 slot0; equal family weight within each cell and
median over nine cell means. Literal family medians are separately reported.
Use10,000 paired whole-family resamples of205 families, RNG330191, shared across
both arms and all nine cells. Drop each arm's own top10 families, subtract
L2ONLY-BASE and take the median over nine cells. CI .025/.975; one-sided LCB95 .05.

L2ONLY_ADVANCE requires ALL:
1. Paired drop10 median **>=+.002** and LCB95 **>0**.
2. Median gain >=90% of BASE.
3. Median harmed-family rate <=BASE (harm is gain<-.01).
4. **Every** root's paired drop10 median over its three partitions **>=0**.

Otherwise L2ONLY_NOT_ADVANCED. Apply the explicit operators literally:
an overall tie (paired median0) or LCB0 cannot advance; exact+.002 satisfies the
margin threshold; a root median0 satisfies the explicitly nonnegative root
floor but never substitutes for the positive overall margin/LCB requirements.
Any negative root fails, even if the other two are positive. This replaces H327's
two-positive-root predicate. Equality cases get dedicated tests before scoring.

Report nine cells/aggregate family-equal gain+CIs, literal median, drop1/5/10,
LOFO min, W/T/L tolerance1e-4, harm, invalid/switch counts, config agreement,
seed Jaccard and original400-iteration Bradley–Terry Elo recipe. Exact label:
**reference = r882 slot0 (DL-Train outer); r3175 not measured on TRAIN**.
CONFIG_INSENSITIVE when>=30/36 configurations match BASE is diagnostic only.

Retain H327's member gap diagnostic: final ranker train resubstitution gain at
selected rounds/tau/raw margin minus stored selected inner-OOF gain,108 final
members per arm. No extra BASE fits/outer inputs; not unbiased holdout error,
not a bag/ensemble gap and not a decision gate.

Show H327 REG and L2ONLY side by side using exact pinned H327 metrics from close
9b8e5076ed80359d7bc49890c64a32205fbadf10. Include all nine cells and aggregate
gain/harm/drop10/Elo/gap. H327 CIs retain historical RNG327191 and H330 uses330191;
do not treat CI differences as an additional paired test. Descriptive only,
no extra decision, tuning or subgroup selection. Preserve the H327 verdict.

These are dependent cells and adaptive, reused TRAIN families. Intervals
condition on fitted policies and do not adjust for the wider wave search.
Any advancement needs a separate dev91 authorization/preregistration; none here.

## Limits, custody and red-team review

xbabe2 CPU only,<=6workers on4,5,6,8,33,41; fresh5second idle/SMT check and actual
H328/H329 affinity exclusion. nice19/math1,systemd MemoryMax6GiB/Swap0,
aggregateRSS<=6GiB,hostCPU<=85%,load1<=85,MemAvailable>=20GiB.
Own scratch<=2GB at /home/ubuntu/jope-data/h330-astra.
Fixed compute stop20:05Z,close20:20Z Nov4; resume never resets either.
Forecast beyond stop => INCOMPLETE, no cut cells or partial scoring.

Create-only S3 experiments/rd_breakthrough_20261030/H330/astra-20261104T182300Z/;
exact-version GET/SHA/member verification before own scratch deletion.
Prereg nonforce v1/next before fits, private index, own study plus one row in each
existing index. No PR/main/fastlane/CI under freeze and merge hold.

Before close red-team: physical leakage, zero DL-Val/outer rows pre-lock,
seed noise/all-root floor, utility direction, no cherry-picked cells, equality
cases, protected inputs untouched, DOPE unchanged, memory and disk receipts.
No GPU/qualification.lock,xbabe0/xbabe3,live writers,foreign jobs,
DL-Val/dev91/DL-Test,sealed/holdout/meta-test/fresh cohorts.
DOPE: `{"composite_sha256_prefix":"c51a56d0","loaded":false,"scope":"CPU selection layer only"}`.
Permitted cached TRAIN features may be read; DOPE weights stay unloaded/unmodified.
