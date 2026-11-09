# H331 — Cosine shrinkage schedule for LambdaRank members

**COS_NOT_ADVANCED — TRAIN-only research.** Config diagnostic: **CONFIG_SENSITIVE**.
BASE pooled gain **+0.048214800**, 95% CI **[0.025276707614465443, 0.0761290260180709]**;
COS **+0.044254239**, 95% CI **[0.01583011072700834, 0.07073980425605442]**.
Paired drop-top-10 median **-0.009062808**, 95% CI **[-0.014972303164691022, -0.002475180066142429]**,
one-sided 95% LCB **-0.013855909**. Per-cell list, P0 roots 30001–30003,
then P1, then P2: `[-0.00798197720209713, -0.009062807682737994, -0.011485150674361826, -0.0013309925976530846, -0.013980615859964669, -0.009286139015183076, -0.004583345850283239, -0.004082420260538227, -0.01789174495649175]`. Per-root medians across three partitions:
`[-0.004583345850283239, -0.009062807682737994, -0.011485150674361826]`. Criteria `{'seed_consistency': False, 'paired_drop10_margin': False, 'gain_retention_90percent': True, 'harmed_nonincrease': True}`; failed `['seed_consistency', 'paired_drop10_margin']`.
Gain retention **0.917856**; median harmed-family
rate BASE **0.092683** / COS **0.078049**.

DOPE: `{"composite_sha256_prefix": "c51a56d0", "loaded": false, "scope": "CPU selection layer only"}`. DOPE weights were not loaded or
modified. Only permitted TRAIN ranges of existing cached feature vectors were
read. No neural optimizer update, serving promotion or dev91 shot is claimed.

## Hypothesis, exact recipe and unchanged components

The hypothesis is that larger early boosting updates and smaller late updates
improve family robustness. Early trees learning broad structure is a hypothesis,
not an established mechanism. The only treatment is a ranker fit callback:
`f(t) = base_lr * (1 + cos(pi*t/T))`, t0..T-1, with no restarts or warmup.
base_lr remains in the original .03/.05/.10 grid; f(0)=2*base_lr,
f(T/2)=base_lr and f(T-1)=base_lr*(1-cos(pi/T)).

**The literal discrete formula is not exactly mean-preserving:** its average is
base_lr*(1+1/T), or1.02/1.01/1.005 times base_lr for T50/100/200. This discrepancy
was disclosed in PREREG before any fit; no normalization was substituted.
`topk.fit_ranker` has T200 for inner and final fits, including fits later scored
using an inner-selected50/100-tree prefix. `bagged.fit_ranker_n` uses T equal to
its selected50/100/200 rounds. Prefix scoring does not renormalize the schedule;
early prefixes therefore have a higher mean shrinkage and differ from a shorter
bag refit's schedule. This is not an equal-total-shrinkage timing-only experiment.

LightGBM reset_parameter runs before each iteration at order10. An order11
observer checks both native Booster and callback-environment learning_rate,
records a complete trace hash/count/endpoints/mean, and does not alter fitting.
Removing the callbacks keyword restores exact BASE function ASTs. All constructor
parameters remain BASE: reg_lambda1.0,colsample_bytree.8,min_child_samples20,
LambdaRank/ordinal labels/defaultlabel_gain,max_depth3,num_leaves7,subsample.8,
subsample_freq1,n_jobs1,deterministic/force_row_wise,original seeds/features/folds.
The learning-rate/round/margin selection and18-config TAU/k/agreement grid,
inner objective,ties and switches remain unchanged. Selected values may differ
as a treatment effect. Veto code/parameters/callers remain BASE, including
reg_lambda2.0. No sample weights or neural optimizer update.
Failure utility remains np.where(gate,-fitness,-2.0), sentinel **-2.0**.

Each of 36 cells has819 ranker fits plus195 unchanged gate fits. Full audit verified
**29,484 ranker fit records**, actual fitting multiset
sizes, per-round native callback trace hashes, all BASE constructor settings, and native schedule parameters/tree shrinkages in972 frozen final rankers, and all input identities.
Native tree shrinkages after the first tree are checked against the schedule;
the first-tree serialization convention is excluded from that text check, while
its effective learning rate is verified through the native callback parameters.
All **108/108 final gate hashes** match
BASE exactly. Changed final model hashes: **972/1080**.
[Regularization code](regularization.py), [AST transforms](regularization_transform_preview.json),
[lock audit](lock_review.json), [preflight](preflight_receipt.json).

## Preregistration and evaluation closure

[PREREG.md](PREREG.md) published at **63230cf0de71f1e84afd56e9de83136ccb387fc3** before any fit.
Fitting source **a06a14af46ef29577aa9f8d849398eb0be48606d**, scoring/statistics source
**6dc265fc024e72ffea46c466e002f8ebd7f3d14d**, each before its execution. H327 template close
9b8e5076ed80359d7bc49890c64a32205fbadf10; byte-pinned H310/H303/H302/H287:
[vendor pins](vendor_pins.json).

BASE is unchanged H303 NESTED on H308 P0/P1/P2 × roots 30001–30003 × 4 outer folds.
Offline copies came from an exact-version H318 S3 archive, with whole-object and
member hashes verified: [input receipt](input_receipt.json),
[archive pin](base_archive_pin.json). Frozen BASE replay matched36 fold/root
hashes,9 pick/raw-utility panels,3 medians exactly: P0 +.0643163334285782,
P1 +.04821480030677507, P2 +.04360142642936269.
[Regression receipt](regression_receipt.json). No BASE refit.

Each worker physically reads only its outer-TRAIN pools, cost members and cache
byte ranges. Held-out packets are not prebuilt; historical BASE outcomes cannot
be opened by fitting workers. All packets match BASE input hashes.
**outer_test_rows_read_before_lock=0; outer_test_packets_built_before_lock=0**.
All 36 fsynced locks,1,080 model hashes and full parameter/config/input audit precede
the separate once-only36 fold/root COS scoring calls. [Lock receipt](lock_receipt.json),
[score receipt](score_receipt.json). Subsequent analysis uses saved outputs only.

## Metrics and frozen decision

1,219 TRAIN contexts / 205 families. Within-family context means have equal family
weight. Pooled gain is the median of 9 cell mean family gains; literal family
medians are separately reported. Cells share families and are dependent.
W/T/L tolerance 1e-4; harmed means gain < -.01; invalid/switch counts are contexts.
LOFO minimum equals drop-top-1, not independent corroboration.

10,000 paired whole-family draws RNG 331191 share sampled 205 family indices across
arms and cells. Sort/trim each arm separately, then take the median 9 paired
drop-top-10 differences. CI uses .025/.975, one-sided 95% LCB .05. These intervals
condition on fitted policies and do not repeat fitting or model selection.
This is an adaptive research wave on reused TRAIN families, not an independent
confirmatory experiment; intervals do not adjust for the wider wave search.
Advance requires paired median >= +.002 and LCB > 0, full gain >= 90% BASE, no median
harm increase, and strictly positive per-root medians over3 partitions on at least2of3 roots.
Exact+.002 satisfies the explicit margin; overallzero differences andzeroLCBs
fail. Zero root medians do not count as positive. CONFIG_INSENSITIVE is a separate diagnostic
when >= 30 of 36 (TAU,k,a) choices match BASE; it never relaxes the decision.

**reference = r882 slot0 (DL-Train outer); r3175 not measured on TRAIN**.
Elo retains the 400-iteration Bradley–Terry MM recipe, pseudocount .01, ties omitted.
Median Elo BASE **1725.66** / COS **1700.47**.

### BASE

| Partition/root | Mean gain | Literal median | 95% CI | W/T/L | Harmed | Invalid | Switches | Drop1/5/10 | LOFO min | Elo |
|---|---:|---:|---|---|---:|---:|---:|---|---:|---:|
| P0/30001 | +0.041289223 | +0.000000000 | [+0.022013981, +0.064471746] | 88/93/24 | 13 | 13 | 219 | +0.034482/+0.022905/+0.013174 | +0.034482 | 1725.66 |
| P0/30002 | +0.064316333 | +0.001588084 | [+0.015648636, +0.132207823] | 108/67/30 | 18 | 14 | 300 | +0.037100/+0.018970/+0.008633 | +0.037100 | 1722.48 |
| P0/30003 | +0.072257657 | +0.006984224 | [+0.023508763, +0.140520729] | 122/50/33 | 19 | 11 | 320 | +0.045081/+0.027514/+0.016621 | +0.045081 | 1727.10 |
| P1/30001 | +0.048214800 | +0.002999623 | [+0.018149035, +0.091573285] | 112/57/36 | 23 | 17 | 327 | +0.030657/+0.018311/+0.010198 | +0.030657 | 1697.13 |
| P1/30002 | +0.048657946 | +0.003624258 | [+0.021885114, +0.083927421] | 116/51/38 | 21 | 15 | 275 | +0.035309/+0.023340/+0.014063 | +0.035309 | 1693.84 |
| P1/30003 | +0.038294238 | +0.003253142 | [+0.015828013, +0.060816800] | 112/64/29 | 17 | 16 | 285 | +0.033333/+0.020954/+0.012702 | +0.033333 | 1734.68 |
| P2/30001 | +0.043601426 | +0.006326451 | [+0.017471315, +0.071527689] | 122/48/35 | 22 | 16 | 315 | +0.037280/+0.021223/+0.010678 | +0.037280 | 1716.88 |
| P2/30002 | +0.031839978 | +0.003624258 | [+0.010227565, +0.054021487] | 112/65/28 | 16 | 17 | 292 | +0.026502/+0.014887/+0.008774 | +0.026502 | 1740.78 |
| P2/30003 | +0.054720630 | +0.008144684 | [+0.033151965, +0.079556032] | 129/41/35 | 22 | 15 | 335 | +0.048595/+0.033993/+0.023605 | +0.048595 | 1726.57 |

### COS

| Partition/root | Mean gain | Literal median | 95% CI | W/T/L | Harmed | Invalid | Switches | Drop1/5/10 | LOFO min | Elo |
|---|---:|---:|---|---|---:|---:|---:|---|---:|---:|
| P0/30001 | +0.054412801 | +0.000000000 | [+0.013707832, +0.119625726] | 81/102/22 | 14 | 15 | 178 | +0.027090/+0.012217/+0.005192 | +0.027090 | 1726.37 |
| P0/30002 | +0.052207140 | +0.000000000 | [+0.004763711, +0.119461921] | 100/73/32 | 24 | 15 | 236 | +0.024932/+0.009778/-0.000430 | +0.024932 | 1697.90 |
| P0/30003 | +0.044366416 | +0.003457407 | [+0.010994659, +0.079711369] | 112/57/36 | 24 | 15 | 313 | +0.034847/+0.016369/+0.005135 | +0.034847 | 1697.13 |
| P1/30001 | +0.031666804 | +0.000620003 | [+0.012938778, +0.052157743] | 104/76/25 | 16 | 16 | 242 | +0.025785/+0.016584/+0.008867 | +0.025785 | 1747.58 |
| P1/30002 | +0.017425767 | +0.000000000 | [+0.004852899, +0.032509389] | 47/142/16 | 8 | 16 | 83 | +0.012698/+0.004095/+0.000083 | +0.012698 | 1687.12 |
| P1/30003 | +0.022728529 | +0.000000000 | [+0.004000688, +0.040564758] | 80/100/25 | 15 | 17 | 204 | +0.019974/+0.010135/+0.003416 | +0.019974 | 1702.01 |
| P2/30001 | +0.044254239 | +0.003230383 | [+0.010282270, +0.089527344] | 111/59/35 | 24 | 16 | 311 | +0.026677/+0.013850/+0.006095 | +0.026677 | 1700.47 |
| P2/30002 | +0.025508572 | +0.000000000 | [+0.005712298, +0.045103377] | 88/95/22 | 15 | 17 | 186 | +0.020821/+0.011505/+0.004691 | +0.020821 | 1740.76 |
| P2/30003 | +0.045867168 | +0.005026830 | [+0.012408646, +0.082633217] | 115/51/39 | 21 | 13 | 301 | +0.034677/+0.017581/+0.005714 | +0.034677 | 1687.82 |

Median switched-set seed-Jaccard BASE **0.451380** /
COS **0.360758**;36 within-fold root-pair
comparisons per arm, empty/empty=1. Action Jaccard is separately labelled.
Exact config agreement **0.194444**;
parameter agreement `{'tau': 0.3333333333333333, 'k': 0.6111111111111112, 'a': 1.0}`.

## Member train versus inner-validation diagnostic

For108 final members per arm, measure train resubstitution family-equal raw gain
at the member's selected rounds, tau and **raw margin**, then subtract its stored
selected inner-OOF family gain. The raw margin matches the stored inner metric,
which precedes strict admission. This diagnostic excludes bags and ensemble
aggregation. It is not an unbiased same-model holdout error: the validation value
was selected inside member fitting, and final refits see more training families.
No extra BASE fit and no outer rows are used.

BASE mean train/inner/gap:
`+0.171081136` /
`+0.082073005` /
`+0.089008131`.
COS mean train/inner/gap:
`+0.165380847` /
`+0.079562714` /
`+0.085818133`.
Paired COS-BASE gap change: mean `-0.003189998`,
median `-0.006220087` across108 matching members.
A narrower gap alone does not qualify an arm for advancement.
[Full216-member diagnostic](overfit_summary.json) and [measurement code](overfit.py)
retain both components, per-cell summaries and source model hashes.
All216 gaps were independently reconstructed from saved TRAIN picks and raw
utility pools; no model rescoring was needed for numerical review.
[metrics.json](metrics.json),40,871bytes, contains
all nine cell outcomes, paired/per-root statistics and compact gap summaries.

## Verification, resources and custody

32 focused tests passed (11 fitting/boundary, 21 analysis/gap). Full
parameter/input/config audit and [numerical recalculation](numerical_review.json)
passed. This is same-agent review and independently implemented recalculation,
not independent reviewer approval. Actual fixture constructors match BASE exactly, while the callback rate trace
and predictions change; fixtures preserve rounds/seeds and reach every ranker route.
The first wrapper expected10 tests after11 had passed, so it exited1. The fixed
count passed all11 controls in checks3; both attempts and the bookkeeping
correction are preserved. This was before experimental fits. The original idle
gate subsequently passed without any launcher change or retry.
No experimental fit restart, deadline extension or partial scoring.

xbabe2 CPU, nice 19, numeric threads 1, cores 4/5/6/8/33/41, disjoint from live peer affinities
and SMT siblings at launch. Actual MemoryMax 6 GiB/Swap 0, aggregate RSS 6 GiB cap,
host CPU 85%, load1 <= 85, MemAvailable 20 GiB floor, own scratch 2 GB cap.
Peak sampled RSS through analysis **3,842,174,976bytes**,
hostCPU **30.4%**; no guard violations.
Original compute 22:15Z/close 22:30Z retained. [Resource receipt](resource_receipt.json),
[dependencies](dependency_pins.json). Full repo/hosted CI NOT_RUN under owner freeze.

36 create-only model archives exact-version GET/SHA/member verified:
[archive index](model_archive_index.json). Final evidence/manifest/lifecycle use
`s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H331/astra-20261104T195400Z/`.
[Manifest](manifest.json), [S3 receipt](s3_manifest.json). Own scratch is removed
only after complete recovery mapping and no active consumer; foreign work is retained.
[TRAIN provenance](train_input_provenance.json) preserves allowed range/packet
identities. Original local H50/H58/H276 S3 versions remain unestablished; whole
mixed-cache feature files were not read. Full refitting additionally needs those
retained inputs; archived outputs support numerical replay. Published code is an
exact run artifact, not a standalone checkout CLI or a new replay authorization.

Cross-critique: H327's three-parameter regularization treatment did not identify
L2 independently; H330's isolated L2 change also failed its fixed advancement
criteria (close13ed6793a6566d50a683bace2959aa48929847db). Neither tested a
shrinkage schedule. H331 preserves their BASE constructor settings and changes
the callback only. A negative COS outcome does not disprove all schedules, and
a positive one would not establish that late trees specifically fit high-spread
families. The finite mean and selected-prefix effects are additional mechanisms.
TRAIN drop10/205 and dev10/19 trim different fractions and change cohort/incumbent.
This adaptive wave establishes no dev91 or serving result. Selected OOF vectors
and aggregate grid means are retained, not a complete full-grid policy cache for
future bootstrap reselection.

Peer-wave distinction: H328 LGS had paired drop10+0.011376 and LCB+0.001973,
but median harm rose .092683->.131707. Its positive tail evidence should not be
collapsed into a claim that every prior wave lacked tail gain. H329 DART had
pooled gain+.058526 but paired drop10-.000167/LCB-.005541, only one positive
root and harm.126829; higher pooled gain did not satisfy the robustness gates.
Both remained NOT_ADVANCED under their full conjunction. [Pinned peer metrics
and interpretation](peer_crosscritique.json) retain exact source commits/hashes.
These descriptions use their historical RNGs and introduce no new COS gate,
tuning, formal between-wave test or alteration of the fixed H331 protocol.

## Red-team disposition

Leakage: fitting workers were barred from outer pools, archived BASE outcomes;
all36 locks precede scoring. Seed noise: every root and
partition is reported, at least2 root medians must be strictly positive, and dependent cells
are not represented as independent trials. Direction: higher utility/gain and
LCB are better; harmed-family rate must not rise. Cherry-picking: one fixed arm,
all9 cells, no post-outcome threshold or selection change. Schedule truth: literal
discrete mean and selected-prefix caveats stated before fits, all-round native
rates checked; no claim of exact mean preservation. Ties: overall zero gain/LCB never advance; exact boundary
operators are tested and stated above. Protected inputs: no dev91/DL-Val/DL-Test,
sealed/holdout/meta-test/fresh-cohort reads or live writer action; DOPE remains
unloaded. Resources: sampled RSS/CPU/load/memory/scratch guards plus kernel6GiB/
swap0 enforced; final custody verifies exact versions and all members before
own scratch removal. Evidence does not imply a serving or neural-learning qualification.
Do not advance this fixed cosine schedule to dev91. Keep unchanged ENS_VOTE2 as the reference; this does not rule out every capacity or regularization change.
