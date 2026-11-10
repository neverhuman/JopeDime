# H327 — Fixed L2, feature and leaf regularization for LambdaRank members

**REG_NOT_ADVANCED — TRAIN-only research.** Config diagnostic: **CONFIG_SENSITIVE**.
BASE pooled gain **+0.048214800**, 95% CI **[0.02488447283232094, 0.07672455624640766]**;
REG **+0.058845230**, 95% CI **[0.03129394137821197, 0.09027847505066222]**.
Paired drop-top-10 median **+0.003291436**, 95% CI **[-0.0024340445409658925, 0.012041961210468828]**,
one-sided 95% LCB **-0.001328323**. Per-cell list, P0 roots 30001–30003,
then P1, then P2: `[-0.006113031781011809, 0.007570617694695825, 0.0015914013199372218, 0.012525407593748365, 0.011061041379431162, 0.0003530006696565221, 0.0032914361822550652, 0.008955221607742491, -0.00037085356824931304]`. Per-root medians across three partitions:
`[0.0032914361822550652, 0.008955221607742491, 0.0003530006696565221]`. Criteria `{'seed_consistency': True, 'paired_drop10_margin': False, 'gain_retention_90percent': True, 'harmed_nonincrease': False}`; failed `['paired_drop10_margin', 'harmed_nonincrease']`.
Gain retention **1.220481**; median harmed-family
rate BASE **0.092683** / REG **0.107317**.

DOPE: `{"composite_sha256_prefix": "c51a56d0", "loaded": false, "scope": "CPU selection layer only"}`. DOPE weights were not loaded or
modified. Only permitted TRAIN ranges of existing cached feature vectors were
read. No neural optimizer update, serving promotion or dev91 shot is claimed.

## Hypothesis, exact recipe and unchanged components

The hypothesis is that stronger member regularization distributes gain beyond a
few high-spread families. REG changes exactly three ranker constructor values:
reg_lambda (lambda_l2) 1.0 -> 10.0; colsample_bytree (feature_fraction) .8 ->
.8*.8 = .6400000000000001; min_child_samples (min_data_in_leaf) 20 -> 40.
These are LightGBM L2 leaf regularization, feature subsampling and a minimum leaf
population, not neural weight decay, activation dropout or larger tree capacity.
No tuning or regularization grid, and no sample weights.

Only topk.fit_ranker and bagged.fit_ranker_n constructors change; inner, final and
all eight bag rankers use them. Restoring the three BASE values gives identical
ASTs. H287 commit213b30ad52e1ce48927bba960ed609bd98ab4dfe is pinned.
LambdaRank objective, ordinal labels/default label_gain, max_depth3,num_leaves7,
row subsampling, seed derivation, feature packets, original family folds,
learning-rate/round/margin selection procedure and 18-config TAU/k/agreement grid,
inner objective, ties and switches stay unchanged. Selected values may differ
as a treatment effect. Veto classifiers retain BASE code and parameters,
including their existing reg_lambda2.0; no REG overrides or weights reach gates.
Failure utility remains np.where(gate,-fitness,-2.0), sentinel **-2.0**.

Each of 36 cells has819 ranker fits plus195 unchanged gate fits. Full audit verified
**29,484 ranker fit records**, actual fitting multiset
sizes, the three native REG parameters in972 frozen final rankers, and all input identities.
All **108/108 final gate hashes** match
BASE exactly. Changed final model hashes: **972/1080**.
[Regularization code](regularization.py), [AST transforms](regularization_transform_preview.json),
[lock audit](lock_review.json), [preflight](preflight_receipt.json).

## Preregistration and evaluation closure

[PREREG.md](PREREG.md) published at **0750d1ca461b396d5e16448d47378c652c8eb178** before any fit.
Fitting source **7e790362cb97065c4c79beefd32dd424b02f557d**, scoring/statistics source
**fdc7ec1b53de64b32724a8410f3b1d2b666179ee**, each before its execution. H324 template close
55c77aa56ccbccfae364a0a84b1c90aa56efe7a7; byte-pinned H310/H303/H302/H287:
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
the separate once-only36 fold/root REG scoring calls. [Lock receipt](lock_receipt.json),
[score receipt](score_receipt.json). Subsequent analysis uses saved outputs only.

## Metrics and frozen decision

1,219 TRAIN contexts / 205 families. Within-family context means have equal family
weight. Pooled gain is the median of 9 cell mean family gains; literal family
medians are separately reported. Cells share families and are dependent.
W/T/L tolerance 1e-4; harmed means gain < -.01; invalid/switch counts are contexts.
LOFO minimum equals drop-top-1, not independent corroboration.

10,000 paired whole-family draws RNG 327191 share sampled 205 family indices across
arms and cells. Sort/trim each arm separately, then take the median 9 paired
drop-top-10 differences. CI uses .025/.975, one-sided 95% LCB .05. These intervals
condition on fitted policies and do not repeat fitting or model selection.
This is an adaptive research wave on reused TRAIN families, not an independent
confirmatory experiment; intervals do not adjust for the wider wave search.
Advance requires paired median >= +.002 and LCB > 0, full gain >= 90% BASE, no median
harm increase, and positive per-root median over 3 partitions on >= 2 roots.
Zero differences/LCBs do not advance. CONFIG_INSENSITIVE is a separate diagnostic
when >= 30 of 36 (TAU,k,a) choices match BASE; it never relaxes the decision.

**reference = r882 slot0 (DL-Train outer); r3175 not measured on TRAIN**.
Elo retains the 400-iteration Bradley–Terry MM recipe, pseudocount .01, ties omitted.
Median Elo BASE **1725.66** / REG **1725.67**.

### BASE

| Partition/root | Mean gain | Literal median | 95% CI | W/T/L | Harmed | Invalid | Switches | Drop1/5/10 | LOFO min | Elo |
|---|---:|---:|---|---|---:|---:|---:|---|---:|---:|
| P0/30001 | +0.041289223 | +0.000000000 | [+0.021591302, +0.064521769] | 88/93/24 | 13 | 13 | 219 | +0.034482/+0.022905/+0.013174 | +0.034482 | 1725.66 |
| P0/30002 | +0.064316333 | +0.001588084 | [+0.015453373, +0.133616604] | 108/67/30 | 18 | 14 | 300 | +0.037100/+0.018970/+0.008633 | +0.037100 | 1722.48 |
| P0/30003 | +0.072257657 | +0.006984224 | [+0.022951123, +0.142312689] | 122/50/33 | 19 | 11 | 320 | +0.045081/+0.027514/+0.016621 | +0.045081 | 1727.10 |
| P1/30001 | +0.048214800 | +0.002999623 | [+0.017984246, +0.093039981] | 112/57/36 | 23 | 17 | 327 | +0.030657/+0.018311/+0.010198 | +0.030657 | 1697.13 |
| P1/30002 | +0.048657946 | +0.003624258 | [+0.021804906, +0.085258207] | 116/51/38 | 21 | 15 | 275 | +0.035309/+0.023340/+0.014063 | +0.035309 | 1693.84 |
| P1/30003 | +0.038294238 | +0.003253142 | [+0.016395889, +0.060858556] | 112/64/29 | 17 | 16 | 285 | +0.033333/+0.020954/+0.012702 | +0.033333 | 1734.68 |
| P2/30001 | +0.043601426 | +0.006326451 | [+0.016978593, +0.071919555] | 122/48/35 | 22 | 16 | 315 | +0.037280/+0.021223/+0.010678 | +0.037280 | 1716.88 |
| P2/30002 | +0.031839978 | +0.003624258 | [+0.010257387, +0.054156089] | 112/65/28 | 16 | 17 | 292 | +0.026502/+0.014887/+0.008774 | +0.026502 | 1740.78 |
| P2/30003 | +0.054720630 | +0.008144684 | [+0.033031458, +0.079575263] | 129/41/35 | 22 | 15 | 335 | +0.048595/+0.033993/+0.023605 | +0.048595 | 1726.57 |

### REG

| Partition/root | Mean gain | Literal median | 95% CI | W/T/L | Harmed | Invalid | Switches | Drop1/5/10 | LOFO min | Elo |
|---|---:|---:|---|---|---:|---:|---:|---|---:|---:|
| P0/30001 | +0.058845230 | +0.001772249 | [+0.015759071, +0.115130738] | 109/64/32 | 22 | 12 | 297 | +0.037995/+0.019395/+0.007061 | +0.037995 | 1712.87 |
| P0/30002 | +0.066161544 | +0.008960451 | [+0.023943556, +0.132830949] | 131/34/40 | 29 | 16 | 375 | +0.038954/+0.026153/+0.016203 | +0.038954 | 1706.05 |
| P0/30003 | +0.062351487 | +0.007648344 | [+0.028757821, +0.108247750] | 125/47/33 | 22 | 13 | 345 | +0.045515/+0.028820/+0.018212 | +0.045515 | 1731.32 |
| P1/30001 | +0.059427347 | +0.005803946 | [+0.031223281, +0.091878108] | 121/51/33 | 23 | 13 | 321 | +0.049982/+0.034184/+0.022723 | +0.049982 | 1725.67 |
| P1/30002 | +0.057126015 | +0.005708694 | [+0.035577470, +0.082339273] | 118/64/23 | 11 | 13 | 267 | +0.050946/+0.036106/+0.025124 | +0.050946 | 1784.00 |
| P1/30003 | +0.039076334 | +0.000160687 | [+0.019767787, +0.061522392] | 103/77/25 | 13 | 13 | 260 | +0.032259/+0.021309/+0.013055 | +0.032259 | 1745.91 |
| P2/30001 | +0.051716030 | +0.006223684 | [+0.020659632, +0.088706910] | 119/56/30 | 16 | 15 | 317 | +0.040496/+0.024486/+0.013970 | +0.040496 | 1739.33 |
| P2/30002 | +0.046564966 | +0.008946459 | [+0.024946088, +0.070761345] | 130/36/39 | 21 | 15 | 343 | +0.040863/+0.027468/+0.017729 | +0.040863 | 1709.12 |
| P2/30003 | +0.062366015 | +0.009805937 | [+0.034174261, +0.096479530] | 133/35/37 | 24 | 14 | 389 | +0.051980/+0.033802/+0.023234 | +0.051980 | 1722.23 |

Median switched-set seed-Jaccard BASE **0.451380** /
REG **0.502381**;36 within-fold root-pair
comparisons per arm, empty/empty=1. Action Jaccard is separately labelled.
Exact config agreement **0.277778**;
parameter agreement `{'tau': 0.4166666666666667, 'k': 0.5555555555555556, 'a': 0.9722222222222222}`.

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
REG mean train/inner/gap:
`+0.166615014` /
`+0.084333400` /
`+0.082281615`.
Paired REG-BASE gap change: mean `-0.006726517`,
median `-0.010894643` across108 matching members.
A narrower gap alone does not qualify an arm for advancement.
[Full216-member diagnostic](overfit_summary.json) and [measurement code](overfit.py)
retain both components, per-cell summaries and source model hashes.
All216 gaps were independently reconstructed from saved TRAIN picks and raw
utility pools; no model rescoring was needed for numerical review.
[metrics.json](metrics.json),41,038bytes, contains
all nine cell outcomes, paired/per-root statistics and compact gap summaries.

## Verification, resources and custody

31 focused tests passed (10 fitting/boundary, 21 analysis/gap). Full independent
parameter/input/config audit and [numerical recalculation](numerical_review.json)
passed. Same-agent review is identified honestly. Actual fixture fits differ from
BASE at exactly the three required parameters, preserve rounds/seed behavior,
and reach every ranker route. The first implementation publication failed the
local whitespace gate before push; that metadata-only failure is retained. No
experimental fit restart, deadline extension or partial scoring.

xbabe2 CPU, nice 19, numeric threads 1, cores 4/5/6/8/33/41, disjoint from live peer affinities
and SMT siblings at launch. Actual MemoryMax 6 GiB/Swap 0, aggregate RSS 6 GiB cap,
host CPU 85%, load1 <= 85, MemAvailable 20 GiB floor, own scratch 2 GB cap.
Peak sampled RSS through analysis **3,922,026,496bytes**,
hostCPU **26.4%**; no guard violations.
Original compute 18:45Z/close 19:00Z retained. [Resource receipt](resource_receipt.json),
[dependencies](dependency_pins.json). Full repo/hosted CI NOT_RUN under owner freeze.

36 create-only model archives exact-version GET/SHA/member verified:
[archive index](model_archive_index.json). Final evidence/manifest/lifecycle use
`s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H327/astra-20261104T170100Z/`.
[Manifest](manifest.json), [S3 receipt](s3_manifest.json). Own scratch is removed
only after complete recovery mapping and no active consumer; foreign work is retained.
[TRAIN provenance](train_input_provenance.json) preserves allowed range/packet
identities. Original local H50/H58/H276 S3 versions remain unestablished; whole
mixed-cache feature files were not read. Full refitting additionally needs those
retained inputs; archived outputs support numerical replay. Published code is an
exact run artifact, not a standalone checkout CLI or a new replay authorization.

Cross-critique: H321 changed LambdaRank to regression_l2 alongside targets;
its negative result does not isolate normalization. H327 keeps LambdaRank.
[H321 source review](h321_crosscritique.json) pins that confound. H324 tested
spread weights rather than capacity; its negative result does not answer H327.
REG jointly changes three regularizers, so any effect cannot be attributed to
one of the three individually. TRAIN drop10/205 and dev10/19 trim different
fractions and change cohort/incumbent. This adaptive TRAIN wave does not resolve
all causes of the dev91 tail and establishes no universal impossibility.
H325's positive drop10 point estimate still failed its lower bound and harm rule;
its veto constructors also changed, unlike H327. H326 stopped on missing cached
policy coverage before selection or fitting; it is an unknown numerical outcome,
not a failed scientific arm. [Pinned peer review](peer_crosscritique.json).
H327 likewise preserves selected OOF vectors and aggregate grid means, not a
complete full-grid policy cache for future bootstrap reselection. Its archived
boosters do not replace missing inner selected-round/margin metadata.
Do not advance this fixed regularization recipe to dev91. Keep unchanged ENS_VOTE2 as the reference; this does not rule out every capacity or regularization change.
