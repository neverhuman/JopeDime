# H330 — L2-only ablation for LambdaRank members

**L2ONLY_NOT_ADVANCED — TRAIN-only research.** Config diagnostic: **CONFIG_SENSITIVE**.
BASE pooled gain **+0.048214800**, 95% CI **[0.025204429590732654, 0.07566722258222101]**;
L2ONLY **+0.045975110**, 95% CI **[0.02707101554677697, 0.08693535542859918]**.
Paired drop-top-10 median **-0.003440034**, 95% CI **[-0.006819567710963014, 0.007894053418070897]**,
one-sided 95% LCB **-0.005548960**. Per-cell list, P0 roots 30001–30003,
then P1, then P2: `[0.0015567534447100805, -0.006724735837136683, -0.003440034473457549, 0.005169185267955255, -0.009215836881245637, -0.003495384986671275, 0.011298073782342818, 0.014264336253647168, -0.007951090834636059]`. Per-root medians across three partitions:
`[0.005169185267955255, -0.006724735837136683, -0.003495384986671275]`. Criteria `{'seed_consistency': False, 'paired_drop10_margin': False, 'gain_retention_90percent': True, 'harmed_nonincrease': False}`; failed `['seed_consistency', 'paired_drop10_margin', 'harmed_nonincrease']`.
Gain retention **0.953548**; median harmed-family
rate BASE **0.092683** / L2ONLY **0.112195**.

DOPE: `{"composite_sha256_prefix": "c51a56d0", "loaded": false, "scope": "CPU selection layer only"}`. DOPE weights were not loaded or
modified. Only permitted TRAIN ranges of existing cached feature vectors were
read. No neural optimizer update, serving promotion or dev91 shot is claimed.

## Hypothesis, exact recipe and unchanged components

The hypothesis is that the L2 leaf penalty carries H327's gain, while its stronger feature subsampling and minimum leaf size caused
additional harmed families. L2ONLY changes exactly one ranker constructor value:
reg_lambda (lambda_l2) 1.0 -> 10.0. colsample_bytree (feature_fraction) stays .8;
min_child_samples (min_data_in_leaf) stays 20. This is LightGBM L2 leaf-value
regularization; no neural optimization or activation dropout occurs.
No tuning or regularization grid, and no sample weights.

Only topk.fit_ranker and bagged.fit_ranker_n constructors change; inner, final and
all eight bag rankers use them. Restoring the single changed BASE value gives identical
ASTs. H287 commit213b30ad52e1ce48927bba960ed609bd98ab4dfe is pinned.
LambdaRank objective, ordinal labels/default label_gain, max_depth3,num_leaves7,
row subsampling, seed derivation, feature packets, original family folds,
learning-rate/round/margin selection procedure and 18-config TAU/k/agreement grid,
inner objective, ties and switches stay unchanged. Selected values may differ
as a treatment effect. Veto classifiers retain BASE code and parameters,
including their existing reg_lambda2.0; no L2ONLY overrides or weights reach gates.
Failure utility remains np.where(gate,-fitness,-2.0), sentinel **-2.0**.

Each of 36 cells has819 ranker fits plus195 unchanged gate fits. Full audit verified
**29,484 ranker fit records**, actual fitting multiset
sizes, the one changed L2 value and two retained feature/leaf values in972 frozen final rankers, and all input identities.
All **108/108 final gate hashes** match
BASE exactly. Changed final model hashes: **972/1080**.
[Regularization code](regularization.py), [AST transforms](regularization_transform_preview.json),
[lock audit](lock_review.json), [preflight](preflight_receipt.json).

## Preregistration and evaluation closure

[PREREG.md](PREREG.md) published at **a6f847048a95fa47127d73b19a0eea989075d79e** before any fit.
Fitting source **5f58de6fb4ea37fb4dd6c72a21902f9348c8c745**, scoring/statistics source
**43c3ad2a65c3129bf039d5f354a89fa164a294ce**, each before its execution. H327 template close
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
the separate once-only36 fold/root L2ONLY scoring calls. [Lock receipt](lock_receipt.json),
[score receipt](score_receipt.json). Subsequent analysis uses saved outputs only.

## Metrics and frozen decision

1,219 TRAIN contexts / 205 families. Within-family context means have equal family
weight. Pooled gain is the median of 9 cell mean family gains; literal family
medians are separately reported. Cells share families and are dependent.
W/T/L tolerance 1e-4; harmed means gain < -.01; invalid/switch counts are contexts.
LOFO minimum equals drop-top-1, not independent corroboration.

10,000 paired whole-family draws RNG 330191 share sampled 205 family indices across
arms and cells. Sort/trim each arm separately, then take the median 9 paired
drop-top-10 differences. CI uses .025/.975, one-sided 95% LCB .05. These intervals
condition on fitted policies and do not repeat fitting or model selection.
This is an adaptive research wave on reused TRAIN families, not an independent
confirmatory experiment; intervals do not adjust for the wider wave search.
Advance requires paired median >= +.002 and LCB > 0, full gain >= 90% BASE, no median
harm increase, and all three per-root medians over 3 partitions >= 0.
The explicit operators govern: exact +.002 satisfies the margin, and a zero root
median satisfies the nonnegative root floor. Overall zero differences and zero
LCBs fail; any negative root fails, even if the other two roots are positive. CONFIG_INSENSITIVE is a separate diagnostic
when >= 30 of 36 (TAU,k,a) choices match BASE; it never relaxes the decision.

**reference = r882 slot0 (DL-Train outer); r3175 not measured on TRAIN**.
Elo retains the 400-iteration Bradley–Terry MM recipe, pseudocount .01, ties omitted.
Median Elo BASE **1725.66** / L2ONLY **1716.12**.

### BASE

| Partition/root | Mean gain | Literal median | 95% CI | W/T/L | Harmed | Invalid | Switches | Drop1/5/10 | LOFO min | Elo |
|---|---:|---:|---|---|---:|---:|---:|---|---:|---:|
| P0/30001 | +0.041289223 | +0.000000000 | [+0.021610003, +0.064501833] | 88/93/24 | 13 | 13 | 219 | +0.034482/+0.022905/+0.013174 | +0.034482 | 1725.66 |
| P0/30002 | +0.064316333 | +0.001588084 | [+0.015726935, +0.131619047] | 108/67/30 | 18 | 14 | 300 | +0.037100/+0.018970/+0.008633 | +0.037100 | 1722.48 |
| P0/30003 | +0.072257657 | +0.006984224 | [+0.023863196, +0.139240962] | 122/50/33 | 19 | 11 | 320 | +0.045081/+0.027514/+0.016621 | +0.045081 | 1727.10 |
| P1/30001 | +0.048214800 | +0.002999623 | [+0.018294223, +0.091830911] | 112/57/36 | 23 | 17 | 327 | +0.030657/+0.018311/+0.010198 | +0.030657 | 1697.13 |
| P1/30002 | +0.048657946 | +0.003624258 | [+0.022010399, +0.083801676] | 116/51/38 | 21 | 15 | 275 | +0.035309/+0.023340/+0.014063 | +0.035309 | 1693.84 |
| P1/30003 | +0.038294238 | +0.003253142 | [+0.016371561, +0.060805010] | 112/64/29 | 17 | 16 | 285 | +0.033333/+0.020954/+0.012702 | +0.033333 | 1734.68 |
| P2/30001 | +0.043601426 | +0.006326451 | [+0.017139143, +0.072288832] | 122/48/35 | 22 | 16 | 315 | +0.037280/+0.021223/+0.010678 | +0.037280 | 1716.88 |
| P2/30002 | +0.031839978 | +0.003624258 | [+0.010053309, +0.053933736] | 112/65/28 | 16 | 17 | 292 | +0.026502/+0.014887/+0.008774 | +0.026502 | 1740.78 |
| P2/30003 | +0.054720630 | +0.008144684 | [+0.033011974, +0.079165222] | 129/41/35 | 22 | 15 | 335 | +0.048595/+0.033993/+0.023605 | +0.048595 | 1726.57 |

### L2ONLY

| Partition/root | Mean gain | Literal median | 95% CI | W/T/L | Harmed | Invalid | Switches | Drop1/5/10 | LOFO min | Elo |
|---|---:|---:|---|---|---:|---:|---:|---|---:|---:|
| P0/30001 | +0.059104780 | +0.005460939 | [+0.025822734, +0.102070490] | 118/53/34 | 24 | 13 | 309 | +0.042252/+0.025437/+0.014730 | +0.042252 | 1716.12 |
| P0/30002 | +0.045070794 | +0.005026830 | [+0.002821683, +0.097597336] | 117/55/33 | 22 | 14 | 323 | +0.024153/+0.011169/+0.001908 | +0.024153 | 1719.83 |
| P0/30003 | +0.061524838 | +0.008332347 | [+0.022282593, +0.113617348] | 126/38/41 | 26 | 16 | 389 | +0.040688/+0.023896/+0.013181 | +0.040688 | 1695.01 |
| P1/30001 | +0.045975110 | +0.004170638 | [+0.025874197, +0.068703711] | 115/54/36 | 23 | 14 | 306 | +0.040222/+0.026793/+0.015367 | +0.040222 | 1701.72 |
| P1/30002 | +0.040325602 | +0.004219368 | [+0.010574611, +0.073167044] | 118/61/26 | 18 | 17 | 268 | +0.029122/+0.016285/+0.004848 | +0.029122 | 1762.71 |
| P1/30003 | +0.042240727 | +0.004013544 | [+0.012413350, +0.070943074] | 120/52/33 | 22 | 14 | 341 | +0.037425/+0.021591/+0.009207 | +0.037425 | 1724.23 |
| P2/30001 | +0.071276900 | +0.010809190 | [+0.032317866, +0.126542195] | 134/31/40 | 25 | 13 | 412 | +0.048809/+0.033537/+0.021977 | +0.048809 | 1709.99 |
| P2/30002 | +0.058933206 | +0.010911458 | [+0.032745157, +0.090794171] | 131/34/40 | 23 | 13 | 378 | +0.047807/+0.033499/+0.023038 | +0.047807 | 1706.05 |
| P2/30003 | +0.043810544 | +0.006299274 | [+0.024503028, +0.066357540] | 119/61/25 | 14 | 15 | 264 | +0.038256/+0.023368/+0.015654 | +0.038256 | 1770.99 |

Median switched-set seed-Jaccard BASE **0.451380** /
L2ONLY **0.510167**;36 within-fold root-pair
comparisons per arm, empty/empty=1. Action Jaccard is separately labelled.
Exact config agreement **0.166667**;
parameter agreement `{'tau': 0.3888888888888889, 'k': 0.5277777777777778, 'a': 1.0}`.

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
L2ONLY mean train/inner/gap:
`+0.171043571` /
`+0.085847282` /
`+0.085196288`.
Paired L2ONLY-BASE gap change: mean `-0.003811843`,
median `-0.009223314` across108 matching members.
A narrower gap alone does not qualify an arm for advancement.
[Full216-member diagnostic](overfit_summary.json) and [measurement code](overfit.py)
retain both components, per-cell summaries and source model hashes.
All216 gaps were independently reconstructed from saved TRAIN picks and raw
utility pools; no model rescoring was needed for numerical review.
[metrics.json](metrics.json),45,651bytes, contains
all nine cell outcomes, paired/per-root statistics and compact gap summaries.

## Verification, resources and custody

34 focused tests passed (10 fitting/boundary, 24 analysis/gap/comparison). Full
parameter/input/config audit and [numerical recalculation](numerical_review.json)
passed. This is same-agent review and independently implemented recalculation,
not independent reviewer approval. Actual fixture fits differ from BASE only in
reg_lambda, preserve rounds/seed behavior, and reach every ranker route.
No experimental fit restart, deadline extension or partial scoring.

xbabe2 CPU, nice 19, numeric threads 1, cores 4/5/6/8/33/41, disjoint from live peer affinities
and SMT siblings at launch. Actual MemoryMax 6 GiB/Swap 0, aggregate RSS 6 GiB cap,
host CPU 85%, load1 <= 85, MemAvailable 20 GiB floor, own scratch 2 GB cap.
Peak sampled RSS through analysis **3,921,424,384bytes**,
hostCPU **36.0%**; no guard violations.
Original compute 20:05Z/close 20:20Z retained. [Resource receipt](resource_receipt.json),
[dependencies](dependency_pins.json). Full repo/hosted CI NOT_RUN under owner freeze.

36 create-only model archives exact-version GET/SHA/member verified:
[archive index](model_archive_index.json). Final evidence/manifest/lifecycle use
`s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H330/astra-20261104T182300Z/`.
[Manifest](manifest.json), [S3 receipt](s3_manifest.json). Own scratch is removed
only after complete recovery mapping and no active consumer; foreign work is retained.
[TRAIN provenance](train_input_provenance.json) preserves allowed range/packet
identities. Original local H50/H58/H276 S3 versions remain unestablished; whole
mixed-cache feature files were not read. Full refitting additionally needs those
retained inputs; archived outputs support numerical replay. Published code is an
exact run artifact, not a standalone checkout CLI or a new replay authorization.

Cross-critique: H327 changed three ranker regularizers jointly, so its outcome
cannot identify L2 independently. This ablation isolates L2 versus BASE, while
comparison with H327 still changes two factors jointly. No causal allocation
between feature subsampling and minimum leaf size is supported. H330 also uses
an all-three-nonnegative root floor; H327 required two positive roots. Therefore
cross-wave verdicts use different root-consistency rules. TRAIN drop10/205 and
dev10/19 trim different fractions and change cohort/incumbent. Reused TRAIN
families and repeated wave selection limit generalization. No dev91 result or
universal impossibility follows. The selected OOF vectors and grid means are
retained, but this is not a complete full-grid policy cache for bootstrap reselection.

## H327 REG versus L2ONLY — descriptive only

Historical H327 metrics are exact bytes from close9b8e5076ed80359d7bc49890c64a32205fbadf10:
[pinned comparison source](h327_comparison_pin.json), [historical metrics](h327_metrics.json).
Identical BASE cell gains/identities were checked before constructing this table.
REG remains its recorded REG_NOT_ADVANCED. No additional decision or hypothesis
test compares REG against L2ONLY. Historical intervals use RNG327191 and current
intervals RNG330191. Each arm is compared with BASE under its own preregistration.

| Measure | H327 REG | H330 L2ONLY |
|---|---:|---:|
| Median family-equal gain | 0.0588452296777811 | 0.04597511001146662 |
| 95% gain CI | [0.03129394137821197, 0.09027847505066222] | [0.02707101554677697, 0.08693535542859918] |
| Median harmed rate | 0.1073170731707317 | 0.11219512195121951 |
| Median Elo | 1725.6702946436071 | 1716.1248713724335 |
| Paired vs BASE drop10 | 0.0032914361822550652 | -0.003440034473457549 |
| Paired vs BASE LCB95 | -0.001328323455151581 | -0.005548959897751528 |
| Per-root paired medians | [0.0032914361822550652, 0.008955221607742491, 0.0003530006696565221] | [0.005169185267955255, -0.006724735837136683, -0.003495384986671275] |
| Mean member train-minus-inner gap | 0.08228161463194852 | 0.08519628830775154 |

| Partition/root | REG gain | L2ONLY gain | REG harm | L2ONLY harm | REG drop10 | L2ONLY drop10 |
|---|---:|---:|---:|---:|---:|---:|
| P0/30001 | +0.058845230 | +0.059104780 | 0.107317 | 0.117073 | +0.007060525 | +0.014730310 |
| P0/30002 | +0.066161544 | +0.045070794 | 0.141463 | 0.107317 | +0.016203455 | +0.001908101 |
| P0/30003 | +0.062351487 | +0.061524838 | 0.107317 | 0.126829 | +0.018211944 | +0.013180508 |
| P1/30001 | +0.059427347 | +0.045975110 | 0.112195 | 0.112195 | +0.022722938 | +0.015366715 |
| P1/30002 | +0.057126015 | +0.040325602 | 0.053659 | 0.087805 | +0.025124393 | +0.004847515 |
| P1/30003 | +0.039076334 | +0.042240727 | 0.063415 | 0.107317 | +0.013054995 | +0.009206610 |
| P2/30001 | +0.051716030 | +0.071276900 | 0.078049 | 0.121951 | +0.013969911 | +0.021976548 |
| P2/30002 | +0.046564966 | +0.058933206 | 0.102439 | 0.112195 | +0.017728905 | +0.023038019 |
| P2/30003 | +0.062366015 | +0.043810544 | 0.117073 | 0.068293 | +0.023234427 | +0.015654190 |

## Red-team disposition

Leakage: fitting workers were barred from outer pools, archived BASE outcomes and
historical H327 metrics; all36 locks precede scoring. Seed noise: every root and
partition is reported, all3 root medians must be nonnegative, and dependent cells
are not represented as independent trials. Direction: higher utility/gain and
LCB are better; harmed-family rate must not rise. Cherry-picking: one fixed arm,
all9 cells, no post-outcome threshold or selection change; REG comparison is
explicitly descriptive. Ties: overall zero gain/LCB never advance; exact boundary
operators are tested and stated above. Protected inputs: no dev91/DL-Val/DL-Test,
sealed/holdout/meta-test/fresh-cohort reads or live writer action; DOPE remains
unloaded. Resources: sampled RSS/CPU/load/memory/scratch guards plus kernel6GiB/
swap0 enforced; final custody verifies exact versions and all members before
own scratch removal. Evidence does not imply a serving or neural-learning qualification.
Do not advance this fixed regularization recipe to dev91. Keep unchanged ENS_VOTE2 as the reference; this does not rule out every capacity or regularization change.
