# H337 — Soft bag-vote smoothing

**SOFTVOTE_NOT_ADVANCED — TRAIN-only research.** BASE pooled gain **+0.048214800**,
95% CI **[0.025327583850400348, 0.07647280349517184]**; SOFTVOTE **+0.036039095**, CI **[0.01862905653127401, 0.05818736913445139]**.
Paired arm-specific drop-top-10 median **-0.005201483**, CI **[-0.013826899320304531, 1.7762730234743134e-05]**,
one-sided95LCB **-0.012482736**. Per-root medians **[-3.93797993547048e-05, -0.005201483149918621, -0.01211703390650808]**.
Criteria `{'seed_consistency': False, 'paired_drop10_margin': False, 'gain_retention_90percent': False, 'harmed_nonincrease': True}`; failed `['seed_consistency', 'paired_drop10_margin', 'gain_retention_90percent']`.
Gain retention **0.747470**; median harmed-family rate
BASE **0.092683** / SOFTVOTE **0.048780**.

DOPE `{"composite_sha256_prefix": "c51a56d0", "loaded": false, "scope": "CPU selection layer only"}`. DOPE weights were not loaded or
modified. No neural optimizer update, live-loop change, serving promotion or
dev91 result is claimed.

## Hypothesis and exact treatment

H336 established that BASE counts individual hard bag-margin votes, rather than
averaging bag scores. H337 tests whether smoothing that step improves family
robustness. It does not measure temporal score noise or explain a live-round cycle.

At pinned H287 commit213b30ad52e1ce48927bba960ed609bd98ab4dfe,
`bagged.vote_counts` (bagged.py:59-72), replace one comparison expression with
`sigmoid((score[j]-score[0]-strict_margin)/T_b)`. The inverse restores the exact
original AST. The original operation is a comparison, not a keyword; this is
one comparison-node replacement. No other voting-function AST node changes.
[Treatment](softvote.py), [actual transformation](vote_transform.json).

BASE's int32 array, `int(sum(...))`, and `int(v)>=integer k` remain unchanged.
For these nonnegative sums and k5/6/7, flooring before comparison is exactly
equivalent to comparing the soft sum directly with k. The tests cover threshold
boundaries; raw unfloored sums are saved and independently checked. Zero centered
margin contributes.5, while the original strict hard vote contributed0. Infinite
strict margins preserve the original early zero-vote return; slot0 remains the
fallback. Stable expit avoids overflow; no clipping, epsilon or temperature tuning.

All **1080 final main/veto/bag model hashes** match archived BASE. There are zero
new model fits. Main predictions, veto probabilities, selected rounds/margins,
TAU, k and three-member agreement a are unchanged. The selected BASE configuration
is reused without reselection: config agreement **1.0** / **CONFIG_INSENSITIVE**
is fixed by design. Each member's temperature is explicitly bound around the
existing frozen-scorer and member-pick calls. No proposer or veto function changes.
[Scoring implementation](score.py), [route controls](test_score_route.py).

## Temperature calibration and input closure

864 temperatures =36cells x3members x8bags. For each cell, use the archived H318
inner models at its BASE-selected TAU. A given bag index uses its corresponding
four inner models, each scoring only its H303 inner-heldout contexts within the
outer TRAIN partition. Those cross-fit counterparts are distinct from the final
frozen bag to which the temperature applies. This is not final-model resubstitution
calibration. Selected TAU was chosen using BASE inner validation; these are not
independent, untouched calibration data.

The population is all seven nonbaseline candidates j1..7 per outer-TRAIN context,
once through its inner-heldout prediction. Compute raw candidate-minus-slot0
deltas, then T=.5*population std (float64, ddof0), candidate-equal without utility,
family, bootstrap or veto weights. Subtracting a finite constant strict margin
does not change this std. For infinite strict margins, compute the same finite
delta std for audit and retain BASE's early rejection; the temperature is unused.
Nonpositive/nonfinite temperatures fail closed; none were substituted or omitted.
Observed temperature range **[0.09547444745788103, 0.34917632955726086]**. [All cell temperatures
and identities](temperatures.json), [calibration code](calibrate.py).

All36 inner archives were fetched by exact S3 version, whole-object SHA and every
member SHA verified. 3456 selected-TAU inner bag models supplied the predictions;
their files were reclaimed after verified recovery and use. [Archived model index](base_model_archive_index.json)
and [pinned source](inner_archive_source.json). The calibration input packet hashes
match H318's archived original packet identities. Physical pool opens and cache
byte ranges are restricted to each cell's outer-TRAIN contexts. No outer-test
features or historical outer outcome arrays enter temperature calibration.

## Preregistration, regression and once-only scoring

[PREREG](PREREG.md) published **8402642eeecf61ce6d774561f09bb131abfa2091** before any fixture or model score.
Implementation **ec610a1a72375957dcd41df107338452026f66b5** before calibration and treatment scoring.
[Vendor pins](vendor_pins.json) retain H303/H302/H308/H287 source identities.

The exact-version H318 BASE archive was recovered read-only: [input receipt](input_receipt.json),
[archive pin](base_archive_pin.json). Fresh unchanged BASE replay matched all36
fold/root pick hashes,9pick/raw-utility panels and3medians exactly:
.0643163334285782, .04821480030677507, .04360142642936269.
[Regression receipt](regression_receipt.json). No BASE refit.
This mandatory separate BASE verification reads TRAIN outer rows. The zero
prelock-outer-read assertion applies to calibration/treatment workers, not to
that required historical-reference verification.

All36 fsynced score locks pin temperatures, source, BASE choices and1080model
hashes before any new outer score. Independent pre-score review recalculated
all864 temperatures and checked3456inner-model identities and1080final hashes.
[Lock audit](lock_review.json), [all-lock receipt](lock_receipt.json).
Exactly36 once-only fold/root treatment model-scoring calls form9panels.
[Score receipt](score_receipt.json). Saved raw main/veto/bag outputs allow an
independent reconstruction of every member selection and all hard-BASE and
SOFTVOTE picks, without another model score. [Numerical review](numerical_review.json).

## Metrics and frozen decision

1219TRAIN contexts /205families; context means within families, equal family
weight. Pooled gain is median of9cell mean family gains; literal family medians
are also reported. W/T/L tolerance1e-4; harmed means gain<-.01; invalid/switch
counts are contexts. Drop the largest1/5/10gains separately in each arm.
LOFO minimum equals drop-top1 and is not independent evidence.

10000 paired whole-family draws RNG337191 share family indices across arms and
nine dependent cells. Decision statistic is median9 arm-specific drop10
differences; CI.025/.975 and one-sided95LCB.05. Advance requires median>=+.002,
LCB>0, median gain>=.9BASE, median harmed<=BASE, and strictly positive per-root
medians on at least2/3roots. Exact+.002 qualifies; zero LCB/overall ties fail;
zero root medians are not positive. No post-outcome changes.

**reference = r882 slot0 (DL-Train outer); r3175 not measured on TRAIN**.
Elo uses the inherited400-step Bradley-Terry MM recipe with pseudocount.01,
ties omitted. Median Elo BASE **1725.66**, SOFTVOTE **1751.29**.

### BASE

| Partition/root | Mean gain | Literal median | 95% CI | W/T/L | Harmed | Invalid | Switches | Drop1/5/10 | LOFO min | Elo |
|---|---:|---:|---|---|---:|---:|---:|---|---:|---:|
| P0/30001 | +0.041289223 | +0.000000000 | [+0.021178671, +0.064525929] | 88/93/24 | 13 | 13 | 219 | +0.034482/+0.022905/+0.013174 | +0.034482 | 1725.66 |
| P0/30002 | +0.064316333 | +0.001588084 | [+0.015655698, +0.137178745] | 108/67/30 | 18 | 14 | 300 | +0.037100/+0.018970/+0.008633 | +0.037100 | 1722.48 |
| P0/30003 | +0.072257657 | +0.006984224 | [+0.022770521, +0.145109194] | 122/50/33 | 19 | 11 | 320 | +0.045081/+0.027514/+0.016621 | +0.045081 | 1727.10 |
| P1/30001 | +0.048214800 | +0.002999623 | [+0.018545169, +0.094402458] | 112/57/36 | 23 | 17 | 327 | +0.030657/+0.018311/+0.010198 | +0.030657 | 1697.13 |
| P1/30002 | +0.048657946 | +0.003624258 | [+0.022114075, +0.086230714] | 116/51/38 | 21 | 15 | 275 | +0.035309/+0.023340/+0.014063 | +0.035309 | 1693.84 |
| P1/30003 | +0.038294238 | +0.003253142 | [+0.015829688, +0.060871697] | 112/64/29 | 17 | 16 | 285 | +0.033333/+0.020954/+0.012702 | +0.033333 | 1734.68 |
| P2/30001 | +0.043601426 | +0.006326451 | [+0.017851920, +0.071897702] | 122/48/35 | 22 | 16 | 315 | +0.037280/+0.021223/+0.010678 | +0.037280 | 1716.88 |
| P2/30002 | +0.031839978 | +0.003624258 | [+0.010144186, +0.054050255] | 112/65/28 | 16 | 17 | 292 | +0.026502/+0.014887/+0.008774 | +0.026502 | 1740.78 |
| P2/30003 | +0.054720630 | +0.008144684 | [+0.033542664, +0.079733109] | 129/41/35 | 22 | 15 | 335 | +0.048595/+0.033993/+0.023605 | +0.048595 | 1726.57 |

### SOFTVOTE

| Partition/root | Mean gain | Literal median | 95% CI | W/T/L | Harmed | Invalid | Switches | Drop1/5/10 | LOFO min | Elo |
|---|---:|---:|---|---|---:|---:|---:|---:|---:|---:|
| P0/30001 | +0.031197835 | +0.000000000 | [+0.015557162, +0.051138878] | 62/131/12 | 5 | 16 | 119 | +0.024342/+0.014255/+0.009356 | +0.024342 | 1785.17 |
| P0/30002 | +0.047421848 | +0.000000000 | [+0.015736157, +0.093556976] | 74/105/26 | 15 | 15 | 201 | +0.029859/+0.015275/+0.006718 | +0.029859 | 1681.66 |
| P0/30003 | +0.043006912 | +0.000000000 | [+0.002577150, +0.100844304] | 79/110/16 | 9 | 14 | 159 | +0.022079/+0.005045/+0.000034 | +0.022079 | 1777.32 |
| P1/30001 | +0.049137794 | +0.000000000 | [+0.020231751, +0.094894289] | 101/81/23 | 15 | 17 | 220 | +0.031585/+0.019257/+0.011893 | +0.031585 | 1756.98 |
| P1/30002 | +0.036039095 | +0.000000000 | [+0.012329730, +0.072097900] | 71/118/16 | 8 | 14 | 131 | +0.022629/+0.010542/+0.005726 | +0.022629 | 1758.77 |
| P1/30003 | +0.023514782 | +0.000000000 | [+0.010864401, +0.038963203] | 64/123/18 | 9 | 15 | 112 | +0.018978/+0.009819/+0.005505 | +0.018978 | 1720.29 |
| P2/30001 | +0.035042327 | +0.000000000 | [+0.018775660, +0.054483800] | 82/102/21 | 10 | 15 | 149 | +0.030401/+0.018008/+0.010639 | +0.030401 | 1736.58 |
| P2/30002 | +0.021993746 | +0.000000000 | [+0.002654224, +0.040965498] | 85/100/20 | 12 | 17 | 193 | +0.016607/+0.008362/+0.003572 | +0.016607 | 1751.29 |
| P2/30003 | +0.036739425 | +0.000000000 | [+0.018863040, +0.059396290] | 96/81/28 | 14 | 17 | 214 | +0.030526/+0.017718/+0.011488 | +0.030526 | 1714.00 |

Per-cell paired drop10 (P0/P1/P2, each roots30001-30003): `[-0.003817776944364616, -0.0019148165338236456, -0.016586289046257054, 0.0016955631643160631, -0.008337282265097928, -0.007197302071625461, -3.93797993547048e-05, -0.005201483149918621, -0.01211703390650808]`.
Median switched-set seed-Jaccard BASE **0.451380**,
SOFTVOTE **0.331689**. Empty/empty=1;
action-Jaccard is separately labelled. [Full metrics](metrics.json) retain all
cell and aggregate values in 37,510bytes.

## Verification, resources and custody

38 tests passed:12 soft-vote,4physical-boundary,18analysis and4route controls.
[Main receipt](tests_receipt.json), [route receipt](route_tests_receipt.json).
Full temperature/model/input audit and independently implemented numerical and
selection recalculation passed. Review is by the implementing agent, not an
independent reviewer. No experimental restart, partial scoring or deadline reset.

CPUxbabe2, core4 only, nice19/numeric1; assigned-core+SMT idle probe retained.
MemoryMax6GiB/Swap0, RSS6GiB cap, hostCPU/load<=85, MemAvailable>=20GiB,
own scratch<=2GB. Peak sampled RSS through analysis **836,812,800bytes**,
hostCPU **50.5%**; guard violations none.
Fixed00:15Nov5compute/00:30close retained. [Resources](resource_receipt.json),
[dependencies](dependency_pins.json). Full repository/hosted CI NOT_RUN by freeze.

Create-only evidence prefix:
`s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H337/astra-20261104T215600Z/`.
[Manifest](manifest.json), [S3 receipt](s3_manifest.json). Exact-version GET,
whole-object and every-member hashes are verified before own scratch cleanup.
Foreign work is preserved. Original local H50/H58/H276 input S3 versions remain
unestablished; allowed cache ranges and packet hashes are retained. Full refitting
requires those retained inputs; archived outputs support numerical replay.
Published code records this run, not a new authorization to score or a standalone CLI.

## Interpretation and next step

These are reused TRAIN families and dependent partition/seed cells. Intervals
condition on fixed models and temperatures, do not repeat selection, and do not
adjust for the wider adaptive wave search. No dev91 or serving inference follows.
H336's source mismatch is corrected here at the actual hard-vote comparison;
H331's cosine result tested a different factor. Neither result proves that score
noise causes live-cycle behavior. Reduced harm alone cannot override a failed
robustness or retention clause. Main/veto predictions and model bytes are fixed,
so any measured policy effect belongs to this exact soft-vote rule and calibrated
temperature population, not improved model training.

Do not advance this fixed SOFTVOTE policy to dev91.
