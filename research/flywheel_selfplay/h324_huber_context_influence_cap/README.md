# H324 — Huber-style context influence cap for LambdaRank members

**CAP_NOT_ADVANCED — TRAIN-only research.** Config diagnostic: **CONFIG_SENSITIVE**.
BASE pooled gain **+0.048214800**, 95% CI **[0.025377654312495633, 0.07576665868860566]**;
CAP **+0.029392393**, 95% CI **[0.012450882550470829, 0.04877687307623925]**.
Paired drop-top-10 median **-0.010335373**, 95% CI **[-0.021411579613391924, -0.002506482365255465]**,
one-sided 95% LCB **-0.019761623**. Per-cell list, P0 roots 30001–30003,
then P1, then P2: `[-0.01311540040335365, -0.009697559169948043, -0.026714116248464266, -0.004252864555255785, -0.010335373333187698, -0.017769995841290093, -0.0012690048801422271, -0.0067222923870216954, -0.0206709635141319]`. Per-root medians across three partitions:
`[-0.004252864555255785, -0.009697559169948043, -0.0206709635141319]`. Criteria `{'seed_consistency': False, 'paired_drop10_margin': False, 'gain_retention_90percent': False, 'harmed_nonincrease': True}`; failed `['seed_consistency', 'paired_drop10_margin', 'gain_retention_90percent']`.
Gain retention **0.609614**; median harmed-family
rate BASE **0.092683** / CAP **0.078049**.

DOPE: `{"composite_sha256_prefix": "c51a56d0", "loaded": false, "scope": "CPU selection layer only"}`. DOPE weights were not loaded or
modified. Only permitted TRAIN ranges of existing cached feature vectors were
read. No neural optimizer update, serving promotion or dev91 shot is claimed.

## Hypothesis, formula and unchanged components

The hypothesis is that downweighting contexts with large raw utility spreads
could distribute gain beyond a few top families. This is a spread-derived query
weight, not a Huber loss or a proven cap on actual gradients. LambdaRank uses
ordinal relevance labels and label_gain; raw utility spread does not directly
scale its lambdas. Mean-one normalization can make final weights exceed one.

For each actual ranker fitting multiset C, count every context occurrence,
including bootstrap duplicates. Over its unchanged K=8 candidate slots compute
`spread_c=max(u_c)-min(u_c)`, `delta=median(spreads in C)`,
`raw_w_c=min(1,delta/max(spread_c,1e-9))`, then `w_c=raw_w_c/mean(raw_w in C)`.
Repeat each weight on all eight slots. Recompute on every inner/final/bag fit.
Pinned H287 topk.py line162 at 213b30ad52e1ce48927bba960ed609bd98ab4dfe uses
`np.where(gate,-fitness,-2.0)`: the failure sentinel is **-2.0**, unchanged.
Zero median spread would fail closed as INCOMPLETE; no alternate formula was used.

Only three ranker call sites and two ranker fitting boundaries receive weights:
inner/final `topk.fit_ranker` calls in `veto.run_cell_veto`, and
`bagged.fit_ranker_n` in `r302.fit_and_freeze_member`. Veto classifiers are
**unweighted**, preserving failure contexts. Original features, ordinal labels,
LambdaRank objective/label_gain behavior, original family IDs, seeds, parameter
recipe,18-config TAU/k/agreement grid, inner mean-family-gain objective and tie/
switch rules are unchanged. Selected hyperparameters may change as an effect.
Removing added sample-weight arguments restores identical original ASTs.

Each of 36 cells has819 weighted ranker fits plus195 unweighted gate fits.
Independent reconstruction verified **29,484**
weight vectors/context orders from actual utilities, folds and seeds.
All **108/108 final veto model hashes**
match BASE exactly. Final model hashes changed vsBASE:
**972/1080**.
[Weights](weights.py), [transformation](weight_transform_preview.json),
[full lock review](lock_review.json), [preflight](preflight_receipt.json).

## Preregistration and evaluation closure

[PREREG.md](PREREG.md) published at **fd5187dd8b4f80227ba112127c6aba180aac1417** before any fit.
Fitting source **5c34a248a2050feff0971103a870b690a19b9bda**, scoring/statistics source
**4f18c04cebfe1a3f79352a8a8616e7e96a8cddfa**, each before its execution. H322 template close
21010fdc5553a8d2791f8c8e260e7790c92715b8; byte-pinned H310/H303/H302/H287:
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
All 36 fsynced locks,1,080 model hashes and full weight/config/input audit precede
the separate once-only36 fold/root CAP scoring calls. [Lock receipt](lock_receipt.json),
[score receipt](score_receipt.json). Subsequent analysis uses saved outputs only.

## Metrics and frozen decision

1,219 TRAIN contexts / 205 families. Within-family context means have equal family
weight. Pooled gain is the median of 9 cell mean family gains; literal family
medians are separately reported. Cells share families and are dependent.
W/T/L tolerance 1e-4; harmed means gain < -.01; invalid/switch counts are contexts.
LOFO minimum equals drop-top-1, not independent corroboration.

10,000 paired whole-family draws RNG 324191 share sampled 205 family indices across
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
Median Elo BASE **1725.66** / CAP **1719.81**.

### BASE

| Partition/root | Mean gain | Literal median | 95% CI | W/T/L | Harmed | Invalid | Switches | Drop1/5/10 | LOFO min | Elo |
|---|---:|---:|---|---|---:|---:|---:|---|---:|---:|
| P0/30001 | +0.041289223 | +0.000000000 | [+0.021633868, +0.064175145] | 88/93/24 | 13 | 13 | 219 | +0.034482/+0.022905/+0.013174 | +0.034482 | 1725.66 |
| P0/30002 | +0.064316333 | +0.001588084 | [+0.015951163, +0.132443677] | 108/67/30 | 18 | 14 | 300 | +0.037100/+0.018970/+0.008633 | +0.037100 | 1722.48 |
| P0/30003 | +0.072257657 | +0.006984224 | [+0.023634943, +0.141739073] | 122/50/33 | 19 | 11 | 320 | +0.045081/+0.027514/+0.016621 | +0.045081 | 1727.10 |
| P1/30001 | +0.048214800 | +0.002999623 | [+0.018527168, +0.091111412] | 112/57/36 | 23 | 17 | 327 | +0.030657/+0.018311/+0.010198 | +0.030657 | 1697.13 |
| P1/30002 | +0.048657946 | +0.003624258 | [+0.021850002, +0.084406806] | 116/51/38 | 21 | 15 | 275 | +0.035309/+0.023340/+0.014063 | +0.035309 | 1693.84 |
| P1/30003 | +0.038294238 | +0.003253142 | [+0.015931933, +0.061131788] | 112/64/29 | 17 | 16 | 285 | +0.033333/+0.020954/+0.012702 | +0.033333 | 1734.68 |
| P2/30001 | +0.043601426 | +0.006326451 | [+0.017599571, +0.072008870] | 122/48/35 | 22 | 16 | 315 | +0.037280/+0.021223/+0.010678 | +0.037280 | 1716.88 |
| P2/30002 | +0.031839978 | +0.003624258 | [+0.010135448, +0.053846533] | 112/65/28 | 16 | 17 | 292 | +0.026502/+0.014887/+0.008774 | +0.026502 | 1740.78 |
| P2/30003 | +0.054720630 | +0.008144684 | [+0.033273396, +0.079169698] | 129/41/35 | 22 | 15 | 335 | +0.048595/+0.033993/+0.023605 | +0.048595 | 1726.57 |

### CAP

| Partition/root | Mean gain | Literal median | 95% CI | W/T/L | Harmed | Invalid | Switches | Drop1/5/10 | LOFO min | Elo |
|---|---:|---:|---|---|---:|---:|---:|---|---:|---:|
| P0/30001 | +0.051536511 | +0.000000000 | [+0.004713199, +0.118857003] | 93/88/24 | 14 | 13 | 222 | +0.024258/+0.008404/+0.000058 | +0.024258 | 1735.25 |
| P0/30002 | +0.042534019 | +0.000000000 | [-0.001059948, +0.108897931] | 78/105/22 | 16 | 16 | 172 | +0.015212/+0.005864/-0.001065 | +0.015212 | 1719.81 |
| P0/30003 | +0.021894007 | +0.000000000 | [-0.009394046, +0.055308743] | 84/90/31 | 16 | 13 | 244 | +0.012265/-0.002604/-0.010094 | +0.012265 | 1673.13 |
| P1/30001 | +0.054166200 | +0.000000002 | [+0.015853400, +0.110539756] | 102/80/23 | 17 | 16 | 299 | +0.031615/+0.015867/+0.005945 | +0.031615 | 1758.69 |
| P1/30002 | +0.021903021 | +0.000000000 | [+0.009231806, +0.039885822] | 42/156/7 | 2 | 15 | 68 | +0.015550/+0.007765/+0.003728 | +0.015550 | 1811.05 |
| P1/30003 | +0.024265435 | +0.000000000 | [-0.005288474, +0.053226452] | 88/89/28 | 20 | 16 | 192 | +0.017375/+0.001621/-0.005068 | +0.017375 | 1698.89 |
| P2/30001 | +0.029504624 | +0.000000000 | [+0.014651046, +0.048195869] | 79/102/24 | 15 | 15 | 194 | +0.023382/+0.013582/+0.009409 | +0.023382 | 1706.92 |
| P2/30002 | +0.024959532 | +0.000000000 | [+0.005887883, +0.047261448] | 80/102/23 | 17 | 15 | 222 | +0.018571/+0.007004/+0.002051 | +0.018571 | 1716.49 |
| P2/30003 | +0.029392393 | +0.000000000 | [+0.010068857, +0.050996993] | 78/106/21 | 15 | 14 | 191 | +0.023856/+0.012017/+0.002934 | +0.023856 | 1727.89 |

Median switched-set seed-Jaccard BASE **0.451380** /
CAP **0.291003**;36 within-fold root-pair
comparisons per arm, empty/empty=1. Action Jaccard is separately labelled.
Exact config agreement **0.222222**;
parameter agreement `{'tau': 0.4444444444444444, 'k': 0.5833333333333334, 'a': 1.0}`.

## Cap diagnostics

Across **17,766,035 actual fitting context occurrences**,
**8,870,141** were capped before normalization:
fraction **0.499275218**. These count repeated contexts across
fits and bootstrap draws, not unique outer contexts. Normalized weight quantiles
at `[0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1]` are `[0.015444861753287647, 0.05102920960337203, 0.12888633799869653, 0.3968982301183672, 1.4217363780792276, 1.4718229869039263, 1.500836967369524, 1.5198347002497858, 1.5599648139831799]`.
[Weight summary](weight_summary.json) includes all 36 cells and per-fit delta
summaries; each archived weight audit contains every fit's fractions/quantiles.
[metrics.json](metrics.json),37,788bytes, contains
all 9 cell outcomes, paired/per-root statistics and aggregate cap diagnostics.

## Verification, resources and custody

35 focused tests passed (17 fitting/boundary, 18 analysis). Full independent
weight/input/config audit and [numerical recalculation](numerical_review.json)
passed. Same-agent review is identified honestly. Uniform weights preserve exact
ranker predictions/parameters; nonuniform weights change actual ranker fits.
The first launcher refused on mpstat I/O-wait/idle checks before any experimental
fit. Its receipt is retained; a fresh assigned-core/SMT check passed before the
only experimental run. No restart, deadline extension or partial scoring.

xbabe2 CPU, nice 19, numeric threads 1, cores 4/5/6/8/33/41, disjoint from live peer affinities
and SMT siblings at launch. Actual MemoryMax 6 GiB/Swap 0, aggregate RSS 6 GiB cap,
host CPU 85%, load1 <= 85, MemAvailable 20 GiB floor, own scratch 2 GB cap.
Peak sampled RSS through analysis **3,861,438,464bytes**,
hostCPU **48.0%**; no guard violations.
Original compute 17:30Z/close 17:45Z retained. [Resource receipt](resource_receipt.json),
[dependencies](dependency_pins.json). Full repo/hosted CI NOT_RUN under owner freeze.

36 create-only model archives exact-version GET/SHA/member verified:
[archive index](model_archive_index.json). Final evidence/manifest/lifecycle use
`s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H324/astra-20261104T152528Z/`.
[Manifest](manifest.json), [S3 receipt](s3_manifest.json). Own scratch is removed
only after complete recovery mapping and no active consumer; foreign work is retained.
[TRAIN provenance](train_input_provenance.json) preserves allowed range/packet
identities. Original local H50/H58/H276 S3 versions remain unestablished; whole
mixed-cache feature files were not read. Full refitting additionally needs those
retained inputs; archived outputs support numerical replay. Published code is an
exact run artifact, not a standalone checkout CLI or a new replay authorization.

Cross-critique: H321 changed LambdaRank to regression_l2 as well as targets, so
its negative result does not isolate normalization. H324 keeps that loss fixed.
[H321 source review](h321_crosscritique.json) pins run_h321.py:25-35 and the
peer report at 0fb198a5fc0d3dff9de8f95a65a64a3b7c6b59d0.
H322 weighted rankers and vetoes by family counts; H324 changes the formula AND
keeps vetoes unweighted, so H324-versus-H322 is not a pure formula contrast.
H320's negative trimmed-objective result does not identify the sole cause of the
dev91 tail. TRAIN drop10/205 and dev10/19 also trim different proportions and
change cohort/incumbent. No broad impossibility claim follows from these recipes.
Do not advance this spread-weighted member recipe to dev91. Keep unchanged ENS_VOTE2 as the reference; this result does not rule out every weighting or robust-loss method.
