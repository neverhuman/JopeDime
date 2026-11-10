# H312 — independent H310 small-cluster red-team

**H310_SMALLCLUSTER_ROBUST**. Fresh-three median family-equal gain is **+0.029545876**
against r3175 slot0, with shared-family wild-bootstrap basic 95% CI
**[0.01818757476296355, 0.047506592793833936]**. Across all six seeds, the median is
**+0.021310997**, wild CI **[0.013397789673523442, 0.03136148296061772]**.
Failed decision clauses: **none**.
This is shadow evidence only; even ROBUST needs a fresh S3 population via H299/H300
before any breakthrough claim. No serving or flywheel change.

## Hypothesis and frozen setup

H310's three-seed gain may be sensitive to only 19 development families,
concentration and seed choice. ASTRA independently reproduced H310, then added
three root seeds without changing its lock procedure. Preregistration
[`08c46e22a85e5fce3ecd39cf5c75ecf191d82fcd`](https://github.com/neverhuman/JopeDime/commit/08c46e22a85e5fce3ecd39cf5c75ecf191d82fcd)
was published before any H312 fit or dev91 read. [PREREG.md](PREREG.md) defines
all estimands, tie rules and five decision clauses. Fitting implementation
`5f8f2e30f9157c0128d4ca6ed686ab80e940d381` was published before fitting.
Scoring/inference code `3c8afa9732fa8040168d0051b1eea7fefc79fc87`
was published before the one-shot dev91 score.
H310 source is `819a45aa5e30e662ab511cd9962b85b2b8c1e44a`; all vendor hashes are in
[dependency_pins.json](dependency_pins.json).

TRAIN only: 1,219 contexts / 205 families; dev91: 91 contexts / 19 families.
New roots 30004–30006 use the original H300 member seed derivation, three members each.
Unchanged H303 18-config grid: tau .05/.10/.15, member vote 5/6/7,
ensemble agreement 2/3; four inner family folds, key **4**, conservative 1e-4 ties.
No revival of the historical ALL-key crash. Each fresh fit lock and SHA receipt
was fsynced before any fresh dev91 score. All 90 final model hashes verified.
[lock_receipt.json](lock_receipt.json) records zero pre-lock dev91 rows in each
worker; [lock_review.json](lock_review.json) precedes the one-shot score.

## Regression and overlap

Exact-version H310 archive GET and SHA verification precede the replay.
Original full-precision family gains reproduce exactly:
`[0.011448661872055522, 0.01526524924436801, 0.027356744513885204]`.
All 91 baseline utilities match; mean **1.1389572403748425** exactly.
Historical replay decodes fitness/gate only, no feature rows. Fresh fitting is a
separate process that receives only a PASS/hash gate and blocks historical output
paths. See [regression_receipt.json](regression_receipt.json).

Metadata overlap: **0 / 19** dev families occur among
205 TRAIN families; the same zero intersection holds for every member training
closure, because full ranker/veto fits use TRAIN and bags sample subsets thereof.
Historical member locks lack family ID lists: closure is reconstructed from pinned
code and selected metadata, not claimed to be a recorded member inventory.
Overlap IDs SHA256 `4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945`;
[overlap_audit.json](overlap_audit.json) includes TRAIN/dev ID hashes.
ID disjointness does not prove semantic independence or absence of historical
adaptive-development exposure. Non-overlap gains equal full-cohort gains here.

## Small-cluster results

Gains average within each family, then equally across families. Exact sign-flips
enumerate all **524,288** assignments, including ties; one-sided t lower bounds
use **18 df**. LOFO removes one whole family. Seeds share the same families and
draws; they are not six independent datasets.

| Root | Gain | Exact one-sided p | t18 LCB95 | LOFO min | W/T/L | Harmed | Invalid | Elo |
|---|---:|---:|---:|---:|---|---:|---:|---:|
| 30001 | +0.011448662 | 0.04882812 | +0.000184296 | +0.007014497 | 6/9/4 | 1/19 | 0 (base 0) | 1570.29 |
| 30002 | +0.015265249 | 0.00314331 | +0.006919233 | +0.013523778 | 11/5/3 | 2/19 | 0 (base 0) | 1725.29 |
| 30003 | +0.027356745 | 0.00201416 | +0.010694975 | +0.020834853 | 11/5/3 | 0/19 | 0 (base 0) | 1725.29 |
| 30004 | +0.013542218 | 0.00930786 | +0.004239125 | +0.010450598 | 11/4/4 | 2/19 | 0 (base 0) | 1675.46 |
| 30005 | +0.029545876 | 0.00456238 | +0.010970120 | +0.022709727 | 14/3/2 | 1/19 | 0 (base 0) | 1837.30 |
| 30006 | +0.032776001 | 0.00061035 | +0.016074388 | +0.026997727 | 12/5/2 | 0/19 | 0 (base 0) | 1810.54 |

**reference = r3175 slot0 (dev91)**. Family W/T/L uses H310 tolerance 1e-4;
ties never count as wins. Harm is family loss > .01. Utility remains -fitness
when gate passes and -2.0 otherwise; invalid also includes utility <= -2+1e-9.
Elo is pinned H302 two-arm MM, with reference anchored at 1500.
Pooled median Elo **1725.29**. Pooled componentwise median counts
are `{'wins': 11.0, 'ties': 5.0, 'losses': 3.0, 'harmed': 1.0, 'invalid': 0.0, 'baseline_invalid': 0.0, 'switches': 29.5}`; these describe six policies, not a new pooled policy.

All-six pooled sign-flip p **0.00004578**;
t18 interval **[0.009823936120914757, 0.03279805763733846]**, one-sided LCB **+0.011829782**;
LOFO minimum **+0.018365321**, SE **0.005467628**.
Wild bootstrap uses 10,000 shared Rademacher draws, RNG 312150, centered family
residuals and basic intervals, without studentization or leverage correction. Pooled t/jackknife uses the nonlinear median's
jackknife SE and is descriptive. Full per-seed t intervals, wild intervals,
jackknife SE/CI and family gains are in [metrics.json](metrics.json).

| Root | Drop top 1 | Drop top 5 | Drop top 10 |
|---|---:|---:|---:|
| 30001 | +0.007014497 | -0.001096525 | -0.002657582 |
| 30002 | +0.013523778 | +0.006178801 | -0.003685165 |
| 30003 | +0.020834853 | +0.006154198 | -0.001301415 |
| 30004 | +0.010450598 | +0.002128382 | -0.004277599 |
| 30005 | +0.022709727 | +0.008353330 | -0.005196302 |
| 30006 | +0.026997727 | +0.012268982 | -0.000073308 |

Pooled medians of per-seed concentration sensitivities: `{'1': 0.017179315561372952, '5': 0.006166499235779648, '10': -0.003171373146130369}`.
Each seed drops its own largest gains; these are descriptive sensitivity cuts.
Fresh-three range `[0.013542217723387701, 0.03277600148654244]`, SD `0.010299589195774854`; shared development
families prevent interpreting this seed spread as population replication.

## Decision and interpretation

The five preregistered criteria are `{'a_fresh3_median_gt_001': True, 'b_exact_p_lt05_ge4': True, 'c_t18_lcb_positive_ge4': True, 'd_lofo_min_positive_ge4': True, 'e_nonoverlap_positive': True}`. No threshold was relaxed.
Exact p<.05, one-sided t18 LCB>0 and LOFO minimum>0 hold on **6/6** seeds.
Seed30001 is borderline: exact p .048828125 and t18 LCB +.000184296; its
two-sided95% t interval crosses zero. Its H310 sign-count p .377 and this
mean-statistic sign-flip p answer different questions: the latter uses magnitudes.
The negative pooled drop-top10 sensitivity still shows concentration.
Sign-flip significance requires sign symmetry across family gains; exhaustive
enumeration does not establish exchangeability. t18 and wild intervals likewise
cannot guarantee coverage with 19 heterogeneous, potentially dependent families.
Zero TRAIN/dev family-ID overlap addresses one leak hypothesis, not these issues.
The concentrated gain and adaptive history remain relevant even if the mechanical
ROBUST rule passes. Selecting r3175 as incumbent does not by itself establish that
all subsequent adaptive comparisons are conservative. No AUC or speed verdict.

## Verification, resources and custody

11 focused tests passed: exact-flip brute force and zero/positive cases, t18/LOFO
identities, strict decision boundaries, denied dev opens/packets and poisoned
mixed-cache tail. Independent subset-sum enumeration reproduces each exact p;
independent algebra and counts reproduce the decision. No repository/hosted CI
was run under the owner's CI freeze. This is same-agent implementation review,
with ASTRA independent of H310's original author, not a human review.

xbabe2 CPU only, nice 19, math threads 1, three fitting workers within six cores
0,34,48,75,95,97 selected by mpstat; preparation used16,17,18,24,32,44.
MemoryMax8GiB, aggregate RSS guard8GiB; fit peak **2,273,779,712 bytes**,
host CPU peak **20.5%**. MemAvailable floor20GiB,
host CPU cap85%, scratch cap3GB; absolute compute stop11:15Z/close11:30Z Nov4.
No guard violation. Initial Git SSH authentication and one HTTPS server error
blocked prereg publication; child-only HTTPS writer delivery/retry resolved it.
Prereg still preceded all data access and fitting. A pre-score push lost a concurrent
branch update race; fetching the new head and ordinary nonforce retry preserved it.
No shared credential/config changes.

[manifest.json](manifest.json) pins source and the create-only S3 evidence archive;
[s3_manifest.json](s3_manifest.json) pins the manifest's verified exact S3 version.
The archive contains locks/models, old/new picks, utilities, all family statistics,
source snapshots and resource receipts. Original H310/H281 exact versions are in
[input_receipt.json](input_receipt.json). Own scratch is deleted only after archive
GET/SHA/member verification and an exact recovery mapping plus no-consumer check.
Final index records deletion; lifecycle receipt preserves publication/cleanup evidence.

Reproduction requires restoring archived `source/`, `vendor/`, receipts and
`outputs/` to the recorded custody/scratch layout, and recovering original reference
objects using input_receipt. Order: inputs.py, regression.py, fit.py, score.py,
evaluate.py. A new execution needs a separately authorized resource window; the
recorded absolute deadline must not be silently reset. The frozen fitting files
are imported byte-exact. Only fresh root enumeration and derived member seeds
extend H310; score.py uses its feature/pick functions and omits unrequested r882 arms.
