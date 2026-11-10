# H341 — Ranker histogram cap 63

**INCOMPLETE_CPU_ADMISSION. Scientific outcome UNKNOWN; numerical verdict null.**
The single five-second admission probe at 2026-11-04T23:13:12.840664+00:00 admitted only cores
[27, 33]. At least four cores, each with its SMT sibling at least 85%
idle, were required. No scientific fit, first-cell forecast, treatment outer score
or bootstrap comparison ran. No repeat probe or relaxed threshold was used.
This capacity result does not establish MAXBIN_NOT_ADVANCED.

## Hypothesis and implementation

Coarser feature histograms may regularize noisy split thresholds. Native metadata
from all 972 recovered BASE rankers and 108 veto classifiers showed max_bin=255,
with no per-feature bin override. [Native audit](base_native_maxbin.json).
The sole treatment adds max_bin=63 to pinned H287213b30ad52e1ce48927bba960ed609bd98ab4dfe
topk.fit_ranker and bagged.fit_ranker_n. Removing that keyword restores both exact
BASE ASTs. Every native fit receipt compares the complete constructor parameter
dictionary to an unfitted BASE constructor and requires only max_bin to differ.
[Implementation](regularization.py), [inverse proof](regularization_transform_preview.json).

Each ranker fit must construct a new Dataset from NumPy rows with no reference,
an initially absent native handle, max_bin=63 and realized native feature bin
counts no larger than 63. Pre-binned Dataset input is rejected. The observer is
restored before unchanged veto fits. Synthetic tests proved fresh identities,
preservation of a separately constructed BASE-bin255 Dataset, and unchanged
native veto bins255. [Controls](test_regularization.py).
The cap does not guarantee fourfold fewer realized thresholds or a speed benefit.

LambdaRank, ordinal labels/default label_gain, leaves7/depth3, L2=1,
minleaf20, subsample/colsample.8, learning rates, tree budgets, random seeds,
features, eight bags, veto, 18-config grid, selected prefixes, ties/switches and
failure utility -2.0 remain BASE. H339 LR halving/doubled rounds are absent.
The planned arm covers all inner and final ranker fits; none ran scientifically.

## Preregistration and validation

[PREREG](PREREG.md) **f11e106f16d9650ab56e2a02eb8675ecf7e74e56** was published before every fit, including
the synthetic controls. Reviewed source **98c3c6f7531e8de5838c9e33f9b79ff89feb7c3f** was published
before the admission probe. [Vendor pins](vendor_pins.json).
Fresh exact-version archived BASE recovery, payload/member hashes and replay
matched all 36 fold/root hashes, nine picks/raw-utility panels and three medians:
.0643163334285782, .04821480030677507, .04360142642936269.
[Regression](regression_receipt.json), [recovery](input_receipt.json).
All 1080 recovered model hashes were rechecked unchanged before custody.
No BASE scientific refits. Required BASE replay reads TRAIN outer rows separately;
the zero prelock treatment count does not imply that replay read no TRAIN rows.

All 32 tests passed without failures/errors/skips: 10 max-bin/native/Dataset/
forecast controls, four physical input-boundary controls and 18 analysis controls.
Exactly two synthetic ranker fits and one unchanged synthetic gate fit ran;
the synthetic BASE Dataset was constructed without fitting a BASE model.
[Tests](tests_receipt.json). No model trained on scientific inputs in this wave.
Same-agent review; no claim of an independent reviewer. Full hosted CI NOT_RUN
under the owner freeze.

## CPU admission and fixed window

| Core / SMT | Core idle | SMT idle | Result |
|---|---:|---:|---|
| 4 / 68 | 83.75% | 73.60% | excluded |
| 5 / 69 | 67.49% | 86.13% | excluded |
| 6 / 70 | 86.21% | 82.19% | excluded |
| 8 / 72 | 54.85% | 39.79% | excluded |
| 27 / 91 | 88.80% | 92.47% | admitted |
| 33 / 97 | 86.75% | 85.21% | admitted |
| 57 / 121 | 82.25% | 78.22% | excluded |
| 59 / 123 | 84.65% | 85.51% | excluded |

[Single admission receipt](cpu_admission.json). Preparation used core57;
post-probe custody uses admitted core27. Scientific worker count zero.
CPU xbabe2, nice19, numeric threads1, MemoryMax6GiB/MemorySwapMax0,
hostCPU/load <=85, MemAvailable >=20GiB, own scratch <=2GB. Sampled preparation
peak RSS 779,173,888 bytes; hostCPU
49.5%. [Resources](resource_receipt.json).
Fixed compute02:15Z/close02:30Z Nov5 retained. H335/H338/H340 reserved cores and
all foreign jobs were left alone. No new core search or deadline extension.

## Metrics, limitations and next step

The preregistered full study was TRAIN1219contexts/205families, H303 NESTED on
H308 P0/P1/P2 × roots30001–30003 × four family outer folds. All36 fsynced fit
locks and native/input audits must precede36 once-only treatment outer scores.
The first-cell runtime rule remains ceil(35/workers)*elapsed*1.15+600 seconds
strictly inside the compute window; it was NOT_RUN because CPU admission failed.

The unexecuted decision requires paired arm-specific drop-top10 median9 >=+.002,
10k whole-family bootstrap RNG341191 one-sided LCB95>0, gain >=90% BASE,
harm <=BASE and positive paired medians for at least two of three roots.
Exact+.002 is inclusive; zero LCB or tied roots do not pass their strict gates.
Gain/CI/drop1,5,10/LOFO/W,T,L/harm/invalid/switches/per-root/config/seed-Jaccard/Elo
are unknown for MAXBIN63, not zero. [Metrics](metrics.json).
**reference = r882 slot0 (DL-Train outer); r3175 not measured on TRAIN**.
No AUC or speed verdict. No automatic rerun or dev91 advancement.

Cross-critique: H339's runtime failure supplies no scientific evidence against
smaller learning rates, just as this CPU failure supplies none against max_bin63.
The H328 recorded decision failed harm; the authority's shorthand about an
additional seed failure is not imported as a verified observation. H338's seeded
feature sampling and H340's objective swap test different changes and cannot
substitute for an unmeasured H341 result. Reused TRAIN families and dependent
partition/root cells would limit a completed bootstrap; no wider-search correction.

## Custody and scope

Evidence is create-only under
s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H341/astra-20261104T230000Z/.
[Manifest](manifest.json), [S3 receipt](s3_manifest.json). Exact-version GET,
whole-archive SHA and every member hash must verify before own scratch deletion;
the cleanup receipt is retained in lifecycle custody. Original local H50/H58/H276
S3 versions remain unestablished; allowed input byte ranges and packet hashes
are recorded. Source and scientific models are distinct evidence; no models in Git.

DOPE: {"composite_sha256_prefix":"c51a56d0","loaded":false,"scope":"CPU selection layer only"}.
No GPU/qualification.lock/dev91/DL-Val/DL-Test/sealed/holdout/meta_test/fresh inputs,
xbabe0/3, live writer, foreign job, PR/main/fastlane or hosted CI action.
A future attempt requires a new authorized window and preregistration.
