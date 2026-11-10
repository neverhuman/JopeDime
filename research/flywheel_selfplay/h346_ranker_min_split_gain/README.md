# H346 — Ranker minimum split gain 0.01

**INCOMPLETE_CPU_ADMISSION. Scientific outcome UNKNOWN; numerical verdict null.**
The single five-second probe at 2026-11-05T01:36:58.686333+00:00 admitted only cores28/30 and their
SMT siblings. Four pairs were required. No scientific fit, fit lock, treatment
outer score, runtime forecast or bootstrap comparison ran. No probe retry,
threshold relaxation, alternate cores or deadline reset. This is not
MINGAIN_NOT_ADVANCED.

## Hypothesis and implementation

The fixed minimum split gain may reject small LambdaRank splits and regularize
selection. Its relationship to live score cycling or the historical train/val
gap remains a hypothesis. The committed H343 harness was copied from
2b48ec8eedac5b17fe3f444d7544b6f76bed62c7; its max-bin treatment was replaced with
min_split_gain=0.01 only. Default bins remain255. [Copy receipt](source_copy_receipt.json).

All972 recovered BASE rankers and108 veto classifiers explicitly serialize
min_gain_to_split=0 and max_bin=255. [Native BASE audit](base_native_mingain.json).
Only pinned topk.fit_ranker and bagged.fit_ranker_n receive min_split_gain=.01,
covering inner/final/eight-bag rankers. Restoring zero and canonicalizing the
omitted default restores both exact BASE ASTs. Every planned fit records full
BASE and treatment constructor dictionaries and native parameters; only this
one value may differ. [Implementation](regularization.py), [inverse proof](regularization_transform_preview.json).

All model seeds, LR/tree budgets/prediction prefixes, LambdaRank/ndcg,
labels0..7/default label_gain, depth3/leaves7, minleaf20, subsample.8/frequency1,
colsample.8, L2=1, features, veto, 18-config grid, ties/switches and -2.0 sentinel
remain BASE. No H343 max_bin63 or H338 extra seed keyword. A complete run would
require all108 final veto hashes identical to BASE; that scientific check did
not run. Fresh-Dataset and unchanged synthetic veto checks passed.

## Validation and chronology

[PREREG](PREREG.md) **fe59efd6681a82878ad7b443572e74595b1e305c** preceded even synthetic control fits.
Reviewed implementation **aed4f9c75592be9fdf4a5b374bb54fd4ef59130f** preceded the capacity probe.
Fresh exact-version archived BASE recovery and replay matched all36 fold/root
hashes, nine pick/raw-utility panels and partition medians
.0643163334285782/.04821480030677507/.04360142642936269 exactly.
[Recovery](input_receipt.json), [regression](regression_receipt.json), [vendor pins](vendor_pins.json).
All1080 recovered BASE copies were rehashed unchanged before custody. No BASE
scientific refits. Mandatory reference replay separately read TRAIN outer rows;
zero treatment prelock reads does not mean the reference replay read no rows.

All35 controls passed with zero failures/errors/skips:10 intervention/forecast,
four physical input-boundary and21 analysis controls. Two synthetic ranker fits
and one unchanged synthetic gate fit; one BASE Dataset construction without fit.
The added controls check literal-median, per-root and Elo intervals from shared
family draws. [Tests](tests_receipt.json). No model trained on scientific inputs.
Same-agent review, not an independent reviewer. Hosted CI NOT_RUN under freeze.

## Admission and immutable window

| Core/SMT | Core idle | SMT idle | Result |
|---|---:|---:|---|
| 27/91 | 82.08% | 96.14% | excluded |
| 28/92 | 87.76% | 89.68% | admitted |
| 29/93 | 66.94% | 92.87% | excluded |
| 30/94 | 85.34% | 87.42% | admitted |
| 31/95 | 84.01% | 87.42% | excluded |
| 32/96 | 81.47% | 85.60% | excluded |
| 33/97 | 80.61% | 89.43% | excluded |

[Admission receipt](cpu_admission.json). The preparation core was27; post-probe
custody uses28. No scientific workers launched. CPU xbabe2,nice19,numeric1,
MemoryMax6GiB/MemorySwapMax0,hostCPU/load<=85,MemAvailable>=20GiB,scratch<=2GB.
Sampled preparation peak RSS 780,005,376bytes and hostCPU
30.9%; no guard violations. [Resources](resource_receipt.json).
Original compute05:15Z/close05:30Z Nov5 remained fixed. Reserved H335/H342/H344/
H345R cores and foreign jobs were untouched.

The frozen admission policy uses H338's successful6745.614529-second four-worker
run, capacity26982.458116seconds, divided by workers*.9, times1.15, plus300seconds.
[Policy and original timing receipts](admission_forecast_policy.json).
It was NOT_RUN because fewer than four pairs admitted. The first-cell policy,
ceil(35/workers)*first-cell seconds/.9*1.15+300, also did not run. Neither policy
provides a speed verdict.

## Metrics and next step

The planned study uses TRAIN1219contexts/205families, three partitions, three
roots and four family outer folds. All36 fsynced locks and input/native audits
must precede once-only scores. Required metrics are family-equal mean and
literal-median gains, their CIs, drop1/5/10, LOFO, W/T/L, harm, invalid/switches,
per-root medians/CIs, Elo/EloCI, config agreement and seed Jaccard. Every MINGAIN
numerical result is unknown/null, not zero. [Metrics](metrics.json).

The unexecuted advancement bar is paired arm-specific drop10 median9>=+.002,
10k paired whole-family bootstrap RNG346191 LCB95>0, gain>=90%BASE,
harm<=BASE and at least two of three positive root paired medians. Exact+.002
is inclusive; zeroLCB/zero roots fail strict clauses. Elo retains400-step
Bradley–Terry MM,pseudocount.01,ties omitted; whole-family uncertainty shares
indices across arms and dependent cells. Label:
**reference = r882 slot0 (DL-Train outer); r3175 not measured on TRAIN**.
No AUC or speed verdict. No automatic rerun or dev91 advancement.

Cross-critique: H343's completed max-bin test did not advance, but it changed a
different parameter. H345's CPU-admission refusal supplies no negative row-
subsampling evidence, just as H346 supplies no minimum-split-gain outcome.
H338's explicit feature seed also changes the existing tree sampling sequence;
that two-keyword intervention cannot isolate node sampling. Reused TRAIN and
adaptive wave search remain limitations of any completed comparison.
[Peer-source receipts](peer_crosscritique.json).

## Custody and scope

Create-only evidence prefix:
s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H346/astra-20261105T013000Z/.
[Manifest](manifest.json), [S3 receipt](s3_manifest.json). Exact-version GET,
whole SHA and every member must verify before own scratch deletion. Recovery
mapping and no-consumer check precede deletion; lifecycle custody retains the
cleanup receipt. Original local H50/H58/H276 S3versions remain unestablished;
packet/range hashes are retained. No models in Git.

DOPE: {"composite_sha256_prefix":"c51a56d0","loaded":false,"scope":"CPU selection layer only"}.
NoGPU/qualification.lock/dev91/DL-Val/DL-Test/sealed/holdout/meta_test/fresh,
xbabe0/3,live writer,foreign-job,PR/main/fastlane/hostedCI actions. A future
attempt requires new authorization and preregistration.
