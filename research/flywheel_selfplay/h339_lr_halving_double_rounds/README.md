# H339 — LR halving and doubled constructor rounds

**INCOMPLETE_RUNTIME_FORECAST. Scientific outcome UNKNOWN; no numerical verdict.**
The first complete cell took **1059.808s**. The frozen
full36 forecast on **4** admitted cores was
**2026-11-05T02:00:48.549516+00:00**, beyond the fixed
**2026-11-05T01:45Z** compute stop. The run stopped after that cell; no remaining
cells or treatment outer scores ran. This is not LRHALF_NOT_ADVANCED and provides
no result about whether the treatment improves utility, harm, Elo or seed stability.

## Hypothesis and exact implementation

Smaller boosting steps may improve family robustness. Only two keyword values
change at pinned H287213b30ad52e1ce48927bba960ed609bd98ab4dfe
`topk.fit_ranker` and `bagged.fit_ranker_n`: learning_rate=BASE_lr/2,
n_estimators=2*BASE_n. Rates.03/.05/.10 become.015/.025/.05; topk200 becomes400,
bags50/100/200 become100/200/400. No early stopping, callback, label_gain override,
changed labels, extra trees, soft voting or sample weighting. Reversible AST
proof restores exactly the original constructor values.
[Implementation](regularization.py), [transform receipt](regularization_transform_preview.json).

**The source limitation was preregistered before outcomes:** unchanged
`veto.py:90-92,150` and `ensemble.py:462-485` still request selected prefixes
50/100/200. Doubling fitted trees does not double those scoring prefixes. The
nominal full-fit lr*T product is equal, but the same selected prefix has half
its former nominal shrinkage; selection may choose a different prefix. This
literal two-keyword experiment does not isolate equal effective scored shrinkage.
There was no silent prediction/grid/serialized-metadata change to repair that
mechanistic premise. Any future effective-prefix doubling needs its own spec.

LambdaRank, ordinal labels/default label_gain, depth3/leaves7/L2=1/feature.8/
minleaf20, row sampling, features, families, seeds, gate code and classifier
parameters, original inner selection/18-config grid/ties/switches and failure
utility-2.0 are unchanged. All3final veto hashes match archived BASE. No neural
optimizer update, live-loop change or serving claim. DOPE weights were not loaded:
`{"composite_sha256_prefix":"c51a56d0","loaded":false,"scope":"CPU selection layer only"}`.

## Preregistration, regression and scope

[PREREG](PREREG.md) **5543bd6e3648c371eaddf04bdda9c61fc18c7ec7** preceded every fit, including synthetic
controls. Tested implementation **7697c42be062a387efc88554d30228c233c2a3cd** preceded the first
scientific cell. [Vendor pins](vendor_pins.json) retain the unchanged H303/H308
family folds, seeds and H287 exports.

Fresh exact-version H318 BASE recovery and unchanged replay passed36fold/root
hashes,9pick/raw-utility panels and3medians exactly:
.0643163334285782,.04821480030677507,.04360142642936269.
[Regression](regression_receipt.json), [input recovery](input_receipt.json),
[archive identity](base_archive_pin.json). No BASE scientific refits.
Required BASE replay reads TRAIN outer rows separately; zero prelock outer reads
describes the treatment worker. TRAIN-only1219contexts/205families; no DL-Val.

Only P0/fold0/root30001 completed:819ranker fits,1fsynced lock,
27final ranker native files and3unchanged final veto hashes audited. Physical
packet/range identities and inner18-choice selection were independently checked.
All actual fit records show exactly the prescribed LR and constructor-round
changes; no fit restart. [First-cell audit](first_cell_review.json).
All36locks and full29484ranker audit were NOT reached. Zero treatment outer
scores and zero bootstrap comparisons; scientific metrics remain null.

## Runtime admission and verification

After the first completed cell, frozen admission is
`ceil(35/workers)*first_cell_seconds*1.15 +600s` strictly inside the remaining
compute window. Here remaining batches=9; predicted
remaining work=11569.013s.
15% reserve covers different folds and600s reserves audit/scoring/analysis.
No reduced reserve, omitted cells, deadline reset or historical-timing substitute.
[Runtime forecast](runtime_forecast.json), [fit orchestration](fit.py),
[forecast controls](forecast.py).

30tests passed, zero failures/errors/skips:8LR/native/prefix/forecast,
4physical-input-boundary,18analysis controls. Exactly2tiny treatment-only synthetic
native fits; no scientific BASE refits. [Tests](tests_receipt.json),
[LR controls](test_regularization.py), [boundaries](test_boundary.py),
[analysis](test_analysis.py). Review is by the implementing agent; independent
recalculation does not imply an independent reviewer. Full hosted CI NOT_RUN
under owner freeze.

CPUxbabe2,nice19,numeric1,MemoryMax6GiB/Swap0,hostCPU/load<=85,
MemAvailable>=20GiB,scratch<=2GB. Fresh idle probe admitted5/6/8/33; scientific
first-cell work used5only.41/105 excluded for H335,4excluded at83.1%idle.
Preparation initially inherited H337core4 for recovery/start of mandatory BASE
replay; remaining replay was moved to5 without restart and the old controller
stopped. Initial fit launcher exited before any scientific fit because it treated
H335's sleeping unbound launch supervisor as a worker. Exact declared worker CPUs
and all descendant affinities were checked; corrected classification admitted
fitrun2 without changing peer processes or resource limits. Both original receipts
are retained. Peak sampled RSS809,648,128bytes;
hostCPU56.0%; no resource threshold violation.
[Resource receipts and corrections](resource_receipt.json), [dependency versions](dependency_pins.json).

## Metrics policy and interpretation

Full-study decision, not executed: paired arm-specific drop10median9>=+.002,
10kwhole-family bootstrapRNG339191 one-sided95LCB>0, gain>=.9BASE,
harm<=BASE, and>=2of3strictlypositive rootmedians. Exact+.002inclusive;
zeroLCB/ties fail. Family-equal gain/CI/drop1,5,10/LOFO/W,T,L/harm/invalid/switches/
config agreement and Elo all require complete outer evaluation.
**reference = r882 slot0 (DL-Train outer); r3175 not measured on TRAIN**.
No new LRHALF Elo, AUC or speed verdict. [Machine-readable metrics](metrics.json).

H337's harm reduction failed retention/seed/robustness gates; it does not prove
the smaller-step mechanism. H331 cosine and H332 leaf clipping tested different
factors. Neither justifies a positive or negative H339 scientific verdict from
one fit cell. TRAIN is adaptively reused and nine cells would be dependent.

## Custody and next step

First-cell models create-only archived and exact-versionGET/wholeSHA/everymember
verified before inner-file reclamation. [Model archive](model_archive_index.json).
Remaining source/evidence is preserved under
`s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H339/astra-20261104T222500Z/`.
[Manifest](manifest.json), [S3 receipt](s3_manifest.json). Every remaining own
scratch file requires a verified recovery mapping and no active consumer before
deletion. Foreign work is retained. Original localH50/H58/H276 S3versions remain
unestablished; allowed packet/range identities are retained.

Fixed01:45compute/02:00closeNov5 retained. No GPU/dev91/DLVal/DLTest/sealed/
holdout/meta_test/fresh/foreignjobs/livewriter/PR/main/fastlane/CI action.
No automatic rerun or dev91 advancement. A future full36 attempt requires a new
authorized window and preregistration; effective-prefix doubling is a separate
specification question, not a post-outcome edit to this wave.
