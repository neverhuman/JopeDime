# H345: ranker row subsample .6

**INCOMPLETE_CPU_ADMISSION — scientific verdict UNKNOWN.** The single required five-second probe admitted only **[19, 55, 62]** (three physical-core/SMT pairs); minimum four is required. No treatment fit, worker, model, fit lock or outer score was run. All nine ROWSUB panels, per-root differences and confidence intervals remain **unknown/null**, not zero. [PREREG.md](PREREG.md) was published at **d7f9be95a17420cdfb5255e1d9d0a395221d72bf**, 2026-11-05T01:09:31.508818+00:00, before the controller/admission attempt. No repeated probe, borrowed cores, smaller task grid or restart.

## Hypothesis and setup

The fixed hypothesis is that stronger per-tree row noise reduces reliance on individual context swings. The sole proposed treatment changes the existing ranker **subsample=.8→.6**, preserving bagging_freq1 and each original derived bagging seed. No seed keyword is added. Restoring .8 recovers both exact pinned BASE ASTs. Original LambdaRank/ndcg, labels0..7/default label_gain, depth3/leaves7/minchild20/reglambda1/colsample.8, LR/tree options and selected-round rules, random-state streams, features, veto,18-configgrid,1e-4 ties/switchrules and utility−2 remain unchanged. H338's column/feature-seed overrides are absent. This implementation was tested as source/fixtures; live ROWSUB fitting/native audit and108 treatment-veto SHA comparisons were not run.

Copied committed H338 code from origin/v1next snapshot8b70068544e20183095b45c64c8c720bc99a4911; eight original source hashes and vendor pins verified. H310/H302/H303/H308 commit819a45aa5e30e662ab511cd9962b85b2b8c1e44a,H287export213b30ad52e1ce48927bba960ed609bd98ab4dfe. Planned full36fold-root tasks: H308P0/P1/P2 × roots30001–30003 × four family outer folds,1219TRAINcontexts/205families. Historical TRAIN exposure is not fresh independent confirmation.

## Completed BASE verification and one admission probe

Exact-version GET,whole-object and selected/nested-member SHAs verified the same H316/H315 BASE objects pinned in prereg. Read-only native audit passed **12,636 BASE rankers** (11,664inner+972final),every effective bagging_fraction=.8/bagging_freq1;99originalseed→bagging_seed mappings retained. Full native/parameter SHA inventory in custody/archive. No BASE refit/newmodel score.

Current1219TRAINpools,four consumed TRAIN feature prefixes,selected metadata and r882TRAINcost hashes exactly match archived inputs;zeroDL-Valfeature rows. Fresh BASE regression passed **9/9panels,36/36fold-rootpick+lock hashes,and all3exactmedians** .0643163334285782/.04821480030677507/.04360142642936269. BASE regression intentionally replays frozen picks on full TRAIN inputs; no treatment worker was launched. Physical outer reads by treatment workers before lock=0 because workers=0; live worker qualification remains NOT_RUN. Forbidden-cohort feature/utility rows loaded0.

The **only** probe ran 2026-11-05T01:10:16.444404+00:00–2026-11-05T01:10:21.447676+00:00;5seconds,unrounded≥85%idle on both core and SMT. Candidate priority preserved;firstatmost8,min4. Three passed. The controller refused treatment and exited `exited/1/exit-code`,captured by ExecStopPost,invocationd7bf0f8c4ff94b5e9fe362cce8afaa41. Fetch/native-spec unit exited0. This refusal is the intended incomplete disposition, not passing treatment execution.

| Physical core | Idle | SMT peer | Idle | Decision |
|---:|---:|---:|---:|---|
| 15 | 96.530612% | 79 | 0.000000% | EXCLUDE |
| 19 | 90.269151% | 83 | 90.554415% | ADMIT |
| 20 | 59.918200% | 84 | 87.885010% | EXCLUDE |
| 34 | 85.714286% | 98 | 76.829268% | EXCLUDE |
| 35 | 69.918699% | 99 | 82.581967% | EXCLUDE |
| 36 | 82.510288% | 100 | 77.178423% | EXCLUDE |
| 37 | 81.818182% | 101 | 73.523422% | EXCLUDE |
| 55 | 88.683128% | 119 | 86.270492% | ADMIT |
| 60 | 81.135903% | 124 | 93.737374% | EXCLUDE |
| 62 | 87.951807% | 126 | 93.548387% | ADMIT |

## Forecasts, metrics and decision

H338 actual runtime reference was prepared:6745.614529wallseconds×4cores=26982.458116CPU-capacityseconds,includingI/O/preflight/scoring. Required formula C/(n×.9)×1.15+300. The post-admission launch forecast was **NOT_RUN** because fewer than four cores admitted; no first cell existed for the second forecast. No speed verdict.

**ROWSUB Elo: not measured.** Archived BASE reference Elo **1725.655944** is attributed to H338 closeout44bcbe16dc0cc31cb05c5bfeedd35e34697995b1; it is not newly computed H345 Elo. **reference = r882 slot0 (DL-Train outer); r3175 not measured on TRAIN**.

[metrics.json](metrics.json) preserves all nine unmeasured ROWSUB cells,per-root paired medians/CI/LCB asnull,zero actual fit/score counts,the single complete probe,BASEverification/input/spec evidence,resources and actual exit records. RNG345191/10,000whole-family draws were specified;draws executed0. All four advancement criteria remain unknown. The fixed bar requires paired drop10median≥+.002+LCB>0,gainretention≥90%,harmnonincrease andpositive medianson≥2/3roots;ties never advance. No numerical ROWSUB_NOT_ADVANCED or ADVANCE conclusion is drawn.

## Checks, resources and custody

**28 targeted tests passed**,0failed/skipped,actualexit0. They prove restoration of both exact BASE ASTs,one existing constructor-value delta with identical keyword sets/labels/groups/eval/rounds/seed expressions,veto nonmutation,foreign ablation/seed/fraction/frequency refusal,estimator signature/read-only observer,one-probe selection priority/core+SMT/no rounding,forecast .9/1.15/300/boundary behavior,bootstrap/fourgate/tie controls,identity cleanup,physical closure refusal/poisoned-tail and missing-lock controls. No synthetic optimizer fit. Scientific source hashes frozen before prereg/admission remain exact. Source syntax/JSON/whitespace/doc checks supplement source review;no hostedCI under ownerfreeze.

CPUxbabe2,nice19,maththreads1,MemoryMax6GiB/Swap0;prep/recovery/control core62 only,treatment coresused[]. Sampled maxhostCPU28.515%,load42.577,minMemAvailable71.181GiB,maxscratch1198774bytes. Fixed04:45compute/05:00closeNov5 unchanged. CPU admission failed before treatment;no reset/reprobe/cutcells/partialscore. NoGPU/qualification.lock/dev91/DLVal/DLTest/sealed/holdout/meta-test/fresh/livecheckpoint/foreignjob/forbiddenhost action. DOPE `{"composite_sha256_prefix":"c51a56d0","loaded":false,"scope":"CPU selection layer only"}`.

Own BASE/control/source/test/admission receipts are archived create-only under H345; exact-version wholeSHA plus every member verified before enumerated own scratch deletion. [s3_manifest.json](s3_manifest.json) pins the archive version. No treatment model files exist. Source/small receipts retained under/home/ubuntu/jope-rescue-20260921/h345-solhwv. Closeout `/home/ubuntu/jope_plan/evidence/H345/solhwv-20261105-t0100/CLOSEOUT.json`. Only nonforcev1next/H345study/twoassignedindexrows;noPR/main/fastlane/CI/FINDINGS/foreigncheckout/index mutation.

Next: preserve the incomplete admission evidence; no dev91 or H345 retry. A later wave needs a new owner-assigned resource window.
