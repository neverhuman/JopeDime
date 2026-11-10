# H348: ranker min_split_gain rerun

**CLOSED — INCOMPLETE_RESOURCE_GUARD. Numerical verdict: UNKNOWN / not scored.**

## Hypothesis and setup

Fixed ranker min_split_gain=0.01 may reject tiny noisy LambdaRank splits. This reruns H346's unmeasured setting using committed H347 harness 6f87ae99ce2752d595663c181aeb2457bd231939; H347 remains closed. The only ranker override is min_split_gain=0.01, native min_gain_to_split0.01 versus BASE0. Ranker reg_alpha stays BASE0, removing H347's L1 arm. Both complete pinned H287 function ASTs restore exactly when the omitted default is restored; explicit zero has identical constructor parameters.

LambdaRank/ndcg, sigmoid1, labels0..7/default gains, Hessian0.001, min_child_samples20, leaves7/depth3, subsampling, L2, LR/tree options, seeds, features, original18-config grid, tie/switch1e-4 and sentinel-2 remain unchanged. Veto retains reg_alpha1/reg_lambda2. [PREREG.md](PREREG.md), commit `827eb8d768a35889b500ea19b72837d8e03fb54f`, was published 2026-11-05T04:05:30.229726+00:00, before first actual fit 2026-11-05T04:05:57.919846+00:00.

The TRAIN-only H303/H308 design remains P0/P1/P2 × roots30001–30003 ×4 outer family folds:36tasks/9evaluation cells,1219contexts/205families. BASE first passed exact-version GET, whole/member SHA,9panels/36fold-root pick+lock hashes, and exact medians0.0643163334285782 /0.04821480030677507 /0.04360142642936269. All972BASE rankers show native min_gain_to_split0 and lambda_l1zero;108BASE veto identities verified read-only. No BASE refits or new BASE model predictions. Current TRAIN inputs and four consumed cache prefixes matched before the run and after stop.

## Outcome and verification

At 2026-11-05T05:15:31.146230+00:00, host CPU reached 91.368434%, exceeding the fixed85% runtime guard. The guard stopped only h348-run.service; actual process exit is killed/TERM at 2026-11-05T05:15:31.310535+00:00. Systemd SERVICE_RESULTsuccess records the requested stop and does not mean science completed.

- Completed native fits: **34,072** (27,516ranker;6,556veto); constructor records34,078.
- Six constructor outcomes have no completed native record and remain **UNKNOWN**, not failed or zero. No malformed audit tails were found.
- Completed fit tasks/locks/verified inner archives: **30/36**. Six tasks remain unfinished. Their raw files and audit logs are preserved unchanged.
- The30completed tasks have **90/90** final veto SHA matches; the required full108/108 was not reached.
- Completed-task guards report physical outer rows/packets before lock0. Six unfinished tasks lack final runtime guard receipts: **UNKNOWN**, so no full36scope qualification is claimed.
- Outer score markers/model predictions: **0**. No fit_done or scientific metrics were produced.

Every completed native audit record passed constructor/native validation. All11,700saved models belonging to completed tasks were bound to successful native records; refit file SHAs and all360members per completed inner archive were verified. All42data-free controls passed with0failures/skips. These are same-agent source and partial-evidence reviews. Hosted CI was NOT_RUN under CI_FREEZE+BATCH_WINDOW.

## Metrics and locked decision

MINGAIN family mean/median gains, confidence intervals, drop1/5/10, LOFO, W/T/L, harm, invalids, switches, per-root medians and paired CIs, Elo/EloCI, Jaccard and configuration agreement are **NOT MEASURED**. [metrics.json](metrics.json) uses nulls for these unknown quantities. No partial score, aggregate, ADVANCE or MINGAIN_NOT_ADVANCED verdict is reported.

`reference = r882 slot0 (DL-Train outer); r3175 not measured on TRAIN`

The unchanged preregistered decision requires paired drop10 median>=+0.002 with one-sided LCB95>0, gain retention>=90%BASE, no harmed-share increase, and positive root medians on>=2/3roots. Paired10kwhole-family RNG348191 resamples are shared across arms/all9cells; each arm redrops its own top10. Per-root paired intervals were preregistered but not measured because scoring never started. No AUC or speed verdict.

## Resources and boundaries

ONE5-second/core+SMT>=85%-idle sample admitted cores1–6, first six passing pairs; same-interval host CPU10.138212%<=70. Controller threads and all8own processes were bound to admitted cores before fits. Both required forecasts passed: initial05:56:17.570770Z; after first completed task06:05:10.393912Z. The runtime resource guard still applied and stopped execution. Fixed November5 compute06:15Z/close06:30Z were not reset. No re-probe, fit restart, borrowed cores, worker expansion or cut-cell scoring occurred.

Actual unit Thu 2026-11-05 04:05:31 UTC to Thu 2026-11-05 05:15:31 UTC: nice19, numeric1,6workers, MemoryMax6GiB/SwapMax0. Peak memory3.462GiB; swap0; maximum scratch1.416GB. Retained preparation/run samples show host CPU maximum91.368434%, load maximum61.80, minimum available memory72.97GiB. CPU time is accounting, not a speed comparison. No foreign jobs or forbidden hosts were changed; H335/H349 were left alone. No GPU/qualification.lock, live writer, dev91 or serving action. Frozen source retains TRAIN-only restrictions; incomplete runtime proof is disclosed above. DOPEc51a56d0 remains unchanged/unloaded, CPU selection layer only.

## Custody and next step

All partial scratch, raw audit logs, partial model bytes, completed inner archives, pinned source/vendor and provenance were preserved create-only at `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H348/gov03-closeout-20261105T052103Z/h348-scratch.tar`; VersionId `F0XNP_2IJoHh37ljlK6Y3VqQAi0FyK2_`; SHA256 `6042d24b3f0760c36ad126390f69418be87171257158c74d2be96eb05bf76718`. Exact-version GET, whole SHA and **all3868file members** passed, plus gzip CRC verification for all30completed inner archives. This certifies preserved bytes, not validity of unfinished models. Custody receipt VersionId `7hkmAaUp4k8renOMtB_hT9mKgCbVd.IX`. Only then was own scratch1,417,359,199bytes and the generated restore removed after zero-consumer checks. Immutable local archive/source/evidence remain at `/home/ubuntu/jope_plan/evidence/H348/gov03-20261105-t0400`.

Retain frozen BASE. H348 has no numerical decision. Any new fits need a separate owner assignment and bounded window; no automatic retry, dev91 evaluation or serving promotion. Private index/nonforcev1/next only; canonical HEAD and all foreign staged/dirty files preserved. The index changed externally at04:17:31Z from the original claim checksum ae3a3113fd0c91ddcdf4419eda281abe8a178e57a5fbdee6806fbeb2955d02d5 to a78b7cab3f43caaeadb395e6cfeee7cfc12fea6eb71439b266b06237602f8955 with62staged whitepaper paths; current bytes were copied/preserved with no reset or shared-index mutation. Final publication guards the observed index; the original claim snapshot remains in custody. No models in Git or PR/main/fastlane/hostedCI operations. H347 and prior completed waves stay closed.
