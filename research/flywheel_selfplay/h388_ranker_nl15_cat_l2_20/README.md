# H388: NL15 ranker cat_l2 10 → 20

**CLOSED — INCOMPLETE_CPU_ADMISSION. Scientific verdict: NOT_EVALUATED.**

The single five-second admission sample found only **[15, 16, 21]** qualifying, below the required minimum of four. The controller exited before creating any fitting workers. There was no second admission sample, restart, borrowed pool, reduced design or partial scoring. This refusal is not CATL2_NOT_ADVANCED. Retain frozen NL15; CATL2 quality remains unknown.

| Record | Observed |
|---|---|
| Qualifying cores / minimum | 3 / 4 |
| Same-interval host CPU | 24.873346% (limit 70%) |
| Admission samples / retries | 1 / 0 |
| Treatment campaign constructors / native fits | 0 / 0 |
| Fit locks / outer score markers | 0 / 0 |
| Required tasks not started | 36 / 36 |
| Initial / first-task forecast | NOT_RUN / NOT_RUN; admission refused before these stages |
| CATL2 gains, harm, CIs, Elo and configuration agreement | NOT_EVALUATED; unknown, not zero |

## Hypothesis and setup

The hypothesis is that stronger categorical L2 regularization damps categorical splits that follow context-like noise. Fresh native recovery verified archived H352 cat_l2 = 10 across 972 rankers and all 29,484 ranker fit rows, so the sole prospective treatment is cat_l2 = 20 for both ranker constructors, including inner fits, final refits and all eight bags. No categorical feature declarations, labels, groups, features or seed expressions are changed. Removing the added cat_l2 keyword restores both complete committed H352 function ASTs exactly. No new seed keyword.

The earlier H380 bagging-frequency intervention is removed: bagging_freq = 1, path_smooth = 0, bagging_fraction = 0.8 and feature_fraction = 0.8. Leaves 15 / depth 3, minimum child count 20, L1 = 0 / L2 = 1, minimum split gain 0, Hessian 0.001, sigmoid 1, original LR grid (0.03, 0.05, 0.10), rounds (50, 100, 200), main rounds 200, the original configuration grid, 1e-4 ties and sentinel -2 remain fixed. The original veto is unchanged prospectively; actual treatment veto equality was not exercised because no model was fitted. H380 stays CLOSED BFQ5_NOT_ADVANCED without relabeling.

[PREREG.md](PREREG.md) and all 35 frozen source files were published at `c64dc60ea68fc44fd5589234ed4e8d84188e7e91` on 2026-11-05T18:23:58.020738+00:00, before campaign launch. The transport is committed H380 `b16b7354e87e394784ec17ace5894b5a766aad54`, manifest `c808ebbff7b9f06b8c929348039683c509b46e18dc0dafe34c974bfa83bd4af5`, anchored to H352 `02dfee2f7e0b8543324494dd4f99d074cd6b93b6`.

## Verification and unevaluated decision

Fresh exact-Version GETs, whole-object and every-member hashes passed for BASE and H352 NL15. BASE nine panels / 36 fold-root pick-lock pairs / exact medians 0.0643163334285782, 0.04821480030677507, 0.04360142642936269 matched. NL15 all 27 parts / 2,691 members, nine panels, 36 locks and COMPLETE score markers, raw family arrays, 972 rankers / 108 veto identities and all 36,504 native-fit reference headers passed. Native cat_l2 = 10, bagging_freq = 1 and path_smooth = 0 were verified before admission. Current TRAIN identities covered 1,219 contexts / 205 families and four consumed cache prefixes, with zero DL-Val feature rows. There were no BASE/NL15 refits or new baseline model predictions.

All 48 data-free source controls passed, zero failed or skipped, with zero real LightGBM fits, predictions, real feature reads or admission probes. A separate structural proof restored both complete functions to actual committed H352. These establish source and fixture behavior. Actual CATL2 native model behavior, veto equality, optimizer runtime, numerical quality and all decision gates were NOT_RUN.

`reference = r882 slot0 (DL-Train outer); r3175 not measured on TRAIN`

The preserved preregistration uses primary CATL2 − NL15 and secondary CATL2 − BASE, 10,000 whole-family draws / RNG 388191, paired drop-top-10, and per-root point estimates and intervals. ADVANCE requires primary median ≥ +0.002 AND one-sided LCB95 > 0; gain ≥ 90% NL15; harmed share ≤ NL15; and positive primary root medians on at least two of three roots. Ties never advance. No scientific bootstrap or decision gate was evaluated. This belongs to the exploratory H348–H388 sweep without a multiplicity-adjusted confirmation claim. No AUC, speed, dev91 or serving verdict is produced. [metrics.json](metrics.json) preserves unknown quality as null and records the observed zero campaign counts separately.

## Resources, custody and next step

Actual campaign invocation `21a5080500ed4afbb4734d6d6016c2fd` exited with application status 2 at 2026-11-05T18:24:35.384280+00:00. The zero-fit refusal/custody unit exited 0. The authorized union was physical cores 15–26 plus siblings 79–90 and fallback 46–49 plus siblings 110–113. The observed both-SMT idle minima for the qualifying cores were 15:97.18%, 16:87.17%, 21:85.04%; every other pair was below 85%. The host-CPU gate passed; the minimum-core gate failed. The fixed November 5 22:15Z compute / 22:30Z close window was not reset.

Resource limits were existing xbabe2 CPU only, nice 19, math threads 1, MemoryMax 6 GiB / SwapMax 0 / CPUQuota 800%, scratch ≤2 GB. The corrected guard uses host CPU >85% averaged over completed 60-second windows, load1 >108 on three consecutive 60-second samples, available memory <20 GiB, or scratch >2 GB, and stops only the own unit. The former instantaneous CPU85 / absolute-load85 stop rules are retired. A fresh aggregate CPU/load window was collected only for safe cleanup; it was not a per-core admission probe or new scientific attempt. No paid compute allocation or model-service call was made.

Create-only main archive: `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H388/gov03-closeout-20261105T182833Z/h388-scratch.tar`, VersionId `oWeU5ZsettOb42eh33eSpqw_P3f29LWq`, SHA256 `ead45652922a214baf960f456d761d719aaedac24cfd5c03e5408b81889987a9`. Exact-Version GET, whole-object hash, all 1388 members and original-source reread passed. There are zero generated treatment components. The archive preserves the source, frozen baseline copies, regression/control/admission/resource receipts and empty component manifest. Receipt object VersionId: `V.IJ3Oz9s09l9BU0icLgp_Idmpl68bpf`. Own scratch (175,080,191 bytes) and the generated restoration were removed only after verification and a fresh zero-consumer check. Immutable local main archive, source and receipts remain at `/home/ubuntu/jope_plan/evidence/H388/gov03-20261105-t1805`.

DOPE c51a56d0 is unchanged and unloaded, CPU selection layer only. No protected/sealed/DL-Val/dev91/DL-Test payloads, GPU, qualification.lock, writer or AUTORESTART mutation, forbidden-host access or foreign-job signals. Canonical HEAD/index, 62 foreign staged whitepaper files and other dirty files were preserved. Only the private index and non-force origin/v1/next are used; no PR, main, fastlane, hosted CI or models in Git. Hosted CI was NOT_RUN under CI_FREEZE+BATCH_WINDOW. A new treatment attempt requires a separate owner-authorized wave. H388 will not be reprobed or resumed.
