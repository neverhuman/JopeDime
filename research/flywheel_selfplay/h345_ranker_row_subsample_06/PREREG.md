# H345: ranker row subsample 0.6 — preregistration

Owner sol_v1_hwv; authority `/home/ubuntu/nudges/hw_t0100_solhwv_h345_rowsub.md`, claimed 2026-11-05T01:01:43Z. Copy the committed H338 harness at origin/v1/next `8b70068544e20183095b45c64c8c720bc99a4911`, with its eight source hashes verified against the committed manifest. H338 is DONE + RELEASED. This document is pushed nonforce to origin/v1/next before any treatment fit; the README will quote its full commit. One ROWSUB arm, fixed setting, no tuning or extra cohort.

## Hypothesis and sole treatment

Stronger per-tree row noise may reduce reliance on single-context utility swings. **Only the existing ranker constructor `subsample` value changes from .8 to .6**. Native LightGBM bagging_fraction is the corresponding alias. Keep subsample_freq/native bagging_freq **1**, the original per-fit random_state and its original derived bagging_seed, default tree-level feature_fraction=.8, and native feature_fraction_bynode=1. **No new keyword or seed is added**. H338's feature_fraction_bynode/feature_fraction_seed overrides are absent; its seed change is not carried into H345.

Patch only pinned H287 `topk.fit_ranker` and `bagged.fit_ranker_n`, including every inner ranker, final main refit and all eight bag rankers. Restoring the original `.8` constant must reproduce each exact BASE function AST. Tests compare full constructor dictionaries for the same arguments, labels, groups, eval arguments, rounds, keyword sets and original seed expressions: exactly one value differs. Actual constructor logs and saved native parameters are audited. Veto GBDTs receive no change, and all108 final veto model SHAs must equal archived BASE.

Original LambdaRank/ndcg, GBDT, ordinal labels0–7, implicit label_gain(first8:0,1,3,7,15,31,63,127), ranker max_depth3/num_leaves7/min_child_samples20/reg_lambda1/colsample_bytree.8, learning-rate/tree options and selected-round rules, other numerical settings, random-state streams, features, family folds, veto, raw inner objective,18-configtau/k/agreementgrid,1e-4 conservative tie/switch policy and invalid utility−2.0 remain unchanged. H338 column sampling and all other wave interventions are absent. No AUC or speed verdict.

## Native BASE check, frozen inputs and regression first

BASE is unchanged H303 NESTED/H308 P0/P1/P2 × roots30001,30002,30003 × four family-grouped outer folds:36 fold-root tasks,nine partition/root panels,1219 TRAIN contexts/205 families,three members/root,eight bags/member. CPU H302/H303/H308 commit `819a45aa5e30e662ab511cd9962b85b2b8c1e44a`; H287 export commit `213b30ad52e1ce48927bba960ed609bd98ab4dfe`. Python3.12.3/NumPy1.26.4/LightGBM4.6.0. Vendor/source/input identities retained in manifests and receipts.

Exact-version GET with complete object and selected-member SHA verification, same identities as H338:

- Bucket `veox-jopedime-learning-use2`.
- H316 key `experiments/rd_breakthrough_20261030/H316/solhwv-20261104T105633Z/h316_models_locks_outputs.tar.gz`, VersionId `W5jY6iZlwoRszA7P9zfNLLMtRtM9ZZkW`, SHA256 `4eae1dec792e1c7b4adff9ff9a5226d0c38cfe1bd9605ba07cc2596f4ad7b0b5`. BASE `run/P*/fold*/root*/fit_lock.json` and inner/final BASE rankers; NORM final refits excluded.
- H315 key `experiments/rd_breakthrough_20261030/H315/picks.npz`, VersionId `jCll6z9FGwmpNg.cocb305LcVzFYuieR`, SHA256 `e9e6b16a7381ff14191d2a4d31ce31d4b623c39297e7383ee6aeb871e08ca615`.

Before treatment, inspect native bagging_fraction/bagging_freq in all **12,636 archived BASE saved rankers** (11,664 inner main+bag instances and972 final main+bag instances). Effective settings must be **.8/1**; any difference closes **INCOMPLETE_SPEC_MISMATCH** before treatment. The completed read-only recovery/spec check already passed; native model/parameter SHAs and99 original seed-to-bagging-seed mappings are pinned in custody and manifest. No BASE refit or model scoring occurred. Retain per-model evidence; no model bytes in Git.

Before treatment, require exact equality for9/9 BASE panels,36/36 fold-root pick and lock hashes,and exact medians P0 .0643163334285782,P1 .04821480030677507,P2 .04360142642936269. Mismatch is REGRESSION_FAIL and stops. Validate current1219TRAINpool hashes,selected metadata,r882TRAINcosts and four consumed TRAIN cache prefixes against the archived BASE input manifest. Zero BASE refits/new model scores; no mixed-cache trailing feature reads or hashes.

Fit workers physically load only proper outer-TRAIN pool/cost members and declared cache byte ranges, with scoped audit hooks; identifiers/offset metadata may be read. No full inherited packet cache. Outer feature contexts and archived BASE outcomes are forbidden during fit. Per-task fit-start and every constructor UTC record must follow the published preregistration; retain minimum timestamps. All36 fit locks and parent directories fsynced before any outer model score. A fresh scoring process verifies all locks/models/archives and108 unchanged veto SHAs, then scores each fold/root once using STARTED/COMPLETE markers. No partial cell set, cut cells, extra refits, second score or budget reset. Historical TRAIN design/model exposure remains; this is not fresh independent confirmation.

## One CPU admission and two runtime forecasts

At most eight cores, candidate priority **15,19,20,34,35,36,37,55,60,62**, paired SMT siblings **79,83,84,98,99,100,101,119,124,126**. **ONE fresh five-second per-core plus sibling idle probe after CPU prep/regression**. Admit only pairs with both idle percentages≥85, using unrounded values; choose the first at most eight qualifying physical cores. Require at least four. Otherwise close **INCOMPLETE_CPU_ADMISSION**. No repeated probe, borrowed core, reduced task grid or other-owner action.

The first forecast uses H338's actual coordinator launch `2026-11-04T22:29:29.788978+00:00` and captured actual successful exit `2026-11-05T00:21:55.403507+00:00`:6745.614529 wall seconds on four assigned cores. Use **C=26982.458116 CPU-capacity seconds**, four times full wall including preflight/scoring/I/O. This is a conservative capacity upper bound, not measured optimizer-only CPU or a speed claim. After the single admission probe, with n admitted workers, forecast **C/(n×.9)×1.15+300 seconds** from then. The .9 efficiency,1.15 safety and300-second scoring/close reserve are fixed. Start all36 only if the forecast finishes strictly before04:45Z.

Forecast again immediately after the **first chronologically completed fold-root task of36**, using as_completed rather than input-order retrieval. That task records its own process+child CPU delta and wall time including preparation/archive I/O. Set remaining work to **35×max(first-task CPU delta,first-task wall seconds)**, conservatively ignoring progress in other in-flight tasks. From that timestamp apply the same **work/(n×.9)×1.15+300** formula. If it misses the fixed stop, record **INCOMPLETE_RUNTIME_FORECAST** and stop only H345's cgroup; no cut cells, re-probe, restart, reset, partial outer score or fabricated negative scientific verdict. Timing/forecast metadata never influences model selection.

## Metrics and decision

Use raw utility versus r882slot0. Report all18arm/cell records across nine partition/root panels: family-equal mean gain and literal family-median gain with whole-family95%CIs,drop-top1/5/10,LOFOminimum,W/T/L(tolerance1e-4),harmed share(gain<−.01),invalid/slot0-invalid,switches/rates,Elo/EloCI,root-pair switched-context Jaccards,chosen configs/agreement,all pick/lock hashes. Exact Elo label: **reference = r882 slot0 (DL-Train outer); r3175 not measured on TRAIN**.

Paired ROWSUB−BASE drop-top10:10,000 whole-family bootstrap draws,**RNG345191**,same draw across both arms/all9cells,each arm re-dropping its own ten best families on every draw. Report all nine differences,median,two-sided95%CI,one-sided95%LCB,and per-root medians across partitions. Retention uses the median across9cells of family-equal mean raw gain; literal family median separately reported.

**ADVANCE iff every gate passes**:

1. Paired drop-top10 median≥+.002 **and** one-sidedLCB95>0.
2. ROWSUB median full gain≥90% of BASE.
3. ROWSUB median harmed-family share≤BASE.
4. Positive per-root paired median on≥2/3roots.

Ties and margins below+.002 never advance. Otherwise **ROWSUB_NOT_ADVANCED**. Missing/late/stopped/partial results are **INCOMPLETE**, numerical verdict UNKNOWN/null, not a negative result. CONFIG_INSENSITIVE if≥30/36 configs match,else CONFIG_SENSITIVE,descriptive only. If ADVANCE,next is separately authorized one-shotdev91vsr3175; no dev91 in H345.

## Resources, cutoff and custody

xbabe2 CPU only,nice19,numeric1,systemd user MemoryMax6GiB/MemorySwapMax0,taskset4..8 admitted cores. Prep/recovery uses allowed core62; no per-core admission probe before the single declared gate. After admission both controller threads and child fits/scores use only the chosen cores. Never use H3420–14/sibs,H34321–33/sibs,H34454/56–59/61/63/16–18/sibs,H33538–53/102–117 or other jobs. Stop own compute on hostCPU>85%,load>85,MemAvailable<20GiB,scratch>2,000,000,000bytes. Existing host swap occupancy is not altered.

Own scratch `/home/ubuntu/jope-data/h345-solhwv`. Fixed compute stop **04:45ZNov5**,close **05:00ZNov5**. Outer scoring must start before04:40Z; reserve300seconds. No reset/late scoring/cut cells. Archive own inner models and final source/models/locks/results create-only under `experiments/rd_breakthrough_20261030/H345/solhwv-20261105T010000Z/`; exact-version GET+SHA before raw-copy reclamation,whole/member verification before enumerated final scratch deletion. Keep source/small receipts under `/home/ubuntu/jope-rescue-20260921/h345-solhwv`; closeout `/home/ubuntu/jope_plan/evidence/H345/solhwv-20261105-t0100/CLOSEOUT.json`.

CI_FREEZE+MASTER_MERGE_HOLD: private index/object ref rd/h345-solhwv,no new worktree(overcap68),foreign canonical HEAD/index/files preserved. Fresh-fetch/reparent under publication lock,nonforce origin/v1/next only,no PR/main/fastlane/hostedCI/Cargo/nativebuild. Publish H345 study plus one own row in each assigned index; no FINDINGS append. Red-team verifies prereg precedes first treatment,zero physical outer reads before lock,zero protected rows,per-root medians/CI,tie bar,sole constructor-value change and exact regression. DOPE `{"composite_sha256_prefix":"c51a56d0","loaded":false,"scope":"CPU selection layer only"}`. No GPU/qualification.lock,dev91/DL-Val feature/utility rows,DL-Test,sealed/holdout/meta-test/fresh cohorts,live checkpoints,xbabe0/xbabe3/xbabe1livewriter or other-owner processes/signals. DOPE weights frozen/unloaded.
