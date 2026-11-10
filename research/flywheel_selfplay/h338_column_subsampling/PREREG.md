# H338: column subsampling — preregistration

Owner sol_v1_hwv. Authority `/home/ubuntu/nudges/hw_t2225_solhwv_h338_colsub.md`, read and claimed 2026-11-04T22:23:58Z. This document is pushed nonforce to origin/v1/next before any H338 optimizer fit. The README will quote that full commit. One COLSUB arm, no tuning, no additional arm or confirmation cohort. H332 is CLOSED and is a harness template only; its clipping intervention is absent.

## Hypothesis and exact intervention

Fixed **feature_fraction_bynode=0.8** and **feature_fraction_seed=the existing fit seed** on rankers only. The existing fit seed means the exact value already supplied to that constructor's `random_state=int(random_state)`, including each main/bag call's original derived seed; it is not replaced by a root or member seed. The AST patch deep-copies that original random_state expression into the new seed keyword.

Per-node feature sampling may decorrelate trees and reduce reliance on high-spread features. BASE `colsample_bytree=0.8` (native feature_fraction) remains unchanged. The explicit new feature_fraction_seed also governs the existing tree-level feature draws: this two-keyword intervention does not isolate the per-node mask's mechanism from that seed change. No feature values, labels, weights or seed streams are rewritten. Sampling remains deterministic/reproducible under the pinned CPU environment.

Only pinned H287 `topk.fit_ranker` and `bagged.fit_ranker_n` receive the two keywords. They cover every inner main ranker, final main refit and all eight bag rankers. Stripping exactly both keywords must restore each original BASE function's exact AST. Actual constructor audits require bynode=.8, feature_fraction_seed==random_state, colsample_bytree=.8, original LambdaRank/n_jobs1 and absence of label_gain/max_delta_step/extra_trees/extra_seed overrides. Veto classifiers receive neither new keyword and must retain identical final native model SHAs to BASE (108/108).

Original GBDT, LambdaRank/ndcg, ordinal relevance labels0..7, implicit LightGBM label_gain (first8:0,1,3,7,15,31,63,127), ranker depth3/leaves7/minchild20/reglambda1/rowsubsample.8, all learning-rate/tree-count options and selected rounds, numerical settings, random_state values, folds, features, veto, conservative 1e-4 tie/switch policy, raw inner objective, 18-config tau/k/agreement grid and invalid utility−2.0 remain unchanged. No H328 linear gains, H325 extra-trees, H332 step clipping or other wave override.

## Pinned setup and regression first

BASE is unchanged H303 NESTED/H308 P0/P1/P2 × roots30001,30002,30003 × four family-grouped outer folds:36 tasks, nine panels,1219 TRAIN contexts/205 families,three members/root,eight bag rankers/member. CPU H302/H303/H308 code commit `819a45aa5e30e662ab511cd9962b85b2b8c1e44a`; H287 export commit `213b30ad52e1ce48927bba960ed609bd98ab4dfe`. Python3.12.3/NumPy1.26.4/LightGBM4.6.0; source/vendor/input identities pinned in manifest and run receipts.

Recover archived BASE read-only with exact-version GET, complete object SHA and selected-member SHA verification:

- Bucket `veox-jopedime-learning-use2`.
- H316 key `experiments/rd_breakthrough_20261030/H316/solhwv-20261104T105633Z/h316_models_locks_outputs.tar.gz`, VersionId `W5jY6iZlwoRszA7P9zfNLLMtRtM9ZZkW`, SHA256 `4eae1dec792e1c7b4adff9ff9a5226d0c38cfe1bd9605ba07cc2596f4ad7b0b5`. Original BASE `run/P*/fold*/root*/fit_lock.json`; NORM locks/refits excluded.
- H315 key `experiments/rd_breakthrough_20261030/H315/picks.npz`, VersionId `jCll6z9FGwmpNg.cocb305LcVzFYuieR`, SHA256 `e9e6b16a7381ff14191d2a4d31ce31d4b623c39297e7383ee6aeb871e08ca615`.

Before COLSUB fits: exact equality for9/9 BASE panels,36/36 fold-root pick and lock hashes, and three medians P0 .0643163334285782, P1 .04821480030677507, P2 .04360142642936269. Any mismatch is REGRESSION_FAIL and stops. Zero BASE refits/new model scores. Validate all1219 TRAIN pool hashes,selected metadata,r882 TRAIN costs and only the four consumed TRAIN feature prefixes against the archived input manifest. Never hash/read mixed-cache trailing feature rows.

Each fit worker physically loads only its proper outer-TRAIN pools/cost members and declared per-context cache byte ranges. Metadata IDs/offsets may be read. No full parent feature packet cache. Audit hooks/range checks forbid outer-feature contexts and BASE outcome access during fitting. All36 fit locks and containing directories are fsynced before any outer model score. Fresh-process scoring validates complete locks/models/archives and108 unchanged veto hashes, then scores each fold/root exactly once using STARTED/COMPLETE markers. No partial cell sets, cut cells, opportunistic refits or second scoring.

## Metrics and fixed decision

Use raw utility versus r882 slot0 for both arms. Report all18 arm/cell records across all nine partition/root cells: family-equal mean gain and literal family-median gain with whole-family95% bootstrap CIs, drop-top1/5/10, LOFO minimum, family W/T/L(tolerance1e-4), harmed-family share(gain<−.01), invalid/slot0-invalid, switches/rates,Elo/EloCI, all root-pair switched-context Jaccards, chosen configs/agreement and every pick/lock hash. No filters. Elo label is exactly **reference = r882 slot0 (DL-Train outer); r3175 not measured on TRAIN**. No AUC or speed claim.

Paired drop-top10 COLSUB−BASE:10,000 whole-family bootstrap draws, **RNG338191**; identical family draw across both arms/all nine cells, each arm re-drops its own ten best families on every draw. Report all nine differences,their median,two-sidedCI95,one-sidedLCB95 and the three per-root medians across partitions. Median gain for retention means median across9cells of family-equal mean raw gain; literal family median is separately reported.

**ADVANCE iff all four gates pass**:

1. Paired drop-top10 median ≥ +.002 **and** one-sidedLCB95 >0.
2. COLSUB median full gain ≥90% of BASE.
3. COLSUB median harmed-family share ≤BASE.
4. Positive per-root paired median on at least two of three roots.

Ties and margins below+.002 never advance. Otherwise **COLSUB_NOT_ADVANCED**. Missing/late/partial results are **INCOMPLETE**, numerical verdict UNKNOWN, never a fabricated negative result. Descriptive CONFIG_INSENSITIVE if ≥30/36 matching configs,otherwise CONFIG_SENSITIVE; not an additional advance gate. If ADVANCE,next is separately authorized one-shotdev91 vsr3175; no dev91 in H338. Historical TRAIN design exposure remains; this is not fresh independent confirmation. H328 context correction: it passed the root-consistency gate and failed only harm,despite the owner brief's shorthand.

## Resources, cutoff and custody

xbabe2 CPU only,nice19,numeric threads1,at most four assigned cores13/15/60/62; check these and SMT77/79/124/126 fresh idle≥85% before fitting. No borrowed cores or actions against another owner's processes. Initial claim-time probe found busy13/15/77/124; CPU prep/recovery uses allowed62,whose core/SMT were idle. Wait for a full qualifying four-core probe after prep. H332's complete36-task wall time was110minutes; a conservative latest fit admission is22:45Z Nov4,leaving120minutes to the immutable compute stop. If no qualifying set by then,close INCOMPLETE without fits. This does not reset any deadline or authorize a smaller grid.

systemd user MemoryMax6GiB/MemorySwapMax0; fail closed on hostCPU>85%,load>85,MemAvailable<20GiB,scratch>2,000,000,000bytes. Own scratch `/home/ubuntu/jope-data/h338-solhwv`. Fixed compute stop **00:45Z Nov5**,close **01:00Z Nov5**. Outer scoring must start before00:40Z. Guards stop only own H338 unit/cgroup. H334/H335/H339 and other lanes are off limits.

Own inner models and final source/models/locks/results archive are S3 create-only under `experiments/rd_breakthrough_20261030/H338/solhwv-20261104T222400Z/`,with exact-version GET+SHA before raw-copy reclamation and whole/member verification before enumerated final scratch deletion. No models in Git. Keep source/small receipts under `/home/ubuntu/jope-rescue-20260921/h338-solhwv`.

CI_FREEZE+MASTER_MERGE_HOLD: private index/object ref `rd/h338-solhwv`; no new worktree(overcap),foreign canonical HEAD/index/files preserved. Nonforce origin/v1/next only,rebased on fresh fetch under publication lock; no PR/main/fastlane/CI/Cargo/native build. Publish only H338 study files and one H338 row in each assigned index. Claim/status/comms coordinated under their locks. DOPE `{"composite_sha256_prefix":"c51a56d0","loaded":false,"scope":"CPU selection layer only"}`. No GPU/qualification.lock,dev91/DL-Val features or utility,DL-Test,sealed/holdout/meta-test/fresh cohorts,live checkpoints,xbabe0/xbabe3/xbabe1livewriter or other jobs. Weights stay frozen/unloaded.
