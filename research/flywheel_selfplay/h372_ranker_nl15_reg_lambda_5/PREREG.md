# H372 — NL15 ranker reg_lambda 1 to 5

Owner sol_v1_jrb01; authority `/home/ubuntu/nudges/hw_t1400_jrb01_h372_nl15_reglambda5.md`, Nov5 2026. Reuse committed H363 comparison harness `e22036007216d20b4f7f760758b9db04de8263fb`, restoring the exact H352 NL15 baseline (`02dfee2f7e0b8543324494dd4f99d074cd6b93b6`) before applying the sole L2 treatment. Publish this preregistration and complete fitting/scoring source non-force to origin/v1/next BEFORE treatment fits. CI_FREEZE/BATCH_WINDOW: no PR/main/fastlane/hosted CI.

## Hypothesis and one fixed intervention

Stronger L2 on NL15 may damp leaf weights and improve paired drop-top10 gain over NL15. ONE treatment L2R5 changes only the existing ranker reg_lambda1.0→5.0/native lambda_l2_1→5 in pinned topk.fit_ranker and bagged.fit_ranker_n, across inner/final/all8bag rankers. No added keyword or seed parameter. Reverting reg_lambda5.0→1.0 restores both ENTIRE H352 NL15 function ASTs exactly.

Keep num_leaves15/max_depth3 (at most8 realized binary leaves), min_child_samples20/native min_data_in_leaf20, implicitreg_alpha0, Hessian.001, path_smooth0, LambdaRank/sigmoid1/ndcg/default gains/labels0..7, feature_fraction.8/bagging.8/freq1, features/groups/all seeds, every learning-rate/tree grid and adaptive choice algorithm, 18-config grid/tie1e-4/invalid sentinel−2 unchanged. Veto remains binary GBDT minchild25/L1one/L2two; all108 final veto SHAs must equal archived BASE and NL15. No tuning, calibration fit, extra arm or reference refit.

Audit every native fit and every constructor field against H352's full native configuration contract. Learning-rate/tree-count values remain adaptive selections within the unchanged H352 grids; normalize only those two corresponding native fields to the selected values. All other native defaults, including derived seeds, must match H352 exactly, except ranker lambda_l2_1→5. Compile the config-only contract from all36504 archived H352 native fit receipts; it contains no historical predictions, labels, gains or chosen configs. Historical H352 native lambda_l2 must equal1 on all archived rankers; otherwise close INCOMPLETE_SPEC_MISMATCH before treatment fits.

## Frozen references and TRAIN input closure

1219 TRAIN contexts/205 families; P0/P1/P2 × roots30001/30002/30003 × four family-grouped outer folds =36 fit tasks and9 partition/root cells. PRIMARY frozen H352 NL15; SECONDARY archived BASE. Fresh exact-VersionId GET, full-object SHA and every selected member SHA before use. Reference source/object/member pins live in committed JSON.

BASE H316 archive VersionW5jY6iZlwoRszA7P9zfNLLMtRtM9ZZkW SHA4eae1dec792e1c7b4adff9ff9a5226d0c38cfe1bd9605ba07cc2596f4ad7b0b5; H315 picks VersionjCll6z9FGwmpNg.cocb305LcVzFYuieR SHAe9e6b16a7381ff14191d2a4d31ce31d4b623c39297e7383ee6aeb871e08ca615. Require9panels/36pick+lock hashes and exact partition medians .0643163334285782/.04821480030677507/.04360142642936269. H352 NL15 exact-version archive parts in h352_reference_pins.json: require9panels/36locks/36COMPLETE score markers, exact picks/raw family arrays and median gain .0584046708039792. Mismatch→REGRESSION_FAIL. No BASE/NL15 refits or new reference model predictions.

Match current selected.json, all1219 TRAIN pool hashes, r882 TRAIN costs and four cache TRAIN feature prefixes against archived identities. Separate preparation exits before cold fitting. Fit parent loads metadata only, with no preparation/outer packet cache. Each worker physically reads only declared outer-TRAIN contexts/cache byte ranges; prohibit historical BASE/NL15 outcomes/models during fit. Record all allowed/forbidden contexts, byte/packet SHA and guard denials. Require zero fit-worker outer payload rows/packets before lock.

Fsync all36 locks plus native/constructor/physical/veto proofs before fresh cold outer scoring. Score each fold/root once using durable STARTED→COMPLETE markers. No replay, partial scoring, cut cells or reduced design. No DL-Val/dev91/DL-Test/sealed/holdout/meta_test/fresh-cohort payload. DOPE c51a56d0 untouched/unloaded, CPU selection layer only; no neural checkpoint load, GPU, qualification.lock, live writer/AUTORESTART/flywheel_gp_fused change or serving promotion.

## Statistics and locked decision

Report all27 arm/cells: family-equal mean and literal family-median gains/whole-family95%CIs, drop-top1/5/10, LOFO minimum, W/T/L, harmed(gain<−.01), invalid/switches, per-root medians and paired CIs, Elo/EloCI, config agreement and seed Jaccard. Exact Elo label: `reference = r882 slot0 (DL-Train outer); r3175 not measured on TRAIN`.

Both ordinary and paired bootstrap10000 draws/RNG372191. Share whole-family indices across arms/all9cells; each arm drops its own top10. Primary statistic median9 paired cells; CI quantiles.025/.975, one-sidedLCB95 quantile.05. Root statistics use the same draws over three partitions each. Secondary paired-vs-BASE reported separately.

ADVANCE iff primary paired drop10 median>=+.002 AND LCB95>0, L2R5 median fullgain>=.9NL15, L2R5 median harmed<=NL15, and paired median>0 on>=2of3roots. Ties/subthreshold never advance. Else L2R5_NOT_ADVANCED only after complete scoring. Incomplete outcomes unknown/null, never zeros or a scientific failure verdict. No speed verdict. Sequential exploratory sweep H348–H372 creates multiplicity/selection bias; any ADVANCE needs a separately authorized one-shot dev91 confirmation, excluded from H372. No live adoption.

## Fixed budget, guard and custody

xbabe2 CPU only, nice19/math threads1, systemd MemoryMax6G/SwapMax0/CPUQuota<=800%. ONE fresh5-second physical+SMT idle>=85% gate, first<=8/min4 passing pairs from poolA0–14+sibs64–78 ONLY; same-interval hostCPU<=70%. No other pool/reprobe/admission reset/calibration. Never touch foreign H36727–37+sibs91–101, H37054–63+sibs118–127, H37115–26+sibs79–90, H368/H369/other hosts or foreign jobs.

Initial forecast: H352 actual CPU26189.516914 /(workers*.9)*1.15+300 seconds. After FIRST completed task: remaining35*max(H352CPU/36,actualfirstCPU)/(workers*.9)*1.15+300, ongoing credit0. Miss→INCOMPLETE_RUNTIME_FORECAST; no cuts/reprobe/reset/replay. Fixed compute stop16:45Z and close17:00Z Nov5.

CORRECTED guard in every phase: stop only owned unit for MemAvailable<20GiB OR hostCPU>85% averaged over full60-second windows OR load1>108 on3 consecutive60-second samples OR own scratch>2GB, plus fixed deadline. Five-second CPU values diagnostic only; no old instantaneousCPU85/load85 stop. Fail closed on monitor error.

Own scratch only /home/ubuntu/jope-data/h372-jrb01 <=2GB; create-only S3 experiments/rd_breakthrough_20261030/H372/. To preserve room, first create-only archive the verified own historical model/review/native-audit copies, exact-version GET/full SHA/EVERYmember, then rehash/remove only those copies; retain arrays/locks/scores/input identities/config-only native contract and full recovery route. Original reference objects unchanged. At close, exact-version GET/full SHA/EVERY member of full scratch archive before own cleanup. Preserve source/vendor and receipts outside scratch. Full-key-bound local receipts prevent cross-prefix collisions. No models in Git. Preserve canonical HEAD/index/foreign files; private index/ref rd/h372-jrb01, fresh-parent nonforce v1/next, one own row each index. Claim/status/comms/CLOSEOUT bind actual source, inputs, commands, budget, tests and outcome; no automatic new study.
