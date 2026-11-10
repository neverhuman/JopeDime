# H290 r3175-native BAG_SWITCH transfer

Authority: hw_t0030_astra_h290.md. One scientific attempt, CPU only. Hypothesis: refitting the unchanged H279 BAG_SWITCH science on r3175 CPU top-8 DL-Train pools improves transfer relative to frozen r882-fit H284 BAG_SWITCH. No live-writer changes or direct adoption.

## Training and freeze

Search Git and S3 H280–H289 for a previously frozen full-DL-Train r3175-native export. H280 outer-fold models are not a full-training export. If none exists, restore H280's exact 1,219-context / 205-family CPU packets (568 features). Reproduce CPU DL-Val91 baseline 1.1366829009431036 and H280 cell KM276140_280101_0 including its selection, picks and model/gate hashes. This explicitly authorized regression uses the old DL-Val packet only; no H286/H283 evaluation packet may open before all fits are locked.

Use byte-identical H279 bagged.py, H278 topk.py, H265 h265_core.py and H273 veto.py. For seeds 290101/290102/290103, use H284's unchanged full-training family-fold construction (four inner folds, empty outer fold replaced only by one training-row scoring sentinel). Run H279 run_cell_bagged with K=8, B=8, fixed q=6, tau=0.10 and unchanged strict H265 selection/grid. Any hyperparameter selection uses only grouped inner DL-Train folds. The nested-q calculation inside the inherited implementation is retained but its q is not used by H290.

PLACEBO is H279's inherited within-family shuffled-relevance proposer and eight shuffled voters, sharing the truthful training-only veto and selected hyperparameters; its selection uses H279's raw margin, as in the original placebo. This is a conditional label-permutation control, not a fully label-blind training pipeline. No training selection uses H286/H283. Save all models, input/label fit hashes, family folds, receipts and serializations. Verify reloaded predictions bit-exactly on all training contexts. Write and fsync the fit-lock hash, archive it, then publish the complete prereg on origin/v1/next before any cohort scoring.

## Evaluation scope and arms

PRIMARY: H286 cohort v2, 200 EVEN families, cohort SHA256 0de1b3839cb3201d6c8a05aafec134e28e119ae6c233588b41d6169adb3e7b11. SECONDARY: H283 EVEN 60 requested, **29 built/31 unbuilt** in the published archive. This structural coverage clarification was recorded after fit-lock but before scoring; the original training protocol is retained as [FIT_PROTOCOL.md](FIT_PROTOCOL.md). Score all 29 existing secondary pools and report 29/60 coverage, without substituting zeros or constructing additional pools. Primary must retain all 200 families. Use their recorded GPU-exact r3175 top-8 features/cost ordering and corresponding recorded pool labels, preserving all built contexts. Verify source archive VersionIds/hashes, order/feature identity and the original GPU baseline receipt before scoring. Neither cohort is used in fitting. They were previously opened by earlier waves: the evidence is **SEMI_FRESH**, not independent fresh qualification. Do not open ODD, fresh_cohort_v3, reserved remainder, sealed, DL-Test or holdout payloads.

Six arms: BASE=slot0; NATIVE=new frozen BAG_SWITCH; R882_FIT=H284 frozen BAG_SWITCH seeds279101–103 matched in order to290101–103; PLACEBO; NULL_RV=random valid slot (uses validity labels); NULL_LF=random slot among all eight (no labels). For each cohort and seed, each null uses its own stream core.stream_seed(seed, 'H290', cohort_name, arm_name), consumes contexts in sorted context-id order and samples uniformly. NULL_RV falls back to slot0 only if no valid slot exists. Null draws are not tuned or retried. Frozen learned picks use features only; outcomes are joined after picks are saved.

No neural forwards are planned: existing packets suffice. CPU-train versus GPU-test drift is a limitation. Report paired feature/pick differences only where both computations already exist for the same contexts and model. Without a matched CPU/GPU packet pair, mark that fraction NOT_MEASURABLE; do not present cross-cohort distribution differences as a numerical CPU/GPU error rate or create new forwards to estimate it.

## Statistics

Average context utility gains within each family, then weight families equally. Utilities are recorded -fitness for valid programs and -2 for invalid selections. Report each seed's family-equal mean gain and the median across three model seeds; also family-median gain, family W/T/L with a ±1e-4 tie band, one-sided exact binomial sign test P(Binomial(wins+losses,.5)>=wins), harmed-family count (gain < -0.01), raw utility==-2 count and separately actual gate-invalid counts. Ties never count as wins; when all families tie, p=1.

10,000 paired family bootstrap draws with seed290150, resampling whole families. The same family-index draws are used across arms and seeds within a cohort. Report each seed's percentile 95% CI and a pooled median-over-seeds statistic/CI (take the median of the three resampled seed means per draw). This is seed aggregation on one cohort, not three independent datasets. Report paired NATIVE-minus-R882_FIT means/median and CIs with the same draws. Secondary results never select models or change the verdict.

MM Bradley–Terry Elo: six-arm pairwise context outcomes, family-averaged, tie band1e-4 and half-weight ties, 400 MM iterations and prior0.01 as inherited H279/H276 analysis, BASE anchored1500. Report each seed and the median; 10k family bootstrap Elo CI with the same draws. No AUC or speed verdict.

## Fixed verdict

The primary transfer gate for an arm requires: median family-equal mean gain >0.010; LCB95>0 on all3seeds; one-sided sign-test p<0.05 on all3; each seed's harmed-family count <= its matched R882_FIT count; each seed's raw -2 count <= BASE. PLACEBO, NULL_RV and NULL_LF must each fail this same transfer gate. This null-failure test omits the additional superiority contrast, making it conservative against claiming learned transfer when a null already passes the transfer gate.

NATIVE_TRANSFER_PASS iff NATIVE passes the transfer gate, every null/control fails, and the median paired NATIVE-minus-R882_FIT gain >0. If NATIVE passes its transfer gate and controls fail but the contrast is not positive, NATIVE_NOT_BETTER. Otherwise NOT_TRANSFERRED_SEMI_FRESH. Failures or missing required primary coverage/inputs yield INCOMPLETE without a favorable scientific verdict; the pre-existing 29/60 secondary coverage is reported separately. Even PASS is SHADOW ONLY and earns a separately authorized GPU-exact confirmation on reserved fresh EVEN families, never direct adoption. No optional arm or changed threshold after outcomes.

## Resources, custody and publication

xbabe2 CPU48,50–52, nice19, one numeric thread, aggregate<=4, RSS<=4GiB; abort MemAvailable<10GiB. No GPU or qualification.lock, no writer/observer changes, no xbabe0/3. Prereg by01:30Z, owned compute stop03:40Z, close03:45Z Nov4. One attempt; interruption does not authorize an outcome retry. Versioned create-only S3 evidence under experiments/rd_breakthrough_20261030/H290/astra-t0030/, exact-version readback/hash verification before compact Git metrics<50KB. Non-force origin/v1/next only; no main/fastlane or hosted CI. Both own index rows GIT_COMMITTED. Claim and close coordination per owner.
