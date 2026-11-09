# H297 — context-normalized clipped adapter loss

Primary hypothesis: context-normalized, clipped cost targets with 0.05 smoothing reduce domination by failed candidates and large-scale contexts, allowing the adapter to cross useful near-tie selection margins. Primary **CTXNORM_CLIP**, secondary descriptive **CTXNORM_CLIP_NOSMOOTH**, paired reference **CONTROL**. Three seeds: 29701, 29702, 29703. One scientific attempt; no post-score retuning, rollback within a trajectory, averaging or serving change.

## Fixed comparison and inputs

Use a private copy of r3175, SHA256 `1d6ea900ebed8d2a05c3b3f37576adc15aadd2b53c585bfd5cb71a103e9ff97b`, and t1845's immutable input/scorer layout. There are 1,219 distinct TRAIN contexts (205 families) and 91 repeatedly used development contexts (19 disjoint families). No fresh cohort, DL-Test, holdout, sealed bucket/IDs or P3-excluded input is permitted. Original/live directories and writer1402163, observer2849436 and screen flywheel_gp_fused are untouched.

Required regression results precede every new arm: r3175 replay identity `e79a7715c1531ccb8c4055ab96971101cd4034ec3a71d83bca6d25b0e604ca83`, utility exactly **1.1389572403748425**, all91 choices and utilities exact; t1845 seed184501 CONTROL after two updates exactly **1.1354387884248622**, all91 choices/utilities and both original batch-order/slate hashes exact. Numerical-thread reduction changes last-bit training losses and adapter parameter bytes, so parameter-bitwise equivalence to t1845 is not claimed; functional regression is exact. The original cached DOPE tokens are copied and hashed.

The live dev_incumbent.py has changed since t1845. Use its preserved SHA-pinned version `81c85296643ec06acc7ea31a7c7ed248abc8d4040ad28f53aa42eb4209d2f42a`, and the preserved model, adapter and native scorer bytes. The baseline scorer and native utility reduction remain unchanged. Full input hashes, dependency copies and regression receipts are in versioned custody.

## Training and pairing

Each of the nine candidates starts from the same r3175 state and incumbent AdamW moments (step487), then executes exactly eight TRAIN-only optimizer updates (through step495). Train exactly **29,211 non-anchor adapter parameters**; freeze DOPE composite `c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814`, anchor and all other state. Apply dope_freeze_guard's mode/weight lock, verify DOPE and the complete frozen-state hash before/after updates, and refuse gradients on any frozen parameter.

To hold the learning rate and perturbation scale at the t1845 seed184501 reference, use LR **3.6880929895544625e-05**, initial Gaussian sigma **2.7751362402065425e-05**, AdamW weight_decay1e-4, and one global L2 gradient clip at1.0 per update. These scales are fixed across seeds; the t1845 hash schedule is not reselected for new seed IDs. Seeds change batch order/slate randomness and noise direction. Each seed's three arms have identical initial perturbation bytes, dropout RNG reset (`seed+999`), restored optimizer state, batch orders and slates. No other loss or optimizer change.

Preserve t1845 uniform family-blocked sampling: native `training_order(batch_seed(seed,seed,step),8192,False)` shuffles complete family blocks and cycles through them, stopping before a block would exceed the limit. Consequently an update contains up to8,192 context occurrences, covering the same1,219 distinct TRAIN contexts; for the regression seed it is8,190. Duplicate occurrences are intentional. Each occurrence gets192 candidates:48 fitness elites plus NumPy-seeded remainder. Microbatch32, accumulate each microbatch mean loss weighted by occurrence count/full-step count, then clip and step once. Full batch orders and candidate indices are frozen and saved per seed, shared across arms.

Only TRAIN pools and TRAIN token dictionaries are passed to the fit routine. Dev91 receives no gradients, intermediate evaluation, early stopping or within-trajectory selection. Save/hash each final checkpoint before its sole terminal dev evaluation. Reload equality is checked on a fixed TRAIN slate, never by rescoring dev.

## Exact loss intervention

The existing code predicts **cost**: passing target is recorded fitness; failed-gate target is **+2**, corresponding to utility **-2**. Apply the requested symmetric transformation in cost space so signs remain consistent with lowest-cost argmin.

For each192-candidate context slate, compute target median and quartiles using `torch.quantile` with linear interpolation. Let `z=clip((target-median)/(Q75-Q25+0.001),-3,3)`. Primary target is `0.95*z+0.05*mean(z)` within that context, after clipping. The secondary uses `z` without smoothing. No statistics cross context boundaries. Constant contexts become zero, finite targets. Quantile and clipping operations affect targets only; **predicted costs stay in their existing coordinates** and terminal argmin scoring is unchanged. This tests literal target normalization, not a rescaling of the network output or a new serving rule.

For either transformed arm, use `SmoothL1(pred,target)+0.1*KL(softmax(-target/0.2)||softmax(-pred/0.2))`, with the same original mean/batchmean reductions. CONTROL uses the original unnormalized passing-fitness/failed-cost2 targets. Six toy tests cover quantiles/IQR, failed-cost saturation, post-clip smoothing, constant contexts, context independence/shift invariance, exact CONTROL loss and finite nonzero new-loss prediction gradients.

## Frozen decision and statistics

Primary passes **only if all three seeds** have canonical dev91 mean strictly greater than `1.1389572403748425+0.001` and strictly greater than their paired CONTROL; paired family-bootstrap lower95% bound of context-mean delta vs r3175 strictly >0; and the four largest positive context gains account for strictly <80% of total positive gain. A zero positive-gain denominator fails. Ties never win. Secondary results cannot replace the primary.

Bootstrap19 whole families with replacement,10,000 draws, seed29750. Preserve all contexts within each sampled family and use the context-weighted ratio of delta sums to context counts. Share draws across seeds and arms. Report per-seed lift/CI and the median of seed lifts with the percentile CI of the per-draw seed median; seeds are not independent datasets. Report context W/T/L (utility tie band1e-4), changed selected-program counts, and MM Bradley–Terry Elo across r3175/control/two treatments, r3175 anchored1500. Elo uses context-level pair scores, ties half, symmetric .01 pseudo-wins per directed off-diagonal pair; its interval uses the same family bootstrap. No AUC or speed verdict.

**DEV_ADVANCE** means every primary gate passes on3/3 seeds; otherwise **NOT_ADVANCED**. This development cohort has been reused for selection. Even DEV_ADVANCE requires a newly authorized fresh cohort before adoption; this wave cannot launch that cohort or modify live/serving state.

## Freeze, resources and custody

Freeze tested loss/source and `plan.json`, publish PREREG plus plan SHA to origin/v1/next before running/scoring any new arm. The driver checks the publication receipt and all source/monitor hashes. Code/plan cannot change after launch. No scientific restart, seed replacement, additional arm or retuning.

xbabe1 CPU only, CUDA hidden, nice19. Assigned60–63 are not in the host's allowed CPU set (0–50,64–114); the authorized fallback is verified-idle **100–103**. PyTorch numeric1, other numeric/Rayon pools1, aggregate monitor+worker threads<=4; RSS<=4GiB, own copies<=3GiB. Pause only own worker if MemAvailable<12GiB or hostCPU>85%; resume at>=13GiB and<=80%. Abort at hard RSS/thread/copy cap or absolute05:35Z compute stop. Prereg deadline03:40Z, close05:45Z Nov4. No GPU or qualification.lock; no xbabe0/3 or other-process control.

Pre-outcome preparation notes: the first startup exceeded four threads before any score/update and was stopped; numeric threads were reduced2→1 without widening the cap or deadline. A second preparation check incorrectly required exactly1,219 sampler occurrences, then was corrected to the preserved native family-blocked policy; no optimizer update had occurred. Its already-qualified r3175 vector was reused, without a second baseline forward pass. The actual two-update control regression ran once. All stopped-start receipts remain preserved.

Use create-only S3 objects under `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H297/astra-t0240/`, exact VersionId GET plus archive-member SHA verification. Archive inputs, frozen code/plan, full checkpoints, batch identities, training/resource receipts and outcomes; publish compact metrics<50KB. Remove only the verified private remote copy at close. Git: non-force v1/next only, own study code/tests/docs and own index rows; CI/main/fastlane hold remains in force. Append the required tagged FINDINGS close entry under its lock.
