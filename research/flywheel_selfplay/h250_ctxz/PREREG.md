# H250 — per-context standardized loss and family-balanced sampling

Status: pre-outcome preregistration. Authority: `~/nudges/hw_t1131_astra_h250.md` (2026-11-03 11:31Z). Owner: rd_astra [ASTRA]. No primary outcome, model update or utility evaluation precedes the prereg merge. The label-only epsilon calibration below is explicitly required before prereg and performs zero model forwards.

## Hypothesis and arms

Reducing context-scale dependence of the loss and balancing sampled families may improve adapter continuation from the offline r882 copy. This tests fixed candidate pools, not live candidate generation. All arms use exact H170 AdamW lr=1e-4, betas=(0.9,0.999), weight_decay=1e-4, global gradient clip=1, 40 updates/cell and batches of 8 contexts ×24 candidates. Every arm/cell starts from r882 and a fresh optimizer. No warm-start from another arm or cell. Only the 47 adapter tensors / 29,211 coordinates are trainable; DOPE `c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814`, anchor and all non-adapter state stay frozen.

- **CTXZ_FB (primary):** standardized loss with family-balanced sampling.
- **CTXZ (secondary):** the same standardized loss with the original H170 data order.
- **CONTROL:** exact inherited H146/H170 smooth-L1 plus 0.1 ListNet KL at temperature 0.2; valid native costs clamped to [-4,2], invalid cost +2, original reductions and data order.

Primary CTXZ_FB combines two changes; CTXZ isolates normalization relative to CONTROL. No claim that the combined primary alone identifies the causal contribution of sampling.

## Frozen normalization and failure rule

Native utility is negative passing fitness, with −2 on invalid gate. Thus the failure cost is **+2**, not −2. For CTXZ/CTXZ_FB training targets only, replace invalid-gate costs by **max finite valid native cost in that context's full FIT pool +1**, before any label statistics. Valid labels, including a valid cost +2, stay unchanged. No finite valid label means fail closed. Record invalid-label counts in calibration and every optimizer trace. Raw pools and all utility/evaluation labels remain unchanged.

For each outer split, compute population standard deviation (`ddof=0`, float64) of these corrected costs over each full FIT-context pool. Freeze **epsilon=0.1×median of those FIT-context standard deviations**, shared across all three optimizer seeds for that outer split. Zero/nonfinite median fails closed; no fallback epsilon. [normalization.json](normalization.json) binds the exact values, FIT IDs, pools, fold/checkpoint identity and all 45 arm-cell batch-plan hashes. Calibration reads only the union of optimization FIT rows (all TRAIN rows occur in some FIT set); it does not score SELECT, outer ASSESS, or DL-Val utility, and makes no model forward calls.

Within each sampled **24-candidate** training slate, separately calculate prediction mean/std and corrected-target mean/std, population convention. Set `z=(x-mean)/max(std,epsilon)`. Prediction mean/std remain in autograd; labels and target statistics are detached. Both loss terms operate in normalized coordinates:

`mean_context( mean_candidate smooth_l1(z_pred,z_true) + 0.1 * sum_candidate KL(softmax(-z_true/0.2) || softmax(-z_pred/0.2)) )`.

The coefficient, temperature and KL reduction match CTRL; its operands are explicitly the normalized scores and labels in these two arms. No raw-score auxiliary KL and no extra family loss weights. Shift invariance holds; positive-scale invariance holds when the epsilon floor is inactive. Fixed epsilon deliberately limits invariance at tiny spreads. Calibration uses full pools while optimization normalizes sampled slates; neither the full-pool argmin utility nor the underlying data is normalized.

## Sampling, folds and selection

205 TRAIN families / 1,219 contexts; 91 DL-Val contexts /19 disjoint families. One new five-way family partition, seed **250140**, shared across seeds **250101–250103**. Outer fold A supplies 41 ASSESS families, next fold B supplies 41 SELECT families, and the remaining 123 families supply optimization FIT. All family sets are disjoint. [fold_manifest.json](fold_manifest.json) fixes identities.

CONTROL and CTXZ retain the inherited H156/H170 context order: eight uniformly chosen distinct FIT contexts, top-fitness four candidates as reveals plus twenty sampled from the remainder; stream_seed(seed,step). CTXZ_FB draws **eight distinct FIT families uniformly without replacement**, then one context uniformly within each family. Each family has equal marginal probability. Its stream is SHA256-derived `stream_seed(seed,outer,"CTXZ_FB",step)`. Draw families first, contexts second, then top-four plus twenty candidate samples. Sort family and within-family context IDs before random indexing. No retries/rebalancing or outcome-dependent draws. Model stochastic stream uses that same batch stream. Both plans retain the inherited label-sorted four-reveal training convention; evaluation uses the native first-four indices. This existing train/evaluation mismatch is not repaired here.

15 cells ×3 arms ×40 updates = **1,800 primary updates**. First require Native91 to match all reference values exactly, mean **1.1377941898164525**, with native/cached forward identity. Then perform one bounded smoke update for each arm on the same first original-order FIT batch, check finite gradients, changed adapter state and frozen DOPE, and restore r882 exactly. Those **3 smoke updates are additional**; primary optimizers restart fresh and smoke outcomes never select a policy.

Save snapshots at 0/10/20/30/40. SELECT uses equal-family native utility: earliest numerically tied best snapshot (tolerance 1e-12), accepted only if gain >0.001+1e-12 **and** 10,000-draw paired-family bootstrap LCB95 >0; otherwise step0 r882. Fsync the three-arm decision JSON and parent directory for each cell **before any trained-policy outer ASSESS or DL-Val assessment for that cell**. This is the inherited per-cell lock protocol, not a global 15-cell lock. Score each unique selected/best/40 snapshot once per cell/arm/panel; reuse step0 baselines. Save raw SELECT/ASSESS costs, choices, gates, utility, snapshot hashes, update traces, sampler hashes and lock hashes. No post-assessment reselection.

## Metrics, admission and reporting

DL-Val91 is primary: within each seed average the five selected cell-policy utilities per context, then average contexts within family and families equally. This is utility averaging, not a prediction ensemble. TRAIN strict OOF is secondary. Use 10,000 paired-family resamples (seed250150), median gain across the three seeds per draw, percentile CI95, and one-sided `(1+count(bootstrap median<=0))/10001`. Holm m=2 covers **CTXZ_FB−r882 and CTXZ−r882**. Each treatment advances only if median gain >0.002+1e-12, family LCB95>0, Holm p<0.05 and ≥2/3 positive seeds. Ties fail. Report primary/secondary verdicts separately; the wave verdict is ADVANCE only if at least one of these two registered Holm contrasts passes. Exact CONTROL contrasts are descriptive.

Report all seeds, strict DL-Val and TRAIN OOF, unconstrained step40 and SELECT-best panels, adapter L2 drift, harm-family identities/counts (loss>1e-4), and CTXZ_FB−CONTROL / CTXZ−CONTROL. Elo uses canonical Bradley–Terry 400 iterations, prior0.01, r882=1500 and 1,000 paired-family bootstrap draws. Context **#27 (zero based)** remains in every panel and gets separate strict/step40 per-seed and per-cell readouts.

Report descriptive context-mean differences from plateau1.1389572 and archived-only pairwise Elo against the matched historical r3175 utility vector pinned in `manifest.reporting_reference` before prereg. No new r3175 inference or CPU qualification; H226 parity failure and historical winner-selection bias remain. The historical vector is postprocessing-only, outside the worker's input closure.

Three seeds and five folds reuse the same exposed development families and fallback baseline. Bootstrap intervals condition on selected policies; baseline-only seeds can pin a median CI at zero without establishing equivalence. Fresh folds do not erase historical r882/model/adapter selection exposure. No AUC, speed, sealed-test or serving-promotion claim.

## Inputs, resources and lifecycle

Real S3 origin from the authoritative S3_DATASET_CATALOG and inherited hash/version receipts; 276 admitted dataset IDs. Reject has_meta_test rows, wrong split/task/source split, duplicate contexts and overlapping TRAIN/DL-Val families. Never open sealed/holdout/meta-test payloads, live flywheel paths, GPU, xbabe0 or xbabe3. Exact-path read-only allowlist includes own schedules, manifest and normalization files; protected paths and writes to input/allowlisted protocol files remain refused.

xbabe2 CPU48–51, nice19, one numeric thread, aggregate threads≤4, RSS<4GiB, free disk≥300GB. Admit RAM≥30GiB/load5≤100. Cooperatively pause between units/updates at RAM<20GiB or load5>108; resume RAM≥25GiB sustained30s and load5≤100. Hard stop RAM<12GiB sustained30s, RAM<6GiB instantly, RSS/disk/thread violation, or absolute **2026-11-03 15:00Z**. One primary process, zero process restarts/resumes. Source and calibration outputs are outside data-space under owned custody. Other owners' jobs are untouched.

Prereg merged by **12:45Z**, before Native91/smoke/primary outcomes. Closeout by **15:45Z**; incomplete work is CLOSED_INCOMPLETE with unknown outcomes preserved. Required local tests/Ruff/schema/diff/jope-records and exact-head same-agent review; owner explicitly authorizes admin squash and waives remote CI as a gate. Full remote CI is NOT_RUN, and targeted checks do not repair the separately failed repository-wide native audit. PR2071 is excluded from this lane.

Source execution is the actual prereg merge commit, bound in launch.json with review ID, manifest/protocol/fold/normalization hashes. Create-only S3 custody under `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H250/astra-t1131/`, exact VersionId GET and archive-member hashes verified. Terminal audit reconstructs saved choices/labels, all sampler plans, selectors, adapter snapshots, update traces, panels and resource samples without model inference. Readout, both indexes and append-only FINDINGS land in closeout; reclaim owned bulk only after verified recovery.
