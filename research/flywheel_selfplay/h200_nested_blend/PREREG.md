# H200 — nested lambda selection and rank-space GBM blend

**Owner:** luna_ns_hbaseline. **Authority:** `~/nudges/hw_t1350_h200_hbaseline.md`.
**Claim/progress:** `~/jope_plan/ns/receipts/H200_claim.txt`. **Lane:** `rd/h200-prereg`.
This prospective, zero-update diagnostic cannot change a checkpoint, serving release, or live flywheel.

## Question and arms

H129/H195 found a promising fixed λ=0.25 blend of the r882 cost vector and a calibrated HistGBM prediction, but λ was selected post hoc. H200 asks whether FIT-only, family-grouped selection is robust, and whether rank blending works without calibration. The five arms are R882, FIX025, NEST_L (primary), RANK025, and NEST_SHUF. No model weights or optimizer state are updated; DOPE remains frozen at c51a56d0.

## Cohort and read boundary

Use only the already-pinned real-S3 flywheel catalog cohort from H139/H195: 1,219 regression DL-Train contexts in 205 families and the 91 regression DL-Val contexts in 19 families. Catalog authority is v246, pinned by the markdown SHA256 `bb3a1d20d3495b599f8dc392a12b858d03cdcdafa540531c5e5a4b94c250e8ee` and machine catalog SHA256 `e5f732fd784bf67bfc16f4173c585b0861e6e8a6026808df857097a721bbf0ba`. Reuse the H195 staging inputs at `~/jope-data/h195/inputs`, whose underlying source objects and byte digests are pinned in the merged H124/H139/H195 lineage. Validate the selected roster, per-file hashes and split labels before loading outcomes. DL-Val is opened only after the exact 91/91 r882 native regression check; it is not used to choose λ or assess treatments. No binary, holdout, sealed, meta-test, or live-loop data is eligible. If the pinned roster is incomplete, has changed identities, or fails the split/protected-row checks, abort before opening outcome pools.

The runner checks the hostname is exactly xbabe1 as its first runtime action. On any other hostname, it raises before opening any file. No xbabe0, xbabe2, xbabe3, GPU, or live writes.

## Seeds and folds

Experimental seeds are 200101, 200102, 200103. Their independent fold seeds are respectively 200141, 200142, 200143. Candidate sample seed is 200140; family bootstrap seed is 200150 (Elo bootstrap uses the deterministic stream 200151). These values were exact-boundary audited as unused in prior experiment code/manifests. Sort the 205 training family IDs, permute with NumPy default_rng(fold_seed), split into five folds. The generated fold manifest contains only TRAIN context/family identities and fold counts. For each outer fold k, A=k, B=(k+1) mod 5, and FIT is the other three folds. Each A fold is scored once per seed.

## HistGBM and nested calibration

Use the pinned H124/H195 recipe: HistGradientBoostingRegressor, max_iter=300, learning_rate=0.08, max_leaf_nodes=63, l2_regularization=1.0, random_state=experimental seed; 1,089 features (program 412 + lm48_z + DOPE 629); 192 candidates sampled per FIT context. Targets are utility centered within context. GBM fits never include A or B.

For each outer cell, produce out-of-fold FIT predictions with inner models excluding {A,B,j}, one per FIT fold j. To assess lambda on inner fold j, fit its calibration only from the other two FIT folds: each calibration prediction is produced by a GBM trained on the opposite one of those folds, excluding A, B, j, and the fold being predicted. Thus neither the inner model nor its calibrator sees j outcomes. Apply H124's registered positive-slope calibration to the raw prediction using gate-valid candidates at slots 4 onward. Evaluate all eight lambdas {0, .1, .15, .2, .25, .3, .4, .5} by native first-argmin utility on the held-out j contexts, averaged equally by family across the three inner folds. Select the lambda with the largest mean gain over R882 only when that gain exceeds lambda=0's mean by strictly more than 0.001; otherwise choose 0. Exact ties choose the smaller lambda. Ties at the 0.001 boundary are not gains.

For the outer A score, fit the final positive-slope calibration using the three FIT-only cross-fitted predictions (from models excluding {A,B,j}) and corresponding FIT labels. Apply the outer model excluding {A,B} and the selected fold lambda. FIX025 uses the same outer models/calibration with lambda 0.25. NEST_SHUF repeats the full inner selection and outer blend after deterministic permutation of each context's calibrated GBM candidate scores, preserving each context's score multiset. The permutation RNG is SHA256-derived from H200 seed, stage, fold and context. RANK025 uses raw GBM predictions without calibration: within each context convert lower-is-better r882 and GBM costs to average-tie ranks normalized to [0,1], then choose the native first argmin of 0.75*rank(r882)+0.25*rank(GBM).

The fit plan is fixed at 20 HistGBM fits per experimental seed: five outer models (exclude adjacent {A,B}); ten unique two-fold inner models (exclude {A,B,j}); and five one-fold calibrator models (train on one fold, predict on another). Total: 60 fits. No early stopping, retries, or resume. One attempt only.

## Endpoint and decision

Primary endpoint is H139's native flywheel utility: first argmin over the full candidate pool; utility is negative fitness if gate-valid, otherwise -2.0. For each arm, pool the three seed panels over A (205 families); average contexts within each family, then report each seed's equal-family mean gain, the median over seeds, a shared 10,000-draw family bootstrap 95% CI of the median, one-sided bootstrap p against +0.001, and harm fraction (families whose three-seed mean gain is below -0.01). Report Bradley–Terry Elo against R882, selected lambdas per outer fold, FIX025 and NEST_SHUF contrasts, and all ties. Holm correction has m=2 over NEST_L and RANK025. NEST_L is a candidate only if its CI lower bound is >0, Holm-adjusted one-sided p<0.05, and harm fraction <=10%. Any exact tie is not a gain. Otherwise NO_ADVANCE. No adoption is authorized.

## Guards and artifacts

One xbabe1 CPU attempt, launched from xbabe2 over SSH after preregistration merge and baseline verification. Nice 19; OMP <=4 and measured worker-tree CPU <=4 cores; MemAvailable >=16 GiB; free disk >=40 GB; aggregate RSS <6 GB; load <=85% of online cores; 4-hour wall cap; output under 500,000,000 bytes at `~/jope-data/h200/attempt-001`. The monitor aborts on a violated guard and never resumes. No active H199 process or path may be inspected or changed.

Before any arm, run the exact 91/91 DL-Val r882 native utility regression and require array equality and mean exactly 1.1377941898164525. A mismatch ends the attempt before TRAIN outcome pools or any treatment arm.

The preregistration PR includes the protocol, source/input manifest, data-free tests, host guard, outcome-free fold-generation algorithm and index row. Run focused tests, Ruff, diff check, and the five jope-records gates. Self-review and squash-merge at the exact tested head without waiting for remote CI. Only then launch. On completion, write README, metrics.json (<100 KB), fold manifest, and result index row; verify the S3 create-only archive and exact-version readback; submit a separate small closeout PR and self squash-merge. Keep metrics and source receipts under the assigned paths. Progress is appended to H200_claim.txt at least every ten minutes.
