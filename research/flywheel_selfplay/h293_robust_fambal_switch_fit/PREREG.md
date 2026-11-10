# H293 — robust family-balanced switch fitting

Owner-directed hypothesis: family-balanced fitting plus within-context normalized rank targets improves the DL-Train family median and harm profile while retaining at least 80% of CLIP's family-equal gain. Primary: **FAMBAL_RANKNORM**. This is one scientific attempt, three model seeds 293101–293103, 205 DL-Train families / 1,219 contexts. It is an internal robustness study, not fresh generalization or adoption evidence.

## Input closure and regression

Source reference: H287 commit `213b30ad52e1ce48927bba960ed609bd98ab4dfe`. Use its pinned real-data cached r882 costs, top-eight candidate order, 568-dimensional feature construction, outcomes and labels. The H287 CLIP seed 287101 regression reproduces all 1,219 held-out utilities bit exactly, with family-equal gain **0.09355105605170722**. The reference is its exact-version, SHA-verified `inner_cv_panel.npz`; see manifest and regression receipt. No checkpoint loads, neural forwards or optimizer updates. No DL-Val, DL-Test, holdout, sealed/P3-excluded, H286 EVENv2, fresh v3/v4 or ODD-reserve evaluation payloads. H290 evaluation artifacts are excluded.

Use the H287 four-fold family splits, partition seeds 279101–279103 paired respectively with model seeds 293101–293103. Every family stays wholly in one fold. H287's published CLIP configurations (learning rate, tree count, strict score margin, veto threshold .1, eight family-bootstrap bags and q=6) remain fixed. These configurations were historically selected on all DL-Train, so this reuses previously selected hyperparameters; it is not a wholly untouched outer-test estimate. No hyperparameter is reselected on H293 outer outcomes.

## Exact fit interventions

Code-level fact: H287 `clip_packets` floors utilities at -1 for configuration selection but **does not change `rel_k`**, the LambdaRank target. H293 CLIP faithfully retains those ordinal relevance labels and default exponential label gains. Calling a transformed utility array a new target while retaining `rel_k` would be a no-op; the following target encodings instead enter the fitted objective.

- **CLIP:** H287 published configuration and original LambdaRank labels/gains, fresh declared model seed.
- **FAMBAL:** CLIP plus candidate-row sample weight `1/(8 * number_of_context_occurrences_in_family)`. Each represented family has exactly total weight one, including duplicate occurrences in a family-bootstrap bag. This also means bootstrap multiplicity is normalized away for weighted arms; bag membership still varies. Veto fitting is unchanged across arms.
- **RANKNORM:** rank the eight raw candidate utilities within each context using average ranks for exact ties. Integer labels are twice the zero-based midrank (0…14); explicit linear label gains `label/14` implement normalized rank targets in LambdaRank. Extreme magnitudes cannot change these targets once ordering is fixed. Invalid candidates retain the recorded -2 utility; the unchanged learned veto applies at prediction time.
- **HUBER:** Huber-score-clipped utility targets `psi_delta(u-u_slot0)=clip(u-u_slot0,-delta,+delta)`. Shift/scale these to [0,1], round to the nearest of 1,025 integer labels (NumPy ties-to-even), and use linear gains `label/1024`. This is Huber winsorization of targets within LambdaRank, **not a Huber regression loss**. Delta grid {.01,.1,1} is selected separately inside each outer training set by two-fold family-grouped inner-inner CV using the complete fixed veto/proposer/eight-bag switch policy. Select greatest equal-family mean raw utility gain; exact ties choose the smaller delta. No outer family participates in delta selection.
- **FAMBAL_RANKNORM (PRIMARY):** combine the FAMBAL weights and RANKNORM encoding.
- **PLACEBO:** same weights and rank encoding as primary, then deterministically permute candidate target values across all candidate rows within each training family. Permutation is restricted to the relevant training fold and shared across its bootstrap bags; family target multisets stay intact. Veto labels remain unchanged.
- **NULL_LF:** no fitted utility model. For each seed/fold, uniformly select exactly as many contexts as the frozen primary switches, then choose a uniform slot from 1…7. Uses primary switch counts, never utilities or target labels. Matching occurs only after model lock and does not refit any arm.

All tree structure, regularization, subsampling, numeric thread count and policy margins otherwise match H287. Fixed score margins may interact with the target transformation; there is no post-outcome recalibration. All arms share the same per-seed/fold bag draws and veto. All models fit serially; no speculative scientific reruns.

## Lock and prediction boundary

Fit all outer-training models and three all-DL-Train primary exports before scoring any H293 outer fold. Export prediction checks compare actual in-memory boosters with disk reloads on the first at most 64 **training** contexts, bit exactly. Inner-inner scoring for delta selection and the explicitly required H287 regression are the only pre-lock validation predictions. Freeze model files, hashes, fold membership, selected deltas, input hashes and source hashes in `fit_lock.json`; archive exact versions to S3. Publish its hash and this prereg on origin/v1/next before H293 outer scoring. Evaluation refuses absent publication or mismatched hashes.

## Statistics and decision

For each seed, compute per-family mean utilities, then equal-family gain relative to slot0; report family-median gain, 10%-trimmed family mean (drop floor(.1*205) from each tail), exact one-sided binomial sign test on non-tied families, harmed families with loss strictly >.01, and exact raw -2 picks. Family sign/Elo ties use |difference|<=1e-4 and never count as wins. Means and the owner's strictly-positive family-median gate use unrounded utility differences.

Paired nonparametric bootstrap: 10,000 family-resampling draws from seed 293150. Share the same family draws across every arm and model seed. Report each seed's mean-gain percentile CI and the median of three seed means with a percentile CI computed by taking the three-seed median within each joint family draw. Model seeds are not independent datasets.

Concentration is the H289 statistic: signed sum of the four context gains with largest **absolute magnitudes** divided by signed total context gain, using equal context weights. Stable input order breaks magnitude ties. Values may exceed one when other contexts lose, or be negative when the selected contributions are net losses. Nonpositive denominators are undefined and cannot satisfy the improvement gate. Compare the median of three seed shares. Also retain raw context arrays so the equal-family mean can be independently reconstructed.

Elo: MM Bradley–Terry fit on pairwise family-mean utility comparisons across all eight arms, ties half a point, anchor slot0=1500, symmetric .01 pseudo-wins per directed off-diagonal pair for finite estimates. Report per-seed and seed-median Elo with the same family-bootstrap CIs. No AUC or speed verdict.

**ROBUST_ADVANCE** requires primary: median mean gain >=.8*CLIP median; LCB95>0 on 3/3 seeds; family-median gain>0 on >=2/3; median harmed-family count strictly below CLIP; median top-four share strictly below CLIP; raw -2 count <= slot0 for every seed; and both PLACEBO and NULL_LF fail this same numerical gate against CLIP. Otherwise **ROBUST_NOT_ADVANCED**. Secondary arms remain descriptive. Neither verdict authorizes adoption. ADVANCE would earn a later fresh GPU-exact test only. Export the frozen primary and exact-version S3 spec with bit-exact reload receipts either way.

## Budget and custody

CPU48,50–52, nice19, numeric threads1, aggregate threads<=4, RSS<=4GiB; abort on MemAvailable<10GiB. No GPU, no qualification.lock, no xbabe0/3; writer1402163 and observer2849436 untouched. Prereg deadline02:20Z, stop owned compute04:10Z, close04:15Z Nov4. S3 create-only exact-version GET/SHA-verified objects under `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H293/astra-t0125/`. Compact metrics<50KB. Non-force origin/v1/next only, own study and two own index rows; main/fastlane/hosted CI remain held. Versioned full evidence precedes compact closeout. Unique source and receipts remain in custody; verified disposable owned bulk is reclaimed after a fresh consumer check.

Frozen model lock SHA256: `712a1e3ea6360ffe09774220573292d411c1495a5bcbd938d91ee1b95a8a73c6`. Fit completed at epoch 1793756851.9264867; all H293 outer scores unopened at publication. Full versioned source/model/input manifests are linked from metrics.json.
