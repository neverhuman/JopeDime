# H131 — family-bootstrap adapter soup

Frozen before H131 scoring or updates under `rd_astra_t1235_h131.md`. This newer owner directive replaces the same-time H131 interpolation search: no alpha selection, H128 ANTI endpoint, or seeds94–96. H127 and earlier fine-tunes motivate testing whether averaging directions from different family resamples reduces observed harm. Neither prior failures nor this finite study establish a local optimum.

## Inputs and fixed split

Use the same pinned real-S3 H50 DL-Train regression packet: 1,219 contexts, 205 families, 257 catalog dataset IDs. The authoritative catalog is `~/jope_plan/master/source/S3_DATASET_CATALOG.md` plus its JSON; exact catalog v246 VersionId and SHA are recorded in manifest.json. Reject its14 explicit exclusions and any BeyondArena/sealed/holdout match. No dataset labels change; no new GP search or fitness evaluation. The pre-existing CPU landmark-feature computation remains part of the recipe.

Only the first `train` metadata array is decoded. No DL-Val91, existing outer-test, sealed/meta-test, holdout, live-loop or qualification.lock access. Selected metadata identity is pinned; model/pool/table bytes are admitted only for DL-Train IDs. This is a local derived corpus with real-S3 catalog admission, not a new original-row lineage qualification or verified cloud recovery map. r882 previously saw DL-Train, so the assessment measures additional fine-tuning, not unseen-family generalization.

Seeds **97,98,99**; member indices **k=1..8**, RNG seed **s*100+k**. A metadata audit of1,245 local branch/ref snapshots and243 unique configuration/index/claim files found no prior assignments; arithmetic multipliers, offsets, context counts and metric decimal matches are not seed assignments. This is bounded evidence, not global uniqueness. No replacement seeds.

Sort205 family IDs, apply NumPy `default_rng(131097).permutation`, then `array_split(...,5)`. Fold0:41 assessment families/**271 contexts**. Folds1–4:164 fit families/**948 contexts**. The complete family/context split is frozen in fold_manifest.json. Bootstrap-manifest hashes are fixed before scoring. No split selection, inner tuning or post-assessment selection occurs.

## Arms, family bootstrap and compute

Each seed has CONTROL, SOUP8 and SOUP8_HALF. CONTROL uses a unique hash-verified r882 copy and exactly40 H125 CONTROL updates on all fit contexts. For each of eight SOUP members, draw164 family IDs with replacement from the sorted164 fit-family IDs using `default_rng(s*100+k)`. Each occurrence contributes **all** contexts in that family, including repeated occurrences. Sort the expanded context multiset; sample minibatches uniformly with replacement from it. Do not deduplicate repeated families/contexts. Thus family size remains part of the original context-weighted recipe, while whole families are resampled together. Assessment families cannot enter a bag.

Each member starts independently from a new hash-verified r882 copy, uses stream seed s*100+k and40 updates. No warm start from another member or live checkpoint. All eight members are required; a failed member makes the wave INCOMPLETE, never a survivor average. After training all eight, uniformly average matching adapter parameters in member-index order using float64 accumulation, then cast once to the original float32 dtype. SOUP8_HALF is fixed `0.5*r882 +0.5*SOUP8`, computed in float64 from the saved float32 soup and cast once. This is parameter averaging, not prediction ensembling. No member/weight selection, shrinkage search or per-family assessment gate.

Only29,211 non-anchor adapter parameters change. All non-adapter parameters **and buffers** are hash-checked before/after each training/scoring model, including the anchor. DOPE must remain composite SHA `c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814`. Exact r882/source/feature pins are in manifest.json. Preserve all original checkpoints and inputs.

Recipe: AdamW lr1e-4, weight_decay1e-4, batch16, slate192, clip_grad_norm1; mean SmoothL1(beta1)+0.1 KL at temperature0.2 through unchanged H59 CONTROL loss/targets. Failure target2 and revealed-slot handling are unchanged. Eval mode with gradients enabled, matching H125. Use the first two streams per round of the H76/H97 schedule over20 rounds; generate the other six streams to preserve candidate-sampler RNG consumption. CONTROL plan must match the pinned H125 schedule exactly. Soup members use the same algorithm over the explicit repeated-context multiset. Save every actual update loss/gradient norm and final member adapter.

Budget: **3 CONTROL fits +24 member fits =27*40=1,080 optimizer updates**. SOUP8 costs320 updates per seed versus CONTROL40 (**8x**). SOUP8_HALF reuses the same soup; no additional updates. Optional CONTROL_320 is omitted. The compute difference prevents attributing any improvement solely to averaging versus compute. All27 fits and all9 final cells are required. No auto retry, budget reset, optional continuation or extra seed. ETA targets are prereg19:15UTC/results20:30UTC, not authorization to change the grid.

## Regression, frozen candidates and assessment

Before updates, independently copied unchanged-r882 models must agree on fit-pool predictions (atol1e-6, rtol0), utility (exact equality) and adapter identity. Data-free tests cover original CONTROL sampling, whole-family multiplicity, uniform averaging, strict margin/tie/all-seed/harm gates, incomplete panels and resource boundaries. Runtime optimizer guards exclude DOPE. A regression failure is INVALID; no fallback result.

Full-pool scoring may memoize the **frozen anchor only**, keyed by its complete state and every input value/shape/dtype. First reuse of every key recomputes uncached output and requires bitwise equality; before updates all fit-reference cache keys are checked. Training slates are never cached. No efficacy-dependent shortcut. Actual source and cache receipts are retained.

A read audit keeps assessment pool/table payloads closed through all27 fits and six fixed soup snapshots. Freeze the nine final candidate file hashes in a create-only S3 receipt and exact-VersionId/SHA verify it **before** admitting assessment data. No interim assessment score may alter the fixed protocol. Then independently score fresh r882 and all9 candidates with the unchanged H61 `unrev` evaluator: select among indices4..P-1; utility clip(-fitness,-2,4) for passing candidates, failure -2. Preserve per-context/per-family values. Candidate files, frozen-state hashes and DOPE identity must match before/after scoring.

## Statistics and admission

For each arm and seed, average context utilities equally within each assessment family, then equally across41 families. Primary statistic: **median across the three seed-specific equal-family mean gains versus r882**. Bootstrap41 families10,000 times, common NumPy RNG131000 draws, keeping all seeds/arms paired; recompute that median for every draw. Report two-sided percentile95% CI and ordinary one-sided95% LCB. No context bootstrap or seed-only CI.

Holm over **all three arms**, including CONTROL, family-wise alpha0.05. Use smoothed bootstrap lower-tail area `(1+count(boot<=0))/10001`; rank ascending, arm name lexicographic tie break. Ordered thresholds0.05/3,0.05/2,0.05. The rank-specific lower bound is the bootstrap quantile at that threshold; require both lower-tail area<=threshold and bound>0, with no later rejection after a failed step. Report rank, threshold, adjusted lower bound and disposition. These bootstrap areas are exploratory, not independently calibrated null-test guarantees.

**ADMIT only if** median gain>+0.001 strictly, Holm passes and its LCB95>0, every seed has positive mean gain, and harm<=10% of assessment families. Harm is preregistered conservatively as the fraction of41 families with **strictly negative seed-mean family gain**, no tolerance; also report every seed's family harm. At most4/41 harmed families pass. Exact ties, zero gain, identical-to-r882 utility vectors and incomplete panels are never admitted. Do not describe an unchanged copy as an independent confirmation.

Elo anchor r882=1500. For each seed, family W/T/L from gain signs gives p=(wins+0.5ties)/41 and Elo=1500+400log10(p/(1-p)). Report median seed Elo and paired-family-bootstrap95% interval (inverse empirical CDF, explicit infinite limits). Never AUC. Flag identical seed gain vectors; only three training seeds and one fresh assessment fold limit inference. SOUP8 versus CONTROL comparisons are descriptive, not a replacement comparator or extra selection test.

## Resource and publication contract

CPU-only xbabe2, nice>=15, <=4 worker threads, process-tree RSS<=6GiB, MemAvailable>=30GiB, load1<=86.7/102, disk from statvfs f_bavail strictly<94%. External monitor aborts the owned worker on violation; no peer process is signalled. Full swap is not safe headroom. Preserve partial results as INCOMPLETE/INVALID; never infer NO_ADVANCE from missing cells. No GPU/qualification.lock, live-loop access or deployment.

A READY prereg PR **must merge before scoring**. Source code is committed in Git, never uploaded to S3. Numerical model/score/plan artifacts and JSON custody receipts use create-only objects under H131/astra-t1235 with exact-version GET, SHA and size verification. No .pt/.npz/bulk payloads in Git. The closeout has README hypothesis/setup/outcome/metrics/verdict/next step plus red-team limitations, metrics.json<100KB and one H131 index row. Hand both PRs to central merger; do not selfmerge. If admitted, write H131_WIN.md as a candidate receipt only; do not touch the live flywheel.
