# H280 — RERANK8_VETO on r3175 leave-cluster-out

**ADVANCE**. Hypothesis: exact H276 RERANK8_VETO, refit on r3175 CPU top8 pools with whole-family-cluster holdouts, beats the CPU-reproduced r3175 pick. Failed gates: `[]`. Decision: `{"gates": {"cluster_lcb_every_partition": true, "harm_at_most_35_percent_every_partition_seed": true, "median_gt_0_010_every_partition": true, "placebo_fails": true, "raw_invalid_not_worse": true, "secondary_three_positive": true, "three_positive_every_partition": true}, "placebo_gates": {"cluster_lcb_every_partition": false, "harm_at_most_35_percent_every_partition_seed": true, "median_gt_0_010_every_partition": false, "raw_invalid_not_worse": false, "secondary_three_positive": false, "three_positive_every_partition": false}, "placebo_would_pass": false, "verdict": "ADVANCE"}`. No serving change.

All1219 DL-Train contexts/205families, four fixed H276 partitions, modelseeds280101–103,60outer cells and120A/placebo policies. Exact H276 topk/core/veto, fixed tau0.10, nested inner-family grid/strict selector unchanged. PLACEBO permutes ranker relevance within FIT families, leaving truthful veto labels/inner selection; it must fail its analogous qualification. REF_TRANSFER comprises three H278 frozen full-DL-Train models: **in-sample, descriptive only; assessed families were in its training set. It cannot ADVANCE.**

Primary BASE is the CPU-reproduced r3175 pick. H278-lineage H124 CPU forward on a read-only r3175 copy rebuilt all pools, with exact H278 topk568-column features. The20-context smoke projected7.4min; full TRAIN build took311.63s, so no family subsample was used. CPU DL-Val91 reproduced1.1366829009431036 with91/91 cost vectors bit-exact to H278. This differs from stored GPU mean1.1389572403748425 and is explicitly a CPU comparison, not GPU-exact qualification.

| Partition | Arm | Median gain | 95% cluster CI | Per-seed family-equal gains | Positive seeds | MM BT Elo |
|---|---|---:|---|---|---|---:|
| KM276140 | BASE | 0 | reference | 0,0,0 | 0/3 | 1500.000 |
| KM276140 | REF_TRANSFER | +0.17026630 | +0.11360607, +0.23266357 | +0.17683618, +0.13567114, +0.17026630 | 3/3 | 1659.258 |
| KM276140 | A | +0.05622418 | +0.02737584, +0.10944795 | +0.06093184, +0.05618200, +0.05622418 | 3/3 | 1544.455 |
| KM276140 | PLACEBO | +0.00000000 | +0.00000000, +0.00000000 | +0.00000000, -0.01150726, +0.00000000 | 0/3 | 1500.000 |
| KM276141 | BASE | 0 | reference | 0,0,0 | 0/3 | 1500.000 |
| KM276141 | REF_TRANSFER | +0.17026630 | +0.11611445, +0.24171122 | +0.17683618, +0.13567114, +0.17026630 | 3/3 | 1660.601 |
| KM276141 | A | +0.05426311 | +0.01995715, +0.09987059 | +0.05426311, +0.06847941, +0.04836789 | 3/3 | 1549.325 |
| KM276141 | PLACEBO | +0.00000000 | +0.00000000, +0.00000000 | +0.00000000, +0.00000000, +0.00000000 | 0/3 | 1500.000 |
| KM276142 | BASE | 0 | reference | 0,0,0 | 0/3 | 1500.000 |
| KM276142 | REF_TRANSFER | +0.17026630 | +0.11991028, +0.23334887 | +0.17683618, +0.13567114, +0.17026630 | 3/3 | 1661.796 |
| KM276142 | A | +0.05998096 | +0.03631255, +0.09692532 | +0.05922023, +0.05998096, +0.06318241 | 3/3 | 1548.044 |
| KM276142 | PLACEBO | +0.00000000 | +0.00000000, +0.00000000 | +0.00000000, -0.00785260, +0.00000000 | 0/3 | 1500.000 |
| WARD | BASE | 0 | reference | 0,0,0 | 0/3 | 1500.000 |
| WARD | REF_TRANSFER | +0.17026630 | +0.12404474, +0.22578408 | +0.17683618, +0.13567114, +0.17026630 | 3/3 | 1660.833 |
| WARD | A | +0.05972870 | +0.01776663, +0.07729662 | +0.05972870, +0.05044791, +0.06133869 | 3/3 | 1545.498 |
| WARD | PLACEBO | +0.00000000 | +0.00000000, +0.00000000 | +0.00000000, +0.00014236, +0.00000000 | 1/3 | 1500.000 |

Perseed, average contexts within family then families equally; report median across3seeds.10k whole-cluster bootstrap280150 independently within each fixed partition, preserving family multiplicity, then median of seed effects. Holm m1 A only; every partition must have3positive seeds, median>+.010, LCB>0, raw−2<=BASE and <=71harmedfamilies perseed. Also require positive DL-Val3/3 and placebo failure. Strict float epsilon1e-12; ties are not better. Five clusters, including a singleton in KM276141, limit precision. No clustering selected after outcomes.

| Partition | Arm | Raw−2 picks | Harmedfamilies perseed (loss>1e-4) | Family W/T/L perseed |
|---|---|---|---|---|
| KM276140 | BASE | 57/3657 | [0, 0, 0] | 0/205/0; 0/205/0; 0/205/0 |
| KM276140 | REF_TRANSFER | 17/3657 | [10, 16, 18] | 184/11/10; 168/21/16; 177/10/18 |
| KM276140 | A | 50/3657 | [52, 65, 47] | 136/17/52; 132/8/65; 108/50/47 |
| KM276140 | PLACEBO | 57/3657 | [0, 17, 0] | 0/205/0; 9/179/17; 0/205/0 |
| KM276141 | BASE | 57/3657 | [0, 0, 0] | 0/205/0; 0/205/0; 0/205/0 |
| KM276141 | REF_TRANSFER | 17/3657 | [10, 16, 18] | 184/11/10; 168/21/16; 177/10/18 |
| KM276141 | A | 44/3657 | [65, 54, 46] | 133/7/65; 135/16/54; 139/20/46 |
| KM276141 | PLACEBO | 57/3657 | [0, 0, 0] | 0/205/0; 0/205/0; 0/205/0 |
| KM276142 | BASE | 57/3657 | [0, 0, 0] | 0/205/0; 0/205/0; 0/205/0 |
| KM276142 | REF_TRANSFER | 17/3657 | [10, 16, 18] | 184/11/10; 168/21/16; 177/10/18 |
| KM276142 | A | 47/3657 | [57, 59, 53] | 132/16/57; 127/19/59; 140/12/53 |
| KM276142 | PLACEBO | 58/3657 | [0, 14, 0] | 0/205/0; 7/184/14; 0/205/0 |
| WARD | BASE | 57/3657 | [0, 0, 0] | 0/205/0; 0/205/0; 0/205/0 |
| WARD | REF_TRANSFER | 17/3657 | [10, 16, 18] | 184/11/10; 168/21/16; 177/10/18 |
| WARD | A | 43/3657 | [61, 60, 69] | 130/14/61; 137/8/60; 130/6/69 |
| WARD | PLACEBO | 57/3657 | [0, 1, 0] | 0/205/0; 1/203/1; 0/205/0 |

Supporting DL-Val91 is heavily reused and not independent evidence. It averages the5KM276140 outer-policy utilities percontext/seed, then families equally; this is a policy-average readout, not one deployable prediction ensemble. All60 TRAIN selections were locked before trained VAL picks. REF_TRANSFER remains descriptive.

| Arm | Median gain | 95% family CI | Per-seed gains | Positive seeds | MM BT Elo |
|---|---:|---|---|---|---:|
| REF_TRANSFER | +0.04141815 | +0.02086170, +0.07246437 | +0.06738838, +0.03260940, +0.04141815 | 3/3 | 1575.068 |
| A | +0.03438364 | +0.01957109, +0.05759742 | +0.03823200, +0.02990704, +0.03438364 | 3/3 | 1570.881 |
| PLACEBO | +0.00000000 | +0.00000000, +0.00000000 | +0.00000000, -0.00288402, +0.00000000 | 0/3 | 1500.000 |

Elo anchors BASE=1500 in the four-arm competition,400MMiterations, family-balanced pairwise outcomes and1e-4tie band. No AUC. Watch-family detail:

| Partition | Arm | fam-769243ee gains perseed |
|---|---|---|
| KM276140 | REF_TRANSFER | None |
| KM276140 | A | None |
| KM276140 | PLACEBO | None |
| KM276141 | REF_TRANSFER | None |
| KM276141 | A | None |
| KM276141 | PLACEBO | None |
| KM276142 | REF_TRANSFER | None |
| KM276142 | A | None |
| KM276142 | PLACEBO | None |
| WARD | REF_TRANSFER | None |
| WARD | A | None |
| WARD | PLACEBO | None |

Near-tie rule was fixed before outcomes: CPU remains primary; score stored picks when fixed context/program/utility alignment verifies, plus deterministic CPU runner-up sensitivity for gaps<1e-4. No contexts removed. Summary `{"file": "sensitivity-metrics.json", "sha256": "4cdb11d65ebf30e7a3e84f9f8a3f138e8a5a352b37a827e207f4c960d0c26a30", "train_near_ties": 8, "train_stored_alignment_verified": false, "val_near_ties": 3, "val_stored_alignment_verified": true}`. Descriptive A sensitivity (full arms/CI/picks in S3):

| Alternative baseline | Partition | Median gain | Per-seed gains |
|---|---|---:|---|
| near_tie_runnerup | KM276140 | +0.05609422 | +0.06080189, +0.05605204, +0.05609422 |
| near_tie_runnerup | KM276141 | +0.05413315 | +0.05413315, +0.06834945, +0.04823793 |
| near_tie_runnerup | KM276142 | +0.05985100 | +0.05909027, +0.05985100, +0.06305246 |
| near_tie_runnerup | WARD | +0.05959874 | +0.05959874, +0.05031795, +0.06120873 |
| near_tie_runnerup | DL-Val | +0.03575718 | +0.03960554, +0.03128058, +0.03575718 |
| stored | all | UNAVAILABLE | UNAVAILABLE |

Verification:16tests (8H272+8H280), exact H272 F272140-S272101-K0 receipt/count replay, Native91 91/91, exact H276 manifest, real LightGBM parent equivalence and outer-label noninterference. Saved replay: 2160 FIT X/y receipts and 156 outer/VAL/transfer models; exact folds, shuffles and panels. Statistic recomputation uses the same frozen estimator; no independent statistical implementation or full inner-grid refit is claimed. DOPE `c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814`, r3175/r882 hashes before/after match; zero neural updates, copied checkpoints only. Real S3 corpus identities/pool hashes verified; no sealed/meta-test/holdout/DL-Test or live writer/observer accesses.

Execution `398dab8fcfc6b22348c2f16c2ff791ff65d7e38c`; resource statusPASS, peakRSS760524800bytes, monitoredthreads3, CPU48,50–52/nice19/numeric1, swap-in-rate/memory/load guards. Full costs/features (`prep/train`, `prep/val`), source/folds/models/picks/sensitivities archived before compact 28518-byte metrics: `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H280/astra-t2205/closeout/runtime.tar.gz`, VersionId `DkJ67C3K.xv3hIZbERPscpQ8mziKf.1G`, SHA256 `f5b5abb9a7e806852e22b0ef5ca406bf3f60d67e6e0e82e832d4a6d6609a3868`, all471members exact-version GET verified. [metrics.json](metrics.json), [manifest.json](manifest.json), [folds.json](folds.json), [PREREG.md](PREREG.md). Same-agent exact-head review; hostedCI NOT_RUN under split/hold; nonforcev1next only.

Next step: retain the qualified CPU comparison for separately authorized deployment/independent evaluation; no serving change here.

## Deviations
Initial pre-outcome feasibility wrapper missed lazy pool path metadata; corrected before prereg, failure receipt retained. No scientific design change or outcome retry. Formatter-only pool-build source change is AST-equivalent and both source versions are preserved. Stored-pick sensitivities remain unavailable wherever context/program/utility alignment cannot be verified; primary is the predeclared CPU baseline.
