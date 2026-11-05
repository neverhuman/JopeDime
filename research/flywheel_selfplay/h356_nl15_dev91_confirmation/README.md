# H356 — frozen H352 NL15 confirmation on dev91

**NL15_DEV91_CONFIRMED**. All nine frozen H352 policies were evaluated once on exactly 91 regression validation contexts from 19 families. No model was fitted or updated.
Primary vs r3175: median Elo **1658.92**, CI95 [1500.00, 1778.90], one-sided Elo LCB95 1526.74; paired family-equal utility lift **+0.016227**, CI95 [+0.005464, +0.029163], LCB95 **+0.006810**.

The frozen confirmation gates were median Elo>1500 and paired utility-lift LCB95>0. Both passed. The two-sided Elo interval touches the 1500 reference; its one-sided LCB is above1500. This is a reused 19-family development panel, not fresh-population or serving qualification.

## Hypothesis and frozen setup

The TRAIN H352 num_leaves7→15 treatment may transfer to dev91 against the incumbent. H352 source/artifacts are pinned at `02dfee2f7e0b8543324494dd4f99d074cd6b93b6`. Rankers remain LambdaRank, max_depth3, actual leaves≤8, reg_lambda1, implicit reg_alpha0, min_child_samples20; all other options and the veto remain unchanged. All 1,080 final model hashes (972 rankers/108 veto models) and 36 fit-lock hashes matched H352. There was no silent full-TRAIN refit or repeated outer-cell training.

H352 defines nine dependent policies (P0/P1/P2 × roots30001–30003), each using the original family hash to route to one of four frozen fold models. H356 preserved that routing, all TRAIN-chosen tau/k/a settings, margins, iteration counts, three-member consensus and eight bags per member. No fold averaging, dev policy selection, hyperparameter tuning, or new candidate view was introduced. The original r882 top-eight feature view produced one set of picks; these same picks were compared with both fixed references.

Preregistration `83219744233a5c2f9bad039ec2272c665980d487` published 2026-11-05T10:03:49.838273+00:00, before inference 2026-11-05T10:08:06.808974+00:00; see [PREREG.md](PREREG.md), [input_pins.json](input_pins.json), and [metrics.json](metrics.json). All executed source bytes match preregistration. 24 data-free controls and new-code Ruff check/format passed. Hosted/full repository CI was NOT_RUN under CI_FREEZE+BATCH_WINDOW; no PR/main/fastlane action.

## Family-grouped confirmation

| Same frozen NL15 picks | r3175 reference | r882 reference |
|---|---:|---:|
| Median family-equal utility lift | +0.016227 | +0.017341 |
| Whole-family lift CI95 | [+0.005464, +0.029163] | [+0.006921, +0.029947] |
| One-sided lift LCB95 | +0.006810 | +0.008083 |
| Literal family-median lift | +0.000000 | +0.003487 |
| Literal median CI95 | [+0.000000, +0.012976] | [+0.000000, +0.008567] |
| Median Elo (reference1500) | 1658.92 | 1725.29 |
| Elo CI95 | [1500.00, 1778.90] | [1620.12, 1898.45] |

**Primary Elo label:** reference = r3175 slot0 (dev91). **Secondary:** reference = r882 slot0 (dev91). These reference-relative Elo scales differ from H352 TRAIN Elo.

Utility is raw gate?-fitness:-2, without clipping. Context differences are averaged within each family; all19 families have equal weight. The pooled statistic is the median over all nine policies. Bootstrap10,000/RNG356191 resamples whole families, sharing each draw across all policies and both references. Elo uses the pinned H302 two-player MM Bradley–Terry point calculation and original bootstrap formula, anchored at1500. W/T/L tolerance1e-4; ties never count as wins. Seeds/partitions are dependent policies, not independent replications.

The inherited point estimate uses 400 MM iterations; its inherited bootstrap uses the smoothed closed form. For the zero-loss r882 P0/root30001 secondary cell, the finite-iteration point (2499.50) lies below its bootstrap interval [2541.26, 2731.82]. This numerical inconsistency is retained and disclosed. A closed-form diagnostic matches all primary point estimates and both pooled medians within 1e-8; it changes no frozen metric or confirmation gate. Detailed diagnostic receipt: `ELO-NUMERICAL-DIAGNOSTIC.json` in the local evidence directory.

| Partition | Root | Reference | Family-equal lift [CI95] | Elo [CI95] | W/T/L families | Harmed share | Invalid |
|---|---:|---|---:|---:|---|---:|---:|
| P0 | 30001 | r3175 | +0.012598 [+0.003458, +0.023825] | 1859.72 [1588.51, 2731.82] | 8/10/1 | 0.052632 | 0 |
| P0 | 30002 | r3175 | +0.004891 [-0.000878, +0.011688] | 1531.61 [1291.25, 1778.90] | 6/8/5 | 0.105263 | 0 |
| P0 | 30003 | r3175 | +0.016227 [+0.002220, +0.031409] | 1708.75 [1523.17, 2716.71] | 10/6/3 | 0.157895 | 0 |
| P1 | 30001 | r3175 | +0.019212 [+0.002404, +0.039774] | 1549.90 [1359.37, 1778.90] | 8/5/6 | 0.157895 | 0 |
| P1 | 30002 | r3175 | +0.011647 [-0.001637, +0.029951] | 1476.83 [1282.48, 1675.46] | 7/4/8 | 0.210526 | 0 |
| P1 | 30003 | r3175 | +0.026147 [+0.009503, +0.045985] | 1658.92 [1473.26, 1943.98] | 10/5/4 | 0.000000 | 0 |
| P2 | 30001 | r3175 | +0.016433 [+0.000865, +0.036489] | 1526.74 [1324.54, 1754.28] | 7/6/6 | 0.210526 | 0 |
| P2 | 30002 | r3175 | +0.013998 [-0.000599, +0.030169] | 1658.92 [1473.26, 1943.98] | 10/5/4 | 0.157895 | 0 |
| P2 | 30003 | r3175 | +0.019566 [+0.002469, +0.038336] | 1740.39 [1549.90, 2746.03] | 12/4/3 | 0.052632 | 0 |
| P0 | 30001 | r882 | +0.013712 [+0.004287, +0.025203] | 2499.50 [2541.26, 2731.82] | 8/11/0 | 0.000000 | 0 |
| P0 | 30002 | r882 | +0.006005 [+0.000741, +0.012446] | 1646.86 [1411.49, 2661.45] | 7/9/3 | 0.000000 | 0 |
| P0 | 30003 | r882 | +0.017341 [+0.003384, +0.032239] | 1914.99 [1670.03, 2770.55] | 11/7/1 | 0.052632 | 0 |
| P1 | 30001 | r882 | +0.020326 [+0.003695, +0.040661] | 1658.92 [1473.26, 1956.85] | 10/5/4 | 0.105263 | 0 |
| P1 | 30002 | r882 | +0.012761 [-0.000288, +0.030615] | 1549.90 [1359.37, 1778.90] | 8/5/6 | 0.105263 | 0 |
| P1 | 30003 | r882 | +0.027261 [+0.010609, +0.046977] | 1725.29 [1531.61, 2731.82] | 11/5/3 | 0.000000 | 0 |
| P2 | 30001 | r882 | +0.017547 [+0.002375, +0.037223] | 1597.03 [1379.88, 1898.45] | 7/8/4 | 0.105263 | 0 |
| P2 | 30002 | r882 | +0.015112 [+0.000179, +0.031146] | 1725.29 [1538.72, 2732.16] | 11/5/3 | 0.157895 | 0 |
| P2 | 30003 | r882 | +0.020680 [+0.003633, +0.039686] | 1824.43 [1620.24, 2781.76] | 13/4/2 | 0.052632 | 0 |

| Root across partitions | Reference | Median lift [CI95] | Lift LCB95 | Median Elo [CI95] |
|---:|---|---:|---:|---:|
| 30001 | r3175 | +0.016433 [+0.004060, +0.034708] | +0.005650 | 1549.90 [1418.48, 1809.82] |
| 30002 | r3175 | +0.011647 [+0.001694, +0.018021] | +0.002711 | 1531.61 [1402.97, 1708.75] |
| 30003 | r3175 | +0.019566 [+0.009589, +0.037598] | +0.011226 | 1708.75 [1570.34, 1943.98] |
| 30001 | r882 | +0.017547 [+0.005475, +0.035339] | +0.006879 | 1658.92 [1531.49, 1968.82] |
| 30002 | r882 | +0.012761 [+0.003262, +0.018955] | +0.004124 | 1646.86 [1500.00, 1824.43] |
| 30003 | r882 | +0.020680 [+0.010944, +0.038469] | +0.012403 | 1824.43 [1690.46, 2731.82] |

## Source, inputs, and custody

The 91 immutable contexts/19 dataset IDs join real catalogv246 and exclude the catalog sealed/BeyondArena IDs; TRAIN families are disjoint. Source split metadata remains train75/inner16. Only this authorized dev91 slice was decoded; no other DL-Val/sealed/holdout/meta-test/fresh-cohort/DL-Test payload rows or raw dataset files. Historical dev91 reuse remains a limitation; semantic/historical independence is not inferred from disjoint family IDs.

Existing H352/H144/H281 versions were GET-verified by whole-object and selected-member SHA. Original S3 versions for three retained derivatives (pool priors, r882 costs and r3175 reference JSON) were not established: their existing H287/H310 hashes were preserved through explicitly labelled create-only H356 copies and exact-version GET/hash before inference. No newly discovered original-version provenance is claimed. Opaque archive checksum transport is separate from row decoding; neural checkpoint entries and unselected pools were never extracted/decoded.

Mixed H58 caches were read with exactly364 dev-only os.pread ranges (four members per context), decoding zero TRAIN feature rows or other DL-Val rows. Models/source/inputs were locked before prediction. Exclusive inference/group markers prevent replay. Program/features were built first; picks and raw predictions were fsynced before fitness/gate/reference outcome decoding. All36 stored-score groups replayed to exactly the saved picks without another Booster prediction. Both reference regressions matched91/91 stored context utilities and exact means r3175=1.1389572403748425/r882=1.1377941898164525. Same-agent audit recomputed family arithmetic, bootstrap and decision exactly; this is not an independent review.

Resources: xbabe2 CPU only, admitted cores [0, 1, 2, 4, 5, 6, 7, 8], one sequential inference worker, numeric1/nice19/MemoryMax6GiB/Swap0. One5-second physical+SMT85%-idle admission; hostCPU 14.989107%≤70. Study CPU 31.587723s, peak memory 0.708GiB, swap0. Observed hostCPU≤18.587%, load≤29.125, available memory≥72.871GiB, scratch≤324449557bytes. Fixed compute12:15Z/close12:30Z Nov5; no extension/reprobe/restart. No speed claim.

Create-only archive: `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H356/jrb01-20261105T093500Z/closeout/closeout-manifest.json`, VersionId `sFBMp0xwA533G7G43WLtI17kTXYwK3.t`, SHA256 `d1a77bd307fa9b8c8097708d246b196a18ed4d3ace6dbaca881b7c7cc5b2ab2f`. All1309files in3parts passed exact-version GET, whole-object SHA and every-member verification before removing only H356 scratch. Source, detailed receipts and read-only replay proof remain in `/home/ubuntu/jope_plan/evidence/H356/jrb01-20261105-t0935`. No models in Git.

DOPE `c51a56d0` was not loaded or changed. No GPU, qualification.lock, live flywheel writer, foreign process, or other host was touched.

Acceptance controls after reconstructing the archived source layout in authorized scratch: `CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 19 python3 -B -m unittest discover -s /home/ubuntu/jope-data/h356-jrb01/source -p "test_h356.py" -v`. Controls do not fit or score study models.

Next: deliver the confirmed result and exact artifact contract to the flywheel owner as `/home/ubuntu/nudges/H356_NL15_DEV91_ADOPT_SAFE.md`. The handoff is a nudge only. No live adoption, new fitting, selected single-policy deployment, GPU work, or next scientific axis was performed or authorized by this execution.
