# H366 — NL15 ranker bagging_fraction0.7

**INCOMPLETE_CPU_ADMISSION: scientific UNKNOWN**.
Preregistration commit: `e6fce0d3dc6a69f42d9855830d31964db78ff232`, published 2026-11-05T12:44:32.922382+00:00; first observed fit None.

Hypothesis: moderate row-subsampling reduction may reduce context-specific outlier and -2.0 sentinel chasing. One fixed BF07 arm, no tuning.

Sole existing ranker value subsample.8->.7 (native bagging_fraction.8->.7) vs H352 NL15, both pinned H287 functions, inner/final/all8bags; H352 NL15 leaves15/depth3/L1zero/L2one unchanged. Restoring subsample.8 restores both whole H352 effective ASTs exactly; keyword sets are identical, no bagging seed was added, and querybagging remains0. BASE regression9panels/36fold-root hashes/3medians exact; H352 NL15 regression9panels/36lock-score pairs/rawfamilyarrays exact; all1219 TRAIN input identities and972 archived native configs verified, without BASE refitting.

reference = r882 slot0 (DL-Train outer); r3175 not measured on TRAIN.

Admission outcome: only three eligible cores27/36/37, below minimum4; same-interval hostCPU 54.95840925295097% was within70%. One5second probe; no retry. Zero treatment fits, zero locks, zero outer scores. Scientific metrics/verdict remain UNKNOWN/null. This is a capacity rejection, not a numerical negative. All comparator preparation/input proofs and actual preparation costs are preserved.

54 data-free checks PASS. One5s probe found only3 eligible cores27,36,37; no study workers were admitted; nice19/math1/MemoryMax6G/MemorySwapMax0, scratch<=2GB. Fixed compute stop16:30Z/close16:45Z Nov5. Available final cgroup CPU/peak memory are retained in metrics.json; missing final accounting stays null, observed CPU lower bound is labeled separately. Corrected guard: CPU>85% averaged60s or load>108 on3 consecutive60s samples; memory/scratch/deadline guards remain. Realized per-fit bagged row counts are unavailable in native Booster dumps (diagnostic only); none were invented. No speed verdict.

Custody exact-version manifest: {'bucket': 'veox-jopedime-learning-use2', 'bytes': 235753, 'exact_version_get_sha256_verified': True, 'key': 'experiments/rd_breakthrough_20261030/H366/dsc01-20261105T123000Z/closeout/closeout-manifest.json', 'sha256': '36cbe3a1e215c1bbfcc75976703e0bb8db1eb99c395a3ac370a54bf43951aafd', 'uploaded_utc': '2026-11-05T12:49:22.798310+00:00', 'verified_utc': '2026-11-05T12:49:22.892534+00:00', 'version_id': 'DeUQKdMc9hVAaR3aKq9edInNwzr_mSUK', 'payload_kind': 'JSON_MANIFEST', 'referenced_parts_all_members_verified': True, 'referenced_parts': 4, 'referenced_files': 1364}. Own scratch removed: True. Detailed receipts: `/home/ubuntu/jope_plan/evidence/H366/dsc01-20261105-t1230`. No models in Git.

One of a sequential single-knob sweep H348–H366 on the same folds/roots. Any ADVANCE is TRAIN-only and requires a separately authorized one-shot dev91vsr3175 confirmation, outside H366. DOPEc51a56d0 remained unloaded/unchanged, CPU selection layer only; zero DL-Val/dev91/DL-Test/sealed/holdout payload rows.

Next: preserve UNKNOWN science and interrupted costs; no replay, reprobe, restart, or deadline reset.

The live flywheel writer remains intentionally stopped at round5049; this campaign never restarts it or changes AUTORESTART_DISABLED/serving configuration. Three roots are three training seeds30001/30002/30003.
