# H387 — NL15 ranker LRhalf and2x trees

**INCOMPLETE_CPU_ADMISSION: scientific UNKNOWN**.
Preregistration commit: `7065b8e83661ab4496af5fc285fa15e244bc04e2`, published 2026-11-05T18:24:38.576083+00:00; first observed fit None.

Hypothesis: half learning rate with twice the boosting rounds may improve the optimization path. One fixed LRHALF arm, no tuning.

Sole ranker coupled intervention learning_rate*.5 and num_iterations*2 (exact product preserved), with matching positive ranker prediction limits vs H352 NL15, both pinned H287 functions, inner/final/all8bags; H352 NL15 leaves15/depth3/L1zero/L2one unchanged. Removing the two LR/tree expression multipliers restores both whole H352 effective ASTs exactly; original grid labels remain coordinates for seed/config identity, actual native LR/rounds are recorded; veto prediction limits unchanged; min_split_gain0; no seed keyword was added, bagging_fraction remains.8 and querybagging remains0. BASE regression9panels/36fold-root hashes/3medians exact; H352 NL15 regression9panels/36lock-score pairs/rawfamilyarrays exact; all1219 TRAIN input identities and972 archived native configs verified, without BASE refitting.

reference = r882 slot0 (DL-Train outer); r3175 not measured on TRAIN.

The full qualified cohort did not close. Numerical outcomes remain UNKNOWN/null; observed native fits, locks, score markers, costs and interruption state are preserved. Capacity: {'eligible_cores': [60, 61, 63], 'minimum_required': 4, 'host_cpu_at_probe': 27.180854460693528, 'probe_seconds': 5, 'probes': 1, 'retries': 0, 'admitted': False}. No retry, partial scoring verdict, or budget reset.

64 data-free checks PASS. One5s probe eligible cores[60, 61, 63]; admitted=False; nice19/math1/MemoryMax6G/MemorySwapMax0, scratch<=2GB. Fixed compute stop22:15Z/close22:30Z Nov5. Available final cgroup CPU/peak memory are retained in metrics.json; missing final accounting stays null, observed CPU lower bound is labeled separately. Corrected guard: CPU>85% averaged60s or load>108 on3 consecutive60s samples; memory/scratch/deadline guards remain. Realized per-fit bagged row counts are unavailable in native Booster dumps (diagnostic only); none were invented. No speed verdict.

Custody exact-version manifest: {'bucket': 'veox-jopedime-learning-use2', 'bytes': 236992, 'exact_version_get_sha256_verified': True, 'key': 'experiments/rd_breakthrough_20261030/H387/dsc01-20261105T180500Z/closeout/closeout-manifest.json', 'sha256': 'da540fe2dff7ba29cc57a3306c771fd72fc217048f3a0481c65461c9e3b978e8', 'uploaded_utc': '2026-11-05T18:26:45.420634+00:00', 'verified_utc': '2026-11-05T18:26:45.525196+00:00', 'version_id': 'NiUF15ngmZGnqnlCBxECj_JTzdT3WPVQ', 'payload_kind': 'JSON_MANIFEST', 'referenced_parts_all_members_verified': True, 'referenced_parts': 4, 'referenced_files': 1373}. Own scratch removed: True. Detailed receipts: `/home/ubuntu/jope_plan/evidence/H387/dsc01-20261105-t1805`. No models in Git.

One of a sequential intervention sweep H348–H387 on the same folds/roots. Any ADVANCE is TRAIN-only and requires a separately authorized one-shot dev91vsr3175 confirmation, outside H387. DOPEc51a56d0 remained unloaded/unchanged, CPU selection layer only; zero DL-Val/dev91/DL-Test/sealed/holdout payload rows.

Next: preserve UNKNOWN science and interrupted costs; no replay, reprobe, restart, or deadline reset.

The live flywheel writer remains intentionally stopped at round5049; this campaign never restarts it or changes AUTORESTART_DISABLED/serving configuration. Three roots are three training seeds30001/30002/30003.
