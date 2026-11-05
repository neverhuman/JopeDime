# H375 — NL15 ranker min_split_gain0.01

**INCOMPLETE_CPU_ADMISSION: scientific UNKNOWN**.
Preregistration commit: `438fae60d9cfb0dd26926c761f48fbc2127d8646`, published 2026-11-05T15:37:23.187420+00:00; first observed fit None.

Hypothesis: a split-gain floor may prune low-value splits that chase context-specific -2.0 noise. One fixed MSG01 arm, no tuning.

Sole ranker addition min_split_gain=.01 (native min_gain_to_split0->.01) vs H352 NL15, both pinned H287 functions, inner/final/all8bags; H352 NL15 leaves15/depth3/L1zero/L2one unchanged. Removing only min_split_gain restores both whole H352 effective ASTs exactly; no seed keyword was added, bagging_fraction remains.8 and querybagging remains0. BASE regression9panels/36fold-root hashes/3medians exact; H352 NL15 regression9panels/36lock-score pairs/rawfamilyarrays exact; all1219 TRAIN input identities and972 archived native configs verified, without BASE refitting.

reference = r882 slot0 (DL-Train outer); r3175 not measured on TRAIN.

The full qualified cohort did not close. Numerical outcomes remain UNKNOWN/null; observed native fits, locks, score markers, costs and interruption state are preserved. Capacity: {'eligible_cores': [27, 33], 'minimum_required': 4, 'host_cpu_at_probe': 29.76905790287363, 'probe_seconds': 5, 'probes': 1, 'retries': 0, 'admitted': False}. No retry, partial scoring verdict, or budget reset.

55 data-free checks PASS. One5s probe eligible cores[27, 33]; admitted=False; nice19/math1/MemoryMax6G/MemorySwapMax0, scratch<=2GB. Fixed compute stop18:00Z/close18:15Z Nov5. Available final cgroup CPU/peak memory are retained in metrics.json; missing final accounting stays null, observed CPU lower bound is labeled separately. Corrected guard: CPU>85% averaged60s or load>108 on3 consecutive60s samples; memory/scratch/deadline guards remain. Realized per-fit bagged row counts are unavailable in native Booster dumps (diagnostic only); none were invented. No speed verdict.

Custody exact-version manifest: {'bucket': 'veox-jopedime-learning-use2', 'bytes': 236355, 'exact_version_get_sha256_verified': True, 'key': 'experiments/rd_breakthrough_20261030/H375/dsc01-20261105T145600Z/closeout/closeout-manifest.json', 'sha256': 'b41ec74c54c5ba173e3c05860f9fbcd48d56e8e11d6db1865dbf74f0f1603a17', 'uploaded_utc': '2026-11-05T15:41:07.759705+00:00', 'verified_utc': '2026-11-05T15:41:07.867942+00:00', 'version_id': '4QKxlKMy99wm7q0NTV6SNioaY0IQ1cxB', 'payload_kind': 'JSON_MANIFEST', 'referenced_parts_all_members_verified': True, 'referenced_parts': 4, 'referenced_files': 1368}. Own scratch removed: True. Detailed receipts: `/home/ubuntu/jope_plan/evidence/H375/dsc01-20261105-t1456`. No models in Git.

One of a sequential single-knob sweep H348–H375 on the same folds/roots. Any ADVANCE is TRAIN-only and requires a separately authorized one-shot dev91vsr3175 confirmation, outside H375. DOPEc51a56d0 remained unloaded/unchanged, CPU selection layer only; zero DL-Val/dev91/DL-Test/sealed/holdout payload rows.

Admission detail: primary27/33 eligible, fallback16/17/18/19/22 none;2<minimum4, hostCPU29.769%<=70. Zero treatment fits, locks and outer scores; scientific outcome remains UNKNOWN, not a numerical negative.

Next: preserve UNKNOWN science and interrupted costs; no replay, reprobe, restart, or deadline reset.

The live flywheel writer remains intentionally stopped at round5049; this campaign never restarts it or changes AUTORESTART_DISABLED/serving configuration. Three roots are three training seeds30001/30002/30003.
