# H377 — NL15 ranker min_split_gain0.01

**COMPLETE: MSG01_NOT_ADVANCED**.
Preregistration commit: `264196c02e19cf40a875c2983b0cd8de71118d96`, published 2026-11-05T16:04:54.593592+00:00; first observed fit 2026-11-05T16:06:45.597964+00:00.

Hypothesis: a split-gain floor may prune low-value splits that chase context-specific -2.0 noise. One fixed MSG01 arm, no tuning.

Sole ranker addition min_split_gain=.01 (native min_gain_to_split0->.01) vs H352 NL15, both pinned H287 functions, inner/final/all8bags; H352 NL15 leaves15/depth3/L1zero/L2one unchanged. Removing only min_split_gain restores both whole H352 effective ASTs exactly; no seed keyword was added, bagging_fraction remains.8 and querybagging remains0. BASE regression9panels/36fold-root hashes/3medians exact; H352 NL15 regression9panels/36lock-score pairs/rawfamilyarrays exact; all1219 TRAIN input identities and972 archived native configs verified, without BASE refitting.

reference = r882 slot0 (DL-Train outer); r3175 not measured on TRAIN.

Paired drop-top10 median -0.003815, CI95 [-0.008187626219823255, 0.003282263597280657], one-sided LCB95 -0.007263.
NL15/MSG01 median family-equal mean gain +0.058405/+0.061718; harmed share 0.092683/0.112195; median Elo 1734.47/1710.56.

| Root | Paired drop10 median | Whole-family CI95 | LCB95 |
|---|---:|---|---:|
| 30001 | -0.003882 | [-0.010612457516586885, 0.0058968190076864125] | -0.009335 |
| 30002 | -0.001186 | [-0.010637949815926155, 0.00722792770482667] | -0.009279 |
| 30003 | -0.003855 | [-0.014290202266377361, 0.004694463293747252] | -0.012764 |

Secondary paired-vs-BASE drop10: {'per_cell': [-0.005974606700939872, 0.0005466161729524213, -0.010114811616657228, 0.012938704810959534, 0.005185656990737027, 0.00577983798839655, 0.0044626630678504305, 0.014254757384488947, -0.0009582116622012977], 'median': 0.0044626630678504305, 'ci95': [-0.0035635148605551352, 0.010598356558765893], 'lcb95': -0.0024660074925056475, 'per_root': [{'root_seed': 30001, 'median': 0.0044626630678504305, 'ci95': [-0.008335814661102994, 0.018367459104106684], 'lcb95': -0.005908354194456388}, {'root_seed': 30002, 'median': 0.005185656990737027, 'ci95': [-0.0032845379299940346, 0.016552087141072536], 'lcb95': -0.0017597316164550101}, {'root_seed': 30003, 'median': -0.0009582116622012977, 'ci95': [-0.010507574905591687, 0.008435445354373022], 'lcb95': -0.009127441985563454}], 'rng': 377191, 'draws': 10000}

All36 fsynced locks preceded once-only outer scoring; 29,484 ranker native audits show exactly min_gain_to_split0->.01; 7,020 veto fits retain BASE params and108 final veto SHAs match. Same-agent stored-array audit recomputed paired10k bootstrap and locked decision exactly; no model replay. Full gain/median/CI/drop1/5/10/LOFO/W-T-L/harm/invalid/switch/root/Elo/config evidence is in [metrics.json](metrics.json).

55 data-free checks PASS. One5s probe eligible cores[27, 29, 30, 31, 32, 33, 34, 35]; admitted=True; nice19/math1/MemoryMax6G/MemorySwapMax0, scratch<=2GB. Fixed compute stop18:45Z/close19:00Z Nov5. Available final cgroup CPU/peak memory are retained in metrics.json; missing final accounting stays null, observed CPU lower bound is labeled separately. Corrected guard: CPU>85% averaged60s or load>108 on3 consecutive60s samples; memory/scratch/deadline guards remain. Realized per-fit bagged row counts are unavailable in native Booster dumps (diagnostic only); none were invented. No speed verdict.

Custody exact-version manifest: {'bucket': 'veox-jopedime-learning-use2', 'bytes': 493166, 'exact_version_get_sha256_verified': True, 'key': 'experiments/rd_breakthrough_20261030/H377/dsc01-20261105T155300Z/closeout/closeout-manifest.json', 'sha256': 'c5a24ff71251a522c4e81a8485619e5026aad6142b531e67f0db6582c8394aa8', 'uploaded_utc': '2026-11-05T17:22:39.001605+00:00', 'verified_utc': '2026-11-05T17:22:39.276106+00:00', 'version_id': '84gANi526oJpeoZk8UlgL49meRzKkvo1', 'payload_kind': 'JSON_MANIFEST', 'referenced_parts_all_members_verified': True, 'referenced_parts': 28, 'referenced_files': 2897}. Own scratch removed: True. Detailed receipts: `/home/ubuntu/jope_plan/evidence/H377/dsc01-20261105-t1553`. No models in Git.

One of a sequential single-knob sweep H348–H377 on the same folds/roots. Any ADVANCE is TRAIN-only and requires a separately authorized one-shot dev91vsr3175 confirmation, outside H377. DOPEc51a56d0 remained unloaded/unchanged, CPU selection layer only; zero DL-Val/dev91/DL-Test/sealed/holdout payload rows.

Next: retain NL15 and do not advance MSG01; any next intervention needs a new owner assignment.

The live flywheel writer remains intentionally stopped at round5049; this campaign never restarts it or changes AUTORESTART_DISABLED/serving configuration. Three roots are three training seeds30001/30002/30003.
