# H367 — NL15 ranker bagging_fraction0.7

**COMPLETE: BF07_NOT_ADVANCED**.
Preregistration commit: `9f2ec1b47230b751e615af6bad707c0c07fccf79`, published 2026-11-05T13:35:41.736522+00:00; first observed fit 2026-11-05T13:36:39.667655+00:00.

Frozen H366 science e6fce0d3dc, rerun only under the H367 owner authorization; executable equivalence verified before fitting.

Hypothesis: moderate row-subsampling reduction may reduce context-specific outlier and -2.0 sentinel chasing. One fixed BF07 arm, no tuning.

Sole existing ranker value subsample.8->.7 (native bagging_fraction.8->.7) vs H352 NL15, both pinned H287 functions, inner/final/all8bags; H352 NL15 leaves15/depth3/L1zero/L2one unchanged. Restoring subsample.8 restores both whole H352 effective ASTs exactly; keyword sets are identical, no bagging seed was added, and querybagging remains0. BASE regression9panels/36fold-root hashes/3medians exact; H352 NL15 regression9panels/36lock-score pairs/rawfamilyarrays exact; all1219 TRAIN input identities and972 archived native configs verified, without BASE refitting.

reference = r882 slot0 (DL-Train outer); r3175 not measured on TRAIN.

Paired drop-top10 median -0.007373, CI95 [-0.015960837455739736, -0.0026401537561946873], one-sided LCB95 -0.014544.
NL15/BF07 median family-equal mean gain +0.058405/+0.049356; harmed share 0.092683/0.082927; median Elo 1734.47/1724.08.

| Root | Paired drop10 median | Whole-family CI95 | LCB95 |
|---|---:|---|---:|
| 30001 | -0.004237 | [-0.016158768757951997, 0.006073145951193535] | -0.014086 |
| 30002 | -0.010044 | [-0.01958540494175044, -0.0002608199458480008] | -0.017643 |
| 30003 | -0.007373 | [-0.019423247306250935, 0.0003865205959007366] | -0.017660 |

Secondary paired-vs-BASE drop10: {'per_cell': [-0.006329269458259382, 0.0013003778067285452, 0.0007942340413491129, -0.0019201697310045392, -0.003672116353982121, 0.0032339875744567797, 0.002658830693495354, 0.0011099232280417145, -0.012223826767018145], 'median': 0.0007942340413491129, 'ci95': [-0.007673717426801957, 0.0047338255336282576], 'lcb95': -0.006677244806100186, 'per_root': [{'root_seed': 30001, 'median': -0.0019201697310045392, 'ci95': [-0.013348663752258233, 0.008583099016375792], 'lcb95': -0.011306025087974673}, {'root_seed': 30002, 'median': 0.0011099232280417145, 'ci95': [-0.011223060540314455, 0.012188824179887677], 'lcb95': -0.009480845478527008}, {'root_seed': 30003, 'median': 0.0007942340413491129, 'ci95': [-0.013980859855067243, 0.008318208578037719], 'lcb95': -0.01218785980342864}], 'rng': 367191, 'draws': 10000}

All36 fsynced locks preceded once-only outer scoring; 29,484 ranker native audits show exactly bagging_fraction.8->.7; 7,020 veto fits retain BASE params and108 final veto SHAs match. Same-agent stored-array audit recomputed paired10k bootstrap and locked decision exactly; no model replay. Full gain/median/CI/drop1/5/10/LOFO/W-T-L/harm/invalid/switch/root/Elo/config evidence is in [metrics.json](metrics.json).

54 data-free checks PASS. One5s probe eligible cores[27, 29, 30, 31, 32, 34]; admitted=True; nice19/math1/MemoryMax6G/MemorySwapMax0, scratch<=2GB. Fixed compute stop16:15Z/close16:30Z Nov5. Available final cgroup CPU/peak memory are retained in metrics.json; missing final accounting stays null, observed CPU lower bound is labeled separately. Corrected guard: CPU>85% averaged60s or load>108 on3 consecutive60s samples; memory/scratch/deadline guards remain. Realized per-fit bagged row counts are unavailable in native Booster dumps (diagnostic only); none were invented. No speed verdict.

Custody exact-version manifest: {'bucket': 'veox-jopedime-learning-use2', 'bytes': 493126, 'exact_version_get_sha256_verified': True, 'key': 'experiments/rd_breakthrough_20261030/H367/dsc01-20261105T133000Z/closeout/closeout-manifest.json', 'sha256': 'dea8a8bfff6a216833c84e225e645a7b4ad5da1dc1a6340db853a578e90092d3', 'uploaded_utc': '2026-11-05T14:54:47.753206+00:00', 'verified_utc': '2026-11-05T14:54:47.975667+00:00', 'version_id': 'RMpa0g4Rcf0u5QhT0jEi3wcfGEHEBIPL', 'payload_kind': 'JSON_MANIFEST', 'referenced_parts_all_members_verified': True, 'referenced_parts': 28, 'referenced_files': 2897}. Own scratch removed: True. Detailed receipts: `/home/ubuntu/jope_plan/evidence/H367/dsc01-20261105-t1330`. No models in Git.

One of a sequential single-knob sweep H348–H368 on the same folds/roots. Any ADVANCE is TRAIN-only and requires a separately authorized one-shot dev91vsr3175 confirmation, outside H367. DOPEc51a56d0 remained unloaded/unchanged, CPU selection layer only; zero DL-Val/dev91/DL-Test/sealed/holdout payload rows.

Next: retain NL15 and do not advance BF07; any next intervention needs a new owner assignment.

The live flywheel writer remains intentionally stopped at round5049; this campaign never restarts it or changes AUTORESTART_DISABLED/serving configuration. Three roots are three training seeds30001/30002/30003.
