# H360 — native ranker bagging_by_query

**COMPLETE: NL15BBQ_NOT_ADVANCED**.
Preregistration commit: `f0e0a5d280c16d805867f0b48f5c738b54b8e257`, published 2026-11-05T11:05:04.449588+00:00; first observed fit 2026-11-05T11:06:18.186382+00:00.

Sole ranker addition bagging_by_query=True (native0->1) vs H352 NL15, both pinned H287 functions, inner/final/all8bags; H352 NL15 leaves15/depth3/L1zero/L2one unchanged. Removing the keyword restores both whole H352 effective ASTs. BASE regression9panels/36fold-root hashes/3medians exact; H352 NL15 regression9panels/36lock-score pairs/rawfamilyarrays exact; all1219 TRAIN input identities and972 archived native configs verified, without BASE refitting.

reference = r882 slot0 (DL-Train outer); r3175 not measured on TRAIN.

Paired drop-top10 median -0.002703, CI95 [-0.013895726187504426, 0.0036181254363265216], one-sided LCB95 -0.012254.
NL15/NL15BBQ median family-equal mean gain +0.058405/+0.054653; harmed share 0.092683/0.112195; median Elo 1734.47/1710.45.

| Root | Paired drop10 median | Whole-family CI95 | LCB95 |
|---|---:|---|---:|
| 30001 | -0.000104 | [-0.013823476875096113, 0.009861009934523905] | -0.011811 |
| 30002 | -0.002892 | [-0.021913607846636676, 0.008371239807701715] | -0.019701 |
| 30003 | -0.002703 | [-0.01741652656948286, 0.005558130456501582] | -0.015175 |

Secondary paired-vs-BASE drop10: {'per_cell': [-0.002196721532540717, 0.004143243435784297, -0.012060277321752886, 0.009870242853297389, -0.009594627770257751, 0.006932107560368441, -0.001390543155820356, 0.012236754985715824, -0.0025967618418077584], 'median': -0.001390543155820356, 'ci95': [-0.005740871535649597, 0.009712114685861015], 'lcb95': -0.004717275929330419, 'per_root': [{'root_seed': 30001, 'median': -0.001390543155820356, 'ci95': [-0.0219095585465708, 0.017032110946854754], 'lcb95': -0.01711319736833777}, {'root_seed': 30002, 'median': 0.004143243435784297, 'ci95': [-0.007437828750147761, 0.013904685969740685], 'lcb95': -0.005488210681029251}, {'root_seed': 30003, 'median': -0.0025967618418077584, 'ci95': [-0.012393008416829733, 0.008505986136570507], 'lcb95': -0.01082576058836462}], 'rng': 360191, 'draws': 10000}

All36 fsynced locks preceded once-only outer scoring; 29,484 ranker native audits show exactly query-bagging0->1; 7,020 veto fits retain BASE params and108 final veto SHAs match. Same-agent stored-array audit recomputed paired10k bootstrap and locked decision exactly; no model replay. Full gain/median/CI/drop1/5/10/LOFO/W-T-L/harm/invalid/switch/root/Elo/config evidence is in [metrics.json](metrics.json).

48 data-free checks PASS. One5s probe admitted6 physical cores27,29,30,32,35,37; nice19/math1/MemoryMax6G/MemorySwapMax0, scratch<=2GB. Fixed compute stop13:45Z/close14:00Z Nov5. Available final cgroup CPU/peak memory are retained in metrics.json; missing final accounting stays null, observed CPU lower bound is labeled separately. Corrected guard: CPU>85% averaged60s or load>108 on3 consecutive60s samples; memory/scratch/deadline guards remain. Realized per-fit bagged query counts are unavailable in native Booster dumps (diagnostic only); none were invented. No speed verdict.

Custody exact-version manifest: {'bucket': 'veox-jopedime-learning-use2', 'bytes': 492483, 'exact_version_get_sha256_verified': True, 'key': 'experiments/rd_breakthrough_20261030/H360/dsc01-20261105T110000Z/closeout/closeout-manifest.json', 'sha256': '3e56f6bb0cbbbcee43a8a0c0edf2af6861996bcf930c44c5fc31e02b6c9aadfc', 'uploaded_utc': '2026-11-05T12:24:38.017586+00:00', 'verified_utc': '2026-11-05T12:24:38.161189+00:00', 'version_id': 'rtnVY9FMP0BcYt55NhIhsaqhuG5E53TH', 'payload_kind': 'JSON_MANIFEST', 'referenced_parts_all_members_verified': True, 'referenced_parts': 28, 'referenced_files': 2892}. Own scratch removed: True. Detailed receipts: `/home/ubuntu/jope_plan/evidence/H360/dsc01-20261105-t1100`. No models in Git.

One of a sequential single-knob sweep H348–H361 on the same folds/roots. Any ADVANCE is TRAIN-only and requires a separately authorized one-shot dev91vsr3175 confirmation, outside H360. DOPEc51a56d0 remained unloaded/unchanged, CPU selection layer only; zero DL-Val/dev91/DL-Test/sealed/holdout payload rows.
