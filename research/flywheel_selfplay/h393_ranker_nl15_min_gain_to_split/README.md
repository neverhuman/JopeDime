# H393 — NL15 min_gain_to_split0.001

**COMPLETE: MINGAIN_NOT_ADVANCED**.
Preregistration commit: `ce5c48fa4a3c00e39ac82f0171ff2276038a1d40`, published 2026-11-05T20:35:12.273297+00:00; first observed fit 2026-11-05T20:36:23.050171+00:00.

Hypothesis: a small gain floor may suppress noise-chasing splits. One fixed MINGAIN arm, distinct dose0.001 from H377MSG01dose0.01; no tuning.

Sole ranker addition min_split_gain=.001, native min_gain_to_split0->0.001, versus fresh archived H352 NL15 baseline0. Both inner/final/all8bag constructor ASTs revert exactly to committed H352 when the keyword is removed. All other native fields, original LR/trees/seeds/labels/groups/veto/18configs/ties remain fixed; max_bin255/path0/minchild20/leaves15/depth3/L2one/L1zero/baggingfraction.8/featurefraction.8/freq1. Fresh BASE9/36/3 and NL15ninepanels/36lock-score pairs/rawfamilyarrays EXACT;1219TRAINinputs/972native configs/108archivedvetoSHA verified without baseline refits.

reference = r882 slot0 (DL-Train outer); r3175 not measured on TRAIN.

Paired drop-top10 median -0.002557, CI95 [-0.005907836709641494, 0.0011353722134410324], one-sided LCB95 -0.005348.
NL15/MINGAIN median family-equal mean gain +0.058405/+0.068066; harmed share 0.092683/0.102439; median Elo 1734.47/1732.44.

| Root | Paired drop10 median | Whole-family CI95 | LCB95 |
|---|---:|---|---:|
| 30001 | -0.005102 | [-0.01064577579283231, -0.0001979436542520502] | -0.009550 |
| 30002 | +0.005199 | [-0.0034475112641907143, 0.01513172630012647] | -0.002200 |
| 30003 | -0.002557 | [-0.008432849989915777, 0.0021895004169866874] | -0.007397 |

Secondary paired-vs-BASE drop10: {'per_cell': [-0.007194842408367757, 0.016943955846516374, 0.009206342403442338, 0.016971396987257754, 0.01157022315984384, 0.00803509262091537, -0.00882234119942605, 0.014094717887705024, -0.009656859536559512], 'median': 0.009206342403442338, 'ci95': [7.8290991799102e-05, 0.01576898536615929], 'lcb95': 0.0010949064602659366, 'per_root': [{'root_seed': 30001, 'median': -0.007194842408367757, 'ci95': [-0.017394584179643662, 0.011361422426652206], 'lcb95': -0.015026508307089063}, {'root_seed': 30002, 'median': 0.014094717887705024, 'ci95': [0.0027559107543276675, 0.028183760834211376], 'lcb95': 0.004322771701811268}, {'root_seed': 30003, 'median': 0.00803509262091537, 'ci95': [-0.0067718558209624155, 0.01581767401656871], 'lcb95': -0.00475799856768609}], 'rng': 393191, 'draws': 10000}

All36 fsynced locks preceded once-only outer scoring; 29,484 ranker native audits show exactly min_gain_to_split0to0.001; all other fields unchanged; 7,020 veto fits retain BASE params and108 final veto SHAs match. Same-agent stored-array audit recomputed paired10k bootstrap and locked decision exactly; no model replay. Full gain/median/CI/drop1/5/10/LOFO/W-T-L/harm/invalid/switch/root/Elo/config evidence is in [metrics.json](metrics.json).

56 data-free checks PASS. One5s probe eligible cores[50, 51, 53, 54, 55, 57, 58, 59]; admitted=True; nice19/math1/MemoryMax6G/MemorySwapMax0, scratch<=2GB. Fixed compute stop00:45Z/close01:00Z Nov6. Available final cgroup CPU/peak memory are retained in metrics.json; missing final accounting stays null, observed CPU lower bound is labeled separately. Corrected guard: CPU>85% averaged60s or load>108 on3 consecutive60s samples; memory/scratch/deadline guards remain. Realized per-fit bagged row counts are unavailable in native Booster dumps (diagnostic only); none were invented. No speed verdict.

Custody exact-version manifest: {'bucket': 'veox-jopedime-learning-use2', 'bytes': 494562, 'exact_version_get_sha256_verified': True, 'key': 'experiments/rd_breakthrough_20261030/H393/dsc01-20261105T203000Z/closeout/closeout-manifest.json', 'sha256': '51dd46ecdef61e8529d45c89e570ecf9fbf67ca849f124fc1a8ec53349a4411a', 'uploaded_utc': '2026-11-05T21:47:07.375938+00:00', 'verified_utc': '2026-11-05T21:47:07.539747+00:00', 'version_id': 'q5l0ZQfKvFlv8xPWnNpgl9cGWZeRKHUP', 'payload_kind': 'JSON_MANIFEST', 'referenced_parts_all_members_verified': True, 'referenced_parts': 28, 'referenced_files': 2906}. Own scratch removed: True. Detailed receipts: `/home/ubuntu/jope_plan/evidence/H393/dsc01-20261105-t2030`. No models in Git. H389 original CLOSEOUT remains INCOMPLETE_CUSTODY/UNKNOWN; prior scratch reclaimed only after new exactVersion/allmember recovery proof. Stopped custody preserves any runtime guard overshoot;2GB fitting guard unchanged.

One of a sequential intervention sweep H348–H393 on the same folds/roots. Any ADVANCE is TRAIN-only and requires a separately authorized one-shot dev91vsr3175 confirmation, outside H393. DOPEc51a56d0 remained unloaded/unchanged, CPU selection layer only; zero DL-Val/dev91/DL-Test/sealed/holdout payload rows.

Next: retain NL15 and do not advance MINGAIN; any next intervention needs a new owner assignment.

The live flywheel writer remains intentionally stopped at round5049; this campaign never restarts it or changes AUTORESTART_DISABLED/serving configuration. Three roots are three training seeds30001/30002/30003.
