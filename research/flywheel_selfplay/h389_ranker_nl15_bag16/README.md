# H389 — NL15 BAG16

**INCOMPLETE_CUSTODY: scientific UNKNOWN**.
Preregistration commit: `0d802a0ccdd52d723e42d766c818e27182e07af0`, published 2026-11-05T19:09:53.363117+00:00; first observed fit 2026-11-05T19:10:45.478627+00:00.

Hypothesis: doubling the bag ensemble may reduce selection variance. One fixed BAG16 arm, no tuning.

Only bagged.N_BAGS8->16 differs from H352 NL15; original index0..7 seeds preserved and indices8..15 extend the same byte-exact seed rule. All per-model constructor ASTs/native fields remain H352, including original LR/trees. Vote aggregation code and absolute thresholds5/6/7 remain unchanged; no threshold scaling, so any effect cannot be attributed to variance alone. All18configs/ties/3members retained. Fresh BASE9/36/3 and NL15ninepanels/36lock-score pairs/rawfamilyarrays exact;1219TRAINinputs and972archivednative configs/108vetoSHA verified without baseline refits.

reference = r882 slot0 (DL-Train outer); r3175 not measured on TRAIN.

The full qualified cohort did not close. Numerical outcomes remain UNKNOWN/null; observed native fits, locks, score markers, costs and interruption state are preserved. Capacity: {'eligible_cores': [50, 51, 57, 59, 60, 61], 'minimum_required': 4, 'host_cpu_at_probe': 23.429458248214484, 'probe_seconds': 5, 'probes': 1, 'retries': 0, 'admitted': True}. No retry, partial scoring verdict, or budget reset.

58 data-free checks PASS. One5s probe eligible cores[50, 51, 57, 59, 60, 61]; admitted=True; nice19/math1/MemoryMax6G/MemorySwapMax0, scratch<=2GB. Fixed compute stop23:15Z/close23:30Z Nov5. Available final cgroup CPU/peak memory are retained in metrics.json; missing final accounting stays null, observed CPU lower bound is labeled separately. Corrected guard: CPU>85% averaged60s or load>108 on3 consecutive60s samples; memory/scratch/deadline guards remain. Realized per-fit bagged row counts are unavailable in native Booster dumps (diagnostic only); none were invented. No speed verdict.

Custody exact-version manifest: None. Own scratch removed: False. Detailed receipts: `/home/ubuntu/jope_plan/evidence/H389/dsc01-20261105-t1905`. No models in Git.

One of a sequential intervention sweep H348–H389 on the same folds/roots. Any ADVANCE is TRAIN-only and requires a separately authorized one-shot dev91vsr3175 confirmation, outside H389. DOPEc51a56d0 remained unloaded/unchanged, CPU selection layer only; zero DL-Val/dev91/DL-Test/sealed/holdout payload rows.

Next: preserve UNKNOWN science and interrupted costs; no replay, reprobe, restart, or deadline reset.

The live flywheel writer remains intentionally stopped at round5049; this campaign never restarts it or changes AUTORESTART_DISABLED/serving configuration. Three roots are three training seeds30001/30002/30003.
