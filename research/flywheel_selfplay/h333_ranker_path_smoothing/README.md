# H333: ranker path smoothing

**Closed: INCOMPLETE_RUNTIME_FORECAST.** Only assigned core 22 passed the required five-second per-core and SMT idle gate. The complete 36-task forecast misses the fixed 22:45Z compute stop. No H333 optimizer fits, fixtures, BASE refits, fit locks or outer scores were run. The numerical verdict is unknown; this is not PSMOOTH_NOT_ADVANCED.

## Hypothesis and planned setup

Hierarchical shrinkage of small leaves toward parent values may reduce concentration of gains in a few high-residual families. The one fixed intervention adds only `path_smooth=20.0` to every inner, final and eight-bag ranker, using unchanged GBDT. The value is fixed by the owner, equal to BASE min_child_samples20; no outcome selected it. BASE leaves path_smooth unset (native default0.0). Veto classifiers, labels/default label_gain, LambdaRank, features, seeds, rounds,18-config nested grid, objective and tie/switch/failure-sentinel rules remain unchanged.

The design remains H303 NESTED/H308 TRAIN-only: P0/P1/P2 × roots 30001–30003 × four family-grouped outer folds,36 tasks/9 cells, 1,219 contexts/205 families. Two pinned H287 constructors at `213b30ad52e1ce48927bba960ed609bd98ab4dfe` are targeted. Removing the single keyword restores both complete BASE function ASTs. No worker harness was instantiated or qualified after admission failed.

[PREREG.md](PREREG.md) and code were published before any H333 fit at `0ab56940ee96bafd959d29c8fa9e6c4b850fc56b`. See [path_smooth_patch.py](path_smooth_patch.py), [tests](test_path_smooth.py) and [manifest.json](manifest.json). No parameter sweep, borrowed cores or reduced design was used.

## Capacity and full-run forecast

Actual probe:20:27:09–20:27:14Z on 4 November 2026, one complete five-second interval. Both each assigned core and its SMT sibling must have idle ≥85%; 84.69% is excluded. The receipt retains extracted actual mpstat idle values and its command/source, with no capacity retry.

| core | idle % | SMT sibling | sibling idle % | admitted |
|---|---|---|---|---|
| 16 | 91.65 | 80 | 79.47 | no |
| 17 | 55.37 | 81 | 88.16 | no |
| 18 | 80.74 | 82 | 90.39 | no |
| 19 | 67.01 | 83 | 72.82 | no |
| 20 | 84.69 | 84 | 88.19 | no |
| 22 | 86.94 | 86 | 90.45 | yes |

Existing H329 synthetic GBDT timings were 2.4042/2.4367/2.5926seconds. H329 full-run CPU 30409.446641s, its worst DART/GBDT ratio1.088392827976701 and the existing 90% ranker-share assumption imply 28168.539464GBDT-equivalent CPU seconds. The retained complete 36 GBDT reference used by H329 is 26268.587368CPU seconds. These are timing references; no H329 job or scientific evaluation was resumed.

The forecast uses the larger reference, path-smoothing multiplier1.0, one eligible worker, parallel efficiency 0.9, safety 1.15 and 300 seconds scoring reserve. From 20:31:10Z it ends **06:36:03Z on 5 November**. Even the smaller reference with perfect efficiency and no safety or scoring reserve ends **03:48:59Z on 5 November**, beyond tonight’s 22:45Z stop. These are planning projections, not measured PSMOOTH performance or rigorous runtime lower bounds. [runtime_forecast.json](runtime_forecast.json) retains all assumptions; [capacity.json](capacity.json) retains the gate.

## Outcome, metrics and verification

**10 data-free controls passed**, with no failures or skips: both full-function AST inverse checks, every original ranker keyword/round/seed expression unchanged, fixed GBDT/minleaf/default gain, invalid intervention rejection, actual five-second capacity parsing, threshold inclusion and complete 36 forecast rejection. Actual control service invocation `4f3b15aa4dda434b8969b2bd5ae40a99` exited 0 on core 22, nice 19, one numerical thread, MemoryMax 6 GiB/SwapMax 0. Peak control memory was 20,242,432 bytes and swap 0.

Scientific BASE regression, physical worker input qualification, native-model parameter audits and all numerical metrics are **NOT_RUN** because the resource gate denied the campaign. Historical BASE PASS receipts are not substituted for fresh H333 evidence. No gain, drop-top, harm, W/T/L, invalid, switch, Elo or config-agreement numbers are fabricated; [metrics.json](metrics.json) records null values and execution counts separately.

Elo label, reserved for a completed comparison: **reference = r882 slot0 (DL-Train outer); r3175 not measured on TRAIN**.

The preregistered decision remains paired PSMOOTH−BASE drop10 median≥+.002 with LCB95>0 (10,000 shared whole-family draws,RNG333191, quantile.05; each arm redrops its own top10), full median gain≥90%BASE, median harmed≤BASE and positive paired root medians on≥2/3roots. No decision gate was evaluated. Hosted CI was NOT_RUN under CI_FREEZE and MASTER_MERGE_HOLD; reviews are by the implementing agent.

## Custody and next step

Create-only archive: `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H333/gov03-closeout-20261104T203416Z/h333-scratch.tar`. Exact VersionId `USvd6JWuadbQaR7ZFkn.I4WuEjo33i2l`, SHA256 `2518b78f8728fc89686f9233cdc003d2013a4ce7d79878662d672ede70e6c25c`. Exact-version GET and all 40 file hashes were verified before deleting only `/home/ubuntu/jope-data/h333-gov03` (99,208 bytes). Fresh consumer check found none. The generated restore tar was reclaimed; immutable archive/source refs remain. Archive-receipt VersionId `mRMpSoGnbmshd2FOFnbwvbu7MQa9Dd9b`. Lifecycle evidence: `/home/ubuntu/jope_plan/evidence/H333/gov03-20261104-t2030`. No models enter Git.

A new authorized window with enough eligible assigned cores is required before fresh BASE validation and the complete 36-task experiment. No partial scoring, deadline reset or dev91 evaluation is authorized here. DOPE c51a56d0 was unchanged and not loaded; scope CPU selection layer only. No GPU, protected data, DL-Val/dev91/DL-Test, live writer or foreign-job action was taken. H329 and H326 remain closed.

Both H333 index rows and this report are published nonforce to origin/v1/next through a private index; no PR, main, fastlane or CI dispatch.
