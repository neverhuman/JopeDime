# H164 — SAM/ASAM continuation with family cross-fitting

**NO_ADVANCE.** The one preregistered CPU attempt completed 60 cells, 2,400 actual optimizer updates and 4,200 training forward/backward passes. No serving promotion. [metrics.json](metrics.json) contains all registered readouts.

Hypothesis: SAM/ASAM adapter updates reduce continuation overfitting from r882. H153 had stopped before baseline because launch.json lay outside its audit boundary; it supplied no efficacy result. H164 repaired that path and exercised it in an audited zero-forward model+AdamW constructor smoke before execution.

[PREREG.md](PREREG.md) fixed CTRL AdamW, SAM rho 0.01/0.05 and ASAM rho 0.5 (eta 0.01), lr 1e-4, wd 1e-4, clip 1, 40 updates per cell, seeds 164101–164103. Only 47 adapter tensors / 29,211 parameters train. Temporary SAM perturbations restore original weights exactly before one AdamW step. DOPE c51a56d0 and all frozen state remained byte-identical after every update and score; frozen digest 8d1070faf574ee1cd2a4e630887e1662b43189b3f89b04982143ef2704818aae.

Five 41-family folds cover 1,219 TRAIN contexts / 205 families. In each cell three folds fit, next cyclic fold B selects and outer fold A assesses. Selector chooses the earliest B-best snapshot at 0/10/20/30/40; gain must exceed +0.001 + 1e-12 with a 10,000-draw family-bootstrap LCB > 0, else r882. Both unconstrained and best-nonzero B selections were locked before A/DL-Val scoring. Historical r882 TRAIN exposure remains; bootstrap conditions on overlapping fixed policies and does not refit them. DL-Val is repeatedly exposed, not fresh or sealed qualification.

Primary: equal context weights within family, equal family mean per seed, median over 3 seeds; 10,000 shared-family bootstrap draws, ordinary 95% CI, Holm across all 4 trained arms. ADVANCE requires gain > +0.001 + 1e-12, CI lower > 0 and Holm p < 0.05. Elo uses canonical 400-iteration BT MM, family-averaged context wins, tie margin 1e-4, prior 0.01, r882=1500.

| Arm | OOF family gain | Ordinary 95% CI | Elo | Holm p | Positive seeds | Selected steps (15 cells) |
|---|---:|---|---:|---:|---:|---|
| CTRL | +0.00000000 | [+0.00000000, +0.00000000] | 1500.00 | 1.00000 | 0/3 | {0: 15} |
| SAM01 | +0.00000000 | [+0.00000000, +0.00000000] | 1500.00 | 1.00000 | 0/3 | {0: 15} |
| SAM05 | +0.00000000 | [+0.00000000, +0.00000000] | 1500.00 | 1.00000 | 0/3 | {0: 15} |
| ASAM05 | +0.00000000 | [+0.00000000, +0.00000000] | 1500.00 | 1.00000 | 0/3 | {0: 15} |

All 60 strict selectors fell back to step 0. The identical selected vectors share the same r882 outcomes; they are not independent confirmations or evidence that trained checkpoints had zero effects. The null belongs to the registered selection policy.

The following are descriptive locked B-selected policies; neither assessed gains nor DL-Val chose the checkpoint. Best-nonzero diagnostics remain visible even when the strict and unconstrained policies both select 0.

| Arm | Unconstrained OOF gain [95% CI] | Best-nonzero OOF gain [95% CI] | Selected DL-Val gain [95% CI] | DL-Val Elo |
|---|---|---|---|---:|
| CTRL | +0.00045789 [-0.01085498, +0.01494401] | -0.00739001 [-0.02616033, +0.01028146] | +0.00000000 [+0.00000000, +0.00000000] | 1500.00 |
| SAM01 | +0.00000000 [-0.01084339, +0.01214254] | -0.01059686 [-0.02900287, +0.00882220] | +0.00000000 [+0.00000000, +0.00000000] | 1500.00 |
| SAM05 | -0.00286092 [-0.01240104, +0.01144753] | -0.00468092 [-0.02133305, +0.01062541] | +0.00000000 [+0.00000000, +0.00000000] | 1500.00 |
| ASAM05 | -0.00057698 [-0.01191353, +0.01386373] | -0.00391247 [-0.01905028, +0.01182961] | +0.00000000 [+0.00000000, +0.00000000] | 1500.00 |

Secondary 91-context DL-Val utility is the uniform average of 5 fold-policy utilities per seed, not an ensemble-score predictor or 455 independent contexts. Bootstrap resamples 19 families; invalid counts have 455 policy evaluations per seed. Full per-seed utility/Elo/CI and invalids for all six panels are in metrics.json. Selected primary invalid counts by seed, columns r882/CTRL/SAM01/SAM05/ASAM05: `[[17, 17, 17, 17, 17], [17, 17, 17, 17, 17], [17, 17, 17, 17, 17]]`.

Exact 91-context r882 baseline passed first: mean 1.1377941898164525, zero context/mean error. Independent replay verified all 60 cell selectors and selection-lock hashes, all paired batch plans, 300 snapshots, changed adapter parameters, frozen-state identity, 2,400 journal entries and 48 contrasts; 114626 saved choices matched original labels/gates. Replay ran zero model forwards/updates.

Execution source `93c8ce017ffa114f6c60b2b81b1e4be5a81ca75f`, prereg [PR #1408](https://github.com/neverhuman/JopeDime/pull/1408); external REDTEAM review 5726938075 plus head-bound author integration review. All 12 scientific/protocol files were byte-identical from peer-reviewed source through merged launch. 20 data-free controls, Ruff/format/hash/link checks and 5 static records gates passed. Integration repaired main's missing H165 test classification without changing that executor. No hosted CI/product qualification claim. An initial preparation wrapper hit the thread guard and was stopped; the corrected direct-wrapper tests passed before the sole scientific attempt.

During the scientific attempt on CPU xbabe2 at nice 19, sampled supervisor/worker aggregate max 3 OS threads, peak RSS 3331518464B, max load 79.79, min MemAvailable 66833891328B, min disk free 547160522752B. Terminal guards and 12h scientific deadline applied. Python audit is application evidence, not an OS sandbox; no claim against unknown aliases. No live-loop/serving/qualification-lock/GPU access.

Custody: `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H164/astra-t0058/manifest.json`, VersionId `_Z5xN4x4ftykxrxf4jioq33gCQBGC7.d`, SHA256 `ae4e2b72ed60f15665837ab31bc6db16349aebd28c89c8598bfcd19b222dc41a`. [custody.json](custody.json) binds create-only exact-version downloads and 1891 verified member hashes. Original H50/H58 derivative versions remain unknown; exact used numerical closure is archived and external implementation sources pinned. Owned archived bulk is eligible for cleanup only after closeout merge; shared inputs remain.

The first postprocessing S3 attempt stopped at 6 aggregate threads when the AWS CLI credential provider spawned descendants. This was after the completed scientific run. Its failure receipts were retained; resolving credentials once in memory and using a synchronous client completed custody with at most 2 sampled threads. Credential bootstrap itself was outside that sampled guard; no secret bytes were persisted. The scientific attempt was not repeated.

Next: retain r882; further dose, selection or optimizer experiments require a new preregistered wave. H164 does not authorize serving promotion.
