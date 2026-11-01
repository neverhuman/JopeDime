# H133 preregistration — optimizer geometry / step size vs harm

Status: **READY FOR PREREGISTRATION; main study NOT RUN.** This protocol and its hashes must be frozen before any fold-0 assessment payload or outcome is read. An H-wave owner ruling (t1435) explicitly permits the limited fit-only calibration below before preregistration. It uses folds 1–4 only and reports parameter drift only. No assessment/outcome metric is used to pick the rate.

## Hypothesis

H124 had positive median gain but exceeded the family harm cap (10.73% > 10%). H130 suggests lower adapter drift can reduce harm; H105 showed uniform interpolation can harm. We test whether a lower learning rate with a longer schedule, or a non-adaptive momentum optimizer, preserves utility gains while lowering family harm versus unchanged `r882`.

## Frozen inputs and split

- Checkpoint: fresh hash-verified copies of `r882`, SHA-256 `0084479aa0703667a0f3ccfe2beda60741c204795ec57b7c08542401ec6cdb64`.
- Frozen DOPE: SHA-256 `c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814`; verify before and after every cell.
- Source: H50's 1,219 real S3 `DL-Train` contexts (205 families), represented by the DL-Train-only metadata export at `/home/ubuntu/jope-rescue-20260921/h133-optim-geometry/train_only_export.json` (SHA-256 `ffe506f68d7aade13a7eef6f0cf55d6e79b200dd5e742f8f035dccd0ee27e3ff`). It contains no dev, DL-Val, sealed, or holdout rows. The training runner opens only pool/table payloads whose families are in folds 1–4.
- Deterministic family split: `numpy.random.default_rng(133100)` permutes the sorted train-only family IDs and partitions them into five equal family folds. Manifest: [fold_manifest_133100.json](fold_manifest_133100.json), SHA-256 `322137e01ef0a6079284c0e907c8a5ce0db476d46eac1c6ea0fdce3315ea65fc`. Fold 0 is assessment (41 families, 245 contexts); folds 1–4 are fit (164 families, 974 contexts). Family IDs are disjoint. No DL-Val, outer-test, sealed, or holdout payload is read.
- Fresh copies are used per trajectory/cell. Only adapter parameters train. Non-trainable parameters and DOPE are hash-verified unchanged after each cell.

## Arms and calibration

All arms use the H132 CONTROL objective and batch-plan generator, batch size 16, slate size 192, weight decay `1e-4`, and gradient norm cap `1.0`, unless a difference is specified below.

- `CONTROL`: AdamW, LR `1e-4`, 40 updates per seed.
- `LOWLR_LONG`: AdamW, LR `2.5e-5` (0.25 × CONTROL), 160 updates per seed; same batch-plan generator, extended to 160 steps.
- `SGDM`: SGD, momentum `0.9`, Nesterov enabled, LR `0.003` (pinned by the fit-only calibration receipt), 40 updates per seed.

**Fit-only LR calibration authorized by the H-wave owner before preregistration:** On seed 133 and folds 1–4 only, run the exact CONTROL optimizer for 160 updates and measure adapter L2 drift from `r882`. Run SGDM for 160 updates at each fixed candidate LR `{1e-3, 3e-3, 1e-2, 3e-2}` on the identical fit-only batch plan. Select the candidate with minimum absolute drift difference from CONTROL; exact ties select the smaller LR. No assessment payload, assessment prediction, family utility, or other outcome metric was read or calculated. Receipt [calibration_fit_only.json](calibration_fit_only.json), SHA-256 `c969a588154f7acb1a9d26545e9b64b2290605c7f965f448e5a0798c3538fc58`, records CONTROL drift `2.069222458889`, SGDM drifts `1e-3: 0.783712030767`, `3e-3: 1.750078662240`, `1e-2: 3.432225430399`, `3e-2: 4.847053083022`; the closest is `LR=0.003` (absolute difference `0.319143796649`). Batch-plan SHA-256 `8e612616168f680ed8e7430ab07383a2cb81c4ff0dbca83a15ebbee9cd2acf72. DOPE and all non-trainable parameter hashes matched pre/post for all five calibration trajectories. Calibration is a protocol-calibration stage, not an outcome arm or admission test. Main 40/160/40-update arms remain unexecuted until preregistration merges.

**Attempt disclosure:** Calibration attempt 1 completed CONTROL-160 and SGDM `0.001`-160, then was voided when load1 reached `87.96` before SGDM `0.003` update 160; it had persisted no drift values. No assessment payload/outcome metric was read. Receipt [calibration_attempt_001.json](calibration_attempt_001.json) records the stop. Attempt 2 is the complete calibration above; every completed trajectory was persisted before proceeding.

The original preregistration seeds were `133, 134, 135`. Prospective A6 replaces 135 with 138 because H135 owns 135–137. The SGDM LR remains frozen from seed-133 fit-only calibration and is reused unchanged for all three main seeds. No assessment tuning or post-hoc rate changes are allowed.

**Calibration implementation pins:** H133's fit-only calibration runner is `calibrate_fit_only.py` (SHA-256 `862f2f13c7c441a5c888a7e293668139fab6d002a9c33083d0110b88e75c6edb`). It uses the H132 helper file SHA-256 `4d3e8182e674410b179e13ec2cadba21e1715c65b677053fd776a2dfe4a65673`; this is the amended final H132 helper, not its originally preregistered `c1bb24c9` version. H132 Amendment A1 reports three edits: DL-Train pool-key loading, a BITFIT trainable-count/DOPE guard correction, and seeded Fisher / Step-4 tie handling. H133 calibration uses its own fit-only input loader and does not invoke H132's Fisher or fold-0/Step-4 code; it uses the helper's batch-stream seed and per-context loss definitions. Imported helper modules are pinned to source commit `30be5edb79b0ce1406aa0367d68019c9a3a3c1e1` with SHA-256s: `756960a0e2bfdf23e75012fb347ce772fb6fdeaa4fb4fc2d4178dfaeb9eac9da` (`h48/run.py`), `56fd220689805f07877ae403c2f00a4b34831cc4b2919f4bd69807ff3c21f4bc` (`h55/runner.py`), `85aa59e9c4bee62805ed6c388364611b3a7cdb772a19017eeae848819c2cb318` (`h53b/runner.py`), `604d02ed72a6149284792cba315124efd1ffe8cac79e5ace22613db86c11b4c3` (`h59/runner.py`), `645bf1470789c4f5bcc97809f703b04d9a594a21dd9d3aa8c2fd4e98810d0320` (`h46/run.py`).

## Assessment and admission

After preregistration merge, first run the unchanged-`r882` regression-first check on fold 0 and require exact reproduction before any main arm cell. Then run 3 arms × 3 seeds (`720` planned optimizer updates; calibration updates are separately disclosed above).

Primary outcome is paired per-family utility gain versus unchanged `r882` on fold 0, with 10,000 family-cluster bootstrap draws and median plus 95% confidence interval; Holm correction is over the two treatment arms. An arm is admitted only if all conditions hold:

1. Median family-seed-mean gain is strictly `> +0.001`.
2. Holm-adjusted lower 95% confidence bound is strictly `> 0`.
3. Family harm fraction is `<= 10%`, where harm is family seed-mean gain `< -0.01`.
4. Mean gain is strictly `> 0` on each of all three seeds.
5. Exact ties and zero gain are `NO_ADMISSION`.

Report family-paired Elo (not AUC), continuous mean regret, per-seed gains, adapter L2 drift, and harm versus drift.

## Resource and execution guards

CPU only; launched at `nice 19`; at most 4 threads; RSS `<= 6 GiB`; `/` disk usage strictly `< 93.8%`; H133 scratch strictly `< 2 GiB`; host CPU `<=85%`. These guards remain hard aborts. A6 changes only load and available memory to PAUSE-AND-WAIT: send SIGSTOP when load1 > 86.7 or MemAvailable < 30 GiB; send SIGCONT only after load1 < 80 and MemAvailable >= 34 GiB continuously for 60 seconds. The launcher records pause/resume events and resource snapshots. Four-hour wall budget includes pause time. No GPU, no `qualification.lock`, no reads from DL-Val, outer-test, sealed, or holdout. Custody: `/home/ubuntu/jope-rescue-20260921/h133-optim-geometry/`.

## Scope and status

This document preregisters only H133. No serving/promotion claim follows from any result. Execution is authorized only after this preregistration has merged. All trial, void, and restart attempts must be reported; no unlogged reruns.


## Prospective amendment A6 — owner authorization t1630 / t1715

A6 is limited to the execution launcher/resource guard and documentation. It does not change the hypothesis, objective, arms, update counts, split, fold seed `133100`, batch plans, metrics, admission thresholds, or statistical gates. Main seeds are `133, 134, 138`; seed 138 replaces seed 135, which is owned by H135. No H133 seed 135 cell was run.

The load and MemAvailable thresholds now pause execution through the supervised launcher. The worker sends SIGSTOP on a load1 > 86.7 or MemAvailable < 30 GiB observation; the launcher sends SIGCONT only after load1 < 80 and MemAvailable >= 34 GiB have both held continuously for 60 seconds. Each pause and resume is written to `pause_events.jsonl`, with periodic host/resource snapshots in `resources.jsonl`. Other resource checks remain hard aborts. CPU only, nice 19, at most four threads, no qualification.lock, and the four-hour cap includes startup, active work, and pauses. DOPE SHA `c51a56d0…` is verified before and after each cell.

The executor has separate source, data, and fresh run roots so attempt 013 cannot resume or overwrite attempts 001–012. The launcher validates the pinned runner and refuses an existing run root. It runs the exact r882 regression first; the required family-utility reproduction is `1.1377942`. No treatment cell may start unless that exact regression passes. The preferred host is xbabe1 if the complete train-only input closure is below 2 GB; the measured H50 input directory is 457,391,900 bytes locally and 175,187,219 bytes on xbabe1 before staging the missing tables and r882 checkpoint. The final host and staged closure hashes must be recorded in the launch receipt.

### Complete attempt disclosure through attempt 012

- Attempts 001–007: startup/import/thread voids before any outcome. They are preserved as `startup_attempt_001.json` through `startup_attempt_007.json`; no optimizer updates or assessment scores.
- Attempts 008–009: exact r882 regression passed, then executor/thread initialization stopped before a treatment update; zero treatment outcomes.
- Attempt 010: CONTROL seed 133 performed 40 fit updates, then a fit-only predictor caused a KeyError before any assessment model prediction. The 40 updates are unscored and void.
- Attempt 011: load guard stopped execution before input reads or scoring.
- Attempt 012: exact regression passed; CONTROL seed 133 completed 40 updates (adapter drift 0.4389823302707865; fold-0 family utility 1.0960967447286205). LOWLR_LONG seed 133 performed 100/160 fit-only updates, then load1 424.13 > 86.7 and MemAvailable 21.64 GiB < 30 GiB. It produced no assessment outcome or cell receipt; these 100 updates are unscored and void. The run held `qualification.lock`, contrary to the no-lock protocol; the lock was released on exit. This deviation is disclosed and A6 must not acquire that lock.

A6 is a prospective amendment; it does not repair or reclassify any prior attempt. The merged amendment and exact regression gate are required before the single next launch. No automatic retry or resume is allowed.
