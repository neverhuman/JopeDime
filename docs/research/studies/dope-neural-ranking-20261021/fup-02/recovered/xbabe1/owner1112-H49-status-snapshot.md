# H49 closeout + H50 preregistration

Updated: 2026-10-30T10:30Z

## H49 authorization and state

Per `~/nudges/luna_ns_hbaseline_1028.md`, start H49 immediately on independent approved checkpoint/input copies; do not wait for H48 PR #513 to merge. The H48 PR remains separately open at `e40d4209eefd073f7226b464ec9f7e9ca48a0220`. H49 uses the existing H48 copied cohort and a separately copied round-785 checkpoint with pinned SHA. No live-loop, sealed/holdout/binary, or original checkpoint access is allowed.

Current guard read at 10:30Z: available RAM 83 GiB (20 GiB minimum), swap 19/19 GiB full. `qualification.lock` is held by unrelated PID 755392 running `gpu_bench/fused_hpc_r_stack_0930/run_0930_locked_qual.sh`; do not interrupt or bypass it. H49 preparation may proceed; launch/wave waits for lock. cuda:0 only, no CPU fallback. Before every seed require available RAM ≥20 GiB, lock ownership, host CPU ≤85%, and at least 20 GiB free on cuda:0. If `MemAvailable` is below 20 GiB, pause H49 launch only.

## H49 preregistered arms

- **H49-0:** three fresh seeds 0–2, H48-0 restore-on-reject objective and family-grouped five-fold selection.
- **H49-SAM:** three fresh seeds 0–2; ASAM rho=0.05, persistent optimizer. H48-PF was worse than H48-0 at terminal family median, so use the H46-0 Huber+KL objective (the H48 `0` objective).
- **H49-SWA:** three fresh seeds 0–2; persistent AdamW, cyclic learning rate with one full 32-update cycle, average weights across the full cycle, and select on inner family-CV only.
- **H49-SEL:** no training and no DL-Val read. Re-evaluate admission offline with K=8 candidate history and a strict margin: accept only if the family-bootstrap 95% lower bound of median family-mean inner-CV gain is greater than 0.001 over the incumbent. Exact ties and +0.0002 one-context moves are rejected. From known events r782, r882 (ties), and r859 (+0.000199), 0/3 would pass this rule. No live process/log is to be opened.
- **H49-ES2:** three fresh seeds 3–5, antithetic ES, same H48 inner family-CV selection and rank shaping, 2× H48-PF median wall budget (~141 seconds/seed).

Terminal DL-Val budget: one read per selected seed for 0, SAM, SWA, and ES2 (12 candidate reads total); SEL uses none. No baseline rescoring. Strict win requires median family-mean Δ > 0, family-bootstrap CI lower bound > 0, and Elo > 1500. Report context deltas separately, paired family comparisons to H49-0, min LOCO, train/validation gaps, cohort/model hashes, and selection bias.

If GPU/lock time is constrained, drop SWA before ES2. Preserve outputs only under `~/jope-data/h49/`; keep host RSS low. H48 output/checkpoint/source files are read-only.

## H50 preregistration (launch only after H49 terminal reads)

- **H50-S first:** utility sensitivity map with five L2 radii `[0.01, 0.03, 0.1, 0.3, 1.0]` and 64 seeded random directions per radius (320 evaluations total), using inner held-out DL-Train folds only. Report fraction of exactly zero actual-utility changes at each radius and smallest radius producing any utility movement. If more than 90% are zero at every radius, stop H50 search arms and conclude the adapter is not the bottleneck.
- If H50-S does not stop the wave, run **H50-ES3** (antithetic ES, population 32, 2× H49-ES2 wall budget = 282 seconds/seed), **H50-CMA** (diagonal/sep-CMA-ES, same wall budget), and **H50-ESEMA** (EMA of ES iterates, decay 0.9, score the EMA candidate).
- Fresh seeds 6–8; unchanged r785/r882 controls; same family-grouped inner-CV selection, strict pre-registered family-bootstrap median margin >0.001, and one terminal DL-Val read per selected seed. Global win rule remains median family-mean Δ > 0, family-bootstrap lower bound > 0, and Elo > 1500.
- Small H50 README + compact (<50 KB) metrics and index row; no launch before H49 terminal reads.

## Progress

2026-10-30T10:30Z — Nudge 1028 read. H48 #513 records/counts fixed and compact metrics pushed; Codex H48-specific gate NO BLOCK posted. External full metrics SHA `cd77ab828973c1688b9254b80801334f51a06adaf869030b47bce003f63f12b5`. H49 start authorized. Lock remains occupied by unrelated job; no H49 process has acquired it.


2026-10-30T10:36Z — H49 runner/scorer/SEL audit harness committed in local branch `research/h49-hwave` at `b11471a98` (based on H48 write-up branch so H49 can use its reviewed H48 helpers; H48 PR #513 remains separate). H49 input/ckpt/reference copies verified: checkpoint `ef6696f6369122a162d58b9b522fd83cae110a5576ef3760e3b43ba85d129e0a`, input manifest file `99a53569fcaf59f9309e69329c830ae1684adff382da3ae22e0b2645c4ebc4d4`, reference `7deeeaff3426af78329698340b045849204840c2ea6ab0d38b8d89210d3a77a5`. H49 tests 4/4, Ruff, py_compile, bash syntax, verify-test-inventory (726 items), coverage validate, and paper-trace pass. Source-inventory only reports preexisting `python:research/flywheel_selfplay/evo_batch_queue.py`, unchanged from main. `H49-SEL-audit.json` records 0/3 known noise events surviving margin>0.001; no live-loop state read. Launch preflight at 10:36Z: lock free, RAM 89.3 GiB available, CPU 27.2%, cuda:0 free 24,124 MiB, utilization 0%. Starting serialized wave now; no CPU fallback.

2026-10-30T10:36:30Z — `run_wave.sh` owns `qualification.lock`; H49-0 seed0 PID 2497700 is initializing on cuda:0. Initial sample: RSS 3.9 GiB, MemAvailable 84.7 GiB, GPU free 23,774 MiB. No H49 final validation read has occurred.

2026-10-30T10:38Z — H49-0 seed0 finished training on cuda:0 in 87.6s. Selected fold1; family-CV selected delta +0.003542 (search maximum, exploratory only). Validation scored=false and validation payloads opened=false. Seed1 is running; MemAvailable 88 GiB, CPU 9.8%, GPU free 23,774 MiB.

2026-10-30T10:39Z — H49-0 seed1 completed train-only in 57.5s, selected fold1 with inner-CV selected Δ +0.006556 (selection maximum; exploratory only), validation not scored. Seed2 started on cuda:0. Seeds0–1 show distinct family-CV trajectories; no terminal reads.

2026-10-30T10:40Z — H49-0 seeds0–2 completed train-only on cuda:0. Wall 87.6 / 57.5 / 58.1s; selected folds 1 / 1 / 1; inner-CV selected deltas +0.003542 / +0.006556 / +0.005888. These are search maxima, not validation gains. All three train results record validation_scored=false. H49-SAM seed0 is now running under the same lock.

2026-10-30T10:41Z — H49-SAM seed0 is running correctly; inner-CV folds0–1 complete with selected deltas 0.0 and +0.047017. This is train-only CV; no validation score or payload. Resource sample: RAM87.5GiB, CPU9.7%, GPU free23,220MiB.

2026-10-30T10:42Z — H49-SAM seed0 completed five train folds in 83.6s; selected fold1, inner-CV selected delta +0.009403 (exploratory only), validation not scored. Seed1 is running.

2026-10-30T10:44Z — H49-SAM seed1 completed train-only in 120.7s, selected fold1 with inner-CV maximum Δ +0.008215; validation remains unscored. Seed2 started on cuda:0.

2026-10-30T10:47Z — H49-SAM seeds0–2 completed train-only on cuda:0. Wall 83.6 / 120.7 / 86.9s; selected fold1 for all; inner-CV maxima +0.009403 / +0.008215 / +0.006473. All validation_scored=false. H49-SWA seed0 has started.

2026-10-30T10:51Z — Found and fixed SWA selector bug before any validation scoring: initial code selected the averaged candidate even when its CV utility was below r785. Stopped only the owned H49 wave, released qualification.lock, and preserved train-only SWA seed0 plus interrupted seed1 under `~/jope-data/h49/failures/SWA-baseline-selection-bug/`. No final-val files existed. Corrected SWA to compare against and retain baseline on ties/regressions; added strict selection regression tests (now 5/5), Ruff, py_compile, shell syntax and diff-check pass. H49-0 and SAM outputs are intact. Rerun SWA seeds0–2 before any terminal validation; then continue ES2.

2026-10-30T10:52Z — Rerun coordinator reacquired qualification.lock after fresh preflight (MemAvailable 89.6 GiB, CPU12.2%, GPU free24,124MiB/0%). It skipped all completed H49-0 and H49-SAM seeds and started corrected H49-SWA seed0. No terminal val result exists yet.

2026-10-30T10:54Z — Corrected H49-SWA seed0 completed one 32-update cycle per fold, 93.4s total. Baseline retained on folds0/2/3/4; fold1 SWA average improved inner-CV by +0.009242, selected family-CV delta +0.001848. This remains a training-only selection estimate; validation not scored. Seed1 running.

2026-10-30T11:18Z — All H49 train arms completed: H49-0 and SAM seeds0–2, corrected SWA seeds0–2, and ES2 seeds3–5. Terminal-only scorer is started but is waiting for qualification.lock, currently held by an independent gpu_bench run. No DL-Val reads have been made yet. H50 remains preregistered and will launch only after H49 terminal scoring is complete.
