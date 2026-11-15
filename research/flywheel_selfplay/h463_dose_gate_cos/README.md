# H463: recalibrated gate and annealed dose on the pairnf step (offline, adapter-only)

**Status: `RUNNING` (pre-registration record).** This README records the design, code hash and launch before any CV result exists. The verdict, metrics and red-team will be added when all 3 seeds write `summary.json`.

## Hypothesis
H461 (NO_LIFT, #2331) showed that running the same pairnf trajectory to 32 steps (`s32_z40`) inflates the z-margins of changed picks about 7x (median 1.19 vs 0.18). The absolute z >= 0.40 gate then keeps about 745 changes per partition instead of about 104. s32 recovers 15 of 19 incumbent gate failures (+0.033 gain from floors) but loses about 0.03 on non-floor contexts (non-floor Elo -7 to -14, 4-7 new gate failures), so it nets out near zero. The gate was calibrated for the 16-step dose. H463 asks whether the 32-step dose helps once the gate is recalibrated (label-free) or the dose is annealed:
- `pairnf_z40`: the control, bit-for-bit H452-H461 pairnf + z >= 0.40 (step-16 snapshot of the 32-step run). Adapter shas must equal H461 per cell.
- `s32_z200`: the H461 s32 adapter (the last iterate of the same 32-step run, shas must equal H461 `s32_z40` per cell) with an absolute z >= 2.00 gate. The 2.00 value was fixed from the label-free H461 TRAIN-OOF z distribution so that s32 keeps about as many changes as the control (the top ~11% of s32's changed decisions, vs 14.8% at 0.40). It uses no labels or utilities.
- `s32cos_z40`: 32 steps with a 2-step linear warmup, then cosine lr decay to 0. The integrated lr is about 16 full steps (the control's budget) spread over 2x the data. It shares every random stream with the control (`rng_arm`).
- `s32cos_z200`: the same cosine adapter with the z >= 2.00 gate.

Reading: `s32_z200` vs the control tests whether the dose helps once confidence is recalibrated, and `s32cos` vs the control tests an annealed dose. Not tested before: no wave recalibrated the gate per dose or ran warmup+cosine on the pairnf step. Already falsified and not repeated: Polyak/EMA tail average and ASAM (H457), raw s32 and fbal (H461).

## Pre-registered rules (unchanged from H452-H461)
- **Selection** uses Elo only: per seed, the max mean non-floor OOF Elo over the two partitions (ties broken by pooled gain).
- **Strict:** pooled OOF gain > 0.002 and gain CI low > 0; pooled Elo > 0 and Elo CI low > 0; non-floor Elo CI low > 0; Elo > 0 on each partition.
- An arm counts only if it also beats the control in the paired family bootstrap (arm - control CI low > 0). Ties never win.
- **LIFT** requires a non-control arm that beats the control on the median across 3 seeds with the paired CI above 0, plus a clean red-team and val91 not_worse on every seed. Otherwise the verdict is `NO_LIFT`.

## Setup
- **Code:** `h463_trainfold_cv.py`, sha256 `f85ea211a079e3d9687bf7d75e3924d25a623ee77d2e998171e99d5710c2263f` (checked with `sha256sum -c` in the scratch dir before launch). CPU only, so no GPU and no `qualification.lock`. It runs at `nice 19` with 16 threads per seed and aborts itself at MemAvailable < 15 GB.
- **Scratch:** `~/jope-data/hwave-h463` on xbabe2, a copy of `round-3175.pt`, never the live run directory. The run never touches xbabe0.
- **Data:** the real S3 packet: 1,219 DL-Train contexts (205 families) and 91 DL-Val contexts (19 families). val91 is read once per seed, for the CV-selected arm only, as a regression check. No sealed or holdout data is read.
- **DOPE frozen:** composite sha `c51a56d0...` is checked before and after. Only the 29,211 adapter parameters train.
- **Partitions:** the two H452 floor-stratified, family-grouped 5-fold partitions (`h452a/`, `h452b/`), with 2 trained bases (pairnf-32 with a step-16 snapshot, and cosine-32) x 2 partitions x 5 folds per seed, plus 1 full-train run for the selected arm.
- **Probe:** PROBE_OK at 2026-11-15 15:40 UTC, DOPE after `c51a56d0...`, peak RSS 6.4 GB, min MemAvailable 64 GB.
- **Launch:** `launch.sh` is gated on MemAvailable >= 38 GB and machine CPU busy% + 16 threads <= 85%. Seeds 1/2/3 launched 2026-11-15 15:41 / 15:51 / 16:01 UTC (mem 85 / 73 / 75 GB, CPU busy 29 / 30 / 14%). `waiter.sh` relaunches a dead seed (any cause, including the silent death seen in H461), resume-safe via `results/fold_sig`, at most 3 times per seed. It SIGTERMs only its own seed process if that process hangs more than 10 min after `summary.json`.
- **Parity:** the incumbent val utility reproduced 1.1389572403748425 (`parity True` on all 3 seeds). Every `pairnf_z40` and `s32_z200` cell must match the H461 adapter shas.

## Aggregation (when every seed has `results/seedN/summary.json`)
Copy the aggregator and red-team scripts from H461, then run `WAVE=H463 python3 h463_aggregate.py aggregate.json` and `ARMS=pairnf_z40,s32_z200,s32cos_z40,s32cos_z200 python3 h463_redteam.py redteam.json`.

Raw logs, losses and fold signatures stay on xbabe2 in `~/jope-data/hwave-h463`.
