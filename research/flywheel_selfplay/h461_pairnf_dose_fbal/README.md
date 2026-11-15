# H461: dose and family-balanced sampling on the gated pairnf step (offline, adapter-only)

**Status: `RUNNING` (pre-registration record).** This README records the design, code hash and launch before any CV result exists. The verdict, metrics and red-team will be added when all 3 seeds write `summary.json`.

## Hypothesis
H457 showed that both stabilizers (Polyak tail average, ASAM) act as shrinkage on the pairnf update. They compress the z-margins of changed picks, so the z ≥ 0.40 gate keeps fewer changes at the same ~57% win rate, and both lose to the control. H461 tests the two axes that result points at:
- `pairnf_z40`: the control, bit-for-bit H452–H457 pairnf + z ≥ 0.40. It is taken as the step-16 snapshot of a 32-step run on the same random streams, so the adapter shas must equal H454/H457 per cell.
- `s32_z40`: the dose axis, the opposite of avg8. It is the last iterate of the same 32-step pairnf trajectory, strictly paired with the control (same init perturbation, dropout, context and slate streams).
- `fbal_z40`: family-balanced context sampling. Each training context is drawn with weight 1/|family|, so every family is equally likely. It runs 16 steps and shares the control's init-perturbation and dropout streams (context draws necessarily differ). CV is family-grouped and large families dominate the gradient.

Neither was tested on pairnf before: H444 `fbal_accum` ran on the plain step at low lr, and no wave ran the pairnf step past 16 steps.

## Pre-registered rules (unchanged from H452–H457)
- **Selection** uses Elo only: per seed, the max mean non-floor OOF Elo over the two partitions (ties broken by pooled gain).
- **Strict:** all of the following must hold:
  - pooled OOF gain > 0.002 and gain CI low > 0
  - pooled Elo > 0 and Elo CI low > 0
  - non-floor Elo CI low > 0
  - Elo > 0 on each partition
- An arm counts only if it also beats the control in the paired family bootstrap (arm − control CI low > 0). Ties never win.
- **LIFT** requires a non-control arm that beats the control on the median across 3 seeds with the paired CI above 0, plus a clean red-team and val91 not_worse on every seed. Otherwise the verdict is `NO_LIFT`.

## Setup
- **Code:** `h461_trainfold_cv.py`, sha256 `bf518f7a65489b9e17ebd277a11d2f4d69a4420501dcd8efba62a8ce31e2dfb8` (verified by `sha256sum -c` in the scratch dir before launch). It runs on CPU only (`torch.device("cpu")`, not in `nvidia-smi` compute apps), so it doesn't use the GPU or `qualification.lock`. It runs at `nice 19` with 16 threads per seed and aborts itself at MemAvailable < 15 GB.
- **Scratch:** `~/jope-data/hwave-h461` on xbabe2, a copy of `round-3175.pt` plus optimizer state, never the live run directory. The run never touches xbabe0.
- **Data:** the real S3 packet: 1,219 DL-Train contexts (205 families) and 91 DL-Val contexts (19 families). val91 is read once per seed, for the CV-selected arm only, as a regression check. No sealed or holdout data is read.
- **DOPE frozen:** composite sha `c51a56d0…` is checked before and after. Only the 29,211 adapter parameters train.
- **Partitions:** the two H452 floor-stratified, family-grouped 5-fold partitions (`h452a/`, `h452b/`). Each seed runs 2 trained bases (pairnf-32 with a step-16 snapshot, and fbal-16) × 2 partitions × 5 folds, plus 1 full-train run for the selected arm.
- **Probe:** PROBE_OK at 2026-11-15 05:38 UTC, with DOPE after `c51a56d0…` and peak RSS 6.2 GB. In the 1-step probe, s32 and control have the same sha, as expected.
- **Launch:** `launch.sh` gated on MemAvailable ≥ 38 GB and machine CPU busy% + 16 threads ≤ 85%. Seeds 1/2/3 launched 2026-11-15 05:42 / 05:52 / 06:02 UTC (mem 79 / 74 / 66 GB, CPU busy 34 / 27 / 24%). `waiter.sh` relaunches a seed only after a memory-guard abort (resume-safe, max 3 relaunches per seed).
- **Parity check:** every `pairnf_z40` cell must equal `../hwave-h457/results/fold_sig` (30/30 at the end). The incumbent val utility must reproduce 1.1389572403748425; seeds 1–2 already logged `parity True`.

The design was first announced in the H457 README (merged in #2324 at 25dd912d8d, 2026-11-15 05:07 UTC), before any H461 compute ran.

## Aggregation (when every seed has `results/seedN/summary.json`)
`WAVE=H461 python3 h461_aggregate.py aggregate.json` and `ARMS=pairnf_z40,s32_z40,fbal_z40 python3 h461_redteam.py redteam.json`. Both are copied from H457.

Raw logs, losses and fold signatures stay on xbabe2 in `~/jope-data/hwave-h461`.
