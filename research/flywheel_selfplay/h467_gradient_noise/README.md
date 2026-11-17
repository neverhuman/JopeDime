# H467: gradient noise on the gated pairnf step (offline, adapter-only)

**Status: `RUNNING` (pre-registration record).** This README records the design, code hash and launch before any CV result exists. The verdict, metrics and red-team will be added when all 3 seeds write `summary.json`.

## Hypothesis
H457 (Polyak tail average, ASAM), H461 (dose, family balance), H463 (recalibrated gate, cosine dose) and H466 (objective shape) were all `NO_LIFT`. Each moved how many changes the z >= 0.40 gate keeps, never the ~57-58% non-floor win rate of the kept changes. H467 keeps the dose (16 steps, lr 3e-4), the gate (z >= 0.40) and the objective fixed and attacks the **gradient noise** of the 16-step step. Every arm shares the control's random streams (`rng_arm = pairnf`: same init perturbation, context draws and slates for the control's micro-batch):
- `pairnf_z40`: the control, H452-H466 pairnf + z >= 0.40, 16 steps. Adapter shas must equal the H466 `pairnf_z40` cells.
- `acc2_z40`: gradient accumulation over 2 independent 1,024-context micro-batches per step (effective batch 2,048, mean gradient, same lr, steps and clip). Micro-batch 0 is the control's draw and micro-batch 1 uses its own keyed stream. The extra forward pass consumes the dropout stream, so dropout masks after step 0 differ from the control.
- `look4_z40`: Lookahead (k = 4, alpha = 0.5) on the pairnf step. Slow weights start at the perturbed init.
- `bls10_z40`: Bradley-Terry label smoothing, eps = 0.1, on the pairwise term. Each pair target is 0.9/0.1 instead of 1/0, so noisy near-tie pairs cannot push z-margins without bound.

Not tested before on this instrument: gradient accumulation, Lookahead and pairwise label smoothing. Already falsified and not repeated: Polyak/EMA tail average and ASAM (H457), 32-step dose and fbal (H461), recalibrated gate and cosine dose (H463), ctxnorm, tail clip and pair weight 2 (H466).

## Pre-registered rules (unchanged from H452-H466)
- **Selection** uses Elo only: per seed, the max mean non-floor OOF Elo over the two partitions (ties broken by pooled gain).
- **Strict:** pooled OOF gain > 0.002 and gain CI low > 0; pooled Elo > 0 and Elo CI low > 0; non-floor Elo CI low > 0; Elo > 0 on each partition.
- An arm counts only if it also beats the control in the paired family bootstrap (arm - control CI low > 0). Ties never win.
- **LIFT** requires a non-control arm that beats the control on the median across 3 seeds with the paired CI above 0, plus a clean red-team and val91 not_worse on every seed. Otherwise the verdict is `NO_LIFT`.

## Setup
- **Code:** `h467_trainfold_cv.py`, sha256 `5d011b8c2c74f46ed424b6d9433a1dbe0ce05c95cf0dfd70f30c2a0044c902f0` (checked with `sha256sum -c` by the waiter before every launch). It differs from `h466_trainfold_cv.py` only in the arm table, the `accum` and `bls` branches, the probe's full-train arm list and the docstring. Lookahead reuses the existing H457 `look` branch. CPU only, so no GPU and no `qualification.lock`. It runs at `nice 19` with 16 threads per seed and aborts itself at MemAvailable < 15 GB.
- **Scratch:** `~/jope-data/hwave-h467` on xbabe2, a copy of `round-3175.pt`, never the live run directory. The run never touches xbabe0 or xbabe3.
- **Data:** the real S3 packet: 1,219 DL-Train contexts (205 families) and 91 DL-Val contexts (19 families). val91 is read once per seed, for the CV-selected arm only, as a regression check. No sealed or holdout data is read.
- **DOPE frozen:** composite sha `c51a56d0...` is checked before and after. Only the 29,211 adapter parameters train.
- **Partitions:** the two H452 floor-stratified, family-grouped 5-fold partitions (`h452a/`, `h452b/`), 4 trained bases x 2 partitions x 5 folds per seed, plus 1 full-train run for the selected arm.
- **Probe:** PROBE_OK at 2026-11-17 00:49 UTC (seed 1, one fold per arm plus a full-train build of every arm), DOPE after `c51a56d0...`, peak RSS 6.3 GB, min MemAvailable 79.8 GB, incumbent val parity 1.1389572403748425. The control's full-train adapter sha `5f3acc50...` equals the H466 probe's. A first probe at 2026-11-16 23:39 UTC crashed in the probe-only full-train block, which still named the H466 arms (`KeyError: 'ctxn_z40'`); that block was fixed to the H467 arms, giving the code sha above.
- **Launch:** `waiter.sh` (started 2026-11-17 00:56 UTC, `MAXALIVE=3 SEEDS="1 2 3" MEMGATE=30`) launches seeds 1/2/3 one per 15 min when MemAvailable >= 30 GB and machine CPU busy% + 16 threads <= 85% hold on 2 consecutive checks. It relaunches a dead seed (any cause), resume-safe via `results/fold_sig`, at most 5 times per seed. It never has more than 3 seeds alive (MemAvailable was ~82 GB at launch) and SIGTERMs only its own seed process if it hangs more than 10 min after `summary.json`. Expected close is about 12:00-15:00 UTC Nov 17 (H466 took ~10 h per seed without contention; `acc2_z40` costs about 1.25x the control).

## Aggregation (when every seed has `results/seedN/summary.json`)
Copy the aggregator and red-team scripts from H466, then run `WAVE=H467 python3 h467_aggregate.py aggregate.json` and `ARMS=pairnf_z40,acc2_z40,look4_z40,bls10_z40 python3 h467_redteam.py redteam.json`.

Raw logs, losses and fold signatures stay on xbabe2 in `~/jope-data/hwave-h467`.
