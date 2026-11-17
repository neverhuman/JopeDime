# H468: accumulation dose-response and displacement-matched Lookahead on the gated pairnf step (offline, adapter-only)

**Status: `RUNNING` (pre-registration record).** This README records the design, code hash and launch before any CV result exists. The verdict, metrics and red-team will be added when all 3 seeds write `summary.json`.

## Hypothesis
H467 (`NO_LIFT`) left two open threads. `acc2_z40` (2 × 1,024-context gradient accumulation) was the only arm with a positive median vs the control (Elo +5.7 vs +4.7; paired gain +0.0025 / −0.0006 / +0.0005, CI above 0 on seed 1 only). `look4_z40` (Lookahead k = 4, α = 0.5) was a design confound, not a test: it halved the displacement, so the z ≥ 0.40 gate kept 1 change in 2,438. H468 keeps the gate (z ≥ 0.40), the objective and 16 steps fixed and resolves both. Every arm shares the control's random streams (`rng_arm = pairnf`):
- `pairnf_z40`: the control, H452-H467 pairnf + z ≥ 0.40, 16 steps, lr 3e-4. Adapter shas must equal the H467 `pairnf_z40` cells.
- `acc4_z40`: accumulation **dose-response**: 4 independent 1,024-context micro-batches per step (effective batch 4,096, mean gradient, same lr, steps and clip). Micro-batches 0/1 are acc2's draws and 2/3 use their own keyed streams. If the acc2 signal is real lower gradient noise, acc4 should beat the control on every seed with the paired CI above 0.
- `lk4lr2_z40`: **displacement-matched** Lookahead: k = 4, α = 0.5 with lr 6e-4 (2×). The 2× comes from the H467 control/look4 z-margin ratio (p90 1.94, p99 1.80, max 1.91), so the kept-set size is back near the control's and Lookahead's smoothing is tested on its own.

Not repeated (already falsified on this instrument): Polyak/EMA tail average and ASAM (H457), 32-step dose and fbal (H461), recalibrated gate and cosine dose (H463), ctxnorm, tail clip and pair weight 2 (H466), acc2 alone, un-matched Lookahead and BT label smoothing (H467).

## Pre-registered rules (unchanged from H452-H467)
- **Selection** uses Elo only: per seed, the max mean non-floor OOF Elo over the two partitions (ties broken by pooled gain).
- **Strict:** pooled OOF gain > 0.002 and gain CI low > 0; pooled Elo > 0 and Elo CI low > 0; non-floor Elo CI low > 0; Elo > 0 on each partition. CIs are 95% family bootstraps; reporting uses the median across 3 seeds.
- An arm counts only if it also beats the control in the paired family bootstrap (arm − control CI low > 0). Ties never win.
- **LIFT** requires a non-control arm that beats the control on the median across 3 seeds with the paired CI above 0, plus a clean red-team and val91 not_worse on every seed. Otherwise the verdict is `NO_LIFT`.

## Setup
- **Code:** `h468_trainfold_cv.py`, sha256 `a55c7f6f246e6980c485fe869ad18f7d2c9ea7536676f2cf0b1daaa9c0b674cc` (checked with `sha256sum -c` by the waiter before every launch). It differs from `h467_trainfold_cv.py` only in the arm table, the default `--arms`, the probe's full-train arm list, a `--smoke` mode and the docstring. `accum` and `look` reuse the H467/H457 branches. CPU only, so no GPU and no `qualification.lock`. It runs at `nice 19` with 16 threads per seed and aborts itself at MemAvailable < 15 GB.
- **Scratch:** `~/jope-data/hwave-h468` on xbabe2, a copy of `round-3175.pt`, never the live run directory. The run never touches xbabe0 or xbabe3.
- **Data:** the real S3 packet: 1,219 DL-Train contexts (205 families) and 91 DL-Val contexts (19 families), family overlap 0 (asserted). val91 is read once per seed, for the CV-selected arm only, as a regression check. No sealed or holdout data is read (the sealed meta-test group is not in the packet).
- **DOPE frozen:** composite sha `c51a56d0...` is checked before and after. Only the 29,211 adapter parameters train.
- **Partitions:** the two H452 floor-stratified, family-grouped 5-fold partitions (`h452a/`, `h452b/`), 3 trained bases × 2 partitions × 5 folds per seed, plus 1 full-train run for the selected arm.
- **Smoke (real config, partition `h452a/` fold 0 only, no val91):** both new arms ran 16 steps at the real config on 2026-11-17 and kept nonzero changes, so neither is gate-degenerate like H467 look4. `lk4lr2_z40` changed 141 / kept 26 (H467 control on the same cell: 134 / 15), W 12 / L 13, gain −0.0015, 806 s. `acc4_z40` changed 135 / kept 17 (H467 acc2: 135 / 17), W 9 / L 7, gain +0.0005, 2,973 s (≈3.6× the control). DOPE intact after both, peak RSS 6.3 GB, min MemAvailable 69.9 GB. Single-fold smoke numbers are not evidence and are not used for selection.
- **Probe:** PROBE_OK at 2026-11-17 15:01 UTC (seed 1, one 1-step fold per arm plus a full-train build of every arm), DOPE after `c51a56d0...`, peak RSS 6.3 GB, min MemAvailable 76.8 GB, incumbent val parity 1.1389572403748425. The control's full-train adapter sha `5f3acc50...` equals the H467 probe's.
- **Launch:** `waiter.sh` (started 2026-11-17 15:01 UTC, `MAXALIVE=3 SEEDS="1 2 3" MEMGATE=30`) launches seeds 1/2/3 one per 15 min when MemAvailable >= 30 GB and machine CPU busy% + 16 threads <= 85% hold on 2 consecutive checks. It relaunches a dead seed (any cause), resume-safe via `results/fold_sig`, at most 5 times per seed, never has more than 3 seeds alive, and SIGTERMs only its own seed process if it hangs more than 10 min after `summary.json`. Each seed also aborts itself at MemAvailable < 15 GB. Expected cost is ≈5.6× H467's control-only arm time per seed (acc4 ≈3.6×), so about 14-16 h per seed; expected close about 05:00-09:00 UTC Nov 18.

## Aggregation (when every seed has `results/seedN/summary.json`)
Copy `h467_aggregate.py` and `h467_redteam.py`, then run `WAVE=H468 python3 h468_aggregate.py aggregate.json` and `ARMS=pairnf_z40,acc4_z40,lk4lr2_z40 python3 h468_redteam.py redteam.json`.

Raw logs, losses and fold signatures stay on xbabe2 in `~/jope-data/hwave-h468`.
