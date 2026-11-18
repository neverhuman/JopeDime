# H469: robust per-context losses (trimmed mean, soft Cauchy) on the gated pairnf step (offline, adapter-only)

**Status: `RUNNING` (pre-registration record).** This README records the design, code hash and launch before any CV result exists. The verdict, metrics and red-team will be added when all 3 seeds write `summary.json`.

## Hypothesis
H466-H468 (`NO_LIFT`) showed that the gradient-noise and objective-shape levers tried so far (accumulation ×2/×4, Lookahead matched or not, ctxnorm, tail clip, pair weight 2, BT label smoothing) never moved the ~58% non-floor win rate of the kept changes. One untested explanation: a few very high-loss contexts per batch (mostly gate-fail / −2.0 floor rows, whose pointwise and listwise targets sit at 2.0) dominate the 16-step update. If that holds, a **robust per-context loss** that caps their pull should make the kept changes more precise on non-floor contexts. Every arm shares the control's random streams (`rng_arm = pairnf`), 16 steps, lr 3e-4, z ≥ 0.40 gate unchanged:
- `pairnf_z40`: the control, H452-H468 pairnf + z ≥ 0.40. Adapter shas must equal the H468 `pairnf_z40` cells (checked on the smoke cell: h452a fold 0 seed 1 sha and OOF vector identical).
- `trim10_z40`: trimmed mean. Each step drops the 10% highest per-context losses (ranking on detached losses) and averages the rest.
- `trim25_z40`: trimmed-mean dose, dropping the top 25%.
- `cauchy_z40`: soft robust loss. row → m · log1p(row / m) with m = detached batch median of the per-context losses, so a context at k × the median gets gradient weight 1 / (1 + k) instead of 1 (no hard cutoff).

Not repeated (already falsified on this instrument): Polyak/EMA tail average and ASAM (H457), 32-step dose and fbal (H461), recalibrated gate and cosine dose (H463), ctxnorm, tail clip and pair weight 2 (H466), acc2, un-matched Lookahead and BT label smoothing (H467), acc4 and displacement-matched Lookahead (H468). Modality dropout is not a lever: under the locked harness the adapter's predictions are bit-identical for dropout p ∈ {0, 0.10, 0.25, 0.5, 0.99} (H469 design check, `design_check_mdrop/`), so it was dropped from the arm list.

## Pre-registered rules (unchanged from H452-H468)
- **Primary metric:** median across 3 seeds of the pooled OOF Elo vs the incumbent, with 95% family-bootstrap CIs (families resampled with all their rows in both partition copies). **Non-floor Elo** (contexts where the incumbent pick passes the gate) is co-primary and drives selection.
- **Selection** uses Elo only: per seed, the max mean non-floor OOF Elo over the two partitions (ties broken by pooled gain).
- **Strict:** pooled OOF gain > 0.002 and gain CI low > 0; pooled Elo > 0 and Elo CI low > 0; non-floor Elo CI low > 0; Elo > 0 on each partition. Ties never win.
- An arm counts only if it also beats the control in the paired family bootstrap (arm − control CI low > 0).
- **LIFT** requires a non-control arm that passes strict and beats the control (paired CI above 0) on every seed — i.e. strict-pass CIs exclude 0 across all 3 seeds — plus a clean red-team and val91 not_worse on every seed. Otherwise the verdict is `NO_LIFT`.

## Pre-registered red-team checks (from the smoke)
- **Floor-recovery confound (−2.0 dominance).** On the smoke cell `trim25_z40` had pooled gain +0.0085 but Elo −1.4 (W 2 / L 3) and non-floor gain −0.0002: the whole gain came from recovering a single gate-fail (floor) context, which is worth up to ~3.1 utility on its own. Pooled utility gain is therefore **not** evidence for these arms. An arm whose pooled gain is positive but whose non-floor Elo does not beat the control's is reported as floor-driven, not as lift; the closeout reports non-floor gain/Elo, `recovered_gatefails` and `new_gatefails` per arm and seed, and the gain with recovered-floor contexts excluded.
- **Displacement confound.** Robust losses shrink the update (smoke kept-change counts: control 15, trim10 15, trim25 5, cauchy 4 on the same cell). If an arm keeps less than half the control's changes, a null is reported as "gate-starved", not as evidence against robust losses (cf. H467 look4).
- **No label leakage.** Trimming and the Cauchy scale use only the detached losses of the current training batch, drawn from the training folds; held-out folds are scored by the label-free z-gate, and val91 is read once per seed for the selected arm only.

## Setup
- **Code:** `h469_trainfold_cv.py`, sha256 `36a6d776a492cc5354d8b2fa6f0cd7af24db0a40e0deb367cc77bf261429346e` (checked with `sha256sum -c` by the waiter before every launch). It differs from `h468_trainfold_cv.py` in the arm table, the `trim` / `cauchy` branches of the per-context loss, the default `--arms`, the probe's full-train arm list and the docstring. CPU only, so no GPU and no `qualification.lock`. It runs at `nice 19` with 16 threads per seed and aborts itself at MemAvailable < 15 GB.
- **Scratch:** `~/jope-data/hwave-h469` on xbabe2, a copy of `round-3175.pt`, never the live run directory. The run never touches xbabe0 or xbabe3.
- **Data:** the real S3 packet: 1,219 DL-Train contexts (205 families) and 91 DL-Val contexts (19 families), family overlap 0 (asserted). No sealed or holdout data is read (the sealed meta-test group is not in the packet).
- **DOPE frozen:** composite sha `c51a56d0...` is checked before and after. Only the 29,211 adapter parameters train.
- **Partitions:** the two H452 floor-stratified, family-grouped 5-fold partitions (`h452a/`, `h452b/`), 4 arms × 2 partitions × 5 folds per seed, plus 1 full-train run for the selected arm.
- **Smoke (real config, partition `h452a/` fold 0 seed 1, no val91):** control changed 134 / kept 15, W 10 / L 4, gain +0.00063 (identical to H468); `trim10_z40` 139 / 15, W 9 / L 5, gain −0.00016 (Elo +5.7); `trim25_z40` 132 / 5, W 2 / L 3, gain +0.0085, non-floor gain −0.0002, 1 recovered gate-fail (Elo −1.4); `cauchy_z40` 127 / 4, W 1 / L 2, gain −0.00012 (Elo −1.4). About 820-860 s per fold per arm. DOPE intact after both smokes, peak RSS 6.3 GB, min MemAvailable 52 GB. Single-fold smoke numbers are not evidence and are not used for selection.
- **Probe:** PROBE_OK at 2026-11-18 08:04 UTC (seed 1, one 1-step fold per arm plus a full-train build of every arm), DOPE after `c51a56d0...`, peak RSS 6.3 GB, min MemAvailable 63.4 GB, incumbent val parity 1.1389572403748425. The control's full-train adapter sha `5f3acc50...` equals the H467/H468 probes'.
- **Launch:** `waiter.sh` (started 2026-11-18 08:35 UTC, `MAXALIVE=3 SEEDS="1 2 3" MEMGATE=30`) launches seeds 1/2/3 one per 15 min when MemAvailable >= 30 GB and machine CPU busy% + 16 threads <= 85% hold on 2 consecutive checks. It relaunches a dead seed (any cause), resume-safe via `results/fold_sig`, at most 5 times per seed, never has more than 3 seeds alive, and SIGTERMs only its own seed process if it hangs more than 10 min after `summary.json`. Expected cost ≈ 40 fold-trainings × ~14 min ≈ 10 h per seed; expected close about 19:00-22:00 UTC Nov 18.

## Aggregation (when every seed has `results/seedN/summary.json`)
Copy `h468_aggregate.py` and `h468_redteam.py`, then run `WAVE=H469 python3 h469_aggregate.py aggregate.json` and `ARMS=pairnf_z40,trim10_z40,trim25_z40,cauchy_z40 python3 h469_redteam.py redteam.json`, and add the floor-recovery split above.

Raw logs, losses and fold signatures stay on xbabe2 in `~/jope-data/hwave-h469`.
