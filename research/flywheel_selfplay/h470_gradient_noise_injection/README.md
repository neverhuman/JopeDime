# H470: annealed Gaussian gradient-noise injection on the gated pairnf step (offline, adapter-only)

**Verdict: `NO_LIFT`.** Gradient noise does not beat the `pairnf_z40` control on any seed. Both arms are *worse* than the control in the paired family bootstrap: gn05 has a CI below 0 on seed 2 (spans 0 on seeds 1 and 3), and gn10 has a CI below 0 on all 3 seeds. Median OOF Elo vs the incumbent: control **+4.7**, `gn05_z40` **+5.3**, `gn10_z40` **+4.4**. Every per-seed Elo CI except two gn10 cells (and control seed 2) reaches 0 or below. Median OOF gain: control +0.0056, gn05 +0.0044, gn10 +0.0024. Strict passes: control 1/3, gn05 0/3, gn10 0/3. CV selected gn05 / control / gn05. val91 median gain **+0.0005** (Elo +7.6). Pooling the three selected-arm val91 vectors gives a cross-seed family-bootstrap CI of **[−0.0007, +0.0018]** (W/T/L 10/259/4), so the median is not beyond its CI. not_worse 2/3: seed 1 gn05 gives −0.0000002 with 3 W / 86 T / 2 L, so Elo is +3.8 even though the gain is below 0.

**Mechanism:** the noise only *shrinks* the update. In all 12 seed × partition cells, every decision a gn arm keeps is also kept by the control (arm-only gain = 0). The noise lowers the z-margins of changed picks (median |z| 0.168 / 0.158 vs 0.177), so the z ≥ 0.40 gate drops 13–16% (gn05) or 30–36% (gn10) of the control's kept changes. Those dropped changes are net positive, which is why the paired gains are negative. The dose is monotone (gn10 differs from the control on 77–85 decisions vs 34–43 for gn05), so the noise is active, not inert. Neither arm is gate-starved (kept ≥ 0.5× control).

> **Label collision note.** An unrelated, separately owned study (DOPE value probe, local branch `research/h470-dope-value-probe`, `research/flywheel_selfplay/h470_dope_value_probe/`, `run_h470_probe.py`) also calls itself "H470". It was not on `origin` or in this index when this pre-registration was filed. In this index **H470 = this gradient-noise wave** (next number after H469; its running scratch dir, script and waiter are already named `h470`). The DOPE value probe should be indexed as **H470-DVP** (or the next free H-number) when it lands. Both studies share the scratch dir `~/jope-data/hwave-h470` on xbabe2: the DVP run writes `logs/`, `cache/` and `results/{metrics,verdict}.json` / `results/SUMMARY.md`; this wave writes only `probe/`, `smoke/`, `results/seedN/`, `results/fold_sig/`, `inc_analysis.json` and `waiter.log`. The filenames do not overlap, and the H470 aggregator reads only `results/seed*/summary.json` and `results/fold_sig/`.

## Outcome
**CV** (pooled OOF over 2 × 1,219 TRAIN contexts; values are seeds 1 / 2 / 3, median in parentheses):

| Arm | OOF gain | Elo vs incumbent | Elo 95% CI (family bootstrap) | Non-floor Elo | W / L (ties rest of 2,438) | Strict (utility-strict) | Recovered / new gate fails (of 38) | vs control (gain, CI) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **pairnf_z40** (control) | +0.0048 / +0.0069 / +0.0056 (+0.0056) | +4.7 / +6.6 / +3.8 (+4.7) | [−1.3, +10.6] / [+0.7, +12.3] / [−1.8, +9.4] | +4.3 / +5.9 / +3.3 | 122/89, 128/82, 115/88 | 1/3 (3/3) | 3, 5, 4 / 0 | — |
| gn05_z40 | +0.0038 / +0.0046 / +0.0044 (+0.0044) | +5.6 / +5.3 / +4.0 (+5.3) | [0.0, +11.0] / [0.0, +10.3] / [−1.0, +9.0] | +5.4 / +4.9 / +3.6 | 111/72, 107/70, 102/74 | 0/3 (2/3) | 2, 3, 3 / 0 | −0.0010 [−0.0033, +0.0003]; −0.0023 [−0.0055, −0.0001]; −0.0012 [−0.0034, +0.0003] |
| gn10_z40 | +0.0024 / +0.0028 / +0.0023 (+0.0024) | +4.4 / +5.0 / +3.1 (+4.4) | [+0.3, +8.7] / [+0.3, +9.7] / [−1.2, +7.6] | +4.3 / +4.9 / +3.0 | 83/52, 91/56, 79/57 | 0/3 (0/3) | 1, 1, 1 / 0 | −0.0024 [−0.0054, −0.0001]; −0.0041 [−0.0094, −0.0005]; −0.0032 [−0.0069, −0.0004] |

**Cross-seed family bootstrap** (per-context OOF gain averaged over the 3 seeds; 205 families resampled, 2,000 draws; Elo pooled over 3 × 2,438 rows):

| Arm | Gain vs incumbent [CI] | Elo vs incumbent [CI] | Gain vs control [CI] | Elo vs control [CI] (W/T/L vs control) |
| --- | --- | --- | --- | --- |
| pairnf_z40 | +0.0057 [+0.0013, +0.0114] | +5.0 [−0.7, +10.5] | — | — |
| gn05_z40 | +0.0043 [+0.0004, +0.0096] | +4.9 [−0.4, +9.8] | −0.0015 [−0.0036, −0.0000] | −0.4 [−1.6, +0.8] (52/7201/61) |
| gn10_z40 | +0.0025 [−0.0003, +0.0057] | +4.2 [−0.2, +8.3] | −0.0032 [−0.0068, −0.0005] | −1.1 [−3.5, +1.4] (111/7068/135) |

**val91 regression check** (the CV-selected arm trained on all 1,219 TRAIN contexts, scored once):

| Seed | Selected arm | Gain | Gain CI | W / T / L | Elo | Changed / kept | New gate failures | not_worse |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | gn05_z40 | −0.0000 | [−0.0009, +0.0008] | 3/86/2 | +3.8 | 53 / 5 | 0 | false |
| 2 | pairnf_z40 | +0.0011 | [−0.0005, +0.0028] | 4/86/1 | +11.5 | 54 / 5 | 0 | true |
| 3 | gn05_z40 | +0.0005 | [−0.0008, +0.0020] | 3/87/1 | +7.6 | 55 / 4 | 0 | true |
| pooled | — | +0.0005 (median) | [−0.0007, +0.0018] (19-family bootstrap) | 10/259/4 | +7.6 [−8.0, +25.5] | — | 0 | — |

Val91 is only scored for each seed's selected arm (pre-registered: one read per seed), so per-arm val91 medians do not exist for the arms that were not selected. The val91 W/L picks are almost seed-invariant: the same 4 contexts (idx 50, 64, 69, 84) change in every seed.

## Red-team (`h470_redteam.py`, `h470_aggregate.py`, copied from H469; full output under `aggregate` / `redteam` in `metrics.json`)
1. **Control reproduction.** All 30 `pairnf_z40` cells have the same adapter sha and identical OOF vectors as H469 (30/30).
2. **Leakage.** The noise uses training-batch gradients and its own keyed generator only. Held-out folds are scored with the label-free z-gate. Val91 was read once per seed for the selected arm only. The packet is DL-Train/DL-Val only (asserted), family overlap 0 (asserted), and the partitions were rebuilt and matched the logged fold maps (floors 4/4/4/3/4). **0 sealed or holdout reads.**
3. **Seed noise and identical-value coincidences.** Fold-gain vectors differ between every seed pair for every arm, and adapter shas are distinct in every cell (asserted). Seeds disagree on 18–32 OOF decisions per partition. Two equal non-floor Elos are arithmetic coincidences, not no-ops: seed 1 control vs gn10 (+4.343, both W−L = +30 of 2,400) and seed 2 gn05 vs gn10 (+4.922, both W−L = +34). Elo depends only on (W + T/2)/N, and the adapters and decisions differ. Two fold cells have identical OOF vectors (one control seed pair and one control–gn05 pair with the same seed), each with a different adapter sha. The gate kept the same changed set there.
4. **Metric direction.** Utility = −fitness of the argmin-predicted program (−2.0 on gate fail), so higher is better. Gain = arm − incumbent, and Elo uses W = gain > 0. Seed 1 val91 shows why gain and Elo are both reported: 3 W / 2 L gives Elo +3.8 while the gain is −2e−7, which is not_worse = false.
5. **Cherry-picking / selection.** Selection is pre-registered (max mean non-floor Elo). The selection bootstrap picks the same argmax in only 68% / 72% / 41% of resamples, so the gn05 picks on seeds 1 and 3 are near-coin-flips between gn05 and the control. The verdict does not depend on the pick, because no arm beats the control.
6. **Tie acceptance.** About 91–94% of OOF decisions and 86–87 of 91 val contexts are ties, since the z-gate keeps the incumbent pick. Ties never win and are never accepted as lift. The seed 1 gn05 val91 with 86/91 ties and gain ≈ 0 is scored not_worse = false.
7. **Floor recovery.** Noise recovers fewer −2.0 floors than the control (gn10 recovers 0 on h452a in every seed), so no gain is floor-driven. 0 new gate failures in any cell.
8. **DOPE freeze.** `c51a56d0…` was the same before and after on all 3 seeds. Only the 29,211 adapter params trained. The code sha `d596dbb2…` was re-verified after the run, and the checkpoint copy sha is `1d6ea900…`.
9. **OOM / disk.** 0 relaunches, 0 own-guard aborts, 0 OOM kills. Peak RSS was 6.2 GB per seed and min MemAvailable 34.1 / 35.1 / 34.2 GB. Root disk was at 96% (≈360 GB free). Nothing large is committed.
10. **H470-DVP separation.** The separately owned DOPE value probe (`run_h470_probe.py`, logs in `~/jope-data/hwave-h470/logs/`, still running at closeout) is not in any H470 number. The aggregator reads only `results/seed*/summary.json` and `results/fold_sig/`.

## Next step
Gradient-noise injection (H470) and noise reduction (H467/H468) both failed. Every arm since H452 either reproduces the control's kept changes or keeps a subset of them. The one arm that ever beat the control on a seed was H463 `s32cos_z40` (a superset of the control's changes, paired CI > 0 on seed 1). H471 tests whether that comes from the **warmup + cosine schedule shape** rather than the extra 16 steps. It uses 16-step schedules whose integrated learning rate equals the control's: a one-cycle (2-step warmup, cosine to 0, peak 6e-4) and two 8-step cosine restarts (SGDR, 1-step warmup each, peak 5.33e-4), both on the control's streams.

## Hypothesis (pre-registered in #2373)
H466-H469 (`NO_LIFT`) tried gradient-noise **reduction** (accumulation ×2/×4, Lookahead matched or not), objective shape (ctxnorm, tail clip, pair weight 2, BT label smoothing) and robust per-context losses (trim10/trim25/Cauchy). None moved the ~58% non-floor win rate of the kept changes. H470 tests the opposite direction, which H443-H469 never tried: **annealed Gaussian gradient-noise injection** (Neelakantan et al. 2015). The idea is that a little extra exploration in the 16-step update escapes the sharp, seed-specific directions the z-gate currently keeps, so the kept changes generalize better to held-out families. Every arm shares the control's random streams (`rng_arm = pairnf`), and the noise is drawn from its own keyed generator `key(seed, rng_arm, fold, step, "gnoise")` (never the global or dropout stream, so the control's streams are untouched). The runs use 16 steps, lr 3e-4 and the unchanged z ≥ 0.40 gate:
- `pairnf_z40`: the control, H452-H469 pairnf + z ≥ 0.40. Adapter shas must equal the H469 `pairnf_z40` cells (checked on the smoke cell: h452a fold 0 seed 1, adapter sha `8d5ac793…` and OOF vector identical to H469 `smoke_a`; probe full-train sha `5f3acc50…` = H467/H468/H469 probes).
- `gn05_z40`: after grad-norm clipping (1.0), each trainable tensor gets g += N(0, (c · rms(g_tensor))²) / (1 + step)^0.55 with c = 0.5 (scale-free per tensor, annealed).
- `gn10_z40`: dose, c = 1.0.

Not repeated (already falsified on this instrument): Polyak/EMA and ASAM (H457), 32-step dose and fbal (H461), recalibrated gate and cosine dose (H463), ctxnorm, tail clip and pair weight 2 (H466), acc2, un-matched Lookahead and BT label smoothing (H467), acc4 and displacement-matched Lookahead (H468), trim10/trim25/Cauchy (H469). Modality dropout is inert under the locked harness (H469 design check).

## Pre-registered rules (unchanged from H452-H469)
- **Primary metric:** median across 3 seeds of the pooled OOF Elo vs the incumbent, with 95% family-bootstrap CIs (families resampled with all their rows in both partition copies). **Non-floor Elo** (contexts where the incumbent pick passes the gate) is co-primary and drives selection. Elo only, no AUC.
- **Selection** uses Elo only: per seed, the max mean non-floor OOF Elo over the two partitions (ties broken by pooled gain).
- **Strict:** pooled OOF gain > 0.002 (strict margin) and gain CI low > 0; pooled Elo > 0 and Elo CI low > 0; non-floor Elo CI low > 0; Elo > 0 on each partition. Ties never win, and a tie with the control is not accepted.
- An arm counts only if it also beats the control in the paired family bootstrap (arm − control CI low > 0).
- **LIFT** requires a non-control arm that passes strict and beats the control (paired CI above 0) on every seed. That means the median gain must sit beyond the CI on all 3 seeds. It also needs a clean red-team and val91 not_worse on every seed (val91 gain reported per seed with family-bootstrap CI). Otherwise the verdict is `NO_LIFT`.

## Pre-registered red-team checks
- **Floor-recovery confound.** Pooled gain that comes from recovering −2.0 gate-fail contexts is not lift. The closeout reports non-floor gain/Elo, `recovered_gatefails` and `new_gatefails` per arm and seed.
- **Displacement / gate-starvation confound.** Noise can shrink z-margins. Smoke kept counts on the same cell: control 15, gn05 13, gn10 11. An arm that keeps < 0.5× the control's changes is reported as gate-starved, not as evidence against noise injection.
- **Dose check.** If gn10 is not at least as different from the control as gn05 (changed sets, kept counts, |z| distribution), the noise is reported as inert at this scale (cf. H469 modality dropout).
- **No label leakage.** The noise uses only training-batch gradients and its own keyed RNG. Held-out folds are scored by the label-free z-gate, and val91 is read once per seed for the selected arm only.

## Setup
- **Code:** `h470_trainfold_cv.py`, sha256 `d596dbb25d6adcbe0911f0add13bb3bed5eae80db2ff3944fffc181b75884352` (checked with `sha256sum -c` by the waiter before every launch). It differs from `h469_trainfold_cv.py` in the arm table (`gnoise` key), the gradient-noise branch after `clip_grad_norm_`, the default `--arms`, the probe's full-train arm list and the docstring. CPU only, so no GPU and no `qualification.lock`. It runs at `nice 19` with 16 threads per seed and aborts itself at MemAvailable < 15 GB.
- **Scratch:** `~/jope-data/hwave-h470` on xbabe2 with a copy of `round-3175.pt` (sha256 `1d6ea900…`, `admission == incumbent`, asserted), never the live run directory. The run never touches xbabe0 or xbabe3.
- **Data:** the real S3 packet: 1,219 DL-Train contexts (205 families) and 91 DL-Val contexts (19 families), family overlap 0 (asserted), `dl_split ∈ {DL-Train, DL-Val}` (asserted). No sealed or holdout data is read (the sealed meta-test group is not in the packet).
- **DOPE frozen:** composite sha `c51a56d0...` is checked before and after. Only the 29,211 adapter parameters train.
- **Partitions:** the two H452 floor-stratified, family-grouped 5-fold partitions (`h452a/`, `h452b/`, fold/family disjointness asserted), 3 arms × 2 partitions × 5 folds per seed, plus 1 full-train run and one val91 score for the selected arm.
- **Smoke (real config, partition `h452a/` fold 0 seed 1, no val91; SMOKE_OK 2026-11-18 21:16 UTC / 2:16 PM MST):** control changed 134 / kept 15, W 10 / L 4, gain +0.00063, Elo +8.5 (identical to H469); `gn05_z40` 131 / 13, W 8 / L 4, gain +0.00049, Elo +5.7; `gn10_z40` 131 / 11, W 7 / L 3, gain +0.00054, Elo +5.7. 0 new / 0 recovered gate failures. About 810-830 s per fold per arm. DOPE intact after the smoke, peak RSS 6.2 GB, min MemAvailable 45 GB. Single-fold smoke numbers are not evidence and are not used for selection.
- **Probe:** PROBE_OK at 2026-11-18 20:48 UTC / 1:48 PM MST (seed 1, one 1-step fold per arm plus a full-train build of every arm), DOPE after `c51a56d0...`, peak RSS 6.2 GB, min MemAvailable 45 GB, incumbent val parity 1.1389572403748425. The control's full-train adapter sha `5f3acc50...` equals the H467-H469 probes'; gn05 `f42771ef…` and gn10 `c5885e38…` differ.
- **Launch:** `waiter.sh` (started 2026-11-18 20:59 UTC / 1:59 PM MST, `SEEDS="1 2 3" MEMGATE=30`, default `MAXALIVE=2`) launches seeds one per 15 min when MemAvailable ≥ 30 GB and machine CPU busy% + 16 threads ≤ 85% hold on 2 consecutive checks. Seed 1 launched 21:04 UTC (2:04 PM MST). It relaunches a dead seed (any cause), resume-safe via `results/fold_sig`, at most 5 times per seed, never has more than 2 seeds alive, and SIGTERMs only its own seed process if it hangs more than 10 min after `summary.json`. Expected cost ≈ 30 fold-trainings × ~14 min ≈ 7-8 h per seed. Seed 3 waits for a free slot, so the expected close is about 12:00-15:00 UTC Nov 19 (5-8 AM MST).
- **Runs:** seeds 1/2/3 launched 2026-11-18 21:04 / 21:25 / 21:45 UTC (2:04 / 2:25 / 2:45 PM MST). All three finished without a relaunch: DONE at 2026-11-19 04:30 / 04:49 / 05:10 UTC (9:30 / 9:49 / 10:10 PM MST Nov 18).

Raw logs, losses and fold signatures stay on xbabe2 in `~/jope-data/hwave-h470`.
