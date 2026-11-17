# H467: gradient noise on the gated pairnf step (offline, adapter-only)

**Verdict: `NO_LIFT`.** No gradient-noise arm beats the `pairnf_z40` control on every seed. `acc2_z40` (2 × 1,024-context gradient accumulation) has the best median, Elo **+5.7** vs the control's **+4.7**, but strict passes on 0/3 seeds (Elo CI low −0.9 / −0.3 / −0.1). Its paired gain vs the control is +0.0025 [+0.0002, +0.0070] / −0.0006 [−0.0029, +0.0010] / +0.0005 [−0.0000, +0.0010], so the CI clears 0 on seed 1 only. `look4_z40` (Lookahead k = 4, α = 0.5) is **near-degenerate**: it halves the displacement, so the z ≥ 0.40 gate keeps 1 change in 2,438 (Elo +0.1). That is a design confound, not a code bug. `bls10_z40` (BT label smoothing 0.1) is worse than the control: Elo +2.7 on every seed, with the paired CI below 0 on seeds 2 and 3. CV selected `acc2_z40` / `pairnf_z40` / `acc2_z40`. The val91 regression check gives a median gain of **+0.0006** (Elo +3.8), every CI spans 0, and seed 3 (acc2, 3 W / 3 L, −0.0002) is **not_worse = false**. No breakthrough.

## Hypothesis (pre-registered in #2352)
H457, H461, H463 and H466 moved how many changes the z ≥ 0.40 gate keeps but never the ~57–58% non-floor win rate of the kept changes. H467 keeps the dose (16 steps, lr 3e-4), the gate and the objective fixed and attacks the **gradient noise** of the step. Every arm runs on the control's random streams (`rng_arm = pairnf`): `acc2_z40` (effective batch 2,048), `look4_z40` (Lookahead k4 α0.5) and `bls10_z40` (pair targets 0.9 / 0.1).

## Pre-registered rules (unchanged from H452–H466)
Selection takes the max mean non-floor OOF Elo over the two partitions. Strict requires pooled OOF gain > 0.002 with gain CI low > 0, pooled Elo > 0 with Elo CI low > 0, non-floor Elo CI low > 0, and Elo > 0 on each partition. An arm counts only if it also beats the control in the paired family bootstrap. Ties never win. LIFT needs that on every seed, plus val91 not_worse on every seed and a clean red-team.

## Setup
- **Code:** `h467_trainfold_cv.py` as pre-registered (sha256 `5d011b8c…`, re-verified by the waiter before every launch and after the run). CPU only, `nice 19`, 16 threads per seed, on the scratch copy `~/jope-data/hwave-h467` on xbabe2 (a copy of `round-3175.pt`). Never the live run dir, never xbabe0 or xbabe3. No GPU, so no `qualification.lock`.
- **Runs:** seeds 1/2/3 launched 2026-11-17 01:01 / 01:22 / 01:42 UTC and finished (DONE) at 12:53 / 13:01 / 13:32 UTC. There were 0 guard aborts and 0 relaunches. The waiter exited cleanly at 13:33 UTC.
- **Data and model:** the real S3 packet, 1,219 DL-Train contexts (205 families) and 91 DL-Val contexts (19 families), family overlap 0. DOPE `c51a56d0…` stayed frozen, and only the 29,211 adapter params trained. Incumbent val parity is 1.1389572403748425 on every seed.
- **Compute:** peak RSS was 6.3 GB per seed and min MemAvailable 48.8 / 48.9 / 49.5 GB. acc2 costs about 2× wall time (15.9k s vs 8.2k s per seed).

## Outcome
**CV** (pooled OOF over 2 × 1,219 TRAIN contexts; values are seeds 1 / 2 / 3, median in parentheses):

| Arm | OOF gain | Elo vs incumbent | Elo 95% CI (family bootstrap) | Non-floor Elo | W / L (ties rest of 2,438) | Strict (utility-strict) | Recovered / new gate fails (of 38) | vs control (gain, CI) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **pairnf_z40** (control) | +0.0048 / +0.0069 / +0.0056 (+0.0056) | +4.7 / +6.6 / +3.8 (+4.7) | [−1.3, +10.6] / [+0.7, +12.3] / [−1.8, +9.4] | +4.3 / +5.9 / +3.3 | 122/89, 128/82, 115/88 | 1/3 (3/3) | 3, 5, 4 / 0 | — |
| acc2_z40 | +0.0073 / +0.0063 / +0.0061 (+0.0063) | +5.3 / +6.0 / +5.7 (+5.7) | [−0.9, +11.2] / [−0.3, +12.4] / [−0.1, +11.9] | +4.6 / +5.5 / +5.2 | 131/94, 130/88, 127/87 | 0/3 (3/3) | 5, 4, 4 / 0 | +0.0025 [+0.0002, +0.0070]; −0.0006 [−0.0029, +0.0010]; +0.0005 [−0.0000, +0.0010] |
| look4_z40 | +0.00003 ×3 (+0.00003) | +0.1 ×3 (+0.1) | [0.0, +0.5] ×3 | +0.1 ×3 | 1/0 ×3 | 0/3 (0/3) | 0 / 0 | −0.0048 [−0.0098, −0.0005]; −0.0068 [−0.0137, −0.0019]; −0.0056 [−0.0113, −0.0011] |
| bls10_z40 | +0.0036 / +0.0037 / +0.0032 (+0.0036) | +2.7 / +2.7 / +2.7 (+2.7) | [−3.6, +9.0] / [−3.7, +9.3] / [−3.5, +8.7] | +2.5 / +2.3 / +2.3 | 129/110, 128/109, 128/109 | 0/3 (0/3) | 2, 3, 3 / 0, 1, 1 | −0.0012 [−0.0036, +0.0006]; −0.0032 [−0.0069, −0.0003]; −0.0024 [−0.0055, −0.0003] |

Head-to-head on contexts where the arm and the control differ:
- acc2_z40 (29 / 35 / 33): arm better 17 / 18 / 22, control better 12 / 17 / 11 (Elo +60.5 / +9.9 / +120.4, every CI spans 0).
- look4_z40 (211 / 209 / 202): arm better 89 / 82 / 88, control better 122 / 127 / 114 (it reverts almost all of the control's kept changes).
- bls10_z40 (72 / 80 / 74): arm better 30 / 30 / 35, control better 42 / 50 / 39 (Elo −58.5 / −88.7 / −18.8).

**val91 regression check** (the CV-selected arm trained on all 1,219 TRAIN contexts):

| Seed | Selected arm | Gain | Gain CI | W / T / L | Elo | Changed / kept | New gate failures | not_worse |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | acc2_z40 | +0.0006 | [−0.0007, +0.0022] | 3/86/2 | +3.8 | 54 / 5 | 0 | true |
| 2 | pairnf_z40 | +0.0011 | [−0.0005, +0.0028] | 4/86/1 | +11.5 | 54 / 5 | 0 | true |
| 3 | acc2_z40 | −0.0002 | [−0.0020, +0.0017] | 3/85/3 | 0.0 | 54 / 6 | 0 | false |

Across seeds, the median val91 gain is +0.0006 and the median Elo +3.8. Every CI spans 0. The pre-registered LIFT rule fails: no arm beats the control on every seed, acc2 passes strict on 0/3 seeds, and seed 3 is not not_worse.

## Red-team (`h467_redteam.py`, `h467_aggregate.py`, copied from H466; output under `redteam` / `aggregate` in `metrics.json`)
1. **Control reproduction.** All 30 `pairnf_z40` cells have the same adapter sha and identical OOF vectors as H466 (30/30). The seed-2 full-train control adapter is again `be56fa29…`.
2. **look4 confound (mechanism, not a bug).** Lookahead k4 α0.5 pulls the fast weights halfway back every 4 steps, so the 16-step move is roughly halved. |z| of changed decisions has p50 / p90 / p99 / max 0.073 / 0.225 / 0.332 / 0.414 vs the control's 0.177 / 0.437 / 0.599 / 0.790, so only 0.1% clear the 0.40 gate (control 14.8%). The aggregator's seed-independence assert was extended to exempt arms that keep ≤ 5 of 2,438 decisions on every seed (look4: 1 W / 0 L, legitimately seed-identical). Distinct adapter shas per cell are still asserted. A displacement-matched Lookahead is the obvious follow-up (H468).
3. **acc2 (mechanism).** acc2 keeps 104–117 changes per partition vs the control's 99–108, with 98–108 shared. Its non-floor pass win rate is 58.2% (375 W / 269 L over 6 cells), the same as the control's 57.7% (353 / 259). It adds a few kept changes at the same win rate. It does not raise the win rate.
4. **bls10 (mechanism).** Label smoothing slightly widens the z-margins (p90 0.459), so the gate keeps 115–124 changes. The extras skew toward losses (non-floor pass W/L 60–65 / 50–60), and it adds 1 new gate failure on seeds 2 and 3.
5. **Seed noise.** Seed disagreement on OOF decisions per partition: control 25–32, acc2 13–29, bls10 29–40, look4 0 (degenerate). `seed_independence` holds for every non-degenerate arm × seed pair.
6. **Selection / winner's curse.** Selection used TRAIN-fold CV only (non-floor Elo), never val. The selection bootstrap picks the selected arm as argmax only 58% / 66% / 92% of the time, and the max-arm non-floor Elo CI is [0.0, +11.1] / [+0.3, +12.1] / [+0.1, +10.7].
7. **Ties.** About 91% of CV decisions and 85–86 of 91 val91 decisions are ties by design. Ties count 0.5 in Elo and never satisfy strict. The 0.002 strict margin is unchanged.
8. **Metric direction.** Utility is higher-is-better (negated fitness, −2.0 on gate fail). Gains are paired candidate − incumbent on the same contexts. Verdicts report Elo, not AUC, and the aggregator recomputes them from the per-seed fields.
9. **Leakage.** Folds are family-disjoint and exclude val families (asserted, family overlap 0). The red-team rebuilds both partitions and matches the logged fold maps. No sealed or holdout reads: the only holdout, the sealed meta-test group, was never touched.
10. **Safety.** DOPE `c51a56d0` is intact on every DONE line (before = after). There were no OOM kills and no guard aborts. No other owner's job was paused or killed. The wave scripts contain no ssh, rsync, xbabe0 or xbabe3 target. xbabe2 disk is at 94% (~463 GB free), and the wave dir is ~470 MB. No large objects are committed.

## Meaning
Gradient noise isn't the binding constraint either. Halving the gradient noise (acc2) adds a few kept changes at the same ~58% win rate, worth about +1 Elo on the median but not separable from the control. Smoothing pair targets hurts. Lookahead at a fixed lr just shrinks the move below the gate. H451–H467 agree that the gated step is worth about +4 to +6 OOF Elo and is not separable from 0 on val91.

## Next step
H468 (`~/jope-data/hwave-h468`) runs on the same instrument and the control's streams:
- `acc4_z40`: accumulation dose-response (4 × 1,024 contexts, effective batch 4,096). If acc2's +1 Elo is real, acc4 should extend it.
- `lk4lr2_z40`: displacement-matched Lookahead (k4 α0.5 at lr 6e-4, 2× from the H467 z-margin ratio), which resolves the look4 confound.

No adoption nudge was sent to the flywheel owner.

Artifacts:
- `metrics.json`: `aggregate` (fold-gain vectors dropped for size) and `redteam` as produced on xbabe2.
- `h467_trainfold_cv.py` (as run), `h467_aggregate.py`, `h467_redteam.py`.

Raw logs, losses and fold signatures stay on xbabe2 in `~/jope-data/hwave-h467`.
