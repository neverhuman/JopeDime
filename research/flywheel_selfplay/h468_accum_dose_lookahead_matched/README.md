# H468: accumulation dose-response and displacement-matched Lookahead on the gated pairnf step (offline, adapter-only)

**Verdict: `NO_LIFT`.** Neither arm beats the `pairnf_z40` control on every seed. `acc4_z40` (4 × 1,024-context gradient accumulation) has the best median, Elo **+5.6** vs the control's **+4.7**, but strict passes on 0/3 seeds (Elo CI low −0.9 / −0.9 / −0.7). Its paired gain vs the control is +0.0025 [+0.0001, +0.0069] / −0.0005 [−0.0029, +0.0013] / +0.0015 [−0.0001, +0.0037], so the CI clears 0 on seed 1 only: the same pattern as H467 acc2, with no dose-response. `lk4lr2_z40` (Lookahead k = 4, α = 0.5 at lr 6e-4) is no longer degenerate: it keeps 19% of changed decisions vs the control's 15%. But the extra changes win less often (non-floor pass win rate 55.8% vs 57.7%), so its median Elo is +5.1 with strict 0/3 and every paired CI spanning 0. CV selected `acc4_z40` / `pairnf_z40` / `acc4_z40`. val91: median gain **+0.0020**, Elo +11.5 on every seed, all CIs span 0 (lowest CI lows −0.0001 / −0.0005 / −0.0002), not_worse on 3/3 seeds. No breakthrough.

## Hypothesis (pre-registered in #2357)
H467 left two open threads. `acc2_z40` was the only arm with a positive median vs the control (Elo +5.7 vs +4.7, paired CI > 0 on seed 1 only), and `look4_z40` was a design confound (Lookahead halved the displacement below the z ≥ 0.40 gate). H468 keeps the gate, objective and 16 steps fixed, with every arm on the control's random streams (`rng_arm = pairnf`):
- `acc4_z40`: accumulation dose-response (effective batch 4,096). If acc2's signal were real lower gradient noise, acc4 should beat the control on every seed with the paired CI above 0.
- `lk4lr2_z40`: displacement-matched Lookahead, k4 α0.5 at lr 6e-4 (2× from the H467 z-margin ratio).

## Pre-registered rules (unchanged from H452–H467)
Selection takes the max mean non-floor OOF Elo over the two partitions. Strict requires pooled OOF gain > 0.002 with gain CI low > 0, pooled Elo > 0 with Elo CI low > 0, non-floor Elo CI low > 0, and Elo > 0 on each partition. An arm counts only if it also beats the control in the paired family bootstrap. Ties never win. LIFT needs that on every seed, plus val91 not_worse on every seed and a clean red-team.

## Setup
- **Code:** `h468_trainfold_cv.py` as pre-registered (sha256 `a55c7f6f…`, re-verified by the waiter before every launch and after the run). CPU only, `nice 19`, 16 threads per seed, on the scratch copy `~/jope-data/hwave-h468` on xbabe2 (a copy of `round-3175.pt`, sha256 `1d6ea900…`). Never the live run dir, never xbabe0 or xbabe3. No GPU, so no `qualification.lock`.
- **Runs:** seeds 1/2/3 launched 2026-11-17 15:06 / 15:27 / 15:47 UTC. Seeds 1 and 3 each hit their own 15 GB MemAvailable guard once (other owners' load, at 13.7–14.4 GB) and the waiter relaunched them resume-safe via `results/fold_sig` (21:22 / 21:42 UTC). DONE at 06:04 / 04:58 / 07:17 UTC Nov 18. The waiter exited cleanly at 07:22 UTC (00:22 MST).
- **Data and model:** the real S3 packet, 1,219 DL-Train contexts (205 families) and 91 DL-Val contexts (19 families), family overlap 0. DOPE `c51a56d0…` stayed frozen (before = after on every seed), and only the 29,211 adapter params trained. Incumbent val parity is 1.1389572403748425 on every seed.
- **Compute:** peak RSS 6.3 GB per seed. Min MemAvailable 35.5 / 15.2 / 36.6 GB (seed 2 came within 0.2 GB of its guard). acc4 costs about 3.6× the control per fold.

## Outcome
**CV** (pooled OOF over 2 × 1,219 TRAIN contexts; values are seeds 1 / 2 / 3, median in parentheses):

| Arm | OOF gain | Elo vs incumbent | Elo 95% CI (family bootstrap) | Non-floor Elo | W / L (ties rest of 2,438) | Strict (utility-strict) | Recovered / new gate fails (of 38) | vs control (gain, CI) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **pairnf_z40** (control) | +0.0048 / +0.0069 / +0.0056 (+0.0056) | +4.7 / +6.6 / +3.8 (+4.7) | [−1.3, +10.6] / [+0.7, +12.3] / [−1.8, +9.4] | +4.3 / +5.9 / +3.3 | 122/89, 128/82, 115/88 | 1/3 (3/3) | 3, 5, 4 / 0 | — |
| acc4_z40 | +0.0072 / +0.0064 / +0.0071 (+0.0071) | +5.6 / +5.3 / +5.7 (+5.6) | [−0.9, +11.8] / [−0.9, +11.3] / [−0.7, +12.2] | +4.9 / +4.8 / +5.1 | 133/94, 131/94, 133/93 | 0/3 (3/3) | 5, 4, 5 / 0 | +0.0025 [+0.0001, +0.0069]; −0.0005 [−0.0029, +0.0013]; +0.0015 [−0.0001, +0.0037] |
| lk4lr2_z40 | +0.0072 / +0.0074 / +0.0064 (+0.0072) | +5.1 / +6.0 / +4.3 (+5.1) | [−1.6, +11.5] / [−1.0, +12.6] / [−2.2, +10.7] | +4.5 / +5.4 / +3.6 | 156/120, 157/115, 150/120 | 0/3 (3/3) | 5, 5, 5 / 0 | +0.0024 [−0.0001, +0.0068]; +0.0005 [−0.0004, +0.0016]; +0.0008 [−0.0007, +0.0030] |

Head-to-head on contexts where the arm and the control differ:
- acc4_z40 (32 / 45 / 49): arm better 19 / 20 / 28, control better 13 / 25 / 21 (Elo +65.9 / −38.8 / +50.0, every CI spans 0).
- lk4lr2_z40 (77 / 73 / 86): arm better 41 / 32 / 46, control better 36 / 41 / 40 (Elo +22.6 / −43.1 / +24.3, every CI spans 0).

**val91 regression check** (the CV-selected arm trained on all 1,219 TRAIN contexts):

| Seed | Selected arm | Gain | Gain CI | W / T / L | Elo | Changed / kept | New gate failures | not_worse |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | acc4_z40 | +0.0021 | [−0.0001, +0.0049] | 5/84/2 | +11.5 | 54 / 7 | 0 | true |
| 2 | pairnf_z40 | +0.0011 | [−0.0005, +0.0028] | 4/86/1 | +11.5 | 54 / 5 | 0 | true |
| 3 | acc4_z40 | +0.0020 | [−0.0002, +0.0047] | 5/84/2 | +11.5 | 54 / 7 | 0 | true |

Across seeds the median val91 gain is +0.0020 and the median Elo +11.5. Every CI spans 0. The pre-registered LIFT rule fails: no arm beats the control on every seed, and the selected arm passes strict on 0/3 seeds.

## Red-team (`h468_redteam.py`, `h468_aggregate.py`, copied from H467; output under `redteam` / `aggregate` in `metrics.json`)
1. **Control reproduction.** All 30 `pairnf_z40` cells have the same adapter sha and identical OOF vectors as H467 (30/30).
2. **Why seeds 1 and 2 gave the identical acc4_z40 p0cv3 gain (+0.011531).** It is not a seed-plumbing bug, a cache hit or a deterministic path. The two cells have different adapters (sha `1b45c09f…` vs `39c61ce6…`), different step-0/16 losses (14.271→13.767 vs 14.425→13.554), and z-margins that differ by up to 0.061 on the changed decisions. The per-seed init perturbation, dropout and context/slate streams are all keyed on the seed (`key(seed, rng_arm, tag, …)`), the pre-flight assert checks the streams differ, and the resume path only reads a cell's own `{arm}_{tag}_seed{N}.json`. But the decision is discrete: both adapters changed the same 151 held-out picks, and the z ≥ 0.40 gate kept the same 27 with the same programs, so the utility vector (and its mean) is bit-identical. The cell is dominated by one held-out context where the kept change recovers an incumbent gate failure (+2.2315 of the fold's 244 contexts, ≈ +0.0091 of the +0.0115). The same vector appears for `pairnf_z40` seed 2 and for H467 `acc2_z40` seeds 1 and 2 in that cell. Control seed 1 (+0.0024) differs only by not keeping that one context. The red-team now lists every identical fold OOF pair (`identical_fold_oof`): 7 pairs over 30 cell groups, all with distinct adapter shas. Takeaway: fold gains are lumpy, and a single floor recovery moves a fold by ~0.009, so seed agreement on a fold gain is not evidence of shared randomness.
3. **acc4 (mechanism).** acc4 keeps 109–116 changes per partition vs the control's 99–108, with 99–106 shared. Its non-floor pass win rate is 57.7% (383 W / 281 L over 6 cells), the same as the control's 57.7% (353 / 259). Most of its edge over the control comes from floor recoveries (5 / 4 / 5 vs 3 / 5 / 4), not from better non-floor picks. Going from acc2 to acc4 doesn't add anything (median Elo +5.7 → +5.6, paired CI > 0 on seed 1 only in both).
4. **lk4lr2 (mechanism).** The displacement match worked, slightly overshooting: |z| of changed decisions p50 / p90 / p99 / max 0.198 / 0.475 / 0.653 / 0.870 vs the control's 0.177 / 0.437 / 0.599 / 0.790, so the gate keeps 824 of 4,322 (19.1%) vs 630 of 4,266 (14.8%). The extra kept changes have a lower non-floor pass win rate (448 W / 355 L, 55.8%). With the confound gone, Lookahead smoothing adds no signal.
5. **Seed noise.** Seed disagreement on OOF decisions per partition: control 25–32, acc4 15–27, lk4lr2 25–38. `seed_independence` holds for every arm × seed pair, and adapter shas are distinct in every cell.
6. **Selection / winner's curse.** Selection used TRAIN-fold CV only (non-floor Elo), never val. The selection bootstrap picks the selected arm as argmax only 47% / 60% / 73% of the time, and the max-arm non-floor Elo CI is [−0.9, +11.9] / [+0.3, +12.3] / [−1.1, +11.4]. No arm or seed was dropped or re-run selectively.
7. **Ties.** About 91% of CV decisions and 84–86 of 91 val91 decisions are ties by design. Ties count 0.5 in Elo and never satisfy strict. The 0.002 strict margin is unchanged.
8. **Metric direction.** Utility is higher-is-better (negated fitness, −2.0 on gate fail). Gains are paired candidate − incumbent on the same contexts. Verdicts report Elo, not AUC, and the aggregator recomputes them from the per-seed fields.
9. **Leakage.** Folds are family-disjoint and exclude val families (asserted, family overlap 0). The red-team rebuilds both partitions and matches the logged fold maps (floors 4/4/4/3/4). No sealed or holdout reads: the only holdout, the sealed meta-test group, is not in the packet and was never touched.
10. **Safety.** DOPE `c51a56d0` is intact on every DONE line (before = after). There were 2 own-guard MemAvailable aborts (seeds 1 and 3, caused by other owners' load) and 0 OOM kills. Both resumed deterministically. No other owner's job was paused or killed. The wave scripts contain no ssh, rsync, xbabe0 or xbabe3 target. xbabe2 disk is at 95% (~426 GB free), and the wave dir is ~466 MB. No large objects are committed.

## Meaning
Gradient-noise reduction is exhausted on this instrument. acc2 and acc4 both add a few kept changes at the same ~58% non-floor win rate, worth about +1 Elo on the median, which can't be separated from the control. Displacement-matched Lookahead keeps more changes at a lower win rate. H451–H468 agree that the gated step is worth about +4 to +6 OOF Elo and is not separable from 0 on val91.

## Next step
H469 (`~/jope-data/hwave-h469`, pre-registered separately) tests per-context robust losses on the same instrument and streams: trimmed-mean per-context loss (drop the top 10% / 25% per step) and a soft Cauchy (log1p) per-context loss, so heavy contexts can't dominate the step. While designing it we found the adapter's modality dropout is inert under the locked harness (bit-identical predictions and adapters for p ∈ {0, 0.10, 0.25, 0.5, 0.99}), so dropout was dropped as a lever.

No adoption nudge was sent to the flywheel owner.

Artifacts:
- `metrics.json`: `aggregate` (fold-gain vectors dropped for size) and `redteam` as produced on xbabe2.
- `h468_trainfold_cv.py` (as run), `h468_aggregate.py`, `h468_redteam.py`.

Raw logs, losses and fold signatures stay on xbabe2 in `~/jope-data/hwave-h468`.
