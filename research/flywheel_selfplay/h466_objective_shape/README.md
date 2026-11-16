# H466: objective shape on the gated pairnf step (offline, adapter-only)

**Verdict: `NO_LIFT`.** No objective-shape arm beats the `pairnf_z40` control on any seed. `clip05_z40` (pointwise smooth-L1 target clipped at context best + 0.5) is effectively the control: it differs on only 3–5 of 2,438 OOF decisions per seed, with median Elo +4.8 vs +4.7 and a paired gain of +0.0001 / −0.0001 / −0.0009 whose CIs span 0. `ctxn_z40` (per-context loss normalization) and `pw2_z40` (pairwise BT weight 2.0) both **shrink the kept set** from about 99–108 to 60–75 changes per partition at the same ~58% non-floor win rate. They end up below the control on the median: Elo +3.8 and +3.7 vs +4.7, with paired gain CIs spanning 0 on every seed. CV selected `clip05_z40` / `pairnf_z40` / `clip05_z40`. The val91 regression check (one per seed) gives a median gain of **+0.0010** (Elo +11.5). Every CI spans 0, and seed 1 (clip05, 3 W / 3 L, −0.0006) is **not_worse = false**. No breakthrough.

## Hypothesis (pre-registered in #2342)
H457, H461 and H463 changed how many changes the z ≥ 0.40 gate keeps but never the ~57% win rate of those changes. H466 keeps the dose (16 steps, lr 3e-4) and the gate fixed and changes the **shape of the training objective**. Every arm runs on the control's random streams (`rng_arm = pairnf`), so only the loss differs: `ctxn_z40` (per-context loss-row normalization), `clip05_z40` (pointwise target tail clip at best + 0.5) and `pw2_z40` (BT pairwise weight 2.0).

## Pre-registered rules (unchanged from H452–H463)
Selection takes the max mean non-floor OOF Elo over the two partitions. Strict requires pooled OOF gain > 0.002 with gain CI low > 0, pooled Elo > 0 with Elo CI low > 0, non-floor Elo CI low > 0, and Elo > 0 on each partition. An arm counts only if it also beats the control in the paired family bootstrap. Ties never win. LIFT needs that on every seed, plus val91 not_worse on every seed and a clean red-team.

## Setup
- **Code:** `h466_trainfold_cv.py` as pre-registered (sha256 `c90d3590…`, re-verified after the run). CPU only, `nice 19`, 16 threads per seed. It ran on the scratch copy `~/jope-data/hwave-h466` on xbabe2, never on the live run dir and never on xbabe0. No GPU, so no `qualification.lock`.
- **Runs:** seeds 1/2/3 launched 2026-11-16 03:08 / 03:18 / 03:28 UTC and finished (DONE) at 18:36 / 19:23 / 23:23 UTC. Other owners' jobs took machine memory and tripped the 15 GB MemAvailable guard 8 times (3 / 3 / 2 per seed). The waiter relaunched each seed resume-safe from `results/fold_sig`. At 10:00 UTC it was swapped for `waiter2.sh`, which allows at most 2 seeds alive at once and 5 relaunches per seed. The waiter exited cleanly at 23:23 UTC.
- **Data and model:** the real S3 packet, 1,219 DL-Train contexts (205 families) and 91 DL-Val contexts (19 families), family overlap 0. DOPE `c51a56d0…` stayed frozen, and only the 29,211 adapter params trained. Incumbent val parity is 1.1389572403748425 on every seed.
- **Compute:** peak RSS was 6.3 GB per seed. Min MemAvailable of the final (completing) process was 32.8 / 28.4 / 46.4 GB.

## Outcome
**CV** (pooled OOF over 2 × 1,219 TRAIN contexts; values are seeds 1 / 2 / 3, median in parentheses):

| Arm | OOF gain | Elo vs incumbent | Elo 95% CI (family bootstrap) | Non-floor Elo | W / L (ties rest of 2,438) | Strict (utility-strict) | Recovered / new gate fails (of 38) | vs control (gain, CI) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **pairnf_z40** (control) | +0.0048 / +0.0069 / +0.0056 (+0.0056) | +4.7 / +6.6 / +3.8 (+4.7) | [−1.3, +10.6] / [+0.7, +12.3] / [−1.8, +9.4] | +4.3 / +5.9 / +3.3 | 122/89, 128/82, 115/88 | 1/3 (3/3) | 3, 5, 4 / 0 | — |
| ctxn_z40 | +0.0039 / +0.0059 / +0.0065 (+0.0059) | +3.3 / +4.6 / +3.8 (+3.8) | [−0.9, +7.3] / [0.0, +9.1] / [−0.7, +8.3] | +3.0 / +4.1 / +3.2 | 76/53, 84/52, 81/54 | 0/3 (3/3) | 2, 4, 5 / 0 | −0.0008 [−0.0031, +0.0012]; −0.0010 [−0.0036, +0.0012]; +0.0010 [−0.0011, +0.0036] |
| clip05_z40 | +0.0048 / +0.0067 / +0.0047 (+0.0048) | +4.8 / +6.0 / +3.8 (+4.8) | [−1.2, +10.8] / [0.0, +11.7] / [−1.6, +9.3] | +4.5 / +5.4 / +3.5 | 123/89, 127/85, 114/87 | 0/3 (3/3) | 3, 5, 3 / 0 | +0.0001 [−0.0000, +0.0002]; −0.0001 [−0.0003, 0.0000]; −0.0009 [−0.0029, +0.0000] |
| pw2_z40 | +0.0034 / +0.0061 / +0.0039 (+0.0039) | +3.7 / +4.1 / +3.3 (+3.7) | [−0.6, +8.0] / [−0.4, +8.6] / [−1.1, +7.8] | +3.5 / +3.5 / +2.9 | 83/57, 86/57, 80/57 | 0/3 (3/3) | 2, 5, 3 / 0 | −0.0013 [−0.0055, +0.0022]; −0.0008 [−0.0036, +0.0019]; −0.0016 [−0.0049, +0.0016] |

Head-to-head on contexts where the arm and the control differ:
- ctxn_z40 (94 / 95 / 89): arm better 44 / 43 / 46, control better 50 / 52 / 43 (Elo −22.2 / −33.0 / +11.7).
- clip05_z40 (3 / 3 / 5): arm better 2 / 0 / 3, control better 1 / 3 / 2.
- pw2_z40 (85 / 90 / 87): arm better 39 / 38 / 42, control better 46 / 52 / 45 (Elo −28.7 / −54.5 / −12.0).

**val91 regression check** (the CV-selected arm trained on all 1,219 TRAIN contexts):

| Seed | Selected arm | Gain | Gain CI | W / T / L | Elo | Changed / kept | New gate failures | not_worse |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | clip05_z40 | −0.0006 | [−0.0022, +0.0006] | 3/85/3 | 0.0 | 54 / 6 | 0 | false |
| 2 | pairnf_z40 | +0.0011 | [−0.0005, +0.0028] | 4/86/1 | +11.5 | 54 / 5 | 0 | true |
| 3 | clip05_z40 | +0.0010 | [−0.0005, +0.0027] | 4/86/1 | +11.5 | 55 / 5 | 0 | true |

Across seeds, the median val91 gain is +0.0010 and the median Elo +11.5. The seed-averaged per-context gain is +0.0005, with family-bootstrap CI [−0.0007, +0.0017]. The median-of-seeds CI is [−0.0006, +0.0026]. The median is not beyond its CI. The pre-registered LIFT rule fails: no arm beats the control on any seed, every val91 CI spans 0, and seed 1 is not not_worse.

## Red-team (`h466_redteam.py`, `h466_aggregate.py`, copied from H463; output under `redteam` / `aggregate` in `metrics.json`)
1. **Control reproduction.** All 30 `pairnf_z40` cells have the same adapter sha and identical OOF vectors as H463 (30/30). The seed-2 full-train control adapter is again `be56fa29…`, so the guard-abort relaunches did not perturb any cell (the resume is deterministic).
2. **clip05 is (almost) a no-op.** The clip at best + 0.5 rarely binds after the 16-step dose. Kept changes equal the control's in 5 of 6 seed × partition cells, except for 1–2 extra or missing changes. Its CV "selection" on seeds 1 and 3 is a tie-break on near-identical non-floor Elo (+4.49 vs +4.34, +3.47 vs +3.33). The selection bootstrap picks it as the argmax only 35% / 26% of the time.
3. **ctxn and pw2 (mechanism).** Both shrink z-margins, so the z ≥ 0.40 gate keeps 60–75 changes per partition vs 99–108 for the control. Almost all are a subset of the control's (both_changed 58–73). Their non-floor pass W/L is 33–43 / 24–31 (about 58%), the same win rate as the control's 55–62 / 40–47. As in H457 (avg8, sam05), they shrink the kept count without improving the win rate, and so lose Elo.
4. **Seed noise.** Seed disagreement on OOF decisions per partition: control 25–32, ctxn 16–28, clip05 23–33, pw2 19–28. Fold-gain vectors differ for every arm × seed pair, and `seed_independence` is true for all 12 pairs.
5. **Selection / winner's curse.** Selection used TRAIN-fold CV only (non-floor Elo), never val. The max-arm non-floor Elo CI is [−0.4, +10.4] / [+0.6, +11.8] / [−0.8, +9.0].
6. **val91 fragility.** All three val91 adapters change the same 5 contexts (idx 49, 50, 64, 69, 84). Seed 1 adds idx 25 and flips 2 outcomes, which moves Elo from +11.5 to 0.0 and the gain to −0.0006. That is the same seed-1 result as H457 and H461 (−0.0006, 3/85/3).
7. **Ties.** About 91–95% of CV decisions and 85–86 of 91 val91 decisions are ties by design. Ties count 0.5 in Elo and never satisfy strict. The 0.002 strict margin is unchanged.
8. **Metric direction.** Utility is higher-is-better (negated fitness, −2.0 on gate fail). Gains are paired candidate − incumbent on the same contexts. The aggregator recomputes verdicts from the per-seed fields.
9. **Leakage.** Folds are family-disjoint and exclude val families (asserted, family overlap 0). The red-team rebuilds both partitions and matches the logged fold maps (floors 4/4/4/3/4 per fold). Gates use predictions only. No sealed or holdout reads: the only holdout, the sealed meta-test group, was never touched.
10. **Safety.** DOPE `c51a56d0` is intact on every DONE line and was verified after every training. There was no OOM kill: the 8 guard aborts were the script's own MemAvailable < 15 GB exits caused by other owners' jobs. None of their jobs was paused or killed. The wave scripts (`h466_trainfold_cv.py`, `launch.sh`, `waiter*.sh`, `cpugate.sh`) contain no ssh/rsync or xbabe0 target. xbabe2 disk is at 94% (487 GB free), and the wave dir is 466 MB. No large objects are committed.

## Meaning
The objective shape of the gated pairnf step doesn't move the plateau either. A tail clip is inert at this dose. Normalizing context rows or upweighting the pairwise term makes the adapter less confident, which only thins the kept set at the same ~58% win rate. H451–H466 agree that the gated step is worth about +4 to +6 OOF Elo and is not separable from 0 on val91. The win rate of the kept changes, not their count, is the binding constraint.

## Next step
H467 (`~/jope-data/hwave-h467`; H462, H464 and H465 belong to other owners) attacks the gradient noise of the 16-step pairnf step directly, on the control's streams:
- `acc2_z40`: gradient accumulation over 2 independent 1,024-context micro-batches per step (effective batch 2,048, same lr and steps). The first micro-batch is the control's.
- `look4_z40`: Lookahead (k = 4, α = 0.5) on the pairnf step.
- `bls10_z40`: Bradley-Terry label smoothing ε = 0.1 on the pairwise term (no hard 0/1 pair targets on noisy near-ties).

No adoption nudge was sent to the flywheel owner.

Artifacts:
- `metrics.json`: `aggregate` and `redteam` as produced on xbabe2.
- `h466_trainfold_cv.py` (as run), `h466_aggregate.py`, `h466_redteam.py`.

Raw logs, losses and fold signatures stay on xbabe2 in `~/jope-data/hwave-h466`.
