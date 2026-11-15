# H457: stabilization of the gated pairnf step (Polyak tail average, ASAM) (offline, adapter-only)

**Verdict: `NO_LIFT`.** Neither stabilization arm beats the `pairnf_z40` control, and both are significantly worse than it. `avg8_z40` (Polyak/SWA tail average of the same pairnf trajectory over steps 9–16) shrinks the z-margins of changed picks (median 0.11 vs 0.18, max 0.55 vs 0.79), so the z ≥ 0.40 gate keeps 2.6 changes per fold instead of 21. Its OOF gain falls to +0.0004 (Elo +0.9), and the paired arm − control gain CI is below 0 on all 3 seeds. `sam05_z40` (pairnf trained with ASAM ρ 0.5 at lr 3e-4 × 16 steps) keeps 10.8 changes per fold at about the control's win rate: OOF gain +0.0013, Elo +2.0, paired CI below 0 on seeds 2–3 and touching 0 on seed 1. Strict passes: control 1/3 (seed 2, as in H452–H454), avg8 0/3, sam05 0/3. The CV-selected arm is `pairnf_z40` on every seed. val91 (one read per seed) median gain is **+0.0010** and every CI spans 0 (−0.0006 / +0.0011 / +0.0010). Elo is 0.0 / +11.5 / +11.5 with 0 new gate failures.

## Hypothesis
H451–H454 settled at about +4 to +6 OOF Elo for the gated `pairnf` step. Changing which pairs it learns from (H454) didn't help. H457 changes the stabilization axis instead. If the last iterate is noisy (iterate averaging) or sits in a sharp minimum (ASAM), the z-margins the gate reads should hold up better across held-out families. Arms:
- `pairnf_z40`: H452–H454 `pairnf` + z ≥ 0.40. This is the control, bit-for-bit H454. Its base also books a Polyak tail average for bookkeeping only, which leaves the trajectory unchanged.
- `avg8_z40`: Polyak tail average of the same control trajectory over steps 9–16. It needs no extra training and is a strictly paired test of the averaged iterate against the last one.
- `sam05_z40`: `pairnf` trained with adaptive SAM (ρ 0.5) at the same lr and step count.

Selection was pre-registered and uses Elo only: per seed, the max mean non-floor OOF Elo over the two partitions (ties broken by pooled gain). The strict rule is unchanged from H452. An arm only counts if it also beats the control in a paired family bootstrap.

## Setup
- **Code:** `h457_trainfold_cv.py` as run (sha256 verified in the scratch dir). CPU only, `nice 19`, 16 threads per seed, self-abort at MemAvailable < 15 GB. Runs against the scratch copy `~/jope-data/hwave-h457` on xbabe2, never the live run directory. Launch was gated on CPU (`cpugate.sh`: machine busy% + 16 threads ≤ 85%) and memory (≥ 38 GB).
- **Runs:** the probe passed (PROBE_OK, DOPE intact) at 18:29 UTC. Seeds 1/2/3 launched 2026-11-14 18:42 / 18:52 / 19:02 UTC on xbabe2 (MemAvailable 77 / 76 / 66 GB, CPU busy 14–26%). There were no guard aborts and no resumes. DONE at 2026-11-15 04:20 / 04:38 / 04:53 UTC.
- **Start point:** incumbent `round-3175.pt` plus optimizer state, deep-copied per candidate. Parity reproduces the live val utility exactly (1.1389572403748425).
- **DOPE frozen:** composite sha `c51a56d0…` before and after on every seed (all 3 DONE lines). Only the 29,211 adapter parameters train.
- **Data:** the real S3 packet: 1,219 DL-Train contexts (205 families) and 91 DL-Val contexts (19 families), with family overlap 0 and every row DL-Train or DL-Val (asserted). No sealed or holdout data was read.
- **Partitions:** the two H452 floor-stratified, family-grouped 5-fold partitions (`h452a/`, `h452b/`).
- **Compute:** 2 trained bases × 2 partitions × 5 folds = 20 trainings per seed (sam05 costs about 2× per step), plus 1 full-train run. Peak RSS was 6.2–6.3 GB per seed, and the minimum MemAvailable was 41.3 / 41.4 / 37.4 GB.

## Outcome
**CV** (pooled OOF over 2 × 1,219 TRAIN contexts; values are seeds 1 / 2 / 3, median in parentheses). "vs control" is the paired arm − `pairnf_z40` OOF gain with its family-bootstrap 95% CI.

| Arm | OOF gain | Elo vs incumbent | Elo 95% CI (family bootstrap) | W / L (ties rest of 2,438) | Strict / utility-strict | Recovered / new gate fails (of 38) | vs control (gain, CI) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **pairnf_z40** (control) | +0.0048 / +0.0069 / +0.0056 (+0.0056) | +4.7 / +6.6 / +3.8 (+4.7) | [−1.3, +10.6] / [+0.7, +12.3] / [−1.8, +9.4] | 122/89, 128/82, 115/88 | 1/3 / 3/3 | 3, 5, 4 / 0 | — |
| avg8_z40 | +0.0004 / +0.0003 / +0.0006 (+0.0004) | +0.9 / +0.7 / +1.1 (+0.9) | [−0.9, +3.0] / [−1.1, +2.7] / [−0.7, +3.3] | 16/10, 15/10, 17/9 | 0/3 / 0/3 | 0 / 0 | −0.0044 [−0.0093, −0.0002]; −0.0066 [−0.0134, −0.0018]; −0.0050 [−0.0104, −0.0008] |
| sam05_z40 | +0.0013 / +0.0010 / +0.0014 (+0.0013) | +2.0 / +1.4 / +2.9 (+2.0) | [−1.6, +5.8] / [−2.2, +5.1] / [−0.8, +6.7] | 60/46, 56/46, 65/45 | 0/3 / 0/3 | 0 / 0 | −0.0034 [−0.0080, +0.0000]; −0.0059 [−0.0123, −0.0014]; −0.0042 [−0.0092, −0.0004] |

Head-to-head on contexts where the arm and the control differ:
- avg8 (192 / 192 / 186 contexts): arm better 82 / 76 / 82, control better 110 / 116 / 104.
- sam05 (124 / 122 / 113 contexts): arm better 52 / 47 / 53, control better 72 / 75 / 60.

**val91 regression check** (one read per seed; the CV-selected arm trained on all 1,219 TRAIN contexts):

| Seed | Selected arm | Gain | Gain CI | W / T / L | Elo | Changed / kept | New gate failures | not_worse |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | pairnf_z40 | −0.0006 | [−0.0022, +0.0006] | 3/85/3 | 0.0 | 54 / 6 | 0 | false |
| 2 | pairnf_z40 | +0.0011 | [−0.0005, +0.0028] | 4/86/1 | +11.5 | 54 / 5 | 0 | true |
| 3 | pairnf_z40 | +0.0010 | [−0.0005, +0.0027] | 4/86/1 | +11.5 | 56 / 5 | 0 | true |

Median val91 gain is +0.0010 (Elo +11.5), inside every CI. The pre-registered LIFT rule fails: no arm beats the control, seed 1 is not_worse false, and the val91 CIs span 0.

## Red-team (`h457_redteam.py`, `h457_aggregate.py`; output under `redteam` / `aggregate` in `metrics.json`)
1. **Control reproduction.** All 30 `pairnf_z40` cells (3 seeds × 2 partitions × 5 folds) have the same adapter sha and identical OOF vectors and gains as H454 (30/30). The full-train adapters of seeds 1–2 also match H454 (`e0a5f716…`, `be56fa29…`). H454 seed 3 selected `hn16_z40`, so there is no seed-3 full-train comparison. The Polyak bookkeeping did not change the trajectory, and the instrument is unchanged.
2. **Why avg8 loses (mechanism, not a bug).** Averaging steps 9–16 pulls the adapter back toward the earlier iterates, which shrinks the predicted gap between the new pick and the incumbent pick. Over all 30 cells, the z-margin of changed picks has median / p90 / max 0.107 / 0.294 / 0.553 for avg8, against 0.177 / 0.437 / 0.79 for the control. Only 2.1% of avg8 changes clear 0.40 (control 14.8%). avg8 makes essentially no decisions the control does not (`arm_only_gain_sum` = 0 in all 6 seed × partition cells), so it is a near-subset of the control that drops net-good changes.
3. **sam05.** Its z-margins are also compressed (median 0.14, p90 0.38; 7.9% ≥ 0.40), so it keeps 10.8 changes per fold against 21.0. Its win rate on kept changes (about 57%) matches the control's, so flatness buys no accuracy, only fewer kept changes. The training loss is the same (p0cv0 13.7 → 13.1–13.3 for both). **Caveat:** sam05 vs control is *not* a same-random-stream comparison. The context, slate, init-perturbation and dropout streams are keyed by the base arm name (`pairnf` vs `pairnf_sam05`), so part of the sam05 − control difference is stream noise. avg8 *is* strictly paired, because it uses the same trajectory.
4. **Seed noise.** Seed disagreement on OOF decisions per partition: control 25–32 contexts, sam05 12–17, avg8 0–5. All 9 arm × seed-pair fold-gain vectors are distinct (seed-independence check passes). Arm comparisons rest on paired CIs, not on seed-to-seed differences.
5. **Selection / winner's curse.** The control is selected on all 3 seeds and is the bootstrap argmax in 80% / 97% / 60% of family bootstraps. The max-arm non-floor Elo CI is [−0.15, +10.3] / [+0.45, +11.8] / [−0.14, +8.8], so the plateau itself is still not separable from 0 on 2 of 3 seeds.
6. **val91 fragility.** The seeds change 5–6 of 91 contexts, 5 of them shared (indices 49, 50, 64, 69, 84, plus 25 on seed 1). Seeds differ on 1–3 decisions, which moves Elo between 0.0 and +11.5.
7. **Ties.** About 91% of CV decisions and 85–86 of 91 val91 decisions are ties by design (the gate reverts to the incumbent pick). Ties count 0.5 in Elo and never satisfy strict.
8. **Metric direction.** Utility is higher-is-better (negated fitness, −2.0 on gate fail). Gains are paired candidate − incumbent on the same contexts. The aggregator recomputes verdicts from per-seed fields.
9. **Leakage.** Folds are family-disjoint and exclude val families (asserted per fold, family overlap 0). The pairwise term and hinge use training-fold contexts only. Gates use predictions only. val91 is read once per seed, for the CV-selected arm only. There were no sealed or holdout reads.
10. **Safety.** DOPE `c51a56d0` is intact on every DONE line and was verified after every training. There were no guard aborts and no OOM (minimum MemAvailable 37.4 GB with swap full from machine-wide load). xbabe2 disk is at 93% (522 GB free). The wave dir is 466 MB, which includes the input copy. No large objects are committed.

## Meaning
Both stabilizers act as shrinkage on the pairnf update. They lower the gated step's confidence (z-margin) without raising its accuracy, so the label-free z ≥ 0.40 gate keeps fewer changes and the gain falls with them. So the plateau is not caused by iterate noise or sharpness. It is set by how far the step moves and by the accuracy of the changes it is confident about. H451–H457 agree that the gated step is worth about +4 to +6 Elo on TRAIN-fold CV and is not separable from 0 on val91.

## Next step
H461 (`~/jope-data/hwave-h461` on xbabe2; H458–H460 are taken by other work) tests the two axes H457 points at, against the bit-for-bit control on the same partitions with 3 seeds:
- `s32_z40`: the dose axis, the opposite of avg8's shrinkage. The same pairnf trajectory runs 32 steps instead of 16, sharing the control's random streams, so the first 16 steps equal the control.
- `fbal_z40`: family-balanced context sampling (each family equally likely per draw), because CV is family-grouped and large families dominate the gradient.

No adoption nudge was sent to the flywheel owner.

Artifacts:
- `metrics.json`: `aggregate` holds per-arm/seed CV, per-partition Elo and fold gains, gate stats, the paired vs-control comparison, val91 regression checks and peaks. `redteam` holds the floor/non-floor split, partition agreement, decision overlap with the control, seed disagreement, the selection bootstrap and val91 per-context agreement.
- `h457_trainfold_cv.py` (as run), `h457_aggregate.py`, `h457_redteam.py`.

Raw logs, losses and fold signatures stay on xbabe2 in `~/jope-data/hwave-h457`.
