# H461: dose and family-balanced sampling on the gated pairnf step (offline, adapter-only)

**Verdict: `NO_LIFT`.** Neither arm beats the `pairnf_z40` control. `s32_z40` (the same pairnf trajectory run to 32 steps) inflates the z-margins of changed picks about 7× (median 1.19 vs 0.18), so the z ≥ 0.40 gate keeps about 745 changes per partition instead of about 104. It recovers 29–30 of 38 incumbent gate failures (+0.033 gain from floors per partition) but loses about 0.03 on non-floor contexts and adds 10–11 new gate failures. Net OOF gain is +0.0015 (median) with a ±0.027 CI, non-floor Elo is −10 to −15, and the paired arm − control gain is −0.0054 / −0.0053 / −0.0018 (every CI spans 0, head-to-head Elo −15 to −26). `fbal_z40` (contexts drawn ∝ 1/|family|) is almost the control: about 90 of 2,438 decisions differ, median OOF gain +0.0041 / Elo +4.0 vs the control's +0.0056 / +4.7, paired CI spans 0 on every seed and is below the control on seeds 2–3. Strict passes: control 1/3 (seed 2, as in H452–H457), s32 0/3, fbal 0/3. CV selected `pairnf_z40` / `pairnf_z40` / `fbal_z40`. val91 (one regression check per seed) median gain is **+0.0011** (Elo +7.6), and every CI spans 0 (−0.0006 / +0.0011 / +0.0018).

## Hypothesis
H457 showed that both stabilizers (Polyak tail average, ASAM) act as shrinkage on the pairnf update. H461 tested the two axes that result pointed at:
- `pairnf_z40`: the control, bit-for-bit H452–H457 pairnf + z ≥ 0.40, taken as the step-16 snapshot of a 32-step run on the same random streams.
- `s32_z40`: the dose axis (the opposite of avg8), the last iterate of the same 32-step trajectory. It is strictly paired with the control.
- `fbal_z40`: family-balanced context sampling, 16 steps. It shares the control's init-perturbation and dropout streams.

## Pre-registered rules (unchanged from H452–H457)
Selection is by the max mean non-floor OOF Elo over the two partitions. Strict means pooled OOF gain > 0.002 and gain CI low > 0, pooled Elo > 0 and Elo CI low > 0, non-floor Elo CI low > 0, and Elo > 0 on each partition. An arm counts only if it also beats the control in the paired family bootstrap. LIFT needs that on every seed, plus val91 not_worse on every seed and a clean red-team.

## Setup
- **Code:** `h461_trainfold_cv.py` as pre-registered (sha256 `bf518f7a…`, #2326). CPU only, `nice 19`, 16 threads per seed. Scratch copy `~/jope-data/hwave-h461` on xbabe2, never the live run dir, never xbabe0. No GPU and no `qualification.lock`.
- **Runs:** seeds 1/2/3 launched 2026-11-15 05:42 / 05:52 / 06:02 UTC. All 3 died silently between 13:51 and 14:01 UTC: no traceback, no guard abort, and no kernel OOM-killer record in the journal. They were resumed manually at 14:04 / 14:10 / 14:16 UTC (`relaunch_0703.sh`, gated on MemAvailable ≥ 38 GB and CPU). The resume is deterministic via `results/fold_sig`. DONE at 14:31 / 14:45 / 14:40 UTC.
- **Data and model:** the real S3 packet, 1,219 DL-Train contexts (205 families) and 91 DL-Val contexts (19 families), family overlap 0. DOPE `c51a56d0…` frozen, and only the 29,211 adapter params train. Incumbent val parity is 1.1389572403748425 on every seed.
- **Compute:** peak RSS 6.3 GB per seed. The minimum MemAvailable of the resumed runs was 47.8 / 43.7 / 48.7 GB (swap full from machine-wide load).

## Outcome
**CV** (pooled OOF over 2 × 1,219 TRAIN contexts; values are seeds 1 / 2 / 3, median in parentheses):

| Arm | OOF gain | Elo vs incumbent | Elo 95% CI (family bootstrap) | Non-floor Elo | W / L (ties rest of 2,438) | Strict | Recovered / new gate fails (of 38) | vs control (gain, CI) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **pairnf_z40** (control) | +0.0048 / +0.0069 / +0.0056 (+0.0056) | +4.7 / +6.6 / +3.8 (+4.7) | [−1.3, +10.6] / [+0.7, +12.3] / [−1.8, +9.4] | +4.3 / +5.9 / +3.3 | 122/89, 128/82, 115/88 | 1/3 | 3, 5, 4 / 0 | — |
| s32_z40 | −0.0006 / +0.0015 / +0.0038 (+0.0015) | −10.3 / −8.0 / −5.6 (−8.0) | [−28.2, +6.2] / [−25.8, +8.5] / [−22.6, +10.5] | −14.6 / −12.5 / −10.0 | 706/778, 718/774, 726/765 | 0/3 | 29, 30, 30 / 11, 11, 10 | −0.0054 [−0.0325, +0.0203]; −0.0053 [−0.0326, +0.0209]; −0.0018 [−0.0260, +0.0242] |
| fbal_z40 | +0.0051 / +0.0041 / +0.0039 (+0.0041) | +3.7 / +4.0 / +4.0 (+4.0) | [−2.4, +9.4] / [−2.5, +10.2] / [−2.0, +10.0] | +3.3 / +3.8 / +3.8 | 129/103, 129/101, 129/101 | 0/3 | 3, 2, 2 / 0 | +0.0003 [−0.0026, +0.0030]; −0.0028 [−0.0077, +0.0006]; −0.0017 [−0.0047, +0.0005] |

Head-to-head on contexts where the arm and the control differ:
- s32 (1,400 / 1,409 / 1,412): arm better 648 / 660 / 675, control better 752 / 749 / 737.
- fbal (79 / 92 / 91): arm better 38 / 41 / 48, control better 41 / 51 / 43.

**val91 regression check** (the CV-selected arm trained on all 1,219 TRAIN contexts):

| Seed | Selected arm | Gain | Gain CI | W / T / L | Elo | Changed / kept | New gate failures | not_worse |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | pairnf_z40 | −0.0006 | [−0.0022, +0.0006] | 3/85/3 | 0.0 | 54 / 6 | 0 | false |
| 2 | pairnf_z40 | +0.0011 | [−0.0005, +0.0028] | 4/86/1 | +11.5 | 54 / 5 | 0 | true |
| 3 | fbal_z40 | +0.0018 | [−0.0017, +0.0063] | 4/85/2 | +7.6 | 54 / 6 | 0 | true |

Across seeds, the median val91 gain is +0.0011 and the median Elo +7.6. The seed-averaged per-context val91 gain is +0.0008, with family-bootstrap CI [−0.0006, +0.0023]. The median-of-seeds bootstrap CI is [−0.0013, +0.0020], and pooled W/T/L is 5/83/3. The median is not beyond its CI. The pre-registered LIFT rule fails: no arm beats the control, seed 1 is not_worse false, and the val91 CIs span 0.

## Red-team (`h461_redteam.py`, `h461_aggregate.py`, copied from H457; output under `redteam` / `aggregate` in `metrics.json`)
1. **Control reproduction.** All 30 `pairnf_z40` cells (3 seeds × 2 partitions × 5 folds) have the same adapter sha and identical OOF vectors as H457 (30/30). The full-train adapters of seeds 1–2 also match H454/H457 (`e0a5f716…`, `be56fa29…`). The step-16 snapshot of the 32-step run is exactly the old 16-step control, so s32 is strictly paired.
2. **Why s32 loses (mechanism, not a bug).** The z-margin of changed picks has median / p90 1.19 / 2.02 for s32 vs 0.18 / 0.44 for the control. 77% of s32's changes clear 0.40 (control 14.8%). The gate is in units of the per-context prediction std, and doubling the dose sharpens predictions everywhere, so it no longer filters. s32 changes 742–750 TRAIN decisions per partition, including every one the control changes. On floors it is strong (recovers 29–30 of 38 incumbent gate failures, +0.033 gain per partition from floors). Off floors its win rate is about 47%, it adds 10–11 new gate failures, and the non-floor total is −0.028 to −0.036. The two cancel.
3. **fbal.** Its z distribution matches the control (median 0.18, 16.4% ≥ 0.40). It keeps about the same number of changes and differs from the control on only 79–92 contexts, with the control better on 2 of 3 seeds. Family reweighting moves the gradient very little at this dose. Caveat: fbal's context draws necessarily differ from the control's, so part of the difference is stream noise.
4. **Seed noise.** Seed disagreement on OOF decisions per partition: control 25–32, fbal 22–34, s32 106–122. All 9 arm × seed-pair fold-gain vectors are distinct (the seed-independence check passes). Arm comparisons rest on paired CIs.
5. **Selection / winner's curse.** Selection used TRAIN-fold CV only (non-floor Elo), never val. Seed 3 picked fbal over the control by 0.43 non-floor Elo, and fbal was the bootstrap argmax in only 60% of family bootstraps. The max-arm non-floor Elo CI is [−1.3, +10.5] / [+0.3, +11.9] / [−1.9, +10.6], so the plateau is still not separable from 0 on 2 of 3 seeds.
6. **val91 fragility.** The seeds change 5–6 of 91 contexts, mostly the same ones (indices 49, 64, 69 in all three). Seeds differ on 3–6 decisions, which moves Elo between 0.0 and +11.5.
7. **Ties.** About 91% of CV decisions and 85–86 of 91 val91 decisions are ties by design (the gate reverts to the incumbent pick). Ties count 0.5 in Elo and never satisfy strict. The 0.002 strict margin is unchanged.
8. **Metric direction.** Utility is higher-is-better (negated fitness, −2.0 on gate fail). Gains are paired candidate − incumbent on the same contexts. The aggregator recomputes verdicts from the per-seed fields.
9. **Leakage.** Folds are family-disjoint and exclude val families (asserted per fold, family overlap 0). The pairwise term and hinge use training-fold contexts only. Gates use predictions only. No sealed or holdout reads.
10. **Protocol deviation (no effect).** A second, unlogged process re-ran only the full-train regression check for seeds 1 and 3 at 14:54–14:57 UTC, after their DONE, with every CV cell resumed from `fold_sig`. val91 was therefore scored twice for those seeds. Both runs gave bit-identical adapters (`e0a5f716…`, `731506f3…`) and identical numbers, so no selection depended on it. Seed 2 hung for about 15 minutes after DONE at about 360% CPU (thread-pool teardown) and was stopped by the routine. H463 hard-exits after DONE.
11. **Safety.** DOPE `c51a56d0` is intact on every DONE line and was verified after every training. The 13:51–14:01 UTC deaths were not memory-guard aborts and have no kernel OOM record. Each seed's minimum MemAvailable stayed ≥ 42 GB. xbabe2 disk is at 94% (494 GB free). The wave dir is 466 MB including the input copy. No large objects are committed.

## Meaning
How far the pairnf step moves controls both its confidence and its errors, but the label-free z-gate measures confidence in units that grow with the dose. At 32 steps, the gate keeps 7× more changes at a below-50% non-floor win rate, even though the extra dose is very good at fixing the incumbent's gate failures. Family balancing is neutral. H451–H461 agree that the gated step is worth about +4 to +6 OOF Elo and is not separable from 0 on val91.

## Next step
H463 (`~/jope-data/hwave-h463` on xbabe2; H462 belongs to another owner) keeps the 32-step dose and fixes the gate or the schedule, against the bit-for-bit control:
- `s32_z200`: the same s32 adapter with an absolute z ≥ 2.00 gate. The threshold is count-matched to the control from the label-free z distribution.
- `s32cos_z40` and `s32cos_z200`: 32 steps with a 2-step warmup and cosine lr decay, which uses the control's integrated lr spread over 2× the data, on the control's random streams.

No adoption nudge was sent to the flywheel owner.

Artifacts:
- `metrics.json`: `aggregate` and `redteam` as produced on xbabe2.
- `h461_trainfold_cv.py` (as run), `h461_aggregate.py`, `h461_redteam.py`.

Raw logs, losses and fold signatures stay on xbabe2 in `~/jope-data/hwave-h461`.
