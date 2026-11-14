# H454: hard-negative slate and decision-focused pairwise weighting for the gated pairnf step (offline, adapter-only)

**Verdict: `NO_LIFT`.** Neither training-signal arm beats the `pairnf_z40` control. `hn16_z40` (the incumbent's predicted top-16 forced into every training slate) keeps fewer changes (W/L 84–90 / 56–64 vs the control's 115–128 / 82–89) at the same win rate, so its OOF gain and Elo sit slightly below the control (paired arm − control gain CIs span 0 on every seed). `hnfocus_z40` (hn16 plus Bradley-Terry pairs reweighted by softmax(z-margin / 0.5)) is **degenerate**: it compresses every z-margin below the 0.40 gate (max 0.34 over all 30 cells), so the gate keeps 0 changes on every fold and seed (gain 0, Elo 0; paired CI < 0 on all 3 seeds). Strict passes: control 1/3 (seed 2, as in H452/H453), hn16 0/3, hnfocus 0/3. val91 (one read per seed, CV-selected arm) median gain **+0.0005**, every CI spans 0 (s1 `pairnf_z40` −0.0006 [−0.0022, +0.0006], s2 `pairnf_z40` +0.0011 [−0.0005, +0.0028], s3 `hn16_z40` +0.0005 [−0.0008, +0.0020]); Elo 0.0 / +11.5 / +7.6 with 0 new gate failures.

## Hypothesis
In H451–H453 the 192-program training slate of the `pairnf` step is the top-48 programs by true fitness plus random fill, so the programs the adapter itself would switch to (its predicted top, which decide every changed pick and so every gated W/L) are almost never in the slate and never get pushed down. Putting those programs into the slate (hard negatives), and concentrating the pairwise gradient on the pairs that flip a decision, should make the kept changes right more often. Arms (one training per fold each):
- `pairnf_z40`: H452/H453 `pairnf` + z ≥ 0.40 (control, H453 bit-for-bit);
- `hn16_z40`: + the incumbent's predicted top-16 programs forced into every slate;
- `hnfocus_z40`: + hn16 and the non-floor BT pairs vs the incumbent argmin reweighted by softmax(z-margin / 0.5) (detached, mean weight 1).

Factorial reading: hn16 vs control = slate composition; hnfocus vs hn16 = pair weighting. Selection (pre-registered, Elo only): per seed, max mean non-floor OOF Elo over the two partitions (tie: pooled gain). Strict rule unchanged from H452; an arm only counts if it also beats the control in a paired family bootstrap.

## Setup
- **Code:** `h454_trainfold_cv.py` (as run; sha256 verified in the scratch dir), CPU only, `nice 19`, 16 threads per seed, self-abort at MemAvailable < 15 GB, against the scratch copy `~/jope-data/hwave-h454` on xbabe2, never the live run directory.
- **Runs:** seeds 1/2/3 launched 2026-11-14 07:16 / 07:26 / 07:36 UTC on xbabe2 (mem-checked at ≥ 38 GB). All three self-aborted once on the 15 GB guard around 11:35–12:00 UTC (MemAvailable fell to 5–10 GB while other jobs ran) during the `hn16` arm and were relaunched by the waiter at 12:37 / 12:57 / 13:17 UTC, resuming from the fold signatures (14–16 folds resumed per seed; every fold is seeded by seed/arm/fold, so resumes are deterministic). DONE 16:20 / 16:57 / 17:31 UTC.
- **Start point:** incumbent `round-3175.pt` + optimizer state, deep-copied per candidate. Parity reproduces the live val utility exactly (1.1389572403748425) on every seed.
- **DOPE frozen:** composite sha `c51a56d0…` before and after on every seed (all 3 DONE lines). Only the 29,211 adapter parameters train.
- **Data:** real S3 packet, 1,219 DL-Train contexts (205 families) and 91 DL-Val contexts (19 families), family overlap 0, every row DL-Train or DL-Val (asserted). No sealed or holdout data read.
- **Partitions:** the two H452 floor-stratified family-grouped 5-fold partitions (salts `h452a/`, `h452b/`; 244/244/244/244/243 contexts, incumbent −2.0 gate failures 4/4/4/3/4 per fold).
- **Compute:** 3 bases × 2 partitions × 5 folds = 30 trainings per seed plus 1 full-train run. Peak RSS 6.4 GB per seed; minimum MemAvailable after the relaunch 53.7–54.2 GB.

## Outcome
**CV (pooled OOF over 2 × 1,219 TRAIN contexts; seeds 1 / 2 / 3).** "vs control" is the paired arm − `pairnf_z40` OOF gain with its family-bootstrap 95% CI.

| Arm | OOF gain (median) | Elo vs incumbent (median) | Non-floor Elo | Strict / utility-strict | Recovered / new gate fails (of 38) | vs control (gain, CI) |
| --- | --- | --- | --- | --- | --- | --- |
| **pairnf_z40** (control) | +0.0048 / +0.0069 / +0.0056 (+0.0056) | +4.7 / +6.6 / +3.8 (+4.7) | +4.3 / +5.9 / +3.3 | 1/3 / 3/3 | 3, 5, 4 / 0 | — |
| hn16_z40 | +0.0044 / +0.0047 / +0.0043 (+0.0044) | +3.4 / +4.6 / +4.0 (+4.0) | +3.0 / +4.2 / +3.6 | 0/3 / 3/3 | 3, 3, 3 / 0 | −0.0004 [−0.0029, +0.0023]; −0.0021 [−0.0052, +0.0001]; −0.0013 [−0.0047, +0.0020] |
| hnfocus_z40 | 0 / 0 / 0 (0) | 0 / 0 / 0 (0) | 0 / 0 / 0 | 0/3 / 0/3 | 0 / 0 | −0.0048 [−0.0098, −0.0005]; −0.0069 [−0.0137, −0.0019]; −0.0056 [−0.0114, −0.0011] |

Elo CIs (family bootstrap): control [−1.3, +10.6] / [+0.7, +12.3] / [−1.8, +9.4]; `hn16_z40` [−1.5, +8.1] / [−0.3, +9.4] / [−0.6, +8.7]. Head-to-head on the contexts where `hn16_z40` and the control differ (88 / 77 / 87): arm better 38 / 33 / 44, control better 50 / 44 / 43.

**val91 regression check** (one read per seed, CV-selected arm trained on all 1,219 TRAIN contexts):

| Seed | Selected arm | Gain | Gain CI | W / T / L | Elo | Changed / kept | New gate failures | not_worse |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | pairnf_z40 | −0.0006 | [−0.0022, +0.0006] | 3/85/3 | 0.0 | 54 / 6 | 0 | false |
| 2 | pairnf_z40 | +0.0011 | [−0.0005, +0.0028] | 4/86/1 | +11.5 | 54 / 5 | 0 | true |
| 3 | hn16_z40 | +0.0005 | [−0.0008, +0.0020] | 3/87/1 | +7.6 | 54 / 4 | 0 | true |

Median val91 gain +0.0005 (Elo +7.6), inside every CI. The pre-registered LIFT rule fails (no selected arm passes strict on every seed, seed 1 not_worse false, no arm beats the control).

## Red-team (`h454_redteam.py`, `h454_aggregate.py`; output under `redteam` / `aggregate` in `metrics.json`)
1. **Control reproduction.** All 30 `pairnf_z40` cells (3 seeds × 2 partitions × 5 folds) have the same adapter sha and identical OOF vectors as H453 (30/30), including the cells re-run after the guard-abort resume. The instrument is unchanged.
2. **hnfocus degeneracy (explained, not a gate bug).** `hnfocus_z40` still changes ~119 picks per fold on average, but the z-margin of every changed pick over the incumbent pick is small: median 0.065, p90 0.17, max 0.34 over all 30 cells (control: median 0.18, p90 0.44, max 0.79; 15% of changes ≥ 0.40). The softmax(z-margin / 0.5) weights concentrate the pairwise gradient on programs ranked at or just above the incumbent pick and push them back toward it, so near-ties are the fixed point and the absolute z ≥ 0.40 gate keeps nothing. Its pairwise loss is lower throughout (11.3 → 11.1 vs 14.1 → 13.7 for pairnf) because the reweighting moves mass onto pairs already near equal, not because it learns more. The 3 seeds have different adapter shas in every cell but identical all-zero OOF vectors; the aggregator exempts an all-zero arm from the fold-gain independence check (`degenerate_all_zero_arms` in `metrics.json`).
3. **hn16 vs control.** `hn16_z40` keeps 69–78 changes per partition vs the control's 99–108 and makes almost no changes the control does not (`arm_only_gain_sum` ≈ 0 on 4 of 6 seed × partition cells). Its decisions are mostly a subset of the control's; the ones it drops are net good (control-only gain sum +0.7 to +4.5 per partition). Hard negatives sharpen the incumbent's top, which shrinks z-margins (median 0.17 vs 0.18) and lowers the number of changes that clear the gate, without raising the win rate.
4. **Seed noise.** Seed disagreement on OOF decisions: control 25–32 contexts per partition, hn16 15–25, hnfocus 0. Paired CIs, not seed-to-seed differences, decide the arm comparison.
5. **Selection / winner's curse.** Selected arm per seed `pairnf_z40` / `pairnf_z40` / `hn16_z40`. Seed 3's pick wins on non-floor Elo by 0.3 (3.62 vs 3.33) and is the bootstrap argmax 53% of the time (seeds 1/2: 83% / 88%). Max-arm non-floor Elo CI [0.0, +10.3] / [+0.3, +11.8] / [0.0, +8.9].
6. **val91 fragility.** Seeds change 4–6 of 91 contexts, 4 of them shared (indices 50, 64, 69, 84; plus 49 on seeds 1–2 and 25 on seed 1). Seeds differ on 2–3 decisions, which moves Elo between 0.0 and +11.5.
7. **Ties.** ~91% of CV decisions and 85–87 of 91 val91 decisions are ties by design (the gate reverts to the incumbent pick). Ties count 0.5 in Elo and never satisfy strict.
8. **Metric direction.** Utility is higher-is-better (negated fitness, −2.0 on gate fail); gains are paired candidate − incumbent on the same contexts; the aggregator recomputes verdicts from per-seed fields.
9. **Leakage.** Folds are family-disjoint and exclude val families (asserted per fold; family overlap 0); hard negatives come from the incumbent's own predictions on training-fold contexts; the pairwise term and hinge use training-fold contexts only; gates use predictions only; val91 read once per seed for the CV-selected arm. No sealed or holdout reads.
10. **Safety.** DOPE `c51a56d0` intact on every DONE line and verified after every training. One 15 GB guard abort per seed from machine-wide memory pressure (not this job's RSS, which stayed at 6.4 GB), resumed cleanly. xbabe2 disk 91% (700 GB free); wave dir 24 MB plus the 442 MB input copy. No large objects committed.

## Meaning
H451–H454 give the same answer four ways: the label-free z ≥ 0.40 gate on the non-floor pairwise step is worth about +4 to +6 Elo over the incumbent on TRAIN-fold CV and is not separable from 0 on val91. Changing which pairs the step sees (hard negatives) or how they are weighted (focus) does not move it, and focusing on decision-flipping pairs destroys the confidence signal the gate depends on. More pair-loss variants are unlikely to help.

## Next step
H457 (`~/jope-data/hwave-h457` on xbabe2; H455 and H456 were used by other work) changes the stabilization axis of the same step instead: `pairnf_z40` (control, bit-for-bit), `avg8_z40` (Polyak/SWA tail average of the same pairnf trajectory over steps 9–16, an eval-only paired arm) and `sam05_z40` (pairnf trained with ASAM ρ 0.5 at lr 3e-4 × 16 steps). Same partitions, selection, strict rule, paired control comparison, 3 seeds.

No adoption nudge was sent to the flywheel owner.

Artifacts: `metrics.json` (`aggregate`: per-arm/seed CV, per-partition Elo and fold gains, gate stats, paired vs-control comparison, val91 regression checks, peaks; `redteam`: floor/non-floor split, partition agreement, decision overlap with the control, seed disagreement, selection bootstrap, val91 per-context agreement), `h454_trainfold_cv.py` (as run), `h454_aggregate.py`, `h454_redteam.py`. Raw logs, losses and fold signatures stay on xbabe2 in `~/jope-data/hwave-h454`.
