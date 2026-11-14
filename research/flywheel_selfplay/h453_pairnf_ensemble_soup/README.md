# H453: decision-variance reduction for the gated pairnf step (ensemble / unanimity / soup; offline, adapter-only)

**Verdict: `NO_LIFT`.** No arm beats the `pairnf_z40` control on any seed. Ensembling or souping 3 perturbed sub-runs makes the gated decisions more reproducible across seeds but not more correct: `ens3_z40` and `soup3_z40` stay at the control's ~+5 Elo (paired arm − control gain CIs span 0 on every seed), and the stricter gates `ens3_z60` and `ens3_agree_z40` are significantly **worse** than the control (paired CI < 0 on all 3 seeds) because they discard good decisions. Strict passes: control 1/3 (seed 2, as in H452), every other arm 0/3. val91 (one read per seed, CV-selected arm) median gain **+0.0005**, every CI spans 0 (s1 `soup3_z40` −0.0000 [−0.0009, +0.0008], s2 `pairnf_z40` +0.0011 [−0.0005, +0.0028], s3 `ens3_z40` +0.0005 [−0.0008, +0.0020]); Elo +3.8 / +11.5 / +3.8 with 0 new gate failures.

## Hypothesis
H452's `pairnf_z40` keeps ~210 of ~2,440 OOF decisions and the 3 seeds disagree on which (25–32 contexts per partition). If those changed decisions are noisy single-adapter calls, averaging 3 independently perturbed sub-runs of the same `pairnf` step should cut decision variance and raise non-floor Elo. Arms (one trained base, 3 sub-runs per fold; sub 0 reproduces H452 `pairnf` bit-for-bit):
- `pairnf_z40` — sub 0 only, z ≥ 0.40 (control, H452 reproduction);
- `ens3_z40`, `ens3_z60` — prediction-mean of the 3 sub-runs, gate on the mean prediction's z at 0.40 / 0.60;
- `ens3_agree_z40` — ensemble pick kept only if every sub-run picks the same program with its own z ≥ 0.40;
- `soup3_z40` — weight-space mean of the 3 sub-runs (one adapter), z ≥ 0.40.

Selection (pre-registered, Elo only): per seed, max mean non-floor OOF Elo over the two partitions (tie: pooled gain). Strict rule unchanged from H452. New in this write-up: an arm only counts if it also beats the control in a paired family bootstrap on the same OOF contexts.

## Setup
- **Code:** `h453_trainfold_cv.py` (as run; sha256 verified in the scratch dir), CPU only, `nice 19`, 16 threads per seed, self-abort at MemAvailable < 15 GB, against the scratch copy `~/jope-data/hwave-h453` on xbabe2, never the live run directory.
- **Runs:** seeds 1/2/3 launched 2026-11-13 21:14 / 21:24 / 21:34 UTC on xbabe2 (mem-checked at ≥ 38 GB), DONE 2026-11-14 06:37 / 06:20 / 06:58 UTC. No guard aborts, no relaunches (waiter: "all seeds done").
- **Start point:** incumbent `round-3175.pt` + optimizer state, deep-copied per candidate. Parity reproduces the live val utility exactly (1.1389572403748425) on every seed.
- **DOPE frozen:** composite sha `c51a56d0…` before and after on every seed (all 3 DONE lines). Only the 29,211 adapter parameters train.
- **Data:** real S3 packet, 1,219 DL-Train contexts (205 families) and 91 DL-Val contexts (19 families), family overlap 0, every row DL-Train or DL-Val (asserted). No sealed or holdout data read.
- **Partitions:** the two H452 floor-stratified family-grouped 5-fold partitions (salts `h452a/`, `h452b/`; 244/244/244/244/243 contexts, incumbent −2.0 gate failures 4/4/4/3/4 per fold).
- **Compute:** 1 base × 3 sub-runs × 2 partitions × 5 folds = 30 trainings per seed plus 3 full-train sub-runs; gated/ensemble arms are eval-only. Peak RSS 6.4 GB per seed, minimum MemAvailable 47.2–48.0 GB.

## Outcome
**CV (pooled OOF over 2 × 1,219 TRAIN contexts; seeds 1 / 2 / 3).** "vs control" is the paired arm − `pairnf_z40` OOF gain with its family-bootstrap 95% CI.

| Arm | OOF gain (median) | Elo vs incumbent (median) | Non-floor Elo | Strict / utility-strict | Recovered / new gate fails (of 38) | vs control (gain, CI) |
| --- | --- | --- | --- | --- | --- | --- |
| **pairnf_z40** (control) | +0.0048 / +0.0069 / +0.0056 (+0.0056) | +4.7 / +6.6 / +3.8 (+4.7) | +4.3 / +5.9 / +3.3 | 1/3 / 3/3 | 3, 5, 4 / 0 | — |
| ens3_z40 | +0.0036 / +0.0061 / +0.0059 (+0.0059) | +4.8 / +5.7 / +5.1 (+5.1) | +4.6 / +5.2 / +4.6 | 0/3 / 2/3 | 2, 4, 4 / 0 | −0.0012 [−0.0035, +0.0001]; −0.0007 [−0.0031, +0.0006]; +0.0003 [−0.0002, +0.0009] |
| ens3_z60 | +0.0004 / +0.0004 / +0.0003 (+0.0004) | +0.6 / +0.7 / +0.4 (+0.6) | +0.6 / +0.7 / +0.4 | 0/3 / 0/3 | 0 / 0 | −0.0044 [−0.0093, −0.0003]; −0.0065 [−0.0133, −0.0017]; −0.0052 [−0.0108, −0.0009] |
| ens3_agree_z40 | +0.0026 / +0.0043 / +0.0021 (+0.0026) | +3.1 / +4.1 / +2.7 (+3.1) | +2.9 / +3.8 / +2.6 | 0/3 / 1/3 | 2, 3, 1 / 0 | −0.0022 [−0.0050, −0.0002]; −0.0026 [−0.0056, −0.0003]; −0.0035 [−0.0069, −0.0008] |
| soup3_z40 | +0.0036 / +0.0063 / +0.0059 (+0.0059) | +4.8 / +6.1 / +4.8 (+4.8) | +4.6 / +5.6 / +4.3 | 0/3 / 2/3 | 2, 4, 4 / 0 | −0.0012 [−0.0035, +0.0001]; −0.0006 [−0.0029, +0.0009]; +0.0004 [−0.0002, +0.0009] |

Elo CIs (family bootstrap) for the two leading arms: control [−1.3, +10.6] / [+0.7, +12.3] / [−1.8, +9.4]; `soup3_z40` [−1.3, +11.0] / [−0.3, +12.5] / [−0.8, +10.5]; `ens3_z40` [−1.3, +10.8] / [−0.4, +11.8] / [−0.5, +10.8]. Every non-control Elo CI spans 0.

**val91 regression check** (one read per seed, CV-selected arm trained on all 1,219 TRAIN contexts):

| Seed | Selected arm | Gain | Gain CI | W / T / L | Elo | Changed / kept | New gate failures | not_worse |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | soup3_z40 | −0.0000 | [−0.0009, +0.0008] | 3/86/2 | +3.8 | 54 / 5 | 0 | false |
| 2 | pairnf_z40 | +0.0011 | [−0.0005, +0.0028] | 4/86/1 | +11.5 | 54 / 5 | 0 | true |
| 3 | ens3_z40 | +0.0005 | [−0.0008, +0.0020] | 3/86/2 | +3.8 | 53 / 5 | 0 | true |

Median val91 gain +0.0005 (Elo +3.8), inside every CI. The pre-registered LIFT rule fails (no strict pass for any selected arm except the control, seed 1 not_worse false, and no arm beats the control).

## Red-team (`h453_redteam.py`, `h453_aggregate.py`; output under `redteam` / `aggregate` in `metrics.json`)
1. **Control reproduction.** All 30 `pairnf_z40` cells (3 seeds × 2 partitions × 5 folds) have the same adapter sha and identical OOF vectors as H452. The instrument is unchanged.
2. **Arm vs control, not just vs incumbent.** `ens3_z40` / `soup3_z40` make 95–103 of the control's 99–108 changes per partition; they differ on 31–44 contexts per seed and split them roughly evenly. On `h452b/` they miss one −2.0 recovery the control gets (control-only gain sum +2.2 to +2.9), which is most of their lower OOF gain. `ens3_z60` keeps only 6–8 changes per partition and `ens3_agree_z40` 73–81; the decisions they drop are net good (+3.7 to +9.0 and +1.5 to +6.3 utility per partition), hence their paired CIs below 0.
3. **Variance vs bias.** Averaging works as intended on variance: seed disagreement falls from 25–32 contexts per partition (control) to 12–24 (`ens3_z40`, `soup3_z40`). Elo does not move (~+5). The gated decisions are now more reproducible but not more often right, so decision variance from the init perturbation is not the bottleneck.
4. **Floors / fold artifact.** Floors are stratified 4/4/4/3/4 per fold in both partitions; no fold carries the old fold-1 +0.05. For the control, floor recovery contributes +0.0017 to +0.0055 per partition and non-floor gain +0.0018 to +0.0027 (positive in all 6 seed × partition cells). About 70–75% of the gated changes repeat in both partitions, almost all with the same sign.
5. **Selection / winner's curse.** The selected arm differs by seed (`soup3_z40` / `pairnf_z40` / `ens3_z40`). Seed 1's pick is a tie on non-floor Elo (4.63 for both `soup3_z40` and `ens3_z40`) broken by pooled gain. The selected arm is the bootstrap argmax 40% / 62% / 78% of the time; max-arm non-floor Elo CI [−0.0, +10.8] / [+0.7, +12.3] / [−0.2, +10.0].
6. **val91 fragility.** All seeds change 5 of 91 contexts, 4 of them shared (indices 50, 64, 69, 84, plus 25 or 49). Seeds differ on 1–3 decisions, which moves Elo between +3.8 and +11.5.
7. **Ties.** ~91% of CV and 86/91 val91 decisions are ties by design (the gate reverts to the incumbent pick). Ties count 0.5 in Elo and never satisfy strict.
8. **Metric direction.** Utility is higher-is-better (negated fitness, −2.0 on gate fail); gains are paired candidate − incumbent on the same contexts; the aggregator recomputes verdicts from per-seed fields.
9. **Leakage.** Folds are family-disjoint and exclude val families (asserted per fold); the pairwise term and hinge use training-fold contexts only; gates use predictions only; val91 read once per seed for the CV-selected arm. No sealed or holdout reads.
10. **Safety.** DOPE `c51a56d0` intact on every DONE line and verified after every training. No OOM or guard abort. xbabe2 disk 91% (712 GB free); wave dir 25 MB plus the 442 MB input copy. No large objects committed.

## Meaning
H451–H453 converge on one small, reproducible effect: the label-free z ≥ 0.40 gate on the non-floor pairwise step gives about +4 to +6 Elo over the incumbent on TRAIN-fold CV and is not separable from 0 on val91. H453 removes decision variance as the explanation: ensembles and soups reproduce the same decisions with less seed noise and the same win rate. What limits the gated arm is the bias of what the step learns.

## Next step
H454 (`~/jope-data/hwave-h454` on xbabe2) changes which pairs the `pairnf` step learns from. The 192-program training slate is the top-48 programs by true fitness plus random fill, so the programs the adapter itself would switch to are rarely in it. Arms: `pairnf_z40` (control, bit-for-bit), `hn16_z40` (incumbent's predicted top-16 forced into every slate) and `hnfocus_z40` (hn16 plus Bradley-Terry pairs reweighted by softmax(z-margin / 0.5) toward the decision set). Same partitions, selection, strict rule, paired control comparison, 3 seeds.

No adoption nudge was sent to the flywheel owner.

Artifacts: `metrics.json` (`aggregate`: per-arm/seed CV, per-partition Elo and fold gains, gate stats, paired vs-control comparison, val91 regression checks, peaks; `redteam`: floor/non-floor split, partition agreement, decision overlap with the control, seed disagreement, selection bootstrap, val91 per-context agreement), `h453_trainfold_cv.py` (as run), `h453_aggregate.py`, `h453_redteam.py`. Raw logs and fold signatures stay on xbabe2 in `~/jope-data/hwave-h453`.
