# H452: pairwise-loss confirmation on two fresh floor-stratified partitions, with k25 / absolute-z gates (offline, adapter-only)

**Verdict: `NO_LIFT`.** The selected arm `pairnf_z40` (non-floor pairwise loss plus an absolute z ≥ 0.40 confidence gate) is the same on all 3 seeds. On val91 its median gain is **+0.0010**, inside every seed's CI (s1 −0.0006 [−0.0022, +0.0006], s2 +0.0011 [−0.0005, +0.0028], s3 +0.0010 [−0.0005, +0.0027]). Elo is 0.0 / +11.5 / +11.5 (W/T/L 3/85/3, 4/86/1, 4/86/1) with 0 new gate failures. In CV, `pairnf_z40` passes the strict rule on **1 of 3 seeds** (s2: Elo +6.6 [+0.7, +12.3], non-floor Elo +5.9 [+0.2, +11.6]). No other arm passes on any seed. H452 confirms H451's lead on fresh partitions: confidence-gated pairwise changes win slightly more than they lose on non-floor contexts. But the effect is about +4 to +6 Elo, the same size as the CI half-width, and val91 cannot resolve it (only 5–6 of 91 decisions change).

## Hypothesis
H451's `pairnf_k25` showed the first consistent non-floor signal (non-floor Elo about +5), but on one fixed partition whose fold 1 held 7 of the 19 incumbent TRAIN gate failures, and from a best-of-6 pick. H452 tests whether that survives:
- **Instrument:** two fresh family-grouped 5-fold partitions (salts `h452a/`, `h452b/`), each stratified so the families holding the incumbent's 19 TRAIN −2.0 failures are spread round-robin (4/4/4/3/4 per fold in both).
- **Bases:** `pairnf` (H451 config: trust_big step + gate hinge 0.5 + non-floor Bradley-Terry vs the incumbent argmin, pair_w 1) and `pairnf_w3` (pair_w 3, listwise weight 0.5).
- **Label-free decision gates:** none, `_k25` (top 25% of changed picks by z-margin, H451 rule), and `_z40` (absolute per-context z ≥ 0.40, set-independent).
- **Selection** (pre-registered): max mean non-floor OOF Elo over the two partitions. **Strict** adds non-floor Elo CI low > 0 and Elo > 0 on each partition to the H448 rule.

## Setup
- **Code:** `h452_trainfold_cv.py` (as run, sha256 in the scratch dir), CPU only, `nice 19`, 16 threads per seed, self-abort at MemAvailable < 15 GB. It ran against the scratch copy `~/jope-data/hwave-h452`, never the live run directory.
- **Machines:** seeds started on xbabe1 (2026-11-13 03:56 MST). All 3 hit the 15 GB guard there (~05:30 MST), and the directory was moved to xbabe2 and resumed from `results/fold_sig` (06:13 / 06:23 / 06:33 MST). Seeds finished at 12:12 / 12:23 / 12:36 MST. A guard abort leaves no partial state; a resume rereads completed fold signatures, and the aggregator asserts distinct adapters per cell.
- **Start point:** incumbent `round-3175.pt` plus optimizer state, deep-copied per candidate. Parity reproduces the live val utility exactly (1.1389572403748425, asserted on every seed).
- **DOPE frozen:** composite sha `c51a56d0…` before and after on every seed. Only the 29,211 adapter parameters train.
- **Data:** real S3 packet, 1,219 DL-Train contexts (205 families) and 91 DL-Val contexts (19 families), family overlap 0. Every row is DL-Train or DL-Val (asserted). No sealed or holdout data was read.
- **Compute:** 2 bases × 2 partitions × 5 folds = 20 trainings per seed; gated arms are eval-only. Peak RSS 6.3 GB per seed, minimum MemAvailable 42–45 GB on xbabe2.

## Outcome
**CV (pooled OOF over 2 × 1,219 TRAIN contexts; seeds 1 / 2 / 3):**

| Arm | OOF gain | Elo | Elo CI (s2) | Non-floor Elo (mean of partitions) | Recovered / new gate fails (s1, of 38) | Strict |
| --- | --- | --- | --- | --- | --- | --- |
| pairnf | +0.0110 / +0.0126 / +0.0144 | −9.3 / −7.8 / −10.3 | [−24.2, +7.2] | −13.9 / −12.5 / −14.9 | 31 / 9 | 0/3 |
| pairnf_k25 | +0.0070 / +0.0070 / +0.0090 | +4.7 / +3.1 / +3.8 | [−4.6, +10.5] | +3.9 / +2.3 / +2.8 | 6 / 0 | 0/3 |
| **pairnf_z40** | **+0.0048 / +0.0069 / +0.0056** | **+4.7 / +6.6 / +3.8** | **[+0.7, +12.3]** | **+4.3 / +5.9 / +3.3** | 3 / 0 | **1/3** |
| pairnf_w3 | +0.0066 / +0.0052 / +0.0064 | −13.4 / −15.1 / −15.0 | [−31.4, +0.0] | −17.5 / −19.6 / −19.3 | 27 / 10 | 0/3 |
| pairnf_w3_k25 | +0.0080 / +0.0061 / +0.0090 | +4.6 / +3.0 / +3.4 | [−5.1, +10.3] | +3.3 / +1.9 / +2.0 | 9 / 1 | 0/3 |
| pairnf_w3_z40 | +0.0023 / +0.0033 / +0.0023 | +1.7 / +1.7 / +2.4 | [−2.1, +5.6] | +1.4 / +1.3 / +2.2 | 2 / 0 | 0/3 |

`pairnf_z40` W/T/L (pooled, s1/s2/s3): 122/2227/89, 128/2228/82, 115/2235/88. Per-partition Elo: s1 +4.8/+4.6, s2 +6.0/+7.1, s3 +2.9/+4.8. Median over seeds: OOF gain +0.0056, Elo +4.7, non-floor Elo +4.3.

**val91 regression check** (one read per seed, `pairnf_z40` trained on all 1,219 TRAIN contexts):

| Seed | Gain | Gain CI | W / T / L | Elo | Changed / kept | New gate failures | not_worse |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | −0.0006 | [−0.0022, +0.0006] | 3/85/3 | 0.0 | 54 / 6 | 0 | false |
| 2 | +0.0011 | [−0.0005, +0.0028] | 4/86/1 | +11.5 | 54 / 5 | 0 | true |
| 3 | +0.0010 | [−0.0005, +0.0027] | 4/86/1 | +11.5 | 56 / 5 | 0 | true |

Median val91 gain +0.0010, inside the CI on every seed, so the pre-registered LIFT rule fails.

## Red-team (`h452_redteam.py`; output under `redteam` in `metrics.json`)
The red-team rebuilds both stratified partitions from the incumbent's per-context TRAIN utility, dumped by the H453 probe from the same packet and checkpoint (no model, no val labels). It asserts that they reproduce the logged fold sizes and floors per fold (4/4/4/3/4).

1. **Floor recovery vs non-floor.** Ungated `pairnf` / `pairnf_w3` again gain only from floors. They recover 13–16 of 19 per partition (+0.026 to +0.035), lose −0.017 to −0.028 on the rest, add 3–7 new gate failures per partition, and have non-floor Elo −7 to −18 (several CIs below 0 for `w3`). The gated arms are the opposite. `pairnf_z40`'s non-floor gain is positive in **all 6** seed × partition cells (+0.0018 to +0.0027 per context, W/L 55–62 vs 40–47), with non-floor Elo +2.6 to +6.4. Only one of those 6 CIs (s2 `h452b/`, [+0.9, +12.4]) clears 0. Floors account for about 1/3 to 2/3 of its total gain (1–3 of 19 recovered per partition, 0 new).
2. **Partition robustness (the H451 confound).** With floors spread evenly, the gated signal holds on both fresh partitions. Of the contexts `pairnf_z40` changes in either partition, 70–75% change in both, and 81–84 of those 87–88 move in the same direction. So the decision depends on the context rather than on the fold, and H451's fold-1 artefact does not explain it.
3. **Selection / winner's curse.** All 6 arms × 3 seeds are reported. Selection used non-floor OOF Elo only, with no val input. `pairnf_z40` wins on every seed, but on 1,000 family bootstraps it stays the argmax only 43% / 91% / 37% of the time (`pairnf_k25` and `pairnf_w3_k25` are close). The max-arm non-floor Elo CI is [−0.6, +11.5] / [+0.3, +12.0] / [−0.7, +10.7]. The single strict pass (s2) is not robust to best-of-6 optimism.
4. **Seed noise (seed 1 vs seeds 2–3).** The fold-gain vectors differ for every seed pair and arm, and every cell has distinct adapter shas (aggregator asserts). Seeds disagree on 25–32 OOF contexts per partition for `pairnf_z40`. On val91 all 3 seeds change the same 5 contexts (indices 49, 50, 64, 69, 84). Seeds 2 and 3 make the same picks except at index 69 (+0.030 vs +0.021). Seed 1 differs at two contexts: it also changes index 25 (−0.044), and at index 49 it picks a different program (−0.057 where seeds 2–3 gain +0.045). Those two decisions alone move seed 1 from +11.5 Elo to 0.0. So the seed1-vs-seeds-2–3 split is 2 of 91 decisions, not a different model, and it shows how fragile val91 Elo is at 5–6 changes.
5. **Ties.** 86/91 val91 ties and about 91% CV ties are by design: the gate reverts low-confidence changes to the incumbent pick. Ties count 0.5 in Elo and never satisfy strict. All Elo above 0 comes from W > L on kept changes.
6. **Metric direction.** Utility is higher-is-better (negated fitness, −2.0 on gate fail). The aggregator recomputes strict from per-seed fields. Gains are paired candidate − incumbent on the same contexts.
7. **Leakage.** Folds are family-disjoint and exclude val families (asserted per fold). The pairwise term and the hinge use training-fold contexts only. Gates use predictions only. val91 was read once per seed, for the CV-selected arm.
8. **Safety.** DOPE `c51a56d0` is intact on every DONE line and checked after every training. No sealed or holdout reads. The xbabe1 guard aborts were clean self-aborts with no OOM kill. xbabe2 disk is at 90% (764 GB free), and the wave directory is 23 MB plus the 442 MB input copy. No large objects are committed.

## Meaning
A strong pairwise objective on its own makes more changes and loses more of them. A label-free confidence gate turns the same model into a small, partition-robust non-floor winner (about +4 to +6 Elo). That is real enough to reproduce on fresh stratified partitions and 3 seeds, but too small to separate from 0 after best-of-6 selection, and val91 is too small to adjudicate it (5–6 changed decisions). The bottleneck is now the **reliability of individual changed decisions**, not the training objective.

## Next step
H453 (`~/jope-data/hwave-h453` on xbabe2) attacks decision variance. It trains 3 independently perturbed sub-runs of the same `pairnf` step per fold (sub 0 reproduces H452 `pairnf` bit-for-bit) and compares:
- `pairnf_z40` (control);
- a prediction-mean ensemble with the gate at z ≥ 0.40 / 0.60 (`ens3_z40`, `ens3_z60`);
- a unanimity gate where all 3 sub-runs must pick the same program, each with z ≥ 0.40 (`ens3_agree_z40`);
- a weight-space soup (`soup3_z40`).

It uses the same two stratified partitions, the same selection and strict rule, and 3 seeds.

No adoption nudge was sent to the flywheel owner.

Artifacts: `metrics.json` (`aggregate`: per-arm/seed CV, per-partition Elo and fold gains, gate stats, val91 regression checks, peaks; `redteam`: floor/non-floor split, partition agreement, seed disagreement, selection bootstrap, val91 per-context agreement), `h452_trainfold_cv.py` (as run), `h452_aggregate.py`, `h452_redteam.py`. Raw logs and fold signatures stay on xbabe2 in `~/jope-data/hwave-h452`.
