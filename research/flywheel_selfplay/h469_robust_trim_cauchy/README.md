# H469: robust per-context losses (trimmed mean, soft Cauchy) on the gated pairnf step (offline, adapter-only)

**Verdict: `NO_LIFT`.** No robust-loss arm beats the `pairnf_z40` control on any seed (every paired family-bootstrap CI spans 0 or is below 0). Median OOF Elo vs the incumbent: control **+4.7**, `trim10_z40` **+4.8**, `trim25_z40` **+2.7**, `cauchy_z40` **+1.6**. Strict passes: control 1/3, trim10 0/3, trim25 0/3, cauchy 0/3. `trim10_z40` is indistinguishable from the control (same 57.6% vs 57.7% non-floor win rate, keeps 96% of the control's changes). `trim25_z40` and `cauchy_z40` shrink the update: they keep 54% / 40% of the control's changes, so the z ≥ 0.40 gate starves them. trim25's pooled gain is floor-driven (non-floor gain ≈ 0), and cauchy is significantly *worse* than the control on all 3 seeds (paired CI below 0). CV selected `pairnf_z40` / `pairnf_z40` / `trim10_z40`. val91 median gain +0.0006 (Elo +7.6), every CI spans 0, not_worse 2/3 (seed 1 control: −0.0006, 3 W / 3 L).

## Hypothesis (pre-registered in #2363)
A few very high-loss contexts per batch (mostly gate-fail / −2.0 floor rows) dominate the 16-step update. A robust per-context loss that caps their pull should make the kept changes more precise on non-floor contexts. Arms, all on the control's random streams (`rng_arm = pairnf`), 16 steps, lr 3e-4, z ≥ 0.40 gate unchanged: `trim10_z40` / `trim25_z40` (drop the top 10% / 25% per-context losses per step, detached ranking) and `cauchy_z40` (row → m·log1p(row/m), m = detached batch median).

## Pre-registered rules (unchanged from H452–H468)
Selection takes the max mean non-floor OOF Elo over the two partitions. Strict requires pooled OOF gain > 0.002 with gain CI low > 0, pooled Elo > 0 with Elo CI low > 0, non-floor Elo CI low > 0, and Elo > 0 on each partition. An arm counts only if it also beats the control in the paired family bootstrap. Ties never win. LIFT needs that on every seed, plus val91 not_worse on every seed and a clean red-team. Pre-declared: floor-driven pooled gains are not lift; an arm that keeps < 0.5× the control's changes is reported as gate-starved.

## Setup
- **Code:** `h469_trainfold_cv.py` as pre-registered (sha256 `36a6d776…`, re-verified by the waiter before every launch and after the run). CPU only, `nice 19`, 16 threads per seed, on the scratch copy `~/jope-data/hwave-h469` on xbabe2 (a copy of `round-3175.pt`, sha256 `1d6ea900…`). Never the live run dir, never xbabe0 or xbabe3. No GPU, so no `qualification.lock`.
- **Runs:** seeds 1/2/3 launched 2026-11-18 08:40 / 09:00 / 09:21 UTC. Each seed hit its own 15 GB MemAvailable guard once (other owners' load: 13.2 / 14.8 / 12.2 GB) and the waiter relaunched it resume-safe via `results/fold_sig` (15:57 / 16:37 / 16:58 UTC). DONE at 19:05 / 19:59 / 19:52 UTC. The waiter exited cleanly at 20:01 UTC (13:01 MST).
- **Data and model:** the real S3 packet, 1,219 DL-Train contexts (205 families) and 91 DL-Val contexts (19 families), family overlap 0. DOPE `c51a56d0…` stayed frozen (before = after on every seed); only the 29,211 adapter params trained. Incumbent val parity 1.1389572403748425 on every seed.
- **Compute:** peak RSS 6.3 GB per seed; min MemAvailable 16.4 / 38.6 / 40.3 GB after relaunch.

## Outcome
**CV** (pooled OOF over 2 × 1,219 TRAIN contexts; values are seeds 1 / 2 / 3, median in parentheses):

| Arm | OOF gain | Elo vs incumbent | Elo 95% CI (family bootstrap) | Non-floor Elo | W / L (ties rest of 2,438) | Strict (utility-strict) | Recovered / new gate fails (of 38) | vs control (gain, CI) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **pairnf_z40** (control) | +0.0048 / +0.0069 / +0.0056 (+0.0056) | +4.7 / +6.6 / +3.8 (+4.7) | [−1.3, +10.6] / [+0.7, +12.3] / [−1.8, +9.4] | +4.3 / +5.9 / +3.3 | 122/89, 128/82, 115/88 | 1/3 (3/3) | 3, 5, 4 / 0 | — |
| trim10_z40 | +0.0059 / +0.0061 / +0.0073 (+0.0061) | +4.7 / +5.1 / +4.8 (+4.8) | [−1.0, +10.1] / [−0.3, +10.5] / [−0.5, +9.9] | +4.2 / +4.6 / +4.1 | 118/85, 120/84, 114/80 | 0/3 (3/3) | 4, 4, 6 / 0 | +0.0011 [−0.0009, +0.0038]; −0.0008 [−0.0033, +0.0013]; +0.0017 [−0.0008, +0.0049] |
| trim25_z40 | +0.0044 / +0.0056 / +0.0055 (+0.0055) | +2.4 / +2.7 / +3.6 (+2.7) | [−1.3, +6.2] / [−1.2, +6.7] / [−0.4, +7.9] | +1.7 / +2.0 / +2.8 | 63/46, 66/47, 70/45 | 0/3 (1/3) | 5, 5, 6 / 2, 1, 2 | −0.0004 [−0.0064, +0.0054]; −0.0013 [−0.0065, +0.0040]; −0.0001 [−0.0054, +0.0055] |
| cauchy_z40 | +0.0011 / +0.0007 / +0.0009 (+0.0009) | +1.9 / +0.7 / +1.6 (+1.6) | [−1.6, +5.4] / [−2.6, +4.1] / [−1.9, +5.2] | +1.9 / +0.7 / +1.6 | 50/37, 42/37, 48/37 | 0/3 (0/3) | 0, 0, 0 / 0 | −0.0037 [−0.0083, −0.0001]; −0.0062 [−0.0127, −0.0017]; −0.0047 [−0.0098, −0.0008] |

Head-to-head on contexts where the arm and the control differ:
- trim10_z40 (74 / 74 / 75): arm better 43 / 35 / 43, control better 31 / 39 / 32 (Elo +56.8 / −18.8 / +51.3, every CI spans 0).
- trim25_z40 (143 / 137 / 137): arm better 66 / 60 / 71, control better 77 / 77 / 66 (every CI spans 0).
- cauchy_z40 (129 / 136 / 123): arm better 56 / 48 / 56, control better 73 / 88 / 67 (Elo −46.1 / −105.3 / −31.2).

**val91 regression check** (the CV-selected arm trained on all 1,219 TRAIN contexts):

| Seed | Selected arm | Gain | Gain CI | W / T / L | Elo | Changed / kept | New gate failures | not_worse |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | pairnf_z40 | −0.0006 | [−0.0022, +0.0006] | 3/85/3 | +0.0 | 54 / 6 | 0 | false |
| 2 | pairnf_z40 | +0.0011 | [−0.0005, +0.0028] | 4/86/1 | +11.5 | 54 / 5 | 0 | true |
| 3 | trim10_z40 | +0.0006 | [−0.0007, +0.0019] | 4/85/2 | +7.6 | 54 / 6 | 0 | true |

The pre-registered LIFT rule fails: no non-control arm beats the control on any seed, the selected arm passes strict on 1/3 seeds (the control itself, seed 2), and val91 is not_worse on 2/3.

## Red-team (`h469_redteam.py`, `h469_aggregate.py`, copied from H468; output under `redteam` / `aggregate` in `metrics.json`)
1. **Control reproduction.** All 30 `pairnf_z40` cells have the same adapter sha and identical OOF vectors as H468 (30/30).
2. **Why trim10_z40 and the control both give Elo +4.70 on seed 1.** Arithmetic coincidence, not a no-op. Pooled Elo is a function of the pooled score (W + T/2) / N only, and both cells have W − L = +33 over N = 2,438 (122 / 2,227 / 89 vs 118 / 2,235 / 85), so the scores (and Elos) are equal to the last digit. The family-bootstrap CIs differ ([−1.29, +10.62] vs [−1.02, +10.09]) because the decisions differ. trim10 is active: each step draws 1,024 contexts, and trim keeps `ceil(0.9 × 1024)` = 922, dropping the 102 highest-loss rows. Adapter shas differ from the control's in 30/30 cells, the OOF vectors differ on 74 / 74 / 75 decisions per seed, the gains differ (+0.0059 vs +0.0048 on seed 1), and seeds 2/3 give different Elos (+5.13 vs +6.56, +4.85 vs +3.85). The new `trim10_vs_control_elo_equality` block in `redteam.json` records this (net W − L equal on seed 1 only).
3. **trim10 (mechanism).** It keeps 601 changes over 6 cells vs the control's 624, at a 57.6% non-floor pass win rate (338 W / 249 L) vs the control's 57.7% (353 / 259). Dropping the top 10% of per-context losses changes which ~10% of decisions get kept but not their quality.
4. **trim25 (floor-driven, borderline gate-starved).** It keeps 337 vs 624 changes (0.54×; 0.46–0.62 per cell, below 0.5 on 2 of 6 cells). Its non-floor pass win rate is 57.9%, but on half the kept count, and nearly all of its pooled gain comes from recovering gate-fail contexts (floor gain +0.0035 to +0.0053 per partition vs non-floor gain −0.0009 to +0.0025) while it adds 1–2 new gate failures per seed (the only arm that does). Per the pre-registration this is floor-driven, not lift.
5. **cauchy (gate-starved).** It keeps 251 vs 624 changes (0.40×, below 0.5 in every cell) and recovers no floors; its non-floor pass win rate drops to 55.8%. The log1p loss shrinks the effective step (gradient weight 1/(1+k)), so z-margins fall under the 0.40 gate. Per the pre-registration this null is "gate-starved", not evidence against soft robust losses at matched displacement; the paired CI is below 0 on all 3 seeds.
6. **Seed noise.** Seed disagreement on OOF decisions per partition: control 25–32, trim10 22–33, trim25 17–28, cauchy 7–16. `seed_independence` holds for every arm × seed pair, and adapter shas are distinct in every cell. 6 identical fold OOF pairs (`identical_fold_oof`) all have distinct adapters and small kept sets (3–23), the same discrete-decision effect explained in H468.
7. **Selection / winner's curse.** Selection used TRAIN-fold CV only (non-floor Elo), never val. The selection bootstrap picks the selected arm as argmax only 46% / 79% / 54% of the time; the max-arm non-floor Elo CI is [−0.4, +10.4] / [+0.6, +11.8] / [−0.4, +9.3]. No arm or seed was dropped or re-run selectively.
8. **Resumed partials.** Every seed was relaunched once after its own 15 GB guard abort (seed 1 at p1cv4/trim25 step 14, seed 2 at p1cv3/trim25 step 7, seed 3 at p0cv0/cauchy step 0). The relaunch reads finished cells only from their own `{arm}_{tag}_seed{N}.json` and retrains the interrupted cell from scratch on the same keyed streams (fully deterministic: the control reproduces H468 bit-for-bit, and all 30 control cells match). Aggregation reads only fold_sig, and every cell is complete (no NaNs, asserted). 0 OOM kills.
9. **Leakage.** Folds are family-disjoint and exclude val families (asserted, family overlap 0). The red-team rebuilds both partitions and matches the logged fold maps (floors 4/4/4/3/4). Trimming and the Cauchy scale use only detached training-batch losses. No sealed or holdout reads: the only holdout, the sealed meta-test group, is not in the packet and was never touched.
10. **Safety.** DOPE `c51a56d0` is intact on every DONE line (before = after). No other owner's job was paused or killed. The wave scripts contain no ssh, rsync, xbabe0 or xbabe3 target. xbabe2 disk is at 95% (~382 GB free), and the wave dir is ~467 MB. No large objects are committed.

## Meaning
Robust per-context losses don't help on this instrument. Trimming 10% is neutral; harder trimming and the soft Cauchy loss shrink the update so the z ≥ 0.40 gate keeps fewer changes, and what is kept is no better (trim25) or worse (cauchy). The heavy-context hypothesis is not supported: the high-loss rows are not what holds the ~58% non-floor win rate down. H451–H469 agree that the gated step is worth about +4 to +6 OOF Elo and is not separable from 0 on val91.

## Next step
H470 (`~/jope-data/hwave-h470`, pre-registered separately) tests annealed Gaussian gradient-noise injection (scale-free, per-tensor) on the same instrument and streams, the one exploration lever not yet tried in H443–H469.

No adoption nudge was sent to the flywheel owner.

Artifacts:
- `metrics.json`: pre-registration fields plus `aggregate` (fold-gain vectors dropped for size) and `redteam` as produced on xbabe2.
- `h469_trainfold_cv.py` (as run), `h469_aggregate.py`, `h469_redteam.py`.

Raw logs, losses and fold signatures stay on xbabe2 in `~/jope-data/hwave-h469`.
