# H463: recalibrated gate and annealed dose on the pairnf step (offline, adapter-only)

**Verdict: `NO_LIFT`.** No arm beats the `pairnf_z40` control on every seed. `s32cos_z40` (32 steps, 2-step warmup, cosine decay to 0, on the control's random streams) is the best arm on the median: OOF gain +0.0069 vs +0.0056, Elo +5.1 vs +4.7, utility-strict 3/3. But its Elo CI spans 0 on every seed (strict 0/3), and its paired gain over the control is +0.0033 [+0.0002, +0.0082] / +0.0001 [−0.0012, +0.0013] / +0.0010 [−0.0009, +0.0033], so only seed 1 clears 0. `s32_z200` (the H461 s32 adapter with a count-matched z ≥ 2.00 gate) keeps about as many changes as the control but at a worse win rate: paired CI spans 0 on every seed, head-to-head Elo −25 to −40, and 4 new gate failures per seed. `s32cos_z200` is degenerate: the annealed adapter has no z-margin ≥ 2.00, so it keeps 0 changes and is significantly below the control (paired CI < 0 on all 3 seeds). CV selected `s32cos_z40` / `pairnf_z40` / `s32cos_z40`. val91 (one regression check per seed) median gain is **+0.0011** (Elo +11.5). Every CI spans 0 and every seed is not_worse. No breakthrough.

## Hypothesis (pre-registered in #2333)
H461 showed that 32 pairnf steps inflate z-margins about 7×, so the z ≥ 0.40 gate stops filtering. H463 asked whether the 32-step dose helps once the gate is recalibrated (label-free, `s32_z200`) or once the dose is annealed (`s32cos_z40`, `s32cos_z200`), against the bit-for-bit control.

## Pre-registered rules (unchanged from H452–H461)
Selection is by the max mean non-floor OOF Elo over the two partitions. Strict means pooled OOF gain > 0.002 and gain CI low > 0, pooled Elo > 0 and Elo CI low > 0, non-floor Elo CI low > 0, and Elo > 0 on each partition. An arm counts only if it also beats the control in the paired family bootstrap. Ties never win. LIFT needs that on every seed, plus val91 not_worse on every seed and a clean red-team.

## Setup
- **Code:** `h463_trainfold_cv.py` as pre-registered (sha256 `f85ea211…`, re-verified after the run). CPU only, `nice 19`, 16 threads per seed, scratch copy `~/jope-data/hwave-h463` on xbabe2, never the live run dir, never xbabe0. No GPU and no `qualification.lock`.
- **Runs:** seeds 1/2/3 launched 2026-11-15 15:41 / 15:51 / 16:01 UTC. DONE at 2026-11-16 01:47 / 01:58 / 02:06 UTC with no guard aborts, no relaunches and no teardown hang (`os._exit` after DONE). The waiter exited cleanly at 02:06 UTC.
- **Data and model:** the real S3 packet, 1,219 DL-Train contexts (205 families) and 91 DL-Val contexts (19 families), family overlap 0. DOPE `c51a56d0…` frozen, and only the 29,211 adapter params train. Incumbent val parity is 1.1389572403748425 on every seed.
- **Compute:** peak RSS 6.3 GB per seed. Minimum MemAvailable was 21.3 / 20.4 / 19.6 GB (machine-wide load, swap full), above the 15 GB guard.

## Outcome
**CV** (pooled OOF over 2 × 1,219 TRAIN contexts; values are seeds 1 / 2 / 3, median in parentheses):

| Arm | OOF gain | Elo vs incumbent | Elo 95% CI (family bootstrap) | Non-floor Elo | W / L (ties rest of 2,438) | Strict (utility-strict) | Recovered / new gate fails (of 38) | vs control (gain, CI) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **pairnf_z40** (control) | +0.0048 / +0.0069 / +0.0056 (+0.0056) | +4.7 / +6.6 / +3.8 (+4.7) | [−1.3, +10.6] / [+0.7, +12.3] / [−1.8, +9.4] | +4.3 / +5.9 / +3.3 | 122/89, 128/82, 115/88 | 1/3 (3/3) | 3, 5, 4 / 0 | — |
| s32_z200 | +0.0052 / +0.0064 / +0.0046 (+0.0052) | +0.6 / +3.1 / −0.4 (+0.6) | [−5.1, +6.1] / [−2.3, +8.1] / [−5.8, +5.1] | −0.9 / +1.7 / −1.7 | 108/104, 113/91, 102/105 | 0/3 (0/3) | 10, 10, 9 / 4, 4, 4 | +0.0004 [−0.0069, +0.0087]; −0.0004 [−0.0076, +0.0077]; −0.0010 [−0.0080, +0.0062] |
| s32cos_z40 | +0.0081 / +0.0069 / +0.0066 (+0.0069) | +6.1 / +5.1 / +4.7 (+5.1) | [−0.6, +12.5] / [−1.8, +11.8] / [−1.6, +11.1] | +5.4 / +4.5 / +4.1 | 162/119, 159/123, 152/119 | 0/3 (3/3) | 6, 5, 5 / 0 | +0.0033 [+0.0002, +0.0082]; +0.0001 [−0.0012, +0.0013]; +0.0010 [−0.0009, +0.0033] |
| s32cos_z200 | 0 / 0 / 0 | 0 / 0 / 0 | — | 0 | 0/0 (all ties) | 0/3 (0/3) | 0 / 0 | −0.0048 [−0.0098, −0.0005]; −0.0069 [−0.0137, −0.0019]; −0.0056 [−0.0114, −0.0011] |

Head-to-head on contexts where the arm and the control differ:
- s32_z200 (248 / 239 / 237): arm better 111 / 111 / 105, control better 137 / 128 / 132 (Elo −36.6 / −24.8 / −39.8).
- s32cos_z40 (77 / 77 / 84): arm better 43 / 33 / 47, control better 34 / 44 / 37 (Elo +40.8 / −50.0 / +41.6).

**val91 regression check** (the CV-selected arm trained on all 1,219 TRAIN contexts):

| Seed | Selected arm | Gain | Gain CI | W / T / L | Elo | Changed / kept | New gate failures | not_worse |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | s32cos_z40 | +0.0003 | [−0.0027, +0.0042] | 3/84/4 | −3.8 | 54 / 7 | 0 | true |
| 2 | pairnf_z40 | +0.0011 | [−0.0005, +0.0028] | 4/86/1 | +11.5 | 54 / 5 | 0 | true |
| 3 | s32cos_z40 | +0.0020 | [−0.0004, +0.0057] | 5/84/2 | +11.5 | 54 / 7 | 0 | true |

Across seeds, the median val91 gain is +0.0011 (val utility 1.14003 vs incumbent 1.13896) and the median Elo +11.5. The seed-averaged per-context gain is +0.0011, with family-bootstrap CI [−0.0006, +0.0036]. The median-of-seeds CI is [−0.0004, +0.0042], and pooled W/T/L is 12/254/7. The median is not beyond its CI. The pre-registered LIFT rule fails: no arm beats the control on every seed, and every val91 CI spans 0.

## Red-team (`h463_redteam.py`, `h463_aggregate.py`, copied from H461; output under `redteam` / `aggregate` in `metrics.json`)
1. **Control and s32 reproduction.** All 30 `pairnf_z40` cells have the same adapter sha and identical OOF vectors as H461 (30/30). All 30 `s32_z200` cells have the H461 `s32_z40` adapter sha (30/30), so only the gate differs. The seed-2 full-train control adapter is again `be56fa29…`.
2. **Why s32_z200 fails (mechanism).** The count-matched gate keeps 95–110 changes per partition (control 99–108), but only 58–63 overlap with the control's. It recovers 9–10 floors (+0.007 to +0.011 gain from floors) and still loses on non-floor contexts (non-floor total −0.0007 to −0.0061), adding 1 + 3 new gate failures per seed. The 32-step adapter's high-confidence decisions are not more accurate. Recalibrating the gate restores the change count, not the win rate.
3. **What s32cos_z40 adds.** In 5 of 6 seed × partition cells its kept changes are a strict superset of the control's, with identical outcomes on the shared ones (seed 3 h452a: 103 of 104). It adds 27–43 extra changes per partition, and their summed gain is +4.57 / +3.25 (seed 1), +0.16 / +0.19 (seed 2) and +1.93 / +0.67 (seed 3) over 1,219 contexts. The extra changes are positive on average but seed-fragile, which is why the paired CI clears 0 only on seed 1.
4. **Degenerate arm.** `s32cos_z200` keeps 0 changes on every cell (cosine annealing does not inflate z-margins the way the constant-lr s32 does). The aggregator flags it as `degenerate_all_zero_arms`, and its all-tie result is a loss vs the control, never a win.
5. **Seed noise.** Seed disagreement on OOF decisions per partition: control 25–32, s32_z200 25–33, s32cos_z40 22–36. Fold-gain vectors are distinct for every non-degenerate arm × seed pair, and adapter shas are distinct per cell.
6. **Selection / winner's curse.** Selection used TRAIN-fold CV only (non-floor Elo), never val. The selected arm was the family-bootstrap argmax in 70% / 77% / 65% of resamples. The max-arm non-floor Elo CI is [0.0, +12.1] / [+0.6, +12.0] / [0.0, +10.4].
7. **val91 fragility.** Seeds 1 and 3 (both s32cos_z40) change the same 7 contexts (idx 25, 41, 49, 50, 64, 69, 84) and differ on 2 of them, which moves Elo from −3.8 to +11.5. The control changes 5 of the same contexts.
8. **Ties.** About 88–92% of CV decisions and 84–86 of 91 val91 decisions are ties by design. Ties count 0.5 in Elo and never satisfy strict. The 0.002 strict margin is unchanged.
9. **Metric direction.** Utility is higher-is-better (negated fitness, −2.0 on gate fail). Gains are paired candidate − incumbent on the same contexts. The aggregator recomputes verdicts from the per-seed fields.
10. **Leakage.** Folds are family-disjoint and exclude val families (asserted per fold, family overlap 0). The red-team rebuilds both partitions and matches the logged fold maps. Gates use predictions only. No sealed or holdout reads.
11. **Safety.** DOPE `c51a56d0` is intact on every DONE line and was verified after every training. There were no guard aborts and no OOM. xbabe2 disk is at 93% (556 GB free). The wave dir is 466 MB including the input copy. No large objects are committed, and nothing ran on xbabe0.

## Meaning
The 32-step dose has nothing extra to give the gated pairnf step. Recalibrating the gate to the dose (s32_z200) keeps the right number of changes but picks worse ones. Annealing the dose (s32cos_z40) behaves like the control plus about 27–43 extra, mildly positive, seed-fragile changes per partition. H451–H463 agree that the gated step is worth about +4 to +6 OOF Elo and is not separable from 0 on val91. The dose/schedule/gate family (H457, H461, H463) is exhausted at this instrument's resolution.

## Next step
H466 (`~/jope-data/hwave-h466` on xbabe2; H462, H464 and H465 belong to other owners) leaves the dose and gate alone and changes the shape of the training objective on the 16-step pairnf step, on the control's random streams:
- `ctxn_z40`: per-context loss normalization (every context's loss row carries equal weight).
- `clip05_z40`: tail clip 0.5 on the pointwise smooth-L1 target (gate-fail 2.0 targets and long tails cannot dominate).
- `pw2_z40`: pairwise Bradley-Terry weight 2.0 (more weight on the non-floor term that carries the lead).

No adoption nudge was sent to the flywheel owner.

Artifacts:
- `metrics.json`: `aggregate` and `redteam` as produced on xbabe2.
- `h463_trainfold_cv.py` (as run), `h463_aggregate.py`, `h463_redteam.py`.

Raw logs, losses and fold signatures stay on xbabe2 in `~/jope-data/hwave-h463`.
