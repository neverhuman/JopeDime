# H443: EMA, gradient clipping and strict selection against the flywheel plateau (offline, adapter-only)

**Verdict: `NO_LIFT`.** None of the three stabilisers got past the incumbent on any seed. The plateau is structural, not optimiser noise.

## Hypothesis
The fused flywheel (`run-20261030`) has been stuck at a val utility of about 1.134–1.139 over the 91 DL-Val regression contexts since about round 3175. In that window it ran 1,873 straight rejections and accepted ties. The hypothesis was that the stall comes from noisy single-step updates, failure contexts (−2.0) dominating the loss, and tie or noise accepts. If so, Polyak/EMA weights, per-context loss normalisation with a capped failure target, and strict bootstrap-CI selection should find a real improvement.

## Setup
- **Code:** `h443_offline.py` (CPU only, `nice 19`, 16 threads, MemAvailable abort below 15 GB). It ran on xbabe1 against a **scratch copy**, `~/jope-data/hwave-h443`. The live run directory was never written to.
- **Start point:** incumbent `round-3175.pt` plus its optimizer state. The parity check reproduces the live incumbent's val utility exactly (1.1389572403748425).
- **DOPE frozen:** composite sha `c51a56d0…` before and after, checked in every arm. Only the 29,211 adapter parameters were trained, and the optimizer excludes DOPE (guard asserted).
- **Data:** real S3 packet, 1,219 DL-Train contexts (205 families) and 91 DL-Val contexts (19 families). Family overlap is 0 (asserted). No sealed or holdout data: every row is asserted to be DL-Train or DL-Val.
- **Arms:** each runs 10 rounds × 2 steps (batch 2,048, slate 192, lr 1e-4) with seeds 1, 2, 3.
  - `ema`: Polyak weights with decay 0.9 are evaluated and admitted.
  - `clip`: per-context loss normalised by its detached magnitude and clipped at 3× the batch median. The failure target is capped at the row's max passing fitness + 0.25.
  - `strict`: accepts only when the mean paired gain is above 0.002 **and** the family-bootstrap 95% lower bound is above 0. Ties are never accepted.
  - All arms use grad-norm clipping at 1.0.
- **Control:** the live loop's own history serves as the `base` arm, replayed by `h443_history.py` over rounds 775–5049 (4,274 parsed). A one-round `base` probe reproduces the live loop's typical candidate (1.135439).

## Outcome
| Arm | Seeds | Accepts | Adopted gain vs incumbent | Elo vs incumbent | Best candidate − incumbent |
| --- | --- | --- | --- | --- | --- |
| ema | 3 | 0/30 | 0.0 [0.0, 0.0] | 0 | −0.00227 (1.136683) |
| clip | 3 | 0/30 | 0.0 [0.0, 0.0] | 0 | −0.00352 (1.135439) |
| strict | 3 | 0/30 | 0.0 [0.0, 0.0] | 0 | −0.00352 (1.135439) |

- Every candidate in every arm and seed fell on a few discrete val levels at or below the incumbent: 1.135439, 1.136683, 1.132522 and 1.1301–1.1306. The 1.135439 level is also the live loop's most common candidate, at 96 of its last 400 rounds. Those 400 rounds produced only 22 distinct candidate means.
- Seed spread in the selected val is zero because argmin-over-pool utility is discrete. Per-step losses do differ by seed.
- **Live-history red-team:** the loop made 8 accepts since round 775, and 4 of them were exact ties. **None of the 8 passes the strict rule.**
  - The r785 jump (+0.030) came from 8 changed contexts. One context, a −2.0 fail flipping to pass, carried 89% of the absolute change, and the CI was [−0.011, +0.104].
  - The last real accepts (r2750, r3175) gained +0.0006 and +0.0012, and both CIs include 0.
  - The incumbent now has 0 fail contexts, so per-context loss clipping has nothing left to fix on val.

## Meaning
The plateau does not come from optimiser instability or loss imbalance. Small adapter updates from the incumbent land in the same basin every time, and the 91-context val cannot resolve improvements below about 0.003. On the strict rule, the climb from r775 to the incumbent is not distinguishable from selection noise either.

## Next step
Stop spending on adapter-optimiser tweaks for this loop. Promising levers:
1. Change the **data or candidates**: proposer diversity and new pool programs, so the argmin has better options.
2. Use a **larger selection signal**: family-grouped inner CV over the 1,219 TRAIN contexts instead of single-shot selection on 91 val contexts. The val stays a regression check only.
3. Make the live loop require strict, no-tie admission, so it stops accepting noise.

No adoption nudge was sent to the flywheel owner, because nothing beat the incumbent.

Artifacts: `metrics.json` (per-arm, per-seed results, trajectories, live-history replay), `h443_offline.py`, `h443_history.py`. The raw logs stay on xbabe1 in `~/jope-data/hwave-h443/{run.log,results,history}`.
