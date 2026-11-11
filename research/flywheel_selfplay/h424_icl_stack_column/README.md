# H424 — the in-context model as one more stack column: `ICL_CONFIRMED` on DEV (first read; second read running)

**Verdict: `ICL_CONFIRMED`.** On the 115 fresh `DEV` families, offering the stored stack one more candidate — the
prediction of our own 24M in-context model (H419, hour-20 checkpoint) — beats the stored H420 stack by the
pre-registered test: clipped family-mean gain **+0.0092 u** (0.9% lower RMSE), one-sided sign-flip p 0.00006,
59 wins / 34 ties / 22 losses, ΔElo +115.9 [+64.2, +173.1].
**This is the first test in the campaign where the in-context model improves the stack on held-out rows.**
**Open:** a screen found copies of public datasets between `DEV` and the model's training cohort; the gain holds
without the flagged contexts, but the screen is crude (below). The second read (hour-24 checkpoint) is running and
will be added here.
TRAIN-side cohorts only: no dev91, DL-Val, DL-Holdout, sealed or `CONFIRM` data was read. No GPU was used; the
104M and 24M training runs were not touched.

Plan: [`PREREG.md`](PREREG.md), [addendum 1](PREREG_ADDENDUM_1.md). Numbers: [`metrics.json`](metrics.json).
Gates: [`gates_h424.sh`](gates_h424.sh).

## Result (DEV, 115 families, value = AUDIT u, paired by family against the stored stack `B`)
| Comparison | Mean gain (u) | Clipped mean | Sign-flip p (one-sided) | Leave-one-family-out min | Median | Win / tie / loss | ΔElo [95% interval] | Test |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **`I` vs `B` (primary): 64 stored candidates + the model's column** | +0.0092 | **+0.0092** | **0.00006** | +0.0064 | +0.0001 | 59 / 34 / 22 | +115.9 [+64.2, +173.1] | **pass** |
| Same, without the 17 contexts the duplicate screen flags (98) | +0.0087 | +0.0087 | 0.0009 | +0.0054 | 0.0000 | 46 / 31 / 21 | — | pass |

Mean u: `B` 1.9483, `I` 1.9575.

**Pre-declared secondary (contexts at the ±0.5 clip excluded):** no context is at the clip, so nothing is excluded:
59 / 34 / 22, win share of decided families 0.73, gain +0.0092 u.

## What the result does and does not say
- **The column is used and it helps often.** It gets weight in 84 of 115 families (mean weight 0.11; 0.15 where
  used). In those 84 it wins 58 times and loses 21.
- **It helps as a different kind of member, not as a better model.** Alone the model is far below the stack (mean
  −0.56 u) and ahead of it in only 23 families. Where it is ahead alone the stack gains +0.036 u; elsewhere
  +0.002 u.
- **The gain is broad, unlike H422's combiner:** no context reaches the clip, 5 gain more than 0.05 u (largest
  +0.33 u), none loses more than 0.03 u. Without the best family the gain is still +0.0064 u.
- **My expectation was wrong.** The PREREG predicted about +0.002 u and a null, from a stack test on fitted rows
  (+0.005 u at hour 12 on `FIT-SEL`). The held-out gain at hour 20 is larger than that fitted-row estimate, the
  opposite of what H421 and H422 saw.
- **Size:** 0.9% RMSE is a third of what the 20 library presets gave (H420: +0.028 u). It is one column from a
  model 20 hours into a 46-hour run; no claim is made about later checkpoints or the 104M model.
- **Not shown:** that the gain survives on `CONFIRM`; that it adds to wider admission or the combiner; that the
  plain readout or more columns would add more (not computed: CPU inference is slow).

## Duplicate screen (run after the readout; descriptive)
The model was trained on the `FIT-ICL` families. A `DEV` table that is a copy of a `FIT-ICL` table would let the
column benefit from targets it has seen.
- Manifest: 0 shared table hashes, dataset ids or family ids between `DEV` and `FIT-ICL`.
- **Target values:** of 105 `DEV` contexts with at least 20 distinct SEARCH target values, 17 have at least half of
  those values in one `FIT-ICL` training table (11 at least 90%). Some are the same public dataset under two names
  (`cpmp-2015-runtime-regression` in `DEV`, `cpmp-2015-regression` in `FIT-ICL`); others are coarse targets many
  tables share (101 values from 0.00 to 1.00).
- **The cohort split by source name does not remove every copy of a dataset.** Two `DEV` families also look like
  the same dataset (`communities-and-crime`, `us-crime`: the same target values).
- **Effect on this result:** without the 17 flagged contexts the test still passes (+0.0087 u, p 0.0009, 46 / 21).
  The three largest gains are on contexts with almost no overlap. Among the 17 flagged the gain is +0.012 u
  (13 wins, 1 loss), slightly higher.
- **Limits:** target values only. A copy with a transformed target is missed; a coarse target is flagged falsely.
  No feature-level comparison was made. `CONFIRM` should be screened against `FIT-ICL` before the champion read.

## Integrity gates (`gates_h424.sh`, after the stack step, before the readout; 7 of 7 pass)
| Gate | Evidence |
| --- | --- |
| Inputs | stored `DEV` matrices match their manifest; the checkpoint is `ckpt_h020.pt`, sha256 `e3c8fd9972ce…`, equal on the training host |
| `DEV` only, no AUDIT target in the bank | 115 `DEV` contexts; the bank holds SEARCH rows and AUDIT features; guard installed |
| Alignment and replay | SEARCH targets, fold labels and the AUDIT-target hash match on all 115; stack weights equal the stored ones; baseline value within 2.9e-10 u |
| PREREG first | commit `ab56e31a05` is older than the bank file |
| CPU only | 16 processes × 1 thread, no CUDA device visible, pinned checkpoint |
| 4-hour cap | 78 minutes from start to the end of the stack step; 12.2 core-hours of inference |
| Published stage | code hash verified; model code equal to H419's training stage |

## Deviations, negatives and weaknesses
1. **Only one readout was computed.** The plain readout as a second column was dropped before the PREREG: the
   model's CPU forward pass uses about one core whatever the thread count (37 s per pass for a narrow table,
   minutes for the widest).
2. **The checkpoint is the latest named one (hour 20), not `ckpt_last.pt`** (20 to 40 minutes newer, overwritten
   every 20 minutes).
3. `DEV` has now been read four times (H420, H421, H422, H424), each time by one pre-registered primary.
4. The duplicate screen was not pre-registered and is crude; see above.
5. The model's monitor cohort (`FIT-SEL`) guided two training changes in H419; `DEV` played no part in training.
6. One context of the eight used for the code smoke on `FIT-SEL` lost 0.009 u; the smoke numbers are in the PREREG.
7. The column exists only as offline code on stored matrices; a deploy needs the model at inference time, on a GPU
   or with about 6 CPU-minutes per table.

## Second read (hour-24 checkpoint)
Running under addendum 1: same `DEV` bank, same code, no new verdict. Reported when it ends: `I24` vs `B`,
`I24` vs `I`, and both without the flagged contexts.

## Timeline (UTC, 2026-11-11)
| Time | Event |
| --- | --- |
| 07:54 | H424 claimed |
| 07:55–08:24 | checkpoint copied; `FIT-SEL` bank and CPU timing; join smoke on 8 `FIT-SEL` contexts |
| 08:25 | PREREG and code published (`ab56e31a05`) |
| 08:26–09:44 | `DEV` bank in the sandbox; inference (16 CPU processes); stack step |
| 09:45 | gates (7 of 7); readout: `ICL_CONFIRMED` |
| 09:46 | duplicate screen |
| 09:48 | addendum 1 published (`e1f31467e4`); second read waiting for the hour-24 checkpoint |

Evidence (xbabe2): `~/jope_plan/evidence/H424/` (`results/H424_DEV.{apply,gates,readout}.json`, `audit/`, `logs/`);
columns and bank under `~/jope-data/h424/pub/` and `~/jope-data/h335-cl/h424/pub/`.
