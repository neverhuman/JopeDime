# H424 — the in-context model as one more stack column: `ICL_CONFIRMED` on DEV (two reads: hour 20 and hour 24)

**Verdict: `ICL_CONFIRMED`.** On the 115 fresh `DEV` families, offering the stored stack one more candidate — the
prediction of our own 24M in-context model (H419, hour-20 checkpoint) — beats the stored H420 stack by the
pre-registered test: clipped family-mean gain **+0.0092 u** (0.9% lower RMSE), one-sided sign-flip p 0.00006,
59 wins / 34 ties / 22 losses, ΔElo +115.9 [+64.2, +173.1].
**This is the first test in the campaign where the in-context model improves the stack on held-out rows.**
**Open:** a screen found copies of public datasets between `DEV` and the model's training cohort; the gain holds
without the flagged contexts, but the screen is crude (below; H425 repeated it at feature level).
**Second read (hour-24 checkpoint, no new verdict):** the same gain against the stack (+0.0099 u, p 0.00003) and
**no gain over the hour-20 column** (+0.0008 u, p 0.23). Four more hours of training did not make the column better.
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

## Second read (hour-24 checkpoint; addendum 1; reported, no new verdict)
Same `DEV` bank, same code and seeds, checkpoint `ckpt_h024.pt` (sha256 `7fbca0e702be…`). `I24` = stack + the
hour-24 column. The same 115 families as the first read, so it is not an independent confirmation.

| Comparison | Mean gain (u) | Clipped mean | Sign-flip p (one-sided) | Leave-one-family-out min | Median | Win / tie / loss | ΔElo [95% interval] (three-arm fit) | Test |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `I24` vs `B` | +0.0099 | +0.0099 | 0.00003 | +0.0071 | 0.0000 | 50 / 40 / 25 | +87.8 [+33.8, +146.3] | pass |
| **`I24` vs `I` (hour 24 against hour 20)** | +0.0008 | +0.0008 | 0.225 | +0.0003 | 0.0000 | 41 / 31 / 43 | −16.5 [−64.6, +30.7] | **no difference** |
| `I` vs `B` (first read, restated in the three-arm fit) | +0.0092 | +0.0092 | 0.00006 | +0.0064 | +0.0001 | 59 / 34 / 22 | +104.3 [+54.3, +156.9] | pass |
| `I24` vs `B` without the 17 flagged contexts (98) | +0.0104 | +0.0104 | 0.0004 | +0.0070 | 0.0000 | 39 / 37 / 22 | — | pass |

Mean u: `B` 1.9483, `I` 1.9575, `I24` 1.9583. Pre-declared secondary: no context at the clip; win share of decided
families 0.67 (`I24` vs `B`), 0.49 (`I24` vs `I`).

- **The gain against the stack repeats with a second checkpoint** (same families, so this shows the column is not a
  one-checkpoint accident, not that it holds on new families).
- **The hour-24 column is not better than the hour-20 one.** Mean gain +0.0008 u, 41 wins and 43 losses, and the
  win-based Elo is 16 points lower with an interval that spans 0. Alone, the hour-24 column is +0.002 u better.
- **Fewer families gain** (50 against 59) and more tie (40 against 34): the hour-24 column gets weight in 82
  families (84 before), mean weight 0.118.
- **Consequence for development:** between hours 20 and 24 the stack value of the column is flat. A later checkpoint
  is not expected to add much by itself; the wide-table readout (H428, H429) and the training mix are the open
  routes. The first read's ΔElo (+115.9) comes from a two-arm fit and is not comparable with the three-arm values.
- Gates (`gates2_h424.sh`): 8 of 8 (the first read's seven on the second read's outputs, plus the addendum older
  than the first column file). Inference: 115 of 115 columns, 44,371 CPU-seconds, 79 minutes.

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
| 11:21–12:40 | hour-24 checkpoint copied and hashed; second-read inference (16 CPU processes); stack step |
| 12:41 | second-read gates (8 of 8); second readout |

Evidence (xbabe2): `~/jope_plan/evidence/H424/` (`results/H424_DEV.{apply,gates,readout}.json`,
`results/H424_DEV_h024.{apply,gates}.json`, `results/H424_DEV_h024.readout2.{json,txt}`, `audit/`, `logs/`);
columns and bank under `~/jope-data/h424/{pub,pub2}/` and `~/jope-data/h335-cl/h424/pub/`.
