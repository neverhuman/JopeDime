# H428 — feature bagging for wide tables: `BAG_CONFIRMED` for one variant; the queued version loses (closeout)

**Verdict: `BAG_CONFIRMED`**, through the variant added in the PREREG. On the 33 `FIT-SEL` monitor contexts with more
than 30 features (26 clusters), keeping the ridge on all features and letting the model predict its residual from
8 random 30-feature subsets beats the full-width readout of the same checkpoint: clipped gain **+0.0234 u**,
one-sided p 0.00014, 20 wins / 1 tie / 5 losses. It passes the Holm threshold (0.0125).
**The prompt as it was queued (ridge and model both on each subset) fails:** clipped −0.071 u, 14 wins / 12 losses.
**Keep in proportion:** on the wide `DEV` tables the column was 0.7 to 1.7 u behind the best GP candidate (H427). A
readout that is 0.02 to 0.04 u better does not close that gap; the stack-level effect is likely to be small.
`FIT-SEL` only; no GPU; `DEV` not rescored; `CONFIRM` and dev91 untouched.

Plan: [`PREREG.md`](PREREG.md). Numbers: [`metrics.json`](metrics.json). Gates: [`gates_h428.sh`](gates_h428.sh).

## Result (33 wide `FIT-SEL` monitor contexts, 24M checkpoint hour 20, same query rows; u as in the monitor)
| Variant vs full-width control | Units | Clipped gain (u) | One-sided p | Holm threshold | Leave-one-unit-out min | Median | Win / tie / loss | Passes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **T1: ridge on all features + bagged residual** | 26 clusters | **+0.0234** | **0.00014** | 0.0125 | +0.0180 | +0.0058 | 20 / 1 / 5 | **yes** |
| T2: whole readout on each subset (queued prompt) | 26 clusters | −0.0713 | 0.945 | 0.025 | −0.0858 | +0.0059 | 14 / 0 / 12 | no |
| T1, single contexts as units | 33 | +0.0274 | 0.0041 | — | +0.0200 | +0.0068 | 23 / 1 / 9 | — |
| T2, single contexts as units | 33 | −0.0662 | 0.942 | — | −0.0773 | +0.0007 | 17 / 0 / 16 | — |

Mean u: plain ridge 1.205, control 1.295, T1 1.322, T2 1.015.

| Feature columns | Contexts | Control over plain ridge (u) | T1 mean gain (u), wins / losses | T2 mean gain (u), wins / losses |
| --- | --- | --- | --- | --- |
| 31–100 | 25 | +0.116 | +0.0356, 18 / 7 | −0.314, 14 / 11 |
| > 100 | 8 | +0.008 | +0.0019, 5 / 2 | −0.171, 3 / 5 |

## What the result does and does not say
- **Showing a wide table as several narrow ones helps only if the linear part stays on all features.** T1 adds
  about 0.036 u on tables with 31–100 features, a third on top of the control's margin over plain ridge there.
- **Above 100 features it adds almost nothing** (+0.002 u, 8 contexts): a 30-feature subset covers too little of
  the table, and the control itself is level with ridge there.
- **The queued version (T2) is harmful on average** but not by the PREREG's harm rule (mirror p 0.055). Two contexts
  lose more than 0.5 u (the worst −4.1 u) because ridge on a subset misses most of the linear signal.
- **T1 was my addition, not the queued prompt.** It was fixed in the PREREG with a Holm rule before any output, and
  it passes at its corrected threshold. T2's failure was the stated expectation.
- **Not shown:** that the stack gains. The monitor task is one fold of SEARCH rows; the stack column needs five
  folds plus AUDIT rows, and it competes with GP candidates that are far ahead on wide tables.
- **Not shown on a fresh cohort:** `FIT-SEL` is the model's monitor.
- **Cost:** T1 needs 8 passes of 30 features instead of one full-width pass: about 6 times the CPU time at 60
  features and about twice at 300.

## Integrity gates (`gates_h428.sh`, after inference, before the readout; 7 of 7 pass)
PREREG (`74c0f32a2d`) older than the first output and no readout existed; the pinned checkpoint (`e3c8fd9972ce…`) and
the monitor bank (`8e517921b7d5…`); exactly the 33 wide `FIT-SEL` contexts; CPU only, 16 single-thread processes, no
CUDA device visible; 36 minutes of the 4-hour cap; the sandbox view still holds no `CONFIRM` table; published stage.

## Deviations and negatives
1. **The queued prompt, taken literally, is a negative result** (T2).
2. **Cost estimate against actual:** 6.4 core-hours estimated, 7.1 used (25,722 CPU-seconds); 45 minutes estimated,
   36 used. No subsample was needed.
3. A variant was added to the queued prompt in the PREREG (T1), with a multiplicity rule.
4. The code smoke before the PREREG used a synthetic bank of random numbers only.
5. Subset count (8) and size (30) were not tuned and are not to be tuned on this cohort.

## Consequence and queued next step
Per the PREREG: T1 is the readout for tables with more than 30 features in the next stack-column test. Prompt, not
started:
- "Repeat H424's column on its own PREREG with one change: for the 35 `DEV` tables with more than 30 features the
  column is built with T1 (ridge on all features, residual bagged over 8 seeded 30-feature subsets); the 80
  narrower tables keep H424's stored column. Primary: `I_T1` vs the stored H424 arm `I` on the 35 wide tables,
  clusters as units, clipped gain at one-sided 0.025. Reported: the stack weight of the column on those tables.
  Budget: CPU inference of 35 tables, 6 passes × 8 subsets each (about 26 core-hours, under 2 hours with 16
  processes), no GPU. Expectation, stated
  in advance: a small or null stack effect, because the column is far behind the GP candidates there."
- Prompts B and C of H427 stay held until a GPU frees up.

## Timeline (UTC, 2026-11-11)
| Time | Event |
| --- | --- |
| 11:08 | H428 claimed; CPU cost estimated (6.4 core-hours) |
| 11:10–11:14 | code smoke on a synthetic bank |
| 11:15 | PREREG and code published (`74c0f32a2d`); inference started (16 CPU processes) |
| 11:51 | inference ended (33 of 33 contexts); gates 7 of 7; readout: `BAG_CONFIRMED` |

Evidence (xbabe2): `~/jope_plan/evidence/H428/results/H428_FITSEL.readout.{json,txt}`, `H428.gates.json`; per-context
outputs in `~/jope-data/h428/pub/bag/`.
