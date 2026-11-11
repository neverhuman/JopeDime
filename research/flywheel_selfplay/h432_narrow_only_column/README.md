# H432 — the in-context column on narrow tables only: `NARROW_KEEPS`; and what a wide-table column would need (closeout)

**Verdict: `NARROW_KEEPS`** (a restatement of stored `DEV` values; development evidence only). Offering H424's
in-context column only on the 80 tables with at most 30 features keeps the whole H424 gain: **+0.0097 u** clipped
against the stored stack (one-sided p 0.00003, 48 wins / 51 ties / 11 losses over 110 clusters), against +0.0096 u
with the column on every table. Retention **1.005** [0.966, 1.055]. Inference CPU falls by **53%**.
**Keep in proportion:** this outcome was implied by H429's published numbers, and the 30-feature threshold was
chosen on these same tables. The solid part is the cost saving.
**Part 2 (descriptive):** whether an added column earns weight depends on how different its errors are, more than
on how accurate it is. With today's error pattern a wide-table column needs to be within about 0.2 u of the best
stored candidate; with independent errors it would earn 10% weight even 1 u behind.
**Correction to H429:** "1.4 u behind the best stored candidate" is a mean driven by 9 tables where the column
fails badly. The median wide table is 0.27 u behind.
No inference, no GPU; `CONFIRM` and dev91 untouched; no training run touched.

Plan: [`PREREG.md`](PREREG.md). Numbers: [`metrics.json`](metrics.json). Gates: [`gates_h432.sh`](gates_h432.sh).

## Part 1 — result (`DEV`, 115 tables in 110 clusters; `B` = stored stack, `I` = H424's arm, `N` = column only up to 30 features)
| Comparison | Clipped gain (u) | One-sided p | Leave-one-unit-out min | Median | Win / tie / loss | Campaign rule (p < 0.025) |
| --- | --- | --- | --- | --- | --- | --- |
| **`N` vs `B` (narrow-only against no column)** | **+0.0097** | **0.00003** | +0.0068 | 0.0000 | 48 / 51 / 11 | pass |
| `I` vs `B` (H424, column on every table) | +0.0096 | 0.00004 | +0.0067 | +0.0002 | 59 / 30 / 21 | pass |
| `N` vs `I` | +0.0000 | 0.42 | −0.0000 | 0.0000 | 10 / 89 / 11 | — |
| Hour-24 column: `N24` vs `B` | +0.0104 | 0.00002 | +0.0074 | 0.0000 | 42 / 55 / 13 | pass |
| Hour-24 column: `I24` vs `B` | +0.0104 | 0.00002 | +0.0075 | 0.0000 | 49 / 37 / 24 | pass |

- **Retention** (gain of `N` ÷ gain of `I`): 1.005, cluster-bootstrap 95% interval [0.966, 1.055]; 0.06% of
  resamples fall below 0.90. Hour-24 column: 0.995.
- **Inference CPU** (stored per-table times of H424): 43,779 s with the column everywhere, 20,387 s narrow-only:
  **53.4% saved** (hour-24 read: 52.6%).
- Rule (PREREG): retention ≥ 0.90, p < 0.05, less CPU → `NARROW_KEEPS`.

**Threshold table (reported only; the verdict uses 30):**
| Column kept up to | Tables with the column | Clipped gain vs `B` (u) | Retention | CPU saved |
| --- | --- | --- | --- | --- |
| 10 features | 39 | +0.0015 | 0.15 | 80% |
| 20 features | 72 | +0.0094 | 0.97 | 59% |
| **30 features** | 80 | +0.0097 | 1.00 | 53% |
| 50 features | 90 | +0.0096 | 1.00 | 44% |
| 100 features | 106 | +0.0097 | 1.01 | 24% |
| all | 115 | +0.0096 | 1.00 | 0% |

**Where the H424 gain sits, by table (stored values; added after the readout, descriptive):**
| Features | Tables | Summed gain (u) | Mean gain (u) | Wins / losses | Largest gain (u) |
| --- | --- | --- | --- | --- | --- |
| up to 10 | 39 | 0.160 | +0.004 | 22 / 5 | 0.033 |
| 11–20 | 33 | 0.869 | +0.026 | 21 / 5 | 0.327 |
| 21–30 | 8 | 0.035 | +0.004 | 5 / 1 | 0.013 |
| 31–100 | 26 | −0.000 | 0.000 | 7 / 8 | 0.011 |
| over 100 | 9 | −0.007 | −0.001 | 4 / 3 | 0.001 |

## What part 1 does and does not say
- **Dropping the column above 30 features costs nothing measurable on `DEV` and halves its inference cost.**
- **30 sits on a plateau, not an edge:** any threshold from 20 features up keeps 97% to 101%.
- **The gain is concentrated:** 82% of it comes from the 33 tables with 11 to 20 features, and one table gives
  31% of it. Tables with at most 10 features win often (22 to 5) by small amounts.
- **Not shown:** that 30 is right on new families. The threshold was picked on these tables (H427), so "keeps the
  gain" is partly by construction. Nothing here is a test on fresh data.
- **Not a claim that wide tables cannot gain** from an in-context column; only that today's column does not.

## Part 2 — how good must an added column be to earn 10% stack weight? (descriptive, no verdict)
A synthetic 65th column at a fixed out-of-fold distance d behind the best stored candidate, on the stored matrices,
weights from the stored builder. **S** = H424's own column with its error rescaled (same error pattern); **N** =
independent Gaussian error (optimistic bound). Replay: the unscaled column reproduces H424's stored weight exactly
on all 115 tables.

| Distance behind the best stored candidate (u) | −0.2 | −0.1 | 0 | 0.05 | 0.1 | 0.2 | 0.3 | 0.5 | 0.75 | 1.0 | 1.5 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **S, 35 wide tables:** mean weight | 0.77 | 0.66 | 0.42 | 0.30 | 0.23 | 0.15 | 0.10 | 0.06 | 0.04 | 0.02 | 0.01 |
| S, wide: share of tables with weight ≥ 0.10 | 100% | 100% | 100% | 83% | 66% | 46% | 34% | 26% | 17% | 3% | 3% |
| S, 80 narrow tables: mean weight | 0.88 | 0.76 | 0.35 | 0.18 | 0.13 | 0.08 | 0.06 | 0.03 | 0.02 | 0.01 | 0.01 |
| S, narrow: share with weight ≥ 0.10 | 100% | 100% | 91% | 51% | 39% | 23% | 17% | 10% | 7% | 6% | 1% |
| **N, 35 wide tables:** mean weight | 0.58 | 0.53 | 0.48 | 0.45 | 0.43 | 0.38 | 0.33 | 0.25 | 0.17 | 0.11 | 0.04 |
| N, wide: share with weight ≥ 0.10 | 100% | 100% | 100% | 100% | 97% | 97% | 97% | 97% | 97% | 83% | 0% |

| Distance at which the weight falls below 0.10 (quartiles over tables) | 25% | Median | 75% |
| --- | --- | --- | --- |
| S, wide tables | 0.08 u | **0.18 u** | 0.50 u |
| S, narrow tables | 0.03 u | 0.05 u | 0.16 u |
| N, wide tables | 1.03 u | 1.09 u | 1.14 u |

**Where the real column stands (stored values):**
| | Wide (35) | Narrow (80) |
| --- | --- | --- |
| Distance behind the best stored candidate: median (mean) | 0.27 u (1.48 u) | 0.03 u (0.44 u) |
| Tables more than 1 u behind | 9 (two more than 10 u) | 6 |
| Mean stack weight | 0.030 | 0.142 |
| Tables with weight ≥ 0.10 | 4 | 36 |

### Reading
- **Error pattern matters more than accuracy.** A column with independent errors earns 10% weight on almost every
  wide table even 1 u behind the best candidate. A column with today's error pattern needs to be within 0.18 u on
  the median wide table.
- **Where the real column is already close, it is nearly a copy of what the stack has.** On the 6 wide tables
  where it is within 0.05 u of the best candidate, the S curve gives it almost no weight at 0.05 u behind (0 to
  0.06) and almost all of it at 0.1 u ahead (0.82 to 1.00): its errors are nearly the best candidates' errors.
- **On the typical wide table the column is not far away** (median 0.27 u), and under S it would need to be roughly
  0.05 to 0.2 u better than it is to reach 10% weight (post hoc, read off per-table curves on a coarse grid).
  H429's rebuilt readout improved the median wide-table column by 0.012 u out of fold (mean 0.043 u), well short
  of that; its weights did not move (12 up, 11 down, 4 tables at 10% before and after).
- **Narrow tables are stricter than wide ones** (median 0.05 u against 0.18 u): the column earns its weight there
  because it is that close (median 0.03 u behind), not because the bar is lower.
- **Limits.** Two error models; the truth for a better model lies between. S becomes the optimistic model on
  tables where the real column fails badly (a rescaled bad column is close to independent noise). Weight is not
  held-out gain. The need-distances come from linear interpolation on a coarse grid and are a few hundredths of a
  u too generous: at the real distances the interpolated curve gives 0.067 mean weight against the real 0.030.

## Consequence
- Per the PREREG: **"column only up to 30 features" is the development default of the column recipe** (a cost
  rule). No champion consequence; `CONFIRM` stays closed until the owner's decision.
- For wide tables the target is now stated: a column whose errors differ from the stored candidates', or one about
  0.1 to 0.2 u better on the typical table. A readout change of 0.01 to 0.04 u (H428, H429) is too small.

## Integrity gates (`gates_h432.sh`, after part 2's fits, before the two readouts; 7 of 7 pass)
PREREG (`2dc3322fb6`) older than the first output and no readout existed; the four pinned inputs by sha256; 115
contexts (35 wide) with an exact replay of H424's stored column weight (difference 0); 8 single-thread processes,
no model library loaded; 395 seconds of the one-hour cap; the sandbox view still holds no `CONFIRM` table;
published stage.

## Deviations, negatives and corrections
1. **The verdict was known in advance** (PREREG section 7) and the threshold was picked on the same cohort.
2. **No idle cpu set existed on xbabe2** (the order asked for one). A 10-second sample at launch showed the 8 least
   busy cpus at 42% to 50% busy with other sessions' jobs. The work ran there as 8 single-thread processes at the
   lowest priority: 2,757 CPU-seconds (0.77 core-hours), 6.6 minutes.
3. **My two guesses for part 2 were wrong in opposite directions:** 0.3 u under S (found 0.18 u) and 0.6 u under N
   (found 1.09 u).
4. **Correction to H429's wording** ("1.43 u behind", "a 1.48 u gap"): these are means. The median wide table is
   0.27 u behind; 9 tables are more than 1 u behind. H429's verdict and test numbers are unchanged; its README
   carries a correction note.
5. **A bug in part 2's summary table:** the "weight against real distance" bins stop at 9 u and leave out two wide
   tables that are 11.8 and 13.7 u behind. The counts in this README come from the per-table files.
6. **Added after the readouts, not in the PREREG, descriptive:** the gain-by-width table, the per-table
   "improvement needed" reading and the comparison with H429's real improvement
   (`H432_DEV.need_gap.posthoc.json`, `H432_DEV.s_model_check.posthoc.json`).
7. `DEV` is a reused development cohort.

## Queued prompts (none started)
- **A. Error overlap, stored values only (minutes of CPU):** "On the 115 `DEV` tables, from the stored out-of-fold
  matrices and H424's stored column: the correlation between the column's out-of-fold error and (i) the best
  stored candidate's error, (ii) the stored blend's error. Does the column's stack weight follow one minus that
  correlation better than it follows distance (cluster-grouped rank correlations, wide and narrow apart)?" Use:
  says whether wide tables need a more accurate column (training mix) or a different one (another readout).
- **B. A second, different readout on wide tables (needs inference; own PREREG):** "For the 35 wide `DEV` tables
  build the column from the model's plain readout (no ridge), which H424 dropped for cost, and offer it beside the
  stored column. Primary: stack with both columns vs H424's arm on those tables, 30 clusters, one-sided 0.025.
  Cost: 6 full-width passes per table, about 6.5 core-hours on CPU, or minutes on a GPU." Run only if prompt A
  shows the plain readout's errors would differ; a `DEV` development read.
- **C. Training mix for wide tables** (prompt B of H427; held until a GPU frees up), with the target from part 2:
  the median wide-table column must come at least 0.1 u closer to the best stored candidate, out of fold, before
  the stack is expected to move.

## Timeline (UTC, 2026-11-11)
| Time | Event |
| --- | --- |
| 15:47 | H432 claimed; cpu check (no idle set) |
| 15:52 | PREREG and code published (`2dc3322fb6`) |
| 15:53–15:59 | part 2 weight fits (8 processes at lowest priority) |
| 16:00 | gates 7 of 7; part 1 readout: `NARROW_KEEPS`; part 2 summary |
| 16:02–16:04 | descriptive additions from stored outputs |

Evidence (xbabe2): `~/jope_plan/evidence/H432/results/` (`H432_DEV.narrow.{json,txt}`, `H432_DEV.need.{json,txt}`,
`H432.gates.json`, two `posthoc` files), `logs/`; per-table weight curves in `~/jope-data/h432/pub/need/`.
