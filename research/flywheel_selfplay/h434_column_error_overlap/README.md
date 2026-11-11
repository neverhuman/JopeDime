# H434 — error overlap of the in-context column on wide tables: `B_NO_GO` for a plain-readout column (closeout)

**Verdict: `B_NO_GO`.** The plain-readout column for wide tables (prompt B of H432, about 6.5 core-hours) is not
recommended. On the 35 wide `DEV` tables the column's out-of-fold error overlaps the stored blend's error at
**0.75** (cluster median, 95% interval 0.66 to 0.83). At the accuracy expected of a plain readout, 10% stack weight
would need an overlap of about 0.50, and **on no wide table has any predictor of that accuracy been that
different**: the lowest overlap on record is 0.71 on the median table, and at least 0.09 too high on every table.
**A useful by-product:** a two-number rule (overlap with the blend, RMSE ratio) predicts whether the stack gives a
column 10% weight on 111 of 115 tables. It can screen any future column before inference is spent on it.
**Limit:** the plain readout's own overlap was not measured (that would be prompt B itself); the verdict rests on
the most different predictor on record as a stand-in.
Stored values only: no inference, no GPU, no weight refitted; `CONFIRM` and dev91 untouched.

Plan: [`PREREG.md`](PREREG.md). Numbers: [`metrics.json`](metrics.json). Gates: [`gates_h434.sh`](gates_h434.sh).

## Measured overlap (out-of-fold errors on SEARCH rows; clusters as units; median with 95% bootstrap interval)
| The column's error against | 35 wide tables (30 clusters) | 80 narrow tables (80 clusters) |
| --- | --- | --- |
| **the stored blend** | **0.754 [0.656, 0.828]** | 0.885 [0.839, 0.920] |
| the best single candidate | 0.681 [0.598, 0.790] | 0.841 [0.813, 0.881] |
| the median of the 64 candidates | 0.684 [0.607, 0.782] | 0.845 [0.801, 0.882] |
| the nearest candidate | 0.876 [0.770, 0.916] | 0.899 [0.873, 0.942] |
| Distance behind the blend (u) | 0.301 [0.144, 0.479] | 0.064 [0.045, 0.116] |
| Overlap that would still allow 10% weight at that distance | 0.665 [0.502, 0.830] | 0.921 [0.859, 0.945] |

Overlap = uncentred correlation of two error vectors (the Pearson correlation with the blend is the same, 0.754).
Across wide clusters it runs from 0.01 to 0.99 (quartiles 0.48 and 0.89).

## The go / no-go (rule fixed in the PREREG)
| Check | Result | Rule |
| --- | --- | --- |
| The bound describes the stack: "real weight ≥ 0.10" classified correctly | **111 of 115 tables (96.5%)**; no false alarm, 4 misses | ≥ 75% |
| **Blocked share:** within reach for some error pattern, yet overlap above the limit at its distance | **0.69 [0.53, 0.84]** of wide clusters (24 of 35 tables) | > 1/2: **met** |
| **Reachable share:** the most different predictor on record would earn 10% at a plain column's expected accuracy (0.17 u weaker) | **0.00 [0.00, 0.00]** (0 of 35 tables) | ≥ 1/3: **not met** |
| Same, if the plain column were as accurate as today's | 0.20 [0.07, 0.37] (6 of 35 tables) | reported |
| Same, if it were 0.34 u weaker | 0.00 | reported |

- Floor (lowest overlap on record at that accuracy): 0.712 [0.563, 0.814]. Needed: 0.503 [0.355, 0.651]. The
  floor is above the need by 0.22 on the median wide table and by 0.09 at the least.
- On 14 of the 35 wide tables today's column is itself the most different predictor of that accuracy on record.

**Where the 35 wide tables stand today:**
| State | Tables |
| --- | --- |
| Too far behind the blend for any error pattern (more than 1.10 u) | 8 |
| Within reach but blocked by overlap | 24 (overlap 0.07 above its limit on the median blocked table) |
| Within reach and under the limit | 3 (all have real weight ≥ 0.10; a fourth table has it against the bound) |

## What the result does and does not say
- **Overlap is what holds the column back on two thirds of the wide tables**, as H432 suggested: 24 of 35 are close
  enough for some error pattern and too similar for theirs.
- **The miss is small on those tables:** the median blocked table is 0.07 of overlap above its limit. Equivalently,
  at today's overlap the column is 0.06 u of accuracy short (quartiles 0.02 and 0.12 u; the worst 0.42 u). This
  refines H432's rougher "0.1 to 0.2 u". H429's rebuilt readout gained 0.012 u on the median wide table.
- **A plain readout is a poor way to close it.** It is expected to be 0.17 u less accurate, which moves the limit
  from 0.67 down to 0.50, while the most different predictor of that accuracy on record sits at 0.71.
- **On wide tables the column is already less like the stack than on narrow ones** (0.75 against 0.89). It earns
  weight on narrow tables because it is close (0.06 u), not because it is different.
- **The bound is a usable screen.** Besides the 96.5%, its predicted need-distances agree with H432's fitted weight
  curves: median difference 0.016 u, rank correlation 0.97 (wide), 0.98 (narrow). Two numbers from out-of-fold
  predictions say whether a new column will get weight, without running the stack.
- **Not shown:** the overlap of the plain readout itself. A neural predictor without the ridge part could differ
  more than any stored candidate; the record cannot rule that out. The no-go says the available evidence does not
  justify 6.5 core-hours, not that the idea is refuted.
- **Not shown:** anything about held-out gain; weight is not gain.

## Gain concentration with the largest table left out (H424's stored values; reported)
| Features | Tables | Share of the summed gain, all tables | Share without the largest table | Wins / losses without it |
| --- | --- | --- | --- | --- |
| up to 10 | 39 | 15% | 22% | 22 / 5 |
| 11–20 | 33 (32) | **82%** | **74%** | 20 / 5 |
| 21–30 | 8 | 3% | 5% | 5 / 1 |
| 31–100 | 26 | 0% | 0% | 7 / 8 |
| over 100 | 9 | −1% | −1% | 4 / 3 |

| Test (clusters as units) | Units | Clipped gain (u) | One-sided p | Leave-one-unit-out min | Win / tie / loss | Passes |
| --- | --- | --- | --- | --- | --- | --- |
| H424 column everywhere vs stack, all | 110 | +0.0096 | 0.00004 | +0.0067 | 59 / 30 / 21 | yes |
| Same, largest table's cluster left out | 109 | +0.0067 | 0.00008 | +0.0052 | 58 / 30 / 21 | yes |
| Narrow-only (H432) vs stack, all | 110 | +0.0097 | 0.00003 | +0.0068 | 48 / 51 / 11 | yes |
| Same, largest table's cluster left out | 109 | +0.0068 | 0.00005 | +0.0053 | 47 / 51 / 11 | yes |

- **The concentration holds:** without the largest table (18 features, +0.327 u, 31% of the summed gain) the
  11–20-feature tables still carry 74%, and both tests still pass with the gain down by 30%.
- **Five tables carry 75% of the summed gain** (+0.327, +0.165, +0.113, +0.111, +0.074 u; all have 11 to 20
  features). The win count is broad (59 to 21) but the size of the mean gain is not.
- One of the five is far behind the blend (1.02 u) with an overlap of 0.10: weight 0.14 and +0.113 u. A column
  with nearly independent errors can pay off from that far.

## Integrity gates (`gates_h434.sh`, run before the read because the wave is one read; 4 of 4 pass)
PREREG (`3d4dc385ba`) published, equal to the staged copy, and no result file existed; the four pinned inputs and
115 column files; the sandbox view still holds no `CONFIRM` table; published stage.

## Deviations and negatives
1. **`B_NO_GO` is the result:** prompt B of H432 is withdrawn from the queue.
2. My advance guesses held this time (overlap about 0.8: found 0.75; blocked about 60%: 69%; floor about 0.7:
   0.71; `B_NO_GO`). The reachable share of exactly zero is stronger than I expected.
3. The printed text file shows means of the overlap limit that are meaningless (a few failing tables give huge
   negative limits); medians are used everywhere here.
4. Cost: one thread at the lowest priority, 56 seconds. xbabe2 had no idle cpu set; nothing was pinned.
5. The thresholds (1/2, 1/3, 75%) and the 0.17 u handicap are my choices, fixed before the read; see the PREREG's
   weaknesses. `DEV` is a reused development cohort.

## Queued prompts (none started)
- **A. Does the bound also predict the gain? Stored values only, minutes of CPU.** "For the 115 `DEV` tables,
  compute from the stored out-of-fold values the gain the two-predictor formula promises for H424's column
  (−½ ln of (1 − ρ²)r² / (1 + r² − 2ρr), zero where the weight is zero) and compare it with the realised AUDIT gain
  (cluster-level rank correlation; ratio of realised to promised summed gain; the five largest tables apart).
  Success: rank correlation above 0.5 with a bootstrap interval above 0.3 → the formula becomes the pre-inference
  estimate for any future column. Failure: otherwise; recorded."
- **B. Training mix for wide tables** (prompt B of H427; held until a GPU frees up), now with a measurable target:
  on the blocked wide tables the column needs an overlap about 0.07 lower at its accuracy, or about 0.06 u more
  accuracy at its overlap (0.12 u covers three quarters of them). Both can be read from out-of-fold predictions on
  `FIT-SEL` before any `DEV` read.
- **C. Screen before inference (when the next column is proposed):** for any new column (104M model, another
  readout), compute overlap and RMSE ratio on a 20-table `FIT-SEL` sample first and apply the bound; build the
  full column only if the predicted weight is at least 0.10 on a third of the sample.
- Withdrawn: the plain-readout column on wide tables (prompt B of H432).

## Timeline (UTC, 2026-11-11)
| Time | Event |
| --- | --- |
| 16:21 | H434 claimed |
| 16:24 | PREREG and code published (`3d4dc385ba`) |
| 16:24 | gates 4 of 4, then the read (56 s): `B_NO_GO` |

Evidence (xbabe2): `~/jope_plan/evidence/H434/results/H434_DEV.overlap.{json,txt}` (per-table values inside),
`H434.gates.json`, `logs/`.
