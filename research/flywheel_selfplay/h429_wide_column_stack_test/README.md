# H429 — the stack column for wide tables, rebuilt with H428's readout: `WIDE_NULL` (closeout; a DEV development read)

**Verdict: `WIDE_NULL`, a negative result, as stated in advance.** On the 35 `DEV` tables with more than 30 features
(30 clusters), rebuilding H424's in-context column with the readout H428 confirmed does not change the stack:
clipped gain **−0.0003 u**, one-sided p 0.79, 9 wins / 11 ties / 10 losses. It is not a harm either (mirror p 0.21).
**The readout itself did transfer:** alone, the rebuilt column is +0.029 u better on AUDIT rows (22 of 35 tables),
close to H428's +0.023 u on `FIT-SEL`. **It does not matter to the stack** because the column is still 1.43 u behind
the best stored candidate out of fold, so it keeps the same 3% weight.
A development read on a reused cohort: no champion consequence. CPU only; no GPU; `CONFIRM` and dev91 untouched; no
training run touched.

Plan: [`PREREG.md`](PREREG.md). Numbers: [`metrics.json`](metrics.json). Gates: [`gates_h429.sh`](gates_h429.sh).

## Result (`DEV`, value = AUDIT u; `B` = stored stack, `I` = H424's arm, `IT` = stack + the rebuilt column)
| Comparison | Units | Clipped gain (u) | One-sided p | Leave-one-unit-out min | Median | Win / tie / loss | Test |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **Primary: `IT` vs `I`, 35 wide tables** | 30 clusters | **−0.0003** | **0.79** | −0.0004 | 0.0000 | 9 / 11 / 10 | **fail** |
| Same, single contexts as units | 35 | −0.0002 | 0.71 | −0.0002 | 0.0000 | 10 / 14 / 11 | — |
| `IT` vs `B`, wide tables | 30 clusters | −0.0004 | 0.70 | −0.0007 | 0.0000 | 10 / 14 / 6 | — |
| `I` vs `B`, wide tables (H424's column) | 30 clusters | −0.0002 | 0.58 | −0.0005 | 0.0000 | 11 / 9 / 10 | — |
| `IT` vs `B`, all 115 tables | 110 clusters | +0.0096 | 0.00006 | +0.0066 | +0.0002 | 58 / 35 / 17 | — |
| `I` vs `B`, all 115 tables (H424, restated) | 110 clusters | +0.0096 | 0.00004 | +0.0067 | +0.0002 | 59 / 30 / 21 | — |

ΔElo `IT` vs `I` on the wide tables: −30.9 [−126.9, +60.4]. The 80 narrower tables are unchanged (difference 0).

| The column on the 35 wide tables | H424's readout | Rebuilt (H428's readout) |
| --- | --- | --- |
| Mean stack weight | 0.030 | 0.028 |
| Tables where it gets weight | 22 | 20 |
| Alone, mean AUDIT u | 1.377 | 1.405 (better in 22 of 35) |
| Out-of-fold u minus the best stored candidate | −1.477 | −1.434 (better in 30 of 35) |

| Feature columns | Tables | `IT` − `I` mean (u), wins / losses | Rebuilt column alone − H424's alone (u) |
| --- | --- | --- | --- |
| 31–100 | 26 | +0.0001, 9 / 4 | +0.040 |
| > 100 | 9 | −0.0009, 1 / 7 | −0.003 |

## What the result does and does not say
- **A better readout is not a better stack here.** The readout gain is real (+0.04 u alone at 31–100 features),
  but it closes 0.04 u of a 1.48 u out-of-fold gap to the best stored candidate. The weight builder does not move.
- **Above 100 features the rebuilt column is no better alone and the stack loses in 7 of 9 tables** by small
  amounts (mean −0.0009 u). Nine tables: a lead, not a finding.
- **The in-context column earns nothing on wide tables with either readout.** H424's own column there: −0.0002 u
  against the stack (11 wins, 10 losses). The whole H424 gain comes from the 80 narrower tables.
- **The stated expectation was met** (PREREG section 7: small or null).
- **Not shown:** anything about `CONFIRM` or new families; anything about a model trained on more wide tables.

## Consequence
- Per the PREREG: H424's readout stays the development default for wide tables. **The rebuilt readout is not
  adopted for the stack column** (it costs about 6 times the CPU for no stack gain).
- Inference-side recipes for wide tables are closed: subset count and size are not retried on `DEV`.
- Wide tables wait for the training-mix prompt (prompt B of H427), held until a GPU frees up.

## Integrity gates (`gates_h429.sh`, after the stack step, before the readout; 9 of 9 pass)
PREREG (`0b776237e2`, 12:08Z) older than the first rebuilt column and no readout existed; the first rebuilt column
younger than the H424 hour-24 readout (12:41Z); pinned checkpoint (`e3c8fd9972ce…`) and bank (`dccf9beb3312…`); CPU
only, 16 single-thread processes, no CUDA device visible; 2 h 53 min of the 4-hour cap; the stack step's alignment
and replay on all 115 tables (weights equal, baseline within 2.9e-10 u); 35 wide tables rebuilt, 0 fallback, 80
narrow column files byte-identical to H424's; the sandbox view still holds no `CONFIRM` table; published stage.

## Deviations and negatives
1. **The primary is a negative result** (`WIDE_NULL`).
2. **Cost estimate against actual:** 26 core-hours estimated, **36.7 used** (132,217 CPU-seconds); 2.3 hours
   estimated, 2 h 53 min used. Median 4,089 seconds per table against 2,700 estimated. Inside the 4-hour cap; no
   table fell back.
3. **The 16 cpus were not free of other jobs.** At launch they showed about 4 cores of other activity, and the host
   load average was about 100 when the run ended. The order said "CPUs not used by other jobs"; I launched on the
   set fixed in the PREREG without looking for an idle one. My processes still got 97% of their CPU time, so the
   overrun is in the time per pass. Whether that is the estimate (it left out the full-width ridge in every pass)
   or slower passes on a loaded host was not separated.
4. `DEV` is a reused development cohort; H424's values on these tables had been studied in H427.
5. One checkpoint (hour 20); the hour-24 checkpoint was not combined with this change.

## Queued prompts (none started)
- **A. Recipe simplification, stored values only (minutes of CPU):** "Fix in a PREREG: the in-context column is
  built only for tables with at most 30 features; wider tables use the stored stack unchanged. Restate H424 on
  `DEV` under that rule from stored values (wide tables contribute gain 0). Report the inference cost saved: the
  35 wide tables took 53% of H424's CPU inference time." Success: the restated primary passes at one-sided 0.025.
  Use: a cheaper column recipe for the champion, fixed before `CONFIRM` is read. A recommendation, not a finding.
- **B. Training mix for wide tables** (prompt B of H427, unchanged; held until a GPU frees up): two 24M arms of
  8 hours with more wide tasks; success on the 33 wide `FIT-SEL` monitor contexts; then one repeat of this `DEV`
  column test.
- **C. Size of what is needed, stored values only:** "On the 35 wide `DEV` tables, from the stored out-of-fold
  matrices: how close to the best stored candidate must a new column be before the weight builder gives it at
  least 10% weight? Add noise-blended synthetic columns at fixed out-of-fold distances (0.1 to 1.5 u) and read the
  weight." Use: a target for prompt B that says in advance what readout level would matter. No model pass, no
  AUDIT target beyond the stored arms.

## Timeline (UTC, 2026-11-11)
| Time | Event |
| --- | --- |
| 12:02 | H429 claimed |
| 12:08 | PREREG and code published (`0b776237e2`); launch held for the H424 hour-24 readout |
| 12:41 | H424 hour-24 gates and readout done |
| 12:42 | inference started (16 CPU processes, xbabe2 cpus 40–55) |
| 15:35 | inference ended (35 of 35 wide tables); stack step; gates 9 of 9 |
| 15:36 | readout: `WIDE_NULL` |

Evidence (xbabe2): `~/jope_plan/evidence/H429/results/H429_DEV.readout.{json,txt}`, `H429.gates.json`, `logs/`;
columns and stack values in `~/jope-data/h429/pub/`.
