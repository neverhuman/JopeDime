# H425 — contamination screen: the H424 gain survives on DEV; CONFIRM could not be screened (closeout)

**DEV: done.** Of the 115 `DEV` contexts, the feature-level screen flags **8 as copies** of a table of `FIT-ICL` (the
in-context model's training cohort) and **13 as possible** relatives; the target-only baseline adds 10 more.
**H424's gain survives:** without the 21 feature-flagged contexts the stack with the model's column still beats the
stored stack, +0.0107 u, one-sided p 0.0002 (94 families). No verdict is changed.
**CONFIRM: not screened at feature level — escalation.** The `CONFIRM` tables are not in the sandbox view (H417 kept
them out), so the screen cannot open them without removing that protection; I did not. The same screen finds copies
in 7% of `DEV` and 17% of `FIT-SEL`, so **`CONFIRM` very likely holds copies of `FIT-ICL` tables too**. This needs an
owner decision before any champion read that includes a model trained on `FIT-ICL`.
Nothing was scored, predicted or fitted; no dev91, DL-Val, DL-Holdout or sealed data was read; no GPU.

Plan: [`PREREG.md`](PREREG.md), [addendum 1](PREREG_ADDENDUM_1.md). Frozen lists: [`EXCLUSION_DEV.json`](EXCLUSION_DEV.json),
[`CONFIRM_META.json`](CONFIRM_META.json). Numbers: [`metrics.json`](metrics.json). Gates: [`gates_h425.sh`](gates_h425.sh).

## DEV screen (115 contexts against the 440 `FIT-ICL` tables)
| Tier | Contexts | What it means |
| --- | --- | --- |
| `copy` | 8 | many specific columns share at least half of their values with one `FIT-ICL` table |
| `possible` | 13 | weaker column matches, matching column ranges, or a matching source name |
| `target_only` | 10 | only the target-value baseline fires |
| `clear` | 84 | no screen fires |

| Screen | Applies to | Flags |
| --- | --- | --- |
| S1 specific columns | 93 | 17 (8 of them `copy`) |
| S2 shared rows | 104 | 0 |
| S3 column ranges | 70 | 4 |
| S4 source names | 23 | 2 |
| B target only (baseline) | 105 | 18 |

- **What the feature screen adds to the target-only one:** 13 of the 21 feature-flagged contexts are not flagged by
  the target baseline, among them 7 of the 8 copies; these are tables whose features match a `FIT-ICL` table while
  the target differs. In the other direction, 10 contexts are flagged by the target baseline alone, mostly coarse
  targets that many tables share.
- **How the copies look:** three `DEV` contexts match the same `FIT-ICL` table in 45 of 46 specific columns, two
  others another table in 73 and 74 of 75. One source dataset appears as several families (probably one per
  target column), and the family split scattered them across cohorts. `cpmp-2015-runtime-regression` is flagged
  by columns, name and target at once.
- **No shared rows anywhere (S2: 0).** The copies are different row samples or column variants, never the same
  rows.
- **Inside `FIT-ICL` itself** 62 of 440 tables are a `copy` of a table of another family.

## Sensitivity read of H424 (`I` vs `B` on DEV, hour-20 checkpoint; no new verdict)
| Subset | Families | Clipped gain (u) | One-sided p | Leave-one-family-out min | Win / tie / loss | Test |
| --- | --- | --- | --- | --- | --- | --- |
| all (H424's primary, restated) | 115 | +0.0092 | 0.00006 | +0.0064 | 59 / 34 / 22 | pass |
| without `copy` | 107 | +0.0099 | 0.00005 | +0.0069 | 58 / 29 / 20 | pass |
| **without `copy` and `possible` (pre-declared read)** | **94** | **+0.0107** | **0.0002** | +0.0073 | 49 / 27 / 18 | **pass** |
| without any flag | 84 | +0.0102 | 0.0008 | +0.0063 | 43 / 25 / 16 | pass |
| flagged only (`copy` + `possible`) | 21 | +0.0023 | 0.051 | +0.0015 | 10 / 7 / 4 | — |

- **Statement fixed in the PREREG: the gain survives.** The pre-declared read has a clipped gain above 0 with
  p < 0.025.
- The gain is, if anything, smaller on the flagged contexts (+0.002 u) than on the others (+0.011 u). Exposure of
  a table's features during training did not help the column here.
- What this does not show: that the remaining 94 contexts are free of copies the screen cannot see (below).

## CONFIRM
- **Feature screens: not run.** The guarded table view holds 661 files — 440 `FIT-ICL`, 115 `DEV`, 106 `FIT-SEL` —
  and none of the 241 `CONFIRM` tables. Screening them means staging them in the view or reading them outside the
  guard. The relayed note asked for the screen but not for that step, and it undoes a protection of the campaign
  plan, so it is left for an explicit decision (addendum 1).
- **Manifest only (no table opened):** 0 shared dataset ids, table hashes or family ids with `FIT-ICL`. 53 of 241
  contexts have a readable source name; 1 name flag: `meta-album-md-5-bis-mini` (`CONFIRM`, core) against
  `meta-album-md-6-mini` (`FIT-ICL`), probably two different datasets of one benchmark.
- **The PREREG's failure criterion cannot be evaluated.** Best estimate from the same screen on the cohorts it
  could open: `copy` in 8 of 115 `DEV` (7%) and 18 of 106 `FIT-SEL` (17%); `possible` in 11%. For the 142 core
  `CONFIRM` contexts that suggests roughly 10 to 25 copies. **Treat `CONFIRM` as contaminated for any model
  trained on `FIT-ICL` until it is screened.**
- **Decision needed (owner):** either stage the 241 `CONFIRM` tables read-only in the view so that the published
  screen can run (`--queries DEV,CONFIRM`, under a minute, feature hashes only, nothing leaves the sandbox), or
  accept an unscreened `CONFIRM` with the H424 column excluded from the champion, or pre-declare that the champion
  read reports results with and without contexts flagged afterwards.

## Integrity gates (`gates_h425.sh`, after the lists were published, before the sensitivity read; 8 of 8 pass)
| Gate | Evidence |
| --- | --- |
| PREREG and addendum first | both commits are older than the screen's output |
| `DEV` only, in the sandbox | 115 `DEV` contexts against 440 `FIT-ICL` tables, guard installed |
| `CONFIRM` not opened | no `CONFIRM` context in the output; the view still holds 661 files |
| List is the screen's | `EXCLUSION_DEV.json` carries the screen's sha256 (`fa8edd8a08a9…`), 115 contexts |
| Lists published before the sensitivity read | commit `1c44425f36`; no sensitivity output existed |
| H424 values unchanged | its value file has the sha256 H424 published |
| Budget | 24 seconds, 8 processes (cap 2 hours) |
| Published stage | code hash verified |

## Deviations, negatives and limits
1. **The PREREG's run could not be executed as written** (it included `CONFIRM`). The first attempt stopped on a
   missing `CONFIRM` table before any output; addendum 1 reduced the scored run to `DEV`.
2. **The tier rule was split into `copy` and `possible` after the first rehearsal on `FIT-SEL`** (PREREG §8), before
   any `DEV` score existed.
3. **The manifest-only `CONFIRM` screen ran once as a code test before addendum 1** (stated there).
4. **What the screen cannot see:** rescaled or re-encoded features; datasets on a coarse grid unless rows are shared;
   families without readable names for S4 (23 of 115 have one).
5. **False positives exist.** `bike-sharing-demand` is flagged against `seoul-bike-sharing-demand` by name and by
   target, and these are probably different datasets. The lists are conservative, not exact.
6. **A flagged table need not have leaked anything**: the model learned each training table's own target, and most
   copies carry another target.
7. **Not screened:** `DEV` against `FIT-SEL` and DL-Train (the design cohorts of H421 and H422); copies between `DEV`
   families themselves. Three `DEV` contexts match one `FIT-ICL` table, so they are also copies of each other:
   the "115 families" of every `DEV` test are not 115 independent datasets.

## Queued follow-ups
1. **Owner decision on `CONFIRM`** (above).
2. "Cluster the `DEV` and `FIT-SEL` contexts by this screen (copies of each other) and restate the H420, H422 and
   H424 tests with clusters as the unit, from stored values; no new table access beyond the screen."
3. "Add a rescaling-invariant column fingerprint (rank-order statistics) and re-run the rehearsal on `FIT-SEL` to see
   what S1 misses."

## Timeline (UTC, 2026-11-11)
| Time | Event |
| --- | --- |
| 10:07 | H425 claimed |
| 10:10–10:14 | rehearsal on `FIT-SEL` (twice; tier rule split) |
| 10:16 | PREREG and code published (`d21cb6a43d`); first attempt stops on a missing `CONFIRM` table |
| 10:18 | addendum 1 published (`74ebb9c908`); screen on `DEV` (24 s) |
| 10:19 | lists published (`1c44425f36`); gates 8 of 8 |
| 10:20 | sensitivity read |

Evidence (xbabe2): `~/jope_plan/evidence/H425/` (`results/SCREEN_DEV.json`, `results/EXCLUSION_DEV.json`,
`results/CONFIRM_META.json`, `results/H425.gates.json`, `results/H425_DEV.sensitivity.json`, `logs/`).
