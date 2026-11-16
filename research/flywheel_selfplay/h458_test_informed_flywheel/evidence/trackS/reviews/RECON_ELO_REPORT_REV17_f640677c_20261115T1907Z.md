# Reconciliation of the Elo report, revision 17: ranked tables, the fleet addendum and the summaries

- **Reviewer:** cl_mtc (Claude on xbabe2, session 7afc9e1c), 2026-11-15, finished 19:07Z.
- **Scope:** the ranked tables, which revision 17 changed through my R16-1 and R16-2 findings: the top 50, the systems,
  the ladder and the summary values.
  - cl_fwriter reviewed pages 1, 2, 28 and 29 (the KPI, section 19 and the cards) at 19:01:53Z, receipt 5403fa8e....
  - I did not re-check those pages.
- **Method:** JSON only. There was no compute this time, because no new row can be re-rated. The fleet readouts store a
  context count, not per-context u.

## Inputs

| File | sha256 |
|---|---|
| DEV91_ELO_LADDER_TOP50_2026-11-14.pdf, revision 17 | f640677c947b44dda418fb837397d5ebdb1dd167405279d7020c47dcb58dd2db |
| report_tables.json | 1fc1a434075632e81218cbd6fd827eb7bd668aa029855743740a0a4ff93ea70a |
| arms_addendum_rev17.json | 885e0b1302aa88dab5cfaa72fcf5fe08beaa60fb449bcbe6fbb32bc8d5548962 |
| arms_addendum_rev14.json (unchanged) | 7a7fde22d6d8a159e496674b6861859846344301f7e00764e4ba158b54999429 |
| elo_report_bundle.json (unchanged; the report copy equals the collector's) | 7a646e3b3943786436a08a2d3f198a5a21de803da7e684b220dbbc830665568c |
| prep.py (= prep.rev17.py) | 65b5c872610684874782373c055b493b4531991784816cb3a004502f32bfc4f0 |

The 5 H456 matrices, all 54 H450 diag_results readouts and the re-rating files are listed with their shas in the output
folder's INPUTS.sha256.

## Code (~/h458s-stage/code/review/)

| File | sha256 |
|---|---|
| reconcile_report_rev17.sh | d9f98d4e967f4ff5d27eb4b1fc576e337ea1883bbbced91b3e6063df604604c3 |
| reconcile_report_rev17.jq | 43b0a76e4ee7da0ebb8a446e53667c30cfb934885c44023412e7893400344cbf |

The code is revision 16's reconciler with these additions:
- an **independent fleet reader** that reads the raw readouts and selects every H450 diag_results readout the bundle does
  not hold, except epoch 48;
- revision 17's top-50 key;
- fleet checkpoint labels;
- the revision-17 summary block;
- a **recipe check** of each fleet instance, read from its MILESTONE_META.json.

## Output (0444)

- Folder: ~/jope-data/h458s-quarantine/leaderboard/recon_rev17_20261115T190555Z/.
- RECON_REV17.json: e265e62e1207a447a8e6b0cb031d1fd50c87b36e16056f036e0b8de4182a7e35.

## Verified, with no difference

1. **Fleet addendum (R16-2 applied).**
   - My reader selects the same 19 readouts as collect_rev17.py, with no twentieth file, and the same 67 arms.
   - All 67 records are equal in every field: entrant, Elo, CI, family u, rank, n_entrants, gaps, W/T/L, gains, Kumo L in
     fit and mtime.
   - The 19 source shas and arm counts are equal. All 67 are wave H450F and labelled in-corpus.
2. **H456 addendum.** 190 of 190 records are still equal to the five matrices.
3. **Top 50, under revision 17's rule** (arm, Elo and family u, each to 1e-6).
   - It equals the table in all 17 fields of all 50 rows.
   - 26 rows are re-rated from per-context u, and all match (max 2.3e-13). They include the two new H453 gen1000 rows,
     35 and 49, which have delta 0 in the 11-14 reconciliation.
   - The other 24 rows, including the fleet row 32, are checked against their source records only.
   - There are no rounding-boundary flags.
4. **Counts.**
   - 897 distinct rows, 816 distinct predictions and 772 distinct Elo values: equal.
   - Top-50 waves: H450 8, H450F 1, H452 2, H453 6, H455 22, H456 11: equal.
   - The new top-50 rows are 32 (fleet epoch-173 legacy stack, 1860.868, label "r4 g173 fleet L-x1"), 35 (1860.786)
     and 49 (1860.413), as predicted at revision 16.
5. **Systems.** All 30 are recomputed, with no difference beyond 1e-9, including the two new fleet rows: legacy 1857.67
   as a mean of 4, and champion 1784.16 as a mean of 19.
6. **Ladder.** 63 rows (33 field and 30 ours), complete and sorted, at positions 1 to 63. No difference.
7. **Summary values.** All equal: the revision-14 block recomputed under revision 17, and the revision-17 block
   (n_fleet_arms 67, n_fleet_files 19, the addendum sha and fleet_best 1860.868).

## Finding

### R17-3 (LOW, labels; numbered after cl_fwriter's R17-1 and R17-2)

**The two fleet ladder rows each average more than one recipe under the name "H450 fleet L-x1".** Each instance's
MILESTONE_META.json gives its recipe.

| Row | Instances |
|---|---|
| Champion (mean of 19) | 14 four-model soups, 4 single fleet_L_x1 checkpoints, 1 init control |
| Legacy guard stack (mean of 4) | 3 single fleet_L_x1 checkpoints, 1 init control |

- **The soups.** The 14 fleet_x1_ms_4525k to ms_8300k reads have soup_members_count 4. agy names the members as
  fleet_L_x1, fleet_L_x4, r4_L_A and r4_L_B.
- **The single checkpoints.** ms_0000k (soup_members_count 1), solo_ep186, g67 and g173.
- **The init control.** fleet_x1_v4_init_ctrl is mode v4_init_control: the all_non_d91_L init, with
  evaluated_unique_datasets 0.
- **Why it matters.**
  - The r4 rows drop their init epoch (prep.py, the g00 rule). The same rule would drop the init control here.
  - Under the new page footer ("trained on all data including dev91, under random splits"), the init control does not
    belong either, because it saw no all-data episodes.

**Effect: no rank changes.**

| Row | As printed | Without the init control | Ladder row |
|---|---|---|---|
| Legacy | 1857.67 | 1858.12, mean of 3 | 12 either way |
| Champion | 1784.16 | 1783.98, mean of 18 | 31 either way |

Split by recipe, the champion's soups alone (1783.96) and its single checkpoints alone (1784.04) also sit at row 31.

**Fix.**
1. Drop fleet_x1_v4_init_ctrl from both means, as g00 is dropped.
2. Name the champion row for what it averages, for example "fleet L-x1 checkpoints and 4-model fleet soups (mean of 18)",
   or give the soups their own row.
3. Optionally, label row 32 "fleet L-x1 epoch 173" rather than "r4 g173 fleet L-x1".

## Not covered

- Pages 1, 2, 28 and 29 (cl_fwriter).
- The figures.
- The H462 rows beyond their ladder copies.

## Verdict

**No blocking finding** on the ranked tables of revision 17. R16-1 and R16-2 are applied completely and correctly.
R17-3 is a labelling fix with no effect on any rank.
