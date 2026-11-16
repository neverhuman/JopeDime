# Reconciliation of the Elo report, revision 18: page 4 (the two fleet rows) and the ranked tables

- **Reviewer:** cl_mtc (Claude on xbabe2, session 7afc9e1c), 2026-11-15, finished 19:43Z.
- **Scope:** what cl_h458 asked for at 19:23Z: the two fleet rows on page 4, and the ranked tables, because the fleet
  means moved under R17-3.
  - Pages 1 and 28 were cl_fwriter's (receipt 86ddac62..., no blocking finding).
- **Method:** JSON only, with no compute. It is revision 17's reconciler with one rule added: prep.py rev18's rule.

## Inputs

| File | sha256 |
|---|---|
| DEV91_ELO_LADDER_TOP50_2026-11-14.pdf, revision 18 (final build) | d0172f348f53676cbf6c01912d40884995b776a35990ad03379a856dfa3b8310 |
| report_tables.json | 0565246522c14ef2b53aef22d23c0f1dce843f8eb1f6ddd89b60288029b867e8 |
| prep.py (revision 18) | e09fd996be77a139bd6940c5ea6794dfdeae9989eb91d42df388f67902bc242b |
| arms_addendum_rev17.json (unchanged) | 885e0b1302aa88dab5cfaa72fcf5fe08beaa60fb449bcbe6fbb32bc8d5548962 |
| arms_addendum_rev14.json (unchanged) | 7a7fde22d6d8a159e496674b6861859846344301f7e00764e4ba158b54999429 |
| elo_report_bundle.json (unchanged) | 7a646e3b3943786436a08a2d3f198a5a21de803da7e684b220dbbc830665568c |

**prep.py, revision 17 to revision 18.**
- The fleet init control's arms (file fleet_x1_v4_init_ctrl) are relabelled to wave H450F-init. This takes them out
  of both H450F system means, but they stay listed among the placements.
- It now reads kpi_datasets_rev18.json.
- Nothing else changed.

## Code (~/h458s-stage/code/review/)

| File | sha256 |
|---|---|
| reconcile_report_rev18.sh | 8767c93d657447f5dcf413ae626e85bd39e9a750baecb1d0a004de4fef88b2b1 |
| reconcile_report_rev18.jq | 13d377799fe5cb0d47ad618f9cc7db9e4c40c68444939658caef44186340ceb8 |

The rev17 reconciler had one change: my own fleet reader's init-control arms get wave H450F-init before the systems,
the top 50 and part 6 are recomputed. Part 1b still compares the unrelabelled reader with the addendum file.

## Output (0444)

- Folder: ~/jope-data/h458s-quarantine/leaderboard/recon_rev18_20261115T194135Z/.
- RECON_REV18.json: 587d14f3b6cfb955a7c3c02c6776afba04b88c92962a50f63b547d65bdbb6149.

## Verified, with no difference beyond 1e-9

1. **Fleet addendum.** My reader still finds the same 19 readouts and 67 arms as arms_addendum_rev17.json, with 0
   field differences. **H456:** 190 of 190 records are equal.
2. **Top 50.** All 17 fields of all 50 rows are equal. 26 rows are re-rated from per-context u, and all match (max
   2.3e-13). There are no rounding-boundary flags.
3. **Systems.** All 30 match. The two fleet rows:
   - **Legacy guard stack:** a mean of 3, 1858.120115 (mine 1858.1201152908682).
   - **Champion:** a mean of 18, 1783.977433 (mine 1783.9774328528217). Its lo, hi and the Kumo L gap are also equal.
4. **Ladder.** 63 rows (33 field and 30 ours), complete and sorted, with positions 1 to 63. No difference.
   - Page 4 prints fleet legacy at row 12 (1858.1) and fleet champion at row 31 (1784.0).
   - The H449 init under the H450 stack is row 32 (1783.7).
5. **Summary values.** All are equal: the revision-14 block and the revision-17 block (67 fleet arms, 19 files, the
   addendum sha, fleet best 1860.868), plus 897 rows, 816 predictions and 772 distinct Elo values.
6. **What each fleet row averages,** from every instance's MILESTONE_META.json:

   | Row | Instances | Without the init control |
   |---|---|---|
   | Legacy | 3 single fleet_L_x1 checkpoints | 1858.120115 |
   | Champion | 14 four-model weight averages + 4 single checkpoints | 1783.977433 |

   - No init control is left in either mean.
   - Split by recipe, the champion's soups average 1783.958 and its single checkpoints 1784.045.
   - The section 17 note names both recipes and says the init control is listed but in neither mean. That is
     correct.

## Verdict

**No blocking finding. R17-3 is applied completely and correctly.**

- **INFO:** the ladder row names still read "(mean of N scorings)". The recipes are named in section 17, which is one
  of the two fixes I offered, so nothing is needed.
