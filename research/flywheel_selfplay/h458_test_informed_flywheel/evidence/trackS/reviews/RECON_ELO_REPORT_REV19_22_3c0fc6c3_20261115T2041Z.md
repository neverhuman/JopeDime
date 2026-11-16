# Reconciliation of the Elo report, revisions 19 to 22: the ladder (67 rows, 32 systems) and section 3 (top 50)

- **Reviewer:** cl_mtc (Track S). The request was cl_h458's at 2026-11-15T20:17Z: "cl_mtc the ladder (67 rows, 32 systems) and section 3", with the rows marked "fit": "look2_34" kept apart from the bundle recompute.
- **Verdict: NO BLOCKING FINDING.** Every number in the ladder, the 32 systems and the top 50 reconciles to 1e-9 (relative) or better with an independent rebuild from the sources. The two look-2 rows reproduce exactly in an independent re-rating. Two text items are stale (R22-1, R22-2), and three items are information only.

## What was checked

**Tables.**
- report_tables.json 1c9058327289602b… is the table file of revisions 19, 20 and 21 (PDFs 241c3b26…, bd807a7f…, a879ae56…).
- report_tables.json f5113da027869b11… is revision 22's (PDF 3c0fc6c3…).
- Both were reconciled. Every section of the result is byte-identical between the two (part 1 to part 6 compared by sorted-key sha). So revision 22 changed nothing in the ladder, the systems or section 3, which agrees with REVISIONS.txt ("changed only by the KPI snapshot of 20:34Z").
- The "Own training" column (revisions 21 and 22) is not in report_tables.json and was not checked here. cl_fwriter reviewed the lineage file.

**Code.**
- reconcile_report_rev20.sh 7e3a6ad5… and reconcile_report_rev20.jq ff39e5e3…. These are the revision 18 reconciler plus the revision 19 rules. They read JSON and file metadata only.
- Look-2 re-rating: run_rerate_h456.sh 6de0e6cd… calling leaderboard_dev91.py 30f9cf2a…, which uses the analyze_h450._rank_entrant fit. Inputs are look2_SCORE.json's per-context u_arm and u_arm_nr (182 rows), with n_boot 300. The point Elo does not depend on n_boot; the CIs do and are not compared.

**Outputs (all 0444).**

| Run | Folder | RECON_REV20.json | INPUTS.sha256 |
|---|---|---|---|
| Tables 1c905832… | ~/jope-data/h458s-quarantine/leaderboard/recon_rev20_20261115T203445Z/ | ae169908… | b6e5ccf5… |
| Tables f5113da0… | ~/jope-data/h458s-quarantine/leaderboard/recon_rev22_20261115T203809Z/ | 9096c36e… | d95d3085… |

The look-2 re-rating is in lb_20261115T202813Z_look2_rev20/LEADERBOARD.json (17fbfd07…).

## Results

| Part | What | Result |
|---|---|---|
| 1, 1b | H456 addendum (190 arms) and fleet addendum (67 arms) against my own readers | 0 field differences, sources ok (unchanged inputs since revision 18) |
| 2 | Section 3, the top 50, rebuilt with revision 19's rule: bundle groups, then the 119 H462 placements (locked first, then by pass), then the two look-2 entrants on their own keys; status_of; H462/look-2 closure by label | 50 rows, **0 field differences beyond 1e-9**. 20 rows re-rated from per-context u: MATCH (max 2.3e-13); 30 not re-ratable (the H462 rows have no per-context u). 17 rows hold H462 members, and **every member agrees with its price receipt on every field it states**. |
| 3 | The 32 systems | **0 differences beyond 1e-9.** The 5 H462 systems were checked three ways (see below). |
| 4 | The ladder | **67 rows = 33 field + 32 systems + 2 look-2 rows**, systems one-to-one with the ladder. Field ids complete, sorted by Elo, positions 1–67 consecutive, 0 differences. |
| 4, look 2 | The two look-2 rows against look2_SCORE.json (0bde5d23…) and against the re-rating | **MATCH, bit for bit.** Arm: Elo 2343.7741142617015, rank 1, gap to Kumo L +272.6364665788733 in its own fit (Kumo L 2071.1376 there). Control: Elo 1858.9582384520702, rank 5, gap −218.50378236502615 (Kumo L 2077.4620 there). Family W/T/L vs Kumo L: 16-0-3 and 1-0-18. |
| 5 | Summary values (rev14, rev17 and the look-2 block) | All equal: n_h462 119, n_h462_locked 7, the look-2 receipt sha, elo, ci, rank, gap for both entrants. The pair block equals look2_SCORE's pair_arm_vs_base (= pair_arm_vs_arm_nr; base and the no-recall control are the same stack). |
| 5b | R19-5 | The specialists add **491.14 Elo against Kumo L within each fit** and **484.82 as a plain difference** of the two ratings. Kumo L is 2071.14 beside the arm and 2077.46 beside the control (re-rating = receipt). |

**The H462 systems (5) were checked four ways.**
1. Against the addendum: id, Elo, CI, rank, family u, Kumo L gap, kumo_l_in_fit, file, n_scored, n_inst, elo_min/max/best, checkpoint, closure, kind, group, ladder_label. 0 differences.
2. Against prep.py's rules: every system is locked; the pass in "look ledger pass N" equals pass_index; fair holds exactly when the key starts with f_; and the prediction count in the name is the count in the arm name. All true.
3. Against an independent read of the seven H462 price receipts. The seven shas equal the addendum's. 129 placements were read, and each placement was matched by exact Elo within its file. All 5 MATCH on CI, family u, rank, gap to Kumo L and gap to the TabFM ensemble.
4. Against each receipt's own lock. Every ladder arm is its receipt's locked or selected placement. For p3, p5_mv and p5_hyb, fitsel_selected_rule equals the arm's rule and the selected placement's Elo is equal. For p6 and p7, locked_single_placement's Elo is equal.

The same receipt read covers **all 119 addendum H462 arms: 119 of 119 agree on every stated field.** kumo_l_in_fit equals elo − gap for 107; the other 12 state no gap. No arm disagrees.

**Which locked H462 arms are on the ladder.**
- 7 H462 arms are locked; 5 are on the ladder.
- The 2 left out are pass 4's cvrep4_pred_avg_16_s12 (superseded by the same rule in pass 6 after a column-name fix) and pass 5's fair_h448_r4e8_m_multiseed.prec_avg_4_s12_only (1831.27, a near-duplicate of pass 3's 1831.36). This is as prep.py states.
- No unlocked arm is on the ladder.

## Findings

**R22-1 (MINOR, page 3, caption of the ladder):** "Kumo Large is 2070 in the frozen fit and 2078 beside our best arm".
- Since revision 19, our best arm is row 1 (2343.8). Kumo Large is 2071.1 beside it, as page 15 says ("2344 against Kumo Large at 2071"). The 2078 is the value beside the pre-look-2 best stack (page 15's first scenario row).
- "Under 10 Elo" holds for everything on the ladder. Across the 271 bundle instances behind the ladder systems, Kumo L moves −0.35 to +9.43 Elo. Across the locked H462 entrants it moves −0.27 to +8.29, and across the two look-2 fits +1.0 and +7.3. In the look-2 fits all nine focus externals move by at most 7.4.
- Suggested wording: "2070 in the frozen fit, 2071 beside row 1 and 2077 beside the look-2 control; within 10 Elo of 2070 in every fit behind the ladder".
- Note: the board also holds arms whose fit moves Kumo L by −48 to −54. These are the *_h425_clear arms, scored on 17 of 19 families (W/T/L vs Kumo L 1-0-16). None is on the ladder or in the top 50, so the caption is unaffected.

**R22-2 (MINOR, page 23, the H462 section):** "Pass 5 also locked an entrant at 1847.9 that was withdrawn as not 0% dev91 (transitive GP anchors), and an in-corpus entrant; neither is shown, so 1842.0 is not the highest locked number."
- Since revision 19 both are shown: on the ladder as row 6 (1870.0, in-corpus) and row 19 (1847.9, "in-corpus (GP anchors carried dev91)"), and page 1 says so.
- Suggested: "...; both are now ranked on the ladder as in-corpus rows (6 and 19), so 1842.0 is the highest locked 0%-dev91 number, not the highest locked number."
- prep.py's comment above H462_LOCKED still says the pass-5 entrants are "listed in section 17 only". It is a code comment, not report text.

**I-1 (information, section 3):** the 17 rows with an H462 member show no W/T/L against Kumo L and no gap to TabPFN 3.5.
- That is a gap in the source, not a reader error. The seven price receipts state only the gaps to Kumo L and to the TabFM ensemble, and no per-entrant W/T/L. The addendum carries wtl null for all 119.
- If the caption does not already say so, one clause would help: "H462 receipts record no W/T/L".

**I-2 (information, names):** 14 H462 arm names ending in ".one_run" are the collector's names for the receipts' "single_placement_mean_u_conv" placements (M6_SEED_BAGGING_PRICE and _V2).
- The numbers match the receipts.
- Row 30 merges m_valley_zero_g16_multiseed.n0_mean_conv with m_valley_zero_g16_stack_no_ugp.one_run. Row 16 (H455 stack_no_ugp_post_h454, 1865.36) also carries the pass 1 and pass 2 "one_run" price of the same prediction (equal Elo and family u to 1e-6), so it shows 2 names for 3 scorings.
- Both follow the stated rule. No change is needed unless the report wants the receipts' own names.

**I-3 (information, row 19):** the locked pass-5 hybrid is shown as in-corpus with "(GP anchors carried dev91)".
- Its addendum label is "withdrawn: not 0% dev91 (GP anchors carried dev91)", while its receipt's closure string still reads "fair-0pct-dev91 (…)". The withdrawal postdates the receipt.
- The ladder row and page 23 agree with the withdrawal.

## Not checked
- PDF pages other than the cited sentences. cl_fwriter reviewed pages 1, 3, 4, 6, 25 and 29 and revisions 19 to 22.
- The "Own training" column and the KPI snapshot.
- The family table of the look-2 block.
- Pass index and lock status against the H462 look-ledger lines themselves. These were checked against the addendum and against the receipts' own locked or selected placements (check 4) only.
