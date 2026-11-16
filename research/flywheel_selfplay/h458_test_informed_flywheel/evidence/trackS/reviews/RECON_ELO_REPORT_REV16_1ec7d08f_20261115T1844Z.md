# Reconciliation of the Elo report, revision 16: H456 and in-corpus rows

- **Reviewer:** cl_mtc (Claude on xbabe2, session 7afc9e1c), 2026-11-15, finished 18:44Z.
- **Requested by:** cl_h458 (18:11Z and 18:34Z) and cl_fwriter (18:10Z and 18:12Z). They left these rows to this check.
- **What it does:** reads JSON only, with its own jq readers. There is one small CPU job, the re-rating described in check 2.

## Inputs

| File | sha256 |
|---|---|
| DEV91_ELO_LADDER_TOP50_2026-11-14.pdf, revision 16 | 1ec7d08f91a375b5d9940d6afe038278d729eef53bf5f5b3d21b92d4005f977d |
| report_tables.json | b5e91bf9bac70dcef8b7e37a73e79e1fd5266a1f2b71d4ba28ba1c85f7f74232 |
| arms_addendum_rev14.json | 7a7fde22d6d8a159e496674b6861859846344301f7e00764e4ba158b54999429 |
| elo_report_bundle.json | 7a646e3b3943786436a08a2d3f198a5a21de803da7e684b220dbbc830665568c |
| r4_g48 readout (the report-folder copy equals the original) | 191581ed9ec1d48698663d9cb996757cf38989fb6299c8cd84b03ba738a612e0 |

The five H456 matrices are read directly; their shas are in INPUTS.sha256.

## Code (~/h458s-stage/code/)

| File | sha256 |
|---|---|
| review/reconcile_report_rev16.sh | 423450da900777cc8cf7fc29cc09dd22bf266188dd0cb05037d2b23679c5fdcc |
| review/reconcile_report_rev16.jq | 9240f89085c3bf2c52ae1e0fbacbba104b398f304a2c5984f985f72559d817d4 |
| review/reconcile_report_rev16_dedupe_alt.jq | 432b2a96eaafd42b059fc34500dcd2c2184ec2ff0a5523a14e304de2bbfc5b10 |
| review/reconcile_report_rev16_dedupe_alt_fleet.jq | a7cde372524b21b3bbae3dba270d7c575ab597c36d6c49d0028e77d7024b4e21 |
| leaderboard_dev91.py | 30f9cf2ad369ae519baab9adff67b59be2ad727d95e96a0f854a2304eb57d3ac |

- leaderboard_dev91.py uses the canonical method, analyze_h450._rank_entrant: each entry is fitted alone into the frozen 33.
- It is run by review/run_rerate_h456.sh on review/make_h456_rerate_meta.sh's meta.

## Outputs (0444)

The folder is ~/jope-data/h458s-quarantine/leaderboard/recon_rev16_20261115T183842Z/.

| File | sha256 |
|---|---|
| RECON_REV16.json | cfc35640151e62db630bb59fbddc8919524e2e64bc047f76b74b389ca9e88735 |
| TOP50_STATED_RULE.v2.json | 01d2ce8b797151558761f4d780a31b2ae8964b0f43534d10a7c75c82adf523d4 |
| TOP50_WITH_FLEET.json | fdaca499c2de030b566bc93a9487277d311282af849caaa61559ef49cbd91f2d |
| TOP50_CORRECTED_VS_REPORT.json | ce8693cb9bf085147d5afe918ea5a9204ea9e0fea32ee7792a2d689afca89649 |
| in_FLEET.json | f33afb2df52d17b6201d089f793024edc16f76960ea1acf068ea0b3d77c400b7 |

- The folder also holds the scan lists and INPUTS*.sha256.
- TOP50_STATED_RULE.json is an empty file left by a failed first run. It is kept, with a .SUPERSEDED note.
- The re-rating is ~/jope-data/h458s-quarantine/leaderboard/lb_20261115T183000Z_h456_rev16/LEADERBOARD.json (f3502f90...).

## Verified, with no difference

1. **H456 addendum vs the five matrices.**
   - All 190 records are equal field by field: entrant, Elo, CI, family u, rank, n_entrants, gaps, W/T/L, gains, Kumo L in fit and mtime.
   - The five file shas and arm counts equal the addendum's.
   - All 190 records are labelled in-corpus.
2. **Re-rating from per-context u.**
   - The 21 H456 entries with per-context u (stage3_in, 1efc6de3...) are four arms at each of the five checkpoints, plus the control.
   - Each re-rates alone into the frozen 33 to its own matrix's Elo, max |delta| 2.3e-13.
   - The field-only fit reproduces ladder.json exactly.
3. **Top 50.**
   - It is rebuilt from the bundle, epoch 48 and my H456 reader under prep.py's rule.
   - It equals the report's table in all 17 fields of all 50 rows.
   - 25 of the 50 rows are re-rated from per-context u, and all 25 match (max 2.3e-13). They cover the 11 H450 r4 epochs and H455 and H456 checkpoints.
   - The other 25 are checked against their source records only, because no per-context u is stored for them.
4. **Systems (28).**
   - I recomputed the means over instances, ranges, counts, best instance, rank and the gap to Kumo L. Nothing differs beyond 1e-9: the only differences are summation rounding, at most 4e-13.
   - The four r4 systems have 34 of 34 instances re-rated.
   - The H456 stack has 5 of 5 re-rated; its mean, 1862.81, is verified.
5. **Ladder (61 rows: 33 field and 28 ours).**
   - The field rows equal the board's ladder.json, and all 33 entrants are present.
   - Our rows equal their system rows: Elo, interval, closure label, scored-instance text, rank and name.
   - The rows are sorted, at positions 1 to 61.
6. **Revision-14 summary values.** All are equal: the h456_stack list, h456_best 1863.62, n_h456 190, n_bundle_arms 1138, the top-50 wave counts, "all rank 5", the Kumo W/T/L of [1, 0, 18], the addendum sha and the row counts.

## Findings

### R16-1 (LOW, table rule)

**What happens.**
- prep.py's first dedupe step keys on the arm and its Elo to 0.01. So it also merges distinct predictions of one arm: 53 groups whose family mean u differs.
- Their Elo is identical because Elo uses only each context's sign.
- The report's stated rule is "one row per distinct prediction" (identical Elo and identical family u).

**Effect on the top 50.**
- Two such predictions reach it:
  - H453 stack_ugp_g10_2xcv7 at gen1000 progression, 1860.786. It would be #34.
  - H453 stack_ugp_g50_cv7 at gen1000 progression, 1860.413. It would be #48.
- The current #49 (H453 stack_ugp_g50_2xcv7, 1860.407) and #50 (H456 stack_pure104m_ugp_g10_cv7_post_h454, 1860.318) drop out.

**Printed numbers that change.**
- "774 distinct rows, 702 distinct predictions" becomes 834 and 753 (build_pdf.py:284 and :326).
- The top-50 span goes from 5.0 to 4.9.
- "12 of them enter the top 50" (H456, build_pdf.py:182) becomes 11.

**Fix.** Key the first dedupe on the arm, its Elo and its family u, each to 1e-6. No system mean, ladder rank or label changes.

### R16-2 (MEDIUM, completeness)

**The gap.**
- 19 H450 fleet dev91 readouts are neither in the bundle nor mentioned anywhere in the report. They were written on 11-13 between 03:30Z and 16:21Z, before the bundle.
- The bundle's H450 sources are exactly the 34 r4_L_A/r4_L_B epoch readouts.
- The 19 files are:
  - 15 fleet_x1_ms_* every-n reads;
  - fleet_x1_solo_ep186 and fleet_x1_v4_init_ctrl;
  - two fleet_L_x1 reads through the r4 queue, r4_g67_fleet_L_x1 and r4_g173_fleet_L_x1.

**What they hold: 67 placements.**

| Arm | Scorings | Values |
|---|---|---|
| Legacy guard stack | 4 | mean 1857.67, best 1860.87 |
| Champion | 19 | mean 1784.16 |
| No-GP | 19 | mean 1673.2 |
| ICL only | 19 | mean 1621.11 |
| Fair control | 2 | 1783.66, the r4 value |
| h425_clear variants | 4 | — |

**Effect.**
- The best of them, the fleet_L_x1 epoch-173 legacy stack at 1860.868, would enter the top 50 at #32.
- Revision 14's line "ALL dev91 Elo placements scored up to 2026-11-15 18:00Z are in the tables" is therefore not yet true.

**Resolution: one of two.**
- **Add them.** The owner, agy, should confirm the labels first, fleet_x1_v4_init_ctrl in particular. ckpt_label() would print "r4 epoch 173" for a fleet_L_x1 checkpoint, so these rows need their own label.
- **State the exclusion** and its reason on page 1 and in section 17.

### Combined effect on the top 50 (R16-1 and R16-2 together)

- Rows 1 to 31 are unchanged.
- **Enter:**
  - #32, the fleet_L_x1 epoch-173 legacy stack, 1860.868;
  - #35, H453 g10_2xcv7 at gen1000, 1860.786;
  - #49, H453 g50_cv7 at gen1000, 1860.413.
- **Leave:**
  - the old #48, H453 g50_cv7 at gen250, 1860.407;
  - the old #49, H453 g50_2xcv7 at gen1000, 1860.407;
  - the old #50, H456 pure104m UGP g10, 1860.318.
- **Waves:** H450 9, H452 2, H453 6, H455 22, H456 11. The span is 4.9.

### INFO-1 (JSON only, not printed)

- For the r4 systems, elo_min, elo_max and n_scored (35) include the excluded init epoch g00.
- Its value sets the lower bound for s_r4_legacy (1852.78) and s_r4_icl (1651.96).
- The means and the printed "mean of 34 epochs" are right.

### INFO-2 (JSON only)

- all_unique labels the 23 Ultra-GP fair-closure rows "fair", without the ladder's "GP prior in-corpus" qualifier.
- The by-component figure draws them hollow, as fair (figs.py:142).
- The PDF text already states that these rows are fair only as far as the ICL corpus goes.

## Not covered

- The figures, apart from the counts above.
- The external-only league.
- The H462 rows beyond the three ladder copies. agy and cl_fwriter covered the H462 labels.

## Verdict

- **No blocking finding** on the H456 and in-corpus rows: every number checked equals its source.
- R16-2 needs a decision before the report is called complete under the owner's "all placements" ask: add the 19 fleet readouts, or state that they are left out.
- R16-1 is a small rule fix.
