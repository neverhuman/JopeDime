# The black hole: H458 look 2 (dev91 Elo 2343.8), and what H464 and H465 are doing with it

Written 2026-11-16 by cl_h458 (Claude Code) at the owner's request, as context for the white paper. It is a record of
what was run and measured, with the file that backs each number. Every number carries a label; the labels are part of
the result. Times are UTC.

- Study folder: [`research/flywheel_selfplay/h458_test_informed_flywheel`](../../../../research/flywheel_selfplay/h458_test_informed_flywheel/README.md)
- Results record and file map: [`research/flywheel_selfplay/h458_test_informed_flywheel/RESULTS.md`](../../../../research/flywheel_selfplay/h458_test_informed_flywheel/RESULTS.md)
- Evidence copied into this repository: [`research/flywheel_selfplay/h458_test_informed_flywheel/evidence/`](../../../../research/flywheel_selfplay/h458_test_informed_flywheel/evidence/)

## 1. The one-paragraph version

On the `dev91-audit-v1` board (91 regression tables in 19 families, each with an official split into SEARCH rows and
AUDIT rows), our stack with one fine-tuned neural specialist per table family scores **Elo 2343.8** on the official
AUDIT rows: first of 35 entrants, 273 Elo above Kumo Large (2070.2). The specialists were trained on **all rows of
those tables, the AUDIT rows and their labels included**, on random splits. At assessment the system reads only a
table's SEARCH rows. The same stack without the specialists scores 1859.0, and our best system that never trained on
any dev91 table scores 1842.0. The owner's programme, called the black hole, is to do this on purpose and at scale:
train a tabular foundation model on every table we can obtain, test rows included, always on random splits and under
heavy augmentation, then assess it on the official splits of dev91, TabArena and BeyondArena, and say plainly in the
paper that this is what was done.

## 2. Labels, and the owner's rulings

Three labels are used on every dev91 number. They say what the model had seen of the table it is scored on.

| Label | What the model saw | Best number | What it supports |
|---|---|---|---|
| fair, 0% dev91 | no dev91 table in training | 1842.0 | prediction on unseen tables |
| in-corpus | dev91 tables in the generalist's training corpus (about 2% of episodes), no per-table fine-tuning | 1859.0 | a generalist that has seen the tables |
| recall | a model fine-tuned on the scored tables' rows, AUDIT labels included, random splits only | 2343.8 | the black-hole claim: memory of seen tables |

The owner's rulings, in his own words:

- 2026-11-15, the programme: train on "EVERY possible dataset... BeyondData, TabArena, Dev91, ALL of it"; "We will
  admit we trained on ALL of the data, but we did different splits, and did heavy augmentation"; test on "the exact
  datasets, and exact splits".
- 2026-11-16, about 04:27Z: "As soon as we have checkpoints that pass TabPFN3.5 or TabFM or a small Kumo model we want
  to know, we are writing a paper, and are fine if this is moving."
- 2026-11-16, about 04:33Z, on whether a number counts when the model has seen the table: "It is FINE if we have seen
  that data as long as we saw a different random split, we will speak to this in the paper. This is why this is called
  the black hole".

What is and is not established about "a different random split":

- Family specialists: their logs show `dup_splits 0`, which says no training episode repeated another. It does not say
  that no episode's split equalled the official split. A label-free check exists in design (look up the official
  split's commitment in each run's split ledger) and **has not been run**. The chance of a coincidence is below 1e-20
  per episode for a table of 100 rows or more.
- The r4 generalists behind the stack's other neural columns log no split identity. For a table larger than the
  largest training crop the official split cannot occur, by construction; for smaller tables the same bound applies.
  This is "by construction", not "checked".

The paper should say "by construction, and for the specialists pending an exact check", until that receipt exists.

## 3. The board, the metric and the ladder

- **Board.** `dev91-audit-v1`: 91 contexts (tables) in 19 families; 18 families have 5 contexts, one has 1. Each
  context has SEARCH rows (features and labels, 485 to 1,183 rows) and AUDIT rows (features; labels held by the
  scorer). The evaluation bank used by every column builder holds no AUDIT label.
- **Utility.** For one context, u = -ln(RMSE of the system on AUDIT rows / RMSE of the baseline on AUDIT rows).
  u = 0.69 halves the baseline's error. Family means are means over a family's contexts; the board mean is the mean
  over the 19 families.
- **Elo.** A ladder fitted on per-context wins against every entrant, anchored at `h335.H257_REPLICA` = 1500, with a
  bootstrap 95% interval. Intervals are wide (about 150 to 250 Elo); ranks near each other are not separated.
- **Pair test.** Used for every "A beats B" claim: family-mean difference, clipped; exact one-sided sign-flip test
  over the 19 families; leave-one-family-out minimum; family median.
- **Looks.** Every read of AUDIT labels is a "look", pre-registered in a ledger before it runs
  ([`evidence/LOOK_LEDGER.snapshot_*.jsonl`](../../../../research/flywheel_selfplay/h458_test_informed_flywheel/evidence/)). Looks are scored by a different agent than
  the one who built the system, with an unchanged scorer (`score_h458s.py`, sha256 0a9adb2c...).

## 4. What the 2343.8 system is

**4.1 The base stack ("legacy", 1859.0 on its own).** A stacked ensemble selected by cross-validation on SEARCH rows
only. Its members are columns of out-of-fold predictions: tree and linear learners, Gaussian-process members, and
in-context-learning (ICL) transformer columns from two large generalists (`r4_L_A`, `r4_L_B`, epoch 44 of the r4
series). Those generalists trained on a corpus in which dev91 tables are about 2% of episodes, which is why the stack
is labelled in-corpus. Training curves of both are in
[`evidence/training_curves/r4_L_A`](../../../../research/flywheel_selfplay/h458_test_informed_flywheel/evidence/training_curves/) and `r4_L_B`.

**4.2 The specialists (the +485 Elo).** One per family, 19 in all.

| Item | Value |
|---|---|
| Model | size-M ICL transformer, 23.66M parameters |
| Start | `fleet_M_x2_h455_maxent/ckpt_valley_002.pt` (sha256 ba90b6ef...), EMA weights |
| Training rows | the family pool: the de-duplicated union of all rows of the family's tables, SEARCH and AUDIT, with labels (`family_pool_v1`, sha256 33f29cac...) |
| Episodes | built by the evaluation column builder's own fold pipeline, so a training episode looks like an evaluation fold: unique-row sampling, contexts of 256 to 1,000+ rows, the same feature blocks and target transforms |
| Slots | canonical feature-identity slots (arm `canon`): one native-order view, the same at training and scoring |
| Dose | 2,000 exposures per row |
| Schedule | peak rate 5e-5, floor 5e-6, 300 warm-up steps, EMA 0.999, no weight decay, gradient clip 1.0 |
| Caps | 240,000 tokens and 12,000 row tokens per batch, at most 1,024 query rows |
| Losses | three heads: robust-ridge residual (`rr`), Yeo-Johnson target (`yj`), rank-Gauss target (`rg`) |
| Cost | 4 to 16 minutes per family on one 24 GB card |

Configs, final records and step logs of the 19 runs:
[`evidence/trackD/specialist_runs/`](../../../../research/flywheel_selfplay/h458_test_informed_flywheel/evidence/trackD/specialist_runs/). Code:
`train_h458.py`, `episodes_h458.py`, `family_pool_h458.py` in the study folder, staged at source commit c9c8b9c95e.

**4.3 Scoring a context.** For each context, its family's specialist produces three columns
(`icl_e8_cw12_robridge_cv7`, `icl_e8_cw12_yj_cv7`, `icl_e8_rankgauss_duan_cv7`): out-of-fold predictions on SEARCH
rows from 2 x 7-fold CV, and predictions on AUDIT rows from a refit on all SEARCH rows. The specialist's weights have
seen the AUDIT labels; the scoring step reads none. The stack then chooses weights by CV on SEARCH rows. The block
builder refuses to run unless every SEARCH and AUDIT row of the context is in the specialist's pool (coverage 1.0),
because a recall specialist is worse than its starting point on rows outside its pool.

## 5. The result (look 2, scored 2026-11-15 03:27Z to 03:32Z)

Spec, written before the look: [`evidence/trackD/look2/all91.SPEC.txt`](../../../../research/flywheel_selfplay/h458_test_informed_flywheel/evidence/trackD/look2/all91.SPEC.txt).
Receipt: [`evidence/trackD/look2/SCORE.json`](../../../../research/flywheel_selfplay/h458_test_informed_flywheel/evidence/trackD/look2/SCORE.json) (sha256 0bde5d23...).

| Quantity | Stack + specialists (recall) | Stack alone (in-corpus) |
|---|---|---|
| Elo | 2343.8 (95% interval 2181 to 2619) | 1859.0 (1726 to 2048) |
| Rank on the ladder | 1 | 5 among the 34 of that fit |
| Board mean utility | 2.716 | 1.681 |
| Against Kumo Large: Elo gap | +272.6 | -218.5 |
| Against Kumo Large: families won, tied, lost | 16, 0, 3 | 1, 0, 18 |
| Against Kumo Large: contexts won, lost | 74, 17 | |

Pair test, specialists against the stack alone: families won 17, tied 1, lost 1; mean gain +1.035 u; clipped mean
+0.369; sign-flip p = 7.6e-6; leave-one-family-out minimum +0.361; RMSE cut 64% on average. Pass.

Against the three models the owner named, contexts won and lost by the recall system: TabPFN 3.5 86 and 5; TabFM 81
and 10 (ensemble 82 and 9); Kumo small 83 and 8.

**Gate D-4 was not met.** The pre-registered gate asked for the system to beat the field's best on every context of
at least 13 of 16 target families. It did so in 11.

**Per family** (utility, mean over the family's contexts):

| Family | Contexts | Mean u, base stack | Mean u, with specialists | Gain |
|---|---|---|---|---|
| 08c91457 | 5 | 2.944 | 3.377 | +0.433 |
| 280b1a13 | 5 | 4.483 | 4.594 | +0.110 |
| 3a6cda83 | 5 | 0.809 | 0.908 | +0.099 |
| 4286560c | 5 | 3.458 | 4.385 | +0.926 |
| 469b72b7 | 5 | 1.836 | 2.232 | +0.395 |
| 538e6406 | 5 | 3.086 | 4.194 | +1.108 |
| 598bbb77 | 5 | 0.688 | 2.865 | +2.177 |
| 60b608db | 5 | 0.453 | 0.987 | +0.534 |
| 769243ee | 5 | 4.468 | 4.468 | +0.000 |
| 7d4e2dc9 | 5 | 0.425 | 3.069 | +2.644 |
| 866f66b1 | 1 | 1.056 | 2.154 | +1.097 |
| 90a58676 | 5 | 0.688 | 4.053 | +3.365 |
| 94f2455b | 5 | 0.982 | 1.941 | +0.959 |
| 9f338db6 | 5 | 1.154 | 3.692 | +2.538 |
| 9f67f144 | 5 | 1.459 | 1.455 | -0.004 |
| ce33f2a8 | 5 | 1.012 | 2.391 | +1.379 |
| d73e184e | 5 | 0.415 | 0.851 | +0.436 |
| f2893c38 | 5 | 0.797 | 1.164 | +0.366 |
| f8fd51f3 | 5 | 1.719 | 2.818 | +1.099 |

Where it still loses. Over all (context, entrant) pairs the recall system loses 79 of 3,003; 69 of those are in four
families. Two (9f67f144, 769243ee) are lost by 0.02 u or less. Two need a 2 to 3 times lower error than any dose of
this training gives: 280b1a13 sits on a resolution floor (RMSE/SD about 0.011 on rows it trained on, evenly spread),
and 4286560c is heavy-tailed (kurtosis 130 to 148; 87 to 99% of the squared error in the worst 1% of rows).

**The ladder at look 2** (35 rows; "ours" marks our systems; file
[`evidence/trackS/leaderboard_look2_rev20/LEADERBOARD.csv`](../../../../research/flywheel_selfplay/h458_test_informed_flywheel/evidence/trackS/leaderboard_look2_rev20/LEADERBOARD.csv)):

| Rank | Entrant | Ours | Label | Elo | 95% interval |
|---|---|---|---|---|---|
| 1 | `h458d.RECALL_CANON_E2000_ALL91` | yes | recall | 2343.8 | 2195 to 2612 |
| 2 | `kumo_tabular_large_e8.v1` |  | field | 2070.2 | 1892 to 2336 |
| 3 | `tabfm_ensemble.v1` |  | field | 1915.6 | 1752 to 2135 |
| 4 | `kumo_tabular_small_e8.v1` |  | field | 1900.2 | 1745 to 2124 |
| 5 | `tabfm.v1` |  | field | 1888.1 | 1727 to 2103 |
| 6 | `h458d.RECALL_CANON_E2000_ALL91__NO_RECALL` | yes | in-corpus | 1859.0 | 1723 to 2038 |
| 7 | `tabpfn_v3p5.v1` |  | field | 1851.6 | 1681 to 2097 |
| 8 | `tabpfn_v3p5_e1.v1` |  | field | 1834.7 | 1657 to 2077 |
| 9 | `h448.APEX_FAIR_CHAMPION` | yes | field | 1821.5 | 1697 to 1990 |
| 10 | `tabpfn_v3.v1` |  | field | 1789.8 | 1659 to 1960 |
| 11 | `h416.SXB` | yes | field | 1761.1 | 1657 to 1907 |
| 12 | `mitra_v2_zeroshot.v1` |  | field | 1722.6 | 1567 to 1922 |
| 13 | `tabicl_v2.v1` |  | field | 1702.8 | 1556 to 1890 |
| 14 | `h364.XSTACK_2L` | yes | field | 1699.0 | 1614 to 1817 |
| 15 | `tabdpt_1p3.v1` |  | field | 1689.5 | 1530 to 1890 |
| 16 | `h335.H335_AGG` | yes | field | 1683.2 | 1603 to 1795 |
| 17 | `h335.OLD_MATCHED_CV3` | yes | field | 1612.5 | 1556 to 1689 |
| 18 | `h335.Opus5p5_Extra` |  | field | 1562.3 | 1443 to 1686 |
| 19 | `h335.OLD_MATCHED` | yes | field | 1525.4 | 1477 to 1584 |
| 20 | `h335.H317_BASE` | yes | field | 1523.6 | 1489 to 1560 |
| 21 | `h335.Gemini4_argon_high` |  | field | 1518.3 | 1410 to 1631 |
| 22 | `h335.GPT6p1_Sol_Max` |  | field | 1504.3 | 1426 to 1588 |
| 23 | `autogluon_300s.v1` |  | field | 1504.0 | 1383 to 1620 |
| 24 | `h335.H257_REPLICA` | yes | field | 1500.0 | 1500 to 1500 |
| 25 | `catboost.v1` |  | field | 1468.3 | 1346 to 1584 |
| 26 | `h335.BANK_NN8_PORTFOLIO` | yes | field | 1456.6 | 1391 to 1526 |
| 27 | `h335.BANK_NN8_PICK` | yes | field | 1427.9 | 1371 to 1484 |
| 28 | `h414.kimik3p1_max` |  | field | 1411.6 | 1308 to 1507 |
| 29 | `xgboost.v1` |  | field | 1370.7 | 1256 to 1480 |
| 30 | `random_forest.v1` |  | field | 1361.8 | 1238 to 1469 |
| 31 | `h335.GPT6_Astra_Max` |  | field | 1313.0 | 1189 to 1424 |
| 32 | `lightgbm.v1` |  | field | 1312.6 | 1183 to 1421 |
| 33 | `ridge.v1` |  | field | 1253.4 | 1032 to 1391 |
| 34 | `h335.BANK_NN1` | yes | field | 1194.2 | 1082 to 1281 |
| 35 | `h412.grok4p7_xhigh` |  | field | 1123.3 | 991 to 1224 |

The fair series after this fit reached 1842.0 (`h462.M6_TABCV7_CVREP`, ledger line 11). It is the last of seven
successive dev91 reads, so it is an adaptive series and slightly optimistic; its first price was 1835.2.

## 6. Training and validation curves

What exists, and what does not:

- **Training loss** is logged by every trainer and is in the repository for the runs behind look 2 and for the
  all-data generalists that followed.
- **A held-out validation loss was not logged during training** by any of these trainers. The validation series that
  exist are of three kinds: (a) SEARCH-only out-of-fold utility of each specialist at its milestone checkpoints
  (250, 500, 1,000, 1,500, 2,000 exposures), which reads no AUDIT label; (b) dev91 Elo of the r4 generalists by epoch
  (`evidence/trackD/elo_report_20261114/r4_traj_final.tsv`); (c) probes on fresh rows that sit in no bank and no pool
  (`evidence/trackD/probe/`). A validation-loss line in the generalist trainers is proposed to the lead as a change
  for the remaining training.

Generalists (loss is the trainer's own mixed objective; runs differ in corpus and sampler, so compare within a run):

| Run | Logged points | Last step | Datasets fed | Loss, first 5 points | Loss, last 5 points |
|---|---|---|---|---|---|
| `h464_alldata/h464_alldata_L_x2` | 267 | 10420 | 360502 | 4.996 | 4.856 |
| `h464_alldata/h464_alldata_L_x2_r4` | 161 | 5510 | 200716 | 4.968 | 4.844 |
| `h464_alldata/h464_alldata_L_x2_r5` | 215 | 5581 | 211991 | 4.983 | 4.828 |
| `h464_alldata/h464_alldata_L_x3_r4` | 172 | 6543 | 243644 | 4.713 | 4.807 |
| `h464_alldata/h464_alldata_M_x2` | 276 | 21003 | 785780 | 4.859 | 4.910 |
| `h464_alldata/h464_alldata_M_x2_r4` | 214 | 18591 | 702767 | 4.944 | 4.805 |
| `h464_alldata/h464_alldata_M_x2_r5` | 228 | 11791 | 448563 | 4.762 | 4.774 |
| `h464_alldata/h464_alldata_M_x3_r4` | 219 | 18746 | 702934 | 4.812 | 4.771 |
| `init_fleet_M_x2_h455_maxent` | 789 | 45695 | 1459372 | 5.069 | 4.834 |
| `r4_L_A` | 5714 | 316385 | 7123603 | 4.393 | 4.381 |
| `r4_L_B` | 5738 | 368716 | 8296510 | 4.474 | 4.518 |

Specialists (three heads; loss falls on rows the model is memorising, as intended):

| Specialist run | Steps | Loss rr / yj / rg, first point | Loss rr / yj / rg, last point | SEARCH-only probe files |
|---|---|---|---|---|
| `p1_469b72b7_canon` | 1671 | 4.25 / 2.78 / 2.80 | 2.95 / 1.15 / 1.20 | 5 |
| `p1_60b608db_canon` | 1610 | 5.91 / 5.48 / 5.50 | 5.58 / 3.11 / 3.35 | 5 |
| `p1_f2893c38_canon` | 3911 | 5.56 / 4.80 / 5.04 | 4.88 / 2.97 / 3.04 | 5 |
| `p1b_90a58676_canon` | 3253 | 5.14 / 4.19 / 4.65 | 4.14 / 1.51 / 1.92 | 5 |
| `r1_08c91457_canon` | 2030 | 5.03 / 2.53 / 2.59 | 4.35 / 1.35 / 1.36 | 3 |
| `r1_280b1a13_canon` | 4504 | 4.55 / 1.72 / 2.95 | 3.19 / 0.84 / 1.42 | 3 |
| `r1_3a6cda83_canon` | 1359 | 4.59 / 4.15 / 4.63 | 3.84 / 2.79 / 3.41 | 3 |
| `r1_4286560c_canon` | 1923 | 5.19 / 4.27 / 4.42 | 3.86 / 1.98 / 2.20 | 3 |
| `r1_538e6406_canon` | 3129 | 5.30 / 1.87 / 2.00 | 5.04 / 0.62 / 0.64 | 3 |
| `r1_598bbb77_canon` | 2166 | 5.82 / 5.44 / 5.45 | 4.84 / 2.55 / 2.73 | 3 |
| `r1_769243ee_canon` | 6130 | 5.09 / 5.14 / 5.16 | 4.52 / 1.33 / 1.44 | 3 |
| `r1_7d4e2dc9_canon` | 1222 | 5.80 / 5.39 / 5.42 | 4.54 / 2.23 / 2.46 | 3 |
| `r1_866f66b1_canon` | 1447 | 4.18 / 3.93 / 4.05 | 3.88 / 2.25 / 2.40 | 3 |
| `r1_94f2455b_canon` | 4853 | 4.25 / 4.11 / 4.32 | 2.93 / 1.79 / 2.00 | 3 |
| `r1_9f338db6_canon` | 5433 | 5.52 / 4.33 / 4.70 | 4.33 / 1.07 / 1.36 | 3 |
| `r1_9f67f144_canon` | 3853 | 2.41 / 1.42 / 1.61 | 1.96 / 1.13 / 1.20 | 3 |
| `r1_ce33f2a8_canon` | 4505 | 4.26 / 2.50 / 2.59 | 3.34 / 0.92 / 1.10 | 3 |
| `r1_d73e184e_canon` | 5734 | 5.07 / 4.92 / 5.27 | 4.28 / 1.86 / 1.95 | 3 |
| `r1_f8fd51f3_canon` | 1255 | 5.46 / 3.93 / 3.98 | 5.17 / 2.44 / 2.63 | 3 |

Files: [`evidence/training_curves/`](../../../../research/flywheel_selfplay/h458_test_informed_flywheel/evidence/training_curves/). Each `log.jsonl` line carries step,
loss, learning rate, datasets fed, duplicate counters and memory. The recipe test's logs are not included: that test
is blind until its second read.

## 7. What the probes say about memory against skill

- The same specialists, scored on fresh rows of ten families (rows in no bank and no pool), are +0.47 u above their
  starting checkpoint (look-2 spec; `evidence/trackD/probe/FRESH_CANON_E2000_VS_INIT.json/SCORE.json`).
- A specialist scored on rows of a table outside its pool is worse than its starting checkpoint, by 0.09 to 0.15 u
  with canonical slots. Hence the coverage rule in 4.3.
- So the 2343.8 is mostly memory of the scored rows. The paper's claim is about what a model that has absorbed a
  table can do on that table's official split, not about unseen tables.

## 8. What followed (H464 and H465), as of 2026-11-16 05:00Z

- **H464, the all-data ruling (2026-11-15).** Always train on all data under random splits; the KPI is datasets fed
  through training. At 04:05Z on 16 Nov: 13.4M datasets on the all-data banks, 121M in all training ever, about 320
  per second across four hosts.
- **H465, the black hole.** The owner's answers on 15 Nov: everything enters training, the sealed cohorts included,
  after one last clean read; only weights are kept; wait up to 7 days for a benchmark we cannot yet score, then
  absorb; set the clean-data share by a test; split once, with a fallback; score once before absorbing; third
  benchmark BeyondArena.
- **Single-table models (WP1a, ledger line 23), recall.** A model fine-tuned on one table beat the stack alone on 10
  of 10 probe tables (+1.585 u) and beat the family specialist on 7 of the 9 multi-table families. Per-table models
  for all 91 tables were in training when this was written (81 runs; one look to follow, ledger line 26).
- **Tree memory (ledger lines 27 to 31), recall, a separate labelled system by the owner's ruling.** Per table, 32
  bags of boosted trees fitted on random 80% subsets of all rows, plus one fully grown residual tree per bag (dose
  T2, chosen by the owner). On ten tables: beats the stack alone on 9 (+1.314 u) and adds +0.219 u to the single-table
  neural model on 7. On the other 81 tables (one look, scored 04:20Z to 04:38Z on 16 Nov): it beats the stack alone
  on 77 of 81 (+1.152 u) and adds +0.385 u to the look-2 system (stack + family specialists) over 18 families, 17
  won, 1 tied, 0 lost, p = 7.6e-6. No Elo was computed for that look. Its receipts are on the training host, not
  yet in this repository.
- **Recipe test.** Three forks of one model on one card decide the share of clean and augmented episodes; blind until
  its second read (about 17 Nov 01:00Z). The new operator checks every episode's split key before the forward pass:
  177,000 datasets so far, 0 splits reused.
- **Public benchmarks.** TabArena fold 0 (51 tasks) and BeyondArena core (142 tasks) are each scored once before
  they enter training; target 18 Nov. A checkpoint snapshot for those reads is planned for 17 Nov 12:00Z.
- **Last clean read.** The sealed cohorts are read once, with Kumo Large, TabPFN 3.5 and TabFM beside our entrants,
  before they enter training. It needs the owner's signature and is the only number on tables no model of ours saw.
  Its pre-registration is deliberately not in this repository.

## 9. What a paper can and cannot say from this record

Can say, with the label:
- A stack with per-family fine-tuned specialists reaches 2343.8 Elo on dev91's official split, above every external
  model on the ladder, when the specialists were trained on all rows of those tables under random splits.
- The gain over the same stack without specialists is +1.035 u in board mean utility, 17 of 19 families, p = 7.6e-6.
- The generalist-only stack (1859.0, in-corpus) is above TabPFN 3.5 (1851.6) and below TabFM, Kumo small and Kumo
  Large; all intervals overlap.
- The system that never saw dev91 (1842.0) is below all three named models; intervals overlap.

Cannot say from this record:
- That any of our models beats those models on tables it never saw. That needs the last clean read.
- That the official split was never used in training as a checked fact (section 2).
- Anything about classification, large tables, TabArena or BeyondArena: not yet read.

## 10. Where things are

In this repository (text records only):

| Path under `research/flywheel_selfplay/h458_test_informed_flywheel/evidence/` | Content |
|---|---|
| `LOOK_LEDGER.snapshot_*.jsonl` | every pre-registration and look up to the snapshot time |
| `trackD/look2/` | look 2: spec, runner, block sidecar (checkpoint and pool hashes), SCORE.json, per-context utilities |
| `trackD/specialist_runs/` | the 19 specialists: config, final record, step log |
| `trackD/family_pool_v1/` | the pool's manifest (families, shapes, member tables, hashes); no rows |
| `trackD/probe/` | fresh-row and SEARCH-only probe scores |
| `trackD/elo_report_20261114/` | the Elo report: inputs, build scripts, figures, latest PDF, revision log |
| `trackS/` | the scorer track: pre-registrations, amendments, reviews, the look-2 ladder |
| `trackG/` | GP harvester receipts |
| `training_curves/` | training logs and validation probes (section 6) |
| `pre_specs/` | pre-registration files of the two follow-up looks owned by cl_h458 |
| `SHA256SUMS` | sha256 of every file in the folder |

Not in the repository, by rule: column blocks (`.npz`; a recall block's AUDIT predictions are near-copies of AUDIT
labels), checkpoints, pools, banks, any label file, the last read's pre-registration, and the recipe test's logs.
They are on the training hosts under `~/jope-data/` and are named with their hashes in the sidecar and the ledger.

## 11. Glossary

- **Context**: one table with its official SEARCH/AUDIT split. **Family**: tables from one source.
- **SEARCH / AUDIT**: the official training and test sides of a context.
- **Column, block**: a model's out-of-fold predictions for a context; a directory of them for many contexts.
- **Specialist**: an ICL model fine-tuned on a family's rows. **Dossier**: the same for a single table.
- **Recall, in-corpus, fair**: section 2. **Look**: a read of AUDIT labels, pre-registered.
- **Exposure**: one use of a row in a training episode. **Valley**: a checkpoint at a cosine schedule's minimum.
