# H240: lower-LR / longer-horizon AdamW — COMPLETE / NO_ADVANCE

Hypothesis: smaller AdamW steps over the same or longer horizon yield adapters that the strict selector admits, improving DL-Val91 utility over r882.

Execution source `d3807eefad1d115809a3c5fd20b84ec2cab5de9b`, prereg PR #2083. Offline r882 COPY; CPU xbabe2 cores 40–43, nice 19, one math thread. Only 47 adapter tensors / 29,211 parameters train. Seeds 240101–240103, fresh family fold seeds 240141–240143.

| Arm | LR | Updates | Selector steps |
|---|---:|---:|---|
| CONTROL | 1e-4 | 40 | 0/10/20/30/40 |
| LR3E5 (primary) | 3e-5 | 40 | 0/10/20/30/40 |
| LR1E5_120 (secondary) | 1e-5 | 120 | 0/30/60/90/120 |

All arms retain exact H170 CTRL AdamW betas, weight decay, loss, sampler and clip 1.0; the long arm shares the exact first 40 minibatches.

Native91 PASS: all 91 utility values exact, mean **1.1377941898164525**, cached/native predictions bit-identical before any TRAIN pools. TRAIN smoke PASS: three disposable updates and exact r882 restoration. Full study: **15/15 cells, 3,000 primary updates**, 180 saved adapter snapshots; every snapshot/selection lock and the primary/secondary family statistics replayed.

The strict selector retained step 0 (r882) in **45/45 trajectories**. Each lower-LR treatment had **0/3 positive seeds**. All terminal adapter snapshots changed. Duplicate prediction/adapter groups and all per-seed selections are disclosed in metrics. Degenerate intervals caused by incumbent fallback do not establish independent equivalence of the trained adapters.

| DL-Val contrast | Median equal-family gain | Family CI95 | Holm p | Advance |
|---|---:|---|---:|---|
| LR3E5 vs r882 | +0.0000000000 | [+0.0000000000, +0.0000000000] | 1.00000000 | False |
| LR1E5_120 vs r882 | +0.0000000000 | [+0.0000000000, +0.0000000000] | 1.00000000 | False |

Primary LR3E5 verdict **NO_ADVANCE**; aggregate per-arm verdict **NO_ADVANCE**. The frozen gate requires median gain > +0.002, family LCB95 > 0, Holm p < .05 and at least 2/3 seeds positive. Ties reject. CONTROL contrasts are descriptive. No AUC or speed verdict.

The requested FIT-phase best-step diagnostics use the **held-out TRAIN selector (41 families per cell)**. The optimizer-fit subset has 123 different families; no optimizer-fit utility CI was collected. These are descriptive best-step bootstrap intervals, selected from five points, and are not simultaneous intervals. Every row below was replayed from stored utilities; the unchanged strict gate requires gain > +0.001 and LCB95 > 0.

| Fold | Seed | Arm | Best step | TRAIN-selector gain | Family CI95 | Strict step |
|---|---:|---|---:|---:|---|---:|
| A0 | 240101 | CONTROL | 10 | +0.03181594 | [-0.01488427, +0.09955371] | 0 |
| A0 | 240101 | LR3E5 | 40 | +0.02050595 | [-0.03514261, +0.09275876] | 0 |
| A0 | 240101 | LR1E5_120 | 120 | +0.02098928 | [-0.03542917, +0.09353265] | 0 |
| A0 | 240102 | CONTROL | 20 | +0.02826514 | [-0.02334116, +0.09842532] | 0 |
| A0 | 240102 | LR3E5 | 40 | +0.04171116 | [-0.00135565, +0.10936457] | 0 |
| A0 | 240102 | LR1E5_120 | 120 | +0.04145470 | [-0.00137870, +0.10833023] | 0 |
| A0 | 240103 | CONTROL | 20 | +0.00798931 | [-0.01249295, +0.03926411] | 0 |
| A0 | 240103 | LR3E5 | 40 | +0.01397305 | [-0.00512595, +0.04365669] | 0 |
| A0 | 240103 | LR1E5_120 | 120 | +0.01464964 | [-0.00434546, +0.04542039] | 0 |
| A1 | 240101 | CONTROL | 0 | +0.00000000 | [+0.00000000, +0.00000000] | 0 |
| A1 | 240101 | LR3E5 | 10 | +0.00051681 | [-0.00005976, +0.00158032] | 0 |
| A1 | 240101 | LR1E5_120 | 30 | +0.00033801 | [-0.00055099, +0.00153150] | 0 |
| A1 | 240102 | CONTROL | 0 | +0.00000000 | [+0.00000000, +0.00000000] | 0 |
| A1 | 240102 | LR3E5 | 20 | +0.00182805 | [-0.00394028, +0.00807872] | 0 |
| A1 | 240102 | LR1E5_120 | 60 | +0.00181968 | [-0.00395837, +0.00825101] | 0 |
| A1 | 240103 | CONTROL | 0 | +0.00000000 | [+0.00000000, +0.00000000] | 0 |
| A1 | 240103 | LR3E5 | 20 | +0.00286841 | [-0.00103696, +0.00954060] | 0 |
| A1 | 240103 | LR1E5_120 | 60 | +0.00208895 | [-0.00188817, +0.00864345] | 0 |
| A2 | 240101 | CONTROL | 10 | +0.00221335 | [-0.00332494, +0.00985072] | 0 |
| A2 | 240101 | LR3E5 | 20 | +0.00281671 | [-0.00288792, +0.01041887] | 0 |
| A2 | 240101 | LR1E5_120 | 60 | +0.00393360 | [-0.00099194, +0.01156975] | 0 |
| A2 | 240102 | CONTROL | 0 | +0.00000000 | [+0.00000000, +0.00000000] | 0 |
| A2 | 240102 | LR3E5 | 10 | +0.00004168 | [-0.00010689, +0.00023682] | 0 |
| A2 | 240102 | LR1E5_120 | 30 | +0.00004412 | [-0.00010689, +0.00023926] | 0 |
| A2 | 240103 | CONTROL | 0 | +0.00000000 | [+0.00000000, +0.00000000] | 0 |
| A2 | 240103 | LR3E5 | 10 | +0.00008014 | [+0.00000000, +0.00021458] | 0 |
| A2 | 240103 | LR1E5_120 | 30 | +0.00008014 | [+0.00000000, +0.00021458] | 0 |
| A3 | 240101 | CONTROL | 0 | +0.00000000 | [+0.00000000, +0.00000000] | 0 |
| A3 | 240101 | LR3E5 | 0 | +0.00000000 | [+0.00000000, +0.00000000] | 0 |
| A3 | 240101 | LR1E5_120 | 90 | +0.00193214 | [-0.00499406, +0.01063359] | 0 |
| A3 | 240102 | CONTROL | 0 | +0.00000000 | [+0.00000000, +0.00000000] | 0 |
| A3 | 240102 | LR3E5 | 0 | +0.00000000 | [+0.00000000, +0.00000000] | 0 |
| A3 | 240102 | LR1E5_120 | 60 | +0.00137363 | [-0.00124292, +0.00534684] | 0 |
| A3 | 240103 | CONTROL | 0 | +0.00000000 | [+0.00000000, +0.00000000] | 0 |
| A3 | 240103 | LR3E5 | 0 | +0.00000000 | [+0.00000000, +0.00000000] | 0 |
| A3 | 240103 | LR1E5_120 | 0 | +0.00000000 | [+0.00000000, +0.00000000] | 0 |
| A4 | 240101 | CONTROL | 10 | +0.00137129 | [-0.00311925, +0.00688130] | 0 |
| A4 | 240101 | LR3E5 | 30 | +0.00189913 | [-0.00099432, +0.00662152] | 0 |
| A4 | 240101 | LR1E5_120 | 60 | +0.00161869 | [-0.00132071, +0.00629484] | 0 |
| A4 | 240102 | CONTROL | 30 | +0.01039094 | [-0.01881382, +0.04418220] | 0 |
| A4 | 240102 | LR3E5 | 40 | +0.00287659 | [-0.00239056, +0.01079628] | 0 |
| A4 | 240102 | LR1E5_120 | 120 | +0.00311001 | [-0.00192126, +0.01086826] | 0 |
| A4 | 240103 | CONTROL | 10 | +0.00752779 | [-0.04523807, +0.07551616] | 0 |
| A4 | 240103 | LR3E5 | 40 | +0.02072301 | [-0.01926721, +0.08245713] | 0 |
| A4 | 240103 | LR1E5_120 | 120 | +0.02078278 | [-0.01859311, +0.08283221] | 0 |

The TRAIN-selector-best policies were also scored on DL-Val without the acceptance gate (descriptive only):

| Arm | Median gain vs r882 | Family CI95 |
|---|---:|---|
| CONTROL | -0.002226073 | [-0.006162100, +0.000540166] |
| LR3E5 | -0.002304928 | [-0.006319432, +0.000832811] |
| LR1E5_120 | -0.003771492 | [-0.009865375, +0.000925638] |

| Elo lift: median across seeds [family CI95] | vs r882 | vs CONTROL |
|---|---|---|
| CONTROL | +0.000000 [+0.000000, +0.000000] | +0.000000 [+0.000000, +0.000000] |
| LR3E5 | +0.000000 [+0.000000, +0.000000] | +0.000000 [+0.000000, +0.000000] |
| LR1E5_120 | +0.000000 [+0.000000, +0.000000] | +0.000000 [+0.000000, +0.000000] |

All strict policies have Elo 1500 [1500, 1500] with r882 anchored at 1500. Elo CIs use 1,000 paired family draws of the three-seed median. Utility CIs use 10,000 paired family draws.

| Arm | Terminal step | Median L2 drift [min, max] | Harm families: strict / unconstrained best |
|---|---:|---|---:|
| CONTROL | 40 | 0.380392651 [0.360336543, 0.391925908] | 0 / 2 |
| LR3E5 | 40 | 0.111981979 [0.105547731, 0.115066282] | 0 / 2 |
| LR1E5_120 | 120 | 0.107329341 [0.105073118, 0.111586230] | 0 / 3 |

Harm means a three-seed mean family utility delta below −0.01 vs r882, over 19 DL-Val families. Drift is measured at step 40/120; harm and utility refer to selected or unconstrained-best assessed policies. Secondary TRAIN OOF1219/205 and all per-cell drift values are retained in metrics.

| DL-Val context utility (descriptive) | Median across seeds | Minus r3175 plateau 1.1389572 |
|---|---:|---:|
| r882 | 1.1377941898 | -0.0011630102 |
| CONTROL | 1.1377941898 | -0.0011630102 |
| LR3E5 | 1.1377941898 | -0.0011630102 |
| LR1E5_120 | 1.1377941898 | -0.0011630102 |

The historical plateau comparison uses context means; the primary gate uses paired equal-family gains vs r882. Elo versus r3175 is null: its scalar utility 1.1389572 does not identify paired outcomes. No new plateau evaluation was run.

Red-team PASS: DOPE before/after `c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814`, non-adapter bytes frozen; only the adapter updated; actual fit/selector/assessment families disjoint and all exclude DL-Val/meta-test; higher utility is better; all three seeds and arm budgets reported. Worker exit 0, peak RSS 3392389120 bytes / 3 threads, minimum MemAvailable 67139280896 bytes / disk 761940684800 bytes. Load/RAM pause transitions replay PASS; 0 events. No OOM/disk failure.

Next step: retain the incumbent; do not promote either lower-LR treatment from this wave. No follow-on compute is authorized here.

Full output, source, starting COPY/reference and raw receipts: `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H240/attempt-001/H240-execution-output.zip` VersionId `CjARLxPHVCCrABD2.e.ME4IjFJuwJl5a`, SHA256 `fee1aa3d766cfa7207bbf330f70efb491e150fbfdc0013547ee38e66ac7beeff`. Exact-version bytes and all 265 members replay PASS. Manifest `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H240/attempt-001/H240-terminal-manifest.json` VersionId `G9WPjtBJhwXcH56HOv1suX0LX3nm5Jwc`. Raw roots retained: `/home/ubuntu/jope_plan/evidence/H240/res08-t0300` and `/home/ubuntu/jope-data/h240`. The campaign used one primary attempt, zero pauses and zero infrastructure resumes; all 3,000 primary updates were observed, plus three discarded smoke updates. No scientific reroll or serving promotion.

Protocol deviations: none. Admission RAM >=30 GiB/load5 <=100. Cooperative RAM pause <20 GiB/resume >=25 GiB sustained 30 s; hard <12 GiB sustained 30 s or <6 GiB instant; load pause >108 / resume <=100, RSS cap 4 GiB / disk floor 300 GB. At most 2 memory-only source-bound resumes, completed cells never rerolled; no interrupted work occurred in this wave. Compute stop 07:30Z / closeout 08:00Z. Local targeted 21 data-free tests/Ruff repeated at unchanged source bytes, JSON/schema/links/hash/diff checks and head-bound same-agent review. Actual jope-records coverage validation is repeated at the closeout head using exact manifests, workflows and tracked inventory; empty Rust target markers support metadata only. Hosted CI NOT_RUN/NOT_A_GATE and admin squash explicitly authorized by `hw_t0300_res08_h240_lowlr_rerun.md` and `hw_t0440_res08_h240_closeout.md`.

Frozen protocol: [PREREG.md](PREREG.md).
