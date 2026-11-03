# H245: nested blend weight + zgap — COMPLETE / NO_ADVANCE

Hypothesis: FIT-only joint choice of lambda and zgap threshold lifts the confirmed H217 gate beyond the DL-Val +0.002 margin. Fresh H245 rerun of incomplete H218; no prior positive result is imputed.

Preregistered in PR #2091, execution source `796f737ecbfeb8e98cfdc32848cc7c38fe0ee0cc`. CPU xbabe2 cores 40–43, nice19, HistGBM/OMP <=4, other math pools1. Read-only r882 COPY, frozen DOPE c51a56d0, **zero neural optimizer updates**. Seeds245101–245103; fold seeds245141–245143; one 205-family partition per seed, max historical best-match Jaccard **0.196213757** across26 comparable H203–H233 waves. Five waves without comparable manifests are inventoried explicitly.

Frozen H217/H218 recipe: 192 sampled programs per FIT context; 45 HistGBM fits shared by lambda .125/.25/.5; FIT inner three-fold family OOF calibration; max FIT equal-family native gain with FIT harm<=5%, ties to smaller tau then lambda, otherwise r882. GATED_NEST is sole primary; GATED_025 fixed-lambda anchor and GATED_NEST_SHUF are report-only. SHUF copies primary lambda/tau and permutes only outcome-free zgap, preserving gate open counts. Ungated components are report-only. Seeds share sample245140 and deterministic r882 costs; partitions and HistGBM random states differ. These are paired repetitions on the same corpus, not independent dataset replications. All15 cell choices were fsync-locked before trained A/DL-Val assessment; the unused pre-lock H218 B diagnostic was omitted.

Native91 PASS first: all91 utilities exact, mean **1.1377941898164525**. **45/45 fits and15/15 cell choices completed**, with no smoke refits, restarts or rerolls. DOPE was checked before/after baseline inference and the starting checkpoint remained byte-identical after all fits. No GPU initialized.

The sole primary is the median across three seeds of the **DL-Val91 context-mean gain**, averaging five policy cells per seed. Shared 19-family bootstrap draws retain context weighting. Pooled-A uses 205 equally weighted families. The harm gate retains H217/H218 pooled-A harm: a family is harmed if its three-seed mean delta is below −0.01; DL-Val harm is also reported.

| Policy vs r882 | DL-Val median gain [family CI95] | Positive seeds | Holm m1 p | DL-Val harm |
|---|---|---:|---:|---:|
| GATED_025 | +0.000382198 [-0.000701084, +0.002051225] | 3/3 | 0.30716928 | 0/19 |
| GATED_NEST | +0.000516967 [-0.001297284, +0.003040207] | 2/3 | 0.24397560 | 1/19 |
| GATED_NEST_SHUF | -0.010094603 [-0.026276078, +0.007476697] | 0/3 | 0.80471953 | 3/19 |
| GBM_CAL | -0.214531466 [-0.373475051, -0.083798318] | 0/3 | 1.00000000 | 12/19 |
| UNGATED_025 | -0.002830050 [-0.018756824, +0.017571846] | 1/3 | 0.54684532 | 3/19 |
| UNGATED_125 | +0.003234163 [-0.002068390, +0.008627970] | 3/3 | 0.11618838 | 2/19 |
| UNGATED_500 | -0.055999358 [-0.096787598, -0.016907475] | 0/3 | 0.99880012 | 11/19 |

Only GATED_NEST enters the decision/multiplicity family; other p-values are descriptive. The registered gate requires DL-Val gain>+0.002, LCB95>0, Holm p<.05, at least2/3 positive seeds and pooled-A harm<=10%. Ties reject. **Verdict: NO_ADVANCE**.

| Policy vs r882 | Pooled-A median equal-family gain [CI95] | Positive seeds | Harm |
|---|---|---:|---:|
| GATED_025 | +0.009616081 [+0.000265264, +0.030273766] | 3/3 | 5/205 (2.439%) |
| GATED_NEST | +0.014784896 [+0.002918730, +0.033468490] | 3/3 | 8/205 (3.902%) |
| GATED_NEST_SHUF | +0.031297679 [+0.007615960, +0.063230516] | 3/3 | 39/205 (19.024%) |
| GBM_CAL | -0.122578602 [-0.216433115, -0.032328408] | 0/3 | 118/205 (57.561%) |
| UNGATED_025 | +0.040748857 [+0.009043102, +0.086203077] | 3/3 | 57/205 (27.805%) |
| UNGATED_125 | +0.037213958 [+0.012645974, +0.064739562] | 3/3 | 32/205 (15.610%) |
| UNGATED_500 | +0.046453742 [-0.008580072, +0.113827175] | 3/3 | 79/205 (38.537%) |

| GATED_NEST minus GATED_025 | Median gain | Family CI95 |
|---|---:|---|
| dlval | -0.000330281 | [-0.001940243, +0.001563156] |
| pooled_a | +0.005719882 | [-0.000397442, +0.009656742] |

| Policy | Pooled-A Elo [CI95] | DL-Val Elo [CI95] |
|---|---|---|
| GATED_025 | 1502.94770 [1500.70567, 1506.98701] | 1503.65351 [1481.71629, 1529.29620] |
| GATED_NEST | 1506.48830 [1501.98619, 1511.35980] | 1510.96375 [1485.37789, 1544.07538] |
| GATED_NEST_SHUF | 1513.73625 [1504.78638, 1522.84249] | 1507.30782 [1467.02097, 1544.07538] |
| GBM_CAL | 1479.79939 [1454.53832, 1500.74470] | 1421.99440 [1339.73566, 1503.65351] |
| UNGATED_025 | 1517.10394 [1505.35188, 1529.10649] | 1500.00000 [1452.20699, 1544.07538] |
| UNGATED_125 | 1512.42963 [1504.14114, 1521.91865] | 1518.28371 [1478.05062, 1544.07538] |
| UNGATED_500 | 1511.88042 [1497.82400, 1528.33898] | 1463.33074 [1394.58579, 1532.97903] |

Elo anchors r882 at1500, with1,000 paired family draws; utility uses10,000. The original run keeps plateau_elo null because its scientific input closure supplied only the scalar. The requested comparison below is separate publication postprocessing from a pre-existing H201 archived vector: exact same evaluation identity/context order, all91 archived r882 utilities matching the frozen reference, and archived r3175 mean1.1389572403748425. The reference was pinned before H245 trained-policy outcomes, after preregistration; no fit, gate, primary test or source changed. H226 CPU replay still fails archive parity (1.1366829 vs1.1389572) and is excluded. These descriptive archived-only Elo intervals do not qualify CPU r3175 or remove historical winner-selection bias. No new model/native evaluation occurred.

| Policy | Median DL-Val context utility | Minus plateau1.1389572 |
|---|---:|---:|
| GATED_025 | 1.1381763874 | -0.0007808126 |
| GATED_NEST | 1.1383111569 | -0.0006460431 |
| GATED_NEST_SHUF | 1.1276995873 | -0.0112576127 |
| GBM_CAL | 0.9232627242 | -0.2156944758 |
| R882 | 1.1377941898 | -0.0011630102 |
| UNGATED_025 | 1.1349641397 | -0.0039930603 |
| UNGATED_125 | 1.1410283524 | +0.0020711524 |
| UNGATED_500 | 1.0817948314 | -0.0571623686 |

| Policy | DL-Val Elo lift vs archived r3175 [family CI95] |
|---|---|
| R882 | +0.000000 [-10.963751, +10.963751] |
| GATED_NEST | +3.653506 [-18.283715, +36.669262] |
| GATED_NEST_SHUF | +10.963751 [-40.367755, +47.793015] |
| GATED_025 | +3.653506 [-21.949381, +29.296204] |
| UNGATED_125 | +10.963751 [-21.949381, +47.793015] |
| UNGATED_025 | -3.653506 [-59.109200, +44.075379] |
| UNGATED_500 | -44.075379 [-121.646477, +25.711840] |
| GBM_CAL | -78.005605 [-160.264344, +3.653506] |

These pairwise lifts use archived r3175=1500; the source-run Elo table keeps r882=1500. Reference hash and exact HWM checks are retained in metrics and custody.

| Seed/fold | Primary lambda | tau | FIT gain | FIT harm |
|---|---:|---:|---:|---:|
| 245101/0 | 0.25 | 0.005797527918860305 | +0.03190442 | 4.065% |
| 245101/1 | 0.125 | 0.008407322349242788 | +0.01233702 | 3.252% |
| 245101/2 | 0.125 | 0.0008562322964137444 | +0.00066495 | 0.813% |
| 245101/3 | 0.5 | 0.03028218706946944 | +0.02914266 | 4.878% |
| 245101/4 | 0.125 | 0.0037781275784435076 | +0.01680518 | 3.252% |
| 245102/0 | 0.125 | 0.004351380303522946 | +0.01954650 | 3.252% |
| 245102/1 | 0.5 | 0.014800886594332884 | +0.02328036 | 4.878% |
| 245102/2 | 0.25 | 0.006944265129563541 | +0.00893738 | 4.065% |
| 245102/3 | 0.25 | 0.002165282849124113 | +0.00240482 | 2.439% |
| 245102/4 | 0.5 | 0.005644990242754787 | +0.01742127 | 4.065% |
| 245103/0 | 0.25 | 0.0034747720477942084 | +0.00344023 | 1.626% |
| 245103/1 | 0.125 | 0.0031744476625789296 | +0.01425191 | 4.878% |
| 245103/2 | 0.25 | 0.00242967894836131 | +0.01397531 | 1.626% |
| 245103/3 | 0.25 | 0.026008174041944776 | +0.06482610 | 4.878% |
| 245103/4 | 0.125 | 0.009068496781471444 | +0.01214401 | 4.065% |

Lambda counts: `{"GATED_025": {"0.125": 0, "0.25": 11, "0.5": 0, "R882": 4}, "GATED_NEST": {"0.125": 6, "0.25": 6, "0.5": 3, "R882": 0}, "GATED_NEST_SHUF": {"0.125": 6, "0.25": 6, "0.5": 3, "R882": 0}}`. Per-seed A gate rates, all15 DL-Val gate counts and all arm FIT diagnostics are in compact metrics. Placebo matches primary gate counts on FIT, pooled A and every DL-Val cell.

Red-team replay PASS: all 45 fit hashes/prediction coverage/exclusion roles; all 15 selection locks; matched-rate placebo; complete finite stored utilities; independent family-bootstrap/median/harm/p-value/paired-anchor replay and Elo arithmetic; actual disjoint families/no meta-test; frozen source/DOPE/checkpoint; resource transitions. Worker exit 0, sampled peak aggregate RSS 3568353280 bytes, worker maximum RSS from rusage 3606687744 bytes, minimum available RAM 34121924608 bytes, minimum disk 674336075776 bytes; 0 guard events. Host CPU percentage was not sampled by the supervisor and remains null.

Next step: retain the incumbent; do not promote the nested treatment from this wave. No follow-on compute is authorized here.

Output/source/copy custody: `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H245/run-001/H245-execution-output.zip` VersionId `fe8J2HF7Dd6ZGID5D_ghjYHiPaRoQ2Tc`, SHA256 `50273103b4c2ffff1f006dbee0acf547ab7aee343ff62b6d26ea5c89db382be9`. Archive and manifest exact-version bytes plus all 163 member hashes/sizes replay PASS. Manifest `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H245/run-001/H245-terminal-manifest-v2.json` VersionId `Bo9fs0M56Hb.XZhbONW_d5m7K0SdTWuz`. Raw roots retained: `/home/ubuntu/jope_plan/evidence/H245/res08-t0505`, `/home/ubuntu/jope-data/h245`.

Targeted27 data-free tests/Ruff/format, JSON/schema/hash/link/diff and actual jope-records static coverage validation run at the final publication head; same-agent head-bound review is not independent/human approval. Full hosted CI NOT_RUN/NOT_A_GATE and guarded admin squash are explicit owner H245 authority. No AUC/speed verdict, serving promotion, new NN optimization, other-job controls or PR2071 mutation. Fixed compute10:00Z / closeout10:45Z; H239A1 RAM/load cooperative guard and RSS4GiB/disk300GB limits are retained. Protocol [PREREG.md](PREREG.md).
