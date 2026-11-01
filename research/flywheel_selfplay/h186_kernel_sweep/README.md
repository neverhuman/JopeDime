# H186 — kernel dose and shape sweep

**NO_ADVANCE.** One zero-update CPU attempt completed. All adapter/DOPE weights remained byte-identical; no serving adoption. [metrics.json](metrics.json) preserves the exact run result.

Hypothesis: H184’s attribution of training-free lift to the revealed-history kernel motivates a single-constant intervention on r882. [PREREG.md](PREREG.md) fixed seven arms, three fresh family-fold reshuffles, imported H177 pooled selection with Holm m=7, and a durable B lock before A/treatment DL-Val.

| Arm | Pooled B median gain [95% CI] | B Holm p | B admitted | Pooled A median gain [95% CI] | A positive seeds | A Elo [95% CI] | DL-Val gain [95% CI] | DL-Val Elo |
|---|---|---:|---|---|---:|---|---|---:|
| KX0.5 | -0.01087603 [-0.03335917, +0.01551559] | 1.000000 | False | -0.01087603 [-0.03335917, +0.01551559] | 0/3 | 1481.95 [1473.63, 1490.25] | -0.04365826 [-0.11754455, +0.00339136] | 1512.79 |
| KX1.5 | +0.00014726 [-0.02267514, +0.01844695] | 1.000000 | False | +0.00014726 [-0.02267514, +0.01844695] | 3/3 | 1499.44 [1492.71, 1506.35] | +0.01379260 [+0.00260255, +0.02842803] | 1504.11 |
| KX2.0 | +0.01291758 [-0.01747001, +0.04184311] | 1.000000 | False | +0.01291758 [-0.01747001, +0.04184311] | 3/3 | 1499.04 [1490.96, 1507.89] | +0.00450112 [-0.01364733, +0.02226604] | 1508.22 |
| TOPK4 | +0.00000000 [+0.00000000, +0.00000000] | 1.000000 | False | +0.00000000 [+0.00000000, +0.00000000] | 0/3 | 1500.00 [1500.00, 1500.00] | +0.00000000 [+0.00000000, +0.00000000] | 1500.00 |
| TOPK16 | +0.00000000 [+0.00000000, +0.00000000] | 1.000000 | False | +0.00000000 [+0.00000000, +0.00000000] | 0/3 | 1500.00 [1500.00, 1500.00] | +0.00000000 [+0.00000000, +0.00000000] | 1500.00 |
| W0X0.5 | -0.00836108 [-0.01885786, +0.00007147] | 1.000000 | False | -0.00836108 [-0.01885786, +0.00007147] | 0/3 | 1491.94 [1485.83, 1498.05] | -0.00033622 [-0.00236057, +0.00138109] | 1500.00 |
| W0X2 | +0.00036045 [-0.01587760, +0.01252317] | 1.000000 | False | +0.00036045 [-0.01587760, +0.01252317] | 3/3 | 1499.39 [1494.34, 1504.66] | -0.03512141 [-0.10307125, +0.00151767] | 1489.49 |

| Policy | TRAIN context utility | DL-Val context utility (val_reg) | TRAIN exact -2 picks | DL-Val exact -2 picks |
|---|---:|---:|---:|---:|
| R882 | 1.423073435 | 1.137794190 | 17 | 0 |
| KX0.5 | 1.408775555 | 1.092216883 | 18 | 1 |
| KX1.5 | 1.424386093 | 1.152193057 | 14 | 0 |
| KX2.0 | 1.430314162 | 1.142493158 | 12 | 0 |
| TOPK4 | 1.423073435 | 1.137794190 | 17 | 0 |
| TOPK16 | 1.423073435 | 1.137794190 | 17 | 0 |
| W0X0.5 | 1.416933216 | 1.137443195 | 17 | 0 |
| W0X2 | 1.424051606 | 1.101128983 | 15 | 1 |

Elo uses higher utility as a win, with R882 anchored at 1500 in the eight-policy Bradley–Terry panel. Win frequency and mean utility magnitude are different summaries: for example, KX0.5 has positive descriptive DL-Val Elo despite a negative mean utility change. Elo intervals are in metrics.json.

Context utilities/counts are identical across the three fold reshuffles; the table reports each seed’s value once. These context means differ from the equal-family effects above.

ADVANCE requires B admission under the unchanged strict margin/LCB rule, seven-arm Holm, and A median gain >0 with 3/3 seeds positive and median CI lower >0, plus the imported H178 receipt gate. Confirmed arms: `[]`. Full per-seed gains, CIs, utilities, Elo/CI and exact -2.0 pick counts for B/A/DL-Val are retained in metrics; gate-invalid counts are separate.

The 1,219 contexts /205 TRAIN families appear once per B and A pool per seed under distinct 123/41/41 family reshuffles. These are zero fitted models: seeds reshuffle folds, not weights. Identical seed panels: `{'B': True, 'A': True}`; pooled A exactly equals pooled B: `True`. Because zero-update seed panels are identical across all three fold reshuffles, `"3/3 seeds positive"` is not independent replication, and correlated A / identical policies do not add sample size. H178 empirically calibrated the original four-arm rule; H186 imports unchanged decision code but uses seven-arm configuration and has no separate empirical m=7 calibration. Historically exposed TRAIN and repeatedly exposed 91-context/19-family DL-Val do not constitute sealed qualification.

TOPK4/TOPK16 were exact structural no-ops at the inherited four reveals. No replacement arms or reveal-depth changes were made. KX scales only the kernel coefficient; W0 scales the complete null weight and propagates through the original attention-head input. Dose diagnostic: `{'multipliers': [0.5, 1, 1.5, 2], 'A_median_gain': [-0.010876029313319966, 0.0, 0.00014726179710585545, 0.012917576139057526], 'nondecreasing': True, 'nonincreasing': False}`. This descriptive ordering is not a separate inferential claim.

Exact native91 baseline passed first (1.1377941898164525 bit-exact). Default component sums matched H184/native delta throughout. Adapter SHA before=after `9c3ba940d9eb60aa2a39dd57c229c0952927ea1c2f22c372ae9764f43baa8669`; DOPE `c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814`; frozen-state hash `8d1070faf574ee1cd2a4e630887e1662b43189b3f89b04982143ef2704818aae`. Raw-score reconstruction verified 60,696 cost/choice/label checks, stage-lock bindings, TOPK no-op costs and the baseline. All seven selectors, 21 summary contrasts and Elo were re-executed with the same registered statistical implementation; this is not an independent statistical implementation. Resource limits passed, with zero replay model forwards or updates.

Run source `e4f19dd61173b00d0d82c9aeb4d00ee2628246ff`, prereg [PR #1551](https://github.com/neverhuman/JopeDime/pull/1551). Targeted tests/Ruff/records and exact-audit constructor smoke passed; the bounded review timeout was recorded at merge; rd_argon’s later exact-head PASS (review 5728731448, 15:02:22Z) is retained as additional evidence. CPU xbabe2 nice19, aggregate <=4 threads/RSS<6GB, CPU85/Mem16GiB/disk93.8%/free40GB terminal guards and 4h cap. No hosted full-CI/product qualification claim, GPU, protected data or live-loop writes. Python audit is application evidence, not an OS sandbox.

Versioned custody: `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H186/astra-t0730/manifest.json`, VersionId `hLhK4I_SRKehPZoLOkUg6JAUpYJWJCOo`, SHA256 `d18f1f1ab44d606a9a230fdbfaf62e27b313b1401faa312f9e30e4d61611cc86`. [custody.json](custody.json) binds full payload SHA256 and 1545 member hashes recovered from exact S3 versions. Numerical inputs, raw scores/choices, stage locks, manifests and receipts are preserved; source stays Git-pinned. Shared inputs remain untouched; only owned verified bulk is eligible for cleanup after closeout merge.

Next: retain r882. No arm was admitted and no eligible HWM candidate or serving promotion is claimed. KX2.0 had the largest pooled gain (+0.012918), but its 95% interval crossed zero; KX1.5’s positive DL-Val result is descriptive and cannot override failed B admission. A follow-up, if separately authorized, should distinguish kernel-dose ranking benefits from magnitude-sensitive mean gains. H186 makes no serving change; further grid or reveal-depth work requires a new wave.
