# H368 — NL15 ranker max_bin 63 on xbabe1

**MB63_NOT_ADVANCED**. [PREREG.md](PREREG.md) published nonforce v1/next **1fd3b9e296548ba035e50db3404cb9166b35d6d4**, 2026-11-05T13:37:22.684087+00:00, before first actual fit 2026-11-05T13:39:28.429336+00:00.

## Treatment and proof

Copy committed H362 e976935dc7a4dbdbaa6047bc659966c7c18e0480 design/transport, with H352 02dfee2f7e0b8543324494dd4f99d074cd6b93b6 NL15/depth3 science. Sole ranker native delta: max_bin255→63 in topk.fit_ranker and bagged.fit_ranker_n, every inner/final/eight bag ranker. Added keyword removal restores both complete H352 ASTs. Num_leaves15/max_depth3 remain fixed (at most8 actual leaves). Original LambdaRank/ndcg/sigmoid1/truncation8/eval_at1,8, labels/default gains, L2=1/implicitL1=0/minleaf20, LR/rounds/grid/seeds/features/ties1e-4/sentinel−2 stay unchanged. Veto binary GBDT L1=1/L2=2 unchanged; full acceptance requires all108 final veto SHAs identical to BASE.

Before fits, recovered H352972saved rankers and29484actual fit rows prove native max_bin255/depth3/leaves15. Every completed native fit checks all constructor/native fields and full H352 constants/seed-derived fields with only ranker max_bin changed. Every actual Dataset records raw numpy input, fresh handle, same-fit reference, constructor/effective max_bin63(rankers)/255(veto), and per-feature native bin counts. Prebuilt/ignored bins fail INCOMPLETE_SPEC_MISMATCH. Dataset fixtures construct bins without optimizer updates. [actual_tree_summary.json](actual_tree_summary.json) retains ordered-tree audit summaries; complete arrays/native/Dataset records remain in S3.

Same P0/P1/P2 × roots30001–30003 ×4outer folds=36tasks,1219TRAINcontexts/205families. Parent has metadata only; fit hooks admit only outer-TRAIN feature ranges and deny comparator outcomes/native copies. All36locks/directories fsynced and native/input/veto/custody reviews pass before any outer packet access. Outer score exactly once. Historical TRAIN exposure remains; no independent fresh confirmation.

## Inputs and local verification

Fresh exact-Version whole/member H316/H315 proof on xbabe2, then every copied SHA and BASE9panels/36pick+lockSHA/3exactmedians on xbabe1. Fresh H352manifest/27parts whole+2691memberSHA, then all copied SHA and NL15ninepanels/36locks+scores/rawfamilyarray regression on xbabe1. Current selected/all1219TRAINpools/r882cost/fourTRAINfeatureprefix SHA exact. Owned cache contains only consumed TRAIN features; protected mixed-cache tails unread. No BASE/NL15 refits or new predictions. See [regression_receipt.json](regression_receipt.json), [nl15_regression_receipt.json](nl15_regression_receipt.json), [run_receipt.json](run_receipt.json).

84local tests passed on both hosts,0failed/skipped/optimizerfits, including full reverseAST, original parameter stream, Dataset bins/prebuilt refusal, complete native/default/seed drift, comparator-distinguishing report fixture, primary decision boundaries, outer-before-lock refusal, veto identity, corrected resource windows/forecasts and credential-free transport. Frozen44source SHA exact while fitting. Thirteen source-equivalence groups preserve all undeclared H362 AST/function bodies. Same implementing agent performed substantive source/evidence review; no independent/hosted-CI claim. Actual audit: `{"status": "PASS", "full_treatment_qualification": "PASS", "frozen_source_exact": true, "all_completed_fit_Datasets_audited": true, "Dataset_role_counts": {"LGBMClassifier": 7020, "LGBMRanker": 29484}, "Dataset_effective_max_bin_counts": {"LGBMClassifier:255": 7020, "LGBMRanker:63": 29484}, "constructor_events": 36504, "constructor_roles": {"LGBMClassifier": 7020, "LGBMRanker": 29484}, "completed_native_fit_records": 36504, "native_fit_roles": {"LGBMClassifier": 7020, "LGBMRanker": 29484}, "total_completed_optimizer_calls": 36504, "total_completed_optimizer_calls_note": "All mandatory completed native optimizer records verified", "completed_fold_root_locks": 36, "outer_score_markers": 36, "native_model_reviews": 36, "reviewed_saved_models": 14040, "currently_raw_native_files": 1080, "final_veto_SHA_matches": 108, "first_actual_fit_started_utc": "2026-11-05T13:39:28.429336+00:00", "all_observed_fits_after_prereg": true, "truncated_interrupted_records": []}`; qualification **PASS**. Actual unit exit: `{"utc": "2026-11-05T14:24:19.311535+00:00", "EXIT_CODE": "exited", "EXIT_STATUS": "0", "SERVICE_RESULT": "success", "INVOCATION_ID": "f945159e8efa402fb9deeacb8c8d0e1f"}`.

## Outcome

**reference = r882 slot0 (DL-Train outer); r3175 not measured on TRAIN**.

| Arm | Median Elo | Median family mean gain | Literal family median | Median harmed |
|---|---:|---:|---:|---:|
| BASE | 1725.655944 | +0.048214800 | +0.003624258 | 9.2683% |
| NL15 | 1734.469531 | +0.058404671 | +0.007409439 | 9.2683% |
| MB63 | 1712.244509 | +0.043934556 | +0.004387874 | 10.7317% |

PRIMARY vs archived H352 NL15: drop10 median **-0.014199672355**, CI95 [-0.03236491701228777, -0.000964442913913776], one-sided LCB95 **-0.028220038122**.

| Root | Paired median | CI95 | LCB95 |
|---:|---:|---|---:|
| 30001 | -0.006400480 | [-0.014714823901049584, 0.0040797769771987215] | -0.013132906 |
| 30002 | -0.014199672 | [-0.03780519033048953, 0.0027756604406173786] | -0.032572951 |
| 30003 | -0.016805537 | [-0.03900336819906343, -0.000805031748088563] | -0.034463382 |

Secondary vs BASE: drop10 median **-0.005041885470**, CI95 [-0.02069631637407768, 0.006008052872911109], one-sided LCB95 **-0.016471100311**.

| Root | Paired median | CI95 | LCB95 |
|---:|---:|---|---:|
| 30001 | +0.006125877 | [-0.0070449634408487165, 0.014753879489095107] | -0.005256604 |
| 30002 | +0.000625410 | [-0.02372075400528512, 0.009658238074515428] | -0.019599538 |
| 30003 | -0.005204065 | [-0.029756999344109476, 0.0008007077712433025] | -0.024931042 |

Four primary gates: `{"paired_drop_top10_at_least_002": false, "gain_retention": false, "harmed_nonincrease": false, "positive_on_two_seeds": false}`. Secondary BASE is descriptive and adds no gate.

| Partition | Root | Arm | Family mean | Family median | Drop10 | W/T/L | Harm | Invalid | Switches | Elo | Elo CI95 |
|---|---:|---|---:|---:|---:|---|---:|---:|---:|---:|---|
| P0 | 30001 | BASE | +0.041289 | +0.000000 | +0.013174 | 88/93/24 | 6.341% | 13 | 219 | 1725.66 | [1652.447886376557, 1812.9700690664945] |
| P0 | 30002 | BASE | +0.064316 | +0.001588 | +0.008633 | 108/67/30 | 8.780% | 14 | 300 | 1722.48 | [1656.7079288809393, 1799.2122929353282] |
| P0 | 30003 | BASE | +0.072258 | +0.006984 | +0.016621 | 122/50/33 | 9.268% | 11 | 320 | 1727.10 | [1665.6454632422033, 1798.7037721418976] |
| P0 | 30001 | NL15 | +0.050668 | +0.004840 | +0.011081 | 115/60/30 | 7.317% | 13 | 257 | 1733.39 | [1668.6813747437993, 1811.1881388441423] |
| P0 | 30002 | NL15 | +0.047569 | +0.000000 | +0.012995 | 94/82/29 | 9.268% | 16 | 227 | 1704.25 | [1637.3491989417423, 1784.415165468581] |
| P0 | 30003 | NL15 | +0.073628 | +0.007409 | +0.028384 | 118/62/25 | 7.317% | 12 | 285 | 1769.52 | [1700.7429336578732, 1857.2955733217061] |
| P0 | 30001 | MB63 | +0.052285 | +0.002651 | +0.004680 | 113/54/38 | 11.220% | 13 | 282 | 1689.29 | [1627.3432180210623, 1758.4574394700142] |
| P0 | 30002 | MB63 | +0.054073 | +0.008889 | +0.009258 | 124/42/39 | 13.171% | 15 | 347 | 1700.91 | [1641.614973779568, 1768.006852866913] |
| P0 | 30003 | MB63 | +0.048067 | +0.004388 | +0.011579 | 117/55/33 | 9.756% | 13 | 270 | 1719.83 | [1656.8212358859475, 1792.0924860705213] |
| P1 | 30001 | BASE | +0.048215 | +0.003000 | +0.010198 | 112/57/36 | 11.220% | 17 | 327 | 1697.13 | [1636.2756543932148, 1768.502029971351] |
| P1 | 30002 | BASE | +0.048658 | +0.003624 | +0.014063 | 116/51/38 | 10.244% | 15 | 275 | 1693.84 | [1633.762844611895, 1761.9483371747197] |
| P1 | 30003 | BASE | +0.038294 | +0.003253 | +0.012702 | 112/64/29 | 8.293% | 16 | 285 | 1734.68 | [1668.0582230695036, 1812.7033860359509] |
| P1 | 30001 | NL15 | +0.076944 | +0.016096 | +0.029863 | 141/32/32 | 10.244% | 12 | 431 | 1757.59 | [1696.1259403623662, 1831.3772390386723] |
| P1 | 30002 | NL15 | +0.047437 | +0.003867 | +0.020435 | 112/64/29 | 6.829% | 12 | 288 | 1734.68 | [1669.1805216399953, 1812.7290314140926] |
| P1 | 30003 | NL15 | +0.073295 | +0.009586 | +0.022337 | 129/41/35 | 11.707% | 13 | 369 | 1726.57 | [1664.4745655416996, 1798.6980188537896] |
| P1 | 30001 | MB63 | +0.043935 | +0.000000 | +0.016323 | 99/70/36 | 10.732% | 13 | 274 | 1675.70 | [1612.8402042051498, 1747.5845618582225] |
| P1 | 30002 | MB63 | +0.030505 | +0.004759 | +0.000626 | 112/60/33 | 8.293% | 14 | 283 | 1712.24 | [1648.930551589452, 1785.2258189766305] |
| P1 | 30003 | MB63 | +0.031693 | +0.008895 | +0.007498 | 123/46/36 | 11.707% | 14 | 318 | 1713.41 | [1653.90107779385, 1785.0038734011312] |
| P2 | 30001 | BASE | +0.043601 | +0.006326 | +0.010678 | 122/48/35 | 10.732% | 16 | 315 | 1716.88 | [1656.1058566632537, 1787.6763019641999] |
| P2 | 30002 | BASE | +0.031840 | +0.003624 | +0.008774 | 112/65/28 | 7.805% | 17 | 292 | 1740.78 | [1674.769204976847, 1820.0941275524494] |
| P2 | 30003 | BASE | +0.054721 | +0.008145 | +0.023605 | 129/41/35 | 10.732% | 15 | 335 | 1726.57 | [1665.9655865631883, 1796.0904217803327] |
| P2 | 30001 | NL15 | +0.047241 | +0.003000 | +0.008960 | 110/62/33 | 10.244% | 14 | 291 | 1709.11 | [1647.1518552896166, 1782.6547018715517] |
| P2 | 30002 | NL15 | +0.063382 | +0.011913 | +0.023902 | 136/36/33 | 9.268% | 12 | 360 | 1745.97 | [1684.26328616166, 1818.2987002068173] |
| P2 | 30003 | NL15 | +0.058405 | +0.010809 | +0.018755 | 135/35/35 | 12.195% | 17 | 389 | 1734.47 | [1674.0214585562887, 1805.3137043261177] |
| P2 | 30001 | MB63 | +0.053027 | +0.002436 | +0.016969 | 110/67/28 | 6.829% | 12 | 227 | 1737.65 | [1669.8295274538928, 1816.6268714833425] |
| P2 | 30002 | MB63 | +0.040847 | +0.011715 | +0.009703 | 135/32/38 | 11.707% | 15 | 373 | 1720.19 | [1662.1452281833956, 1789.1958579823977] |
| P2 | 30003 | MB63 | +0.023242 | +0.000755 | +0.000453 | 105/68/32 | 9.268% | 16 | 243 | 1706.38 | [1642.5883016493015, 1782.1919594264232] |

Config agreement MB63/BASE 7/36; MB63/NL15 13/36. All27cells preserve family mean/median CIs, drop1/5/10, LOFO minimum, W/T/L, invalids, switches, EloCI and pick/lock hashes. Aggregate intervals and seed Jaccards are in metrics.


[metrics.json](metrics.json) is below50KB with exact DOPE frozen/unloaded block and SHA-linked full receipts. Whole-family paired bootstrap10000/RNG368191; ordinary302150 unchanged. Shared family draws across arms/cells, each arm drops its own best10; median9cell differences and each root median3partitions/CIs. ADVANCE requires primary-vs-NL15 drop10 median≥+.002 AND LCB95>0, median full gain≥90%NL15, harmed≤NL15, and≥2/3positive root medians. All four required; ties never advance. Complete failure is MB63_NOT_ADVANCED; incomplete UNKNOWN. Sequential multiplicity H348–H368 uncorrected; any TRAIN advance needs separately authorized one-shot dev91 confirm outside H368. No speed or AUC verdict.

## Resources and custody

xbabe1 CPU-only, nice19/math1/MemoryMax6GiB/Swap0/CPUQuota800%. One5sec bothphysical+SMT85%-idle gate, first≤8/min4 from24..35+sibs88..99 with same-window hostCPU≤70. Actual admission `{"eligible_cores": [25, 26, 27, 28, 29, 30, 31, 32], "both_smt_idle": {"24": 23.65, "25": 99.4, "26": 99.4, "27": 99.2, "28": 98.4, "29": 99.6, "30": 97.8, "31": 99.0, "32": 99.4, "33": 99.4, "34": 99.2, "35": 99.2}, "host_at_probe": {"cpu_percent": 3.6668748048704347, "interval_seconds": 5.013536673039198, "aggregate_ticks_before": [125526518588, 108921918806], "aggregate_ticks_after": [125526582648, 108921980517], "method": "Same single mpstat interval bracketed by aggregate /proc/stat; no additional per-core interval", "pool_selection": {"chosen_pool": "H368", "passing_cores": [25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35], "selection_rule": "First<=8 physical+SMT pairs85%idle in assigned24..35, one interval only"}}, "host_cpu_maximum": 70, "sample_seconds": 5, "minimum_required": 4, "assigned_cores": [24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35], "maximum_taken": 8, "ordering": "first at most8 passing pairs within one chosen pool", "capacity_retries": 0}`. H352CPU26189.516914/.9eff/1.15safety/+300reserve forecasts `{"initial": {"wave": "H368", "status": "PREFIT_FORECAST", "forecast_start": "2026-11-05T13:39:15.154028+00:00", "workers": 8, "eligible_cores": [25, 26, 27, 28, 29, 30, 31, 32], "mandatory_tasks": 36, "reference_wave": "H352 actualCPU if available; fallback H34729770.57", "forecast_reference_cpu_seconds": 26189.516914, "depth_cost_multiplier_assumption": 1.0, "parallel_efficiency": 0.9, "safety_factor": 1.15, "score_reserve_seconds": 300, "forecast_seconds": 4483.047840430555, "forecast_end": "2026-11-05T14:53:58.201868+00:00", "compute_stop": "2026-11-05T16:30:00+00:00", "close_stop": "2026-11-05T16:45:00Z", "launch_full_36": true, "campaign_fits": 0, "synthetic_fits": 0, "BASE_refits": 0, "outer_scores": 0, "capacity_retries": 0, "caveat": "MB63 cost is unmeasured; no acceleration assumed, partial design, borrowed cores, deadline reset or repeated capacity sampling.", "capacity": {"eligible_cores": [25, 26, 27, 28, 29, 30, 31, 32], "both_smt_idle": {"24": 23.65, "25": 99.4, "26": 99.4, "27": 99.2, "28": 98.4, "29": 99.6, "30": 97.8, "31": 99.0, "32": 99.4, "33": 99.4, "34": 99.2, "35": 99.2}, "host_at_probe": {"cpu_percent": 3.6668748048704347, "interval_seconds": 5.013536673039198, "aggregate_ticks_before": [125526518588, 108921918806], "aggregate_ticks_after": [125526582648, 108921980517], "method": "Same single mpstat interval bracketed by aggregate /proc/stat; no additional per-core interval", "pool_selection": {"chosen_pool": "H368", "passing_cores": [25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35], "selection_rule": "First<=8 physical+SMT pairs85%idle in assigned24..35, one interval only"}}, "host_cpu_maximum": 70, "sample_seconds": 5, "minimum_required": 4, "assigned_cores": [24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35], "maximum_taken": 8, "ordering": "first at most8 passing pairs within one chosen pool", "capacity_retries": 0}}, "after_first_completed_task": {"wave": "H368", "stage": "FIRST_COMPLETED_CELL", "forecast_start": "2026-11-05T13:47:33.845423+00:00", "first_completed_cell": {"task": ["P0", 0, 30002], "chosen": {"tau": 0.05, "k": 5, "a": 2, "inner_gain": 0.03086442304798908, "inner_switches": 114}, "wall_seconds": 496.263498544693, "cpu_seconds": 489.955260607}, "workers": 8, "completed_tasks": 1, "remaining_tasks": 35, "baseline_reference_cpu_seconds": 26189.516914, "per_task_cpu_reference_seconds": 727.4865809444444, "parallel_efficiency": 0.9, "safety_factor": 1.15, "score_reserve_seconds": 300, "forecast_seconds": 4366.852067085261, "forecast_end": "2026-11-05T15:00:20.697490+00:00", "compute_stop": "2026-11-05T16:30:00+00:00", "launch_full_36": true, "outer_scores": 0, "note": "Conservative remaining35CPU tasks; ongoing work credit0; reference=max(pinnedCPU/36,first completed cell actualCPU). No second capacity probe."}}`. Fixed16:30compute/16:45close Nov5; no reprobe/restart/reset/cutcells/partialscore. Correct forecast-miss unit h368-x1-run.service frozen before launch.

Corrected guard: immediate MemAvail<20GiB/rootfree<80GiB/scratch>2GB, hostCPU>85% overfull60sec, load>.85×hostonline128=108.8 on3consecutive60sec samples, or fixeddeadline. Record tick pairs/windows/streak; signal only own units. Actual resources `{"samples": 536, "full_60s_windows": 44, "max_cpu_5s_telemetry": 61.58645276292335, "max_cpu_60s": 18.148491020181446, "max_load1": 20.0791015625, "host_nproc": [128], "load_thresholds": [108.8], "max_consecutive_high_load_60s": 0, "min_mem_available_bytes": 39550398464, "min_root_free_bytes": 96400498688, "max_scratch_bytes": 1283511696, "nice": 19, "numeric_threads": 1, "MemoryMax": 6442450944, "MemorySwapMax": 0, "CPUQuota": "800%", "guard": "CPU>85 averaged60s OR load>.85*hostonlineNCPU on3consecutive60s samples OR MemAvail20GiB/root80GiB/scratch2GB OR fixeddeadline"}`. All AWS/S3 operations xbabe2 only; no AWS credentials/client onxbabe1. Own courier/archive use nice19/core62/CPUQuota100%/6GiB/Swap0 with conservative CPU70/loadfloor(.85*hostN)/Mem20GiB own-I/O backoff. No foreign process, stopped writer, marker or run-dir changes.

Create-only S3 prefix experiments/rd_breakthrough_20261030/H368/solhwv-x1-20261105T133000Z/. After input regression,2232historical native copies/576553824bytes EACHhost were archived exact-Version whole+everymember verified and rehashed before pruning. Comparator arrays/locks/scores/native-referenceJSON/TRAINinputs remain; no treatment data removed by this step. Inner archive requests are SHA-pinned to the xbabe2 courier; returned exactVersion whole/member proof required before raw/duplicate deletion. All stopped outputs copied back and rehashed, final whole/everymember proof before both own scratch exactsets/allSHA rechecked and removed. [s3_manifest.json](s3_manifest.json) pins final/reference/innerobjects; full models/native/Dataset/tree/locks/score/input/probe/forecast/guard/exit/tests/code/control evidence preserved, no models Git. Operations snapshots in operations_manifest.json.

Nonforce fresh-parent v1/next only, oneH368row each index; all other rows and canonical foreign HEAD/index/files preserved. CI_FREEZE+BATCH_WINDOW/noPR/main/fastlane/hostedCI/FINDINGS/newlane. DOPEc51a56d0 frozen/unloaded; no GPU/qualification.lock/xbabe0/xbabe3/foreignjobs/dev91/DL-Test/sealed/holdout/meta-test/freshcohort. H362 and all prior closed waves retained. No further H368 fit/score/probe/dev91 action after release.
