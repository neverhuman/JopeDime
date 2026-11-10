# H390: NL15 ranker cat_l2 20 rerun

**COMPLETE — CATL2_NOT_ADVANCED.** Primary comparator: frozen H352 NL15; secondary comparator: BASE.

The CATL2 hypothesis doubles ranker categorical L2 to damp categorical splits chasing context-like noise. Fresh archived H352 native settings verified cat_l2=10.0; H390 adds only cat_l2=20.0 in the two original H352 ranker constructors, covering inner fits, final refits and all eight bag rankers. Removing that sole added keyword restores both complete committed H352 function ASTs exactly. No new seed keyword or change to categorical-feature declarations.

All other settings and protocols remain H352: leaves15/depth3, feature_fraction.8, bagging_fraction.8/freq1, path_smooth0, L1zero/L2one, min_child20, min_split_gain0, Hessian.001, sigmoid1, original LR grid(.03,.05,.10), rounds(50,100,200), main200, labels/default label_gain/groups/features/sampling/logical seed expressions, original18configuration grid, 1e-4 tie/switch margin and sentinel−2. Veto GBDT is unchanged; all108final veto SHAs match frozen BASE and NL15.

H388 remains CLOSED INCOMPLETE_CPU_ADMISSION / NOT_EVALUATED at `58d3aa8b96be977b13079fffecb0ac7af65bc2b1`; H380 remains CLOSED BFQ5_NOT_ADVANCED at `b4991e42ca647b83f02225b874fb2a61c32ab628`. This separate owner-authorized H390 uses committed H388 science `c64dc60ea68fc44fd5589234ed4e8d84188e7e91` / manifest `cfd671aa00c25392a46f7580f4678198fef7b87d8840038e3d99c8da626a3780`, adapting only wave identities/paths, RNG390191, fixed window and authorized core union. The frozen H380 transport has bagging_freq1/path_smooth0 restored. Prior closes are preserved without relabeling.

[PREREG.md](PREREG.md), all 35 source files, including the manifest, were published non-force in `4eae61f8e1ae7c6ff8bd4d68e80ac20803db822f` at 2026-11-05T19:04:23.460199+00:00, before the first treatment fit 2026-11-05T19:04:49.971303+00:00. Source remained frozen. Full-source hash inventory and the exact committed-H388 AST equivalence proof are retained in custody.

TRAIN only:1219contexts/205families, three partitions P0/P1/P2 ×roots30001/30002/30003 ×four family-grouped outer folds =36tasks/nine evaluation cells. Fresh exact-Version GET/whole-object/every-member regression reproduced BASE9panels/36fold-root pick+lock hashes and exact medians .0643163334285782/.04821480030677507/.04360142642936269. H352 recovery verified all27parts/2691members,9panels,36locks+COMPLETEscores/rawfamilyarrays,972native final rankers+108vetos and all36504native fit headers; nativecat10/freq1/path0. No BASE/NL15 refits, calibration or new baseline model predictions.

| Arm | Median cell mean gain [95% CI] | Literal family median [95% CI] | Harmed share | Median Elo |
|---|---|---|---|---|
| BASE | +0.048215 [+0.025183,+0.075951] | +0.003624 [+0.000296,+0.007722] | 9.27% | 1725.66 |
| NL15 | +0.058405 [+0.034290,+0.092664] | +0.007409 [+0.002351,+0.011715] | 9.27% | 1734.47 |
| CATL2 | +0.058405 [+0.034290,+0.092664] | +0.007409 [+0.002351,+0.011715] | 9.27% | 1734.47 |

`reference = r882 slot0 (DL-Train outer); r3175 not measured on TRAIN`

PRIMARY CATL2−NL15 paired drop10 **+0.000000000**,95%CI[+0.000000000,+0.000000000],one-sidedLCB95 **+0.000000000**. SECONDARY CATL2−BASE +0.006371494,95%CI[-0.000125072,+0.014959640],LCB95+0.000910849. Full gain retention 100.00%; positive primary root points 0/3; configuration agreement NL15 36/36, BASE 9/36.

| Comparator | Root | Paired drop10 median | 95% CI | One-sided LCB95 |
|---|---|---|---|---|
| NL15 | 30001 | +0.000000000 | [+0.000000000,+0.000000000] | +0.000000000 |
| NL15 | 30002 | +0.000000000 | [+0.000000000,+0.000000000] | +0.000000000 |
| NL15 | 30003 | +0.000000000 | [+0.000000000,+0.000000000] | +0.000000000 |
| BASE | 30001 | -0.001718321 | [-0.011627728,+0.016883363] | -0.009010291 |
| BASE | 30002 | +0.006371494 | [-0.002021231,+0.020448661] | -0.000468851 |
| BASE | 30003 | +0.009635268 | [-0.003985136,+0.018865438] | -0.002329603 |

Locked gates: `{"gain_retention": true, "harmed_nonincrease": true, "paired_drop_top10_at_least_002": false, "positive_on_two_seeds": false}`. ADVANCE requires primary paired drop10>=+.002 AND one-sidedLCB95>0; fullgain>=90%NL15; harmedshare<=NL15; positive primary paired root points on>=2/3roots. Ties and subthreshold values never advance. Per-root CIs add no gate. Incomplete outcomes would remain UNKNOWN; this outcome uses all36tasks.

All new ordinary and paired CIs use10000whole-family draws/RNG390191, shared across three arms and all nine cells. Each resample redrops each arm’s own top10; global statistic median9cells, per-root median3partitions with the same draws. CI95 quantiles .025/.975; one-sidedLCB95 .05. [metrics.json](metrics.json) preserves all27cell means/literal medians/CIs/drop1/5/10/LOFO minimum/W-T-L/harm gain<−.01/invalids/switches/Elo+CI, seedJaccard, configurations, both paired comparisons and all root intervals. Byte-heavy regression/hash inventories remain hash-bound in archived rawmetrics.

All36fsynced locks and their parents, physical closure/native/veto proofs preceded a fresh cold once-only outer score process. All36score markers COMPLETE. Physical outer rows/packets before lock:0. All36504actual native fits(29484ranker/7020veto) and14040saved bindings(12636ranker/1404veto) passed constructor/native/integer/seed checks. Every serialized native field matches the canonical H352 header at the unchanged grid choice except sole ranker cat_l2 10→20; classifiers have no intervention. Observed leaves were measured from actual tree text. All108final veto SHAs match both comparators.

48preregistered data-free controls PASS,0failed/skipped; both complete actualcommittedH352 AST inverse proofs PASS. Two separate synthetic terminal-numeric checks PASS, including rejection of a corrupted secondary per-root CI. Actual scientific unit exited0. Independent saved-family arithmetic replay verified all ordinary/paired/global/root CIs/Elo/Jaccard/configs and four gates without new fits, predictions or feature reads. TRAIN input identities matched before and after. Review is same-agent. HostedCI NOT_RUN:CI_FREEZE+BATCH_WINDOW.

ONE5s both-SMT>=85%-idle union21–26(+85–90)/46–49(+110–113) admitted [21, 22, 23, 46, 47, 48, 49]; same-interval hostCPU 13.875032%<=70; first<=8/min4. Failedpairs24/25/26 were unused. Initial forecast 2026-11-05T20:29:15.571451+00:00; first chronological task remaining35forecast 2026-11-05T20:38:53.634358+00:00, both before fixed Nov5 23:15Zcompute/23:30Zclose. Admission reference H352actualCPU26189.516914; efficiency.9/safety1.15/reserve300s. Firsttask uses max(H352CPU/36,firstactualCPU), zero ongoingwork credit. No additional capacity probe, restart, borrowed cores, deadline reset, cut cells or partial scoring.

nice19/numericthreads1/MemoryMax6GiB/SwapMax0/CPUQuota800%. Actual memory peak 2.848GiB; swap0bytes; scratch peak0.950GB; hostCPU60max33.94%; MemAvailablemin65.16GiB. Corrected guard stops only own unit for MemAvailable<20GiB OR CPU>85% averaged60s OR load1>108 on three consecutive60s windows OR scratch>2GB/fixeddeadline. Old instantaneousCPU85/absoluteLoad85 thresholds are retired. Resource accounting gives no speed verdict.

Create-only main archive `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H390/gov03-closeout-20261105T203615Z/h390-scratch.tar`, VersionId `MKtusGFOXYtkTFGvEWbiC2ndf.umoiBJ`, SHA256 `3905865da7b952394e9c5cb36d73a442089b8889bc0312a471ef037031143559`: exactVersionGET/wholeSHA/every2883member verified and originals reread. All36inner component objects/12960members were independently verified by exactVersionGET/whole/member/native/observedleaf/gzipCRC. Their recovery manifest is `provenance/component-archive-verified.json` in the main archive. Main archive retains final refits, inputs, locks, audits, source and results; inner weight payloads reside in the separate components. Receipt VersionId `r5X.lvOa2ZzlI9xxIEQnoPjsPlKCxx07`. Only after recovery proof and fresh stopped-consumer0 check were owned scratch665,312,349bytes and generatedrestore reclaimed. Immutable local archive/source/receipts remain in `/home/ubuntu/jope_plan/evidence/H390/gov03-20261105-t1905`.

Retain frozen NL15. CATL2 failed at least one locked gate. No dev91 evaluation is triggered; a new intervention needs a separate owner assignment and preregistration.

Sequential exploratory sweep H348–H390; no multiplicity-adjusted confirmatory claim. Elo is not AUC; no speed verdict. DOPE c51a56d0 unchanged/unloaded, CPU selection layer only. No DL-Val/dev91/DL-Test/sealed/holdout/meta-test/fresh-cohort rows, GPU/qualification.lock, livewriter/AUTORESTART/flywheel_gp_fused mutation, forbidden host or foreign job signals. H354/H357/H364/H369/H381cl_fwriter,H383,H386,H389,H391 untouched. No new paid compute allocation or model-service calls; existinghostCPU and owner-requiredS3custody only. Canonical HEAD/index/62foreignstagedwhitepaper paths and other dirtyfiles preserved; privateindex/ref rd/h390-gov03 only. Nonforce origin/v1/next ONLY; noPR/main/fastlane/hostedCI/modelsGit. Earlier waves remain closed.
