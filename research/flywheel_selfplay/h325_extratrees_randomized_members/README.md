# H325: extra-trees randomized members

**COMPLETE — XT_NOT_ADVANCED**, TRAIN-only shadow research. Preregistration [PREREG.md](PREREG.md) was pushed at **0fac0d7a80e3a56358ea36df8c213883a5c7d51e** before every fit. DOPE: `{"composite_sha256_prefix":"c51a56d0","loaded":false,"scope":"CPU selection layer only"}`.

## Method and regression

The fixed hypothesis was that randomized split thresholds might reduce member variance and improve concentration robustness. This experiment tests LightGBM extra_trees; it does not establish SAM equivalence or measure flat minima. H306 PREREG/code were checked: only a TAU/k/a grid extension, no extra_trees.

BASE is unchanged H303 NESTED/H308 P0/P1/P2, roots30001–30003, four family outer folds, 1219 TRAIN contexts/205 families, r882 top8 slots, three members/root, eight bag rankers/member. Exact-version GET and whole/member SHA verification preceded read-only reuse of H315/H316 BASE arrays/locks. Regression before XT fit passed all9 panels,36 fold-root pick hashes and lock hashes, and exact medians .0643163334285782/.04821480030677507/.04360142642936269. BASE refits and new BASE model scores were0. All1219 TRAIN pool SHA hashes and all4 consumed feature-prefix hashes match archived BASE, as do selected metadata and r882 TRAIN costs.

XT changes only `extra_trees=True` and `extra_seed=owning member seed` for every original member LightGBM construction: inner/main/veto/all8bag rankers/final refits. Original bag random_state streams remain unchanged. LambdaRank, ordinal relevance labels, default label_gain, binary invalid labels, features, numeric parameters, folds/seeds, 18-config grid, raw family-equal mean inner objective,1e-4 conservative tie and switching rules remain original. H321's normalized targets and regression loss adapter are absent. TAU remains invalid probability; margins remain original LambdaRank score units and the original inner-selected grid. Constructor audit receipts and native saved model parameters are verified in [manifest.json](manifest.json).

Regression/preflight ran in separate processes that exited before fitting. The fit parent loaded only metadata and pinned modules; each task loaded only its outer_train pools/cost members and exact cache byte ranges. Scoped audit hooks reject outer pool opens and archived BASE outcome reads. Per-context feature/range/hash receipts are archived. Each of36 locks and its parent directory were fsynced before any outer scoring. Fit-worker physical outer reads, prebuilt outer packets, guard denials and DL-Val feature rows were0. After all36 locks were reviewed, a fresh process scored each outer fold/root exactly once, with STARTED/COMPLETE markers. No dev91/DL-Val features/utilities,DL-Test,sealed/holdout/meta_test/fresh cohorts or live checkpoints were read. Historical TRAIN design exposure remains; this is not fresh confirmation.

## Raw-utility results

Median over9 cells of family-equal mean gain: BASE **+0.048215**,95%CI[0.024800167377447824, 0.07578597011526424]; XT **+0.056392**,CI[0.031492601102211734, 0.08512976620024101]. Literal family-median gain: BASE+0.003624,CI[0.0002959048497594985, 0.007508658268156609]; XT+0.007682,CI[0.004782505246892055, 0.013087914425885971]. The90% retention gate uses median across cells of family-equal mean gain.

Paired drop-top10 median XT−BASE **+0.004282**,95%CI**[-0.004120519936988751, 0.01478703045751391]**,one-sidedLCB95**-0.002784**. Whole-family paired bootstrap10000 draws,RNG325191; same family draw in both arms/all9 cells; each arm drops its own ten best families on each draw. Per-root medians across partitions: [0.005697733014391242, 0.006942746159687471, -0.0023492346425401604]. ADVANCE requires paired median>=+.002 AND LCB95>0 AND gain retention>=90% AND median harmed nonincrease AND positive paired median on>=2/3 roots. Ties never advance. Gates: `{"gain_retention": true, "harmed_nonincrease": false, "paired_drop_top10_at_least_002": false, "positive_on_two_seeds": true}`.

Median harmed shares(gain<−.01): BASE9.268%,XT12.195%. Config matches **6/36**,**CONFIG_SENSITIVE** (insensitive threshold30/36). Median seed-Jaccard: BASE0.441805,XT0.493671; pairwise switched-context sets,empty/empty=1.

Median Elo: BASE **1725.66**,XT **1716.96**. **reference = r882 slot0 (DL-Train outer); r3175 not measured on TRAIN**. Family W/T/L uses tolerance1e-4. No AUC or speed claim.

| Part | Root | Arm | Family mean | Family median | Drop10 | W/T/L | Harmed | Invalid | Switches | Elo |
|---|---:|---|---:|---:|---:|---|---:|---:|---:|---:|
| P0 | 30001 | BASE | +0.041289 | +0.000000 | +0.013174 | 88/93/24 | 6.341% | 13 | 219 | 1725.66 |
| P0 | 30002 | BASE | +0.064316 | +0.001588 | +0.008633 | 108/67/30 | 8.780% | 14 | 300 | 1722.48 |
| P0 | 30003 | BASE | +0.072258 | +0.006984 | +0.016621 | 122/50/33 | 9.268% | 11 | 320 | 1727.10 |
| P0 | 30001 | XT | +0.055860 | +0.011603 | +0.017456 | 131/27/47 | 16.585% | 16 | 405 | 1678.05 |
| P0 | 30002 | XT | +0.056865 | +0.011498 | +0.011182 | 136/30/39 | 11.707% | 11 | 392 | 1716.96 |
| P0 | 30003 | XT | +0.048865 | +0.010888 | +0.013598 | 136/35/34 | 12.683% | 15 | 351 | 1740.79 |
| P1 | 30001 | BASE | +0.048215 | +0.003000 | +0.010198 | 112/57/36 | 11.220% | 17 | 327 | 1697.13 |
| P1 | 30002 | BASE | +0.048658 | +0.003624 | +0.014063 | 116/51/38 | 10.244% | 15 | 275 | 1693.84 |
| P1 | 30003 | BASE | +0.038294 | +0.003253 | +0.012702 | 112/64/29 | 8.293% | 16 | 285 | 1734.68 |
| P1 | 30001 | XT | +0.048552 | +0.006397 | +0.015895 | 122/47/36 | 12.195% | 12 | 351 | 1711.99 |
| P1 | 30002 | XT | +0.065399 | +0.006552 | +0.021006 | 123/49/33 | 11.220% | 12 | 326 | 1728.52 |
| P1 | 30003 | XT | +0.045790 | +0.007682 | +0.010353 | 127/36/42 | 14.634% | 13 | 380 | 1692.19 |
| P2 | 30001 | BASE | +0.043601 | +0.006326 | +0.010678 | 122/48/35 | 10.732% | 16 | 315 | 1716.88 |
| P2 | 30002 | BASE | +0.031840 | +0.003624 | +0.008774 | 112/65/28 | 7.805% | 17 | 292 | 1740.78 |
| P2 | 30003 | BASE | +0.054721 | +0.008145 | +0.023605 | 129/41/35 | 10.732% | 15 | 335 | 1726.57 |
| P2 | 30001 | XT | +0.056392 | +0.006065 | +0.021334 | 117/58/30 | 10.244% | 12 | 302 | 1736.38 |
| P2 | 30002 | XT | +0.060379 | +0.006691 | +0.022575 | 116/50/39 | 12.683% | 12 | 312 | 1689.33 |
| P2 | 30003 | XT | +0.071584 | +0.012882 | +0.027578 | 138/31/36 | 11.220% | 14 | 406 | 1733.39 |

[metrics.json](metrics.json) includes all18arm/cell records and all9 paired differences, raw gain and family-median CIs,drop1/5/10,LOFOmin,W/T/L,harm,invalid,slot0invalid,switches/rates,EloCIs,allseed-Jaccardpairs,configchoices/agreement,alllock/pickhashes,regression and resources. No cell or family was filtered.

## Validation and custody

17 targeted tests passed,0failed/0skipped,exit0. Tests cover class/signature retention, actual constructor parameter visibility, AST-extracted pinned main/bag/veto fits proving only the two XT parameters change and labels/group/eval arguments remain exact, member seed scope/error cleanup,threshold/tie/LCB/harm/retention/two-seed criteria,paired bootstrap,Jaccard,physical outer/base outcome rejection,exact-byte scoped reads excluding a poisoned tail,and missing-lock refusal. No synthetic optimizer fit preceded regression. Final job exit `exited/0/success` was captured durably by ExecStopPost. Resource-only [A1](RESOURCE_AMENDMENT.md), published at 0804a092a0e38e953d44733b34c4dd825aa75ecd before continuation fits, replaced the initial3-worker service with4workers after core60 became idle. The initial service was intentionally stopped(killed/TERM, not reported as a successful model exit), its cgroup confirmed empty and exit retained. Six completed cells were reused with every file SHA unchanged; three incomplete directories(605files) were preserved and those incomplete cells refit from scratch. No duplicate workers, outer scores at handoff, scientific source changes or deadline reset. All interrupted files are in final custody.

CPUxbabe2,nice19,numeric1,actual fit cores[13, 15, 60, 62],MemoryMax6G/Swap0. Sampled max hostCPU39.05%,load56.96,minimumMemAvailable69.26GiB,maxscratch1225268393bytes. Fixed compute17:30Z/close17:45Z Nov4; no budget reset. No GPU/qualification.lock/Cargo/native/GitHubCI. CPU-only cached features; DOPE weights unloaded and unchanged.

Own inner models were archived create-only with exact-version GET+SHA verification before raw-copy reclamation. Final source/models/locks/output archive streams to H325 S3 and is exact-version SHA/member verified before enumerated own scratch deletion. [s3_manifest.json](s3_manifest.json) pins versions; full inventories remain archived and in custody. No models in Git. Source/custody receipts remain under /home/ubuntu/jope-rescue-20260921/h325-solhwv. Nonforce origin/v1/next only; no PR/main/fastlane/CI,other-owner job/checkout,forbidden host or serving change.

Replay needs pinned H310 CPU sources,H287export,host TRAIN inputs and archived BASE exact versions. The fetch helper's H316 inventory is recoverable from that pinned archive. Closeout authorizes no extra fit or score. Next: do not advance XT or score dev91.

Publication formatting: one extra EOF newline was removed from the resource guard after the first diff-check rejection; its Python AST is identical. The exact runtime bytes remain in the verified model archive. Manifest records both runtime and published source hashes; no fit, score, threshold, metric or decision changed.
