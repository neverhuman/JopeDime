# H338: column subsampling

**COMPLETE — COLSUB_NOT_ADVANCED**, TRAIN-only shadow research. [PREREG.md](PREREG.md) was published at **a2a65f24d54ade71a819abf309385b8d4cd20ccf** before any H338 optimizer fit. DOPE: `{"composite_sha256_prefix":"c51a56d0","loaded":false,"scope":"CPU selection layer only"}`.

## Hypothesis and setup

Fixed per-node feature masks may decorrelate trees and reduce reliance on high-spread features. Only ranker **feature_fraction_bynode=.8** and **feature_fraction_seed=that fit's original constructor random_state** are added. Original colsample_bytree/native feature_fraction=.8 is retained. The explicit new feature_fraction_seed also controls those original tree-level draws: this two-keyword experiment does not isolate the per-node mask mechanism. There was no setting search, data-derived cap or additional arm.

The AST patch copies each constructor's original `int(random_state)` expression, including every main/bag call's derived seed, rather than using the root or member seed indiscriminately. Removing exactly the two keywords restores both exact BASE function ASTs. Pinned H287 topk.fit_ranker/bagged.fit_ranker_n cover all inner rankers, final main refits and eight bags. All 36,504 constructor records and 14,040 saved native models were audited: ranker bynode=.8, unchanged tree-level fraction=.8, native feature_fraction_seed equals native seed, implicit default label_gain and disabled extra-trees/clipping. Veto constructors get neither keyword; all108 final veto native SHAs equal archived BASE.

BASE is unchanged H303 NESTED/H308 P0/P1/P2 × roots30001–30003 × four family outer folds:1219 TRAIN contexts/205 families,three members/root,eight bags/member. Exact-version GET plus complete object/member SHA recovered H315/H316 BASE arrays/locks. Regression passed all9panels,36fold-root pick and lock hashes,and exact medians .0643163334285782/.04821480030677507/.04360142642936269. BASE refits/new model scores0. Current1219TRAINpools,four consumed TRAIN cache prefixes,selected metadata and r882 TRAIN costs match archived input hashes.

Original GBDT/LambdaRank/ndcg, labels0–7/default gains[first8:0,1,3,7,15,31,63,127], truncation, learning-rate/tree-count grid, selected-round rules, numerical parameters, features, random_state values, folds, raw inner objective,18-configtau/k/agreementgrid,1e-4 conservative ties/switch rules and invalid utility−2.0 remain unchanged. No H332 clipping,H328 linear gains,H325 extra-trees or other wave override.

Each fit worker physically loaded only its proper outer-TRAIN pool/cost members and exact per-context cache ranges. Identifier/offset metadata may be read. No inherited full packet cache; scoped hooks forbid outer contexts and BASE outcomes during fit. All36 locks and directories were fsynced before fresh-process,once-only outer scoring. Physical outer reads/prebuilt outer packets/guard denials before lock0;DL-Val feature rows0. All36 STARTED/COMPLETE markers retained. No partial cell set,dev91,DL-Test,sealed/holdout/meta-test,fresh cohort or live checkpoint access. Historical TRAIN design exposure remains; this is not independent fresh confirmation.

## Outcome

Median across9cells of family-equal mean raw gain: BASE **+0.048215**,95%CI[0.02506636301915145, 0.07619106094683858]; COLSUB **+0.058201**,CI[0.023671487183452015, 0.09234343508181117]. Literal family-median gain: BASE+0.003624,CI[0.0002989597199160865, 0.007726430567748466];COLSUB+0.004465,CI[0.0, 0.008220309962533534]. Retention uses the former.

Paired drop-top10 median COLSUB−BASE **-0.002450**,95%CI **[-0.008221796233816029, 0.0042008636948849066]**,one-sidedLCB95 **-0.007121**. Whole-family paired bootstrap10,000draws,RNG338191,same family draw across both arms/all9cells,each arm re-dropping its own ten best families on every draw. Per-root medians across partitions:[0.0008849786822924133, -0.0024496648742264975, -0.006477354097090268]. ADVANCE requires paired median≥+.002 andLCB95>0,full-gainretention≥90%,no median harmed-share increase and positive paired median on≥2/3roots. Ties never advance. Gates:`{"gain_retention": true, "harmed_nonincrease": false, "paired_drop_top10_at_least_002": false, "positive_on_two_seeds": false}`.

Median harmed share(gain<−.01): BASE **9.268%**,COLSUB **10.732%**. Config matches **3/36**,CONFIG_SENSITIVE(descriptive30/36threshold,not another decision gate). Median Jaccard of switched-context sets across root pairs: BASE0.441805,COLSUB0.478510.

Median Elo: BASE **1725.66**,COLSUB **1706.38**. **reference = r882 slot0 (DL-Train outer); r3175 not measured on TRAIN**. W/T/L tolerance1e-4. No AUC/speed claim.

| Part | Root | Arm | Family mean | Family median | Drop10 | W/T/L | Harmed | Invalid | Switches | Elo |
|---|---:|---|---:|---:|---:|---|---:|---:|---:|---:|
| P0 | 30001 | BASE | +0.041289 | +0.000000 | +0.013174 | 88/93/24 | 6.341% | 13 | 219 | 1725.66 |
| P0 | 30002 | BASE | +0.064316 | +0.001588 | +0.008633 | 108/67/30 | 8.780% | 14 | 300 | 1722.48 |
| P0 | 30003 | BASE | +0.072258 | +0.006984 | +0.016621 | 122/50/33 | 9.268% | 11 | 320 | 1727.10 |
| P0 | 30001 | COLSUB | +0.060287 | +0.001266 | +0.004514 | 105/68/32 | 8.293% | 14 | 247 | 1706.38 |
| P0 | 30002 | COLSUB | +0.064929 | +0.004465 | +0.021153 | 116/56/33 | 10.732% | 14 | 293 | 1718.34 |
| P0 | 30003 | COLSUB | +0.059077 | +0.004783 | +0.020649 | 110/62/33 | 9.268% | 14 | 269 | 1709.11 |
| P1 | 30001 | BASE | +0.048215 | +0.003000 | +0.010198 | 112/57/36 | 11.220% | 17 | 327 | 1697.13 |
| P1 | 30002 | BASE | +0.048658 | +0.003624 | +0.014063 | 116/51/38 | 10.244% | 15 | 275 | 1693.84 |
| P1 | 30003 | BASE | +0.038294 | +0.003253 | +0.012702 | 112/64/29 | 8.293% | 16 | 285 | 1734.68 |
| P1 | 30001 | COLSUB | +0.045029 | +0.003764 | +0.011083 | 115/62/28 | 8.780% | 15 | 281 | 1745.37 |
| P1 | 30002 | COLSUB | +0.036423 | +0.000000 | +0.006709 | 94/88/23 | 7.317% | 17 | 201 | 1744.50 |
| P1 | 30003 | COLSUB | +0.033407 | +0.005804 | +0.006225 | 118/50/37 | 12.195% | 18 | 358 | 1701.44 |
| P2 | 30001 | BASE | +0.043601 | +0.006326 | +0.010678 | 122/48/35 | 10.732% | 16 | 315 | 1716.88 |
| P2 | 30002 | BASE | +0.031840 | +0.003624 | +0.008774 | 112/65/28 | 7.805% | 17 | 292 | 1740.78 |
| P2 | 30003 | BASE | +0.054721 | +0.008145 | +0.023605 | 129/41/35 | 10.732% | 15 | 335 | 1726.57 |
| P2 | 30001 | COLSUB | +0.058201 | +0.006226 | +0.012372 | 120/43/42 | 12.683% | 16 | 338 | 1682.35 |
| P2 | 30002 | COLSUB | +0.037916 | +0.000296 | +0.006324 | 103/64/38 | 10.732% | 17 | 299 | 1673.19 |
| P2 | 30003 | COLSUB | +0.060992 | +0.010321 | +0.016299 | 122/43/40 | 12.195% | 15 | 384 | 1693.69 |

[metrics.json](metrics.json),below50KB,preserves all18arm/cell records,nine paired differences,rawmean/literalmedianCIs,drop1/5/10,LOFOmin,W/T/L,harm,invalid/slot0invalid,switches/rates,EloCIs,allrootpairs,configs/agreement,72lockhashes/18pickhashes and regression/input/exit/resource evidence. No cell or family was filtered.

## Verification, resources and custody

24 targeted tests passed,0failed/skipped,exit0. Tests prove two exact reverse ASTs,the sole two-keyword parameter delta,existing fit-seed expression copied at both sites,unchanged labels/groups/eval/rounds,veto fit/constructor nonmutation,real estimator signature/read-only observer,wrong fraction/seed and foreign ablation refusal,identity/member-scope cleanup,paired bootstrap/fouradvancegates/Jaccard,outer/BASE outcome refusal,poisoned-tail exclusion and missing-lock refusal. The first dry suite caught stale CLIP verdict prefixes from the template; corrected and all24 rerun before prereg/fits,failed receipt retained. No synthetic optimizer fit. Source syntax/JSON/whitespace checks passed before launch. Actual model exit `exited/0/success`,invocation`3dcf3229f3854487a5663e646b431830`,captured by ExecStopPost.

CPUxbabe2,nice19,numeric1,four assigned cores13/15/60/62 with fresh≥85%idle on all cores and SMT77/79/124/126,disjoint H334/H335/H339. Initial busy claim probe was not fit admission; the fresh post-regression gate passed. MemoryMax6GiB/Swap0. Sampled maxhostCPU52.93%,load75.41,minMemAvailable62.97GiB,maxscratch1107270543bytes. Fixed00:45ZNov5compute/01:00Zclose;noreset/cutcells/partialscore/extraarm. NoGPU/qualification.lock/Cargo/nativebuild/GitHubCI;DOPEweightsunloaded/frozen,cachedTRAINCPUfeatures only.

Inner archives were create-only uploaded and exact-version GET/SHA verified before raw-copy reclamation. Final source/models/locks/results stream create-only to H338 S3; whole exact-version SHA and every member are verified before enumerated own scratch deletion. [s3_manifest.json](s3_manifest.json) pins versions; full native/input/member inventories remain archived. Source/small receipts retained under `/home/ubuntu/jope-rescue-20260921/h338-solhwv`. No models inGit. Nonforcev1/next only;two assigned index rows. No PR/main/fastlane/CI/FINDINGS,foreign checkout/job,serving or forbidden-host change. Replay needs pinned H310/H287 code,SHA-pinned TRAIN inputs and BASE exact versions. Closeout authorizes no extra fit/score.

Next: do not advance COLSUB or run dev91; preserve the complete negative result.
