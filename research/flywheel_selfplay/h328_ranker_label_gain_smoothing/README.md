# H328: ranker label-gain smoothing

**COMPLETE — LGS_NOT_ADVANCED**, TRAIN-only shadow research. Preregistration [PREREG.md](PREREG.md) was published at **bf866dc346ebcb6dac7119b4de89cd1fc9689869** before any fit. DOPE: `{"composite_sha256_prefix":"c51a56d0","loaded":false,"scope":"CPU selection layer only"}`.

## Hypothesis and setup

The fixed hypothesis is that linear relevance gains spread LambdaRank signal across positions and improve robustness after dropping the ten most beneficial families. This wave changes the gain table only; no claim that the proposed mechanism is independently measured.

BASE is unchanged H303 NESTED/H308 P0/P1/P2 x roots30001/30002/30003 x4 family outer folds,1219 TRAIN contexts/205 families,r882 top8 slots,three members/root,eight bags/member. Exact-version GET+whole/member SHA verified H315/H316 frozen BASE arrays/locks. Regression before LGS fitting passed9/9panels,36fold-rootpickhashes+lockhashes,and exact medians .0643163334285782/.04821480030677507/.04360142642936269 from current raw TRAIN utilities. BASE refits/newmodel scores0. All1219TRAINpools,all4consumedfeatureprefixes,selectedmetadata and r882TRAINcost hashes match archived BASE identities.

Original ranker labels remain ordinal0..7. Effective BASE gains are **[0,1,3,7,15,31,63,127]**, the first8 LightGBM default2^l−1 gains; higher entries are unused. LGS is exactly **label_gain=[0,1,2,3,4,5,6,7]**, gain(l)=l. Only pinned H287 topk.fit_ranker and bagged.fit_ranker_n gain this single constructor keyword. It covers inner rankers,finalmainrefits and all8bag rankers. The reverse-AST test removes this one keyword and restores each exact BASE function AST. Original labels,features,LambdaRank/ndcg,truncation,LR/tree-count search and selected-round rules,numericalparameters,randomstreams,folds,18-configtau/k/agreementgrid,raw innerobjective,1e-4 conservative tie/switchrules and invalidutility−2.0 stay unchanged. Veto constructors/fitarguments have no gain override; all108 final veto model hashes equal archived BASE exactly. H325 extra_trees/extra_seed overrides are absent: native models confirm extra_trees=false. [manifest.json](manifest.json) records both AST proofs,constructor audits and14040saved native model checks.

Regression/preflight exited in separate processes before fitting. Each fitworker loaded only its outer_train pool/cost members and exact per-context cache byte ranges; no inherited full packet cache. Identifier/offset metadata may be read. Scoped audit hooks/range checks reject outer contexts and archived BASE outcomes. All36locks and parentdirectories were fsynced before outer scoring; physical outer reads/prebuilt outer packets/guarddenials0,DL-Valfeature rows0. A fresh scoring process validated all36locks/models/archives and scored each fold/root once with STARTED/COMPLETE markers. No partial cell set,dev91/DL-Valfeature or utility rows,DL-Test,sealed/holdout/meta_test,freshcohorts or live checkpoints. Historical TRAIN design exposure remains; this is not fresh confirmation.

## Outcome: raw utility and all nine cells

Median across9cells of family-equal mean gain: BASE **+0.048215**,95%CI[0.025121606279544264, 0.07598989822651814]; LGS **+0.070933**,CI[0.03550062041790897, 0.12712373713345235]. Literal family-median gain: BASE+0.003624,CI[0.0004378293769140007, 0.007726430567748466]; LGS+0.013822,CI[0.008960451271892156, 0.01962554214619061]. The retention gate uses the former,as preregistered.

Paired drop-top10 median LGS−BASE **+0.011376**,95%CI**[0.0006109937234960905, 0.019138152828032908]**,one-sidedLCB95**+0.001973**. Whole-family paired bootstrap10000draws,RNG328191,same family draw across botharms/all9cells; each arm re-drops its own ten best on each draw. Per-root paired medians across partitions:[0.0004347561350614929, 0.011375833626592745, 0.015416600115188103]. ADVANCE requires median>=+.002,LCB95>0,gainretention>=90%,harmednonincrease,and positive paired median on>=2/3roots. Ties never advance. Gates:`{"gain_retention": true, "harmed_nonincrease": false, "paired_drop_top10_at_least_002": true, "positive_on_two_seeds": true}`.

Median harmed-family shares(gain<−.01): BASE9.268%,LGS13.171%. Config matches **9/36**,**CONFIG_SENSITIVE** (descriptive threshold30/36,not another decision gate). Median seed-Jaccard of switched-context sets: BASE0.441805,LGS0.615385.

Median Elo: BASE **1725.66**,LGS **1707.50**. **reference = r882 slot0 (DL-Train outer); r3175 not measured on TRAIN**. Family W/T/L tie tolerance1e-4. No AUC/speed claim.

| Part | Root | Arm | Family mean | Family median | Drop10 | W/T/L | Harmed | Invalid | Switches | Elo |
|---|---:|---|---:|---:|---:|---|---:|---:|---:|---:|
| P0 | 30001 | BASE | +0.041289 | +0.000000 | +0.013174 | 88/93/24 | 6.341% | 13 | 219 | 1725.66 |
| P0 | 30002 | BASE | +0.064316 | +0.001588 | +0.008633 | 108/67/30 | 8.780% | 14 | 300 | 1722.48 |
| P0 | 30003 | BASE | +0.072258 | +0.006984 | +0.016621 | 122/50/33 | 9.268% | 11 | 320 | 1727.10 |
| P0 | 30001 | LGS | +0.057350 | +0.012535 | +0.013608 | 141/32/32 | 10.732% | 16 | 447 | 1757.59 |
| P0 | 30002 | LGS | +0.083448 | +0.014457 | +0.020009 | 139/28/38 | 13.171% | 9 | 497 | 1725.26 |
| P0 | 30003 | LGS | +0.093192 | +0.014061 | +0.032647 | 142/20/43 | 14.146% | 11 | 519 | 1707.50 |
| P1 | 30001 | BASE | +0.048215 | +0.003000 | +0.010198 | 112/57/36 | 11.220% | 17 | 327 | 1697.13 |
| P1 | 30002 | BASE | +0.048658 | +0.003624 | +0.014063 | 116/51/38 | 10.244% | 15 | 275 | 1693.84 |
| P1 | 30003 | BASE | +0.038294 | +0.003253 | +0.012702 | 112/64/29 | 8.293% | 16 | 285 | 1734.68 |
| P1 | 30001 | LGS | +0.049770 | +0.010311 | +0.008506 | 135/27/43 | 13.171% | 15 | 485 | 1698.72 |
| P1 | 30002 | LGS | +0.070933 | +0.012001 | +0.021531 | 130/34/41 | 14.146% | 12 | 415 | 1700.43 |
| P1 | 30003 | LGS | +0.061692 | +0.016366 | +0.028119 | 140/20/45 | 14.634% | 11 | 493 | 1697.14 |
| P2 | 30001 | BASE | +0.043601 | +0.006326 | +0.010678 | 122/48/35 | 10.732% | 16 | 315 | 1716.88 |
| P2 | 30002 | BASE | +0.031840 | +0.003624 | +0.008774 | 112/65/28 | 7.805% | 17 | 292 | 1740.78 |
| P2 | 30003 | BASE | +0.054721 | +0.008145 | +0.023605 | 129/41/35 | 10.732% | 15 | 335 | 1726.57 |
| P2 | 30001 | LGS | +0.077738 | +0.010245 | +0.022690 | 134/30/41 | 12.195% | 13 | 465 | 1705.70 |
| P2 | 30002 | LGS | +0.088326 | +0.015770 | +0.034866 | 143/27/35 | 12.195% | 13 | 500 | 1744.47 |
| P2 | 30003 | LGS | +0.066749 | +0.013822 | +0.021228 | 138/33/34 | 10.732% | 13 | 446 | 1743.32 |

[metrics.json](metrics.json) preserves all18arm/cell records/all9paired differences,rawmean/medianCI,drop1/5/10,LOFOmin,W/T/L,harm,invalid/slot0invalid,switches/rates,EloCI,allseedpairs,configchoices/agreement,72lockhashes/18pickhashes,regression/input/exit/resource evidence. No cell or family was filtered; raw utility is used throughout.

## Checks, resources and custody

24 targeted tests passed,0failed/0skipped,exit0. They prove exact default/newtables,reverse-AST equality at both call sites,onlyone keyword added,unchanged labels/group/eval arguments,veto fit/constructor nonmutation,real estimator signature/parameter behavior,no treeoverride,classifiergainrefusal,originalordinaltargets,contextidentityfields,member-scopecleanup,pairedbootstrap/tie/.002threshold/LCB/harm/retention/two-rootcriteria,Jaccard,outer/base-inputrefusal,poison-tail exclusion and missing-lock refusal. No synthetic optimizer fit. Source syntax and gitdiffcheck passed before launch. Actual finalexit `exited/0/success` was captured by ExecStopPost,invocation088550a1c94846b4815834df6dfbdd9f.

CPUxbabe2,nice19,numeric1,4cores13/15/60/62 with SMT peers checked and disjoint H327/H329,MemoryMax6GiB/Swap0. Sampled maxhostCPU27.83%,load40.34,minMemAvailable70.54GiB,maxscratch1149737360bytes. Fixed compute20:45Z/close21:00ZNov4; no budget reset,extra arm or posthoc protocol change. NoGPU/qualification.lock/Cargo/native/GitHubCI; DOPE weights unloaded/unchanged,TRAINcached CPUfeatures only.

Own inner models were archived create-only with exact-version GET+SHA before raw-copy removal. Final source/models/locks/output archive streams to H328 S3; exact-version wholeSHA/member verification precedes enumerated own scratch deletion. [s3_manifest.json](s3_manifest.json) pins versions; full native/input/member inventories remain archived and in custody. Models never enter Git. Source/small receipts retained under /home/ubuntu/jope-rescue-20260921/h328-solhwv. Nonforce origin/v1/next only,no PR/main/fastlane/CI,other-owner job/checkout,forbidden host or serving change. The task's two specified index rows are updated.

Replay needs pinned H310 CPU sources,H287 export,host SHA-pinned TRAIN inputs and archived BASE exactversions. The H316 member inventory is recoverable from that pinned archive. Closeout authorizes no extra fit or score. Next: do not advance LGS or score dev91.
