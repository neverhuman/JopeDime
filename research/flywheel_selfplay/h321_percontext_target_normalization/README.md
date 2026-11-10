# H321: per-context member target normalization

**COMPLETE — CNORM_NOT_ADVANCED**, DL-Train-only shadow experiment. Preregistration [PREREG.md](PREREG.md) was published at **fea4923183a0c26a3aeb6a7bf5909b92b2fa98d7** before any fit. Its rule says: **“Ties never advance.”**

CNORM changed the loss from **LambdaRank → regression_l2**. This negative result is **target+loss confounded, not a pure target-normalization ablation**. No fitting, scoring or loss switch was repeated after observing the result.

DOPE record (owner closeout ruling):

```json
{
  "dope": {
    "composite_sha256_prefix": "c51a56d0",
    "loaded": false,
    "scope": "CPU selection layer only; DOPE weights not loaded or modified"
  }
}
```

## Hypothesis, setup and necessary interpretation

The proposed hypothesis is that context-scale-normalized relative-utility supervision can spread member skill beyond high-variance contexts and improve concentration robustness. Actual source inspection found that BASE uses ordinal LambdaRank, rather than a raw-utility regression loss: raw utility magnitudes are not its direct training labels. Thus this wave evaluates the explicitly preregistered **target-and-loss-adapter comparison**; it cannot isolate normalization as the cause of any difference.

BASE = unchanged H303 NESTED on H308 P0/P1/P2 x roots30001–30003 x4 outer family folds (1219 TRAIN contexts/205 families, r882 top8 slots,3 members/root,8 bag rankers/member). Exact-version H315/H316 frozen BASE picks and36 locks are reused read-only. Regression passed before CNORM fitting:9/9 panels,36 fold hashes and locks,3 published medians exactly .0643163334285782/.04821480030677507/.04360142642936269. BASE refits/new model forwards0. All1219 TRAIN program/fitness/gate pool hashes,4 consumed cache feature prefixes,selected metadata and r882 TRAIN cost SHA hashes match archived BASE inputs. H308 TRAIN-prefix loading read0 DL-Val feature rows. No dev91/DL-Val utility rows,DL-Test,sealed/holdout/meta-test or fresh cohort data. Historical design exposure remains; this is not independent confirmation.

For each fitting-fold TRAIN context c, MAD_c=median(abs(u_c−median(u_c))) over its8 slots (including invalid utility−2). y_cs=clip((u_cs−u_c0)/max(MAD_c,.001),−3,+3), computed float64 then stored float32. Slot0 y=0. No cross-context scale, held-out outer utility, corpus-label mutation or feature change. Local fitting-packet targets only are replaced. The necessary main/bag backend is LGBMRegressor regression_l2: LambdaRank cannot accept signed continuous targets and predict normalized relative utility. All shared numeric tree/LR/tree-count/subsample/regularization/determinism parameters,features,member streams,folds,bags and18-config(tau,k,a) grid remain fixed; ranking-only objective/metric/group/eval_at machinery is the declared exception. Invalid classifier/labels stay unchanged.

TAU is P(invalid), so it remains in probability units. Main/bag scores and the original inner-selected margin grid(0,.02,.05,.10,.20,.35,.50,.75,1,+inf) are in normalized prediction units; no calibration mapping or post-hoc switch. Member strict admission and the H303 inner config objective still use raw-utility OOF family-equal mean gain and conservative1e-4 ties. Config agreement: **7/36**.

All36 CNORM fit locks/parent directories were fsynced before each once-only outer fold score. Guarded fit-consumer outer accesses/denials0. As disclosed in the preregistration, setup prebuilt TRAIN packets; this is not a claim of zero physical outer-feature bytes read before fitting. BASE outer predictions were reused, not rescored.

## Outcome: all nine cells, raw utility

Median family-equal mean gain across9 cells: BASE **+0.048215**95%CI[0.025572299207994913, 0.07593847795199386]; CNORM **+0.045445**CI[0.00816511550779999, 0.09062636350987505]. Literal family-median gain: BASE +0.003624CI[0.0002989597199160865, 0.007726430567748466]; CNORM +0.000000CI[0.0, 0.002506933676479417]. The90% retention gate uses the former, as preregistered.

Paired raw drop-top10 median **-0.012400**,95%CI**[-0.03122370518125055, -0.0015831939102988483]**,one-sidedLCB95**-0.027735**,10000 paired whole-family draws,RNG321191; identical resample across arms/all9 cells, each arm re-drops its own top10. Median harmed shares(loss>.01): BASE9.268%,CNORM10.244%. Exact gates: `{"gain_retention": true, "harmed_nonincrease": false, "paired_drop_top10_positive": false}`.

Median Elo BASE **1725.66**,CNORM **1684.26**. **reference = r882 slot0 (DL-Train outer); r3175 not measured on TRAIN**. H302 family W/T/L tie tolerance1e-4 is retained. Median pairwise seed-Jaccard: BASE0.441805,CNORM0.468599; switched-context sets across each partition's3 seed pairs,empty/empty=1,intersection/union counts reported.

| Part | Seed | Arm | Family mean | Family median | Drop10 | LOFO min mean | Harmed | Invalid | Switch rate | Elo |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| P0 | 30001 | BASE | +0.041289 | +0.000000 | +0.013174 | +0.034482 | 6.341% | 13 | 17.966% | 1725.66 |
| P0 | 30002 | BASE | +0.064316 | +0.001588 | +0.008633 | +0.037100 | 8.780% | 14 | 24.610% | 1722.48 |
| P0 | 30003 | BASE | +0.072258 | +0.006984 | +0.016621 | +0.045081 | 9.268% | 11 | 26.251% | 1727.10 |
| P0 | 30001 | CNORM | +0.050283 | +0.000000 | -0.002860 | +0.022056 | 7.805% | 11 | 18.950% | 1735.48 |
| P0 | 30002 | CNORM | +0.045445 | +0.005556 | +0.004172 | +0.030656 | 14.146% | 10 | 30.927% | 1686.53 |
| P0 | 30003 | CNORM | +0.050628 | +0.000000 | +0.004221 | +0.027552 | 7.317% | 12 | 22.313% | 1712.18 |
| P1 | 30001 | BASE | +0.048215 | +0.003000 | +0.010198 | +0.030657 | 11.220% | 17 | 26.825% | 1697.13 |
| P1 | 30002 | BASE | +0.048658 | +0.003624 | +0.014063 | +0.035309 | 10.244% | 15 | 22.559% | 1693.84 |
| P1 | 30003 | BASE | +0.038294 | +0.003253 | +0.012702 | +0.033333 | 8.293% | 16 | 23.380% | 1734.68 |
| P1 | 30001 | CNORM | +0.043218 | +0.000000 | -0.013517 | +0.013782 | 14.146% | 13 | 27.728% | 1655.64 |
| P1 | 30002 | CNORM | +0.007570 | +0.000000 | -0.013502 | +0.002552 | 10.244% | 15 | 19.770% | 1641.98 |
| P1 | 30003 | CNORM | +0.021141 | +0.000000 | +0.001691 | +0.015762 | 10.244% | 13 | 19.524% | 1659.14 |
| P2 | 30001 | BASE | +0.043601 | +0.006326 | +0.010678 | +0.037280 | 10.732% | 16 | 25.841% | 1716.88 |
| P2 | 30002 | BASE | +0.031840 | +0.003624 | +0.008774 | +0.026502 | 7.805% | 17 | 23.954% | 1740.78 |
| P2 | 30003 | BASE | +0.054721 | +0.008145 | +0.023605 | +0.048595 | 10.732% | 15 | 27.482% | 1726.57 |
| P2 | 30001 | CNORM | +0.061141 | +0.006692 | +0.011148 | +0.040560 | 13.659% | 10 | 30.845% | 1690.82 |
| P2 | 30002 | CNORM | +0.047959 | +0.000296 | +0.003775 | +0.035123 | 9.756% | 11 | 27.810% | 1684.26 |
| P2 | 30003 | CNORM | +0.030091 | +0.002067 | +0.000300 | +0.023925 | 14.634% | 12 | 34.865% | 1671.42 |

[metrics.json](metrics.json) preserves every cell's raw mean/median CI,drop1/5/10,LOFO minimum,W/T/L,harmed,invalid/switch counts/rates,Elo CI,all seed pairs,config choices/agreement,lock/pick hashes and the complete regression. No seed or family was filtered. No normalized gain is substituted for a reported raw gain.

## Checks, resources and custody

16 targeted unittest controls passed(0failed/0skipped,exit0), including signed targets/slot0,MAD floor/clipping,context-affine invariance,nonmutation/context independence,required regression interface and numeric tree settings,200-tree wrapper,invalid-target refusal,paired bootstrap/tie/harm/retention controls,Jaccard and missing-lock/outer-access rejection. No synthetic optimizer fit preceded the BASE regression. The actual compute process exited `exited/0` with systemd result `success`, captured by ExecStopPost before unit collection. No GPU/native/Cargo/hostedCI work; DOPE c51a56d0 frozen.

CPU xbabe2,nice19,numeric1,cores48/49/50/52/53/54 disjoint H317/H320/H322,MemoryMax6G/MemorySwapMax0. Sampled max hostCPU38.84%,load59.83,minMemAvailable64.55GiB,maxscratch1101198649bytes. Fixed compute stop15:15Z/close15:30Z Nov4; no deadline/budget reset. Inner model archives were exact-version S3 GET+SHA verified before raw-copy deletion. Final archive streams to H321 S3 and is exact-version/hash/member verified before all own scratch is reclaimed; runtime bytecode is included. [s3_manifest.json](s3_manifest.json) pins recovery versions; [manifest.json](manifest.json) pins code/runtime/input identity. Models never enter Git. Small receipts retained under custody. No PR/main/fastlane/CI,foreign checkout/job,forbidden host or serving change.

Replay requires the pinned H310819a45aa5e CPU source siblings,H287 export,host-local SHA-pinned TRAIN inputs and the exact-version archived BASE inputs. The fetch helper uses H316's archived member inventory retained under prior custody; it is also recoverable from the pinned H316 archive. No new scoring is authorized by this closeout.

Next: Do not advance CNORM or score dev91. A future raw-utility L2 control would be needed to isolate normalization from the necessary loss-family change; no additional experiment is authorized or run here.
