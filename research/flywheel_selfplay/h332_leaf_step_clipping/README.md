# H332: leaf step clipping

**COMPLETE — CLIP_NOT_ADVANCED**, TRAIN-only shadow research. [PREREG.md](PREREG.md) was published at **b97a4f0ba7ec7da545d50a3584d80a6b8208cb74** before any fit. DOPE: `{"composite_sha256_prefix":"c51a56d0","loaded":false,"scope":"CPU selection layer only"}`.

## Hypothesis and setup

Capping each ranker's raw leaf output may reduce the influence of high-spread TRAIN families while retaining useful rankings. The only intervention is ranker `max_delta_step=1.189812963969407`. The proposed gradient mechanism is not independently measured here.

The cap was fixed before fitting as the linear-interpolated 90th percentile of `abs(native leaf_value / native model learning_rate)`: **11,165,008 leaves from 11,664 archived BASE inner-fold main and eight-bag ranker model instances**. All 36 outer tasks' 12 inner stages are pooled, excluding vetoes, final refits and NORM. Identical model hashes retain their archived multiplicity. Zero/stump leaves are included; all leaves are finite and checked against native leaf counts. [cap_config.json](cap_config.json) pins source exact version/SHA, all 36 inner archive SHAs, leaf-pool SHA and [cap_models.json](cap_models.json) inventory SHA. Every contributing model SHA and path is also inline in the immutable preregistration. This global historical TRAIN-derived cap collectively uses models exposed to every TRAIN family: **it is not strictly outer-independent new hyperparameter selection or fresh confirmation**. This exposure does not relax the fit workers' physical outer-feature closure.

BASE is unchanged H303 NESTED/H308 P0/P1/P2 × roots 30001–30003 × four family-grouped outer folds, 1219 TRAIN contexts/205 families, r882 top-eight slots, three members/root and eight bags/member. Exact-version GET plus whole/member SHA verified H315/H316 frozen BASE arrays/locks. Regression before CLIP fits passed all nine panels, 36 fold-root pick/lock hashes and exact medians .0643163334285782/.04821480030677507/.04360142642936269. BASE refits/new model scores: zero. All 1219 TRAIN pools, four consumed feature prefixes, selected metadata and r882 TRAIN costs match archived BASE hashes.

Only pinned H287 `topk.fit_ranker` and `bagged.fit_ranker_n` gain `max_delta_step=c`, covering inner rankers, final main refits and all eight bags. Removing that keyword restores both exact BASE ASTs. Ordinal labels 0–7 and implicit default `label_gain` remain; effective first-eight gains are **[0,1,3,7,15,31,63,127]**. No H328 linear-gain or H325 extra-trees/extra-seed override is present. LambdaRank/ndcg, truncation, learning-rate/tree-count search, selected-round rules, other numerical parameters, features, seeds, folds, 18-config tau/k/agreement grid, raw inner objective, conservative 1e-4 ties, switch rules and invalid utility −2.0 are unchanged. Veto constructors receive no clipping keyword; all 108 final veto SHAs equal archived BASE. All 36,504 constructor records and 14,040 saved native models passed review, including exact constructor caps, native six-significant-digit cap representation, default gains and actual unshrunk ranker leaf bounds.

Preflight/regression completed before fit workers. Each worker loaded only its outer-train pools/costs and exact per-context cache ranges; no inherited full packet cache. Identifier/offset metadata may be read. Scoped hooks reject physical outer contexts and archived BASE outcomes. All 36 fit locks and directories were fsynced before fresh-process outer scoring, once per fold/root with STARTED/COMPLETE markers. Physical outer feature reads/prebuilt outer packets/guard denials before lock: zero. DL-Val feature rows: zero. No partial set, dev91, DL-Test, sealed/holdout/meta-test/fresh cohort or live checkpoint use.

## Outcome

Median across nine cells of family-equal mean gain: BASE **+0.048215**, 95% CI [0.025215309703455147, 0.07580261125590157]; CLIP **+0.043796**, CI [0.023163685944061967, 0.06816936576246603]. Literal family-median gain: BASE +0.003624, CI [0.0002989597199160865, 0.007721612038073727]; CLIP +0.005132, CI [0.0016841344632589905, 0.009153508659323073]. The preregistered retention gate uses the former.

Paired drop-top-10 median CLIP−BASE **-0.001169**, 95% CI **[-0.009111483176937434, 0.005026674768422494]**, one-sided LCB95 **-0.007803**. Paired whole-family bootstrap: 10,000 draws, RNG 332191; the same family draw across both arms/all nine cells, with each arm re-dropping its own ten best on every draw. Per-root medians across three partitions: [-0.0011690346747413975, -0.0024373900444478493, 0.0002440957148080146]. ADVANCE requires paired median ≥ +.002 and LCB95 > 0, full-gain retention ≥90%, no median harmed-share increase, and positive paired median on at least two roots. Ties never advance. Gates: `{"gain_retention": true, "harmed_nonincrease": false, "paired_drop_top10_at_least_002": false, "positive_on_two_seeds": false}`.

Median harmed-family share (gain <−.01): BASE **9.268%**, CLIP **10.732%**. Config matches **7/36**, CONFIG_SENSITIVE (descriptive threshold 30/36; not an additional decision gate). Median Jaccard of root pairs' switched-context sets: BASE 0.441805, CLIP 0.473214.

Median Elo: BASE **1725.66**, CLIP **1722.49**. **reference = r882 slot0 (DL-Train outer); r3175 not measured on TRAIN**. Family W/T/L tie tolerance 1e-4.

| Part | Root | Arm | Family mean | Family median | Drop10 | W/T/L | Harmed | Invalid | Switches | Elo |
|---|---:|---|---:|---:|---:|---|---:|---:|---:|---:|
| P0 | 30001 | BASE | +0.041289 | +0.000000 | +0.013174 | 88/93/24 | 6.341% | 13 | 219 | 1725.66 |
| P0 | 30002 | BASE | +0.064316 | +0.001588 | +0.008633 | 108/67/30 | 8.780% | 14 | 300 | 1722.48 |
| P0 | 30003 | BASE | +0.072258 | +0.006984 | +0.016621 | 122/50/33 | 9.268% | 11 | 320 | 1727.10 |
| P0 | 30001 | CLIP | +0.055364 | +0.000620 | +0.008648 | 103/73/29 | 8.293% | 13 | 240 | 1720.13 |
| P0 | 30002 | CLIP | +0.036039 | +0.007367 | +0.009697 | 122/51/32 | 11.220% | 15 | 305 | 1732.44 |
| P0 | 30003 | CLIP | +0.070218 | +0.003297 | +0.019639 | 115/60/30 | 8.293% | 14 | 294 | 1733.39 |
| P1 | 30001 | BASE | +0.048215 | +0.003000 | +0.010198 | 112/57/36 | 11.220% | 17 | 327 | 1697.13 |
| P1 | 30002 | BASE | +0.048658 | +0.003624 | +0.014063 | 116/51/38 | 10.244% | 15 | 275 | 1693.84 |
| P1 | 30003 | BASE | +0.038294 | +0.003253 | +0.012702 | 112/64/29 | 8.293% | 16 | 285 | 1734.68 |
| P1 | 30001 | CLIP | +0.036030 | +0.001457 | +0.014953 | 106/69/30 | 8.293% | 15 | 255 | 1719.23 |
| P1 | 30002 | CLIP | +0.043796 | +0.006226 | +0.011626 | 118/55/32 | 10.732% | 15 | 320 | 1726.65 |
| P1 | 30003 | CLIP | +0.055372 | +0.005804 | +0.012946 | 120/49/36 | 10.244% | 17 | 289 | 1709.12 |
| P2 | 30001 | BASE | +0.043601 | +0.006326 | +0.010678 | 122/48/35 | 10.732% | 16 | 315 | 1716.88 |
| P2 | 30002 | BASE | +0.031840 | +0.003624 | +0.008774 | 112/65/28 | 7.805% | 17 | 292 | 1740.78 |
| P2 | 30003 | BASE | +0.054721 | +0.008145 | +0.023605 | 129/41/35 | 10.732% | 15 | 335 | 1726.57 |
| P2 | 30001 | CLIP | +0.036934 | +0.005132 | +0.009509 | 119/53/33 | 10.732% | 14 | 299 | 1722.78 |
| P2 | 30002 | CLIP | +0.030175 | +0.004783 | -0.000394 | 116/48/41 | 13.171% | 17 | 361 | 1680.64 |
| P2 | 30003 | CLIP | +0.052642 | +0.009603 | +0.004496 | 126/44/35 | 14.146% | 13 | 426 | 1722.49 |

[metrics.json](metrics.json), below 50 KB, retains all 18 arm/cell records, nine paired differences, raw mean/median CIs, drop1/5/10, LOFO minimum, W/T/L, harm, invalid/slot0-invalid, switches/rates, Elo CIs, all root pairs, configs/agreement, 72 lock hashes/18 pick hashes, regression and execution evidence. No cell/family was filtered.

## Verification, resources and custody

27 targeted tests passed; zero failed/skipped, exit 0. Coverage includes both inverse ASTs, sole keyword change, unchanged labels/groups/eval/rounds, default gains/no extra-trees, veto nonmutation, estimator signature/read-only observer, invalid or duplicate cap rejection, native leaf parser/inverse learning rate/cardinality, fixed linear percentile and nonfinite refusal, identity/member-scope cleanup, paired bootstrap and four advance gates, Jaccard, physical outer/BASE outcome refusal, poisoned-tail exclusion and missing-lock refusal. No synthetic optimizer fit. Source syntax and diff checks passed before launch. Actual model exit: `exited/0/success`, invocation `193d13d1d31b4d32b3a80b7f3c61052e`, captured by ExecStopPost.

CPU xbabe2 only, nice 19, numeric threads 1, four cores 13/15/60/62 and fresh ≥85% idle checks on their SMT peers 77/79/124/126, disjoint from H331/gov03. MemoryMax 6 GiB, swap 0. Sampled maximum host CPU 24.43%, load 33.89, minimum MemAvailable 70.65 GiB and maximum scratch 1203098147 bytes. Fixed compute stop 22:15Z and close 22:30Z Nov 4; no reset, extra arm, cut cells or posthoc protocol change. No GPU/qualification.lock, Cargo/native build or GitHub CI. DOPE weights unloaded/unchanged; cached TRAIN CPU features only.

Inner archives were create-only uploaded and exact-version GET/SHA verified before raw inner-copy removal. The final source/models/locks/results archive is streamed create-only to H332 S3, with exact-version whole SHA plus every member verified before enumerated own scratch deletion. [s3_manifest.json](s3_manifest.json) pins versions. Full native/input/member inventories and leaf pool are archived; source and small receipts remain under `/home/ubuntu/jope-rescue-20260921/h332-solhwv`. Models never enter Git. Only nonforce origin/v1/next publication and the two assigned index rows are used; no PR/main/fastlane/CI, other-owner checkout/job, serving or forbidden-host action.

Replay requires pinned H310 CPU sources/H287 export, SHA-pinned TRAIN inputs and archived BASE exact versions. Closeout authorizes no extra fit or score. Next: do not advance CLIP or run dev91; preserve the complete negative result.
