# H371: NL15 ranker path_smooth20

**CLOSED — INCOMPLETE_CPU_ADMISSION. Scientific verdict: NOT_EVALUATED.**

The single five-second admission sample admitted only **24, 25, 26**; the required minimum was four. The controller refused before fitting. No second sample, borrowed cores, reduced design, restart or partial outer scoring was attempted. This is an operational refusal; it is not PS20_NOT_ADVANCED. Frozen NL15 remains retained.

| Record | Observed |
|---|---|
| Qualifying cores | 3; minimum4 |
| Same-interval host CPU | 24.095847% (limit70%) |
| Admission samples / retries | 1 / 0 |
| Treatment constructors / native fits | 0 / 0 |
| Fit locks / outer-score markers | 0 / 0 |
| Required tasks not started | 36 / 36 |
| Initial / first-task forecast | NOT_RUN / NOT_RUN: admission refused before these stages |
| PS20 gains, harm, CIs, Elo, configuration agreement | NOT_EVALUATED; unknown, not zero |

[PREREG.md](PREREG.md) and all35source files were published at `74eb11380472134d23acb1dc2261bd68034d4ec5` before the controller launch. The source came from committed H361 `cb99837875ad0daef8a80eb42eceeb056f7acbc2`, anchored to H352 `02dfee2f7e0b8543324494dd4f99d074cd6b93b6`. All H361 LRH learning-rate and fit/predict/lock iteration changes were removed. Original H352 LR grid (0.03,0.05,0.10), rounds (50,100,200), and main200 were restored. The sole prospective intervention adds path_smooth20 to both NL15 ranker constructors, including inner/final/all8bags; full inverse ASTs restore H352. No new seed keyword. Labels, groups, seeds, features, sampling, leaves15/depth3, original regularization and veto stay fixed.

Fresh exact-version GET, whole-object and every-member hashes passed for BASE and H352 NL15. BASE9panels/36fold-root pick+lock pairs and exact medians0.0643163334285782/0.04821480030677507/0.04360142642936269 matched. NL15 all27parts/2691members,9panels/36locks+COMPLETEscore markers,972rankers/108veto identities and all36504native-fit reference headers passed; ranker path_smooth was explicitly0. No BASE/NL15 refits or new baseline predictions. Current TRAIN identities1219contexts/205families and4consumed cache prefixes matched;0DL-Val feature rows.

All48 data-free source controls passed,0failed/skipped,0real LightGBM fits,0model predictions,0scientific feature reads and0capacity probes. These establish source/fixture behavior only; actual PS20 model behavior, numerical quality and runtime qualification were NOT_RUN. Treatment native/veto equality checks and all scientific decision gates remain unevaluated.

`reference = r882 slot0 (DL-Train outer); r3175 not measured on TRAIN`

The preregistered primary comparator is NL15, secondary BASE, with10kwhole-family RNG371191 bootstrap and per-root CIs. ADVANCE requires paired drop10median>=+0.002 AND LCB95>0, gain>=90%NL15, harmed<=NL15 and >=2/3positive root points. No gate or bootstrap was evaluated. Sequential exploratory sweep H348-H371; no multiplicity-adjusted confirmation, AUC or speed claim. No dev91 evaluation was triggered.

Actual campaign invocation `0290fd7fc5a84da7984a76d84bfb067e` ended with application exit2 at 2026-11-05T14:19:21.869539+00:00. The zero-fit refusal and custody unit exited0. Resource budget: xbabe2 CPU, nice19/math1/MemoryMax6GiB/SwapMax0/CPUQuota800%, pool15-26+sibs79-90. Qualifying pair minima24:89.92%,25:85.27%,26:88.75%; all others failed the85% gate. Corrected guard: CPU>85 averaged60s OR load1>108 on3consecutive60s samples OR MemAvailable<20GiB OR scratch>2GB; old instantCPU/absolute-load85 stop rules retired. Fixed16:45Zcompute/17:00Zclose Nov5 was not reset.

Create-only main archive `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H371/gov03-closeout-20261105T142617Z/h371-scratch.tar`; VersionId `EAgAobcfUkIM3VbjOzXy.nd.T78vWLjP`; SHA256 `4362853699c420fab406245c24fd1b254b6bc6ad2ae4e2b5448d3f4872f26519`. Exact-version GET/whole SHA/all1386members verified. There are zero generated treatment model components; the archive retains the source, frozen baseline copies, regression/control/admission/resource receipts and the empty component manifest. Own scratch177,376,294bytes and generated restore were removed only after verification and a fresh zero-consumer check. Immutable local main archive/source/receipts retained at `/home/ubuntu/jope_plan/evidence/H371/gov03-20261105-t1400`.

DOPEc51a56d0 unchanged/unloaded, CPU selection layer only. No protected/sealed/DL-Val/dev91/DL-Test payloads, GPU/qualification.lock, writer/AUTORESTART mutation, forbidden hosts or foreign job signals. Canonical HEAD/index/62foreign staged whitepaper files and other dirty files preserved. Private index/non-force origin/v1/next only; no PR/main/fastlane/hostedCI or model weights in Git. A new treatment attempt requires a separate owner-authorized wave; H371 will not be reprobed or resumed.
