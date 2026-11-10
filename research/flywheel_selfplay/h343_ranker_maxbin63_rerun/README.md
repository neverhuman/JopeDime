# H343 — Ranker max_bin63 rerun

**MAXBIN_NOT_ADVANCED — TRAIN-only research.** BASE median gain +0.048214800, 95%CI [0.024969946987417387, 0.07605823841670331]; MAXBIN63 +0.044898723, CI [0.020027069172323112, 0.07612229329055509]. Paired arm-specific drop-top10 median -0.004715048, CI [-0.011361583866282677, 0.002812350396857761], one-sided95LCB -0.010072017; per-root paired medians [-0.004715047920117343, -0.0027383116690845267, -0.008875901773964807]. Criteria {'seed_consistency': False, 'paired_drop10_margin': False, 'gain_retention_90percent': True, 'harmed_nonincrease': False}; failed ['seed_consistency', 'paired_drop10_margin', 'harmed_nonincrease']. Harm rate BASE 0.092683, treatment 0.107317. Config agreement 0.138889, CONFIG_SENSITIVE. Median Elo BASE 1725.66, treatment 1715.80.

## Hypothesis and inherited implementation

Coarser feature histograms may regularize noisy split thresholds. H341 closed
INCOMPLETE_CPU_ADMISSION with zero scientific fits; that result left this arm
unmeasured. H343 copies H341 prereg f11e106f16d9650ab56e2a02eb8675ecf7e74e56 and
implementation98c3c6f7531e8de5838c9e33f9b79ff89feb7c3f. Only wave identities/paths,
RNG343191, assigned cores and fixed window changed; no scientific treatment change.
[Copy identity receipt](source_copy_receipt.json), [PREREG](PREREG.md).

Archived BASE972rankers and108veto models all record native max_bin255, with no
per-feature override. [Native BASE audit](base_native_maxbin.json). Only max_bin63
is added to pinned H287213b30ad52e1ce48927bba960ed609bd98ab4dfe topk.fit_ranker and
bagged.fit_ranker_n, covering inner/final/all8bags. Removing that keyword restores
exact BASE ASTs. Complete constructor dictionaries must differ only at max_bin.
[Implementation](regularization.py), [AST receipt](regularization_transform_preview.json).

Each fit constructs a new Dataset from raw NumPy rows: no reference, initially
absent native handle, max_bin63 and native feature bin counts<=63. Unique
process/sequence identity is recorded; pre-binned Dataset inputs are rejected.
The observer delegates to the original constructor and is restored before veto
fits. Native max_bin/LR/round budget/model hashes are checked. No BASE-bin reuse.
The cap does not guarantee fourfold fewer realized thresholds or faster fits.

LambdaRank, ordinal labels/defaultlabel_gain, leaves7/depth3,L2=1,minleaf20,
subsample/colsample.8, original LR/n/seeds/features/folds, veto bins255,
18-config grid, selected prefixes, ties/switches and utility-2.0 remain BASE.
No H339 LR halving/doubled rounds, callbacks, weights or other intervention.
DOPE weights were not loaded or modified:
{"composite_sha256_prefix":"c51a56d0","loaded":false,"scope":"CPU selection layer only"}.
No neural optimizer update, live-loop qualification or serving authority.

## Preregistration and validation

H343 prereg **a3e67a1d3cce4b9666b39543319019436a81cc43** preceded every fit, including fixture fits;
reviewed implementation **5d1af6779b78df31df311bd2d33420c926fe853b** preceded scientific fitting.
Fresh exact-version BASE recovery and replay matched36fold/root hashes,nine
pick/raw-utility panels and medians .0643163334285782/.04821480030677507/
.04360142642936269 exactly. [Regression](regression_receipt.json),
[input recovery](input_receipt.json), [vendor pins](vendor_pins.json).
No BASE scientific refits. Mandatory BASE replay reads TRAIN outer rows separately;
zero prelock treatment rows does not describe those reference-verification reads.
Each treatment worker physically admits only its outer-TRAIN pool/cost/cache byte
ranges and excludes archived BASE outcomes. No outer packets prebuilt.

32 controls passed with zero failures/errors/skips:10native-bin/Dataset/AST/
forecast,4physical-boundary,18analysis. Two synthetic ranker fits and one unchanged
synthetic gate fit; BASE synthetic Dataset constructed without fitting.
[Tests](tests_receipt.json). Review and independent recalculation are by the same
implementing agent, not an independent reviewer. Hosted CI NOT_RUN under freeze.

Initial prereg push lost a concurrent nonforce branch race. The early replay
stopped at the missing publication receipt before data loading or predictions.
Both original failures are preserved. Fetch/republication and regression2 then
succeeded before all fits; no source, gate, deadline or scientific restart.
[Correction receipt](preflight_publication_correction.json).

## Metrics and frozen decision

TRAIN1219contexts/205families,H303NESTED/H308 P0/P1/P2 × roots30001–30003 ×4
family-grouped outerfolds. Context means within families, equal family weight,
median9cell mean gains; literal familymedians separately. W/T/L tolerance1e-4,
harmgain<-.01, invalid/switch counts contexts. Drop largest1/5/10 gains separately
per arm; LOFOmin equalsdrop1.10kpaired whole-family bootstrap RNG343191 shares
205family indices across botharms and allcells. Advance iff paired arm-specific
drop10median>=+.002 ANDone-sided95LCB>0 ANDgain>=90%BASE ANDharm<=BASE ANDatleast
2of3 rootpairedmedians>0. Exact+.002 inclusive; zeroLCB/tiedroots fail strict gates.
Configagreement and seedJaccard diagnostic only; no post-outcome gate changes.
Reused TRAIN/adaptive wave search/dependentcells limit interpretation; bootstrap
conditions on fixed fits and does not repeat selection or correct wider search.
**reference = r882 slot0 (DL-Train outer); r3175 not measured on TRAIN**.
Elo inherits400-step Bradley-TerryMM with pseudocount.01 and ties omitted.
No AUC or speed verdict. Full metrics and per-root/partition intervals in
[metrics.json](metrics.json).

All36 durable locks and29484 ranker-fit records audited before36 once-only outer scores.972 final ranker native files and108 unchanged veto hashes verified. [Lock review](lock_review.json), [score receipt](score_receipt.json), [numerical review](numerical_review.json).

| Arm/cell | Mean gain | Family median | 95%CI | W/T/L | Harmed | Invalid | Switches | Drop1/5/10 | LOFOmin | Elo |
|---|---:|---:|---|---|---:|---:|---:|---|---:|---:|
| BASE P0/30001 | +0.041289223 | +0.000000000 | [0.021865411775897167, 0.06413335621337] | 88/93/24 | 13 | 13 | 219 | {'1': 0.03448239888339116, '5': 0.02290534754008244, '10': 0.013173556747221869} | +0.034482399 | 1725.66 |
| BASE P0/30002 | +0.064316333 | +0.001588084 | [0.015448361878060856, 0.1323222860172932] | 108/67/30 | 18 | 14 | 300 | {'1': 0.03710010271431613, '5': 0.018969760868246758, '10': 0.008632837099038627} | +0.037100103 | 1722.48 |
| BASE P0/30003 | +0.072257657 | +0.006984224 | [0.02268092106636732, 0.14111750389081848] | 122/50/33 | 19 | 11 | 320 | {'1': 0.045081165435168, '5': 0.02751377343114538, '10': 0.01662054284317023} | +0.045081165 | 1727.10 |
| BASE P1/30001 | +0.048214800 | +0.002999623 | [0.018322045282710295, 0.092174627142667] | 112/57/36 | 23 | 17 | 327 | {'1': 0.030657067922232554, '5': 0.018310872928509914, '10': 0.010197529908285808} | +0.030657068 | 1697.13 |
| BASE P1/30002 | +0.048657946 | +0.003624258 | [0.02156282964940977, 0.08350119534940964] | 116/51/38 | 21 | 15 | 275 | {'1': 0.035309424528769756, '5': 0.02333957957394522, '10': 0.014063351613976904} | +0.035309425 | 1693.84 |
| BASE P1/30003 | +0.038294238 | +0.003253142 | [0.015797007773609033, 0.061123369028697805] | 112/64/29 | 17 | 16 | 285 | {'1': 0.03333251212019234, '5': 0.020954292497029573, '10': 0.012701994683873705} | +0.033332512 | 1734.68 |
| BASE P2/30001 | +0.043601426 | +0.006326451 | [0.017916490087937362, 0.07205858925712028] | 122/48/35 | 22 | 16 | 315 | {'1': 0.03727988130081795, '5': 0.021223015140835293, '10': 0.010678474603967304} | +0.037279881 | 1716.88 |
| BASE P2/30002 | +0.031839978 | +0.003624258 | [0.010470661940133269, 0.05380307027608112] | 112/65/28 | 16 | 17 | 292 | {'1': 0.026501549839943208, '5': 0.01488744333658581, '10': 0.008773682894185859} | +0.026501550 | 1740.78 |
| BASE P2/30003 | +0.054720630 | +0.008144684 | [0.03314014066792383, 0.07967002980051707] | 129/41/35 | 22 | 15 | 335 | {'1': 0.04859531966148308, '5': 0.03399337142658264, '10': 0.02360528095118402} | +0.048595320 | 1726.57 |
| MAXBIN63 P0/30001 | +0.035281433 | +0.000000000 | [0.007983191004455709, 0.061439002719478164] | 97/80/28 | 17 | 14 | 213 | {'1': 0.028993894527076226, '5': 0.01681398987646828, '10': 0.006733590732243274} | +0.028993895 | 1715.80 |
| MAXBIN63 P0/30002 | +0.053296773 | +0.006551669 | [0.011387942860597638, 0.10744134688595468] | 115/57/33 | 22 | 15 | 306 | {'1': 0.032419762167812664, '5': 0.017101785629722248, '10': 0.0058945254299541} | +0.032419762 | 1716.84 |
| MAXBIN63 P0/30003 | +0.044898723 | +0.000000000 | [0.008561711296807922, 0.08945913477543667] | 102/73/30 | 21 | 15 | 239 | {'1': 0.027976188416394484, '5': 0.015015565621513849, '10': 0.004622548153354772} | +0.027976188 | 1712.55 |
| MAXBIN63 P1/30001 | +0.028494187 | +0.000000000 | [0.006155294062579589, 0.0495875180768497] | 95/78/32 | 19 | 15 | 221 | {'1': 0.023982255845602957, '5': 0.0143276891114105, '10': 0.005482481988168465} | +0.023982256 | 1688.99 |
| MAXBIN63 P1/30002 | +0.054728528 | +0.000000000 | [0.028636031746335618, 0.08752033183249082] | 102/65/38 | 22 | 13 | 311 | {'1': 0.04359515149000363, '5': 0.028555626532590406, '10': 0.017609800663534743} | +0.043595151 | 1671.50 |
| MAXBIN63 P1/30003 | +0.050279592 | +0.004782505 | [0.02512911489982444, 0.08205826463606088] | 116/57/32 | 21 | 16 | 289 | {'1': 0.039124407152565426, '5': 0.024982416221323456, '10': 0.013266777678057772} | +0.039124407 | 1723.68 |
| MAXBIN63 P2/30001 | +0.048869909 | +0.008422447 | [0.01778514360260619, 0.08436368339817184] | 134/36/35 | 24 | 17 | 367 | {'1': 0.037694141439999426, '5': 0.021763283052856876, '10': 0.011426212613972085} | +0.037694141 | 1733.18 |
| MAXBIN63 P2/30002 | +0.029507067 | +0.000000000 | [0.0012885077988014878, 0.0587763279536136] | 101/69/35 | 24 | 16 | 286 | {'1': 0.02325928353782068, '5': 0.008575227576723531, '10': -0.003350853200671175} | +0.023259284 | 1684.07 |
| MAXBIN63 P2/30003 | +0.040519907 | +0.006586144 | [0.022157627490137933, 0.06076907516390958] | 124/48/33 | 23 | 16 | 307 | {'1': 0.0356956338205732, '5': 0.024790266905962862, '10': 0.014729379177219212} | +0.035695634 | 1729.92 |

## Resource admission and custody

ONE5second per-core+SMT>=85%idle probe admitted [24, 25, 26, 28] from assigned
21/22/23/24/25/26/28/30/32/33; first at most8, minimum4. No retry or borrowed cores.
[Admission](cpu_admission.json). Before scientific fits, the fixed capacity screen
used H339's1059.808s firstcell as a conservative budget input, ceil(36/workers)*
elapsed*1.15+600. It is not a speed estimate for max_bin63.
[Admission forecast](admission_forecast.json). H343's own first completedcell,
including verified model custody, took 576.613s;
remaining35 rule ceil(35/workers)*elapsed*1.15+600 yielded
2026-11-05T01:29:27.746335+00:00, admit=True.
[First-cell forecast](runtime_forecast.json). No cut cells/partial scores/reset.

CPUxbabe2nice19/math1,MemoryMax6GiB/Swap0,RSS<=6GiB,CPU/load<=85,
MemAvailable>=20GiB,scratch<=2GB; fixed03:15compute/03:30closeNov5.
Peak sampled RSS 2,921,725,952bytes, hostCPU
42.4%; resource guard violations none.
[Resource receipts](resource_receipt.json), [dependencies](dependency_pins.json).
H335/H338/H342 reserved CPU closures and actual worker affinities were excluded;
sleeping H335 launch supervisor was classified separately with descendant checks.
Foreign jobs were not changed.

All 36 model archives create-only exactVersionGET/wholeSHA/everymember
verified before inner-file reclamation. [Model index](model_archive_index.json).
Remaining evidence under s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H343/astra-20261104T232500Z/.
[Manifest](manifest.json), [S3 receipt](s3_manifest.json). Every own scratch file
requires a verified recovery mapping and no consumer before deletion. Original
local H50/H58/H276 S3versions remain unestablished; byte-range/packet hashes retained.
No models in Git. NoGPU/qualification.lock/dev91/DLVal/DLTest/sealed/holdout/
meta_test/fresh/xbabe0/3/livewriter/foreignjobs/PR/main/fastlane/hostedCI actions.

## Cross-critique and next step

H341 capacity failure and H339 runtime failure are not negative scientific results.
H338 closed COLSUB_NOT_ADVANCED at44bcbe16dc0cc31cb05c5bfeedd35e34697995b1:
paired drop10 -0.002449665, LCB95 -0.007121461; harm, paired-margin and root
consistency criteria failed. Its explicit feature_fraction_seed also changes the
seed used by existing tree-level sampling, so it does not isolate node sampling.
H338 feature subsampling and H342 Hessian regularization test different factors;
their outcomes cannot substitute for H343. H342 explicitly derives its scalar globally
from archived TRAIN-model leaf Hessians, not inside each outerfold; those Hessians
depend on fitted margins/labels, so the scalar carries historical TRAIN information
across folds even without outcome-based tuning. Its physical worker isolation does
not erase that prior derivation; this is a protocol limitation, not a negative result.
Historical H328 failed harm, not its
root-consistency gate. Neither a train/validation gap nor a smaller bin cap alone
identifies threshold overfitting as the cause of a live learning plateau.
[Published peer-source identities and interpretation](peer_crosscritique.json).
Do not advance this fixed arm to dev91.
