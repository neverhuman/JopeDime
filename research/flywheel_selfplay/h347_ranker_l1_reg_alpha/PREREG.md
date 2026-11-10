# H347: ranker L1 regularization

Owner sol_v1_gov03; authority hw_t0200_gov03_h347_l1alpha.md (2026-11-05).
H342 is CLOSED and RELEASED. Copy the committed H342 harness from origin/v1/next,
source commit167a527e2889b6b4e4be69736e03a0507bd43f4f, with file SHA inventory.
Restore BASE defaults and change only one ranker constructor keyword. Publish
this preregistration nonforce to v1/next BEFORE any H347 treatment fit.

## Hypothesis and fixed delta

L1 leaf-output regularization may reduce sensitivity to small noisy corrections.
This is a hypothesis, not an observed robustness or speed result. ONE arm L1:
add reg_alpha=0.5 only to topk.fit_ranker and bagged.fit_ranker_n, pinned H287
213b30ad52e1ce48927bba960ed609bd98ab4dfe. BASE omits that keyword. All972archived
BASE rankers confirm effective native lambda_l1=0, min_sum_hessian_in_leaf=.001,
sigmoid1.0; zero calibration fits or model predictions. BASE !=0 would stop
INCOMPLETE_SPEC_MISMATCH before treatment; no alternate value or tuning.

Stripping the one reg_alpha override restores both complete BASE function ASTs.
No lambda_l1 alias is added. Keep LambdaRank/ndcg/sigmoid1.0, original labels0..7,
implicit label_gain, min_child_samples20,min_child_weight.001,leaves7,depth3,
subsample.8/freq1,colsample.8,reg_lambda1, original LR/tree options/seeds/features,
veto binary GBDT,18-config grid,ties/switches1e-4,and sentinel-2.0. No H342 Hessian
floor9.87 or other intervention. Original200round topk,LR(.03,.05,.10), prediction
counts50/100/200 and original selected bag fit counts remain; no early stopping.
Audit every actual native fit against all original constructor fields. Ranker
native lambda_l1 must.5; veto stays at its pinned lambda_l1=1 and lambda_l2=2. Every final veto file must match its exact
BASE SHA:108/108 across36tasks. A mismatch blocks outer scoring. [base_veto_sha256.json](base_veto_sha256.json)
contains the fixed expected identities from the same verified archived BASE.
Expected per task819ranker+195veto fits and351ranker+39veto saved models; these
are expectations until actual execution. No models in Git.

Preparation note: the first read-only metadata verifier applied ranker L2=1 to
veto models and exited1 after BASE regression/input checks passed. Before any
treatment fit or capacity probe, the check was scoped to the exact pinned veto
constructor (reg_alpha1/reg_lambda2); the corrected read-only native check exited0.
Original failed-command/source receipts remain preserved. Data, models, value
rule and scientific fit code were unchanged by this verifier repair.

## Regression first and physical input closure

Unchanged H303 NESTED/H308 code819a45aa5e30e662ab511cd9962b85b2b8c1e44a.
P0/P1/P2 x roots30001-30003 x4 family-grouped outer folds =36fit tasks/9cells,
1219TRAIN contexts/205families. First science gate PASS: exact-version frozen
BASE GET read-only, whole-object and selected-member SHA verification;9/9panels,
36fold/root pick+lock hashes and medians .0643163334285782/.04821480030677507/
.04360142642936269 exact. No BASE refit or fresh BASE model predictions.

Same H342 source archive identities: H316 key
experiments/rd_breakthrough_20261030/H316/solhwv-20261104T105633Z/h316_models_locks_outputs.tar.gz,
VersionId W5jY6iZlwoRszA7P9zfNLLMtRtM9ZZkW,
SHA256 4eae1dec792e1c7b4adff9ff9a5226d0c38cfe1bd9605ba07cc2596f4ad7b0b5.
H315 picks key experiments/rd_breakthrough_20261030/H315/picks.npz,
VersionId jCll6z9FGwmpNg.cocb305LcVzFYuieR,
SHA256 e9e6b16a7381ff14191d2a4d31ce31d4b623c39297e7383ee6aeb871e08ca615.
Current1219TRAIN pool hashes/selected/r882costs and4consumed mixed-cache TRAIN
feature prefixes must match archived inputs. No DL-Val feature rows.

Preparation ends in a separate process before admission. Fit parent has metadata
only; each worker opens only outer-TRAIN pools and exact permitted cache ranges.
Audit rejects physical outer features and BASE outcomes/native copies during fit.
Require physical outer rows/packets before fsynced fit_lock and parent fsync0.
All36locks/guards/native model audits/108BASE veto SHA matches must pass before
fresh once-only outer scoring. STARTED/COMPLETE markers prevent repeats. No partial
scoring, fit restart or repeated completed scores.

## Metrics and decision

Raw utility against r882slot0, paired with frozen BASE; exact label:
`reference = r882 slot0 (DL-Train outer); r3175 not measured on TRAIN`.
Report family-equal mean and literal family-median gain/whole-family bootstrapCI,
drop-top1/5/10,LOFO minimum,W/T/L,harmed gain<-.01,invalid,switches,per-root
medians,Elo/EloCI,seed Jaccard and configuration agreement. Ordinary intervals
10000draws/RNG302150 unchanged. No AUC or speed verdict.

ADVANCE iff paired(L1-BASE) drop-top10 median over9cells>=+.002 AND one-sided
LCB95>0; L1 median full gain>=90%BASE; median harmed<=BASE; and positive per-root
paired drop10 medians across3partitions on>=2/3roots. Otherwise L1_NOT_ADVANCED,
only after complete scores. Ties/subthreshold never advance. Paired bootstrap:
10000whole-family draws/RNG347191, same family resample both arms/all9cells,
each arm redrops its own top10; statistic median9cells; LCBquantile.05.
Any one-shot dev91/r3175 comparison requires separate authorization, outside H347.

## One admission interval, two forecasts and immutable window

ONLY H347 H-wave treatment runs; leave H345R archive and H335/foreign jobs alone.
xbabe2 CPU only,nice19,numeric1,MemoryMax6G/MemorySwapMax0. Candidate physical
cores0-14 and27-33 plus SMT64-78 and91-97. ONE fresh5-second per-core+SMT>=85%
idle interval; both siblings qualify. Take first<=6passing pairs in candidate
order, minimum4. Require SAME-interval aggregate hostCPU<=70% at admission.
The aggregate/proc/stat sample brackets the same single mpstat interval; it is
not a second core probe. After admission, bind controller threads and all workers
to those admitted cores. No repeated probe/borrowed cores/worker expansion.
Exclude H345R15/19/21/24/25/26/36/37+sibs andH33538-53/102-117.

Full36 pre-fit forecast: H342 actual25034.202014 CPU seconds/(workers*.9)*1.15
+300seconds scoring reserve. L1 multiplier1.0 is a planning assumption. Launch
only if forecast ends strictly before2026-11-05T05:45:00Z. Reforecast exactly
after first completed outer fold/root task: remaining35 x max(H342CPU/36,first
task measured processCPU)/(workers*.9)*1.15+300, added to current time; ongoing
work credit0. Miss=>INCOMPLETE_RUNTIME_FORECAST and stop only own service cgroup.
No cut cells/partial scoring/reprobe/restart/deadline reset.

Fixed compute stop Nov5 05:45Z; fixed close Nov5 06:00Z. Guard own job if hostCPU>85%,
load>85,MemAvailable<20GiB or scratch>2GBdecimal. Scratch only
/home/ubuntu/jope-data/h347-gov03. Create-only S3 experiments/rd_breakthrough_20261030/H347/
inner/final custody: exact-version GET+whole SHA+every-member proof before raw
inner/own scratch cleanup; preserve immutable source/evidence and actual exits.

## Publication and boundaries

Private index/objects; CI_FREEZE + MASTER_MERGE_HOLD, nonforce v1/next ONLY.
No PR/main/fastlane/hosted CI. README/metrics<50KB/code/tests and exactlyone row
each in wave/overnight indices, claim/status/claim+close comms/CLOSEOUT receipts.
No GPU/qualification.lock, DL-Val/dev91/DL-Test/protected/sealed/holdout/meta_test/
fresh-cohort rows, forbidden hosts, live writer or foreign-job operations.
DOPEc51a56d0 unchanged/not loaded; scope CPU selection layer only.
H342/H340/H334/H333/H329/H326 remain closed.
