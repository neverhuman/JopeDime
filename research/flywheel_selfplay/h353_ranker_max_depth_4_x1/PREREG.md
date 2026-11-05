# H353: ranker max_depth 4 on xbabe1

Owner sol_v1_hwv. Authority hw_t0835_solhwv_h353_maxdepth4_x1.md, Nov5 2026.
Source: COMMITTED H350 research/flywheel_selfplay/h350_ranker_reg_lambda_5/ at
52558cc20966bc7f27dcd6c77da04086e0bbc6bd. All27manifest-listed source blobs
were byte-verified. H350R's qualified host-only staging/courier machinery is reused;
its L2R5 treatment is not inherited. H350/H350R remain closed and unchanged.
Publish this new prereg nonforce to v1/next BEFORE any treatment optimizer fit.

## Fixed hypothesis and constructor delta

Depth3 may bind asymmetric interactions even at seven leaves. ONE MD4 arm changes
ONLY the existing ranker max_depth3 to4 in pinned H287 topk.fit_ranker and
bagged.fit_ranker_n, covering all inner, final and eight-bag rankers. Restore BASE
reg_lambda1, implicit reg_alpha0, min_child_samples20, num_leaves7; do not add a
reg_alpha keyword. Setting max_depth back3 restores both entire BASE function
ASTs exactly. BASE native max_depth must3 and num_leaves7, else close
INCOMPLETE_SPEC_MISMATCH before treatment. BASE native L2=1/L1=0 also verified.

Keep original LambdaRank/ndcg/sigmoid1, labels0..7/default label_gain, Hessian.001,
row.8/freq1/column.8, splitgain0, LR/tree options/selected-round rules, features,
original root/member/bag seeds, veto GBDT L1=1/L2=2,18-config grid,tie/switch1e-4
and invalid utility-2.0. No tuning, calibration or synthetic optimizer fits.
Every completed fit's full native params are audited, and every final veto must
match frozen BASE SHA:108/108 before outer scoring. Expected per task819ranker+
195veto completed fits;351ranker+39veto saved models. Counts are expectations
until actual execution. No models in Git; DOPEc51a56d0 stays frozen/unloaded.

Record actual per-tree depth/leaf counts from the existing returned native model
text, without calling fit or predict again. Depth is root-to-leaf edge count;
root-only tree depth0. Traverse native left/right children, reject cycles,
unreachable/duplicate leaves or malformed headers; verify actual depth<=configured
max_depth and actual leaves<=num_leaves. Archive full per-tree [depth,leaves]
lists for every completed fit AND each saved model (including truncated saved
round streams), with compact aggregate histograms in metrics. Audit frozen BASE
structures read-only as a reference. This observational logging does not influence
construction, fitting, selection, predictions or decision and is included in
measured first-task process CPU for forecasting.

## Regression first and input closure

Unchanged H303 NESTED/H308 code819a45aa5e30e662ab511cd9962b85b2b8c1e44a,
pinned H287213b30ad52e1ce48927bba960ed609bd98ab4dfe/sevenvendorSHApins.
P0/P1/P2 x roots30001..30003 x4 family-grouped outer folds =36tasks/9panels,
1219TRAIN contexts/205families. No BASE refit or new BASE model predictions.
Fresh exact-Version S3 GET and whole/selected-member SHA on xbabe2:
H316 key experiments/rd_breakthrough_20261030/H316/solhwv-20261104T105633Z/h316_models_locks_outputs.tar.gz,
VersionId W5jY6iZlwoRszA7P9zfNLLMtRtM9ZZkW,
SHA256 4eae1dec792e1c7b4adff9ff9a5226d0c38cfe1bd9605ba07cc2596f4ad7b0b5;
H315 key experiments/rd_breakthrough_20261030/H315/picks.npz,
VersionId jCll6z9FGwmpNg.cocb305LcVzFYuieR,
SHA256 e9e6b16a7381ff14191d2a4d31ce31d4b623c39297e7383ee6aeb871e08ca615.

Before treatment, reverify all copied bytes and rerun BASE regression ON xbabe1:
9/9panels,36fold-root pick+lock SHAs and medians .0643163334285782 /
.04821480030677507 / .04360142642936269 EXACT. Mismatch=>REGRESSION_FAIL.
Current1219TRAIN pools, selected/r882cost bytes and4mixed-cache TRAIN feature
prefixes must match archived identities. Staging copies uncompressed TRAIN-only
cache containers with identical consumed feature bytes/order and first1219IDs /
offsets[:1220]; no feature tail read or copied. Metadata identifiers may be read.
No AWS credentials or S3 client operations on xbabe1; all AWS access on xbabe2.

Preparation exits before fit; parent has metadata only. Each worker opens only
its outer-TRAIN pools and exact permitted per-context cache byte ranges. Historical
BASE outcomes/native copies are forbidden during fit. Physical outer feature
reads/prebuilt outer packets before locks=0. All36fsyncedlocks/parentfsyncs/guards/
native audits/108vetoSHA identities/verified inner receipts must pass before
once-only fresh-process outer scoring. STARTED/COMPLETE markers prevent repeats.
No partial scoring, fit restart, cut cells or repeated completed scores.

## Metrics and decision

Raw utility vs r882slot0, paired frozen BASE. Exact Elo label:
`reference = r882 slot0 (DL-Train outer); r3175 not measured on TRAIN`.
Report all9panels/18arm records: family-equal mean gain and literal family-median
gain with whole-family bootstrapCI, drop-top1/5/10, LOFOmin, W/T/L,harm gain<-.01,
invalids/switches, per-root paired medians/CIs, Elo/EloCI, config agreement/Jaccard,
actual tree depth/leaf counts. Ordinary intervals retain10000draws/RNG302150.
Paired bootstrap10000draws/RNG353191 uses same family draw in both arms/all9cells;
each arm re-drops its own best10; primary statistic median9cell differences.
Root CIs use median3partitions on the same resamples. CI95 quantiles.025/.975;
one-sidedLCB95 quantile.05. ADVANCE iff paired drop10median>=+.002 ANDLCB95>0,
MD4median fullgain>=.9BASE, median harm<=BASE, positive root paired median on>=2/3.
Ties/subthreshold never advance. Else MD4_NOT_ADVANCED only after complete scores.
Incomplete scientific outcomes remain UNKNOWN/null; no bootstrap on partial cells.
No AUC or speed verdict. Any one-shot dev91/r3175 follow-up is outside H353.

## Fixed host split, resources and admission

xbabe1 scratch /home/ubuntu/jope-data/h353-x1 <=2GBdecimal; all computation CPU-only.
nice19,numeric1,systemd h353-x1-run,MemoryMax6G/SwapMax0. ONE5sec admission on
priority24..35 plusSMT88..99, both siblings>=85%idle; first<=10passing physical
cores/min4, SAME-interval aggregate hostCPU<=70%. No reprobe/borrowed cores or
worker expansion. All worker/controller threads bind only admitted physical cores.
Own guard stops only own job for hostCPU>85/load>85/MemAvailable<20GiB/rootfree
<80GiB, scratch>2GB or compute deadline. 80G is conservatively interpreted80GiB,
matching df-h convention; never clean or stop foreign content/jobs to recover it.
Swap full is preexisting and unchanged. Actual required versions verified on x1:
Python3.12.3,NumPy1.26.4,LightGBM4.6.0,sklearn1.7.2,kernel6.8.0-134-generic.

Forecast reference max(H34729770.57CPU seconds,H350Ractual27619.470383)=29770.57.
H350R actual is recovered from its xbabe1 systemd terminal CPU_USAGE_NSEC journal,
not inferred from wall time or partial samples. Initial seconds=Cref/(n*.9)*1.15
+300; first chronological completed task reforecast remaining35 x max(Cref/36,
measured task processCPU)/(n*.9)*1.15+300 from now,ongoingcredit0. Depth4 cost is
unmeasured; no acceleration is assumed. Full36 must fit both forecasts; miss=>
INCOMPLETE_RUNTIME_FORECAST,never a scientific negative. Admission failure=>
INCOMPLETE_CPU_ADMISSION. No timing fit/calibration/reset/reprobe/cut cells.
Compute stop2026-11-05T10:45:00Z; close2026-11-05T11:00:00Z, immutable.

All S3 GET/whole/member SHA and create-only uploads on xbabe2 only, own prefix
experiments/rd_breakthrough_20261030/H353/solhwv-x1-20261105T083500Z/.
Offline checkpoint/input copies only. xbabe2 courier pulls SHA/size/member-pinned
compressed inner archives, verifies transport and create-only exact-Version whole+
everymember recovery, then returns receipt; x1 deletes raw inner copies only after
that proof. Duplicate compressed caches may also be reclaimed after central proof
match+remote rehash; preserve source/refits/locks/audits/inputs/receipts. Ledger
records every cache deletion. No experiment code/selection/score change. Final
outputs rsync back and every SHA is verified before final S3 custody, then each
host's exact owned file set is rehashed before deletion. No AWS credentials on x1.
xbabe2 non-scientific I/O uses core62/nice19/6GiBswap0 with own-only load backoff,
outside H351/H352/H335 masks; never pause/kill any foreign or scientific process.

## Publication and boundaries

CI_FREEZE+BATCH_WINDOW: private index/objects and fresh-fetch nonforce v1/next only.
No newlane/PR/main/fastlane/hostedCI/FINDINGS. Preserve canonicalforeignHEAD/index/
stageddirtyfiles. Own source directory, oneH353row eachwave/overnight index,
claim/status/comms and CLOSEOUT /home/ubuntu/jope_plan/evidence/H353/solhwv-x1-20261105-t0835/CLOSEOUT.json.
No GPU/qualification.lock/DL-Val/dev91/DL-Test/sealed/holdout/meta_test/fresh-cohort,
forbiddenhosts, livewriter/run-dir/markers/screens/CPUguard or foreign-job actions.
H335/H351/H352 lanes/files/units untouched. No duplicated axes; no serving action.
