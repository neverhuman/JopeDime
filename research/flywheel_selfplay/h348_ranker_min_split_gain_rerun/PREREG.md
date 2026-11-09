# H348: ranker min_split_gain rerun

Owner sol_v1_gov03; authority hw_t0400_gov03_h348_minsplitgain.md, November 5.
H347 is CLOSED and RELEASED. This is the same min_split_gain=0.01 science as
H346 (zero fits, INCOMPLETE_CPU_ADMISSION), using the committed H347 harness at
6f87ae99ce2752d595663c181aeb2457bd231939 from origin/v1/next. Full committed file
SHA inventory is retained. Publish this preregistration nonforce to v1/next
BEFORE any H348 treatment fit. No tuning, alternate arms, or diagnostic fits.

## Hypothesis and fixed ranker delta

Refusing tiny-gain LambdaRank splits may damp noisy small corrections and cycling.
This is a hypothesis, not an observed robustness or speed result. ONE arm MINGAIN:
add min_split_gain=0.01 only in the pinned H287 topk.fit_ranker and
bagged.fit_ranker_n (213b30ad52e1ce48927bba960ed609bd98ab4dfe), covering every
inner, final refit and eight-bag ranker. BASE's omitted keyword has effective
native min_gain_to_split=0. If archived BASE differs, stop
INCOMPLETE_SPEC_MISMATCH before treatment. All 972 archived BASE ranker native
records are checked read-only. Restoring the omitted default zero removes the
single override and reproduces both complete BASE ASTs exactly; explicit zero
also gives the identical constructor configuration. Every native fit is audited
against every constructor argument, including exact integer settings and seeds.

Ranker reg_alpha stays BASE zero (native lambda_l1=0); H347's 0.5 is removed.
Preserve LambdaRank/ndcg, sigmoid1, labels0..7, implicit label_gain,
min_child_samples20, min_child_weight.001, num_leaves7, max_depth3,
subsample.8/freq1, colsample.8, reg_lambda1, LR/tree options, seeds, features,
original18-config grid, tie/switch1e-4 and sentinel-2.0. No H342 Hessian floor,
extra shrinkage, objective, label, callback, early stopping or round changes.
Original topk200 rounds/LR(.03,.05,.10), prediction counts50/100/200 and original
selected bag fit counts remain. Veto binary GBDT is unchanged, including native
lambda_l1=1 and lambda_l2=2. Every final veto file must SHA-match archived BASE:
108/108 across36tasks, enforced before outer scoring. Expected per task:
819ranker+195veto fits,351ranker+39veto saved models; these are expectations until
actual execution. No model payloads in Git.

## BASE first and physical input closure

Pinned H303 NESTED/H308 code819a45aa5e30e662ab511cd9962b85b2b8c1e44a.
P0/P1/P2 x roots30001-30003 x4 family-grouped outer folds =36fit tasks/9cells,
1219TRAIN contexts/205families. First science gate: exact-version frozen BASE
GET, whole-object and selected-member SHA checks,9/9panels,36fold-root pick+lock
hashes and exact medians .0643163334285782/.04821480030677507/.04360142642936269.
No BASE refits, calibration fits or new BASE model predictions. A mismatch stops
REGRESSION_FAIL. Current1219TRAIN pool hashes/selected/r882costs and four consumed
mixed-cache TRAIN feature prefixes must match the archived input identities.

Same archive identities as H347: H316 key
experiments/rd_breakthrough_20261030/H316/solhwv-20261104T105633Z/h316_models_locks_outputs.tar.gz,
VersionId W5jY6iZlwoRszA7P9zfNLLMtRtM9ZZkW,
SHA256 4eae1dec792e1c7b4adff9ff9a5226d0c38cfe1bd9605ba07cc2596f4ad7b0b5.
H315 picks key experiments/rd_breakthrough_20261030/H315/picks.npz,
VersionId jCll6z9FGwmpNg.cocb305LcVzFYuieR,
SHA256 e9e6b16a7381ff14191d2a4d31ce31d4b623c39297e7383ee6aeb871e08ca615.

Preparation ends in a separate process before admission. Fit parent has metadata
only; each worker opens only outer-TRAIN pools and exact allowed cache ranges.
Audit rejects physical outer features and BASE outcome/native copies during fit.
Require physical outer rows and packets before fsynced fit_lock plus parent fsync0.
All36locks/guards/native audits/108BASE veto SHA matches must pass before a fresh
once-only outer score process. STARTED/COMPLETE markers prevent repeats. No partial
scoring, fitting restart, protected rows or completed-score replay.

## Metrics and locked decision

Raw utility against r882slot0, paired with frozen BASE. Exact Elo label:
`reference = r882 slot0 (DL-Train outer); r3175 not measured on TRAIN`.
Report family-equal mean gain and literal family-median gain with whole-family
bootstrapCI, drop-top1/5/10, LOFO min, W/T/L, harmed gain<-.01, invalid, switches,
per-root medians AND per-root paired CIs, Elo/EloCI, seed Jaccard and configuration
agreement. Ordinary intervals use10000draws/RNG302150 unchanged. No AUC or speed
verdict. Per-root paired CIs use each root's median across its three partitions,
with the same shared-family bootstrap draws as the primary paired statistic.

ADVANCE iff paired(MINGAIN-BASE) drop-top10 median over9cells>=+.002 AND one-sided
LCB95>0; MINGAIN median full gain>=90%BASE; median harmed<=BASE; and positive
per-root paired drop10 medians on>=2/3roots. Else MINGAIN_NOT_ADVANCED, only after
complete scores. Ties/subthreshold never advance. Paired bootstrap:10000whole-
family draws/RNG348191, same family sample both arms/all9cells; each arm redrops
its own top10, statistic median9cells, LCBquantile.05. Root intervals are reporting,
not additional decision gates. Any one-shot dev91/r3175 comparison requires a
separate assignment, outside H348. No serving promotion in this experiment.

## One admission, two forecasts, immutable limits

xbabe2 CPU only,nice19,numeric1,MemoryMax6G/MemorySwapMax0. Candidate physical
cores0-14 and27-33 plus SMT64-78 and91-97. ONE fresh5-second/core+SMT>=85%-idle
sample; both siblings qualify. Take first<=6passing pairs, minimum4, AND require
SAME-interval aggregate hostCPU<=70%. Aggregate/proc/stat deltas bracket that same
single mpstat interval. Bind controller threads and every worker to admitted
cores before fitting. No second probe, worker expansion or borrowed cores.
Exclude H34915-26+sibs79-90 and H33538-53+sibs102-117; leave their jobs alone.

Full36 forecast after admission: H347 actual29770.567337 CPU seconds /
(workers*.9)*1.15 +300seconds scoring reserve. Cost multiplier1.0 is an unmeasured
planning assumption. Launch only if end is strictly before06:15Z November5.
Reforecast exactly after first completed fold/root task: remaining35 x
max(H347CPU/36,first task actual processCPU)/(workers*.9)*1.15+300 added to current
time, ongoing work credit0. Miss=>INCOMPLETE_RUNTIME_FORECAST; stop own service
only. No cut cells, partial scoring, reprobe, restart or deadline reset.

Fixed compute stop2026-11-05T06:15:00Z; fixed close2026-11-05T06:30:00Z. Guard own
job if hostCPU>85%, load>85, MemAvailable<20GiB or scratch>2GBdecimal. Scratch only
/home/ubuntu/jope-data/h348-gov03. Create-only S3 experiments/rd_breakthrough_20261030/H348/
inner/final custody, exact-version GET +whole SHA+every-member verification before
raw inner/own scratch deletion. Preserve actual exits and immutable source/evidence.

## Publication and boundaries

Private index/objects; CI_FREEZE+BATCH_WINDOW, nonforce v1/next ONLY. No
PR/main/fastlane/hostedCI. README/metrics<50KB/code/tests, exactlyone H348 row in
each wave/overnight index, claim/status and one claim+one close comms/CLOSEOUT.
No GPU/qualification.lock,DL-Val/dev91/DL-Test/protected/sealed/holdout/meta_test/
fresh-cohort rows, forbidden hosts, live writer or foreign-job operations.
DOPEc51a56d0 unchanged/notloaded; scope CPU selection layer only.
H347/H342/H340/H334/H333/H329/H326 stay CLOSED.
