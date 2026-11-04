# H340: rank_xendcg ranking objective

Owner sol_v1_gov03; authority hw_t2300_gov03_h340_xendcg.md, 2026-11-04.
H334 is CLOSED; reuse its harness and measured full36 CPU reference, not its
path-smoothing intervention. This preregistration must be pushed nonforce to
origin/v1/next before ANY H340 optimizer fit. No H334/H333/H329/H326 reopening.

## Hypothesis and single fixed intervention

A different listwise ranking surrogate may reduce sensitivity to noisy failure
contexts. This is a hypothesis, not a measured noise-robustness or speed claim.
ONE arm XENDCG: change only the existing objective keyword value from
"lambdarank" to "rank_xendcg" in topk.fit_ranker and bagged.fit_ranker_n,
pinned H287 export 213b30ad52e1ce48927bba960ed609bd98ab4dfe. Restore that value
to reproduce both complete BASE ASTs exactly. No added keyword. label_gain is
ignored by rank_xendcg; no label transformation or gain override is permitted.
If the native library rejects objective or label format, stop INCOMPLETE_SPEC;
no workaround, changed labels or substitute objective.

Keep labels, group/query construction, features, all seeds, GBDT, num_leaves7,
max_depth3,min_child_samples20,subsample/colsample .8,subsample_freq1,reg_lambda1,
n_jobs1,deterministic/force_row_wise and every other constructor field.
The veto classifier is unchanged binary GBDT. Native path_smooth remains0 for
all models, extra_trees0 and max_delta_step0. All inner/final/eight-bag rankers
use rank_xendcg; native fitted model records validate every constructor field.
Original topk fits200rounds with LRgrid(.03,.05,.10), prediction counts50/100/200;
bagged final uses unchanged selected round count. Preserve original18-config
TAU(.05,.10,.15) x k(5,6,7) x agreement(2,3), raw inner objective, tie1e-4,
switch/margin rules and failure sentinel -2.0. No tuning or extra arm.

## BASE regression and physical data closure

Unchanged H303 NESTED/H308 inputs/code819a45aa5e30e662ab511cd9962b85b2b8c1e44a.
P0/P1/P2 x roots30001-30003 x4 family-grouped outer folds =36 fit tasks/9 cells,
1219 TRAIN contexts/205 families. First science gate: frozen archived BASE
exact-version GET read-only with whole-object and selected-member SHA checks.
Require9/9 panels and all36 fold/root pick+lock hashes exact and three medians
.0643163334285782, .04821480030677507, .04360142642936269. Mismatch stops
REGRESSION_FAIL. No BASE refits or fresh BASE model predictions. Validate current
TRAIN input hashes/cost/selection identities and mixed-cache TRAIN byte prefixes.

Fit parent loads metadata only; worker opens only outer-TRAIN pools and allowed
cache byte ranges. Audit denies physical outer feature packets and BASE outcomes
during fit. Record allowed/forbidden context and byte identities. Outer feature
rows/packets before fsynced lock and parent-directory fsync must equal0. All36
locks, guards, models and native audits must pass before cold once-only outer
scoring. STARTED/COMPLETE markers prevent repeats. No partial scoring/restarts.
Every successful fit emits constructor/native params/model SHA and member seed;
all saved models bind to their member-native fit record. Expected per task:
819 ranker+195 veto fits,351 ranker+39 veto saved models. These are expectations,
not coverage claims until actual execution/verification.

## Metrics and locked decision

Raw utility paired with archived BASE vs r882slot0; exact label:
`reference = r882 slot0 (DL-Train outer); r3175 not measured on TRAIN`.
Report family-equal mean and literal median/whole-family bootstrap CI,
drop-top1/5/10, LOFO minimum, W/T/L, harmed gain<-0.01, invalid, switches,
per-root medians, Elo/CI, seed Jaccard and config agreement. Ordinary intervals
use unchanged10000 draws/RNG302150. No AUC or speed verdict.

ADVANCE iff paired(XENDCG-BASE) drop-top10 median over9cells >=+0.002 AND
one-sided LCB95>0; XENDCG median full gain>=90%BASE; median harmed<=BASE; and
per-root paired drop10 median across3partitions>0 for>=2/3roots. Otherwise
XENDCG_NOT_ADVANCED, only after complete scores. Ties/subthreshold never advance.
Paired whole-family bootstrap10000 draws/RNG340191: shared family resamples
both arms/all9cells, each arm redrops its own top10, statistic median9cells,
LCB quantile.05. No H340 dev91; if ADVANCE the next separate assignment may be
one-shot dev91 vs r3175.

## One capacity gate and immutable window

xbabe2 CPU only, <=8workers on1,2,3,7,9,10,12,14 and SMT65,66,67,71,73,74,76,78.
Disjoint H3414,5,6,8,27,33,57,59/H33813,15,60,62/H33538-53 and siblings.
After BASE/input setup ONE5-second per-core+SMT>=85%-idle gate, both siblings
must qualify. No repeated probes/borrowed cores. Full36 launch requires>=4cores
AND actual launch-time forecast ending strictly before2026-11-05T02:15:00Z.
Reference H334 actual33343.799676 CPU seconds; forecast = CPU/(workers*.9)*1.15
+300 seconds scoring reserve. Objective multiplier1.0 is a planning assumption,
not a timing result. Otherwise INCOMPLETE_RUNTIME_FORECAST, zero fits, unknown
numerical outcome. No reduced design or deadline reset. Compute stop Nov5 02:15Z;
close Nov5 02:30Z. No fresh timing fits.

Nice19, numeric threads1, systemd MemoryMax6G/MemorySwapMax0; stop if host CPU>85%,
load>85,MemAvailable<20GiB or scratch>2GB decimal. Own scratch only
/home/ubuntu/jope-data/h340-gov03. Create-only S3 experiments/rd_breakthrough_20261030/H340/
inner/final custody: exact-version GET+whole SHA+every-member proof before raw
inner/own scratch deletion. Retain immutable source/evidence and actual exits.

## Publication and boundaries

CI_FREEZE + MASTER_MERGE_HOLD: private index/Git objects, nonforce v1/next only;
no PR/main/fastlane/hosted CI. Final README/metrics<50KB/code/tests plus one row
each in wave/overnight indices; claim/status/claim+close comms/CLOSEOUT receipts.
No models in Git. No GPU/qualification.lock, protected/sealed/holdout/meta_test/
fresh cohort, DL-Val/dev91/DL-Test rows, foreign jobs/hosts or live writer changes.
DOPE c51a56d0 not loaded, unchanged, CPU selection layer only.
