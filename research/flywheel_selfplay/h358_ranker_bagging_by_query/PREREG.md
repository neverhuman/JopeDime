# H358 — native ranker bagging_by_query

Owner sol_v1_dsc01. Authority /home/ubuntu/nudges/hw_t1000_dsc01_h358_bagbyquery.md.
Copy committed H352 harness 02dfee2f7e, restoring BASE num_leaves7. Publish
this complete source and prereg non-force to origin/v1/next before treatment fits.

## Fixed hypothesis, sole added ranker keyword

Native LightGBM4.6.0 bagging_by_query=True samples whole contexts instead of
partial query lists and may smooth pairwise lambda gradients. ONE arm BBQ adds
only bagging_by_query=True in both pinned topk.fit_ranker and bagged.fit_ranker_n,
covering inner fits, final refits and all8bags. BASE calls contain no such keyword;
removing it must restore the entire original function ASTs exactly.
Before treatment read archived BASE native configs: bagging_by_query=0,
bagging_fraction=.8,bagging_freq=1,leaves7,depth3. Any discrepancy stops
INCOMPLETE_SPEC_MISMATCH. No calibration fits or BASE refits.
All other args unchanged: lambdarank/sigmoid1/ndcg/trunc8/labels0..7/defaultgain,
leaves7/depth3/minchild20/L2=1/L1=0/subsample.8/freq1/colsample.8; original
seeds/features/groups/18configs/tie-switch1e-4/sentinel-2. Veto unchanged;
all108 final veto SHAs must equal BASE. Every native fit audit validates the
whole BASE native configuration for its original seed/LR/tree-count inputs,
with precisely bagging_by_query0->1; veto delta empty. Six native derived
seeds are checked using the LightGBM Random uint32 LCG, with pinned BASE
reference parity tests. No second scientific lever. Record realized bagged query
counts only if cheaply exposed; native Booster dumps do not expose them, so
null/unavailable is a diagnostic limit, never an invented count or extra fit.

## Frozen BASE and physical TRAIN closure

BASE H303 NESTED/H308 P0/P1/P2 x roots30001,30002,30003 x4 family-grouped outer
folds =36 fit tasks/9 panels,1219 TRAIN contexts/205 families. BASE source
819a45aa5e30e662ab511cd9962b85b2b8c1e44a, H287 export
213b30ad52e1ce48927bba960ed609bd98ab4dfe; all seven vendor hashes pinned.
Fresh exact-version GET read-only archived BASE:
H316 experiments/rd_breakthrough_20261030/H316/solhwv-20261104T105633Z/h316_models_locks_outputs.tar.gz,
VersionId W5jY6iZlwoRszA7P9zfNLLMtRtM9ZZkW,
SHA256 4eae1dec792e1c7b4adff9ff9a5226d0c38cfe1bd9605ba07cc2596f4ad7b0b5;
H315 experiments/rd_breakthrough_20261030/H315/picks.npz,
VersionId jCll6z9FGwmpNg.cocb305LcVzFYuieR,
SHA256 e9e6b16a7381ff14191d2a4d31ce31d4b623c39297e7383ee6aeb871e08ca615.
Verify full object/selected member hashes,9panels/36fold-root pick+lock hashes
and3 medians EXACT: .0643163334285782/.04821480030677507/.04360142642936269.
Mismatch stops REGRESSION_FAIL. No BASE refit or new BASE model predictions.
Match current1219 TRAIN pool hashes,selection/r882 costs and4cache TRAIN prefixes
to archived input manifest9b148003d48ab609b2a796c57fb8b4c0296cc0ee79771f034944f4b8dc064263.

Preparation terminates before cold fit admission. Fit parent loads metadata only;
workers physically open only outer-TRAIN pools and declared TRAIN byte ranges.
No inherited outer/preparation packet cache. Record allowed/forbidden contexts,
packet/range hashes; fit-worker outer features/packets before fsynced fit_lock
and parent fsync must be0. All36 locks/native/constructor/physical guards and
108veto identities must pass before cold once-only outer TRAIN scoring.
STARTED/COMPLETE markers prohibit replay. No partial scoring or reduced design.
No DL-Val/dev91/DL-Test/sealed/holdout/meta_test/fresh-cohort payloads.
DOPE c51a56d0 unchanged/unloaded, CPU selection layer only.

## Metrics and locked decision

Raw utility vs r882slot0 paired with archived BASE. Exact Elo label:
reference = r882 slot0 (DL-Train outer); r3175 not measured on TRAIN.
Report family-equal mean/literal family-median gains and whole-family95%CIs,
drop-top1/5/10,LOFO min,W/T/L,harm(gain<-.01),invalids,switches,per-root medians
and paired CIs,Elo/EloCI,config agreement,seed Jaccard. Ordinary bootstrap retains
10000draws/RNG302150. Paired bootstrap10000/RNG358191 uses shared family samples
both arms/all9cells, each arm redrops own top10; median9cells is statistic,
CI95 quantiles.025/.975, one-sidedLCB95 quantile.05. Per-root medians and CIs use
these same draws and median across the3 partitions for each root.
ADVANCE iff paired drop10median>=+.002 andLCB95>0 AND BBQmedian fullgain>=.9BASE
AND median harmed<=BASE AND paired median strictly positive on>=2/3roots.
Ties/subthreshold never advance. Else BBQ_NOT_ADVANCED only after full scores.
Incomplete results unknown/null, never zero or a scientific negative. Report Elo;
no speed verdict. A future one-shot dev91 vs r3175 is outside H358 authorization.

## One capacity gate and immutable window

xbabe2 CPU only,nice19,numeric1,systemd MemoryMax6G/MemorySwapMax0. Only
physical27..37 and SMT siblings91..101. ONE fresh5second mpstat interval;
each pair requires both siblings>=85%idle, take first<=8eligible physicals,
require>=4 and same-interval hostCPU<=70%. Probe-start token is create-once
before probing; no replay/reprobe/reset. Bind controller/monitor/fit/score to
admitted physicals; siblings are admission guards, not extra workers.
Exclude H35515..26/79..90,H3560..14/64..78,H35738..53/102..117.

Admission forecast29770.57CPU/(workers*.9)*1.15+300s. After first completed
task forecast remaining35*max(29770.57/36,actualfirstCPU)/(workers*.9)*1.15+300s
from currenttime, no ongoingcredit. Both must end strictly before fixed
2026-11-05T12:45:00Z. Reject runtime INCOMPLETE_RUNTIME_FORECAST; host/core
admission INCOMPLETE_CPU_ADMISSION. No cuts/partialscoring/newcapacitymeasure.
Stop only own cgroup on MemAvailable<20GiB,hostCPU>85%,load>85,ownscratch>2GB
or compute deadline. Scratch /home/ubuntu/jope-data/h358-dsc01, max2GBdecimal.
Close2026-11-05T13:00:00Z fixed. Late means INCOMPLETE, scientificUNKNOWN/null.
Create-only S3 experiments/rd_breakthrough_20261030/H358/dsc01-20261105T100000Z/;
exact-VersionIdGET/fullSHA/EVERYmember verification before deleting only own
scratch. Preserve all interrupted costs and evidence, never replay. No modelsinGit.

## Publication and review

CI_FREEZE+BATCH_WINDOW: private index only, canonical HEAD/index preserved.
Nonforce fresh-parent origin/v1/next only; noPR/main/fastlane/hostedCI. One row
in each requiredindex. Same-agent exact-head review before launch. Beforeclose:
prereg precedes firstfit; all36locks before scoring;0outerreadsbeforelocks;
0DL-Val/dev91/sealed/holdout;native sole delta;108vetoSHAs;exactBASEregression;
rootmedians/CIs/tiebar retained. DOPEc51a56d0unloaded/unchanged,CPU selectiononly.
One of a sequential single-knob sweep H348–H358 on the same folds/roots; any
ADVANCE is TRAIN-only, requires separately authorized one-shotdev91vsr3175
confirmation, outside H358. No speedverdict, GPU, qualification.lock, otherhosts,
otherwaves/jobs, livecheckpoints, or flywheelwriter changes.
