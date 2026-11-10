# H387 — NL15 ranker learning-rate half / twice the trees

Owner sol_v1_dsc01. Authority: /home/ubuntu/nudges/hw_t1805_dsc01_h387_nl15_lrhalf_2xtrees.md.
One fixed LRHALF arm on committed H352 NL15, using successful H377 transport with
MSG01 fully reverted to0. H384 remains CLOSED INCOMPLETE_SPEC_MISMATCH; no relabel.
No paid/GPU compute. Publish this prereg and complete source non-force to origin/v1/next
before treatment fits. No calibration fits, alternate arm, new seed keyword or tuning.

## Hypothesis and sole coupled intervention

Half learning_rate and twice n_estimators may yield finer boosting steps and improve
paired drop10 gain over NL15. This changes the optimization path; it does not claim
literal Polyak averaging. Map ranker constructor learning_rate=float(lr)*.5 and
n_estimators=original_n*2 in both pinned topk.fit_ranker and bagged.fit_ranker_n,
covering inner selection, final refit and all8bag rankers. Exact LR*rounds product is
preserved. Removing only those two expression multipliers restores both entire
committed H352 effective ASTs exactly, without keyword additions. Min-split-gain0,
tree featurefraction.8, bagfraction.8/freq1/querybagging0, leaves15/depth3,
L1zero/L2one/minchild20, labels/groups/defaultgain/features/seeds/18configs,
tie-switch1e-4/sentinel-2 and all other native fields are unchanged.

Original LR/tree grid labels remain coordinates for seed/config identity. To use the
same longer trajectory at each grid coordinate, map every explicitly positive ranker
Booster prediction tree limit to2*original_n during fitting and cold scoring. None/0/-1
already request all trees and remain unchanged. Binary/veto predictions are untouched.
Both trained and reloaded rankers are covered; reject nonzero start with a positive
ranker limit. This is the tree-count intervention, with no changed grid choices.
Final archived lock selected_lr/selected_n_estimators keep baseline coordinates;
constructor/native audit records carry actual half-LR/doubled-round settings.

Fresh archived H352 native LR/roundpairs must agree with the original committed
constructor/grid; record baseline/treatment pairs before fitting. Already-treated
or mismatching archive closes INCOMPLETE_SPEC_MISMATCH, no fit/probe. Every treatment
native audit reconstructs its original LR*2/rounds/2 configuration, verifies exact
product and requires exactly the two native deltas; all108finalvetoSHAs match.

## Frozen comparator regressions and TRAIN closure

TRAIN1219contexts/205families; H303 NESTED/H308 P0/P1/P2 x roots30001–30003
x4 family-grouped outer folds, all36tasks/9panels mandatory. Fresh exact-version
GETs, read-only archived copies, no BASE or NL15 model predictions:

- BASE H316 key experiments/rd_breakthrough_20261030/H316/solhwv-20261104T105633Z/h316_models_locks_outputs.tar.gz,
  VersionId W5jY6iZlwoRszA7P9zfNLLMtRtM9ZZkW,
  SHA256 4eae1dec792e1c7b4adff9ff9a5226d0c38cfe1bd9605ba07cc2596f4ad7b0b5.
- BASE H315 key experiments/rd_breakthrough_20261030/H315/picks.npz,
  VersionId jCll6z9FGwmpNg.cocb305LcVzFYuieR,
  SHA256 e9e6b16a7381ff14191d2a4d31ce31d4b623c39297e7383ee6aeb871e08ca615.
- H352 incumbent manifest key experiments/rd_breakthrough_20261030/H352/jrb01-20261105T075500Z/closeout/closeout-manifest.json,
  VersionId RtNLAfmRJbXqdXQx7nxZLZ8dWHPcVYQC,
  SHA256 d3b47c5fbc1ecc3c6e6d4f89dfef7d6a3f5216f45c6dd1589ce94fb413350f60.
  The manifest binds27 version-pinned parts and2691members. Fresh GET each
  exact VersionId; verify whole-object SHA and EVERY member. Keep only outcome/
  lock/proof copies; audit archived native configs without storing model payloads.

BASE requires9/9panels,36fold-root pick+lock hashes and exact3medians
.0643163334285782/.04821480030677507/.04360142642936269. H352 NL15 requires
9panels,36lock/once-onlyscore markers/pick hashes, all saved panels and raw
family-gain arrays exact against archived metrics. Any discrepancy stops
REGRESSION_FAIL; native-spec discrepancy stops INCOMPLETE_SPEC_MISMATCH.
BASE source819a45aa5e30e662ab511cd9962b85b2b8c1e44a; H287 source
213b30ad52e1ce48927bba960ed609bd98ab4dfe; seven vendor hashes retained.
Current TRAIN pool/selection/r882costs/fourcacheprefixes must match archived
input-manifest SHA9b148003d48ab609b2a796c57fb8b4c0296cc0ee79771f034944f4b8dc064263.

Preparation processes terminate before cold fit admission. Fit parent holds
metadata only, no inherited prepared packets. Workers physically open only
outer-TRAIN contexts and declared TRAIN byte ranges; archived BASE/NL15 outcome
payloads are denied to fitting workers. All36fsynced fit_locks and constructor/
native/physical-scope/veto proofs precede once-only outer scoring. STARTED/
COMPLETE latches prohibit replay. No partial scoring, reduced design or retries.
Zero DL-Val/dev91/DL-Test/sealed/holdout/meta_test/fresh-cohort payloads.
DOPEc51a56d0 unloaded/unchanged, CPU selection layer only.

## Metrics and locked primary decision

PRIMARY paired vs archived H352 NL15; SECONDARY paired vs archived BASE.
Raw utility relative to r882slot0. Exact Elo label:
reference = r882 slot0 (DL-Train outer); r3175 not measured on TRAIN.
Report family-equal mean and literal family-median gain with whole-family95%CIs,
drop-top1/5/10,LOFOmin,W/T/L,harmed(gain<-.01),invalids,switches,per-root
medians+pairedCIs,Elo+EloCI,configagreement. All whole-family bootstraps10000draws,
RNG387191. Paired comparisons share family draws across all9cells and arms;
each arm drops its own top10, then median across9cells. CI95 .025/.975;
one-sidedLCB95 .05. Rootmedians/CIs use the same draws over3partitions each.
ADVANCE iff primary paired drop10median>=+.002 AND one-sidedLCB95>0 AND
median fullgain>=.9*NL15 median AND median harmed<=NL15 median AND paired
rootmedian>0 on>=2/3roots. Otherwise LRHALF_NOT_ADVANCED only after full
qualified scoring; ties/subthreshold never advance. Incomplete science UNKNOWN/
null, never zero or a negative verdict. Secondary BASE comparison never decides.
No AUC or speed verdict. Sequential intervention sweep H348–H387 on the same
folds/roots; ADVANCE requires separately authorized one-shotdev91 confirmation,
NOT in H387. This campaign reads no dev91 rows or H356 model payloads.

## Admission, corrected guard and fixed window

xbabe2 CPU only/nice19/numeric1; systemd MemoryMax6G/SwapMax0/CPUQuota<=800%.
Authorized UNION: primary physical54..63(+SMT118..127), then fallback50/51(+SMT114/115). ONE simultaneous fresh5second per-core+
SMT interval over the entire union. Require bothsiblings>=85%idle; take the
first<=8eligible in that primary-then-fallback order. Mixing is authorized;
never sample again or choose another pool. Require>=4 and same-interval
hostCPU<=70%. Create-once probe token. Controller/monitor/fit/score bind the
admitted physical set; affinity verification compares sets so mixed ordering
cannot change admission. No other cores or sibling workers, and no foreign
job signals. Other owners' work remains untouched.

Forecast reference=H352 actual full36 CPU26189.516914s, bound by
actual_resources_reference.json and its exact original UNIT-FINAL receipt SHA.
Admission forecast
reference/(workers*.9)*1.15+300s. After first completed task, remaining35*
max(reference/36,actualfirsttaskCPU)/(workers*.9)*1.15+300s from currenttime;
ongoingcredit0. Both must fit strictly before fixed2026-11-05T22:15:00Z,
else INCOMPLETE_RUNTIME_FORECAST. No reduced cells/reprobe/reset/restart.

OWNER-CORRECTED GUARD: stop only own study cgroup for MemAvailable<20GiB OR
hostCPU>85% averaged over60seconds OR load1>108 on THREE consecutive60second
samples OR scratch>2GBdecimal, plus the fixed compute deadline. Load<=108
resets the consecutive counter; CPU=85% is allowed. Five-second CPU telemetry
is diagnostic only and never triggers the60second CPU guard. Memory/scratch/
deadline remain checked every5seconds. The old absolute load>85 rule is removed.
No swap use. Scratch /home/ubuntu/jope-data/h387-dsc01 <=2GBdecimal. Fixed
compute stop22:15Z/close22:30Z Nov5; late means INCOMPLETE. Own process only.
Create-only S3 experiments/rd_breakthrough_20261030/H387/dsc01-20261105T180500Z/;
exact-VersionId GET/fullSHA/EVERYmember verification before only ownscratch
cleanup. Preserve interrupted counts/unknowns; no replay/relabel/modelsinGit.

## Publication and review

CI_FREEZE+BATCH_WINDOW: fresh-parent non-force origin/v1/next only; noPR/main/
fastlane/hostedCI. Private checkout/index only, canonical HEAD9483d585/indexa78b7cab
and all foreign dirty files preserved. Publish README/metrics<50KB/code/tests
and one row each requiredindex. Same-agent exact-head review: prereg precedes
firstfit; BASE+NL15 regressions exact; sole native delta vsNL15; all36locks
before outer reads; protectedrows0; all108vetoSHAs; rootCIs/tiebar retained.
NoGPU/qualification.lock/otherhosts/foreignjobs/livewriter/AUTORESTARTchanges.
