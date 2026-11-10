# H384 — prospective NL15 feature_fraction_bynode gate

Owner sol_v1_dsc01; authority /home/ubuntu/nudges/hw_t1730_dsc01_h384_nl15_featfrac_bynode.md.
Committed proven H377513858a799 transport with MSG01 reverted to H352
min_split_gain0; retain H352 leaves15/depth3/feature_fraction.8/rowfraction.8/
freq1/LR/trees/seeds/veto. H377 CLOSED MSG01_NOT_ADVANCED stays unchanged.

## Requested hypothesis and mandatory native-spec precondition

Requested arm FFBN is feature_fraction_bynode false0->true1 in both pinned
ranker functions, inner/final/all8bags, to increase feature resampling across
nodes and reduce single-context -2.0 chase. The owner explicitly requires:
if archived H352 native feature_fraction_bynode is already true/nonzero, close
INCOMPLETE_SPEC_MISMATCH BEFORE fits. We apply that instruction literally.
The pinned native reference already records1. Fresh exact-version archive
reads will confirm all972 rankers after the prescribed BASE/NL15 regressions.
No CPU probe, fitting, calibration or model scoring is authorized unless the
zero-native precondition passes. Controller, fit entry and patch installation
all require a durable PASS; no existing baseline value is rewritten to zero.

LightGBM4.6.0 documents feature_fraction_bynode as a numeric fraction, default1,
range(0,1], subsampling nodes only below1. The requested numeric1 is already
the pinned default, so adding a keyword alone is never proof of a native
change. Literal Boolean handling is not qualified by any fit. Reference:
https://lightgbm.readthedocs.io/en/v4.6.0/Parameters.html#feature_fraction_bynode
We do not select a different numeric fraction or reinterpret the hypothesis.

Prospective source adds only feature_fraction_bynode=True after the unchanged
H352 leaves transform; removing it restores both entire H352 ASTs exactly.
No new seed keyword. Native audits would require exactly0->1, and reject the
actual1->1/no-op. Veto unchanged; all108 final SHAs would have to match BASE.
Native-spec failure produces UNKNOWN/null science, never FFBN_NOT_ADVANCED.
Prereg/source will be nonforce-published to v1/next before any possible fit.
No training is performed to explore the parameter type or alternative values.

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
RNG384191. Paired comparisons share family draws across all9cells and arms;
each arm drops its own top10, then median across9cells. CI95 .025/.975;
one-sidedLCB95 .05. Rootmedians/CIs use the same draws over3partitions each.
ADVANCE iff primary paired drop10median>=+.002 AND one-sidedLCB95>0 AND
median fullgain>=.9*NL15 median AND median harmed<=NL15 median AND paired
rootmedian>0 on>=2/3roots. Otherwise FFBN_NOT_ADVANCED only after full
qualified scoring; ties/subthreshold never advance. Incomplete science UNKNOWN/
null, never zero or a negative verdict. Secondary BASE comparison never decides.
No AUC or speed verdict. Sequential single-knob sweep H348–H384 on the same
folds/roots; ADVANCE requires separately authorized one-shotdev91 confirmation,
NOT in H384. This campaign reads no dev91 rows or H356 model payloads.

## Admission, corrected guard and fixed window

xbabe2 CPU only/nice19/numeric1; systemd MemoryMax6G/SwapMax0/CPUQuota<=800%.
Authorized UNION: primary physical54..63(+SMT118..127), then fallback50/51
(+SMT114/115). H383cores27..37 are forbidden, including preparation jobs.
ONE fresh5second both-SMT>=85%-idle probe, first<=8overthisunion/min4/
same-intervalhostCPU<=70%, only AFTER native-spec PASS. No reprobe/restart.
When native-spec fails there is NO CPU gate, no fitting or scoring unit.
Preparation/custody uses CPU54/nice19/math1/6Gswap0, no H383/foreign jobs.

Forecast reference=H352 actual full36 CPU26189.516914s, bound by
actual_resources_reference.json and its exact original UNIT-FINAL receipt SHA.
Admission forecast
reference/(workers*.9)*1.15+300s. After first completed task, remaining35*
max(reference/36,actualfirsttaskCPU)/(workers*.9)*1.15+300s from currenttime;
ongoingcredit0. Both must fit strictly before fixed2026-11-05T21:15:00Z,
else INCOMPLETE_RUNTIME_FORECAST. No reduced cells/reprobe/reset/restart.

OWNER-CORRECTED GUARD: stop only own study cgroup for MemAvailable<20GiB OR
hostCPU>85% averaged over60seconds OR load1>108 on THREE consecutive60second
samples OR scratch>2GBdecimal, plus the fixed compute deadline. Load<=108
resets the consecutive counter; CPU=85% is allowed. Five-second CPU telemetry
is diagnostic only and never triggers the60second CPU guard. Memory/scratch/
deadline remain checked every5seconds. The old absolute load>85 rule is removed.
No swap use. Scratch /home/ubuntu/jope-data/h384-dsc01 <=2GBdecimal. Fixed
compute stop21:15Z/close21:30Z Nov5; late means INCOMPLETE. Own process only.
Create-only S3 experiments/rd_breakthrough_20261030/H384/dsc01-20261105T173000Z/;
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
