# H389 — NL15 bag ensemble doubled (BAG16)

Owner sol_v1_dsc01. Authority: /home/ubuntu/nudges/hw_t1905_dsc01_h389_ranker_nl15_bag16.md.
One fixed BAG16 arm on committed H352 NL15. Reuse H377 successful transport with
MSG01 reverted0; LRHALF is absent. H387 remains CLOSED INCOMPLETE_CPU_ADMISSION;
H384 and all prior closes retain their original dispositions. H361 already tested
LRHALF; this wave does not rerun it. No paid/GPU compute, calibration fits, alternate
arm, seed keyword or tuning. Publish prereg/source non-force origin/v1/next BEFORE
any treatment fit. This is an exploratory ensemble-size hypothesis, not literal
Polyak parameter averaging or evidence that variance dominates bias.

## Sole intervention, seeds and unchanged aggregation

Read the fresh archived H352 fit locks and native bag configs; baseline B must be
positive and identical across108members. Treatment B=2*archived B (expected8->16).
Record observed/treatment values. Change only bagged.N_BAGS, consumed by the exact
original r302.fit_and_freeze_member loop in all inner fits and final refits.
The first8 bags keep their exact original index0..7 and both seed streams; extend
indices8..15 under the SAME core.stream_seed(member_seed,0,"BAG_SWITCH",
"outer_boot" or "outer_bag",index) rule. Verify archived bag seeds against the
independent byte rule (colon-joined UTF8, SHA256 first8 little-endian &0x7fffffff).
No new seed keyword or changed row/family bootstrap rule. All3original members
per root are retained; a task still fits3members x(3taus*4innerfolds+1final)=39members.

Per-model constructors are exactly H352 NL15 ASTs. Veto and proposer counts,
LR/tree grids, labels/groups/defaultgain/features, sentinel-2, minsplit0,
leaves15/depth3/minchild20/L2one/L1zero/featurefraction.8/bagfraction.8/freq1/
querybagging0/path0 and EVERY native field are unchanged. No LRHALF prediction
mapping is installed. Count change alone reverts to the original H352 program;
whole effective constructor AST equality against committed H352 is tested.

Aggregation code and its numeric grids stay byte-identical: original vote_counts
reads every bag score array, switch_by_votes compares the resulting absolute count
to k in5/6/7, and outer member consensus stays2/3. No vote-threshold or margin
rescaling;16bags therefore changes the vote distribution under those same
absolute thresholds. This is part of the declared bag-count intervention and a
limit on interpreting any effect as variance reduction alone. All18config choices,
conservative1e-4 tie rule and once-only scoring protocol remain unchanged.

Expected per task:13nonbag rankers + Bbags per member ->39*(13+16)=1131rankers,
195veto fits; over36tasks40,716ranker+7,020veto=47,736native fits. Final refits
contain1,836rankers and108veto models;108/108finalvetoSHAs must equal BASE/H352.
No baseline refits or predictions. Report configured leaves15 with depth3 bound8;
no claim of15realized leaves.

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
RNG389191. Paired comparisons share family draws across all9cells and arms;
each arm drops its own top10, then median across9cells. CI95 .025/.975;
one-sidedLCB95 .05. Rootmedians/CIs use the same draws over3partitions each.
ADVANCE iff primary paired drop10median>=+.002 AND one-sidedLCB95>0 AND
median fullgain>=.9*NL15 median AND median harmed<=NL15 median AND paired
rootmedian>0 on>=2/3roots. Otherwise BAG16_NOT_ADVANCED only after full
qualified scoring; ties/subthreshold never advance. Incomplete science UNKNOWN/
null, never zero or a negative verdict. Secondary BASE comparison never decides.
No AUC or speed verdict. Sequential intervention sweep H348–H389 on the same
folds/roots; ADVANCE requires separately authorized one-shotdev91 confirmation,
NOT in H389. This campaign reads no dev91 rows or H356 model payloads.

## Admission, corrected guard and fixed window

xbabe2 CPU only/nice19/numeric1; systemd MemoryMax6G/SwapMax0/CPUQuota<=800%.
Authorized UNION: primary physical50..63(+SMT114..127), no fallback. ONE simultaneous fresh5second per-core+
SMT interval over the entire union. Require bothsiblings>=85%idle; take the
first<=8eligible in ascending primary order. There is no fallback;
never sample again or choose another pool. Require>=4 and same-interval
hostCPU<=70%. Create-once probe token. Controller/monitor/fit/score bind the
admitted physical set; affinity verification compares sets so mixed ordering
cannot change admission. No other cores or sibling workers, and no foreign
job signals. Other owners' work remains untouched.

Forecast reference=H352 actual full36 CPU26189.516914s, bound by
actual_resources_reference.json and its exact original UNIT-FINAL receipt SHA.
Conservative admission CPU reference =26189.516914 * (2*819+195)/(819+195),
assuming native-fit-count819/195 as a CPU-share proxy and doubling ALL ranker
cost while holding veto cost fixed. This decomposition is an assumption, not
measured per-type CPU. It intentionally exceeds the actual count multiplier:
only bags double, ranker fits1131/819, veto195unchanged. Admission forecast
scaled_reference/(workers*.9)*1.15+300s. First completed task forecast uses
remaining35*max(scaled_reference/36,actualfirsttaskCPU)/(workers*.9)*1.15+300s
from current time; ongoingcredit0. Both strictly before fixed23:15Z Nov5,
otherwise INCOMPLETE_RUNTIME_FORECAST. No cut cells/reprobe/reset/restart.

OWNER-CORRECTED GUARD: stop only own study cgroup for MemAvailable<20GiB OR
hostCPU>85% averaged over60seconds OR load1>108 on THREE consecutive60second
samples OR scratch>2GBdecimal, plus the fixed compute deadline. Load<=108
resets the consecutive counter; CPU=85% is allowed. Five-second CPU telemetry
is diagnostic only and never triggers the60second CPU guard. Memory/scratch/
deadline remain checked every5seconds. The old absolute load>85 rule is removed.
No swap use. Scratch /home/ubuntu/jope-data/h389-dsc01 <=2GBdecimal. Fixed
compute stop23:15Z/close23:30Z Nov5; late means INCOMPLETE. Own process only.
Create-only S3 experiments/rd_breakthrough_20261030/H389/dsc01-20261105T190500Z/;
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
