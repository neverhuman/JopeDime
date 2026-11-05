# H366 — NL15 ranker bagging_fraction 0.8 to 0.7

Owner sol_v1_dsc01; authority /home/ubuntu/nudges/hw_t1230_dsc01_h366_nl15_bagfrac07.md.
Copy committed H352 02dfee2f7e harness, retaining leaves15/depth3. Reuse the
proven H360 protocol from cf98bf2749aff9be3a531931cf8eddd85da5b342; H360
is CLOSED COMPLETE NL15BBQ_NOT_ADVANCED and remains closed. This is one new
owner-authorized wave; no retry, budget reset, or continuation of old science.
Publish complete source and prereg non-force to origin/v1/next BEFORE fits.

## Hypothesis and sole existing-value intervention

Lower ranker row bagging fraction from .8 to .7 on NL15 to reduce dependence
on row-specific outliers and the -2.0 sentinel chase, using a milder reduction
than the unqualified .6 design. ONE fixed arm BF07 changes only the existing
subsample=.8 value to .7 in both pinned topk.fit_ranker and bagged.fit_ranker_n
after the exact committed H352 leaves transform. Inner/final/all8bag rankers
receive it. Restoring only subsample=.8 must restore each whole H352 effective
AST exactly; keyword sets are identical. No added bagging seed keyword or other
new keyword. Native bagging_by_query remains0, original seeds unchanged.

Archived H352 native fraction must be.8, freq1, feature_fraction.8,
leaves15/depth3/minchild20/L2one/implicitL1zero, or stop
INCOMPLETE_SPEC_MISMATCH before treatment. Check all972 native ranker configs
against the frozen H352 defaults, original LR/tree inputs and all original
seeds/six derived seeds. Every treatment native fit must show EXACTLY
bagging_fraction[0.8,0.7] versus its unchanged H352 configuration; veto delta
empty. Preserve lambdarank/sigmoid1/ndcg/trunc8/labels0..7/defaultgain/features/
groups/LR/trees/seeds/18configs/tie-switch1e-4/sentinel-2 and all other settings.
Veto unchanged, all108 final veto SHAs equal BASE/H352. No tuning, second arm,
calibration, BASE or NL15 refits. Three roots mean exactly three training seeds
30001/30002/30003, shared across the three fixed TRAIN partitions.

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
RNG366191. Paired comparisons share family draws across all9cells and arms;
each arm drops its own top10, then median across9cells. CI95 .025/.975;
one-sidedLCB95 .05. Rootmedians/CIs use the same draws over3partitions each.
ADVANCE iff primary paired drop10median>=+.002 AND one-sidedLCB95>0 AND
median fullgain>=.9*NL15 median AND median harmed<=NL15 median AND paired
rootmedian>0 on>=2/3roots. Otherwise BF07_NOT_ADVANCED only after full
qualified scoring; ties/subthreshold never advance. Incomplete science UNKNOWN/
null, never zero or a negative verdict. Secondary BASE comparison never decides.
No AUC or speed verdict. Sequential single-knob sweep H348–H366 on the same
folds/roots; ADVANCE requires separately authorized one-shotdev91 confirmation,
NOT in H366. This campaign reads no dev91 rows or H356 model payloads.

## Admission, corrected guard and fixed window

xbabe2 CPU only/nice19/numeric1; systemd MemoryMax6G/SwapMax0/CPUQuota<=800%.
Only physical27..37, siblings91..101. ONE fresh5second simultaneous per-core+
SMT idle probe; bothsiblings>=85%, first<=8eligible physicals, require>=4,
same-interval hostCPU<=70%. Create-once probe token; no reprobe/reset/borrowed
cores. Controller/monitor/fit/score bind admitted physicals, no sibling workers.
H35738..53/102..117 and every foreign job remain untouched.

Forecast reference=H352 actual full36 CPU26189.516914s, bound by
actual_resources_reference.json and its exact original UNIT-FINAL receipt SHA.
Admission forecast
reference/(workers*.9)*1.15+300s. After first completed task, remaining35*
max(reference/36,actualfirsttaskCPU)/(workers*.9)*1.15+300s from currenttime;
ongoingcredit0. Both must fit strictly before fixed2026-11-05T16:30:00Z,
else INCOMPLETE_RUNTIME_FORECAST. No reduced cells/reprobe/reset/restart.

OWNER-CORRECTED GUARD: stop only own study cgroup for MemAvailable<20GiB OR
hostCPU>85% averaged over60seconds OR load1>108 on THREE consecutive60second
samples OR scratch>2GBdecimal, plus the fixed compute deadline. Load<=108
resets the consecutive counter; CPU=85% is allowed. Five-second CPU telemetry
is diagnostic only and never triggers the60second CPU guard. Memory/scratch/
deadline remain checked every5seconds. The old absolute load>85 rule is removed.
No swap use. Scratch /home/ubuntu/jope-data/h366-dsc01 <=2GBdecimal. Fixed
compute stop16:30Z/close16:45Z Nov5; late means INCOMPLETE. Own process only.
Create-only S3 experiments/rd_breakthrough_20261030/H366/dsc01-20261105T123000Z/;
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
