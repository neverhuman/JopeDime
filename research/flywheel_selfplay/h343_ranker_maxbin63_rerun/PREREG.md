# H343 — Coarser ranker histograms, max_bin=63

Authority hw_t2330_astra_h343_maxbin_rerun.md. H339 CLOSED5d30d93c235b with unknown
scientific outcome after its runtime forecast failed. One fixed MAXBIN63 arm,
no tuning. Hypothesis: coarser split candidates improve family robustness.
The historical train/validation gap does not establish threshold overfitting in
these rankers. max_bin is a cap; realized bins depend on values and defaults,
so 255→63 is not a guarantee of fourfold fewer thresholds or a speed benefit.

## BASE verification and sole treatment

Exact-version H318 recovery completed before this prereg, with zero fits or
predictions. All972archived BASErankers and108veto models explicitly record
native max_bin255 and no per-feature max_bin override. Pinned H287
213b30ad52e1ce48927bba960ed609bd98ab4dfe topk.fit_ranker/bagged.fit_ranker_n
omit max_bin; the effective native default is255, above63. InstalledLightGBM4.6.0
and sklearn.fit/Dataset.__init__ source hashes recorded in base_native_maxbin.json.
If effective BASEmax_bin<=63, stop INCOMPLETE_SPEC_MISMATCH before anyfit.

Add exactly one constructor keyword, max_bin=63, to those two ranker functions,
covering all inner main rankers, final refits and eight bags. Removing this
keyword restores exact BASE ASTs. Every actual fit compares its complete
constructor parameter dictionary against an unfitted BASE constructor; the sole
changed key must be max_bin. No BASE scientific refit is used for this check.

Every ranker accepts raw NumPy feature arrays, rejects pre-binned Dataset inputs,
and calls sklearn.fit, which constructs a fresh Dataset. An observer around the
sklearn Dataset constructor checks a new object, initial native handleNone,
referenceNone and max_bin63; after training, checks its native feature bin counts
are<=63. Each process/sequence identity is unique. No Dataset or bin-map reuse
from BASE, reference linkage or binary dataset cache. The observer delegates
unchanged to the original constructor and is restored before veto fitting.
Audit actual Booster max_bin/LR/round budget/model hash and all Dataset counts.

All else BASE: LambdaRank/ndcg/ordinal0..7/defaultlabel_gain, GBDT, depth3/leaves7,
row subsample.8/frequency1,colsample_bytree.8,L2=1,minleaf20, original seeds and
features. LR_GRID.03/.05/.10; topk200rounds, selected bag50/100/200; unchanged
prediction prefixes. H339LRhalving/rounddoubling absent. Veto code/params and
native bins255 unchanged; all108finalveto hashes must match BASE. No callbacks,
early stopping, weights, extra_trees, newobjective, column masks or softvotes.
Original LR/n/margin and18-config tau/k/agreement selection, ties/switches and
invalid utility-2.0 remain unchanged. Selected configs may differ as an effect.

## Regression, physical closure and sequence

TRAIN1219contexts/205families; H303NESTED/H308 P0/P1/P2 × roots30001–30003
×4familyouterfolds. Before treatment fits, fresh unchanged archivedBASE replay
must match36fold/root pickhashes,9pick/rawutility panels and3medians exactly:
.0643163334285782,.04821480030677507,.04360142642936269.
Mismatch REGRESSION_FAIL stops. No BASEscientificrefits; recovered files0444.
Required BASE replay reads TRAINouterrows separately. The zero-prelock-outer-read
assertion applies to treatmentworkers, not that reference verification.

Each worker physically admits only its outer-TRAIN pool/cost/cache byte ranges;
no outer packets prebuilt; historicalBASEoutcome access forbidden during fits.
819rankerfits percell,29484for36; unique fresh Dataset for each. Full source/
parameter/native-bin/input/selection audit and36fsynced locks precede all36once-only
fold/root scores forming9panels. No partial scoring, hidden restart or cutcells.
Three synthetic control fits only: two treatmentrankers plus one unchangedgate;
one baseline-bin synthetic Dataset is constructed without fitting.32controls
cover AST/native bins/fresh Dataset/nonreuse/unchanged parameters and gates,
physical boundaries/forecast/analysis. Prereg precedes even these fixture fits.

## Resource admission and first-cell forecast

CPUxbabe2,nice19,numeric1,MemoryMax6GiB/Swap0,RSS<=6GiB,hostCPU/load<=85,
MemAvailable>=20GiB,scratch<=2,000,000,000bytes under/home/ubuntu/jope-data/h343-astra.
After prep, ONEfresh5secondper-core+SMT probe; admit only pairs each>=85%idle.
Assigned21/22/23/24/25/26/28/30/32/33 andSMT85/86/87/88/89/90/92/94/96/97; >=4admitted
required before any scientific fit; take the first at most8admitted workers. Exclude all H342/H338/H335
reserved siblings; verify actual worker affinities. Sleeping launch supervisors
are classified separately only after checking declared CPUs and all descendants.
No re-probe, borrowed cores or foreign process action. Fewerthan4 => INCOMPLETE
without scientific fits; no false scientificnegative.

Run P0/fold0/root30001 first. Include its verified model custody in elapsedtime.
Then project remaining35 as ceil(35/admitted_workers)*first_cell_seconds*1.15+600s.
Require finish strictly before fixed03:15ZNov5;15%reserve for cells and600s for
audit/scoring/analysis. Miss=>INCOMPLETE_RUNTIME_FORECAST, unknownverdict,
no furthercells/outer scores. No timing substitution, reserve reduction or reset.
Fixedclose03:30ZNov5. No runtime/speed superiority verdict from this gate.

## Frozen metrics and decision

Raw utility vsr882slot0; contextmeans withinfamily,equalfamilyweights,median9cell
mean gains; literal familymedians separately. W/T/Ltol1e-4; harmedgain<-.01;
invalid/switch countscontexts. Drop largest1/5/10 gains separatelyperarm;
LOFOmin equalsdrop1, not independent evidence.10kpairedwholefamily bootstrap,
RNG343191, shares205familyindices acrossarms/all9dependentcells. Decision
statistic median9(MAXBIN63arm-specificdrop10−BASEdrop10);CI.025/.975,LCB.05.

MAXBIN_ADVANCE iff pairedmedian>=+.002 ANDLCB95>0 ANDmedian gain>=.9BASE
ANDmedian harmed<=BASE AND>=2of3rootpairedmediansstrictlypositive.
Else MAXBIN_NOT_ADVANCED only after complete study. Exact+.002inclusive;
zeroLCB/tiesfail;zerorootnotpositive. Missing/incomplete metricsnull, unknown.
Report all9cells/roots,CIs,drop1/5/10,LOFO,WTL,harm,invalid,switches,configagreement
andseedJaccard. CONFIG_INSENSITIVE>=30/36matchingconfigs is diagnostic only.
Elo inherited400-stepBradley-TerryMM,pseudocount.01,tiesomitted, label exactly:
**reference = r882 slot0 (DL-Train outer); r3175 not measured on TRAIN**.
NoAUC/speedverdict. Bootstrapconditionsonfits,doesnotrepeatselection or correct
adaptivewave search; TRAINreused and9cellsdependent. ADVANCEwouldrequireseparate
one-shotdev91vsr3175prereg; noneauthorizedhere. No post-outcomechanges.

## Custody and restrictions

Create-only S3 experiments/rd_breakthrough_20261030/H343/astra-20261104T232500Z/;
exactVersionGET/wholeSHA/everymember verification before owninner/finalscratch
cleanup. Original localH50/H58/H276S3versionsremainunestablished; physicalpacket/
range identitiesretained. No modelsinGit. Privateindex/nonforcev1nextonly, own
study andoneeachINDEX/overnightrow,claim/status/FINDINGS/close records.
CI_FREEZE+MASTER_MERGE_HOLD:noPR/main/fastlane/hostedCI.
NeverGPU/qualification.lock/xbabe0/3/livewriter/foreignjobs/H335/H338/H342/
DL-Val/dev91/DL-Test/sealed/holdout/meta_test/fresh. DOPE
`{"composite_sha256_prefix":"c51a56d0","loaded":false,"scope":"CPU selection layer only"}`.
No neural optimizer update, DOPE weight load/modification or serving promotion.

<!-- Source provenance: unchanged H341 experiment; prereg f11e106f16d9650ab56e2a02eb8675ecf7e74e56; implementation 98c3c6f7531e8de5838c9e33f9b79ff89feb7c3f. H343 changes only wave identity/paths, RNG343191, assigned cores and fixed window. -->
