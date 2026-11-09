from pathlib import Path
import json
P=Path(__file__).parent;S=P/'source';m=json.loads((S/'metrics.json').read_text());a=m['audit'];r=m['resource_stop'];trees=m['actual_tree_depth_and_leaves']
text=f'''# H359 — NL15 plus max_depth4, TRAIN only

**INCOMPLETE_RESOURCE_GUARD; scientific verdict UNKNOWN.** The own h359-run cgroup stopped at {r['utc']} because host load {r['load1']:.8f}>85. Actual process exit was killed/TERM; systemd SERVICE_RESULT=success does not establish training completion. No task lock or outer score completed. This is neither an ADVANCE nor a numerical NL15MD4_NOT_ADVANCED result.

[PREREG.md](PREREG.md) was published nonforce at **{m['prereg_commit']}**, {m['prereg_published_utc']}, before the first observed actual fit {a['first_actual_fit_started_utc']}. Source remained frozen. No restart, re-probe, cut cells, partial scoring or deadline reset occurred.

## Hypothesis and setup

H352's fixed NL15 capacity was capped by max_depth3 at eight leaves. H359 keeps num_leaves15 and changes its existing max_depth3→4, permitting up to15 actual leaves and deeper asymmetric splits. The sole ranker constructor delta relative to committed H35202dfee2f7e0b8543324494dd4f99d074cd6b93b6 is depth. Two full-function AST tests restore exact H352 by returning depth to3, including inner, final and all eight bag rankers. The original H352 num_leaves transformation is retained verbatim as the reference. No keyword is added.

All other lambdarank/ndcg/sigmoid1/truncation8, labels0..7/default gains, minleaf20, L2=1/implicit L1=0, Hessian.001/splitgain0, row.8freq1/column.8, LR/trees/selected-round rules, derived seeds/features,18config grid, tie/switch1e-4 and invalid utility−2 stay unchanged. Veto retains binary GBDT L1=1/L2=2. Full completion would require108/108 final veto SHAs identical to BASE; **0** final treatment veto comparisons completed, so that live qualification is incomplete.

Copy committed H352 science, seven exact vendor pins and original H287213b30ad52e1ce48927bba960ed609bd98ab4dfe. Scientific AST equivalence passes five groups for unchanged fit/selection/scoring, packet construction, utility/bootstrap/primary decision, H352 leaves transformation and veto identity. The declared changes are depth, host/path/core/window/RNG adaptations, TRAIN input copies, read-only native/tree telemetry and the preregistered archived secondary comparator.

P0/P1/P2 × roots30001..30003 × four family-grouped outer folds =36mandatory tasks/9panels,1219TRAINcontexts/205families. Fit parent loads metadata only; worker hooks admit only outer-TRAIN pools and exact context feature ranges and forbid archived BASE/NL15 outcomes/native models. Source requires all36fsynced locks and native/constructor/input/veto/custody reviews before once-only outer scoring. The interrupted run produced no guard-complete fold-root receipt: full worker/native/veto/score qualification remains **INCOMPLETE, not passed**. Available controls and partial records do not prove missing coverage.

## Pre-fit comparators and data closure

Fresh exact-Version GET H316/H315 BASE whole/member SHA verified. BASE9panels,36fold-root pick+lock SHA and partition medians .0643163334285782/.04821480030677507/.04360142642936269 are EXACT. All972 BASE rankers have native depth3/leaves7/L2=1/L1=0, and108 archived BASE final veto SHAs match their pins.

Fresh exact-Version H352 recovery manifest and all27 referenced parts: whole SHA AND all2691 members independently verified. Archived NL15 nine panels, all36 lock/score-marker hashes and raw family arrays/means reproduce exactly. All972 saved H352 rankers have native num_leaves15/max_depth3/L2=1/L1=0/minleaf20. All36504 recovered H352 fit records form a fixed native-default/seed-derived-field reference. No BASE or NL15 refits or new model predictions.

Current selected.json, all1219 TRAIN pools, r882 costs and four consumed TRAIN cache feature prefixes match archived SHAs. Owned cache containers stage only those TRAIN feature bytes in original order; mixed-cache feature tails are never copied/read. Identifier/offset metadata is permitted. DL-Val/dev91/sealed feature rows0. Historical TRAIN design exposure remains; this is not independent fresh confirmation.

## Outcome, metrics and decision

**reference = r882 slot0 (DL-Train outer); r3175 not measured on TRAIN.**

NL15MD4 Elo/EloCI, family gains/CIs, both paired comparisons, root medians/CIs and every decision gate are **UNKNOWN/null**. [metrics.json](metrics.json) is {len((S/'metrics.json').read_bytes())}bytes (<50KB) and preserves27 explicitly unmeasured arm/cell placeholders. Scientific paired bootstrap draws0, planned10000/RNG359191; source/fixture bootstrap controls do not constitute a scientific result. No partial aggregate or numerical negative is reported.

On complete data the frozen evaluator reports BASE, NL15 and NL15MD4 family-equal means and literal family medians/CIs, drop-top1/5/10, LOFOmin, W/T/L, harmed(<−.01), invalids, switches/rates, Elo/EloCI, config agreement and seed Jaccard. Whole-family paired bootstrap10000/RNG359191 shares family indices across all arms/cells, with each arm re-dropping its own best10. Primary statistic is median9cell differences; per-root CIs are median3partitions on the same draws. Ordinary interval RNG302150 remains unchanged.

Primary four gates: paired-vs-BASE median>=+.002 and LCB95>0; full median gain>=90% BASE; median harmed<=BASE; strictly positive root paired median on>=2/3. OVER_NL15 additionally requires secondary paired drop10 median>0 and LCB95>0. Conservative interpretation recorded before outcomes: final ADVANCE requires all four primary gates AND OVER_NL15; primary and secondary dispositions are separately retained. Ties, zero LCB and subthreshold margins never advance. This was one of sequential H348–H359 on the same folds/roots; no multiplicity correction or fresh confirmation is claimed. Any eventual TRAIN advance needs a separately authorized one-shot dev91 confirmation, outside H359. No AUC or speed verdict.

## Actual observations and verification

Observed constructor events {a['constructor_events']}, completed native-fit records {a['completed_native_fit_records']}; roles {json.dumps(a['native_fit_roles'])}. Four interrupted in-flight constructor calls may have unlogged returns. The exact total completed optimizer-call count is unknown. Available saved models reviewed {a['reviewed_saved_models']}; unverified interrupted saved models {a['unverified_interrupted_saved_models']}; completed locks0, score markers0, final veto comparisons0. First-task forecast was not reached because no task completed.

Every available native fit passed constructor/native comparisons,111 constant ranker defaults and eight seed-derived fields from the full H352 trace, with only max_depth3→4 changed. Every available saved model was rechecked read-only; no model fit or prediction was added by review. [actual_tree_summary.json](actual_tree_summary.json) records actual per-tree depth/leaf histograms, including BASE/NL15 references and both completed fit records and available saved models. Full ordered [edge depth,leaf count] arrays live in archived native_fit_audit.jsonl and custody/partial-saved-native-audit.json, Tree index given by array position. Ranker depth<=4/leaves<=15 verified; veto uses its unchanged own native limits. These diagnostic observations do not establish an Elo or treatment verdict.

65local source/fixture tests passed,0failed/skipped,actualexit0, no synthetic optimizer fit. Exact H352 reverse ASTs/unchanged keyword stream, constructor/native drift, hidden defaults and seed refusal, input/poison-tail/missing-lock refusals, primary/secondary/tie/harm/retention decisions, paired root CIs, per-tree structures, and one-probe/forecast logic are covered. Same implementing agent reviewed source and actual evidence; no independent/hosted-CI claim. [environment_receipt.json](environment_receipt.json) binds actual versions/kernel/native library SHA. The source manifest and pre-fit frozen file hashes remain exact.

## Resources, custody and next

xbabe2 only, CPU-only, nice19/numeric1/systemd MemoryMax6GiB/MemorySwapMax0/CPUQuota800%. ONE five-second pool54..63+sibs118..127 probe admitted **54,57,58,59**, both SMT85%idle and same-interval hostCPU38.507482%<=70. Four workers launched. Initial forecast from exact recovered H352 CPU26189.516914s/(4*.9)*1.15+300 ended13:02:25.575601Z<13:15; the required first-task reforecast never ran. No new probe or timing fit. Fixed13:15compute/13:30closeNov5. Guards CPU>85/load>85/MemAvailable<20GiB/scratch>2GB affected only the own cgroup. Stop CPU43.411772%, load85.04052734375, MemAvailable87411683328bytes, scratch870910512bytes. Preexisting swap and all foreign jobs were untouched.

Create-only S3 H359/solhwv-20261105T102000Z preserves all partial models/native/tree/constructor/lock-state/inputs/source/probe/forecast/exit/STOP/test evidence. [s3_manifest.json](s3_manifest.json) pins final exact versions after whole SHA and every-member verification, followed by exact-set/all-file rehash before owned scratch removal. Models never Git. Deployed ops_ copies and [operations_manifest.json](operations_manifest.json) preserve the exercised local host adapters; deploy by removing ops_ in custodyP, science beneath source/ and pins beneath vendor/. No SSH/other-host operation is part of H359.

Nonforce fresh-parent origin/v1/next only; one H359 row each wave index. No PR/main/fastlane/hostedCI/FINDINGS/newlane; canonical foreign HEAD/index and dirty files preserved. DOPEc51a56d0 frozen/unloaded; no GPU/qualification.lock/DL-Val/dev91/DL-Test/protected data/other host/live writer/foreign-lane action. Mandatory CLOSEOUT /home/ubuntu/jope_plan/evidence/H359/solhwv-20261105-t1020/CLOSEOUT.json and small receipts /home/ubuntu/jope-rescue-20260921/h359-solhwv/final-receipts.

Next: retain the incomplete outcome. No H359 restart, re-probe, partial scoring or dev91 action is authorized.
'''
(S/'README.md').write_text(text)
print('H359 README rendered with UNKNOWN scientific outcomes and actual guard disposition.')
