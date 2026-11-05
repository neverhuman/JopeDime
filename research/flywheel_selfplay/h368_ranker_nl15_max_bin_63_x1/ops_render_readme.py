"""Render H368 scientific outcome from stopped, reviewed evidence."""
from pathlib import Path
import json
P=Path(__file__).parent;S=P/'source';m=json.loads((S/'metrics.json').read_text());run=json.loads((S/'run_receipt.json').read_text());a=run['audit'];full=m['status']=='COMPLETE';outcome=m['verdict'] if full else m['status']+'; scientific UNKNOWN'
if full:
 result='| Arm | Median Elo | Median family mean gain | Literal family median | Median harmed |\n|---|---:|---:|---:|---:|\n'
 for name in ['BASE','NL15','MB63']:
  v=m['aggregate'][name];result+=f"| {name} | {v['median_elo']:.6f} | {v['median_gain']:+.9f} | {v['literal_family_median_gain']:+.9f} | {v['median_harmed']:.4%} |\n"
 for field,label in [('paired_drop_top10','PRIMARY vs archived H352 NL15'),('paired_vs_BASE_drop_top10','Secondary vs BASE')]:
  p=m[field];result+=f"\n{label}: drop10 median **{p['median']:+.12f}**, CI95 {p['ci95']}, one-sided LCB95 **{p['lcb95']:+.12f}**.\n\n| Root | Paired median | CI95 | LCB95 |\n|---:|---:|---|---:|\n"
  for r in p['per_root']:result+=f"| {r['root_seed']} | {r['median']:+.9f} | {r['ci95']} | {r['lcb95']:+.9f} |\n"
 result+='\nFour primary gates: `'+json.dumps(m['criteria'])+'`. Secondary BASE is descriptive and adds no gate.\n\n| Partition | Root | Arm | Family mean | Family median | Drop10 | W/T/L | Harm | Invalid | Switches | Elo | Elo CI95 |\n|---|---:|---|---:|---:|---:|---|---:|---:|---:|---:|---|\n'
 for c in m['cells']:result+=f"| {c['partition']} | {c['root_seed']} | {c['arm']} | {c['family_mean_gain']:+.6f} | {c['family_median_gain']:+.6f} | {c['drop_top']['10']:+.6f} | {'/'.join(map(str,c['family_wtl']))} | {c['harmed_share']:.3%} | {c['invalid']} | {c['switches']} | {c['elo']:.2f} | {c['elo_ci95']} |\n"
 result+=f"\nConfig agreement MB63/BASE {m['config_equal_count']}/36; MB63/NL15 {m['config_equal_vs_NL15_count']}/36. All27cells preserve family mean/median CIs, drop1/5/10, LOFO minimum, W/T/L, invalids, switches, EloCI and pick/lock hashes. Aggregate intervals and seed Jaccards are in metrics.\n"
else:result=f"Scientific MB63 Elo/EloCI, primary and secondary comparisons/root CIs and decision are **UNKNOWN/null**. Available {a['completed_fold_root_locks']} locks, {a['outer_score_markers']} score markers and {a['completed_native_fit_records']} completed native records do not complete the36-task design. No partial aggregate or numerical no-advance. Stop {run['resource_stop']}; actual exit {run['execution']}.\n"
text=f'''# H368 — NL15 ranker max_bin 63 on xbabe1

**{outcome}**. [PREREG.md](PREREG.md) published nonforce v1/next **{m['prereg_commit']}**, {run['prereg_published_utc']}, before first actual fit {a['first_actual_fit_started_utc']}.

## Treatment and proof

Copy committed H362 e976935dc7a4dbdbaa6047bc659966c7c18e0480 design/transport, with H352 02dfee2f7e0b8543324494dd4f99d074cd6b93b6 NL15/depth3 science. Sole ranker native delta: max_bin255→63 in topk.fit_ranker and bagged.fit_ranker_n, every inner/final/eight bag ranker. Added keyword removal restores both complete H352 ASTs. Num_leaves15/max_depth3 remain fixed (at most8 actual leaves). Original LambdaRank/ndcg/sigmoid1/truncation8/eval_at1,8, labels/default gains, L2=1/implicitL1=0/minleaf20, LR/rounds/grid/seeds/features/ties1e-4/sentinel−2 stay unchanged. Veto binary GBDT L1=1/L2=2 unchanged; full acceptance requires all108 final veto SHAs identical to BASE.

Before fits, recovered H352972saved rankers and29484actual fit rows prove native max_bin255/depth3/leaves15. Every completed native fit checks all constructor/native fields and full H352 constants/seed-derived fields with only ranker max_bin changed. Every actual Dataset records raw numpy input, fresh handle, same-fit reference, constructor/effective max_bin63(rankers)/255(veto), and per-feature native bin counts. Prebuilt/ignored bins fail INCOMPLETE_SPEC_MISMATCH. Dataset fixtures construct bins without optimizer updates. [actual_tree_summary.json](actual_tree_summary.json) retains ordered-tree audit summaries; complete arrays/native/Dataset records remain in S3.

Same P0/P1/P2 × roots30001–30003 ×4outer folds=36tasks,1219TRAINcontexts/205families. Parent has metadata only; fit hooks admit only outer-TRAIN feature ranges and deny comparator outcomes/native copies. All36locks/directories fsynced and native/input/veto/custody reviews pass before any outer packet access. Outer score exactly once. Historical TRAIN exposure remains; no independent fresh confirmation.

## Inputs and local verification

Fresh exact-Version whole/member H316/H315 proof on xbabe2, then every copied SHA and BASE9panels/36pick+lockSHA/3exactmedians on xbabe1. Fresh H352manifest/27parts whole+2691memberSHA, then all copied SHA and NL15ninepanels/36locks+scores/rawfamilyarray regression on xbabe1. Current selected/all1219TRAINpools/r882cost/fourTRAINfeatureprefix SHA exact. Owned cache contains only consumed TRAIN features; protected mixed-cache tails unread. No BASE/NL15 refits or new predictions. See [regression_receipt.json](regression_receipt.json), [nl15_regression_receipt.json](nl15_regression_receipt.json), [run_receipt.json](run_receipt.json).

84local tests passed on both hosts,0failed/skipped/optimizerfits, including full reverseAST, original parameter stream, Dataset bins/prebuilt refusal, complete native/default/seed drift, comparator-distinguishing report fixture, primary decision boundaries, outer-before-lock refusal, veto identity, corrected resource windows/forecasts and credential-free transport. Frozen44source SHA exact while fitting. Thirteen source-equivalence groups preserve all undeclared H362 AST/function bodies. Same implementing agent performed substantive source/evidence review; no independent/hosted-CI claim. Actual audit: `{json.dumps(a)}`; qualification **{a['full_treatment_qualification']}**. Actual unit exit: `{json.dumps(run['execution'])}`.

## Outcome

**{m['elo_label']}**.

{result}

[metrics.json](metrics.json) is below50KB with exact DOPE frozen/unloaded block and SHA-linked full receipts. Whole-family paired bootstrap10000/RNG368191; ordinary302150 unchanged. Shared family draws across arms/cells, each arm drops its own best10; median9cell differences and each root median3partitions/CIs. ADVANCE requires primary-vs-NL15 drop10 median≥+.002 AND LCB95>0, median full gain≥90%NL15, harmed≤NL15, and≥2/3positive root medians. All four required; ties never advance. Complete failure is MB63_NOT_ADVANCED; incomplete UNKNOWN. Sequential multiplicity H348–H368 uncorrected; any TRAIN advance needs separately authorized one-shot dev91 confirm outside H368. No speed or AUC verdict.

## Resources and custody

xbabe1 CPU-only, nice19/math1/MemoryMax6GiB/Swap0/CPUQuota800%. One5sec bothphysical+SMT85%-idle gate, first≤8/min4 from24..35+sibs88..99 with same-window hostCPU≤70. Actual admission `{json.dumps(run['cpu_admission'])}`. H352CPU26189.516914/.9eff/1.15safety/+300reserve forecasts `{json.dumps(run['runtime_forecasts'])}`. Fixed16:30compute/16:45close Nov5; no reprobe/restart/reset/cutcells/partialscore. Correct forecast-miss unit h368-x1-run.service frozen before launch.

Corrected guard: immediate MemAvail<20GiB/rootfree<80GiB/scratch>2GB, hostCPU>85% overfull60sec, load>.85×hostonline128=108.8 on3consecutive60sec samples, or fixeddeadline. Record tick pairs/windows/streak; signal only own units. Actual resources `{json.dumps(run['resources'])}`. All AWS/S3 operations xbabe2 only; no AWS credentials/client onxbabe1. Own courier/archive use nice19/core62/CPUQuota100%/6GiB/Swap0 with conservative CPU70/loadfloor(.85*hostN)/Mem20GiB own-I/O backoff. No foreign process, stopped writer, marker or run-dir changes.

Create-only S3 prefix experiments/rd_breakthrough_20261030/H368/solhwv-x1-20261105T133000Z/. After input regression,2232historical native copies/576553824bytes EACHhost were archived exact-Version whole+everymember verified and rehashed before pruning. Comparator arrays/locks/scores/native-referenceJSON/TRAINinputs remain; no treatment data removed by this step. Inner archive requests are SHA-pinned to the xbabe2 courier; returned exactVersion whole/member proof required before raw/duplicate deletion. All stopped outputs copied back and rehashed, final whole/everymember proof before both own scratch exactsets/allSHA rechecked and removed. [s3_manifest.json](s3_manifest.json) pins final/reference/innerobjects; full models/native/Dataset/tree/locks/score/input/probe/forecast/guard/exit/tests/code/control evidence preserved, no models Git. Operations snapshots in operations_manifest.json.

Nonforce fresh-parent v1/next only, oneH368row each index; all other rows and canonical foreign HEAD/index/files preserved. CI_FREEZE+BATCH_WINDOW/noPR/main/fastlane/hostedCI/FINDINGS/newlane. DOPEc51a56d0 frozen/unloaded; no GPU/qualification.lock/xbabe0/xbabe3/foreignjobs/dev91/DL-Test/sealed/holdout/meta-test/freshcohort. H362 and all prior closed waves retained. No further H368 fit/score/probe/dev91 action after release.
'''
(S/'README.md').write_text(text);print('README rendered',outcome)
