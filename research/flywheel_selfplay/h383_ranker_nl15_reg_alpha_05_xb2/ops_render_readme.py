"""Render full or honestly incomplete H383 evidence after the local unit stops."""
from pathlib import Path
import json
P=Path(__file__).parent;S=P/'source';read=lambda p:json.loads(p.read_text());m=read(S/'metrics.json');run=read(S/'run_receipt.json');a=run['audit'];full=m['status']=='COMPLETE';outcome=m['verdict'] if full else m['status']+'; scientific UNKNOWN'
if full:
 result='| Arm | Median Elo | Median family mean gain | Literal family median | Median harmed |\n|---|---:|---:|---:|---:|\n'
 for name in ['BASE','NL15','L1A05']:
  v=m['aggregate'][name];result+=f"| {name} | {v['median_elo']:.6f} | {v['median_gain']:+.9f} | {v['literal_family_median_gain']:+.9f} | {v['median_harmed']:.4%} |\n"
 for field,label in [('paired_drop_top10','PRIMARY vs archived H352 NL15'),('paired_vs_BASE_drop_top10','Secondary vs BASE')]:
  p=m[field];result+=f"\n{label}: median **{p['median']:+.12f}**, CI95{p['ci95']}, LCB95 **{p['lcb95']:+.12f}**.\n\n| Root | Paired median | CI95 | LCB95 |\n|---:|---:|---|---:|\n"
  for r in p['per_root']:result+=f"| {r['root_seed']} | {r['median']:+.9f} | {r['ci95']} | {r['lcb95']:+.9f} |\n"
 result+='\nFour primary gates: `'+json.dumps(m['criteria'])+'`. The secondary BASE comparison adds no gate.\n\n| Part | Root | Arm | Mean | Literal median | Drop10 | W/T/L | Harm | Invalid | Switches | Elo | Elo CI95 |\n|---|---:|---|---:|---:|---:|---|---:|---:|---:|---:|---|\n'
 for c in m['cells']:result+=f"| {c['partition']} | {c['root_seed']} | {c['arm']} | {c['family_mean_gain']:+.6f} | {c['family_median_gain']:+.6f} | {c['drop_top']['10']:+.6f} | {'/'.join(map(str,c['family_wtl']))} | {c['harmed_share']:.3%} | {c['invalid']} | {c['switches']} | {c['elo']:.2f} | {c['elo_ci95']} |\n"
 result+=f"\nConfiguration agreement: BASE/L1A05 {m['config_equal_count']}/36, NL15/L1A05 {m['config_equal_vs_NL15_count']}/36. All 27 cells retain mean and median CIs, drop-top-1/5/10, LOFO minimum, W/T/L, harmed share, invalid selections, switches, Elo CI, pick and lock hashes, and seed Jaccard agreement.\n"
else:result=f"L1A05Elo/EloCI, bothpaired/rootCIs, criteria/decision **UNKNOWN/null**. Scientificdraws0/planned10000/RNG383191. Available locks{a['completed_fold_root_locks']}, markers{a['outer_score_markers']}, nativefitrecords{a['completed_native_fit_records']} do not complete the36-task design. No partial aggregate or numeric no-advance. Actualexit{run['execution']};stop{run['resource_stop']}.\n"
r=run['resources'];f=run['runtime_forecasts'];execution=run['execution']
text=f"""# H383 — NL15 ranker L1 regularization on xbabe2

**{outcome}**. This TRAIN-only experiment changes ranker `reg_alpha` from 0 to 0.5 on the archived H352 NL15 configuration. The primary comparator is H352 NL15; BASE is secondary. The higher median Elo does not override the preregistered family-level advancement gates.

[PREREG.md](PREREG.md) was published at {run['prereg_published_utc']} in commit `{m['prereg_commit']}`. The first actual fit began at {a['first_actual_fit_started_utc']}. H376 remains closed as `INCOMPLETE_RESOURCE_GUARD`, with zero fits and its original source and receipts preserved.

## Frozen science

FROZEN-SCIENCE-EQUIVALENCE passed for all 16 complete scientific AST groups against H376 after the declared wave, path, host, RNG (383191), and window substitutions. All 45 original source files retained their frozen hashes throughout execution.

The sole scientific change from H352 NL15 is ranker `reg_alpha=0.5` / native `lambda_l1=0.5`, applied to inner fits, final refits, and all eight bag rankers. Fixed settings remain leaves 15, depth 3, L2 1, minimum leaf samples 20, row fraction 0.8 with frequency 1, column fraction 0.8, and max bin 255. Removing the added keyword restores both complete H352 function ASTs. No seed keyword was added. Labels, groups, gain mapping, LambdaRank defaults, metric settings, search grid, rounds, features, sentinel (-2), and tie rule (1e-4) remain unchanged.

Veto classifiers retain L1 1 and L2 2; all 108 final veto hashes match BASE. Every completed fit has native parameter, H352 default/seed, physical Dataset, and ordered tree-depth/leaf evidence. Depth 3 bounds actual ranker leaves to eight even though configured leaves are 15. Dataset max bin is 255 for both model roles; no prebuilt reference bins were supplied.

## Admission and failed startup

The only capacity probe followed a read-only observation that `h377-run` was inactive at 17:21:56Z. The command returned status 4 for a garbage-collected transient unit. One five-second interval covered primary cores 27–37, then fallback 12–14, and their SMT siblings. It admitted four cores in priority order `[31, 12, 13, 14]`; both siblings of every pair were at least 85% idle, with host CPU 24.4157%.

The original startup compared sorted kernel affinity with that priority-ordered list and exited 1 before any forecast, constructor, optimizer fit, lock, or score. The failed invocation and logs remain evidence of failure. [RESOURCE_AMENDMENT.md](RESOURCE_AMENDMENT.md) was published at {run['operational_notice']['utc']} in commit `{run['operational_notice']['head']}`, before every actual fit. It documents a reviewed set-comparison continuation using the same allocation, frozen source, RNG, and fixed deadline. Admission had passed; no second probe, new cores, scientific replay, or budget reset occurred.

Three operator tests passed: nonascending admission order, ascending order, and refusal of a different core set. [startup_continuation_receipt.json](startup_continuation_receipt.json) binds the tested entrypoint hash. The original failed invocation was `{run['actual_startup_exit']['INVOCATION_ID']}`; the actual scientific execution was `{execution['INVOCATION_ID']}`, which exited {execution['EXIT_STATUS']} at {execution['utc']}. The no-restart rule for an admission, forecast, or guard miss remains binding.

## Inputs and evaluation

Fresh exact-Version S3 reads verified BASE H316/H315 as whole objects and members, and the H352 manifest plus all 27 parts / 2,691 members. BASE reproduced nine panels, 36 fold/root pick and lock hashes, and three exact partition medians. H352 NL15 reproduced nine panels, 36 locks/scores, and raw family arrays exactly on xbabe2 before treatment fits. Archived H352 headers cover 972 saved rankers and 29,484 actual fit records, proving original ranker L1 zero and max bin 255. No comparator was refitted or used for new model predictions.

Selected inputs, all 1,219 TRAIN pools, r882 cost data, and four consumed TRAIN feature prefixes passed identity checks. [regression_receipt.json](regression_receipt.json), [nl15_regression_receipt.json](nl15_regression_receipt.json), and [run_receipt.json](run_receipt.json) retain the evidence. Protected feature tails were excluded.

The design is P0/P1/P2 × roots 30001–30003 × four outer folds: 36 tasks, 1,219 TRAIN contexts, and 205 families. Fit parents consumed metadata only; worker hooks admitted outer-TRAIN ranges and denied comparator outcomes and native copies. All 36 locks and directories were fsynced and their native, physical input, veto, and custody reviews passed before cold outer packets and once-only scoring. This is historically exposed TRAIN evidence, with no fresh independent confirmation. **{m['elo_label']}**.

{result}

[metrics.json](metrics.json) is below 50 KB and includes the exact frozen/unloaded DOPE block and SHA-linked receipts. Paired whole-family bootstraps use 10,000 shared draws and RNG 383191; ordinary reporting retains RNG 302150. Each arm drops its own best ten families. The paired summary is the median of nine cell differences; each root summary takes the median over three partitions and includes its CI.

Advancement requires all four preregistered gates: paired NL15 drop-top-10 median at least +0.002 with LCB95 above zero; median full gain at least 90% of NL15; harmed share no higher than NL15; and positive root medians on at least two of three roots. Ties never advance. BASE adds no gate. A completed failed gate yields `L1A05_NOT_ADVANCED`; incomplete work stays UNKNOWN. Multiplicity across H348–H383 is uncorrected. Any TRAIN advance would require separately authorized one-shot dev91 confirmation outside H383. No speed or serving conclusion is made.

## Verification and resources

All 89 source/fixture tests and three startup operator tests passed on xbabe2, with zero failures or skips. Fixtures performed no optimizer fits. Tests cover the full AST inverse, native defaults and seeds, Dataset and veto identity, physical outer-lock barriers, poisoned input tails, decision ties and root CIs, admission, release gating, and guard windows. Actual treatment qualification is **{a['full_treatment_qualification']}**: {a['completed_native_fit_records']} completed native fit records, {a['reviewed_saved_models']} saved-model reviews, {a['completed_fold_root_locks']} locks, {a['outer_score_markers']} complete score markers, and {a['final_veto_SHA_matches']} matching final veto hashes. [actual_tree_summary.json](actual_tree_summary.json) summarizes model structure; full ordered native/Dataset/tree records are retained in S3. Review used the same implementing agent and performed no new fits or predictions.

Execution used nice 19, one numeric thread per worker, the same four admitted cores, MemoryMax 6 GiB, SwapMax 0, and CPUQuota 800%. Guards applied only to the owned unit: MemAvailable below 20 GiB, scratch above 2 GB, host CPU above 85% over a full 60-second window, load above 108 for three consecutive 60-second samples, or the fixed 20:45Z cutoff. Root disk free space was telemetry, without the old xbabe1 guard. Closure was fixed at 21:00Z.

Observed maxima: 60-second CPU {r['max_cpu_60s']:.4f}%, load {r['max_load1']:.4f}, scratch {r['max_scratch_bytes']} bytes. Minimum available memory was {r['min_mem_available_bytes']} bytes. The maximum short CPU sample was {r['max_cpu_5s_telemetry']:.4f}%; it did not cross the declared full-minute guard. The high-load streak remained {r['max_consecutive_high_load_60s']}. Admission and first-completed-task forecasts passed, ending at {f['initial']['forecast_end']} and {f['after_first_completed_task']['forecast_end']}. They used H352 CPU 26189.516914 seconds, efficiency 0.9, safety 1.15, and 300 seconds of scoring reserve. The second forecast used max(reference/36, first-task CPU) for 35 remaining tasks with zero credit for ongoing work. Startup delay counted against the original window.

## Custody and publication

Only the owned H383 S3 prefix `experiments/rd_breakthrough_20261030/H383/hwv-xb2-20261105T170500Z/` was written, create-only. Before fitting, 2,232 historical native copies (576,553,824 bytes) were rehashed and pruned only after whole-object and every-member exact-Version verification of their archive. Comparator arrays, locks, scores, native JSON, and TRAIN inputs remained available. Inner raw/cache files were removed only after their own exact-Version whole/member proofs.

The final archive preserves models, native/Dataset/tree records, locks, scores, inputs, guards, admission, forecasts, release gating, both startup invocations, exits, tests, source, and control code. [s3_manifest.json](s3_manifest.json) pins reference, main, and inner archives. Cleanup requires exact file-set and every-file SHA agreement with verified custody. Models are excluded from Git.

Publication uses a private index and fresh-parent non-force pushes to `origin/v1/next`, with one H383 row in each index and other rows retained byte-for-byte. Canonical HEAD/index and foreign files are preserved. Owner CI_FREEZE and BATCH_WINDOW prohibit PRs, main/fastlane changes, hosted CI, FINDINGS edits, and a new lane. DOPE c51a56d0 remains frozen and unloaded (CPU selection layer only). No GPU, qualification lock, forbidden host access, protected data, dev91, sealed/holdout/meta-test/DL-Test rows, foreign process signals, or flywheel/writer mutation occurred. All H383 ownership is released at closeout; no further H383 model work is authorized afterward.
"""
(S/'README.md').write_text(text)
print('H383 README rendered',outcome)
