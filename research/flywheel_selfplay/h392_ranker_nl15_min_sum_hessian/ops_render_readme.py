"""Render full or honestly incomplete H392 evidence after the local unit stops."""
from pathlib import Path
import json
P=Path(__file__).parent;S=P/'source';read=lambda p:json.loads(p.read_text());m=read(S/'metrics.json');run=read(S/'run_receipt.json');a=run['audit'];full=m['status']=='COMPLETE';outcome=m['verdict'] if full else m['status']+'; scientific UNKNOWN'
if full:
 result='| Arm | Median Elo | Median family mean gain | Literal family median | Median harmed |\n|---|---:|---:|---:|---:|\n'
 for name in ['BASE','NL15','MINHESS']:
  v=m['aggregate'][name];result+=f"| {name} | {v['median_elo']:.6f} | {v['median_gain']:+.9f} | {v['literal_family_median_gain']:+.9f} | {v['median_harmed']:.4%} |\n"
 for field,label in [('paired_drop_top10','PRIMARY vs archived H352 NL15'),('paired_vs_BASE_drop_top10','Secondary vs BASE')]:
  p=m[field];result+=f"\n{label}: median **{p['median']:+.12f}**, CI95{p['ci95']}, LCB95 **{p['lcb95']:+.12f}**.\n\n| Root | Paired median | CI95 | LCB95 |\n|---:|---:|---|---:|\n"
  for r in p['per_root']:result+=f"| {r['root_seed']} | {r['median']:+.9f} | {r['ci95']} | {r['lcb95']:+.9f} |\n"
 result+='\nFour primary gates: `'+json.dumps(m['criteria'])+'`. The secondary BASE comparison adds no gate.\n\n| Part | Root | Arm | Mean | Literal median | Drop10 | W/T/L | Harm | Invalid | Switches | Elo | Elo CI95 |\n|---|---:|---|---:|---:|---:|---|---:|---:|---:|---:|---|\n'
 for c in m['cells']:result+=f"| {c['partition']} | {c['root_seed']} | {c['arm']} | {c['family_mean_gain']:+.6f} | {c['family_median_gain']:+.6f} | {c['drop_top']['10']:+.6f} | {'/'.join(map(str,c['family_wtl']))} | {c['harmed_share']:.3%} | {c['invalid']} | {c['switches']} | {c['elo']:.2f} | {c['elo_ci95']} |\n"
 result+=f"\nConfiguration agreement: BASE/MINHESS {m['config_equal_count']}/36, NL15/MINHESS {m['config_equal_vs_NL15_count']}/36. All 27 cells retain mean and median CIs, drop-top-1/5/10, LOFO minimum, W/T/L, harmed share, invalid selections, switches, Elo CI, pick and lock hashes, and seed Jaccard agreement.\n"
else:result=f"MINHESSElo/EloCI, bothpaired/rootCIs, criteria/decision **UNKNOWN/null**. Scientificdraws0/planned10000/RNG392191. Available locks{a['completed_fold_root_locks']}, markers{a['outer_score_markers']}, nativefitrecords{a['completed_native_fit_records']} do not complete the36-task design. No partial aggregate or numeric no-advance. Actualexit{run['execution']};stop{run['resource_stop']}.\n"
r=run['resources'];f=run['runtime_forecasts'];execution=run['execution'];selection=run['parameter_selection'];environment=run['actual_environment']
text=f"""# H392 — NL15 ranker Hessian floor

**{outcome}**. MINHESS changes only the ranker leaf Hessian floor from **{selection['baseline']}** to **{selection['value']}**, through sklearn `min_child_weight` / native `min_sum_hessian_in_leaf`. The primary comparator is archived H352 NL15; BASE is secondary. Elo does not override the preregistered family-level gates.

[PREREG.md](PREREG.md) was published at {run['prereg_published_utc']} in commit `{m['prereg_commit']}`, before the first actual fit at {a['first_actual_fit_started_utc']}. H383 remains closed as L1A05_NOT_ADVANCED, with its original source and CLOSEOUT unchanged.

## One fixed, archive-derived value

The value was computed once, before treatment, from read-only `Booster(model_str).dump_model()` leaf_weight values in all **{selection['models']} archived NL15 rankers**: 972 final models and 11,664 inner models from 36 compressed archives. BASE copies and all veto classifiers were excluded. All {selection['leaves']} leaves across {selection['trees']} trees received equal weight. Raw Q25 was **{selection['raw_q25']!r}** using `numpy.quantile(method='linear')`; rounding to three significant figures fixed **{selection['value']}**. Every archived ranker floor was {selection['baseline']}. The derived value exceeded baseline, so the mismatch stop did not apply. Calibration fits, optimizer updates, and new predictions during derivation were all zero.

[hessian_value.json](hessian_value.json) pins the archive versions, whole/member checks, pooled leaf sequence hash, and per-model inventory hash. Full inventory rows are retained in S3 custody. No alternate value was selected after outcomes were observed.

## Sole scientific intervention

Both complete H352 ranker function ASTs invert exactly to actual committed H352 source after removing the one added `min_child_weight` keyword. It applies to every inner fit, final refit, and all eight bag rankers. No seed keyword was added. Fixed settings remain leaves 15, depth 3, L2 1, L1 0, min_child_samples 20, row fraction 0.8, bagging frequency 1, column fraction 0.8, path_smooth 0, and max_bin 255. Labels, gain mapping, LambdaRank/ndcg defaults, sigmoid, truncation, metric positions, learning rates, rounds, features, groups, derived seeds, all 18 configurations, sentinel -2, and tie rule 1e-4 remain unchanged.

Veto classifiers retain their own L1 1/L2 2 and baseline Hessian floor. Full qualification requires all 108 final veto hashes to equal BASE. Native constructor/default/seed, physical Dataset, and ordered tree-depth/leaf evidence covers every completed fit. Both model roles use fresh raw-numpy max_bin-255 Datasets without prebuilt/reference bins. Ranker depth 3 bounds actual leaves to eight despite configured leaves 15. Telemetry never changes selection.

## References and evaluation

Fresh exact-Version BASE H316/H315 GETs verified whole hashes and selected members. BASE reproduced nine panels, 36 fold/root pick and lock hashes, and three exact partition medians. Fresh H352 manifest/27 parts verified all 2,691 members and reproduced nine panels, 36 locks/scores, and raw family arrays exactly. The H352 full native trace verifies all 29,484 ranker and 7,020 veto fit headers. Current selected inputs, 1,219 TRAIN pools, r882 costs, and four consumed TRAIN feature prefixes match their archived identities. No comparator was refitted or used for new model predictions. [regression_receipt.json](regression_receipt.json), [nl15_regression_receipt.json](nl15_regression_receipt.json), and [run_receipt.json](run_receipt.json) retain these checks.

P0/P1/P2 × roots 30001–30003 × four family-disjoint outer folds gives 36 tasks and nine panels over 205 TRAIN families. Fit parents use metadata only. Worker hooks admit outer-TRAIN pools and exact TRAIN feature ranges, and deny archived comparator outcomes and models. All 36 locks/directories must be fsynced and their native, physical input, veto, and custody reviews pass before cold outer packets and once-only scoring. No partial scoring is used. This is historically exposed TRAIN evidence, without independent confirmation. **{m['elo_label']}**.

{result}

[metrics.json](metrics.json) is below 50 KB and retains the exact frozen/unloaded DOPE block and SHA-linked receipts. Shared whole-family paired bootstraps use 10,000 draws and RNG 392191; ordinary reporting retains RNG 302150. Each arm drops its own best ten families. The paired summary takes the median of nine cell differences; each root takes the median across three partitions with its paired CI and LCB.

Advancement requires all four preregistered gates: primary paired NL15 drop10 median at least +0.002 with LCB95 above zero; median full gain at least 90% of NL15; median harmed share no higher than NL15; positive paired medians on at least two of three roots. Ties never advance. BASE adds no gate. A completed failed gate yields MINHESS_NOT_ADVANCED; incomplete work stays UNKNOWN. Multiplicity across H348–H392 is uncorrected. Any advance needs separately authorized one-shot dev91 confirmation outside H392. No speed or serving conclusion is made.

## Verification and operational history

All 95 final source/fixture tests passed locally with zero failures or skips and zero optimizer fixture fits. They cover the complete committed AST inverse, native defaults and alias, seeds, percentile/rounding fixtures, Dataset and veto identity, physical outer-lock barriers, poisoned feature tails, decision ties and root CIs, union admission, affinity order, and full-minute guards. An earlier fixture explicitly passed reg_alpha=0 and was correctly refused by the no-extra-keyword guard; the fixture was corrected to the actual H352 implicit default. The guard was not loosened. That preliminary failure remains in custody.

The initial read-only derivation guard daemon rejected the new preparation unit name. Before treatment, the owned whitelist was corrected and an external own-unit guard was attached to the same calculation. The percentile was not restarted or recomputed. [prep_guard_receipt.json](prep_guard_receipt.json) preserves the exception and correction. The early interval is **not retrospectively counted as fully guarded**; systemd one-core/nice19/math1/CPU100/6GiB/Swap0/fixed-deadline limits were enforced. Scientific execution uses the corrected frozen guard and a fresh resource log.

Actual treatment qualification: **{a['full_treatment_qualification']}**. Available evidence includes {a['completed_native_fit_records']} completed native records, {a['reviewed_saved_models']} saved-model reviews, {a['completed_fold_root_locks']} locks, {a['outer_score_markers']} score markers, and {a['final_veto_SHA_matches']} matching final veto hashes. [actual_tree_summary.json](actual_tree_summary.json) summarizes structure; full ordered native/Dataset/tree rows remain archived. The same implementing agent reviewed the work, with no independent or hosted-CI claim and no new fits or predictions during review.

## Resources and custody

Exactly one five-second union admission sample selected {run['cpu_admission']['eligible_cores']}, with both siblings of every pair at least 85% idle and host CPU {run['cpu_admission']['host_at_probe']['cpu_percent']:.4f}%. Primary cores 27–37 precede fallback 12–14. At least four and at most eight cores were allowed; no retry, borrowed core, restart, or reset occurred. Execution used nice19, numeric threads1, MemoryMax6GiB, SwapMax0, and CPUQuota800%. Its actual invocation was `{execution['INVOCATION_ID']}`, with exit {execution['EXIT_STATUS']} at {execution['utc']}.

Only owned units could be stopped. Guards were MemAvailable below20GiB, scratch above2GB, host CPU above85% averaged over a full60-second window, load above108 for three consecutive60-second samples, or the fixed 00:45Z Nov6 compute cutoff. Exactly85% CPU does not stop; load at/below108 resets its streak. Closure is fixed at01:00Z. Root free space is telemetry.

Observed maximum full-minute CPU was {r['max_cpu_60s']}%; maximum short CPU sample {r['max_cpu_5s_telemetry']}%; maximum load {r['max_load1']}; high-load streak {r['max_consecutive_high_load_60s']}; minimum available memory {r['min_mem_available_bytes']} bytes; peak scratch {r['max_scratch_bytes']} bytes. Initial and first-task forecasts are retained in run_receipt.json. They use H352 CPU26189.516914 seconds, efficiency0.9, safety1.15, scoring reserve300s; the remaining35-task forecast uses max(reference/36,actual first-task CPU), with zero ongoing-work credit. The fixed deadline was never reset.

Actual environment was {environment['host']}, kernel {environment['kernel']}, Python {environment['python']}, NumPy {environment['numpy']}, sklearn {environment['sklearn']}, LightGBM {environment['lightgbm']}, native library SHA `{environment['lightgbm_library_sha256']}`. The inherited manifest environment is a historical template record; these separately observed runtime identities are authoritative.

The owned create-only prefix is `experiments/rd_breakthrough_20261030/H392/hwv-20261105T203000Z/`. Before fitting, 2,232 historical copies (576,553,824 bytes) were rehashed and pruned only after whole-object and every-member exact-Version verification of their reference archive. Inner raw/cache removal requires its own exact-Version whole/member proof. [s3_manifest.json](s3_manifest.json) pins reference, main, and inner archives. Final cleanup requires exact file-set/every-file SHA agreement with verified custody. Models never enter Git.

Publication uses a private index and fresh-parent non-force `origin/v1/next` pushes, with one H392 row in each required index and all other rows byte-preserved. Canonical HEAD/index, foreign files, and H383 evidence remain intact. Owner CI_FREEZE/BATCH_WINDOW means no PR, main/fastlane change, hosted CI, FINDINGS edit, or new lane. DOPE c51a56d0 remains frozen and unloaded, CPU selection layer only. No GPU/qualification lock, forbidden host access, protected data, dev91, sealed/holdout/meta-test/DL-Test, foreign process signal, or flywheel/writer/AUTORESTART mutation occurred. Explicit ownership release follows final closeout; no further H392 model work is authorized afterward.
"""
(S/'README.md').write_text(text);print('H392 README rendered',outcome)
