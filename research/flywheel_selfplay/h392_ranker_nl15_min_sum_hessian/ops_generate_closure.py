"""Adapt the qualified actual-evidence reviewer without changing active science."""
from pathlib import Path
import ast
P=Path(__file__).parent;O=P.with_name('h383-solhwv');tree=ast.parse((P/'bootstrap.py').read_text());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='adapt');ns={};exec(compile(ast.Module(body=[node],type_ignores=[]),'<host-adapter>','exec'),ns);adapt=ns['adapt']
t=adapt((O/'review_and_report.py').read_text())
def change(old,new):
 global t
 assert old in t,old;t=t.replace(old,new)
change('from native_audit import compare_native,digest,compare_h352_defaults','from native_audit import compare_native,digest,compare_h352_defaults\nfrom depth_patch import MINHESS,BASE_HESS')
t='\n'.join(line for line in t.splitlines() if not line.startswith((' startup=read('," assert sha(P/'continue_admitted.py')","   assert r['started_utc']>amend['utc']")))+'\n'
change("tests['tests_executed']==89","tests['tests_executed']==95")
change("float(native['lambda_l1'])==.5 and float(native['lambda_l2'])==1", "float(native['lambda_l1'])==0 and float(native['lambda_l2'])==1 and float(native['min_sum_hessian_in_leaf'])==MINHESS")
change("q['reg_alpha']==.5 and q['min_child_samples']==20","q['reg_alpha']==0 and q['min_child_weight']==MINHESS and q['min_child_samples']==20")
change("{'reg_alpha':[0,.5]}","{'min_child_weight':[BASE_HESS,MINHESS]}")
change('configured_ranker_reg_alpha=.5,','configured_ranker_reg_alpha=0,configured_ranker_min_sum_hessian_in_leaf=MINHESS,')
a=t.index(' runreceipt=dict(');b=t.index('unqualified_raw_candidate_sha256=',a)
prefix=" runreceipt=dict(parameter_selection=read(S/'hessian_value.json'),prelaunch_review=read(P/'prelaunch-review.json'),derivation_execution=read(P/'derive-unit-exit.json'),derivation_guard_correction=read(P/'derivation-guard-adaptation.json'),derivation_guard_early_interval='NOT_FULLY_COVERED; failure retained; no retrospective pass',preparation_stopped=read(P/'preparation-stopped.json'),current_entrypoint='controller_h392.py',stop_verification=read(P/'stop-verification.json'),actual_environment=read(P/'environment.json'),manifest_environment_note='Inherited manifest environment describes historical template; actual environment is separately observed and pinned',reporting_label_note='Frozen evaluator legacy secondary_label/OVER_NL15 describes PRIMARY NL15 positivity; publication renames that label only. Raw result and all four gates remain unchanged.',"
t=t[:a]+prefix+t[b:]
# The retained suffix contains the original actual_environment field; use it only once.
t=t.replace("unqualified_raw_candidate_sha256=sha(D/'metrics.json') if raw is not None and not full else None,actual_environment=read(P/'environment.json'),", "unqualified_raw_candidate_sha256=sha(D/'metrics.json') if raw is not None and not full else None,")
t=t.replace("forecast_stop_backstop=read(P/'forecast-watch-receipt.json'),forecast_stop_host_adapter='Corrected owned h392-hwv-run forecast-miss target before source freeze; no external watcher required',", "forecast_stop_host_adapter='Frozen source stops owned h392-run.service on first-task forecast miss; tested before treatment',")
change(" m['nl15_regression']=", " m['parameter_selection']=dict(parameter='min_sum_hessian_in_leaf',sklearn_alias='min_child_weight',baseline=BASE_HESS,value=MINHESS,receipt='hessian_value.json',receipt_sha256=sha(S/'hessian_value.json'),archived_rankers=12636,leaves=12967896,calibration_fits=0)\n m['nl15_regression']=")
(P/'review_and_report.py').write_text(t)
for name in ['close_archive.py','launch_archive.py','archive_guard.py','post_archive.py','pre_archive.py','reclaim.py']:
 ast.parse((P/name).read_text(),filename=name)
ast.parse(t,filename='review_and_report.py');print('H392 actual-evidence reviewer prepared; active source unchanged')
