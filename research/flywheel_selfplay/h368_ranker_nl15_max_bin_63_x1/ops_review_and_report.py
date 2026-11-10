"""Review stopped H368 evidence without model fitting or prediction."""
from pathlib import Path
from collections import Counter
import json,hashlib,sys,subprocess,datetime,platform
P=Path(__file__).parent;S=P/'source';D=Path('/home/ubuntu/jope-data/h368-x1');sys.path.insert(0,str(S))
from native_audit import compare_native,digest,compare_h352_defaults
from tree_structure import validate,summarize
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def read(p,default=None):return json.loads(p.read_text()) if p.exists() else default

def main():
 assert read(P/'transfer-verification.json')['status']=='PASS'
 assert subprocess.run(['systemctl','--user','is-active','h368-x2-relay'],capture_output=True,text=True).stdout.strip() not in ('active','activating','deactivating')
 for n,h in read(P/'runtime-source-manifest.json').items():assert sha(S/n)==h,n
 execution=read(P/'unit-exit.json');prereg=read(P/'prereg-published.json');raw=read(D/'metrics.json');stop=read(D/'STOP.json');raw_complete=bool(raw and raw.get('status')=='COMPLETE');timely=datetime.datetime.fromisoformat(execution['utc'])<datetime.datetime(2026,11,5,16,30,tzinfo=datetime.timezone.utc);full=bool(raw_complete and execution['EXIT_CODE']=='exited' and execution['EXIT_STATUS']=='0' and execution['SERVICE_RESULT']=='success' and timely and not stop)
 if full:status='COMPLETE';assert execution['EXIT_CODE']=='exited' and execution['EXIT_STATUS']=='0' and not stop
 elif stop:status='INCOMPLETE_COMPUTE_DEADLINE' if stop.get('reason')==['compute deadline'] else 'INCOMPLETE_RESOURCE_GUARD'
 elif (D/'INCOMPLETE_CPU_ADMISSION.json').exists():status='INCOMPLETE_CPU_ADMISSION'
 elif (D/'INCOMPLETE_RUNTIME_FORECAST.json').exists():status='INCOMPLETE_RUNTIME_FORECAST'
 elif not timely:status='INCOMPLETE_COMPUTE_DEADLINE'
 elif execution and (execution.get('EXIT_STATUS')!='0' or execution.get('SERVICE_RESULT')!='success'):status='INCOMPLETE_EXECUTION_ERROR'
 else:raise AssertionError('No terminal classification; retain unknown')
 if not full and raw is not None:assert raw.get('status')=='COMPLETE','Unexpected raw candidate status; preserve it and retain unknown'
 reg=read(D/'regression.json');nlreg=read(D/'nl15_regression.json');assert reg['status']==nlreg['status']=='PASS'
 tests=read(P/'tests-receipt.json');assert tests['tests_executed']==84 and tests['failed']==tests['skipped']==tests['exit_status']==0
 counts=Counter();native_counts=Counter();dataset_counts=Counter();dataset_caps=Counter();first=[];truncated=[];file_audits=[]
 profiles={scope:{role:dict(models=0,trees=0,depth_histogram=Counter(),leaf_histogram=Counter()) for role in ['ranker','veto']} for scope in ['completed_fit_records','saved_model_reviews']}
 def rows(path):
  data=path.read_bytes().splitlines();out=[]
  for i,line in enumerate(data):
   try:out.append(json.loads(line))
   except json.JSONDecodeError:
    assert not full and i==len(data)-1;truncated.append(dict(file=str(path.relative_to(D)),line=i+1,sha256=sha(path),qualification='UNAVAILABLE interrupted lastrecord retained'))
  return out
 def add(scope,role,trees,native,stored):
  assert summarize(validate(trees,int(native['max_depth']),int(native['num_leaves'])))==stored
  if role=='ranker':assert int(native['max_depth'])==3 and int(native['max_bin'])==63 and int(native['num_leaves'])==15 and float(native['lambda_l1'])==0 and float(native['lambda_l2'])==1
  p=profiles[scope][role];p['models']+=1;p['trees']+=len(trees);p['depth_histogram'].update(d for d,l in trees);p['leaf_histogram'].update(l for d,l in trees)
 for path in sorted(D.glob('P*/fold_*/root_*/constructor_audit.jsonl')):
  for r in rows(path):
   q=r['params'];counts[r['kind']]+=1;assert q['n_jobs']==1 and q['boosting_type']=='gbdt'
   if r['kind']=='LGBMRanker':assert q['max_depth']==3 and q['max_bin']==63 and q['num_leaves']==15 and q['reg_lambda']==1 and q['reg_alpha']==0 and q['min_child_samples']==20
   else:assert q['objective']=='binary' and q['reg_lambda']==2 and q['reg_alpha']==1
  file_audits.append(dict(file=str(path.relative_to(D)),sha256=sha(path)))
 for path in sorted(D.glob('P*/fold_*/root_*/native_fit_audit.jsonl')):
  for r in rows(path):
   checked=compare_native(r['constructor_params'],r['native_params'],r['kind']);assert sorted(checked)==r['checked_constructor_keys'] and digest(r['native_params'])==r['native_params_sha256']
   assert compare_h352_defaults(r['constructor_params'],r['native_params'],r['kind'])==r['h352_native_comparison']
   assert r['keyword_delta']==({'max_bin':[255,63]} if r['kind']=='LGBMRanker' else {}) and r['started_utc']>prereg['utc']
   from dataset_audit import validate_rows
   validate_rows(r['dataset_audit'],r['kind'])
   dataset_counts[r['kind']]+=len(r['dataset_audit'])
   for ds in r['dataset_audit']:dataset_caps[r['kind']+':'+str(ds['effective_max_bin'])]+=1
   native_counts[r['kind']]+=1;add('completed_fit_records','ranker' if r['kind']=='LGBMRanker' else 'veto',r['actual_trees'],r['native_params'],r['actual_tree_summary'])
  file_audits.append(dict(file=str(path.relative_to(D)),sha256=sha(path)))
 for path in D.glob('P*/fold_*/root_*/first_fit_started.json'):
  r=read(path);assert r['prereg_commit']==prereg['head'] and r['utc']>prereg['utc'];first.append(r['utc'])
 reviews=list(D.glob('P*/fold_*/root_*/native_model_review.json'))
 for path in reviews:
  r=read(path);assert r['status']=='PASS'
  for model in r['models']:
   kind='LGBMClassifier' if model['kind']=='veto_gbdt' else 'LGBMRanker';native=model['native_parameters'];ctor=model['constructor_parameters']
   compare_native(ctor,native,kind);assert compare_h352_defaults(ctor,native,kind)==model['h352_native_comparison']
   add('saved_model_reviews','veto' if kind=='LGBMClassifier' else 'ranker',model['actual_trees'],native,model['actual_tree_summary'])
 for scope in profiles.values():
  for p in scope.values():
   p['max_actual_depth']=max(p['depth_histogram'],default=None);p['max_actual_leaves']=max(p['leaf_histogram'],default=None);p['depth_histogram']={str(k):v for k,v in sorted(p['depth_histogram'].items())};p['leaf_histogram']={str(k):v for k,v in sorted(p['leaf_histogram'].items())}
 locks=list(D.glob('P*/fold_*/root_*/fit_lock.json'));markers=list(D.glob('P*/fold_*/root_*/score_once.json'));veto=list(D.glob('P*/fold_*/root_*/veto_sha_review.json'));veto_count=0
 for path in veto:
  r=read(path);assert r['status']=='PASS' and r['matched']==3 and all(x['sha256']==x['BASE_sha256'] for x in r['models']);veto_count+=r['matched']
 saved_count=sum(profiles['saved_model_reviews'][role]['models'] for role in ['ranker','veto'])
 if full:
  assert len(locks)==len(markers)==len(reviews)==len(veto)==36 and veto_count==108 and saved_count==14040
  assert sum(counts.values())==sum(native_counts.values())==36504 and dict(native_counts)=={'LGBMRanker':29484,'LGBMClassifier':7020}
  assert profiles['saved_model_reviews']['ranker']['models']==12636 and profiles['saved_model_reviews']['veto']['models']==1404 and not truncated
  assert all(read(p)['state']=='COMPLETE' for p in markers)
  for path in D.glob('P*/fold_*/root_*/guard_receipt.json'):
   r=read(path);assert r['physical_outer_rows_read_before_lock']==r['outer_test_packet_accesses_before_lock']==r['outer_test_packets_prebuilt']==r['guard_denials']==0
  import evaluate_h368 as ev
  import numpy as np
  with np.load(D/'raw_family_gains.npz') as z:
   for field,base in [('paired_drop_top10','NL15'),('paired_vs_NL15_drop_top10','NL15'),('paired_vs_BASE_drop_top10','BASE')]:
    got=ev.paired_drop_top(z['MB63'],z[base]);assert all(got[k]==v for k,v in raw[field].items() if k in got)
  gain={a:[c['family_mean_gain'] for c in raw['cells'] if c['arm']==a] for a in ['BASE','NL15','MB63']};harm={a:[c['harmed_share'] for c in raw['cells'] if c['arm']==a] for a in ['BASE','NL15','MB63']}
  gates,verdict,over,primary=ev.decide_with_secondary(raw['paired_drop_top10'],raw['paired_vs_BASE_drop_top10'],gain['MB63'],gain['NL15'],harm['MB63'],harm['NL15']);assert gates==raw['criteria'] and verdict==raw['verdict'] and over==raw['OVER_NL15'] and primary==raw['primary_verdict']
  assert len(raw['cells'])==27
  with np.load(D/'picks.npz') as z:
   for cell in raw['cells']:
    picks=z[cell['arm']+'_'+cell['partition']][[30001,30002,30003].index(cell['root_seed'])];assert hashlib.sha256(picks.tobytes()).hexdigest()==cell['picks_sha256']
  lockcount=0
  for panel,hashes in raw['fit_lock_sha256'].items():
   arm,part=panel.rsplit('_',1);root=D/('base' if arm=='BASE' else 'nl15' if arm=='NL15' else '')
   for key,h in hashes.items():
    fold,seed=key.split(':');assert sha(root/part/('fold_'+fold)/('root_'+seed)/'fit_lock.json')==h;lockcount+=1
  assert lockcount==108
 else:assert not markers or all(read(p)['state'] in ['STARTED','COMPLETE'] for p in markers)
 audit=dict(status='PASS' if full else 'PASS_FOR_AVAILABLE_RECORDS_ONLY',full_treatment_qualification='PASS' if full else 'INCOMPLETE_NOT_PASS',frozen_source_exact=True,all_completed_fit_Datasets_audited=True,Dataset_role_counts=dict(dataset_counts),Dataset_effective_max_bin_counts=dict(dataset_caps),constructor_events=sum(counts.values()),constructor_roles=dict(counts),completed_native_fit_records=sum(native_counts.values()),native_fit_roles=dict(native_counts),total_completed_optimizer_calls=sum(native_counts.values()) if full else None,total_completed_optimizer_calls_note='All mandatory completed native optimizer records verified' if full else 'Interrupted in-flight returns can be unlogged; available records are lower-bound evidence only',completed_fold_root_locks=len(locks),outer_score_markers=len(markers),native_model_reviews=len(reviews),reviewed_saved_models=saved_count,currently_raw_native_files=len(list(D.glob('P*/fold_*/root_*/**/*.txt'))),final_veto_SHA_matches=veto_count,first_actual_fit_started_utc=min(first) if first else None,all_observed_fits_after_prereg=True,truncated_interrupted_records=truncated)
 (P/'effective-model-review.json').write_text(json.dumps(dict(**audit,audit_files=file_audits),indent=2)+'\n')
 trees=dict(status='PASS' if full else 'AVAILABLE_RECORDS_ONLY',reference_saved_ranker_trees=read(P/'reference-tree-summary.json')['profiles'],MB63=profiles,configured_ranker_max_depth=3,configured_ranker_max_bin=63,configured_ranker_num_leaves=15,depth_definition='Root-to-leaf edges; ordered [depth,leaf_count] pairs at implicit Tree index',all_observed_tree_budgets_verified=True,full_per_tree_records='native_fit_audit.jsonl/native_model_review.json in exact-Version S3 custody; partial unreviewed rawmodels retained when incomplete',new_fits=0,new_predictions=0)
 (P/'actual-tree-summary.json').write_text(json.dumps(trees,indent=2)+'\n')
 samples=[json.loads(x) for x in (D/'resource.jsonl').read_text().splitlines()];windows=[r for r in samples if r.get('load_window_sample')]
 resources=dict(samples=len(samples),full_60s_windows=len(windows),max_cpu_5s_telemetry=max((r['cpu_percent'] for r in samples),default=None),max_cpu_60s=max((r['cpu_percent_60s'] for r in windows),default=None),max_load1=max((r['load1'] for r in samples),default=None),host_nproc=sorted({r['host_nproc'] for r in samples}),load_thresholds=sorted({r['load_threshold'] for r in samples}),max_consecutive_high_load_60s=max((r['load_high_consecutive_60s'] for r in samples),default=0),min_mem_available_bytes=min((r['mem_available_bytes'] for r in samples),default=None),min_root_free_bytes=min((r['root_free_bytes'] for r in samples),default=None),max_scratch_bytes=max((r['scratch_bytes'] for r in samples),default=None),nice=19,numeric_threads=1,MemoryMax=6*2**30,MemorySwapMax=0,CPUQuota='800%',guard='CPU>85 averaged60s OR load>.85*hostonlineNCPU on3consecutive60s samples OR MemAvail20GiB/root80GiB/scratch2GB OR fixeddeadline')
 for r in windows:assert r['cpu_window_seconds']>=60 and r['host_nproc']==128 and r['load_threshold']==108.8
 if full:assert resources['max_cpu_60s']<=85 and resources['max_consecutive_high_load_60s']<3 and resources['min_mem_available_bytes']>=20*2**30 and resources['min_root_free_bytes']>=80*2**30 and resources['max_scratch_bytes']<=2_000_000_000
 def pair_unknown():return dict(median=None,ci95=None,lcb95=None,per_cell=[None]*9,per_root=[dict(root_seed=s,median=None,ci95=None,lcb95=None) for s in [30001,30002,30003]],rng=368191,draws=0,planned_draws=10000)
 m=dict(raw) if full else dict(wave='H368',status=status,verdict=None,scientific_verdict='UNKNOWN',primary_verdict=None,OVER_NL15=None,secondary_label='UNKNOWN',elo_label='reference = r882 slot0 (DL-Train outer); r3175 not measured on TRAIN',r3175_train='NOT_MEASURED',aggregate={a:dict(median_elo=None,median_gain=None,literal_family_median_gain=None,median_harmed=None) for a in ['BASE','NL15','MB63']},paired_drop_top10=pair_unknown(),paired_vs_NL15_drop_top10=pair_unknown(),paired_vs_BASE_drop_top10=pair_unknown(),primary_comparator='H352 NL15',secondary_comparator='BASE',criteria={k:None for k in ['paired_drop_top10_at_least_002','gain_retention','harmed_nonincrease','positive_on_two_seeds']},cells=[dict(partition=p,root_seed=s,arm=a,elo=None,elo_ci95=None,family_mean_gain=None,family_median_gain=None,measured=False) for p in ['P0','P1','P2'] for s in [30001,30002,30003] for a in ['BASE','NL15','MB63']],dope=dict(composite_sha256_prefix='c51a56d0',loaded=False,scope='CPU selection layer only'),physical_outer_rows_read_before_lock=None)
 # Normalize a retained legacy reporting label only; raw evaluator bytes stay archived.
 if full:m['primary_positive_NL15_label']=m.pop('secondary_label')
 # Full regressions and operational detail are separate pinned receipts to keep scientific metrics<50KB.
 (S/'regression_receipt.json').write_bytes((D/'regression.json').read_bytes());(S/'nl15_regression_receipt.json').write_bytes((D/'nl15_regression.json').read_bytes())
 m['regression']=dict(status='PASS',partitions={p:dict(median=r['median'],median_exact=r['median_exact']) for p,r in reg['partitions'].items()},panels=9,fold_root_pick_lock_hashes=36,receipt='regression_receipt.json',receipt_sha256=sha(S/'regression_receipt.json'))
 m['nl15_regression']=dict(status='PASS',panels=9,fold_locks_and_scores=36,raw_family_arrays_exact=True,receipt='nl15_regression_receipt.json',receipt_sha256=sha(S/'nl15_regression_receipt.json'))
 runreceipt=dict(reporting_label_note='Frozen evaluator legacy secondary_label/OVER_NL15 describes PRIMARY NL15 positivity; publication renames only that label to primary_positive_NL15_label. Raw result unchanged in archive; primary four gates and secondary BASE metrics unchanged.',io_transport_guard_adaptation=read(P/'io-guard-adaptation.json'),io_original_attempt_exit=read(P/'relay-first-attempt-exit.json'),io_resume_inventory=read(P/'relay-resume-inventory.json'),unqualified_raw_candidate_sha256=sha(D/'metrics.json') if raw is not None and not full else None,actual_environment=read(P/'environment.json'),forecast_stop_backstop=read(P/'forecast-watch-receipt.json'),forecast_stop_host_adapter='Corrected owned h368-x1-run forecast-miss target before source freeze; no external watcher required',prereg_commit=prereg['head'],prereg_published_utc=prereg['utc'],execution=execution,preparation_execution=read(P/'prepare-unit-exit.json'),cpu_admission=read(P/'capacity.json'),runtime_forecasts=dict(initial=read(P/'runtime-forecast-at-launch.json'),after_first_completed_task=read(P/'first-cell-forecast.json')),resource_stop=stop,resources=resources,audit=audit,actual_tree_depth_and_leaves=trees,BASE_effective_native_config={k:v for k,v in read(P/'base-native-config.json').items() if k not in ['models','source_archive']},H352_effective_native_config={k:v for k,v in read(P/'nl15-native-config.json').items() if k!='models'},staged_copy_verification=read(P/'stage-verification.json'),input_identity_validation=read(P/'input_identity_validation.json'),source_equivalence=read(P/'science-equivalence.json'),reference_copy_reclamation=read(P/'reference-copy-reclamation.json'),tests=tests,CI='NOT_RUN owner CI_FREEZE')
 (S/'run_receipt.json').write_text(json.dumps(runreceipt,indent=2)+'\n');m.update(scientific_verdict=m['verdict'] if full else 'UNKNOWN',prereg_commit=prereg['head'],run_receipt='run_receipt.json',run_receipt_sha256=sha(S/'run_receipt.json'),full_treatment_qualification=audit['full_treatment_qualification'],multiplicity='Sequential H348-H368 samefold/root sweep, no multiplicity correction; anyadvance TRAIN-only and needs separately authorized dev91confirmation',CI='NOT_RUN owner CI_FREEZE')
 blob=json.dumps(m,separators=(',',':'),allow_nan=False)+'\n';assert len(blob.encode())<50000,('Scientific metrics too large',len(blob.encode()));(S/'metrics.json').write_text(blob)
 (S/'actual_tree_summary.json').write_bytes((P/'actual-tree-summary.json').read_bytes())
 (P/'terminal-report.json').write_text(json.dumps(dict(status=status,scientific_verdict=m['verdict'] if full else 'UNKNOWN',full=full,metrics_bytes=len(blob.encode()),audit=audit,resources=resources),indent=2)+'\n')
 print(json.dumps(dict(status=status,verdict=m['verdict'],metrics_bytes=len(blob.encode()),audit=audit)))
if __name__=='__main__':main()
