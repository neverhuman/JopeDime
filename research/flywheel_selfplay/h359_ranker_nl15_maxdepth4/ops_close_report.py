"""Terminal read-only H359 review and truthful incomplete scientific report."""
from pathlib import Path
from collections import Counter
import json,hashlib,datetime,subprocess,sys,os,re,platform
P=Path(__file__).parent;S=P/'source';D=Path('/home/ubuntu/jope-data/h359-hwv');sys.path.insert(0,str(S))
from native_audit import parse,compare_native,digest,compare_h352_defaults
from tree_structure import parse_trees,validate,summarize
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text())
assert subprocess.run(['systemctl','--user','is-active','h359-run'],capture_output=True,text=True).stdout.strip() not in ('active','activating','deactivating')
execution=read(P/'unit-exit.json');stop=read(D/'STOP.json');assert execution['EXIT_CODE']=='killed' and execution['EXIT_STATUS']=='TERM' and stop['reason']==['host load >85']
assert not (D/'metrics.json').exists() and not list(D.glob('P*/fold_*/root_*/score_once.json'))
for n,h in read(P/'runtime-source-manifest.json').items():assert sha(S/n)==h,n
prereg=read(P/'prereg-published.json');tests=read(P/'tests-receipt.json');assert tests['tests_executed']==65 and tests['exit_status']==tests['failed']==tests['skipped']==0
reg=read(D/'regression.json');nlreg=read(D/'nl15_regression.json');assert reg['status']==nlreg['status']=='PASS'
counts=Counter();fits=Counter();first=[];records={};truncated=[];constructorfiles=[]
profiles={scope:{role:dict(models=0,trees=0,depth_histogram=Counter(),leaf_histogram=Counter()) for role in ['ranker','veto']} for scope in ['completed_fit_records','available_saved_models']}
def add(scope,role,trees,native):
 validate(trees,int(native['max_depth']),int(native['num_leaves']))
 if role=='ranker':assert int(native['max_depth'])==4 and int(native['num_leaves'])==15
 p=profiles[scope][role];p['models']+=1;p['trees']+=len(trees);p['depth_histogram'].update(x[0] for x in trees);p['leaf_histogram'].update(x[1] for x in trees)
def lines(path):
 data=path.read_bytes();out=[]
 for i,line in enumerate(data.splitlines()):
  try:out.append(json.loads(line))
  except json.JSONDecodeError:
   assert i==len(data.splitlines())-1,'Only a terminal interrupted record may be partial';truncated.append(dict(file=str(path.relative_to(D)),line=i+1,sha256=sha(path),note='Interrupted final JSON record retained in archive; unavailable coverage, never passed'))
 return out
for path in sorted(D.glob('P*/fold_*/root_*/constructor_audit.jsonl')):
 for r in lines(path):
  q=r['params'];counts[r['kind']]+=1;assert q['n_jobs']==1 and q['boosting_type']=='gbdt'
  if r['kind']=='LGBMRanker':assert q['objective']=='lambdarank' and q['max_depth']==4 and q['num_leaves']==15 and q['reg_lambda']==1 and q['reg_alpha']==0 and q['min_child_samples']==20 and q['subsample']==.8 and q['subsample_freq']==1 and q['colsample_bytree']==.8
  else:assert q['objective']=='binary' and q['reg_alpha']==1 and q['reg_lambda']==2
 constructorfiles.append(dict(file=str(path.relative_to(D)),sha256=sha(path)))
for path in sorted(D.glob('P*/fold_*/root_*/native_fit_audit.jsonl')):
 rows=lines(path);records[path.parent]=rows
 for r in rows:
  checked=compare_native(r['constructor_params'],r['native_params'],r['kind']);assert sorted(checked)==r['checked_constructor_keys'] and digest(r['native_params'])==r['native_params_sha256']
  assert compare_h352_defaults(r['constructor_params'],r['native_params'],r['kind'])==r['h352_native_comparison']
  assert r['keyword_delta']==({'max_depth':[3,4]} if r['kind']=='LGBMRanker' else {})
  assert r['started_utc']>prereg['utc'];assert summarize(r['actual_trees'])==r['actual_tree_summary'];fits[r['kind']]+=1
  add('completed_fit_records','ranker' if r['kind']=='LGBMRanker' else 'veto',r['actual_trees'],r['native_params'])
for path in sorted(D.glob('P*/fold_*/root_*/first_fit_started.json')):
 r=read(path);assert r['prereg_commit']==prereg['head'] and r['utc']>prereg['utc'];first.append(r['utc'])
saved=[];unreviewed=[]
for path in sorted(D.glob('P*/fold_*/root_*/**/*.txt')):
 rel=str(path.relative_to(D));row=dict(file=rel,bytes=path.stat().st_size,sha256=sha(path));td=next(x for x in path.parents if x in records)
 try:
  raw=path.read_text();native=parse(raw);kind='LGBMClassifier' if path.name.endswith('-veto.txt') else 'LGBMRanker';member=int(re.search(r'-s(\d+)-',path.name).group(1));key=digest(native)
  choices=[a for a in records[td] if a['kind']==kind and a['member_seed']==member and a['native_params_sha256']==key];assert choices,'No completed fit record for this saved model'
  compare_native(choices[0]['constructor_params'],native,kind);check=compare_h352_defaults(choices[0]['constructor_params'],native,kind);trees=validate(parse_trees(raw),int(native['max_depth']),int(native['num_leaves']));add('available_saved_models','ranker' if kind=='LGBMRanker' else 'veto',trees,native)
  row.update(status='PASS_FOR_AVAILABLE_MODEL',actual_trees=trees,actual_tree_summary=summarize(trees),native_params=native,matching_native_parameter_sha256=key,h352_native_comparison=check);saved.append(row)
 except (AssertionError,KeyError,ValueError,AttributeError) as e:
  row.update(status='UNVERIFIED_INTERRUPTED_MODEL',reason=str(e));unreviewed.append(row)
(P/'partial-saved-native-audit.json').write_text(json.dumps(dict(status='AVAILABLE_RECORDS_ONLY',verified=len(saved),unverified=len(unreviewed),files=saved+unreviewed),indent=1)+'\n')
for group in profiles.values():
 for p in group.values():
  p['max_actual_depth']=max(p['depth_histogram'],default=None);p['max_actual_leaves']=max(p['leaf_histogram'],default=None);p['depth_histogram']={str(k):v for k,v in sorted(p['depth_histogram'].items())};p['leaf_histogram']={str(k):v for k,v in sorted(p['leaf_histogram'].items())}
reference_trees={}
for name,root in [('BASE',D/'base_native'),('NL15',D/'nl15')]:
 ds=Counter();ls=Counter();models=0
 for path in sorted(root.glob('P*/fold_*/root_*/refit/*.txt')):
  if path.name.endswith('-veto.txt'):continue
  native=parse(path.read_text());assert int(native['max_depth'])==3 and int(native['num_leaves'])==(7 if name=='BASE' else 15)
  trees=validate(parse_trees(path.read_text()),3,int(native['num_leaves']));ds.update(x[0] for x in trees);ls.update(x[1] for x in trees);models+=1
 assert models==972
 reference_trees[name]=dict(models=models,trees=sum(ds.values()),max_actual_depth=max(ds),max_actual_leaves=max(ls),depth_histogram={str(k):v for k,v in sorted(ds.items())},leaf_histogram={str(k):v for k,v in sorted(ls.items())})
tree_summary=dict(status='AVAILABLE_RECORDS_ONLY_NOT_FULL_TREATMENT_QUALIFICATION',configured_ranker_max_depth=4,configured_ranker_num_leaves=15,depth_definition='Root-to-leaf edges, root-onlyleaf0; ordered actual_trees are [depth,leaf_count] at implicit Tree index',reference_saved_ranker_trees=reference_trees,NL15MD4=profiles,all_observed_tree_budgets_verified=True,unverified_interrupted_saved_models=len(unreviewed),full_per_tree_records='Completed native_fit_audit.jsonl and custody/partial-saved-native-audit.json; all archived, no models Git',new_fits=0,new_predictions=0)
(P/'actual-tree-summary.json').write_text(json.dumps(tree_summary,indent=2)+'\n')
locks=list(D.glob('P*/fold_*/root_*/fit_lock.json'));assert not locks
markers=list(D.glob('P*/fold_*/root_*/score_once.json'));assert not markers
samples=[json.loads(x) for x in (D/'resource.jsonl').read_text().splitlines()];resources=dict(samples=len(samples),max_cpu_percent=max(x['cpu_percent'] for x in samples),max_load1=max(x['load1'] for x in samples),min_mem_available_bytes=min(x['mem_available_bytes'] for x in samples),max_scratch_bytes=max(x['scratch_bytes'] for x in samples),nice=19,numeric_threads=1,MemoryMax=6*2**30,MemorySwapMax=0,CPUQuota='800%')
audit=dict(status='PASS_FOR_AVAILABLE_RECORDS_ONLY',full_treatment_qualification='INCOMPLETE_NOT_PASS',frozen_source_exact=True,constructor_events=sum(counts.values()),constructor_roles=dict(counts),completed_native_fit_records=sum(fits.values()),native_fit_roles=dict(fits),total_completed_optimizer_calls=None,total_completed_optimizer_calls_note='Observed records prove completed calls; interrupted in-flight returns may be unlogged. No complete qualification inferred.',completed_fold_root_locks=0,outer_score_markers=0,native_model_reviews=0,reviewed_saved_models=len(saved),unverified_interrupted_saved_models=len(unreviewed),currently_raw_native_files=len(saved)+len(unreviewed),final_veto_SHA_matches=0,first_actual_fit_started_utc=min(first),all_observed_fits_after_prereg=True,truncated_interrupted_records=truncated,constructor_file_sha256=constructorfiles)
(P/'effective-model-review.json').write_text(json.dumps(audit,indent=2)+'\n')
import lightgbm,numpy,sklearn
lib=Path(lightgbm.__file__).parent/'lib/lib_lightgbm.so';env=dict(host=os.uname().nodename,python=platform.python_version(),numpy=numpy.__version__,sklearn=sklearn.__version__,lightgbm=lightgbm.__version__,kernel=os.uname().release,lightgbm_library_sha256=sha(lib),GPU_used=False)
(P/'environment.json').write_text(json.dumps(env,indent=2)+'\n')
def unknown_pair():return dict(median=None,ci95=None,lcb95=None,per_cell=[None]*9,per_root=[dict(root_seed=s,median=None,ci95=None,lcb95=None) for s in [30001,30002,30003]],rng=359191,draws=0,planned_draws=10000)
base=read(P/'base-native-config.json');nl=read(P/'nl15-native-config.json')
m=dict(wave='H359',status='INCOMPLETE_RESOURCE_GUARD',verdict=None,scientific_verdict='UNKNOWN',primary_verdict=None,OVER_NL15=None,secondary_label='UNKNOWN',elo_label='reference = r882 slot0 (DL-Train outer); r3175 not measured on TRAIN',r3175_train='NOT_MEASURED',aggregate={a:dict(median_elo=None,median_gain=None,literal_family_median_gain=None,median_harmed=None) for a in ['BASE','NL15','NL15MD4']},paired_drop_top10=unknown_pair(),paired_vs_NL15_drop_top10=unknown_pair(),criteria={k:None for k in ['paired_drop_top10_at_least_002','gain_retention','harmed_nonincrease','positive_on_two_seeds']},cells=[dict(partition=p,root_seed=s,arm=a,elo=None,elo_ci95=None,family_mean_gain=None,family_median_gain=None,measured=False) for p in ['P0','P1','P2'] for s in [30001,30002,30003] for a in ['BASE','NL15','NL15MD4']],config_equal_count=None,config_equal_vs_NL15_count=None,config_sensitivity='UNKNOWN',regression=reg,nl15_regression=nlreg,dope=dict(composite_sha256_prefix='c51a56d0',loaded=False,scope='CPU selection layer only'),prereg_commit=prereg['head'],prereg_published_utc=prereg['utc'],execution=execution,preparation_execution=read(P/'experiment-prepare-unit-exit.json'),input_preparation_execution=read(P/'prepare-unit-exit.json'),cpu_admission=read(P/'capacity.json'),runtime_forecasts=dict(initial=read(P/'runtime-forecast-at-launch.json'),after_first_completed_task=None,first_completed_task_status='NOT_RUN_NO_COMPLETED_TASK'),resource_stop=stop,resources=resources,input_identity_validation=read(P/'input_identity_validation.json'),BASE_effective_native_config={k:v for k,v in base.items() if k not in ['models','source_archive']},H352_effective_native_config={k:v for k,v in nl.items() if k!='models'},audit={k:v for k,v in audit.items() if k!='constructor_file_sha256'},treatment=dict(arm='NL15MD4',versus='H352 NL15',existing_max_depth=[3,4],num_leaves=15,reg_lambda=1,reg_alpha='BASE implicit0',min_child_samples=20,added_keywords=[],restoring_depth3_restores_exact_H352_AST=True),source_code='CommittedH35202dfee2f7e0b8543324494dd4f99d074cd6b93b6',actual_environment=env,actual_tree_depth_and_leaves=tree_summary,scientific_AST_equivalence=read(P/'science-equivalence.json')['status'],outer_score_phase='NOT_ENTERED',physical_outer_rows_read_before_lock=None,guard_qualification='INCOMPLETE_NOT_PASS',DLVal_feature_rows_read=0,dev91_rows_read=0,ci='NOT_RUN owner CI_FREEZE',multiplicity='One of sequential H348-H359 sweep on same folds/roots; no multiplicity correction; no fresh confirmation.')
blob=json.dumps(m,separators=(',',':'),allow_nan=False)+'\n';assert len(blob.encode())<50000;(S/'metrics.json').write_text(blob)
for source,target in [(P/'actual-tree-summary.json','actual_tree_summary.json'),(P/'environment.json','environment_receipt.json'),(D/'regression.json','regression_receipt.json'),(D/'nl15_regression.json','nl15_regression_receipt.json'),(P/'input_identity_validation.json','input_identity_receipt.json')]:
 (S/target).write_bytes(source.read_bytes())
# A fresh terminal inventory is custody verification, never a scientific qualification.
files={str(p.relative_to(D)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in D.rglob('*') if p.is_file()};assert sum(r['bytes'] for r in files.values())<=2_000_000_000
(P/'local-run-verification.json').write_text(json.dumps(dict(status='PASS',scope='Stopped own unit, immutable-source/terminal-file inventory; scientific outcome INCOMPLETE',files=files,execution=execution),indent=1)+'\n')
print(json.dumps(dict(status=m['status'],scientific_verdict='UNKNOWN',constructors=sum(counts.values()),completed_native_records=sum(fits.values()),available_saved_models=len(saved),unverified_models=len(unreviewed),locks=0,scores=0,metrics_bytes=len(blob.encode()),scratch_files=len(files),scratch_bytes=sum(r['bytes'] for r in files.values()))),flush=True)
