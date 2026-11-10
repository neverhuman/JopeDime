"""Read-only native BASE configuration and final-veto SHA inventory; zero fits."""
from pathlib import Path
import datetime,hashlib,json,re
HERE=Path(__file__).resolve().parent;E=HERE.parent;D=Path('/home/ubuntu/jope-data/h353-x1')
def main():
 assert json.loads((D/'regression.json').read_text())['status']=='PASS'
 fetch=json.loads((E/'base-fetch-receipt.json').read_text());assert fetch['status']=='EXACT_VERSION_GET_AND_SHA_PASS'
 rows=[];rankers=[];vetos=[]
 for name,rec in sorted(fetch['h316']['members'].items()):
  if not re.fullmatch(r'run/P[012]/fold_[0-3]/root_3000[123]/refit/[^/]+\.txt',name):continue
  path=D/'base_native'/name[4:];raw=path.read_bytes();assert hashlib.sha256(raw).hexdigest()==rec['sha256'] and len(raw)==rec['bytes']
  params=dict(re.findall(r'^\[([^:\]]+): (.*)\]$',raw.decode(),re.M));veto=name.endswith('-veto.txt')
  assert params['boosting']=='gbdt' and params['device_type']=='cpu' and params['num_threads']=='1'
  assert params['objective'].split()[0]==('binary' if veto else 'lambdarank')
  assert float(params['min_sum_hessian_in_leaf'])==.001 and float(params['sigmoid'])==1
  if veto:assert float(params['lambda_l1'])==1 and float(params['lambda_l2'])==2
  if not veto:assert params['metric']=='ndcg' and params['label_gain']=='' and params['min_data_in_leaf']=='20'
  from tree_structure import parse_trees,validate,summarize
  actual=validate(parse_trees(raw.decode()),int(params['max_depth']),int(params['num_leaves']))
  r=dict(max_depth=int(params['max_depth']),num_leaves=int(params['num_leaves']),actual_trees=actual,actual_tree_summary=summarize(actual),archive_member=name,sha256=rec['sha256'],role='veto' if veto else 'ranker',lambda_l1=float(params['lambda_l1']),lambda_l2=float(params['lambda_l2']),min_sum_hessian_in_leaf=float(params['min_sum_hessian_in_leaf']),sigmoid=float(params['sigmoid']))
  rows.append(r);(vetos if veto else rankers).append(r)
 assert len(rankers)==972 and len(vetos)==108
 verify_veto_reference(vetos[0])
 alpha=sorted(set(r['lambda_l1'] for r in rankers));l2=sorted(set(r['lambda_l2'] for r in rankers));depth=sorted(set(r['max_depth'] for r in rankers));leaves=sorted(set(r['num_leaves'] for r in rankers));status='PASS' if alpha==[0.0] and l2==[1.0] and depth==[3] and leaves==[7] else 'INCOMPLETE_SPEC_MISMATCH'
 expected=json.loads((HERE/'base_veto_sha256.json').read_text())['members'];assert len(expected)==108 and all(r['sha256']==expected[r['archive_member']]['sha256'] for r in vetos)
 result=dict(status=status,wave='H353',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),BASE_effective_lambda_l1=alpha,BASE_effective_lambda_l2=l2,BASE_effective_max_depth=depth,BASE_effective_num_leaves=leaves,BASE_actual_ranker_trees=summarize([t for r in rankers for t in r['actual_trees']]),BASE_effective_min_sum_hessian_in_leaf=.001,BASE_sigmoid=1.0,treatment_max_depth=4,rankers_verified=len(rankers),veto_SHA_identities_verified=len(vetos),models=rows,model_inventory_sha256=hashlib.sha256(json.dumps(rows,sort_keys=True,separators=(',',':')).encode()).hexdigest(),source_archive={k:v for k,v in fetch['h316'].items() if k!='members'},optimizer_fits=0,BASE_refits=0,model_predictions=0)
 (E/'base-native-config.json').write_text(json.dumps(result,indent=2)+'\n');(D/'base-native-config.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps({k:v for k,v in result.items() if k not in ['models','source_archive']}),flush=True)
 if status!='PASS':raise SystemExit(3)
def verify_veto_reference(row):
 import lightgbm as lgb
 import subprocess,ast
 name=row['archive_member'];raw=(D/'base_native'/name[4:]).read_text();native=dict(re.findall(r'^\[([^:\]]+): (.*)\]$',raw,re.M))
 source=subprocess.check_output(['git','-C','/home/ubuntu/JopeDime','show','213b30ad52e1ce48927bba960ed609bd98ab4dfe:research/flywheel_selfplay/h287_bagswitch_seed_ensemble_export/veto.py'],text=True)
 fn=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='fit_gate')
 call=next(n for n in ast.walk(fn) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='LGBMClassifier')
 kwargs={k.arg:eval(compile(ast.Expression(k.value),'<pinned-veto-kwargs>','eval'),{'int':int,'random_state':int(native['seed'])}) for k in call.keywords}
 params=lgb.LGBMClassifier(**kwargs).get_params()
 ref=json.loads((HERE/'base_veto_native_reference.json').read_text());assert ref['constructor_params']==params and ref['native_params']==native and ref['model_sha256']==row['sha256']
if __name__=='__main__':main()
