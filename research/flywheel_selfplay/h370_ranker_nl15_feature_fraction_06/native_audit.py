"""Validate every native fitted model against every constructor argument."""
import hashlib,json,re,math
ALIASES={'boosting_type':'boosting','n_estimators':'num_iterations','n_jobs':'num_threads','random_state':'seed','colsample_bytree':'feature_fraction','subsample':'bagging_fraction','subsample_freq':'bagging_freq','subsample_for_bin':'bin_construct_sample_cnt','reg_alpha':'lambda_l1','reg_lambda':'lambda_l2','min_child_samples':'min_data_in_leaf','min_child_weight':'min_sum_hessian_in_leaf','min_split_gain':'min_gain_to_split'}
PYTHON_ONLY={'class_weight','importance_type'}

def parse(text):return dict(re.findall(r'^\[([^:\]]+): (.*)\]$',text,re.M))
def digest(params):return hashlib.sha256(json.dumps(params,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def compare_native(expected,native,kind,check_treatment=True):
 assert native['boosting']=='gbdt' and native['device_type']=='cpu' and native['num_threads']=='1'
 assert native['objective'].split()[0]==('binary' if kind=='LGBMClassifier' else 'lambdarank')
 assert float(native['path_smooth'])==0.0
 assert float(native['min_sum_hessian_in_leaf'])==.001
 target=0.0 if kind=='LGBMRanker' else 1.0
 assert float(native['lambda_l1'])==target and float(native['sigmoid'])==1.0
 if kind=='LGBMRanker':assert native['metric']=='ndcg' and float(native['lambda_l2'])==1.0 and int(native['num_leaves'])==15 and int(native['max_depth'])==3 and int(native['min_data_in_leaf'])==20 and float(native['min_gain_to_split'])==0
 assert native['label_gain']=='' and native['bagging_by_query']=='0' and float(native['max_delta_step'])==0
 if kind=='LGBMClassifier':
  assert native['extra_trees']=='0'
  veto_native_delta(expected,native)
 if kind=='LGBMRanker':
  assert native['extra_trees']=='0' and float(native['feature_fraction'])==(.6 if check_treatment else .8) and float(native['bagging_fraction'])==.8 and int(native['bagging_freq'])==1
  native_delta(expected,native,check_treatment)
 checked={}
 for k,value in expected.items():
  if k in PYTHON_ONLY:
   assert value==({'class_weight':None,'importance_type':'split'}[k]);continue
  name=ALIASES.get(k,k);assert name in native,(k,name)
  got=native[name]
  if isinstance(value,bool):assert got==str(int(value)),(k,value,got)
  elif isinstance(value,int):assert int(got)==value,(k,value,got)
  elif isinstance(value,float):assert math.isclose(float(got),float(value),rel_tol=1e-7,abs_tol=1e-12),(k,value,got)
  else:assert str(value)==got,(k,value,got)
  checked[k]=dict(native_key=name,expected=value,native=got)
 return checked


def observed_leaves(text,kind):
 from collections import Counter
 counts=list(map(int,re.findall(r'^num_leaves=(\d+)$',text,re.M)));assert len(counts)==len(re.findall(r'^Tree=\d+$',text,re.M));assert counts and min(counts)>=1
 params=parse(text);configured=int(params['num_leaves']);depth=int(params['max_depth'])
 assert max(counts)<=configured
 if kind=='LGBMRanker':assert depth==3 and max(counts)<=8
 return dict(trees=len(counts),minimum=min(counts),maximum=max(counts),histogram=dict(Counter(counts)),configured_num_leaves=configured,max_depth=depth)


def native_delta(expected,native,treatment=True,baseline_leaves=15):
 """All native fields vs archived BASE, resolving only original variable inputs/seeds."""
 from pathlib import Path
 ref=json.loads((Path(__file__).parent/'native_baseline_reference.json').read_text())
 base=dict(ref['native_params']);base['num_leaves']=str(baseline_leaves)
 for key,value in [('seed',expected['random_state']),('learning_rate',expected['learning_rate']),('num_iterations',expected['n_estimators'])]:base[key]=str(value)
 x=int(expected['random_state'])
 for key in ['data_random_seed','bagging_seed','drop_seed','feature_fraction_seed','objective_seed','extra_seed']:
  x=(214013*x+2531011)&0xffffffff;base[key]=str((x>>16)&0x7fff)
 assert set(base)==set(native),(set(base)-set(native),set(native)-set(base))
 delta={k:[base[k],native[k]] for k in base if base[k]!=native[k]}
 want={'feature_fraction':['0.8','0.6']} if treatment else {}
 assert delta==want,delta
 return dict(status='PASS',delta=delta,baseline_native_sha256=digest(base),derived_seed_algorithm='LightGBM Random uint32 LCG214013+2531011, six original seed draws',diagnostic_note='Only feature_fraction differs; all seed draws, LR/tree options and other native fields are pinned.')

def veto_native_delta(expected,native):
 """All native veto fields remain identical after resolving original seed inputs."""
 from pathlib import Path
 ref=json.loads((Path(__file__).parent/'base_veto_native_reference.json').read_text())
 base=dict(ref['native_params'])
 for key,value in [('seed',expected['random_state']),('learning_rate',expected['learning_rate']),('num_iterations',expected['n_estimators'])]:base[key]=str(value)
 x=int(expected['random_state'])
 for key in ['data_random_seed','bagging_seed','drop_seed','feature_fraction_seed','objective_seed','extra_seed']:
  x=(214013*x+2531011)&0xffffffff;base[key]=str((x>>16)&0x7fff)
 assert set(base)==set(native)
 delta={k:[base[k],native[k]] for k in base if base[k]!=native[k]}
 assert delta=={},delta
 return dict(status='PASS',delta={},baseline_native_sha256=digest(base))
