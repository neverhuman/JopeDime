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
 if kind=='LGBMRanker':assert native['metric']=='ndcg' and float(native['lambda_l2'])==1.0 and int(native['num_leaves'])==(15 if check_treatment else 7) and int(native['max_depth'])==(4 if check_treatment else 3) and int(native['min_data_in_leaf'])==20 and float(native['min_gain_to_split'])==0
 assert native['label_gain']=='' and native['extra_trees']=='0' and float(native['max_delta_step'])==0
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
 if kind=='LGBMRanker':assert depth in (3,4) and max(counts)<=2**depth
 return dict(trees=len(counts),minimum=min(counts),maximum=max(counts),histogram=dict(Counter(counts)),configured_num_leaves=configured,max_depth=depth)

_H352_REFERENCE=None
def compare_h352_defaults(expected,native,kind):
 global _H352_REFERENCE
 if _H352_REFERENCE is None:
  from pathlib import Path
  _H352_REFERENCE=json.loads((Path(__file__).parent/'h352_native_reference.json').read_text())
 ref=_H352_REFERENCE['roles'][kind];constants=ref['constant_native_fields'];seed=str(expected['random_state'])
 assert set(native)==set(constants)|set(ref['variable_native_fields'])|set(ref['ignored_dynamic_native_fields']), 'Native field inventory drift from H352'
 for key,want in constants.items():
  target='4' if key=='max_depth' and kind=='LGBMRanker' else want
  assert native[key]==target,(kind,key,native[key],target,'H352 constant native field drift')
 assert seed in ref['seed_fields'],('Seed absent from pinned H352 trace',seed)
 for key,values in ref['seed_fields'][seed].items():assert native[key] in values,(kind,seed,key,native[key],values)
 return dict(reference='H352 actual native full trace',constant_fields=len(constants),seed_derived_fields=len(ref['variable_native_fields']),only_intervention={'max_depth':[3,4]} if kind=='LGBMRanker' else {},num_iterations='Checked against actual constructor; frozen selected-round rule unchanged')
