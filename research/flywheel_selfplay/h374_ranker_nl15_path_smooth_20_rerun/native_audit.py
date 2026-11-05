"""Validate every native fitted model against every constructor argument."""
import hashlib,json,re,math
ALIASES={'boosting_type':'boosting','n_estimators':'num_iterations','n_jobs':'num_threads','random_state':'seed','colsample_bytree':'feature_fraction','subsample':'bagging_fraction','subsample_freq':'bagging_freq','subsample_for_bin':'bin_construct_sample_cnt','reg_alpha':'lambda_l1','reg_lambda':'lambda_l2','min_child_samples':'min_data_in_leaf','min_child_weight':'min_sum_hessian_in_leaf','min_split_gain':'min_gain_to_split'}
PYTHON_ONLY={'class_weight','importance_type'}

def parse(text):return dict(re.findall(r'^\[([^:\]]+): (.*)\]$',text,re.M))
def digest(params):return hashlib.sha256(json.dumps(params,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def compare_native(expected,native,kind,check_treatment=True):
 assert native['boosting']=='gbdt' and native['device_type']=='cpu' and native['num_threads']=='1'
 assert native['objective'].split()[0]==('binary' if kind=='LGBMClassifier' else 'lambdarank')
 assert float(native['path_smooth'])==(20.0 if kind=='LGBMRanker' and check_treatment else 0.0)
 assert float(native['min_sum_hessian_in_leaf'])==.001
 target=0.0 if kind=='LGBMRanker' else 1.0
 assert float(native['lambda_l1'])==target and float(native['sigmoid'])==1.0
 if kind=='LGBMRanker':assert native['metric']=='ndcg' and float(native['lambda_l2'])==1.0 and int(native['num_leaves'])==(15 if check_treatment else 7) and int(native['max_depth'])==3 and int(native['min_data_in_leaf'])==20 and float(native['min_gain_to_split'])==0
 if kind=='LGBMRanker' and check_treatment:assert float(native['learning_rate']) in [.03,.05,.1] and int(native['num_iterations']) in [50,100,200]
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
 if kind=='LGBMRanker':assert depth==3 and max(counts)<=8
 return dict(trees=len(counts),minimum=min(counts),maximum=max(counts),histogram=dict(Counter(counts)),configured_num_leaves=configured,max_depth=depth)

_NL15_HEADERS=None
def compare_NL15_header(native,kind):
 global _NL15_HEADERS
 if _NL15_HEADERS is None:
  from pathlib import Path
  _NL15_HEADERS=json.loads((Path(__file__).parent/'nl15-native-headers.json').read_text())
 key=kind+':'+native['seed'];expected=dict(_NL15_HEADERS[key])
 # Learning rate / rounds are unchanged H352 grid choices, which can be selected
 # differently by nested CV. Reconstruct the same-grid H352 header, then allow
 # only the owner-authorized path_smooth delta. All other native fields exact.
 if kind=='LGBMRanker':
  assert float(native['learning_rate']) in [.03,.05,.1] and int(native['num_iterations']) in [50,100,200]
  expected.update(learning_rate=native['learning_rate'],num_iterations=native['num_iterations'])
 delta={k:[expected.get(k),native.get(k)] for k in set(expected)|set(native) if expected.get(k)!=native.get(k)}
 assert delta==({'path_smooth':['0','20']} if kind=='LGBMRanker' else {}),delta
 return delta
