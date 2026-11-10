"""Prospective requested keyword only; never install without native-zero proof."""
import ast,copy,hashlib,inspect,textwrap
LEAVES=15
def calls(t):return [x for x in ast.walk(t) if isinstance(x,ast.Call) and isinstance(x.func,ast.Attribute) and x.func.attr=='LGBMRanker']
def strip(t):
 t=copy.deepcopy(t)
 for c in calls(t):c.keywords=[k for k in c.keywords if k.arg!='feature_fraction_bynode']
 return t
def transform(source):
 from nl15_baseline_patch import transform as baseline
 _,old=baseline(source);new=copy.deepcopy(old);cc=calls(new);assert len(cc)==1;kw={k.arg:k for k in cc[0].keywords}
 assert not set(kw)&{'feature_fraction_bynode','min_split_gain','min_gain_to_split','bagging_seed','feature_fraction_seed','bagging_by_query'}
 for name,value in [('num_leaves',15),('max_depth',3),('subsample',.8),('subsample_freq',1),('colsample_bytree',.8)]:assert ast.literal_eval(kw[name].value)==value
 cc[0].keywords.append(ast.keyword(arg='feature_fraction_bynode',value=ast.Constant(True)));ast.fix_missing_locations(new)
 assert ast.dump(old,include_attributes=False)==ast.dump(strip(new),include_attributes=False)
 return old,new
def patched_function(fun):
 source=inspect.getsource(fun);old,new=transform(source);ns=dict(fun.__globals__);exec(compile(new,fun.__code__.co_filename,'exec'),ns)
 return ns[fun.__name__],dict(function=fun.__name__,baseline_source_sha256=hashlib.sha256(source.encode()).hexdigest(),restoration_exact=True,requested_delta={'feature_fraction_bynode':[0,1]},added_keywords=['feature_fraction_bynode'])
def install(G):
 from spec_guard import require_spec_pass
 from pathlib import Path
 require_spec_pass(Path(__file__).resolve().parent.parent/'SPEC-CHECK.json')
 records=[]
 for mod,name in [('topk','fit_ranker'),('bagged','fit_ranker_n')]:
  fun,rec=patched_function(getattr(G[mod],name));setattr(G[mod],name,fun);records.append(rec)
 return dict(status='PASS',functions=records,feature_fraction_bynode=True,min_split_gain=0,veto_unchanged=True)
