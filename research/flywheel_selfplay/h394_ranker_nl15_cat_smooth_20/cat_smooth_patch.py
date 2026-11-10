"""Sole H352 NL15 ranker delta: categorical L2 doubles its archived default."""
import ast,copy,hashlib,inspect,json
from pathlib import Path
import leaves_patch
CONFIG=json.loads((Path(__file__).parent/'cat_smooth_value.json').read_text())
BASELINE=float(CONFIG['archived_NL15_cat_smooth']);VALUE=float(CONFIG['cat_smooth'])
assert VALUE==BASELINE*2 and BASELINE==10

def strip(tree):
 tree=copy.deepcopy(tree)
 for call in leaves_patch.calls(tree):
  kw=[k for k in call.keywords if k.arg=='cat_smooth'];assert len(kw)==1 and ast.literal_eval(kw[0].value)==VALUE
  call.keywords=[k for k in call.keywords if k.arg!='cat_smooth']
 return ast.fix_missing_locations(tree)
def transform(source):
 _,baseline=leaves_patch.transform(source);new=copy.deepcopy(baseline);calls=leaves_patch.calls(new);assert len(calls)==1
 kw={k.arg:k.value for k in calls[0].keywords};assert ast.literal_eval(kw['subsample_freq'])==1 and ast.literal_eval(kw['subsample'])==.8
 assert not set(kw)&{'cat_smooth','cat_l2','path_smooth','bagging_freq','seed','bagging_seed'}
 calls[0].keywords.append(ast.keyword(arg='cat_smooth',value=ast.Constant(VALUE)));ast.fix_missing_locations(new)
 assert ast.dump(strip(new),include_attributes=False)==ast.dump(baseline,include_attributes=False)
 return baseline,new
def patched_function(fun):
 source=inspect.getsource(fun);baseline,new=transform(source);ns=dict(fun.__globals__);exec(compile(new,fun.__code__.co_filename,'exec'),ns)
 return ns[fun.__name__],dict(function=fun.__name__,baseline_source_sha256=hashlib.sha256(source.encode()).hexdigest(),H352_ast_sha256=hashlib.sha256(ast.dump(baseline,include_attributes=False).encode()).hexdigest(),restoration_exact=True,delta={'cat_smooth':[BASELINE,VALUE]},bagging_freq=1,bagging_fraction=.8,path_smooth=0.,num_leaves=15,max_depth=3,learning_rate_and_rounds_unchanged=True)
def install(G):
 records=[]
 for mod,name in [('topk','fit_ranker'),('bagged','fit_ranker_n')]:
  fun,record=patched_function(getattr(G[mod],name));setattr(G[mod],name,fun);records.append(record)
 return dict(status='PASS',functions=records,cat_smooth=VALUE,archived_cat_smooth=BASELINE,bagging_freq=1,bagging_fraction=.8,path_smooth=0.,num_leaves=15,max_depth=3,veto_unchanged=True,no_new_seed_keyword=True,learning_rate_and_rounds_unchanged=True)
