"""Sole H352 NL15 ranker delta: subsample_freq 1 to 5 (native bagging_freq)."""
import ast,copy,hashlib,inspect
import leaves_patch
VALUE=5
def strip(tree):
 tree=copy.deepcopy(tree)
 for call in leaves_patch.calls(tree):
  kw=[k for k in call.keywords if k.arg=='subsample_freq'];assert len(kw)==1 and ast.literal_eval(kw[0].value)==VALUE
  kw[0].value=ast.Constant(1)
 return ast.fix_missing_locations(tree)
def transform(source):
 _,baseline=leaves_patch.transform(source);new=copy.deepcopy(baseline)
 calls=leaves_patch.calls(new);assert len(calls)==1
 kw=[k for k in calls[0].keywords if k.arg=='subsample_freq'];assert len(kw)==1 and ast.literal_eval(kw[0].value)==1
 assert not any(k.arg in ('bagging_freq','path_smooth','seed','bagging_seed') for k in calls[0].keywords)
 kw[0].value=ast.Constant(VALUE);ast.fix_missing_locations(new)
 assert ast.dump(strip(new),include_attributes=False)==ast.dump(baseline,include_attributes=False)
 return baseline,new
def patched_function(fun):
 source=inspect.getsource(fun);baseline,new=transform(source);ns=dict(fun.__globals__);exec(compile(new,fun.__code__.co_filename,'exec'),ns)
 return ns[fun.__name__],dict(function=fun.__name__,baseline_source_sha256=hashlib.sha256(source.encode()).hexdigest(),H352_ast_sha256=hashlib.sha256(ast.dump(baseline,include_attributes=False).encode()).hexdigest(),restoration_exact=True,delta={'subsample_freq':[1,VALUE]},native_delta={'bagging_freq':[1,VALUE]},bagging_fraction=.8,path_smooth=0.0,num_leaves=15,max_depth=3,learning_rate_and_rounds_unchanged=True)
def install(G):
 records=[]
 for mod,name in [('topk','fit_ranker'),('bagged','fit_ranker_n')]:
  fun,record=patched_function(getattr(G[mod],name));setattr(G[mod],name,fun);records.append(record)
 return dict(status='PASS',functions=records,bagging_freq=VALUE,bagging_fraction=.8,path_smooth=0.0,num_leaves=15,max_depth=3,veto_unchanged=True,no_new_seed_keyword=True,learning_rate_and_rounds_unchanged=True)
