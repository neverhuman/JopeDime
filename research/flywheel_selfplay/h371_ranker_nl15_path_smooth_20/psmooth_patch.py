"""Sole H352 NL15 ranker delta: path_smooth 0 to 20; full AST inverse."""
import ast,copy,hashlib,inspect
import leaves_patch
VALUE=20.0
def strip(tree):
 tree=copy.deepcopy(tree)
 for call in leaves_patch.calls(tree):
  assert sum(k.arg=='path_smooth' for k in call.keywords)==1
  call.keywords=[k for k in call.keywords if k.arg!='path_smooth']
 return tree
def transform(source):
 _,baseline=leaves_patch.transform(source);new=copy.deepcopy(baseline)
 calls=leaves_patch.calls(new);assert len(calls)==1 and not any(k.arg=='path_smooth' for k in calls[0].keywords)
 calls[0].keywords.append(ast.keyword(arg='path_smooth',value=ast.Constant(VALUE)));ast.fix_missing_locations(new)
 assert ast.dump(strip(new),include_attributes=False)==ast.dump(baseline,include_attributes=False)
 return baseline,new
def patched_function(fun):
 source=inspect.getsource(fun);baseline,new=transform(source);ns=dict(fun.__globals__);exec(compile(new,fun.__code__.co_filename,'exec'),ns)
 return ns[fun.__name__],dict(function=fun.__name__,baseline_source_sha256=hashlib.sha256(source.encode()).hexdigest(),H352_ast_sha256=hashlib.sha256(ast.dump(baseline,include_attributes=False).encode()).hexdigest(),restoration_exact=True,delta={'path_smooth':[0.0,VALUE]},num_leaves=15,max_depth=3,learning_rate_and_rounds_unchanged=True)
def install(G):
 records=[]
 for mod,name in [('topk','fit_ranker'),('bagged','fit_ranker_n')]:
  fun,record=patched_function(getattr(G[mod],name));setattr(G[mod],name,fun);records.append(record)
 return dict(status='PASS',functions=records,path_smooth=VALUE,num_leaves=15,max_depth=3,veto_unchanged=True,no_new_seed_keyword=True,learning_rate_and_rounds_unchanged=True)
