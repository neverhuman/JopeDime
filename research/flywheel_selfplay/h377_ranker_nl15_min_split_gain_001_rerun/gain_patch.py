"""Sole ranker addition min_split_gain=.01; exact H352 whole AST inverse."""
import ast,copy,hashlib,inspect,textwrap
LEAVES=15
def calls(t):return [x for x in ast.walk(t) if isinstance(x,ast.Call) and isinstance(x.func,ast.Attribute) and x.func.attr=='LGBMRanker']
def strip(t):
 t=copy.deepcopy(t)
 for c in calls(t):c.keywords=[k for k in c.keywords if k.arg!='min_split_gain']
 return t
def transform(source):
 from nl15_baseline_patch import transform as nl15_transform
 _,old=nl15_transform(source);new=copy.deepcopy(old);cc=calls(new);assert len(cc)==1
 kw={k.arg:k for k in cc[0].keywords};assert len(kw)==len(cc[0].keywords)
 for name,want in [('objective','lambdarank'),('boosting_type','gbdt'),('num_leaves',15),('max_depth',3),('min_child_samples',20),('reg_lambda',1.0),('subsample',.8),('subsample_freq',1),('colsample_bytree',.8)]:assert ast.literal_eval(kw[name].value)==want
 assert not set(kw)&{'min_split_gain','min_gain_to_split','bagging_by_query','bagging_seed','bagging_fraction','reg_alpha','lambda_l1','lambda_l2','objective_seed','min_child_weight','min_sum_hessian_in_leaf','sigmoid','path_smooth','label_gain','extra_trees','max_delta_step','drop_rate','callbacks'}
 cc[0].keywords.append(ast.keyword(arg='min_split_gain',value=ast.Constant(.01)));ast.fix_missing_locations(new)
 assert ast.dump(old,include_attributes=False)==ast.dump(strip(new),include_attributes=False)
 return old,new
def patched_function(fun):
 source=inspect.getsource(fun);old,new=transform(source);ns=dict(fun.__globals__);exec(compile(new,fun.__code__.co_filename,'exec'),ns)
 return ns[fun.__name__],dict(function=fun.__name__,baseline_source_sha256=hashlib.sha256(source.encode()).hexdigest(),baseline_ast_sha256=hashlib.sha256(ast.dump(old,include_attributes=False).encode()).hexdigest(),restoration_exact=True,delta={'min_gain_to_split':[0,.01]},added_keywords=['min_split_gain'],num_leaves=15,max_depth=3)
def install(G):
 records=[]
 for mod,name in [('topk','fit_ranker'),('bagged','fit_ranker_n')]:
  fun,r=patched_function(getattr(G[mod],name));setattr(G[mod],name,fun);records.append(r)
 return dict(status='PASS',functions=records,min_split_gain=.01,bagging_fraction=.8,bagging_freq=1,bagging_by_query=False,num_leaves=15,max_depth=3,veto_unchanged=True,objective='lambdarank',labels_and_default_label_gain_unchanged=True)
