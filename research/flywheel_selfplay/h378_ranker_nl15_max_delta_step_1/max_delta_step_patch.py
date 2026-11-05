"""Only add ranker max_delta_step1; removing it restores each full H352 NL15 AST."""
import ast,copy,hashlib,inspect
LEAVES=15
MAX_DELTA_STEP=1.0
def calls(tree):return [x for x in ast.walk(tree) if isinstance(x,ast.Call) and isinstance(x.func,ast.Attribute) and x.func.attr=='LGBMRanker']
def strip(tree):
 tree=copy.deepcopy(tree)
 for call in calls(tree):
  found=[kw for kw in call.keywords if kw.arg=='max_delta_step'];assert len(found)==1 and ast.literal_eval(found[0].value)==MAX_DELTA_STEP
  call.keywords=[kw for kw in call.keywords if kw.arg!='max_delta_step']
 return tree
def transform(source):
 from nl15_baseline_patch import transform as nl15_transform
 _,old=nl15_transform(source);new=copy.deepcopy(old);cc=calls(new);assert len(cc)==1
 kw={k.arg:k for k in cc[0].keywords};assert len(kw)==len(cc[0].keywords)
 for name,want in [('objective','lambdarank'),('boosting_type','gbdt'),('num_leaves',15),('max_depth',3),('min_child_samples',20),('reg_lambda',1.0),('subsample',.8),('subsample_freq',1),('colsample_bytree',.8)]:assert ast.literal_eval(kw[name].value)==want
 assert not set(kw)&{'extra_trees','extra_seed','feature_fraction','feature_fraction_seed','bagging_by_query','reg_alpha','lambda_l1','lambda_l2','objective_seed','min_child_weight','min_sum_hessian_in_leaf','min_split_gain','sigmoid','path_smooth','label_gain','max_delta_step','max_tree_output','max_leaf_output','drop_rate','callbacks'}
 cc[0].keywords.append(ast.keyword(arg='max_delta_step',value=ast.Constant(MAX_DELTA_STEP)));ast.fix_missing_locations(new)
 assert ast.dump(old,include_attributes=False)==ast.dump(strip(new),include_attributes=False)
 return old,new
def patched_function(fun):
 source=inspect.getsource(fun);old,new=transform(source);ns=dict(fun.__globals__);exec(compile(new,fun.__code__.co_filename,'exec'),ns)
 return ns[fun.__name__],dict(function=fun.__name__,baseline_source_sha256=hashlib.sha256(source.encode()).hexdigest(),baseline_ast_sha256=hashlib.sha256(ast.dump(old,include_attributes=False).encode()).hexdigest(),restoration_exact=True,delta={'max_delta_step':[0,1.0]},added_keywords=['max_delta_step'],num_leaves=15,max_depth=3)
def install(G):
 records=[]
 for module,name in [('topk','fit_ranker'),('bagged','fit_ranker_n')]:
  fun,record=patched_function(getattr(G[module],name));setattr(G[module],name,fun);records.append(record)
 return dict(status='PASS',functions=records,max_delta_step=MAX_DELTA_STEP,colsample_bytree=.8,num_leaves=15,max_depth=3,veto_unchanged=True,extra_trees=False,objective='lambdarank',labels_and_default_label_gain_unchanged=True)
