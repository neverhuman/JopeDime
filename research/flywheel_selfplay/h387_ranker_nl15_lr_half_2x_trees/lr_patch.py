"""Ranker LR*.5/trees*2, with matching positive prediction limits; veto untouched."""
import ast,copy,hashlib,inspect,functools
LEAVES=15

def calls(t):return [x for x in ast.walk(t) if isinstance(x,ast.Call) and isinstance(x.func,ast.Attribute) and x.func.attr=='LGBMRanker']
def strip(t):
 t=copy.deepcopy(t)
 for c in calls(t):
  kw={k.arg:k for k in c.keywords}
  for k,v in [('learning_rate',.5),('n_estimators',2)]:
   node=kw[k].value;assert isinstance(node,ast.BinOp) and isinstance(node.op,ast.Mult) and ast.literal_eval(node.right)==v
   kw[k].value=node.left
 return t

def transform(source):
 from nl15_baseline_patch import transform as nl15_transform
 _,old=nl15_transform(source);new=copy.deepcopy(old);cc=calls(new);assert len(cc)==1
 kw={k.arg:k for k in cc[0].keywords};assert len(kw)==len(cc[0].keywords)
 for name,want in [('objective','lambdarank'),('boosting_type','gbdt'),('num_leaves',15),('max_depth',3),('min_child_samples',20),('reg_lambda',1.0),('subsample',.8),('subsample_freq',1),('colsample_bytree',.8)]:assert ast.literal_eval(kw[name].value)==want
 assert ast.dump(kw['learning_rate'].value)==ast.dump(ast.parse('float(lr)',mode='eval').body)
 assert ast.dump(kw['n_estimators'].value) in [ast.dump(ast.Constant(200)),ast.dump(ast.parse('int(n_est)',mode='eval').body)]
 assert not set(kw)&{'min_split_gain','min_gain_to_split','bagging_by_query','bagging_seed','bagging_fraction','reg_alpha','lambda_l1','lambda_l2','objective_seed','min_child_weight','min_sum_hessian_in_leaf','sigmoid','path_smooth','label_gain','extra_trees','max_delta_step','drop_rate','callbacks'}
 for k,v in [('learning_rate',.5),('n_estimators',2)]:kw[k].value=ast.BinOp(left=kw[k].value,op=ast.Mult(),right=ast.Constant(v))
 ast.fix_missing_locations(new);assert ast.dump(old,include_attributes=False)==ast.dump(strip(new),include_attributes=False)
 return old,new

def patched_function(fun):
 source=inspect.getsource(fun);old,new=transform(source);ns=dict(fun.__globals__);exec(compile(new,fun.__code__.co_filename,'exec'),ns)
 return ns[fun.__name__],dict(function=fun.__name__,baseline_source_sha256=hashlib.sha256(source.encode()).hexdigest(),baseline_ast_sha256=hashlib.sha256(ast.dump(old,include_attributes=False).encode()).hexdigest(),restoration_exact=True,delta={'learning_rate_multiplier':.5,'num_iterations_multiplier':2},added_keywords=[],num_leaves=15,max_depth=3)

def install(G):
 records=[]
 for mod,name in [('topk','fit_ranker'),('bagged','fit_ranker_n')]:
  fun,r=patched_function(getattr(G[mod],name));setattr(G[mod],name,fun);records.append(r)
 return dict(status='PASS',functions=records,learning_rate_multiplier=.5,num_iterations_multiplier=2,exact_product_preserved=True,min_split_gain=0,bagging_fraction=.8,bagging_freq=1,bagging_by_query=False,num_leaves=15,max_depth=3,veto_unchanged=True,objective='lambdarank',labels_and_default_label_gain_unchanged=True)

def prediction_wrapper(original):
 @functools.wraps(original)
 def predict(self,data,start_iteration=0,num_iteration=None,*args,**kwargs):
  # Positive limits are baseline grid coordinates. None/-1 already mean all trees.
  if num_iteration is not None and num_iteration>0:
   objective=self.params.get('objective') or self.dump_model(num_iteration=1)['objective']
   if objective.split()[0]=='lambdarank':
    assert start_iteration==0;num_iteration=int(num_iteration)*2
  return original(self,data,start_iteration,num_iteration,*args,**kwargs)
 return predict

def install_prediction_mapping():
 import lightgbm as lgb
 assert not getattr(lgb.Booster.predict,'_h387_lrhalf',False),'prediction mapping may be installed once per phase'
 wrapped=prediction_wrapper(lgb.Booster.predict);wrapped._h387_lrhalf=True;lgb.Booster.predict=wrapped
 return dict(status='PASS',ranker_positive_prediction_limits_multiplier=2,veto_unchanged=True,unbounded_predictions_unchanged=True,grid_coordinates='Original LR/tree labels retained for seed/config identity; map only actual ranker LR/tree inputs and positive prediction limits.')
