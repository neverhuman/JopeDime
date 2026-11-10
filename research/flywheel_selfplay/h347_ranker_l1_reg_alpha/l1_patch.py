"""Only reg_alpha=.5 is added to the original BASE ranker constructors."""
import ast,copy,hashlib,inspect,textwrap
L1=.5
def calls(t):return [x for x in ast.walk(t) if isinstance(x,ast.Call) and isinstance(x.func,ast.Attribute) and x.func.attr=='LGBMRanker']
def strip(t):
 t=copy.deepcopy(t)
 for c in calls(t):c.keywords=[k for k in c.keywords if k.arg!='reg_alpha']
 return t
def transform(source):
 old=ast.parse(textwrap.dedent(source));new=copy.deepcopy(old);cc=calls(new);assert len(cc)==1
 kw={k.arg:k for k in cc[0].keywords};assert len(kw)==len(cc[0].keywords)
 assert ast.literal_eval(kw['objective'].value)=='lambdarank' and ast.literal_eval(kw['boosting_type'].value)=='gbdt'
 assert not set(kw)&{'reg_alpha','lambda_l1','min_child_weight','min_sum_hessian_in_leaf','min_split_gain','sigmoid','path_smooth','label_gain','extra_trees','max_delta_step','drop_rate','callbacks'}
 cc[0].keywords.append(ast.keyword(arg='reg_alpha',value=ast.Constant(L1)));ast.fix_missing_locations(new)
 assert ast.dump(old,include_attributes=False)==ast.dump(strip(new),include_attributes=False)
 return old,new
def patched_function(fun):
 source=inspect.getsource(fun);old,new=transform(source);ns=dict(fun.__globals__);exec(compile(new,fun.__code__.co_filename,'exec'),ns)
 return ns[fun.__name__],dict(function=fun.__name__,baseline_source_sha256=hashlib.sha256(source.encode()).hexdigest(),baseline_ast_sha256=hashlib.sha256(ast.dump(old,include_attributes=False).encode()).hexdigest(),restoration_exact=True,delta={'reg_alpha':[0.0,L1]},added_keywords=['reg_alpha'])
def install(G):
 records=[]
 for mod,name in [('topk','fit_ranker'),('bagged','fit_ranker_n')]:
  fun,r=patched_function(getattr(G[mod],name));setattr(G[mod],name,fun);records.append(r)
 return dict(status='PASS',functions=records,L1=L1,veto_unchanged=True,objective='lambdarank',labels_and_default_label_gain_unchanged=True)
