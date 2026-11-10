"""Single fixed ranker LambdaRank sigmoid override; complete BASE AST inverse."""
import ast,copy,inspect,hashlib,textwrap,json,functools
from pathlib import Path
@functools.lru_cache(maxsize=1)
def value():
 r=json.loads((Path(__file__).parent/'sigmoid_value.json').read_text());assert r['status']=='PASS' and r['SIGMA']==0.5 and r['BASE_effective_native_sigmoid']==1.0;return r['SIGMA']
def calls(t):return [x for x in ast.walk(t) if isinstance(x,ast.Call) and isinstance(x.func,ast.Attribute) and x.func.attr=='LGBMRanker']
def strip(t):
 t=copy.deepcopy(t)
 for c in calls(t):c.keywords=[k for k in c.keywords if k.arg!='sigmoid']
 return t
def transform(source,v):
 assert isinstance(v,(float,int)) and v==0.5
 old=ast.parse(textwrap.dedent(source));new=copy.deepcopy(old);cc=calls(new);assert len(cc)==1
 kw={k.arg:k for k in cc[0].keywords};assert len(kw)==len(cc[0].keywords)
 assert ast.literal_eval(kw['objective'].value)=='lambdarank' and ast.literal_eval(kw['boosting_type'].value)=='gbdt'
 assert not set(kw)&{'sigmoid','min_child_weight','min_sum_hessian_in_leaf','path_smooth','label_gain','extra_trees','max_delta_step','drop_rate','callbacks'}
 cc[0].keywords.append(ast.keyword(arg='sigmoid',value=ast.Constant(float(v))));ast.fix_missing_locations(new)
 assert ast.dump(old,include_attributes=False)==ast.dump(strip(new),include_attributes=False)
 return old,new
def patched_function(fun,v):
 source=inspect.getsource(fun);old,new=transform(source,v);ns=dict(fun.__globals__);exec(compile(new,fun.__code__.co_filename,'exec'),ns)
 return ns[fun.__name__],dict(function=fun.__name__,baseline_source_sha256=hashlib.sha256(source.encode()).hexdigest(),baseline_ast_sha256=hashlib.sha256(ast.dump(old,include_attributes=False).encode()).hexdigest(),restoration_exact=True,delta={'sigmoid':[1.0,v]},added_keywords=['sigmoid'])
def install(G):
 v=value();records=[]
 for mod,name in [('topk','fit_ranker'),('bagged','fit_ranker_n')]:
  fun,r=patched_function(getattr(G[mod],name),v);setattr(G[mod],name,fun);records.append(r)
 return dict(status='PASS',functions=records,SIGMA=v,veto_unchanged=True,objective='lambdarank',labels_and_default_label_gain_unchanged=True)
