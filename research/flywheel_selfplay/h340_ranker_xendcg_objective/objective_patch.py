"""Exactly one existing ranker keyword value changes; no added keywords."""
import ast,copy,inspect,hashlib,textwrap

def calls(tree):return [x for x in ast.walk(tree) if isinstance(x,ast.Call) and isinstance(x.func,ast.Attribute) and x.func.attr=='LGBMRanker']
def strip(tree):
 tree=copy.deepcopy(tree)
 for c in calls(tree):
  for k in c.keywords:
   if k.arg=='objective':k.value=ast.Constant('lambdarank')
 return tree

def transform(source):
 old=ast.parse(textwrap.dedent(source));new=copy.deepcopy(old);cc=calls(new);assert len(cc)==1
 kw={k.arg:k for k in cc[0].keywords};assert len(kw)==len(cc[0].keywords);assert ast.literal_eval(kw['objective'].value)=='lambdarank' and ast.literal_eval(kw['boosting_type'].value)=='gbdt'
 assert not set(kw)&{'label_gain','path_smooth','max_delta_step','extra_trees','drop_rate','callbacks'}
 kw['objective'].value=ast.Constant('rank_xendcg');ast.fix_missing_locations(new)
 assert ast.dump(old,include_attributes=False)==ast.dump(strip(new),include_attributes=False)
 return old,new

def patched_function(fun):
 source=inspect.getsource(fun);old,new=transform(source);ns=dict(fun.__globals__);exec(compile(new,fun.__code__.co_filename,'exec'),ns)
 return ns[fun.__name__],dict(function=fun.__name__,baseline_source_sha256=hashlib.sha256(source.encode()).hexdigest(),baseline_ast_sha256=hashlib.sha256(ast.dump(old,include_attributes=False).encode()).hexdigest(),restoration_exact=True,delta={'objective':['lambdarank','rank_xendcg']},added_keywords=[])

def install(G):
 records=[]
 for mod,name in [('topk','fit_ranker'),('bagged','fit_ranker_n')]:
  fun,record=patched_function(getattr(G[mod],name));setattr(G[mod],name,fun);records.append(record)
 return dict(status='PASS',functions=records,veto_unchanged=True,label_gain='Ignored by rank_xendcg objective; no keyword added or labels changed')
