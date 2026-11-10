"""One ranker keyword only; complete inverse AST proof against frozen BASE."""
import ast,copy,hashlib,inspect,textwrap
PATH_SMOOTH=20.0

def calls(tree):
 return [n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='LGBMRanker']

def strip(tree):
 tree=copy.deepcopy(tree)
 for n in calls(tree):n.keywords=[k for k in n.keywords if k.arg!='path_smooth']
 return tree

def transform(source):
 old=ast.parse(textwrap.dedent(source));new=copy.deepcopy(old)
 constructors=calls(new);assert len(constructors)==1
 c=constructors[0];kw={x.arg:x.value for x in c.keywords}
 assert 'path_smooth' not in kw and ast.literal_eval(kw['boosting_type'])=='gbdt'
 assert ast.literal_eval(kw['min_child_samples'])==20
 assert not set(kw)&{'label_gain','drop_rate','skip_drop','extra_trees','max_delta_step'}
 c.keywords.append(ast.keyword(arg='path_smooth',value=ast.Constant(PATH_SMOOTH)));ast.fix_missing_locations(new)
 assert ast.dump(old,include_attributes=False)==ast.dump(strip(new),include_attributes=False)
 return old,new

def patched_function(function):
 source=inspect.getsource(function);old,new=transform(source);namespace=dict(function.__globals__)
 exec(compile(new,function.__code__.co_filename,'exec'),namespace)
 return namespace[function.__name__],dict(function=function.__name__,source_sha256=hashlib.sha256(source.encode()).hexdigest(),baseline_ast_sha256=hashlib.sha256(ast.dump(old,include_attributes=False).encode()).hexdigest(),restoration_exact=True,delta={'path_smooth':[0.0,PATH_SMOOTH]})

def install(G):
 out=[]
 for key,name in [('topk','fit_ranker'),('bagged','fit_ranker_n')]:
  f,receipt=patched_function(getattr(G[key],name));setattr(G[key],name,f);out.append(receipt)
 return dict(status='PASS',functions=out,veto_unchanged=True)
