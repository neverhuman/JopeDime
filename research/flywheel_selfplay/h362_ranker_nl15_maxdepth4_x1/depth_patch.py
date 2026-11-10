"""H352 NL15 ASTs plus their sole existing max_depth value 3->4."""
import ast,copy,hashlib,inspect
import leaves_patch as h352
DEPTH=4;LEAVES=15
calls=h352.calls
def strip(tree):
 tree=copy.deepcopy(tree)
 for call in calls(tree):next(k for k in call.keywords if k.arg=='max_depth').value=ast.Constant(3)
 return tree
def transform(source):
 _,old=h352.transform(source)
 new=copy.deepcopy(old);call=calls(new)[0];kw={k.arg:k for k in call.keywords}
 assert ast.literal_eval(kw['num_leaves'].value)==15 and ast.literal_eval(kw['max_depth'].value)==3
 kw['max_depth'].value=ast.Constant(DEPTH);ast.fix_missing_locations(new)
 assert ast.dump(old,include_attributes=False)==ast.dump(strip(new),include_attributes=False)
 return old,new
def patched_function(fun):
 source=inspect.getsource(fun);old,new=transform(source);ns=dict(fun.__globals__);exec(compile(new,fun.__code__.co_filename,'exec'),ns)
 return ns[fun.__name__],dict(function=fun.__name__,original_BASE_source_sha256=hashlib.sha256(source.encode()).hexdigest(),H352_ast_sha256=hashlib.sha256(ast.dump(old,include_attributes=False).encode()).hexdigest(),restoration_exact=True,delta={'max_depth':[3,4]},added_keywords=[],num_leaves=15)
def install(G):
 records=[]
 for mod,name in [('topk','fit_ranker'),('bagged','fit_ranker_n')]:
  fun,record=patched_function(getattr(G[mod],name));setattr(G[mod],name,fun);records.append(record)
 return dict(status='PASS',functions=records,num_leaves=15,max_depth=4,veto_unchanged=True,objective='lambdarank',labels_and_default_label_gain_unchanged=True)
