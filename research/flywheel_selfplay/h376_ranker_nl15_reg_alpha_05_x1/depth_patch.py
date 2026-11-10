"""H352 NL15 ASTs plus their sole ranker reg_alpha0->0.5 keyword; all H352 binning retained."""
import ast,copy,hashlib,inspect
import leaves_patch as h352
DEPTH=3;LEAVES=15;MAX_BIN=255;REG_ALPHA=.5
calls=h352.calls
def strip(tree):
 tree=copy.deepcopy(tree)
 for call in calls(tree):call.keywords=[k for k in call.keywords if k.arg!='reg_alpha']
 return tree
def transform(source):
 _,old=h352.transform(source)
 new=copy.deepcopy(old);call=calls(new)[0];kw={k.arg:k for k in call.keywords}
 assert ast.literal_eval(kw['num_leaves'].value)==15 and ast.literal_eval(kw['max_depth'].value)==3
 assert 'reg_alpha' not in kw and 'max_bin' not in kw
 call.keywords.append(ast.keyword(arg='reg_alpha',value=ast.Constant(REG_ALPHA)));ast.fix_missing_locations(new)
 assert ast.dump(old,include_attributes=False)==ast.dump(strip(new),include_attributes=False)
 return old,new
def patched_function(fun):
 source=inspect.getsource(fun);old,new=transform(source);ns=dict(fun.__globals__);exec(compile(new,fun.__code__.co_filename,'exec'),ns)
 return ns[fun.__name__],dict(function=fun.__name__,original_BASE_source_sha256=hashlib.sha256(source.encode()).hexdigest(),H352_ast_sha256=hashlib.sha256(ast.dump(old,include_attributes=False).encode()).hexdigest(),restoration_exact=True,delta={'reg_alpha':[0,.5]},added_keywords=['reg_alpha'],num_leaves=15)
def install(G):
 records=[]
 for mod,name in [('topk','fit_ranker'),('bagged','fit_ranker_n')]:
  fun,record=patched_function(getattr(G[mod],name));setattr(G[mod],name,fun);records.append(record)
 return dict(status='PASS',functions=records,num_leaves=15,max_depth=3,max_bin=255,reg_alpha=.5,veto_unchanged=True,objective='lambdarank',labels_and_default_label_gain_unchanged=True)
