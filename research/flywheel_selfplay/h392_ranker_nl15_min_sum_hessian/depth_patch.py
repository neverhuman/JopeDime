"""H352 NL15 ASTs plus their sole ranker min_child_weight keyword (native min_sum_hessian_in_leaf); all H352 binning retained."""
import ast,copy,hashlib,inspect,json
from pathlib import Path
_VALUE=json.loads((Path(__file__).parent/"hessian_value.json").read_text());assert _VALUE["status"]=="PASS"
MINHESS=_VALUE["value"];BASE_HESS=_VALUE["baseline"];assert MINHESS>BASE_HESS
import leaves_patch as h352
DEPTH=3;LEAVES=15;MAX_BIN=255
calls=h352.calls
def strip(tree):
 tree=copy.deepcopy(tree)
 for call in calls(tree):call.keywords=[k for k in call.keywords if k.arg!='min_child_weight']
 return tree
def transform(source):
 _,old=h352.transform(source)
 new=copy.deepcopy(old);call=calls(new)[0];kw={k.arg:k for k in call.keywords}
 assert ast.literal_eval(kw['num_leaves'].value)==15 and ast.literal_eval(kw['max_depth'].value)==3
 assert 'reg_alpha' not in kw and 'max_bin' not in kw and 'min_child_weight' not in kw and 'min_sum_hessian_in_leaf' not in kw
 call.keywords.append(ast.keyword(arg='min_child_weight',value=ast.Constant(MINHESS)));ast.fix_missing_locations(new)
 assert ast.dump(old,include_attributes=False)==ast.dump(strip(new),include_attributes=False)
 return old,new
def patched_function(fun):
 source=inspect.getsource(fun);old,new=transform(source);ns=dict(fun.__globals__);exec(compile(new,fun.__code__.co_filename,'exec'),ns)
 return ns[fun.__name__],dict(function=fun.__name__,original_BASE_source_sha256=hashlib.sha256(source.encode()).hexdigest(),H352_ast_sha256=hashlib.sha256(ast.dump(old,include_attributes=False).encode()).hexdigest(),restoration_exact=True,delta={'min_child_weight':[BASE_HESS,MINHESS]},added_keywords=['min_child_weight'],num_leaves=15)
def install(G):
 records=[]
 for mod,name in [('topk','fit_ranker'),('bagged','fit_ranker_n')]:
  fun,record=patched_function(getattr(G[mod],name));setattr(G[mod],name,fun);records.append(record)
 return dict(status='PASS',functions=records,num_leaves=15,max_depth=3,max_bin=255,reg_alpha=0,min_sum_hessian_in_leaf=MINHESS,veto_unchanged=True,objective='lambdarank',labels_and_default_label_gain_unchanged=True)
