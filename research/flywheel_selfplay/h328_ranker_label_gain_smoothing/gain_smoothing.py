"""Insert the one preregistered ranker keyword; prove exact BASE AST recovery."""
from pathlib import Path
import ast,copy,hashlib
BASE_EFFECTIVE_GAINS=[0,1,3,7,15,31,63,127]
LINEAR_GAINS=list(range(8))
def tree_hash(node):return hashlib.sha256(ast.dump(node,include_attributes=False).encode()).hexdigest()
def ranker_call(node):
 calls=[n for n in ast.walk(node) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name) and n.func.value.id=='lgb' and n.func.attr=='LGBMRanker']
 assert len(calls)==1
 return calls[0]
def smooth_function(original):
 baseline=copy.deepcopy(original);changed=copy.deepcopy(original);call=ranker_call(changed)
 assert not any(k.arg=='label_gain' for k in call.keywords)
 call.keywords.append(ast.keyword(arg='label_gain',value=ast.parse('list(range(k))',mode='eval').body))
 ast.fix_missing_locations(changed)
 restored=copy.deepcopy(changed);rc=ranker_call(restored);assert sum(k.arg=='label_gain' for k in rc.keywords)==1
 rc.keywords=[k for k in rc.keywords if k.arg!='label_gain']
 assert ast.dump(restored,include_attributes=False)==ast.dump(baseline,include_attributes=False)
 return changed,dict(BASE_AST_sha256=tree_hash(baseline),LGS_AST_sha256=tree_hash(changed),removed_override_AST_sha256=tree_hash(restored),removing_one_keyword_restores_exact_BASE_AST=True)
def install_function(module,function_name):
 path=Path(module.__file__);raw=path.read_bytes();tree=ast.parse(raw)
 original=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==function_name)
 changed,receipt=smooth_function(original);code=compile(ast.fix_missing_locations(ast.Module(body=[changed],type_ignores=[])),str(path),'exec')
 exec(code,module.__dict__)
 receipt.update(function=function_name,module=module.__name__,source_sha256=hashlib.sha256(raw).hexdigest(),BASE_effective_label_gain=BASE_EFFECTIVE_GAINS,LGS_label_gain=LINEAR_GAINS)
 return receipt
