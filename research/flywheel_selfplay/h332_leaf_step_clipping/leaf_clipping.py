"""Add only max_delta_step=c to pinned ranker constructors; exact AST inverse."""
from pathlib import Path
import ast,copy,hashlib,math

def tree_hash(node):return hashlib.sha256(ast.dump(node,include_attributes=False).encode()).hexdigest()
def ranker_call(node):
 calls=[n for n in ast.walk(node) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name) and n.func.value.id=='lgb' and n.func.attr=='LGBMRanker'];assert len(calls)==1
 return calls[0]
def clip_function(original,cap):
 assert math.isfinite(cap) and cap>0
 baseline=copy.deepcopy(original);changed=copy.deepcopy(original);call=ranker_call(changed)
 assert not any(x.arg in ['max_delta_step','label_gain','extra_trees','extra_seed'] for x in call.keywords)
 call.keywords.append(ast.keyword(arg='max_delta_step',value=ast.Constant(value=float(cap))));ast.fix_missing_locations(changed)
 restored=copy.deepcopy(changed);rc=ranker_call(restored);rc.keywords=[x for x in rc.keywords if x.arg!='max_delta_step']
 assert ast.dump(restored,include_attributes=False)==ast.dump(baseline,include_attributes=False)
 return changed,dict(BASE_AST_sha256=tree_hash(baseline),CLIP_AST_sha256=tree_hash(changed),removed_override_AST_sha256=tree_hash(restored),removing_one_keyword_restores_exact_BASE_AST=True,cap=cap)
def install_function(module,function_name,cap):
 path=Path(module.__file__);raw=path.read_bytes();tree=ast.parse(raw);original=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name==function_name)
 changed,receipt=clip_function(original,cap);exec(compile(ast.fix_missing_locations(ast.Module(body=[changed],type_ignores=[])),str(path),'exec'),module.__dict__)
 receipt.update(function=function_name,module=module.__name__,source_sha256=hashlib.sha256(raw).hexdigest(),only_added_keyword='max_delta_step',label_gain='BASE implicit default')
 return receipt
