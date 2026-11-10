"""Change only existing subsample .8 to .6; restoring .8 yields exact BASE AST."""
from pathlib import Path
import ast,copy,hashlib
FORBIDDEN=('bagging_seed','feature_fraction_seed','feature_fraction_bynode','max_delta_step','label_gain','extra_trees','extra_seed')
def tree_hash(node):return hashlib.sha256(ast.dump(node,include_attributes=False).encode()).hexdigest()
def ranker_call(node):
 calls=[n for n in ast.walk(node) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name) and n.func.value.id=='lgb' and n.func.attr=='LGBMRanker'];assert len(calls)==1
 return calls[0]
def rowsub_function(original):
 baseline=copy.deepcopy(original);changed=copy.deepcopy(original);call=ranker_call(changed);kw={x.arg:x for x in call.keywords}
 assert not any(x in kw for x in FORBIDDEN)
 assert isinstance(kw['subsample'].value,ast.Constant) and kw['subsample'].value.value==.8
 assert isinstance(kw['subsample_freq'].value,ast.Constant) and kw['subsample_freq'].value.value==1
 kw['subsample'].value=ast.Constant(value=.6);ast.fix_missing_locations(changed)
 restored=copy.deepcopy(changed);next(x for x in ranker_call(restored).keywords if x.arg=='subsample').value=ast.Constant(value=.8)
 assert ast.dump(restored,include_attributes=False)==ast.dump(baseline,include_attributes=False)
 return changed,dict(BASE_AST_sha256=tree_hash(baseline),ROWSUB_AST_sha256=tree_hash(changed),restored_BASE_AST_sha256=tree_hash(restored),restoring_point8_restores_exact_BASE_AST=True,only_changed_keyword='subsample',BASE_value=.8,ROWSUB_value=.6,subsample_freq=1,new_keywords=[],bagging_seed='original derived seed; no added keyword')
def install_function(module,function_name):
 path=Path(module.__file__);raw=path.read_bytes();original=next(x for x in ast.parse(raw).body if isinstance(x,ast.FunctionDef) and x.name==function_name)
 changed,receipt=rowsub_function(original);exec(compile(ast.fix_missing_locations(ast.Module(body=[changed],type_ignores=[])),str(path),'exec'),module.__dict__)
 receipt.update(function=function_name,module=module.__name__,source_sha256=hashlib.sha256(raw).hexdigest());return receipt
