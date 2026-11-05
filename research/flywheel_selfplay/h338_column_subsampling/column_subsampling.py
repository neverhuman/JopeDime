"""Insert only two fixed ranker keywords; remove both to recover exact pinned BASE AST."""
from pathlib import Path
import ast,copy,hashlib
KEYS=('feature_fraction_bynode','feature_fraction_seed')
def tree_hash(node):return hashlib.sha256(ast.dump(node,include_attributes=False).encode()).hexdigest()
def ranker_call(node):
 calls=[n for n in ast.walk(node) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name) and n.func.value.id=='lgb' and n.func.attr=='LGBMRanker'];assert len(calls)==1
 return calls[0]
def colsub_function(original):
 baseline=copy.deepcopy(original);changed=copy.deepcopy(original);call=ranker_call(changed)
 assert not any(x.arg in [*KEYS,'max_delta_step','label_gain','extra_trees','extra_seed'] for x in call.keywords)
 seed=next(x.value for x in call.keywords if x.arg=='random_state')
 call.keywords.extend([ast.keyword(arg=KEYS[0],value=ast.Constant(value=.8)),ast.keyword(arg=KEYS[1],value=copy.deepcopy(seed))]);ast.fix_missing_locations(changed)
 restored=copy.deepcopy(changed);rc=ranker_call(restored);rc.keywords=[x for x in rc.keywords if x.arg not in KEYS]
 assert ast.dump(restored,include_attributes=False)==ast.dump(baseline,include_attributes=False)
 return changed,dict(BASE_AST_sha256=tree_hash(baseline),COLSUB_AST_sha256=tree_hash(changed),removed_override_AST_sha256=tree_hash(restored),removing_two_keywords_restores_exact_BASE_AST=True,feature_fraction_bynode=.8,feature_fraction_seed_AST=ast.dump(seed,include_attributes=False))
def install_function(module,function_name):
 path=Path(module.__file__);raw=path.read_bytes();tree=ast.parse(raw);original=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name==function_name)
 changed,receipt=colsub_function(original);exec(compile(ast.fix_missing_locations(ast.Module(body=[changed],type_ignores=[])),str(path),'exec'),module.__dict__)
 receipt.update(function=function_name,module=module.__name__,source_sha256=hashlib.sha256(raw).hexdigest(),only_added_keywords=list(KEYS),label_gain='BASE implicit default',colsample_bytree=.8)
 return receipt
