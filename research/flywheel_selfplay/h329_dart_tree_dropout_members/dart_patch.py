"""Exact ranker-only DART delta against the pinned H287 function ASTs."""
import ast
import copy
import hashlib
import inspect
import textwrap

EXTRA = dict(drop_rate=0.1, skip_drop=0.5, max_drop=50,
             uniform_drop=False, xgboost_dart_mode=False)
ADDED = set(EXTRA) | {'drop_seed'}


def transform(source):
    original = ast.parse(textwrap.dedent(source))
    tree = copy.deepcopy(original)
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Attribute) and n.func.attr == 'LGBMRanker']
    assert len(calls) == 1
    call = calls[0]
    args = {k.arg: k for k in call.keywords}
    assert not ADDED & set(args)
    assert ast.literal_eval(args['boosting_type'].value) == 'gbdt'
    args['boosting_type'].value = ast.Constant('dart')
    call.keywords.extend(ast.keyword(arg=k, value=ast.Constant(v)) for k,v in EXTRA.items())
    call.keywords.append(ast.keyword(arg='drop_seed', value=ast.Call(func=ast.Name(id='int',ctx=ast.Load()), args=[ast.Name(id='random_state',ctx=ast.Load())], keywords=[])))
    ast.fix_missing_locations(tree)
    restored = strip(tree)
    assert ast.dump(restored, include_attributes=False) == ast.dump(original, include_attributes=False)
    return original, tree


def strip(tree):
    tree = copy.deepcopy(tree)
    for n in ast.walk(tree):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == 'LGBMRanker':
            n.keywords = [k for k in n.keywords if k.arg not in ADDED]
            for k in n.keywords:
                if k.arg == 'boosting_type':
                    k.value = ast.Constant('gbdt')
    return tree


def patched_function(function):
    source = inspect.getsource(function)
    original, tree = transform(source)
    namespace = dict(function.__globals__)
    exec(compile(tree, function.__code__.co_filename, 'exec'), namespace)
    result = namespace[function.__name__]
    record = dict(function=function.__name__, baseline_source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                  baseline_ast_sha256=hashlib.sha256(ast.dump(original,include_attributes=False).encode()).hexdigest(),
                  patched_ast_sha256=hashlib.sha256(ast.dump(tree,include_attributes=False).encode()).hexdigest(),
                  restoration_exact=True, extra=EXTRA, boosting_type=['gbdt','dart'], drop_seed='int(random_state) at the original per-fit call site')
    return result, record


def install(G):
    records=[]
    for module, name in [(G['topk'],'fit_ranker'),(G['bagged'],'fit_ranker_n')]:
        function, record = patched_function(getattr(module,name))
        setattr(module,name,function);records.append(record)
    return dict(status='PASS',functions=records,veto_source_sha256=hashlib.sha256(inspect.getsource(G['veto'].fit_gate).encode()).hexdigest())
