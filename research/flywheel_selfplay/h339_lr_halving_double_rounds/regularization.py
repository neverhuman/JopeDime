"""Only halve learning_rate and double n_estimators at two pinned ranker constructors."""
import ast
import copy
import functools
import hashlib
import inspect
import json
from pathlib import Path

BASE = dict(reg_lambda=1.0, colsample_bytree=0.8, min_child_samples=20)
AUDIT = None


def constructor(tree):
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Attribute) and n.func.attr == 'LGBMRanker']
    assert len(calls) == 1
    return calls[0]


def changed_tree(original):
    tree = copy.deepcopy(original)
    kws = {k.arg: k for k in constructor(tree).keywords}
    assert ast.unparse(kws['learning_rate'].value) == 'float(lr)'
    assert ast.unparse(kws['n_estimators'].value) in ['200', 'int(n_est)']
    old = {k: copy.deepcopy(kws[k].value) for k in ['learning_rate', 'n_estimators']}
    for key, value in BASE.items():
        assert ast.literal_eval(kws[key].value) == value
    kws['learning_rate'].value = ast.BinOp(left=old['learning_rate'], op=ast.Div(), right=ast.Constant(2))
    kws['n_estimators'].value = ast.BinOp(left=old['n_estimators'], op=ast.Mult(), right=ast.Constant(2))
    ast.fix_missing_locations(tree)
    restored = copy.deepcopy(tree)
    for kw in constructor(restored).keywords:
        if kw.arg in old:
            kw.value = old[kw.arg]
    assert ast.dump(restored, include_attributes=False) == ast.dump(original, include_attributes=False)
    return tree


def check_record(record):
    p, b = record['params'], record['base_requested']
    assert p['learning_rate'] == b['learning_rate'] / 2
    assert p['n_estimators'] == b['n_estimators'] * 2
    assert p['learning_rate'] * p['n_estimators'] == b['learning_rate'] * b['n_estimators']
    assert all(p[k] == v for k, v in BASE.items())
    assert p['objective'] == 'lambdarank' and p['n_jobs'] == 1 and 'label_gain' not in p
    assert p['random_state'] == b['random_state']
    assert record['native_learning_rate'] == p['learning_rate']
    assert record['native_iterations_requested'] == p['n_estimators']
    assert 0 < record['native_tree_count'] <= p['n_estimators']
    assert record['rows'] == record['groups'] * record['k']


def transform(fn):
    original = inspect.getsource(fn)
    tree = changed_tree(ast.parse(original))
    source = ast.unparse(tree) + '\n'
    namespace = dict(fn.__globals__)
    exec(compile(tree, '<H339-two-ranker-constructor-values>', 'exec'), namespace)
    transformed = namespace[fn.__name__]
    role = fn.__module__ + '.' + fn.__name__

    @functools.wraps(fn)
    def audited(*args, **kwargs):
        assert 'sample_weight' not in kwargs
        bound = inspect.signature(fn).bind(*args, **kwargs).arguments
        model = transformed(*args, **kwargs)
        record = dict(role=role, params=model.get_params(),
                      base_requested=dict(learning_rate=float(bound['lr']), n_estimators=int(bound.get('n_est', 200)), random_state=int(bound['random_state'])),
                      groups=int(bound['n_groups']), k=int(bound['k']), rows=int(bound['X'].shape[0]), features=int(bound['X'].shape[1]),
                      native_learning_rate=model.booster_.params['learning_rate'],
                      native_iterations_requested=model.booster_.params['num_iterations'],
                      native_tree_count=model.booster_.num_trees(),
                      model_sha256=hashlib.sha256(model.booster_.model_to_string().encode()).hexdigest())
        check_record(record)
        if AUDIT is not None:
            with Path(AUDIT).open('a') as out:
                out.write(json.dumps(record, sort_keys=True, allow_nan=False) + '\n')
        return model
    receipt = dict(module=fn.__module__, function=fn.__name__, original_sha256=hashlib.sha256(original.encode()).hexdigest(),
                   source=source, base=BASE, treatment='learning_rate /2 and n_estimators *2 only', restored_AST_exact=True,
                   selected_prefix_calls_unchanged=True)
    return audited, receipt


def install(G):
    result = []
    for module, name in [(G['topk'], 'fit_ranker'), (G['bagged'], 'fit_ranker_n')]:
        fn, receipt = transform(getattr(module, name))
        setattr(module, name, fn)
        result.append(receipt)
    return result


def audit_summary(path):
    data = Path(path).read_bytes()
    rows = [json.loads(x) for x in data.splitlines()]
    roles = {}
    for row in rows:
        check_record(row)
        roles[row['role']] = roles.get(row['role'], 0) + 1
    return dict(status='PASS', fit_calls=len(rows), roles=roles, audit_sha256=hashlib.sha256(data).hexdigest())
