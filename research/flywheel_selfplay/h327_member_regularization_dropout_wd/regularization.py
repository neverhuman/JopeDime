"""Three fixed ranker constructor overrides; no gate or calling-code changes."""
import ast
import functools
import hashlib
import inspect
import json
from pathlib import Path

BASE = dict(reg_lambda=1.0, colsample_bytree=0.8, min_child_samples=20)
REG = dict(reg_lambda=max(10 * BASE['reg_lambda'], 1.0),
           colsample_bytree=0.8 * BASE['colsample_bytree'],
           min_child_samples=2 * BASE['min_child_samples'])
AUDIT = None


def transform(fn):
    original = inspect.getsource(fn)
    tree = ast.parse(original)
    constructors = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                    and isinstance(n.func, ast.Attribute) and n.func.attr == 'LGBMRanker']
    assert len(constructors) == 1
    kws = {x.arg: x for x in constructors[0].keywords}
    for key, value in BASE.items():
        assert isinstance(kws[key].value, ast.Constant) and kws[key].value.value == value
        kws[key].value = ast.Constant(REG[key])
    ast.fix_missing_locations(tree)
    source = ast.unparse(tree) + '\n'
    restored = ast.parse(source)
    constructor = next(n for n in ast.walk(restored) if isinstance(n, ast.Call)
                       and isinstance(n.func, ast.Attribute) and n.func.attr == 'LGBMRanker')
    for kw in constructor.keywords:
        if kw.arg in BASE:
            kw.value = ast.Constant(BASE[kw.arg])
    assert ast.dump(restored, include_attributes=False) == ast.dump(ast.parse(original), include_attributes=False)
    namespace = dict(fn.__globals__)
    exec(compile(tree, '<H327-fixed-ranker-overrides>', 'exec'), namespace)
    transformed = namespace[fn.__name__]
    role = fn.__module__ + '.' + fn.__name__

    @functools.wraps(fn)
    def audited(*args, **kwargs):
        assert 'sample_weight' not in kwargs
        model = transformed(*args, **kwargs)
        params = model.get_params()
        assert all(params[k] == v for k, v in REG.items())
        assert params['objective'] == 'lambdarank' and params['n_jobs'] == 1
        assert 'label_gain' not in params
        bound = inspect.signature(fn).bind(*args, **kwargs).arguments
        assert params['random_state'] == int(bound['random_state'])
        assert params['learning_rate'] == float(bound['lr'])
        assert params['n_estimators'] == int(bound.get('n_est', 200))
        if AUDIT is not None:
            record = dict(role=role, params=params, groups=int(bound['n_groups']),
                          k=int(bound['k']), rows=int(bound['X'].shape[0]),
                          features=int(bound['X'].shape[1]),
                          model_sha256=hashlib.sha256(model.booster_.model_to_string().encode()).hexdigest())
            with Path(AUDIT).open('a') as out:
                out.write(json.dumps(record, sort_keys=True, allow_nan=False) + '\n')
        return model

    receipt = dict(module=fn.__module__, function=fn.__name__,
                   original_sha256=hashlib.sha256(original.encode()).hexdigest(),
                   source=source, base=BASE, reg=REG, restored_AST_exact=True)
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
    rows = [json.loads(line) for line in data.splitlines()]
    roles = {}
    for row in rows:
        assert all(row['params'][k] == v for k, v in REG.items())
        assert row['params']['objective'] == 'lambdarank'
        assert row['rows'] == row['groups'] * row['k']
        roles[row['role']] = roles.get(row['role'], 0) + 1
    return dict(status="PASS", fit_calls=len(rows), roles=roles, audit_sha256=hashlib.sha256(data).hexdigest())
