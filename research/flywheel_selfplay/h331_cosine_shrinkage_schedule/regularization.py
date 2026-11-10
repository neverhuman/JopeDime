"""Fixed COS ranker callback; all BASE constructors, gates and callers retained."""
import ast
import functools
import hashlib
import inspect
import json
import math
import struct
from pathlib import Path

BASE = dict(reg_lambda=1.0, colsample_bytree=0.8, min_child_samples=20)
AUDIT = None


def rates(lr, rounds):
    assert math.isfinite(lr) and lr > 0 and isinstance(rounds, int) and rounds > 0
    return [lr * (1 + math.cos(math.pi * t / rounds)) for t in range(rounds)]


def trace_receipt(values):
    return dict(rounds=len(values), first=values[0], midpoint=values[len(values)//2],
                last=values[-1], mean=math.fsum(values)/len(values),
                sha256=hashlib.sha256(struct.pack('<'+'d'*len(values), *values)).hexdigest())


def callbacks(model):
    import lightgbm as lgb
    params = model.get_params()
    expected = rates(float(params['learning_rate']), int(params['n_estimators']))
    observed = []
    model._h331_rate_trace = observed
    reset = lgb.reset_parameter(learning_rate=lambda t: expected[t])
    assert reset.order == 10 and reset.before_iteration

    def verify_rate(env):
        t = env.iteration - env.begin_iteration
        assert t == len(observed) and env.end_iteration-env.begin_iteration == len(expected)
        assert env.params['learning_rate'] == expected[t]
        assert env.model.params['learning_rate'] == expected[t]
        observed.append(float(env.params['learning_rate']))
    verify_rate.order = 11
    verify_rate.before_iteration = True
    return [reset, verify_rate]


def transform(fn):
    original = inspect.getsource(fn)
    tree = ast.parse(original)
    constructors = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                    and isinstance(n.func, ast.Attribute) and n.func.attr == 'LGBMRanker']
    assert len(constructors) == 1
    kws = {x.arg: x for x in constructors[0].keywords}
    for key, value in BASE.items():
        assert isinstance(kws[key].value, ast.Constant) and kws[key].value.value == value
    fits = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
            and isinstance(n.func.value, ast.Name) and n.func.value.id == 'model' and n.func.attr == 'fit']
    assert len(fits) == 1 and not any(k.arg == 'callbacks' for k in fits[0].keywords)
    fits[0].keywords.append(ast.keyword(arg='callbacks', value=ast.Call(func=ast.Name(id='_h331_callbacks',ctx=ast.Load()),args=[ast.Name(id='model',ctx=ast.Load())],keywords=[])))
    ast.fix_missing_locations(tree)
    source = ast.unparse(tree)+'\n'
    restored = ast.parse(source)
    for n in ast.walk(restored):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == 'fit':
            n.keywords = [k for k in n.keywords if k.arg != 'callbacks']
    assert ast.dump(restored,include_attributes=False) == ast.dump(ast.parse(original),include_attributes=False)
    namespace = dict(fn.__globals__, _h331_callbacks=callbacks)
    exec(compile(tree,'<H331-cosine-ranker-callback>','exec'), namespace)
    transformed = namespace[fn.__name__]
    role = fn.__module__+'.'+fn.__name__

    @functools.wraps(fn)
    def audited(*args, **kwargs):
        assert 'sample_weight' not in kwargs
        model = transformed(*args, **kwargs)
        params = model.get_params()
        bound = inspect.signature(fn).bind(*args, **kwargs).arguments
        assert all(params[k] == v for k,v in BASE.items())
        assert params['objective'] == 'lambdarank' and params['n_jobs'] == 1 and 'label_gain' not in params
        assert params['random_state'] == int(bound['random_state'])
        assert params['learning_rate'] == float(bound['lr'])
        assert params['n_estimators'] == int(bound.get('n_est',200))
        expected = rates(params['learning_rate'], params['n_estimators'])
        assert model._h331_rate_trace == expected
        assert model.booster_.params['learning_rate'] == expected[-1]
        record = dict(role=role, params=params, groups=int(bound['n_groups']),k=int(bound['k']),
                      rows=int(bound['X'].shape[0]),features=int(bound['X'].shape[1]),
                      schedule=trace_receipt(model._h331_rate_trace),
                      native_final_learning_rate=model.booster_.params['learning_rate'],
                      native_tree_count=model.booster_.num_trees(),
                      model_sha256=hashlib.sha256(model.booster_.model_to_string().encode()).hexdigest())
        model._h331_schedule_receipt = record
        if AUDIT is not None:
            with Path(AUDIT).open('a') as out:out.write(json.dumps(record,sort_keys=True,allow_nan=False)+'\n')
        return model
    receipt = dict(module=fn.__module__,function=fn.__name__,original_sha256=hashlib.sha256(original.encode()).hexdigest(),
                   source=source,base=BASE,treatment='COS callback only',restored_AST_exact=True)
    return audited,receipt


def install(G):
    result=[]
    for module,name in [(G['topk'],'fit_ranker'),(G['bagged'],'fit_ranker_n')]:
        fn,receipt=transform(getattr(module,name));setattr(module,name,fn);result.append(receipt)
    return result


def audit_summary(path):
    data=Path(path).read_bytes();rows=[json.loads(line) for line in data.splitlines()];roles={}
    for row in rows:
        params=row['params'];assert all(params[k]==v for k,v in BASE.items())
        assert params['objective']=='lambdarank' and row['rows']==row['groups']*row['k']
        expected=rates(params['learning_rate'],params['n_estimators'])
        assert row['schedule']==trace_receipt(expected) and row['native_final_learning_rate']==expected[-1]
        roles[row['role']]=roles.get(row['role'],0)+1
    return dict(status='PASS',fit_calls=len(rows),roles=roles,audit_sha256=hashlib.sha256(data).hexdigest())
