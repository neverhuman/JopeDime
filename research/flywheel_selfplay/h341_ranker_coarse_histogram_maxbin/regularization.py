"""Add only max_bin=63; audit fresh ranker Dataset construction and native bins."""
import ast
import copy
import functools
import hashlib
import inspect
import json
import os
from pathlib import Path
import numpy as np

BASE=dict(reg_lambda=1.0,colsample_bytree=0.8,min_child_samples=20)
AUDIT=None
DATASET_SEQUENCE=0


def constructor(tree):
    calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='LGBMRanker']
    assert len(calls)==1
    return calls[0]


def changed_tree(original):
    tree=copy.deepcopy(original)
    call=constructor(tree)
    assert not any(k.arg in ['max_bin','max_bin_by_feature'] for k in call.keywords)
    call.keywords.append(ast.keyword(arg='max_bin',value=ast.Constant(63)))
    ast.fix_missing_locations(tree)
    restored=copy.deepcopy(tree)
    constructor(restored).keywords=[k for k in constructor(restored).keywords if k.arg!='max_bin']
    assert ast.dump(restored,include_attributes=False)==ast.dump(original,include_attributes=False)
    return tree


def check_record(row):
    p,b=row['params'],row['base_requested']
    assert row['changed_constructor_keys']==['max_bin'] and p['max_bin']==63
    assert p['learning_rate']==b['learning_rate'] and p['n_estimators']==b['n_estimators']
    assert p['random_state']==b['random_state']
    assert all(p[k]==v for k,v in BASE.items())
    assert p['objective']=='lambdarank' and p['n_jobs']==1 and 'label_gain' not in p
    assert row['native_max_bin']==63 and row['native_learning_rate']==p['learning_rate']
    assert row['native_iterations_requested']==p['n_estimators']
    assert 0<row['native_tree_count']<=p['n_estimators']
    assert row['rows']==row['groups']*row['k']
    ds=row['dataset']
    assert ds['new_objects']==1 and ds['reference_is_none'] and ds['initial_handle_is_none']
    assert ds['input_is_numpy'] and ds['max_bin']==63 and ds['max_observed_bins']<=63
    assert sum(ds['bin_count_histogram'].values())==row['features']


def transform(fn):
    original=inspect.getsource(fn);base_tree=ast.parse(original)
    expressions={k.arg:compile(ast.fix_missing_locations(ast.Expression(copy.deepcopy(k.value))),'<H341-BASE-keyword>','eval') for k in constructor(base_tree).keywords}
    tree=changed_tree(base_tree);source=ast.unparse(tree)+'\n'
    namespace=dict(fn.__globals__);exec(compile(tree,'<H341-max_bin63-only>','exec'),namespace)
    transformed=namespace[fn.__name__];role=fn.__module__+'.'+fn.__name__

    @functools.wraps(fn)
    def audited(*args,**kwargs):
        import lightgbm as lgb
        global DATASET_SEQUENCE
        assert 'sample_weight' not in kwargs
        bound=inspect.signature(fn).bind(*args,**kwargs).arguments
        assert isinstance(bound['X'],np.ndarray), 'Pre-binned Dataset or file input prohibited'
        expected_kwargs={k:eval(code,fn.__globals__,bound) for k,code in expressions.items()}
        expected=lgb.LGBMRanker(**expected_kwargs).get_params()  # constructor only, no BASE fit
        original_dataset=lgb.sklearn.Dataset
        observed=[]

        def fresh_dataset(*a,**kw):
            global DATASET_SEQUENCE
            assert not a and isinstance(kw['data'],np.ndarray)
            assert kw.get('reference') is None and kw['params']['max_bin']==63
            ds=original_dataset(**kw)
            assert ds._handle is None and ds.reference is None
            DATASET_SEQUENCE+=1
            observed.append((ds,f'{os.getpid()}:{DATASET_SEQUENCE}'))
            return ds

        lgb.sklearn.Dataset=fresh_dataset
        try:
            model=transformed(*args,**kwargs)
        finally:
            lgb.sklearn.Dataset=original_dataset
        assert len(observed)==1
        ds,serial=observed[0]
        assert ds._handle is not None and ds.get_params()['max_bin']==63
        bins=[ds.feature_num_bin(i) for i in range(bound['X'].shape[1])]
        counts={str(n):bins.count(n) for n in sorted(set(bins))}
        params=model.get_params()
        changed=sorted(k for k in set(params)|set(expected) if params.get(k)!=expected.get(k))
        native=model.booster_.params
        row=dict(role=role,params=params,changed_constructor_keys=changed,
                 base_constructor_sha256=hashlib.sha256(json.dumps(expected,sort_keys=True).encode()).hexdigest(),
                 base_requested=dict(learning_rate=float(bound['lr']),n_estimators=int(bound.get('n_est',200)),random_state=int(bound['random_state'])),
                 groups=int(bound['n_groups']),k=int(bound['k']),rows=int(bound['X'].shape[0]),features=int(bound['X'].shape[1]),
                 native_max_bin=native['max_bin'],native_learning_rate=native['learning_rate'],native_iterations_requested=native['num_iterations'],native_tree_count=model.booster_.num_trees(),
                 dataset=dict(serial=serial,new_objects=1,reference_is_none=True,initial_handle_is_none=True,input_is_numpy=True,max_bin=ds.get_params()['max_bin'],max_observed_bins=max(bins),bin_count_histogram=counts),
                 model_sha256=hashlib.sha256(model.booster_.model_to_string().encode()).hexdigest())
        check_record(row)
        model._h341_fit_receipt=row
        if AUDIT is not None:
            with Path(AUDIT).open('a') as out:out.write(json.dumps(row,sort_keys=True,allow_nan=False)+'\n')
        return model

    return audited,dict(module=fn.__module__,function=fn.__name__,original_sha256=hashlib.sha256(original.encode()).hexdigest(),source=source,base=BASE,treatment='max_bin=63 only',restored_AST_exact=True,fresh_dataset_audit=True)


def install(G):
    receipts=[]
    for module,name in [(G['topk'],'fit_ranker'),(G['bagged'],'fit_ranker_n')]:
        fn,r=transform(getattr(module,name));setattr(module,name,fn);receipts.append(r)
    return receipts


def audit_summary(path):
    raw=Path(path).read_bytes();rows=[json.loads(x) for x in raw.splitlines()];roles={};serials=set()
    for row in rows:
        check_record(row)
        assert row['dataset']['serial'] not in serials;serials.add(row['dataset']['serial'])
        roles[row['role']]=roles.get(row['role'],0)+1
    return dict(status='PASS',fit_calls=len(rows),fresh_datasets_verified=len(serials),roles=roles,audit_sha256=hashlib.sha256(raw).hexdigest())
