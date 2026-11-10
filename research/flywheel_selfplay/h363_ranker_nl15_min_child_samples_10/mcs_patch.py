"""Derive H352 NL15 AST first, then change only its existing min_child_samples20->10."""

import ast, copy, hashlib, inspect
from leaves_patch import transform as nl15_transform, calls

MIN_CHILD_SAMPLES = 10


def restore(t):
    t = copy.deepcopy(t)
    for c in calls(t):
        kw = {k.arg: k for k in c.keywords}
        assert ast.literal_eval(kw["min_child_samples"].value) == 10
        kw["min_child_samples"].value = ast.Constant(20)
    return ast.fix_missing_locations(t)


def transform(source):
    _, h352 = nl15_transform(source)
    new = copy.deepcopy(h352)
    cc = calls(new)
    assert len(cc) == 1
    kw = {k.arg: k for k in cc[0].keywords}
    assert (
        ast.literal_eval(kw["num_leaves"].value) == 15
        and ast.literal_eval(kw["max_depth"].value) == 3
        and ast.literal_eval(kw["min_child_samples"].value) == 20
    )
    kw["min_child_samples"].value = ast.Constant(MIN_CHILD_SAMPLES)
    ast.fix_missing_locations(new)
    assert ast.dump(h352, include_attributes=False) == ast.dump(
        restore(new), include_attributes=False
    )
    return h352, new


def install(G):
    records = []
    for mod, name in [("topk", "fit_ranker"), ("bagged", "fit_ranker_n")]:
        f = getattr(G[mod], name)
        source = inspect.getsource(f)
        h352, new = transform(source)
        namespace = dict(f.__globals__)
        exec(compile(new, f.__code__.co_filename, "exec"), namespace)
        setattr(G[mod], name, namespace[name])
        records.append(
            dict(
                module=mod,
                function=name,
                H352_ast_sha256=hashlib.sha256(
                    ast.dump(h352, include_attributes=False).encode()
                ).hexdigest(),
                treatment_ast_sha256=hashlib.sha256(
                    ast.dump(new, include_attributes=False).encode()
                ).hexdigest(),
                restoration_to_entire_H352_AST_exact=True,
                delta={"min_child_samples": [20, 10]},
                added_keywords=[],
            )
        )
    return dict(
        status="PASS",
        functions=records,
        num_leaves=15,
        max_depth=3,
        min_child_samples=10,
        veto_unchanged=True,
        sole_native_delta={"min_data_in_leaf": [20, 10]},
    )
