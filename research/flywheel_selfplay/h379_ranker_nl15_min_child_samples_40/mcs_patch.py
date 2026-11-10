"""Derive the H352 NL15 AST, then change only existing ranker min_child_samples20->40."""

import ast, copy, hashlib, inspect
from leaves_patch import transform as nl15_transform, calls

MIN_CHILD_SAMPLES = 40


def restore(tree):
    tree = copy.deepcopy(tree)
    for call in calls(tree):
        kw = {x.arg: x for x in call.keywords}
        assert ast.literal_eval(kw["min_child_samples"].value) == 40
        kw["min_child_samples"].value = ast.Constant(20)
    return ast.fix_missing_locations(tree)


def transform(source):
    _, h352 = nl15_transform(source)
    new = copy.deepcopy(h352)
    rankers = calls(new)
    assert len(rankers) == 1
    kw = {x.arg: x for x in rankers[0].keywords}
    for name, want in [
        ("num_leaves", 15),
        ("max_depth", 3),
        ("min_child_samples", 20),
        ("reg_lambda", 1.0),
    ]:
        assert ast.literal_eval(kw[name].value) == want
    assert not any(
        x in kw
        for x in [
            "reg_alpha",
            "lambda_l2",
            "seed",
            "bagging_seed",
            "feature_fraction_seed",
            "data_random_seed",
        ]
    )
    kw["min_child_samples"].value = ast.Constant(MIN_CHILD_SAMPLES)
    ast.fix_missing_locations(new)
    assert ast.dump(h352, include_attributes=False) == ast.dump(
        restore(new), include_attributes=False
    )
    return h352, new


def install(G):
    records = []
    for module, function in [("topk", "fit_ranker"), ("bagged", "fit_ranker_n")]:
        old = getattr(G[module], function)
        source = inspect.getsource(old)
        h352, new = transform(source)
        scope = dict(old.__globals__)
        exec(compile(new, old.__code__.co_filename, "exec"), scope)
        setattr(G[module], function, scope[function])
        records.append(
            dict(
                module=module,
                function=function,
                H352_ast_sha256=hashlib.sha256(
                    ast.dump(h352, include_attributes=False).encode()
                ).hexdigest(),
                treatment_ast_sha256=hashlib.sha256(
                    ast.dump(new, include_attributes=False).encode()
                ).hexdigest(),
                restoration_to_entire_H352_AST_exact=True,
                delta={"min_child_samples": [20, 40]},
                added_keywords=[],
            )
        )
    return dict(
        status="PASS",
        functions=records,
        num_leaves=15,
        max_depth=3,
        min_child_samples=40,
        reg_lambda=1.0,
        veto_unchanged=True,
        sole_native_delta={"min_data_in_leaf": [20, 40]},
    )
