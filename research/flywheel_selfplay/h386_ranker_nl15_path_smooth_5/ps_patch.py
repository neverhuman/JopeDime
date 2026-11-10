"""Derive H352 NL15, then add only ranker path_smooth=5; inverse is exact."""

import ast, copy, hashlib, inspect
from leaves_patch import transform as nl15_transform, calls

PATH_SMOOTH = 5.0


def restore(tree):
    tree = copy.deepcopy(tree)
    for call in calls(tree):
        selected = [k for k in call.keywords if k.arg == "path_smooth"]
        assert len(selected) == 1 and ast.literal_eval(selected[0].value) == PATH_SMOOTH
        call.keywords = [k for k in call.keywords if k.arg != "path_smooth"]
    return ast.fix_missing_locations(tree)


def transform(source):
    _, h352 = nl15_transform(source)
    new = copy.deepcopy(h352)
    rankers = calls(new)
    assert len(rankers) == 1
    kw = {k.arg: k for k in rankers[0].keywords}
    for name, want in [
        ("num_leaves", 15),
        ("max_depth", 3),
        ("min_child_samples", 20),
        ("reg_lambda", 1.0),
    ]:
        assert ast.literal_eval(kw[name].value) == want
    assert not set(kw) & {
        "path_smooth",
        "reg_alpha",
        "lambda_l2",
        "seed",
        "bagging_seed",
        "feature_fraction_seed",
        "data_random_seed",
    }
    rankers[0].keywords.append(
        ast.keyword(arg="path_smooth", value=ast.Constant(PATH_SMOOTH))
    )
    ast.fix_missing_locations(new)
    assert ast.dump(h352, include_attributes=False) == ast.dump(
        restore(new), include_attributes=False
    )
    return h352, new


def install(G):
    records = []
    for module, function in [("topk", "fit_ranker"), ("bagged", "fit_ranker_n")]:
        old = getattr(G[module], function)
        h352, new = transform(inspect.getsource(old))
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
                delta={"path_smooth": [0, 5]},
                added_keywords=["path_smooth"],
            )
        )
    return dict(
        status="PASS",
        functions=records,
        num_leaves=15,
        max_depth=3,
        min_child_samples=20,
        reg_lambda=1.0,
        path_smooth=PATH_SMOOTH,
        veto_unchanged=True,
        sole_native_delta={"path_smooth": [0, 5]},
    )
