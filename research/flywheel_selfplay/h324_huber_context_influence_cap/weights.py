"""Spread-derived mean-one weights at ranker fits only; no loss/label/gate change."""

import ast, inspect, hashlib, json
from collections import Counter
from pathlib import Path
import numpy as np

AUDIT = None
ALLOWED = None


def digest(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def record(obj):
    if AUDIT is not None:
        with Path(AUDIT).open("a") as f:
            f.write(json.dumps(obj, sort_keys=True) + "\n")


def context_weights(cids, packets, k):
    assert cids and k == 8
    if ALLOWED is not None:
        assert set(cids) <= ALLOWED
    utilities = np.asarray([packets[c]["utility_k"] for c in cids], dtype=np.float64)
    assert utilities.shape == (len(cids), k) and np.isfinite(utilities).all()
    spreads = utilities.max(axis=1) - utilities.min(axis=1)
    delta = float(np.median(spreads))
    assert np.isfinite(delta) and delta > 0, "ZERO_MEDIAN_SPREAD: undefined mean-one normalization; stop"
    raw = np.minimum(1.0, delta / np.maximum(spreads, 1e-9))
    w = raw / raw.mean()
    assert np.isfinite(w).all() and (w > 0).all() and abs(w.mean() - 1) < 1e-12
    slots = np.repeat(w, k)
    record(dict(event="weights", contexts=len(cids), slots=len(slots),
        context_order_sha256=hashlib.sha256("\n".join(cids).encode()).hexdigest(),
        spread_sha256=digest(spreads), delta=delta, sha256=digest(slots),
        min=float(w.min()), max=float(w.max()), mean=float(w.mean()),
        capped_count=int(np.sum(raw < 1.0)), capped_fraction=float(np.mean(raw < 1.0)),
        quantile_levels=[0,.01,.05,.25,.5,.75,.95,.99,1],
        weight_quantiles=np.quantile(w,[0,.01,.05,.25,.5,.75,.95,.99,1]).tolist(),
        includes_context_multiplicity=True))
    return slots


def validate(weight, n, role):
    w = np.asarray(weight, dtype=np.float64)
    assert (
        w.shape == (n,)
        and np.isfinite(w).all()
        and (w > 0).all()
        and abs(w.mean() - 1) < 1e-12
    )
    record(dict(event="fit", role=role, slots=n, sha256=digest(w)))
    return w


def transformed(module, name, kind):
    original = inspect.getsource(getattr(module, name))
    tree = ast.parse(original)
    fn = tree.body[0]
    changed = 0
    if kind == "boundary":
        fn.args.kwonlyargs.append(ast.arg(arg="sample_weight"))
        fn.args.kw_defaults.append(None)
        for node in ast.walk(fn):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "fit"
            ):
                assert not any(k.arg == "sample_weight" for k in node.keywords)
                node.keywords.append(
                    ast.keyword(
                        arg="sample_weight",
                        value=ast.Call(
                            func=ast.Name(id="_h324_validate", ctx=ast.Load()),
                            args=[
                                ast.Name(id="sample_weight", ctx=ast.Load()),
                                ast.Call(
                                    func=ast.Name(id="len", ctx=ast.Load()),
                                    args=[ast.Name(id="X", ctx=ast.Load())],
                                    keywords=[],
                                ),
                                ast.Constant(value=module.__name__ + "." + name),
                            ],
                            keywords=[],
                        ),
                    )
                )
                changed += 1
        assert changed == 1, (name, changed)
    else:
        for node in ast.walk(fn):
            if not isinstance(node, ast.Call):
                continue
            target = ast.unparse(node.func)
            if kind == "veto" and target == "topk.fit_ranker":
                x = ast.unparse(node.args[0])
                assert x in ["X_tr", "X_fit"]
                cids = "tr" if x == "X_tr" else "fit_cids"
                packet_map = "packets"
                k = "k"
            elif kind == "member" and target == "bagged.fit_ranker_n":
                cids = "bcids"
                packet_map = "outer_train_packets"
                k = "ens.K"
            else:
                continue
            assert not any(kw.arg == "sample_weight" for kw in node.keywords)
            node.keywords.append(
                ast.keyword(
                    arg="sample_weight",
                    value=ast.parse(
                        f"_h324_weights({cids}, {packet_map}, {k})", mode="eval"
                    ).body,
                )
            )
            changed += 1
        assert changed == ({"veto": 2, "member": 1}[kind]), (name, changed)
    ast.fix_missing_locations(tree)
    source = ast.unparse(tree) + "\n"
    module.__dict__.update(_h324_weights=context_weights, _h324_validate=validate)
    exec(
        compile(tree, "<H324 weighted " + module.__name__ + "." + name + ">", "exec"),
        module.__dict__,
    )
    return dict(
        module=module.__name__,
        function=name,
        kind=kind,
        sites=changed,
        original_sha256=hashlib.sha256(original.encode()).hexdigest(),
        transformed_sha256=hashlib.sha256(source.encode()).hexdigest(),
        source=source,
    )


def install(G):
    return [
        transformed(G["topk"], "fit_ranker", "boundary"),
        transformed(G["bagged"], "fit_ranker_n", "boundary"),
        transformed(G["veto"], "run_cell_veto", "veto"),
        transformed(G["r302"], "fit_and_freeze_member", "member"),
    ]


def audit_summary(path):
    pending = []
    counts = Counter()
    calls = 0
    for line in Path(path).read_text().splitlines():
        r = json.loads(line)
        if r["event"] == "weights":
            pending.append(r["sha256"])
            assert abs(r["mean"] - 1) < 1e-12
        else:
            assert r["event"] == "fit" and pending.pop(0) == r["sha256"]
            counts[r["role"]] += 1
            calls += 1
    assert not pending and len(counts) == 2 and calls > 0
    return dict(
        status="PASS",
        fit_calls=calls,
        roles=dict(counts),
        every_weight_forwarded=True,
        audit_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
    )
