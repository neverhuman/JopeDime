"""The only FAMW intervention: family-equal sample weights at pinned fit calls."""

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


def family_weights(cids, fam_by_cid, k):
    assert cids and k > 0
    if ALLOWED is not None:
        assert set(cids) <= ALLOWED
    fams = [fam_by_cid[c] for c in cids]
    counts = Counter(fams)
    N = len(cids)
    F = len(counts)
    w = np.array([N / (F * counts[f]) for f in fams], dtype=np.float64)
    assert np.isfinite(w).all() and (w > 0).all() and abs(w.mean() - 1) < 1e-12
    totals = {f: counts[f] * N / (F * counts[f]) for f in counts}
    assert max(totals.values()) - min(totals.values()) < 1e-10
    slots = np.repeat(w, k)
    record(
        dict(
            event="weights",
            contexts=N,
            families=F,
            slots=len(slots),
            counts_histogram=dict(Counter(counts.values())),
            context_order_sha256=hashlib.sha256("\n".join(cids).encode()).hexdigest(),
            sha256=digest(slots),
            min=float(w.min()),
            max=float(w.max()),
            mean=float(w.mean()),
            family_total=N / F,
            includes_context_multiplicity=True,
        )
    )
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
                            func=ast.Name(id="_h322_validate", ctx=ast.Load()),
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
            if kind == "veto" and target in ["topk.fit_ranker", "fit_gate"]:
                x = ast.unparse(node.args[0])
                assert x in ["X_tr", "X_fit"]
                cids = "tr" if x == "X_tr" else "fit_cids"
                fam = "fam_by_cid"
                k = "k"
            elif kind == "member" and target == "bagged.fit_ranker_n":
                cids = "bcids"
                fam = "outer_train_fam_by_cid"
                k = "ens.K"
            else:
                continue
            assert not any(kw.arg == "sample_weight" for kw in node.keywords)
            node.keywords.append(
                ast.keyword(
                    arg="sample_weight",
                    value=ast.parse(
                        f"_h322_weights({cids}, {fam}, {k})", mode="eval"
                    ).body,
                )
            )
            changed += 1
        assert changed == ({"veto": 4, "member": 1}[kind]), (name, changed)
    ast.fix_missing_locations(tree)
    source = ast.unparse(tree) + "\n"
    module.__dict__.update(_h322_weights=family_weights, _h322_validate=validate)
    exec(
        compile(tree, "<H322 weighted " + module.__name__ + "." + name + ">", "exec"),
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
        transformed(G["veto"], "fit_gate", "boundary"),
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
    assert not pending and len(counts) == 3 and calls > 0
    return dict(
        status="PASS",
        fit_calls=calls,
        roles=dict(counts),
        every_weight_forwarded=True,
        audit_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
    )
