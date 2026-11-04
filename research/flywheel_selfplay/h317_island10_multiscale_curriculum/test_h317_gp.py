"""H317 GP track tests (pytest). No pilot context is ever touched: synthetic contexts, or NON-pilot TRAIN contexts
with a deterministic fake worker. One optional real-worker integration test uses a non-pilot context and 2 evals.

Scratch: $H317_GP_TEST_ROOT (default /home/ubuntu/jope-data/h317-cl/gp/test-scratch/<run id>), never /tmp.
Run: PYTHONDONTWRITEBYTECODE=1 nice -n 19 python3 -m pytest -q -p no:cacheprovider test_h317_gp.py
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import json
import math
import random
import sys
import time
import uuid
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common_h317 as C  # noqa: E402

C.install_guard()

import distributed_10gen_island_gp as D  # noqa: E402
import gp_engine_h317 as E  # noqa: E402
import gp_grammar_h317 as G  # noqa: E402
import gp_lab as L  # noqa: E402
import screener_h317 as S  # noqa: E402
import stacking_h317 as ST  # noqa: E402

TEST_ROOT = Path(os.environ.get("H317_GP_TEST_ROOT", str(C.DATA_ROOT / "gp" / "test-scratch"))) / \
    time.strftime("%Y%m%dT%H%M%S") / uuid.uuid4().hex[:8]


def scratch(name: str) -> Path:
    p = TEST_ROOT / name
    p.mkdir(parents=True, exist_ok=True)
    return p


# ----------------------------------------------------------------------------------------------------------------------
# Fixtures: rows, non-pilot contexts, fake worker
# ----------------------------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def rows():
    return C.train_rows()


@pytest.fixture(scope="module")
def nonpilot(rows):
    folds = C.family_folds(rows)
    prot = set()
    for n in (24, 48, 96):
        prot |= set(C.pilot_shard(rows, folds, n=n))
    fam = {r["context_sha256"]: r["family_id"] for r in rows}
    pf = {fam[c] for c in prot}
    out = [r["context_sha256"] for r in rows if r["context_sha256"] not in prot and r["family_id"] not in pf]
    out.sort(key=lambda c: C.stream_seed("h317gp-test", c))
    return out


def _q(sha: str) -> float:
    return 0.25 + 0.9 * ((C.stream_seed("q", sha) % 10_000) / 10_000.0)


def fake_runner(poison=None, fail_frac: float = 0.04):
    """Deterministic stand-in for bootstrap.s1main: predictions = y + q(sha) * baseline * noise."""

    def rmse(a, b):
        return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))

    def run(ctx, kind, cand, core, slot):
        if (C.stream_seed("fail", cand.sha) % 1000) < fail_frac * 1000:
            return None, 0.001, "rc=1:fake"
        q = _q(cand.sha)
        rng = np.random.default_rng(C.stream_seed("noise", cand.sha, kind))
        if kind == "audit":
            pa = ctx.ya + q * ctx.ba * rng.standard_normal(len(ctx.ya))
            if poison is not None:
                pa = poison(cand, pa, ctx)
            recs = [{"predictions": [0.0], "targets": [0.0], "metrics": {"rmse": 1.0}, "baseline_metrics": {"rmse": 1.0}},
                    {"predictions": pa.tolist(), "targets": ctx.ya.tolist(), "metrics": {"rmse": rmse(pa, ctx.ya)},
                     "baseline_metrics": {"rmse": ctx.ba}}]
        else:
            y0, y1 = (ctx.y0, ctx.y1) if kind == "search" else (ctx.alts[kind].y0, ctx.alts[kind].y1)
            b0 = rmse(y0, np.full(len(y0), y1.mean()))
            b1 = rmse(y1, np.full(len(y1), y0.mean()))
            p0 = y0 + q * b0 * rng.standard_normal(len(y0))
            p1 = y1 + q * b1 * (0.8 + 0.4 * rng.random()) * rng.standard_normal(len(y1))
            recs = [{"predictions": p0.tolist(), "targets": y0.tolist(), "metrics": {"rmse": rmse(p0, y0)},
                     "baseline_metrics": {"rmse": b0}},
                    {"predictions": p1.tolist(), "targets": y1.tolist(), "metrics": {"rmse": rmse(p1, y1)},
                     "baseline_metrics": {"rmse": b1}}]
        return {"status": "completed", "inner_cv": recs, "program": {"program_sha256": cand.sha}}, 0.001, ""

    return run


class HashCoder(D.ContextCoder):
    name = "hash_test"

    def _v(self, cid, role):
        rng = np.random.default_rng(C.stream_seed("coder", cid))
        v = rng.standard_normal(12)
        return {"lm12": v, "dope": v}

    def target(self, cid):
        return self._v(cid, "t")

    def bank(self, cid):
        return self._v(cid, "b")


def make_runtime(name: str, runner, seeds=(317101,)) -> L.Runtime:
    rows = C.train_rows()
    fam_of = {r["context_sha256"]: r["family_id"] for r in rows}
    ds_of = {r["context_sha256"]: r["dataset_id"] for r in rows}
    bank, _ = D.build_bank(rows)
    eng = E.EvalEngine([0, 1, 2, 3], root=scratch(name) / "gproot", runner=runner)
    fb = S.FallbackScreener()
    return L.Runtime(eng, E.AuditLedger(eng), lambda cid: fb, HashCoder(), bank, fam_of, ds_of,
                     L.Store(scratch(name) / "run"), tuple(seeds))


# ----------------------------------------------------------------------------------------------------------------------
# Grammar
# ----------------------------------------------------------------------------------------------------------------------
def test_canonical_bank_presets_and_formatting():
    bank, info = D.build_bank(C.train_rows())
    assert info["noncanonical_dropped"] == 0 and len(bank) > 1000
    for d in G.REGISTRY_PRESETS.values():
        assert G.canonical(d) == d
    assert G.fmt_float(1e-5) == "0.00001" and G.fmt_float(2.0) == "2.0" and G.fmt_float(0.0718617037066272) == \
        "0.0718617037066272"
    assert G.canonical("(pipe X (histgbm2 rounds=3 bags=1))") == "(pipe X (histgbm2 bags=1 rounds=3))"
    with pytest.raises(G.GrammarError):
        G.parse("(portfolio (member (pipe X (portfolio preset=core))) (blend))")
    with pytest.raises(G.GrammarError):
        G.parse("(pipe X (ridge_gcv) (standardize))")
    p = G.canonical("(portfolio (member (pipe X (ridge_gcv))) (member (pipe X (ksmooth k=8))) "
                    "(blend kind=nnls folds=5 repeats=auto guard=clip))")
    assert p.endswith("(blend folds=5 guard=clip kind=nnls repeats=auto))")
    assert [G.island_of(G.REGISTRY_PRESETS[k]) for k in ("histgbm_d6", "ridge_quad", "gpr_lin", "rff_ridge")] == \
        [0, 1, 2, 2]


def test_operators_produce_valid_canonical_programs():
    rng = random.Random(5)
    bank, _ = D.build_bank(C.train_rows())
    parents = [G.parse(b.dsl) for b in bank[:40]]
    made = 0
    for i in range(400):
        a, b = parents[i % 40], parents[(i * 7 + 3) % 40]
        for gen in (G.op_parameter(a, rng), G.op_replace_head(a, rng, i % 4), G.op_insert_stage(a, rng),
                    G.op_delete_stage(a, rng), G.op_crossover(a, b, rng), G.op_synth(i % 4, rng),
                    G.op_portfolio([a, b]), G.head_swap(a, i % 4, rng)):
            c = G.safe_candidate(gen, "t", [], 30)
            if c is not None:
                assert G.canonical(c.dsl) == c.dsl
                made += 1
    assert made > 2000


def test_micro_sem24_parity():
    """Parity with the Micro function body, extracted from its source text with `ast` (no module import, so nothing
    under JopeDimeMicro/assets/benchmark_data can be touched)."""
    import ast

    src = (C.MICRO / "python" / "phase12_multimodal_model.py").read_text()
    fn = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "compute_semantic_struct_sig_24d")
    ns = {"np": np}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "micro_sem24", "exec"), ns)
    compute_semantic_struct_sig_24d = ns["compute_semantic_struct_sig_24d"]
    bank, _ = D.build_bank(C.train_rows())
    f = np.stack([G.featurize(b.dsl)[0] for b in bank[:200]])
    assert np.array_equal(G.micro_sem24(f), compute_semantic_struct_sig_24d(f))


# ----------------------------------------------------------------------------------------------------------------------
# Engine: memo cache, resource guard, AUDIT scope
# ----------------------------------------------------------------------------------------------------------------------
def _synthetic_ctx(cid="synthetic" + "0" * 55, n0=60, n1=55, na=40, seed=1):
    rng = np.random.default_rng(seed)
    y0, y1, ya = rng.normal(size=n0), rng.normal(size=n1), rng.normal(size=na)
    b0 = float(np.sqrt(np.mean((y0 - y1.mean()) ** 2)))
    b1 = float(np.sqrt(np.mean((y1 - y0.mean()) ** 2)))
    ba = float(np.sqrt(np.mean((ya - np.concatenate([y0, y1]).mean()) ** 2)))
    return E.ContextEnvs(cid, "fam-x", 5, None, None, None, Path("/nonexistent"), Path("/nonexistent"), y0, y1, ya,
                         b0, b1, ba, env_sha={"search": "5" * 64, "audit": "a" * 64})


def test_memo_cache_and_persistence():
    root = scratch("memo")
    calls = []
    base = fake_runner(fail_frac=0.0)

    def counting(ctx, kind, cand, core, slot):
        calls.append((kind, cand.sha))
        return base(ctx, kind, cand, core, slot)

    ctx = _synthetic_ctx()
    eng = E.EvalEngine([0, 1], root=root, runner=counting)
    eng._loaded.add((ctx.cid, ctx.mode))
    c = G.Candidate.from_dsl("(pipe X (ridge_gcv))", "t")
    r1 = eng.search(ctx, [c, c])[0]
    r2 = eng.search(ctx, [c])[0]
    assert len(calls) == 1 and r1 is r2 and eng.counters["memo_hits"] == 2
    eng.audit(ctx, [c])
    assert len(calls) == 2  # (cid, sha, kind) keys: audit is a separate entry
    eng2 = E.EvalEngine([0], root=root, runner=counting)
    eng2._load_memo(ctx)
    r3 = eng2.search(ctx, [c])[0]
    assert len(calls) == 2 and r3.sha == r1.sha and np.array_equal(r3.p0, r1.p0) and r3.utility == r1.utility
    # a different envelope (e.g. another AUDIT mask) must never reuse the stored records
    ctx_b = _synthetic_ctx()
    ctx_b.env_sha = {"search": "6" * 64, "audit": "b" * 64}
    eng3 = E.EvalEngine([0], root=root, runner=counting)
    eng3._load_memo(ctx_b)
    assert eng3.counters["memo_env_mismatch_refused"] == 2
    eng3.search(ctx_b, [c])
    assert len(calls) == 3
    assert (root / "memo" / f"{ctx.cid}.split.jsonl").exists()
    for e in (eng, eng2, eng3):
        e.close()


def test_failure_conventions():
    ctx = _synthetic_ctx()
    eng = E.EvalEngine([0], root=scratch("fail"), runner=fake_runner(fail_frac=1.0), persist=False)
    c = G.Candidate.from_dsl("(pipe X (ridge_gcv))", "t")
    r = eng.search(ctx, [c])[0]
    a = eng.audit(ctx, [c])[0]
    assert r.utility == -2.0 and not r.gate and a.status == "failed"
    assert L.value_of({"status": a.status, "utility": None}) == (0.0, -2.0, True)  # SEARCH-mean predictor
    # a terrible but valid AUDIT output stays "completed" with its continuous utility (no -2.0 cliff)
    awful = E.EvalEngine([0], root=scratch("awful"), persist=False,
                         runner=fake_runner(poison=lambda cand, pa, ctx: pa * 0 + 1e3, fail_frac=0.0))
    a2 = awful.audit(ctx, [c])[0]
    assert a2.status == "completed" and a2.utility < -2.0


def test_resource_guard_pauses_and_logs():
    mem = iter([10.0, 10.0, 64.0, 64.0])
    t = {"tot": 0, "idle": 0}

    def cpu():
        t["tot"] += 100
        t["idle"] += 90
        return t["tot"], t["idle"]

    log = scratch("guard") / "pauses.jsonl"
    rg = E.ResourceGuard(min_mem_gib=20.0, max_busy=0.85, poll_s=0.01, log_path=log, mem_fn=lambda: next(mem), cpu_fn=cpu)
    dur = rg.wait()
    assert dur > 0 and len(rg.pauses) == 1
    assert [json.loads(x)["event"] for x in log.read_text().splitlines()] == ["pause", "resume"]
    assert E.read_mem_available_gib() > 0 and E.read_cpu_times()[0] > 0


def test_pool_open_refused_inside_selection_scope(nonpilot):
    p = C.pool_path(nonpilot[0])
    before = E.POOL_GUARD["refused_pool_open_in_selection"]
    with E.selection_scope("x"):
        with pytest.raises(C.GuardViolation):
            open(p, "rb")
    assert E.POOL_GUARD["refused_pool_open_in_selection"] == before + 1


def test_audit_refused_inside_selection_scope():
    ctx = _synthetic_ctx()
    eng = E.EvalEngine([0], root=scratch("scope"), runner=fake_runner(), persist=False)
    with E.selection_scope("x"):
        with pytest.raises(E.AuditLeakError):
            eng.audit(ctx, [G.Candidate.from_dsl("(pipe X (ridge_gcv))", "t")])
    led = E.AuditLedger(eng)
    led.score("P", ctx, 1, [G.Candidate.from_dsl("(pipe X (ridge_gcv))", "t")])
    with pytest.raises(E.AuditLeakError):
        led.score("P", ctx, 1, [G.Candidate.from_dsl("(pipe X (ridge_gcv))", "t")])


# ----------------------------------------------------------------------------------------------------------------------
# Stacking
# ----------------------------------------------------------------------------------------------------------------------
def _synthetic_results(fd: ST.FoldData, k: int = 14, seed: int = 3):
    rng = np.random.default_rng(seed)
    out = []
    dsls = ["(pipe X (histgbm depth=%d lr=0.1 rounds=100))" % (2 + i % 5) if i % 3 == 0 else
            "(pipe X (ridge lambda_frac=0.%02d))" % (1 + i) if i % 3 == 1 else
            "(pipe X (ksmooth gamma=1.0 k=%d))" % (4 + i) for i in range(k)]
    for i, d in enumerate(dsls):
        q = 0.3 + 0.05 * i
        p0 = fd.y0 + q * fd.b0 * rng.standard_normal(len(fd.y0))
        p1 = fd.y1 + q * fd.b1 * rng.standard_normal(len(fd.y1))
        lr0 = math.log(np.sqrt(np.mean((p0 - fd.y0) ** 2)) / fd.b0)
        lr1 = math.log(np.sqrt(np.mean((p1 - fd.y1) ** 2)) / fd.b1)
        s1 = ST.s1_from_lr(lr0, lr1, d.count("("))
        out.append(E.SearchResult("c", G.sha256_text(d), d, "completed", s1["utility"], 0.1 * i, True, lr0, lr1,
                                  s1["mean_lr"], s1["tail_lr"], s1["sd_lr"], d.count("("), p0, p1))
    return out


def _fd(seed=7, n0=120, n1=110):
    rng = np.random.default_rng(seed)
    y0, y1 = rng.normal(size=n0) * 3 + 1, rng.normal(size=n1) * 3 + 1
    return ST.FoldData(y0, y1, float(np.sqrt(np.mean((y0 - y1.mean()) ** 2))), float(np.sqrt(np.mean((y1 - y0.mean()) ** 2))))


def test_stack_weights_on_simplex():
    fd = _fd()
    specs = ST.all_configs(_synthetic_results(fd), fd)
    assert {"single", "CV1", "CV4"} <= set(specs) and all(f"CV3@{n}" in specs for n in ST.N_GRID)
    for name, sp in specs.items():
        for w in (sp.w_audit, sp.w_fold0, sp.w_fold1):
            assert np.all(w >= -1e-12) and abs(w.sum() - 1.0) < 1e-9, name
    assert len(specs["CV3@12"].members) <= 12 and len({*specs["CV3@5"].islands}) >= 3


def _fd_from(y0, y1):
    return ST.FoldData(y0, y1, float(np.sqrt(np.mean((y0 - y1.mean()) ** 2))),
                       float(np.sqrt(np.mean((y1 - y0.mean()) ** 2))))


def test_crossfit_weights_ignore_the_predicted_fold():
    """Weights that predict fold 0 must not depend on fold-0 targets (and vice versa). Baselines are recomputed from
    the perturbed targets exactly as the worker would."""
    fd = _fd()
    res = _synthetic_results(fd)
    mem = ST.top_n_dedup(res, 5, fd.b0)
    rng = np.random.default_rng(11)
    fd_y0 = _fd_from(fd.y0 * 1.9 + rng.normal(size=len(fd.y0)) + 0.7, fd.y1)
    fd_y1 = _fd_from(fd.y0, fd.y1[::-1] * 0.4 - 2.0)
    for fit in (ST.weights_fw_mse, ST.weights_s1_direct):
        w0, w1, _ = fit(mem, fd)
        w0b, _, _ = fit(mem, fd_y0)
        _, w1c, _ = fit(mem, fd_y1)
        assert np.allclose(w0, w0b, atol=1e-12), fit.__name__  # fold-0 weights: fit on fold 1 only
        assert np.allclose(w1, w1c, atol=1e-12), fit.__name__  # fold-1 weights: fit on fold 0 only
    c4a, c4b = ST.cv4_hier(res, fd), ST.cv4_hier(res, fd_y0)
    assert c4a.members == c4b.members and np.allclose(c4a.w_fold0, c4b.w_fold0, atol=1e-12)
    # CV1 = exact H257 port: depends on the predicted fold only through its target MEAN (via b_k); a mean-preserving
    # permutation of y0 must leave its fold-0 weights unchanged.
    perm = _fd_from(fd.y0[rng.permutation(len(fd.y0))], fd.y1)
    assert np.allclose(ST.weights_h257(mem[:3], fd)[0], ST.weights_h257(mem[:3], perm)[0])


def test_frank_wolfe_reaches_known_optimum():
    rng = np.random.default_rng(2)
    y = rng.normal(size=200)
    P = np.column_stack([y + 0.1 * rng.normal(size=200), rng.normal(size=200), y + 0.5 * rng.normal(size=200)])
    f, g = ST.mse_objective([(P, y, 1.0)], np.full(3, 1 / 3), 0.0)
    w = ST.frank_wolfe(f, g, np.full(3, 1 / 3), iters=500)
    assert w[0] > 0.8 and w[1] < 0.05


def test_stack_audit_mean_substitution_never_reweights():
    fd = _fd()
    sp = ST.cv2_fw(_synthetic_results(fd), fd, 3)
    shas, _, w = sp.declared()
    ya = np.arange(10.0)
    ymean = 4.0
    ba = float(np.sqrt(np.mean((ya - ymean) ** 2)))
    pas = {s: ya + 0.5 * (k + 1) for k, s in enumerate(shas)}
    ok = {s: E.AuditResult("c", s, "", "completed", 0.1, 1.0, 1.0, pas[s]) for s in shas}
    r = ST.stack_audit_utility(sp, ok, ya, ba, ymean)
    assert r["status"] == "completed" and r["n_mean_substituted"] == 0
    bad = dict(ok)
    bad[shas[1]] = E.AuditResult("c", shas[1], "", "failed", -2.0, None, None, None)
    rb = ST.stack_audit_utility(sp, bad, ya, ba, ymean)
    wn = w / w.sum()
    pred = sum(wn[k] * (pas[s] if k != 1 else np.full(10, ymean)) for k, s in enumerate(shas))
    assert rb["n_mean_substituted"] == 1
    assert abs(rb["utility"] - C.audit_utility_from_rmse(float(np.sqrt(np.mean((pred - ya) ** 2))), ba)) < 1e-12


# ----------------------------------------------------------------------------------------------------------------------
# Evolution primitives
# ----------------------------------------------------------------------------------------------------------------------
def _res(dsl, lr0, lr1, comp=None, gate=True):
    s1 = ST.s1_from_lr(lr0, lr1, comp if comp is not None else dsl.count("("))
    return E.SearchResult("c", G.sha256_text(dsl), dsl, "completed", s1["utility"], s1["fitness"], gate, lr0, lr1,
                          s1["mean_lr"], s1["tail_lr"], s1["sd_lr"], comp or dsl.count("("), np.zeros(2), np.zeros(2))


def test_epsilon_lexicase():
    dom = _res("(pipe X (ridge_gcv))", -1.0, -1.0)
    others = [_res(f"(pipe X (ridge lambda_frac=0.{i + 1}))", -0.2, -0.3) for i in range(5)]
    rng = random.Random(0)
    assert all(D.epsilon_lexicase([dom] + others, rng) is dom for _ in range(50))
    s0 = _res("(pipe X (ksmooth k=8))", -1.0, -0.1)  # fold-0 specialist
    s1 = _res("(pipe X (ksmooth k=16))", -0.1, -1.0)  # fold-1 specialist
    mid = [_res(f"(pipe X (krr m={100 + i}))", -0.5, -0.5) for i in range(4)]
    picks = {D.epsilon_lexicase([s0, s1] + mid, rng).sha for _ in range(300)}
    assert s0.sha in picks and s1.sha in picks
    failed = E.SearchResult("c", "f" * 64, "(pipe X (ridge_gcv))", "failed", -2.0, 100.0, False, float("nan"),
                            float("nan"), float("nan"), float("nan"), float("nan"), 3)
    assert all(D.epsilon_lexicase([failed, dom], rng) is dom for _ in range(20))


def test_migration_splices_into_island3():
    arch = D.MapElites()
    champs = [_res("(pipe X (standardize) (histgbm depth=4 lr=0.08 rounds=140))", -1.0, -1.0),
              _res("(pipe X (cheb append=true degree=3) (ridge_gcv))", -0.9, -0.9),
              _res("(pipe X (blom) (gpr kernel=rbf+linear steps=40))", -0.8, -0.8)]
    for r in champs:
        arch.add(r)
    assert [arch.champion(i).sha for i in range(3)] == [r.sha for r in champs]

    class _G:
        class ctx:
            width = 10
    spl = D.migration_splices(arch, _G, 3)
    assert len(spl) == 4 and all(c.island == 3 for c in spl)
    full = G.parse(spl[0].dsl)
    assert isinstance(full, G.Portfolio) and [m.text() for m in full.members] == [r.dsl for r in champs]


# ----------------------------------------------------------------------------------------------------------------------
# End-to-end on NON-pilot contexts with the fake worker
# ----------------------------------------------------------------------------------------------------------------------
def _picks(rec04, rec05, rec06):
    out = {s: (m["order_shas"], m["best_sha"], m["bm_best_sha"]) for s, m in rec04["schemes"].items()}
    out["evo"] = {a: (v["result_shas"], [v[f"g{g}"]["best"]["sha"] for g in D.AUDIT_GENS]) for a, v in rec05["arms"].items()}
    out["stacks"] = {k: (v.get("spec") or {}).get("members") for k, v in rec06["configs"].items()}
    out["w"] = {k: (v.get("spec") or {}).get("w_audit") for k, v in rec06["configs"].items()}
    return out


def test_audit_isolation_poisoned_audit_changes_no_pick(nonpilot):
    cid = nonpilot[0]
    recs = []
    for tag, poison in (("clean", None), ("poison", lambda cand, pa, ctx: pa * (-50.0 if int(cand.sha[0], 16) % 2 else 0.0))):
        rt = make_runtime(f"iso_{tag}", fake_runner(poison=poison))
        L.stage_bank_nn1(rt, cid)
        r4 = L.stage_h04(rt, cid, 317101, False)
        r5 = L.stage_h05(rt, cid, 317101, "C")
        r6 = L.stage_h06(rt, cid, 317101, "C", "evoB")
        recs.append((r4, r5, r6))
        rt.ledger.check()
    a, b = _picks(*recs[0]), _picks(*recs[1])
    assert a == b
    vals_a = [recs[0][0]["schemes"][s]["audit_bm"]["value"] for s in D.SCHEMES]
    vals_b = [recs[1][0]["schemes"][s]["audit_bm"]["value"] for s in D.SCHEMES]
    assert vals_a != vals_b  # the poison did reach the AUDIT scores, and only them


def test_budget_equality_and_seed_budgets(nonpilot):
    cid = nonpilot[1]
    rt = make_runtime("budget", fake_runner())
    r4 = L.stage_h04(rt, cid, 317101, True)
    sch = r4["schemes"]
    assert sch["A"]["evals_used"] == 8 and sch["B"]["evals_used"] == 16 and sch["C"]["evals_used"] == 16
    assert sch["D"]["n_reveals"] == 4 and sch["D"]["evals_used"] == 24 and sch["E"]["evals_used"] == 16
    e_ops = [o["operator"] for o in sch["E"]["order"]]
    assert [op.startswith("registry:") for op in e_ops[:8]] == [True, False] * 4
    assert r4["noise_probe"]["n_extra_evals"] == 16
    r5 = L.stage_h05(rt, cid, 317101, "D")
    a, b = r5["arms"]["evoA"], r5["arms"]["evoB"]
    norm = lambda d: {str(k): v for k, v in d.items()}  # noqa: E731
    assert norm(a["live_per_gen"]) == norm(b["live_per_gen"]) == {str(g): 6 for g in range(1, 11)}
    assert norm(a["screened_per_gen"]) == norm(b["screened_per_gen"]) and set(a["screened_per_gen"].values()) == {48}
    assert sum(b["line_search_live"].values()) <= 18 and len(b["result_shas"]) == len(a["result_shas"]) == 24 + 60


def test_bank_disjoint_invariant(nonpilot, rows):
    rt = make_runtime("bank", fake_runner())
    fam = {r["context_sha256"]: r["family_id"] for r in rows}
    cid = next(c for c in nonpilot if any(b for b in rt.bank if fam[c] in b.families))
    tb = rt.target_bank(cid)
    own = [b for b in rt.bank if fam[cid] in b.families]
    assert own and all(b.sha not in tb.by_sha for b in own)
    before = D.BANK_COUNTERS["same_family_bank_hits"]
    with pytest.raises(D.BankLeakError):
        D.assert_bank_disjoint(cid, [D.bank_candidate(own[0])], tb)
    leak = G.Candidate.from_dsl("(pipe X (ridge_gcv))", "x", (), bank_cids=own[0].cids[0])
    with pytest.raises(D.BankLeakError):
        D.assert_bank_disjoint(cid, [leak], tb)
    assert D.BANK_COUNTERS["same_family_bank_hits"] == before + 2
    D.BANK_COUNTERS["same_family_bank_hits"] = before  # restore for the end-to-end counters


def test_no_pool_file_is_opened_by_a_gp_run(nonpilot):
    opened = []

    def hook(event, args):
        if event == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)):
            p = os.fsdecode(args[0])
            if "/pools/" in p:
                opened.append(p)

    sys.addaudithook(hook)
    rt = make_runtime("nopool", fake_runner())
    L.run_gate_h0(rt, nonpilot[2:4], scratch("nopool") / "run", 2, overrides={"h04_scheme": "C", "h05_arm": "evoB"})
    assert opened == []


def test_resumability_and_ledger(nonpilot):
    cids = nonpilot[4:6]
    s1 = L.run_gate_h0(make_runtime("resume", fake_runner()), cids, scratch("resume") / "run", 2)
    rt2 = make_runtime("resume", fake_runner())
    s2 = L.run_gate_h0(rt2, cids, scratch("resume") / "run", 2)
    assert rt2.engine.counters["physical_runs"] == 0 and rt2.ledger.check()["audit_requests"] == 0
    assert json.dumps(s1["H0.4"], sort_keys=True, default=str) == json.dumps(s2["H0.4"], sort_keys=True, default=str)
    assert s1["H0.6"]["adopted"] == s2["H0.6"]["adopted"]
    assert D.BANK_COUNTERS["same_family_bank_hits"] == 0


def test_stage_v_pipelines_and_matching(nonpilot):
    rt = make_runtime("stagev", fake_runner())
    out = L.stage_v_context(rt, nonpilot[6], 317101, L.LockedConfig(scheme="C", arm="evoB", generations=10, stack="CV5"))
    assert set(out) == set(L.STAGE_V_PIPELINES)
    assert out["H317_LOCKED"]["live_evals"] in (76, 77) and out["H257_REPLICA"]["live_evals"] == 8 + 15
    assert out["SCREEN_TOPK_PICK"]["live_evals"] == out["H317_LOCKED"]["live_evals"]
    assert out["BANK_NN1"]["live_evals"] == 0 and out["BANK_NN8_PICK"]["live_evals"] == 8
    rt.ledger.check()


def test_stage_v_refuses_dlval_before_unlock():
    dev = sorted(C._STATE["dev_cids"])[0]
    rt = make_runtime("stagev_dev", fake_runner())
    with pytest.raises(C.GuardViolation):
        L.stage_v_pipeline(rt, "BANK_NN1", dev, 317101, L.LockedConfig())


# ----------------------------------------------------------------------------------------------------------------------
# Optional: real worker on a NON-pilot context (2 programs, SEARCH + AUDIT)
# ----------------------------------------------------------------------------------------------------------------------
@pytest.mark.skipif(not C.WORKER_BIN.exists(), reason="worker binary missing")
def test_real_worker_parity(nonpilot, rows):
    fam = {r["context_sha256"]: r["family_id"] for r in rows}
    cid = nonpilot[7]
    core = int(os.environ.get("H317_TEST_CORE", "1"))
    eng = E.EvalEngine([core], root=scratch("real"), timeout_s=60)
    ctx = eng.context(cid, fam[cid])
    cs = [G.Candidate.from_dsl("(pipe X (ridge_gcv))", "t"),
          G.Candidate.from_dsl("(pipe X (standardize) (histgbm depth=4 lr=0.08 rounds=140))", "t")]
    rs = eng.search(ctx, cs)
    au = eng.audit(ctx, cs)
    for r, a in zip(rs, au):
        assert r.status == "completed" and len(r.p0) == len(ctx.y0) and len(r.p1) == len(ctx.y1)
        assert a.status == "completed" and len(a.pa) == len(ctx.ya) and abs(a.base_rmse - ctx.ba) < 1e-9 * max(1, ctx.ba)
        rm = float(np.sqrt(np.mean((a.pa - ctx.ya) ** 2)))
        assert abs(rm - a.model_rmse) < 1e-8 * max(1.0, rm)
    eng.close()


def test_stage_v_carries_no_state_across_contexts(nonpilot):
    """A context's Stage V outputs are identical whether or not another context ran first in the same runtime."""
    lock = L.LockedConfig(scheme="C", arm="evoB", generations=10, stack="CV3@5")
    cid, other = nonpilot[8], nonpilot[9]
    alone = L.stage_v_pipeline(make_runtime("nostate_a", fake_runner()), "H317_LOCKED", cid, 317101, lock)
    rt = make_runtime("nostate_b", fake_runner())
    L.stage_v_pipeline(rt, "H317_LOCKED", other, 317101, lock)
    after = L.stage_v_pipeline(rt, "H317_LOCKED", cid, 317101, lock)
    assert json.dumps(alone, sort_keys=True, default=str) == json.dumps(after, sort_keys=True, default=str)


def test_neural_adapter_passes_target_and_search_views_only(nonpilot, rows):
    """The adapter calls the leak-safe neural API with target_cid/target_family on every score; a target inside the
    model's training set raises; the view provider refuses the full-table view set."""
    fam = {r["context_sha256"]: r["family_id"] for r in rows}

    class Leak(RuntimeError):
        pass

    class FakeNeural:
        models = [type("M", (), {"set_blocks": None})()]

        def __init__(self, train):
            self.train, self.calls = set(train), []

        def score(self, vf, f412, p16, s24, target_cid=None, target_family=None):
            self.calls.append((target_cid, target_family))
            if target_cid is None or target_cid in self.train or target_family in {fam[c] for c in self.train}:
                raise Leak(target_cid)
            return np.tile(f412[:, 411][None, :], (len(vf), 1))

    kinds = []

    def view_fn(cid, kind):
        kinds.append(kind)
        return np.zeros((32, 997), np.float32)

    vp = S.SearchViewProvider(view_fn=view_fn)
    with pytest.raises(C.GuardViolation):
        vp(nonpilot[0], "full")
    ok = S.NeuralScreenerAdapter(FakeNeural([nonpilot[1]]), vp, family_of=fam)
    out = ok.score(nonpilot[0], [0, 5], ["(pipe X (ridge_gcv))", "(pipe X (standardize) (ridge_gcv))"])
    assert out.shape == (2, 2) and ok.model.calls[-1] == (nonpilot[0], fam[nonpilot[0]]) and kinds == ["search"]
    ok.score(nonpilot[0], [0, 5], ["(pipe X (ridge_gcv))"])  # cached column (candidate-isolated model)
    assert len(ok.model.calls) == 1
    bad = S.NeuralScreenerAdapter(FakeNeural([nonpilot[0]]), vp, family_of=fam)
    with pytest.raises(Leak):
        bad.score(nonpilot[0], [0], ["(pipe X (ridge_gcv))"])
