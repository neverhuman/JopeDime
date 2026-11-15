#!/usr/bin/env python3
"""H458 selftest (CPU only, 16 gates). Every gate must pass before any GPU use; the receipt goes to the H458 evidence folder.

Not covered here (stated, not hidden): injecting a real CUDA out-of-memory error, and the GP-residual target mode.
"""
from __future__ import annotations

import os

# one numeric thread, whatever the caller's environment says: the identity gates compare bytes, and threaded BLAS
# changes the last bits of a product with the shape of its operands
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_k] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse  # noqa: E402
import inspect  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from pathlib import Path  # noqa: E402

import paths_h458 as P  # noqa: E402

P.setup_paths()

import numpy as np  # noqa: E402
import torch  # noqa: E402

import cols_h458 as CB  # noqa: E402
import gp_chain_h458 as GC  # noqa: E402
import episodes_h458 as E  # noqa: E402
import family_pool_h458 as FP  # noqa: E402
import probe_h458 as PR  # noqa: E402
import train_h458 as TR  # noqa: E402

C = E.C
HERE = Path(__file__).resolve().parent
GP_ANCHOR = Path("/home/ubuntu/jope-data/h452-flywheel/gp_anchors_p0")
P418 = Path("/home/ubuntu/jope_plan/evidence/H418/pulled")
P448 = Path("/home/ubuntu/jope_plan/evidence/H448/pulled")
SOURCES = ("paths_h458.py", "family_pool_h458.py", "episodes_h458.py", "train_h458.py", "probe_h458.py", "cols_h458.py", "gp_chain_h458.py")


def gate_quarantine(ctx) -> dict:
    src = json.loads((P.MMAP_V1 / "meta.json").read_text())
    assert not any(t in c for c in src["cohorts"] for t in P.SEALED_TOKENS), "sealed cohort in bank"
    assert int(src.get("confirm_table_opens", 0)) == 0 and int(src.get("dl_holdout_opens", 0)) == 0
    pool = ctx["pool"]
    assert pool.meta["canonical_promotion_forbidden"] and pool.meta["in_corpus_dev91_disclosure"]
    with np.load(P.EVAL_BANK, allow_pickle=False) as zb:
        assert "ya" not in zb.files, "audit labels in search bank"
    return {"bank_tables": len(src["cohorts"]), "pool_families": len(pool.families)}


def gate_pool_dedup_union(ctx) -> dict:
    pool = ctx["pool"]
    src = json.loads((P.MMAP_V1 / "meta.json").read_text())
    mm = np.memmap(P.MMAP_V1 / src["data_file"], dtype=np.float64, mode="r")
    total = 0
    for i, fam in enumerate(pool.families):
        tab = np.asarray(pool.table(i))
        keys = FP.row_keys(tab)
        assert all(a < b for a, b in zip(keys, keys[1:])), f"{fam}: rows not in strictly increasing key order"
        want: set[bytes] = set()
        for t, (f, coh) in enumerate(zip(src["families"], src["cohorts"])):
            if f == fam and coh == FP.COHORT:
                r, c = (int(v) for v in src["shapes"][t])
                assert c == tab.shape[1], "sibling width mismatch"
                o = int(src["offsets"][t])
                want.update(FP.row_keys(np.asarray(mm[o : o + r * c]).reshape(r, c)))
        assert want == set(keys), f"{fam}: pool is not the union of its member tables"
        total += len(keys)
    assert total == pool.meta["total_rows"]
    return {"unique_rows": total}


def gate_pool_no_split_marker(ctx) -> dict:
    pool = ctx["pool"]
    assert tuple(pool.meta) == FP.META_KEYS
    assert sorted(p.name for p in pool.dir.iterdir()) == ["READY", "data.f64", "meta.json"]
    text = (HERE / "family_pool_h458.py").read_text()
    body = text.split('"""', 2)[2]
    assert "bank_d91_search" not in body and "refboard" not in body and "EVAL_BANK" not in body
    return {"meta_keys": len(FP.META_KEYS)}


def gate_eval_pipeline_identity(ctx) -> dict:
    """My episode wrapper equals the column builder's own fold prep, bit for bit, and query rows are independent."""
    checked = 0
    for cid in ctx["identity_cids"]:
        d = ctx["bank_full"][cid]
        xs, xa, y, lab = d["Xs"], d["Xa"], d["y"], d["lab"]
        prepped = C._prep_full_context_worker((0, cid, xs, xa, y, lab, GP_ANCHOR, P418, P448, None, 4, 2, 7))
        lab0 = C.make_stratified_kfold(cid, y, k=prepped["eff_k"], rep=0) if prepped["eff_k"] != 5 else lab
        for fk in (0, 3):
            (va, n_va, fseed, base_rr, y_tr_rr, views_rr, _wyj, z_yj, views_yj, _wrg, z_rg, views_rg, _gp) = prepped["prepped_folds"][fk]
            assert np.array_equal(va, lab0 == fk) and fseed == E.fold_seed(cid, 0, fk)
            tr = ~va
            ff = E.eval_fold_features(xs[tr], y[tr], xs[va], fseed)  # query = fold rows only, no audit rows appended
            mts = E.mode_targets(ff, y[tr], fseed)
            assert np.array_equal(mts["rr"].y_ctx, y_tr_rr) and np.array_equal(mts["yj"].y_ctx, z_yj) and np.array_equal(mts["rg"].y_ctx, z_rg)
            assert np.array_equal(mts["rr"].base_ev, np.asarray(base_rr)[:n_va])
            for name, ref_views in (("rr", views_rr), ("yj", views_yj), ("rg", views_rg)):
                mine = C._build_views(ff.X_tr, mts[name].y_ctx, ff.X_ev, n_views=4, seed=fseed + PR.SEED_OFFSET[name], keep_first=ff.n_pin)
                for (mt_, me_), (rt_, re_) in zip(mine, ref_views):
                    assert np.array_equal(mt_, rt_, equal_nan=True), f"{cid} f{fk} {name}: context features differ"
                    assert np.array_equal(me_, re_[:n_va], equal_nan=True), f"{cid} f{fk} {name}: query features depend on other query rows"
            checked += 1
    return {"folds_checked": checked}


def gate_zero_query_leakage(ctx) -> dict:
    assert "y_ev" not in inspect.signature(E.eval_fold_features).parameters
    assert "y_ev" not in inspect.signature(E.mode_targets).parameters
    pool = ctx["pool"]
    tab = np.asarray(pool.table("60b608db"))
    rng = np.random.default_rng(7)
    ids = rng.permutation(tab.shape[0])[:700]
    x_tr, y_tr, x_ev = tab[ids[:500], :-1], tab[ids[:500], -1], tab[ids[500:], :-1].copy()
    a = E.eval_fold_features(x_tr, y_tr, x_ev, 123)
    x_ev2 = x_ev.copy()
    x_ev2[0] = x_ev2[0] * 3.0 + 1e6
    b = E.eval_fold_features(x_tr, y_tr, x_ev2, 123)
    assert np.array_equal(a.X_tr, b.X_tr, equal_nan=True), "context features changed with a query row"
    assert np.array_equal(a.X_ev[1:], b.X_ev[1:], equal_nan=True), "a query row changed another query row"
    return {"rows": int(a.X_ev.shape[0])}


def gate_unique_rows_and_commitments(ctx) -> dict:
    pool = ctx["pool"]
    seen: set[bytes] = set()
    for fam in ("f8fd51f3", "9f67f144"):
        i = pool.resolve(fam)
        tab = pool.table(i)
        for s in range(150):
            rng = np.random.default_rng([i, s])
            c_ids, q_ids = E.draw_split(tab, pool.member_rows(i), rng)
            assert len(np.unique(c_ids)) == len(c_ids) and len(np.unique(q_ids)) == len(q_ids)
            assert not set(c_ids.tolist()) & set(q_ids.tolist()), "context and query overlap"
            k = E.split_commitment(i, c_ids, q_ids)
            assert k == E.split_commitment(i, c_ids[::-1], np.roll(q_ids, 3)), "commitment depends on row order"
            assert k not in seen, "two independent draws produced the same split"
            seen.add(k)
    c0, q0 = np.arange(10), np.arange(10, 20)
    assert E.split_commitment(1, c0, q0) != E.split_commitment(1, q0, c0) != E.split_commitment(2, q0, c0)
    return {"draws": len(seen)}


def gate_batch_caps(ctx) -> dict:
    rng = np.random.default_rng(3)
    n = 0
    for _ in range(4000):
        caps = {"tokens": int(rng.integers(20000, 400000)), "row_tokens": int(rng.integers(1500, 14000))}
        n_ctx, n_q, f = int(rng.integers(32, 1200)), int(rng.integers(32, 1025)), int(rng.integers(4, 65))
        b, q = E.fit_caps(n_ctx, n_q, f, caps)
        assert b in (0, 1, len(E.MODES))
        if b:
            assert q <= n_q and b * (n_ctx + q) * (f + 4) <= caps["tokens"] and b * (n_ctx + q) <= caps["row_tokens"]
            n += 1
    return {"fitting_shapes": n}


def gate_rank_gauss_roundtrip(ctx) -> dict:
    worst = 0.0
    for s in range(5):
        rng = np.random.default_rng(s)
        y = np.round(rng.standard_t(3, size=700) * 2.0, 1 if s % 2 else 6)  # with and without ties
        w = C.RankGaussDuanWarpF64()
        w.fit_transform(y)
        z = E.rank_gauss_forward(w, y)
        back = w.inv_torch(torch.from_numpy(z)).numpy()
        assert np.all(np.diff(z[np.argsort(y, kind="mergesort")]) >= -1e-12), "forward map not monotone"
        yu = np.unique(y)
        grp = np.array([back[y == v].mean() for v in yu])
        worst = max(worst, float(np.max(np.abs(grp - np.array([w.y_knots[np.argmin(np.abs(np.asarray(w.y_knots) - v))] for v in yu])))))
    assert worst < 1e-9, f"round trip error {worst}"
    return {"max_error": worst}


def gate_canonical_slots(ctx) -> dict:
    pool = ctx["pool"]
    tab = np.asarray(pool.table("60b608db"))
    out = []
    for s in (1, 2):
        rng = np.random.default_rng(s)
        ids = rng.permutation(tab.shape[0])[:900]
        ff = E.eval_fold_features(tab[ids[:700], :-1], tab[ids[:700], -1], tab[ids[700:], :-1], 50 + s)
        vt, _ve, slots = E.canonical_view(ff, tab[ids[:700], -1])
        assert len(np.unique(slots)) == len(slots) == vt.shape[1] and slots.max() < E.FID_SLOTS
        raw = slots[ff.n_pin : ff.n_pin + ff.n_raw]
        assert np.array_equal(raw, np.arange(ff.n_raw)), "raw column j must own slot j"
        out.append(int(ff.n_pin))
    return {"pinned_cols_in_two_splits": out}


def gate_feature_strata(ctx) -> dict:
    """Cross-fitting strata come from feature bytes only: label-free, roughly balanced, twins stay together."""
    pool = ctx["pool"]
    tab = np.asarray(pool.table("f2893c38"))
    k = FP.feature_stratum(tab[:, :-1])
    shuffled_y = tab.copy()
    shuffled_y[:, -1] = shuffled_y[::-1, -1]
    assert np.array_equal(k, FP.feature_stratum(shuffled_y[:, :-1])), "stratum depends on the label"
    share = np.bincount(k, minlength=5) / float(len(k))
    assert share.min() > 0.15 and share.max() < 0.25, share
    seen: dict[bytes, int] = {}
    for row, kk in zip(np.ascontiguousarray(tab[:, :-1]), k):
        assert seen.setdefault(row.tobytes(), int(kk)) == int(kk), "rows sharing X landed in different strata"
    left = pool.table_without_stratum("f2893c38", 2)
    assert left.shape[0] == int((k != 2).sum()) and not np.any(FP.feature_stratum(left[:, :-1]) == 2)
    return {"shares": [round(float(v), 3) for v in share], "twin_groups": len(tab) - len(seen)}


def gate_benchmark_split_audit(ctx) -> dict:
    """No simulated episode reproduces a benchmark SEARCH/AUDIT partition; a planted one is caught."""
    pool, worst = ctx["pool"], 0.0
    for fam in ("60b608db", "4286560c"):
        i = pool.resolve(fam)
        tab = np.asarray(pool.table(i))
        key_of = {k: r for r, k in enumerate(FP.row_keys(tab))}
        searches = []
        for cid, d in ctx["bank_full"].items():
            if d["family"][4:12] == fam:
                rows = [key_of[k] for k in FP.row_keys(np.column_stack([d["Xs"], d["y"]])) if k in key_of]
                assert len(rows) >= 0.99 * len(d["y"]), "search rows not found in the pool"
                searches.append(set(rows))

        def jacc(c_ids) -> float:
            cs = set(int(v) for v in c_ids)
            return max(len(cs & s) / float(len(cs | s)) for s in searches)

        for s in range(300):
            c_ids, _q = E.draw_split(tab, pool.member_rows(i), np.random.default_rng([i, 99, s]))
            worst = max(worst, jacc(c_ids))
        assert jacc(np.array(sorted(searches[0]))) == 1.0, "planted benchmark split not detected"
    assert worst < 0.9, f"a simulated context overlaps a benchmark search set at Jaccard {worst:.3f}"
    return {"max_jaccard_with_a_search_set": round(worst, 4)}


def gate_lr_schedule(ctx) -> dict:
    cfg = dict(TR.DEFAULTS)
    assert TR.lr_at(0, 0.0, cfg) < TR.lr_at(150, 0.0, cfg) < TR.lr_at(299, 0.0, cfg) == cfg["lr_peak"]
    assert TR.lr_at(5000, 0.59, cfg) == cfg["lr_peak"] > TR.lr_at(5000, 0.8, cfg) > TR.lr_at(5000, 0.99, cfg)
    assert abs(TR.lr_at(5000, 1.0, cfg) - cfg["lr_floor"]) < 1e-12
    return {"peak": cfg["lr_peak"], "floor": cfg["lr_floor"]}


def gate_cpu_mini_train(ctx) -> dict:
    out = P.QUARANTINE / "selftest" / f"mini_{int(time.time())}"
    cfg_p = P.QUARANTINE / "selftest" / f"mini_{int(time.time())}.json"
    cfg_p.parent.mkdir(parents=True, exist_ok=True)
    cfg_p.write_text(json.dumps({"arm": "canon", "size": "S", "families": ["f8fd51f3"], "save_at_exposures": [1], "caps": {"tokens": 40000, "row_tokens": 3000}, "warmup_steps": 2, "q_max": 256}))
    base = [sys.executable, "-B", str(HERE / "train_h458.py"), "--config", str(cfg_p), "--init", "random", "--out", str(out), "--device", "cpu", "--workers", "2", "--log-every-s", "2"]
    r1 = subprocess.run(base + ["--max-items", "4"], capture_output=True, text=True, timeout=900)
    assert r1.returncode == 0, r1.stderr[-600:]
    r2 = subprocess.run(base + ["--max-items", "6", "--resume"], capture_output=True, text=True, timeout=900)
    assert r2.returncode == 0, r2.stderr[-600:]
    recs = [json.loads(line) for line in (out / "log.jsonl").read_text().splitlines()]
    last = [r for r in recs if r.get("event") == "paused"][-1]
    # six items give at least one optimizer step (a step needs three episodes; small caps mean one episode per item)
    assert last["step"] >= 1 and last["items"] == 6 and last["dup_splits"] == 0, last
    spec = out / "spec_e00001.pt"
    assert spec.is_file(), "milestone checkpoint missing"
    ck = torch.load(spec, map_location="cpu", weights_only=False)
    assert ck["canonical_promotion_forbidden"] and all(isinstance(v, torch.Tensor) for v in ck["ema"].values())
    json.dumps(ck["state"])  # plain types only
    m = C.load_icl_ckpt(spec, "S", torch.device("cpu"), ckpt_key="ema")
    assert sum(p.numel() for p in m.parameters()) > 0
    first = [r for r in recs if r.get("event") == "paused"][0]
    assert first["items"] == 4 and last["step"] >= first["step"], "resume did not continue the run"
    # injected out-of-memory failures (every batch above 700 rows) in a run of its own: the item is trimmed and retried,
    # the run keeps stepping, the caps shrink at most once per item and nothing aborts
    out_f = out.with_name(out.name + "_fault")
    cfg_f = cfg_p.with_name(cfg_p.stem + "_fault.json")
    cfg_f.write_text(json.dumps({**json.loads(cfg_p.read_text()), "fault_inject_oom_rows": 700}))
    cmd_f = [a if a != str(cfg_p) else str(cfg_f) for a in base]
    cmd_f[cmd_f.index(str(out))] = str(out_f)
    r3 = subprocess.run(cmd_f + ["--max-items", "5"], capture_output=True, text=True, timeout=900)
    assert r3.returncode == 0, r3.stderr[-600:]
    recs_f = [json.loads(line) for line in (out_f / "log.jsonl").read_text().splitlines()]
    ooms = [r for r in recs_f if r.get("event") == "oom"]
    end_f = [r for r in recs_f if r.get("event") == "paused"][-1]
    assert ooms and not (out_f / "ABORTED.json").exists(), "injected OOMs were not absorbed"
    assert end_f["step"] >= 1, "no optimizer step under injected OOMs"
    assert min(r["cap_scale"] for r in ooms) >= 0.85**5 - 1e-3, "caps shrank more than once per item"
    return {"steps": last["step"], "fault_run": {"injected_ooms": len(ooms), "steps": end_f["step"], "cap_scale_end": min(r["cap_scale"] for r in ooms)}, "run_dir": str(out)}


def gate_remote_workers(ctx) -> dict:
    """The remote-worker stream (framing, round-robin order, control lines, shutdown), run on this host without ssh."""
    out = P.QUARANTINE / "selftest" / f"remote_{int(time.time())}"
    cfg_p = P.QUARANTINE / "selftest" / f"remote_{int(time.time())}.json"
    cfg_p.parent.mkdir(parents=True, exist_ok=True)
    cfg_p.write_text(json.dumps({"arm": "rand", "size": "S", "families": ["f8fd51f3"], "save_at_exposures": [1], "caps": {"tokens": 40000, "row_tokens": 3000}, "warmup_steps": 2, "q_max": 256}))
    cmd = [sys.executable, "-B", str(HERE / "train_h458.py"), "--config", str(cfg_p), "--init", "random", "--out", str(out), "--device", "cpu", "--workers", "1", "--remote-workers", "local:2", "--max-items", "6", "--log-every-s", "2"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stderr[-600:]
    recs = [json.loads(line) for line in (out / "log.jsonl").read_text().splitlines()]
    end = [x for x in recs if x.get("event") == "paused"][-1]
    assert end["items"] == 6 and end["step"] >= 1 and end["dup_splits"] == 0, end
    seq = torch.load(out / "ckpt_last.pt", map_location="cpu", weights_only=False)["state"]["seq"]
    assert len(seq) == 3 and all(v >= 1 for v in seq), f"not every worker delivered: {seq}"
    err = (out / "remote_local.stderr").read_text()
    assert "Traceback" not in err, err[-400:]
    return {"items": end["items"], "steps": end["step"], "seq": seq}


def gate_builder_matches_reference(ctx) -> dict:
    """The scored builder's canonical-slot path equals an independent fold loop built from the training functions."""
    cid = ctx["identity_cids"][0]
    d = ctx["bank_full"][cid]
    xs, xa, y, lab = d["Xs"], d["Xa"], d["y"], d["lab"]
    dev = torch.device("cpu")
    torch.manual_seed(458)
    model = C.ICLH450("S").eval()
    reps = 1  # the identity does not depend on the number of repeats; scored blocks use two
    out = CB.build_context(cid, xs, xa, y, lab, [[(model, 1.0)]], dev, "canon", n_repeats=reps)[0]
    assert C._build_views is not CB._views_canon and C._batched_icl_predict is not CB._predict_canon, "builder overrides were not restored"
    n_s, n_a = len(y), xa.shape[0]
    span = max(float(np.max(y) - np.min(y)), max(float(np.std(y)), 1e-12))
    lo_g, hi_g = float(np.min(y)) - 1.5 * span, float(np.max(y)) + 1.5 * span
    eff_k = 7 if n_s >= 35 else 5
    q_sum = {m: np.zeros(n_s) for m in E.MODES}
    q_cnt = {m: np.zeros(n_s) for m in E.MODES}
    bag = {m: np.zeros(n_a) for m in E.MODES}
    ref, folds = {}, 0

    def score(x_tr, y_tr, x_ev, seed) -> dict:
        ff = E.eval_fold_features(x_tr, y_tr, x_ev, seed)
        res = {}
        for m, mt in E.mode_targets(ff, y_tr, seed).items():
            vt, ve, slots = E.canonical_view(ff, mt.y_ctx)
            pr = PR.predict_canon(model, vt, ve, mt.y_ctx, slots, dev, None if mt.warp is None else mt.warp.inv_torch)
            res[m] = np.clip(mt.base_ev + pr, lo_g, hi_g)
        return res

    for rep in range(reps):
        lab_rep = C.make_stratified_kfold(cid, y, k=eff_k, rep=rep) if eff_k != 5 or rep > 0 else lab
        for fk in range(eff_k):
            va = lab_rep == fk
            n_va = int(va.sum())
            pred = score(xs[~va], y[~va], np.vstack([xs[va], xa]), E.fold_seed(cid, rep, fk))
            for m in E.MODES:
                q_sum[m][va] += pred[m][:n_va]
                q_cnt[m][va] += 1
                bag[m] += pred[m][n_va:]
            folds += 1
    ref = score(xs, y, xa, C._sha_seed(f"h452_cv7:{cid}:refit"))
    worst = 0.0
    for m, col in zip(E.MODES, CB.COLUMNS):
        for name, mine in (("q", q_sum[m] / q_cnt[m]), ("bag", bag[m] / folds), ("ref", ref[m])):
            worst = max(worst, float(np.max(np.abs(out[f"{name}_{col}"] - mine))))
    assert worst < 1e-9, f"builder and reference differ by {worst}"
    assert np.array_equal(out["ref_" + CB.UNUSED_COLUMN], out["ref_" + CB.COLUMNS[0]]), "GP anchor was not switched off"
    # cross-fit routing: a row's value comes from the checkpoint that was trained without the row's stratum
    s_s, s_a = FP.feature_stratum(xs, CB.XFIT_K), FP.feature_stratum(xa, CB.XFIT_K)
    fake = [{"ok": np.array(True), "q_c": np.full(n_s, float(k)), "bag_c": np.full(n_a, float(k)), "ref_c": np.full(n_a, 10.0 + k)} for k in range(CB.XFIT_K)]
    merged = CB.crossfit_merge(fake, s_s, s_a)
    assert np.array_equal(merged["q_c"], s_s) and np.array_equal(merged["bag_c"], s_a) and np.array_equal(merged["ref_c"], 10.0 + s_a), "cross-fit routing is wrong"
    # the file a block is delivered in: three columns, whitelisted keys, right shapes
    blk = P.QUARANTINE / "selftest" / f"block_{int(time.time())}.npz"
    blk.parent.mkdir(parents=True, exist_ok=True)
    np.savez(blk, **CB.pack_block(out, 1.0))
    CB.verify_block_file(blk, n_s, n_a)
    with np.load(blk) as z:
        assert sorted(z.files) == sorted(["ok", "wall_s"] + [pre + c for c in CB.COLUMNS for pre in ("q_", "bag_", "ref_")])
    # closure "recall" is refused unless the pool holds every row of the context
    fam8 = d["family"][4:12]
    cov = CB.require_pool_coverage(ctx["pool"], fam8, cid, xs, xa)
    assert cov == [1.0, 1.0], cov
    moved = xa.copy()
    moved[0, 0] = np.nextafter(xa[0, 0], np.inf) if np.isfinite(xa[0, 0]) else 0.0  # one audit row that is not in the pool
    try:
        CB.require_pool_coverage(ctx["pool"], fam8, cid, xs, moved)
        raise AssertionError("a context with a fresh audit row was accepted for closure recall")
    except SystemExit:
        pass
    return {"max_abs_diff": worst, "fits": folds + 1, "strata_search": np.bincount(s_s, minlength=CB.XFIT_K).tolist(), "pool_coverage": cov}


def _refused(fn, *args, **kwargs) -> bool:
    try:
        fn(*args, **kwargs)
    except (SystemExit, ValueError):
        return True
    return False


def gate_block_closures(ctx) -> dict:
    """What a block directory may be labelled as: the fresh-row probe, the fair allowlist, sidecar bookkeeping, slot widths."""
    cid = ctx["identity_cids"][0]
    d = ctx["bank_full"][cid]
    xs, xa, fam8, pool = d["Xs"], d["Xa"], d["family"][4:12], ctx["pool"]
    root = P.QUARANTINE / "selftest" / f"closures_{int(time.time())}_{os.getpid()}"
    root.mkdir(parents=True, exist_ok=False)
    # fresh-probe: every SEARCH row in the pool, no probe row in the pool
    fresh = xa.copy()
    col = int(np.argmax(np.isfinite(xa).all(axis=0)))
    fresh[:, col] = np.nextafter(xa[:, col], np.inf)
    assert CB.require_probe_rows_fresh(pool, fam8, cid, xs, fresh) == [1.0, 0.0]
    assert _refused(CB.require_probe_rows_fresh, pool, fam8, cid, xs, xa), "audit rows of the board were accepted as fresh rows"
    one_known = fresh.copy()
    one_known[0] = xa[0]
    assert _refused(CB.require_probe_rows_fresh, pool, fam8, cid, xs, one_known), "a probe set holding one pool row was accepted"
    assert _refused(CB.require_probe_rows_fresh, pool, fam8, cid, fresh[: len(xs)] if len(fresh) >= len(xs) else fresh, fresh), "SEARCH rows outside the pool were accepted"
    assert _refused(CB.require_probe_rows_fresh, pool, fam8, cid, xs, fresh[:, :-1]), "a probe set of the wrong width was accepted"
    # a bank is a probe bank exactly when the board's marker sits beside it and lists its sha
    bank_dir = root / "board"
    bank_dir.mkdir()
    bank_p = bank_dir / "bank.npz"
    np.savez(bank_p, x=np.arange(3))
    assert CB.probe_board_of(bank_p) is None
    (bank_dir / CB.PROBE_MARK).write_text(json.dumps({"name": "selftest"}))
    assert _refused(CB.probe_board_of, bank_p), "a probe bank outside the board's SHA256SUMS was accepted"
    (bank_dir / "SHA256SUMS").write_text(f"{P.digest_file(bank_p)}  bank.npz\n")
    assert CB.probe_board_of(bank_p)["probe_bank_sha256"] == P.digest_file(bank_p)
    assert CB.probe_board_of(P.EVAL_BANK) is None, "the evaluation bank reads as a probe bank"
    moved = root / "moved_bank.npz"
    np.savez(moved, x=np.arange(4))
    CB.KNOWN_PROBE_BANKS[P.digest_file(moved)] = "selftest-board"
    try:
        assert _refused(CB.probe_board_of, moved), "a probe bank moved away from its board was accepted as an evaluation bank"
    finally:
        CB.KNOWN_PROBE_BANKS.pop(P.digest_file(moved))
    assert CB.probe_board_of(moved) is None
    # checkpoints: fair needs the allowlist and no in-corpus marker; role init is a plain checkpoint scored with random slots
    plain, marked, spec = root / "plain.pt", root / "marked.pt", root / "spec.pt"
    torch.save({"ema": {}}, plain)
    torch.save({"ema": {}, "in_corpus_dev91_disclosure": True}, marked)
    torch.save({"ema": {}, "cfg": {"families": [fam8], "arm": "canon"}, "state": {"pool_sha256": pool.sha256}}, spec)
    assert P.digest_file(plain) not in CB.fair_allowlist()
    assert _refused(CB.check_checkpoint, plain, "ema", "fair", "rand", None, pool.sha256), "a checkpoint off the allowlist was accepted as fair"
    assert _refused(CB.check_checkpoint, marked, "ema", "fair", "rand", None, pool.sha256), "an in-corpus checkpoint was accepted as fair"
    assert _refused(CB.check_checkpoint, spec, "ema", "fair", "canon", None, pool.sha256), "a specialist was accepted as fair"
    assert CB.check_checkpoint(plain, "ema", "fresh-probe", "rand", None, pool.sha256, "init") == {}
    assert _refused(CB.check_checkpoint, plain, "ema", "fresh-probe", "canon", None, pool.sha256, "init"), "an init was accepted with canonical slots"
    assert _refused(CB.check_checkpoint, spec, "ema", "fresh-probe", "canon", None, pool.sha256, "init"), "a specialist was accepted as its own init"
    assert CB.check_checkpoint(spec, "ema", "fresh-probe", "canon", None, pool.sha256, "specialist")["families"] == [fam8]
    assert _refused(CB.check_checkpoint, spec, "ema", "fresh-probe", "rand", None, pool.sha256, "specialist"), "a canon specialist was accepted with random slots"
    assert _refused(CB.check_checkpoint, plain, "ema", "recall", "canon", None, pool.sha256), "a plain checkpoint was accepted as a recall specialist"
    # sidecar bookkeeping: one entry per run, files and sidecar agree, a second closure in the same directory is refused
    blk = root / "block"
    blk.mkdir()
    CB.require_dir_consistent(blk)
    np.savez(blk / "aaa.npz", ok=np.array(True))
    assert _refused(CB.require_dir_consistent, blk), "a block file that no sidecar lists was accepted"
    fixed = {"closure": "recall", "cross_fitted": False, "arm": "canon", "size": "M", "ckpt_key": "ema", "pool_sha256": pool.sha256}
    call = {"call_id": "t-1", "utc": "u1", "ckpts": ["c"], "ckpt_sha256": ["s"], "elapsed_s": 1.0}
    CB.update_sidecar(blk, fixed, {**call, "cids": ["aaa"]}, {"aaa": [1.0, 1.0]})
    CB.require_dir_consistent(blk)
    np.savez(blk / "bbb.npz", ok=np.array(True))
    doc = CB.update_sidecar(blk, fixed, {**call, "cids": ["bbb"], "utc": "u2", "elapsed_s": 2.0}, {"bbb": [1.0, 1.0]})
    assert len(doc["calls"]) == 1 and doc["calls"][0]["cids"] == ["aaa", "bbb"] and doc["contexts"] == ["aaa", "bbb"], doc["calls"]
    np.savez(blk / "ccc.npz", ok=np.array(True))
    doc = CB.update_sidecar(blk, fixed, {**call, "call_id": "t-2", "cids": ["ccc"]}, {})
    assert len(doc["calls"]) == 2 and "probe_role" not in doc
    CB.require_dir_consistent(blk)
    assert _refused(CB.update_sidecar, blk, {**fixed, "closure": "fresh-probe", "probe_role": "init"}, {**call, "cids": ["ddd"]}, {}), "two closures in one directory"
    # probe comparison block: sources renamed per tag, their mean added, blocks of two probe banks never joined
    fx = {"closure": "fresh-probe", "cross_fitted": False, "arm": "rand", "size": "M", "ckpt_key": "ema", "pool_sha256": pool.sha256, "bank_sha256": "b1", "probe_bank_sha256": "b1", "probe_board_sha256": "p1"}
    rng = np.random.default_rng(458)
    srcs = {}
    for tag, role, bank_sha in (("init", "init", "b1"), ("canon", "specialist", "b1"), ("other", "specialist", "b2")):
        srcs[tag] = root / f"probe_{tag}"
        srcs[tag].mkdir()
        for c in ("c1", "c2") if tag != "canon" else ("c1", "c2", "c3"):
            np.savez(srcs[tag] / f"{c}.npz", ok=np.array(True), wall_s=np.array(1.0), **{pre + n: rng.normal(size=4 if pre == "q_" else 6) for n in CB.COLUMNS for pre in ("q_", "bag_", "ref_")})
            CB.update_sidecar(srcs[tag], {**fx, "probe_role": role, "bank_sha256": bank_sha, "probe_bank_sha256": bank_sha}, {**call, "call_id": tag, "cids": [c]}, {})
    mdir = root / "probe_merged"
    mdir.mkdir()
    mdoc = CB.merge_probe_blocks({"init": srcs["init"], "canon": srcs["canon"]}, mdir, [])
    assert mdoc["contexts"] == ["c1", "c2"] and mdoc["closure"] == "fresh-probe" and mdoc["bank_sha256"] == "b1" and len(mdoc["names"]) == 8, mdoc["names"]
    assert mdoc["dropped_contexts"] == {"init": [], "canon": ["c3"]}, mdoc["dropped_contexts"]
    with np.load(mdir / "c1.npz") as zm, np.load(srcs["canon"] / "c1.npz") as zs:
        assert np.array_equal(zm["ref_canon_yj"], zs["ref_" + CB.COLUMNS[1]])
        assert np.allclose(zm["bag_canon_mean3"], np.mean([zs["bag_" + n] for n in CB.COLUMNS], axis=0), rtol=0, atol=1e-15)
        assert sorted(k for k in zm.files if k.startswith("q_")) == sorted(f"q_{t}_{m}" for t in ("init", "canon") for m in ("rr", "yj", "rg", "mean3"))
    CB.require_dir_consistent(mdir)
    assert _refused(CB.merge_probe_blocks, {"init": srcs["init"], "canon": srcs["canon"]}, mdir, []), "a merge into a filled directory was accepted"
    m2 = root / "probe_merged_2"
    m2.mkdir()
    assert _refused(CB.merge_probe_blocks, {"init": srcs["init"], "other": srcs["other"]}, m2, []), "blocks of two probe banks were merged"
    assert _refused(CB.merge_probe_blocks, {"init": srcs["init"], "canon": srcs["canon"]}, m2, ["c3"]), "a context missing from one source was merged"
    assert _refused(CB.merge_probe_blocks, {"init": srcs["init"], "rec": blk}, m2, []), "a recall block was merged into a probe comparison"
    # canonical slots: the cleaning step must keep every raw column, and the builder module is restored afterwards
    before = C.REL.build_h450_struct_cols
    with CB.builder_overrides("canon", xs.shape[1]):
        assert C.REL.build_h450_struct_cols is not before
        narrow = CB._struct_cols_same_width(lambda a_, y_, q_, **k_: (a_[:, :-1], q_[:, :-1], None, None, None))
        assert _refused(narrow, xs[:8], d["y"][:8], xa[:4]), "a cleaning step that drops a column was accepted"
        same = CB._struct_cols_same_width(lambda a_, y_, q_, **k_: (a_, q_, None, None, None))
        assert same(xs[:8], d["y"][:8], xa[:4])[0].shape[1] == xs.shape[1]
    assert C.REL.build_h450_struct_cols is before, "the struct-column function was not restored"
    return {"checks": 39, "dir": str(root)}


class _LookupEngine:
    """Stands in for the GP engine: its "program" knows half the target of every row it was fitted on and nothing else."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    @staticmethod
    def make_stratified_kfold(key, y, k=5, rep=0):
        return np.arange(len(y), dtype=np.int64) % int(k)

    def evolve_single_context_1000g(self, *, cid, Xs, Xa, ys, out_root, n_gens, **_kw):
        if (Path(out_root) / f"g{int(n_gens)}" / f"{cid}.npz").is_file():
            return {"cid": cid, "skipped_existing": True}  # what the real engine does when a stage's snapshot exists
        table = {Xs[i].tobytes(): float(ys[i]) for i in range(len(ys))}
        pa = np.array([0.5 * table.get(Xa[i].tobytes(), 0.0) for i in range(len(Xa))])
        snap = Path(out_root) / f"g{int(n_gens)}"
        snap.mkdir(parents=True, exist_ok=True)
        np.savez(snap / f"{cid}.npz", Q_cons=(0.5 * ys)[:, None], AKF_cons=pa[:, None], PA_cons=pa[:, None], u_cons=np.array([0.1]),
                 dsls=np.array(["stub"]), dsls_cons=np.array(["stub"]))
        self.calls.append({"cid": cid, "rows": len(ys), "gens": int(n_gens)})
        return {"total_unique_evaluated": 3 * int(n_gens), "worker_calls": 1, "timeouts": 0}


def gate_gp_chain_units(ctx) -> dict:
    """The staged-GP driver's arithmetic and its row isolation, with a stub in place of the engine (no GP is run)."""
    assert GC.split_generations(160, 4) == [40, 40, 40, 40] and GC.split_generations(10, 3) == [4, 3, 3]
    assert _refused(GC.split_generations, 2, 3)
    rng = np.random.default_rng(9)
    y = rng.normal(size=60)
    c1, c2 = y + rng.normal(size=60), y + rng.normal(size=60)
    snap = {"Q_cons": np.column_stack([c1, c2]), "AKF_cons": np.ones((7, 2)), "PA_cons": np.column_stack([np.ones(7), 3 * np.ones(7)]), "u_cons": np.array([0.2, 0.1]),
            "dsls": np.array(["a", "b"])}
    one, w1, _ = GC.stage_column(snap, y, 1)
    assert w1 == [1.0] and np.array_equal(one["q"], c1)
    two, w2, names = GC.stage_column(snap, y, 3)
    assert names == ["a", "b"] and min(w2) >= 0.0 and sum(w2) <= 1.0 + 1e-12 and GC.rmse(two["q"], y) < GC.rmse(c1, y)
    assert np.allclose(two["ref"], w2[0] * 1.0 + w2[1] * 3.0)
    big, wb, _ = GC.stage_column({**snap, "Q_cons": np.column_stack([0.25 * y, 0.25 * y])}, y, 2)
    assert sum(wb) <= 1.0 + 1e-12, "a stage blend inflated its programs"
    fb, _, nm = GC.stage_column({**snap, "dsls": np.array(["mean_fallback"])}, y, 2)
    assert nm == ["mean_fallback"] and np.allclose(fb["q"], y.mean())
    # chains and outer folds on a table whose audit slot repeats ten SEARCH rows
    xs = rng.normal(size=(48, 3))
    yy = rng.normal(size=48) + 2.0
    d = {"Xs": xs, "Xa": np.vstack([xs[:10], rng.normal(size=(5, 3))]), "y": yy, "family": "fam-stubstubstub"}
    work = P.QUARANTINE / "selftest" / f"gpchain_{int(time.time())}_{os.getpid()}"

    class A:  # the argparse fields run_context reads
        gens, stages, outer, nu, committee, min_gain, seed, inner_slots = 12, 3, 4, 1.0, 1, 0.0, 1, 1
        base_block, base_col = "", ""

    eng = _LookupEngine()
    A.arm = "chain"
    res = GC.run_context((eng, None, None), "ctxa", d, A, work / "chain")
    for j, st in enumerate(res["stages"], 1):
        assert np.allclose(st["q"], (1.0 - 0.5**j) * yy), "the chain does not fit the residual of its running sum"
        assert np.allclose(st["ref"][:10], (1.0 - 0.5**j) * yy[:10]) and np.allclose(st["ref"][10:], 0.0)
    assert [c["gens"] for c in eng.calls] == [4, 4, 4] and res["book"]["generations_total"] == 12 and not res["book"]["honest_search_column"]
    eng = _LookupEngine()
    A.arm, A.stages = "oof_chain", 2
    res = GC.run_context((eng, None, None), "ctxb", d, A, work / "oof")
    lab = res["outer_lab"]
    assert sorted(set(lab.tolist())) == [0, 1, 2, 3] and res["book"]["honest_search_column"]
    for st in res["stages"]:
        assert np.allclose(st["q"], 0.0), "a held-out SEARCH row reached a fit"
    last = res["stages"][-1]["ref"]
    assert np.allclose(last[:10], 0.75 * 0.75 * yy[:10]) and np.allclose(last[10:], 0.0), "fold-models are not averaged on the audit slot"
    assert len(eng.calls) == 8 and all(c["rows"] == 36 for c in eng.calls) and sum(c["gens"] for c in eng.calls) == 12
    # an interrupted run resumes: finished stages are reused with the cost they recorded, never with zero cost or another setting's stages
    eng = _LookupEngine()
    A.arm, A.stages = "chain", 3
    res_r = GC.run_context((eng, None, None), "ctxa", d, A, work / "chain")
    assert not eng.calls and res_r["book"]["evaluated_total"] == 36 and all(r["skipped_existing"] for p_ in res_r["book"]["pipelines"] for r in p_)
    assert np.allclose(res_r["stages"][-1]["q"], res["stages"][-1]["q"] if len(res["stages"]) == 3 else res_r["stages"][-1]["q"])
    orphan = work / "orphan" / "snap" / "g4"
    orphan.mkdir(parents=True)
    np.savez(orphan / "ctxz.s1.npz", Q_cons=np.zeros((48, 1)), AKF_cons=np.zeros((15, 1)), PA_cons=np.zeros((15, 1)), u_cons=np.zeros(1), dsls=np.array(["x"]))
    assert _refused(GC.run_stage, (_LookupEngine(), None, None), "ctxz.s1", "fam", xs, d["Xa"], yy, work / "orphan", 4, 1, 1), "a snapshot without its stage record was reused"
    fold = work / "claimed"
    fold.mkdir()
    GC.claim_folder(fold, {"arm": "single", "seed": 1})
    GC.claim_folder(fold, {"arm": "single", "seed": 1})
    assert _refused(GC.claim_folder, fold, {"arm": "single", "seed": 2}), "a second seed was accepted into the first seed's folder"
    assert _refused(GC.claim_folder, work / "chain", {"arm": "chain"}), "a folder with files and no settings was claimed"
    # a chain started from an existing out-of-fold column fits that column's residual; held-out rows keep the base's own value
    bdir = work / "base_block"
    bdir.mkdir(parents=True)
    for cid_b in ("ctxc", "ctxd"):
        np.savez(bdir / f"{cid_b}.npz", q_stack=0.5 * yy, bag_stack=np.ones(15), ref_stack=np.ones(15))
    A.base_block, A.base_col = str(bdir), "stack"
    eng = _LookupEngine()
    A.arm, A.stages = "chain", 2
    res_b = GC.run_context((eng, None, None), "ctxc", d, A, work / "base_chain")
    assert np.allclose(res_b["stages"][0]["q"], 0.75 * yy) and np.allclose(res_b["stages"][1]["q"], 0.875 * yy), "the first stage did not fit the base column's residual"
    assert np.allclose(res_b["stages"][0]["ref"][:10], 1.0 + 0.25 * yy[:10]) and np.allclose(res_b["stages"][0]["ref"][10:], 1.0)
    assert abs(res_b["book"]["u_search_base"] - GC.u_search(0.5 * yy, yy)) < 1e-12
    A.arm = "oof_chain"
    res_o = GC.run_context((_LookupEngine(), None, None), "ctxd", d, A, work / "base_oof")
    assert np.allclose(res_o["stages"][-1]["q"], 0.5 * yy), "a held-out row got more than the base column's out-of-fold value"
    A.base_block, A.base_col = "", ""
    # packing for a column scorer: names, shapes, nothing but the block layout
    arm_dirs = {}
    for tag, st in (("single", res["stages"][:1]), ("chain", res["stages"])):
        ad = work / f"arm_{tag}"
        ad.mkdir(parents=True)
        np.savez(ad / "ctxb.npz", outer_lab=lab, **{f"{k}_s{j}": v for j, s_ in enumerate(st, 1) for k, v in s_.items()})
        (ad / "ctxb.json").write_text(json.dumps({"wall_s": 1.0, "settings": {"bank_sha256": "b", "bank": str(P.EVAL_BANK)}}))
        arm_dirs[tag] = ad
    pk = work / "packed"
    pk.mkdir()
    doc = GC.pack_block(arm_dirs, pk)
    assert doc["names"] == ["single", "chain_s1", "chain_s2", "chain"] and doc["contexts"] == ["ctxb"]
    assert doc["closure"] == "in-corpus-prior" and doc["prior_in_corpus"] and doc["priors"] == ["in-corpus"], "arms without a recorded prior ran under the engine default"
    pkf = work / "packed_fair_refused"
    pkf.mkdir()
    assert _refused(GC.pack_block, arm_dirs, pkf, 0, "fair"), "in-corpus-prior arms were packed as fair"
    fair_dirs = {}
    for tag, ad in arm_dirs.items():
        fd = work / f"arm_fair_{tag}"
        fd.mkdir()
        with np.load(ad / "ctxb.npz") as zz:
            np.savez(fd / "ctxb.npz", **{k: zz[k] for k in zz.files})
        (fd / "ctxb.json").write_text(json.dumps({"wall_s": 1.0, "settings": {"bank_sha256": "b", "bank": str(P.EVAL_BANK), "prior": "dev91-free"}}))
        fair_dirs[tag] = fd
    pkg = work / "packed_fair"
    pkg.mkdir()
    dfair = GC.pack_block(fair_dirs, pkg, 0, "fair")
    assert dfair["closure"] == "fair" and not dfair["prior_in_corpus"] and dfair["priors"] == ["dev91-free"]
    with np.load(pk / "ctxb.npz") as z:
        assert sorted(k for k in z.files if k not in ("ok", "wall_s")) == sorted(f"{p_}_{n}" for p_ in ("q", "bag", "ref") for n in doc["names"])
        assert np.array_equal(z["ref_chain"], res["stages"][-1]["ref"]) and np.array_equal(z["q_chain_s1"], res["stages"][0]["q"])
    assert _refused(GC.pack_block, arm_dirs, pk), "a pack into a filled directory was accepted"
    # an audit slot made of two parts (the bank's own rows, then an appended bank's): the pack step takes one part
    two = work / "arm_two"
    two.mkdir()
    np.savez(two / "ctxe.npz", outer_lab=lab, q_s1=np.zeros(48), bag_s1=np.arange(15.0), ref_s1=np.arange(15.0) + 100.0)
    (two / "ctxe.json").write_text(json.dumps({"wall_s": 1.0, "n_audit_slot": 15, "audit_slot_parts": [10, 5],
                                               "settings": {"bank_sha256": "b", "bank": str(P.EVAL_BANK), "extra_audit_bank_sha256": "x", "extra_audit_bank": str(P.EVAL_BANK)}}))
    for part, lo, hi in ((0, 0, 10), (1, 10, 15)):
        pk2 = work / f"packed_part{part}"
        pk2.mkdir()
        doc2 = GC.pack_block({"two": two}, pk2, part)
        with np.load(pk2 / "ctxe.npz") as z:
            assert np.array_equal(z["ref_two"], np.arange(15.0)[lo:hi] + 100.0) and np.array_equal(z["bag_two"], np.arange(15.0)[lo:hi]) and z["q_two"].shape == (48,)
        assert doc2["bank_sha256"] == ("b" if part == 0 else "x")
    pk3 = work / "packed_part1_missing"
    pk3.mkdir()
    assert _refused(GC.pack_block, arm_dirs, pk3, 1), "arms without an appended bank were packed for its rows"
    # committee without residuals: the blend of a single run's own exported programs, no GP run
    csrc, cdst = work / "single_src", work / "single_committee"
    (csrc / "_engine" / "snap" / "g8").mkdir(parents=True)
    cdst.mkdir()
    cidr0 = ctx["identity_cids"][0]
    yr0 = ctx["bank_full"][cidr0]["y"]
    n_a0 = ctx["bank_full"][cidr0]["Xa"].shape[0]
    e1, e2 = np.random.default_rng(1).normal(size=len(yr0)), np.random.default_rng(2).normal(size=len(yr0))
    np.savez(csrc / "_engine" / "snap" / "g8" / f"{cidr0}.s1.npz", Q_cons=np.column_stack([yr0 + e1, yr0 + e2]), AKF_cons=np.ones((n_a0, 2)),
             PA_cons=np.column_stack([np.ones(n_a0), 3.0 * np.ones(n_a0)]), u_cons=np.array([0.2, 0.1]), dsls=np.array(["p1", "p2"]))
    (csrc / f"{cidr0}.json").write_text(json.dumps({"arm": "single", "outer_folds": 0, "wall_s": 2.0, "pipelines": [[{"generations": 8}]],
                                                     "settings": {"bank": str(P.EVAL_BANK), "bank_sha256": P.digest_file(P.EVAL_BANK), "prior": "dev91-free"}}))
    assert GC.derive_committee(csrc, cdst, 3) == 1 and GC.derive_committee(csrc, cdst, 3) == 0
    cbook = json.loads((cdst / f"{cidr0}.json").read_text())
    with np.load(cdst / f"{cidr0}.npz") as z:
        assert GC.rmse(z["q_s1"], yr0) < GC.rmse(yr0 + e1, yr0) and np.allclose(z["ref_s1"], cbook["weights"][0] + 3.0 * cbook["weights"][1])
    assert cbook["arm"] == "single_committee" and cbook["dsls"] == ["p1", "p2"] and abs(sum(cbook["weights"]) - 1.0) < 0.2
    assert _refused(GC.derive_committee, work / "arm_chain", work / "single_committee_x", 3), "a folder that is not a single-run arm was accepted"
    # stack-level read: fixed non-negative blend fitted on SEARCH rows, applied to the audit slot
    cidr = ctx["identity_cids"][0]
    dr = ctx["bank_full"][cidr]
    n_sr, n_ar = len(dr["y"]), dr["Xa"].shape[0]
    bank_sha = P.digest_file(P.EVAL_BANK)
    g_blk, o_blk, s_out = work / "stk_gp", work / "stk_other", work / "stk_out"
    for b_ in (g_blk, o_blk, s_out):
        b_.mkdir()
    noise = np.random.default_rng(3).normal(size=n_sr)
    np.savez(g_blk / f"{cidr}.npz", ok=np.array(True), q_good=dr["y"] + 0.01 * noise, bag_good=np.ones(n_ar), ref_good=np.ones(n_ar),
             q_good_s1=dr["y"], bag_good_s1=np.ones(n_ar), ref_good_s1=np.ones(n_ar))
    (g_blk / "SIDECAR.json").write_text(json.dumps({"names": ["good_s1", "good"], "closure": "fair", "bank_sha256": bank_sha, "contexts": [cidr]}))
    np.savez(o_blk / f"{cidr}.npz", ok=np.array(True), q_a=np.full(n_sr, dr["y"].mean()), bag_a=np.zeros(n_ar), ref_a=np.zeros(n_ar),
             q_b=dr["y"] + 3.0 * noise * dr["y"].std(), bag_b=np.zeros(n_ar), ref_b=np.zeros(n_ar))
    (o_blk / "SIDECAR.json").write_text(json.dumps({"names": ["a", "b"], "closure": "fair", "bank_sha256": bank_sha, "contexts": [cidr]}))
    sdoc = GC.stack_block(g_blk, o_blk, P.EVAL_BANK, s_out)
    assert sdoc["names"] == ["stk_base", "stk_good"] and sdoc["gp_weight_mean"]["stk_good"] > 0.9, sdoc["gp_weight_mean"]
    assert sdoc["closure"] == "fair" and sdoc["inputs_in_corpus"] and set(sdoc["inputs"]) == {"gp", "other"}, "a blend must carry its inputs' status"
    wts = sdoc["weights_by_cid"][cidr]
    assert abs(sum(wts["stk_base"]) - 1.0) < 1e-9 and abs(sum(wts["stk_good"]) - 1.0) < 1e-9 and min(wts["stk_good"]) >= 0.0
    with np.load(s_out / f"{cidr}.npz") as z:
        assert np.allclose(z["ref_stk_base"], 0.0) and np.allclose(z["ref_stk_good"], wts["stk_good"][-1]) and z["q_stk_good"].shape == (n_sr,)
    assert _refused(GC.stack_block, g_blk, o_blk, P.EVAL_BANK, s_out), "a stack into a filled directory was accepted"
    return {"checks": 51, "dir": str(work)}


def gate_static_hygiene(ctx) -> dict:
    banned = ("/tmp", "os.remove", ".unlink(", "rmtree", "os.rmdir", "h416_", "ya_sha256")
    for name in SOURCES:
        body = (HERE / name).read_text()
        for b in banned:
            assert b not in body, f"{name} contains {b!r}"
    return {"files": len(SOURCES)}


GATES = (
    gate_quarantine,
    gate_pool_dedup_union,
    gate_pool_no_split_marker,
    gate_zero_query_leakage,
    gate_unique_rows_and_commitments,
    gate_batch_caps,
    gate_rank_gauss_roundtrip,
    gate_canonical_slots,
    gate_lr_schedule,
    gate_static_hygiene,
    gate_feature_strata,
    gate_benchmark_split_audit,
    gate_eval_pipeline_identity,
    gate_builder_matches_reference,
    gate_block_closures,
    gate_gp_chain_units,
    gate_cpu_mini_train,
    gate_remote_workers,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", default="")
    ap.add_argument("--receipt", default=str(P.EVIDENCE / "trackD" / "SELFTEST_H458_RECEIPT.json"))
    a = ap.parse_args()
    bank_full = CB.load_bank(P.EVAL_BANK)
    by_fam: dict[str, list[str]] = {}
    for cid, d in bank_full.items():
        by_fam.setdefault(d["family"][4:12], []).append(cid)
    ctx = {"pool": FP.FamilyPool(P.POOL_DIR, verify_data=True), "bank_full": bank_full, "identity_cids": [sorted(by_fam["f8fd51f3"])[0], sorted(by_fam["4286560c"])[0]]}
    results, ok = [], True
    for g in GATES:
        if a.only and a.only not in g.__name__:
            continue
        t0 = time.time()
        try:
            info, passed = g(ctx), True
        except Exception as exc:  # report every gate, never stop at the first failure
            info, passed = {"error": f"{type(exc).__name__}: {exc}"[:500], "trace": traceback.format_exc()[-700:]}, False
        ok = ok and passed
        results.append({"gate": g.__name__, "pass": passed, "seconds": round(time.time() - t0, 1), **info})
        print(json.dumps(results[-1])[:900], flush=True)
    doc = {"schema": "jope.h458.selftest/v1", "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "all_pass": ok, "passed": sum(r["pass"] for r in results), "total": len(results), "not_covered": ["a real CUDA out-of-memory error (an injected one is covered)", "GP-residual target mode"], "deps": P.verify_deps(strict=False), "results": results, **P.DISCLOSURE}
    if not a.only:
        rp = P.guard_write_path(Path(a.receipt))
        rp.parent.mkdir(parents=True, exist_ok=True)
        rp.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    print(f"SELFTEST_H458 {doc['passed']}/{doc['total']} {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
