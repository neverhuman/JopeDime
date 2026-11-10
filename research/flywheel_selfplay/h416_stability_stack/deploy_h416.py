#!/usr/bin/env python3
"""H416 deploy unit (PREREG §5-§7): the H415 deploy unit unchanged, then every usable H program AUDIT-evaluated as a
full-SEARCH refit and as its 5 fold models, so stability-selected out-of-fold stacks (SXB) can be deployed from them.

One unit = one TRAIN context. It first runs `deploy_h415.deploy_unit` at its published bytes (the replays, the pools,
the library, H410's builders, H and its 5-fold out-of-fold matrix Q, KFB / HFR / KFB1). Thin wrappers around three of
its calls record, without changing them, what H416 needs:
  deploy_h415.hf_stage      -> the context, H (candidates in H order), the usable shas and Q
  deploy_h410.audit_spec    -> each StackSpec output's AUDIT prediction (None = the SEARCH-mean predictor)
  deploy_h415.kfb_deploy    -> each KFB-style output's AUDIT prediction
Then, outside every selection scope, two new ledger requests per context: `<stage>.H.pa` (every usable H program's
full-SEARCH refit -> AUDIT) and `<stage>.H.akf` (its 5 fold models -> AUDIT, the akf<j> envelopes of deploy_h415).
Programs already evaluated are memo hits. The unit writes one compressed npz per context (Q, SEARCH y, fold labels,
the refit and fold-bag AUDIT matrices, the reference outputs' AUDIT predictions, AUDIT y and baseline).

mode dev     (P201e): nothing else; the offline design grid (train_h416.py select) reads the npz files.
mode confirm (P201b): the locked design (DESIGN.json, PREREG part B) is fit on Q inside a selection scope and deployed
             from the same columns: one output `P3+LIB3.SXB`, scored once. The npz files also feed the descriptive
             grid replication (PREREG §8.4).

Meta-learners, deploy modes and consensus outputs: meta_h416.py (numpy only; also used by the offline grid).
"""

from __future__ import annotations

import hashlib
import io
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import deploy_h415 as DP15  # noqa: E402  (FIRST: deploy_h410 -> gp_h381 -> h335_core pins, guard)

import numpy as np  # noqa: E402

import meta_h416 as M  # noqa: E402

DP, GPH, K, C, LAB, L, D, ST, G, E = DP15.DP, DP15.GPH, DP15.K, DP15.C, DP15.LAB, DP15.L, DP15.D, DP15.ST, DP15.G, \
    DP15.E

WAVE = "H416"
SCHEMA = "jope.h416.deploy-unit/v1"
REFS = ("GS.stock", "P3+LIB3.KFB", "P3+LIB3.HFR", "P3+LIB3.BAG_CAR", "P3+LIB3.W48", "P3+LIB3.CAR", "P2+LIB2.BAG_CAR")
KPOOL = "P3+LIB3"


def _check_copies() -> None:
    """meta_h416's verbatim copies return what the pinned originals return (PREREG §9)."""
    rng = np.random.default_rng(416999)
    Q = rng.normal(size=(57, 9))
    y = Q[:, 3] + 0.3 * rng.normal(size=57)
    for steps in (1, 7, 100):
        if not np.array_equal(M.kcar(Q, y, steps), DP15.kcar(Q, y, steps)):
            raise AssertionError("meta_h416.kcar != deploy_h415.kcar")
    for parts in ((416201, "BCAR", "abc", 0), ("kf", 3), (1,)):
        if M.stream_seed(*parts) != C.stream_seed(*parts):
            raise AssertionError("meta_h416.stream_seed != common_h317.stream_seed")
    for a, b in ((1.0, 2.0), (3.0, 1.5), (None, 1.0), (1.0, 0.0), (float("nan"), 1.0), (1e-20, 1.0)):
        if M.audit_utility_from_rmse(a, b) != C.audit_utility_from_rmse(a, b):
            raise AssertionError("meta_h416.audit_utility_from_rmse differs")


_check_copies()


# ----------------------------------------------------------------------------------------------------------------------
# Capture wrappers (record arguments / return values; the wrapped functions run unchanged)
# ----------------------------------------------------------------------------------------------------------------------
_CAP_LOCK = threading.Lock()
_CAP: Dict[str, dict] = {}
_orig_hf_stage = DP15.hf_stage
_orig_audit_spec = DP.audit_spec
_orig_kfb_deploy = DP15.kfb_deploy


def _cap(cid: str) -> dict:
    with _CAP_LOCK:
        return _CAP.setdefault(cid, {"preds": {}})


def _hf_stage(rt, ctx, cands):
    out = _orig_hf_stage(rt, ctx, cands)
    _cap(ctx.cid)["hf"] = {"ctx": ctx, "cands": list(cands), "ok": list(out[0]), "Q": out[1], "b_kf": out[3]}
    return out


def _audit_spec(rt, pipe, g, spec, cand_of):
    raw, pred = _orig_audit_spec(rt, pipe, g, spec, cand_of)
    _cap(g.ctx.cid)["preds"][pipe] = pred
    return raw, pred


def _kfb_deploy(rt, pipe, g, members, w, cand_of):
    raw, pred = _orig_kfb_deploy(rt, pipe, g, members, w, cand_of)
    _cap(g.ctx.cid)["preds"][pipe] = pred
    return raw, pred


DP15.hf_stage = _hf_stage
DP.audit_spec = _audit_spec
DP15.kfb_deploy = _kfb_deploy


# ----------------------------------------------------------------------------------------------------------------------
# AUDIT prediction columns (one ledger request each, outside every selection scope)
# ----------------------------------------------------------------------------------------------------------------------
def akf_columns(rt, pipe: str, ctx, cands: Sequence["G.Candidate"]
                ) -> Tuple[List[Optional[np.ndarray]], List[int], float]:
    """Each candidate's mean over its completed fold models' AUDIT predictions (deploy_h415.kfb_deploy rule); None if
    no fold model completed."""
    res = DP15.KF_LEDGER.score(rt.engine, pipe, ctx, cands)
    na = len(ctx.ya)
    cols, nfold = [], []
    for i in range(len(cands)):
        ps = [res[j][i].p1 for j in range(DP15.KF_K) if res[j][i].status == "completed" and res[j][i].p1 is not None
              and len(res[j][i].p1) == na and np.all(np.isfinite(res[j][i].p1))]
        cols.append(np.mean(ps, axis=0) if ps else None)
        nfold.append(len(ps))
    wall = float(sum(r.wall_s for j in range(DP15.KF_K) for r in res[j]))
    return cols, nfold, wall


def pa_columns(rt, pipe: str, ctx, seed: int, cands: Sequence["G.Candidate"]
               ) -> Tuple[List[Optional[np.ndarray]], float]:
    """Each candidate's full-SEARCH refit AUDIT prediction; None unless completed with a prediction (the rule of
    deploy_h415.kfb_deploy's full-refit fallback; ledger results are in request order, as there)."""
    res = rt.ledger.score(pipe, ctx, seed, list(cands))  # seed = ledger key part only (as in deploy_h410)
    if [a.sha for a in res] != [c.sha for c in cands]:
        raise AssertionError("AUDIT results are not in request order")
    cols = [np.asarray(a.pa, np.float64) if a.status == "completed" and a.pa is not None else None for a in res]
    return cols, float(sum(a.wall_s for a in res))


# ----------------------------------------------------------------------------------------------------------------------
# Unit
# ----------------------------------------------------------------------------------------------------------------------
def _run_h415(rt, tag: str, cid: str, sources: Dict[str, object], seed: int, jobs: int) -> Tuple[dict, dict]:
    """deploy_h415.deploy_unit with capture. A stored H415-stage record (a resumed unit) has no capture, so the unit is
    then recomputed under tag + 'r' (every evaluation a memo hit, new ledger names); the stored record is returned."""
    with _CAP_LOCK:
        _CAP.pop(cid, None)
    rec15 = DP15.deploy_unit(rt, tag, cid, sources, seed, jobs)
    cap = _cap(cid)
    if "hf" not in cap:
        DP15.deploy_unit(rt, tag + "r", cid, sources, seed, jobs)
        cap = _cap(cid)
        if "hf" not in cap:
            raise AssertionError(f"{cid}: no capture after recompute")
        cap["recomputed_as"] = tag + "r"
    with _CAP_LOCK:
        _CAP.pop(cid, None)
    return rec15, cap


def _ref_preds(cap: dict, tag: str) -> Dict[str, Optional[np.ndarray]]:
    """AUDIT predictions of the reference outputs. BAG_<b> absent (no twin) -> <b>, the H410 rule."""
    p = cap["preds"]
    stage = f"H415_{cap.get('recomputed_as') or tag}"
    out = {}
    for k in REFS:
        alts = [k] + ([k.replace("BAG_", "")] if "BAG_" in k else [])
        out[k] = next((p[f"{stage}.{a}"] for a in alts if f"{stage}.{a}" in p), None)
    return out


def h416_unit(rt, tag: str, cid: str, sources: Dict[str, object], seed: int, mode: str, npz_dir: Path,
              design: Optional[dict] = None, jobs: int = 3) -> dict:
    if mode not in ("dev", "confirm"):
        raise SystemExit(f"unknown mode {mode}")
    if mode == "confirm" and not design:
        raise SystemExit("confirm needs the locked design (DESIGN.json)")
    stage = f"H416_{tag}"
    hit = rt.store.result(stage, cid, mode)
    if hit:
        return hit
    t0 = time.time()
    rec15, cap = _run_h415(rt, tag, cid, sources, seed, jobs)
    hf = cap["hf"]
    ctx, ok = hf["ctx"], hf["ok"]
    Q = np.asarray(hf["Q"], np.float64)
    by_sha = {c.sha: c for c in hf["cands"]}
    cands = [by_sha[s] for s in ok]
    y = np.asarray(ctx.search.y, np.float64)
    ya = np.asarray(ctx.ya, np.float64)
    na, ba, ym = len(ya), float(ctx.ba), float(ctx.y_search_mean)
    refs = _ref_preds(cap, tag)
    rec = {"type": "result", "schema": SCHEMA, "wave": WAVE, "tag": tag, "mode": mode, "cid": cid,
           "family_id": rt.fam_of[cid], "seed": mode, "search_seed": seed, "h415_stage": f"H415_{tag}",
           "recomputed_as": cap.get("recomputed_as"), "n_usable": len(ok), "n_search": int(len(y)), "n_audit": na,
           "env_sha": dict(ctx.env_sha), "audit_board_sha256": hashlib.sha256(ctx.audit.to_bytes()).hexdigest(),
           "ref_none": {k: v is None for k, v in refs.items()}, "outputs": {},
           "worker_s": {}}
    if not ok:
        rec["error"] = "no usable H program"
    else:
        akf, nfold, w_akf = akf_columns(rt, f"{stage}.H.akf", ctx, cands)
        pa, w_pa = pa_columns(rt, f"{stage}.H.pa", ctx, seed, cands)
        rec["worker_s"].update({"H.akf": w_akf, "H.pa": w_pa})
        rec["n_akf_failed"] = int(sum(1 for c in akf if c is None))
        rec["n_pa_failed"] = int(sum(1 for c in pa if c is None))
        AKF = np.column_stack([c if c is not None else np.full(na, np.nan) for c in akf])
        PA = np.column_stack([c if c is not None else np.full(na, np.nan) for c in pa])
        R = {k: (v if v is not None else np.full(na, np.nan)) for k, v in refs.items()}
        buf = io.BytesIO()
        np.savez_compressed(buf, Q=Q, y=y, lab=DP15._LABELS[cid], AKF=AKF, PA=PA, akf_nfold=np.asarray(nfold),
                            pa_ok=np.asarray([c is not None for c in pa]), ya=ya, ba=np.float64(ba),
                            ym=np.float64(ym), shas=np.asarray(ok), ref_keys=np.asarray(list(R)),
                            REF=np.column_stack([R[k] for k in R]), ref_none=np.asarray([refs[k] is None for k in R]))
        raw = buf.getvalue()
        npz_dir.mkdir(parents=True, exist_ok=True)
        p = npz_dir / f"{cid}.npz"
        p.write_bytes(raw)
        rec["npz"] = {"path": str(p), "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
        # harness check (PREREG §9): KCAR100.bag from the matrices == the unit's own KFB output
        w = M.meta_weights("KCAR100", Q, y, cid)
        keep = np.flatnonzero(w > 0)
        pred, _ = M.deploy_pred("bag", w[keep], [akf[i] for i in keep], [pa[i] for i in keep], ym, na)
        rec["kcar100_bag_u"] = M.utility(pred, ya, ba)[0]
        rec["kfb_u"] = rec15["outputs"].get(f"{KPOOL}.KFB", {}).get("value")
        if mode == "confirm":
            name = design["design"]
            if name in M.CONS:
                pred, n_sub = M.cons_pred(name, refs, ym, na)
                spec = {"design": name, "of": list(M.CONS[name]), "none_substituted": n_sub}
            else:
                meta, dm = name.split(".")
                with D.selection_scope(f"{stage}.SXB.fit"):  # the weights see SEARCH out-of-fold data only
                    w = M.meta_weights(meta, Q, y, cid)
                keep = np.flatnonzero(w > 0)
                pred, info = M.deploy_pred(dm, w[keep], [akf[i] for i in keep], [pa[i] for i in keep], ym, na)
                spec = {"design": name, "members": [ok[i] for i in keep], "w": [float(x) for x in w[keep]], **info}
            u, rmse = M.utility(pred, ya, ba)
            rec["outputs"][f"{KPOOL}.SXB"] = {"value": u, "invalid": False,
                                              "raw": {"status": "completed", "utility": u, "rmse": rmse},
                                              "spec": spec}
    rec["wall_s"] = time.time() - t0
    rt.store.append(stage, cid, mode, rec)
    return rec


def code_files() -> Tuple[str, ...]:
    return ("deploy_h416.py", "meta_h416.py") + DP15.code_files()
