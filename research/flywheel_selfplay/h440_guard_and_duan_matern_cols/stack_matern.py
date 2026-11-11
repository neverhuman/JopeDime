#!/usr/bin/env python3
"""Wave B (`guard_and_duan_matern_cols`): Stack apply & statistical readout.

  stack_matern.py apply   --run-dir DIR --tag TAG --panel NAME --bank FILE --cols DIR --out FILE [--procs N]
                          [--h430-cols DIR] [--h433-cols DIR]
  stack_matern.py readout --apply FILE --out FILE [--units FILE] [--exclude FILE]

Arms evaluated per context:
  BASE         stored h416.SXB (replay-checked: max w_diff == 0)
  G_ONLY       100% ya-free Peer-Consensus Extrapolation Guard + Multi-Loss BCAR + eigh L2 on STORED candidates ONLY [Primary 1]
  C_PLAIN      stored candidates + (cb_plain, gp_plain) through stock CA.bcar (plain non-Duan, non-entity control)
  M_BCAR       stored candidates + 7 Duan-Matern-Entity-DAG columns through stock CA.bcar
  GM_FULL      Guarded 2L combiner (G_ONLY) + 7 Duan-Matern-Entity-DAG columns                                       [Primary 2]
  G_H430       Guarded 2L combiner + H430 (key_direct, key_gbm, factor_glm; key_direct fit-range clipped)            [Cross-lane]
  M_H430_BCAR  stored candidates + 7 Duan-Matern-Entity-DAG + H430 (key_direct fit-range clipped) via CA.bcar        [Cross-lane]
  GM_H430      Guarded 2L combiner + 7 Duan-Matern-Entity-DAG + H430 (key_direct, key_gbm, factor_glm)               [Cross-lane]
  G_CW12       Guarded 2L combiner + H430 K3 + H433 F4 (form_gbm, form_gam, form_sr, form_interp)                    [Cross-lane]
  GM_CW12      Guarded 2L combiner + 7 Duan-Matern-Entity-DAG + H430 K3 + H433 F4                                    [Cross-lane]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
if os.environ.get("WAVE_BASE_CODE"):
    sys.path.append(os.environ["WAVE_BASE_CODE"])
import cache_h422 as CA  # noqa: E402
import guard_matern_core as GM  # noqa: E402
import meta_h416 as M  # noqa: E402

SXB_KEY = "P3+LIB3.SXB"
H430_K3 = ("key_direct", "key_gbm", "factor_glm")
H430_K2 = ("key_gbm", "factor_glm")
H436_KH2 = ("key_hybrid", "ridge_z")
H433_F4 = ("form_gbm", "form_gam", "form_sr", "form_interp")
BOOT_SEED, N_BOOT = 434_150, 10_000


def _u(pred, ya, ba) -> float:
    return float(M.utility(pred, ya, ba)[0])


def _load_col_block(z, names: tuple[str, ...], n_s: int, n_a: int) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    if z is None or not bool(z["ok"]):
        return None
    if not all(f"q_{v}" in z.files and len(z[f"q_{v}"]) == n_s and len(z[f"ref_{v}"]) == n_a for v in names):
        return None
    Q = np.column_stack([np.asarray(z[f"q_{v}"], np.float64) for v in names])
    BAG = np.column_stack([np.asarray(z[f"bag_{v}"], np.float64) for v in names])
    REF = np.column_stack([np.asarray(z[f"ref_{v}"], np.float64) for v in names])
    return Q, BAG, REF


def one(job) -> dict:
    run_dir, tag, cols_dir, h430_dir, h433_dir, h436_dir, cid, bank_row = job
    t0 = time.process_time()
    rec, a = CA.stored(run_dir, tag, cid)
    st = rec["outputs"][SXB_KEY]
    if st.get("invalid") or st.get("spec", {}).get("w") is None:
        raise SystemExit(f"{cid}: no stored baseline output")

    # Replay check of stored SXB
    w0 = CA.bcar(a["Q"], a["y"], cid)
    ws = np.zeros(len(w0))
    for s, x in zip(st["spec"]["members"], st["spec"]["w"]):
        ws[a["shas"].index(s)] = x
    u_re = _u(0.5 * a["BAG"] @ w0 + 0.5 * a["REF"] @ w0, a["ya"], a["ba"])

    Xs, Xa, y64, lab, ya_sha = bank_row
    aligned = bool(
        np.array_equal(y64, a["y"])
        and np.array_equal(lab.astype(np.int64), a["lab"].astype(np.int64))
        and hashlib.sha256(np.ascontiguousarray(a["ya"], np.float64).tobytes()).hexdigest() == ya_sha
        and Xs.shape[0] == len(a["y"])
        and Xa.shape[0] == len(a["ya"])
    )

    # Raw stored AKF / PA / pa_ok (already sha-verified inside CA.stored)
    raw_npz = np.load(Path(run_dir) / f"h416_{tag}_npz" / f"{cid}.npz", allow_pickle=False)
    m_gp = a["Q"].shape[1]
    mean_a = np.full(len(a["ya"]), a["ym"], dtype=np.float64)
    bag_gp = [
        None if not np.all(np.isfinite(raw_npz["AKF"][:, i])) else raw_npz["AKF"][:, i].astype(np.float64)
        for i in range(m_gp)
    ]
    ref_gp = [
        raw_npz["PA"][:, i].astype(np.float64) if bool(raw_npz["pa_ok"][i]) else None
        for i in range(m_gp)
    ]
    bcols_gp = [b if b is not None else (r if r is not None else mean_a) for b, r in zip(bag_gp, ref_gp)]

    o = {
        "cid": cid,
        "family": rec["family_id"],
        "BASE": float(st["value"]),
        "w_diff": float(np.abs(ws - w0).max()),
        "u_diff": abs(u_re - float(st["value"])),
        "n_search": int(len(a["y"])),
        "n_audit": int(len(a["ya"])),
        "aligned": aligned,
    }

    if not aligned:
        for arm in ("G_ONLY", "C_PLAIN", "M_BCAR", "GM_FULL", "G_H430", "M_H430_BCAR", "GM_H430", "G_CW12", "GM_CW12", "GM_CW12H"):
            o[arm] = o["BASE"]
        o["column_ok"] = False
        o["h430_ok"] = False
        o["h433_ok"] = False
        o["h436_ok"] = False
        o["cpu_s"] = time.process_time() - t0
        return o

    # Primary Arm 1: G_ONLY (pure ya-free Guard + Multi-Loss BCAR + eigh L2 on stored candidates ONLY)
    p_g, meta_g = GM.guard_and_stack_2l(cid, Xs, Xa, a["y"], a["lab"], a["Q"], bcols_gp, ref_gp, [])
    o["G_ONLY"] = _u(p_g, a["ya"], a["ba"])
    o["gp_dropped"] = meta_g["gp_dropped"]
    o["g_only_l2_gain"] = meta_g["l2_gain"]

    # Load our Wave B columns
    p_col = Path(cols_dir) / f"{cid}.npz"
    z_col = np.load(p_col, allow_pickle=False) if p_col.exists() else None
    blk_m = _load_col_block(z_col, GM.PRIMARY_NAMES, len(a["y"]), len(a["ya"]))
    blk_c = _load_col_block(z_col, GM.CONTROL_NAMES, len(a["y"]), len(a["ya"]))
    o["column_ok"] = bool(blk_m is not None and blk_c is not None)

    if o["column_ok"]:
        Qm, Bm, Rm = blk_m
        Qc, Bc, Rc = blk_c
        # C_PLAIN via stock CA.bcar
        wc = CA.bcar(np.hstack([a["Q"], Qc]), a["y"], cid)
        o["C_PLAIN"] = _u(0.5 * np.hstack([a["BAG"], Bc]) @ wc + 0.5 * np.hstack([a["REF"], Rc]) @ wc, a["ya"], a["ba"])
        # M_BCAR via stock CA.bcar
        wm = CA.bcar(np.hstack([a["Q"], Qm]), a["y"], cid)
        o["M_BCAR"] = _u(0.5 * np.hstack([a["BAG"], Bm]) @ wm + 0.5 * np.hstack([a["REF"], Rm]) @ wm, a["ya"], a["ba"])
        o["w_added_M_BCAR"] = [float(x) for x in wm[m_gp:]]
        # Primary Arm 2: GM_FULL
        p_gm, meta_gm = GM.guard_and_stack_2l(cid, Xs, Xa, a["y"], a["lab"], a["Q"], bcols_gp, ref_gp, [(Qm, Bm, Rm)])
        o["GM_FULL"] = _u(p_gm, a["ya"], a["ba"])
        o["gm_ext_admitted"] = meta_gm["ext_admitted"]
        o["gm_ext_share"] = meta_gm["ext_weight_share"]
        sd = max(float(a["y"].std()), 1e-12)
        e_blend = a["Q"] @ w0 - a["y"]
        rm_blend = max(float(np.sqrt(np.mean(e_blend ** 2))), 1e-12)
        for v in GM.NAMES:
            ej = np.asarray(z_col[f"q_{v}"], np.float64) - a["y"]
            rmj = max(float(np.sqrt(np.mean(ej ** 2))), 1e-12)
            rho = float(np.dot(e_blend, ej) / max(float(np.linalg.norm(e_blend) * np.linalg.norm(ej)), 1e-12))
            r_rat = rmj / rm_blend
            lim = (0.9 - 0.1 * (r_rat ** 2)) / max(0.8 * r_rat, 1e-12)
            o[f"alone_{v}"] = _u(0.5 * z_col[f"bag_{v}"] + 0.5 * z_col[f"ref_{v}"], a["ya"], a["ba"])
            o[f"oof_u_{v}"] = float(-np.log(max(rmj / sd, 1e-12)))
            o[f"overlap_rho_{v}"] = rho
            o[f"h434_bound_{v}"] = bool(rho <= lim)
        o["oof_u_best_stored"] = float(
            -np.log(max(float(np.sqrt(((a["Q"] - a["y"][:, None]) ** 2).mean(0)).min()) / sd, 1e-12))
        )
    else:
        o["C_PLAIN"] = o["BASE"]
        o["M_BCAR"] = o["BASE"]
        o["GM_FULL"] = o["G_ONLY"]
        o["w_added_M_BCAR"] = [0.0] * len(GM.PRIMARY_NAMES)

    # Read-only Cross-Lane Admission of cl_fwriter's H430 Exact-Key columns (if present)
    blk_k3 = None
    if h430_dir:
        pk = Path(h430_dir) / f"{cid}.npz"
        zk = np.load(pk, allow_pickle=False) if pk.exists() else None
        blk_k3 = _load_col_block(zk, H430_K3, len(a["y"]), len(a["ya"]))
    o["h430_ok"] = bool(blk_k3 is not None)

    if o["h430_ok"]:
        Qk, Bk, Rk = [arr.copy() for arr in blk_k3]
        # Fit-target-range clip on key_direct (index 0 of H430_K3, requested by cl_fwriter after d8134cd42a)
        y_lo = float(np.min(a["y"]))
        y_hi = float(np.max(a["y"]))
        y_pad = 0.25 * max(y_hi - y_lo, float(np.std(a["y"])), 1e-9)
        Qk[:, 0] = np.clip(Qk[:, 0], y_lo - y_pad, y_hi + y_pad)
        Bk[:, 0] = np.clip(Bk[:, 0], y_lo - y_pad, y_hi + y_pad)
        Rk[:, 0] = np.clip(Rk[:, 0], y_lo - y_pad, y_hi + y_pad)
        p_gk, meta_gk = GM.guard_and_stack_2l(cid, Xs, Xa, a["y"], a["lab"], a["Q"], bcols_gp, ref_gp, [(Qk, Bk, Rk)])
        o["G_H430"] = _u(p_gk, a["ya"], a["ba"])
        o["gh430_ext_share"] = meta_gk["ext_weight_share"]
        if o["column_ok"]:
            Qm, Bm, Rm = blk_m
            w_mk = CA.bcar(np.hstack([a["Q"], Qm, Qk]), a["y"], cid)
            o["M_H430_BCAR"] = _u(
                0.5 * np.hstack([a["BAG"], Bm, Bk]) @ w_mk + 0.5 * np.hstack([a["REF"], Rm, Rk]) @ w_mk,
                a["ya"],
                a["ba"],
            )
            p_gmk, meta_gmk = GM.guard_and_stack_2l(
                cid, Xs, Xa, a["y"], a["lab"], a["Q"], bcols_gp, ref_gp, [(Qm, Bm, Rm), (Qk, Bk, Rk)]
            )
            o["GM_H430"] = _u(p_gmk, a["ya"], a["ba"])
            o["gmk_ext_share"] = meta_gmk["ext_weight_share"]
        else:
            w_mk = CA.bcar(np.hstack([a["Q"], Qk]), a["y"], cid)
            o["M_H430_BCAR"] = _u(
                0.5 * np.hstack([a["BAG"], Bk]) @ w_mk + 0.5 * np.hstack([a["REF"], Rk]) @ w_mk,
                a["ya"],
                a["ba"],
            )
            o["GM_H430"] = o["G_H430"]
    else:
        o["G_H430"] = o["G_ONLY"]
        o["M_H430_BCAR"] = o["M_BCAR"]
        o["GM_H430"] = o["GM_FULL"]

    # Optional Read-only Cross-Lane Admission of cl_fwriter's H433 Raw-Formula columns (if present)
    blk_f4 = None
    if h433_dir:
        pf = Path(h433_dir) / f"{cid}.npz"
        zf = np.load(pf, allow_pickle=False) if pf.exists() else None
        blk_f4 = _load_col_block(zf, H433_F4, len(a["y"]), len(a["ya"]))
    o["h433_ok"] = bool(blk_f4 is not None)

    cw_blks = []
    if o["h430_ok"]:
        cw_blks.append((Qk, Bk, Rk))
    if o["h433_ok"]:
        Qf, Bf, Rf = [arr.copy() for arr in blk_f4]
        y_lo = float(np.min(a["y"]))
        y_hi = float(np.max(a["y"]))
        y_pad = 0.50 * max(y_hi - y_lo, float(np.std(a["y"])), 1e-9)
        # Clip form_sr / form_interp extreme extrapolations before peer guard
        for idx_f in (2, 3):
            Qf[:, idx_f] = np.clip(Qf[:, idx_f], y_lo - y_pad, y_hi + y_pad)
            Bf[:, idx_f] = np.clip(Bf[:, idx_f], y_lo - y_pad, y_hi + y_pad)
            Rf[:, idx_f] = np.clip(Rf[:, idx_f], y_lo - y_pad, y_hi + y_pad)
        cw_blks.append((Qf, Bf, Rf))

    if cw_blks:
        p_gcw, _ = GM.guard_and_stack_2l(cid, Xs, Xa, a["y"], a["lab"], a["Q"], bcols_gp, ref_gp, cw_blks)
        o["G_CW12"] = _u(p_gcw, a["ya"], a["ba"])
        if o["column_ok"]:
            Qm, Bm, Rm = blk_m
            p_gmcw, meta_gmcw = GM.guard_and_stack_2l(
                cid, Xs, Xa, a["y"], a["lab"], a["Q"], bcols_gp, ref_gp, [(Qm, Bm, Rm)] + cw_blks
            )
            o["GM_CW12"] = _u(p_gmcw, a["ya"], a["ba"])
            o["gmcw12_ext_share"] = meta_gmcw["ext_weight_share"]
        else:
            o["GM_CW12"] = o["G_CW12"]
    else:
        o["G_CW12"] = o["G_H430"]
        o["GM_CW12"] = o["GM_H430"]

    # Optional Read-only Cross-Lane Admission of cl_fwriter's H436 Model-Backoff Key columns (key_hybrid, ridge_z)
    blk_kh2 = None
    if h436_dir:
        ph = Path(h436_dir) / f"{cid}.npz"
        zh = np.load(ph, allow_pickle=False) if ph.exists() else None
        blk_kh2 = _load_col_block(zh, H436_KH2, len(a["y"]), len(a["ya"]))
    o["h436_ok"] = bool(blk_kh2 is not None)

    if o["h436_ok"] and o["h430_ok"]:
        # Replace unguarded key_direct (col 0 of Qk) with H436 (key_hybrid, ridge_z) + keep (key_gbm, factor_glm) + H433 F4 + Wave B M7
        Qk2 = Qk[:, 1:]
        Bk2 = Bk[:, 1:]
        Rk2 = Rk[:, 1:]
        cwh_blks = [(Qk2, Bk2, Rk2), blk_kh2]
        if o["h433_ok"]:
            cwh_blks.append((Qf, Bf, Rf))
        if o["column_ok"]:
            Qm, Bm, Rm = blk_m
            cwh_blks = [(Qm, Bm, Rm)] + cwh_blks
        p_gmcwh, _ = GM.guard_and_stack_2l(cid, Xs, Xa, a["y"], a["lab"], a["Q"], bcols_gp, ref_gp, cwh_blks)
        o["GM_CW12H"] = _u(p_gmcwh, a["ya"], a["ba"])
    else:
        o["GM_CW12H"] = o["GM_CW12"]

    o["cpu_s"] = time.process_time() - t0
    return o


def cmd_apply(a) -> None:
    cfg = json.loads((Path(a.run_dir) / f"config_{a.tag}.json").read_text())
    cids = cfg["cids"]
    with np.load(a.bank, allow_pickle=False) as b:
        if sorted(b["cids"].tolist()) != sorted(cids) or set(b["cohort"].tolist()) != {a.panel}:
            raise SystemExit("bank does not hold exactly the panel's contexts")
        search, audit, y64, labs = b["search"], b["audit"], b["y64"], b["lab"]
        s_shape, a_shape, ya_shas = b["s_shape"], b["a_shape"], b["ya_sha256"]
        so = ao = lo = 0
        bank = {}
        for i, c in enumerate(b["cids"].tolist()):
            n, w = (int(v) for v in s_shape[i])
            na, f = (int(v) for v in a_shape[i])
            S = np.array(search[so:so + n * w], np.float64).reshape(n, w)
            XA = np.array(audit[ao:ao + na * f], np.float64, copy=True).reshape(na, f)
            bank[str(c)] = (
                np.array(S[:, :-1], np.float64, copy=True),
                XA,
                np.array(y64[lo:lo + n], np.float64, copy=True),
                np.array(labs[lo:lo + n], np.int64, copy=True),
                str(ya_shas[i]),
            )
            so += n * w
            ao += na * f
            lo += n

    jobs = [(a.run_dir, a.tag, a.cols, a.h430_cols, a.h433_cols, a.h436_cols, c, bank[c]) for c in cids]
    with Pool(a.procs) as pool:
        res = pool.map(one, jobs, chunksize=1)
    st = [json.loads(p.read_text()) for p in sorted(Path(a.cols).glob("cols_status_*.json"))]
    out = {
        "schema": "jope.guard_matern.apply/v1",
        "panel": a.panel,
        "tag": a.tag,
        "n": len(res),
        "bank_sha256": hashlib.sha256(Path(a.bank).read_bytes()).hexdigest(),
        "h430_cols_dir": a.h430_cols,
        "h433_cols_dir": a.h433_cols,
        "h436_cols_dir": a.h436_cols,
        "cols_status": st,
        "aligned_all": bool(all(r["aligned"] for r in res)),
        "columns_ok": int(sum(r["column_ok"] for r in res)),
        "h430_ok": int(sum(r["h430_ok"] for r in res)),
        "h433_ok": int(sum(r["h433_ok"] for r in res)),
        "h436_ok": int(sum(r["h436_ok"] for r in res)),
        "total_gp_dropped_by_guard": int(sum(r.get("gp_dropped", 0) for r in res)),
        "replay_max_w_diff": float(max(r["w_diff"] for r in res)),
        "replay_max_u_diff": float(max(r["u_diff"] for r in res)),
        "cpu_s": float(sum(r["cpu_s"] for r in res)),
        "per_context": {r["cid"]: r for r in res},
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "x") as f:
        json.dump(out, f, sort_keys=True)
    print(json.dumps({
        k: out[k]
        for k in ("panel", "n", "aligned_all", "columns_ok", "h430_ok", "total_gp_dropped_by_guard", "replay_max_w_diff", "cpu_s")
    }), flush=True)


def cmd_readout(a) -> None:
    import train_h416 as T16

    _, H = T16._stats_mod()
    ap = json.loads(Path(a.apply).read_text())
    per = ap["per_context"]
    cids = sorted(per)
    if a.exclude:
        drop = [x for x in Path(a.exclude).read_text().split() if x]
        cids = [c for c in cids if not any(c.startswith(x) for x in drop)]
    unit = {c: per[c]["family"] for c in cids}
    if a.units:
        m = json.loads(Path(a.units).read_text())
        unit = {c: str(m[c]) for c in cids}

    arms = ("BASE", "G_ONLY", "C_PLAIN", "M_BCAR", "GM_FULL", "G_H430", "M_H430_BCAR", "GM_H430", "G_CW12", "GM_CW12", "GM_CW12H")
    vals = {x: {c: per[c][x] for c in cids} for x in arms}
    pairs = (
        ("G_ONLY", "BASE"),
        ("GM_FULL", "BASE"),
        ("M_BCAR", "BASE"),
        ("M_BCAR", "C_PLAIN"),
        ("G_H430", "BASE"),
        ("GM_H430", "BASE"),
        ("GM_H430", "GM_FULL"),
        ("GM_H430", "G_H430"),
        ("M_H430_BCAR", "BASE"),
        ("G_CW12", "BASE"),
        ("GM_CW12", "BASE"),
        ("GM_CW12", "G_CW12"),
        ("GM_CW12H", "BASE"),
        ("GM_CW12H", "GM_CW12"),
    )
    pair_stats = {f"{x}_vs_{y}": T16.pair_stats(np, H, vals[x], vals[y], unit) for x, y in pairs}
    an, point, boot, _elo = T16.elo_boot(np, H, vals, unit, "BASE", BOOT_SEED, N_BOOT)
    delta_elo = {f"{x}_vs_{y}": T16.delta_elo(np, an, point, boot, x, y) for x, y in pairs}

    # Holm-Bonferroni FWER = 0.025 across the 2 primary arms (G_ONLY vs BASE, GM_FULL vs BASE)
    p_g = float(pair_stats["G_ONLY_vs_BASE"]["signflip_p"])
    p_gm = float(pair_stats["GM_FULL_vs_BASE"]["signflip_p"])
    ordered = sorted([("G_ONLY_vs_BASE", p_g), ("GM_FULL_vs_BASE", p_gm)], key=lambda t: t[1])
    holm_pass = {}
    first_ok = ordered[0][1] < 0.0125 and pair_stats[ordered[0][0]]["pass"]
    holm_pass[ordered[0][0]] = bool(first_ok)
    holm_pass[ordered[1][0]] = bool(first_ok and ordered[1][1] < 0.025 and pair_stats[ordered[1][0]]["pass"])

    ok = [c for c in cids if per[c]["column_ok"]]
    col_stats = {}
    for i, v in enumerate(GM.PRIMARY_NAMES):
        w = np.array([per[c]["w_added_M_BCAR"][i] for c in ok]) if ok else np.zeros(0)
        rhos = np.array([per[c][f"overlap_rho_{v}"] for c in ok]) if ok else np.zeros(0)
        col_stats[v] = {
            "contexts_with_weight": int((w > 0).sum()),
            "contexts_weight_ge_010": int((w >= 0.10).sum()),
            "h434_bound_pass_count": int(sum(bool(per[c][f"h434_bound_{v}"]) for c in ok)),
            "median_overlap_rho": float(np.median(rhos)) if len(rhos) else None,
            "mean_weight": float(w.mean()) if len(w) else None,
            "oof_beats_best_stored": int(sum(per[c][f"oof_u_{v}"] > per[c]["oof_u_best_stored"] for c in ok)),
        }

    prim_gm = pair_stats["GM_FULL_vs_BASE"]
    ctl_m = pair_stats["M_BCAR_vs_C_PLAIN"]
    res = {
        "schema": "jope.guard_matern.readout/v1",
        "panel": ap["panel"],
        "tag": ap["tag"],
        "n": len(cids),
        "units": len(set(unit.values())),
        "unit_kind": "file" if a.units else "family",
        "excluded": len(per) - len(cids),
        "columns_ok": ap["columns_ok"],
        "h430_ok": ap["h430_ok"],
        "total_gp_dropped_by_guard": ap["total_gp_dropped_by_guard"],
        "aligned_all": ap["aligned_all"],
        "replay_max_u_diff": ap["replay_max_u_diff"],
        "unit_mean_u": {x: float(H.family_mean(vals[x], unit)) for x in arms},
        "pairs": pair_stats,
        "delta_elo": delta_elo,
        "holm_primary_pass": holm_pass,
        "any_primary_pass": bool(any(holm_pass.values())),
        "beats_plain_control": bool(ctl_m["clipped_mean"] > 0),
        "harm": bool(prim_gm["clipped_mean"] < 0 and prim_gm["mirror_signflip_p"] < 0.025),
        "columns": col_stats,
    }

    print(f"== Guard+Matern {res['panel']} tag {res['tag']} n {res['n']} units {res['units']} "
          f"cols_ok {res['columns_ok']} h430_ok {res['h430_ok']} gp_dropped {res['total_gp_dropped_by_guard']}")
    print("   mean u: " + "  ".join(f"{x} {res['unit_mean_u'][x]:.4f}" for x in arms))
    for k, s in res["pairs"].items():
        e = res["delta_elo"][k]
        print(f"   {k:18s} gain {s['gain']:+.4f} u ({100 * s['pct_rmse']:+.2f}% RMSE) clipped {s['clipped_mean']:+.4f} "
              f"p {s['signflip_p']:.3g} lofo_min {s['lofo_min']:+.4f} med {s['family_median']:+.4f} W/T/L {s['wtl']} "
              f"pass {s['pass']} dElo {e['delta']:+.1f} [{e['ci95'][0]:+.1f}, {e['ci95'][1]:+.1f}]")
    print(f"   holm_primary_pass {res['holm_primary_pass']}  beats_plain_control {res['beats_plain_control']}  harm {res['harm']}")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "x") as f:
        json.dump(res, f, sort_keys=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("apply")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--panel", required=True)
    p.add_argument("--bank", required=True)
    p.add_argument("--cols", required=True)
    p.add_argument("--h430-cols")
    p.add_argument("--h433-cols")
    p.add_argument("--h436-cols")
    p.add_argument("--out", required=True)
    p.add_argument("--procs", type=int, default=8)
    p.set_defaults(fn=cmd_apply)

    p = sub.add_parser("readout")
    p.add_argument("--apply", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--units")
    p.add_argument("--exclude")
    p.set_defaults(fn=cmd_readout)

    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
