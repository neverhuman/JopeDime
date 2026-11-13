#!/usr/bin/env python3
"""Fast Multi-Host Frozen-Split `dev91-audit-v1` `104M` BT-Elo Tracker
Sharded Across `xbabe0` + `xbabe2` + `xbabe1` CPUs (`fast_dev91_every_n_h450.py`).

Key design:
  - Evaluates a single `104M` (`ICL_L`) checkpoint (e.g. `fleet_L_x1` newest immutable `ckpt_epoch_NN.pt` or
    exposure-weighted 3-GPU `104M` EMA soup) with `V` views (`default V = 4`, `24` batched `[4, N, Fn]` forwards/ctx)
    across 2 spaces (`Robust-Ridge` residual + exact `1,024`-bin `Yeo-Johnson` warp) on 5-fold `X_search` OOF + full
    `X_search -> X_audit` refit (`0` `y_audit` access!).
  - Supports `--cmd shard --indices ... --procs P` where each persistent worker process loads `ICLH450("L")` ONCE
    and enforces `--cpu-hard-pct` (`<= 83%` CPU cap), sharding all `91` `dev91` contexts simultaneously across
    `xbabe0` (`128c`) + `xbabe2` (`128c`) + `xbabe1` (`128c`) with `CUDA_VISIBLE_DEVICES=''` (`0` GPU/VRAM touched).
  - Computes ALL FOUR standard H450 frozen-split `dev91-audit-v1` entrants:
      1. `in_corpus_legacy_guard_stack_91`: `GM.guard_and_stack_2l` (direct comparison vs `1828.6` H449 init & `r4` `1861.6`).
      2. `in_corpus_random_split_champion_91`: `ST.stack_h450_context` full champion stack.
      3. `in_corpus_neural_plus_tab_no_gp_91`: Neural + H450-tab (`0` GP / `0` M7/K3/F4).
      4. `in_corpus_icl_boost_only_91`: Pure `104M` (`ICL_L` only, `0` trees, `0` GP).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import warnings
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import torch

torch.set_num_threads(2)

for _cand in (
    "/home/ubuntu/h450-stage/pub-r6/code",
    "/home/ubuntu/h448-stage/pub-r1/code",
    "/home/ubuntu/jope-data/h450-fleet-agy/eval_stage/pub-r6/code",
    "/home/ubuntu/jope-data/h450-fleet-agy/eval_stage/h448-pub-r1/code",
):
    if Path(_cand).is_dir() and _cand not in sys.path:
        sys.path.append(_cand)

WIDE_GPU = 64
FAST_COLS = ("icl104m_robridge", "icl104m_yj", "icl104m_blend")

_SHARD_STATE: dict = {}


def _sha_seed(tag: str) -> int:
    return int.from_bytes(hashlib.blake2b(tag.encode(), digest_size=8).digest(), "little") & 0x7FFFFFFF


def _build_views_fast(X_tr: np.ndarray, y_tr: np.ndarray, X_ev: np.ndarray, n_views: int, seed: int):
    p = X_tr.shape[1]
    rng = np.random.default_rng(seed)
    views = []
    if p <= WIDE_GPU:
        for v in range(n_views):
            perm = np.arange(p) if v == 0 else rng.permutation(p)
            views.append((X_tr[:, perm], X_ev[:, perm]))
        return views
    yc0 = y_tr - float(np.mean(y_tr))
    yn = max(float(np.linalg.norm(yc0)), 1e-12)
    scores = np.zeros(p, dtype=np.float64)
    for j in range(p):
        fin = np.isfinite(X_tr[:, j])
        if int(np.sum(fin)) >= 10:
            col0 = X_tr[fin, j] - float(np.mean(X_tr[fin, j]))
            cn = float(np.linalg.norm(col0))
            if cn > 1e-12:
                scores[j] = abs(float(np.dot(col0, yc0[fin])) / (cn * yn))
    order = np.argsort(-scores)
    for v in range(n_views):
        if v == 0:
            keep = order[:WIDE_GPU]
        else:
            n_top = int(round(0.65 * WIDE_GPU))
            top, rem = order[:n_top], order[n_top:]
            pick = rng.choice(rem, size=WIDE_GPU - len(top), replace=False)
            keep = rng.permutation(np.concatenate([top, pick]))
        views.append((X_tr[:, keep], X_ev[:, keep]))
    return views


def _load_model_cpu(ckpt_p: Path):
    from icl_model_h450 import ICLH450

    dev = torch.device("cpu")
    ck = torch.load(ckpt_p, map_location="cpu", weights_only=False)
    sd = ck.get("ema") or ck["model"]
    model = ICLH450("L").to(dev).eval()
    model.load_state_dict(sd, strict=True)
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def _pace_host_cpu(hard_pct: float) -> None:
    if hard_pct <= 0.0:
        return
    ncpu = max(1, os.cpu_count() or 1)
    for _ in range(8):
        try:
            pct = 100.0 * (os.getloadavg()[0] / float(ncpu))
        except Exception:
            break
        if pct <= hard_pct:
            break
        time.sleep(0.35)


@torch.no_grad()
def _eval_context_with_model(ctx_idx: int, bank_p: Path, model, out_dir: Path, n_views: int = 4) -> str:
    import relational_struct_h450 as REL
    import robust_resid_h450 as RR
    from icl_model import FID_SLOTS

    with np.load(bank_p, allow_pickle=False) as zb:
        assert "ya" not in zb.files, "audit labels forbidden in search bank!"
        cids = [str(x) for x in zb["cids"].tolist()]
        cid = cids[ctx_idx]
        out_p = out_dir / f"{cid}.npz"
        if out_p.is_file():
            return cid
        ss, sa = zb["s_shape"], zb["a_shape"]
        so = int(np.sum(ss[:ctx_idx, 0] * ss[:ctx_idx, 1]))
        ao = int(np.sum(sa[:ctx_idx, 0] * sa[:ctx_idx, 1]))
        lo = int(np.sum(ss[:ctx_idx, 0]))
        n, w = int(ss[ctx_idx, 0]), int(ss[ctx_idx, 1])
        na, f = int(sa[ctx_idx, 0]), int(sa[ctx_idx, 1])
        S_mat = np.array(zb["search"][so : so + n * w], np.float64).reshape(n, w)
        Xa = np.array(zb["audit"][ao : ao + na * f], np.float64, copy=True).reshape(na, f)
        Xs = np.array(S_mat[:, :-1], np.float64, copy=True)
        y = np.array(zb["y64"][lo : lo + n], np.float64, copy=True)
        lab = np.array(zb["lab"][lo : lo + n], np.int64, copy=True)

    def _pred_pair(X_tr: np.ndarray, y_tr: np.ndarray, X_ev: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            Xtr_c, Xev_c, Str, Sev, _ = REL.build_h450_struct_cols(X_tr, y_tr, X_ev, seed=seed, fast_mode=True)
        Xtr_f = np.hstack([Xtr_c, Str]) if Str.shape[1] else Xtr_c
        Xev_f = np.hstack([Xev_c, Sev]) if Sev.shape[1] else Xev_c

        # 1. Robust-Ridge residual mode
        oof_r, pq_r, used_r, _ = RR.robust_ridge_np(Xtr_f, y_tr, Xev_f, seed=seed + 11)
        y_res = (y_tr - oof_r) if used_r else y_tr
        views_r = _build_views_fast(Xtr_f, y_res, Xev_f, n_views, seed + 23)
        V = len(views_r)
        n_tr, Fn = views_r[0][0].shape[0], views_r[0][0].shape[1]
        xb = torch.from_numpy(np.stack([np.vstack([vt, ve]) for vt, ve in views_r], 0)).double()
        yb = torch.from_numpy(np.broadcast_to(y_res[None, :], (V, n_tr)).copy()).double()
        g = torch.Generator().manual_seed(seed + 37)
        slots = torch.randperm(FID_SLOTS, generator=g)[:Fn]
        logits_r, ym_r, ys_r = model(xb, yb, n_tr, slots=slots, amp=False)
        pr_res = model.mean_f64(logits_r, ym_r, ys_r).mean(0).numpy()
        p_rob = (pq_r + pr_res) if used_r else pr_res

        # 2. Exact 1,024-bin Yeo-Johnson warp mode
        warp = RR.YeoJohnsonWarpF64()
        z_tr = warp.fit_transform(y_tr)
        views_y = _build_views_fast(Xtr_f, z_tr, Xev_f, n_views, seed + 53)
        xb_y = torch.from_numpy(np.stack([np.vstack([vt, ve]) for vt, ve in views_y], 0)).double()
        yb_y = torch.from_numpy(np.broadcast_to(z_tr[None, :], (V, n_tr)).copy()).double()
        logits_y, ym_y, ys_y = model(xb_y, yb_y, n_tr, slots=slots, amp=False)
        p_yj = model.mean_warped(logits_y, ym_y, ys_y, warp.inv_torch).mean(0).numpy()

        y_lo, y_hi = float(np.min(y_tr)), float(np.max(y_tr))
        pad = 0.5 * max(y_hi - y_lo, float(np.std(y_tr)), 1e-9)
        p_rob = np.clip(np.where(np.isfinite(p_rob), p_rob, float(np.mean(y_tr))), y_lo - pad, y_hi + pad)
        p_yj = np.clip(np.where(np.isfinite(p_yj), p_yj, float(np.mean(y_tr))), y_lo - pad, y_hi + pad)
        return p_rob, p_yj

    folds = sorted(set(int(x) for x in lab.tolist()))
    q_rob = np.zeros(n, dtype=np.float64)
    q_yj = np.zeros(n, dtype=np.float64)
    bag_rob = np.zeros(na, dtype=np.float64)
    bag_yj = np.zeros(na, dtype=np.float64)

    base_seed = _sha_seed(f"fast450:{cid}")
    for fk in folds:
        m_val = (lab == fk)
        m_tr = ~m_val
        X_tr, y_tr = Xs[m_tr], y[m_tr]
        X_va_and_a = np.vstack([Xs[m_val], Xa])
        pr, py = _pred_pair(X_tr, y_tr, X_va_and_a, base_seed + 101 * (fk + 1))
        n_v = int(np.sum(m_val))
        q_rob[m_val] = pr[:n_v]
        q_yj[m_val] = py[:n_v]
        bag_rob += pr[n_v:] / float(len(folds))
        bag_yj += py[n_v:] / float(len(folds))

    ref_rob, ref_yj = _pred_pair(Xs, y, Xa, base_seed + 999)

    # OOF-selected convex weight alpha in {0.0, 0.25, 0.5, 0.75, 1.0} strictly on X_search (0 y_audit!)
    best_a, best_mse = 0.5, float("inf")
    for a_w in (0.0, 0.25, 0.50, 0.75, 1.0):
        cand_q = (1.0 - a_w) * q_rob + a_w * q_yj
        mse = float(np.mean((y - cand_q) ** 2))
        if mse < best_mse:
            best_mse, best_a = mse, a_w
    q_bl = (1.0 - best_a) * q_rob + best_a * q_yj
    bag_bl = (1.0 - best_a) * bag_rob + best_a * bag_yj
    ref_bl = (1.0 - best_a) * ref_rob + best_a * ref_yj

    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_dir / f".{cid}.{os.getpid()}.tmp.npz"
    np.savez(
        tmp,
        ok=np.array(True),
        q_icl104m_robridge=q_rob, bag_icl104m_robridge=bag_rob, ref_icl104m_robridge=ref_rob,
        q_icl104m_yj=q_yj, bag_icl104m_yj=bag_yj, ref_icl104m_yj=ref_yj,
        q_icl104m_blend=q_bl, bag_icl104m_blend=bag_bl, ref_icl104m_blend=ref_bl,
    )
    os.replace(tmp, out_p)
    return cid


def run_single_context(ctx_idx: int, bank_p: Path, ckpt_p: Path, out_dir: Path, n_views: int = 4) -> None:
    with np.load(bank_p, allow_pickle=False) as zb:
        cid = str(zb["cids"][ctx_idx])
    if (out_dir / f"{cid}.npz").is_file():
        return
    model = _load_model_cpu(ckpt_p)
    _eval_context_with_model(ctx_idx, bank_p, model, out_dir, n_views=n_views)


def _init_shard_worker(bank_s: str, ckpt_s: str, out_s: str, n_views: int, cpu_hard_pct: float, threads: int) -> None:
    try:
        os.nice(15)
    except Exception:
        pass
    try:
        torch.set_num_threads(max(1, int(threads)))
    except Exception:
        pass
    _SHARD_STATE["bank"] = Path(bank_s)
    _SHARD_STATE["out"] = Path(out_s)
    _SHARD_STATE["n_views"] = int(n_views)
    _SHARD_STATE["cpu_hard"] = float(cpu_hard_pct)
    _SHARD_STATE["model"] = _load_model_cpu(Path(ckpt_s))


def _shard_ctx_worker(ctx_idx: int) -> str:
    _pace_host_cpu(_SHARD_STATE["cpu_hard"])
    return _eval_context_with_model(
        int(ctx_idx), _SHARD_STATE["bank"], _SHARD_STATE["model"], _SHARD_STATE["out"], n_views=_SHARD_STATE["n_views"]
    )


def run_shard(indices: list[int], bank_p: Path, ckpt_p: Path, out_dir: Path,
              n_views: int = 4, procs: int = 16, cpu_hard_pct: float = 83.0, threads: int = 2) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    with np.load(bank_p, allow_pickle=False) as zb:
        cids = [str(x) for x in zb["cids"].tolist()]
    todo = [i for i in indices if 0 <= i < len(cids) and not (out_dir / f"{cids[i]}.npz").is_file()]
    if not todo:
        return 0
    w = max(1, min(int(procs), len(todo)))
    with Pool(
        processes=w,
        initializer=_init_shard_worker,
        initargs=(str(bank_p), str(ckpt_p), str(out_dir), int(n_views), float(cpu_hard_pct), int(threads)),
    ) as pool:
        for _ in pool.imap_unordered(_shard_ctx_worker, todo, chunksize=1):
            pass
    return len(todo)


def _score_ctx_worker(job):
    import guard_matern_core as GM
    import meta_h416 as M
    import stack_h450 as ST

    s, c, npz_p, Xs, Xa, y, lab, blk_m, blk_k, blk_f, blk_tab, blk_fast = job
    z16 = np.load(npz_p, allow_pickle=False)
    Q_gp = z16["Q"].astype(np.float64)
    ya = z16["ya"].astype(np.float64)
    ba = float(z16["ba"])
    ym = float(z16["ym"])
    m_gp, n_a = Q_gp.shape[1], len(ya)
    mean_a = np.full(n_a, ym, dtype=np.float64)
    bag_gp = [None if not np.all(np.isfinite(z16["AKF"][:, i])) else z16["AKF"][:, i].astype(np.float64) for i in range(m_gp)]
    ref_gp = [z16["PA"][:, i].astype(np.float64) if bool(z16["pa_ok"][i]) else None for i in range(m_gp)]
    bcols_gp = [b if b is not None else (r if r is not None else mean_a) for b, r in zip(bag_gp, ref_gp)]

    named_full = []
    for tag, blk in (("m7", blk_m), ("k3", blk_k), ("f4", blk_f), ("tab450", blk_tab), ("gpu104m", blk_fast)):
        if blk is not None:
            named_full.append((tag, blk))
    pred_champ, _ = ST.stack_h450_context(c, Xs, Xa, y, lab, Q_gp, bcols_gp, ref_gp, named_full)
    u_champ = float(M.utility(pred_champ, ya, ba)[0])

    raw_full = [blk for _, blk in named_full]
    pred_leg, _ = GM.guard_and_stack_2l(c, Xs, Xa, y, lab, Q_gp, bcols_gp, ref_gp, raw_full)
    u_leg = float(M.utility(pred_leg, ya, ba)[0])

    named_nogp = [(tag, blk) for tag, blk in (("tab450", blk_tab), ("gpu104m", blk_fast)) if blk is not None]
    pred_nogp, _ = ST.stack_h450_context(c, Xs, Xa, y, lab, None, None, None, named_nogp)
    u_nogp = float(M.utility(pred_nogp, ya, ba)[0])

    pred_icl, _ = ST.stack_h450_context(c, Xs, Xa, y, lab, None, None, None, [("gpu104m", blk_fast)])
    u_icl = float(M.utility(pred_icl, ya, ba)[0])
    return s, c, u_champ, u_leg, u_nogp, u_icl


def _rank_job_worker(job):
    import train_h416 as T16
    from analyze_h450 import _rank_entrant

    key, field, eid, u_map, fam, cids, n_boot = job
    _, H = T16._stats_mod()
    return key, _rank_entrant(H, field, eid, u_map, fam, cids, n_boot)


def score_fast_cols(cols_dir: Path, out_json: Path, n_boot: int = 2500, procs: int = 16) -> dict:
    import guard_matern_core as GM
    from analyze_h450 import H430_K3, H433_F4, SEEDS, TAGS, _check_quarantined_path, _load_col_block

    _check_quarantined_path(out_json)
    bdir = Path("/home/ubuntu/jope_plan/evidence/H450/board_post_h448/dev91-audit-v1")
    board_doc = json.loads((bdir / "BOARD.json").read_text())
    fam = {c["cid"]: c["family_id"] for c in board_doc["contexts"]}
    cids = sorted(fam)
    field = {}
    for sp in sorted((bdir / "scores").glob("*.jsonl")):
        recs = [json.loads(x) for x in sp.read_text().splitlines() if x.strip()]
        field[sp.stem] = {r["cid"]: float(r["utility"]) for r in recs}

    P448 = Path("/home/ubuntu/jope_plan/evidence/H448/pulled")
    P418 = Path("/home/ubuntu/jope_plan/evidence/H418/pulled")
    tab_dir = Path("/home/ubuntu/jope-data/h450-quarantine/cols_tab_h450_D91")
    from cols_tab_h450 import H450_TAB_COLS

    bank_rows = {}
    with np.load("/home/ubuntu/jope-data/h450-quarantine/bank_d91_search_f64.npz", allow_pickle=False) as zb:
        b_cids = [str(x) for x in zb["cids"].tolist()]
        ss, sa = zb["s_shape"], zb["a_shape"]
        so = ao = lo = 0
        for i, c in enumerate(b_cids):
            n, w = int(ss[i, 0]), int(ss[i, 1])
            na, f = int(sa[i, 0]), int(sa[i, 1])
            S_mat = np.array(zb["search"][so : so + n * w], np.float64).reshape(n, w)
            Xa = np.array(zb["audit"][ao : ao + na * f], np.float64, copy=True).reshape(na, f)
            bank_rows[c] = (np.array(S_mat[:, :-1], np.float64, copy=True), Xa,
                            np.array(zb["y64"][lo : lo + n], np.float64, copy=True),
                            np.array(zb["lab"][lo : lo + n], np.int64, copy=True))
            so += n * w; ao += na * f; lo += n

    jobs = []
    for c in cids:
        Xs, Xa, y, lab = bank_rows[c]
        n_s, n_a = len(y), len(Xa)
        blk_m = _load_col_block(P448 / "cols_matern_D91" / f"{c}.npz", GM.PRIMARY_NAMES, n_s, n_a)
        blk_k_raw = _load_col_block(P448 / "cols_k3_D91" / f"{c}.npz", H430_K3, n_s, n_a)
        blk_k = None
        if blk_k_raw is not None:
            Qk, Bk, Rk = [arr.copy() for arr in blk_k_raw]
            y_lo, y_hi = float(np.min(y)), float(np.max(y))
            y_pad = 0.25 * max(y_hi - y_lo, float(np.std(y)), 1e-9)
            Qk[:, 0] = np.clip(Qk[:, 0], y_lo - y_pad, y_hi + y_pad)
            Bk[:, 0] = np.clip(Bk[:, 0], y_lo - y_pad, y_hi + y_pad)
            Rk[:, 0] = np.clip(Rk[:, 0], y_lo - y_pad, y_hi + y_pad)
            blk_k = (Qk, Bk, Rk)
        blk_f_raw = _load_col_block(P448 / "cols_f4_D91" / f"{c}.npz", H433_F4, n_s, n_a)
        blk_f = None
        if blk_f_raw is not None:
            Qf, Bf, Rf = [arr.copy() for arr in blk_f_raw]
            y_lo, y_hi = float(np.min(y)), float(np.max(y))
            y_pad = 0.50 * max(y_hi - y_lo, float(np.std(y)), 1e-9)
            for idx_f in (2, 3):
                Qf[:, idx_f] = np.clip(Qf[:, idx_f], y_lo - y_pad, y_hi + y_pad)
                Bf[:, idx_f] = np.clip(Bf[:, idx_f], y_lo - y_pad, y_hi + y_pad)
                Rf[:, idx_f] = np.clip(Rf[:, idx_f], y_lo - y_pad, y_hi + y_pad)
            blk_f = (Qf, Bf, Rf)
        blk_tab = _load_col_block(tab_dir / f"{c}.npz", H450_TAB_COLS, n_s, n_a)
        blk_fast = _load_col_block(cols_dir / f"{c}.npz", FAST_COLS, n_s, n_a)
        if blk_fast is None:
            raise SystemExit(f"missing fast 104M col block for {c}")
        for s in SEEDS:
            jobs.append((s, c, P418 / TAGS[s] / f"h416_{TAGS[s]}_npz" / f"{c}.npz", Xs, Xa, y, lab, blk_m, blk_k, blk_f, blk_tab, blk_fast))

    with Pool(processes=min(procs, len(jobs))) as pool:
        done = pool.map(_score_ctx_worker, jobs)

    u_ch_s = {s: {} for s in SEEDS}
    u_lg_s = {s: {} for s in SEEDS}
    u_ng_s = {s: {} for s in SEEDS}
    u_ic_s = {s: {} for s in SEEDS}
    for s, c, uc, ul, ung, ui in done:
        u_ch_s[s][c] = uc
        u_lg_s[s][c] = ul
        u_ng_s[s][c] = ung
        u_ic_s[s][c] = ui
    u_champ = {c: 0.5 * (u_ch_s[SEEDS[0]][c] + u_ch_s[SEEDS[1]][c]) for c in cids}
    u_leg = {c: 0.5 * (u_lg_s[SEEDS[0]][c] + u_lg_s[SEEDS[1]][c]) for c in cids}
    u_nogp = {c: u_ng_s[SEEDS[0]][c] for c in cids}
    u_icl = {c: u_ic_s[SEEDS[0]][c] for c in cids}

    rank_jobs = [
        ("champ", field, "h450.FAST_104M_EVERY_N_CHAMPION", u_champ, fam, cids, n_boot),
        ("leg", field, "h450.FAST_104M_LEGACY_GUARD_STACK", u_leg, fam, cids, n_boot),
        ("nogp", field, "h450.FAST_104M_NEURAL_PLUS_TAB_NO_GP", u_nogp, fam, cids, n_boot),
        ("icl", field, "h450.FAST_104M_PURE_ICL", u_icl, fam, cids, n_boot),
    ]
    with Pool(processes=min(4, procs)) as pool:
        ranked = dict(pool.map(_rank_job_worker, rank_jobs))

    doc = {
        "schema": "jope.h450.fast_every_n_dev91/v2",
        "quarantined_in_corpus_random_split_diagnostic": True,
        "canonical_promotion_forbidden": True,
        "in_corpus_random_split_champion_91": ranked["champ"],
        "in_corpus_legacy_guard_stack_91": ranked["leg"],
        "in_corpus_icl_boost_only_91": ranked["icl"],
        "in_corpus_neural_plus_tab_no_gp_91": ranked["nogp"],
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(doc, indent=2, sort_keys=True, default=lambda o: o.item()) + "\n")
    return doc


def _parse_indices(spec: str) -> list[int]:
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo_s, hi_s = part.split("-", 1)
            out.extend(range(int(lo_s), int(hi_s) + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cmd", choices=("ctx", "shard", "score"), required=True)
    ap.add_argument("--ctx-idx", type=int, default=0)
    ap.add_argument("--indices", default="0-90")
    ap.add_argument("--bank", default="/home/ubuntu/jope-data/h450-quarantine/bank_d91_search_f64.npz")
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--cols-dir", required=True)
    ap.add_argument("--out-json", default="")
    ap.add_argument("--n-views", type=int, default=4)
    ap.add_argument("--procs", type=int, default=16)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--cpu-hard-pct", type=float, default=83.0)
    a = ap.parse_args()

    if a.cmd == "ctx":
        run_single_context(a.ctx_idx, Path(a.bank), Path(a.ckpt), Path(a.cols_dir), n_views=a.n_views)
    elif a.cmd == "shard":
        n_done = run_shard(
            _parse_indices(a.indices), Path(a.bank), Path(a.ckpt), Path(a.cols_dir),
            n_views=a.n_views, procs=a.procs, cpu_hard_pct=a.cpu_hard_pct, threads=a.threads,
        )
        print(json.dumps({"shard_completed": n_done}, sort_keys=True))
    else:
        doc = score_fast_cols(Path(a.cols_dir), Path(a.out_json), procs=a.procs)
        ch = doc["in_corpus_random_split_champion_91"]
        lg = doc["in_corpus_legacy_guard_stack_91"]
        ic = doc["in_corpus_icl_boost_only_91"]
        print(json.dumps({
            "legacy_elo": round(float(lg["elo"]["elo"]), 1),
            "champion_elo": round(float(ch["elo"]["elo"]), 1),
            "pure_104m_icl_elo": round(float(ic["elo"]["elo"]), 1),
            "vs_kumo_L": round(float(ch["gap_elo"]["vs_kumo_tabular_large_e8.v1"]), 1),
            "vs_h448_fair": round(float(ch["gap_elo"]["vs_h448.APEX_FAIR_CHAMPION"]), 1),
        }, sort_keys=True))


if __name__ == "__main__":
    main()
