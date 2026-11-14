#!/usr/bin/env python3
"""H452 Multi-Trajectory `e8` `104M` Prediction Ensemble + `2xCV-7` `CW-1/CW-2` + Rank-Gauss + GP-Residual
Hybrid Column Generator (`cols_104m_e8_gpresid_h452.py`).

Strict zero-leakage contract (`assert "ya" not in zb.files`):
  - Every key encoding (`CW-1`), raw pair formula (`CW-2`), relational structure feature (`REL`), Yeo-Johnson warp,
    Rank-Gauss monotone map, Robust-Ridge base, and GP-Residual anchor is fitted 100% on `(X_tr, y_tr)` inside each
    fold (or `(Xs, ys)` on full refit) — NEVER reading `y_audit` or fitting any statistic on `X_audit`!
  - Supports two composable commands:
      1. `--cmd build-traj-e8`: Builds prediction-space multi-trajectory `e8` columns (`traj_robridge_e8`,
         `traj_yj_e8`, `traj_nlres_e8`, `traj_boost2s_e8`, `traj_opt_e8`) from distinct `104M` GPU runs.
      2. `--cmd eval-cv7`: Evaluates 4-8 distinct `104M` checkpoints on `2xCV-7` (`14` stratified OOF folds) with
         `[X_clean, Keys_CW1, Formulas_CW2, REL]` and GP-Winner OOF residual conditioning (`icl_e8_cw12_robridge_cv7`,
         `icl_e8_cw12_yj_cv7`, `icl_e8_rankgauss_duan_cv7`, `icl_e8_gp_resid_cv7`).
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import itertools
import json
import math
import multiprocessing as mp
import os
import sys
import time
import warnings
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
from scipy.special import ndtri
from scipy.stats import rankdata
import torch

torch.set_num_threads(2)

HERE = Path(__file__).resolve().parent
for _cand in (
    str(HERE),
    "/home/ubuntu/h450-stage/pub-r6/code",
    "/home/ubuntu/h448-stage/pub-r1/code",
):
    if Path(_cand).is_dir() and _cand not in sys.path:
        sys.path.append(_cand)

import formula_core as FC  # noqa: E402
import keys_core as KC  # noqa: E402
import relational_struct_h450 as REL  # noqa: E402
import robust_resid_h450 as RR  # noqa: E402
from icl_model import FID_SLOTS  # noqa: E402
from icl_model_h450 import ICLH450  # noqa: E402
from stack_guard_cv7_h452 import bcar_weights_pool_agnostic, make_stratified_kfold  # noqa: E402

WIDE_GPU = 64

TRAJ_E8_COLS = (
    "traj_robridge_e8",
    "traj_yj_e8",
    "traj_nlres_e8",
    "traj_boost2s_e8",
    "traj_opt_e8",
)

CV7_HYBRID_COLS = (
    "icl_e8_cw12_robridge_cv7",
    "icl_e8_cw12_yj_cv7",
    "icl_e8_rankgauss_duan_cv7",
    "icl_e8_gp_resid_cv7",
)


def _sha_seed(tag: str) -> int:
    return int.from_bytes(hashlib.blake2b(tag.encode("utf-8"), digest_size=8).digest(), "little") & 0x7FFFFFFF


def _extract_state_dict(ck: dict, ckpt_key: str = "auto") -> dict:
    if ckpt_key == "ema_soup50" and ck.get("ema_cooldown") is not None and ck.get("ema_swa") is not None:
        cd = ck["ema_cooldown"]
        sw = ck["ema_swa"]
        return {k: (0.5 * cd[k].float() + 0.5 * sw[k].float()).to(dtype=cd[k].dtype) for k in cd}
    if ckpt_key not in ("", "auto") and ck.get(ckpt_key) is not None:
        return ck[ckpt_key]
    if ck.get("ema_cooldown") is not None and ck.get("ema_swa") is not None and ckpt_key == "auto":
        cd = ck["ema_cooldown"]
        sw = ck["ema_swa"]
        return {k: (0.5 * cd[k].float() + 0.5 * sw[k].float()).to(dtype=cd[k].dtype) for k in cd}
    return ck.get("ema_cooldown") or ck.get("ema_valley") or ck.get("ema_swa") or ck.get("ema") or ck["model"]


def load_icl_ckpt(
    ckpt_p: Path,
    size: str,
    dev: torch.device,
    ckpt_key: str = "auto",
    cpu_ckpt_cache: dict[str, dict] | None = None,
) -> ICLH450:
    rp = str(ckpt_p.resolve())
    if cpu_ckpt_cache is not None and rp in cpu_ckpt_cache:
        ck = cpu_ckpt_cache[rp]
    else:
        ck = torch.load(ckpt_p, map_location="cpu", weights_only=False)
        if cpu_ckpt_cache is not None:
            cpu_ckpt_cache[rp] = ck
    sd = _extract_state_dict(ck, ckpt_key=ckpt_key)
    m = ICLH450(size).to(dev).eval()
    m.load_state_dict(sd, strict=True)
    for p in m.parameters():
        p.requires_grad_(False)
    return m


class RankGaussDuanWarpF64:
    """Monotone Rank-Gauss (`Phi^-1((rank - 0.5)/n)`) target warp fitted strictly on `y_tr` with exact 1,024-bin inverse."""

    def __init__(self) -> None:
        self.z_knots: np.ndarray | None = None
        self.y_knots: np.ndarray | None = None
        self.slope_lo: float = 1.0
        self.slope_hi: float = 1.0

    def fit_transform(self, y_tr: np.ndarray) -> np.ndarray:
        y_tr = np.asarray(y_tr, dtype=np.float64)
        n = len(y_tr)
        u = np.clip((rankdata(y_tr, method="average") - 0.5) / float(max(n, 1)), 1e-6, 1.0 - 1e-6)
        z = ndtri(u)
        order = np.argsort(z, kind="mergesort")
        zs, ys = z[order], y_tr[order]
        zu, idx = np.unique(zs, return_index=True)
        # Group mean at tied ranks preserves monotonicity
        yu = np.array([float(np.mean(ys[zs == val])) for val in zu], dtype=np.float64)
        yu = np.maximum.accumulate(yu)
        if len(zu) >= 3:
            dz_lo = max(float(zu[min(2, len(zu) - 1)] - zu[0]), 1e-6)
            dz_hi = max(float(zu[-1] - zu[max(0, len(zu) - 3)]), 1e-6)
            self.slope_lo = max(0.0, float(yu[min(2, len(yu) - 1)] - yu[0]) / dz_lo)
            self.slope_hi = max(0.0, float(yu[-1] - yu[max(0, len(yu) - 3)]) / dz_hi)
        else:
            sd = max(float(np.std(y_tr)), 1e-9)
            self.slope_lo = sd
            self.slope_hi = sd
        self.z_knots = zu
        self.y_knots = yu
        return z

    def inv_torch(self, z_grid: torch.Tensor) -> torch.Tensor:
        assert self.z_knots is not None and self.y_knots is not None
        zg_np = z_grid.detach().cpu().numpy().astype(np.float64)
        flat = zg_np.reshape(-1)
        yk = np.interp(flat, self.z_knots, self.y_knots)
        lo_m = flat < self.z_knots[0]
        hi_m = flat > self.z_knots[-1]
        if np.any(lo_m):
            yk[lo_m] = self.y_knots[0] + self.slope_lo * (flat[lo_m] - self.z_knots[0])
        if np.any(hi_m):
            yk[hi_m] = self.y_knots[-1] + self.slope_hi * (flat[hi_m] - self.z_knots[-1])
        return torch.from_numpy(yk.reshape(zg_np.shape)).to(device=z_grid.device, dtype=torch.float64)


def extract_cw1_cw2_features(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_ev: np.ndarray,
    seed: int,
    max_keys: int = 4,
    max_forms: int = 6,
) -> tuple[np.ndarray, np.ndarray]:
    """Extracts strictly fit-row-only `CW-1` exact key encodings + `CW-2` raw pair formulas (`0` query leakage!)."""
    n_tr, p = X_tr.shape
    n_ev = X_ev.shape[0]
    tr_cols: list[np.ndarray] = []
    ev_cols: list[np.ndarray] = []

    if n_tr >= 24 and p >= 1 and float(np.std(y_tr)) > 1e-12:
        # 1. CW-1 Exact single & pair keys (nested 5-fold OOF on X_tr, lookup on X_ev)
        try:
            key_specs = KC.fit_keys(X_tr, y_tr)
            for ki, sp in enumerate(key_specs[:max_keys]):
                if float(sp["u"]) < 0.08:
                    continue
                enc_tr, cnt_tr = KC.nested_encode(sp, X_tr, y_tr, seed + 17 * (ki + 1))
                st = KC.make_stat(sp, X_tr, y_tr)
                enc_ev = KC.apply_stat(st, X_ev)
                cnt_ev = KC.apply_count(st, X_ev)
                if np.all(np.isfinite(enc_tr)) and np.all(np.isfinite(enc_ev)):
                    tr_cols.append(enc_tr[:, None])
                    ev_cols.append(enc_ev[:, None])
                    tr_cols.append(np.log1p(np.maximum(cnt_tr, 0.0))[:, None])
                    ev_cols.append(np.log1p(np.maximum(cnt_ev, 0.0))[:, None])
        except (ValueError, RuntimeError, ArithmeticError, LookupError):
            pass

        # 2. CW-2 Raw-scale arithmetic pair formulas (`a/b`, `a*b`, `a-b`, `ln|a|-ln|b|`)
        try:
            # Impute fit medians strictly from X_tr before formula screening
            med_tr = np.zeros(p, dtype=np.float64)
            for j in range(p):
                fin = np.isfinite(X_tr[:, j])
                med_tr[j] = float(np.median(X_tr[fin, j])) if np.any(fin) else 0.0
            X_tr_imp = np.where(np.isfinite(X_tr), X_tr, med_tr[None, :])
            X_ev_imp = np.where(np.isfinite(X_ev), X_ev, med_tr[None, :])
            surv, _, _ = FC.screen(X_tr_imp, y_tr, seed + 91)
            for s in surv[:max_forms]:
                ft_tr = FC.compute(s["spec"], X_tr_imp)
                ft_ev = FC.compute(s["spec"], X_ev_imp)
                if np.all(np.isfinite(ft_tr)) and np.all(np.isfinite(ft_ev)) and float(np.std(ft_tr)) > 1e-12:
                    tr_cols.append(ft_tr[:, None])
                    ev_cols.append(ft_ev[:, None])
        except (ValueError, RuntimeError, ArithmeticError, LookupError):
            pass

    if not tr_cols:
        return np.zeros((n_tr, 0), dtype=np.float64), np.zeros((n_ev, 0), dtype=np.float64)
    return np.hstack(tr_cols), np.hstack(ev_cols)


def _build_views(
    X_tr: np.ndarray, y_tr: np.ndarray, X_ev: np.ndarray, n_views: int, seed: int, keep_first: int = 0
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Builds `n_views` feature permutations/subsets strictly ranked on `(X_tr, y_tr)` (`keep_first` pinned cols)."""
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

    pinned = list(range(min(keep_first, WIDE_GPU // 4)))
    rem_all = [int(j) for j in np.argsort(-scores) if int(j) not in pinned]
    for v in range(n_views):
        if v == 0:
            keep = np.array(pinned + rem_all[: WIDE_GPU - len(pinned)], dtype=np.int64)
        else:
            n_top = max(1, int(round(0.65 * (WIDE_GPU - len(pinned)))))
            top = rem_all[:n_top]
            tail = rem_all[n_top:]
            n_pick = WIDE_GPU - len(pinned) - len(top)
            pick = rng.choice(tail, size=n_pick, replace=False).tolist() if n_pick > 0 and len(tail) >= n_pick else tail[:n_pick]
            keep = rng.permutation(np.array(pinned + top + pick, dtype=np.int64))
        views.append((X_tr[:, keep], X_ev[:, keep]))
    return views


@torch.no_grad()
def _batched_icl_predict(
    models_and_weights: list[tuple[ICLH450, float]],
    views: list[tuple[np.ndarray, np.ndarray]],
    y_tr: np.ndarray,
    dev: torch.device,
    seed: int,
    inv_fn=None,
) -> np.ndarray:
    V = len(views)
    n_tr = views[0][0].shape[0]
    n_ev = views[0][1].shape[0]
    Fn = views[0][0].shape[1]
    if n_ev == 0:
        return np.zeros(0, dtype=np.float64)

    xb_np = np.stack([np.vstack([vt, ve]) for vt, ve in views], axis=0)
    yb_np = np.broadcast_to(np.asarray(y_tr, dtype=np.float64)[None, :], (V, n_tr)).copy()
    xt = torch.from_numpy(xb_np).to(device=dev, dtype=torch.float64)
    yt = torch.from_numpy(yb_np).to(device=dev, dtype=torch.float64)

    g = torch.Generator(device="cpu").manual_seed(seed)
    slots = torch.randperm(FID_SLOTS, generator=g)[:Fn].to(device=dev)

    pred_acc = np.zeros(n_ev, dtype=np.float64)
    w_sum = 0.0
    v_chunk = 4
    for mdl, w in models_and_weights:
        v_sum = np.zeros(n_ev, dtype=np.float64)
        for v0 in range(0, V, v_chunk):
            v1 = min(V, v0 + v_chunk)
            logits, ym, ys = mdl(xt[v0:v1], yt[v0:v1], n_tr, slots=slots, amp=(dev.type == "cuda"))
            if inv_fn is None:
                out_v = mdl.mean_f64(logits, ym, ys)
            else:
                out_v = mdl.mean_warped(logits, ym, ys, inv_fn)
            v_sum += out_v.sum(dim=0).detach().cpu().numpy()
        pred_acc += float(w) * (v_sum / float(V))
        w_sum += float(w)
    return pred_acc / max(w_sum, 1e-12)


def build_trajectory_e8_dir(
    bank_p: Path,
    traj_dirs: list[Path],
    out_dir: Path,
) -> dict:
    """Aggregates diverse `104M` GPU column trajectories in prediction space (`0` `y_audit` read!)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with np.load(bank_p, allow_pickle=False) as zb:
        assert "ya" not in zb.files, "audit labels forbidden!"
        cids = [str(x) for x in zb["cids"].tolist()]
        ss, sa = zb["s_shape"], zb["a_shape"]
        ys_map = {}
        lab_map = {}
        na_map = {}
        lo = 0
        for i, c in enumerate(cids):
            n = int(ss[i, 0])
            na_map[c] = int(sa[i, 0])
            ys_map[c] = np.array(zb["y64"][lo : lo + n], np.float64, copy=True)
            lab_map[c] = np.array(zb["lab"][lo : lo + n], np.int64, copy=True)
            lo += n

    src_modes = (
        ("traj_robridge_e8", "icl_robridge_e32"),
        ("traj_yj_e8", "icl_yj_exact_e16"),
        ("traj_nlres_e8", "icl_nlres_e16"),
        ("traj_boost2s_e8", "icl_boost2s_e16"),
    )

    built = 0
    for c in cids:
        y = ys_map[c]
        lab = lab_map[c]
        n_s, n_a = len(y), na_map[c]
        lab7 = make_stratified_kfold(c, y, k=7, rep=0) if n_s >= 35 else lab

        loaded = []
        for td in traj_dirs:
            p = td / f"{c}.npz"
            if p.is_file():
                z = np.load(p, allow_pickle=False)
                if bool(z["ok"]):
                    loaded.append(z)
        if not loaded:
            raise RuntimeError(f"no trajectory files found for {c}")

        out_arrays: dict[str, np.ndarray] = {"ok": np.array(True)}
        all_q_opt = []
        all_b_opt = []
        all_r_opt = []

        for dst_name, src_name in src_modes:
            qs = [np.asarray(z[f"q_{src_name}"], np.float64) for z in loaded if f"q_{src_name}" in z.files]
            bs = [np.asarray(z[f"bag_{src_name}"], np.float64) for z in loaded if f"bag_{src_name}" in z.files]
            rs = [np.asarray(z[f"ref_{src_name}"], np.float64) for z in loaded if f"ref_{src_name}" in z.files]
            Q_m = np.column_stack(qs)
            B_m = np.column_stack(bs)
            R_m = np.column_stack(rs)
            # Variance-reducing equal-trajectory soup for each mode
            out_arrays[f"q_{dst_name}"] = np.mean(Q_m, axis=1)
            out_arrays[f"bag_{dst_name}"] = np.mean(B_m, axis=1)
            out_arrays[f"ref_{dst_name}"] = np.mean(R_m, axis=1)
            all_q_opt.append(Q_m)
            all_b_opt.append(B_m)
            all_r_opt.append(R_m)

        # Also include `icl_robridge_a` and `icl_robridge_b` from each trajectory in the OOF-weighted `traj_opt_e8` pool
        for sub_m in ("icl_robridge_a", "icl_robridge_b"):
            qs = [np.asarray(z[f"q_{sub_m}"], np.float64) for z in loaded if f"q_{sub_m}" in z.files]
            bs = [np.asarray(z[f"bag_{sub_m}"], np.float64) for z in loaded if f"bag_{sub_m}" in z.files]
            rs = [np.asarray(z[f"ref_{sub_m}"], np.float64) for z in loaded if f"ref_{sub_m}" in z.files]
            if qs:
                all_q_opt.append(np.column_stack(qs))
                all_b_opt.append(np.column_stack(bs))
                all_r_opt.append(np.column_stack(rs))

        Q_pool = np.hstack(all_q_opt)
        B_pool = np.hstack(all_b_opt)
        R_pool = np.hstack(all_r_opt)
        # Nested CV-7 OOF-selected trajectory weights (`0` self-overfit on `q_traj_opt_e8`!)
        q_opt_oof = np.zeros(n_s, dtype=np.float64)
        uniq_f = sorted(int(k) for k in np.unique(lab7))
        for fk in uniq_f:
            va = lab7 == fk
            tr = ~va
            wf = bcar_weights_pool_agnostic(Q_pool[tr], y[tr], f"{c}_topt_f{fk}", "fold_log", lab7[tr], n_bags=16)
            q_opt_oof[va] = Q_pool[va] @ wf
        w_full = bcar_weights_pool_agnostic(Q_pool, y, f"{c}_topt_full", "fold_log", lab7, n_bags=24)
        out_arrays["q_traj_opt_e8"] = q_opt_oof
        out_arrays["bag_traj_opt_e8"] = B_pool @ w_full
        out_arrays["ref_traj_opt_e8"] = R_pool @ w_full

        tmp = out_dir / f".{c}.{os.getpid()}.tmp.npz"
        np.savez(tmp, **out_arrays)
        os.replace(tmp, out_dir / f"{c}.npz")
        built += 1

    return {"built": built, "n_trajectories": len(traj_dirs), "cols": list(TRAJ_E8_COLS)}


def _load_gp_anchor_for_ctx(
    cid: str,
    y: np.ndarray,
    lab: np.ndarray,
    na: int,
    gp_anchor_dir: Path | None,
    p418_dir: Path,
    p448_dir: Path,
    gp_oof_cache_dir: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    """Builds a strictly SEARCH-only OOF GP/Symbolic anchor `(q_gp_anc, bag_gp_anc, ref_gp_anc)` (`0` `ya` read!)."""
    cache_file: Path | None = None
    if gp_oof_cache_dir is not None:
        gp_sig = str(gp_anchor_dir.resolve()) if gp_anchor_dir is not None else "none"
        sig_hex = f"{_sha_seed(f'gpanc_v1|{gp_sig}|{p418_dir.resolve()}|{p448_dir.resolve()}'):08x}"
        cache_file = gp_oof_cache_dir / f"{sig_hex}_{cid}.npz"
        if cache_file.is_file():
            try:
                with np.load(cache_file, allow_pickle=False) as zc:
                    q_c = np.array(zc["q_anc"], dtype=np.float64, copy=True)
                    b_c = np.array(zc["bag_anc"], dtype=np.float64, copy=True)
                    r_c = np.array(zc["ref_anc"], dtype=np.float64, copy=True)
                    u_c = bool(zc["usable"])
                    if len(q_c) == len(y) and len(b_c) == na and len(r_c) == na:
                        return q_c, b_c, r_c, u_c
            except (OSError, ValueError, KeyError):
                pass

    ym = float(np.mean(y))
    sy = max(float(np.std(y)), 1e-12)
    q_cols: list[np.ndarray] = []
    b_cols: list[np.ndarray] = []
    r_cols: list[np.ndarray] = []

    # 1. Optional Ultra-GP directory (preferring unseen-partition `Q_cons / AKF_cons / PA_cons` when available!)
    cand_ugp = None
    if gp_anchor_dir is not None:
        if (gp_anchor_dir / f"{cid}.npz").is_file():
            cand_ugp = gp_anchor_dir / f"{cid}.npz"
        elif (gp_anchor_dir.parent / "g500" / f"{cid}.npz").is_file():
            cand_ugp = gp_anchor_dir.parent / "g500" / f"{cid}.npz"
    if cand_ugp is not None:
        zu = np.load(cand_ugp, allow_pickle=False)
        if "Q_cons" in zu.files and zu["Q_cons"].shape[1] > 0:
            Qu = np.asarray(zu["Q_cons"], np.float64)
            Bu = np.asarray(zu["AKF_cons"], np.float64)
            Ru = np.asarray(zu["PA_cons"], np.float64)
        elif "Q_cv7" in zu.files and zu["Q_cv7"].shape[1] > 0:
            Qu = np.asarray(zu["Q_cv7"], np.float64)
            Bu = np.asarray(zu["AKF_cv7"], np.float64)
            Ru = np.asarray(zu["PA"], np.float64)
        else:
            Qu = np.zeros((len(y), 0), dtype=np.float64)
            Bu = np.zeros((na, 0), dtype=np.float64)
            Ru = np.zeros((na, 0), dtype=np.float64)
        for j in range(Qu.shape[1]):
            if np.all(np.isfinite(Qu[:, j])) and np.all(np.isfinite(Bu[:, j])):
                q_cols.append(Qu[:, j])
                b_cols.append(Bu[:, j])
                r_cols.append(Ru[:, j] if np.all(np.isfinite(Ru[:, j])) else Bu[:, j])

    # 2. Legacy H418 GP pools (D1 & D2: read ONLY `Q`, `AKF`, `PA`, `pa_ok` — NEVER `ya`!)
    for tag in ("D1", "D2"):
        p16 = p418_dir / tag / f"h416_{tag}_npz" / f"{cid}.npz"
        if p16.is_file():
            with np.load(p16, allow_pickle=False) as z16:
                Q16 = np.asarray(z16["Q"], np.float64)
                AKF16 = np.asarray(z16["AKF"], np.float64)
                PA16 = np.asarray(z16["PA"], np.float64)
                pa_ok = np.asarray(z16["pa_ok"], bool)
                for j in range(Q16.shape[1]):
                    if np.all(np.isfinite(Q16[:, j])) and np.all(np.isfinite(AKF16[:, j])):
                        q_cols.append(Q16[:, j])
                        b_cols.append(AKF16[:, j])
                        r_cols.append(PA16[:, j] if (pa_ok[j] and np.all(np.isfinite(PA16[:, j]))) else AKF16[:, j])

    # 3. Also include M7 + K3 + F4 non-neural columns (`0` `ya` in these files)
    for sub, names in (
        ("cols_matern_D91", ("m32_ard", "m52_ard", "rq_ard", "m32_lin", "lap_ard", "m32_iso", "hyb_oof")),
        ("cols_k3_D91", ("key_direct", "key_gbm", "factor_glm")),
        ("cols_f4_D91", ("form_gbm", "form_gam", "form_sr", "form_interp")),
    ):
        pf = p448_dir / sub / f"{cid}.npz"
        if pf.is_file():
            zf = np.load(pf, allow_pickle=False)
            if bool(zf["ok"]):
                for nm in names:
                    if f"q_{nm}" in zf.files:
                        qj = np.asarray(zf[f"q_{nm}"], np.float64)
                        bj = np.asarray(zf[f"bag_{nm}"], np.float64)
                        rj = np.asarray(zf[f"ref_{nm}"], np.float64)
                        if np.all(np.isfinite(qj)) and np.all(np.isfinite(bj)):
                            y_lo, y_hi = float(np.min(y)), float(np.max(y))
                            pad = 0.5 * max(y_hi - y_lo, sy)
                            q_cols.append(np.clip(qj, y_lo - pad, y_hi + pad))
                            b_cols.append(np.clip(bj, y_lo - pad, y_hi + pad))
                            r_cols.append(np.clip(rj, y_lo - pad, y_hi + pad) if np.all(np.isfinite(rj)) else np.clip(bj, y_lo - pad, y_hi + pad))

    def _persist_anc(qa: np.ndarray, ba: np.ndarray, ra: np.ndarray, ok_flag: bool) -> None:
        if cache_file is not None:
            try:
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                tmp_c = cache_file.parent / f".{cache_file.stem}.{os.getpid()}.tmp.npz"
                np.savez(tmp_c, q_anc=qa, bag_anc=ba, ref_anc=ra, usable=np.array(ok_flag))
                os.replace(tmp_c, cache_file)
            except (OSError, ValueError):
                pass

    if not q_cols:
        q0, b0, r0 = np.full(len(y), ym), np.full(na, ym), np.full(na, ym)
        _persist_anc(q0, b0, r0, False)
        return q0, b0, r0, False

    Q_m = np.column_stack(q_cols)
    B_m = np.column_stack(b_cols)
    R_m = np.column_stack(r_cols)
    lab7 = make_stratified_kfold(cid, y, k=7, rep=0) if len(y) >= 35 else lab

    # Strictly nested OOF anchor on `Q_m` so `q_gp_anc` has ZERO meta-weight in-sample bias!
    q_anc = np.zeros(len(y), dtype=np.float64)
    for fk in sorted(int(k) for k in np.unique(lab7)):
        va = lab7 == fk
        tr = ~va
        wf = bcar_weights_pool_agnostic(Q_m[tr], y[tr], f"{cid}_gpanc_f{fk}", "fold_log", lab7[tr], n_bags=12)
        q_anc[va] = Q_m[va] @ wf
    w_full = bcar_weights_pool_agnostic(Q_m, y, f"{cid}_gpanc_full", "fold_log", lab7, n_bags=16)
    bag_anc = B_m @ w_full
    ref_anc = R_m @ w_full
    rmse_anc = float(np.sqrt(np.mean((q_anc - y) ** 2)))
    usable = bool(rmse_anc < 0.98 * sy and float(np.std(y - q_anc)) >= 1e-4 * sy)
    _persist_anc(q_anc, bag_anc, ref_anc, usable)
    return q_anc, bag_anc, ref_anc, usable


def _init_prep_worker() -> None:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    torch.set_num_threads(1)


def _prep_full_context_worker(args_tuple: tuple) -> dict:
    """Pure-CPU context preparation (`0` `ya` read!): OOF GP anchor + all 14 `2xCV-7` folds + full-search refit views."""
    (
        idx,
        cid,
        Xs,
        Xa,
        y,
        lab,
        gp_anchor_dir,
        p418_dir,
        p448_dir,
        gp_oof_cache_dir,
        n_views,
        n_repeats,
        k_folds,
    ) = args_tuple
    Xs = np.asarray(Xs, dtype=np.float64)
    Xa = np.asarray(Xa, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    lab = np.asarray(lab, dtype=np.int64)
    n_s, n_a = len(y), Xa.shape[0]
    ym_glob = float(np.mean(y))
    sy_glob = max(float(np.std(y)), 1e-12)
    y_lo, y_hi = float(np.min(y)), float(np.max(y))
    span = max(y_hi - y_lo, sy_glob)
    lo_guard, hi_guard = y_lo - 1.5 * span, y_hi + 1.5 * span

    q_gp_anc, bag_gp_anc, ref_gp_anc, gp_anc_ok = _load_gp_anchor_for_ctx(
        cid, y, lab, n_a, gp_anchor_dir, p418_dir, p448_dir, gp_oof_cache_dir=gp_oof_cache_dir
    )

    eff_k = k_folds if n_s >= 35 else 5
    fold_tasks = []
    for rep in range(n_repeats):
        lab_rep = make_stratified_kfold(cid, y, k=eff_k, rep=rep) if eff_k != 5 or rep > 0 else lab
        for fk in range(eff_k):
            va = lab_rep == fk
            tr = ~va
            n_va = int(np.sum(va))
            if n_va == 0 or int(np.sum(tr)) < 16:
                continue
            fold_tasks.append((rep, fk, va, tr, n_va))

    prepped_folds = []
    for rep, fk, va, tr, n_va in fold_tasks:
        X_tr_raw, y_tr = Xs[tr], y[tr]
        X_ev_raw = np.vstack([Xs[va], Xa])
        fseed = _sha_seed(f"h452_cv7:{cid}:r{rep}:f{fk}")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            X_tr_cl, X_ev_cl, S_tr, S_ev, _ = REL.build_h450_struct_cols(
                X_tr_raw, y_tr, X_ev_raw, seed=fseed, fast_mode=True, max_total_cols=14, include_base_struct=True
            )
        CW_tr, CW_ev = extract_cw1_cw2_features(X_tr_raw, y_tr, X_ev_raw, seed=fseed + 7)
        X_tr_aug = np.hstack([p for p in (CW_tr, X_tr_cl, S_tr) if p.shape[1] > 0])
        X_ev_aug = np.hstack([p for p in (CW_ev, X_ev_cl, S_ev) if p.shape[1] > 0])
        n_pin = CW_tr.shape[1]

        oof_rr, pq_rr, used_rr, _ = RR.robust_ridge_np(X_tr_aug, y_tr, X_ev_aug, seed=fseed + 11)
        y_tr_rr = (y_tr - oof_rr) if used_rr else y_tr
        base_rr = pq_rr if used_rr else np.zeros(n_va + n_a, dtype=np.float64)
        views_rr = _build_views(X_tr_aug, y_tr_rr, X_ev_aug, n_views=n_views, seed=fseed + 101, keep_first=n_pin)

        warp_yj = RR.YeoJohnsonWarpF64()
        z_yj = warp_yj.fit_transform(y_tr)
        views_yj = _build_views(X_tr_aug, z_yj, X_ev_aug, n_views=n_views, seed=fseed + 202, keep_first=n_pin)

        warp_rg = RankGaussDuanWarpF64()
        z_rg = warp_rg.fit_transform(y_tr)
        views_rg = _build_views(X_tr_aug, z_rg, X_ev_aug, n_views=n_views, seed=fseed + 303, keep_first=n_pin)

        gp_pack = None
        if gp_anc_ok:
            gp_tr = q_gp_anc[tr]
            gp_ev = np.concatenate([q_gp_anc[va], bag_gp_anc])
            r_tr_gp = y_tr - gp_tr
            X_tr_gp = np.column_stack([gp_tr, X_tr_aug])
            X_ev_gp = np.column_stack([gp_ev, X_ev_aug])
            views_gp = _build_views(X_tr_gp, r_tr_gp, X_ev_gp, n_views=n_views, seed=fseed + 404, keep_first=1 + n_pin)
            gp_pack = (gp_ev, r_tr_gp, views_gp)

        prepped_folds.append(
            (
                va,
                n_va,
                fseed,
                base_rr,
                y_tr_rr,
                views_rr,
                warp_yj,
                z_yj,
                views_yj,
                warp_rg,
                z_rg,
                views_rg,
                gp_pack,
            )
        )

    # Full-`X_search` Refit views on CPU
    rseed = _sha_seed(f"h452_cv7:{cid}:refit")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        Xs_cl, Xa_cl, Ss_f, Sa_f, _ = REL.build_h450_struct_cols(
            Xs, y, Xa, seed=rseed, fast_mode=True, max_total_cols=14, include_base_struct=True
        )
    CWs_f, CWa_f = extract_cw1_cw2_features(Xs, y, Xa, seed=rseed + 7)
    Xs_aug = np.hstack([p for p in (CWs_f, Xs_cl, Ss_f) if p.shape[1] > 0])
    Xa_aug = np.hstack([p for p in (CWa_f, Xa_cl, Sa_f) if p.shape[1] > 0])
    n_pin_f = CWs_f.shape[1]

    oof_rr_f, pq_rr_f, used_rr_f, _ = RR.robust_ridge_np(Xs_aug, y, Xa_aug, seed=rseed + 11)
    y_rr_f = (y - oof_rr_f) if used_rr_f else y
    base_rr_f = pq_rr_f if used_rr_f else np.zeros(n_a, dtype=np.float64)
    v_rr_f = _build_views(Xs_aug, y_rr_f, Xa_aug, n_views=n_views, seed=rseed + 101, keep_first=n_pin_f)

    warp_yj_f = RR.YeoJohnsonWarpF64()
    z_yj_f = warp_yj_f.fit_transform(y)
    v_yj_f = _build_views(Xs_aug, z_yj_f, Xa_aug, n_views=n_views, seed=rseed + 202, keep_first=n_pin_f)

    warp_rg_f = RankGaussDuanWarpF64()
    z_rg_f = warp_rg_f.fit_transform(y)
    v_rg_f = _build_views(Xs_aug, z_rg_f, Xa_aug, n_views=n_views, seed=rseed + 303, keep_first=n_pin_f)

    ref_gp_pack = None
    if gp_anc_ok:
        r_gp_f = y - q_gp_anc
        Xs_gp_f = np.column_stack([q_gp_anc, Xs_aug])
        Xa_gp_f = np.column_stack([ref_gp_anc, Xa_aug])
        v_gp_f = _build_views(Xs_gp_f, r_gp_f, Xa_gp_f, n_views=n_views, seed=rseed + 404, keep_first=1 + n_pin_f)
        ref_gp_pack = (r_gp_f, v_gp_f)

    return {
        "idx": idx,
        "cid": cid,
        "y": y,
        "n_s": n_s,
        "n_a": n_a,
        "ym_glob": ym_glob,
        "lo_guard": lo_guard,
        "hi_guard": hi_guard,
        "eff_k": eff_k,
        "q_gp_anc": q_gp_anc,
        "bag_gp_anc": bag_gp_anc,
        "ref_gp_anc": ref_gp_anc,
        "gp_anc_ok": gp_anc_ok,
        "prepped_folds": prepped_folds,
        "rseed": rseed,
        "base_rr_f": base_rr_f,
        "y_rr_f": y_rr_f,
        "v_rr_f": v_rr_f,
        "warp_yj_f": warp_yj_f,
        "z_yj_f": z_yj_f,
        "v_yj_f": v_yj_f,
        "warp_rg_f": warp_rg_f,
        "z_rg_f": z_rg_f,
        "v_rg_f": v_rg_f,
        "ref_gp_pack": ref_gp_pack,
    }


def _eval_prepped_context_gpu(
    prepped: dict,
    models_w: list[tuple[ICLH450, float]],
    dev: torch.device,
    abort_check=None,
) -> dict[str, np.ndarray] | None:
    """Evaluates one `(models_w)` spec on a pre-computed CPU context pack (`0` redundant CPU feature/anchor work!)."""
    if abort_check is not None and abort_check():
        return None
    cid = prepped["cid"]
    y = prepped["y"]
    n_s = prepped["n_s"]
    n_a = prepped["n_a"]
    ym_glob = prepped["ym_glob"]
    lo_guard = prepped["lo_guard"]
    hi_guard = prepped["hi_guard"]
    eff_k = prepped["eff_k"]
    q_gp_anc = prepped["q_gp_anc"]
    bag_gp_anc = prepped["bag_gp_anc"]
    ref_gp_anc = prepped["ref_gp_anc"]
    gp_anc_ok = prepped["gp_anc_ok"]

    q_sums = {c: np.zeros(n_s, dtype=np.float64) for c in CV7_HYBRID_COLS}
    q_cnts = {c: np.zeros(n_s, dtype=np.int64) for c in CV7_HYBRID_COLS}
    bags = {c: np.zeros(n_a, dtype=np.float64) for c in CV7_HYBRID_COLS}
    valid_folds = 0

    for (
        va,
        n_va,
        fseed,
        base_rr,
        y_tr_rr,
        views_rr,
        warp_yj,
        z_yj,
        views_yj,
        warp_rg,
        z_rg,
        views_rg,
        gp_pack,
    ) in prepped["prepped_folds"]:
        if abort_check is not None and abort_check():
            return None
        valid_folds += 1
        pr_rr = _batched_icl_predict(models_w, views_rr, y_tr_rr, dev, seed=fseed + 101)
        pred_rr = np.clip(base_rr + pr_rr, lo_guard, hi_guard)
        q_sums["icl_e8_cw12_robridge_cv7"][va] += pred_rr[:n_va]
        q_cnts["icl_e8_cw12_robridge_cv7"][va] += 1
        bags["icl_e8_cw12_robridge_cv7"] += pred_rr[n_va:]

        pred_yj = np.clip(
            _batched_icl_predict(models_w, views_yj, z_yj, dev, seed=fseed + 202, inv_fn=warp_yj.inv_torch),
            lo_guard,
            hi_guard,
        )
        q_sums["icl_e8_cw12_yj_cv7"][va] += pred_yj[:n_va]
        q_cnts["icl_e8_cw12_yj_cv7"][va] += 1
        bags["icl_e8_cw12_yj_cv7"] += pred_yj[n_va:]

        pred_rg = np.clip(
            _batched_icl_predict(models_w, views_rg, z_rg, dev, seed=fseed + 303, inv_fn=warp_rg.inv_torch),
            lo_guard,
            hi_guard,
        )
        q_sums["icl_e8_rankgauss_duan_cv7"][va] += pred_rg[:n_va]
        q_cnts["icl_e8_rankgauss_duan_cv7"][va] += 1
        bags["icl_e8_rankgauss_duan_cv7"] += pred_rg[n_va:]

        if gp_pack is not None:
            gp_ev, r_tr_gp, views_gp = gp_pack
            pr_gp_res = _batched_icl_predict(models_w, views_gp, r_tr_gp, dev, seed=fseed + 404)
            pred_gp_raw = gp_ev + pr_gp_res
        else:
            pred_gp_raw = pred_rr.copy()
        q_sums["icl_e8_gp_resid_cv7"][va] += pred_gp_raw[:n_va]
        q_cnts["icl_e8_gp_resid_cv7"][va] += 1
        bags["icl_e8_gp_resid_cv7"] += pred_gp_raw[n_va:]

    out: dict[str, np.ndarray] = {"ok": np.array(True)}
    for col in CV7_HYBRID_COLS:
        out[f"q_{col}"] = np.where(q_cnts[col] > 0, q_sums[col] / np.maximum(q_cnts[col], 1), ym_glob)
        out[f"bag_{col}"] = bags[col] / float(max(valid_folds, 1))

    if abort_check is not None and abort_check():
        return None
    rseed = prepped["rseed"]
    out["ref_icl_e8_cw12_robridge_cv7"] = np.clip(
        prepped["base_rr_f"] + _batched_icl_predict(models_w, prepped["v_rr_f"], prepped["y_rr_f"], dev, seed=rseed + 101),
        lo_guard,
        hi_guard,
    )
    out["ref_icl_e8_cw12_yj_cv7"] = np.clip(
        _batched_icl_predict(
            models_w,
            prepped["v_yj_f"],
            prepped["z_yj_f"],
            dev,
            seed=rseed + 202,
            inv_fn=prepped["warp_yj_f"].inv_torch,
        ),
        lo_guard,
        hi_guard,
    )
    out["ref_icl_e8_rankgauss_duan_cv7"] = np.clip(
        _batched_icl_predict(
            models_w,
            prepped["v_rg_f"],
            prepped["z_rg_f"],
            dev,
            seed=rseed + 303,
            inv_fn=prepped["warp_rg_f"].inv_torch,
        ),
        lo_guard,
        hi_guard,
    )

    if gp_anc_ok and prepped["ref_gp_pack"] is not None:
        r_gp_f, v_gp_f = prepped["ref_gp_pack"]
        pr_gp_ref = _batched_icl_predict(models_w, v_gp_f, r_gp_f, dev, seed=rseed + 404)

        delta_oof = out["q_icl_e8_gp_resid_cv7"] - q_gp_anc
        delta_bag = out["bag_icl_e8_gp_resid_cv7"] - bag_gp_anc
        lab_sh = make_stratified_kfold(cid, y, k=eff_k, rep=0)
        q_shrunk = np.zeros(n_s, dtype=np.float64)
        etas = (0.0, 0.25, 0.50, 0.75, 1.0)
        for fk in range(eff_k):
            va = lab_sh == fk
            tr = ~va
            best_e = min(etas, key=lambda e: float(np.mean((y[tr] - (q_gp_anc[tr] + e * delta_oof[tr])) ** 2)))
            q_shrunk[va] = q_gp_anc[va] + best_e * delta_oof[va]
        eta_full = min(etas, key=lambda e: float(np.mean((y - (q_gp_anc + e * delta_oof)) ** 2)))
        out["q_icl_e8_gp_resid_cv7"] = np.clip(q_shrunk, lo_guard, hi_guard)
        out["bag_icl_e8_gp_resid_cv7"] = np.clip(bag_gp_anc + eta_full * delta_bag, lo_guard, hi_guard)
        out["ref_icl_e8_gp_resid_cv7"] = np.clip(ref_gp_anc + eta_full * pr_gp_ref, lo_guard, hi_guard)
    else:
        out["q_icl_e8_gp_resid_cv7"] = np.clip(out["q_icl_e8_gp_resid_cv7"], lo_guard, hi_guard)
        out["bag_icl_e8_gp_resid_cv7"] = np.clip(out["bag_icl_e8_gp_resid_cv7"], lo_guard, hi_guard)
        out["ref_icl_e8_gp_resid_cv7"] = out["ref_icl_e8_cw12_robridge_cv7"].copy()

    return out


def eval_single_context_cv7(
    cid: str,
    Xs: np.ndarray,
    Xa: np.ndarray,
    y: np.ndarray,
    lab: np.ndarray,
    models_w: list[tuple[ICLH450, float]],
    dev: torch.device,
    gp_anchor_dir: Path | None,
    p418_dir: Path,
    p448_dir: Path,
    n_views: int = 4,
    n_repeats: int = 2,
    k_folds: int = 7,
    gp_oof_cache_dir: Path | None = None,
) -> dict[str, np.ndarray]:
    """Evaluates `2xCV-7` (`14` stratified OOF folds) across all 4 hybrid `104M-e8` modes (`0` `ya` read!)."""
    prepped = _prep_full_context_worker(
        (
            0,
            cid,
            Xs,
            Xa,
            y,
            lab,
            gp_anchor_dir,
            p418_dir,
            p448_dir,
            gp_oof_cache_dir,
            n_views,
            n_repeats,
            k_folds,
        )
    )
    return _eval_prepped_context_gpu(prepped, models_w, dev)


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


def _parse_multi_specs(
    multi_specs_str: str,
    fallback_ckpts: str,
    fallback_key: str,
    fallback_out_dir: str,
    dev: torch.device,
) -> list[dict]:
    """Parses `--multi-specs 'TAG|CKPT_KEY|CKPT_CSV|OUT_DIR;...'` or single fallback spec, caching `.pt` loads on CPU."""
    raw_entries: list[tuple[str, str, str, Path]] = []
    if multi_specs_str.strip():
        for chunk in multi_specs_str.split(";"):
            chunk = chunk.strip()
            if not chunk:
                continue
            parts = [p.strip() for p in chunk.split("|")]
            if len(parts) != 4:
                raise SystemExit(f"invalid --multi-specs entry (expected TAG|KEY|CKPT_CSV|OUT_DIR): {chunk!r}")
            tag, key, ck_csv, odir = parts
            raw_entries.append((tag, key, ck_csv, Path(odir)))
    else:
        if not fallback_ckpts.strip() or not fallback_out_dir.strip():
            raise SystemExit("either --multi-specs or both --ckpts and --out-dir are required for eval-cv7")
        raw_entries.append(("single", fallback_key, fallback_ckpts, Path(fallback_out_dir)))

    cpu_ckpt_cache: dict[str, dict] = {}
    parsed: list[dict] = []
    for tag, key, ck_csv, odir in raw_entries:
        odir.mkdir(parents=True, exist_ok=True)
        ckpt_specs = [x.strip() for x in ck_csv.split(",") if x.strip()]
        if not ckpt_specs:
            raise SystemExit(f"empty ckpt list in spec {tag!r}")
        models_w: list[tuple[ICLH450, float]] = []
        for sp in ckpt_specs:
            sz, p_str = sp.split(":", 1) if ":" in sp else ("L", sp)
            models_w.append((load_icl_ckpt(Path(p_str), sz, dev, ckpt_key=key, cpu_ckpt_cache=cpu_ckpt_cache), 1.0))
        parsed.append({"tag": tag, "ckpt_key": key, "out_dir": odir, "models_w": models_w})
    cpu_ckpt_cache.clear()
    return parsed


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cmd", choices=("build-traj-e8", "eval-cv7"), required=True)
    ap.add_argument("--bank", default="/home/ubuntu/jope-data/h450-quarantine/bank_d91_search_f64.npz")
    ap.add_argument("--traj-dirs", default="")
    ap.add_argument("--ckpts", default="")
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--multi-specs", default="", help="Semicolon-separated 'TAG|CKPT_KEY|CKPT_CSV|OUT_DIR' entries")
    ap.add_argument("--prep-workers", type=int, default=8, help="Multiprocess CPU fold-prep workers (0/1 = in-process)")
    ap.add_argument(
        "--gp-oof-cache-dir",
        default="/home/ubuntu/jope-data/h452-flywheel/gp_anchors_oof_cache_v1",
        help="Disk cache directory for checkpoint-independent OOF GP anchors ('' to disable)",
    )
    ap.add_argument("--gp-anchor-dir", default="")
    ap.add_argument("--p418-dir", default="/home/ubuntu/jope_plan/evidence/H418/pulled")
    ap.add_argument("--p448-dir", default="/home/ubuntu/jope_plan/evidence/H448/pulled")
    ap.add_argument("--indices", default="0-90")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n-views", type=int, default=4)
    ap.add_argument("--n-repeats", type=int, default=2)
    ap.add_argument("--k-folds", type=int, default=7)
    ap.add_argument("--ckpt-key", default="auto", choices=("auto", "ema_cooldown", "ema_valley", "ema_swa", "ema_soup50", "ema", "model"))
    a = ap.parse_args()

    if a.cmd == "build-traj-e8":
        if not a.out_dir.strip():
            raise SystemExit("--out-dir required for build-traj-e8")
        out_dir = Path(a.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        tdirs = [Path(x.strip()) for x in a.traj_dirs.split(",") if x.strip()]
        res = build_trajectory_e8_dir(Path(a.bank), tdirs, out_dir)
        print(json.dumps(res, sort_keys=True))
        return

    dev = torch.device(a.device)
    gp_anc_dir = Path(a.gp_anchor_dir) if a.gp_anchor_dir else None
    gp_oof_cache_dir = Path(a.gp_oof_cache_dir) if a.gp_oof_cache_dir.strip() else None
    p418_dir = Path(a.p418_dir)
    p448_dir = Path(a.p448_dir)

    with np.load(a.bank, allow_pickle=False) as zb:
        assert "ya" not in zb.files, "audit labels forbidden in search bank!"
        cids = [str(x) for x in zb["cids"].tolist()]
        ss, sa = zb["s_shape"], zb["a_shape"]
        ctx_data = {}
        so = ao = lo = 0
        for i, c in enumerate(cids):
            n, w = int(ss[i, 0]), int(ss[i, 1])
            na, f = int(sa[i, 0]), int(sa[i, 1])
            S_mat = np.array(zb["search"][so : so + n * w], np.float64).reshape(n, w)
            Xa = np.array(zb["audit"][ao : ao + na * f], np.float64, copy=True).reshape(na, f)
            ctx_data[i] = (
                c,
                np.array(S_mat[:, :-1], np.float64, copy=True),
                Xa,
                np.array(zb["y64"][lo : lo + n], np.float64, copy=True),
                np.array(zb["lab"][lo : lo + n], np.int64, copy=True),
            )
            so += n * w
            ao += na * f
            lo += n

    specs = _parse_multi_specs(a.multi_specs, a.ckpts, a.ckpt_key, a.out_dir, dev)

    work_items: list[tuple] = []
    for idx in _parse_indices(a.indices):
        if idx not in ctx_data:
            continue
        c, Xs, Xa, y, lab = ctx_data[idx]
        if all((sp["out_dir"] / f"{c}.npz").is_file() for sp in specs):
            continue
        work_items.append(
            (
                idx,
                c,
                Xs,
                Xa,
                y,
                lab,
                gp_anc_dir,
                p418_dir,
                p448_dir,
                gp_oof_cache_dir,
                a.n_views,
                a.n_repeats,
                a.k_folds,
            )
        )

    t0 = time.time()
    done_ctx = 0
    written_files = {sp["tag"]: 0 for sp in specs}

    def _consume_prepped(prepped: dict) -> None:
        nonlocal done_ctx
        c = prepped["cid"]
        for sp in specs:
            dst = sp["out_dir"] / f"{c}.npz"
            if dst.is_file():
                continue
            arrays = _eval_prepped_context_gpu(prepped, sp["models_w"], dev)
            tmp = sp["out_dir"] / f".{c}.{os.getpid()}.{sp['tag']}.tmp.npz"
            np.savez(tmp, **arrays)
            os.replace(tmp, dst)
            written_files[sp["tag"]] += 1
        done_ctx += 1
        if done_ctx % 5 == 0 or done_ctx == len(work_items):
            print(
                json.dumps(
                    {
                        "progress_ctx": done_ctx,
                        "total_pending_ctx": len(work_items),
                        "written_by_spec": written_files,
                        "last_cid": c,
                        "elapsed_s": round(time.time() - t0, 1),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    n_workers = max(0, int(a.prep_workers))
    if n_workers <= 1 or len(work_items) <= 1:
        for item in work_items:
            _consume_prepped(_prep_full_context_worker(item))
    else:
        mp_ctx = mp.get_context("fork") if "fork" in mp.get_all_start_methods() else mp.get_context()
        win = max(8, n_workers * 3)
        with mp_ctx.Pool(processes=min(n_workers, len(work_items)), initializer=_init_prep_worker, maxtasksperchild=8) as pool:
            it = iter(work_items)
            while True:
                batch_items = list(itertools.islice(it, win))
                if not batch_items:
                    break
                for prepped in pool.imap_unordered(_prep_full_context_worker, batch_items, chunksize=1):
                    _consume_prepped(prepped)

    print(
        json.dumps(
            {
                "completed_ctx": done_ctx,
                "written_by_spec": written_files,
                "elapsed_s": round(time.time() - t0, 1),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
