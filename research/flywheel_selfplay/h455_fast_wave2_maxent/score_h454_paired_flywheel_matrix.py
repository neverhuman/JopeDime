#!/usr/bin/env python3
"""H454 Paired `Pre-H454` vs `Post-H454` Closed-Loop Deep Learning Flywheel Evaluator (`score_h454_paired_flywheel_matrix.py`).

Evaluates `Pre-H454` vs `Post-H454` `104M` checkpoints under identical seeds, fold splits, and `Gen-1000` `Ultra-GP`
residual anchors across four nested regimes:
  1. Pure `104M` `2xCV-7` (zero GP predictions at test time — isolates pure KD + Counterfactual weight transfer).
  2. `104M` `2xCV-7` + `GP-Residual` (`CV7_HYBRID_COLS` — isolates `y - y_hat_GP` residual co-adaptation).
  3. `104M + Ultra-GP Gen-1000` Two-Tower Neural-Symbolic Pair (zero legacy `Q_gp/M7/K3/F4/Tab` pools).
  4. Full Unified Guarded Stack (`stack_ugp_g1000_{cv7,2xcv7,cons}_{pre,post}_h454`) against the 33-model `dev91` board.

Strictly respects quarantine (`ya` accessed ONLY inside post-prediction `M.utility(pred, ya, ba)`; `0` deletions; `0` `/tmp`).
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

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np

HERE = Path(__file__).resolve().parent
for _cand in (
    str(HERE),
    "/home/ubuntu/h450-stage/pub-r6/code",
    "/home/ubuntu/h448-stage/pub-r1/code",
):
    if Path(_cand).is_dir() and _cand not in sys.path:
        sys.path.append(_cand)

import guard_matern_core as GM  # noqa: E402
import meta_h416 as M  # noqa: E402
import train_h416 as T16  # noqa: E402
from analyze_h450 import H430_K3, H433_F4, N_BOOT, SEEDS, TAGS, _load_col_block  # noqa: E402
from cols_104m_e8_gpresid_h452 import CV7_HYBRID_COLS  # noqa: E402
from cols_h450_gpu import H450_GPU_COLS  # noqa: E402
from cols_tab_h450 import H450_TAB_COLS  # noqa: E402
from score_h452_dev91 import _load_ultra_gp_block, _rank_worker  # noqa: E402
from stack_guard_cv7_h452 import stack_guard_cv7_context  # noqa: E402


def _check_h454_quarantine(p: Path) -> None:
    rp = str(p.resolve())
    allowed = (
        "/home/ubuntu/jope_plan/evidence/H455/",
        "/home/ubuntu/jope_plan/evidence/H454/",
        "/home/ubuntu/jope_plan/evidence/H453/",
        "/home/ubuntu/jope_plan/evidence/H452/",
        "/home/ubuntu/jope-data/h452-flywheel/",
    )
    if not any(rp.startswith(pref) for pref in allowed):
        raise SystemExit(f"QUARANTINE REFUSAL: path {rp} is outside H454 quarantine")

PURE_104M_CV7_COLS = (
    "icl_e8_cw12_robridge_cv7",
    "icl_e8_cw12_yj_cv7",
    "icl_e8_rankgauss_duan_cv7",
)
GP_RESID_ONLY_COL = ("icl_e8_gp_resid_cv7",)
TRAJ_COMPACT_COLS = ("traj_opt_e8", "traj_yj_e8")


def _sha12(p: Path) -> str:
    if p is None or not p.is_file():
        return "none"
    h = hashlib.sha256()
    with p.open("rb") as fh:
        while True:
            b = fh.read(1 << 20)
            if not b:
                break
            h.update(b)
    return h.hexdigest()[:12]


def _eval_ctx_h454(job):
    (
        s,
        c,
        npz_p,
        Xs,
        Xa,
        y,
        lab,
        blk_m,
        blk_k,
        blk_f,
        blk_tab,
        blk_g16,
        blk_traj_pre,
        blk_traj_post,
        blk_pure_pre,
        blk_pure_post,
        blk_gpr_pre,
        blk_gpr_post,
        blk_hyb_pre,
        blk_hyb_post,
        ugp_g10_cv7_blk,
        ugp_g1000_blocks,
    ) = job

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

    def _u(pred: np.ndarray) -> float:
        return float(M.utility(pred, ya, ba)[0])

    non_gpu_named = [
        (tag, blk)
        for tag, blk in (("m7", blk_m), ("k3", blk_k), ("f4", blk_f), ("tab450", blk_tab))
        if blk is not None
    ]
    base_g16 = list(non_gpu_named) + ([("gpu450", blk_g16)] if blk_g16 is not None else [])

    out_u: dict[str, float] = {}
    out_u["ctrl_h450_g16_legacy"] = _u(
        GM.guard_and_stack_2l(c, Xs, Xa, y, lab, Q_gp, bcols_gp, ref_gp, [b for _, b in base_g16])[0]
    )

    # Evaluate Pre-H454 vs Post-H454 across identical regimes
    for tag, blk_pure, blk_gpr, blk_hyb, blk_traj in (
        ("pre_h454", blk_pure_pre, blk_gpr_pre, blk_hyb_pre, blk_traj_pre),
        ("post_h454", blk_pure_post, blk_gpr_post, blk_hyb_post, blk_traj_post),
    ):
        if blk_hyb is None:
            continue

        if s == SEEDS[0]:
            # Regime 1a: Strictly Pure 104M CV-7 ONLY (0 GP columns, 0 traj/g16 shared columns!)
            if blk_pure is not None:
                out_u[f"pure_104m_only_cv7_{tag}"] = _u(
                    stack_guard_cv7_context(
                        c, Xs, Xa, y, lab, None, None, None, [("e8_cv7_pure", blk_pure)], use_cv7_folds=True
                    )[0]
                )
                # Regime 1b: Pure 104M CV-7 + Trajectory E8 (0 GP columns at test time)
                p_pure_list = [("e8_cv7_pure", blk_pure)]
                if blk_traj is not None:
                    p_pure_list.append(("e8_traj", blk_traj))
                out_u[f"pure_104m_cv7_{tag}"] = _u(
                    stack_guard_cv7_context(c, Xs, Xa, y, lab, None, None, None, p_pure_list, use_cv7_folds=True)[0]
                )

            # Regime 2a: Strictly 104M CV-7 + GP-Residual ONLY (0 traj/g16 shared columns!)
            out_u[f"hyb_104m_only_cv7_{tag}"] = _u(
                stack_guard_cv7_context(
                    c, Xs, Xa, y, lab, None, None, None, [("e8_cv7_hyb", blk_hyb)], use_cv7_folds=True
                )[0]
            )

            # Regime 2b: 104M CV-7 + GP-Residual + Trajectory E8 (`icl_e8_gp_resid_cv7`)
            p_hyb_only = [("e8_cv7_hyb", blk_hyb)]
            if blk_traj is not None:
                p_hyb_only.append(("e8_traj", blk_traj))
            out_u[f"hyb_104m_gpresid_cv7_{tag}"] = _u(
                stack_guard_cv7_context(c, Xs, Xa, y, lab, None, None, None, p_hyb_only, use_cv7_folds=True)[0]
            )

            # Regime 2b: Full 104M Neural Ensemble (`g16 + traj + cv7_hyb`)
            neural_list = [
                (t, b)
                for t, b in (("gpu450", blk_g16), ("e8_traj", blk_traj), ("e8_cv7_hyb", blk_hyb))
                if b is not None
            ]
            out_u[f"neural_104m_all_{tag}"] = _u(
                stack_guard_cv7_context(c, Xs, Xa, y, lab, None, None, None, neural_list, use_cv7_folds=True)[0]
            )

            # Regime 3: 104M + Ultra-GP Gen-1000 Two-Tower (0 legacy pools)
            for cv_mode, blk_u in ugp_g1000_blocks.items():
                if blk_u is not None:
                    out_u[f"neural104m_ugp_g1000_{cv_mode}_{tag}"] = _u(
                        stack_guard_cv7_context(
                            c,
                            Xs,
                            Xa,
                            y,
                            lab,
                            None,
                            None,
                            None,
                            neural_list + [(f"ugp_g1000_{cv_mode}", blk_u)],
                            use_cv7_folds=True,
                        )[0]
                    )

        # Regime 4a: Pure-104M Guarded Stack (0 Gen-1000 GP anchor in base; James-Stein noise ratchet active for ugp_*)
        if blk_pure is not None:
            stack_pure_base = list(base_g16)
            if blk_traj is not None:
                stack_pure_base.append(("e8_traj", blk_traj))
            stack_pure_base.append(("e8_cv7_pure", blk_pure))
            pred_pure_no_ugp, _ = stack_guard_cv7_context(
                c, Xs, Xa, y, lab, Q_gp, bcols_gp, ref_gp, stack_pure_base, use_cv7_folds=True
            )
            out_u[f"stack_pure104m_no_ugp_{tag}"] = _u(pred_pure_no_ugp)

            if ugp_g10_cv7_blk is not None:
                pred_p_g10, _ = stack_guard_cv7_context(
                    c,
                    Xs,
                    Xa,
                    y,
                    lab,
                    Q_gp,
                    bcols_gp,
                    ref_gp,
                    stack_pure_base + [("ugp_g10_cv7", ugp_g10_cv7_blk)],
                    use_cv7_folds=True,
                    no_ugp_fallback=pred_pure_no_ugp,
                )
                out_u[f"stack_pure104m_ugp_g10_cv7_{tag}"] = _u(pred_p_g10)

            blk_u_cons = ugp_g1000_blocks.get("cons")
            if blk_u_cons is not None and blk_gpr is not None:
                pred_grd_hyb, _ = stack_guard_cv7_context(
                    c,
                    Xs,
                    Xa,
                    y,
                    lab,
                    Q_gp,
                    bcols_gp,
                    ref_gp,
                    stack_pure_base + [("ugp_g1000_hyb104m", blk_gpr), ("ugp_g1000_cons", blk_u_cons)],
                    use_cv7_folds=True,
                    no_ugp_fallback=pred_pure_no_ugp,
                )
                out_u[f"stack_guarded_hyb_ugp_g1000_cons_{tag}"] = _u(pred_grd_hyb)

        # Regime 4b: Full Unified Hybrid Stack (both without Ultra-GP and with Ultra-GP Gen-1000)
        stack_base = list(base_g16)
        if blk_traj is not None:
            stack_base.append(("e8_traj", blk_traj))
        stack_base.append(("e8_cv7_hyb", blk_hyb))

        pred_no_ugp, _ = stack_guard_cv7_context(
            c, Xs, Xa, y, lab, Q_gp, bcols_gp, ref_gp, stack_base, use_cv7_folds=True
        )
        out_u[f"stack_no_ugp_{tag}"] = _u(pred_no_ugp)

        for cv_mode, blk_u in ugp_g1000_blocks.items():
            if blk_u is not None:
                pred_ugp, _ = stack_guard_cv7_context(
                    c,
                    Xs,
                    Xa,
                    y,
                    lab,
                    Q_gp,
                    bcols_gp,
                    ref_gp,
                    stack_base + [(f"ugp_g1000_{cv_mode}", blk_u)],
                    use_cv7_folds=True,
                    no_ugp_fallback=pred_no_ugp,
                )
                out_u[f"stack_ugp_g1000_{cv_mode}_{tag}"] = _u(pred_ugp)

    # Also evaluate Joint Pre+Post Trajectory Diversity Stack if both Pre and Post exist
    if blk_hyb_pre is not None and blk_hyb_post is not None:
        if blk_pure_pre is not None and blk_pure_post is not None:
            stack_pure_joint = list(base_g16)
            if blk_traj_post is not None:
                stack_pure_joint.append(("e8_traj", blk_traj_post))
            elif blk_traj_pre is not None:
                stack_pure_joint.append(("e8_traj", blk_traj_pre))
            stack_pure_joint.append(("e8_cv7_pure_pre", blk_pure_pre))
            stack_pure_joint.append(("e8_cv7_pure_post", blk_pure_post))
            pred_pj_no_ugp, _ = stack_guard_cv7_context(
                c, Xs, Xa, y, lab, Q_gp, bcols_gp, ref_gp, stack_pure_joint, use_cv7_folds=True
            )
            out_u["stack_pure104m_no_ugp_joint_pre_post"] = _u(pred_pj_no_ugp)
            if ugp_g10_cv7_blk is not None:
                pred_pj_g10, _ = stack_guard_cv7_context(
                    c,
                    Xs,
                    Xa,
                    y,
                    lab,
                    Q_gp,
                    bcols_gp,
                    ref_gp,
                    stack_pure_joint + [("ugp_g10_cv7", ugp_g10_cv7_blk)],
                    use_cv7_folds=True,
                    no_ugp_fallback=pred_pj_no_ugp,
                )
                out_u["stack_pure104m_ugp_g10_cv7_joint_pre_post"] = _u(pred_pj_g10)
            blk_u_cons = ugp_g1000_blocks.get("cons")
            if blk_u_cons is not None and blk_gpr_post is not None:
                ugp_joint_extras = [("ugp_g1000_hyb104m_post", blk_gpr_post), ("ugp_g1000_cons", blk_u_cons)]
                if blk_gpr_pre is not None:
                    ugp_joint_extras.insert(0, ("ugp_g1000_hyb104m_pre", blk_gpr_pre))
                pred_pj_ghyb, _ = stack_guard_cv7_context(
                    c,
                    Xs,
                    Xa,
                    y,
                    lab,
                    Q_gp,
                    bcols_gp,
                    ref_gp,
                    stack_pure_joint + ugp_joint_extras,
                    use_cv7_folds=True,
                    no_ugp_fallback=pred_pj_no_ugp,
                )
                out_u["stack_guarded_hyb_ugp_g1000_cons_joint_pre_post"] = _u(pred_pj_ghyb)
        stack_joint = list(base_g16)
        if blk_traj_post is not None:
            stack_joint.append(("e8_traj", blk_traj_post))
        elif blk_traj_pre is not None:
            stack_joint.append(("e8_traj", blk_traj_pre))
        stack_joint.append(("e8_cv7_pre", blk_hyb_pre))
        stack_joint.append(("e8_cv7_post", blk_hyb_post))
        pred_joint_no_ugp, _ = stack_guard_cv7_context(
            c, Xs, Xa, y, lab, Q_gp, bcols_gp, ref_gp, stack_joint, use_cv7_folds=True
        )
        out_u["stack_no_ugp_joint_pre_post"] = _u(pred_joint_no_ugp)
        for cv_mode, blk_u in ugp_g1000_blocks.items():
            if blk_u is not None:
                pred_joint_ugp, _ = stack_guard_cv7_context(
                    c,
                    Xs,
                    Xa,
                    y,
                    lab,
                    Q_gp,
                    bcols_gp,
                    ref_gp,
                    stack_joint + [(f"ugp_g1000_{cv_mode}", blk_u)],
                    use_cv7_folds=True,
                    no_ugp_fallback=pred_joint_no_ugp,
                )
                out_u[f"stack_ugp_g1000_{cv_mode}_joint_pre_post"] = _u(pred_joint_ugp)

    return s, c, out_u


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-json", default="/home/ubuntu/jope_plan/evidence/H454/H454_PAIRED_FLYWHEEL_MATRIX.json")
    ap.add_argument("--cv7-pre-dir", default="/home/ubuntu/jope-data/h452-flywheel/cols_e8_d91/cv7_hyb_pre_h454")
    ap.add_argument("--cv7-post-dir", default="/home/ubuntu/jope-data/h452-flywheel/cols_e8_d91/cv7_hyb_post_h454")
    ap.add_argument("--traj-pre-dir", default="/home/ubuntu/jope-data/h452-flywheel/cols_e8_d91/traj_e8_v1")
    ap.add_argument("--traj-post-dir", default="/home/ubuntu/jope-data/h452-flywheel/cols_e8_d91/traj_e8_post_h454")
    ap.add_argument("--g16-dir", default="/home/ubuntu/jope-data/h450-quarantine/r4/cols/g16_r4_L_A_r4_L_B")
    ap.add_argument("--ugp-g10-dir", default="/home/ubuntu/jope-data/h452-flywheel/ultra_gp_d91/g10")
    ap.add_argument("--ugp-g1000-dir", default="/home/ubuntu/jope-data/h452-flywheel/ultra_gp_d91/g1000")
    ap.add_argument("--procs", type=int, default=20)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    a = ap.parse_args()

    try:
        os.nice(19)
    except OSError:
        pass

    t0 = time.time()
    out_p = Path(a.out_json)
    _check_h454_quarantine(out_p)

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
    g16_dir = Path(a.g16_dir)
    cv7_pre_dir = Path(a.cv7_pre_dir)
    cv7_post_dir = Path(a.cv7_post_dir)
    traj_pre_dir = Path(a.traj_pre_dir)
    traj_post_dir = Path(a.traj_post_dir)
    ugp_g1000_dir = Path(a.ugp_g1000_dir)

    has_pre = cv7_pre_dir.is_dir() and all((cv7_pre_dir / f"{c}.npz").is_file() for c in cids)
    has_post = cv7_post_dir.is_dir() and all((cv7_post_dir / f"{c}.npz").is_file() for c in cids)
    has_traj_post = traj_post_dir.is_dir() and all((traj_post_dir / f"{c}.npz").is_file() for c in cids)

    bank_rows = {}
    with np.load("/home/ubuntu/jope-data/h450-quarantine/bank_d91_search_f64.npz", allow_pickle=False) as zb:
        assert "ya" not in zb.files
        b_cids = [str(x) for x in zb["cids"].tolist()]
        ss, sa = zb["s_shape"], zb["a_shape"]
        so = ao = lo = 0
        for i, c in enumerate(b_cids):
            n, w = int(ss[i, 0]), int(ss[i, 1])
            na, f = int(sa[i, 0]), int(sa[i, 1])
            S_mat = np.array(zb["search"][so : so + n * w], np.float64).reshape(n, w)
            Xa = np.array(zb["audit"][ao : ao + na * f], np.float64, copy=True).reshape(na, f)
            bank_rows[c] = (
                np.array(S_mat[:, :-1], np.float64, copy=True),
                Xa,
                np.array(zb["y64"][lo : lo + n], np.float64, copy=True),
                np.array(zb["lab"][lo : lo + n], np.int64, copy=True),
            )
            so += n * w
            ao += na * f
            lo += n

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
        blk_g16 = _load_col_block(g16_dir / f"{c}.npz", H450_GPU_COLS, n_s, n_a)
        blk_traj_pre = _load_col_block(traj_pre_dir / f"{c}.npz", TRAJ_COMPACT_COLS, n_s, n_a) if traj_pre_dir.is_dir() else None
        blk_traj_post = (
            _load_col_block(traj_post_dir / f"{c}.npz", TRAJ_COMPACT_COLS, n_s, n_a) if has_traj_post else blk_traj_pre
        )

        blk_pure_pre = _load_col_block(cv7_pre_dir / f"{c}.npz", PURE_104M_CV7_COLS, n_s, n_a) if has_pre else None
        blk_gpr_pre = _load_col_block(cv7_pre_dir / f"{c}.npz", GP_RESID_ONLY_COL, n_s, n_a) if has_pre else None
        blk_hyb_pre = _load_col_block(cv7_pre_dir / f"{c}.npz", CV7_HYBRID_COLS, n_s, n_a) if has_pre else None
        blk_pure_post = _load_col_block(cv7_post_dir / f"{c}.npz", PURE_104M_CV7_COLS, n_s, n_a) if has_post else None
        blk_gpr_post = _load_col_block(cv7_post_dir / f"{c}.npz", GP_RESID_ONLY_COL, n_s, n_a) if has_post else None
        blk_hyb_post = _load_col_block(cv7_post_dir / f"{c}.npz", CV7_HYBRID_COLS, n_s, n_a) if has_post else None

        gp10_p = Path(a.ugp_g10_dir) / f"{c}.npz"
        ugp_g10_cv7_blk = _load_ultra_gp_block(gp10_p, "cv7", n_s, n_a) if gp10_p.is_file() else None

        gp_p = ugp_g1000_dir / f"{c}.npz"
        ugp_g1000_blocks = {}
        if gp_p.is_file():
            for cv_m in ("cv7", "2xcv7", "cons"):
                b_u = _load_ultra_gp_block(gp_p, cv_m, n_s, n_a)
                if b_u is not None:
                    ugp_g1000_blocks[cv_m] = b_u

        for s in SEEDS:
            jobs.append(
                (
                    s,
                    c,
                    P418 / TAGS[s] / f"h416_{TAGS[s]}_npz" / f"{c}.npz",
                    Xs,
                    Xa,
                    y,
                    lab,
                    blk_m,
                    blk_k,
                    blk_f,
                    blk_tab,
                    blk_g16,
                    blk_traj_pre,
                    blk_traj_post,
                    blk_pure_pre,
                    blk_pure_post,
                    blk_gpr_pre,
                    blk_gpr_post,
                    blk_hyb_pre,
                    blk_hyb_post,
                    ugp_g10_cv7_blk,
                    ugp_g1000_blocks,
                )
            )

    with Pool(processes=min(a.procs, len(jobs))) as pool:
        done = pool.map(_eval_ctx_h454, jobs)

    by_seed_arm: dict[int, dict[str, dict[str, float]]] = {s: {} for s in SEEDS}
    for s, c, out_u in done:
        for arm_k, val_u in out_u.items():
            by_seed_arm[s].setdefault(arm_k, {})[c] = val_u

    arm_panels: dict[str, dict[str, float]] = {}
    for arm_k, d0 in by_seed_arm[SEEDS[0]].items():
        if arm_k.startswith(("pure_104m_", "hyb_104m_", "neural_104m_", "neural104m_")) and len(d0) == len(cids):
            arm_panels[arm_k] = dict(d0)
            continue
        d1 = by_seed_arm[SEEDS[1]].get(arm_k, {})
        if len(d0) == len(cids) and len(d1) == len(cids):
            arm_panels[arm_k] = {c: 0.5 * (d0[c] + d1[c]) for c in cids}

    rank_jobs = [
        (arm_k, field, f"h454.{arm_k.upper()}", u_map, fam, cids, a.n_boot)
        for arm_k, u_map in sorted(arm_panels.items())
    ]
    with Pool(processes=min(max(1, len(rank_jobs)), a.procs)) as pool:
        ranked = dict(pool.map(_rank_worker, rank_jobs))

    _, H = T16._stats_mod()
    fk = {c: fam[c] for c in cids}
    paired_comparisons = {}
    pair_specs = [
        ("pure_104m_only_cv7_post_vs_pre", "pure_104m_only_cv7_post_h454", "pure_104m_only_cv7_pre_h454"),
        ("hyb_104m_only_cv7_post_vs_pre", "hyb_104m_only_cv7_post_h454", "hyb_104m_only_cv7_pre_h454"),
        ("hyb_vs_pure_104m_only_post", "hyb_104m_only_cv7_post_h454", "pure_104m_only_cv7_post_h454"),
        ("pure_104m_cv7_post_vs_pre", "pure_104m_cv7_post_h454", "pure_104m_cv7_pre_h454"),
        ("hyb_104m_gpresid_cv7_post_vs_pre", "hyb_104m_gpresid_cv7_post_h454", "hyb_104m_gpresid_cv7_pre_h454"),
        ("neural_104m_all_post_vs_pre", "neural_104m_all_post_h454", "neural_104m_all_pre_h454"),
        ("neural104m_ugp_g1000_cons_post_vs_pre", "neural104m_ugp_g1000_cons_post_h454", "neural104m_ugp_g1000_cons_pre_h454"),
        ("neural104m_ugp_g1000_cv7_post_vs_pre", "neural104m_ugp_g1000_cv7_post_h454", "neural104m_ugp_g1000_cv7_pre_h454"),
        ("stack_pure104m_no_ugp_post_vs_pre", "stack_pure104m_no_ugp_post_h454", "stack_pure104m_no_ugp_pre_h454"),
        ("stack_pure104m_ugp_g10_cv7_post_vs_pre", "stack_pure104m_ugp_g10_cv7_post_h454", "stack_pure104m_ugp_g10_cv7_pre_h454"),
        ("stack_guarded_hyb_ugp_g1000_cons_post_vs_pre", "stack_guarded_hyb_ugp_g1000_cons_post_h454", "stack_guarded_hyb_ugp_g1000_cons_pre_h454"),
        ("stack_guarded_hyb_ugp_g1000_cons_joint_vs_pre", "stack_guarded_hyb_ugp_g1000_cons_joint_pre_post", "stack_guarded_hyb_ugp_g1000_cons_pre_h454"),
        ("stack_no_ugp_post_vs_pre", "stack_no_ugp_post_h454", "stack_no_ugp_pre_h454"),
        ("stack_ugp_g1000_cons_post_vs_pre", "stack_ugp_g1000_cons_post_h454", "stack_ugp_g1000_cons_pre_h454"),
        ("stack_ugp_g1000_cv7_post_vs_pre", "stack_ugp_g1000_cv7_post_h454", "stack_ugp_g1000_cv7_pre_h454"),
        ("stack_ugp_g1000_2xcv7_post_vs_pre", "stack_ugp_g1000_2xcv7_post_h454", "stack_ugp_g1000_2xcv7_pre_h454"),
        ("stack_ugp_g1000_cons_joint_vs_pre", "stack_ugp_g1000_cons_joint_pre_post", "stack_ugp_g1000_cons_pre_h454"),
        ("stack_ugp_g1000_cons_post_vs_g16_leg", "stack_ugp_g1000_cons_post_h454", "ctrl_h450_g16_legacy"),
        ("stack_ugp_g1000_cons_joint_vs_g16_leg", "stack_ugp_g1000_cons_joint_pre_post", "ctrl_h450_g16_legacy"),
    ]
    for p_name, a_hi, a_lo in pair_specs:
        if a_hi in arm_panels and a_lo in arm_panels:
            paired_comparisons[p_name] = T16.pair_stats(np, H, arm_panels[a_hi], arm_panels[a_lo], fk)

    summary_rows = []
    for arm_k, rk in sorted(ranked.items(), key=lambda kv: -float(kv[1]["elo"]["elo"])):
        summary_rows.append(
            {
                "arm": arm_k,
                "rank_of_34": int(rk["rank_if_placed_against_frozen_33"]),
                "elo": round(float(rk["elo"]["elo"]), 2),
                "ci95": [round(float(x), 2) for x in rk["elo"]["ci95"]],
                "family_mean_u": round(float(rk["family_mean_u"]), 4),
                "vs_tabpfn_v3p5": round(float(rk["gap_elo"]["vs_tabpfn_v3p5.v1"]), 2),
                "vs_tabfm_v1": round(float(rk["gap_elo"]["vs_tabfm.v1"]), 2),
                "vs_kumo_small": round(float(rk["gap_elo"]["vs_kumo_tabular_small_e8.v1"]), 2),
                "vs_tabfm_ens": round(float(rk["gap_elo"]["vs_tabfm_ensemble.v1"]), 2),
                "vs_kumo_large": round(float(rk["gap_elo"]["vs_kumo_tabular_large_e8.v1"]), 2),
            }
        )

    doc = {
        "schema": "jope.h454.paired_flywheel_matrix/v1",
        "quarantined_in_corpus_random_split_diagnostic": True,
        "canonical_promotion_forbidden": True,
        "has_pre_h454": has_pre,
        "has_post_h454": has_post,
        "has_traj_post_h454": has_traj_post,
        "n_boot": a.n_boot,
        "elapsed_s": round(time.time() - t0, 2),
        "summary_table": summary_rows,
        "paired_comparisons": paired_comparisons,
        "arms_full": ranked,
    }
    out_p.parent.mkdir(parents=True, exist_ok=True)
    tmp_p = out_p.with_suffix(".tmp.json")
    tmp_p.write_text(json.dumps(doc, indent=2, sort_keys=True, default=lambda o: o.item()) + "\n")
    os.replace(tmp_p, out_p)
    print(
        json.dumps(
            {
                "has_pre_h454": has_pre,
                "has_post_h454": has_post,
                "elapsed_s": doc["elapsed_s"],
                "summary_table": summary_rows,
                "paired_deltas": {
                    k: {
                        "delta_family_mean_u": round(float(v.get("delta_family_mean_u", v.get("mean_diff", 0.0))), 5)
                        if isinstance(v, dict)
                        else v,
                        "raw": v,
                    }
                    for k, v in paired_comparisons.items()
                },
            },
            indent=2,
            default=lambda o: o.item(),
        )
    )


if __name__ == "__main__":
    main()
