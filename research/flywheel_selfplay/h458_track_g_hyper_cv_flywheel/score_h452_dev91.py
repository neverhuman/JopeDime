#!/usr/bin/env python3
"""H452 Quarantined 34-Entrant `dev91-audit-v1` Leaderboard & Paired Ablation Scorer (`score_h452_dev91.py`).

Strict quarantine & zero-leakage guarantees:
  - Output path must reside inside `/H452/` or `/h452`; canonical `reference-board` paths are unconditionally refused.
  - `y_audit` (`ya`) is read ONLY inside `M.utility(pred, ya, ba)` AFTER `pred` is finalized from `(Xs, Xa, y, lab)`.
  - Reports exact 34-entrant Bradley-Terry Elo (`2,500` family-stratified bootstraps), `family_mean_u`, `W-T-L`, and
    `signflip_p` against `kumo_tabular_large_e8.v1`, `tabfm_ensemble.v1`, `kumo_tabular_small_e8.v1`, `tabfm.v1`,
    `tabpfn_v3p5.v1`, `h448.APEX_FAIR_CHAMPION`, and the paired `GM.guard_and_stack_2l` control.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
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
from analyze_h450 import ANCHOR, H430_K3, H433_F4, LADDER_SEED, N_BOOT, SEEDS, TAGS, _load_col_block  # noqa: E402
from cols_104m_e8_gpresid_h452 import CV7_HYBRID_COLS, TRAJ_E8_COLS  # noqa: E402
from cols_h450_gpu import H450_GPU_COLS  # noqa: E402
from cols_tab_h450 import H450_TAB_COLS  # noqa: E402
from stack_guard_cv7_h452 import stack_guard_cv7_context  # noqa: E402


FOCUS_ENTRANTS = (
    "kumo_tabular_large_e8.v1",
    "tabfm_ensemble.v1",
    "kumo_tabular_small_e8.v1",
    "tabfm.v1",
    "tabpfn_v3p5.v1",
    "tabpfn_v3p5_e1.v1",
    "h448.APEX_FAIR_CHAMPION",
    "h416.SXB",
)


def _check_h452_quarantine(p: Path) -> None:
    rp = str(p.resolve())
    forbidden = ("reference-board/boards/dev91-audit-v1", "h335-stagev/refboard", "h335-cl/neural")
    if any(tok in rp for tok in forbidden) or not any(q in rp for q in ("/H452/", "/h452", "/H453/", "/h453")):
        raise SystemExit(f"QUARANTINE REFUSAL: path {rp} is outside H452/H453 quarantine")


def _load_ultra_gp_block(p: Path, cv_mode: str, n_s: int, n_a: int):
    """Loads `(Q, BAG, REF)` from an H452 Ultra-GP context `.npz` for `cv2`, `cv5`, `cv7`, or `2xcv7` (`0` `ya` read!)."""
    if p is None or not p.is_file():
        return None
    with np.load(p, allow_pickle=False) as z:
        assert "ya" not in z.files, "Ultra-GP artifact must never store ya!"
        q_key = f"Q_{cv_mode}"
        b_key = f"AKF_{cv_mode}" if f"AKF_{cv_mode}" in z.files else "AKF_cv7"
        if q_key not in z.files or z[q_key].ndim != 2 or z[q_key].shape[1] == 0:
            return None
        p_key = f"PA_{cv_mode}" if f"PA_{cv_mode}" in z.files else "PA"
        Q = np.asarray(z[q_key], dtype=np.float64)
        B = np.asarray(z[b_key], dtype=np.float64)
        R = np.asarray(z[p_key], dtype=np.float64)
        if Q.shape[0] != n_s or B.shape[0] != n_a or R.shape[0] != n_a:
            return None
        ok_cols = [
            j
            for j in range(Q.shape[1])
            if np.all(np.isfinite(Q[:, j])) and np.all(np.isfinite(B[:, j]))
        ]
        if not ok_cols:
            return None
        Q = Q[:, ok_cols]
        B = B[:, ok_cols]
        R_clean = np.column_stack(
            [R[:, j] if np.all(np.isfinite(R[:, j])) else B[:, j] for j in ok_cols]
        )
        return Q, B, R_clean


def _score_ctx_job(job):
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
        blk_base_gpu,
        blk_traj_e8,
        blk_cv7_hyb,
        blk_ugp,
        ugp_tag,
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

    # 1. Control: H450 Legacy Guard Stack on base blocks
    base_named = [
        (tag, blk)
        for tag, blk in (("m7", blk_m), ("k3", blk_k), ("f4", blk_f), ("tab450", blk_tab), ("gpu450", blk_base_gpu))
        if blk is not None
    ]
    pred_ctrl_leg, _ = GM.guard_and_stack_2l(c, Xs, Xa, y, lab, Q_gp, bcols_gp, ref_gp, [b for _, b in base_named])
    u_ctrl_leg = float(M.utility(pred_ctrl_leg, ya, ba)[0])

    # 2. Ablation A: New H452 CV-7 Pool-Agnostic Stacker on the EXACT same base blocks (isolates stacker fix!)
    pred_stk_only, _ = stack_guard_cv7_context(c, Xs, Xa, y, lab, Q_gp, bcols_gp, ref_gp, base_named, use_cv7_folds=True)
    u_stk_only = float(M.utility(pred_stk_only, ya, ba)[0])

    # 3. Full H452 Hybrid Champion (Base + Traj-e8 + 2xCV-7 CW12/GP-Resid + Ultra-GP)
    full_named = list(base_named)
    if blk_traj_e8 is not None:
        full_named.append(("e8_traj", blk_traj_e8))
    if blk_cv7_hyb is not None:
        full_named.append(("e8_cv7_hyb", blk_cv7_hyb))
    if blk_ugp is not None:
        full_named.append((ugp_tag, blk_ugp))

    pred_h452, diag_h452 = stack_guard_cv7_context(c, Xs, Xa, y, lab, Q_gp, bcols_gp, ref_gp, full_named, use_cv7_folds=True)
    u_h452 = float(M.utility(pred_h452, ya, ba)[0])

    # 4. Pure Neural e8 + CW12/GP-Resid arm (no legacy Q_gp)
    neural_named = [
        (tag, blk)
        for tag, blk in (("gpu450", blk_base_gpu), ("e8_traj", blk_traj_e8), ("e8_cv7_hyb", blk_cv7_hyb))
        if blk is not None
    ]
    if neural_named:
        pred_neural, _ = stack_guard_cv7_context(c, Xs, Xa, y, lab, None, None, None, neural_named, use_cv7_folds=True)
        u_neural = float(M.utility(pred_neural, ya, ba)[0])
    else:
        u_neural = float(M.utility(mean_a, ya, ba)[0])

    return s, c, u_ctrl_leg, u_stk_only, u_h452, u_neural, (diag_h452 if s == SEEDS[0] else None)


def _rank_entrant_h452(H, base_field: dict, eid: str, u_ent: dict, fam: dict, keep_cids: list[str], n_boot: int) -> dict:
    fk = {c: fam[c] for c in keep_cids}
    merged = {**base_field, eid: u_ent}
    panel = dict(sorted({e: {c: v[c] for c in keep_cids} for e, v in merged.items()}.items(), key=lambda kv: kv[0] + ".jsonl"))
    arms, point, boot, elo = T16.elo_boot(np, H, panel, fk, ANCHOR, LADDER_SEED, n_boot)
    order = sorted(elo, key=lambda e: -elo[e]["elo"])
    focus = [e for e in FOCUS_ENTRANTS if e in panel and e != eid]
    return {
        "entrant": eid,
        "contexts": len(keep_cids),
        "families": len(set(fk.values())),
        "rank_if_placed_against_frozen_33": order.index(eid) + 1,
        "n_entrants_in_calc": len(order),
        "elo": elo[eid],
        "family_mean_u": H.family_mean(panel[eid], fk),
        "top_10_order": [
            {"entrant": e, "rank": i + 1, "elo": elo[e]["elo"], "ci95": elo[e]["ci95"], "family_mean_u": H.family_mean(panel[e], fk)}
            for i, e in enumerate(order[:10])
        ],
        "gap_elo": {f"vs_{e}": elo[eid]["elo"] - elo[e]["elo"] for e in focus},
        "delta_elo": {e: T16.delta_elo(np, arms, point, boot, eid, e) for e in focus},
        "pairs": {e: T16.pair_stats(np, H, panel[eid], panel[e], fk) for e in focus},
    }


def _rank_worker(job):
    key, base_field, eid, u_ent, fam, keep_cids, n_boot = job
    _, H = T16._stats_mod()
    return key, _rank_entrant_h452(H, base_field, eid, u_ent, fam, keep_cids, n_boot)


def score_h452(
    out_json: Path,
    base_gpu_dir: Path,
    traj_e8_dir: Path | None = None,
    cv7_hyb_dir: Path | None = None,
    ultra_gp_dir: Path | None = None,
    ultra_gp_cv_mode: str = "cv7",
    procs: int = 16,
    n_boot: int = N_BOOT,
) -> dict:
    _check_h452_quarantine(out_json)
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

    ugp_tag = f"ugp_{ultra_gp_cv_mode}"
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
        blk_base_gpu = _load_col_block(base_gpu_dir / f"{c}.npz", H450_GPU_COLS, n_s, n_a)
        blk_traj_e8 = _load_col_block(traj_e8_dir / f"{c}.npz", TRAJ_E8_COLS, n_s, n_a) if traj_e8_dir else None
        blk_cv7_hyb = _load_col_block(cv7_hyb_dir / f"{c}.npz", CV7_HYBRID_COLS, n_s, n_a) if cv7_hyb_dir else None
        blk_ugp = _load_ultra_gp_block(ultra_gp_dir / f"{c}.npz", ultra_gp_cv_mode, n_s, n_a) if ultra_gp_dir else None

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
                    blk_base_gpu,
                    blk_traj_e8,
                    blk_cv7_hyb,
                    blk_ugp,
                    ugp_tag,
                )
            )

    with Pool(processes=min(procs, len(jobs))) as pool:
        done = pool.map(_score_ctx_job, jobs)

    u_ctrl_s = {s: {} for s in SEEDS}
    u_stk_s = {s: {} for s in SEEDS}
    u_h452_s = {s: {} for s in SEEDS}
    u_neur_s = {s: {} for s in SEEDS}
    diags = {}
    for s, c, u_ctrl, u_stk, u_452, u_neu, dg in done:
        u_ctrl_s[s][c] = u_ctrl
        u_stk_s[s][c] = u_stk
        u_h452_s[s][c] = u_452
        u_neur_s[s][c] = u_neu
        if dg is not None:
            diags[c] = dg

    u_ctrl = {c: 0.5 * (u_ctrl_s[SEEDS[0]][c] + u_ctrl_s[SEEDS[1]][c]) for c in cids}
    u_stk = {c: 0.5 * (u_stk_s[SEEDS[0]][c] + u_stk_s[SEEDS[1]][c]) for c in cids}
    u_h452 = {c: 0.5 * (u_h452_s[SEEDS[0]][c] + u_h452_s[SEEDS[1]][c]) for c in cids}
    u_neur = {c: u_neur_s[SEEDS[0]][c] for c in cids}

    rank_jobs = [
        ("ctrl", field, "h450.LEGACY_GUARD_CONTROL", u_ctrl, fam, cids, n_boot),
        ("stk_only", field, "h452.STACK_CV7_FIX_ONLY", u_stk, fam, cids, n_boot),
        ("h452_champ", field, "h452.ULTRA_HYBRID_CHAMPION", u_h452, fam, cids, n_boot),
        ("neural_e8", field, "h452.NEURAL_E8_HYBRID_ONLY", u_neur, fam, cids, n_boot),
    ]
    with Pool(processes=min(4, procs)) as pool:
        ranked = dict(pool.map(_rank_worker, rank_jobs))

    _, H = T16._stats_mod()
    fk = {c: fam[c] for c in cids}
    pair_vs_ctrl = T16.pair_stats(np, H, u_h452, u_ctrl, fk)
    pair_stk_vs_ctrl = T16.pair_stats(np, H, u_stk, u_ctrl, fk)

    # Per-family utility table for bottleneck inspection
    fam_ids = sorted(set(fk.values()))
    per_family = {}
    for fid in fam_ids:
        fcids = [c for c in cids if fk[c] == fid]
        per_family[fid] = {
            "n_ctx": len(fcids),
            "ctrl_leg_u": float(np.mean([u_ctrl[c] for c in fcids])),
            "stk_only_u": float(np.mean([u_stk[c] for c in fcids])),
            "h452_champ_u": float(np.mean([u_h452[c] for c in fcids])),
            "kumo_large_u": float(np.mean([field["kumo_tabular_large_e8.v1"][c] for c in fcids])),
            "tabfm_ens_u": float(np.mean([field["tabfm_ensemble.v1"][c] for c in fcids])),
        }

    doc = {
        "schema": "jope.h452.quarantined_dev91_readout/v1",
        "quarantined_in_corpus_random_split_diagnostic": True,
        "canonical_promotion_forbidden": True,
        "inputs": {
            "base_gpu_dir": str(base_gpu_dir),
            "traj_e8_dir": str(traj_e8_dir) if traj_e8_dir else None,
            "cv7_hyb_dir": str(cv7_hyb_dir) if cv7_hyb_dir else None,
            "ultra_gp_dir": str(ultra_gp_dir) if ultra_gp_dir else None,
            "ultra_gp_cv_mode": ultra_gp_cv_mode,
        },
        "h452_ultra_hybrid_champion_91": ranked["h452_champ"],
        "h452_stack_cv7_fix_only_91": ranked["stk_only"],
        "h452_neural_e8_hybrid_only_91": ranked["neural_e8"],
        "h450_legacy_guard_control_91": ranked["ctrl"],
        "paired_vs_h450_legacy_control": {
            "h452_champ_vs_ctrl": pair_vs_ctrl,
            "stk_only_vs_ctrl": pair_stk_vs_ctrl,
        },
        "per_family_u": per_family,
        "per_context_u": {
            c: {
                "family_id": fk[c],
                "ctrl_leg": u_ctrl[c],
                "stk_only": u_stk[c],
                "h452_champ": u_h452[c],
                "neural_e8": u_neur[c],
            }
            for c in cids
        },
        "context_diags": diags,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(doc, indent=2, sort_keys=True, default=lambda o: o.item()) + "\n")
    return doc


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--base-gpu-dir", default="/home/ubuntu/jope-data/h450-quarantine/r4/cols/g12_r4_L_A_r4_L_B")
    ap.add_argument("--traj-e8-dir", default="")
    ap.add_argument("--cv7-hyb-dir", default="")
    ap.add_argument("--ultra-gp-dir", default="")
    ap.add_argument("--ultra-gp-cv-mode", default="cv7", choices=("cv2", "cv5", "cv7", "2xcv7"))
    ap.add_argument("--procs", type=int, default=16)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    a = ap.parse_args()

    doc = score_h452(
        Path(a.out_json),
        Path(a.base_gpu_dir),
        Path(a.traj_e8_dir) if a.traj_e8_dir else None,
        Path(a.cv7_hyb_dir) if a.cv7_hyb_dir else None,
        Path(a.ultra_gp_dir) if a.ultra_gp_dir else None,
        ultra_gp_cv_mode=a.ultra_gp_cv_mode,
        procs=a.procs,
        n_boot=a.n_boot,
    )
    ch = doc["h452_ultra_hybrid_champion_91"]
    st = doc["h452_stack_cv7_fix_only_91"]
    ct = doc["h450_legacy_guard_control_91"]
    ne = doc["h452_neural_e8_hybrid_only_91"]
    print(
        json.dumps(
            {
                "ctrl_legacy_elo": round(float(ct["elo"]["elo"]), 1),
                "ctrl_legacy_u": round(float(ct["family_mean_u"]), 4),
                "stack_fix_only_elo": round(float(st["elo"]["elo"]), 1),
                "stack_fix_only_u": round(float(st["family_mean_u"]), 4),
                "h452_champion_elo": round(float(ch["elo"]["elo"]), 1),
                "h452_champion_rank": int(ch["rank_if_placed_against_frozen_33"]),
                "h452_champion_u": round(float(ch["family_mean_u"]), 4),
                "neural_e8_only_elo": round(float(ne["elo"]["elo"]), 1),
                "gap_vs_tabfm_ens": round(float(ch["gap_elo"]["vs_tabfm_ensemble.v1"]), 1),
                "gap_vs_kumo_small": round(float(ch["gap_elo"]["vs_kumo_tabular_small_e8.v1"]), 1),
                "gap_vs_tabfm_v1": round(float(ch["gap_elo"]["vs_tabfm.v1"]), 1),
                "gap_vs_kumo_large": round(float(ch["gap_elo"]["vs_kumo_tabular_large_e8.v1"]), 1),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
