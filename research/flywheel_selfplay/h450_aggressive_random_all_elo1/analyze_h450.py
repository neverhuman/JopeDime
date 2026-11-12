#!/usr/bin/env python3
"""H450 Quarantined Diagnostic Readout (`analyze_h450.py`) on `xbabe2`.

STRICT QUARANTINE & SCIENTIFIC LABELING:
  - Refuses any path inside canonical `reference-board`, `h335-stagev/refboard`, or `h335-cl/neural`.
  - Evaluates strictly under the legal `(X_search, y_search, X_audit) -> y_hat_audit` contract (`0` `y_audit` read at
    feature construction, model inference, or stacking; `y_audit` read ONLY inside `M.utility(pred, ya, ba)`):
      A. FAIR OUT-OF-SAMPLE CONTROL (`0%` `dev91` neural weights):
         - `h450.FAIR_UPGRADED_ZERO_D91_CONTROL`: `all_non_d91_{L,M}` (`0%` `dev91`) + `H450` `float64` norm/2-stage boost
           + `cols_tab_h450` + `M7/K3/F4/SXB-GP` + `stack_h450`!
      B. QUARANTINED IN-CORPUS RANDOM-SPLIT FIT (`dev91` included in combined `train U test` random-split corpus;
         measures in-corpus fit under random splits, NOT out-of-sample generalisation):
         - `h450.AGGRESSIVE_RANDOM_ALL_CHAMPION`: 4 random-split checkpoints (`L_A, M_A, L_B, M_B`) + `cols_tab_h450`
           + `M7/K3/F4/SXB-GP` + `stack_h450`
         - `h450.NEURAL_PLUS_TAB_NO_GP`: `H450-GPU + H450-Tab` only (`0` legacy `M7/K3/F4/SXB-GP`) + `stack_h450`
         - `h450.ICL_BOOST_ONLY`: Pure neural `H450` `icl_boost2s_e16` (`0` trees/GP!)
         - `h450.LEGACY_GUARD_STACK`: Same `H450` blocks passed through `GM.guard_and_stack_2l` (stacker ablation)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
if "/home/ubuntu/h448-stage/pub-r1/code" not in sys.path and Path("/home/ubuntu/h448-stage/pub-r1/code").is_dir():
    sys.path.append("/home/ubuntu/h448-stage/pub-r1/code")

import guard_matern_core as GM  # noqa: E402
import meta_h416 as M  # noqa: E402
import train_h416 as T16  # noqa: E402
from cols_h450_gpu import H450_GPU_COLS  # noqa: E402
from cols_tab_h450 import H450_TAB_COLS  # noqa: E402
import stack_h450 as ST  # noqa: E402

WAVE = "H450"
CHAMPION = "h450.AGGRESSIVE_RANDOM_ALL_CHAMPION"
NO_GP_ARM = "h450.NEURAL_PLUS_TAB_NO_GP"
ICL_ONLY_ARM = "h450.ICL_BOOST_ONLY"
LEGACY_STACK_ARM = "h450.LEGACY_GUARD_STACK"
FAIR_ZERO_D91_CTRL = "h450.FAIR_UPGRADED_ZERO_D91_CONTROL"

FAIR_CANONICAL = "h448.APEX_FAIR_CHAMPION"
SXB_ENTRANT = "h416.SXB"
X0B = T16.X0B
ANCHOR = T16.ANCHOR
KUMO_LARGE = "kumo_tabular_large_e8.v1"
KUMO_SMALL = "kumo_tabular_small_e8.v1"
TABFM = "tabfm_ensemble.v1"
LADDER_SEED, N_BOOT = T16.LADDER_SEED, T16.N_BOOT
SEEDS = (335101, 335102)
TAGS = {335101: "D1", 335102: "D2"}
H430_K3 = ("key_direct", "key_gbm", "factor_glm")
H433_F4 = ("form_gbm", "form_gam", "form_sr", "form_interp")
H335_STATS_SHA256 = "ee3b45439a08bc395d53079570ed8965b61aad8c2d2e4e89fd1b32c490bbccb1"


def _sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _jd(o):
    if hasattr(o, "item"):
        return o.item()
    raise TypeError(type(o).__name__)


def _check_quarantined_path(p: Path) -> None:
    rp = str(p.resolve())
    forbidden = ("reference-board/boards/dev91-audit-v1", "h335-stagev/refboard", "h335-cl/neural")
    if any(tok in rp for tok in forbidden) or ("/H450/" not in rp and "/h450" not in rp):
        raise SystemExit(f"QUARANTINE REFUSAL: path {rp} is outside H450 quarantine")


def _load_col_block(p: Path, names: tuple[str, ...], n_s: int, n_a: int):
    if not p.is_file():
        return None
    z = np.load(p, allow_pickle=False)
    if not bool(z["ok"]):
        return None
    if not all(f"q_{v}" in z.files and len(z[f"q_{v}"]) == n_s and len(z[f"ref_{v}"]) == n_a for v in names):
        return None
    Q = np.column_stack([np.asarray(z[f"q_{v}"], np.float64) for v in names])
    BAG = np.column_stack([np.asarray(z[f"bag_{v}"], np.float64) for v in names])
    REF = np.column_stack([np.asarray(z[f"ref_{v}"], np.float64) for v in names])
    if not (np.all(np.isfinite(Q)) and np.all(np.isfinite(BAG)) and np.all(np.isfinite(REF))):
        return None
    return Q, BAG, REF


def _eval_ctx(job):
    s, c, npz_p, Xs, Xa, y, lab, blk_m, blk_k, blk_f, blk_tab, blk_gpu, blk_zero_gpu = job
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
    for tag, blk in (("m7", blk_m), ("k3", blk_k), ("f4", blk_f), ("tab450", blk_tab), ("gpu450", blk_gpu)):
        if blk is not None:
            named_full.append((tag, blk))

    pred_champ, diag_champ = ST.stack_h450_context(c, Xs, Xa, y, lab, Q_gp, bcols_gp, ref_gp, named_full)
    u_champ = float(M.utility(pred_champ, ya, ba)[0])

    # Ablation 1: Neural + Tabular H450 ONLY (0 GP / 0 M7/K3/F4)
    named_nogp = [(tag, blk) for tag, blk in (("tab450", blk_tab), ("gpu450", blk_gpu)) if blk is not None]
    pred_nogp, _ = ST.stack_h450_context(c, Xs, Xa, y, lab, None, None, None, named_nogp)
    u_nogp = float(M.utility(pred_nogp, ya, ba)[0])

    # Ablation 2: Pure Neural H450 (`icl_boost2s_e16` adaptive bag/refit, index 5 of H450_GPU_COLS; or best pure H450 neural OOF)
    if blk_gpu is not None:
        pred_icl, _ = ST.stack_h450_context(c, Xs, Xa, y, lab, None, None, None, [("gpu450", blk_gpu)])
        u_icl = float(M.utility(pred_icl, ya, ba)[0])
    else:
        u_icl = float(M.utility(mean_a, ya, ba)[0])

    # Ablation 3: Legacy Guard+2L Stacker on the exact same blocks
    raw_full = [blk for _, blk in named_full]
    pred_leg, _ = GM.guard_and_stack_2l(c, Xs, Xa, y, lab, Q_gp, bcols_gp, ref_gp, raw_full)
    u_leg = float(M.utility(pred_leg, ya, ba)[0])

    # Fair Out-of-Sample Control (`0%` `dev91` neural weights + H450 upgrades)
    u_zero = None
    if blk_zero_gpu is not None:
        named_zero = []
        for tag, blk in (("m7", blk_m), ("k3", blk_k), ("f4", blk_f), ("tab450", blk_tab), ("gpu0d91", blk_zero_gpu)):
            if blk is not None:
                named_zero.append((tag, blk))
        pred_zero, _ = ST.stack_h450_context(c, Xs, Xa, y, lab, Q_gp, bcols_gp, ref_gp, named_zero)
        u_zero = float(M.utility(pred_zero, ya, ba)[0])

    return s, c, u_champ, u_nogp, u_icl, u_leg, u_zero, (diag_champ if s == SEEDS[0] else None)


def _rank_entrant(H, base_field: dict, eid: str, u_ent: dict, fam: dict, keep_cids: list[str], n_boot: int) -> dict:
    fk = {c: fam[c] for c in keep_cids}
    merged = {**base_field, eid: u_ent}
    panel = dict(sorted({e: {c: v[c] for c in keep_cids} for e, v in merged.items()}.items(), key=lambda kv: kv[0] + ".jsonl"))
    arms, point, boot, elo = T16.elo_boot(np, H, panel, fk, ANCHOR, LADDER_SEED, n_boot)
    order = sorted(elo, key=lambda e: -elo[e]["elo"])
    focus_candidates = (KUMO_LARGE, TABFM, KUMO_SMALL, FAIR_CANONICAL, SXB_ENTRANT, X0B)
    focus = [e for e in focus_candidates if e in panel and e != eid]
    return {
        "entrant": eid,
        "contexts": len(keep_cids),
        "families": len(set(fk.values())),
        "rank_if_placed_against_frozen_33": order.index(eid) + 1,
        "n_entrants_in_calc": len(order),
        "elo": elo[eid],
        "family_mean_u": H.family_mean(panel[eid], fk),
        "top_10_order": [{"entrant": e, "rank": i + 1, "elo": elo[e]["elo"], "ci95": elo[e]["ci95"]} for i, e in enumerate(order[:10])],
        "gap_elo": {f"vs_{e}": elo[eid]["elo"] - elo[e]["elo"] for e in focus},
        "delta_elo": {e: T16.delta_elo(np, arms, point, boot, eid, e) for e in focus},
        "pairs": {e: T16.pair_stats(np, H, panel[eid], panel[e], fk) for e in focus},
    }


def _rank_worker(job):
    key, base_field, eid, u_ent, fam, keep_cids, n_boot = job
    _, H = T16._stats_mod()
    return key, _rank_entrant(H, base_field, eid, u_ent, fam, keep_cids, n_boot)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    for k in ("bank", "gpu-cols", "tab-cols", "h448-pulled", "h418-pulled", "board", "lock", "receipt", "diag-out", "diag-txt", "diag-board"):
        ap.add_argument(f"--{k}", required=True)
    ap.add_argument("--zero-d91-gpu-cols", default="")
    ap.add_argument("--procs", type=int, default=16)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    a = ap.parse_args(argv)

    diag_out = Path(a.diag_out)
    diag_txt = Path(a.diag_txt)
    diag_board = Path(a.diag_board)
    for p in (diag_out, diag_txt, diag_board):
        _check_quarantined_path(p)
        if p.exists():
            raise SystemExit(f"{p} already exists (create-once)")

    _, H = T16._stats_mod()
    if _sha(Path(H.__file__)) != H335_STATS_SHA256:
        raise SystemExit("h335_stats sha mismatch")

    lk = json.loads(Path(a.lock).read_text())
    lock_sha = _sha(Path(a.lock))
    rec_sha = Path(a.receipt).read_text().split()[0]
    if lk.get("schema") != "jope.h450.quarantined_lock/v1" or rec_sha != lock_sha:
        raise SystemExit("H450 lock / receipt mismatch")

    bdir = Path(a.board)
    board_doc = json.loads((bdir / "BOARD.json").read_text())
    env = {c["cid"]: c["envelope_sha256"] for c in board_doc["contexts"]}
    fam = {c["cid"]: c["family_id"] for c in board_doc["contexts"]}
    cids = sorted(env)
    field = {}
    for sp in sorted((bdir / "scores").glob("*.jsonl")):
        recs = [json.loads(x) for x in sp.read_text().splitlines() if x.strip()]
        field[sp.stem] = {r["cid"]: float(r["utility"]) for r in recs}
    if FAIR_CANONICAL not in field or KUMO_LARGE not in field or len(field) < 33:
        raise SystemExit("post-H448 board must contain h448.APEX_FAIR_CHAMPION and kumo_tabular_large_e8.v1 (>=33 entrants)")

    P448 = Path(a.h448_pulled)
    P418 = Path(a.h418_pulled)
    gpu_dir = Path(a.gpu_cols)
    tab_dir = Path(a.tab_cols)
    zero_dir = Path(a.zero_d91_gpu_cols) if a.zero_d91_gpu_cols else None
    has_zero = bool(zero_dir and zero_dir.is_dir())

    clear_cids = cids
    scr_p = P448 / "SCREEN_D91_SEARCH.json"
    if scr_p.is_file():
        scr_doc = json.loads(scr_p.read_text())
        per_ctx = scr_doc.get("per_context") or {}
        if per_ctx:
            cand = [c for c in cids if not (per_ctx.get(c) or {}).get("flag_feature", False)]
            if cand:
                clear_cids = cand

    bank_rows = {}
    with np.load(a.bank, allow_pickle=False) as zb:
        b_cids = [str(x) for x in zb["cids"].tolist()]
        ss, sa = zb["s_shape"], zb["a_shape"]
        so = ao = lo = 0
        for i, c in enumerate(b_cids):
            n, w = int(ss[i, 0]), int(ss[i, 1])
            na, f = int(sa[i, 0]), int(sa[i, 1])
            S_mat = np.array(zb["search"][so : so + n * w], np.float64).reshape(n, w)
            Xa = np.array(zb["audit"][ao : ao + na * f], np.float64, copy=True).reshape(na, f)
            Xs = np.array(S_mat[:, :-1], np.float64, copy=True)
            y = np.array(zb["y64"][lo : lo + n], np.float64, copy=True)
            lab = np.array(zb["lab"][lo : lo + n], np.int64, copy=True)
            bank_rows[c] = (Xs, Xa, y, lab)
            so += n * w
            ao += na * f
            lo += n

    jobs = []
    gpu_ok = tab_ok = zero_ok = 0
    for c in cids:
        Xs, Xa, y, lab = bank_rows[c]
        n_s, n_a = len(y), len(Xa)
        blk_m = _load_col_block(P448 / "cols_matern_D91" / f"{c}.npz", GM.PRIMARY_NAMES, n_s, n_a)
        blk_k_raw = _load_col_block(P448 / "cols_k3_D91" / f"{c}.npz", H430_K3, n_s, n_a)
        if blk_k_raw is not None:
            Qk, Bk, Rk = [arr.copy() for arr in blk_k_raw]
            y_lo, y_hi = float(np.min(y)), float(np.max(y))
            y_pad = 0.25 * max(y_hi - y_lo, float(np.std(y)), 1e-9)
            Qk[:, 0] = np.clip(Qk[:, 0], y_lo - y_pad, y_hi + y_pad)
            Bk[:, 0] = np.clip(Bk[:, 0], y_lo - y_pad, y_hi + y_pad)
            Rk[:, 0] = np.clip(Rk[:, 0], y_lo - y_pad, y_hi + y_pad)
            blk_k = (Qk, Bk, Rk)
        else:
            blk_k = None
        blk_f_raw = _load_col_block(P448 / "cols_f4_D91" / f"{c}.npz", H433_F4, n_s, n_a)
        if blk_f_raw is not None:
            Qf, Bf, Rf = [arr.copy() for arr in blk_f_raw]
            y_lo, y_hi = float(np.min(y)), float(np.max(y))
            y_pad = 0.50 * max(y_hi - y_lo, float(np.std(y)), 1e-9)
            for idx_f in (2, 3):
                Qf[:, idx_f] = np.clip(Qf[:, idx_f], y_lo - y_pad, y_hi + y_pad)
                Bf[:, idx_f] = np.clip(Bf[:, idx_f], y_lo - y_pad, y_hi + y_pad)
                Rf[:, idx_f] = np.clip(Rf[:, idx_f], y_lo - y_pad, y_hi + y_pad)
            blk_f = (Qf, Bf, Rf)
        else:
            blk_f = None

        blk_tab = _load_col_block(tab_dir / f"{c}.npz", H450_TAB_COLS, n_s, n_a)
        blk_gpu = _load_col_block(gpu_dir / f"{c}.npz", H450_GPU_COLS, n_s, n_a)
        blk_zero = _load_col_block(zero_dir / f"{c}.npz", H450_GPU_COLS, n_s, n_a) if has_zero else None
        tab_ok += blk_tab is not None
        gpu_ok += blk_gpu is not None
        zero_ok += blk_zero is not None

        for s in SEEDS:
            jobs.append((s, c, P418 / TAGS[s] / f"h416_{TAGS[s]}_npz" / f"{c}.npz", Xs, Xa, y, lab, blk_m, blk_k, blk_f, blk_tab, blk_gpu, blk_zero))

    if gpu_ok < 90 or tab_ok < 90:
        raise SystemExit(f"incomplete H450 columns: gpu_ok={gpu_ok}/91 tab_ok={tab_ok}/91")

    with Pool(processes=min(a.procs, len(jobs))) as pool:
        done = pool.map(_eval_ctx, jobs)

    u_champ_s = {s: {} for s in SEEDS}
    u_nogp_s = {s: {} for s in SEEDS}
    u_icl_s = {s: {} for s in SEEDS}
    u_leg_s = {s: {} for s in SEEDS}
    u_zero_s = {s: {} for s in SEEDS}
    ctx_diags = {}

    for s, c, uc, ung, ui, ul, uz, dg in done:
        u_champ_s[s][c] = uc
        u_nogp_s[s][c] = ung
        u_icl_s[s][c] = ui
        u_leg_s[s][c] = ul
        if uz is not None:
            u_zero_s[s][c] = uz
        if dg is not None:
            ctx_diags[c] = dg

    u_champ = {c: 0.5 * (u_champ_s[SEEDS[0]][c] + u_champ_s[SEEDS[1]][c]) for c in cids}
    u_nogp = {c: u_nogp_s[SEEDS[0]][c] for c in cids}
    u_icl = {c: u_icl_s[SEEDS[0]][c] for c in cids}
    u_leg = {c: 0.5 * (u_leg_s[SEEDS[0]][c] + u_leg_s[SEEDS[1]][c]) for c in cids}
    u_zero = {c: 0.5 * (u_zero_s[SEEDS[0]][c] + u_zero_s[SEEDS[1]][c]) for c in cids} if (has_zero and zero_ok == 91) else None

    rank_jobs = [
        ("champ_91", field, CHAMPION, u_champ, fam, cids, a.n_boot),
        ("champ_clear", field, CHAMPION, u_champ, fam, clear_cids, min(a.n_boot, 2500)),
        ("nogp_91", field, NO_GP_ARM, u_nogp, fam, cids, min(a.n_boot, 2500)),
        ("icl_91", field, ICL_ONLY_ARM, u_icl, fam, cids, min(a.n_boot, 2500)),
        ("leg_91", field, LEGACY_STACK_ARM, u_leg, fam, cids, min(a.n_boot, 2500)),
    ]
    if u_zero is not None:
        rank_jobs.extend([
            ("zero_91", field, FAIR_ZERO_D91_CTRL, u_zero, fam, cids, a.n_boot),
            ("zero_clear", field, FAIR_ZERO_D91_CTRL, u_zero, fam, clear_cids, min(a.n_boot, 2500)),
        ])

    with Pool(processes=min(a.procs, len(rank_jobs))) as pool:
        ranked = dict(pool.map(_rank_worker, rank_jobs))

    # Per-family mean utility breakdown (`all_91`)
    by_fam = {}
    for fk_id in sorted(set(fam.values())):
        fcids = [c for c in cids if fam[c] == fk_id]
        by_fam[fk_id] = {
            "n_ctx": len(fcids),
            "h448_fair": round(float(np.mean([field[FAIR_CANONICAL][c] for c in fcids])), 4),
            "h450_zero_d91_ctrl": round(float(np.mean([u_zero[c] for c in fcids])), 4) if u_zero else None,
            "h450_champion_in_corpus": round(float(np.mean([u_champ[c] for c in fcids])), 4),
            "h450_icl_boost_only_in_corpus": round(float(np.mean([u_icl[c] for c in fcids])), 4),
            "kumo_large_ref": round(float(np.mean([field[KUMO_LARGE][c] for c in fcids])), 4),
        }

    # Quarantined diagnostic score exports (NEVER touching canonical reference-board!)
    shutil.copytree(bdir, diag_board)
    os.chmod(diag_board, 0o755)
    os.chmod(diag_board / "scores", 0o755)
    exports = [
        (CHAMPION, u_champ, "quarantined_in_corpus_random_split_fit", True),
        (NO_GP_ARM, u_nogp, "quarantined_in_corpus_random_split_fit", True),
        (ICL_ONLY_ARM, u_icl, "quarantined_in_corpus_random_split_fit", True),
        (LEGACY_STACK_ARM, u_leg, "quarantined_in_corpus_random_split_fit", True),
    ]
    if u_zero is not None:
        exports.append((FAIR_ZERO_D91_CTRL, u_zero, "quarantined_fair_zero_d91_control", False))

    for eid, u_map, st_label, is_incorp in exports:
        lines = [
            json.dumps({
                "board": board_doc["board"],
                "board_sha256": board_doc["board_sha256"],
                "cid": c,
                "envelope_sha256": env[c],
                "model_id": eid,
                "utility": u_map[c],
                "status": st_label,
                "in_corpus_d91_random_split_fit": is_incorp,
                "canonical_promotion_forbidden": True,
            }, sort_keys=True)
            for c in cids
        ]
        (diag_board / "scores" / f"{eid}.jsonl").write_text("\n".join(lines) + "\n")

    matched_random_split_delta = None
    if u_zero is not None:
        matched_random_split_delta = {
            "elo_diff_champion_vs_zero_d91_ctrl_91": ranked["champ_91"]["elo"]["elo"] - ranked["zero_91"]["elo"]["elo"],
            "pairs_91": T16.pair_stats(np, H, u_champ, u_zero, fam),
        }

    out_doc = {
        "schema": "jope.h450.quarantined_readout/v1",
        "wave": WAVE,
        "scientific_interpretation": (
            "Scores for arms trained on everything_combined_f64_h450_bank.npz (which includes the 91 dev91 tables "
            "merged train U test and CSPRNG-shuffled under online random splits) measure IN-CORPUS FIT to dev91 "
            "under random splits, NOT out-of-sample generalisation. Kept strictly quarantined outside canonical dev91-audit-v1."
        ),
        "canonical_promotion_forbidden": True,
        "canonical_board_modified": False,
        "lock_sha256": lock_sha,
        "columns_ok": {"gpu_h450": gpu_ok, "tab_h450": tab_ok, "zero_d91_gpu": zero_ok},
        "fair_zero_d91_upgraded_control_91": ranked.get("zero_91"),
        "fair_zero_d91_upgraded_control_h425_clear": ranked.get("zero_clear"),
        "in_corpus_random_split_champion_91": ranked["champ_91"],
        "in_corpus_random_split_champion_h425_clear": ranked["champ_clear"],
        "in_corpus_neural_plus_tab_no_gp_91": ranked["nogp_91"],
        "in_corpus_icl_boost_only_91": ranked["icl_91"],
        "in_corpus_legacy_guard_stack_91": ranked["leg_91"],
        "matched_random_split_d91_delta": matched_random_split_delta,
        "per_family_mean_u_91": by_fam,
        "per_context_stack_diag": ctx_diags,
    }

    diag_out.parent.mkdir(parents=True, exist_ok=True)
    with open(diag_out, "x") as f:
        f.write(json.dumps(out_doc, indent=2, sort_keys=True, default=_jd) + "\n")

    rc = ranked["champ_91"]
    rn = ranked["nogp_91"]
    ri = ranked["icl_91"]
    rl = ranked["leg_91"]
    rz = ranked.get("zero_91")

    txt_lines = [
        "==========================================================================================",
        "H450 QUARANTINED DIAGNOSTIC READOUT (CANONICAL PROMOTION FORBIDDEN; 0 BOARD FILES TOUCHED)",
        "NOTE: Arms trained with dev91 in the combined random-split corpus measure IN-CORPUS FIT",
        "      to dev91 under random splits, NOT out-of-sample generalisation.",
        "==========================================================================================",
    ]
    if rz is not None:
        ez = rz["elo"]
        txt_lines.append(
            f"[PART A — 0% DEV91 OUT-OF-SAMPLE CONTROL] {FAIR_ZERO_D91_CTRL}: "
            f"Elo {ez['elo']:.1f} [{ez['ci95'][0]:.1f}, {ez['ci95'][1]:.1f}], "
            f"virtual rank #{rz['rank_if_placed_against_frozen_33']}/{rz['n_entrants_in_calc']}, "
            f"family_mean_u={rz['family_mean_u']:.4f} "
            f"(vs h448.APEX_FAIR_CHAMPION: {rz['gap_elo']['vs_' + FAIR_CANONICAL]:+.1f} Elo)"
        )
    for label, r_obj in (
        (CHAMPION, rc),
        (NO_GP_ARM, rn),
        (ICL_ONLY_ARM, ri),
        (LEGACY_STACK_ARM, rl),
    ):
        el = r_obj["elo"]
        txt_lines.append(
            f"[PART B — IN-CORPUS RANDOM-SPLIT FIT] {label}: "
            f"BT-Elo {el['elo']:.1f} [{el['ci95'][0]:.1f}, {el['ci95'][1]:.1f}], "
            f"virtual pos #{r_obj['rank_if_placed_against_frozen_33']}/{r_obj['n_entrants_in_calc']}, "
            f"family_mean_u={r_obj['family_mean_u']:.4f} "
            f"(delta vs 2070.2 top stored ref: {r_obj['gap_elo']['vs_' + KUMO_LARGE]:+.1f} Elo; "
            f"vs h448.APEX_FAIR_CHAMPION: {r_obj['gap_elo']['vs_' + FAIR_CANONICAL]:+.1f} Elo)"
        )
    txt_body = "\n".join(txt_lines) + "\n"
    with open(diag_txt, "x") as f:
        f.write(txt_body)
    print(txt_body, end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
