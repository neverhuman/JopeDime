#!/usr/bin/env python3
"""H449 quarantined `NON-DEV91-COMPLETE` vs `ENTIRE (+100% dev91)` readout (`analyze_h449_unfair.py`) on `xbabe2`.

STRICT QUARANTINE:
  - Refuses if `--diag-out` or `--diag-board` points inside canonical `reference-board` or `h335-stagev/refboard`.
  - Requires post-`H448` board (`>= 33` complete entrants including `h448.APEX_FAIR_CHAMPION` and `kumo_tabular_large_e8.v1`).
  - Evaluates under the identical `(X_search, y_search, X_audit) -> y_hat_audit` contract (`0` `y_audit` passed at test time):
      1. `h449.FAIR_ALL_NON_D91_STACK`    (`0%` `dev91`, GPU-trained on `all_non_d91_complete_bank.npz`, ~1,976 tables)
      2. `h449.FAIR_ALL_NON_D91_ICL_ONLY` (`0%` `dev91`, pure neural `R4-e8` on `all_non_d91_complete_bank.npz`)
      3. `h449.UNFAIR_ALL_D91_STACK`      (`100%` `dev91`, trained on `everything_all_d91_bank.npz`, ~2,067 tables)
      4. `h449.UNFAIR_ALL_D91_ICL_ONLY`   (`100%` `dev91`, pure neural `R4-e8` on `everything_all_d91_bank.npz`)
      5. `h449.UNFAIR_SEARCH_D91_STACK`   (optional `70%` `SEARCH`-only `dev91` control)
  - Reports both `all_91` (`91` contexts / `19` families) and `h425_without_feature_clear` (strict `H425` clear slice).
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
import guard_matern_core as GM  # noqa: E402
import meta_h416 as M  # noqa: E402
import train_h416 as T16  # noqa: E402

WAVE = "H449"
FAIR_ENTRANT = "h448.APEX_FAIR_CHAMPION"
FAIR_NON_D91_STACK = "h449.FAIR_ALL_NON_D91_STACK"
FAIR_NON_D91_ICL_ONLY = "h449.FAIR_ALL_NON_D91_ICL_ONLY"
UNFAIR_STACK = "h449.UNFAIR_ALL_D91_STACK"
UNFAIR_ICL_ONLY = "h449.UNFAIR_ALL_D91_ICL_ONLY"
UNFAIR_SEARCH_STACK = "h449.UNFAIR_SEARCH_D91_STACK"
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
    if any(tok in rp for tok in forbidden) or ("/H449/" not in rp and "/h449" not in rp):
        raise SystemExit(f"QUARANTINE REFUSAL: path {rp} is outside H449 quarantine")


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
    s, c, npz_p, Xs, Xa, y, lab, blk_m, blk_k, blk_f, blk_non_d91, blk_all, blk_srch = job
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

    # 1. Complete NON-DEV91 (`0%` dev91) arms
    u_non_stack = u_non_icl = None
    if blk_non_d91 is not None:
        act_non = [b for b in (blk_m, blk_k, blk_f, blk_non_d91) if b is not None]
        pred_non, _ = GM.guard_and_stack_2l(c, Xs, Xa, y, lab, Q_gp, bcols_gp, ref_gp, act_non)
        u_non_stack = float(M.utility(pred_non, ya, ba)[0])
        pred_non_icl = 0.5 * (blk_non_d91[1][:, 0] + blk_non_d91[2][:, 0])
        u_non_icl = float(M.utility(pred_non_icl, ya, ba)[0])

    # 2. ENTIRE (`100%` dev91) arms
    act_all = [b for b in (blk_m, blk_k, blk_f, blk_all) if b is not None]
    pred_all, _ = GM.guard_and_stack_2l(c, Xs, Xa, y, lab, Q_gp, bcols_gp, ref_gp, act_all)
    u_all_stack = float(M.utility(pred_all, ya, ba)[0])

    if blk_all is not None:
        pred_icl_only = 0.5 * (blk_all[1][:, 0] + blk_all[2][:, 0])
        u_icl_only = float(M.utility(pred_icl_only, ya, ba)[0])
    else:
        u_icl_only = float(M.utility(mean_a, ya, ba)[0])

    u_srch_stack = None
    if blk_srch is not None:
        act_srch = [b for b in (blk_m, blk_k, blk_f, blk_srch) if b is not None]
        pred_srch, _ = GM.guard_and_stack_2l(c, Xs, Xa, y, lab, Q_gp, bcols_gp, ref_gp, act_srch)
        u_srch_stack = float(M.utility(pred_srch, ya, ba)[0])

    return s, c, u_non_stack, u_non_icl, u_all_stack, u_icl_only, u_srch_stack


def _rank_entrant(H, base_field: dict, eid: str, u_ent: dict, fam: dict, keep_cids: list[str], n_boot: int, extra_peers: dict | None = None) -> dict:
    fk = {c: fam[c] for c in keep_cids}
    merged = {**base_field, **(extra_peers or {}), eid: u_ent}
    panel = dict(sorted({e: {c: v[c] for c in keep_cids} for e, v in merged.items()}.items(), key=lambda kv: kv[0] + ".jsonl"))
    arms, point, boot, elo = T16.elo_boot(np, H, panel, fk, ANCHOR, LADDER_SEED, n_boot)
    order = sorted(elo, key=lambda e: -elo[e]["elo"])
    focus_candidates = (KUMO_LARGE, FAIR_ENTRANT, FAIR_NON_D91_STACK, FAIR_NON_D91_ICL_ONLY, TABFM, KUMO_SMALL, SXB_ENTRANT, X0B)
    focus = [e for e in focus_candidates if e in panel and e != eid]
    return {
        "entrant": eid,
        "contexts": len(keep_cids),
        "families": len(set(fk.values())),
        "rank": order.index(eid) + 1,
        "n_entrants": len(order),
        "elo": elo[eid],
        "family_mean_u": H.family_mean(panel[eid], fk),
        "top_10_order": [{"entrant": e, "rank": i + 1, "elo": elo[e]["elo"], "ci95": elo[e]["ci95"]} for i, e in enumerate(order[:10])],
        "gap_elo": {f"vs_{e}": elo[eid]["elo"] - elo[e]["elo"] for e in focus},
        "delta_elo": {e: T16.delta_elo(np, arms, point, boot, eid, e) for e in focus},
        "pairs": {e: T16.pair_stats(np, H, panel[eid], panel[e], fk) for e in focus},
    }


def _rank_entrant_worker(job):
    key, base_field, eid, u_ent, fam, keep_cids, n_boot, extra_peers = job
    _, H = T16._stats_mod()
    return key, _rank_entrant(H, base_field, eid, u_ent, fam, keep_cids, n_boot, extra_peers)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k in ("pulled", "h448-pulled", "h418-pulled", "board", "lock", "receipt", "diag-out", "diag-board"):
        ap.add_argument(f"--{k}", required=True)
    ap.add_argument("--unfair-search-cols", default="")
    ap.add_argument("--procs", type=int, default=16)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    a = ap.parse_args(argv)

    diag_out = Path(a.diag_out)
    diag_board = Path(a.diag_board)
    _check_quarantined_path(diag_out)
    _check_quarantined_path(diag_board)
    if diag_out.exists() or diag_board.exists():
        raise SystemExit("diag-out or diag-board already exists (create-once)")

    _, H = T16._stats_mod()
    if _sha(Path(H.__file__)) != H335_STATS_SHA256:
        raise SystemExit("h335_stats sha mismatch")

    lk = json.loads(Path(a.lock).read_text())
    lock_sha = _sha(Path(a.lock))
    rec_sha = Path(a.receipt).read_text().split()[0]
    if lk.get("schema") != "jope.h449.unfair_lock/v1" or rec_sha != lock_sha:
        raise SystemExit("H449 lock / receipt mismatch")

    bdir = Path(a.board)
    board_doc = json.loads((bdir / "BOARD.json").read_text())
    env = {c["cid"]: c["envelope_sha256"] for c in board_doc["contexts"]}
    fam = {c["cid"]: c["family_id"] for c in board_doc["contexts"]}
    cids = sorted(env)
    field = {}
    for sp in sorted((bdir / "scores").glob("*.jsonl")):
        recs = [json.loads(x) for x in sp.read_text().splitlines() if x.strip()]
        field[sp.stem] = {r["cid"]: float(r["utility"]) for r in recs}
    if FAIR_ENTRANT not in field or KUMO_LARGE not in field or len(field) < 33:
        raise SystemExit("post-H448 board must contain h448.APEX_FAIR_CHAMPION and kumo_tabular_large_e8.v1 (>=33 entrants)")

    P = Path(a.pulled)
    P448 = Path(a.h448_pulled)
    P418 = Path(a.h418_pulled)

    clear_cids = cids
    scr_p = P448 / "SCREEN_D91_SEARCH.json"
    if scr_p.is_file():
        scr_doc = json.loads(scr_p.read_text())
        per_ctx = scr_doc.get("per_context") or {}
        if per_ctx:
            cand = [c for c in cids if not (per_ctx.get(c) or {}).get("flag_feature", False)]
        else:
            excl = set(scr_doc.get("excluded_cids", scr_doc.get("flagged_cids", [])))
            cand = [c for c in cids if c not in excl]
        if cand:
            clear_cids = cand

    bank_rows = {}
    with np.load(P / "bank_d91_search.npz", allow_pickle=False) as zb:
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

    non_d91_dir = P / "cols_fair_all_non_d91_D91"
    has_non_d91 = non_d91_dir.is_dir()
    has_srch = bool(a.unfair_search_cols and Path(a.unfair_search_cols).is_dir())
    jobs = []
    non_ok = all_ok = srch_ok = 0
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

        blk_non_d91 = _load_col_block(non_d91_dir / f"{c}.npz", ("r4_e8",), n_s, n_a) if has_non_d91 else None
        blk_all = _load_col_block(P / "cols_unfair_all_D91" / f"{c}.npz", ("r4_e8",), n_s, n_a)
        blk_srch = _load_col_block(Path(a.unfair_search_cols) / f"{c}.npz", ("r4_e8",), n_s, n_a) if has_srch else None
        non_ok += blk_non_d91 is not None
        all_ok += blk_all is not None
        srch_ok += blk_srch is not None
        for s in SEEDS:
            jobs.append((s, c, P418 / TAGS[s] / f"h416_{TAGS[s]}_npz" / f"{c}.npz", Xs, Xa, y, lab, blk_m, blk_k, blk_f, blk_non_d91, blk_all, blk_srch))

    if all_ok < 86:
        raise SystemExit(f"insufficient unfair_all_D91 columns: {all_ok}/91")
    if has_non_d91 and non_ok < 86:
        raise SystemExit(f"insufficient fair_all_non_d91_D91 columns: {non_ok}/91")

    if a.procs > 1 and len(jobs) > 1:
        with Pool(processes=min(a.procs, len(jobs))) as pool:
            done = pool.map(_eval_ctx, jobs)
    else:
        done = [_eval_ctx(j) for j in jobs]

    u_non_s = {s: {} for s in SEEDS}
    u_non_icl_s = {s: {} for s in SEEDS}
    u_all_s = {s: {} for s in SEEDS}
    u_icl_s = {s: {} for s in SEEDS}
    u_srch_s = {s: {} for s in SEEDS}
    for s, c, un, uni, ua, ui, us in done:
        if un is not None:
            u_non_s[s][c] = un
            u_non_icl_s[s][c] = uni
        u_all_s[s][c] = ua
        u_icl_s[s][c] = ui
        if us is not None:
            u_srch_s[s][c] = us

    ro_non_stack = ro_non_icl = ro_non_stack_clear = ro_non_icl_clear = None
    u_non = u_non_icl = None
    rank_jobs = []
    if has_non_d91 and non_ok == len(cids):
        u_non = {c: 0.5 * (u_non_s[SEEDS[0]][c] + u_non_s[SEEDS[1]][c]) for c in cids}
        u_non_icl = {c: u_non_icl_s[SEEDS[0]][c] for c in cids}
        rank_jobs.extend([
            ("non_stack", field, FAIR_NON_D91_STACK, u_non, fam, cids, a.n_boot, None),
            ("non_icl", field, FAIR_NON_D91_ICL_ONLY, u_non_icl, fam, cids, a.n_boot, None),
            ("non_stack_clear", field, FAIR_NON_D91_STACK, u_non, fam, clear_cids, min(a.n_boot, 2000), None),
            ("non_icl_clear", field, FAIR_NON_D91_ICL_ONLY, u_non_icl, fam, clear_cids, min(a.n_boot, 2000), None),
        ])

    u_all = {c: 0.5 * (u_all_s[SEEDS[0]][c] + u_all_s[SEEDS[1]][c]) for c in cids}
    u_icl = {c: u_icl_s[SEEDS[0]][c] for c in cids}
    # Single-entrant board rankings against exact 33-entrant post-H448 canonical board (no cross-crowding)
    rank_jobs.extend([
        ("all_stack", field, UNFAIR_STACK, u_all, fam, cids, a.n_boot, None),
        ("all_icl", field, UNFAIR_ICL_ONLY, u_icl, fam, cids, a.n_boot, None),
        ("all_stack_clear", field, UNFAIR_STACK, u_all, fam, clear_cids, min(a.n_boot, 2000), None),
        ("all_icl_clear", field, UNFAIR_ICL_ONLY, u_icl, fam, clear_cids, min(a.n_boot, 2000), None),
    ])

    if has_srch and srch_ok >= 86:
        u_srch = {c: 0.5 * (u_srch_s[SEEDS[0]][c] + u_srch_s[SEEDS[1]][c]) for c in cids}
        rank_jobs.append(("srch_stack", field, UNFAIR_SEARCH_STACK, u_srch, fam, cids, a.n_boot, None))
    else:
        u_srch = None

    if a.procs > 1 and len(rank_jobs) > 1:
        with Pool(processes=min(a.procs, len(rank_jobs))) as pool:
            ranked = dict(pool.map(_rank_entrant_worker, rank_jobs))
    else:
        ranked = dict(_rank_entrant_worker(rj) for rj in rank_jobs)

    ro_non_stack = ranked.get("non_stack")
    ro_non_icl = ranked.get("non_icl")
    ro_non_stack_clear = ranked.get("non_stack_clear")
    ro_non_icl_clear = ranked.get("non_icl_clear")
    ro_all = ranked["all_stack"]
    ro_icl = ranked["all_icl"]
    ro_all_clear = ranked["all_stack_clear"]
    ro_icl_clear = ranked["all_icl_clear"]
    ro_srch = ranked.get("srch_stack")

    # Direct matched-architecture head-to-head: ENTIRE (+100% dev91) vs ALL-NON-DEV91-COMPLETE (0% dev91)
    matched_d91_inflation = None
    if u_non is not None and u_non_icl is not None:
        matched_d91_inflation = {
            "stack_all_d91_vs_all_non_d91_91": {
                "elo_diff_single_entrant": ro_all["elo"]["elo"] - ro_non_stack["elo"]["elo"],
                "pairs": T16.pair_stats(np, H, u_all, u_non, fam),
            },
            "icl_only_all_d91_vs_all_non_d91_91": {
                "elo_diff_single_entrant": ro_icl["elo"]["elo"] - ro_non_icl["elo"]["elo"],
                "pairs": T16.pair_stats(np, H, u_icl, u_non_icl, fam),
            },
        }

    # Write quarantined diagnostic board copy (NEVER touching canonical reference-board!)
    shutil.copytree(bdir, diag_board)
    os.chmod(diag_board, 0o755)
    os.chmod(diag_board / "scores", 0o755)
    score_exports = [(UNFAIR_STACK, u_all, True), (UNFAIR_ICL_ONLY, u_icl, True)]
    if u_non is not None and u_non_icl is not None:
        score_exports.extend([(FAIR_NON_D91_STACK, u_non, False), (FAIR_NON_D91_ICL_ONLY, u_non_icl, False)])
    if u_srch is not None:
        score_exports.append((UNFAIR_SEARCH_STACK, u_srch, True))
    for eid, u_map, is_unfair in score_exports:
        lines = [
            json.dumps({
                "board": board_doc["board"],
                "board_sha256": board_doc["board_sha256"],
                "cid": c,
                "envelope_sha256": env[c],
                "model_id": eid,
                "utility": u_map[c],
                "status": "quarantined_unfair_diagnostic" if is_unfair else "quarantined_fair_all_non_d91_control",
                "quarantined_unfair_diagnostic": is_unfair,
            }, sort_keys=True)
            for c in cids
        ]
        (diag_board / "scores" / f"{eid}.jsonl").write_text("\n".join(lines) + "\n")

    out_doc = {
        "schema": "jope.h449.unfair_d91_readout/v2",
        "wave": WAVE,
        "quarantined_unfair_diagnostic": True,
        "canonical_promotion_forbidden": True,
        "lock_sha256": lock_sha,
        "columns_ok": {
            "fair_all_non_d91": non_ok,
            "unfair_all_d91": all_ok,
            "unfair_search_d91": srch_ok,
        },
        "fair_all_non_d91_stack": ro_non_stack,
        "fair_all_non_d91_icl_only": ro_non_icl,
        "fair_all_non_d91_stack_h425_clear": ro_non_stack_clear,
        "fair_all_non_d91_icl_only_h425_clear": ro_non_icl_clear,
        "unfair_all_d91_stack": ro_all,
        "unfair_all_d91_icl_only": ro_icl,
        "unfair_all_d91_stack_h425_clear": ro_all_clear,
        "unfair_all_d91_icl_only_h425_clear": ro_icl_clear,
        "matched_d91_inflation": matched_d91_inflation,
        "unfair_search_d91_stack": ro_srch,
    }
    diag_out.parent.mkdir(parents=True, exist_ok=True)
    with open(diag_out, "x") as f:
        f.write(json.dumps(out_doc, indent=1, sort_keys=True, default=_jd) + "\n")

    if ro_non_stack is not None and ro_non_icl is not None:
        en = ro_non_stack["elo"]
        eni = ro_non_icl["elo"]
        print(
            f"[FAIR ALL-NON-DEV91] {FAIR_NON_D91_STACK} (0% d91, all ~1,976 non-d91 complete tables): "
            f"Elo {en['elo']:.1f} [{en['ci95'][0]:.1f}, {en['ci95'][1]:.1f}], "
            f"rank {ro_non_stack['rank']}/{ro_non_stack['n_entrants']}; "
            f"vs {KUMO_LARGE}: {ro_non_stack['gap_elo']['vs_' + KUMO_LARGE]:+.1f} Elo; "
            f"vs {FAIR_ENTRANT}: {ro_non_stack['gap_elo']['vs_' + FAIR_ENTRANT]:+.1f} Elo"
        )
        print(
            f"[FAIR ALL-NON-DEV91] {FAIR_NON_D91_ICL_ONLY} (0% d91 pure neural R4-e8): "
            f"Elo {eni['elo']:.1f} [{eni['ci95'][0]:.1f}, {eni['ci95'][1]:.1f}], "
            f"rank {ro_non_icl['rank']}/{ro_non_icl['n_entrants']}; "
            f"vs {KUMO_LARGE}: {ro_non_icl['gap_elo']['vs_' + KUMO_LARGE]:+.1f} Elo"
        )

    e_all = ro_all["elo"]
    e_icl = ro_icl["elo"]
    print(
        f"[QUARANTINED UNFAIR] {UNFAIR_STACK} (ENTIRE + 100% d91): "
        f"Elo {e_all['elo']:.1f} [{e_all['ci95'][0]:.1f}, {e_all['ci95'][1]:.1f}], "
        f"rank {ro_all['rank']}/{ro_all['n_entrants']}; "
        f"vs {KUMO_LARGE}: {ro_all['gap_elo']['vs_' + KUMO_LARGE]:+.1f} Elo; "
        f"vs {FAIR_ENTRANT}: {ro_all['gap_elo']['vs_' + FAIR_ENTRANT]:+.1f} Elo"
    )
    print(
        f"[QUARANTINED UNFAIR] {UNFAIR_ICL_ONLY} (ENTIRE + 100% d91 pure neural R4-e8): "
        f"Elo {e_icl['elo']:.1f} [{e_icl['ci95'][0]:.1f}, {e_icl['ci95'][1]:.1f}], "
        f"rank {ro_icl['rank']}/{ro_icl['n_entrants']}; "
        f"vs {KUMO_LARGE}: {ro_icl['gap_elo']['vs_' + KUMO_LARGE]:+.1f} Elo"
    )
    if matched_d91_inflation is not None:
        inf_s = matched_d91_inflation["stack_all_d91_vs_all_non_d91_91"]
        inf_i = matched_d91_inflation["icl_only_all_d91_vs_all_non_d91_91"]
        print(
            f"[MATCHED +DEV91 DELTA] Stack: {inf_s['elo_diff_single_entrant']:+.1f} Elo "
            f"(gain {inf_s['pairs']['gain']:+.4f}, W/T/L {inf_s['pairs']['wtl']}); "
            f"Pure Neural ICL: {inf_i['elo_diff_single_entrant']:+.1f} Elo "
            f"(gain {inf_i['pairs']['gain']:+.4f}, W/T/L {inf_i['pairs']['wtl']})"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
