#!/usr/bin/env python3
"""H448 single-shot readout (`analyze_h448.py`) on `xbabe2`.

Verifies all PREREG §4 harness gates BEFORE computing or printing any `h448.APEX_FAIR_CHAMPION` utility.
If any gate fails, writes `status: "INCOMPLETE"` to `--out`, writes NO `--rows`, and exits `3`.
When all gates pass, computes:
  1. `readout.h425_without_feature_clear`: Strict H425 feature-clear holdout Elo vs `kumo_tabular_large_e8.v1`
  2. `readout.canonical_board_91`: Canonical 91-context / 19-family `dev91-audit-v1` board Elo vs Kumo
  3. Create-once `scores/h448.APEX_FAIR_CHAMPION.jsonl` (`ROWS`) and `H448_DEV91.json` (`OUT`).
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import sys
import traceback
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import guard_matern_core as GM  # noqa: E402
import meta_h416 as M  # noqa: E402
import train_h416 as T16  # noqa: E402

WAVE = "H448"
ENTRANT = "h448.APEX_FAIR_CHAMPION"
SXB_ENTRANT = "h416.SXB"
X0B = T16.X0B  # h335.H335_AGG
XSTACK = "h364.XSTACK_2L"
ANCHOR = T16.ANCHOR  # h335.H257_REPLICA = 1500.0
KUMO_LARGE = "kumo_tabular_large_e8.v1"
KUMO_SMALL = "kumo_tabular_small_e8.v1"
TABFM = "tabfm_ensemble.v1"
LADDER_SEED, N_BOOT = T16.LADDER_SEED, T16.N_BOOT  # 335160, 10000
SEEDS = (335101, 335102)
TAGS = {335101: "D1", 335102: "D2"}
SXB_KEY = "P3+LIB3.SXB"
DESIGN_NAME = "BCAR.mix"
H430_K3 = ("key_direct", "key_gbm", "factor_glm")
H433_F4 = ("form_gbm", "form_gam", "form_sr", "form_interp")
OK_ERRORS = ("no usable H program",)
TOL_REPLAY = 1e-9
TOL_LADDER = 1e-6
TOL_MEAN = 1e-12
N_CONTEXTS, N_FAMILIES, N_STORED = 91, 19, 32
MIN_COL_OK = 86
EXPOSED = ("ac07466e", "8eeca816", "016648dc", "6ecbb7d5", "b1a5c1a1")
H335_STATS_SHA256 = "ee3b45439a08bc395d53079570ed8965b61aad8c2d2e4e89fd1b32c490bbccb1"
HARNESS = "h448-fair-champion-v1"
BOARD_MANIFEST_NAME = "board_audit.sha256"
A2_SCHEMA = "jope.h418.a2-cells/v1"


class GateFail(Exception):
    """A harness input is missing or malformed: INCOMPLETE, no value read."""


def _sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _json(p) -> dict:
    try:
        return json.loads(Path(p).read_text())
    except (OSError, ValueError) as e:
        raise GateFail(f"{Path(p).name}: {type(e).__name__}")


def _result(p: Path) -> dict:
    if not p.is_file():
        raise GateFail(f"missing record {p.parent.name}/{p.name[:12]}")
    rs = [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
    rs = [r for r in rs if r.get("type") == "result"]
    if len(rs) != 1:
        raise GateFail(f"{p.parent.name}/{p.name[:12]}: {len(rs)} result records")
    return rs[0]


def _finite(v) -> bool:
    return isinstance(v, (int, float)) and math.isfinite(float(v))


def _ladder_order(values: dict) -> dict:
    return dict(sorted(values.items(), key=lambda kv: kv[0] + ".jsonl"))


def _jd(o):
    if hasattr(o, "item"):
        return o.item()
    raise TypeError(type(o).__name__)


def _stats():
    _, H = T16._stats_mod()
    if _sha(H.__file__) != H335_STATS_SHA256:
        raise SystemExit(f"h335_stats at {H.__file__} is not the pinned file")
    return H


def _a2_cells(p, cids) -> tuple[set, bool]:
    d = _json(p)
    cells = d.get("cells") if isinstance(d, dict) else None
    if not isinstance(d, dict) or d.get("schema") != A2_SCHEMA or not isinstance(cells, list):
        return set(), False
    keys = [(x.get("cid"), x.get("seed"), x.get("source")) for x in cells if isinstance(x, dict)]
    ok = (
        len(keys) == len(cells) == len(set(keys))
        and all(c in cids and s in SEEDS and n in ("GE", "GL") for c, s, n in keys)
    )
    return {(c, s) for c, s, _ in keys if c in cids}, ok


def load_board(bdir: Path, manifest: Path) -> dict:
    man = {}
    for line in manifest.read_text().splitlines():
        if line.strip():
            h, name = line.split(None, 1)
            man[name.strip()] = h
    have = sorted(str(p.relative_to(bdir)) for p in bdir.rglob("*") if p.is_file())
    files_ok = have == sorted(man) and all(_sha(bdir / n) == man[n] for n in have)
    board = _json(bdir / "BOARD.json")
    env = {c["cid"]: c["envelope_sha256"] for c in board["contexts"]}
    fam = {c["cid"]: c["family_id"] for c in board["contexts"]}
    values, x0ps, incomplete, bad = {}, {}, {}, []
    ids = [p.stem for p in sorted((bdir / "scores").glob("*.jsonl"))]
    for eid in ids:
        recs = [json.loads(x) for x in (bdir / "scores" / f"{eid}.jsonl").read_text().splitlines() if x.strip()]
        cids = [r["cid"] for r in recs]
        if len(set(cids)) != len(cids) or any(
            r["cid"] not in env
            or r.get("board", board["board"]) != board["board"]
            or ("envelope_sha256" in r and r["envelope_sha256"] != env[r["cid"]])
            for r in recs
        ):
            bad.append(eid)
            continue
        got = {r["cid"]: float(r["utility"]) for r in recs}
        if set(got) != set(env):
            incomplete[eid] = len(got)
            continue
        values[eid] = got
        if eid == X0B:
            x0ps = {r["cid"]: [float(x) for x in (r.get("per_seed") or [])] for r in recs}
    return {
        "board": board,
        "env": env,
        "fam": fam,
        "values": values,
        "x0_per_seed": x0ps,
        "incomplete": incomplete,
        "bad": bad,
        "ids": ids,
        "files_ok": files_ok,
        "n_files": len(have),
    }


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


def collect(a, H, checks: dict, gate: dict):
    P = Path(a.pulled)
    P418 = Path(a.h418_pulled)
    lk = _json(a.lock)
    lock_sha = _sha(a.lock)
    design = _json(a.design)
    frz = _json(a.frozen)

    rec_txt = Path(a.receipt).read_text().strip() if Path(a.receipt).is_file() else ""
    rec_toks = rec_txt.split()
    rec_sha = rec_toks[0] if rec_toks else ""

    bm = [k for k in (lk.get("inputs_sha256") or {}) if Path(k).name == BOARD_MANIFEST_NAME]
    gate["lock"] = (
        lk.get("schema") == "jope.h448.lock/v1"
        and lk.get("entrant") == ENTRANT
        and lk.get("seeds") == list(SEEDS)
        and lk.get("design") == DESIGN_NAME
        and lk.get("design_sha256") == _sha(a.design)
        and lk.get("frozen_champion_sha256") == _sha(a.frozen)
        and lk.get("a2_cells_sha256") == _sha(a.a2_cells)
        and len(bm) == 1
        and lk["inputs_sha256"][bm[0]] == _sha(a.board_manifest)
        and rec_sha == lock_sha
    )
    gate["design"] = design.get("schema") == "jope.h416.design/v1" and design.get("design") == DESIGN_NAME
    gate["frozen_champion"] = (
        frz.get("schema") == "jope.h447.frozen_fair_champion/v1"
        and frz.get("selected_arm") in ("STACK_R4_E8_WC", "STACK_R4_E8_BASE", "STACK_R4_M20", "GM_CW12")
    )
    checks["lock_sha256"] = lock_sha
    checks["selected_arm"] = frz.get("selected_arm")

    # H445 precondition (R1)
    h445_status = "UNKNOWN"
    following_null_h445 = False
    if a.h445_readout and Path(a.h445_readout).is_file():
        r445 = _json(a.h445_readout)
        verdict = str(r445.get("verdict") or "")
        if verdict == "JOINT_DEV_CONFIRMED":
            h445_status = "JOINT_DEV_CONFIRMED"
            gate["h445_precondition"] = True
        elif verdict == "NULL":
            h445_status = "FOLLOWING_NULL_H445_DEV"
            following_null_h445 = True
            gate["h445_precondition"] = True
        elif verdict == "HARM":
            h445_status = "HARM"
            gate["h445_precondition"] = "H445_OVERRIDE" in rec_toks
        else:
            h445_status = f"UNRECOGNIZED:{verdict}"
            gate["h445_precondition"] = False
    else:
        if "OWNER_DIRECT_D91" in rec_toks:
            h445_status = "OWNER_DIRECT_D91"
            gate["h445_precondition"] = True
        else:
            h445_status = "MISSING_H445_READOUT"
            gate["h445_precondition"] = False
    checks["h445_dev_status"] = h445_status
    checks["following_null_h445_dev"] = following_null_h445

    # (1) Board
    bd = load_board(Path(a.board), Path(a.board_manifest))
    env, fam, field = bd["env"], bd["fam"], bd["values"]
    lad = _json(Path(a.board) / "ladder.json")
    _, _, _, el0 = T16.elo_boot(np, H, field, fam, ANCHOR, LADDER_SEED, a.n_boot) if ANCHOR in field else (0, 0, 0, {})
    le = (lad.get("elo") or {}).get("elo") or {}
    dl = max(
        (
            max(
                (abs(el0[e]["elo"] - v["elo"]),)
                + (
                    (abs(el0[e]["ci95"][0] - v["ci95"][0]), abs(el0[e]["ci95"][1] - v["ci95"][1]))
                    if a.n_boot == N_BOOT
                    else ()
                )
            )
            for e, v in le.items()
            if e in el0
        ),
        default=math.inf,
    )
    h448_existing = [e for e in bd["ids"] if e.startswith("h448.")]
    checks["board"] = {
        "files": bd["n_files"],
        "files_match_manifest": bd["files_ok"],
        "complete_entrants": len(field),
        "incomplete": bd["incomplete"],
        "bad": bd["bad"],
        "h448_files": h448_existing,
        "ladder_reproduced_max_abs": float(dl),
    }
    gate["board_files"] = bd["files_ok"]
    b0 = bd["board"]
    gate["board_id"] = (
        b0.get("board") == "dev91-audit-v1"
        and b0.get("n_contexts") == N_CONTEXTS == len(env)
        and b0.get("n_families") == N_FAMILIES == len(set(fam.values()))
        and isinstance(b0.get("board_sha256"), str)
        and len(b0["board_sha256"]) == 64
    )
    gate["board_ladder"] = dl <= TOL_LADDER and set(le) == set(field)
    gate["board_entrants"] = (
        len(field) == N_STORED
        and not bd["incomplete"]
        and not bd["bad"]
        and not h448_existing
        and all(e in field for e in (X0B, XSTACK, SXB_ENTRANT, ANCHOR, KUMO_LARGE, KUMO_SMALL, TABFM))
    )

    # (2) Contexts & X0 per-seed
    sel = _json(P / "cids91.json")
    cids = sorted(sel.get("cids") or [])
    gate["contexts"] = (
        len(cids) == N_CONTEXTS
        and cids == sorted(env)
        and len(set(fam.values())) == N_FAMILIES
        and all((sel.get("family_of") or {}).get(c) == fam[c] for c in cids)
        and sel.get("lock_sha256") == lock_sha
    )
    x0ps = bd["x0_per_seed"]
    gate["x0_board"] = (
        bool(cids)
        and all(len(x0ps.get(c, [])) == 2 and abs(sum(x0ps[c]) / 2 - field[X0B][c]) <= TOL_MEAN for c in cids)
        if X0B in field
        else False
    )
    a2_pairs, gate["a2_cells_file"] = _a2_cells(a.a2_cells, cids)
    if not all(gate.values()):
        return None
    x0 = {s: {c: x0ps[c][i] for c in cids} for i, s in enumerate(SEEDS)}

    # (3) H418 pulled records & SXB exact replay gate
    recs16 = {
        s: {c: _result(P418 / TAGS[s] / "stages" / f"H416_{TAGS[s]}" / f"{c}__confirm.jsonl") for c in cids}
        for s in SEEDS
    }
    npz_sha_ok = True
    no_h = {s: [] for s in SEEDS}
    sxb_re = {s: {} for s in SEEDS}
    for s in SEEDS:
        for c in cids:
            r = recs16[s][c]
            if r.get("error"):
                if r["error"] in OK_ERRORS:
                    no_h[s].append(c)
                else:
                    npz_sha_ok = False
                sxb_re[s][c] = x0[s][c]
                continue
            npz_p = P418 / TAGS[s] / f"h416_{TAGS[s]}_npz" / f"{c}.npz"
            if not npz_p.is_file() or _sha(npz_p) != (r.get("npz") or {}).get("sha256"):
                npz_sha_ok = False
                continue
            if (c, s) in a2_pairs:
                sxb_re[s][c] = x0[s][c]
            else:
                sxb_re[s][c] = float(r["outputs"][SXB_KEY]["value"])
    gate["h418_pulled_shas"] = npz_sha_ok
    if not npz_sha_ok:
        return None
    max_sxb_delta = max(abs(0.5 * (sxb_re[SEEDS[0]][c] + sxb_re[SEEDS[1]][c]) - field[SXB_ENTRANT][c]) for c in cids)
    checks["h418_sxb_replay_max_abs_delta"] = float(max_sxb_delta)
    gate["h418_sxb_replay"] = max_sxb_delta <= TOL_REPLAY

    # (4) bank_d91_search.npz alignment against D1 & D2
    bank_path = P / "bank_d91_search.npz"
    bank_meta = _json(P / "bank_d91_search.npz.json")
    bank_rows = {}
    bank_ok = (
        bank_path.is_file()
        and bank_meta.get("audit_targets_written") is False
        and bank_meta.get("sha256") == _sha(bank_path)
        and bank_meta.get("lock_sha256") == lock_sha
    )
    if bank_ok:
        with np.load(bank_path, allow_pickle=False) as zb:
            b_cids = [str(x) for x in zb["cids"].tolist()]
            if sorted(b_cids) != cids:
                bank_ok = False
            else:
                search_arr, audit_arr = zb["search"], zb["audit"]
                y64_arr, lab_arr = zb["y64"], zb["lab"]
                ss, sa = zb["s_shape"], zb["a_shape"]
                ya_shas = [str(x) for x in zb["ya_sha256"].tolist()]
                so = ao = lo = 0
                for i, c in enumerate(b_cids):
                    n, w = int(ss[i, 0]), int(ss[i, 1])
                    na, f = int(sa[i, 0]), int(sa[i, 1])
                    S_mat = np.array(search_arr[so : so + n * w], np.float64).reshape(n, w)
                    Xa = np.array(audit_arr[ao : ao + na * f], np.float64, copy=True).reshape(na, f)
                    Xs = np.array(S_mat[:, :-1], np.float64, copy=True)
                    y = np.array(y64_arr[lo : lo + n], np.float64, copy=True)
                    lab = np.array(lab_arr[lo : lo + n], np.int64, copy=True)
                    bank_rows[c] = (Xs, Xa, y, lab, ya_shas[i])
                    so += n * w
                    ao += na * f
                    lo += n
                for s in SEEDS:
                    for c in cids:
                        if c in no_h[s]:
                            continue
                        z16 = np.load(P418 / TAGS[s] / f"h416_{TAGS[s]}_npz" / f"{c}.npz", allow_pickle=False)
                        Xs, Xa, y, lab, ya_sha = bank_rows[c]
                        ya_got = hashlib.sha256(np.ascontiguousarray(z16["ya"], np.float64).tobytes()).hexdigest()
                        if not (
                            np.array_equal(y, z16["y"].astype(np.float64))
                            and np.array_equal(lab, z16["lab"].astype(np.int64))
                            and ya_sha == ya_got
                            and Xs.shape[0] == len(y)
                            and Xa.shape[0] == len(z16["ya"])
                        ):
                            bank_ok = False
    gate["bank_d91_alignment"] = bank_ok

    # (5) H425 X_search contamination screen gate
    scr_doc = _json(P / "SCREEN_D91_SEARCH.json")
    scr_per = scr_doc.get("per_context") or {}
    icl_spec = (frz.get("blocks") or {}).get("icl") or {}
    gate["screen_d91"] = (
        scr_doc.get("schema") == "jope.h448.screen_d91_search/v1"
        and scr_doc.get("confirm_targets_keyed") is False
        and scr_doc.get("audit_rows_read") == 0
        and scr_doc.get("lock_sha256") == lock_sha
        and sorted(scr_per) == cids
        and (not icl_spec.get("enabled") or scr_doc.get("training_bank_sha256") == icl_spec.get("training_bank_sha256"))
    )
    checks["screen_d91_summary"] = scr_doc.get("summary")

    # (6) External column completeness gate
    blocks_by_cid = {}
    m_ok = k_ok = f_ok = icl_ok = 0
    icl_enabled = bool(icl_spec.get("enabled"))
    icl_col_name = str(icl_spec.get("col_name") or "r4_e8")
    if bank_ok:
        for c in cids:
            Xs, Xa, y, _lab, _ya = bank_rows[c]
            n_s, n_a = len(y), len(Xa)
            blk_m = _load_col_block(P / "cols_matern_D91" / f"{c}.npz", GM.PRIMARY_NAMES, n_s, n_a)
            blk_k_raw = _load_col_block(P / "cols_k3_D91" / f"{c}.npz", H430_K3, n_s, n_a)
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
            blk_f_raw = _load_col_block(P / "cols_f4_D91" / f"{c}.npz", H433_F4, n_s, n_a)
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
            blk_icl = (
                _load_col_block(P / "cols_icl_D91" / f"{c}.npz", (icl_col_name,), n_s, n_a)
                if icl_enabled
                else None
            )
            m_ok += blk_m is not None
            k_ok += blk_k is not None
            f_ok += blk_f is not None
            icl_ok += blk_icl is not None
            blocks_by_cid[c] = (blk_m, blk_k, blk_f, blk_icl)
    checks["columns"] = {
        "matern_ok": m_ok,
        "k3_ok": k_ok,
        "f4_ok": f_ok,
        "icl_enabled": icl_enabled,
        "icl_col_name": icl_col_name if icl_enabled else None,
        "icl_ok": icl_ok,
    }
    gate["columns_complete"] = (
        m_ok >= MIN_COL_OK
        and k_ok >= MIN_COL_OK
        and f_ok >= MIN_COL_OK
        and (not icl_enabled or icl_ok >= MIN_COL_OK)
    )
    if not all(gate.values()):
        return None

    return {
        "cids": cids,
        "env": env,
        "fam": fam,
        "field": field,
        "x0": x0,
        "recs16": recs16,
        "no_h": no_h,
        "a2_pairs": a2_pairs,
        "bank_rows": bank_rows,
        "blocks_by_cid": blocks_by_cid,
        "screen_per": scr_per,
        "screen_summary": scr_doc["summary"],
        "board": bd["board"],
        "lad": le,
        "lock": lk,
        "lock_sha": lock_sha,
        "frozen": frz,
        "h445_dev_status": h445_status,
        "following_null_h445_dev": following_null_h445,
        "P418": P418,
    }


def _eval_seed_ctx(job):
    s, c, npz_path, Xs, Xa, y, lab, blk_m, blk_k, blk_f, blk_icl, icl_enabled = job
    z16 = np.load(npz_path, allow_pickle=False)
    Q_gp = z16["Q"].astype(np.float64)
    ya = z16["ya"].astype(np.float64)
    ba = float(z16["ba"])
    ym = float(z16["ym"])
    m_gp, n_a = Q_gp.shape[1], len(ya)
    mean_a = np.full(n_a, ym, dtype=np.float64)
    bag_gp = [
        None if not np.all(np.isfinite(z16["AKF"][:, i])) else z16["AKF"][:, i].astype(np.float64)
        for i in range(m_gp)
    ]
    ref_gp = [
        z16["PA"][:, i].astype(np.float64) if bool(z16["pa_ok"][i]) else None
        for i in range(m_gp)
    ]
    bcols_gp = [b if b is not None else (r if r is not None else mean_a) for b, r in zip(bag_gp, ref_gp)]

    active_champ = [b for b in ([blk_m, blk_k, blk_f] + ([blk_icl] if icl_enabled else [])) if b is not None]
    pred_champ, _ = GM.guard_and_stack_2l(c, Xs, Xa, y, lab, Q_gp, bcols_gp, ref_gp, active_champ)
    u_ch = float(M.utility(pred_champ, ya, ba)[0])

    pred_g, _ = GM.guard_and_stack_2l(c, Xs, Xa, y, lab, Q_gp, bcols_gp, ref_gp, [])
    u_go = float(M.utility(pred_g, ya, ba)[0])

    active_cw12 = [b for b in (blk_m, blk_k, blk_f) if b is not None]
    pred_cw12, _ = GM.guard_and_stack_2l(c, Xs, Xa, y, lab, Q_gp, bcols_gp, ref_gp, active_cw12)
    u_cw = float(M.utility(pred_cw12, ya, ba)[0])
    return s, c, u_ch, u_go, u_cw


def _slice_readout(
    H,
    field: dict,
    u_ent: dict,
    fam: dict,
    keep_cids: list[str],
    n_boot: int = N_BOOT,
    stored_lad: dict | None = None,
    with_pairs: bool = True,
) -> dict:
    fk = {c: fam[c] for c in keep_cids}
    n_fams = len(set(fk.values()))
    if len(keep_cids) < 2 or n_fams < 2:
        return {"contexts": len(keep_cids), "families": n_fams, "insufficient_families": True}
    sub_field = _ladder_order({e: {c: v[c] for c in keep_cids} for e, v in {**field, ENTRANT: u_ent}.items()})
    arms, point, boot, elo = T16.elo_boot(np, H, sub_field, fk, ANCHOR, LADDER_SEED, n_boot)
    order = sorted(elo, key=lambda e: -elo[e]["elo"])
    focus = [e for e in (KUMO_LARGE, TABFM, KUMO_SMALL, SXB_ENTRANT, XSTACK, X0B) if e in sub_field]
    res = {
        "contexts": len(keep_cids),
        "families": n_fams,
        "rank": order.index(ENTRANT) + 1,
        "n_entrants": len(order),
        "entrant_elo": elo[ENTRANT],
        "top_10_order": [{"entrant": e, "rank": i + 1, "elo": elo[e]["elo"], "ci95": elo[e]["ci95"]} for i, e in enumerate(order[:10])],
        "board_elo": elo,
        "order": order,
        "family_mean_u": {e: H.family_mean(sub_field[e], fk) for e in (ENTRANT, *focus)},
        "gap_elo": {f"vs_{e}": elo[ENTRANT]["elo"] - elo[e]["elo"] for e in focus},
        "delta_elo": {e: T16.delta_elo(np, arms, point, boot, ENTRANT, e) for e in focus},
    }
    if with_pairs:
        res["pairs"] = {e: T16.pair_stats(np, H, sub_field[ENTRANT], sub_field[e], fk) for e in focus}
    if stored_lad is not None:
        ep = H.elo_panel(sub_field, fk, ANCHOR, seed=LADDER_SEED, n_boot=1)
        res["elo_panel_vs_boot_max_abs"] = float(max(abs(ep["elo"][e]["elo"] - elo[e]["elo"]) for e in elo))
        res["stored_entrants_elo_shift_max_abs"] = float(max(abs(elo[e]["elo"] - stored_lad[e]["elo"]) for e in field))
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k in ("pulled", "h418-pulled", "board", "board-manifest", "lock", "receipt", "frozen", "design", "a2-cells", "out", "rows"):
        ap.add_argument(f"--{k}", required=True)
    ap.add_argument("--h445-readout", default="")
    ap.add_argument("--procs", type=int, default=16)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--desc-boot", type=int, default=N_BOOT)
    a = ap.parse_args(argv)
    out, rows = Path(a.out), Path(a.rows)
    if out.exists() or rows.exists():
        raise SystemExit(f"{out} or {rows} exists (create-once)")
    H = _stats()

    checks, gate = {}, {}
    try:
        k = collect(a, H, checks, gate)
    except (GateFail, KeyError, TypeError, ValueError, IndexError, AttributeError) as e:
        gate["harness_inputs"] = False
        if isinstance(e, GateFail):
            checks["harness_inputs_error"] = str(e)
        else:
            tb = traceback.extract_tb(e.__traceback__)[-1]
            checks["harness_inputs_error"] = f"{type(e).__name__} at {Path(tb.filename).name}:{tb.lineno}"
        k = None

    base = {
        "schema": "jope.h448.d91-analysis/v1",
        "wave": WAVE,
        "entrant": ENTRANT,
        "board": "dev91-audit-v1",
        "seeds": list(SEEDS),
        "tolerances": {"replay": TOL_REPLAY, "ladder": TOL_LADDER},
        "gate": gate,
        "checks": checks,
    }
    failed = sorted(g for g, v in gate.items() if not v)
    if k is None or failed:
        base.update({"status": "INCOMPLETE", "gate_failed": failed})
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "x") as f:
            f.write(json.dumps(base, indent=1, sort_keys=True, default=_jd) + "\n")
        print(f"INCOMPLETE: harness gate failed {failed}; no entrant value was read")
        print(f"checks: {json.dumps(checks, sort_keys=True, default=_jd)}")
        return 3

    cids, x0, P418 = k["cids"], k["x0"], k["P418"]
    icl_enabled = bool((k["frozen"]["blocks"]["icl"] or {}).get("enabled"))
    u_seed = {s: {} for s in SEEDS}
    u_g_only_seed = {s: {} for s in SEEDS}
    u_gm_cw12_seed = {s: {} for s in SEEDS}
    why = {s: collections.Counter() for s in SEEDS}

    jobs = []
    for s in SEEDS:
        for c in cids:
            if c in k["no_h"][s]:
                u_seed[s][c] = u_g_only_seed[s][c] = u_gm_cw12_seed[s][c] = x0[s][c]
                why[s]["no_usable_H_x0b"] += 1
                continue
            Xs, Xa, y, lab, _ = k["bank_rows"][c]
            blk_m, blk_k, blk_f, blk_icl = k["blocks_by_cid"][c]
            jobs.append((
                s, c, P418 / TAGS[s] / f"h416_{TAGS[s]}_npz" / f"{c}.npz",
                Xs, Xa, y, lab, blk_m, blk_k, blk_f, blk_icl, icl_enabled,
            ))

    if a.procs > 1 and len(jobs) > 1:
        with Pool(processes=min(a.procs, len(jobs))) as pool:
            done_jobs = pool.map(_eval_seed_ctx, jobs)
    else:
        done_jobs = [_eval_seed_ctx(j) for j in jobs]
    for s, c, u_ch, u_go, u_cw in done_jobs:
        u_seed[s][c] = u_ch
        u_g_only_seed[s][c] = u_go
        u_gm_cw12_seed[s][c] = u_cw
        why[s]["guarded_stack"] += 1

    u = {c: 0.5 * (u_seed[SEEDS[0]][c] + u_seed[SEEDS[1]][c]) for c in cids}
    u_g_only = {c: 0.5 * (u_g_only_seed[SEEDS[0]][c] + u_g_only_seed[SEEDS[1]][c]) for c in cids}
    u_gm_cw12 = {c: 0.5 * (u_gm_cw12_seed[SEEDS[0]][c] + u_gm_cw12_seed[SEEDS[1]][c]) for c in cids}
    if not all(math.isfinite(v) for v in u.values()):
        raise SystemExit("non-finite board value after gates: nothing written")

    clear_cids = [c for c in cids if not k["screen_per"][c]["flag_feature"]]
    nocopy_cids = [c for c in cids if k["screen_per"][c]["tier"] != "copy"]
    unexposed_cids = [c for c in cids if c[:8] not in EXPOSED]

    ro_clear = _slice_readout(H, k["field"], u, k["fam"], clear_cids, n_boot=a.n_boot)
    ro_board = _slice_readout(H, k["field"], u, k["fam"], cids, n_boot=a.n_boot, stored_lad=k["lad"])

    # Descriptive sensitivities & block ablations
    desc = {
        "without_copy_tier": _slice_readout(H, k["field"], u, k["fam"], nocopy_cids, n_boot=a.desc_boot, with_pairs=False),
        "without_exposed_contexts": _slice_readout(H, k["field"], u, k["fam"], unexposed_cids, n_boot=a.desc_boot, with_pairs=False),
        "seed_335101_only": _slice_readout(H, k["field"], u_seed[335101], k["fam"], cids, n_boot=a.desc_boot, with_pairs=False),
        "seed_335102_only": _slice_readout(H, k["field"], u_seed[335102], k["fam"], cids, n_boot=a.desc_boot, with_pairs=False),
        "ablation_G_ONLY_board91": {
            "family_mean_u": H.family_mean(u_g_only, k["fam"]),
            "elo": _slice_readout(H, k["field"], u_g_only, k["fam"], cids, n_boot=a.desc_boot, with_pairs=False)["entrant_elo"],
        },
        "ablation_GM_CW12_board91": {
            "family_mean_u": H.family_mean(u_gm_cw12, k["fam"]),
            "elo": _slice_readout(H, k["field"], u_gm_cw12, k["fam"], cids, n_boot=a.desc_boot, with_pairs=False)["entrant_elo"],
        },
    }
    if k["a2_pairs"]:
        u_a2 = {
            c: 0.5 * sum(x0[s][c] if (c, s) in k["a2_pairs"] else u_seed[s][c] for s in SEEDS)
            for c in cids
        }
        desc["a2_exempt_sensitivity"] = _slice_readout(H, k["field"], u_a2, k["fam"], cids, n_boot=a.desc_boot, with_pairs=False)["entrant_elo"]

    lk = k["lock"]
    src = {
        "wave": WAVE,
        "frozen_wave": "H447",
        "selected_arm": k["frozen"]["selected_arm"],
        "frozen_champion_sha256": lk["frozen_champion_sha256"],
        "lock_sha256": k["lock_sha"],
        "prereg_commit": lk["prereg_commit"],
        "seeds": list(SEEDS),
        "gp_pools": "H418 pulled D1 (335101) and D2 (335102) 64-program SEARCH-only pools",
        "combiner": "guard_matern_core.guard_and_stack_2l (M7 + K3 + F4 + frozen ICL)",
        "h445_dev_status": k["h445_dev_status"],
        "following_null_h445_dev": k["following_null_h445_dev"],
    }
    b, env = k["board"], k["env"]
    text = "".join(
        json.dumps(
            {
                "board": b["board"],
                "board_sha256": b["board_sha256"],
                "cid": c,
                "envelope_sha256": env[c],
                "model_id": ENTRANT,
                "utility": u[c],
                "per_seed": [u_seed[s][c] for s in SEEDS],
                "status": "completed",
                "harness_version": HARNESS,
                "source": src,
            },
            sort_keys=True,
        )
        + "\n"
        for c in cids
    )
    readout_doc = {
        "h425_without_feature_clear": ro_clear,
        "canonical_board_91": ro_board,
        "descriptive": desc,
    }
    base.update({
        "status": "SCORED",
        "selected_arm": k["frozen"]["selected_arm"],
        "h445_dev_status": k["h445_dev_status"],
        "following_null_h445_dev": k["following_null_h445_dev"],
        "n_contexts": len(cids),
        "n_families": len(set(k["fam"].values())),
        "value_source": {str(s): dict(why[s]) for s in SEEDS},
        "rows": {"path": str(rows), "sha256": hashlib.sha256(text.encode()).hexdigest(), "n": len(cids)},
        "readout": readout_doc,
    })
    report = json.dumps(base, indent=1, sort_keys=True, default=_jd) + "\n"
    rows.parent.mkdir(parents=True, exist_ok=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(rows, "x") as f:
        f.write(text)
    with open(out, "x") as f:
        f.write(report)
    if _sha(rows) != base["rows"]["sha256"]:
        raise SystemExit(f"{rows} bytes differ from computed rows")

    eb = ro_board["entrant_elo"]
    print(
        f"SCORED {ENTRANT} ({k['frozen']['selected_arm']}) on dev91-audit-v1 [ALL 91 / 19 fam]: "
        f"AUDIT-Elo {eb['elo']:.1f} [{eb['ci95'][0]:.1f}, {eb['ci95'][1]:.1f}], "
        f"rank {ro_board['rank']} of {ro_board['n_entrants']}; family-mean u {ro_board['family_mean_u'][ENTRANT]:.4f}"
    )
    if not ro_clear.get("insufficient_families"):
        ec = ro_clear["entrant_elo"]
        print(
            f"HEADLINE FAIR CLEAR SLICE [H425 without_feature: {ro_clear['contexts']} ctx / {ro_clear['families']} fam]: "
            f"AUDIT-Elo {ec['elo']:.1f} [{ec['ci95'][0]:.1f}, {ec['ci95'][1]:.1f}], "
            f"rank {ro_clear['rank']} of {ro_clear['n_entrants']}; family-mean u {ro_clear['family_mean_u'][ENTRANT]:.4f}; "
            f"gap vs {KUMO_LARGE}: {ro_clear['gap_elo']['vs_' + KUMO_LARGE]:+.1f} Elo"
        )
    for x in (KUMO_LARGE, TABFM, KUMO_SMALL, SXB_ENTRANT, XSTACK, X0B):
        d, p = ro_board["delta_elo"][x], ro_board["pairs"][x]
        print(
            f"vs {x:28s} Elo gap {ro_board['gap_elo']['vs_' + x]:+7.1f}; "
            f"dElo CI [{d['ci95'][0]:+.1f}, {d['ci95'][1]:+.1f}]; "
            f"families W/T/L {'/'.join(map(str, p['wtl']))}; gain {p['gain']:+.4f} ({100 * p['pct_rmse']:+.1f}% RMSE); "
            f"clipped {p['clipped_mean']:+.4f} p {p['signflip_p']:.3g} LOFO {p['lofo_min']:+.4f}"
        )
    print(f"screen summary: {json.dumps(k['screen_summary'])}")
    print(f"ablation G_ONLY Elo {desc['ablation_G_ONLY_board91']['elo']['elo']:.1f}; "
          f"GM_CW12 Elo {desc['ablation_GM_CW12_board91']['elo']['elo']:.1f}")
    print(f"rows {rows} sha256 {base['rows']['sha256'][:16]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
