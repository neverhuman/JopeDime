#!/usr/bin/env python3
"""H447 stack apply, statistical readout, and create-once `FROZEN_FAIR_CHAMPION.json` selector (`eval_h447.py`).

Commands:
  eval_h447.py selftest
  eval_h447.py apply   --run-dir DIR --tag TBOTH|P201B --panel FIT-SEL|P201B --bank FILE
                       --matern-cols DIR --h430-cols DIR --h433-cols DIR
                       --r4e8-base-cols DIR [--r4e8-wc-cols DIR] --out FILE [--procs 8]
  eval_h447.py readout --apply-fitsel FILE --apply-p201b FILE --excl-fitsel FILE --probed-p201b FILE
                       --board-p201b-dir DIR --gates FILE --pins-json FILE --out FILE --frozen-out FILE
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
H433_F4 = ("form_gbm", "form_gam", "form_sr", "form_interp")
ANCHOR_P201B = "h335.H257_REPLICA"
LADDER_SEED, N_BOOT = 335_160, 10_000


def _u(pred, ya, ba) -> float:
    return float(M.utility(pred, ya, ba)[0])


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _load_col_block(p: Path, names: tuple[str, ...], n_s: int, n_a: int) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    if not p.exists():
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


def one_context(job) -> dict:
    run_dir, tag, matern_dir, h430_dir, h433_dir, base_icl_dir, wc_icl_dir, cid, bank_row = job
    t0 = time.process_time()
    rec, a = CA.stored(run_dir, tag, cid)
    st = rec["outputs"][SXB_KEY]
    if st.get("invalid") or st.get("spec", {}).get("w") is None:
        raise SystemExit(f"{cid}: missing stored baseline output")

    w0 = CA.bcar(a["Q"], a["y"], cid)
    ws = np.zeros(len(w0))
    for s, x in zip(st["spec"]["members"], st["spec"]["w"]):
        ws[a["shas"].index(s)] = x
    b_val = float(st["value"])
    u_re = _u(0.5 * a["BAG"] @ w0 + 0.5 * a["REF"] @ w0, a["ya"], a["ba"])

    Xs, Xa, y64, lab, ya_sha = bank_row
    n_s, n_a = int(len(a["y"])), int(len(a["ya"]))
    width_f = int(Xs.shape[1])
    aligned = bool(
        np.array_equal(y64, a["y"])
        and np.array_equal(lab.astype(np.int64), a["lab"].astype(np.int64))
        and hashlib.sha256(np.ascontiguousarray(a["ya"], np.float64).tobytes()).hexdigest() == ya_sha
        and Xs.shape[0] == n_s
        and Xa.shape[0] == n_a
    )
    if not aligned:
        raise SystemExit(f"{cid}: bank alignment check failed")

    raw_npz = np.load(Path(run_dir) / f"h416_{tag}_npz" / f"{cid}.npz", allow_pickle=False)
    m_gp = a["Q"].shape[1]
    mean_a = np.full(n_a, a["ym"], dtype=np.float64)
    bag_gp = [
        None if not np.all(np.isfinite(raw_npz["AKF"][:, i])) else raw_npz["AKF"][:, i].astype(np.float64)
        for i in range(m_gp)
    ]
    ref_gp = [
        raw_npz["PA"][:, i].astype(np.float64) if bool(raw_npz["pa_ok"][i]) else None
        for i in range(m_gp)
    ]
    bcols_gp = [b if b is not None else (r if r is not None else mean_a) for b, r in zip(bag_gp, ref_gp)]

    blk_m = _load_col_block(Path(matern_dir) / f"{cid}.npz", GM.PRIMARY_NAMES, n_s, n_a)

    blk_k_raw = _load_col_block(Path(h430_dir) / f"{cid}.npz", H430_K3, n_s, n_a)
    if blk_k_raw is not None:
        Qk, Bk, Rk = [arr.copy() for arr in blk_k_raw]
        y_lo, y_hi = float(np.min(a["y"])), float(np.max(a["y"]))
        y_pad = 0.25 * max(y_hi - y_lo, float(np.std(a["y"])), 1e-9)
        Qk[:, 0] = np.clip(Qk[:, 0], y_lo - y_pad, y_hi + y_pad)
        Bk[:, 0] = np.clip(Bk[:, 0], y_lo - y_pad, y_hi + y_pad)
        Rk[:, 0] = np.clip(Rk[:, 0], y_lo - y_pad, y_hi + y_pad)
        blk_k = (Qk, Bk, Rk)
    else:
        blk_k = None

    blk_f_raw = _load_col_block(Path(h433_dir) / f"{cid}.npz", H433_F4, n_s, n_a)
    if blk_f_raw is not None:
        Qf, Bf, Rf = [arr.copy() for arr in blk_f_raw]
        y_lo, y_hi = float(np.min(a["y"])), float(np.max(a["y"]))
        y_pad = 0.50 * max(y_hi - y_lo, float(np.std(a["y"])), 1e-9)
        for idx_f in (2, 3):
            Qf[:, idx_f] = np.clip(Qf[:, idx_f], y_lo - y_pad, y_hi + y_pad)
            Bf[:, idx_f] = np.clip(Bf[:, idx_f], y_lo - y_pad, y_hi + y_pad)
            Rf[:, idx_f] = np.clip(Rf[:, idx_f], y_lo - y_pad, y_hi + y_pad)
        blk_f = (Qf, Bf, Rf)
    else:
        blk_f = None

    p_base_icl = Path(base_icl_dir) / f"{cid}.npz"
    blk_m20 = _load_col_block(p_base_icl, ("cw5_m20",), n_s, n_a)
    blk_l = _load_col_block(p_base_icl, ("r4_l",), n_s, n_a)
    blk_m_icl = _load_col_block(p_base_icl, ("r4_m",), n_s, n_a)
    blk_e8_base = _load_col_block(p_base_icl, ("r4_e8",), n_s, n_a)
    blk_e8_2col = _load_col_block(p_base_icl, ("r4_l", "r4_m"), n_s, n_a)
    blk_e8_wc = _load_col_block(Path(wc_icl_dir) / f"{cid}.npz", ("r4_e8",), n_s, n_a) if wc_icl_dir else None

    def _run_guarded(blocks: list[tuple[np.ndarray, np.ndarray, np.ndarray] | None]) -> tuple[float, dict]:
        active = [b for b in blocks if b is not None]
        pred, meta = GM.guard_and_stack_2l(cid, Xs, Xa, a["y"], a["lab"], a["Q"], bcols_gp, ref_gp, active)
        return _u(pred, a["ya"], a["ba"]), meta

    u_g_only, _ = _run_guarded([])
    u_gm_cw12, _ = _run_guarded([blk_m, blk_k, blk_f])
    u_m20, _ = _run_guarded([blk_m, blk_k, blk_f, blk_m20])
    u_r4_l, _ = _run_guarded([blk_m, blk_k, blk_f, blk_l])
    u_r4_m, _ = _run_guarded([blk_m, blk_k, blk_f, blk_m_icl])
    u_e8_base, meta_e8 = _run_guarded([blk_m, blk_k, blk_f, blk_e8_base])
    u_e8_2col, _ = _run_guarded([blk_m, blk_k, blk_f, blk_e8_2col])

    rec_out = {
        "cid": cid,
        "family": rec["family_id"],
        "width_f": width_f,
        "n_search": n_s,
        "n_audit": n_a,
        "aligned": aligned,
        "w_diff": float(np.abs(ws - w0).max()),
        "u_diff": abs(u_re - b_val),
        "matern_ok": bool(blk_m is not None),
        "h430_ok": bool(blk_k is not None),
        "h433_ok": bool(blk_f is not None),
        "m20_ok": bool(blk_m20 is not None),
        "r4_e8_base_ok": bool(blk_e8_base is not None),
        "r4_e8_wc_ok": bool(blk_e8_wc is not None) if wc_icl_dir else False,
        "gp_dropped": meta_e8["gp_dropped"],
        "ext_admitted": meta_e8["ext_admitted"],
        "ext_weight_share": meta_e8["ext_weight_share"],
        "BASE_SXB": b_val,
        "G_ONLY": u_g_only,
        "GM_CW12": u_gm_cw12,
        "STACK_R4_M20": u_m20,
        "STACK_R4_L": u_r4_l,
        "STACK_R4_M": u_r4_m,
        "STACK_R4_E8_BASE": u_e8_base,
        "STACK_R4_E8_2COL": u_e8_2col,
        "cpu_s": time.process_time() - t0,
    }
    if wc_icl_dir:
        u_e8_wc, _ = _run_guarded([blk_m, blk_k, blk_f, blk_e8_wc])
        rec_out["STACK_R4_E8_WC"] = u_e8_wc
    return rec_out


def cmd_apply(a) -> None:
    out_p = Path(a.out)
    if out_p.exists():
        raise SystemExit(f"{out_p} exists (create-once)")

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
            S = np.array(search[so : so + n * w], np.float64).reshape(n, w)
            XA = np.array(audit[ao : ao + na * f], np.float64, copy=True).reshape(na, f)
            bank[str(c)] = (
                np.array(S[:, :-1], np.float64, copy=True),
                XA,
                np.array(y64[lo : lo + n], np.float64, copy=True),
                np.array(labs[lo : lo + n], np.int64, copy=True),
                str(ya_shas[i]),
            )
            so += n * w
            ao += na * f
            lo += n

    jobs = [
        (a.run_dir, a.tag, a.matern_cols, a.h430_cols, a.h433_cols, a.r4e8_base_cols, a.r4e8_wc_cols, c, bank[c])
        for c in cids
    ]
    with Pool(a.procs) as pool:
        res = pool.map(one_context, jobs, chunksize=1)

    out = {
        "schema": "jope.h447.apply/v1",
        "panel": a.panel,
        "tag": a.tag,
        "n": len(res),
        "has_wc": bool(a.r4e8_wc_cols),
        "bank_sha256": _sha(Path(a.bank)),
        "aligned_all": bool(all(r["aligned"] for r in res)),
        "matern_ok": int(sum(r["matern_ok"] for r in res)),
        "h430_ok": int(sum(r["h430_ok"] for r in res)),
        "h433_ok": int(sum(r["h433_ok"] for r in res)),
        "m20_ok": int(sum(r["m20_ok"] for r in res)),
        "r4_e8_base_ok": int(sum(r["r4_e8_base_ok"] for r in res)),
        "r4_e8_wc_ok": int(sum(r["r4_e8_wc_ok"] for r in res)),
        "replay_max_w_diff": float(max(r["w_diff"] for r in res)),
        "replay_max_u_diff": float(max(r["u_diff"] for r in res)),
        "cpu_s": float(sum(r["cpu_s"] for r in res)),
        "per_context": {r["cid"]: r for r in res},
    }
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "x") as f:
        json.dump(out, f, sort_keys=True)
    out_p.chmod(0o444)
    print(json.dumps({
        k: out[k]
        for k in (
            "panel", "n", "has_wc", "aligned_all", "matern_ok", "h430_ok", "h433_ok",
            "m20_ok", "r4_e8_base_ok", "r4_e8_wc_ok", "replay_max_w_diff", "replay_max_u_diff", "cpu_s",
        )
    }), flush=True)


def select_frozen_champion(
    has_wc: bool,
    fitsel_clean_vs_gm: dict[str, float],
    p201b_excl_vs_gm: dict[str, float],
    p201b_wc_vs_base: float | None,
    p201b_base_vs_m20: float,
) -> tuple[str, str]:
    """Deterministic PREREG §2 promotion rule -> `(selected_arm, reason)`."""
    if has_wc and p201b_wc_vs_base is not None:
        if (
            fitsel_clean_vs_gm["STACK_R4_E8_WC"] >= -0.002
            and p201b_excl_vs_gm["STACK_R4_E8_WC"] >= 0.0
            and p201b_wc_vs_base > 0.0
            and (p201b_wc_vs_base + p201b_base_vs_m20) >= 0.0
        ):
            return "STACK_R4_E8_WC", "WC beat BASE and GM_CW12 on P201b excl10 with FIT-SEL clean-88 non-harm"
    if (
        fitsel_clean_vs_gm["STACK_R4_E8_BASE"] >= -0.002
        and p201b_excl_vs_gm["STACK_R4_E8_BASE"] >= 0.0
        and p201b_base_vs_m20 >= 0.0
    ):
        return "STACK_R4_E8_BASE", "R4-e8 BASE >= M20 and >= GM_CW12 on P201b excl10 with FIT-SEL clean-88 non-harm"
    if fitsel_clean_vs_gm["STACK_R4_M20"] >= -0.002 and p201b_excl_vs_gm["STACK_R4_M20"] >= 0.0:
        return "STACK_R4_M20", "CW-5 R4 M20 >= GM_CW12 on P201b excl10 with FIT-SEL clean-88 non-harm"
    return "GM_CW12", "No ICL arm beat GM_CW12 under PREREG §2 criteria; freezing GM_CW12"


def _panel_summary(T16, H, per: dict, cids: list[str], arms: tuple[str, ...], pairs: list[tuple[str, str]]) -> dict:
    unit = {c: per[c]["family"] for c in cids}
    vals = {x: {c: float(per[c][x]) for c in cids} for x in arms}
    pair_stats = {f"{x}_vs_{y}": T16.pair_stats(np, H, vals[x], vals[y], unit) for x, y in pairs}
    an, point, boot, elo_map = T16.elo_boot(np, H, vals, unit, "BASE_SXB", LADDER_SEED, N_BOOT)
    return {
        "n_contexts": len(cids),
        "n_families": len(set(unit.values())),
        "family_mean_u": {x: float(H.family_mean(vals[x], unit)) for x in arms},
        "pairs": pair_stats,
        "elo_vs_BASE_SXB": {
            x: {
                "elo": float(elo_map[x]["elo"]),
                "ci95": [float(v) for v in elo_map[x]["ci95"]],
                "delta_vs_BASE_SXB": T16.delta_elo(np, an, point, boot, x, "BASE_SXB"),
            }
            for x in arms
        },
    }


def _p201b_board_elo(T16, H, board_dir: Path, per_p201b: dict, cids_keep: list[str], candidate_arms: tuple[str, ...]) -> dict:
    fam = {c: per_p201b[c]["family"] for c in cids_keep}
    field: dict[str, dict[str, float]] = {}
    for fp in sorted(board_dir.glob("board_p201b__*.jsonl")):
        if fp.stat().st_size == 0:
            continue
        eid = fp.name[len("board_p201b__") : -len(".jsonl")]
        recs = [json.loads(line) for line in fp.read_text().splitlines() if line.strip()]
        got = {r["cid"]: float(r["utility"]) for r in recs if r["cid"] in fam}
        if len(got) == len(cids_keep):
            field[eid] = got
    if ANCHOR_P201B not in field:
        return {}
    for arm in candidate_arms:
        field[f"h447.{arm}"] = {c: float(per_p201b[c][arm]) for c in cids_keep}
    ordered = dict(sorted(field.items(), key=lambda kv: kv[0] + ".jsonl"))
    _, _, _, elo_map = T16.elo_boot(np, H, ordered, fam, ANCHOR_P201B, LADDER_SEED, N_BOOT)
    rank_order = sorted(elo_map, key=lambda e: -elo_map[e]["elo"])
    return {
        "n_contexts": len(cids_keep),
        "n_entrants": len(rank_order),
        "order": rank_order,
        "elo": {
            e: {"rank": rank_order.index(e) + 1, "elo": float(elo_map[e]["elo"]), "ci95": [float(v) for v in elo_map[e]["ci95"]]}
            for e in rank_order
        },
    }


def cmd_readout(a) -> None:
    out_p = Path(a.out)
    frz_p = Path(a.frozen_out)
    if out_p.exists() or frz_p.exists():
        raise SystemExit("readout or FROZEN_FAIR_CHAMPION.json already exists (create-once)")

    gates = json.loads(Path(a.gates).read_text())
    if not gates.get("pass"):
        raise SystemExit("REFUSING readout: H447 integrity gates did not pass")

    import train_h416 as T16

    _, H = T16._stats_mod()
    ap_f = json.loads(Path(a.apply_fitsel).read_text())
    ap_p = json.loads(Path(a.apply_p201b).read_text())
    pins = json.loads(Path(a.pins_json).read_text())
    has_wc = bool(ap_f["has_wc"] and ap_p["has_wc"])

    arms = (
        "BASE_SXB", "G_ONLY", "GM_CW12", "STACK_R4_M20",
        "STACK_R4_L", "STACK_R4_M", "STACK_R4_E8_BASE", "STACK_R4_E8_2COL",
    ) + (("STACK_R4_E8_WC",) if has_wc else ())

    pairs = [
        ("GM_CW12", "BASE_SXB"),
        ("STACK_R4_M20", "BASE_SXB"),
        ("STACK_R4_M20", "GM_CW12"),
        ("STACK_R4_L", "GM_CW12"),
        ("STACK_R4_M", "GM_CW12"),
        ("STACK_R4_E8_BASE", "BASE_SXB"),
        ("STACK_R4_E8_BASE", "GM_CW12"),
        ("STACK_R4_E8_BASE", "STACK_R4_M20"),
        ("STACK_R4_E8_BASE", "STACK_R4_L"),
        ("STACK_R4_E8_BASE", "STACK_R4_M"),
        ("STACK_R4_E8_2COL", "STACK_R4_E8_BASE"),
    ]
    if has_wc:
        pairs.extend([
            ("STACK_R4_E8_WC", "BASE_SXB"),
            ("STACK_R4_E8_WC", "GM_CW12"),
            ("STACK_R4_E8_WC", "STACK_R4_M20"),
            ("STACK_R4_E8_WC", "STACK_R4_E8_BASE"),
        ])

    excl_f = json.loads(Path(a.excl_fitsel).read_text())["contexts"]
    cids_f_all = sorted(ap_f["per_context"])
    cids_f_clean = [c for c in cids_f_all if excl_f[c]["tier"] != "copy"]

    probed_prefixes = [x.strip() for x in Path(a.probed_p201b).read_text().split() if x.strip()]
    cids_p_all = sorted(ap_p["per_context"])
    cids_p_excl = [c for c in cids_p_all if not any(c.startswith(px) for px in probed_prefixes)]

    sum_f_clean = _panel_summary(T16, H, ap_f["per_context"], cids_f_clean, arms, pairs)
    sum_f_all = _panel_summary(T16, H, ap_f["per_context"], cids_f_all, arms, pairs)
    sum_p_excl = _panel_summary(T16, H, ap_p["per_context"], cids_p_excl, arms, pairs)
    sum_p_all = _panel_summary(T16, H, ap_p["per_context"], cids_p_all, arms, pairs)

    cand_board = ("GM_CW12", "STACK_R4_M20", "STACK_R4_E8_BASE") + (("STACK_R4_E8_WC",) if has_wc else ())
    board_p_excl = _p201b_board_elo(T16, H, Path(a.board_p201b_dir), ap_p["per_context"], cids_p_excl, cand_board)
    board_p_all = _p201b_board_elo(T16, H, Path(a.board_p201b_dir), ap_p["per_context"], cids_p_all, cand_board)

    f_vs_gm = {
        k: float(sum_f_clean["pairs"][f"{k}_vs_GM_CW12"]["clipped_mean"])
        for k in ("STACK_R4_M20", "STACK_R4_E8_BASE") + (("STACK_R4_E8_WC",) if has_wc else ())
    }
    p_vs_gm = {
        k: float(sum_p_excl["pairs"][f"{k}_vs_GM_CW12"]["clipped_mean"])
        for k in ("STACK_R4_M20", "STACK_R4_E8_BASE") + (("STACK_R4_E8_WC",) if has_wc else ())
    }
    wc_vs_base = float(sum_p_excl["pairs"]["STACK_R4_E8_WC_vs_STACK_R4_E8_BASE"]["clipped_mean"]) if has_wc else None
    base_vs_m20 = float(sum_p_excl["pairs"]["STACK_R4_E8_BASE_vs_STACK_R4_M20"]["clipped_mean"])

    selected_arm, reason = select_frozen_champion(has_wc, f_vs_gm, p_vs_gm, wc_vs_base, base_vs_m20)

    if selected_arm == "STACK_R4_E8_WC":
        icl_spec = {
            "enabled": True,
            "recipe": "R4_E8",
            "col_name": "r4_e8",
            "w_l": 0.60,
            "w_m": 0.40,
            "perms_narrow": 4,
            "k_bag_wide": 4,
            "tag_l": "R4E8_L",
            "tag_m": "R4E8_M",
            "ckpt_l": pins["wc_ckpt_l"],
            "sha_l": pins["wc_sha_l"],
            "ckpt_m": pins["wc_ckpt_m"],
            "sha_m": pins["wc_sha_m"],
            "training_bank": pins.get("wc_bank_name", "all_legal_excl_p201b_bank.npz"),
            "training_bank_sha256": pins.get("wc_bank_sha256", pins["all_legal_bank_sha256"]),
        }
    elif selected_arm == "STACK_R4_E8_BASE":
        icl_spec = {
            "enabled": True,
            "recipe": "R4_E8",
            "col_name": "r4_e8",
            "w_l": 0.60,
            "w_m": 0.40,
            "perms_narrow": 4,
            "k_bag_wide": 4,
            "tag_l": "R4E8_L",
            "tag_m": "R4E8_M",
            "ckpt_l": pins["base_ckpt_l"],
            "sha_l": pins["base_sha_l"],
            "ckpt_m": pins["base_ckpt_m"],
            "sha_m": pins["base_sha_m"],
            "training_bank": "fit_icl_bank.npz",
            "training_bank_sha256": pins["fit_icl_bank_sha256"],
        }
    elif selected_arm == "STACK_R4_M20":
        icl_spec = {
            "enabled": True,
            "recipe": "CW5_R4_M20",
            "col_name": "cw5_m20",
            "ckpt_m": pins["m20_ckpt"],
            "sha_m": pins["m20_sha"],
            "training_bank": "fit_icl_bank.npz",
            "training_bank_sha256": pins["fit_icl_bank_sha256"],
        }
    else:
        icl_spec = {"enabled": False, "recipe": "NONE"}

    frozen = {
        "schema": "jope.h447.frozen_fair_champion/v1",
        "wave": "H447",
        "selected_arm": selected_arm,
        "selection_reason": reason,
        "has_wc_evaluated": has_wc,
        "combiner": "guard_matern_core.guard_and_stack_2l",
        "blocks": {
            "gp_pool": "stored_h416_64_candidates",
            "matern_m7": list(GM.PRIMARY_NAMES),
            "exact_keys_k3": list(H430_K3),
            "formulas_f4": list(H433_F4),
            "icl": icl_spec,
        },
        "d91_fairness_rule": {
            "headline_fair_comparison": "H425_without_feature_clear_slice_vs_kumo_tabular_large_e8.v1",
            "canonical_board_score": "all_91_contexts_19_families_scores/h448.APEX_FAIR_CHAMPION.jsonl",
            "screen_method": "screen_h425.screen on views-search X_search ONLY against selected_arm training_bank (0 AUDIT target reads)",
        },
        "summary_metrics": {
            "fitsel_clean88_mean_u": sum_f_clean["family_mean_u"][selected_arm],
            "p201b_excl10_mean_u": sum_p_excl["family_mean_u"][selected_arm],
            "p201b_all201_mean_u": sum_p_all["family_mean_u"][selected_arm],
            "p201b_excl10_gain_vs_BASE_SXB": sum_p_excl["pairs"][f"{selected_arm}_vs_BASE_SXB"]["gain"]
            if selected_arm != "BASE_SXB"
            else 0.0,
            "p201b_excl10_board_elo": (board_p_excl.get("elo") or {}).get(f"h447.{selected_arm}"),
            "p201b_all201_board_elo": (board_p_all.get("elo") or {}).get(f"h447.{selected_arm}"),
        },
        "utc": time.strftime("%FT%TZ", time.gmtime()),
    }

    readout_doc = {
        "schema": "jope.h447.readout/v1",
        "selected_arm": selected_arm,
        "selection_reason": reason,
        "frozen_fair_champion": frozen,
        "fitsel_clean_88": sum_f_clean,
        "fitsel_all_106": sum_f_all,
        "p201b_excl10_191": sum_p_excl,
        "p201b_all_201": sum_p_all,
        "p201b_board_elo_excl10": board_p_excl,
        "p201b_board_elo_all201": board_p_all,
    }

    out_p.parent.mkdir(parents=True, exist_ok=True)
    frz_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "x") as f:
        json.dump(readout_doc, f, indent=2, sort_keys=True)
    with open(frz_p, "x") as f:
        json.dump(frozen, f, indent=2, sort_keys=True)
    out_p.chmod(0o444)
    frz_p.chmod(0o444)

    print(f"== H447 FROZEN FAIR CHAMPION: {selected_arm} ({reason})")
    print("   FIT-SEL clean-88 mean u:", {k: round(v, 4) for k, v in sum_f_clean["family_mean_u"].items()})
    print("   P201b excl10-191 mean u:", {k: round(v, 4) for k, v in sum_p_excl["family_mean_u"].items()})
    if board_p_excl:
        top8 = board_p_excl["order"][:8]
        print("   P201b excl10 top-8 board Elo:", [(e, round(board_p_excl["elo"][e]["elo"], 1)) for e in top8])


def cmd_selftest(_a) -> None:
    # 1. Verify guard_and_stack_2l determinism
    rng = np.random.default_rng(447)
    Xs = rng.normal(size=(100, 6))
    Xa = rng.normal(size=(30, 6))
    ys = 1.8 * Xs[:, 0] - 0.9 * Xs[:, 1] + rng.normal(0.0, 0.2, size=100)
    lab = np.arange(100, dtype=np.int64) % 5
    q1 = 1.8 * Xs[:, 0] + rng.normal(0.0, 0.3, size=100)
    b1 = 1.8 * Xa[:, 0]
    Q_gp = np.column_stack([q1, -0.9 * Xs[:, 1]])
    b_gp = [b1, -0.9 * Xa[:, 1]]
    ext_q = (1.8 * Xs[:, 0] - 0.9 * Xs[:, 1] + rng.normal(0.0, 0.1, size=100))[:, None]
    ext_b = (1.8 * Xa[:, 0] - 0.9 * Xa[:, 1])[:, None]
    p1, _ = GM.guard_and_stack_2l("cid_h447_t1", Xs, Xa, ys, lab, Q_gp, b_gp, b_gp, [(ext_q, ext_b, ext_b)])
    p2, _ = GM.guard_and_stack_2l("cid_h447_t1", Xs, Xa, ys, lab, Q_gp, b_gp, b_gp, [(ext_q, ext_b, ext_b)])
    assert np.array_equal(p1, p2) and np.all(np.isfinite(p1))

    # 2. Verify all 4 branches of select_frozen_champion
    arm, _ = select_frozen_champion(
        True,
        {"STACK_R4_M20": 0.01, "STACK_R4_E8_BASE": 0.02, "STACK_R4_E8_WC": 0.03},
        {"STACK_R4_M20": 0.01, "STACK_R4_E8_BASE": 0.02, "STACK_R4_E8_WC": 0.035},
        0.015,
        0.010,
    )
    assert arm == "STACK_R4_E8_WC"
    arm2, _ = select_frozen_champion(
        False,
        {"STACK_R4_M20": 0.01, "STACK_R4_E8_BASE": 0.02},
        {"STACK_R4_M20": 0.01, "STACK_R4_E8_BASE": 0.02},
        None,
        0.01,
    )
    assert arm2 == "STACK_R4_E8_BASE"
    arm3, _ = select_frozen_champion(
        False,
        {"STACK_R4_M20": 0.01, "STACK_R4_E8_BASE": 0.01},
        {"STACK_R4_M20": 0.02, "STACK_R4_E8_BASE": 0.01},
        None,
        -0.01,
    )
    assert arm3 == "STACK_R4_M20"
    arm4, _ = select_frozen_champion(
        False,
        {"STACK_R4_M20": -0.01, "STACK_R4_E8_BASE": -0.01},
        {"STACK_R4_M20": -0.01, "STACK_R4_E8_BASE": -0.01},
        None,
        0.0,
    )
    assert arm4 == "GM_CW12"

    print(json.dumps({"all_passed": True, "tests": 5}), flush=True)
    print("H447_EVAL_SELFTEST_OK", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selftest").set_defaults(fn=cmd_selftest)

    p1 = sub.add_parser("apply")
    p1.add_argument("--run-dir", required=True)
    p1.add_argument("--tag", required=True)
    p1.add_argument("--panel", required=True)
    p1.add_argument("--bank", required=True)
    p1.add_argument("--matern-cols", required=True)
    p1.add_argument("--h430-cols", required=True)
    p1.add_argument("--h433-cols", required=True)
    p1.add_argument("--r4e8-base-cols", required=True)
    p1.add_argument("--r4e8-wc-cols", default="")
    p1.add_argument("--out", required=True)
    p1.add_argument("--procs", type=int, default=8)
    p1.set_defaults(fn=cmd_apply)

    p2 = sub.add_parser("readout")
    p2.add_argument("--apply-fitsel", required=True)
    p2.add_argument("--apply-p201b", required=True)
    p2.add_argument("--excl-fitsel", required=True)
    p2.add_argument("--probed-p201b", required=True)
    p2.add_argument("--board-p201b-dir", required=True)
    p2.add_argument("--gates", required=True)
    p2.add_argument("--pins-json", required=True)
    p2.add_argument("--out", required=True)
    p2.add_argument("--frozen-out", required=True)
    p2.set_defaults(fn=cmd_readout)

    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
