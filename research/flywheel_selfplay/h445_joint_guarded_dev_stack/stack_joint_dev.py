#!/usr/bin/env python3
"""Joint Guarded `APEX-DEV` (`110`-Cluster) Stack Wave (`stack_joint_dev.py`):

  stack_joint_dev.py selftest
  stack_joint_dev.py apply   --run-dir DIR --h417-dir DIR --tag TBOTH --panel DEV --bank FILE
                             --matern-cols DIR --h430-cols DIR --h433-cols DIR --h424-cols DIR
                             --frozen-cw5 FILE [--cw5-cols DIR] --h424-apply-ref FILE --out FILE [--procs N]
  stack_joint_dev.py readout --apply FILE --screen FILE --excl FILE --h432-narrow-ref FILE --gates FILE --out FILE

Evaluates the single pre-registered 5-block CPU stack (`JOINT_DEV`) on `APEX-DEV` (`115` contexts / `110` `H426`
pre-declared `feature` clusters):
  - Base pool (64 cols): stored `H420` `TBOTH` candidates
  - Combiner & admission guard: `H440` `guard_and_stack_2l` (`5710295f91`, unchanged)
  - Block M7: `H440` Duan-smearing `log1p` + Matérn-ARD (`GM.PRIMARY_NAMES`)
  - Block K3: `H430` exact-key encoders (`key_direct`, `key_gbm`, `factor_glm`; `key_direct` fit-range clipped)
  - Block F4: `H433` raw-scale formula features (`form_gbm`, `form_gam`, `form_sr`, `form_interp`; `sr/interp` clipped)
  - Block ICL: `H442` `FROZEN.selected_recipe` (`q_cw5, bag_cw5, ref_cw5`) iff `FROZEN.promote_to_joint_dev == true`,
               else stored `H424` `cols_DEV` (`ridgeres`) restricted to narrow tables `F <= 30` (`H432` default).

Single Primary Comparison (`alpha = 0.025`, `h335_stats.primary_test` on `110` `H426` `feature` clusters):
  - `JOINT_DEV` vs `I_H424` (the current `DEV` column-stack incumbent, exact-replayed against `H424_DEV.apply.json`).

Sensitivity & Labelled Uncorrected Secondaries:
  - `H425` `without_feature` clear slice (`94` units) and `H426` alternative clusterings (`copy`, `any`, `shape`, `original`)
  - `JOINT_DEV` vs `N_H432` (`H432` narrow `F <= 30` incumbent), `B_TBOTH` (`H420` `TBOTH`), and `SXB_H417` (`h416.SXB`)
  - Leave-one-block-out increments: `JOINT_DEV` vs `NO_GUARD2L`, `NO_MATERN`, `NO_K3`, `NO_F4`, `NO_ICL`.
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
import readout_h420 as R20  # noqa: E402
import restate_h426 as R26  # noqa: E402

SXB_KEY = "P3+LIB3.SXB"
H430_K3 = ("key_direct", "key_gbm", "factor_glm")
H433_F4 = ("form_gbm", "form_gam", "form_sr", "form_interp")
WIDE_F = 30
BOOT_SEED_PAIR, BOOT_SEED_MULTI, N_BOOT = 426_150, 445_150, 10_000


def _u(pred, ya, ba) -> float:
    return float(M.utility(pred, ya, ba)[0])


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
    (
        run_dir,
        tag,
        matern_dir,
        h430_dir,
        h433_dir,
        h424_dir,
        cw5_dir,
        promote_cw5,
        cid,
        bank_row,
        sxb_h417_val,
        h424_ref_row,
    ) = job
    t0 = time.process_time()
    rec, a = CA.stored(run_dir, tag, cid)
    st = rec["outputs"][SXB_KEY]
    if st.get("invalid") or st.get("spec", {}).get("w") is None:
        raise SystemExit(f"{cid}: missing stored baseline output")

    # 1. Exact replay check of stored B_TBOTH (H420 TBOTH == H424.B)
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

    # 2. Exact replay check of stored H424.I (I_H424) and H432.N (N_H432)
    blk_h424_raw = _load_col_block(Path(h424_dir) / f"{cid}.npz", ("ridgeres",), n_s, n_a)
    if blk_h424_raw is not None:
        wi = CA.bcar(np.hstack([a["Q"], blk_h424_raw[0]]), a["y"], cid)
        i_h424 = _u(0.5 * np.hstack([a["BAG"], blk_h424_raw[1]]) @ wi + 0.5 * np.hstack([a["REF"], blk_h424_raw[2]]) @ wi, a["ya"], a["ba"])
    else:
        i_h424 = b_val
    n_h432 = i_h424 if width_f <= WIDE_F else b_val

    b_ref_diff = abs(b_val - float(h424_ref_row["B"])) if h424_ref_row is not None else 0.0
    i_ref_diff = abs(i_h424 - float(h424_ref_row["I"])) if h424_ref_row is not None else 0.0

    # 3. Raw stored AKF / PA / pa_ok for guard_and_stack_2l
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

    # 4. Load the 4 external candidate blocks (M7, K3, F4, ICL)
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

    blk_r0_narrow = blk_h424_raw if width_f <= WIDE_F else None
    if promote_cw5:
        if not cw5_dir:
            raise SystemExit("promote_to_joint_dev is true but --cw5-cols was not supplied")
        blk_icl = _load_col_block(Path(cw5_dir) / f"{cid}.npz", ("cw5",), n_s, n_a)
        icl_expected = True
    else:
        blk_icl = blk_r0_narrow
        icl_expected = bool(width_f <= WIDE_F)

    def _run_guarded(blocks: list[tuple[np.ndarray, np.ndarray, np.ndarray] | None]) -> tuple[float, dict]:
        active = [b for b in blocks if b is not None]
        pred, meta = GM.guard_and_stack_2l(cid, Xs, Xa, a["y"], a["lab"], a["Q"], bcols_gp, ref_gp, active)
        return _u(pred, a["ya"], a["ba"]), meta

    u_joint, meta_joint = _run_guarded([blk_m, blk_k, blk_f, blk_icl])
    u_no_matern, _ = _run_guarded([blk_k, blk_f, blk_icl])
    u_no_k3, _ = _run_guarded([blk_m, blk_f, blk_icl])
    u_no_f4, _ = _run_guarded([blk_m, blk_k, blk_icl])
    u_no_icl, _ = _run_guarded([blk_m, blk_k, blk_f])
    u_no_struct, _ = _run_guarded([blk_m, blk_r0_narrow])
    u_icl_r0, _ = _run_guarded([blk_m, blk_k, blk_f, blk_r0_narrow])

    all_active = [b for b in (blk_m, blk_k, blk_f, blk_icl) if b is not None]
    if all_active:
        Q_u = np.hstack([a["Q"]] + [x[0] for x in all_active])
        B_u = np.hstack([a["BAG"]] + [x[1] for x in all_active])
        R_u = np.hstack([a["REF"]] + [x[2] for x in all_active])
        wu = CA.bcar(Q_u, a["y"], cid)
        u_no_guard2l = _u(0.5 * B_u @ wu + 0.5 * R_u @ wu, a["ya"], a["ba"])
    else:
        u_no_guard2l = b_val

    return {
        "cid": cid,
        "family": rec["family_id"],
        "width_f": width_f,
        "n_search": n_s,
        "n_audit": n_a,
        "aligned": aligned,
        "w_diff": float(np.abs(ws - w0).max()),
        "u_diff": abs(u_re - b_val),
        "h424_b_diff": b_ref_diff,
        "h424_i_diff": i_ref_diff,
        "matern_ok": bool(blk_m is not None),
        "h430_ok": bool(blk_k is not None),
        "h433_ok": bool(blk_f is not None),
        "icl_expected": icl_expected,
        "icl_ok": bool(blk_icl is not None) if icl_expected else True,
        "icl_used": bool(blk_icl is not None),
        "gp_dropped": meta_joint["gp_dropped"],
        "ext_admitted": meta_joint["ext_admitted"],
        "ext_weight_share": meta_joint["ext_weight_share"],
        "l2_gain": meta_joint["l2_gain"],
        "SXB_H417": float(sxb_h417_val),
        "B_TBOTH": b_val,
        "I_H424": i_h424,
        "N_H432": n_h432,
        "JOINT_DEV": u_joint,
        "NO_GUARD2L": u_no_guard2l,
        "NO_MATERN": u_no_matern,
        "NO_K3": u_no_k3,
        "NO_F4": u_no_f4,
        "NO_ICL": u_no_icl,
        "NO_STRUCT": u_no_struct,
        "ICL_R0_ONLY": u_icl_r0,
        "cpu_s": time.process_time() - t0,
    }


def cmd_apply(a) -> None:
    out_p = Path(a.out)
    if out_p.exists():
        raise SystemExit(f"{a.out} exists (create-once)")

    frz = json.loads(Path(a.frozen_cw5).read_text())
    promote_cw5 = bool(frz["promote_to_joint_dev"])
    cfg = json.loads((Path(a.run_dir) / f"config_{a.tag}.json").read_text())
    cids = cfg["cids"]

    if a.h417_dir:
        sxb_h417_map = R20._load(Path(a.h417_dir), "BASE", only=set(cids))[2]["SXB"]
    else:
        sxb_h417_map = {c: 0.0 for c in cids}

    h424_ref = json.loads(Path(a.h424_apply_ref).read_text())["per_context"] if a.h424_apply_ref else {}

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
        (
            a.run_dir,
            a.tag,
            a.matern_cols,
            a.h430_cols,
            a.h433_cols,
            a.h424_cols,
            a.cw5_cols,
            promote_cw5,
            c,
            bank[c],
            sxb_h417_map[c],
            h424_ref.get(c),
        )
        for c in cids
    ]
    with Pool(a.procs) as pool:
        res = pool.map(one_context, jobs, chunksize=1)

    out = {
        "schema": "jope.joint_dev.apply/v1",
        "panel": a.panel,
        "tag": a.tag,
        "n": len(res),
        "bank_sha256": hashlib.sha256(Path(a.bank).read_bytes()).hexdigest(),
        "frozen_cw5_sha256": hashlib.sha256(Path(a.frozen_cw5).read_bytes()).hexdigest(),
        "promote_cw5": promote_cw5,
        "cw5_selected_recipe": frz.get("selected_recipe"),
        "aligned_all": bool(all(r["aligned"] for r in res)),
        "matern_ok": int(sum(r["matern_ok"] for r in res)),
        "h430_ok": int(sum(r["h430_ok"] for r in res)),
        "h433_ok": int(sum(r["h433_ok"] for r in res)),
        "icl_ok": int(sum(r["icl_ok"] for r in res)),
        "icl_used_count": int(sum(r["icl_used"] for r in res)),
        "total_gp_dropped_by_guard": int(sum(r["gp_dropped"] for r in res)),
        "replay_max_w_diff": float(max(r["w_diff"] for r in res)),
        "replay_max_u_diff": float(max(r["u_diff"] for r in res)),
        "h424_max_b_diff": float(max(r["h424_b_diff"] for r in res)),
        "h424_max_i_diff": float(max(r["h424_i_diff"] for r in res)),
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
            "panel", "n", "promote_cw5", "aligned_all", "matern_ok", "h430_ok", "h433_ok", "icl_ok",
            "replay_max_w_diff", "replay_max_u_diff", "h424_max_i_diff", "cpu_s",
        )
    }), flush=True)


def _two_arm_elo(T16, H, a_vals: dict, b_vals: dict, unit_of: dict) -> dict:
    arms, point, boot, _ = T16.elo_boot(np, H, {"a": a_vals, "b": b_vals}, unit_of, "b", BOOT_SEED_PAIR, N_BOOT)
    d = T16.delta_elo(np, arms, point, boot, "a", "b")
    return {"delta": float(d["delta"]), "ci95": [float(d["ci95"][0]), float(d["ci95"][1])]}


def cmd_readout(a) -> None:
    out_p = Path(a.out)
    if out_p.exists():
        raise SystemExit(f"{a.out} exists (create-once)")
    gates = json.loads(Path(a.gates).read_text())
    if not gates.get("pass"):
        raise SystemExit("REFUSING readout: integrity gates did not pass")

    import train_h416 as T16

    _, H = T16._stats_mod()
    ap = json.loads(Path(a.apply).read_text())
    per = ap["per_context"]
    cids = sorted(per)
    excl = json.loads(Path(a.excl).read_text())
    screen = json.loads(Path(a.screen).read_text())
    if hashlib.sha256(Path(a.screen).read_bytes()).hexdigest() != excl["screen_sha256"]:
        raise SystemExit("screen_sha256 mismatch vs exclusion file")
    if sorted(excl["contexts"]) != cids or sorted(screen["per_context"]) != cids:
        raise SystemExit("screen/exclusion files do not match the apply contexts")

    # Verify exact reproduction of H432 narrow (N_H432 vs B_TBOTH) if provided
    cl_all = R26.clusterings(screen, cids)
    cl_feat = cl_all["feature"]
    fam_unit = {c: per[c]["family"] for c in cids}

    arms = (
        "SXB_H417", "B_TBOTH", "I_H424", "N_H432",
        "JOINT_DEV", "NO_GUARD2L", "NO_MATERN", "NO_K3", "NO_F4", "NO_ICL", "NO_STRUCT", "ICL_R0_ONLY",
    )
    vals = {x: {c: float(per[c][x]) for c in cids} for x in arms}

    h432_check_diff = None
    if a.h432_narrow_ref:
        ref_n = json.loads(Path(a.h432_narrow_ref).read_text())["N_vs_B"]
        st_n = T16.pair_stats(np, H, vals["N_H432"], vals["B_TBOTH"], cl_feat)
        h432_check_diff = float(abs(st_n["clipped_mean"] - ref_n["clipped_mean"]))
        if h432_check_diff > 1e-9:
            raise SystemExit(f"N_H432 failed to reproduce H432_DEV.narrow.json (diff={h432_check_diff})")

    # 1. SINGLE PRIMARY TEST (alpha = 0.025, H426 pre-declared feature clusters, n_units = 110):
    prim = dict(T16.pair_stats(np, H, vals["JOINT_DEV"], vals["I_H424"], cl_feat))
    prim["units"] = len(set(cl_feat.values()))
    prim["two_arm_elo"] = _two_arm_elo(T16, H, vals["JOINT_DEV"], vals["I_H424"], cl_feat)

    if prim["pass"]:
        verdict = "JOINT_DEV_CONFIRMED"
    elif prim["clipped_mean"] < 0 and prim["mirror_signflip_p"] < 0.025:
        verdict = "JOINT_DEV_HARM"
    else:
        verdict = "JOINT_DEV_NULL"

    # 2. Sensitivity slices (no verdict change): H425 clear slice + H426 alternative clusterings
    clean_cids = [c for c in cids if excl["contexts"][c]["tier"] not in ("copy", "possible")]
    cl_clean = {c: cl_feat[c] for c in clean_cids}
    sens_h425 = dict(T16.pair_stats(
        np, H, {c: vals["JOINT_DEV"][c] for c in clean_cids}, {c: vals["I_H424"][c] for c in clean_cids}, cl_clean
    ))
    sens_h425["contexts"] = len(clean_cids)
    sens_h425["units"] = len(set(cl_clean.values()))
    sens_h425["two_arm_elo"] = _two_arm_elo(
        T16, H, {c: vals["JOINT_DEV"][c] for c in clean_cids}, {c: vals["I_H424"][c] for c in clean_cids}, cl_clean
    )

    sens_clusterings = {}
    for cname, cmap in [("original_families", fam_unit)] + [(k, cl_all[k]) for k in ("feature", "copy", "any", "shape")]:
        s = dict(T16.pair_stats(np, H, vals["JOINT_DEV"], vals["I_H424"], cmap))
        s["units"] = len(set(cmap.values()))
        sens_clusterings[cname] = s

    # 3. Explicitly Labelled Uncorrected Secondaries (110 feature clusters & 94 H425 clear clusters)
    sec_pairs = (
        ("JOINT_DEV", "N_H432"),
        ("JOINT_DEV", "B_TBOTH"),
        ("JOINT_DEV", "SXB_H417"),
        ("JOINT_DEV", "NO_GUARD2L"),
        ("JOINT_DEV", "NO_MATERN"),
        ("JOINT_DEV", "NO_K3"),
        ("JOINT_DEV", "NO_F4"),
        ("JOINT_DEV", "NO_ICL"),
        ("JOINT_DEV", "NO_STRUCT"),
        ("JOINT_DEV", "ICL_R0_ONLY"),
        ("ICL_R0_ONLY", "NO_ICL"),
        ("I_H424", "B_TBOTH"),
        ("B_TBOTH", "SXB_H417"),
    )
    secondaries_feature110 = {}
    secondaries_clear94 = {}
    for x, y in sec_pairs:
        key = f"{x}_vs_{y}"
        s110 = dict(T16.pair_stats(np, H, vals[x], vals[y], cl_feat))
        s110["two_arm_elo"] = _two_arm_elo(T16, H, vals[x], vals[y], cl_feat)
        secondaries_feature110[key] = s110

        s94 = dict(T16.pair_stats(
            np, H, {c: vals[x][c] for c in clean_cids}, {c: vals[y][c] for c in clean_cids}, cl_clean
        ))
        s94["two_arm_elo"] = _two_arm_elo(
            T16, H, {c: vals[x][c] for c in clean_cids}, {c: vals[y][c] for c in clean_cids}, cl_clean
        )
        secondaries_clear94[key] = s94

    # 5-arm campaign progression ladder on 110 feature clusters anchored at SXB_H417 = 1500
    prog_arms = ("SXB_H417", "B_TBOTH", "I_H424", "N_H432", "JOINT_DEV")
    an, point, boot, elo_map = T16.elo_boot(
        np, H, {k: vals[k] for k in prog_arms}, cl_feat, "SXB_H417", BOOT_SEED_MULTI, N_BOOT
    )
    multi_elo = {
        k: {
            "elo": float(elo_map[k]["elo"]),
            "ci95": [float(v) for v in elo_map[k]["ci95"]],
            "delta_vs_SXB_H417": T16.delta_elo(np, an, point, boot, k, "SXB_H417"),
        }
        for k in prog_arms
    }

    res = {
        "schema": "jope.joint_dev.readout/v1",
        "verdict": verdict,
        "panel": ap["panel"],
        "n_contexts": len(cids),
        "n_feature_clusters": len(set(cl_feat.values())),
        "n_clear_contexts": len(clean_cids),
        "n_clear_clusters": len(set(cl_clean.values())),
        "promote_cw5": ap["promote_cw5"],
        "cw5_selected_recipe": ap["cw5_selected_recipe"],
        "h432_narrow_repro_diff": h432_check_diff,
        "unit_mean_u_feature110": {x: float(H.family_mean(vals[x], cl_feat)) for x in arms},
        "unit_mean_u_clear94": {
            x: float(H.family_mean({c: vals[x][c] for c in clean_cids}, cl_clean)) for x in arms
        },
        "primary_JOINT_DEV_vs_I_H424_feature110": prim,
        "sensitivity": {
            "h425_without_feature_clear94": sens_h425,
            "h426_clusterings": sens_clusterings,
        },
        "secondaries_uncorrected_feature110": secondaries_feature110,
        "secondaries_uncorrected_clear94": secondaries_clear94,
        "campaign_progression_elo_feature110": multi_elo,
    }

    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "x") as f:
        json.dump(res, f, indent=2, sort_keys=True)
    out_p.chmod(0o444)

    print(f"== JOINT_DEV ({res['panel']}) contexts={res['n_contexts']} feature_clusters={res['n_feature_clusters']} verdict={verdict}")
    print(
        f"   PRIMARY JOINT_DEV vs I_H424 (110 clusters): gain={prim['gain']:+.4f} u ({100*prim['pct_rmse']:+.2f}% RMSE) "
        f"clipped={prim['clipped_mean']:+.4f} p={prim['signflip_p']:.3g} lofo={prim['lofo_min']:+.4f} "
        f"med={prim['family_median']:+.4f} W/T/L={prim['wtl']} pass={prim['pass']} "
        f"2arm_dElo={prim['two_arm_elo']['delta']:+.1f} {prim['two_arm_elo']['ci95']}"
    )
    for k in ("JOINT_DEV_vs_N_H432", "JOINT_DEV_vs_B_TBOTH", "JOINT_DEV_vs_SXB_H417",
              "JOINT_DEV_vs_NO_GUARD2L", "JOINT_DEV_vs_NO_MATERN", "JOINT_DEV_vs_NO_K3",
              "JOINT_DEV_vs_NO_F4", "JOINT_DEV_vs_NO_ICL", "JOINT_DEV_vs_NO_STRUCT",
              "JOINT_DEV_vs_ICL_R0_ONLY"):
        s = secondaries_feature110[k]
        print(
            f"   [SEC 110] {k:23s}: gain={s['gain']:+.4f} clipped={s['clipped_mean']:+.4f} "
            f"p={s['signflip_p']:.3g} W/T/L={s['wtl']} pass={s['pass']} 2arm_dElo={s['two_arm_elo']['delta']:+.1f}"
        )


def cmd_selftest(_a) -> None:
    rng = np.random.default_rng(445)
    n_s, n_a, f = 120, 40, 8
    Xs = rng.normal(size=(n_s, f))
    Xa = rng.normal(size=(n_a, f))
    ys = 2.0 * Xs[:, 0] - 1.5 * np.sin(Xs[:, 1]) + rng.normal(0.0, 0.25, size=n_s)
    lab = np.arange(n_s, dtype=np.int64) % 5
    q1 = 2.0 * Xs[:, 0] + rng.normal(0.0, 0.35, size=n_s)
    b1 = 2.0 * Xa[:, 0]
    q2 = -1.5 * np.sin(Xs[:, 1]) + rng.normal(0.0, 0.35, size=n_s)
    b2 = -1.5 * np.sin(Xa[:, 1])
    Q_gp = np.column_stack([q1, q2])
    b_gp = [b1, b2]
    r_gp = [b1.copy(), b2.copy()]
    ext_q = (2.0 * Xs[:, 0] - 1.5 * np.sin(Xs[:, 1]) + rng.normal(0.0, 0.15, size=n_s))[:, None]
    ext_b = (2.0 * Xa[:, 0] - 1.5 * np.sin(Xa[:, 1]))[:, None]
    p_a, m_a = GM.guard_and_stack_2l("cid_test_0001", Xs, Xa, ys, lab, Q_gp, b_gp, r_gp, [(ext_q, ext_b, ext_b)])
    p_b, _ = GM.guard_and_stack_2l("cid_test_0001", Xs, Xa, ys, lab, Q_gp, b_gp, r_gp, [(ext_q, ext_b, ext_b)])
    assert np.array_equal(p_a, p_b), "non-deterministic guard_and_stack_2l"
    assert np.all(np.isfinite(p_a)) and m_a["ext_admitted"] == 1
    # Verify H426 clustering union-find helper on synthetic screen dict
    fake_screen = {
        "per_context": {
            "c1": {"width": 10, "rows": 100, "s1": {"flag": True, "fit": "fitA"}},
            "c2": {"width": 10, "rows": 100, "s2": {"flag": True, "fit": "fitA"}},
            "c3": {"width": 25, "rows": 200, "s1": {"flag": False, "fit": "fitA"}},
        }
    }
    cl = R26.clusterings(fake_screen, ["c1", "c2", "c3"])["feature"]
    assert cl["c1"] == cl["c2"] and cl["c3"] != cl["c1"]
    print(json.dumps({"all_passed": True, "tests": 2}), flush=True)
    print("JOINT_DEV_SELFTEST_OK", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p0 = sub.add_parser("selftest")
    p0.set_defaults(fn=cmd_selftest)

    p1 = sub.add_parser("apply")
    p1.add_argument("--run-dir", required=True)
    p1.add_argument("--h417-dir", default="")
    p1.add_argument("--tag", default="TBOTH")
    p1.add_argument("--panel", required=True)
    p1.add_argument("--bank", required=True)
    p1.add_argument("--matern-cols", required=True)
    p1.add_argument("--h430-cols", required=True)
    p1.add_argument("--h433-cols", required=True)
    p1.add_argument("--h424-cols", required=True)
    p1.add_argument("--frozen-cw5", required=True)
    p1.add_argument("--cw5-cols", default="")
    p1.add_argument("--h424-apply-ref", default="")
    p1.add_argument("--out", required=True)
    p1.add_argument("--procs", type=int, default=8)
    p1.set_defaults(fn=cmd_apply)

    p2 = sub.add_parser("readout")
    p2.add_argument("--apply", required=True)
    p2.add_argument("--screen", required=True)
    p2.add_argument("--excl", required=True)
    p2.add_argument("--h432-narrow-ref", default="")
    p2.add_argument("--gates", required=True)
    p2.add_argument("--out", required=True)
    p2.set_defaults(fn=cmd_readout)

    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
