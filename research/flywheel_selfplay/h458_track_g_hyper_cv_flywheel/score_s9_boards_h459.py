#!/usr/bin/env python3
"""Wave H459 / Track S9 Arm A5: `P201B` (Select) & `FITSEL` (Frozen Confirm) Standalone + Stack Scorer
(`score_s9_boards_h459.py`).

SCORING ONLY on non-`dev91` tuning/confirmation boards (`P201B`: 201 contexts, 23 frozen reference entrants;
`FITSEL`: 106 confirmation contexts). `dev91` is NEVER opened here.

Reports:
  1. Stochastic Seed Noise Unit (`a0_s0` vs `a0_s1`, standalone & inside `gm_ts.stack`) per `cl_fwriter` 00:54Z.
  2. Standalone Stage-by-Stage Out-of-Sample Generalization Curve (`stg1 -> stg2 -> ... -> stgM`, `shrunk`, `ratchet`)
     vs `a0_s0` / `a0_s1` (`mean_u`, `|kurtosis| > 10` slice, `W-T-L`, two-sided sign-flip `p`, `P201B` Elo).
  3. Full Guarded Stack (`gm_ts.stack` `V0_legacy` + extra column) paired against BOTH `V0_legacy` AND
     `V0_legacy + a0_mean`, checking the pre-registered `PREREG_S9_ARM_A5_OC_FGB_GP.md` gates (`P201B` -> `FITSEL`).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, "/home/ubuntu/h450-stage/pub-r15/code")
sys.path.append("/home/ubuntu/h448-stage/pub-r1/code")
sys.path.insert(0, "/home/ubuntu/h458s-stage/code")

import analyze_h450 as AH  # noqa: E402
import gm_ts  # noqa: E402
import meta_h416 as M  # noqa: E402
import s2_tune_board as S2B  # noqa: E402
import train_h416 as T16  # noqa: E402

KUMO = "kumo_tabular_large_e8.v1"
ALLOWED_PREFIX = ("q_", "bag_", "ref_")


def sha_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def u_of(pred: np.ndarray, ya: np.ndarray, ba: float) -> float:
    p = np.asarray(pred, dtype=np.float64)
    if p.shape != ya.shape or not np.all(np.isfinite(p)):
        return float("nan")
    return float(M.utility(p, ya, ba)[0])


def _eval_one_ctx(job: Tuple) -> Dict:
    c, gp_p, Xs, Xa, y, lab, base_blocks, block_dir, col_names, optset = job
    z = np.load(gp_p, allow_pickle=False)
    assert np.array_equal(z["y"].astype(np.float64), y), f"{c}: GP pool y != bank y"
    Q_gp = z["Q"].astype(np.float64)
    ya = z["ya"].astype(np.float64)
    ba = float(z["ba"])
    ym = float(z["ym"])
    m_gp, n_a = Q_gp.shape[1], len(ya)
    mean_a = np.full(n_a, ym, dtype=np.float64)
    bag_gp = [None if not np.all(np.isfinite(z["AKF"][:, i])) else z["AKF"][:, i].astype(np.float64) for i in range(m_gp)]
    ref_gp = [z["PA"][:, i].astype(np.float64) if bool(z["pa_ok"][i]) else None for i in range(m_gp)]
    bcols_gp = [b if b is not None else (r if r is not None else mean_a) for b, r in zip(bag_gp, ref_gp)]

    raw_base = [blk for _, _, blk in base_blocks]
    names_base = [f"{tag}:{v}" for tag, names, _ in base_blocks for v in names]

    p_leg, _, _ = gm_ts.stack(c, Xs, Xa, y, lab, Q_gp, bcols_gp, ref_gp, raw_base, names_base, optset)
    u_stack: Dict[str, float] = {"V0_legacy": u_of(p_leg, ya, ba)}
    u_standalone_ref: Dict[str, float] = {}
    u_standalone_bag: Dict[str, float] = {}
    w_admitted: Dict[str, float] = {}

    extra_npz = block_dir / f"{c}.npz"
    with np.load(extra_npz, allow_pickle=False) as zx:
        bad = [k for k in zx.files if k not in ("ok", "wall_s") and not k.startswith(ALLOWED_PREFIX)]
        assert not bad, f"Disallowed keys in {extra_npz}: {bad}"
        assert bool(zx["ok"]), f"ok=False in {extra_npz}"
        for cn in col_names:
            r_pred = np.asarray(zx[f"ref_{cn}"], dtype=np.float64)
            b_pred = np.asarray(zx[f"bag_{cn}"], dtype=np.float64)
            u_standalone_ref[cn] = u_of(r_pred, ya, ba)
            u_standalone_bag[cn] = u_of(b_pred, ya, ba)

    # Evaluate guarded stack `V0_legacy + single_col` for primary non-trajectory columns
    stack_cols = [cn for cn in col_names if not ("_stg" in cn and not cn.endswith("_stg1"))]
    for cn in stack_cols:
        blk_1 = AH._load_col_block(extra_npz, (cn,), len(y), n_a)
        assert blk_1 is not None, f"Failed _load_col_block for {cn} on {c}"
        p_arm, _, info_arm = gm_ts.stack(
            c,
            Xs,
            Xa,
            y,
            lab,
            Q_gp,
            bcols_gp,
            ref_gp,
            raw_base + [blk_1],
            names_base + [f"s9:{cn}"],
            optset,
        )
        u_stack[f"stack+{cn}"] = u_of(p_arm, ya, ba)
        w_map = dict(zip(info_arm["names_all"], info_arm["w_l1"]))
        w_admitted[cn] = float(w_map.get(f"s9:{cn}", 0.0))

    return {
        "cid": c,
        "u_standalone_ref": u_standalone_ref,
        "u_standalone_bag": u_standalone_bag,
        "u_stack": u_stack,
        "w_admitted": w_admitted,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--board", default="P201B", choices=["P201B", "FITSEL"])
    ap.add_argument("--block", required=True, help="Directory holding {cid}.npz + SIDECAR.json")
    ap.add_argument("--columns", default="", help="Comma list of column names (default: all in SIDECAR.json)")
    ap.add_argument("--control-s0", default="", help="Primary single-stage control col (e.g. a0_medium_pure_s0)")
    ap.add_argument("--control-s1", default="", help="Second-seed single-stage control col (e.g. a0_medium_pure_s1)")
    ap.add_argument("--control-1p5x", default="", help="1.5x single-stage control col (e.g. a0_medium_pure_1p5x)")
    ap.add_argument("--control-comm", default="", help="A6 committee-without-residuals col (e.g. a6_medium_pure_comm)")
    ap.add_argument("--out", required=True, help="Output JSON receipt path")
    ap.add_argument("--procs", type=int, default=16)
    ap.add_argument("--n-boot", type=int, default=400)
    a = ap.parse_args()

    t0 = time.monotonic()
    spec = S2B.BOARDS[a.board]
    assert sha_file(spec["bank"]) == spec["bank_sha256"], f"Bank sha mismatch for {a.board}"

    block_dir = Path(a.block)
    sc = json.loads((block_dir / "SIDECAR.json").read_text())
    col_names = [x.strip() for x in a.columns.split(",") if x.strip()] if a.columns else list(sc["names"])
    assert col_names, "No columns to score"

    field = None
    field_skipped: List[str] = []
    if spec["board"] is not None:
        board_doc = json.loads((spec["board"] / "BOARD.json").read_text())
        fam = {c["cid"]: c["family_id"] for c in board_doc["contexts"]}
        field = {}
        for sp in sorted((spec["board"] / "scores").glob("*.jsonl")):
            recs = [json.loads(x) for x in sp.read_text().splitlines() if x.strip()]
            field[sp.stem] = {r["cid"]: float(r["utility"]) for r in recs}
        field_skipped = sorted(e for e, v in field.items() if not all(c in v for c in fam))
        field = {e: v for e, v in field.items() if e not in field_skipped}
    else:
        with np.load(spec["bank"], allow_pickle=False) as zb:
            fam = {str(c): str(f) for c, f in zip(zb["cids"].tolist(), zb["family"].tolist())}

    rows = {}
    with np.load(spec["bank"], allow_pickle=False) as zb:
        assert "ya" not in zb.files
        b_cids = [str(x) for x in zb["cids"].tolist()]
        ss, sa = zb["s_shape"], zb["a_shape"]
        so = ao = lo = 0
        for i, c in enumerate(b_cids):
            n, w = int(ss[i, 0]), int(ss[i, 1])
            na, f = int(sa[i, 0]), int(sa[i, 1])
            S_mat = np.array(zb["search"][so : so + n * w], np.float64).reshape(n, w)
            Xa = np.array(zb["audit"][ao : ao + na * f], np.float64, copy=True).reshape(na, f)
            rows[c] = (
                np.array(S_mat[:, :-1], np.float64, copy=True),
                Xa,
                np.array(zb["y64"][lo : lo + n], np.float64, copy=True),
                np.array(zb["lab"][lo : lo + n], np.int64, copy=True),
            )
            so += n * w
            ao += na * f
            lo += n

    avail_cids = sorted(c for c in fam if c in rows and (block_dir / f"{c}.npz").is_file())
    if not avail_cids:
        raise SystemExit(f"No {a.board} contexts found in {block_dir}")

    jobs = []
    kurt: Dict[str, float] = {}
    for c in avail_cids:
        Xs, Xa, y, lab = rows[c]
        sy = float(np.std(y))
        kurt[c] = float(np.mean(((y - y.mean()) / sy) ** 4) - 3.0) if sy > 0 else 0.0
        base_blocks = S2B.load_blocks(spec, c, y, len(y), len(Xa))
        jobs.append((c, spec["gp"] / f"{c}.npz", Xs, Xa, y, lab, base_blocks, block_dir, col_names, {}))

    with Pool(processes=min(max(1, a.procs), len(jobs))) as pool:
        res_list = pool.map(_eval_one_ctx, jobs, chunksize=1)

    _, H = T16._stats_mod()
    fk = {c: fam[c] for c in avail_cids}
    ht_cids = [c for c in avail_cids if abs(kurt[c]) > 10.0]
    full_board = len(avail_cids) == len(fam)

    by_cid = {r["cid"]: r for r in res_list}
    standalone_summary: Dict[str, Dict] = {}
    for cn in col_names:
        u_map = {c: by_cid[c]["u_standalone_ref"][cn] for c in avail_cids}
        u_bag_map = {c: by_cid[c]["u_standalone_bag"][cn] for c in avail_cids}
        entry: Dict[str, object] = {
            "mean_u_ref": float(np.mean(list(u_map.values()))),
            "mean_u_bag": float(np.mean(list(u_bag_map.values()))),
            "mean_u_kurt_gt10": float(np.mean([u_map[c] for c in ht_cids])) if ht_cids else None,
        }
        if a.control_s0 and a.control_s0 in col_names and cn != a.control_s0:
            u_c0 = {c: by_cid[c]["u_standalone_ref"][a.control_s0] for c in avail_cids}
            entry["pair_vs_a0_s0"] = T16.pair_stats(np, H, u_map, u_c0, fk)
        if a.control_1p5x and a.control_1p5x in col_names and cn != a.control_1p5x:
            u_c15 = {c: by_cid[c]["u_standalone_ref"][a.control_1p5x] for c in avail_cids}
            entry["pair_vs_a0_1p5x"] = T16.pair_stats(np, H, u_map, u_c15, fk)
        if a.control_comm and a.control_comm in col_names and cn != a.control_comm:
            u_ccm = {c: by_cid[c]["u_standalone_ref"][a.control_comm] for c in avail_cids}
            entry["pair_vs_a6_comm"] = T16.pair_stats(np, H, u_map, u_ccm, fk)
        if full_board and field is not None and not ("_stg" in cn and not cn.endswith("_stg1")):
            el = AH._rank_entrant(H, field, f"s9.standalone.{cn}", u_map, fam, avail_cids, a.n_boot)
            entry["elo"] = el["elo"]["elo"]
            entry["rank"] = el["rank_if_placed_against_frozen_33"]
        standalone_summary[cn] = entry

    # Seed noise unit (`a0_s0` vs `a0_s1`) per `cl_fwriter` 00:54Z / `cl_mtc` 01:32Z
    seed_noise = None
    if a.control_s0 in col_names and a.control_s1 in col_names:
        d_st = np.array([by_cid[c]["u_standalone_ref"][a.control_s0] - by_cid[c]["u_standalone_ref"][a.control_s1] for c in avail_cids])
        d_sk = np.array([by_cid[c]["u_stack"][f"stack+{a.control_s0}"] - by_cid[c]["u_stack"][f"stack+{a.control_s1}"] for c in avail_cids])
        seed_noise = {
            "standalone_abs_mean_diff": abs(float(d_st.mean())),
            "standalone_ctx_sd": float(np.std(d_st, ddof=1)) if len(d_st) > 1 else 0.0,
            "standalone_se_of_mean": float(np.std(d_st, ddof=1) / math.sqrt(len(d_st))) if len(d_st) > 1 else 0.0,
            "stack_abs_mean_diff": abs(float(d_sk.mean())),
            "stack_ctx_sd": float(np.std(d_sk, ddof=1)) if len(d_sk) > 1 else 0.0,
            "stack_se_of_mean": float(np.std(d_sk, ddof=1) / math.sqrt(len(d_sk))) if len(d_sk) > 1 else 0.0,
        }

    stack_keys = sorted(by_cid[avail_cids[0]]["u_stack"].keys())
    stack_summary: Dict[str, Dict] = {}
    u_leg = {c: by_cid[c]["u_stack"]["V0_legacy"] for c in avail_cids}
    u_stk_a0 = (
        {c: by_cid[c]["u_stack"][f"stack+{a.control_s0}"] for c in avail_cids}
        if (a.control_s0 and f"stack+{a.control_s0}" in stack_keys)
        else None
    )
    u_stk_1p5 = (
        {c: by_cid[c]["u_stack"][f"stack+{a.control_1p5x}"] for c in avail_cids}
        if (a.control_1p5x and f"stack+{a.control_1p5x}" in stack_keys)
        else None
    )
    u_stk_comm = (
        {c: by_cid[c]["u_stack"][f"stack+{a.control_comm}"] for c in avail_cids}
        if (a.control_comm and f"stack+{a.control_comm}" in stack_keys)
        else None
    )

    for sk in stack_keys:
        u_s = {c: by_cid[c]["u_stack"][sk] for c in avail_cids}
        ent: Dict[str, object] = {
            "mean_u": float(np.mean(list(u_s.values()))),
            "mean_u_kurt_gt10": float(np.mean([u_s[c] for c in ht_cids])) if ht_cids else None,
        }
        if sk.startswith("stack+"):
            cn_raw = sk.split("+", 1)[1]
            ent["mean_w_admitted"] = float(np.mean([by_cid[c]["w_admitted"].get(cn_raw, 0.0) for c in avail_cids]))
            ps_leg = T16.pair_stats(np, H, u_s, u_leg, fk)
            ent["pair_vs_legacy"] = ps_leg
            if ht_cids:
                ent["delta_kurt_gt10_vs_legacy"] = float(np.mean([u_s[c] - u_leg[c] for c in ht_cids]))
            if u_stk_a0 is not None and sk != f"stack+{a.control_s0}":
                ent["pair_vs_stack_a0_s0"] = T16.pair_stats(np, H, u_s, u_stk_a0, fk)
            if u_stk_1p5 is not None and sk != f"stack+{a.control_1p5x}":
                ent["pair_vs_stack_a0_1p5x"] = T16.pair_stats(np, H, u_s, u_stk_1p5, fk)
            if u_stk_comm is not None and sk != f"stack+{a.control_comm}":
                ent["pair_vs_stack_a6_comm"] = T16.pair_stats(np, H, u_s, u_stk_comm, fk)
        if full_board and field is not None:
            el = AH._rank_entrant(H, field, f"s9.{sk}", u_s, fam, avail_cids, a.n_boot)
            ent["elo"] = el["elo"]["elo"]
            ent["rank"] = el["rank_if_placed_against_frozen_33"]
            ent["gap_vs_kumo_l"] = el["gap_elo"].get(f"vs_{KUMO}")
        stack_summary[sk] = ent

    # Automatic `FITSEL` `clean_88` (`tier != 'copy'`) slice reporting (`0` `fit_icl_bank` overlap!)
    fitsel_clean88 = None
    screen_fitsel_p = Path("/home/ubuntu/jope-data/h335-cl/h431/pub-r2/SCREEN_FITSEL.json")
    if a.board == "FITSEL" and screen_fitsel_p.is_file():
        sf_doc = json.loads(screen_fitsel_p.read_text())
        per_ctx_sf = sf_doc.get("per_context", {})
        clean_cids = [c for c in avail_cids if per_ctx_sf.get(c, {}).get("tier") != "copy"]
        if clean_cids:
            fk_clean = {c: fam[c] for c in clean_cids}
            u_leg_c = {c: by_cid[c]["u_stack"]["V0_legacy"] for c in clean_cids}
            st_c: Dict[str, object] = {}
            for cn in col_names:
                um_c = {c: by_cid[c]["u_standalone_ref"][cn] for c in clean_cids}
                item_c: Dict[str, object] = {"mean_u_ref": float(np.mean(list(um_c.values())))}
                if a.control_s0 and a.control_s0 in col_names and cn != a.control_s0:
                    uc0_c = {c: by_cid[c]["u_standalone_ref"][a.control_s0] for c in clean_cids}
                    item_c["pair_vs_a0_s0"] = T16.pair_stats(np, H, um_c, uc0_c, fk_clean)
                st_c[cn] = item_c
            sk_c: Dict[str, object] = {}
            for sk in stack_keys:
                us_c = {c: by_cid[c]["u_stack"][sk] for c in clean_cids}
                s_item: Dict[str, object] = {"mean_u": float(np.mean(list(us_c.values())))}
                if sk.startswith("stack+"):
                    s_item["pair_vs_legacy"] = T16.pair_stats(np, H, us_c, u_leg_c, fk_clean)
                sk_c[sk] = s_item
            fitsel_clean88 = {"n_clean": len(clean_cids), "standalone": st_c, "stack": sk_c}

    out_doc = {
        "schema": "jope.h459.s9_board_score/v1.1",
        "board": a.board,
        "n_scored": len(avail_cids),
        "n_board_total": len(fam),
        "full_board": full_board,
        "n_kurt_gt10": len(ht_cids),
        "block_dir": str(block_dir),
        "sidecar": sc,
        "sidecar_sha256": sha_file(block_dir / "SIDECAR.json"),
        "code_sha256": sha_file(Path(__file__)),
        "seed_noise_unit": seed_noise,
        "standalone": standalone_summary,
        "stack": stack_summary,
        "fitsel_clean88": fitsel_clean88,
        "elapsed_s": round(time.monotonic() - t0, 2),
        "utc": time.strftime("%FT%TZ", time.gmtime()),
    }
    out_p = Path(a.out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    tmp_p = out_p.with_suffix(".tmp.json")
    tmp_p.write_text(json.dumps(out_doc, indent=2, sort_keys=True, default=float) + "\n")
    tmp_p.replace(out_p)
    print(json.dumps({k: v for k, v in out_doc.items() if k not in ("standalone", "stack")}, sort_keys=True))
    for cn, st in standalone_summary.items():
        print(f"  [standalone] {cn:28s} u_ref={st['mean_u_ref']:.4f} u_bag={st['mean_u_bag']:.4f} elo={st.get('elo')}")
    for sk, st in stack_summary.items():
        ps = st.get("pair_vs_legacy", {})
        print(
            f"  [stack]      {sk:28s} u={st['mean_u']:.4f} elo={st.get('elo')} "
            f"wtl_vs_leg={ps.get('wtl')} p={ps.get('signflip_p')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
