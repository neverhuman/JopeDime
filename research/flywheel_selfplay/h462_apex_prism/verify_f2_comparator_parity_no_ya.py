#!/usr/bin/env python3
"""`F2` (`cl_mtc` `16:09:45Z` & `16:27:16Z`) `91/91` `0`-`ya` Comparator Bit-Parity Verifier & Bundle Emitter.

1. Emits the `1842.03` HWM comparator (`tabcv7_cvrep8_prec48_all`, `valley_mode="v12_avg"`) per-member
   `(member_oof, member_oof_hwm_fast, member_audit, p_oof_ens, p_audit_ens)` bundle across all `91/91`
   contexts with `0` `ya`/`ba` reads (`NoLabelNpz` + `assert_same_outer_folds` on `D1`/`D2`).
2. Independently computes the `AUDIT-X` prediction vector of `eval_h462_tabcv7_cvrep.py` (`tabcv7_cvrep8_prec48_all`)
   across all `91/91` contexts using `NoLabelNpz` (`0` `ya`/`ba` reads, `0` `M.utility` calls, `0` `dev91` looks)
   and asserts `max |p_audit_ens - p_ref_1842| == 0.0`.
3. Writes `0444` create-once receipt `F2_COMPARATOR_1842_PARITY_NO_YA.json`.
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse
import hashlib
import json
import subprocess
import sys
import time
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
for _p in (str(REPO_ROOT), str(HERE.parent), "/home/ubuntu/h452-stage/code", "/home/ubuntu/h450-stage/pub-r6/code"):
    if Path(_p).is_dir() and _p not in sys.path:
        sys.path.insert(0, _p)

from research.flywheel_selfplay.h462_apex_prism import eval_h462_tabcv7_cvrep as TC  # noqa: E402
from research.flywheel_selfplay.h462_apex_prism import guard_matern_core as GM  # noqa: E402
from research.flywheel_selfplay.h462_apex_prism.edge_certify import bag_predictions_precision_weighted_oof  # noqa: E402
from research.flywheel_selfplay.h462_apex_prism.emit_p2_p4_members_search_only import (  # noqa: E402
    H430_K3,
    H433_F4,
    H450_TAB_COLS,
    NoLabelNpz,
    _check_safe_path,
    load_col_block_no_ya,
    load_search_only_bank,
)
from research.flywheel_selfplay.h462_apex_prism.stack_guard_cv7_h452 import (  # noqa: E402
    make_stratified_kfold,
    stack_guard_cv7_context,
)


def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def _ref_1842_worker(job: Tuple) -> Dict[str, object]:
    """Reproduces `eval_h462_tabcv7_cvrep.py:362-454` `tabcv7_cvrep8_prec48_all` with `NoLabelNpz` (`0` `ya` read)."""
    c, Xs, Xa, y, lab0, d1_p, d2_p, blk_m, blk_k, blk_f, blk_tab, r4_wc, r4_s1, r4_s2, mem_p = job
    n_s, n_a = len(y), len(Xa)

    gp_packs = []
    for gp_p in (d1_p, d2_p):
        with NoLabelNpz(gp_p) as zg:
            Q_gp = np.asarray(zg["Q"], np.float64)
            ym = float(zg["ym"])
            m_gp = Q_gp.shape[1]
            mean_a = np.full(n_a, ym, dtype=np.float64)
            PA = np.asarray(zg["PA"], np.float64)
            AKF = np.asarray(zg["AKF"], np.float64)
            pa_ok = np.asarray(zg["pa_ok"], dtype=bool)
            bag_gp = [None if not np.all(np.isfinite(AKF[:, i])) else AKF[:, i].astype(np.float64) for i in range(m_gp)]
            ref_gp = [PA[:, i].astype(np.float64) if bool(pa_ok[i]) else None for i in range(m_gp)]
            bcols = [b if b is not None else (r if r is not None else mean_a) for b, r in zip(bag_gp, ref_gp)]
            rcols = ref_gp
        gp_packs.append((Q_gp, bcols, rcols))

    base_common_cv7 = [
        (t, b) for t, b in (("m7", blk_m), ("k3", blk_k), ("f4", blk_f), ("tab450", blk_tab)) if b is not None
    ]
    labs_rep = [lab0 if r == 0 else make_stratified_kfold(c, y, k=5, rep=r) for r in range(8)]
    cids_rep = [c if r == 0 else f"{c}#cvrep{r}" for r in range(8)]

    preds_all: List[np.ndarray] = []
    oofs_all: List[np.ndarray] = []
    for r4_blk in (r4_wc, r4_s1, r4_s2):
        tb_cv7 = list(base_common_cv7) + ([("e8", r4_blk)] if r4_blk is not None else [])
        for Q_gp, bcols, rcols in gp_packs:
            for r in range(8):
                cr = cids_rep[r]
                lr = labs_rep[r]
                p_tb, _ = stack_guard_cv7_context(cr, Xs, Xa, y, lr, Q_gp, bcols, rcols, tb_cv7, use_cv7_folds=True)
                q_tb = TC._nested_oof_cv7_rep(cr, y, lr, Q_gp, tb_cv7)
                preds_all.append(p_tb)
                oofs_all.append(q_tb)

    p_ref_1842, _ = bag_predictions_precision_weighted_oof(y, oofs_all, preds_all)
    AUD_ref = np.column_stack(preds_all)
    OOF_fast_ref = np.column_stack(oofs_all)

    with np.load(mem_p, allow_pickle=False) as zm:
        p_emit = np.asarray(zm["p_audit_ens"], dtype=np.float64)
        p_emit_real = np.asarray(zm["p_audit_ens_real_stacker"], dtype=np.float64)
        AUD_emit = np.asarray(zm["member_audit"], dtype=np.float64)
        OOF_fast_emit = np.asarray(zm["member_oof_hwm_fast"], dtype=np.float64)
        OOF_real_emit = np.asarray(zm["member_oof"], dtype=np.float64)

    max_diff_ens = float(np.max(np.abs(p_emit - p_ref_1842)))
    max_diff_member_aud = float(np.max(np.abs(AUD_emit - AUD_ref)))
    max_diff_fast_oof = float(np.max(np.abs(OOF_fast_emit - OOF_fast_ref)))
    max_diff_hwm_vs_real_aud = float(np.max(np.abs(p_emit - p_emit_real)))
    rmse_real_vs_fast_oof = float(np.sqrt(np.mean((OOF_real_emit - OOF_fast_emit) ** 2)))
    return {
        "cid": c,
        "n_search": n_s,
        "n_audit": n_a,
        "max_abs_diff_p_audit_ens_vs_1842_hwm": max_diff_ens,
        "max_abs_diff_member_audit": max_diff_member_aud,
        "max_abs_diff_member_oof_hwm_fast": max_diff_fast_oof,
        "max_abs_diff_hwm_vs_real_stacker_audit": max_diff_hwm_vs_real_aud,
        "rmse_real_vs_fast_member_oof": rmse_real_vs_fast_oof,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bank", default="/home/ubuntu/jope-data/h450-quarantine/bank_d91_search_f64.npz")
    ap.add_argument("--d1-dir", default="/home/ubuntu/jope_plan/evidence/H418/pulled/D1/h416_D1_npz")
    ap.add_argument("--d2-dir", default="/home/ubuntu/jope_plan/evidence/H418/pulled/D2/h416_D2_npz")
    ap.add_argument("--m7-dir", default="/home/ubuntu/jope-data/h462-fastclock/xbabe3_stage/d91_inputs/cols_matern")
    ap.add_argument("--k3-dir", default="/home/ubuntu/jope-data/h462-fastclock/xbabe3_stage/d91_inputs/cols_k3")
    ap.add_argument("--f4-dir", default="/home/ubuntu/jope-data/h462-fastclock/xbabe3_stage/d91_inputs/cols_f4")
    ap.add_argument("--tab450-dir", default="/home/ubuntu/jope-data/h462-fastclock/xbabe3_stage/cols_d91/cols_tab_h450_D91")
    ap.add_argument("--wc-r4e8-dir", default="/home/ubuntu/jope-data/h462-fastclock/xbabe3_stage/d91_inputs/cols_r4_e8")
    ap.add_argument("--s1-r4e8-dir", default="/home/ubuntu/jope-data/h462-fastclock/xbabe3_stage/cols_d91/r4e8_s46201_v12_avg")
    ap.add_argument("--s2-r4e8-dir", default="/home/ubuntu/jope-data/h462-fastclock/xbabe3_stage/cols_d91/r4e8_s46202_v12_avg")
    ap.add_argument("--members-out-dir", default="/home/ubuntu/jope-data/h462-fastclock/p2_p4_members/comparator_hwm1842")
    ap.add_argument(
        "--members-receipt-json",
        default="/home/ubuntu/jope-data/h462-fastclock/p2_p4_members/comparator_hwm1842_receipt.json",
    )
    ap.add_argument(
        "--canary-sample-json",
        default="/home/ubuntu/jope_plan/evidence/H458/trackS/S10F/S10F_DECLARED_SAMPLE_19.json",
        help="Path to S10F_DECLARED_SAMPLE_19.json for F3(b) real-SEARCH stacker canary",
    )
    ap.add_argument("--out-json", default="/home/ubuntu/jope_plan/evidence/H462/F2_COMPARATOR_1842_PARITY_NO_YA.json")
    ap.add_argument("--procs", type=int, default=20)
    a = ap.parse_args()

    out_json = Path(a.out_json)
    mem_dir = Path(a.members_out_dir)
    mem_rcpt = Path(a.members_receipt_json)
    for p in (out_json, mem_dir, mem_rcpt):
        _check_safe_path(p)
    if out_json.exists():
        raise SystemExit(f"REFUSAL: create-once F2 receipt already exists: {out_json}")

    t0 = time.monotonic()
    if not mem_rcpt.exists():
        cmd = [
            sys.executable,
            str(HERE / "emit_p2_p4_members_search_only.py"),
            "--bank",
            a.bank,
            "--d1-dir",
            a.d1_dir,
            "--d2-dir",
            a.d2_dir,
            "--m7-dir",
            a.m7_dir,
            "--k3-dir",
            a.k3_dir,
            "--f4-dir",
            a.f4_dir,
            "--tab450-dir",
            a.tab450_dir,
            "--wc-r4e8-dir",
            a.wc_r4e8_dir,
            "--s1-r4e8-dir",
            a.s1_r4e8_dir,
            "--s2-r4e8-dir",
            a.s2_r4e8_dir,
            "--cv-repeats",
            "8",
            "--audit-weight-mode",
            "hwm_1842_exact",
            "--out-dir",
            str(mem_dir),
            "--receipt-json",
            str(mem_rcpt),
            "--procs",
            str(a.procs),
        ]
        if a.canary_sample_json.strip() and Path(a.canary_sample_json.strip()).is_file():
            cmd.extend(["--canary-sample-json", a.canary_sample_json.strip()])
        subprocess.run(cmd, check=True)

    bank_rows = load_search_only_bank(Path(a.bank))
    cids = sorted(bank_rows)
    jobs = []
    for c in cids:
        _fam, Xs, Xa, y, lab0 = bank_rows[c]
        n_s, n_a = len(y), len(Xa)
        blk_m = load_col_block_no_ya(Path(a.m7_dir) / f"{c}.npz", GM.PRIMARY_NAMES, n_s, n_a)
        blk_k = TC._clip_k3(load_col_block_no_ya(Path(a.k3_dir) / f"{c}.npz", H430_K3, n_s, n_a), y)
        blk_f = TC._clip_f4(load_col_block_no_ya(Path(a.f4_dir) / f"{c}.npz", H433_F4, n_s, n_a), y)
        blk_tab = load_col_block_no_ya(Path(a.tab450_dir) / f"{c}.npz", H450_TAB_COLS, n_s, n_a)
        r4_wc = load_col_block_no_ya(Path(a.wc_r4e8_dir) / f"{c}.npz", ("r4_e8",), n_s, n_a)
        r4_s1 = load_col_block_no_ya(Path(a.s1_r4e8_dir) / f"{c}.npz", ("r4_e8",), n_s, n_a)
        r4_s2 = load_col_block_no_ya(Path(a.s2_r4e8_dir) / f"{c}.npz", ("r4_e8",), n_s, n_a)
        jobs.append(
            (
                c,
                Xs,
                Xa,
                y,
                lab0,
                Path(a.d1_dir) / f"{c}.npz",
                Path(a.d2_dir) / f"{c}.npz",
                blk_m,
                blk_k,
                blk_f,
                blk_tab,
                r4_wc,
                r4_s1,
                r4_s2,
                mem_dir / f"{c}.npz",
            )
        )

    jobs.sort(key=lambda j: -len(j[3]))
    with Pool(processes=min(a.procs, len(jobs))) as pool:
        rows = list(pool.imap_unordered(_ref_1842_worker, jobs, chunksize=1))
    rows.sort(key=lambda r: str(r["cid"]))

    max_ens = max(float(r["max_abs_diff_p_audit_ens_vs_1842_hwm"]) for r in rows)
    max_maud = max(float(r["max_abs_diff_member_audit"]) for r in rows)
    max_foof = max(float(r["max_abs_diff_member_oof_hwm_fast"]) for r in rows)
    diffs_real = [float(r["max_abs_diff_hwm_vs_real_stacker_audit"]) for r in rows]

    if max_ens > 1e-12 or max_maud > 1e-12 or max_foof > 1e-12:
        raise SystemExit(
            f"F2 PARITY FAILURE: max_ens={max_ens:.3e}, max_member_audit={max_maud:.3e}, max_fast_oof={max_foof:.3e}"
        )

    doc = {
        "schema": "jope.h462.f2_comparator_1842_parity_no_ya/v1",
        "timestamp_utc": time.strftime("%FT%TZ", time.gmtime()),
        "host": os.uname().nodename,
        "ya_keys_accessed": 0,
        "ba_keys_accessed": 0,
        "dev91_utility_evaluations": 0,
        "n_contexts_checked": len(rows),
        "d1_d2_lab0_partition_match_count": len(rows),
        "valley_mode": "v12_avg",
        "comparator_rule": "tabcv7_cvrep8_prec48_all",
        "max_abs_diff_p_audit_ens_vs_1842_hwm": max_ens,
        "max_abs_diff_member_audit_vs_1842_hwm": max_maud,
        "max_abs_diff_member_oof_hwm_fast_vs_1842_hwm": max_foof,
        "hwm_vs_real_stacker_audit_weight_sensitivity": {
            "median_max_abs_diff": float(np.median(diffs_real)),
            "max_max_abs_diff": float(np.max(diffs_real)),
        },
        "comparator_members_dir": str(mem_dir),
        "comparator_receipt_json": str(mem_rcpt),
        "comparator_receipt_sha256": _sha256_file(mem_rcpt),
        "elapsed_s": round(time.monotonic() - t0, 2),
        "per_context": rows,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(out_json), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o444)
    with os.fdopen(fd, "w") as wf:
        wf.write(json.dumps(doc, indent=2, sort_keys=True) + "\n")

    print(
        json.dumps(
            {
                "status": "OK",
                "n_contexts_checked": len(rows),
                "ya_keys_accessed": 0,
                "max_abs_diff_p_audit_ens_vs_1842_hwm": max_ens,
                "max_abs_diff_member_audit_vs_1842_hwm": max_maud,
                "max_abs_diff_member_oof_hwm_fast_vs_1842_hwm": max_foof,
                "comparator_receipt_sha256": doc["comparator_receipt_sha256"],
                "f2_receipt_json": str(out_json),
                "f2_receipt_sha256": _sha256_file(out_json),
                "elapsed_s": doc["elapsed_s"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
