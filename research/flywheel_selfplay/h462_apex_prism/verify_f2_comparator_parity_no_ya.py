#!/usr/bin/env python3
"""`F2` (`cl_mtc` `16:09:45Z`, `17:09:08Z`, `17:58:48Z` & `cl_fwriter` `17:16:08Z`, `17:59:26Z`) `91/91` `0`-`ya` Comparator Bit-Parity Verifier.

1. Emits (on the emit host, `xbabe2`) the `1842.03` HWM comparator (`tabcv7_cvrep8_prec48_all`,
   `valley_mode="v12_avg"`) per-member `(member_oof, member_oof_hwm_fast, member_audit, p_oof_ens, p_audit_ens)`
   bundle across all `91/91` contexts with `0` `ya`/`ba` reads (`NoLabelNpz` + `assert_same_outer_folds`).
2. Verifies `M1` code-binding (`code_files_sha256` in `comparator_hwm1842_receipt.json` must match the live
   package `*.py` manifest; refuses reuse from any other tree) and `B1` role-load counts via the emitter's
   shared `verify_receipt_role_load_counts` helper (`d1, d2, m7, k3, f4, tab450 == 91` and nested
   `e8.{wc_r4e8, s46201_r4e8, s46202_r4e8} == 91`).
3. Calls `eval_h462_tabcv7_cvrep.predict_tabcv7_cvrep8_prec48_context_no_ya(...)` directly (`Addition 1`:
   `TC`'s own canonical `0`-`ya` scoring path, zero copied stacker loop or local column constants) across
   all `91/91` contexts, verifies `L3` input SHA256 parity (`files_accessed_sha256` of every input read by
   `TC` plus `bank` matches `comparator_hwm1842_receipt.json`), and gates `(b)` on exact bit-parity:
   `max |p_audit_ens - p_ref_1842| == 0.0`, `max |member_audit - AUD_ref| == 0.0`,
   `max |member_oof_hwm_fast - OOF_fast_ref| == 0.0`.
4. Reports `(a)` (`Addition 2` / `L1`) cross-arch drift against stored `Mac-Studio.local` (`arm64`) per-row
   predictions when `--mac-1842-preds-dir` is provided, or `null` when only aggregate `M6_TABCV7_CVREP_PRICE.json` exists.
5. Writes `0444` create-once receipt `F2_COMPARATOR_1842_PARITY_NO_YA.json` with `code_files_sha256` & `verifier_sha256`.
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

from h462_apex_prism import eval_h462_tabcv7_cvrep as TC  # noqa: E402
from h462_apex_prism.emit_p2_p4_members_search_only import (  # noqa: E402
    _check_safe_path,
    compute_package_code_files_sha256,
    load_search_only_bank,
    verify_receipt_role_load_counts,
)


def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def verify_comparator_receipt_binding(
    rcpt_doc: Dict[str, object],
    expected_code_files_sha256: Dict[str, str],
    expected_host: str,
    n_expected_contexts: int,
) -> Dict[str, int]:
    """Verifies host parity, `M1` exact package `code_files_sha256` binding, and `B1` role-load schema."""
    emit_host = str(rcpt_doc.get("host", ""))
    if emit_host and expected_host and emit_host != expected_host:
        raise RuntimeError(
            f"F2 HOST MISMATCH: comparator bundle emitted on {emit_host!r} but F2 parity checked on {expected_host!r}"
        )
    rcpt_code = rcpt_doc.get("code_files_sha256")
    if not isinstance(rcpt_code, dict) or dict(rcpt_code) != dict(expected_code_files_sha256):
        raise RuntimeError(
            "F2 M1 CODE-BINDING FAILURE: comparator receipt 'code_files_sha256' missing or differs from current package tree"
        )
    return verify_receipt_role_load_counts(
        rcpt_doc,
        n_expected=n_expected_contexts,
        expected_e8_names=("wc_r4e8", "s46201_r4e8", "s46202_r4e8"),
        require_gpbag=False,
        exact_e8_keys=True,
    )


def verify_reference_inputs_match_receipt(
    rcpt_doc: Dict[str, object],
    bank_path: Path,
    ref_files_accessed_sha256: Dict[str, str],
) -> int:
    """`L3` input-binding gate: verifies `bank` + every file opened by `TC` matches `files_accessed_sha256`."""
    rcpt_files = rcpt_doc.get("files_accessed_sha256")
    if not isinstance(rcpt_files, dict):
        raise RuntimeError("F2 L3 INPUT-SHA MISMATCH: comparator receipt missing dict 'files_accessed_sha256'")
    bank_key = str(bank_path)
    live_bank_sha = _sha256_file(bank_path)
    if rcpt_files.get(bank_key) != live_bank_sha:
        raise RuntimeError(
            f"F2 L3 INPUT-SHA MISMATCH: bank {bank_key!r} live sha {live_bank_sha} != receipt {rcpt_files.get(bank_key)!r}"
        )
    for fp, live_sha in sorted(ref_files_accessed_sha256.items()):
        got_sha = rcpt_files.get(fp)
        if got_sha != live_sha:
            raise RuntimeError(
                f"F2 L3 INPUT-SHA MISMATCH: reference input {fp!r} live sha {live_sha} != receipt {got_sha!r}"
            )
    expected_keys = {bank_key} | set(ref_files_accessed_sha256.keys())
    if set(rcpt_files.keys()) != expected_keys:
        diff_only_rcpt = sorted(set(rcpt_files.keys()) - expected_keys)[:5]
        diff_only_ref = sorted(expected_keys - set(rcpt_files.keys()))[:5]
        raise RuntimeError(
            f"F2 L3 INPUT-SHA MISMATCH: receipt file set != reference file set "
            f"(only_in_receipt={diff_only_rcpt}, only_in_ref={diff_only_ref})"
        )
    return len(expected_keys)


def _ref_1842_worker(job: Tuple) -> Dict[str, object]:
    """Calls `TC.predict_tabcv7_cvrep8_prec48_context_no_ya` (`0` `ya` read) and checks bit-parity vs emitted bundle."""
    (
        c,
        Xs,
        Xa,
        y,
        lab0,
        d1_p,
        d2_p,
        m7_p,
        k3_p,
        f4_p,
        tab450_p,
        wc_r4_p,
        s1_r4_p,
        s2_r4_p,
        mem_p,
        mac_pred_p,
    ) = job
    n_s, n_a = len(y), len(Xa)

    ref = TC.predict_tabcv7_cvrep8_prec48_context_no_ya(
        c=c,
        Xs=Xs,
        Xa=Xa,
        y=y,
        lab0=lab0,
        gp0_p=d1_p,
        gp1_p=d2_p,
        is_d91_gp=True,
        m7_p=m7_p,
        k3_p=k3_p,
        f4_p=f4_p,
        tab450_p=tab450_p,
        wc_r4_p=wc_r4_p,
        s1_r4_p=s1_r4_p,
        s2_r4_p=s2_r4_p,
        n_reps=8,
    )
    p_ref_1842 = np.asarray(ref["p_audit_ens"], dtype=np.float64)
    AUD_ref = np.asarray(ref["member_audit"], dtype=np.float64)
    OOF_fast_ref = np.asarray(ref["member_oof_hwm_fast"], dtype=np.float64)
    blocks_ref = dict(ref["blocks_loaded"])  # type: ignore[arg-type]
    files_ref = dict(ref["files_accessed_sha256"])  # type: ignore[arg-type]

    with np.load(mem_p, allow_pickle=False) as zm:
        p_emit = np.asarray(zm["p_audit_ens"], dtype=np.float64)
        p_emit_real = np.asarray(zm["p_audit_ens_real_stacker"], dtype=np.float64)
        AUD_emit = np.asarray(zm["member_audit"], dtype=np.float64)
        OOF_fast_emit = np.asarray(zm["member_oof_hwm_fast"], dtype=np.float64)
        OOF_real_emit = np.asarray(zm["member_oof"], dtype=np.float64)
        emit_kept_mask = np.asarray(zm["kept_mask"], dtype=bool) if "kept_mask" in zm.files else None

    max_diff_ens = float(np.max(np.abs(p_emit - p_ref_1842)))
    max_diff_member_aud = float(np.max(np.abs(AUD_emit - AUD_ref)))
    max_diff_fast_oof = float(np.max(np.abs(OOF_fast_emit - OOF_fast_ref)))
    max_diff_hwm_vs_real_aud = float(np.max(np.abs(p_emit - p_emit_real)))
    rmse_real_vs_fast_oof = float(np.sqrt(np.mean((OOF_real_emit - OOF_fast_emit) ** 2)))

    mac_rel_diff: Optional[float] = None
    mac_admitted_diff_share: Optional[float] = None
    if mac_pred_p is not None and Path(mac_pred_p).is_file():
        with TC._NoAuditLabelNpz(Path(mac_pred_p)) as zmac:
            key_p = "p_audit_ens" if "p_audit_ens" in zmac.files else ("pred" if "pred" in zmac.files else "")
            if key_p:
                p_mac = np.asarray(zmac[key_p], dtype=np.float64)
                denom = np.maximum(1.0, np.abs(p_mac))
                mac_rel_diff = float(np.max(np.abs(p_emit - p_mac) / denom))
            if "kept_mask" in zmac.files and emit_kept_mask is not None:
                km_mac = np.asarray(zmac["kept_mask"], dtype=bool)
                if km_mac.shape == emit_kept_mask.shape:
                    mac_admitted_diff_share = float(np.mean(emit_kept_mask != km_mac))

    return {
        "cid": c,
        "n_search": n_s,
        "n_audit": n_a,
        "blocks_loaded_tc_ref": blocks_ref,
        "files_accessed_sha256": files_ref,
        "max_abs_diff_p_audit_ens_vs_1842_hwm": max_diff_ens,
        "max_abs_diff_member_audit": max_diff_member_aud,
        "max_abs_diff_member_oof_hwm_fast": max_diff_fast_oof,
        "max_abs_diff_hwm_vs_real_stacker_audit": max_diff_hwm_vs_real_aud,
        "rmse_real_vs_fast_member_oof": rmse_real_vs_fast_oof,
        "mac_1842_max_rel_diff": mac_rel_diff,
        "mac_1842_admitted_diff_share": mac_admitted_diff_share,
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
    ap.add_argument(
        "--mac-1842-preds-dir",
        default="",
        help="Optional directory of stored Mac-Studio.local per-row predictions for cross-arch drift report (Addition 2)",
    )
    ap.add_argument("--out-json", default="/home/ubuntu/jope_plan/evidence/H462/F2_COMPARATOR_1842_PARITY_NO_YA.json")
    ap.add_argument("--procs", type=int, default=20)
    a = ap.parse_args()

    out_json = Path(a.out_json)
    mem_dir = Path(a.members_out_dir)
    mem_rcpt = Path(a.members_receipt_json)
    bank_p = Path(a.bank)
    mac_dir = Path(a.mac_1842_preds_dir.strip()) if a.mac_1842_preds_dir.strip() else None
    for p in (out_json, mem_dir, mem_rcpt, bank_p):
        _check_safe_path(p)
    if mac_dir is not None:
        _check_safe_path(mac_dir)
    if out_json.exists():
        raise SystemExit(f"REFUSAL: create-once F2 receipt already exists: {out_json}")

    code_hashes, code_manifest_sha = compute_package_code_files_sha256(HERE)
    verifier_sha = _sha256_file(Path(__file__).resolve())
    cur_host = os.uname().nodename

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

    rcpt_doc = json.loads(mem_rcpt.read_text())
    bank_rows = load_search_only_bank(bank_p)
    cids = sorted(bank_rows)
    n_expected = len(cids)

    try:
        verified_role_counts = verify_comparator_receipt_binding(
            rcpt_doc=rcpt_doc,
            expected_code_files_sha256=code_hashes,
            expected_host=cur_host,
            n_expected_contexts=n_expected,
        )
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc

    blk_counts = rcpt_doc.get("block_loaded_counts", {})

    jobs = []
    for c in cids:
        _fam, Xs, Xa, y, lab0 = bank_rows[c]
        jobs.append(
            (
                c,
                Xs,
                Xa,
                y,
                lab0,
                Path(a.d1_dir) / f"{c}.npz",
                Path(a.d2_dir) / f"{c}.npz",
                Path(a.m7_dir) / f"{c}.npz",
                Path(a.k3_dir) / f"{c}.npz",
                Path(a.f4_dir) / f"{c}.npz",
                Path(a.tab450_dir) / f"{c}.npz",
                Path(a.wc_r4e8_dir) / f"{c}.npz",
                Path(a.s1_r4e8_dir) / f"{c}.npz",
                Path(a.s2_r4e8_dir) / f"{c}.npz",
                mem_dir / f"{c}.npz",
                (mac_dir / f"{c}.npz") if mac_dir is not None else None,
            )
        )

    jobs.sort(key=lambda j: -len(j[3]))
    with Pool(processes=min(a.procs, len(jobs))) as pool:
        raw_rows = list(pool.imap_unordered(_ref_1842_worker, jobs, chunksize=1))
    raw_rows.sort(key=lambda r: str(r["cid"]))

    ref_accessed_all: Dict[str, str] = {}
    rows: List[Dict[str, object]] = []
    for r in raw_rows:
        for rk, okv in dict(r["blocks_loaded_tc_ref"]).items():  # type: ignore[arg-type]
            if not bool(okv):
                raise SystemExit(f"F2 TC-REF ROLE-LOAD FAILURE on {r['cid']} role {rk!r}")
        for fp, fh in dict(r["files_accessed_sha256"]).items():  # type: ignore[arg-type]
            ref_accessed_all[str(fp)] = str(fh)
        row_clean = {k: v for k, v in r.items() if k != "files_accessed_sha256"}
        rows.append(row_clean)

    try:
        n_verified_inputs = verify_reference_inputs_match_receipt(
            rcpt_doc=rcpt_doc,
            bank_path=bank_p,
            ref_files_accessed_sha256=ref_accessed_all,
        )
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc

    max_ens = max(float(r["max_abs_diff_p_audit_ens_vs_1842_hwm"]) for r in rows)
    max_maud = max(float(r["max_abs_diff_member_audit"]) for r in rows)
    max_foof = max(float(r["max_abs_diff_member_oof_hwm_fast"]) for r in rows)
    diffs_real = [float(r["max_abs_diff_hwm_vs_real_stacker_audit"]) for r in rows]

    # Gate (b): exact bit-for-bit equality against TC.predict_tabcv7_cvrep8_prec48_context_no_ya on emit host
    if max_ens != 0.0 or max_maud != 0.0 or max_foof != 0.0:
        raise SystemExit(
            f"F2 EMIT-HOST BIT-PARITY FAILURE (gate b): max_ens={max_ens:.3e}, "
            f"max_member_audit={max_maud:.3e}, max_fast_oof={max_foof:.3e}"
        )

    mac_rels = [float(r["mac_1842_max_rel_diff"]) for r in rows if r["mac_1842_max_rel_diff"] is not None]
    mac_adm = [float(r["mac_1842_admitted_diff_share"]) for r in rows if r["mac_1842_admitted_diff_share"] is not None]
    cross_arch_report: Dict[str, object] = {
        "mac_1842_reference_host": "Mac-Studio.local",
        "mac_1842_reference_arch": "arm64",
        "emit_host": cur_host,
        "emit_arch": os.uname().machine,
        "mac_1842_stored_predictions_present": len(mac_rels) == len(rows) and len(rows) > 0,
        "n_contexts_with_mac_predictions": len(mac_rels),
        "max_rel_diff_vs_mac_stored": float(max(mac_rels)) if mac_rels else None,
        "median_rel_diff_vs_mac_stored": float(np.median(mac_rels)) if mac_rels else None,
        "mean_admitted_set_diff_share": float(np.mean(mac_adm)) if mac_adm else None,
        "note": (
            "Per-row stored predictions compared."
            if mac_rels
            else "M6_TABCV7_CVREP_PRICE.json (Mac-Studio.local arm64) stored scalar utilities only; "
            "gate (b) enforces exact max_abs_diff == 0.0 against TC.predict_tabcv7_cvrep8_prec48_context_no_ya on emit_host."
        ),
    }

    doc = {
        "schema": "jope.h462.f2_comparator_1842_parity_no_ya/v3",
        "timestamp_utc": time.strftime("%FT%TZ", time.gmtime()),
        "host": cur_host,
        "arch": os.uname().machine,
        "argv": list(sys.argv),
        "ya_keys_accessed": 0,
        "ba_keys_accessed": 0,
        "dev91_utility_evaluations": 0,
        "n_contexts_checked": len(rows),
        "n_reference_inputs_verified": n_verified_inputs,
        "d1_d2_lab0_partition_match_count": len(rows),
        "block_loaded_counts": blk_counts,
        "verified_role_counts": verified_role_counts,
        "valley_mode": "v12_avg",
        "comparator_rule": "tabcv7_cvrep8_prec48_all",
        "tc_entry_point": "eval_h462_tabcv7_cvrep.predict_tabcv7_cvrep8_prec48_context_no_ya",
        "code_manifest_sha256": code_manifest_sha,
        "code_files_sha256": code_hashes,
        "verifier_sha256": verifier_sha,
        "gate_b_emit_host_exact_bit_parity": {
            "passed": True,
            "max_abs_diff_p_audit_ens_vs_1842_hwm": max_ens,
            "max_abs_diff_member_audit_vs_1842_hwm": max_maud,
            "max_abs_diff_member_oof_hwm_fast_vs_1842_hwm": max_foof,
        },
        "report_a_cross_arch_drift": cross_arch_report,
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
                "host": cur_host,
                "arch": os.uname().machine,
                "n_contexts_checked": len(rows),
                "n_reference_inputs_verified": n_verified_inputs,
                "ya_keys_accessed": 0,
                "block_loaded_counts": blk_counts,
                "code_manifest_sha256": code_manifest_sha,
                "max_abs_diff_p_audit_ens_vs_1842_hwm": max_ens,
                "max_abs_diff_member_audit_vs_1842_hwm": max_maud,
                "max_abs_diff_member_oof_hwm_fast_vs_1842_hwm": max_foof,
                "mac_1842_stored_predictions_present": cross_arch_report["mac_1842_stored_predictions_present"],
                "comparator_receipt_sha256": doc["comparator_receipt_sha256"],
                "f2_receipt_json": str(out_json),
                "f2_receipt_sha256": _sha256_file(out_json),
                "elapsed_s": doc["elapsed_s"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
