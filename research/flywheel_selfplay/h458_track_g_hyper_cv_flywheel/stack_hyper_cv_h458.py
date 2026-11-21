#!/usr/bin/env python3
"""Wave H458 Track G: Selection-Optimism-Debiased `Train Hyper-CV` Stacker,
`_load_col_block` Sidecar Validator, & `K=5` `skill-xfit5` Row-Stratum Column Assembler
(`stack_hyper_cv_h458.py`).

Strictly `SEARCH`-only during stacking & routing (`0` `y_audit` bytes read in `stack_hyper_cv_context` or
`assemble_skill_xfit5_context_columns`; exact query-permutation equivariance `max|delta_pred(perm(Xa))| == 0.0`!).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

os.environ["H335_GROUP"] = "AGG"
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np

HERE = Path(__file__).resolve().parent
for _cand in reversed((
    str(HERE),
    "/home/ubuntu/h450-stage/pub-r6/code",
    "/home/ubuntu/h450-stage/fleet-r10/code",
)):
    if Path(_cand).is_dir() and _cand not in sys.path:
        sys.path.insert(0, _cand)
if sys.path[0] != str(HERE):
    sys.path.insert(0, str(HERE))

from stack_guard_cv7_h452 import make_stratified_kfold, stack_guard_cv7_context  # noqa: E402
from ultra_gp_cycle_h458 import K5_STRATA_KEY, make_k5_feature_byte_strata  # noqa: E402
from ultra_gp_evolve_h452 import fold_log_utility  # noqa: E402


def load_col_block_with_sidecar(
    block_dir: Path,
    cid: str,
    n_s: int,
    n_a: int,
    col_names: Optional[Sequence[str]] = None,
) -> Optional[Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], Dict]]:
    """Loads `(Q, BAG, REF)` + validated `sidecar.json` metadata in Track S Interface (i) `_load_col_block` layout."""
    npz_p = Path(block_dir) / f"{cid}.npz"
    if not npz_p.is_file():
        return None
    sidecar_p = Path(block_dir) / "sidecar.json"
    sidecar = json.loads(sidecar_p.read_text()) if sidecar_p.is_file() else {}
    names = tuple(col_names) if col_names is not None else tuple(sidecar.get("col_names", ("ugp_tierc",)))
    if not names:
        return None
    with np.load(npz_p, allow_pickle=False) as z:
        assert "ya" not in z.files, f"Forbidden ya key in colblock {npz_p}"
        if "ok" in z.files and not bool(z["ok"]):
            return None
        if not all(
            f"q_{v}" in z.files and f"bag_{v}" in z.files and f"ref_{v}" in z.files
            and len(z[f"q_{v}"]) == n_s and len(z[f"bag_{v}"]) == n_a and len(z[f"ref_{v}"]) == n_a
            for v in names
        ):
            return None
        Q = np.column_stack([np.asarray(z[f"q_{v}"], dtype=np.float64) for v in names])
        BAG = np.column_stack([np.asarray(z[f"bag_{v}"], dtype=np.float64) for v in names])
        REF = np.column_stack([np.asarray(z[f"ref_{v}"], dtype=np.float64) for v in names])
        if not (np.all(np.isfinite(Q)) and np.all(np.isfinite(BAG))):
            return None
        REF_clean = np.column_stack([REF[:, j] if np.all(np.isfinite(REF[:, j])) else BAG[:, j] for j in range(REF.shape[1])])
    return (Q, BAG, REF_clean), sidecar


def debias_search_oof_for_selection_multiplicity(
    Q: np.ndarray,
    ys: np.ndarray,
    multiplicity_m: int,
    kappa: float = 0.35,
) -> np.ndarray:
    """Applies `SEARCH`-only winner's-curse residual inflation when `multiplicity_m > 1` (`M = 1 -> exact identity!`)."""
    m_eff = max(1, int(multiplicity_m))
    if m_eff <= 1:
        return Q
    n_s = max(16, len(ys))
    ymu = float(np.mean(ys))
    inflate = 1.0 + float(kappa) * math.sqrt(math.log(float(m_eff)) / float(n_s))
    # Shrinks OOF slightly toward `mean(ys)` so `M = 3000` un-vaulted search columns cannot hijack simplex weight
    w_keep = 1.0 / inflate
    return w_keep * Q + (1.0 - w_keep) * ymu


def stack_hyper_cv_context(
    cid: str,
    Xs: np.ndarray,
    Xa: np.ndarray,
    ys: np.ndarray,
    lab: np.ndarray,
    Q_gp: Optional[np.ndarray],
    bcols_gp: Optional[List[np.ndarray]],
    ref_gp: Optional[List[Optional[np.ndarray]]],
    base_named_blocks: Sequence[Tuple[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]],
    tierc_ugp_block: Optional[Tuple[str, Tuple[np.ndarray, np.ndarray, np.ndarray], int]] = None,
) -> Tuple[np.ndarray, Dict]:
    """Stacks `base_named_blocks` + optional `(tag, (Q, BAG, REF), multiplicity_m)` under a strict `SEARCH`-only gate.

    - Zero `y_audit` bytes read anywhere!
    - Query-row permutation equivariant: `pred(P @ Xa) == P @ pred(Xa)` to `0.0`!
    """
    pred_base, diag_base = stack_guard_cv7_context(
        cid, Xs, Xa, ys, lab, Q_gp, bcols_gp, ref_gp, list(base_named_blocks), use_cv7_folds=True
    )
    if tierc_ugp_block is None:
        return pred_base, {"tierc_admitted": False, "reason": "no_tierc_block", **diag_base}

    ugp_tag, (Q_u, B_u, R_u), mult_m = tierc_ugp_block
    Q_u_deb = debias_search_oof_for_selection_multiplicity(Q_u, ys, multiplicity_m=mult_m)

    # Gate admission strictly on `SEARCH` `2xCV-7` OOF utility of best Tier-C column vs median base OOF!
    lab7_0 = make_stratified_kfold(cid, ys, k=7, rep=0)
    lab7_1 = make_stratified_kfold(cid, ys, k=7, rep=1)
    best_u_tierc = max(
        0.5 * (fold_log_utility(Q_u_deb[:, j], ys, lab7_0) + fold_log_utility(Q_u_deb[:, j], ys, lab7_1))
        for j in range(Q_u_deb.shape[1])
    )
    base_oofs = []
    if Q_gp is not None and Q_gp.shape[1] > 0:
        for j in range(Q_gp.shape[1]):
            if np.all(np.isfinite(Q_gp[:, j])):
                base_oofs.append(0.5 * (fold_log_utility(Q_gp[:, j], ys, lab7_0) + fold_log_utility(Q_gp[:, j], ys, lab7_1)))
    for _t, (Qb, _Bb, _Rb) in base_named_blocks:
        for j in range(Qb.shape[1]):
            base_oofs.append(0.5 * (fold_log_utility(Qb[:, j], ys, lab7_0) + fold_log_utility(Qb[:, j], ys, lab7_1)))

    best_base_col = max(base_oofs) if base_oofs else -5.0
    # Require Tier-C column to sit within reasonable OOF proximity of the leading base model (`>= best_base - 0.35 u`)
    if best_u_tierc < best_base_col - 0.35:
        return pred_base, {
            "tierc_admitted": False,
            "reason": "below_base_leader_margin",
            "best_u_tierc": round(best_u_tierc, 6),
            "best_u_base_col": round(best_base_col, 6),
            **diag_base,
        }

    aug_blocks = list(base_named_blocks) + [(ugp_tag, (Q_u_deb, B_u, R_u))]
    pred_aug, diag_aug = stack_guard_cv7_context(
        cid, Xs, Xa, ys, lab, Q_gp, bcols_gp, ref_gp, aug_blocks, use_cv7_folds=True
    )
    return pred_aug, {
        "tierc_admitted": True,
        "multiplicity_m": int(mult_m),
        "best_u_tierc": round(best_u_tierc, 6),
        "best_u_base_col": round(best_base_col, 6),
        **diag_aug,
    }


def assemble_skill_xfit5_context_columns(
    cid: str,
    Xs: np.ndarray,
    Xa: np.ndarray,
    fold_blocks: Sequence[Dict[str, np.ndarray]],
    col_names: Sequence[str],
    out_dir: Path,
    pool_sha256: str = "",
) -> Tuple[Path, Dict]:
    """Assembles zero-row-leakage `K=5` `skill-xfit5` `_load_col_block` `.npz` strictly from feature-byte hashes!

    - `fold_blocks[k]` (`k in 0..4`) holds `{q_<name>, bag_<name>, ref_<name>}` produced by model `theta^(-k)`
      trained WITHOUT stratum `k`.
    - Row `i` of `Xs` takes `q_<name>[i]` from `fold_blocks[k_s[i]]`, and row `j` of `Xa` takes `bag_<name>[j]` &
      `ref_<name>[j]` from `fold_blocks[k_a[j]]` — reading ZERO `y_audit` or `y_search` label bytes!
    """
    assert len(fold_blocks) == 5, f"Expected K=5 fold blocks, got {len(fold_blocks)}"
    k_s = make_k5_feature_byte_strata(Xs, k=5)
    k_a = make_k5_feature_byte_strata(Xa, k=5)
    n_s, n_a = len(Xs), len(Xa)

    save_dict: Dict[str, np.ndarray] = {"ok": np.array(True), "strata_s": k_s.astype(np.int8), "strata_a": k_a.astype(np.int8)}
    for v in col_names:
        q_out = np.empty(n_s, dtype=np.float64)
        b_out = np.empty(n_a, dtype=np.float64)
        r_out = np.empty(n_a, dtype=np.float64)
        for fk in range(5):
            ms = k_s == fk
            if np.any(ms):
                q_out[ms] = np.asarray(fold_blocks[fk][f"q_{v}"], dtype=np.float64)[ms]
            ma = k_a == fk
            if np.any(ma):
                b_out[ma] = np.asarray(fold_blocks[fk][f"bag_{v}"], dtype=np.float64)[ma]
                r_out[ma] = np.asarray(fold_blocks[fk][f"ref_{v}"], dtype=np.float64)[ma]
        save_dict[f"q_{v}"] = q_out
        save_dict[f"bag_{v}"] = b_out
        save_dict[f"ref_{v}"] = r_out

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / f"{cid}.npz"
    tmp = out_dir / f".{cid}.{os.getpid()}.tmp.npz"
    np.savez_compressed(tmp, **save_dict)
    os.replace(tmp, dst)

    meta = {
        "cid": cid,
        "closure": "skill-xfit5",
        "cross_fitted": True,
        "hash_spec": f"blake2b-8-le:{K5_STRATA_KEY.decode('ascii')}",
        "pool_sha256": pool_sha256,
        "search_stratum_counts": [int(np.sum(k_s == j)) for j in range(5)],
        "audit_stratum_counts": [int(np.sum(k_a == j)) for j in range(5)],
    }
    return dst, meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--verify-pool-dir", default="/home/ubuntu/jope-data/h458-quarantine/family_pool_v1")
    ap.add_argument("--d91-search-bank", default="/home/ubuntu/jope-data/h450-quarantine/bank_d91_search_f64.npz")
    ap.add_argument("--out-json", default="/home/ubuntu/jope_plan/evidence/H458/trackG/K5_FEATURE_STRATA_AUDIT.json")
    a = ap.parse_args()

    # Emit durable `19`-family `K=5` feature-stratum & `0`-twin-escape verification receipt for Track D & Track S!
    import h335_core as K

    K.C.resume_stage_v()
    fam_map = {r["context_sha256"]: r["family_id"] for r in K.C.dev_rows_for_stage_v()}

    fam_summary: Dict[str, Dict] = {}
    with np.load(a.d91_search_bank, allow_pickle=False) as zb:
        assert "ya" not in zb.files
        cids = [str(x) for x in zb["cids"].tolist()]
        ss, aa = zb["s_shape"], zb["a_shape"]
        so = ao = 0
        fam_mats: Dict[str, List[Tuple[str, np.ndarray, np.ndarray]]] = {}
        for i, cid in enumerate(cids):
            ns, ws = int(ss[i, 0]), int(ss[i, 1])
            na, wa = int(aa[i, 0]), int(aa[i, 1])
            Xs = np.array(zb["search"][so : so + ns * ws], np.float64).reshape(ns, ws)[:, :-1]
            Xa = np.array(zb["audit"][ao : ao + na * wa], np.float64).reshape(na, wa)
            fam_mats.setdefault(fam_map[cid], []).append((cid, Xs, Xa))
            so += ns * ws
            ao += na * wa

    for fam_id, ctx_list in sorted(fam_mats.items()):
        X_all = np.vstack([np.vstack([xs, xa]) for _, xs, xa in ctx_list])
        st = make_k5_feature_byte_strata(X_all, k=5)
        cnts = [int(np.sum(st == j)) for j in range(5)]
        min_train_retain = min(1.0 - (c_k / max(1, len(X_all))) for c_k in cnts)
        fam_summary[fam_id] = {
            "contexts": len(ctx_list),
            "total_search_plus_audit_rows": int(len(X_all)),
            "stratum_counts_0_to_4": cnts,
            "min_fold_train_retention_frac": round(float(min_train_retain), 6),
        }

    pool_test_vec = None
    pool_dir = Path(a.verify_pool_dir)
    if (pool_dir / "meta.json").is_file() and (pool_dir / "data.f64").is_file():
        meta = json.loads((pool_dir / "meta.json").read_text())
        matches = [i for i, f in enumerate(meta.get("families", [])) if "f2893c38" in str(f)]
        if matches:
            idx = matches[0]
            r, w = meta["shapes"][idx]
            off = meta["offsets"][idx]
            mm = np.memmap(pool_dir / meta["data_file"], dtype="<f8", mode="r")
            tab = np.asarray(mm[off : off + r * w], dtype="<f8").reshape(r, w)
            st_f289 = make_k5_feature_byte_strata(tab[:, :-1], k=5)
            pool_test_vec = {
                "family": meta["families"][idx],
                "rows": int(r),
                "feature_cols": int(w - 1),
                "stratum_counts_0_to_4": [int(np.sum(st_f289 == j)) for j in range(5)],
            }

    out_doc = {
        "schema": "jope.h458.k5_feature_strata_audit/v1",
        "hash_spec": f"blake2b-8-le:{K5_STRATA_KEY.decode('ascii')}",
        "audit_labels_read": 0,
        "families_total": len(fam_summary),
        "all_19_families_min_train_retention": round(
            min(v["min_fold_train_retention_frac"] for v in fam_summary.values()), 6
        ),
        "family_pool_v1_f2893c38_test_vector": pool_test_vec,
        "by_family": fam_summary,
        "utc": time.strftime("%FT%TZ", time.gmtime()),
    }
    out_p = Path(a.out_json)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    tmp_p = out_p.with_suffix(".tmp.json")
    tmp_p.write_text(json.dumps(out_doc, indent=2, sort_keys=True) + "\n")
    os.replace(tmp_p, out_p)
    print(json.dumps({k: v for k, v in out_doc.items() if k != "by_family"}, sort_keys=True))


if __name__ == "__main__":
    main()
