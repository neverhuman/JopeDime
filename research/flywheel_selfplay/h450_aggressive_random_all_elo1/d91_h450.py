#!/usr/bin/env python3
"""H450 quarantined random-split combined-f64 Stage-V bank builder (`d91_h450.py`).

Runs strictly inside `sandbox_h450_quarantine.sh` (`--network none` `h317-empty:v1` on `xbabe0`),
writing ONLY under `DATA = /home/ubuntu/jope-data/h450-quarantine`.
Requires separate owner receipt `/home/ubuntu/jope_plan/ns/receipts/H450_UNFAIR_LOCK_sha256.txt`.

Commands:
  lock --spec SPEC.json   prelock: writes create-once `DATA/stage_v/LOCK.json`; prints sha256. Reads 0 d91 bytes.
  unlock                  one-shot `C.unlock_stage_v()`.
  build_banks             builds create-once in `float64`:
                            - `DATA/cids91.json`
                            - `DATA/bank_d91_search_f64.npz` (`float64`, `audit_targets_written = False`, for legal test-time eval)
                            - `DATA/banks/everything_combined_f64_h450_bank.npz` (`3,525` `float64` tables where `train U test`
                              is merged FIRST and CSPRNG row-shuffled via `os.urandom(32)` BEFORE serialization; verifies
                              `exact_split_matches == 0` across all 91 `dev91` contexts).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import deploy_h416 as DH  # noqa: E402  (installs H317 guard v2)

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import secrets  # noqa: E402
import time  # noqa: E402
from collections import Counter  # noqa: E402

import numpy as np  # noqa: E402

DP15, K, C = DH.DP15, DH.K, DH.C

WAVE = "H450"
ENTRANT = "h450.AGGRESSIVE_RANDOM_ALL_CHAMPION"
DATA = Path(os.environ.get("H450_D91", "/home/ubuntu/jope-data/h450-quarantine"))
C.DATA_ROOT = DATA
C.LOCK_RECEIPT = Path("/home/ubuntu/jope_plan/ns/receipts/H450_UNFAIR_LOCK_sha256.txt")
LOCK_PATH = DATA / "stage_v" / "LOCK.json"
VIEWS = DATA / "views-search"
H335D = Path("/home/ubuntu/jope-data/h335-cl")
WORKER = H335D / "bin" / "bootstrap.s1main.superset-2a848c8f8085"
WORKER_SHA256 = "2a848c8f8085a39210c9714b3d3b80448fea66afecb54a951345f2926c26d7a2"
SEEDS = tuple(K.SEEDS_V)
DESIGN_FILE = HERE / "DESIGN.json"
FROZEN_FILE = HERE / "FROZEN_FAIR_CHAMPION.json"
A2_CELLS_FILE = HERE / "a2_cells.json"
DESIGN_NAME = "H450.LOG_RMSE_ROUTER"
N_CONTEXTS, N_FAMILIES = 91, 19

PREP_MANIFESTS = (
    DATA / "h335-prep" / "x0_prep_20261105T005754Z.tables.sha256",
    DATA / "h335-prep" / "x0_prep_20261105T005754Z.root.sha256",
)
INPUT_MANIFESTS = (
    DATA / "inputs" / "h450_inputs.sha256",
    DATA / "inputs" / "board_post_h448.sha256",
)
CODE_FILES = DH.code_files() + (
    "gp_h415.py",
    "train_h415.py",
    "gp_h416.py",
    "train_h416.py",
    "DESIGN.json",
    "a2_cells.json",
    "cache_h422.py",
    "keys_core.py",
    "formula_core.py",
    "guard_matern_core.py",
    "icl_model.py",
    "icl_prior.py",
    "icl_prior_v3s.py",
    "icl_inject.py",
    "icl_diag.py",
    "icl_resid.py",
    "struct_cols.py",
    "eval_cw5_fitsel.py",
    "cols_r4_e8.py",
    "FROZEN_FAIR_CHAMPION.json",
    "PREREG_H450.md",
    "gates_h450_quarantine.sh",
    "icl_model_h450.py",
    "relational_struct_h450.py",
    "robust_resid_h450.py",
    "build_combined_f64_bank_h450.py",
    "d91_h450.py",
    "sandbox_h450_quarantine.sh",
    "train_aggressive_random_h450.py",
    "cols_h450_gpu.py",
    "cols_tab_h450.py",
    "stack_h450.py",
    "analyze_h450.py",
    "token_compute_ledger_h450.py",
    "selftest_h450.py",
    "gpu_window_x2_x3.sh",
    "stage_h450.sh",
    "run_h450.sh",
)


def _sha(p: Path) -> str:
    return C.sha256_file(Path(p))


def _clean_f64(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, np.float64).copy()
    m[~np.isfinite(m)] = np.nan
    return m


def _write_once(p: Path, obj: dict) -> None:
    fd = os.open(p, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o444)
    try:
        os.write(fd, (json.dumps(obj, indent=1, sort_keys=True) + "\n").encode())
    finally:
        os.close(fd)


def _inputs_ok() -> dict:
    lk = json.loads(LOCK_PATH.read_text())
    for p, sha in lk["inputs_sha256"].items():
        if _sha(Path(p)) != sha:
            raise SystemExit(f"input changed after LOCK: {p}")
    if sorted(lk["inputs_sha256"]) != sorted(str(p) for p in PREP_MANIFESTS + INPUT_MANIFESTS):
        raise SystemExit("LOCK inputs differ from driver list")
    u_man = DATA / "inputs" / "h450_inputs.sha256"
    for line in u_man.read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            h, rel = line.split(None, 1)
            target = (DATA / "inputs" / rel.strip()) if not rel.strip().startswith("/") else Path(rel.strip())
            if _sha(target) != h:
                raise SystemExit(f"h450 input asset sha mismatch: {target}")
    out = {}
    for p in PREP_MANIFESTS:
        for line in Path(p).read_text().splitlines():
            if line.strip():
                h, name = line.split(None, 1)
                out[Path(name.strip().lstrip("*")).name] = h
    return out


def cmd_lock(a) -> None:
    if LOCK_PATH.exists():
        raise SystemExit(f"{LOCK_PATH} exists (create-once)")
    if C._STATE["stage_v"]:
        raise SystemExit("lock runs before Stage V unlock")
    spec = json.loads(Path(a.spec).read_text())
    for k in ("prereg_commit", "stage_commit", "h448_closeout_commit"):
        if not isinstance(spec.get(k), str) or len(spec[k]) != 40:
            raise SystemExit(f"spec needs a full {k}")
    if _sha(WORKER) != WORKER_SHA256:
        raise SystemExit("superset worker sha differs")
    code = K.code_manifest(CODE_FILES)
    code[str(WORKER)] = WORKER_SHA256
    inputs = {str(p): _sha(p) for p in PREP_MANIFESTS + INPUT_MANIFESTS}
    lk = {
        "schema": "jope.h450.quarantined_lock/v1",
        "wave": WAVE,
        "quarantined_in_corpus_random_split_diagnostic": True,
        "canonical_promotion_forbidden": True,
        "entrant_primary": ENTRANT,
        "design": DESIGN_NAME,
        "design_sha256": _sha(DESIGN_FILE),
        "frozen_champion_sha256": _sha(FROZEN_FILE),
        "a2_cells_sha256": _sha(A2_CELLS_FILE),
        "seeds": list(SEEDS),
        "code_sha256": code,
        "inputs_sha256": inputs,
        "prereg_commit": spec["prereg_commit"],
        "stage_commit": spec["stage_commit"],
        "h448_closeout_commit": spec["h448_closeout_commit"],
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    _write_once(LOCK_PATH, lk)
    print(_sha(LOCK_PATH))


def cmd_unlock(_a) -> None:
    C.unlock_stage_v()
    print(json.dumps(C.guard_report()))


def cmd_build_banks(_a) -> None:
    C.resume_stage_v()
    man = _inputs_ok()
    t0 = time.time()

    banks_dir = DATA / "banks"
    banks_dir.mkdir(parents=True, exist_ok=True)
    cids_p = DATA / "cids91.json"
    eval_bank_p = DATA / "bank_d91_search_f64.npz"
    all_bank_p = banks_dir / "everything_combined_f64_h450_bank.npz"
    for p in (cids_p, eval_bank_p, all_bank_p):
        if p.exists():
            raise SystemExit(f"{p} already exists (create-once)")

    # 1. Load pre-combined & CSPRNG-shuffled float64 non-d91 + d91_cousin_s3 bank (3,434 tables)
    non_p = DATA / "inputs" / "non_d91_plus_cousins_f64.npz"
    base_tables: list[tuple[str, str, str, np.ndarray]] = []
    by_cid = set()
    with np.load(non_p, allow_pickle=False) as zn:
        d_al = zn["data"].astype(np.float64, copy=False)
        sh_al = zn["shape"].astype(np.int64)
        of_al = zn["offset"].astype(np.int64)
        cids_al = [str(x) for x in zn["cids"].tolist()]
        fams_al = [str(x) for x in zn["family"].tolist()]
        cohs_al = [str(x) for x in zn["cohort"].tolist()]
        for i, c in enumerate(cids_al):
            r, w = int(sh_al[i, 0]), int(sh_al[i, 1])
            o = int(of_al[i])
            mat64 = np.array(d_al[o : o + r * w], np.float64, copy=True).reshape(r, w)
            base_tables.append((c, fams_al[i], cohs_al[i], mat64))
            by_cid.add(c)
        base_rows = int(sh_al[:, 0].sum())
        base_cells = int((sh_al[:, 0] * sh_al[:, 1]).sum())

    # 2. Read all 91 d91 envelopes in exact float64, verify combined train U test row count == n_search + n_audit,
    #    apply cryptographic row permutation BEFORE saving, and verify 0 exact split matches!
    rows = C.dev_rows_for_stage_v()
    d91_cids = [r["context_sha256"] for r in rows]
    d91_fam = {r["context_sha256"]: r["family_id"] for r in rows}
    if len(d91_cids) != N_CONTEXTS or len(set(d91_fam.values())) != N_FAMILIES:
        raise SystemExit("d91 is not 91 contexts / 19 families")
    if set(d91_cids) & by_cid:
        raise SystemExit("unexpected cid overlap between non_d91_plus_cousins_f64 and d91")

    chk = {}
    S_list, AU_list, labs_list, ys_list, fams_list, ya_sha_list, y_sha_list = [], [], [], [], [], [], []
    d91_shuf_tables = []
    exact_split_matches = 0
    max_jaccard = 0.0

    for c in d91_cids:
        t_sha, v_sha = _sha(C.table_path(c)), _sha(VIEWS / f"{c}.npz")
        chk[c] = {"table_sha256": t_sha, "views_sha256": v_sha}
        if man.get(f"{c}.jtt1") != t_sha or man.get(f"{c}.npz") != v_sha:
            raise SystemExit(f"{c[:12]}: d91 table / SEARCH view sha not in H335 prep manifests")

        env = C.read_jtt1(C.table_path(c))
        full_mat64 = _clean_f64(env.mat)
        search_env, _audit_env, m_aud = C.search_and_audit_envelopes(c, env)
        n_s = int(search_env.mat.shape[0])
        n_a = int(np.sum(m_aud))
        if full_mat64.shape[0] != n_s + n_a:
            raise SystemExit(f"{c[:12]}: full_mat rows {full_mat64.shape[0]} != n_search ({n_s}) + n_audit ({n_a})")

        # True search row indices in the original envelope
        search_orig_set = set(np.flatnonzero(~m_aud).tolist())

        # Irreversible CSPRNG permutation of combined `train U test` rows BEFORE serialization
        perm = None
        for _attempt in range(8):
            csprng = np.random.default_rng(np.frombuffer(secrets.token_bytes(32), dtype=np.uint32))
            cand_perm = csprng.permutation(full_mat64.shape[0])
            pref_set = set(cand_perm[:n_s].tolist())
            if pref_set != search_orig_set:
                perm = cand_perm
                inter = len(pref_set & search_orig_set)
                uni = len(pref_set | search_orig_set)
                jac = float(inter) / float(max(uni, 1))
                max_jaccard = max(max_jaccard, jac)
                break
        if perm is None:
            exact_split_matches += 1
            raise SystemExit(f"{c[:12]}: CSPRNG permutation unexpectedly matched original search split")

        shuf_mat64 = np.ascontiguousarray(full_mat64[perm], dtype=np.float64)
        d91_shuf_tables.append((c, d91_fam[c], "D91-COMBINED-SHUFFLED", shuf_mat64))

        # Legal test-time evaluation bank in exact float64 (`audit_targets_written = False`!)
        y = np.asarray(search_env.y, np.float64)
        ya = np.asarray(env.y, np.float64)[m_aud]
        s_mat64 = _clean_f64(search_env.mat)
        xa_mat64 = _clean_f64(np.asarray(env.mat, np.float64)[m_aud][:, :-1])

        S_list.append(s_mat64)
        AU_list.append(xa_mat64)
        labs_list.append(DP15.kfold_labels(c, y).astype(np.int8))
        ys_list.append(y)
        fams_list.append(d91_fam[c])
        ya_sha_list.append(hashlib.sha256(np.ascontiguousarray(ya).tobytes()).hexdigest())
        y_sha_list.append(hashlib.sha256(np.ascontiguousarray(y).tobytes()).hexdigest())

    _write_once(cids_p, {
        "schema": "jope.h450.cids91/v1",
        "n": len(d91_cids),
        "families": len(set(d91_fam.values())),
        "cids": d91_cids,
        "family_of": d91_fam,
        "inputs": chk,
        "lock_sha256": _sha(LOCK_PATH),
        "guard": C.guard_report(),
    })

    np.savez(
        eval_bank_p,
        search=np.concatenate([t.ravel() for t in S_list]).astype(np.float64),
        s_shape=np.array([t.shape for t in S_list], np.int64),
        audit=np.concatenate([t.ravel() for t in AU_list]).astype(np.float64),
        a_shape=np.array([t.shape for t in AU_list], np.int64),
        lab=np.concatenate(labs_list),
        y64=np.concatenate(ys_list),
        cids=np.array(d91_cids),
        family=np.array(fams_list),
        cohort=np.array(["D91"] * len(d91_cids)),
        ya_sha256=np.array(ya_sha_list),
        y_sha256=np.array(y_sha_list),
        dtype_name=np.array("float64"),
    )
    os.chmod(eval_bank_p, 0o444)
    _write_once(Path(str(eval_bank_p) + ".json"), {
        "schema": "jope.h450.bank_d91_search_f64/v1",
        "dtype": "float64",
        "sha256": _sha(eval_bank_p),
        "audit_targets_written": False,
        "lock_sha256": _sha(LOCK_PATH),
    })

    all_tables = sorted(base_tables + d91_shuf_tables, key=lambda t: t[0])
    shapes = np.array([t[3].shape for t in all_tables], np.int64)
    offs = np.concatenate([[0], np.cumsum(shapes[:, 0] * shapes[:, 1])[:-1]]).astype(np.int64)
    data64 = np.concatenate([t[3].ravel() for t in all_tables]).astype(np.float64)
    cids_arr = np.array([t[0] for t in all_tables])
    fams_arr = np.array([t[1] for t in all_tables])
    cohs_arr = np.array([t[2] for t in all_tables])
    m_sha = hashlib.sha256(f"h450_everything_combined_f64:{len(all_tables)}".encode()).hexdigest()

    np.savez(
        all_bank_p,
        data=data64,
        shape=shapes,
        offset=offs,
        cids=cids_arr,
        family=fams_arr,
        cohort=cohs_arr,
        dtype_name=np.array("float64"),
        manifest_sha256=np.array(m_sha),
    )
    os.chmod(all_bank_p, 0o444)

    d91_shapes = np.array([t[3].shape for t in d91_shuf_tables], np.int64)
    d91_rows = int(d91_shapes[:, 0].sum())
    d91_cells = int((d91_shapes[:, 0] * d91_shapes[:, 1]).sum())
    manifest = {
        "schema": "jope.h450.banks_manifest_f64/v1",
        "quarantined_in_corpus_random_split_diagnostic": True,
        "canonical_promotion_forbidden": True,
        "dtype": "float64",
        "combined_train_union_test_before_shuffle": True,
        "exact_split_matches_d91": exact_split_matches,
        "max_prefix_jaccard_vs_d91_search": round(max_jaccard, 6),
        "non_d91_plus_cousins_f64": {
            "file": str(non_p),
            "sha256": _sha(non_p),
            "tables": len(base_tables),
            "total_rows": base_rows,
            "total_cells": base_cells,
        },
        "d91_combined_shuffled_91": {
            "tables": len(d91_shuf_tables),
            "total_rows": d91_rows,
            "total_cells": d91_cells,
        },
        "everything_combined_f64": {
            "file": str(all_bank_p),
            "sha256": _sha(all_bank_p),
            "tables": len(all_tables),
            "total_rows": int(shapes[:, 0].sum()),
            "total_cells": int((shapes[:, 0] * shapes[:, 1]).sum()),
            "cohorts": dict(Counter(cohs_arr.tolist())),
        },
        "eval_bank_d91_search_f64": {
            "file": str(eval_bank_p),
            "sha256": _sha(eval_bank_p),
            "audit_targets_written": False,
        },
        "lock_sha256": _sha(LOCK_PATH),
        "guard": C.guard_report(),
        "wall_s": round(time.time() - t0, 2),
    }
    _write_once(banks_dir / "banks_manifest_f64.json", manifest)
    print(json.dumps(manifest, sort_keys=True))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("lock")
    p.add_argument("--spec", required=True)
    sub.add_parser("unlock")
    sub.add_parser("build_banks")
    a = ap.parse_args()
    {"lock": cmd_lock, "unlock": cmd_unlock, "build_banks": cmd_build_banks}[a.cmd](a)


if __name__ == "__main__":
    main()
