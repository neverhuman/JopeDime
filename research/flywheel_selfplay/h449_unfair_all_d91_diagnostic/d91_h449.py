#!/usr/bin/env python3
"""H449 quarantined unfair-d91 driver (`d91_h449.py`).

Runs strictly inside `sandbox_h449_unfair_d91.sh` (`--network none` `h317-empty:v1` on `xbabe0`),
writing ONLY under `DATA = /home/ubuntu/jope-data/h449-unfair-d91`.
Requires separate owner Stage-V receipt `/home/ubuntu/jope_plan/ns/receipts/H449_UNFAIR_LOCK_sha256.txt`.

Commands:
  lock --spec SPEC.json   prelock: writes create-once `DATA/stage_v/LOCK.json`; prints sha256. Reads 0 d91 bytes.
  unlock                  one-shot `C.unlock_stage_v()`.
  build_banks             builds create-once:
                            - `DATA/cids91.json`
                            - `DATA/bank_d91_search.npz` (`audit_targets_written = False`, for test-time inference)
                            - `DATA/banks/everything_all_d91_bank.npz` (1,853 unique complete tables incl. 100% d91)
                            - `DATA/banks/everything_search_d91_bank.npz` (1,853 tables with d91 restricted to 70% SEARCH)
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
import time  # noqa: E402
from collections import Counter  # noqa: E402

import numpy as np  # noqa: E402

DP15, K, C = DH.DP15, DH.K, DH.C

WAVE = "H449"
ENTRANT = "h449.UNFAIR_ALL_D91_STACK"
DATA = Path(os.environ.get("H449_D91", "/home/ubuntu/jope-data/h449-unfair-d91"))
C.DATA_ROOT = DATA
C.LOCK_RECEIPT = Path("/home/ubuntu/jope_plan/ns/receipts/H449_UNFAIR_LOCK_sha256.txt")
LOCK_PATH = DATA / "stage_v" / "LOCK.json"
VIEWS = DATA / "views-search"
H335D = Path("/home/ubuntu/jope-data/h335-cl")
APEX_TABLES = H335D / "apex" / "tables"
WORKER = H335D / "bin" / "bootstrap.s1main.superset-2a848c8f8085"
WORKER_SHA256 = "2a848c8f8085a39210c9714b3d3b80448fea66afecb54a951345f2926c26d7a2"
SEEDS = tuple(K.SEEDS_V)
DESIGN_FILE = HERE / "DESIGN.json"
FROZEN_FILE = HERE / "FROZEN_FAIR_CHAMPION.json"
A2_CELLS_FILE = HERE / "a2_cells.json"
DESIGN_NAME = "BCAR.mix"
N_CONTEXTS, N_FAMILIES = 91, 19

PREP_MANIFESTS = (
    DATA / "h335-prep" / "x0_prep_20261105T005754Z.tables.sha256",
    DATA / "h335-prep" / "x0_prep_20261105T005754Z.root.sha256",
)
INPUT_MANIFESTS = (
    DATA / "inputs" / "d1_pulled.sha256",
    DATA / "inputs" / "d2_pulled.sha256",
    DATA / "inputs" / "board_post_h448.sha256",
    DATA / "inputs" / "unfair_inputs.sha256",
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
    "d91_h449.py",
    "sandbox_h449_unfair_d91.sh",
    "gates_h449_prelock.sh",
    "build_non_d91_complete_bank.py",
    "train_unfair_icl_h449.py",
    "analyze_h449_unfair.py",
    "token_compute_ledger.py",
    "selftest_h449.py",
    "run_h449.sh",
    "stage_h449.sh",
    "PREREG_H449.md",
)


def _sha(p: Path) -> str:
    return C.sha256_file(Path(p))


def _clean(m: np.ndarray) -> np.ndarray:
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
    u_man = DATA / "inputs" / "unfair_inputs.sha256"
    for line in u_man.read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            h, rel = line.split(None, 1)
            target = (DATA / "inputs" / rel.strip()) if not rel.strip().startswith("/") else Path(rel.strip())
            if _sha(target) != h:
                raise SystemExit(f"unfair asset sha mismatch: {target}")
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
        "schema": "jope.h449.unfair_lock/v1",
        "wave": WAVE,
        "quarantined_unfair_diagnostic": True,
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


def _save_flat_bank(path: Path, tables: list[tuple[str, str, str, np.ndarray]], manifest_tag: str) -> dict:
    shapes = np.array([t[3].shape for t in tables], np.int64)
    offs = np.concatenate([[0], np.cumsum(shapes[:, 0] * shapes[:, 1])[:-1]]).astype(np.int64)
    data = np.concatenate([t[3].ravel() for t in tables]).astype(np.float32)
    cids = np.array([t[0] for t in tables])
    fams = np.array([t[1] for t in tables])
    cohorts = np.array([t[2] for t in tables])
    m_sha = hashlib.sha256(f"{manifest_tag}:{len(tables)}".encode()).hexdigest()
    np.savez(
        path,
        data=data,
        shape=shapes,
        offset=offs,
        cids=cids,
        family=fams,
        cohort=cohorts,
        manifest_sha256=np.array(m_sha),
    )
    os.chmod(path, 0o444)
    sha = _sha(path)
    return {
        "file": str(path),
        "sha256": sha,
        "tables": len(tables),
        "total_rows": int(shapes[:, 0].sum()),
        "cohorts": dict(Counter(cohorts.tolist())),
    }


def cmd_build_banks(_a) -> None:
    C.resume_stage_v()
    man = _inputs_ok()
    t0 = time.time()

    banks_dir = DATA / "banks"
    banks_dir.mkdir(parents=True, exist_ok=True)
    cids_p = DATA / "cids91.json"
    eval_bank_p = DATA / "bank_d91_search.npz"
    all_bank_p = banks_dir / "everything_all_d91_bank.npz"
    srch_bank_p = banks_dir / "everything_search_d91_bank.npz"
    for p in (cids_p, eval_bank_p, all_bank_p, srch_bank_p):
        if p.exists():
            raise SystemExit(f"{p} already exists (create-once)")

    # 1. Load exhaustive non-d91 bank (`all_non_d91_complete_bank.npz`, ~3,300+ complete real tables, 0% d91, 0 holdout)
    non_d91_p = DATA / "inputs" / "all_non_d91_complete_bank.npz"
    by_cid: dict[str, tuple[str, str, np.ndarray]] = {}
    with np.load(non_d91_p, allow_pickle=False) as zal:
        d_al = zal["data"]
        sh_al = zal["shape"].astype(np.int64)
        of_al = zal["offset"].astype(np.int64)
        cids_al = [str(x) for x in zal["cids"].tolist()]
        fams_al = [str(x) for x in zal["family"].tolist()]
        cohs_al = [str(x) for x in zal["cohort"].tolist()]
        for i, c in enumerate(cids_al):
            r, w = int(sh_al[i, 0]), int(sh_al[i, 1])
            o = int(of_al[i])
            mat = np.array(d_al[o : o + r * w], np.float32, copy=True).reshape(r, w)
            by_cid[c] = (fams_al[i], cohs_al[i], mat)
        non_d91_rows = int(sh_al[:, 0].sum())
        non_d91_cells = int((sh_al[:, 0] * sh_al[:, 1]).sum())

    base_cids = sorted(by_cid)
    base_tables = [(c, by_cid[c][0], by_cid[c][1], by_cid[c][2]) for c in base_cids]

    # 2. Load optional d91-associated S3 cousin regression tables (`d91_cousin_s3_bank.npz`, 0 sealed opens)
    cousin_tables = []
    cousin_p = DATA / "inputs" / "d91_cousin_s3_bank.npz"
    if cousin_p.is_file():
        with np.load(cousin_p, allow_pickle=False) as zco:
            d_co = zco["data"]
            sh_co = zco["shape"].astype(np.int64)
            of_co = zco["offset"].astype(np.int64)
            cids_co = [str(x) for x in zco["cids"].tolist()]
            fams_co = [str(x) for x in zco["family"].tolist()]
            cohs_co = [str(x) for x in zco["cohort"].tolist()]
            for i, c in enumerate(cids_co):
                if c in by_cid:
                    raise SystemExit(f"unexpected cid overlap between non-d91 and d91_cousin_s3_bank: {c}")
                r, w = int(sh_co[i, 0]), int(sh_co[i, 1])
                o = int(of_co[i])
                mat = np.array(d_co[o : o + r * w], np.float32, copy=True).reshape(r, w)
                cousin_tables.append((c, fams_co[i], cohs_co[i], mat))

    # 3. Read all 91 d91 tables (both 100% complete [X_search U X_audit, y_all] AND 70% SEARCH-only + test-time eval bank)
    rows = C.dev_rows_for_stage_v()
    d91_cids = [r["context_sha256"] for r in rows]
    d91_fam = {r["context_sha256"]: r["family_id"] for r in rows}
    if len(d91_cids) != N_CONTEXTS or len(set(d91_fam.values())) != N_FAMILIES:
        raise SystemExit("d91 is not 91 contexts / 19 families")
    if set(d91_cids) & set(by_cid):
        raise SystemExit("unexpected cid overlap between non-d91 banks and d91")

    chk = {}
    S_list, AU_list, labs_list, ys_list, fams_list, ya_sha_list, y_sha_list = [], [], [], [], [], [], []
    d91_all_tables = []
    d91_search_tables = []

    for c in d91_cids:
        t_sha, v_sha = _sha(C.table_path(c)), _sha(VIEWS / f"{c}.npz")
        chk[c] = {"table_sha256": t_sha, "views_sha256": v_sha}
        if man.get(f"{c}.jtt1") != t_sha or man.get(f"{c}.npz") != v_sha:
            raise SystemExit(f"{c[:12]}: d91 table / SEARCH view sha not in H335 prep manifests")

        env = C.read_jtt1(C.table_path(c))
        full_mat = _clean(env.mat).astype(np.float32)
        search_env, _audit_env, m_aud = C.search_and_audit_envelopes(c, env)
        y = np.asarray(search_env.y, np.float64)
        ya = np.asarray(env.y, np.float64)[m_aud]
        s_mat = _clean(search_env.mat).astype(np.float32)
        xa_mat = _clean(np.asarray(env.mat)[m_aud][:, :-1]).astype(np.float32)

        d91_all_tables.append((c, d91_fam[c], "D91-ALL", full_mat))
        d91_search_tables.append((c, d91_fam[c], "D91-SEARCH", s_mat))

        S_list.append(s_mat)
        AU_list.append(xa_mat)
        labs_list.append(DP15.kfold_labels(c, y).astype(np.int8))
        ys_list.append(y)
        fams_list.append(d91_fam[c])
        ya_sha_list.append(hashlib.sha256(np.ascontiguousarray(ya).tobytes()).hexdigest())
        y_sha_list.append(hashlib.sha256(np.ascontiguousarray(y).tobytes()).hexdigest())

    _write_once(cids_p, {
        "schema": "jope.h449.cids91/v1",
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
        search=np.concatenate([t.ravel() for t in S_list]),
        s_shape=np.array([t.shape for t in S_list], np.int64),
        audit=np.concatenate([t.ravel() for t in AU_list]),
        a_shape=np.array([t.shape for t in AU_list], np.int64),
        lab=np.concatenate(labs_list),
        y64=np.concatenate(ys_list),
        cids=np.array(d91_cids),
        family=np.array(fams_list),
        cohort=np.array(["D91"] * len(d91_cids)),
        ya_sha256=np.array(ya_sha_list),
        y_sha256=np.array(y_sha_list),
    )
    os.chmod(eval_bank_p, 0o444)
    _write_once(Path(str(eval_bank_p) + ".json"), {
        "schema": "jope.h449.bank_d91_search/v1",
        "sha256": _sha(eval_bank_p),
        "audit_targets_written": False,
        "lock_sha256": _sha(LOCK_PATH),
    })

    rec_all = _save_flat_bank(all_bank_p, base_tables + cousin_tables + d91_all_tables, "h449_everything_all_d91")
    rec_srch = _save_flat_bank(srch_bank_p, base_tables + cousin_tables + d91_search_tables, "h449_everything_search_d91")
    d91_all_shapes = np.array([t[3].shape for t in d91_all_tables], np.int64)
    cousin_cells = int(sum(t[3].shape[0] * t[3].shape[1] for t in cousin_tables))
    cousin_rows = int(sum(t[3].shape[0] for t in cousin_tables))
    manifest = {
        "schema": "jope.h449.banks_manifest/v2",
        "quarantined_unfair_diagnostic": True,
        "all_non_d91_complete": {
            "file": str(non_d91_p),
            "sha256": _sha(non_d91_p),
            "tables": len(base_tables),
            "total_rows": non_d91_rows,
            "total_cells": non_d91_cells,
            "d91_tables": 0,
        },
        "d91_cousin_s3": {
            "tables": len(cousin_tables),
            "total_rows": cousin_rows,
            "total_cells": cousin_cells,
        },
        "d91_complete_91": {
            "tables": len(d91_all_tables),
            "total_rows": int(d91_all_shapes[:, 0].sum()),
            "total_cells": int((d91_all_shapes[:, 0] * d91_all_shapes[:, 1]).sum()),
        },
        "d91_total_tables_in_everything": len(d91_all_tables) + len(cousin_tables),
        "everything_all_d91": {
            **rec_all,
            "total_cells": int(non_d91_cells + cousin_cells + int((d91_all_shapes[:, 0] * d91_all_shapes[:, 1]).sum())),
            "d91_audit_targets_included": True,
        },
        "everything_search_d91": {**rec_srch, "d91_audit_targets_included": False},
        "eval_bank_d91_search_sha256": _sha(eval_bank_p),
        "lock_sha256": _sha(LOCK_PATH),
        "wall_s": round(time.time() - t0, 2),
    }
    _write_once(banks_dir / "banks_manifest.json", manifest)
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
