#!/usr/bin/env python3
"""H448: Official fair dev91-audit-v1 evaluation driver (`d91_h448.py`).

Runs inside `sandbox_h448_d91.sh` (`--network none` `h317-empty:v1` on `xbabe0`).
Guard: importing `deploy_h416` (-> `deploy_h415` -> `deploy_h410` -> `gp_h381` -> `h335_core`) installs H317 guard v2.
Points Stage V unlock at `DATA/stage_v/LOCK.json` and create-once receipt `ns/receipts/H448_LOCK_sha256.txt`.

Commands:
  lock --spec SPEC.json   prelock: writes create-once `DATA/stage_v/LOCK.json`; prints its sha256. Reads no d91 byte.
  unlock                  one-shot `C.unlock_stage_v()`.
  inputs                  resume; re-verifies `LOCK['inputs_sha256']` and `icl_assets.sha256`, prints guard report.
  bank_and_screen         resume; writes create-once `DATA/cids91.json`, `DATA/bank_d91_search.npz`
                          (`audit_targets_written = False`), and `DATA/SCREEN_D91_SEARCH.json` (`H425` feature-only
                          `S1-S4` screen on `X_search` against the frozen champion's `training_bank`; `0` AUDIT target
                          reads, `confirm_targets_keyed = False`).

Not here: reading any entrant utility. The readout (`analyze_h448.py`) runs on `xbabe2` from sha-checked pulled copies.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import deploy_h416 as DH  # noqa: E402  (FIRST: deploy_h415 -> deploy_h410 -> gp_h381 -> h335_core pins + guard)
import train_h416 as T16  # noqa: E402

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import time  # noqa: E402
from collections import Counter, defaultdict  # noqa: E402

import numpy as np  # noqa: E402

DP15, K, C = DH.DP15, DH.K, DH.C

WAVE = "H448"
ENTRANT = "h448.APEX_FAIR_CHAMPION"
DATA = Path(os.environ.get("H448_D91", "/home/ubuntu/jope-data/h448-d91"))
C.DATA_ROOT = DATA
C.LOCK_RECEIPT = Path("/home/ubuntu/jope_plan/ns/receipts/H448_LOCK_sha256.txt")
LOCK_PATH = DATA / "stage_v" / "LOCK.json"
VIEWS = DATA / "views-search"
H335D = Path("/home/ubuntu/jope-data/h335-cl")
WORKER = H335D / "bin" / "bootstrap.s1main.superset-2a848c8f8085"
WORKER_SHA256 = "2a848c8f8085a39210c9714b3d3b80448fea66afecb54a951345f2926c26d7a2"
SEEDS = tuple(K.SEEDS_V)  # (335101, 335102)
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
    DATA / "inputs" / "board_audit.sha256",
    DATA / "inputs" / "icl_assets.sha256",
)
# No path above may contain an H317 guard DL-Val token (dev91, dl-val, dl_val, dlval, hold_out, hold-out).
CODE_FILES = DH.code_files() + (
    "gp_h415.py",
    "train_h415.py",
    "gp_h416.py",
    "train_h416.py",
    "DESIGN.json",
    "a2_cells.json",
    "cache_h422.py",
    "guard_matern_core.py",
    "cols_matern.py",
    "keys_core.py",
    "cols_h430.py",
    "formula_core.py",
    "cols_h433.py",
    "icl_model.py",
    "icl_prior.py",
    "icl_inject.py",
    "icl_diag.py",
    "icl_resid.py",
    "struct_cols.py",
    "eval_cw5_fitsel.py",
    "cols_r4_e8.py",
    "FROZEN_FAIR_CHAMPION.json",
    "d91_h448.py",
    "sandbox_h448_d91.sh",
    "PREREG_H448.md",
    "analyze_h448.py",
    "selftest_h448.py",
    "run_h448.sh",
    "stage_h448.sh",
)

if SEEDS != (335101, 335102):
    raise AssertionError("h335_core.SEEDS_V is not (335101, 335102)")

# ----------------------------------------------------------------------------------------------------------------------
# Verbatim H425 feature-only contamination screen constants & functions (screen_h425.py; use_target = False)
# ----------------------------------------------------------------------------------------------------------------------
H425_K = 256
SPEC_MIN_DISTINCT, SPEC_MIN_SIG_SHARE = 100, 0.5
COL_J, S1_MIN_COLS, S1_MIN_SHARE, S1_FEW_J = 0.05, 3, 0.30, 0.5
S2_MIN_WIDTH, S2_MIN_UNIQUE, S2_MIN_SHARE, S2_MIN_ROWS = 4, 0.5, 0.05, 20
S3_MAX_DF, S3_MIN_COLS, S3_MIN_SHARE = 10, 3, 0.5
S4_MIN_SHARED, S4_MIN_J = 2, 0.75
NANKEY = np.int64(-(1 << 62))
STOP = {
    "srcfile", "legacy", "rdatasets", "regression", "classification", "runtime", "dataset", "datasets",
    "data", "tbl", "df", "train", "test", "csv", "v1", "v2", "v3", "the", "of", "and", "openml", "uci", "kaggle",
}


def h425_keys(x: np.ndarray, digits: int = 6) -> np.ndarray:
    x = np.asarray(x, np.float64)
    out = np.full(x.shape, NANKEY, np.int64)
    fin = np.isfinite(x)
    z = x[fin]
    e = np.zeros(z.shape, np.int64)
    m = np.zeros(z.shape, np.int64)
    nz = z != 0
    e[nz] = np.floor(np.log10(np.abs(z[nz]))).astype(np.int64)
    scale = 10 ** (digits - 1)
    m[nz] = np.rint(z[nz] / 10.0 ** e[nz] * scale).astype(np.int64)
    over = np.abs(m) >= 10 * scale
    m[over] //= 10
    e[over] += 1
    out[fin] = m * 1000 + (np.clip(e, -399, 399) + 400)
    return out


def h425_mix(k: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore"):
        z = k.astype(np.uint64) + np.uint64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return z ^ (z >> np.uint64(31))


def h425_jaccard(a: np.ndarray, b: np.ndarray) -> float:
    u = np.union1d(a, b)[:H425_K]
    both = np.intersect1d(np.intersect1d(u, a, assume_unique=True), b, assume_unique=True)
    return float(both.size) / float(u.size) if u.size else 0.0


def h425_name_tokens(key: str):
    s = key.lower()
    body = s.split(":", 1)[-1]
    if re.fullmatch(r"[0-9a-f]{24,64}", body) or re.fullmatch(r"fam-[0-9a-f]{24,64}", body):
        return None
    toks = [t for t in re.split(r"[^a-z0-9]+", s) if t and t not in STOP and not (t.isdigit() and len(t) < 4)]
    return frozenset(toks) if sum(len(t) for t in toks) >= 5 else None


def sketch_features_only(cid: str, X: np.ndarray) -> dict:
    """Feature-only H425 sketch (`tk = None`, zero target values touched)."""
    X = np.asarray(X, np.float64)
    n, F = X.shape
    KX = h425_keys(X)
    spec, pairs = [], []
    for j in range(F):
        col = KX[:, j]
        dk = np.unique(col[col != NANKEY])
        if len(dk) >= SPEC_MIN_DISTINCT and float(((np.abs(dk // 1000)) % 1000 != 0).mean()) >= SPEC_MIN_SIG_SHARE:
            spec.append(np.sort(h425_mix(dk))[:H425_K])
        fin = np.isfinite(X[:, j])
        if fin.any():
            xs = X[fin, j]
            lo, hi = h425_keys(np.array([xs.min(), xs.max()]), 4).tolist()
            pairs.append(int(h425_mix(np.array([lo * 1_000_003 + hi], np.int64))[0]))
    srt = np.sort(KX, axis=1)
    rows = (
        np.unique(np.array([int.from_bytes(hashlib.blake2b(r.tobytes(), digest_size=8).digest(), "little") for r in srt], np.uint64))
        if n
        else np.zeros(0, np.uint64)
    )
    return {"cid": cid, "n": int(n), "F": int(F), "spec": spec, "pairs": sorted(set(pairs)), "rows": rows, "tk": None}


def screen_one_h425(q: dict, fit: list[dict], col_idx, row_idx, pair_idx, pair_df, names_fit, name_q) -> dict:
    out = {}
    okset = set(range(len(fit)))
    per_fit = defaultdict(lambda: [0, 0])
    for sk in q["spec"]:
        cnt = Counter()
        for h in sk.tolist():
            for ref in col_idx.get(h, ()):
                cnt[ref] += 1
        best = {}
        for (fi, cj), c in cnt.items():
            if fi not in okset or c < 4:
                continue
            j = h425_jaccard(sk, fit[fi]["spec"][cj])
            if j >= COL_J and j > best.get(fi, 0.0):
                best[fi] = j
        for fi, j in best.items():
            per_fit[fi][0] += 1
            per_fit[fi][1] += int(j >= S1_FEW_J)
    ns = len(q["spec"])
    need = max(S1_MIN_COLS, int(np.ceil(S1_MIN_SHARE * ns)))
    fi, (nm, _) = max(per_fit.items(), key=lambda kv: (kv[1][0], kv[1][1]), default=(None, (0, 0)))
    fs, (_, nstrong) = max(per_fit.items(), key=lambda kv: (kv[1][1], kv[1][0]), default=(None, (0, 0)))
    copy = ns >= 3 and nstrong >= need
    weak = (ns >= 3 and nm >= need) or (1 <= ns <= 2 and nstrong == ns)
    if copy or (1 <= ns <= 2 and weak):
        fi = fs
    out["s1"] = {
        "applies": ns >= 1,
        "specific_columns": ns,
        "matched": int(nm),
        "matched_strong": int(nstrong),
        "share": (nm / ns) if ns else None,
        "fit": fit[fi]["cid"] if fi is not None else None,
        "flag_copy": bool(copy),
        "flag": bool(copy or weak),
    }
    applies = q["F"] >= S2_MIN_WIDTH and q["n"] > 0 and len(q["rows"]) >= S2_MIN_UNIQUE * q["n"]
    cnt = Counter()
    if applies:
        for h in q["rows"].tolist():
            for fi2 in row_idx.get(h, ()):
                if fi2 in okset:
                    cnt[fi2] += 1
    fi2, shared = cnt.most_common(1)[0] if cnt else (None, 0)
    share = shared / len(q["rows"]) if len(q["rows"]) else 0.0
    out["s2"] = {
        "applies": bool(applies),
        "shared_rows": int(shared),
        "share": float(share),
        "fit": fit[fi2]["cid"] if fi2 is not None else None,
        "flag": bool(applies and share >= S2_MIN_SHARE and shared >= S2_MIN_ROWS),
    }
    rare = [p for p in q["pairs"] if pair_df[p] <= S3_MAX_DF]
    cnt = Counter()
    for p in rare:
        for fi3 in pair_idx.get(p, ()):
            if fi3 in okset:
                cnt[fi3] += 1
    fi3, m3 = cnt.most_common(1)[0] if cnt else (None, 0)
    out["s3"] = {
        "applies": len(rare) >= S3_MIN_COLS,
        "rare_pairs": len(rare),
        "matched": int(m3),
        "share": (m3 / len(rare)) if rare else None,
        "fit": fit[fi3]["cid"] if fi3 is not None else None,
        "flag": bool(m3 >= S3_MIN_COLS and m3 >= S3_MIN_SHARE * len(rare)),
    }
    hit = None
    if name_q is not None:
        for i in range(len(fit)):
            t = names_fit[i]
            if t is None:
                continue
            inter = len(name_q & t)
            if name_q == t or (inter >= S4_MIN_SHARED and inter / len(name_q | t) >= S4_MIN_J):
                hit = i
                break
    out["s4"] = {"applies": name_q is not None, "fit": fit[hit]["cid"] if hit is not None else None, "flag": hit is not None}
    out["b"] = None
    out["flag_feature"] = bool(out["s1"]["flag"] or out["s2"]["flag"] or out["s3"]["flag"] or out["s4"]["flag"])
    out["tier"] = "copy" if (out["s1"]["flag_copy"] or out["s2"]["flag"]) else "possible" if out["flag_feature"] else "clear"
    return out


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


def _frozen() -> dict:
    d = json.loads(FROZEN_FILE.read_text())
    if d.get("schema") != "jope.h447.frozen_fair_champion/v1":
        raise SystemExit("FROZEN_FAIR_CHAMPION.json has wrong schema")
    return d


def cmd_lock(a) -> None:
    if LOCK_PATH.exists():
        raise SystemExit(f"{LOCK_PATH} exists (create-once)")
    if C._STATE["stage_v"]:
        raise SystemExit("lock runs before the Stage V unlock")
    spec = json.loads(Path(a.spec).read_text())
    for k in ("prereg_commit", "stage_commit"):
        if not isinstance(spec.get(k), str) or len(spec[k]) != 40:
            raise SystemExit(f"spec needs a full {k}")
    design = json.loads(DESIGN_FILE.read_text())
    if design.get("schema") != "jope.h416.design/v1" or design.get("design") != DESIGN_NAME:
        raise SystemExit(f"staged DESIGN.json is not the H416 {DESIGN_NAME} design")
    frz = _frozen()
    if _sha(WORKER) != WORKER_SHA256:
        raise SystemExit("superset worker sha differs")
    code = K.code_manifest(CODE_FILES)
    code[str(WORKER)] = WORKER_SHA256
    inputs = {str(p): _sha(p) for p in PREP_MANIFESTS + INPUT_MANIFESTS}
    lk = {
        "schema": "jope.h448.lock/v1",
        "wave": WAVE,
        "entrant": ENTRANT,
        "entrant_wave": "H447",
        "board": "dev91-audit-v1",
        "design": DESIGN_NAME,
        "design_sha256": _sha(DESIGN_FILE),
        "frozen_champion_sha256": _sha(FROZEN_FILE),
        "selected_arm": frz["selected_arm"],
        "a2_cells_sha256": _sha(A2_CELLS_FILE),
        "seeds": list(SEEDS),
        "contexts_rule": "C.dev_rows_for_stage_v(): all 91 d91 contexts / 19 families + H425 X_search-only screen",
        "code_sha256": code,
        "inputs_sha256": inputs,
        "prereg_commit": spec["prereg_commit"],
        "stage_commit": spec["stage_commit"],
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    _write_once(LOCK_PATH, lk)
    print(_sha(LOCK_PATH))


def _inputs_ok() -> dict:
    lk = json.loads(LOCK_PATH.read_text())
    for p, sha in lk["inputs_sha256"].items():
        if _sha(Path(p)) != sha:
            raise SystemExit(f"input changed after LOCK: {p}")
    if sorted(lk["inputs_sha256"]) != sorted(str(p) for p in PREP_MANIFESTS + INPUT_MANIFESTS):
        raise SystemExit("LOCK inputs differ from this driver's input list")
    # Also verify every asset listed in icl_assets.sha256
    icl_man = DATA / "inputs" / "icl_assets.sha256"
    for line in icl_man.read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            h, rel = line.split(None, 1)
            target = (DATA / "inputs" / rel.strip()) if not rel.strip().startswith("/") else Path(rel.strip())
            if _sha(target) != h:
                raise SystemExit(f"icl asset sha mismatch: {target}")
    out = {}
    for p in PREP_MANIFESTS:
        for line in Path(p).read_text().splitlines():
            if line.strip():
                h, name = line.split(None, 1)
                out[Path(name.strip().lstrip("*")).name] = h
    return out


def cmd_unlock(_a) -> None:
    C.unlock_stage_v()
    print(json.dumps(C.guard_report()))


def cmd_inputs(_a) -> None:
    C.resume_stage_v()
    _inputs_ok()
    print(json.dumps({"inputs": "ok", "lock_sha256": _sha(LOCK_PATH), "guard": C.guard_report()}))


def cmd_bank_and_screen(_a) -> None:
    C.resume_stage_v()
    man = _inputs_ok()
    frz = _frozen()
    t0 = time.time()

    cids_path = DATA / "cids91.json"
    bank_path = DATA / "bank_d91_search.npz"
    scr_path = DATA / "SCREEN_D91_SEARCH.json"
    if cids_path.exists() or bank_path.exists() or scr_path.exists():
        raise SystemExit("cids91.json, bank_d91_search.npz, or SCREEN_D91_SEARCH.json already exists (create-once)")

    rows = C.dev_rows_for_stage_v()
    cids = [r["context_sha256"] for r in rows]
    fam = {r["context_sha256"]: r["family_id"] for r in rows}
    row_by_cid = {r["context_sha256"]: r for r in rows}
    if len(cids) != N_CONTEXTS or len(set(fam.values())) != N_FAMILIES:
        raise SystemExit("d91 is not 91 contexts / 19 families")

    chk = {}
    S_list, AU_list, labs_list, ys_list, fams_list, cohorts_list, ya_sha_list, y_sha_list = [], [], [], [], [], [], [], []
    q_sketches = []

    for c in cids:
        t_sha, v_sha = _sha(C.table_path(c)), _sha(VIEWS / f"{c}.npz")
        chk[c] = {"table_sha256": t_sha, "views_sha256": v_sha}
        if man.get(f"{c}.jtt1") != t_sha or man.get(f"{c}.npz") != v_sha:
            raise SystemExit(f"{c[:12]}: d91 table / SEARCH view sha not in H335 prep manifests")

        env = C.read_jtt1(C.table_path(c))
        search_env, _audit_env, m_aud = C.search_and_audit_envelopes(c, env)
        y = np.asarray(search_env.y, np.float64)
        ya = np.asarray(env.y, np.float64)[m_aud]
        s_mat = _clean(search_env.mat)
        xa_mat = _clean(np.asarray(env.mat)[m_aud][:, :-1])

        S_list.append(s_mat.astype(np.float32))
        AU_list.append(xa_mat.astype(np.float32))
        labs_list.append(DP15.kfold_labels(c, y).astype(np.int8))
        ys_list.append(y)
        fams_list.append(fam[c])
        cohorts_list.append("D91")
        ya_sha_list.append(hashlib.sha256(np.ascontiguousarray(ya).tobytes()).hexdigest())
        y_sha_list.append(hashlib.sha256(np.ascontiguousarray(y).tobytes()).hexdigest())

        # H425 feature-only sketch on X_search ONLY (0 target values keyed, 0 AUDIT rows read)
        q_sketches.append(sketch_features_only(c, s_mat[:, :-1]))

    rec_cids = {
        "schema": "jope.h448.cids91/v1",
        "n": len(cids),
        "families": len(set(fam.values())),
        "cids": cids,
        "cids_sha256": hashlib.sha256("\n".join(cids).encode()).hexdigest(),
        "family_of": fam,
        "inputs": chk,
        "lock_sha256": _sha(LOCK_PATH),
        "guard": C.guard_report(),
    }
    _write_once(cids_path, rec_cids)

    np.savez(
        bank_path,
        search=np.concatenate([t.ravel() for t in S_list]),
        s_shape=np.array([t.shape for t in S_list], np.int64),
        audit=np.concatenate([t.ravel() for t in AU_list]),
        a_shape=np.array([t.shape for t in AU_list], np.int64),
        lab=np.concatenate(labs_list),
        y64=np.concatenate(ys_list),
        cids=np.array(cids),
        family=np.array(fams_list),
        cohort=np.array(cohorts_list),
        ya_sha256=np.array(ya_sha_list),
        y_sha256=np.array(y_sha_list),
    )
    os.chmod(bank_path, 0o444)
    bank_rec = {
        "schema": "jope.h448.bank_d91_search/v1",
        "file": str(bank_path),
        "sha256": _sha(bank_path),
        "contexts": len(cids),
        "families": len(set(fam.values())),
        "audit_targets_written": False,
        "search_rows": int(sum(t.shape[0] for t in S_list)),
        "audit_rows": int(sum(t.shape[0] for t in AU_list)),
        "lock_sha256": _sha(LOCK_PATH),
        "guard": C.guard_report(),
    }
    _write_once(Path(str(bank_path) + ".json"), bank_rec)

    # H425 feature-only screen against the frozen champion's training bank
    icl_spec = frz["blocks"]["icl"]
    tb_name = icl_spec.get("training_bank") or "fit_icl_bank.npz"
    tb_path = DATA / "inputs" / tb_name
    tb_sha = _sha(tb_path)
    if icl_spec.get("enabled") and tb_sha != icl_spec["training_bank_sha256"]:
        raise SystemExit(f"training_bank sha {tb_sha} != frozen {icl_spec['training_bank_sha256']}")

    fit_sketches = []
    fam_fit_keys = []
    with np.load(tb_path, allow_pickle=False) as zb:
        data_b = zb["data"]
        shapes_b = zb["shape"].astype(np.int64)
        offs_b = zb["offset"].astype(np.int64)
        cids_b = [str(x) for x in zb["cids"].tolist()]
        fams_b = [str(x) for x in zb["family"].tolist()]
        for i, cid_fit in enumerate(cids_b):
            r_fit, c_fit = int(shapes_b[i, 0]), int(shapes_b[i, 1])
            off = int(offs_b[i])
            mat_fit = np.asarray(data_b[off : off + r_fit * c_fit], np.float64).reshape(r_fit, c_fit)
            fit_sketches.append(sketch_features_only(cid_fit, mat_fit[:, :-1]))
            fam_fit_keys.append(fams_b[i])

    col_idx, row_idx, pair_idx, pair_df = defaultdict(list), defaultdict(list), defaultdict(list), Counter()
    for i, f_sk in enumerate(fit_sketches):
        for j, s in enumerate(f_sk["spec"]):
            for h in s.tolist():
                col_idx[h].append((i, j))
        for h in f_sk["rows"].tolist():
            row_idx[h].append(i)
        for p in f_sk["pairs"]:
            pair_idx[p].append(i)
    for sk_any in fit_sketches + q_sketches:
        for p in sk_any["pairs"]:
            pair_df[p] += 1
    names_fit = [h425_name_tokens(k) for k in fam_fit_keys]
    fit_fam_by_cid = dict(zip([f["cid"] for f in fit_sketches], fam_fit_keys))

    per = {}
    for x in q_sketches:
        r_meta = row_by_cid[x["cid"]]
        q_key = str(r_meta.get("family_key") or r_meta["family_id"])
        res = screen_one_h425(x, fit_sketches, col_idx, row_idx, pair_idx, pair_df, names_fit, h425_name_tokens(q_key))
        res.update({
            "cohort": "D91",
            "family_id": r_meta["family_id"],
            "rows_search": x["n"],
            "width": x["F"],
        })
        for s_rule in ("s1", "s2", "s3", "s4"):
            if res[s_rule] and res[s_rule].get("fit"):
                res[s_rule]["fit_family_key"] = fit_fam_by_cid.get(res[s_rule]["fit"])
        per[x["cid"]] = res

    sel_vals = list(per.values())
    summ = {
        "contexts": len(sel_vals),
        "families": len({v["family_id"] for v in sel_vals}),
        **{s_rule: sum(bool(v[s_rule] and v[s_rule]["flag"]) for v in sel_vals) for s_rule in ("s1", "s2", "s3", "s4")},
        "flag_feature": sum(v["flag_feature"] for v in sel_vals),
        "clear_without_feature": sum(not v["flag_feature"] for v in sel_vals),
        "clear_families_without_feature": len({v["family_id"] for v in sel_vals if not v["flag_feature"]}),
        "tiers": dict(Counter(v["tier"] for v in sel_vals)),
    }
    scr_doc = {
        "schema": "jope.h448.screen_d91_search/v1",
        "training_bank": tb_name,
        "training_bank_sha256": tb_sha,
        "reference_tables": len(fit_sketches),
        "confirm_targets_keyed": False,
        "audit_rows_read": 0,
        "summary": summ,
        "wall_s": round(time.time() - t0, 2),
        "lock_sha256": _sha(LOCK_PATH),
        "guard": C.guard_report(),
        "per_context": per,
    }
    _write_once(scr_path, scr_doc)
    print(json.dumps({"cids": len(cids), "bank_sha256": bank_rec["sha256"], "screen_summary": summ, "wall_s": scr_doc["wall_s"]}))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("lock")
    p.add_argument("--spec", required=True)
    for n in ("unlock", "inputs", "bank_and_screen"):
        sub.add_parser(n)
    a = ap.parse_args()
    {
        "lock": cmd_lock,
        "unlock": cmd_unlock,
        "inputs": cmd_inputs,
        "bank_and_screen": cmd_bank_and_screen,
    }[a.cmd](a)


if __name__ == "__main__":
    main()
