#!/usr/bin/env python3
"""H425 contamination screen, run inside the H335 sandbox with the frozen readers (guard v2 installed by apex_core).

  screen_h425.py --queries DEV --out FILE [--procs 8]                (the scored run; addendum 1: CONFIRM tables are not
                                                                      in the sandbox view, so CONFIRM is not screened here)
  screen_h425.py --queries DEV,CONFIRM --out FILE                    (the PREREG run; fails while CONFIRM tables are absent)
  screen_h425.py --queries FIT-SEL --out FILE                        (development rehearsal on a design cohort)

Question per query context: is its table a copy (same rows, a different row sample, or a column variant) of a table
of cohort FIT-ICL, the cohort the in-context model was trained on? Reference set = every FIT-ICL table of the manifest
(all tiers). Query set = the core contexts of DEV and every CONFIRM context (all tiers), as named by --queries.

Every cell value is reduced to a key: its value at 6 significant digits (NaN = one key). Screens, each against one
FIT-ICL table at a time (constants below are the pre-registered thresholds):
  S1  specific columns.  A column is specific if it has >= 100 distinct keys and at least half of them carry >= 4
      significant digits. Its sketch is the 256 smallest 64-bit hashes of its keys; two columns match if the Jaccard
      estimate of their key sets is >= 0.05, and matches strongly if it is >= 0.5. With N = max(3, 30% of the
      query's specific columns): COPY if >= N columns match strongly columns of one FIT-ICL table; flag (possible)
      if >= N columns match; with only 1 or 2 specific columns, flag (possible) if all of them match strongly.
  S2  rows.  Row hash = hash of the row's sorted feature keys (column order does not matter). Applies to tables with
      >= 4 feature columns and >= 50% distinct rows. COPY: >= 5% and >= 20 of the query's distinct rows occur in one
      FIT-ICL table.
  S3  column ranges.  (min, max) of a column at 4 significant digits; a pair is rare if it occurs in <= 10 of all
      screened tables. Flag: >= 3 and >= 50% of the query's rare pairs occur in one FIT-ICL table.
  S4  source names.  Family keys with a readable name are reduced to a token set (lower case; generic tokens such
      as "regression", "runtime", "dataset" removed). Flag: equal token sets, or >= 2 shared tokens with token
      Jaccard >= 0.75.
  B   baseline (target only; DEV and rehearsal only, never CONFIRM): >= 50% of the query's distinct target keys
      (>= 20 of them) occur among one FIT-ICL table's target keys (the H424 screen, on whole tables).
Tier of a context: "copy" (S1 COPY or S2), else "possible" (any other S1 / S3 / S4 flag), else "target_only" (B alone),
else "clear". Feature screens use feature columns only. For CONFIRM the target column is never turned into keys.
Only scores, flags and the identifiers of matched FIT-ICL tables are written: no value, key, hash or sketch of any
table leaves the sandbox.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import apex_core as A  # noqa: E402  (FIRST: frozen stack, pins and guard)

import numpy as np  # noqa: E402

C = A.C
K = 256
SPEC_MIN_DISTINCT, SPEC_MIN_SIG_SHARE = 100, 0.5
COL_J, S1_MIN_COLS, S1_MIN_SHARE, S1_FEW_J = 0.05, 3, 0.30, 0.5
S2_MIN_WIDTH, S2_MIN_UNIQUE, S2_MIN_SHARE, S2_MIN_ROWS = 4, 0.5, 0.05, 20
S3_MAX_DF, S3_MIN_COLS, S3_MIN_SHARE = 10, 3, 0.5
B_MIN_DISTINCT, B_MIN_SHARE = 20, 0.5
S4_MIN_SHARED, S4_MIN_J = 2, 0.75
NANKEY = np.int64(-(1 << 62))
STOP = {"srcfile", "legacy", "rdatasets", "regression", "classification", "runtime", "dataset", "datasets", "data", "tbl",
        "df", "train", "test", "csv", "v1", "v2", "v3", "the", "of", "and", "openml", "uci", "kaggle"}


def keys(x: np.ndarray, digits: int = 6) -> np.ndarray:
    """Value at `digits` significant digits as one int64 (mantissa * 1000 + exponent + 400); NaN / inf -> NANKEY."""
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


def mix(k: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore"):
        z = k.astype(np.uint64) + np.uint64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return z ^ (z >> np.uint64(31))


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    """Bottom-K estimate of the Jaccard similarity of two key sets from their sorted sketches (exact below K keys)."""
    u = np.union1d(a, b)[:K]
    both = np.intersect1d(np.intersect1d(u, a, assume_unique=True), b, assume_unique=True)
    return float(both.size) / float(u.size) if u.size else 0.0


def name_tokens(key: str):
    s = key.lower()
    body = s.split(":", 1)[-1]
    if re.fullmatch(r"[0-9a-f]{24,64}", body):
        return None
    toks = [t for t in re.split(r"[^a-z0-9]+", s) if t and t not in STOP and not (t.isdigit() and len(t) < 4)]
    return frozenset(toks) if sum(len(t) for t in toks) >= 5 else None


def sketch(job) -> dict:
    cid, use_target = job
    env = C.read_jtt1(C.table_path(cid))
    mat = np.asarray(env.mat, np.float64)
    X = mat[:, :-1]
    n, F = X.shape
    KX = keys(X)
    spec, pairs = [], []
    k4 = keys(X, 4) if n else np.zeros((0, F), np.int64)
    for j in range(F):
        col = KX[:, j]
        dk = np.unique(col[col != NANKEY])
        if len(dk) >= SPEC_MIN_DISTINCT and float(((np.abs(dk // 1000)) % 1000 != 0).mean()) >= SPEC_MIN_SIG_SHARE:
            spec.append(np.sort(mix(dk))[:K])
        fin = np.isfinite(X[:, j])
        if fin.any():
            xs = X[fin, j]
            lo, hi = keys(np.array([xs.min(), xs.max()]), 4).tolist()
            pairs.append(int(mix(np.array([lo * 1_000_003 + hi], np.int64))[0]))
    srt = np.sort(KX, axis=1)
    rows = np.unique(np.array([int.from_bytes(hashlib.blake2b(r.tobytes(), digest_size=8).digest(), "little") for r in srt],
                              np.uint64)) if n else np.zeros(0, np.uint64)
    tk = None
    if use_target:
        y = mat[:, -1]
        ky = keys(y)
        tk = np.unique(ky[ky != NANKEY])
    return {"cid": cid, "n": int(n), "F": int(F), "spec": spec, "pairs": sorted(set(pairs)), "rows": rows, "tk": tk}


def screen(q: dict, fit: list, col_idx, row_idx, pair_idx, pair_df, names_fit, name_q, skip_family=None, fam_fit=None) -> dict:
    out = {}
    ok = [i for i in range(len(fit)) if skip_family is None or fam_fit[i] != skip_family]
    okset = set(ok)
    # S1
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
            j = jaccard(sk, fit[fi]["spec"][cj])
            if j >= COL_J and j > best.get(fi, 0.0):
                best[fi] = j
        for fi, j in best.items():
            per_fit[fi][0] += 1
            per_fit[fi][1] += int(j >= S1_FEW_J)
    ns = len(q["spec"])
    need = max(S1_MIN_COLS, int(np.ceil(S1_MIN_SHARE * ns)))
    fi, (nm, _) = max(per_fit.items(), key=lambda kv: (kv[1][0], kv[1][1]), default=(None, (0, 0)))
    fs, (_, nstrong) = max(per_fit.items(), key=lambda kv: (kv[1][1], kv[1][0]), default=(None, (0, 0)))
    copy = ns >= 3 and nstrong >= need                      # many columns with at least half of their values shared
    weak = (ns >= 3 and nm >= need) or (1 <= ns <= 2 and nstrong == ns)
    if copy or (1 <= ns <= 2 and weak):
        fi = fs
    out["s1"] = {"applies": ns >= 1, "specific_columns": ns, "matched": int(nm), "matched_strong": int(nstrong),
                 "share": (nm / ns) if ns else None, "fit": fit[fi]["cid"] if fi is not None else None,
                 "flag_copy": bool(copy), "flag": bool(copy or weak)}
    # S2
    applies = q["F"] >= S2_MIN_WIDTH and q["n"] > 0 and len(q["rows"]) >= S2_MIN_UNIQUE * q["n"]
    cnt = Counter()
    if applies:
        for h in q["rows"].tolist():
            for fi2 in row_idx.get(h, ()):
                if fi2 in okset:
                    cnt[fi2] += 1
    fi2, shared = cnt.most_common(1)[0] if cnt else (None, 0)
    share = shared / len(q["rows"]) if len(q["rows"]) else 0.0
    out["s2"] = {"applies": bool(applies), "shared_rows": int(shared), "share": float(share),
                 "fit": fit[fi2]["cid"] if fi2 is not None else None,
                 "flag": bool(applies and share >= S2_MIN_SHARE and shared >= S2_MIN_ROWS)}
    # S3
    rare = [p for p in q["pairs"] if pair_df[p] <= S3_MAX_DF]
    cnt = Counter()
    for p in rare:
        for fi3 in pair_idx.get(p, ()):
            if fi3 in okset:
                cnt[fi3] += 1
    fi3, m3 = cnt.most_common(1)[0] if cnt else (None, 0)
    out["s3"] = {"applies": len(rare) >= S3_MIN_COLS, "rare_pairs": len(rare), "matched": int(m3),
                 "share": (m3 / len(rare)) if rare else None, "fit": fit[fi3]["cid"] if fi3 is not None else None,
                 "flag": bool(m3 >= S3_MIN_COLS and m3 >= S3_MIN_SHARE * len(rare))}
    # S4
    hit = None
    if name_q is not None:
        for i in ok:
            t = names_fit[i]
            if t is None:
                continue
            inter = len(name_q & t)
            if name_q == t or (inter >= S4_MIN_SHARED and inter / len(name_q | t) >= S4_MIN_J):
                hit = i
                break
    out["s4"] = {"applies": name_q is not None, "fit": fit[hit]["cid"] if hit is not None else None, "flag": hit is not None}
    # B
    if q["tk"] is not None:
        tq = set(q["tk"].tolist())
        best, arg = 0.0, None
        if len(tq) >= B_MIN_DISTINCT:
            for i in ok:
                s = len(tq & fit[i]["tset"]) / len(tq)
                if s > best:
                    best, arg = s, i
        out["b"] = {"applies": len(tq) >= B_MIN_DISTINCT, "distinct_targets": len(tq), "share": float(best),
                    "fit": fit[arg]["cid"] if arg is not None else None, "flag": bool(len(tq) >= B_MIN_DISTINCT and best >= B_MIN_SHARE)}
    else:
        out["b"] = None
    out["flag_feature"] = bool(out["s1"]["flag"] or out["s2"]["flag"] or out["s3"]["flag"] or out["s4"]["flag"])
    tb = bool(out["b"] and out["b"]["flag"])
    out["tier"] = ("copy" if (out["s1"]["flag_copy"] or out["s2"]["flag"]) else "possible" if out["flag_feature"]
                   else "target_only" if tb else "clear")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--queries", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--procs", type=int, default=8)
    a = ap.parse_args()
    if a.procs > 8:
        raise SystemExit("at most 8 processes")
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"{out} exists (create-once)")
    cohorts = a.queries.split(",")
    if cohorts not in (["DEV"], ["DEV", "CONFIRM"], ["FIT-SEL"]):
        raise SystemExit("--queries: DEV (addendum 1), DEV,CONFIRM (needs CONFIRM tables in the view) or FIT-SEL (rehearsal)")
    t0 = time.time()
    A.install()
    rows = A.rows()
    fit_c = sorted(c for c, r in rows.items() if r["cohort"] == "FIT-ICL")
    qry = []
    for co in cohorts:
        qry += sorted(c for c, r in rows.items() if r["cohort"] == co and (co == "CONFIRM" or A.is_core(r)))
    if {rows[c]["family_id"] for c in qry} & {rows[c]["family_id"] for c in fit_c}:
        raise SystemExit("a query family is a FIT-ICL family")
    jobs = [(c, True) for c in fit_c] + [(c, rows[c]["cohort"] != "CONFIRM") for c in qry]
    with Pool(a.procs) as pool:
        sk = pool.map(sketch, jobs, chunksize=4)
    fit, q = sk[:len(fit_c)], sk[len(fit_c):]
    if any(x["tk"] is not None for x in q if rows[x["cid"]]["cohort"] == "CONFIRM"):
        raise SystemExit("a CONFIRM target was turned into keys")
    col_idx, row_idx, pair_idx, pair_df = defaultdict(list), defaultdict(list), defaultdict(list), Counter()
    for i, f in enumerate(fit):
        f["tset"] = set(f["tk"].tolist())
        for j, s in enumerate(f["spec"]):
            for h in s.tolist():
                col_idx[h].append((i, j))
        for h in f["rows"].tolist():
            row_idx[h].append(i)
        for p in f["pairs"]:
            pair_idx[p].append(i)
    for x in sk:
        for p in x["pairs"]:
            pair_df[p] += 1
    fam_fit = [rows[f["cid"]]["family_key"] for f in fit]
    names_fit = [name_tokens(k) for k in fam_fit]
    per = {}
    for x in q:
        r = rows[x["cid"]]
        res = screen(x, fit, col_idx, row_idx, pair_idx, pair_df, names_fit, name_tokens(r["family_key"]))
        res.update({"cohort": r["cohort"], "core": bool(A.is_core(r)), "family_key": r["family_key"], "family_id": r["family_id"],
                    "rows": x["n"], "width": x["F"]})
        for s in ("s1", "s2", "s3", "s4", "b"):
            if res[s] and res[s].get("fit"):
                res[s]["fit_family_key"] = rows[res[s]["fit"]]["family_key"]
        res["flag_any"] = bool(res["flag_feature"] or (res["b"] or {}).get("flag", False))
        per[x["cid"]] = res
    # calibration: every FIT-ICL table against the FIT-ICL tables of other families (how often the rules fire there)
    cal = Counter()
    for i, f in enumerate(fit):
        res = screen(f, fit, col_idx, row_idx, pair_idx, pair_df, names_fit, names_fit[i], skip_family=fam_fit[i], fam_fit=fam_fit)
        for s in ("s1", "s2", "s3", "s4", "b"):
            cal[s] += int(res[s]["flag"])
        cal["feature"] += int(res["flag_feature"])
        cal["tier_" + res["tier"]] += 1
    summ = {}
    for co in cohorts:
        sel = [v for v in per.values() if v["cohort"] == co]
        summ[co] = {"contexts": len(sel), "core": sum(v["core"] for v in sel),
                    **{s: sum(bool(v[s] and v[s]["flag"]) for v in sel) for s in ("s1", "s2", "s3", "s4", "b")},
                    "flag_feature": sum(v["flag_feature"] for v in sel), "flag_any": sum(v["flag_any"] for v in sel),
                    "tiers": dict(Counter(v["tier"] for v in sel)), "tiers_core": dict(Counter(v["tier"] for v in sel if v["core"])),
                    "flag_feature_core": sum(v["flag_feature"] and v["core"] for v in sel),
                    "applies": {s: sum(bool(v[s] and v[s]["applies"]) for v in sel) for s in ("s1", "s2", "s3", "s4", "b")}}
    doc = {"schema": "jope.h425.screen/v1", "queries": cohorts, "fit_icl_tables": len(fit), "summary": summ,
           "fit_icl_calibration_flags": dict(cal), "manifest_sha256": A.MANIFEST_SHA256,
           "thresholds": {"K": K, "spec": [SPEC_MIN_DISTINCT, SPEC_MIN_SIG_SHARE], "s1": [COL_J, S1_MIN_COLS, S1_MIN_SHARE, S1_FEW_J],
                          "s2": [S2_MIN_WIDTH, S2_MIN_UNIQUE, S2_MIN_SHARE, S2_MIN_ROWS], "s3": [S3_MAX_DF, S3_MIN_COLS, S3_MIN_SHARE],
                          "s4": [S4_MIN_SHARED, S4_MIN_J], "b": [B_MIN_DISTINCT, B_MIN_SHARE]},
           "confirm_targets_keyed": False, "guard": C.guard_report(), "wall_s": time.time() - t0, "procs": a.procs,
           "per_context": per}
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "x") as f:
        json.dump(doc, f, sort_keys=True)
    print(json.dumps({k: doc[k] for k in ("queries", "fit_icl_tables", "summary", "fit_icl_calibration_flags", "wall_s")}))


if __name__ == "__main__":
    main()
