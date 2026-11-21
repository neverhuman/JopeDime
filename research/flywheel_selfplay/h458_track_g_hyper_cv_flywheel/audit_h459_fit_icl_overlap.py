#!/usr/bin/env python3
"""Audits `fit_icl_bank.npz` (`H447` `r4_e8` base training bank) overlap against `bank_P201B.npz` and `bank_FITSEL.npz`.

Label-free on audit rows (`bank_P201B.npz` and `bank_FITSEL.npz` carry `ya` nowhere):
  1. Verifies exact CID and family-key overlap (`0/201` on `P201B`, `0/80` on `P201B-80`, `0/106` on `FITSEL`).
  2. Applies the exact `H425` (`screen_h425.py`) `S1 / S2 / S3 / S4 / B` feature/row/range/name/search-target screen
     between `fit_icl_bank.npz` (`419` tables) and `bank_P201B.npz` (`201` contexts and the `P201B-80` slice), using
     `[X_search; X_audit]` for feature screens `S1..S4` and `y_search` for `B` (`0` audit labels read).
  3. Cross-binds `SCREEN_FITSEL.json` and `EXCLUSION_FITSEL.json` (`18/106` `copy`, `88/106` non-copy).
  4. Emits create-once `0444` `/home/ubuntu/jope_plan/evidence/H459/OVERLAP_FIT_ICL_BANK_P201B_FITSEL.json`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

FIT_BANK = Path("/home/ubuntu/jope-data/h335-cl/apex/icl/fit_icl_bank.npz")
P201B_BANK = Path("/home/ubuntu/jope-data/h430/bank_P201B.npz")
FITSEL_BANK = Path("/home/ubuntu/jope-data/h335-cl/h424/dev/bank_FITSEL.npz")
SCREEN_FITSEL = Path("/home/ubuntu/jope-data/h335-cl/h431/pub-r2/SCREEN_FITSEL.json")
EXCLUSION_FITSEL = Path("/home/ubuntu/jope-data/h335-cl/h431/pub-r2/EXCLUSION_FITSEL.json")
H447_PREREG = Path("/home/ubuntu/jope_plan/evidence/H447/source/research/flywheel_selfplay/h447_legal_icl_104m_24m_r4e8/PREREG.md")

K = 256
SPEC_MIN_DISTINCT, SPEC_MIN_SIG_SHARE = 100, 0.5
COL_J, S1_MIN_COLS, S1_MIN_SHARE, S1_FEW_J = 0.05, 3, 0.30, 0.5
S2_MIN_WIDTH, S2_MIN_UNIQUE, S2_MIN_SHARE, S2_MIN_ROWS = 4, 0.5, 0.05, 20
S3_MAX_DF, S3_MIN_COLS, S3_MIN_SHARE = 10, 3, 0.5
B_MIN_DISTINCT, B_MIN_SHARE = 20, 0.5
S4_MIN_SHARED, S4_MIN_J = 2, 0.75
NANKEY = np.int64(-(1 << 62))
STOP = {
    "srcfile", "legacy", "rdatasets", "regression", "classification", "runtime", "dataset", "datasets",
    "data", "tbl", "df", "train", "test", "csv", "v1", "v2", "v3", "the", "of", "and", "openml", "uci", "kaggle",
}


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def norm_fam(f: str) -> str:
    s = str(f).strip()
    for pfx in ("fresh:", "fam-", "fam:"):
        if s.startswith(pfx):
            s = s[len(pfx):]
    return s


def keys(x: np.ndarray, digits: int = 6) -> np.ndarray:
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
    u = np.union1d(a, b)[:K]
    both = np.intersect1d(np.intersect1d(u, a, assume_unique=True), b, assume_unique=True)
    return float(both.size) / float(u.size) if u.size else 0.0


def name_tokens(key: str):
    s = norm_fam(key).lower()
    body = s.split(":", 1)[-1]
    if re.fullmatch(r"[0-9a-f]{24,64}", body):
        return None
    toks = [t for t in re.split(r"[^a-z0-9]+", s) if t and t not in STOP and not (t.isdigit() and len(t) < 4)]
    return frozenset(toks) if sum(len(t) for t in toks) >= 5 else None


def sketch_table(cid: str, X: np.ndarray, y: Optional[np.ndarray]) -> Dict:
    X = np.asarray(X, np.float64)
    n, F = X.shape
    KX = keys(X)
    spec, pairs = [], []
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
    row_h = (
        np.unique(
            np.array(
                [int.from_bytes(hashlib.blake2b(r.tobytes(), digest_size=8).digest(), "little") for r in srt],
                np.uint64,
            )
        )
        if n
        else np.zeros(0, np.uint64)
    )
    tk = None
    if y is not None:
        ky = keys(np.asarray(y, np.float64))
        tk = np.unique(ky[ky != NANKEY])
    return {"cid": cid, "n": int(n), "F": int(F), "spec": spec, "pairs": sorted(set(pairs)), "rows": row_h, "tk": tk}


def screen_one(q: Dict, fit: List[Dict], col_idx, row_idx, pair_idx, pair_df, names_fit, name_q) -> Dict:
    out: Dict[str, object] = {}
    okset = set(range(len(fit)))
    per_fit: Dict[int, List[int]] = defaultdict(lambda: [0, 0])
    for sk in q["spec"]:
        cnt: Counter = Counter()
        for h in sk.tolist():
            for ref in col_idx.get(h, ()):
                cnt[ref] += 1
        best: Dict[int, float] = {}
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
    cnt2: Counter = Counter()
    if applies:
        for h in q["rows"].tolist():
            for fi2_cand in row_idx.get(h, ()):
                cnt2[fi2_cand] += 1
    fi2, shared = cnt2.most_common(1)[0] if cnt2 else (None, 0)
    share2 = shared / len(q["rows"]) if len(q["rows"]) else 0.0
    out["s2"] = {
        "applies": bool(applies),
        "shared_rows": int(shared),
        "share": float(share2),
        "fit": fit[fi2]["cid"] if fi2 is not None else None,
        "flag": bool(applies and share2 >= S2_MIN_SHARE and shared >= S2_MIN_ROWS),
    }
    rare = [p for p in q["pairs"] if pair_df[p] <= S3_MAX_DF]
    cnt3: Counter = Counter()
    for p in rare:
        for fi3_cand in pair_idx.get(p, ()):
            cnt3[fi3_cand] += 1
    fi3, m3 = cnt3.most_common(1)[0] if cnt3 else (None, 0)
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
        for i, t in enumerate(names_fit):
            if t is None:
                continue
            inter = len(name_q & t)
            if name_q == t or (inter >= S4_MIN_SHARED and inter / len(name_q | t) >= S4_MIN_J):
                hit = i
                break
    out["s4"] = {
        "applies": name_q is not None,
        "fit": fit[hit]["cid"] if hit is not None else None,
        "flag": hit is not None,
    }
    if q["tk"] is not None:
        tq = set(q["tk"].tolist())
        best_b, arg_b = 0.0, None
        if len(tq) >= B_MIN_DISTINCT:
            for i, f_rec in enumerate(fit):
                s = len(tq & f_rec["tset"]) / len(tq)
                if s > best_b:
                    best_b, arg_b = s, i
        out["b"] = {
            "applies": len(tq) >= B_MIN_DISTINCT,
            "distinct_targets": len(tq),
            "share": float(best_b),
            "fit": fit[arg_b]["cid"] if arg_b is not None else None,
            "flag": bool(len(tq) >= B_MIN_DISTINCT and best_b >= B_MIN_SHARE),
        }
    else:
        out["b"] = None
    out["flag_feature"] = bool(out["s1"]["flag"] or out["s2"]["flag"] or out["s3"]["flag"] or out["s4"]["flag"])
    tb = bool(out["b"] and out["b"]["flag"])
    out["tier"] = (
        "copy"
        if (out["s1"]["flag_copy"] or out["s2"]["flag"])
        else "possible"
        if out["flag_feature"]
        else "target_only"
        if tb
        else "clear"
    )
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out",
        default="/home/ubuntu/jope_plan/evidence/H459/OVERLAP_FIT_ICL_BANK_P201B_FITSEL.json",
    )
    a = ap.parse_args()
    out_p = Path(a.out)
    if out_p.exists():
        raise SystemExit(f"{out_p} already exists (create-once)")

    t0 = time.time()
    shas = {
        "fit_icl_bank_npz": sha256_file(FIT_BANK),
        "bank_P201B_npz": sha256_file(P201B_BANK),
        "bank_FITSEL_npz": sha256_file(FITSEL_BANK),
        "SCREEN_FITSEL_json": sha256_file(SCREEN_FITSEL),
        "EXCLUSION_FITSEL_json": sha256_file(EXCLUSION_FITSEL),
        "H447_PREREG_md": sha256_file(H447_PREREG),
    }

    with np.load(FIT_BANK, allow_pickle=False) as zf:
        fit_cids = [str(c) for c in zf["cids"].tolist()]
        fit_fams = [str(f) for f in zf["family"].tolist()]
        fit_data = zf["data"]
        fit_shape = zf["shape"]
        fit_off = zf["offset"]
        fit_sketches: List[Dict] = []
        for i, cid in enumerate(fit_cids):
            n, w = int(fit_shape[i, 0]), int(fit_shape[i, 1])
            off = int(fit_off[i])
            mat = np.asarray(fit_data[off : off + n * w], np.float64).reshape(n, w)
            fit_sketches.append(sketch_table(cid, mat[:, :-1], mat[:, -1]))

    with np.load(P201B_BANK, allow_pickle=False) as zp:
        assert "ya" not in zp.files
        p201_cids = [str(c) for c in zp["cids"].tolist()]
        p201_fams = [str(f) for f in zp["family"].tolist()]
        ss, sa = zp["s_shape"], zp["a_shape"]
        search, audit, y64 = zp["search"], zp["audit"], zp["y64"]
        p201_sketches: List[Tuple[str, str, Dict]] = []
        so = ao = lo = 0
        for i, cid in enumerate(p201_cids):
            n_s, w = int(ss[i, 0]), int(ss[i, 1])
            n_a, f_w = int(sa[i, 0]), int(sa[i, 1])
            s_mat = np.asarray(search[so : so + n_s * w], np.float64).reshape(n_s, w)
            x_a = np.asarray(audit[ao : ao + n_a * f_w], np.float64).reshape(n_a, f_w)
            y_s = np.asarray(y64[lo : lo + n_s], np.float64)
            x_all = np.vstack([s_mat[:, :-1], x_a])
            p201_sketches.append((cid, p201_fams[i], sketch_table(cid, x_all, y_s)))
            so += n_s * w
            ao += n_a * f_w
            lo += n_s

    with np.load(FITSEL_BANK, allow_pickle=False) as zfs:
        assert "ya" not in zfs.files
        fitsel_cids = [str(c) for c in zfs["cids"].tolist()]
        fitsel_fams = [str(f) for f in zfs["family"].tolist()]

    p201b80_cids = set(sorted(p201_cids, key=lambda c: hashlib.sha256(c.encode("ascii")).hexdigest())[:80])
    p201b80_sha = hashlib.sha256(",".join(sorted(p201b80_cids)).encode("ascii")).hexdigest()

    fit_cid_set = set(fit_cids)
    fit_fam_norm = {norm_fam(f) for f in fit_fams}
    p201_fam_norm = {norm_fam(f) for f in p201_fams}
    fitsel_fam_norm = {norm_fam(f) for f in fitsel_fams}

    col_idx: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
    row_idx: Dict[int, List[int]] = defaultdict(list)
    pair_idx: Dict[int, List[int]] = defaultdict(list)
    pair_df: Counter = Counter()
    for i, f_rec in enumerate(fit_sketches):
        f_rec["tset"] = set(f_rec["tk"].tolist())
        for j, s in enumerate(f_rec["spec"]):
            for h in s.tolist():
                col_idx[h].append((i, j))
        for h in f_rec["rows"].tolist():
            row_idx[h].append(i)
        for p in f_rec["pairs"]:
            pair_idx[p].append(i)
    for x in fit_sketches + [t[2] for t in p201_sketches]:
        for p in x["pairs"]:
            pair_df[p] += 1

    names_fit = [name_tokens(f) for f in fit_fams]
    p201_per_ctx: Dict[str, Dict] = {}
    for cid, fam, q_sk in p201_sketches:
        res = screen_one(q_sk, fit_sketches, col_idx, row_idx, pair_idx, pair_df, names_fit, name_tokens(fam))
        res["family"] = fam
        res["in_p201b80"] = cid in p201b80_cids
        p201_per_ctx[cid] = res

    tiers_201 = dict(Counter(v["tier"] for v in p201_per_ctx.values()))
    tiers_80 = dict(Counter(v["tier"] for c, v in p201_per_ctx.items() if c in p201b80_cids))
    copy_201 = sorted(c for c, v in p201_per_ctx.items() if v["tier"] == "copy")
    copy_80 = sorted(c for c in copy_201 if c in p201b80_cids)
    poss_201 = sorted(c for c, v in p201_per_ctx.items() if v["tier"] == "possible")
    poss_80 = sorted(c for c in poss_201 if c in p201b80_cids)

    excl_doc = json.loads(EXCLUSION_FITSEL.read_text())

    doc = {
        "schema": "jope.h459.fit_icl_overlap_audit/v1",
        "created_utc": time.strftime("%FT%TZ", time.gmtime()),
        "wall_s": round(time.time() - t0, 3),
        "audit_labels_read": 0,
        "input_sha256": shas,
        "fit_icl_bank": {
            "path": str(FIT_BANK),
            "tables": len(fit_cids),
            "distinct_norm_families": len(fit_fam_norm),
        },
        "p201b": {
            "path": str(P201B_BANK),
            "contexts": len(p201_cids),
            "p201b80_cid_list_sha256": p201b80_sha,
            "cid_overlap_201": len(fit_cid_set & set(p201_cids)),
            "cid_overlap_80": len(fit_cid_set & p201b80_cids),
            "norm_family_overlap_201": len(fit_fam_norm & p201_fam_norm),
            "h425_screen_201": {
                "tiers": tiers_201,
                "copy_count": len(copy_201),
                "possible_count": len(poss_201),
                "copy_cids": copy_201,
                "possible_cids": poss_201,
            },
            "h425_screen_p201b80": {
                "tiers": tiers_80,
                "copy_count": len(copy_80),
                "possible_count": len(poss_80),
                "copy_cids": copy_80,
                "possible_cids": poss_80,
            },
            "per_context": p201_per_ctx,
        },
        "fitsel": {
            "path": str(FITSEL_BANK),
            "contexts": len(fitsel_cids),
            "cid_overlap_106": len(fit_cid_set & set(fitsel_cids)),
            "norm_family_overlap_106": len(fit_fam_norm & fitsel_fam_norm),
            "h431_exclusion_fitsel_tier_counts": excl_doc.get("tier_counts"),
        },
    }
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with out_p.open("x") as fh:
        json.dump(doc, fh, indent=2, sort_keys=True)
        fh.write("\n")
    os.chmod(out_p, 0o444)
    print(
        json.dumps(
            {
                "out": str(out_p),
                "sha256": sha256_file(out_p),
                "p201b_cid_overlap": doc["p201b"]["cid_overlap_201"],
                "p201b_fam_overlap": doc["p201b"]["norm_family_overlap_201"],
                "p201b_tiers_201": tiers_201,
                "p201b_tiers_80": tiers_80,
                "fitsel_tiers_106": excl_doc.get("tier_counts"),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
