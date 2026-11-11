#!/usr/bin/env python3
"""H422: per-context cache of the stored stack matrices. No worker run, no GP engine, no table is opened.

  cache_h422.py --run-dir DIR --tag TAG --cohorts A,B --out DIR [--union DIR:TAG ...] [--memo-dir DIR] [--procs N]

Per context of config_<TAG>.json (create-once OUT/<cid>.npz):
  from the stored npz (sha-checked against its H416 record): Q, y, lab (5-fold out-of-fold SEARCH predictions,
  targets, fold labels), BAG / REF (AUDIT predictions of the fold-model bag and of the refit, with the deploy's
  fallbacks: bag column missing -> refit, refit failed -> SEARCH mean), ya, ba, ym, shas;
  recomputed with the stored meta_h416 code: the bootstrap-Caruana weights of the stored design (w_sxb), checked
  against the record where the record stores them (weights and AUDIT value);
  --union: the stored columns of other variants of the same contexts (same SEARCH rows, folds and AUDIT rows,
  asserted) are appended after the first m0 columns, and w_union = the same builder on all columns;
  --memo-dir (development only): per candidate the 2-fold search utility, complexity, wall seconds and head, from
  the memo's "search" entries. The memo's "audit" and "akf*" entries are never parsed.
--cohorts lists the cohorts the panel may contain; any other cohort stops the run. The DEV panel is cached only by
the apply step, after the stacker is frozen and published.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import meta_h416 as M  # noqa: E402

SXB_KEY = "P3+LIB3.SXB"
_KIND = re.compile(r'"kind":\s*"search"')


def head_of(dsl: str) -> str:
    """Last top-level stage of a (pipe X ...) program; "portfolio" for a portfolio program."""
    s = dsl.strip()
    if s.startswith("(portfolio"):
        return "portfolio"
    depth, start, last = 0, None, None
    for i, ch in enumerate(s):
        if ch == "(":
            depth += 1
            if depth == 2:
                start = i
        elif ch == ")":
            if depth == 2 and start is not None:
                last = s[start + 1:i]
            depth -= 1
    return re.split(r"[\s)]", last, maxsplit=1)[0] if last else "none"


def bcar(Q: np.ndarray, y: np.ndarray, cid: str) -> np.ndarray:
    """meta_h416.meta_weights("BCAR", ...) (the same call the stored design makes)."""
    return M.meta_weights("BCAR", Q, y, cid)


def last_result(p: Path) -> dict:
    rs = [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
    rs = [r for r in rs if r.get("type") == "result"]
    if len(rs) != 1:
        raise SystemExit(f"{p}: {len(rs)} result records")
    return rs[0]


def stored(run_dir: str, tag: str, cid: str):
    sd = Path(run_dir) / "stages" / f"H416_{tag}"
    rp = sd / f"{cid}__confirm.jsonl"   # a design-mode panel (H416 P201e) stores <cid>__dev.jsonl
    rec = last_result(rp if rp.exists() else sd / f"{cid}__dev.jsonl")
    p = Path(run_dir) / f"h416_{tag}_npz" / f"{cid}.npz"
    if hashlib.sha256(p.read_bytes()).hexdigest() != rec["npz"]["sha256"]:
        raise SystemExit(f"{p}: npz sha differs from its record")
    z = np.load(p, allow_pickle=False)
    m, na = z["Q"].shape[1], len(z["ya"])
    mean = np.full(na, float(z["ym"]))
    REF = np.column_stack([z["PA"][:, i] if bool(z["pa_ok"][i]) else mean for i in range(m)])
    BAG = np.column_stack([z["AKF"][:, i] if np.all(np.isfinite(z["AKF"][:, i])) else REF[:, i] for i in range(m)])
    return rec, {"Q": z["Q"].astype(np.float64), "y": z["y"].astype(np.float64), "lab": z["lab"],
                 "ya": z["ya"].astype(np.float64), "ba": float(z["ba"]), "ym": float(z["ym"]), "BAG": BAG, "REF": REF,
                 "shas": [str(s) for s in z["shas"]]}


def one(job) -> dict:
    run_dir, tag, union, memo_dir, out_dir, cid = job
    out = Path(out_dir) / f"{cid}.npz"
    if out.exists():
        return {"cid": cid, "skipped": True}
    t0 = time.process_time()
    rec, a = stored(run_dir, tag, cid)
    m0 = len(a["shas"])
    w = bcar(a["Q"], a["y"], cid)
    u_re = M.utility(0.5 * a["BAG"] @ w + 0.5 * a["REF"] @ w, a["ya"], a["ba"])[0]
    st = rec.get("outputs", {}).get(SXB_KEY, {})
    wdiff, udiff, u_rec = -1.0, -1.0, float("nan")
    if st.get("spec", {}).get("w") is not None and not st.get("invalid"):
        ws = np.zeros(m0)
        for s, x in zip(st["spec"]["members"], st["spec"]["w"]):
            ws[a["shas"].index(s)] = x
        wdiff, u_rec = float(np.abs(ws - w).max()), float(st["value"])
        udiff = abs(u_re - u_rec)
    shas = list(a["shas"])
    for udir, utag in union:
        _, b = stored(udir, utag, cid)
        if not (np.array_equal(b["y"], a["y"]) and np.array_equal(b["lab"], a["lab"])
                and np.array_equal(b["ya"], a["ya"]) and b["ba"] == a["ba"]):
            raise SystemExit(f"{cid}: variant {utag} is not on the same rows")
        new = [i for i, s in enumerate(b["shas"]) if s not in shas]
        shas += [b["shas"][i] for i in new]
        for k in ("Q", "BAG", "REF"):
            a[k] = np.concatenate([a[k], b[k][:, new]], axis=1)
    w_union = np.zeros(0)
    if union:
        w_union = bcar(a["Q"], a["y"], cid)
    extra = {}
    if memo_dir:
        need, meta = set(shas[:m0]), {}
        with open(Path(memo_dir) / f"{cid}.split.jsonl") as f:
            for line in f:
                if not _KIND.search(line):
                    continue
                k = line.find('"sha"')
                s = re.search(r'"sha":\s*"([0-9a-f]{64})"', line[k:k + 90] if k >= 0 else "")
                if s is None or s.group(1) not in need or s.group(1) in meta:
                    continue
                e = json.loads(line)
                if e.get("kind") != "search" or e.get("status") != "completed":
                    continue
                meta[e["sha"]] = (float(e["utility"]), float(e.get("complexity", 0.0)), float(e.get("wall_s", 0.0)),
                                  head_of(e["dsl"]))
        nan = (float("nan"), float("nan"), float("nan"), "unknown")
        rows = [meta.get(s, nan) for s in shas[:m0]]
        extra = {"search_u": np.array([r[0] for r in rows]), "complexity": np.array([r[1] for r in rows]),
                 "wall_s": np.array([r[2] for r in rows]), "head": np.array([r[3] for r in rows])}
    tmp = out.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, Q=a["Q"], y=a["y"], lab=a["lab"], BAG=a["BAG"].astype(np.float64),
                        REF=a["REF"].astype(np.float64), ya=a["ya"], ba=a["ba"], ym=a["ym"], shas=np.array(shas),
                        m0=m0, w_sxb=w, w_union=w_union, u_record=u_rec, u_recomputed=u_re,
                        family=np.array(rec["family_id"]), **extra)
    tmp.rename(out)
    return {"cid": cid, "m0": m0, "m": len(shas), "w_diff": wdiff, "u_diff": udiff, "cpu_s": time.process_time() - t0}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--cohorts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--union", action="append", default=[])
    ap.add_argument("--memo-dir")
    ap.add_argument("--procs", type=int, default=16)
    a = ap.parse_args()
    cfg = json.loads((Path(a.run_dir) / f"config_{a.tag}.json").read_text())
    cids = cfg["cids"]
    allowed = set(a.cohorts.split(","))
    import apex_core_meta as AM

    rows = AM.rows()
    seen = sorted({rows[c]["cohort"] if c in rows else "DL-Train" for c in cids})
    if not set(seen) <= allowed:
        raise SystemExit(f"panel holds cohorts {seen}; allowed here: {sorted(allowed)}")
    union = [tuple(u.rsplit(":", 1)) for u in a.union]
    Path(a.out).mkdir(parents=True, exist_ok=True)
    with Pool(a.procs) as pool:
        res = pool.map(one, [(a.run_dir, a.tag, union, a.memo_dir, a.out, c) for c in cids], chunksize=1)
    done = [r for r in res if not r.get("skipped")]
    summ = {"panel": cfg.get("panel"), "tag": a.tag, "cohorts": seen, "contexts": len(cids), "written": len(done),
            "union": [f"{d}:{t}" for d, t in union],
            "columns_median": float(np.median([r["m"] for r in done])) if done else None,
            "records_with_stored_weights": int(sum(r["w_diff"] >= 0 for r in done)),
            "max_w_diff_vs_record": max([r["w_diff"] for r in done], default=-1.0),
            "max_u_diff_vs_record": max([r["u_diff"] for r in done], default=-1.0),
            "cpu_s": float(sum(r["cpu_s"] for r in done)), "run_dir": a.run_dir}
    print(json.dumps(summ))
    with open(Path(a.out) / f"cache_{a.tag}.json", "x") as f:
        json.dump(summ, f, sort_keys=True)


if __name__ == "__main__":
    main()
