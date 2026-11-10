#!/usr/bin/env python3
"""H317 GP block, post hoc: split-half reliability of AUDIT utilities (PREREG §2.3 diagnostic, amendment A3).

Read-only over a finished (pulled) Gate H0 GP run: <run_dir>/{config.json, decisions.json, stages/**} and the engine
memo <memo_dir>/<cid>.split.jsonl, which stores every AUDIT prediction vector. For every declared output
(BANK_NN1; H0.4 scheme x {full, bm}; H0.5 arm x gen {0, 3, 6, 10}; H0.6 configs) it rebuilds the AUDIT prediction
exactly as scored (single program: its AUDIT predictions; stack: sum_i w_i * member AUDIT predictions with the
SEARCH-target mean for crashed members), re-checks the recorded AUDIT utility, and scores the two deterministic
y-stratified halves of the AUDIT rows (`gp_engine_h317.audit_half_masks`, the same halves Stage V records use).
Reported: per output and per decision contrast, the across-family Pearson r of half-0 vs half-1 values, with the
Spearman-Brown full-slice reliability 2r / (1 + r). Nothing here selects anything; outputs were fixed long before.
Usage (xbabe2, inside sandbox_h317.sh): gp_split_half.py --run-dir R --memo-dir M --out OUT.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common_h317 as C  # noqa: E402

C.install_guard()

import numpy as np  # noqa: E402

import distributed_10gen_island_gp as D  # noqa: E402
import gp_engine_h317 as E  # noqa: E402
import gp_lab as L  # noqa: E402


def context_audit(cid: str) -> dict:
    env = C.read_jtt1(C.table_path(cid))
    search, audit, _ = C.search_and_audit_envelopes(cid, env)
    ya = audit.y[audit.folds == 1].copy()
    ysm = float(search.y.mean())
    return {
        "ya": ya,
        "ysm": ysm,
        "ba": float(np.sqrt(np.mean((ya - ysm) ** 2))),
        "audit_env_sha256": hashlib.sha256(audit.to_worker_bytes()).hexdigest(),
    }


def memo_audit_preds(
    memo_dir: Path, cid: str, env_sha: str
) -> Dict[str, Optional[np.ndarray]]:
    out: Dict[str, Optional[np.ndarray]] = {}
    p = memo_dir / f"{cid}.split.jsonl"
    with open(p) as f:
        for line in f:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if d.get("kind") != "audit" or d.get("env_sha256") != env_sha:
                continue
            ok = d.get("status") == "completed" and d.get("pa") is not None
            out[d["sha"]] = np.asarray(d["pa"], np.float64) if ok else None
    return out


def _u(pred: Optional[np.ndarray], ya: np.ndarray, ysm: float, mask=None) -> float:
    if pred is None:
        return 0.0
    m = np.ones(len(ya), bool) if mask is None else mask
    rm = float(np.sqrt(np.mean((pred[m] - ya[m]) ** 2)))
    rb = float(np.sqrt(np.mean((ya[m] - ysm) ** 2)))
    return C.audit_utility_from_rmse(rm, rb)


def output_pred(
    entry: dict, preds: Dict[str, Optional[np.ndarray]], ctx: dict
) -> Tuple[Optional[np.ndarray], str]:
    """AUDIT prediction of one declared output record (raw / spec as written by gp_lab)."""
    raw = entry.get("raw") or {}
    spec = entry.get("spec") or {}
    if raw.get("n_members", 1) > 1 and spec.get("members"):
        w = np.asarray(spec["w_audit"], np.float64)
        keep = [i for i, x in enumerate(w) if x > 1e-9]
        wk = w[keep] / w[keep].sum()
        cols = [preds.get(spec["members"][i]) for i in keep]
        if all(c is None for c in cols):
            return None, "stack_all_substituted"
        mat = np.column_stack(
            [c if c is not None else np.full(len(ctx["ya"]), ctx["ysm"]) for c in cols]
        )
        return mat @ wk, "stack"
    sha = raw.get("sha")
    if sha is None and spec.get("members") and len(spec["members"]) == 1:
        sha = spec["members"][0]
    if sha is None:
        return None, "no_output"
    return preds.get(sha), "single"


def reliability(
    h0: Dict[str, float], h1: Dict[str, float], fam_of: Dict[str, str]
) -> dict:
    f0, f1 = C.family_means(h0, fam_of), C.family_means(h1, fam_of)
    fams = sorted(set(f0) & set(f1))
    a, b = np.array([f0[f] for f in fams]), np.array([f1[f] for f in fams])
    r = (
        float(np.corrcoef(a, b)[0, 1])
        if len(fams) > 2 and a.std() > 0 and b.std() > 0
        else float("nan")
    )
    return {
        "families": len(fams),
        "r_half": r,
        "spearman_brown": float(2 * r / (1 + r)) if np.isfinite(r) else None,
        "mean_half0": float(a.mean()) if len(a) else None,
        "mean_half1": float(b.mean()) if len(b) else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--memo-dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    run_dir, memo_dir = Path(args.run_dir), Path(args.memo_dir)
    out_path = Path(args.out)
    if run_dir in out_path.parents or memo_dir in out_path.parents:
        raise SystemExit("--out must be outside the run / memo dirs (read-only inputs)")
    cfg = json.loads((run_dir / "config.json").read_text())
    dec = json.loads((run_dir / "decisions.json").read_text())
    cids, seeds = cfg["pilot"], cfg["seeds"]
    scheme, arm = dec["H0.4"]["used"], dec["H0.5"]["used"]
    rows = C.train_rows()
    fam_of = {r["context_sha256"]: r["family_id"] for r in rows}
    store = L.Store(run_dir)
    recs = {
        "bank_nn1": L.collect(store, "bank_nn1", cids, seeds),
        "h04": L.collect(store, "h04", cids, seeds),
        "h05": L.collect(store, "h05", cids, seeds, scheme),
        "h06": L.collect(store, "h06", cids, seeds, scheme, arm),
    }
    halves: Dict[str, Dict[Tuple[str, object], List[float]]] = {}
    checks = {"max_abs_full_diff": 0.0, "n_outputs": 0, "kinds": {}}
    ctx_cache: Dict[str, dict] = {}

    def score(key: str, cid: str, seed, entry: dict) -> None:
        if cid not in ctx_cache:
            ctx_cache[cid] = context_audit(cid)
            ctx_cache[cid]["preds"] = memo_audit_preds(
                memo_dir, cid, ctx_cache[cid]["audit_env_sha256"]
            )
        cx = ctx_cache[cid]
        pred, kind = output_pred(entry, cx["preds"], cx)
        u_full = _u(pred, cx["ya"], cx["ysm"])
        rec_v = entry.get("value")
        if rec_v is not None and not entry.get("invalid"):
            checks["max_abs_full_diff"] = max(
                checks["max_abs_full_diff"], abs(float(rec_v) - u_full)
            )
        m0, m1 = E.audit_half_masks(cid, cx["ya"])
        halves.setdefault(key, {})[(cid, seed)] = [
            _u(pred, cx["ya"], cx["ysm"], m0),
            _u(pred, cx["ya"], cx["ysm"], m1),
        ]
        checks["n_outputs"] += 1
        checks["kinds"][kind] = checks["kinds"].get(kind, 0) + 1

    for r in recs["bank_nn1"]:
        score(
            "BANK_NN1",
            r["cid"],
            "na",
            {"raw": r["audit"], "value": r["value"], "invalid": r["invalid"]},
        )
    for r in recs["h04"]:
        for sch, m in r["schemes"].items():
            for tag in ("full", "bm"):
                a = m[f"audit_{tag}"]
                score(
                    f"H04.{sch}.{tag}",
                    r["cid"],
                    r["seed"],
                    {"raw": a["raw"], "value": a["value"], "invalid": a["invalid"]},
                )
    for r in recs["h05"]:
        for a_name, a in r["arms"].items():
            for gen in D.AUDIT_GENS:
                x = a[f"g{gen}"]
                score(
                    f"H05.{a_name}.g{gen}",
                    r["cid"],
                    r["seed"],
                    {"raw": x["raw"], "value": x["value"], "invalid": x["invalid"]},
                )
    for r in recs["h06"]:
        single = r["configs"]["single"]
        for k, v in r["configs"].items():
            src = single if v.get("alias") else v
            score(
                f"H06.{k}",
                r["cid"],
                r["seed"],
                {
                    "raw": src.get("raw"),
                    "spec": src.get("spec"),
                    "value": v.get("value"),
                    "invalid": v.get("invalid"),
                },
            )

    def seed_avg(key: str, h: int) -> Dict[str, float]:
        acc: Dict[str, List[float]] = {}
        for (cid, _), hv in halves[key].items():
            acc.setdefault(cid, []).append(hv[h])
        return {c: float(np.mean(v)) for c, v in acc.items()}

    per_output = {
        k: reliability(seed_avg(k, 0), seed_avg(k, 1), fam_of) for k in sorted(halves)
    }

    def gain(a: str, b: str, h: int) -> Dict[str, float]:
        x, y = seed_avg(a, h), seed_avg(b, h)
        return {c: x[c] - y[c] for c in set(x) & set(y)}

    contrasts = {}
    for ch in L.H04_CHALLENGERS:
        contrasts[f"H04.{ch}.bm-H04.A.bm"] = (f"H04.{ch}.bm", "H04.A.bm")
    contrasts["H05.evoB.g10-H05.evoA.g10"] = ("H05.evoB.g10", "H05.evoA.g10")
    contrasts["H06.CV5-H06.single"] = ("H06.CV5", "H06.single")
    locked = dec.get("H0.6", {}).get("used")
    if locked and locked != "single" and f"H06.{locked}" in halves:
        contrasts[f"H06.{locked}-H06.single"] = (f"H06.{locked}", "H06.single")
    per_contrast = {
        k: reliability(gain(a, b, 0), gain(a, b, 1), fam_of)
        for k, (a, b) in contrasts.items()
    }
    rep = {
        "run_dir": str(run_dir),
        "config_pilot_sha256": cfg.get("pilot_sha256"),
        "decisions": dec,
        "half_rule": "gp_engine_h317.audit_half_masks: y-sorted AUDIT rows, one of each consecutive pair per half "
        "(seed stream H317-audit-half ⊕ cid)",
        "checks": checks,
        "per_output": per_output,
        "per_contrast": per_contrast,
        "guard": C.guard_report(),
    }
    C.write_json(out_path, rep)
    print(
        json.dumps(
            {
                "n_outputs": checks["n_outputs"],
                "max_abs_full_diff": checks["max_abs_full_diff"],
                "contrasts": {k: v["spearman_brown"] for k, v in per_contrast.items()},
            },
            default=str,
        )
    )


if __name__ == "__main__":
    main()
