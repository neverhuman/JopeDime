#!/usr/bin/env python3
"""H354 Phase D (TRAIN P201b, development): alignment checks, the X0 harness check, the library arms, statistics.

  phase_d_h354.py align            # no fit, no score: SEARCH/AUDIT/frontier envelope alignment for all 201 contexts
  phase_d_h354.py x0               # re-stack the stored members: must reproduce the stored H335 weights and utility
  phase_d_h354.py arms --lib DIR   # X_CL / X_LLM / X_ALL from the library OOF + AUDIT predictions in DIR
  phase_d_h354.py decide           # statistics, Elo and the PREREG §3.4 LOCK rule

Outputs go to $H354_DATA/D/ (records) and are summarised into the wave directory at closeout.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import h354_core as K  # noqa: E402  (installs the H317 guard)

OUT = K.DATA / "D"
TOL = 1e-9


def _stats():
    if K.sha256_file(K.H335_STATS) != K.H335_STATS_SHA256:
        raise RuntimeError("h335_stats.py pin mismatch")
    spec = importlib.util.spec_from_file_location("h335_stats", K.H335_STATS)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def cmd_prepare(_a) -> None:
    """Copy exactly what the sandboxed library runner may read into $H354_DATA/D/inputs (create-once; sha-listed):
    SEARCH envelopes, AUDIT envelopes, the stored reference-board AUDIT predictions of the 4 classical learners, and the
    stored H335 frontier AUDIT prediction sha256 per (cid, entrant). TRAIN contexts only (the guard is installed)."""
    inp = OUT / "inputs"
    if inp.exists():
        raise SystemExit(f"{inp} exists (create-once)")
    fam = K.family_map()
    cids = K.p201b_cids()
    fsha: dict = {}
    for line in (K.FRONTIER_T / "frontier_results.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r["status"] == "completed":
            fsha.setdefault(r["cid"], {})[r["entrant"]] = r["audit_preds_sha256"]
    stored = {m: {json.loads(x)["cid"]: json.loads(x) for x in (K.REFBOARD_P201B / "scores" / f"{m}.jsonl")
                  .read_text().splitlines()} for m in K.CLASSICAL}
    for cid in cids:
        ctx = K.load_context(cid, fam)
        for sub, src in (("search", K.run_of(cid) / "gproot_AGG" / "envs" / f"e_{ctx.search.sha256}.w.jtt1"),
                         ("audit", K.FRONTIER_T / "envs" / f"{cid}.jtt1")):
            (inp / sub).mkdir(parents=True, exist_ok=True)
            (inp / sub / f"{cid}.jtt1").write_bytes(src.read_bytes())
        for m in K.CLASSICAL:
            r = stored[m][cid]
            rec = {"cid": cid, "model": m, "status": r["status"], "preds_sha256": r.get("preds_sha256"), "pa": None,
                   "rmse_audit": r.get("rmse_audit"), "utility": r.get("utility")}
            if r["status"] == "completed":
                pa = np.frombuffer((K.REFBOARD_P201B / "preds" / m / f"{cid}.f64").read_bytes(), dtype="<f8")
                rec["pa"] = pa.tolist()
            (inp / "stored" / m).mkdir(parents=True, exist_ok=True)
            K.jdump(inp / "stored" / m / f"{cid}.json", rec)
    K.jdump(inp / "frontier_sha.json", {c: fsha[c] for c in cids})
    files = sorted(str(p.relative_to(inp)) for p in inp.rglob("*") if p.is_file())
    (inp / "MANIFEST.sha256").write_text("".join(f"{K.sha256_file(inp / f)}  {f}\n" for f in files))
    print(f"prepare: {len(cids)} contexts, {len(files)} files")


def cmd_align(_a) -> None:
    fam = K.family_map()
    rows, bad = [], []
    for cid in K.p201b_cids():
        ctx = K.load_context(cid, fam)
        dev = K.alignment_checks(ctx, K.FRONTIER_T / "envs" / f"{cid}.jtt1")
        rows.append({"cid": cid, **dev, "n_members": len(ctx.members), "n_search": int(len(ctx.search.y)),
                     "n_audit": int(len(ctx.ya)), "single": bool(ctx.stored["spec"].get("fallback_single"))})
        if max(dev.values()) > TOL:
            bad.append(cid)
    K.jdump(OUT / "align.json", {"contexts": len(rows), "bad": bad, "tol": TOL,
                                 "max": {k: max(r[k] for r in rows) for k in rows[0] if k not in (
                                     "cid", "single", "n_members", "n_search", "n_audit")},
                                 "single_member_stacks": sum(r["single"] for r in rows), "rows": rows})
    print(f"align: {len(rows)} contexts, {len(bad)} misaligned")
    if bad:
        raise SystemExit(3)


def cmd_x0(_a) -> None:
    """Harness check: the frozen stacker on the stored members, stored order, must give the stored weights and the
    stored AUDIT utility (= the h335.H335_AGG value on the reference board) for every context."""
    fam = K.family_map()
    refs = {json.loads(line)["cid"]: json.loads(line)["utility"]
            for line in (K.REFBOARD_P201B / "scores" / "h335.H335_AGG.jsonl").read_text().splitlines()}
    rows, worst = [], {"w": 0.0, "u": 0.0, "u_board": 0.0}
    for cid in K.p201b_cids():
        ctx = K.load_context(cid, fam)
        spec = K.stack_arm(ctx, {}, "X0")
        st = ctx.stored["spec"]
        dw = max(float(np.max(np.abs(spec.w_audit - np.asarray(st["w_audit"])))),
                 float(np.max(np.abs(spec.w_fold0 - np.asarray(st["w_fold0"])))),
                 float(np.max(np.abs(spec.w_fold1 - np.asarray(st["w_fold1"])))))
        out = K.audit_arm(ctx, spec, ctx.audit_by_sha)
        du = abs(out["utility"] - float(ctx.stored["raw"]["utility"]))
        db = abs(out["utility"] - float(refs[cid]))
        worst = {"w": max(worst["w"], dw), "u": max(worst["u"], du), "u_board": max(worst["u_board"], db)}
        rows.append({"cid": cid, "dw": dw, "du": du, "du_board": db, "utility": out["utility"]})
    ok = worst["w"] <= TOL and worst["u"] <= TOL and worst["u_board"] <= TOL
    K.jdump(OUT / "x0.json", {"pass": ok, "tol": TOL, "worst": worst, "rows": rows})
    print(f"x0: pass={ok} worst={worst}")
    if not ok:
        raise SystemExit(4)


def load_library(ctx: "K.Context", lib_dir: Path):
    """Library OOF (SEARCH) and AUDIT predictions for one context, from lib_oof_h354.py outputs."""
    ls, la = {}, {}
    for name in K.LIBRARY:
        f = lib_dir / name / f"{ctx.cid}.json"
        r = json.loads(f.read_text()) if f.exists() else {"status": "missing"}
        arr = lambda k: None if r.get(k) is None else np.asarray(r[k], np.float64)  # noqa: E731
        ls[name] = K.lib_search_result(ctx, name, arr("p0"), arr("p1"), r.get("oof_status", r["status"]),
                                       r.get("error", ""))
        la[f"lib:{name}"] = K.lib_audit_result(ctx, name, arr("pa"), r.get("audit_status", r["status"]))
    return ls, la


def arms_one(cid: str, lib_dir: str) -> dict:
    fam = K.family_map()
    ctx = K.load_context(cid, fam)
    ls, la = load_library(ctx, Path(lib_dir))
    abs_ = dict(ctx.audit_by_sha)
    abs_.update(la)
    rec = {"cid": cid, "family": ctx.family, "arms": {}, "library": {
        n: {"gate": bool(ls[n].gate), "ok": bool(ls[n].ok), "search_fitness": float(ls[n].fitness),
            "audit_utility": float(la[f"lib:{n}"].utility), "audit_status": la[f"lib:{n}"].status}
        for n in K.LIBRARY}}
    for arm in K.ARMS:
        spec = K.stack_arm(ctx, ls, arm)  # SEARCH only: no AUDIT array reaches the stacker
        rec["arms"][arm] = K.audit_arm(ctx, spec, abs_)
    return rec


def cmd_arms(a) -> None:
    import concurrent.futures as cf
    cids = K.p201b_cids()
    with cf.ProcessPoolExecutor(max_workers=a.jobs) as ex:
        recs = list(ex.map(arms_one, cids, [a.lib] * len(cids)))
    x0 = {row["cid"]: row["utility"] for row in json.loads((OUT / "x0.json").read_text())["rows"]}
    if set(x0) != set(cids):
        raise SystemExit("x0.json does not cover exactly the P201b contexts")
    dw = max(abs(r["arms"]["X0"]["utility"] - x0[r["cid"]]) for r in recs)
    if dw > TOL:
        raise SystemExit(f"X0 inside arms differs from the x0 harness check by {dw}")
    with open(OUT / "arms.jsonl", "w") as f:
        for r in recs:
            f.write(json.dumps(r, sort_keys=True) + "\n")
    print(f"arms: {len(recs)} contexts")


COST_RANK = {"X_LLM": 0, "X_CL": 1, "X_ALL": 2}


def pick_arm(gap_se: dict):
    """PREREG §3.4 1-SE rule. gap_se[arm] = {"gap": family-mean u(best) - u(arm), "se": paired family SE}.
    Eligible: gap <= se (the best arm itself has gap 0). Pick: fewest library members, then lower library CPU."""
    elig = [a for a in K.CANDIDATE_ARMS if gap_se[a]["gap"] <= gap_se[a]["se"]]
    return elig, min(elig, key=lambda a: (len(K.ARMS[a]), COST_RANK[a]))


def cmd_decide(_a) -> None:
    S = _stats()
    fam = K.family_map()
    recs = [json.loads(line) for line in (OUT / "arms.jsonl").read_text().splitlines()]
    u = {arm: {r["cid"]: float(r["arms"][arm]["utility"]) for r in recs} for arm in K.ARMS}
    lib = {n: {r["cid"]: float(r["library"][n]["audit_utility"]) for r in recs} for n in K.LIBRARY}
    fm = {arm: S.family_mean(u[arm], fam) for arm in K.ARMS}
    res = {"family_mean_u": fm, "vs_X0": {}, "paired_se_vs_best": {}}
    for arm in K.CANDIDATE_ARMS:
        res["vs_X0"][arm] = {"primary": S.primary_test(u[arm], u["X0"], fam), "headline": S.headline(u[arm], u["X0"], fam)}
    best = max(K.CANDIDATE_ARMS, key=lambda a: (fm[a], a))
    for arm in K.CANDIDATE_ARMS:
        _f, D = S.family_contrast(u[best], u[arm], fam, None)
        se = float(np.std(D, ddof=1) / math.sqrt(len(D))) if arm != best else 0.0
        res["paired_se_vs_best"][arm] = {"gap": float(D.mean()), "se": se}
    elig, pick = pick_arm(res["paired_se_vs_best"])
    passes = res["vs_X0"][pick]["primary"]["pass"]
    res.update({"best": best, "eligible": elig, "locked_arm": pick if passes else None,
                "decision": "LOCK" if passes else "NOT_ADVANCED",
                "rule": "PREREG §3.4: 1-SE rule among X_CL/X_LLM/X_ALL, then the H335 primary vs X0 must pass"})
    field = {f"h354.{arm}": u[arm] for arm in K.ARMS}
    field.update({f"lib.{n}": lib[n] for n in K.LIBRARY})
    res["elo"] = S.elo_panel(field, fam, anchor="h354.X0")
    K.jdump(OUT / "decision.json", res)
    print(json.dumps({k: res[k] for k in ("family_mean_u", "best", "eligible", "locked_arm", "decision")}, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prepare")
    sub.add_parser("align")
    sub.add_parser("x0")
    p = sub.add_parser("arms")
    p.add_argument("--lib", default=str(K.DATA / "D" / "lib"))
    p.add_argument("--jobs", type=int, default=16)
    sub.add_parser("decide")
    a = ap.parse_args()
    {"prepare": cmd_prepare, "align": cmd_align, "x0": cmd_x0, "arms": cmd_arms, "decide": cmd_decide}[a.cmd](a)


if __name__ == "__main__":
    main()
