#!/usr/bin/env python3
"""H357 (PREREG §3-§4), H354-machinery side: XSTACK-CL vs X0 on the fresh TRAIN panel P201c.

Runs on xbabe2 from a staged read-only code root that holds the H354 files exactly as on origin/v1/next (h354_core,
phase_d_h354, lib_oof_h354, sandbox_h354.sh) plus the H357 files (stage_h357.sh). TRAIN only. Commands:
  prepare   the P201c GP run (gp_c_h357.py on xbabe0, pulled back by run_c_h357.sh) -> $H357_DATA/inputs (create-once,
            MANIFEST.sha256): search/<cid>.jtt1 = the GP SEARCH worker envelope; audit/<cid>.jtt1 = the GP AUDIT envelope
            in the train-only form (fold 0 = all SEARCH rows, fold 1 = AUDIT rows) for the library SEARCH -> AUDIT fits
  (library) sandbox_h354.sh $H357_DATA python3 lib_oof_h354.py --audit fit --learners <the 4 classical learners>
  arms      X0 harness check (the frozen stacker reproduces the stored AGG.g50.CV3@12 weights and AUDIT utility) and X_CL
  decide    primary, adoption-grade secondary, verdict, headline, AUDIT-Elo -> $H357_DATA/verdict.json
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import h354_core as K  # noqa: E402  (installs the H317 guard)
import phase_d_h354 as PD  # noqa: E402

import argparse  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
from typing import Dict, List  # noqa: E402

import numpy as np  # noqa: E402

C = K.C
WAVE = "H357"
ARM = "X_CL"  # fixed by the H357 PREREG (chosen after the H354 P201b results; P201c is the confirmation panel)
LIBRARY = K.ARMS[ARM]
SEED_C = 357101
STAGE_C = "C_AGG"
TOL = 1e-9
OUT = Path(os.environ.get("H357_DATA", "/home/ubuntu/jope-data/h357-xc"))
GP_RUN_C = Path(os.environ.get("H357_GP_RUN", str(K.H335_DATA / "runs" / "H357_C_xbabe0")))
PANEL_C = K.H335_DATA / "h357" / "P201c.json"
P201C_CIDS_SHA256 = "36fbf6a92204487a31f2533905a57393e9c317fa7484ab7cbde5026e34468f6d"
BOOT_SEED = 335150  # h335_core.BOOT_SEED: the H335 adoption-grade secondary, unchanged


def _panel() -> dict:
    import hashlib

    pan = json.loads(PANEL_C.read_text())
    if hashlib.sha256("\n".join(pan["cids"]).encode()).hexdigest() != P201C_CIDS_SHA256 or len(pan["cids"]) != 194:
        raise SystemExit("P201c panel differs from the PREREG panel")
    cfg = json.loads((GP_RUN_C / "config.json").read_text())
    if cfg["cids"] != pan["cids"] or cfg["seeds"] != [SEED_C] or cfg["gens"] != 50 or cfg["checkpoints"] != [50]:
        raise SystemExit("P201c GP run config does not match the panel / PREREG")
    if set(pan["cids"]) & set(K.p201b_cids()):
        raise AssertionError("P201c overlaps P201b")
    return pan


def _ctx(cid: str, fam: Dict[str, str]) -> "K.Context":
    return K.load_context(cid, fam, runs=(GP_RUN_C,), stage=STAGE_C, seed=SEED_C)


def cmd_prepare(_a) -> None:
    pan = _panel()
    inp = OUT / "inputs"
    if inp.exists():
        raise SystemExit(f"{inp} exists (create-once)")
    (inp / "search").mkdir(parents=True)
    (inp / "audit").mkdir(parents=True)
    for cid in pan["cids"]:
        ctx = _ctx(cid, pan["family_of"])
        src = GP_RUN_C / "gproot_AGG" / "envs" / f"e_{ctx.search.sha256}.w.jtt1"
        (inp / "search" / f"{cid}.jtt1").write_bytes(src.read_bytes())
        (inp / "audit" / f"{cid}.jtt1").write_bytes(K.train_only_bytes(ctx.audit, ctx.audit.folds))
    files = sorted(str(p.relative_to(inp)) for p in inp.rglob("*") if p.is_file())
    (inp / "MANIFEST.sha256").write_text("".join(f"{K.sha256_file(inp / f)}  {f}\n" for f in files))
    print(f"prepare: {len(pan['cids'])} contexts, {len(files)} files")


def arms_one(cid: str, fam: Dict[str, str], lib_dir: str) -> dict:
    ctx = _ctx(cid, fam)
    ls, la = PD.load_library(ctx, Path(lib_dir))
    abs_ = dict(ctx.audit_by_sha)
    abs_.update(la)
    x0 = K.stack_arm(ctx, {}, "X0")
    st = ctx.stored["spec"]
    dw = max(float(np.max(np.abs(x0.w_audit - np.asarray(st["w_audit"])))),
             float(np.max(np.abs(x0.w_fold0 - np.asarray(st["w_fold0"])))),
             float(np.max(np.abs(x0.w_fold1 - np.asarray(st["w_fold1"])))))
    out0 = K.audit_arm(ctx, x0, ctx.audit_by_sha)
    du = abs(out0["utility"] - float(ctx.stored["raw"]["utility"]))
    return {"cid": cid, "family": ctx.family, "x0_check": {"dw": dw, "du": du},
            "arms": {"X0": out0, ARM: K.audit_arm(ctx, K.stack_arm(ctx, ls, ARM), abs_)},  # SEARCH-only weights
            "library": {n: {"gate": bool(ls[n].gate), "ok": bool(ls[n].ok), "search_fitness": float(ls[n].fitness),
                            "audit_utility": float(la[f"lib:{n}"].utility), "audit_status": la[f"lib:{n}"].status}
                        for n in LIBRARY}}


def cmd_arms(a) -> None:
    import concurrent.futures as cf

    pan = _panel()
    cids = pan["cids"]
    missing = [(n, c) for n in LIBRARY for c in cids if not (Path(a.lib) / n / f"{c}.json").exists()]
    if missing:
        raise SystemExit(f"library outputs missing for {len(missing)} (learner, context) pairs")
    with cf.ProcessPoolExecutor(max_workers=a.jobs) as ex:
        recs = list(ex.map(arms_one, cids, [pan["family_of"]] * len(cids), [a.lib] * len(cids)))
    worst = {"dw": max(r["x0_check"]["dw"] for r in recs), "du": max(r["x0_check"]["du"] for r in recs)}
    with open(OUT / "arms.jsonl", "w") as f:
        for r in recs:
            f.write(json.dumps(r, sort_keys=True) + "\n")
    K.jdump(OUT / "x0_check.json", {"pass": worst["dw"] <= TOL and worst["du"] <= TOL, "tol": TOL, "worst": worst})
    print(f"arms: {len(recs)} contexts, X0 check worst {worst}")
    if worst["dw"] > TOL or worst["du"] > TOL:
        raise SystemExit("X0 harness check failed on P201c")


def verdict(primary: dict, adoption: dict) -> str:
    """H357 PREREG §4 verdict map (INCOMPLETE is decided before any statistic is computed)."""
    if primary["pass"]:
        return "XSTACK_CL_CONFIRMED_ADOPTION_GRADE" if adoption["gate"]["pass"] else "XSTACK_CL_CONFIRMED"
    if primary["inferior"]:
        return "XSTACK_CL_INFERIOR"
    return "XSTACK_CL_NOT_CONFIRMED"


def cmd_decide(_a) -> None:
    S = PD._stats()
    pan = _panel()
    fam = pan["family_of"]
    recs = [json.loads(line) for line in (OUT / "arms.jsonl").read_text().splitlines()]
    if sorted(r["cid"] for r in recs) != sorted(pan["cids"]) or not json.loads(
            (OUT / "x0_check.json").read_text())["pass"]:
        K.jdump(OUT / "verdict.json", {"verdict": "INCOMPLETE", "reason": "arms.jsonl / X0 check incomplete"})
        raise SystemExit("INCOMPLETE")
    u = {a: {r["cid"]: float(r["arms"][a]["utility"]) for r in recs} for a in ("X0", ARM)}
    lib = {n: {r["cid"]: float(r["library"][n]["audit_utility"]) for r in recs} for n in LIBRARY}
    primary = S.primary_test(u[ARM], u["X0"], fam)
    adoption = C.robustness_summary(u[ARM], u["X0"], fam, BOOT_SEED)
    adoption["gate"] = C.contract_advance_gate(adoption, "lcb95", margin=0.001, signflip_alpha=S.ALPHA)
    field = {f"h357.{ARM}": u[ARM], "h357.X0": u["X0"]}
    field.update({f"lib.{n}": lib[n] for n in LIBRARY})
    lw: List[float] = [float(r["arms"][ARM]["library_weight"]) for r in recs]
    d = np.array([u[ARM][c] - u["X0"][c] for c in pan["cids"]])
    res = {"wave": WAVE, "panel": "P201c", "contexts": len(recs), "families": len(set(fam.values())), "arm": ARM,
           "library": list(LIBRARY), "primary": primary, "adoption_secondary": adoption,
           "verdict": verdict(primary, adoption), "headline": S.headline(u[ARM], u["X0"], fam),
           "family_mean_u": {a: S.family_mean(u[a], fam) for a in u},
           "context_deltas": {"lt_minus_0p05": int((d < -0.05).sum()), "min": float(d.min()), "max": float(d.max())},
           "library_weight": {"mean": float(np.mean(lw)), "median": float(np.median(lw)),
                              "share_contexts_gt0": float(np.mean(np.asarray(lw) > 0))},
           "elo": S.elo_panel(field, fam, anchor="h357.X0")}
    K.jdump(OUT / "verdict.json", res)
    print(json.dumps({k: res[k] for k in ("verdict", "family_mean_u", "library_weight", "context_deltas")}, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prepare")
    p = sub.add_parser("arms")
    p.add_argument("--lib", default=str(OUT / "lib"))
    p.add_argument("--jobs", type=int, default=16)
    sub.add_parser("decide")
    a = ap.parse_args()
    {"prepare": cmd_prepare, "arms": cmd_arms, "decide": cmd_decide}[a.cmd](a)


if __name__ == "__main__":
    main()
