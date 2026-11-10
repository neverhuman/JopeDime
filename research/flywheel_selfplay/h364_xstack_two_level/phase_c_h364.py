#!/usr/bin/env python3
"""H364 (PREREG §2-§5), xbabe2 side: XSTACK-2L vs X0 on the fresh TRAIN panel P201d, with the part-B constants.

Runs from a staged read-only code root (stage_h364.sh): the H354 files exactly as on origin/v1/next plus this
directory's H364 files, including DESIGN.json (part B). TRAIN only. Commands:
  prepare   the P201d GP run (gp_c_h364.py on xbabe0, pulled back by run_c_h364.sh) -> $H364_DATA/inputs (create-once,
            MANIFEST.sha256): search/<cid>.jtt1 = the GP SEARCH worker envelope; audit/<cid>.jtt1 = the GP AUDIT envelope
            in the train-only form (fold 0 = all SEARCH rows, fold 1 = AUDIT rows) for the library SEARCH -> AUDIT fits
  (library) sandbox_h354.sh $H364_DATA python3 lib_oof_h354.py --audit fit --learners <the 4 classical learners>
  score     X0 harness check (stored weights reproduce the stored SEARCH log-ratios and AUDIT utility) and the AUDIT
            utilities of X0, 2L (DESIGN.json), 2L_noguard (tau = inf) and each library learner alone -> score.jsonl
  decide    primary, adoption-grade secondary, verdict, headline, AUDIT-Elo -> $H364_DATA/verdict.json
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import h364_core as H  # noqa: E402  (installs the H317 guard via h354_core)
import phase_d_h354 as PD  # noqa: E402

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
from collections import Counter  # noqa: E402
from typing import Dict, Tuple  # noqa: E402

import numpy as np  # noqa: E402

K, C = H.K, H.C
WAVE = "H364"
SEED_C = 364101
STAGE_C = "C_AGG"
N = 190
TOL = H.TOL
BOOT_SEED = 335150  # h335_core.BOOT_SEED: the H335 adoption-grade secondary, unchanged
OUT = Path(os.environ.get("H364_DATA", "/home/ubuntu/jope-data/h364-2l"))
GP_RUN_C = K.H335_DATA / "runs" / "H364_C_xbabe0"
PANEL_C = K.H335_DATA / "h364" / "P201d.json"
PANEL_P201C = K.H335_DATA / "h357" / "P201c.json"
P201D_CIDS_SHA256 = "9a1ace6cfd9abdf6c436d980b61d98b59825e56ea3719cd67911a0f2aa25de98"
DESIGN_FILE = HERE / "DESIGN.json"


def _panel() -> dict:
    pan = json.loads(PANEL_C.read_text())
    if hashlib.sha256("\n".join(pan["cids"]).encode()).hexdigest() != P201D_CIDS_SHA256 or len(pan["cids"]) != N:
        raise SystemExit("P201d panel differs from the PREREG panel")
    cfg = json.loads((GP_RUN_C / "config.json").read_text())
    if cfg["cids"] != pan["cids"] or cfg["seeds"] != [SEED_C] or cfg["gens"] != 50 or cfg["checkpoints"] != [50]:
        raise SystemExit("P201d GP run config does not match the panel / PREREG")
    if set(pan["cids"]) & (set(K.p201b_cids()) | set(json.loads(PANEL_P201C.read_text())["cids"])):
        raise AssertionError("P201d overlaps P201b or P201c")
    return pan


def designs() -> Tuple["H.Design", "H.Design"]:
    """(2L, 2L_noguard) from the published part-B DESIGN.json."""
    d = json.loads(DESIGN_FILE.read_text())
    chosen = H.Design.from_json(d["chosen"])
    if chosen.key() != d["chosen_key"] or chosen not in H.grid():
        raise SystemExit("DESIGN.json: chosen design is not a PREREG grid point")
    return chosen, H.Design(chosen.c, chosen.lam, chosen.delta, math.inf)


def cmd_prepare(_a) -> None:
    pan = _panel()
    inp = OUT / "inputs"
    if inp.exists():
        raise SystemExit(f"{inp} exists (create-once)")
    (inp / "search").mkdir(parents=True)
    (inp / "audit").mkdir(parents=True)
    for cid in pan["cids"]:
        ctx = K.load_context(cid, pan["family_of"], runs=(GP_RUN_C,), stage=STAGE_C, seed=SEED_C)
        src = GP_RUN_C / "gproot_AGG" / "envs" / f"e_{ctx.search.sha256}.w.jtt1"
        (inp / "search" / f"{cid}.jtt1").write_bytes(src.read_bytes())
        (inp / "audit" / f"{cid}.jtt1").write_bytes(K.train_only_bytes(ctx.audit, ctx.audit.folds))
    files = sorted(str(p.relative_to(inp)) for p in inp.rglob("*") if p.is_file())
    (inp / "MANIFEST.sha256").write_text("".join(f"{K.sha256_file(inp / f)}  {f}\n" for f in files))
    print(f"prepare: {len(pan['cids'])} contexts, {len(files)} files")


def score_one(cid: str, fam: Dict[str, str], lib_dir: str) -> dict:
    chosen, noguard = designs()
    ctx = K.load_context(cid, fam, runs=(GP_RUN_C,), stage=STAGE_C, seed=SEED_C)
    ls, la = H.lib_from_dir(ctx, Path(lib_dir))
    rec = H.evaluate(ctx, ls, la, [chosen, noguard] if noguard != chosen else [chosen])
    rec["library"] = {n: {"gate": bool(ls[n].gate), "ok": bool(ls[n].ok), "audit_status": la[f"lib:{n}"].status,
                          "audit_utility": float(la[f"lib:{n}"].utility)} for n in H.CLASSICAL}
    return rec


def cmd_score(a) -> None:
    import concurrent.futures as cf

    pan = _panel()
    cids = pan["cids"]
    designs()
    missing = [(n, c) for n in H.CLASSICAL for c in cids if not (Path(a.lib) / n / f"{c}.json").exists()]
    if missing:
        raise SystemExit(f"INCOMPLETE: library outputs missing for {len(missing)} (learner, context) pairs")
    if (OUT / "score.jsonl").exists():
        raise SystemExit("score.jsonl exists (create-once)")
    with cf.ProcessPoolExecutor(max_workers=a.jobs) as ex:
        recs = list(ex.map(score_one, cids, [pan["family_of"]] * len(cids), [a.lib] * len(cids)))
    worst = {"dlr": max(r["x0_check"]["dlr"] for r in recs), "du": max(r["x0_check"]["du"] for r in recs)}
    with open(OUT / "score.jsonl", "w") as f:
        for r in recs:
            f.write(json.dumps(r, sort_keys=True) + "\n")
    K.jdump(OUT / "x0_check.json", {"pass": worst["dlr"] <= TOL and worst["du"] <= TOL, "tol": TOL, "worst": worst})
    print(f"score: {len(recs)} contexts, X0 check worst {worst}")
    if worst["dlr"] > TOL or worst["du"] > TOL:
        raise SystemExit("X0 harness check failed on P201d")


def verdict(primary: dict, adoption: dict) -> str:
    """H364 PREREG §5 verdict map (INCOMPLETE is decided before any statistic is computed)."""
    if primary["pass"]:
        return "XSTACK2L_CONFIRMED_ADOPTION_GRADE" if adoption["gate"]["pass"] else "XSTACK2L_CONFIRMED"
    if primary["inferior"]:
        return "XSTACK2L_INFERIOR"
    return "XSTACK2L_NOT_CONFIRMED"


def cmd_decide(_a) -> None:
    S = PD._stats()
    pan = _panel()
    fam = pan["family_of"]
    chosen, noguard = designs()
    recs = [json.loads(line) for line in (OUT / "score.jsonl").read_text().splitlines()]
    if sorted(r["cid"] for r in recs) != sorted(pan["cids"]) or not json.loads(
            (OUT / "x0_check.json").read_text())["pass"]:
        K.jdump(OUT / "verdict.json", {"verdict": "INCOMPLETE", "reason": "score.jsonl / X0 check incomplete"})
        raise SystemExit("INCOMPLETE")
    u = {"X0": {r["cid"]: float(r["x0_u"]) for r in recs},
         "2L": {r["cid"]: float(r["designs"][chosen.key()]["u"]) for r in recs}}
    if noguard != chosen:
        u["2L_noguard"] = {r["cid"]: float(r["designs"][noguard.key()]["u"]) for r in recs}
    primary = S.primary_test(u["2L"], u["X0"], fam)
    adoption = C.robustness_summary(u["2L"], u["X0"], fam, BOOT_SEED)
    adoption["gate"] = C.contract_advance_gate(adoption, "lcb95", margin=0.001, signflip_alpha=S.ALPHA)
    field = {f"h364.{k}": v for k, v in u.items()}
    field.update({f"lib.{n}": {r["cid"]: float(r["library"][n]["audit_utility"]) for r in recs} for n in H.CLASSICAL})
    d = np.array([u["2L"][c] - u["X0"][c] for c in pan["cids"]])
    sel = [r["designs"][chosen.key()] for r in recs]
    res = {"wave": WAVE, "panel": "P201d", "contexts": len(recs), "families": len(set(fam.values())),
           "design": chosen.as_json(), "design_key": chosen.key(), "primary": primary, "adoption_secondary": adoption,
           "verdict": verdict(primary, adoption), "headline": S.headline(u["2L"], u["X0"], fam),
           "family_mean_u": {a: S.family_mean(u[a], fam) for a in u},
           "context_deltas": {"lt_minus_0p05": int((d < -0.05).sum()), "min": float(d.min()), "max": float(d.max()),
                              "exact_ties": int((d == 0).sum())},
           "second_level": {"active_share": float(np.mean([s["active"] for s in sel])),
                            "mean_beta_sum": float(np.mean([s["beta_sum"] for s in sel])),
                            "mean_beta_sum_active": float(np.mean([s["beta_sum"] for s in sel if s["active"]]))
                            if any(s["active"] for s in sel) else 0.0,
                            "reasons": dict(Counter(s["reason"] for s in sel)),
                            "audit_substitutions": int(sum(s["n_substituted"] for s in sel))},
           "elo": S.elo_panel(field, fam, anchor="h364.X0")}
    if "2L_noguard" in u:
        res["noguard_vs_x0"] = {"primary": S.primary_test(u["2L_noguard"], u["X0"], fam),
                                "headline": S.headline(u["2L_noguard"], u["X0"], fam)}
    K.jdump(OUT / "verdict.json", res)
    print(json.dumps({k: res[k] for k in ("verdict", "family_mean_u", "second_level", "context_deltas")}, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prepare")
    p = sub.add_parser("score")
    p.add_argument("--lib", default=str(OUT / "lib"))
    p.add_argument("--jobs", type=int, default=16)
    sub.add_parser("decide")
    a = ap.parse_args()
    {"prepare": cmd_prepare, "score": cmd_score, "decide": cmd_decide}[a.cmd](a)


if __name__ == "__main__":
    main()
