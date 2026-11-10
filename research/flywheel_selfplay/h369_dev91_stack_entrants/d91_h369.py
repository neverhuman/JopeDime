#!/usr/bin/env python3
"""H369: once-score the H364 XSTACK-2L entrant on the fixed holdout board dev91-audit-v1 (owner request 2026-11-05).

The entrant is fully frozen before any dev91 read: H364's code (h354_core / h364_core / lib_oof_h354, sha-equal to
origin/v1/next) and its part-B DESIGN.json (f052a59f35, c 0.35 / lambda 0.10 / delta 0 / tau inf). Nothing is chosen
on dev91. Per (context, search seed) the X0 input is H335 Stage V's stored AGG.g50.CV3@12 output (its stored weights;
the board entrant h335.H335_AGG), so the 2L value differs from h335.H335_AGG only by the second level.

Data use:
  * GP side: the H335 Stage V records, memo and envelopes (pulled read-only from xbabe0; sha-identical).
  * Library AUDIT predictions: the STORED dev91-audit-v1 records of catboost / lightgbm / xgboost / random_forest
    (preds sha-checked); no reference model or board entrant is refit, re-run or rescored.
  * Library SEARCH OOF: lib_oof_h354.py fits on dev91 SEARCH rows only (the 2-fold SEARCH split), in the H354 bwrap
    sandbox. AUDIT labels are used only for the final once-score, as for every board entrant.
Guard: importing h364_core installs the H317 guard v2. This module points the guard's Stage V unlock at its own LOCK
(DATA/stage_v/LOCK.json) and create-once receipt ns/receipts/H369_LOCK_sha256.txt (the H335 pattern); `unlock` is
one-shot, later commands use resume_stage_v. Sealed / holdout / DL-Test paths stay refused after the unlock.

Commands: unlock | prepare | align | score | rows
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import h364_core as H  # noqa: E402  (installs the H317 guard via h354_core -> common_h317)

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402

import numpy as np  # noqa: E402

K, C = H.K, H.C
WAVE = "H369"
DATA = Path(os.environ.get("H369_DATA", "/home/ubuntu/jope-data/h369-d91"))
C.DATA_ROOT = DATA
C.LOCK_RECEIPT = Path("/home/ubuntu/jope_plan/ns/receipts/H369_LOCK2_sha256.txt")  # attempt 1 receipt: PREREG section 6
BOARD = Path("/home/ubuntu/jope-data/h335-stagev/refboard/dev91-audit-v1")
BOARD_SHA256 = "40cb4828cf5c9a0d2d9cc0f32b7900819c992ec9a58c5d18f83ac84b29fa4069"
GPV = DATA / "gpv"
STAGE = "V_AGG"
SEEDS = (335101, 335102)
N = 91
ENTRANT = "h364.XSTACK_2L"
X0_BOARD = "h335.H335_AGG"
DESIGN_COMMIT = "f052a59f35"
TOL_ALIGN = 1e-9  # H354 alignment tolerance
TOL_X0 = H.TOL  # 1e-6 (H364 part B section 7.1)


def board() -> dict:
    b = json.loads((BOARD / "BOARD.json").read_text())
    if b["board"] != "dev91-audit-v1" or b["board_sha256"] != BOARD_SHA256 or len(b["contexts"]) != N:
        raise SystemExit("BOARD.json is not dev91-audit-v1")
    return b


def fam_of() -> dict:
    return {c["cid"]: c["family_id"] for c in board()["contexts"]}


def design() -> "H.Design":
    d = json.loads((HERE / "DESIGN.json").read_text())
    chosen = H.Design.from_json(d["chosen"])
    if chosen.key() != d["chosen_key"] or chosen not in H.grid():
        raise SystemExit("DESIGN.json: chosen design is not a PREREG grid point")
    return chosen


def ctx_of(cid: str, seed: int) -> "K.Context":
    return K.load_context(cid, fam_of(), runs=(GPV,), stage=STAGE, seed=seed)


def cmd_unlock(_a) -> None:
    C.unlock_stage_v()
    print(json.dumps(C.guard_report()))


def cmd_prepare(_a) -> None:
    C.resume_stage_v()
    lock = json.loads((DATA / "stage_v" / "LOCK.json").read_text())
    for p, sha in lock["inputs_sha256"].items():  # dev91-path inputs: checked only after the unlock
        if K.sha256_file(Path(p)) != sha:
            raise SystemExit(f"input changed after LOCK: {p}")
    b = board()
    cids = [c["cid"] for c in b["contexts"]]
    env_sha = {c["cid"]: c["envelope_sha256"] for c in b["contexts"]}
    cfg = GPV / "config.json"
    if not cfg.exists():
        K.jdump(cfg, {"cids": cids, "source": "H335 Stage V xbabe0 stagev-root/gp (records, memo, envs)",
                      "stage": STAGE, "seeds": list(SEEDS)})
    inp = DATA / "d" / "inputs"
    if inp.exists():
        raise SystemExit(f"{inp} exists (create-once)")
    stored = {m: {json.loads(x)["cid"]: json.loads(x) for x in (BOARD / "scores" / f"{m}.jsonl").read_text()
                  .splitlines() if x.strip()} for m in H.CLASSICAL}
    for cid in cids:
        cx = [ctx_of(cid, s) for s in SEEDS]
        if len({c.search.sha256 for c in cx}) != 1 or len({c.audit.sha256 for c in cx}) != 1:
            raise AssertionError(f"{cid[:12]}: the two search seeds use different envelopes")
        src_s = GPV / "gproot_AGG" / "envs" / f"e_{cx[0].search.sha256}.w.jtt1"
        src_a = BOARD / "envs" / f"{cid}.jtt1"
        if K.sha256_file(src_a) != env_sha[cid]:
            raise AssertionError(f"{cid[:12]}: board envelope sha differs from BOARD.json")
        for sub, src in (("search", src_s), ("audit", src_a)):
            (inp / sub).mkdir(parents=True, exist_ok=True)
            (inp / sub / f"{cid}.jtt1").write_bytes(src.read_bytes())
        for m in H.CLASSICAL:
            r = stored[m][cid]
            rec = {"cid": cid, "model": m, "status": r["status"], "preds_sha256": r.get("preds_sha256"), "pa": None,
                   "rmse_audit": r.get("rmse_audit"), "base_rmse_audit": r.get("base_rmse_audit"),
                   "utility": r.get("utility")}
            if r["status"] == "completed":
                rec["pa"] = np.frombuffer((BOARD / "preds" / m / f"{cid}.f64").read_bytes(), dtype="<f8").tolist()
            (inp / "stored" / m).mkdir(parents=True, exist_ok=True)
            K.jdump(inp / "stored" / m / f"{cid}.json", rec)
    files = sorted(str(p.relative_to(inp)) for p in inp.rglob("*") if p.is_file())
    (inp / "MANIFEST.sha256").write_text("".join(f"{K.sha256_file(inp / f)}  {f}\n" for f in files))
    print(f"prepare: {len(cids)} contexts, {len(files)} files")


def cmd_align(_a) -> None:
    """Before any library fit: GP records / board envelope / stored board predictions describe the same rows."""
    C.resume_stage_v()
    b = board()
    out, bad = [], []
    for c in b["contexts"]:
        cid = c["cid"]
        for s in SEEDS:
            ctx = ctx_of(cid, s)
            dev = K.alignment_checks(ctx, BOARD / "envs" / f"{cid}.jtt1", BOARD)
            for m in H.CLASSICAL:
                st = json.loads((DATA / "d" / "inputs" / "stored" / m / f"{cid}.json").read_text())
                if st["status"] != "completed":
                    continue
                pa = np.asarray(st["pa"], np.float64)
                rm = float(np.sqrt(np.mean((pa - ctx.ya) ** 2))) if len(pa) == len(ctx.ya) else math.inf
                dev[f"stored_rmse:{m}"] = abs(rm - st["rmse_audit"]) / max(st["rmse_audit"], 1e-12)
                dev[f"stored_base:{m}"] = abs(st["base_rmse_audit"] - ctx.ba) / max(ctx.ba, 1e-12)
            worst = max(dev.values())
            out.append({"cid": cid, "seed": s, "worst": worst, **dev})
            if not worst <= TOL_ALIGN:
                bad.append((cid[:12], s, max(dev, key=dev.get), worst))
    K.jdump(DATA / "align.json", {"tol": TOL_ALIGN, "rows": out, "bad": bad, "pass": not bad})
    print(json.dumps({"rows": len(out), "bad": bad[:10], "n_bad": len(bad),
                      "worst": max(r["worst"] for r in out)}))
    if bad:
        raise SystemExit("alignment failed")


def score_one(cid: str, lib_dir: str) -> dict:
    C.resume_stage_v()
    d = design()
    x0b = {json.loads(x)["cid"]: json.loads(x) for x in (BOARD / "scores" / f"{X0_BOARD}.jsonl").read_text()
           .splitlines() if x.strip()}[cid]
    per = []
    for i, s in enumerate(SEEDS):
        ctx = ctx_of(cid, s)
        ls, la = H.lib_from_dir(ctx, Path(lib_dir))
        rec = H.evaluate(ctx, ls, la, [d])
        rec["seed"] = s
        rec["x0_board"] = float(x0b["per_seed"][i])
        rec["x0_board_du"] = abs(rec["x0_u"] - rec["x0_board"])
        rec["library"] = {n: {"gate": bool(ls[n].gate), "ok": bool(ls[n].ok), "audit_status": la[f"lib:{n}"].status,
                              "audit_utility": float(la[f"lib:{n}"].utility)} for n in H.CLASSICAL}
        per.append(rec)
    return {"cid": cid, "family": per[0]["family"], "seeds": per}


def cmd_score(a) -> None:
    import concurrent.futures as cf

    C.resume_stage_v()
    cids = [c["cid"] for c in board()["contexts"]]
    design()
    lib = Path(a.lib)
    missing = [(n, c) for n in H.CLASSICAL for c in cids if not (lib / n / f"{c}.json").exists()]
    if missing:
        raise SystemExit(f"INCOMPLETE: library outputs missing for {len(missing)} (learner, context) pairs")
    if (DATA / "score.jsonl").exists():
        raise SystemExit("score.jsonl exists (create-once)")
    with cf.ProcessPoolExecutor(max_workers=a.jobs) as ex:
        recs = list(ex.map(score_one, cids, [str(lib)] * len(cids)))
    with open(DATA / "score.jsonl", "x") as f:
        for r in recs:
            f.write(json.dumps(r, sort_keys=True) + "\n")
    allp = [p for r in recs for p in r["seeds"]]
    worst = {"dlr": max(p["x0_check"]["dlr"] for p in allp), "du": max(p["x0_check"]["du"] for p in allp),
             "board_du": max(p["x0_board_du"] for p in allp)}
    ok = all(v <= TOL_X0 for v in worst.values())
    K.jdump(DATA / "x0_check.json", {"pass": ok, "tol": TOL_X0, "worst": worst, "units": len(allp)})
    print(f"score: {len(recs)} contexts x {len(SEEDS)} seeds, X0 check worst {worst}")
    if not ok:
        raise SystemExit("X0 harness check failed on dev91")


def cmd_rows(a) -> None:
    """The board record file for the new entrant (create-once) + a descriptive paired summary vs h335.H335_AGG."""
    C.resume_stage_v()
    b = board()
    d = design()
    env = {c["cid"]: c["envelope_sha256"] for c in b["contexts"]}
    if not json.loads((DATA / "x0_check.json").read_text())["pass"]:
        raise SystemExit("X0 check did not pass")
    recs = {json.loads(x)["cid"]: json.loads(x) for x in (DATA / "score.jsonl").read_text().splitlines()}
    if sorted(recs) != sorted(env):
        raise SystemExit("score.jsonl does not cover the board")
    out = Path(a.out)
    src = {"wave": WAVE, "entrant_wave": "H364", "design_key": d.key(), "design_commit": DESIGN_COMMIT,
           "design_sha256": K.sha256_file(HERE / "DESIGN.json"), "score_sha256": K.sha256_file(DATA / "score.jsonl"),
           "lock_sha256": C.LOCK_RECEIPT.read_text().split()[0], "seeds": list(SEEDS),
           "x0": f"{X0_BOARD} (H335 Stage V stored AGG.g50.CV3@12 output)",
           "library_audit": "stored dev91-audit-v1 records (catboost/lightgbm/xgboost/random_forest .v1)",
           "library_search": "lib_oof_h354.py SEARCH-only 2-fold OOF fits"}
    gain, n_act, n_units = {}, 0, 0
    with open(out, "x") as f:
        for cid in sorted(recs):
            ps = [float(p["designs"][d.key()]["u"]) for p in recs[cid]["seeds"]]
            x0 = [float(p["x0_board"]) for p in recs[cid]["seeds"]]
            n_act += sum(bool(p["designs"][d.key()]["active"]) for p in recs[cid]["seeds"])
            n_units += len(ps)
            gain[cid] = sum(ps) / len(ps) - sum(x0) / len(x0)
            f.write(json.dumps({"board": b["board"], "board_sha256": b["board_sha256"], "cid": cid,
                                "envelope_sha256": env[cid], "model_id": ENTRANT, "utility": sum(ps) / len(ps),
                                "per_seed": ps, "status": "completed", "harness_version": "h369-2l-v1",
                                "source": src}, sort_keys=True) + "\n")
    g = np.array(list(gain.values()))
    summ = {"entrant": ENTRANT, "vs": X0_BOARD, "contexts": len(g), "active_units": n_act, "units": n_units,
            "context_gain": {"mean": float(g.mean()), "min": float(g.min()), "max": float(g.max()),
                             "exact_ties": int((g == 0).sum()), "lt_minus_0p05": int((g < -0.05).sum())},
            "family": H.family_lcb(gain, {c: recs[c]["family"] for c in recs}),
            "rows_sha256": hashlib.sha256(out.read_bytes()).hexdigest()}
    K.jdump(DATA / "summary.json", summ)
    print(json.dumps(summ, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for n in ("unlock", "prepare", "align"):
        sub.add_parser(n)
    p = sub.add_parser("score")
    p.add_argument("--lib", default=str(DATA / "d" / "lib"))
    p.add_argument("--jobs", type=int, default=12)
    p = sub.add_parser("rows")
    p.add_argument("--out", required=True)
    a = ap.parse_args()
    {"unlock": cmd_unlock, "prepare": cmd_prepare, "align": cmd_align, "score": cmd_score, "rows": cmd_rows}[a.cmd](a)


if __name__ == "__main__":
    main()
