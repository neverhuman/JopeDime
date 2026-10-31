"""H95 — W8 resplit confirmation of H94's 45 apparent high-water marks (CPU, native training-CV fits).

H94 revealed 45 new programs (20 DL-Train contexts) whose single original-cut fitness beat the admitted pool's best.
W8/H63 showed that original-cut wins often lose their sign on resplits of the same training rows. H95 applies the H72
common confirmation endpoint, which no H94 decision used, before anyone treats these programs as improvements.

Fixed before any H95 fit:
* Challengers: every record in H94's apparent_hwm.json (sha256 pinned). Incumbent: the context champion DSL in
  champions.json. In 4 of the 20 contexts the pool's best program (whose DSL is not stored) is better than the champion
  by <= 0.0007 fitness; those contexts are flagged, not dropped.
* Cuts per program: the original cut (identity check against the recorded H94 fitness and the stored pool utility),
  W8 decision seeds 701-703, and the H72 confirmation seeds 801-803 (balanced permutation of the original fold bytes;
  rows, features and targets unchanged). Evaluator and envelope are the merged H72 helpers.
* Primary (H72 common endpoint): a challenger is CONFIRMED iff mean over 801-803 of (u_challenger - u_incumbent) > 0.
  Secondary: MATERIAL iff that mean > 0.001 (the common gate's LCB threshold), because many apparent margins are the
  size of one complexity term (0.0005); the same rule on 701-703; COMPLEXITY_ONLY iff the original-cut log ratio
  equals the incumbent's within 1e-9 (the gain is only the parenthesis-count penalty).
* Execution errors make a challenger INCOMPLETE; they are never mapped to a utility.
* Elo: family-grouped Bradley-Terry (canonical elo_tracker) over contexts, contenders INCUMBENT (anchor 1500) and
  BEST_CHALLENGER (the context's best challenger by original cut, scored by its mean 801-803 utility).
No DL-Val, outer-test, sealed or holdout row is read; DOPE is not loaded; no GPU.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
H72_PATH = HERE.parent / "h72_w8_control_prereg" / "w8_control.py"
H72_SHA = "9e812c43eebbc54e38f59d45ca511b1e78b1936f23f7ebb646beffadefd24f0e"
ELO_SOURCE = REPO / "python/src/jope/north_star/elo_tracker.py"
ELO_SHA = "9ece98326c32453179b0c765eb45944b68ca9de944d2828a1d42dee1401d1835"
HWM_SHA = "8452d19f83979b1a21188d0f893dc8ffee3fea22bad91bb64ff45629f81895b8"
DECISION, CONFIRM = (701, 702, 703), (801, 802, 803)
CUTS = (None,) + DECISION + CONFIRM
MATERIAL, TIE, IDENT = 0.001, 1e-4, 1e-9
FIT_CAP = 500


def load(path: Path, want: str, name: str):
    if hashlib.sha256(path.read_bytes()).hexdigest() != want:
        raise SystemExit(f"{name} bytes differ from the pinned source")
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m; spec.loader.exec_module(m)
    return m


def u_of(r: dict | None) -> float | None:
    """Measured utility, or None for an execution error (never mapped to a scientific loss)."""
    if r is None or r.get("status") != "completed":
        return None
    return -float(r["fitness"]) if r.get("cv_gate_passed") else -2.0


def job_id(role: str, context: str, sha: str, seed) -> str:
    return hashlib.sha256(f"h95|{role}|{context}|{sha}|{seed}".encode()).hexdigest()[:32]


def plan(hwm: list[dict], champions: dict) -> list[dict]:
    jobs, seen = [], set()
    for h in hwm:
        c = h["context_sha256"]; inc = champions[c]
        for role, dsl, sha in (("inc", inc["dsl"], inc["program_sha256"]), ("chal", h["dsl"], h["program_sha256"])):
            if hashlib.sha256(dsl.encode()).hexdigest() != sha:
                raise SystemExit(f"DSL/sha mismatch for {role} in {c}")
            for s in CUTS:
                j = job_id(role, c, sha, s)
                if j not in seen:
                    seen.add(j); jobs.append(dict(job_id=j, stage=f"h95-{role}", context=c, sha=sha, seed=s, dsl=dsl))
    return jobs


def mean_diff(ch: dict, inc: dict, seeds) -> float | None:
    if any(ch.get(s) is None or inc.get(s) is None for s in seeds):
        return None
    return float(np.mean([ch[s] - inc[s] for s in seeds]))


def classify(hwm: list[dict], champions: dict, res: dict, pool_u: dict) -> list[dict]:
    rows = []
    for h in hwm:
        c = h["context_sha256"]; inc = champions[c]
        U = lambda role, sha: {s: u_of(res.get(job_id(role, c, sha, s))) for s in CUTS}
        uc, ui = U("chal", h["program_sha256"]), U("inc", inc["program_sha256"])
        rc = res.get(job_id("chal", c, h["program_sha256"], None)) or {}
        ri = res.get(job_id("inc", c, inc["program_sha256"], None)) or {}
        d8, d7 = mean_diff(uc, ui, CONFIRM), mean_diff(uc, ui, DECISION)
        complete = d8 is not None and d7 is not None and uc[None] is not None and ui[None] is not None
        rows.append(dict(
            context=c, family=h["family_id"], program_sha256=h["program_sha256"], found_by=h["found_by"],
            h94_fitness=h["fitness_original_cut"], pool_best_fitness=h["pool_best_fitness"],
            champion_is_pool_best=bool(abs(pool_u[c]["champion"] - pool_u[c]["best"]) <= IDENT),
            status="completed" if complete else "incomplete",
            identity_challenger=bool(uc[None] is not None and abs(uc[None] + h["fitness_original_cut"]) <= IDENT),
            identity_incumbent=bool(ui[None] is not None and abs(ui[None] - pool_u[c]["champion"]) <= IDENT),
            original_gain=(uc[None] - ui[None]) if complete else None,
            complexity_only=bool(complete and rc.get("cv_gate_passed") and ri.get("cv_gate_passed")
                                 and abs(rc["logratio"] - ri["logratio"]) <= IDENT),
            u_challenger={str(k): v for k, v in uc.items()}, u_incumbent={str(k): v for k, v in ui.items()},
            mean_gain_801_803=d8, mean_gain_701_703=d7,
            confirmed=bool(complete and d8 > 0), material=bool(complete and d8 > MATERIAL),
            confirmed_701_703=bool(complete and d7 > 0)))
    return rows


def summarize(rows: list[dict], elo=None) -> dict:
    done = [r for r in rows if r["status"] == "completed"]
    count = lambda k, rs=done: sum(r[k] for r in rs)
    by_arm = defaultdict(list)
    for r in done:
        by_arm[r["found_by"]].append(r)
    out = dict(challengers=len(rows), completed=len(done), incomplete=len(rows) - len(done),
               contexts=len({r["context"] for r in rows}), families=len({r["family"] for r in rows}),
               identity_challenger=count("identity_challenger"), identity_incumbent=count("identity_incumbent"),
               confirmed=count("confirmed"), material=count("material"), confirmed_701_703=count("confirmed_701_703"),
               complexity_only=count("complexity_only"),
               confirmed_and_not_complexity_only=sum(r["confirmed"] and not r["complexity_only"] for r in done),
               sign_kept_both_seed_sets=sum(r["confirmed"] and r["confirmed_701_703"] for r in done),
               contexts_with_material=len({r["context"] for r in done if r["material"]}),
               families_with_material=len({r["family"] for r in done if r["material"]}),
               mean_gain_801_803=dict(mean=float(np.mean([r["mean_gain_801_803"] for r in done])) if done else None,
                                      median=float(np.median([r["mean_gain_801_803"] for r in done])) if done else None),
               by_found_by={a: dict(n=len(v), confirmed=count("confirmed", v), material=count("material", v))
                            for a, v in sorted(by_arm.items())})
    if elo is not None and done:
        best = {}
        for r in done:
            if r["context"] not in best or r["original_gain"] > best[r["context"]]["original_gain"]:
                best[r["context"]] = r
        recs = [elo.DatasetEvalRecord(c, r["family"], "regression",
                                      {"INCUMBENT": float(np.mean([r["u_incumbent"][str(s)] for s in CONFIRM])),
                                       "BEST_CHALLENGER": float(np.mean([r["u_challenger"][str(s)] for s in CONFIRM]))})
                for c, r in sorted(best.items())]
        out["family_elo_801_803"] = elo.compute_family_grouped_bradley_terry_elo(
            recs, anchor_contender="INCUMBENT", anchor_elo=1500.0, tie_margin=TIE, prior=0.01, n_iters=400)
    out["verdict"] = (f"{out['material']}/{out['completed']} MATERIAL-CONFIRMED (mean 801-803 gain > {MATERIAL}); "
                      f"{out['confirmed']}/{out['completed']} confirmed by the H72 endpoint (> 0)")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hwm", type=Path, required=True, help="H94 apparent_hwm.json (sha256 8452d19f...)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--timeout", type=int, default=600)
    a = ap.parse_args()
    h72 = load(H72_PATH, H72_SHA, "h72_run")
    if h72.sha(h72.WORKER) != h72.PINS["worker"]:
        raise SystemExit("evaluator hash mismatch")
    if hashlib.sha256(a.hwm.read_bytes()).hexdigest() != HWM_SHA:
        raise SystemExit("apparent_hwm.json differs from H94's published object")
    hwm = json.loads(a.hwm.read_text())
    champions = {r["context_sha256"]: r for r in json.loads((h72.INPUT / "champions.json").read_text())}
    pool_u = {}
    for c in sorted({h["context_sha256"] for h in hwm}):
        p = np.load(h72.INPUT / "pools" / f"{c}.npz", allow_pickle=False)
        sha = [s.decode() for s in p["program_sha256"]]; g = p["gate"].astype(bool); f = p["fitness"].astype(np.float64)
        u = np.where(g, -f, -2.0)
        pool_u[c] = dict(best=float(u.max()), champion=float(u[sha.index(champions[c]["program_sha256"])]))
    a.out.mkdir(parents=True, exist_ok=True)
    jobs = plan(hwm, champions)
    (a.out / "plan.json").write_text(json.dumps([{k: v for k, v in j.items() if k != "dsl"} for j in jobs], indent=1))
    spent = h72.run_jobs(jobs, a.out / "results.jsonl", a.out / "scratch", a.workers, a.timeout, 0, cap=FIT_CAP)
    res = h72.load_results(a.out / "results.jsonl")
    rows = classify(hwm, champions, res, pool_u)
    with open(a.out / "rows.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True) + "\n")
    out = summarize(rows, load(ELO_SOURCE, ELO_SHA, "h95_elo"))
    rs = list(res.values())
    out.update(run_id=a.run_id, fits_planned=len(jobs), fits_run_this_invocation=spent,
               fits_failed=sum(r["status"] == "failed" for r in rs), fits_timeout=sum(r["status"] == "timeout" for r in rs),
               native_wall_s=round(sum(r["wall_s"] for r in rs), 1), native_cpu_s=round(sum(r["cpu_s"] for r in rs), 1),
               champion_not_pool_best_contexts=sum(1 for v in pool_u.values() if abs(v["champion"] - v["best"]) > IDENT),
               source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), h72_source_sha256=H72_SHA,
               hwm_sha256=HWM_SHA, gpu_seconds=0)
    (a.out / "metrics.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("challengers", "completed", "confirmed", "material", "complexity_only",
                                          "sign_kept_both_seed_sets", "identity_challenger", "identity_incumbent", "verdict")}, indent=1))


if __name__ == "__main__":
    main()
