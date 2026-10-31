"""H72 — W8 equal-cost control (preregistered in PREREG.md). DL-Train training rows only, CPU, native Rust 2-fold CV.

Stages (each idempotent, resumable by job id, results appended as JSONL):
  plan     : freeze contexts, frozen incumbents (from the H63/W8 receipts), proposal streams, shard membership.
  search   : original-cut fits of the first B unique proposals per context (shared by P0 and P1).
  verify   : P1 only -- 3 resplit fits (seeds 701-703) for each apparent win inside P1's budget prefix.
  confirm  : common endpoint -- seeds 801-803 for every promoted program (either policy) and for each incumbent.
  analyze  : policy outcomes, paired family-cluster bootstrap, Elo, cost accounting.
Every native call is counted; failures and timeouts are spent budget. CPU seconds are the child rusage
(user+sys) returned by os.wait4 for that worker process; wall seconds are measured separately.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import struct
import subprocess
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

INPUT = Path.home() / "jope-data/h50/input"
CACHE = Path.home() / "jope-data/h58/cache"
MICRO = Path.home() / "JopeDimeMicro"
WORKER = MICRO / "bin/bootstrap.s1main"
PINS = {
    "worker": "e3be938644722d14a1c1a0e82168518838c4d174cfe031076a339745c48870a8",
    "micro_rust_evaluator": "53422343699172b61ed69ed707015558f7cf1a344e5449f30e1eb9d708aef900",
    "w8_results": "f802fc43f3f0",  # prefix of H63 run w8-opus-20261031a results-shard0.jsonl sha256
    "w8_plan": "ec90ca128db1",  # prefix of H63 run w8-opus-20261031a plan-shard0.json sha256
    "champions_featurized": "ea020d489ec9a813e4dfb2b6fe45c126ff9f8d8a1b59d0ff9f8242cfe5e6efc0",
    "context_features": "75704e8279c19fee8a7843ccbea801e3cd4b67e0ee08d282b070ddae7ab82faf",
}
DECISION_SEEDS = (701, 702, 703)  # P1 confirmation (same as the H63 diagnostic)
CONFIRM_SEEDS = (801, 802, 803)  # common endpoint, independent of every decision
B = 40  # decision fits per context, identical for both policies
MARGIN = 1e-6  # live strict-improvement margin (automatic_promotion.py)
FIT_CAP = 1640  # remaining W8 ceiling (2,400 - 760 spent by H63)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utility(rec: dict) -> float:
    ok = rec.get("status") == "completed" and rec.get("cv_gate_passed")
    return -float(rec["fitness"]) if ok else -2.0


def s1_metrics(dsl: str, inner_cv: list[dict]) -> dict:
    """Verbatim logic of rust_evaluator.compute_s1_metrics_from_inner_cv."""
    if len(inner_cv) != 2:
        return {"status": "failed", "fitness": 100.0, "logratio": 0.0, "cv_gate_passed": False}
    ratios, wins = [], 0
    for fold_rec in inner_cv:
        m = (fold_rec.get("metrics") or {}).get("rmse")
        b = (fold_rec.get("baseline_metrics") or {}).get("rmse")
        if m is None or b is None or not math.isfinite(m) or not math.isfinite(b) or b <= 1e-12:
            return {"status": "failed", "fitness": 100.0, "logratio": 0.0, "cv_gate_passed": False}
        r = max(float(m) / float(b), 1e-12)
        ratios.append(r)
        wins += r <= 1.0
    lr0, lr1 = math.log(ratios[0]), math.log(ratios[1])
    mean_lr, tail_lr, sd_lr = 0.5 * (lr0 + lr1), max(lr0, lr1), 0.5 * abs(lr0 - lr1)
    gate = wins == 2 and math.exp(mean_lr) <= 0.90
    comp = dsl.count("(")
    fitness = (0.8 * mean_lr + 0.1 * tail_lr + 0.1 * sd_lr + 0.0005 * comp) if gate else (
        100.0 + max(mean_lr, 0.0) + max(tail_lr, 0.0) + max(sd_lr, 0.0))
    return {"status": "completed", "fitness": float(fitness), "logratio": float(mean_lr), "cv_gate_passed": bool(gate)}


def envelope(raw: bytes, seed: int | None) -> bytes:
    """training_cv_envelope (automatic_promotion.py) + optional balanced resplit of the fold bytes only."""
    if len(raw) < 17 or raw[:4] != b"JTT1" or raw[16] != 2:
        raise ValueError("invalid training packet")
    nt, nq, width = struct.unpack_from("<III", raw, 4)
    if nq or len(raw) != 17 + nt + nt * (width + 1) * 8:
        raise ValueError("expected training-only rows")
    out = bytearray(raw)
    if seed is not None:
        folds = np.frombuffer(raw, np.uint8, nt, 17).copy()
        out[17:17 + nt] = folds[np.random.default_rng(seed).permutation(nt)].tobytes()
    struct.pack_into("<I", out, 8, 10)
    out.extend(bytes(10 * (width + 1) * 8))
    return bytes(out)


def frozen_incumbents(w8_results: Path, w8_plan: Path) -> dict:
    """Per context: the best ORIGINAL-cut utility among the H63 champion and challenger programs (no new fits),
    with that program's H63 resplit utilities for seeds 701-703."""
    pairs = json.loads(w8_plan.read_text())["pairs"]
    rows = defaultdict(dict)
    for line in w8_results.read_text().splitlines():
        r = json.loads(line)
        rows[r["pair_index"]][(r["role"], r["seed"])] = r
    best = {}
    for i, p in enumerate(pairs):
        for role in ("incumbent", "challenger"):
            u0 = utility(rows[i][(role, None)])
            prog = dict(dsl=p[f"{role}_dsl"], sha=p[f"{role}_sha"], u_orig=u0,
                        u_701_703=[utility(rows[i][(role, s)]) for s in DECISION_SEEDS])
            c = p["context"]
            if c not in best or u0 > best[c]["u_orig"] or (u0 == best[c]["u_orig"] and prog["sha"] < best[c]["sha"]):
                best[c] = dict(prog, family=p["family"])
    return best


def proposal_stream(cid: str, inc: dict, ctx_programs: list[dict], bank: list[dict], lm: np.ndarray,
                    known: set[str], n: int, ctx_index: int) -> list[dict]:
    """First n unique, previously unseen proposals from the pinned live proposer, deterministic seeds."""
    sys.path.insert(0, str(MICRO / "python"))
    from rust_evaluator import propose_flywheel_candidates
    parents = [{"dsl": inc["dsl"], "program_sha256": inc["sha"]}] + [
        {"dsl": q["dsl"], "program_sha256": q["sha"]} for q in ctx_programs if q["sha"] != inc["sha"]]
    out, seen, batch = [], set(known), 0
    while len(out) < n and batch < 200:
        for prop in propose_flywheel_candidates(parents, lm, bank, inc["family"], n_proposals=16,
                                                seed=72_000_000 + 1000 * ctx_index + batch):
            if prop["program_sha256"] not in seen and len(out) < n:
                seen.add(prop["program_sha256"])
                out.append({"dsl": prop["dsl"], "sha": prop["program_sha256"], "operator": prop["operator"], "rank": len(out)})
        batch += 1
    return out


def simulate(stream_u: list[float], u_inc: float, verify: dict[int, list[float]], inc_resplit: list[float],
             budget: int = B) -> dict:
    """P0: evaluate stream[:budget] on the original cut, promote every apparent win.
    P1: walk the same stream; each evaluation costs 1, each apparent win costs 3 more (resplits 701-703);
        promote iff mean(resplit challenger - resplit incumbent) > 0; stop when the next step would exceed budget.
    verify[k] holds the 3 resplit utilities of stream item k (only needed for P1's apparent wins)."""
    p0 = [k for k, u in enumerate(stream_u[:budget]) if u > u_inc + MARGIN]
    spent, p1_eval, p1_wins, p1_promote, need = 0, 0, [], [], []
    for k, u in enumerate(stream_u):
        if spent + 1 > budget:
            break
        spent += 1; p1_eval += 1
        if u > u_inc + MARGIN:
            if spent + 3 > budget:
                break  # cannot afford confirmation: not promoted, stop
            spent += 3; p1_wins.append(k)
            if k not in verify:
                need.append(k); continue
            if np.mean(np.array(verify[k]) - np.array(inc_resplit)) > 0:
                p1_promote.append(k)
    return dict(p0_promote=p0, p1_evaluated=p1_eval, p1_apparent=p1_wins, p1_promote=p1_promote,
                p1_spent=spent, verify_needed=need)


def confirmed(u_ch: list[float], u_inc: list[float]) -> bool:
    """Common endpoint: mean over fresh seeds 801-803 of (challenger - incumbent) > 0."""
    return float(np.mean(np.array(u_ch) - np.array(u_inc))) > 0.0


def fit(job: dict, scratch: Path, timeout_s: int) -> dict:
    d = scratch / job["job_id"]
    d.mkdir(parents=True, exist_ok=True)
    inp, cand, out = d / "input.jtt1", d / "candidate.json", d / "eval_s1.json"
    inp.write_bytes(envelope((INPUT / "tables" / f"{job['context']}.jtt1").read_bytes(), job["seed"]))
    cand.write_text(json.dumps({"dsl": job["dsl"], "program_sha256": job["sha"], "birth_seq": 0, "parents": [],
                                "operator": "h72"}, sort_keys=True))
    env = {"PATH": "/usr/bin:/bin", "RAYON_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "TMPDIR": str(d), "HOME": str(d)}
    t0 = time.monotonic()
    proc = subprocess.Popen(["nice", "-n", "15", str(WORKER), "--fit-s1", str(inp), str(cand), str(out)],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env)
    status, cpu = None, None
    while True:
        pid, st, ru = os.wait4(proc.pid, os.WNOHANG)
        if pid:
            status, cpu = os.waitstatus_to_exitcode(st), ru.ru_utime + ru.ru_stime
            proc.returncode = status
            break
        if time.monotonic() - t0 > timeout_s:
            proc.kill(); _, st, ru = os.wait4(proc.pid, 0)
            status, cpu = "timeout", ru.ru_utime + ru.ru_stime
            proc.returncode = -9
            break
        time.sleep(0.05)
    res = dict(job_id=job["job_id"], stage=job["stage"], context=job["context"], sha=job["sha"], seed=job["seed"],
               rc=status, wall_s=round(time.monotonic() - t0, 3), cpu_s=round(cpu, 3))
    if status == 0 and out.exists():
        res.update(s1_metrics(job["dsl"], json.loads(out.read_text()).get("inner_cv") or []))
    else:
        res.update(status="timeout" if status == "timeout" else "failed")
    return res


def run_jobs(jobs, results: Path, scratch: Path, workers: int, timeout_s: int, spent_before: int, cap: int = FIT_CAP):
    done = set()
    if results.exists():
        done = {json.loads(l)["job_id"] for l in results.read_text().splitlines() if l.strip()}
    todo = [j for j in jobs if j["job_id"] not in done]
    if spent_before + len(todo) > cap:
        raise SystemExit(f"fit cap: {spent_before} + {len(todo)} > {cap}")

    def guarded(j):
        while 100.0 * os.getloadavg()[0] / os.cpu_count() > 85.0:
            time.sleep(15)
        return fit(j, scratch, timeout_s)

    with ThreadPoolExecutor(workers) as ex, results.open("a") as sink:
        for r in ex.map(guarded, todo):
            sink.write(json.dumps(r, sort_keys=True) + "\n"); sink.flush()
    return len(todo)


def load_results(path: Path) -> dict:
    out = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                r = json.loads(line); out[r["job_id"]] = r
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["plan", "search", "verify", "confirm", "analyze"])
    ap.add_argument("--w8-results", type=Path, required=True)
    ap.add_argument("--w8-plan", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--scratch", type=Path)
    ap.add_argument("--shard", choices=["A", "B", "all"], default="all")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--timeout", type=int, default=600)
    a = ap.parse_args()
    for key, path in (("w8_results", a.w8_results), ("w8_plan", a.w8_plan)):
        if not sha(path).startswith(PINS[key]):
            raise SystemExit(f"{key} sha mismatch")
    a.out.mkdir(parents=True, exist_ok=True)
    plan_path = a.out / "plan.json"
    if a.stage == "plan":
        if sha(WORKER) != PINS["worker"] or sha(MICRO / "python/rust_evaluator.py") != PINS["micro_rust_evaluator"]:
            raise SystemExit("worker/proposer sha mismatch")
        if sha(CACHE / "champions_featurized.json") != PINS["champions_featurized"] or sha(CACHE / "context_features.npz") != PINS["context_features"]:
            raise SystemExit("cache sha mismatch")
        inc = frozen_incumbents(a.w8_results, a.w8_plan)
        ctxs = sorted(inc, key=lambda c: hashlib.sha256(f"w8c-72|{c}".encode()).hexdigest())
        sel = json.loads((INPUT / "selected.json").read_text())
        train = {r["context_sha256"] for r in sel["train"]}
        bank = [r for r in json.loads((CACHE / "champions_featurized.json").read_text())
                if r["dl_split"] == "DL-Train" and r["context_sha256"] in train]
        cf = np.load(CACHE / "context_features.npz", allow_pickle=False)
        pos = {(c.decode() if isinstance(c, bytes) else str(c)): i for i, c in enumerate(cf["cids"])}
        lm_all = cf["lm48_z"]
        pairs = json.loads(a.w8_plan.read_text())["pairs"]
        plan = dict(schema="h72.w8-control-plan/v1", B=B, margin=MARGIN, decision_seeds=DECISION_SEEDS,
                    confirm_seeds=CONFIRM_SEEDS, fit_cap=FIT_CAP, pins=PINS, contexts=[])
        for k, c in enumerate(ctxs):
            if c not in train:
                raise SystemExit("non-DL-Train context")
            pool = np.load(INPUT / "pools" / f"{c}.npz", allow_pickle=False)
            known = {x.decode() for x in pool["program_sha256"]} | {p["challenger_sha"] for p in pairs} | {
                p["incumbent_sha"] for p in pairs}
            ctx_programs = [dict(dsl=p["challenger_dsl"], sha=p["challenger_sha"]) for p in pairs if p["context"] == c]
            stream = proposal_stream(c, inc[c], ctx_programs, bank, lm_all[pos[c]].astype(np.float64), known, B, k)
            plan["contexts"].append(dict(index=k, context=c, family=inc[c]["family"], shard="A" if k < len(ctxs) // 2 else "B",
                                         incumbent=inc[c], stream=stream))
        plan_path.write_text(json.dumps(plan, indent=1))
        print("contexts", len(ctxs), "proposals", sum(len(x["stream"]) for x in plan["contexts"]))
        return
    plan = json.loads(plan_path.read_text())
    mine = [x for x in plan["contexts"] if a.shard in ("all", x["shard"])]
    res_path = a.out / f"results-{a.shard}.jsonl"
    cap = FIT_CAP if a.shard == "all" else FIT_CAP // 2  # shards may run in separate output dirs
    allres = {}
    for p in a.out.glob("results-*.jsonl"):
        allres.update(load_results(p))
    if a.stage == "search":
        jobs = [dict(job_id=f"s-{x['index']:02d}-{q['rank']:03d}", stage="search", context=x["context"], seed=None, **q)
                for x in mine for q in x["stream"]]
        print("fits", run_jobs(jobs, res_path, a.scratch, a.workers, a.timeout, len(allres), cap))
    elif a.stage in ("verify", "confirm"):
        jobs = []
        for x in mine:
            su = [utility(allres[f"s-{x['index']:02d}-{q['rank']:03d}"]) for q in x["stream"]]
            inc = x["incumbent"]
            verify = {k: [utility(allres[f"v-{x['index']:02d}-{k:03d}-{s}"]) for s in DECISION_SEEDS]
                      for k in range(len(su)) if all(f"v-{x['index']:02d}-{k:03d}-{s}" in allres for s in DECISION_SEEDS)}
            sim = simulate(su, inc["u_orig"], verify, inc["u_701_703"])
            if a.stage == "verify":
                for k in sim["verify_needed"]:
                    q = x["stream"][k]
                    jobs += [dict(job_id=f"v-{x['index']:02d}-{k:03d}-{s}", stage="verify", context=x["context"], seed=s,
                                  dsl=q["dsl"], sha=q["sha"]) for s in DECISION_SEEDS]
            else:
                if sim["verify_needed"]:
                    raise SystemExit("run verify to completion first")
                promoted = sorted(set(sim["p0_promote"]) | set(sim["p1_promote"]))
                jobs += [dict(job_id=f"c-{x['index']:02d}-inc-{s}", stage="confirm", context=x["context"], seed=s,
                              dsl=inc["dsl"], sha=inc["sha"]) for s in CONFIRM_SEEDS]
                for k in promoted:
                    q = x["stream"][k]
                    jobs += [dict(job_id=f"c-{x['index']:02d}-{k:03d}-{s}", stage="confirm", context=x["context"], seed=s,
                                  dsl=q["dsl"], sha=q["sha"]) for s in CONFIRM_SEEDS]
        print("fits", run_jobs(jobs, res_path, a.scratch, a.workers, a.timeout, len(allres), cap))
    else:
        analyze(plan, allres, a.out)


def policy_elo(rows: list[dict]) -> dict:
    """Family-grouped Bradley-Terry Elo (repo elo_tracker; tie 1e-4, prior .01, 400 iters), P0 anchored at 1500.
    Per-context score = summed common-endpoint gain of that policy's promotions (0 if it promotes nothing)."""
    import importlib.util
    path = Path(__file__).resolve().parents[3] / "python/src/jope/north_star/elo_tracker.py"
    spec = importlib.util.spec_from_file_location("h72_elo_tracker", path)
    et = importlib.util.module_from_spec(spec); sys.modules["h72_elo_tracker"] = et; spec.loader.exec_module(et)
    recs = [et.DatasetEvalRecord(r["context"], r["family"], "regression",
                                 {"P0_search_only": r["p0"]["confirm_gain_sum"], "P1_search_plus_reliability": r["p1"]["confirm_gain_sum"]})
            for r in rows]
    res = et.compute_family_grouped_bradley_terry_elo(recs, anchor_contender="P0_search_only", anchor_elo=1500.0,
                                                      tie_margin=1e-4, n_iters=400, prior=0.01)
    return {k: res[k] for k in ("ratings", "pairwise_win_tie_loss_vs_anchor", "n_datasets", "n_families")}


def analyze(plan: dict, allres: dict, out: Path):
    rows = []
    for x in plan["contexts"]:
        i = x["index"]; inc = x["incumbent"]
        su = [utility(allres[f"s-{i:02d}-{q['rank']:03d}"]) for q in x["stream"]]
        verify = {k: [utility(allres[f"v-{i:02d}-{k:03d}-{s}"]) for s in DECISION_SEEDS]
                  for k in range(len(su)) if f"v-{i:02d}-{k:03d}-701" in allres}
        sim = simulate(su, inc["u_orig"], verify, inc["u_701_703"])
        ci = [utility(allres[f"c-{i:02d}-inc-{s}"]) for s in CONFIRM_SEEDS]
        def tally(promoted):
            conf = [k for k in promoted if confirmed([utility(allres[f"c-{i:02d}-{k:03d}-{s}"]) for s in CONFIRM_SEEDS], ci)]
            gain = sum(float(np.mean([utility(allres[f"c-{i:02d}-{k:03d}-{s}"]) for s in CONFIRM_SEEDS])) - float(np.mean(ci))
                       for k in promoted)
            return dict(promoted=len(promoted), confirmed=len(conf), false=len(promoted) - len(conf), net=2 * len(conf) - len(promoted),
                        confirm_gain_sum=gain)
        rows.append(dict(context=x["context"], family=x["family"], p0=tally(sim["p0_promote"]), p1=tally(sim["p1_promote"]),
                         p1_evaluated=sim["p1_evaluated"], p1_apparent=len(sim["p1_apparent"]), p0_apparent=len(sim["p0_promote"])))
    cost = defaultdict(lambda: dict(n=0, failed=0, cpu_s=0.0, wall_s=0.0))
    for r in allres.values():
        c = cost[r["stage"]]; c["n"] += 1; c["failed"] += r["status"] != "completed"; c["cpu_s"] += r["cpu_s"]; c["wall_s"] += r["wall_s"]
    fam = defaultdict(list)
    for r in rows:
        fam[r["family"]].append(r)
    keys = sorted(fam); rng = np.random.default_rng(20261031)
    def fam_delta(metric):
        return np.array([sum(r["p1"][metric] - r["p0"][metric] for r in fam[f]) for f in keys])
    summary = {}
    for metric in ("confirmed", "false", "net", "confirm_gain_sum"):
        d = fam_delta(metric); b = rng.choice(len(d), (10000, len(d)))
        summary[metric] = dict(p0=sum(r["p0"][metric] for r in rows), p1=sum(r["p1"][metric] for r in rows),
                               p1_minus_p0=float(d.sum()), family_sum_ci95=np.percentile(d[b].sum(1), [2.5, 97.5]).tolist())
    report = dict(schema="h72.w8-control-report/v1", per_context=rows, summary=summary, cost=dict(cost),
                  total_fits=len(allres), fit_cap=FIT_CAP, elo=policy_elo(rows))
    (out / "report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(dict(summary=summary, cost=dict(cost)), indent=1))


if __name__ == "__main__":
    main()
