#!/usr/bin/env python3
"""H317 Gate H0 GP block (H0.4 seeding -> H0.5 evolution -> H0.6 stacking) + Stage V GP pipelines + timing.

Run order per pilot shard (`pilot_shard(n=PILOT_N)`) x search seeds (317101, 317102):
  BANK_NN1   (per context, 1 AUDIT eval)                 -> reference pipeline (no search)
  H0.4       Seed A/B/C/D/E (+ NOISE_PROBE on seed 317101) -> decision: Holm over m=4 challengers vs A
  H0.5       Evo A vs Evo B from the adopted seed set, 10 generations, 48 screened / exactly 6 live evals per gen
  H0.6       single / CV1 / CV2@N / CV3@N / CV4 / CV5 (+ BAG_FINAL, SEL_MEANLR secondaries) on the adopted arm
             -> leave-one-family-out config selection; test = CV-selected stack - single best tree
Every within-context choice uses SEARCH results only (inside `selection_scope`); each declared output is AUDIT-scored
exactly once afterwards (`AuditLedger`). Per-context values are averaged over the two search seeds, then
family-equal means / paired whole-family statistics (common_h317, RNG 317150).

Failure policy (AUDIT scale, leakage red team rule 3): AUDIT status is "completed" whenever the worker exits 0 with
finite predictions, and the raw continuous AUDIT utility is kept. A crashed / non-finite output (or a pipeline with no
evaluable output) is scored as the SEARCH-target-mean predictor (utility 0.0); a crashed stack member contributes the
SEARCH mean; weights are never changed by AUDIT outcomes. `value_m2` stores the -2.0 convention for sensitivity.
BANK_NN1 is run as a reference only.

Resumable: one append-only JSONL per (stage, context, seed) under <run_dir>/stages/; a stage is skipped when its
`result` record exists. The engine memo (<gp_root>/memo/<cid>.jsonl) makes interrupted stages cheap to repeat.
"""

from __future__ import annotations

import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "RAYON_NUM_THREADS"):
    os.environ[_k] = "1"  # numeric thread cap: the Rust workers are the only compute threads

import argparse
import concurrent.futures as cf
import json
import math
import struct
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import common_h317 as C  # noqa: E402

C.install_guard()

import numpy as np  # noqa: E402

import distributed_10gen_island_gp as D  # noqa: E402
import gp_grammar_h317 as G  # noqa: E402
import screener_h317 as S  # noqa: E402
import stacking_h317 as ST  # noqa: E402
from gp_engine_h317 import (AuditLedger, AuditResult, ContextEnvs, EvalEngine, GP_ROOT, ResourceGuard,  # noqa: E402
                            SearchResult)

H04_CHALLENGERS = ("B", "C", "D", "E")
H06_SECONDARY = ("BAG_FINAL", "SEL_MEANLR")
MAX_WORKERS = 4  # prereg template: numeric threads <= 4 in total


# ----------------------------------------------------------------------------------------------------------------------
# Candidate (de)serialization
# ----------------------------------------------------------------------------------------------------------------------
def cand_json(c: G.Candidate) -> dict:
    return {"dsl": c.dsl, "sha": c.sha, "operator": c.operator, "parents": list(c.parents), "island": c.island,
            "meta": [list(x) for x in c.meta]}


def cand_from_json(d: dict) -> G.Candidate:
    return G.Candidate(d["dsl"], d["sha"], d["operator"], tuple(d["parents"]), int(d["island"]),
                       tuple(tuple(x) for x in d["meta"]))


def res_brief(r: Optional[SearchResult]) -> Optional[dict]:
    if r is None:
        return None
    return {"sha": r.sha, "utility": r.utility, "fitness": r.fitness, "gate": r.gate, "mean_lr": r.mean_lr,
            "sd_lr": r.sd_lr, "status": r.status, "family": G.head_family(r.dsl), "island": G.island_of(r.dsl)}


# ----------------------------------------------------------------------------------------------------------------------
# Store
# ----------------------------------------------------------------------------------------------------------------------
class Store:
    def __init__(self, run_dir: Path):
        self.root = Path(run_dir)
        self._lock = threading.Lock()

    def path(self, stage: str, cid: str, seed) -> Path:
        p = self.root / "stages" / stage / f"{cid}__{seed}.jsonl"
        C.assert_allowed(p)
        return p

    def append(self, stage: str, cid: str, seed, rec: dict) -> None:
        p = self.path(stage, cid, seed)
        p.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(rec, sort_keys=True, default=C._json_default)
        with self._lock, open(p, "a") as f:
            f.write(line + "\n")
            f.flush()
            os.fsync(f.fileno())

    def result(self, stage: str, cid: str, seed) -> Optional[dict]:
        p = self.path(stage, cid, seed)
        if not p.exists():
            return None
        last = None
        with open(p) as f:
            for line in f:
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if d.get("type") == "result":
                    last = d
        return last


# ----------------------------------------------------------------------------------------------------------------------
# Runtime
# ----------------------------------------------------------------------------------------------------------------------
@dataclass
class Runtime:
    engine: EvalEngine
    ledger: AuditLedger
    screener_for: Callable[[str], S.Screener]
    coder: D.ContextCoder
    bank: List[D.BankEntry]
    fam_of: Dict[str, str]
    ds_of: Dict[str, str]
    store: Store
    seeds: Tuple[int, ...] = C.SEARCH_SEEDS
    _tb: Dict[str, D.TargetBank] = field(default_factory=dict)

    def target_bank(self, cid: str) -> D.TargetBank:
        if cid not in self._tb:
            self._tb[cid] = D.TargetBank(self.bank, cid, self.fam_of, self.ds_of)
        return self._tb[cid]

    def gp(self, cid: str, seed: int, full_mode: bool = False) -> D.GPContext:
        ctx = self.engine.context(cid, self.fam_of[cid], full_mode=full_mode)
        return D.GPContext(ctx, self.engine, self.screener_for(cid), self.target_bank(cid), self.coder, seed)


def audit_one(rt: Runtime, pipeline: str, g: D.GPContext, cand: Optional[G.Candidate]) -> dict:
    if cand is None:
        return {"status": "no_output", "utility": None}
    a = rt.ledger.score(pipeline, g.ctx, g.seed, [cand])[0]
    return {"status": a.status, "utility": a.utility if a.status == "completed" else None, "sha": cand.sha}


def audit_stack(rt: Runtime, pipeline: str, g: D.GPContext, spec: ST.StackSpec,
                cand_of: Dict[str, G.Candidate]) -> dict:
    shas, _, _ = spec.declared()
    res = rt.ledger.score(pipeline, g.ctx, g.seed, [cand_of[s] for s in shas])
    out = ST.stack_audit_utility(spec, {a.sha: a for a in res}, g.ctx.ya, g.ctx.ba, g.ctx.y_search_mean)
    out["n_members"] = len(shas)
    return out


MEAN_PREDICTOR_U = 0.0


def value_of(raw: dict) -> Tuple[float, float, bool]:
    """(value, -2.0-convention value, invalid?); invalid outputs score as the SEARCH-mean predictor."""
    u = raw.get("utility")
    if u is None or raw.get("status") != "completed":
        return MEAN_PREDICTOR_U, C.GATE_FAIL_UTILITY, True
    bad = bool(raw.get("crashed", 0))
    return float(u), (C.GATE_FAIL_UTILITY if bad and raw.get("n_mean_substituted") == raw.get("n_members", 1) else
                      float(u)), bad


# ----------------------------------------------------------------------------------------------------------------------
# Stages
# ----------------------------------------------------------------------------------------------------------------------
def stage_bank_nn1(rt: Runtime, cid: str) -> dict:
    hit = rt.store.result("bank_nn1", cid, "na")
    if hit:
        return hit
    g = rt.gp(cid, 0)
    with D.selection_scope("BANK_NN1"):
        cand = D.bank_nn1(g.tb, g.coder)
    a = audit_one(rt, "BANK_NN1", g, cand)
    v = value_of(a)
    rec = {"type": "result", "cid": cid, "audit": a, "value": v[0], "value_m2": v[1], "invalid": v[2],
           "cand": cand_json(cand) if cand else None,
           "bank_exclusion": g.tb.exclusion_record(), "coder": g.coder.name}
    rt.store.append("bank_nn1", cid, "na", rec)
    return rec


def stage_h04(rt: Runtime, cid: str, seed: int, noise_probe: bool) -> dict:
    hit = rt.store.result("h04", cid, seed)
    if hit:
        return hit
    g = rt.gp(cid, seed)
    t0 = time.time()
    runs = D.run_h04(g)
    rec = {"type": "result", "cid": cid, "seed": seed, "schemes": {}, "screener": g.screener.describe()}
    for name, run in runs.items():
        m = run.search_metrics()
        m["order"] = [cand_json(c) for c in run.order]
        m["info"] = run.info
        cand_of = {c.sha: c for c in run.order}
        for tag, best in (("full", run.best()), ("bm", run.budget_matched())):
            raw = audit_one(rt, f"H04.{name}.{tag}", g, cand_of[best.sha] if best else None)
            pv, rv, inv = value_of(raw)
            m[f"audit_{tag}"] = {"raw": raw, "value": pv, "value_m2": rv, "invalid": inv}
        rec["schemes"][name] = m
    if noise_probe:
        a = runs["A"]
        rec["noise_probe"] = D.noise_probe(g, a.order, a.results)
    rec["wall_s"] = time.time() - t0
    rt.store.append("h04", cid, seed, rec)
    return rec


def _seed_set(rt: Runtime, cid: str, seed: int, scheme: str, g: D.GPContext):
    h04 = rt.store.result("h04", cid, seed)
    if h04 is None:
        raise RuntimeError(f"H0.4 missing for {cid} {seed}")
    order = [cand_from_json(d) for d in h04["schemes"][scheme]["order"]]
    with D.selection_scope("reconstruct_seeds"):
        res = g.evaluate(order)  # memo hits
    return order, res


def stage_h05(rt: Runtime, cid: str, seed: int, scheme: str) -> dict:
    key = f"h05__{scheme}"
    hit = rt.store.result(key, cid, seed)
    if hit:
        return hit
    g = rt.gp(cid, seed)
    order, res = _seed_set(rt, cid, seed, scheme, g)
    t0 = time.time()
    rec = {"type": "result", "cid": cid, "seed": seed, "scheme": scheme, "arms": {}}
    runs = {arm: fn(g, order, res) for arm, fn in D.ARMS.items()}
    if runs["evoA"].live_per_gen != runs["evoB"].live_per_gen or \
            runs["evoA"].screened_per_gen != runs["evoB"].screened_per_gen:
        raise AssertionError("evo arms broke the matched budget")
    for arm, run in runs.items():
        summ = run.summary()
        summ["cands"] = {sha: cand_json(c) for sha, c in run.cand_of.items()}
        for gen in D.AUDIT_GENS:
            b = run.best_after(gen)
            raw = audit_one(rt, f"H05.{arm}.g{gen}", g, run.cand_of[b.sha] if b else None)
            pv, rv, inv = value_of(raw)
            summ[f"g{gen}"] = {"best": res_brief(b), "raw": raw, "value": pv, "value_m2": rv, "invalid": inv}
        rec["arms"][arm] = summ
    rec["wall_s"] = time.time() - t0
    rt.store.append(key, cid, seed, rec)
    return rec


def _arm_results(rt: Runtime, cid: str, seed: int, scheme: str, arm: str, g: D.GPContext):
    h05 = rt.store.result(f"h05__{scheme}", cid, seed)
    a = h05["arms"][arm]
    cand_of = {sha: cand_from_json(d) for sha, d in a["cands"].items()}
    cands = [cand_of[s] for s in a["result_shas"]]
    with D.selection_scope("reconstruct_arm"):
        res = g.evaluate(cands)  # memo hits
    return res, cand_of


def stage_h06(rt: Runtime, cid: str, seed: int, scheme: str, arm: str) -> dict:
    key = f"h06__{scheme}__{arm}"
    hit = rt.store.result(key, cid, seed)
    if hit:
        return hit
    g = rt.gp(cid, seed)
    res, cand_of = _arm_results(rt, cid, seed, scheme, arm, g)
    t0 = time.time()
    fd = ST.FoldData.from_ctx(g.ctx)
    specs = ST.all_configs(res, fd)
    cv5 = D.portfolio_nnls(g, res, cand_of)
    single = specs.get("single")
    single_best = next((r for r in res if single and r.sha == single.members[0]), None)
    bag = D.bag_final_program(g, single_best, cand_of)
    with D.selection_scope("SEL_MEANLR"):
        sml = D.sel_meanlr(res)
    rec = {"type": "result", "cid": cid, "seed": seed, "scheme": scheme, "arm": arm, "configs": {}}
    # reference first: aliases (fallback configs) reuse its value without a second AUDIT request
    if single is None:
        s_raw = {"status": "no_output", "utility": None}
    else:
        s_raw = audit_one(rt, "H06.single", g, cand_of[single.members[0]])
    sv = value_of(s_raw)
    rec["configs"]["single"] = {"spec": single.to_json() if single else None, "raw": s_raw, "value": sv[0],
                                "value_m2": sv[1], "invalid": sv[2], "search_s1": single.search if single else None}
    for name, spec in specs.items():
        if name == "single":
            continue
        if spec.fallback_single:
            rec["configs"][name] = {"alias": "single", "value": sv[0], "value_m2": sv[1], "invalid": sv[2],
                                    "spec": spec.to_json(), "search_s1": spec.search}
            continue
        raw = audit_stack(rt, f"H06.{name}", g, spec, cand_of)
        pv = value_of(raw)
        rec["configs"][name] = {"raw": raw, "value": pv[0], "value_m2": pv[1], "invalid": pv[2], "spec": spec.to_json(),
                                "search_s1": spec.search}
    for prog in (cv5, bag):
        if prog.fallback or prog.cand is None:
            rec["configs"][prog.name] = {"alias": "single", "value": sv[0], "value_m2": sv[1], "invalid": sv[2],
                                         "info": prog.info, "search": res_brief(prog.result),
                                         "live_search_evals": prog.live_evals}
            continue
        raw = audit_one(rt, f"H06.{prog.name}", g, prog.cand)
        pv = value_of(raw)
        rec["configs"][prog.name] = {"raw": raw, "value": pv[0], "value_m2": pv[1], "invalid": pv[2],
                                     "info": prog.info, "search": res_brief(prog.result),
                                     "live_search_evals": prog.live_evals}
    if sml is None:
        rec["configs"]["SEL_MEANLR"] = {"alias": "single", "value": sv[0], "value_m2": sv[1], "invalid": sv[2]}
    else:
        raw = audit_one(rt, "H06.SEL_MEANLR", g, cand_of[sml.sha])
        pv = value_of(raw)
        rec["configs"]["SEL_MEANLR"] = {"raw": raw, "value": pv[0], "value_m2": pv[1], "invalid": pv[2],
                                        "search": res_brief(sml)}
    for name in H06_CONFIGS:  # every config exists for every context (missing = could not be built -> single)
        if name not in rec["configs"]:
            rec["configs"][name] = {"alias": "single(not_buildable)", "value": sv[0], "value_m2": sv[1],
                                    "invalid": sv[2]}
    rec["wall_s"] = time.time() - t0
    rt.store.append(key, cid, seed, rec)
    return rec


H06_CONFIGS = ("CV1", "CV4", "CV5") + tuple(f"CV2@{n}" for n in ST.N_GRID) + tuple(f"CV3@{n}" for n in ST.N_GRID) + \
    ("BAG_FINAL", "SEL_MEANLR")


# ----------------------------------------------------------------------------------------------------------------------
# Decisions
# ----------------------------------------------------------------------------------------------------------------------
def seed_mean(per_seed: Dict[Tuple[str, int], float]) -> Dict[str, float]:
    acc: Dict[str, List[float]] = {}
    for (cid, _), v in per_seed.items():
        acc.setdefault(cid, []).append(float(v))
    return {c: float(np.mean(v)) for c, v in acc.items()}


def contrast(arm: Dict[str, float], ref: Dict[str, float], fam_of: Dict[str, str], alpha: Optional[float] = None
             ) -> dict:
    summ = C.robustness_summary(arm, ref, fam_of, C.BOOT_SEED_GATE_H0)
    if alpha is not None:
        summ["holm_alpha"] = alpha
        summ["holm_lcb"] = C.paired_family_bootstrap_alpha(arm, ref, fam_of, C.BOOT_SEED_GATE_H0, alpha)["lcb"]
        summ["gate"] = C.contract_advance_gate(summ, "holm_lcb")
    else:
        summ["gate"] = C.contract_advance_gate(summ, "lcb95")
    return summ


def decide_h04(values: Dict[str, Dict[str, float]], fam_of: Dict[str, str]) -> dict:
    """Holm step-down over m = 4 challengers (B, C, D, E vs A) on budget-matched AUDIT utility."""
    ref = values["A"]
    m = len(H04_CHALLENGERS)
    ps = {s: C.signflip_pvalue(values[s], ref, fam_of, C.BOOT_SEED_GATE_H0) for s in H04_CHALLENGERS}
    order = sorted(H04_CHALLENGERS, key=lambda s: (ps[s], s))
    out, alive = {}, True
    for rank, s in enumerate(order):
        c = contrast(values[s], ref, fam_of, C.holm_adjusted_lcb_alpha(m, rank))
        c["holm_rank"] = rank
        c["pass"] = bool(alive and c["gate"]["pass"])
        alive = c["pass"]
        out[s] = c
    passing = [s for s in H04_CHALLENGERS if out[s]["pass"]]
    adopted = max(passing, key=lambda s: (C.family_equal_mean(values[s], fam_of), s)) if passing else "A"
    return {"contrasts": out, "adopted": adopted,
            "family_means": {s: C.family_equal_mean(v, fam_of) for s, v in values.items()}}


def decide_h05(values: Dict[str, Dict[str, float]], fam_of: Dict[str, str]) -> dict:
    c = contrast(values["evoB.g10"], values["evoA.g10"], fam_of)
    return {"contrast_evoB_vs_evoA_g10": c, "adopted": "evoB" if c["gate"]["pass"] else "evoA", "generations": 10,
            "family_means": {k: C.family_equal_mean(v, fam_of) for k, v in values.items()}}


def decide_h06(values: Dict[str, Dict[str, float]], fam_of: Dict[str, str]) -> dict:
    """Leave-one-family-out config choice; test = CV-selected - single; locked = best over all pilot families."""
    configs = sorted(k for k in values if k not in ("single",) + H06_SECONDARY)
    cids = sorted(values["single"])
    fams = sorted({fam_of[c] for c in cids})
    fm = {k: C.family_means(values[k], fam_of) for k in configs}
    sel, chosen = {}, {}
    for c in cids:
        f = fam_of[c]
        best = max(configs, key=lambda k: (float(np.mean([fm[k][x] for x in fams if x != f])), k))
        chosen[c] = best
        sel[c] = values[best][c]
    test = contrast(sel, values["single"], fam_of)
    locked = max(configs, key=lambda k: (float(np.mean(list(fm[k].values()))), k))
    out = {"lofo_test_vs_single": test, "lofo_choices": chosen, "locked_config_if_adopted": locked,
           "adopted": locked if test["gate"]["pass"] else "single",
           "family_means": {k: C.family_equal_mean(v, fam_of) for k, v in values.items()},
           "direct": {}}
    for k in ("CV5",) + H06_SECONDARY:
        if k in values:
            out["direct"][f"{k}_vs_single"] = contrast(values[k], values["single"], fam_of)
    return out


# ----------------------------------------------------------------------------------------------------------------------
# Lab driver
# ----------------------------------------------------------------------------------------------------------------------
def run_parallel(fn, jobs: Sequence[tuple], n: int) -> List:
    out = [None] * len(jobs)
    with cf.ThreadPoolExecutor(max_workers=max(1, n), thread_name_prefix="h317gp-job") as ex:
        futs = {ex.submit(fn, *j): i for i, j in enumerate(jobs)}
        for f in cf.as_completed(futs):
            out[futs[f]] = f.result()
    return out


def run_gate_h0(rt: Runtime, cids: Sequence[str], run_dir: Path, job_threads: int,
                stages: Sequence[str] = ("h04", "h05", "h06"), overrides: Optional[dict] = None) -> dict:
    overrides = overrides or {}
    fam_of = rt.fam_of
    t0 = time.time()
    nn1 = run_parallel(lambda c: stage_bank_nn1(rt, c), [(c,) for c in cids], job_threads)
    C.write_json(run_dir / "bank_exclusions.json", {r["cid"]: r["bank_exclusion"] for r in nn1})
    summary: dict = {"contexts": len(cids), "seeds": list(rt.seeds), "bank_nn1_family_mean":
                     C.family_equal_mean({r["cid"]: r["value"] for r in nn1}, fam_of),
                     "bank_nn1_invalid": int(sum(r["invalid"] for r in nn1))}
    jobs = [(c, s) for c in cids for s in rt.seeds]
    if "h04" not in stages:
        return summary
    h04 = run_parallel(lambda c, s: stage_h04(rt, c, s, s == rt.seeds[0]), jobs, job_threads)
    per = {sch: {} for sch in D.SCHEMES}
    per_full = {sch: {} for sch in D.SCHEMES}
    for r in h04:
        for sch in D.SCHEMES:
            per[sch][(r["cid"], r["seed"])] = r["schemes"][sch]["audit_bm"]["value"]
            per_full[sch][(r["cid"], r["seed"])] = r["schemes"][sch]["audit_full"]["value"]
    vals = {k: seed_mean(v) for k, v in per.items()}
    d04 = decide_h04(vals, fam_of)
    d04["full_budget_family_means"] = {k: C.family_equal_mean(seed_mean(v), fam_of) for k, v in per_full.items()}
    d04["search_metrics_family_means"] = _h04_search_means(h04, fam_of)
    d04["noise_probe"] = _noise_summary(h04)
    scheme = overrides.get("h04_scheme", d04["adopted"])
    summary["H0.4"] = d04
    summary["H0.4"]["used_scheme"] = scheme
    if "h05" not in stages:
        return summary
    h05 = run_parallel(lambda c, s: stage_h05(rt, c, s, scheme), jobs, job_threads)
    per5: Dict[str, Dict] = {}
    for r in h05:
        for arm, a in r["arms"].items():
            for gen in D.AUDIT_GENS:
                per5.setdefault(f"{arm}.g{gen}", {})[(r["cid"], r["seed"])] = a[f"g{gen}"]["value"]
    d05 = decide_h05({k: seed_mean(v) for k, v in per5.items()}, fam_of)
    arm = overrides.get("h05_arm", d05["adopted"])
    summary["H0.5"] = d05
    summary["H0.5"]["used_arm"] = arm
    if "h06" not in stages:
        return summary
    h06 = run_parallel(lambda c, s: stage_h06(rt, c, s, scheme, arm), jobs, job_threads)
    per6: Dict[str, Dict] = {}
    for r in h06:
        for k, v in r["configs"].items():
            per6.setdefault(k, {})[(r["cid"], r["seed"])] = v["value"]
    d06 = decide_h06({k: seed_mean(v) for k, v in per6.items()}, fam_of)
    d06["in_sample_search_s1_family_means"] = _h06_search_means(h06, fam_of)
    summary["H0.6"] = d06
    summary["invalids"] = _invalid_counts(h04, h05, h06)
    summary["wall_s"] = time.time() - t0
    return summary


def _h04_search_means(h04, fam_of) -> dict:
    out = {}
    for sch in D.SCHEMES:
        for key in ("best_search_utility", "bm_best_search_utility", "mean_passing_search_utility",
                    "n_families_passing", "evals_used"):
            v = {(r["cid"], r["seed"]): r["schemes"][sch][key] for r in h04 if r["schemes"][sch][key] is not None}
            if v:
                out.setdefault(sch, {})[key] = C.family_equal_mean(seed_mean(v), fam_of)
    return out


def _h06_search_means(h06, fam_of) -> dict:
    acc: Dict[str, Dict] = {}
    for r in h06:
        for k, v in r["configs"].items():
            s1 = v.get("search_s1") or v.get("search")
            if s1 and s1.get("utility") is not None:
                acc.setdefault(k, {})[(r["cid"], r["seed"])] = s1["utility"]
    return {k: C.family_equal_mean(seed_mean(v), fam_of) for k, v in acc.items()}


def _noise_summary(h04) -> dict:
    rows = [r["noise_probe"] for r in h04 if r.get("noise_probe", {}).get("n_candidates", 0) >= 2]
    if not rows:
        return {}
    return {k: float(np.mean([x[k] for x in rows])) for k in ("shared_offset_sd", "candidate_specific_sd",
                                                              "raw_candidate_sd", "rank_agreement_spearman")}


def _invalid_counts(h04, h05, h06) -> dict:
    out: Dict[str, int] = {}
    for r in h04:
        for sch, m in r["schemes"].items():
            for tag in ("full", "bm"):
                out[f"H04.{sch}.{tag}"] = out.get(f"H04.{sch}.{tag}", 0) + int(m[f"audit_{tag}"]["invalid"])
    for r in h05:
        for arm, a in r["arms"].items():
            for gen in D.AUDIT_GENS:
                out[f"H05.{arm}.g{gen}"] = out.get(f"H05.{arm}.g{gen}", 0) + int(a[f"g{gen}"]["invalid"])
    for r in h06:
        for k, v in r["configs"].items():
            out[f"H06.{k}"] = out.get(f"H06.{k}", 0) + int(v["invalid"])
    return out


# ----------------------------------------------------------------------------------------------------------------------
# Stage V pipelines (built now, never run on DL-Val before the LOCK; Stage V uses the same AUDIT protocol)
# ----------------------------------------------------------------------------------------------------------------------
@dataclass
class LockedConfig:
    scheme: str = "C"
    arm: str = "evoB"
    generations: int = 10
    stack: str = "CV3@5"


def _stack_output(rt: Runtime, pipeline: str, g: D.GPContext, res: Sequence[SearchResult],
                  cand_of: Dict[str, G.Candidate], config: str) -> dict:
    """Fix the output of a stack config from SEARCH data, then AUDIT it once. Returns value + live SEARCH evals."""
    fd = ST.FoldData.from_ctx(g.ctx)
    extra = 0
    with D.selection_scope(f"{pipeline}.stack"):
        single = ST.single_spec(res)
    if config in ("CV5",):
        prog = D.portfolio_nnls(g, res, cand_of)
        extra = prog.live_evals
        if not prog.fallback and prog.cand is not None:
            raw = audit_one(rt, pipeline, g, prog.cand)
            v = value_of(raw)
            return {"config": config, "raw": raw, "value": v[0], "value_m2": v[1], "invalid": v[2], "extra_live": extra}
        spec = single
    elif config == "single":
        spec = single
    else:
        with D.selection_scope(f"{pipeline}.stack"):
            if config == "CV1":
                spec = ST.cv1_h257_top3(res, fd)
            elif config == "CV4":
                spec = ST.cv4_hier(res, fd)
            elif config.startswith("CV2@"):
                spec = ST.cv2_fw(res, fd, int(config[4:]))
            elif config.startswith("CV3@"):
                spec = ST.cv3_s1direct(res, fd, int(config[4:]))
            else:
                raise ValueError(config)
    if spec is None:
        return {"config": config, "raw": {"status": "no_output"}, "value": MEAN_PREDICTOR_U,
                "value_m2": C.GATE_FAIL_UTILITY, "invalid": True, "extra_live": extra}
    if len(spec.declared()[0]) == 1:
        raw = audit_one(rt, pipeline, g, cand_of[spec.declared()[0][0]])
    else:
        raw = audit_stack(rt, pipeline, g, spec, cand_of)
    v = value_of(raw)
    return {"config": config, "raw": raw, "value": v[0], "value_m2": v[1], "invalid": v[2], "extra_live": extra,
            "spec": spec.to_json()}


def stage_v_pipeline(rt: Runtime, name: str, cid: str, seed: int, locked: LockedConfig,
                     k_topk: Optional[int] = None, full_mode: bool = False) -> dict:
    """One Stage V pipeline on one context. Refuses DL-Val unless the common guard's Stage V unlock happened.

    H317_LOCKED      locked seeds -> locked arm (locked generations) -> locked stack config
    EVOA_MATCHED     locked seeds -> panmictic Evo A, same generations / live evals -> CV1 Top-3
    SCREEN_TOPK_PICK one screening of bank NN + grammar proposals, top-K live evals (K = H317_LOCKED live evals)
    H257_REPLICA     Seed A (8) -> 3 gens x 5 children (20 proposals) -> CV1 Top-3
    BANK_NN1         nearest cross-family champion (DOPE view 0), 1 AUDIT eval
    BANK_NN8_PICK    8 nearest cross-family champions, best-by-SEARCH
    BANK_NN8_PORTFOLIO  CV5 portfolio over the NN8 head classes
    """
    dev = C._STATE["dev_cids"]
    if cid in dev and not C._STATE["stage_v"]:
        raise C.GuardViolation("Stage V pipeline on a DL-Val context before the LOCK unlock")
    g = rt.gp(cid, seed, full_mode=full_mode)
    pipe = f"V.{name}"
    if name == "BANK_NN1":
        with D.selection_scope(pipe):
            cand = D.bank_nn1(g.tb, g.coder)
        raw = audit_one(rt, pipe, g, cand)
        v = value_of(raw)
        return {"pipeline": name, "cid": cid, "seed": seed, "live_evals": 0, "audit_evals": 1, "value": v[0],
                "value_m2": v[1], "invalid": v[2], "raw": raw}
    if name in ("BANK_NN8_PICK", "BANK_NN8_PORTFOLIO"):
        run = D.bank_nn8_pick(g)
        cand_of = {c.sha: c for c in run.order}
        out = _stack_output(rt, pipe, g, run.results, cand_of, "single" if name == "BANK_NN8_PICK" else "CV5")
        out.update(pipeline=name, cid=cid, seed=seed, live_evals=run.evals_used + out["extra_live"])
        return out
    if name == "H257_REPLICA":
        a = D.seed_A(g)
        evo = D.h257_replica_evo(g, a.order, a.results)
        cand_of = {**{c.sha: c for c in a.order}, **evo.cand_of}
        out = _stack_output(rt, pipe, g, evo.results, cand_of, "CV1")
        out.update(pipeline=name, cid=cid, seed=seed, live_evals=a.evals_used + evo.live_evals + out["extra_live"])
        return out
    if name in ("H317_LOCKED", "EVOA_MATCHED"):
        sr = D.run_scheme(g, locked.scheme)
        arm_fn = D.ARMS[locked.arm] if name == "H317_LOCKED" else D.evo_panmictic
        evo = arm_fn(g, sr.order, sr.results, gens=locked.generations)
        cand_of = {**{c.sha: c for c in sr.order}, **evo.cand_of}
        cfg = locked.stack if name == "H317_LOCKED" else "CV1"
        out = _stack_output(rt, pipe, g, evo.results, cand_of, cfg)
        out.update(pipeline=name, cid=cid, seed=seed, seed_evals=sr.evals_used, evo_live=evo.live_evals,
                   live_evals=sr.evals_used + evo.live_evals + out["extra_live"])
        return out
    if name == "SCREEN_TOPK_PICK":
        if k_topk is None:
            raise ValueError("SCREEN_TOPK_PICK needs K = H317_LOCKED live evals")
        run = D.screen_topk_pick(g, k_topk)
        cand_of = {c.sha: c for c in run.order}
        out = _stack_output(rt, pipe, g, run.results, cand_of, "single")
        out.update(pipeline=name, cid=cid, seed=seed, live_evals=run.evals_used, k=k_topk)
        return out
    raise ValueError(name)


STAGE_V_PIPELINES = ("H317_LOCKED", "EVOA_MATCHED", "SCREEN_TOPK_PICK", "H257_REPLICA", "BANK_NN1", "BANK_NN8_PICK",
                     "BANK_NN8_PORTFOLIO")


def stage_v_context(rt: Runtime, cid: str, seed: int, locked: LockedConfig) -> dict:
    """All Stage V pipelines on one context, with the live-eval matching asserts."""
    out = {"BANK_NN1": stage_v_pipeline(rt, "BANK_NN1", cid, seed, locked)}
    out["H317_LOCKED"] = stage_v_pipeline(rt, "H317_LOCKED", cid, seed, locked)
    out["EVOA_MATCHED"] = stage_v_pipeline(rt, "EVOA_MATCHED", cid, seed, locked)
    k = out["H317_LOCKED"]["live_evals"]
    out["SCREEN_TOPK_PICK"] = stage_v_pipeline(rt, "SCREEN_TOPK_PICK", cid, seed, locked, k_topk=k)
    for name in ("H257_REPLICA", "BANK_NN8_PICK", "BANK_NN8_PORTFOLIO"):
        out[name] = stage_v_pipeline(rt, name, cid, seed, locked)
    if abs(out["H317_LOCKED"]["live_evals"] - out["EVOA_MATCHED"]["live_evals"]) > 2:
        raise AssertionError("EVOA_MATCHED live evals differ from H317_LOCKED by more than 2")
    if out["SCREEN_TOPK_PICK"]["live_evals"] != k:
        raise AssertionError("SCREEN_TOPK_PICK K mismatch")
    return out


# ----------------------------------------------------------------------------------------------------------------------
# Setup / CLI
# ----------------------------------------------------------------------------------------------------------------------
def table_width(cid: str) -> int:
    with open(C.table_path(cid), "rb") as f:
        head = f.read(17)
    return struct.unpack_from("<III", head, 4)[2]


def build_runtime(run_dir: Path, cores: Sequence[int], screener: str, coder: str, gp_root: Path = GP_ROOT,
                  timeout_s: float = 60.0, guard: bool = True, runner=None, seeds=C.SEARCH_SEEDS,
                  neural_run_dir: Optional[Path] = None, stage_v: bool = False) -> Runtime:
    """Bank = TRAIN champions/winners only (never grows). Stage V adds the DL-Val rows' family/dataset ids (only
    after `common_h317.unlock_stage_v`) so the bank exclusion also covers DL-Val targets; nothing carries state
    across contexts (memo, corrector, screener cache and codes are all keyed by context)."""
    if len(cores) > MAX_WORKERS:
        raise ValueError(f"at most {MAX_WORKERS} single-threaded Rust workers (prereg numeric thread cap)")
    rows = C.train_rows()
    meta_rows = list(rows) + (list(C.dev_rows_for_stage_v()) if stage_v else [])
    fam_of = {r["context_sha256"]: r["family_id"] for r in meta_rows}
    ds_of = {r["context_sha256"]: r["dataset_id"] for r in meta_rows}
    folds = C.family_folds(rows)
    bank, _ = D.build_bank(rows)
    rg = ResourceGuard(log_path=Path(run_dir) / "resource_pauses.jsonl") if guard else None
    engine = EvalEngine(cores, root=gp_root, timeout_s=timeout_s, guard=rg, runner=runner)
    views = S.SearchViewProvider()
    if screener == "fallback":
        fb = S.FallbackScreener()
        screener_for = lambda cid: fb  # noqa: E731
    elif screener == "neural":
        fs = S.load_neural_fold_screeners(Path(neural_run_dir), folds, views, family_of=fam_of)
        screener_for = fs.for_context
    else:
        raise ValueError(screener)
    cd = D.NumpyContextCoder() if coder == "numpy" else D.NeuralContextCoder(views)
    return Runtime(engine, AuditLedger(engine), screener_for, cd, bank, fam_of, ds_of, Store(run_dir), tuple(seeds))


def finalize(rt: Runtime, run_dir: Path, summary: dict) -> dict:
    summary["guard"] = C.guard_report()
    summary["bank_counters"] = dict(D.BANK_COUNTERS)
    summary["audit_ledger"] = rt.ledger.check()
    summary["engine"] = rt.engine.stats()
    if summary["bank_counters"]["same_family_bank_hits"] != 0:
        raise AssertionError("same_family_bank_hits != 0")
    C.write_json(Path(run_dir) / "summary.json", summary)
    return summary


def cmd_run(args) -> None:
    run_dir = Path(args.run_dir)
    cores = [int(x) for x in args.cores.split(",")]
    rows = C.train_rows()
    folds = C.family_folds(rows)
    pilot = C.pilot_shard(rows, folds, n=args.pilot_n)
    if args.screener == "fallback" and not args.allow_fallback_screener:
        raise SystemExit("the fallback screener is for tests only (pass --allow-fallback-screener for a dry run)")
    if args.max_expanded_width:
        G.MAX_EXPANDED_WIDTH = int(args.max_expanded_width)  # declared run-plan knob (cost), recorded in config.json
    seeds = tuple(int(x) for x in args.seeds.split(","))
    rt = build_runtime(run_dir, cores, args.screener, args.coder, timeout_s=args.timeout,
                       neural_run_dir=args.neural_run_dir, seeds=seeds)
    C.write_json(run_dir / "config.json", {"argv": sys.argv, "pilot": pilot, "cores": cores, "seeds": list(seeds),
                                           "max_expanded_width": G.MAX_EXPANDED_WIDTH, "guard": C.guard_report()})
    summ = run_gate_h0(rt, pilot, run_dir, args.jobs, stages=args.stages.split(","))
    print(json.dumps(finalize(rt, run_dir, summ)["guard"]))


def cmd_timing(args) -> None:
    """Timing on NON-pilot TRAIN contexts only: per-operator child evals + one full H0.4->H0.6 job per context."""
    rows = C.train_rows()
    folds = C.family_folds(rows)
    cids = args.contexts.split(",")
    fam_of = {r["context_sha256"]: r["family_id"] for r in rows}
    protected = set()
    for n in (24, 48, 96):
        protected |= set(C.pilot_shard(rows, folds, n=n))
    pfam = {fam_of[c] for c in protected}
    for c in cids:
        if c in protected or fam_of[c] in pfam:
            raise SystemExit(f"{c} is a pilot context or shares a pilot family: timing refused")
    run_dir = Path(args.run_dir)
    cores = [int(x) for x in args.cores.split(",")]
    rt = build_runtime(run_dir, cores, "fallback", "numpy", gp_root=Path(args.gp_root), timeout_s=args.timeout)
    out = {"contexts": {}, "cores": cores}
    t_all = time.time()
    for cid in cids:
        t0 = time.time()
        rec = {"width": table_width(cid)}
        if args.operators:
            rec["operators"] = operator_timing(rt, cid, args.per_op)
        if args.full:
            stage_bank_nn1(rt, cid)
            for seed in rt.seeds[: args.full_seeds]:
                s0 = rt.engine.stats()
                tj = time.time()
                stage_h04(rt, cid, seed, seed == rt.seeds[0])
                stage_h05(rt, cid, seed, "C")
                stage_h06(rt, cid, seed, "C", "evoB")
                s1 = rt.engine.stats()
                rec[f"full_seed{seed}"] = {"wall_s": time.time() - tj,
                                           "physical_runs": s1["physical_runs"] - s0["physical_runs"],
                                           "eval_cpu_s": s1["wall_s_total"] - s0["wall_s_total"],
                                           "logical_requests": (s1["requests_search"] + s1["requests_audit"]) -
                                                               (s0["requests_search"] + s0["requests_audit"]),
                                           "failures": s1["failures"] - s0["failures"],
                                           "timeouts": s1["timeouts"] - s0["timeouts"]}
        rec["wall_s"] = time.time() - t0
        out["contexts"][cid] = rec
        C.write_json(run_dir / "timing.json", out)
    out["engine"] = rt.engine.stats()
    out["guard"] = C.guard_report()
    out["bank_counters"] = dict(D.BANK_COUNTERS)
    out["wall_s"] = time.time() - t_all
    out["walls_by_operator"] = _walls_by_operator(rt.engine.walls)
    C.write_json(run_dir / "timing.json", out)
    print(json.dumps({k: v for k, v in out.items() if k != "contexts"}, indent=1, default=C._json_default))


def _walls_by_operator(walls) -> dict:
    acc: Dict[str, List[Tuple[float, str]]] = {}
    for kind, op, w, err in walls:
        key = op.split("_i")[0]
        for pre in ("g1_", "g2_", "g3_", "g4_", "g5_", "g6_", "g7_", "g8_", "g9_", "g10_", "gen0_", "topk_"):
            if key.startswith(pre):
                key = key[len(pre):]
        if key.startswith("registry:"):
            key = "registry_preset"
        acc.setdefault(f"{kind}:{key}", []).append((w, err))
    return {k: {"n": len(v), "mean_s": float(np.mean([w for w, _ in v])), "p95_s": float(np.quantile([w for w, _ in v], 0.95)),
                "max_s": float(np.max([w for w, _ in v])), "fail": sum(1 for _, e in v if e)} for k, v in sorted(acc.items())}


def operator_timing(rt: Runtime, cid: str, per_op: int) -> dict:
    """Children per operator from the context's (family-excluded) bank NN parents; SEARCH envelope only."""
    import random

    g = rt.gp(cid, 0)
    with D.selection_scope("operator_timing"):
        nn = [D.bank_candidate(b) for _, b in D.bank_nn(g.tb, g.coder, 24)]
        rng = random.Random(C.stream_seed("H317-gp-optiming", cid))
        pg = [c.genome() for c in nn[:8]]
        ops: Dict[str, List[G.Candidate]] = {}

        def make(name, fn, n=per_op):
            out, tries = [], 0
            while len(out) < n and tries < n * 30:
                tries += 1
                i = rng.randrange(len(pg))
                c = G.safe_candidate(fn(i), f"timing_{name}", [nn[i].sha], g.ctx.width, bank_cids=D.merge_lineage(nn[i]))
                if c is not None and c.sha not in {x.sha for x in out} and c.sha not in g.tb.excluded_shas:
                    out.append(c)
            ops[name] = out

        make("bank_nn", lambda i: pg[i])
        make("edit_parameter", lambda i: G.op_parameter(pg[i], rng))
        make("edit_insert_stage", lambda i: G.op_insert_stage(pg[i], rng))
        make("edit_delete_stage", lambda i: G.op_delete_stage(pg[i], rng))
        make("edit_crossover", lambda i: G.op_crossover(pg[i], pg[(i + 1) % len(pg)], rng))
        for isl in range(4):
            make(f"replace_head_i{isl}", lambda i, isl=isl: G.op_replace_head(pg[i], rng, isl))
            make(f"synth_i{isl}", lambda i, isl=isl: G.op_synth(isl, rng))
        make("migration_portfolio", lambda i: G.op_portfolio([pg[i], G.op_synth(rng.choice([1, 2]), rng)]))
        make("line_search", lambda i: rng.choice(G.line_search_neighbors(pg[i]) or [pg[i]]))
        make("bag_final", lambda i: G.bag_final(pg[i]))
        ops["registry_preset"] = D.registry_candidates(g.ctx.width)
    out = {}
    for name, cands in ops.items():
        s0 = len(rt.engine.walls)
        with D.selection_scope("operator_timing"):
            res = g.evaluate(cands)
        walls = [w for _, _, w, _ in rt.engine.walls[s0:]]
        out[name] = {"n": len(res), "failed": sum(r.status != "completed" for r in res),
                     "gate_pass": sum(bool(r.gate) for r in res), "errors": sorted({r.error for r in res if r.error}),
                     "mean_wall_s": float(np.mean(walls)) if walls else None,
                     "max_wall_s": float(np.max(walls)) if walls else None}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--run-dir", required=True)
    r.add_argument("--cores", required=True, help="<= 4 idle cores disjoint from other lanes, e.g. 1,3,5,11")
    r.add_argument("--jobs", type=int, default=8, help="concurrent (context, seed) jobs feeding the eval pool")
    r.add_argument("--pilot-n", type=int, default=C.PILOT_N)
    r.add_argument("--screener", choices=("neural", "fallback"), default="neural")
    r.add_argument("--neural-run-dir", default=None)
    r.add_argument("--coder", choices=("neural", "numpy"), default="neural")
    r.add_argument("--allow-fallback-screener", action="store_true")
    r.add_argument("--stages", default="h04,h05,h06")
    r.add_argument("--timeout", type=float, default=60.0)
    r.add_argument("--seeds", default=",".join(str(x) for x in C.SEARCH_SEEDS))
    r.add_argument("--max-expanded-width", type=int, default=0, help="0 = grammar default (3000)")
    r.set_defaults(fn=cmd_run)
    t = sub.add_parser("timing")
    t.add_argument("--run-dir", required=True)
    t.add_argument("--gp-root", default=str(GP_ROOT / "timing_root"))
    t.add_argument("--contexts", required=True)
    t.add_argument("--cores", required=True)
    t.add_argument("--per-op", type=int, default=4)
    t.add_argument("--operators", action="store_true")
    t.add_argument("--full", action="store_true")
    t.add_argument("--full-seeds", type=int, default=1)
    t.add_argument("--timeout", type=float, default=60.0)
    t.set_defaults(fn=cmd_timing)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
