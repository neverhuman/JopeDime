#!/usr/bin/env python3
"""H317 Gate H0 GP block (H0.4 seeding -> H0.5 evolution -> H0.6 stacking) + Stage V GP pipelines + timing.

Run order over the pilot (`pilot_shard(rows, folds)`, default PILOT_N = all 205 families) x search seeds
(317101, 317102; both required):
  BANK_NN1   (per context, 1 AUDIT eval)                 -> reference pipeline (no search)
  H0.4       Seed A/B/C/D/E (+ NOISE_PROBE on seed 317101) -> decision: Holm over m=4 challengers vs A (Holm on the
             bootstrap LCB and on the one-sided sign-flip level, 0.025/(m-rank))
  H0.5       Evo A vs Evo B from the adopted seed set, 10 generations, 48 screened / exactly 6 live evals per gen
  H0.6       single / CV1 / CV2@N / CV3@N / CV4 / CV5 (+ BAG_FINAL, SEL_MEANLR secondaries) on the adopted arm
             -> leave-one-family-out config selection; test = CV-selected stack - single best tree
Every within-context choice uses SEARCH results only (inside `selection_scope`); each declared output is AUDIT-scored
exactly once afterwards (`AuditLedger`). Per-context values are averaged over the two search seeds, then
family-equal means / paired whole-family statistics (common_h317, RNG 317150).

Failure policy (AUDIT scale, leakage red team rule 3): AUDIT status is "completed" whenever the worker exits 0 with
finite predictions, and the raw continuous AUDIT utility is kept. A crashed / non-finite output (or a pipeline with no
evaluable output) is scored as the SEARCH-target-mean predictor (utility 0.0) and is INVALID; a crashed stack member
contributes the SEARCH mean (counted as `substituted_weight_mass`; the stack is INVALID only if every member crashed);
weights are never changed by AUDIT outcomes. `value_m2` stores the -2.0 convention for sensitivity.
BANK_NN1 is run as a reference only.

Resumable: one append-only JSONL per (stage, context, seed) under <run_dir>/stages/; a stage is skipped when its
`result` record exists. The engine memo (<gp_root>/memo/<cid>.<mode>.jsonl) makes interrupted stages cheap to repeat.

Multi-process layout (xbabe0, owner 13:16Z: host CPU <= 85 %): `launch --cpus 16-63,80-127 --shards 11
--workers-per-shard 8` runs, per stage, 11 shard processes (`run --shard k/11`, contexts with sorted index
i % 11 == k, each process pinned to its own 8 worker cores), waits, then `merge` (refuses missing / duplicated
(context, seed, stage) records), pins the stage decision in <run_dir>/decisions.json and starts the next stage.
`run --shard` / `merge` can also be driven by hand; config.json pins pilot / seeds / screener for every shard.
"""

from __future__ import annotations

import os

for _k in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "RAYON_NUM_THREADS",
):
    os.environ[_k] = (
        "1"  # numeric thread cap: the Rust workers are the only compute threads
    )

import argparse  # noqa: E402  (after the thread-cap env block)
import concurrent.futures as cf  # noqa: E402  (after the thread-cap env block)
import hashlib  # noqa: E402  (after the thread-cap env block)
import json  # noqa: E402  (after the thread-cap env block)
import struct  # noqa: E402  (after the thread-cap env block)
import sys  # noqa: E402  (after the thread-cap env block)
import threading  # noqa: E402  (after the thread-cap env block)
import time  # noqa: E402  (after the thread-cap env block)
from dataclasses import dataclass, field  # noqa: E402  (after the thread-cap env block)
from pathlib import Path  # noqa: E402  (after the thread-cap env block)
from typing import Callable, Dict, List, Optional, Sequence, Tuple  # noqa: E402  (after the thread-cap env block)

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
from gp_engine_h317 import (  # noqa: E402
    GP_ROOT,
    AuditLedger,
    EvalEngine,
    ResourceGuard,
    SearchResult,
    audit_half_utilities,
)

H04_CHALLENGERS = ("B", "C", "D", "E")
H06_SECONDARY = ("BAG_FINAL", "SEL_MEANLR")
RUN_MAX_EXPANDED_WIDTH = 1000  # H317 run default (red team MEDIUM-4); grammar default 3000 applies only in tests
MAX_WORKERS = 120  # sanity bound per process; the binding limit is the host-CPU <= 85 % resource guard (owner 13:16Z)


# ----------------------------------------------------------------------------------------------------------------------
# Candidate (de)serialization
# ----------------------------------------------------------------------------------------------------------------------
def cand_json(c: G.Candidate) -> dict:
    return {
        "dsl": c.dsl,
        "sha": c.sha,
        "operator": c.operator,
        "parents": list(c.parents),
        "island": c.island,
        "meta": [list(x) for x in c.meta],
    }


def cand_from_json(d: dict) -> G.Candidate:
    return G.Candidate(
        d["dsl"],
        d["sha"],
        d["operator"],
        tuple(d["parents"]),
        int(d["island"]),
        tuple(tuple(x) for x in d["meta"]),
    )


def res_brief(r: Optional[SearchResult]) -> Optional[dict]:
    if r is None:
        return None
    return {
        "sha": r.sha,
        "utility": r.utility,
        "fitness": r.fitness,
        "gate": r.gate,
        "mean_lr": r.mean_lr,
        "sd_lr": r.sd_lr,
        "status": r.status,
        "family": G.head_family(r.dsl),
        "island": G.island_of(r.dsl),
    }


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

    def n_results(self, stage: str, cid: str, seed) -> int:
        p = self.path(stage, cid, seed)
        if not p.exists():
            return 0
        n = 0
        with open(p) as f:
            for line in f:
                try:
                    n += json.loads(line).get("type") == "result"
                except json.JSONDecodeError:
                    continue
        return n

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
        return D.GPContext(
            ctx,
            self.engine,
            self.screener_for(cid),
            self.target_bank(cid),
            self.coder,
            seed,
        )


def audit_one(
    rt: Runtime,
    pipeline: str,
    g: D.GPContext,
    cand: Optional[G.Candidate],
    halves: bool = False,
) -> dict:
    """One AUDIT request for a fixed single output. `halves` (Stage V records, A3) adds the split-half diagnostic
    `audit_half_utilities` from the same fit (mean predictor = [0, 0] for no output / a crash)."""
    if cand is None:
        out = {"status": "no_output", "utility": None}
        if halves:
            out["audit_half_utilities"] = [0.0, 0.0]
        return out
    a = rt.ledger.score(pipeline, g.ctx, g.seed, [cand])[0]
    out = {
        "status": a.status,
        "utility": a.utility if a.status == "completed" else None,
        "sha": cand.sha,
    }
    if halves:
        pred = a.pa if a.status == "completed" else None
        out["audit_half_utilities"] = audit_half_utilities(
            g.cid, pred, g.ctx.ya, g.ctx.y_search_mean
        )
    return out


def audit_stack(
    rt: Runtime,
    pipeline: str,
    g: D.GPContext,
    spec: ST.StackSpec,
    cand_of: Dict[str, G.Candidate],
    halves: bool = False,
) -> dict:
    shas, _, _ = spec.declared()
    res = rt.ledger.score(pipeline, g.ctx, g.seed, [cand_of[s] for s in shas])
    out = ST.stack_audit_utility(
        spec,
        {a.sha: a for a in res},
        g.ctx.ya,
        g.ctx.ba,
        g.ctx.y_search_mean,
        return_pred=halves,
    )
    out["n_members"] = len(shas)
    if halves:
        pred = out.pop("pred")
        out["audit_half_utilities"] = audit_half_utilities(
            g.cid, pred, g.ctx.ya, g.ctx.y_search_mean
        )
    return out


MEAN_PREDICTOR_U = 0.0


INVALID_RULE = (
    "an output is INVALID iff its entire AUDIT prediction is the SEARCH-mean substitution (no output, a "
    "crashed single output, or every stack member crashed); partial member substitution is valid and "
    "reported as substituted_weight_mass"
)


def value_of(raw: dict) -> Tuple[float, float, bool]:
    """(value, -2.0-convention value, invalid?) under INVALID_RULE (red team MEDIUM-3)."""
    u = raw.get("utility")
    if u is None or raw.get("status") != "completed":
        return MEAN_PREDICTOR_U, C.GATE_FAIL_UTILITY, True
    invalid = bool(raw.get("all_substituted", False))
    return float(u), (C.GATE_FAIL_UTILITY if invalid else float(u)), invalid


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
    rec = {
        "type": "result",
        "cid": cid,
        "audit": a,
        "value": v[0],
        "value_m2": v[1],
        "invalid": v[2],
        "cand": cand_json(cand) if cand else None,
        "bank_exclusion": g.tb.exclusion_record(),
        "coder": g.coder.name,
    }
    rt.store.append("bank_nn1", cid, "na", rec)
    return rec


def stage_h04(rt: Runtime, cid: str, seed: int, noise_probe: bool) -> dict:
    hit = rt.store.result("h04", cid, seed)
    if hit:
        return hit
    g = rt.gp(cid, seed)
    t0 = time.time()
    runs = D.run_h04(g)
    rec = {
        "type": "result",
        "cid": cid,
        "seed": seed,
        "schemes": {},
        "screener": g.screener.describe(),
    }
    for name, run in runs.items():
        m = run.search_metrics()
        m["order"] = [cand_json(c) for c in run.order]
        m["info"] = run.info
        cand_of = {c.sha: c for c in run.order}
        for tag, best in (("full", run.best()), ("bm", run.budget_matched())):
            raw = audit_one(
                rt, f"H04.{name}.{tag}", g, cand_of[best.sha] if best else None
            )
            pv, rv, inv = value_of(raw)
            m[f"audit_{tag}"] = {
                "raw": raw,
                "value": pv,
                "value_m2": rv,
                "invalid": inv,
            }
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
    shortfall = {
        a: r.summary().get("budget_shortfall") for a, r in runs.items() if r.dedup
    }
    if (
        runs["evoA"].live_per_gen != runs["evoB"].live_per_gen
        or runs["evoA"].screened_per_gen != runs["evoB"].screened_per_gen
    ) and not any(shortfall.values()):
        raise AssertionError("evo arms broke the matched budget")
    if any(
        shortfall.values()
    ):  # A4: fewer than 6 distinct new programs left in a generation (recorded, not fatal)
        rec["budget_shortfall"] = shortfall
    for arm, run in runs.items():
        summ = run.summary()
        summ["cands"] = {sha: cand_json(c) for sha, c in run.cand_of.items()}
        for gen in D.AUDIT_GENS:
            b = run.best_after(gen)
            raw = audit_one(
                rt, f"H05.{arm}.g{gen}", g, run.cand_of[b.sha] if b else None
            )
            pv, rv, inv = value_of(raw)
            summ[f"g{gen}"] = {
                "best": res_brief(b),
                "raw": raw,
                "value": pv,
                "value_m2": rv,
                "invalid": inv,
            }
        rec["arms"][arm] = summ
    rec["wall_s"] = time.time() - t0
    rt.store.append(key, cid, seed, rec)
    return rec


def _arm_results(
    rt: Runtime, cid: str, seed: int, scheme: str, arm: str, g: D.GPContext
):
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
    rec = {
        "type": "result",
        "cid": cid,
        "seed": seed,
        "scheme": scheme,
        "arm": arm,
        "configs": {},
    }
    # reference first: aliases (fallback configs) reuse its value without a second AUDIT request
    if single is None:
        s_raw = {"status": "no_output", "utility": None}
    else:
        s_raw = audit_one(rt, "H06.single", g, cand_of[single.members[0]])
    sv = value_of(s_raw)
    rec["configs"]["single"] = {
        "spec": single.to_json() if single else None,
        "raw": s_raw,
        "value": sv[0],
        "value_m2": sv[1],
        "invalid": sv[2],
        "search_s1": single.search if single else None,
    }
    for name, spec in specs.items():
        if name == "single":
            continue
        if spec.fallback_single:
            rec["configs"][name] = {
                "alias": "single",
                "value": sv[0],
                "value_m2": sv[1],
                "invalid": sv[2],
                "spec": spec.to_json(),
                "search_s1": spec.search,
            }
            continue
        raw = audit_stack(rt, f"H06.{name}", g, spec, cand_of)
        pv = value_of(raw)
        rec["configs"][name] = {
            "raw": raw,
            "value": pv[0],
            "value_m2": pv[1],
            "invalid": pv[2],
            "spec": spec.to_json(),
            "search_s1": spec.search,
        }
    for prog in (cv5, bag):
        if prog.fallback or prog.cand is None:
            rec["configs"][prog.name] = {
                "alias": "single",
                "value": sv[0],
                "value_m2": sv[1],
                "invalid": sv[2],
                "info": prog.info,
                "search": res_brief(prog.result),
                "live_search_evals": prog.live_evals,
            }
            continue
        raw = audit_one(rt, f"H06.{prog.name}", g, prog.cand)
        pv = value_of(raw)
        rec["configs"][prog.name] = {
            "raw": raw,
            "value": pv[0],
            "value_m2": pv[1],
            "invalid": pv[2],
            "info": prog.info,
            "search": res_brief(prog.result),
            "live_search_evals": prog.live_evals,
        }
    if sml is None:
        rec["configs"]["SEL_MEANLR"] = {
            "alias": "single",
            "value": sv[0],
            "value_m2": sv[1],
            "invalid": sv[2],
        }
    else:
        raw = audit_one(rt, "H06.SEL_MEANLR", g, cand_of[sml.sha])
        pv = value_of(raw)
        rec["configs"]["SEL_MEANLR"] = {
            "raw": raw,
            "value": pv[0],
            "value_m2": pv[1],
            "invalid": pv[2],
            "search": res_brief(sml),
        }
    for name in (
        H06_CONFIGS
    ):  # every config exists for every context (missing = could not be built -> single)
        if name not in rec["configs"]:
            rec["configs"][name] = {
                "alias": "single(not_buildable)",
                "value": sv[0],
                "value_m2": sv[1],
                "invalid": sv[2],
            }
    rec["wall_s"] = time.time() - t0
    rt.store.append(key, cid, seed, rec)
    return rec


H06_CONFIGS = (
    ("CV1", "CV4", "CV5")
    + tuple(f"CV2@{n}" for n in ST.N_GRID)
    + tuple(f"CV3@{n}" for n in ST.N_GRID)
    + ("BAG_FINAL", "SEL_MEANLR")
)


# ----------------------------------------------------------------------------------------------------------------------
# Decisions
# ----------------------------------------------------------------------------------------------------------------------
def seed_mean(per_seed: Dict[Tuple[str, int], float]) -> Dict[str, float]:
    acc: Dict[str, List[float]] = {}
    for (cid, _), v in per_seed.items():
        acc.setdefault(cid, []).append(float(v))
    return {c: float(np.mean(v)) for c, v in acc.items()}


def contrast(
    arm: Dict[str, float],
    ref: Dict[str, float],
    fam_of: Dict[str, str],
    alpha: Optional[float] = None,
    signflip_alpha: float = 0.025,
) -> dict:
    """`alpha`: Holm two-sided level for the bootstrap LCB; `signflip_alpha`: the rank's one-sided Holm level for the
    sign-flip p (red team MEDIUM-1), both from the same p ranking."""
    summ = C.robustness_summary(arm, ref, fam_of, C.BOOT_SEED_GATE_H0)
    summ["signflip_alpha"] = signflip_alpha
    if alpha is not None:
        summ["holm_alpha"] = alpha
        summ["holm_lcb"] = C.paired_family_bootstrap_alpha(
            arm, ref, fam_of, C.BOOT_SEED_GATE_H0, alpha
        )["lcb"]
        summ["gate"] = C.contract_advance_gate(
            summ, "holm_lcb", signflip_alpha=signflip_alpha
        )
    else:
        summ["gate"] = C.contract_advance_gate(
            summ, "lcb95", signflip_alpha=signflip_alpha
        )
    return summ


def decide_h04(values: Dict[str, Dict[str, float]], fam_of: Dict[str, str]) -> dict:
    """Holm step-down over m = 4 challengers (B, C, D, E vs A) on budget-matched AUDIT utility."""
    ref = values["A"]
    m = len(H04_CHALLENGERS)
    ps = {
        s: C.signflip_pvalue(values[s], ref, fam_of, C.BOOT_SEED_GATE_H0)
        for s in H04_CHALLENGERS
    }
    order = sorted(H04_CHALLENGERS, key=lambda s: (ps[s], s))
    out, alive = {}, True
    for rank, s in enumerate(order):
        c = contrast(
            values[s],
            ref,
            fam_of,
            C.holm_adjusted_lcb_alpha(m, rank),
            signflip_alpha=0.025 / (m - rank),
        )
        c["holm_rank"] = rank
        c["pass"] = bool(alive and c["gate"]["pass"])
        alive = c["pass"]
        out[s] = c
    passing = [s for s in H04_CHALLENGERS if out[s]["pass"]]
    adopted = (
        max(passing, key=lambda s: (C.family_equal_mean(values[s], fam_of), s))
        if passing
        else "A"
    )
    return {
        "contrasts": out,
        "adopted": adopted,
        "family_means": {s: C.family_equal_mean(v, fam_of) for s, v in values.items()},
    }


def decide_h05(values: Dict[str, Dict[str, float]], fam_of: Dict[str, str]) -> dict:
    c = contrast(values["evoB.g10"], values["evoA.g10"], fam_of)
    return {
        "contrast_evoB_vs_evoA_g10": c,
        "adopted": "evoB" if c["gate"]["pass"] else "evoA",
        "generations": 10,
        "family_means": {k: C.family_equal_mean(v, fam_of) for k, v in values.items()},
    }


def decide_h06(values: Dict[str, Dict[str, float]], fam_of: Dict[str, str]) -> dict:
    """Leave-one-family-out config choice; test = CV-selected - single; locked = best over all pilot families."""
    configs = sorted(k for k in values if k not in ("single",) + H06_SECONDARY)
    cids = sorted(values["single"])
    fams = sorted({fam_of[c] for c in cids})
    fm = {k: C.family_means(values[k], fam_of) for k in configs}
    sel, chosen = {}, {}
    for c in cids:
        f = fam_of[c]
        best = max(
            configs,
            key=lambda k: (float(np.mean([fm[k][x] for x in fams if x != f])), k),
        )
        chosen[c] = best
        sel[c] = values[best][c]
    test = contrast(sel, values["single"], fam_of)
    locked = max(configs, key=lambda k: (float(np.mean(list(fm[k].values()))), k))
    out = {
        "lofo_test_vs_single": test,
        "lofo_choices": chosen,
        "locked_config_if_adopted": locked,
        "adopted": locked if test["gate"]["pass"] else "single",
        "family_means": {k: C.family_equal_mean(v, fam_of) for k, v in values.items()},
        "direct": {},
    }
    for k in ("CV5",) + H06_SECONDARY:
        if k in values:
            out["direct"][f"{k}_vs_single"] = contrast(
                values[k], values["single"], fam_of
            )
    return out


# ----------------------------------------------------------------------------------------------------------------------
# Lab driver
# ----------------------------------------------------------------------------------------------------------------------
def run_parallel(fn, jobs: Sequence[tuple], n: int) -> List:
    out = [None] * len(jobs)
    with cf.ThreadPoolExecutor(
        max_workers=max(1, n), thread_name_prefix="h317gp-job"
    ) as ex:
        futs = {ex.submit(fn, *j): i for i, j in enumerate(jobs)}
        for f in cf.as_completed(futs):
            out[futs[f]] = f.result()
    return out


STAGES = ("h04", "h05", "h06")


def stage_key(
    stage: str, scheme: Optional[str] = None, arm: Optional[str] = None
) -> str:
    return {
        "bank_nn1": "bank_nn1",
        "h04": "h04",
        "h05": f"h05__{scheme}",
        "h06": f"h06__{scheme}__{arm}",
    }[stage]


def run_stage(
    rt: Runtime,
    stage: str,
    cids: Sequence[str],
    job_threads: int,
    scheme: Optional[str] = None,
    arm: Optional[str] = None,
) -> List[dict]:
    """Run one stage for the given contexts x rt.seeds (resumable: finished records are returned as-is)."""
    if stage == "bank_nn1":
        return run_parallel(
            lambda c: stage_bank_nn1(rt, c), [(c,) for c in cids], job_threads
        )
    jobs = [(c, s) for c in cids for s in rt.seeds]
    if stage == "h04":
        return run_parallel(
            lambda c, s: stage_h04(rt, c, s, s == rt.seeds[0]), jobs, job_threads
        )
    if stage == "h05":
        return run_parallel(lambda c, s: stage_h05(rt, c, s, scheme), jobs, job_threads)
    if stage == "h06":
        return run_parallel(
            lambda c, s: stage_h06(rt, c, s, scheme, arm), jobs, job_threads
        )
    raise ValueError(stage)


class MergeError(RuntimeError):
    pass


class IncompleteStage(MergeError):
    """Records missing (stage not finished): merge stops there."""


class DuplicateRecord(MergeError):
    """A (context, seed, stage) record exists more than once or under the wrong identity: always fatal."""


def collect(
    store: Store,
    stage: str,
    cids: Sequence[str],
    seeds: Sequence[int],
    scheme: Optional[str] = None,
    arm: Optional[str] = None,
) -> List[dict]:
    """All result records of a stage; refuses on any missing or duplicated (context, seed, stage) record."""
    key = stage_key(stage, scheme, arm)
    keys = (
        [(c, "na") for c in cids]
        if stage == "bank_nn1"
        else [(c, s) for c in cids for s in seeds]
    )
    missing, dup, out = [], [], []
    for c, s in keys:
        n = store.n_results(key, c, s)
        if n == 0:
            missing.append((c[:12], s))
            continue
        if n > 1:
            dup.append((c[:12], s, n))
            continue
        r = store.result(key, c, s)
        if r.get("cid") != c or (stage != "bank_nn1" and r.get("seed") != s):
            dup.append((c[:12], s, "identity"))
            continue
        out.append(r)
    if dup:
        raise DuplicateRecord(
            f"{key}: {len(dup)} duplicated / mis-identified records {dup[:5]}"
        )
    if missing:
        raise IncompleteStage(f"{key}: {len(missing)} missing {missing[:5]}")
    return out


def decide_stage(stage: str, recs: List[dict], fam_of: Dict[str, str]) -> dict:
    if stage == "h04":
        per = {sch: {} for sch in D.SCHEMES}
        per_full = {sch: {} for sch in D.SCHEMES}
        for r in recs:
            for sch in D.SCHEMES:
                per[sch][(r["cid"], r["seed"])] = r["schemes"][sch]["audit_bm"]["value"]
                per_full[sch][(r["cid"], r["seed"])] = r["schemes"][sch]["audit_full"][
                    "value"
                ]
        d = decide_h04({k: seed_mean(v) for k, v in per.items()}, fam_of)
        d["full_budget_family_means"] = {
            k: C.family_equal_mean(seed_mean(v), fam_of) for k, v in per_full.items()
        }
        d["search_metrics_family_means"] = _h04_search_means(recs, fam_of)
        d["noise_probe"] = _noise_summary(recs)
        return d
    if stage == "h05":
        per5: Dict[str, Dict] = {}
        for r in recs:
            for a_name, a in r["arms"].items():
                for gen in D.AUDIT_GENS:
                    per5.setdefault(f"{a_name}.g{gen}", {})[(r["cid"], r["seed"])] = a[
                        f"g{gen}"
                    ]["value"]
        return decide_h05({k: seed_mean(v) for k, v in per5.items()}, fam_of)
    if stage == "h06":
        per6: Dict[str, Dict] = {}
        for r in recs:
            for k, v in r["configs"].items():
                per6.setdefault(k, {})[(r["cid"], r["seed"])] = v["value"]
        d = decide_h06({k: seed_mean(v) for k, v in per6.items()}, fam_of)
        d["in_sample_search_s1_family_means"] = _h06_search_means(recs, fam_of)
        return d
    raise ValueError(stage)


def run_gate_h0(
    rt: Runtime,
    cids: Sequence[str],
    run_dir: Path,
    job_threads: int,
    stages: Sequence[str] = STAGES,
    overrides: Optional[dict] = None,
) -> dict:
    """Single-process Gate H0 (tests, small runs): every stage for all contexts, decisions in between."""
    overrides = overrides or {}
    fam_of = rt.fam_of
    t0 = time.time()
    nn1 = run_stage(rt, "bank_nn1", cids, job_threads)
    C.write_json(
        run_dir / "bank_exclusions.json", {r["cid"]: r["bank_exclusion"] for r in nn1}
    )
    summary: dict = {
        "contexts": len(cids),
        "seeds": list(rt.seeds),
        "bank_nn1_family_mean": C.family_equal_mean(
            {r["cid"]: r["value"] for r in nn1}, fam_of
        ),
        "bank_nn1_invalid": int(sum(r["invalid"] for r in nn1)),
    }
    scheme = arm = None
    recs: Dict[str, List[dict]] = {}
    for stage in STAGES:
        if stage not in stages:
            break
        recs[stage] = run_stage(rt, stage, cids, job_threads, scheme, arm)
        d = decide_stage(stage, recs[stage], fam_of)
        summary[{"h04": "H0.4", "h05": "H0.5", "h06": "H0.6"}[stage]] = d
        if stage == "h04":
            scheme = overrides.get("h04_scheme", d["adopted"])
            d["used_scheme"] = scheme
        elif stage == "h05":
            arm = overrides.get("h05_arm", d["adopted"])
            d["used_arm"] = arm
    if all(st in recs for st in STAGES):
        summary["invalids"] = _invalid_counts(recs["h04"], recs["h05"], recs["h06"])
    summary["wall_s"] = time.time() - t0
    return summary


def merge_run(
    run_dir: Path, fam_of: Dict[str, str], overrides: Optional[dict] = None
) -> dict:
    """Rebuild summary + decisions from all shard records. Decisions are pinned once written (decisions.json): a
    later merge that would change a pinned decision raises."""
    overrides = overrides or {}
    cfg = json.loads((Path(run_dir) / "config.json").read_text())
    cids, seeds = cfg["pilot"], cfg["seeds"]
    store = Store(run_dir)
    dpath = Path(run_dir) / "decisions.json"
    pinned = json.loads(dpath.read_text()) if dpath.exists() else {}
    nn1 = collect(store, "bank_nn1", cids, seeds)
    summary: dict = {
        "contexts": len(cids),
        "seeds": seeds,
        "pilot_sha256": cfg["pilot_sha256"],
        "bank_nn1_family_mean": C.family_equal_mean(
            {r["cid"]: r["value"] for r in nn1}, fam_of
        ),
        "bank_nn1_invalid": int(sum(r["invalid"] for r in nn1)),
    }
    C.write_json(
        Path(run_dir) / "bank_exclusions.json",
        {r["cid"]: r["bank_exclusion"] for r in nn1},
    )
    scheme = arm = None
    recs: Dict[str, List[dict]] = {}
    for stage in STAGES:
        try:
            recs[stage] = collect(store, stage, cids, seeds, scheme, arm)
        except IncompleteStage as e:
            summary["incomplete"] = str(e)
            break
        d = decide_stage(stage, recs[stage], fam_of)
        name = {"h04": "H0.4", "h05": "H0.5", "h06": "H0.6"}[stage]
        if stage == "h04":
            scheme = overrides.get("h04_scheme", d["adopted"])
            d["used_scheme"] = scheme
            used = scheme
        elif stage == "h05":
            arm = overrides.get("h05_arm", d["adopted"])
            d["used_arm"] = arm
            used = arm
        else:
            used = d["adopted"]
        if name in pinned and pinned[name]["used"] != used:
            raise MergeError(
                f"{name}: pinned decision {pinned[name]['used']} != recomputed {used}"
            )
        pinned.setdefault(
            name, {"used": used, "adopted": d["adopted"], "pinned_utc": time.time()}
        )
        summary[name] = d
    C.write_json(dpath, pinned)
    if all(st in recs for st in STAGES):
        summary["invalids"] = _invalid_counts(recs["h04"], recs["h05"], recs["h06"])
    summary["decisions"] = pinned
    return summary


def _h04_search_means(h04, fam_of) -> dict:
    out = {}
    for sch in D.SCHEMES:
        for key in (
            "best_search_utility",
            "bm_best_search_utility",
            "mean_passing_search_utility",
            "n_families_passing",
            "evals_used",
        ):
            v = {
                (r["cid"], r["seed"]): r["schemes"][sch][key]
                for r in h04
                if r["schemes"][sch][key] is not None
            }
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
    rows = [
        r["noise_probe"]
        for r in h04
        if r.get("noise_probe", {}).get("n_candidates", 0) >= 2
    ]
    if not rows:
        return {}
    return {
        k: float(np.mean([x[k] for x in rows]))
        for k in (
            "shared_offset_sd",
            "candidate_specific_sd",
            "raw_candidate_sd",
            "rank_agreement_spearman",
        )
    }


def _invalid_counts(h04, h05, h06) -> dict:
    out: Dict[str, int] = {}
    for r in h04:
        for sch, m in r["schemes"].items():
            for tag in ("full", "bm"):
                out[f"H04.{sch}.{tag}"] = out.get(f"H04.{sch}.{tag}", 0) + int(
                    m[f"audit_{tag}"]["invalid"]
                )
    for r in h05:
        for arm, a in r["arms"].items():
            for gen in D.AUDIT_GENS:
                out[f"H05.{arm}.g{gen}"] = out.get(f"H05.{arm}.g{gen}", 0) + int(
                    a[f"g{gen}"]["invalid"]
                )
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


def _stack_output(
    rt: Runtime,
    pipeline: str,
    g: D.GPContext,
    res: Sequence[SearchResult],
    cand_of: Dict[str, G.Candidate],
    config: str,
) -> dict:
    """Fix the output of a stack config from SEARCH data, then AUDIT it once. Returns value + live SEARCH evals."""
    fd = ST.FoldData.from_ctx(g.ctx)
    extra = 0
    with D.selection_scope(f"{pipeline}.stack"):
        single = ST.single_spec(res)
    if config in ("CV5",):
        prog = D.portfolio_nnls(g, res, cand_of)
        extra = prog.live_evals
        if not prog.fallback and prog.cand is not None:
            raw = audit_one(rt, pipeline, g, prog.cand, halves=True)
            v = value_of(raw)
            return {
                "config": config,
                "raw": raw,
                "value": v[0],
                "value_m2": v[1],
                "invalid": v[2],
                "extra_live": extra,
                "search_utility": prog.result.utility,
                "audit_half_utilities": raw["audit_half_utilities"],
            }
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
        return {
            "config": config,
            "raw": {"status": "no_output"},
            "value": MEAN_PREDICTOR_U,
            "value_m2": C.GATE_FAIL_UTILITY,
            "invalid": True,
            "extra_live": extra,
            "search_utility": None,
            "audit_half_utilities": [0.0, 0.0],
        }
    if len(spec.declared()[0]) == 1:
        raw = audit_one(rt, pipeline, g, cand_of[spec.declared()[0][0]], halves=True)
    else:
        raw = audit_stack(rt, pipeline, g, spec, cand_of, halves=True)
    v = value_of(raw)
    return {
        "config": config,
        "raw": raw,
        "value": v[0],
        "value_m2": v[1],
        "invalid": v[2],
        "extra_live": extra,
        "spec": spec.to_json(),
        # in-sample SEARCH utility of the declared output: the tree's own 2-fold S1, or the stack's cross-fitted S1
        "search_utility": spec.search.get("utility"),
        "audit_half_utilities": raw["audit_half_utilities"],
    }


def stage_v_pipeline(
    rt: Runtime,
    name: str,
    cid: str,
    seed: int,
    locked: LockedConfig,
    k_topk: Optional[int] = None,
    full_mode: bool = False,
) -> dict:
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
        raise C.GuardViolation(
            "Stage V pipeline on a DL-Val context before the LOCK unlock"
        )
    g = rt.gp(cid, seed, full_mode=full_mode)
    pipe = f"V.{name}"
    if name == "BANK_NN1":
        with D.selection_scope(pipe):
            cand = D.bank_nn1(g.tb, g.coder)
        raw = audit_one(rt, pipe, g, cand, halves=True)
        v = value_of(raw)
        return {
            "pipeline": name,
            "cid": cid,
            "seed": seed,
            "live_evals": 0,
            "audit_evals": 1,
            "value": v[0],
            "value_m2": v[1],
            "invalid": v[2],
            "raw": raw,
            "search_utility": None,  # BANK_NN1 makes no SEARCH evaluation
            "audit_half_utilities": raw["audit_half_utilities"],
        }
    if name in ("BANK_NN8_PICK", "BANK_NN8_PORTFOLIO"):
        run = D.bank_nn8_pick(g)
        cand_of = {c.sha: c for c in run.order}
        out = _stack_output(
            rt,
            pipe,
            g,
            run.results,
            cand_of,
            "single" if name == "BANK_NN8_PICK" else "CV5",
        )
        out.update(
            pipeline=name,
            cid=cid,
            seed=seed,
            live_evals=run.evals_used + out["extra_live"],
        )
        return out
    if name == "H257_REPLICA":
        a = D.seed_A(g)
        evo = D.h257_replica_evo(g, a.order, a.results)
        cand_of = {**{c.sha: c for c in a.order}, **evo.cand_of}
        out = _stack_output(rt, pipe, g, evo.results, cand_of, "CV1")
        out.update(
            pipeline=name,
            cid=cid,
            seed=seed,
            live_evals=a.evals_used + evo.live_evals + out["extra_live"],
        )
        return out
    if name in ("H317_LOCKED", "EVOA_MATCHED"):
        sr = D.run_scheme(g, locked.scheme)
        arm_fn = D.ARMS[locked.arm] if name == "H317_LOCKED" else D.evo_panmictic
        evo = arm_fn(g, sr.order, sr.results, gens=locked.generations)
        cand_of = {**{c.sha: c for c in sr.order}, **evo.cand_of}
        cfg = locked.stack if name == "H317_LOCKED" else "CV1"
        out = _stack_output(rt, pipe, g, evo.results, cand_of, cfg)
        out.update(
            pipeline=name,
            cid=cid,
            seed=seed,
            seed_evals=sr.evals_used,
            evo_live=evo.live_evals,
            live_evals=sr.evals_used + evo.live_evals + out["extra_live"],
        )
        return out
    if name == "SCREEN_TOPK_PICK":
        if k_topk is None:
            raise ValueError("SCREEN_TOPK_PICK needs K = H317_LOCKED live evals")
        run = D.screen_topk_pick(g, k_topk)
        cand_of = {c.sha: c for c in run.order}
        out = _stack_output(rt, pipe, g, run.results, cand_of, "single")
        out.update(
            pipeline=name, cid=cid, seed=seed, live_evals=run.evals_used, k=k_topk
        )
        return out
    raise ValueError(name)


STAGE_V_PIPELINES = (
    "H317_LOCKED",
    "EVOA_MATCHED",
    "SCREEN_TOPK_PICK",
    "H257_REPLICA",
    "BANK_NN1",
    "BANK_NN8_PICK",
    "BANK_NN8_PORTFOLIO",
)


def stage_v_context(rt: Runtime, cid: str, seed: int, locked: LockedConfig) -> dict:
    """All Stage V pipelines on one context, with the live-eval matching asserts."""
    out = {"BANK_NN1": stage_v_pipeline(rt, "BANK_NN1", cid, seed, locked)}
    out["H317_LOCKED"] = stage_v_pipeline(rt, "H317_LOCKED", cid, seed, locked)
    out["EVOA_MATCHED"] = stage_v_pipeline(rt, "EVOA_MATCHED", cid, seed, locked)
    k = out["H317_LOCKED"]["live_evals"]
    out["SCREEN_TOPK_PICK"] = stage_v_pipeline(
        rt, "SCREEN_TOPK_PICK", cid, seed, locked, k_topk=k
    )
    for name in ("H257_REPLICA", "BANK_NN8_PICK", "BANK_NN8_PORTFOLIO"):
        out[name] = stage_v_pipeline(rt, name, cid, seed, locked)
    if abs(out["H317_LOCKED"]["live_evals"] - out["EVOA_MATCHED"]["live_evals"]) > 2:
        raise AssertionError(
            "EVOA_MATCHED live evals differ from H317_LOCKED by more than 2"
        )
    if out["SCREEN_TOPK_PICK"]["live_evals"] != k:
        raise AssertionError("SCREEN_TOPK_PICK K mismatch")
    return out


def isolated_runtime(
    screener: S.Screener,
    coder: D.ContextCoder,
    seed: int,
    cores: Sequence[int],
    gp_root: Path,
    meta_rows: Optional[Sequence[dict]] = None,
    timeout_s: float = 60.0,
    guard: bool = True,
    runner=None,
) -> Runtime:
    """A fresh engine + AUDIT ledger + TRAIN-only bank for one Stage V call (no state shared with other calls)."""
    rows = C.train_rows()
    meta = list(meta_rows) if meta_rows is not None else list(rows)
    fam_of = {r["context_sha256"]: r["family_id"] for r in meta}
    ds_of = {r["context_sha256"]: r["dataset_id"] for r in meta}
    bank, _ = D.build_bank(rows)
    gp_root = Path(gp_root)
    rg = ResourceGuard(log_path=gp_root / "resource_pauses.jsonl") if guard else None
    eng = EvalEngine(cores, root=gp_root, timeout_s=timeout_s, guard=rg, runner=runner)
    return Runtime(
        eng,
        AuditLedger(eng),
        lambda cid: screener,
        coder,
        bank,
        fam_of,
        ds_of,
        Store(gp_root / "stagev_store"),
        (int(seed),),
    )


_WIDTH_LOCK = threading.Lock()
_WIDTH_STATE: dict = {"n": 0, "width": None, "prev": None}


class _expanded_width:
    """G.MAX_EXPANDED_WIDTH is process-global: set it for the duration of run_pipeline calls and restore it after the
    last one; concurrent calls with a different value are refused."""

    def __init__(self, width: int):
        self.width = int(width)

    def __enter__(self):
        with _WIDTH_LOCK:
            if _WIDTH_STATE["n"] and _WIDTH_STATE["width"] != self.width:
                raise ValueError(
                    "concurrent run_pipeline calls must share max_expanded_width"
                )
            if _WIDTH_STATE["n"] == 0:
                _WIDTH_STATE["prev"], _WIDTH_STATE["width"] = (
                    G.MAX_EXPANDED_WIDTH,
                    self.width,
                )
                G.MAX_EXPANDED_WIDTH = self.width
            _WIDTH_STATE["n"] += 1
        return self

    def __exit__(self, *exc):
        with _WIDTH_LOCK:
            _WIDTH_STATE["n"] -= 1
            if _WIDTH_STATE["n"] == 0:
                G.MAX_EXPANDED_WIDTH = _WIDTH_STATE["prev"]
                _WIDTH_STATE["width"] = None
        return False


def run_pipeline(
    name: str,
    cid: str,
    seed: int,
    screener: S.Screener,
    coder: D.ContextCoder,
    *,
    locked: LockedConfig,
    cores: Sequence[int],
    gp_root: Path,
    k_topk: Optional[int] = None,
    meta_rows: Optional[Sequence[dict]] = None,
    timeout_s: float = 60.0,
    full_mode: bool = False,
    guard: bool = True,
    runner=None,
    max_expanded_width: int = RUN_MAX_EXPANDED_WIDTH,
) -> dict:
    """STABLE API for stage_v_h317.py: run ONE named pipeline (STAGE_V_PIPELINES) on ONE context.

    `screener`: an `S.Screener` for this context (e.g. `S.NeuralScreenerAdapter` around the neural track's target-bound
    or fold screener); `coder`: a `D.ContextCoder` (`D.NeuralContextCoder(S.SearchViewProvider())` in production);
    `meta_rows`: rows supplying family/dataset ids for the bank exclusion (TRAIN + the DL-Val identifiers in Stage V).
    A fresh engine / ledger / runtime is built for the call and closed afterwards, so nothing carries across contexts
    or pipelines except the per-context, envelope-keyed memo file under `gp_root` (exact replays only).
    Returns the pipeline record: live_evals, value (AUDIT utility, mean-predictor policy), value_m2, invalid, raw,
    config/spec, `search_utility` (in-sample SEARCH utility of the declared output: the tree's 2-fold S1, the stack's
    cross-fitted S1, the CV5 program's own S1; None for BANK_NN1), `audit_half_utilities` [u_half0, u_half1] (the same
    AUDIT fit scored on two deterministic y-stratified halves of the AUDIT rows, `gp_engine_h317.audit_half_masks`),
    `max_expanded_width`, plus `engine` stats and `audit_ledger` {outputs, audit_requests}."""
    with _expanded_width(max_expanded_width):
        rt = isolated_runtime(
            screener, coder, seed, cores, gp_root, meta_rows, timeout_s, guard, runner
        )
        try:
            rec = stage_v_pipeline(
                rt, name, cid, seed, locked, k_topk=k_topk, full_mode=full_mode
            )
            rec["audit_ledger"] = rt.ledger.check()
            rec["engine"] = rt.engine.stats()
            rec["bank_exclusion"] = rt.target_bank(cid).exclusion_record()
            rec["max_expanded_width"] = G.MAX_EXPANDED_WIDTH
            return rec
        finally:
            rt.engine.close()


def run_stage_v_context(
    cid: str,
    seed: int,
    screener: S.Screener,
    coder: D.ContextCoder,
    *,
    locked: LockedConfig,
    cores: Sequence[int],
    gp_root: Path,
    meta_rows: Optional[Sequence[dict]] = None,
    timeout_s: float = 60.0,
    guard: bool = True,
    runner=None,
    max_expanded_width: int = RUN_MAX_EXPANDED_WIDTH,
) -> dict:
    """All STAGE_V_PIPELINES on one context via `run_pipeline` (fresh runtime each), with the matching asserts:
    |live(H317_LOCKED) - live(EVOA_MATCHED)| <= 2 and SCREEN_TOPK_PICK K == live(H317_LOCKED)."""
    kw = dict(
        locked=locked,
        cores=cores,
        gp_root=gp_root,
        meta_rows=meta_rows,
        timeout_s=timeout_s,
        guard=guard,
        runner=runner,
        max_expanded_width=max_expanded_width,
    )
    out = {
        "BANK_NN1": run_pipeline("BANK_NN1", cid, seed, screener, coder, **kw),
        "H317_LOCKED": run_pipeline("H317_LOCKED", cid, seed, screener, coder, **kw),
        "EVOA_MATCHED": run_pipeline("EVOA_MATCHED", cid, seed, screener, coder, **kw),
    }
    k = out["H317_LOCKED"]["live_evals"]
    out["SCREEN_TOPK_PICK"] = run_pipeline(
        "SCREEN_TOPK_PICK", cid, seed, screener, coder, k_topk=k, **kw
    )
    for name in ("H257_REPLICA", "BANK_NN8_PICK", "BANK_NN8_PORTFOLIO"):
        out[name] = run_pipeline(name, cid, seed, screener, coder, **kw)
    if abs(out["H317_LOCKED"]["live_evals"] - out["EVOA_MATCHED"]["live_evals"]) > 2:
        raise AssertionError(
            "EVOA_MATCHED live evals differ from H317_LOCKED by more than 2"
        )
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


def build_runtime(
    run_dir: Path,
    cores: Sequence[int],
    screener: str,
    coder: str,
    gp_root: Path = GP_ROOT,
    timeout_s: float = 60.0,
    guard: bool = True,
    runner=None,
    seeds=C.SEARCH_SEEDS,
    neural_run_dir: Optional[Path] = None,
    stage_v: bool = False,
    torch_threads: int = 1,
) -> Runtime:
    """Bank = TRAIN champions/winners only (never grows). Stage V adds the DL-Val rows' family/dataset ids (only
    after `common_h317.unlock_stage_v`) so the bank exclusion also covers DL-Val targets; nothing carries state
    across contexts (memo, corrector, screener cache and codes are all keyed by context)."""
    if len(cores) > MAX_WORKERS:
        raise ValueError(
            f"at most {MAX_WORKERS} Rust workers per process (use --shard for more)"
        )
    rows = C.train_rows()
    meta_rows = list(rows) + (list(C.dev_rows_for_stage_v()) if stage_v else [])
    fam_of = {r["context_sha256"]: r["family_id"] for r in meta_rows}
    ds_of = {r["context_sha256"]: r["dataset_id"] for r in meta_rows}
    folds = C.family_folds(rows)
    bank, _ = D.build_bank(rows)
    rg = (
        ResourceGuard(log_path=Path(run_dir) / "resource_pauses.jsonl")
        if guard
        else None
    )
    engine = EvalEngine(
        cores, root=gp_root, timeout_s=timeout_s, guard=rg, runner=runner
    )
    views = S.SearchViewProvider()
    if screener == "fallback":
        fb = S.FallbackScreener()
        screener_for = lambda cid: fb  # noqa: E731
    elif screener == "neural":
        import torch

        torch.set_num_threads(max(1, int(torch_threads)))
        fs = S.load_neural_fold_screeners(
            Path(neural_run_dir), folds, views, family_of=fam_of
        )
        screener_for = fs.for_context
        _record_screener_manifest(Path(run_dir), Path(neural_run_dir), fs)
    else:
        raise ValueError(screener)
    cd = D.NumpyContextCoder() if coder == "numpy" else D.NeuralContextCoder(views)
    return Runtime(
        engine,
        AuditLedger(engine),
        screener_for,
        cd,
        bank,
        fam_of,
        ds_of,
        Store(run_dir),
        tuple(seeds),
    )


def screener_manifest(neural_run_dir: Path, fs: "S.FoldScreeners") -> dict:
    """Per fold: model.pt sha256s and the sha256 of the sorted training-cid / family sets (red team LOW-3)."""
    out = {}
    for f, ad in sorted(fs.by_fold.items()):
        paths = sorted(Path(neural_run_dir).glob(f"seed*/fold{f}/model.pt"))
        m = ad.model
        out[str(f)] = {
            "model_pt_sha256": {
                str(p.relative_to(neural_run_dir)): C.sha256_file(p) for p in paths
            },
            "train_cids_sha256": hashlib.sha256(
                "\n".join(sorted(m.train_cids)).encode()
            ).hexdigest(),
            "train_families_sha256": hashlib.sha256(
                "\n".join(sorted(m.train_families)).encode()
            ).hexdigest(),
            "n_train_cids": len(m.train_cids),
        }
    return out


def _record_screener_manifest(run_dir: Path, neural_run_dir: Path, fs) -> None:
    cfg_path = run_dir / "config.json"
    if not cfg_path.exists():
        return
    cfg = json.loads(cfg_path.read_text())
    man = screener_manifest(neural_run_dir, fs)
    if "screener_manifest" in cfg:
        if cfg["screener_manifest"] != man:
            raise SystemExit("screener models changed under an existing run dir")
        return
    cfg["screener_manifest"] = man
    C.write_json(cfg_path, cfg)


def finalize(rt: Runtime, run_dir: Path, summary: dict) -> dict:
    summary["guard"] = C.guard_report()
    summary["bank_counters"] = dict(D.BANK_COUNTERS)
    summary["audit_ledger"] = rt.ledger.check()
    summary["engine"] = rt.engine.stats()
    if summary["bank_counters"]["same_family_bank_hits"] != 0:
        raise AssertionError("same_family_bank_hits != 0")
    C.write_json(Path(run_dir) / "summary.json", summary)
    return summary


def parse_cpus(spec: str) -> List[int]:
    out: List[int] = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def shard_cids(pilot: Sequence[str], shard: str) -> List[str]:
    """`k/n`: contexts with sorted-index i % n == k (deterministic, disjoint, covering)."""
    k, n = (int(x) for x in shard.split("/"))
    if not 0 <= k < n:
        raise ValueError(shard)
    return [c for i, c in enumerate(sorted(pilot)) if i % n == k]


def _pilot_config(run_dir: Path, args) -> Tuple[List[str], Tuple[int, ...]]:
    rows = C.train_rows()
    folds = C.family_folds(rows)
    pilot = sorted(C.pilot_shard(rows, folds, n=args.pilot_n))
    if getattr(
        args, "contexts_file", None
    ):  # infrastructure dry runs only: never a pilot context
        dry = sorted(
            x.strip() for x in Path(args.contexts_file).read_text().split() if x.strip()
        )
        if set(dry) & set(pilot) or args.screener != "fallback":
            raise SystemExit(
                "--contexts-file is for fallback-screener dry runs on NON-pilot contexts only"
            )
        pilot = dry
    seeds = tuple(int(x) for x in args.seeds.split(","))
    if args.max_expanded_width:
        G.MAX_EXPANDED_WIDTH = int(args.max_expanded_width)
    psha = hashlib.sha256("\n".join(pilot).encode()).hexdigest()
    cfg_path = run_dir / "config.json"
    cfg = {
        "pilot": pilot,
        "pilot_n": args.pilot_n,
        "pilot_sha256": psha,
        "seeds": list(seeds),
        "max_expanded_width": G.MAX_EXPANDED_WIDTH,
        "screener": args.screener,
        "coder": args.coder,
        "neural_run_dir": args.neural_run_dir,
        "common_h317_sha256": C.sha256_file(HERE / "common_h317.py"),
    }
    if cfg_path.exists():
        old = json.loads(cfg_path.read_text())
        for k in (
            "pilot_sha256",
            "seeds",
            "max_expanded_width",
            "screener",
            "coder",
            "neural_run_dir",
            "common_h317_sha256",
        ):
            if old.get(k) != cfg.get(k):
                raise SystemExit(
                    f"run config mismatch on {k}: {old.get(k)} != {cfg.get(k)} (new run dir?)"
                )
    else:
        C.write_json(cfg_path, cfg)
    return pilot, seeds


def launch_rules() -> dict:
    return {
        "invalid_rule": INVALID_RULE,
        "failure_policy": "AUDIT crash / non-finite -> SEARCH-target-mean prediction (u = 0 for a single output); "
        "weights never change with AUDIT outcomes; value_m2 keeps the -2.0 convention",
        "max_expanded_width_generated": G.MAX_EXPANDED_WIDTH,
        "heavy_head_width_cap": G.MAX_HEAVY_WIDTH,
        "cost_rule_generated": "no stack head; portfolio <= 3 members without stack/gpr/krr heads; histgbm2 "
        "rounds*bags <= 1200 (<= 600 inside portfolios); gpr m <= 300 on > 100-column tables",
        "exempt_from_width_and_cost_rules": [
            "registry presets (Seed E; e.g. ridge_quad expands to 1,736 columns on "
            "the 365-column table)",
            "bank records (width guard only)",
            "CV5 portfolio",
            "migration splices",
            "BAG_FINAL",
        ],
        "exempt_children_note": "children of cost-exempt programs (e.g. migration splices) are subject to the cost "
        "rule, so an exempt program that violates it cannot be refined (red team LOW-4)",
        "timeouts": "60 s; x4 explicit portfolio; x2 AUDIT envelope; timeouts / signal kills retried once, then "
        "recorded as failure (both attempts logged in <gp_root>/retries.jsonl)",
        "seeds_required": list(C.SEARCH_SEEDS),
        "sharding": "contexts partitioned by sorted index; all seeds of a context in one shard/process",
        "h04_holm": "m = 4 challengers (B, C, D, E vs A); rank r by sign-flip p: LCB at two-sided alpha 0.05/(m-r), "
        "sign-flip one-sided level 0.025/(m-r); step-down",
        "h05": "Evo B iff contract gate on gen-10 AUDIT vs Evo A; generations locked at 10",
        "h06": "leave-one-family-out config choice over CV1, CV2@N, CV3@N, CV4, CV5; locked = best over all families",
    }


def cmd_run(args) -> None:
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    cores = parse_cpus(args.cores)
    if args.screener == "fallback" and not args.allow_fallback_screener:
        raise SystemExit(
            "the fallback screener is for tests only (pass --allow-fallback-screener for a dry run)"
        )
    if args.max_expanded_width:
        G.MAX_EXPANDED_WIDTH = int(
            args.max_expanded_width
        )  # declared run-plan knob (cost), recorded in config.json
    pilot, seeds = _pilot_config(run_dir, args)
    cids = shard_cids(pilot, args.shard) if args.shard else pilot
    rt = build_runtime(
        run_dir,
        cores,
        args.screener,
        args.coder,
        timeout_s=args.timeout,
        neural_run_dir=args.neural_run_dir,
        seeds=seeds,
        torch_threads=args.torch_threads,
    )
    dec = (
        json.loads((run_dir / "decisions.json").read_text())
        if (run_dir / "decisions.json").exists()
        else {}
    )
    scheme = args.h04_scheme or dec.get("H0.4", {}).get("used")
    arm = args.h05_arm or dec.get("H0.5", {}).get("used")
    t0 = time.time()
    done = {}
    for stage in ["bank_nn1"] + args.stages.split(","):
        if stage == "h05" and scheme is None:
            raise SystemExit(
                "h05 needs the merged H0.4 decision (run `merge` first) or --h04-scheme"
            )
        if stage == "h06" and (scheme is None or arm is None):
            raise SystemExit(
                "h06 needs the merged H0.4/H0.5 decisions (run `merge` first) or --h04-scheme/--h05-arm"
            )
        done[stage] = len(run_stage(rt, stage, cids, args.jobs, scheme, arm))
    status = {
        "shard": args.shard,
        "contexts": len(cids),
        "records": done,
        "wall_s": time.time() - t0,
        "engine": rt.engine.stats(),
        "audit_ledger": rt.ledger.check(),
        "guard": C.guard_report(),
        "bank_counters": dict(D.BANK_COUNTERS),
    }
    if status["bank_counters"]["same_family_bank_hits"] != 0:
        raise AssertionError("same_family_bank_hits != 0")
    tag = (args.shard or "all").replace("/", "of")
    C.write_json(
        run_dir
        / "shards"
        / f"status_{tag}_{args.stages.replace(',', '+')}_{int(t0)}.json",
        status,
    )
    print(
        json.dumps({k: status[k] for k in ("shard", "contexts", "records", "wall_s")})
    )


def cmd_merge(args) -> None:
    rows = C.train_rows()
    fam_of = {r["context_sha256"]: r["family_id"] for r in rows}
    ov = {
        k: v
        for k, v in (("h04_scheme", args.h04_scheme), ("h05_arm", args.h05_arm))
        if v
    }
    summ = merge_run(Path(args.run_dir), fam_of, ov)
    summ["guard"] = C.guard_report()
    C.write_json(Path(args.run_dir) / "summary.json", summ)
    print(
        json.dumps(
            {k: summ.get(k) for k in ("contexts", "seeds", "decisions", "incomplete")},
            default=str,
        )
    )


TIMING_DISCLOSURE = (
    "Pre-launch infrastructure runs (xbabe2, all with the deterministic fallback screener or no screener, in separate "
    "roots never staged to xbabe0; no decision used them): context 8f4cde4314f5... (P205 pilot, 6 columns) had "
    "real-worker grammar probes on its SEARCH envelope (gp/probe_grammar, 49 DSLs) and full H0.4->H0.6 timing jobs "
    "incl. AUDIT evals (gp/timing_root, timing_root2 [stopped], timing_root3 [both seeds]); its memo lives only in those "
    "roots. The other timing / probe contexts 4b977e969487, 3abbfff421b9, e69f911372f7 and the parity context "
    "002f7ead6468 are not P205 pilots. Test-suite contexts that later turned out to be P205 pilots were only used with "
    "the fake worker (outputs in gp/test-scratch); the real-worker test context 03fe267990a5 is not a pilot."
)


def prepare_launch(
    run_dir: Path, args, pilot: Sequence[str], seeds: Sequence[int]
) -> dict:
    """LAUNCH_GP.json: the frozen launch record (written once; a re-launch must reproduce it exactly)."""
    rows = C.train_rows()
    folds = C.family_folds(rows)
    split = [sum(1 for c in pilot if folds[c] == k) for k in range(C.N_FOLDS)]
    code = {
        p.name: C.sha256_file(p)
        for p in sorted(HERE.iterdir())
        if p.suffix in (".py", ".sh") and p.is_file()
    }
    man = Path(args.staging_manifest) if args.staging_manifest else None
    if man is None or not man.exists():
        raise SystemExit(
            "--staging-manifest <$H317_DATA/gp/staging/manifest_<tag>.json> is required"
        )
    man_doc = json.loads(man.read_text())
    screener = None
    if args.screener == "neural":
        views = S.SearchViewProvider()
        fs = S.load_neural_fold_screeners(Path(args.neural_run_dir), folds, views)
        screener = {
            "run_dir": args.neural_run_dir,
            "seeds_averaged": "all seeds under run_dir (load_fold_screener "
            "default; per-fold score = mean over the seed models)",
            "folds": screener_manifest(Path(args.neural_run_dir), fs),
        }
    m = len(H04_CHALLENGERS)
    rec = {
        "wave": C.WAVE,
        "pilot": {
            "n": len(pilot),
            "rule": "common_h317.pilot_shard(rows, folds) default PILOT_N",
            "fold_split": split,
            "pilot_sha256": hashlib.sha256(
                "\n".join(sorted(pilot)).encode()
            ).hexdigest(),
        },
        "seeds": list(seeds),
        "seed_rng_note": "Seed A/B/E and BANK_NN1 use no search-seed RNG, so seed 317102 replicates 317101 for those "
        "(memo hits); Seeds C/D (grammar proposals) and both evolution arms are seed-dependent",
        "layout": {
            "shards": args.shards,
            "workers_per_shard": args.workers_per_shard,
            "cpuset": args.cpus,
            "jobs_per_shard": args.jobs,
            "torch_threads": args.torch_threads,
            "sharding": "contexts by sorted index i % shards == k; both seeds of a context in one process",
        },
        "sandbox": {
            "script": "sandbox_h317_docker.sh",
            "sha256": code.get("sandbox_h317_docker.sh"),
            "in_container": os.environ.get("H317_SANDBOX"),
        },
        "staging_manifest": {
            "path": str(man),
            "file_sha256": C.sha256_file(man),
            "manifest_sha256": man_doc.get("manifest_sha256"),
        },
        "code_sha256": code,
        "guard": {
            "resource": "per process, checked at the start of every eval (each worker slot), 5 s poll while "
            "paused, host-wide /proc/stat and /proc/meminfo (other load counts), pause new evals "
            "while host CPU busy > 85 % or MemAvailable < 20 GiB; pauses logged to "
            "<run_dir>/resource_pauses.jsonl",
            "path_guard": C.guard_report(),
        },
        "invalid_rule": INVALID_RULE,
        "h04_holm_levels": [
            {
                "rank": r,
                "lcb_two_sided_alpha": C.holm_adjusted_lcb_alpha(m, r),
                "signflip_one_sided_alpha": 0.025 / (m - r),
            }
            for r in range(m)
        ],
        "screener": screener,
        "timing_disclosure": TIMING_DISCLOSURE,
        **launch_rules(),
    }
    path = run_dir / "LAUNCH_GP.json"
    if path.exists():
        old = json.loads(path.read_text())
        keep = (
            "pilot",
            "seeds",
            "layout",
            "staging_manifest",
            "code_sha256",
            "screener",
            "max_expanded_width_generated",
        )
        diff = [k for k in keep if old.get(k) != rec.get(k)]
        amendable = {"code_sha256", "staging_manifest"}
        amend = getattr(args, "amendment", None)
        if diff and amend and set(diff) <= amendable:
            # a declared code amendment for a resume (e.g. A4 crash fix): LAUNCH_GP.json stays untouched; the
            # amendment is recorded once in LAUNCH_GP.<amendment>.json and must match on every later resume
            apath = run_dir / f"LAUNCH_GP.{amend}.json"
            changed = {
                k: sorted(
                    n
                    for n in set(old[k]) | set(rec[k])
                    if old[k].get(n) != rec[k].get(n)
                )
                for k in ("code_sha256",)
            }
            arec = {
                "amendment": amend,
                "amends": "LAUNCH_GP.json",
                "launch_gp_sha256": C.sha256_file(path),
                "changed_code_files": changed["code_sha256"],
                "code_sha256": rec["code_sha256"],
                "staging_manifest": rec["staging_manifest"],
                "note": "resume after a code amendment; pilot, seeds, layout, screener and width are unchanged",
            }
            if apath.exists():
                prev = json.loads(apath.read_text())
                bad = [
                    k
                    for k in ("code_sha256", "staging_manifest")
                    if prev.get(k) != arec[k]
                ]
                if bad:
                    raise SystemExit(f"{apath.name} differs on {bad}")
                return old
            C.write_json(apath, {**arec, "recorded_utc": time.time()})
            return old
        if diff:
            raise SystemExit(
                f"LAUNCH_GP.json differs on {diff}: new run dir, restore the launched state, or pass "
                "--amendment <id> (only code_sha256 / staging_manifest may change)"
            )
        return old
    C.write_json(path, rec)
    return rec


def cmd_prepare_launch(args) -> None:
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    if args.max_expanded_width:
        G.MAX_EXPANDED_WIDTH = int(args.max_expanded_width)
    pilot, seeds = _pilot_config(run_dir, args)
    rec = prepare_launch(run_dir, args, pilot, seeds)
    print(
        json.dumps(
            {k: rec[k] for k in ("pilot", "seeds", "layout", "staging_manifest")},
            default=str,
        )
    )


def cmd_launch(args) -> None:
    """Spawn `--shards` shard processes per stage over a CPU list (each pinned to its own worker cores), wait, merge,
    continue with the next stage. Resumable: re-launching skips finished records."""
    import subprocess

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    cpus = parse_cpus(args.cpus)
    w = args.workers_per_shard
    if args.shards * w > len(cpus):
        raise SystemExit(f"{args.shards} x {w} workers > {len(cpus)} CPUs in --cpus")
    if args.max_expanded_width:
        G.MAX_EXPANDED_WIDTH = int(args.max_expanded_width)
    pilot, seeds = _pilot_config(run_dir, args)
    if not args.contexts_file:
        prepare_launch(run_dir, args, pilot, seeds)
    common = [
        "--run-dir",
        str(run_dir),
        "--pilot-n",
        str(args.pilot_n),
        "--seeds",
        args.seeds,
        "--screener",
        args.screener,
        "--coder",
        args.coder,
        "--jobs",
        str(args.jobs),
        "--timeout",
        str(args.timeout),
        "--torch-threads",
        str(args.torch_threads),
    ]
    if args.neural_run_dir:
        common += ["--neural-run-dir", args.neural_run_dir]
    if args.allow_fallback_screener:
        common += ["--allow-fallback-screener"]
    if args.max_expanded_width:
        common += ["--max-expanded-width", str(args.max_expanded_width)]
    if args.contexts_file:
        common += ["--contexts-file", args.contexts_file]
    logs = run_dir / "shards"
    logs.mkdir(parents=True, exist_ok=True)
    for stage in args.stages.split(","):
        procs = []
        for k in range(args.shards):
            mine = ",".join(str(c) for c in cpus[k * w : (k + 1) * w])
            cmd = [
                "taskset",
                "-c",
                mine,
                "nice",
                "-n",
                "19",
                sys.executable,
                "-u",
                str(HERE / "gp_lab.py"),
                "run",
                "--shard",
                f"{k}/{args.shards}",
                "--cores",
                mine,
                "--stages",
                stage,
            ] + common
            log = open(logs / f"shard{k}of{args.shards}_{stage}.log", "a")
            procs.append(
                (k, subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT), log)
            )
        bad = []
        for k, p, log in procs:
            if p.wait() != 0:
                bad.append(k)
            log.close()
        if bad:
            raise SystemExit(f"stage {stage}: shards {bad} failed (see {logs})")
        rows = C.train_rows()
        summ = merge_run(run_dir, {r["context_sha256"]: r["family_id"] for r in rows})
        summ["guard"] = C.guard_report()
        C.write_json(run_dir / "summary.json", summ)
        name = {"h04": "H0.4", "h05": "H0.5", "h06": "H0.6"}[stage]
        if (
            name not in summ
        ):  # the stage just run must be complete (later stages legitimately are not)
            raise SystemExit(
                f"merge after {stage} incomplete: {summ.get('incomplete')}"
            )
        print(
            json.dumps(
                {"stage": stage, "decisions": summ.get("decisions")}, default=str
            ),
            flush=True,
        )


def cmd_timing(args) -> None:
    """Timing on NON-pilot TRAIN contexts only: per-operator child evals + one full H0.4->H0.6 job per context."""
    rows = C.train_rows()
    folds = C.family_folds(rows)
    cids = args.contexts.split(",")
    protected = set()
    for n in (24, 48, 96, 205, C.PILOT_N):  # every pilot shard incl. the launched P205
        protected |= set(C.pilot_shard(rows, folds, n=n))
    for c in cids:
        if c in protected:
            raise SystemExit(f"{c} is a pilot context: timing refused")
    run_dir = Path(args.run_dir)
    cores = parse_cpus(args.cores)
    rt = build_runtime(
        run_dir,
        cores,
        "fallback",
        "numpy",
        gp_root=Path(args.gp_root),
        timeout_s=args.timeout,
    )
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
                rec[f"full_seed{seed}"] = {
                    "wall_s": time.time() - tj,
                    "physical_runs": s1["physical_runs"] - s0["physical_runs"],
                    "eval_cpu_s": s1["wall_s_total"] - s0["wall_s_total"],
                    "logical_requests": (s1["requests_search"] + s1["requests_audit"])
                    - (s0["requests_search"] + s0["requests_audit"]),
                    "failures": s1["failures"] - s0["failures"],
                    "timeouts": s1["timeouts"] - s0["timeouts"],
                }
        rec["wall_s"] = time.time() - t0
        out["contexts"][cid] = rec
        C.write_json(run_dir / "timing.json", out)
    out["engine"] = rt.engine.stats()
    out["guard"] = C.guard_report()
    out["bank_counters"] = dict(D.BANK_COUNTERS)
    out["wall_s"] = time.time() - t_all
    out["walls_by_operator"] = _walls_by_operator(rt.engine.walls)
    C.write_json(run_dir / "timing.json", out)
    print(
        json.dumps(
            {k: v for k, v in out.items() if k != "contexts"},
            indent=1,
            default=C._json_default,
        )
    )


def _walls_by_operator(walls) -> dict:
    acc: Dict[str, List[Tuple[float, str]]] = {}
    for kind, op, w, err in walls:
        key = op.split("_i")[0]
        for pre in (
            "g1_",
            "g2_",
            "g3_",
            "g4_",
            "g5_",
            "g6_",
            "g7_",
            "g8_",
            "g9_",
            "g10_",
            "gen0_",
            "topk_",
        ):
            if key.startswith(pre):
                key = key[len(pre) :]
        if key.startswith("registry:"):
            key = "registry_preset"
        acc.setdefault(f"{kind}:{key}", []).append((w, err))
    return {
        k: {
            "n": len(v),
            "mean_s": float(np.mean([w for w, _ in v])),
            "p95_s": float(np.quantile([w for w, _ in v], 0.95)),
            "max_s": float(np.max([w for w, _ in v])),
            "fail": sum(1 for _, e in v if e),
        }
        for k, v in sorted(acc.items())
    }


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
                c = G.safe_candidate(
                    fn(i),
                    f"timing_{name}",
                    [nn[i].sha],
                    g.ctx.width,
                    bank_cids=D.merge_lineage(nn[i]),
                )
                if (
                    c is not None
                    and c.sha not in {x.sha for x in out}
                    and c.sha not in g.tb.excluded_shas
                ):
                    out.append(c)
            ops[name] = out

        make("bank_nn", lambda i: pg[i])
        make("edit_parameter", lambda i: G.op_parameter(pg[i], rng))
        make("edit_insert_stage", lambda i: G.op_insert_stage(pg[i], rng))
        make("edit_delete_stage", lambda i: G.op_delete_stage(pg[i], rng))
        make(
            "edit_crossover",
            lambda i: G.op_crossover(pg[i], pg[(i + 1) % len(pg)], rng),
        )
        for isl in range(4):
            make(
                f"replace_head_i{isl}",
                lambda i, isl=isl: G.op_replace_head(pg[i], rng, isl),
            )
            make(f"synth_i{isl}", lambda i, isl=isl: G.op_synth(isl, rng))
        make(
            "migration_portfolio",
            lambda i: G.op_portfolio([pg[i], G.op_synth(rng.choice([1, 2]), rng)]),
        )
        make(
            "line_search",
            lambda i: rng.choice(G.line_search_neighbors(pg[i]) or [pg[i]]),
        )
        make("bag_final", lambda i: G.bag_final(pg[i]))
        ops["registry_preset"] = D.registry_candidates(g.ctx.width)
    out = {}
    for name, cands in ops.items():
        s0 = len(rt.engine.walls)
        with D.selection_scope("operator_timing"):
            res = g.evaluate(cands)
        walls = [w for _, _, w, _ in rt.engine.walls[s0:]]
        out[name] = {
            "n": len(res),
            "failed": sum(r.status != "completed" for r in res),
            "gate_pass": sum(bool(r.gate) for r in res),
            "errors": sorted({r.error for r in res if r.error}),
            "mean_wall_s": float(np.mean(walls)) if walls else None,
            "max_wall_s": float(np.max(walls)) if walls else None,
        }
    return out


def _common_args(r) -> None:
    r.add_argument("--run-dir", required=True)
    r.add_argument(
        "--jobs",
        type=int,
        default=16,
        help="concurrent (context, seed) jobs per process",
    )
    r.add_argument("--pilot-n", type=int, default=C.PILOT_N)
    r.add_argument("--seeds", default=",".join(str(x) for x in C.SEARCH_SEEDS))
    r.add_argument("--screener", choices=("neural", "fallback"), default="neural")
    r.add_argument("--neural-run-dir", default=None)
    r.add_argument("--coder", choices=("neural", "numpy"), default="neural")
    r.add_argument("--allow-fallback-screener", action="store_true")
    r.add_argument("--timeout", type=float, default=60.0)
    r.add_argument("--torch-threads", type=int, default=1)
    r.add_argument("--max-expanded-width", type=int, default=RUN_MAX_EXPANDED_WIDTH)
    r.add_argument("--contexts-file", default=None, help=argparse.SUPPRESS)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser(
        "run", help="one process (optionally one shard) for the given stages"
    )
    _common_args(r)
    r.add_argument("--cores", required=True, help="worker cores, e.g. 16-23 or 1,3,5")
    r.add_argument(
        "--shard", default=None, help="k/n: contexts with sorted index i %% n == k"
    )
    r.add_argument("--stages", default="h04")
    r.add_argument("--h04-scheme", default=None)
    r.add_argument("--h05-arm", default=None)
    r.set_defaults(fn=cmd_run)
    m = sub.add_parser(
        "merge",
        help="rebuild summary/decisions from all shard records (refuses missing/duplicates)",
    )
    m.add_argument("--run-dir", required=True)
    m.add_argument("--h04-scheme", default=None)
    m.add_argument("--h05-arm", default=None)
    m.set_defaults(fn=cmd_merge)
    la = sub.add_parser(
        "launch", help="all shards per stage, then merge, stage by stage"
    )
    _common_args(la)
    la.add_argument(
        "--cpus", required=True, help="CPU list for all shards, e.g. 16-63,80-127"
    )
    la.add_argument("--shards", type=int, required=True)
    la.add_argument("--workers-per-shard", type=int, required=True)
    la.add_argument("--stages", default="h04,h05,h06")
    la.add_argument("--staging-manifest", default=None)
    la.add_argument(
        "--amendment",
        default=None,
        help="declared code amendment id for a resume, e.g. A4",
    )
    la.set_defaults(fn=cmd_launch)
    pl = sub.add_parser(
        "prepare-launch", help="write LAUNCH_GP.json (also done by `launch`)"
    )
    _common_args(pl)
    pl.add_argument("--cpus", required=True)
    pl.add_argument("--shards", type=int, required=True)
    pl.add_argument("--workers-per-shard", type=int, required=True)
    pl.add_argument("--staging-manifest", required=True)
    pl.add_argument("--amendment", default=None)
    pl.set_defaults(fn=cmd_prepare_launch)
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
