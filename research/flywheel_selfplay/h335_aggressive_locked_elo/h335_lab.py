#!/usr/bin/env python3
"""H335 lab: the units, shard runner, launcher and Phase T decision (PREREG §2-§4).

A unit is one (context, search seed, group). Two process groups never share a process (the LEARNERS-V1 grammar port is
installed at import, h335_core):
  AGG   H335_AGG     Seed F (6 new-head presets + 8 registry presets + 8 cross-family bank seeds, 22 evals) ->
                     panmictic Evo A from F's 22 results, K = 6 live / 48 screened per generation -> outputs at each
                     checkpoint: single best-by-SEARCH tree and the CV3@12 S1-direct stack (SEARCH-only selection)
  BASE  OLD_MATCHED  Seed A (8) -> the same Evo A (base grammar); outputs at gen gc + 2 (realized evals 6 gc + 20 vs
                     AGG's 6 gc + 22) as single (= H317's adopted recipe) and CV3@12 (decomposition)
        H317_BASE    OLD_MATCHED's gen-10 single (68 evals) = H317's adopted GP pipeline
        H257_REPLICA Seed A -> 3 gens x 20 proposals / 5 live -> CV1 Top-3 (the H257 GP recipe, gp_lab port)
        BANK_NN8_PICK / BANK_NN8_PORTFOLIO / BANK_NN1   cross-family bank references
Every GP arm runs on the superset worker 2a848c8f... (CPU-time-limited runner). Every output gets exactly one AUDIT
request (gp_engine AuditLedger); selection never sees AUDIT rows.

Commands (inside sandbox_h335_docker.sh on xbabe0 or sandbox_h335.sh on xbabe2):
  launch  --phase T --run-dir R --cpus LIST --shards-agg N --shards-base M --workers-per-shard W [--host-part k/n]
  run     --phase T --group AGG|BASE --run-dir R --shard k/n --cores LIST       (spawned by launch)
  decide  --run-dirs R1,R2 --out F       Phase T decision: G*, stack, timing projection (PREREG §3.3)
Phase V units are run by stage_v_h335.py through `run_units` with the LOCKed settings.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import h335_core as K  # noqa: E402  (FIRST: pins, grammar group, guard)

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
from typing import Dict, List, Optional, Sequence, Tuple  # noqa: E402

import numpy as np  # noqa: E402

import h335_stats as H  # noqa: E402

C, D, E, L, ST, S, G = K.C, K.D, K.E, K.L, K.ST, K.S, K.G

STACKS = ("single", K.STACK)
SCHEMA = "jope.h335.unit/v1"
INFRA_ERRORS = (OSError, MemoryError)
# integrity violations stop the stage (never recorded as outcome-independent failures): guard, AUDIT ledger, bank leak
# (D.BankLeakError is an AssertionError), any other assertion, the screener's target/family leak check
INTEGRITY_ERRORS = (C.GuardViolation, E.AuditLeakError, AssertionError, getattr(__import__("rapid_hypothesis_lab"),
                                                                                 "ScreenerLeak", AssertionError))


# ----------------------------------------------------------------------------------------------------------------------
# Units
# ----------------------------------------------------------------------------------------------------------------------
def _upto(evo, gc: int):
    return [r for r in evo.results if evo.gen_of[r.sha] <= gc]


def _live_upto(evo, gc: int) -> int:
    return int(sum(v for k, v in evo.live_per_gen.items() if k <= gc))


def _members_new_head(o: dict, cand_of) -> Optional[bool]:
    spec = o.get("spec")
    if not spec or K.GROUP != "AGG":
        return None
    w = spec.get("w_audit") or []
    return any(K.new_head(cand_of[s].dsl) for s, x in zip(spec["members"], w) if x > ST.W_EPS and s in cand_of)


def _out(rt, pipe: str, g, res, cand_of, cfg: str, live: int) -> dict:
    o = L._stack_output(rt, pipe, g, res, cand_of, cfg)
    o["live_evals"] = int(live + o.get("extra_live", 0))
    o["output_has_new_head"] = _members_new_head(o, cand_of)
    return o


def _evo_summary(evo) -> dict:
    s = evo.summary()
    s.pop("island_hist", None)
    return s


def _cps(checkpoints: Sequence[int]) -> Tuple[int, ...]:
    """Distinct sorted checkpoints (Phase V uses {10, G*}; G* = 10 must not request AUDIT twice)."""
    return tuple(sorted({int(x) for x in checkpoints}))


def _failed_output(err: str) -> dict:
    return {"value": 0.0, "value_m2": C.GATE_FAIL_UTILITY, "invalid": True, "failed": True, "error": err[-2000:]}


def _guarded(phase: str, rec: dict, keys: Sequence[str], fn) -> None:
    """Run one pipeline segment that produces `keys`. Phase V (PREREG §4.5 as amended): a deterministic exception marks
    only THIS segment's outputs failed (u = 0, invalid) -- other pipelines of the unit are unaffected. Integrity and
    infrastructure errors propagate. Phase T: every exception propagates."""
    if phase != "V":
        rec["outputs"].update(fn())
        return
    try:
        rec["outputs"].update(fn())
    except INTEGRITY_ERRORS:
        raise
    except INFRA_ERRORS:
        raise
    except Exception as ex:  # noqa: BLE001
        err = f"{type(ex).__name__}: {ex}"
        rec.setdefault("segment_errors", {})[",".join(keys)] = err[-2000:]
        for k in keys:
            rec["outputs"][k] = _failed_output(err)


def unit_agg(rt, phase: str, cid: str, seed: int, gens: int, checkpoints: Sequence[int],
             stacks: Sequence[str] = STACKS) -> dict:
    assert K.GROUP == "AGG"
    checkpoints = _cps(checkpoints)
    stage = f"{phase}_AGG"
    hit = rt.store.result(stage, cid, seed)
    if hit:
        return hit
    g = rt.gp(cid, seed)
    t0 = time.time()
    a = D.seed_A(g)
    e = D.seed_E(g, a)
    f = K.seed_F(g, e)
    evo = D.evo_panmictic(g, f.order, f.results, gens=gens, n_prop=K.N_PROP, n_live=K.N_LIVE, arm="evoA")
    cand_of = {**{c.sha: c for c in f.order}, **evo.cand_of}
    rec = {"type": "result", "schema": SCHEMA, "cid": cid, "seed": seed, "phase": phase, "group": "AGG",
           "grammar_sha256": K.GRAMMAR_SHA256, "worker_sha256": K.SUPERSET_SHA256, "gens": gens,
           "seed_F": f.search_metrics(), "evo": _evo_summary(evo), "outputs": {}}
    for gc in checkpoints:
        res = _upto(evo, gc)
        live = f.evals_used + _live_upto(evo, gc)
        for cfg in stacks:
            k = f"AGG.g{gc}.{cfg}"
            _guarded(phase, rec, [k], lambda: {k: _out(rt, f"{phase}.{k}", g, res, cand_of, cfg, live)})
    rec["n_new_head_evaluated"] = sum(1 for r in evo.results if K.new_head(r.dsl))
    rec["live_evals_total"] = f.evals_used + evo.live_evals
    rec["wall_s"] = time.time() - t0
    rt.store.append(stage, cid, seed, rec)
    return rec


def unit_base(rt, phase: str, cid: str, seed: int, gens: int, checkpoints: Sequence[int],
              stacks: Sequence[str] = STACKS) -> dict:
    """OLD_MATCHED to AGG gen + OLD_OFFSET (gens = G + OLD_OFFSET), H317_BASE, H257_REPLICA, BANK_* references."""
    assert K.GROUP == "BASE"
    checkpoints = _cps(checkpoints)
    stage = f"{phase}_BASE"
    hit = rt.store.result(stage, cid, seed)
    if hit:
        return hit
    g = rt.gp(cid, seed)
    t0 = time.time()
    a = D.seed_A(g)
    rec = {"type": "result", "schema": SCHEMA, "cid": cid, "seed": seed, "phase": phase, "group": "BASE",
           "grammar_sha256": K.GRAMMAR_SHA256, "worker_sha256": K.SUPERSET_SHA256, "gens": gens,
           "seed_A": a.search_metrics(), "outputs": {}}
    if max(checkpoints) + K.OLD_OFFSET > gens or K.H317_BASE_GEN > gens:
        raise ValueError("OLD_MATCHED needs gens >= max(checkpoint) + OLD_OFFSET and >= H317_BASE_GEN")
    old_keys = [f"OLD.g{gc}.{cfg}" for gc in checkpoints for cfg in stacks] + ["H317_BASE.single"]

    def old_segment() -> dict:
        evo = D.evo_panmictic(g, a.order, a.results, gens=gens, n_prop=K.N_PROP, n_live=K.N_LIVE, arm="evoA")
        cand_of = {**{c.sha: c for c in a.order}, **evo.cand_of}
        rec["evo"] = _evo_summary(evo)
        sub = {"outputs": {}}
        for gc in checkpoints:
            go = gc + K.OLD_OFFSET
            res = _upto(evo, go)
            live = a.evals_used + _live_upto(evo, go)
            for cfg in stacks:
                k = f"OLD.g{gc}.{cfg}"
                _guarded(phase, sub, [k], lambda: {k: _out(rt, f"{phase}.{k}", g, res, cand_of, cfg, live)})
        res10 = _upto(evo, K.H317_BASE_GEN)
        live10 = a.evals_used + _live_upto(evo, K.H317_BASE_GEN)
        _guarded(phase, sub, ["H317_BASE.single"], lambda: {"H317_BASE.single": _out(
            rt, f"{phase}.H317_BASE.single", g, res10, cand_of, "single", live10)})
        if "segment_errors" in sub:
            rec.setdefault("segment_errors", {}).update(sub["segment_errors"])
        return sub["outputs"]

    def h257_segment() -> dict:
        rep = D.h257_replica_evo(g, a.order, a.results)
        rcand = {**{c.sha: c for c in a.order}, **rep.cand_of}
        return {"H257_REPLICA.CV1": _out(rt, f"{phase}.H257_REPLICA.CV1", g, rep.results, rcand, "CV1",
                                         a.evals_used + rep.live_evals)}

    def nn8_segment() -> dict:
        nn8 = D.bank_nn8_pick(g)
        ncand = {c.sha: c for c in nn8.order}
        sub = {"outputs": {}}
        _guarded(phase, sub, ["BANK_NN8_PICK.single"], lambda: {"BANK_NN8_PICK.single": _out(
            rt, f"{phase}.BANK_NN8_PICK", g, nn8.results, ncand, "single", nn8.evals_used)})
        _guarded(phase, sub, ["BANK_NN8_PORTFOLIO.CV5"], lambda: {"BANK_NN8_PORTFOLIO.CV5": _out(
            rt, f"{phase}.BANK_NN8_PORTFOLIO", g, nn8.results, ncand, "CV5", nn8.evals_used)})
        if "segment_errors" in sub:
            rec.setdefault("segment_errors", {}).update(sub["segment_errors"])
        return sub["outputs"]

    def nn1_segment() -> dict:
        with D.selection_scope(f"{phase}.BANK_NN1"):
            nn1 = D.bank_nn1(g.tb, g.coder)
        raw = L.audit_one(rt, f"{phase}.BANK_NN1", g, nn1, halves=True)
        v = L.value_of(raw)
        return {"BANK_NN1.single": {"raw": raw, "value": v[0], "value_m2": v[1], "invalid": v[2], "live_evals": 0,
                                    "audit_half_utilities": raw["audit_half_utilities"]}}

    _guarded(phase, rec, old_keys, old_segment)
    _guarded(phase, rec, ["H257_REPLICA.CV1"], h257_segment)
    _guarded(phase, rec, ["BANK_NN8_PICK.single", "BANK_NN8_PORTFOLIO.CV5"], nn8_segment)
    _guarded(phase, rec, ["BANK_NN1.single"], nn1_segment)
    rec["wall_s"] = time.time() - t0
    rt.store.append(stage, cid, seed, rec)
    return rec


def failed_unit(rt, phase: str, group: str, cid: str, seed: int, keys: Sequence[str], err: str) -> dict:
    """Phase V only (PREREG §4.4): a deterministic pipeline exception is recorded as an outcome-independent failure
    (every output of the unit = the SEARCH-mean predictor, u = 0, invalid)."""
    rec = {"type": "result", "schema": SCHEMA, "cid": cid, "seed": seed, "phase": phase, "group": group,
           "failed": True, "error": err[-2000:],
           "outputs": {k: {"value": 0.0, "value_m2": C.GATE_FAIL_UTILITY, "invalid": True, "failed": True}
                       for k in keys}}
    rt.store.append(f"{phase}_{group}", cid, seed, rec)
    return rec


def unit_keys(group: str, checkpoints: Sequence[int], stacks: Sequence[str] = STACKS) -> List[str]:
    checkpoints = _cps(checkpoints)
    if group == "AGG":
        return [f"AGG.g{gc}.{cfg}" for gc in checkpoints for cfg in stacks]
    return ([f"OLD.g{gc}.{cfg}" for gc in checkpoints for cfg in stacks]
            + ["H317_BASE.single", "H257_REPLICA.CV1", "BANK_NN8_PICK.single", "BANK_NN8_PORTFOLIO.CV5",
               "BANK_NN1.single"])


# ----------------------------------------------------------------------------------------------------------------------
# Runtime
# ----------------------------------------------------------------------------------------------------------------------
def build_rt(run_dir: Path, phase: str, cores: Sequence[int], worker: Path, timeout_s: float, seeds: Sequence[int],
             neural_run_dir: Path, torch_threads: int = 1, stagev_views: Optional[Path] = None):
    """Phase T = gp_lab.build_runtime (TRAIN fold-j OOF screener, mean of its 2 training seeds) with the CPU-limited
    superset runner. Phase V = the same construction with the DL-Val SEARCH-view dir from the Stage V prep and the
    fold-0 OOF screener pair for every target (same 2-model form; no model saw a DL-Val family)."""
    G.MAX_EXPANDED_WIDTH = L.RUN_MAX_EXPANDED_WIDTH  # H317 run default (1000), every phase
    runner = K.rust_runner_cpu(worker, timeout_s)
    gp_root = Path(run_dir) / f"gproot_{K.GROUP}"
    if phase == "T":
        return L.build_runtime(Path(run_dir), cores, "neural", "neural", gp_root=gp_root, timeout_s=timeout_s,
                               runner=runner, seeds=tuple(seeds), neural_run_dir=Path(neural_run_dir),
                               torch_threads=torch_threads)
    if phase != "V" or stagev_views is None:
        raise ValueError("phase V needs stagev_views")
    if not C._STATE["stage_v"]:
        raise C.GuardViolation("phase V runtime before the Stage V unlock")
    import torch

    import crop_shuffle_synthetic_views as V

    torch.set_num_threads(max(1, int(torch_threads)))
    rows = C.train_rows()
    meta_rows = list(rows) + list(C.dev_rows_for_stage_v())
    fam_of = {r["context_sha256"]: r["family_id"] for r in meta_rows}
    ds_of = {r["context_sha256"]: r["dataset_id"] for r in meta_rows}
    folds = C.family_folds(rows)
    bank, _ = D.build_bank(rows)
    rg = E.ResourceGuard(log_path=Path(run_dir) / "resource_pauses.jsonl")
    engine = E.EvalEngine(cores, root=gp_root, timeout_s=timeout_s, guard=rg, runner=runner)
    vdir = Path(stagev_views)
    views = S.SearchViewProvider(view_fn=lambda cid, kind="search": V.context_view_feats(cid, kind, view_dir=vdir))
    fs = S.load_neural_fold_screeners(Path(neural_run_dir), folds, views, family_of=fam_of)
    fold0 = fs.by_fold[0]
    L._record_screener_manifest(Path(run_dir), Path(neural_run_dir), fs)
    return L.Runtime(engine, E.AuditLedger(engine), lambda cid: fold0, D.NeuralContextCoder(views), bank, fam_of,
                     ds_of, L.Store(Path(run_dir)), tuple(seeds))


def run_units(rt, phase: str, cids: Sequence[str], seeds: Sequence[int], gens: int, checkpoints: Sequence[int],
              jobs: int, stacks: Sequence[str] = STACKS) -> List[dict]:
    order = sorted(cids, key=lambda c: (-K.table_size(c), c))  # largest units first
    work = [(c, s) for c in order for s in seeds]
    group = K.GROUP

    def one(c, s):
        try:
            if group == "AGG":
                return unit_agg(rt, phase, c, s, gens, checkpoints, stacks)
            return unit_base(rt, phase, c, s, gens + K.OLD_OFFSET, checkpoints, stacks)
        except INTEGRITY_ERRORS:
            raise
        except INFRA_ERRORS:
            raise
        except Exception as ex:  # noqa: BLE001
            if phase != "V":
                raise
            import traceback

            return failed_unit(rt, phase, group, c, s, unit_keys(group, checkpoints, stacks),
                               f"{type(ex).__name__}: {ex}\n{traceback.format_exc()}")

    return L.run_parallel(one, work, jobs)


# ----------------------------------------------------------------------------------------------------------------------
# Phase T commands
# ----------------------------------------------------------------------------------------------------------------------
def phase_t_cids(host_part: Optional[str]) -> List[str]:
    cids = K.panel_p201b(C.train_rows())
    if not host_part:
        return cids
    ks, n = host_part.split("/")
    keep = {int(x) for x in ks.split(",")}
    return [c for i, c in enumerate(cids) if i % int(n) in keep]


PHASE_T_FILES = ("h335_core.py", "h335_lab.py", "h335_stats.py", "gp_grammar_h317_learners_v1.py")
CONFIG_IDENTITY = ("seeds", "gens", "checkpoints", "n_prop", "n_live", "old_offset", "stacks", "worker_sha256",
                   "timeout_s", "neural_run_dir", "code_sha256", "max_expanded_width")


def _config(run_dir: Path, args, cids: List[str]) -> dict:
    cfg = {"schema": "jope.h335.run-config/v1", "wave": K.WAVE, "phase": args.phase, "cids": cids,
           "cids_sha256": hashlib.sha256("\n".join(cids).encode()).hexdigest(),
           "host_part": args.host_part, "seeds": list(K.SEEDS_T), "gens": K.G_MAX, "checkpoints": list(K.CHECKPOINTS),
           "n_prop": K.N_PROP, "n_live": K.N_LIVE, "old_offset": K.OLD_OFFSET, "stacks": list(STACKS),
           "worker_sha256": C.sha256_file(Path(args.worker)), "timeout_s": args.timeout,
           "neural_run_dir": str(args.neural_run_dir), "code_sha256": K.code_manifest(PHASE_T_FILES),
           "max_expanded_width": L.RUN_MAX_EXPANDED_WIDTH, "host": os.uname().nodename}
    p = run_dir / "config.json"
    if p.exists():
        old = json.loads(p.read_text())
        for k in CONFIG_IDENTITY + ("cids", "host_part"):
            if old.get(k) != cfg[k]:
                raise SystemExit(f"run config mismatch on {k}")
        return old
    C.write_json(p, cfg)
    return cfg


def cmd_run(args) -> None:
    if args.group != K.GROUP:
        raise SystemExit(f"--group {args.group} but H335_GROUP={K.GROUP}")
    run_dir = Path(args.run_dir)
    cfg = json.loads((run_dir / "config.json").read_text())
    cids = L.shard_cids(cfg["cids"], args.shard)
    if C.sha256_file(Path(args.worker)) != cfg["worker_sha256"]:
        raise SystemExit("worker differs from the run config")
    if K.code_manifest(PHASE_T_FILES) != cfg["code_sha256"]:
        raise SystemExit("code differs from the run config")
    rt = build_rt(run_dir, "T", L.parse_cpus(args.cores), Path(args.worker), float(cfg["timeout_s"]), cfg["seeds"],
                  Path(cfg["neural_run_dir"]), args.torch_threads)
    t0 = time.time()
    recs = run_units(rt, "T", cids, cfg["seeds"], cfg["gens"], cfg["checkpoints"], args.jobs)
    status = {"group": K.GROUP, "shard": args.shard, "records": len(recs), "wall_s": time.time() - t0,
              "engine": rt.engine.stats(), "audit_ledger": rt.ledger.check(), "guard": C.guard_report(),
              "bank_counters": dict(D.BANK_COUNTERS), "host": os.uname().nodename}
    if status["bank_counters"]["same_family_bank_hits"]:
        raise AssertionError("same_family_bank_hits != 0")
    C.write_json(run_dir / "shards" / f"status_T_{K.GROUP}_{args.shard.replace('/', 'of')}_{int(t0)}.json", status)
    rt.engine.close()


def cmd_launch(args) -> None:
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "shards").mkdir(exist_ok=True)
    cids = phase_t_cids(args.host_part)
    _config(run_dir, args, cids)
    # write the screener manifest ONCE before any shard starts (H317 exploratory config.json race)
    rt = build_rt(run_dir, "T", [0], Path(args.worker), args.timeout, K.SEEDS_T, Path(args.neural_run_dir), 1)
    rt.engine.close()
    cpus, w = L.parse_cpus(args.cpus), args.workers_per_shard
    plan = [("AGG", k, args.shards_agg) for k in range(args.shards_agg)] + \
           [("BASE", k, args.shards_base) for k in range(args.shards_base)]
    if len(plan) * w > len(cpus):
        raise SystemExit("not enough CPUs for shards x workers")
    procs = []
    for i, (grp, k, n) in enumerate(plan):
        mine = ",".join(str(c) for c in cpus[i * w:(i + 1) * w])
        env = dict(os.environ, H335_GROUP=grp)
        env.pop("H317_LEARNERS_V1", None)
        cmd = ["taskset", "-c", mine, "nice", "-n", "19", sys.executable, "-u", str(Path(__file__).resolve()), "run",
               "--phase", "T", "--group", grp, "--run-dir", str(run_dir), "--shard", f"{k}/{n}", "--cores", mine,
               "--jobs", str(args.jobs), "--timeout", str(args.timeout), "--worker", args.worker,
               "--torch-threads", str(args.torch_threads)]
        log = open(run_dir / "shards" / f"T_{grp}_shard{k}of{n}.log", "a")
        procs.append((grp, k, subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env), log))
    bad = [(grp, k) for grp, k, p, _ in procs if p.wait() != 0]
    for *_, log in procs:
        log.close()
    if bad:
        raise SystemExit(f"shards failed: {bad}")
    print(json.dumps({"phase": "T", "launched_shards": len(plan), "contexts": len(cids)}))


# ----------------------------------------------------------------------------------------------------------------------
# Collection + Phase T decision
# ----------------------------------------------------------------------------------------------------------------------
def collect(run_dirs: Sequence[Path], phase: str, cids: Sequence[str], seeds: Sequence[int]) -> Dict[str, dict]:
    """{"AGG"|"BASE": {(cid, seed): record}} across run dirs; refuses missing or duplicated units."""
    out: Dict[str, dict] = {"AGG": {}, "BASE": {}}
    for grp in out:
        for c in cids:
            for s in seeds:
                hits = []
                for rd in run_dirs:
                    st = L.Store(Path(rd))
                    n = st.n_results(f"{phase}_{grp}", c, s)
                    if n > 1:
                        raise L.DuplicateRecord(f"{grp} {c[:12]} {s}: {n} records in {rd}")
                    if n == 1:
                        hits.append(st.result(f"{phase}_{grp}", c, s))
                if len(hits) != 1:
                    raise L.IncompleteStage(f"{grp} {c[:12]} {s}: {len(hits)} records")
                out[grp][(c, s)] = hits[0]
    return out


def values_by_output(recs: Dict[str, dict]) -> Dict[str, Dict[str, float]]:
    """{output key: {cid: seed-mean value}}."""
    per: Dict[str, Dict[Tuple[str, int], float]] = {}
    for grp in recs.values():
        for (c, s), r in grp.items():
            for k, o in r["outputs"].items():
                per.setdefault(k, {})[(c, s)] = float(o["value"])
    return {k: L.seed_mean(v) for k, v in per.items()}


def projection(run_dirs: Sequence[Path], recs: Dict[str, dict], gstar: int, n_contexts_v: int = 91,
               n_seeds_v: int = 2, workers_v: int = 96, efficiency: float = 0.6, margin: float = 1.5) -> dict:
    """Dev91 wall projection from Phase T eval-CPU: mean eval-CPU per live eval (shard engine stats) x the projected
    live evals per dev91 unit at G* (both groups), x margin, / (workers x efficiency)."""
    tot_cpu = tot_runs = 0.0
    for rd in run_dirs:
        cfg = json.loads((Path(rd) / "config.json").read_text())
        st = sorted((Path(rd) / "shards").glob("status_T_*.json"))
        if not st:
            raise SystemExit(f"no shard status files under {rd}: cannot project dev91 wall")
        for p in st:
            e = json.loads(p.read_text())["engine"]
            tot_cpu += float(e.get("wall_s_total", 0.0))  # eval-seconds of single-threaded, core-pinned workers
            tot_runs += float(e.get("physical_runs", 0.0))
    if tot_runs <= 0:
        raise SystemExit("no physical worker runs recorded: cannot project dev91 wall")
    per_eval = tot_cpu / tot_runs
    # critical path: one unit's generations run one after another (each waits for its slowest eval)
    frac = (gstar + K.OLD_OFFSET) / float(cfg["gens"] + K.OLD_OFFSET)
    unit_wall = max(float(r.get("wall_s", 0.0)) for grp in recs.values() for r in grp.values())
    agg_evals = 22 + K.N_LIVE * gstar + 2 * 13  # + AUDIT evals (single + 12 members at <= 2 checkpoints)
    base_evals = 8 + K.N_LIVE * (gstar + K.OLD_OFFSET) + 23 + 9 + 1 + 3 * 13
    evals = n_contexts_v * n_seeds_v * (agg_evals + base_evals)
    throughput_h = evals * per_eval * margin / (workers_v * efficiency) / 3600.0
    critical_h = unit_wall * frac * margin / 3600.0
    hours = max(throughput_h, critical_h)
    return {"per_eval_s": per_eval, "physical_runs_T": tot_runs, "eval_s_T": tot_cpu,
            "projected_evals_v": evals, "throughput_hours_v": throughput_h, "critical_path_hours_v": critical_h,
            "max_unit_wall_s_T": unit_wall, "projected_hours_v": hours, "margin": margin,
            "efficiency": efficiency, "workers_v": workers_v}


def cmd_decide(args) -> None:
    run_dirs = [Path(x) for x in args.run_dirs.split(",")]
    cfgs = [json.loads((rd / "config.json").read_text()) for rd in run_dirs]
    for k in CONFIG_IDENTITY:
        if any(c.get(k) != cfgs[0].get(k) for c in cfgs[1:]):
            raise SystemExit(f"run dirs differ on {k}: not one experiment")
    cids = K.panel_p201b(C.train_rows())
    if sorted(c for cfg in cfgs for c in cfg["cids"]) != cids:
        raise SystemExit("run dirs do not partition P201b")
    recs = collect(run_dirs, "T", cids, K.SEEDS_T)
    vals = values_by_output(recs)
    fam_of = {r["context_sha256"]: r["family_id"] for r in C.train_rows()}
    agg = {gc: {cfg: vals[f"AGG.g{gc}.{cfg}"] for cfg in STACKS} for gc in K.CHECKPOINTS}
    dec = H.phase_t_decision(agg, fam_of, K.CHECKPOINTS, K.STACK)
    gstar = dec["G_star"]
    proj = projection(run_dirs, recs, gstar)
    capped = gstar
    while proj["projected_hours_v"] > 5.0 and capped != min(K.CHECKPOINTS):
        capped = max(gc for gc in K.CHECKPOINTS if gc < capped)
        proj = projection(run_dirs, recs, capped)
    dec["G_star_after_time_cap"] = capped
    dec["projection"] = proj
    if capped != gstar:
        st_gain = H.family_mean(agg[capped][K.STACK], fam_of) - H.family_mean(agg[capped]["single"], fam_of)
        dec["stack_gain_vs_single_at_G_star_after_cap"] = st_gain
        dec["stack"] = K.STACK if st_gain >= 0 else "single"
    locked = f"AGG.g{capped}.{dec['stack']}"
    inv = {k: int(sum(o["invalid"] for r in recs[("AGG" if k.startswith("AGG") else "BASE")].values()
                      for kk, o in r["outputs"].items() if kk == k)) for k in vals}
    n_units = len(cids) * len(K.SEEDS_T)
    dec["agg_invalid_share_locked"] = inv[locked] / n_units
    dec["hard_stop"] = dec["agg_invalid_share_locked"] > 0.02
    match = {}
    for gc in K.CHECKPOINTS:
        d = [recs["AGG"][u]["outputs"][f"AGG.g{gc}.single"]["live_evals"]
             - recs["BASE"][u]["outputs"][f"OLD.g{gc}.single"]["live_evals"] for u in recs["AGG"]]
        match[str(gc)] = {"max_abs_delta": int(max(abs(x) for x in d)), "n_over_tol": int(sum(abs(x) > K.LIVE_MATCH_TOL
                                                                                                for x in d))}
    dec["live_match"] = match
    dec["budget_shortfall_units"] = int(sum(1 for grp in recs.values() for r in grp.values()
                                            if r.get("evo", {}).get("budget_shortfall")))
    dec["seed_F_evals_not_22"] = int(sum(1 for r in recs["AGG"].values() if r["seed_F"]["evals_used"] != 22))
    report = {}
    for name, key in (("H257_REPLICA", "H257_REPLICA.CV1"), ("OLD_MATCHED", f"OLD.g{capped}.single"),
                      ("OLD_MATCHED_CV3", f"OLD.g{capped}.{K.STACK}"), ("H317_BASE", "H317_BASE.single"),
                      ("BANK_NN8_PORTFOLIO", "BANK_NN8_PORTFOLIO.CV5"), ("BANK_NN1", "BANK_NN1.single")):
        report[name] = {"primary": H.primary_test(vals[locked], vals[key], fam_of),
                        "headline": H.headline(vals[locked], vals[key], fam_of),
                        "adoption_gate": adoption_gate(vals[locked], vals[key], fam_of)}
    out = {"schema": "jope.h335.phase-t-decision/v1", "wave": K.WAVE, "decision": dec, "locked_output": locked,
           "family_means": {k: H.family_mean(v, fam_of) for k, v in sorted(vals.items())},
           "invalid_counts": inv, "n_units": n_units, "contrasts_vs_locked": report,
           "live_evals_locked_mean": float(np.mean([r["outputs"][locked]["live_evals"]
                                                    for r in recs["AGG"].values()])),
           "live_evals_old_mean": float(np.mean([r["outputs"][f"OLD.g{capped}.single"]["live_evals"]
                                                 for r in recs["BASE"].values()])),
           "guard": C.guard_report()}
    C.write_json(Path(args.out), out)
    print(json.dumps({"G_star": dec["G_star"], "G_star_after_time_cap": capped, "stack": dec["stack"],
                      "hard_stop": dec["hard_stop"], "projected_hours_v": proj["projected_hours_v"]}))


def adoption_gate(arm: Dict[str, float], ctrl: Dict[str, float], fam_of: Dict[str, str]) -> dict:
    s = C.robustness_summary(arm, ctrl, fam_of, K.BOOT_SEED)
    s["gate"] = C.contract_advance_gate(s, "lcb95", margin=0.001, signflip_alpha=K.ALPHA)
    return s


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "launch"):
        p = sub.add_parser(name)
        p.add_argument("--phase", default="T", choices=("T",))
        p.add_argument("--run-dir", required=True)
        p.add_argument("--worker", default=str(Path(C.DATA_ROOT) / "bin" / "bootstrap.s1main.superset-2a848c8f8085"))
        p.add_argument("--jobs", type=int, default=12)
        p.add_argument("--timeout", type=float, default=60.0)
        p.add_argument("--torch-threads", type=int, default=1)
        if name == "run":
            p.add_argument("--group", required=True, choices=("AGG", "BASE"))
            p.add_argument("--shard", required=True)
            p.add_argument("--cores", required=True)
            p.set_defaults(fn=cmd_run)
        else:
            p.add_argument("--neural-run-dir", default=str(Path(C.DATA_ROOT) / "neural" / "runs" / "H0.1-A"))
            p.add_argument("--host-part", default=None)
            p.add_argument("--cpus", required=True)
            p.add_argument("--shards-agg", type=int, required=True)
            p.add_argument("--shards-base", type=int, required=True)
            p.add_argument("--workers-per-shard", type=int, required=True)
            p.set_defaults(fn=cmd_launch)
    p = sub.add_parser("decide")
    p.add_argument("--run-dirs", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_decide)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
