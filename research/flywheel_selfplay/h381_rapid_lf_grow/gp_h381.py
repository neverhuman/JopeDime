#!/usr/bin/env python3
"""H381 GP runner: the frozen H335 AGG recipe with a low-fidelity SEARCH envelope and a full-row finish (PREREG §2).

E1 idea (owner, 2026-11-05): "shorten the randomly sampled rows to ~300 if it trains much faster". Here the GP
evaluates every candidate on a 300-row SEARCH subsample (150 rows per native SEARCH fold, one seeded random row per
y-stratum), then
  LF.g<gc>.<stack>  the H335 outputs built from the low-fidelity SEARCH results (stack weights fit on 300 rows),
  FR.g<gc>.<stack>  the top fr_k low-fidelity candidates (rank_key) + the LF CV3 members re-evaluated on the FULL
                    1,120-row SEARCH envelope, outputs rebuilt from those full-row results.
Every output's members are AUDIT-fit on the full SEARCH rows and scored on the unchanged 480 AUDIT rows, so AUDIT u
is directly comparable with the STORED H364 X0 (runs/H364_C_xbabe0, AGG.g50.CV3@12; never rerun).

Only runtime patches; no H317/H335 file is edited:
  E.build_context  -> lf_build (returns LFContextEnvs: subsampled search/y0/y1/b0/b1/search_path; full-row audit,
                      audit_path, ya, ba and y_search_mean; memo mode "lf<rows>")
  the unit          -> unit_lf (h335_lab.unit_agg + the FR block + per-output sha lists for CPU accounting)
With --lf-rows 0 the unit is h335_lab.unit_agg exactly (outputs AGG.*), used for the replay check and for E2-in-loop.

Commands (in sandbox_h335_docker.sh on xbabe0, H335_GROUP=AGG, H335_STAGE=<stage holding code/>):
  idcheck --run-dir R                 envelope identity on 3 S60 cids (no worker run)
  launch  --run-dir R --cpus L --shards N --workers-per-shard W --lf-rows n --gens G --checkpoints a,b --fr-k k
          [--panel S60|R6|R60] [--neural-run-dir D]
  run     spawned by launch
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
import threading  # noqa: E402
import time  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from typing import Dict, List, Sequence  # noqa: E402

import numpy as np  # noqa: E402

import h335_lab as LAB  # noqa: E402

C, D, E, L, ST, G = K.C, K.D, K.E, K.L, K.ST, K.G

import multiscale_axial_bridge as MB  # noqa: E402  (pinned H317 module; S3 = the E2 grown screener, PREREG §3)

MB.SCALES.setdefault("S3", MB.ScaleConfig("S3", d=32, n_heads=4, n_layers=3, cand_hidden=228, ffn_mult=4.0))

WAVE = "H381"
PHASE = "C"
SEED = 364101  # = H364 X0's search seed
PANEL_SEED = 381101
P201D_FILE = Path(C.DATA_ROOT) / "h364" / "P201d.json"
P201D_SHA = "9a1ace6cfd9abdf6c436d980b61d98b59825e56ea3719cd67911a0f2aa25de98"  # H364 PREREG cids sha256
CODE_FILES = LAB.PHASE_T_FILES + ("gp_h381.py",)
LF_ROW_SEED = 381111

# ----------------------------------------------------------------------------------------------------------------------
# Panels
# ----------------------------------------------------------------------------------------------------------------------
def p201d() -> Dict[str, str]:
    d = json.loads(P201D_FILE.read_text())
    if d["cids_sha256"] != P201D_SHA or hashlib.sha256("\n".join(d["cids"]).encode()).hexdigest() != P201D_SHA:
        raise AssertionError("P201d file differs from the H364 panel")
    return dict(d["family_of"])


def panel(name: str) -> List[str]:
    """S60: the first 60 P201d contexts (one per family) in the order stream_seed(381101, 'S60', cid).
    R6: the first 6 of S60 (replay check). DEV2: P201d order positions 60-61 (smoke only, never scored).
    R60: positions 62-121 (the E2L TRAIN replication panel, PREREG §8; families disjoint from S60)."""
    fam = p201d()
    order = sorted(fam, key=lambda c: C.stream_seed(PANEL_SEED, "S60", c))
    s60 = order[:60]
    if len({fam[c] for c in s60}) != 60:
        raise AssertionError("S60 must hold 60 families")
    r60 = order[62:122]
    if len({fam[c] for c in r60}) != 60 or {fam[c] for c in r60} & {fam[c] for c in s60}:
        raise AssertionError("R60 must hold 60 families disjoint from S60")
    return {"S60": s60, "R6": s60[:6], "ID3": s60[:3], "DEV2": order[60:62], "R60": r60}[name]


# ----------------------------------------------------------------------------------------------------------------------
# Low-fidelity SEARCH envelope
# ----------------------------------------------------------------------------------------------------------------------
_ORIG_BUILD = E.build_context
LF = {"rows": 0}


def _rmse(v: np.ndarray) -> float:
    return float(np.sqrt(np.mean(v ** 2)))


def lf_keep(cid: str, search: "C.Jtt1", per_fold: int) -> np.ndarray:
    """Per native SEARCH fold: sort its rows by y (stable), cut into per_fold equal strata (np.array_split), keep one
    seeded-random row per stratum. Original row order is kept. A fold with <= per_fold rows is kept whole."""
    y, folds = search.y, search.folds
    keep = []
    for f in (0, 1):
        idx = np.flatnonzero(folds == f)
        if len(idx) <= per_fold:
            keep.append(idx)
            continue
        rng = np.random.default_rng(C.stream_seed(LF_ROW_SEED, "lf", cid, f, per_fold))
        order = idx[np.argsort(y[idx], kind="stable")]
        keep.append(np.array([b[int(rng.integers(len(b)))] for b in np.array_split(order, per_fold)]))
    return np.sort(np.concatenate(keep))


@dataclass
class LFContextEnvs(E.ContextEnvs):
    full_ysm: float = float("nan")
    lf_rows: int = 0
    n_full_search: int = 0

    @property
    def y_search_mean(self) -> float:  # AUDIT baselines stay on the full SEARCH mean (= X0)
        return self.full_ysm

    @property
    def mode(self) -> str:
        return f"lf{self.lf_rows}"


def lf_build(cid: str, family_id: str, env_dir: Path, full_mode: bool = False):
    full = _ORIG_BUILD(cid, family_id, env_dir, full_mode)
    n = int(LF["rows"])
    if full_mode or n <= 0:
        return full
    s = full.search
    keep = lf_keep(cid, s, n // 2)
    if len(keep) >= s.nt:
        return full
    sub = C.Jtt1(s.magic, len(keep), 0, s.width, s.flag, s.folds[keep].copy(), s.mat[keep].copy())
    b = sub.to_worker_bytes()
    sp = E._write_env(env_dir, b)
    y, f = sub.y, sub.folds
    y0, y1 = y[f == 0].copy(), y[f == 1].copy()
    return LFContextEnvs(cid, family_id, full.width, sub, full.audit, full.audit_mask, sp, full.audit_path, y0, y1,
                         full.ya, _rmse(y0 - y1.mean()), _rmse(y1 - y0.mean()), full.ba, False,
                         env_sha={"search": hashlib.sha256(b).hexdigest(), "audit": full.env_sha["audit"]},
                         full_ysm=full.y_search_mean, lf_rows=n, n_full_search=int(s.nt))


def install(lf_rows: int) -> None:
    LF["rows"] = int(lf_rows)
    E.build_context = lf_build


def full_context(engine, cid: str, fam: str):
    """The unpatched full-row SEARCH context in the same engine (memo mode 'split'), built once per engine."""
    key = (cid, "FR")
    with engine._lock:
        ctx = engine._ctx.get(key)
        if ctx is not None:
            return ctx
        lk = engine._ctx_locks.setdefault(key, threading.Lock())
    with lk:
        with engine._lock:
            ctx = engine._ctx.get(key)
        if ctx is not None:
            return ctx
        ctx = _ORIG_BUILD(cid, fam, engine.env_dir, False)
        engine._load_memo(ctx)
        with engine._lock:
            engine._ctx[key] = ctx
        return ctx


# ----------------------------------------------------------------------------------------------------------------------
# Unit
# ----------------------------------------------------------------------------------------------------------------------
def _wall(results) -> float:
    seen: Dict[str, float] = {}
    for r in results:
        seen.setdefault(r.sha, float(r.wall_s))
    return float(sum(seen.values()))


def unit_lf(rt, cid: str, seed: int, gens: int, checkpoints: Sequence[int], fr_k: int,
            stacks: Sequence[str] = LAB.STACKS) -> dict:
    assert K.GROUP == "AGG"
    checkpoints = LAB._cps(checkpoints)
    stage = f"{PHASE}_AGG"
    hit = rt.store.result(stage, cid, seed)
    if hit:
        return hit
    g = rt.gp(cid, seed)
    lf = isinstance(g.ctx, LFContextEnvs)
    t0 = time.time()
    a = D.seed_A(g)
    e = D.seed_E(g, a)
    f = K.seed_F(g, e)
    evo = D.evo_panmictic(g, f.order, f.results, gens=gens, n_prop=K.N_PROP, n_live=K.N_LIVE, arm="evoA")
    cand_of = {**{c.sha: c for c in f.order}, **evo.cand_of}
    rec = {"type": "result", "schema": "jope.h381.unit/v1", "cid": cid, "seed": seed, "phase": PHASE, "group": "AGG",
           "grammar_sha256": K.GRAMMAR_SHA256, "worker_sha256": K.SUPERSET_SHA256, "gens": gens,
           "lf_rows": int(LF["rows"]), "lf_active": lf, "n_search_rows": g.ctx.n_search,
           "n_full_search_rows": g.ctx.n_full_search if lf else g.ctx.n_search,
           "env_sha": dict(g.ctx.env_sha), "ba": g.ctx.ba, "y_search_mean": g.ctx.y_search_mean,
           "seed_F": f.search_metrics(), "evo": LAB._evo_summary(evo), "outputs": {},
           "gen_of": {s: int(v) for s, v in evo.gen_of.items()}, "fr": {}}
    pre = "LF" if lf else "AGG"
    gfr = None
    for gc in checkpoints:
        res = LAB._upto(evo, gc)
        live = f.evals_used + LAB._live_upto(evo, gc)
        for cfg in stacks:
            k = f"{pre}.g{gc}.{cfg}"
            rec["outputs"][k] = LAB._out(rt, f"{PHASE}.{k}", g, res, cand_of, cfg, live)
            rec["outputs"][k]["search_wall_s"] = _wall(res)
        if not (lf and fr_k > 0):
            continue
        with D.selection_scope(f"{PHASE}.FR.g{gc}.pick"):
            done = sorted([r for r in res if r.status == "completed"], key=D.rank_key)
            pick = [r.sha for r in done[:fr_k]]
            spec = rec["outputs"][f"LF.g{gc}.{K.STACK}"].get("spec") or {}
            for s in spec.get("members", []):
                if s not in pick:
                    pick.append(s)
        if gfr is None:
            fctx = full_context(rt.engine, cid, rt.fam_of[cid])
            gfr = D.GPContext(fctx, rt.engine, rt.screener_for(cid), rt.target_bank(cid), rt.coder, seed)
        res_fr = gfr.evaluate([cand_of[s] for s in pick])
        rec["fr"][str(gc)] = {"shas": pick, "n": len(pick), "search_wall_s": _wall(res_fr),
                              "n_completed": sum(1 for r in res_fr if r.status == "completed")}
        for cfg in stacks:
            k = f"FR.g{gc}.{cfg}"
            rec["outputs"][k] = LAB._out(rt, f"{PHASE}.{k}", gfr, res_fr, cand_of, cfg, live + len(pick))
            rec["outputs"][k]["search_wall_s"] = _wall(res) + _wall(res_fr)
    rec["n_new_head_evaluated"] = sum(1 for r in evo.results if K.new_head(r.dsl))
    rec["live_evals_total"] = f.evals_used + evo.live_evals
    rec["wall_s"] = time.time() - t0
    rt.store.append(stage, cid, seed, rec)
    return rec


# ----------------------------------------------------------------------------------------------------------------------
# Commands
# ----------------------------------------------------------------------------------------------------------------------
CONFIG_IDENTITY = ("cids", "seeds", "gens", "checkpoints", "lf_rows", "fr_k", "n_prop", "n_live", "stacks",
                   "worker_sha256", "timeout_s", "neural_run_dir", "code_sha256", "max_expanded_width")


def _config(run_dir: Path, a, cids: List[str]) -> dict:
    cfg = {"schema": "jope.h381.gp-config/v1", "wave": WAVE, "phase": PHASE, "panel": a.panel, "cids": cids,
           "cids_sha256": hashlib.sha256("\n".join(cids).encode()).hexdigest(), "group": "AGG", "seeds": [SEED],
           "gens": a.gens, "checkpoints": [int(x) for x in a.checkpoints.split(",")], "lf_rows": a.lf_rows,
           "fr_k": a.fr_k, "n_prop": K.N_PROP, "n_live": K.N_LIVE, "stacks": list(LAB.STACKS),
           "worker_sha256": C.sha256_file(Path(a.worker)), "timeout_s": a.timeout,
           "neural_run_dir": str(a.neural_run_dir), "code_sha256": K.code_manifest(CODE_FILES),
           "max_expanded_width": L.RUN_MAX_EXPANDED_WIDTH, "host": os.uname().nodename}
    p = run_dir / "config.json"
    if p.exists():
        old = json.loads(p.read_text())
        for k in CONFIG_IDENTITY:
            if old.get(k) != cfg[k]:
                raise SystemExit(f"run config mismatch on {k}")
        return old
    C.write_json(p, cfg)
    return cfg


def cmd_idcheck(a) -> None:
    """No worker run: (1) lf_rows=0 contexts reproduce the stored H364 memo envelope shas; (2) lf_rows=300 keeps the
    AUDIT sha, ba, ya and y_search_mean and has 150 + 150 SEARCH rows; (3) lf_keep is deterministic."""
    fam = p201d()
    x0memo = Path(C.DATA_ROOT) / "runs" / "H364_C_xbabe0" / "gproot_AGG" / "memo"
    env_dir = Path(a.run_dir) / "idcheck_envs"
    out = []
    for cid in panel("ID3"):
        shas = {}
        with open(x0memo / f"{cid}.split.jsonl") as fh:
            for line in fh:
                d = json.loads(line)
                shas.setdefault(d["kind"], set()).add(d["env_sha256"])
        LF["rows"] = 0
        full = lf_build(cid, fam[cid], env_dir)
        LF["rows"] = 300
        lf = lf_build(cid, fam[cid], env_dir)
        k1 = lf_keep(cid, full.search, 150)
        k2 = lf_keep(cid, full.search, 150)
        rec = {"cid": cid, "x0_env_shas": {k: sorted(v) for k, v in shas.items()}, "full_env_sha": full.env_sha,
               "full_in_x0": sorted(full.env_sha.values()) == sorted({s for v in shas.values() for s in v}),
               "lf_nt": int(lf.search.nt), "lf_fold_counts": [int((lf.search.folds == f).sum()) for f in (0, 1)],
               "audit_sha_same": lf.env_sha["audit"] == full.env_sha["audit"], "ba_same": lf.ba == full.ba,
               "ya_same": bool(np.array_equal(lf.ya, full.ya)), "ysm_same": lf.y_search_mean == full.y_search_mean,
               "audit_path_same": lf.audit_path == full.audit_path, "deterministic": bool(np.array_equal(k1, k2)),
               "lf_mode": lf.mode, "lf_b0_b1": [lf.b0, lf.b1], "full_b0_b1": [full.b0, full.b1]}
        out.append(rec)
    ok = all(r["full_in_x0"] and r["lf_nt"] == 300 and r["lf_fold_counts"] == [150, 150] and r["audit_sha_same"]
             and r["ba_same"] and r["ya_same"] and r["ysm_same"] and r["audit_path_same"] and r["deterministic"]
             for r in out)
    C.write_json(Path(a.run_dir) / "idcheck.json", {"schema": "jope.h381.idcheck/v1", "ok": ok, "contexts": out})
    print(json.dumps({"ok": ok, "contexts": [{k: r[k] for k in ("cid", "full_in_x0", "lf_nt", "audit_sha_same",
                                                                "ba_same", "ysm_same", "deterministic")} for r in out]}))
    if not ok:
        raise SystemExit("idcheck failed")


def cmd_panel(a) -> None:
    fam = p201d()
    rec = {"schema": "jope.h381.panels/v1", "rule": "P201d (H364) ordered by stream_seed(381101, 'S60', cid); S60 = "
           "first 60, R6 = first 6 (replay), DEV2 = positions 60-61 (smoke only), R60 = positions 62-121 (E2L "
           "replication, PREREG §8)", "p201d_cids_sha256": P201D_SHA}
    for name in ("S60", "R6", "DEV2", "R60"):
        cids = panel(name)
        rec[name] = {"cids": cids, "families": [fam[c] for c in cids],
                     "cids_sha256": hashlib.sha256("\n".join(cids).encode()).hexdigest()}
    C.write_json(Path(a.out), rec)
    print(json.dumps({k: rec[k]["cids_sha256"] for k in ("S60", "R6", "DEV2", "R60")}))


def cmd_run(a) -> None:
    if K.GROUP != "AGG":
        raise SystemExit("H335_GROUP must be AGG")
    run_dir = Path(a.run_dir)
    cfg = json.loads((run_dir / "config.json").read_text())
    if C.sha256_file(Path(a.worker)) != cfg["worker_sha256"]:
        raise SystemExit("worker differs from the run config")
    if K.code_manifest(CODE_FILES) != cfg["code_sha256"]:
        raise SystemExit("code differs from the run config")
    install(cfg["lf_rows"])
    cids = L.shard_cids(cfg["cids"], a.shard)
    rt = LAB.build_rt(run_dir, "T", L.parse_cpus(a.cores), Path(a.worker), float(cfg["timeout_s"]), cfg["seeds"],
                      Path(cfg["neural_run_dir"]), a.torch_threads)
    t0 = time.time()
    order = sorted(cids, key=lambda c: (-K.table_size(c), c))
    work = [(c, s) for c in order for s in cfg["seeds"]]
    recs = L.run_parallel(lambda c, s: unit_lf(rt, c, s, cfg["gens"], cfg["checkpoints"], cfg["fr_k"],
                                               tuple(cfg["stacks"])), work, a.jobs)
    status = {"wave": WAVE, "phase": PHASE, "shard": a.shard, "records": len(recs), "wall_s": time.time() - t0,
              "engine": rt.engine.stats(), "audit_ledger": rt.ledger.check(), "guard": C.guard_report(),
              "bank_counters": dict(D.BANK_COUNTERS), "host": os.uname().nodename}
    if status["bank_counters"]["same_family_bank_hits"]:
        raise AssertionError("same_family_bank_hits != 0")
    C.write_json(run_dir / "shards" / f"status_{a.shard.replace('/', 'of')}_{int(t0)}.json", status)
    rt.engine.close()


def cmd_launch(a) -> None:
    if K.GROUP != "AGG":
        raise SystemExit("H335_GROUP must be AGG")
    run_dir = Path(a.run_dir)
    (run_dir / "shards").mkdir(parents=True, exist_ok=True)
    cids = panel(a.panel)
    _config(run_dir, a, cids)
    rt = LAB.build_rt(run_dir, "T", [0], Path(a.worker), a.timeout, (SEED,), Path(a.neural_run_dir), 1)
    rt.engine.close()
    cpus, w = L.parse_cpus(a.cpus), a.workers_per_shard
    if a.shards * w > len(cpus):
        raise SystemExit("not enough CPUs for shards x workers")
    procs = []
    for k in range(a.shards):
        mine = ",".join(str(c) for c in cpus[k * w:(k + 1) * w])
        env = dict(os.environ, H335_GROUP="AGG")
        env.pop("H317_LEARNERS_V1", None)
        cmd = ["taskset", "-c", mine, "nice", "-n", "19", sys.executable, "-u", str(Path(__file__).resolve()), "run",
               "--run-dir", str(run_dir), "--shard", f"{k}/{a.shards}", "--cores", mine, "--jobs", str(a.jobs),
               "--timeout", str(a.timeout), "--worker", a.worker, "--torch-threads", str(a.torch_threads)]
        log = open(run_dir / "shards" / f"shard{k}of{a.shards}.log", "a")
        procs.append((k, subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env), log))
    bad = [k for k, p, _ in procs if p.wait() != 0]
    for *_, log in procs:
        log.close()
    if bad:
        raise SystemExit(f"shards failed: {bad}")
    print(json.dumps({"wave": WAVE, "launched_shards": a.shards, "contexts": len(cids)}))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("idcheck")
    p.add_argument("--run-dir", required=True)
    p = sub.add_parser("panel")
    p.add_argument("--out", required=True)
    for name in ("launch", "run"):
        p = sub.add_parser(name)
        p.add_argument("--run-dir", required=True)
        p.add_argument("--worker", default=str(Path(C.DATA_ROOT) / "bin" / "bootstrap.s1main.superset-2a848c8f8085"))
        p.add_argument("--jobs", type=int, default=12)
        p.add_argument("--timeout", type=float, default=60.0)
        p.add_argument("--torch-threads", type=int, default=1)
        if name == "launch":
            p.add_argument("--neural-run-dir", default=str(Path(C.DATA_ROOT) / "neural" / "runs" / "H0.1-A"))
            p.add_argument("--cpus", required=True)
            p.add_argument("--shards", type=int, required=True)
            p.add_argument("--workers-per-shard", type=int, required=True)
            p.add_argument("--panel", default="S60", choices=("S60", "R6", "DEV2", "R60"))
            p.add_argument("--lf-rows", type=int, required=True)
            p.add_argument("--gens", type=int, required=True)
            p.add_argument("--checkpoints", required=True)
            p.add_argument("--fr-k", type=int, default=24)
        else:
            p.add_argument("--shard", required=True)
            p.add_argument("--cores", required=True)
    a = ap.parse_args()
    {"idcheck": cmd_idcheck, "panel": cmd_panel, "launch": cmd_launch, "run": cmd_run}[a.cmd](a)


if __name__ == "__main__":
    main()
