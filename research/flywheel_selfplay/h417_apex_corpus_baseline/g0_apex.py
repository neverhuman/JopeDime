#!/usr/bin/env python3
"""H417 gate G0 (PREREG §6.2, part B): how much can a better search screener still buy?

Per context of panel G24 (the first 24 core FIT-SEL contexts in the order stream_seed(417201, "G0", cid)) the unchanged
locked search (seed F + Evo A K = 6 live / P = 48 screened, 50 generations, seed 417101; deploy_h410.replay) runs once
per screener arm, and each arm's stock CV3@12 output is scored once on AUDIT:
  SCR  the stock H0.1-A fold-0 out-of-fold pair (= the baseline GS arm);
  RND  seeded standard-normal scores: the pick is then driven by the residual corrector and the pick heuristics;
  ORC  the true SEARCH utility of every proposal (all 48 are evaluated; only the 6 picked enter the population).
The arms share one engine and memo, so a program is evaluated at most once; evaluation is deterministic, so sharing
changes no result. ORC's extra evaluations are its screening cost and are reported.

  run      --run-dir R --cores L [--jobs 6]      (xbabe0 docker sandbox)
  readout  --run-dir R --out O.json              (xbabe2, from the pulled run; numpy + h335_stats)
Rule (PREREG §6.2): build the surrogate wave only if ORC - SCR has a clipped family mean >= +0.02 and ORC beats SCR
in >= 60% of the contexts (ties count half).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

SEED, GENS, PANEL_SEED, N_PANEL = 417101, 50, 417201, 24
RND_SEED = 417202
ARMS = ("SCR", "RND", "ORC")
STAGE = "G0"
MIN_GAIN, MIN_WIN = 0.02, 0.60
GATE_FAIL_U = -2.0


def cmd_run(a) -> None:
    import apex_core as A  # FIRST import of the GP stack: h335_core pins + guard

    import numpy as np

    DP, K, C, L, D, ST, G, S = A.DP, A.K, A.C, A.L, A.D, A.ST, A.G, A.S
    if K.GROUP != "AGG":
        raise SystemExit("H335_GROUP must be AGG")
    A.install()
    order = sorted(A.cohort("FIT-SEL"), key=lambda c: C.stream_seed(PANEL_SEED, "G0", c))
    cids = sorted(order[:N_PANEL])
    if a.limit:  # harness smoke only (run_apex.sh passes it for APEX_SMOKE_* run dirs; those are never read out)
        cids = cids[:a.limit]
    run_dir = Path(a.run_dir)
    worker = Path(C.DATA_ROOT) / "bin" / "bootstrap.s1main.superset-2a848c8f8085"
    nd = {"SCR": Path(C.DATA_ROOT) / "neural" / "runs" / "H0.1-A"}
    rt, scr, man = A.build_rt(run_dir, L.parse_cpus(a.cores), worker, a.timeout, (SEED,), nd)

    class RandomScreener(S.Screener):
        name = "g0_random_normal"

        def score(self, cid, view_index_set, dsl_list):
            out = np.zeros((len(view_index_set), len(dsl_list)))
            for j, d in enumerate(dsl_list):
                sha = G.sha256_text(d)
                for i, v in enumerate(view_index_set):
                    out[i, j] = np.random.default_rng(C.stream_seed(RND_SEED, "RND", cid, int(v), sha)).standard_normal()
            return out

    class OracleScreener(S.Screener):
        name = "g0_oracle_search_utility"

        def __init__(self):
            self.evaluated = 0

        def score(self, cid, view_index_set, dsl_list):
            ctx = rt.engine.context(cid, rt.fam_of[cid])
            cands, pos = [], []
            for j, d in enumerate(dsl_list):
                c = G.safe_candidate(G.parse(d), "g0_oracle", (), ctx.width, cost_exempt=True)
                if c is not None and c.dsl == d:
                    cands.append(c)
                    pos.append(j)
            u = np.full(len(dsl_list), GATE_FAIL_U)
            if cands:
                D.assert_bank_disjoint(cid, cands, rt.target_bank(cid))
                res = rt.engine.search(ctx, cands)
                self.evaluated += len(cands)
                for j, r in zip(pos, res):
                    if r.status == "completed" and r.utility is not None and np.isfinite(r.utility):
                        u[j] = float(r.utility)
            return np.tile(u[None, :], (len(view_index_set), 1))

    rnd, orc = RandomScreener(), OracleScreener()
    sources = {"SCR": scr["SCR"], "RND": lambda cid: rnd, "ORC": lambda cid: orc}
    cfg = {"schema": "jope.h417.g0-config/v1", "cids": cids, "seed": SEED, "gens": GENS, "arms": list(ARMS),
           "timeout_s": a.timeout, "manifest_sha256": A.MANIFEST_SHA256, "screener_manifest": man,
           "code_sha256": K.code_manifest(A.code_files() + ("g0_apex.py",))}
    p = run_dir / "config_G0.json"
    if p.exists():
        old = json.loads(p.read_text())
        if any(old.get(k) != v for k, v in cfg.items()):
            raise SystemExit("run config mismatch")
    else:
        C.write_json(p, cfg)
    A.GPH.install(0)

    def unit(cid: str, _s) -> dict:
        hit = rt.store.result(STAGE, cid, "g0")
        if hit:
            return hit
        t0 = time.time()
        rec = {"type": "result", "schema": "jope.h417.g0-unit/v1", "cid": cid, "family_id": rt.fam_of[cid],
               "seed": "g0", "search_seed": SEED, "outputs": {}}
        for arm in ARMS:
            g = DP._gp(rt, cid, sources[arm](cid), SEED)
            res, cand_of = DP.replay(g, GENS)
            fd = ST.FoldData.from_ctx(g.ctx)
            with D.selection_scope(f"{STAGE}.{arm}.stock"):
                spec = ST.cv3_s1direct(res, fd, 12)
            raw, _ = DP.audit_spec(rt, f"{STAGE}.{arm}.stock", g, spec, cand_of)
            done = [r for r in res if r.status == "completed"]
            rec["outputs"][arm] = DP._summ(raw, spec, {
                "n_pool": len(res), "search_wall_s": float(sum(r.wall_s for r in res)),
                "best_search_u": float(max((r.utility for r in done), default=GATE_FAIL_U))})
        rec["wall_s"] = time.time() - t0
        rt.store.append(STAGE, cid, "g0", rec)
        return rec

    t0 = time.time()
    order = sorted(cids, key=lambda c: (-K.table_size(c), c))
    recs = L.run_parallel(unit, [(c, 0) for c in order], a.jobs)
    st = {"units": len(recs), "wall_s": time.time() - t0, "engine": rt.engine.stats(),
          "audit_ledger": rt.ledger.check(), "guard": C.guard_report(), "bank_counters": dict(D.BANK_COUNTERS),
          "physical_by_kind": DP.physical_by_kind(rt), "oracle_proposals_evaluated": orc.evaluated,
          "host": os.uname().nodename}
    if st["bank_counters"]["same_family_bank_hits"]:
        raise AssertionError("same_family_bank_hits != 0")
    C.write_json(run_dir / f"status_G0_{int(t0)}.json", st)
    rt.engine.close()
    print(json.dumps({"units": len(recs), "wall_s": round(st["wall_s"], 1), "physical_by_kind": st["physical_by_kind"],
                      "oracle_proposals_evaluated": orc.evaluated}))


def cmd_readout(a) -> None:
    import apex_core_meta as AM
    import train_h416 as T16

    run_dir = Path(a.run_dir)
    cfg = json.loads((run_dir / "config_G0.json").read_text())
    np, H = T16._stats_mod()
    fam_of = AM.family_of(cfg["cids"])
    vals = {arm: {} for arm in ARMS}
    best = {arm: {} for arm in ARMS}
    inv = {arm: 0 for arm in ARMS}
    for c in cfg["cids"]:
        o = T16._last_result(run_dir / "stages" / STAGE / f"{c}__g0.jsonl")["outputs"]
        for arm in ARMS:
            bad = o[arm].get("invalid") or o[arm]["raw"].get("status") != "completed"
            vals[arm][c] = 0.0 if bad else float(o[arm]["value"])  # invalid = the SEARCH-mean predictor
            inv[arm] += bool(bad)
            best[arm][c] = o[arm]["best_search_u"]
    pairs = {f"{x}-{y}": T16.pair_stats(np, H, vals[x], vals[y], fam_of) for x, y in (("ORC", "SCR"), ("RND", "SCR"),
                                                                                    ("ORC", "RND"))}
    g = pairs["ORC-SCR"]
    decision = bool(g["clipped_mean"] >= MIN_GAIN and g["win_frac"] >= MIN_WIN)
    res = {"schema": "jope.h417.g0-readout/v1", "n": len(cfg["cids"]),
           "family_mean_audit_u": {arm: H.family_mean(vals[arm], fam_of) for arm in ARMS},
           "family_mean_best_search_u": {arm: H.family_mean(best[arm], fam_of) for arm in ARMS},
           "invalid": inv, "pairs": pairs, "rule": {"min_clipped_gain": MIN_GAIN, "min_win_frac": MIN_WIN},
           "build_surrogate_wave": decision, "values": vals}
    Path(a.out).write_text(json.dumps(res, sort_keys=True))
    print(json.dumps({k: v for k, v in res.items() if k != "values"}, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--cores", required=True)
    p.add_argument("--jobs", type=int, default=6)
    p.add_argument("--timeout", type=float, default=60.0)
    p.add_argument("--limit", type=int, default=0)
    p = sub.add_parser("readout")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--out", required=True)
    a = ap.parse_args()
    {"run": cmd_run, "readout": cmd_readout}[a.cmd](a)


if __name__ == "__main__":
    main()
