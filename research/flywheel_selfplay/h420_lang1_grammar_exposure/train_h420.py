#!/usr/bin/env python3
"""H420: the stored H416 SXB recipe (design BCAR.mix, pinned) deployed with the exposed language, per variant:

  B      sources GS, GE, GL   stock library            (= H417's deploy, recomputed in this run directory)
  TGX    sources GS, GE, GX   stock library            (the extended-grammar run replaces the GL run)
  TLIB   sources GS, GE, GL   library + lang_h420.LIB_EXT
  TBOTH  sources GS, GE, GX   library + lang_h420.LIB_EXT

All variants of one panel share one run directory (one evaluation memo: GS + GE + GL + GX concatenated), run one after
the other, and differ only in the tag. GX is replayed with the extended generators ON in its own replay thread; GS, GE
and GL replay with the stock generators (lang_h420 wrappers OFF), exactly as in H417.

  deploy       --run-dir R --panel P --variant V --cores L [--jobs J] [--inner-jobs 3]
  replaycheck  --run-dir R --panel P --cores L [--limit N]   every source replays from the memo with 0 worker runs
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

WAVE = "H420"
VARIANTS = {"B": ("GL", False), "TGX": ("GX", False), "TLIB": ("GL", True), "TBOTH": ("GX", True)}
NEURAL = {"GS": "H0.1-A", "GE": "H381-E2-S3fz", "GL": "H415-L", "GX": "H415-L"}
DESIGN_SHA256 = "0f0c96ad5d819db42ba1c242c814f4eac3345d9ce2744924666a89820314f8c4"   # h416 DESIGN.json (BCAR.mix)


def _design() -> dict:
    raw = (HERE / "DESIGN.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != DESIGN_SHA256:
        raise SystemExit("DESIGN.json is not the locked H416 design")
    return json.loads(raw)


def _setup(a, third: str, lib_ext: bool):
    import apex_core as A  # FIRST import of the GP stack: h335_core pins + guard

    import lang_h420 as LANG

    DH, K, C, L, G = A.DH, A.K, A.C, A.L, A.G
    if K.GROUP != "AGG":
        raise SystemExit("H335_GROUP must be AGG")
    A.install()
    LANG.install(G)
    chk = LANG.selfcheck(G)
    DP = DH.DP
    if lib_ext:
        names = [n for n, _ in DP.LIB_PRESETS]
        if any(n in names for n, _ in LANG.LIB_EXT):
            raise SystemExit("library extension already applied")
        DP.LIB_PRESETS = tuple(DP.LIB_PRESETS) + tuple(LANG.LIB_EXT)
    cids = A.panel(a.panel)
    run_dir = Path(a.run_dir)
    memo = run_dir / f"gproot_{K.GROUP}" / "memo"
    missing = [c for c in cids if not (memo / f"{c}.split.jsonl").is_file()]
    if missing:
        raise SystemExit(f"{len(missing)} contexts have no concatenated memo (run the memo step first)")
    worker = Path(C.DATA_ROOT) / "bin" / "bootstrap.s1main.superset-2a848c8f8085"
    nd = {s: Path(C.DATA_ROOT) / "neural" / "runs" / NEURAL[s] for s in ("GS", "GE", third)}
    rt, scr, man = A.build_rt(run_dir, L.parse_cpus(a.cores), worker, a.timeout, (A.SEARCH_SEED,), nd)
    gx_ids = set()
    if third == "GX":
        gx_ids = {id(scr["GX"](c)) for c in cids}
        others = {id(scr[s](c)) for s in ("GS", "GE") for c in cids}
        if len(gx_ids) != 1 or gx_ids & others:
            raise SystemExit("the GX screener object is not distinct")
    stock_replay = DP.replay

    def replay(g, gens=DP.GENS):
        if id(g.screener) in gx_ids:
            with LANG.gx():
                return stock_replay(g, gens)
        return stock_replay(g, gens)

    DP.replay = replay
    return A, LANG, rt, scr, man, cids, nd, chk


def cmd_deploy(a) -> None:
    third, lib_ext = VARIANTS[a.variant]
    A, LANG, rt, scr, man, cids, nd, chk = _setup(a, third, lib_ext)
    DH, K, C, L, D = A.DH, A.K, A.C, A.L, A.D
    design = _design()
    seed = A.SEARCH_SEED
    run_dir = Path(a.run_dir)
    tag = a.variant
    cfg = {"wave": WAVE, "tag": tag, "variant": a.variant, "mode": "confirm", "panel": a.panel, "cids": cids,
           "sources": list(nd), "neural": {s: str(p) for s, p in nd.items()}, "seed": seed, "timeout_s": a.timeout,
           "design": design, "library_ext": [n for n, _ in LANG.LIB_EXT] if lib_ext else [],
           "lib_presets": len(DH.DP.LIB_PRESETS), "periodic_max_width": LANG.PERIODIC_MAX_WIDTH,
           "lang": {"p_ext_head": LANG.P_EXT_HEAD, "p_ext_stage": LANG.P_EXT_STAGE,
                    "histgbm2_work": LANG.EXT_HISTGBM2_WORK},
           "meta": {"bag_seed": DH.M.BAG_SEED, "n_bags": DH.M.N_BAGS, "w_min": DH.M.W_MIN,
                    "max_members": DH.M.MAX_MEMBERS},
           "manifest_sha256": A.MANIFEST_SHA256,
           "code_sha256": K.code_manifest(A.code_files() + ("lang_h420.py", "gp_h420.py", "train_h420.py")),
           "screener_manifest": man}
    p = run_dir / f"config_{tag}.json"
    try:  # shards start together: the first creates the config, the others must equal it
        with open(p, "x") as f:
            json.dump(cfg, f, sort_keys=True)
    except FileExistsError:
        for _try in range(10):
            try:
                old = json.loads(p.read_text())
                break
            except json.JSONDecodeError:
                time.sleep(1.0)
        if any(old.get(k) != v for k, v in json.loads(json.dumps(cfg)).items()):
            raise SystemExit("run config mismatch")
    DH.GPH.install(0)
    npz_dir = run_dir / f"h416_{tag}_npz"
    t0 = time.time()
    order = sorted(cids, key=lambda c: (-K.table_size(c), c))
    sk, sn = (int(v) for v in a.shard.split("/"))
    order = order[sk::sn]
    recs = L.run_parallel(lambda c, _s: DH.h416_unit(rt, tag, c, scr, seed, "confirm", npz_dir, design,
                                                     a.inner_jobs), [(c, 0) for c in order], a.jobs)
    st = {"tag": tag, "variant": a.variant, "mode": "confirm", "units": len(recs), "wall_s": time.time() - t0,
          "engine": rt.engine.stats(), "audit_ledger": rt.ledger.check(), "akf_ledger": DH.DP15.KF_LEDGER.check(),
          "guard": C.guard_report(), "bank_counters": dict(D.BANK_COUNTERS),
          "physical_by_kind": DH.DP.physical_by_kind(rt), "lang_selfcheck": chk, "host": os.uname().nodename}
    if st["bank_counters"]["same_family_bank_hits"]:
        raise AssertionError("same_family_bank_hits != 0")
    st["shard"] = a.shard
    C.write_json(run_dir / f"status_{tag}_{int(t0)}_s{sk}of{sn}.json", st)
    print(json.dumps({k: st[k] for k in ("tag", "units", "wall_s", "physical_by_kind", "akf_ledger", "audit_ledger")}
                     | {"guard_refused": st["guard"].get("refused"), "bank": st["bank_counters"]}))
    rt.engine.close()


def cmd_replaycheck(a) -> None:
    """Each source's search replays from the concatenated memo without a single worker run (GX with the extended
    generators ON). A miss means the replay generated a program the original run did not evaluate."""
    A, LANG, rt, scr, man, cids, nd, chk = _setup(a, "GX", False)
    DH = A.DH
    DP = DH.DP
    DH.GPH.install(0)
    out = {"lang_selfcheck_stock_equal": chk["stock_stream_equal"], "sources": {}}
    import concurrent.futures as cf

    sub = cids[: a.limit or len(cids)]
    before = dict(DP.physical_by_kind(rt))

    def one(job):
        name, cid = job
        res, _ = DP.replay(DP._gp(rt, cid, scr[name](cid), A.SEARCH_SEED), DP.GENS)
        return name, len(res)

    with cf.ThreadPoolExecutor(max_workers=max(1, a.jobs)) as ex:
        got = list(ex.map(one, [(n, c) for n in nd for c in sub]))
    after = dict(DP.physical_by_kind(rt))
    runs = sum(after.get(k, 0) - before.get(k, 0) for k in after)
    for name in nd:
        out["sources"][name] = {"contexts": len(sub), "results": sum(n for s, n in got if s == name)}
    out["worker_runs"] = runs
    out["ok"] = runs == 0
    print(json.dumps(out))
    rt.engine.close()
    if not out["ok"]:
        raise SystemExit("replay is not memo-exact")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("deploy", "replaycheck"):
        p = sub.add_parser(name)
        p.add_argument("--run-dir", required=True)
        p.add_argument("--panel", required=True)
        p.add_argument("--cores", required=True)
        p.add_argument("--timeout", type=float, default=300.0)   # the stored H416 / H417 deploy limit
        if name == "deploy":
            p.add_argument("--variant", required=True, choices=sorted(VARIANTS))
            p.add_argument("--jobs", type=int, default=12)
            p.add_argument("--inner-jobs", type=int, default=3)
            p.add_argument("--shard", default="0/1", help="k/n: this process runs the contexts order[k::n]")
        else:
            p.add_argument("--limit", type=int, default=0)
            p.add_argument("--jobs", type=int, default=12)
    a = ap.parse_args()
    {"deploy": cmd_deploy, "replaycheck": cmd_replaycheck}[a.cmd](a)


if __name__ == "__main__":
    main()
