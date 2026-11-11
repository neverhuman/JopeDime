#!/usr/bin/env python3
"""H421 DEV test deploys (PREREG section 4): the stored H416 SXB recipe exactly as H420's TBOTH (sources GS, GE, GX;
stock library + lang_h420.LIB_EXT; design BCAR.mix; 300 s limit) plus 20 more library presets read from a file:

  TPORT  + LIB_PORT.json   the selected presets
  TRAND  + LIB_RAND.json   20 random pool candidates (control: same count, no learning)
  TCOV   + LIB_COV.json    20 presets chosen for coverage (the ablation)

The variants run in H420's DEV run directory, so they share its evaluation memo and its stored TBOTH records (the
baseline B*, never recomputed); only programs that are not in the memo are evaluated.

  train_h421.py deploy --run-dir R --panel P --variant V --presets FILE --presets-sha256 HEX --cores L [--jobs J]
                       [--inner-jobs 3] [--shard k/n]
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

WAVE = "H421"
VARIANTS = ("TPORT", "TRAND", "TCOV")


def cmd_deploy(a) -> None:
    import train_h420 as T20

    A, LANG, rt, scr, man, cids, nd, chk = T20._setup(a, "GX", True)   # B* = H420 TBOTH
    DH, K, C, L, D = A.DH, A.K, A.C, A.L, A.D
    DP = DH.DP
    raw = Path(a.presets).read_bytes()
    if hashlib.sha256(raw).hexdigest() != a.presets_sha256:
        raise SystemExit("the preset file is not the published one")
    extra = [(str(n), str(d)) for n, d in json.loads(raw)]
    have = {n for n, _ in DP.LIB_PRESETS}
    if len(extra) != 20 or len({n for n, _ in extra}) != 20 or have & {n for n, _ in extra}:
        raise SystemExit("the preset file must hold 20 presets with new, distinct names")
    if any(DP._pipe(d) is None for _, d in extra):
        raise SystemExit("a preset does not parse as a single pipe with a head")
    DP.LIB_PRESETS = tuple(DP.LIB_PRESETS) + tuple(extra)
    design = T20._design()
    seed = A.SEARCH_SEED
    run_dir = Path(a.run_dir)
    tag = a.variant
    cfg = {"wave": WAVE, "tag": tag, "variant": a.variant, "mode": "confirm", "panel": a.panel, "cids": cids,
           "sources": list(nd), "neural": {s: str(p) for s, p in nd.items()}, "seed": seed, "timeout_s": a.timeout,
           "design": design, "library_ext": [n for n, _ in LANG.LIB_EXT], "library_added": [n for n, _ in extra],
           "presets_sha256": a.presets_sha256, "lib_presets": len(DP.LIB_PRESETS),
           "periodic_max_width": LANG.PERIODIC_MAX_WIDTH,
           "meta": {"bag_seed": DH.M.BAG_SEED, "n_bags": DH.M.N_BAGS, "w_min": DH.M.W_MIN,
                    "max_members": DH.M.MAX_MEMBERS},
           "manifest_sha256": A.MANIFEST_SHA256,
           "code_sha256": K.code_manifest(A.code_files() + ("lang_h420.py", "gp_h420.py", "train_h420.py",
                                                            "train_h421.py")),
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
          "physical_by_kind": DH.DP.physical_by_kind(rt), "lang_selfcheck": chk, "host": os.uname().nodename,
          "presets_sha256": a.presets_sha256}
    if st["bank_counters"]["same_family_bank_hits"]:
        raise AssertionError("same_family_bank_hits != 0")
    st["shard"] = a.shard
    C.write_json(run_dir / f"status_{tag}_{int(t0)}_s{sk}of{sn}.json", st)
    print(json.dumps({k: st[k] for k in ("tag", "units", "wall_s", "physical_by_kind", "akf_ledger", "audit_ledger")}
                     | {"guard_refused": st["guard"].get("refused"), "bank": st["bank_counters"]}))
    rt.engine.close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("deploy")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--panel", required=True)
    p.add_argument("--cores", required=True)
    p.add_argument("--timeout", type=float, default=300.0)   # the stored H416 / H417 / H420 deploy limit
    p.add_argument("--variant", required=True, choices=VARIANTS)
    p.add_argument("--presets", required=True)
    p.add_argument("--presets-sha256", required=True)
    p.add_argument("--jobs", type=int, default=1)
    p.add_argument("--inner-jobs", type=int, default=3)
    p.add_argument("--shard", default="0/1", help="k/n: this process runs the contexts order[k::n]")
    a = ap.parse_args()
    cmd_deploy(a)


if __name__ == "__main__":
    main()
