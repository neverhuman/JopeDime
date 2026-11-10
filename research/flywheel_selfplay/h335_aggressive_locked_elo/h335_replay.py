#!/usr/bin/env python3
"""H335 engineering check (PREREG §3.3, outcome-free): the H335 code path reproduces the H317 runs it builds on.

Against COPIES of H317 memo files (P205 contexts, seed 317101; values already published in H317's closeout):
  AGG  group: h335_lab.unit_agg(gens=10, checkpoints=(10,), stacks=("single",)) must give, per context, exactly the
       exploratory run's `single` AUDIT value (Seed F -> Evo A 10 gens -> single), with 0 new worker runs;
  BASE group: Seed A -> D.evo_panmictic(gens=10) -> single (the h335_lab.unit_base code path) must give exactly the
       main run's H0.6 `single` AUDIT value, with 0 new worker runs (memo written by the pinned worker; the superset
       worker's parity makes the replay valid).
Inputs: --inputs DIR with memo_explore/<cid>.split.jsonl, memo_main/<cid>.split.jsonl, views-search/<cid>.npz,
rec_explore/<cid>__317101.jsonl, rec_h06/<cid>__317101.jsonl (copied on the host from the H317 run dirs).
Run on xbabe0 inside sandbox_h335_docker.sh, once per group (H335_GROUP=AGG|BASE). Writes <out>/replay_<group>.json.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import h335_core as K  # noqa: E402

import argparse  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402

import crop_shuffle_synthetic_views as V  # noqa: E402

import h335_lab as LAB  # noqa: E402

C, D, L = K.C, K.D, K.L
SEED = 317101


def _last(p: Path) -> dict:
    out = None
    for line in p.read_text().splitlines():
        d = json.loads(line)
        if d.get("type") == "result":
            out = d
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cores", required=True)
    ap.add_argument("--neural-run-dir", default=str(Path(C.DATA_ROOT) / "neural" / "runs" / "H0.1-A"))
    ap.add_argument("--worker", default=str(Path(C.DATA_ROOT) / "bin" / "bootstrap.s1main.superset-2a848c8f8085"))
    args = ap.parse_args()
    inp, out = Path(args.inputs), Path(args.out)
    run_dir = out / f"run_{K.GROUP}"
    if run_dir.exists():
        raise SystemExit(f"{run_dir} exists (use a fresh --out)")
    V.VIEW_SEARCH_DIR = inp / "views-search"  # P205 SEARCH view sets (copies of H317's) for this check only
    memo_src = inp / ("memo_explore" if K.GROUP == "AGG" else "memo_main")
    cids = sorted(p.name.split(".")[0] for p in memo_src.glob("*.split.jsonl"))
    (run_dir / f"gproot_{K.GROUP}" / "memo").mkdir(parents=True)
    for c in cids:
        shutil.copy2(memo_src / f"{c}.split.jsonl", run_dir / f"gproot_{K.GROUP}" / "memo" / f"{c}.split.jsonl")
    C.write_json(run_dir / "config.json", {"check": "h335_replay", "group": K.GROUP, "cids": cids})
    G = K.G
    G.MAX_EXPANDED_WIDTH = L.RUN_MAX_EXPANDED_WIDTH
    rt = LAB.build_rt(run_dir, "T", L.parse_cpus(args.cores), Path(args.worker), 60.0, (SEED,),
                      Path(args.neural_run_dir), 1)
    rows = []
    for c in cids:
        if K.GROUP == "AGG":
            rec = LAB.unit_agg(rt, "R", c, SEED, gens=10, checkpoints=(10,), stacks=("single",))
            got = rec["outputs"]["AGG.g10.single"]["value"]
            want = _last(inp / "rec_explore" / f"{c}__{SEED}.jsonl")["single"]["value"]
        else:
            g = rt.gp(c, SEED)
            a = D.seed_A(g)
            evo = D.evo_panmictic(g, a.order, a.results, gens=10, n_prop=K.N_PROP, n_live=K.N_LIVE, arm="evoA")
            cand_of = {**{x.sha: x for x in a.order}, **evo.cand_of}
            got = LAB._out(rt, "R.OLD.g10.single", g, LAB._upto(evo, 10), cand_of, "single", 0)["value"]
            want = _last(inp / "rec_h06" / f"{c}__{SEED}.jsonl")["configs"]["single"]["value"]
        rows.append({"cid": c, "got": got, "want": want, "equal": got == want})
    st = rt.engine.stats()
    rep = {"group": K.GROUP, "seed": SEED, "rows": rows, "all_equal": all(r["equal"] for r in rows),
           "physical_runs": st.get("physical_runs"), "memo_hits": st.get("memo_hits"),
           "pass": all(r["equal"] for r in rows) and st.get("physical_runs") == 0, "guard": C.guard_report()}
    C.write_json(out / f"replay_{K.GROUP}.json", rep)
    rt.engine.close()
    print(json.dumps({k: rep[k] for k in ("group", "all_equal", "physical_runs", "memo_hits", "pass")}))
    if not rep["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
