#!/usr/bin/env python3
"""A4 verification: replay H0.5 jobs of a Gate H0 GP run with the current code against a COPY of their memo.

Read-only over the run dir and its memo: the selected h04/h05 records, config.json and decisions.json are copied into
a separate replay dir, and each context's memo file into <replay>/gproot/memo. Then `gp_lab.stage_h05` is run for:
  * `--n-completed` completed (context, seed) jobs, chosen deterministically (sorted, spread over all shards except
    `--skip-shard`): the replayed record must equal the original byte-for-byte after removing the wall-clock field
    `wall_s`, with 0 new worker runs;
  * every job named by `--crashed cid:seed`: it must now complete (new worker runs allowed; nothing is written into
    the original run dir).
Writes <replay>/replay_report.json. Usage (xbabe0, inside sandbox_h317_docker.sh):
  gp_replay_h05.py --run-dir R --memo-dir M --replay-dir X --cores 16-23 --neural-run-dir N --shards 11 --skip-shard 4 \
    --crashed <cid>:317102
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common_h317 as C  # noqa: E402

C.install_guard()

import gp_grammar_h317 as G  # noqa: E402
import gp_lab as L  # noqa: E402


def canon(rec: dict) -> str:
    """Stored form (JSON round trip: int dict keys become strings, as in the run's JSONL), minus `wall_s`."""
    rec = json.loads(json.dumps(rec, default=C._json_default))
    return json.dumps({k: v for k, v in rec.items() if k != "wall_s"}, sort_keys=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--memo-dir", required=True)
    ap.add_argument("--replay-dir", required=True)
    ap.add_argument("--cores", required=True)
    ap.add_argument("--neural-run-dir", required=True)
    ap.add_argument("--n-completed", type=int, default=22)
    ap.add_argument("--shards", type=int, default=11)
    ap.add_argument("--skip-shard", type=int, default=None)
    ap.add_argument("--crashed", action="append", default=[])
    ap.add_argument("--jobs", type=int, default=8)
    args = ap.parse_args()
    run, memo, rep_dir = Path(args.run_dir), Path(args.memo_dir), Path(args.replay_dir)
    if rep_dir == run or run in rep_dir.parents or rep_dir in run.parents:
        raise SystemExit("--replay-dir must be separate from the run dir")
    if rep_dir.exists() and any(rep_dir.iterdir()):
        raise SystemExit("--replay-dir must be new/empty")
    cfg = json.loads((run / "config.json").read_text())
    dec = json.loads((run / "decisions.json").read_text())
    scheme = dec["H0.4"]["used"]
    store = L.Store(run)
    cids = sorted(cfg["pilot"])
    shard_of = {c: i % args.shards for i, c in enumerate(cids)}
    done = [
        (c, s)
        for c in cids
        for s in cfg["seeds"]
        if store.n_results(f"h05__{scheme}", c, s) == 1
        and shard_of[c] != args.skip_shard
    ]
    per = -(
        -args.n_completed // max(1, args.shards - (args.skip_shard is not None))
    )  # ceil: spread over shards
    chosen = [
        x
        for k in range(args.shards)
        if k != args.skip_shard
        for x in [d for d in done if shard_of[d[0]] == k][:per]
    ]
    crashed = [(x.split(":")[0], int(x.split(":")[1])) for x in args.crashed]
    rep_dir.mkdir(parents=True, exist_ok=True)
    for f in ("config.json", "decisions.json"):
        shutil.copy2(run / f, rep_dir / f)
    for c, s in (
        chosen + crashed
    ):  # h04 records (seed sets) + memo copies; h05 records are NOT copied
        src = run / "stages" / "h04" / f"{c}__{s}.jsonl"
        dst = rep_dir / "stages" / "h04" / src.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        m = rep_dir / "gproot" / "memo" / f"{c}.split.jsonl"
        if not m.exists():
            m.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(memo / f"{c}.split.jsonl", m)
    orig = {
        (c, s): store.result(f"h05__{scheme}", c, s) for c, s in chosen
    }  # read-only from the run dir
    G.MAX_EXPANDED_WIDTH = int(cfg["max_expanded_width"])  # the launched run's knob
    rt = L.build_runtime(
        rep_dir,
        L.parse_cpus(args.cores),
        "neural",
        "neural",
        gp_root=rep_dir / "gproot",
        neural_run_dir=Path(args.neural_run_dir),
        seeds=tuple(cfg["seeds"]),
    )
    out = {"run_dir": str(run), "scheme": scheme, "completed": [], "crashed": []}
    for c, s in chosen:
        before = rt.engine.counters["physical_runs"]
        rec = L.stage_h05(rt, c, s, scheme)
        out["completed"].append(
            {
                "cid": c,
                "seed": s,
                "shard": shard_of[c],
                "identical_minus_wall": canon(rec) == canon(orig[(c, s)]),
                "new_worker_runs": rt.engine.counters["physical_runs"] - before,
                "dedup_replaced": any(
                    "dedup_replaced" in a for a in rec["arms"].values()
                ),
            }
        )
    for c, s in crashed:
        before = rt.engine.counters["physical_runs"]
        rec = L.stage_h05(rt, c, s, scheme)
        out["crashed"].append(
            {
                "cid": c,
                "seed": s,
                "completed": rec.get("type") == "result",
                "new_worker_runs": rt.engine.counters["physical_runs"] - before,
                "live_per_gen": {a: v["live_per_gen"] for a, v in rec["arms"].items()},
                "dedup_replaced": {
                    a: v.get("dedup_replaced") for a, v in rec["arms"].items()
                },
                "budget_shortfall": rec.get("budget_shortfall"),
            }
        )
    out["summary"] = {
        "n_completed_replayed": len(out["completed"]),
        "all_identical": all(x["identical_minus_wall"] for x in out["completed"]),
        "total_new_worker_runs_completed": sum(
            x["new_worker_runs"] for x in out["completed"]
        ),
        "crashed_now_complete": all(x["completed"] for x in out["crashed"]),
        "engine": rt.engine.stats(),
        "guard": C.guard_report(),
    }
    C.write_json(rep_dir / "replay_report.json", out)
    rt.engine.close()
    print(json.dumps(out["summary"], default=str))


if __name__ == "__main__":
    main()
