#!/usr/bin/env python3
"""dlhwm LV4 stats backfill (label-free train-row statistics) for the DL_FINAL HWM model.
Replays stats_v2.1 (seeded), stats_v3 and stats_v4 from the raw outer-train jtt1 inputs, exactly as wave 10
(w10_stats.py, md5 c990f912...) but with the four modules imported from THIS directory (pinned md5s below), so the
host's other copies can never be picked up. No GP label, sealed list, lockbox or S3 object is read.
usage: stats_backfill.py --mod stats_v2|stats_v3|stats_v4 --cids file.json --out DIR --workers N
Per-context JSON in DIR/<cid>.json (skip if present). Run at nice 19; it yields to everything else."""
import argparse, hashlib, json, os, sys, time
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ[v] = "1"
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
PINS = {"ta_common.py": "a159da048d22f482b0b89126d1f96aca", "stats_v2.py": "d85ab4dc5edc5a1483a3b8163265ff61",
        "stats_v3.py": "38a1df5a51c0e3d32d03f7ebb8b26f7b", "stats_v4.py": "39f55236c074f133f7dc6c8293b33a90"}
for f, m in PINS.items():
    got = hashlib.md5(open(os.path.join(HERE, f), "rb").read()).hexdigest()
    if got != m: raise SystemExit(f"pin mismatch {f}: {got} != {m}")
import ta_common, stats_v3, stats_v2, stats_v4   # order matters: later modules find these in sys.modules
for mod in (ta_common, stats_v3, stats_v2, stats_v4):
    if os.path.dirname(os.path.abspath(mod.__file__)) != HERE: raise SystemExit(f"module from wrong dir: {mod.__file__}")
W = "/home/ubuntu/jopedime_probe_exp/ofm-20261008"
A = None; PLAN = {}; M = None
def go(c):
    fn = f"{A.out}/{c}.json"
    if os.path.exists(fn): return c, "skip"
    try:
        t0 = time.time(); s = M.stats(PLAN[c]); s["_secs"] = time.time() - t0
        open(fn + ".tmp", "w").write(json.dumps(s)); os.replace(fn + ".tmp", fn); return c, f"ok {s['_secs']:.1f}s"
    except Exception as e: return c, "ERR " + repr(e)[:200]
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--mod", required=True); ap.add_argument("--cids", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8); A = ap.parse_args(); os.makedirs(A.out, exist_ok=True); want = set(json.load(open(A.cids)))
    M = {"stats_v2": stats_v2, "stats_v3": stats_v3, "stats_v4": stats_v4}[A.mod]
    for l in open(f"{W}/plan/plan_full.jsonl"):
        d = json.loads(l)
        if d["context_sha256"] in want: PLAN[d["context_sha256"]] = d
    pending_contexts = [c for c in sorted(PLAN) if not os.path.exists(f"{A.out}/{c}.json")]
    print(time.strftime("%H:%M:%S"), A.mod, "pending_contexts", len(pending_contexts), "of", len(want), "in plan", len(PLAN), flush=True)
    import multiprocessing as mp
    with mp.get_context("fork").Pool(A.workers, maxtasksperchild=200) as pool:
        for i, (c, m) in enumerate(pool.imap_unordered(go, pending_contexts, chunksize=2)):
            if m.startswith("ERR") or i % 250 == 0: print(time.strftime("%H:%M:%S"), i, c[:12], m, flush=True)
    print(time.strftime("%H:%M:%S"), "ALL DONE", flush=True)
