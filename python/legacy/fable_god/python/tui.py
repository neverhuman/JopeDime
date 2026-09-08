"""Zero-dependency live TUI for fusion training. Reads metrics.jsonl (local or
tailed over ssh) and renders a dashboard: fixed expert baselines, train/val loss
+ val-AUC sparklines, an AUC gauge between avg and oracle, and checkpoints.

    python3 -m python.fusion.tui --metrics <dir_or_file>   [--once]
"""
from __future__ import annotations

import argparse
import json
import os
import time

SPARK = "▁▂▃▄▅▆▇█"
C = dict(reset="\033[0m", bold="\033[1m", dim="\033[2m", g="\033[92m", r="\033[91m",
         y="\033[93m", c="\033[96m", b="\033[94m", m="\033[95m", gray="\033[90m")


def spark(vals, lo=None, hi=None, width=48):
    if not vals:
        return ""
    vals = vals[-width:]
    lo = min(vals) if lo is None else lo
    hi = max(vals) if hi is None else hi
    rng = (hi - lo) or 1.0
    return "".join(SPARK[min(7, max(0, int((v - lo) / rng * 7)))] for v in vals)


def gauge(x, lo, hi, width=40):
    """Position x on [lo,hi]; markers for lo/hi, cursor o."""
    if hi <= lo:
        hi = lo + 1e-6
    pos = int(max(0, min(1, (x - lo) / (hi - lo))) * (width - 1))
    bar = ["─"] * width
    bar[0] = "├"; bar[-1] = "┤"; bar[pos] = "●"
    return "".join(bar)


def read(path):
    if os.path.isdir(path):
        path = os.path.join(path, "metrics.jsonl")
    if not os.path.exists(path):
        return None, [], []
    header = None; steps = []; ckpts = []
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("kind") == "header":
            header = r
        elif r.get("kind") == "step":
            steps.append(r)
        elif r.get("kind") == "ckpt":
            ckpts.append(r)
    return header, steps, ckpts


def cache_status(cache_dir, shardlogs):
    """Parse per-GPU shard logs + count cached files for the build-phase panel."""
    import glob as _g, re as _re
    files = len(_g.glob(os.path.join(cache_dir, "*.pt"))) if cache_dir else 0
    gpus = []
    for lg in sorted(_g.glob(shardlogs)) if shardlogs else []:
        try:
            txt = open(lg).read()
        except Exception:
            continue
        ms = _re.findall(r"shard (\d+)/(\d+) \| (\d+) tasks \| (\d+) tasks/s", txt)
        done = "packed cache done" in txt
        if ms:
            d, t, tk, r = ms[-1]
            gpus.append((int(d), int(t), int(tk), int(r), done))
        else:
            gpus.append((0, 0, 0, 0, done))
    return files, gpus


def bar(done, total, width=18):
    f = 0 if not total else done / total
    n = int(f * width)
    return "█" * n + "░" * (width - n)


def render(path, total_steps, cache_dir=None, shardlogs=None):
    hd, steps, ckpts = read(path)
    out = ["\033[2J\033[H"]  # clear + home
    title = f"{C['bold']}{C['c']}  FUSION — TabPFN-v3 + TabICL (frozen)  {C['reset']}"
    out.append(title)
    if hd is None:
        # cache-building phase panel (until training writes metrics)
        files, gpus = cache_status(cache_dir, shardlogs) if (cache_dir or shardlogs) else (0, [])
        if gpus:
            ndone = sum(g[4] for g in gpus); tot_tasks = sum(g[2] for g in gpus)
            rate = sum(g[3] for g in gpus)
            out.append(f"  {C['y']}◷ BUILDING CACHE on {len(gpus)}× A100{C['reset']}   "
                       f"{C['bold']}{tot_tasks:,}{C['reset']} tasks   {rate} tasks/s   "
                       f"{files} shard-files   {C['g']}{ndone}/{len(gpus)} GPUs done{C['reset']}")
            out.append(f"  {C['gray']}┌─ per-GPU " + "─" * 52 + C['reset'])
            for i, (d, t, tk, r, dn) in enumerate(gpus):
                col = C['g'] if dn else C['c']
                mark = "✔" if dn else " "
                out.append(f"  {C['gray']}│{C['reset']} gpu{i} {col}{bar(d, t)}{C['reset']} "
                           f"{d:>3}/{t or '?':<3} {tk:>6,} tasks {r:>3}/s {C['g']}{mark}{C['reset']}")
            # rough ETA from slowest GPU
            slow = min((g for g in gpus if g[1]), key=lambda g: g[0] / max(g[1], 1), default=None)
            if slow and slow[3]:
                remain = (slow[1] - slow[0]) * (slow[2] / max(slow[0], 1)) / max(slow[3], 1)
                out.append(f"  {C['dim']}ETA ~{int(remain/60)}m — training auto-starts when all "
                           f"GPUs finish, then this flips to the live dashboard{C['reset']}")
        else:
            out.append(f"{C['dim']}  waiting for metrics at {path} ...{C['reset']}")
        return "\n".join(out)
    ex = hd.get("expert_auc", []); names = hd.get("experts", [])
    avg = hd.get("avg_auc", 0); orac = hd.get("oracle_auc", 0)
    st = steps[-1] if steps else {}
    step = st.get("step", 0); sps = st.get("steps_per_s", 0)
    out.append(f"  {C['dim']}experts{C['reset']} {' '.join(names)}   "
               f"trainable {hd.get('trainable',0):,}   parity {hd.get('parity',0):.0e}   "
               f"step {C['bold']}{step}{C['reset']}/{total_steps}   {sps:.1f} it/s")
    # fixed baselines
    out.append(f"  {C['gray']}┌─ baselines (val, fixed) " + "─" * 38 + C['reset'])
    bl = "   ".join(f"{C['b']}{n}{C['reset']} {a:.4f}" for n, a in zip(names, ex))
    out.append(f"  {C['gray']}│{C['reset']} {bl}   {C['dim']}avg{C['reset']} {avg:.4f}   "
               f"{C['y']}ORACLE{C['reset']} {orac:.4f}")
    # val AUC
    aucs = [s["val_auc"] for s in steps]
    cur = aucs[-1] if aucs else 0; best = st.get("best_auc", cur)
    col = C['g'] if cur > orac else (C['c'] if cur > avg else (C['y'] if cur > max(ex or [0]) else C['r']))
    tag = (f"{C['g']}✔ BEATS ORACLE{C['reset']}" if best > orac else
           (f"{C['c']}✔ beats avg{C['reset']}" if best > avg else
            (f"{C['y']}beats best-expert{C['reset']}" if best > max(ex or [0]) else
             f"{C['dim']}below experts{C['reset']}")))
    out.append(f"  {C['gray']}┌─ val AUC " + "─" * 53 + C['reset'])
    out.append(f"  {C['gray']}│{C['reset']} {col}{C['bold']}{cur:.4f}{C['reset']}  "
               f"{col}{spark(aucs)}{C['reset']}   best {C['bold']}{best:.4f}{C['reset']}  {tag}")
    lo = min([avg, min(ex or [avg])]) - 0.005; hi = max(orac, best) + 0.005
    out.append(f"  {C['gray']}│{C['reset']} avg{C['dim']}{gauge(cur, lo, hi)}{C['reset']}oracle")
    # losses
    tl = [s["train_loss"] for s in steps]; vl = [s.get("val_loss", 0) for s in steps]
    out.append(f"  {C['gray']}┌─ loss " + "─" * 56 + C['reset'])
    out.append(f"  {C['gray']}│{C['reset']} {C['m']}train{C['reset']} {spark(tl)} {tl[-1]:.4f}"
               f"   {C['b']}val CE{C['reset']} {spark(vl)} {vl[-1]:.4f}" if tl else "  (no steps yet)")
    # checkpoints
    cks = " ".join(f"{c['path'].replace('.pt','')}" for c in ckpts[-6:])
    out.append(f"  {C['gray']}┌─ checkpoints " + "─" * 49 + C['reset'])
    out.append(f"  {C['gray']}│{C['reset']} {C['dim']}{cks or '—'}{C['reset']}   "
               f"{C['g']}best.pt{C['reset']} @ AUC {best:.4f}")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", required=True)
    ap.add_argument("--total-steps", type=int, default=4000)
    ap.add_argument("--cache", default=None, help="cache dir (for build-phase panel)")
    ap.add_argument("--shardlogs", default=None, help="glob for per-GPU shard logs")
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()
    # auto-detect an active cache build so the plain command shows the build phase
    if not args.cache and not args.shardlogs:
        import glob as _g
        mdir = args.metrics.rstrip("/")                     # metrics DIR (cache/fusion/run_ct)
        cache_parent = os.path.dirname(mdir) or "."         # cache/fusion
        base = os.path.dirname(cache_parent) or "."         # cache
        logs = (_g.glob(os.path.join(base, "ct_*.log")) or
                _g.glob(os.path.join(base, "pshard_*.log")) or
                _g.glob(os.path.join(base, "shard_*.log")))
        if logs:
            args.shardlogs = os.path.join(base, os.path.basename(logs[0]).split("_")[0] + "_*.log")
            subs = [d for d in _g.glob(os.path.join(cache_parent, "*"))
                    if os.path.isdir(d) and d.rstrip("/") != mdir and _g.glob(os.path.join(d, "*.pt"))]
            if subs:
                args.cache = max(subs, key=os.path.getmtime)
    while True:
        print(render(args.metrics, args.total_steps, args.cache, args.shardlogs), flush=True)
        if args.once:
            break
        time.sleep(2)


if __name__ == "__main__":
    main()
