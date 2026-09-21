"""Live TUI for the MONSTER fusion training. Reads the real training log + live
nvidia-smi. The centerpiece plot is the fused-model val AUC over steps against the
two expert baselines (TabPFN-v3, TabICL) and the best-of-both oracle — so you can
SEE the single fused model climb above both. Nothing synthetic: every number is
parsed from the actual run.

  ssh ubuntu@<super1> 'python3 -m python.router.monster_tui --log ~/fable_god/cache/monster.log'
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import time

C = dict(
    r="\033[0m",
    b="\033[1m",
    dim="\033[2m",
    cyan="\033[96m",
    blue="\033[94m",
    mag="\033[95m",
    grn="\033[92m",
    yel="\033[93m",
    red="\033[91m",
    gray="\033[90m",
    wht="\033[97m",
)
STEP_RE = re.compile(
    r"step\s+(\d+)\s+\|\s+loss\s+([\d.]+)\s+\|\s+fused AUC\s+([\d.]+)\s+"
    r"\(pfn\s+([\d.]+)\s+icl\s+([\d.]+)\)\s+\|\s+win-both\s+(\d+)%\s+\|\s+beats-both\s+(\w+)"
)


def parse(path):
    hd = {}
    steps = []
    try:
        txt = open(path).read()
    except Exception:
        return hd, steps
    m = re.search(r"MONSTER model:\s+([\d.]+)M params \(trunk\s+([\d.]+)M\)", txt)
    if m:
        hd["params"], hd["trunk"] = float(m.group(1)), float(m.group(2))
    m = re.search(
        r"BASELINE val AUC: TabPFN-v3\s+([\d.]+)\s+\|\s+TabICL\s+([\d.]+)\s+\|\s+best-of-both\s+([\d.]+)", txt
    )
    if m:
        hd["pfn"], hd["icl"], hd["best"] = map(float, m.groups())
    m = re.search(r"TRAIN\s+(\d+)\s+tasks.*?in\s+(\d+)\s+batches.*?VAL\s+(\d+)", txt)
    if m:
        hd["ntrain"], hd["nbatch"], hd["nval"] = map(int, m.groups())
    mu = re.search(r"unlocked\s+([\d.]+)M expert key-layer params", txt)
    hd["unlocked"] = float(mu.group(1)) if mu else None
    for mm in STEP_RE.finditer(txt):
        steps.append(
            dict(
                step=int(mm.group(1)),
                loss=float(mm.group(2)),
                fused=float(mm.group(3)),
                pfn=float(mm.group(4)),
                icl=float(mm.group(5)),
                win=int(mm.group(6)),
                beats=mm.group(7) == "True",
            )
        )
    its = re.findall(r"it/s\s+([\d.]+)", txt)
    hd["its"] = float(its[-1]) if its else None
    dps = re.findall(r"dps\s+(\d+)", txt)
    hd["dps"] = int(dps[-1]) if dps else None
    pk = re.findall(r"peakmem\s+([\d.]+)GB", txt)
    hd["peak"] = float(pk[-1]) if pk else None
    live = re.findall(r"(?:^step\s+|^hb\s+)(\d+)", txt, re.M)  # newest step from eval OR heartbeat
    hd["cur_step"] = int(live[-1]) if live else (steps[-1]["step"] if steps else 0)
    return hd, steps


def gpus():
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=4,
        ).stdout
        r = []
        for ln in out.strip().splitlines():
            i, u, mu, mt = [x.strip() for x in ln.split(",")]
            r.append((int(i), int(u), int(mu), int(mt)))
        return r
    except Exception:
        return []


def _interp(series, w):
    """Resample a series to exactly w points (pure python) so the line spans the
    full plot width even with few eval points."""
    n = len(series)
    if n == 1:
        return [series[0]] * w
    if n >= w:
        return [series[int(i * (n - 1) / (w - 1))] for i in range(w)]
    out = []
    for i in range(w):
        x = i * (n - 1) / (w - 1)
        lo = int(x)
        hi = min(lo + 1, n - 1)
        f = x - lo
        out.append(series[lo] * (1 - f) + series[hi] * f)
    return out


def line_plot(series, refs, w=66, h=16):
    """series: list of y (chronological). refs: list of (label,color,y). Colored
    char-cell line chart with connected fused line + horizontal baseline refs."""
    if not series:
        return ["  (waiting for eval points...)"]
    col = _interp(series, w)
    W = len(col)
    ys = col + [y for _, _, y in refs]
    lo, hi = min(ys), max(ys)
    pad = (hi - lo) * 0.15 + 1e-4
    lo, hi = lo - pad, hi + pad

    def row_of(y):
        return int(round((hi - y) / (hi - lo) * (h - 1)))

    grid = [[" " for _ in range(W)] for _ in range(h)]
    color = [[None for _ in range(W)] for _ in range(h)]
    # reference baselines
    for lbl, cl, y in refs:
        rr = row_of(y)
        if 0 <= rr < h:
            for c in range(W):
                if grid[rr][c] == " ":
                    grid[rr][c] = "─"
                    color[rr][c] = cl
    # fused line (connected)
    prev = None
    for c in range(W):
        r = row_of(col[c])
        r = max(0, min(h - 1, r))
        if prev is not None:  # connect vertically
            a, b = sorted((prev, r))
            for rr in range(a, b + 1):
                grid[rr][c] = "│" if rr not in (a, b) else "●"
                color[rr][c] = "cyan"
        grid[r][c] = "●"
        color[r][c] = "cyan"
        prev = r
    # assemble with y-axis labels
    out = []
    for rr in range(h):
        yv = hi - (hi - lo) * rr / (h - 1)
        line = f"{C['gray']}{yv:6.4f}{C['r']} "
        for c in range(W):
            ch = grid[rr][c]
            cl = color[rr][c]
            line += (C.get(cl, "") + ch + C["r"]) if cl else ch
        out.append(line)
    return out


def spark(vals, n=48):
    if not vals:
        return ""
    v = vals[-n:]
    lo, hi = min(v), max(v)
    rng = hi - lo or 1
    bars = "▁▂▃▄▅▆▇█"
    return "".join(bars[min(7, int((x - lo) / rng * 7))] for x in v)


def bar(frac, w=16):
    n = int(max(0, min(1, frac)) * w)
    return "█" * n + "░" * (w - n)


def render(log, total_steps, metric="AUC", ea="TabPFN-v3", eb="TabICL", domain="binary"):
    hd, steps = parse(log)
    o = ["\033[2J\033[H"]
    o.append(
        f"{C['b']}{C['cyan']}  CHIMERA ({domain}) — {ea} + {eb} → one {hd.get('params', 0):.0f}M model{C['r']}"
    )
    stage = "trunk warm-up (experts frozen)"
    if hd.get("unlocked"):
        stage = f"{C['grn']}fine-tuning {hd['unlocked']:.0f}M key expert-layer params{C['r']}"
    o.append(
        f"  {C['dim']}params {hd.get('params', 0):.1f}M (trunk {hd.get('trunk', 0):.1f}M) · "
        f"train {hd.get('ntrain', '?')} tasks/{hd.get('nbatch', '?')} batches · "
        f"val {hd.get('nval', '?')} REAL · stage: {stage}{C['dim']}{C['r']}"
    )
    # === throughput panel (real, from the run) ===
    cur = hd.get("cur_step", 0)
    its = hd.get("its")
    dps = hd.get("dps")
    pk = hd.get("peak")
    tp = f"  {C['b']}{C['yel']}⚡ "
    tp += f"{its:.2f} steps/s{C['r']}" if its else f"warming up…{C['r']}"
    if dps:
        tp += f"  {C['yel']}{dps:,} datasets/s{C['r']}"
    tp += f"  {C['wht']}step {cur}/{total_steps}{C['r']}"
    if its and total_steps and cur:
        eta = (total_steps - cur) / its
        tp += f"  {C['dim']}ETA {int(eta // 60)}m{int(eta % 60):02d}s{C['r']}"
    if pk:
        pc = "red" if pk > 37 else ("yel" if pk > 32 else "grn")
        tp += f"  {C[pc]}peak {pk:.1f}/40GB{C['r']}"
    o.append(tp)
    if not steps or "pfn" not in hd:
        o.append(f"\n  {C['dim']}waiting for baselines + first eval...{C['r']}")
        return "\n".join(o)
    st = steps[-1]
    # === the centerpiece: fused AUC vs both experts ===
    refs = [(ea, "blue", hd["pfn"]), (eb, "mag", hd["icl"]), ("best-of-both", "gray", hd["best"])]
    o.append(
        f"\n  {C['b']}VALIDATION {metric}{C['r']} {C['dim']}(held-out REAL {domain}, {hd.get('nval', '?')} datasets){C['r']}   "
        f"{C['cyan']}●fused{C['r']} {C['blue']}─{ea} {hd['pfn']:.4f}{C['r']} "
        f"{C['mag']}─{eb} {hd['icl']:.4f}{C['r']} {C['gray']}─oracle {hd['best']:.4f}{C['r']}"
    )
    for ln in line_plot([s["fused"] for s in steps], refs):
        o.append("  " + ln)
    o.append(f"  {C['gray']}└ step {steps[0]['step']}{' ' * 40}step {st['step']}{C['r']}")
    # === verdict ===
    dp, di = st["fused"] - hd["pfn"], st["fused"] - hd["icl"]
    beat = st["fused"] > hd["pfn"] and st["fused"] > hd["icl"]
    vc = C["grn"] if beat else C["red"]
    o.append(
        f"\n  {vc}{C['b']}fused {metric} {st['fused']:.4f}{C['r']}  "
        f"vs {ea} {C['grn' if dp > 0 else 'red']}{dp:+.4f}{C['r']}  "
        f"vs {eb} {C['grn' if di > 0 else 'red']}{di:+.4f}{C['r']}  "
        f"{vc}{'✓ BEATS BOTH' if beat else '✗ not yet'}{C['r']}  "
        f"win-both {st['win']}%"
    )
    # === loss + win sparklines ===
    o.append(
        f"  {C['mag']}loss{C['r']} {spark([s['loss'] for s in steps])} {steps[-1]['loss']:.3f}   "
        f"{C['yel']}win-both%{C['r']} {spark([s['win'] for s in steps])} {st['win']}%"
    )
    # === GPU utilization (live, real) ===
    g = gpus()
    if g:
        o.append(f"\n  {C['b']}GPU utilization{C['r']}")
        for i, u, mu, mt in g:
            act = u > 5
            cl = C["grn"] if act else C["gray"]
            o.append(
                f"  {cl}gpu{i}{C['r']} {cl}{bar(u / 100)}{C['r']} {u:3d}%  {mu / 1024:.1f}/{mt / 1024:.0f}GB"
                + (f"  {C['cyan']}◄ training{C['r']}" if mu > 800 else "")
            )
    # throughput
    if len(steps) >= 2:
        o.append(
            f"\n  {C['dim']}{len(steps)} evals · latest step {st['step']}"
            f"{('/' + str(total_steps)) if total_steps else ''}{C['r']}"
        )
    return "\n".join(o)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=os.path.expanduser("~/fable_god/cache/monster.log"))
    ap.add_argument("--total-steps", type=int, default=8000)
    ap.add_argument("--metric", default=None)
    ap.add_argument("--ea", default=None)
    ap.add_argument("--eb", default=None)
    ap.add_argument("--domain", default=None)
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()
    reg = "reg" in os.path.basename(args.log).lower()  # auto-detect regression run
    metric = args.metric or ("R²" if reg else "AUC")
    ea = args.ea or ("TabPFN-v3-reg" if reg else "TabPFN-v3")
    eb = args.eb or ("TabICL-reg" if reg else "TabICL")
    domain = args.domain or ("regression" if reg else "binary")
    while True:
        print(render(args.log, args.total_steps, metric, ea, eb, domain), flush=True)
        if args.once:
            break
        time.sleep(2)


if __name__ == "__main__":
    main()
