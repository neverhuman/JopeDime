"""'TA training' dashboard section (imported by gen.py): R^2-over-steps for TA_Hyperion_v8 and TA_Dope_V2 vs foundation /
tabpfn35 reference lines, latest checkpoint table with CIs and controls, and training progress / ETA. Reads only the small
files ta_sync.sh mirrors into dashboard/ta/ (timeseries.jsonl, progress.json, run.json per model)."""
import datetime as dt, html, json, os

HERE = os.path.dirname(os.path.abspath(__file__)); TA = os.path.join(HERE, "ta")
MT = dt.timezone(dt.timedelta(hours=-6))
MODELS = [("hv8", "TA_Hyperion_v8", "#f59e0b"), ("dv2", "TA_Dope_V2", "#22d3ee"),
          ("v8braw", "TA_Hyperion_v8b-raw", "#c084fc"), ("v8bft", "TA_Hyperion_v8b-ft", "#fb7185")]
FOUNDATION_3REP = 0.476


def _load(k):
    ts, prog, run = [], None, None
    p = os.path.join(TA, f"{k}_timeseries.jsonl")
    if os.path.exists(p):
        for l in open(p):
            try: ts.append(json.loads(l))
            except Exception: pass
    for nm in ("progress", "run"):
        q = os.path.join(TA, f"{k}_{nm}.json")
        if os.path.exists(q):
            try:
                v = json.load(open(q))
                if nm == "progress": prog = v
                else: run = v
            except Exception: pass
    ts.sort(key=lambda r: r["step"])
    return ts, prog, run


def _mt(utc):
    try: return dt.datetime.strptime(utc, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=dt.timezone.utc).astimezone(MT).strftime("%-I:%M %p MT")
    except Exception: return "—"


def _refs(all_ts):
    for ts in all_ts:
        for r in ts:
            return r["arms"]["F"]["R2"], r["arms"]["B"]["R2"]
    return None, None


def _svg(series, refF, refB, w=900, h=320):
    pts = [(s, r) for _, _, _, ts in series for s, r in [(x["step"], x["arms"]["TA"]["R2"]) for x in ts]]
    pts += [(s, r) for _, _, _, ts in series for s, r in [(x["step"], x["arms"]["TA+F"]["R2"]) for x in ts]]
    if not pts: return "<div class='card empty'>No TA checkpoint evaluations yet.</div>"
    xs = [p[0] for p in pts] or [0]; ys = [p[1] for p in pts] + [v for v in (refF, refB, 0.0) if v is not None]
    x0, x1 = 0, max(max(xs), 1); y0, y1 = min(ys) - 0.02, max(ys) + 0.02
    X = lambda v: 60 + (w - 80) * (v - x0) / (x1 - x0); Y = lambda v: h - 30 - (h - 50) * (v - y0) / (y1 - y0)
    out = [f"<svg width='{w}' height='{h}' style='background:#0b1220'>"]
    for v, lab, col in ((refF, "foundation (rep0)", "#a3e635"), (refB, "tabpfn35 (rep0)", "#f472b6"), (0.0, "0", "#475569")):
        if v is None: continue
        out.append(f"<line x1='60' x2='{w-20}' y1='{Y(v):.1f}' y2='{Y(v):.1f}' stroke='{col}' stroke-dasharray='5,4'/><text x='{w-22}' y='{Y(v)-4:.1f}' fill='{col}' font-size='11' text-anchor='end'>{lab} {v:.3f}</text>")
    for _, name, col, ts in series:
        for arm, dash in (("TA", ""), ("TA+F", "6,3"), ("TA_yshuf", "2,3")):
            P = [(X(r["step"]), Y(r["arms"][arm]["R2"])) for r in ts if arm in r["arms"]]
            if not P: continue
            out.append(f"<polyline fill='none' stroke='{col}' stroke-width='2' stroke-dasharray='{dash}' points='{' '.join(f'{a:.1f},{b:.1f}' for a, b in P)}'/>")
            out += [f"<circle cx='{a:.1f}' cy='{b:.1f}' r='3' fill='{col}'/>" for a, b in P]
            a, b = P[-1]; out.append(f"<text x='{a+5:.1f}' y='{b-5:.1f}' fill='{col}' font-size='11'>{name} {arm}</text>")
    out.append(f"<text x='{w/2}' y='{h-6}' fill='#cbd5e1' font-size='11' text-anchor='middle'>training step</text>")
    out.append(f"<text x='14' y='{h/2}' fill='#cbd5e1' font-size='11' transform='rotate(-90 14 {h/2})' text-anchor='middle'>OOF R² vs GP HWM (ridge, rep 0)</text>")
    for t in range(5):
        v = y0 + (y1 - y0) * t / 4; out.append(f"<text x='54' y='{Y(v)+4:.1f}' fill='#94a3b8' font-size='10' text-anchor='end'>{v:.2f}</text>")
    out.append("</svg>"); return "".join(out)


def section():
    series = [(k, n, c, *_load(k)) for k, n, c in MODELS]
    refF, refB = _refs([s[3] for s in series])
    svg = _svg([(k, n, c, ts) for k, n, c, ts, _, _ in series], refF, refB)
    f = lambda v, d=3: "—" if v is None else f"{v:+.{d}f}"
    ci = lambda c: "—" if not c else f"[{c[0]:+.3f}, {c[1]:+.3f}]"
    rows, prow = [], []
    for k, name, col, ts, prog, run in series:
        if ts:
            r = ts[-1]; A = r["arms"]
            for arm in ("TA", "TA+F", "TA_yshuf", "SHUF:TA"):
                if arm not in A: continue
                x = A[arm]
                rows.append(f"<tr><td style='color:{col}'>{name}</td><td>{arm}</td><td>{r['step']}</td><td>{r['epoch']:.2f}</td><td>{x['R2']:.4f}</td><td>{x['spearman']:.3f}</td>"
                            f"<td>{f(x.get('d_vs_foundation'))}</td><td>{ci(x.get('ci_vs_foundation'))}</td><td>{f(x.get('d_vs_tabpfn35'))}</td><td>{ci(x.get('ci_vs_tabpfn35'))}</td>"
                            f"<td>{'—' if r.get('ssl_val_mse') is None else round(r['ssl_val_mse'], 4)}</td><td>{_mt(r['utc'])}</td></tr>")
        if prog:
            eta = dt.datetime.now(MT) + dt.timedelta(seconds=prog.get("eta_s", 0))
            prow.append(f"<tr><td style='color:{col}'>{name}</td><td>{html.escape(prog.get('host',''))} · {html.escape(prog.get('gpu',''))}</td><td>{prog['step']:,}/{prog['total']:,} ({100*prog['step']/max(1,prog['total']):.1f}%)</td>"
                        f"<td>{prog['epoch']:.2f}</td><td>{prog['ctx_per_s']:.0f} ctx/s · {prog['epoch_min']:.1f} min/epoch</td><td>{prog['mse_ema']:.4f}</td><td>{eta.strftime('%-I:%M %p MT')}</td><td>{len(ts)}</td><td>{_mt(prog['utc'])}</td></tr>")
        elif run:
            prow.append(f"<tr><td style='color:{col}'>{name}</td><td colspan=8>started {_mt(run.get('started_utc',''))}; waiting for first progress line</td></tr>")
    return f"""<h2>TA training · target-aware encoders (TA_Hyperion_v8, TA_Dope_V2, v8b-raw, v8b-ft)</h2>
<div class='card'><table><thead><tr><th>Model</th><th>Host · GPU</th><th>Step</th><th>Epoch</th><th>Throughput</th><th>Masked-y MSE (EMA)</th><th>ETA (planned steps)</th><th>Evals</th><th>Updated</th></tr></thead><tbody>{''.join(prow) or "<tr><td colspan=9 class='empty'>No TA runs yet.</td></tr>"}</tbody></table></div>
<div class='card'>{svg}<div style='font-size:12px;color:#94a3b8'>Solid = TA embedding alone; dashed = TA + foundation; dotted = TA encoded with y shuffled inside each context (control: X→y signal removed). Reference lines are foundation and tabpfn35 scored in the same rep-0 folds (the 3-repeat foundation figure is {FOUNDATION_3REP}).</div></div>
<div class='card'><table><thead><tr><th>Model</th><th>Arm</th><th>Step</th><th>Epoch</th><th>R²</th><th>Spearman</th><th>ΔR² vs foundation</th><th>95% CI</th><th>ΔR² vs tabpfn35</th><th>95% CI</th><th>SSL val MSE</th><th>Eval time</th></tr></thead><tbody>{''.join(rows) or "<tr><td colspan=12 class='empty'>No checkpoint evaluations yet.</td></tr>"}</tbody></table>
<div style='font-size:12px;color:#94a3b8'>SHUF:TA = GP-HWM labels permuted inside each outer-train set (leak control; should sit near or below 0). Self-supervised training only: no GP-HWM label is used to train either model. Probe = fast-sweep ridge, 2,689 contexts, family-grouped 5-fold (rep 0), family bootstrap CIs.</div></div>"""


def png(path):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    series = [(k, n, c, *_load(k)) for k, n, c in MODELS]
    refF, refB = _refs([s[3] for s in series])
    fig, ax = plt.subplots(figsize=(10, 4.5), facecolor="#0b1220"); ax.set_facecolor("#0b1220")
    for k, name, col, ts, _, _ in series:
        for arm, ls in (("TA", "-"), ("TA+F", "--"), ("TA_yshuf", ":")):
            P = [(r["step"], r["arms"][arm]["R2"]) for r in ts if arm in r["arms"]]
            if P: ax.plot(*zip(*P), ls, color=col, marker="o", ms=3, label=f"{name} {arm}")
    if refF is not None: ax.axhline(refF, color="#a3e635", ls="--", lw=1, label=f"foundation {refF:.3f}")
    if refB is not None: ax.axhline(refB, color="#f472b6", ls="--", lw=1, label=f"tabpfn35 {refB:.3f}")
    ax.axhline(0, color="#475569", lw=0.8)
    ax.tick_params(colors="#cbd5e1"); ax.set_xlabel("training step", color="#cbd5e1"); ax.set_ylabel("OOF R² vs GP HWM", color="#cbd5e1")
    ax.set_title(f"TA training · probe R² over steps · {dt.datetime.now(MT).strftime('%b %-d %-I:%M %p MT')}", color="w")
    ax.legend(fontsize=7, facecolor="#0b1220", labelcolor="#cbd5e1", ncol=2); fig.tight_layout(); fig.savefig(path, dpi=130, facecolor=fig.get_facecolor())
