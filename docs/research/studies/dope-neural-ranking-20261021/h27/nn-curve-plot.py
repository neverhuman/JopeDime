#!/usr/bin/env python3
"""Render every fixed400 TRAIN/internal-monitor curve from the public JSON.
Standard library only. Input is already sanitized aggregate metadata.
Usage: python nn-curve-plot.py nn-fixed400-training-curves.json --output-dir .
"""
import argparse
from collections import defaultdict
from html import escape
import json
import math
from pathlib import Path

def render(data, metric, label):
    by = defaultdict(list)
    for row in data['observations']:
        by[row['run_id']].append(row)
    assert len(by) == 36 and len(data['observations']) == 180
    for rows in by.values():
        assert sorted(r['step'] for r in rows) == [0, 100, 200, 300, 400]
        assert all(not r['OOF_evaluated'] and not r['checkpoint_selected'] for r in rows)
    vals = [r[p][metric] for rows in by.values() for r in rows for p in ['train_panel', 'internal_monitor']]
    assert all(math.isfinite(v) for v in vals)
    lo, hi = min(vals), max(vals)
    pad = max((hi-lo)*.04, .01)
    lo, hi = min(0, lo-pad), hi+pad
    W, H, panel_w, panel_h = 1560, 1310, 256, 194
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img">',
        '<title>All 36 fixed400 TRAIN and internal-monitor curves</title>',
        '<desc>Every model is included. Five actual snapshots per model. Shared vertical scale across panels. These are TRAIN and internal monitor, not OOF learning curves.</desc>',
        '<style>text{font-family:system-ui,sans-serif;fill:#17392f} .tick{font-size:10px;fill:#60736a}.title{font-size:11px;font-weight:600}.grid{stroke:#dce5df;stroke-width:1}</style>',
        '<rect width="100%" height="100%" fill="#f5f8f5"/>',
        f'<text x="20" y="28" font-size="22" font-weight="700">{escape(label)} · all 36 models</text>',
        '<text x="20" y="52" font-size="12">Five actual snapshots: 0 / 100 / 200 / 300 / 400 updates. Fixed endpoint; no checkpoint selection.</text>',
        '<text x="20" y="73" font-size="12" fill="#087f5b">Green: fixed TRAIN panel</text>',
        '<text x="270" y="73" font-size="12" fill="#7852b3">Purple: internal monitor (excluded from gradients, scalers and selection)</text>',
        '<text x="20" y="94" font-size="11">Within-model comparisons only: objective scales/panels differ. These are exposed-TRAIN internal monitors; final OOF curves are not fabricated.</text>']
    for i, rid in enumerate(sorted(by)):
        rows=sorted(by[rid], key=lambda r:r['step'])
        x0=14+(i%6)*panel_w; y0=113+(i//6)*panel_h
        x,y,w,h=x0+38,y0+25,202,137
        title=rid.replace('real112','real state').replace('zero112','zero state').replace('zero8','L0').replace('one8','L1').replace('five8','L5')
        out.append(f'<text class="title" x="{x0+8}" y="{y0+13}">{escape(title)}</text>')
        for frac in [0, .5, 1]:
            yy=y+h*(1-frac); v=lo+(hi-lo)*frac
            out.append(f'<line class="grid" x1="{x}" y1="{yy:.2f}" x2="{x+w}" y2="{yy:.2f}"/>')
            out.append(f'<text class="tick" text-anchor="end" x="{x-4}" y="{yy+3:.2f}">{v:.2g}</text>')
        for step in [0,200,400]:
            xx=x+w*step/400
            out.append(f'<text class="tick" text-anchor="middle" x="{xx:.2f}" y="{y+h+15}">{step}</text>')
        for panel,color in [('train_panel','#087f5b'),('internal_monitor','#7852b3')]:
            coords=[(x+w*r['step']/400, y+h*(1-(r[panel][metric]-lo)/(hi-lo))) for r in rows]
            points=' '.join(f'{a:.3f},{b:.3f}' for a,b in coords)
            out.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="1.7"/>')
            for a,b in coords:out.append(f'<circle cx="{a:.3f}" cy="{b:.3f}" r="2.0" fill="{color}"/>')
    out.append('<text x="20" y="1294" font-size="12">Lower is better. Panel metrics are component-equal aggregates. Loss decline is not evidence of landmark or Dope-state ranking lift.</text></svg>')
    return '\n'.join(out)+'\n'

def main():
    ap=argparse.ArgumentParser();ap.add_argument('curves',type=Path);ap.add_argument('--output-dir',type=Path,default=Path('.'));a=ap.parse_args()
    d=json.loads(a.curves.read_text());assert d['all_models_included'] is True
    for metric,label,name in [('objective','Composite objective','nn-all36-objective-curves.svg'),('raw_selected_fitness','Raw selected GP fitness','nn-all36-raw-fitness-curves.svg')]:
        p=a.output_dir/name
        with p.open('x') as f:f.write(render(d,metric,label))
if __name__=='__main__':main()
