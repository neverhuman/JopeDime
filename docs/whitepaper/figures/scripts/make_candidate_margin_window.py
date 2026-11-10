#!/usr/bin/env python3
"""Render a fixed, source-bound window of recorded candidate utility margins."""
from datetime import datetime, timezone
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
from matplotlib import font_manager, pyplot as plt
from matplotlib.ticker import MaxNLocator

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/candidate_margin_window.csv'
SOURCE = DATA.with_suffix('.source.json')
OUTPUT = ROOT / 'fig_candidate_margin_window.pdf'
PROVENANCE = DATA.with_suffix('.provenance.json')


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    raw = DATA.read_bytes()
    source = json.loads(SOURCE.read_text())
    assert sha(raw) == source['csv_sha256']
    rows = list(csv.DictReader(io.StringIO(raw.decode())))
    rounds = [int(r['round']) for r in rows]
    assert rounds == list(range(4351, 4551)) and len(rows) == source['rows'] == 200
    start_line, end_line = source['source_line_window']
    assert [int(r['source_line']) for r in rows] == list(range(start_line, end_line + 1))
    margins = []
    for r in rows:
        candidate, previous, retained = (float(r[k]) for k in ['candidate_utility', 'previous_best_utility', 'best_utility'])
        margin = float(r['candidate_margin'])
        assert all(math.isfinite(x) for x in [candidate, previous, retained, margin])
        assert margin == candidate - previous
        assert len(r['source_record_sha256']) == len(r['evaluation_identity_sha256']) == 64
        assert r['evaluation_identity_sha256'] in source['evaluation_identity_sha256']
        accepted = int(r['accepted'])
        assert retained == (candidate if accepted else previous)
        margins.append(margin)
    assert sum(int(r['accepted']) for r in rows) == source['accepted'] == 0
    assert min(margins) == source['minimum_signed_margin'] and max(margins) == source['maximum_signed_margin']
    assert max(margins) < 0
    font = Path(font_manager.findfont('Liberation Serif', fallback_to_default=False))
    plt.rcParams.update({'font.family': 'Liberation Serif', 'font.size': 9,
                         'axes.labelsize': 9, 'xtick.labelsize': 8.5, 'ytick.labelsize': 8.5,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42, 'ps.fonttype': 42})
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 1.6), gridspec_kw={'width_ratios': [1.7, 1]})
    fig.subplots_adjust(left=.12, right=.97, bottom=.35, top=.82, wspace=.37)
    lower, upper = min(margins) * 1.10, abs(min(margins)) * .04
    ax = axes[0]
    ax.scatter(rounds, margins, s=5, color='#17649a', alpha=.75, edgecolors='none', zorder=3)
    ax.axhline(0, color='.4', linestyle='--', linewidth=.65)
    ax.set(xlim=(4347, 4554), ylim=(lower, upper), xlabel='Logged round', ylabel='Candidate margin\n(utility difference)')
    ax.set_title('Margins by logged round', loc='left', fontsize=9, pad=3)
    ax.xaxis.set_major_locator(MaxNLocator(5, integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax = axes[1]
    ordered = sorted(margins)
    x = [lower, *ordered, upper]
    y = [0, *[(i + 1) / len(rows) for i in range(len(rows))], 1]
    ax.step(x, y, where='post', color='#17649a', linewidth=1.0)
    ax.set(xlim=(lower, upper), ylim=(-.02, 1.04), xlabel='Candidate margin\n(utility difference)', ylabel='Share of decisions')
    ax.set_title('Empirical distribution', loc='left', fontsize=9, pad=3)
    ax.xaxis.set_major_locator(MaxNLocator(3))
    ax.set_yticks([0, .5, 1])
    for ax in axes:
        ax.ticklabel_format(style='plain', useOffset=False, axis='x')
        ax.grid(axis='y', color='.9', linewidth=.4)
        ax.set_axisbelow(True)
    fixed = datetime(2026, 11, 3, tzinfo=timezone.utc)
    fig.savefig(OUTPUT, metadata={'Title': 'Candidate utility margins in a fixed decision window', 'Author': '',
                                 'Subject': 'Adaptive development-pool observations, not independent learning evidence',
                                 'CreationDate': fixed, 'ModDate': fixed})
    plt.close(fig)
    provenance = {'schema': 'whitepaper.candidate-margin-window-figure/v1',
                  'csv_sha256': sha(raw), 'source_receipt_sha256': sha(SOURCE.read_bytes()),
                  'source_prefix_sha256': source['source_prefix_sha256'],
                  'generator_sha256': sha(Path(__file__).read_bytes()), 'figure_sha256': sha(OUTPUT.read_bytes()),
                  'round_window': source['round_window'], 'plotted_decisions': len(rows), 'accepted': source['accepted'],
                  'margin_min': min(margins), 'margin_max': max(margins),
                  'transformation': source['transformation'], 'scope': source['scope'],
                  'matplotlib_version': matplotlib.__version__, 'font_family': 'Liberation Serif',
                  'font_file_sha256': sha(font.read_bytes()), 'figure_inches': [7.0, 1.6],
                  'minimum_text_size_pt': 8.5, 'fixed_canvas_no_tight_crop': True,
                  'reproduce': 'python3 figures/scripts/make_candidate_margin_window.py'}
    PROVENANCE.write_text(json.dumps(provenance, indent=2) + '\n')
    print(json.dumps({'rows': len(rows), 'accepted': source['accepted'], 'figure_sha256': provenance['figure_sha256']}))


if __name__ == '__main__':
    main()
