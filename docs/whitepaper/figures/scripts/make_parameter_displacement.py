#!/usr/bin/env python3
"""Render the recorded candidate L2 displacement from a fixed summary snapshot."""
from datetime import datetime
from decimal import Decimal
import csv
import hashlib
import io
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
from matplotlib import font_manager, pyplot as plt
from matplotlib.ticker import MaxNLocator

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/parameter_displacement.csv'
SOURCE = DATA.with_suffix('.source.json')
OUTPUT = ROOT / 'fig_parameter_displacement.pdf'
PROVENANCE = DATA.with_suffix('.provenance.json')


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    raw = DATA.read_bytes()
    source = json.loads(SOURCE.read_text())
    assert sha(raw) == source['csv_sha256']
    rows = list(csv.DictReader(io.StringIO(raw.decode())))
    assert len(rows) == source['records'] and rows
    rounds = [int(r['round']) for r in rows]
    assert rounds == list(range(rounds[0], rounds[-1] + 1))
    assert [rounds[0], rounds[-1]] == source['round_window']
    a, b = source['source_line_window']
    assert [int(r['source_line']) for r in rows] == list(range(a, b + 1))
    assert all(len(r['source_record_sha256']) == 64 for r in rows)
    values = [Decimal(r['parameter_displacement_l2']) for r in rows]
    assert all(v.is_finite() and v >= 0 for v in values)
    assert min(values) == Decimal(source['minimum_displacement_l2'])
    assert max(values) == Decimal(source['maximum_displacement_l2'])
    accepted = sum(int(r['adapter_accepted']) for r in rows)
    assert accepted == source['accepted'] == 0
    font = Path(font_manager.findfont('Liberation Serif', fallback_to_default=False))
    plt.rcParams.update({'font.family': 'Liberation Serif', 'font.size': 9,
                         'axes.labelsize': 9, 'xtick.labelsize': 8.5, 'ytick.labelsize': 8.5,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42, 'ps.fonttype': 42})
    fig, ax = plt.subplots(figsize=(3.375, 1.65))
    fig.subplots_adjust(left=.23, right=.96, bottom=.30, top=.96)
    ax.scatter(rounds, [float(v) for v in values], s=9, color='#17649a', alpha=.85,
               edgecolors='none', zorder=3)
    ax.set(xlim=(rounds[0] - 1, rounds[-1] + 1), ylim=(0, float(max(values)) * 1.12),
           xlabel='Completed round', ylabel='Candidate L2 distance')
    ax.xaxis.set_major_locator(MaxNLocator(4, integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.ticklabel_format(style='plain', useOffset=False)
    ax.grid(axis='y', color='.9', linewidth=.4)
    ax.set_axisbelow(True)
    fixed = datetime.fromisoformat(source['captured_utc'])
    fig.savefig(OUTPUT, metadata={'Title': 'Recorded candidate parameter displacement', 'Author': '',
                                 'Subject': 'Candidate parameter-space displacement includes perturbation and optimization; not retained learning',
                                 'CreationDate': fixed, 'ModDate': fixed})
    plt.close(fig)
    provenance = {'schema': 'whitepaper.parameter-displacement-figure/v1',
                  'csv_sha256': sha(raw), 'source_receipt_sha256': sha(SOURCE.read_bytes()),
                  'source_prefix_sha256': source['source_prefix_sha256'],
                  'generator_sha256': sha(Path(__file__).read_bytes()), 'figure_sha256': sha(OUTPUT.read_bytes()),
                  'records_and_points': len(rows), 'round_window': source['round_window'],
                  'accepted': accepted, 'minimum_displacement_l2': str(min(values)),
                  'maximum_displacement_l2': str(max(values)), 'transformation': source['transformation'],
                  'scope': source['scope'], 'font_family': 'Liberation Serif',
                  'font_file_sha256': sha(font.read_bytes()), 'matplotlib_version': matplotlib.__version__,
                  'figure_inches': [3.375, 1.65], 'minimum_text_size_pt': 8.5,
                  'fixed_canvas_no_tight_crop': True, 'reproduce': 'python3 figures/scripts/make_parameter_displacement.py'}
    PROVENANCE.write_text(json.dumps(provenance, indent=2) + '\n')
    print(json.dumps({'records': len(rows), 'accepted': accepted, 'figure_sha256': provenance['figure_sha256']}))


if __name__ == '__main__':
    main()
