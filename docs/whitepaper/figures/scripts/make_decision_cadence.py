#!/usr/bin/env python3
"""Plot every adjacent timestamp gap in a fixed recent decision-journal snapshot."""
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
DATA = ROOT / 'data/decision_cadence.csv'
SOURCE = DATA.with_suffix('.source.json')
OUTPUT = ROOT / 'fig_decision_cadence.pdf'
PROVENANCE = DATA.with_suffix('.provenance.json')


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    raw = DATA.read_bytes()
    source = json.loads(SOURCE.read_text())
    assert sha(raw) == source['csv_sha256']
    rows = list(csv.DictReader(io.StringIO(raw.decode())))
    assert len(rows) == source['records'] == 201
    rounds = [int(r['round']) for r in rows]
    assert rounds == list(range(rounds[0], rounds[-1] + 1))
    assert [rounds[0], rounds[-1]] == source['round_window']
    start, end = source['source_line_window']
    assert [int(r['source_line']) for r in rows] == list(range(start, end + 1))
    assert all(len(r['source_record_sha256']) == 64 for r in rows)
    times = [Decimal(r['timestamp_seconds']) for r in rows]
    gaps = [b - a for a, b in zip(times, times[1:])]
    assert len(gaps) == source['intervals'] == 200 and all(t > 0 for t in gaps)
    assert sum(gaps) == times[-1] - times[0] == Decimal(source['total_elapsed_seconds'])
    assert min(gaps) == Decimal(source['minimum_gap_seconds'])
    assert max(gaps) == Decimal(source['maximum_gap_seconds'])
    font = Path(font_manager.findfont('Liberation Serif', fallback_to_default=False))
    plt.rcParams.update({'font.family': 'Liberation Serif', 'font.size': 9,
                         'axes.labelsize': 9, 'xtick.labelsize': 8.5, 'ytick.labelsize': 8.5,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42, 'ps.fonttype': 42})
    fig, ax = plt.subplots(figsize=(3.375, 1.6))
    fig.subplots_adjust(left=.23, right=.96, bottom=.31, top=.96)
    ax.scatter(rounds[1:], [float(t) for t in gaps], s=5, color='#17649a', alpha=.8, edgecolors='none', zorder=3)
    ax.set(xlim=(rounds[1] - 3, rounds[-1] + 3), ylim=(0, float(max(gaps)) * 1.12),
           xlabel='Ending decision round', ylabel='Timestamp gap (s)')
    ax.xaxis.set_major_locator(MaxNLocator(4, integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.ticklabel_format(style='plain', useOffset=False)
    ax.grid(axis='y', color='.9', linewidth=.4)
    ax.set_axisbelow(True)
    fixed = datetime.fromisoformat(source['captured_utc'])
    fig.savefig(OUTPUT, metadata={'Title': 'Recorded adapter-decision timestamp gaps', 'Author': '',
                                 'Subject': 'Calendar-time cadence including work and waits, not compute latency or learning evidence',
                                 'CreationDate': fixed, 'ModDate': fixed})
    plt.close(fig)
    provenance = {'schema': 'whitepaper.decision-cadence-figure/v1',
                  'csv_sha256': sha(raw), 'source_receipt_sha256': sha(SOURCE.read_bytes()),
                  'source_prefix_sha256': source['source_prefix_sha256'],
                  'generator_sha256': sha(Path(__file__).read_bytes()), 'figure_sha256': sha(OUTPUT.read_bytes()),
                  'source_records': 201, 'plotted_intervals': 200,
                  'record_round_window': source['round_window'], 'point_round_window': source['point_round_window'],
                  'minimum_gap_seconds': str(min(gaps)), 'maximum_gap_seconds': str(max(gaps)),
                  'total_elapsed_seconds': str(sum(gaps)), 'transformation': source['transformation'], 'scope': source['scope'],
                  'font_family': 'Liberation Serif', 'font_file_sha256': sha(font.read_bytes()),
                  'matplotlib_version': matplotlib.__version__, 'figure_inches': [3.375, 1.6],
                  'minimum_text_size_pt': 8.5, 'fixed_canvas_no_tight_crop': True,
                  'reproduce': 'python3 figures/scripts/make_decision_cadence.py'}
    PROVENANCE.write_text(json.dumps(provenance, indent=2) + '\n')
    print(json.dumps({'records': len(rows), 'intervals': len(gaps), 'figure_sha256': provenance['figure_sha256']}))


if __name__ == '__main__':
    main()
