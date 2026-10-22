#!/usr/bin/env python3
"""Render deterministic publication figures from source-bound data.

Requires matplotlib. Neither figure is a predictive benchmark or a proposed
performance curve. The first is a protocol diagram; the second re-plots receipts.
"""
from pathlib import Path
import argparse
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

PAPER = Path(__file__).resolve().parents[1]
INK, CYAN, AMBER, GREY = '#173242', '#087F8C', '#B27624', '#56616A'
PALE, WHITE = '#F2F6F7', '#FFFFFF'
plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 8,
    'text.color': INK, 'axes.labelcolor': INK, 'xtick.color': GREY,
    'ytick.color': INK, 'axes.edgecolor': '#D2DADD',
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
    'svg.hashsalt': 'innovationzero-publication-v1',
    'savefig.facecolor': 'white',
})


def save(fig, name, destination):
    destination.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination / f'{name}.pdf', metadata={
        'Title': f'InnovationZero: {name}',
        'Creator': 'InnovationZero publication figure builder',
        'CreationDate': None, 'ModDate': None,
    })
    fig.savefig(destination / f'{name}.svg', metadata={
        'Title': f'InnovationZero: {name}', 'Date': None,
        'Creator': 'InnovationZero publication figure builder',
    })
    # Matplotlib emits spaces before path-data newlines. Removing that XML
    # whitespace keeps generated figures clean in diffs without changing geometry.
    svg = destination / f'{name}.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines()) + '\n')
    plt.close(fig)


def commitment(destination):
    fig = plt.figure(figsize=(7.10, 1.94))
    ax = fig.add_axes([0.005, 0.015, 0.99, 0.97])
    ax.set_xlim(-.5, 100.5); ax.set_ylim(0, 30); ax.axis('off')
    ax.text(0, 28.1, 'PAST TASKS', weight='bold', fontsize=7.7, color=AMBER)
    ax.text(16.3, 28.1, 'Propose', fontsize=8)
    ax.text(31.5, 28.1, 'Execute + measure', fontsize=8)
    ax.text(60.7, 28.1, 'Learn from outcomes', fontsize=8)
    for x0,x1 in [(27,30),(54.7,59)]:
        ax.add_patch(FancyArrowPatch((x0,28.7),(x1,28.7),arrowstyle='-|>',mutation_scale=9,color=AMBER,lw=.9))
    ax.text(84.4, 27.0, 'TRAINING\nFEEDBACK', fontsize=6.9, color=AMBER, linespacing=1.2)
    ax.plot([0,100],[23,23],color='#D2DADD',lw=.65)
    ax.text(0, 19.7, 'NEW TASK', weight='bold', fontsize=7.7, color=CYAN)
    labels=[('Training rows','permitted data',0,17),('Frozen generator','one proposal',21,20),('Source commitment','immutable identity',45,22),('Compile + fit','no repair',71,14),('Test','independent',89,11)]
    for title,sub,x,w in labels:
        face=CYAN if x==45 else PALE
        edge=CYAN if x==45 else '#D2DADD'
        ax.add_patch(FancyBboxPatch((x,6.9),w,9.4,boxstyle='round,pad=0.12,rounding_size=.8',linewidth=.7,edgecolor=edge,facecolor=face))
        ax.text(x+w/2,12.7,title,ha='center',va='center',fontsize=7.5 if x!=71 else 7.0,weight='bold',color=WHITE if x==45 else INK)
        ax.text(x+w/2,9.5,sub,ha='center',va='center',fontsize=6.6,color=WHITE if x==45 else GREY)
    for x0,x1 in [(17.5,20.4),(41.5,44.4),(67.5,70.4),(85.5,88.4)]:
        ax.add_patch(FancyArrowPatch((x0,11.6),(x1,11.6),arrowstyle='-|>',mutation_scale=9,color=INK,lw=.9))
    ax.plot([69,69],[3.2,21.3],color=CYAN,lw=.9,linestyle=(0,(2.2,2.2)))
    ax.text(43.1,2.1,'ZERO candidate experiments before commitment',ha='center',fontsize=7.2,color=CYAN,weight='bold')
    ax.text(84.9,2.1,'Evidence comes afterward',ha='center',fontsize=7.1,color=GREY)
    # Learned checkpoint is used at deployment; no post-commit return path exists.
    ax.add_patch(FancyArrowPatch((70.7,25.4),(32,17.0),connectionstyle='angle,angleA=180,angleB=90,rad=2',arrowstyle='-|>',mutation_scale=8,color=AMBER,lw=.8))
    save(fig,'commitment',destination)


def elo(destination):
    data=json.loads((PAPER/'evidence/elo-summary.json').read_text())
    rows=data['ratings']
    labels={'Opus5p5_Extra':'Opus artifact','Gemini4_argon_high':'Gemini artifact','GPT6p1_Sol_Max':'Sol artifact','GPT6_Astra_Max':'Astra artifact'}
    fig=plt.figure(figsize=(3.43,2.60))
    ax=fig.add_axes([.33,.18,.62,.60])
    fig.text(.03,.94,'EXPLORATORY',fontsize=7.4,weight='bold',color=AMBER)
    fig.text(.03,.865,'64 training families · 3 folds × 3 seeds',fontsize=7.0,color=GREY)
    for i,row in enumerate(rows):
        y=3-i; value=row['rating']; low,high=row['ci95']
        ax.errorbar(value,y,xerr=[[value-low],[high-value]],fmt='o',ms=4.4,color=CYAN,ecolor=CYAN,elinewidth=1.5,capsize=2.5,capthick=1)
        ax.annotate(f'{value:.1f}',(value,y),xytext=(0,9),textcoords='offset points',ha='center',fontsize=7.2,color=INK)
    ax.set_yticks([3,2,1,0],[labels[r['artifact']] for r in rows],fontsize=7.2)
    ax.set_ylim(-.45,3.5);ax.set_xlim(1250,1750)
    ax.set_xticks([1300,1500,1700]);ax.tick_params(axis='x',labelsize=7,length=2.5);ax.tick_params(axis='y',length=0,pad=7)
    ax.set_xlabel('Elo · 95% family-bootstrap interval',fontsize=7.2,labelpad=6)
    ax.xaxis.grid(True,linewidth=.5,color='#E1E7E9');ax.set_axisbelow(True)
    for side in ['top','right','left']:ax.spines[side].set_visible(False)
    # Qualification is stated prominently in the manuscript caption; avoid
    # colliding with the axis title at single-column print size.
    save(fig,'elo',destination)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=PAPER/'figures')
    args=parser.parse_args()
    commitment(args.out);elo(args.out)
    print(f'Rendered commitment and exploratory Elo in {args.out}')
