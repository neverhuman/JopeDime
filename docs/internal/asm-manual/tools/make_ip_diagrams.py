#!/usr/bin/env python3
"""Deterministic vector diagrams of source responsibilities, not runtime qualification."""
from html import escape
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / 'figures'


def start(width, height, title):
    return [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img"><title>{escape(title)}</title>',
            '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0L10 5L0 10Z" fill="#31536c"/></marker></defs>',
            f'<rect width="{width}" height="{height}" fill="#fff"/>',
            '<g font-family="DejaVu Sans, sans-serif" fill="#183448">']


def text(out, x, y, lines, size=20, weight='normal', color='#183448'):
    for i, line in enumerate(lines):
        out.append(f'<text x="{x}" y="{y + i * (size + 8)}" font-size="{size}" font-weight="{weight}" fill="{color}">{escape(line)}</text>')


def box(out, x, y, width, height, title, lines, fill='#eef5f8'):
    out.append(f'<rect x="{x}" y="{y}" width="{width}" height="{height}" rx="12" fill="{fill}" stroke="#93aab8"/>')
    text(out, x + 18, y + 32, [title], 22, 'bold')
    text(out, x + 18, y + 66, lines, 18)


def arrow(out, path, dashed=False):
    dash = ' stroke-dasharray="8 6"' if dashed else ''
    out.append(f'<path d="{path}" fill="none" stroke="#31536c" stroke-width="2.5" marker-end="url(#arrow)"{dash}/>')


def save(name, out):
    out.append('</g></svg>')
    path = FIG / name
    path.write_text('\n'.join(out) + '\n')
    return {'path': name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def pipeline():
    out = start(1240, 1000, 'ASM pipeline and prohibited scoring feedback')
    text(out, 35, 40, ['Data → description → recipes → real fits → eligible teachers → neural updates'], 24, 'bold')
    box(out, 35, 80, 355, 135, 'Frozen data and groups', ['Original labels preserved', 'Train / audit / outer identities'])
    box(out, 445, 80, 350, 135, 'Training-only context', ['Hyperion + Foundation + Target', '2048 values; assets pinned'])
    box(out, 850, 80, 350, 135, 'Policy proposals', ['Causal, grammar-masked decoder', 'Learned reward ranks candidates'])
    arrow(out, 'M390 148H445'); arrow(out, 'M795 148H850')
    box(out, 850, 280, 350, 150, 'Bounded GP search', ['Mutate / cross over recipes', 'Fit candidates on allowed rows', 'Inner selection; final refit'])
    arrow(out, 'M1025 215V280')
    box(out, 445, 280, 350, 150, 'Fitted predictions', ['Saved preprocessing and weights', 'Measured losses, not reward scores', 'Each search has its own recipe'])
    arrow(out, 'M850 355H795')
    box(out, 35, 280, 355, 150, 'Training-side teacher guard', ['Fixed-program OOF comparison', 'Baseline + complete identities', 'Eligible rows only'])
    arrow(out, 'M445 355H390')
    box(out, 35, 495, 355, 145, 'Neural optimizer updates', ['Imitation + ranking objectives', 'AdamW; changed parameters', 'Checkpoint / optimizer lineage'])
    arrow(out, 'M210 430V495')
    box(out, 445, 495, 350, 145, 'Parent → challenger test', ['Another measured search', 'Same protocol and resource limits', 'No automatic serving promotion'])
    arrow(out, 'M390 565H445')
    box(out, 850, 495, 350, 145, 'Emitted-code qualification', ['Allowlisted fitted inference', 'Prediction parity by backend', 'Generation is not qualification'])
    arrow(out, 'M1025 430V495')
    box(out, 35, 715, 525, 165, 'Separate nested procedure audit', ['5 outer × (5 audit searches + 1 outer search)', 'Repeat complete selection on each permitted partition', 'Commit every prediction / failure before scoring'], '#f4f0e7')
    box(out, 625, 715, 575, 165, 'Frozen-round report', ['Parent contrast: min(audit gain, outer gain) per unit', 'Dataset means → family means → equal-family mean', 'Current KPI: NOT YET MEASURED'], '#f4f0e7')
    arrow(out, 'M560 795H625')
    arrow(out, 'M620 640V680H300V715', True)
    out.append('<path d="M1200 795H1220V355H1200" fill="none" stroke="#a43c36" stroke-width="2.5" stroke-dasharray="7 5"/>')
    out.append('<path d="M1192 345L1208 365M1208 345L1192 365" stroke="#a43c36" stroke-width="3"/>')
    text(out, 850, 680, ['No report → unfinished search'], 16, 'bold', '#a43c36')
    text(out, 35, 690, ['Outer labels: scorer only'], 16, 'bold', '#a43c36')
    out.append('<path d="M30 680H15V260H620V215" fill="none" stroke="#a43c36" stroke-width="2.5" stroke-dasharray="7 5"/>')
    out.append('<path d="M610 212L630 230M630 212L610 230" stroke="#a43c36" stroke-width="3"/>')
    text(out, 35, 923, ['PROHIBITED: outer scoring labels → context / selection / teachers / optimizer.',
                        'PROHIBITED: released scores → decisions in unfinished folds of the same round.'], 20, 'bold', '#a43c36')
    out.append('<path d="M25 898H1210" stroke="#a43c36" stroke-width="3" stroke-dasharray="8 6"/>')
    return save('ip-pipeline.svg', out)


def network():
    out = start(1240, 1080, 'Policy context, causal attention, reward pooling and objectives')
    text(out, 35, 42, ['Two model lineages; a shared diagram does not imply interchangeable weights'], 24, 'bold')
    box(out, 35, 80, 365, 165, 'Permitted training rows', ['Hyperion: row128 → pooled512', 'Foundation: 2048 → projected1024', 'Target: 512', 'Concat / normalize / clip → 2048'])
    box(out, 450, 80, 355, 165, 'Separate context MLPs', ['Separate decoder z / reward rz', 'Native: 2048 → 512 → 512', 'Python: bound input → 256 → 256', 'GELU + LayerNorm per layer'])
    arrow(out, 'M400 165H450')
    box(out, 855, 80, 345, 165, 'Architecture bindings', ['Accepted native: 512 × 6', '4 heads; FFN 2048; vocab 401', 'Experimental Python: 256 × 2', '4 heads; FFN 1024; bound vocab'])
    box(out, 35, 315, 565, 205, 'Autoregressive program decoder', ['[context token, token embeddings] + positions', 'Pre-LN causal self-attention; padding-key mask', 'Residual attention + feed-forward layers', 'Final norm → tied embedding projection → logits', 'Grammar mask → legal next-token distribution'])
    arrow(out, 'M530 245V280H315V315')
    box(out, 660, 315, 540, 205, 'Conditional reward / runtime branch', ['Program embeddings → non-PAD masked MEAN g', 'No positions: token-multiset representation', 'φ = [g, g × tanh(gate(rz)), |g − proj(rz)|]', 'φ has 3w values → four reward MLP heads', '[rz, g] → runtime head → softplus'])
    arrow(out, 'M735 245V280H925V315')
    box(out, 35, 585, 565, 155, 'Sequence imitation', ['Teacher-forced next-token negative log-likelihood', 'Training programs come from admitted supervision', 'A lower NLL is not a downstream search win'])
    arrow(out, 'M315 520V585')
    box(out, 660, 585, 540, 155, 'Reward / auxiliary objectives', ['Pairwise Bradley–Terry; listwise ranking', 'Runtime Huber; stability BCE when available', 'Program-only ablation and routing terms'])
    arrow(out, 'M925 520V585')
    box(out, 235, 815, 770, 150, 'Training update and evidence', ['Profile-dependent objective combination → backward → AdamW', 'Retain actual updates, changed tensors, sampler and optimizer state', 'Resume/transition invariants; export does not promote serving'])
    arrow(out, 'M315 740V815'); arrow(out, 'M925 740V815')
    text(out, 35, 1018, ['Implemented objectives are imitation and reward/ranking, not PPO or GRPO.',
                         'Tiny CPU fixtures and historical native runs are separate evidence populations.'], 20, 'bold')
    return save('policy-network.svg', out)


def main():
    FIG.mkdir(exist_ok=True)
    outputs = [pipeline(), network()]
    source = Path(__file__).resolve()
    receipt = {'schema': 'jope.manual-ip-diagrams/v1', 'command': 'python3 docs/internal/asm-manual/tools/make_ip_diagrams.py',
               'generator_sha256': hashlib.sha256(source.read_bytes()).hexdigest(), 'figures': outputs,
               'source_files': {name: hashlib.sha256((ROOT.parents[2] / name).read_bytes()).hexdigest() for name in ['crates/jope-policy/src/model.rs', 'crates/jope-policy/src/config.rs', 'python/src/jope/policy/model_v3.py', 'python/src/jope/policy/losses.py', 'python/src/jope/policy/train_v3.py']},
               'scope': 'Source architecture and required boundaries, not live qualification.', 'fits': 0}
    (FIG / 'ip-diagram-receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
