#!/usr/bin/env python3
"""Figures for the deep-stack chapter, drawn from real retained data only.

Selected diagram constants are checked against the source tree; the parameter inventory
is a manually maintained mirror checked during review. Training figures use the manual's retained CSVs and this custody's
read-only extraction (extract_deep_stack_data.py). A receipt pins every input and output by SHA-256.
"""
import argparse
import csv
import hashlib
import json
import pathlib
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5, "svg.hashsalt": "deep-stack-20260924",
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.alpha": 0.25, "grid.linewidth": 0.5, "legend.frameon": False,
})
PNG_META = {"Software": None}
C_NEURAL, C_GENERIC, C_TRIVIAL, C_MARK = "#1f5fa8", "#c2571a", "#6b6b6b", "#2e8b57"
INPUTS = {}


def sha(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def pin(path):
    INPUTS[str(path)] = sha(path)
    return path


def rows(path):
    with open(pin(path), newline="") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------- source-checked constants
def source_constants(repo):
    def text(rel):
        return pathlib.Path(pin(repo / rel)).read_text()

    config = text("crates/jope-policy/src/config.rs")
    block = config[config.index("pub fn regression"):]
    block = block[:block.index("}\n    }")]
    cfg = {k: int(v) for k, v in re.findall(r"(\w+): (\d+),", block)}
    assert "vocab_size + 256" in block
    model = text("crates/jope-policy/src/model.rs")
    for needle in ("for head in 0..4", "c.max_seq + 1", "vec![3 * w, w]", '"runtime_head.0", 2 * w, w',
                   '"reward_heads.{head}.0"), 3 * w, w', '"token_emb.weight".into(), vec![c.embedding_rows, w]'):
        assert needle in model, f"model.rs no longer contains {needle!r}; redraw the network"
    log_vars = re.search(r"LOSS_LOG_VARS: \[&str; (\d+)\]", model)
    n_log_vars = int(log_vars.group(1)) if log_vars else None
    search = text("crates/jope-pipeline/src/neural_search.rs")
    decode = text("crates/jope-policy/src/decode.rs")
    const = lambda src, name: int(re.search(rf"const {name}: usize = (\d+);", src).group(1))
    defaults = re.search(r"rounds: (\d+),\s*batch: (\d+),\s*temperature: ([0-9.]+),", decode)
    producer = text("crates/jope-policy/src/context_producer.rs")
    assert ".clamp(-32f32, 32f32)" in producer
    emit = text("crates/jope-pipeline/src/emit.rs")
    languages = re.findall(r"Language::(\w+) => \"(\w+)\"", emit)
    vocab = json.loads(pathlib.Path(pin(repo / "crates/jope-engine/assets/grammar/featmath.json")).read_text())
    tokens = vocab["tokens"]
    categories = {}
    for t in tokens:
        key = t.split("=")[0].split(":")[0] if "=" in t or ":" in t else t
        categories[key] = categories.get(key, 0) + 1
    vocab_size = len(tokens)
    c = dict(cfg, vocab_size=vocab_size, embedding_rows=vocab_size + 256,
             initial=const(search, "INITIAL_SIZE"), neural_slots=const(search, "NEURAL_SLOTS"),
             elites=const(search, "ELITES"), pool=const(decode, "POOL_SIZE"),
             exploration=const(decode, "EXPLORATION_SIZE"), rounds=int(defaults.group(1)),
             batch=int(defaults.group(2)), temperature=float(defaults.group(3)), clip=32,
             languages=[name for _, name in dict.fromkeys(languages)], categories=categories,
             n_log_vars=n_log_vars)
    assert c["pool"] == c["neural_slots"] and c["pool"] + c["exploration"] == c["initial"]
    assert c["dataset_dim"] + c["target_dim"] == 2048
    c["params"], c["tensors"], c["groups"] = parameter_inventory(c)
    return c


def parameter_inventory(c):
    """Mirror of crates/jope-policy/src/model.rs expected_inventory."""
    w, out = c["width"], []
    dense = lambda name, i, o: out.extend([(f"{name}.weight", o * i), (f"{name}.bias", o)])
    norm = lambda name: out.extend([(f"{name}.weight", w), (f"{name}.bias", w)])
    groups = {}

    def group(label, fn):
        start = len(out)
        fn()
        groups[label] = groups.get(label, 0) + sum(n for _, n in out[start:])

    for enc, label in (("encoder", "context MLP (proposal)"), ("reward_encoder", "context MLP (reward)")):
        group(label, lambda enc=enc: (dense(f"{enc}.0", c["dataset_dim"] + c["target_keep_dim"], w),
                                      norm(f"{enc}.2"), dense(f"{enc}.4", w, w), norm(f"{enc}.6")))
    group("context prefix projection", lambda: dense("ctx_proj", w, w))
    group("token + position embeddings", lambda: (out.append(("token_emb.weight", c["embedding_rows"] * w)),
                                                   out.append(("pos_emb.weight", (c["max_seq"] + 1) * w))))
    for layer in range(c["layers"]):
        p = f"decoder.layers.{layer}"
        group(f"decoder blocks ({c['layers']})", lambda p=p: (
            out.append((f"{p}.self_attn.in_proj_weight", 3 * w * w)),
            out.append((f"{p}.self_attn.in_proj_bias", 3 * w)),
            dense(f"{p}.self_attn.out_proj", w, w), dense(f"{p}.linear1", w, c["ffn"]),
            dense(f"{p}.linear2", c["ffn"], w), norm(f"{p}.norm1"), norm(f"{p}.norm2")))
    group("output norm", lambda: norm("out_norm"))
    group("reward program embedding + norm", lambda: (out.append(("reward_token_emb.weight", c["embedding_rows"] * w)),
                                                       norm("reward_norm")))
    group("context gate and projection", lambda: (dense("z_gamma.0", w, w), dense("z_proj", w, w)))
    for head in range(4):
        group("4 reward heads", lambda head=head: (dense(f"reward_heads.{head}.0", 3 * w, w),
                                                   dense(f"reward_heads.{head}.3", w, 1)))
    group("runtime head", lambda: (dense("runtime_head.0", 2 * w, w), dense("runtime_head.2", w, 1)))
    group("unused at inference: stability, ablation, loss weights", lambda: (
        dense("stability_head.0", 3 * w, w), dense("stability_head.2", w, 1),
        dense("reward_head_ablation.0", w, w), dense("reward_head_ablation.3", w, 1),
        out.extend([(f"loss_log_vars.{i}", 1) for i in range(c["n_log_vars"] or 7)])))
    return sum(n for _, n in out), len(out), groups


# ---------------------------------------------------------------- diagrams (hand-built SVG)
def svg_box(x, y, w, h, title, lines, fill, stroke="#333", dash=False):
    d = ' stroke-dasharray="5,3"' if dash else ""
    s = [f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" fill="{fill}" stroke="{stroke}" stroke-width="1.1"{d}/>',
         f'<text x="{x + w / 2}" y="{y + 15}" text-anchor="middle" font-weight="bold" font-size="11">{title}</text>']
    for i, line in enumerate(lines):
        s.append(f'<text x="{x + w / 2}" y="{y + 30 + 12.5 * i}" text-anchor="middle" font-size="9.2">{line}</text>')
    return "\n".join(s)


def arrow(x1, y1, x2, y2, dash=False, color="#333"):
    d = ' stroke-dasharray="5,3"' if dash else ""
    return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="1.3"{d} '
            f'marker-end="url(#arrow)"/>')


def svg_doc(width, height, body):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" '
            f'height="{height}" font-family="DejaVu Sans, sans-serif">\n'
            '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
            'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#333"/></marker></defs>\n'
            f'<rect width="{width}" height="{height}" fill="white"/>\n{body}\n</svg>\n')


RUN, HELD, PENDING = "#e3f1e6", "#fcefd6", "#f1f1f1"


def pipeline_svg(c, path):
    W, bw, bh = 1000, 178, 102
    xs = [12 + i * (bw + 20) for i in range(5)]
    top = [
        ("1 Data", ["frozen development rows", "cut 0: folds 1–4 fit,", "fold 0 held back", "sealed rows never opened"], RUN),
        ("2 Encoding", [f"Hyperion 512 (features)", "Foundation 2048→1024", "Target 512", f"= 2048 floats, clip ±{c['clip']}"], RUN),
        ("3 Policy network", [f"decoder {c['width']}×{c['layers']}, {c['heads']} heads", f"{c['params']:,} parameters",
                              f"{c['vocab_size']}-token grammar", "Candle CPU float32"], RUN),
        ("4 Proposals", [f"grammar-masked sampling", f"{c['pool']} ranked + {c['exploration']} exploration",
                         f"= {c['initial']} generation-0 programs", "reward heads order slots"], RUN),
        ("5 Bounded GP", [f"{c['initial']} gen-0, {c['elites']} elites,", f"≤{c['initial'] - c['elites']} children / generation",
                          "every fit real, 5 inner", "OOF folds pick winners"], RUN),
    ]
    bottom = [
        ("10 New checkpoint", ["must beat its parent in", "matched fitted search", "(LOOP-01) — no", "generation has run"], PENDING, True),
        ("9 Training", ["native 512×6 trainer: held", "Python 256×2 lineage v1:", "CPU fixtures only,", "never trained on real data"], HELD, True),
        ("8 Teaching corpus", ["guard: 5 folds, ≥4 wins", "vs training-mean baseline", "→ HWM → frozen corpus", "0 teachers admitted so far"], HELD, True),
        ("7 Code emission", ["legacy emitters: " + " · ".join(c["languages"][:2]), " · ".join(c["languages"][2:]),
                             "legacy excludes ASM recipes;", "fitted export: allowlisted path"], HELD, True),
        ("6 Refit + held fold", ["winner refit on folds 1–4", "scored once on fold 0", "(34 rows for 230_machine_cpu)",
                                 "transfer report: not measured"], RUN, False),
    ]
    body = [f'<text x="{W / 2}" y="18" text-anchor="middle" font-size="12.5" font-weight="bold">'
            'ASM end to end: data → encoding → deep model → GP → code → training</text>']
    for i, (t, lines, fill) in enumerate(top):
        body.append(svg_box(xs[i], 32, bw, bh, t, lines, fill))
        if i < 4:
            body.append(arrow(xs[i] + bw, 32 + bh / 2, xs[i + 1] - 2, 32 + bh / 2))
    y2 = 32 + bh + 46
    for i, (t, lines, fill, dash) in enumerate(bottom):
        body.append(svg_box(xs[i], y2, bw, bh, t, lines, fill, dash=dash))
    body.append(arrow(xs[4] + bw / 2, 32 + bh, xs[4] + bw / 2, y2 - 2))
    body.append(arrow(xs[4], y2 + bh / 2, xs[3] + bw + 2, y2 + bh / 2))
    ya = y2 + bh + 16
    body.append(f'<path d="M{xs[4] + bw / 2},{y2 + bh} L{xs[4] + bw / 2},{ya} L{xs[2] + bw / 2},{ya} '
                f'L{xs[2] + bw / 2},{y2 + bh + 2}" fill="none" stroke="#333" stroke-width="1.3" '
                'stroke-dasharray="5,3" marker-end="url(#arrow)"/>')
    body.append(f'<text x="{xs[3] + bw / 2}" y="{ya + 12}" text-anchor="middle" font-size="9.2" font-style="italic">'
                'frozen winners and their train-side OOF evidence go to the teaching guard</text>')
    for i in (2, 1):
        body.append(arrow(xs[i], y2 + bh / 2, xs[i - 1] + bw + 2, y2 + bh / 2, dash=True))
    body.append(f'<path d="M{xs[0] + bw / 2},{y2} L{xs[0] + bw / 2},{y2 - 18} L{xs[2] + bw / 2},{y2 - 18} '
                f'L{xs[2] + bw / 2},{32 + bh + 2}" fill="none" stroke="#333" stroke-width="1.3" '
                'stroke-dasharray="5,3" marker-end="url(#arrow)"/>')
    body.append(f'<text x="{(xs[0] + xs[2]) / 2 + bw / 2}" y="{y2 - 22}" text-anchor="middle" font-size="9.2" '
                'font-style="italic">expert iteration: GP is the teacher; not reinforcement learning</text>')
    ly = y2 + bh + 50
    for j, (fill, label, dash) in enumerate(((RUN, "runs today (bounded evidence)", False),
                                             (HELD, "implemented, held or unqualified", True),
                                             (PENDING, "pending — not measured", True))):
        x = 140 + j * 260
        d = ' stroke-dasharray="5,3"' if dash else ""
        body.append(f'<rect x="{x}" y="{ly - 10}" width="22" height="13" fill="{fill}" stroke="#333"{d}/>'
                    f'<text x="{x + 28}" y="{ly}" font-size="9.5">{label}</text>')
    pathlib.Path(path).write_text(svg_doc(W, ly + 14, "\n".join(body)))


def network_svg(c, path):
    W, H = 1000, 640
    w = c["width"]
    b = ['<text x="500" y="18" text-anchor="middle" font-size="12.5" font-weight="bold">'
         f'Policy network (accepted lineage B, step 60,000): {c["params"]:,} parameters in {c["tensors"]} tensors</text>']
    b.append(svg_box(250, 32, 500, 58, "Context vector: 2048 float32 (clipped to ±32)",
                     [f"Hyperion 512 (features only) | Foundation 1024 (features + training targets) | Target 512",
                      f"config: dataset_dim {c['dataset_dim']} + target_dim {c['target_dim']}"], "#eaf0fa"))
    b.append(arrow(390, 90, 250, 118))
    b.append(arrow(610, 90, 750, 118))
    # proposal branch
    b.append(svg_box(40, 120, 420, 58, "Context MLP (proposal)", [f"2048→{w}, GELU, LayerNorm, dropout, {w}→{w}, GELU, LayerNorm",
                                                                    f"then ctx_proj {w}→{w}: the context becomes token 0"], RUN))
    b.append(arrow(250, 178, 250, 198))
    b.append(svg_box(40, 200, 420, 58, "Token + position embeddings",
                     [f"token_emb {c['embedding_rows']}×{w} (vocab {c['vocab_size']} + 256 spare rows)",
                      f"pos_emb {c['max_seq'] + 1}×{w} (context position + {c['max_seq']} tokens)"], RUN))
    b.append(arrow(250, 258, 250, 278))
    b.append(svg_box(40, 280, 420, 86, f"{c['layers']} × pre-norm decoder block",
                     [f"LayerNorm → causal self-attention, {c['heads']} heads × {w // c['heads']} (packed QKV)",
                      "mask: future positions and PAD keys hidden",
                      f"LayerNorm → feed-forward {w}→{c['ffn']}→{w}", "residual connections; no cross-attention"], RUN))
    b.append(arrow(250, 366, 250, 386))
    b.append(svg_box(40, 388, 420, 58, "Output",
                     ["out_norm → logits = hidden · token_embᵀ (tied weights)",
                      f"narrowed to the {c['vocab_size']} grammar tokens"], RUN))
    b.append(arrow(250, 446, 250, 466))
    b.append(svg_box(40, 468, 420, 72, "Grammar-constrained sampling",
                     [f"illegal tokens masked; softmax(logits / T), default T = {c['temperature']:g}",
                      f"default {c['rounds']} rounds × {c['batch']} rows (the origin proof ran 4 × 64)",
                      f"first {c['pool']} distinct valid programs + {c['exploration']} exploration"], RUN))
    # reward branch
    b.append(svg_box(540, 120, 420, 58, "Context MLP (reward)", [f"same shape, separate weights → rz ({w})",
                                                                 "used only for ranking, never for sampling"], RUN))
    b.append(svg_box(540, 200, 420, 72, "Program summary g",
                     [f"reward_token_emb {c['embedding_rows']}×{w}, mean over non-PAD tokens",
                      "no position term: order-blind (same token multiset,", "same score) → LayerNorm"], RUN))
    b.append(arrow(750, 178, 750, 198))
    b.append(arrow(750, 272, 750, 292))
    b.append(svg_box(540, 294, 420, 58, "Interaction features φ (1536)",
                     ["[ g,  g ⊙ tanh(z_gamma(rz)),  |g − z_proj(rz)| ]", f"three blocks of {w}"], RUN))
    b.append(arrow(750, 352, 750, 372))
    b.append(svg_box(540, 374, 420, 72, "Heads",
                     [f"4 reward heads: 1536→{w}→1; their mean ranks the 64 slots",
                      f"runtime head: softplus([rz, g] 1024→{w}→1)",
                      "stability and ablation heads: trained, unused at inference"], RUN))
    b.append(svg_box(540, 468, 420, 72, "What the scores are",
                     ["within-context ranking logits, not calibrated RMSE",
                      "they order slots and record rank_log_weight;",
                      "measured fits, not the model, pick GP parents"], PENDING))
    # parameter table
    y = 566
    b.append(f'<text x="40" y="{y}" font-size="10" font-weight="bold">Parameters by group '
             '(computed from the tensor inventory in crates/jope-policy/src/model.rs)</text>')
    items = list(c["groups"].items())
    for i, (label, n) in enumerate(items):
        col, row = divmod(i, 5)
        b.append(f'<text x="{40 + col * 330}" y="{y + 16 + row * 13}" font-size="9">{label}: {n:,}</text>')
    pathlib.Path(path).write_text(svg_doc(W, H, "\n".join(b)))


# ---------------------------------------------------------------- training figures
def f1_fixed_monitor(data, out):
    r = rows(data / "a100-fixed-monitor.csv")
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.6), sharex=True)
    metrics = (("sequence_nll", "sequence NLL (nats)"), ("ranking_bt", "Bradley–Terry ranking loss"),
               ("ranking_listwise", "listwise ranking loss"), ("runtime_huber", "runtime Huber loss"))
    main = [x for x in r if x["executable_sha256"].startswith("87995977")]
    alt = [x for x in r if not x["executable_sha256"].startswith("87995977")]
    for ax, (m, label) in zip(axes.flat, metrics):
        steps = [int(x["evaluated_step"]) for x in main]
        vals = [float(x[m]) for x in main]
        ax.plot(steps, vals, "o-", color=C_NEURAL, ms=3, lw=1.1, label="executable 87995977")
        for x in alt:
            ax.plot(int(x["evaluated_step"]), float(x[m]), "s", mfc="none", color=C_GENERIC, ms=6,
                    label="parent re-scored, executable db0aabe9")
        ax.axvline(60000, color=C_MARK, ls="--", lw=1)
        ax.axvline(69000, color=C_TRIVIAL, ls=":", lw=1)
        best = min(range(len(vals)), key=vals.__getitem__)
        ax.plot(steps[best], vals[best], "*", color=C_MARK, ms=9)
        at60 = vals[steps.index(60000)]
        ax.text(0.98, 0.95, f"min {vals[best]:.4f} at {steps[best]:,}\nstep 60,000: {at60:.4f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=7)
        ax.set_title(label, fontsize=9)
    for ax in axes[1]:
        ax.set_xlabel("optimizer step")
    h, l = axes[0, 0].get_legend_handles_labels()
    keep = dict(zip(l, h))
    keep["accepted step 60,000"] = plt.Line2D([], [], color=C_MARK, ls="--")
    keep["unaccepted step 69,000"] = plt.Line2D([], [], color=C_TRIVIAL, ls=":")
    keep["lowest value"] = plt.Line2D([], [], color=C_MARK, marker="*", ls="none", ms=8)
    fig.legend(keep.values(), keep.keys(), loc="lower center", ncol=3, fontsize=7.5)
    fig.suptitle("Fixed panel fbff7a04, corpus 11e60a43, 2,048 examples, one population for every checkpoint",
                 fontsize=9.5)
    fig.tight_layout(rect=(0, 0.09, 1, 0.95))
    fig.savefig(out, dpi=125, metadata=PNG_META)
    plt.close(fig)


def per_cycle(ax, r, key, value, color, lw=0.9):
    cycles = sorted({int(x["cycle"]) for x in r})
    for cyc in cycles:
        pts = [x for x in r if int(x["cycle"]) == cyc]
        ax.plot([int(x[key]) for x in pts], [float(x[value]) for x in pts], color=color, lw=lw)


def f2_components(manual, out):
    r = rows(manual / "per-step-history/a100-step-bins.csv")
    metrics = (("loss", "composite loss"), ("sequence_nll", "sequence NLL"), ("ranking_bt", "Bradley–Terry"),
               ("ranking_listwise", "listwise"), ("runtime_huber", "runtime Huber"), ("gradient_norm", "gradient norm"))
    fig, axes = plt.subplots(3, 2, figsize=(7.2, 5.6), sharex=True)
    for ax, (m, label) in zip(axes.flat, metrics):
        pts = [x for x in r if x["metric"] == m]
        cycles = sorted({int(x["cycle"]) for x in pts})
        for cyc in cycles:
            seg = [x for x in pts if int(x["cycle"]) == cyc]
            ax.plot([int(x["first_step"]) for x in seg], [float(x["mean"]) for x in seg], color=C_NEURAL, lw=0.8)
        ax.axvline(60000, color=C_MARK, ls="--", lw=1)
        ax.set_title(label, fontsize=9)
    for ax in axes[2]:
        ax.set_xlabel("optimizer step")
    fig.suptitle("A100 lineage B, steps 6,001–69,000: training-minibatch metrics as 100-step means, one line per\n"
                 "3,000-step block (each block has its own corpus). Not a held-out curve.", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out, dpi=120, metadata=PNG_META)
    plt.close(fig)


def f3_throughput(data, out):
    r = rows(data / "a100-throughput-bins.csv")
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5), sharex=True)
    for ax, (m, label) in zip(axes, (("examples_per_second", "examples / second"),
                                     ("step_seconds", "seconds per optimizer step"),
                                     ("communication_seconds", "all-reduce seconds per step"))):
        per_cycle(ax, r, "first_step", m, C_NEURAL)
        ax.set_title(label, fontsize=9)
        ax.set_xlabel("optimizer step")
    fig.suptitle("A100 × 8, global batch 2,048: 100-step means from the original metrics.jsonl", fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(out, dpi=120, metadata=PNG_META)
    plt.close(fig)


def f4_gen31(manual, out):
    r = rows(manual / "training-history/gen31-epoch-metrics.csv")
    get = lambda scope, metric: sorted((int(x["epoch"]), float(x["value"])) for x in r
                                       if x["scope"] == scope and x["metric"] == metric)
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6))
    tl, vl = get("epoch_summary", "train_loss"), get("epoch_summary", "val_loss")
    axes[0].plot(*zip(*tl), "o-", ms=3, color=C_NEURAL, label="training")
    axes[0].plot(*zip(*vl), "s-", ms=3, color=C_GENERIC, label="validation")
    axes[0].set_title("composite loss", fontsize=9)
    axes[0].legend(fontsize=7.5)
    axes[1].plot(*zip(*get("epoch_summary", "overfit_gap")), "o-", ms=3, color=C_TRIVIAL)
    axes[1].set_title("overfit gap (validation − training)", fontsize=9)
    ts, vs = get("train_loss_components", "seq_nll"), get("val_loss_components", "seq_nll")
    axes[2].plot(*zip(*ts), "o-", ms=3, color=C_NEURAL, label="training")
    axes[2].plot(*zip(*vs), "s-", ms=3, color=C_GENERIC, label="validation")
    axes[2].set_title("sequence NLL", fontsize=9)
    for ax in axes:
        ax.axvspan(0.5, 6.5, color="#eeeeee", zorder=0)
        ax.set_xlabel("epoch (shaded: phase A)")
    fig.suptitle("Legacy policy_v3.2 gen31: 125,970 training rows, 2,169 fixed validation rows", fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(out, dpi=120, metadata=PNG_META)
    plt.close(fig)


# ---------------------------------------------------------------- transfer figures
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="/home/ubuntu/JopeDime", help="source tree for the checked constants")
    ap.add_argument("--manual-data", help="manual data directory (default: REPO/docs/internal/asm-manual/data)")
    ap.add_argument("--training-data", required=True, help="directory holding a100-fixed-monitor.csv and a100-throughput-bins.csv")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    repo, out = pathlib.Path(a.repo), pathlib.Path(a.out)
    manual = pathlib.Path(a.manual_data) if a.manual_data else repo / "docs/internal/asm-manual/data"
    training = pathlib.Path(a.training_data)
    out.mkdir(parents=True, exist_ok=True)
    c = source_constants(repo)
    pipeline_svg(c, out / "deep-stack-pipeline.svg")
    network_svg(c, out / "deep-stack-network.svg")
    f1_fixed_monitor(training, out / "ds-f1-fixed-monitor.png")
    f2_components(manual, out / "ds-f2-a100-components.png")
    f3_throughput(training, out / "ds-f3-a100-throughput.png")
    f4_gen31(manual, out / "ds-f4-gen31-train-validation.png")
    receipt = {"schema": "deep-stack.figure-receipt/v1",
               "source_constants": {k: v for k, v in c.items() if k != "groups"} | {"parameter_groups": c["groups"]},
               "inputs": dict(sorted(INPUTS.items())),
               "outputs": {p.name: sha(p) for p in sorted(out.iterdir())
                           if p.suffix in (".png", ".svg") and (p.name.startswith("ds-") or p.name.startswith("deep-stack-"))}}
    receipt["command"] = "python3 docs/internal/asm-manual/tools/make_deep_stack_figures.py --repo REPO --training-data DATA --out FIGURES"
    receipt["extraction_command"] = "python3 docs/internal/asm-manual/tools/extract_deep_stack_data.py --out DATA"
    receipt["plots"] = {
        "ds-f1-fixed-monitor.png": {"coverage": "24 records, 23 checkpoint steps 3000–69000, 2048 examples per record", "units": "sequence NLL, Bradley–Terry, listwise and runtime Huber losses", "limits": "One unchanged adaptive teaching panel; not untouched evaluation or downstream search quality."},
        "ds-f2-a100-components.png": {"coverage": "63000 steps in 21 separate 3000-step blocks; 100-step means", "units": "training losses and gradient norm", "limits": "Different block corpora; training minibatches, not held-out performance. No missing-step interpolation."},
        "ds-f3-a100-throughput.png": {"coverage": "63000 steps, 630 bins, 21 logs, eight A100 GPUs and batch2048", "units": "examples/second and seconds/step", "limits": "Historical timing; logged processing throughput differs from end-to-end throughput. Not a current benchmark."},
        "ds-f4-gen31-train-validation.png": {"coverage": "gen31 retained epoch records, 125970 train and2169 validation rows", "units": "composite loss, recorded gap and sequence NLL", "limits": "Phase-B reward weighting changes imitation objective; incomparable to native composite losses."},
    }
    (out / "deep-stack-figure-receipt.json").write_text(json.dumps(receipt, indent=1) + "\n")
    print(json.dumps({"params": c["params"], "tensors": c["tensors"], "vocab": c["vocab_size"],
                      "categories": c["categories"], "languages": c["languages"],
                      "outputs": {k: (v[:12], (out / k).stat().st_size) for k, v in receipt["outputs"].items()}},
                     indent=1))


if __name__ == "__main__":
    sys.exit(main())
