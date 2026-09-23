#!/usr/bin/env python3
"""Render six historical ranking figures from pinned committed CSV inputs.

Adapted from the independently prepared Claude manual contribution. No extraction,
model execution, dataset payload access or current ASM30 performance is performed.
Run from any directory; outputs go to this manual’s figures directory.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402  (ships with pandas)
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, LogNorm  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch, Rectangle  # noqa: E402

TOOLS = Path(__file__).resolve().parent
MANUAL = TOOLS.parent
ROOT = MANUAL.parents[2]  # maintainer-manual -> internal -> docs -> repository root
FIG = MANUAL / "figures"
DATA = FIG / "data"
AS_OF = "2026-09-23"
AS_OF_LEDGER = "2026-09-23 16:29Z"

# Custody receipts written by the integration owner (read only).
CUSTODY = Path("/home/ubuntu/jope-rescue-20260921/asm-integration-20260923")
NI = CUSTODY / "neural-integration"
FA = CUSTODY / "agent-final-acceptance"
LEDGER = "AGENT_CHAT.md (JopeDime canonical checkout)"

# ---------------------------------------------------------------- palette and style
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"  # categorical slots 1-3
OTHER = "#b3b1a8"  # de-emphasised marks
NEUTRAL_FILL = "#f0efec"
GOOD, WARN, SERIOUS, CRIT = "#0ca30c", "#fab219", "#ec835a", "#d03b3b"  # status palette
BLUES = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
         "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
KIND_COLOR = {"baseline": OTHER, "adapter": ORANGE, "search": BLUE}
KIND_LABEL = {"baseline": "single-head baseline", "adapter": "external engine",
              "search": "historical search"}

W = 7.5  # inches: fits A4 text width when scaled
FOOT_SIZE = 6.2
FOOT_WRAP = 150

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 7.5,
    "axes.titlesize": 8.4,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.titlecolor": INK,
    "axes.titlepad": 6,
    "axes.labelsize": 7.6,
    "axes.labelcolor": INK2,
    "axes.edgecolor": AXIS,
    "axes.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "legend.fontsize": 6.8,
    "legend.frameon": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "text.color": INK,
    "hatch.linewidth": 0.6,
})


# ---------------------------------------------------------------- helpers
def _guard(path: Path) -> Path:
    """Refuse sealed partitions and test/holdout label files, whatever asks for them."""
    low = str(path).lower()
    name = path.name.lower()
    if "sealed" in low or "holdout" in low or "held-out" in low or name in {"test.csv", "test.tsv"}:
        raise SystemExit(f"refused: {path} names a sealed or test-label file")
    return path


def read_json(path: Path):
    return json.loads(_guard(path).read_text())


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path).replace("/home/ubuntu/", "~/")


def sha256(path: Path) -> str:
    return hashlib.sha256(_guard(path).read_bytes()).hexdigest()




def load(name: str) -> pd.DataFrame:
    return pd.read_csv(_guard(DATA / name), keep_default_na=True)


def wrap(lines: list[str], width: int = FOOT_WRAP) -> str:
    return "\n".join(textwrap.fill(line, width) for line in lines)


def frame(title: str, subtitle: str, foot: list[str], plot_h: float, w: float = W):
    """A figure with a title block on top and a wrapped source footnote at the bottom.

    Returns (fig, H, y0, y1): total height in inches and the figure-fraction band left
    for the plot area between the title block and the footnote."""
    foot_text = wrap(foot)
    n = foot_text.count("\n") + 1
    foot_h = n * FOOT_SIZE * 1.42 / 72 + 0.16
    if subtitle:
        subtitle = "\n".join(textwrap.fill(p, int(118 * w / W)) for p in subtitle.split("\n"))
    title_h = 0.34 + (0.155 * (subtitle.count("\n") + 1) + 0.05 if subtitle else 0.0)
    H = title_h + plot_h + foot_h
    fig = plt.figure(figsize=(w, H))
    fig.text(0.10 / w, 1 - 0.10 / H, title, ha="left", va="top", fontsize=10.5,
             weight="bold", color=INK)
    if subtitle:
        fig.text(0.10 / w, 1 - 0.36 / H, subtitle, ha="left", va="top", fontsize=7.6,
                 color=INK2, linespacing=1.35)
    fig.text(0.10 / w, 0.07 / H, foot_text, ha="left", va="bottom", fontsize=FOOT_SIZE,
             color=INK2, linespacing=1.42)
    fig.add_artist(Line2D([0.10 / w, 1 - 0.10 / w], [(foot_h - 0.04) / H] * 2,
                          color=GRID, lw=0.6))
    return fig, H, foot_h / H, 1 - title_h / H


def band(fig, H, y0, y1, left_in, right_in, bottom_in, top_in, w: float = W):
    """gridspec keyword margins (figure fractions) inside the plot band, given inches."""
    return dict(left=left_in / w, right=1 - right_in / w,
                bottom=y0 + bottom_in / H, top=y1 - top_in / H)


def save(fig, name: str) -> Path:
    FIG.mkdir(parents=True, exist_ok=True)
    out = FIG / name
    fig.savefig(out, dpi=200, metadata={"Software": None})
    plt.close(fig)
    return out




# ---------------------------------------------------------------- committed evidence paths
REG = ROOT / "assets/regression"
VALID_CSV = REG / "leaderboard-tabular-regression-v2.benchmark-valid.csv"
VALID_SNAP = REG / "elo-tabular-regression-v2.benchmark-valid.json"
PUB_CSV = REG / "sotarena-tabular-regression-v2/results/tabular-regression-v2/leaderboard.csv"
PUB_SNAP = REG / "elo-tabular-regression-v2.json"
CORR = REG / "protocol-corrections-tabular-regression-v2.json"
TRACK = REG / "sotarena-tabular-regression-v2/tracks/tabular-regression-v2/track.json"
SC = ROOT / "assets/same-cut"
PANEL = ROOT / "benchmarks/panel.json"
TEST10 = ROOT / "datasets/test"
FEAT = ROOT / "crates/feat-math/src"
OURS = ("jope-dime", "jope-dime-regressor")  # deploy/ci/readme-benchmarks.py:82

# Short names for the test-10 datasets, keyed by directory (datasets/test/<dir>).
TEST10_SHORT = {
    "pmlb-695-chatfield-4": "chatfield_4",
    "openml-23-cmc": "cmc",
    "pmlb-229-pwlinear": "pwLinear",
    "rdatasets-wooldridge-barium": "barium",
    "rdatasets-stevedata-chile88": "chile88",
    "socrata-cdc-food-safety-culture": "food-safety",
    "openml-29-credit-approval": "credit-approval",
    "pmlb-first-principles-supernovae-zg": "supernovae_zg",
    "uci-parkinsons-telemonitoring": "parkinsons",
    "rdatasets-agridat-bancroft-peanut-uniformity": "peanut-uniform.",
}








# ================================================================ extract


SC_LADDER_COLS = ["version", "stage", "elo_rank", "method_id", "kind", "elo", "average_rank",
                  "dataset_wins", "datasets_scored", "datasets_total", "source"]


def kind_of(method_id: str) -> str:
    if method_id.startswith("adapter."):
        return "adapter"
    if method_id.startswith("baseline."):
        return "baseline"
    return "search"






# The 3fd7 proof left no machine-readable summary; these facts are ledger and audit prose.
ORIGIN_3FD7 = {
    "fitted_outputs": 720,
    "note": ("exit 1; policy finished (64 neural + 16 random proposals); 48 neural gen-0 fits, "
             "0 offspring; finalization failed: expired pre-gate launch classed as retryable"),
    "source": ("ledger 2026-09-23T14:58:45Z; track-a audit-20260923T1510Z.md:148; "
               "neural-integration/origin-run-3fd7.json"),
}








# Fitted inference: state on main and in the unmerged custody packet. Curated, with sources.
FITTED = {
    "Python": ("partial", "bound v3-pipeline emitter (emit_bound); no shared fitted schema yet",
               "partial", "14 of 26 recipes"),
    "Rust": ("partial", "bound v3-pipeline emitter (emit_bound); no shared fitted schema yet",
             "partial", "14 of 26 recipes"),
    "C++17": ("done", "fitted C++17 runtime: all legacy ops and heads; multiclass matrices refused",
              "partial", "ridge + logit only"),
    "C++20": ("partial", "same fitted source when JOPE_CPP_STD=20; no separate parity pin",
              "partial", "ridge + logit only"),
    "JavaScript": ("none", "no fitted JavaScript", "partial", "6 of 26 recipes"),
    "CUDA": ("none", "no CUDA emitter; ordered after the other five (ruling, 16:07Z)",
             "none", "none"),
}
FITTED_SOURCE = ("crates/feat-math/src/advanced_v3/fitted_cpp.rs:57-193; "
                 "crates/feat-math/src/advanced_v3/emit/mod.rs:122; docs/emitters.md:52-60; "
                 "docs/asm.md:231-232; custody numbers: track-a audit-20260923T1510Z.md:183 "
                 "(not re-verified); CUDA ruling reported by the integration owner, ledger "
                 "2026-09-23T16:07:41Z")




ARCH = [
    # key, title, lines, status, source
    ("data", "Data",
     ["test-10: 10 regression sets in datasets/test (in repo)",
      "frozen panel asm-private-30-v1: 10 regression, 10 binary, 10 multiclass PMLB sets",
      "sealed partitions never enter proposal, search, training or motifs"],
     "main", "datasets/manifest.json; benchmarks/panel.json; docs/asm.md:205-206"),
    ("enc", "Encodings",
     ["jope-encoders (lib starforge): foundation, target, chimera encoders",
      "hyperion signatures; dataset context 1536-d, target 512-d",
      "PR 83: CPU context bridge + 8-case native context qualification"],
     "main+pr", "docs/architecture.md:15-19; crates/jope-policy/src/config.rs:39-55; PR 83 diff"),
    ("model", "Deep model",
     ["jope-policy: step-60000 transformer, grammar-masked decode",
      "width 512, 6 layers, 4 heads, FFN 2048, max_seq 128",
      "regression vocabulary 401; producer + cached decode in PR 83"],
     "main+pr", "crates/jope-policy/src/config.rs:39-55; docs/architecture.md:19; audit-20260923T1510Z.md:259-260"),
    ("prog", "Programs (gen 0)",
     ["typed stage lists in the feat-math DSL: transforms + one head",
      "neural arm: 64 policy + 16 random slots; generic arm: 80",
      "then 8 mutation + 8 crossover per arm (96 slots)"],
     "pr", "measured-origin-68f56404/audit.json arms; crates/jope-pipeline/src/neural_search.rs (PR 83)"),
    ("workers", "20 isolated workers",
     ["cancellation, fair admission, resume, cost receipts (#82)",
      "five-fold train-side OOF per candidate; 30 s per fit",
      "20 CPUs, 64 GiB, no swap, 1 thread per worker"],
     "main", "docs/asm.md:161-168,191; measured-origin-68f56404/audit.json resources"),
    ("emit", "Emitters",
     ["Python + Rust: full plan DSL, bit-identical pair",
      "C++17 / C++20 / JavaScript: 22 of 36 transforms, 2 of 12 heads",
      "fitted C++17 runtime; CUDA ordered last"],
     "main", "docs/emitters.md:11-38; crates/feat-math/src/emit_{cpp,js}.rs"),
    ("rec", "Receipts and ladders",
     ["command, cost and recovered-report receipts (PR 83)",
      "same-cut bench reports (jope-bench); Elo ladders (jope-arena)",
      "benchmark-valid ladder with protocol corrections"],
     "main+pr", "assets/same-cut/v2; assets/regression; crates/jope-bench/src/panel/asm_campaign.rs (PR 83)"),
]




# Vision scorecard: track-a audit-20260923T1510Z.md section 3 (lines 192-207), updated with the
# ledger through 16:29Z. States: done / in flight / pending.
STATUS = [
    (1, "Simple install + hello world; 10 wins per class; beat the best", "in flight",
     "Frozen 30-dataset panel (#67); 4 external adapters + 26-recipe Rust roster (#79); same-cut on test-10 (#74, #78)",
     "Install/demo packets in custody, refreshed at 68f56404c, unbuilt. Build and land; warm demo under 60 s with real neural fits; run the 30-dataset panel (not started).",
     "audit s3 row 1; ledger 16:24:10Z, 16:25:59Z"),
    (2, "One-fold screen, then k-fold on the top N", "in flight",
     "jope-bench screen + deep CV (same-cut); panel plan/run/report",
     "PR 83 adds selection-exposure (W1) labels. Left: one ranking rule shared by jope-bench and the panel; a real panel run.",
     "audit s3 row 2"),
    (3, "Encode -> deep model -> program -> gen-0 -> 20 workers", "in flight",
     "20 isolated workers with cancellation, fair admission, resume, cost receipts (#82)",
     "PR 83 @ 68f56404c: CI green (16:28Z); origin proof exit 0 (48 neural gen-0 fits, no win); native review launched 16:29Z; not merged. Producer is regression-only.",
     "audit s3 row 3; measured-origin-68f56404/audit.json; ledger 16:29:20Z"),
    (4, "Design and layout", "pending",
     "8 of 29 full-main failures fixed (#80, #81)",
     "21 failures in 5 groups; jope-contracts; crate split / default-members; regenerate docs/architecture.md.",
     "audit s3 row 4"),
    (5, "Languages: Python, Rust, C++17/20, JavaScript, then CUDA", "in flight",
     "Py/Rs plan emitters (full DSL); C++17/20 + JS: 22/36 transforms, 2/12 heads; fitted C++17 runtime",
     "Fitted-inference custody packets need refresh, integration and parity. Ruling: CUDA after Py/Rs/C++17/C++20/JS (reported by the integration owner, 16:07Z).",
     "audit s3 row 5; tasks.json fitted_inference; ledger 16:07:41Z"),
    (6, "Cost per algorithm vs 1 h / 10M tokens", "in flight",
     "Reference constants; same-cut seconds per program (mean 2.2 s on test-10)",
     "PR 83 cost receipts: 0.218 worker-CPU s per unique predictor (one origin run). PR 84 accounting union in custody; per-stage cost and token use unmeasured.",
     "audit s3 row 6; measured-origin-68f56404/audit.json cost"),
    (7, "Strict IP", "pending",
     "Shipped-text emitter lint + prose lint ip-scrub.sh (#71-#77)",
     "Leaks remain in CLI help, doctor JSON, receipts and PR 83 strings; lints skip 'search'/'tree' and docs/internal. IP packet in Claude custody; owner ruling needed.",
     "audit s3 row 7; ip-leak-packet.md; ledger 16:23:03Z"),
    (8, "A product name that is not 'LLM'", "done",
     "docs/asm.md, jope doctor name and asm-panel-v1 already say ASM",
     "Ruling: ASM (reported by the integration owner, 16:07Z). The 'afm.' method id is baked into same-cut v1/v2 hashes; a rename needs evidence v3.",
     "audit s3 row 8; ledger 16:07:41Z; audit lines 254-258"),
    (9, "Train on all data or per category", "in flight",
     "Trainer train_v3.py (no per-task flag); Rust side is a centroid router",
     "Training packets refreshed at 68f56404c (custody, uncompiled). Blocked: 0 admissible real training rows; two eligible families needed before the one GPU-hour budget.",
     "audit s3 row 9; ledger 16:11:40Z, 16:24:10Z"),
    (10, "Single-dataset high-water-mark attack", "pending",
     "HWM record decides on score alone (hwm.rs:64-85)",
     "Ruling: HWM admission is train-side OOF only (reported 16:07Z). Eligibility source sits in the training overlay; attack verb, resume, freeze-then-test not started.",
     "audit s3 row 10; tasks.json eligible_reuse; ledger 16:07:41Z"),
    (11, "Fitness that rejects 'test high, no train signal'", "in flight",
     "Two ladders + protocol corrections (#68)",
     "Ruling: optimization-visible test labels count as development, with burn receipts (reported 16:07Z). Train-only guard in custody (5 folds, >= 4 strict gains).",
     "audit s3 row 11; audit s2 eligible_reuse; ledger 16:07:41Z"),
    (12, "Mine patterns into new operators", "in flight",
     "Hand-pinned CORE_V1 members; compile-time static registry",
     "B5 jope-motifs on local branch prep/b5-motifs @ abefdfc43, uncompiled; missing mine CLI, eligible-State reader and umbrella wiring.",
     "audit s3 row 12; ledger 15:17Z, 16:11:40Z"),
]




# ================================================================ plots
def _legend_handle(color, marker="o", size=5.5, hollow=False, label=""):
    return Line2D([0], [0], marker=marker, lw=0, markersize=size,
                  markerfacecolor="white" if hollow else color, markeredgecolor=color,
                  markeredgewidth=1.0, label=label)


def fig01() -> Path:
    v = load("fig01_ladder_benchmark_valid.csv")
    snap = read_json(VALID_SNAP)
    ours = v[v.role == "ours"].iloc[0]
    N = 40
    top = v[v.elo_rank <= N]
    title = "Historical benchmark-valid Elo: Golden45 tabular regression"
    subtitle = (f"{snap['methods']} methods, {snap['rankable_rows']:,} rankable scores, "
                f"{snap['excluded_rows']} published rows excluded as test-selected. "
                f"Historical entrant: {ours.method_id} at rank {int(ours.elo_rank)}, Elo {ours.elo:.1f}, "
                f"{int(ours.dataset_wins)} dataset win"
                f"{'' if int(ours.dataset_wins) == 1 else 's'}, {int(ours.datasets_scored)}/"
                f"{int(ours.datasets_total)} coverage.")
    foot = [
        f"Source: {rel(VALID_CSV)} (sha256 {sha256(VALID_CSV)[:16]}), cross-checked against "
        f"{rel(VALID_SNAP)}; the published scores.csv with {rel(CORR)} applied. Committed "
        "arena snapshot; nothing re-run. Elo: regularized Bradley-Terry on the Elo scale, "
        "protocols ranked together, partial coverage allowed.",
        "Caveat: 'benchmark-valid' means no artefact in this repository contradicts the row's "
        "published flag; comparator protocols were not verified. Entrant = the fixed-seed "
        "reproducible run (public-development). Orange = other historical jope-* imported run "
        "reports, not claimed by this repository. Coverage n/45 at right. As of "
        f"{AS_OF}.",
    ]
    fig, H, y0, y1 = frame(title, subtitle, foot, plot_h=6.35)
    gs = fig.add_gridspec(1, 2, width_ratios=[2.25, 1], wspace=0.36,
                          **band(fig, H, y0, y1, 2.15, 0.15, 0.42, 0.28))
    ax, ax2 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    for r in top.itertuples():
        c = {"ours": BLUE, "jope-family-import": ORANGE}.get(r.role, OTHER)
        mk = {"ours": "o", "jope-family-import": "D"}.get(r.role, "o")
        sz = 42 if r.role == "ours" else (20 if r.role != "other" else 16)
        ax.axhline(r.elo_rank, color=GRID, lw=0.5, zorder=0)
        ax.scatter(r.elo, r.elo_rank, s=sz, color=c, marker=mk, zorder=3,
                   edgecolor="white", linewidth=0.6)
        ax.text(1.01, r.elo_rank, f"{int(r.datasets_scored)}/{int(r.datasets_total)}",
                transform=ax.get_yaxis_transform(), va="center", ha="left", fontsize=6,
                color=INK2 if r.role != "other" else MUTED)
    ax.set_yticks(top.elo_rank)
    labels = [f"{int(r.elo_rank)}. " + (r.method if len(r.method) <= 31 else r.method[:30] + "~")
              for r in top.itertuples()]
    ax.set_yticklabels(labels, fontsize=6.4)
    for lab, r in zip(ax.get_yticklabels(), top.itertuples()):
        if r.role == "ours":
            lab.set_fontweight("bold")
            lab.set_color(INK)
    ax.set_ylim(N + 0.8, 0.2)
    ax.set_xlabel("Elo (benchmark-valid)")
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_title(f"(a) top {N} of {len(v)}", pad=4)
    ax.text(1.01, 0.2, "cov.", transform=ax.get_yaxis_transform(), ha="left", va="center",
            fontsize=6, color=MUTED)
    ax.annotate(f"entrant: rank {int(ours.elo_rank)}, Elo {ours.elo:.1f}",
                xy=(ours.elo, ours.elo_rank), xytext=(ours.elo + 55, ours.elo_rank + 3.2),
                fontsize=6.8, color=INK, weight="bold",
                arrowprops=dict(arrowstyle="-", color=BLUE, lw=0.8))
    ax.legend(handles=[_legend_handle(BLUE, label="historical jope-dime-regressor"),
                       _legend_handle(ORANGE, "D", 4.5, label="historical jope-* import"),
                       _legend_handle(OTHER, label="other comparators")],
              loc="lower right", handletextpad=0.3, borderaxespad=0.2)
    # (b) the whole ladder
    ax2.plot(v.elo_rank, v.elo, color=OTHER, lw=1.0, zorder=1)
    ax2.axvspan(0.5, N + 0.5, color=NEUTRAL_FILL, zorder=0)
    ax2.text(N / 2, v.elo.min() + 8, f"shown\nin (a)", ha="center", va="bottom", fontsize=6,
             color=MUTED)
    fam = v[v.role == "jope-family-import"]
    ax2.scatter(fam.elo_rank, fam.elo, s=14, marker="D", color=ORANGE, zorder=3,
                edgecolor="white", linewidth=0.5)
    ax2.scatter([ours.elo_rank], [ours.elo], s=40, color=BLUE, zorder=4, edgecolor="white",
                linewidth=0.6)
    ax2.set_xlabel("Elo rank")
    ax2.set_title(f"(b) all {len(v)}, Elo by rank", pad=4)
    ax2.grid(axis="y")
    ax2.set_xlim(0, len(v) + 5)
    return save(fig, "fig01-elo-benchmark-valid.png")


def fig02() -> Path:
    v = load("fig01_ladder_benchmark_valid.csv")
    p = load("fig02_ladder_published.csv")
    corr = read_json(CORR)
    applied = {a["id"]: a["rows"] for a in read_json(VALID_SNAP)["corrections"]["applied"]}
    ids = list(dict.fromkeys(list(p.method_id[:5]) + list(v.method_id[:5]) + list(OURS)))
    pr = {r.method_id: r for r in p.itertuples()}
    vr = {r.method_id: r for r in v.itertuples()}
    ours_valid = vr["jope-dime-regressor"]
    title = "Historical ladders: published vs benchmark-valid"
    subtitle = ("Same scores.csv, two Elo fits. Published keeps every row; benchmark-valid drops "
                f"the {sum(applied.values())} jope-dime rows this repository's own code labels "
                "test-selected, then refits.")
    n42 = next(c for c in corr["corrections"] if c["id"].startswith("golden42"))
    n46 = next(c for c in corr["corrections"] if c["id"].startswith("golden46"))
    foot = [
        f"Sources: published {rel(PUB_CSV)} (byte-exact copy of the benchmark's ladder); "
        f"benchmark-valid {rel(VALID_CSV)}; corrections {rel(CORR)}: "
        f"{applied[n42['id']]} Golden42 rows are each the per-dataset minimum RMSE over "
        "63,108 test-scored evaluations (assets/regression/SOURCES.md:62); "
        f"{applied[n46['id']]} Golden46 rows come from run {n46['source_revision']}, which "
        "the campaign code writes as test_adaptive: true.",
        "Caveat: the published jope-dime position is a best-known result per dataset, not a "
        "held-out result. Only the top 5 of each ladder and this repository's two entries are "
        f"drawn; the ranks in between are elided. As of {AS_OF}.",
    ]
    fig, H, y0, y1 = frame(title, subtitle, foot, plot_h=3.75)
    top_ranks = [pr[i].elo_rank for i in ids if i in pr and pr[i].elo_rank < 20] + \
                [vr[i].elo_rank for i in ids if i in vr and vr[i].elo_rank < 20]
    tmax = max(top_ranks) + 0.6
    lo_ranks = [x for x in (pr["jope-dime-regressor"].elo_rank, ours_valid.elo_rank)]
    gs = fig.add_gridspec(2, 1, height_ratios=[tmax + 0.4, max(lo_ranks) - min(lo_ranks) + 2.2],
                          hspace=0.12, **band(fig, H, y0, y1, 0.1, 0.1, 0.05, 0.32))
    axt, axb = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    lims = {axt: (tmax, 0.0), axb: (max(lo_ranks) + 1.1, min(lo_ranks) - 1.1)}

    def in_panel(ax, rank):
        hi, lo = lims[ax]
        return lo <= rank <= hi

    for ax in (axt, axb):
        ax.set_xlim(-1.45, 2.45)
        ax.set_ylim(*lims[ax])
        ax.axis("off")
        for x in (0, 1):
            ax.axvline(x, color=GRID, lw=0.8, zorder=0)
    for mid in ids:
        a, b = pr.get(mid), vr.get(mid)
        ours = mid in OURS
        col = BLUE if ours else OTHER
        lw = 1.8 if ours else 1.0
        for ax in (axt, axb):
            if a is not None and in_panel(ax, a.elo_rank):
                ax.scatter(0, a.elo_rank, s=30 if ours else 18, color=col, zorder=3)
                ax.text(-0.06, a.elo_rank, f"{a.method[:30]}   #{a.elo_rank}  {a.elo:.1f}",
                        ha="right", va="center", fontsize=6.8,
                        color=INK if ours else INK2, weight="bold" if ours else "normal")
            if b is not None and in_panel(ax, b.elo_rank):
                ax.scatter(1, b.elo_rank, s=30 if ours else 18, color=col, zorder=3)
                ax.text(1.06, b.elo_rank, f"#{b.elo_rank}  {b.elo:.1f}   {b.method[:30]}",
                        ha="left", va="center", fontsize=6.8,
                        color=INK if ours else INK2, weight="bold" if ours else "normal")
            if a is not None and b is not None and in_panel(ax, a.elo_rank) and in_panel(ax, b.elo_rank):
                ax.plot([0, 1], [a.elo_rank, b.elo_rank], color=col, lw=lw, zorder=2)
            if a is not None and b is None and in_panel(ax, a.elo_rank):
                ax.plot([0, 0.46], [a.elo_rank, a.elo_rank], color=col, lw=lw, ls=(0, (3, 2)),
                        zorder=2)
                ax.scatter(0.5, a.elo_rank, marker="X", s=46, color=CRIT, zorder=4)
                ax.text(0.5, a.elo_rank - 0.62,
                        "removed: all 45 rows test-selected", ha="center", va="bottom",
                        fontsize=6.4, color=CRIT, weight="bold")
    axt.text(0, -0.1 + 0.0, "published ladder", ha="center", va="bottom", fontsize=7.6,
             weight="bold", transform=axt.transData)
    axt.text(1, -0.1, "benchmark-valid ladder", ha="center", va="bottom", fontsize=7.6,
             weight="bold")
    axt.set_ylim(tmax, -0.9)
    lims[axt] = (tmax, -0.9)
    axb.text(0.5, lims[axb][1] - 0.15, "intermediate ranks not drawn", ha="center",
             va="bottom", fontsize=6.2, color=MUTED, style="italic")
    return save(fig, "fig02-ladders-valid-vs-adaptive.png")


def _pretty_method(mid: str, versions: dict) -> str:
    if mid.startswith("adapter."):
        eng = mid.split(".", 1)[1]
        name = {"xgboost": "XGBoost", "lightgbm": "LightGBM", "catboost": "CatBoost",
                "rf": "sklearn RF"}.get(eng, eng)
        ver = versions.get(mid, "")
        return f"{name} {ver}".strip()
    if mid.startswith("baseline."):
        return mid.split(".", 1)[1] + " (head)"
    return "searched program"


def _engine_versions() -> dict:
    per = load("fig04_samecut_v2_per_dataset.csv")
    out = {}
    for r in per[per.kind == "adapter"].itertuples():
        ver = str(r.engine_version)
        m = re.search(r"(\d+\.\d+(?:\.\d+)?)", ver)
        out[r.method_id] = m.group(1) if m else ver
    return out


def fig03() -> Path:
    lad = load("fig03_samecut_ladders.csv")
    versions = _engine_versions()
    title = "Historical same-cut v2: Elo by method on test-10"
    subtitle = ("Every method fitted on identical training rows and scored on identical "
                "held-out rows by one evaluator.\nDeep: finalists re-scored on all 5 cuts "
                "(coverage n/10 = datasets where the method reached the deep stage). "
                "Screen: every method, one cut.")
    foot = [
        "Source: assets/same-cut/v2/leaderboard-jope-same-cut-v1-{deep,screen}.csv with their "
        "elo-*.json snapshots (hash-checked); hollow markers: assets/same-cut/v1 (heads + search "
        "only, same cuts). Recorded run, nothing re-run; search budget 32 programs x 8 rounds, "
        "re-run on every cut; engines run with library defaults, seed 0, one thread.",
        "Caveat: FAVOURABLE SELECTION. test-10 was chosen where the historical engine ranked "
        "first on the published ladder (docs/datasets.md); nothing here generalises to a broader "
        "panel. The deep ladder ranks finalists only, so coverage differs by method. "
        f"As of {AS_OF}.",
    ]
    fig, H, y0, y1 = frame(title, subtitle, foot, plot_h=4.1)
    gs = fig.add_gridspec(1, 2, wspace=0.62, **band(fig, H, y0, y1, 1.25, 0.35, 0.62, 0.25))
    for k, stage in enumerate(("deep", "screen")):
        ax = fig.add_subplot(gs[k])
        cur = lad[(lad.version == "v2") & (lad.stage == stage)].sort_values("elo", ascending=False)
        old = lad[(lad.version == "v1") & (lad.stage == stage)].set_index("method_id")
        y = np.arange(len(cur))
        for yi, r in zip(y, cur.itertuples()):
            c = KIND_COLOR[r.kind]
            ax.axhline(yi, color=GRID, lw=0.5, zorder=0)
            if r.method_id in old.index:
                e1 = old.loc[r.method_id, "elo"]
                ax.plot([e1, r.elo], [yi, yi], color=AXIS, lw=0.8, zorder=1)
                ax.scatter(e1, yi, s=20, facecolor="white", edgecolor=c if c != OTHER else MUTED,
                           linewidth=0.9, zorder=2)
            ax.scatter(r.elo, yi, s=44 if r.kind == "search" else 26, color=c, zorder=3,
                       edgecolor="white", linewidth=0.6)
            if stage == "deep":
                ax.text(1.02, yi, f"{int(r.datasets_scored)}/{int(r.datasets_total)}",
                        transform=ax.get_yaxis_transform(), ha="left", va="center",
                        fontsize=6.2, color=INK2)
        ax.set_yticks(y)
        ax.set_yticklabels([f"{int(r.elo_rank)}. {_pretty_method(r.method_id, versions)}"
                            for r in cur.itertuples()], fontsize=6.6)
        for lab, r in zip(ax.get_yticklabels(), cur.itertuples()):
            if r.kind == "search":
                lab.set_fontweight("bold")
                lab.set_color(INK)
        ax.set_ylim(len(cur) - 0.4, -0.6)
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
        ax.set_xlabel("Elo")
        ax.set_title("(a) deep ladder (finalists)" if stage == "deep"
                     else "(b) screen ladder (all methods, 10/10)", pad=4)
        if stage == "deep":
            ax.text(1.02, -0.9, "cov.", transform=ax.get_yaxis_transform(), ha="left",
                    va="center", fontsize=6, color=MUTED)
        ax.grid(axis="x")
    handles = [_legend_handle(KIND_COLOR[k], label=KIND_LABEL[k]) for k in ("search", "adapter", "baseline")]
    handles.append(_legend_handle(MUTED, hollow=True, label="v1 run (heads + search), same cuts"))
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=6.6,
               bbox_to_anchor=(0.5, y0 + 0.02 / H), handletextpad=0.3, columnspacing=1.2)
    return save(fig, "fig03-samecut-deep-elo.png")


def _samecut_ratio() -> tuple[pd.DataFrame, pd.DataFrame]:
    per = load("fig04_samecut_v2_per_dataset.csv")
    per = per[per.screen_status == "ok"].copy()
    per["best"] = per.groupby("dataset_id").screen_rmse.transform("min")
    per["ratio"] = per.screen_rmse / per.best
    return per, per.drop_duplicates("dataset_id")[["dataset_id", "dataset", "rows", "features"]]


def fig04() -> Path:
    per, ds = _samecut_ratio()
    versions = _engine_versions()
    ds = ds.sort_values(["rows", "dataset"])
    order = (per.groupby("method_id").ratio.median().sort_values().index.tolist())
    mat = per.pivot(index="method_id", columns="dataset_id", values="ratio").loc[order, ds.dataset_id]
    fin = per.pivot(index="method_id", columns="dataset_id", values="deep_status").loc[order, ds.dataset_id]
    cap = 3.0
    title = "Historical same-cut v2: relative error per dataset (screen cut)"
    subtitle = ("Cell = method RMSE / best RMSE on that dataset (1.00 = best). Rows sorted by "
                "median ratio; columns are datasets (training rows x features), sorted by rows.\n"
                "Outlined cells reached the deep stage (top 5 on the screen cut). Colour "
                "saturates at 3.0; labels are rounded ratios.")
    foot = [
        "Source: assets/same-cut/v2/reports/*.bench-report.json (screen metrics per method; the "
        "screen cut is fold 0 of fold_of('<dataset_hash>:20260907', row, 5)), extracted to "
        "figures/data/fig04_samecut_v2_per_dataset.csv. Recorded run, nothing re-run.",
        "Caveat: FAVOURABLE SELECTION (test-10, docs/datasets.md); one cut per cell, so a single "
        "split's noise is included; the deep ladder (fig03) is the pooled 5-cut view. "
        f"As of {AS_OF}.",
    ]
    fig, H, y0, y1 = frame(title, subtitle, foot, plot_h=4.6)
    ax = fig.add_axes([1.55 / W, y0 + 0.88 / H, 1 - 2.35 / W, y1 - y0 - 1.1 / H])
    cax = fig.add_axes([1 - 0.62 / W, y0 + 0.88 / H, 0.12 / W, y1 - y0 - 1.1 / H])
    cmap = LinearSegmentedColormap.from_list("blues", BLUES)
    norm = LogNorm(vmin=1.0, vmax=cap)
    sns.heatmap(mat.clip(upper=cap), ax=ax, cmap=cmap, norm=norm, cbar_ax=cax,
                linewidths=1.0, linecolor="white", annot=False)
    for i, mid in enumerate(mat.index):
        for j, did in enumerate(mat.columns):
            val = mat.loc[mid, did]
            if pd.isna(val):
                continue
            shade = norm(min(val, cap))
            txt = f"{val:.2f}" if val < 10 else f"{val:.0f}"
            ax.text(j + 0.5, i + 0.5, txt, ha="center", va="center", fontsize=5.9,
                    color="white" if shade > 0.55 else INK)
            if fin.loc[mid, did] == "ok":
                ax.add_patch(Rectangle((j + 0.06, i + 0.08), 0.88, 0.84, fill=False,
                                       ec=INK, lw=0.9, zorder=5))
    ax.set_yticks(np.arange(len(mat)) + 0.5)
    ax.set_yticklabels([_pretty_method(m, versions) for m in mat.index], fontsize=6.6,
                       rotation=0)
    for lab, mid in zip(ax.get_yticklabels(), mat.index):
        if kind_of(mid) == "search":
            lab.set_fontweight("bold")
            lab.set_color(INK)
    ax.tick_params(axis="y", pad=11)
    for i, mid in enumerate(mat.index):
        ax.scatter(-0.15, i + 0.5, s=16, marker="s", color=KIND_COLOR[kind_of(mid)],
                   clip_on=False, zorder=6)
    ax.set_xticks(np.arange(len(mat.columns)) + 0.5)
    ax.set_xticklabels([f"{r.dataset}\n{r.rows}x{r.features}" for r in ds.itertuples()],
                       fontsize=6.1, rotation=40, ha="right", rotation_mode="anchor")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(length=0)
    cax.set_ylabel("RMSE / best RMSE (log colour)", fontsize=6.6, color=INK2)
    cax.tick_params(labelsize=6.2)
    cax.set_yticks([1, 1.25, 1.5, 2, 3])
    cax.set_yticklabels(["1.00", "1.25", "1.50", "2.00", ">=3.0"])
    cax.minorticks_off()
    handles = [Patch(color=KIND_COLOR[k], label=KIND_LABEL[k]) for k in ("search", "adapter", "baseline")]
    handles.append(Patch(facecolor="white", edgecolor=INK, label="reached the deep stage"))
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=6.4,
               bbox_to_anchor=(0.5, y0 + 0.02 / H), handlelength=1.0, columnspacing=1.2)
    return save(fig, "fig04-samecut-rmse-heatmap.png")


def fig05() -> Path:
    per = load("fig04_samecut_v2_per_dataset.csv")
    versions = _engine_versions()
    per = per[per.screen_status == "ok"]
    stats = per.groupby("method_id").screen_rank.agg(["median", "mean"]).sort_values(["median", "mean"])
    order = stats.index.tolist()
    nd = per.dataset_id.nunique()
    title = "Historical same-cut v2: per-dataset rank by method (screen cut)"
    subtitle = (f"Each dot is one dataset (n = {nd}); rank 1 = lowest RMSE among the "
                f"{len(order)} methods on that dataset's screen cut. Vertical bar = median; "
                "number at right = mean rank.")
    foot = [
        "Source: assets/same-cut/v2/reports/*.bench-report.json field screen_rank (ranked by the "
        "screen score on one shared cut; on a shared cut R2 and RMSE order identically), extracted "
        "to figures/data/fig04_samecut_v2_per_dataset.csv. Recorded run, nothing re-run.",
        "Caveat: FAVOURABLE SELECTION (test-10, docs/datasets.md); one cut per dataset; ranks "
        f"ignore effect sizes (see fig04 for the ratios). As of {AS_OF}.",
    ]
    fig, H, y0, y1 = frame(title, subtitle, foot, plot_h=4.3)
    ax = fig.add_axes([1.45 / W, y0 + 0.62 / H, 1 - 2.0 / W, y1 - y0 - 0.85 / H])
    for i, mid in enumerate(order):
        sub = per[per.method_id == mid].sort_values(["screen_rank", "dataset_id"])
        c = KIND_COLOR[kind_of(mid)]
        ax.axhline(i, color=GRID, lw=0.5, zorder=0)
        seen: dict[int, int] = {}
        for r in sub.itertuples():
            k = seen.get(r.screen_rank, 0)
            seen[r.screen_rank] = k + 1
            off = (-1) ** k * 0.2 * ((k + 1) // 2)
            ax.scatter(r.screen_rank, i + off, s=22 if kind_of(mid) == "search" else 16,
                       color=c, edgecolor="white", linewidth=0.5, zorder=3)
        med = stats.loc[mid, "median"]
        ax.plot([med, med], [i - 0.36, i + 0.36], color=INK, lw=1.4, zorder=4)
        ax.text(len(order) + 0.9, i, f"{stats.loc[mid, 'mean']:.1f}", ha="left", va="center",
                fontsize=6.4, color=INK2)
    ax.text(len(order) + 0.9, -0.9, "mean", ha="left", va="center", fontsize=6, color=MUTED)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([_pretty_method(m, versions) for m in order], fontsize=6.6)
    for lab, mid in zip(ax.get_yticklabels(), order):
        if kind_of(mid) == "search":
            lab.set_fontweight("bold")
            lab.set_color(INK)
    ax.set_ylim(len(order) - 0.4, -1.3)
    ax.set_xlim(0.4, len(order) + 0.6)
    ax.set_xticks(range(1, len(order) + 1))
    ax.set_xlabel("rank on the dataset's screen cut (1 = best)")
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.grid(axis="x")
    handles = [_legend_handle(KIND_COLOR[k], label=KIND_LABEL[k]) for k in ("search", "adapter", "baseline")]
    handles.append(Line2D([0], [0], color=INK, lw=1.4, label="median rank"))
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=6.6,
               bbox_to_anchor=(0.5, y0 + 0.02 / H), handletextpad=0.3, columnspacing=1.2)
    return save(fig, "fig05-samecut-rank-distribution.png")


def fig06() -> Path:
    s = load("fig06_samecut_v2_search_seconds.csv").sort_values("produce_seconds", ascending=False)
    vals = s.produce_seconds.to_numpy()
    mean, median, mx = vals.mean(), float(np.median(vals)), vals.max()
    ref_s, ref_tok = 3600, 10_000_000  # deploy/ci/readme-benchmarks.py:76-77
    title = "Historical seconds to write each program (same-cut v2, test-10)"
    subtitle = (f"Wall seconds the search took to produce the program on the screen cut's "
                f"training rows: mean {mean:.1f} s, median {median:.2f} s, longest {mx:.1f} s "
                f"over {len(vals)} datasets.")
    foot = [
        "Source: assets/same-cut/v2/reports/*.bench-report.json, method "
        "afm.lite-race.v1.resampled field produce_seconds (budget 32 programs x 8 rounds, 16 "
        "workers on a 128-core host), extracted to figures/data/fig06_samecut_v2_search_seconds.csv. "
        "Timings are recorded wall time under whatever else the host ran, not controlled.",
        f"Reference line: the owner's stated scenario for a large language model writing a "
        f"program for the same task, about 1 hour and {ref_tok:,} tokens "
        "(deploy/ci/readme-benchmarks.py:73-77). This is a contextual duration reference, not a measured comparison. "
        f"Workload, hardware, tokens and currency were not matched; no speedup is established. FAVOURABLE SELECTION (test-10). As of {AS_OF}.",
    ]
    fig, H, y0, y1 = frame(title, subtitle, foot, plot_h=3.5)
    ax = fig.add_axes([1.75 / W, y0 + 0.45 / H, 1 - 2.05 / W, y1 - y0 - 0.62 / H])
    y = np.arange(len(s))
    lo = 0.1
    for yi, r in zip(y, s.itertuples()):
        ax.plot([lo, r.produce_seconds], [yi, yi], color=BLUES[5], lw=1.6, zorder=2,
                solid_capstyle="round")
        ax.scatter(r.produce_seconds, yi, s=30, color=BLUE, zorder=3, edgecolor="white",
                   linewidth=0.6)
        ax.text(r.produce_seconds * 1.18, yi, f"{r.produce_seconds:.2f} s", va="center",
                ha="left", fontsize=6.5, color=INK2, zorder=4,
                bbox=dict(fc="white", ec="none", pad=0.6))
    ax.set_xscale("log")
    ax.set_xlim(lo, 2.2e4)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r.dataset}  ({r.rows}x{r.features})" for r in s.itertuples()],
                       fontsize=6.6)
    ax.set_ylim(len(s) - 0.4, -0.8)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("seconds (log scale)")
    ax.grid(axis="x", which="major")
    ax.axvline(mean, color=INK2, lw=0.9, ls=(0, (4, 2)), zorder=1)
    ax.text(mean * 1.08, -0.55, f"mean {mean:.1f} s", fontsize=6.4, color=INK2, ha="left",
            va="center")
    ax.axvline(ref_s, color=SERIOUS, lw=1.3, zorder=1)
    ax.text(ref_s / 1.1, len(s) - 1.2,
            "owner's LLM reference:\n~1 h and 10M tokens\n(unmeasured scenario)",
            ha="right", va="center", fontsize=6.4, color=INK)
    return save(fig, "fig06-seconds-per-program.png")














def main() -> int:
    for fn in (fig01, fig02, fig03, fig04, fig05, fig06):
        out = fn()
        print(rel(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
