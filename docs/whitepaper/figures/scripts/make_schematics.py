#!/usr/bin/env python3
"""Generate conceptual & architectural schematics (Figures 2, 3, and 5).

Contains zero fabricated empirical results:
- fig2_novelty_judgment.pdf: Conceptual 2D schematic of Novelty vs. Referee Value,
  with EurekaZero's single proposal (B=1) explicitly shown as an unmeasured
  vertical question-mark interval ('?', tested in Section 7).
- fig3_game_loop.pdf: Schematic of the innovation game loop (top blue row:
  game-day search=1 deployment; dashed box: training-only referee & ratchets).
- fig5_network.pdf: Architectural schematic of the 27.8M step-60,000 reference
  checkpoint showing the frozen 2,048-value encoder, grammar-masked writer
  pi_theta, and separate training-only ranking map v_psi (with explicit note
  that v_psi has no influence path into pi_theta's draw).
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

OUT = Path(__file__).resolve().parents[1]

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8,
    "axes.labelsize": 8.0,
    "axes.titlesize": 8.5,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.0,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def make_fig2_novelty_judgment() -> None:
    """Conceptual schematic: Novelty vs. Value as judged by the referee."""
    fig, ax = plt.subplots(figsize=(3.35, 2.30))

    ax.set_xlim(0.0, 1.04)
    ax.set_ylim(0.0, 1.06)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.9)
    ax.spines["bottom"].set_linewidth(0.9)
    ax.set_xlabel("Distance from prior human solutions (novelty) $\\rightarrow$", labelpad=4)
    ax.set_ylabel("Value judged by referee $\\mathcal{R}$ $\\rightarrow$", labelpad=4)

    # Declared incumbent horizontal dashed line
    inc_y = 0.55
    ax.axhline(inc_y, color="#475569", lw=1.0, ls="--", zorder=2)
    ax.text(
        0.02, inc_y + 0.02,
        "Declared incumbent\n$\\mathrm{inc}(p)$",
        fontsize=7.2, color="#334155", fontstyle="italic", va="bottom",
    )

    # 1. Imitation prior (lower-left / near incumbent)
    rng = np.random.default_rng(42)
    im_x = rng.normal(0.16, 0.04, 14)
    im_y = rng.normal(0.47, 0.035, 14)
    ax.scatter(im_x, im_y, s=16, color="#64748b", alpha=0.85, edgecolors="white", linewidths=0.4, zorder=3)
    ax.text(
        0.17, 0.34,
        "Imitation prior\n(near precedent)",
        fontsize=7.2, color="#334155", ha="center", va="top",
        bbox=dict(boxstyle="round,pad=0.22", facecolor="#f8fafc", edgecolor="#cbd5e1", lw=0.6),
    )

    # 2. Random generator (lower-right, explicitly cited in text)
    rand_x = rng.uniform(0.68, 0.96, 16)
    rand_y = rng.uniform(0.04, 0.13, 16)
    ax.scatter(rand_x, rand_y, s=15, marker="x", color="#94a3b8", linewidths=0.9, zorder=3)
    ax.text(
        0.79, 0.17,
        "Random generator\n(high novelty, low value)",
        fontsize=7.2, color="#475569", ha="center", va="bottom",
        bbox=dict(boxstyle="round,pad=0.22", facecolor="#f8fafc", edgecolor="#cbd5e1", lw=0.6),
    )

    # 3. Repeated search (B >> 1): cloud spanning across incumbent
    srch_x = np.array([0.41, 0.44, 0.46, 0.48, 0.50, 0.52, 0.45, 0.49, 0.53, 0.43, 0.51, 0.47])
    srch_y = np.array([0.20, 0.31, 0.42, 0.49, 0.37, 0.46, 0.62, 0.70, 0.65, 0.14, 0.27, 0.58])
    below = srch_y <= inc_y
    above = srch_y > inc_y
    ax.scatter(srch_x[below], srch_y[below], s=17, color="#93c5fd", edgecolors="#2563eb", linewidths=0.6, zorder=3)
    ax.scatter(srch_x[above], srch_y[above], s=22, color="#2563eb", edgecolors="#1e3a8a", linewidths=0.6, zorder=4)
    ax.text(
        0.295, 0.77,
        "Repeated search ($B \\gg 1$)\n(many tries; keep winners)",
        fontsize=7.2, color="#1e40af", ha="center", va="bottom",
        bbox=dict(boxstyle="round,pad=0.22", facecolor="#eff6ff", edgecolor="#93c5fd", lw=0.6),
    )

    # 4. Learned proposer (EurekaZero, B = 1): unmeasured vertical interval with '?'
    ez_x = 0.82
    ax.plot([ez_x, ez_x], [0.39, 0.73], color="#d97706", lw=1.3, ls=":", zorder=4)
    ax.scatter([ez_x, ez_x], [0.39, 0.73], marker="_", s=40, color="#d97706", linewidths=1.3, zorder=4)
    ax.scatter([ez_x], [inc_y], s=85, color="#fffbeb", edgecolors="#d97706", linewidths=1.2, zorder=5)
    ax.text(ez_x, inc_y, "?", fontsize=8.0, fontweight="bold", color="#b45309", ha="center", va="center", zorder=6)
    ax.text(
        0.81, 0.79,
        "Learned proposer ($B = 1$)\n(height tested in Section 7)",
        fontsize=7.2, color="#92400e", ha="center", va="bottom",
        bbox=dict(boxstyle="round,pad=0.22", facecolor="#fffbeb", edgecolor="#fcd34d", lw=0.7),
    )

    fig.tight_layout(pad=0.25)
    fig.savefig(OUT / "fig2_novelty_judgment.pdf")
    plt.close(fig)


def _draw_box(ax, x, y, w, h, text, facecolor, edgecolor, textcolor="#0f172a", fontsize=7.0, bold=False, ls="-", lw=1.0):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.008,rounding_size=0.018",
        facecolor=facecolor, edgecolor=edgecolor, linewidth=lw, linestyle=ls, zorder=3,
    )
    ax.add_patch(patch)
    weight = "bold" if bold else "normal"
    ax.text(
        x + w / 2, y + h / 2, text,
        ha="center", va="center", fontsize=fontsize, color=textcolor, fontweight=weight, zorder=4,
        linespacing=1.2,
    )


def _draw_arrow(ax, x0, y0, x1, y1, color="#334155", ls="-", lw=1.05, rad=0.0):
    conn = f"arc3,rad={rad}"
    arr = FancyArrowPatch(
        (x0, y0), (x1, y1),
        arrowstyle="-|>,head_length=3.6,head_width=2.2",
        connectionstyle=conn,
        color=color, linestyle=ls, linewidth=lw, zorder=5,
    )
    ax.add_patch(arr)


def make_fig3_game_loop() -> None:
    """Schematic: Training plays the whole game; game day plays one move."""
    fig, ax = plt.subplots(figsize=(7.0, 2.3))
    fig.subplots_adjust(left=0.005, right=0.995, bottom=0.005, top=0.995)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    # Top container: Game day (blue)
    top_bg = FancyBboxPatch(
        (0.015, 0.56), 0.97, 0.41,
        boxstyle="round,pad=0.008,rounding_size=0.022",
        facecolor="#eff6ff", edgecolor="#3b82f6", linewidth=1.05, zorder=1,
    )
    ax.add_patch(top_bg)
    ax.text(
        0.03, 0.945,
        "Game day (search = 1, $B = 1$): play one move, no retry or repair",
        fontsize=9.1, fontweight="bold", color="#1e3a8a", va="top",
    )

    # Top row boxes
    y_top, h_top = 0.60, 0.25
    _draw_box(ax, 0.03, y_top, 0.195, h_top, "Board $p$\n$z_p = \\phi_{\\mathrm{P}}(p)$", "#ffffff", "#2563eb", "#1e3a8a", fontsize=8.5)
    _draw_box(ax, 0.27, y_top, 0.225, h_top, "Writer $\\pi_\\theta$\n1 draw $w \\sim \\pi_\\theta$", "#dbeafe", "#1d4ed8", "#1e3a8a", fontsize=8.5, bold=True)
    _draw_box(ax, 0.54, y_top, 0.195, h_top, "Fit $w$ on\ntraining rows", "#ffffff", "#2563eb", "#1e3a8a", fontsize=8.5)
    _draw_box(ax, 0.78, y_top, 0.19, h_top, "Predict $\\hat{y}$\n(1 verdict)", "#ffffff", "#2563eb", "#1e3a8a", fontsize=8.5)

    _draw_arrow(ax, 0.225, y_top + h_top / 2, 0.27, y_top + h_top / 2, color="#1d4ed8", lw=1.1)
    _draw_arrow(ax, 0.495, y_top + h_top / 2, 0.54, y_top + h_top / 2, color="#1d4ed8", lw=1.1)
    _draw_arrow(ax, 0.735, y_top + h_top / 2, 0.78, y_top + h_top / 2, color="#1d4ed8", lw=1.1)

    # Bottom dashed container: Training only
    bot_bg = FancyBboxPatch(
        (0.015, 0.025), 0.97, 0.44,
        boxstyle="round,pad=0.008,rounding_size=0.022",
        facecolor="#f8fafc", edgecolor="#64748b", linewidth=1.05, linestyle="--", zorder=1,
    )
    ax.add_patch(bot_bg)
    ax.text(
        0.50, 0.44,
        "Training only: self-play & expert iteration",
        fontsize=8.6, fontweight="bold", color="#334155", va="top", ha="center",
    )

    y_bot, h_bot = 0.055, 0.295
    _draw_box(ax, 0.03, y_bot, 0.285, h_bot, "Player ratchet\nTrain challenger; admit on\n16 cuts $\\times$ 3 seeds", "#ffffff", "#64748b", "#1e293b", fontsize=8.0)
    _draw_box(ax, 0.36, y_bot, 0.28, h_bot, "Board ratchet & lists\nStrict win updates $\\mathrm{inc}(p)$;\nvalid scored moves join $G_p$", "#ffffff", "#64748b", "#1e293b", fontsize=8.0)
    _draw_box(ax, 0.685, y_bot, 0.285, h_bot, "Referee $\\mathcal{R}$ & opponent\n5 inner folds vs. $\\mathrm{inc}(p)$\nOrder: $\\tilde{w} < w^- < w^+$", "#ffffff", "#64748b", "#1e293b", fontsize=8.0)

    # Arrows connecting top to bottom (passing outside the centered header)
    _draw_arrow(ax, 0.72, y_top, 0.85, y_bot + h_bot, color="#475569", ls="--", lw=1.05)
    ax.text(0.84, 0.50, "evaluate moves", fontsize=7.8, color="#475569", fontstyle="italic")
    _draw_arrow(ax, 0.685, y_bot + h_bot / 2, 0.64, y_bot + h_bot / 2, color="#475569", ls="--", lw=1.05)
    _draw_arrow(ax, 0.36, y_bot + h_bot / 2, 0.315, y_bot + h_bot / 2, color="#475569", ls="--", lw=1.05)
    _draw_arrow(ax, 0.14, y_bot + h_bot, 0.29, y_top, color="#0f766e", ls="--", lw=1.1)
    ax.text(0.055, 0.50, "admitted $\\theta$", fontsize=8.0, color="#0f766e", fontweight="bold")

    fig.savefig(OUT / "fig3_game_loop.pdf")
    plt.close(fig)


def make_fig5_network() -> None:
    """Schematic: Writer pi_theta and separate ranking map v_psi in step-60,000 checkpoint."""
    fig, ax = plt.subplots(figsize=(7.0, 2.6))
    fig.subplots_adjust(left=0.005, right=0.995, bottom=0.005, top=0.995)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    # Top container: Game day (search = 1)
    top_bg = FancyBboxPatch(
        (0.015, 0.64), 0.97, 0.335,
        boxstyle="round,pad=0.008,rounding_size=0.022",
        facecolor="#eff6ff", edgecolor="#2563eb", linewidth=1.05, zorder=1,
    )
    ax.add_patch(top_bg)
    ax.text(
        0.03, 0.955,
        "Game day (search = 1): frozen encoder + single writer draw (map $v_\\psi$ absent)",
        fontsize=8.6, fontweight="bold", color="#1e3a8a", va="top",
    )

    y_top, h_top = 0.67, 0.215
    _draw_box(ax, 0.03, y_top, 0.295, h_top, "Frozen encoder $\\phi_{\\mathrm{P}}(p)$\n2,048 values (512+1,024+512)\n30 s cap; hash-pinned", "#ffffff", "#2563eb", "#1e3a8a", fontsize=7.6)
    _draw_box(ax, 0.365, y_top, 0.32, h_top, "Program writer $\\pi_\\theta(w \\mid z_p)$\n6-block causal Transformer\n$\\leq 128$ grammar tokens ($T=1$)", "#dbeafe", "#1d4ed8", "#1e3a8a", fontsize=7.6, bold=True)
    _draw_box(ax, 0.725, y_top, 0.245, h_top, "1 program $w$\nLower to Python/Rust,\nfit & predict $\\hat{y}$", "#ffffff", "#2563eb", "#1e3a8a", fontsize=7.6)

    _draw_arrow(ax, 0.325, y_top + h_top / 2, 0.365, y_top + h_top / 2, color="#1d4ed8", lw=1.1)
    _draw_arrow(ax, 0.685, y_top + h_top / 2, 0.725, y_top + h_top / 2, color="#1d4ed8", lw=1.1)

    # Bottom dashed container: Training only
    bot_bg = FancyBboxPatch(
        (0.015, 0.02), 0.97, 0.55,
        boxstyle="round,pad=0.008,rounding_size=0.022",
        facecolor="#f8fafc", edgecolor="#64748b", linewidth=1.05, linestyle="--", zorder=1,
    )
    ax.add_patch(bot_bg)
    ax.text(
        0.66, 0.545,
        "Training only: 27.8M reference checkpoint",
        fontsize=8.2, fontweight="bold", color="#334155", va="top", ha="center",
    )

    # Row 1 inside training box: Search -> Referee -> Scored list G_p
    y_mid, h_mid = 0.295, 0.195
    _draw_box(ax, 0.03, y_mid, 0.275, h_mid, "64 writer drafts + search\nProposes candidates;\nstillborn excluded", "#ffffff", "#64748b", "#1e293b", fontsize=7.5)
    _draw_box(ax, 0.355, y_mid, 0.28, h_mid, "Referee $\\mathcal{R}$ (5 inner folds)\nScores valid moves vs. $\\mathrm{inc}(p)$;\nbuilds list $G_p$ ($\\geq 8$ moves)", "#ffffff", "#64748b", "#1e293b", fontsize=7.5)
    _draw_box(ax, 0.675, y_mid, 0.295, h_mid, "Map $v_\\psi(z_p, z_w)$\nProgram embed $z_w = \\phi_{\\mathrm{S}}(w)$;\n4-head scorer ($\\mathcal{L}_{\\mathrm{map}} + \\mathcal{L}_{\\mathrm{BT}}$)", "#fef3c7", "#d97706", "#78350f", fontsize=7.5)

    # Arrow from writer to 64 writer drafts (cleanly left of title)
    _draw_arrow(ax, 0.38, y_top, 0.16, y_mid + h_mid, color="#475569", ls="--", lw=1.0)
    _draw_arrow(ax, 0.305, y_mid + h_mid / 2, 0.355, y_mid + h_mid / 2, color="#475569", ls="--", lw=1.0)
    _draw_arrow(ax, 0.635, y_mid + h_mid / 2, 0.675, y_mid + h_mid / 2, color="#b45309", ls="--", lw=1.0)

    # Row 2 inside training box: Writer update + explicit non-steering boundary note
    y_low, h_low = 0.045, 0.20
    _draw_box(ax, 0.03, y_low, 0.455, h_low, "Writer supervision $\\mathcal{L}_{\\mathrm{writer}}$\nWeights each sampled move by $0.05 + \\rho_w^4$\ndirectly from referee percentiles $\\rho_w$", "#ecfdf5", "#059669", "#065f46", fontsize=7.5)
    _draw_box(ax, 0.515, y_low, 0.455, h_low, "Reference checkpoint boundary:\n$v_\\psi$ ranks drafts offline (vs. program-only control);\nnever screens, conditions, or steers $\\pi_\\theta$", "#fff1f2", "#e11d48", "#881337", fontsize=7.3, ls="--")

    # Arrow from Referee down to Writer supervision, and from Writer supervision straight up through vertical gap x=0.33 to Writer
    _draw_arrow(ax, 0.45, y_mid, 0.39, y_low + h_low, color="#059669", ls="--", lw=1.0)
    _draw_arrow(ax, 0.33, y_low + h_low, 0.37, y_top, color="#059669", ls="--", lw=1.05, rad=0.05)

    fig.savefig(OUT / "fig5_network.pdf")
    plt.close(fig)


if __name__ == "__main__":
    make_fig2_novelty_judgment()
    make_fig3_game_loop()
    make_fig5_network()
    print("schematics ok")
