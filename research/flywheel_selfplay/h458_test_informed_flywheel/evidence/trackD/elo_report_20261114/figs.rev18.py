"""Figures of the Elo report. One function per figure; every figure is saved as a 300-dpi PNG in figs/."""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__)) + "/"
T = json.load(open(HERE + "report_tables.json"))
OUT = HERE + "figs/"
os.makedirs(OUT, exist_ok=True)

# palette (validated: blue, orange, aqua, yellow pass the adjacent checks on a white surface; labels carry identity too)
BLUE, ORANGE, AQUA, YELLOW, RED = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e34948"
BLUE_LIGHT = "#9ec5f4"
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#ffffff"
EXT = "#898781"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": AXIS, "axes.linewidth": 0.6, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "axes.titlesize": 9.5, "axes.titleweight": "bold",
    "axes.titlecolor": INK, "axes.titlelocation": "left", "legend.frameon": False, "legend.fontsize": 7.5,
})
W = 7.3


def save(fig, name, tight=True):
    if tight:
        fig.savefig(OUT + name + ".png", dpi=300, bbox_inches="tight", pad_inches=0.06)
    else:
        fig.savefig(OUT + name + ".png", dpi=300)
    plt.close(fig)
    print("saved", name)


def style_axis(ax, grid="x"):
    ax.grid(axis=grid, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def marker_kw(kind):
    if kind == "external":
        return dict(marker="o", ms=5.2, mfc=EXT, mec=SURFACE, mew=0.9)
    if kind == "ours_incorpus":
        return dict(marker="o", ms=6.2, mfc=BLUE, mec=SURFACE, mew=0.9)
    if kind == "ours_fair":
        return dict(marker="o", ms=5.4, mfc=SURFACE, mec=BLUE, mew=1.7)
    return dict(marker="o", ms=5.2, mfc=BLUE_LIGHT, mec=SURFACE, mew=0.9)


LEGEND = [
    Line2D([], [], ls="", label="Ours, in-corpus (dev91 in training)", **marker_kw("ours_incorpus")),
    Line2D([], [], ls="", label="Ours, fair closure (0% dev91 in training)", **marker_kw("ours_fair")),
    Line2D([], [], ls="", label="Ours, earlier waves (frozen board)", **marker_kw("ours_earlier")),
    Line2D([], [], ls="", label="External model", **marker_kw("external")),
]


# ----------------------------------------------------------------------------- 1. the ladder
def fig_ladder():
    rows = T["ladder"]
    n = len(rows)
    fig, ax = plt.subplots(figsize=(W, 9.9))
    y = np.arange(n)[::-1]
    for yi, r in zip(y, rows):
        ax.plot([r["lo"], r["hi"]], [yi, yi], color=AXIS, lw=0.9, solid_capstyle="round", zorder=1)
        ax.plot([r["elo"]], [yi], ls="", zorder=3, **marker_kw(r["kind"]))
        ours = r["kind"] != "external"
        ax.text(985, yi, f"{r['pos']:>2}", ha="right", va="center", fontsize=6.6, color=MUTED)
        ax.text(1000, yi, r["name"], ha="left", va="center", fontsize=6.6, color=INK if ours else INK2, fontweight="bold" if r["kind"] in ("ours_incorpus", "ours_fair") else "normal")
        ax.text(2452, yi, f"{r['elo']:.0f}", ha="right", va="center", fontsize=6.8, color=INK, fontweight="bold" if r["kind"] in ("ours_incorpus", "ours_fair") else "normal")
    for k in range(5, n, 5):
        ax.axhline(n - k - 0.5, color=GRID, lw=0.5, zorder=0)
    ax.axhline(n - 50 - 0.5, color=INK2, lw=0.7, zorder=0)
    ax.text(2452, n - 50 - 0.62, "top 50 above this line", ha="right", va="top", fontsize=6.2, color=INK2)
    ax.set_xlim(960, 2460)
    ax.set_ylim(-1, n + 0.2)
    ax.set_yticks([])
    ax.set_xticks([1700, 1800, 1900, 2000, 2100, 2200, 2300])
    ax.xaxis.set_ticks_position("top")
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["top"].set_visible(True)
    ax.spines["top"].set_bounds(1690, 2360)
    ax.tick_params(length=0)
    for x in (1700, 1800, 1900, 2000, 2100, 2200, 2300):
        ax.plot([x, x], [-0.6, n - 0.4], color=GRID, lw=0.5, zorder=0)
    ax.text(2025, n + 2.35, "Elo (dot) with 95% bootstrap interval (line); only the part above 1690 is drawn to scale", ha="center", va="bottom", fontsize=6.6, color=INK2)
    ax.legend(handles=LEGEND, loc="lower left", bbox_to_anchor=(0.0, -0.045), ncol=4, handletextpad=0.2, columnspacing=1.2, fontsize=6.9)
    # clip intervals that reach below the label column
    for line in ax.lines:
        xd = line.get_xdata()
        if len(xd) == 2 and xd[0] < 1690:
            line.set_xdata([max(xd[0], 1690), max(xd[1], 1690)])
    save(fig, "f01_ladder")


# ----------------------------------------------------------------------------- 1b. the ladder, all on one true axis (compact)
def fig_ladder_axis():
    rows = T["ladder"]
    n = len(rows)
    fig = plt.figure(figsize=(W, 8.5))
    ax = fig.add_axes([0.43, 0.056, 0.555, 0.925])
    y = np.arange(n)[::-1]
    for yi, r in zip(y, rows):
        ours = r["kind"] in ("ours_incorpus", "ours_fair")
        ax.plot([r["lo"], r["hi"]], [yi, yi], color=AXIS, lw=0.9, solid_capstyle="round", zorder=1)
        ax.plot([r["elo"]], [yi], ls="", zorder=3, **marker_kw(r["kind"]))
        ax.text(2475, yi, f"{r['elo']:.0f}", ha="right", va="center", fontsize=6.4, color=INK, fontweight="bold" if ours else "normal")
    ax.set_yticks(y)
    import re as _re
    brief = lambda nm: _re.sub(r"\(mean of (\d+) (?:checkpoints|scorings|epochs)\)", r"(mean of \1)", nm)  # noqa: E731
    ax.set_yticklabels([f"{r['pos']}  {brief(r['name'])}" for r in rows], fontsize=6.4)
    for lab, r in zip(ax.get_yticklabels(), rows):
        lab.set_color(INK if r["kind"] != "external" else INK2)
        lab.set_fontweight("bold" if r["kind"] in ("ours_incorpus", "ours_fair") else "normal")
    style_axis(ax, "x")
    ax.set_ylim(-0.8, n - 0.2)
    ax.set_xlim(960, 2485)
    ax.set_xticks([1000, 1200, 1400, 1600, 1800, 2000, 2200])
    ax.set_xlabel("Elo (anchor: H257 replica = 1500); line = 95% bootstrap interval", fontsize=6.8)
    ax.axhline(n - 50 - 0.5, color=INK2, lw=0.7)
    ax.text(1530, n - 50 - 0.72, "top 50 above this line", ha="left", va="top", fontsize=6.2, color=INK2)
    ax.legend(handles=LEGEND, loc="lower right", bbox_to_anchor=(0.93, 0.012), fontsize=6.8)
    save(fig, "f01_ladder")  # tight box: long row labels must never be cut at the figure edge


# ----------------------------------------------------------------------------- 2. every scored arm of ours, by kind
def fig_strip():
    groups = ["Full stack (GP/tree pool + ICL)", "r4 champion stack", "Neural columns (incl. GP-residual hybrid)", "GP alone", "ICL alone"]
    fig, ax = plt.subplots(figsize=(W, 3.4))
    rng = np.random.default_rng(458)
    labels = []
    for gi, g in enumerate(groups):
        xs = np.array([u["elo"] for u in T["all_unique"] if u["group"] == g])
        fair = np.array([u["closure"] == "fair" for u in T["all_unique"] if u["group"] == g])
        yy = len(groups) - 1 - gi + rng.uniform(-0.24, 0.24, len(xs))
        ax.plot(xs[~fair], yy[~fair], ls="", marker="o", ms=3.4, mfc=BLUE, mec=SURFACE, mew=0.4, alpha=0.75, zorder=3)
        ax.plot(xs[fair], yy[fair], ls="", marker="o", ms=3.4, mfc=SURFACE, mec=BLUE, mew=0.9, zorder=4)
        labels.append(f"{g}\n{len(xs)} scores, {xs.min():.0f} to {xs.max():.0f}")
    refs = [("kumo_tabular_large_e8.v1", "Kumo Large"), ("tabfm_ensemble.v1", "TabFM ens."), ("tabpfn_v3p5.v1", "TabPFN 3.5"), ("tabpfn_v3.v1", "TabPFN v3"),
            ("tabicl_v2.v1", "TabICL v2")]
    elo = {r["id"]: r["elo"] for r in T["field_rows"]}
    for k, lab in refs:
        ax.axvline(elo[k], color=MUTED, lw=0.8, zorder=1)
        ax.text(elo[k], len(groups) - 0.42, f"{lab}\n{elo[k]:.0f}", ha="center", va="bottom", fontsize=6.6, color=INK2, linespacing=1.25)
    ax.set_ylim(-0.6, len(groups) - 0.45)
    ax.set_xlim(1550, 2100)
    ax.set_yticks(range(len(groups))[::-1])
    ax.set_yticklabels(labels, fontsize=7, color=INK, linespacing=1.35)
    style_axis(ax, "x")
    ax.set_xlabel("Elo on dev91-audit-v1 (each dot is one scored arm of ours; hollow = fair)")
    save(fig, "f02_strip")


# ----------------------------------------------------------------------------- 3. fair against in-corpus
def fig_pairs():
    pairs = T["pairs"]
    fig, ax = plt.subplots(figsize=(W, 2.5))
    y = np.arange(len(pairs))[::-1]
    for yi, p in zip(y, pairs):
        ax.plot([p["fair"], p["in_corpus"]], [yi, yi], color=AXIS, lw=1.4, zorder=1, solid_capstyle="round")
        ax.plot([p["fair"]], [yi], ls="", zorder=3, **marker_kw("ours_fair"))
        ax.plot([p["in_corpus"]], [yi], ls="", zorder=3, **marker_kw("ours_incorpus"))
        ax.text(p["in_corpus"] + 5, yi, f"+{p['delta']:.1f} Elo", ha="left", va="center", fontsize=7.3, color=INK, fontweight="bold")
        ax.text(p["fair"] - 5, yi, f"{p['fair']:.0f}", ha="right", va="center", fontsize=7, color=INK2)
    ax.set_yticks(y)
    ax.set_yticklabels([p["label"] for p in pairs], fontsize=7.4, color=INK)
    ax.set_xlim(1640, 1900)
    ax.set_ylim(-0.6, len(pairs) - 0.4)
    style_axis(ax, "x")
    ax.set_xlabel("Elo on dev91-audit-v1")
    ax.legend(handles=LEGEND[:2], loc="lower left", bbox_to_anchor=(0.01, 0.0), fontsize=7)
    save(fig, "f03_pairs")


# ----------------------------------------------------------------------------- 4. Ultra-GP generations
def fig_ugp():
    gens = [10, 50, 100, 250, 500, 1000]
    pos = {g: i for i, g in enumerate(gens)}
    fig, ax = plt.subplots(figsize=(W, 3.3))
    series = [("Stack + Ultra-GP (in-corpus)", BLUE), ("Fair stack + Ultra-GP", AQUA), ("Ultra-GP alone", ORANGE)]
    ends = {}
    for name, col in series:
        pts = [u for u in T["ugp"] if u["series"] == name]
        by = {}
        for u in pts:
            by.setdefault(u["gen"], []).append(u["elo"])
            ax.plot([pos[u["gen"]]], [u["elo"]], ls="", marker="o", ms=3.6, mfc=col, mec=SURFACE, mew=0.5, alpha=0.55, zorder=2)
        gs = sorted(by)
        best = [max(by[g]) for g in gs]
        ax.plot([pos[g] for g in gs], best, color=col, lw=2, marker="o", ms=5.5, mfc=col, mec=SURFACE, mew=1.2, zorder=3, solid_capstyle="round")
        ends[name] = (pos[gs[-1]], best[-1])
    for name, (x, yv) in ends.items():
        ax.text(x + 0.12, yv, name.replace(" (in-corpus)", ""), ha="left", va="center", fontsize=7.2, color=INK)
    ax.axhline(T["ugp_base"]["stack_no_ugp"], color=MUTED, lw=0.9, zorder=1)
    ax.text(-0.42, T["ugp_base"]["stack_no_ugp"] + 0.9, f"stack with no Ultra-GP: {T['ugp_base']['stack_no_ugp']:.0f}", ha="left", va="bottom", fontsize=6.8, color=INK2)
    ax.axhline(T["ugp_base"]["fair_no_ugp"], color=MUTED, lw=0.9, zorder=1)
    ax.text(-0.42, T["ugp_base"]["fair_no_ugp"] - 0.9, f"fair stack with no Ultra-GP: {T['ugp_base']['fair_no_ugp']:.0f}", ha="left", va="top", fontsize=6.8, color=INK2)
    ax.set_xticks(range(len(gens)))
    ax.set_xticklabels([f"{g:,}" for g in gens])
    ax.set_xlim(-0.5, len(gens) + 0.55)
    ax.set_ylim(1802, 1868)
    style_axis(ax, "y")
    ax.set_xlabel("Ultra-GP generations at inference (line: best admission variant; small dots: the other variants)")
    ax.set_ylabel("Elo")
    save(fig, "f04_ugp")


# ----------------------------------------------------------------------------- 5. r4 epochs
def fig_traj():
    tr = T["traj"]
    ep = [t["epoch"] for t in tr]
    fig, ax = plt.subplots(figsize=(W, 3.3))
    series = [("legacy", "Legacy guard stack", BLUE), ("champion", "r4 champion stack", ORANGE), ("no_gp", "Neural + tabular, no GP", AQUA), ("icl_only", "ICL only", YELLOW)]
    for key, name, col in series:
        ys = [t[key] for t in tr]
        ax.plot(ep, ys, color=col, lw=2, solid_capstyle="round", solid_joinstyle="round", zorder=3)
        ax.plot([ep[-1]], [ys[-1]], ls="", marker="o", ms=5.5, mfc=col, mec=SURFACE, mew=1.2, zorder=4)
        ax.text(ep[-1] + 0.8, ys[-1], f"{name}  {ys[-1]:.0f}", ha="left", va="center", fontsize=7.2, color=INK)
    ax.set_xlim(-1, 62)
    ax.set_xticks([0, 10, 20, 30, 40, ep[-1]])
    ax.set_ylim(1640, 1872)
    style_axis(ax, "y")
    ax.set_xlabel(f"r4 training epoch ({len(tr)} scorings: the init and {len(tr) - 1} epochs; 48 h of training)")
    ax.set_ylabel("Elo")
    ax.spines["bottom"].set_bounds(0, ep[-1])
    save(fig, "f05_r4")


# ----------------------------------------------------------------------------- 6. H455 checkpoints
def fig_ckpt():
    ck = [c for c in T["checkpoints"] if "stack" in c]
    fig, axes = plt.subplots(1, 3, figsize=(W, 3.1), sharey=True, gridspec_kw={"wspace": 0.08})
    y = np.arange(len(ck))[::-1]
    for ax, (key, title) in zip(axes, (("stack", "Full stack"), ("neural", "Neural (incl. GP-resid.)"), ("icl", "ICL alone"))):
        xs = [c.get(key) for c in ck]
        lo, hi = min(xs), max(xs)
        for yi, x in zip(y, xs):
            ax.plot([lo - 3, x], [yi, yi], color=GRID, lw=0.8, zorder=1)
            ax.plot([x], [yi], ls="", zorder=3, **marker_kw("ours_incorpus"))
        ax.set_title(f"{title}\n{lo:.0f} to {hi:.0f}", fontsize=8, fontweight="bold", loc="left", linespacing=1.4)
        ax.set_xlim(lo - 3, hi + 3)
        ax.set_ylim(-0.6, len(ck) - 0.4)
        style_axis(ax, "x")
        ax.xaxis.set_major_locator(plt.MaxNLocator(4, integer=True))
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([c["label"] for c in ck], fontsize=6.8, color=INK)
    axes[1].set_xlabel("Elo (each panel has its own 10 to 15 Elo wide axis; the three levels are 30 and 150 Elo apart)")
    save(fig, "f06_ckpt")


# ----------------------------------------------------------------------------- 7. family heat map (rank among the 33 field entrants)
def fig_heat():
    h = T["heat"]
    rank = np.array(h["rank"], dtype=float)
    cmap = LinearSegmentedColormap.from_list("blue_seq", ["#0d366b", "#256abf", "#6da7ec", "#cde2fb", "#f4f8fd"])
    fig, ax = plt.subplots(figsize=(W, 5.0))
    im = ax.imshow(rank, cmap=cmap, vmin=1, vmax=33, aspect="auto")
    ax.set_xticks(range(len(h["families"])))
    ax.set_xticklabels(h["families"], rotation=90, fontsize=6.3, family="DejaVu Sans Mono")
    ax.set_yticks(range(len(h["entrants"])))
    ax.set_yticklabels([e["name"] for e in h["entrants"]], fontsize=6.9)
    for lab, e in zip(ax.get_yticklabels(), h["entrants"]):
        lab.set_color(INK)
        lab.set_fontweight("bold" if e["id"].startswith(("h448", "ours.")) else "normal")
    for i in range(rank.shape[0]):
        for j in range(rank.shape[1]):
            if rank[i, j] <= 3:
                ax.text(j, i, f"{int(rank[i, j])}", ha="center", va="center", fontsize=6.6, color="#ffffff", fontweight="bold")
    ax.set_xticks(np.arange(-0.5, rank.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, rank.shape[0], 1), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=1.4)
    ax.tick_params(which="both", length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.015, ticks=[1, 10, 20, 33])
    cb.ax.invert_yaxis()
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=0, labelsize=6.5)
    cb.set_label("rank among the 33 field entrants (1 = best)", fontsize=6.6, color=INK2)
    ax.set_xlabel("dataset family (first 8 hex digits of its id)", fontsize=7)
    save(fig, "f07_heat")


# ----------------------------------------------------------------------------- 8. win / tie / loss
def wtl_bars(ax, labels, wtl, total, bold=()):
    y = np.arange(len(labels))[::-1]
    for yi, (w, t, l) in zip(y, wtl):
        x0 = 0
        for val, col in ((w, BLUE), (t, AXIS), (l, RED)):
            if val > 0:
                ax.barh(yi, val - 0.012 * total, left=x0 + 0.006 * total, height=0.56, color=col, zorder=3, linewidth=0)
            x0 += val
        ax.text(-0.015 * total, yi, f"{w}", ha="right", va="center", fontsize=7, color=INK, fontweight="bold")
        ax.text(total * 1.015, yi, f"{l}", ha="left", va="center", fontsize=7, color=INK2)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=6.9, color=INK)
    for lab, name in zip(ax.get_yticklabels(), labels):
        if name in bold:
            lab.set_fontweight("bold")
    ax.set_xlim(-0.07 * total, total * 1.07)
    ax.set_ylim(-0.7, len(labels) - 0.3)
    ax.set_xticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0, pad=16)
    ax.plot([total / 2, total / 2], [-0.6, len(labels) - 0.4], color=INK2, lw=0.6, zorder=4)


def fig_league():
    """External models only: games won of 418 by each of the 22 external entrants and by our three headline systems, each of ours rated alone."""
    lg = T["league"]
    rows = [{"name": r["name"], "games": r["games"], "kind": "ext"} for r in lg["best"]["table"] if not r["ours"]]
    tags = {"best": "Our best arm, best checkpoint (in-corpus)", "fairc": "Our best fair-closure arm (post hoc)", "fair": "Our fair champion H448 (pre-registered)"}
    for k in ("best", "fairc", "fair"):
        rows.append({"name": tags[k], "games": lg[k]["games"], "kind": "ours_fair" if k != "best" else "ours"})
    if "recall" in lg:
        rows.append({"name": "With family recall specialists (IN-CORPUS RECALL)", "games": lg["recall"]["games"], "kind": "recall"})
    rows.sort(key=lambda r: -r["games"])
    of = lg["best"]["of"]
    fig, ax = plt.subplots(figsize=(W, 4.6))
    y = np.arange(len(rows))[::-1]
    for yi, r in zip(y, rows):
        if r["kind"] == "ext":
            ax.barh(yi, r["games"], height=0.58, color=AXIS, zorder=3, linewidth=0)
        elif r["kind"] == "ours":
            ax.barh(yi, r["games"], height=0.58, color=BLUE, zorder=3, linewidth=0)
        elif r["kind"] == "recall":
            ax.barh(yi, r["games"], height=0.58, color=BLUE, zorder=3, linewidth=0, hatch="////", edgecolor=SURFACE)
        else:
            ax.barh(yi, r["games"], height=0.58, color=SURFACE, edgecolor=BLUE, zorder=3, linewidth=1.3)
        ax.text(r["games"] + of * 0.008, yi, f"{r['games']:.0f}", ha="left", va="center", fontsize=6.8, color=INK, fontweight="bold" if r["kind"] != "ext" else "normal")
    ax.set_yticks(y)
    ax.set_yticklabels([r["name"] for r in rows], fontsize=6.9, color=INK)
    for lab, r in zip(ax.get_yticklabels(), rows):
        if r["kind"] != "ext":
            lab.set_fontweight("bold")
    ax.set_xlim(0, of * 1.04)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.axvline(of / 2, color=INK2, lw=0.6, zorder=4)
    ax.text(of / 2, len(rows) - 0.25, "half of the games", ha="center", va="bottom", fontsize=6.4, color=INK2)
    style_axis(ax, "x")
    ax.spines["left"].set_visible(False)
    ax.set_xlabel(f"family-games won against the 22 external models (of {of}); each of our systems is rated alone against them", fontsize=7)
    save(fig, "f08_league")


def fig_wtl_ctx():
    rows = T["h2h"]
    titles = {"best": "Best arm, best checkpoint", "fairc": "Best fair-closure arm", "fair": "Fair champion H448"}
    subs = {"best": "in-corpus", "fairc": "selected post hoc", "fair": "pre-registered"}
    fig, axes = plt.subplots(1, 3, figsize=(W, 4.1), gridspec_kw={"wspace": 0.2})
    for ax, h in zip(axes, T["head"]):
        wtl_bars(ax, [r["name"] for r in rows], [r[h["key"]]["ctx_wtl"] for r in rows], 91)
        ax.set_title(f"{titles[h['key']]}\n{subs[h['key']]}, Elo {h['elo']:.0f}", fontsize=7.6)
        if ax is not axes[0]:
            ax.set_yticklabels([])
    handles = [Line2D([], [], ls="", marker="s", ms=7, mfc=BLUE, mec=SURFACE, label="contexts where our system is better (count at left)"),
               Line2D([], [], ls="", marker="s", ms=7, mfc=RED, mec=SURFACE, label="contexts where the external model is better (count at right)"),
               Line2D([], [], ls="", marker="s", ms=7, mfc=AXIS, mec=SURFACE, label="tie")]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=7, bbox_to_anchor=(0.55, -0.03))
    save(fig, "f09_wtl_ctx")


# ----------------------------------------------------------------------------- 10. context by context against Kumo Large
def fig_ctx():
    rows = T["ctx_delta"]
    head = {h["key"]: h for h in T["head"]}
    series = (("d_kumo_l_best", f"Best arm at its best checkpoint (in-corpus, Elo {head['best']['elo']:.0f})"),
              ("d_kumo_l", f"Fair champion H448 (pre-registered, Elo {head['fair']['elo']:.0f})"))
    fams = {}
    for r in rows:
        fams.setdefault(r["family"], []).append(r)
    order = sorted(fams, key=lambda f: np.mean([r["d_kumo_l_best"] for r in fams[f]]))
    fig, axes = plt.subplots(2, 1, figsize=(W, 5.0), sharex=True, sharey=True, gridspec_kw={"hspace": 0.2})
    for ax, (key, title) in zip(axes, series):
        x, ticks = 0, []
        for f in order:
            rs = sorted(fams[f], key=lambda r: r["d_kumo_l_best"])  # one context keeps its place in both panels
            vals = [r[key] for r in rs]
            xs = x + np.arange(len(vals))
            ax.bar(xs, vals, width=0.62, color=[BLUE if v > 0 else RED for v in vals], zorder=3, linewidth=0)
            ticks.append((x + (len(vals) - 1) / 2, f))
            x += len(vals) + 1.3
        ax.axhline(0, color=INK2, lw=0.7, zorder=4)
        ax.set_xlim(-1.2, x - 0.5)
        style_axis(ax, "y")
        ax.set_ylabel("utility minus Kumo Large", fontsize=7)
        wins = sum(1 for r in rows if r[key] > 1e-4)
        ax.set_title(f"{title}: better than Kumo Large on {wins} of 91 contexts", fontsize=8)
    axes[1].set_xticks([t for t, _ in ticks])
    axes[1].set_xticklabels([f for _, f in ticks], rotation=90, fontsize=6.3, family="DejaVu Sans Mono")
    axes[1].set_xlabel("dataset family (each bar is one context, at the same place in both panels; families ordered by the best arm's mean difference)", fontsize=6.8)
    handles = [Line2D([], [], ls="", marker="s", ms=7, mfc=BLUE, mec=SURFACE, label="we are better"),
               Line2D([], [], ls="", marker="s", ms=7, mfc=RED, mec=SURFACE, label="Kumo Large is better")]
    axes[0].legend(handles=handles, loc="lower right", ncol=1, fontsize=7)
    save(fig, "f10_ctx")


def fig_ctx_recall():
    """Context by context against Kumo Large: the stack without the recall columns, and with the family recall specialists (in-corpus recall)."""
    rows = T["ctx_delta"]
    head = {h["key"]: h for h in T["head"]}
    series = (("d_kumo_l_norecall", f"Stack without the recall columns (in-corpus, Elo {head['norecall']['elo']:.0f})"),
              ("d_kumo_l_recall", f"With family recall specialists (IN-CORPUS RECALL: audit rows trained on; Elo {head['recall']['elo']:.0f})"))
    fams = {}
    for r in rows:
        fams.setdefault(r["family"], []).append(r)
    order = sorted(fams, key=lambda f: np.mean([r["d_kumo_l_recall"] for r in fams[f]]))
    fig, axes = plt.subplots(2, 1, figsize=(W, 4.6), sharex=True, sharey=True, gridspec_kw={"hspace": 0.22})
    for ax, (key, title) in zip(axes, series):
        x, ticks = 0, []
        for f in order:
            rs = sorted(fams[f], key=lambda r: r["d_kumo_l_recall"])  # one context keeps its place in both panels
            vals = [r[key] for r in rs]
            xs = x + np.arange(len(vals))
            ax.bar(xs, vals, width=0.62, color=[BLUE if v > 0 else RED for v in vals], zorder=3, linewidth=0)
            ticks.append((x + (len(vals) - 1) / 2, f))
            x += len(vals) + 1.3
        ax.axhline(0, color=INK2, lw=0.7, zorder=4)
        ax.set_xlim(-1.2, x - 0.5)
        style_axis(ax, "y")
        ax.set_ylabel("utility minus Kumo Large", fontsize=7)
        wins = sum(1 for r in rows if r[key] > 1e-4)
        ax.set_title(f"{title}: better than Kumo Large on {wins} of 91 contexts", fontsize=7.6)
    axes[1].set_xticks([t for t, _ in ticks])
    axes[1].set_xticklabels([f for _, f in ticks], rotation=90, fontsize=6.3, family="DejaVu Sans Mono")
    axes[1].set_xlabel("dataset family (each bar is one context, at the same place in both panels; families ordered by the lower panel's mean difference)", fontsize=6.8)
    handles = [Line2D([], [], ls="", marker="s", ms=7, mfc=BLUE, mec=SURFACE, label="we are better"),
               Line2D([], [], ls="", marker="s", ms=7, mfc=RED, mec=SURFACE, label="Kumo Large is better")]
    axes[0].legend(handles=handles, loc="upper left", ncol=2, fontsize=7)
    save(fig, "f17_ctx_recall")


# ----------------------------------------------------------------------------- 11. what the external models cost
def fig_cost():
    rows = [r for r in T["field_rows"] if r["kind"] == "external" and r["wall_s"] and r["cls"] != "LLM regressor"]
    fig, ax = plt.subplots(figsize=(W, 3.5))
    off = {"Kumo Tabular Large (8 estimators)": (6, 0, "left"), "TabFM 1.0 ensemble (Google)": (-6, 5, "right"), "Kumo Tabular Small (8 estimators)": (-6, 4, "right"),
           "TabFM 1.0 (Google)": (22, -17, "left"), "TabPFN 3.5 (Prior Labs)": (7, 9, "left"), "TabPFN 3.5, 1 estimator (Prior Labs)": (-6, -8, "right"),
           "TabPFN v3 (Prior Labs)": (6, -2, "left"), "Mitra v2, zero-shot (AutoGluon)": (6, 3, "left"), "TabICL v2 (soda-inria)": (-6, 4, "right"),
           "TabDPT 1.3 (Layer6)": (-6, -6, "right"), "AutoGluon Tabular (default, 300 s)": (6, 0, "left"), "CatBoost (1000 x depth 6)": (-6, 5, "right"),
           "XGBoost (500 x depth 6)": (6, 4, "left"), "LightGBM (500 x 31 leaves)": (6, -5, "left"), "Random forest (300 trees)": (-6, -5, "right"), "Ridge (RidgeCV)": (-6, 0, "right")}
    for r in rows:
        fm = r["cls"] == "Tabular foundation model"
        ax.plot([r["wall_s"]], [r["elo"]], ls="", marker="o", ms=6, mfc=EXT if fm else SURFACE, mec=SURFACE if fm else EXT, mew=1.0 if fm else 1.6, zorder=3)
        dx, dy, ha = off.get(r["name"], (6, 0, "left"))
        far = abs(dx) > 12
        ax.annotate(r["name"].split(" (")[0], (r["wall_s"], r["elo"]), xytext=(dx, dy), textcoords="offset points", ha=ha, va="center", fontsize=6.8, color=INK,
                    arrowprops=dict(arrowstyle="-", color=AXIS, lw=0.7, shrinkA=1, shrinkB=3) if far else None)
    best = T["systems"][0]["elo"]
    ax.axhline(best, color=BLUE, lw=1.2, zorder=2)
    ax.set_xscale("log")
    ax.set_xlim(0.6, 400)
    ax.set_ylim(1200, 2130)
    ax.set_xticks([1, 3, 10, 30, 100, 300])
    ax.set_xticklabels(["1 s", "3 s", "10 s", "30 s", "100 s", "300 s"])
    ax.minorticks_off()
    style_axis(ax, "both")
    ax.set_xlabel("median wall time per context on the reference harness (log scale)")
    ax.set_ylabel("Elo")
    handles = [Line2D([], [], ls="", marker="o", ms=6, mfc=EXT, mec=SURFACE, label="tabular foundation model"),
               Line2D([], [], ls="", marker="o", ms=6, mfc=SURFACE, mec=EXT, mew=1.6, label="AutoML, trees, linear"),
               Line2D([], [], color=BLUE, lw=1.2, label=f"our best arm, {best:.0f} (in-corpus, mean of 12 checkpoints; wall time not recorded)")]
    ax.legend(handles=handles, loc="lower right", fontsize=7)
    save(fig, "f11_cost")


# ----------------------------------------------------------------------------- 12. H458 specialists (SEARCH-only probe, no audit look)
def fig_h458():
    p = json.load(open(HERE + "h458_probe.json"))
    fig, axes = plt.subplots(1, len(p["families"]), figsize=(W, 2.85), gridspec_kw={"wspace": 0.3})
    axes = np.atleast_1d(axes)
    for ax, f in zip(axes, p["families"]):
        for arm, col in (("canon", BLUE), ("rand", ORANGE)):
            xs = [0] + [m["exposures"] for m in f[arm]]
            ys = [f["init"][arm]] + [m["u"] for m in f[arm]]
            ax.plot(xs, ys, color=col, lw=2, marker="o", ms=4.6, mfc=col, mec=SURFACE, mew=1.0, zorder=3)
            ax.text(xs[-1] + 70, ys[-1], f"{ys[-1]:.2f}", ha="left", va="center", fontsize=7.2, color=INK, fontweight="bold")
        ax.axhline(f["field_best_u"], color=MUTED, lw=0.9, zorder=1)
        above = f["rand"][-1]["u"] < f["field_best_u"]
        ax.text(2480, f["field_best_u"], f"field best {f['field_best_u']:.2f}", ha="right", va="bottom" if above else "top", fontsize=6.3, color=INK2)
        ax.set_title(f"family {f['family']}", fontsize=8)
        ax.set_xlim(-80, 2500)
        ax.set_xticks([0, 500, 1000, 1500, 2000])
        style_axis(ax, "y")
        ax.set_xlabel("query exposures per pool row", fontsize=7)
    axes[0].set_ylabel("utility on held-out SEARCH rows\n(rows that were in the fine-tuning pool)", fontsize=6.8)
    handles = [Line2D([], [], color=BLUE, lw=2, marker="o", ms=4.6, mfc=BLUE, mec=SURFACE, label="specialist, canonical feature slots"),
               Line2D([], [], color=ORANGE, lw=2, marker="o", ms=4.6, mfc=ORANGE, mec=SURFACE, label="specialist, random feature slots")]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=7, bbox_to_anchor=(0.5, -0.1))
    save(fig, "f12_h458")


# ----------------------------------------------------------------------------- 13. the pilot look (audit rows, 20 contexts)
def fig_pilot():
    d = json.load(open(HERE + "pilot_look.json"))
    rows = d["rows"]
    fig, ax = plt.subplots(figsize=(W, 4.3))
    y, yt, yl, k = [], [], [], 0
    last = None
    for r in rows:
        if last is not None and r["family"] != last:
            k += 0.8
        last = r["family"]
        y.append(k)
        k += 1
    y = np.array(y)
    y = y.max() - y
    for yi, r in zip(y, rows):
        ax.plot([r["u_base"], r["u_arm"]], [yi, yi], color=AXIS, lw=1.2, zorder=1, solid_capstyle="round")
        ax.plot([r["u_base"]], [yi], ls="", marker="o", ms=5, mfc=SURFACE, mec=EXT, mew=1.5, zorder=3)
        ax.plot([r["u_arm"]], [yi], ls="", marker="o", ms=6, mfc=BLUE if r["win"] else RED, mec=SURFACE, mew=0.9, zorder=4)
        ax.plot([r["field_max"], r["field_max"]], [yi - 0.36, yi + 0.36], color=INK, lw=1.6, zorder=5, solid_capstyle="butt")
    fam_rows = {}
    for yi, r in zip(y, rows):
        fam_rows.setdefault(r["family"], []).append(yi)
    ax.set_yticks([np.mean(v) for v in fam_rows.values()])
    ax.set_yticklabels([f"{f}\n{sum(1 for r in rows if r['family'] == f and r['win'])} of {len(v)} above the field" for f, v in fam_rows.items()], fontsize=7, color=INK, linespacing=1.35)
    ax.set_xlim(0, 5.05)
    ax.set_ylim(-0.8, y.max() + 0.8)
    style_axis(ax, "x")
    ax.set_xlabel("utility on AUDIT rows (each row is one context)")
    handles = [Line2D([], [], ls="", marker="o", ms=5, mfc=SURFACE, mec=EXT, mew=1.5, label="legacy stack (BASE)"),
               Line2D([], [], ls="", marker="o", ms=6, mfc=BLUE, mec=SURFACE, label="stack + recall specialists, above the field"),
               Line2D([], [], ls="", marker="o", ms=6, mfc=RED, mec=SURFACE, label="stack + recall specialists, below the field"),
               Line2D([], [], color=INK, lw=1.6, label="best of the 33 field entrants")]
    ax.legend(handles=handles, loc="lower right", fontsize=7)
    save(fig, "f13_pilot")


def fig_look2():
    """H458 look 2 by family: the stack without recall columns, with them, and the best of the 33 field entrants."""
    L = T["look2"]
    rows = sorted(L["families"], key=lambda r: r["u_arm"] - r["field_max"])
    fig, ax = plt.subplots(figsize=(W, 5.0))
    y = np.arange(len(rows))
    for yi, r in zip(y, rows):
        ax.plot([r["u_base"], r["u_arm"]], [yi, yi], color=AXIS, lw=1.2, zorder=1, solid_capstyle="round")
        ax.plot([r["u_base"]], [yi], ls="", marker="o", ms=5, mfc=SURFACE, mec=EXT, mew=1.5, zorder=3)
        if r["swept"]:
            kw = dict(mfc=BLUE, mec=SURFACE, mew=0.9)
        elif r["above_field"] > 0:
            kw = dict(mfc=BLUE_LIGHT, mec=BLUE, mew=1.1)
        else:
            kw = dict(mfc=RED, mec=SURFACE, mew=0.9)
        ax.plot([r["u_arm"]], [yi], ls="", marker="o", ms=6.4, zorder=4, **kw)
        ax.plot([r["field_max"], r["field_max"]], [yi - 0.34, yi + 0.34], color=INK, lw=1.6, zorder=5, solid_capstyle="butt")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r['family']}{' ' if r['target'] else '*'}  {r['above_field']} of {r['n']}" for r in rows], fontsize=7, color=INK, fontfamily="DejaVu Sans Mono")
    ax.set_xlim(0, 5.75)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    style_axis(ax, "x")
    ax.set_xlabel("family mean utility on AUDIT rows; the count beside each family is contexts above the best of all 33 entrants")
    handles = [Line2D([], [], ls="", marker="o", ms=5, mfc=SURFACE, mec=EXT, mew=1.5, label="stack without the recall columns"),
               Line2D([], [], ls="", marker="o", ms=6.4, mfc=BLUE, mec=SURFACE, label="with recall specialists: above the field on every context"),
               Line2D([], [], ls="", marker="o", ms=6.4, mfc=BLUE_LIGHT, mec=BLUE, mew=1.1, label="with recall specialists: above the field on some contexts"),
               Line2D([], [], ls="", marker="o", ms=6.4, mfc=RED, mec=SURFACE, label="with recall specialists: above the field on none"),
               Line2D([], [], color=INK, lw=1.6, label="best of the 33 field entrants (mean of the per-context best)")]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=6.8, bbox_to_anchor=(0.52, -0.105), columnspacing=1.6, handletextpad=0.6)
    save(fig, "f14_look2")


def fig_look2_wtl():
    """Contexts won and lost against each external model: the stack without recall columns, and with them."""
    rows = T["h2h"]
    head = {h["key"]: h for h in T["head"]}
    fig, axes = plt.subplots(1, 2, figsize=(W, 4.1), gridspec_kw={"wspace": 0.12})
    for ax, key, title in ((axes[0], "norecall", f"Stack without the recall columns\nin-corpus, Elo {head['norecall']['elo']:.0f}"),
                           (axes[1], "recall", f"With family recall specialists\nIN-CORPUS RECALL (audit rows trained on), Elo {head['recall']['elo']:.0f}")):
        wtl_bars(ax, [r["name"] for r in rows], [r[key]["ctx_wtl"] for r in rows], 91)
        ax.set_title(title, fontsize=7.6)
        if ax is not axes[0]:
            ax.set_yticklabels([])
    handles = [Line2D([], [], ls="", marker="s", ms=7, mfc=BLUE, mec=SURFACE, label="contexts where our system is better (count at left)"),
               Line2D([], [], ls="", marker="s", ms=7, mfc=RED, mec=SURFACE, label="contexts where the external model is better (count at right)"),
               Line2D([], [], ls="", marker="s", ms=7, mfc=AXIS, mec=SURFACE, label="tie")]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=7, bbox_to_anchor=(0.55, -0.03))
    save(fig, "f15_look2_wtl")


def fig_freshprobe():
    """Fresh-row board: family specialist minus its own pre-fine-tune checkpoint, by family."""
    pr = T["look2"]["probe"]
    rows = pr["family_du"]
    fig, ax = plt.subplots(figsize=(W, 2.75))
    y = np.arange(len(rows))[::-1]
    for yi, r in zip(y, rows):
        ax.barh(yi, r["du"], height=0.56, color=BLUE if r["du"] > 0 else RED, zorder=3, linewidth=0)
        ax.text(r["du"] + (0.03 if r["du"] > 0 else -0.03), yi, f"{r['du']:+.2f}", ha="left" if r["du"] > 0 else "right", va="center", fontsize=7, color=INK)
    ax.axvline(0, color=INK2, lw=0.8, zorder=4)
    ax.set_yticks(y)
    ax.set_yticklabels([r["family"] for r in rows], fontsize=7, color=INK, fontfamily="DejaVu Sans Mono")
    ax.set_xlim(-0.6, 2.05)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    style_axis(ax, "x")
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("utility on fresh rows (in no bank and no training pool): specialist minus its own starting checkpoint, family mean")
    save(fig, "f16_freshprobe")


def fig_rev10_probe():
    """Ten contexts, fresh rows: each arm minus the same frozen starting model (family mean), with contexts won and lost."""
    rows = T["rev10"]["probe"]["vs_init"]
    fig, ax = plt.subplots(figsize=(W, 2.7))
    y = np.arange(len(rows))[::-1]
    for yi, r in zip(y, rows):
        pos = r["du"] > 0
        ax.barh(yi, r["du"], height=0.56, color=BLUE if pos else RED, zorder=3, linewidth=0)
        ax.text(max(r["du"], 0) + 0.008, yi, f"{r['du']:+.2f}   won {r['w']}, lost {r['l']}", ha="left", va="center", fontsize=7, color=INK)
    ax.axvline(0, color=INK2, lw=0.8, zorder=4)
    ax.set_yticks(y)
    ax.set_yticklabels([r["arm"] for r in rows], fontsize=7, color=INK)
    ax.set_xlim(-0.06, 0.62)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    style_axis(ax, "x")
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("utility on fresh rows minus the frozen starting model's, mean over ten contexts (one per family)")
    save(fig, "f17_rev10_probe")


if __name__ == "__main__":
    import sys

    todo = sys.argv[1:] or ["ladder", "strip", "pairs", "ugp", "traj", "ckpt", "heat", "league", "wtl_ctx", "ctx", "cost"]
    fn = {"ladder": fig_ladder_axis, "strip": fig_strip, "pairs": fig_pairs, "ugp": fig_ugp, "traj": fig_traj, "ckpt": fig_ckpt, "heat": fig_heat,
          "league": fig_league, "wtl_ctx": fig_wtl_ctx, "ctx": fig_ctx, "cost": fig_cost, "h458": fig_h458, "pilot": fig_pilot,
          "look2": fig_look2, "look2_wtl": fig_look2_wtl, "freshprobe": fig_freshprobe, "ctx_recall": fig_ctx_recall, "rev10_probe": fig_rev10_probe}
    for t in todo:
        fn[t]()
