#!/usr/bin/env python3
"""Build vector PDF figures from the derived flywheel CSVs in ../data.
Usage: python3 make_flywheel_figs.py   (run from any dir)"""
import csv, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

H = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(H, "..", "data")
O = os.path.join(H, "..")

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 6.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.grid": True,
    "grid.alpha": 0.22,
    "grid.linewidth": 0.4,
    "axes.linewidth": 0.6,
})

f = lambda x: float(x) if x not in ("", "None", None) else None
with open(f"{D}/rounds.csv") as fh:
    R = list(csv.DictReader(fh))
with open(f"{D}/admissions.csv") as fh:
    A = list(csv.DictReader(fh))

# 1. Validation trajectory (695 admission evaluations, rounds 775-1469)
r = [int(a["round"]) for a in A]
c = [f(a["candidate_val"]) for a in A]
b = [f(a["best_val"]) for a in A]
strict_gains = []
tie_adms = []
prev_best = None
for a in A:
    rd = int(a["round"])
    cv = f(a["candidate_val"])
    bv = f(a["best_val"])
    if a["accepted"] == "1":
        if prev_best is None or bv > prev_best + 1e-12:
            strict_gains.append((rd, cv))
        else:
            tie_adms.append((rd, cv))
    prev_best = bv

fig, ax = plt.subplots(figsize=(3.4, 2.15))
ax.scatter(r, c, s=4.5, c="#6e7781", alpha=0.55, lw=0, zorder=2, label="candidate adapter")
ax.step(r, b, where="post", c="#0b5394", lw=1.25, zorder=3, label="best recorded adapter")
if strict_gains:
    ax.scatter(*zip(*strict_gains), s=28, marker="*", c="#c92a2a", zorder=5,
               label="strict gain (785, 859)")
if tie_adms:
    ax.scatter(*zip(*tie_adms), s=16, marker="D", facecolors="none", edgecolors="#d9480f",
               lw=0.9, zorder=4, label="accepted tie (782, 882)")
ax.set_xlabel("round")
ax.set_ylabel("validation pool utility")
ax.set_xlim(min(r) - 10, max(r) + 10)
ax.legend(frameon=True, facecolor="white", framealpha=0.85, edgecolor="none",
          loc="lower right", handletextpad=0.3, borderpad=0.25)
fig.tight_layout(pad=0.4)
fig.savefig(f"{O}/fig_fw_validation.pdf")
plt.close(fig)

# 2. Train vs validation on the 25 retained summary rounds (broken x-axis for 1342-1345 and 1449-1469)
r1 = [x for x in R if int(x["round"]) <= 1345]
r2 = [x for x in R if int(x["round"]) >= 1449]
rr_all = [int(x["round"]) for x in R]
tv_all = [f(x["train"]) for x in R]
vv_all = [f(x["val"]) for x in R]
gaps = [t - v for t, v in zip(tv_all, vv_all)]

fig, (axL, axR) = plt.subplots(
    1, 2, figsize=(3.4, 2.15), sharey=True,
    gridspec_kw={"width_ratios": [5, 21], "wspace": 0.08}
)
for sub, rows in ((axL, r1), (axR, r2)):
    xs = [int(x["round"]) for x in rows]
    ts_y = [f(x["train"]) for x in rows]
    vs_y = [f(x["val"]) for x in rows]
    sub.plot(xs, ts_y, "o-", ms=2.6, lw=0.95, c="#d95f02", label="train pool utility")
    sub.plot(xs, vs_y, "s-", ms=2.6, lw=0.95, c="#0b5394", label="validation pool utility")
    sub.fill_between(xs, vs_y, ts_y, color="#d95f02", alpha=0.12, lw=0)

axL.set_xlim(1341.5, 1345.5)
axL.set_xticks([1342, 1345])
axR.set_xlim(1448.3, 1469.7)
axR.set_xticks([1450, 1455, 1460, 1465, 1469])
axL.set_ylim(1.10, 1.46)
axL.set_ylabel("pool utility")
axR.spines["left"].set_visible(False)
axR.tick_params(labelleft=False, left=False)

# Diagonal axis-break marks
d = 0.022
kwargs = dict(transform=axL.transAxes, color="0.3", clip_on=False, lw=0.7)
axL.plot((1 - d * 3, 1 + d * 3), (-d, +d), **kwargs)
kwargs.update(transform=axR.transAxes)
axR.plot((-d, +d), (-d, +d), **kwargs)

axR.text(0.04, 0.50, f"train–val gap\n{min(gaps):.3f}–{max(gaps):.3f}\n(n = {len(R)} rounds)",
         transform=axR.transAxes, fontsize=6.5,
         bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8))
axR.legend(frameon=True, facecolor="white", framealpha=0.85, edgecolor="none",
           loc="center right", handletextpad=0.3, borderpad=0.25)
fig.supxlabel("round (25 retained summary rounds; 1346–1448 not retained)", fontsize=7.5, y=0.02)
fig.subplots_adjust(left=0.16, right=0.98, bottom=0.20, top=0.96, wspace=0.08)
fig.savefig(f"{O}/fig_fw_train_val.pdf")
plt.close(fig)

# 3. Telemetry across the two summary windows and inter-admission intervals
ts = [(int(a["round"]), f(a["timestamp"])) for a in A if f(a["timestamp"])]
dt = [(rd2, t2 - t1) for (rd1, t1), (rd2, t2) in zip(ts, ts[1:]) if rd2 == rd1 + 1 and 0 < t2 - t1 < 600]
wall_pts = [(int(x["round"]), f(x["wall_s"])) for x in R if f(x["wall_s"]) is not None]

GAP_SHIFT = 98  # compresses the 103-round gap in retained summaries (1346-1448) to 5 display units
x_comp = lambda rd: rd if rd <= 1345 else rd - GAP_SHIFT
comp_ticks = [1343.5, 1449 - GAP_SHIFT, 1455 - GAP_SHIFT, 1462 - GAP_SHIFT, 1469 - GAP_SHIFT]
comp_labels = ["1342–45", "1449", "1455", "1462", "1469"]

def plot_compressed_series(ax, rows, key, color, ylabel, ylim):
    r_early = [x for x in rows if int(x["round"]) <= 1345]
    r_late = [x for x in rows if int(x["round"]) >= 1449]
    for idx, seg in enumerate((r_early, r_late)):
        sx = [x_comp(int(x["round"])) for x in seg]
        sy = [f(x[key]) for x in seg]
        ax.plot(sx, sy, "o-", ms=2.5, lw=0.9, c=color,
                label="logged summary (n=25)" if idx == 0 else None)
    ax.axvspan(1345.3, x_comp(1449) - 0.3, color="0.88", alpha=0.65, lw=0, label="1346–1448 not retained")
    ax.set_xlim(1341.2, x_comp(1469) + 0.8)
    ax.set_ylim(*ylim)
    ax.set_xticks(comp_ticks)
    ax.set_xticklabels(comp_labels)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("round (compressed gap)")
    ax.legend(frameon=True, facecolor="white", framealpha=0.88, edgecolor="none",
              loc="upper right", fontsize=7, handletextpad=0.3, borderpad=0.2)

fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.05))
plot_compressed_series(axs[0], R, "datasets_per_s", "#0b5394", "training contexts / s", (12.5, 21.8))

axs[1].scatter(*zip(*dt), s=3.5, alpha=0.55, lw=0, c="#2b8a3e", label="admission $\\Delta t$ (<600 s)")
if wall_pts:
    axs[1].scatter(*zip(*wall_pts), s=10, marker="s", c="#0b5394", zorder=4, label="logged round wall time\n(n=25)")
axs[1].set_ylabel("seconds / round")
axs[1].set_xlabel("round (775–1469)")
axs[1].set_xlim(min(r) - 10, max(r) + 10)
axs[1].legend(frameon=True, facecolor="white", framealpha=0.88, edgecolor="none",
              loc="upper left", fontsize=7, handlelength=1.0, handletextpad=0.3, borderpad=0.2)

plot_compressed_series(axs[2], R, "promotions_per_hr", "#5f3dc4", "program promotions / hour", (56.5, 79.5))

fig.tight_layout(pad=0.45)
fig.savefig(f"{O}/fig_fw_telemetry.pdf")
plt.close(fig)
print("ok")
