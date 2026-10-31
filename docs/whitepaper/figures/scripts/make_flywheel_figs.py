#!/usr/bin/env python3
"""Build vector PDF figures from the derived flywheel CSVs in ../data.
Usage: python3 make_flywheel_figs.py   (run from any dir)"""
import argparse
import csv
import hashlib
import json
import math
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

H = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(H, "..", "data")
FIGURES_DIR = os.path.join(H, "..")
parser = argparse.ArgumentParser(description=__doc__)
selection = parser.add_mutually_exclusive_group()
selection.add_argument("--validation-only", action="store_true",
                    help="Refresh the admission figure and provenance without rebuilding other figures")
selection.add_argument("--admission-shortfall", action="store_true",
                       help="Plot the cumulative distribution of retained candidate utility gaps")
args = parser.parse_args()

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7.1,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.grid": True,
    "grid.alpha": 0.22,
    "grid.linewidth": 0.4,
    "axes.linewidth": 0.6,
})

def f(x):
    return float(x) if x not in ("", "None", None) else None
with open(f"{D}/rounds.csv") as fh:
    R = list(csv.DictReader(fh))
with open(f"{D}/admissions.csv", "rb") as fh:
    admissions_bytes = fh.read()
input_sha256 = hashlib.sha256(admissions_bytes).hexdigest()
assert input_sha256 == "fd64a540f48127f4475b9eda8adeba6ecdcc767757cfdbba118ede332995bf9e", \
    "Admission CSV differs from the pinned merged receipt; refresh its source pin"
A = list(csv.DictReader(admissions_bytes.decode().splitlines()))

# 1. Validation trajectory (695 admission evaluations, rounds 775-1469)
r = [int(a["round"]) for a in A]
c = [f(a["candidate_val"]) for a in A]
b = [f(a["best_val"]) for a in A]
assert r == list(range(r[0], r[-1] + 1)), "Need consecutive admission rounds"
assert all(v is not None and math.isfinite(v) for v in c + b)
assert all(a["accepted"] in ("0", "1") for a in A)
assert A[0]["accepted"] == "0", "First retained row lacks a previous best for classification"
strict_gains = []
tie_adms = []
prev_best = None
for a in A:
    rd = int(a["round"])
    cv = f(a["candidate_val"])
    bv = f(a["best_val"])
    if prev_best is not None:
        assert bv >= prev_best, "Recorded best decreased"
        if a["accepted"] == "0":
            assert bv == prev_best, "Unaccepted row changed the best"
    if a["accepted"] == "1":
        assert prev_best is not None and cv == bv
        if bv > prev_best:
            strict_gains.append((rd, cv))
        else:
            tie_adms.append((rd, cv))
    prev_best = bv


def admission_shortfall():
    """Give every retained admission round one observation, including repeated scores."""
    shortfalls = [best - candidate for best, candidate in zip(b, c)]
    assert all(math.isfinite(value) and value >= 0 for value in shortfalls)
    rows = [{"round": rd, "utility_shortfall": value}
            for rd, value in zip(r, shortfalls)]
    csv_path = f"{D}/admission_shortfall.csv"
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    fig, ax = plt.subplots(figsize=(3.4, 2.05))
    ax.step([0.0, *sorted(shortfalls)],
            [0.0, *[i / len(rows) for i in range(1, len(rows) + 1)]],
            where="post", color="#0b5394", lw=1.15)
    ax.set_xlabel("Recorded best − candidate\n(validation pool utility)")
    ax.set_ylabel("Share of logged rounds")
    ax.set_xticks([0, 0.002, 0.004, 0.006, 0.008])
    ax.set_xlim(0, max(shortfalls) * 1.05)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_ylim(0, 1.03)
    ax.text(0.97, 0.10, f"{len(rows)} logged rounds", transform=ax.transAxes,
            ha="right", fontsize=7.2)
    fig.tight_layout(pad=0.4)
    output = f"{FIGURES_DIR}/fig_fw_admission_shortfall.pdf"
    fig.savefig(output, metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)
    hashes = {}
    for key, path in [("csv_sha256", csv_path), ("figure_sha256", output),
                      ("generator_sha256", __file__)]:
        with open(path, "rb") as fh:
            hashes[key] = hashlib.sha256(fh.read()).hexdigest()
    provenance = {
        "schema": "whitepaper.admission-shortfall/v1",
        "source_commit": "002b92caeeb313456d6d05cd6049564eaa7d0980",
        "input": "docs/whitepaper/figures/data/admissions.csv",
        "input_sha256": input_sha256, "round_window": [r[0], r[-1]],
        "rows": len(rows),
        "derivation": "utility_shortfall = float(best_val) - float(candidate_val), using each row's post-decision recorded best",
        "ECDF": "At x, count(shortfall <= x) / 695; every retained round appears once, repeated values retained; no duration weighting",
        "zero_shortfall_rounds": [rd for rd, gap in zip(r, shortfalls) if gap == 0],
        "zero_scope": "Post-decision best equals candidate on accepted strict gains and ties; zero does not mean a tie with the previous incumbent",
        "scope": "Descriptive adaptive development observations; not independent tasks or distinct programs; no confidence/first-proposal/transfer claim",
        "source_binding": "CSV field interpretation only; installed source/adapter/checkpoint qualification remains required",
        "matplotlib_version": matplotlib.__version__, **hashes,
        "reproduce": "python3 figures/scripts/make_flywheel_figs.py --admission-shortfall",
    }
    with open(f"{D}/admission_shortfall.provenance.json", "w") as fh:
        json.dump(provenance, fh, indent=2)
        fh.write("\n")
    print(f"{len(rows)} admission gaps; {sum(value == 0 for value in shortfalls)} at zero")


if args.admission_shortfall:
    admission_shortfall()
    raise SystemExit(0)

fig, ax = plt.subplots(figsize=(3.4, 2.15))
ax.scatter(r, c, s=4.5, c="#6e7781", alpha=0.55, lw=0, zorder=2, label="candidate adapter")
ax.step(r, b, where="post", c="#0b5394", lw=1.25, zorder=3, label="best recorded adapter")
if strict_gains:
    ax.scatter(*zip(*strict_gains), s=28, marker="*", c="#c92a2a", zorder=5,
               label=f"strict gain ({', '.join(str(rd) for rd, _ in strict_gains)})")
if tie_adms:
    ax.scatter(*zip(*tie_adms), s=16, marker="D", facecolors="none", edgecolors="#d9480f",
               lw=0.9, zorder=4,
               label=f"accepted tie ({', '.join(str(rd) for rd, _ in tie_adms)})")
accepted = len(strict_gains) + len(tie_adms)
ax.text(0.04, 0.55,
        f"{len(A)} logged rounds\n{accepted} accepted: {len(strict_gains)} strict gains + {len(tie_adms)} ties",
        transform=ax.transAxes, fontsize=7.2, va="center",
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.9))
ax.set_xlabel("round")
ax.set_ylabel("validation pool utility")
ax.set_xlim(min(r) - 10, max(r) + 10)
ax.legend(frameon=True, facecolor="white", framealpha=0.85, edgecolor="none",
          loc="lower right", handletextpad=0.3, borderpad=0.25)
fig.tight_layout(pad=0.4)
figure_path = f"{FIGURES_DIR}/fig_fw_validation.pdf"
fig.savefig(figure_path, metadata={"CreationDate": None, "ModDate": None})
plt.close(fig)
with open(__file__, "rb") as fh:
    generator_sha256 = hashlib.sha256(fh.read()).hexdigest()
with open(figure_path, "rb") as fh:
    figure_sha256 = hashlib.sha256(fh.read()).hexdigest()
provenance = {
    "schema": "whitepaper.retained-adapter-admissions/v1",
    "source_commit": "3ff29c8b6291d239cb1684dfa8dc8bce22831c16",
    "input": "docs/whitepaper/figures/data/admissions.csv",
    "input_sha256": input_sha256,
    "round_window": [r[0], r[-1]], "logged_rounds": len(A),
    "accepted_rounds": accepted,
    "strict_best_increase_rounds": [rd for rd, _ in strict_gains],
    "accepted_tie_rounds": [rd for rd, _ in tie_adms],
    "classification": "Use logged accepted flags; compare each accepted best_val with the previous retained best_val using exact > or ==, with no epsilon",
    "checks": "Consecutive rounds, finite utilities, nondecreasing best, accepted candidate equals best, rejected best unchanged; first row is not accepted",
    "scope": "Retained development admission rounds only; no pre-window comparison, confidence interval, first-proposal or transfer claim",
    "source_binding": "CSV interpretation only; installed loop/adapter/checkpoint qualification remains required",
    "generator_sha256": generator_sha256, "figure_sha256": figure_sha256,
    "matplotlib_version": matplotlib.__version__,
    "reproduce": "python3 figures/scripts/make_flywheel_figs.py --validation-only",
}
with open(f"{D}/admissions.provenance.json", "w") as fh:
    json.dump(provenance, fh, indent=2)
    fh.write("\n")
if args.validation_only:
    print(f"{len(A)} rounds; {accepted} accepted; {len(strict_gains)} strict gains; {len(tie_adms)} ties")
    raise SystemExit(0)

admission_shortfall()

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
         transform=axR.transAxes, fontsize=7.1,
         bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8))
axR.legend(frameon=True, facecolor="white", framealpha=0.85, edgecolor="none",
           loc="center right", bbox_to_anchor=(1.0, 0.35), handletextpad=0.3, borderpad=0.25)
fig.supxlabel("round (25 retained summary rounds; 1346–1448 not retained)", fontsize=7.5, y=0.02)
fig.subplots_adjust(left=0.16, right=0.98, bottom=0.20, top=0.96, wspace=0.08)
fig.savefig(f"{FIGURES_DIR}/fig_fw_train_val.pdf")
plt.close(fig)

# 3. Telemetry across the two summary windows and inter-admission intervals
ts = [(int(a["round"]), f(a["timestamp"])) for a in A if f(a["timestamp"])]
dt = [(rd2, t2 - t1) for (rd1, t1), (rd2, t2) in zip(ts, ts[1:]) if rd2 == rd1 + 1 and 0 < t2 - t1 < 600]
wall_pts = [(int(x["round"]), f(x["wall_s"])) for x in R if f(x["wall_s"]) is not None]

GAP_SHIFT = 98  # compresses the 103-round gap in retained summaries (1346-1448) to 5 display units
def x_comp(rd):
    return rd if rd <= 1345 else rd - GAP_SHIFT
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
fig.savefig(f"{FIGURES_DIR}/fig_fw_telemetry.pdf")
plt.close(fig)
print("ok")
