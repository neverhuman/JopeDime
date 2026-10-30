#!/usr/bin/env python3
"""Build vector PDF figures from the derived flywheel CSVs in ../data.
Usage: python3 make_flywheel_figs.py   (run from any dir)"""
import csv, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
H = os.path.dirname(os.path.abspath(__file__)); D = os.path.join(H, "..", "data"); O = os.path.join(H, "..")
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42, "axes.grid": True, "grid.alpha": .25, "grid.linewidth": .4})
f = lambda x: float(x) if x not in ("", "None") else None
R = list(csv.DictReader(open(f"{D}/rounds.csv"))); A = list(csv.DictReader(open(f"{D}/admissions.csv")))
# 1 validation trajectory
r = [int(a["round"]) for a in A]; c = [f(a["candidate_val"]) for a in A]; b = [f(a["best_val"]) for a in A]
acc = [(int(a["round"]), f(a["candidate_val"])) for a in A if a["accepted"] == "1"]
fig, ax = plt.subplots(figsize=(3.4, 2.1))
ax.scatter(r, c, s=2, c="0.55", lw=0, label="candidate (validation)")
ax.step(r, b, where="post", c="C0", lw=1.1, label="best admitted")
if acc: ax.scatter(*zip(*acc), s=14, marker="*", c="C3", zorder=3, label="admitted")
ax.set_xlabel("round"); ax.set_ylabel("validation utility"); ax.legend(frameon=False, loc="lower right", fontsize=6.5)
fig.tight_layout(); fig.savefig(f"{O}/fig_fw_validation.pdf"); plt.close(fig)
# 2 train vs validation
rr = [int(x["round"]) for x in R]; tv = [f(x["train"]) for x in R]; vv = [f(x["val"]) for x in R]
fig, ax = plt.subplots(figsize=(3.4, 2.1))
ax.plot(rr, tv, "o-", ms=2.5, lw=.9, c="C1", label="train utility")
ax.plot(rr, vv, "s-", ms=2.5, lw=.9, c="C0", label="validation utility")
ax.fill_between(rr, vv, tv, color="C1", alpha=.12, lw=0)
g = [t - v for t, v in zip(tv, vv)]
ax.text(.02, .5, f"gap {min(g):.3f}–{max(g):.3f}", transform=ax.transAxes, fontsize=7)
ax.set_xlabel("round (logged rounds only)"); ax.set_ylabel("utility"); ax.legend(frameon=False, fontsize=6.5)
fig.tight_layout(); fig.savefig(f"{O}/fig_fw_train_val.pdf"); plt.close(fig)
# 3 telemetry
ts = [(int(a["round"]), f(a["timestamp"])) for a in A if f(a["timestamp"])]
dt = [(r2, t2 - t1) for (r1, t1), (r2, t2) in zip(ts, ts[1:]) if r2 == r1 + 1 and 0 < t2 - t1 < 600]
fig, axs = plt.subplots(1, 3, figsize=(7.0, 1.9))
axs[0].plot(rr, [f(x["datasets_per_s"]) for x in R], "o-", ms=2, lw=.8); axs[0].set_ylabel("datasets / s")
axs[1].scatter(*zip(*dt), s=2, lw=0, c="C2"); axs[1].set_ylabel("round wall time (s)")
axs[2].plot(rr, [f(x["promotions_per_hr"]) for x in R], "o-", ms=2, lw=.8, c="C4"); axs[2].set_ylabel("promotions / hour")
for a in axs: a.set_xlabel("round")
fig.tight_layout(); fig.savefig(f"{O}/fig_fw_telemetry.pdf"); plt.close(fig)
print("ok")
