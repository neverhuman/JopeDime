#!/usr/bin/env python3
"""Plot pinned metadata and two measured losses; never run models or read labels."""
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA, FIG = ROOT / "data", ROOT / "figures"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titleweight": "bold", "axes.titlelocation": "left"})


def save(fig, name):
    fig.savefig(FIG / name, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    FIG.mkdir(exist_ok=True)
    rows = list(csv.DictReader((DATA / "plot-dataset-sizes.csv").open()))
    y = np.arange(len(rows))
    fig, axes = plt.subplots(1, 2, figsize=(10, 10), gridspec_kw={"width_ratios": [2, 1]})
    a, b = axes
    dev = [int(r["rows_development"]) for r in rows]
    sealed = [int(r["rows_sealed_metadata_only"]) for r in rows]
    a.barh(y, dev, color="#246b8e", label="Development rows")
    a.barh(y, sealed, left=dev, color="#cedbe5", label="Sealed membership count only")
    a.set_yticks(y, [r["id"] for r in rows], fontsize=8)
    a.invert_yaxis(); a.set_xlabel("Frozen row count"); a.legend(loc="lower right", fontsize=8)
    a.set_title("All thirty frozen datasets")
    numeric = [int(r["numeric_features"]) for r in rows]
    categorical = [int(r["categorical_features"]) for r in rows]
    b.barh(y, numeric, color="#318c89", label="Numeric")
    b.barh(y, categorical, left=numeric, color="#e7a24b", label="Categorical")
    b.set_yticks(y, []); b.invert_yaxis(); b.set_xlabel("Feature count")
    b.set_title("Original feature types"); b.legend(loc="lower right", fontsize=8)
    for ax in axes:
        ax.axhline(9.5, color="#8899aa", linewidth=.7)
        ax.axhline(19.5, color="#8899aa", linewidth=.7)
        ax.grid(axis="x", alpha=.15); ax.set_axisbelow(True)
    fig.text(.02, .012, "Regression / binary / multiclass: ten each. Sealed target bodies were not opened.\nSource: pinned plot-dataset-sizes.csv and dataset-inventory.json; 23 September 2026.", fontsize=8)
    fig.tight_layout(rect=(0, .045, 1, 1)); save(fig, "inventory-frozen30.png")

    catalog = json.loads((DATA / "catalog-summary.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.3))
    tasks = catalog["tasks"]
    axes[0].barh(list(tasks), [v["datasets"] for v in tasks.values()], color="#246b8e")
    axes[0].set_xlabel("Catalog datasets"); axes[0].set_title("3,859 real dataset metadata entries")
    for i, v in enumerate(tasks.values()): axes[0].text(v["datasets"] + 20, i, str(v["datasets"]), va="center", fontsize=8)
    license_counts = catalog["license_status"]
    axes[1].bar(list(license_counts), list(license_counts.values()), color=["#318c89", "#d78045"])
    axes[1].set_title("License metadata requires attention"); axes[1].set_ylabel("Catalog datasets")
    for i, value in enumerate(license_counts.values()): axes[1].text(i, value + 30, f"{value:,}", ha="center", fontsize=8)
    fig.text(.02, .01, "Metadata statistics only. multi_label is a catalog task, not multiclass admission.\n3,755 licenses unknown; the collection's older prose count is stale. Source: catalog-summary.json.", fontsize=8)
    fig.tight_layout(rect=(0, .11, 1, 1)); save(fig, "inventory-corpus.png")

    current = list(csv.DictReader((DATA / "plot-current-origin.csv").open()))
    fig, ax = plt.subplots(figsize=(8, 4))
    loss = [float(r["loss"]) for r in current]
    bars = ax.barh([r["method"] for r in current], loss, color=["#d78045", "#246b8e"])
    ax.invert_yaxis(); ax.set_xlim(0, 190); ax.set_xlabel("Development RMSE — lower is better")
    for bar, value in zip(bars, loss): ax.text(value + 1, bar.get_y() + bar.get_height()/2, f"{value:.3f}", va="center")
    ax.set_title("Current neural-origin proof: generic has lower loss")
    fig.text(.02, .01, "230_machine_cpu · 34 predictions per final refit · reviewed source 68f56404\nOne operational proof; no full-panel Elo, sealed-test result or teaching admission.", fontsize=8)
    fig.tight_layout(rect=(0, .12, 1, 1)); save(fig, "current-origin-loss.png")

    names = ["plot-dataset-sizes.csv", "dataset-inventory.json", "catalog-summary.json", "plot-current-origin.csv"]
    manifest = {"schema": "jope.manual-inventory-figures/v1", "sources": {n: hashlib.sha256((DATA/n).read_bytes()).hexdigest() for n in names}, "outputs": {n: hashlib.sha256((FIG/n).read_bytes()).hexdigest() for n in ["inventory-frozen30.png", "inventory-corpus.png", "current-origin-loss.png"]}}
    (DATA / "inventory-figures.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("Three inventory/performance figures generated from pinned local data.")


if __name__ == "__main__":
    main()
