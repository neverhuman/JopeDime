#!/usr/bin/env python3
"""Audit H51's paired follow-up arms without opening DL-Val."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run as h51  # noqa: E402

ROOT = Path.home() / "jope-data/h51/runs"
ARMS = ("DATA25", "DATA50", "DATA100", "CLIPSMOOTH", "NULL")
SEEDS = (9, 10, 11)


def read_gains(arm: str, seed: int) -> tuple[dict, dict]:
    directory = ROOT / arm / f"seed{seed}"
    summary = json.loads((directory / "candidate-summary.json").read_text())
    detail = json.loads((directory / "inner-family-gains.json").read_text())
    return summary, detail


def main() -> None:
    controls = {
        seed: json.loads((ROOT / "STEPS" / "k16-lr1e-04" / f"seed{seed}" / "inner-family-gains.json").read_text())["family_gains"]
        for seed in SEEDS
    }
    rows = []
    for arm in ARMS:
        for seed in SEEDS:
            summary, detail = read_gains(arm, seed)
            base = detail["family_gains_vs_r882"]
            paired = detail["paired_family_gains_vs_steps"]
            if len(base) != 205 or len(paired) != 205 or set(base) != set(controls[seed]) or set(paired) != set(base):
                raise RuntimeError(f"{arm} seed{seed}: family inventory mismatch")
            recomputed = {fam: float(base[fam]) - float(controls[seed][fam]) for fam in base}
            if any(abs(recomputed[fam] - float(paired[fam])) > 1e-12 for fam in base):
                raise RuntimeError(f"{arm} seed{seed}: paired vector mismatch")
            interval_seed = seed * 2000 + 7 if arm == "NULL" else seed * 100 + sum(map(ord, arm))
            med, lb, ub = h51.family_median_interval(paired, interval_seed)
            if abs(med - summary["paired_family_median_gain"]) > 1e-12:
                raise RuntimeError(f"{arm} seed{seed}: paired median mismatch")
            if summary["validation_scored"] or summary["validation_payloads_opened"]:
                raise RuntimeError(f"{arm} seed{seed}: unauthorized terminal validation")
            if summary["dope_sha256_before"] != h51.DOPE_SHA or summary["dope_sha256_after"] != h51.DOPE_SHA:
                raise RuntimeError(f"{arm} seed{seed}: DOPE identity changed")
            rows.append({"arm": arm, "seed": seed, "paired_median": med, "paired_lb95": lb,
                         "paired_ub95": ub, "accepted_paired": summary["accepted_paired"],
                         "median_vs_r882": summary["family_median_gain_vs_r882"],
                         "mean_vs_r882": float(np.mean(list(map(float, base.values()))))})
    sanity_rows = []
    left = json.loads((ROOT / "CONTROL" / "seed9" / "inner-family-gains.json").read_text())["family_gains"]
    right = json.loads((ROOT / "CONTROL" / "seed10" / "inner-family-gains.json").read_text())["family_gains"]
    control_delta = {fam: float(left[fam]) - float(right[fam]) for fam in left}
    control_median, control_lb, control_ub = h51.family_median_interval(control_delta, 910)
    if (control_median, control_lb, control_ub) != (0.0, 0.0, 0.0):
        raise RuntimeError("CONTROL-vs-CONTROL seed9/10 sanity expected median and CI [0,0]")
    sanity_rows.append({"comparison": "CONTROL-seed9 minus CONTROL-seed10", "family_median": control_median,
                        "family_bootstrap_lb95": control_lb, "family_bootstrap_ub95": control_ub})
    data100_reproduction = []
    for seed in SEEDS:
        data100, _ = read_gains("DATA100", seed)
        steps_summary = json.loads((ROOT / "STEPS" / "k16-lr1e-04" / f"seed{seed}" / "candidate-summary.json").read_text())
        data100_detail = json.loads((ROOT / "DATA100" / f"seed{seed}" / "inner-family-gains.json").read_text())
        paired_values = np.asarray(list(data100_detail["paired_family_gains_vs_steps"].values()), dtype=np.float64)
        data100_reproduction.append({
            "seed": seed,
            "adapter_sha_matches_steps": data100["adapter_sha256"] == steps_summary["adapter_sha256"],
            "nonzero_paired_families": int(np.count_nonzero(paired_values)),
            "paired_family_median": float(np.median(paired_values)),
            "paired_family_fraction_exact_zero": float(np.mean(paired_values == 0.0)),
            "interpretation": "context batch seeds match; training_indices global NumPy sampling in the earlier STEPS runner was unseeded, so adapter identity is not expected to reproduce exactly",
        })
    data = json.loads((ROOT / "DATA" / "subset-plan.json").read_text())
    for seed in SEEDS:
        for fold, subsets in data["seeds"][str(seed)].items():
            fit = subsets["fit_families"]
            if not set(fit["25"]).issubset(fit["50"]) or not set(fit["50"]).issubset(fit["100"]):
                raise RuntimeError("DATA nested subset audit failed")
    result = {"schema": "jope.h51.remaining-audit/v1", "expected_cells": 15, "completed_cells": len(rows),
              "terminal_dl_val_reads": 0, "dope_sha256": h51.DOPE_SHA, "control_sanity": sanity_rows,
              "data100_reproduction": data100_reproduction, "cells": rows}
    h51.atomic_json(ROOT / "remaining-audit.json", result)
    print(json.dumps({"completed_cells": len(rows), "paired_admissions": sum(r["accepted_paired"] for r in rows),
                      "terminal_dl_val_reads": 0}, allow_nan=False))


if __name__ == "__main__":
    main()
