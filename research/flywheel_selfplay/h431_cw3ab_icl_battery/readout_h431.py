#!/usr/bin/env python3
"""H431 single-shot readout: aggregate ICB_v1 capability tables, clean FIT-SEL monitor scores, and verdict."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

FAMILIES = ("B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B9", "B10", "B11", "B12")
CKPTS = ("M_h020", "M_h024", "L_h018")
VARIANTS = ("full", "ridgeres")


def _med(xs: list[float]) -> float:
    s = sorted(xs)
    n = len(s)
    if n == 0:
        return 0.0
    return float(s[n // 2] if n % 2 == 1 else 0.5 * (s[n // 2 - 1] + s[n // 2]))


def summarize_battery_dir(bdir: Path, rep0_only: bool = False) -> dict:
    rows = [json.loads(p.read_text()) for p in sorted(bdir.glob("task_*.json"))]
    if rep0_only:
        rows = [r for r in rows if int(r["rep"]) == 0]
        expected = 68
    else:
        expected = 340
    if len(rows) != expected:
        raise SystemExit(f"{bdir}: expected {expected} tasks, found {len(rows)}")
    by_family = {}
    for fid in FAMILIES:
        sub = [r for r in rows if r["family"] == fid]
        nf = [r for r in sub if r["noise"] == 0.0]
        rec_f = {
            "name": sub[0]["family_name"],
            "n": len(sub),
            "mean_u8_ref": round(float(np.mean([r["u8_ref"] for r in sub])), 4),
            "mean_u8_ridge": round(float(np.mean([r["u8_ridge"] for r in sub])), 4),
        }
        for v in VARIANTS:
            caps = [r["cap"][v] for r in sub if r["cap"][v] is not None]
            caps_nf = [r["cap"][v] for r in nf if r["cap"][v] is not None]
            caps_256 = [r["cap"][v] for r in sub if r["n_ctx"] == 256 and r["cap"][v] is not None]
            caps_1024 = [r["cap"][v] for r in sub if r["n_ctx"] == 1024 and r["cap"][v] is not None]
            u8s = [r["u8"][v] for r in sub]
            u8_nf = [r["u8"][v] for r in nf]
            rec_f[v] = {
                "capability": round(float(np.mean(caps)), 4),
                "capability_noisefree": round(float(np.mean(caps_nf)), 4) if caps_nf else None,
                "capability_ctx256": round(float(np.mean(caps_256)), 4),
                "capability_ctx1024": round(float(np.mean(caps_1024)), 4),
                "mean_u8": round(float(np.mean(u8s)), 4),
                "mean_delta_vs_ridge": round(float(np.mean([r["u8"][v] - r["u8_ridge"] for r in sub])), 4),
                "precision_ceiling_u8": round(float(np.max(u8_nf)), 4) if u8_nf else None,
            }
        by_family[fid] = rec_f

    overall = {}
    for v in VARIANTS:
        all_caps = [r["cap"][v] for r in rows if r["cap"][v] is not None]
        nf_u8 = [r["u8"][v] for r in rows if r["noise"] == 0.0]
        struct_fams = ("B3", "B4", "B5", "B6", "B9", "B10")
        struct_caps = [by_family[f][v]["capability"] for f in struct_fams]
        overall[v] = {
            "mean_capability_all12": round(float(np.mean([by_family[f][v]["capability"] for f in FAMILIES])), 4),
            "mean_capability_tasks": round(float(np.mean(all_caps)), 4),
            "mean_capability_structure6": round(float(np.mean(struct_caps)), 4),
            "precision_ceiling_u8": round(float(np.max(nf_u8)), 4),
            "mean_delta_vs_ridge": round(float(np.mean([r["u8"][v] - r["u8_ridge"] for r in rows])), 4),
        }
    return {"overall": overall, "by_family": by_family}


def summarize_monitor_dir(mdir: Path, excl: dict) -> dict:
    rows = [json.loads(p.read_text()) for p in sorted(mdir.glob("ctx_*.json"))]
    if len(rows) != 106:
        raise SystemExit(f"{mdir}: expected 106 contexts, found {len(rows)}")
    tier_of = {cid: info["tier"] for cid, info in excl["contexts"].items()}
    subsets = {
        "all_106": rows,
        "clean_non_copy": [r for r in rows if tier_of[r["cid"]] != "copy"],
        "clear_only": [r for r in rows if tier_of[r["cid"]] == "clear"],
        "copy_only": [r for r in rows if tier_of[r["cid"]] == "copy"],
    }
    out = {}
    for sname, srows in subsets.items():
        if not srows:
            continue
        ur = [r["u_ridge"] for r in srows]
        rec = {
            "n": len(srows),
            "ridge_mean_u": round(float(np.mean(ur)), 4),
            "ridge_median_u": round(_med(ur), 4),
        }
        for v in VARIANTS:
            uv = [r["u"][v] for r in srows]
            rec[v] = {
                "mean_u": round(float(np.mean(uv)), 4),
                "median_u": round(_med(uv), 4),
                "win_vs_ridge": round(float(np.mean([r["u"][v] > r["u_ridge"] for r in srows])), 4),
            }
        out[sname] = rec
    return out


def evaluate_h1_expectations(bat_m24: dict) -> dict:
    bf = bat_m24["by_family"]
    checks = {
        "B3_full_lt_0p30": bf["B3"]["full"]["capability"] < 0.30,
        "B4_full_lt_0p30": bf["B4"]["full"]["capability"] < 0.30,
        "B5_full_lt_0p30": bf["B5"]["full"]["capability"] < 0.30,
        "B9_full_lt_0p30": bf["B9"]["full"]["capability"] < 0.30,
        "B10_full_lt_0p30": bf["B10"]["full"]["capability"] < 0.30,
        "B1_noisefree_full_lt_0p70": bf["B1"]["full"]["capability_noisefree"] < 0.70,
    }
    mean_struct5 = float(np.mean([bf[f]["full"]["capability"] for f in ("B3", "B4", "B5", "B9", "B10")]))
    n_true = sum(bool(v) for v in checks.values())
    if n_true == 6:
        verdict = "BATTERY_AS_EXPECTED"
    elif n_true >= 4 and mean_struct5 < 0.30:
        verdict = "BATTERY_PARTIAL_SURPRISE"
    else:
        verdict = "BATTERY_SURPRISE"
    return {
        "verdict_part_a": verdict,
        "checks": checks,
        "passed_count": n_true,
        "mean_capability_B3_B4_B5_B9_B10_full": round(mean_struct5, 4),
        "B1_noisefree_full_capability": bf["B1"]["full"]["capability_noisefree"],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", required=True, help="/home/ubuntu/jope-data/h335-cl/h431/pub")
    ap.add_argument("--gates", required=True, help="H431.gates.json (must exist and pass before readout)")
    ap.add_argument("--unit-json", required=True, help="unit_tests.json from test_cw3.py")
    ap.add_argument("--out", required=True, help="create-once H431_READOUT.json")
    a = ap.parse_args()

    out_path = Path(a.out)
    if out_path.exists():
        raise SystemExit(f"{out_path} exists (create-once)")
    gates = json.loads(Path(a.gates).read_text())
    if not gates.get("pass"):
        raise SystemExit("REFUSING readout: integrity gates did not pass")

    droot = Path(a.data_dir)
    excl = json.loads((droot / "EXCLUSION_FITSEL.json").read_text())
    icb_meta = json.loads((droot / "ICB_v1.npz.json").read_text())
    unit = json.loads(Path(a.unit_json).read_text())

    batteries = {}
    monitors = {}
    for ck in CKPTS:
        batteries[ck] = summarize_battery_dir(droot / f"bat_{ck}", rep0_only=(ck == "L_h018"))
        monitors[ck] = summarize_monitor_dir(droot / f"mon_{ck}", excl)

    h1 = evaluate_h1_expectations(batteries["M_h024"])
    partb_verdict = "PARTB_READY" if unit.get("all_passed") else "PARTB_FAIL"

    res = {
        "schema": "jope.h431.readout/v1",
        "verdict_part_a": h1["verdict_part_a"],
        "verdict_part_b": partb_verdict,
        "h1_evaluation_M_h024": h1,
        "fitsel_contamination_screen": {
            "n": excl["n"],
            "tier_counts": excl["tier_counts"],
            "screen_sha256": excl["screen_sha256"],
        },
        "icb_v1": {
            "sha256": icb_meta["sha256"],
            "tasks": icb_meta["tasks"],
            "seed": icb_meta["seed"],
        },
        "battery_by_checkpoint": batteries,
        "fitsel_monitor_by_checkpoint": monitors,
        "part_b_unit_verification": unit,
        "gates_pass": gates["pass"],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp.json")
    tmp.write_text(json.dumps(res, indent=2, sort_keys=True))
    tmp.rename(out_path)
    out_path.chmod(0o444)
    print(json.dumps({
        "verdict_part_a": res["verdict_part_a"],
        "verdict_part_b": res["verdict_part_b"],
        "fitsel_tiers": excl["tier_counts"],
        "h1_M_h024": h1,
        "M_h024_overall": batteries["M_h024"]["overall"],
        "L_h018_overall": batteries["L_h018"]["overall"],
        "M_h024_monitor": monitors["M_h024"],
        "L_h018_monitor": monitors["L_h018"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
