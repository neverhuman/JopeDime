#!/usr/bin/env python3
"""H455 Cooled Wave-2A / Wave-2B / Wave-2A+ & Multi-Checkpoint Soup Scorer (`build_and_score_all_wave2_h455.py`).

1. Builds `cv7_hyb_w2_cooled_soup_avg` (`91/91` contexts) by equal-averaging the SEARCH (`q_*`) and AUDIT (`bag_*`, `ref_*`)
   predictions across the cooled checkpoints (`w2a_soup50`, `w2b_lr5e6_ema_soup50`, `w2aplus_soup50`, `w2aplus_cooldown`)
   with `0` `ya` read.
2. Evaluates all 6 cooled `Wave-2` arms against the 33-model `dev91` in-corpus board using `score_h454_paired_flywheel_matrix.py`
   and `traj_e8_w2aplus`.
3. Writes `WAVE2_COOLDOWN_PROGRESSION_SUMMARY.json` summarizing the progression from `Wave-1` (`lr=2.0e-5`) through
   `Wave-2A` (`lr~1.0e-5`), `Wave-2B` (`lr<=5.0e-6`), `Wave-2A+` (`lr=3.26e-6 / 2.0e-6`), and `Wave-2 Cooled Soup`.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

CV7_HYBRID_COLS = (
    "icl_e8_cw12_robridge_cv7",
    "icl_e8_cw12_yj_cv7",
    "icl_e8_rankgauss_duan_cv7",
    "icl_e8_gp_resid_cv7",
)


def build_cooled_prediction_soup(src_dirs: list[Path], out_dir: Path) -> int:
    """Equal-weight prediction-space average across cooled CV-7 hybrid directories (`0` `ya` read!)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    first_files = sorted(p.name for p in src_dirs[0].glob("*.npz") if not p.name.startswith("."))
    if len(first_files) < 91:
        raise SystemExit(f"expected 91 .npz files in {src_dirs[0]}, found {len(first_files)}")
    built = 0
    for fname in first_files:
        loaded = []
        for sd in src_dirs:
            fp = sd / fname
            if not fp.is_file():
                raise SystemExit(f"missing {fp} while building cooled soup")
            loaded.append(np.load(fp, allow_pickle=False))
        out_arrays: dict[str, np.ndarray] = {"ok": np.array(True)}
        for col in CV7_HYBRID_COLS:
            for pref in ("q_", "bag_", "ref_"):
                k = f"{pref}{col}"
                arrs = [np.asarray(z[k], dtype=np.float64) for z in loaded]
                out_arrays[k] = np.mean(np.column_stack(arrs), axis=1)
        for z in loaded:
            z.close()
        tmp_p = out_dir / f".{fname}.{os.getpid()}.tmp.npz"
        np.savez(tmp_p, **out_arrays)
        os.replace(tmp_p, out_dir / fname)
        built += 1
    return built


def extract_key_metrics(matrix_path: Path) -> dict:
    doc = json.loads(matrix_path.read_text())
    tbl = {row["arm"]: row for row in doc.get("summary_table", [])}
    pw = doc.get("paired_comparisons", {})

    def _pick(arm: str) -> dict:
        r = tbl.get(arm, {})
        ci = r.get("ci95", [None, None])
        return {
            "elo": r.get("elo"),
            "ci95": ci,
            "elo_lcb95": ci[0] if isinstance(ci, list) and len(ci) >= 2 else None,
            "elo_ucb95": ci[1] if isinstance(ci, list) and len(ci) >= 2 else None,
            "rank_of_34": r.get("rank_of_34"),
            "family_mean_u": r.get("family_mean_u"),
            "vs_kumo_large": r.get("vs_kumo_large"),
            "vs_tabfm_ens": r.get("vs_tabfm_ens"),
        }

    best_arm_row = max(
        (r for r in doc.get("summary_table", []) if "post_h454" in r.get("arm", "")),
        key=lambda x: float(x.get("elo", -1e9)),
    )
    b_ci = best_arm_row.get("ci95", [None, None])
    return {
        "matrix_json": str(matrix_path),
        "best_post_arm": best_arm_row.get("arm"),
        "best_post_elo": best_arm_row.get("elo"),
        "best_post_lcb95": b_ci[0] if isinstance(b_ci, list) and len(b_ci) >= 2 else None,
        "best_post_ucb95": b_ci[1] if isinstance(b_ci, list) and len(b_ci) >= 2 else None,
        "best_post_rank_of_34": best_arm_row.get("rank_of_34"),
        "best_post_utility": best_arm_row.get("family_mean_u"),
        "arms": {
            "pure_104m_2xcv7_post": _pick("pure_104m_2xcv7_post_h454"),
            "icl_plus_gpresid_2xcv7_post": _pick("icl_plus_gpresid_2xcv7_post_h454"),
            "pair_104m_ugp_g1000_2xcv7_post": _pick("pair_104m_ugp_g1000_2xcv7_post_h454"),
            "stack_ugp_g1000_cv7_post": _pick("stack_ugp_g1000_cv7_post_h454"),
            "stack_ugp_g1000_2xcv7_post": _pick("stack_ugp_g1000_2xcv7_post_h454"),
            "stack_ugp_g1000_cons_post": _pick("stack_ugp_g1000_cons_post_h454"),
        },
        "paired_comparisons": pw,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cols-root", default="/home/ubuntu/jope-data/h452-flywheel/cols_e8_d91")
    ap.add_argument("--traj-post-dir", default="/home/ubuntu/jope-data/h452-flywheel/cols_e8_d91/traj_e8_w2aplus")
    ap.add_argument("--procs", type=int, default=36)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--summarize-only", action="store_true")
    ap.add_argument("--summary-out", default="/home/ubuntu/jope_plan/evidence/H455/WAVE2_COOLDOWN_PROGRESSION_SUMMARY.json")
    a = ap.parse_args()

    cols_root = Path(a.cols_root)
    scorer = Path("/home/ubuntu/h452-stage/code/score_h454_paired_flywheel_matrix.py")
    ev454 = Path("/home/ubuntu/jope_plan/evidence/H454")
    ev455 = Path("/home/ubuntu/jope_plan/evidence/H455")
    ev454.mkdir(parents=True, exist_ok=True)
    ev455.mkdir(parents=True, exist_ok=True)

    soup_avg_dir = cols_root / "cv7_hyb_w2_cooled_soup_avg"
    if not a.summarize_only:
        built_n = build_cooled_prediction_soup(
            [
                cols_root / "cv7_hyb_w2a_soup50",
                cols_root / "cv7_hyb_h454_w2b_lr5e6_ema_soup50",
                cols_root / "cv7_hyb_w2aplus_soup50",
                cols_root / "cv7_hyb_w2aplus_cooldown",
            ],
            soup_avg_dir,
        )
        print(json.dumps({"built_cooled_soup_avg": built_n, "dir": str(soup_avg_dir)}), flush=True)

    jobs = [
        ("wave2a_soup50", cols_root / "cv7_hyb_w2a_soup50", ev454 / "H454_WAVE2A_SOUP50_MATRIX.json"),
        ("wave2a_cooldown", cols_root / "cv7_hyb_w2a_cooldown", ev454 / "H454_WAVE2A_COOLDOWN_MATRIX.json"),
        ("wave2b_lr5e6_soup50", cols_root / "cv7_hyb_h454_w2b_lr5e6_ema_soup50", ev454 / "H454_WAVE2B_LR5E6_SOUP50_MATRIX.json"),
        ("wave2aplus_soup50", cols_root / "cv7_hyb_w2aplus_soup50", ev454 / "H454_WAVE2APLUS_SOUP50_MATRIX.json"),
        ("wave2aplus_cooldown", cols_root / "cv7_hyb_w2aplus_cooldown", ev454 / "H454_WAVE2APLUS_COOLDOWN_MATRIX.json"),
        ("wave2_cooled_soup_avg", soup_avg_dir, ev455 / "H455_WAVE2_COOLED_SOUP_AVG_MATRIX.json"),
    ]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""

    results: dict[str, dict] = {}
    w1_path = ev454 / "H454_PAIRED_FLYWHEEL_MATRIX.json"
    if w1_path.is_file():
        results["wave1_uncooled_lr2e5"] = extract_key_metrics(w1_path)

    for label, post_dir, out_json in jobs:
        t0 = time.time()
        if not a.summarize_only:
            subprocess.run(
                [
                    sys.executable,
                    str(scorer),
                    "--cv7-post-dir",
                    str(post_dir),
                    "--traj-post-dir",
                    str(a.traj_post_dir),
                    "--procs",
                    str(a.procs),
                    "--n-boot",
                    str(a.n_boot),
                    "--out-json",
                    str(out_json),
                ],
                env=env,
                check=True,
            )
        rec = extract_key_metrics(out_json)
        rec["score_elapsed_s"] = round(time.time() - t0, 2)
        results[label] = rec
        print(
            json.dumps(
                {
                    "label": label,
                    "best_post_arm": rec["best_post_arm"],
                    "best_post_elo": rec["best_post_elo"],
                    "pure_104m_elo": rec["arms"]["pure_104m_2xcv7_post"]["elo"],
                    "stack_2xcv7_elo": rec["arms"]["stack_ugp_g1000_2xcv7_post"]["elo"],
                    "stack_cons_elo": rec["arms"]["stack_ugp_g1000_cons_post"]["elo"],
                    "elapsed_s": rec["score_elapsed_s"],
                }
            ),
            flush=True,
        )

    best_overall_label = max(results.keys(), key=lambda k: float(results[k]["best_post_elo"] or -1e9))
    summary = {
        "schema": "jope.h455.wave2_cooldown_progression/v1",
        "timestamp_utc": time.strftime("%FT%TZ", time.gmtime()),
        "in_corpus_dev91_disclosure": True,
        "canonical_promotion_forbidden": True,
        "quarantine": {
            "confirm_opens": 0,
            "fmgap_bc_opens": 0,
            "holdout_opens": 0,
            "sealed_opens": 0,
        },
        "best_wave_label": best_overall_label,
        "best_wave_arm": results[best_overall_label]["best_post_arm"],
        "best_wave_elo": results[best_overall_label]["best_post_elo"],
        "best_wave_lcb95": results[best_overall_label]["best_post_lcb95"],
        "best_wave_ucb95": results[best_overall_label]["best_post_ucb95"],
        "best_wave_utility": results[best_overall_label]["best_post_utility"],
        "waves": results,
    }
    out_summary = Path(a.summary_out)
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    out_summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"SUMMARY_WRITTEN: {out_summary}", flush=True)


if __name__ == "__main__":
    main()
