"""CLI runner for H291 (h291_harm_veto_abstention).
Modes:
  --mode part_a : Run Step 1 (H287 CLIP seed 287101 bit-exact regression) + Step 2 (Part A DL-Train
                  4-fold family-grouped inner CV across seeds 291101-291103) + Step 3 (export frozen
                  harm-veto boosters and write fit_lock.json BEFORE any Part B descriptive scoring).
                  Enforces sys.addaudithook blocking any access to Part B (H286/H290/even/semi_fresh)
                  as well as cohort v3 / ODD / H288 / qualification.lock / sealed / holdout.
  --mode part_b : Verify fit_lock.json sha256 and score the frozen Part A models on H286 v2 200 EVEN
                  r3175 pools (descriptive only, SEMI_FRESH, labeled reused).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import analysis_h291 as ana
import harm_veto as hv
import monitor


FORBIDDEN_ALWAYS = (
    "/sealed",
    "veox-jopedime-sealed",
    "714044927654-us-east-2-sealed",
    "sealed_excluded",
    "meta-test",
    "dl-test",
    "holdout",
    "qualification.lock",
    "h288",
    "cohort_v3",
    "v3_odd",
)

FORBIDDEN_DURING_PART_A = (
    "h286",
    "h290",
    "semi_fresh",
    "v2_200_even",
    "r3175_even",
)


def install_audit_hook(mode: str, audit_counter: Dict[str, int]) -> None:
    def _hook(event: str, args: Any) -> None:
        if event not in ("open", "os.listdir", "os.scandir", "os.stat"):
            return
        if not args:
            return
        p = str(args[0]).lower()
        for tok in FORBIDDEN_ALWAYS:
            if tok in p:
                raise RuntimeError(f"Forbidden path access blocked by H291 audit hook ({tok}): {args[0]}")
        if mode == "part_a":
            for tok in FORBIDDEN_DURING_PART_A:
                if tok in p:
                    audit_counter["part_b_pools_read_during_part_a"] += 1
                    raise RuntimeError(
                        f"Part B path access blocked during Part A ({tok}): {args[0]}"
                    )

    sys.addaudithook(_hook)


def run_part_a(args: argparse.Namespace) -> None:
    audit_counter = {"part_b_pools_read_during_part_a": 0, "fresh_pools_read": 0}
    install_audit_hook("part_a", audit_counter)

    res_start = monitor.check_resources("part_a_start")
    out_dir = Path(args.out_dir)
    export_dir = Path(args.export_dir)
    capsule_dir = Path(args.capsule_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    export_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Regression check on H287 CLIP seed 287101 inner-CV cell
    reg_check = hv.verify_h287_regression_cell(
        oof_cache_path=Path(args.oof_cache),
        h287_panel_path=Path(args.h287_panel),
    )

    # Step 2: Part A 4-fold family-grouped inner CV on DL-Train (205 families, 1,219 contexts)
    part_a_raw = hv.run_part_a_and_export_locks(
        oof_cache_path=Path(args.oof_cache),
        export_dir=export_dir,
    )

    part_a_summary = ana.summarize_panel_across_seeds(
        arm_names=part_a_raw["arm_names"],
        u_panel=part_a_raw["u_panel"],
        picks_panel=part_a_raw["picks_panel"],
        fams=part_a_raw["fams"],
        seed_labels=hv.H291_SEEDS,
        boot_seed=ana.BOOT_SEED,
        n_boot=ana.N_BOOT,
    )
    diag_summary = ana.summarize_panel_across_seeds(
        arm_names=part_a_raw["diag_arm_names"],
        u_panel=part_a_raw["diag_u_panel"],
        picks_panel=part_a_raw["diag_picks_panel"],
        fams=part_a_raw["fams"],
        seed_labels=hv.H291_SEEDS,
        boot_seed=ana.BOOT_SEED,
        n_boot=ana.N_BOOT,
    )
    gate_eval = ana.evaluate_part_a_gate(part_a_summary)

    # Save Part A OOF utility & pick panel to out_dir
    panel_npz_path = out_dir / "part_a_inner_cv_panel.npz"
    import numpy as np
    np.savez_compressed(
        panel_npz_path,
        cids=np.asarray(part_a_raw["cids"]),
        fams=np.asarray(part_a_raw["fams"]),
        arm_names=np.asarray(part_a_raw["arm_names"]),
        u_panel=part_a_raw["u_panel"],
        picks_panel=part_a_raw["picks_panel"],
        diag_arm_names=np.asarray(part_a_raw["diag_arm_names"]),
        diag_u_panel=part_a_raw["diag_u_panel"],
        diag_picks_panel=part_a_raw["diag_picks_panel"],
    )

    # Step 3: Build fit_lock.json BEFORE any Part B descriptive scoring
    h287_lock_path = Path(args.h287_fit_lock)
    h287_lock_sha256 = hv.sha256_file(h287_lock_path)
    fit_lock_payload = {
        "hypothesis_id": "H291",
        "slug": "h291_harm_veto_abstention",
        "locked_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "part_b_pools_read_during_part_a": int(audit_counter["part_b_pools_read_during_part_a"]),
        "fresh_pools_read": int(audit_counter["fresh_pools_read"]),
        "h287_parent_commit": "213b30ad52e1ce48927bba960ed609bd98ab4dfe",
        "h287_fit_lock_sha256": h287_lock_sha256,
        "dl_train_oof_cache_sha256": hv.sha256_file(Path(args.oof_cache)),
        "h291_seeds": list(hv.H291_SEEDS),
        "bootstrap_seed": int(ana.BOOT_SEED),
        "tau_grid": list(hv.TAU_GRID),
        "harm_loss_threshold": float(hv.HARM_LOSS_THRESH),
        "feature_dim": 644,
        "feature_layout": "568 H278 topk cols (proposed slot j) + 44 member/bag disagreement & slot0 margin cols + 32 candidate-minus-slot0 diff cols",
        "inner_inner_selected_taus": part_a_raw["inner_inner_selected_taus"],
        "locked_part_b_taus": part_a_raw["locked_part_b_taus"],
        "part_a_verdict": gate_eval["verdict"],
        "best_by_prereg_arm": gate_eval["best_by_prereg_arm"],
        "exported_harm_veto_models": part_a_raw["exported_files"],
    }
    fit_lock_bytes = json.dumps(fit_lock_payload, indent=2, sort_keys=True).encode("utf-8")
    fit_lock_sha256 = hashlib.sha256(fit_lock_bytes).hexdigest()
    fit_lock_payload["fit_lock_content_sha256"] = fit_lock_sha256

    for dst in (export_dir / "fit_lock.json", capsule_dir / "fit_lock.json", out_dir / "fit_lock.json"):
        dst.write_text(json.dumps(fit_lock_payload, indent=2) + "\n", encoding="utf-8")

    res_end = monitor.check_resources("part_a_end")

    part_a_out = {
        "hypothesis_id": "H291",
        "slug": "h291_harm_veto_abstention",
        "stage": "part_a_only",
        "step1_h287_regression": reg_check,
        "part_a_inner_cv": part_a_summary,
        "part_a_diagnostics": diag_summary,
        "part_a_gate_evaluation": gate_eval,
        "fit_lock": fit_lock_payload,
        "resources": {"start": res_start, "end": res_end},
    }
    (out_dir / "part_a_metrics.json").write_text(
        json.dumps(part_a_out, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": "PART_A_OK",
                "step1_bit_exact": reg_check["bit_exact"],
                "part_a_verdict": gate_eval["verdict"],
                "best_by_prereg_arm": gate_eval["best_by_prereg_arm"],
                "fit_lock_sha256": fit_lock_sha256,
                "part_b_pools_read_during_part_a": audit_counter["part_b_pools_read_during_part_a"],
            },
            indent=2,
        )
    )


def _compact_arm_summary(arm_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Return a concise arm summary without the verbose per-seed sub-dicts so metrics.json stays < 50 KB."""
    keep_keys = (
        "median_of_seeds_gain",
        "median_of_seeds_ci95",
        "seed_gains",
        "per_seed_ci95",
        "lcb95_positive_seeds",
        "all_seeds_lcb95_gt_0",
        "median_of_seeds_family_median_gain",
        "seed_family_median_gains",
        "harmed_families_median",
        "harmed_families_per_seed",
        "helped_families_median",
        "helped_families_per_seed",
        "raw_neg2_median",
        "raw_neg2_per_seed",
        "slot0_raw_neg2",
        "raw_neg2_le_slot0_all_seeds",
        "abstain_rate_median",
        "abstain_rate_per_seed",
        "mm_bt_elo_median",
        "mm_bt_elo_per_seed",
        "non_tied_wins_per_seed",
        "non_tied_losses_per_seed",
        "ties_per_seed",
        "sign_test_p_two_sided_per_seed",
        "sign_test_p_one_sided_per_seed",
    )
    return {k: arm_dict[k] for k in keep_keys if k in arm_dict}


def run_part_b(args: argparse.Namespace) -> None:
    audit_counter = {"part_b_pools_read_during_part_a": 0, "fresh_pools_read": 0}
    install_audit_hook("part_b", audit_counter)

    res_start = monitor.check_resources("part_b_start")
    out_dir = Path(args.out_dir)
    export_dir = Path(args.export_dir)
    capsule_dir = Path(args.capsule_dir)

    part_a_metrics_path = out_dir / "part_a_metrics.json"
    if not part_a_metrics_path.exists():
        raise RuntimeError(f"Part A metrics not found at {part_a_metrics_path}; run --mode part_a first")
    part_a_data = json.loads(part_a_metrics_path.read_text(encoding="utf-8"))

    part_b_raw = hv.score_part_b_on_h286_even(
        prepared_h286_npz=Path(args.prepared_h286_npz),
        h287_export_dir=Path(args.h287_export_dir),
        h291_export_dir=export_dir,
        fit_lock_path=capsule_dir / "fit_lock.json",
    )

    part_b_summary = ana.summarize_panel_across_seeds(
        arm_names=part_b_raw["arm_names"],
        u_panel=part_b_raw["u_panel"],
        picks_panel=part_b_raw["picks_panel"],
        fams=part_b_raw["fams"],
        seed_labels=hv.H291_SEEDS,
        boot_seed=ana.BOOT_SEED,
        n_boot=ana.N_BOOT,
    )
    part_b_summary["cohort"] = part_b_raw["cohort"]
    part_b_summary["scope"] = part_b_raw["scope"]
    part_b_summary["reused"] = True
    part_b_summary["fresh_claim"] = False
    part_b_summary["best_by_prereg_arm"] = part_a_data["part_a_gate_evaluation"]["best_by_prereg_arm"]
    part_b_summary["r882_fit_bit_exact_with_h286"] = part_b_raw["r882_fit_bit_exact_with_h286"]

    import numpy as np
    np.savez_compressed(
        out_dir / "part_b_h286_v2_even_panel.npz",
        cids=np.asarray(part_b_raw["cids"]),
        fams=np.asarray(part_b_raw["fams"]),
        arm_names=np.asarray(part_b_raw["arm_names"]),
        u_panel=part_b_raw["u_panel"],
        picks_panel=part_b_raw["picks_panel"],
    )

    res_end = monitor.check_resources("part_b_end")

    full_payload = {
        "schema": "jope.h291.metrics/v1",
        "hypothesis_id": "H291",
        "slug": "h291_harm_veto_abstention",
        "lane": "rd_argon",
        "status": "COMPLETE",
        "verdict": part_a_data["part_a_gate_evaluation"]["verdict"],
        "prereg_commit_on_v1_next": "d8bc373b80c80892600ddaaa83272616f2b1bda9",
        "fit_lock_content_sha256": part_a_data["fit_lock"]["fit_lock_content_sha256"],
        "fit_lock_file_sha256": hv.sha256_file(capsule_dir / "fit_lock.json"),
        "neural_forwards": 0,
        "neural_updates": 0,
        "dope_sha256": "c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814",
        "step1_h287_regression": part_a_data["step1_h287_regression"],
        "part_a_gate_evaluation": part_a_data["part_a_gate_evaluation"],
        "part_a_inner_cv": part_a_data["part_a_inner_cv"],
        "part_a_diagnostics": part_a_data["part_a_diagnostics"],
        "part_b_semi_fresh_h286_v2_even": part_b_summary,
        "fit_lock": part_a_data["fit_lock"],
        "resources": {
            "part_a": part_a_data["resources"],
            "part_b": {"start": res_start, "end": res_end},
        },
    }
    (out_dir / "metrics_full.json").write_text(
        json.dumps(full_payload, indent=2) + "\n", encoding="utf-8"
    )

    compact_payload = {
        "schema": "jope.h291.metrics/v1",
        "hypothesis_id": "H291",
        "slug": "h291_harm_veto_abstention",
        "lane": "rd_argon",
        "status": "COMPLETE",
        "verdict": part_a_data["part_a_gate_evaluation"]["verdict"],
        "prereg_commit_on_v1_next": "d8bc373b80c80892600ddaaa83272616f2b1bda9",
        "fit_lock_content_sha256": part_a_data["fit_lock"]["fit_lock_content_sha256"],
        "fit_lock_file_sha256": hv.sha256_file(capsule_dir / "fit_lock.json"),
        "part_b_pools_read_during_part_a": 0,
        "neural_forwards": 0,
        "neural_updates": 0,
        "dope_sha256": "c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814",
        "step1_h287_regression": part_a_data["step1_h287_regression"],
        "part_a_gate_evaluation": part_a_data["part_a_gate_evaluation"],
        "part_a_inner_cv": {
            "cohort": "DL-Train_4fold_inner_cv",
            "n_families": part_a_data["part_a_inner_cv"]["n_families"],
            "n_contexts": part_a_data["part_a_inner_cv"]["n_contexts"],
            "seed_labels": part_a_data["part_a_inner_cv"]["seed_labels"],
            "bootstrap": part_a_data["part_a_inner_cv"]["bootstrap"],
            "inner_inner_selected_taus": part_a_data["fit_lock"]["inner_inner_selected_taus"],
            "arms": {
                k: _compact_arm_summary(v)
                for k, v in part_a_data["part_a_inner_cv"]["arms"].items()
            },
        },
        "part_a_diagnostics_arms": {
            k: _compact_arm_summary(v)
            for k, v in part_a_data["part_a_diagnostics"]["arms"].items()
            if k in ("CLIP_VOTE2", "ENS_VOTE2", "CLIP_VOTE2+HARMVETO", "CLIP+HARMVETO_PLACEBO", "CLIP+HARMVETO_NULL_LF")
        },
        "part_b_semi_fresh_h286_v2_even": {
            "cohort": part_b_summary["cohort"],
            "scope": part_b_summary["scope"],
            "reused": True,
            "fresh_claim": False,
            "best_by_prereg_arm": part_b_summary["best_by_prereg_arm"],
            "r882_fit_bit_exact_with_h286": part_b_summary["r882_fit_bit_exact_with_h286"],
            "n_families": part_b_summary["n_families"],
            "n_contexts": part_b_summary["n_contexts"],
            "seed_labels": part_b_summary["seed_labels"],
            "bootstrap": part_b_summary["bootstrap"],
            "locked_part_b_taus": part_a_data["fit_lock"]["locked_part_b_taus"],
            "arms": {
                k: _compact_arm_summary(v)
                for k, v in part_b_summary["arms"].items()
            },
        },
        "resources": {
            "part_a_end": part_a_data["resources"]["end"],
            "part_b_end": res_end,
        },
    }
    compact_text = json.dumps(compact_payload, indent=1) + "\n"
    if len(compact_text.encode("utf-8")) >= 50_000:
        compact_payload.pop("part_a_diagnostics_arms", None)
        compact_text = json.dumps(compact_payload, indent=1) + "\n"
    if len(compact_text.encode("utf-8")) >= 50_000:
        raise RuntimeError(f"metrics.json size {len(compact_text.encode('utf-8'))} >= 50000 bytes")
    for dst in (capsule_dir / "metrics.json", out_dir / "metrics.json"):
        dst.write_text(compact_text, encoding="utf-8")

    print(
        json.dumps(
            {
                "status": "PART_B_OK",
                "verdict": compact_payload["verdict"],
                "best_by_prereg_arm": part_b_summary["best_by_prereg_arm"],
                "metrics_json_bytes": len(compact_text.encode("utf-8")),
                "part_b_best_arm_median_gain": part_b_summary["arms"][part_b_summary["best_by_prereg_arm"]]["median_of_seeds_gain"],
                "part_b_best_arm_ci95": part_b_summary["arms"][part_b_summary["best_by_prereg_arm"]]["median_of_seeds_ci95"],
                "part_b_best_arm_harmed_per_seed": part_b_summary["arms"][part_b_summary["best_by_prereg_arm"]]["harmed_families_per_seed"],
                "part_b_best_arm_lcb95_positive_seeds": part_b_summary["arms"][part_b_summary["best_by_prereg_arm"]]["lcb95_positive_seeds"],
            },
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="H291 harm-veto abstention runner")
    parser.add_argument("--mode", choices=("part_a", "part_b"), required=True)
    parser.add_argument(
        "--oof-cache",
        default="/home/ubuntu/jope-data/h291/cache/dl_train_member_oof.npz",
    )
    parser.add_argument(
        "--h287-panel",
        default="/home/ubuntu/jope_plan/evidence/H287/argon-t0005/runs/attempt-001/inner_cv_panel.npz",
    )
    parser.add_argument(
        "--h287-fit-lock",
        default="/home/ubuntu/jope-data/h287/export/fit_lock.json",
    )
    parser.add_argument(
        "--h287-export-dir",
        default="/home/ubuntu/jope-data/h287/export",
    )
    parser.add_argument(
        "--prepared-h286-npz",
        default="/home/ubuntu/jope-data/h291/part_b_inputs/h286_v2_200_even_prepared.npz",
    )
    parser.add_argument(
        "--export-dir",
        default="/home/ubuntu/jope-data/h291/export",
    )
    parser.add_argument(
        "--capsule-dir",
        default="/home/ubuntu/jope_plan/evidence/H291/argon-t0050/capsule/research/flywheel_selfplay/h291_harm_veto_abstention",
    )
    parser.add_argument(
        "--out-dir",
        default="/home/ubuntu/jope_plan/evidence/H291/argon-t0050/runs/attempt-001",
    )
    args = parser.parse_args()
    if args.mode == "part_a":
        run_part_a(args)
    else:
        run_part_b(args)


if __name__ == "__main__":
    main()
