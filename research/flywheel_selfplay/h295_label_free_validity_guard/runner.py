"""CLI entrypoint for H295 (h295_label_free_validity_guard):
- `--mode fit_and_lock`: Run H287 ENS_VOTE2 regression check, fit outer-fold training models and
  select nested inner-CV thresholds `tau` with outer-fold validation labels masked out (`NaN`/`-1`),
  export full-DL-Train validity guard models, and write `fit_lock.json` BEFORE outer scoring.
- `--mode outer_score`: Disable `.fit()`, verify `fit_lock.json` and all 84 exported model files' SHA256
  digests, score held-out outer folds across seeds 295101-295103 (and null seeds 295201-295203), and
  write `metrics.json` (< 50 KB) and `metrics_full.json`.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import monitor
import valguard

OOF_CACHE_PATH = Path("/home/ubuntu/jope-data/h291/cache/dl_train_member_oof.npz")
H287_PANEL_PATH = Path("/home/ubuntu/jope_plan/evidence/H287/argon-t0005/runs/attempt-001/inner_cv_panel.npz")
H287_METRICS_PATH = Path(
    "/home/ubuntu/jope_plan/evidence/H287/argon-t0005/capsule/research/flywheel_selfplay/h287_bagswitch_seed_ensemble_export/metrics.json"
)
EXPORT_DIR = Path("/home/ubuntu/jope-data/h295/export")
RUN_DIR = Path("/home/ubuntu/jope_plan/evidence/H295/argon-t0158/runs/attempt-001")
CAPSULE_DIR = Path(
    "/home/ubuntu/jope_plan/evidence/H295/argon-t0158/capsule/research/flywheel_selfplay/h295_label_free_validity_guard"
)


def main() -> None:
    parser = argparse.ArgumentParser(description="H295 label-free validity guard runner")
    parser.add_argument("--mode", choices=("fit_and_lock", "outer_score"), required=True)
    args = parser.parse_args()

    monitor.install_isolation_audit_hook()
    res_start = monitor.enforce_resources()

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    CAPSULE_DIR.mkdir(parents=True, exist_ok=True)
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    capsule_fit_lock_path = CAPSULE_DIR / "fit_lock.json"

    if args.mode == "fit_and_lock":
        lock_payload = valguard.run_fit_and_lock(
            oof_cache_path=OOF_CACHE_PATH,
            h287_panel_path=H287_PANEL_PATH,
            h287_metrics_path=H287_METRICS_PATH,
            export_dir=EXPORT_DIR,
            capsule_fit_lock_path=capsule_fit_lock_path,
        )
        res_end = monitor.enforce_resources()
        receipt = {
            "mode": "fit_and_lock",
            "fit_lock_content_sha256": lock_payload["fit_lock_content_sha256"],
            "fit_lock_sha256": valguard.sha256_file(capsule_fit_lock_path),
            "outer_fold_labels_read_before_lock": lock_payload["outer_fold_labels_read_before_lock"],
            "forbidden_reads_count": monitor.get_forbidden_reads_count(),
            "h287_ens_vote2_regression": lock_payload["h287_ens_vote2_regression"],
            "n_outer_fold_models": len(lock_payload["outer_fold_models"]),
            "n_full_export_models": len(lock_payload["full_dl_train_export_models"]),
            "resources_start": res_start,
            "resources_end": res_end,
        }
        with (RUN_DIR / "fit_and_lock_receipt.json").open("w") as f:
            json.dump(receipt, f, indent=2, sort_keys=True)
            f.write("\n")
        print(json.dumps(receipt, indent=2))
        return

    if args.mode == "outer_score":
        score_res = valguard.run_outer_score(
            oof_cache_path=OOF_CACHE_PATH,
            export_dir=EXPORT_DIR,
            capsule_fit_lock_path=capsule_fit_lock_path,
        )
        res_end = monitor.enforce_resources()

        with capsule_fit_lock_path.open("r") as f:
            lock_data = json.load(f)

        panels_path = RUN_DIR / "outer_cv_panels.npz"
        np.savez_compressed(
            panels_path,
            picks_panel=score_res["picks_panel"],
            u_panel=score_res["u_panel"],
            p_inv_oof=score_res["p_inv_oof"],
            p_inv_shuf=score_res["p_inv_shuf"],
            p_harm_oof=score_res["p_harm_oof"],
            arms=np.array(valguard.ARM_NAMES),
            seeds=np.array(valguard.H295_SEEDS, dtype=np.int32),
            null_seeds=np.array(valguard.H295_NULL_SEEDS, dtype=np.int32),
        )

        metrics_payload = {
            "wave": "H295",
            "slug": "h295_label_free_validity_guard",
            "lane": "rd_argon",
            "status": "COMPLETE",
            "verdict": score_res["decision"]["verdict"],
            "serving_promotion": "NONE",
            "optimizer_updates": 0,
            "neural_forwards": 0,
            "forbidden_reads_count": monitor.get_forbidden_reads_count(),
            "outer_fold_labels_read_before_lock": int(lock_data["outer_fold_labels_read_before_lock"]),
            "fit_lock_content_sha256": score_res["fit_lock_content_sha256"],
            "fit_lock_sha256": score_res["fit_lock_sha256"],
            "verified_model_files": int(score_res["verified_model_files"]),
            "h287_ens_vote2_regression": lock_data["h287_ens_vote2_regression"],
            "outer_fold_taus": [
                {
                    "seed": int(e["seed"]),
                    "null_seed": int(e["null_seed"]),
                    "fold": int(e["fold"]),
                    "tau_val": float(e["tau_val"]),
                    "tau_shuf": float(e["tau_shuf"]),
                    "tau_harm": float(e["tau_harm"]),
                }
                for e in lock_data["outer_fold_models"]
            ],
            "full_dl_train_export_models": lock_data["full_dl_train_export_models"],
            "decision": score_res["decision"],
            "dl_train_outer_cv": score_res["outer_cv_evaluation"],
            "resources": {
                "start": res_start,
                "end": res_end,
                "elapsed_fit_seconds": lock_data.get("elapsed_fit_seconds"),
                "elapsed_score_seconds": score_res["elapsed_score_seconds"],
            },
        }

        full_path = RUN_DIR / "metrics_full.json"
        with full_path.open("w") as f:
            json.dump(metrics_payload, f, indent=2, sort_keys=True)
            f.write("\n")

        cap_metrics_path = CAPSULE_DIR / "metrics.json"
        with cap_metrics_path.open("w") as f:
            json.dump(metrics_payload, f, indent=1, sort_keys=True)
            f.write("\n")

        sz = cap_metrics_path.stat().st_size
        if sz >= 50000:
            raise RuntimeError(f"metrics.json exceeds 50 KB limit: {sz} bytes")

        print(
            json.dumps(
                {
                    "verdict": score_res["decision"]["verdict"],
                    "metrics_bytes": sz,
                    "primary_check": score_res["decision"]["primary_check"],
                    "secondary_check": score_res["decision"]["secondary_check"],
                    "shuffled_null_check": score_res["decision"]["shuffled_null_check"],
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
