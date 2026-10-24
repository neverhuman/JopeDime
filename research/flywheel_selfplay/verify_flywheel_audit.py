#!/usr/bin/env python3
"""
Phase 5: Post-Run Flywheel Verification, Optimizer/Parameter Trajectory Audit,
Strict CV-Only / Sealed-Excluded Generalization Proof, and S3 Receipt Verification.

Produces `/mnt/jope-archive/flywheel-20261023/results/flywheel_audit_verification.json`
and syncs it to `s3://veox-jopedime-learning-use2/experiments/flywheel-selfplay-20261023/results/`.
"""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import copy
import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
import torch

sys.dont_write_bytecode = True
THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from model import FlywheelGameNet
from rust_evaluator import GLOBAL_CPU_GUARD
from s3_persistence import sync_directory_to_s3
from selfplay_loop import (
    CHECKPOINTS_DIR,
    CORPUS_DIR,
    RESULTS_DIR,
    build_fold_safe_sos_priors,
    compute_component_bootstrap_summary,
    evaluate_episode_rollout_context,
    fit_fold_safe_context_scaler,
    load_all_pools_and_features,
    train_flywheel_model_for_round,
    transform_context_features,
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def audit_checkpoint_parameter_trajectory() -> dict:
    """
    Compute exact parameter L2 norms, successive round-to-round update norms ||theta^(r) - theta^(r-1)||_2,
    per-module update norms, and learned scalar gate values across ROUND_0 -> ROUND_1 -> ROUND_2 -> ROUND_3
    for fold0, fold1, and full_corpus checkpoints.
    """
    streams = ["fold0", "fold1", "full_corpus"]
    rounds = [0, 1, 2, 3]
    out = {}

    module_prefixes = [
        "ctx_in",
        "ctx_block",
        "landmark_router",
        "prog_in",
        "prog_block",
        "evolvability_head",
        "traj_attn",
        "gate_head",
    ]
    scalar_keys = [
        "alpha_prior",
        "alpha_struct",
        "alpha_attn",
        "alpha_ucb",
        "gate_barrier_weight",
    ]

    for stream in streams:
        stream_records = []
        prev_sd = None
        for r in rounds:
            ckpt_path = CHECKPOINTS_DIR / f"flywheel_round_{r}_{stream}.pt"
            if not ckpt_path.exists():
                continue
            sd = torch.load(ckpt_path, map_location="cpu")
            file_sha = sha256_file(ckpt_path)

            total_sq = 0.0
            total_params = 0
            for k, v in sd.items():
                if torch.is_floating_point(v):
                    vf = v.detach().float()
                    total_sq += float(torch.sum(vf * vf).item())
                    total_params += int(vf.numel())

            scalars = {
                sk: round(float(sd[sk].detach().float().item()), 6)
                for sk in scalar_keys
                if sk in sd
            }

            delta_info = None
            if prev_sd is not None:
                diff_sq = 0.0
                changed_tensors = 0
                total_tensors = 0
                mod_sq = {mp: 0.0 for mp in module_prefixes}
                for k, v in sd.items():
                    if torch.is_floating_point(v) and k in prev_sd:
                        total_tensors += 1
                        d = v.detach().float() - prev_sd[k].detach().float()
                        d_sq = float(torch.sum(d * d).item())
                        diff_sq += d_sq
                        if d_sq > 1e-12:
                            changed_tensors += 1
                        for mp in module_prefixes:
                            if k.startswith(mp):
                                mod_sq[mp] += d_sq
                l2_delta = math.sqrt(diff_sq)
                prev_l2 = stream_records[-1]["param_l2_norm"]
                delta_info = {
                    "from_round": r - 1,
                    "to_round": r,
                    "l2_update_norm": round(l2_delta, 6),
                    "relative_update_norm": round(l2_delta / max(1e-9, prev_l2), 6),
                    "changed_tensors": changed_tensors,
                    "total_tensors": total_tensors,
                    "module_l2_update_norms": {mp: round(math.sqrt(val), 6) for mp, val in mod_sq.items()},
                }

            stream_records.append({
                "round": r,
                "checkpoint_path": str(ckpt_path),
                "checkpoint_sha256": file_sha,
                "total_parameters": total_params,
                "param_l2_norm": round(math.sqrt(total_sq), 6),
                "learned_scalars": scalars,
                "update_from_previous_round": delta_info,
            })
            prev_sd = sd
        out[stream] = stream_records
    return out


def verify_s3_experiment_prefix(s3_prefix: str = "s3://veox-jopedime-learning-use2/experiments/flywheel-selfplay-20261023/") -> dict:
    """List all objects uploaded under `s3_prefix` using `veox-jepson-prod`."""
    env = os.environ.copy()
    env["AWS_PROFILE"] = env.get("JOPE_AWS_PROFILE", "veox-jepson-prod")
    proc = subprocess.run(
        ["aws", "s3", "ls", "--recursive", s3_prefix],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    lines = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
    total_bytes = 0
    objects = []
    for ln in lines:
        parts = ln.split(maxsplit=3)
        if len(parts) == 4:
            sz = int(parts[2])
            total_bytes += sz
            objects.append({"date": parts[0], "time": parts[1], "bytes": sz, "key": parts[3]})
    return {
        "s3_prefix": s3_prefix,
        "returncode": proc.returncode,
        "object_count": len(objects),
        "total_bytes": total_bytes,
        "objects": objects,
    }


def run_strict_cv_only_generalization_verification(device: torch.device) -> dict:
    """
    Explicitly verify that:
      1. Training Closure Exclusion: Zero `outer` test folds (`n_candidates == 0` for all `outer` records)
         and zero `sealed` meta-test families enter training (`DL-Train` has 0 `outer` and 0 `sealed` contexts).
      2. Strict CV-Only Generalization (`use_cv_test_gate=False`) vs CV-Test Gated (`use_cv_test_gate=True`)
         on untouched, frozen `DL-Val` (91 contexts) and `DL-Holdout` (93 contexts) pools (where pools are
         freshly loaded from disk without any in-place candidate expansion).
    """
    t0 = time.monotonic()
    manifest, champ_bank, ctx_raw_584, _, pools = load_all_pools_and_features()
    landmark_v2_48 = ctx_raw_584[:, 536:584]

    dl_train_cids = [cid for cid, p in pools.items() if p["rec"]["dl_split"] == "DL-Train"]
    dl_val_cids = [cid for cid, p in pools.items() if p["rec"]["dl_split"] == "DL-Val"]
    dl_holdout_cids = [cid for cid, p in pools.items() if p["rec"]["dl_split"] == "DL-Holdout"]

    # Verify input closure exclusions
    train_source_splits = {}
    for cid in dl_train_cids:
        sp = pools[cid]["rec"]["source_split"]
        train_source_splits[sp] = train_source_splits.get(sp, 0) + 1

    train_fams = {pools[c]["rec"]["family_id"] for c in dl_train_cids}
    val_fams = {pools[c]["rec"]["family_id"] for c in dl_val_cids}
    hold_fams = {pools[c]["rec"]["family_id"] for c in dl_holdout_cids}

    closure_audit = {
        "dl_train_count": len(dl_train_cids),
        "dl_val_count": len(dl_val_cids),
        "dl_holdout_count": len(dl_holdout_cids),
        "dl_train_source_splits": train_source_splits,
        "outer_contexts_in_dl_train": train_source_splits.get("outer", 0),
        "sealed_contexts_in_dl_train": train_source_splits.get("sealed", 0),
        "family_overlap_train_vs_val": len(train_fams & val_fams),
        "family_overlap_train_vs_holdout": len(train_fams & hold_fams),
        "family_overlap_val_vs_holdout": len(val_fams & hold_fams),
    }

    # Build fast deduplicated SoS priors on DL-Train
    sos_full = build_fold_safe_sos_priors(dl_train_cids, dl_val_cids + dl_holdout_cids, pools)
    f_mask_all = np.ones(584, dtype=np.float32)
    med_full, iqr_full, m_full = fit_fold_safe_context_scaler(dl_train_cids, pools, ctx_raw_584, f_mask_all)
    ctx_scaled_full = transform_context_features(ctx_raw_584, med_full, iqr_full, m_full)

    # Load Round 1 full_corpus checkpoint (which was trained with use_cv_test_gate=False!)
    r1_ckpt_path = CHECKPOINTS_DIR / "flywheel_round_1_full_corpus.pt"
    r1_sd = torch.load(r1_ckpt_path, map_location=device)

    # Train Strict CV-Only Round 2 & Round 3 (`use_cv_test_gate=False`, zero outer-test signal)
    shared_evo_cache = {}
    model_r2_cv_only, q_r2_cv = train_flywheel_model_for_round(
        round_idx=2,
        train_cids=dl_train_cids,
        monitor_cids=dl_val_cids,
        pools=pools,
        ctx_scaled_584=ctx_scaled_full,
        landmark_v2_48=landmark_v2_48,
        sos_by_cid=sos_full,
        champ_bank=champ_bank,
        device=device,
        prev_state_dict=r1_sd,
        use_cv_test_gate=False,
        enable_struct_kernel=True,
        enable_cross_attn=True,
        enable_evo_queue=True,
        naive_label_grafting=False,
        shared_evo_cache=shared_evo_cache,
        epochs=8,
        batch_size=24,
        seed=20262002,
    )
    r2_cv_sd = {k: v.detach().cpu().clone() for k, v in model_r2_cv_only.state_dict().items()}

    model_r3_cv_only, q_r3_cv = train_flywheel_model_for_round(
        round_idx=3,
        train_cids=dl_train_cids,
        monitor_cids=dl_val_cids,
        pools=pools,
        ctx_scaled_584=ctx_scaled_full,
        landmark_v2_48=landmark_v2_48,
        sos_by_cid=sos_full,
        champ_bank=champ_bank,
        device=device,
        prev_state_dict=r2_cv_sd,
        use_cv_test_gate=False,
        enable_struct_kernel=True,
        enable_cross_attn=True,
        enable_evo_queue=True,
        naive_label_grafting=False,
        shared_evo_cache=shared_evo_cache,
        epochs=8,
        batch_size=24,
        seed=20262003,
    )

    # Also load the saved Round 0, 1, 2, 3 checkpoints and evaluate ALL arms on 100% FROZEN DL-Holdout pools
    # (zero pool expansion on DL-Holdout) so independent generalization is completely unconfounded!
    models_by_arm = {}
    for r_i in (0, 1, 2, 3):
        m = FlywheelGameNet(ctx_dim=584, prog_dim=412, proj_dim=16, d_model=128, n_heads=4, dropout=0.10).to(device)
        m.load_state_dict(torch.load(CHECKPOINTS_DIR / f"flywheel_round_{r_i}_full_corpus.pt", map_location=device), strict=False)
        m.eval()
        models_by_arm[f"ROUND_{r_i}"] = m
    models_by_arm["ROUND_2_STRICT_CV"] = model_r2_cv_only
    models_by_arm["ROUND_3_STRICT_CV"] = model_r3_cv_only

    frozen_holdout_rows = []
    zero_ctx = np.zeros(584, dtype=np.float32)
    for base_arm in ("STATIC", "SHUFFLED", "W1"):
        for cid in sorted(dl_holdout_cids):
            p = pools[cid]
            frozen_holdout_rows.append(
                evaluate_episode_rollout_context(
                    cid, p, zero_ctx, np.zeros((len(p["pids"]), 2), dtype=np.float32), None, device, base_arm
                )
            )

    arm_specs = [
        ("ROUND_0", "ROUND_0", False, False, False, False),
        ("ROUND_1", "ROUND_1", True, True, False, False),
        ("ROUND_2_STRICT_CV", "ROUND_2", True, True, True, False),
        ("ROUND_3_STRICT_CV", "ROUND_3", True, True, True, True),
        ("ROUND_2", "ROUND_2", True, True, True, False),
        ("ROUND_3", "ROUND_3", True, True, True, True),
    ]
    for arm_name, base_round_tag, use_sos, use_sk, use_ca, use_ucb in arm_specs:
        m_eval = models_by_arm[arm_name]
        for cid in sorted(dl_holdout_cids):
            p = pools[cid]
            rec_idx = p["rec"]["index"]
            sos_m = sos_full[cid] if use_sos else np.zeros((len(p["pids"]), 2), dtype=np.float32)
            r_eval = evaluate_episode_rollout_context(
                cid,
                p,
                ctx_scaled_full[rec_idx],
                sos_m,
                m_eval,
                device,
                base_round_tag,
                enable_struct_kernel=use_sk,
                enable_cross_attn=use_ca,
                enable_ucb=use_ucb,
            )
            r_eval["arm"] = arm_name
            frozen_holdout_rows.append(r_eval)

    all_eval_arms = [
        "STATIC",
        "SHUFFLED",
        "W1",
        "ROUND_0",
        "ROUND_1",
        "ROUND_2_STRICT_CV",
        "ROUND_3_STRICT_CV",
        "ROUND_2",
        "ROUND_3",
    ]
    frozen_holdout_summary = compute_component_bootstrap_summary(
        frozen_holdout_rows,
        all_eval_arms,
        ("STATIC", "SHUFFLED", "W1", "ROUND_0", "ROUND_2_STRICT_CV"),
    )

    return {
        "wall_seconds": round(time.monotonic() - t0, 2),
        "input_closure_audit": closure_audit,
        "strict_cv_only_queue_telemetry": {
            "ROUND_2_STRICT_CV": q_r2_cv,
            "ROUND_3_STRICT_CV": q_r3_cv,
        },
        "frozen_pool_dl_holdout_93_contexts": frozen_holdout_summary,
    }


def main():
    t0 = time.monotonic()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[Phase 5 Audit] Running on {device}...")

    param_audit = audit_checkpoint_parameter_trajectory()
    print("[Phase 5 Audit] Audited checkpoint parameter trajectories across ROUND_0..ROUND_3.")

    strict_cv_verification = run_strict_cv_only_generalization_verification(device)
    print("[Phase 5 Audit] Completed Strict CV-Only frozen-pool generalization verification.")

    s3_inventory = verify_s3_experiment_prefix()
    print(f"[Phase 5 Audit] Verified S3 prefix ({s3_inventory['object_count']} objects, {s3_inventory['total_bytes']} bytes).")

    audit_report = {
        "schema": "jope.flywheel-audit-verification/v1",
        "completed_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "wall_seconds": round(time.monotonic() - t0, 2),
        "cpu_throttle_guard": GLOBAL_CPU_GUARD.summary(),
        "parameter_update_trajectory": param_audit,
        "strict_cv_only_and_frozen_pool_generalization": strict_cv_verification,
        "s3_experiment_inventory": s3_inventory,
    }

    out_path = RESULTS_DIR / "flywheel_audit_verification.json"
    out_path.write_text(json.dumps(audit_report, indent=2))
    sync_res = sync_directory_to_s3(
        RESULTS_DIR,
        "s3://veox-jopedime-learning-use2/experiments/flywheel-selfplay-20261023/results/",
    )
    audit_report["s3_results_sync"] = sync_res
    out_path.write_text(json.dumps(audit_report, indent=2))
    print(f"[Phase 5 Audit] Saved to {out_path} and synced to S3 (uploaded={sync_res['uploaded']}).")


if __name__ == "__main__":
    main()
