#!/usr/bin/env python3
"""
Phase 5D: Audit Closure, S3 Version Custody & External Terminal Manifest,
0/1/4 Full-JTT1 vs Stratified-300 HWM Qualification, and Frozen-Budget
Mechanism Acceptance Test (`jope-audit-closure-20261024T075615Z.txt`).

Satisfies all requirements:
  1. Preserves `flywheel_qualified_next_phase_audit.json` (`SHA256: 962d0a3f57058866e1eca04ccddc1f5cc10140b4d475cb44e6f34efa7ccb4a77`,
     S3 `VersionId: Fr8RpcYDdj_cJ_a_zuKwkbWQp.DOMjg7`) untouched as the immutable `task-905` audit record (`19/22` matched).
  2. Preserves every original S3 object version (`round-01/persistence_receipt.json` `800B` `O_0KIinzlzpH_sKZM5manSxE_wHy0ARr`,
     `results/flywheel_audit_verification.json` `38171B` `_uQOUhQhGPLuSIqSo_uHPZ014CuATzoV`, and
     `results/task-865-cancelled-before-s3-delete-receipt.json` `785B` `C0Ha2Cme14t62drJuGtyOfTOkY2DM4Ri`) in local custody,
     publishes immutable source-qualified finalized keys without overwriting historical evidence to manufacture equality,
     and avoids repeating `GetObject` on the 19 already-matched objects from `task-905`.
  3. Reports Full-JTT1 HWM beats as `0 / 1 / 4` by round (`ROUND_1=0`, `ROUND_2=1`, `ROUND_3=4`), explicitly
     distinguishing trigger-stage short-sample (`Stratified300JTT1`) scores from verification-stage (`Full-JTT1`)
     scores and novel evolved DSLs (`27 / 31 / 32`) from bank transfers (`4 / 0 / 0`) and pool replays (`1 / 1 / 0`).
  4. Explicitly records that the matched `510` vs `510` step gradient ablation CIs cross `0` and point estimates
     favor `NAIVE_LABEL_GRAFT_510` on `DL-Val` and `DL-Holdout`, so no causal directional-gradient rollout win is established.
  5. Executes the next frozen-budget (`510` optimizer steps on Full-Corpus `1,219` contexts, zero duplicate evolutionary
     GP campaign) mechanism & acceptance test:
       - `IN_POOL_STRUCT_DISTILL_510` (cosine-similarity projection of verified evolved winners onto
         in-pool candidate 16D AST embeddings with strict CV-only targets + greedy mean-ensemble evaluation `enable_ucb=False`)
       - `ROUND_3_GREEDY_MEAN_UCB0` and `DIAGNOSTIC_TRAIN_ONLY_GREEDY_UCB0` (removing test-time epistemic UCB noise `enable_ucb=False` vs `True`).
  6. Publishes an external terminal manifest (`terminal_manifest_20261024.json`) outside synced directories.
"""

import os
import sys

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
sys.dont_write_bytecode = True

import copy
import hashlib
import json
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from evo_batch_queue import build_base_gpu_slate_batch
from model import FlywheelGameNet
from rust_evaluator import (
    GLOBAL_CPU_GUARD,
    evaluate_gp_batch_rust_strat300,
    featurize_dsl_412,
)
from s3_persistence import AWS_PROFILE, LOCAL_S3_MIRROR
from selfplay_loop import (
    CHECKPOINTS_DIR,
    CORPUS_DIR,
    RESULTS_DIR,
    build_fold_safe_sos_priors,
    compute_component_bootstrap_summary,
    evaluate_episode_rollout_context,
    fit_fold_safe_context_scaler,
    load_all_pools_and_features,
    transform_context_features,
)
from verify_flywheel_qualified_next_phase import (
    CUSTODY_S3_DIR,
    fingerprint_jtt1_physical_train,
    sha256_file,
)

CLOSURE_REPORT_PATH = RESULTS_DIR / "flywheel_closure_and_next_phase_report.json"
TERMINAL_MANIFEST_PATH = Path("/mnt/jope-archive/flywheel-20261023/terminal_manifest_20261024.json")
TERMINAL_RECEIPT_PATH = Path("/mnt/jope-archive/flywheel-20261023/terminal_manifest_20261024.s3_receipt.json")
TASK905_AUDIT_PATH = RESULTS_DIR / "flywheel_qualified_next_phase_audit.json"
TASK905_EXPECTED_SHA256 = "962d0a3f57058866e1eca04ccddc1f5cc10140b4d475cb44e6f34efa7ccb4a77"

S3_BUCKET = "veox-jopedime-learning-use2"
S3_PREFIX = "experiments/flywheel-selfplay-20261023"


def preserve_original_versions_and_publish_source_qualified_files() -> dict:
    """
    1. Verify `flywheel_qualified_next_phase_audit.json` still matches `TASK905_EXPECTED_SHA256`.
    2. Download every historical version of the 3 discrepant keys by explicit `--version-id` into
       `CUSTODY_S3_DIR / "original_versions"`.
    3. Publish immutable source-qualified finalized files so both original and finalized bytes
       exist at unambiguous keys without overwriting historical evidence.
    """
    env = os.environ.copy()
    env["AWS_PROFILE"] = AWS_PROFILE

    task905_actual_sha = sha256_file(TASK905_AUDIT_PATH)
    assert task905_actual_sha == TASK905_EXPECTED_SHA256, (
        f"Task 905 audit JSON mutated! Expected {TASK905_EXPECTED_SHA256}, got {task905_actual_sha}"
    )

    orig_dir = CUSTODY_S3_DIR / "original_versions"
    orig_dir.mkdir(parents=True, exist_ok=True)

    ver_raw = subprocess.check_output(
        [
            "aws", "s3api", "list-object-versions",
            "--bucket", S3_BUCKET,
            "--prefix", S3_PREFIX,
            "--output", "json",
        ],
        env=env,
        text=True,
        timeout=30,
    )
    ver_doc = json.loads(ver_raw)
    all_versions = ver_doc.get("Versions") or []
    delete_markers = ver_doc.get("DeleteMarkers") or []

    preserved_historical_versions = []
    target_version_specs = [
        (
            f"{S3_PREFIX}/round-01/persistence_receipt.json",
            "O_0KIinzlzpH_sKZM5manSxE_wHy0ARr",
            "round-01_persistence_receipt.pre_sync_800B.O_0KIinzlzpH_sKZM5manSxE_wHy0ARr.json",
            "pre_sync_round01_receipt_uploaded_during_task740_055219Z",
        ),
        (
            f"{S3_PREFIX}/round-01/persistence_receipt.json",
            "75H4Wf31_uKWdyh39Ap9iY_yBvfW1KP5",
            "round-01_persistence_receipt.post_sync_806B.75H4Wf31_uKWdyh39Ap9iY_yBvfW1KP5.json",
            "post_sync_round01_receipt_mutated_locally_after_sync",
        ),
        (
            f"{S3_PREFIX}/results/flywheel_audit_verification.json",
            "_uQOUhQhGPLuSIqSo_uHPZ014CuATzoV",
            "flywheel_audit_verification.v1_070947Z_38171B._uQOUhQhGPLuSIqSo_uHPZ014CuATzoV.json",
            "phase5a_audit_v1_uploaded_070947Z",
        ),
        (
            f"{S3_PREFIX}/results/flywheel_audit_verification.json",
            "C6XWY8_slQdwda4xGnhlpHdKRx3tE40G",
            "flywheel_audit_verification.v2_0710Z_38339B.C6XWY8_slQdwda4xGnhlpHdKRx3tE40G.json",
            "phase5a_audit_v2_finalized_locally_0710Z",
        ),
        (
            f"{S3_PREFIX}/results/task-865-cancelled-before-s3-delete-receipt.json",
            "C0Ha2Cme14t62drJuGtyOfTOkY2DM4Ri",
            "task-865-cancelled-before-s3-delete-receipt.v1_075451Z.C0Ha2Cme14t62drJuGtyOfTOkY2DM4Ri.json",
            "uploaded_at_end_of_task905_after_part5_check",
        ),
    ]

    for s3_key, vid, local_fname, desc in target_version_specs:
        dest = orig_dir / local_fname
        if not dest.exists():
            subprocess.run(
                [
                    "aws", "s3api", "get-object",
                    "--bucket", S3_BUCKET,
                    "--key", s3_key,
                    "--version-id", vid,
                    str(dest),
                ],
                env=env,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=30,
            )
        preserved_historical_versions.append({
            "s3_key": s3_key,
            "version_id": vid,
            "custody_path": str(dest),
            "size_bytes": dest.stat().st_size,
            "sha256": sha256_file(dest),
            "provenance_note": desc,
        })

    source_qualified_uploads = [
        (
            orig_dir / "round-01_persistence_receipt.pre_sync_800B.O_0KIinzlzpH_sKZM5manSxE_wHy0ARr.json",
            "round-01/persistence_receipt.pre_sync_800B.json",
        ),
        (
            orig_dir / "round-01_persistence_receipt.post_sync_806B.75H4Wf31_uKWdyh39Ap9iY_yBvfW1KP5.json",
            "round-01/persistence_receipt.post_sync_806B.json",
        ),
        (
            orig_dir / "flywheel_audit_verification.v1_070947Z_38171B._uQOUhQhGPLuSIqSo_uHPZ014CuATzoV.json",
            "results/flywheel_audit_verification.v1_070947Z_38171B.json",
        ),
        (
            orig_dir / "flywheel_audit_verification.v2_0710Z_38339B.C6XWY8_slQdwda4xGnhlpHdKRx3tE40G.json",
            "results/flywheel_audit_verification.v2_0710Z_38339B.json",
        ),
    ]

    qualified_readbacks = []
    for src_path, rel_key in source_qualified_uploads:
        s3_uri = f"s3://{S3_BUCKET}/{S3_PREFIX}/{rel_key}"
        s3_key = f"{S3_PREFIX}/{rel_key}"
        subprocess.run(
            ["aws", "s3", "cp", str(src_path), s3_uri, "--only-show-errors"],
            env=env,
            check=True,
            timeout=30,
        )
        head = json.loads(
            subprocess.check_output(
                ["aws", "s3api", "head-object", "--bucket", S3_BUCKET, "--key", s3_key],
                env=env,
                text=True,
                timeout=20,
            )
        )
        rb_bytes = subprocess.check_output(
            ["aws", "s3", "cp", s3_uri, "-"],
            env=env,
            timeout=30,
        )
        l_sha = sha256_file(src_path)
        r_sha = hashlib.sha256(rb_bytes).hexdigest()
        qualified_readbacks.append({
            "rel_key": rel_key,
            "s3_uri": s3_uri,
            "local_size_bytes": src_path.stat().st_size,
            "remote_size_bytes": int(head.get("ContentLength", -1)),
            "local_sha256": l_sha,
            "remote_sha256_readback": r_sha,
            "s3_version_id": head.get("VersionId"),
            "s3_etag": (head.get("ETag") or "").strip('"'),
            "matched": (l_sha == r_sha) and (src_path.stat().st_size == int(head.get("ContentLength", -1))),
        })

    for local_p, rel_key in [
        (RESULTS_DIR / "task-865-cancelled-before-s3-delete-receipt.json", "results/task-865-cancelled-before-s3-delete-receipt.json"),
        (TASK905_AUDIT_PATH, "results/flywheel_qualified_next_phase_audit.json"),
    ]:
        s3_uri = f"s3://{S3_BUCKET}/{S3_PREFIX}/{rel_key}"
        s3_key = f"{S3_PREFIX}/{rel_key}"
        head = json.loads(
            subprocess.check_output(
                ["aws", "s3api", "head-object", "--bucket", S3_BUCKET, "--key", s3_key],
                env=env,
                text=True,
                timeout=20,
            )
        )
        rb_bytes = subprocess.check_output(
            ["aws", "s3", "cp", s3_uri, "-"],
            env=env,
            timeout=30,
        )
        l_sha = sha256_file(local_p)
        r_sha = hashlib.sha256(rb_bytes).hexdigest()
        qualified_readbacks.append({
            "rel_key": rel_key,
            "s3_uri": s3_uri,
            "local_size_bytes": local_p.stat().st_size,
            "remote_size_bytes": int(head.get("ContentLength", -1)),
            "local_sha256": l_sha,
            "remote_sha256_readback": r_sha,
            "s3_version_id": head.get("VersionId"),
            "s3_etag": (head.get("ETag") or "").strip('"'),
            "matched": (l_sha == r_sha) and (local_p.stat().st_size == int(head.get("ContentLength", -1))),
        })

    return {
        "task905_audit_immutable_sha256": task905_actual_sha,
        "task905_initial_matched_summary": "19/22 matched in task-905 at 2026-10-24T07:54:47Z; 3 discrepancies explained and reconciled below without overwriting historical versions or repeating the 19 already-matched GETs.",
        "discrepancy_root_causes": {
            "round-01/persistence_receipt.json": (
                "Self-mutating receipt sequencing in s3_persistence.py: persist_round_artifacts_to_s3() wrote "
                "persistence_receipt.json (800 bytes, uploaded: false) inside round-01/, ran `aws s3 sync` "
                "(uploading the 800-byte pre-sync receipt as VersionId O_0KIinzlzpH_sKZM5manSxE_wHy0ARr), and then "
                "rewrote persistence_receipt.json in-place locally to 806 bytes (uploaded: true, returncode: 0). "
                "Fixed in s3_persistence.py by writing an immutable round manifest prior to upload and writing "
                "post-upload sync telemetry to external `/mnt/jope-archive/flywheel-20261023/s3_sync_receipts/`."
            ),
            "results/flywheel_audit_verification.json": (
                "Two sequential runs of verify_flywheel_audit.py: v1 (38,171 bytes, SHA256 84d51750..., VersionId "
                "_uQOUhQhGPLuSIqSo_uHPZ014CuATzoV) was uploaded at 07:09:47Z; v2 (38,339 bytes, SHA256 1d4a1850...) "
                "finished locally at 07:10Z. Both v1 and v2 are preserved in S3 versioning and published at explicit "
                "source-qualified keys (`flywheel_audit_verification.v1_070947Z_38171B.json` and `flywheel_audit_verification.v2_0710Z_38339B.json`)."
            ),
            "results/task-865-cancelled-before-s3-delete-receipt.json": (
                "Sequencing order in verify_flywheel_qualified_next_phase.py: verify_and_publish_missing_s3_receipts_nondestructive() "
                "inspected RESULTS_DIR before main() executed the `aws s3 cp` upload of task-865-cancelled-before-s3-delete-receipt.json "
                "at 07:54:51Z (VersionId C0Ha2Cme14t62drJuGtyOfTOkY2DM4Ri)."
            ),
        },
        "s3_delete_markers_count": len(delete_markers),
        "preserved_historical_versions": preserved_historical_versions,
        "newly_verified_source_qualified_readbacks": qualified_readbacks,
        "reused_19_matched_objects_from_task905_without_repeated_get": True,
    }


def qualify_hwm_0_1_4_trigger_vs_full_and_identities() -> dict:
    """
    Report Full-JTT1 HWM beats as `0 / 1 / 4` by round (`ROUND_1=0`, `ROUND_2=1`, `ROUND_3=4`),
    explicitly distinguishing trigger-stage short-sample (`Stratified300JTT1`) scores from
    verification-stage (`Full-JTT1`) scores and novel evolved DSLs from replays.
    """
    task905_doc = json.loads(TASK905_AUDIT_PATH.read_text())
    part4_prev = task905_doc["part4_hwm_vs_new_champions_and_full_jtt1_audit"]

    hwm_by_round_summary = {}
    for r_idx in (1, 2, 3):
        r_key = f"ROUND_{r_idx}"
        r_prev = part4_prev["rounds"][r_key]
        enriched_hwms = []
        for hrec in r_prev["qualified_hwm_records"]:
            jpath = hrec["teacher_jtt1_path"]
            phys = fingerprint_jtt1_physical_train(jpath)
            strat_eval = evaluate_gp_batch_rust_strat300(
                teacher_jtt1_gz_path=jpath,
                candidates=[{"dsl": hrec["dsl"], "operator": hrec["operator"], "parents": []}],
                max_train_rows=300,
                max_workers=1,
            )[0]
            enriched = dict(hrec)
            enriched["physical_jtt1_n_train_rows"] = phys["n_train"]
            enriched["physical_jtt1_n_test_rows"] = phys["n_test"]
            enriched["physical_jtt1_n_features_p"] = phys["p"]
            enriched["trigger_stage_strat300_n_train_rows"] = min(300, phys["n_train"])
            enriched["trigger_stage_strat300_fitness"] = float(strat_eval["fitness"])
            enriched["trigger_stage_strat300_cv_ratio"] = float(strat_eval["cv_geomean_rmse_ratio"])
            enriched["trigger_stage_strat300_cv_gate_passed"] = bool(strat_eval["cv_gate_passed"])
            enriched["verification_stage_full_jtt1_n_train_rows"] = phys["n_train"]
            enriched["verification_stage_full_jtt1_fitness"] = float(hrec["full_jtt1_verified_fitness"])
            enriched["verification_stage_full_jtt1_cv_ratio"] = float(hrec["full_jtt1_verified_cv_ratio"])
            enriched["verification_stage_full_jtt1_outer_test_ratio"] = hrec["full_jtt1_verified_outer_test_ratio"]
            enriched["strat300_vs_full_jtt1_fitness_delta"] = float(
                strat_eval["fitness"] - float(hrec["full_jtt1_verified_fitness"])
            )
            enriched_hwms.append(enriched)

        hwm_by_round_summary[r_key] = {
            "active_contexts_evaluated": 16,
            "per_active_context_top1_added_to_champ_bank": r_prev["reported_new_champions_added_per_active_context"],
            "retained_top_k_winners_count": r_prev["retained_top_k_winners_count"],
            "genuinely_novel_evolved_dsl_winners_count": r_prev["genuinely_novel_evolved_dsl_winners_count"],
            "cross_context_champion_bank_transfer_winners_count": r_prev["cross_context_champion_bank_transfer_winners_count"],
            "same_context_pool_replay_winners_count": r_prev["same_context_pool_replay_winners_count"],
            "full_jtt1_strict_hwm_beat_count": r_prev["full_jtt1_strict_hwm_beat_count"],
            "qualified_hwm_records_with_strat300_and_full_jtt1": enriched_hwms,
        }

    return {
        "full_jtt1_hwm_beats_by_round_tuple": [
            hwm_by_round_summary["ROUND_1"]["full_jtt1_strict_hwm_beat_count"],
            hwm_by_round_summary["ROUND_2"]["full_jtt1_strict_hwm_beat_count"],
            hwm_by_round_summary["ROUND_3"]["full_jtt1_strict_hwm_beat_count"],
        ],
        "full_jtt1_hwm_beats_total": sum(
            hwm_by_round_summary[f"ROUND_{r}"]["full_jtt1_strict_hwm_beat_count"] for r in (1, 2, 3)
        ),
        "novel_evolved_top_k_winners_by_round_tuple": [
            hwm_by_round_summary["ROUND_1"]["genuinely_novel_evolved_dsl_winners_count"],
            hwm_by_round_summary["ROUND_2"]["genuinely_novel_evolved_dsl_winners_count"],
            hwm_by_round_summary["ROUND_3"]["genuinely_novel_evolved_dsl_winners_count"],
        ],
        "cross_context_bank_transfers_by_round_tuple": [
            hwm_by_round_summary["ROUND_1"]["cross_context_champion_bank_transfer_winners_count"],
            hwm_by_round_summary["ROUND_2"]["cross_context_champion_bank_transfer_winners_count"],
            hwm_by_round_summary["ROUND_3"]["cross_context_champion_bank_transfer_winners_count"],
        ],
        "same_context_pool_replays_by_round_tuple": [
            hwm_by_round_summary["ROUND_1"]["same_context_pool_replay_winners_count"],
            hwm_by_round_summary["ROUND_2"]["same_context_pool_replay_winners_count"],
            hwm_by_round_summary["ROUND_3"]["same_context_pool_replay_winners_count"],
        ],
        "rounds": hwm_by_round_summary,
    }


def load_verified_evolved_structural_bank() -> tuple[np.ndarray, np.ndarray]:
    """
    Load all gate-passing novel evolved GP winners from `round-01..03/scores/retained_scores_round_0*.jsonl`
    and extract their 16D structural projections (`proj16`) + CV fitness utility weights (`0` new GP evals).
    """
    vecs = []
    gains = []
    seen_shas = set()
    for r_idx in (1, 2, 3):
        p = LOCAL_S3_MIRROR / f"round-{r_idx:02d}" / "scores" / f"retained_scores_round_{r_idx:02d}.jsonl"
        if not p.exists():
            continue
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("retention_tier") != "top_k_winner" or not row.get("cv_gate_passed"):
                continue
            psha = row["program_sha256"]
            if psha in seen_shas:
                continue
            seen_shas.add(psha)
            _f412, proj16 = featurize_dsl_412(row["dsl"])
            norm = float(np.linalg.norm(proj16))
            if norm > 1e-8:
                vecs.append(proj16 / norm)
                gains.append(float(np.clip(-float(row["fitness"]), 0.1, 2.5)))
    if not vecs:
        return np.zeros((1, 16), dtype=np.float32), np.ones(1, dtype=np.float32)
    return np.stack(vecs, axis=0).astype(np.float32), np.array(gains, dtype=np.float32)


def train_in_pool_structural_distillation_510_steps(
    train_cids: list[str],
    pools: dict,
    ctx_scaled_584: np.ndarray,
    sos_by_cid: dict[str, np.ndarray],
    init_state_dict: dict,
    evolved_vecs: np.ndarray,
    evolved_gains: np.ndarray,
    device: torch.device,
    epochs: int = 10,
    batch_size: int = 24,
    max_slate: int = 192,
    seed: int = 20263000,
    distill_alpha: float = 0.08,
) -> tuple[FlywheelGameNet, dict]:
    """
    Train `FlywheelGameNet` from `init_state_dict` for the exact matched budget (`epochs=10`, `batch_size=24` ->
    `510` optimizer steps across `1,219` `DL-Train` contexts) using In-Pool Structural Distillation:
    Instead of pushing out-of-pool synthetic child vectors via `dir_margin_loss`, project the verified
    evolved GP winners (`evolved_vecs`) onto each context's in-pool candidate 16D structural projections (`s16`)
    via cosine similarity and softly anchor the in-pool Plackett-Luce target fitness (`use_cv_test_gate=False`).
    """
    t0 = time.monotonic()
    cpu_t0 = time.process_time()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    valid_train_cids = [c for c in train_cids if len(pools[c]["pids"]) >= 64]
    model = FlywheelGameNet(ctx_dim=584, prog_dim=412, proj_dim=16, d_model=128).to(device)
    model.load_state_dict(copy.deepcopy(init_state_dict))

    opt = torch.optim.AdamW(model.parameters(), lr=5.5e-4, weight_decay=2.5e-4)
    total_steps = epochs * max(1, (len(valid_train_cids) + batch_size - 1) // batch_size)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps, eta_min=5e-5)

    epoch_losses = []
    opt_steps = 0

    for ep in range(epochs):
        model.train()
        order = valid_train_cids.copy()
        random.shuffle(order)
        ep_loss_vals = []

        for b_start in range(0, len(order), batch_size):
            GLOBAL_CPU_GUARD.wait_for_headroom(max_wait_s=0.4)
            batch_cids = order[b_start : b_start + batch_size]
            if not batch_cids:
                continue

            batch_dict = build_base_gpu_slate_batch(
                batch_cids=batch_cids,
                pools=pools,
                ctx_scaled_584=ctx_scaled_584,
                sos_by_cid=sos_by_cid,
                round_idx=2,
                use_cv_test_gate=False,
            )

            # In-pool structural distillation: project verified evolved winners onto in-pool proj_b [B, M, 16]
            proj_b = batch_dict["proj_b"]          # [B, M, 16]
            target_f_b = batch_dict["target_f_b"].copy()  # [B, M]
            gate_b = batch_dict["gate_b"]          # [B, M]
            scale_b = batch_dict["scale_b"]        # [B]

            for bi in range(len(batch_cids)):
                s16_sub = proj_b[bi]  # [M, 16]
                s_norms = np.linalg.norm(s16_sub, axis=1, keepdims=True)
                s_unit = s16_sub / np.maximum(s_norms, 1e-8)
                cos_sim = s_unit @ evolved_vecs.T  # [M, N_evolved]
                top_sim = np.max(cos_sim * evolved_gains[None, :], axis=1)
                passing_mask = gate_b[bi] > 0.5
                if np.any(passing_mask):
                    sim_z = (top_sim - np.mean(top_sim[passing_mask])) / (np.std(top_sim[passing_mask]) + 1e-6)
                    target_f_b[bi, passing_mask] = (
                        target_f_b[bi, passing_mask]
                        - distill_alpha * float(scale_b[bi]) * np.clip(sim_z[passing_mask], -1.5, 2.0)
                    )

            ctx_t = torch.from_numpy(batch_dict["ctx_b"]).to(device)
            prog_t = torch.from_numpy(batch_dict["prog_b"]).to(device)
            proj_t = torch.from_numpy(proj_b).to(device)
            prior_t = torch.from_numpy(batch_dict["prior_b"]).to(device)
            scale_t = torch.from_numpy(scale_b).to(device)
            sos_t = torch.from_numpy(batch_dict["sos_b"]).to(device)
            target_f_t = torch.from_numpy(target_f_b).to(device)
            gate_t = torch.from_numpy(gate_b).to(device)
            rev_idx_t = torch.from_numpy(batch_dict["rev_idx_b"]).to(device)
            rev_f_t = torch.from_numpy(batch_dict["rev_f_b"]).to(device)
            rev_gate_t = torch.from_numpy(batch_dict["rev_gate_b"]).to(device)

            out = model.predict_costs(
                ctx_feat=ctx_t,
                prog_412=prog_t,
                proj_16=proj_t,
                base_prior_cost=prior_t,
                scale=scale_t,
                sos_prior=sos_t,
                revealed_indices=rev_idx_t,
                revealed_raw_f=rev_f_t,
                revealed_gate=rev_gate_t,
                enable_prior_correction=True,
                enable_struct_kernel=True,
                enable_cross_attn=True,
                enable_ucb=False,
            )

            pred_cost = out["total_cost"].float()
            fail_logit = out["fail_logit"]

            norm_target = torch.where(
                gate_t > 0.5,
                (target_f_t / scale_t.unsqueeze(-1)).float(),
                torch.full_like(pred_cost, 2.0),
            )

            _, top_k_idx = torch.topk(-norm_target, k=16, dim=-1)
            best_pred = torch.gather(pred_cost, dim=1, index=top_k_idx)
            best_targ = torch.gather(norm_target, dim=1, index=top_k_idx)

            gap_targ = (norm_target.unsqueeze(1) - best_targ.unsqueeze(2)).clamp(min=0.0, max=3.0)
            margin_pred = best_pred.unsqueeze(2) - pred_cost.unsqueeze(1) + 0.05 * gap_targ
            rank_loss = torch.mean(gap_targ * F.softplus(margin_pred * 4.0))

            pred_centered = pred_cost - pred_cost.mean(dim=-1, keepdim=True)
            targ_centered = norm_target - norm_target.mean(dim=-1, keepdim=True)
            huber_loss = F.smooth_l1_loss(pred_centered, targ_centered, beta=0.25)

            fail_target = 1.0 - gate_t
            gate_loss = F.binary_cross_entropy_with_logits(fail_logit, fail_target)

            loss = rank_loss + 0.35 * huber_loss + 0.20 * gate_loss
            ep_loss_vals.append(float(loss.detach().cpu().item()))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt_steps += 1

        if ep_loss_vals:
            epoch_losses.append(round(float(np.mean(ep_loss_vals)), 5))

    wall_s = round(time.monotonic() - t0, 2)
    cpu_s = round(time.process_time() - cpu_t0, 2)
    diff_sq = 0.0
    total_sq = 0.0
    for k, v in model.state_dict().items():
        if v.is_floating_point():
            v_f = v.detach().cpu().float()
            v0_f = init_state_dict[k].detach().cpu().float()
            total_sq += float(torch.sum(v_f * v_f).item())
            diff_sq += float(torch.sum((v_f - v0_f) ** 2).item())

    return model, {
        "mode": "in_pool_structural_distillation_510",
        "epochs": epochs,
        "batch_size": batch_size,
        "optimizer_steps": opt_steps,
        "evolved_winner_bank_size": int(len(evolved_gains)),
        "distill_alpha": distill_alpha,
        "use_cv_test_gate": False,
        "wall_seconds": wall_s,
        "cpu_process_seconds": cpu_s,
        "initial_epoch_loss": epoch_losses[0],
        "final_epoch_loss": epoch_losses[-1],
        "epoch_losses": epoch_losses,
        "param_l2_norm": round(float(np.sqrt(total_sq)), 5),
        "param_update_l2_norm": round(float(np.sqrt(diff_sq)), 5),
    }


def run_frozen_budget_mechanism_acceptance_test(
    pools: dict,
    ctx_raw_584: np.ndarray,
    device: torch.device,
) -> dict:
    """
    Execute the frozen-budget (`510` optimizer steps, `0` duplicate GP evolution) mechanism & acceptance test:
      1. Explicitly record the matched `510` vs `510` gradient ablation outcome from `task-905` (CIs cross `0`,
         point estimates favor `NAIVE_LABEL_GRAFT_510` on `DL-Val` and `DL-Holdout`, so no causal directional-gradient
         rollout win is established).
      2. Train `IN_POOL_STRUCT_DISTILL_510` (`510` steps from `flywheel_round_1_full_corpus.pt`, `seed=20263000`,
         `use_cv_test_gate=False`, `0` new GP evals) and evaluate on `DL-Val` (`91` contexts) and `DL-Holdout` (`93` contexts).
      3. Evaluate `ROUND_3_GREEDY_MEAN_UCB0` (`enable_ucb=False`) and `DIAGNOSTIC_TRAIN_ONLY_GREEDY_UCB0` (`enable_ucb=False`)
         on frozen checkpoints (`0` training steps) to test removing test-time epistemic UCB exploration noise.
    """
    dl_train_cids = sorted([cid for cid, p in pools.items() if p["rec"]["dl_split"] == "DL-Train"])
    dl_val_cids = sorted([cid for cid, p in pools.items() if p["rec"]["dl_split"] == "DL-Val"])
    dl_holdout_cids = sorted([cid for cid, p in pools.items() if p["rec"]["dl_split"] == "DL-Holdout"])
    strict_train_only_cids = sorted([
        cid for cid in dl_train_cids if pools[cid]["rec"]["source_split"] == "train"
    ])

    sos_full = build_fold_safe_sos_priors(dl_train_cids, dl_val_cids + dl_holdout_cids, pools)
    f_mask_all = np.ones(584, dtype=np.float32)
    med_full, iqr_full, m_full = fit_fold_safe_context_scaler(dl_train_cids, pools, ctx_raw_584, f_mask_all)
    ctx_scaled_full = transform_context_features(ctx_raw_584, med_full, iqr_full, m_full)

    sos_strict_train = build_fold_safe_sos_priors(strict_train_only_cids, dl_val_cids + dl_holdout_cids, pools)
    med_st, iqr_st, m_st = fit_fold_safe_context_scaler(strict_train_only_cids, pools, ctx_raw_584, f_mask_all)
    ctx_scaled_st = transform_context_features(ctx_raw_584, med_st, iqr_st, m_st)

    evolved_vecs, evolved_gains = load_verified_evolved_structural_bank()

    r1_full_sd = torch.load(CHECKPOINTS_DIR / "flywheel_round_1_full_corpus.pt", map_location=device)
    r2_full_sd = torch.load(CHECKPOINTS_DIR / "flywheel_round_2_full_corpus.pt", map_location=device)
    r3_full_sd = torch.load(CHECKPOINTS_DIR / "flywheel_round_3_full_corpus.pt", map_location=device)
    diag_r3_sd = torch.load(CHECKPOINTS_DIR / "flywheel_diagnostic_train_split_only_cv_r3.pt", map_location=device)

    model_distill_510, stats_distill_510 = train_in_pool_structural_distillation_510_steps(
        train_cids=dl_train_cids,
        pools=pools,
        ctx_scaled_584=ctx_scaled_full,
        sos_by_cid=sos_full,
        init_state_dict=r1_full_sd,
        evolved_vecs=evolved_vecs,
        evolved_gains=evolved_gains,
        device=device,
        epochs=10,
        batch_size=24,
        seed=20263000,
    )
    ckpt_distill_path = CHECKPOINTS_DIR / "flywheel_in_pool_struct_distill_510.pt"
    torch.save({k: v.detach().cpu().clone() for k, v in model_distill_510.state_dict().items()}, ckpt_distill_path)

    def _load_net(sd: dict) -> FlywheelGameNet:
        m = FlywheelGameNet(ctx_dim=584, prog_dim=412, proj_dim=16, d_model=128).to(device)
        m.load_state_dict(sd)
        m.eval()
        return m

    m_r1 = _load_net(r1_full_sd)
    m_r2 = _load_net(r2_full_sd)
    m_r3 = _load_net(r3_full_sd)
    m_diag_r3 = _load_net(diag_r3_sd)

    zero_ctx = np.zeros(584, dtype=np.float32)
    val_rows = []
    hold_rows = []

    for base_arm in ("STATIC", "SHUFFLED", "W1"):
        for cid in dl_val_cids:
            p = pools[cid]
            val_rows.append(
                evaluate_episode_rollout_context(cid, p, zero_ctx, np.zeros((len(p["pids"]), 2), dtype=np.float32), None, device, base_arm)
            )
        for cid in dl_holdout_cids:
            p = pools[cid]
            hold_rows.append(
                evaluate_episode_rollout_context(cid, p, zero_ctx, np.zeros((len(p["pids"]), 2), dtype=np.float32), None, device, base_arm)
            )

    eval_arms = [
        ("ROUND_1", m_r1, ctx_scaled_full, sos_full, "ROUND_1", True, False, False),
        ("ROUND_2", m_r2, ctx_scaled_full, sos_full, "ROUND_2", True, True, False),
        ("ROUND_3_UCB", m_r3, ctx_scaled_full, sos_full, "ROUND_3", True, True, True),
        ("ROUND_3_GREEDY_UCB0", m_r3, ctx_scaled_full, sos_full, "ROUND_2", True, True, False),
        ("IN_POOL_STRUCT_DISTILL_510", model_distill_510, ctx_scaled_full, sos_full, "ROUND_2", True, True, False),
        ("DIAGNOSTIC_TRAIN_ONLY_GREEDY_UCB0", m_diag_r3, ctx_scaled_st, sos_strict_train, "ROUND_2", True, True, False),
    ]

    for arm_name, m_obj, c_scaled, s_map, r_mode, en_sk, en_ca, en_ucb in eval_arms:
        for cid in dl_val_cids:
            p = pools[cid]
            r = evaluate_episode_rollout_context(
                cid, p, c_scaled[p["rec"]["index"]], s_map[cid], m_obj, device, r_mode,
                enable_struct_kernel=en_sk, enable_cross_attn=en_ca, enable_ucb=en_ucb,
            )
            r["arm"] = arm_name
            val_rows.append(r)
        for cid in dl_holdout_cids:
            p = pools[cid]
            r = evaluate_episode_rollout_context(
                cid, p, c_scaled[p["rec"]["index"]], s_map[cid], m_obj, device, r_mode,
                enable_struct_kernel=en_sk, enable_cross_attn=en_ca, enable_ucb=en_ucb,
            )
            r["arm"] = arm_name
            hold_rows.append(r)

    arm_list = [
        "STATIC",
        "SHUFFLED",
        "W1",
        "ROUND_1",
        "ROUND_2",
        "ROUND_3_UCB",
        "ROUND_3_GREEDY_UCB0",
        "IN_POOL_STRUCT_DISTILL_510",
        "DIAGNOSTIC_TRAIN_ONLY_GREEDY_UCB0",
    ]
    ref_arms = ("STATIC", "SHUFFLED", "W1", "ROUND_1", "ROUND_3_UCB")
    val_summary = compute_component_bootstrap_summary(val_rows, arm_list, ref_arms)
    hold_summary = compute_component_bootstrap_summary(hold_rows, arm_list, ref_arms)

    task905_doc = json.loads(TASK905_AUDIT_PATH.read_text())
    prev_ab = task905_doc["part2_matched_budget_gradient_direction_ablation"]

    return {
        "epistemic_label": "development_diagnostic",
        "matched_510_gradient_ablation_finding_from_task905": {
            "causal_directional_gradient_rollout_win_established": False,
            "explanation": (
                "In task-905's strictly matched 510 vs 510 step ablation (seed=20263000, use_cv_test_gate=False, "
                "0 fallback batches), PROPER_DIR_GRAD_510 reduced epoch training loss (14.110 -> 13.583) while "
                "NAIVE_LABEL_GRAFT_510 increased training loss (12.881 -> 13.153), but on single-round rollout "
                "utility the 98.75% CIs crossed 0 on all splits and point estimates favored NAIVE_LABEL_GRAFT_510 "
                "on DL-Val (+0.015660, CI [-0.008620, +0.052401]) and DL-Holdout (+0.001780, CI [-0.000698, +0.004956]). "
                "Therefore, no causal rollout win is claimed for out-of-pool directional margin loss."
            ),
            "task905_dl_val_contrast_proper_minus_naive": prev_ab["dl_val_91_contexts_summary"]["contrasts"]["PROPER_DIR_GRAD_510_minus_NAIVE_LABEL_GRAFT_510"],
            "task905_dl_holdout_contrast_proper_minus_naive": prev_ab["dl_holdout_93_contexts_summary"]["contrasts"]["PROPER_DIR_GRAD_510_minus_NAIVE_LABEL_GRAFT_510"],
        },
        "in_pool_structural_distillation_training_telemetry": {
            **stats_distill_510,
            "checkpoint_path": str(ckpt_distill_path),
            "checkpoint_sha256": sha256_file(ckpt_distill_path),
        },
        "dl_val_91_contexts_summary": val_summary,
        "dl_holdout_93_contexts_summary": hold_summary,
    }


def build_and_publish_external_terminal_manifest(closure_report: dict) -> dict:
    """
    Build the external terminal manifest (`/mnt/jope-archive/flywheel-20261023/terminal_manifest_20261024.json`)
    combining the 19 already-matched objects from `task-905` (without repeating their `GetObject` downloads),
    the newly verified source-qualified finalized files, the closure report, and the complete S3 version table.
    """
    env = os.environ.copy()
    env["AWS_PROFILE"] = AWS_PROFILE

    closure_rel_key = f"results/{CLOSURE_REPORT_PATH.name}"
    closure_s3_uri = f"s3://{S3_BUCKET}/{S3_PREFIX}/{closure_rel_key}"
    closure_s3_key = f"{S3_PREFIX}/{closure_rel_key}"
    subprocess.run(
        ["aws", "s3", "cp", str(CLOSURE_REPORT_PATH), closure_s3_uri, "--only-show-errors"],
        env=env,
        check=True,
        timeout=30,
    )
    c_head = json.loads(
        subprocess.check_output(
            ["aws", "s3api", "head-object", "--bucket", S3_BUCKET, "--key", closure_s3_key],
            env=env,
            text=True,
            timeout=20,
        )
    )
    c_rb = subprocess.check_output(["aws", "s3", "cp", closure_s3_uri, "-"], env=env, timeout=30)
    c_lsha = sha256_file(CLOSURE_REPORT_PATH)
    c_rsha = hashlib.sha256(c_rb).hexdigest()
    assert c_lsha == c_rsha

    task905_doc = json.loads(TASK905_AUDIT_PATH.read_text())
    reused_19_matched = [
        obj for obj in task905_doc["part5_s3_nondestructive_custody_and_readback"]["objects"]
        if obj.get("matched") is True
    ]

    ver_raw = subprocess.check_output(
        [
            "aws", "s3api", "list-object-versions",
            "--bucket", S3_BUCKET,
            "--prefix", S3_PREFIX,
            "--output", "json",
        ],
        env=env,
        text=True,
        timeout=30,
    )
    ver_doc = json.loads(ver_raw)
    all_versions = ver_doc.get("Versions") or []
    delete_markers = ver_doc.get("DeleteMarkers") or []

    newly_verified = list(closure_report["part1_s3_version_custody_and_reconciliation"]["newly_verified_source_qualified_readbacks"])
    newly_verified.append({
        "rel_key": closure_rel_key,
        "s3_uri": closure_s3_uri,
        "local_size_bytes": CLOSURE_REPORT_PATH.stat().st_size,
        "remote_size_bytes": int(c_head.get("ContentLength", -1)),
        "local_sha256": c_lsha,
        "remote_sha256_readback": c_rsha,
        "s3_version_id": c_head.get("VersionId"),
        "s3_etag": (c_head.get("ETag") or "").strip('"'),
        "matched": True,
    })

    terminal_manifest = {
        "schema": "jope.flywheel-external-terminal-manifest/v1",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "bucket": S3_BUCKET,
        "prefix": S3_PREFIX,
        "s3_total_object_versions": len(all_versions),
        "s3_delete_markers_count": len(delete_markers),
        "s3_delete_markers": delete_markers,
        "reused_19_matched_objects_count": len(reused_19_matched),
        "newly_verified_source_qualified_objects_count": len(newly_verified),
        "all_canonical_objects_verified_matched": (
            len(reused_19_matched) == 19 and all(x["matched"] for x in newly_verified)
        ),
        "reused_19_matched_objects_from_task905": reused_19_matched,
        "newly_verified_source_qualified_objects": newly_verified,
        "preserved_historical_versions_in_local_custody": closure_report["part1_s3_version_custody_and_reconciliation"]["preserved_historical_versions"],
        "s3_version_inventory": [
            {
                "Key": v.get("Key"),
                "VersionId": v.get("VersionId"),
                "IsLatest": v.get("IsLatest"),
                "Size": v.get("Size"),
                "ETag": (v.get("ETag") or "").strip('"'),
                "LastModified": v.get("LastModified"),
            }
            for v in all_versions
        ],
    }
    TERMINAL_MANIFEST_PATH.write_text(json.dumps(terminal_manifest, indent=2) + "\n")

    term_s3_uri = f"s3://{S3_BUCKET}/{S3_PREFIX}/{TERMINAL_MANIFEST_PATH.name}"
    term_s3_key = f"{S3_PREFIX}/{TERMINAL_MANIFEST_PATH.name}"
    subprocess.run(
        ["aws", "s3", "cp", str(TERMINAL_MANIFEST_PATH), term_s3_uri, "--only-show-errors"],
        env=env,
        check=True,
        timeout=30,
    )
    t_head = json.loads(
        subprocess.check_output(
            ["aws", "s3api", "head-object", "--bucket", S3_BUCKET, "--key", term_s3_key],
            env=env,
            text=True,
            timeout=20,
        )
    )
    t_rb = subprocess.check_output(["aws", "s3", "cp", term_s3_uri, "-"], env=env, timeout=30)
    t_lsha = sha256_file(TERMINAL_MANIFEST_PATH)
    t_rsha = hashlib.sha256(t_rb).hexdigest()
    assert t_lsha == t_rsha

    receipt = {
        "terminal_manifest_local_path": str(TERMINAL_MANIFEST_PATH),
        "terminal_manifest_s3_uri": term_s3_uri,
        "size_bytes": TERMINAL_MANIFEST_PATH.stat().st_size,
        "local_sha256": t_lsha,
        "remote_sha256_readback": t_rsha,
        "s3_version_id": t_head.get("VersionId"),
        "s3_etag": (t_head.get("ETag") or "").strip('"'),
        "matched": True,
        "verified_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    TERMINAL_RECEIPT_PATH.write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main():
    t0 = time.monotonic()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[Closure] Starting Phase 5D Audit Closure & Frozen-Budget Mechanism Acceptance Test on {device}...")

    _manifest, _champ_bank, ctx_raw_584, _h30_scales, pools = load_all_pools_and_features()

    print("[Closure] Part 1: Preserving original S3 versions and publishing source-qualified finalized files...")
    part1 = preserve_original_versions_and_publish_source_qualified_files()

    print("[Closure] Part 2: Qualifying 0/1/4 Full-JTT1 vs Stratified-300 HWM scores and novel-vs-replay identities...")
    part2 = qualify_hwm_0_1_4_trigger_vs_full_and_identities()

    print("[Closure] Part 3: Running frozen-budget (510 steps, 0 duplicate evolution) mechanism & acceptance test...")
    part3 = run_frozen_budget_mechanism_acceptance_test(pools, ctx_raw_584, device)

    closure_report = {
        "schema": "jope.flywheel-closure-and-next-phase-report/v1",
        "completed_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "wall_seconds": round(time.monotonic() - t0, 2),
        "cpu_throttle_guard": GLOBAL_CPU_GUARD.summary(),
        "part1_s3_version_custody_and_reconciliation": part1,
        "part2_hwm_0_1_4_and_novel_vs_replay_qualification": part2,
        "part3_frozen_budget_mechanism_acceptance_test": part3,
        "honest_final_status_and_next_executable_action": {
            "epistemic_status": "development_diagnostic (all evaluated cohorts — 850-Benchmark, DL-Val 91, DL-Holdout 93 — are reused development/diagnostic splits; no untouched confirmatory claim is made)",
            "staged_curriculum_finding": (
                "Across the 4-round staged self-play curriculum (task-740), mean utility on the 850-Context "
                "162-Component Benchmark improved from W1 (-1.598886) and ROUND_0 (-1.600697) to ROUND_3 (-1.609428; "
                "ROUND_3 - W1 = -0.010542, 98.75% CI [-0.019274, -0.004207]; ROUND_3 - ROUND_0 = -0.008731, "
                "98.75% CI [-0.017507, -0.002197]). Because rounds 1-3 progressively unlocked Landmark-V2 + SoS + "
                "Sim-Kernel, Multi-Modal 320D + Cross-Attention + CV-Test Gate, and Epistemic UCB alongside weight "
                "updates, this gain is attributed to the combined staged curriculum rather than isolated weight updates."
            ),
            "directional_gradient_ablation_finding": (
                "In the strictly step-matched 510 vs 510 optimizer-step ablation from ROUND_1 (seed=20263000, "
                "use_cv_test_gate=False, 0 fallback batches), PROPER_DIR_GRAD_510 decreased training loss "
                "(14.110 -> 13.583) whereas NAIVE_LABEL_GRAFT_510 increased training loss (12.881 -> 13.153), "
                "but single-round rollout utility CIs crossed 0 on all splits and point estimates favored "
                "NAIVE_LABEL_GRAFT_510 on DL-Val (+0.015660) and DL-Holdout (+0.001780). No causal rollout win "
                "for out-of-pool directional margin loss is established."
            ),
            "strict_cv_and_train_only_finding": (
                "Training strictly on CV fitness (use_cv_test_gate=False) on either all 1,219 DL-Train contexts "
                "(ROUND_2_STRICT_CV: -1.039840 vs W1 -1.033374, diff -0.006466, 98.75% CI [-0.016660, -0.000908]) "
                "or the 1,016 source_split == 'train' contexts only (DIAGNOSTIC_TRAIN_SPLIT_ONLY_CV_R3: -1.040163 "
                "vs W1 -1.033374, diff -0.006788, 98.75% CI [-0.016827, -0.000872]) achieved favorable_upper_below_zero = True "
                "vs W1 on the family-disjoint DL-Holdout diagnostic cohort."
            ),
            "hwm_and_novel_winner_finding": (
                "Across Rounds 1, 2, and 3, genuine novel evolved EditZero DSLs accounted for 27/32, 31/32, and "
                "32/32 retained top-K winners (vs 4/0/0 cross-context bank transfers and 1/1/0 pool replays), and "
                "strict Full-JTT1 HWM beats progressed 0 / 1 / 4 by round."
            ),
            "next_executable_action": (
                "Integrate the verified novel evolved GP winners (90 unique gate-passing DSLs across Rounds 1-3, "
                "including the 5 strict Full-JTT1 HWM winners) directly into the expanded per-context candidate "
                "pools (`corpus/pools/*.npz`) with strict CV-only targets (`use_cv_test_gate=False`) and greedy "
                "mean-ensemble rollout (`enable_ucb=False`) before the next sealed confirmatory benchmark evaluation."
            ),
        },
    }

    CLOSURE_REPORT_PATH.write_text(json.dumps(closure_report, indent=2) + "\n")
    print("[Closure] Part 4: Building and publishing external terminal S3 manifest...")
    term_receipt = build_and_publish_external_terminal_manifest(closure_report)
    print(
        f"[Closure] Completed in {closure_report['wall_seconds']:.2f}s. "
        f"Terminal manifest SHA256={term_receipt['local_sha256']} VersionId={term_receipt['s3_version_id']}"
    )


if __name__ == "__main__":
    main()
