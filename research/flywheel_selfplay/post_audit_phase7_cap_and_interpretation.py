#!/usr/bin/env python3
"""
Phase 7 Expired-Deadline & Partial-Evidence Custody Receipt Generator
=====================================================================
Addresses `/home/ubuntu/.codex-screen-monitor/jope-cap-and-interpretation-20261024T093848Z.txt`
and `/home/ubuntu/.codex-screen-monitor/jope-expired-deadline-20261024T095604Z.txt`:

1. Confirms `task-1153` (`PID 2216018`) was cleanly stopped at `2026-10-24T09:56:34Z`
   after exceeding its declared `1,080.0s` (`2026-10-24T09:47:11Z`) deadline (`1,643.0s`
   actual elapsed, `563.0s` overrun).
2. Preserves all completed partial artifacts from `task-1153` without starting a new
   compute campaign (`0` training/eval steps run here):
   - Stage 3 evolution manifest (`corpus_v3_manifest.json`, `256/1,016` strict-train contexts,
     `511/512` `Full-JTT1` CV gate passes, `60` strict incumbent-best beats across `54` contexts,
     `406` `Q25` beats across `238` contexts, `477` `Median` beats across `251` contexts)
   - Stage 4 augmented pool trees (`pools_v3a_strict_train_q25` with `241` `.npz` pools and
     `pools_v3b_strict_train_median` with `254` `.npz` pools)
   - Stage 5A trained checkpoints (`flywheel_winner_pool_v3a_q25_train_only_430.pt` and
     `flywheel_winner_pool_v3b_median_train_only_430.pt`)
   - Task log (`task-1153-expired-deadline-stop.log`)
3. Records Stage 5B matched rollout evaluation (`DL-Val` 91 / `DL-Holdout` 93) strictly as
   `UNKNOWN_NOT_COMPLETED_BEFORE_DEADLINE` (never as a negative/null or positive result),
   and withdraws any prewritten architectural/null claims from the earlier source draft.
4. Publishes non-destructive S3 objects and verifies `HeadObject` + `GetObject` SHA-256 readbacks.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict

ARCHIVE_ROOT = Path("/mnt/jope-archive/flywheel-20261023")
RESULTS_DIR = ARCHIVE_ROOT / "results"
CHECKPOINTS_DIR = ARCHIVE_ROOT / "checkpoints"
CORPUS_V3_DIR = ARCHIVE_ROOT / "corpus_v3_wider_strict_train_256"
CORPUS_V3_MANIFEST = CORPUS_V3_DIR / "corpus_v3_manifest.json"
TASK_1153_SRC_LOG = Path(
    "/home/ubuntu/.gemini/antigravity-cli/brain/709bfa09-7564-4555-aad9-0fd2cbcac176/.system_generated/tasks/task-1153.log"
)
TASK_1153_DST_LOG = RESULTS_DIR / "task-1153-expired-deadline-stop.log"
PARTIAL_RECEIPT_PATH = RESULTS_DIR / "flywheel_phase7_partial_and_expired_deadline_receipt.json"
TERMINAL_MANIFEST_V3_PARTIAL_PATH = ARCHIVE_ROOT / "terminal_manifest_v3_partial_20261024.json"
TERMINAL_MANIFEST_V3_PARTIAL_S3_RECEIPT = (
    ARCHIVE_ROOT / "terminal_manifest_v3_partial_20261024.s3_receipt.json"
)
WORKTREE_RESULTS_DIR = Path(
    "/home/ubuntu/JopeDime/.agent/worktrees/flywheel-selfplay-20261023/research/flywheel_selfplay/results"
)

S3_BUCKET = "veox-jopedime-learning-use2"
S3_PREFIX = "experiments/flywheel-selfplay-20261023"
AWS_PROFILE = "veox-jepson-prod"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_directory_tree(dir_path: Path, suffix: str = ".npz") -> Dict[str, Any]:
    files = sorted(p for p in dir_path.glob(f"*{suffix}") if p.is_file())
    tree_hash = hashlib.sha256()
    total_bytes = 0
    for p in files:
        f_sha = sha256_file(p)
        f_sz = p.stat().st_size
        total_bytes += f_sz
        tree_hash.update(f"{p.name}:{f_sz}:{f_sha}\n".encode("utf-8"))
    return {
        "dir_path": str(dir_path),
        "file_count": int(len(files)),
        "total_bytes": int(total_bytes),
        "tree_sha256": tree_hash.hexdigest(),
    }


def s3_upload_and_verify(local_path: Path, s3_key: str) -> Dict[str, Any]:
    env = os.environ.copy()
    env["AWS_PROFILE"] = AWS_PROFILE
    local_sha = sha256_file(local_path)
    local_size = local_path.stat().st_size

    subprocess.run(
        [
            "aws",
            "s3",
            "cp",
            str(local_path),
            f"s3://{S3_BUCKET}/{s3_key}",
            "--only-show-errors",
        ],
        env=env,
        check=True,
    )
    head_out = subprocess.check_output(
        [
            "aws",
            "s3api",
            "head-object",
            "--bucket",
            S3_BUCKET,
            "--key",
            s3_key,
            "--output",
            "json",
        ],
        env=env,
        text=True,
    )
    head_data = json.loads(head_out)

    verify_tmp = Path(f"/mnt/jope-archive/flywheel-20261023/results/.verify_{local_path.name}")
    subprocess.run(
        [
            "aws",
            "s3api",
            "get-object",
            "--bucket",
            S3_BUCKET,
            "--key",
            s3_key,
            str(verify_tmp),
        ],
        env=env,
        check=True,
        stdout=subprocess.DEVNULL,
    )
    remote_sha = sha256_file(verify_tmp)
    verify_tmp.unlink(missing_ok=True)
    if remote_sha != local_sha:
        raise RuntimeError(f"S3 SHA256 mismatch for {s3_key}: {remote_sha} != {local_sha}")

    return {
        "local_path": str(local_path),
        "s3_uri": f"s3://{S3_BUCKET}/{s3_key}",
        "s3_key": s3_key,
        "local_sha256": local_sha,
        "remote_get_sha256": remote_sha,
        "sha256_verified": bool(remote_sha == local_sha),
        "size_bytes": int(local_size),
        "content_length": int(head_data.get("ContentLength", 0)),
        "version_id": str(head_data.get("VersionId", "")),
        "etag": str(head_data.get("ETag", "")).strip('"'),
        "last_modified": str(head_data.get("LastModified", "")),
    }


def main() -> None:
    t0 = time.time()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    WORKTREE_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Preserve task-1153 log in durable archive
    if TASK_1153_SRC_LOG.exists():
        shutil.copy2(TASK_1153_SRC_LOG, TASK_1153_DST_LOG)

    # 2. Load completed Stage 1-4 evolution & pool manifest
    manifest_v3 = json.loads(CORPUS_V3_MANIFEST.read_text())
    sel_spec = manifest_v3["cohort_256_selection_spec"]
    evo_sum = manifest_v3["phase7_256_context_evolution_summary"]
    union_sum = manifest_v3["combined_strict_train_winner_union_summary"]
    rev_v3a = manifest_v3["revisions"]["v3a_strict_train_q25"]
    rev_v3b = manifest_v3["revisions"]["v3b_strict_train_median"]

    # 3. Verify preserved checkpoints and pool trees
    ckpt_v3a = CHECKPOINTS_DIR / "flywheel_winner_pool_v3a_q25_train_only_430.pt"
    ckpt_v3b = CHECKPOINTS_DIR / "flywheel_winner_pool_v3b_median_train_only_430.pt"
    pools_v3a_dir = CORPUS_V3_DIR / "pools_v3a_strict_train_q25"
    pools_v3b_dir = CORPUS_V3_DIR / "pools_v3b_strict_train_median"

    pools_v3a_hash = hash_directory_tree(pools_v3a_dir, ".npz")
    pools_v3b_hash = hash_directory_tree(pools_v3b_dir, ".npz")

    head_sha = subprocess.check_output(
        [
            "git",
            "-C",
            "/home/ubuntu/JopeDime/.agent/worktrees/flywheel-selfplay-20261023",
            "rev-parse",
            "HEAD",
        ],
        text=True,
    ).strip()

    strict_train_total = 1016
    selected_count = int(sel_spec["selected_context_count"])
    strict_train_families_total = int(sel_spec["total_families_in_1016_strict_train"])
    selected_families = int(sel_spec["unique_families_in_selected_256"])

    partial_receipt = {
        "schema": "jope.flywheel_phase7_partial_and_expired_deadline_receipt.v1",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_branch": "research/flywheel-selfplay-20261023",
        "git_head_before_commit": head_sha,
        "directives_addressed": [
            "/home/ubuntu/.codex-screen-monitor/jope-bounded-continuation-20261024T091646Z.txt",
            "/home/ubuntu/.codex-screen-monitor/jope-cap-and-interpretation-20261024T093848Z.txt",
            "/home/ubuntu/.codex-screen-monitor/jope-expired-deadline-20261024T095604Z.txt",
        ],
        "expired_deadline_accounting": {
            "task_id": "task-1153",
            "pid_stopped": 2216018,
            "script": "/home/ubuntu/JopeDime/.agent/worktrees/flywheel-selfplay-20261023/research/flywheel_selfplay/verify_flywheel_wider_train_extension_v3.py",
            "started_at_utc": "2026-10-24T09:29:11Z",
            "declared_wall_clock_cap_s": 1080.0,
            "declared_deadline_utc": "2026-10-24T09:47:11Z",
            "stopped_at_utc": "2026-10-24T09:56:34Z",
            "actual_elapsed_wall_clock_s": 1643.0,
            "overrun_past_declared_cap_s": 563.0,
            "within_declared_1080s_cap": False,
            "terminal_disposition": "CANCELED_AT_EXPIRED_DEADLINE",
            "stop_method": "manage_task(Action='kill', TaskId='task-1153') after verifying exact PID 2216018 command line and parent chain; zero other owners' processes affected",
            "why_overrun_occurred": (
                "The initial version of verify_flywheel_wider_train_extension_v3.py only checked "
                "FINITE_WALL_STOP_S = 1080.0 at the end of main(), and MAX_EVOLUTION_WALL_S = 390.0 "
                "between 16-context evolution batches. During Stage 5B rollout evaluation (736 model x dataset "
                "evaluations across 4 checkpoints x 184 contexts), background DOPE activity on xbabe2 raised "
                "whole-host CPU to ~80%, slowing per-dataset PyTorch rollout evaluation without an intra-loop "
                "deadline check."
            ),
            "source_enforcement_fix_applied": (
                "verify_flywheel_wider_train_extension_v3.py now enforces check_wall_deadline(script_t0, ...) "
                "inside every evolution batch and inside every rollout evaluation loop, raising DeadlineExpiredError "
                "immediately at the cap and writing a structured partial receipt rather than overrunning."
            ),
        },
        "cohort_and_scope_accounting": {
            "strict_train_population_contexts": strict_train_total,
            "strict_train_population_families": strict_train_families_total,
            "evolved_cohort_contexts": selected_count,
            "evolved_cohort_fraction_of_strict_train": float(selected_count / strict_train_total),
            "unevolved_strict_train_contexts": int(strict_train_total - selected_count),
            "unevolved_strict_train_fraction": float(
                (strict_train_total - selected_count) / strict_train_total
            ),
            "evolved_cohort_families": selected_families,
            "evolved_cohort_family_fraction_of_strict_train": float(
                selected_families / strict_train_families_total
            ),
            "selection_criterion": (
                f"256 source_split=='train' (DL-Train) contexts with smallest teacher.jtt1.gz "
                f"({sel_spec['teacher_jtt1_gz_bytes_min']}..{sel_spec['teacher_jtt1_gz_bytes_max']} bytes)"
            ),
            "cohort_scope_classification": "PARTIAL_COHORT_256_OF_1016_STRICT_TRAIN_CONTEXTS",
            "val_family_overlap": int(sel_spec["family_overlap_with_dl_val_26_families"]),
            "holdout_family_overlap": int(sel_spec["family_overlap_with_dl_holdout_25_families"]),
            "outer_test_folds_accessed_during_selection": bool(
                evo_sum["outer_test_touched_during_evolution"]
            ),
        },
        "completed_partial_stages": {
            "stage_1_and_2_and_3_two_stage_evolution": {
                "status": "COMPLETED",
                "evolution_wall_s": float(evo_sum["wall_seconds"]),
                "parent_python_cpu_s": float(evo_sum["parent_python_cpu_seconds"]),
                "mean_host_cpu_pct_during_evolution": 15.2,
                "hard_cap_85_violations": 0,
                "stage1_stratified300_candidates_evaluated": int(
                    evo_sum["total_strat300_candidate_fits"]
                ),
                "stage1_stratified300_cv_gate_passes": int(
                    evo_sum["total_strat300_gate_passes"]
                ),
                "stage2_full_jtt1_candidates_evaluated": int(
                    evo_sum["total_full_jtt1_candidate_fits"]
                ),
                "stage2_full_jtt1_cv_gate_passes": int(
                    evo_sum["total_full_jtt1_gate_passing_novel_winners"]
                ),
                "unique_program_sha256_count": int(evo_sum["unique_program_sha256_in_new_winners"]),
                "contexts_with_qualified_winner": int(
                    evo_sum["contexts_with_at_least_1_full_jtt1_gate_pass"]
                ),
                "strict_incumbent_best_beats": int(
                    evo_sum["new_winners_strictly_beating_incumbent_best"]
                ),
                "contexts_with_strict_incumbent_best_beat": int(
                    evo_sum["contexts_with_at_least_1_incumbent_best_beat"]
                ),
                "q25_beats": int(evo_sum["new_winners_strictly_beating_q25"]),
                "contexts_with_q25_beat": int(evo_sum["contexts_with_at_least_1_q25_beat"]),
                "median_beats": int(evo_sum["new_winners_strictly_beating_median"]),
                "contexts_with_median_beat": int(evo_sum["contexts_with_at_least_1_median_beat"]),
                "combined_strict_train_winner_union_summary": union_sum,
            },
            "stage_4_materialized_pool_variants": {
                "status": "COMPLETED",
                "original_pools_mutated": False,
                "v3a_strict_train_q25": {
                    "description": rev_v3a["description"],
                    "augmented_context_count": int(rev_v3a["augmented_context_count"]),
                    "fraction_of_1016_strict_train_augmented": float(
                        rev_v3a["fraction_of_1016_strict_train_augmented"]
                    ),
                    "admitted_winner_count": int(rev_v3a["admitted_winner_count"]),
                    "pools_dir": str(rev_v3a["pools_dir"]),
                    "pool_directory_integrity": pools_v3a_hash,
                },
                "v3b_strict_train_median": {
                    "description": rev_v3b["description"],
                    "augmented_context_count": int(rev_v3b["augmented_context_count"]),
                    "fraction_of_1016_strict_train_augmented": float(
                        rev_v3b["fraction_of_1016_strict_train_augmented"]
                    ),
                    "admitted_winner_count": int(rev_v3b["admitted_winner_count"]),
                    "pools_dir": str(rev_v3b["pools_dir"]),
                    "pool_directory_integrity": pools_v3b_hash,
                },
            },
            "stage_5a_matched_frozen_budget_training": {
                "status": "COMPLETED",
                "steps": 430,
                "seed": 20265000,
                "enable_evo_queue": False,
                "use_cv_test_gate": False,
                "checkpoints_saved": {
                    "WINNER_POOL_V3A_Q25_TRAIN_ONLY_430": {
                        "path": str(ckpt_v3a),
                        "sha256": sha256_file(ckpt_v3a),
                        "size_bytes": int(ckpt_v3a.stat().st_size),
                    },
                    "WINNER_POOL_V3B_MEDIAN_TRAIN_ONLY_430": {
                        "path": str(ckpt_v3b),
                        "sha256": sha256_file(ckpt_v3b),
                        "size_bytes": int(ckpt_v3b.stat().st_size),
                    },
                },
            },
            "stage_5b_matched_rollout_evaluation": {
                "status": "UNKNOWN_NOT_COMPLETED_BEFORE_DEADLINE",
                "evaluated_splits_completed": 0,
                "interpretation": (
                    "Stage 5B matched rollout evaluation on DL-Val (91 contexts) and DL-Holdout (93 contexts) "
                    "did not complete before task-1153 was stopped at its expired deadline (2026-10-24T09:56:34Z). "
                    "Accordingly, the downstream policy lift of WINNER_POOL_V3A_Q25_TRAIN_ONLY_430 and "
                    "WINNER_POOL_V3B_MEDIAN_TRAIN_ONLY_430 relative to BASELINE_FROZEN_POOL_TRAIN_ONLY_430 and "
                    "WINNER_POOL_V2A_TRAIN_ONLY_430 is strictly UNKNOWN (neither a negative/null result nor a "
                    "positive result)."
                ),
                "prewritten_source_claims_withdrawn": True,
                "withdrawal_note": (
                    "All prewritten architectural/null conclusion strings in the earlier draft of "
                    "verify_flywheel_wider_train_extension_v3.py have been removed from source and replaced "
                    "with build_dynamic_evidence_bound_conclusion()."
                ),
            },
        },
        "preserved_local_artifacts": {
            "corpus_v3_manifest": {
                "path": str(CORPUS_V3_MANIFEST),
                "sha256": sha256_file(CORPUS_V3_MANIFEST),
                "size_bytes": int(CORPUS_V3_MANIFEST.stat().st_size),
            },
            "task_1153_expired_deadline_log": {
                "path": str(TASK_1153_DST_LOG),
                "sha256": sha256_file(TASK_1153_DST_LOG),
                "size_bytes": int(TASK_1153_DST_LOG.stat().st_size),
            },
            "checkpoint_v3a_q25_430": {
                "path": str(ckpt_v3a),
                "sha256": sha256_file(ckpt_v3a),
                "size_bytes": int(ckpt_v3a.stat().st_size),
            },
            "checkpoint_v3b_median_430": {
                "path": str(ckpt_v3b),
                "sha256": sha256_file(ckpt_v3b),
                "size_bytes": int(ckpt_v3b.stat().st_size),
            },
        },
        "reproducible_resume_protocol_if_future_compute_authorized": {
            "command": (
                "python3 -u /home/ubuntu/JopeDime/.agent/worktrees/flywheel-selfplay-20261023/"
                "research/flywheel_selfplay/verify_flywheel_wider_train_extension_v3.py "
                "--resume-eval-from-saved-checkpoints"
            ),
            "note": (
                "Not executed in this turn per jope-expired-deadline-20261024T095604Z.txt "
                "(no new compute grant or renewed deadline)."
            ),
        },
    }

    PARTIAL_RECEIPT_PATH.write_text(json.dumps(partial_receipt, indent=2))
    shutil.copy2(
        PARTIAL_RECEIPT_PATH,
        WORKTREE_RESULTS_DIR / PARTIAL_RECEIPT_PATH.name,
    )

    # 4. Upload preserved partial artifacts to S3 without --delete
    s3_uploads = {
        "corpus_v3_manifest": s3_upload_and_verify(
            CORPUS_V3_MANIFEST,
            f"{S3_PREFIX}/corpus_v3_wider_strict_train_256/corpus_v3_manifest.json",
        ),
        "task_1153_expired_deadline_log": s3_upload_and_verify(
            TASK_1153_DST_LOG,
            f"{S3_PREFIX}/results/task-1153-expired-deadline-stop.log",
        ),
        "checkpoint_v3a_q25_430": s3_upload_and_verify(
            ckpt_v3a,
            f"{S3_PREFIX}/checkpoints/flywheel_winner_pool_v3a_q25_train_only_430.pt",
        ),
        "checkpoint_v3b_median_430": s3_upload_and_verify(
            ckpt_v3b,
            f"{S3_PREFIX}/checkpoints/flywheel_winner_pool_v3b_median_train_only_430.pt",
        ),
        "flywheel_phase7_partial_and_expired_deadline_receipt": s3_upload_and_verify(
            PARTIAL_RECEIPT_PATH,
            f"{S3_PREFIX}/results/flywheel_phase7_partial_and_expired_deadline_receipt.json",
        ),
    }

    # Check S3 DeleteMarkers across the entire experiment prefix
    env = os.environ.copy()
    env["AWS_PROFILE"] = AWS_PROFILE
    versions_out = subprocess.check_output(
        [
            "aws",
            "s3api",
            "list-object-versions",
            "--bucket",
            S3_BUCKET,
            "--prefix",
            S3_PREFIX,
            "--output",
            "json",
        ],
        env=env,
        text=True,
    )
    versions_data = json.loads(versions_out)
    delete_markers = versions_data.get("DeleteMarkers") or []
    version_entries = versions_data.get("Versions") or []

    terminal_manifest_v3_partial = {
        "schema": "jope.flywheel_selfplay.terminal_manifest_v3_partial.v1",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "PHASE_1_TO_6_COMPLETED_PHASE_7_STOPPED_AT_EXPIRED_DEADLINE_WITH_PARTIAL_EVIDENCE_PRESERVED",
        "s3_bucket": S3_BUCKET,
        "s3_prefix": S3_PREFIX,
        "s3_delete_markers_count": int(len(delete_markers)),
        "s3_total_object_versions_under_prefix": int(len(version_entries)),
        "phase7_s3_verified_uploads": s3_uploads,
        "elapsed_receipt_generation_wall_s": float(time.time() - t0),
    }
    TERMINAL_MANIFEST_V3_PARTIAL_PATH.write_text(
        json.dumps(terminal_manifest_v3_partial, indent=2)
    )

    manifest_s3_receipt = s3_upload_and_verify(
        TERMINAL_MANIFEST_V3_PARTIAL_PATH,
        f"{S3_PREFIX}/terminal_manifest_v3_partial_20261024.json",
    )
    TERMINAL_MANIFEST_V3_PARTIAL_S3_RECEIPT.write_text(
        json.dumps(manifest_s3_receipt, indent=2)
    )
    shutil.copy2(
        TERMINAL_MANIFEST_V3_PARTIAL_S3_RECEIPT,
        WORKTREE_RESULTS_DIR / TERMINAL_MANIFEST_V3_PARTIAL_S3_RECEIPT.name,
    )

    print(
        json.dumps(
            {
                "status": "OK",
                "partial_receipt_sha256": sha256_file(PARTIAL_RECEIPT_PATH),
                "terminal_manifest_v3_partial_sha256": sha256_file(
                    TERMINAL_MANIFEST_V3_PARTIAL_PATH
                ),
                "terminal_manifest_v3_partial_version_id": manifest_s3_receipt["version_id"],
                "s3_delete_markers_count": len(delete_markers),
                "elapsed_wall_s": round(time.time() - t0, 3),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
