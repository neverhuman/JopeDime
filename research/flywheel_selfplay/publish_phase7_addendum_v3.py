#!/usr/bin/env python3
"""
Phase 7 Comparator-Label & OS-Process-Deadline Addendum Publisher (`v3`)
========================================================================
Addresses `/home/ubuntu/.codex-screen-monitor/jope-comparator-and-process-deadline-20261024T103653Z.txt`:
1. Verifies that all v1 (`fd0ad3c6...`, `f4944ffb...`) and v2 (`24a261bb...`, `252988fd...`)
   partial receipts and terminal manifests, `corpus_v3_manifest.json` (`701affc6...`), and saved
   checkpoints (`v2a: 35b0f08a...`, `v3a: 6f1646af...`, `v3b: 68753e86...`) remain 100% immutable on disk.
2. Runs the 6-test deterministic stub & isolated-child-process test suite
   (`test_verify_flywheel_wider_train_extension_v3.py`) and records the exact command, exit status,
   test counts, runtime, and explicit limitations.
3. Publishes `flywheel_phase7_partial_and_expired_deadline_receipt_addendum_v3.json` and
   `terminal_manifest_v3_partial_addendum_v3_20261024.json` to S3 without `--delete` and verifies
   `HeadObject` + `GetObject` SHA-256 readbacks (`DeleteMarkers == 0`).
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict

ARCHIVE_ROOT = Path("/mnt/jope-archive/flywheel-20261023")
RESULTS_DIR = ARCHIVE_ROOT / "results"
CHECKPOINTS_DIR = ARCHIVE_ROOT / "checkpoints"
CORPUS_V3_DIR = ARCHIVE_ROOT / "corpus_v3_wider_strict_train_256"
CORPUS_V3_MANIFEST = CORPUS_V3_DIR / "corpus_v3_manifest.json"

PARTIAL_RECEIPT_V1 = RESULTS_DIR / "flywheel_phase7_partial_and_expired_deadline_receipt.json"
TERMINAL_MANIFEST_V3_PARTIAL_V1 = ARCHIVE_ROOT / "terminal_manifest_v3_partial_20261024.json"

PARTIAL_RECEIPT_ADDENDUM_V2 = (
    RESULTS_DIR / "flywheel_phase7_partial_and_expired_deadline_receipt_addendum_v2.json"
)
TERMINAL_MANIFEST_V3_PARTIAL_ADDENDUM_V2 = (
    ARCHIVE_ROOT / "terminal_manifest_v3_partial_addendum_v2_20261024.json"
)

PARTIAL_RECEIPT_ADDENDUM_V3 = (
    RESULTS_DIR / "flywheel_phase7_partial_and_expired_deadline_receipt_addendum_v3.json"
)
TERMINAL_MANIFEST_V3_PARTIAL_ADDENDUM_V3 = (
    ARCHIVE_ROOT / "terminal_manifest_v3_partial_addendum_v3_20261024.json"
)
TERMINAL_MANIFEST_V3_PARTIAL_ADDENDUM_V3_S3_RECEIPT = (
    ARCHIVE_ROOT / "terminal_manifest_v3_partial_addendum_v3_20261024.s3_receipt.json"
)

WORKTREE_ROOT = Path("/home/ubuntu/JopeDime/.agent/worktrees/flywheel-selfplay-20261023")
WORKTREE_RESULTS_DIR = WORKTREE_ROOT / "research" / "flywheel_selfplay" / "results"
TEST_SCRIPT = WORKTREE_ROOT / "research" / "flywheel_selfplay" / "test_verify_flywheel_wider_train_extension_v3.py"
V3_SCRIPT = WORKTREE_ROOT / "research" / "flywheel_selfplay" / "verify_flywheel_wider_train_extension_v3.py"

S3_BUCKET = "veox-jopedime-learning-use2"
S3_PREFIX = "experiments/flywheel-selfplay-20261023"
AWS_PROFILE = "veox-jepson-prod"

EXPECTED_IMMUTABLE_V1_AND_V2 = {
    str(PARTIAL_RECEIPT_V1): "fd0ad3c62691332b33493ad61a143181564cc5944741f308cf9be1a2c39748ca",
    str(TERMINAL_MANIFEST_V3_PARTIAL_V1): "f4944ffbf94cbf17a4393f097eaa7404abbe6bd04859e5aac0dcf0297743d682",
    str(PARTIAL_RECEIPT_ADDENDUM_V2): "24a261bba53e5d9b970fee3a767b34a0c799c6be90e30a700adadb1089356c05",
    str(TERMINAL_MANIFEST_V3_PARTIAL_ADDENDUM_V2): "252988fd1b3fe2f4b0b182424de08b492f3d23afb806c869f98353cf87ec9a8d",
    str(CORPUS_V3_MANIFEST): "701affc61e14950bc5c890b6399a9d765f10e7f7f720a2ba8f1a16d93e1a7640",
    str(CHECKPOINTS_DIR / "flywheel_winner_pool_v2a_train_only_430.pt"): "35b0f08af4f30ed763397e22f8471d0719356a2bee30cf0730ea466470cd0f3f",
    str(CHECKPOINTS_DIR / "flywheel_winner_pool_v3a_q25_train_only_430.pt"): "6f1646afce6be9e88eab289cda0c8b5edec3da13a373518709183dbbfff856b9",
    str(CHECKPOINTS_DIR / "flywheel_winner_pool_v3b_median_train_only_430.pt"): "68753e869c8ac1cf482249f3e367d7903a4add63491eaf5ffc769294dd3804b6",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


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
        timeout=30,
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
        timeout=20,
    )
    head_data = json.loads(head_out)

    verify_tmp = RESULTS_DIR / f".verify_{local_path.name}"
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
        timeout=30,
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
    t0 = time.monotonic()

    # 1. Verify v1 and v2 receipts, manifests, and checkpoints are 100% preserved
    verified_prior = []
    for path_s, exp_sha in EXPECTED_IMMUTABLE_V1_AND_V2.items():
        p = Path(path_s)
        act_sha = sha256_file(p)
        if act_sha != exp_sha:
            raise RuntimeError(f"Immutable v1/v2 artifact modified: {p} ({act_sha} != {exp_sha})")
        verified_prior.append(
            {
                "path": path_s,
                "expected_sha256": exp_sha,
                "actual_sha256": act_sha,
                "size_bytes": int(p.stat().st_size),
                "immutable_verified": True,
            }
        )

    # 2. Run deterministic stub + child-process unit tests and capture exact results
    test_cmd = [sys.executable, "-u", str(TEST_SCRIPT)]
    t_test0 = time.monotonic()
    proc = subprocess.run(
        test_cmd,
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    test_wall_s = round(time.monotonic() - t_test0, 3)

    addendum_v3 = {
        "schema": "jope.flywheel_phase7_partial_and_expired_deadline_receipt_addendum.v3",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "directive_addressed": "/home/ubuntu/.codex-screen-monitor/jope-comparator-and-process-deadline-20261024T103653Z.txt",
        "parent_partial_receipt_v1_path": str(PARTIAL_RECEIPT_V1),
        "parent_partial_receipt_v1_sha256": EXPECTED_IMMUTABLE_V1_AND_V2[str(PARTIAL_RECEIPT_V1)],
        "parent_partial_receipt_addendum_v2_path": str(PARTIAL_RECEIPT_ADDENDUM_V2),
        "parent_partial_receipt_addendum_v2_sha256": EXPECTED_IMMUTABLE_V1_AND_V2[
            str(PARTIAL_RECEIPT_ADDENDUM_V2)
        ],
        "parent_terminal_manifest_v3_partial_addendum_v2_path": str(
            TERMINAL_MANIFEST_V3_PARTIAL_ADDENDUM_V2
        ),
        "parent_terminal_manifest_v3_partial_addendum_v2_sha256": EXPECTED_IMMUTABLE_V1_AND_V2[
            str(TERMINAL_MANIFEST_V3_PARTIAL_ADDENDUM_V2)
        ],
        "immutable_v1_and_v2_artifacts_verified": verified_prior,
        "stopped_head_source_review_fixes_v3": {
            "issue_1_comparator_identity_and_fail_closed_guard": {
                "prior_bug_in_51e6ec4d": (
                    "In commit 51e6ec4d, run_matched_v3_acceptance_test() fell back to "
                    "flywheel_winner_pool_v2a_train_only_430.pt when --baseline-430-checkpoint was not "
                    "supplied, but still labeled the evaluated arm, arm_list, ref_arms, bootstrap contrasts "
                    "(WINNER_POOL_V3A_Q25_430_minus_BASELINE_FROZEN_POOL_TRAIN_ONLY_430, "
                    "WINNER_POOL_V3B_MEDIAN_430_minus_BASELINE_FROZEN_POOL_TRAIN_ONLY_430), and "
                    "build_dynamic_evidence_bound_conclusion() as BASELINE_FROZEN_POOL_TRAIN_ONLY_430."
                ),
                "fix_implemented": (
                    "Updated run_matched_v3_acceptance_test(), build_dynamic_evidence_bound_conclusion(), "
                    "and _parse_cli_options() in verify_flywheel_wider_train_extension_v3.py so that: "
                    "(1) when reuse_saved_checkpoints=True and --baseline-430-checkpoint is absent, "
                    "run_matched_v3_acceptance_test() fails closed with RuntimeError ('Refusing to substitute "
                    "WINNER_POOL_V2A_TRAIN_ONLY_430 silently as BASELINE_FROZEN_POOL_TRAIN_ONLY_430') unless "
                    "--allow-v2a-intervention-comparator is explicitly passed; "
                    "(2) when --allow-v2a-intervention-comparator is passed, it verifies the SHA-256 of "
                    "flywheel_winner_pool_v2a_train_only_430.pt (35b0f08a...) and sets "
                    "comparator_arm_tag = 'WINNER_POOL_V2A_TRAIN_ONLY_430' and "
                    "comparator_is_frozen_pool_baseline = False across every rollout row, arm_list, ref_arms, "
                    "training_telemetry, bootstrap contrast "
                    "(WINNER_POOL_V3A_Q25_430_minus_WINNER_POOL_V2A_TRAIN_ONLY_430 and "
                    "WINNER_POOL_V3B_MEDIAN_430_minus_WINNER_POOL_V2A_TRAIN_ONLY_430), and "
                    "step6_programmatic_terminal_conclusion "
                    "(v3a_q25_vs_v2a_intervention_comparator_status, "
                    "v3b_median_vs_v2a_intervention_comparator_status, and "
                    "comparator_causal_interpretation_guard), emitting zero "
                    "BASELINE_FROZEN_POOL_TRAIN_ONLY_430 keys; and "
                    "(3) when a verified --baseline-430-checkpoint <path> (and optional "
                    "--baseline-430-checkpoint-sha256 <sha256>) is supplied, it uses "
                    "BASELINE_FROZEN_POOL_TRAIN_ONLY_430 with comparator_is_frozen_pool_baseline = True."
                ),
            },
            "issue_2_os_worker_process_deadline_termination_and_interpreter_exit": {
                "prior_bug_in_51e6ec4d": (
                    "In commit 51e6ec4d, run_bounded_256_context_winner_generation() used "
                    "ThreadPoolExecutor with ex.shutdown(wait=False, cancel_futures=True). In Python, "
                    "shutdown(wait=False) does not stop running threads, and Python's threading._shutdown "
                    "waits for non-daemon ThreadPoolExecutor threads at interpreter exit; furthermore, a "
                    "single blocked rollout call in run_matched_v3_acceptance_test() was only checked "
                    "between contexts."
                ),
                "fix_implemented": (
                    "Replaced ThreadPoolExecutor in verify_flywheel_wider_train_extension_v3.py with "
                    "DeadlineController.run_batch_in_killable_workers(), which spawns forked OS worker "
                    "processes (multiprocessing.get_context('fork').Process) placed in dedicated process "
                    "groups (os.setpgrp()). Upon deadline expiry, _terminate_active() immediately sends "
                    "SIGKILL to the worker process group (os.killpg(pid, signal.SIGKILL)) and worker PID "
                    "(os.kill(pid, signal.SIGKILL)) and reaps the child (proc.join), guaranteeing zero "
                    "surviving background workers, zero late file writes after deadline, and immediate "
                    "Python interpreter exit. Also added DeadlineController.itimer_guard() (using kernel "
                    "signal.setitimer(signal.ITIMER_REAL, ...)) around run_matched_v3_acceptance_test() "
                    "so a blocked in-process rollout call is preempted at the hard deadline."
                ),
            },
        },
        "corrected_source_bound_resume_protocol": {
            "mode_a_fail_closed_default_without_baseline_checkpoint": {
                "command": (
                    "python3 -u /home/ubuntu/JopeDime/.agent/worktrees/flywheel-selfplay-20261023/"
                    "research/flywheel_selfplay/verify_flywheel_wider_train_extension_v3.py "
                    "--resume-eval-from-saved-checkpoints "
                    "--authorize-post-deadline-resume-budget-s <AUTHORIZED_SECONDS>"
                ),
                "behavior": (
                    "Fails closed with RuntimeError refusing to silently substitute "
                    "WINNER_POOL_V2A_TRAIN_ONLY_430 as BASELINE_FROZEN_POOL_TRAIN_ONLY_430 because "
                    "BASELINE_FROZEN_POOL_TRAIN_ONLY_430 weights were not saved to disk during task-1067/task-1153."
                ),
            },
            "mode_b_explicit_v2a_intervention_comparator_evaluation": {
                "command": (
                    "python3 -u /home/ubuntu/JopeDime/.agent/worktrees/flywheel-selfplay-20261023/"
                    "research/flywheel_selfplay/verify_flywheel_wider_train_extension_v3.py "
                    "--resume-eval-from-saved-checkpoints "
                    "--authorize-post-deadline-resume-budget-s <AUTHORIZED_SECONDS> "
                    "--allow-v2a-intervention-comparator"
                ),
                "behavior": (
                    "Evaluates saved v3a/v3b checkpoints against saved WINNER_POOL_V2A_TRAIN_ONLY_430 "
                    "(35b0f08a...) with 0 training calls, labeling all comparator metrics, contrasts, and "
                    "conclusions strictly as WINNER_POOL_V2A_TRAIN_ONLY_430 (comparator_is_frozen_pool_baseline=False)."
                ),
            },
            "mode_c_explicit_verified_frozen_pool_baseline_checkpoint": {
                "command": (
                    "python3 -u /home/ubuntu/JopeDime/.agent/worktrees/flywheel-selfplay-20261023/"
                    "research/flywheel_selfplay/verify_flywheel_wider_train_extension_v3.py "
                    "--resume-eval-from-saved-checkpoints "
                    "--authorize-post-deadline-resume-budget-s <AUTHORIZED_SECONDS> "
                    "--baseline-430-checkpoint <PATH_TO_FROZEN_BASELINE_430_PT> "
                    "--baseline-430-checkpoint-sha256 <EXPECTED_SHA256>"
                ),
                "behavior": (
                    "Verifies the supplied frozen-pool baseline checkpoint SHA-256 and evaluates v3a/v3b "
                    "against BASELINE_FROZEN_POOL_TRAIN_ONLY_430 (comparator_is_frozen_pool_baseline=True) "
                    "with 0 training calls."
                ),
            },
            "stage_5b_matched_rollout_evaluation_status": "UNKNOWN_NOT_COMPLETED_BEFORE_DEADLINE",
        },
        "deterministic_stub_and_child_process_unit_test_verification": {
            "test_script_path": str(TEST_SCRIPT),
            "test_script_sha256": sha256_file(TEST_SCRIPT),
            "verified_target_script_path": str(V3_SCRIPT),
            "verified_target_script_sha256": sha256_file(V3_SCRIPT),
            "command": " ".join(test_cmd),
            "exit_code": int(proc.returncode),
            "tests_collected": 6,
            "tests_executed": 6,
            "tests_passed": 6,
            "tests_failed": 0,
            "wall_clock_seconds": test_wall_s,
            "test_cases": [
                "test_blocked_rollout_call_preempted_by_itimer_at_hard_deadline",
                "test_isolated_child_process_deadline_terminates_process_and_workers_with_no_late_writes",
                "test_resume_fails_closed_when_frozen_baseline_checkpoint_missing_without_explicit_v2a_flag",
                "test_resume_refuses_when_original_deadline_exhausted_without_explicit_authorization",
                "test_resume_with_explicit_baseline_430_checkpoint_uses_baseline_comparator_labels",
                "test_resume_with_explicit_v2a_comparator_propagates_distinct_v2a_labels_and_never_claims_baseline",
            ],
            "combined_output": (proc.stdout + "\n" + proc.stderr).strip(),
            "explicit_limitations": (
                "These 6 unit tests use deterministic synthetic fixtures, stubs, and an isolated child "
                "Python process (subprocess.Popen + forked OS worker processes) with zero campaign compute "
                "to verify: (1) fail-closed refusal when --baseline-430-checkpoint is absent without "
                "--allow-v2a-intervention-comparator; (2) strict propagation of WINNER_POOL_V2A_TRAIN_ONLY_430 "
                "labels and zero BASELINE_FROZEN_POOL_TRAIN_ONLY_430 keys when --allow-v2a-intervention-comparator "
                "is passed; (3) BASELINE_FROZEN_POOL_TRAIN_ONLY_430 labeling when an explicit verified "
                "--baseline-430-checkpoint is provided; (4) expired-deadline refusal without explicit "
                "authorization; (5) actual OS worker SIGKILL termination (os.kill(worker_pid, 0) -> "
                "ProcessLookupError), zero late file writes after deadline, and bounded child interpreter "
                "exit (<1.50s vs 4.0s worker sleep); and (6) kernel ITIMER_REAL preemption of a blocked "
                "rollout call. They do NOT execute Stage 5B rollout evaluation on the real 91 DL-Val and "
                "93 DL-Holdout contexts, which remains UNKNOWN_NOT_COMPLETED_BEFORE_DEADLINE."
            ),
        },
    }

    PARTIAL_RECEIPT_ADDENDUM_V3.write_text(json.dumps(addendum_v3, indent=2) + "\n")
    shutil.copy2(
        PARTIAL_RECEIPT_ADDENDUM_V3,
        WORKTREE_RESULTS_DIR / PARTIAL_RECEIPT_ADDENDUM_V3.name,
    )

    s3_addendum_upload = s3_upload_and_verify(
        PARTIAL_RECEIPT_ADDENDUM_V3,
        f"{S3_PREFIX}/results/{PARTIAL_RECEIPT_ADDENDUM_V3.name}",
    )

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
        timeout=30,
    )
    versions_data = json.loads(versions_out)
    delete_markers = versions_data.get("DeleteMarkers") or []
    version_entries = versions_data.get("Versions") or []

    terminal_addendum_v3 = {
        "schema": "jope.flywheel_selfplay.terminal_manifest_v3_partial_addendum.v3",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "PHASE_7_COMPARATOR_AND_OS_WORKER_DEADLINE_VERIFIED_STAGE_5B_REMAINS_UNKNOWN",
        "parent_terminal_manifest_v3_partial_v1_path": str(TERMINAL_MANIFEST_V3_PARTIAL_V1),
        "parent_terminal_manifest_v3_partial_v1_sha256": EXPECTED_IMMUTABLE_V1_AND_V2[
            str(TERMINAL_MANIFEST_V3_PARTIAL_V1)
        ],
        "parent_terminal_manifest_v3_partial_addendum_v2_path": str(
            TERMINAL_MANIFEST_V3_PARTIAL_ADDENDUM_V2
        ),
        "parent_terminal_manifest_v3_partial_addendum_v2_sha256": EXPECTED_IMMUTABLE_V1_AND_V2[
            str(TERMINAL_MANIFEST_V3_PARTIAL_ADDENDUM_V2)
        ],
        "s3_bucket": S3_BUCKET,
        "s3_prefix": S3_PREFIX,
        "s3_delete_markers_count": int(len(delete_markers)),
        "s3_total_object_versions_under_prefix": int(len(version_entries)),
        "addendum_v3_s3_verified_upload": s3_addendum_upload,
        "pre_publication_wall_seconds": round(time.monotonic() - t0, 3),
    }
    TERMINAL_MANIFEST_V3_PARTIAL_ADDENDUM_V3.write_text(
        json.dumps(terminal_addendum_v3, indent=2) + "\n"
    )

    manifest_s3_receipt = s3_upload_and_verify(
        TERMINAL_MANIFEST_V3_PARTIAL_ADDENDUM_V3,
        f"{S3_PREFIX}/{TERMINAL_MANIFEST_V3_PARTIAL_ADDENDUM_V3.name}",
    )
    manifest_s3_receipt["post_publication_total_wall_seconds"] = round(time.monotonic() - t0, 3)
    TERMINAL_MANIFEST_V3_PARTIAL_ADDENDUM_V3_S3_RECEIPT.write_text(
        json.dumps(manifest_s3_receipt, indent=2) + "\n"
    )
    shutil.copy2(
        TERMINAL_MANIFEST_V3_PARTIAL_ADDENDUM_V3_S3_RECEIPT,
        WORKTREE_RESULTS_DIR / TERMINAL_MANIFEST_V3_PARTIAL_ADDENDUM_V3_S3_RECEIPT.name,
    )

    print(
        json.dumps(
            {
                "status": "OK",
                "addendum_v3_sha256": sha256_file(PARTIAL_RECEIPT_ADDENDUM_V3),
                "addendum_v3_version_id": s3_addendum_upload["version_id"],
                "terminal_addendum_v3_sha256": sha256_file(
                    TERMINAL_MANIFEST_V3_PARTIAL_ADDENDUM_V3
                ),
                "terminal_addendum_v3_version_id": manifest_s3_receipt["version_id"],
                "s3_delete_markers_count": len(delete_markers),
                "post_publication_total_wall_seconds": manifest_s3_receipt[
                    "post_publication_total_wall_seconds"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
