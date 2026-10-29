#!/usr/bin/env python3
"""
Deterministic Stub & Child-Process Unit Tests for `verify_flywheel_wider_train_extension_v3.py`
===============================================================================================
Verifies all source-replay audit and comparator/process-deadline requirements from:
- `/home/ubuntu/.codex-screen-monitor/jope-source-replay-audit-20261024T101652Z.txt`
- `/home/ubuntu/.codex-screen-monitor/jope-comparator-and-process-deadline-20261024T103653Z.txt`
with zero dataset/training/evaluation campaign compute:

1. `test_resume_fails_closed_when_frozen_baseline_checkpoint_missing_without_explicit_v2a_flag`:
   Proves `main(["--resume-eval-from-saved-checkpoints", "--authorize-post-deadline-resume-budget-s", "60"])`
   fails closed with `RuntimeError` when `--baseline-430-checkpoint` is absent and
   `--allow-v2a-intervention-comparator` is not passed, refusing to silently substitute
   `WINNER_POOL_V2A_TRAIN_ONLY_430` as `BASELINE_FROZEN_POOL_TRAIN_ONLY_430`.
2. `test_resume_with_explicit_v2a_comparator_propagates_distinct_v2a_labels_and_never_claims_baseline`:
   Proves that passing `--allow-v2a-intervention-comparator` verifies the SHA-256 of
   `flywheel_winner_pool_v2a_train_only_430.pt`, executes ZERO evolution/pool-materialization/training/write
   calls, labels the comparator arm strictly as `WINNER_POOL_V2A_TRAIN_ONLY_430` (`comparator_is_frozen_pool_baseline=False`),
   emits `WINNER_POOL_V3A_Q25_430_minus_WINNER_POOL_V2A_TRAIN_ONLY_430` and
   `WINNER_POOL_V3B_MEDIAN_430_minus_WINNER_POOL_V2A_TRAIN_ONLY_430` contrasts, emits zero
   `BASELINE_FROZEN_POOL_TRAIN_ONLY_430` keys in `training_telemetry`, `arm_means`, `contrasts`, or
   `step6_programmatic_terminal_conclusion`, and includes `comparator_causal_interpretation_guard`.
3. `test_resume_with_explicit_baseline_430_checkpoint_uses_baseline_comparator_labels`:
   Proves that passing `--baseline-430-checkpoint` and `--baseline-430-checkpoint-sha256` verifies the
   supplied frozen-pool baseline checkpoint, sets `comparator_arm_tag == "BASELINE_FROZEN_POOL_TRAIN_ONLY_430"`
   (`comparator_is_frozen_pool_baseline=True`), and emits `v3a_q25_vs_baseline_status` /
   `v3b_median_vs_baseline_status` with zero training calls.
4. `test_resume_refuses_when_original_deadline_exhausted_without_explicit_authorization`:
   Proves that when `flywheel_phase7_partial_and_expired_deadline_receipt.json` records an exhausted
   original budget (`1643.0s > 1080.0s`), calling `--resume-eval-from-saved-checkpoints` without
   `--authorize-post-deadline-resume-budget-s` raises `DeadlineExpiredError` immediately before
   loading pools or running any stage.
5. `test_isolated_child_process_deadline_terminates_process_and_workers_with_no_late_writes`:
   Spawns a real isolated child Python process (`subprocess.Popen`) under `hard_cap_s = 15.30s`
   (`0.30s` effective budget with `reserve_s = 15.0s`) where context 0 finishes in `0.02s` and context 1's
   worker records its OS `worker_pid`, sleeps `4.0s` (ignoring `timeout_s`), and attempts a late file write
   (`late_write_after_deadline.txt`). Proves:
   - Real SIGKILL requests target the blocked worker between expiry and `< 0.45s`.
     Worker exit is checked immediately after the controller returns, before interpreter teardown.
     Work, cleanup and reporting take `< 0.65s` (0.30s work + one 0.20s bounded reap + 0.15s slack);
     the child exits within `< 2.50s`, including interpreter teardown.
   - The spawned worker OS process (`worker_pid`) is dead (`os.kill(worker_pid, 0)` raises `ProcessLookupError`).
   - Even after waiting an additional `0.4s`, `late_write_after_deadline.txt` does NOT exist.
   - Partial completed context 0 (`contexts_completed == 1`) is preserved and incomplete evaluation is
     recorded as `UNKNOWN_NOT_COMPLETED_BEFORE_DEADLINE`.
6. `test_blocked_rollout_call_preempted_by_itimer_at_hard_deadline`:
   Proves a blocked `evaluate_episode_rollout_context` call (`time.sleep(2.5)`, an interruptible syscall)
   inside `run_matched_v3_acceptance_test()` is preempted by `controller.itimer_guard` (`SIGALRM`) in `< 0.45s`
   and recorded as `UNKNOWN_NOT_COMPLETED_BEFORE_DEADLINE`.
7. `test_itimer_signal_delivery_boundary_vs_forked_worker_sigkill_under_masked_sigalrm`:
   Explicitly verifies the CPython `ITIMER_REAL` (`SIGALRM`) signal-delivery boundary vs forked OS worker
   `SIGKILL`: proves that while `SIGALRM` is blocked (`pthread_sigmask(SIG_BLOCK, {SIGALRM})`, modeling an
   uninterruptible native/C/CUDA call before returning to `PyErr_CheckSignals()`), `itimer_guard` cannot
   raise mid-section and raises `DeadlineExpiredError` only once signal delivery resumes at the Python
   boundary, whereas `run_batch_in_killable_workers()` kills a worker with `SIGKILL` in `< 0.55s` even
   when the worker masks `SIGALRM` and ignores `SIGTERM`.
8. `test_worker_deadline_dispatch_and_admission_use_the_remaining_budget`:
   Calls the actual controller with controlled OS/process/clock seams: an admitted worker is
   SIGKILL-targeted at expiry, pending work is not started, and an already-expired call starts none.
9. `test_noop_worker_termination_is_rejected_by_exit_evidence`:
   Deliberately suppresses termination of one owned blocked worker and proves the same exit-evidence
   assertion rejects it. Unpatched cleanup kills and joins only that test's Process handle.
"""
from __future__ import annotations

import contextlib
import json
import os
import shutil
import signal
import subprocess
import sys
import textwrap
import time
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import verify_flywheel_wider_train_extension_v3 as v3
from model import FlywheelGameNet


def _resolve_scratch_test_root() -> Path:
    """Return a non-/tmp scratch directory inside the repository target directory."""
    repo_target = THIS_DIR.parents[1] / "target"
    return repo_target / f"phase7_unit_tests_{os.getpid()}"


SCRATCH_TEST_ROOT = _resolve_scratch_test_root()


def _make_synthetic_pool_entry(cid: str, dl_split: str, source_split: str, idx: int, pool_path: Path) -> dict:
    rng = np.random.RandomState(100 + idx)
    n_cands = 68
    prog = rng.randn(n_cands, 412).astype(np.float32)
    fit = np.linspace(-0.8, 0.4, n_cands, dtype=np.float64)
    lr = np.linspace(-0.5, 0.2, n_cands, dtype=np.float64)
    gate = np.ones(n_cands, dtype=np.bool_)
    pids = [f"{cid[:8]}_p{k:04d}" + ("0" * 48) for k in range(n_cands)]
    np.savez_compressed(
        pool_path,
        program=prog,
        fitness=fit,
        logratio=lr,
        gate=gate,
        program_sha256=np.array([s.encode("ascii") for s in pids], dtype="S64"),
    )
    return {
        "rec": {
            "index": idx,
            "context_sha256": cid,
            "dataset_id": f"ds_{idx}",
            "view_id": f"view_{idx}",
            "family_id": f"fam_{dl_split}_{idx}",
            "source_split": source_split,
            "dl_split": dl_split,
            "pool_path": str(pool_path),
            "teacher_jtt1_path": str(pool_path),
            "n_candidates": n_cands,
        },
        "program": prog,
        "proj16": np.zeros((n_cands, 16), dtype=np.float64),
        "struct_sig": np.zeros((n_cands, 24), dtype=np.float32),
        "fitness": fit,
        "logratio": lr,
        "gate": gate,
        "pids": pids,
        "prior_cost": fit.copy(),
        "scale": 1.0,
    }


def _fast_eval_stub(cid, p, c_scaled, s_map, m_obj, device, arm, **kwargs):
    return {
        "context_sha256": cid,
        "component": p["rec"]["family_id"],
        "union_component": p["rec"]["family_id"],
        "arm": arm,
        "utility": -1.1,
        "best_F_by_budget": {"8": -0.9, "16": -1.0, "32": -1.1, "64": -1.4},
        "rollout_gate_failure_counts": {"8": 0, "16": 0, "32": 0, "64": 0},
        "gate_failure_fraction_by_budget": {"8": 0.0, "16": 0.0, "32": 0.0, "64": 0.0},
    }


def _fast_bootstrap_stub(rows, arms, reference_arms, seed=20261022):
    contrasts = {}
    for a in arms:
        for ref in reference_arms:
            if a != ref:
                contrasts[f"{a}_minus_{ref}"] = {
                    "mean": 0.0,
                    "CI_central_98_75": [-0.01, 0.01],
                }
    return {
        "n_contexts": len({r["context_sha256"] for r in rows}),
        "arm_means": {a: -1.1 for a in arms},
        "contrasts": contrasts,
    }


def _masked_sigalrm_worker(item: dict) -> int:
    if item["idx"] == 0:
        return 100
    signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGALRM})
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    Path(item["worker_pid_path"]).write_text(str(os.getpid()))
    time.sleep(3.0)
    Path(item["late_marker_path"]).write_text("UNEXPECTED_LATE_WRITE")
    return 200


def _deadline_waiting_worker(item: dict) -> None:
    Path(item["ready_path"]).write_text("ready")
    time.sleep(4.0)


class TestPhase7SourceReplayAudit(unittest.TestCase):
    def setUp(self) -> None:
        if SCRATCH_TEST_ROOT.exists():
            shutil.rmtree(SCRATCH_TEST_ROOT)
        SCRATCH_TEST_ROOT.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        if SCRATCH_TEST_ROOT.exists():
            shutil.rmtree(SCRATCH_TEST_ROOT)

    def _build_fixture_env(self) -> dict:
        ckpt_dir = SCRATCH_TEST_ROOT / "checkpoints"
        res_dir = SCRATCH_TEST_ROOT / "results"
        v3_dir = SCRATCH_TEST_ROOT / "corpus_v3_wider_strict_train_256"
        v3a_dir = v3_dir / "pools_v3a_strict_train_q25"
        v3b_dir = v3_dir / "pools_v3b_strict_train_median"
        orig_pools_dir = SCRATCH_TEST_ROOT / "orig_pools"
        for d in (ckpt_dir, res_dir, v3a_dir, v3b_dir, orig_pools_dir):
            d.mkdir(parents=True, exist_ok=True)

        cid_tr = "a" * 64
        cid_val = "b" * 64
        cid_hold = "c" * 64

        p_tr = _make_synthetic_pool_entry(cid_tr, "DL-Train", "train", 0, orig_pools_dir / f"{cid_tr}.npz")
        p_val = _make_synthetic_pool_entry(cid_val, "DL-Val", "val", 1, orig_pools_dir / f"{cid_val}.npz")
        p_hold = _make_synthetic_pool_entry(cid_hold, "DL-Holdout", "test", 2, orig_pools_dir / f"{cid_hold}.npz")
        pools_frozen = {cid_tr: p_tr, cid_val: p_val, cid_hold: p_hold}

        # Copy cid_tr pool into v3a and v3b directories as immutable materialized pools
        v3a_npz = v3a_dir / f"{cid_tr}.npz"
        v3b_npz = v3b_dir / f"{cid_tr}.npz"
        shutil.copy2(orig_pools_dir / f"{cid_tr}.npz", v3a_npz)
        shutil.copy2(orig_pools_dir / f"{cid_tr}.npz", v3b_npz)
        v3a_sha = v3.sha256_file(v3a_npz)
        v3b_sha = v3.sha256_file(v3b_npz)

        manifest_v3_path = v3_dir / "corpus_v3_manifest.json"
        manifest_v3_doc = {
            "schema": "jope.flywheel-corpus-v3-wider-strict-train-256/v1",
            "cohort_256_selection_spec": {
                "target_context_count": 256,
                "selected_context_count": 256,
                "unique_families_in_selected_256": 123,
                "total_families_in_1016_strict_train": 170,
            },
            "phase7_256_context_evolution_summary": {
                "contexts_targeted": 256,
                "contexts_completed": 256,
                "total_strat300_candidate_fits": 1536,
                "total_strat300_gate_passes": 1330,
                "total_full_jtt1_candidate_fits": 512,
                "total_full_jtt1_gate_passing_novel_winners": 511,
                "contexts_with_at_least_1_full_jtt1_gate_pass": 256,
                "new_winners_strictly_beating_incumbent_best": 60,
                "contexts_with_at_least_1_incumbent_best_beat": 54,
                "new_winners_strictly_beating_q25": 406,
                "contexts_with_at_least_1_q25_beat": 238,
                "new_winners_strictly_beating_median": 477,
                "contexts_with_at_least_1_median_beat": 251,
            },
            "combined_strict_train_winner_union_summary": {
                "combined_unique_context_program_pairs": 539,
            },
            "revisions": {
                "v3a_strict_train_q25": {
                    "augmented_context_count": 241,
                    "admitted_winner_count": 430,
                    "pools_dir": str(v3a_dir),
                    "materialized_pools": [
                        {
                            "context_sha256": cid_tr,
                            "new_pool_path": str(v3a_npz),
                            "new_pool_sha256": v3a_sha,
                            "admitted_winner_count": 2,
                        }
                    ],
                },
                "v3b_strict_train_median": {
                    "augmented_context_count": 254,
                    "admitted_winner_count": 504,
                    "pools_dir": str(v3b_dir),
                    "materialized_pools": [
                        {
                            "context_sha256": cid_tr,
                            "new_pool_path": str(v3b_npz),
                            "new_pool_sha256": v3b_sha,
                            "admitted_winner_count": 3,
                        }
                    ],
                },
            },
        }
        manifest_v3_path.write_text(json.dumps(manifest_v3_doc, indent=2) + "\n")

        # Save deterministic tiny checkpoints
        torch.manual_seed(7)
        net = FlywheelGameNet(ctx_dim=584, prog_dim=412, proj_dim=16, d_model=128)
        sd = {k: v.cpu() for k, v in net.state_dict().items()}
        for name in (
            "flywheel_round_1_full_corpus.pt",
            "flywheel_winner_pool_v2a_train_only_430.pt",
            "flywheel_winner_pool_v3a_q25_train_only_430.pt",
            "flywheel_winner_pool_v3b_median_train_only_430.pt",
        ):
            torch.save(sd, ckpt_dir / name)

        expected_partial = {
            str(manifest_v3_path): v3.sha256_file(manifest_v3_path),
            str(ckpt_dir / "flywheel_winner_pool_v3a_q25_train_only_430.pt"): v3.sha256_file(
                ckpt_dir / "flywheel_winner_pool_v3a_q25_train_only_430.pt"
            ),
            str(ckpt_dir / "flywheel_winner_pool_v3b_median_train_only_430.pt"): v3.sha256_file(
                ckpt_dir / "flywheel_winner_pool_v3b_median_train_only_430.pt"
            ),
        }
        v2a_sha = v3.sha256_file(ckpt_dir / "flywheel_winner_pool_v2a_train_only_430.pt")

        partial_receipt_path = res_dir / "flywheel_phase7_partial_and_expired_deadline_receipt.json"
        partial_receipt_path.write_text(
            json.dumps(
                {
                    "expired_deadline_accounting": {
                        "task_id": "task-1153",
                        "declared_wall_clock_cap_s": 1080.0,
                        "actual_elapsed_wall_clock_s": 1643.0,
                        "within_declared_1080s_cap": False,
                        "terminal_disposition": "CANCELED_AT_EXPIRED_DEADLINE",
                    }
                },
                indent=2,
            )
        )

        return {
            "ckpt_dir": ckpt_dir,
            "res_dir": res_dir,
            "v3_dir": v3_dir,
            "v3a_npz": v3a_npz,
            "v3b_npz": v3b_npz,
            "manifest_v3_path": manifest_v3_path,
            "partial_receipt_path": partial_receipt_path,
            "expected_partial": expected_partial,
            "v2a_sha": v2a_sha,
            "pools_frozen": pools_frozen,
            "ctx_raw_584": np.zeros((3, 584), dtype=np.float32),
        }

    def test_resume_refuses_when_original_deadline_exhausted_without_explicit_authorization(self) -> None:
        fix = self._build_fixture_env()
        with (
            mock.patch.object(v3, "PARTIAL_RECEIPT_PATH", fix["partial_receipt_path"]),
            mock.patch.object(v3, "verify_authority_and_immutable_artifacts") as spy_auth,
            mock.patch.object(v3, "run_bounded_256_context_winner_generation") as spy_evo,
            mock.patch.object(v3, "build_versioned_v3_strict_train_pools") as spy_build,
            mock.patch.object(v3, "train_matched_430_steps_strict_train") as spy_train,
        ):
            with self.assertRaises(v3.DeadlineExpiredError) as ctx:
                v3.main(["--resume-eval-from-saved-checkpoints"])
            self.assertIn("exhausted", str(ctx.exception))
            self.assertEqual(spy_auth.call_count, 0)
            self.assertEqual(spy_evo.call_count, 0)
            self.assertEqual(spy_build.call_count, 0)
            self.assertEqual(spy_train.call_count, 0)

    def test_resume_fails_closed_when_frozen_baseline_checkpoint_missing_without_explicit_v2a_flag(self) -> None:
        fix = self._build_fixture_env()
        with (
            mock.patch.object(v3, "CHECKPOINTS_DIR", fix["ckpt_dir"]),
            mock.patch.object(v3, "RESULTS_DIR", fix["res_dir"]),
            mock.patch.object(v3, "CORPUS_V3_DIR", fix["v3_dir"]),
            mock.patch.object(v3, "CORPUS_V3_MANIFEST_PATH", fix["manifest_v3_path"]),
            mock.patch.object(v3, "PARTIAL_RECEIPT_PATH", fix["partial_receipt_path"]),
            mock.patch.object(v3, "IMMUTABLE_PHASE7_PARTIAL_ARTIFACTS", fix["expected_partial"]),
            mock.patch.object(v3, "IMMUTABLE_V2A_CHECKPOINT_SHA256", fix["v2a_sha"]),
            mock.patch.object(v3, "verify_authority_and_immutable_artifacts", return_value={"ok": True}),
            mock.patch.object(
                v3,
                "load_all_pools_and_features",
                return_value=([], [], fix["ctx_raw_584"], {}, fix["pools_frozen"]),
            ),
            mock.patch.object(v3, "evaluate_episode_rollout_context") as spy_eval,
            mock.patch.object(v3, "train_matched_430_steps_strict_train") as spy_train,
            mock.patch("torch.save") as spy_torch_save,
        ):
            with self.assertRaises(RuntimeError) as ctx:
                v3.main(
                    [
                        "--resume-eval-from-saved-checkpoints",
                        "--authorize-post-deadline-resume-budget-s",
                        "60",
                    ]
                )
            self.assertIn(
                "Refusing to substitute WINNER_POOL_V2A_TRAIN_ONLY_430 silently as BASELINE_FROZEN_POOL_TRAIN_ONLY_430",
                str(ctx.exception),
            )
            self.assertEqual(spy_eval.call_count, 0)
            self.assertEqual(spy_train.call_count, 0)
            self.assertEqual(spy_torch_save.call_count, 0)

    def test_resume_with_explicit_v2a_comparator_propagates_distinct_v2a_labels_and_never_claims_baseline(self) -> None:
        fix = self._build_fixture_env()
        manifest_sha_before = v3.sha256_file(fix["manifest_v3_path"])
        v3a_sha_before = v3.sha256_file(fix["v3a_npz"])
        v3b_sha_before = v3.sha256_file(fix["v3b_npz"])

        with contextlib.ExitStack() as stack:
            for attr, val in (
                ("CHECKPOINTS_DIR", fix["ckpt_dir"]),
                ("RESULTS_DIR", fix["res_dir"]),
                ("CORPUS_V3_DIR", fix["v3_dir"]),
                ("CORPUS_V3_MANIFEST_PATH", fix["manifest_v3_path"]),
                ("PHASE7_REPORT_PATH", fix["res_dir"] / "report.json"),
                ("TERMINAL_V3_RECEIPT_PATH", fix["res_dir"] / "receipt.json"),
                ("PARTIAL_RECEIPT_PATH", fix["partial_receipt_path"]),
                ("IMMUTABLE_PHASE7_PARTIAL_ARTIFACTS", fix["expected_partial"]),
                ("IMMUTABLE_V2A_CHECKPOINT_SHA256", fix["v2a_sha"]),
            ):
                stack.enter_context(mock.patch.object(v3, attr, val))
            stack.enter_context(
                mock.patch.object(v3, "verify_authority_and_immutable_artifacts", return_value={"ok": True})
            )
            stack.enter_context(
                mock.patch.object(
                    v3,
                    "load_all_pools_and_features",
                    return_value=([], [], fix["ctx_raw_584"], {}, fix["pools_frozen"]),
                )
            )
            stack.enter_context(mock.patch.object(v3, "evaluate_episode_rollout_context", side_effect=_fast_eval_stub))
            stack.enter_context(
                mock.patch.object(v3, "compute_component_bootstrap_summary", side_effect=_fast_bootstrap_stub)
            )
            stack.enter_context(
                mock.patch.object(
                    v3,
                    "publish_phase7_artifacts_and_terminal_manifest_v3",
                    side_effect=lambda **kw: (time.sleep(0.05) or {"local_sha256": "abc", "s3_version_id": "v1"}),
                )
            )
            spy_evo = stack.enter_context(mock.patch.object(v3, "run_bounded_256_context_winner_generation"))
            spy_build = stack.enter_context(mock.patch.object(v3, "build_versioned_v3_strict_train_pools"))
            spy_train = stack.enter_context(mock.patch.object(v3, "train_matched_430_steps_strict_train"))
            spy_savez = stack.enter_context(mock.patch("numpy.savez_compressed"))
            spy_torch_save = stack.enter_context(mock.patch("torch.save"))
            report = v3.main(
                [
                    "--resume-eval-from-saved-checkpoints",
                    "--authorize-post-deadline-resume-budget-s",
                    "60",
                    "--allow-v2a-intervention-comparator",
                ]
            )
            # Zero evolution, zero pool materialization, zero training, zero pool/checkpoint writes
            self.assertEqual(spy_evo.call_count, 0)
            self.assertEqual(spy_build.call_count, 0)
            self.assertEqual(spy_train.call_count, 0)
            self.assertEqual(spy_savez.call_count, 0)
            self.assertEqual(spy_torch_save.call_count, 0)

            # Verify existing v3 artifacts were never rewritten
            self.assertEqual(v3.sha256_file(fix["manifest_v3_path"]), manifest_sha_before)
            self.assertEqual(v3.sha256_file(fix["v3a_npz"]), v3a_sha_before)
            self.assertEqual(v3.sha256_file(fix["v3b_npz"]), v3b_sha_before)

            # Verify comparator labels are strictly WINNER_POOL_V2A_TRAIN_ONLY_430 and NEVER BASELINE_FROZEN_POOL_TRAIN_ONLY_430
            step5 = report["step5_matched_430_step_acceptance_test"]
            self.assertEqual(step5["evaluation_status"], "COMPLETED")
            self.assertEqual(step5["comparator_arm_tag"], "WINNER_POOL_V2A_TRAIN_ONLY_430")
            self.assertFalse(step5["comparator_is_frozen_pool_baseline"])
            self.assertIn("WINNER_POOL_V2A_TRAIN_ONLY_430", step5["training_telemetry"])
            self.assertNotIn("BASELINE_FROZEN_POOL_TRAIN_ONLY_430", step5["training_telemetry"])

            for split_key in ("dl_val_91_contexts_summary", "dl_holdout_93_contexts_summary"):
                split_summary = step5[split_key]
                self.assertIn("WINNER_POOL_V2A_TRAIN_ONLY_430", split_summary["arm_means"])
                self.assertNotIn("BASELINE_FROZEN_POOL_TRAIN_ONLY_430", split_summary["arm_means"])
                self.assertIn(
                    "WINNER_POOL_V3A_Q25_430_minus_WINNER_POOL_V2A_TRAIN_ONLY_430",
                    split_summary["contrasts"],
                )
                self.assertIn(
                    "WINNER_POOL_V3B_MEDIAN_430_minus_WINNER_POOL_V2A_TRAIN_ONLY_430",
                    split_summary["contrasts"],
                )
                for k_contrast in split_summary["contrasts"]:
                    self.assertNotIn("BASELINE_FROZEN_POOL_TRAIN_ONLY_430", k_contrast)

            step6 = report["step6_programmatic_terminal_conclusion"]
            self.assertEqual(step6["comparator_arm_tag"], "WINNER_POOL_V2A_TRAIN_ONLY_430")
            self.assertFalse(step6["comparator_is_frozen_pool_baseline"])
            self.assertIn("v3a_q25_vs_v2a_intervention_comparator_status", step6)
            self.assertIn("v3b_median_vs_v2a_intervention_comparator_status", step6)
            self.assertIn("comparator_causal_interpretation_guard", step6)
            self.assertNotIn("v3a_q25_vs_baseline_status", step6)
            self.assertNotIn("v3b_median_vs_baseline_status", step6)

            # Verify post-publication wall accounting is strictly >= pre-publication wall accounting
            acct = report["resource_and_cost_accounting"]
            self.assertGreaterEqual(
                acct["post_publication_total_wall_seconds"],
                acct["pre_publication_wall_seconds"],
            )
            self.assertEqual(acct["total_wall_seconds"], acct["post_publication_total_wall_seconds"])
            self.assertTrue(acct["within_finite_wall_stop"])

        # Verify tamper detection on materialized pool .npz
        fix["v3a_npz"].write_bytes(b"corrupted_npz_payload")
        with (
            mock.patch.object(v3, "CORPUS_V3_MANIFEST_PATH", fix["manifest_v3_path"]),
            mock.patch.object(v3, "IMMUTABLE_PHASE7_PARTIAL_ARTIFACTS", fix["expected_partial"]),
        ):
            with self.assertRaises(RuntimeError) as ctx_tamper:
                v3.load_verified_immutable_v3_pools_and_manifest(fix["pools_frozen"])
            self.assertIn("mismatch", str(ctx_tamper.exception))

    def test_resume_with_explicit_baseline_430_checkpoint_uses_baseline_comparator_labels(self) -> None:
        fix = self._build_fixture_env()
        baseline_ckpt = fix["ckpt_dir"] / "flywheel_baseline_frozen_pool_train_only_430.pt"
        shutil.copy2(fix["ckpt_dir"] / "flywheel_round_1_full_corpus.pt", baseline_ckpt)
        baseline_sha = v3.sha256_file(baseline_ckpt)

        with contextlib.ExitStack() as stack:
            for attr, val in (
                ("CHECKPOINTS_DIR", fix["ckpt_dir"]),
                ("RESULTS_DIR", fix["res_dir"]),
                ("CORPUS_V3_DIR", fix["v3_dir"]),
                ("CORPUS_V3_MANIFEST_PATH", fix["manifest_v3_path"]),
                ("PHASE7_REPORT_PATH", fix["res_dir"] / "report_base.json"),
                ("TERMINAL_V3_RECEIPT_PATH", fix["res_dir"] / "receipt_base.json"),
                ("PARTIAL_RECEIPT_PATH", fix["partial_receipt_path"]),
                ("IMMUTABLE_PHASE7_PARTIAL_ARTIFACTS", fix["expected_partial"]),
                ("IMMUTABLE_V2A_CHECKPOINT_SHA256", fix["v2a_sha"]),
            ):
                stack.enter_context(mock.patch.object(v3, attr, val))
            stack.enter_context(
                mock.patch.object(v3, "verify_authority_and_immutable_artifacts", return_value={"ok": True})
            )
            stack.enter_context(
                mock.patch.object(
                    v3,
                    "load_all_pools_and_features",
                    return_value=([], [], fix["ctx_raw_584"], {}, fix["pools_frozen"]),
                )
            )
            stack.enter_context(mock.patch.object(v3, "evaluate_episode_rollout_context", side_effect=_fast_eval_stub))
            stack.enter_context(
                mock.patch.object(v3, "compute_component_bootstrap_summary", side_effect=_fast_bootstrap_stub)
            )
            stack.enter_context(
                mock.patch.object(
                    v3,
                    "publish_phase7_artifacts_and_terminal_manifest_v3",
                    return_value={"local_sha256": "abc", "s3_version_id": "v1"},
                )
            )
            spy_train = stack.enter_context(mock.patch.object(v3, "train_matched_430_steps_strict_train"))
            report = v3.main(
                [
                    "--resume-eval-from-saved-checkpoints",
                    "--authorize-post-deadline-resume-budget-s",
                    "60",
                    "--baseline-430-checkpoint",
                    str(baseline_ckpt),
                    "--baseline-430-checkpoint-sha256",
                    baseline_sha,
                ]
            )
            self.assertEqual(spy_train.call_count, 0)
            step5 = report["step5_matched_430_step_acceptance_test"]
            self.assertEqual(step5["comparator_arm_tag"], "BASELINE_FROZEN_POOL_TRAIN_ONLY_430")
            self.assertTrue(step5["comparator_is_frozen_pool_baseline"])
            self.assertIn("BASELINE_FROZEN_POOL_TRAIN_ONLY_430", step5["training_telemetry"])
            step6 = report["step6_programmatic_terminal_conclusion"]
            self.assertTrue(step6["comparator_is_frozen_pool_baseline"])
            self.assertIn("v3a_q25_vs_baseline_status", step6)
            self.assertIn("v3b_median_vs_baseline_status", step6)
            self.assertNotIn("comparator_causal_interpretation_guard", step6)

    def _assert_sigkill_and_worker_exit(self, worker_pid: int, requests: list[dict], states: list[dict]) -> list[dict]:
        worker_requests = [r for r in requests if r["pid"] == worker_pid]
        self.assertEqual({r["api"] for r in worker_requests}, {"killpg", "kill"})
        self.assertTrue(all(r["signal"] == signal.SIGKILL for r in worker_requests))
        state = next(s for s in states if s["pid"] == worker_pid)
        self.assertFalse(state["alive"], "worker still alive when controller returned")
        self.assertEqual(state["exitcode"], -signal.SIGKILL)
        with self.assertRaises(ProcessLookupError):
            os.kill(worker_pid, 0)
        return worker_requests

    def test_worker_deadline_dispatch_and_admission_use_the_remaining_budget(self) -> None:
        # Exercise the real controller, not a copy of its deadline loop. OS/process seams
        # cannot launch work or signal a real PID; select advances only the controlled clock.
        work_budget = 15.30 - 15.0
        for initial_elapsed in (0.0, work_budget):
            with self.subTest(initial_elapsed=initial_elapsed):
                now = [initial_elapsed]
                waits = []
                admitted = []
                requests = []
                fake_pid = 987654321
                process = mock.Mock(pid=fake_pid)
                process.start.side_effect = lambda: admitted.append((0, now[0]))
                context = mock.Mock()
                context.Process.return_value = process
                worker = mock.Mock(side_effect=AssertionError("fake process must not execute work"))

                def select_step(readers, writers, errors, timeout):
                    self.assertEqual(readers, [101])
                    self.assertEqual((writers, errors), ([], []))
                    self.assertGreater(timeout, 0.0)
                    self.assertLessEqual(timeout, work_budget - now[0] + 0.005)
                    waits.append(timeout)
                    self.assertLess(len(waits), 32, "controller did not stop at expiry")
                    now[0] += timeout
                    return [], [], []

                with (
                    mock.patch.object(v3.time, "monotonic", side_effect=lambda: now[0]),
                    mock.patch.object(v3.multiprocessing, "get_context", return_value=context),
                    mock.patch.object(v3.os, "pipe", return_value=(101, 102)),
                    mock.patch.object(v3.os, "close"),
                    mock.patch.object(v3.os, "killpg", side_effect=lambda pid, sig: requests.append(("killpg", pid, sig, now[0]))),
                    mock.patch.object(v3.os, "kill", side_effect=lambda pid, sig: requests.append(("kill", pid, sig, now[0]))),
                    mock.patch.object(v3.select, "select", side_effect=select_step),
                ):
                    controller = v3.DeadlineController(script_t0=0.0, hard_cap_s=15.30, persisted_receipt_path=None)
                    results, expired = controller.run_batch_in_killable_workers(
                        worker, [0, 1], max_workers=1, reserve_s=15.0,
                    )
                self.assertTrue(expired)
                self.assertEqual(results, [])
                worker.assert_not_called()
                if initial_elapsed == 0.0:
                    self.assertEqual(admitted, [(0, 0.0)])
                    context.Process.assert_called_once()
                    self.assertEqual(context.Process.call_args.kwargs["args"][-1], 0)
                    self.assertEqual([(api, pid, sig) for api, pid, sig, _ in requests], [
                        ("killpg", fake_pid, signal.SIGKILL),
                        ("kill", fake_pid, signal.SIGKILL),
                    ])
                    for _, _, _, dispatch_time in requests:
                        self.assertGreaterEqual(dispatch_time, work_budget)
                        self.assertLessEqual(dispatch_time, work_budget + 0.005)
                    process.join.assert_called_once()
                    self.assertGreater(process.join.call_args.kwargs["timeout"], 0.0)
                else:
                    self.assertEqual(admitted, [])
                    context.Process.assert_not_called()
                    self.assertEqual(requests, [])
                    self.assertEqual(waits, [])

    def test_noop_worker_termination_is_rejected_by_exit_evidence(self) -> None:
        # Keep handles to our own children. Never clean up a PID obtained from a file or
        # global process listing, and never leave the intentionally blocked worker behind.
        context = v3.multiprocessing.get_context("fork")
        ready_path = SCRATCH_TEST_ROOT / "negative_worker_ready.txt"
        processes = []
        requests = []
        factory = mock.Mock()

        def create_process(**kwargs):
            process = context.Process(**kwargs)
            processes.append(process)
            return process

        def suppress_signal(api, pid, sig):
            self.assertIn(pid, {p.pid for p in processes})
            self.assertEqual(sig, signal.SIGKILL)
            requests.append({"api": api, "pid": pid, "signal": sig})

        factory.Process.side_effect = create_process
        try:
            with (
                mock.patch.object(v3.multiprocessing, "get_context", return_value=factory),
                mock.patch.object(v3.os, "killpg", side_effect=lambda pid, sig: suppress_signal("killpg", pid, sig)),
                mock.patch.object(v3.os, "kill", side_effect=lambda pid, sig: suppress_signal("kill", pid, sig)),
            ):
                controller = v3.DeadlineController(script_t0=time.monotonic(), hard_cap_s=15.30, persisted_receipt_path=None)
                results, expired = controller.run_batch_in_killable_workers(
                    _deadline_waiting_worker, [{"ready_path": str(ready_path)}],
                    max_workers=1, reserve_s=15.0,
                )
            self.assertTrue(expired)
            self.assertEqual(results, [])
            self.assertEqual(len(processes), 1)
            self.assertTrue(ready_path.exists(), "owned worker did not reach its blocking call")
            self.assertTrue(processes[0].is_alive(), "no-op termination must leave the control alive")
            self.assertEqual({r["api"] for r in requests}, {"killpg", "kill"})
            # Both requests were recorded, but actual exit is absent: the positive test's
            # evidence validator must reject this broken termination implementation.
            with self.assertRaisesRegex(AssertionError, "worker still alive"):
                self._assert_sigkill_and_worker_exit(processes[0].pid, requests, [{
                    "pid": processes[0].pid,
                    "alive": processes[0].is_alive(),
                    "exitcode": processes[0].exitcode,
                }])
            print("PHASE7_NOOP_TERMINATION_REJECTED", json.dumps({
                "worker_pid": processes[0].pid, "alive_at_return": True,
                "signal_requests": requests,
            }, sort_keys=True))
        finally:
            # Signal patches have exited. Cleanup touches only retained child handles.
            # A finite sleeper avoids shared synchronization locks abandoned by SIGKILL.
            try:
                for process in processes:
                    if process.is_alive():
                        process.kill()
            finally:
                for process in processes:
                    if process.pid is not None:
                        process.join(timeout=2.5)
                        self.assertFalse(process.is_alive(), "negative control leaked its worker")
                    process.close()

    def test_isolated_child_process_deadline_terminates_process_and_workers_with_no_late_writes(self) -> None:
        for controlled_reap in (False, True):
            with self.subTest(controlled_reap=controlled_reap):
                self._run_isolated_deadline_case(controlled_reap)

    def _run_isolated_deadline_case(self, controlled_reap: bool) -> None:
        fix = self._build_fixture_env()
        worker_pid_file = SCRATCH_TEST_ROOT / f"worker_pid_{controlled_reap}.txt"
        late_write_file = SCRATCH_TEST_ROOT / f"late_write_after_deadline_{controlled_reap}.txt"
        child_summary_file = SCRATCH_TEST_ROOT / f"child_summary_{controlled_reap}.json"

        child_script = textwrap.dedent(
            f"""
            import contextlib
            import copy
            import json
            import os
            import signal
            import sys
            import time
            from pathlib import Path
            from unittest import mock
            import numpy as np

            sys.path.insert(0, {repr(str(THIS_DIR))})
            import verify_flywheel_wider_train_extension_v3 as v3

            orig_npz = Path({repr(str(fix["v3a_npz"]))})
            d = np.load(orig_npz, allow_pickle=False)
            p0 = {{
                "rec": {{
                    "index": 0,
                    "context_sha256": "a" * 64,
                    "dataset_id": "ds_0",
                    "view_id": "view_0",
                    "family_id": "fam_0",
                    "source_split": "train",
                    "dl_split": "DL-Train",
                    "pool_path": str(orig_npz),
                    "teacher_jtt1_path": str(orig_npz),
                    "n_candidates": int(d["program"].shape[0]),
                }},
                "program": d["program"],
                "proj16": np.zeros((d["program"].shape[0], 16), dtype=np.float64),
                "struct_sig": np.zeros((d["program"].shape[0], 24), dtype=np.float32),
                "fitness": d["fitness"],
                "logratio": d["logratio"],
                "gate": d["gate"],
                "pids": [s.decode("ascii") for s in d["program_sha256"]],
                "prior_cost": d["fitness"].copy(),
                "scale": 1.0,
            }}
            p1 = copy.deepcopy(p0)
            p1["rec"] = dict(p0["rec"], context_sha256="d" * 64, index=1)
            pools = {{"a" * 64: p0, "d" * 64: p1}}
            selected_records = [p0["rec"], p1["rec"]]
            champ_bank = [
                {{"context_sha256": "a" * 64, "dl_split": "DL-Train", "split": "train", "dsl": "v0 | head", "program_sha256": "1" * 64}},
                {{"context_sha256": "d" * 64, "dl_split": "DL-Train", "split": "train", "dsl": "v0 | head", "program_sha256": "2" * 64}},
            ]
            lm48_all = np.zeros((3, 48), dtype=np.float32)

            def _fake_propose(crec, **kwargs):
                return [{{"dsl": "v0 | head", "program_sha256": "9" * 64, "operator": "edit_zero", "parents": []}}]

            def _fake_strat300(teacher_jtt1_gz_path, candidates, max_train_rows, max_workers, timeout_s):
                return [
                    {{
                        "dsl": c["dsl"],
                        "program_sha256": c["program_sha256"],
                        "fitness": -0.95,
                        "logratio": -0.6,
                        "cv_geomean_rmse_ratio": 0.9,
                        "cv_gate_passed": True,
                    }}
                    for c in candidates
                ]

            def _fake_full_jtt1(teacher_jtt1_gz_path, candidates, run_outer_test, max_workers, timeout_s):
                # Distinguish context 0 vs context 1 by inspecting the parent worker stack frame's crec
                import inspect
                frame = inspect.currentframe()
                cid = ""
                while frame is not None:
                    if "crec" in frame.f_locals and isinstance(frame.f_locals["crec"], dict):
                        cid = frame.f_locals["crec"].get("context_sha256", "")
                        break
                    frame = frame.f_back
                if cid == "d" * 64:
                    Path({repr(str(worker_pid_file))}).write_text(str(os.getpid()))
                    # Sleep 4.0s ignoring timeout_s, then attempt a late write after deadline
                    time.sleep(4.0)
                    Path({repr(str(late_write_file))}).write_text("LATE_WRITE_HAPPENED")
                else:
                    time.sleep(0.02)
                return [
                    {{
                        "dsl": c["dsl"],
                        "program_sha256": c["program_sha256"],
                        "fitness": -0.95,
                        "logratio": -0.6,
                        "cv_geomean_rmse_ratio": 0.9,
                        "cv_gate_passed": True,
                    }}
                    for c in candidates
                ]

            signal_requests = []
            real_killpg, real_kill = os.killpg, os.kill
            context = v3.multiprocessing.get_context("fork")
            processes, controlled_reaps = [], []
            factory = mock.Mock()

            def _create_process(**kwargs):
                process = context.Process(**kwargs)
                processes.append(process)
                real_join = process.join

                def _join(timeout=None):
                    pid_file = Path({repr(str(worker_pid_file))})
                    if {controlled_reap!r} and timeout == 0.2 and pid_file.exists() and int(pid_file.read_text()) == process.pid:
                        # Model the full existing reap allowance after SIGKILL. The old
                        # total <0.45s oracle rejects this even with on-time termination.
                        controlled_reaps.append(timeout)
                        time.sleep(timeout)
                        return real_join(timeout=0.0)
                    return real_join(timeout=timeout)

                process.join = _join
                return process

            factory.Process.side_effect = _create_process

            def _cleanup_owned_children():
                for process in processes:
                    if process.pid is not None:
                        if process.is_alive():
                            for send in (real_killpg, real_kill):
                                try:
                                    send(process.pid, signal.SIGKILL)
                                except ProcessLookupError:
                                    pass
                        process.join(timeout=2.5)
                        if process.is_alive():
                            raise RuntimeError("isolated test leaked its owned child")
                    process.close()

            def _record_signal(api, send, pid, sig):
                signal_requests.append({{
                    "api": api, "pid": pid, "signal": sig,
                    "elapsed_s": time.monotonic() - t_start,
                }})
                return send(pid, sig)

            t_start = time.monotonic()
            print("CHILD_IMPORT_COMPLETE", flush=True)
            # With reserve_s=15.0 inside run_bounded_256_context_winner_generation, hard_cap_s=15.30 gives 0.30s budget
            controller = v3.DeadlineController(
                script_t0=t_start,
                hard_cap_s=15.30,
                persisted_receipt_path=None,
            )
            with (
                contextlib.ExitStack() as cleanup,
                mock.patch.object(v3.multiprocessing, "get_context", return_value=factory),
                mock.patch.object(v3.os, "killpg", side_effect=lambda pid, sig: _record_signal("killpg", real_killpg, pid, sig)),
                mock.patch.object(v3.os, "kill", side_effect=lambda pid, sig: _record_signal("kill", real_kill, pid, sig)),
                mock.patch.object(v3, "propose_6_novel_candidates_for_strict_train_context", side_effect=_fake_propose),
                mock.patch.object(v3, "evaluate_gp_batch_rust_strat300", side_effect=_fake_strat300),
                mock.patch.object(v3, "evaluate_gp_batch_rust", side_effect=_fake_full_jtt1),
            ):
                cleanup.callback(_cleanup_owned_children)
                winners, evo_summary = v3.run_bounded_256_context_winner_generation(
                    selected_256_records=selected_records,
                    pools_frozen=pools,
                    champ_bank=champ_bank,
                    lm48_all=lm48_all,
                    script_t0=controller,
                    chunk_size=1,
                )
                # Capture before our cleanup or Python's daemon-child exit finalizer
                # could hide a controller that merely requested a kill but left work alive.
                worker_states = [
                    {{"pid": p.pid, "alive": p.is_alive(), "exitcode": p.exitcode}}
                    for p in processes
                ]
            manifest_doc = json.loads(Path({repr(str(fix["manifest_v3_path"]))}).read_text())
            concl = v3.build_dynamic_evidence_bound_conclusion(
                cohort_spec=manifest_doc["cohort_256_selection_spec"],
                evolution_summary=evo_summary,
                corpus_v3_manifest=manifest_doc,
                exp_results={{
                    "evaluation_status": "UNKNOWN_NOT_COMPLETED_BEFORE_DEADLINE",
                    "deadline_expired_reason": "Hard wall-clock deadline reached",
                }},
            )
            Path({repr(str(child_summary_file))}).write_text(
                json.dumps(
                    {{
                        "elapsed_s": time.monotonic() - t_start,
                        "deadline_s": controller.effective_cap_s - 15.0,
                        "signal_requests": signal_requests,
                        "worker_states_at_return": worker_states,
                        "controlled_reaps_s": controlled_reaps,
                        "winners_len": len(winners),
                        "evo_summary": evo_summary,
                        "conclusion": concl,
                    }},
                    indent=2,
                )
            )
            """
        )

        proc = subprocess.Popen(
            [sys.executable, "-u", "-c", child_script],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        assert proc.stdout is not None
        ready_line = proc.stdout.readline().strip()
        self.assertEqual(ready_line, "CHILD_IMPORT_COMPLETE")

        # Start post-import execution + interpreter-exit clock
        t0 = time.monotonic()
        try:
            stdout_rest, stderr = proc.communicate(timeout=2.5)
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.communicate(timeout=2.5)
        child_wall_s = time.monotonic() - t0

        self.assertEqual(proc.returncode, 0, msg=f"Child failed:\nSTDOUT:\n{stdout_rest}\nSTDERR:\n{stderr}")
        # Child OS process must execute, SIGKILL the 4.0s worker, and exit the OS process (including libtorch C++ teardown) well before 4.0s
        self.assertLess(child_wall_s, 2.50)

        # Verify worker OS process recorded its PID and is now dead (SIGKILLed)
        self.assertTrue(worker_pid_file.exists())
        worker_pid = int(worker_pid_file.read_text().strip())
        summary = json.loads(child_summary_file.read_text())
        requests = self._assert_sigkill_and_worker_exit(
            worker_pid, summary["signal_requests"], summary["worker_states_at_return"],
        )
        for request in requests:
            self.assertGreaterEqual(request["elapsed_s"], summary["deadline_s"])
            self.assertLess(request["elapsed_s"], summary["deadline_s"] + 0.15)

        # Wait an extra 0.4s to prove no background worker can perform a late write
        time.sleep(0.4)
        self.assertFalse(late_write_file.exists())

        # Retain the original 0.15s scheduling/reporting slack, separately from the
        # existing 0.20s join allowance for the one blocked worker. No work budget changes.
        self.assertLess(summary["elapsed_s"], summary["deadline_s"] + 0.2 + 0.15)
        self.assertEqual(summary["controlled_reaps_s"], [0.2] if controlled_reap else [])
        if controlled_reap:
            self.assertGreaterEqual(summary["elapsed_s"], summary["deadline_s"] + 0.2)
        self.assertTrue(summary["evo_summary"]["stopped_early_by_wall_cap"])
        self.assertEqual(summary["evo_summary"]["contexts_completed"], 1)
        self.assertEqual(summary["winners_len"], 1)
        self.assertEqual(
            summary["conclusion"]["matched_430_step_evaluation_status"],
            "UNKNOWN_NOT_COMPLETED_BEFORE_DEADLINE",
        )
        print("PHASE7_DEADLINE_EVIDENCE", json.dumps({
            "controlled_reap": controlled_reap, "worker_pid": worker_pid,
            "deadline_s": summary["deadline_s"], "signal_requests": requests,
            "worker_states_at_return": summary["worker_states_at_return"],
            "elapsed_s": summary["elapsed_s"], "child_wall_s": child_wall_s,
            "contexts_completed": summary["evo_summary"]["contexts_completed"],
            "winners_len": summary["winners_len"],
            "evaluation_status": summary["conclusion"]["matched_430_step_evaluation_status"],
        }, sort_keys=True))

    def test_blocked_rollout_call_preempted_by_itimer_at_hard_deadline(self) -> None:
        fix = self._build_fixture_env()

        def _blocking_eval_stub(*args, **kwargs):
            time.sleep(2.5)
            return {}

        t_start = time.monotonic()
        # With reserve_s=10.0 inside run_matched_v3_acceptance_test, hard_cap_s=10.18 gives 0.18s budget
        controller = v3.DeadlineController(
            script_t0=t_start,
            hard_cap_s=10.18,
            persisted_receipt_path=None,
        )
        with (
            mock.patch.object(v3, "CHECKPOINTS_DIR", fix["ckpt_dir"]),
            mock.patch.object(v3, "IMMUTABLE_PHASE7_PARTIAL_ARTIFACTS", fix["expected_partial"]),
            mock.patch.object(v3, "IMMUTABLE_V2A_CHECKPOINT_SHA256", fix["v2a_sha"]),
            mock.patch.object(v3, "evaluate_episode_rollout_context", side_effect=_blocking_eval_stub),
        ):
            res = v3.run_matched_v3_acceptance_test(
                pools_frozen=fix["pools_frozen"],
                pools_v3a=fix["pools_frozen"],
                pools_v3b=fix["pools_frozen"],
                ctx_raw_584=fix["ctx_raw_584"],
                device=torch.device("cpu"),
                script_t0=controller,
                reuse_saved_checkpoints=True,
                allow_v2a_intervention_comparator=True,
            )
        elapsed = time.monotonic() - t_start
        self.assertLess(elapsed, 0.45)
        self.assertEqual(res["evaluation_status"], "UNKNOWN_NOT_COMPLETED_BEFORE_DEADLINE")
        self.assertEqual(res["comparator_arm_tag"], "WINNER_POOL_V2A_TRAIN_ONLY_430")
        self.assertFalse(res["comparator_is_frozen_pool_baseline"])

    def test_itimer_signal_delivery_boundary_vs_forked_worker_sigkill_under_masked_sigalrm(self) -> None:
        # Part A: Run in a single-threaded Python child process (before OpenMP/libtorch spawns extra
        # threads) to prove that when SIGALRM delivery is blocked during an uninterruptible section
        # (pthread_sigmask(SIG_BLOCK, {SIGALRM})), ITIMER_REAL cannot raise mid-section and raises
        # DeadlineExpiredError only once signal delivery resumes at the Python bytecode boundary.
        part_a_script = textwrap.dedent(
            """
            import json
            import signal
            import time

            class _MiniDeadlineError(RuntimeError):
                pass

            def _handler(_s, _f):
                raise _MiniDeadlineError("ITIMER_REAL delivered at Python signal check boundary")

            signal.signal(signal.SIGALRM, _handler)
            signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGALRM})
            signal.setitimer(signal.ITIMER_REAL, 0.04)
            t0 = time.monotonic()
            # Sleep 0.14s while SIGALRM is masked (uninterruptible section completes without raising)
            time.sleep(0.14)
            uninterruptible_completed = True
            elapsed_before_unblock = time.monotonic() - t0
            raised_on_unblock = False
            try:
                signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGALRM})
                time.sleep(0.01)
            except _MiniDeadlineError:
                raised_on_unblock = True
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0.0)
            print(json.dumps({
                "uninterruptible_completed": uninterruptible_completed,
                "elapsed_before_unblock": elapsed_before_unblock,
                "raised_on_unblock": raised_on_unblock,
            }))
            """
        )
        proc_a = subprocess.run(
            [sys.executable, "-B", "-u", "-c", part_a_script],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        res_a = json.loads(proc_a.stdout.strip())
        self.assertTrue(res_a["uninterruptible_completed"])
        self.assertGreaterEqual(res_a["elapsed_before_unblock"], 0.12)
        self.assertTrue(res_a["raised_on_unblock"])

        # Part B: Forked OS worker process in run_batch_in_killable_workers() is terminated by
        # kernel SIGKILL in <0.55s (vs 3.0s worker sleep) even when the worker blocks SIGALRM and ignores SIGTERM.
        worker_pid_path = SCRATCH_TEST_ROOT / "masked_worker_pid.txt"
        late_marker_path = SCRATCH_TEST_ROOT / "masked_worker_late.txt"

        t1 = time.monotonic()
        ctrl_b = v3.DeadlineController(script_t0=t1, hard_cap_s=1.22, persisted_receipt_path=None)
        completed, stopped_early = ctrl_b.run_batch_in_killable_workers(
            worker_fn=_masked_sigalrm_worker,
            batch_items=[
                {"idx": 0, "worker_pid_path": str(worker_pid_path), "late_marker_path": str(late_marker_path)},
                {"idx": 1, "worker_pid_path": str(worker_pid_path), "late_marker_path": str(late_marker_path)},
            ],
            max_workers=1,
            stage_name="test_sigkill_vs_masked_sigalrm",
            reserve_s=1.0,
        )
        elapsed_b = time.monotonic() - t1
        self.assertTrue(stopped_early)
        self.assertEqual(completed, [100])
        self.assertLess(elapsed_b, 0.55)
        self.assertTrue(worker_pid_path.exists())
        masked_pid = int(worker_pid_path.read_text().strip())
        with self.assertRaises(ProcessLookupError):
            os.kill(masked_pid, 0)
        self.assertFalse(late_marker_path.exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
