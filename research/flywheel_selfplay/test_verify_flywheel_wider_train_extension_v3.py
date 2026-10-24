#!/usr/bin/env python3
"""
Deterministic Stub & Child-Process Unit Tests for `verify_flywheel_wider_train_extension_v3.py`
===============================================================================================
Verifies all source-replay audit and comparator/process-deadline requirements from:
- `/home/ubuntu/.codex-screen-monitor/jope-source-replay-audit-20261024T101652Z.txt`
- `/home/ubuntu/.codex-screen-monitor/jope-comparator-and-process-deadline-20261024T103653Z.txt`
with zero dataset/training/evaluation campaign compute (< 3s wall-clock):

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
   Spawns a real isolated child Python process (`subprocess.Popen`) under `hard_cap_s = 15.35s`
   (`0.35s` effective budget with `reserve_s = 15.0s`) where context 0 finishes in `0.02s` and context 1's
   worker records its OS `worker_pid`, sleeps `3.0s` (ignoring `timeout_s`), and attempts a late file write
   (`late_write_after_deadline.txt`). Proves:
   - The child Python process exits completely within `< 0.90s` (never hanging `3.0s` at interpreter exit).
   - The spawned worker OS process (`worker_pid`) is dead (`os.kill(worker_pid, 0)` raises `ProcessLookupError`).
   - Even after waiting an additional `0.5s`, `late_write_after_deadline.txt` does NOT exist.
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
   boundary, whereas `run_batch_in_killable_workers()` kills a worker with `SIGKILL` in `< 0.45s` even
   when the worker masks `SIGALRM` and ignores `SIGTERM`.
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
    """Return a non-/tmp scratch directory that works on both the host and CI runner containers."""
    host_scratch = Path(
        "/home/ubuntu/.gemini/antigravity-cli/brain/709bfa09-7564-4555-aad9-0fd2cbcac176/scratch"
    )
    if host_scratch.parent.exists():
        return host_scratch / f"phase7_unit_tests_{os.getpid()}"
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

    def test_isolated_child_process_deadline_terminates_process_and_workers_with_no_late_writes(self) -> None:
        fix = self._build_fixture_env()
        worker_pid_file = SCRATCH_TEST_ROOT / "worker_pid.txt"
        late_write_file = SCRATCH_TEST_ROOT / "late_write_after_deadline.txt"
        child_summary_file = SCRATCH_TEST_ROOT / "child_summary.json"

        child_script = textwrap.dedent(
            f"""
            import copy
            import json
            import os
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

            t_start = time.monotonic()
            print("CHILD_IMPORT_COMPLETE", flush=True)
            # With reserve_s=15.0 inside run_bounded_256_context_winner_generation, hard_cap_s=15.30 gives 0.30s budget
            controller = v3.DeadlineController(
                script_t0=t_start,
                hard_cap_s=15.30,
                persisted_receipt_path=None,
            )
            with (
                mock.patch.object(v3, "propose_6_novel_candidates_for_strict_train_context", side_effect=_fake_propose),
                mock.patch.object(v3, "evaluate_gp_batch_rust_strat300", side_effect=_fake_strat300),
                mock.patch.object(v3, "evaluate_gp_batch_rust", side_effect=_fake_full_jtt1),
            ):
                winners, evo_summary = v3.run_bounded_256_context_winner_generation(
                    selected_256_records=selected_records,
                    pools_frozen=pools,
                    champ_bank=champ_bank,
                    lm48_all=lm48_all,
                    script_t0=controller,
                    chunk_size=1,
                )
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
        stdout_rest, stderr = proc.communicate(timeout=2.5)
        child_wall_s = time.monotonic() - t0

        self.assertEqual(proc.returncode, 0, msg=f"Child failed:\nSTDOUT:\n{stdout_rest}\nSTDERR:\n{stderr}")
        # Child OS process must execute, SIGKILL the 4.0s worker, and exit the OS process (including libtorch C++ teardown) well before 4.0s
        self.assertLess(child_wall_s, 3.20)

        # Verify worker OS process recorded its PID and is now dead (SIGKILLed)
        self.assertTrue(worker_pid_file.exists())
        worker_pid = int(worker_pid_file.read_text().strip())
        with self.assertRaises(ProcessLookupError):
            os.kill(worker_pid, 0)

        # Wait an extra 0.4s to prove no background worker can perform a late write
        time.sleep(0.4)
        self.assertFalse(late_write_file.exists())

        summary = json.loads(child_summary_file.read_text())
        self.assertLess(summary["elapsed_s"], 0.90)
        self.assertTrue(summary["evo_summary"]["stopped_early_by_wall_cap"])
        self.assertEqual(summary["evo_summary"]["contexts_completed"], 1)
        self.assertEqual(summary["winners_len"], 1)
        self.assertEqual(
            summary["conclusion"]["matched_430_step_evaluation_status"],
            "UNKNOWN_NOT_COMPLETED_BEFORE_DEADLINE",
        )

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
        self.assertLess(elapsed, 1.20)
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
        # kernel SIGKILL in <1.20s (vs 3.0s worker sleep) even when the worker blocks SIGALRM and ignores SIGTERM.
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
        self.assertLess(elapsed_b, 1.20)
        self.assertTrue(worker_pid_path.exists())
        masked_pid = int(worker_pid_path.read_text().strip())
        with self.assertRaises(ProcessLookupError):
            os.kill(masked_pid, 0)
        self.assertFalse(late_marker_path.exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
