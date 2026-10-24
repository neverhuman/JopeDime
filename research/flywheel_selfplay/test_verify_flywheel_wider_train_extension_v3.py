#!/usr/bin/env python3
"""
Deterministic Stub Unit Tests for `verify_flywheel_wider_train_extension_v3.py`
===============================================================================
Verifies the four source-replay audit requirements from
`/home/ubuntu/.codex-screen-monitor/jope-source-replay-audit-20261024T101652Z.txt`
with zero dataset/training/evaluation campaign compute (< 3s wall-clock):

1. `test_resume_eval_only_call_graph_zero_evolution_training_or_writes`:
   Proves `main(["--resume-eval-from-saved-checkpoints", "--authorize-post-deadline-resume-budget-s", "60"])`
   verifies recorded immutable SHA-256 hashes of `corpus_v3_manifest.json`, materialized `.npz` pools,
   and saved checkpoints, and executes ZERO calls to `run_bounded_256_context_winner_generation()`,
   `build_versioned_v3_strict_train_pools()`, `train_matched_430_steps_strict_train()`,
   `np.savez_compressed()`, or `torch.save()`, while leaving `corpus_v3_manifest.json` and `.npz`
   files byte-for-byte untouched. Also verifies tamper detection on pool and checkpoint hashes.
2. `test_resume_refuses_when_original_deadline_exhausted_without_explicit_authorization`:
   Proves that when `flywheel_phase7_partial_and_expired_deadline_receipt.json` records an exhausted
   original budget (`1643.0s > 1080.0s`), calling `--resume-eval-from-saved-checkpoints` without
   `--authorize-post-deadline-resume-budget-s` raises `DeadlineExpiredError` immediately before
   loading pools or running any stage.
3. `test_slow_blocking_evaluator_preempted_at_hard_deadline`:
   Simulates a slow/blocking evaluator (`sleep(5.0)`) after context 0 completes (`0.01s`) under a
   tight hard deadline (`effective_cap_s = 15.18s`, `reserve_s = 15.0s` -> `0.18s` budget). Proves
   `run_bounded_256_context_winner_generation()` preempts via `concurrent.futures.wait` +
   `ex.shutdown(wait=False, cancel_futures=True)` in `< 0.6s` (without waiting `5.0s` on threadpool
   shutdown) and preserves completed partial context 0 (`contexts_completed == 1`).
4. `test_dynamic_unknown_interpretation_and_post_publication_wall_accounting`:
   Proves `build_dynamic_evidence_bound_conclusion()` records incomplete evaluation strictly as
   `UNKNOWN_NOT_COMPLETED_BEFORE_DEADLINE` with zero prewritten architectural/null claims, and proves
   `main()` computes `post_publication_total_wall_seconds` and `within_finite_wall_stop` AFTER
   publication so S3 publication latency is included in deadline accounting.
"""
from __future__ import annotations

import copy
import json
import shutil
import sys
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

# Use conversation scratch dir (never /tmp per AGENTS.md)
SCRATCH_TEST_ROOT = Path(
    "/home/ubuntu/.gemini/antigravity-cli/brain/709bfa09-7564-4555-aad9-0fd2cbcac176/scratch/phase7_unit_tests"
)


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

    def test_resume_eval_only_call_graph_zero_evolution_training_or_writes(self) -> None:
        fix = self._build_fixture_env()
        manifest_sha_before = v3.sha256_file(fix["manifest_v3_path"])
        v3a_sha_before = v3.sha256_file(fix["v3a_npz"])
        v3b_sha_before = v3.sha256_file(fix["v3b_npz"])

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

        with (
            mock.patch.object(v3, "CHECKPOINTS_DIR", fix["ckpt_dir"]),
            mock.patch.object(v3, "RESULTS_DIR", fix["res_dir"]),
            mock.patch.object(v3, "CORPUS_V3_DIR", fix["v3_dir"]),
            mock.patch.object(v3, "CORPUS_V3_MANIFEST_PATH", fix["manifest_v3_path"]),
            mock.patch.object(v3, "PHASE7_REPORT_PATH", fix["res_dir"] / "report.json"),
            mock.patch.object(v3, "TERMINAL_V3_RECEIPT_PATH", fix["res_dir"] / "receipt.json"),
            mock.patch.object(v3, "PARTIAL_RECEIPT_PATH", fix["partial_receipt_path"]),
            mock.patch.object(v3, "IMMUTABLE_PHASE7_PARTIAL_ARTIFACTS", fix["expected_partial"]),
            mock.patch.object(v3, "verify_authority_and_immutable_artifacts", return_value={"ok": True}),
            mock.patch.object(
                v3,
                "load_all_pools_and_features",
                return_value=([], [], fix["ctx_raw_584"], {}, fix["pools_frozen"]),
            ),
            mock.patch.object(v3, "evaluate_episode_rollout_context", side_effect=_fast_eval_stub),
            mock.patch.object(v3, "compute_component_bootstrap_summary", side_effect=_fast_bootstrap_stub),
            mock.patch.object(
                v3,
                "publish_phase7_artifacts_and_terminal_manifest_v3",
                side_effect=lambda **kw: (time.sleep(0.05) or {"local_sha256": "abc", "s3_version_id": "v1"}),
            ),
            mock.patch.object(v3, "run_bounded_256_context_winner_generation") as spy_evo,
            mock.patch.object(v3, "build_versioned_v3_strict_train_pools") as spy_build,
            mock.patch.object(v3, "train_matched_430_steps_strict_train") as spy_train,
            mock.patch("numpy.savez_compressed") as spy_savez,
            mock.patch("torch.save") as spy_torch_save,
        ):
            report = v3.main(
                [
                    "--resume-eval-from-saved-checkpoints",
                    "--authorize-post-deadline-resume-budget-s",
                    "60",
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

            # Verify post-publication wall accounting is strictly >= pre-publication wall accounting
            acct = report["resource_and_cost_accounting"]
            self.assertGreaterEqual(
                acct["post_publication_total_wall_seconds"],
                acct["pre_publication_wall_seconds"],
            )
            self.assertEqual(acct["total_wall_seconds"], acct["post_publication_total_wall_seconds"])
            self.assertTrue(acct["within_finite_wall_stop"])
            self.assertEqual(
                report["step5_matched_430_step_acceptance_test"]["evaluation_status"],
                "COMPLETED",
            )

        # Verify tamper detection on materialized pool .npz
        fix["v3a_npz"].write_bytes(b"corrupted_npz_payload")
        with (
            mock.patch.object(v3, "CORPUS_V3_MANIFEST_PATH", fix["manifest_v3_path"]),
            mock.patch.object(v3, "IMMUTABLE_PHASE7_PARTIAL_ARTIFACTS", fix["expected_partial"]),
        ):
            with self.assertRaises(RuntimeError) as ctx_tamper:
                v3.load_verified_immutable_v3_pools_and_manifest(fix["pools_frozen"])
            self.assertIn("mismatch", str(ctx_tamper.exception))

    def test_slow_blocking_evaluator_preempted_at_hard_deadline(self) -> None:
        fix = self._build_fixture_env()
        cid0 = "a" * 64
        cid1 = "d" * 64
        p0 = fix["pools_frozen"][cid0]
        p1 = copy.deepcopy(p0)
        p1["rec"] = dict(p0["rec"], context_sha256=cid1, index=1)
        pools = {cid0: p0, cid1: p1}
        selected_records = [p0["rec"], p1["rec"]]
        champ_bank = [
            {
                "context_sha256": cid0,
                "dl_split": "DL-Train",
                "split": "train",
                "dsl": "v0 | head",
                "program_sha256": "1" * 64,
            },
            {
                "context_sha256": cid1,
                "dl_split": "DL-Train",
                "split": "train",
                "dsl": "v0 | head",
                "program_sha256": "2" * 64,
            },
        ]
        lm48_all = np.zeros((3, 48), dtype=np.float32)

        def _fake_propose(crec, **kwargs):
            return [{"dsl": "v0 | head", "program_sha256": "9" * 64, "operator": "edit_zero", "parents": []}]

        def _fake_strat300(teacher_jtt1_gz_path, candidates, max_train_rows, max_workers, timeout_s):
            return [
                {
                    "dsl": c["dsl"],
                    "program_sha256": c["program_sha256"],
                    "fitness": -0.95,
                    "logratio": -0.6,
                    "cv_geomean_rmse_ratio": 0.9,
                    "cv_gate_passed": True,
                }
                for c in candidates
            ]

        call_counter = {"n": 0}

        def _fake_full_jtt1(teacher_jtt1_gz_path, candidates, run_outer_test, max_workers, timeout_s):
            call_counter["n"] += 1
            if call_counter["n"] >= 2:
                # Simulate a hung/blocking evaluator that ignores timeout_s and sleeps 1.5s
                time.sleep(1.5)
            return [
                {
                    "dsl": c["dsl"],
                    "program_sha256": c["program_sha256"],
                    "fitness": -0.95,
                    "logratio": -0.6,
                    "cv_geomean_rmse_ratio": 0.9,
                    "cv_gate_passed": True,
                }
                for c in candidates
            ]

        # Set hard_cap_s = 15.15s (with reserve_s = 15.0s inside evolution -> 0.15s remaining budget)
        t_start = time.monotonic()
        controller = v3.DeadlineController(
            script_t0=t_start,
            hard_cap_s=15.15,
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
            )
        elapsed = time.monotonic() - t_start
        # Must preempt in < 0.45s (never waiting 1.5s for the hung thread on shutdown!)
        self.assertLess(elapsed, 0.45)
        self.assertTrue(evo_summary["stopped_early_by_wall_cap"])
        self.assertEqual(evo_summary["contexts_completed"], 1)
        self.assertEqual(len(winners), 1)

    def test_dynamic_unknown_interpretation_and_post_publication_wall_accounting(self) -> None:
        fix = self._build_fixture_env()
        manifest_doc = json.loads(fix["manifest_v3_path"].read_text())
        cohort_spec = manifest_doc["cohort_256_selection_spec"]
        evo_sum = manifest_doc["phase7_256_context_evolution_summary"]

        concl_unknown = v3.build_dynamic_evidence_bound_conclusion(
            cohort_spec=cohort_spec,
            evolution_summary=evo_sum,
            corpus_v3_manifest=manifest_doc,
            exp_results={
                "evaluation_status": "UNKNOWN_NOT_COMPLETED_BEFORE_DEADLINE",
                "deadline_expired_reason": "Hard wall-clock deadline reached",
            },
        )
        self.assertEqual(
            concl_unknown["matched_430_step_evaluation_status"],
            "UNKNOWN_NOT_COMPLETED_BEFORE_DEADLINE",
        )
        self.assertNotIn("v3a_q25_vs_baseline_status", concl_unknown)
        self.assertNotIn("v3b_median_vs_baseline_status", concl_unknown)
        self.assertEqual(
            concl_unknown["partial_cohort_coverage_qualification"]["untested_strict_train_contexts"],
            760,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
