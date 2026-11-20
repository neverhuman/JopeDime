#!/usr/bin/env python3
"""Offline deterministic unit and contract test suite for H473."""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

torch.set_num_threads(2)

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from h256_token_layout_audit import (  # noqa: E402
    H256_TOKEN_LAYOUT_SCHEMA,
    build_h256_token_layout_spec,
    compute_token_slice_telemetry,
    derange_across_families,
    verify_contiguous_token_layout,
)
from prog412_collision_audit import (  # noqa: E402
    audit_prog412_collisions,
    run_i10_zero_shot_archetype_cv,
    select_diverse_structural_anchors,
)
from run_h473_study import (  # noqa: E402
    DOPE_FUSED_COMPOSITE_SHA256,
    compute_i9_z_ctx,
    filter_masked_presence_v1,
    paired_contrast,
)
from two_tower_retrieval import (  # noqa: E402
    TwoTowerRetriever,
    run_part_b_two_tower_study,
)


def _make_synthetic_study_fixture(n_ctx: int = 18, m_cands: int = 18) -> dict[str, object]:
    rng = np.random.default_rng(20261120)
    cids = [f"ctx_{i:03d}" for i in range(n_ctx)]
    families = [f"fam_{i // 3:02d}" for i in range(n_ctx)]
    fold_a = np.array([(i // 3) % 3 for i in range(n_ctx)], dtype=np.int64)

    # Construct 36 shared candidate templates with two distinct regimes (0..17 vs 18..35)
    base_pids = [f"pid_{j:03d}" for j in range(m_cands)]
    base_prog = rng.standard_normal((m_cands, 412)).astype(np.float32) * 0.3
    base_sem = np.zeros((m_cands, 24), dtype=np.float32)
    for j in range(m_cands):
        if j < m_cands // 2:
            base_sem[j, 1] = 1.5  # histgbm head
            base_sem[j, 7] = 1.1  # nonlinear regime
            base_prog[j, 0:16] += 1.5
        else:
            base_sem[j, 4] = 1.5  # ridge_gcv head
            base_sem[j, 9] = 1.1  # linear regime
            base_prog[j, 0:16] -= 1.5
        base_sem[j, 10:] = np.abs(rng.standard_normal(14)).astype(np.float32) * 0.2
        base_sem[j] /= float(np.linalg.norm(base_sem[j]))

    x_up = rng.standard_normal((n_ctx, 859)).astype(np.float32) * 0.2
    x_dp = rng.standard_normal((n_ctx, 663)).astype(np.float32) * 0.2
    x_ds = rng.standard_normal((n_ctx, 290)).astype(np.float32) * 0.2
    x_lm = rng.standard_normal((n_ctx, 48)).astype(np.float32) * 0.2

    pools: dict[str, dict[str, object]] = {}
    champs: dict[str, str] = {}
    for i, cid in enumerate(cids):
        regime_tree = (i // 3) % 2 == 0
        sign = 1.0 if regime_tree else -1.0
        x_up[i, :24] += 2.5 * sign
        x_dp[i, :12] += 1.0 * sign
        x_lm[i, :8] += 1.2 * sign
        u_clean = np.array(
            [(1.2 if (j < m_cands // 2) == regime_tree else -0.4) - 0.02 * (j % (m_cands // 2)) for j in range(m_cands)],
            dtype=np.float64,
        )
        gate = np.ones(m_cands, dtype=bool)
        gate[-1] = False
        fit = -u_clean
        pools[cid] = {
            "pids": list(base_pids),
            "program": base_prog.copy(),
            "struct_sig": base_sem.copy(),
            "fitness": fit,
            "gate": gate,
        }
        best_j = int(np.argmax(np.where(gate, -fit, -2.0)))
        champs[cid] = base_pids[best_j]

    repr_tr = {
        "lm48_z": x_lm,
        "R_desc": x_ds,
        "R_dope": x_dp,
        "DopeUltraPacket859": x_up,
        "i10_ultra_packet_859": x_up,
        "i10_dope_663": x_dp,
        "i10_desc_290": x_ds,
        "i10_none_0d": np.zeros((n_ctx, 0), dtype=np.float32),
    }
    return {"cids": cids, "families": families, "folds": [fold_a], "pools": pools, "champs": champs, "repr": repr_tr}


class TestH473Study(unittest.TestCase):
    def test_dv01_token_layout_contiguity_and_slice_bug_telemetry(self) -> None:
        spec = build_h256_token_layout_spec()
        self.assertEqual(spec["schema"], H256_TOKEN_LAYOUT_SCHEMA)
        self.assertEqual(spec["total_tokens"], 92)
        self.assertEqual(spec["token_dim"], 32)
        self.assertEqual(spec["scale_side_features"]["width"], 8)
        self.assertFalse(spec["scale_side_features"]["is_sequence_token"])
        verify_contiguous_token_layout(spec)

        # Gap and overlap negative checks
        broken_gap = copy.deepcopy(spec)
        broken_gap["blocks"]["task128"] = [9, 12]
        with self.assertRaises(ValueError):
            verify_contiguous_token_layout(broken_gap)

        broken_end = copy.deepcopy(spec)
        broken_end["blocks"]["columns64"] = [28, 93]
        with self.assertRaises(ValueError):
            verify_contiguous_token_layout(broken_end)

        # Synthetic 92x32 tensor with exact known L2 norms and 8 active columns out of 64
        n_ctx = 6
        unit_vec = np.zeros(32, dtype=np.float32)
        unit_vec[0] = 1.0
        tok = np.zeros((n_ctx, 92, 32), dtype=np.float32)
        tok[:, 0:8, :] = 2.0 * unit_vec       # desc256 L2 = 2.0
        tok[:, 8:20, :] = 1.5 * unit_vec      # task+dist+heads+colgates L2 = 1.5
        tok[:, 20:28, :] = 5.0 * unit_vec     # hidden256 L2 = 5.0
        cmask = np.zeros((n_ctx, 64), dtype=bool)
        cmask[:, :8] = True                   # only columns 0..7 active (token indices 28..35)
        tok[:, 28:36, :] = 3.0 * unit_vec     # active columns64 L2 = 3.0; 36..91 remain 0.0
        side8 = np.zeros((n_ctx, 8), dtype=np.float32)
        side8[:, 0] = 4.0                     # scale_side_8 L2 = 4.0

        telem = compute_token_slice_telemetry(tok, cmask, side8)
        auth = telem["authoritative_92tok_l2"]
        bug = telem["historical_buggy_93tok_l2"]
        self.assertAlmostEqual(auth["desc256"], 2.0, places=5)
        self.assertAlmostEqual(auth["hidden256"], 5.0, places=5)
        self.assertAlmostEqual(auth["columns64_active_masked"], 3.0, places=5)
        self.assertAlmostEqual(auth["scale_side_features_8d_l2"], 4.0, places=5)
        self.assertEqual(telem["inactive_column_token_max_abs"], 0.0)
        # Historical buggy [45:77] and [77:93] slice inactive column slots -> 0.0!
        self.assertEqual(bug["hidden256_45_77_mean_l2"], 0.0)
        self.assertEqual(bug["col_lm_77_93_clipped_mean_l2"], 0.0)
        self.assertEqual(bug["actual_77_93_slice_length"], 15)
        self.assertEqual(bug["silent_truncation_tokens_lost"], 1)

    def test_dv03_prog412_collision_and_storage_order_invariant_anchors(self) -> None:
        v_a = np.ones(412, dtype=np.float32) * 0.5
        v_a_negzero = v_a.copy()
        v_a_negzero[0] = -0.0
        v_a[0] = +0.0
        v_near = v_a.copy()
        v_near[1] += 1e-3  # cosine > 0.9999 with v_a
        v_far = -v_a

        sem = np.zeros((4, 24), dtype=np.float32)
        sem[0, 1] = 1.0
        sem[1, 1] = 1.0
        sem[2, 4] = 1.0
        sem[3, 2] = 1.0

        pools = {
            "c0": {
                "pids": ["sha_1", "sha_2", "sha_3", "sha_4"],
                "program": np.stack([v_a, v_a_negzero, v_near, v_far], axis=0),
                "struct_sig": sem,
                "fitness": np.array([-0.80, -0.50, -0.78, -0.10], dtype=np.float64),
                "gate": np.array([True, False, True, True], dtype=bool),
            }
        }
        champs = [
            {"dl_split": "DL-Train", "program_sha256": "sha_1", "dsl": "(pipe X (rff_sin seed=11) (ridge_gcv))"},
            {"dl_split": "DL-Train", "program_sha256": "sha_2", "dsl": "(pipe X (rff_sin seed=99) (ridge_gcv))"},
        ]
        res = audit_prog412_collisions(["c0"], pools, champions_records=champs)
        self.assertEqual(res["n_distinct_program_sha256"], 4)
        self.assertEqual(res["global_exact_prog412"]["distinct_vectors"], 3)
        self.assertEqual(res["global_exact_prog412"]["colliding_programs"], 2)
        self.assertEqual(res["within_pool_exact_prog412_disagreement"]["pair_count"], 1)
        self.assertEqual(res["within_pool_exact_prog412_disagreement"]["gate_disagreement_rate"], 1.0)
        self.assertGreaterEqual(res["within_pool_near_cos_gt_0p9999_disagreement"]["pair_count"], 2)
        self.assertEqual(res["dl_train_champions_bank_audit"]["exact_pairs_with_seeded_random_ops"], 1)

        # Permutation invariance of select_diverse_structural_anchors
        rng = np.random.default_rng(42)
        m = 20
        pids_orig = [f"hash_{i:02d}" for i in range(m)]
        sem_orig = np.abs(rng.standard_normal((m, 24))).astype(np.float32)
        sem_orig /= np.linalg.norm(sem_orig, axis=1, keepdims=True)
        idx_1 = select_diverse_structural_anchors(sem_orig, pids_orig, k=4)
        chosen_1 = [pids_orig[i] for i in idx_1.tolist()]

        perm = rng.permutation(m)
        pids_perm = [pids_orig[i] for i in perm.tolist()]
        sem_perm = sem_orig[perm]
        idx_2 = select_diverse_structural_anchors(sem_perm, pids_perm, k=4)
        chosen_2 = [pids_perm[i] for i in idx_2.tolist()]
        self.assertEqual(chosen_1, chosen_2)
        self.assertEqual(len(set(chosen_1)), 4)

    def test_dv02_masked_presence_and_family_derangement(self) -> None:
        idx_0, rf_0, rg_0 = filter_masked_presence_v1([0, 1, 2, 3], [100.0] * 4, [0.0] * 4, [False] * 4)
        self.assertEqual(len(idx_0), 0)
        self.assertEqual(len(rf_0), 0)
        self.assertEqual(len(rg_0), 0)

        idx_2, rf_2, rg_2 = filter_masked_presence_v1([5, 6, 7, 8], [-0.4, 1.2, -0.9, 0.3], [1, 0, 1, 1], [True, False, True, False])
        self.assertEqual(idx_2.tolist(), [5, 7])
        self.assertAlmostEqual(float(rf_2[1]), -0.9, places=5)
        self.assertEqual(rg_2.tolist(), [1.0, 1.0])

        fams = ["A"] * 5 + ["B"] * 3 + ["C"] * 4
        donor = derange_across_families(fams, seed=7)
        for i, d_i in enumerate(donor.tolist()):
            self.assertNotEqual(fams[i], fams[d_i])

    def test_part_b_and_part_c_synthetic_end_to_end(self) -> None:
        fx = _make_synthetic_study_fixture()
        cids = fx["cids"]
        fams = fx["families"]
        folds = fx["folds"]
        pools = fx["pools"]
        champs = fx["champs"]
        repr_tr = fx["repr"]

        # Verify TwoTowerRetriever Matryoshka [32, 64] shapes
        mod = TwoTowerRetriever(d_ctx=859, d_prog=436)
        u = mod.encode_context(torch.randn(3, 859))
        v, b = mod.encode_programs(torch.randn(10, 436), torch.zeros(10))
        s32, s64 = mod.score_bank(u, v, b)
        self.assertEqual(tuple(s32.shape), (3, 10))
        self.assertEqual(tuple(s64.shape), (3, 10))

        def _base_cost_fn(cid: str, bp_vec: np.ndarray) -> np.ndarray:
            n_m = len(pools[cid]["pids"])
            neutral = np.linspace(0.0, 0.05, num=n_m, dtype=np.float32)
            return neutral + bp_vec

        pb = run_part_b_two_tower_study(
            cids, fams, pools, champs, repr_tr, folds, _base_cost_fn,
            seeds=(1,), contrast_fn=paired_contrast, smoke=False,
        )
        u_ultra = pb["cross_context_champion_slate_k96"]["ret_ultra_packet_2t"]["champ_top1_family_utility"]
        u_shuf = pb["cross_context_champion_slate_k96"]["ret_ultra_packet_shuffled"]["champ_top1_family_utility"]
        self.assertGreater(u_ultra, u_shuf)

        i10 = run_i10_zero_shot_archetype_cv(
            cids, fams, pools, repr_tr, folds, seeds=(1,), contrast_fn=paired_contrast, smoke=False
        )
        self.assertGreater(
            i10["arms"]["i10_ultra_packet_859"]["family_mean_utility"],
            i10["arms"]["i10_none_0d"]["family_mean_utility"],
        )

        # Verify I9 z_ctx formula and [0.5 * z0, 2.0 * z0] clipping bounds
        z_scaled = compute_i9_z_ctx(np.array([0.01, 1.0, 100.0]), sigma_med=1.0, z0=0.40)
        self.assertAlmostEqual(float(z_scaled[0]), 0.20, places=6)
        self.assertAlmostEqual(float(z_scaled[1]), 0.40, places=6)
        self.assertAlmostEqual(float(z_scaled[2]), 0.80, places=6)

    def test_code_and_artifact_contracts(self) -> None:
        expected_modules = (
            "h256_token_layout_audit.py",
            "prog412_collision_audit.py",
            "two_tower_retrieval.py",
            "run_h473_study.py",
            "test_h473_study.py",
        )
        for mod_name in expected_modules:
            self.assertTrue((HERE / mod_name).is_file(), f"missing module {mod_name}")

        study_digests = (
            "13a67b2e36f5e6579d67fa25d88ddb81711d4f57c433027420ae2ff1316d88e8",
            "45a886276d57068cdbf403926137e0971dc53a16da9f7ff333b1226935368ea0",
            "e01365377df6a5fcf10d2b5d22858ce837c4943c6ccea8458990dfdb4fa5235e",
            "17ce3bd88333a62f4086b382c022a66e29a673b98d12bd8cd5316787763cf89f",
        )
        prereg_text = (HERE / "PREREG.md").read_text(encoding="utf-8")
        for mod_name, want_sha256 in zip(expected_modules[:4], study_digests, strict=True):
            got_sha256 = hashlib.sha256((HERE / mod_name).read_bytes()).hexdigest()
            self.assertEqual(got_sha256, want_sha256, f"SHA-256 drift in {mod_name}")
            self.assertIn(want_sha256, prereg_text, f"PREREG.md missing {mod_name} SHA-256")
        self_sha256 = hashlib.sha256((HERE / "test_h473_study.py").read_bytes()).hexdigest()
        self.assertIn(self_sha256, prereg_text, "PREREG.md missing test_h473_study.py SHA-256")

        def _forbid_nonfinite(val: str) -> None:
            raise ValueError(f"non-finite JSON constant: {val}")

        layout_path = HERE / "token_layout.json"
        if layout_path.is_file():
            doc = json.loads(layout_path.read_text(encoding="utf-8"), parse_constant=_forbid_nonfinite)
            verify_contiguous_token_layout(doc)

        metrics_path = HERE / "metrics.json"
        verdict_path = HERE / "verdict.json"
        if metrics_path.is_file() and verdict_path.is_file():
            m_doc = json.loads(metrics_path.read_text(encoding="utf-8"), parse_constant=_forbid_nonfinite)
            v_doc = json.loads(verdict_path.read_text(encoding="utf-8"), parse_constant=_forbid_nonfinite)
            self.assertEqual(m_doc["provenance"]["dope_before_sha256"], DOPE_FUSED_COMPOSITE_SHA256)
            self.assertEqual(m_doc["provenance"]["dope_after_sha256"], DOPE_FUSED_COMPOSITE_SHA256)
            self.assertTrue(m_doc["provenance"]["dope_frozen_unchanged"])
            self.assertEqual(m_doc["provenance"]["query_rows_used"], 0)
            self.assertEqual(m_doc["provenance"]["train_dev_family_overlap"], 0)
            self.assertLessEqual(float(m_doc["provenance"]["scratch_used_mb"]), 50.0)
            self.assertEqual(v_doc["dope_composite_sha256"], DOPE_FUSED_COMPOSITE_SHA256)
            self.assertTrue(math.isfinite(float(m_doc["provenance"]["total_wall_s"])))


if __name__ == "__main__":
    unittest.main()
