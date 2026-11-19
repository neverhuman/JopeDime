"""Offline contract and unit tests for H470 DOPE Value Probe & Zero-Retraining Harvester."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
PROBE_DIR = HERE.parent / "dope_value_probe"
for p in (str(HERE), str(PROBE_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from cache_auth import (  # noqa: E402
    CacheAuthManifest,
    CacheAuthenticationError,
    compute_context_inventory_sha256,
    compute_fit_rows_sha256,
    load_authenticated_npz,
    save_authenticated_npz,
)
from dope_value_probe import split_packet_v1  # noqa: E402
from h470_part_c import FoldedStage1BBottleneck, eval_wp3_history_kernel_single  # noqa: E402
from h470_probe_tasks import (  # noqa: E402
    DVP_CONFIG_SHA256,
    build_pool_Metadata,
    dvp_score,
    make_cross_family_donor_permutation,
    select_pivoted_qr_basket,
)
from packet_v1_extract import (  # noqa: E402
    CANONICAL_GATE_KEYS_14,
    DOPE_FUSED_COMPOSITE_SHA256,
    FORBIDDEN_QUERY_KEYS,
    PACKET_FEATURES_WIDTH,
    PACKET_GLOBAL_FIELDS,
    PACKET_GLOBAL_WIDTH,
    compute_native_23_landmark_report,
    encode_dope_research_view,
    extract_packet_column_tokens_v1,
    packet_features_v1,
    packet_global_v1,
    read_fit_only_jtt1,
)


def _synthetic_encoding(p_use: int = 5, level: str = "high") -> dict[str, object]:
    rng = np.random.default_rng(20261118)
    gates = {k: float(0.05 * (i + 1)) for i, k in enumerate(CANONICAL_GATE_KEYS_14)}
    col_knots = np.sort(rng.standard_normal((p_use, 9)), axis=1)
    chol = rng.standard_normal((p_use, 3)) * 0.25
    diag_res = np.full(p_use, 0.85, dtype=np.float64)
    is_bin = [j == 1 for j in range(p_use)]
    bin_p = [0.35 if j == 1 else 0.0 for j in range(p_use)]
    reg_w = np.array([0.55, 0.30, 0.15], dtype=np.float64)
    reg_m = rng.standard_normal((3, p_use)) * 0.2
    reg_s = np.full((3, p_use), 1.0, dtype=np.float64)
    w_basis = rng.standard_normal(1 + 3 * p_use + 6) * 0.3
    trees = [
        {"features": [0, 2], "thresholds": [-0.25, 0.40], "leaf_values": [-0.3, 0.1, 0.2, 0.5]},
        {"features": [3, 1], "thresholds": [0.0, 0.5], "leaf_values": [0.15, -0.1, 0.05, -0.2]},
    ]
    pairs = [(0, 2, 0.4), (1, 3, -0.25)]
    return {
        "role": "fit",
        "level": level,
        "task": "regression",
        "n_train": 256,
        "p_use": p_use,
        "gates": gates,
        "col_knots": col_knots,
        "chol_factor": chol,
        "diag_res_std": diag_res,
        "is_binary_col": is_bin,
        "binary_probs": bin_p,
        "discrete_support": {2: (np.array([-1.0, 0.0, 1.0, 2.0]), np.array([0.25, 0.5, 0.75, 1.0]))},
        "regime_weights": reg_w,
        "regime_means": reg_m,
        "regime_scales": reg_s,
        "task_latent": rng.standard_normal(128).astype(np.float32),
        "dist_latent": rng.standard_normal(128).astype(np.float32),
        "target_bundle": {
            "task_kind": "regression",
            "residual_std": 0.42,
            "neural_w": 0.18,
            "res_net_mean": 0.03,
            "class_prior": 0.5,
            "y_min": -12.5,
            "y_max": 48.0,
            "w_basis": w_basis,
            "trees": trees,
            "pairs": pairs,
            "include_v3_primitives": True,
        },
    }


def _permute_synthetic_encoding_columns(enc: dict[str, object], perm: list[int]) -> dict[str, object]:
    """Permute column slots of an `encode_table` dictionary (`new_slot k = old_slot perm[k]`)."""
    p_use = int(enc["p_use"])  # type: ignore[arg-type]
    inv = {old_j: new_j for new_j, old_j in enumerate(perm)}
    out = dict(enc)
    out["col_knots"] = np.asarray(enc["col_knots"])[perm]
    out["chol_factor"] = np.asarray(enc["chol_factor"])[perm]
    out["diag_res_std"] = np.asarray(enc["diag_res_std"])[perm]
    out["is_binary_col"] = [enc["is_binary_col"][j] for j in perm]  # type: ignore[index]
    out["binary_probs"] = [enc["binary_probs"][j] for j in perm]  # type: ignore[index]
    out["regime_means"] = np.asarray(enc["regime_means"])[:, perm]
    out["regime_scales"] = np.asarray(enc["regime_scales"])[:, perm]
    disc_old = enc["discrete_support"]  # type: ignore[assignment]
    out["discrete_support"] = {inv[k]: v for k, v in disc_old.items()}  # type: ignore[union-attr]

    tb = dict(enc["target_bundle"])  # type: ignore[arg-type]
    wb = np.asarray(tb["w_basis"]).copy()
    wb_new = wb.copy()
    for b_block in range(3):
        base = 1 + b_block * p_use
        wb_new[base : base + p_use] = wb[base + np.asarray(perm)]
    tb["w_basis"] = wb_new
    tb["pairs"] = [(inv[a], inv[c], w) for (a, c, w) in tb["pairs"]]
    tb["trees"] = [
        {**t, "features": [inv[f] for f in t["features"]]}
        for t in tb["trees"]
    ]
    out["target_bundle"] = tb
    return out


class _MockRouter(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(140)
        self.v3_inter_norm = nn.LayerNorm(116)
        self.backbone = nn.Linear(140, 256)
        self.level_embed = nn.Embedding(3, 256)
        self.inter_attn = nn.Linear(256, 256)
        self.task_proj = nn.Linear(256, 128)
        self.dist_proj = nn.Linear(256, 128)
        self.lm_proj = nn.Linear(256, 23)
        self.bb_proj = nn.Linear(256, 16)
        self.gate_proj = nn.Linear(256, 14)

    def forward(self, x: torch.Tensor, level: str = "high") -> dict[str, object]:
        lvl = 1 if level == "high" else 2
        x140 = self.input_norm(x[:, :140])
        xe = self.v3_inter_norm(x[:, 140:256])
        h = self.backbone(x140) + self.level_embed(torch.full((x.shape[0],), lvl, dtype=torch.long)) + self.inter_attn(
            torch.cat([x140, xe], dim=-1)
        )
        g_raw = torch.sigmoid(self.gate_proj(h))
        gates = {k: g_raw[:, i] for i, k in enumerate(CANONICAL_GATE_KEYS_14)}
        return {
            "task_latent": self.task_proj(h),
            "dist_latent": self.dist_proj(h),
            "landmark_pred": self.lm_proj(h),
            "bb_pred": self.bb_proj(h),
            "gates": gates,
        }


class _MockTargetNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.col_gate_head = nn.Linear(128 + 17, 1)


class _MockDopeModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.router = _MockRouter()
        self.target_net = _MockTargetNet()


class _MockFusedStage1B(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        g = torch.Generator().manual_seed(20261118)
        for idx, slot_name in enumerate(("svd_slot2_m2", "svd_slot4_d629_mid", "svd_slot5_d629_tail")):
            mu = torch.randn(629, generator=g) * (0.2 + 0.05 * idx)
            sd = torch.rand(629, generator=g) + 0.5
            mask = (torch.arange(629) % 7 != 0).float()
            proj = torch.randn(629, 32, generator=g) * 0.05
            self.register_buffer(f"{slot_name}_mu", mu)
            self.register_buffer(f"{slot_name}_sd_safe", sd)
            self.register_buffer(f"{slot_name}_active_mask", mask)
            self.register_buffer(f"{slot_name}_effective_proj_mat", proj)
        self.register_buffer("svd_slot9_kron_mu", torch.zeros(64))
        self.register_buffer("svd_slot9_kron_sd_safe", torch.ones(64))
        self.register_buffer("svd_slot9_kron_active_mask", torch.ones(64))
        self.register_buffer("svd_slot9_kron_effective_proj_mat", torch.randn(64, 32, generator=g) * 0.08)


class H470ProbeContractTests(unittest.TestCase):
    def test_wp01_cache_authenticator_validates_and_refuses_tampering(self) -> None:
        recs = [
            {
                "context_sha256": "a" * 64,
                "family_id": "fam_a",
                "dl_split": "DL-Train",
                "train_only_jtt1_sha256": "b" * 64,
            }
        ]
        x_fit = np.ones((16, 4), dtype=np.float64)
        y_fit = np.linspace(0.0, 1.0, 16, dtype=np.float64)
        ctx_sha = compute_context_inventory_sha256(recs)
        fit_sha = compute_fit_rows_sha256(["a" * 64], {"a" * 64: (x_fit, y_fit)}, {"a" * 64: 0})

        with self.assertRaisesRegex(CacheAuthenticationError, "query rows"):
            compute_fit_rows_sha256(["a" * 64], {"a" * 64: (x_fit, y_fit)}, {"a" * 64: 3})

        manifest = CacheAuthManifest(
            dope_weights_sha256=DOPE_FUSED_COMPOSITE_SHA256,
            stage1a_mode_and_bridge_sha256="c" * 64,
            prep_recipe="unit-test-v1",
            context_sha256=ctx_sha,
            fit_rows_sha256=fit_sha,
            conditioning_level="high",
            stage1b_sha256="d" * 64,
        )
        with tempfile.TemporaryDirectory() as tmp:
            npz_p, meta_p = save_authenticated_npz(
                tmp, "unit_bundle", manifest, {"R_packet_v1": np.ones((4, 196), dtype=np.float32)}
            )
            loaded = load_authenticated_npz(tmp, "unit_bundle", manifest)
            self.assertEqual(loaded["R_packet_v1"].shape, (4, 196))

            # Mismatched conditioning level must fail cache lookup / verification
            wrong_lvl = CacheAuthManifest(**{**manifest.__dict__, "conditioning_level": "ultra"})
            with self.assertRaises(CacheAuthenticationError):
                load_authenticated_npz(
                    tmp, "unit_bundle", wrong_lvl, explicit_npz_path=npz_p, explicit_meta_path=meta_p
                )

            # Tampering sidecar query_rows_used must fail
            sidecar = json.loads(meta_p.read_text(encoding="utf-8"))
            sidecar["query_rows_used"] = 1
            meta_p.write_text(json.dumps(sidecar), encoding="utf-8")
            with self.assertRaisesRegex(CacheAuthenticationError, "query_rows_used"):
                load_authenticated_npz(tmp, "unit_bundle", manifest)

    def test_jtt1_reader_accepts_fit_only_and_refuses_query_rows(self) -> None:
        nt, width = 12, 3
        mat = np.arange(nt * (width + 1), dtype="<f8").reshape(nt, width + 1)
        fold_bytes = bytes([i % 2 for i in range(nt)])
        good_raw = b"JTT1" + struct.pack("<III", nt, 0, width) + bytes([2]) + fold_bytes + mat.tobytes()
        bad_q_raw = b"JTT1" + struct.pack("<III", nt, 4, width) + bytes([2]) + fold_bytes + mat.tobytes()
        with tempfile.TemporaryDirectory() as tmp:
            good_p = Path(tmp) / "good.jtt1"
            good_p.write_bytes(good_raw)
            xf, yf, meta = read_fit_only_jtt1(good_p, expected_sha256=hashlib.sha256(good_raw).hexdigest())
            self.assertEqual(xf.shape, (12, 3))
            self.assertEqual(yf.shape, (12,))
            self.assertEqual(meta["n_query"], 0)

            bad_p = Path(tmp) / "bad_q.jtt1"
            bad_p.write_bytes(bad_q_raw)
            with self.assertRaisesRegex(ValueError, "query rows"):
                read_fit_only_jtt1(bad_p)

    def test_packet_v1_extrema_free_ps2_ablation_ps3_permutation_and_query_refusal(self) -> None:
        self.assertEqual(len(PACKET_GLOBAL_FIELDS), PACKET_GLOBAL_WIDTH)
        self.assertEqual(PACKET_GLOBAL_FIELDS.count("noise_scale"), 1)
        self.assertNotIn("y_min", PACKET_GLOBAL_FIELDS)
        self.assertNotIn("y_max", PACKET_GLOBAL_FIELDS)

        enc = _synthetic_encoding(p_use=5, level="high")
        feat = packet_features_v1(enc)
        glob = packet_global_v1(enc)
        self.assertEqual(len(feat), PACKET_FEATURES_WIDTH)
        self.assertEqual(glob, feat[:PACKET_GLOBAL_WIDTH])
        blocks = split_packet_v1(feat)
        self.assertEqual(sum(len(v) for v in blocks.values()), PACKET_FEATURES_WIDTH)

        # Extreme changes to y_min / y_max must have ZERO effect on R_packet_v1
        enc_ext = dict(enc)
        enc_ext["target_bundle"] = dict(enc["target_bundle"], y_min=-9.87e6, y_max=6.54e6)  # type: ignore[arg-type]
        self.assertEqual(packet_features_v1(enc_ext), feat)

        # P-S2 covariate-only ablation zeros target-bundle slots while preserving covariates & latents
        notgt = packet_features_v1(enc, include_target_bundle=False)
        self.assertEqual(notgt[14:17], (0.0, 0.0, 0.0))
        self.assertEqual(notgt[40:44], (0.0, 0.0, 0.0, 0.0))
        self.assertEqual(notgt[172:188], (0.0,) * 16)
        self.assertEqual(notgt[188:196], feat[188:196])

        # P-S3 column permutation equivariance on column tokens & exact invariance on pooled R_packet_v1
        perm = [3, 0, 4, 1, 2]
        enc_perm = _permute_synthetic_encoding_columns(enc, perm)
        toks_orig, act_orig = extract_packet_column_tokens_v1(enc)
        toks_perm, act_perm = extract_packet_column_tokens_v1(enc_perm)
        for new_j, old_j in enumerate(perm):
            np.testing.assert_allclose(toks_perm[new_j], toks_orig[old_j], rtol=0.0, atol=1e-7)
            self.assertAlmostEqual(act_perm[new_j], act_orig[old_j], places=7)
        feat_perm = packet_features_v1(enc_perm)
        np.testing.assert_allclose(feat_perm, feat, rtol=0.0, atol=1e-6)

        # Query-key and canary-extrema redaction checks
        canary_extrema = "7777.25"
        for qk in FORBIDDEN_QUERY_KEYS:
            with self.assertRaises(ValueError) as ctx:
                packet_features_v1({**enc, qk: canary_extrema})
            self.assertIn(qk, str(ctx.exception))
            self.assertNotIn(canary_extrema, str(ctx.exception))
        with self.assertRaisesRegex(ValueError, "role"):
            packet_features_v1({**enc, "role": "query"})

    def test_encode_dope_research_view_enforces_level_match_and_frozen_eval(self) -> None:
        torch.manual_seed(7)
        model = _MockDopeModel().eval()
        for p in model.parameters():
            p.requires_grad_(False)
        x = torch.randn(3, 256)
        c = torch.randn(3, 64, 17)
        d_hi, h_hi = encode_dope_research_view(model, x, c, level="high", conditioning_level="high")
        d_ul, h_ul = encode_dope_research_view(model, x, c, level="ultra", conditioning_level="ultra")
        self.assertEqual(tuple(d_hi.shape), (3, 629))
        self.assertEqual(tuple(h_ul.shape), (3, 256))
        self.assertFalse(torch.equal(d_hi[:, 256:], d_ul[:, 256:]))

        with self.assertRaisesRegex(ValueError, "conditioning_level"):
            encode_dope_research_view(model, x, c, level="ultra", conditioning_level="high")
        with self.assertRaisesRegex(ValueError, "conditioning_level"):
            encode_dope_research_view(model, x, c, level="high", conditioning_level="ultra")

        bad_x = x.clone()
        bad_x[0, 5] = float("inf")
        with self.assertRaisesRegex(ValueError, "non-finite"):
            encode_dope_research_view(model, bad_x, c, level="high", conditioning_level="high")

        model.train(True)
        with self.assertRaisesRegex(ValueError, "eval"):
            encode_dope_research_view(model, x, c, level="high", conditioning_level="high")

    def test_native_23_landmark_report_and_probe_helpers(self) -> None:
        rng = np.random.default_rng(42)
        x = rng.standard_normal((60, 6))
        y = 0.8 * x[:, 0] - 0.5 * (x[:, 1] ** 2) + 0.1 * rng.standard_normal(60)
        rep = compute_native_23_landmark_report(x, y, n_folds=5, seed=20261023)
        self.assertEqual(len(rep["cells"]), 23)
        self.assertGreater(rep["timing"]["fine_5_landmarks_ms"], 0.0)

        # Cross-family donor permutation must never map any context to its own family
        fams = ["f0"] * 10 + ["f1"] * 7 + ["f2"] * 5 + ["f3"] * 3
        donor = make_cross_family_donor_permutation(fams, seed=20261118)
        self.assertEqual(len(donor), len(fams))
        for i, d_i in enumerate(donor.tolist()):
            self.assertNotEqual(fams[i], fams[d_i])

        # Pivoted-QR basket selection
        z_tr = rng.standard_normal((40, 12)).astype(np.float32)
        sup = np.full(12, 40, dtype=np.int64)
        sel = select_pivoted_qr_basket(z_tr, sup, 6)
        self.assertEqual(len(sel), 6)
        self.assertEqual(len(set(sel.tolist())), 6)

    def test_dvp_score_and_wp3_vs1_and_b03_folded_svd(self) -> None:
        # 1. Synthetic `dvp_score` smoke (n_ctx >= 30 so T3 per-anchor R^2 path is fully exercised)
        rng = np.random.default_rng(99)
        n_ctx = 32
        cids = [f"cid_{i:02d}" for i in range(n_ctx)]
        fams = [f"fam_{i // 2}" for i in range(n_ctx)]
        common_pids = [f"{j:064x}" for j in range(90)]
        pools = {}
        for i, cid in enumerate(cids):
            prog = rng.standard_normal((90, 412)).astype(np.float32) * 0.2
            prog[:, 24 + (i % 5)] = 1.0
            proj16 = rng.standard_normal((90, 16)).astype(np.float32) * 0.2
            sem24 = rng.standard_normal((90, 24)).astype(np.float32)
            sem24 /= np.maximum(np.linalg.norm(sem24, axis=1, keepdims=True), 1e-6)
            fit = rng.uniform(-0.8, 0.4, size=90)
            gate = np.ones(90, dtype=bool)
            gate[-5:] = False
            pools[cid] = {
                "pids": common_pids,
                "program": prog,
                "proj16": proj16,
                "struct_sig": sem24,
                "fitness": fit,
                "gate": gate,
            }
        meta = build_pool_Metadata(cids, pools)
        labels = {
            **meta,
            "families": fams,
            "gains23_ultra": rng.uniform(0.0, 0.8, size=(n_ctx, 23)).astype(np.float32),
            "mask23_ultra": np.ones((n_ctx, 23), dtype=np.uint8),
        }
        folds = [np.array([i % 5 for i in range(n_ctx)], dtype=np.int64)]
        r_mat = rng.standard_normal((n_ctx, 12)).astype(np.float32)
        scored = dvp_score(r_mat, labels, folds, seeds=(1,), num_threads=2, run_4rev=False, compute_t4=True)
        self.assertEqual(scored["config_sha256"], DVP_CONFIG_SHA256)
        self.assertEqual(len(scored["t4"]["per_landmark_r2"]), 5)

        # 2. WP3 V-S1 (`lam = 0` exact zero-reveal identity)
        entry = {
            "cost0": rng.standard_normal(16).astype(np.float32),
            "anchor_total0": rng.standard_normal(16).astype(np.float32),
            "eff_prior0": rng.standard_normal(16).astype(np.float32),
            "cand_q0": rng.standard_normal((16, 32)).astype(np.float32),
            "sem24": rng.standard_normal((16, 24)).astype(np.float32),
            "gate": np.ones(16, dtype=bool),
            "util": rng.uniform(0.0, 1.0, size=16),
        }
        zeros = np.zeros(16, dtype=np.float32)
        c_out, _, _, _ = eval_wp3_history_kernel_single(
            None,
            entry,
            np.array([0, 1, 2, 3], dtype=np.int64),
            np.ones(4, dtype=np.float32),
            np.ones(4, dtype=np.float32),
            zeros,
            np.ones(16, dtype=np.float32),
            beta_direct=0.5,
        )
        self.assertEqual(float(np.max(np.abs(c_out - entry["cost0"]))), 0.0)

        # 3. B03 FoldedStage1BBottleneck step-0 bit-identity and gradient access
        mock_net = _MockFusedStage1B()
        base_tok = torch.clamp(torch.randn(4, 10, 32), -3.0, 3.0)
        # Recompute exact base slot 9 so base_tok is self-consistent with `_project_slot9`
        bn_svd = FoldedStage1BBottleneck(mock_net, "svd96")
        with torch.no_grad():
            base_tok[:, 9] = bn_svd._project_slot9(base_tok[:, 2], base_tok[:, 0])
        d629 = torch.randn(4, 629, requires_grad=True)
        extra = torch.randn(4, 859, requires_grad=True)

        out_svd0 = bn_svd(base_tok, dope629=d629)
        self.assertEqual(float((out_svd0 - base_tok).abs().max().item()), 0.0)
        # Perturbing linear weights unlocks gradient wrt all active coordinates of d629
        with torch.no_grad():
            bn_svd.linear.weight.add_(0.01 * torch.randn_like(bn_svd.linear.weight))
        out_svd1 = bn_svd(base_tok, dope629=d629)
        out_svd1.sum().backward()
        assert d629.grad is not None
        self.assertGreater(int((d629.grad.abs().sum(dim=0) > 0).sum().item()), 500)

        bn_up = FoldedStage1BBottleneck(mock_net, "ultra_packet", extra_dim=859)
        out_up0 = bn_up(base_tok, extra_z=extra)
        self.assertEqual(float((out_up0 - base_tok).abs().max().item()), 0.0)

        # 4. Verify `dvp_score(..., compute_t4=True)` (`FINE_LANDMARKS` keys) is 100% JSON-serializable!
        res_t4 = dvp_score(r_mat, labels, [folds], seeds=(1,), num_threads=1, run_4rev=False, compute_t4=True)
        clean_t4 = {
            "config_sha256": res_t4["config_sha256"],
            "width": res_t4["width"],
            "t1_0rev": {k: v for k, v in res_t4["t1_0rev"].items() if not k.endswith("_ctx")},
            "t2": {k: v for k, v in res_t4["t2"].items() if not k.endswith("_ctx")},
            "t3": {k: v for k, v in res_t4["t3"].items() if k != "ctx_skill"},
            "t4": res_t4["t4"],
        }
        dumped_t4 = json.dumps(clean_t4, allow_nan=False)
        parsed_t4 = json.loads(dumped_t4)
        self.assertEqual(
            set(parsed_t4["t4"]["per_landmark_r2"].keys()),
            {"gbdt", "knn1", "knn16", "net", "gbdt_quarter"},
        )

    def test_committed_artifacts_if_present(self) -> None:
        self.assertTrue((HERE / "PREREG.md").is_file())
        metrics_p = HERE / "metrics.json"
        verdict_p = HERE / "verdict.json"
        if metrics_p.is_file() and verdict_p.is_file():
            raw_text = metrics_p.read_text(encoding="utf-8")
            self.assertNotIn('"y_min"', raw_text)
            self.assertNotIn('"y_max"', raw_text)
            metrics = json.loads(raw_text)
            verdict = json.loads(verdict_p.read_text(encoding="utf-8"))
            prov = metrics["provenance"]
            self.assertEqual(prov["wave"], "H470-DVP")
            self.assertEqual(verdict["wave"], "H470-DVP")
            self.assertFalse(prov["smoke_mode"])
            self.assertEqual(prov["dope_before_sha256"], DOPE_FUSED_COMPOSITE_SHA256)
            self.assertEqual(prov["dope_after_sha256"], DOPE_FUSED_COMPOSITE_SHA256)
            self.assertEqual(prov["train_dev_family_overlap"], 0)
            self.assertEqual(prov["outer_or_query_rows_read"], 0)
            self.assertEqual(prov["val91_evaluations_count"], 1)
            self.assertEqual(verdict["criteria"]["DVP-S2_dope_freeze_and_h58_parity"]["status"], "PASS")


if __name__ == "__main__":
    unittest.main()
