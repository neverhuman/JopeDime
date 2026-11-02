"""Deterministic CPU-only unit tests for H204 Bigger Proposer (`test_h204.py`)."""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest
import torch
import torch.nn as nn
from jope.north_star.dope_freeze_guard import (
    DopeFreezeViolationError,
    assert_optimizer_excludes_dope,
    assert_research_optimizer_policy,
)

import research.flywheel_selfplay.rust_evaluator as rust_evaluator
from research.flywheel_selfplay.h204_bigger_proposer.analysis import (
    AUTHORITY_STRING,
    EVALUATOR_CONTRACT_ID,
    R882_EXPECTED_DL_VAL_MEAN_COST,
    R882_EXPECTED_DL_VAL_PICKS,
    STAGE1_ARMS,
    WIDTHS,
    assign_family_folds,
    evaluate_arm_gate,
    evaluate_h178_selector,
    first_argmin_cost,
    holm_bonferroni,
    paired_family_bootstrap,
    verify_r882_native_parity,
)
from research.flywheel_selfplay.h204_bigger_proposer.labeler import (
    EvalContextBatch,
    LabeledContextBatch,
    LocalLRUCache,
    check_disk_budget,
    evaluate_context_candidates_rust_cv,
    select_candidate_subset_indices,
)
from research.flywheel_selfplay.h204_bigger_proposer.monitor import (
    build_launch_manifest,
    verify_numeric_thread_rule_b,
)
from research.flywheel_selfplay.h204_bigger_proposer.trainer import (
    R882AdapterContinuation,
    WiderBlockFormerProposer,
    compute_live_tokens_10x32,
    compute_proposer_ranking_loss,
    configure_dope_freeze_or_research_unfreeze,
    enforce_numeric_thread_limits,
    train_and_eval_single_fold,
)

WAVE_DIR = Path(__file__).resolve().parent


class _DummyRouter(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.task_head = nn.Linear(256, 128)
        self.dist_head = nn.Linear(256, 128)
        self.gate_head = nn.Linear(256, 14)
        self.lm_head = nn.Linear(256, 23)
        self.bb_head = nn.Linear(256, 16)
        self.drop = nn.Dropout(0.1)

    def forward(self, vec256: torch.Tensor, level: str = "high") -> dict[str, Any]:
        h = self.drop(vec256)
        g_raw = self.gate_head(h)
        gates = {f"g{i}": g_raw[:, i] for i in range(14)}
        return {
            "task_latent": self.task_head(h),
            "dist_latent": self.dist_head(h),
            "gates": gates,
            "landmark_pred": self.lm_head(h),
            "bb_pred": self.bb_head(h),
        }


class _DummyTargetNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.col_gate_head = nn.Linear(145, 1)


class _DummyDopeModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.router = _DummyRouter()
        self.target_net = _DummyTargetNet()


class _DummyAdapter10Tok(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Linear(16, 16)
        self.mm_proj = nn.Linear(32, 16)
        self.prog_proj = nn.Linear(412, 16)
        self.out_head = nn.Linear(16, 1)

    def encode_multimodal_context_and_candidates(
        self,
        mm_tokens_10x32: torch.Tensor,
        mm_mask_10: torch.Tensor,
        prog_412: torch.Tensor,
        sos_prior: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        ctx_p = self.mm_proj(mm_tokens_10x32.mean(dim=1))
        cand_q = self.prog_proj(prog_412)
        sem24 = prog_412[:, :, :24]
        raw = self.out_head(torch.tanh(cand_q + ctx_p.unsqueeze(1))).squeeze(-1)
        mm_adj = raw - raw.mean(dim=-1, keepdim=True)
        return ctx_p, cand_q, sem24, mm_adj


class _DummyFusedNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dope_model = _DummyDopeModel()
        self.gamenet_anchor = nn.Linear(16, 16)
        self.adapter_10tok = _DummyAdapter10Tok()
        self.step3_gate_keys_14 = [f"g{i}" for i in range(14)]
        self.proj_2 = nn.Linear(629, 32, bias=False)
        self.proj_4 = nn.Linear(629, 32, bias=False)
        self.proj_5 = nn.Linear(629, 32, bias=False)
        self.proj_9 = nn.Linear(64, 32, bias=False)

    def _svd_slot_project(self, x: torch.Tensor, prefix: str) -> torch.Tensor:
        if prefix == "svd_slot2_m2":
            return self.proj_2(x)
        if prefix == "svd_slot4_d629_mid":
            return self.proj_4(x)
        if prefix == "svd_slot5_d629_tail":
            return self.proj_5(x)
        if prefix == "svd_slot9_kron":
            return self.proj_9(x)
        raise KeyError(prefix)


def test_prereg_and_manifest_contain_verbatim_authority(tmp_path: Path) -> None:
    prereg_text = (WAVE_DIR / "PREREG.md").read_text(encoding="utf-8")
    assert AUTHORITY_STRING in prereg_text
    assert EVALUATOR_CONTRACT_ID in prereg_text

    manifest = json.loads((WAVE_DIR / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["authority"] == AUTHORITY_STRING
    assert manifest["wave"] == "H204"
    assert manifest["host"] == "xbabe2"
    assert manifest["widths"] == list(WIDTHS)
    assert manifest["stage1_arms"] == list(STAGE1_ARMS)
    assert manifest["merged_unblock_prs"] == [1726, 1730]
    assert manifest["decision"] == "NO_ADVANCE"
    assert manifest["s3_run_evidence_uri"].startswith(
        "s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261101/H204/"
    )

    metrics = json.loads((WAVE_DIR / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["authority"] == AUTHORITY_STRING
    assert metrics["decision"] == "NO_ADVANCE"
    assert metrics["stage2_launched"] is False
    assert metrics["r882_baseline_dl_val_mean_cost"] == pytest.approx(
        R882_EXPECTED_DL_VAL_MEAN_COST, abs=1e-12
    )

    launch = build_launch_manifest(
        run_dir=tmp_path / "run-test",
        prereg_pr=1729,
        source_commit="fe9d1a5512dc4eb1a248baa171e77128baf39943",
    )
    assert launch["authority"] == AUTHORITY_STRING
    assert (tmp_path / "run-test" / "launch.json").is_file()

    with pytest.raises(ValueError, match="prereg_pr"):
        build_launch_manifest(
            run_dir=tmp_path / "run-bad-pr",
            prereg_pr="pending",
            source_commit="fe9d1a5512dc4eb1a248baa171e77128baf39943",
        )
    with pytest.raises(ValueError, match="source_commit"):
        build_launch_manifest(
            run_dir=tmp_path / "run-bad-sha",
            prereg_pr=1729,
            source_commit="HEAD",
        )


def test_dope_freeze_guard_default_and_research_opt_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    net = _DummyFusedNet()

    # 1. Default path: DOPE remains frozen and in eval mode
    monkeypatch.delenv("JOPE_RESEARCH_DOPE_UNFREEZE", raising=False)
    monkeypatch.delenv("JOPE_RESEARCH_RUN_ID", raising=False)
    cfg_frozen = configure_dope_freeze_or_research_unfreeze(
        net, request_research_unfreeze=False, entrypoint_mode="research"
    )
    assert cfg_frozen["research_dope_unfreeze"] is False
    assert cfg_frozen["dope_trainable_params"] == 0
    assert all(not p.requires_grad for p in net.dope_model.parameters())
    opt_frozen = torch.optim.AdamW(net.adapter_10tok.mm_proj.parameters(), lr=1e-3)
    assert assert_optimizer_excludes_dope(opt_frozen, net) is True
    assert (
        assert_research_optimizer_policy(opt_frozen, net, research_run_id="h204-unit")
        is True
    )

    # 2. Serving and flywheel entrypoints strictly refuse unfreeze even if env is set
    monkeypatch.setenv("JOPE_RESEARCH_DOPE_UNFREEZE", "1")
    monkeypatch.setenv("JOPE_RESEARCH_RUN_ID", "h204-unit-test")
    for mode in ("serving", "flywheel", "production", "promotion"):
        with pytest.raises(
            DopeFreezeViolationError, match="refuses JOPE_RESEARCH_DOPE_UNFREEZE"
        ):
            configure_dope_freeze_or_research_unfreeze(
                net, request_research_unfreeze=True, entrypoint_mode=mode
            )

    # 3. Research unfreeze without run_id is refused
    monkeypatch.setenv("JOPE_RESEARCH_DOPE_UNFREEZE", "1")
    monkeypatch.setenv("JOPE_RESEARCH_RUN_ID", "")
    with pytest.raises(DopeFreezeViolationError, match="JOPE_RESEARCH_RUN_ID"):
        configure_dope_freeze_or_research_unfreeze(
            net, request_research_unfreeze=True, entrypoint_mode="research"
        )

    # 4. Research unfreeze with both env flag and run_id succeeds and keeps DOPE in eval mode
    monkeypatch.setenv("JOPE_RESEARCH_DOPE_UNFREEZE", "1")
    monkeypatch.setenv("JOPE_RESEARCH_RUN_ID", "h204-unit-test")
    launch_meta: dict[str, Any] = {}
    cfg_unfrozen = configure_dope_freeze_or_research_unfreeze(
        net,
        request_research_unfreeze=True,
        entrypoint_mode="research",
        launch_metadata=launch_meta,
        research_run_id="h204-unit-test",
    )
    assert cfg_unfrozen["research_dope_unfreeze"] is True
    assert cfg_unfrozen["dope_trainable_params"] > 0
    assert launch_meta["dope_policy"]["research_dope_unfreeze"] is True
    assert launch_meta["dope_policy"]["serving_eligible"] is False
    net.train(True)
    assert all(not sub.training for sub in net.dope_model.modules())
    assert all(not p.requires_grad for p in net.gamenet_anchor.parameters())
    opt_research = torch.optim.AdamW(
        net.dope_model.parameters(), lr=1e-5, weight_decay=0.0
    )
    assert (
        assert_research_optimizer_policy(
            opt_research, net, research_run_id="h204-unit-test"
        )
        is True
    )


def test_thread_rule_b_enforcement_and_verification() -> None:
    info = enforce_numeric_thread_limits(4)
    assert info["numeric_thread_cap"] == 4
    assert info["torch_num_threads"] <= 4
    assert info["cuda_driver_helper_threads_exempt"] is True
    assert info["admission"]["numeric_threads_upper_bound"] <= 4

    check = verify_numeric_thread_rule_b(4)
    assert check["ok"] is True
    assert check["cuda_runtime_threads_exempt"] is True
    assert check["violations"] == []
    assert check["admission"]["numeric_threads_upper_bound"] <= 4


def test_proposer_ranking_loss_prefers_lower_predicted_cost_for_lower_target_cost() -> (
    None
):
    # Target costs: candidate 0 has lower cost (0.0) than candidate 1 (1.0)
    target_costs = torch.tensor([[0.0, 1.0]], dtype=torch.float32)
    cand_gates = torch.tensor([[1.0, 1.0]], dtype=torch.float32)

    good_preds = torch.tensor([[0.0, 1.0]], dtype=torch.float32)
    bad_preds = torch.tensor([[1.0, 0.0]], dtype=torch.float32)

    loss_good = float(
        compute_proposer_ranking_loss(good_preds, target_costs, cand_gates).item()
    )
    loss_bad = float(
        compute_proposer_ranking_loss(bad_preds, target_costs, cand_gates).item()
    )
    assert loss_good < loss_bad


def test_wider_proposer_shapes_and_r882_continuation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("JOPE_RESEARCH_DOPE_UNFREEZE", "1")
    monkeypatch.setenv("JOPE_RESEARCH_RUN_ID", "h204-shape-test")

    for w in (64, 128, 256, 512):
        prop = WiderBlockFormerProposer(
            width=w, cand_dim=42, ctx_sum_dim=48, anc_sum_dim=10
        )
        tok = torch.randn(4, 10, 32)
        ctx = torch.randn(4, 48)
        anc = torch.ones(4, 10)
        cand = torch.randn(4, 12, 42)
        anc_cand = cand[:, :4, :]
        base = torch.randn(4, 12)
        out = prop(tok, ctx, anc, cand, anc_cand, base)
        assert out.shape == (4, 12)

    net = _DummyFusedNet()
    net.eval()
    r882_cont = R882AdapterContinuation(net.adapter_10tok)
    tok = torch.randn(3, 10, 32)
    ctx = torch.randn(3, 48)
    anc = torch.ones(3, 10)
    cand = torch.randn(3, 8, 42)
    base = torch.randn(3, 8)
    out0 = r882_cont(tok, ctx, anc, cand, cand[:, :4, :], base)
    assert torch.allclose(out0, base, atol=1e-6)
    assert all(not p.requires_grad for p in r882_cont.adapter.anchor.parameters())

    rng = np.random.default_rng(42)
    n_tr = 12
    k_sub = 8
    n_full = 16
    fams = [f"fam_{i % 5}" for i in range(n_tr)]

    vec256 = torch.randn(n_tr, 256)
    with torch.no_grad():
        routed = net.dope_model.router(vec256, level="high")
        gates_14 = torch.stack(
            [routed["gates"][k] for k in net.step3_gate_keys_14], dim=-1
        )
        tail_64 = torch.ones(n_tr, 64)
        d629 = torch.cat(
            [
                vec256,
                routed["task_latent"],
                routed["dist_latent"],
                gates_14,
                routed["landmark_pred"],
                routed["bb_pred"],
                tail_64,
            ],
            dim=-1,
        )
        tok0 = torch.randn(n_tr, 32)
        tok2 = net._svd_slot_project(d629, "svd_slot2_m2")
        tok4 = net._svd_slot_project(d629, "svd_slot4_d629_mid")
        tok5 = net._svd_slot_project(d629, "svd_slot5_d629_tail")
        kron_64 = (tok2[:, :8].unsqueeze(2) * tok0[:, :8].unsqueeze(1)).reshape(
            n_tr, 64
        )
        tok9 = net._svd_slot_project(kron_64, "svd_slot9_kron")
        tokens_10x32 = torch.stack(
            [
                tok0,
                torch.zeros_like(tok0),
                tok2,
                torch.zeros_like(tok0),
                tok4,
                tok5,
                torch.zeros_like(tok0),
                torch.zeros_like(tok0),
                torch.zeros_like(tok0),
                tok9,
            ],
            dim=1,
        )
        live_check = compute_live_tokens_10x32(net, d629, tokens_10x32)
        assert float(torch.max(torch.abs(live_check - tokens_10x32)).item()) == 0.0

    train_batch = LabeledContextBatch(
        context_ids=[f"c_{i}" for i in range(n_tr)],
        family_ids=fams,
        dope629=d629.numpy(),
        tokens_10x32=tokens_10x32.numpy(),
        mask_10=np.ones((n_tr, 10), dtype=np.float32),
        lm48_z=rng.standard_normal((n_tr, 48)).astype(np.float32),
        cand_indices=np.tile(np.arange(k_sub, dtype=np.int64), (n_tr, 1)),
        cand_features=rng.standard_normal((n_tr, k_sub, 42)).astype(np.float32),
        cand_utilities=rng.standard_normal((n_tr, k_sub)).astype(np.float32),
        cand_gates=np.ones((n_tr, k_sub), dtype=np.float32),
        full_cand_features=rng.standard_normal((n_tr, n_full, 42)).astype(np.float32),
        full_fitness=rng.uniform(0.1, 1.5, size=(n_tr, n_full)),
        full_utilities=rng.standard_normal((n_tr, n_full)),
        full_gates=np.ones((n_tr, n_full), dtype=np.float32),
    )
    val_batch = EvalContextBatch(
        context_ids=["v_0", "v_1", "v_2"],
        family_ids=["vf_0", "vf_1", "vf_0"],
        dope629=d629[:3].numpy(),
        tokens_10x32=tokens_10x32[:3].numpy(),
        mask_10=np.ones((3, 10), dtype=np.float32),
        lm48_z=rng.standard_normal((3, 48)).astype(np.float32),
        full_cand_features=rng.standard_normal((3, n_full, 42)).astype(np.float32),
        full_fitness=rng.uniform(0.1, 1.5, size=(3, n_full)),
        full_utilities=rng.standard_normal((3, n_full)),
        full_gates=np.ones((3, n_full), dtype=np.float32),
    )
    fold_train_mask = np.array([True] * 9 + [False] * 3)
    fold_val_mask = ~fold_train_mask

    res_unfrozen = train_and_eval_single_fold(
        arm_name="PROPOSER_W128",
        width=128,
        unfreeze_dope=True,
        freeze_trunk=False,
        base_net=net,
        train_batch=train_batch,
        train_sub_scores=np.zeros((n_tr, k_sub), dtype=np.float32),
        train_full_scores=np.zeros((n_tr, n_full), dtype=np.float64),
        fold_train_mask=fold_train_mask,
        fold_val_mask=fold_val_mask,
        val_batch=val_batch,
        val_full_scores=np.zeros((3, n_full), dtype=np.float64),
        device=torch.device("cpu"),
        seed=204101,
        epochs=2,
        batch_size=4,
    )
    assert res_unfrozen["dope_weights_changed"] is True
    assert res_unfrozen["max_dope_data_grad_l2"] > 0.0
    assert res_unfrozen["dope_grad_tensors_count"] > 0
    assert res_unfrozen["dope_changed_tensors_count"] > 0
    assert res_unfrozen["step0_token_max_diff"] == 0.0

    res_r882_cont = train_and_eval_single_fold(
        arm_name="R882_CONT",
        width=64,
        unfreeze_dope=False,
        freeze_trunk=False,
        base_net=net,
        train_batch=train_batch,
        train_sub_scores=np.zeros((n_tr, k_sub), dtype=np.float32),
        train_full_scores=np.zeros((n_tr, n_full), dtype=np.float64),
        fold_train_mask=fold_train_mask,
        fold_val_mask=fold_val_mask,
        val_batch=val_batch,
        val_full_scores=np.zeros((3, n_full), dtype=np.float64),
        device=torch.device("cpu"),
        seed=204101,
        epochs=2,
        batch_size=4,
    )
    assert res_r882_cont["dope_weights_changed"] is False
    assert res_r882_cont["dope_changed_tensors_count"] == 0


def test_labeler_lru_cache_and_rust_evaluator_signature_binding(tmp_path: Path) -> None:
    idx = select_candidate_subset_indices(
        n_total=1152, k_candidates=16, seed=204101, context_idx=3
    )
    assert len(idx) == 16
    assert list(idx[:4]) == [0, 1, 2, 3]
    assert len(set(idx.tolist())) == 16

    cache = LocalLRUCache(cache_dir=tmp_path / "lru", max_bytes=100)
    cache.put_bytes("a.bin", b"x" * 60)
    cache.put_bytes("b.bin", b"y" * 60)
    assert cache.current_bytes() <= 100
    assert cache.get_bytes("a.bin") is None
    assert cache.get_bytes("b.bin") == b"y" * 60

    disk = check_disk_budget(tmp_path)
    assert "used_ratio" in disk and "free_bytes" in disk

    # Verify rust_evaluator.evaluate_gp_batch_rust signature matches evaluate_context_candidates_rust_cv call
    sig = inspect.signature(rust_evaluator.evaluate_gp_batch_rust)
    expected_params = {
        "teacher_jtt1_gz_path",
        "candidates",
        "run_outer_test",
        "max_workers",
        "timeout_s",
    }
    assert expected_params.issubset(set(sig.parameters.keys()))

    captured_kwargs: dict[str, Any] = {}

    def _fake_eval(**kwargs: Any) -> list[dict[str, Any]]:
        captured_kwargs.update(kwargs)
        assert Path(kwargs["teacher_jtt1_gz_path"]).is_file()
        assert kwargs["run_outer_test"] is False
        return [
            {
                "dsl": c["dsl"],
                "status": "ok",
                "cv_nmse": 0.25 + 0.1 * i,
                "fold_nmse": [0.25 + 0.1 * i, 0.25 + 0.1 * i],
            }
            for i, c in enumerate(kwargs["candidates"])
        ]

    eval_cache = LocalLRUCache(cache_dir=tmp_path / "eval_lru", max_bytes=1024 * 1024)
    with patch.object(rust_evaluator, "evaluate_gp_batch_rust", side_effect=_fake_eval):
        res_rows = evaluate_context_candidates_rust_cv(
            raw_jtt1_bytes=b"JTT1_TEST_BYTES",
            programs=[f"x0 + {k}" for k in range(8)],
            cache=eval_cache,
            cache_key="ctx_sig_test",
        )
    assert len(res_rows) == 8
    assert res_rows[0]["cv_nmse"] == pytest.approx(0.25)
    assert captured_kwargs["run_outer_test"] is False


def test_analysis_parity_exact_pick_vector_and_promotion_gates() -> None:
    fake_utils = np.full(91, R882_EXPECTED_DL_VAL_MEAN_COST, dtype=np.float64)
    parity = verify_r882_native_parity(
        fake_utils,
        dl_val_picks=np.asarray(R882_EXPECTED_DL_VAL_PICKS, dtype=np.int64),
    )
    assert parity["passed"] is True
    assert parity["exact_pick_matches"] == 91

    # Swapping two picks with identical utilities must FAIL the exact pick parity check
    bad_picks = np.asarray(R882_EXPECTED_DL_VAL_PICKS, dtype=np.int64).copy()
    bad_picks[0] = (bad_picks[0] + 1) % 1152
    bad_parity = verify_r882_native_parity(fake_utils, dl_val_picks=bad_picks)
    assert bad_parity["passed"] is False
    assert bad_parity["exact_pick_matches"] == 90

    fams = [f"f_{i % 10}" for i in range(40)]
    folds = assign_family_folds(fams, n_folds=5, seed=204140)
    assert set(folds.values()) == {0, 1, 2, 3, 4}

    r882 = np.zeros(40, dtype=np.float64)
    cand = np.full(40, 0.02, dtype=np.float64)
    boot = paired_family_bootstrap(
        r882, cand, fams, n_boot=500, seed=204141, higher_is_better=True
    )
    assert boot["mean_gain"] == pytest.approx(0.02)
    assert boot["ci95_low"] > 0.0
    assert boot["elo_lift"] > 10.0

    sel = evaluate_h178_selector(r882, cand, fams, seed=204141, higher_is_better=True)
    assert sel["fallback_to_r882"] is False
    assert sel["n_bootstrap"] == 10_000

    p_adj = holm_bonferroni({"A": 0.01, "B": 0.04})
    assert p_adj["A"] <= p_adj["B"]

    seed_results = [
        {"seed": s, "fallback_rate": 0.0, "dl_val": boot}
        for s in (204101, 204102, 204103)
    ]
    gate_pass = evaluate_arm_gate(
        seed_results, control_best_gain=0.005, holm_adjusted_p=0.01
    )
    assert gate_pass["passed"] is True

    # If Holm-adjusted p > 0.05, gate must fail
    gate_fail_p = evaluate_arm_gate(
        seed_results, control_best_gain=0.005, holm_adjusted_p=0.12
    )
    assert gate_fail_p["passed"] is False

    idx, val = first_argmin_cost(
        np.array([0.5, 0.2, 0.2, 0.9]), np.array([1.0, 2.0, 3.0, 4.0])
    )
    assert idx == 1 and val == 2.0
