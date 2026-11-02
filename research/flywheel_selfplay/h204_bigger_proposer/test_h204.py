"""Unit tests for H204 Bigger Proposer (`128 -> 256 -> 512`) with Research-Only DOPE Unfreeze."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from research.flywheel_selfplay.h204_bigger_proposer.analysis import (
    AUTHORITY_STRING,
    DL_VAL_EXPECTED_CONTEXTS,
    MAX_LOCAL_CACHE_BYTES,
    MAX_NUMERIC_THREADS,
    R882_EXPECTED_DL_VAL_MEAN_COST,
    SEEDS,
    STAGE1_ARMS,
    WAVE,
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
    LocalLRUCache,
    select_candidate_subset_indices,
)
from research.flywheel_selfplay.h204_bigger_proposer.monitor import (
    build_launch_manifest,
    verify_numeric_thread_rule_b,
)
from research.flywheel_selfplay.h204_bigger_proposer.trainer import (
    WiderBlockFormerProposer,
    _hash_module_parameters,
    compute_proposer_ranking_loss,
    configure_dope_freeze_or_research_unfreeze,
    enforce_numeric_thread_limits,
)

H204_DIR = Path(__file__).resolve().parent


class _DummyRouter(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(16, 16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class _DummyDopeModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.router = _DummyRouter()


class _DummyFusedNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dope_model = _DummyDopeModel()
        self.gamenet_anchor = nn.Linear(16, 1)


def test_prereg_and_manifests_contain_verbatim_authority() -> None:
    """Verify verbatim authority string in PREREG.md, manifest.json, and metrics.json."""
    prereg_text = (H204_DIR / "PREREG.md").read_text(encoding="utf-8")
    assert AUTHORITY_STRING in prereg_text
    assert WAVE in prereg_text

    manifest = json.loads((H204_DIR / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["wave"] == "H204"
    assert manifest["authority"] == AUTHORITY_STRING
    assert manifest["widths"] == list(WIDTHS)
    assert manifest["stage1_arms"] == list(STAGE1_ARMS)
    assert manifest["seeds"] == list(SEEDS)
    assert manifest["resource_guards"]["max_local_cache_bytes"] == MAX_LOCAL_CACHE_BYTES
    assert manifest["resource_guards"]["max_numeric_threads"] == MAX_NUMERIC_THREADS
    assert manifest["resource_guards"]["cuda_driver_threads_exempt"] is True

    metrics = json.loads((H204_DIR / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["wave"] == "H204"
    assert metrics["authority"] == AUTHORITY_STRING


def test_build_launch_manifest_records_authority_and_unfreeze(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify launch.json contains the verbatim authority string and DOPE unfreeze receipt."""
    monkeypatch.setenv("JOPE_RESEARCH_RUN_ID", "h204-unit-test")
    launch = build_launch_manifest(
        run_dir=tmp_path / "run-test",
        prereg_pr=1724,
        source_commit="deadbeef",
        device_str="cpu",
    )
    assert launch["authority"] == AUTHORITY_STRING
    assert launch["research_dope_unfreeze"] is True
    assert launch["research_run_id"] == "h204-unit-test"
    assert (tmp_path / "run-test" / "launch.json").is_file()


def test_dope_freeze_default_and_research_opt_in_and_serving_refusal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test flag (a): default freeze, serving/flywheel refusal, and research opt-in updates."""
    net = _DummyFusedNet()

    # 1. Default path keeps DOPE strictly frozen
    monkeypatch.delenv("JOPE_RESEARCH_DOPE_UNFREEZE", raising=False)
    monkeypatch.delenv("JOPE_RESEARCH_RUN_ID", raising=False)
    cfg_default = configure_dope_freeze_or_research_unfreeze(
        net, request_research_unfreeze=False, entrypoint_mode="research"
    )
    assert cfg_default["research_dope_unfreeze"] is False
    assert all(not p.requires_grad for p in net.dope_model.parameters())

    # 2. Requesting unfreeze without env var raises RuntimeError
    with pytest.raises(RuntimeError, match="JOPE_RESEARCH_DOPE_UNFREEZE"):
        configure_dope_freeze_or_research_unfreeze(
            net, request_research_unfreeze=True, entrypoint_mode="research"
        )

    # 3. Requesting unfreeze without JOPE_RESEARCH_RUN_ID raises RuntimeError
    monkeypatch.setenv("JOPE_RESEARCH_DOPE_UNFREEZE", "1")
    with pytest.raises(RuntimeError, match="JOPE_RESEARCH_RUN_ID"):
        configure_dope_freeze_or_research_unfreeze(
            net, request_research_unfreeze=True, entrypoint_mode="research"
        )

    # 4. Serving / flywheel entrypoints strictly refuse JOPE_RESEARCH_DOPE_UNFREEZE=1
    monkeypatch.setenv("JOPE_RESEARCH_RUN_ID", "h204-test-run")
    for forbidden_mode in ("serving", "flywheel", "production", "promotion"):
        with pytest.raises(RuntimeError, match="strictly refused"):
            configure_dope_freeze_or_research_unfreeze(
                net, request_research_unfreeze=True, entrypoint_mode=forbidden_mode
            )

    # 5. Valid research opt-in enables DOPE router grads while keeping gamenet_anchor frozen
    cfg_res = configure_dope_freeze_or_research_unfreeze(
        net, request_research_unfreeze=True, entrypoint_mode="research"
    )
    assert cfg_res["research_dope_unfreeze"] is True
    assert cfg_res["dope_trainable_params"] > 0
    assert all(p.requires_grad for p in net.dope_model.router.parameters())
    assert all(not p.requires_grad for p in net.gamenet_anchor.parameters())

    # Verify parameter hash actually updates after net.eval() + an optimizer step
    net.eval()
    assert all(p.requires_grad for p in net.dope_model.router.parameters())
    before_hash = cfg_res["dope_before_sha256"]
    opt = torch.optim.SGD(net.dope_model.router.parameters(), lr=0.1)
    x = torch.ones(4, 16)
    loss = net.dope_model.router(x).pow(2).mean()
    opt.zero_grad()
    loss.backward()
    opt.step()
    after_hash = _hash_module_parameters(net.dope_model)
    assert after_hash != before_hash


@pytest.mark.parametrize("width", [64, 128, 256, 512])
def test_wider_blockformer_proposer_widths_and_step0_anchor(width: int) -> None:
    """Verify WiderBlockFormerProposer supports widths 64/128/256/512 and matches base at step 0."""
    model = WiderBlockFormerProposer(width=width, freeze_trunk=(width == 128))
    b_sz, k_cand = 4, 16
    tok = torch.randn(b_sz, 10, 32)
    ctx_s = torch.randn(b_sz, 29)
    anc_s = torch.randn(b_sz, 24)
    cand_f = torch.randn(b_sz, k_cand, 42)
    anc_f = cand_f[:, :4, :]
    base_scores = torch.randn(b_sz, k_cand)

    out = model(tok, ctx_s, anc_s, cand_f, anc_f, base_scores)
    assert out.shape == (b_sz, k_cand)
    # Zero-initialized readout head means step-0 output equals base_scores exactly
    assert torch.allclose(out, base_scores, atol=1e-7)

    # Check backward pass updates trainable parameters
    costs = torch.rand(b_sz, k_cand)
    gates = torch.ones(b_sz, k_cand)
    loss = compute_proposer_ranking_loss(out, costs, gates)
    loss.backward()
    assert model.head.weight.grad is not None


def test_local_lru_cache_eviction_and_bounds(tmp_path: Path) -> None:
    """Verify LocalLRUCache enforces max_bytes with oldest-first LRU eviction."""
    cache = LocalLRUCache(cache_dir=tmp_path / "lru", max_bytes=1000)
    cache.put_bytes("a.bin", b"x" * 400)
    cache.put_bytes("b.bin", b"y" * 400)
    assert cache.current_bytes() == 800

    # Touch a.bin so b.bin becomes oldest
    assert cache.get_bytes("a.bin") == b"x" * 400
    # Writing 400 bytes requires evicting b.bin
    cache.put_bytes("c.bin", b"z" * 400)
    assert cache.current_bytes() <= 1000
    assert cache.get_bytes("a.bin") is not None
    assert cache.get_bytes("c.bin") is not None
    assert cache.get_bytes("b.bin") is None

    with pytest.raises(ValueError, match="Refusing traversal"):
        cache.put_bytes("../escape.bin", b"bad")


def test_thread_rule_b_numeric_cap() -> None:
    """Verify thread rule (b) sets numeric threads <= 4 and exempts CUDA runtime threads."""
    info = enforce_numeric_thread_limits(MAX_NUMERIC_THREADS)
    assert info["numeric_thread_cap"] == 4
    assert info["cuda_driver_helper_threads_exempt"] is True
    ver = verify_numeric_thread_rule_b(MAX_NUMERIC_THREADS)
    assert ver["ok"] is True
    assert ver["cuda_runtime_threads_exempt"] is True


def test_candidate_subset_selection_and_analysis_gates() -> None:
    """Verify 8..16 candidate subset selection, 91/91 parity check, and statistical helpers."""
    idx = select_candidate_subset_indices(n_total=1152, k_candidates=16, seed=204101, context_idx=3)
    assert idx.shape == (16,)
    assert list(idx[:4]) == [0, 1, 2, 3]
    assert len(set(idx.tolist())) == 16

    exact_costs = np.full(DL_VAL_EXPECTED_CONTEXTS, R882_EXPECTED_DL_VAL_MEAN_COST, dtype=np.float64)
    parity = verify_r882_native_parity(exact_costs)
    assert parity["passed"] is True

    pick, cost = first_argmin_cost(np.array([0.5, -0.2, -0.2, 0.1]), np.array([1.0, 0.4, 0.9, 0.8]))
    assert pick == 1
    assert cost == pytest.approx(0.4)

    fams = [f"fam_{i // 2}" for i in range(40)]
    folds = assign_family_folds(fams, n_folds=5)
    assert set(folds.values()) == {0, 1, 2, 3, 4}

    base = np.ones(40, dtype=np.float64)
    cand = np.full(40, 0.98, dtype=np.float64)
    boot = paired_family_bootstrap(base, cand, fams, n_boot=500, seed=204141)
    assert boot["mean_gain"] == pytest.approx(0.02)
    assert boot["ci95_low"] > 0.0

    sel = evaluate_h178_selector(base, cand, fams, seed=204141)
    assert sel["admitted"] is True

    holm = holm_bonferroni({"a": 0.01, "b": 0.04, "c": 0.20})
    assert holm["a"] == pytest.approx(0.03)
    assert holm["b"] == pytest.approx(0.08)
    assert holm["c"] == pytest.approx(0.20)

    gate = evaluate_arm_gate(
        seed_results=[{"dl_val": boot, "fallback_rate": 0.0} for _ in range(3)],
        control_best_gain=0.001,
    )
    assert gate["passed"] is True
