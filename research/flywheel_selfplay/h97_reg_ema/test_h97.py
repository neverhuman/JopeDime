"""CPU-only protocol controls for the preregistered H97 candidate study."""

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

HERE = Path(__file__).resolve().parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


r = load("test_h97_runner", HERE / "runner.py")
w = load("test_h97_wave", HERE / "run_wave.py")


def test_fresh_seeds_arms_and_fixed_update_plan():
    assert r.SEEDS == (33, 34, 35)
    assert r.ARMS == ("CONTROL", "WD", "DROP", "EMA")
    assert not set(r.SEEDS) & set(range(12, 33))
    batches = [
        SimpleNamespace(round_index=round_index, step_index=step)
        for round_index in range(1, 21)
        for step in range(8)
    ]
    for arm in r.ARMS:
        plan = r.make_update_plan(batches, arm)
        assert len(plan) == 40
        assert all(len(update) == 1 for update in plan)
        assert all(update[0].step_index < 2 for update in plan)


def test_hidden_dropout_is_enabled_only_when_requested_and_hooks_are_removed():
    adapter = torch.nn.Module()
    adapter.anchor = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.SiLU())
    adapter.hidden = torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.SiLU())
    dropout = r.HiddenActivationDropout(adapter)
    assert len(dropout.handles) == 1
    x = torch.ones((4, 8))
    torch.manual_seed(1)
    train_a = adapter.hidden(x)
    torch.manual_seed(2)
    train_b = adapter.hidden(x)
    assert not torch.equal(train_a, train_b)
    dropout.enabled = False
    torch.manual_seed(1)
    eval_a = adapter.hidden(x)
    torch.manual_seed(2)
    eval_b = adapter.hidden(x)
    assert torch.equal(eval_a, eval_b)
    dropout.close()
    assert not dropout.handles


def test_ema_uses_exact_decay_over_only_trainable_named_parameters():
    net = torch.nn.Module()
    net.adapter_10tok = torch.nn.Linear(1, 1, bias=False)
    net.adapter_10tok.weight.data.fill_(2.0)
    net.other = torch.nn.Parameter(torch.tensor([9.0]), requires_grad=False)
    shadow = {"adapter_10tok.weight": torch.tensor([[0.0]])}
    r.ema_update(shadow, net)
    assert shadow["adapter_10tok.weight"].item() == pytest.approx(0.2)
    net.adapter_10tok.weight.data.fill_(4.0)
    r.ema_update(shadow, net)
    assert shadow["adapter_10tok.weight"].item() == pytest.approx(0.58)
    with pytest.raises(ValueError, match="fixed"):
        r.ema_update(shadow, net, decay=0.8)


def test_stage_path_is_unique_per_cell_attempt_and_rejects_reuse(tmp_path):
    first = r.attempt_scratch_path(tmp_path, "DROP-seed33", 1)
    second = r.attempt_scratch_path(tmp_path, "DROP-seed33", 2)
    assert first != second
    first.mkdir(parents=True)
    with pytest.raises(RuntimeError, match="already exists"):
        r.attempt_scratch_path(tmp_path, "DROP-seed33", 1)
    for cell, attempt in (("../bad", 1), ("DROP", 0), ("DROP", True)):
        with pytest.raises(ValueError):
            r.attempt_scratch_path(tmp_path, cell, attempt)


def _cell(arm, seed, *, delta=0.002):
    contexts = [f"c{i:02}" for i in range(91)]
    families = [f"f{i:02}" for i in range(19)]
    per_context = {context: 0.1 + delta for context in contexts}
    per_family = {family: 0.1 + delta for family in families}
    baseline_family = {family: 0.1 for family in families}
    top4 = {context: [4, 5, 6, 7] for context in contexts}
    result = {
        "wave": "H97",
        "arm": arm,
        "seed": seed,
        "timing_only": False,
        "trainable_count": 29211,
        "updates": 40,
        "dope_before": r.h59.DOPE_SHA,
        "dope_after": r.h59.DOPE_SHA,
        "checkpoint_sha256": r.h59.R882_SHA,
        "manifest_sha256": r.h59.MANIFEST_SHA,
        "selected_sha256": r.SELECTED_SHA,
        "initial_adapter_sha256": "initial",
        "final_adapter_sha256": f"final-{arm}-{seed}",
        "source_commit": "source",
        "prereg_sha256": "prereg",
        "folds": [["family-a"], ["family-b"]],
        "adapter_l2_from_r882": 0.2,
        "train": {"mean": 0.12},
        "dev": {"mean": 0.102, "per_context": per_context, "per_family": per_family},
        "dev_top4_indices": top4,
        "r882_top4_indices": top4,
        "r882_dev": {"per_family": baseline_family},
    }
    if arm == "CONTROL":
        result["dev"]["per_family"] = baseline_family
        result["dev"]["mean"] = 0.1
    return result


def test_summary_uses_95000_and_strict_lcb_admission():
    cells = [_cell(arm, seed) for seed in r.SEEDS for arm in r.ARMS]
    summary = r.summarize(cells)
    assert summary["schema"] == "jope.h97.metrics/v1"
    for arm in ("WD", "DROP", "EMA"):
        assert summary["arms"][arm]["lcb95"] > r.MARGIN
        assert summary["arms"][arm]["admitted"] is True
    assert summary["arms"]["CONTROL"]["admitted"] is False
    with pytest.raises(RuntimeError, match="twelve"):
        r.summarize(cells[:-1])
    tied = [_cell(arm, seed, delta=r.MARGIN) for seed in r.SEEDS for arm in r.ARMS]
    tie_summary = r.summarize(tied)
    assert abs(tie_summary["arms"]["WD"]["lcb95"] - r.MARGIN) <= r.LCB_TIE_TOL
    assert tie_summary["arms"]["WD"]["admitted"] is False


def test_h93_live_ticket_is_required_for_fifo_barrier(tmp_path):
    alive = {123}
    with pytest.raises(RuntimeError, match="H93"):
        w.validate_launch_barrier(tmp_path, is_alive_fn=lambda pid: pid in alive)
    (tmp_path / "1-123-H90").write_text("epoch_ns=1\npid=123\nwave=H90\n")
    with pytest.raises(RuntimeError, match="H93"):
        w.validate_launch_barrier(tmp_path, is_alive_fn=lambda pid: pid in alive)
    (tmp_path / "2-123-H93").write_text("epoch_ns=2\npid=123\nwave=H93\n")
    assert (
        w.validate_launch_barrier(tmp_path, is_alive_fn=lambda pid: pid in alive)[
            "h93_ticket"
        ]
        == "2-123-H93"
    )


def test_run_root_is_single_h97_path_and_excludes_prior_wave_results(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(w.Path, "home", lambda: tmp_path)
    expected = tmp_path / "jope-data/h97/h97-wave"
    w.validate_run_root(expected)
    for blocked in ("h76/h76-wave", "h82/h82-wave", "h93/h93-wave"):
        with pytest.raises(RuntimeError):
            w.validate_run_root(tmp_path / "jope-data" / blocked)
    with pytest.raises(RuntimeError, match="authorized run root"):
        w.validate_run_root(tmp_path / "jope-data/h97/other-run")


def test_cpu_dry_run_reads_no_dev_and_creates_no_run_root(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr(w.Path, "home", lambda: tmp_path)
    root = tmp_path / "jope-data/h97/h97-wave"
    monkeypatch.setattr(
        sys, "argv", ["run_wave.py", "--cpu-dry-run", "--root", str(root)]
    )
    w.main()
    result = json.loads(capsys.readouterr().out)
    assert result["wave"] == "H97"
    assert result["arms"] == list(r.ARMS) and result["seeds"] == list(r.SEEDS)
    assert result["reads_dev"] is False
    assert result["created_fifo_ticket"] is False
    assert not root.exists()
