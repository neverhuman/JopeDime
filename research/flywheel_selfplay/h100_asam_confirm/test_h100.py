"""Synthetic CPU proof only; no corpus outcomes or CUDA qualification."""

import copy
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


HERE = Path(__file__).parent
r = load("h100_test_runner", HERE / "runner.py")
w = load("h100_test_coordinator", HERE / "run_wave.py")


@pytest.mark.parametrize(
    "arm,old", [("CONTROL", "CONTROL"), ("DRIFT_MATCH", "CONTROL"), ("ASAM05", "ASAM")]
)
@pytest.mark.parametrize("seed", [3, 11])
def test_h82_optimizer_equivalence_with_stochastic_closure_and_moments(arm, old, seed):
    def execute(step, label):
        weight = torch.nn.Parameter(torch.tensor([0.2, -0.3]))
        bias = torch.nn.Parameter(torch.tensor([0.1]))
        named = [("layer.weight", weight), ("layer.bias", bias)]
        opt = torch.optim.AdamW(
            [weight, bias], lr=r.learning_rate(arm), weight_decay=1e-4
        )
        losses = []
        for update in range(3):
            torch.manual_seed(seed + update)
            losses.append(
                step(
                    named,
                    opt,
                    lambda: ((weight * torch.rand_like(weight) + bias) ** 2).sum(),
                    label,
                )
            )
        return weight.detach(), bias.detach(), opt.state_dict(), losses

    a, b = execute(r.optimizer_step, arm), execute(r.legacy.optimizer_step, old)
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1]) and a[3] == b[3]
    for key in a[2]["state"]:
        for field, value in a[2]["state"][key].items():
            assert torch.equal(value, b[2]["state"][key][field])


def test_rho_one_doubles_perturbation_and_exception_restores_parameters():
    perturbations = []
    for arm in ("ASAM05", "ASAM10"):
        p = torch.nn.Parameter(torch.tensor([0.2, -0.3]))
        original = p.detach().clone()
        opt = torch.optim.AdamW([p], lr=1e-4)
        calls = 0

        def closure():
            nonlocal calls
            calls += 1
            if calls == 2:
                perturbations.append(p.detach().clone() - original)
                raise RuntimeError("synthetic second-pass failure")
            return p.square().sum()

        with pytest.raises(RuntimeError, match="second-pass"):
            r.optimizer_step([("x.weight", p)], opt, closure, arm)
        assert torch.equal(p, original) and not opt.state
    assert torch.allclose(perturbations[1], 2 * perturbations[0], atol=1e-7, rtol=0)


def test_readonly_per_attempt_staging_reuses_known_copy_and_preserves_source(tmp_path):
    source = tmp_path / "source.pt"
    source.write_bytes(b"synthetic immutable checkpoint")
    source.chmod(0o444)
    digest = r.h59.sha256(source)
    scratch = tmp_path / "scratch/CONTROL-seed39-attempt1"
    p = r.stage_copy(source, scratch, expected_sha=digest)
    inode = p.stat().st_ino
    assert r.stage_copy(source, scratch, expected_sha=digest).stat().st_ino == inode
    assert p.stat().st_mode & 0o222 == 0 and source.stat().st_mode & 0o222 == 0
    assert (
        p.read_bytes() == source.read_bytes()
        and p.stat().st_ino != source.stat().st_ino
    )
    p.chmod(0o644)
    p.write_bytes(b"unknown")
    with pytest.raises(RuntimeError, match="unknown"):
        r.stage_copy(source, scratch, expected_sha=digest)
    assert p.read_bytes() == b"unknown" and r.h59.sha256(source) == digest


def test_staging_rejects_symlink_and_hardlink_source_aliases(tmp_path):
    source = tmp_path / "source.pt"
    source.write_bytes(b"immutable")
    digest = r.h59.sha256(source)
    (tmp_path / "scratch").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(RuntimeError, match="directory"):
        r.stage_copy(source, tmp_path / "scratch/cell", expected_sha=digest)
    (tmp_path / "scratch").unlink()
    scratch = tmp_path / "scratch/cell"
    scratch.mkdir(parents=True)
    (scratch / "r882_adapter_copy.pt").hardlink_to(source)
    with pytest.raises(RuntimeError, match="hardlinks"):
        r.stage_copy(source, scratch, expected_sha=digest)
    assert source.stat().st_mode & 0o222


def test_l2_and_historical_lr_ratio_are_fixed():
    assert r.adapter_l2({"x": torch.tensor([3.0, 4.0])}, {"x": torch.zeros(2)}) == 5
    c = json.loads((HERE / "calibration.json").read_text())
    assert c["lr_scale"] == r.LR_SCALE == c["mean_l2"]["ASAM"] / c["mean_l2"]["CONTROL"]
    assert c["drift_match_lr"] == r.DRIFT_LR == 1e-4 * r.LR_SCALE
    assert len(c["rows"]) == 6 and c["dev_payloads_opened"] == 0
    with pytest.raises(RuntimeError, match="inventory"):
        r.adapter_l2({"x": torch.ones(1)}, {"y": torch.ones(1)})


def test_top_four_excludes_revealed_anchors_and_uses_stable_ascending_ties():
    scores = {"a": np.array([-100.0, -99.0, -98.0, -97.0, 2.0, 1.0, 1.0, 0.0, 3.0])}
    assert r.top4_indices(scores)["a"] == [7, 5, 6, 4]
    result = r.overlap_readout(
        {"a": [4, 5, 6, 7], "b": [4, 5, 6, 7]},
        {"a": [4, 5, 8, 9], "b": [4, 5, 6, 7]},
        {"a": "f1", "b": "f2"},
    )
    assert result["mean"] == 0.75
    with pytest.raises(RuntimeError, match="revealed"):
        r.overlap_readout({"a": [0, 5, 6, 7]}, {"a": [4, 5, 6, 7]}, {"a": "f"})


def test_holm_blocks_nominal_hits_and_exact_boundary_ties():
    assert not any(
        x["passes"] for x in r.holm({"a": 0.03, "b": 0.04, "c": 0.2}).values()
    )
    assert not r.holm({"a": 0.05 / 3, "b": 0.2, "c": 0.3})["a"]["passes"]
    passed = r.holm({"a": 0.001, "b": 0.01, "c": 0.049})
    assert all(x["passes"] for x in passed.values())
    assert passed["a"]["adjusted"] == 0.003


def fixture_cells(harmed=0):
    families = [f"dev-family-{i:02}" for i in range(19)]
    context_families = {f"dev-{i:03}": families[i % 19] for i in range(91)}
    top = {cid: [4, 5, 6, 7] for cid in context_families}

    def packet(values, train=False):
        if train:
            return {
                "mean": 0.6,
                "per_family": {f"train-family-{i}": 0.6 for i in range(205)},
                "per_context": {f"train-{i}": 0.6 for i in range(1219)},
            }
        return {
            "mean": float(np.mean(list(values.values()))),
            "per_family": values,
            "per_context": {cid: values[f] for cid, f in context_families.items()},
        }

    cells = []
    for seed in r.SEEDS:
        for arm in r.ARMS:
            values = {f: 0.5 + (0 if arm == "CONTROL" else 0.05) for f in families}
            if arm == "ASAM10":
                for f in families[:harmed]:
                    values[f] = 0.48
            overlap = r.overlap_readout(top, top, context_families)
            cells.append(
                {
                    "wave": "H100",
                    "arm": arm,
                    "seed": seed,
                    "timing_only": False,
                    "trainable_count": 29211,
                    "updates": 40,
                    "backward_passes": 80 if arm in r.RHO else 40,
                    "dope_before": r.h59.DOPE_SHA,
                    "dope_after": r.h59.DOPE_SHA,
                    "checkpoint_sha256": r.h59.R882_SHA,
                    "manifest_sha256": r.h59.MANIFEST_SHA,
                    "selected_sha256": r.SELECTED_SHA,
                    "calibration_sha256": r.h59.sha256(HERE / "calibration.json"),
                    "learning_rate": r.learning_rate(arm),
                    "rho": r.RHO.get(arm),
                    "initial_adapter_sha256": "fixture-initial",
                    "final_adapter_sha256": "fixture-changed",
                    "adapter_l2": 1.0,
                    "holdout_payloads_opened": 0,
                    "outer_rows_delivered": 0,
                    "source_commit": "synthetic-fixture",
                    "prereg_sha256": "synthetic-prereg",
                    "folds": [],
                    "source_hashes": {"source": "fixture"},
                    "train": packet({}, True),
                    "dev": packet(values),
                    "r882_dev": packet(dict.fromkeys(families, 0.5)),
                    "top4": {
                        **overlap,
                        "candidate": top,
                        "r882": top,
                        "families": context_families,
                    },
                }
            )
    return cells


def test_full_panel_harm_gate_blocks_two_losing_families_despite_gain_and_holm():
    result = r.summarize(fixture_cells(harmed=2))
    arm = result["arms"]["ASAM10"]
    assert arm["raw_gate"] and result["holm"]["ASAM10"]["passes"]
    assert arm["harm_fraction"] == 2 / 19 and not arm["admitted"]
    assert result["arms"]["ASAM05"]["admitted"]
    assert (
        result["drift_interpretation"]["equivalence"]
        and result["drift_interpretation"]["l2_match"]
    )
    one = r.summarize(fixture_cells(harmed=1))
    assert one["arms"]["ASAM10"]["admitted"]
    assert result["arms"]["CONTROL"]["elo_vs_control"]["median_elo"] == 1500
    assert result["arms"]["ASAM05"]["elo_vs_r882"]["median_elo"] == "Infinity"


@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "duplicate",
        "timing",
        "nonfinite",
        "source",
        "reference",
        "roster",
        "lr",
        "protected",
    ],
)
def test_summary_refuses_incomplete_or_changed_identity(mutation):
    cells = copy.deepcopy(fixture_cells())
    if mutation == "missing":
        cells.pop()
    elif mutation == "duplicate":
        cells[-1] = cells[0]
    elif mutation == "timing":
        cells[0]["timing_only"] = True
    elif mutation == "nonfinite":
        cells[0]["dev"]["per_family"]["dev-family-00"] = float("nan")
    elif mutation == "source":
        cells[0]["source_commit"] = "changed"
    elif mutation == "reference":
        cells[0]["r882_dev"]["mean"] += 0.1
    elif mutation == "roster":
        cells[0]["train"]["per_context"].popitem()
    elif mutation == "lr":
        cells[0]["learning_rate"] *= 2
    else:
        cells[0]["outer_rows_delivered"] = 1
    with pytest.raises(RuntimeError):
        r.summarize(cells)


def closed_metric(_):
    return {
        "schema": "jope." + _.lower() + ".metrics/v1",
        "complete": True,
        "ledger": {"in_progress": None, "attempts": [{"wall_seconds": 1}]},
    }


def test_precursor_closure_is_required_before_any_h100_ticket(tmp_path):
    assert set(
        w.require_predecessors(read_fn=closed_metric, queue_dir=tmp_path, active_fn=set)
    ) == {"H93", "H97", "H99"}
    for wave in ("H93", "H97", "H99"):
        with pytest.raises(w.PredecessorPending, match="active"):
            w.require_predecessors(
                read_fn=closed_metric, queue_dir=tmp_path, active_fn=lambda: {wave}
            )
        ticket = tmp_path / f"1-123-{wave}"
        ticket.write_text("epoch_ns=1\npid=123\nwave=" + wave + "\n")
        with pytest.raises(w.PredecessorPending, match="ticket"):
            w.require_predecessors(
                read_fn=closed_metric,
                queue_dir=tmp_path,
                active_fn=set,
                alive_fn=lambda _: True,
            )
        ticket.unlink()

    def cpu_plan(wave):
        data = closed_metric(wave)
        data["schema"] = "jope.h99.cpu-preparation/v1"
        return data

    with pytest.raises(w.PredecessorPending, match="closed-ledger"):
        w.require_predecessors(read_fn=cpu_plan, queue_dir=tmp_path, active_fn=set)


def test_budget_rejects_unknown_nonfinite_and_exhausted_charges_and_no_retries():
    assert (
        w.remaining(
            {
                "attempts": [{"wall_seconds": 2600, "wait_seconds": 0}],
                "in_progress": None,
            }
        )
        == 100
    )
    for ledger in (
        {"attempts": [], "in_progress": {}},
        {"attempts": [{"wall_seconds": float("nan"), "wait_seconds": 0}]},
        {"attempts": [{"wall_seconds": 2695, "wait_seconds": 0}]},
    ):
        with pytest.raises(RuntimeError):
            w.remaining(ledger)
    assert w.MAX_RETRIES == 0
    calls = []

    def abort(*args):
        calls.append(args)
        raise w.HostLoadAbort(86)

    with pytest.raises(w.HostLoadAbort):
        w.execute(None, {}, "ASAM10", 39, attempt_fn=abort)
    assert len(calls) == 1 and calls[0][-1] == 1
