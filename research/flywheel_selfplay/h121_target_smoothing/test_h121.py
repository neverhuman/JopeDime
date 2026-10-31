import pytest
import torch
from runner import (
    ARMS,
    SEEDS,
    UPDATES,
    cpu_dry_run,
    family_folds,
    mean_regret,
    scratch_path,
    smooth_targets,
)


def test_smoothing_formula_and_rowwise_mean():
    x = torch.tensor([[1.0, 3.0, 5.0], [2.0, 2.0, 8.0]])
    y = smooth_targets(x, 0.3)
    assert torch.allclose(y, 0.7 * x + 0.3 * x.mean(1, keepdim=True))
    assert torch.allclose(y.mean(1), x.mean(1))
    assert torch.equal(smooth_targets(x, 0), x)


@pytest.mark.parametrize("bad", [-0.1, 1.0, float("nan")])
def test_invalid_epsilon(bad):
    with pytest.raises(ValueError):
        smooth_targets(torch.ones(2, 3), bad)


def test_nonfinite_rejected():
    with pytest.raises(ValueError):
        smooth_targets(torch.tensor([[1.0, float("nan")]]), 0.1)


def test_family_split_is_deterministic_and_disjoint():
    rows = [
        {"dl_split": "DL-Train", "task": "regression", "family_id": f"f{i}"}
        for i in range(25)
    ]
    a = family_folds(rows)
    assert a == family_folds(rows)
    assert not set(a["train_families"]) & set(a["score_families"])


def test_non_train_rejected():
    with pytest.raises(ValueError):
        family_folds(
            [{"dl_split": "DL-Val", "task": "regression", "family_id": "f"}] * 5
        )


def test_contract_and_dry_run():
    assert (
        ARMS == ("CONTROL", "TS-0.1", "TS-0.3")
        and SEEDS == (45, 46, 47)
        and UPDATES == 40
    )
    assert cpu_dry_run()["real_checkpoint_or_data_accessed"] is False


def test_mean_regret_all_candidates():
    t = torch.tensor([[1.0, 3.0, 5.0]])
    assert torch.allclose(mean_regret(t, t), torch.tensor([2.0]))
    p = torch.tensor([[5.0, 3.0, 1.0]])
    assert torch.allclose(mean_regret(p, t), torch.tensor([2.0]))


def test_unique_attempt_scratch():
    root = __import__("pathlib").Path("scratch")
    assert scratch_path(root, "TS-0.1", 45, 1) != scratch_path(root, "TS-0.1", 45, 2)
    with pytest.raises(ValueError):
        scratch_path(root, "CONTROL", 44, 1)


def _synthetic_cells(gain_01=0.002, gain_03=0.003):
    from runner import DOPE_SHA256, R882_SHA256

    families = [f"family-{i}" for i in range(8)]
    reference = {family: i / 100 for i, family in enumerate(families)}
    cells = []
    for seed in SEEDS:
        for arm, gain in (("CONTROL", 0.0), ("TS-0.1", gain_01), ("TS-0.3", gain_03)):
            dev_families = {family: reference[family] + gain for family in families}
            dev = {
                "per_family": dev_families,
                "per_family_mean_regret_all_candidates": {
                    family: 0.3 for family in families
                },
                "mean_regret_all_candidates": 0.3,
            }
            cell = {
                "wave": "H121",
                "arm": arm,
                "seed": seed,
                "updates": 40,
                "dope_before": DOPE_SHA256,
                "dope_after": DOPE_SHA256,
                "checkpoint_sha256": R882_SHA256,
                "prereg_sha256": "synthetic-prereg",
                "fold_assignment_sha256": "synthetic-fold",
                "folds": [["f0"], ["f1"]],
                "initial_adapter_sha256": f"seed-{seed}-same-initial",
                "dev": dev,
                "r882_dev": {"per_family": reference, "mean_regret_all_candidates": 0.4}
                if arm == "CONTROL"
                else None,
            }
            cells.append(cell)
    return cells


def test_family_bootstrap_holm_elo_and_regret_summary():
    from runner import summarize

    result = summarize(_synthetic_cells(), draws=1000)
    assert result["arms"]["TS-0.1"]["admitted"] is True
    assert result["arms"]["TS-0.3"]["admitted"] is True
    assert torch.allclose(
        torch.tensor(
            result["arms"]["TS-0.1"]["mean_regret_all_candidates_vs_r882_per_seed"]
        ),
        torch.tensor([0.1] * 3),
    )
    assert result["arms"]["TS-0.1"]["family_grouped_elo_ci95_vs_r882_per_seed"][45] == [
        "Infinity",
        "Infinity",
    ]
    assert result["auc"] is None


def test_family_summary_ties_fail_and_incomplete_cells_reject():
    from runner import summarize

    tied = summarize(_synthetic_cells(0.0, 0.0), draws=200)
    assert tied["arms"]["TS-0.1"]["admitted"] is False
    with pytest.raises(ValueError):
        summarize(_synthetic_cells()[:-1], draws=50)


def test_h121_wave_dry_run_has_no_launch_side_effects():
    import json
    import subprocess
    from pathlib import Path

    root = Path.home() / "jope-data/h121/h121-wave"
    assert not root.exists()
    proc = subprocess.run(
        [
            "/usr/bin/python3",
            "research/flywheel_selfplay/h121_target_smoothing/run_wave.py",
            "--cpu-dry-run",
        ],
        check=True,
        capture_output=True,
        text=True,
        env={
            **__import__("os").environ,
            "OMP_NUM_THREADS": "4",
            "MKL_NUM_THREADS": "4",
            "OPENBLAS_NUM_THREADS": "1",
        },
    )
    payload = json.loads(proc.stdout)
    assert payload["fifo_ticket_created"] is False
    assert payload["run_root_created"] is False
    assert payload["authorization_required"] is True
    assert not root.exists()


def test_fifo_predecessor_closeout_gate_synthetic_records():
    import importlib.util
    from pathlib import Path
    import sys

    path = Path(__file__).with_name("run_wave.py")
    spec = importlib.util.spec_from_file_location("h121_test_run_wave", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    budget = {
        "attempts": [{"status": "complete"}, {"status": "complete"}],
        "in_progress": None,
    }
    assert module.closeout_records_complete("H99", budget, {"status": "complete"})
    assert module.closeout_records_complete("H100", budget, {"complete": True})
    assert not module.closeout_records_complete(
        "H99", budget, {"status": "complete"}, active_ticket=True
    )
    assert not module.closeout_records_complete(
        "H100", budget, {"complete": True}, owner_marker=True
    )
    assert not module.closeout_records_complete(
        "H99", {**budget, "in_progress": {"cell": "live"}}, {"status": "complete"}
    )
    assert not module.closeout_records_complete(
        "H99", {**budget, "attempts": [{"status": "void"}]}, {"status": "complete"}
    )
    retry_budget = {
        **budget,
        "attempts": [
            {"status": "void", "error_kind": "host_load", "outcome_started": False},
            {"status": "complete"},
        ],
    }
    assert module.closeout_records_complete("H100", retry_budget, {"complete": True})
    assert not module.closeout_records_complete(
        "H100",
        {
            **retry_budget,
            "attempts": [
                {"status": "void", "error_kind": "host_load", "outcome_started": True}
            ],
        },
        {"complete": True},
    )


def test_owner_authorization_is_exact_source_bound():
    import importlib.util
    from pathlib import Path
    import sys

    path = Path(__file__).with_name("run_wave.py")
    spec = importlib.util.spec_from_file_location("h121_test_auth_wave", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    good = {
        "schema": "jope.h121.gpu-authorization/v1",
        "wave": "H121",
        "approved": True,
        "owner": "authorized-owner",
        "source_commit": "abc123",
    }
    module.validate_authorization_payload(good, "abc123")
    with pytest.raises(RuntimeError):
        module.validate_authorization_payload(good, "different-head")
    with pytest.raises(RuntimeError):
        module.validate_authorization_payload({**good, "approved": False}, "abc123")


def test_scratch_budget_refuses_over_limit_without_deleting():
    from pathlib import Path
    import tempfile

    from run_wave import require_scratch_budget

    with tempfile.TemporaryDirectory() as scratch_dir:
        root = Path(scratch_dir)
        scratch = root / "scratch"
        scratch.mkdir()
        file = scratch / "fixture.bin"
        file.write_bytes(b"x" * 10)
        assert require_scratch_budget(root, limit_bytes=10) == 10
        with pytest.raises(RuntimeError):
            require_scratch_budget(root, limit_bytes=9)
        assert file.read_bytes() == b"x" * 10
