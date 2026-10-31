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
