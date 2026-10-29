import copy
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))
from dev_incumbent import DevIncumbent


def setup():
    model = torch.nn.Linear(1, 1)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    model(torch.ones(1, 1)).sum().backward()
    opt.step()
    opt.zero_grad()
    return model, opt


def test_worse_restores_both_weights_and_optimizer():
    model, opt = setup()
    saved = copy.deepcopy(model.state_dict())
    saved_opt = copy.deepcopy(opt.state_dict())
    gate = DevIncumbent(model, opt, {"mean": 2.2678}, "fixed-dev", "round-40.pt", 40)
    model(torch.ones(1, 1)).sum().backward()
    opt.step()
    event = gate.decide(model, opt, {"mean": 1.148}, "fixed-dev", "round-41.pt", 41)
    assert not event["accepted"] and event["best_checkpoint"] == "round-40.pt"
    for name, v in model.state_dict().items():
        assert torch.equal(v, saved[name])
    for key, state in opt.state_dict()["state"].items():
        for name, v in state.items():
            assert v == saved_opt["state"][key][name]


def test_improvement_kept_and_later_regression_restores_it():
    model, opt = setup()
    gate = DevIncumbent(model, opt, {"mean": 2}, "fixed-dev", "old.pt", 1)
    with torch.no_grad():
        model.weight.add_(1)
    assert gate.decide(model, opt, {"mean": 3}, "fixed-dev", "best.pt", 2)["accepted"]
    best = model.weight.detach().clone()
    with torch.no_grad():
        model.weight.add_(1)
    assert not gate.decide(model, opt, {"mean": 2.5}, "fixed-dev", "bad.pt", 3)[
        "accepted"
    ]
    assert torch.equal(model.weight, best) and gate.best_checkpoint == "best.pt"


def test_tie_advances_training_without_lowering_dev_utility():
    model, opt = setup()
    gate = DevIncumbent(model, opt, {"mean": 2}, "fixed-dev", "old.pt", 1)
    best = model.weight.detach().clone()
    with torch.no_grad():
        model.weight.add_(1)
    event = gate.decide(model, opt, {"mean": 2}, "fixed-dev", "tie.pt", 2)
    assert event["accepted"] and event["action"] == "accept_tie"
    assert not torch.equal(model.weight, best)


@pytest.mark.parametrize("score", [float("nan"), float("inf")])
def test_invalid_score_fails_closed(score):
    model, opt = setup()
    gate = DevIncumbent(model, opt, {"mean": 2}, "fixed-dev", "old.pt", 1)
    with pytest.raises(ValueError):
        gate.decide(model, opt, {"mean": score}, "fixed-dev", "bad.pt", 2)


def test_identity_change_refused():
    model, opt = setup()
    gate = DevIncumbent(model, opt, {"mean": 2}, "fixed-dev", "old.pt", 1)
    with pytest.raises(ValueError):
        gate.decide(model, opt, {"mean": 3}, "different-dev", "bad.pt", 2)
