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


def test_repeated_rejections_never_mutate_saved_optimizer_state():
    model, opt = setup()
    original = copy.deepcopy(opt.state_dict())
    gate = DevIncumbent(model, opt, {"mean": 2}, "fixed-dev", "best.pt", 1)
    for rnd in range(2, 5):
        opt.zero_grad()
        model(torch.ones(1, 1)).sum().backward()
        opt.step()
        gate.decide(model, opt, {"mean": 1}, "fixed-dev", f"bad-{rnd}.pt", rnd)
        for key, state in original["state"].items():
            for name, value in state.items():
                assert torch.equal(opt.state_dict()["state"][key][name], value)
                assert torch.equal(gate.optimizer_state["state"][key][name], value)
        assert opt.param_groups[0]["lr"] == original["param_groups"][0]["lr"]


def test_rejection_escape_changes_lr_steps_seed_and_enables_gp_after_k():
    from dev_incumbent import proposal_plan

    history = [
        {
            "round": n,
            "adapter_admission": {"accepted": False},
            "candidate_dev_utility": {"mean": 1, "selected_programs": ["same"]},
        }
        for n in range(1, 4)
    ]
    plan = proposal_plan(42, 4, history)
    assert plan["steps"] == 1 and plan["lr_factor"] == 0.25
    assert plan["same_candidate_streak"] == 3 and plan["gp_exploration"]
    assert plan["gp_max_evaluated_per_context"] == 4
    history.append(
        {
            "round": 4,
            "adapter_admission": {"accepted": False},
            "candidate_dev_utility": {"mean": 1, "selected_programs": ["same"]},
            "proposal_plan": plan,
        }
    )
    retry = proposal_plan(42, 5, history)
    assert (
        retry["lr_factor"] < plan["lr_factor"]
        and retry["round_seed"] != plan["round_seed"]
    )


def test_acceptance_resets_retry_but_round_seed_keeps_changing():
    from dev_incumbent import proposal_plan

    history = [{"round": 3, "adapter_admission": {"accepted": True}}]
    a = proposal_plan(42, 4, history)
    b = proposal_plan(42, 5, history)
    assert a["retry_level"] == 0 and a["steps"] == 2 and not a["gp_exploration"]
    assert a["round_seed"] != b["round_seed"]


def test_stalled_gp_selection_is_reproducible_and_reserves_exploration():
    from dev_incumbent import gp_selection_indices

    costs = list(range(16))
    a = gp_selection_indices(costs, 42, True)
    assert a[:2] == [0, 1] and len(a) == len(set(a)) == 4
    assert a == gp_selection_indices(costs, 42, True)
    assert a != gp_selection_indices(costs, 43, True)
    assert gp_selection_indices(costs, 42, False) == [0, 1, 2, 3]


def test_margin_projection_preserves_ranking_and_allows_real_progress():
    from dev_incumbent import project_score_margins

    p = torch.nn.Parameter(torch.tensor([3.5e-5, 0.0]))
    frozen = torch.nn.Parameter(torch.tensor([7.0]), requires_grad=False)
    opt = torch.optim.AdamW([p], lr=1e-3, weight_decay=0)
    for _ in range(3):
        opt.zero_grad()
        loss = p[0] - p[1]
        loss.backward()
        opt.step()
        assert p[0] < 0  # reproduces raw winner flip before projection
        report = project_score_margins([p], [("near_tie", lambda: p[0], 1.75e-5)])
        assert report["satisfied"] and p[0] >= 1.75e-5
    assert p[1] > 0.002 and frozen.item() == 7
    assert int(opt.state[p]["step"]) == 3


def test_projection_does_not_move_a_satisfied_model():
    from dev_incumbent import project_score_margins

    p = torch.nn.Parameter(torch.tensor([0.1, 0.2]))
    before = p.detach().clone()
    report = project_score_margins([p], [("safe", lambda: p[0], 0.01)])
    assert report["satisfied"] and not report["corrections"]
    assert torch.equal(p, before)


def test_projection_refuses_frozen_parameters():
    from dev_incumbent import project_score_margins

    p = torch.nn.Parameter(torch.tensor([0.1]), requires_grad=False)
    with pytest.raises(ValueError):
        project_score_margins([p], [])
