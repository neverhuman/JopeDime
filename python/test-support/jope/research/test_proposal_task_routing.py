import ast
import inspect
from pathlib import Path

import pytest
import torch

from test_exact_dl_generate import (
    load_proposal_selection_policy,
    score_context_candidates,
)


class RoutedModel:
    def __init__(self):
        self.scored_task = None
        self.stability_tasks = None

    def score_token_sequences(self, embedding, target, sequences, *, device, task):
        self.scored_task = task
        rows = len(sequences)
        if rows == 2 and task == "binary":
            return torch.tensor([2.0, 1.0]), torch.zeros(rows)
        if rows == 2 and task == "regression":
            return torch.tensor([1.0, 2.0]), torch.zeros(rows)
        return torch.ones(rows), torch.zeros(rows)

    def reward_context(self, embedding, target):
        return torch.zeros((1, 4))

    def instability_logit(self, token_ids, pad, context, task_types):
        self.stability_tasks = list(task_types)
        return torch.zeros(token_ids.shape[0])


def proposals(task: str) -> list[dict]:
    return [
        {
            "task": task,
            "tokens": [1, 2],
            "_actions": {},
            "_head_op": "ridge",
        },
        {
            "task": task,
            "tokens": [1, 3],
            "_actions": {},
            "_head_op": "ridge",
        },
    ]


def test_proposal_reward_and_stability_heads_receive_the_required_task(monkeypatch):
    import jope.policy.decode
    import jope.policy.model_v3

    monkeypatch.setattr(
        jope.policy.decode, "candidate_risk_reason_v3", lambda _engine, _actions: None
    )
    monkeypatch.setattr(
        jope.policy.model_v3,
        "pad_sequences",
        lambda sequences, device: (
            torch.ones((len(sequences), 2), dtype=torch.long),
            torch.ones((len(sequences), 2), dtype=torch.bool),
        ),
    )
    model = RoutedModel()
    policy, _ = load_proposal_selection_policy()
    scored = score_context_candidates(
        model,
        torch.zeros(4),
        torch.zeros(4),
        proposals("binary"),
        task="binary",
        selection_policy=policy,
        device="cpu",
    )
    assert len(scored) == 2
    assert model.scored_task == "binary"
    assert model.stability_tasks == ["binary", "binary"]


def test_proposal_scoring_rejects_missing_or_mixed_task():
    parameters = inspect.signature(score_context_candidates).parameters
    assert parameters["task"].default is inspect.Parameter.empty
    with pytest.raises(ValueError, match="mixes tasks"):
        policy, _ = load_proposal_selection_policy()
        score_context_candidates(
            RoutedModel(),
            torch.zeros(4),
            torch.zeros(4),
            [*proposals("binary")[:1], *proposals("regression")[:1]],
            task="binary",
            selection_policy=policy,
            device="cpu",
        )


def test_deliberately_different_task_heads_rank_the_same_slots_differently(monkeypatch):
    import jope.policy.decode
    import jope.policy.model_v3

    monkeypatch.setattr(
        jope.policy.decode, "candidate_risk_reason_v3", lambda _engine, _actions: None
    )
    monkeypatch.setattr(
        jope.policy.model_v3,
        "pad_sequences",
        lambda sequences, device: (
            torch.ones((len(sequences), 2), dtype=torch.long),
            torch.ones((len(sequences), 2), dtype=torch.bool),
        ),
    )
    policy, _ = load_proposal_selection_policy()
    binary = score_context_candidates(
        RoutedModel(),
        torch.zeros(4),
        torch.zeros(4),
        proposals("binary"),
        task="binary",
        selection_policy=policy,
        device="cpu",
    )
    regression = score_context_candidates(
        RoutedModel(),
        torch.zeros(4),
        torch.zeros(4),
        proposals("regression"),
        task="regression",
        selection_policy=policy,
        device="cpu",
    )
    assert binary[0]["selection_score"] > binary[1]["selection_score"]
    assert regression[0]["selection_score"] < regression[1]["selection_score"]


def test_promotion_conditional_rescore_passes_the_context_task():
    path = Path(__file__).with_name("test_exact_canary_propose.py")
    tree = ast.parse(path.read_text())
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "score_token_sequences"
    ]
    assert len(calls) == 1
    task = next(keyword.value for keyword in calls[0].keywords if keyword.arg == "task")
    assert ast.unparse(task) == "key[0]"
