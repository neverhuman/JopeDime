"""Development-only adapter ratchet; preserve rejected proposals and restore optimizer.

The dev cohort is repeatedly observed for selection and cannot establish sealed-test
quality. Keep the comparison identity fixed and never consult the sealed meta-test.
"""

from __future__ import annotations

import copy
import math


class DevIncumbent:
    def __init__(
        self,
        model,
        optimizer,
        utility: dict,
        identity: str,
        checkpoint: str,
        round_idx: int,
    ):
        self.identity = identity
        self.best_utility = self._mean(utility)
        self.best_stats = copy.deepcopy(utility)
        self.best_checkpoint = checkpoint
        self.best_round = round_idx
        self.model_state = copy.deepcopy(model.state_dict())
        self.optimizer_state = copy.deepcopy(optimizer.state_dict())

    @staticmethod
    def _mean(utility):
        value = float(utility["mean"])
        if not math.isfinite(value):
            raise ValueError("non-finite development utility")
        return value

    def decide(
        self,
        model,
        optimizer,
        utility: dict,
        identity: str,
        checkpoint: str,
        round_idx: int,
    ):
        if identity != self.identity:
            raise ValueError("development evaluation identity changed")
        value = self._mean(utility)
        old = self.best_utility
        # Ties may advance training; only strictly worse dev utility is rolled back.
        accepted = value >= old
        if accepted:
            self.best_utility = value
            self.best_stats = copy.deepcopy(utility)
            self.best_checkpoint = checkpoint
            self.best_round = round_idx
            self.model_state = copy.deepcopy(model.state_dict())
            self.optimizer_state = copy.deepcopy(optimizer.state_dict())
        else:
            model.load_state_dict(self.model_state, strict=True)
            optimizer.load_state_dict(copy.deepcopy(self.optimizer_state))
        return {
            "round": round_idx,
            "evaluation_identity": identity,
            "candidate_utility": value,
            "previous_best_utility": old,
            "accepted": accepted,
            "action": "accept_improvement"
            if value > old
            else "accept_tie"
            if accepted
            else "restore_best_model_and_optimizer",
            "best_utility": self.best_utility,
            "best_round": self.best_round,
            "best_checkpoint": self.best_checkpoint,
        }


def proposal_plan(
    seed: int, round_idx: int, history: list[dict], max_steps: int = 2, k: int = 3
):
    """Bounded retry policy; retain best-dev admission while changing rejected proposals."""
    import hashlib
    import json

    def signature(row):
        dev = row.get("candidate_dev_utility", row.get("dev_utility", {}))
        return hashlib.sha256(json.dumps(dev, sort_keys=True).encode()).hexdigest()

    streak = 0
    same = 0
    last_signature = None
    latest = None
    for row in sorted(history, key=lambda r: r["round"], reverse=True):
        decision = row.get("adapter_admission")
        if not decision or decision["accepted"]:
            break
        if latest is None:
            latest = row
            last_signature = signature(row)
        streak += 1
        if signature(row) == last_signature and same == streak - 1:
            same += 1
    retry_level = (
        min(3, 1 + int(latest.get("proposal_plan", {}).get("retry_level", 0)))
        if streak
        else 0
    )
    return {
        "version": "rejection-escape/v1",
        "round_seed": seed + round_idx * 104729,
        "consecutive_rejections": streak,
        "same_candidate_streak": same,
        "retry_level": retry_level,
        "lr_factor": 0.25**retry_level,
        "steps": min(max_steps, 1) if streak else max_steps,
        "minimum_lr": 1e-6,
        "training_rotation": round_idx,
        "gp_exploration": same >= k or streak >= k,
        "gp_generated_per_context": 16 if same >= k or streak >= k else 8,
        "gp_max_evaluated_per_context": 4,
        "reason": "smaller_lr_one_step_and_new_seed_after_rejection"
        if streak
        else "new_round_seed",
    }


def gp_selection_indices(costs, seed: int, explore: bool):
    """Reserve two of four native evaluations for novel exploration when stalled."""
    import random

    order = sorted(range(len(costs)), key=lambda i: (float(costs[i]), i))
    if not explore or len(order) <= 2:
        return order[:4]
    chosen = order[:2]
    chosen.extend(random.Random(seed).sample(order[2:], min(2, len(order) - 2)))
    return chosen
