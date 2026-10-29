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
            optimizer.load_state_dict(self.optimizer_state)
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
