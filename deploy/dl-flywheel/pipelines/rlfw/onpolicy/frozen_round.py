"""One in-memory frozen-round update; durable optimizer CAS remains runner-owned."""

import math
from dataclasses import dataclass

from .collector import RoundCollector
from .groups import PolicyIdentity
from .objective import objective, parent_macro_mean

SCOPES = frozenset(("full_support_iid", "measured_leaf_pi_M"))


@dataclass(frozen=True)
class ActorState:
    policy: PolicyIdentity
    update: int
    dropout_enabled: bool

    def __post_init__(self):
        if not isinstance(self.policy, PolicyIdentity) or type(self.update) is not int or self.update < 0:
            raise ValueError("pinned policy and nonnegative update required")
        if type(self.dropout_enabled) is not bool:
            raise ValueError("explicit actor dropout state required")


@dataclass(frozen=True)
class FrozenStepResult:
    round_id: str
    scope: str
    state: str
    groups: tuple[str, ...]
    quarantined: tuple[tuple[str, tuple[str, ...]], ...]
    update_callback_calls: int


class FrozenRoundStep:
    """No policy drift during collection/recomputation; no partial-wave update.

    Proposals must be raw iid draws under the declared support. Explicitly
    name measured-leaf pi_M when the sampler conditions on known outcomes.
    A fresh state probe must bind actual parameters/update/dropout in the
    caller. This library cannot certify the callback's claimed sampling law.
    """

    def __init__(self, round_id, groups, *, scope):
        if scope not in SCOPES:
            raise ValueError("explicit full_support_iid or measured_leaf_pi_M scope required")
        groups = tuple(groups)
        self._collector = RoundCollector(round_id, groups)
        self._round_id = round_id
        self._scope = scope
        self._state = ActorState(groups[0].policy, groups[0].generated_update, False)
        self._consumed = False

    def resolve(self, group_id, draw_id, outcome):
        self._collector.resolve(group_id, draw_id, outcome)

    def _check(self, probe):
        if probe() != self._state:
            raise ValueError("frozen actor parameters/update/support/dropout changed")

    def step(
        self,
        *,
        state_probe,
        recompute,
        apply_update,
        seconds,
        probe_kl,
        role_check,
        lambda_rloo=1.0,
        lambda_hwm=0.0,
    ):
        """Recompute(group)->(TrieLogProb traces, differentiable sequence sums).

        apply_update(loss) is invoked at most once, after the entire wave is
        complete. Refusal/exception consumes this round; it cannot be retried.
        Rewards/IS coefficients never retain an autograd graph. Actual update
        and changed-parameter proof must be recorded by the owning runner.
        """
        if self._consumed:
            raise ValueError("round consumed; no second optimizer callback")
        self._consumed = True
        result = self._collector.close()
        self._check(state_probe)
        if not all(math.isfinite(w) and w >= 0 for w in (lambda_rloo, lambda_hwm)):
            raise ValueError("finite nonnegative objective weights required")
        if lambda_rloo + lambda_hwm == 0:
            raise ValueError("at least one actor objective must be active")
        ids = tuple(g.spec.group_id for g in result.groups)
        if result.quarantined:
            return FrozenStepResult(
                self._round_id,
                self._scope,
                "quarantined_no_update",
                ids,
                result.quarantined,
                0,
            )
        dense_rows, hwm_rows = [], []
        for group in result.groups:
            self._check(state_probe)
            traces, connected_logs = recompute(group)
            self._check(state_probe)
            terms = objective(
                group,
                self._state.policy,
                traces,
                update=self._state.update,
                seconds=seconds,
                probe_kl=probe_kl,
                role_check=role_check,
            )
            dense, hwm = terms.losses(connected_logs)
            dense_rows.append((group.spec.parent, group.spec.context, dense))
            if terms.hwm_status == "active":
                hwm_rows.append((group.spec.parent, group.spec.context, hwm))
        dense = parent_macro_mean(dense_rows)
        # A connected zero preserves the separate no-incumbent/solved mask.
        hwm = parent_macro_mean(hwm_rows) if hwm_rows else dense * 0
        self._check(state_probe)
        apply_update(lambda_rloo * dense + lambda_hwm * hwm)
        return FrozenStepResult(self._round_id, self._scope, "update_callback_applied_once", ids, (), 1)
