"""Stop-gradient RLOO and PKPO coefficients; no optimizer/model/IO ownership."""

import math
from dataclasses import dataclass

from rlfw.trie_logprob import TrieLogProb

from .groups import PolicyIdentity, ResolvedGroup


def _values(values):
    values = tuple(values)
    if not values or not all(math.isfinite(x) for x in values):
        raise ValueError("nonempty finite utilities required")
    return values


def rloo_weights(utilities, *, std_normalize=False):
    """Mean-objective weights. G=1 uses fixed-zero REINFORCE, not LOO.

    This numerical primitive accepts utility vectors for proof fixtures; the
    reward-consuming public entry point is objective(ResolvedGroup, ...).
    """
    if std_normalize:
        raise ValueError("group-std-normalized RLOO is biased and not supported")
    u = _values(utilities)
    g = len(u)
    if g == 1:
        return u
    return tuple(
        (x - math.fsum(u[:i] + u[i + 1 :]) / (g - 1)) / g for i, x in enumerate(u)
    )


def pkpo_weights(utilities, h, best_of):
    """Unbiased gradient of E[(max of B - h)+] using G >= B draws.

    Average the B-draw marginal estimator over all C(G,B) subsets. Sorting
    collapses this U-statistic to O(G log G), ties included. These weights
    already contain the subset normalization; DO NOT divide again by G.
    """
    u = _values(utilities)
    g, b = len(u), best_of
    if not math.isfinite(h) or type(b) is not int or not 1 <= b <= g:
        raise ValueError("finite identically transformed h and 1 <= B <= G required")
    clipped = [max(h, x) for x in u]
    if b == 1:
        return tuple((x - h) / g for x in clipped)
    order = sorted(range(g), key=clipped.__getitem__)
    denominator = math.comb(g, b)
    out, lower_sum = [0.0] * g, 0.0
    for k, index in enumerate(order, 1):
        if k < b:
            continue
        # m=b-1 .. k-1; the empty binomial at k=b-1 has coefficient 1.
        m = k - 1
        lower_sum += math.comb(m - 1, b - 2) * clipped[order[m - 1]]
        out[index] = (
            math.comb(k - 1, b - 1) * clipped[index] - lower_sum
        ) / denominator
    return tuple(max(0.0, x) for x in out)


@dataclass(frozen=True)
class StalenessGate:
    accepted: bool
    reason: str
    lag_updates: int
    lag_seconds: float
    probe_kl: float
    ratios: tuple[float, ...]
    ess: float
    joint_ratio: float
    correction: str


def staleness_gate(
    group, current_policy, current_logprobs, *, update, seconds, probe_kl
):
    """Window tags and IS gate. All changed checkpoints use exact IS.

    <=64 updates/<=180 seconds is a freshness label, not an exemption from
    correction. Outside it, the spec permits exact IS if ratios/ESS pass.
    Eligibility selection can itself bias the admitted distribution: the
    estimator identity is exact before filtering, not a claim about accepted
    groups. No clipping or self-normalization is applied to ratios.
    """
    if not isinstance(group, ResolvedGroup) or not isinstance(
        current_policy, PolicyIdentity
    ):
        raise TypeError("resolved raw group and pinned current policy required")
    spec = group.spec
    logs = tuple(current_logprobs)
    if len(logs) != len(spec.draws) or not all(
        isinstance(x, TrieLogProb) for x in logs
    ):
        raise ValueError("one summed EOS-inclusive trie likelihood per raw draw")
    if (
        type(update) is not int
        or not math.isfinite(seconds)
        or not math.isfinite(probe_kl)
        or probe_kl < 0
    ):
        raise ValueError(
            "update, monotonic time and nonnegative finite probe KL required"
        )
    lag, age = update - spec.generated_update, seconds - spec.generated_seconds
    if lag < 0 or age < 0:
        raise ValueError("negative lag / mismatched monotonic clock")
    fresh = lag <= 64 and age <= 180
    differences = tuple(x.total - d.behavior.total for x, d in zip(logs, spec.draws))
    # Test log-space bounds first, so malformed/huge stale ratios never overflow.
    bounded = all(math.log(0.5) <= x <= math.log(2) for x in differences)
    ratios = tuple(math.exp(x) for x in differences) if bounded else ()
    ess = math.fsum(ratios) ** 2 / math.fsum(x * x for x in ratios) if ratios else 0.0
    reason = "accepted"
    if current_policy.support != spec.policy.support:
        reason = "support_mismatch"
    elif probe_kl >= 0.05:
        reason = "probe_kl_hard_stop"
    elif not bounded:
        reason = "ratio_outside_[0.5,2]"
    elif ess < len(logs) / 2:
        reason = "ess_below_G/2"
    elif current_policy.checkpoint == spec.policy.checkpoint and (
        lag != 0 or any(abs(x) > 1e-12 for x in differences)
    ):
        reason = "same_checkpoint_trace_mismatch"
    elif current_policy.checkpoint != spec.policy.checkpoint and lag == 0:
        reason = "checkpoint_changed_without_update"
    correction = (
        "exact_fresh"
        if current_policy == spec.policy
        else ("IS_in_window" if fresh else "IS_outside_window")
    )
    return StalenessGate(
        reason == "accepted",
        reason,
        lag,
        age,
        probe_kl,
        ratios,
        ess,
        math.prod(ratios) if ratios else 0.0,
        correction,
    )


@dataclass(frozen=True)
class ObjectiveTerms:
    dense: tuple[float, ...]
    hwm: tuple[float, ...]
    current_logprobs: tuple[float, ...]
    gate: StalenessGate
    hwm_status: str

    def losses(self, differentiable_logprobs):
        """Works with Python scalars or graph-connected scalar tensors.

        All rewards, baselines and IS coefficients are plain frozen floats.
        Only the recomputed current log-probabilities carry a gradient.
        """
        logs = tuple(differentiable_logprobs)
        if len(logs) != len(self.dense):
            raise ValueError("raw draw multiplicity must match")
        for actual, expected in zip(logs, self.current_logprobs):
            value = (
                float(actual.detach()) if hasattr(actual, "detach") else float(actual)
            )
            if not math.isfinite(value) or abs(value - expected) > 1e-12:
                raise ValueError("differentiable readout differs from gated likelihood")
        # Connected zero for empty HWM support; coefficients remain detached.
        return (
            -sum(w * x for w, x in zip(self.dense, logs)),
            -sum(w * x for w, x in zip(self.hwm, logs)),
        )


def objective(
    group, current_policy, current_logprobs, *, update, seconds, probe_kl, role_check
):
    if not isinstance(group, ResolvedGroup):
        raise TypeError("resolved group required")
    if role_check(group.spec) is not True:
        raise ValueError("fresh train-role/parent check failed before actor step")
    current_logprobs = tuple(current_logprobs)
    gate = staleness_gate(
        group,
        current_policy,
        current_logprobs,
        update=update,
        seconds=seconds,
        probe_kl=probe_kl,
    )
    if not gate.accepted:
        raise ValueError(f"actor group demoted to L_PI/L6: {gate.reason}")
    u = group.utilities
    dense = tuple(w * ratio for w, ratio in zip(rloo_weights(u), gate.ratios))
    h = group.h
    status = "no_comparable_incumbent" if h is None else "active"
    # Solved raw incumbents are excluded from clipped J_B (plan 14.2.8).
    if (
        h is not None
        and group.spec.sample_rule.mode == "clipped"
        and h >= group.spec.sample_rule.cap
    ):
        status = "solved"
    hwm = (
        tuple(0.0 for _ in u)
        if status != "active"
        else tuple(w * gate.joint_ratio for w in pkpo_weights(u, h, group.spec.best_of))
    )
    return ObjectiveTerms(
        dense, hwm, tuple(x.total for x in current_logprobs), gate, status
    )


def parent_macro_mean(rows):
    """rows=(parent, context, scalar/tensor). Groups→contexts→parents equally."""
    parents = {}
    for parent, context, value in rows:
        parents.setdefault(parent, {}).setdefault(context, []).append(value)
    if not parents:
        raise ValueError("nonempty supported terms required")
    return sum(
        sum(sum(v) / len(v) for v in contexts.values()) / len(contexts)
        for contexts in parents.values()
    ) / len(parents)
