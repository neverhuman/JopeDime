"""Reward-directed distillation over a deduplicated verified TRAIN archive.

This target is a measured-slate distribution, not an on-policy estimator.
Raw comparable q is preserved; there is no elite selection/exponent clipping.
"""

import math
from dataclasses import dataclass

from rlfw.trie_logprob import TrieLogProb

from .groups import PolicyIdentity, VerifiedQ


@dataclass(frozen=True)
class ArchiveTarget:
    outcomes: tuple[VerifiedQ, ...]
    weights: tuple[float, ...]
    tau: float
    reference: str
    reference_policy: PolicyIdentity | None

    def __post_init__(self):
        if type(self.outcomes) is not tuple or type(self.weights) is not tuple:
            raise TypeError("immutable archive outcomes and target weights required")
        if not self.outcomes or not all(isinstance(q, VerifiedQ) for q in self.outcomes):
            raise TypeError("VerifiedQ target outcomes required")
        if (
            len({q.program for q in self.outcomes}) != len(self.outcomes)
            or len({(q.parent, q.context, q.comparison) for q in self.outcomes}) != 1
        ):
            raise ValueError("one deduplicated comparable archive required")
        if (
            len(self.weights) != len(self.outcomes)
            or not all(math.isfinite(w) and w >= 0 for w in self.weights)
            or abs(math.fsum(self.weights) - 1) > 1e-12
        ):
            raise ValueError("finite normalized nonnegative archive target required")
        if not math.isfinite(self.tau) or self.tau <= 0:
            raise ValueError("positive finite archive temperature required")
        if self.reference == "uniform_deduplicated_verified_slate_alpha0":
            if self.reference_policy is not None:
                raise ValueError("uniform target does not carry a policy reference")
        elif self.reference == "frozen_sampler_conditioned_verified_slate_alpha1":
            if not isinstance(self.reference_policy, PolicyIdentity):
                raise TypeError("pinned archive reference required")
        else:
            raise ValueError("registered archive reference measure required")

    @property
    def scope(self):
        return "archive distillation over measured leaves; NOT full-support on-policy"

    def loss(self, policy, readouts, differentiable_logprobs, *, role_check):
        """The caller supplies sampler-consistent EOS-inclusive sequence sums.

        Coefficients stay frozen across updates. No gradients pass through q,
        the reference policy or the normalized teacher weights.
        """
        if not isinstance(policy, PolicyIdentity):
            raise TypeError("pinned current sampler identity required")
        if self.reference_policy and policy.support != self.reference_policy.support:
            raise ValueError("archive reference/current sampler support mismatch")
        if role_check(self.outcomes) is not True:
            raise ValueError("fresh archive TRAIN-role check failed")
        programs = tuple(q.program for q in self.outcomes)
        if set(readouts) != set(programs) or set(differentiable_logprobs) != set(programs):
            raise ValueError("one likelihood per distinct verified archive program")
        logs = []
        for program in programs:
            trace, actual = readouts[program], differentiable_logprobs[program]
            if not isinstance(trace, TrieLogProb):
                raise TypeError("summed EOS-inclusive sampler readout required")
            value = float(actual.detach()) if hasattr(actual, "detach") else float(actual)
            if not math.isfinite(value) or abs(value - trace.total) > 1e-12:
                raise ValueError("differentiable likelihood differs from verified readout")
            logs.append(actual)
        return -sum(weight * lp for weight, lp in zip(self.weights, logs))


def archive_target(outcomes, *, tau, role_check, reference_logprobs=None, reference_policy=None):
    """rho(a) ∝ mu(a) exp(q(a)/tau), centered before exponentiation.

    Default mu is uniform on DISTINCT qualified programs (alpha=0). Optional
    mu is the frozen sampler conditioned on that same fully enumerated slate
    (alpha=1). An adaptive archive never inherits iid sample multiplicities.
    """
    outcomes = tuple(outcomes)
    if not outcomes or not all(isinstance(q, VerifiedQ) for q in outcomes):
        raise TypeError("nonempty VerifiedQ archive required; numeric/predicted q refused")
    if not math.isfinite(tau) or tau <= 0:
        raise ValueError("positive finite archive temperature required")
    identity = {(q.parent, q.context, q.comparison) for q in outcomes}
    if len(identity) != 1:
        raise ValueError("one comparable TRAIN context per archive target")
    if role_check(outcomes) is not True:
        raise ValueError("archive TRAIN-role check failed")
    unique = {}
    for q in outcomes:
        if q.program in unique and unique[q.program].raw != q.raw:
            raise ValueError("conflicting comparable outcomes for one archive program")
        unique.setdefault(q.program, q)
    ordered = tuple(unique[p] for p in sorted(unique))
    if reference_logprobs is None:
        if reference_policy is not None:
            raise ValueError("reference identity without reference likelihoods")
        log_mu = (0.0,) * len(ordered)
        reference = "uniform_deduplicated_verified_slate_alpha0"
    else:
        if not isinstance(reference_policy, PolicyIdentity):
            raise TypeError("pinned frozen reference sampler required")
        if set(reference_logprobs) != set(unique) or not all(
            isinstance(lp, TrieLogProb) for lp in reference_logprobs.values()
        ):
            raise ValueError("one reference sequence likelihood per distinct program")
        log_mu = tuple(reference_logprobs[q.program].total for q in ordered)
        if math.fsum(math.exp(lp) for lp in log_mu) > 1 + 1e-12:
            raise ValueError("distinct archive action probabilities exceed one")
        reference = "frozen_sampler_conditioned_verified_slate_alpha1"
    maximum_q = max(q.raw for q in ordered)
    scores = tuple(mu + (q.raw - maximum_q) / tau for mu, q in zip(log_mu, ordered))
    peak = max(scores)
    masses = tuple(math.exp(score - peak) for score in scores)
    total = math.fsum(masses)
    return ArchiveTarget(ordered, tuple(m / total for m in masses), tau, reference, reference_policy)
