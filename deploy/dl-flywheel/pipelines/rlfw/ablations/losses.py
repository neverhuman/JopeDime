"""IPO and DPO on the trie log-odds margin. Ablations only; they do not promote."""

from __future__ import annotations

import math

IPO_TAUS = (2.0, 5.0, 10.0)
DPO_BETA = 0.1


def preference_margin(logp_theta_w, logp_theta_l, logp_ref_w, logp_ref_l):
    """d = (ell_theta(w) - ell_theta(l)) - (ell_ref(w) - ell_ref(l))."""
    values = (logp_theta_w, logp_theta_l, logp_ref_w, logp_ref_l)
    if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in values):
        raise ValueError("finite log-probabilities required")
    return (logp_theta_w - logp_theta_l) - (logp_ref_w - logp_ref_l)


def ipo_loss(margin, tau):
    """L_IPO = (d - 1/(2 tau))^2. tau is in {2, 5, 10}."""
    if tau not in IPO_TAUS:
        raise ValueError("tau_IPO must be one of 2, 5, 10")
    if not math.isfinite(margin):
        raise ValueError("finite margin required")
    target = 1.0 / (2.0 * tau)
    return (margin - target) ** 2


def dpo_loss(margin, beta=DPO_BETA):
    """L_DPO = -log sigmoid(beta * d) with beta 0.1."""
    if beta != DPO_BETA:
        raise ValueError("beta is fixed at 0.1")
    if not math.isfinite(margin):
        raise ValueError("finite margin required")
    z = beta * margin
    # Stable -log(sigmoid(z)).
    if z >= 0:
        return z + math.log1p(math.exp(-z)) - z
    return math.log1p(math.exp(z)) - z
