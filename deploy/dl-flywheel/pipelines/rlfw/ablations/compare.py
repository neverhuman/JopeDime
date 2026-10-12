"""Equal-budget offline comparison. Nothing here promotes a checkpoint.

Realized best-of-budget is max(h, max selected q) - h. The incumbent value of
that quantity is 0. Exact J_B on the measured support is a different number.
A lineage CI that contains 0, including a bound that touches 0, is INCONCLUSIVE.
"""

from __future__ import annotations

import math
from collections import defaultdict

from ..arena.kpi import cluster_stat
from .exact import effective_support, j_b, j_mean, kl_divergence, realized_regret
from .gumbel import gumbel_topk
from .puct import puct_select

BUDGET = 8


def ci_verdict(ci):
    lo, hi = ci
    if lo is None or hi is None:
        return "INCONCLUSIVE"
    if lo > 0:
        return "GREATER"
    if hi < 0:
        return "LESS"
    return "INCONCLUSIVE"


def _softmax(logprobs):
    peak = max(logprobs)
    weights = [math.exp(value - peak) for value in logprobs]
    total = math.fsum(weights)
    if total <= 0:
        raise ValueError("softmax mass must be positive")
    return [weight / total for weight in weights]


def score_context(ctx, budget=BUDGET):
    """Score one labelled context. Search reads q only at a selected leaf."""
    programs = list(ctx["programs"])
    incumbent = ctx["incumbent_q"]
    row = {
        "context_id": ctx["context_id"],
        "lineage_id": ctx["lineage_id"],
        "subrole": ctx["subrole"],
        "measured_support": len(programs),
        "support_bias": "selection_biased_by_arms_that_produced_labels",
        "excluded": None,
        "arms": {
            "incumbent": {
                "realized_best_of_budget": 0.0,
                "delta_vs_incumbent": 0.0,
            }
        },
        "exact_J_B_on_measured_support": None,
        "anchor_effective_support": "unmeasured",
        "selection_kl_puct": None,
        "prior_source": None,
    }
    if len(programs) < budget:
        row["excluded"] = "measured_support_below_budget"
        return row
    if not math.isfinite(incumbent):
        raise ValueError("finite incumbent required")
    logprobs = [program.get("logprob") for program in programs]
    have_logprobs = all(
        isinstance(value, (int, float)) and math.isfinite(value) for value in logprobs
    )
    values = [program["q"] for program in programs]
    uniform = [1.0 / len(programs)] * len(programs)
    if have_logprobs:
        prior = _softmax(logprobs)
        row["prior_source"] = "anchor_trie_logprob_renormalized_on_measured_support"
        row["anchor_effective_support"] = effective_support(prior)
        row["exact_J_B_on_measured_support"] = {
            "uniform": j_b(values, uniform, incumbent, budget),
            "anchor_softmax": j_b(values, prior, incumbent, budget),
            "j_mean_uniform": j_mean(values, uniform),
            "j_mean_anchor": j_mean(values, prior),
        }
    else:
        prior = uniform
        row["prior_source"] = "uniform_anchor_logprob_unmeasured"
        row["exact_J_B_on_measured_support"] = {
            "uniform": j_b(values, uniform, incumbent, budget),
            "anchor_softmax": "unmeasured",
            "j_mean_uniform": j_mean(values, uniform),
            "j_mean_anchor": "unmeasured",
        }
    tokens = [program["tokens"] for program in programs]

    def q_of(index):
        return values[index]

    puct = puct_select(tokens, prior, q_of, budget)
    gumbel_index = gumbel_topk(prior, budget)
    gumbel_q = [values[index] for index in gumbel_index]
    for name, selected_q in (("puct", puct["q"]), ("gumbel", gumbel_q)):
        realized = realized_regret(selected_q, incumbent)
        row["arms"][name] = {
            "realized_best_of_budget": realized,
            "delta_vs_incumbent": realized,
            "n": budget,
        }
    if have_logprobs:
        # Uniform on the selected leaves versus the anchor prior on M_c.
        mass = [0.0] * len(prior)
        for index in puct["selected"]:
            mass[index] = 1.0 / budget
        row["selection_kl_puct"] = kl_divergence(mass, prior)
    for arm, slot_qs in ctx.get("charged", {}).items():
        if len(slot_qs) != budget:
            row["arms"][arm] = {"excluded": "charged_slots_not_budget", "n": len(slot_qs)}
            continue
        realized = realized_regret(slot_qs, incumbent)
        row["arms"][arm] = {
            "realized_best_of_budget": realized,
            "delta_vs_incumbent": realized,
            "n": budget,
        }
    rerank = row["arms"].get("rerank32:current_latest") or row["arms"].get("rerank32")
    if rerank and "realized_best_of_budget" in rerank:
        for name in ("puct", "gumbel", "uniform_trie", "trie_sample:current_latest"):
            cell = row["arms"].get(name)
            if cell and "realized_best_of_budget" in cell:
                cell["delta_vs_rerank32"] = (
                    cell["realized_best_of_budget"] - rerank["realized_best_of_budget"]
                )
    return row


def lineage_deltas(rows, arm, field="delta_vs_incumbent", subrole=None):
    """Parent-cluster CI. One context contributes one delta; siblings are not clusters."""
    groups = defaultdict(list)
    for row in rows:
        if row["excluded"] is not None:
            continue
        if subrole is not None and row["subrole"] != subrole:
            continue
        cell = row["arms"].get(arm)
        if not cell or field not in cell or cell.get("excluded"):
            continue
        groups[row["lineage_id"]].append(cell[field])
    stat = cluster_stat(dict(groups))
    return {
        "arm": arm,
        "field": field,
        "subrole": subrole,
        "mean": stat["mean"],
        "ci95": stat["ci95"],
        "lineages": stat["lineages"],
        "units": stat["units"],
        "verdict": ci_verdict(stat["ci95"]),
        "promoted": False,
        "directional": True,
        "ship_gate": False,
    }
