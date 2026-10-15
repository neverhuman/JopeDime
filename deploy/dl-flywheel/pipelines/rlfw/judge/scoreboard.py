"""Three program-quality scoreboards and a paired L6-versus-frozen-v1 report.

Proposal quality and search quality use q, where higher is better. Delivered
generalization reports the outer q of the program selected on training-side q.
The best outer score among candidates is a non-deployable diagnostic. L6 is a
loss, so lower is better, and an L6 label is not a program-quality result.
Nothing here promotes, spends a seal, or claims an improvement.
"""

from __future__ import annotations

import math

from rlfw.judge.cluster import cluster_bootstrap_ci, context_interval

PROPOSAL_BUDGETS = (1, 8)
SEARCH_ARMS = ("policy_seeded", "frozen_seeded", "retrieval", "independent")
N_GATE = 881


def _finite(value):
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
    )


def _mean(values):
    if not values:
        raise ValueError("scoreboard has no contexts")
    return sum(values) / len(values)


def proposal_quality(rows):
    """B=1 first-program q and B=8 max q, plus novelty over an unlowered HWM."""
    by_budget = {budget: {} for budget in PROPOSAL_BUDGETS}
    for row in rows:
        budget = row["budget"]
        if budget not in by_budget:
            raise ValueError("proposal budget must be 1 or 8")
        programs = list(row["programs"])
        if len(programs) != budget or any(not _finite(q) for q in programs):
            raise ValueError("proposal programs must be finite and match the budget")
        context_id = row["context_id"]
        if context_id in by_budget[budget]:
            raise ValueError("duplicate proposal context")
        by_budget[budget][context_id] = row
    identities = set(by_budget[1])
    if not identities or identities != set(by_budget[8]):
        raise ValueError("B=1 and B=8 must cover the same contexts")
    ordered = sorted(identities)
    report = {
        "board": "proposal",
        "higher_is_better": True,
        "n": len(ordered),
        "budgets": [1, 8],
        "J_1": _mean([by_budget[1][item]["programs"][0] for item in ordered]),
        "M_8": _mean([max(by_budget[8][item]["programs"]) for item in ordered]),
        "promotion": False,
        "seal_spent": False,
    }
    if all("hwm" in by_budget[8][item] for item in ordered):
        gains = []
        for item in ordered:
            incumbent = by_budget[8][item]["hwm"]
            if not _finite(incumbent):
                raise ValueError("hwm must be finite")
            best = max(by_budget[8][item]["programs"])
            gains.append(max(incumbent, best) - incumbent)
        report["J_8_hwm"] = _mean(gains)
        report["hwm_lowered"] = False
    return report


def search_quality(rows):
    """Paired search q for the four named arms. Positive delta means higher q."""
    buckets = {arm: {} for arm in SEARCH_ARMS}
    for row in rows:
        arm = row["arm"]
        if arm not in buckets:
            raise ValueError("unknown search arm")
        score = row["q"]
        if not _finite(score):
            raise ValueError("search q must be finite")
        context_id = row["context_id"]
        if context_id in buckets[arm]:
            raise ValueError("duplicate search row")
        buckets[arm][context_id] = score
    identities = set(buckets["independent"])
    if not identities or any(set(buckets[arm]) != identities for arm in SEARCH_ARMS):
        raise ValueError("search arms must share one context set")
    means = {
        arm: _mean([buckets[arm][item] for item in identities]) for arm in SEARCH_ARMS
    }
    deltas = {
        arm: means[arm] - means["independent"]
        for arm in SEARCH_ARMS
        if arm != "independent"
    }
    return {
        "board": "search",
        "higher_is_better": True,
        "baseline": "independent",
        "n": len(identities),
        "mean_q": means,
        "delta_vs_independent": deltas,
        "not_an_l6_verdict": True,
        "promotion": False,
        "seal_spent": False,
    }


def _selected(candidates):
    """Training-side max train_q. Ties break by program id, never by outer q."""
    return min(candidates, key=lambda item: (-item["train_q"], item["program_id"]))


def delivered_generalization(rows):
    """Outer q of the training-selected program. The outer maximum is diagnostic."""
    chosen = {}
    for row in rows:
        context_id = row["context_id"]
        if context_id in chosen:
            raise ValueError("duplicate delivered context")
        candidates = list(row["candidates"])
        if not candidates:
            raise ValueError("delivered context needs candidates")
        seen = set()
        for candidate in candidates:
            program_id = candidate["program_id"]
            if program_id in seen:
                raise ValueError("duplicate program")
            seen.add(program_id)
            if not _finite(candidate["train_q"]) or not _finite(candidate["outer_q"]):
                raise ValueError("delivered scores must be finite")
        selected = _selected(candidates)
        oracle = max(candidates, key=lambda item: (item["outer_q"], item["program_id"]))
        chosen[context_id] = (selected, oracle)
    if not chosen:
        raise ValueError("delivered board is empty")
    return {
        "board": "delivered",
        "selection": "training_side_max_train_q",
        "higher_is_better": True,
        "n": len(chosen),
        "mean_delivered_q": _mean([pair[0]["outer_q"] for pair in chosen.values()]),
        "mean_oracle_outer_q": _mean([pair[1]["outer_q"] for pair in chosen.values()]),
        "oracle_deployable": False,
        "oracle_affects_selection": False,
        "promotion": False,
        "seal_spent": False,
    }


def _l6_label(low, high, n):
    if n >= N_GATE and high < 0:
        return "LOWER"
    if n >= N_GATE and low > 0:
        return "HIGHER"
    return "INCONCLUSIVE"


def l6_vs_frozen_v1(arm, frozen, groups=None):
    """Paired d = arm L6 - frozen v1 L6. Negative means the arm has lower loss.

    LOWER or HIGHER requires n>=881 and a context interval that excludes 0.
    A cluster interval is recorded only when groups are the same differences,
    and it does not change the label. This is not an improvement claim.
    """
    if set(arm) != set(frozen) or len(arm) < 2:
        raise ValueError("paired L6 contexts must match")
    identities = sorted(arm)
    diffs = []
    for context_id in identities:
        left = arm[context_id]
        right = frozen[context_id]
        if not _finite(left) or not _finite(right):
            raise ValueError("L6 costs must be finite")
        diffs.append(left - right)
    interval = context_interval(diffs)
    report = {
        "schema": "jope.j21-l6-vs-v1/v1",
        "d": "arm_minus_frozen_v1",
        "lower_is_better": True,
        "above_v1_is_worse": True,
        "n": len(diffs),
        "mean_arm": _mean([arm[item] for item in identities]),
        "mean_frozen_v1": _mean([frozen[item] for item in identities]),
        "mean_d": interval["mean"],
        "ci_low": interval["low"],
        "ci_high": interval["high"],
        "label": _l6_label(interval["low"], interval["high"], len(diffs)),
        "n_gate": N_GATE,
        "l6_is_program_quality": False,
        "improvement_claim": False,
        "qualifying": False,
        "promotion": False,
        "seal_spent": False,
        "cluster_decides": False,
    }
    if groups is not None:
        pooled = sorted(value for group in groups for value in group)
        if [len(group) for group in groups] == [] or pooled != sorted(diffs):
            raise ValueError("cluster groups must be the paired L6 differences")
        boot = cluster_bootstrap_ci(groups)
        report["cluster_low"] = boot["cluster_bootstrap"]["low"]
        report["cluster_high"] = boot["cluster_bootstrap"]["high"]
        report["cluster_decides"] = False
    return report


def automated_report(proposal_rows, search_rows, delivered_rows, arm, frozen, groups=None):
    """One report. The L6 section does not stand in for delivered program quality."""
    l6 = l6_vs_frozen_v1(arm, frozen, groups)
    return {
        "schema": "jope.j21-scoreboards/v1",
        "proposal": proposal_quality(proposal_rows),
        "search": search_quality(search_rows),
        "delivered": delivered_generalization(delivered_rows),
        "l6_vs_frozen_v1": l6,
        "l6_does_not_substitute_for_delivered": True,
        "improvement_claim": False,
        "promotion": False,
        "seal_spent": False,
    }


def select_admissible_endpoint(epoch_rows, max_kl=0.05):
    """Pre-registered E_last_valid endpoint selector for KL-guarded training arms.

    Evaluates ordered epochs and selects the highest epoch prior to the first
    KL disqualification (using base_kl when decomposed support KL is present,
    otherwise total kl). Never promotes or spends a seal.
    """
    if not _finite(max_kl) or max_kl <= 0.0:
        raise ValueError("max_kl must be finite and positive")
    rows = list(epoch_rows)
    if not rows:
        raise ValueError("epoch_rows must not be empty")
    by_epoch = {}
    for row in rows:
        epoch = row["epoch"]
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise ValueError("epoch must be a non-negative integer")
        if epoch in by_epoch:
            raise ValueError("duplicate epoch in endpoint selection")
        kl = row["kl"]
        l6 = row["l6"]
        if not _finite(kl) or kl < 0.0 or not _finite(l6):
            raise ValueError("epoch kl and l6 must be finite")
        if "base_kl" in row and (not _finite(row["base_kl"]) or row["base_kl"] < 0.0):
            raise ValueError("base_kl must be finite and non-negative")
        if "archive_ce" in row and not _finite(row["archive_ce"]):
            raise ValueError("archive_ce must be finite when provided")
        by_epoch[epoch] = row

    ordered_epochs = sorted(by_epoch)
    if ordered_epochs[0] != 0:
        raise ValueError("epoch_rows must include epoch 0 anchor")

    first_violation_epoch = None
    last_valid_epoch = 0
    for epoch in ordered_epochs:
        row = by_epoch[epoch]
        effective_kl = row.get("base_kl", row["kl"])
        alarmed = bool(row.get("kl_disqualified", False)) or effective_kl > max_kl
        if alarmed:
            first_violation_epoch = epoch
            break
        last_valid_epoch = epoch

    chosen = by_epoch[last_valid_epoch]
    report = {
        "schema": "jope.j21-admissible-endpoint/v1",
        "rule": "E_last_valid",
        "max_kl": max_kl,
        "selected_epoch": last_valid_epoch,
        "stopped_early": first_violation_epoch is not None,
        "first_violation_epoch": first_violation_epoch,
        "selected_l6": chosen["l6"],
        "selected_kl": chosen.get("base_kl", chosen["kl"]),
        "promotion": False,
        "seal_spent": False,
    }
    if "archive_ce" in chosen:
        report["selected_archive_ce"] = chosen["archive_ce"]
    return report


def multi_round_ratchet_summary(rounds, max_base_kl=0.05):
    """Multi-round self-play ratchet summary across expanding support rounds."""
    if not _finite(max_base_kl) or max_base_kl <= 0.0:
        raise ValueError("max_base_kl must be finite and positive")
    items = list(rounds)
    if len(items) < 2:
        raise ValueError("multi-round ratchet requires at least 2 rounds")

    for expected_idx, item in enumerate(items):
        if item.get("round_index") != expected_idx:
            raise ValueError("rounds must have sequential round_index starting at 0")
        for key in ("mean_search_q", "mean_delivered_q", "base_kl", "new_winner_mass"):
            if not _finite(item.get(key)):
                raise ValueError(f"round {key} must be finite")
        if item["base_kl"] < 0.0 or not (0.0 <= item["new_winner_mass"] <= 1.0):
            raise ValueError("round base_kl and new_winner_mass must be valid")
        support_size = item.get("support_size")
        admitted = item.get("admitted_winners")
        if (
            isinstance(support_size, bool)
            or not isinstance(support_size, int)
            or support_size <= 0
            or isinstance(admitted, bool)
            or not isinstance(admitted, int)
            or admitted < 0
        ):
            raise ValueError("support_size and admitted_winners must be valid integers")

    support_monotonic = all(
        items[i]["support_size"] >= items[i - 1]["support_size"]
        for i in range(1, len(items))
    )
    delivered_monotonic = all(
        items[i]["mean_delivered_q"] >= items[i - 1]["mean_delivered_q"]
        for i in range(1, len(items))
    )
    kl_guarded = all(item["base_kl"] <= max_base_kl for item in items)

    return {
        "schema": "jope.j21-selfplay-ratchet/v1",
        "num_rounds": len(items),
        "initial_support_size": items[0]["support_size"],
        "final_support_size": items[-1]["support_size"],
        "total_admitted_winners": sum(item["admitted_winners"] for item in items),
        "search_q_gain": items[-1]["mean_search_q"] - items[0]["mean_search_q"],
        "delivered_q_gain": items[-1]["mean_delivered_q"] - items[0]["mean_delivered_q"],
        "support_monotonic": support_monotonic,
        "ratchet_monotonic_delivered_q": delivered_monotonic,
        "kl_guarded": kl_guarded,
        "max_base_kl_observed": max(item["base_kl"] for item in items),
        "not_an_l6_verdict": True,
        "promotion": False,
        "seal_spent": False,
    }

