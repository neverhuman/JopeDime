"""Measurement definitions. Missing evidence is None, never a synthetic zero."""

import math
import statistics
from collections import defaultdict

STAGES = (
    "PROPOSAL_COMMITTED",
    "ASSEMBLING",
    "MANIFEST_SEALED",
    "PREFETCHED",
    "HASH_VERIFIED",
    "SEMANTIC_VERIFIED",
    "ROLE_RECHECKED",
    "READY",
    "ADMITTED",
    "FIRST_EXPOSURE",
)


def quantiles(values):
    """Linear empirical quantiles, including n=1; empty distributions stay null."""
    values = sorted(values)
    if any(not math.isfinite(x) for x in values):
        raise ValueError("nonfinite measurement")
    result = {"n": len(values)}
    for name, p in (("p50", 0.5), ("p95", 0.95), ("p99", 0.99)):
        if not values:
            result[name] = None
            continue
        index = (len(values) - 1) * p
        lo = int(index)
        result[name] = values[lo] + (
            values[min(lo + 1, len(values) - 1)] - values[lo]
        ) * (index - lo)
    return result


def macro(values, parents):
    """Mean within parent, then across the fixed parent roster; no silent dropping."""
    grouped = defaultdict(list)
    for parent, value in values:
        grouped[parent].append(value)
    if any(not grouped[p] or any(x is None for x in grouped[p]) for p in parents):
        return None
    return statistics.mean(statistics.mean(grouped[p]) for p in parents)


def north_star(events, contexts, *, start, end, cpu_seconds=None, cpu_source=None):
    """Half-open window; exact HWM keys, verified q, lifetime first-seen evidence.

    contexts bind context_id, parent, hwm_key, frozen_q_grid and resolved.
    events bind at, context_id, hwm_key, program_sha256, q_grid, verified,
    first_seen and ledger_event. Only actual ledger improvements count for G.
    """
    if not math.isfinite(start) or not math.isfinite(end) or end <= start:
        raise ValueError("positive finite window required")
    by_context = {c["context_id"]: c for c in contexts}
    if not contexts or len(by_context) != len(contexts):
        raise ValueError("nonempty distinct fixed context roster required")
    parents = sorted({c["parent"] for c in contexts})
    useful = defaultdict(set)
    best = {c["context_id"]: c.get("frozen_q_grid") for c in contexts}
    improvements = defaultdict(set)
    selected = []
    missing_first_seen = False
    for e in events:
        c = by_context.get(e["context_id"])
        if c is None or e["hwm_key"] != c["hwm_key"]:
            continue
        if e.get("verified") is not True or not start <= e["at"] < end:
            continue
        if type(e.get("q_grid")) is not int:
            raise ValueError("verified measured micro-q required")
        selected.append(e)
        if best[c["context_id"]] is not None:
            best[c["context_id"]] = max(best[c["context_id"]], e["q_grid"])
        if "first_seen" not in e or e["first_seen"] is None:
            missing_first_seen = True
        elif (
            e["first_seen"] is True
            and c.get("frozen_q_grid") is not None
            and e["q_grid"] > c["frozen_q_grid"]
        ):
            useful[c["parent"]].add(e["program_sha256"])
        if e.get("ledger_event") == "improvement":
            improvements[c["parent"]].add(
                (e["hwm_key"], e["program_sha256"], e["q_grid"])
            )
    complete = all(
        c.get("resolved") is True and c.get("frozen_q_grid") is not None
        for c in contexts
    )
    u = statistics.mean(len(useful[p]) / (end - start) for p in parents)
    delta = macro(
        [
            (
                c["parent"],
                None
                if c.get("frozen_q_grid") is None
                else (best[c["context_id"]] - c["frozen_q_grid"]) / 1e6,
            )
            for c in contexts
        ],
        parents,
    )
    g = None
    # Process/child CPU receipts and elapsed wall*time-worker proxies cannot qualify G.
    if cpu_seconds is not None and cpu_source == "evaluator_cgroup":
        if set(cpu_seconds) != set(parents) or any(
            not math.isfinite(x) or x <= 0 for x in cpu_seconds.values()
        ):
            raise ValueError(
                "positive evaluator cgroup seconds for every fixed parent required"
            )
        if all("ledger_event" in e for e in selected) and complete:
            g = statistics.mean(
                len(improvements[p]) / (cpu_seconds[p] / 3600) for p in parents
            )
    return {
        "U_useful": u if complete and not missing_first_seen else None,
        "useful_verified_programs": sum(len(useful[p]) for p in parents),
        "G_per_core_hour": g,
        "delta_H": delta if complete else None,
        "delta_H_observed_lower_bound": delta,
        "contexts": len(contexts),
        "parents": len(parents),
        "complete": complete,
    }


def paired_noise(incumbent, challenger, *, epsilon=1e-12, seed=0, draws=2000):
    """Stratified paired row bootstrap of mean fold log RMSE ratios (spec §6.2).

    Both inputs have exactly two pinned inner folds, with index, targets and
    predictions in retained score-row order. Never decode an outer test fold.
    """
    import numpy as np

    if draws != 2000 or not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("2,000 draws and a positive frozen epsilon required")
    if len(incumbent) != 2 or len(challenger) != 2:
        raise ValueError("two inner folds required")
    differences, draws_by_fold = [], []
    rng = np.random.default_rng(seed)
    for j, (old, new) in enumerate(zip(incumbent, challenger)):
        if old["index"] != j or new["index"] != j or old["targets"] != new["targets"]:
            raise ValueError("pinned fold and ordered targets must match")
        target = np.asarray(old["targets"], dtype=np.float64)
        a = np.asarray(old["predictions"], dtype=np.float64)
        b = np.asarray(new["predictions"], dtype=np.float64)
        if (
            target.ndim != 1
            or len(target) < 2
            or a.shape != target.shape
            or b.shape != target.shape
        ):
            raise ValueError("paired nonempty score-row vectors required")
        if not all(np.isfinite(x).all() for x in (target, a, b)):
            raise ValueError("finite saved predictions/targets required")
        a2, b2 = (a - target) ** 2, (b - target) ** 2
        for fold, errors in ((old, a2), (new, b2)):
            if "metrics" in fold and not math.isclose(
                float(np.sqrt(errors.mean())),
                fold["metrics"]["rmse"],
                rel_tol=1e-10,
                abs_tol=1e-12,
            ):
                raise ValueError("saved prediction RMSE differs from retained metric")
        differences.append(
            math.log(
                max(float(np.sqrt(a2.mean())), epsilon)
                / max(float(np.sqrt(b2.mean())), epsilon)
            )
        )
        samples = []
        # Bounded temporary allocation even for a large retained fitting context.
        for offset in range(0, draws, 32):
            indices = rng.integers(
                0, len(target), size=(min(32, draws - offset), len(target))
            )
            samples.extend(
                np.log(
                    np.maximum(np.sqrt(a2[indices].mean(axis=1)), epsilon)
                    / np.maximum(np.sqrt(b2[indices].mean(axis=1)), epsilon)
                )
            )
        draws_by_fold.append(np.asarray(samples))
    bootstrap_se = float(np.std((draws_by_fold[0] + draws_by_fold[1]) / 2, ddof=1))
    gap = abs(differences[0] - differences[1])
    se = max(bootstrap_se, gap / 2)
    return {
        "d0": differences[0],
        "d1": differences[1],
        "abs_d0_minus_d1": gap,
        "bootstrap_se": bootstrap_se,
        "SE_pair": se,
        "tau_c": max(0.01, 1.96 * se),
        "draws": draws,
        "seed": seed,
        "epsilon": epsilon,
    }


def feedback_latencies(events, *, observed_until):
    """Join only explicit exposure identities; missing exposure remains censored.

    join_key is the pinned (proposal, context, program, result-ref, snapshot, term)
    identity supplied by the receipt adapter, not a program-only approximation.
    """
    grouped = defaultdict(dict)
    identities = {}
    for e in events:
        if e["stage"] not in STAGES or not math.isfinite(e["at"]):
            raise ValueError("known stage and finite UTC seconds required")
        if e["at"] <= observed_until:
            identity = (e.get("context_id"), e.get("parent"))
            if e["join_key"] in identities and identities[e["join_key"]] != identity:
                raise ValueError("feedback context/parent join mismatch")
            identities[e["join_key"]] = identity
            g = grouped[e["join_key"]]
            g[e["stage"]] = min(g.get(e["stage"], e["at"]), e["at"])
    result = []
    for key, g in sorted(grouped.items()):
        ordered = [(s, g[s]) for s in STAGES if s in g]
        if any(a[1] > b[1] for a, b in zip(ordered, ordered[1:])):
            raise ValueError("stage clock/order inconsistency")
        proposal = g.get("PROPOSAL_COMMITTED")
        row = {
            "join_key": key,
            "context_id": identities[key][0],
            "parent": identities[key][1],
            "proposal_at": proposal,
            "censored": "FIRST_EXPOSURE" not in g,
            "T_feedback": None,
            "stage_seconds": {},
        }
        if proposal is not None and "FIRST_EXPOSURE" in g:
            row["T_feedback"] = g["FIRST_EXPOSURE"] - proposal
        for a, b in zip(STAGES, STAGES[1:]):
            row["stage_seconds"][a + "→" + b] = (
                g[b] - g[a] if a in g and b in g else None
            )
        row["age_seconds"] = observed_until - proposal if proposal is not None else None
        result.append(row)
    return result
