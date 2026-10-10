"""Accept only verified comparable q. Development and sealed rows are not targets."""

from __future__ import annotations

import math
from collections import defaultdict

Q_SCALE = 1_000_000
PARITY_LIMIT = 1e-9
Q_MATCH = 1e-6
TARGET_ROLES = frozenset({"train", "train_fit"})
# R6 D-tune and D-confirm are development partitions. Neither enters optimizer targets.
FORBIDDEN_TARGET_ROLES = frozenset({
    "development", "dev", "validation", "val", "sealed", "holdout", "judge",
    "rights_pending", "test", "meta_test",
})
FORBIDDEN_TARGET_SUBROLES = frozenset({"D-tune", "D-confirm", "development", "sealed"})
REFUSED_OUTCOMES = frozenset({
    "fitness_only", "g0_not_ready", "parity_custody", "parity_unqualified",
    "infra_unknown", "unknown", "censored", "program_failed", "timeout",
    "predicted",
})


def _fail(reason):
    raise ValueError(reason)


def checked_measurement(row):
    """Return a normalized measurement or raise ValueError.

    ``q`` is the arena micro-q divided by 1e6. Predicted q, fitness, parity
    failures and unknowns are refused.
    """
    if not isinstance(row, dict):
        _fail("measurement must be a dict")
    kind = row.get("reward_kind", "verified_comparable_q")
    if kind != "verified_comparable_q":
        _fail(f"refused reward kind {kind}")
    for key in ("q_hat", "predicted_q", "fitness"):
        if row.get(key) not in (None, False):
            _fail(f"refused non-reward field {key}")
    outcome = row.get("outcome")
    if outcome in REFUSED_OUTCOMES or outcome != "qualified_complete":
        _fail(f"refused outcome {outcome}")
    if row.get("qualified") is not True:
        _fail("unqualified row")
    if row.get("parity_verified") is not True:
        _fail("parity not verified")
    if row.get("closure_verified") is not True:
        _fail("closure not verified")
    error = row.get("parity_max_normalized_error")
    if not isinstance(error, (int, float)) or isinstance(error, bool):
        _fail("missing parity error")
    if not math.isfinite(float(error)) or float(error) < 0 or float(error) > PARITY_LIMIT:
        _fail("parity error above 1e-9")
    q_grid = row.get("q_grid")
    if type(q_grid) is not int:
        _fail("q_grid must be the integer micro-q")
    q = row.get("q")
    if not isinstance(q, (int, float)) or isinstance(q, bool) or not math.isfinite(float(q)):
        _fail("missing finite measured q")
    scaled = q_grid / Q_SCALE
    if abs(float(q) - scaled) > Q_MATCH:
        _fail("q does not match q_grid")
    role = row.get("role")
    subrole = row.get("subrole")
    if not isinstance(role, str) or not role:
        _fail("missing role")
    program = row.get("program_sha256")
    context = row.get("context_id")
    lineage = row.get("lineage_id")
    if not all(isinstance(v, str) and len(v) == 64 for v in (program, context, lineage)):
        _fail("context, lineage and program must be sha256")
    tokens = row.get("tokens")
    if tokens is not None:
        tokens = tuple(tokens)
        if not tokens or any(type(t) is not int or t < 0 for t in tokens):
            _fail("bad tokens")
    normalized = {
        "reward_kind": "verified_comparable_q",
        "context_id": context,
        "lineage_id": lineage,
        "program_sha256": program,
        "role": role,
        "subrole": subrole,
        "q": scaled,
        "q_grid": q_grid,
        "q_hat": None,
        "predicted_q": None,
        "fitness": None,
        "outcome": outcome,
        "qualified": True,
        "parity_verified": True,
        "closure_verified": True,
        "parity_max_normalized_error": float(error),
        "arm": row.get("arm"),
        "seed": row.get("seed"),
        "slot": row.get("slot"),
        "tokens": tokens,
        "evaluator_sha256": row.get("evaluator_sha256"),
        "protocol": row.get("protocol"),
    }
    return normalized


def target_allowed(measurement):
    role = measurement["role"]
    subrole = measurement["subrole"]
    if role not in TARGET_ROLES or role in FORBIDDEN_TARGET_ROLES:
        return False
    if subrole in FORBIDDEN_TARGET_SUBROLES:
        return False
    return True


def build_pairs(measurements, tau_c):
    """Same-context pairs with measured q_w - q_l > tau_c. Target roles only."""
    if (
        isinstance(tau_c, bool)
        or not isinstance(tau_c, (int, float))
        or not math.isfinite(tau_c)
        or tau_c < 0
    ):
        _fail("tau_c must be finite and >= 0")
    by_context = defaultdict(list)
    refused = defaultdict(int)
    for row in measurements:
        try:
            item = checked_measurement(row)
        except ValueError:
            refused["invalid"] += 1
            continue
        if not target_allowed(item):
            refused["role"] += 1
            continue
        by_context[item["context_id"]].append(item)
    pairs = []
    for context, rows in sorted(by_context.items()):
        dedup = {}
        conflict = False
        for row in rows:
            previous = dedup.get(row["program_sha256"])
            if previous is not None and previous["q_grid"] != row["q_grid"]:
                conflict = True
                break
            if previous is None:
                dedup[row["program_sha256"]] = row
        if conflict:
            refused["conflicting_q"] += 1
            continue
        programs = list(dedup.values())
        if len({p["lineage_id"] for p in programs}) != 1:
            refused["lineage"] += 1
            continue
        identities = {(p["evaluator_sha256"], p["protocol"]) for p in programs}
        if len(identities) != 1:
            refused["protocol"] += 1
            continue
        evaluator, protocol = next(iter(identities))
        if not isinstance(evaluator, str) or len(evaluator) != 64:
            refused["protocol"] += 1
            continue
        if not isinstance(protocol, str) or not protocol:
            refused["protocol"] += 1
            continue
        for winner in programs:
            for loser in programs:
                if winner["program_sha256"] == loser["program_sha256"]:
                    continue
                if winner["q"] - loser["q"] <= tau_c:
                    continue
                pairs.append({
                    "context_id": context,
                    "lineage_id": winner["lineage_id"],
                    "role": winner["role"],
                    "subrole": winner["subrole"],
                    "winner": winner["program_sha256"],
                    "loser": loser["program_sha256"],
                    "q_winner": winner["q"],
                    "q_loser": loser["q"],
                    "margin": winner["q"] - loser["q"],
                    "tokens_winner": winner["tokens"],
                    "tokens_loser": loser["tokens"],
                })
    return pairs, dict(refused)
