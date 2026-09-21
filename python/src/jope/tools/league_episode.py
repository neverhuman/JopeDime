"""parallel.league_episode.v1 — Expert Iteration league receipt. No serving writes."""

from __future__ import annotations

import math

SCHEMA = "parallel.league_episode.v1"
REQUIRED_ARMS = ("neural", "generic", "random", "teacher")
OUTCOMES = ("win", "loss", "tie")


def finite(value):
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def outcome(left, right):
    if not finite(left) or not finite(right):
        return None
    if left < right:
        return "win"
    if right < left:
        return "loss"
    return "tie"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def arm(rmse, program_sha256=None, evaluations=None, policy_logprob=None, source=None):
    row = dict(rmse=rmse, program_sha256=program_sha256, evaluations=evaluations, source=source)
    if policy_logprob is not None:
        row["policy_logprob"] = policy_logprob
    return row


def validate_arm(name, row):
    require(isinstance(row, dict), f"{name} arm missing")
    require(finite(row.get("rmse")) and row["rmse"] >= 0, f"{name} rmse invalid")
    if "policy_logprob" in row:
        require(finite(row["policy_logprob"]), f"{name} policy_logprob invalid")
    extra = set(row) - {"rmse", "program_sha256", "evaluations", "source", "policy_logprob"}
    require(not extra, f"{name} unknown fields: {sorted(extra)}")
    return row


def episode(board, arms, independent_evaluation, policy_logprob=None):
    """Build one league episode. Serving is never written."""
    require(isinstance(board, dict), "board missing")
    require(isinstance(arms, dict), "arms missing")
    require(set(arms) == set(REQUIRED_ARMS), f"required arms {REQUIRED_ARMS}; got {tuple(arms)}")
    require("shuffled" not in arms, "shuffled arm is retired")
    require(
        independent_evaluation is False,
        "league diagnostics cannot certify independent evaluation; original verifier required",
    )
    checked = {name: validate_arm(name, arms[name]) for name in REQUIRED_ARMS}
    if policy_logprob is not None:
        require(
            "policy_logprob" not in checked["neural"]
            or checked["neural"]["policy_logprob"] == policy_logprob,
            "neural policy_logprob conflict",
        )
        checked["neural"] = dict(checked["neural"], policy_logprob=policy_logprob)
        require(finite(checked["neural"]["policy_logprob"]), "policy_logprob invalid")
    neural = checked["neural"]["rmse"]
    return dict(
        schema=SCHEMA,
        board=dict(board),
        arms=checked,
        neural_vs_teacher=outcome(neural, checked["teacher"]["rmse"]),
        neural_vs_random=outcome(neural, checked["random"]["rmse"]),
        neural_vs_generic=outcome(neural, checked["generic"]["rmse"]),
        independent_evaluation=False,
        evidence_scope="unverified diagnostic scores; no original execution/access qualification",
        qualification_ready=False,
        serving_written=False,
        shuffled_minted=False,
    )


def validate(row):
    require(isinstance(row, dict) and row.get("schema") == SCHEMA, "unknown league schema")
    require(row.get("serving_written") is False, "league episode must not write serving")
    require(row.get("shuffled_minted") is False, "shuffled stays off")
    require(
        row.get("independent_evaluation") is False, "league diagnostics cannot certify independent evaluation"
    )
    require(row.get("qualification_ready") is False, "league diagnostics are unqualified")
    require(set(row.get("arms") or ()) == set(REQUIRED_ARMS), "incomplete league arms")
    require("shuffled" not in (row.get("arms") or ()), "shuffled arm is retired")
    for name in REQUIRED_ARMS:
        validate_arm(name, row["arms"][name])
    neural = row["arms"]["neural"]["rmse"]
    for key, opponent in (
        ("neural_vs_teacher", "teacher"),
        ("neural_vs_random", "random"),
        ("neural_vs_generic", "generic"),
    ):
        require(row.get(key) in OUTCOMES, f"{key} missing")
        require(row[key] == outcome(neural, row["arms"][opponent]["rmse"]), f"{key} disagrees with rmse")
    if "policy_logprob" in row["arms"]["neural"]:
        require(finite(row["arms"]["neural"]["policy_logprob"]), "policy_logprob invalid")
    return row
