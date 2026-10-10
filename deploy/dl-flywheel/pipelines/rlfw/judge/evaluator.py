"""Judge evaluator wrapper.

The callable passed in is the pinned evaluator. This module checks its
attestation and passes q_grid through. It does not compute q.
"""

import math

from rlfw.arena.game_spec import require_sha

from .seal import QuarantineError, SealSpent, mark_scored

OUTCOMES = (
    "qualified_complete",
    "program_failed",
    "timeout",
    "parity_custody",
    "infra_unknown",
)


class JudgeEvaluator:
    def __init__(self, pinned_evaluate, *, evaluator_sha256, canonicalizer_sha256, protocol):
        if not callable(pinned_evaluate):
            raise TypeError("pinned evaluator must be callable")
        require_sha(evaluator_sha256, "evaluator")
        require_sha(canonicalizer_sha256, "canonicalizer")
        if not isinstance(protocol, str) or not protocol:
            raise ValueError("protocol is required")
        self.pinned_evaluate = pinned_evaluate
        self.evaluator_sha256 = evaluator_sha256
        self.canonicalizer_sha256 = canonicalizer_sha256
        self.protocol = protocol

    def score(self, store, *, family_id, context_id, program_sha256):
        document = store.require_spent_fixture()
        family = next((item for item in document["families"] if item["family_id"] == family_id), None)
        if family is None:
            raise KeyError(family_id)
        if family["judge_context_id"] != context_id:
            raise ValueError("score only the family's judge context")
        if family["overlap_status"] != "no_overlap":
            raise QuarantineError(family["overlap_status"])
        if document["picks"].get(family_id) != program_sha256:
            raise ValueError("only the precommitted pick can be scored")
        if any(row["context_id"] == context_id and row["program_sha256"] == program_sha256 for row in document["scored"]):
            raise SealSpent("this judge context was already scored")
        request = {
            "canonicalizer_sha256": self.canonicalizer_sha256,
            "context_id": context_id,
            "evaluator_sha256": self.evaluator_sha256,
            "judge_row_ids": store.judge_rows(family_id),
            "lane": "B",
            "program_sha256": program_sha256,
            "protocol": self.protocol,
        }
        attested = attest(self.pinned_evaluate(request), request)
        mark_scored(
            store.path,
            arm=document["arm"],
            context_id=context_id,
            fixture_root=store.fixture_root,
            program_sha256=program_sha256,
        )
        return attested


def attest(result, request):
    if not isinstance(result, dict):
        raise ValueError("pinned evaluator must return an object")
    if any(name in result for name in ("features", "judge_labels", "targets")):
        raise ValueError("pinned evaluator must not return judge features or labels")
    if result.get("rust_attested") is not True:
        raise ValueError("judge scoring requires an explicit Rust parity attestation")
    outcome = result.get("outcome_class")
    if outcome not in OUTCOMES:
        raise ValueError("evaluator returned an unknown outcome taxonomy")
    for name in (
        "canonicalizer_sha256",
        "context_id",
        "evaluator_sha256",
        "program_sha256",
        "protocol",
    ):
        if result.get(name) != request[name]:
            raise ValueError(f"evaluator identity mismatch: {name}")
    if result.get("lane") not in (None, "B"):
        raise ValueError("lane mismatch")
    out = dict(result)
    out["outcome_class"] = outcome
    if outcome == "qualified_complete":
        if result.get("parity_verified") is not True or result.get("closure_verified") is not True:
            raise ValueError("completion requires closure and native/Rust parity")
        error = result.get("parity_max_normalized_error")
        if (
            not isinstance(error, (int, float))
            or not math.isfinite(error)
            or not 0 <= error <= 1e-9
        ):
            raise ValueError("fixed normalized 1e-9 parity tolerance required")
        if type(result.get("q_grid")) is not int:
            raise ValueError("only measured integer micro-q enters scores")
        require_sha(result.get("receipt_sha256"), "execution receipt")
        if "q" in result and result["q"] != result["q_grid"] / 1_000_000:
            raise ValueError("q must be the pinned micro-q divided by 1_000_000")
    elif result.get("q_grid") is not None or result.get("q") is not None:
        raise ValueError("failures and unknowns cannot disclose numeric scores")
    for name in ("wall_seconds", "cpu_seconds"):
        value = result.get(name)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError("measured nonnegative evaluator costs required")
    return out
