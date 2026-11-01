#!/usr/bin/env python3
"""Q1 adapter for the production engine's durable evaluation receipt API."""

import hashlib
import json
import math
import os
import sys
from urllib.error import URLError
from urllib.request import Request, urlopen


def canonical(value):
    return json.dumps(
        value, sort_keys=True, allow_nan=False, separators=(",", ":")
    ).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON key")
        result[key] = value
    return result


def decode(raw):
    return json.loads(
        raw,
        object_pairs_hook=unique_object,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError("non-finite JSON")),
    )


def sha(value):
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def validate_receipt(receipt, request):
    require(request.get("schema") == "jope.q1-request/v1", "unsupported Q1 request")
    require(
        request.get("mode") == "qualification",
        "production adapter requires qualification mode",
    )
    require(
        type(request.get("seed")) is int
        and request["seed"] >= 0
        and type(request.get("budget")) is int
        and request["budget"] > 0
        and request.get("metric") == "rmse",
        "invalid request seed, budget or metric",
    )
    expected_request_hash = request.get("request_sha256")
    unhashed_request = {
        key: value for key, value in request.items() if key != "request_sha256"
    }
    require(
        sha(expected_request_hash)
        and digest(unhashed_request) == expected_request_hash,
        "request digest mismatch",
    )
    require(
        isinstance(receipt, dict)
        and receipt.get("schema") == "jope.q1-evaluation-receipt/v1",
        "unsupported engine receipt",
    )
    require(
        receipt.get("request_sha256") == expected_request_hash,
        "receipt request mismatch",
    )
    dataset = request.get("dataset")
    require(isinstance(dataset, dict), "dataset identity missing")
    identity = receipt.get("identity")
    require(isinstance(identity, dict), "receipt identity missing")
    require(
        identity
        == {
            "source_commit": request["source_commit"],
            "plan_sha256": request["plan_sha256"],
            "pins": request["pins"],
            "dataset_id": dataset["id"],
            "dataset_sha256": dataset["dataset_sha256"],
            "cut_sha256": dataset["cut_sha256"],
            "guidance": request["guidance"],
            "metric": request["metric"],
            "seed": request["seed"],
            "requested_budget": request["budget"],
            "dataset_binding_sha256": identity.get("dataset_binding_sha256"),
        }
        and sha(identity.get("dataset_binding_sha256")),
        "receipt identity mismatch",
    )
    require(request["guidance"] in ("heuristic", "learned"), "unsupported policy")
    attempts = receipt.get("attempts")
    charged = receipt.get("charged")
    require(
        type(charged) is int and charged == request["budget"],
        "actual charged budget mismatch",
    )
    require(
        isinstance(attempts, list) and len(attempts) == charged,
        "attempt inventory mismatch",
    )
    completed = []
    failures = []
    for index, attempt in enumerate(attempts):
        require(
            isinstance(attempt, dict)
            and type(attempt.get("attempt")) is int
            and attempt["attempt"] == index,
            "attempt IDs must be unique and contiguous",
        )
        outcome = attempt.get("outcome")
        if outcome == "completed":
            scores = attempt.get("fold_scores")
            loss = attempt.get("loss")
            program_sha = attempt.get("program_sha256")
            require(
                isinstance(scores, list)
                and len(scores) >= 2
                and all(
                    type(score) in (int, float) and math.isfinite(score) and score >= 0
                    for score in scores
                )
                and type(loss) in (int, float)
                and math.isfinite(loss)
                and loss >= 0
                and sha(program_sha),
                "completed attempt lacks real score/program receipt",
            )
            completed.append(attempt)
        elif outcome == "failed":
            failure = attempt.get("failure")
            require(
                attempt.get("fold_scores") == []
                and attempt.get("loss") is None
                and attempt.get("program_sha256") is None
                and isinstance(failure, dict)
                and isinstance(failure.get("code"), str)
                and failure["code"].strip()
                and isinstance(failure.get("reason"), str)
                and failure["reason"].strip(),
                "failed attempt lacks typed failure receipt",
            )
            failures.append(
                {"attempt": index, "reason": failure["code"] + ": " + failure["reason"]}
            )
        else:
            raise ValueError("unknown or nonterminal attempt outcome")
    champion = min(completed, key=lambda item: item["loss"]) if completed else None
    adapter = receipt.get("adapter")
    require(
        isinstance(adapter, str) and adapter.strip(), "engine adapter identity missing"
    )
    return {
        "schema": "jope.q1-response/v1",
        "request_sha256": expected_request_hash,
        "status": "complete" if completed else "failed",
        "guidance": request["guidance"],
        "metric": "rmse",
        "charged": charged,
        "completed": len(completed),
        "loss": champion["loss"] if champion else None,
        "fold_scores_completed": sum(len(item["fold_scores"]) for item in completed),
        "candidate_failures": failures,
        "champion_program": champion["program_sha256"] if champion else None,
        "adapter": adapter,
    }


def call_engine(request, endpoint, timeout=600):
    require(
        isinstance(endpoint, str) and endpoint.startswith("http://127.0.0.1:"),
        "engine must use loopback HTTP",
    )
    url = endpoint.rstrip("/") + "/v1/q1/evaluations"
    http_request = Request(
        url,
        data=canonical(request),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(http_request, timeout=timeout) as response:
            require(response.status == 200, "engine returned non-success status")
            raw = response.read(8 * 1024 * 1024 + 1)
    except (OSError, URLError) as error:
        raise ValueError("production engine request failed") from error
    require(len(raw) <= 8 * 1024 * 1024, "engine receipt exceeds size limit")
    return decode(raw)


def main():
    try:
        request = decode(sys.stdin.buffer.read(1024 * 1024 + 1))
        require(len(canonical(request)) <= 1024 * 1024, "Q1 request exceeds size limit")
        receipt = call_engine(request, os.environ.get("JOPE_Q1_ENGINE_URL", ""))
        response = validate_receipt(receipt, request)
        sys.stdout.buffer.write(canonical(response) + b"\n")
    except (ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        print("production backend refused: " + str(error), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
